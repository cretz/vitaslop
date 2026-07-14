//! The browser engine host: the web mirror of `vitaslop-native::Vm`. It runs the
//! transpiler's emitted guest module on the browser's own `WebAssembly` engine
//! (there is no wasmtime in the browser) and services the guest's `env.svc` /
//! `env.import` host traps by calling back into the engine-agnostic
//! `vitaslop-runtime` (which is compiled into *this* wasm-bindgen module).
//!
//! # The two-instance memory seam
//! The runtime and the guest are two separate `WebAssembly` instances with two
//! separate linear memories: the runtime cannot borrow the guest's memory as a
//! Rust `&mut [u8]` the way native wasmtime can. So guest memory is reached
//! through [`vitaslop_runtime::GuestMemory`], backed here by a `Uint8Array` view
//! over the guest instance's `ArrayBuffer` ([`JsGuestMemory`]). This costs a JS
//! round-trip per access, but host calls happen only at kernel/GXM boundaries
//! (tens per frame), never in the hot CPU loop - the loop is pure guest wasm at
//! full browser speed. See the runtime's `GuestMemory` docs.
//!
//! # Import wiring
//! The guest imports `env.svc : (i32 imm) -> ()` and `env.import : (i32 index) ->
//! ()`. We supply both as JS closures. They need the guest instance's globals and
//! memory, which only exist *after* instantiation - a chicken-and-egg. The
//! closures capture a shared, initially-empty [`ExportsCell`] that we fill once the
//! instance is built; imports fire only during execution, so the cell is always
//! populated by then. A host handler that asks to halt throws a sentinel
//! `JsValue`, which unwinds the guest call exactly as native's trap-based halt.
//!
//! Which trap a run wires depends on its host convention: the Vita cube routes
//! `env.import` to a `VitaEnv`; the ARM conformance corpus routes `env.svc` to a
//! Linux-EABI `SvcDispatch`. Either may be `None`.

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{Function, Object, Reflect, Uint8Array, WebAssembly};
use vitaslop_runtime::{Flags, GuestMemory, ImportDispatch, SvcDispatch};
use vitaslop_transpiler::abi;
use wasm_bindgen::prelude::*;

/// A `Uint8Array` view over the guest instance's linear memory, rebased so guest
/// address `A` is byte `A - base`. Created fresh per host call because
/// `memory.grow` detaches and replaces the underlying `ArrayBuffer`.
struct JsGuestMemory {
    view: Uint8Array,
}

impl GuestMemory for JsGuestMemory {
    fn len(&self) -> usize {
        self.view.length() as usize
    }
    fn read(&self, off: usize, buf: &mut [u8]) {
        self.view
            .subarray(off as u32, (off + buf.len()) as u32)
            .copy_to(buf);
    }
    fn write(&mut self, off: usize, bytes: &[u8]) {
        self.view
            .subarray(off as u32, (off + bytes.len()) as u32)
            .copy_from(bytes);
    }
}

/// The guest instance handles the host closures and accessors need: its memory,
/// the 16 register globals, and the 4 NZCV flag globals. Filled after instantiation.
struct GuestExports {
    memory: WebAssembly::Memory,
    regs: [WebAssembly::Global; abi::REG_COUNT],
    /// VFP single-precision argument registers s0..s15 (raw bits), the float arg
    /// and return file for the hardfloat NID path.
    vfp: [WebAssembly::Global; vitaslop_runtime::VFP_ARG_COUNT],
    flags: [WebAssembly::Global; abi::FLAG_COUNT],
}

impl GuestExports {
    fn read_regs(&self) -> [u32; abi::REG_COUNT] {
        let mut regs = [0u32; abi::REG_COUNT];
        for (i, g) in self.regs.iter().enumerate() {
            // An i32 global reads back as a signed JS number; recover the bits.
            regs[i] = g.value().as_f64().unwrap_or(0.0) as i64 as u32;
        }
        regs
    }
    fn write_regs(&self, regs: &[u32; abi::REG_COUNT]) {
        for (i, g) in self.regs.iter().enumerate() {
            // ToInt32 in the setter wraps a u32-valued f64 to the right bits.
            g.set_value(&JsValue::from_f64(regs[i] as f64));
        }
    }
    fn read_vfp(&self) -> [u32; vitaslop_runtime::VFP_ARG_COUNT] {
        let mut vfp = [0u32; vitaslop_runtime::VFP_ARG_COUNT];
        for (i, g) in self.vfp.iter().enumerate() {
            vfp[i] = g.value().as_f64().unwrap_or(0.0) as i64 as u32;
        }
        vfp
    }
    fn write_vfp(&self, vfp: &[u32; vitaslop_runtime::VFP_ARG_COUNT]) {
        for (i, g) in self.vfp.iter().enumerate() {
            g.set_value(&JsValue::from_f64(vfp[i] as f64));
        }
    }
    fn read_flags(&self) -> Flags {
        let get = |g: &WebAssembly::Global| g.value().as_f64().unwrap_or(0.0) != 0.0;
        // Flag globals are exported in N, Z, C, V order (see abi::flag_export).
        Flags {
            n: get(&self.flags[0]),
            z: get(&self.flags[1]),
            c: get(&self.flags[2]),
            v: get(&self.flags[3]),
        }
    }
    /// A rebased byte view over the current memory buffer.
    fn memory_view(&self) -> JsGuestMemory {
        JsGuestMemory { view: Uint8Array::new(&self.memory.buffer()) }
    }
}

type ExportsCell = Rc<RefCell<Option<GuestExports>>>;

/// The sentinel a halting host handler throws to unwind the guest call. Caught in
/// [`WebVm::call`], which reports a clean halt rather than an error.
fn halt_sentinel() -> JsValue {
    JsValue::from_str("vitaslop:halt")
}

/// A transpiled guest module instantiated on the browser's `WebAssembly` engine,
/// ready for host-driven execution. Mirrors `vitaslop-native::Vm`.
pub struct WebVm {
    instance: WebAssembly::Instance,
    exports: ExportsCell,
    halted: Rc<RefCell<bool>>,
    // The import closures must outlive every guest call that can invoke them, so
    // this Vm holds them alive for as long as it lives.
    _svc: Closure<dyn FnMut(i32) -> Result<(), JsValue>>,
    _import: Closure<dyn FnMut(i32) -> Result<(), JsValue>>,
    _dispatch_miss: Closure<dyn FnMut(i32, i32) -> Result<(), JsValue>>,
}

impl WebVm {
    /// Instantiate `wasm` (the transpiler's emitted module for a guest loaded at
    /// `base`), seed the guest image `code` and stack pointer, and wire the host
    /// traps: `env.svc` to `svc` (the ARM/Linux path) and `env.import` to `import`
    /// (the Vita NID path). Either may be `None` (a no-op trap). `mem_bytes` sizes
    /// the provisioned region (its top is the initial sp), matching the value the
    /// module was emitted with.
    pub fn new(
        wasm: &[u8],
        code: &[u8],
        base: u32,
        mem_bytes: u32,
        mut svc: Option<Box<dyn SvcDispatch>>,
        mut import: Option<Box<dyn ImportDispatch>>,
    ) -> Result<WebVm, JsValue> {
        let exports: ExportsCell = Rc::new(RefCell::new(None));
        let halted = Rc::new(RefCell::new(false));

        // env.svc: read regs, hand the handler a memory view, service the svc,
        // write regs back, throw to unwind on halt.
        let svc_closure = {
            let exports = exports.clone();
            let halted = halted.clone();
            Closure::wrap(Box::new(move |imm: i32| -> Result<(), JsValue> {
                let cell = exports.borrow();
                let ex = cell.as_ref().expect("exports set before first call");
                let mut regs = ex.read_regs();
                let outcome = match svc.as_mut() {
                    Some(h) => {
                        let mut mem = ex.memory_view();
                        h.svc(imm as u32, &mut regs, &mut mem, base)
                    }
                    None => vitaslop_runtime::SvcOutcome::Continue,
                };
                ex.write_regs(&regs);
                finish(&halted, outcome)
            }) as Box<dyn FnMut(i32) -> Result<(), JsValue>>)
        };

        // env.import: same shape for the Vita NID trap.
        let import_closure = {
            let exports = exports.clone();
            let halted = halted.clone();
            Closure::wrap(Box::new(move |index: i32| -> Result<(), JsValue> {
                let cell = exports.borrow();
                let ex = cell.as_ref().expect("exports set before first call");
                let mut regs = ex.read_regs();
                let mut vfp = ex.read_vfp();
                let outcome = match import.as_mut() {
                    Some(h) => {
                        let mut mem = ex.memory_view();
                        h.dispatch(index as u32, &mut regs, &mut vfp, &mut mem, base)
                    }
                    None => vitaslop_runtime::SvcOutcome::Continue,
                };
                ex.write_regs(&regs);
                ex.write_vfp(&vfp);
                finish(&halted, outcome)
            }) as Box<dyn FnMut(i32) -> Result<(), JsValue>>)
        };

        // env.dispatch_miss: an indirect call that resolves to no translated function
        // throws with the faulting (target, caller) addresses, turning an opaque
        // `unreachable` trap into a debuggable message.
        let dispatch_miss_closure = Closure::wrap(Box::new(move |target: i32, caller: i32| -> Result<(), JsValue> {
            Err(JsValue::from_str(&format!(
                "indirect dispatch to unknown target {:#010x} from f_{:x}",
                target as u32, caller as u32
            )))
        }) as Box<dyn FnMut(i32, i32) -> Result<(), JsValue>>);

        let env_obj = Object::new();
        Reflect::set(&env_obj, &JsValue::from_str(abi::SVC_NAME), svc_closure.as_ref())?;
        Reflect::set(&env_obj, &JsValue::from_str(abi::IMPORT_NAME), import_closure.as_ref())?;
        Reflect::set(&env_obj, &JsValue::from_str(abi::DISPATCH_MISS_NAME), dispatch_miss_closure.as_ref())?;
        let imports = Object::new();
        Reflect::set(&imports, &JsValue::from_str(abi::IMPORT_MODULE), &env_obj)?;

        let module = WebAssembly::Module::new(&Uint8Array::from(wasm).into())?;
        let instance = WebAssembly::Instance::new(&module, &imports)?;
        let exports_obj = instance.exports();

        // Pull the guest handles the closures and accessors need; fill the cell.
        let memory = Reflect::get(&exports_obj, &JsValue::from_str(abi::MEMORY_EXPORT))?
            .dyn_into::<WebAssembly::Memory>()?;
        let regs = read_globals::<{ abi::REG_COUNT }>(&exports_obj, |i| abi::reg_export(i))?;
        let vfp = read_globals::<{ vitaslop_runtime::VFP_ARG_COUNT }>(&exports_obj, |i| {
            abi::vfp_s_export(i as u8)
        })?;
        let flags = read_globals::<{ abi::FLAG_COUNT }>(&exports_obj, |i| {
            abi::flag_export(FLAG_ORDER[i]).to_string()
        })?;
        *exports.borrow_mut() = Some(GuestExports { memory: memory.clone(), regs, vfp, flags });

        // Seed the guest image at rebased offset 0 and set sp to the top of the
        // provisioned region (matches native Vm::new).
        let view = Uint8Array::new(&memory.buffer());
        view.subarray(0, code.len() as u32).copy_from(code);
        {
            let cell = exports.borrow();
            let ex = cell.as_ref().unwrap();
            ex.regs[abi::SP].set_value(&JsValue::from_f64(base.wrapping_add(mem_bytes) as f64));
        }

        Ok(WebVm {
            instance,
            exports,
            halted,
            _svc: svc_closure,
            _import: import_closure,
            _dispatch_miss: dispatch_miss_closure,
        })
    }

    /// Call the guest function exported at `addr`, running until it returns or a
    /// host handler halts it. A clean halt is `Ok`, mirroring native `Vm::call`.
    pub fn call(&self, addr: u32) -> Result<(), JsValue> {
        let exports_obj = self.instance.exports();
        let func = Reflect::get(&exports_obj, &JsValue::from_str(&abi::func_export(addr)))?
            .dyn_into::<Function>()?;
        match func.call0(&JsValue::NULL) {
            Ok(_) => Ok(()),
            Err(e) => {
                if *self.halted.borrow() {
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Seed guest register `i` before a call (e.g. a case's input registers).
    pub fn set_reg(&self, i: usize, v: u32) {
        let cell = self.exports.borrow();
        let ex = cell.as_ref().expect("instantiated");
        ex.regs[i].set_value(&JsValue::from_f64(v as f64));
    }

    /// The final ARM register file (r0..r15).
    pub fn regs(&self) -> [u32; abi::REG_COUNT] {
        self.exports.borrow().as_ref().expect("instantiated").read_regs()
    }

    /// The final NZCV condition flags.
    pub fn flags(&self) -> Flags {
        self.exports.borrow().as_ref().expect("instantiated").read_flags()
    }
}

/// Flag globals in the ABI's export order (N, Z, C, V), so index i of the flag
/// array is the flag whose export name is `flag_export(FLAG_ORDER[i])`.
const FLAG_ORDER: [abi::Flag; abi::FLAG_COUNT] =
    [abi::Flag::N, abi::Flag::Z, abi::Flag::C, abi::Flag::V];

/// Write back regs already done by the caller; turn a host outcome into the
/// closure's return, throwing the halt sentinel to unwind the guest call.
fn finish(halted: &Rc<RefCell<bool>>, outcome: vitaslop_runtime::SvcOutcome) -> Result<(), JsValue> {
    use vitaslop_runtime::SvcOutcome;
    match outcome {
        // A worker ending is the process ending on this single-worker path.
        SvcOutcome::Halt | SvcOutcome::ThreadExit => {
            *halted.borrow_mut() = true;
            Err(halt_sentinel())
        }
        // This run-to-completion path has no scheduler to yield to, so a blocking
        // hint (Yield at a flip, or a would-block wait) just keeps running,
        // identical to the native sync Vm. The browser's cooperative-scheduler path
        // will handle these distinctly.
        SvcOutcome::Continue | SvcOutcome::Yield | SvcOutcome::Block => Ok(()),
    }
}

/// Fetch `N` exported globals, named `name(0)..name(N-1)`, into a fixed array.
fn read_globals<const N: usize>(
    exports: &JsValue,
    name: impl Fn(usize) -> String,
) -> Result<[WebAssembly::Global; N], JsValue> {
    let mut v = Vec::with_capacity(N);
    for i in 0..N {
        let g = Reflect::get(exports, &JsValue::from_str(&name(i)))?
            .dyn_into::<WebAssembly::Global>()?;
        v.push(g);
    }
    Ok(v.try_into().expect("N globals"))
}
