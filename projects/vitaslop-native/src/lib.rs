//! Native host mechanisms shared by every non-browser host (desktop and, one
//! day, mobile): the wasmtime-backed engine that runs the transpiler's WASM,
//! and later the mmap image source and worker threads. Never compiled to
//! wasm32, so no engine cfg gymnastics.
//!
//! The core type is [`Vm`]: a transpiled module instantiated for host-driven
//! execution - seed guest memory and registers, call exported guest functions
//! by address, read state back. The browser host will mirror this shape over
//! `WebAssembly`. [`run`]/[`run_arm`] are thin conformance-oriented wrappers.

pub use vitaslop_runtime::{
    capture, nid, render, CtrlFrame, DeterministicWorld, Flags, GuestMemory, ImportDispatch, Record,
    Replay, RunResult, SvcOutcome, VitaEnv, VitaState, World, WorldEvent,
};

pub mod sched;
pub use sched::{FrameStop, Scheduler};

pub mod threaded;
pub use threaded::{RunReport, ThreadSpawn, ThreadedScheduler};

pub mod wgpu_render;
pub use wgpu_render::WgpuRenderer;
pub use vitaslop_transpiler::abi;
use vitaslop_transpiler::{self as transpiler};
use wasmtime::{Caller, Config, Engine, Instance, Linker, Module, Store, Val};

/// A host `svc`/`import` handler: given the call selector (the `svc` imm or the
/// NID import index), the guest registers (mutable - write return values into
/// them, e.g. r0), guest memory (rebased: offset 0 is guest address `base`), the
/// image `base`, and the output sink, service the call and say whether to
/// continue or halt. Any register the handler changes is written back to the
/// guest register file after it returns.
pub type SvcHandler = fn(
    selector: u32,
    regs: &mut [u32; abi::REG_COUNT],
    mem: &mut [u8],
    base: u32,
    out: &mut Vec<u8>,
) -> SvcOutcome;

/// The host ABI a run uses. Injected by the caller so the engine carries no
/// syscall convention of its own: the arm conformance harness passes a Linux
/// one, Vita will pass a NID-based one.
pub struct HostAbi<'a> {
    /// Syscall numbers (guest r7) that do not return, so the transpiler can end
    /// decoding at a `svc` with a statically-known one of them.
    pub noreturn_svc: &'a [u32],
    /// Services an ARM `svc`.
    pub svc: SvcHandler,
    /// Services a Vita NID `import` call, by dense index.
    pub import: SvcHandler,
}

fn noop_handler(
    _selector: u32,
    _regs: &mut [u32; abi::REG_COUNT],
    _mem: &mut [u8],
    _base: u32,
    _out: &mut Vec<u8>,
) -> SvcOutcome {
    SvcOutcome::Continue
}

impl Default for HostAbi<'_> {
    fn default() -> Self {
        HostAbi { noreturn_svc: &[], svc: noop_handler, import: noop_handler }
    }
}

/// Default guest memory provisioned for a run (image + stack + heap), in bytes
/// from `base`.
pub const DEFAULT_MEM_BYTES: u32 = 64 * 1024 * 1024;

/// Host state threaded through a run: captured output, the exit flag, and an
/// optional stateful Vita import environment. When `import_env` is set, the
/// `env.import` trap routes to it (the Vita NID path); otherwise it falls back to
/// the fn-pointer handler in `HostAbi` (the ARM/Linux conformance path).
struct Host {
    output: Vec<u8>,
    halted: bool,
    base: u32,
    import_fn: SvcHandler,
    import_env: Option<Box<dyn ImportDispatch>>,
    /// The instantiated module, so a host call can re-enter guest code (run a
    /// thread entry). Set once, right after instantiation.
    instance: Option<Instance>,
}

/// Errors running a program end to end.
#[derive(Debug)]
pub enum RunError {
    Transpile(transpiler::Error),
    Wasm(String),
}

impl From<transpiler::Error> for RunError {
    fn from(e: transpiler::Error) -> Self {
        RunError::Transpile(e)
    }
}
impl From<wasmtime::Error> for RunError {
    fn from(e: wasmtime::Error) -> Self {
        RunError::Wasm(e.to_string())
    }
}
impl From<wasmtime::MemoryAccessError> for RunError {
    fn from(e: wasmtime::MemoryAccessError) -> Self {
        RunError::Wasm(e.to_string())
    }
}

/// A sentinel error a host handler returns to unwind the wasm stack on `exit`.
#[derive(Debug)]
struct HaltUnwind;
impl std::fmt::Display for HaltUnwind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "guest halted")
    }
}
impl std::error::Error for HaltUnwind {}

/// The error the `dispatch_miss` host trap returns when an indirect call resolves to
/// no translated function. Carries the faulting target and its caller so the trap
/// message pinpoints the unmapped `blx`/`bx` target instead of a bare `unreachable`.
#[derive(Debug)]
struct DispatchMiss {
    target: u32,
    caller: u32,
}
impl std::fmt::Display for DispatchMiss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "indirect dispatch to unknown target {:#010x} from f_{:x}",
            self.target, self.caller
        )
    }
}
impl std::error::Error for DispatchMiss {}

/// A transpiled module, instantiated and ready for host-driven execution.
pub struct Vm {
    store: Store<Host>,
    instance: Instance,
    base: u32,
}

/// The result of a fuel-bounded guest call ([`Vm::call_bounded`]).
#[derive(Debug, PartialEq, Eq)]
pub enum CallOutcome {
    /// The function returned (or a host handler cleanly halted the run).
    Returned,
    /// The fuel budget was exhausted mid-execution (still-running loop).
    OutOfFuel,
    /// A real trap (unreachable stub, memory fault, ...); carries the detail.
    Trap(String),
}

impl Vm {
    /// Transpile `code` (loaded at `base`) and instantiate it under `host_abi`.
    /// The image is seeded into guest memory (rebased) and sp is set to the top
    /// of the `mem_bytes` region. `externs` wires NID import stubs.
    pub fn new(
        code: &[u8],
        base: u32,
        thumb: bool,
        entries: &[u32],
        externs: &[transpiler::Extern],
        mem_bytes: u32,
        host_abi: &HostAbi,
    ) -> Result<Vm, RunError> {
        let artifact = transpiler::transpile(&transpiler::Program {
            code,
            base,
            thumb,
            entries,
            arm_entries: &[],
            externs,
            redirects: &[],
            noreturn_svc: host_abi.noreturn_svc,
            mem_bytes,
            // Vita modules take function addresses (thread entries, GXM
            // callbacks); discover them. Safe for the ARM corpus too - those
            // cases materialize no code pointers, so the closure is unchanged.
            discover_code_pointers: true,
            // The single-instance `Vm` defines its own memory.
            import_memory: false,
        })?;

        // Validate first for a precise error (wasmtime only names the function).
        wasmparser::validate(&artifact.wasm)
            .map_err(|e| RunError::Wasm(format!("invalid module: {e}")))?;

        let engine = Engine::default();
        let module = Module::from_binary(&engine, &artifact.wasm)?;
        let mut store = Store::new(
            &engine,
            Host {
                output: Vec::new(),
                halted: false,
                base,
                import_fn: host_abi.import,
                import_env: None,
                instance: None,
            },
        );

        let mut linker = Linker::new(&engine);
        bind_host(&mut linker, abi::SVC_NAME, host_abi.svc)?;
        bind_import(&mut linker)?;
        bind_dispatch_miss(&mut linker)?;
        let instance = linker.instantiate(&mut store, &module)?;
        // Record the instance so an import handler can re-enter guest code.
        store.data_mut().instance = Some(instance);

        let mut vm = Vm { store, instance, base };
        vm.write_mem(base, code)?;
        vm.set_reg(abi::SP, base.wrapping_add(mem_bytes));
        Ok(vm)
    }

    /// Instantiate a multi-module linked title ([`vitaslop_runtime::link::LinkedProgram`])
    /// for a headless boot. Unlike [`Vm::new`] this transpiles *leniently* (a
    /// handful of still-unlifted functions become trapping stubs so the whole game
    /// still builds), carries the inter-module `redirects`, seeds the combined
    /// image, and runs under a fuel budget so a non-terminating render loop can be
    /// bounded. Returns the VM and the addresses that became stubs.
    pub fn from_linked(
        linked: &vitaslop_runtime::link::LinkedProgram,
        host_abi: &HostAbi,
    ) -> Result<(Vm, Vec<(u32, u32)>), RunError> {
        let built = transpiler::transpile_lenient(&linked.program());
        wasmparser::validate(&built.artifact.wasm)
            .map_err(|e| RunError::Wasm(format!("invalid module: {e}")))?;

        let engine = Engine::new(Config::new().consume_fuel(true))
            .map_err(|e| RunError::Wasm(e.to_string()))?;
        let module = Module::from_binary(&engine, &built.artifact.wasm)?;
        let mut store = Store::new(
            &engine,
            Host {
                output: Vec::new(),
                halted: false,
                base: linked.base,
                import_fn: host_abi.import,
                import_env: None,
                instance: None,
            },
        );
        store.set_fuel(u64::MAX).map_err(|e| RunError::Wasm(e.to_string()))?;

        let mut linker = Linker::new(&engine);
        bind_host(&mut linker, abi::SVC_NAME, host_abi.svc)?;
        bind_import(&mut linker)?;
        bind_dispatch_miss(&mut linker)?;
        let instance = linker.instantiate(&mut store, &module)?;
        store.data_mut().instance = Some(instance);

        let mut vm = Vm { store, instance, base: linked.base };
        vm.write_mem(linked.base, &linked.image)?;
        vm.set_reg(abi::SP, linked.base.wrapping_add(linked.mem_bytes));
        // Report stubs as (guest addr, wasm function index) so a trap backtrace can
        // be attributed to a known stub vs a real miscompile.
        let stubs = built
            .stubbed
            .iter()
            .copied()
            .zip(built.stub_wasm_indices.iter().copied())
            .collect();
        Ok((vm, stubs))
    }

    /// Call the guest function at `addr` with a bounded instruction budget (`fuel`),
    /// so a render loop that never returns still stops. Distinguishes a clean
    /// return/halt from budget exhaustion from a real trap - the boot probe treats
    /// exhaustion as "still running", not a failure.
    pub fn call_bounded(&mut self, addr: u32, fuel: u64) -> CallOutcome {
        self.store.set_fuel(fuel).ok();
        let func = match self
            .instance
            .get_typed_func::<(), ()>(&mut self.store, &abi::func_export(addr))
        {
            Ok(f) => f,
            Err(e) => return CallOutcome::Trap(e.to_string()),
        };
        match func.call(&mut self.store, ()) {
            Ok(()) => CallOutcome::Returned,
            Err(e) => {
                if self.store.data().halted {
                    CallOutcome::Returned
                } else if e.downcast_ref::<wasmtime::Trap>() == Some(&wasmtime::Trap::OutOfFuel) {
                    CallOutcome::OutOfFuel
                } else {
                    let detail = match e.downcast_ref::<wasmtime::Trap>() {
                        Some(t) => format!("{t:?}: {e}"),
                        None => e.to_string(),
                    };
                    CallOutcome::Trap(detail)
                }
            }
        }
    }

    /// Attach a stateful Vita import environment. Once set, the `env.import` trap
    /// routes NID calls to it instead of the fn-pointer handler. Set before the
    /// first `call`.
    pub fn set_import_env(&mut self, env: Box<dyn ImportDispatch>) {
        self.store.data_mut().import_env = Some(env);
    }

    /// Reclaim the import environment after a run (to read its captured state).
    pub fn take_import_env(&mut self) -> Option<Box<dyn ImportDispatch>> {
        self.store.data_mut().import_env.take()
    }

    /// Write `bytes` at guest address `addr`.
    pub fn write_mem(&mut self, addr: u32, bytes: &[u8]) -> Result<(), RunError> {
        let mem = self.memory();
        mem.write(&mut self.store, (addr - self.base) as usize, bytes)?;
        Ok(())
    }

    /// Read `len` bytes at guest address `addr`.
    pub fn read_mem(&mut self, addr: u32, len: usize) -> Result<Vec<u8>, RunError> {
        let mem = self.memory();
        let mut buf = vec![0u8; len];
        mem.read(&mut self.store, (addr - self.base) as usize, &mut buf)?;
        Ok(buf)
    }

    /// Set guest register `i`.
    pub fn set_reg(&mut self, i: usize, v: u32) {
        self.global(&abi::reg_export(i)).set(&mut self.store, Val::I32(v as i32)).unwrap();
    }

    /// Get guest register `i`.
    pub fn get_reg(&mut self, i: usize) -> u32 {
        self.global(&abi::reg_export(i)).get(&mut self.store).i32().unwrap() as u32
    }

    /// Set single-precision VFP register S`n` from an f32.
    pub fn set_s(&mut self, n: u8, v: f32) {
        self.global(&abi::vfp_s_export(n))
            .set(&mut self.store, Val::I32(v.to_bits() as i32))
            .unwrap();
    }

    /// Read single-precision VFP register S`n` as an f32.
    pub fn get_s(&mut self, n: u8) -> f32 {
        f32::from_bits(self.global(&abi::vfp_s_export(n)).get(&mut self.store).i32().unwrap() as u32)
    }

    /// Read the raw 32 bits of VFP register S`n`.
    pub fn get_s_bits(&mut self, n: u8) -> u32 {
        self.global(&abi::vfp_s_export(n)).get(&mut self.store).i32().unwrap() as u32
    }

    /// Read the N,Z,C,V flags.
    pub fn flags(&mut self) -> Flags {
        let mut f = |flag| {
            self.global(abi::flag_export(flag)).get(&mut self.store).i32().unwrap() != 0
        };
        Flags {
            n: f(abi::Flag::N),
            z: f(abi::Flag::Z),
            c: f(abi::Flag::C),
            v: f(abi::Flag::V),
        }
    }

    /// The captured host output so far.
    pub fn output(&self) -> &[u8] {
        &self.store.data().output
    }

    /// Call the guest function exported at `addr`, running until it returns or a
    /// host handler halts it (`exit`). A clean halt is `Ok`.
    pub fn call(&mut self, addr: u32) -> Result<(), RunError> {
        let func = self
            .instance
            .get_typed_func::<(), ()>(&mut self.store, &abi::func_export(addr))?;
        match func.call(&mut self.store, ()) {
            Ok(()) => Ok(()),
            Err(e) => {
                if self.store.data().halted {
                    Ok(())
                } else {
                    let detail = match e.downcast_ref::<wasmtime::Trap>() {
                        Some(t) => format!("{t:?}: {e}"),
                        None => e.to_string(),
                    };
                    Err(RunError::Wasm(detail))
                }
            }
        }
    }

    fn memory(&mut self) -> wasmtime::Memory {
        self.instance
            .get_memory(&mut self.store, abi::MEMORY_EXPORT)
            .expect("module exports memory")
    }
    fn global(&mut self, name: &str) -> wasmtime::Global {
        self.instance
            .get_global(&mut self.store, name)
            .expect("module exports the global")
    }
}

/// Bind a host handler (`svc` or `import`) as the named import.
fn bind_host(
    linker: &mut Linker<Host>,
    name: &'static str,
    handler: SvcHandler,
) -> Result<(), RunError> {
    linker.func_wrap(
        abi::IMPORT_MODULE,
        name,
        move |mut caller: Caller<'_, Host>, selector: i32| -> Result<(), wasmtime::Error> {
            let mut regs = [0u32; abi::REG_COUNT];
            for (i, r) in regs.iter_mut().enumerate() {
                *r = read_reg(&mut caller, i);
            }
            let mem = caller
                .get_export(abi::MEMORY_EXPORT)
                .and_then(|e| e.into_memory())
                .expect("module exports memory");
            let outcome = {
                let (bytes, host) = mem.data_and_store_mut(&mut caller);
                handler(selector as u32, &mut regs, bytes, host.base, &mut host.output)
            };
            // Write back any registers the handler set (e.g. the r0 return value).
            for (i, &v) in regs.iter().enumerate() {
                write_reg(&mut caller, i, v);
            }
            if let SvcOutcome::Halt = outcome {
                caller.data_mut().halted = true;
                return Err(HaltUnwind.into());
            }
            Ok(())
        },
    )?;
    Ok(())
}

/// Bind the `env.import` NID trap. Routes to the stateful import environment if
/// one is attached, else to the fn-pointer fallback in `Host`.
fn bind_import(linker: &mut Linker<Host>) -> Result<(), RunError> {
    linker.func_wrap(
        abi::IMPORT_MODULE,
        abi::IMPORT_NAME,
        move |mut caller: Caller<'_, Host>, selector: i32| -> Result<(), wasmtime::Error> {
            let mut regs = [0u32; abi::REG_COUNT];
            for (i, r) in regs.iter_mut().enumerate() {
                *r = read_reg(&mut caller, i);
            }
            // The Vita is hardfloat, so the NID path also carries the VFP argument
            // registers (s0..s15). Read them alongside the core registers so a
            // handler can marshal float args and returns.
            let mut vfp = [0u32; vitaslop_runtime::VFP_ARG_COUNT];
            for (i, s) in vfp.iter_mut().enumerate() {
                *s = read_vfp(&mut caller, i);
            }
            let mem = caller
                .get_export(abi::MEMORY_EXPORT)
                .and_then(|e| e.into_memory())
                .expect("module exports memory");
            let outcome = {
                let (bytes, host) = mem.data_and_store_mut(&mut caller);
                let base = host.base;
                match host.import_env.as_mut() {
                    // Native wasmtime shares one address space with the guest, so
                    // the rebased slice backs `GuestMemory` directly (zero-copy).
                    Some(env) => {
                        let mut mem = vitaslop_runtime::SliceMemory(bytes);
                        env.dispatch(selector as u32, &mut regs, &mut vfp, &mut mem, base)
                    }
                    None => (host.import_fn)(selector as u32, &mut regs, bytes, base, &mut host.output),
                }
            };
            for (i, &v) in regs.iter().enumerate() {
                write_reg(&mut caller, i, v);
            }
            for (i, &v) in vfp.iter().enumerate() {
                write_vfp(&mut caller, i, v);
            }
            if let SvcOutcome::Halt = outcome {
                caller.data_mut().halted = true;
                return Err(HaltUnwind.into());
            }
            // Drain any synchronous guest re-entries the call raised (a thread
            // start). Each runs the thread entry to completion on its own stack,
            // transparently to the interrupted thread: the whole register file is
            // saved and restored around the call, so only the entry's return value
            // (captured before the restore) and its side effects persist.
            run_reentries(&mut caller);
            Ok(())
        },
    )?;
    Ok(())
}

/// Bind `env.dispatch_miss`: an indirect call whose target matches no translated
/// function traps here with the faulting `(target, caller)` addresses, so an
/// unmapped `blx`/`bx` target is a one-line report instead of an opaque
/// `unreachable`. (A future lazy-discovery path could compile the target and resume
/// here instead of trapping.)
fn bind_dispatch_miss(linker: &mut Linker<Host>) -> Result<(), RunError> {
    linker.func_wrap(
        abi::IMPORT_MODULE,
        abi::DISPATCH_MISS_NAME,
        move |_caller: Caller<'_, Host>, target: i32, caller: i32| -> Result<(), wasmtime::Error> {
            Err(DispatchMiss { target: target as u32, caller: caller as u32 }.into())
        },
    )?;
    Ok(())
}

/// Run every pending guest re-entry the last import call raised.
fn run_reentries(caller: &mut Caller<'_, Host>) {
    while let Some(re) = caller.data_mut().import_env.as_mut().and_then(|e| e.take_reentry()) {
        // Save the full register context of the interrupted thread.
        let saved_regs: [u32; abi::REG_COUNT] =
            std::array::from_fn(|i| read_reg(caller, i));
        let saved_vfp: [u32; vitaslop_runtime::VFP_ARG_COUNT] =
            std::array::from_fn(|i| read_vfp(caller, i));
        // The worker ending itself (a normal return, or an exit-thread halt) must
        // not end the interrupted thread, so shield its halt flag too.
        let saved_halted = caller.data().halted;

        // Seed the entry's arguments (r0 = arg length, r1 = arg pointer) and its
        // own stack, then run it to completion.
        write_reg(caller, 0, re.arg_len);
        write_reg(caller, 1, re.arg_ptr);
        write_reg(caller, abi::SP, re.stack_top);
        let export = abi::func_export(re.entry);
        let worker_ret = if let Some(instance) = caller.data().instance {
            match instance.get_typed_func::<(), ()>(&mut *caller, &export) {
                Ok(func) => {
                    // A clean return or a guest halt inside the worker both end
                    // the entry; either way its r0 is the return value.
                    let _ = func.call(&mut *caller, ());
                    read_reg(caller, 0)
                }
                // The entry was never transpiled (should not happen: thread
                // entries are discovered as code pointers). Report 0.
                Err(_) => 0,
            }
        } else {
            0
        };

        // Restore the interrupted thread's context and record the result.
        for (i, &v) in saved_regs.iter().enumerate() {
            write_reg(caller, i, v);
        }
        for (i, &v) in saved_vfp.iter().enumerate() {
            write_vfp(caller, i, v);
        }
        caller.data_mut().halted = saved_halted;
        if let Some(env) = caller.data_mut().import_env.as_mut() {
            env.set_thread_exit(re.thid, worker_ret);
        }
    }
}

fn read_reg(caller: &mut Caller<'_, Host>, i: usize) -> u32 {
    caller
        .get_export(&abi::reg_export(i))
        .and_then(|e| e.into_global())
        .expect("module exports registers")
        .get(&mut *caller)
        .i32()
        .expect("register global is i32") as u32
}

fn write_reg(caller: &mut Caller<'_, Host>, i: usize, v: u32) {
    caller
        .get_export(&abi::reg_export(i))
        .and_then(|e| e.into_global())
        .expect("module exports registers")
        .set(&mut *caller, Val::I32(v as i32))
        .expect("register global is mutable i32");
}

fn read_vfp(caller: &mut Caller<'_, Host>, i: usize) -> u32 {
    caller
        .get_export(&abi::vfp_s_export(i as u8))
        .and_then(|e| e.into_global())
        .expect("module exports vfp registers")
        .get(&mut *caller)
        .i32()
        .expect("vfp global is i32") as u32
}

fn write_vfp(caller: &mut Caller<'_, Host>, i: usize, v: u32) {
    caller
        .get_export(&abi::vfp_s_export(i as u8))
        .and_then(|e| e.into_global())
        .expect("module exports vfp registers")
        .set(&mut *caller, Val::I32(v as i32))
        .expect("vfp global is mutable i32");
}

/// Transpile an ARM code image, run it from `entry`, and return the final
/// register file, flags, and output. Seeds `(index, value)` registers.
pub fn run_arm(
    code: &[u8],
    base: u32,
    entry: u32,
    in_regs: &[(usize, u32)],
    host_abi: &HostAbi,
) -> Result<RunResult, RunError> {
    run(code, base, entry, false, &[], in_regs, host_abi)
}

/// The general conformance driver: `thumb` selects decode mode, `externs` wires
/// NID import stubs.
pub fn run(
    code: &[u8],
    base: u32,
    entry: u32,
    thumb: bool,
    externs: &[transpiler::Extern],
    in_regs: &[(usize, u32)],
    host_abi: &HostAbi,
) -> Result<RunResult, RunError> {
    let mut vm = Vm::new(code, base, thumb, &[entry], externs, DEFAULT_MEM_BYTES, host_abi)?;
    for &(i, v) in in_regs {
        vm.set_reg(i, v);
    }
    vm.call(entry)?;
    let mut regs = [0u32; abi::REG_COUNT];
    for (i, r) in regs.iter_mut().enumerate() {
        *r = vm.get_reg(i);
    }
    let flags = vm.flags();
    Ok(RunResult { regs, flags, output: vm.output().to_vec() })
}

#[cfg(test)]
mod switch_tests {
    use super::{run, HostAbi};

    /// A real GCC-shape Thumb-2 `tbh` switch (assembled with the Vita toolchain):
    ///   cmp r0,#3 ; bhi .Ldefault ; tbh [pc, r0, lsl #1] ; <4-entry table> ;
    ///   case0: r0=10 ; case1: r0=20 ; case2: r0=30 ; case3: r0=40 ; default: r0=99.
    /// This exercises the computed-jump terminator end to end on the real engine:
    /// each in-range index must dispatch to its own case body, and an out-of-range
    /// index must take the range-check branch to the default. A depth or index bug
    /// in the `br_table` landing pads would send a case to the wrong body.
    const SWITCH: [u8; 36] = [
        0x03, 0x28, 0x0d, 0xd8, 0xdf, 0xe8, 0x10, 0xf0, 0x04, 0x00, 0x06, 0x00, 0x08, 0x00, 0x0a,
        0x00, 0x0a, 0x20, 0x70, 0x47, 0x14, 0x20, 0x70, 0x47, 0x1e, 0x20, 0x70, 0x47, 0x28, 0x20,
        0x70, 0x47, 0x63, 0x20, 0x70, 0x47,
    ];

    #[test]
    fn tbh_switch_dispatches_each_case() {
        let base = 0x10000u32;
        let abi = HostAbi::default();
        // In-range indices land on their own case body.
        for (idx, want) in [(0u32, 10u32), (1, 20), (2, 30), (3, 40)] {
            let r = run(&SWITCH, base, base, true, &[], &[(0, idx)], &abi).expect("run");
            assert_eq!(r.regs[0], want, "tbh index {idx} dispatched wrong");
        }
        // Out-of-range indices take the range-check branch to the default body.
        for idx in [4u32, 7, 100] {
            let r = run(&SWITCH, base, base, true, &[], &[(0, idx)], &abi).expect("run");
            assert_eq!(r.regs[0], 99, "out-of-range index {idx} should hit default");
        }
    }

    // Each of the following is a Vita-toolchain-assembled Thumb-2 function that
    // exercises one newly-lifted instruction family end to end on the engine, and
    // is checked against a reference computed here. All return their result in r0
    // (or r0:r1 for a 64-bit result). Base and `bx lr` return are shared.
    fn run1(code: &[u8], ins: &[(usize, u32)]) -> [u32; 16] {
        let base = 0x10000u32;
        run(code, base, base, true, &[], ins, &HostAbi::default()).expect("run").regs
    }

    #[test]
    fn blx_lr_dispatches_to_target_not_return() {
        // A compiler sometimes uses `lr` as the scratch that holds an indirect call
        // target (`blx lr`). The call itself writes the return address into `lr`, so
        // the target MUST be captured before `lr` is overwritten - otherwise the
        // dispatch goes to the return address (mid-function) and traps. This assembles:
        //   start: push {r7,lr}; mov lr,r0; blx lr; pop {r7,pc}
        //   target(@+8): movs r0,#0x42; bx lr
        // with r0 seeded to the target (Thumb bit set). A correct `blx lr` dispatches
        // to `target`, which returns 0x42; the pre-fix behaviour dispatched to the
        // `pop` address and trapped in the dispatcher (UnreachableCodeReached).
        let code = [
            0x80, 0xb5, // push {r7, lr}
            0x86, 0x46, // mov lr, r0
            0xf0, 0x47, // blx lr
            0x80, 0xbd, // pop {r7, pc}
            0x42, 0x20, // movs r0, #0x42
            0x70, 0x47, // bx lr
        ];
        let base = 0x10000u32;
        let target = base + 0x08;
        let mut vm = super::Vm::new(
            &code,
            base,
            true,
            &[base, target],
            &[],
            super::DEFAULT_MEM_BYTES,
            &HostAbi::default(),
        )
        .expect("new");
        vm.set_reg(0, target | 1); // r0 = target with the Thumb bit set
        vm.call(base).expect("blx lr must dispatch to the target, not trap");
        assert_eq!(vm.get_reg(0), 0x42, "blx lr dispatched to the wrong function");
    }

    #[test]
    fn arm_forward_branch_lands_exactly() {
        // ARM mode (thumb=false), assembled with the Vita toolchain:
        //   b good ; mov r0,#99 ; good: mov r0,#7 ; bx lr
        // The forward branch must land exactly on `good`; the historical ARM bug
        // added the pc+8 bias twice, landing 8 bytes off and running the wrong path.
        let code = [
            0x00, 0x00, 0x00, 0xea, // b good
            0x63, 0x00, 0xa0, 0xe3, // mov r0, #99  (must be skipped)
            0x07, 0x00, 0xa0, 0xe3, // good: mov r0, #7
            0x1e, 0xff, 0x2f, 0xe1, // bx lr
        ];
        let base = 0x10000u32;
        let r = run(&code, base, base, false, &[], &[], &HostAbi::default()).expect("run");
        assert_eq!(r.regs[0], 7, "ARM forward branch mis-targeted");
    }

    #[test]
    fn arm_backward_branch_loop() {
        // ARM mode: mov r0,#3 ; loop: subs r0,r0,#1 ; bne loop ; bx lr
        // A backward conditional branch whose target must be exact for the loop to
        // terminate at zero rather than run away or exit early.
        let code = [
            0x03, 0x00, 0xa0, 0xe3, // mov r0, #3
            0x01, 0x00, 0x50, 0xe2, // loop: subs r0, r0, #1
            0xfd, 0xff, 0xff, 0x1a, // bne loop
            0x1e, 0xff, 0x2f, 0xe1, // bx lr
        ];
        let base = 0x10000u32;
        let r = run(&code, base, base, false, &[], &[], &HostAbi::default()).expect("run");
        assert_eq!(r.regs[0], 0, "ARM loop did not terminate at zero");
    }

    #[test]
    fn rbit_reverses_bits() {
        // rbit r0, r0 ; bx lr
        let code = [0x90, 0xfa, 0xa0, 0xf0, 0x70, 0x47];
        for x in [0x0000_0001u32, 0x1234_5678, 0xFFFF_0000, 0xDEAD_BEEF, 0] {
            assert_eq!(run1(&code, &[(0, x)])[0], x.reverse_bits(), "rbit {x:#x}");
        }
    }

    #[test]
    fn orn_or_not() {
        // orn r0, r0, r1 ; bx lr
        let code = [0x60, 0xea, 0x01, 0x00, 0x70, 0x47];
        for (a, b) in [(0x0f0f_0f0fu32, 0x00ff_00ffu32), (0, 0xFFFF_FFFF), (0xAAAA_AAAA, 0x5555_5555)] {
            assert_eq!(run1(&code, &[(0, a), (1, b)])[0], a | !b, "orn {a:#x},{b:#x}");
        }
    }

    // NEON logical ops on D registers assembled from r0:r1 and r2:r3, result in r0:r1.
    const NEON_INPUTS: [(u32, u32, u32, u32); 3] = [
        (0x0f0f_0f0f, 0xffff_0000, 0x00ff_00ff, 0x1234_5678),
        (0xdead_beef, 0xcafe_babe, 0x0000_ffff, 0xffff_0000),
        (0, 0xFFFF_FFFF, 0xFFFF_FFFF, 0),
    ];

    #[test]
    fn neon_vand_vorr_vbic() {
        let vand = [0x41, 0xec, 0x10, 0x0b, 0x43, 0xec, 0x11, 0x2b, 0x00, 0xef, 0x11, 0x01, 0x51, 0xec, 0x10, 0x0b, 0x70, 0x47];
        let vorr = [0x41, 0xec, 0x10, 0x0b, 0x43, 0xec, 0x11, 0x2b, 0x20, 0xef, 0x11, 0x01, 0x51, 0xec, 0x10, 0x0b, 0x70, 0x47];
        let vbic = [0x41, 0xec, 0x10, 0x0b, 0x43, 0xec, 0x11, 0x2b, 0x10, 0xef, 0x11, 0x01, 0x51, 0xec, 0x10, 0x0b, 0x70, 0x47];
        for (lo0, hi0, lo1, hi1) in NEON_INPUTS {
            let ins = [(0, lo0), (1, hi0), (2, lo1), (3, hi1)];
            let a = run1(&vand, &ins);
            assert_eq!((a[0], a[1]), (lo0 & lo1, hi0 & hi1), "vand");
            let o = run1(&vorr, &ins);
            assert_eq!((o[0], o[1]), (lo0 | lo1, hi0 | hi1), "vorr");
            let b = run1(&vbic, &ins);
            assert_eq!((b[0], b[1]), (lo0 & !lo1, hi0 & !hi1), "vbic");
        }
    }

    #[test]
    fn vfp_double_add() {
        // vmov d0,r0,r1 ; vmov d1,r2,r3 ; vadd.f64 d0,d0,d1 ; vmov r0,r1,d0 ; bx lr
        let code = [0x41, 0xec, 0x10, 0x0b, 0x43, 0xec, 0x11, 0x2b, 0x30, 0xee, 0x01, 0x0b, 0x51, 0xec, 0x10, 0x0b, 0x70, 0x47];
        for (x, y) in [(1.5f64, 2.25f64), (-3.0, 0.125), (1e300, 1e300)] {
            let (xb, yb) = (x.to_bits(), y.to_bits());
            let ins = [(0, xb as u32), (1, (xb >> 32) as u32), (2, yb as u32), (3, (yb >> 32) as u32)];
            let r = run1(&code, &ins);
            let got = ((r[1] as u64) << 32) | r[0] as u64;
            assert_eq!(got, (x + y).to_bits(), "vadd.f64 {x}+{y}");
        }
    }

    #[test]
    fn vfp_double_cvt_roundtrip() {
        // vmov s0,r0 ; vcvt.f64.s32 d1,s0 ; vcvt.s32.f64 s2,d1 ; vmov r0,s2 ; bx lr
        let code = [0x00, 0xee, 0x10, 0x0a, 0xb8, 0xee, 0xc0, 0x1b, 0xbd, 0xee, 0xc1, 0x1b, 0x11, 0xee, 0x10, 0x0a, 0x70, 0x47];
        for x in [0i32, 1, -1, 123456, -987654, i32::MAX] {
            assert_eq!(run1(&code, &[(0, x as u32)])[0] as i32, x, "s32->f64->s32 {x}");
        }
        // vmov s0,r0 ; vcvt.f64.u32 d1,s0 ; vmov r0,r1,d1 ; bx lr
        let u2d = [0x00, 0xee, 0x10, 0x0a, 0xb8, 0xee, 0x40, 0x1b, 0x51, 0xec, 0x11, 0x0b, 0x70, 0x47];
        for x in [0u32, 1, 0x8000_0000, 0xFFFF_FFFF, 42] {
            let r = run1(&u2d, &[(0, x)]);
            let got = ((r[1] as u64) << 32) | r[0] as u64;
            assert_eq!(got, (x as f64).to_bits(), "u32->f64 {x}");
        }
    }

    // --- NEON single-element load/store (vld1/vst1 element forms) --------------
    //
    // These exercise the element load/store lift end to end. Eight bytes of test
    // data are placed at image offset 0x40, and r4 is seeded to its guest address;
    // the code loads from / stores to `[r4]` and reports through r0(:r1). The
    // instruction bytes were verified to decode against capstone (the certified
    // NEON structure-load/store decode).

    /// Data offset within the image and its guest address helper.
    const DATA_OFF: usize = 0x40;

    /// Run `code` with `data` placed at image offset 0x40 and r4 = its address,
    /// plus any extra input registers. Returns the register file.
    fn run_mem(code: &[u8], data: [u8; 8], regs: &[(usize, u32)]) -> [u32; 16] {
        let base = 0x10000u32;
        let mut image = code.to_vec();
        image.resize(DATA_OFF, 0);
        image.extend_from_slice(&data);
        let mut ins = vec![(4usize, base + DATA_OFF as u32)];
        ins.extend_from_slice(regs);
        run(&image, base, base, true, &[], &ins, &HostAbi::default()).expect("run").regs
    }

    #[test]
    fn vld1_lane_16_inserts_one_lane() {
        // vmov d0,r0,r1 ; vld1.16 {d0[1]},[r4] ; vmov r0,r1,d0 ; bx lr
        let code = [
            0x41, 0xec, 0x10, 0x0b, 0xa4, 0xf9, 0x4f, 0x04, 0x51, 0xec, 0x10, 0x0b, 0x70, 0x47,
        ];
        // data low 16 bits = 0xDEAD (little-endian bytes AD DE).
        let data = [0xAD, 0xDE, 0, 0, 0, 0, 0, 0];
        let r = run_mem(&code, data, &[(0, 0x1111_2222), (1, 0x3333_4444)]);
        // lane 1 (bits 16..31 of the low word) becomes 0xDEAD; the rest is untouched.
        assert_eq!(r[0], 0xDEAD_2222, "vld1.16 lane 1 low word");
        assert_eq!(r[1], 0x3333_4444, "vld1.16 high word untouched");
    }

    #[test]
    fn vld1_lane_8_inserts_one_byte() {
        // vmov d0,r0,r1 ; vld1.8 {d0[0]},[r4] ; vmov r0,r1,d0 ; bx lr
        let code = [
            0x41, 0xec, 0x10, 0x0b, 0xa4, 0xf9, 0x0f, 0x00, 0x51, 0xec, 0x10, 0x0b, 0x70, 0x47,
        ];
        let data = [0x5A, 0, 0, 0, 0, 0, 0, 0];
        let r = run_mem(&code, data, &[(0, 0x1111_22FF), (1, 0xAAAA_BBBB)]);
        assert_eq!(r[0], 0x1111_225A, "vld1.8 lane 0 replaces low byte only");
        assert_eq!(r[1], 0xAAAA_BBBB, "vld1.8 high word untouched");
    }

    #[test]
    fn vld1_all_lanes_32_broadcasts() {
        // vmov d0,r0,r1 ; vld1.32 {d0[]},[r4] ; vmov r0,r1,d0 ; bx lr
        let code = [
            0x41, 0xec, 0x10, 0x0b, 0xa4, 0xf9, 0x8f, 0x0c, 0x51, 0xec, 0x10, 0x0b, 0x70, 0x47,
        ];
        let data = [0xEF, 0xBE, 0xAD, 0xDE, 0, 0, 0, 0]; // 0xDEADBEEF
        let r = run_mem(&code, data, &[(0, 0), (1, 0)]);
        // both 32-bit lanes become the loaded word.
        assert_eq!(r[0], 0xDEAD_BEEF, "vld1.32 broadcast low lane");
        assert_eq!(r[1], 0xDEAD_BEEF, "vld1.32 broadcast high lane");
    }

    #[test]
    fn vld1_all_lanes_16_broadcasts() {
        // vld1.16 {d0[]},[r4]: replicate the 16-bit element across all four lanes.
        // vmov d0,r0,r1 ; vld1.16 {d0[]},[r4] ; vmov r0,r1,d0 ; bx lr
        let code = [
            0x41, 0xec, 0x10, 0x0b, 0xa4, 0xf9, 0x4f, 0x0c, 0x51, 0xec, 0x10, 0x0b, 0x70, 0x47,
        ];
        let data = [0x34, 0x12, 0, 0, 0, 0, 0, 0]; // 0x1234
        let r = run_mem(&code, data, &[(0, 0), (1, 0)]);
        assert_eq!(r[0], 0x1234_1234, "vld1.16 broadcast low word");
        assert_eq!(r[1], 0x1234_1234, "vld1.16 broadcast high word");
    }

    /// Reference IEEE half-precision (f16) -> f32, via float arithmetic (exact for
    /// every finite f16). Used to check the branchless lift.
    fn f16_to_f32_ref(h: u16) -> f32 {
        let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
        let exp = ((h >> 10) & 0x1f) as i32;
        let mant = (h & 0x3ff) as f32;
        if exp == 0 {
            sign * mant * 2f32.powi(-24) // subnormal (zero when mant == 0)
        } else if exp == 0x1f {
            if mant == 0.0 { sign * f32::INFINITY } else { f32::NAN }
        } else {
            sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
        }
    }

    #[test]
    fn vcvtb_f32_from_f16_matches_reference() {
        // vmov s0,r0 ; vcvtb.f32.f16 s0,s0 ; vmov r0,s0 ; bx lr
        let code = [
            0x00, 0xee, 0x10, 0x0a, 0xb2, 0xee, 0x40, 0x0a, 0x10, 0xee, 0x10, 0x0a, 0x70, 0x47,
        ];
        // Edge cases + a broad stride sweep of the 16-bit space (a full sweep would be
        // 65536 wasm instantiations).
        let mut cases: Vec<u16> = vec![
            0x0000, 0x8000, // +/-0
            0x3c00, 0xbc00, // +/-1.0
            0x0001, 0x8001, // smallest subnormal +/-
            0x03ff, 0x83ff, // largest subnormal +/-
            0x0400, // smallest normal
            0x7bff, 0xfbff, // largest finite +/-
            0x7c00, 0xfc00, // +/-inf
            0x7e00, 0x7c01, 0xfe00, // NaNs
            0x3555, 0x4248, 0xc248, // ~1/3, ~pi, ~-pi
        ];
        cases.extend((0..65536u32).step_by(137).map(|x| x as u16));
        for h in cases {
            let got_bits = run1(&code, &[(0, h as u32)])[0];
            let got = f32::from_bits(got_bits);
            let want = f16_to_f32_ref(h);
            if want.is_nan() {
                assert!(got.is_nan(), "vcvtb f16 {h:#06x} should be NaN, got {got_bits:#010x}");
            } else {
                assert_eq!(got_bits, want.to_bits(), "vcvtb f16 {h:#06x} -> {got} want {want}");
            }
        }
    }

    #[test]
    fn vcvtt_f32_from_f16_uses_top_half() {
        // vmov s0,r0 ; vcvtt.f32.f16 s0,s0 ; vmov r0,s0 ; bx lr
        // vcvtt reads the TOP 16 bits of s0 as the f16.
        let code = [
            0x00, 0xee, 0x10, 0x0a, 0xb2, 0xee, 0xc0, 0x0a, 0x10, 0xee, 0x10, 0x0a, 0x70, 0x47,
        ];
        for h in [0x3c00u16, 0xc248, 0x0001, 0x7bff] {
            // put the f16 in the top half; the bottom half must be ignored.
            let input = ((h as u32) << 16) | 0xBEEF;
            let got = f32::from_bits(run1(&code, &[(0, input)])[0]);
            assert_eq!(got.to_bits(), f16_to_f32_ref(h).to_bits(), "vcvtt f16 {h:#06x}");
        }
    }

    #[test]
    fn vst1_lane_16_stores_one_lane() {
        // vmov d0,r0,r1 ; vst1.16 {d0[1]},[r4] ; ldrh r0,[r4] ; bx lr
        let code = [
            0x41, 0xec, 0x10, 0x0b, 0x84, 0xf9, 0x4f, 0x04, 0x20, 0x88, 0x70, 0x47,
        ];
        let r = run_mem(&code, [0; 8], &[(0, 0xBEEF_1234), (1, 0)]);
        // lane 1 = bits 16..31 of the low word = 0xBEEF; ldrh reads it back.
        assert_eq!(r[0], 0xBEEF, "vst1.16 stored lane 1");
    }
}
