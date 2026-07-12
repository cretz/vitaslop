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
    capture, nid, render, CtrlFrame, DeterministicWorld, Flags, ImportDispatch, Record, Replay,
    RunResult, SvcOutcome, VitaEnv, VitaState, World, WorldEvent,
};

pub mod wgpu_render;
pub use wgpu_render::WgpuRenderer;
pub use vitaslop_transpiler::abi;
use vitaslop_transpiler::{self as transpiler};
use wasmtime::{Caller, Engine, Instance, Linker, Module, Store, Val};

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

/// A transpiled module, instantiated and ready for host-driven execution.
pub struct Vm {
    store: Store<Host>,
    instance: Instance,
    base: u32,
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
            externs,
            noreturn_svc: host_abi.noreturn_svc,
            mem_bytes,
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
            },
        );

        let mut linker = Linker::new(&engine);
        bind_host(&mut linker, abi::SVC_NAME, host_abi.svc)?;
        bind_import(&mut linker)?;
        let instance = linker.instantiate(&mut store, &module)?;

        let mut vm = Vm { store, instance, base };
        vm.write_mem(base, code)?;
        vm.set_reg(abi::SP, base.wrapping_add(mem_bytes));
        Ok(vm)
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
            let mem = caller
                .get_export(abi::MEMORY_EXPORT)
                .and_then(|e| e.into_memory())
                .expect("module exports memory");
            let outcome = {
                let (bytes, host) = mem.data_and_store_mut(&mut caller);
                let base = host.base;
                match host.import_env.as_mut() {
                    Some(env) => env.dispatch(selector as u32, &mut regs, bytes, base),
                    None => (host.import_fn)(selector as u32, &mut regs, bytes, base, &mut host.output),
                }
            };
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
