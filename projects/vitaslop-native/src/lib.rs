//! Native host mechanisms shared by every non-browser host (desktop and, one
//! day, mobile): the wasmtime-backed engine that runs the transpiler's WASM,
//! and later the mmap image source and worker threads. Never compiled to
//! wasm32, so no engine cfg gymnastics.

pub use vitaslop_runtime::{RunResult, SvcOutcome};
use vitaslop_transpiler::{self as transpiler, abi};
use wasmtime::{Caller, Engine, Linker, Module, Store, Val};

/// A host `svc` handler: given the syscall number and args (guest r7, r0, r1,
/// r2), the guest memory, and the output sink, service the call and say whether
/// to continue or halt.
pub type SvcHandler =
    fn(nr: u32, r0: u32, r1: u32, r2: u32, mem: &[u8], out: &mut Vec<u8>) -> SvcOutcome;

/// The host ABI a run uses. Injected by the caller so the engine carries no
/// syscall convention of its own: the arm conformance harness passes a Linux
/// one, Vita will pass a NID-based one.
pub struct HostAbi<'a> {
    /// Syscall numbers (guest r7) that do not return, so the transpiler can end a
    /// block at a `svc` with a statically-known one of them.
    pub noreturn_svc: &'a [u32],
    /// Services a `svc`.
    pub svc: SvcHandler,
}

/// Host state threaded through a run: captured output and the exit flag.
struct Host {
    output: Vec<u8>,
    halted: bool,
}

/// Errors running an arm program end to end.
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

/// Transpile an arm code image to WASM, run it via wasmtime under the ABI, and
/// return the final register file and captured output. `in_regs` seeds
/// `(index, value)` registers before entry; the rest default to 0.
pub fn run_arm(
    code: &[u8],
    base: u32,
    entry: u32,
    in_regs: &[(usize, u32)],
    host_abi: &HostAbi,
) -> Result<RunResult, RunError> {
    let artifact = transpiler::transpile(&transpiler::Program {
        code,
        base,
        entries: &[entry],
        externs: &[],
        noreturn_svc: host_abi.noreturn_svc,
    })?;

    let engine = Engine::default();
    let module = Module::from_binary(&engine, &artifact.wasm)?;
    let mut store = Store::new(
        &engine,
        Host {
            output: Vec::new(),
            halted: false,
        },
    );

    let svc = host_abi.svc;
    let mut linker = Linker::new(&engine);
    linker.func_wrap(
        abi::SVC_MODULE,
        abi::SVC_NAME,
        move |mut caller: Caller<'_, Host>, _imm: i32| {
            // Registers live in globals; read the syscall number (r7) and args.
            let r7 = read_reg(&mut caller, 7);
            let r0 = read_reg(&mut caller, 0);
            let r1 = read_reg(&mut caller, 1);
            let r2 = read_reg(&mut caller, 2);
            let mem = caller
                .get_export(abi::MEMORY_EXPORT)
                .and_then(|e| e.into_memory())
                .expect("module exports memory");
            let (bytes, host) = mem.data_and_store_mut(&mut caller);
            if let SvcOutcome::Halt = svc(r7, r0, r1, r2, bytes, &mut host.output) {
                host.halted = true;
            }
        },
    )?;

    let instance = linker.instantiate(&mut store, &module)?;
    let mem = instance
        .get_memory(&mut store, abi::MEMORY_EXPORT)
        .expect("module exports memory");

    // Seed the guest image (identity-mapped) into memory, and the requested
    // registers into their globals.
    mem.write(&mut store, base as usize, code)?;
    for &(i, v) in in_regs {
        instance
            .get_global(&mut store, &abi::reg_export(i))
            .expect("module exports registers")
            .set(&mut store, Val::I32(v as i32))?;
    }

    // Drive block-by-block until a block returns HALT or a svc exits.
    let mut pc = entry;
    loop {
        let func = instance.get_typed_func::<(), i32>(&mut store, &abi::block_export(pc))?;
        let next = func.call(&mut store, ())?;
        if next == abi::HALT || store.data().halted {
            break;
        }
        pc = next as u32;
    }

    // Read the final register file back out of the globals.
    let mut regs = [0u32; abi::REG_COUNT];
    for (i, r) in regs.iter_mut().enumerate() {
        *r = instance
            .get_global(&mut store, &abi::reg_export(i))
            .expect("module exports registers")
            .get(&mut store)
            .i32()
            .expect("register global is i32") as u32;
    }
    Ok(RunResult {
        regs,
        output: store.into_data().output,
    })
}

/// Read guest register `i` out of its wasm global (used by the `svc` host).
fn read_reg(caller: &mut Caller<'_, Host>, i: usize) -> u32 {
    let g = caller
        .get_export(&abi::reg_export(i))
        .and_then(|e| e.into_global())
        .expect("module exports registers");
    g.get(&mut *caller).i32().expect("register global is i32") as u32
}
