//! >>> CODE FOUND AT RUN TIME, on the native engines: the shared translation record
//! (`vitaslop_runtime::runtime_code`) plus what wasmtime adds - each side module compiled once for
//! the process, and installed into the instance that missed by growing its table and
//! instantiating the module against that instance's own imports and exports.

use std::sync::Mutex;

use vitaslop_runtime::runtime_code as core;
use vitaslop_transpiler::{self as transpiler, abi};
use wasmtime::{Caller, Engine, Extern, Instance, Linker, Module, Ref};

pub use core::RuntimeStats;

/// A run's runtime translations on a native engine. Shared by every instance of the run.
pub struct RuntimeCode {
    core: core::RuntimeCode,
    engine: Engine,
    /// Side module `i`'s compiled form, made the first time an instance needs it.
    modules: Mutex<Vec<Option<Module>>>,
}

impl RuntimeCode {
    /// The record for a run of the main module made from `program` under `settings` (taken on
    /// the thread that made it), whose functions are `main_addrs` (ascending) and whose entry
    /// table is at layout offset `runtime_off`.
    pub fn new(
        engine: &Engine,
        program: transpiler::OwnedProgram,
        main_addrs: Vec<u32>,
        settings: transpiler::EmitSettings,
        runtime_off: u64,
    ) -> RuntimeCode {
        RuntimeCode {
            core: core::RuntimeCode::new(program, main_addrs, settings, runtime_off),
            engine: engine.clone(),
            modules: Mutex::new(Vec::new()),
        }
    }

    /// What has been translated so far.
    pub fn stats(&self) -> RuntimeStats {
        self.core.stats()
    }

    /// Side module `side`'s compiled form, compiling it from `wasm` the first time.
    fn module(&self, side: usize, wasm: &[u8]) -> Result<Module, String> {
        let mut modules = self.modules.lock().unwrap_or_else(|p| p.into_inner());
        if modules.len() <= side {
            modules.resize(side + 1, None);
        }
        if let Some(m) = &modules[side] {
            return Ok(m.clone());
        }
        let m = Module::from_binary(&self.engine, wasm).map_err(|e| format!("side module {side} does not compile: {e}"))?;
        modules[side] = Some(m.clone());
        Ok(m)
    }
}

/// The base of the calling instance's linear memory.
fn memory_base<T>(caller: &mut Caller<'_, T>) -> Option<*mut u8> {
    match caller.get_export(abi::MEMORY_EXPORT)? {
        Extern::Memory(m) => Some(m.data_ptr(&*caller)),
        Extern::SharedMemory(m) => Some(m.data().as_ptr() as *mut u8),
        _ => None,
    }
}

/// What a dispatch miss leaves for its engine to do.
pub enum Miss {
    /// The calling instance's table already holds the code: call this slot.
    Ready(i32),
    /// Instantiate `module` against `imports` in the calling store (which writes it into the
    /// table, already grown), then call `slot`.
    Install { slot: i32, module: Module, imports: Vec<Extern> },
}

/// The engine-independent half of `env.dispatch_miss`: translate or find the code at `target`
/// and get the CALLING instance's table ready for it. With no `runtime` record the miss is the
/// old loud trap.
pub fn prepare_miss<T>(
    caller: &mut Caller<'_, T>,
    runtime: Option<&RuntimeCode>,
    linker: Option<&Linker<T>>,
    target: u32,
    from: u32,
) -> Result<Miss, wasmtime::Error> {
    let (Some(runtime), Some(linker)) = (runtime, linker) else {
        return Err(wasmtime::Error::msg(format!(
            "indirect dispatch to unknown target {target:#010x} from f_{from:x}"
        )));
    };
    let mem = memory_base(caller).ok_or_else(|| wasmtime::Error::msg("the main module exports no memory"))?;
    // SAFETY: `mem` is the base of this run's linear memory, and the entry table lies inside its
    // layout at `table_off`.
    let resolved = unsafe { runtime.core.resolve(target, from, mem.add(runtime.core.table_off() as usize)) }
        .map_err(wasmtime::Error::msg)?;
    let table = caller
        .get_export(abi::TABLE_EXPORT)
        .and_then(Extern::into_table)
        .ok_or_else(|| wasmtime::Error::msg("the main module exports no table"))?;
    let (slot, slot_end) = (resolved.slot, resolved.slot_end);
    let size = table.size(&*caller);
    if size > u64::from(slot) && matches!(table.get(&mut *caller, u64::from(slot)), Some(Ref::Func(Some(_)))) {
        return Ok(Miss::Ready(slot as i32));
    }
    if size < u64::from(slot_end) {
        table.grow(&mut *caller, u64::from(slot_end) - size, Ref::Func(None))?;
    }
    let module = runtime.module(resolved.side, &resolved.wasm).map_err(wasmtime::Error::msg)?;
    let mut imports = Vec::new();
    for import in module.imports() {
        let found = if import.module() == abi::MAIN_MODULE {
            caller.get_export(import.name())
        } else {
            linker.get(&mut *caller, import.module(), import.name()).ok()
        };
        imports.push(found.ok_or_else(|| {
            wasmtime::Error::msg(format!("side module import {}.{} is not provided", import.module(), import.name()))
        })?);
    }
    Ok(Miss::Install { slot: slot as i32, module, imports })
}

/// `env.dispatch_miss` on a synchronous native engine: [`prepare_miss`], then install.
pub fn dispatch_miss<T>(
    caller: &mut Caller<'_, T>,
    runtime: Option<&RuntimeCode>,
    linker: Option<&Linker<T>>,
    target: u32,
    from: u32,
) -> Result<i32, wasmtime::Error> {
    match prepare_miss(caller, runtime, linker, target, from)? {
        Miss::Ready(slot) => Ok(slot),
        Miss::Install { slot, module, imports } => {
            Instance::new(&mut *caller, &module, &imports)?;
            Ok(slot)
        }
    }
}
