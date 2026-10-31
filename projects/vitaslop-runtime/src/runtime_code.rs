//! >>> CODE FOUND AT RUN TIME - the half every engine shares.
//!
//! The dispatcher misses when the guest branches to an address that is no translated function
//! and no entry of the runtime entry table (`vitaslop_transpiler::RUNTIME_ENTRY_PAIRS`). That
//! used to end the run. Some code is only ever reached that way - an interpreter's opcode
//! handlers, whose addresses the program computes from a base and a table of offsets, are the
//! standing example - so instead the miss comes here: the code at the target, with the rest of
//! the unseen stretch of code around it (guessed - see the transpiler's stretch sweep), is
//! translated into a SIDE MODULE (`vitaslop_transpiler::transpile_side`) and recorded in the
//! entry table, and the engine installs it into the instance that missed. Every other instance installs it when IT
//! first misses on it, so nothing has to stop the world. What an engine adds is compiling the
//! bytes and instantiating them (`vitaslop-native`'s `runtime_code`, the browser's).
//!
//! Nothing is kept across runs: the translations live in this process only. That is a rule, not
//! an omission - a cache here would hide what the miss path costs, and the miss path is the thing
//! that has to be fast.
//!
//! A run that keeps finding code is stopped by name at `RUNTIME_MODULE_LIMIT` side modules (or,
//! structurally, `RUNTIME_ENTRY_LIMIT` entries): past that it is not a few stretches of code
//! discovery could not see, it is a title generating code (which this does not support yet) or
//! a discovery regression, and either one must be looked at.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use vitaslop_transpiler as transpiler;

/// A run's runtime translations, shared by every instance and worker of the run.
pub struct RuntimeCode {
    program: transpiler::OwnedProgram,
    main_addrs: Vec<u32>,
    settings: transpiler::EmitSettings,
    /// Linear-memory byte offset of the runtime entry table: its layout offset plus the host
    /// offset the guest region sits at.
    table_off: u64,
    state: Mutex<State>,
}

/// One side module.
struct Side {
    wasm: Arc<[u8]>,
    /// Table slots `[.., slot_end)` hold its functions.
    slot_end: u32,
}

/// How many side modules a run may make before it is stopped by name - see the module doc.
/// MEASURED: an interpreter-driven title needs 5; this suite's embedded LuaJIT, whose library
/// functions are reached through tables built at run time and scattered over its image, 48.
pub const RUNTIME_MODULE_LIMIT: usize = 128;

/// How many entries the stretch sweep (`vitaslop_transpiler::transpile_side`) may add over a
/// whole run: half the entry table, so guessing can never crowd out code the guest asks for.
const SWEEP_BUDGET: usize = transpiler::RUNTIME_ENTRY_LIMIT as usize / 2;

#[derive(Default)]
struct State {
    /// Every runtime entry: target (Thumb bit clear) -> (table slot, side module).
    entries: HashMap<u32, (u32, usize)>,
    /// What the stretch sweeps have learned so far.
    memo: transpiler::SideMemo,
    /// Entries the stretch sweep added (the rest were asked for, or called by what was).
    swept: usize,
    sides: Vec<Side>,
    next_slot: u32,
    /// Wall time spent translating, in microseconds.
    busy_us: u64,
}

/// What a run translated at run time - for the report a host prints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeStats {
    pub entries: usize,
    /// Of `entries`, how many the stretch sweep guessed.
    pub swept: usize,
    pub modules: usize,
    pub busy_us: u64,
}

/// Where a target's translation lives: call `slot`, once side module `side` (whose bytes are
/// `wasm`, filling slots up to `slot_end`) is installed in the calling instance.
pub struct Resolved {
    pub slot: u32,
    pub side: usize,
    pub wasm: Arc<[u8]>,
    pub slot_end: u32,
}

impl RuntimeCode {
    /// The record for a run of the main module made from `program` under `settings` (taken on
    /// the thread that made it), whose functions are `main_addrs` (ascending) and whose entry
    /// table is at layout offset `runtime_off`.
    pub fn new(
        program: transpiler::OwnedProgram,
        main_addrs: Vec<u32>,
        settings: transpiler::EmitSettings,
        runtime_off: u64,
    ) -> RuntimeCode {
        let table_off = runtime_off + u64::from(program.host_off);
        let state = State { next_slot: main_addrs.len() as u32, ..State::default() };
        RuntimeCode { program, main_addrs, settings, table_off, state: Mutex::new(state) }
    }

    /// Linear-memory byte offset of the runtime entry table.
    pub fn table_off(&self) -> u64 {
        self.table_off
    }

    /// What has been translated so far.
    pub fn stats(&self) -> RuntimeStats {
        let st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        RuntimeStats { entries: st.entries.len(), swept: st.swept, modules: st.sides.len(), busy_us: st.busy_us }
    }

    /// The table slot and side module of `target` (Thumb bit = the mode to lift it in, `from` the
    /// caller, for messages), translating it if nobody has. `table` is the runtime entry table in
    /// this process's address space, where a new entry is recorded.
    ///
    /// # Safety
    /// `table` must point at the whole, naturally aligned runtime entry table of this run's
    /// memory, which every instance reads.
    pub unsafe fn resolve(&self, target: u32, from: u32, table: *mut u8) -> Result<Resolved, String> {
        let key = target & !1;
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(&(slot, side)) = st.entries.get(&key) {
            let s = &st.sides[side];
            return Ok(Resolved { slot, side, wasm: s.wasm.clone(), slot_end: s.slot_end });
        }
        let code_end = self.program.base.wrapping_add(self.program.code.len() as u32);
        if key < self.program.base || key >= code_end {
            return Err(format!(
                "indirect dispatch to unknown target {target:#010x} from f_{from:x}: not in the image \
                 ({:#010x}..{code_end:#010x})",
                self.program.base
            ));
        }
        if st.sides.len() >= RUNTIME_MODULE_LIMIT {
            return Err(format!(
                "RUNTIME MODULE LIMIT: {target:#010x} (from f_{from:x}) would be side module {} of at most                  {RUNTIME_MODULE_LIMIT} ({} entries so far, {} of them swept) - this title keeps reaching code                  discovery did not find; find out why before raising the limit",
                st.sides.len() + 1,
                st.entries.len(),
                st.swept
            ));
        }
        // `perf::now_ms`, not `Instant`: a browser worker has no `Instant` (it panics there).
        let started = crate::perf::now_ms().unwrap_or(0.0);
        self.settings.apply();
        let slot_base = st.next_slot;
        let program = self.program.as_program();
        let sweep = SWEEP_BUDGET.saturating_sub(st.swept);
        let side =
            transpiler::transpile_side(&program, &self.main_addrs, &mut st.memo, &[(key, target & 1 != 0)], slot_base, sweep)
                .map_err(|e| format!("runtime code at {target:#010x} (from f_{from:x}) does not translate: {e:?}"))?;
        let fresh = side.funcs.iter().filter(|(a, _)| !st.entries.contains_key(a)).count();
        if st.entries.len() + fresh > transpiler::RUNTIME_ENTRY_LIMIT as usize {
            return Err(format!(
                "RUNTIME CODE LIMIT: {target:#010x} (from f_{from:x}) would be runtime entry {} of at most {} \
                 ({} side modules so far) - this title reaches far more code than discovery found; \
                 find out why before raising the limit",
                st.entries.len() + fresh,
                transpiler::RUNTIME_ENTRY_LIMIT,
                st.sides.len()
            ));
        }
        let index = st.sides.len();
        let slot_end = slot_base + side.funcs.len() as u32;
        for (i, &(addr, _)) in side.funcs.iter().enumerate() {
            // A function another side module already holds keeps that slot: one address, one
            // translation in the table, whichever instance asks.
            if st.entries.contains_key(&addr) {
                continue;
            }
            let slot = slot_base + i as u32;
            // SAFETY: the caller's contract - `table` is the whole entry table.
            unsafe { insert(table, addr, slot) };
            st.entries.insert(addr, (slot, index));
            st.memo.record(addr, side.ends[i]);
            if side.swept[i] {
                st.swept += 1;
            }
        }
        let wasm: Arc<[u8]> = side.wasm.into();
        st.sides.push(Side { wasm: wasm.clone(), slot_end });
        st.next_slot = slot_end;
        let us = ((crate::perf::now_ms().unwrap_or(0.0) - started) * 1000.0).max(0.0) as u64;
        st.busy_us += us;
        let n = st.sides.len();
        if n.is_power_of_two() || crate::knobs::var("VITASLOP_LOG").is_ok() {
            tracing::info!(
                target: "vitaslop::status",
                "runtime code: {target:#010x} ({}) from f_{from:x}: {} function(s) ({} swept) in {us} us; \
                 {} entries ({} swept) in {n} side module(s), {} us in all",
                if target & 1 != 0 { "Thumb" } else { "ARM" },
                side.funcs.len(),
                side.swept.iter().filter(|&&g| g).count(),
                st.entries.len(),
                st.swept,
                st.busy_us
            );
        }
        let (slot, _) = st.entries[&key];
        Ok(Resolved { slot, side: index, wasm, slot_end })
    }
}

/// Record `key -> slot` in the entry table at `table`: value first and key last, both atomic, so
/// a dispatcher on another thread sees either no key or a whole pair.
///
/// # Safety
/// `table` must point at a whole, naturally aligned runtime entry table.
unsafe fn insert(table: *mut u8, key: u32, slot: u32) {
    let mut h = transpiler::runtime_entry_hash(key);
    loop {
        // SAFETY: `h` < RUNTIME_ENTRY_PAIRS, so both words are inside the table.
        let (k, v) = unsafe {
            let pair = table.add(h as usize * 8);
            (AtomicU32::from_ptr(pair.cast()), AtomicU32::from_ptr(pair.add(4).cast()))
        };
        match k.load(Ordering::Acquire) {
            0 => {
                v.store(slot, Ordering::Relaxed);
                k.store(key, Ordering::Release);
                return;
            }
            existing if existing == key => return,
            _ => h = (h + 1) & (transpiler::RUNTIME_ENTRY_PAIRS - 1),
        }
    }
}
