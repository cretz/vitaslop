//! WASM backend: lower each [`ir::Func`] to one wasm function and assemble the
//! module.
//!
//! # Control flow
//! A function's basic blocks become a **dispatch loop**: a `loop` wrapping a
//! `br_table` on a `$bb` (current-block) local, with each block's code emitted in
//! ascending address order. Straight-line fall-through between adjacent blocks
//! needs no branch (control flows through the block boundary); only real jumps
//! and loop back-edges pay a `br` back to the dispatch. Direct calls are wasm
//! `call`s, returns are wasm `return`s - so the wasm call stack mirrors the guest
//! call stack and stays out of the dispatch machinery. This keeps hot loops
//! entirely in-function (no host round-trips) while handling arbitrary intra-
//! function control flow correctly. A relooper that recovers structured loops/
//! ifs for even better codegen can replace the dispatch loop later without
//! touching lowering.
//!
//! # Memory
//! Guest addresses are rebased: linear offset = guest address - `base` (see
//! [`crate::abi`]). Every load/store subtracts `base` before touching memory.
//!
//! # Flags
//! N,Z,C,V are computed eagerly into their globals by the flag statements, using
//! an i64 widening for an always-correct unsigned carry. Lazy flags (compute a
//! condition only where consumed) are a later optimization; the IR seam
//! ([`ir::Stmt::FlagsAdd`]/[`ir::Stmt::FlagsLogic`]) already isolates the choice.

use std::borrow::Cow;
use std::collections::BTreeMap;

use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, DataSection, ElementSection, Elements, ExportKind,
    ExportSection, Function, FunctionSection, GlobalSection, GlobalType, ImportSection,
    Instruction as W, MemArg, MemorySection, MemoryType, Module, NameMap, NameSection, RefType,
    TableSection, TableType, TypeSection, ValType,
};

use crate::abi;
use crate::ir::{BinOp, Block, ConditionCode, Func, MemSize, Stmt, Term, Value};

/// `VITASLOP_ARM_AT_FRAME=<n>` - hold every trapping diagnostic DISARMED until the
/// run reaches display frame `n`.
///
/// Almost every diagnostic here fires on its FIRST hit, and the first hit of
/// anything interesting is during boot - which makes them useless for a question
/// about frame 2000 of a live game. Arming them by frame fixes that generally, in
/// one place, instead of each knob growing its own "skip the first N" counter (and
/// instead of bisecting such a counter by hand, which is a long, dull dead end).
///
/// The gate is a single 4-byte word in LINEAR MEMORY rather than a wasm global,
/// because the scheduler runs each guest thread as its own instance: a global is
/// per-instance, so arming it would mean reaching into every live thread's store,
/// while linear memory is shared by all of them and the host can write it at any
/// moment. The word sits on its own page above the dispatch table (see
/// [`emit_module`]), so it can never collide with guest memory.
///
/// Zero cost when unset: no gate is emitted at all and the module is byte-identical.
pub fn arm_at_frame() -> Option<u64> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<u64>> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_ARM_AT_FRAME").ok().and_then(|s| s.trim().parse().ok())
    })
}

thread_local! {
    /// Linear-memory byte offset of the "diagnostics armed" word for the module
    /// being emitted on this thread, or 0 when this build has no frame gate. A
    /// thread-local (not a global) because emission is single-threaded per module
    /// while a test binary may emit several modules at once.
    static ARM_WORD_OFF: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// AND the "diagnostics are armed" condition into the value already on the stack.
/// A no-op when this build has no frame gate, so an ungated diagnostic keeps its
/// exact previous shape.
fn and_armed(f: &mut Function) {
    let off = ARM_WORD_OFF.with(|c| c.get());
    if off == 0 {
        return;
    }
    f.instruction(&W::I32Const(0));
    f.instruction(&W::I32Load(MemArg { offset: off, align: 2, memory_index: 0 }));
    f.instruction(&W::I32And);
}

/// Diagnostic store watchpoint. When `VITASLOP_WATCH_STORE=<hex guest addr>` is set
/// at transpile time, every word store to that exact guest address is preceded by an
/// `unreachable`, so the first writer traps with a full wasm backtrace (and the
/// stored value is visible in the register dump). Used to catch which code path
/// writes - or fails to write - a specific object field (e.g. a NULL vtable slot).
/// Cached once; parsing happens at emit time only, never at guest runtime.
fn watch_store_addr() -> Option<u32> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<u32>> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_WATCH_STORE").ok().and_then(|s| {
            u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
        })
    })
}

/// When set alongside `VITASLOP_WATCH_STORE`, only a store of a non-zero value to
/// the watched address traps (so a memset-to-zero of a fresh allocation is skipped
/// and the trap pinpoints the code that writes the real value, e.g. a vtable set).
fn watch_store_nonzero() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| std::env::var("VITASLOP_WATCH_STORE_NZ").is_ok())
}

/// Diagnostic read watchpoint. When `VITASLOP_WATCH_READ=<hex guest addr>` is set at
/// transpile time, every load whose address equals that exact guest address AND whose
/// loaded value is non-zero is preceded by an `unreachable`, so the first non-zero
/// reader traps with a full backtrace. Paired with `VITASLOP_TRACK_PC` this pinpoints
/// the exact guest instruction that consumes a field - the mirror of the store
/// watchpoint, used to find who reads a value (e.g. which code consumes an input-edge
/// field that no static reference reveals because the object is heap-allocated). The
/// non-zero filter skips the idle reads (a per-frame poll that sees 0) so the trap
/// lands on the consumer that actually acts on a set value. Zero cost when unset.
fn watch_read_addr() -> Option<u32> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<u32>> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_WATCH_READ").ok().and_then(|s| {
            u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
        })
    })
}

/// When set alongside `VITASLOP_WATCH_READ`, only a load of a non-zero value from the
/// watched address traps (skip idle polls that see 0). Default (unset) traps on the
/// first read of any value, which confirms whether the consumer runs at all.
fn watch_read_nonzero() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| std::env::var("VITASLOP_WATCH_READ_NZ").is_ok())
}

/// When `VITASLOP_WASM_NAMES` is set, emit a wasm `name` custom section labelling
/// every function with its guest address (`g_<addr>`), so a wasmtime trap backtrace
/// prints the guest function directly instead of a bare module index that has to be
/// hand-mapped. Purely a debugging aid: the name section is never touched during
/// execution (zero runtime cost), but it grows the module by ~1.5% and adds a small
/// per-instantiation parse, so it stays off for shipped builds. Cached once; the
/// module is byte-identical to a normal build when unset.
fn emit_wasm_names() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| std::env::var("VITASLOP_WASM_NAMES").is_ok())
}

/// When `VITASLOP_TRACK_PC` is set, each basic block writes its own guest start
/// address into [`GUEST_PC_GLOBAL`] before executing, so a trap's register dump can
/// report exactly which guest instruction faulted (block granularity - the block is
/// short enough that the fault site is unambiguous once disassembled). A debugging
/// aid with a small runtime cost (one `global.set` per block), so it stays off for
/// shipped builds; the module is byte-identical to a normal build when unset.
fn track_pc() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| std::env::var("VITASLOP_TRACK_PC").is_ok())
}

/// Function indices of the host imports (imports occupy the low function-index
/// space, in declaration order).
const SVC_FUNC: u32 = 0;
const IMPORT_FUNC: u32 = 1;
/// `env.dispatch_miss(target, caller)`: the indirect-call dispatcher calls this
/// when a runtime function-pointer matches no translated function, so an unmapped
/// target becomes a reported, debuggable trap instead of an opaque `unreachable`.
const DISPATCH_MISS_FUNC: u32 = 2;
/// Number of imported functions before the guest functions. Re-exported through
/// [`abi::IMPORT_FUNC_COUNT`] so hosts mapping a trap backtrace stay in lockstep.
pub(crate) const IMPORT_FUNCS: u32 = abi::IMPORT_FUNC_COUNT;

/// WASM global index of the diagnostic store-watchpoint "armed" latch, appended
/// after the whole register file (see [`emit_module`]).
const WATCH_ARMED_GLOBAL: u32 = abi::TOTAL_GLOBAL_COUNT;

/// WASM global index of the diagnostic guest-PC tracker, appended just after the
/// watchpoint latch (see [`track_pc`] and [`emit_module`]). Holds the address of
/// the basic block currently executing; on a trap the host reads it back to name
/// the faulting guest instruction.
const GUEST_PC_GLOBAL: u32 = abi::TOTAL_GLOBAL_COUNT + 1;

/// WASM global index of the read-watchpoint match counter, appended after the guest-PC
/// tracker. Counts matching loads so `VITASLOP_WATCH_READ_SKIP` can skip the first N
/// (init/idle) hits and trap on a later one - the way to see a consumer that runs every
/// frame without the trap always landing on the first (startup) read.
const WATCH_READ_COUNT_GLOBAL: u32 = abi::TOTAL_GLOBAL_COUNT + 2;

/// WASM global index of the store-watchpoint match counter (appended after `TP_GLOBAL`).
/// Lets `VITASLOP_WATCH_STORE_SKIP` skip the first N matching stores and trap on a later
/// one - e.g. to catch a map's node-count *decrement* past its earlier increments.
const WATCH_STORE_COUNT_GLOBAL: u32 = abi::TOTAL_GLOBAL_COUNT + 4;

/// Number of matching store-watchpoint hits to skip before trapping (`VITASLOP_WATCH_
/// STORE_SKIP`, default 0 = trap on the first).
fn watch_store_skip() -> u32 {
    use std::sync::OnceLock;
    static CELL: OnceLock<u32> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_WATCH_STORE_SKIP")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// Number of matching read-watchpoint hits to skip before trapping (`VITASLOP_WATCH_
/// READ_SKIP`, default 0 = trap on the first). Lets the trap land past startup reads.
fn watch_read_skip() -> u32 {
    use std::sync::OnceLock;
    static CELL: OnceLock<u32> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_WATCH_READ_SKIP")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// Optional guest-PC EXCLUDE window for the read watchpoint (`VITASLOP_WATCH_READ_
/// PC_EXCL=<lo>-<hi>` hex): when set, a match only traps if the currently-executing
/// block's guest address (from `VITASLOP_TRACK_PC`) is OUTSIDE `[lo, hi)`. This peels
/// one consumer of a hot field apart from others - e.g. a menu/dialog read of an input
/// field that a per-frame input *system* (a known address window) also reads, which
/// would otherwise always trap first. Requires `VITASLOP_TRACK_PC`.
fn watch_read_pc_exclude() -> Option<(u32, u32)> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<(u32, u32)>> = OnceLock::new();
    *CELL.get_or_init(|| {
        let s = std::env::var("VITASLOP_WATCH_READ_PC_EXCL").ok()?;
        let (lo, hi) = s.trim().split_once('-')?;
        let lo = u32::from_str_radix(lo.trim().trim_start_matches("0x"), 16).ok()?;
        let hi = u32::from_str_radix(hi.trim().trim_start_matches("0x"), 16).ok()?;
        Some((lo, hi))
    })
}

/// Diagnostic callee-saved-register guard. When `VITASLOP_GUARD_REG=<n>` (a single
/// ARM register number, e.g. `7`) is set at transpile time, the value of that register
/// is snapshotted into a scratch local immediately before every direct `Call` and
/// indirect `blx`/`bx`, and compared against the register right after the call returns;
/// a mismatch traps (`unreachable`) so the first callee that fails to preserve a
/// callee-saved register (a mislifted push/pop or LDM/STM, or a wrong indirect
/// dispatch) is pinpointed. Pair with `VITASLOP_TRACK_PC` + `VITASLOP_WASM_NAMES`: the
/// trap's `guest_block` names the call site and the backtrace names the caller, so the
/// call target in that block is the culprit. Zero cost and byte-identical when unset.
fn guard_reg() -> Option<u8> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<u8>> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_GUARD_REG")
            .ok()
            .and_then(|s| s.trim().parse::<u8>().ok())
            .filter(|&r| (r as usize) < 15)
    })
}

/// Diagnostic guest-function entry tracer. When `VITASLOP_TRACE_FUNCS=<hex>[,<hex>...]`
/// is set at transpile time, each listed guest function emits `svc #<its own address>`
/// as its first instruction, so the host `svc` handler logs the entry (address + the
/// incoming argument registers) before the body runs. Guest `svc` immediates are 24-bit,
/// so an immediate with the top bit set (a guest address, always >= 0x81000000) is
/// unambiguously a trace marker, never a real syscall. Zero cost and byte-identical when
/// unset. Pairs with `VITASLOP_WASM_NAMES` to name the call chain.
fn trace_funcs() -> &'static std::collections::BTreeSet<u32> {
    use std::sync::OnceLock;
    static CELL: OnceLock<std::collections::BTreeSet<u32>> = OnceLock::new();
    CELL.get_or_init(|| {
        std::env::var("VITASLOP_TRACE_FUNCS")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| {
                        u32::from_str_radix(t.trim().trim_start_matches("0x"), 16).ok()
                    })
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// Diagnostic indirect-call tracer. `VITASLOP_TRACE_INDIRECT=<lo>-<hi>` (hex, inclusive)
/// makes the module dispatcher log every indirect (`blx`/`bx`) call whose resolved target
/// lands in `[lo, hi]`, by routing the target through the `svc` handler (which prints the
/// target address plus the live argument registers r0..r3 and lr = the caller's return).
/// This reveals the runtime vtable dispatch graph that static call-graph analysis cannot
/// follow. Zero cost / byte-identical when unset.
fn trace_indirect_range() -> Option<(u32, u32)> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<(u32, u32)>> = OnceLock::new();
    *CELL.get_or_init(|| {
        let s = std::env::var("VITASLOP_TRACE_INDIRECT").ok()?;
        let (lo, hi) = s.split_once('-')?;
        let lo = u32::from_str_radix(lo.trim().trim_start_matches("0x"), 16).ok()?;
        let hi = u32::from_str_radix(hi.trim().trim_start_matches("0x"), 16).ok()?;
        Some((lo, hi))
    })
}

/// Diagnostic per-basic-block execution tracer. `VITASLOP_TRACE_BLOCKS=<lo>-<hi>` (hex,
/// inclusive) makes every basic block whose guest start address lands in `[lo, hi]` emit
/// `svc #<block address>` as its first instruction, so the host `svc` handler logs the
/// block entry (address + live registers r0..r3, r8, lr) in execution order. This is the
/// ground-truth "which path did the function actually take" trace: run it over a single
/// function's address span and compare the observed block sequence to the static CFG to
/// find where a mis-lifted branch/computation steers control the wrong way. Zero cost /
/// byte-identical when unset.
fn trace_blocks_range() -> Option<(u32, u32)> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<(u32, u32)>> = OnceLock::new();
    *CELL.get_or_init(|| {
        let s = std::env::var("VITASLOP_TRACE_BLOCKS").ok()?;
        let (lo, hi) = s.split_once('-')?;
        let lo = u32::from_str_radix(lo.trim().trim_start_matches("0x"), 16).ok()?;
        let hi = u32::from_str_radix(hi.trim().trim_start_matches("0x"), 16).ok()?;
        Some((lo, hi))
    })
}

/// Diagnostic forced return. `VITASLOP_FORCE_RET=<hex addr>:<dec value>[,...]` makes each
/// listed guest function immediately `return value` (value left in the r0 global) as its
/// first action, skipping its body. This tests downstream causality: force a readiness /
/// predicate function to a fixed result and observe how far the boot then progresses,
/// without needing to find and fix its real producer first. Zero cost / byte-identical when
/// unset. Pairs with `VITASLOP_TRACE_FUNCS` to confirm the forced function actually runs.
fn force_ret() -> &'static std::collections::BTreeMap<u32, u32> {
    use std::sync::OnceLock;
    static CELL: OnceLock<std::collections::BTreeMap<u32, u32>> = OnceLock::new();
    CELL.get_or_init(|| {
        std::env::var("VITASLOP_FORCE_RET")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|t| {
                        let (a, v) = t.trim().split_once(':')?;
                        let addr = u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?;
                        let val: u32 = v.trim().parse().ok()?;
                        Some((addr, val))
                    })
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// When `VITASLOP_TRAP_HALT` is set, a `Term::Halt` (a block that ran off the end of decoded
/// code, i.e. an undecoded-op cutoff) traps instead of returning, so the first such cutoff
/// actually reached at runtime faults loudly instead of silently returning an incomplete
/// function. A debugging aid to drive the decode-gap grind along the real boot path.
fn trap_halt() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| std::env::var("VITASLOP_TRAP_HALT").is_ok())
}

/// Scratch local holding the pre-call snapshot of the guarded register (see
/// [`guard_reg`]). Only declared when the guard is enabled, so an unguarded build is
/// byte-identical; it follows the v128 scratch locals.
const L_GUARD: u32 = L_V128C + 1;

/// Store-watchpoint mode, from `VITASLOP_WATCH_STORE_MODE` (default `any`):
/// `any` traps on any store to the address, `nz` only on a non-zero store, `arm`
/// arms on a non-zero store and traps on a later zero store (catches a field that
/// is set correctly then wrongly cleared).
fn watch_store_arm_mode() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| std::env::var("VITASLOP_WATCH_STORE_ARM").is_ok())
}

// Scratch locals used by flag computation. Local 0 is `$bb`.
const L_BB: u32 = 0;
const L_T0: u32 = 1;
const L_T1: u32 = 2;
const L_T2: u32 = 3;
/// Fourth i32 scratch: the carry-in of add/sub-family flag computation (an
/// immediate, or the C flag for adc/sbc).
const L_T3: u32 = 4;
/// The four APSR `GE` bits (bits [3:0]) that a byte-wise parallel add/sub
/// (`uadd8`) deposits and a later `sel` reads. A dedicated local, not one of the
/// `L_T*` scratches, because `sel` can be several instructions after the
/// `uadd8` and the scratches are clobbered in between (the byte-search loop in
/// `strlen` interleaves loads).
const L_GE: u32 = 5;
const L_I32_COUNT: u32 = 6;
/// i64 scratch, used to split/merge a double register across its two aliased
/// single-register halves. Index follows the i32 locals.
const L_D64: u32 = L_I32_COUNT;
/// Two `v128` scratch locals, used by NEON emission to hold a quad register for
/// read-modify-write (writing one D lane of an upper-bank quad) and to stage the
/// two operands of the ops that read each twice (`vabd`/`vabdl`). Follow the i64
/// scratch.
const L_V128A: u32 = L_D64 + 1;
const L_V128B: u32 = L_D64 + 2;
/// A third `v128` scratch, used by the two-register permutes (`vtrn`/`vzip`/`vuzp`) to stash the
/// first result register while the second is computed and written (a plain low-bank `neon_set`
/// itself reuses `L_V128A`, so the staged result must live elsewhere).
const L_V128C: u32 = L_D64 + 3;

/// The exported linear-memory layout, returned by [`emit_module`] so the host can
/// provision a shared memory that exactly matches what the module declares.
pub struct EmitOutput {
    /// The emitted WASM module bytes.
    pub wasm: Vec<u8>,
    /// Total linear-memory pages the module declares: the guest region plus the
    /// dispatch address table appended above it (see [`emit_module`]). A host that
    /// *imports* the memory (the preemptive scheduler) must create it with exactly
    /// this many pages; a host that lets the module define its own memory ignores
    /// this (the module already carries the right size).
    pub mem_pages: u32,
    /// Linear-memory byte offset of the "diagnostics armed" word, when this build
    /// was emitted with `VITASLOP_ARM_AT_FRAME` (see [`arm_at_frame`]). The host
    /// writes 1 there once the run reaches the armed frame; until then every
    /// trapping diagnostic is inert. `None` in an ordinary build.
    pub arm_word_off: Option<u64>,
}

/// Assemble the full wasm module for `funcs`. `func_index` maps a guest function
/// address to its wasm function index. `mem_bytes` sizes the guest linear memory;
/// `base` is the guest image base for the address rebase.
///
/// # Indirect-call dispatch
/// Guest `blx`/`bx reg` targets a runtime function-pointer, which the dispatcher
/// ([`emit_dispatch`]) resolves to a translated function. Resolution is O(log n):
/// the ascending guest addresses of all `funcs` are emitted as a data segment just
/// above the guest region, the dispatcher binary-searches it, and a `call_indirect`
/// through a dense funcref table (`table[i]` = the i-th function in ascending order)
/// jumps to the match. The search array is small (4 bytes per function) and stays
/// hot in cache; the funcref table is the only per-instance cost (one funcref per
/// function), which matters because the preemptive scheduler instantiates the module
/// once per guest thread.
pub fn emit_module(
    funcs: &[Func],
    func_index: &BTreeMap<u32, u32>,
    base: u32,
    mem_bytes: u32,
    inline_imports: &[crate::InlineImport],
    import_memory: bool,
) -> EmitOutput {
    let inline = InlineImports::new(inline_imports, mem_bytes);
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], []); // svc / import: (i32) -> ()
    let host_ty = 0;
    types.ty().function([], []); // guest function: () -> ()
    let func_ty = 1;
    types.ty().function([ValType::I32, ValType::I32], []); // dispatch / dispatch_miss
    let dispatch_ty = 2;

    // The guest region occupies whole pages from offset 0; the dispatch address
    // table is appended immediately above it (page-aligned), so the module declares
    // more pages than the guest itself uses. `addr_table_off` is the byte offset of
    // that table, and `n` its entry count (one 4-byte ascending guest address per
    // translated function).
    let n = funcs.len() as u32;
    let guest_pages = (mem_bytes as u64).div_ceil(abi::PAGE_SIZE as u64).max(1);
    let addr_table_off = guest_pages * abi::PAGE_SIZE as u64;
    let addr_table_bytes = n as u64 * 4;
    let addr_table_pages = addr_table_bytes.div_ceil(abi::PAGE_SIZE as u64);
    // One more page above the dispatch table holds the "diagnostics armed" word
    // (see `arm_at_frame`), and only when that knob is set - an ordinary build's
    // memory layout is unchanged. Its own page, so no guest allocation or dispatch
    // entry can ever share a cache line with it.
    let arm_word_off =
        arm_at_frame().map(|_| (guest_pages + addr_table_pages) * abi::PAGE_SIZE as u64);
    ARM_WORD_OFF.with(|c| c.set(arm_word_off.unwrap_or(0)));
    let total_pages = guest_pages + addr_table_pages + u64::from(arm_word_off.is_some());

    // Preemptive multithreading (the native `ThreadedScheduler`) runs each guest
    // thread as its own instance so their register globals stay independent, but
    // they must share one address space. wasm gives us exactly one tool for that:
    // a `shared` linear memory imported into every instance. When `import_memory`
    // is set the module imports `env.memory` (shared, fixed size) instead of
    // defining its own; single-instance hosts leave it off and get the original
    // self-contained module unchanged. The memory index is 0 either way (it is the
    // only memory), so every load/store and the `memory` export are identical.
    let mut imports = ImportSection::new();
    imports.import(abi::IMPORT_MODULE, abi::SVC_NAME, wasm_encoder::EntityType::Function(host_ty));
    imports.import(abi::IMPORT_MODULE, abi::IMPORT_NAME, wasm_encoder::EntityType::Function(host_ty));
    // `env.dispatch_miss(target, caller)`: reported when an indirect call resolves to
    // no known function (see `emit_dispatch`). Declared as a function import, so it
    // takes function index `DISPATCH_MISS_FUNC` before any guest function.
    imports.import(
        abi::IMPORT_MODULE,
        abi::DISPATCH_MISS_NAME,
        wasm_encoder::EntityType::Function(dispatch_ty),
    );
    if import_memory {
        // A shared memory must declare a maximum; the guest never grows memory, so
        // pin max == min at the provisioned size (guest region + dispatch table).
        imports.import(
            abi::IMPORT_MODULE,
            abi::MEMORY_EXPORT,
            wasm_encoder::EntityType::Memory(MemoryType {
                minimum: total_pages,
                maximum: Some(total_pages),
                memory64: false,
                shared: true,
                page_size_log2: None,
            }),
        );
    }

    let mut function_section = FunctionSection::new();
    for _ in funcs {
        function_section.function(func_ty);
    }
    // The indirect-call dispatcher (the last defined function): `(target, caller)`.
    // It binary-searches the address table and `call_indirect`s the match, or reports
    // an unmapped target to `dispatch_miss` - see `emit_dispatch`.
    function_section.function(dispatch_ty);

    // The dense funcref table the dispatcher's `call_indirect` jumps through:
    // `table[i]` is the i-th translated function in ascending-address order, so the
    // index the binary search returns indexes it directly. Sized exactly to the
    // function count (this is the only per-instance dispatch cost).
    let mut tables = TableSection::new();
    tables.table(TableType {
        element_type: RefType::FUNCREF,
        table64: false,
        minimum: n as u64,
        maximum: Some(n as u64),
        shared: false,
    });

    let mut mems = MemorySection::new();
    if !import_memory {
        mems.memory(MemoryType {
            minimum: total_pages,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
    }

    // Globals, in ABI order: 16 registers + 4 integer flags (i32), then the VFP/NEON
    // register file - 32 single-precision S registers (raw-bit i32) and 8 upper
    // quad registers Q8..Q15 (v128) - then 4 FP condition flags (i32).
    let mut globals = GlobalSection::new();
    let i32_global =
        GlobalType { val_type: ValType::I32, mutable: true, shared: false };
    let v128_global =
        GlobalType { val_type: ValType::V128, mutable: true, shared: false };
    for _ in 0..abi::GLOBAL_COUNT {
        globals.global(i32_global, &ConstExpr::i32_const(0));
    }
    for _ in 0..abi::VFP_S_COUNT {
        globals.global(i32_global, &ConstExpr::i32_const(0));
    }
    for _ in 0..abi::VFP_Q_HI_COUNT {
        globals.global(v128_global, &ConstExpr::v128_const(0));
    }
    for _ in 0..abi::FP_FLAG_COUNT {
        globals.global(i32_global, &ConstExpr::i32_const(0));
    }
    // Three extra i32 globals, always present so their indices are stable and unused
    // (zero, no runtime cost) in a normal build: the store watchpoint's "armed" latch
    // (see `watch_store_addr`), the guest-PC tracker (see `track_pc`), and the
    // read-watchpoint match counter (see `watch_read_skip`).
    globals.global(i32_global, &ConstExpr::i32_const(0)); // WATCH_ARMED_GLOBAL
    globals.global(i32_global, &ConstExpr::i32_const(0)); // GUEST_PC_GLOBAL
    globals.global(i32_global, &ConstExpr::i32_const(0)); // WATCH_READ_COUNT_GLOBAL
    // The per-thread pointer (TPIDRURO / TLS base). Per-instance, set by the host at
    // thread instantiation; read by `MRC p15,0,Rt,c13,c0,3` (see `abi::TP_GLOBAL`).
    globals.global(i32_global, &ConstExpr::i32_const(0)); // TP_GLOBAL
    globals.global(i32_global, &ConstExpr::i32_const(0)); // WATCH_STORE_COUNT_GLOBAL

    let mut exports = ExportSection::new();
    exports.export(abi::MEMORY_EXPORT, ExportKind::Memory, 0);
    for i in 0..abi::REG_COUNT {
        exports.export(&abi::reg_export(i), ExportKind::Global, abi::reg_global(i));
    }
    for f in [abi::Flag::N, abi::Flag::Z, abi::Flag::C, abi::Flag::V] {
        exports.export(abi::flag_export(f), ExportKind::Global, abi::flag_global(f));
    }
    // Export the VFP/NEON register file so the host can seed/read FP state (tests,
    // GXM capture). S0..S31 (i32), Q8..Q15 (v128), and the FP flags. The host only
    // marshals the low-bank S registers; the v128 quads are exported for
    // completeness (JS cannot read them, but native hosts and tools can).
    for n in 0..abi::VFP_S_COUNT as u8 {
        exports.export(&abi::vfp_s_export(n), ExportKind::Global, abi::vfp_s_global(n));
    }
    for q in abi::VFP_Q_HI_FIRST as u8..(abi::VFP_Q_HI_FIRST + abi::VFP_Q_HI_COUNT) as u8 {
        exports.export(&abi::vfp_qhi_export(q), ExportKind::Global, abi::vfp_qhi_global(q));
    }
    for f in [abi::Flag::N, abi::Flag::Z, abi::Flag::C, abi::Flag::V] {
        exports.export(abi::fp_flag_export(f), ExportKind::Global, abi::fp_flag_global(f));
    }
    // The guest-PC tracker, so the host can read the faulting block address on a trap
    // (zero unless `VITASLOP_TRACK_PC` is set at emit time; see `track_pc`).
    exports.export(abi::GUEST_PC_EXPORT, ExportKind::Global, GUEST_PC_GLOBAL);
    // The per-thread pointer, so the host seeds each thread's TLS base at instantiation.
    exports.export(abi::TP_EXPORT, ExportKind::Global, abi::TP_GLOBAL);

    // Populate the dense funcref table: table[i] = the i-th translated function
    // (wasm index IMPORT_FUNCS + i), matching the ascending-address order of `funcs`
    // and of the address table the dispatcher searches. One contiguous active
    // segment; skipped when there are no functions (an empty table needs no init).
    let mut elements = ElementSection::new();
    if n > 0 {
        let entries: Vec<u32> = (0..n).map(|i| IMPORT_FUNCS + i).collect();
        elements.active(Some(0), &ConstExpr::i32_const(0), Elements::Functions(Cow::Owned(entries)));
    }

    let mut code = CodeSection::new();
    for (i, func) in funcs.iter().enumerate() {
        let idx = IMPORT_FUNCS + i as u32;
        exports.export(&abi::func_export(func.addr), ExportKind::Func, idx);
        code.function(&emit_func(func, func_index, base, &inline));
    }
    code.function(&emit_dispatch(funcs, addr_table_off));

    // The dispatcher's search array: each function's guest address as a little-endian
    // u32, in ascending order (so a binary search finds a target and its dense index
    // in one shot). Emitted as an active data segment just above the guest region, so
    // it initializes at instantiation with no host cooperation.
    let mut data = DataSection::new();
    if n > 0 {
        let mut bytes = Vec::with_capacity(funcs.len() * 4);
        for func in funcs {
            bytes.extend_from_slice(&func.addr.to_le_bytes());
        }
        data.active(0, &ConstExpr::i32_const(addr_table_off as i32), bytes);
    }

    let mut module = Module::new();
    module
        .section(&types)
        .section(&imports)
        .section(&function_section)
        .section(&tables)
        .section(&mems)
        .section(&globals)
        .section(&exports)
        .section(&elements)
        .section(&code)
        .section(&data);

    // Optional debug `name` section (see `emit_wasm_names`): map each wasm function
    // index to a human-readable name so trap backtraces symbolize directly. Imports
    // occupy indices 0..IMPORT_FUNCS, guest functions IMPORT_FUNCS.., and the
    // dispatcher is last. Emitted only when opted in; the module is otherwise
    // byte-identical to a shipped build.
    if emit_wasm_names() {
        let mut names = NameMap::new();
        names.append(SVC_FUNC, "svc");
        names.append(IMPORT_FUNC, "host_import");
        names.append(DISPATCH_MISS_FUNC, "dispatch_miss");
        for (i, func) in funcs.iter().enumerate() {
            names.append(IMPORT_FUNCS + i as u32, &format!("g_{:08x}", func.addr));
        }
        names.append(IMPORT_FUNCS + n, "dispatch");
        let mut name_section = NameSection::new();
        name_section.functions(&names);
        module.section(&name_section);
    }
    EmitOutput { wasm: module.finish(), mem_pages: total_pages as u32, arm_word_off }
}

/// Emit the indirect-call dispatcher: `(target: i32, caller: i32) -> ()`. It masks
/// the Thumb bit off the runtime function-pointer and binary-searches the ascending
/// address table (a data segment at `addr_table_off`; see [`emit_module`]) for the
/// target. On a hit it `call_indirect`s the dense funcref table at the found index
/// (`table[i]` is the i-th function in the same ascending order) and returns. On a
/// miss - a target that is no known function entry - it reports `(target, caller)` to
/// `env.dispatch_miss`, which traps with a debuggable message; the trailing
/// `unreachable` guards against the host returning instead of trapping.
///
/// Resolution is O(log n) with a search array that stays hot in cache, replacing the
/// old O(n) linear address compare. `funcs` must be in ascending-address order (it
/// is - `emit_module` receives the functions sorted), matching both the address
/// table and the funcref table.
fn emit_dispatch(funcs: &[Func], addr_table_off: u64) -> Function {
    // Locals beyond the two params: lo, hi, mid, v (the loaded table entry).
    const P_TARGET: u32 = 0;
    const P_CALLER: u32 = 1;
    const L_LO: u32 = 2;
    const L_HI: u32 = 3;
    const L_MID: u32 = 4;
    const L_V: u32 = 5;
    let mut f = Function::new([(4, ValType::I32)]);

    // target &= ~1  (clear the Thumb bit; function addresses are even).
    f.instruction(&W::LocalGet(P_TARGET));
    f.instruction(&W::I32Const(!1));
    f.instruction(&W::I32And);
    f.instruction(&W::LocalSet(P_TARGET));

    // lo = 0; hi = n.
    f.instruction(&W::I32Const(0));
    f.instruction(&W::LocalSet(L_LO));
    f.instruction(&W::I32Const(funcs.len() as i32));
    f.instruction(&W::LocalSet(L_HI));

    // block $done { loop $loop { ... } }  -- breaking to $done means "not found".
    f.instruction(&W::Block(BlockType::Empty));
    f.instruction(&W::Loop(BlockType::Empty));

    // if lo >= hi { break to $done }  (unsigned; lo/hi are small non-negative counts).
    f.instruction(&W::LocalGet(L_LO));
    f.instruction(&W::LocalGet(L_HI));
    f.instruction(&W::I32GeU);
    f.instruction(&W::BrIf(1)); // -> $done (out of the loop)

    // mid = (lo + hi) >> 1.
    f.instruction(&W::LocalGet(L_LO));
    f.instruction(&W::LocalGet(L_HI));
    f.instruction(&W::I32Add);
    f.instruction(&W::I32Const(1));
    f.instruction(&W::I32ShrU);
    f.instruction(&W::LocalSet(L_MID));

    // v = addr_table[mid]  (load u32 at addr_table_off + mid*4).
    f.instruction(&W::LocalGet(L_MID));
    f.instruction(&W::I32Const(4));
    f.instruction(&W::I32Mul);
    f.instruction(&W::I32Load(MemArg { offset: addr_table_off, align: 2, memory_index: 0 }));
    f.instruction(&W::LocalSet(L_V));

    // if v == target { [trace if in range]; call_indirect table[mid]; return }
    f.instruction(&W::LocalGet(L_V));
    f.instruction(&W::LocalGet(P_TARGET));
    f.instruction(&W::I32Eq);
    f.instruction(&W::If(BlockType::Empty));
    // Diagnostic: log resolved indirect targets in the configured range (see
    // `trace_indirect_range`). The target is passed to the `svc` handler as its
    // selector; because guest addresses have bit 31 set, the handler treats it as a
    // trace marker and logs it with the live registers (r0 = `this`, lr = caller).
    if let Some((lo, hi)) = trace_indirect_range() {
        f.instruction(&W::LocalGet(P_TARGET));
        f.instruction(&W::I32Const(lo as i32));
        f.instruction(&W::I32GeU);
        f.instruction(&W::LocalGet(P_TARGET));
        f.instruction(&W::I32Const(hi as i32));
        f.instruction(&W::I32LeU);
        f.instruction(&W::I32And);
        f.instruction(&W::If(BlockType::Empty));
        f.instruction(&W::LocalGet(P_TARGET));
        f.instruction(&W::Call(SVC_FUNC));
        f.instruction(&W::End);
    }
    f.instruction(&W::LocalGet(L_MID));
    f.instruction(&W::CallIndirect { type_index: 1 /* guest () -> () */, table_index: 0 });
    f.instruction(&W::Return);
    f.instruction(&W::End);

    // else narrow the range: if v < target { lo = mid + 1 } else { hi = mid }.
    f.instruction(&W::LocalGet(L_V));
    f.instruction(&W::LocalGet(P_TARGET));
    f.instruction(&W::I32LtU);
    f.instruction(&W::If(BlockType::Empty));
    f.instruction(&W::LocalGet(L_MID));
    f.instruction(&W::I32Const(1));
    f.instruction(&W::I32Add);
    f.instruction(&W::LocalSet(L_LO));
    f.instruction(&W::Else);
    f.instruction(&W::LocalGet(L_MID));
    f.instruction(&W::LocalSet(L_HI));
    f.instruction(&W::End);

    f.instruction(&W::Br(0)); // continue $loop
    f.instruction(&W::End); // loop
    f.instruction(&W::End); // block $done

    // Not found: report the unmapped target (with its caller) and trap.
    f.instruction(&W::LocalGet(P_TARGET));
    f.instruction(&W::LocalGet(P_CALLER));
    f.instruction(&W::Call(DISPATCH_MISS_FUNC));
    f.instruction(&W::Unreachable);
    f.instruction(&W::End); // function body
    f
}

/// Emit one guest function as a wasm function: a dispatch loop over its blocks.
fn emit_func(
    func: &Func,
    func_index: &BTreeMap<u32, u32>,
    base: u32,
    inline: &InlineImports,
) -> Function {
    // Locals: $bb + i32 scratch temps (flag computation), then one i64 scratch
    // (double-register split/merge) and one v128 scratch (NEON quad staging).
    let mut f = if guard_reg().is_some() {
        Function::new([
            (L_I32_COUNT, ValType::I32),
            (1, ValType::I64),
            (3, ValType::V128),
            (1, ValType::I32), // L_GUARD: pre-call snapshot for the CSR guard
        ])
    } else {
        Function::new([
            (L_I32_COUNT, ValType::I32),
            (1, ValType::I64),
            (3, ValType::V128),
        ])
    };

    // A stub for an un-liftable function: trap if ever executed.
    if func.stub {
        f.instruction(&W::Unreachable);
        f.instruction(&W::End);
        return f;
    }

    // Diagnostic entry tracer (opt-in): announce this function's entry to the host
    // `svc` handler, which logs the address and incoming argument registers. Emitted
    // before any block so it fires exactly once per call, on entry (see `trace_funcs`).
    if trace_funcs().contains(&func.addr) {
        f.instruction(&W::I32Const(func.addr as i32));
        f.instruction(&W::Call(SVC_FUNC));
    }

    // Diagnostic forced return (opt-in): set r0 to the configured value and return before
    // running the body, so a readiness/predicate function can be pinned to test downstream.
    if let Some(&val) = force_ret().get(&func.addr) {
        f.instruction(&W::I32Const(val as i32));
        f.instruction(&W::GlobalSet(abi::reg_global(0)));
        f.instruction(&W::Return);
    }

    let n = func.blocks.len() as u32;

    // Single-block functions need no dispatch machinery.
    if n == 1 {
        emit_block(&mut f, &func.blocks[0], func, func_index, base, inline, 0);
        f.instruction(&W::End);
        return f;
    }

    // block $exit ; loop $loop ; block $B{n-1} ... block $B0 ; br_table ...
    f.instruction(&W::Block(BlockType::Empty)); // $exit
    f.instruction(&W::Loop(BlockType::Empty)); // $loop
    for _ in 0..n {
        f.instruction(&W::Block(BlockType::Empty));
    }
    // At the innermost point, $B0 is depth 0 .. $B{n-1} is depth n-1.
    f.instruction(&W::LocalGet(L_BB));
    let targets: Vec<u32> = (0..n).collect();
    f.instruction(&W::BrTable(targets.into(), n /* default -> $loop */));
    // Close $B0 (its body follows), then $B1, ...
    for (k, block) in func.blocks.iter().enumerate() {
        f.instruction(&W::End); // closes $B{k}
        emit_block(&mut f, block, func, func_index, base, inline, n - 1 - k as u32);
    }
    f.instruction(&W::End); // loop
    f.instruction(&W::End); // $exit block
    f.instruction(&W::End); // function body
    f
}

/// Emit one basic block's statements and terminator. `loop_depth` is the wasm
/// branch depth from within this block's code to the enclosing dispatch `loop`.
fn emit_block(
    f: &mut Function,
    block: &Block,
    func: &Func,
    func_index: &BTreeMap<u32, u32>,
    base: u32,
    inline: &InlineImports,
    loop_depth: u32,
) {
    // Diagnostic guest-PC tracking (opt-in): record this block's start address before
    // running it, so a trap's register dump can name the faulting instruction.
    if track_pc() {
        f.instruction(&W::I32Const(block.addr as i32));
        f.instruction(&W::GlobalSet(GUEST_PC_GLOBAL));
    }
    // Diagnostic per-block execution trace (opt-in): announce this block's entry to the
    // host `svc` handler in execution order (see `trace_blocks_range`).
    if let Some((lo, hi)) = trace_blocks_range() {
        if block.addr >= lo && block.addr <= hi {
            f.instruction(&W::I32Const(block.addr as i32));
            f.instruction(&W::Call(SVC_FUNC));
        }
    }
    for stmt in &block.stmts {
        emit_stmt(f, stmt, func_index, base, inline, func.addr);
    }
    emit_term(f, &block.term, func, base, loop_depth);
}

/// Re-dispatch to the block at `target` address: set `$bb`, branch to the loop.
/// `extra` accounts for any `if`/`block` frames open between here and the loop.
fn goto(f: &mut Function, func: &Func, target: u32, loop_depth: u32, extra: u32) {
    let idx = func
        .block_index(target)
        .unwrap_or_else(|| panic!("branch target {target:#x} is not a block in f_{:x}", func.addr))
        as i32;
    f.instruction(&W::I32Const(idx));
    f.instruction(&W::LocalSet(L_BB));
    f.instruction(&W::Br(loop_depth + extra));
}

fn emit_term(f: &mut Function, term: &Term, func: &Func, base: u32, loop_depth: u32) {
    match term {
        Term::Fallthrough => {} // flow into the next block's code
        Term::Return => {
            f.instruction(&W::Return);
        }
        // A `Halt` is a block that ran off the end of decoded code - almost always the
        // boundary just before an instruction the decoder could not lift. Normally it
        // returns (letting the rest of the program run, at the risk of an incomplete
        // function silently corrupting state downstream); with `VITASLOP_TRAP_HALT` it
        // traps instead, so the first undecoded-op cutoff actually reached at runtime
        // faults loudly with a backtrace (pair with `VITASLOP_TRACK_PC`/`_WASM_NAMES`).
        Term::Halt => {
            if trap_halt() {
                f.instruction(&W::Unreachable);
            } else {
                f.instruction(&W::Return);
            }
        }
        Term::Unreachable => {
            f.instruction(&W::Unreachable);
        }
        Term::Jump(target) => {
            goto(f, func, *target, loop_depth, 0);
        }
        // Computed jump-table dispatch. A `br_table` on the guest index selects one
        // of `n` landing pads; each sets `$bb` to the target block and re-enters the
        // dispatch loop. This is the wasm-native jump table: one `br_table` (plus
        // the loop's own dispatch `br_table`), no memory load, no linear compare
        // chain. The pads are `n+1` nested blocks - `T_0..T_{n-1}` plus an outer
        // `default` - so that `br_table` index `v` exits exactly `v` frames to land
        // after `T_v`'s `end`. From the pad for target `i`, `n - i` switch frames
        // are still open, so the branch back to the loop is `loop_depth + (n - i)`.
        Term::Switch { index, targets, default } => {
            let n = targets.len() as u32;
            for _ in 0..=n {
                f.instruction(&W::Block(BlockType::Empty));
            }
            emit_value(f, index, base);
            let table: Vec<u32> = (0..n).collect();
            f.instruction(&W::BrTable(table.into(), n /* default -> outer block */));
            for (i, &target) in targets.iter().enumerate() {
                f.instruction(&W::End); // closes T_i
                let idx = func.block_index(target).unwrap_or_else(|| {
                    panic!("switch target {target:#x} is not a block in f_{:x}", func.addr)
                }) as i32;
                f.instruction(&W::I32Const(idx));
                f.instruction(&W::LocalSet(L_BB));
                f.instruction(&W::Br(loop_depth + n - i as u32));
            }
            f.instruction(&W::End); // closes the default (outer) block
            match default {
                // The range check already routed out-of-range indices away, so this
                // is faithful when known and unreachable in practice.
                Some(d) => goto(f, func, *d, loop_depth, 0),
                None => {
                    f.instruction(&W::Unreachable);
                }
            }
        }
        Term::Branch { cond, taken } => {
            emit_cond(f, *cond);
            f.instruction(&W::If(BlockType::Empty));
            goto(f, func, *taken, loop_depth, 1); // +1 for the `if` frame
            f.instruction(&W::End);
        }
        Term::BranchZero { reg, nonzero, taken } => {
            f.instruction(&W::GlobalGet(abi::reg_global(*reg as usize)));
            f.instruction(&W::I32Eqz); // reg == 0
            if *nonzero {
                f.instruction(&W::I32Eqz); // reg != 0
            }
            f.instruction(&W::If(BlockType::Empty));
            goto(f, func, *taken, loop_depth, 1);
            f.instruction(&W::End);
        }
    }
}

/// Push a 0/1 i32 for `cond` computed from the flag globals.
fn emit_cond(f: &mut Function, cond: ConditionCode) {
    use ConditionCode::*;
    fn get(f: &mut Function, flag: abi::Flag) {
        f.instruction(&W::GlobalGet(abi::flag_global(flag)));
    }
    fn eqz(f: &mut Function) {
        f.instruction(&W::I32Eqz);
    }
    match cond {
        EQ => get(f, abi::Flag::Z),
        NE => { get(f, abi::Flag::Z); eqz(f); }
        HS => get(f, abi::Flag::C),
        LO => { get(f, abi::Flag::C); eqz(f); }
        MI => get(f, abi::Flag::N),
        PL => { get(f, abi::Flag::N); eqz(f); }
        VS => get(f, abi::Flag::V),
        VC => { get(f, abi::Flag::V); eqz(f); }
        HI => {
            // C && !Z
            get(f, abi::Flag::C);
            get(f, abi::Flag::Z);
            eqz(f);
            f.instruction(&W::I32And);
        }
        LS => {
            // !C || Z  ==  !(C && !Z)
            get(f, abi::Flag::C);
            get(f, abi::Flag::Z);
            eqz(f);
            f.instruction(&W::I32And);
            eqz(f);
        }
        GE => { get(f, abi::Flag::N); get(f, abi::Flag::V); f.instruction(&W::I32Eq); }
        LT => { get(f, abi::Flag::N); get(f, abi::Flag::V); f.instruction(&W::I32Ne); }
        GT => {
            // !Z && (N == V)
            get(f, abi::Flag::N);
            get(f, abi::Flag::V);
            f.instruction(&W::I32Eq);
            get(f, abi::Flag::Z);
            eqz(f);
            f.instruction(&W::I32And);
        }
        LE => {
            // Z || (N != V)
            get(f, abi::Flag::N);
            get(f, abi::Flag::V);
            f.instruction(&W::I32Ne);
            get(f, abi::Flag::Z);
            f.instruction(&W::I32Or);
        }
        AL => { f.instruction(&W::I32Const(1)); }
    }
}

/// Emit the read-watchpoint trap check. Precondition: the load's memory offset (guest
/// addr - base) is in `L_T0` and the loaded value is on top of the stack. Consumes the
/// value into `L_T1`, traps (`unreachable`) when the offset matches the watched address
/// (and, with `VITASLOP_WATCH_READ_NZ`, the value is non-zero) once more than
/// `VITASLOP_WATCH_READ_SKIP` earlier matches have passed, then leaves the value on the
/// stack. Shared by the integer and VFP-single load paths.
fn emit_read_watch_check(f: &mut Function, w: u32, base: u32) {
    f.instruction(&W::LocalSet(L_T1)); // value -> L_T1
    f.instruction(&W::LocalGet(L_T0));
    f.instruction(&W::I32Const(w.wrapping_sub(base) as i32));
    f.instruction(&W::I32Eq);
    if watch_read_nonzero() {
        f.instruction(&W::LocalGet(L_T1));
        f.instruction(&W::I32Eqz);
        f.instruction(&W::I32Eqz);
        f.instruction(&W::I32And);
    }
    if let Some((lo, hi)) = watch_read_pc_exclude() {
        // AND (guest_pc < lo OR guest_pc >= hi): only trap when the reader is OUTSIDE
        // the excluded address window (the known per-frame consumer).
        f.instruction(&W::GlobalGet(GUEST_PC_GLOBAL));
        f.instruction(&W::I32Const(lo as i32));
        f.instruction(&W::I32LtU);
        f.instruction(&W::GlobalGet(GUEST_PC_GLOBAL));
        f.instruction(&W::I32Const(hi as i32));
        f.instruction(&W::I32GeU);
        f.instruction(&W::I32Or);
        f.instruction(&W::I32And);
    }
    and_armed(f);
    f.instruction(&W::If(BlockType::Empty));
    // Matched: bump the counter and trap once past the skip window.
    f.instruction(&W::GlobalGet(WATCH_READ_COUNT_GLOBAL));
    f.instruction(&W::I32Const(1));
    f.instruction(&W::I32Add);
    f.instruction(&W::GlobalSet(WATCH_READ_COUNT_GLOBAL));
    f.instruction(&W::GlobalGet(WATCH_READ_COUNT_GLOBAL));
    f.instruction(&W::I32Const(watch_read_skip() as i32));
    f.instruction(&W::I32GtU);
    f.instruction(&W::If(BlockType::Empty));
    f.instruction(&W::Unreachable);
    f.instruction(&W::End);
    f.instruction(&W::End);
    f.instruction(&W::LocalGet(L_T1)); // value back on the stack
}

/// Snapshot the guarded callee-saved register into [`L_GUARD`] just before a call
/// (no-op unless `VITASLOP_GUARD_REG` is set). See [`guard_reg`].
fn guard_snapshot(f: &mut Function) {
    if let Some(r) = guard_reg() {
        f.instruction(&W::GlobalGet(abi::reg_global(r as usize)));
        f.instruction(&W::LocalSet(L_GUARD));
    }
}

/// After a call returns, trap if the guarded register differs from its pre-call
/// snapshot - the callee failed to preserve it (no-op unless the guard is set).
fn guard_check(f: &mut Function) {
    if let Some(r) = guard_reg() {
        f.instruction(&W::GlobalGet(abi::reg_global(r as usize)));
        f.instruction(&W::LocalGet(L_GUARD));
        f.instruction(&W::I32Ne);
        and_armed(f);
        f.instruction(&W::If(BlockType::Empty));
        f.instruction(&W::Unreachable);
        f.instruction(&W::End);
    }
}

fn emit_stmt(
    f: &mut Function,
    stmt: &Stmt,
    func_index: &BTreeMap<u32, u32>,
    base: u32,
    inline: &InlineImports,
    func_addr: u32,
) {
    match stmt {
        Stmt::SetReg(r, v) => {
            emit_value(f, v, base);
            f.instruction(&W::GlobalSet(abi::reg_global(*r as usize)));
        }
        Stmt::Store { addr, data, size } => {
            if let Some(w) = watch_store_addr() {
                // Diagnostic: trap on a store to the watched guest address so the
                // writer surfaces with a backtrace (and the value in the reg dump).
                // Save offset and data in scratch locals, test the condition, trap,
                // then perform the real store from the locals.
                emit_addr(f, addr, base);
                f.instruction(&W::LocalSet(L_T0)); // L_T0 = guest addr - base
                emit_value(f, data, base);
                f.instruction(&W::LocalSet(L_T1)); // L_T1 = value
                if watch_store_arm_mode() {
                    // arm mode: on a store to the watched address, arm the latch when
                    // the value is non-zero, and trap on a zero store once armed.
                    f.instruction(&W::LocalGet(L_T0));
                    f.instruction(&W::I32Const(w.wrapping_sub(base) as i32));
                    f.instruction(&W::I32Eq);
                    and_armed(f);
                    f.instruction(&W::If(BlockType::Empty));
                    f.instruction(&W::LocalGet(L_T1));
                    f.instruction(&W::If(BlockType::Empty)); // value != 0 -> arm
                    f.instruction(&W::I32Const(1));
                    f.instruction(&W::GlobalSet(WATCH_ARMED_GLOBAL));
                    f.instruction(&W::Else); // value == 0 -> trap if armed
                    f.instruction(&W::GlobalGet(WATCH_ARMED_GLOBAL));
                    f.instruction(&W::If(BlockType::Empty));
                    f.instruction(&W::Unreachable);
                    f.instruction(&W::End);
                    f.instruction(&W::End);
                    f.instruction(&W::End);
                } else {
                    f.instruction(&W::LocalGet(L_T0));
                    f.instruction(&W::I32Const(w.wrapping_sub(base) as i32));
                    f.instruction(&W::I32Eq);
                    if watch_store_nonzero() {
                        // AND value != 0.
                        f.instruction(&W::LocalGet(L_T1));
                        f.instruction(&W::I32Eqz);
                        f.instruction(&W::I32Eqz);
                        f.instruction(&W::I32And);
                    }
                    and_armed(f);
                    f.instruction(&W::If(BlockType::Empty));
                    // Matched: bump the counter and trap once past the skip window.
                    f.instruction(&W::GlobalGet(WATCH_STORE_COUNT_GLOBAL));
                    f.instruction(&W::I32Const(1));
                    f.instruction(&W::I32Add);
                    f.instruction(&W::GlobalSet(WATCH_STORE_COUNT_GLOBAL));
                    f.instruction(&W::GlobalGet(WATCH_STORE_COUNT_GLOBAL));
                    f.instruction(&W::I32Const(watch_store_skip() as i32));
                    f.instruction(&W::I32GtU);
                    f.instruction(&W::If(BlockType::Empty));
                    f.instruction(&W::Unreachable);
                    f.instruction(&W::End);
                    f.instruction(&W::End);
                }
                f.instruction(&W::LocalGet(L_T0));
                f.instruction(&W::LocalGet(L_T1));
                f.instruction(&store_op(*size));
            } else {
                emit_addr(f, addr, base);
                emit_value(f, data, base);
                f.instruction(&store_op(*size));
            }
        }
        Stmt::FlagsAdd { a, b, cin } => emit_flags_add(f, a, b, cin, base),
        Stmt::FlagsLogic { value, carry } => {
            emit_value(f, value, base);
            f.instruction(&W::LocalTee(L_T0));
            f.instruction(&W::I32Eqz);
            f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::Z)));
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::I32Const(31));
            f.instruction(&W::I32ShrU);
            f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::N)));
            if let Some(c) = carry {
                emit_value(f, c, base);
                f.instruction(&W::I32Const(1));
                f.instruction(&W::I32And);
                f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::C)));
            }
        }
        Stmt::Svc(imm) => {
            f.instruction(&W::I32Const(*imm as i32));
            f.instruction(&W::Call(SVC_FUNC));
        }
        Stmt::Import(index) => emit_import(f, *index, base, inline),
        Stmt::Rbit { rd, rm } => {
            // Reverse all 32 bits with the classic swap network, over a scratch
            // local (the input is read twice per step). Four masked adjacent-group
            // swaps, then a halfword swap (a rotate by 16).
            emit_value(f, rm, base);
            f.instruction(&W::LocalSet(L_T0));
            for (shift, mask) in [(1u32, 0x5555_5555u32), (2, 0x3333_3333), (4, 0x0f0f_0f0f), (8, 0x00ff_00ff)] {
                // ((x >> shift) & mask) | ((x & mask) << shift)
                f.instruction(&W::LocalGet(L_T0));
                f.instruction(&W::I32Const(shift as i32));
                f.instruction(&W::I32ShrU);
                f.instruction(&W::I32Const(mask as i32));
                f.instruction(&W::I32And);
                f.instruction(&W::LocalGet(L_T0));
                f.instruction(&W::I32Const(mask as i32));
                f.instruction(&W::I32And);
                f.instruction(&W::I32Const(shift as i32));
                f.instruction(&W::I32Shl);
                f.instruction(&W::I32Or);
                f.instruction(&W::LocalSet(L_T0));
            }
            // Swap the two 16-bit halves: (x << 16) | (x >> 16) == rotr 16.
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::I32Const(16));
            f.instruction(&W::I32Rotr);
            f.instruction(&W::GlobalSet(abi::reg_global(*rd as usize)));
        }
        Stmt::MulLong { rdlo, rdhi, rn, rm, signed } => {
            // Extend both operands to i64 (sign- or zero-extend), multiply, and
            // split the 64-bit product into its low and high 32-bit halves. The
            // full product is computed before either register is written, so an
            // operand that aliases a destination reads its old value first.
            let extend = |f: &mut Function| {
                if *signed {
                    f.instruction(&W::I64ExtendI32S);
                } else {
                    f.instruction(&W::I64ExtendI32U);
                }
            };
            emit_value(f, rn, base);
            extend(f);
            emit_value(f, rm, base);
            extend(f);
            f.instruction(&W::I64Mul);
            f.instruction(&W::LocalTee(L_D64)); // full product, kept for the high half
            f.instruction(&W::I32WrapI64); // low 32 bits
            f.instruction(&W::GlobalSet(abi::reg_global(*rdlo as usize)));
            f.instruction(&W::LocalGet(L_D64));
            f.instruction(&W::I64Const(32));
            f.instruction(&W::I64ShrU);
            f.instruction(&W::I32WrapI64); // high 32 bits
            f.instruction(&W::GlobalSet(abi::reg_global(*rdhi as usize)));
        }
        Stmt::Call { target } => {
            // A direct call to a discovered function. In the strict build every callee
            // is discovered, so the lookup always succeeds. In the lenient build a
            // block reached only speculatively (runaway decode of data) can hold a
            // `bl` to a bogus address that is not a real function; trap there rather
            // than fail the whole emit - the block is off the real execution path, and
            // if somehow reached it faults loudly.
            guard_snapshot(f);
            match func_index.get(target) {
                Some(&idx) => {
                    f.instruction(&W::Call(idx));
                }
                None => {
                    f.instruction(&W::Unreachable);
                }
            }
            guard_check(f);
        }
        Stmt::CallIndirect { addr, set_lr } => {
            guard_snapshot(f);
            // Push the runtime target address and this function's own address (as the
            // caller, for a `dispatch_miss` report), then call the module dispatcher,
            // which resolves the target to the matching translated function. The
            // dispatcher is the last defined function: IMPORT_FUNCS + one per guest
            // func. Its signature is `(target, caller)`, so both are on the stack.
            let dispatch = IMPORT_FUNCS + func_index.len() as u32;
            match set_lr {
                // `blx rN`: snapshot the target BEFORE writing lr, because the target
                // register can be lr itself (a compiler using lr as call-target
                // scratch) - writing lr first would dispatch to the return address.
                Some(lr) => {
                    emit_value(f, addr, base);
                    f.instruction(&W::LocalSet(L_T0));
                    f.instruction(&W::I32Const(*lr as i32));
                    f.instruction(&W::GlobalSet(abi::reg_global(14)));
                    f.instruction(&W::LocalGet(L_T0)); // target
                    f.instruction(&W::I32Const(func_addr as i32)); // caller
                    f.instruction(&W::Call(dispatch));
                }
                // `bx rN` tail call: lr is untouched.
                None => {
                    emit_value(f, addr, base); // target
                    f.instruction(&W::I32Const(func_addr as i32)); // caller
                    f.instruction(&W::Call(dispatch));
                }
            }
            guard_check(f);
        }
        Stmt::Guard(cond, body) => {
            emit_cond(f, *cond);
            f.instruction(&W::If(BlockType::Empty));
            for s in body {
                emit_stmt(f, s, func_index, base, inline, func_addr);
            }
            f.instruction(&W::End);
        }
        Stmt::Vfp(op) => emit_vfp(f, op),
        Stmt::VfpMem { reg, addr, load } => emit_vfp_mem(f, *reg, addr, *load, base),
        Stmt::Neon(op) => emit_neon(f, op, base),
        Stmt::SetThreadPtr(v) => {
            emit_value(f, v, base);
            f.instruction(&W::GlobalSet(abi::TP_GLOBAL));
        }
        Stmt::Uadd8 { rd, rn, rm } => emit_uadd8(f, *rd, *rn, *rm),
        Stmt::Sel { rd, rn, rm } => emit_sel(f, *rd, *rn, *rm),
        Stmt::ShiftRegFlags { kind, rd, rn, amount, set_flags } => {
            emit_shift_reg_flags(f, *kind, *rd, rn, amount, *set_flags, base)
        }
    }
}

/// Push byte `i` (0..3) of register `r`, zero-extended: `(r >> 8i) & 0xff`.
fn push_reg_byte(f: &mut Function, r: u8, i: u32) {
    f.instruction(&W::GlobalGet(abi::reg_global(r as usize)));
    if i != 0 {
        f.instruction(&W::I32Const((8 * i) as i32));
        f.instruction(&W::I32ShrU);
    }
    f.instruction(&W::I32Const(0xff));
    f.instruction(&W::I32And);
}

/// `uadd8 rd, rn, rm`: four independent byte adds, each depositing its unsigned
/// carry-out into an APSR GE bit (held in `L_GE`) for a later `sel`. The full
/// result is staged in `L_T2` and the GE mask in `L_GE` before `rd` is written,
/// so `rd` aliasing `rn`/`rm` (e.g. `uadd8 r2, r2, ip`) is safe.
fn emit_uadd8(f: &mut Function, rd: u8, rn: u8, rm: u8) {
    f.instruction(&W::I32Const(0));
    f.instruction(&W::LocalSet(L_T2)); // result accumulator
    f.instruction(&W::I32Const(0));
    f.instruction(&W::LocalSet(L_GE)); // GE bits accumulator
    for i in 0..4u32 {
        // L_T0 = rn.byte[i] + rm.byte[i]  (0..510)
        push_reg_byte(f, rn, i);
        push_reg_byte(f, rm, i);
        f.instruction(&W::I32Add);
        f.instruction(&W::LocalSet(L_T0));
        // result |= (L_T0 & 0xff) << 8i
        f.instruction(&W::LocalGet(L_T2));
        f.instruction(&W::LocalGet(L_T0));
        f.instruction(&W::I32Const(0xff));
        f.instruction(&W::I32And);
        if i != 0 {
            f.instruction(&W::I32Const((8 * i) as i32));
            f.instruction(&W::I32Shl);
        }
        f.instruction(&W::I32Or);
        f.instruction(&W::LocalSet(L_T2));
        // GE |= (L_T0 >> 8) << i   (the sum is <= 510, so bit 8 is the carry-out)
        f.instruction(&W::LocalGet(L_GE));
        f.instruction(&W::LocalGet(L_T0));
        f.instruction(&W::I32Const(8));
        f.instruction(&W::I32ShrU);
        if i != 0 {
            f.instruction(&W::I32Const(i as i32));
            f.instruction(&W::I32Shl);
        }
        f.instruction(&W::I32Or);
        f.instruction(&W::LocalSet(L_GE));
    }
    f.instruction(&W::LocalGet(L_T2));
    f.instruction(&W::GlobalSet(abi::reg_global(rd as usize)));
}

/// `sel rd, rn, rm`: for each byte, pick `rn`'s byte where the GE bit (in `L_GE`)
/// is set, else `rm`'s. Branchless per byte via a 0x00/0xff mask; staged in
/// `L_T2` before writing `rd` so aliasing is safe.
fn emit_sel(f: &mut Function, rd: u8, rn: u8, rm: u8) {
    f.instruction(&W::I32Const(0));
    f.instruction(&W::LocalSet(L_T2)); // result accumulator
    for i in 0..4u32 {
        // L_T1 = mask = 0 - ((GE >> i) & 1)   (0x00000000 or 0xffffffff)
        f.instruction(&W::I32Const(0));
        f.instruction(&W::LocalGet(L_GE));
        if i != 0 {
            f.instruction(&W::I32Const(i as i32));
            f.instruction(&W::I32ShrU);
        }
        f.instruction(&W::I32Const(1));
        f.instruction(&W::I32And);
        f.instruction(&W::I32Sub);
        f.instruction(&W::LocalSet(L_T1));
        // L_T0 = rn.byte[i], L_T3 = rm.byte[i]
        push_reg_byte(f, rn, i);
        f.instruction(&W::LocalSet(L_T0));
        push_reg_byte(f, rm, i);
        f.instruction(&W::LocalSet(L_T3));
        // chosen = rm ^ ((rn ^ rm) & mask)  == mask ? rn : rm
        f.instruction(&W::LocalGet(L_T3));
        f.instruction(&W::LocalGet(L_T0));
        f.instruction(&W::LocalGet(L_T3));
        f.instruction(&W::I32Xor);
        f.instruction(&W::LocalGet(L_T1));
        f.instruction(&W::I32And);
        f.instruction(&W::I32Xor);
        // result |= chosen << 8i
        if i != 0 {
            f.instruction(&W::I32Const((8 * i) as i32));
            f.instruction(&W::I32Shl);
        }
        f.instruction(&W::LocalGet(L_T2));
        f.instruction(&W::I32Or);
        f.instruction(&W::LocalSet(L_T2));
    }
    f.instruction(&W::LocalGet(L_T2));
    f.instruction(&W::GlobalSet(abi::reg_global(rd as usize)));
}

/// Emit the exact ARM register-controlled shift `lsl/lsr/asr Rd, Rn, Rm` where the
/// amount is a runtime value (`Rm[7:0]`, 0..255). wasm shifts mask the amount mod 32,
/// so `lsl` by >=32 must be forced to 0 (not `Rn << (amt & 31)`); and ARM's shifter
/// carry-out for a ZERO amount is the OLD carry (unchanged), which no constant carry
/// expression captures - so both the result and the carry are modeled explicitly. Sets
/// N,Z,C when `set_flags` (a shift never affects V). Scratch: L_T0=value, L_T1=amount,
/// L_T2=result, L_T3=carry. `value`/`amount` are read before `rd` is written, so a
/// shift whose amount or source aliases `rd` (e.g. `lsls r0, r3, r0`) is correct.
fn emit_shift_reg_flags(
    f: &mut Function,
    kind: crate::ir::ShiftKind,
    rd: u8,
    rn: &Value,
    amount: &Value,
    set_flags: bool,
    base: u32,
) {
    use crate::ir::ShiftKind::*;
    emit_value(f, rn, base);
    f.instruction(&W::LocalSet(L_T0)); // value
    emit_value(f, amount, base);
    f.instruction(&W::I32Const(0xff));
    f.instruction(&W::I32And);
    f.instruction(&W::LocalSet(L_T1)); // amt = Rm[7:0]

    // result -> L_T2. wasm Select is `a b c -> (c != 0 ? a : b)`.
    match kind {
        Lsl => {
            // amt < 32 ? val << amt : 0
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32Shl);
            f.instruction(&W::I32Const(0));
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32Const(32));
            f.instruction(&W::I32LtU);
            f.instruction(&W::Select);
        }
        Lsr => {
            // amt < 32 ? val >>u amt : 0
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32ShrU);
            f.instruction(&W::I32Const(0));
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32Const(32));
            f.instruction(&W::I32LtU);
            f.instruction(&W::Select);
        }
        Asr => {
            // amt < 32 ? val >>s amt : val >>s 31  (arithmetic sign-fill for amt>=32)
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32ShrS);
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::I32Const(31));
            f.instruction(&W::I32ShrS);
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32Const(32));
            f.instruction(&W::I32LtU);
            f.instruction(&W::Select);
        }
    }
    f.instruction(&W::LocalSet(L_T2));
    f.instruction(&W::LocalGet(L_T2));
    f.instruction(&W::GlobalSet(abi::reg_global(rd as usize)));

    if !set_flags {
        return;
    }
    // Z = (result == 0); N = result[31].
    f.instruction(&W::LocalGet(L_T2));
    f.instruction(&W::I32Eqz);
    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::Z)));
    f.instruction(&W::LocalGet(L_T2));
    f.instruction(&W::I32Const(31));
    f.instruction(&W::I32ShrU);
    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::N)));
    // Shifter carry-out for a NON-zero amount -> L_T3 (the amt==0 case is folded in
    // by the final select, which keeps the old C).
    match kind {
        Lsl => {
            // amt <= 32 ? (val >>u (32 - amt)) & 1 : 0   [amt in 1..=32 -> shift 0..=31]
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::I32Const(32));
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32Sub);
            f.instruction(&W::I32ShrU);
            f.instruction(&W::I32Const(1));
            f.instruction(&W::I32And);
            f.instruction(&W::I32Const(0));
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32Const(32));
            f.instruction(&W::I32LeU);
            f.instruction(&W::Select);
        }
        Lsr => {
            // amt <= 32 ? (val >>u (amt - 1)) & 1 : 0    [amt in 1..=32 -> shift 0..=31]
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32Const(1));
            f.instruction(&W::I32Sub);
            f.instruction(&W::I32ShrU);
            f.instruction(&W::I32Const(1));
            f.instruction(&W::I32And);
            f.instruction(&W::I32Const(0));
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32Const(32));
            f.instruction(&W::I32LeU);
            f.instruction(&W::Select);
        }
        Asr => {
            // (val >>s min(amt - 1, 31)) & 1   [amt>=32 -> the sign bit; ASR carry is
            // defined for all amounts, so no amt>32 zero branch].
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32Const(1));
            f.instruction(&W::I32Sub);
            f.instruction(&W::I32Const(31));
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32Const(32));
            f.instruction(&W::I32LtU);
            f.instruction(&W::Select); // min(amt-1, 31)
            f.instruction(&W::I32ShrS);
            f.instruction(&W::I32Const(1));
            f.instruction(&W::I32And);
        }
    }
    f.instruction(&W::LocalSet(L_T3));
    // C = (amt == 0) ? old_C : L_T3
    f.instruction(&W::GlobalGet(abi::flag_global(abi::Flag::C)));
    f.instruction(&W::LocalGet(L_T3));
    f.instruction(&W::LocalGet(L_T1));
    f.instruction(&W::I32Eqz);
    f.instruction(&W::Select);
    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::C)));
}

// --- VFP / floating-point emission ---------------------------------------

/// Push S`n` interpreted as an f32.
fn get_s_f32(f: &mut Function, n: u8) {
    f.instruction(&W::GlobalGet(abi::vfp_s_global(n)));
    f.instruction(&W::F32ReinterpretI32);
}

/// Store the f32 on the stack into S`n` (as raw bits).
fn set_s_f32(f: &mut Function, n: u8) {
    f.instruction(&W::I32ReinterpretF32);
    f.instruction(&W::GlobalSet(abi::vfp_s_global(n)));
}

/// Push the raw 64 bits of D`n` as an i64. Low bank (n < 16): merge the two S
/// halves. Upper bank (n >= 16): extract `i64x2` lane `n & 1` of the quad `q(n/2)`.
fn get_d_bits(f: &mut Function, n: u8) {
    if (n as usize) < abi::VFP_D_HI_FIRST {
        let lo = 2 * n;
        let hi = 2 * n + 1;
        f.instruction(&W::GlobalGet(abi::vfp_s_global(hi)));
        f.instruction(&W::I64ExtendI32U);
        f.instruction(&W::I64Const(32));
        f.instruction(&W::I64Shl);
        f.instruction(&W::GlobalGet(abi::vfp_s_global(lo)));
        f.instruction(&W::I64ExtendI32U);
        f.instruction(&W::I64Or);
    } else {
        f.instruction(&W::GlobalGet(abi::vfp_qhi_global(n / 2)));
        f.instruction(&W::I64x2ExtractLane(n & 1));
    }
}

/// Store the raw i64 on the stack into D`n`. Low bank (n < 16): split into the two
/// S halves. Upper bank (n >= 16): replace `i64x2` lane `n & 1` of quad `q(n/2)`,
/// leaving the sibling D untouched. Uses the i64 scratch local.
fn set_d_bits(f: &mut Function, n: u8) {
    if (n as usize) < abi::VFP_D_HI_FIRST {
        let lo = 2 * n;
        let hi = 2 * n + 1;
        f.instruction(&W::LocalSet(L_D64));
        f.instruction(&W::LocalGet(L_D64));
        f.instruction(&W::I32WrapI64);
        f.instruction(&W::GlobalSet(abi::vfp_s_global(lo)));
        f.instruction(&W::LocalGet(L_D64));
        f.instruction(&W::I64Const(32));
        f.instruction(&W::I64ShrU);
        f.instruction(&W::I32WrapI64);
        f.instruction(&W::GlobalSet(abi::vfp_s_global(hi)));
    } else {
        // Read-modify-write the quad so the sibling D lane survives.
        let q = abi::vfp_qhi_global(n / 2);
        f.instruction(&W::LocalSet(L_D64));
        f.instruction(&W::GlobalGet(q));
        f.instruction(&W::LocalGet(L_D64));
        f.instruction(&W::I64x2ReplaceLane(n & 1));
        f.instruction(&W::GlobalSet(q));
    }
}

/// Push D`n` interpreted as an f64 (its raw 64 bits reinterpreted).
fn get_d_f64(f: &mut Function, n: u8) {
    get_d_bits(f, n);
    f.instruction(&W::F64ReinterpretI64);
}

/// Store the f64 on the stack into D`n` (as raw bits).
fn set_d_f64(f: &mut Function, n: u8) {
    f.instruction(&W::I64ReinterpretF64);
    set_d_bits(f, n);
}

fn fbinop(op: crate::ir::FBinOp) -> W<'static> {
    use crate::ir::FBinOp::*;
    match op {
        Add => W::F32Add,
        Sub => W::F32Sub,
        Mul => W::F32Mul,
        Div => W::F32Div,
    }
}

fn fbinop64(op: crate::ir::FBinOp) -> W<'static> {
    use crate::ir::FBinOp::*;
    match op {
        Add => W::F64Add,
        Sub => W::F64Sub,
        Mul => W::F64Mul,
        Div => W::F64Div,
    }
}

fn emit_vfp(f: &mut Function, op: &crate::ir::VfpOp) {
    use crate::ir::VfpOp::*;
    match op {
        Bin32 { op, rd, rn, rm } => {
            get_s_f32(f, *rn);
            get_s_f32(f, *rm);
            f.instruction(&fbinop(*op));
            set_s_f32(f, *rd);
        }
        MulAcc32 { rd, rn, rm, sub, neg } => {
            // rd = (-rd if neg else rd) +/- (rn * rm), non-fused.
            get_s_f32(f, *rd);
            if *neg {
                f.instruction(&W::F32Neg);
            }
            get_s_f32(f, *rn);
            get_s_f32(f, *rm);
            f.instruction(&W::F32Mul);
            f.instruction(if *sub { &W::F32Sub } else { &W::F32Add });
            set_s_f32(f, *rd);
        }
        NegMul32 { rd, rn, rm } => {
            get_s_f32(f, *rn);
            get_s_f32(f, *rm);
            f.instruction(&W::F32Mul);
            f.instruction(&W::F32Neg);
            set_s_f32(f, *rd);
        }
        Neg32 { rd, rm } => {
            get_s_f32(f, *rm);
            f.instruction(&W::F32Neg);
            set_s_f32(f, *rd);
        }
        Abs32 { rd, rm } => {
            get_s_f32(f, *rm);
            f.instruction(&W::F32Abs);
            set_s_f32(f, *rd);
        }
        Sqrt32 { rd, rm } => {
            get_s_f32(f, *rm);
            f.instruction(&W::F32Sqrt);
            set_s_f32(f, *rd);
        }
        Mov32 { rd, rm } => {
            // Raw bit copy (no float interpretation).
            f.instruction(&W::GlobalGet(abi::vfp_s_global(*rm)));
            f.instruction(&W::GlobalSet(abi::vfp_s_global(*rd)));
        }
        ScalarToCore { rt, s } => {
            f.instruction(&W::GlobalGet(abi::vfp_s_global(*s)));
            f.instruction(&W::GlobalSet(abi::reg_global(*rt as usize)));
        }
        CoreToScalar { s, rt } => {
            f.instruction(&W::GlobalGet(abi::reg_global(*rt as usize)));
            f.instruction(&W::GlobalSet(abi::vfp_s_global(*s)));
        }
        SetImmS { s, bits } => {
            f.instruction(&W::I32Const(*bits as i32));
            f.instruction(&W::GlobalSet(abi::vfp_s_global(*s)));
        }
        SetImmD { d, lo, hi } => {
            if (*d as usize) < abi::VFP_D_HI_FIRST {
                // Aliased low double: write the two S halves directly.
                f.instruction(&W::I32Const(*lo as i32));
                f.instruction(&W::GlobalSet(abi::vfp_s_global(2 * d)));
                f.instruction(&W::I32Const(*hi as i32));
                f.instruction(&W::GlobalSet(abi::vfp_s_global(2 * d + 1)));
            } else {
                // Upper-bank double: an i64 lane of the quad (set_d_bits does the RMW).
                let bits = ((*hi as u64) << 32) | (*lo as u64);
                f.instruction(&W::I64Const(bits as i64));
                set_d_bits(f, *d);
            }
        }
        Cmp32 { rn, rm } => emit_vfp_cmp(f, *rn, *rm),
        MrsNzcv => {
            // Copy FP flags -> integer NZCV.
            for flag in [abi::Flag::N, abi::Flag::Z, abi::Flag::C, abi::Flag::V] {
                f.instruction(&W::GlobalGet(abi::fp_flag_global(flag)));
                f.instruction(&W::GlobalSet(abi::flag_global(flag)));
            }
        }
        MrsFpscr { rt } => {
            // rt = (N<<31) | (Z<<30) | (C<<29) | (V<<28); other FPSCR bits zero.
            for (i, flag) in [abi::Flag::N, abi::Flag::Z, abi::Flag::C, abi::Flag::V].iter().enumerate() {
                f.instruction(&W::GlobalGet(abi::fp_flag_global(*flag)));
                f.instruction(&W::I32Const(31 - i as i32));
                f.instruction(&W::I32Shl);
                if i > 0 {
                    f.instruction(&W::I32Or);
                }
            }
            f.instruction(&W::GlobalSet(abi::reg_global(*rt as usize)));
        }
        CvtToInt { rd, rm, signed } => {
            // Round toward zero, saturating (matches ARM vcvt.{s,u}32.f32).
            get_s_f32(f, *rm);
            f.instruction(if *signed { &W::I32TruncSatF32S } else { &W::I32TruncSatF32U });
            f.instruction(&W::GlobalSet(abi::vfp_s_global(*rd)));
        }
        CvtFromInt { rd, rm, signed } => {
            f.instruction(&W::GlobalGet(abi::vfp_s_global(*rm)));
            f.instruction(if *signed { &W::F32ConvertI32S } else { &W::F32ConvertI32U });
            set_s_f32(f, *rd);
        }

        // --- Double precision (f64) ---
        Bin64 { op, rd, rn, rm } => {
            get_d_f64(f, *rn);
            get_d_f64(f, *rm);
            f.instruction(&fbinop64(*op));
            set_d_f64(f, *rd);
        }
        MulAcc64 { rd, rn, rm, sub, neg } => {
            get_d_f64(f, *rd);
            if *neg {
                f.instruction(&W::F64Neg);
            }
            get_d_f64(f, *rn);
            get_d_f64(f, *rm);
            f.instruction(&W::F64Mul);
            f.instruction(if *sub { &W::F64Sub } else { &W::F64Add });
            set_d_f64(f, *rd);
        }
        NegMul64 { rd, rn, rm } => {
            get_d_f64(f, *rn);
            get_d_f64(f, *rm);
            f.instruction(&W::F64Mul);
            f.instruction(&W::F64Neg);
            set_d_f64(f, *rd);
        }
        Neg64 { rd, rm } => {
            get_d_f64(f, *rm);
            f.instruction(&W::F64Neg);
            set_d_f64(f, *rd);
        }
        Abs64 { rd, rm } => {
            get_d_f64(f, *rm);
            f.instruction(&W::F64Abs);
            set_d_f64(f, *rd);
        }
        Sqrt64 { rd, rm } => {
            get_d_f64(f, *rm);
            f.instruction(&W::F64Sqrt);
            set_d_f64(f, *rd);
        }
        Mov64 { rd, rm } => {
            // Raw 64-bit copy (no float interpretation).
            get_d_bits(f, *rm);
            set_d_bits(f, *rd);
        }
        Cmp64 { rn, rm } => emit_vfp_cmp64(f, *rn, *rm),
        CvtF64FromInt { d, s, signed } => {
            f.instruction(&W::GlobalGet(abi::vfp_s_global(*s)));
            f.instruction(if *signed { &W::F64ConvertI32S } else { &W::F64ConvertI32U });
            set_d_f64(f, *d);
        }
        CvtIntFromF64 { s, d, signed } => {
            get_d_f64(f, *d);
            f.instruction(if *signed { &W::I32TruncSatF64S } else { &W::I32TruncSatF64U });
            f.instruction(&W::GlobalSet(abi::vfp_s_global(*s)));
        }
        CvtF64FromF32 { d, s } => {
            get_s_f32(f, *s);
            f.instruction(&W::F64PromoteF32);
            set_d_f64(f, *d);
        }
        CvtF32FromF64 { s, d } => {
            get_d_f64(f, *d);
            f.instruction(&W::F32DemoteF64);
            set_s_f32(f, *s);
        }
        CvtF32FromHalf { sd, sm, top } => {
            // IEEE f16 -> f32, branchless (Giesen): scale the exponent/mantissa bits by
            // a float multiply, force the exponent for inf/NaN, then splice the sign.
            // Magic 0x7780_0000 = 2^112 (the f16->f32 exponent bias difference); the
            // inf/NaN threshold 0x4780_0000 = 65536.0.
            const MAGIC: i32 = 0x7780_0000u32 as i32;
            const INF_NAN_THRESHOLD: i32 = 0x4780_0000u32 as i32;
            // h = the selected 16-bit half of sm, in the low bits (rest zero).
            f.instruction(&W::GlobalGet(abi::vfp_s_global(*sm)));
            if *top {
                f.instruction(&W::I32Const(16));
                f.instruction(&W::I32ShrU);
            } else {
                f.instruction(&W::I32Const(0xffff));
                f.instruction(&W::I32And);
            }
            f.instruction(&W::LocalSet(L_T1)); // h
            // o.u = bits( f32((h & 0x7fff) << 13) * 2^112 )
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32Const(0x7fff));
            f.instruction(&W::I32And);
            f.instruction(&W::I32Const(13));
            f.instruction(&W::I32Shl);
            f.instruction(&W::F32ReinterpretI32);
            f.instruction(&W::I32Const(MAGIC));
            f.instruction(&W::F32ReinterpretI32);
            f.instruction(&W::F32Mul);
            f.instruction(&W::I32ReinterpretF32);
            f.instruction(&W::LocalSet(L_T0)); // o.u
            // result = o.u | (o.u >=u threshold ? 0x7f80_0000 : 0) | sign
            f.instruction(&W::LocalGet(L_T0));
            // inf/NaN mask:
            f.instruction(&W::I32Const(0));
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::I32Const(INF_NAN_THRESHOLD));
            f.instruction(&W::I32GeU);
            f.instruction(&W::I32Sub); // 0 - cond = 0 or 0xffff_ffff
            f.instruction(&W::I32Const(0x7f80_0000u32 as i32));
            f.instruction(&W::I32And);
            f.instruction(&W::I32Or);
            // sign = (h & 0x8000) << 16:
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32Const(0x8000));
            f.instruction(&W::I32And);
            f.instruction(&W::I32Const(16));
            f.instruction(&W::I32Shl);
            f.instruction(&W::I32Or);
            f.instruction(&W::GlobalSet(abi::vfp_s_global(*sd)));
        }
        DoubleToCore { rt, rt2, d } => {
            get_d_bits(f, *d);
            f.instruction(&W::LocalTee(L_D64));
            f.instruction(&W::I32WrapI64); // low 32 -> rt
            f.instruction(&W::GlobalSet(abi::reg_global(*rt as usize)));
            f.instruction(&W::LocalGet(L_D64));
            f.instruction(&W::I64Const(32));
            f.instruction(&W::I64ShrU);
            f.instruction(&W::I32WrapI64); // high 32 -> rt2
            f.instruction(&W::GlobalSet(abi::reg_global(*rt2 as usize)));
        }
        CoreToDouble { d, rt, rt2 } => {
            f.instruction(&W::GlobalGet(abi::reg_global(*rt2 as usize)));
            f.instruction(&W::I64ExtendI32U);
            f.instruction(&W::I64Const(32));
            f.instruction(&W::I64Shl);
            f.instruction(&W::GlobalGet(abi::reg_global(*rt as usize)));
            f.instruction(&W::I64ExtendI32U);
            f.instruction(&W::I64Or);
            set_d_bits(f, *d);
        }
    }
}

/// Set the FP condition flags from comparing D`rn` against D`rm` (or `+0.0`), the
/// f64 twin of [`emit_vfp_cmp`]. N=less, Z=equal, C=not-less, V=unordered.
fn emit_vfp_cmp64(f: &mut Function, rn: u8, rm: Option<u8>) {
    let push_b = |f: &mut Function| match rm {
        Some(m) => get_d_f64(f, m),
        None => {
            f.instruction(&W::F64Const(0.0f64.into()));
        }
    };
    get_d_f64(f, rn);
    push_b(f);
    f.instruction(&W::F64Lt);
    f.instruction(&W::GlobalSet(abi::fp_flag_global(abi::Flag::N)));
    get_d_f64(f, rn);
    push_b(f);
    f.instruction(&W::F64Eq);
    f.instruction(&W::GlobalSet(abi::fp_flag_global(abi::Flag::Z)));
    get_d_f64(f, rn);
    push_b(f);
    f.instruction(&W::F64Lt);
    f.instruction(&W::I32Eqz);
    f.instruction(&W::GlobalSet(abi::fp_flag_global(abi::Flag::C)));
    get_d_f64(f, rn);
    get_d_f64(f, rn);
    f.instruction(&W::F64Ne);
    push_b(f);
    push_b(f);
    f.instruction(&W::F64Ne);
    f.instruction(&W::I32Or);
    f.instruction(&W::GlobalSet(abi::fp_flag_global(abi::Flag::V)));
}

/// Set the FP condition flags (FPSCR N,Z,C,V) from comparing S`rn` against S`rm`
/// (or +0.0 when `rm` is `None`). N=less, Z=equal, C=not-less, V=unordered.
fn emit_vfp_cmp(f: &mut Function, rn: u8, rm: Option<u8>) {
    let push_b = |f: &mut Function| match rm {
        Some(m) => get_s_f32(f, m),
        None => {
            f.instruction(&W::F32Const(0.0f32.into()));
        }
    };
    // N = (a < b)
    get_s_f32(f, rn);
    push_b(f);
    f.instruction(&W::F32Lt);
    f.instruction(&W::GlobalSet(abi::fp_flag_global(abi::Flag::N)));
    // Z = (a == b)
    get_s_f32(f, rn);
    push_b(f);
    f.instruction(&W::F32Eq);
    f.instruction(&W::GlobalSet(abi::fp_flag_global(abi::Flag::Z)));
    // C = !(a < b)
    get_s_f32(f, rn);
    push_b(f);
    f.instruction(&W::F32Lt);
    f.instruction(&W::I32Eqz);
    f.instruction(&W::GlobalSet(abi::fp_flag_global(abi::Flag::C)));
    // V = unordered = (a != a) | (b != b)
    get_s_f32(f, rn);
    get_s_f32(f, rn);
    f.instruction(&W::F32Ne);
    push_b(f);
    push_b(f);
    f.instruction(&W::F32Ne);
    f.instruction(&W::I32Or);
    f.instruction(&W::GlobalSet(abi::fp_flag_global(abi::Flag::V)));
}

/// One VFP register <-> memory transfer. S = 4-byte raw i32; D = 8-byte raw i64.
fn emit_vfp_mem(f: &mut Function, reg: crate::ir::VfpReg, addr: &Value, load: bool, base: u32) {
    use crate::ir::VfpReg::*;
    match reg {
        S(n) => {
            if load {
                emit_addr(f, addr, base);
                if let Some(w) = watch_read_addr() {
                    // Read watchpoint over VFP single loads too (analog/float fields are
                    // read via `vldr`, invisible to the integer-load watch).
                    f.instruction(&W::LocalTee(L_T0));
                    f.instruction(&W::I32Load(mem_arg()));
                    emit_read_watch_check(f, w, base);
                    f.instruction(&W::GlobalSet(abi::vfp_s_global(n)));
                    return;
                }
                f.instruction(&W::I32Load(mem_arg()));
                f.instruction(&W::GlobalSet(abi::vfp_s_global(n)));
            } else {
                emit_addr(f, addr, base);
                f.instruction(&W::GlobalGet(abi::vfp_s_global(n)));
                f.instruction(&W::I32Store(mem_arg()));
            }
        }
        D(n) => {
            if load {
                emit_addr(f, addr, base);
                f.instruction(&W::I64Load(mem_arg()));
                set_d_bits(f, n);
            } else {
                emit_addr(f, addr, base);
                get_d_bits(f, n);
                f.instruction(&W::I64Store(mem_arg()));
            }
        }
    }
}

// --- NEON data-processing emission ----------------------------------------
//
// Every NEON operand is materialized as a wasm `v128`; a `D` register uses the low
// 64 bits. The register banks are handled by `neon_get`/`neon_set` (see the model
// in `abi`): the upper bank Q8..Q15 are `v128` globals accessed directly, the low
// bank is assembled from / scattered to the `s` globals. The widening family maps
// onto the wasm `extend_low` / `extmul_low` / `extadd_pairwise` primitives, which
// widen the low 64 bits of a `v128` - exactly the NEON long/wide semantics.

/// Push NEON register `reg` as a `v128` (a `D` lands in the low 64 bits).
fn neon_get(f: &mut Function, reg: crate::ir::NeonReg) {
    use crate::ir::NeonReg::*;
    match reg {
        Q(k) => {
            if (k as usize) >= abi::VFP_Q_HI_FIRST {
                f.instruction(&W::GlobalGet(abi::vfp_qhi_global(k)));
            } else {
                // Low bank: assemble the quad from its four aliased S registers.
                let s = 4 * k;
                f.instruction(&W::GlobalGet(abi::vfp_s_global(s)));
                f.instruction(&W::I32x4Splat);
                f.instruction(&W::GlobalGet(abi::vfp_s_global(s + 1)));
                f.instruction(&W::I32x4ReplaceLane(1));
                f.instruction(&W::GlobalGet(abi::vfp_s_global(s + 2)));
                f.instruction(&W::I32x4ReplaceLane(2));
                f.instruction(&W::GlobalGet(abi::vfp_s_global(s + 3)));
                f.instruction(&W::I32x4ReplaceLane(3));
            }
        }
        D(n) => {
            get_d_bits(f, n);
            f.instruction(&W::I64x2Splat);
        }
    }
}

/// Store the `v128` on the stack into NEON register `reg`. A `Q` writes all 128
/// bits; a `D` writes only the low 64 (leaving the sibling half of an upper-bank
/// quad intact). Uses the `L_V128A` scratch to fan a low-bank quad out to its S
/// halves (safe: any staged operands are already consumed by this point).
fn neon_set(f: &mut Function, reg: crate::ir::NeonReg) {
    use crate::ir::NeonReg::*;
    match reg {
        Q(k) => {
            if (k as usize) >= abi::VFP_Q_HI_FIRST {
                f.instruction(&W::GlobalSet(abi::vfp_qhi_global(k)));
            } else {
                let s = 4 * k;
                f.instruction(&W::LocalSet(L_V128A));
                for lane in 0..4u8 {
                    f.instruction(&W::LocalGet(L_V128A));
                    f.instruction(&W::I32x4ExtractLane(lane));
                    f.instruction(&W::GlobalSet(abi::vfp_s_global(s + lane)));
                }
            }
        }
        D(n) => {
            f.instruction(&W::I64x2ExtractLane(0));
            set_d_bits(f, n);
        }
    }
}

/// Lanewise `i{bits}x{n}.add`.
fn simd_add(bits: u8) -> W<'static> {
    match bits {
        8 => W::I8x16Add,
        16 => W::I16x8Add,
        32 => W::I32x4Add,
        64 => W::I64x2Add,
        _ => unreachable!("neon add width {bits}"),
    }
}

/// Lanewise `i{bits}x{n}.sub`.
fn simd_sub(bits: u8) -> W<'static> {
    match bits {
        8 => W::I8x16Sub,
        16 => W::I16x8Sub,
        32 => W::I32x4Sub,
        64 => W::I64x2Sub,
        _ => unreachable!("neon sub width {bits}"),
    }
}

/// Lanewise `i{bits}x{n}.mul` (undefined for 8-bit: wasm has no `i8x16.mul`).
fn simd_mul(bits: u8) -> W<'static> {
    match bits {
        16 => W::I16x8Mul,
        32 => W::I32x4Mul,
        64 => W::I64x2Mul,
        _ => unreachable!("neon mul width {bits}"),
    }
}

/// Lanewise floating-point `f{bits}x{n}.{add|sub|mul}` for a NEON `.f32`/`.f64`
/// vector op. NEON float SIMD is F32 (and F16, which wasm SIMD has no lanewise
/// arithmetic for - filtered out at lift).
fn simd_fadd(bits: u8) -> W<'static> {
    match bits {
        32 => W::F32x4Add,
        64 => W::F64x2Add,
        _ => unreachable!("neon fadd width {bits}"),
    }
}
fn simd_fsub(bits: u8) -> W<'static> {
    match bits {
        32 => W::F32x4Sub,
        64 => W::F64x2Sub,
        _ => unreachable!("neon fsub width {bits}"),
    }
}
fn simd_fmul(bits: u8) -> W<'static> {
    match bits {
        32 => W::F32x4Mul,
        64 => W::F64x2Mul,
        _ => unreachable!("neon fmul width {bits}"),
    }
}
fn simd_fmax(bits: u8) -> W<'static> {
    match bits {
        32 => W::F32x4Max,
        64 => W::F64x2Max,
        _ => unreachable!("neon fmax width {bits}"),
    }
}
fn simd_fmin(bits: u8) -> W<'static> {
    match bits {
        32 => W::F32x4Min,
        64 => W::F64x2Min,
        _ => unreachable!("neon fmin width {bits}"),
    }
}

/// Lanewise signed/unsigned min.
fn simd_min(bits: u8, signed: bool) -> W<'static> {
    match (bits, signed) {
        (8, true) => W::I8x16MinS,
        (8, false) => W::I8x16MinU,
        (16, true) => W::I16x8MinS,
        (16, false) => W::I16x8MinU,
        (32, true) => W::I32x4MinS,
        (32, false) => W::I32x4MinU,
        _ => unreachable!("neon min width {bits}"),
    }
}

/// Lanewise signed/unsigned max.
fn simd_max(bits: u8, signed: bool) -> W<'static> {
    match (bits, signed) {
        (8, true) => W::I8x16MaxS,
        (8, false) => W::I8x16MaxU,
        (16, true) => W::I16x8MaxS,
        (16, false) => W::I16x8MaxU,
        (32, true) => W::I32x4MaxS,
        (32, false) => W::I32x4MaxU,
        _ => unreachable!("neon max width {bits}"),
    }
}

/// Lanewise integer abs / neg.
fn simd_abs(bits: u8) -> W<'static> {
    match bits {
        8 => W::I8x16Abs,
        16 => W::I16x8Abs,
        32 => W::I32x4Abs,
        64 => W::I64x2Abs,
        _ => unreachable!("neon abs width {bits}"),
    }
}
fn simd_neg(bits: u8) -> W<'static> {
    match bits {
        8 => W::I8x16Neg,
        16 => W::I16x8Neg,
        32 => W::I32x4Neg,
        64 => W::I64x2Neg,
        _ => unreachable!("neon neg width {bits}"),
    }
}

/// Widen the low half of a `v128` from `bits`-wide elements to `2*bits` (signed or
/// unsigned zero/sign extension).
fn simd_extend_low(bits: u8, signed: bool) -> W<'static> {
    match (bits, signed) {
        (8, true) => W::I16x8ExtendLowI8x16S,
        (8, false) => W::I16x8ExtendLowI8x16U,
        (16, true) => W::I32x4ExtendLowI16x8S,
        (16, false) => W::I32x4ExtendLowI16x8U,
        (32, true) => W::I64x2ExtendLowI32x4S,
        (32, false) => W::I64x2ExtendLowI32x4U,
        _ => unreachable!("neon extend width {bits}"),
    }
}

/// Widen-and-multiply the low halves of two `v128`s (`bits` -> `2*bits`).
fn simd_extmul_low(bits: u8, signed: bool) -> W<'static> {
    match (bits, signed) {
        (8, true) => W::I16x8ExtMulLowI8x16S,
        (8, false) => W::I16x8ExtMulLowI8x16U,
        (16, true) => W::I32x4ExtMulLowI16x8S,
        (16, false) => W::I32x4ExtMulLowI16x8U,
        (32, true) => W::I64x2ExtMulLowI32x4S,
        (32, false) => W::I64x2ExtMulLowI32x4U,
        _ => unreachable!("neon extmul width {bits}"),
    }
}

/// Pairwise-add adjacent `bits`-wide elements, widening to `2*bits` (8->16 or
/// 16->32 only; wasm has no 32->64 form).
fn simd_extadd_pairwise(bits: u8, signed: bool) -> W<'static> {
    match (bits, signed) {
        (8, true) => W::I16x8ExtAddPairwiseI8x16S,
        (8, false) => W::I16x8ExtAddPairwiseI8x16U,
        (16, true) => W::I32x4ExtAddPairwiseI16x8S,
        (16, false) => W::I32x4ExtAddPairwiseI16x8U,
        _ => unreachable!("neon extadd-pairwise width {bits}"),
    }
}

/// Lanewise integer equality (`i{bits}x{n}.eq`), producing all-ones on true.
fn simd_cmp_eq(bits: u8) -> W<'static> {
    match bits {
        8 => W::I8x16Eq,
        16 => W::I16x8Eq,
        32 => W::I32x4Eq,
        64 => W::I64x2Eq,
        _ => unreachable!("neon cmpeq width {bits}"),
    }
}

/// Lanewise integer greater-than (`i{bits}x{n}.gt_s`/`gt_u`).
fn simd_cmp_gt(bits: u8, signed: bool) -> W<'static> {
    match (bits, signed) {
        (8, true) => W::I8x16GtS,
        (8, false) => W::I8x16GtU,
        (16, true) => W::I16x8GtS,
        (16, false) => W::I16x8GtU,
        (32, true) => W::I32x4GtS,
        (32, false) => W::I32x4GtU,
        (64, true) => W::I64x2GtS,
        _ => unreachable!("neon cmpgt width {bits} signed {signed}"),
    }
}

/// Lanewise integer greater-or-equal (`i{bits}x{n}.ge_s`/`ge_u`).
fn simd_cmp_ge(bits: u8, signed: bool) -> W<'static> {
    match (bits, signed) {
        (8, true) => W::I8x16GeS,
        (8, false) => W::I8x16GeU,
        (16, true) => W::I16x8GeS,
        (16, false) => W::I16x8GeU,
        (32, true) => W::I32x4GeS,
        (32, false) => W::I32x4GeU,
        (64, true) => W::I64x2GeS,
        _ => unreachable!("neon cmpge width {bits} signed {signed}"),
    }
}

/// Lanewise `i{bits}x{n}.shl` (the shift count is an `i32` operand, taken modulo the lane width).
fn simd_shl(bits: u8) -> W<'static> {
    match bits {
        8 => W::I8x16Shl,
        16 => W::I16x8Shl,
        32 => W::I32x4Shl,
        64 => W::I64x2Shl,
        _ => unreachable!("neon shl width {bits}"),
    }
}

/// Lanewise `i{bits}x{n}.shr_s`/`shr_u` (the shift count is an `i32` operand, modulo the lane width).
fn simd_shr(bits: u8, signed: bool) -> W<'static> {
    match (bits, signed) {
        (8, true) => W::I8x16ShrS,
        (8, false) => W::I8x16ShrU,
        (16, true) => W::I16x8ShrS,
        (16, false) => W::I16x8ShrU,
        (32, true) => W::I32x4ShrS,
        (32, false) => W::I32x4ShrU,
        (64, true) => W::I64x2ShrS,
        (64, false) => W::I64x2ShrU,
        _ => unreachable!("neon shr width {bits}"),
    }
}

/// A v128 constant with `val`'s low `bits` bits replicated into every `bits`-wide lane, for the
/// per-lane insert masks of `vsli`/`vsri`.
fn splat_lane_mask(bits: u8, val: u64) -> i128 {
    let nbytes = (bits / 8) as usize;
    let v = val.to_le_bytes();
    let mut bytes = [0u8; 16];
    let mut off = 0;
    while off < 16 {
        bytes[off..off + nbytes].copy_from_slice(&v[0..nbytes]);
        off += nbytes;
    }
    i128::from_le_bytes(bytes)
}

/// Emit an immediate NEON shift (`vshr`/`vsra`/`vshl`/`vsli`/`vsri`). wasm SIMD takes the shift
/// count modulo the lane width, so a right shift by the full element width (a valid NEON encoding)
/// is special-cased: a logical one yields zero, an arithmetic one is clamped to `bits-1` (which
/// already produces the sign broadcast). Left shifts are always in `0..bits-1`.
fn emit_shift_imm(
    f: &mut Function,
    op: crate::ir::NeonShift,
    ty: crate::ir::NeonType,
    dst: crate::ir::NeonReg,
    src: crate::ir::NeonReg,
    amount: u8,
) {
    use crate::ir::NeonShift::*;
    let bits = ty.bits;
    let amt = amount as u32;
    // Push `src >> amt` (arithmetic iff `ty.signed`), handling the shift-out-everything case.
    let push_shifted_src = |f: &mut Function| {
        if !ty.signed && amt >= bits as u32 {
            f.instruction(&W::V128Const(0)); // logical shift by >= width clears the lane
        } else {
            neon_get(f, src);
            f.instruction(&W::I32Const(amt.min(bits as u32 - 1) as i32));
            f.instruction(&simd_shr(bits, ty.signed));
        }
    };
    match op {
        Shr => {
            push_shifted_src(f);
            neon_set(f, dst);
        }
        Sra => {
            neon_get(f, dst);
            push_shifted_src(f);
            f.instruction(&simd_add(bits));
            neon_set(f, dst);
        }
        Shl => {
            neon_get(f, src);
            f.instruction(&W::I32Const(amt as i32));
            f.instruction(&simd_shl(bits));
            neon_set(f, dst);
        }
        Sli => {
            // dst = (dst & lowmask) | (src << amt); lowmask keeps the low `amt` bits of dst.
            let lowmask = if amt == 0 { 0 } else { ((1u128 << amt) - 1) as u64 };
            neon_get(f, dst);
            f.instruction(&W::V128Const(splat_lane_mask(bits, lowmask)));
            f.instruction(&W::V128And);
            neon_get(f, src);
            f.instruction(&W::I32Const(amt as i32));
            f.instruction(&simd_shl(bits));
            f.instruction(&W::V128Or);
            neon_set(f, dst);
        }
        Sri => {
            // dst = (dst & highmask) | (src >>u amt); highmask keeps the high `amt` bits of dst.
            let highmask = if amt >= bits as u32 {
                u64::MAX
            } else {
                !(((1u128 << (bits as u32 - amt)) - 1) as u64)
            };
            neon_get(f, dst);
            f.instruction(&W::V128Const(splat_lane_mask(bits, highmask)));
            f.instruction(&W::V128And);
            if amt >= bits as u32 {
                f.instruction(&W::V128Const(0));
            } else {
                neon_get(f, src);
                f.instruction(&W::I32Const(amt as i32));
                f.instruction(&simd_shr(bits, false));
            }
            f.instruction(&W::V128Or);
            neon_set(f, dst);
        }
    }
}

fn emit_neon(f: &mut Function, op: &crate::ir::NeonStmt, base: u32) {
    use crate::ir::NeonStmt::*;
    match op {
        Bin { op: bop, ty, dst, a, b } => {
            use crate::ir::NeonBin::*;
            match bop {
                Add | Sub | Mul => {
                    neon_get(f, *a);
                    neon_get(f, *b);
                    f.instruction(&match (bop, ty.float) {
                        (Add, true) => simd_fadd(ty.bits),
                        (Sub, true) => simd_fsub(ty.bits),
                        (_, true) => simd_fmul(ty.bits),
                        (Add, false) => simd_add(ty.bits),
                        (Sub, false) => simd_sub(ty.bits),
                        (_, false) => simd_mul(ty.bits),
                    });
                    neon_set(f, *dst);
                }
                Max | Min => {
                    neon_get(f, *a);
                    neon_get(f, *b);
                    f.instruction(&match (matches!(bop, Max), ty.float) {
                        (true, true) => simd_fmax(ty.bits),
                        (false, true) => simd_fmin(ty.bits),
                        (true, false) => simd_max(ty.bits, ty.signed),
                        (false, false) => simd_min(ty.bits, ty.signed),
                    });
                    neon_set(f, *dst);
                }
                Abd => {
                    // |a - b| = max(a, b) - min(a, b), avoiding the overflow of a
                    // straight sub-then-abs. Both operands are read twice.
                    neon_get(f, *a);
                    f.instruction(&W::LocalSet(L_V128A));
                    neon_get(f, *b);
                    f.instruction(&W::LocalSet(L_V128B));
                    f.instruction(&W::LocalGet(L_V128A));
                    f.instruction(&W::LocalGet(L_V128B));
                    f.instruction(&simd_max(ty.bits, ty.signed));
                    f.instruction(&W::LocalGet(L_V128A));
                    f.instruction(&W::LocalGet(L_V128B));
                    f.instruction(&simd_min(ty.bits, ty.signed));
                    f.instruction(&simd_sub(ty.bits));
                    neon_set(f, *dst);
                }
            }
        }
        MulAcc { ty, dst, a, b, sub } => {
            neon_get(f, *dst);
            neon_get(f, *a);
            neon_get(f, *b);
            f.instruction(&if ty.float { simd_fmul(ty.bits) } else { simd_mul(ty.bits) });
            f.instruction(&match (*sub, ty.float) {
                (true, true) => simd_fsub(ty.bits),
                (false, true) => simd_fadd(ty.bits),
                (true, false) => simd_sub(ty.bits),
                (false, false) => simd_add(ty.bits),
            });
            neon_set(f, *dst);
        }
        PairAdd { ty, dst, a, b } => {
            // No wasm horizontal add: gather the even and odd source elements of the
            // (a : b) concatenation with two shuffles, then add. `a` is source bytes
            // 0..16, `b` is 16..32.
            let ebytes = (ty.bits / 8) as usize;
            let cnt = 8 / ebytes; // elements per D register
            let mut xmask = [0u8; 16];
            let mut ymask = [0u8; 16];
            for k in 0..cnt {
                let (src_base, within) =
                    if k < cnt / 2 { (0usize, k) } else { (16usize, k - cnt / 2) };
                let even = src_base + (2 * within) * ebytes;
                let odd = src_base + (2 * within + 1) * ebytes;
                for j in 0..ebytes {
                    xmask[k * ebytes + j] = (even + j) as u8;
                    ymask[k * ebytes + j] = (odd + j) as u8;
                }
            }
            neon_get(f, *a);
            f.instruction(&W::LocalSet(L_V128A));
            neon_get(f, *b);
            f.instruction(&W::LocalSet(L_V128B));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128B));
            f.instruction(&W::I8x16Shuffle(xmask));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128B));
            f.instruction(&W::I8x16Shuffle(ymask));
            // `vpadd.f32` adds the same gathered even/odd lanes with float arithmetic.
            f.instruction(&if ty.float { simd_fadd(ty.bits) } else { simd_add(ty.bits) });
            neon_set(f, *dst);
        }
        Widen { ty, dst, a } => {
            neon_get(f, *a);
            f.instruction(&simd_extend_low(ty.bits, ty.signed));
            neon_set(f, *dst);
        }
        WideAddSub { sub, wide, ty, dst, a, b } => {
            // Wide form: `a` is already a Q of 2*bits elements; long form: widen `a`.
            neon_get(f, *a);
            if !wide {
                f.instruction(&simd_extend_low(ty.bits, ty.signed));
            }
            neon_get(f, *b);
            f.instruction(&simd_extend_low(ty.bits, ty.signed));
            f.instruction(&if *sub { simd_sub(ty.bits * 2) } else { simd_add(ty.bits * 2) });
            neon_set(f, *dst);
        }
        WideMul { acc, sub, ty, dst, a, b } => {
            if *acc {
                neon_get(f, *dst);
            }
            neon_get(f, *a);
            neon_get(f, *b);
            f.instruction(&simd_extmul_low(ty.bits, ty.signed));
            if *acc {
                f.instruction(&if *sub { simd_sub(ty.bits * 2) } else { simd_add(ty.bits * 2) });
            }
            neon_set(f, *dst);
        }
        WideAbd { acc, ty, dst, a, b } => {
            let w = ty.bits * 2;
            neon_get(f, *a);
            f.instruction(&simd_extend_low(ty.bits, ty.signed));
            f.instruction(&W::LocalSet(L_V128A));
            neon_get(f, *b);
            f.instruction(&simd_extend_low(ty.bits, ty.signed));
            f.instruction(&W::LocalSet(L_V128B));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128B));
            f.instruction(&simd_max(w, ty.signed));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128B));
            f.instruction(&simd_min(w, ty.signed));
            f.instruction(&simd_sub(w));
            if *acc {
                f.instruction(&W::LocalSet(L_V128A));
                neon_get(f, *dst);
                f.instruction(&W::LocalGet(L_V128A));
                f.instruction(&simd_add(w));
            }
            neon_set(f, *dst);
        }
        PairLong { acc, ty, dst, a } => {
            neon_get(f, *a);
            f.instruction(&simd_extadd_pairwise(ty.bits, ty.signed));
            if *acc {
                f.instruction(&W::LocalSet(L_V128A));
                neon_get(f, *dst);
                f.instruction(&W::LocalGet(L_V128A));
                f.instruction(&simd_add(ty.bits * 2));
            }
            neon_set(f, *dst);
        }
        Unary { neg, ty, dst, a } => {
            neon_get(f, *a);
            f.instruction(&if ty.float {
                if *neg { W::F32x4Neg } else { W::F32x4Abs }
            } else if *neg {
                simd_neg(ty.bits)
            } else {
                simd_abs(ty.bits)
            });
            neon_set(f, *dst);
        }
        MovImm { ty, dst, imm } => {
            let mut bytes = [0u8; 16];
            match ty.bits {
                8 => bytes = [*imm as u8; 16],
                16 => {
                    let h = (*imm as u16).to_le_bytes();
                    for i in 0..8 {
                        bytes[2 * i] = h[0];
                        bytes[2 * i + 1] = h[1];
                    }
                }
                32 => {
                    let w = imm.to_le_bytes();
                    for i in 0..4 {
                        bytes[4 * i..4 * i + 4].copy_from_slice(&w);
                    }
                }
                _ => unreachable!("neon vmov.i width {}", ty.bits),
            }
            f.instruction(&W::V128Const(i128::from_le_bytes(bytes)));
            neon_set(f, *dst);
        }
        Bitwise { op, dst, a, b } => {
            use crate::ir::NeonBitwise::*;
            match op {
                And | Or | Xor => {
                    neon_get(f, *a);
                    neon_get(f, *b);
                    f.instruction(&match op {
                        And => W::V128And,
                        Or => W::V128Or,
                        _ => W::V128Xor,
                    });
                }
                // `a & ~b`.
                Bic => {
                    neon_get(f, *a);
                    neon_get(f, *b);
                    f.instruction(&W::V128AndNot);
                }
                // `a | ~b`.
                Orn => {
                    neon_get(f, *a);
                    neon_get(f, *b);
                    f.instruction(&W::V128Not);
                    f.instruction(&W::V128Or);
                }
                // The insert/select forms via `v128.bitselect(v1, v2, c) =
                // (v1 & c) | (v2 & ~c)`, each with its own operand roles - `dst`'s
                // current value is one of the inputs, read before it is rewritten.
                Bsl => {
                    // dst = (a & dst) | (b & ~dst)
                    neon_get(f, *a);
                    neon_get(f, *b);
                    neon_get(f, *dst);
                    f.instruction(&W::V128Bitselect);
                }
                Bit => {
                    // dst = (a & b) | (dst & ~b)
                    neon_get(f, *a);
                    neon_get(f, *dst);
                    neon_get(f, *b);
                    f.instruction(&W::V128Bitselect);
                }
                Bif => {
                    // dst = (dst & b) | (a & ~b)
                    neon_get(f, *dst);
                    neon_get(f, *a);
                    neon_get(f, *b);
                    f.instruction(&W::V128Bitselect);
                }
            }
            neon_set(f, *dst);
        }
        DupCore { ty, dst, rt } => {
            // Broadcast the low `ty.bits` bits of a core register to every lane.
            f.instruction(&W::GlobalGet(abi::reg_global(*rt as usize)));
            f.instruction(&match ty.bits {
                8 => W::I8x16Splat,
                16 => W::I16x8Splat,
                32 => W::I32x4Splat,
                _ => unreachable!("vdup core width {}", ty.bits),
            });
            neon_set(f, *dst);
        }
        DupLane { esize, dst, src, lane } => {
            // Broadcast one lane of source `Dsrc` to every lane. `neon_get(D)` splats
            // the 64-bit doubleword across both halves of the v128, so the wanted
            // element sits at `lane` within the low half; extract it, then splat.
            neon_get(f, crate::ir::NeonReg::D(*src));
            f.instruction(&match esize {
                8 => W::I8x16ExtractLaneU(*lane),
                16 => W::I16x8ExtractLaneU(*lane),
                32 => W::I32x4ExtractLane(*lane),
                _ => unreachable!("vdup lane width {esize}"),
            });
            f.instruction(&match esize {
                8 => W::I8x16Splat,
                16 => W::I16x8Splat,
                32 => W::I32x4Splat,
                _ => unreachable!("vdup lane width {esize}"),
            });
            neon_set(f, *dst);
        }
        MovLane { to_core, bits, signed, dreg, lane, rt } => {
            // `neon_get(D)` splats the 64-bit doubleword across the v128, so lane
            // `lane` (< lanes-per-D) sits in the low half - exactly where `ReplaceLane`
            // writes and `neon_set(D)` (which keeps the low 64 bits) reads back.
            let d = crate::ir::NeonReg::D(*dreg);
            if *to_core {
                neon_get(f, d);
                f.instruction(&match (*bits, *signed) {
                    (8, false) => W::I8x16ExtractLaneU(*lane),
                    (8, true) => W::I8x16ExtractLaneS(*lane),
                    (16, false) => W::I16x8ExtractLaneU(*lane),
                    (16, true) => W::I16x8ExtractLaneS(*lane),
                    (32, _) => W::I32x4ExtractLane(*lane),
                    _ => unreachable!("vmov lane->core width {bits}"),
                });
                f.instruction(&W::GlobalSet(abi::reg_global(*rt as usize)));
            } else {
                neon_get(f, d);
                f.instruction(&W::GlobalGet(abi::reg_global(*rt as usize)));
                f.instruction(&match bits {
                    8 => W::I8x16ReplaceLane(*lane),
                    16 => W::I16x8ReplaceLane(*lane),
                    32 => W::I32x4ReplaceLane(*lane),
                    _ => unreachable!("vmov core->lane width {bits}"),
                });
                neon_set(f, d);
            }
        }
        MovImm64 { dst, val } => {
            // `vmov.i64`: every doubleword lane gets the full 64-bit pattern.
            let lo = val.to_le_bytes();
            let mut bytes = [0u8; 16];
            bytes[0..8].copy_from_slice(&lo);
            bytes[8..16].copy_from_slice(&lo);
            f.instruction(&W::V128Const(i128::from_le_bytes(bytes)));
            neon_set(f, *dst);
        }
        ShiftImm { op, ty, dst, src, amount } => {
            emit_shift_imm(f, *op, *ty, *dst, *src, *amount);
        }
        Ext { dst, a, b, byte_off } => {
            // The destination byte width picks which shuffle: a `Q` extract is a full
            // 16-byte window into `a:b`; a `D` extract is an 8-byte window (the source
            // doublewords sit in the low half of each staged v128, so `b`'s bytes are at
            // shuffle indices 16..24). Unused upper lanes of a `D` result are dropped by
            // `neon_set`.
            let q = matches!(dst, crate::ir::NeonReg::Q(_));
            let width: u8 = if q { 16 } else { 8 };
            let mut mask = [0u8; 16];
            for i in 0..width {
                let src_byte = *byte_off + i;
                mask[i as usize] = if src_byte < width {
                    src_byte // from `a`
                } else {
                    16 + (src_byte - width) // from `b`
                };
            }
            neon_get(f, *a);
            neon_get(f, *b);
            f.instruction(&W::I8x16Shuffle(mask));
            neon_set(f, *dst);
        }
        CvtFloatInt { to_int, signed, dst, src } => {
            neon_get(f, *src);
            f.instruction(&match (to_int, signed) {
                // Float->int rounds toward zero; wasm's saturating trunc matches NEON's
                // out-of-range clamping (NEON VCVT saturates rather than wrapping).
                (true, true) => W::I32x4TruncSatF32x4S,
                (true, false) => W::I32x4TruncSatF32x4U,
                (false, true) => W::F32x4ConvertI32x4S,
                (false, false) => W::F32x4ConvertI32x4U,
            });
            neon_set(f, *dst);
        }
        Cmp { op, ty, dst, a, b } => {
            use crate::ir::NeonCmp::*;
            neon_get(f, *a);
            neon_get(f, *b);
            f.instruction(&match (op, ty.float) {
                (Eq, true) => W::F32x4Eq,
                (Gt, true) => W::F32x4Gt,
                (Ge, true) => W::F32x4Ge,
                (Eq, false) => simd_cmp_eq(ty.bits),
                (Gt, false) => simd_cmp_gt(ty.bits, ty.signed),
                (Ge, false) => simd_cmp_ge(ty.bits, ty.signed),
                // Le/Lt reach the IR only as compare-against-zero (`CmpZero`), never the
                // register-register `Cmp`, since the assembler folds them into Ge/Gt.
                (Le, _) | (Lt, _) => unreachable!("vcle/vclt only exist against #0"),
            });
            neon_set(f, *dst);
        }
        CmpZero { op, ty, dst, src } => {
            use crate::ir::NeonCmp::*;
            // `src <rel> 0`. Le/Lt reuse the Ge/Gt primitives with the operands swapped
            // (`src <= 0` is `0 >= src`, `src < 0` is `0 > src`), so no le/lt op is needed.
            let zero = W::V128Const(0);
            let swap = matches!(op, Le | Lt);
            if swap {
                f.instruction(&zero);
                neon_get(f, *src);
            } else {
                neon_get(f, *src);
                f.instruction(&zero);
            }
            f.instruction(&match (op, ty.float) {
                (Eq, true) => W::F32x4Eq,
                (Gt, true) | (Lt, true) => W::F32x4Gt,
                (Ge, true) | (Le, true) => W::F32x4Ge,
                (Eq, false) => simd_cmp_eq(ty.bits),
                (Gt, false) | (Lt, false) => simd_cmp_gt(ty.bits, ty.signed),
                (Ge, false) | (Le, false) => simd_cmp_ge(ty.bits, ty.signed),
            });
            neon_set(f, *dst);
        }
        CmpAbs { ge, dst, a, b } => {
            // `|a| >= |b|` / `|a| > |b|`, f32 lanes.
            neon_get(f, *a);
            f.instruction(&W::F32x4Abs);
            neon_get(f, *b);
            f.instruction(&W::F32x4Abs);
            f.instruction(&if *ge { W::F32x4Ge } else { W::F32x4Gt });
            neon_set(f, *dst);
        }
        PairMinMax { min, dst, a, b } => {
            // f32 pairwise max/min, doubleword only: gather the even/odd f32 lanes of the
            // concatenation `a : b` (a is bytes 0..16, b is 16..32) with two shuffles, then
            // reduce. Mirrors `PairAdd` for the two-lane f32 case.
            let mut xmask = [0u8; 16];
            let mut ymask = [0u8; 16];
            // Two D registers -> two source elements each. Output lane 0/1 <- pairs of `a`,
            // output lane... only the low two lanes are meaningful for a D result.
            for k in 0..2usize {
                let (src_base, within) = if k < 1 { (0usize, k) } else { (16usize, k - 1) };
                let even = src_base + (2 * within) * 4;
                let odd = src_base + (2 * within + 1) * 4;
                for j in 0..4 {
                    xmask[k * 4 + j] = (even + j) as u8;
                    ymask[k * 4 + j] = (odd + j) as u8;
                }
            }
            neon_get(f, *a);
            f.instruction(&W::LocalSet(L_V128A));
            neon_get(f, *b);
            f.instruction(&W::LocalSet(L_V128B));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128B));
            f.instruction(&W::I8x16Shuffle(xmask));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128B));
            f.instruction(&W::I8x16Shuffle(ymask));
            f.instruction(&if *min { W::F32x4Min } else { W::F32x4Max });
            neon_set(f, *dst);
        }
        Rev { esize, container, dst, src } => {
            // Reverse the `esize`-bit elements within each `container`-bit group: a pure byte
            // permutation of the source with itself.
            let ebytes = (*esize / 8) as usize;
            let cbytes = (*container / 8) as usize;
            let nel = cbytes / ebytes;
            let mut mask = [0u8; 16];
            for i in 0..16usize {
                let c = i / cbytes;
                let p = i % cbytes;
                let e = p / ebytes;
                let j = p % ebytes;
                mask[i] = (c * cbytes + (nel - 1 - e) * ebytes + j) as u8;
            }
            neon_get(f, *src);
            f.instruction(&W::LocalSet(L_V128A));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::I8x16Shuffle(mask));
            neon_set(f, *dst);
        }
        ShiftReg { sat: _, ty, dst, src, amt } => {
            // Per-lane variable shift-left by a signed amount (`vshl` register form): each lane of
            // `src` is shifted by the signed low byte of the matching `amt` lane; a negative amount
            // shifts right (arithmetic when the type is signed, else logical). wasm SIMD has no
            // vector variable shift, so extract, shift, and reinsert each lane. `sat` (VQSHL) would
            // additionally saturate the left-shift overflow; the unsaturated VSHL is emitted here and
            // the saturating form is gated to lift as unsupported in `neon_emittable`.
            let w = ty.bits as i32;
            let signed = ty.signed;
            let lanes = 16 / (ty.bits as usize / 8);
            neon_get(f, *src);
            f.instruction(&W::LocalSet(L_V128A));
            neon_get(f, *amt);
            f.instruction(&W::LocalSet(L_V128B));
            neon_get(f, *src);
            f.instruction(&W::LocalSet(L_V128C)); // result accumulator, lanes overwritten below
            let extract_amt = |k: u8| -> W<'static> {
                match ty.bits {
                    8 => W::I8x16ExtractLaneU(k),
                    16 => W::I16x8ExtractLaneU(k),
                    _ => W::I32x4ExtractLane(k),
                }
            };
            let extract_src = |k: u8| -> W<'static> {
                match (ty.bits, signed) {
                    (8, true) => W::I8x16ExtractLaneS(k),
                    (8, false) => W::I8x16ExtractLaneU(k),
                    (16, true) => W::I16x8ExtractLaneS(k),
                    (16, false) => W::I16x8ExtractLaneU(k),
                    _ => W::I32x4ExtractLane(k),
                }
            };
            let replace = |k: u8| -> W<'static> {
                match ty.bits {
                    8 => W::I8x16ReplaceLane(k),
                    16 => W::I16x8ReplaceLane(k),
                    _ => W::I32x4ReplaceLane(k),
                }
            };
            for k in 0..lanes as u8 {
                // s = sign-extended low byte of amt lane k.
                f.instruction(&W::LocalGet(L_V128B));
                f.instruction(&extract_amt(k));
                f.instruction(&W::I32Const(24));
                f.instruction(&W::I32Shl);
                f.instruction(&W::I32Const(24));
                f.instruction(&W::I32ShrS);
                f.instruction(&W::LocalSet(L_T0)); // s
                // x = extended src lane k.
                f.instruction(&W::LocalGet(L_V128A));
                f.instruction(&extract_src(k));
                f.instruction(&W::LocalSet(L_T1)); // x
                // shifted = (s >= 0) ? (s >= w ? 0 : x << s)
                //                    : (r=-s >= w ? sign/0 : x >> r)
                f.instruction(&W::LocalGet(L_T0));
                f.instruction(&W::I32Const(0));
                f.instruction(&W::I32GeS);
                f.instruction(&W::If(BlockType::Result(ValType::I32)));
                {
                    f.instruction(&W::LocalGet(L_T0));
                    f.instruction(&W::I32Const(w));
                    f.instruction(&W::I32GeS);
                    f.instruction(&W::If(BlockType::Result(ValType::I32)));
                    f.instruction(&W::I32Const(0));
                    f.instruction(&W::Else);
                    f.instruction(&W::LocalGet(L_T1));
                    f.instruction(&W::LocalGet(L_T0));
                    f.instruction(&W::I32Shl);
                    f.instruction(&W::End);
                }
                f.instruction(&W::Else);
                {
                    f.instruction(&W::I32Const(0));
                    f.instruction(&W::LocalGet(L_T0));
                    f.instruction(&W::I32Sub);
                    f.instruction(&W::LocalSet(L_T2)); // r = -s
                    f.instruction(&W::LocalGet(L_T2));
                    f.instruction(&W::I32Const(w));
                    f.instruction(&W::I32GeS);
                    f.instruction(&W::If(BlockType::Result(ValType::I32)));
                    if signed {
                        // arithmetic shift by (w-1) replicates the sign bit across the lane.
                        f.instruction(&W::LocalGet(L_T1));
                        f.instruction(&W::I32Const(w - 1));
                        f.instruction(&W::I32ShrS);
                    } else {
                        f.instruction(&W::I32Const(0));
                    }
                    f.instruction(&W::Else);
                    f.instruction(&W::LocalGet(L_T1));
                    f.instruction(&W::LocalGet(L_T2));
                    f.instruction(&if signed { W::I32ShrS } else { W::I32ShrU });
                    f.instruction(&W::End);
                }
                f.instruction(&W::End);
                f.instruction(&W::LocalSet(L_T1)); // shifted result
                f.instruction(&W::LocalGet(L_V128C));
                f.instruction(&W::LocalGet(L_T1));
                f.instruction(&replace(k));
                f.instruction(&W::LocalSet(L_V128C));
            }
            f.instruction(&W::LocalGet(L_V128C));
            neon_set(f, *dst);
        }
        Test { ty, dst, a, b } => {
            // `(a AND b) != 0` per lane: AND, compare-equal-zero (all-ones where zero), then invert.
            neon_get(f, *a);
            neon_get(f, *b);
            f.instruction(&W::V128And);
            f.instruction(&W::V128Const(0));
            f.instruction(&simd_cmp_eq(ty.bits));
            f.instruction(&W::V128Not);
            neon_set(f, *dst);
        }
        Narrow { esize, dst, src } => {
            // Truncate each `2*esize`-bit source element to its low `esize` bits, packing the results
            // into the low 8 bytes (the `D` result). A pure byte gather from the source.
            let rbytes = (*esize / 8) as usize;
            let n = 8 / rbytes; // result elements (fill the low 64 bits)
            let mut mask = [0u8; 16];
            for i in 0..n {
                for j in 0..rbytes {
                    mask[i * rbytes + j] = (i * 2 * rbytes + j) as u8;
                }
            }
            neon_get(f, *src);
            f.instruction(&W::LocalSet(L_V128A));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::I8x16Shuffle(mask));
            neon_set(f, *dst);
        }
        MulScalar { ty, dst, a, src, lane, acc, sub } => {
            // dst = [dst -/+] a * broadcast(D[src].lane). Push the accumulator first (if any) so it
            // sits under the product for the trailing add/sub.
            if *acc {
                neon_get(f, *dst);
            }
            neon_get(f, *a);
            // Broadcast the scalar lane: `neon_get(D)` splats the doubleword, so the wanted lane
            // sits at index `lane` of the low half; extract it and splat across the vector.
            neon_get(f, crate::ir::NeonReg::D(*src));
            match ty.bits {
                16 => {
                    f.instruction(&W::I16x8ExtractLaneU(*lane));
                    f.instruction(&W::I16x8Splat);
                }
                _ => {
                    f.instruction(&W::I32x4ExtractLane(*lane));
                    f.instruction(&W::I32x4Splat);
                }
            }
            f.instruction(&if ty.float { simd_fmul(ty.bits) } else { simd_mul(ty.bits) });
            if *acc {
                f.instruction(&match (*sub, ty.float) {
                    (true, true) => simd_fsub(ty.bits),
                    (false, true) => simd_fadd(ty.bits),
                    (true, false) => simd_sub(ty.bits),
                    (false, false) => simd_add(ty.bits),
                });
            }
            neon_set(f, *dst);
        }
        RecipEstimate { sqrt, dst, src } => {
            // Full-precision reciprocal: 1 / (sqrt?)(src). f32x4 only.
            f.instruction(&f32x4_splat_const(1.0));
            neon_get(f, *src);
            if *sqrt {
                f.instruction(&W::F32x4Sqrt);
            }
            f.instruction(&W::F32x4Div);
            neon_set(f, *dst);
        }
        RecipStep { sqrt, dst, a, b } => {
            // vrecps: 2 - a*b. vrsqrts: (3 - a*b) / 2 (multiply by 0.5, exact). Non-fused.
            f.instruction(&f32x4_splat_const(if *sqrt { 3.0 } else { 2.0 }));
            neon_get(f, *a);
            neon_get(f, *b);
            f.instruction(&W::F32x4Mul);
            f.instruction(&W::F32x4Sub);
            if *sqrt {
                f.instruction(&f32x4_splat_const(0.5));
                f.instruction(&W::F32x4Mul);
            }
            neon_set(f, *dst);
        }
        Permute { op, esize, a, b } => {
            // Two shuffles of the concatenation `a:b` produce the two result registers. Both results
            // depend on both inputs, so stage the inputs in scratch, stash the first result (a) in
            // `L_V128C`, write b, then write a (a plain low-bank `neon_set` clobbers `L_V128A`).
            let q = matches!(a, crate::ir::NeonReg::Q(_));
            let (mask_a, mask_b) = permute_masks(*op, *esize, q);
            neon_get(f, *a);
            f.instruction(&W::LocalSet(L_V128A));
            neon_get(f, *b);
            f.instruction(&W::LocalSet(L_V128B));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128B));
            f.instruction(&W::I8x16Shuffle(mask_a));
            f.instruction(&W::LocalSet(L_V128C));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128B));
            f.instruction(&W::I8x16Shuffle(mask_b));
            neon_set(f, *b);
            f.instruction(&W::LocalGet(L_V128C));
            neon_set(f, *a);
        }
        ElemMem { d, esize, lane, addr, load } => emit_elem_mem(f, *d, *esize, *lane, addr, *load, base),
    }
}

/// A `v128` constant with `v` broadcast into all four f32 lanes.
fn f32x4_splat_const(v: f32) -> W<'static> {
    let bits = v.to_bits().to_le_bytes();
    let mut b = [0u8; 16];
    for i in 0..4 {
        b[i * 4..i * 4 + 4].copy_from_slice(&bits);
    }
    W::V128Const(i128::from_le_bytes(b))
}

/// Byte-shuffle masks for the two result registers of a two-register permute
/// ([`crate::ir::NeonStmt::Permute`]). Input `a` is shuffle bytes 0..width, `b` is 16..16+width
/// (`width` = 16 for the quad form, 8 for the double form). Elements are `esize` bits.
fn permute_masks(op: crate::ir::PermuteOp, esize: u8, q: bool) -> ([u8; 16], [u8; 16]) {
    use crate::ir::PermuteOp::*;
    let ebytes = (esize / 8) as usize;
    let width = if q { 16 } else { 8 };
    let n = width / ebytes; // elements per register
    // Per-output-element source: `(from_b, index)`.
    let mut a_el: Vec<(bool, usize)> = Vec::with_capacity(n);
    let mut b_el: Vec<(bool, usize)> = Vec::with_capacity(n);
    match op {
        Trn => {
            // Transpose adjacent pairs: a_new = [a0,b0,a2,b2,...], b_new = [a1,b1,a3,b3,...].
            for k in 0..n / 2 {
                a_el.push((false, 2 * k));
                a_el.push((true, 2 * k));
                b_el.push((false, 2 * k + 1));
                b_el.push((true, 2 * k + 1));
            }
        }
        Zip => {
            // Interleave into [a0,b0,a1,b1,...]; low n -> a_new, high n -> b_new.
            let comb = |i: usize| if i % 2 == 0 { (false, i / 2) } else { (true, i / 2) };
            for j in 0..n {
                a_el.push(comb(j));
            }
            for j in 0..n {
                b_el.push(comb(n + j));
            }
        }
        Uzp => {
            // De-interleave the concatenation c = [a0..an, b0..bn]: a_new = c[even], b_new = c[odd].
            let c = |i: usize| if i < n { (false, i) } else { (true, i - n) };
            for j in 0..n {
                a_el.push(c(2 * j));
            }
            for j in 0..n {
                b_el.push(c(2 * j + 1));
            }
        }
    }
    let expand = |src: &[(bool, usize)]| -> [u8; 16] {
        let mut m = [0u8; 16];
        for (j, (from_b, idx)) in src.iter().enumerate() {
            let base = if *from_b { 16 } else { 0 } + idx * ebytes;
            for wb in 0..ebytes {
                m[j * ebytes + wb] = (base + wb) as u8;
            }
        }
        m
    };
    (expand(&a_el), expand(&b_el))
}

/// Emit a NEON single-element load/store ([`crate::ir::NeonStmt::ElemMem`]). A `D`
/// register is a raw 64-bit value ([`get_d_bits`]/[`set_d_bits`]); an element is
/// `esize` bits at lane offset `lane * esize`. A lane load reads `esize` bits and
/// splices them into that field of `d`; a broadcast load replicates the read
/// element across all `64/esize` lanes (a multiply by the per-lane "1" constant); a
/// lane store extracts the field and writes `esize` bits out.
fn emit_elem_mem(
    f: &mut Function,
    d: u8,
    esize: u8,
    lane: crate::ir::ElemLane,
    addr: &Value,
    load: bool,
    base: u32,
) {
    use crate::ir::ElemLane;
    let elem_mask: i64 = if esize >= 64 { -1 } else { (1i64 << esize) - 1 };
    let load_op = match esize {
        8 => W::I32Load8U(mem_arg()),
        16 => W::I32Load16U(mem_arg()),
        _ => W::I32Load(mem_arg()),
    };
    let store_op = match esize {
        8 => W::I32Store8(mem_arg()),
        16 => W::I32Store16(mem_arg()),
        _ => W::I32Store(mem_arg()),
    };
    match (lane, load) {
        (ElemLane::One(idx), true) => {
            let shift = (idx as i64) * (esize as i64);
            // cleared = d & ~(elem_mask << shift)
            get_d_bits(f, d);
            f.instruction(&W::I64Const(!(elem_mask << shift)));
            f.instruction(&W::I64And);
            // field = (zext(mem) & elem_mask) << shift
            emit_addr(f, addr, base);
            f.instruction(&load_op);
            f.instruction(&W::I64ExtendI32U);
            f.instruction(&W::I64Const(elem_mask));
            f.instruction(&W::I64And);
            f.instruction(&W::I64Const(shift));
            f.instruction(&W::I64Shl);
            // d = cleared | field
            f.instruction(&W::I64Or);
            set_d_bits(f, d);
        }
        (ElemLane::One(idx), false) => {
            // mem = (d >> shift) truncated to esize bits (the store width truncates).
            let shift = (idx as i64) * (esize as i64);
            emit_addr(f, addr, base);
            get_d_bits(f, d);
            f.instruction(&W::I64Const(shift));
            f.instruction(&W::I64ShrU);
            f.instruction(&W::I32WrapI64);
            f.instruction(&store_op);
        }
        (ElemLane::All, true) => {
            // d = (zext(mem) & elem_mask) * broadcast_ones
            let broadcast_ones: i64 = match esize {
                8 => 0x0101_0101_0101_0101u64 as i64,
                16 => 0x0001_0001_0001_0001u64 as i64,
                32 => 0x0000_0001_0000_0001u64 as i64,
                _ => 1,
            };
            emit_addr(f, addr, base);
            f.instruction(&load_op);
            f.instruction(&W::I64ExtendI32U);
            f.instruction(&W::I64Const(elem_mask));
            f.instruction(&W::I64And);
            f.instruction(&W::I64Const(broadcast_ones));
            f.instruction(&W::I64Mul);
            set_d_bits(f, d);
        }
        (ElemLane::All, false) => unreachable!("store to all lanes has no encoding"),
    }
}

/// N,Z,C,V for `a + b + cin`. `cin` is a runtime value (0 or 1) - an immediate for
/// add/sub/cmp, the C flag for adc/sbc. Uses i64 for an always-correct unsigned
/// carry. `cin` is emitted once into a local, since a flag read must not be
/// duplicated if it ever had a cost.
fn emit_flags_add(f: &mut Function, a: &Value, b: &Value, cin: &Value, base: u32) {
    emit_value(f, a, base);
    f.instruction(&W::LocalSet(L_T0)); // a
    emit_value(f, b, base);
    f.instruction(&W::LocalSet(L_T1)); // b
    emit_value(f, cin, base);
    f.instruction(&W::LocalSet(L_T3)); // cin
    // res = a + b + cin
    f.instruction(&W::LocalGet(L_T0));
    f.instruction(&W::LocalGet(L_T1));
    f.instruction(&W::I32Add);
    f.instruction(&W::LocalGet(L_T3));
    f.instruction(&W::I32Add);
    f.instruction(&W::LocalSet(L_T2)); // res
    // Z = res == 0
    f.instruction(&W::LocalGet(L_T2));
    f.instruction(&W::I32Eqz);
    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::Z)));
    // N = res >> 31
    f.instruction(&W::LocalGet(L_T2));
    f.instruction(&W::I32Const(31));
    f.instruction(&W::I32ShrU);
    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::N)));
    // C = (a_u64 + b_u64 + cin) >> 32
    f.instruction(&W::LocalGet(L_T0));
    f.instruction(&W::I64ExtendI32U);
    f.instruction(&W::LocalGet(L_T1));
    f.instruction(&W::I64ExtendI32U);
    f.instruction(&W::I64Add);
    f.instruction(&W::LocalGet(L_T3));
    f.instruction(&W::I64ExtendI32U);
    f.instruction(&W::I64Add);
    f.instruction(&W::I64Const(32));
    f.instruction(&W::I64ShrU);
    f.instruction(&W::I32WrapI64);
    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::C)));
    // V = (~(a^b) & (a^res)) >> 31
    f.instruction(&W::LocalGet(L_T0));
    f.instruction(&W::LocalGet(L_T1));
    f.instruction(&W::I32Xor);
    f.instruction(&W::I32Const(-1));
    f.instruction(&W::I32Xor);
    f.instruction(&W::LocalGet(L_T0));
    f.instruction(&W::LocalGet(L_T2));
    f.instruction(&W::I32Xor);
    f.instruction(&W::I32And);
    f.instruction(&W::I32Const(31));
    f.instruction(&W::I32ShrU);
    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::V)));
}

/// Emit a guest address as a linear-memory offset (guest addr - base).
fn emit_addr(f: &mut Function, addr: &Value, base: u32) {
    emit_value(f, addr, base);
    f.instruction(&W::I32Const(base as i32));
    f.instruction(&W::I32Sub);
}

fn emit_value(f: &mut Function, v: &Value, base: u32) {
    match v {
        Value::Imm(x) => {
            f.instruction(&W::I32Const(*x as i32));
        }
        Value::Reg(r) => {
            f.instruction(&W::GlobalGet(abi::reg_global(*r as usize)));
        }
        Value::Not(a) => {
            emit_value(f, a, base);
            f.instruction(&W::I32Const(-1));
            f.instruction(&W::I32Xor);
        }
        Value::Flag(flag) => {
            // Flag globals hold 0 or 1, so a plain read is the flag's value.
            f.instruction(&W::GlobalGet(abi::flag_global(*flag)));
        }
        Value::CarryAddResult => {
            // `emit_flags_add` left `a + b + cin` in L_T2; reuse it (see the IR doc).
            f.instruction(&W::LocalGet(L_T2));
        }
        Value::Clz(a) => {
            emit_value(f, a, base);
            f.instruction(&W::I32Clz);
        }
        Value::ThreadPtr => {
            f.instruction(&W::GlobalGet(abi::TP_GLOBAL));
        }
        Value::Bin(op, a, b) => {
            emit_value(f, a, base);
            emit_value(f, b, base);
            f.instruction(&binop(*op));
        }
        Value::Load { addr, size, signed } => {
            emit_addr(f, addr, base);
            if let Some(w) = watch_read_addr() {
                // Diagnostic: trap on a load from the watched guest address so the
                // reader surfaces with a backtrace (guest PC via VITASLOP_TRACK_PC).
                f.instruction(&W::LocalTee(L_T0)); // L_T0 = guest addr - base (kept)
                f.instruction(&load_op(*size, *signed));
                emit_read_watch_check(f, w, base);
            } else {
                f.instruction(&load_op(*size, *signed));
            }
        }
    }
}

fn binop(op: BinOp) -> W<'static> {
    match op {
        BinOp::Add => W::I32Add,
        BinOp::Sub => W::I32Sub,
        BinOp::And => W::I32And,
        BinOp::Or => W::I32Or,
        BinOp::Xor => W::I32Xor,
        BinOp::Shl => W::I32Shl,
        BinOp::Lsr => W::I32ShrU,
        BinOp::Asr => W::I32ShrS,
        BinOp::Ror => W::I32Rotr,
        BinOp::Mul => W::I32Mul,
    }
}

fn mem_arg() -> MemArg {
    MemArg { offset: 0, align: 0, memory_index: 0 }
}

/// Which host imports may be emitted inline, and how far into linear memory an
/// inline load may reach. Built once per module from
/// [`Program::inline_imports`](crate::Program::inline_imports).
#[derive(Default)]
pub struct InlineImports {
    ops: BTreeMap<u32, crate::InlineOp>,
    /// Guest region size in bytes - the bound an inline load must stay inside.
    mem_bytes: u32,
}

impl InlineImports {
    fn new(list: &[crate::InlineImport], mem_bytes: u32) -> Self {
        InlineImports {
            ops: list.iter().map(|i| (i.import, i.op)).collect(),
            mem_bytes,
        }
    }

    /// The inline form of import `index`, together with the highest REBASED address
    /// at which it may be used. `None` when the import has no inline form, or when
    /// guest memory is too small for the load to ever be in range (in which case the
    /// host call is not merely correct but the only option).
    fn lower(&self, index: u32) -> Option<(crate::InlineOp, u32)> {
        let op = *self.ops.get(&index)?;
        // The load reads 4 bytes at `offset + op.offset()`, so the last rebased
        // address it may start from is `mem_bytes - 4 - op.offset()`.
        let limit = self.mem_bytes.checked_sub(4)?.checked_sub(op.offset())?;
        Some((op, limit))
    }
}

/// Emit a host-import call: either the real trap, or - for an import with an inline
/// form - the guest-memory read it amounts to.
///
/// The inline form is guarded so it is EXACTLY equivalent to the host call, never
/// merely equivalent in the expected case. `r0 - base` is compared unsigned against
/// the highest address the load may start from, which rejects both a pointer below
/// the image base (the subtraction wraps to a huge value - this is the null-pointer
/// case) and one too near the end of guest memory, in a single comparison. Either way
/// the real host call runs, so the handler stays the definition of the behaviour and
/// the odd cases keep their exact old semantics.
fn emit_import(f: &mut Function, index: u32, base: u32, inline: &InlineImports) {
    let Some((op, limit)) = inline.lower(index) else {
        f.instruction(&W::I32Const(index as i32));
        f.instruction(&W::Call(IMPORT_FUNC));
        return;
    };
    let crate::InlineOp::LoadShiftMask { offset, shift, mask } = op;
    // t0 = r0 - base, the rebased address of the pointer argument.
    f.instruction(&W::GlobalGet(abi::reg_global(0)));
    f.instruction(&W::I32Const(base as i32));
    f.instruction(&W::I32Sub);
    f.instruction(&W::LocalTee(L_T0));
    f.instruction(&W::I32Const(limit as i32));
    f.instruction(&W::I32GtU);
    f.instruction(&W::If(BlockType::Empty));
    f.instruction(&W::I32Const(index as i32));
    f.instruction(&W::Call(IMPORT_FUNC));
    f.instruction(&W::Else);
    f.instruction(&W::LocalGet(L_T0));
    f.instruction(&W::I32Load(MemArg { offset: offset as u64, align: 0, memory_index: 0 }));
    if shift != 0 {
        f.instruction(&W::I32Const(shift as i32));
        f.instruction(&W::I32ShrU);
    }
    if mask != u32::MAX {
        f.instruction(&W::I32Const(mask as i32));
        f.instruction(&W::I32And);
    }
    f.instruction(&W::GlobalSet(abi::reg_global(0)));
    f.instruction(&W::End);
}

fn load_op(size: MemSize, signed: bool) -> W<'static> {
    match (size, signed) {
        (MemSize::Byte, false) => W::I32Load8U(mem_arg()),
        (MemSize::Byte, true) => W::I32Load8S(mem_arg()),
        (MemSize::Half, false) => W::I32Load16U(mem_arg()),
        (MemSize::Half, true) => W::I32Load16S(mem_arg()),
        (MemSize::Word, _) => W::I32Load(mem_arg()),
    }
}

fn store_op(size: MemSize) -> W<'static> {
    match size {
        MemSize::Byte => W::I32Store8(mem_arg()),
        MemSize::Half => W::I32Store16(mem_arg()),
        MemSize::Word => W::I32Store(mem_arg()),
    }
}
