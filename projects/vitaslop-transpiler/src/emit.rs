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
    BlockType, CodeSection, ConstExpr, DataSection, ElementSection, Elements, Encode, ExportKind,
    ExportSection, Function, FunctionSection, GlobalSection, GlobalType, ImportSection,
    Instruction as W, MemArg, MemorySection, MemoryType, Module, NameMap, NameSection, RefType,
    TableSection, TableType, TypeSection, ValType,
};

use crate::abi;
use crate::ir::{BinOp, Block, ConditionCode, FlagMask, Func, MemSize, Stmt, Term, Value};

/// Wasmtime's own fuel cost for one operator, from `wasmtime_environ`'s
/// `default_operator_cost`: an operator that generates no machine code costs nothing,
/// and everything else costs one.
///
/// This is transcribed rather than invented. The browser's software fuel exists to give
/// the browser the clock signal wasmtime gives native, so the two are only comparable if
/// they bill the same operators - and the difference is not small. `end` alone is one of
/// the most common operators in this codegen (every predicated guest instruction lowers
/// to an `if`/`end` pair), so billing the zero-cost set inflates the browser's idea of
/// how much work it has done, and its clock with it.
fn operator_cost(i: &W) -> u32 {
    match i {
        // Nop and drop generate no code.
        W::Nop | W::Drop => 0,
        // Control flow may create branches, but is generally cheap and free. Note the
        // absence of `if`, which does pay for its conditional check.
        W::Block(_) | W::Loop(_) | W::Unreachable | W::Return | W::Else | W::End => 0,
        _ => 1,
    }
}

/// Whether wasmtime flushes its buffered fuel BEFORE emitting this operator, from the
/// match in `wasmtime_internal_cranelift`'s `fuel_before_op`.
///
/// This is what makes fuel a count of the operators actually EXECUTED rather than of the
/// operators present: a buffered charge is committed at every point control could leave
/// the current straight line, so the arm of an `if` that is not taken is never billed.
/// Charging a whole guest block up front instead - the obvious cheap design - bills every
/// untaken arm in it, and on this codegen that alone ran the browser's clock several
/// times fast.
fn operator_flushes(i: &W) -> bool {
    matches!(
        i,
        // Leaving this function, or entering another one: the counter has to be
        // committed because it may be read while control is elsewhere.
        W::Unreachable
            | W::Return
            | W::Call(_)
            | W::CallIndirect { .. }
            | W::ReturnCall(_)
            | W::ReturnCallIndirect { .. }
            | W::ReturnCallRef(_)
            | W::Throw(_)
            | W::ThrowRef
            // A loop header, so the code before it is counted once rather than per turn.
            | W::Loop(_)
            // A branch whose edge is not known until runtime.
            | W::If(_)
            | W::Br(_)
            | W::BrIf(_)
            | W::BrTable(..)
            | W::BrOnNull(_)
            | W::BrOnNonNull(_)
            // Leaving a scope: there are several ways out, so this is the only chance.
            | W::End
            | W::Else
    )
    // `Block` is deliberately absent: entering one is unconditional, so it is
    // straight-line code and the exit accounts for it.
}

/// A function body under construction: the encoded instruction bytes, and - when the
/// build opted into software fuel - wasmtime's fuel accounting emitted inline as they go.
///
/// # Why the accounting lives here
/// Every instruction in a body passes through [`Body::instruction`], which is the only
/// place that can see both an operator and its position. Wasmtime does its accounting at
/// exactly this seam (`fuel_before_op`, called per operator as it lowers), so mirroring
/// it here mirrors it exactly, rather than approximating it from the IR - and no caller
/// has to remember to charge anything.
///
/// Fuel bookkeeping itself emits through [`Body::untolled`] and is never billed: native
/// meters a module that has none of it, so billing our own instrumentation would charge
/// the browser for work native never does.
///
/// Encoding into a plain `Vec<u8>` rather than straight into a `wasm_encoder::Function`
/// costs nothing and keeps the locals declaration - which depends on choices made after
/// the body is under way - out of the picture until [`Body::into_function`].
struct Body {
    bytes: Vec<u8>,
    /// Whether to emit fuel accounting at all. Captured once at construction so a native
    /// build pays a single branch per instruction and emits byte-identical code.
    fuelled: bool,
    /// Fuel charged but not yet committed to the counter - wasmtime's `fuel_consumed`.
    pending: u32,
    /// Guest ARM instructions retired but not yet committed. Rides the SAME commit as
    /// `pending` (see [`Body::flush`]), which is why the clock costs nothing.
    pending_arm: u32,
    /// Every operator this body has emitted, by the same cost rule, whether or not this
    /// build meters fuel. Counted unconditionally because it is a property of the CODE,
    /// and a native build that never meters is exactly the build whose expansion factor
    /// the calibration needs (see [`Expansion`]).
    billed: u64,
    /// Of `billed`, the moves of ARM core state (registers and flags) to and from the
    /// instance's globals. See [`Expansion::core_state_ops`].
    core_state: u64,
}

impl Body {
    fn new() -> Self {
        let fuelled = fuel_interval() != 0;
        Body {
            bytes: Vec::new(),
            fuelled,
            // Wasmtime starts a function at one, so that even an empty one costs
            // something. Entering a function is real work: it is a call.
            pending: u32::from(fuelled),
            pending_arm: 0,
            billed: 1,
            core_state: 0,
        }
    }

    /// Emit one instruction of translated guest work, billing it as wasmtime would.
    fn instruction(&mut self, i: &W) -> &mut Self {
        let cost = operator_cost(i);
        self.billed += u64::from(cost);
        // The 16 registers and the 4 flags occupy the first globals, in that order (see
        // `abi`), so one range test identifies a core-state move.
        if let W::GlobalGet(g) | W::GlobalSet(g) = i {
            if *g < abi::REG_COUNT as u32 + abi::FLAG_COUNT as u32 {
                self.core_state += 1;
            }
        }
        if self.fuelled {
            self.pending += cost;
            if operator_flushes(i) {
                self.flush();
            }
        }
        i.encode(&mut self.bytes);
        self
    }

    /// Emit one instruction of fuel bookkeeping, which is not guest work and is neither
    /// billed nor a flush point.
    fn untolled(&mut self, i: &W) -> &mut Self {
        i.encode(&mut self.bytes);
        self
    }

    /// Commit the buffered charge to the work counter. A no-op when nothing is buffered,
    /// which is common - a flush point immediately after another one buffers nothing, and
    /// emitting a `+= 0` for it would be pure code size on the hottest path there is.
    ///
    /// # One `i64.add` commits BOTH counters
    /// [`abi::WORK_GLOBAL`] packs guest instructions in its high half and operators in its
    /// low half, both counting UP, so one add of a packed constant advances both with no
    /// borrow between them. That is what makes billing the clock in guest instructions
    /// FREE: it is the same four operators the operator-only commit already cost.
    fn flush(&mut self) {
        if !self.fuelled || (self.pending == 0 && self.pending_arm == 0) {
            return;
        }
        let packed = abi::pack_work(self.pending_arm, self.pending);
        self.pending = 0;
        self.pending_arm = 0;
        self.untolled(&W::GlobalGet(abi::WORK_GLOBAL));
        self.untolled(&W::I64Const(packed));
        self.untolled(&W::I64Add);
        self.untolled(&W::GlobalSet(abi::WORK_GLOBAL));
    }

    /// Buffer a basic block's GUEST INSTRUCTION count, committed by the next [`flush`]
    /// alongside the operator charge. Costs nothing on its own - see [`flush`].
    fn charge_guest_instructions(&mut self, n: u32) {
        if self.fuelled {
            self.pending_arm += n;
        }
    }

    fn into_function<L>(self, locals: L) -> Function
    where
        L: IntoIterator<Item = (u32, ValType)>,
        L::IntoIter: ExactSizeIterator,
    {
        debug_assert_eq!(
            self.pending, 0,
            "a finished body must have committed its fuel; every body ends in `end`, \
             which is a flush point"
        );
        let mut f = Function::new(locals);
        f.raw(self.bytes);
        f
    }
}

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
fn and_armed(f: &mut Body) {
    let off = ARM_WORD_OFF.with(|c| c.get());
    if off == 0 {
        return;
    }
    f.instruction(&W::I32Const(0));
    f.instruction(&W::I32Load(MemArg { offset: off, align: 2, memory_index: 0 }));
    f.instruction(&W::I32And);
}

// --- guest-store dirty map ------------------------------------------------
//
// One byte per 4 KB page of linear memory, set to 1 by every translated store. The
// runtime reads it to answer "can this region of guest memory have changed since I
// last looked?" exactly, without reading the region.
//
// # Why it exists
// A texture lives in guest memory and the capture needs its PIXELS, so it retains a
// snapshot and compares it against guest memory once per scene to find out whether it
// is still current (see `TextureSnapshots`). That compare is EXACT and it is
// enormous: measured on a live race, 116.8 MB a frame, 40% of the whole frame, and it
// re-reads 0.0 MB - it is paying memory bandwidth to prove nothing changed.
//
// The Vita does none of this: GXM hands the GPU a pointer and the SGX reads texture
// memory through the MMU when it rasterises. The compare is an artefact of decoupling
// capture from render, and the exact way to remove it is to know which pages the guest
// wrote - which the guest itself can say, for the cost of a few instructions per store.
//
// # Why this is not on by default on every engine
// The mark is real wasm instructions, and on NATIVE wasmtime bills every operator it
// executes - so a native build with this on would burn fuel for host bookkeeping and
// the game clock, which is priced in fuel, would speed up with it. That is fitting a
// constant to an emulator artefact, which this project does not do. Emitted
// [`Body::untolled`], so on an engine whose fuel is the transpiler's OWN software
// counter (the browser) the marks are invisible to the clock, exactly as they should
// be. So: the browser turns this on and drops the compare; native leaves it off and
// keeps the compare. Both are exact, by different means.
thread_local! {
    /// Whether modules emitted on this thread mark their stores. `u8::MAX` is the
    /// "never set" sentinel, so a host can ask for OFF explicitly and mean it.
    static DIRTY_TRACKING: std::cell::Cell<u8> = const { std::cell::Cell::new(u8::MAX) };
    /// Linear-memory byte offset of the dirty map for the module being emitted on this
    /// thread, or 0 when this build has none. Thread-local for the same reason as
    /// [`ARM_WORD_OFF`].
    static DIRTY_OFF: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Turn guest-store dirty tracking on or off for modules emitted on this thread after
/// this call. A host with a software fuel counter calls this; one billed by the engine
/// leaves it alone and emits byte-identical code. Overrides `VITASLOP_DIRTY_PAGES`,
/// which is the same knob for a native experiment (the browser has no environment).
pub fn set_dirty_tracking(on: bool) {
    DIRTY_TRACKING.with(|c| c.set(u8::from(on)));
}

/// Does this build mark guest stores? See [`set_dirty_tracking`].
pub fn dirty_tracking() -> bool {
    use std::sync::OnceLock;
    static FROM_ENV: OnceLock<bool> = OnceLock::new();
    match DIRTY_TRACKING.with(|c| c.get()) {
        u8::MAX => *FROM_ENV.get_or_init(|| std::env::var("VITASLOP_DIRTY_PAGES").is_ok()),
        n => n != 0,
    }
}

/// Log2 of the dirty map's granule. 4 KB, the wasm page, so a page index is a plain
/// `addr >> 12` and the runtime's page arithmetic needs no second unit.
pub const DIRTY_SHIFT: u32 = 12;

/// Byte offset of the EPOCH within the dirty block. The block leads with it so a host
/// that knows `dirty_off` knows both.
pub const DIRTY_EPOCH_OFF: u64 = 0;

/// Byte offset of the page map within the dirty block: page `p`'s stamp is the byte at
/// `dirty_off + DIRTY_MAP_OFF + p`.
pub const DIRTY_MAP_OFF: u64 = 4;

/// Mark the page of the rebased address on top of the stack, LEAVING that address on
/// the stack for the store that follows.
///
/// Seven operators, all [`untolled`](Body::untolled) - this is host bookkeeping, not
/// guest work, and billing it would charge the game clock for something only one
/// engine does.
///
/// # It stamps an EPOCH, not a flag, and that is what makes it exact
/// A flag would have to be cleared by whoever reads it, and readers here are
/// per-TEXTURE while the granule is a page: two textures sharing one page would clear
/// each other's evidence, and the second would serve stale pixels. So the guest writes
/// the CURRENT EPOCH - a byte the host keeps in linear memory just below the map - and
/// nothing is ever cleared. A reader asks "was any of my pages stamped at or after the
/// epoch I recorded when I last read these bytes?", which is a question every reader
/// can answer independently and none can spoil for another.
///
/// The epoch lives in linear MEMORY rather than a global because a guest thread is its
/// own module instance: a global would be per-thread, and a store on one thread has to
/// be visible to a reader that ran on another.
///
/// # The address is the store's START, and that is exactly enough
/// A store spanning a page boundary stamps only the page it STARTS in. It is not
/// widened here, because the widening costs as much again on the hottest path in the
/// module and the reader can do it for free: the largest translated store is 8 bytes
/// (`i64.store`, a VFP double), so a store that reaches into page P can only have
/// started in P or P-1. A reader asking about pages `[first, last]` therefore reads
/// `[first - 1, last]` and misses nothing. `GuestMemory::take_dirty` does that, and
/// says so.
fn emit_dirty_mark(f: &mut Body, addr_local: u32) {
    let off = DIRTY_OFF.with(|c| c.get());
    if off == 0 {
        return;
    }
    f.untolled(&W::LocalTee(addr_local));
    f.untolled(&W::I32Const(DIRTY_SHIFT as i32));
    f.untolled(&W::I32ShrU);
    f.untolled(&W::I32Const(0));
    f.untolled(&W::I32Load8U(MemArg {
        offset: off + DIRTY_EPOCH_OFF,
        align: 0,
        memory_index: 0,
    }));
    f.untolled(&W::I32Store8(MemArg {
        offset: off + DIRTY_MAP_OFF,
        align: 0,
        memory_index: 0,
    }));
    f.untolled(&W::LocalGet(addr_local));
}

/// Stamp every page `[addr_local, addr_local + len_local)` touches with the current epoch,
/// for an inline form that writes a RANGE rather than a word. Consumes nothing from the
/// stack and leaves nothing on it.
///
/// # Why the single-address mark cannot serve
/// [`emit_dirty_mark`] stamps only the page the address starts in, and that is exact ONLY
/// because the largest translated store is eight bytes, so a reader that also looks one page
/// below misses nothing. A form whose length the guest chooses breaks that argument outright:
/// a 64 KB `memcpy` crosses sixteen pages and stamping the first would report the other
/// fifteen as untouched. What that produces is not an error - it is a texture the host
/// believes it has already uploaded, drawn from bytes the guest has since overwritten
/// ([[vitaslop-guest-store-stamps]]).
///
/// The host's own writes already stamp the same range (`SharedView::stamp_written`), so this
/// is not a new contract - it is the emitted side of one that exists, and an inline form that
/// skipped it would be the first writer of guest memory that stamps nothing.
///
/// # Which inline forms owe a stamp
/// Only the ones that can write where a TEXTURE SNAPSHOT might be looking. The storing forms
/// above ([`crate::InlineOp::StoreArg`] and its neighbours) all write into a structure the
/// guest has handed to GXM or the kernel as private state - a context block, a lightweight
/// mutex work area, a `SceKernelSysClock` - which is never a texture's own bytes, so they
/// stamp nothing and cost nothing. A form that writes wherever the guest points it has no
/// such argument available and must stamp. Widen that judgement only with a reason, not by
/// analogy: an unstamped write to texture memory is a stale picture with nothing reporting.
///
/// Untolled throughout, like [`emit_dirty_mark`] and for the same reason: this is host
/// bookkeeping that only one engine does, and billing it would charge the game clock for it.
fn emit_dirty_range(f: &mut Body, addr_local: u32, len_local: u32) {
    let off = DIRTY_OFF.with(|c| c.get());
    if off == 0 {
        return;
    }
    // A zero-length write touches no page, and `last = addr + len - 1` would underflow into
    // a stamp of the whole map. The guard is a branch rather than a clamp because zero is a
    // real case (`memcpy(d, s, 0)` is legal C and the handler accepts it), not an error.
    f.untolled(&W::LocalGet(len_local));
    f.untolled(&W::If(BlockType::Empty));
    // dest = map + (addr >> DIRTY_SHIFT). `memory.fill` takes its address from the stack
    // with no immediate offset, so the block base is added here rather than ridden in a
    // MemArg the way the single-page mark rides it.
    f.untolled(&W::LocalGet(addr_local));
    f.untolled(&W::I32Const(DIRTY_SHIFT as i32));
    f.untolled(&W::I32ShrU);
    f.untolled(&W::I32Const((off + DIRTY_MAP_OFF) as i32));
    f.untolled(&W::I32Add);
    // value = the current epoch, read from its own word just below the map.
    f.untolled(&W::I32Const(0));
    f.untolled(&W::I32Load8U(MemArg { offset: off + DIRTY_EPOCH_OFF, align: 0, memory_index: 0 }));
    // count = ((addr + len - 1) >> SHIFT) - (addr >> SHIFT) + 1, the pages the range spans.
    f.untolled(&W::LocalGet(addr_local));
    f.untolled(&W::LocalGet(len_local));
    f.untolled(&W::I32Add);
    f.untolled(&W::I32Const(1));
    f.untolled(&W::I32Sub);
    f.untolled(&W::I32Const(DIRTY_SHIFT as i32));
    f.untolled(&W::I32ShrU);
    f.untolled(&W::LocalGet(addr_local));
    f.untolled(&W::I32Const(DIRTY_SHIFT as i32));
    f.untolled(&W::I32ShrU);
    f.untolled(&W::I32Sub);
    f.untolled(&W::I32Const(1));
    f.untolled(&W::I32Add);
    f.untolled(&W::MemoryFill(0));
    f.untolled(&W::End);
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
pub(crate) const IMPORT_FUNC: u32 = 1;
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

// Software fuel: how many guest loop iterations a thread may run before it must give
// the scheduler a turn. 0 (the default) emits no fuel points at all and the module is
// byte-identical to a build without this feature.
//
// # Why this exists, and why only some hosts want it
// A guest thread can only be taken off the CPU at a point the engine can interrupt.
// Natively that is free: wasmtime's `fuel_async_yield_interval` interrupts a thread
// after a fixed amount of EXECUTION, whatever it is doing. The browser's WebAssembly
// engine has no such counter - there, a guest thread leaves the CPU only when it calls
// out to the host, which is why browser preemption counts HOST CALLS
// (`VITASLOP_BROWSER_QUANTUM_CALLS`).
//
// That covers a busy-wait that polls the host, and it does NOT cover a guest loop that
// makes no host call at all. Such a loop was long assumed not to exist in this title;
// it does. Measured: with host-call preemption alone the browser reached display flip 2
// and then burned 100% CPU indefinitely with a completely FLAT host-call count, while
// native ran the same boot to flip 45 in 4.3 s. The loop spins on a word another thread
// writes, so it can only ever end if something else is allowed to run.
//
// So a host with no engine fuel asks for fuel in the CODE. The counter lives in a wasm
// global (`abi::FUEL_GLOBAL`), which is per-instance - and the preemptive scheduler
// runs one instance per guest thread, so each thread gets its own quantum for free.
//
// # Where the check is emitted, and why that placement is both complete and cheap
// Only on LOOP BACK EDGES: a re-entry of the dispatch loop whose target block address
// is at or below the branching block's. That is complete, because every cycle in a
// control-flow graph contains at least one edge to an address no higher than its
// source - take the lowest-addressed block in the cycle and look at the edge entering
// it. And it is cheap, because straight-line code and forward branches emit nothing;
// only the code that can actually spin pays.
//
// Function entry deliberately gets NO check. An unbounded cycle through CALLS cannot
// spin here: calls are real wasm calls, so a call cycle grows the wasm stack and traps
// rather than running forever.
//
// Thread-local for the same reason as `ARM_WORD_OFF`: emission is single-threaded per
// module, while a test binary emits several modules at once on several threads. A
// process-global here would let one test's fuel setting silently change every other
// test's module.
thread_local! {
    static FUEL_INTERVAL: std::cell::Cell<u32> = const { std::cell::Cell::new(u32::MAX) };
}

/// Set the fuel interval for modules emitted on this thread after this call (0 disables).
/// A host with no engine-level fuel calls this before transpiling; one with real fuel
/// leaves it alone and pays nothing. Overrides `VITASLOP_FUEL`, which is the same knob
/// for a native experiment (the browser has no environment to read one from).
pub fn set_fuel_interval(n: u32) {
    FUEL_INTERVAL.with(|c| c.set(n));
}

/// The fuel interval this build emits with: an explicit [`set_fuel_interval`] if one was
/// made on this thread, else `VITASLOP_FUEL`, else 0 (no fuel). `u32::MAX` is the "never
/// set" sentinel, so a host CAN ask for 0 explicitly and mean it.
pub fn fuel_interval() -> u32 {
    use std::sync::OnceLock;
    static FROM_ENV: OnceLock<u32> = OnceLock::new();
    match FUEL_INTERVAL.with(|c| c.get()) {
        u32::MAX => *FROM_ENV.get_or_init(|| {
            std::env::var("VITASLOP_FUEL").ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0)
        }),
        n => n,
    }
}

/// `VITASLOP_FLAG_POISON=0|1` - the FALSIFIER for the flag-liveness pass
/// ([`crate::flags`]). Instead of leaving an elided flag alone, store this constant into
/// it, so a flag the analysis called dead and something then READS carries a value that
/// is wrong rather than merely stale.
///
/// # Why this can be trusted where a code review cannot
/// Dead-flag elimination is the one optimisation here whose mistakes are invisible: the
/// wrong answer is not a crash or a wrong pixel, it is a flag that happens to hold the
/// right value anyway on the run you tried. The ARM conformance corpus checks all four
/// flags but only at a case's END, where every flag is live by construction, so it cannot
/// see a mid-function claim at all.
///
/// # How to run it: TWO poisoned builds against each other, never against the plain one
/// `VITASLOP_FLAG_POISON=0` and `VITASLOP_FLAG_POISON=1` emit the same NUMBER of
/// operators and differ only in one constant. So they burn identical fuel, advance the
/// guest clock identically, and schedule identically - a perfect A/B in which the single
/// variable is the value an elided flag holds. Run the title in both and compare the
/// renders: identical means nothing read a flag the analysis called dead, because a read
/// would have to resolve differently for 0 than for 1. A difference names a bug in the
/// analysis, loudly.
///
/// It takes a VALUE rather than being a plain switch for exactly that reason: a flag holds
/// 0 or 1, so a single poison constant leaves half of all wrong reads accidentally right.
///
/// **Do NOT compare a poisoned build against the ordinary one.** The stores go through
/// [`Body::untolled`], which excludes them from the SOFTWARE fuel counter - and that
/// counter is the browser's. Native has no software fuel; it meters with wasmtime, which
/// bills every operator in the module including these. MEASURED: the same 7450-frame race
/// burns 339,995,880,296 fuel poisoned against 325,596,212,794 plain, so the poisoned run
/// has a different clock and a different thread interleaving and its render legitimately
/// differs. `untolled` means "not OUR fuel", not "free".
///
/// RESULT, 2026-08-15f: both poison arms rendered SHA-256 `50164a52...` at frame 7450 of
/// `campaign-race.recipe`, bit-identical, on identical fuel. No elided flag was read.
fn flag_poison() -> Option<i32> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<i32>> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_FLAG_POISON").ok().and_then(|s| s.trim().parse().ok())
    })
}

/// Store the poison value into `flag`, if this build is a falsifier build. Untolled, so
/// it is invisible to the fuel counter and therefore to the clock. See [`flag_poison`].
fn poison_flag(f: &mut Body, flag: abi::Flag) {
    if let Some(v) = flag_poison() {
        f.untolled(&W::I32Const(v));
        f.untolled(&W::GlobalSet(abi::flag_global(flag)));
    }
}

/// Emit a fuel point: decrement this thread's quantum and, when it runs out, hand the
/// scheduler a turn. A no-op unless the build opted into fuel, so an ordinary module is
/// byte-identical.
///
/// The reload lives in the emitted code rather than in the host because the guest stack
/// resumes exactly where it suspended: a host that forgot to refill would re-yield on
/// the very next back edge, turning a preemption into a livelock of its own. `i32.le_s`
/// rather than `i32.eqz` for the same reason - a counter that somehow went negative must
/// still preempt, not run forever undetected.
/// Test the fuel counter and, when it has run out, hand the scheduler a turn. Emitted on
/// LOOP BACK EDGES only; a no-op unless the build opted into fuel, so an ordinary module
/// is byte-identical.
///
/// # The unit is wasmtime's fuel, because that is what native's quantum is measured in
/// The counter drives the browser's virtual game clock (a preemption charges
/// `charge_cpu_quantum`), so it has to measure guest EXECUTION the way native's engine
/// fuel does. It now does so by construction: [`Body`] reproduces wasmtime's own
/// accounting - its operator cost table and its flush points - so the browser's quantum
/// and native's `fuel_async_yield_interval` are the same number of the same thing rather
/// than two scales that have to be fitted to each other.
///
/// Four cheaper models were measured against native's clock curve on the same title and
/// recipe, and every one of them is wrong in a way no constant can absorb:
///
/// | model                            | ~f10,300 | ~f29,300 | ~f40,800 |
/// |----------------------------------|----------|----------|----------|
/// | flat 1 per back edge             | -1.6%    | -5.0%    | -15.4%   |
/// | the back-edge block's IR size    | -11.1%   | -5.8%    | -14.6%   |
/// | every block's IR size, up front  | -12%     | -12%     | -12%     |
/// | every block's wasm size, up front| about 3.4x FAST                |
///
/// Each failure named the next mistake. Weighting the back-edge block alone changed
/// nothing, because a loop body is usually SEVERAL blocks and only the one carrying the
/// edge was charged. Charging every block fixed that but left a uniform 12%, because an
/// IR statement is a PROXY for a wasm instruction whose ratio varies by screen. Counting
/// wasm instructions instead overshot by 3.4x, for two reasons that only the real
/// algorithm settles: wasmtime bills NOTHING for `end`/`block`/`loop`/`return`/`drop`,
/// which are a large share of this codegen, and it bills only the operators actually
/// EXECUTED, where an up-front block charge bills every untaken `if` arm as well.
///
/// The BRANCH stays on back edges only. Straight-line code cannot spin, so only the code
/// that can actually livelock pays for the test.
fn emit_fuel_check(f: &mut Body) {
    let n = fuel_interval();
    if n == 0 {
        return;
    }
    // Commit anything still buffered first: the test must see what this thread has just
    // done, not what it had done one basic block ago.
    f.flush();
    // The counter is ADVANCED by `Body`'s own accounting; this only tests it. Advancing
    // here as well would double-charge the block carrying the back edge.
    //
    // The operator half is the LOW 32 bits of `WORK_GLOBAL`, so the test masks it out and
    // compares it against the interval. It counts UP (both halves do, so the packed commit
    // needs no borrow), hence `ge_u` against the interval rather than the old `le_s`
    // against zero. `ge_u` also keeps the "a counter that somehow overshot must still
    // preempt" property the signed test had.
    f.untolled(&W::GlobalGet(abi::WORK_GLOBAL));
    f.untolled(&W::I64Const(abi::WORK_OPS_MASK));
    f.untolled(&W::I64And);
    f.untolled(&W::I64Const(i64::from(n)));
    f.untolled(&W::I64GeU);
    f.untolled(&W::If(BlockType::Empty));
    f.untolled(&W::I32Const(abi::FUEL_SELECTOR as i32));
    f.untolled(&W::Call(IMPORT_FUNC));
    // Clear the OPERATOR half and keep the instruction half: the quantum restarts, the
    // guest's retired-instruction total does not. Zeroing the whole global here would
    // reset the clock's counter every quantum and the game clock would never advance.
    f.untolled(&W::GlobalGet(abi::WORK_GLOBAL));
    f.untolled(&W::I64Const(!abi::WORK_OPS_MASK));
    f.untolled(&W::I64And);
    f.untolled(&W::GlobalSet(abi::WORK_GLOBAL));
    f.untolled(&W::End);
}

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
///
/// Honours [`arm_at_frame`], which is what makes it usable on a live title: a virtual call
/// in an engine's refcount or resource path runs millions of times during boot, so an
/// ungated trace of one is gigabytes of the wrong window. Gated, the same knob answers "who
/// called addRef/release on this object, in the 60 frames around the trap".
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

/// `VITASLOP_WATCH_STORE_LOG` - LOG each store to the watched address (the storing
/// function's address plus the live register file, through the `svc` trace path) and
/// carry on, instead of trapping on the (skip+1)-th.
///
/// The trapping form answers "who wrote this" one writer at a time: each additional
/// writer costs another whole run with `VITASLOP_WATCH_STORE_SKIP` bumped, and it can
/// never say how many writers there are in total - the run just stops trapping, which
/// is indistinguishable from having mis-set the skip. Logging enumerates the complete
/// write history in ONE run, which is what a refcount question needs: "how many times
/// was this incremented, by whom, before the decrement that freed it" is a question
/// about the whole sequence, not about any single writer.
///
/// Pairs with [`arm_at_frame`] (the address of a heap object only exists late) and with
/// `VITASLOP_WATCH_STORE_NZ`.
fn watch_store_log() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| std::env::var("VITASLOP_WATCH_STORE_LOG").is_ok())
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
/// The rebased address of a store, held across the guest-store dirty mark
/// ([`emit_dirty_mark`]) so the mark can index the map and still hand the address to
/// the store. Its own local, not one of the `L_T*` scratches, because a store's
/// address is on the stack at points where those are live (the watchpoint path parks
/// the address in `L_T0` and the value in `L_T1`).
///
/// Declared even in a build with tracking off, where nothing reads it: a wasm local is
/// inert and the engine drops it, and making the declaration conditional would make
/// every index after it depend on a knob.
const L_DIRTY: u32 = 6;
const L_I32_COUNT: u32 = 7;
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
    /// Linear-memory byte offset of the host-mirror block, when some inline import
    /// reads it ([`crate::InlineOp::LoadMirror`]). `None` when none does, which
    /// leaves the memory layout exactly as it was.
    pub mirror_off: Option<u64>,
    /// Linear-memory byte offset of the GUEST-STORE DIRTY MAP, when this build was
    /// emitted with store tracking on ([`dirty_tracking`]). One byte per 4 KB page of
    /// the whole linear memory, set to 1 by every translated store. `None` in a build
    /// without it, which leaves the memory layout exactly as it was.
    ///
    /// See [`emit_dirty_mark`] for what the guest writes and why, and
    /// `TextureSnapshots` in the runtime for what reads it.
    pub dirty_off: Option<u64>,
    /// How many wasm operators this module emits per GUEST INSTRUCTION - see
    /// [`Expansion`].
    pub expansion: Expansion,
}

/// The module's CODE EXPANSION: guest instructions lifted, and wasm operators emitted for
/// them.
///
/// # Why a host needs this number
/// The emulator's game clock is charged per unit of engine FUEL, and a unit of fuel is one
/// executed wasm operator. So the emulated Vita's CPU SPEED is
/// `fuel_rate / expansion` - and `expansion` is a property of this transpiler's codegen,
/// not of the device. Improve the codegen and the emulated console silently gets faster,
/// which is a faithfulness change nobody asked for and nothing reports.
///
/// Until now that factor was a guess written into a doc comment on the calibration
/// constant ("order 0.2-0.5 M ARM instructions"). This measures it. It is STATIC - every
/// emitted operator counted once, not weighted by how often it runs - so it is a property
/// of the build rather than of a run, which is exactly what makes it comparable between
/// two builds.
///
/// The executed figure, which is what the clock actually rides on, is the `fuel ... /frame`
/// line a run reports; compare the two builds' ratio of THAT when a run is available.
#[derive(Clone, Copy, Debug, Default)]
pub struct Expansion {
    /// Guest ARM/Thumb instructions lifted into emitted functions.
    pub arm_instructions: u64,
    /// Wasm operators emitted for them, counted by the same cost rule the fuel counter
    /// uses, so this is directly comparable with a run's fuel figures.
    pub emitted_ops: u64,
    /// Of those, how many are a `global.get`/`global.set` of the ARM CORE STATE - the 16
    /// registers and the four flags.
    ///
    /// This is the share of the translated code that exists only to move guest state
    /// between the instance's globals and the operand stack. In both V8 and Cranelift a
    /// mutable global is a load/store against the instance rather than a machine register,
    /// so it is a memory access per guest operand. It is broken out because it is the
    /// candidate for the next large codegen change (promoting a function's registers into
    /// wasm LOCALS, which the engine can register-allocate), and a change that size wants
    /// its ceiling measured before it is attempted rather than after.
    pub core_state_ops: u64,
}

impl Expansion {
    /// Wasm operators per guest instruction, or 0.0 when nothing was lifted.
    pub fn per_instruction(&self) -> f64 {
        if self.arm_instructions == 0 {
            return 0.0;
        }
        self.emitted_ops as f64 / self.arm_instructions as f64
    }

    /// Share of emitted operators that are core-state moves, as a percentage.
    pub fn core_state_share(&self) -> f64 {
        if self.emitted_ops == 0 {
            return 0.0;
        }
        100.0 * self.core_state_ops as f64 / self.emitted_ops as f64
    }
}

/// Host-mirror slots per page. The block is one page, which is far more than the
/// handful of system values that can ever qualify (a value only belongs here if it
/// cannot change while guest code runs), so overflowing it means the rule was
/// abandoned rather than that the block wants growing.
const MIRROR_SLOTS_PER_PAGE: u32 = abi::PAGE_SIZE / 4;

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
    //
    // Above those, and only when something actually reads it, one page holds the
    // HOST MIRROR block (see `crate::InlineOp::LoadMirror`). It is placed here for
    // the same reason as the armed word: outside the guest region, so no guest
    // allocation can reach it and no guest store can corrupt it.
    // Sized from the TOP slot each op touches, not its base: a pair form reads two words,
    // and sizing from the base would leave its high word past the end of the page.
    let mirror_slots = inline_imports.iter().filter_map(|i| i.op.top_mirror_slot()).max();
    if let Some(top) = mirror_slots {
        assert!(
            top < MIRROR_SLOTS_PER_PAGE,
            "host-mirror slot {top} does not fit the one-page block ({MIRROR_SLOTS_PER_PAGE} slots)",
        );
    }
    let mirror_off =
        mirror_slots.map(|_| (guest_pages + addr_table_pages) * abi::PAGE_SIZE as u64);
    let arm_word_off = arm_at_frame().map(|_| {
        (guest_pages + addr_table_pages + u64::from(mirror_off.is_some())) * abi::PAGE_SIZE as u64
    });
    ARM_WORD_OFF.with(|c| c.set(arm_word_off.unwrap_or(0)));
    let pages_below_dirty = guest_pages
        + addr_table_pages
        + u64::from(mirror_off.is_some())
        + u64::from(arm_word_off.is_some());
    // The GUEST-STORE DIRTY MAP tops the layout, one byte per 4 KB page - see
    // `emit_dirty_mark`. It covers the WHOLE linear memory, itself included, rather
    // than only the guest region: the mark is emitted before its store, so a store to
    // an address outside the guest region would otherwise index past the map, and
    // "off the end of the map" must not mean "into the next block". Covering
    // everything makes any in-memory address a valid index, and an address past the
    // end of memory traps on the mark exactly as it would have on the store.
    //
    // The map's granule is [`DIRTY_SHIFT`] (4 KB), which is NOT the wasm page (64 KB):
    // one byte per 4 KB of the memory, so an index is a plain `addr >> 12` on the
    // hottest path in the module. Sizing is self-referential (the map's own pages need
    // bytes of map), so it is solved by iterating to a fixed point - two rounds at any
    // real size.
    let mut dirty_pages = 0u64;
    if dirty_tracking() {
        loop {
            let total_bytes = (pages_below_dirty + dirty_pages) * abi::PAGE_SIZE as u64;
            let block_bytes = DIRTY_MAP_OFF + (total_bytes >> DIRTY_SHIFT);
            let next = block_bytes.div_ceil(abi::PAGE_SIZE as u64);
            if next == dirty_pages {
                break;
            }
            dirty_pages = next;
        }
    }
    let dirty_off =
        (dirty_pages > 0).then(|| pages_below_dirty * abi::PAGE_SIZE as u64);
    DIRTY_OFF.with(|c| c.set(dirty_off.unwrap_or(0)));
    let total_pages = pages_below_dirty + dirty_pages;
    // Built here rather than at the top of the function because an inline mirror read
    // needs the block's address, which is part of the layout just computed.
    let inline = InlineImports::new(inline_imports, mem_bytes, mirror_off);

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
    // The indirect-call dispatcher: `(target, caller)`. It binary-searches the address
    // table and `call_indirect`s the match, or reports an unmapped target to
    // `dispatch_miss` - see `emit_dispatch`.
    function_section.function(dispatch_ty);
    // The instance reset (see `abi::RESET_EXPORT` and `emit_reset`), appended AFTER the
    // dispatcher so no existing function index moves - the funcref table's entries and
    // every wasm-backtrace-to-guest-function mapping are stated in terms of them.
    function_section.function(func_ty);

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
    // The software WORK counter (see `emit_fuel_check` and `abi::WORK_GLOBAL`): guest
    // instructions in its high half, operators since the last yield in its low half. Both
    // start at zero - the operator half counts UP to the interval now, so a fresh thread
    // is at the START of its quantum rather than needing to be seeded with one.
    let i64_global = GlobalType { val_type: ValType::I64, mutable: true, shared: false };
    globals.global(i64_global, &ConstExpr::i64_const(0)); // WORK_GLOBAL

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
    // The software fuel counter, so a host can see how much of a thread's quantum is
    // left. Always exported (like `guest_pc`) so the export list does not depend on a
    // build option; it reads 0 and never moves unless fuel was asked for.
    exports.export(abi::FUEL_EXPORT, ExportKind::Global, abi::WORK_GLOBAL);

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
    let mut expansion = Expansion::default();
    for (i, func) in funcs.iter().enumerate() {
        let idx = IMPORT_FUNCS + i as u32;
        exports.export(&abi::func_export(func.addr), ExportKind::Func, idx);
        code.function(&emit_func(func, func_index, base, &inline, &mut expansion));
    }
    code.function(&emit_dispatch(funcs, addr_table_off));
    code.function(&emit_reset());
    exports.export(abi::RESET_EXPORT, ExportKind::Func, IMPORT_FUNCS + n + 1);

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
        names.append(IMPORT_FUNCS + n + 1, "reset");
        let mut name_section = NameSection::new();
        name_section.functions(&names);
        module.section(&name_section);
    }
    EmitOutput {
        wasm: module.finish(),
        mem_pages: total_pages as u32,
        arm_word_off,
        mirror_off,
        dirty_off,
        expansion,
    }
}

/// Emit the instance RESET function: `() -> ()`, exported as [`abi::RESET_EXPORT`].
///
/// It writes every per-instance global back to the value the globals section gives it at
/// instantiation, IN THE SAME ORDER that section declares them - the whole ARM register
/// file and flags, the VFP/NEON file (S registers, the Q8..Q15 quads, the FP flags), the
/// diagnostic latches, `tp`, and the fuel counter. After this call an instance is
/// indistinguishable from a fresh one, which is what lets a host REUSE it for the next
/// guest thread instead of instantiating the module again.
///
/// # Why that matters here
/// An instance of a retail title is a funcref table with one entry per translated
/// function - 106,572 of them on one measured title - and every instantiation allocates
/// and eagerly initializes the whole table. A guest that creates a thread per frame
/// therefore hands the browser's GC a fresh copy of that table sixty times a second; the
/// measured renderer went from 875 MB to 2.19 GB in five seconds and was killed.
///
/// # It is emitted UNBILLED
/// Deliberately built as a raw `Function` rather than through [`Body`]: this is host
/// bookkeeping, not guest execution, and billing it would charge the game clock for work
/// the device never does (and would differ between the two engines, since only the
/// browser reuses instances).
fn emit_reset() -> Function {
    let mut f = Function::new([]);
    let mut g = 0u32;
    let mut zero_i32 = |f: &mut Function, g: &mut u32| {
        f.instruction(&W::I32Const(0));
        f.instruction(&W::GlobalSet(*g));
        *g += 1;
    };
    // 16 registers + the 4 integer flags.
    for _ in 0..abi::GLOBAL_COUNT {
        zero_i32(&mut f, &mut g);
    }
    // S0..S31, as raw bits.
    for _ in 0..abi::VFP_S_COUNT {
        zero_i32(&mut f, &mut g);
    }
    // Q8..Q15. These are the reason this function exists: a `v128` global cannot be read
    // or written from JavaScript at all, so a host-side reset could never clear them.
    for _ in 0..abi::VFP_Q_HI_COUNT {
        f.instruction(&W::V128Const(0));
        f.instruction(&W::GlobalSet(g));
        g += 1;
    }
    // The 4 FP condition flags.
    for _ in 0..abi::FP_FLAG_COUNT {
        zero_i32(&mut f, &mut g);
    }
    // The diagnostic latches, `tp`, and the store-watchpoint counter, in the order
    // `emit_module` declares them: WATCH_ARMED, GUEST_PC, WATCH_READ_COUNT, TP,
    // WATCH_STORE_COUNT. `tp` is among them: it is per-THREAD, so a reused instance that
    // kept the previous thread's TLS base would reach another thread's `__thread`
    // variables. The host sets the new thread's value right after this call.
    for _ in 0..5 {
        zero_i32(&mut f, &mut g);
    }
    // A hard assert, not a debug one: this runs ONCE per emit, and if the globals section
    // ever grows a field without this walk growing with it, the reset would quietly write
    // the wrong globals and a reused instance would carry a previous thread's state.
    assert_eq!(
        g,
        abi::WORK_GLOBAL,
        "emit_reset must walk the globals section in declaration order; it stopped at {g} \
         but the work counter is global {}",
        abi::WORK_GLOBAL,
    );
    // The work counter, BOTH halves, exactly as instantiation leaves them. A reused
    // instance that carried the previous thread's instruction total would make the
    // scheduler's next delta enormous, which is a game clock that jumps by hours - and one
    // that carried its operator half would preempt the new thread almost immediately.
    f.instruction(&W::I64Const(0));
    f.instruction(&W::GlobalSet(abi::WORK_GLOBAL));
    f.instruction(&W::End);
    f
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
    let mut f = Body::new();

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
        and_armed(&mut f);
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
    f.into_function([(4, ValType::I32)])
}

/// Emit one guest function as a wasm function: a dispatch loop over its blocks.
fn emit_func(
    func: &Func,
    func_index: &BTreeMap<u32, u32>,
    base: u32,
    inline: &InlineImports,
    // Accumulates this function's contribution to the module's `Expansion`.
    expansion: &mut Expansion,
) -> Function {
    // Locals: $bb + i32 scratch temps (flag computation), then one i64 scratch
    // (double-register split/merge) and one v128 scratch (NEON quad staging).
    let locals: Vec<(u32, ValType)> = if guard_reg().is_some() {
        vec![
            (L_I32_COUNT, ValType::I32),
            (1, ValType::I64),
            (3, ValType::V128),
            (1, ValType::I32), // L_GUARD: pre-call snapshot for the CSR guard
        ]
    } else {
        vec![
            (L_I32_COUNT, ValType::I32),
            (1, ValType::I64),
            (3, ValType::V128),
        ]
    };
    let mut f = Body::new();

    // A stub for an un-liftable function: trap if ever executed.
    if func.stub {
        f.instruction(&W::Unreachable);
        f.instruction(&W::End);
        return f.into_function(locals);
    }

    // Diagnostic entry tracer (opt-in): announce this function's entry to the host
    // `svc` handler, which logs the address and incoming argument registers. Emitted
    // before any block so it fires exactly once per call, on entry (see `trace_funcs`).
    // Honours `arm_at_frame`: a function on an engine's hot path (a resource request, an
    // allocator) is entered thousands of times during boot, so an ungated trace of one
    // buries the frames actually being asked about.
    if trace_funcs().contains(&func.addr) {
        f.instruction(&W::I32Const(1));
        and_armed(&mut f);
        f.instruction(&W::If(BlockType::Empty));
        f.instruction(&W::I32Const(func.addr as i32));
        f.instruction(&W::Call(SVC_FUNC));
        f.instruction(&W::End);
    }

    // Diagnostic forced return (opt-in): set r0 to the configured value and return before
    // running the body, so a readiness/predicate function can be pinned to test downstream.
    if let Some(&val) = force_ret().get(&func.addr) {
        f.instruction(&W::I32Const(val as i32));
        f.instruction(&W::GlobalSet(abi::reg_global(0)));
        f.instruction(&W::Return);
    }

    let n = func.blocks.len() as u32;

    // The guest instructions this function lifts. Counted from the blocks rather than
    // from the emitted code, which is the point of the measurement.
    expansion.arm_instructions += func.blocks.iter().map(|b| u64::from(b.arm_count)).sum::<u64>();

    // Single-block functions need no dispatch machinery.
    if n == 1 {
        emit_block(&mut f, &func.blocks[0], func, func_index, base, inline, 0);
        f.instruction(&W::End);
        expansion.emitted_ops += f.billed;
        expansion.core_state_ops += f.core_state;
        return f.into_function(locals);
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
    expansion.emitted_ops += f.billed;
    expansion.core_state_ops += f.core_state;
    f.into_function(locals)
}

/// Emit one basic block's statements and terminator. `loop_depth` is the wasm
/// branch depth from within this block's code to the enclosing dispatch `loop`.
fn emit_block(
    f: &mut Body,
    block: &Block,
    func: &Func,
    func_index: &BTreeMap<u32, u32>,
    base: u32,
    inline: &InlineImports,
    loop_depth: u32,
) {
    // >>> THE EMULATED CPU CLOCK'S UNIT. Add this block's GUEST INSTRUCTION count to the
    // per-thread counter both engines' schedulers read (`abi::ARM_COUNT_GLOBAL`).
    //
    // Per BLOCK rather than per instruction: `arm_count` is a compile-time constant for
    // the whole block and a block runs to completion or traps, so one add is exact for
    // every path that retires. Four operators per block, and they are UNTOLLED - this is
    // our instrumentation, and billing it as guest work would charge the guest for the
    // clock that measures it (the same rule `Body` already applies to fuel bookkeeping).
    //
    // A block with no guest instructions (a synthesised dispatch or trap block) emits
    // nothing, so the hottest zero-work paths stay byte-identical.
    // The emulated CPU clock's unit. Buffered here and committed by the next flush point
    // in the SAME `i64.add` that commits the operator charge, so it emits NO code of its
    // own - see `Body::flush` and `abi::WORK_GLOBAL`.
    f.charge_guest_instructions(block.arm_count);
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
    // Nothing here charges the software fuel counter: `Body` does it, per operator, as
    // the block below is emitted (see [`emit_fuel_check`]). The CHECK is what is placed
    // by hand, and only on back edges.
    for stmt in &block.stmts {
        emit_stmt(f, stmt, func_index, base, inline, func.addr);
    }
    emit_term(f, &block.term, func, base, loop_depth, block.addr);
}

/// Re-dispatch to the block at `target` address: set `$bb`, branch to the loop.
/// `extra` accounts for any `if`/`block` frames open between here and the loop. `from`
/// is the address of the block doing the branching, which is what makes a back edge
/// recognisable (see [`emit_fuel_check`]).
///
/// The fuel check goes FIRST and is self-contained (`if`..`end` closes before the
/// branch), so it cannot disturb `loop_depth + extra`.
fn goto(f: &mut Body, func: &Func, target: u32, loop_depth: u32, extra: u32, from: u32) {
    if target <= from {
        emit_fuel_check(f);
    }
    let idx = func
        .block_index(target)
        .unwrap_or_else(|| {
            // Name the blocks that DO exist. Without them this says only that something
            // does not fit, and the first question - "is the target just outside this
            // function, or is the function nonsense?" - needs the answer inline.
            let blocks: Vec<String> =
                func.blocks.iter().take(12).map(|b| format!("{:#x}", b.addr)).collect();
            panic!(
                "branch target {target:#x} is not a block in f_{:x} ({} block(s): {}{})",
                func.addr,
                func.blocks.len(),
                blocks.join(" "),
                if func.blocks.len() > 12 { " ..." } else { "" }
            )
        })
        as i32;
    f.instruction(&W::I32Const(idx));
    f.instruction(&W::LocalSet(L_BB));
    f.instruction(&W::Br(loop_depth + extra));
}

fn emit_term(f: &mut Body, term: &Term, func: &Func, base: u32, loop_depth: u32, from: u32) {
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
            goto(f, func, *target, loop_depth, 0, from);
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
                // A switch arm is a dispatch-loop re-entry like any other, so a backward
                // one is a back edge and needs its fuel point. This pad open-codes what
                // `goto` does (its branch depth counts the switch frames), so it has to
                // repeat the check rather than inherit it.
                if target <= from {
                    emit_fuel_check(f);
                }
                f.instruction(&W::I32Const(idx));
                f.instruction(&W::LocalSet(L_BB));
                f.instruction(&W::Br(loop_depth + n - i as u32));
            }
            f.instruction(&W::End); // closes the default (outer) block
            match default {
                // The range check already routed out-of-range indices away, so this
                // is faithful when known and unreachable in practice.
                Some(d) => goto(f, func, *d, loop_depth, 0, from),
                None => {
                    f.instruction(&W::Unreachable);
                }
            }
        }
        Term::Branch { cond, taken } => {
            emit_cond(f, *cond);
            f.instruction(&W::If(BlockType::Empty));
            goto(f, func, *taken, loop_depth, 1, from); // +1 for the `if` frame
            f.instruction(&W::End);
        }
        Term::BranchZero { reg, nonzero, taken } => {
            f.instruction(&W::GlobalGet(abi::reg_global(*reg as usize)));
            f.instruction(&W::I32Eqz); // reg == 0
            if *nonzero {
                f.instruction(&W::I32Eqz); // reg != 0
            }
            f.instruction(&W::If(BlockType::Empty));
            goto(f, func, *taken, loop_depth, 1, from);
            f.instruction(&W::End);
        }
    }
}

/// Push a 0/1 i32 for `cond` computed from the flag globals.
fn emit_cond(f: &mut Body, cond: ConditionCode) {
    use ConditionCode::*;
    fn get(f: &mut Body, flag: abi::Flag) {
        f.instruction(&W::GlobalGet(abi::flag_global(flag)));
    }
    fn eqz(f: &mut Body) {
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
fn emit_read_watch_check(f: &mut Body, w: u32, base: u32) {
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
fn guard_snapshot(f: &mut Body) {
    if let Some(r) = guard_reg() {
        f.instruction(&W::GlobalGet(abi::reg_global(r as usize)));
        f.instruction(&W::LocalSet(L_GUARD));
    }
}

/// After a call returns, trap if the guarded register differs from its pre-call
/// snapshot - the callee failed to preserve it (no-op unless the guard is set).
fn guard_check(f: &mut Body) {
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
    f: &mut Body,
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
                    if watch_store_log() {
                        // Log the writer (its own address, through the `svc` trace
                        // marker path) and carry on, so one run enumerates them all.
                        f.instruction(&W::I32Const(func_addr as i32));
                        f.instruction(&W::Call(SVC_FUNC));
                    } else {
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
                    }
                    f.instruction(&W::End);
                }
                f.instruction(&W::LocalGet(L_T0));
                emit_dirty_mark(f, L_DIRTY);
                f.instruction(&W::LocalGet(L_T1));
                f.instruction(&store_op(*size));
            } else {
                emit_addr(f, addr, base);
                emit_dirty_mark(f, L_DIRTY);
                emit_value(f, data, base);
                f.instruction(&store_op(*size));
            }
        }
        Stmt::FlagsAdd { a, b, cin, live } => emit_flags_add(f, a, b, cin, *live, base),
        // A logical flag update whose result nothing can observe is dropped whole - unlike
        // `FlagsAdd`, nothing reads back its intermediate. `value` is a pure expression by
        // construction (`ir::Value` has no side effects; a `Load` cannot fault, since the
        // guest's whole address space is the linear memory), so not evaluating it is not
        // observable either.
        Stmt::FlagsLogic { value, carry, live } => {
            if !live.has(abi::Flag::Z) {
                poison_flag(f, abi::Flag::Z);
            }
            if !live.has(abi::Flag::N) {
                poison_flag(f, abi::Flag::N);
            }
            match (live.has(abi::Flag::Z), live.has(abi::Flag::N)) {
                (false, false) => {}
                // Both: the original sequence exactly - `tee` feeds Z from the stack and
                // leaves the value in the scratch local for N.
                (true, true) => {
                    emit_value(f, value, base);
                    f.instruction(&W::LocalTee(L_T0));
                    f.instruction(&W::I32Eqz);
                    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::Z)));
                    f.instruction(&W::LocalGet(L_T0));
                    f.instruction(&W::I32Const(31));
                    f.instruction(&W::I32ShrU);
                    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::N)));
                }
                // Only one of them: the value is used once, so it never reaches the
                // scratch local at all.
                (true, false) => {
                    emit_value(f, value, base);
                    f.instruction(&W::I32Eqz);
                    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::Z)));
                }
                (false, true) => {
                    emit_value(f, value, base);
                    f.instruction(&W::I32Const(31));
                    f.instruction(&W::I32ShrU);
                    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::N)));
                }
            }
            if let Some(c) = carry {
                if live.has(abi::Flag::C) {
                    emit_value(f, c, base);
                    f.instruction(&W::I32Const(1));
                    f.instruction(&W::I32And);
                    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::C)));
                } else {
                    poison_flag(f, abi::Flag::C);
                }
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
            let extend = |f: &mut Body| {
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
        Stmt::ShiftRegFlags { kind, rd, rn, amount, set_flags, live } => {
            emit_shift_reg_flags(f, *kind, *rd, rn, amount, *set_flags, *live, base)
        }
    }
}

/// Push byte `i` (0..3) of register `r`, zero-extended: `(r >> 8i) & 0xff`.
fn push_reg_byte(f: &mut Body, r: u8, i: u32) {
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
fn emit_uadd8(f: &mut Body, rd: u8, rn: u8, rm: u8) {
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
fn emit_sel(f: &mut Body, rd: u8, rn: u8, rm: u8) {
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
    f: &mut Body,
    kind: crate::ir::ShiftKind,
    rd: u8,
    rn: &Value,
    amount: &Value,
    set_flags: bool,
    live: FlagMask,
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
    if live.has(abi::Flag::Z) {
        f.instruction(&W::LocalGet(L_T2));
        f.instruction(&W::I32Eqz);
        f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::Z)));
    } else {
        poison_flag(f, abi::Flag::Z);
    }
    if live.has(abi::Flag::N) {
        f.instruction(&W::LocalGet(L_T2));
        f.instruction(&W::I32Const(31));
        f.instruction(&W::I32ShrU);
        f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::N)));
    } else {
        poison_flag(f, abi::Flag::N);
    }
    // The exact shifter carry-out is the expensive half of this form - a per-kind
    // sequence plus a select against the old C for a zero amount. Skipping it when no
    // reader can see C is the whole point of the mask here.
    if !live.has(abi::Flag::C) {
        poison_flag(f, abi::Flag::C);
        return;
    }
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
fn get_s_f32(f: &mut Body, n: u8) {
    f.instruction(&W::GlobalGet(abi::vfp_s_global(n)));
    f.instruction(&W::F32ReinterpretI32);
}

/// Store the f32 on the stack into S`n` (as raw bits).
fn set_s_f32(f: &mut Body, n: u8) {
    f.instruction(&W::I32ReinterpretF32);
    f.instruction(&W::GlobalSet(abi::vfp_s_global(n)));
}

/// Push the raw 64 bits of D`n` as an i64. Low bank (n < 16): merge the two S
/// halves. Upper bank (n >= 16): extract `i64x2` lane `n & 1` of the quad `q(n/2)`.
fn get_d_bits(f: &mut Body, n: u8) {
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
fn set_d_bits(f: &mut Body, n: u8) {
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
fn get_d_f64(f: &mut Body, n: u8) {
    get_d_bits(f, n);
    f.instruction(&W::F64ReinterpretI64);
}

/// Store the f64 on the stack into D`n` (as raw bits).
fn set_d_f64(f: &mut Body, n: u8) {
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

fn emit_vfp(f: &mut Body, op: &crate::ir::VfpOp) {
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
fn emit_vfp_cmp64(f: &mut Body, rn: u8, rm: Option<u8>) {
    let push_b = |f: &mut Body| match rm {
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
fn emit_vfp_cmp(f: &mut Body, rn: u8, rm: Option<u8>) {
    let push_b = |f: &mut Body| match rm {
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
fn emit_vfp_mem(f: &mut Body, reg: crate::ir::VfpReg, addr: &Value, load: bool, base: u32) {
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
                emit_dirty_mark(f, L_DIRTY);
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
                emit_dirty_mark(f, L_DIRTY);
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
fn neon_get(f: &mut Body, reg: crate::ir::NeonReg) {
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
fn neon_set(f: &mut Body, reg: crate::ir::NeonReg) {
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
    f: &mut Body,
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
    let push_shifted_src = |f: &mut Body| {
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

fn emit_neon(f: &mut Body, op: &crate::ir::NeonStmt, base: u32) {
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
    f: &mut Body,
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
            emit_dirty_mark(f, L_DIRTY);
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
/// Emit the flag computation for `a + b + cin`, computing only the flags in `live`.
///
/// # The result is always computed, the flags are not
/// `L_T2` (the sum) is a value in its own right, not only a flag input: an `adc`/`sbc`
/// that sets flags reads it back as [`Value::CarryAddResult`], because the C flag it
/// would otherwise re-read has already been overwritten with the carry-OUT. So the sum
/// stays unconditional and only the four flag derivations are gated, leaving a floor of
/// six operators even when every flag is dead. Dropping that floor would mean proving no
/// following statement reads [`Value::CarryAddResult`], and a wholly-dead compare is rare
/// enough that the proof is not worth its risk - the flags are where the cost is.
///
/// Each block below is exactly the code that was there before `live` existed, so a fully
/// live statement emits byte-for-byte what it always did.
fn emit_flags_add(f: &mut Body, a: &Value, b: &Value, cin: &Value, live: FlagMask, base: u32) {
    // >>> THE OPERANDS ONLY NEED KEEPING FOR THE CARRY AND THE OVERFLOW.
    //
    // `a`, `b` and `cin` went into scratch locals because C re-reads all three (widened to
    // i64) and V re-reads `a` and `b`. Z and N read only the SUM. So when the expensive
    // pair is dead the three stores and their three reads are dead with them, and the sum
    // is built straight on the stack: nine operators become three. That is the `cmp`
    // feeding a conditional branch - the shape this whole pass exists for - so it is worth
    // spelling the three cases out rather than always paying the general one.
    //
    // Evaluation ORDER is `a`, then `b`, then `cin` in every case, exactly as before: the
    // operands can contain loads, and reordering them would reorder guest reads.
    let keep_operands = live.has(abi::Flag::C) || live.has(abi::Flag::V);
    let keep_cin = live.has(abi::Flag::C);
    if keep_operands {
        emit_value(f, a, base);
        f.instruction(&W::LocalSet(L_T0)); // a
        emit_value(f, b, base);
        f.instruction(&W::LocalSet(L_T1)); // b
        f.instruction(&W::LocalGet(L_T0));
        f.instruction(&W::LocalGet(L_T1));
        f.instruction(&W::I32Add);
        if keep_cin {
            emit_value(f, cin, base);
            f.instruction(&W::LocalTee(L_T3)); // cin, kept for the i64 carry below
        } else {
            emit_value(f, cin, base);
        }
        f.instruction(&W::I32Add);
        f.instruction(&W::LocalSet(L_T2)); // res
    } else {
        // res = a + b + cin, entirely on the stack.
        emit_value(f, a, base);
        emit_value(f, b, base);
        f.instruction(&W::I32Add);
        emit_value(f, cin, base);
        f.instruction(&W::I32Add);
        f.instruction(&W::LocalSet(L_T2)); // res
    }
    // Z = res == 0
    if live.has(abi::Flag::Z) {
        f.instruction(&W::LocalGet(L_T2));
        f.instruction(&W::I32Eqz);
        f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::Z)));
    } else {
        poison_flag(f, abi::Flag::Z);
    }
    // N = res >> 31
    if live.has(abi::Flag::N) {
        f.instruction(&W::LocalGet(L_T2));
        f.instruction(&W::I32Const(31));
        f.instruction(&W::I32ShrU);
        f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::N)));
    } else {
        poison_flag(f, abi::Flag::N);
    }
    // C = (a_u64 + b_u64 + cin) >> 32 - nine operators and the only i64 arithmetic in the
    // integer path, so this is the single most valuable one to skip.
    if live.has(abi::Flag::C) {
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
    } else {
        poison_flag(f, abi::Flag::C);
    }
    // V = (~(a^b) & (a^res)) >> 31
    if live.has(abi::Flag::V) {
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
    } else {
        poison_flag(f, abi::Flag::V);
    }
}

/// Split `v` into a dynamic part and a constant addend such that
/// `v == dynamic + constant` in WRAPPING 32-bit arithmetic, which is the only arithmetic
/// either ARM or wasm `i32` does here.
///
/// Used by [`emit_addr`] to fold the rebase subtraction into an address's own
/// displacement. Every ARM addressing mode that is not a bare register carries one -
/// `[r1, #8]`, `[sp, #-4]`, a literal pool address - so this is one of the most common
/// shapes in the whole instruction stream.
fn split_const_addend(v: &Value) -> (Option<&Value>, u32) {
    match v {
        Value::Imm(k) => (None, *k),
        Value::Bin(BinOp::Add, a, b) => match (&**a, &**b) {
            (Value::Imm(k), other) | (other, Value::Imm(k)) => {
                let (dynamic, c) = split_const_addend(other);
                (dynamic, c.wrapping_add(*k))
            }
            _ => (Some(v), 0),
        },
        // `a - k` is `a + (-k)`; the negation is exact in wrapping arithmetic.
        Value::Bin(BinOp::Sub, a, b) => match &**b {
            Value::Imm(k) => {
                let (dynamic, c) = split_const_addend(a);
                (dynamic, c.wrapping_sub(*k))
            }
            _ => (Some(v), 0),
        },
        _ => (Some(v), 0),
    }
}

/// Emit a guest address as a linear-memory offset (guest addr - base).
///
/// # The rebase is free on any address with a displacement
/// The obvious form is `<address expression>; i32.const base; i32.sub`, and for
/// `[r1, #8]` that is five operators: read r1, push 8, add, push base, subtract. But
/// `(r1 + 8) - base` and `r1 + (8 - base)` are the same value - wrapping addition is
/// associative, and both ARM and wasm `i32` wrap - so the two constants fold into one and
/// it becomes three: read r1, push `8 - base`, add. An absolute address folds all the way
/// to a single constant, and a displacement that happens to equal the base folds to
/// nothing at all.
///
/// This is an identity, not an approximation: no address is out of range here that was in
/// range before, and no access moves. That matters because the alternative - putting the
/// displacement in the load's own `MemArg.offset` - is NOT an identity: wasm adds that
/// offset in 64 bits without wrapping, so a guest pointer just below `base` with a
/// positive displacement would trap where it used to read. Two operators are not worth a
/// difference in when the emulator faults.
fn emit_addr(f: &mut Body, addr: &Value, base: u32) {
    let (dynamic, constant) = split_const_addend(addr);
    let offset = constant.wrapping_sub(base);
    match dynamic {
        None => {
            f.instruction(&W::I32Const(offset as i32));
        }
        Some(v) => {
            emit_value(f, v, base);
            if offset != 0 {
                f.instruction(&W::I32Const(offset as i32));
                f.instruction(&W::I32Add);
            }
        }
    }
}

fn emit_value(f: &mut Body, v: &Value, base: u32) {
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

#[cfg(test)]
mod addr_fold_tests {
    use super::*;

    /// Evaluate the split the way [`emit_addr`] does, given a value for the dynamic part.
    /// The whole optimisation is the claim that this equals the unfolded
    /// `(address) - base`, so the test states that claim directly.
    fn folded(addr: &Value, base: u32, dynamic_value: u32) -> u32 {
        let (dynamic, constant) = split_const_addend(addr);
        let offset = constant.wrapping_sub(base);
        match dynamic {
            None => offset,
            Some(_) => dynamic_value.wrapping_add(offset),
        }
    }

    /// Evaluate the address expression itself, then subtract the base - the form the
    /// emitter used before the fold. Only the shapes the fold recognises are needed.
    fn unfolded(addr: &Value, base: u32, dynamic_value: u32) -> u32 {
        fn eval(v: &Value, r: u32) -> u32 {
            match v {
                Value::Imm(k) => *k,
                Value::Reg(_) => r,
                Value::Bin(BinOp::Add, a, b) => eval(a, r).wrapping_add(eval(b, r)),
                Value::Bin(BinOp::Sub, a, b) => eval(a, r).wrapping_sub(eval(b, r)),
                _ => unreachable!("not a shape these tests build"),
            }
        }
        eval(addr, dynamic_value).wrapping_sub(base)
    }

    /// The fold is an IDENTITY, including at the wrap. `base` is a real Vita region base
    /// and the register values include the ones that make the arithmetic wrap round zero,
    /// which is the only place an associativity claim could fail.
    #[test]
    fn folding_the_rebase_into_the_displacement_changes_no_address() {
        let base = 0x8000_0000u32;
        let shapes = [
            Value::Reg(1),
            Value::Imm(0x8100_0000),
            Value::Bin(BinOp::Add, Box::new(Value::Reg(1)), Box::new(Value::Imm(8))),
            Value::Bin(BinOp::Add, Box::new(Value::Imm(8)), Box::new(Value::Reg(1))),
            Value::Bin(BinOp::Sub, Box::new(Value::Reg(1)), Box::new(Value::Imm(4))),
            // A displacement that cancels the base exactly, which folds to no operator.
            Value::Bin(BinOp::Add, Box::new(Value::Reg(1)), Box::new(Value::Imm(base))),
            // Nested, as a pre-indexed form with two constant steps lowers.
            Value::Bin(
                BinOp::Add,
                Box::new(Value::Bin(
                    BinOp::Add,
                    Box::new(Value::Reg(1)),
                    Box::new(Value::Imm(16)),
                )),
                Box::new(Value::Imm(0x20)),
            ),
        ];
        for r in [0u32, 1, 0x7FFF_FFFF, base, base + 0x1000, 0xFFFF_FFFF, base.wrapping_sub(1)] {
            for shape in &shapes {
                assert_eq!(
                    folded(shape, base, r),
                    unfolded(shape, base, r),
                    "shape {shape:?} at r={r:#x}"
                );
            }
        }
    }

    /// A dynamic subtrahend is NOT a constant addend and must not be folded, or
    /// `[r1, -r2]` would rebase by the wrong amount.
    #[test]
    fn a_register_operand_is_never_folded_into_the_constant() {
        let v = Value::Bin(BinOp::Sub, Box::new(Value::Reg(1)), Box::new(Value::Reg(2)));
        let (dynamic, c) = split_const_addend(&v);
        assert!(dynamic.is_some());
        assert_eq!(c, 0);
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

/// A one-word access at a fixed byte offset from whatever address is on the stack.
fn word_at(offset: u32) -> MemArg {
    MemArg { offset: offset as u64, align: 0, memory_index: 0 }
}

/// Which host imports may be emitted inline, and how far into linear memory an
/// inline load may reach. Built once per module from
/// [`Program::inline_imports`](crate::Program::inline_imports).
#[derive(Default)]
pub struct InlineImports {
    ops: BTreeMap<u32, crate::InlineOp>,
    /// Guest region size in bytes - the bound an inline load must stay inside.
    mem_bytes: u32,
    /// Linear-memory offset of the host-mirror block, when the layout reserved one.
    mirror_off: Option<u64>,
}

/// How a given import lowers inline, with the operand the emitter needs.
enum InlineLowering {
    /// Read through the pointer in r0, falling back to the host call unless the
    /// rebased pointer is `<= limit`.
    Guest { offset: u32, shift: u32, mask: u32, plus: u32, limit: u32 },
    /// Read through the pointer in r0 and shift the whole word LEFT, falling back to
    /// the host call when the pointer is out of range OR the loaded word exceeds `max`
    /// (the clamped case, which only the handler defines - see
    /// [`crate::InlineOp::LoadScaled`]).
    GuestScaled { offset: u32, max: u32, shl: u32, limit: u32 },
    /// Read the host-mirror word at this fixed linear-memory offset. No guard: the
    /// address is a constant inside the module's own reserved page, so there is no
    /// out-of-range case to fall back for.
    Mirror { off: u64 },
    /// Read the 64-bit host-mirror value at this offset into r0/r1. No guard, same
    /// reason as [`InlineLowering::Mirror`].
    MirrorPair { off: u64 },
    /// Store the 64-bit host-mirror value at this offset through the guest pointer in
    /// r0, then set r0 = 0. Guarded on the pointer, which must admit an EIGHT-byte
    /// store rather than the usual four.
    MirrorStorePair { off: u64, limit: u32 },
    /// Store r1 at `r0 + offset` and set r0 = 0. Guarded on the pointer exactly like the
    /// reading forms; a rejected pointer runs the handler, which keeps defining what
    /// writing through it means.
    ArgStore { offset: u32, limit: u32 },
    /// Rewrite the `mask << shift` bitfield of the word at `r0 + offset` from r1 and set
    /// r0 = 0, leaving every other bit of that word alone. Guarded on the pointer exactly
    /// like [`InlineLowering::ArgStore`] - the word is READ as well as written, so a
    /// rejected pointer would otherwise load garbage and store it back.
    ArgStoreField { offset: u32, shift: u32, mask: u32, limit: u32 },
    /// Store r2 at `r0 + offset + r1 * 4` and set r0 = 0, when `r1 < count`. Guarded on
    /// BOTH the pointer and the index: an index past the end is the handler's case, and
    /// `limit` is computed against the LAST element so an in-bounds index can never reach
    /// past the end of guest memory.
    ArgStoreIndexed { offset: u32, count: u32, limit: u32 },
    /// Copy `words` words from the pointer in r2 into the slot at `r0 + offset + r1 * stride`,
    /// stamping r2 itself ahead of them and a zero behind them. Guarded on BOTH pointers, on
    /// the index, and on r2 being non-null - see [`crate::InlineOp::CopyArgIndexed`].
    ArgCopyIndexed { offset: u32, stride: u32, count: u32, words: u32, limit: u32, src_limit: u32 },
    /// Take (`lock`) or release (`!lock`) an uncontended lightweight mutex whose state is
    /// the four words at `layout` from the pointer in r0, using the current thread id at
    /// `thread_off` in the host-mirror block. Guarded on the pointer like every other
    /// pointer form, and on its own predicate besides - see
    /// [`crate::InlineOp::LwMutexLock`].
    LwMutex { layout: crate::LwMutexLayout, thread_off: u64, limit: u32, lock: bool },
    /// Move or compare `r2` bytes between the pointers in r0 and (for the two-pointer
    /// kinds) r1. Guarded on both pointers AND the length, since the length is what decides
    /// how far past a pointer the access reaches - see [`emit_bulk_guard`].
    Bulk { kind: BulkKind, mem_bytes: u32 },
}

/// Which bulk operation an [`InlineLowering::Bulk`] performs. One enum rather than three
/// lowerings because the guard is identical for all three and only the body differs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BulkKind {
    /// `sceClibMemcpy`: r0 <- r1, r2 bytes, r0 preserved.
    Copy,
    /// `sceClibMemset`: r0 <- low byte of r1, r2 bytes, r0 preserved.
    Fill,
    /// `sceClibMemcmp`: r0 = the first differing byte pair's difference over r2 bytes.
    Compare,
}

impl BulkKind {
    /// Whether r1 is a POINTER (and so needs its own bound) rather than a fill byte.
    fn reads_second_pointer(self) -> bool {
        matches!(self, BulkKind::Copy | BulkKind::Compare)
    }

    /// Whether the form WRITES guest memory, and therefore owes the dirty map a stamp.
    fn writes(self) -> bool {
        matches!(self, BulkKind::Copy | BulkKind::Fill)
    }
}

impl InlineImports {
    fn new(list: &[crate::InlineImport], mem_bytes: u32, mirror_off: Option<u64>) -> Self {
        InlineImports {
            ops: list.iter().map(|i| (i.import, i.op)).collect(),
            mem_bytes,
            mirror_off,
        }
    }

    /// How import `index` lowers inline. `None` when the import has no inline form, or
    /// when guest memory is too small for the load to ever be in range (in which case
    /// the host call is not merely correct but the only option).
    fn lower(&self, index: u32) -> Option<InlineLowering> {
        match *self.ops.get(&index)? {
            crate::InlineOp::LoadShiftMask { offset, shift, mask, plus } => {
                // The load reads 4 bytes at `r0 - base + offset`, so the last rebased
                // address it may start from is `mem_bytes - 4 - offset`.
                let limit = self.mem_bytes.checked_sub(4)?.checked_sub(offset)?;
                Some(InlineLowering::Guest { offset, shift, mask, plus, limit })
            }
            crate::InlineOp::LoadScaled { offset, max, shl } => {
                let limit = self.mem_bytes.checked_sub(4)?.checked_sub(offset)?;
                Some(InlineLowering::GuestScaled { offset, max, shl, limit })
            }
            crate::InlineOp::LoadMirror { slot } => {
                // The block is reserved by the same layout pass that fills `mirror_off`
                // from these very ops, so a mirror op without a block is a bug here, not
                // a condition to paper over with a host call.
                let base = self.mirror_off.expect("mirror op emitted with no mirror block");
                Some(InlineLowering::Mirror { off: base + slot as u64 * 4 })
            }
            crate::InlineOp::LoadMirrorPair { slot } => {
                let base = self.mirror_off.expect("mirror op emitted with no mirror block");
                Some(InlineLowering::MirrorPair { off: base + slot as u64 * 4 })
            }
            crate::InlineOp::StoreMirrorPair { slot } => {
                let base = self.mirror_off.expect("mirror op emitted with no mirror block");
                // EIGHT bytes are written from the rebased pointer, so the last address
                // the store may start at is `mem_bytes - 8`, not `- 4`.
                let limit = self.mem_bytes.checked_sub(8)?;
                Some(InlineLowering::MirrorStorePair { off: base + slot as u64 * 4, limit })
            }
            crate::InlineOp::StoreArg { offset } => {
                // Four bytes at `r0 - base + offset`, same arithmetic as the reading forms.
                let limit = self.mem_bytes.checked_sub(4)?.checked_sub(offset)?;
                Some(InlineLowering::ArgStore { offset, limit })
            }
            crate::InlineOp::StoreArgField { offset, shift, mask } => {
                // One word, read and written at the same address, so the same bound as a
                // plain store.
                let limit = self.mem_bytes.checked_sub(4)?.checked_sub(offset)?;
                Some(InlineLowering::ArgStoreField { offset, shift, mask, limit })
            }
            crate::InlineOp::StoreArgIndexed { offset, count } => {
                // The LAST element is the one that decides the limit: an index guarded only
                // against `count` still has to land inside memory, so the pointer bound is
                // computed for `offset + (count - 1) * 4`. Bounding on element zero would
                // let a pointer near the end of memory pass the guard and store past it.
                let last = offset.checked_add(count.checked_sub(1)?.checked_mul(4)?)?;
                let limit = self.mem_bytes.checked_sub(4)?.checked_sub(last)?;
                Some(InlineLowering::ArgStoreIndexed { offset, count, limit })
            }
            crate::InlineOp::CopyArgIndexed { offset, stride, count, words } => {
                // The destination bound is computed against the LAST WORD of the LAST slot,
                // for the same reason the indexed store's is: an index inside `count` still
                // has to land inside memory, and bounding on slot zero would let a pointer
                // near the end of memory pass and write past it.
                let last_slot = offset.checked_add(count.checked_sub(1)?.checked_mul(stride)?)?;
                let last_word = last_slot.checked_add(words.checked_add(1)?.checked_mul(4)?)?;
                let limit = self.mem_bytes.checked_sub(4)?.checked_sub(last_word)?;
                // ...and the SOURCE against the last word it reads. Two pointers, two bounds:
                // a guard on one of them is not a guard.
                let src_limit = self.mem_bytes.checked_sub(words.checked_mul(4)?)?;
                Some(InlineLowering::ArgCopyIndexed { offset, stride, count, words, limit, src_limit })
            }
            crate::InlineOp::LwMutexLock { layout, thread_slot }
            | crate::InlineOp::LwMutexUnlock { layout, thread_slot } => {
                let base = self.mirror_off.expect("mirror op emitted with no mirror block");
                // The pointer must admit the LAST word of the layout, not the first: the
                // guard is one comparison for four accesses.
                let limit = self.mem_bytes.checked_sub(4)?.checked_sub(layout.top())?;
                let lock = matches!(*self.ops.get(&index)?, crate::InlineOp::LwMutexLock { .. });
                Some(InlineLowering::LwMutex {
                    layout,
                    thread_off: base + thread_slot as u64 * 4,
                    limit,
                    lock,
                })
            }
            // No constant limit to precompute: the bound depends on the LENGTH the guest
            // passes, so the whole comparison is built at runtime from `mem_bytes`.
            crate::InlineOp::MemCopy => Some(InlineLowering::Bulk {
                kind: BulkKind::Copy,
                mem_bytes: self.mem_bytes,
            }),
            crate::InlineOp::MemFill => Some(InlineLowering::Bulk {
                kind: BulkKind::Fill,
                mem_bytes: self.mem_bytes,
            }),
            crate::InlineOp::MemCompare => Some(InlineLowering::Bulk {
                kind: BulkKind::Compare,
                mem_bytes: self.mem_bytes,
            }),
        }
    }
}

/// Emit a host-import call: either the real trap, or - for an import with an inline
/// form - the memory read it amounts to.
///
/// A pointer-reading inline form is guarded so it is EXACTLY equivalent to the host
/// call, never merely equivalent in the expected case. `r0 - base` is compared
/// unsigned against the highest address the load may start from, which rejects both a
/// pointer below the image base (the subtraction wraps to a huge value - this is the
/// null-pointer case) and one too near the end of guest memory, in a single
/// comparison. Either way the real host call runs, so the handler stays the definition
/// of the behaviour and the odd cases keep their exact old semantics.
///
/// A host-mirror read takes no pointer and needs no guard: its address is a constant
/// inside a page this module reserved, so it is always in range. What makes IT exact
/// is the host-side contract in [`crate::InlineOp::LoadMirror`], not a guard here.
/// Emit the shared pointer guard for an inline form that touches guest memory through
/// r0, leaving the emitter positioned inside the IN-RANGE arm (the caller emits its body
/// and the closing `End`).
///
/// `L_T0` holds the rebased address on that arm. The single unsigned compare rejects both
/// a pointer below the image base (the subtraction wraps to a huge value, which is the
/// null-pointer case) and one too near the end of guest memory; either way the real host
/// call runs, so the handler keeps defining those cases.
fn emit_pointer_guard(f: &mut Body, base: u32, limit: u32, index: u32) {
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
}

/// Emit the guard a BULK form needs, leaving the emitter inside the IN-RANGE arm with the
/// rebased destination in `L_T0`, the rebased second pointer (or the raw r1) in `L_T1`, and
/// the byte count in `L_T2`.
///
/// # Why the length is part of the bound
/// Every other pointer form reaches a distance named at emit time, so its bound is a
/// constant computed once in [`InlineImports::lower`]. Here the guest chooses the distance,
/// so the bound moves with it: the access at `p` reaches `p + len - 1`, and the only
/// admissible pointers are those with `len` bytes left in front of them. That is
/// `len <= mem_bytes` and `p - base <= mem_bytes - len`, per pointer.
///
/// The length term is not redundant with the pointer terms and cannot be dropped: without
/// it `mem_bytes - len` wraps to a huge value for an absurd length and every pointer passes.
/// The terms are combined with `or` rather than nested `if`s so the fallback is a single
/// `call`, and `or` evaluates both sides - which is why the wrapped subtraction is computed
/// even on the rejecting path and must be harmless there. It is: it is arithmetic on
/// operands, not an access.
fn emit_bulk_guard(f: &mut Body, base: u32, mem_bytes: u32, kind: BulkKind, index: u32) {
    // t2 = len, and the first rejecting term: a length larger than memory itself.
    f.instruction(&W::GlobalGet(abi::reg_global(2)));
    f.instruction(&W::LocalTee(L_T2));
    f.instruction(&W::I32Const(mem_bytes as i32));
    f.instruction(&W::I32GtU);
    // ...or the destination leaves less than `len` bytes in front of it. The single
    // unsigned compare rejects a pointer below the image base as well, by wrapping.
    f.instruction(&W::GlobalGet(abi::reg_global(0)));
    f.instruction(&W::I32Const(base as i32));
    f.instruction(&W::I32Sub);
    f.instruction(&W::LocalTee(L_T0));
    f.instruction(&W::I32Const(mem_bytes as i32));
    f.instruction(&W::LocalGet(L_T2));
    f.instruction(&W::I32Sub);
    f.instruction(&W::I32GtU);
    f.instruction(&W::I32Or);
    if kind.reads_second_pointer() {
        // ...or the source does. Two pointers, two bounds: a guard on one of them is not a
        // guard.
        f.instruction(&W::GlobalGet(abi::reg_global(1)));
        f.instruction(&W::I32Const(base as i32));
        f.instruction(&W::I32Sub);
        f.instruction(&W::LocalTee(L_T1));
        f.instruction(&W::I32Const(mem_bytes as i32));
        f.instruction(&W::LocalGet(L_T2));
        f.instruction(&W::I32Sub);
        f.instruction(&W::I32GtU);
        f.instruction(&W::I32Or);
    }
    f.instruction(&W::If(BlockType::Empty));
    f.instruction(&W::I32Const(index as i32));
    f.instruction(&W::Call(IMPORT_FUNC));
    f.instruction(&W::Else);
}

fn emit_import(f: &mut Body, index: u32, base: u32, inline: &InlineImports) {
    let (offset, shift, mask, plus, limit) = match inline.lower(index) {
        None => {
            f.instruction(&W::I32Const(index as i32));
            f.instruction(&W::Call(IMPORT_FUNC));
            return;
        }
        Some(InlineLowering::Mirror { off }) => {
            // r0 = the mirror word. Address zero plus a constant `offset`, so the whole
            // read is one `i32.load` against a literal.
            f.instruction(&W::I32Const(0));
            f.instruction(&W::I32Load(MemArg { offset: off, align: 0, memory_index: 0 }));
            f.instruction(&W::GlobalSet(abi::reg_global(0)));
            return;
        }
        Some(InlineLowering::MirrorPair { off }) => {
            // r0 = low word, r1 = high word: the ARM EABI's 64-bit return pair.
            f.instruction(&W::I32Const(0));
            f.instruction(&W::I32Load(MemArg { offset: off, align: 0, memory_index: 0 }));
            f.instruction(&W::GlobalSet(abi::reg_global(0)));
            f.instruction(&W::I32Const(0));
            f.instruction(&W::I32Load(MemArg { offset: off + 4, align: 0, memory_index: 0 }));
            f.instruction(&W::GlobalSet(abi::reg_global(1)));
            return;
        }
        Some(InlineLowering::MirrorStorePair { off, limit }) => {
            emit_pointer_guard(f, base, limit, index);
            // *(u64 *)r0 = the mirror pair. Written as two i32 stores rather than one
            // i64 store because the guest pointer carries no alignment guarantee, and
            // the two halves are two separate mirror words in any case.
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::I32Const(0));
            f.instruction(&W::I32Load(MemArg { offset: off, align: 0, memory_index: 0 }));
            f.instruction(&W::I32Store(mem_arg()));
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::I32Const(0));
            f.instruction(&W::I32Load(MemArg { offset: off + 4, align: 0, memory_index: 0 }));
            f.instruction(&W::I32Store(MemArg { offset: 4, align: 0, memory_index: 0 }));
            // The handler returns 0 on success, and the guarded path is the success path.
            f.instruction(&W::I32Const(0));
            f.instruction(&W::GlobalSet(abi::reg_global(0)));
            f.instruction(&W::End);
            return;
        }
        Some(InlineLowering::ArgStore { offset, limit }) => {
            emit_pointer_guard(f, base, limit, index);
            // *(u32 *)(r0 + offset) = r1
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::GlobalGet(abi::reg_global(1)));
            f.instruction(&W::I32Store(MemArg { offset: offset as u64, align: 0, memory_index: 0 }));
            // The handler returns 0, and the guarded path is the one it would have taken.
            f.instruction(&W::I32Const(0));
            f.instruction(&W::GlobalSet(abi::reg_global(0)));
            f.instruction(&W::End);
            return;
        }
        Some(InlineLowering::ArgStoreField { offset, shift, mask, limit }) => {
            emit_pointer_guard(f, base, limit, index);
            // *(u32 *)(r0 + offset) = (old & !(mask << shift)) | ((r1 & mask) << shift).
            // The address is pushed once for the store and once for the load inside it,
            // rather than parked in a local, because `L_T0` already holds it and a second
            // `local.get` is what the reading forms cost too.
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::I32Load(MemArg { offset: offset as u64, align: 0, memory_index: 0 }));
            f.instruction(&W::I32Const(!(mask << shift) as i32));
            f.instruction(&W::I32And);
            f.instruction(&W::GlobalGet(abi::reg_global(1)));
            f.instruction(&W::I32Const(mask as i32));
            f.instruction(&W::I32And);
            if shift != 0 {
                f.instruction(&W::I32Const(shift as i32));
                f.instruction(&W::I32Shl);
            }
            f.instruction(&W::I32Or);
            f.instruction(&W::I32Store(MemArg { offset: offset as u64, align: 0, memory_index: 0 }));
            // The handler returns 0, and the guarded path is the one it would have taken.
            f.instruction(&W::I32Const(0));
            f.instruction(&W::GlobalSet(abi::reg_global(0)));
            f.instruction(&W::End);
            return;
        }
        Some(InlineLowering::ArgStoreIndexed { offset, count, limit }) => {
            emit_pointer_guard(f, base, limit, index);
            // In range. Hand an out-of-bounds INDEX back to the handler - it is the one
            // that defines that case (it reports it), and storing past the array here
            // would overwrite a neighbouring field with nothing to say so.
            f.instruction(&W::GlobalGet(abi::reg_global(1)));
            f.instruction(&W::I32Const(count as i32));
            f.instruction(&W::I32GeU);
            f.instruction(&W::If(BlockType::Empty));
            f.instruction(&W::I32Const(index as i32));
            f.instruction(&W::Call(IMPORT_FUNC));
            f.instruction(&W::Else);
            // *(u32 *)(r0 + offset + r1 * 4) = r2
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::GlobalGet(abi::reg_global(1)));
            f.instruction(&W::I32Const(2));
            f.instruction(&W::I32Shl);
            f.instruction(&W::I32Add);
            f.instruction(&W::GlobalGet(abi::reg_global(2)));
            f.instruction(&W::I32Store(MemArg { offset: offset as u64, align: 0, memory_index: 0 }));
            f.instruction(&W::I32Const(0));
            f.instruction(&W::GlobalSet(abi::reg_global(0)));
            f.instruction(&W::End); // the index guard's `if`
            f.instruction(&W::End); // the pointer guard's `if`
            return;
        }
        Some(InlineLowering::ArgCopyIndexed { offset, stride, count, words, limit, src_limit }) => {
            emit_pointer_guard(f, base, limit, index);
            // The destination is in range. Three more conditions have to hold before this can
            // be a copy, and they are combined into ONE branch so the fallback is a single
            // `call` rather than three nested ones: the sampler unit must be inside the array,
            // and the source pointer must be non-null AND in range. Every one of them is a
            // case the handler defines - an out-of-range unit is REPORTED, a null texture
            // UNBINDS - so falling back is not a safety net here, it is the specification.
            //
            // r1 < count
            f.instruction(&W::GlobalGet(abi::reg_global(1)));
            f.instruction(&W::I32Const(count as i32));
            f.instruction(&W::I32LtU);
            // r2 != 0
            f.instruction(&W::GlobalGet(abi::reg_global(2)));
            f.instruction(&W::I32Const(0));
            f.instruction(&W::I32Ne);
            f.instruction(&W::I32And);
            // (r2 - base) <= src_limit, the same single unsigned compare the destination
            // guard uses, and it rejects a below-image pointer by wrapping.
            f.instruction(&W::GlobalGet(abi::reg_global(2)));
            f.instruction(&W::I32Const(base as i32));
            f.instruction(&W::I32Sub);
            f.instruction(&W::LocalTee(L_T1));
            f.instruction(&W::I32Const(src_limit as i32));
            f.instruction(&W::I32LeU);
            f.instruction(&W::I32And);
            f.instruction(&W::If(BlockType::Empty));
            // dst = (r0 - base) + offset + r1 * stride, held in T2 because every store below
            // reads it. `offset` rides in each store's MemArg, so T2 is the slot base.
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::GlobalGet(abi::reg_global(1)));
            f.instruction(&W::I32Const(stride as i32));
            f.instruction(&W::I32Mul);
            f.instruction(&W::I32Add);
            f.instruction(&W::LocalTee(L_T2));
            // dst[0] = r2 - the source address, kept for IDENTITY only. `LocalTee` above left
            // the address on the stack for this first store.
            f.instruction(&W::GlobalGet(abi::reg_global(2)));
            f.instruction(&W::I32Store(MemArg { offset: offset as u64, align: 0, memory_index: 0 }));
            // dst[1 + k] = *(u32 *)(r2 + 4k) - the control words, copied BY VALUE at the
            // moment of the bind, which is the whole reason this is a copy form.
            for k in 0..words {
                f.instruction(&W::LocalGet(L_T2));
                f.instruction(&W::LocalGet(L_T1));
                f.instruction(&W::I32Load(MemArg { offset: (k * 4) as u64, align: 0, memory_index: 0 }));
                f.instruction(&W::I32Store(MemArg {
                    offset: (offset + 4 + k * 4) as u64,
                    align: 0,
                    memory_index: 0,
                }));
            }
            // dst[words + 1] = 0 - "not from a precomputed state". Written, not left alone:
            // the slot is reused, and a unit bound by a precomputed state and then re-bound
            // directly would otherwise keep the old provenance for the rest of the run.
            f.instruction(&W::LocalGet(L_T2));
            f.instruction(&W::I32Const(0));
            f.instruction(&W::I32Store(MemArg {
                offset: (offset + 4 + words * 4) as u64,
                align: 0,
                memory_index: 0,
            }));
            f.instruction(&W::I32Const(0));
            f.instruction(&W::GlobalSet(abi::reg_global(0)));
            f.instruction(&W::Else);
            f.instruction(&W::I32Const(index as i32));
            f.instruction(&W::Call(IMPORT_FUNC));
            f.instruction(&W::End); // the combined index/source guard
            f.instruction(&W::End); // the destination pointer guard
            return;
        }
        Some(InlineLowering::LwMutex { layout, thread_off, limit, lock }) => {
            emit_pointer_guard(f, base, limit, index);
            // In range. Two values are read once and used twice, so they go in locals:
            // the current thread id from the mirror, and the recursion count (tested in
            // the predicate, then incremented or decremented).
            f.instruction(&W::I32Const(0));
            f.instruction(&W::I32Load(MemArg { offset: thread_off, align: 0, memory_index: 0 }));
            f.instruction(&W::LocalSet(L_T2));
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::I32Load(word_at(layout.count)));
            f.instruction(&W::LocalSet(L_T1));

            // The predicate, built on the stack. Every term is a comparison, so each
            // leaves exactly 0 or 1 and the combining `and`/`or` are bitwise-safe; a raw
            // word used as a truth value here would AND its BITS with the terms either
            // side and admit takes that should have fallen back.
            //
            // r1 == 1: the lock/unlock COUNT argument. Anything else is the handler's.
            f.instruction(&W::GlobalGet(abi::reg_global(1)));
            f.instruction(&W::I32Const(1));
            f.instruction(&W::I32Eq);
            // ...and the work area names ITSELF, so this pointer is the canonical mutex
            // rather than a byte copy of one (which carries the original's id).
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::I32Load(word_at(layout.id)));
            f.instruction(&W::GlobalGet(abi::reg_global(0)));
            f.instruction(&W::I32Eq);
            f.instruction(&W::I32And);
            // ...and nothing is parked on it. Only the host can wake a parked thread, so
            // a mutex with waiters stays entirely on the host.
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::I32Load(word_at(layout.waiters)));
            f.instruction(&W::I32Eqz);
            f.instruction(&W::I32And);
            if lock {
                // ...and it is free OR already mine (a recursive take).
                f.instruction(&W::LocalGet(L_T1));
                f.instruction(&W::I32Eqz);
                f.instruction(&W::LocalGet(L_T0));
                f.instruction(&W::I32Load(word_at(layout.owner)));
                f.instruction(&W::LocalGet(L_T2));
                f.instruction(&W::I32Eq);
                f.instruction(&W::I32Or);
                f.instruction(&W::I32And);
            } else {
                // ...and it is held, AND held by me. Both, not either: releasing a mutex
                // this thread does not own is an error only the handler defines, and
                // decrementing a zero count inline would wrap it to four billion.
                f.instruction(&W::LocalGet(L_T1));
                f.instruction(&W::I32Eqz);
                f.instruction(&W::I32Eqz);
                f.instruction(&W::I32And);
                f.instruction(&W::LocalGet(L_T0));
                f.instruction(&W::I32Load(word_at(layout.owner)));
                f.instruction(&W::LocalGet(L_T2));
                f.instruction(&W::I32Eq);
                f.instruction(&W::I32And);
            }

            f.instruction(&W::If(BlockType::Empty));
            if lock {
                // owner = cur. A no-op on the recursive arm, which is what lets one
                // branch serve both cases.
                f.instruction(&W::LocalGet(L_T0));
                f.instruction(&W::LocalGet(L_T2));
                f.instruction(&W::I32Store(word_at(layout.owner)));
            }
            // count += 1 (take) or -= 1 (release). The release deliberately leaves `owner`
            // alone: every reader tests `count` first, and thid 0 is a real thread, so
            // there is no owner value that could mean "nobody".
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::LocalGet(L_T1));
            f.instruction(&W::I32Const(1));
            f.instruction(if lock { &W::I32Add } else { &W::I32Sub });
            f.instruction(&W::I32Store(word_at(layout.count)));
            f.instruction(&W::I32Const(0));
            f.instruction(&W::GlobalSet(abi::reg_global(0)));
            f.instruction(&W::Else);
            f.instruction(&W::I32Const(index as i32));
            f.instruction(&W::Call(IMPORT_FUNC));
            f.instruction(&W::End); // the predicate's `if`
            f.instruction(&W::End); // the pointer guard's `if`
            return;
        }
        Some(InlineLowering::Bulk { kind, mem_bytes }) => {
            emit_bulk_guard(f, base, mem_bytes, kind, index);
            // In range. The dirty stamp goes FIRST for the writing kinds, so a reader that
            // races the copy sees the page marked before any of its bytes change rather
            // than after some of them have - the same order `emit_dirty_mark` uses, and the
            // same reason.
            if kind.writes() {
                emit_dirty_range(f, L_T0, L_T2);
            }
            match kind {
                BulkKind::Copy => {
                    // memmove(dst, src, len). `memory.copy` is specified to behave as if
                    // the source were read in full before the destination is written, which
                    // is exactly what the handler's read-then-write does - so the two agree
                    // on an OVERLAPPING copy as well as on an ordinary one.
                    f.instruction(&W::LocalGet(L_T0));
                    f.instruction(&W::LocalGet(L_T1));
                    f.instruction(&W::LocalGet(L_T2));
                    f.instruction(&W::MemoryCopy { src_mem: 0, dst_mem: 0 });
                    // r0 is left alone: it is the destination, which is what the handler
                    // returns.
                }
                BulkKind::Fill => {
                    // memset(dst, ch, len). `memory.fill` truncates its value operand to a
                    // byte, which is the handler's `ch as u8`. r1 is passed raw for that
                    // reason - masking it here would be a second spelling of one rule.
                    f.instruction(&W::LocalGet(L_T0));
                    f.instruction(&W::GlobalGet(abi::reg_global(1)));
                    f.instruction(&W::LocalGet(L_T2));
                    f.instruction(&W::MemoryFill(0));
                }
                BulkKind::Compare => {
                    // r0 = 0 up front: it is the answer for equal buffers AND for a zero
                    // length, so the loop only ever has to write the differing case.
                    f.instruction(&W::I32Const(0));
                    f.instruction(&W::GlobalSet(abi::reg_global(0)));
                    // The one form that loops, because a comparison has no bulk instruction
                    // and the count is a runtime value.
                    //
                    // No fuel CHECK is emitted on this back edge, which makes the loop a
                    // region the scheduler cannot preempt. That is deliberate and it is
                    // what the host call already was: a handler runs to completion too.
                    // It cannot livelock, because the trip count is `r2` and the guard
                    // above has already bounded `r2` by the size of guest memory.
                    f.instruction(&W::Block(BlockType::Empty));
                    f.instruction(&W::Loop(BlockType::Empty));
                    // Nothing left to compare: the buffers are equal, and r0 already says so.
                    f.instruction(&W::LocalGet(L_T2));
                    f.instruction(&W::I32Eqz);
                    f.instruction(&W::BrIf(1));
                    // t3 = a[i] - b[i], both ZERO-extended, which is the difference C
                    // requires the sign of and what `crate::mem_compare` computes.
                    f.instruction(&W::LocalGet(L_T0));
                    f.instruction(&W::I32Load8U(mem_arg()));
                    f.instruction(&W::LocalGet(L_T1));
                    f.instruction(&W::I32Load8U(mem_arg()));
                    f.instruction(&W::I32Sub);
                    f.instruction(&W::LocalTee(L_T3));
                    f.instruction(&W::If(BlockType::Empty));
                    f.instruction(&W::LocalGet(L_T3));
                    f.instruction(&W::GlobalSet(abi::reg_global(0)));
                    // Out of the `if`, the `loop` and the `block` at once: the first
                    // difference is the answer and nothing after it is read.
                    f.instruction(&W::Br(2));
                    f.instruction(&W::End); // the difference test
                    // Advance both pointers and count one byte off.
                    f.instruction(&W::LocalGet(L_T0));
                    f.instruction(&W::I32Const(1));
                    f.instruction(&W::I32Add);
                    f.instruction(&W::LocalSet(L_T0));
                    f.instruction(&W::LocalGet(L_T1));
                    f.instruction(&W::I32Const(1));
                    f.instruction(&W::I32Add);
                    f.instruction(&W::LocalSet(L_T1));
                    f.instruction(&W::LocalGet(L_T2));
                    f.instruction(&W::I32Const(1));
                    f.instruction(&W::I32Sub);
                    f.instruction(&W::LocalSet(L_T2));
                    f.instruction(&W::Br(0));
                    f.instruction(&W::End); // the loop
                    f.instruction(&W::End); // the block
                }
            }
            f.instruction(&W::End); // the bulk guard's `if`
            return;
        }
        Some(InlineLowering::GuestScaled { offset, max, shl, limit }) => {
            emit_pointer_guard(f, base, limit, index);
            // In range. Load the word, and hand the CLAMPED case back to the handler:
            // `word > max` is exactly where `read(p).min(cap) * k` stops being a shift.
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::I32Load(MemArg { offset: offset as u64, align: 0, memory_index: 0 }));
            f.instruction(&W::LocalTee(L_T1));
            f.instruction(&W::I32Const(max as i32));
            f.instruction(&W::I32GtU);
            f.instruction(&W::If(BlockType::Empty));
            f.instruction(&W::I32Const(index as i32));
            f.instruction(&W::Call(IMPORT_FUNC));
            f.instruction(&W::Else);
            f.instruction(&W::LocalGet(L_T1));
            if shl != 0 {
                f.instruction(&W::I32Const(shl as i32));
                f.instruction(&W::I32Shl);
            }
            f.instruction(&W::GlobalSet(abi::reg_global(0)));
            f.instruction(&W::End);
            f.instruction(&W::End); // the pointer guard's `if`
            return;
        }
        Some(InlineLowering::Guest { offset, shift, mask, plus, limit }) => {
            (offset, shift, mask, plus, limit)
        }
    };
    emit_pointer_guard(f, base, limit, index);
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
    // The bias, dropped entirely when it is zero - so every form that had no bias before
    // this existed still emits exactly the instructions it did.
    if plus != 0 {
        f.instruction(&W::I32Const(plus as i32));
        f.instruction(&W::I32Add);
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
