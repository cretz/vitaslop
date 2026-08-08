//! Vita-agnostic ARMv7-A + Thumb-2 (later NEON/VFP) to WASM transpiler.
//!
//! Input is a code image, entry-point addresses, and import/relocation facts
//! ([`Program`]). Output is a WASM module plus the guest-function-to-export map
//! ([`Artifact`]). Nothing Vita-specific belongs here.
//!
//! Pipeline: decode + discover ([`lower`]) -> per-function CFG IR ([`ir`]) ->
//! wasm emission ([`emit`]). Each guest function becomes one wasm function whose
//! body is a dispatch loop over its basic blocks; see [`emit`] and [`abi`] for
//! the control-flow and state model, and the crate README for the rationale.

pub mod abi;
mod emit;
/// The one emit-time diagnostic knob a HOST also needs to see: the frame at which
/// trapping diagnostics become live (`VITASLOP_ARM_AT_FRAME`). The host arms them by
/// writing [`Artifact::arm_word_off`] when the run reaches it.
pub use emit::arm_at_frame;
pub use emit::set_fuel_interval;
/// The fuel interval modules emitted on this thread carry, so a HOST can read the
/// software counter the emitted code maintains: the counter runs DOWN from this and
/// reloads to it, so only a host that knows the interval can difference it.
pub use emit::fuel_interval;
/// The guest-store DIRTY MAP: whether to emit it, and where a host reads it. The map
/// is one byte per 4 KB page holding the epoch of the last store into that page, laid
/// out at [`Artifact::dirty_off`] as `[epoch byte][map]` (see [`DIRTY_EPOCH_OFF`] and
/// [`DIRTY_MAP_OFF`]). A host that stamps its own reads against the epoch can prove a
/// region of guest memory unchanged without reading it.
pub use emit::{set_dirty_tracking, DIRTY_EPOCH_OFF, DIRTY_MAP_OFF, DIRTY_SHIFT};
mod ir;
mod lower;

use std::collections::BTreeMap;

pub use ir::ConditionCode;
use lower::Imports;

/// A code image to transpile: the ARM/Thumb blob, where it loads, which
/// addresses to start decoding from, how guest imports wire to host functions,
/// and the guest memory size.
pub struct Program<'a> {
    /// The ARM/Thumb code/data image.
    pub code: &'a [u8],
    /// Guest address the image loads at (and the linear-memory rebase origin).
    pub base: u32,
    /// Default decode mode for [`entries`](Self::entries) and their transitively
    /// discovered callees: true for Thumb-2, false for ARM. A title is overwhelmingly
    /// one mode (Vita user code is Thumb); the few functions in the other mode are
    /// listed in [`arm_entries`](Self::arm_entries).
    pub thumb: bool,
    /// Entry points to discover in the default [`thumb`](Self::thumb) mode (each
    /// becomes a function, transitively pulling in its direct callees).
    pub entries: &'a [u32],
    /// Entry points to discover in ARM mode regardless of [`thumb`](Self::thumb) -
    /// even code pointers a Thumb title reaches through a `blx` into an ARM stub.
    /// Discovered tentatively (a bad guess that fails to decode is dropped), like
    /// address-taken code pointers.
    pub arm_entries: &'a [u32],
    /// Extern (imported-function) wiring: each maps a guest stub address to a
    /// dense host-import index.
    pub externs: &'a [Extern],
    /// Inter-module redirects: each maps an import-stub address to the guest
    /// address of the function that satisfies it in another module of the same
    /// link. A `bl`/`b` to such a stub becomes a direct guest call to the target
    /// (no host trap), so a call across module boundaries is as cheap as one
    /// within a module. Empty for a single-module link.
    pub redirects: &'a [Redirect],
    /// Host imports to emit INLINE rather than as a host trap (see [`InlineImport`]).
    /// Empty leaves every import a host call, which is what every earlier build did
    /// and what the ARM conformance corpus still wants.
    pub inline_imports: &'a [InlineImport],
    /// Syscall numbers (guest r7) that do not return, so a `svc` with a
    /// statically-known one of them ends decoding (before trailing data).
    pub noreturn_svc: &'a [u32],
    /// Total guest memory to provision, in bytes from `base`. The host keeps all
    /// guest allocations (image, stack, heap) within `[base, base + mem_bytes)`.
    pub mem_bytes: u32,
    /// Also discover address-taken code pointers (functions materialized via
    /// `movw`/`movt` but never directly called - e.g. a thread entry passed to
    /// sceKernelCreateThread). Such candidates are transpiled tentatively: one
    /// that fails to decode is skipped, so a mis-identified constant can never
    /// break the build. Vita modules set this; the tightly controlled ARM
    /// conformance corpus leaves it off so its output stays exactly as before.
    pub discover_code_pointers: bool,
    /// Import a shared linear memory (`env.memory`) instead of defining one, so
    /// several instances of this module can share one guest address space while
    /// keeping independent register globals. Only the preemptive multi-thread
    /// scheduler (`vitaslop_native::ThreadedScheduler`) needs this; every single-
    /// instance host leaves it off and gets the original self-contained module.
    pub import_memory: bool,
}

/// A guest address that dispatches to a host import (the Vita NID mechanism): a
/// `bl`/`blx` to `addr` becomes a host call with dense index `import`.
pub struct Extern {
    pub addr: u32,
    pub import: u32,
}

/// An inter-module redirect: a `bl`/`b` to import stub `addr` is retargeted to
/// the guest function at `target` (an export of another module in the same link),
/// lowering to a direct guest call instead of a host import trap.
pub struct Redirect {
    pub addr: u32,
    pub target: u32,
    /// The instruction set `target` is written in, taken from the Thumb bit of the
    /// exported address (which `target` itself has cleared). It cannot be inferred
    /// from the call site: the caller reaches an ARM veneer, and the veneer is not
    /// what runs. A wrongly-moded callee decodes into a different function without
    /// erroring, so this has to travel with the redirect.
    pub thumb: bool,
}

/// A host import whose whole behaviour is one guest-memory read, emitted INLINE
/// instead of trapping to the host.
///
/// # Why
/// Some system calls are pure accessors over a guest structure - the GXM shader
/// reflection getters are the clearest case: given a `SceGxmProgramParameter *` they
/// return one bitfield of one word. On a real title those dominate the host-call
/// traffic by a wide margin (measured: four of them are 60% of every host call a
/// gameplay frame makes), and each costs a wasm-to-host transition plus marshalling
/// the guest register file both ways, to compute a shift and a mask.
///
/// Emitting the load directly turns each into a handful of wasm instructions with no
/// boundary crossing at all.
///
/// # Staying exactly equivalent
/// An inlined call must be indistinguishable from the host one, including on the
/// error paths, so the emitted code FALLS BACK to the real host call whenever the
/// address is not one the inline load can serve (below the image base, or too near
/// the end of guest memory) - see [`emit`]. The host handler remains the definition
/// of the behaviour; the inline form is an optimisation of the common case only.
///
/// # The one observable difference
/// An inlined call never reaches the host, so it does NOT appear in the host-call
/// trace, the call histogram, or `Capture::call_count`. That is a deliberate
/// trade and a safe one for what is inlined here - the ordered-timeline tracer
/// already filters `sceGxmProgram*` out as noise - but it means a NID must not be
/// inlined if anyone would go looking for it in a trace. It does not touch the
/// determinism signature, which folds the render stream and the egress ledger, not
/// the call trace.
pub struct InlineImport {
    /// The dense host-import index this replaces (the same index `Extern::import`
    /// carries).
    pub import: u32,
    pub op: InlineOp,
}

/// What an [`InlineImport`] computes.
///
/// Deliberately a tiny closed set rather than a general expression language: every
/// operation admitted here has to be proven equivalent to its host handler, and a
/// small set is what makes that provable. Widen it only with a matching test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InlineOp {
    /// `r0 = (u32_at(r0 + offset) >> shift) & mask` - read a word at a fixed offset
    /// from the pointer argument and extract a bitfield. `mask` is applied after the
    /// shift, so `mask = u32::MAX` means "the whole word".
    LoadShiftMask { offset: u32, shift: u32, mask: u32 },
    /// `r0 = u32_at(mirror_base + slot * 4)` - read slot `slot` of the HOST MIRROR
    /// block, a small run of words the host keeps up to date in linear memory (see
    /// [`Artifact::mirror_off`]). Takes no guest argument and reads nothing the guest
    /// owns.
    ///
    /// This covers the other shape of pure accessor: one that returns a value of the
    /// SYSTEM rather than of a guest structure. `sceDisplayGetVcount` is the case that
    /// motivated it - a vblank spin calls it tens of thousands of times a frame, and
    /// its answer is a pure function of the virtual clock, which can only change while
    /// no guest code is running. So a word the host refreshes at every such point is
    /// not an approximation of the call, it is the call's answer.
    ///
    /// The host side of that contract is the whole correctness argument, and it is not
    /// optional: a module emitted with any `LoadMirror` op must be run on a host that
    /// refreshes the block before it resumes guest code. A host that does not is a
    /// configuration error, and callers are expected to reject it outright rather than
    /// let the guest read a stale word.
    LoadMirror { slot: u32 },
    /// `r0 = u32_at(r0 + offset) << shl`, but ONLY when the loaded word is `<= max`;
    /// otherwise the real host call runs.
    ///
    /// The value guard is what makes this exact rather than nearly exact. It exists for
    /// a handler shaped `read(p).min(cap) * k` - a size read out of a structure, scaled,
    /// and clamped so a header we failed to resolve cannot ask for an absurd allocation.
    /// The clamp is not expressible as a shift and a mask, so instead of approximating
    /// it the inline form handles only the unclamped case and hands the other one back
    /// to the handler, which stays the definition of the answer. Same principle as the
    /// pointer guard in [`emit`](crate::emit): fall back rather than approximate.
    LoadScaled { offset: u32, max: u32, shl: u32 },
    /// `u32_at(r0) = mirror[slot]; u32_at(r0 + 4) = mirror[slot + 1]; r0 = 0` - store a
    /// 64-bit host-mirror value THROUGH the guest pointer in r0, then return success.
    ///
    /// The out-parameter twin of [`InlineOp::LoadMirror`], and it exists because the
    /// most-called clock reader on a real title does not return its answer in a
    /// register: `sceKernelGetProcessTime(SceKernelSysClock *)` writes it through a
    /// pointer. Being an out-parameter is a calling convention, not behaviour - the
    /// value written is the same pure function of the virtual clock - so the same
    /// mirror contract covers it.
    ///
    /// Guarded like every other pointer form: an out-of-range pointer runs the handler.
    StoreMirrorPair { slot: u32 },
    /// `r0 = mirror[slot]; r1 = mirror[slot + 1]` - read a 64-bit host-mirror value into
    /// the ARM EABI's 64-bit return pair.
    ///
    /// The register-returning twin of [`InlineOp::StoreMirrorPair`], for the wide
    /// spelling of the same clock read. No pointer, so no guard.
    LoadMirrorPair { slot: u32 },
}

impl InlineOp {
    /// The operation's meaning, over the word it reads. The single definition of what
    /// the emitted code must compute, so a test can hold a host handler to it.
    ///
    /// For the pair forms this is the LOW word only; the high word is `mirror[slot + 1]`
    /// unchanged, which needs no definition. For [`InlineOp::LoadScaled`] this is
    /// meaningful only when [`InlineOp::falls_back`] is false - the guarded case has no
    /// inline answer by construction.
    pub fn eval(self, word: u32) -> u32 {
        match self {
            InlineOp::LoadShiftMask { shift, mask, .. } => (word >> shift) & mask,
            // The mirror word IS the answer; the host computed it.
            InlineOp::LoadMirror { .. } => word,
            InlineOp::LoadScaled { shl, .. } => word << shl,
            // The pair forms deliver the mirror words untouched, wherever they land.
            InlineOp::StoreMirrorPair { .. } | InlineOp::LoadMirrorPair { .. } => word,
        }
    }

    /// Whether the loaded `word` sends this op to the host call instead of computing an
    /// answer inline. Only [`InlineOp::LoadScaled`] can, and a test pins the boundary:
    /// the point of the guard is that the handler, not this, defines the clamped case.
    pub fn falls_back(self, word: u32) -> bool {
        match self {
            InlineOp::LoadScaled { max, .. } => word > max,
            _ => false,
        }
    }

    /// Byte offset from the pointer argument at which the word is read, for the forms
    /// that read through a guest pointer. `None` for a form that does not take one.
    pub fn offset(self) -> Option<u32> {
        match self {
            InlineOp::LoadShiftMask { offset, .. } => Some(offset),
            InlineOp::LoadScaled { offset, .. } => Some(offset),
            // Writes through r0 rather than reading through it, so it has no read offset
            // even though it is a pointer form. `emit_import` guards it on its own terms.
            InlineOp::StoreMirrorPair { .. } => None,
            InlineOp::LoadMirror { .. } | InlineOp::LoadMirrorPair { .. } => None,
        }
    }

    /// The host-mirror slot this op reads, if it reads one. For a pair form this is the
    /// LOW slot; the layout must also reserve `slot + 1`, which
    /// [`InlineOp::top_mirror_slot`] reports.
    pub fn mirror_slot(self) -> Option<u32> {
        match self {
            InlineOp::LoadShiftMask { .. } | InlineOp::LoadScaled { .. } => None,
            InlineOp::LoadMirror { slot } => Some(slot),
            InlineOp::StoreMirrorPair { slot } | InlineOp::LoadMirrorPair { slot } => Some(slot),
        }
    }

    /// The HIGHEST mirror slot this op touches, which is what the memory layout must
    /// size the block against. A pair form reads two words, so sizing the block from
    /// [`InlineOp::mirror_slot`] alone would leave its high word off the end of the
    /// reserved page - a read of whatever follows, which for a clock is a garbage
    /// timestamp rather than an obvious failure.
    pub fn top_mirror_slot(self) -> Option<u32> {
        match self {
            InlineOp::StoreMirrorPair { slot } | InlineOp::LoadMirrorPair { slot } => Some(slot + 1),
            other => other.mirror_slot(),
        }
    }
}

/// The transpiler output: the WASM blob plus the map the runtime needs to enter
/// guest code by address.
pub struct Artifact {
    /// The emitted WASM module bytes.
    pub wasm: Vec<u8>,
    /// One entry per transpiled function: its guest address and wasm export name.
    pub funcs: Vec<FuncExport>,
    /// Total linear-memory pages the module declares (the guest region plus the
    /// appended indirect-dispatch address table). A host that imports a shared memory
    /// must create it with exactly this many pages; a host that lets the module define
    /// its own memory can ignore this. See [`emit::EmitOutput`].
    pub mem_pages: u32,
    /// Linear-memory offset of the "diagnostics armed" word, present only when this
    /// build was emitted with `VITASLOP_ARM_AT_FRAME` (see [`emit::arm_at_frame`]).
    pub arm_word_off: Option<u64>,
    /// Linear-memory byte offset of the HOST MIRROR block, present only when some
    /// inline import reads it ([`InlineOp::LoadMirror`]). Slot `n` is the word at
    /// `mirror_off + n * 4`.
    ///
    /// A host running this module MUST keep those words current - see
    /// [`InlineOp::LoadMirror`]. `Some` here is therefore a REQUIREMENT on the host,
    /// not an optional extra: a host that cannot refresh the block must refuse to run
    /// the module rather than let the guest read a word that never changes.
    pub mirror_off: Option<u64>,
    /// Linear-memory byte offset of the GUEST-STORE DIRTY MAP, present only when this
    /// build was emitted with store tracking on ([`emit::dirty_tracking`]). One byte
    /// per 4 KB page of the whole linear memory, set to 1 by every translated store.
    ///
    /// A host may read it or ignore it; what it may NOT do is assume a page is clean
    /// in a build that has no map. `None` means "this module tracks nothing", which is
    /// why it is an `Option` rather than an offset of zero.
    pub dirty_off: Option<u64>,
}

/// A transpiled function: the guest address it starts at and its wasm export.
pub struct FuncExport {
    pub addr: u32,
    pub export: String,
}

/// Why transpilation failed.
#[derive(Debug)]
pub enum Error {
    /// The decoder could not decode the bytes at `addr`.
    Decode { addr: u32 },
    /// A decoded instruction is not lifted yet.
    Unsupported {
        addr: u32,
        opcode: yaxpeax_arm::armv7::Opcode,
    },
    /// An operand had an unexpected shape for its opcode.
    Operand { addr: u32 },
}

/// The linker inserts Thumb->ARM interworking veneers so Thumb code (e.g. newlib,
/// linked without -nostdlib) can `bl` an ARM import stub. Each veneer has a fixed
/// five-word shape:
///   bx   pc          ; 0x4778     Thumb -> ARM, pc word-aligned to veneer+4
///   b.n  .-2         ; 0xe7fd     never executed (bx pc reads pc = veneer+4)
///   ldr  ip, [pc]    ; 0xe59fc000 ip = the offset word below
///   add  pc, ip, pc  ; 0xe08cf00f pc(here)+8 + ip  ->  stub = veneer+16+off
///   .word off
/// Resolve each veneer to its stub and, when the stub is a known import, return
/// `(veneer_addr, import_index)`. Aliasing the veneer's entry to that import lets
/// a `bl veneer` lower straight to the import call - no ARM lifting and no
/// computed-branch support needed. Artifacts with no veneers (our -nostdlib ones,
/// which `blx` the stub directly) yield nothing here, so this is purely additive.
fn scan_veneers(code: &[u8], base: u32, imports: &BTreeMap<u32, u32>) -> Vec<(u32, u32)> {
    let rd16 = |o: usize| u16::from_le_bytes([code[o], code[o + 1]]);
    let rd32 = |o: usize| u32::from_le_bytes([code[o], code[o + 1], code[o + 2], code[o + 3]]);
    let mut out = Vec::new();
    let mut o = 0usize;
    while o + 16 <= code.len() {
        if rd16(o) == 0x4778 && rd32(o + 4) == 0xe59f_c000 && rd32(o + 8) == 0xe08c_f00f {
            let veneer = base.wrapping_add(o as u32);
            let stub = veneer.wrapping_add(16).wrapping_add(rd32(o + 12));
            if let Some(&idx) = imports.get(&stub) {
                out.push((veneer, idx));
            }
        }
        o += 2;
    }
    out
}

/// Whether the halfword at `off` opens a Thumb function: one of the `push` encodings a
/// non-leaf function starts with (`push {..}` / `push {.., lr}` / the wide
/// `stmdb sp!, {..}`). Used to filter GUESSED function pointers - everything that passes
/// is still only seeded tentatively.
fn looks_like_thumb_entry(code: &[u8], base: u32, target: u32) -> bool {
    let Some(off) = target.checked_sub(base).map(|d| d as usize) else { return false };
    if target & 1 != 0 || off + 2 > code.len() {
        return false;
    }
    let first = u16::from_le_bytes([code[off], code[off + 1]]);
    (first & 0xFE00) == 0xB400 || first == 0xE92D
}

/// Scan the whole image's CODE for `movw`/`movt` pairs that materialize a Thumb function
/// pointer, and return the targets.
///
/// This is the companion to [`scan_stored_code_pointers`], and it exists because a
/// function pointer is not always in a table: a callback registered at runtime, a C++
/// lambda captured into a heap object, a `std::function` - all of these BUILD the address
/// in code with `movw`/`movt` and store it somewhere no static scan can see.
/// `Program::discover_code_pointers` already picks those up, but only inside functions
/// that were themselves discovered, so a chain of pointer-reached code stays invisible:
/// the producer is only called through a pointer, so it is never discovered, so the
/// pointer it produces is never seen, and the trail goes cold at the first link.
/// Scanning the image linearly does not care how the producer is reached.
///
/// Both `movw`/`movt` are the Thumb-2 T3/T1 encodings:
/// `movw`: `11110 i 100100 imm4 | 0 imm3 Rd imm8`, `movt` the same with bit 7 of the
/// first halfword set, and the value is `imm4:i:imm3:imm8`.
fn scan_materialized_code_pointers(code: &[u8], base: u32) -> Vec<u32> {
    /// How far after a `movw` its `movt` may sit. A compiler emits them adjacent or with
    /// a couple of instructions between; a wide window only adds false pairings.
    const PAIR_WINDOW: usize = 16;
    let rd16 = |o: usize| u16::from_le_bytes([code[o], code[o + 1]]);
    // (Rd, imm16) of a movw/movt at `o`, if it is one.
    let parts = |o: usize, movt: bool| -> Option<(u8, u32)> {
        if o + 4 > code.len() {
            return None;
        }
        let hw1 = rd16(o);
        let want = if movt { 0xF2C0 } else { 0xF240 };
        if hw1 & 0xFBF0 != want {
            return None;
        }
        let hw2 = rd16(o + 2);
        if hw2 & 0x8000 != 0 {
            return None; // not the second halfword of a 32-bit data-processing op
        }
        let i = ((hw1 >> 10) & 1) as u32;
        let imm4 = (hw1 & 0xF) as u32;
        let imm3 = ((hw2 >> 12) & 7) as u32;
        let imm8 = (hw2 & 0xFF) as u32;
        Some((((hw2 >> 8) & 0xF) as u8, (imm4 << 12) | (i << 11) | (imm3 << 8) | imm8))
    };

    let mut out = Vec::new();
    let mut o = 0usize;
    while o + 4 <= code.len() {
        if let Some((rd, lo)) = parts(o, false) {
            let mut p = o + 4;
            while p + 4 <= code.len() && p < o + 4 + PAIR_WINDOW * 2 {
                if let Some((rd2, hi)) = parts(p, true) {
                    if rd2 == rd {
                        let value = (hi << 16) | lo;
                        if value & 1 == 1 && looks_like_thumb_entry(code, base, value & !1) {
                            out.push(value & !1);
                        }
                        break;
                    }
                }
                p += 2;
            }
        }
        o += 2;
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// `(entry, last block address)` for every discovered function, so a sweep candidate that
/// lands inside one already-understood function can be skipped. Approximate on purpose:
/// the last block's own length is not counted, which keeps the test to two comparisons
/// and can only ever admit a candidate, never wrongly reject one.
fn discovered_spans(funcs: &BTreeMap<u32, ir::Func>) -> Vec<(u32, u32)> {
    funcs
        .values()
        .filter_map(|f| Some((f.addr, f.blocks.last()?.addr)))
        .collect()
}

/// Every address in the image that opens like a Thumb function, in ascending order.
///
/// The last resort for finding code nothing points at *that we can see*. A title's
/// pointer-reached code forms chains - a handler registered by a function that is itself
/// only registered - and the call graph, the image's pointer tables and its `movw`/`movt`
/// materializations between them still do not always reach the far end. A linear sweep
/// does not care how code is reached.
///
/// Everything here is a guess, so callers seed it TENTATIVELY (a candidate that fails to
/// decode, or decodes into something malformed, is dropped) and skip anything already
/// inside a discovered function - a `push` in the middle of a real function is not a
/// second function, and emitting it as one would put a bogus entry in the dispatch table.
fn sweep_thumb_entries(code: &[u8], base: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut o = 0usize;
    while o + 2 <= code.len() {
        let addr = base.wrapping_add(o as u32);
        if looks_like_thumb_entry(code, base, addr) {
            out.push(addr);
        }
        o += 2;
    }
    out
}

/// Scan the whole image for stored Thumb function POINTERS and return their targets.
///
/// A C++ vtable, a table of state handlers, a registered callback saved into a struct
/// initializer - all of these put a function's address in DATA, and a dispatch through
/// one is the single thing a call-graph walk cannot follow. `discover_code_pointers`
/// finds the pointers a function MATERIALIZES with `movw`/`movt`; this finds the ones the
/// linker wrote into the image.
///
/// The filter is deliberately narrow, because everything found here is guessed: a
/// candidate must be a 4-byte-aligned word, odd (a Thumb pointer), in bounds, and land on
/// a plausible function prologue - one of the `push` encodings that opens a non-leaf
/// function. Everything that survives is seeded as TENTATIVE, so a guess that fails to
/// decode, or decodes into something malformed, is dropped rather than emitted.
///
/// A leaf function whose address is only ever stored in data and which does not open with
/// a `push` is still missed. That is the honest trade: a wider filter guesses more, and a
/// dispatch to an address we did not discover already fails loudly rather than silently.
fn scan_stored_code_pointers(code: &[u8], base: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut o = 0usize;
    while o + 4 <= code.len() {
        let w = u32::from_le_bytes([code[o], code[o + 1], code[o + 2], code[o + 3]]);
        o += 4;
        if w & 1 == 0 {
            continue;
        }
        let target = w & !1;
        if looks_like_thumb_entry(code, base, target) {
            out.push(target);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Transpile `program` into a WASM module and its dispatch map.
/// One pending discovery target: its address, whether it is `tentative` (a guessed
/// code pointer that is dropped on a decode failure rather than erroring), and which
/// `thumb`/ARM mode to decode it in.
struct WorkItem {
    addr: u32,
    tentative: bool,
    thumb: bool,
}

impl WorkItem {
    /// A hard target (a decode failure is a real error), decoded in `thumb` mode.
    fn hard(addr: u32, thumb: bool) -> WorkItem {
        WorkItem { addr, tentative: false, thumb }
    }
    /// A tentative target (dropped on a decode failure), decoded in `thumb` mode.
    fn tentative(addr: u32, thumb: bool) -> WorkItem {
        WorkItem { addr, tentative: true, thumb }
    }
}

/// The initial discovery worklist: hard entries in the program's default mode, plus
/// tentative ARM entries (even code pointers reached through a `blx` into ARM code).
fn seed_worklist(program: &Program) -> Vec<WorkItem> {
    let mut work: Vec<WorkItem> =
        program.entries.iter().map(|&a| WorkItem::hard(a, program.thumb)).collect();
    work.extend(program.arm_entries.iter().map(|&a| WorkItem::tentative(a, false)));
    work
}

pub fn transpile(program: &Program) -> Result<Artifact, Error> {
    let mut import_map: BTreeMap<u32, u32> =
        program.externs.iter().map(|e| (e.addr, e.import)).collect();
    // Alias Thumb->ARM interworking veneers to the imports they trampoline to.
    for (veneer, idx) in scan_veneers(program.code, program.base, &import_map) {
        import_map.insert(veneer, idx);
    }
    let redirect_map: BTreeMap<u32, (u32, bool)> =
        program.redirects.iter().map(|r| (r.addr, (r.target, r.thumb))).collect();
    let imports = Imports::new(&import_map, &redirect_map);

    // Discover the transitive closure from the entries: direct callees are hard
    // (a decode failure is a real bug and propagates), while address-taken code
    // pointers are tentative (a mis-identified constant that fails to decode is
    // silently skipped, never breaking the build).
    let mut funcs: BTreeMap<u32, ir::Func> = BTreeMap::new();
    let mut work = seed_worklist(program);
    while let Some(WorkItem { addr, tentative, thumb }) = work.pop() {
        if funcs.contains_key(&addr) {
            continue;
        }
        // Never lift an import/redirect stub's unresolved placeholder as a function
        // (see the note in `transpile_lenient`): a dispatch that reaches an unresolved
        // stub must trap loudly, not silently run the `mvn r0,#0; bx lr` no-op.
        if import_map.contains_key(&addr) || redirect_map.contains_key(&addr) {
            continue;
        }
        let found = match lower::discover(
            program.code,
            program.base,
            addr,
            thumb,
            &imports,
            program.noreturn_svc,
            program.discover_code_pointers,
            false,
        ) {
            Ok(found) => found,
            // A tentative code pointer that does not decode was not a function;
            // drop it. A hard callee failure is a genuine error.
            Err(_) if tentative => continue,
            Err(e) => return Err(e),
        };
        // A tentative guess that decoded into a malformed function (a branch to a
        // non-block - common when ARM-decoding data or a Thumb address) was never a
        // real function; drop it so it never reaches emission.
        if tentative && !found.func.well_formed() {
            continue;
        }
        // Direct callees inherit this function's tentativeness (a callee reached only
        // from a guess is itself a guess) but NOT its mode: `discover` reports the mode
        // per callee, because `blx <label>` interworks and an ARM callee decoded as
        // Thumb silently becomes a different function rather than failing. Discovered
        // code pointers are tentative Thumb (materialized with the Thumb bit set).
        let callee: fn(u32, bool) -> WorkItem = if tentative { WorkItem::tentative } else { WorkItem::hard };
        work.extend(found.callees.into_iter().map(|(a, t)| callee(a, t)));
        work.extend(found.code_pointers.into_iter().map(|a| WorkItem::tentative(a, true)));
        work.extend(found.arm_code_pointers.into_iter().map(|a| WorkItem::tentative(a, false)));
        funcs.insert(addr, found.func);
    }

    // Assign wasm function indices (imports occupy the low indices) in ascending
    // address order, and build the address -> index map for call lowering.
    let ordered: Vec<ir::Func> = funcs.into_values().collect();
    let func_index: BTreeMap<u32, u32> = ordered
        .iter()
        .enumerate()
        .map(|(i, f)| (f.addr, emit::IMPORT_FUNCS + i as u32))
        .collect();

    let emit::EmitOutput { wasm, mem_pages, arm_word_off, mirror_off, dirty_off } =
        emit::emit_module(
            &ordered,
            &func_index,
            program.base,
            program.mem_bytes,
            program.inline_imports,
            program.import_memory,
        );
    let funcs = ordered
        .iter()
        .map(|f| FuncExport {
            addr: f.addr,
            export: abi::func_export(f.addr),
        })
        .collect();
    Ok(Artifact { wasm, funcs, mem_pages, arm_word_off, mirror_off, dirty_off })
}

/// The output of a lenient whole-program build ([`transpile_lenient`]): the module
/// plus the addresses that were emitted as trapping stubs.
pub struct LenientArtifact {
    pub artifact: Artifact,
    /// Guest addresses of functions that could not be lowered and became trapping
    /// stubs. Empty means a fully-translated program. Any of these reached at
    /// runtime faults (a real, visible signal that the code is on the hot path).
    pub stubbed: Vec<u32>,
    /// The wasm function indices of the stubs (parallel to [`stubbed`](Self::stubbed)),
    /// so a runtime `unreachable` trap backtrace ("wasm function N") can be told
    /// apart from a genuine miscompile - a stub is expected, anything else is a bug.
    pub stub_wasm_indices: Vec<u32>,
}

/// Transpile like [`transpile`], but never abort: a function that fails to lower
/// becomes a trapping stub instead of an error, so the whole program still emits a
/// valid, runnable module. This is what lets a real title boot while a handful of
/// exotic instructions remain unlifted - the boot path proceeds, and any stub it
/// actually calls faults loudly instead of silently mistranslating. Discovery from
/// a stubbed function's body stops (its callees are unknown), which is correct: we
/// could not decode past the unlifted instruction anyway.
pub fn transpile_lenient(program: &Program) -> LenientArtifact {
    let mut import_map: BTreeMap<u32, u32> =
        program.externs.iter().map(|e| (e.addr, e.import)).collect();
    for (veneer, idx) in scan_veneers(program.code, program.base, &import_map) {
        import_map.insert(veneer, idx);
    }
    let redirect_map: BTreeMap<u32, (u32, bool)> =
        program.redirects.iter().map(|r| (r.addr, (r.target, r.thumb))).collect();
    let imports = Imports::new(&import_map, &redirect_map);

    let mut funcs: BTreeMap<u32, ir::Func> = BTreeMap::new();
    let mut stubbed = Vec::new();
    let mut work = seed_worklist(program);
    // A redirect's target is an ordinary guest function, and it is reachable even when no
    // direct call to its stub exists - a vtable slot can hold the stub's address alone.
    // Seed the targets so the thunks added below always have something to call.
    work.extend(program.redirects.iter().map(|r| WorkItem::tentative(r.target, r.thumb)));
    // Function pointers the call graph cannot reach: the ones the linker wrote into the
    // image (vtables, handler tables) and the ones code materializes with `movw`/`movt`
    // anywhere in the image, discovered or not. Both are guesses, so both are tentative.
    if program.discover_code_pointers {
        let guessed = scan_stored_code_pointers(program.code, program.base)
            .into_iter()
            .chain(scan_materialized_code_pointers(program.code, program.base));
        work.extend(guessed.map(|a| WorkItem::tentative(a, true)));
    }
    // Two rounds: the call graph and everything pointed at, then - only over what is
    // still unaccounted for - the linear prologue sweep. The sweep runs second because it
    // is the weakest evidence there is, and running it last means it only ever proposes
    // code that no stronger route already claimed.
    for round in 0..2 {
        if round == 1 {
            if !program.discover_code_pointers {
                break;
            }
            let claimed = discovered_spans(&funcs);
            work.extend(
                sweep_thumb_entries(program.code, program.base)
                    .into_iter()
                    .filter(|a| !claimed.iter().any(|&(lo, hi)| (lo..=hi).contains(a)))
                    .map(|a| WorkItem::tentative(a, true)),
            );
        }
        while let Some(WorkItem { addr, tentative, thumb }) = work.pop() {
            if funcs.contains_key(&addr) {
                continue;
            }
            // Never lift an import or redirect stub's unresolved placeholder (`mvn r0,#0;
            // bx lr`) as a function - that would make a dispatch to it a silent no-op
            // returning -1 (e.g. a `memset` reached through a function pointer doing
            // nothing). Direct calls to a stub are already resolved to the import/callee,
            // and register-indirect calls with a tracked target are too (see
            // `lower::discover`). Stubs get real thunks after discovery instead.
            if import_map.contains_key(&addr) || redirect_map.contains_key(&addr) {
                continue;
            }
            match lower::discover(
                program.code,
                program.base,
                addr,
                thumb,
                &imports,
                program.noreturn_svc,
                program.discover_code_pointers,
                true,
            ) {
                // A tentative guess that decoded into a malformed function (a branch to
                // a non-block) was never real; drop it before it can be emitted.
                Ok(found) if tentative && !found.func.well_formed() => {}
                Ok(found) => {
                    let callee: fn(u32, bool) -> WorkItem =
                        if tentative { WorkItem::tentative } else { WorkItem::hard };
                    work.extend(found.callees.into_iter().map(|(a, t)| callee(a, t)));
                    work.extend(found.code_pointers.into_iter().map(|a| WorkItem::tentative(a, true)));
                    work.extend(found.arm_code_pointers.into_iter().map(|a| WorkItem::tentative(a, false)));
                    funcs.insert(addr, found.func);
                }
                // A tentative code pointer that does not decode was never a function.
                Err(_) if tentative => {}
                // A hard callee we cannot lower becomes a trapping stub, so the rest of
                // the program still builds and runs.
                Err(_) => {
                    stubbed.push(addr);
                    funcs.insert(addr, ir::Func::new_stub(addr));
                }
            }
        }
    }

    // Give every import and redirect stub a callable thunk. A DYNAMIC pointer to one - a
    // vtable slot, a registered callback - reaches the stub's address with nothing for
    // the indirect dispatcher to land on, because a direct call was resolved at lift time
    // and the placeholder bytes were deliberately not lifted (above). Thunks are one
    // statement each and are emitted for every stub rather than only the reachable ones,
    // because which stubs a title takes the address of is exactly what static discovery
    // cannot know. A redirect whose target did not survive discovery gets no thunk, so a
    // dispatch to it stays a loud miss rather than a call into nothing.
    for (&addr, &import) in &import_map {
        funcs.insert(addr, ir::Func::new_import_thunk(addr, true, import));
    }
    for (&addr, &(target, _)) in &redirect_map {
        if import_map.contains_key(&addr) || !funcs.contains_key(&target) {
            continue;
        }
        funcs.insert(addr, ir::Func::new_redirect_thunk(addr, true, target));
    }

    let ordered: Vec<ir::Func> = funcs.into_values().collect();
    let func_index: BTreeMap<u32, u32> = ordered
        .iter()
        .enumerate()
        .map(|(i, f)| (f.addr, emit::IMPORT_FUNCS + i as u32))
        .collect();
    let emit::EmitOutput { wasm, mem_pages, arm_word_off, mirror_off, dirty_off } = emit::emit_module(
        &ordered,
        &func_index,
        program.base,
        program.mem_bytes,
        program.inline_imports,
        program.import_memory,
    );
    let funcs = ordered
        .iter()
        .map(|f| FuncExport { addr: f.addr, export: abi::func_export(f.addr) })
        .collect();
    stubbed.sort_unstable();
    let stub_wasm_indices = stubbed.iter().map(|a| func_index[a]).collect();
    LenientArtifact {
        artifact: Artifact { wasm, funcs, mem_pages, arm_word_off, mirror_off, dirty_off },
        stubbed,
        stub_wasm_indices,
    }
}

/// Diagnostic: discover the single function at `addr` and return its lowered IR
/// (blocks, statements, terminators) as a human-readable string, or `None` if it
/// does not decode. This is the authoritative decode/lowering the emitter uses, so
/// a trap's `guest_block` (from `VITASLOP_TRACK_PC`) can be read against the exact
/// statements - unlike a naive linear disassembly, which misaligns after any op the
/// decoder skips. `addr`'s Thumb bit is ignored; ARM mode is used when the address
/// is one of the program's [`arm_entries`](Program::arm_entries).
pub fn dump_func(program: &Program, addr: u32) -> Option<String> {
    use std::fmt::Write;
    let addr = addr & !1;
    let mut import_map: BTreeMap<u32, u32> =
        program.externs.iter().map(|e| (e.addr, e.import)).collect();
    for (veneer, idx) in scan_veneers(program.code, program.base, &import_map) {
        import_map.insert(veneer, idx);
    }
    let redirect_map: BTreeMap<u32, (u32, bool)> =
        program.redirects.iter().map(|r| (r.addr, (r.target, r.thumb))).collect();
    let imports = Imports::new(&import_map, &redirect_map);
    let seeded_thumb = !program.arm_entries.contains(&addr);
    let disc = |thumb: bool| {
        lower::discover(
            program.code,
            program.base,
            addr,
            thumb,
            &imports,
            program.noreturn_svc,
            program.discover_code_pointers,
            true,
        )
    };
    // Try the mode the program seeded this address in first, then the other mode, so
    // an ARM/Thumb misclassification is visible (a function stubbed only because it
    // was decoded in the wrong mode still dumps cleanly in the right one).
    let (thumb, found) = match disc(seeded_thumb) {
        Ok(f) => (seeded_thumb, f),
        Err(e_seeded) => {
            // The seeded mode is the one the program actually uses; surface its error
            // even if the other mode happens to decode into a garbage function.
            let other = disc(!seeded_thumb);
            return Some(format!(
                "== g_{addr:08x}: seeded thumb={seeded_thumb} FAILED: {e_seeded:?}; \
                 other mode: {} ==\n",
                match other {
                    Ok(f) => format!("decoded {} blocks", f.func.blocks.len()),
                    Err(e) => format!("also failed: {e:?}"),
                }
            ));
        }
    };
    let f = found.func;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "== function g_{:08x} (seeded_thumb={seeded_thumb}, decoded thumb={thumb}, {} blocks, stub={}) ==",
        f.addr,
        f.blocks.len(),
        f.stub
    );
    for b in &f.blocks {
        let _ = writeln!(s, "  block {:#010x}:", b.addr);
        for st in &b.stmts {
            let _ = writeln!(s, "    {st:?}");
        }
        let _ = writeln!(s, "    term: {:?}", b.term);
    }
    Some(s)
}

/// A single function the report could not translate: the discovery root it was
/// reached from, and the error (whose `addr` pinpoints the offending instruction).
pub struct ReportFailure {
    pub root: u32,
    pub error: Error,
}

/// The result of a resilient, diagnostic-only discovery pass: how many functions
/// translated cleanly and the ones that failed.
pub struct Report {
    /// Guest addresses of functions that discovered and lowered without error.
    pub ok: Vec<u32>,
    /// Functions that failed to translate (skipped rather than aborting).
    pub failures: Vec<ReportFailure>,
}

/// Diagnostic: attempt to discover and lower every reachable function, but on a
/// failure record it and continue instead of aborting at the first one. Unlike
/// [`transpile`] this never emits a module and never returns `Err` - it is purely
/// for sizing what CPU work remains (which instructions block which functions),
/// so bring-up can see the whole gap at once rather than one instruction per run.
pub fn transpile_report(program: &Program) -> Report {
    let import_map: BTreeMap<u32, u32> =
        program.externs.iter().map(|e| (e.addr, e.import)).collect();
    let redirect_map: BTreeMap<u32, (u32, bool)> =
        program.redirects.iter().map(|r| (r.addr, (r.target, r.thumb))).collect();
    let imports = Imports::new(&import_map, &redirect_map);

    let mut done: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut ok = Vec::new();
    let mut failures = Vec::new();
    let mut work = seed_worklist(program);
    while let Some(WorkItem { addr, tentative, thumb }) = work.pop() {
        if !done.insert(addr) {
            continue;
        }
        match lower::discover(
            program.code,
            program.base,
            addr,
            thumb,
            &imports,
            program.noreturn_svc,
            program.discover_code_pointers,
            false,
        ) {
            Ok(found) if tentative && !found.func.well_formed() => {}
            Ok(found) => {
                let callee: fn(u32, bool) -> WorkItem =
                    if tentative { WorkItem::tentative } else { WorkItem::hard };
                work.extend(found.callees.into_iter().map(|(a, t)| callee(a, t)));
                work.extend(found.code_pointers.into_iter().map(|a| WorkItem::tentative(a, true)));
                work.extend(found.arm_code_pointers.into_iter().map(|a| WorkItem::tentative(a, false)));
                ok.push(addr);
            }
            // A tentative code pointer that does not decode was never a function.
            Err(_) if tentative => {}
            Err(error) => failures.push(ReportFailure { root: addr, error }),
        }
    }
    Report { ok, failures }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Software fuel must land on LOOP BACK EDGES and nowhere else. Both halves matter
    /// and they fail in opposite directions: a missing fuel point on a back edge is the
    /// browser livelock this feature exists to fix, and a spurious one on a forward
    /// branch is a tax on straight-line guest code, which is most of it.
    ///
    /// Matched structurally on the emitted call - `i32.const FUEL_SELECTOR` (a one-byte
    /// signed LEB `-1`) followed by `call $host_import` - and asserted by COUNT, so the
    /// test fails loudly if the shape ever stops being emitted rather than quietly
    /// passing on a pattern that no longer matches anything.
    #[test]
    fn fuel_is_emitted_on_back_edges_and_only_on_back_edges() {
        // `i32.const -1` (0x41 0x7f) then `call 1` (0x10 0x01) - see abi::FUEL_SELECTOR
        // and emit's IMPORT_FUNC.
        const FUEL_CALL: [u8; 4] = [0x41, 0x7f, 0x10, 0x01];
        fn fuel_points(wasm: &[u8]) -> usize {
            wasm.windows(FUEL_CALL.len()).filter(|w| *w == FUEL_CALL).count()
        }
        let build = |code: &[u8], fuel: u32| -> Vec<u8> {
            set_fuel_interval(fuel);
            let a = transpile(&Program {
                code,
                base: 0x10000,
                thumb: false,
                entries: &[0x10000],
                arm_entries: &[],
                externs: &[],
                redirects: &[],
                inline_imports: &[],
                noreturn_svc: &[],
                mem_bytes: 0x20000,
                discover_code_pointers: false,
                import_memory: false,
            })
            .expect("transpile");
            wasmparser::validate(&a.wasm).expect("valid wasm");
            set_fuel_interval(u32::MAX); // leave this thread as we found it
            a.wasm
        };

        // A loop: mov r1,#0 / subs r0,r0,#1 / bne back to the subs / bx lr.
        let looping: [u8; 16] = [
            0x00, 0x10, 0xa0, 0xe3, // mov r1, #0
            0x01, 0x00, 0x50, 0xe2, // subs r0, r0, #1
            0xfd, 0xff, 0xff, 0x1a, // bne 0x10004  (backward)
            0x1e, 0xff, 0x2f, 0xe1, // bx lr
        ];
        // No loop: cmp r0,#0 / beq forward over one instruction / mov r1,#1 / bx lr.
        let straight: [u8; 16] = [
            0x00, 0x00, 0x50, 0xe3, // cmp r0, #0
            0x00, 0x00, 0x00, 0x0a, // beq 0x1000c  (forward)
            0x01, 0x10, 0xa0, 0xe3, // mov r1, #1
            0x1e, 0xff, 0x2f, 0xe1, // bx lr
        ];

        assert_eq!(
            fuel_points(&build(&looping, 1000)),
            1,
            "a single guest loop must get exactly one fuel point on its back edge"
        );
        assert_eq!(
            fuel_points(&build(&straight, 1000)),
            0,
            "a forward-only branch must cost nothing - fuel is for loops"
        );
        // And the whole feature must vanish when not asked for, so a native build is
        // byte-identical to one from before fuel existed.
        assert_eq!(
            fuel_points(&build(&looping, 0)),
            0,
            "fuel disabled must emit no fuel points at all"
        );
        assert_eq!(
            build(&looping, 0),
            build(&looping, u32::MAX),
            "an unset fuel interval and an explicit 0 must produce the same module"
        );
    }

    /// An emitted function must charge itself EXACTLY what wasmtime would charge it.
    ///
    /// This is the whole point of the mechanism. The charge drives the browser's virtual
    /// game clock, and that clock is only comparable with native's because native's
    /// quantum is wasmtime fuel over the very same module - so any divergence in the rule
    /// is a divergence in game time, which a frame-keyed recipe then reads as the browser
    /// being on a different screen from native.
    ///
    /// Both halves of wasmtime's rule are load-bearing and each was got wrong in turn:
    /// billing one per instruction (rather than nothing for `end`/`block`/`loop`/`return`/
    /// `drop`) and billing a whole guest block up front (rather than only the operators
    /// actually reached) each ran the clock several times fast.
    ///
    /// The cost table below is a deliberate second statement of `emit::operator_cost`,
    /// over `wasmparser`'s operators rather than `wasm_encoder`'s. A test that imported
    /// the emitter's own table would agree with it by construction and prove nothing.
    #[test]
    fn a_function_charges_itself_exactly_what_wasmtime_would() {
        use wasmparser::{Operator, Payload};

        // `wasmtime_environ::default_operator_cost`: operators that generate no machine
        // code are free, everything else is one operation.
        fn wasmtime_cost(op: &Operator) -> i64 {
            match op {
                Operator::Nop | Operator::Drop => 0,
                Operator::Block { .. }
                | Operator::Loop { .. }
                | Operator::Unreachable
                | Operator::Return
                | Operator::Else
                | Operator::End => 0,
                _ => 1,
            }
        }

        const INTERVAL: i32 = 1000;
        let audit = |code: &[u8], what: &str| {
            set_fuel_interval(INTERVAL as u32);
            let wasm = transpile(&Program {
                code,
                base: 0x10000,
                thumb: false,
                entries: &[0x10000],
                arm_entries: &[],
                externs: &[],
                redirects: &[],
                inline_imports: &[],
                noreturn_svc: &[],
                mem_bytes: 0x20000,
                discover_code_pointers: false,
                import_memory: false,
            })
            .expect("transpile")
            .wasm;
            set_fuel_interval(u32::MAX); // leave this thread as we found it
            wasmparser::validate(&wasm).expect("valid wasm");

            // Every body EXCEPT the last, which is `emit_reset` (see `abi::RESET_EXPORT`).
            // That one is host bookkeeping - it restores an instance's globals so the host
            // can reuse it - and is emitted deliberately UNBILLED, because native never runs
            // it and billing it would charge the game clock for work only one engine does.
            // Auditing it against wasmtime's rule would therefore assert the opposite of
            // what it is for.
            let bodies: Vec<_> = wasmparser::Parser::new(0)
                .parse_all(&wasm)
                .filter_map(|p| match p.expect("parse") {
                    Payload::CodeSectionEntry(body) => Some(body),
                    _ => None,
                })
                .collect();
            let audited = bodies.len() - 1;
            for body in bodies.into_iter().take(audited) {
                let ops: Vec<Operator> = body
                    .get_operators_reader()
                    .expect("operators")
                    .into_iter()
                    .collect::<Result<_, _>>()
                    .expect("operators");

                // Walk the body, splitting it into the emitter's own bookkeeping (which
                // native does not have and so must never be billed) and the translated
                // guest code (which must be billed by wasmtime's rule).
                //
                // Wasmtime charges a function one unit for being entered at all, so that
                // an empty one still costs something; `Body::new` starts at 1 to match.
                let mut charged: i64 = 0;
                let mut owed: i64 = 1;
                let mut i = 0;
                while i < ops.len() {
                    // A commit:  global.get $fuel ; i32.const N ; i32.sub ; global.set $fuel
                    if let [
                        Operator::GlobalGet { global_index: g },
                        Operator::I32Const { value },
                        Operator::I32Sub,
                        Operator::GlobalSet { global_index: h },
                    ] = ops[i..(i + 4).min(ops.len())]
                    {
                        if g == abi::FUEL_GLOBAL && h == abi::FUEL_GLOBAL {
                            assert!(value > 0, "{what}: a commit of {value} is dead code");
                            charged += value as i64;
                            i += 4;
                            continue;
                        }
                    }
                    // A back-edge test:  global.get $fuel ; i32.const 0 ; i32.le_s ; if
                    //   ; i32.const -1 ; call $host ; i32.const INTERVAL ; global.set $fuel ; end
                    if let [
                        Operator::GlobalGet { global_index: g },
                        Operator::I32Const { value: 0 },
                        Operator::I32LeS,
                        Operator::If { .. },
                    ] = ops[i..(i + 4).min(ops.len())]
                    {
                        if g == abi::FUEL_GLOBAL {
                            assert!(
                                matches!(
                                    ops[i + 6],
                                    Operator::I32Const { value } if value == INTERVAL
                                ),
                                "{what}: a back-edge test must reload the whole interval"
                            );
                            i += 9;
                            continue;
                        }
                    }
                    owed += wasmtime_cost(&ops[i]);
                    i += 1;
                }
                assert_eq!(
                    charged, owed,
                    "{what}: the body commits {charged} fuel where wasmtime bills {owed}"
                );
            }
        };

        // Straight-line only, so every operator in the body is reached and the totals can
        // be compared as flat sums.
        audit(
            &[
                0x00, 0x10, 0xa0, 0xe3, // mov r1, #0
                0x01, 0x00, 0x90, 0xe0, // adds r0, r0, r1
                0x1e, 0xff, 0x2f, 0xe1, // bx lr
            ],
            "straight line",
        );
        // With real control flow: a dispatch loop, a back edge carrying a fuel test, and
        // predicated code, which is where an up-front per-block charge went wrong.
        audit(
            &[
                0x00, 0x10, 0xa0, 0xe3, // mov r1, #0
                0x01, 0x00, 0x50, 0xe2, // subs r0, r0, #1
                0x01, 0x10, 0x81, 0x12, // addne r1, r1, #1   (predicated)
                0xfc, 0xff, 0xff, 0x1a, // bne 0x10004        (backward)
                0x1e, 0xff, 0x2f, 0xe1, // bx lr
            ],
            "a loop with predication",
        );
    }

    #[test]
    fn transpiles_arm_hello() {
        // adr r0,msg / mov r1,#13 / svc #0 / svc #1  (base 0x10000, ARM).
        let code: [u8; 16] = [
            0x08, 0x00, 0x8f, 0xe2, // adr r0, msg
            0x0d, 0x10, 0xa0, 0xe3, // mov r1, #13
            0x00, 0x00, 0x00, 0xef, // svc #0
            0x01, 0x00, 0x00, 0xef, // svc #1
        ];
        let artifact = transpile(&Program {
            code: &code,
            base: 0x10000,
            thumb: false,
            entries: &[0x10000],
            arm_entries: &[],
            externs: &[],
            redirects: &[],
            inline_imports: &[],
            noreturn_svc: &[],
            mem_bytes: 0x20000,
            discover_code_pointers: false,
            import_memory: false,
        })
        .expect("transpile");
        assert!(!artifact.wasm.is_empty());
        assert_eq!(artifact.funcs.len(), 1);
        assert_eq!(artifact.funcs[0].addr, 0x10000);
        assert_eq!(artifact.funcs[0].export, "f_10000");
        // The module must validate.
        wasmparser::validate(&artifact.wasm).expect("valid wasm");
    }

    /// A mirror-reading import must lower to a plain load of the reserved block, with
    /// NO host call left behind - and the block must be a page the module actually
    /// declares, above everything the guest can reach.
    ///
    /// Worth pinning here because the failure is silent and remote: a load of the wrong
    /// address reads a word nobody writes, which for the clock is a vblank spin that can
    /// never be satisfied - the title simply stops, thousands of frames from the cause.
    #[test]
    fn a_mirror_import_lowers_to_a_load_of_the_reserved_block() {
        // Thumb `bl` to the import stub at 0x10010, then `bx lr`.
        let code: [u8; 8] = [
            0x00, 0xf0, 0x06, 0xf8, // bl 0x10010
            0x70, 0x47, // bx lr
            0x00, 0x00,
        ];
        const SLOT: u32 = 3;
        let program = |inline: &[InlineImport]| -> Artifact {
            transpile(&Program {
                code: &code,
                base: 0x10000,
                thumb: true,
                entries: &[0x10000],
                arm_entries: &[],
                externs: &[Extern { addr: 0x10010, import: 0 }],
                redirects: &[],
                inline_imports: inline,
                noreturn_svc: &[],
                mem_bytes: 0x20000,
                discover_code_pointers: false,
                import_memory: false,
            })
            .expect("transpile")
        };

        let plain = program(&[]);
        assert_eq!(plain.mirror_off, None, "no mirror op means no block and no layout change");

        let mirrored =
            program(&[InlineImport { import: 0, op: InlineOp::LoadMirror { slot: SLOT } }]);
        wasmparser::validate(&mirrored.wasm).expect("valid wasm");
        let off = mirrored.mirror_off.expect("a mirror op reserves the block");
        assert_eq!(
            mirrored.mem_pages,
            plain.mem_pages + 1,
            "the block is one more declared page"
        );
        assert!(
            off >= u64::from(0x20000u32),
            "the block must sit above the guest region, not inside it"
        );

        // The emitted body must contain the load of slot SLOT, and no call to the
        // import it replaced. The un-inlined build is checked to DO make that call, so
        // the negative assertion below cannot pass by looking at the wrong thing.
        assert!(
            scan_body(&plain.wasm).1,
            "without an inline form the import must be a host call",
        );
        let (loads, calls) = scan_body(&mirrored.wasm);
        assert!(
            loads.contains(&(off + u64::from(SLOT) * 4)),
            "expected an i32.load of the slot at {:#x}; found loads at {loads:?}",
            off + u64::from(SLOT) * 4
        );
        assert!(!calls, "the inlined import must not also emit a host call");
    }

    /// Byte offsets of every `i32.load` in the module's guest functions, and whether
    /// any of them calls the host-import trap.
    fn scan_body(wasm: &[u8]) -> (Vec<u64>, bool) {
        use wasmparser::{Chunk, Parser, Payload};
        let mut loads = Vec::new();
        let mut calls = false;
        let mut parser = Parser::new(0);
        let mut input = wasm;
        loop {
            let (payload, consumed) = match parser.parse(input, true).expect("parse") {
                Chunk::Parsed { payload, consumed } => (payload, consumed),
                Chunk::NeedMoreData(_) => panic!("truncated module"),
            };
            if let Payload::CodeSectionEntry(body) = &payload {
                let mut ops = body.get_operators_reader().expect("ops");
                while !ops.eof() {
                    match ops.read().expect("op") {
                        wasmparser::Operator::I32Load { memarg } => loads.push(memarg.offset),
                        // The host-call trap the inline form replaces.
                        wasmparser::Operator::Call { function_index }
                            if function_index == crate::emit::IMPORT_FUNC =>
                        {
                            calls = true
                        }
                        _ => {}
                    }
                }
            }
            if let Payload::End(_) = payload {
                break;
            }
            input = &input[consumed..];
        }
        (loads, calls)
    }

    #[test]
    fn transpiles_indirect_call() {
        // Thumb: `blx r0` (indirect call through a function pointer) then `bx lr`.
        // Exercises the indirect-call lowering + the module dispatcher emission.
        let code: [u8; 4] = [
            0x80, 0x47, // blx r0
            0x70, 0x47, // bx lr
        ];
        let artifact = transpile(&Program {
            code: &code,
            base: 0x10000,
            thumb: true,
            entries: &[0x10000],
            arm_entries: &[],
            externs: &[],
            redirects: &[],
            inline_imports: &[],
            noreturn_svc: &[],
            mem_bytes: 0x20000,
            discover_code_pointers: false,
            import_memory: false,
        })
        .expect("transpile indirect");
        // One guest function plus the emitted dispatcher must produce valid wasm.
        assert_eq!(artifact.funcs.len(), 1);
        wasmparser::validate(&artifact.wasm).expect("valid wasm with dispatcher");
    }

    #[test]
    fn scan_veneers_resolves_interworking_stub() {
        // A Thumb->ARM interworking veneer trampolining to a stub 0x10 past it.
        // Layout at base: bx pc / b.n .-2 / ldr ip,[pc] / add pc,ip,pc / .word off.
        let mut code = vec![0u8; 0x40];
        let put16 = |c: &mut [u8], o: usize, v: u16| c[o..o + 2].copy_from_slice(&v.to_le_bytes());
        let put32 = |c: &mut [u8], o: usize, v: u32| c[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put16(&mut code, 0x00, 0x4778); // bx pc
        put16(&mut code, 0x02, 0xe7fd); // b.n .-2
        put32(&mut code, 0x04, 0xe59f_c000); // ldr ip, [pc]
        put32(&mut code, 0x08, 0xe08c_f00f); // add pc, ip, pc
        put32(&mut code, 0x0c, 0x0000_0010); // .word 0x10 -> stub = base+16+16 = base+0x20
        let base = 0x8100_0000;
        let mut imports = BTreeMap::new();
        imports.insert(base + 0x20, 7u32); // the stub at base+0x20 -> import 7
        let found = scan_veneers(&code, base, &imports);
        assert_eq!(found, vec![(base, 7)]);
    }

    #[test]
    fn transpiles_cube_memset() {
        // The cube's Thumb memset at 0x81000948 (cbz, subs, strb[!], cmp, bne,
        // bx lr): a 4-block loop that stresses the dispatch machinery.
        let code: [u8; 18] = [
            0x32, 0xb1, 0x01, 0x3a, 0x43, 0x1e, 0x02, 0x44, 0x03, 0xf8, 0x01, 0x1f, 0x93, 0x42,
            0xfb, 0xd1, 0x70, 0x47,
        ];
        let base = 0x8100_0948;
        let artifact = transpile(&Program {
            code: &code,
            base,
            thumb: true,
            entries: &[base],
            arm_entries: &[],
            externs: &[],
            redirects: &[],
            inline_imports: &[],
            noreturn_svc: &[],
            mem_bytes: 0x1_0000,
            discover_code_pointers: false,
            import_memory: false,
        })
        .expect("transpile memset");
        if let Err(e) = wasmparser::validate(&artifact.wasm) {
            dump_last_func(&artifact.wasm);
            panic!("invalid memset module: {e}");
        }
    }

    /// Print the operators of the last function body with a running control depth.
    fn dump_last_func(wasm: &[u8]) {
        use wasmparser::{Parser, Payload};
        let mut depth = 0i32;
        for payload in Parser::new(0).parse_all(wasm) {
            if let Ok(Payload::CodeSectionEntry(body)) = payload {
                let mut reader = body.get_operators_reader().unwrap();
                while let Ok(op) = reader.read() {
                    use wasmparser::Operator::*;
                    let before = depth;
                    match op {
                        Block { .. } | Loop { .. } | If { .. } => depth += 1,
                        End => depth -= 1,
                        _ => {}
                    }
                    eprintln!("depth {before}->{depth}: {op:?}");
                }
            }
        }
    }

    #[test]
    fn import_memory_mode_imports_shared_memory_and_validates() {
        // Same tiny ARM program, once self-contained and once importing a shared
        // memory: both must validate, and only the shared build declares the
        // `env.memory` import (shared, with a maximum).
        let code: [u8; 8] = [
            0x0d, 0x10, 0xa0, 0xe3, // mov r1, #13
            0x00, 0x00, 0x00, 0xef, // svc #0
        ];
        let mk = |import_memory| {
            transpile(&Program {
                code: &code,
                base: 0x10000,
                thumb: false,
                entries: &[0x10000],
                arm_entries: &[],
                externs: &[],
                redirects: &[],
            inline_imports: &[],
                noreturn_svc: &[],
                mem_bytes: 0x20000,
                discover_code_pointers: false,
                import_memory,
            })
            .expect("transpile")
            .wasm
        };

        let plain = mk(false);
        let shared = mk(true);
        wasmparser::validate(&plain).expect("plain module valid");
        wasmparser::validate(&shared).expect("shared-memory module valid");

        // Walk imports: the shared build must import a shared memory named
        // `env.memory`; the plain build must not import any memory.
        let has_shared_mem_import = |wasm: &[u8]| {
            use wasmparser::{Parser, Payload, TypeRef};
            for payload in Parser::new(0).parse_all(wasm) {
                if let Ok(Payload::ImportSection(reader)) = payload {
                    for imp in reader.into_imports() {
                        let imp = imp.unwrap();
                        if let TypeRef::Memory(mt) = imp.ty {
                            assert_eq!(imp.module, "env");
                            assert_eq!(imp.name, "memory");
                            assert!(mt.shared, "imported memory must be shared");
                            assert!(mt.maximum.is_some(), "shared memory needs a maximum");
                            return true;
                        }
                    }
                }
            }
            false
        };
        assert!(has_shared_mem_import(&shared), "shared build imports env.memory");
        assert!(!has_shared_mem_import(&plain), "plain build imports no memory");
    }
}
