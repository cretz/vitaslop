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
/// Whether emitted modules hold the ARM register file in wasm LOCALS along each
/// straight-line run (`VITASLOP_PROMOTE_REGS`), and the per-thread override a test uses
/// to emit both arms in one process. See [`promote`].
pub use emit::{promote_registers, set_promote_registers};
/// The A/B arm for the flag carry/overflow forms - see [`emit::flags_wide_c`]. The browser
/// has no environment, so it selects the arm through the setter.
pub use emit::{flags_wide_c, set_flags_wide_c};
/// The ablation that prices a DISPATCH RE-ENTRY - see [`emit::dispatch_all`]. Selected
/// through the setter for the same reason: the browser has no environment, and the browser
/// is the engine whose indirect-branch cost is the question.
pub use emit::{dispatch_all, set_dispatch_all};
/// The guest-address wasm NAME SECTION - see [`emit::emit_wasm_names`]. Selected through the
/// setter so a browser profile can name the guest functions it samples.
pub use emit::{emit_wasm_names, set_wasm_names};
/// The fuel interval modules emitted on this thread carry, so a HOST can read the
/// software counter the emitted code maintains: the counter runs DOWN from this and
/// reloads to it, so only a host that knows the interval can difference it.
pub use emit::fuel_interval;
/// The lowering categories operator cost is attributed to - see [`emit::Expansion::by_stmt`].
pub use emit::StmtKind;
/// The guest-store DIRTY MAP: whether to emit it, and where a host reads it. The map
/// is one byte per 4 KB page holding the epoch of the last store into that page, laid
/// out at [`Artifact::dirty_off`] as `[epoch byte][map]` (see [`DIRTY_EPOCH_OFF`] and
/// [`DIRTY_MAP_OFF`]). A host that stamps its own reads against the epoch can prove a
/// region of guest memory unchanged without reading it.
pub use emit::{set_dirty_tracking, DIRTY_EPOCH_OFF, DIRTY_MAP_OFF, DIRTY_SHIFT};
mod flags;
mod ir;
mod lower;
/// Holding the ARM register file in wasm LOCALS instead of globals: the policy, and the
/// model that prices it. Public because the price is a number a host reports.
pub mod promote;

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
///
/// The storing forms extend that trade to the WRITE instruments, and they extend it
/// FURTHER than the reading forms do - which is worth stating plainly, because the obvious
/// guess is wrong. An inlined [`InlineOp::StoreArg`] writes guest memory as a plain wasm
/// store, so `VITASLOP_HOST_WRITE_WATCH` - which reports writes a HOST CALL makes - does not
/// see it. Neither does `VITASLOP_WATCH_STORE`: that watchpoint is emitted around the
/// translated guest STORES ([`emit`]'s `Stmt::Store` arm) and nothing in `emit_import` goes
/// near it, so an inlined store is invisible to BOTH watches at once. So before concluding
/// from a silent write watch that nobody wrote a context field, run with
/// `VITASLOP_NO_INLINE_IMPORTS=1` and the host call - and with it `VITASLOP_HOST_WRITE_WATCH`
/// - comes back.
///
/// The guest-store DIRTY MAP is the one write instrument an inline form does not simply drop
/// out of, because a texture read against a stale stamp is a wrong picture rather than a
/// missing diagnostic line. See `emit::emit_dirty_range` for which forms owe it a stamp and
/// why the others do not.
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
    /// `r0 = ((u32_at(r0 + offset) >> shift) & mask) + plus` - read a word at a fixed offset
    /// from the pointer argument, extract a bitfield, and add a fixed bias. `mask` is applied
    /// after the shift, so `mask = u32::MAX` means "the whole word"; `plus = 0` means "the
    /// field itself", which is what most getters want.
    ///
    /// # Why a bias belongs in the same form
    /// A hardware field is not always the number the API returns. GXM stores a texture's
    /// width and height as SIZE MINUS ONE, so `sceGxmTextureGetWidth` is the field plus one -
    /// and that `+ 1` is part of DECODING the field, not a second operation on top of it.
    /// Splitting it into its own variant would leave two forms that must be kept in step for
    /// one idea; the bias is one `i32.add` the emitter drops entirely when it is zero, so an
    /// unbiased form stays byte-identical to what it was before this existed.
    LoadShiftMask { offset: u32, shift: u32, mask: u32, plus: u32 },
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
    /// `r0 = mirror[slot]`, plus a countdown that PARKS the thread when the same word has
    /// been read `mirror[budget]` times inside one resume.
    ///
    /// [`InlineOp::LoadMirror`] with a spin guard, and the guard is exact rather than
    /// heuristic. The mirror contract is that a slot cannot change while guest code runs,
    /// so every read inside one resume returns the SAME word - a thread that has taken
    /// thousands of them is in a loop it cannot leave, and it will leave it only when the
    /// clock reaches the next vblank, which its own spinning is what pays for.
    ///
    /// The budget lives in the block so the host refreshes it exactly where it refreshes
    /// everything else: once per resume, at the one point that makes the count mean "reads
    /// since this thread was scheduled". Nothing in the guard is TOLLED - it must not move
    /// the fuel counter or the clock, or it would change the schedule it is measuring.
    ///
    /// See [`abi::VBLANK_PARK_SELECTOR`] for what the host does with it and what it is
    /// worth. `budget` names a slot, not a count: the count is the word in it.
    LoadMirrorParking { slot: u32, budget: u32 },
    /// `r0 = value` - the whole call. For a handler that returns a constant and does
    /// NOTHING else.
    ///
    /// # Why a no-op is worth emitting
    /// It looks like the least valuable form here, and on the desktop it nearly is. In the
    /// browser a host call is a boundary crossing whose cost is almost entirely marshalling
    /// [[vitaslop-browser-host-call-cost]], so a call that computes nothing still costs the
    /// full crossing - and a real title makes them in bulk. MEASURED on a retail racer's
    /// race: `sceNgsPatchGetInfo` and `sceNgsVoicePatchSetVolumesMatrix` are **198 calls per
    /// guest frame EACH** - 32% of everything the title calls - and each is 1.14 us of pure
    /// crossing in desktop Chrome, ~1.0 ms per presented frame between them. On a phone, where
    /// a crossing is ~20 us, the same pair is most of a frame.
    ///
    /// # Admissibility, which is narrower than it looks
    /// The handler must be EXACTLY a constant return. Not "returns 0 today": a handler that
    /// records a call, fills an out-parameter, or touches host state that anything reads is a
    /// different program, and inlining it makes whatever it did stop happening silently. The
    /// caller decides ([`vitaslop_runtime::vita::inline_op`]), because only the runtime knows
    /// what a NID means.
    ///
    /// # What is given up
    /// An inlined call never reaches the host, so the NID leaves the call histogram and the
    /// host-call trace - and for a STUB that matters more than for the other forms, because
    /// "which unimplemented calls does this title make" is a question the histogram is how
    /// anyone answers. So the runtime prints the inlined stub list once at link time and keeps
    /// a scoped switch to put them all back on the host.
    RetConst { value: u32 },
    /// Nothing at all - not even a write to r0.
    ///
    /// The twin of [`InlineOp::RetConst`] for a handler that returns VOID: one whose Rust
    /// signature yields `()`, so the dispatcher leaves r0 holding whatever the guest passed
    /// in it. `RetConst { value: 0 }` is NOT the same program for such a call - it would hand
    /// back 0 where the call used to hand back its own first argument - and the difference is
    /// invisible until a caller reads the result.
    ///
    /// `sceKernelSetGPO` is the case: it sets a devkit LED, its handler stores a word nothing
    /// ever reads, and a retail title calls it **116 times per guest frame**. Admissible for
    /// exactly the same reason `RetConst` is, and under the same rule - the handler must do
    /// nothing that anything observes - with the extra requirement that it return void.
    Nop,
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
    /// `u32_at(r0 + offset) = r1; r0 = 0` - store the SECOND argument through the pointer
    /// argument at a fixed offset, then return success.
    ///
    /// # What this is for
    /// Every form above reads. This one writes, and it exists because the largest block of
    /// host calls a real title makes in steady gameplay is not getters at all - it is the
    /// GXM draw state: `sceGxmSetVertexProgram`, `sceGxmSetCullMode`,
    /// `sceGxmSetFrontDepthFunc` and their neighbours, measured at 248 calls per frame EACH
    /// on one title, EIGHT of them from the same call site as every
    /// `sceGxmDrawPrecomputed`. Nine crossings a draw, eight of them one-word state writes.
    ///
    /// A setter is only inlinable once its state lives in guest memory - which is where the
    /// hardware keeps it, so this is the faithful shape as well as the fast one. See
    /// `vitaslop_runtime::vita::gxmctx`.
    ///
    /// # Why it is exactly the host call
    /// The handler writes one word at a fixed offset from a guest pointer and returns 0.
    /// So does this. The pointer guard is the same one the reading forms use, so an
    /// out-of-range pointer still reaches the handler and keeps its old semantics -
    /// including the case where the handler would decline to write at all.
    ///
    /// A handler that does anything ELSE - reports, resolves a handle, sizes an allocation -
    /// must not use this form. That is not a style rule: the inline call never reaches the
    /// host, so whatever else the handler did simply stops happening, silently.
    StoreArg { offset: u32 },
    /// `u32_at(r0 + offset + r1 * 4) = r2; r0 = 0`, but ONLY when `r1 < count`; otherwise
    /// the real host call runs.
    ///
    /// The indexed twin of [`InlineOp::StoreArg`], for the setters shaped
    /// `set(context, index, value)` - `sceGxmSetVertexStream` is the one that pays here, at
    /// 239 calls a frame. The bound is what makes it exact: an index past the end of the
    /// array is a case only the handler defines (it reports it), and writing past the array
    /// inline would corrupt whatever field follows with nothing to say so.
    StoreArgIndexed { offset: u32, count: u32 },
    /// `w = u32_at(r0 + offset); u32_at(r0 + offset) = (w & !(mask << shift)) | ((r1 & mask) << shift); r0 = 0`
    /// - write one BITFIELD of a word through the pointer argument, leaving every other bit
    /// alone, then return success.
    ///
    /// # Why a field form and not a store
    /// [`InlineOp::StoreArg`] writes a whole word, which is right for a context field that
    /// owns its word and wrong for the case here: a `SceGxmTexture`'s control word 0 packs
    /// eight independent settings, and `sceGxmTextureSetMagFilter` changes one of them.
    /// Storing the word would clear the address mode, the mip count and the LOD bias with it,
    /// and the picture that results is not obviously a texture-state bug - it is a texture
    /// that samples wrongly, which reads as a decode problem.
    ///
    /// This is the read-modify-write twin of [`InlineOp::LoadShiftMask`], over the same
    /// `(offset, shift, mask)` a getter reads, so a setter and its getter can be given the
    /// SAME field constants and cannot disagree about where the field is.
    ///
    /// # The value is MASKED, not rejected
    /// A value wider than the field is truncated, because that is what the hardware does with
    /// it and what the handler already does. This form is only admissible for a setter whose
    /// handler uses its argument AS PASSED: a setter that pre-shifts (because its enum is
    /// already in control-word position, like `SceGxmTextureMipFilter`) computes something
    /// else and must stay on the host.
    ///
    /// # Read-modify-write with no yield point
    /// Like the lock forms, this loads a word, computes, and stores it with no loop and no
    /// call in between, so neither engine can preempt inside it - see
    /// [`InlineOp::LwMutexLock`] for the whole argument. It needs less than the lock does:
    /// two threads racing on one texture's control word is a data race the guest wrote, and
    /// the host handler was no more atomic.
    StoreArgField { offset: u32, shift: u32, mask: u32 },
    /// `w = u32_at(r0 + offset); u32_at(r0 + offset) = (w & !mask) | (r1 & mask); r0 = 0` -
    /// [`InlineOp::StoreArgField`] for a setter whose argument is ALREADY in field position.
    ///
    /// # Why this is a second form and not a flag on the first
    /// `StoreArgField` shifts the argument UP into the field, which is right for an enum
    /// numbered from zero (`SceGxmTextureFilter` is 0..3) and wrong for one whose constants are
    /// the control-word bits themselves (`SCE_GXM_TEXTURE_MIP_FILTER_ENABLED` IS `0x200`). The
    /// handlers for those shift the argument DOWN first, so giving them the shifting form would
    /// mask `0x200` to zero and store "disabled" for every call that asked for enabled -
    /// silently, visible only as absent mip filtering. The two really are different programs,
    /// and the way to keep them from being confused is to make the emitter carry both.
    ///
    /// The mask here is the field IN PLACE (`mask << shift` of the getter's pair), so a caller
    /// still derives it from the one `(shift, mask)` constant its getter reads and the two
    /// cannot drift apart.
    ///
    /// Everything else - the truncation of a wider value, the single load/compute/store with no
    /// yield point, the admissibility rule that the handler must do this and nothing else -
    /// is exactly as [`InlineOp::StoreArgField`] states it.
    StoreArgFieldInPlace { offset: u32, mask: u32 },
    /// `for i in 0..count: u32_at(r0 + offset + 4*i) = raw_bits(s_i); r0 = 0` - store the
    /// first `count` VFP single-precision ARGUMENT registers, as raw bits, into a
    /// contiguous word run at a fixed offset from the pointer argument, then return
    /// success.
    ///
    /// # What this is for
    /// The multi-float context setters. `sceGxmSetViewport(context, 6 floats)` is the one
    /// that pays: its handler is exactly "store the six argument floats' bits into six
    /// consecutive context words and return 0" - a hardfloat AAPCS call carries them in
    /// s0..s5 - and it is called ~12 times a frame on a racing title, each one a full
    /// crossing to move 24 bytes the emitted code already holds in globals.
    ///
    /// # Admissibility
    /// Same rule as [`InlineOp::StoreArg`]: the handler must store its arguments AS PASSED,
    /// contiguously, and do nothing else - no report, no host state, no derived value. The
    /// bits are stored raw (`f32::to_bits` of what the guest put in s`i`), which is exactly
    /// what the handler's `v.to_bits()` writes; no float operation happens on either path,
    /// so there is no rounding to disagree about.
    ///
    /// Guarded like every pointer form, against the LAST word of the run.
    StoreVfpRun { offset: u32, count: u32 },
    /// `for i in 0..count: u32_at(r0 + offset + 4*i) = int_arg(i + 1); r0 = 0` - store the
    /// call's integer arguments AFTER the pointer (`r1..r3`, then the guest stack at
    /// `sp + 4*(n-4)` for argument `n`) as a contiguous word run at a fixed offset from the
    /// pointer argument, then return success.
    ///
    /// # What this is for
    /// The multi-word context setters whose values arrive in core registers.
    /// `sceGxmSetRegionClip(context, mode, xMin, yMin, xMax, yMax)` is the one that pays:
    /// its handler stores the five arguments into five consecutive context words
    /// (`REGION_CLIP_MODE` then the four bounds) - and with six arguments the AAPCS puts
    /// the last two on the GUEST STACK, so the run's tail is two plain loads from `sp`,
    /// exactly the loads `GuestCtx::arg` performs on the host path.
    ///
    /// # Admissibility
    /// Same rule as [`InlineOp::StoreArg`]: arguments stored AS PASSED, contiguously,
    /// nothing else. A handler that masks an argument (`& 0xff`), reports, or touches host
    /// state must not use this form.
    ///
    /// # Guards
    /// The destination is guarded against the LAST word of the run, and - when the run
    /// reaches past r3 - the STACK POINTER is guarded against the last stack word read, so
    /// a thread with a garbage sp falls back to the handler (whose `read_u32` defines that
    /// case) instead of trapping in emitted code.
    StoreArgRun { offset: u32, count: u32 },
    /// `dst = r0 + offset + r1 * stride; dst[0] = r2; dst[1..=words] = *(r2 .. r2 + 4*words);
    /// dst[words + 1] = 0; r0 = 0` - copy N words THROUGH a second pointer into an indexed
    /// slot, recording where they came from.
    ///
    /// # Why a copy form exists at all
    /// Every storing form above writes a value the guest passed. This one writes bytes it
    /// FETCHES, and that is not a generalisation for its own sake - it is the only shape that
    /// can serve `sceGxmSetFragmentTexture`, which is the largest single block of host calls
    /// left in steady gameplay on a real title: 1,275 crossings per display frame, 1,025 of
    /// them from ONE call site, measured over a 100-frame window of live racing.
    ///
    /// GXM copies a texture's control words BY VALUE at bind time
    /// (`vitaslop-texture-binding-by-value`). Storing the POINTER and reading it at draw time
    /// would be a different program - one where a texture the guest re-initialised between
    /// bind and draw renders with its new contents - so `StoreArg` cannot serve this call
    /// however much it looks like a setter. Copying the words at the moment of the bind is
    /// what the hardware does, so this form is the faithful shape as well as the fast one.
    ///
    /// # Why each guard is there
    /// - **Both pointers are bounds-checked**, the destination against its LAST word and the
    ///   source against `4 * words`. Two pointers, two guards; either failing runs the handler.
    /// - **`r1 < count`** keeps an out-of-range sampler unit on the host, which is the side
    ///   that reports it. Writing past the array would corrupt whatever field follows.
    /// - **`r2 != 0`** sends the UNBIND case to the handler. A null texture is not a copy at
    ///   all - it clears the unit - and it is rare enough that a second inline arm would be
    ///   more code than it saves.
    ///
    /// The trailing zero word is the "this came from a precomputed state" flag, which a
    /// direct bind always clears. It is written rather than left alone because the slot is
    /// reused: a unit bound by a precomputed state and then re-bound directly would otherwise
    /// keep the old provenance and mislabel itself in every later report.
    CopyArgIndexed { offset: u32, stride: u32, count: u32, words: u32 },
    /// Bump a RING CURSOR in the structure r0 points at, hand the block back through r1,
    /// and record what was handed out - when everything the answer depends on is already in
    /// guest memory. Everything else runs the real host call.
    ///
    /// ```text
    /// h    = u32_at(r0 + l.program)                  -- the bound program handle
    /// cur  = align(u32_at(r0 + l.ring_cursor), l.align)
    /// take = u32_at(r0 + l.context_magic_at) == l.context_magic
    ///      & u32_at(h + l.program_magic_at) == l.program_magic
    ///      & u32_at(r0 + l.ring_base) != 0
    ///      & cur + u32_at(h + l.prog_alloc) <= u32_at(r0 + l.ring_end)
    /// if take {
    ///     u32_at(r0 + l.ring_cursor) = cur + u32_at(h + l.prog_alloc)
    ///     u32_at(r0 + l.record + 0)  = cur
    ///     u32_at(r0 + l.record + 4)  = u32_at(h + l.prog_size)
    ///     u32_at(r0 + l.record + 8)  = u32_at(h + l.prog_header)
    ///     u32_at(r1) = cur; r0 = 0
    /// } else { host call }
    /// ```
    ///
    /// # Why an ALLOCATING call can be inlined at all
    /// It looks like the one shape that cannot: the handler resolves a handle, reflects a
    /// program's interface to size a buffer, allocates, and leaves three facts behind for
    /// the draw to read. Every one of those was a reason to keep crossing, and every one of
    /// them turned out to be a fact that does not change after the program is CREATED.
    /// Moving the size to the handle it belongs to ([`vitaslop_runtime::vita::gxmprog`]) and
    /// the ring plus the bound record to the context block
    /// ([`vitaslop_runtime::vita::gxmctx`]) leaves an allocation that is a bump of one word,
    /// and both of those are where the hardware keeps the same facts: GXM's default uniform
    /// buffer IS a driver-recycled ring inside the memory the guest handed over.
    ///
    /// This is the largest single item left in a gameplay frame's host-call budget -
    /// MEASURED per frame at **1,189 crossings on one title (53% of everything it calls)**
    /// and 601 on another - and in the browser a crossing is ~1.4 us of marshalling alone.
    ///
    /// # Why each term of the guard is there
    /// - **The context magic** identifies r0 as a block this engine laid out. Without it an
    ///   arbitrary pointer would be read as a ring and the guest handed a "buffer" at an
    ///   address computed from whatever was there.
    /// - **The program magic** does the same for the bound handle, and it is the term that
    ///   covers "nothing is bound" (the word is 0) as well as "this is not our handle".
    /// - **A non-zero ring base** is how a context whose ring was never attached - the host
    ///   allocates it, and an exhausted heap declines - reaches the handler, which is the
    ///   side that can allocate one.
    /// - **The fit** sends a scene that overruns the ring to the handler, because WRAPPING
    ///   aliases two live buffers and that is a fidelity loss the handler reports once.
    ///   Inlining the wrap would make it silent.
    ///
    /// # What it does NOT do, and must not
    /// The handler also POISONS the vertex buffer when `VITASLOP_GXM_UNIFORM_POISON` is set,
    /// which is the diagnostic that separates "the guest wrote this" from "this is the last
    /// draw's uniforms still in the ring". An inlined call never reaches the host, so that
    /// fill would simply stop happening - which is why the runtime withholds this form
    /// entirely while the poison knob is on rather than emitting an approximation of it.
    ///
    /// # Read-modify-write with no yield point
    /// The cursor is loaded, bumped and stored with no loop and no call in between, so
    /// neither engine can preempt inside it and two threads cannot be handed the same
    /// block - see [`InlineOp::LwMutexLock`] for the whole argument. The host handler was no
    /// more atomic than this.
    ReserveUniformBuffer { layout: UniformRingLayout },
    /// Copy `r3` floats from the pointer at `[sp]` into TWO places - the uniform buffer in
    /// r0 at the register the parameter record in r1 names, and the engine's fallback SA
    /// bank - then return success. Everything else runs the real host call.
    ///
    /// ```text
    /// src  = u32_at(sp)
    /// idx  = u32_at(r1 + l.param_index_at)              -- the parameter's resource_index
    /// at   = idx + r2                                    -- the first REGISTER written
    /// bank = mirror[l.bank_slot]
    /// take = ((u32_at(r1 + l.param_packed_at) >> l.type_shift) & l.type_mask) != l.f16_type
    ///      & idx <= l.max_regs & r2 <= l.max_regs & r3 <= l.max_regs & at + r3 <= l.max_regs
    ///      & bank != 0
    ///      & every pointer admits 4*r3 bytes
    /// if take {
    ///     memcpy(r0 + at*4, src, r3*4)
    ///     memcpy(bank + l.bank_data_at + at*4, src, r3*4)
    ///     if at + r3 > u32_at(bank + l.bank_len_at) { u32_at(bank + l.bank_len_at) = at + r3 }
    ///     r0 = 0
    /// } else { host call }
    /// ```
    ///
    /// # Why this one is worth a form of its own
    /// After the GXM draw state, the texture binds and the default-uniform reserves were
    /// inlined, `sceGxmSetUniformDataF` is what a real title has LEFT: **1,106 calls a frame
    /// on one racing title, 58% of every host call it still makes.** It is also the last
    /// GXM call in the per-draw sequence that was not a plain fact about guest memory - and
    /// it turned out to be one, once the fallback bank moved into guest memory beside
    /// everything else the draw path reads (`vitaslop_runtime::host::SA_BANK_DATA`).
    ///
    /// # Why the copy is exactly what the handler does
    /// The handler reads each component with `read_f32` and writes it back with `to_bits`,
    /// which is bit-preserving in both directions, so a component's four bytes arrive
    /// unchanged - a NaN payload included. `memory.copy` moves the same bytes. The two
    /// destinations are written in the same order the handler writes them, and neither can
    /// overlap the source in a way the handler would have seen differently: `memory.copy`
    /// is specified to read the source in full before writing, which is what the handler's
    /// read-into-a-`Vec`-then-write does.
    ///
    /// # The FIFTH ARGUMENT
    /// `sourceData` is the fifth parameter, which AAPCS puts on the STACK. This is the only
    /// form that reads one, and it reads it exactly where `GuestCtx::arg(4)` does: the word
    /// at `sp`. Guarded like any other pointer - a stack pointer near the end of memory runs
    /// the handler.
    ///
    /// # What it refuses, and why each refusal is the handler's case
    /// - **An F16 parameter.** Two components share a register, so the write is a
    ///   read-modify-write per half and a byte copy is simply a different program. The
    ///   handler keeps it ([`vitaslop_runtime::host::VitaState::set_uniform_halves`]).
    /// - **A null or unreadable parameter record**, where the handler defines the base as 0.
    /// - **A negative or absurd `resource_index`**, which the handler clamps - and a clamp
    ///   is not expressible here, the same reason [`InlineOp::LoadScaled`] has a value guard.
    /// - **A write past the bank's ceiling**, which the handler drops from the bank while
    ///   still writing the buffer. Two different destinations disagreeing is precisely the
    ///   sort of case to leave in one place.
    /// - **No bank at all**, which is an arena that could not place one.
    SetUniformData { layout: UniformDataLayout },
    /// Apply a precomputed vertex/fragment STATE (r1) to the context block (r0), when both
    /// carry this engine's identity stamps - one bulk `memory.copy` of the state's arrays
    /// block into the context, the stage's three-word uniform record, and (fragment only)
    /// the program handle. Everything else runs the real host call.
    ///
    /// # Why a state bind can be inlined at all
    /// `sceGxmSetPrecomputed{Vertex,Fragment}State` used to read a HOST-side table keyed by
    /// the state's address, which forced the crossing AND could not follow a state the
    /// guest `memcpy`s (the identical by-value defect the precomputed-DRAW family fixed by
    /// moving into the guest block - see `vitaslop_runtime::vita::gxmstate`). With the
    /// state living in guest memory - its struct words plus an arrays block laid out
    /// exactly as the context block's own texture/uniform tables - the bind is a copy
    /// between two guest structures, which is what the hardware's own bind is. The
    /// reflected uniform size is memoised into the struct at INIT, the same "a fact fixed
    /// at creation belongs in the handle" move `ReserveUniformBuffer` rests on.
    ///
    /// The two binds are **48 calls per frame** on one title's race - every remaining
    /// non-draw GXM crossing it makes in steady gameplay.
    ///
    /// # The NULL-state arm, which on a real title is most of the traffic
    /// A race UNBINDS the precomputed state between draw batches - `state == 0`, twenty-plus
    /// times a frame - and the handler's null arm is as pure as the copy: the fragment bind
    /// does nothing but return success, the vertex bind zeroes the stage's table and
    /// record. Both are emitted inline (the vertex zeroing still behind the context magic).
    ///
    /// # Why each guard is there
    /// - **The context magic** identifies r0 as a block this engine laid out.
    /// - **The state magic** (stage-specific) identifies r1 as a state THIS engine
    ///   initialised, with this packing; an uninitialised struct runs the handler, which
    ///   defines that case (it clears / declines exactly as it did when the table missed).
    /// - **All three pointers are bounds-checked** - context, struct, arrays block - each
    ///   against the LAST byte it reaches.
    ///
    /// # No yield point
    /// Loads, one `memory.copy` (or `memory.fill`) and a few stores; no loop, no call, so
    /// neither engine can preempt inside it ([`InlineOp::LwMutexLock`] states the whole
    /// argument).
    BindPrecomputedState { layout: BindStateLayout },
    /// Take a recursive lock whose state lives in the guest WORK AREA pointed to by r0,
    /// when it is uncontended. Everything else runs the real host call.
    ///
    /// ```text
    /// take = r1 == 1
    ///      & u32_at(r0 + layout.id) == r0
    ///      & u32_at(r0 + layout.waiters) == 0
    ///      & (u32_at(r0 + layout.count) == 0 | u32_at(r0 + layout.owner) == mirror[thread_slot])
    /// if take { owner = mirror[thread_slot]; count += 1; r0 = 0 } else { host call }
    /// ```
    ///
    /// # Why a lock can be inlined at all
    /// `sceKernelLockLwMutex` is a USERSPACE function on the device. A lightweight mutex has
    /// no kernel handle - its state lives in the caller-provided
    /// `SceKernelLwMutexWork` - and the uncontended take is a compare-and-swap of that work
    /// area with no syscall at all; only CONTENTION enters the kernel. So this form is not an
    /// optimisation bolted onto a system call, it is the shape the call actually has, and the
    /// fallback arm is the syscall the hardware would also have made.
    ///
    /// The pair is the largest single block of host calls left in steady gameplay on a real
    /// title once the GXM draw state was inlined: 101,155 lock/unlock crossings in one profile
    /// window, 28,316 of each at a single call site.
    ///
    /// # Why each term of the guard is there
    /// - **`r1 == 1`** is the `lockCount` argument. A lock of any other count (including the
    ///   illegal zero) is the handler's case; folding a multi-count acquire into `count += 1`
    ///   would silently under-count the recursion and release the mutex early.
    /// - **`id == r0`** identifies the work area as ITSELF. The kernel keeps an id inside the
    ///   work area, and a caller may operate on a byte COPY of it staged elsewhere - a C++
    ///   wrapper putting its embedded work struct on the stack. A copy carries the ORIGINAL's
    ///   id, so it fails this test and the host resolves it. A never-created work area is
    ///   zeroed, so it fails too and the host adopts it.
    /// - **`waiters == 0`** keeps every mutex with a parked thread on the host, which is the
    ///   only place that can wake one.
    /// - **`count == 0 | owner == cur`** is free-or-mine. Testing `count` rather than `owner`
    ///   for freeness is what lets `owner` be left stale on release: the main thread's id is
    ///   0 by convention, so there is no owner value that can mean "nobody".
    ///
    /// The two writes are correct on both arms at once: `count + 1` takes a free mutex to 1
    /// and a recursive one to `n + 1`, and re-writing `owner` with the value it already holds
    /// is a no-op on the recursive arm. That is what collapses the two cases into one branch.
    ///
    /// # Why a plain read-modify-write is enough, when the hardware needs a CAS
    /// This tests the count and then stores `count + 1`, which is only safe if the scheduler
    /// cannot take the baton away in between - otherwise two threads read "free" and both
    /// take the mutex, with no error and no host call to notice it. The device needs an
    /// atomic compare-and-swap because its three cores really do run at once. Here they do
    /// not: there is one baton, and it changes hands only at a SUSPENSION POINT.
    ///
    /// Both engines put those in the same places, and neither is inside this sequence:
    /// wasmtime emits its `fuel_check` at function entry and loop headers only (an
    /// `if`/`else`/`end` merely bumps a local counter), and the browser's software fuel
    /// check is emitted on back edges only. This form has no loop and no call on the path
    /// that writes, so it runs to completion or not at all.
    ///
    /// That is a property of the emitted code, so it is pinned by a test
    /// (`a_lock_form_has_no_suspension_point`) rather than left as an argument. If either
    /// engine ever gains a finer preemption point, this needs real atomics - not a re-run
    /// of the test.
    LwMutexLock { layout: LwMutexLayout, thread_slot: u32 },
    /// Release a lock taken by [`InlineOp::LwMutexLock`], when nothing is parked on it.
    ///
    /// ```text
    /// drop = r1 == 1
    ///      & u32_at(r0 + layout.id) == r0
    ///      & u32_at(r0 + layout.waiters) == 0
    ///      & u32_at(r0 + layout.count) != 0
    ///      & u32_at(r0 + layout.owner) == mirror[thread_slot]
    /// if drop { count -= 1; r0 = 0 } else { host call }
    /// ```
    ///
    /// The mirror image of the lock, with the free-or-mine disjunction replaced by a
    /// conjunction: releasing requires that this thread really holds it. An unlock by a
    /// non-owner, or of an already-free mutex, is an ERROR the handler defines and reports;
    /// inline it would silently underflow the count.
    ///
    /// `owner` is deliberately NOT cleared when the count reaches zero. Every reader tests
    /// `count` first, and there is no owner value that means "nobody" (thid 0 is the main
    /// thread), so a stale owner is unobservable while a sentinel would be a second encoding
    /// of the same fact.
    LwMutexUnlock { layout: LwMutexLayout, thread_slot: u32 },
    /// `memmove(r0, r1, r2); r0 unchanged` - copy the r2 bytes at the pointer in r1 to the
    /// pointer in r0, and leave the destination in r0 as the return value.
    ///
    /// # Why a bulk form exists
    /// Every form above moves a fixed, small number of words named at emit time. This one
    /// moves a count the guest supplies, and it exists because `sceClibMemcpy`,
    /// `sceClibMemset` and `sceClibMemcmp` together are 508,181 crossings of a real title's
    /// 3.85 M - 13% of every host call it makes, with 403,687 memcmps at a SINGLE call site.
    /// They are also the cleanest inline candidates in that tally: pure functions of guest
    /// memory, with no kernel object, no handle to resolve and no host state anywhere near
    /// them ([[vitaslop-guest-state-is-what-makes-a-call-inlinable]]).
    ///
    /// # Why it is exactly the host call
    /// The handler reads the source into a buffer and then writes that buffer to the
    /// destination, which is `memmove` semantics - an overlapping copy sees the ORIGINAL
    /// bytes. `memory.copy` is specified the same way, so the two agree on the overlapping
    /// case as well as the ordinary one. Picking a form with `memcpy` semantics instead
    /// would differ from the handler exactly where C says the program is already wrong,
    /// which is the worst place to differ.
    ///
    /// # The length is part of the guard
    /// Both pointers AND the length are checked, because the length is what decides how far
    /// past a pointer the access reaches: `len <= mem_bytes` and `p - base <= mem_bytes - len`
    /// for each pointer. A rejected call runs the handler, which keeps its own truncating
    /// behaviour at the end of memory (`GuestCtx::read_bytes` clamps and returns short) rather
    /// than having it approximated here.
    MemCopy,
    /// `memset(r0, r1 & 0xff, r2); r0 unchanged` - fill the r2 bytes at the pointer in r0
    /// with the low byte of r1, and leave the destination in r0 as the return value.
    ///
    /// `memory.fill` truncates its value operand to a byte, which is exactly what the
    /// handler's `ch as u8` does. Guarded on the destination and the length like
    /// [`InlineOp::MemCopy`].
    MemFill,
    /// `r0 = memcmp(r0, r1, r2)` - the difference of the first differing byte pair over the
    /// r2 bytes at the pointers in r0 and r1, or 0 when they are equal.
    ///
    /// The only form that emits a LOOP, and the only one that can: the count is a runtime
    /// value and there is no bulk instruction for a comparison. See
    /// [`crate::emit`] for why that loop carries no fuel check.
    ///
    /// The answer is the ZERO-EXTENDED difference `a[i] - b[i]`, which is what the handler
    /// computes and what C requires the SIGN of. [`mem_compare`] is the one definition of
    /// it, called by the handler and asserted against the emitted code.
    MemCompare,
}

/// The answer `sceClibMemcmp` gives for `a` and `b`: the difference of the first differing
/// byte pair, or 0.
///
/// One definition, called by the host handler and asserted against the emitted
/// [`InlineOp::MemCompare`] loop, so the two cannot drift. Written here rather than in the
/// runtime because the emitter is what has to reproduce it, and a definition that lives
/// beside the thing it defines is one a reader of either can find.
///
/// The bytes are zero-extended before subtracting - `0xff` against `0x01` is +254, not -2 -
/// which is what C requires the sign of and what the ARM code the guest would otherwise have
/// run computes.
pub fn mem_compare(a: &[u8], b: &[u8]) -> i32 {
    for (p, q) in a.iter().zip(b.iter()) {
        if p != q {
            return *p as i32 - *q as i32;
        }
    }
    0
}

/// Byte offsets of the four words an [`InlineOp::LwMutexLock`] reads out of a lightweight
/// mutex's guest work area.
///
/// The transpiler does not choose these - the runtime owns the layout (see
/// `vitaslop_runtime::vita::lwwork`) and passes it in, so the emitted code and the host
/// handlers read one set of numbers rather than two copies that can drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LwMutexLayout {
    /// The work area's own address, stamped at create. See the `id == r0` term.
    pub id: u32,
    /// The owning thread's SceUID, meaningful only while `count` is non-zero.
    pub owner: u32,
    /// Recursion depth; zero means free.
    pub count: u32,
    /// How many threads the host has parked on this mutex. Non-zero sends every
    /// operation to the host, which is the only side that can wake one.
    pub waiters: u32,
}

/// Where an [`InlineOp::ReserveUniformBuffer`] finds every word it reads and writes.
///
/// The transpiler does not choose any of these - the runtime owns both structures
/// (`vitaslop_runtime::vita::gxmctx` and `::gxmprog`) and passes them in, so the emitted
/// code and the host handler read one set of numbers rather than two copies that can drift.
/// The `*_at` offsets are byte offsets from the pointer named in their prefix: `ctx_` from
/// the context in r0, `prog_` from the bound program handle read out of it.
/// Where an [`InlineOp::BindPrecomputedState`] finds everything it copies: the context
/// block it writes (r0), the precomputed-state STRUCT it reads (r1), and the state's
/// ARRAYS block, whose guest address sits in the struct.
///
/// One layout type serves both stages: the vertex bind copies the non-default
/// uniform-buffer table and writes the vertex uniform record; the fragment bind copies the
/// 16-unit texture-binding array, writes the fragment uniform record, and additionally
/// stores the program HANDLE into the context (`has_prog`) - binding a fragment state
/// leaves the context bound to its program, exactly as `sceGxmSetFragmentProgram` would.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BindStateLayout {
    /// The context block's identity stamp, and the value it must hold.
    pub ctx_magic_at: u32,
    pub ctx_magic: u32,
    /// The state STRUCT's identity stamp (stage-specific), and the value it must hold. A
    /// struct without it - never initialised, or written by an engine version with a
    /// different packing - runs the handler, which defines that case.
    pub st_magic_at: u32,
    pub st_magic: u32,
    /// Where the state struct keeps the guest address of its ARRAYS block.
    pub st_block_at: u32,
    /// The state struct's default-uniform-buffer address, its memoised size in bytes, and
    /// its `SceGxmProgram *` - the three words the stage's uniform record receives.
    pub st_buf_at: u32,
    pub st_size_at: u32,
    pub st_header_at: u32,
    /// The state struct's program HANDLE word (read only when `has_prog`).
    pub st_handle_at: u32,
    /// The stage's three-word uniform record in the context block.
    pub ctx_record: u32,
    /// The bulk copy: `copy_bytes` bytes from the arrays block to `ctx + copy_dst`.
    pub copy_dst: u32,
    pub copy_bytes: u32,
    /// >>> WHEN NON-ZERO, THE COPY IS PER-SLOT AND SKIPS EMPTY ONES: `copy_bytes` is treated
    /// >>> as `copy_bytes / copy_slot_stride` slots of this size, and a slot whose FIRST WORD
    /// >>> is zero is left alone instead of copied.
    ///
    /// The fragment stage's copy is the sixteen-unit TEXTURE array, and that array is a block
    /// this engine allocates and zeroes ([`gxmstate`-side `ensure_state_block`]); the only
    /// thing that ever fills a slot is a `sceGxmPrecomputedFragmentStateSetTexture`. So a zero
    /// slot is an UNWRITTEN value, not a guest request to unbind, and copying it over the
    /// context destroys a binding the guest made through `sceGxmSetFragmentTexture`.
    ///
    /// MEASURED on PCSE00120: 19,603 direct binds, **0** textures ever put into a state, ~1,286
    /// state binds a frame - and its title-screen art did not draw, because the state binds
    /// erased the sprite texture between the bind and the immediate `sceGxmDraw` that sampled
    /// it. The handler skips empty slots for that reason and this exists so the emitted form
    /// does the same: the two must leave byte-identical state, which is the whole contract the
    /// inline forms rest on.
    ///
    /// Zero keeps the plain bulk `memory.copy` - which is what the VERTEX stage wants, because
    /// its copy is the uniform-buffer TABLE, where a zero entry really does mean "no buffer
    /// bound" and replacing it is correct.
    pub copy_slot_stride: u32,
    /// Context slot the program handle is stored to, when `has_prog`.
    pub ctx_prog: u32,
    pub has_prog: bool,
}

impl BindStateLayout {
    /// The highest offset reached from the CONTEXT pointer - the guard admits the last
    /// word, not the first.
    pub fn ctx_top(self) -> u32 {
        let mut top = self.ctx_magic_at.max(self.ctx_record + 8);
        top = top.max(self.copy_dst + self.copy_bytes - 4);
        if self.has_prog {
            top = top.max(self.ctx_prog);
        }
        top
    }

    /// The highest offset reached from the STATE STRUCT pointer.
    pub fn st_top(self) -> u32 {
        let mut top =
            self.st_magic_at.max(self.st_block_at).max(self.st_buf_at).max(self.st_size_at).max(self.st_header_at);
        if self.has_prog {
            top = top.max(self.st_handle_at);
        }
        top
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniformRingLayout {
    /// Offset of the context block's identity stamp, and the value it must hold.
    pub ctx_magic_at: u32,
    pub ctx_magic: u32,
    /// Offset of the bound program HANDLE within the context block. Which stage this form
    /// serves is decided here and at [`Self::record`], and nowhere else.
    pub ctx_program: u32,
    /// The ring's three words: base, one-past-the-end, and the next free byte. All guest
    /// ADDRESSES, so the bumped cursor is the answer the guest wants with no rebasing.
    pub ctx_ring_base: u32,
    pub ctx_ring_end: u32,
    pub ctx_ring_cursor: u32,
    /// Offset of this stage's three-word bound-uniform record: `[buffer, size, header]`.
    pub record: u32,
    /// Offset of the handle's identity stamp, and the value it must hold.
    pub prog_magic_at: u32,
    pub prog_magic: u32,
    /// The handle's memoised `default uniform buffer` size in bytes (recorded), the bytes
    /// a reserve takes from the ring for it (never smaller), and the `SceGxmProgram *`.
    pub prog_size: u32,
    pub prog_alloc: u32,
    pub prog_header: u32,
    /// Alignment every handed-out block gets. A power of two; the ring base has it too, so
    /// aligning the absolute cursor and aligning an offset from the base agree.
    pub align: u32,
}

impl UniformRingLayout {
    /// The highest offset reached from the CONTEXT pointer, which is what its bound must be
    /// computed against - the guard has to admit the last word, not the first.
    pub fn ctx_top(self) -> u32 {
        self.ctx_magic_at
            .max(self.ctx_program)
            .max(self.ctx_ring_base)
            .max(self.ctx_ring_end)
            .max(self.ctx_ring_cursor)
            .max(self.record + 8)
    }

    /// The highest offset reached from the PROGRAM HANDLE, for the same reason.
    pub fn prog_top(self) -> u32 {
        self.prog_magic_at.max(self.prog_size).max(self.prog_alloc).max(self.prog_header)
    }
}

/// Where an [`InlineOp::SetUniformData`] finds the two records it reads.
///
/// As with [`UniformRingLayout`], the runtime owns every number here (the GXM parameter
/// record's own layout, and its fallback SA bank's) and passes them in, so the emitted code
/// and the handler cannot drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniformDataLayout {
    /// Host-mirror slot holding the SA bank's guest address.
    pub bank_slot: u32,
    /// Byte offsets within the bank: its high-water float count, and its first float.
    pub bank_len_at: u32,
    pub bank_data_at: u32,
    /// Byte offset of the parameter record's packed word, and the field within it that
    /// carries the component TYPE - plus the value of that field which means F16, the one
    /// case this form must refuse.
    pub param_packed_at: u32,
    pub type_shift: u32,
    pub type_mask: u32,
    pub f16_type: u32,
    /// Byte offset of the parameter's `resource_index` - the register it starts at.
    pub param_index_at: u32,
    /// The bank's capacity in registers, which is also the ceiling the handler refuses past.
    pub max_regs: u32,
}

impl UniformDataLayout {
    /// The highest offset reached from the PARAMETER pointer, which is what its bound is
    /// computed against.
    pub fn param_top(self) -> u32 {
        self.param_packed_at.max(self.param_index_at)
    }

    /// The highest byte offset reached from the BANK pointer: its last float.
    pub fn bank_top(self) -> u32 {
        self.bank_len_at.max(self.bank_data_at + self.max_regs * 4 - 4)
    }
}

impl LwMutexLayout {
    /// The highest offset the layout reaches, which is what a pointer bound must be
    /// computed against - the guard has to admit the LAST word, not the first.
    pub fn top(self) -> u32 {
        self.id.max(self.owner).max(self.count).max(self.waiters)
    }
}

impl InlineOp {
    /// The operation's meaning, over the word it reads. The single definition of what
    /// the emitted code must compute, so a test can hold a host handler to it.
    ///
    /// For the pair forms this is the LOW word only; the high word is `mirror[slot + 1]`
    /// unchanged, which needs no definition. For [`InlineOp::LoadScaled`] this is
    /// meaningful only when [`InlineOp::falls_back`] is false - the guarded case has no
    /// inline answer by construction.
    ///
    /// The STORE forms have no answer in r0 beyond the success code, so their meaning is
    /// [`InlineOp::store_offset`] instead: what they write and where. `eval` returns 0 for
    /// them, which IS the r0 they leave.
    pub fn eval(self, word: u32) -> u32 {
        match self {
            InlineOp::LoadShiftMask { shift, mask, plus, .. } => {
                ((word >> shift) & mask).wrapping_add(plus)
            }
            // The mirror word IS the answer; the host computed it.
            InlineOp::LoadMirror { .. } | InlineOp::LoadMirrorParking { .. } => word,
            InlineOp::LoadScaled { shl, .. } => word << shl,
            // The pair forms deliver the mirror words untouched, wherever they land.
            InlineOp::StoreMirrorPair { .. } | InlineOp::LoadMirrorPair { .. } => word,
            // A void setter: the guest gets the success code and nothing else.
            InlineOp::StoreArg { .. }
            | InlineOp::StoreArgIndexed { .. }
            | InlineOp::StoreArgField { .. }
            | InlineOp::StoreArgFieldInPlace { .. } => 0,
            // Void run setters: the guest gets the success code. What they WRITE is a run of
            // argument words, which a one-word `eval` cannot express - the execution test in
            // `vitaslop-native/tests/inline_imports.rs` holds both to their definition.
            InlineOp::StoreVfpRun { .. } | InlineOp::StoreArgRun { .. } => 0,
            // A copy: what it writes comes from the SOURCE pointer, not from a loaded word,
            // so a one-word `eval` cannot express it. Its meaning is the layout it writes -
            // held against the handler by `the_texture_binding_layout_is_closed`.
            InlineOp::CopyArgIndexed { .. } => 0,
            // A taken lock returns success; a refused one never gets here (the host call
            // answers instead). Their real definition is a whole work-area state machine
            // over four words, which `eval`'s one-word signature cannot express - see
            // `vitaslop_runtime::vita::lwwork::fast_lock`, which the emitted code is held
            // against directly.
            InlineOp::LwMutexLock { .. } | InlineOp::LwMutexUnlock { .. } => 0,
            // A successful reserve returns 0, and a refused one never gets here (the host
            // call answers instead). Its real meaning is a bump over two structures, which
            // `eval`'s one-word signature cannot express - the execution test in
            // `vitaslop-native/tests/inline_imports.rs` is what holds it to its handler.
            InlineOp::ReserveUniformBuffer { .. } => 0,
            // A successful uniform write returns 0; its meaning is two byte ranges, which
            // `eval`'s one-word signature cannot express any more than the copy form's.
            InlineOp::SetUniformData { .. } => 0,
            // A successful bind returns 0; its meaning is a copy between two guest
            // structures, held to its handler by the execution test and the runtime's
            // layout equivalence tests.
            InlineOp::BindPrecomputedState { .. } => 0,
            // A bulk form's meaning is a RANGE of memory, which a one-word `eval` cannot
            // express any more than it can express the copy form's. `MemCopy` and `MemFill`
            // return the destination they were handed, so 0 here is not their r0 - the
            // emitted code leaves r0 alone, and the execution test in
            // `vitaslop-native/tests/inline_imports.rs` is what holds all three to their
            // handlers.
            InlineOp::MemCopy | InlineOp::MemFill | InlineOp::MemCompare => 0,
            // The whole answer, and it does not depend on the word - there is no word. The
            // constant IS the definition, so `eval` returning it is exact rather than a
            // stand-in the way the zeros above are.
            InlineOp::RetConst { value } => value,
            // r0 is left ALONE, which a one-word `eval` cannot express any more than it can
            // express a store form's range. Its meaning is that nothing happens.
            InlineOp::Nop => 0,
        }
    }

    /// Whether the loaded `word` sends this op to the host call instead of computing an
    /// answer inline. Only [`InlineOp::LoadScaled`] can, and a test pins the boundary:
    /// the point of the guard is that the handler, not this, defines the clamped case.
    ///
    /// This is about a loaded WORD. [`InlineOp::StoreArgIndexed`] also has a value guard,
    /// but on its INDEX argument rather than on anything it read, so it has its own
    /// predicate ([`InlineOp::falls_back_on_index`]) instead of overloading this one - two
    /// different quantities under one name is how an off-by-one gets written.
    pub fn falls_back(self, word: u32) -> bool {
        match self {
            InlineOp::LoadScaled { max, .. } => word > max,
            _ => false,
        }
    }

    /// Whether index argument `index` sends this op to the host call. Only
    /// [`InlineOp::StoreArgIndexed`] has an index at all; every other form answers false
    /// for every index, which is trivially true of a form that has none.
    pub fn falls_back_on_index(self, index: u32) -> bool {
        match self {
            InlineOp::StoreArgIndexed { count, .. } | InlineOp::CopyArgIndexed { count, .. } => {
                index >= count
            }
            _ => false,
        }
    }

    /// Byte offset from the pointer argument at which the word is read, for the forms
    /// that read through a guest pointer. `None` for a form that does not take one, and
    /// for the forms that WRITE through it - see [`InlineOp::store_offset`].
    pub fn offset(self) -> Option<u32> {
        match self {
            InlineOp::LoadShiftMask { offset, .. } => Some(offset),
            InlineOp::LoadScaled { offset, .. } => Some(offset),
            // Writes through r0 rather than reading through it, so it has no read offset
            // even though it is a pointer form. `emit_import` guards it on its own terms.
            InlineOp::StoreMirrorPair { .. } => None,
            InlineOp::StoreArg { .. } | InlineOp::StoreArgIndexed { .. } => None,
            // Reads the word it is about to rewrite, so its offset is a `store_offset` -
            // reporting it here would make a getter test read the field mid-update.
            InlineOp::StoreArgField { .. } | InlineOp::StoreArgFieldInPlace { .. } => None,
            // Reads through r2 and writes through r0, so neither pointer's offset is "the"
            // offset.
            InlineOp::CopyArgIndexed { .. } => None,
            InlineOp::LoadMirror { .. }
            | InlineOp::LoadMirrorParking { .. }
            | InlineOp::LoadMirrorPair { .. } => None,
            // Take no pointer and read nothing.
            InlineOp::RetConst { .. } | InlineOp::Nop => None,
            // Reads four words and writes two, so no single offset describes it.
            InlineOp::LwMutexLock { .. } | InlineOp::LwMutexUnlock { .. } => None,
            // Reaches from the pointer itself for a length the guest supplies; there is no
            // fixed offset to name.
            InlineOp::MemCopy | InlineOp::MemFill | InlineOp::MemCompare => None,
            // Reads through two pointers and writes through three; no single offset
            // describes it, and its layout is what a test holds it to instead.
            InlineOp::ReserveUniformBuffer { .. } => None,
            // Reads a record, a stack word and a source buffer, and writes two ranges.
            InlineOp::SetUniformData { .. } => None,
            // Write-only runs; their offsets are store offsets.
            InlineOp::StoreVfpRun { .. } | InlineOp::StoreArgRun { .. } => None,
            // Reads a struct and a block, writes the context; no single offset names it.
            InlineOp::BindPrecomputedState { .. } => None,
        }
    }

    /// Byte offset from the pointer argument at which an argument-storing form WRITES its
    /// word - for [`InlineOp::StoreArgIndexed`], the offset of element ZERO.
    ///
    /// Deliberately separate from [`InlineOp::offset`]: a test that holds a getter to its
    /// handler reads the word at `offset()`, and a test that holds a SETTER to its handler
    /// reads the word at `store_offset()` AFTER running it. Same number shape, opposite
    /// direction, and conflating them would make the setter test read the word the handler
    /// was about to overwrite.
    pub fn store_offset(self) -> Option<u32> {
        match self {
            InlineOp::StoreArg { offset } => Some(offset),
            InlineOp::StoreArgIndexed { offset, .. } => Some(offset),
            InlineOp::CopyArgIndexed { offset, .. } => Some(offset),
            InlineOp::StoreArgField { offset, .. } => Some(offset),
            InlineOp::StoreArgFieldInPlace { offset, .. } => Some(offset),
            _ => None,
        }
    }

    /// The host-mirror slot this op reads, if it reads one. For a pair form this is the
    /// LOW slot; the layout must also reserve `slot + 1`, which
    /// [`InlineOp::top_mirror_slot`] reports.
    pub fn mirror_slot(self) -> Option<u32> {
        match self {
            InlineOp::LoadShiftMask { .. } | InlineOp::LoadScaled { .. } => None,
            InlineOp::StoreArg { .. } | InlineOp::StoreArgIndexed { .. } => None,
            InlineOp::StoreArgField { .. } | InlineOp::StoreArgFieldInPlace { .. } => None,
            InlineOp::StoreVfpRun { .. } | InlineOp::StoreArgRun { .. } => None,
            InlineOp::CopyArgIndexed { .. } => None,
            InlineOp::LoadMirror { slot } => Some(slot),
            // Names the VALUE slot; the budget slot is covered by `top_mirror_slot`, which
            // is what the layout pass sizes the block from.
            InlineOp::LoadMirrorParking { slot, .. } => Some(slot),
            InlineOp::StoreMirrorPair { slot } | InlineOp::LoadMirrorPair { slot } => Some(slot),
            // The lock forms read the mirror too - the CURRENT THREAD, which is the one
            // fact about the take that is not in the work area. Naming the slot here is
            // what makes the layout pass reserve the block for them as well; a lock form
            // omitted from this list would read a word nobody ever writes and take every
            // mutex on behalf of thread zero.
            InlineOp::LwMutexLock { thread_slot, .. }
            | InlineOp::LwMutexUnlock { thread_slot, .. } => Some(thread_slot),
            InlineOp::MemCopy | InlineOp::MemFill | InlineOp::MemCompare => None,
            // Read nothing at all, mirror included.
            InlineOp::RetConst { .. } | InlineOp::Nop => None,
            // Everything it reads is in the two guest structures it is handed.
            InlineOp::ReserveUniformBuffer { .. } => None,
            // Reads the SA bank's ADDRESS out of the block - the one slot that is not a
            // value the guest asked for. See `vitaslop_runtime::vita::mirror::SLOT_SA_BANK`.
            InlineOp::SetUniformData { layout } => Some(layout.bank_slot),
            // Everything it reads is in the guest structures it is handed.
            InlineOp::BindPrecomputedState { .. } => None,
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
            // BOTH its slots have to be inside the block: the budget is written by the same
            // snapshot and decremented by the emitted code.
            InlineOp::LoadMirrorParking { slot, budget } => Some(slot.max(budget)),
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
    /// Wasm operators emitted per GUEST INSTRUCTION for this build.
    ///
    /// A host should report it, because it is the hidden term in the emulated CPU's
    /// speed: the game clock is charged per unit of fuel and a unit of fuel is an
    /// executed wasm operator, so the emulated Vita runs at `fuel rate / expansion`.
    /// Improve the codegen and the console gets faster unless the calibration moves with
    /// it. See [`emit::Expansion`].
    pub expansion: emit::Expansion,
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

    let funcs = ordered
        .iter()
        .map(|f| FuncExport {
            addr: f.addr,
            export: abi::func_export(f.addr),
        })
        .collect();
    let emit::EmitOutput { wasm, mem_pages, arm_word_off, mirror_off, dirty_off, expansion } =
        emit::emit_module(
            ordered,
            &func_index,
            program.base,
            program.mem_bytes,
            program.inline_imports,
            program.import_memory,
        );
    Ok(Artifact { wasm, funcs, mem_pages, arm_word_off, mirror_off, dirty_off, expansion })
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
    /// Guest addresses INSIDE lifted functions where an instruction did not decode, so
    /// the block was cut there and a trapping block put in its place.
    ///
    /// **This is the decode-gap list that matters, and it is not the diagnostic report's.**
    /// [`transpile_report`] walks the call graph from the entry points, so it never
    /// reaches a function whose only caller is a vtable slot - and a C++ engine keeps its
    /// hot per-object work exactly there. Those functions are lifted here anyway (the
    /// pointer scan finds them), gaps and all, so this is the only place their gaps
    /// appear. A gap on a hot path does not look like a gap at runtime: the block simply
    /// ends early, the rest of the loop never runs, and the failure surfaces as a wild
    /// pointer somewhere else entirely.
    pub decode_gaps: Vec<u32>,
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
    // Decode gaps inside functions that DID lift; see `LenientArtifact::decode_gaps`.
    let mut decode_gaps: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
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
                    decode_gaps.extend(found.trap_leaders.iter().copied());
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

    report_lifted_size(&funcs);
    let ordered: Vec<ir::Func> = funcs.into_values().collect();
    let func_index: BTreeMap<u32, u32> = ordered
        .iter()
        .enumerate()
        .map(|(i, f)| (f.addr, emit::IMPORT_FUNCS + i as u32))
        .collect();
    let funcs = ordered
        .iter()
        .map(|f| FuncExport { addr: f.addr, export: abi::func_export(f.addr) })
        .collect();
    let emit::EmitOutput { wasm, mem_pages, arm_word_off, mirror_off, dirty_off, expansion } = emit::emit_module(
        ordered,
        &func_index,
        program.base,
        program.mem_bytes,
        program.inline_imports,
        program.import_memory,
    );
    stubbed.sort_unstable();
    let stub_wasm_indices = stubbed.iter().map(|a| func_index[a]).collect();
    LenientArtifact {
        artifact: Artifact { wasm, funcs, mem_pages, arm_word_off, mirror_off, dirty_off, expansion },
        stubbed,
        stub_wasm_indices,
        decode_gaps: decode_gaps.into_iter().collect(),
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
            // Guest instructions committed across the whole module, so the clock's half of
            // every commit is audited too. See where it is asserted, below the loop.
            let mut lifted: i64 = 0;
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
                let mut instructions: i64 = 0;
                let mut owed: i64 = 1;
                let mut i = 0;
                while i < ops.len() {
                    // A commit:  global.get $work ; i64.const PACKED ; i64.add ; global.set $work
                    //
                    // PACKED carries BOTH counters (see `abi::pack_work`): guest
                    // instructions in the high 32 bits, operators in the low 32. One add
                    // advances both, which is what makes billing the clock in guest
                    // instructions cost no extra code at all.
                    if let [
                        Operator::GlobalGet { global_index: g },
                        Operator::I64Const { value },
                        Operator::I64Add,
                        Operator::GlobalSet { global_index: h },
                    ] = ops[i..(i + 4).min(ops.len())]
                    {
                        if g == abi::WORK_GLOBAL && h == abi::WORK_GLOBAL {
                            let ops_half = value & abi::WORK_OPS_MASK;
                            let instr_half = value >> abi::WORK_INSTR_SHIFT;
                            assert!(
                                ops_half > 0 || instr_half > 0,
                                "{what}: a commit of 0 is dead code"
                            );
                            // The halves must not have bled into each other. Both count
                            // UP precisely so the add cannot borrow; a negative here would
                            // mean that reasoning is wrong.
                            assert!(instr_half >= 0, "{what}: the instruction half went negative");
                            charged += ops_half;
                            instructions += instr_half;
                            i += 4;
                            continue;
                        }
                    }
                    // A back-edge test:
                    //   global.get $work ; i64.const MASK ; i64.and ; i64.const INTERVAL ;
                    //   i64.ge_u ; if ; i32.const -1 ; call $host ;
                    //   global.get $work ; i64.const !MASK ; i64.and ; global.set $work ; end
                    if let [
                        Operator::GlobalGet { global_index: g },
                        Operator::I64Const { value: mask },
                        Operator::I64And,
                        Operator::I64Const { value: interval },
                    ] = ops[i..(i + 4).min(ops.len())]
                    {
                        if g == abi::WORK_GLOBAL && mask == abi::WORK_OPS_MASK {
                            assert_eq!(
                                interval, INTERVAL as i64,
                                "{what}: a back-edge test must compare the whole interval"
                            );
                            // The yield must clear ONLY the operator half - clearing the
                            // whole global would reset the clock's instruction total every
                            // quantum and the game clock would never advance.
                            assert!(
                                matches!(
                                    ops[i + 9],
                                    Operator::I64Const { value } if value == !abi::WORK_OPS_MASK
                                ),
                                "{what}: a yield must preserve the guest-instruction half"
                            );
                            i += 12;
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
                // The clock's half rides the same commits, so it is auditable by the same
                // walk. Accumulated across the module rather than asserted per body: the
                // dispatcher and the reset thunk lift no guest code and correctly commit
                // no instructions, so a per-body assert would fire on them.
                lifted += instructions;
            }
            // The module lifted guest code, so its commits must have carried a guest
            // instruction count. Zero here means the clock's half of the packed commit
            // was dropped - a game clock that never advances, which is the livelock
            // `Signal::fuel` documents arriving by a new route.
            assert!(
                lifted > 0,
                "{what}: the module committed no guest instructions at all, so the clock \
                 would never advance"
            );
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

    /// The lightweight-mutex forms must contain NO SUSPENSION POINT, and this is the whole
    /// reason a non-atomic read-modify-write is safe in guest code.
    ///
    /// They load the count, test it, and store `count + 1`. If the scheduler could take the
    /// baton away in between, two threads would both read "free" and both take the mutex -
    /// a mutual-exclusion violation with no error and no host call, surfacing later as
    /// corrupt data somewhere else. On hardware that window does not exist because the
    /// uncontended take is a compare-and-swap; here it does not exist because of where the
    /// two engines can preempt, and those are the only two facts holding it up:
    ///
    /// - **Native (wasmtime).** `fuel_check` - the only thing that yields - is emitted at
    ///   FUNCTION ENTRY and LOOP HEADERS only (`wasmtime-internal-cranelift`,
    ///   `fuel_function_entry` / `translate_loop_header`). An `if`/`else`/`end` merely
    ///   increments a function-local counter; it cannot yield. A `call` can, because the
    ///   callee may.
    /// - **Browser.** [`emit::emit_fuel_check`] is emitted on BACK EDGES only, and says so.
    ///
    /// So the sequence is safe exactly while it contains no loop and no call on the path
    /// that writes. That is what this asserts. It will fail if anyone puts a loop in the
    /// emitted form, and it is the place to come back to if either engine ever gains a
    /// finer preemption point - at which point this needs real atomics, not a re-test.
    #[test]
    fn a_lock_form_has_no_suspension_point() {
        use wasmparser::{Operator, Payload};
        // Thumb `bl` to the import stub at 0x10010, then `bx lr`.
        let code: [u8; 8] = [
            0x00, 0xf0, 0x06, 0xf8, // bl 0x10010
            0x70, 0x47, // bx lr
            0x00, 0x00,
        ];
        let layout = LwMutexLayout { id: 0, owner: 4, count: 8, waiters: 12 };
        for op in [
            InlineOp::LwMutexLock { layout, thread_slot: 3 },
            InlineOp::LwMutexUnlock { layout, thread_slot: 3 },
        ] {
            let artifact = transpile(&Program {
                code: &code,
                base: 0x10000,
                thumb: true,
                entries: &[0x10000],
                arm_entries: &[],
                externs: &[Extern { addr: 0x10010, import: 0 }],
                redirects: &[],
                inline_imports: &[InlineImport { import: 0, op }],
                noreturn_svc: &[],
                mem_bytes: 0x20000,
                discover_code_pointers: false,
                import_memory: false,
            })
            .expect("transpile");
            wasmparser::validate(&artifact.wasm).expect("valid wasm");
            // The FIRST body is the translated guest function; the rest is emitter
            // bookkeeping (the dispatcher, `emit_reset`) that no guest lock runs through.
            let body = wasmparser::Parser::new(0)
                .parse_all(&artifact.wasm)
                .filter_map(|p| match p.expect("parse") {
                    Payload::CodeSectionEntry(b) => Some(b),
                    _ => None,
                })
                .next()
                .expect("the guest function is emitted");
            let ops: Vec<Operator> = body
                .get_operators_reader()
                .expect("operators")
                .into_iter()
                .collect::<Result<_, _>>()
                .expect("operators");
            let loops = ops.iter().filter(|o| matches!(o, Operator::Loop { .. })).count();
            assert_eq!(loops, 0, "{op:?} must emit no loop - a loop header is a yield point");
            // Two calls, and both are the FALLBACK: one from the pointer guard and one from
            // the predicate. Any third call would be on the served path, where a yield is
            // exactly the race this test exists to rule out.
            let calls = ops.iter().filter(|o| matches!(o, Operator::Call { .. })).count();
            assert_eq!(calls, 2, "{op:?} must call the host on its two refusal arms and nowhere else");
        }
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

    /// A long straight-line run of register arithmetic, transpiled both ways.
    ///
    /// Three things have to be true at once and each fails silently on its own:
    /// promotion must actually FIRE (a policy that promotes nothing passes every
    /// correctness test there is), the unpromoted module must be BYTE-IDENTICAL to one
    /// built before promotion existed, and the promoted module must still be valid wasm
    /// with its register file properly written back.
    ///
    /// The straight-line shape is the point: the policy only promotes within a run
    /// bounded by branches and calls, so a corpus of one-instruction cases - which is
    /// what the ARM corpus is - exercises none of it.
    #[test]
    fn promotion_fires_on_a_straight_run_and_changes_nothing_when_off() {
        // add r0,r0,r1 / add r0,r0,r1 / add r0,r0,r1 / add r0,r0,r1 / bx lr.
        // r0 is touched eight times and r1 four, both far past the threshold.
        const ADD_R0_R0_R1: [u8; 4] = [0x01, 0x00, 0x80, 0xe0];
        const BX_LR: [u8; 4] = [0x1e, 0xff, 0x2f, 0xe1];
        let mut code = Vec::new();
        for _ in 0..4 {
            code.extend_from_slice(&ADD_R0_R0_R1);
        }
        code.extend_from_slice(&BX_LR);

        let build = |promote: bool| -> Vec<u8> {
            set_promote_registers(promote);
            let a = transpile(&Program {
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
            wasmparser::validate(&a.wasm).expect("valid wasm");
            a.wasm
        };

        let plain = build(false);
        let promoted = build(true);

        // Count register-file traffic in each. `local.get`/`local.set`/`local.tee` of the
        // promoted range replace `global.get`/`global.set` of globals 0..20.
        fn core_global_ops(wasm: &[u8]) -> usize {
            use wasmparser::{Operator, Parser, Payload};
            let mut n = 0;
            for payload in Parser::new(0).parse_all(wasm) {
                if let Ok(Payload::CodeSectionEntry(body)) = payload {
                    let mut reader = body.get_operators_reader().unwrap();
                    while let Ok(op) = reader.read() {
                        if let Operator::GlobalGet { global_index } | Operator::GlobalSet
                        { global_index } = op
                        {
                            if promote::is_core(global_index) {
                                n += 1;
                            }
                        }
                    }
                }
            }
            n
        }

        let plain_ops = core_global_ops(&plain);
        let promoted_ops = core_global_ops(&promoted);
        assert!(
            promoted_ops < plain_ops,
            "promotion must actually fire on a straight run: {plain_ops} core-global \
             accesses unpromoted, {promoted_ops} promoted"
        );

        // And it must be entirely absent when not asked for. This is the guarantee that
        // lets the two arms be compared at all: the OFF arm is the build that shipped.
        set_promote_registers(false);
        let plain_again = build(false);
        assert_eq!(plain, plain_again, "an unpromoted build must be deterministic");
    }
}

/// Report the size of the lifted program, once, before it is handed to the emitter.
///
/// # Why this exists
/// The transpiler is the allocation peak of the whole system and nothing said how big it was.
/// MEASURED on one retail title: a **9,975 MB** desktop working-set peak and a ~5,000 MB steady
/// state, for a 4.9 MB ARM module - and in a browser, where wasm32 caps the address space at
/// 4 GiB and linear memory never shrinks, the same build died in `lower::discover` with
/// `std::alloc::rust_oom`. Three other titles peak at 1.4-2.7 GB, so this is a property of the
/// TITLE's code size, not a constant. A number that decides whether a title can run at all
/// should not have to be recovered from a stack trace.
///
/// Statements are counted, not measured: `Stmt` is an enum whose size the compiler is free to
/// change, so a byte estimate here would be a number that silently goes stale. The counts are
/// what scale with the title, and `size_of` is printed beside them so the product is available
/// without pinning it.
fn report_lifted_size(funcs: &std::collections::BTreeMap<u32, ir::Func>) {
    let blocks: usize = funcs.values().map(|f| f.blocks.len()).sum();
    let stmts: usize = funcs.values().flat_map(|f| &f.blocks).map(|b| b.stmts.len()).sum();
    let arm: u64 = funcs.values().flat_map(|f| &f.blocks).map(|b| b.arm_count as u64).sum();
    let stubs = funcs.values().filter(|f| f.stub).count();
    eprintln!(
        "transpile: lifted {} functions ({stubs} stubs), {blocks} blocks, {stmts} statements, \
         {arm} guest instructions; size_of::<Stmt>()={} B, so the statement vectors alone are \
         about {:.0} MB",
        funcs.len(),
        std::mem::size_of::<ir::Stmt>(),
        (stmts * std::mem::size_of::<ir::Stmt>()) as f64 / 1e6,
    );
}
