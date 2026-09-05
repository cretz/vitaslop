//! Vita host-call implementations, grouped by module (one file per Sce* module,
//! mirroring the vita-headers layout). Each module holds the handler bodies; the
//! single [`dispatch`] match below routes a function NID straight to its handler.

pub mod at9;
pub mod audio;
pub mod audioin;
pub mod camera;
pub mod cfmt;
pub mod ctrl;
pub mod dbg;
pub mod display;
pub mod fiber;
pub mod gesture;
pub mod net;
pub mod fios2;
pub mod gxm;
pub mod gxmctx;
pub mod gxmstate;
pub mod gxmprog;
pub mod http;
pub mod iofilemgr;
pub mod jpeg;
pub mod jpegenc;
pub mod libkernel;
pub mod livearea;
pub mod location;
pub mod lwsync;
pub mod lwwork;
pub mod mirror;
pub mod ngs;
pub mod processmgr;
pub mod pgf;
pub mod pvf;
pub mod sce_xml;
pub mod services;
pub mod sync;
pub mod sysmem;
pub mod threadmgr;
pub mod touch;
pub mod audiodec;
pub mod avcdec;
pub mod video;

use crate::host::{GuestCtx, VitaState};
use crate::nid::{
    audio as audio_nid, audiodec as ad_nid, ctrl as ctrl_nid, dbg as dbg_nid, display as display_nid,
    fiber as fiber_nid, fios2 as fios2_nid, gxm as gxm_nid,
    audioin as audioin_nid, http as http_nid, iofilemgr as io_nid, libkernel as lk_nid,
    livearea as livearea_nid, lwsync as lw_nid, pgf as pgf_nid, xml as xml_nid,
    net as net_nid, ngs as ngs_nid,
    processmgr as pm_nid, pvf as pvf_nid, services as sv_nid, sync as sync_nid,
    sysmem as sm_nid, threadmgr as tm_nid, videodec as vd_nid,
};
use crate::{nid, SvcOutcome};

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

/// The inline form of host import `func_nid` - the code the transpiler emits straight
/// into the guest instead of trapping to the host - or `None` for every NID with real
/// behaviour, which is nearly all of them.
///
/// The one place that decides what may be inlined, because only this crate knows what
/// a NID means. Three shapes qualify and no others:
/// - a pure read through a guest pointer (the GXM reflection getters);
/// - a read of a host value that cannot change while guest code runs (the mirror block -
///   see [`mirror`]);
/// - a read-modify-write of state the guest OWNS, whose contended cases still reach the
///   host. That is the GXM context setters ([`gxmctx`]) and the uncontended half of a
///   lightweight mutex ([`lwwork`]), and the second one is only admissible because on the
///   device that half is userspace too;
/// - a BULK move or compare over guest memory and nothing else (the `sceClibMem*` trio in
///   [`libkernel`]), which is the first shape whose reach the guest chooses rather than the
///   emitter, and so the first whose guard is arithmetic rather than a constant.
pub fn inline_op(func_nid: u32) -> Option<vitaslop_transpiler::InlineOp> {
    if no_inline_imports() {
        return None;
    }
    if func_nid == gxm_nid::SET_FRAGMENT_TEXTURE && no_inline_texture() {
        return None;
    }
    if no_inline_clib() && is_clib_bulk(func_nid) {
        return None;
    }
    if is_uniform_reserve(func_nid) && (no_inline_reserve() || uniform_poison()) {
        return None;
    }
    if func_nid == gxm_nid::SET_UNIFORM_DATA_F && (no_inline_uniform_data() || uniform_watch()) {
        return None;
    }
    gxm::inline_op(func_nid)
        .or_else(|| display::inline_op(func_nid))
        .or_else(|| libkernel::inline_op(func_nid))
        .or_else(|| (!no_inline_lwmutex()).then(|| lwsync::inline_op(func_nid)).flatten())
        .or_else(|| (!no_inline_stubs()).then(|| stub_inline_op(func_nid)).flatten())
}

/// Whether `func_nid`'s handler can only ever CONTINUE, so the transpiler may route the
/// call through the non-suspending trap ([`vitaslop_transpiler::abi::IMPORT_FAST_NAME`]).
///
/// # Why this list is written out rather than derived
/// The admissibility test is the SHAPE of the dispatch arm: every NID here is dispatched
/// as `cont!(handler(..))` in [`dispatch`], an unconditional `Continue` around a handler
/// that returns nothing - so it cannot block, reschedule, flip or exit whatever the guest
/// passes it. (`sceKernelTryLockLwMutex` is the one arm here that returns an outcome, and
/// both of its paths are `Continue`: a contended try-lock fails rather than parks.) A
/// handler that later grows a parking path must leave this list in the same edit; the
/// browser turns a fast call that suspends into a loud run-ending error, never a thread
/// left unparked.
///
/// The list is the race's own host-call profile: draws and scene boundaries are ~90% of a
/// retail race frame's calls, and the rest of it is the allocator, the mixer, the
/// lightweight signal/unlock side of the title's thread handoffs, and input polling.
/// `VITASLOP_NO_FAST_IMPORT=1` routes every call through the suspending trap again (the
/// A/B arm).
pub fn fast_nid(func_nid: u32) -> bool {
    if no_fast_import() {
        return false;
    }
    matches!(
        func_nid,
        gxm_nid::DRAW
            | gxm_nid::DRAW_PRECOMPUTED
            | gxm_nid::BEGIN_SCENE
            | gxm_nid::END_SCENE
            | gxm_nid::SET_VISIBILITY_BUFFER
            | gxm_nid::COLOR_SURFACE_GET_DATA
            | gxm_nid::COLOR_SURFACE_GET_STRIDE_IN_PIXELS
            | gxm_nid::PAD_HEARTBEAT
            | lw_nid::SIGNAL_LW_COND
            // (the lightweight-mutex lock/unlock pair is INLINED - `lwsync::inline_op` - so
            // it never reaches a trap and is not named here)
            | lw_nid::TRY_LOCK_LW_MUTEX
            | sync_nid::UNLOCK_MUTEX
            | sync_nid::SIGNAL_COND
            | lk_nid::CLIB_MSPACE_MALLOC
            | lk_nid::CLIB_MSPACE_MEMALIGN
            | lk_nid::CLIB_MSPACE_FREE
            | lk_nid::GET_TLS_ADDR
            | ngs_nid::VOICE_GET_STATE_DATA
            | ngs_nid::SYSTEM_UPDATE
            | ngs_nid::VOICE_SET_PARAMS_BLOCK
            | pm_nid::POWER_TICK
            | sv_nid::APP_MGR_GET_APP_STATE
            | sv_nid::SYSTEM_GESTURE_UPDATE_TOUCH_RECOGNIZER
            | sv_nid::SYSTEM_GESTURE_GET_TOUCH_EVENTS_COUNT
            | sv_nid::TOUCH_READ
    )
}

fn no_fast_import() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_NO_FAST_IMPORT"))
}

/// The calls whose handler is EXACTLY `ctx.ret(0)` - a constant return and nothing else -
/// so the transpiler can emit the constant and never cross the boundary
/// ([`vitaslop_transpiler::InlineOp::RetConst`]).
///
/// # Why this list is written out rather than derived
/// "The handler currently returns 0" is not the admissibility test; "the handler is DEFINED
/// as a constant return" is. A stub that grows a body later must lose its inline form in the
/// same edit, and a list in one place is what makes that a visible edit rather than a silent
/// divergence between a dispatch arm and the code emitted for it. `the_inlined_stubs_are_stubs`
/// in `vitaslop-native/tests/inline_imports.rs` asserts each one still dispatches to a bare
/// return.
///
/// # What these cost, which is why a no-op is worth inlining at all
/// MEASURED in desktop Chrome on a retail racer's race: `sceNgsPatchGetInfo` and
/// `sceNgsVoicePatchSetVolumesMatrix` are **198 calls per guest frame each**, together 32% of
/// every host call the title makes, at ~1.14 us of pure crossing each. Nothing is computed on
/// either side of that.
///
/// `sceKernelSetGPO` is here for the same reason and needs one extra word: its handler writes
/// `VitaState::gpo`, which NOTHING reads - it is a devkit LED - and logs at `vitaslop::gpo`.
/// It is withheld while that log target is selected, the same way the uniform forms are
/// withheld while their diagnostics are on: an instrument that reports nothing because the
/// call it watches was inlined imitates its own subject
/// [[vitaslop-instrument-failure-imitating-its-subject]].
fn stub_inline_op(func_nid: u32) -> Option<vitaslop_transpiler::InlineOp> {
    use crate::nid::{ctrl as c, ngs as n, sysmem as sm};
    let is_stub = matches!(
        func_nid,
        n::SYSTEM_SET_FLAGS
            | n::SYSTEM_RELEASE
            | n::RACK_RELEASE
            | n::VOICE_RESUME
            | n::VOICE_BYPASS_MODULE
            | n::VOICE_GET_PARAMS_OUT_OF_RANGE
            | n::VOICE_PATCH_SET_VOLUMES_MATRIX
            | n::VOICE_PATCH_SET_VOLUME
            | n::PATCH_GET_INFO
            | n::PATCH_REMOVE_ROUTING
            | n::SYSTEM_LOCK
            | n::SYSTEM_UNLOCK
            | n::AT9_GET_SECTION_DETAILS
            | c::SET_SAMPLING_MODE
    );
    if is_stub {
        return Some(vitaslop_transpiler::InlineOp::RetConst { value: 0 });
    }
    // `sceKernelSetGPO` returns VOID, so its handler leaves r0 holding the argument - which
    // makes it a `Nop` and not a `RetConst { value: 0 }`. Handing back 0 where the call used
    // to hand back its own argument is a different program, and `the_inlined_stubs_are_stubs`
    // catches exactly that (it did).
    (func_nid == sm::SET_GPO && !gpo_log_selected()).then_some(vitaslop_transpiler::InlineOp::Nop)
}

/// Whether the log filter names the `vitaslop::gpo` target, which WITHHOLDS the inline form
/// for `sceKernelSetGPO`. See [`stub_inline_op`].
fn gpo_log_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::log_filter().contains("gpo"))
}

/// `VITASLOP_NO_INLINE_STUBS`: route the constant-return stubs through the host, leaving
/// every other inline form on.
///
/// A SCOPED A/B switch like the others, and here it is mostly a DIAGNOSTIC switch rather than
/// a price tag: an inlined call is absent from the call histogram, and the histogram is how
/// "which unimplemented calls does this title make" gets answered. Set this and every stub is
/// back on the host and back in the counts. The link line that lists what was inlined says the
/// same thing without a rerun.
///
/// Read through [`crate::knobs`] and listed in `OVERRIDABLE`, so a live page on a PHONE can
/// throw it between two runs of one build - which is the machine where a crossing is ~20 us and
/// this family is worth the most. Read at LINK time, so it must be set for the whole run.
fn no_inline_stubs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_NO_INLINE_STUBS"))
}

/// The three NIDs [`no_inline_clib`] scopes over. Named here rather than tested against
/// `libkernel::inline_op`'s output, because the switch is about which CALLS are rerouted and
/// the answer must not change if a fourth form is ever added to that function for a NID this
/// switch was never meant to cover.
fn is_clib_bulk(func_nid: u32) -> bool {
    use crate::nid::libkernel as lk;
    matches!(func_nid, lk::CLIB_MEMCPY | lk::CLIB_MEMSET | lk::CLIB_MEMCMP)
}

/// `VITASLOP_NO_INLINE_CLIB`: route `sceClibMemcpy`, `sceClibMemset` and `sceClibMemcmp`
/// through the host, leaving every other inline form on.
///
/// A SCOPED A/B switch like [`no_inline_lwmutex`], and it is BOTH of the things those two
/// switches are separately: a price tag and a correctness falsifier.
///
/// As a price tag, it is the only way to learn what this family is worth. The three are 13%
/// of a real title's host calls, so turning the whole mechanism off to measure them buys a
/// baseline that also moves ~11,000 other calls a frame and every preemption point with them
/// - against which 500,000 crossings over a run are not separable.
///
/// As a falsifier, it is the switch to reach for first if a title ever renders differently
/// after this change. These forms are the first that WRITE a range the guest sizes, so they
/// are the first that could plausibly corrupt memory rather than merely answer wrongly - and
/// they are also the first that owe the guest-store dirty map a range stamp, which is a
/// browser-only path a desktop run cannot exercise at all. Both arms write the same bytes by
/// construction (the handler and `emit_dirty_range`'s `memory.copy` do the same move), so a
/// picture that CHANGES when this is set is a real divergence and one that does not clears
/// the forms.
///
/// Read through [`crate::knobs`] and listed in `OVERRIDABLE`, so a live page on a PHONE can
/// throw it between two runs of one build - which is the only machine whose answer counts.
/// Read at LINK time, so it must be set for the whole run.
fn no_inline_clib() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_NO_INLINE_CLIB"))
}

/// The two NIDs [`no_inline_reserve`] and the poison knob scope over.
fn is_uniform_reserve(func_nid: u32) -> bool {
    use crate::nid::gxm as g;
    matches!(
        func_nid,
        g::RESERVE_VERTEX_DEFAULT_UNIFORM_BUFFER | g::RESERVE_FRAGMENT_DEFAULT_UNIFORM_BUFFER
    )
}

/// `VITASLOP_NO_INLINE_RESERVE`: route `sceGxmReserve{Vertex,Fragment}DefaultUniformBuffer`
/// through the host, leaving every other inline form on.
///
/// A SCOPED A/B switch like [`no_inline_texture`], and for the same two reasons at once.
/// As a price tag it is the only way to weigh this family: it is 1,189 crossings a frame on
/// one title, and the whole-mechanism switch moves ~11,000 and every preemption point with
/// them, so that baseline cannot separate it. As a falsifier it is the first thing to reach
/// for if a title's uniforms ever look wrong after this change - both arms compute the same
/// buffer from the same three words (`gxmctx::UNIFORM_RING_*` and the handle's memoised
/// size), so a picture that CHANGES when this is set is a real divergence between the two
/// writers and one that does not clears the form.
///
/// Read through [`crate::knobs`] and listed in `OVERRIDABLE`, so a live page on a PHONE can
/// throw it between two runs of one build. Read at LINK time, so it must be set for the
/// whole run.
fn no_inline_reserve() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_NO_INLINE_RESERVE"))
}

/// Whether `VITASLOP_GXM_UNIFORM_POISON` is on, which WITHHOLDS the inline reserve.
///
/// Not a knob of its own - it is the diagnostic that fills a freshly reserved vertex buffer
/// with a quiet NaN so a lane the guest never wrote is distinguishable from one it wrote
/// zero into (`crate::host::poison_uniform_buffer`). An inlined call never reaches the host,
/// so the fill would simply stop happening and the instrument would report "the guest wrote
/// every lane" for every draw - an instrument whose failure imitates its subject
/// [[vitaslop-instrument-failure-imitating-its-subject]]. So the form is withheld while the
/// poison is on rather than emitting an approximation of it.
fn uniform_poison() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var_os("VITASLOP_GXM_UNIFORM_POISON").is_some())
}

/// `VITASLOP_NO_INLINE_UNIFORM_DATA`: route `sceGxmSetUniformDataF` through the host,
/// leaving every other inline form on.
///
/// The scoped A/B switch for the largest single host call a real title has left once the
/// draw state, the texture binds and the default-uniform reserves are inlined - **1,106
/// calls a frame on a retail race, 58% of what it still makes.** It is also the
/// falsifier to reach for first if a title's uniforms ever look wrong after this change,
/// which matters more here than for the other forms: this one writes the bytes a shader
/// reads, so a fault in it is a wrong PICTURE rather than a missing one.
///
/// Read through [`crate::knobs`] and listed in `OVERRIDABLE`, so a live page on a PHONE can
/// throw it between two runs of one build. Read at LINK time, so it must be set for the
/// whole run.
fn no_inline_uniform_data() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_NO_INLINE_UNIFORM_DATA"))
}

/// Whether `VITASLOP_UNIFORM_WATCH` names anything, which WITHHOLDS the inline form.
///
/// The watch names the guest code that wrote a uniform, and it lives inside the handler
/// (`vita::gxm::report_uniform_write`). An inlined call never reaches the host, so with the
/// form on, the one instrument for "who wrote this uniform" would report NOTHING and read as
/// "nobody writes it" - the failure mode
/// [[vitaslop-instrument-failure-imitating-its-subject]] is about. Same treatment
/// [`uniform_poison`] gets, and for the same reason.
fn uniform_watch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_UNIFORM_WATCH").is_ok_and(|s| !s.trim().is_empty()))
}

/// `VITASLOP_NO_INLINE_LWMUTEX`: route the lightweight-mutex lock and unlock through the
/// host, leaving every other inline form on.
///
/// A SCOPED A/B switch, and it exists because the whole-mechanism one
/// ([`no_inline_imports`]) cannot answer what one family is worth. Turning everything off
/// changes ~11,000 host calls a frame and moves every preemption point with them; against
/// that baseline a family worth ~1,000 calls is inside the noise, which is exactly the
/// position the draw-state work was left in on the browser.
///
/// It also has to be reachable from a PHONE, which is the only machine whose answer counts
/// (a desktop crossing is cheap enough that a count-based win barely registers there). So
/// it is read through [`crate::knobs`] and listed in `OVERRIDABLE`, and a live page can
/// throw it between two runs of the same build.
///
/// Read at LINK time, so it must be set for the whole run.
fn no_inline_lwmutex() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_NO_INLINE_LWMUTEX"))
}

/// `VITASLOP_NO_INLINE_TEXTURE`: route `sceGxmSetFragmentTexture` through the host,
/// leaving every other inline form on.
///
/// A SCOPED A/B switch like [`no_inline_lwmutex`], but it exists for a different reason, and
/// the difference is the point: this one is a CORRECTNESS falsifier, not a price tag. The
/// inline copy form took over a call every fragment texture bind in the title goes through
/// (1,275 a frame in a race), so it is the one change that can make a texture DISAPPEAR - and
/// the report that some text went missing came from a phone, against a build whose desktop
/// "verification" compared two different screens. Without a knob, answering it means
/// rebuilding an old revision and getting the user to run it; with one, it is a line typed
/// into the live page's knobs box and one run.
///
/// It only answers anything because both paths write the SAME bytes: the handler ends in
/// `gxmctx::set_texture_binding` and the emitted code writes that slot directly, held together
/// by `the_texture_binding_layout_is_closed`. So a picture that CHANGES when this is set is a
/// real divergence between the two writers, and one that does not clears the inline form.
///
/// Read at LINK time, so it must be set for the whole run.
fn no_inline_texture() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_NO_INLINE_TEXTURE"))
}

/// Which of the two writers of the context's texture slots reach the host this run, as
/// `(direct binds, precomputed-state binds)`.
///
/// Read by the empty-bindings report, which counts both and must say which half of its own
/// tally is complete: a report that presents a partial count as a complete one is worse than no
/// count [[vitaslop-inlined-host-calls-escape-both-watchpoints]].
///
/// **The two knobs are NOT interchangeable, and reading them as if they were cost a run.**
/// `VITASLOP_NO_INLINE_TEXTURE` routes only `sceGxmSetFragmentTexture`; the
/// `BindPrecomputedState` form stays inlined under it, so a run with that knob set reports
/// `0 precomputed-state binds` on a title making thousands of them. Only
/// `VITASLOP_NO_INLINE_IMPORTS` makes both halves complete.
pub(crate) fn texture_slot_writers_are_host_routed() -> (bool, bool) {
    let all = no_inline_imports();
    (all || no_inline_texture(), all)
}

/// `VITASLOP_NO_INLINE_IMPORTS`: route every host call through the host, even the
/// ones the transpiler could emit inline.
///
/// This is the A/B switch for the inline mechanism, and it earns its keep because
/// inlining changes how much wasm the guest executes, which changes fuel consumption,
/// which changes WHERE the preemptive scheduler switches threads - so an inlined build
/// legitimately reports a different determinism signature without computing anything
/// differently. Turning inlining off is how a signature is compared against a
/// pre-inlining run, which is the only way to tell a real behaviour change from that
/// re-interleaving. Read at LINK time, so it must be set for the whole run.
///
/// Read through [`crate::knobs`], not `std::env`. The browser has no environment
/// [[vitaslop-browser-has-no-env]], so an `env` read here made the A/B switch for the whole
/// inline mechanism unreachable on the ONE engine whose host-call cost is the reason the
/// mechanism exists: a phone pays ~36% of its host-call time in the CROSSING, against a
/// desktop where inlining ~1,089 calls a frame is worth 3.5%. Measuring what inlining buys
/// there could only ever be extrapolated from here. Exactly the defect the call-site
/// profiler below had, and for the same reason.
///
/// (Do not name another knob with its literal `VITASLOP_*` token in a doc comment here: the
/// `KNOBS.md` indexer treats any such token as a READ SITE, and the cross-reference then
/// overwrites that knob's real entry with this one's summary.)
fn no_inline_imports() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_NO_INLINE_IMPORTS"))
}

/// Diagnostic call-site profiler (`VITASLOP_DBG_CALLSITES`): counts host calls
/// keyed by (function NID, guest return address). A busy-wait spin shows up as one
/// (nid, lr) pair with an enormous count - the exact instruction to investigate.
///
/// Read through [`crate::knobs`], not `std::env`. The browser has no environment
/// [[vitaslop-browser-has-no-env]], so an `env` read here made this profiler
/// unreachable on the one engine whose host-call cost is in question: a phone spends
/// ~16 ms of a 56 ms guest frame on ~4,950 host calls, and "which NIDs are those"
/// could only be asked of the desktop.
/// Also SAMPLABLE at runtime, for the same reason the browser's host-call timer is: the guest
/// stack scan it does per call costs roughly as much as the call, so pinning it on for a whole
/// run means the only way to learn which NIDs are hot is to run at half speed for as long as you
/// want to watch. A caller that samples it for a window at a time gets the same histogram at a
/// fraction of the cost - the counts are cumulative, so a sampled window is a fair sample of
/// the calls made during it.
static DBG_CALLSITES: LazyLock<bool> =
    LazyLock::new(|| crate::knobs::flag("VITASLOP_DBG_CALLSITES"));

/// Sampling override for [`DBG_CALLSITES`], set by a frontend that profiles in bursts.
static DBG_CALLSITES_SAMPLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Turn the call-site profiler on or off for the next stretch of the run.
pub fn set_callsite_profiling(on: bool) {
    DBG_CALLSITES_SAMPLED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the profiler is recording right now, for an observer OUTSIDE the run - a watchdog
/// that reports nothing has to be able to say whether that is "nothing happened" or "nothing
/// was being recorded", and those are opposite readings.
pub fn callsite_profiling_on() -> bool {
    callsite_profiling()
}

/// Whether the profiler is recording right now - the knob forces it on permanently, and a
/// frontend can sample it on top of that.
#[inline]
fn callsite_profiling() -> bool {
    *DBG_CALLSITES || DBG_CALLSITES_SAMPLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The guest code range scanned for the game-level caller in [`dispatch`] (env
/// `VITASLOP_CODE_RANGE=lo-hi`, hex). A title's executable module is not always in
/// the same place, so the profiler's "skip the libc/lock veneer, find the first
/// return address in the game's own code" heuristic needs the right window. Default
/// covers a typical executable's .text just above IMAGE_BASE.
static CALLSITE_CODE_RANGE: LazyLock<(u32, u32)> = LazyLock::new(|| {
    std::env::var("VITASLOP_CODE_RANGE")
        .ok()
        .and_then(|s| {
            let (lo, hi) = s.split_once('-')?;
            let p = |x: &str| u32::from_str_radix(x.trim().trim_start_matches("0x"), 16).ok();
            Some((p(lo)?, p(hi)?))
        })
        .unwrap_or((0x8110_0000, 0x8130_0000))
});
static CALLSITE_HIST: Mutex<BTreeMap<(u32, u32), u64>> = Mutex::new(BTreeMap::new());

/// Print the guest call chain the first time a chosen NID is called from each thread
/// (env `VITASLOP_BACKTRACE=<func-nid-hex>[@LO-HI]`, the frame window optional). The
/// scan range is [`CALLSITE_CODE_RANGE`], so `VITASLOP_CODE_RANGE` widens it to a
/// title whose own code sits outside the default window.
static BACKTRACE_AT: LazyLock<Option<(u32, u64, u64)>> = LazyLock::new(|| {
    let s = std::env::var("VITASLOP_BACKTRACE").ok()?;
    let (nid_s, win) = match s.split_once('@') {
        Some((n, w)) => (n, parse_frame_window(w)),
        None => (s.as_str(), (0, u64::MAX)),
    };
    let nid_ = u32::from_str_radix(nid_s.trim().trim_start_matches("0x"), 16).ok()?;
    Some((nid_, win.0, win.1))
});
/// (nid, thread) pairs already reported, so the backtrace prints once per thread
/// instead of every frame.
static BACKTRACE_DONE: Mutex<std::collections::BTreeSet<(u32, i32)>> =
    Mutex::new(std::collections::BTreeSet::new());

/// Ordered-timeline trace (env `VITASLOP_TRACE_ORDER`): print every *meaningful*
/// host call live, in global order, with a monotonic index and thread id. Unlike
/// the counting profiler this shows the boot NARRATIVE and the exact point it
/// flatlines into a pure lock/poll spin. The high-frequency lock/unlock, shader-
/// reflection and per-draw GXM state calls are filtered so the interesting sequence
/// is not drowned out (a single front-end frame issues thousands of `sceGxmSet*`).
///
/// The value is a DISPLAY-FRAME WINDOW, because the interesting moment is usually
/// thousands of frames into a boot and tracing from frame 0 costs hundreds of
/// megabytes and minutes of wall time to reach it: `1`/`all` traces everything,
/// `LO-HI` traces an inclusive range, `LO-` traces from `LO` to the end. Anything
/// outside the window costs one integer compare.
static TRACE_ORDER: LazyLock<Option<(u64, u64)>> =
    LazyLock::new(|| std::env::var("VITASLOP_TRACE_ORDER").ok().map(|s| parse_frame_window(&s)));
static TRACE_ORDER_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `VITASLOP_HOSTCALL_WATCH=<hex addr>[,<hex addr>...]` - print every host call that passes one
/// of these guest addresses in any of its first four arguments.
///
/// This is the "what was ever done to this object" watch, and it is the one of the three that
/// can see an ABSENCE. A `SceGxmTexture` that reads as sixteen zero bytes at bind time was
/// either never initialised or initialised somewhere else, and no watchpoint on WRITES can tell
/// those apart - neither fires. A complete list of the calls that did name the struct settles
/// it: if `sceGxmTextureInitLinear` is not in it, the guest never called it.
static HOSTCALL_WATCH: LazyLock<Option<std::collections::HashSet<u32>>> = LazyLock::new(|| {
    let spec = std::env::var("VITASLOP_HOSTCALL_WATCH").ok()?;
    let set: std::collections::HashSet<u32> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            u32::from_str_radix(s.trim_start_matches("0x"), 16)
                .unwrap_or_else(|e| panic!("VITASLOP_HOSTCALL_WATCH: {s:?} is not a hex address: {e}"))
        })
        .collect();
    (!set.is_empty()).then_some(set)
});

/// Parse a diagnostic's display-frame window: `LO-HI` inclusive, `LO-` open-ended,
/// and anything else (including `1` and `all`) the whole run. A malformed bound is
/// the whole run rather than silently a different window.
fn parse_frame_window(spec: &str) -> (u64, u64) {
    let spec = spec.trim();
    match spec.split_once('-') {
        Some((lo, hi)) => {
            let lo = lo.trim().parse::<u64>().unwrap_or(0);
            let hi = if hi.trim().is_empty() { u64::MAX } else { hi.trim().parse::<u64>().unwrap_or(u64::MAX) };
            (lo, hi)
        }
        None => (0, u64::MAX),
    }
}

/// Describe a host call exactly as the guest made it: `r0`-`r3`, `lr`/`sp`, the first
/// few stack words (arguments five and up, under AAPCS), and a short dump of whatever
/// each pointer-looking argument points at.
///
/// This is the RE tool for a library with NO PUBLISHED PROTOTYPE. The NID database gives
/// a name and nothing else, and the title's own call is then the only evidence for the
/// signature: which arguments are pointers, how big a work area is, what a descriptor's
/// leading fields hold. Printed by the unimplemented-NID hard-fail, which is the moment
/// that evidence is on the wire.
///
/// A "pointer" here is any value inside the guest address space; that over-selects (a
/// large integer looks like one), so the dump is EVIDENCE, never a decoded signature.
fn describe_call_args(ctx: &mut crate::host::GuestCtx) -> String {
    use core::fmt::Write;
    let base = ctx.base;
    let lo = base;
    let hi = base.saturating_add(0x2000_0000);
    let sp = ctx.regs[13];
    let mut s = String::new();
    let _ = writeln!(
        s,
        "  call: r0={:#010x} r1={:#010x} r2={:#010x} r3={:#010x} lr={:#010x} sp={sp:#010x}",
        ctx.regs[0], ctx.regs[1], ctx.regs[2], ctx.regs[3], ctx.regs[14],
    );
    let stack: Vec<String> =
        (0..4).map(|i| format!("{:#010x}", ctx.read_u32(sp.wrapping_add(i * 4)))).collect();
    let _ = writeln!(s, "  stack args (sp+0..12): [{}]", stack.join(" "));
    for (i, &v) in [ctx.regs[0], ctx.regs[1], ctx.regs[2], ctx.regs[3]].iter().enumerate() {
        if !(lo..hi).contains(&v) {
            continue;
        }
        let words: Vec<String> =
            (0..8).map(|k| format!("{:#010x}", ctx.read_u32(v.wrapping_add(k * 4)))).collect();
        let _ = writeln!(s, "  *r{i} ({v:#010x}): [{}]", words.join(" "));
    }
    s
}

/// The plausible RETURN ADDRESSES on the calling thread's stack, innermost first.
///
/// A host call has no guest backtrace: the guest's frames are wasm frames the host cannot
/// walk, and `lr` names only the immediate caller. What the stack DOES hold is every
/// frame's saved `lr`, and a Thumb return address is recognisable - it is in the loaded
/// image and odd. That is not a real unwind (a saved data word can look like one), but it
/// is enough to name the code path that made a call, which is the whole question when a
/// title asks for something absurd.
pub fn guest_return_trail(ctx: &crate::host::GuestCtx, words: u32) -> String {
    let sp = ctx.regs[13];
    let lo = ctx.base;
    let hi = ctx.base.saturating_add(0x0200_0000);
    let mut out = Vec::new();
    for i in 0..words {
        let v = ctx.read_u32(sp.wrapping_add(i * 4));
        if v & 1 == 1 && (lo..hi).contains(&v) {
            out.push(format!("{:#010x}", v & !1));
        }
    }
    out.join(" <- ")
}

/// Print the hottest call sites (by count) gathered when `VITASLOP_DBG_CALLSITES` is
/// set. Call from a probe after the run to localize a spin.
pub fn dump_call_sites(top: usize) {
    eprintln!("{}", call_sites_report(top));
}

/// Forget every call site recorded so far, so the counts that follow describe a WINDOW
/// rather than the run.
///
/// A boot-inclusive tally and a steady-gameplay tally measure different programs, and the
/// difference is not small: on one title `sceKernelDelayThread` reads as the #1 call from
/// boot, and ~46,000 of that total is a single LOADING-phase call site that appears in no
/// steady window at all. Ranking work to remove off a cumulative tally has mis-ranked this
/// project's host-call list three times. Differencing two samples answers the same
/// question, but only if somebody remembers to take two - this makes the windowed reading
/// the cheap one.
pub fn reset_call_sites() {
    CALLSITE_HIST.lock().unwrap().clear();
}

/// The call-site tally as it stands right now, for DIFFERENCING two samples taken while the
/// run is still going.
///
/// [`reset_call_sites`] gives a window to anything that can call it at both ends. A run that
/// HANGS cannot: the frame that would close the window never arrives, so the only reading
/// available is the cumulative one, and a cumulative tally on a hung run is mostly the boot
/// that got there. Two snapshots seconds apart are a window that needs no cooperation from
/// the guest at all, which is what a watchdog observing from outside has to work with.
pub fn call_sites_snapshot() -> BTreeMap<(u32, u32), u64> {
    CALLSITE_HIST.lock().unwrap().clone()
}

/// The calls made SINCE `before`, hottest first - the same shape as [`call_sites_report`] but
/// over a window rather than from boot.
///
/// Returns `None` when nothing at all was called in the window, which is a real reading and
/// not an empty report: it says the guest is burning CPU in code that makes no host calls, and
/// that is a different instrument's question ([`crate::vita::dump_call_sites`] cannot see it).
pub fn call_sites_delta_report(before: &BTreeMap<(u32, u32), u64>, top: usize) -> Option<String> {
    use std::fmt::Write;
    let after = call_sites_snapshot();
    let mut delta: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    for (k, n) in after.iter() {
        let d = n.saturating_sub(*before.get(k).unwrap_or(&0));
        if d > 0 {
            delta.insert(*k, d);
        }
    }
    if delta.is_empty() {
        return None;
    }
    let mut per_nid: BTreeMap<u32, u64> = BTreeMap::new();
    for ((nid_, _), n) in delta.iter() {
        *per_nid.entry(*nid_).or_default() += n;
    }
    let total: u64 = delta.values().sum();
    let mut byn: Vec<_> = per_nid.iter().collect();
    byn.sort_by(|a, b| b.1.cmp(a.1));
    let mut v: Vec<_> = delta.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    let mut s = format!("--- calls IN THE WINDOW by NID: count ({total} total) ---
");
    for (nid_, n) in byn.into_iter().take(top) {
        let _ = writeln!(s, "  {n:>12}  {}", nid::name(*nid_));
    }
    let _ = writeln!(s, "--- hottest call sites IN THE WINDOW (nid, caller lr): count ---");
    for ((nid_, lr), n) in v.into_iter().take(top) {
        let _ = writeln!(s, "  {n:>12}  {} @ lr={lr:#010x}", nid::name(*nid_));
    }
    Some(s)
}

/// The hottest call sites as text, so a live session can print them too rather than only
/// an after-the-run probe. Counts are CUMULATIVE from boot (or from the last
/// [`reset_call_sites`]), which is what makes them useful: sample twice around a known
/// number of display frames and the difference is calls-per-frame, which is how you tell a
/// title pacing itself off the display from one running its own loop at a rate we are not
/// presenting at.
pub fn call_sites_report(top: usize) -> String {
    use std::fmt::Write;
    let h = CALLSITE_HIST.lock().unwrap();
    if h.is_empty() {
        return "no call sites recorded - set VITASLOP_DBG_CALLSITES=1 on the run".into();
    }
    // Aggregate per NID as well as per call site: "how often does this title ask for the
    // time" is a question about the NID, not about which of its call sites asked.
    let mut per_nid: BTreeMap<u32, u64> = BTreeMap::new();
    for ((nid_, _), n) in h.iter() {
        *per_nid.entry(*nid_).or_default() += n;
    }
    let mut v: Vec<_> = h.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    let mut byn: Vec<_> = per_nid.iter().collect();
    byn.sort_by(|a, b| b.1.cmp(a.1));
    let mut s = String::from("--- calls by NID: count ---\n");
    for (nid_, n) in byn.into_iter().take(top) {
        let _ = writeln!(s, "  {n:>12}  {}", nid::name(*nid_));
    }
    let _ = writeln!(s, "--- hottest call sites (nid, caller lr): count ---");
    for ((nid_, lr), n) in v.into_iter().take(top) {
        let _ = writeln!(s, "  {n:>12}  {} @ lr={lr:#010x}", nid::name(*nid_));
    }
    s
}

/// Route a NID call straight to its handler. Function NIDs are globally unique, so
/// one match over every implemented NID is unambiguous; `library_nid` is only for
/// logging an unimplemented call. This is a single decision - the compiler lowers
/// the match to one binary-decision tree / jump table - rather than the old chain
/// that re-probed each module's own match in turn (up to ~13 calls deep on a cold
/// NID). At tens of millions of host calls per frame that flat routing is the hot
/// path, so it lives in one place; the handler bodies stay in their per-module
/// files. An unhandled NID is recorded and returns 0 so the run continues and the
/// gap shows up in the capture.
pub fn dispatch(
    library_nid: u32,
    func_nid: u32,
    ctx: &mut GuestCtx,
    st: &mut VitaState,
) -> SvcOutcome {
    // A handler that returns `()` leaves the guest running; wrap its call so the arm
    // yields `Continue`. Handlers that can suspend a thread (blocking waits, the
    // frame flip, process/thread exit) return the `SvcOutcome` directly instead.
    macro_rules! cont {
        ($call:expr) => {{
            $call;
            SvcOutcome::Continue
        }};
    }

    // Diagnostic (env-gated): tally host calls by (nid, game-level caller), so a hot
    // busy-wait spin's exact site is visible without printing millions of lines. The
    // immediate LR is usually a thin libc lock wrapper, so scan the guest stack for
    // the first return address in the main module's code range - the game loop that
    // is actually spinning. Dumped by [`dump_call_sites`]. Zero cost when unset.
    if callsite_profiling() {
        let mut caller = ctx.regs[14];
        let sp = ctx.regs[13];
        let (lo, hi) = *CALLSITE_CODE_RANGE;
        for i in 0..40u32 {
            let v = ctx.read_u32(sp.wrapping_add(i * 4));
            if (lo..hi).contains(&v) {
                caller = v;
                break;
            }
        }
        *CALLSITE_HIST.lock().unwrap().entry((func_nid, caller)).or_insert(0u64) += 1;
    }

    // Diagnostic (env `VITASLOP_BACKTRACE`): the guest CALL CHAIN at a chosen host
    // call, once. "Which NID is being called" answers what a thread is doing; only the
    // chain answers WHERE in the game that thread is, and a stalled title's stuck
    // state machine is usually several frames up from the innermost call it makes.
    // The chain is a stack SCAN (every word in the code range, innermost first), not a
    // frame-pointer walk - ARM leaf frames often keep no frame pointer at all - so it
    // may contain stale slots; it is a set of candidates ordered by depth, not proof.
    if let Some((want_nid, lo_f, hi_f)) = *BACKTRACE_AT {
        if func_nid == want_nid && (lo_f..=hi_f).contains(&st.cur_frame()) {
            let key = (want_nid, st.current_thread());
            if BACKTRACE_DONE.lock().unwrap().insert(key) {
                let (lo, hi) = *CALLSITE_CODE_RANGE;
                let sp = ctx.regs[13];
                let mut chain = vec![format!("{:#010x}", ctx.regs[14])];
                for i in 0..256u32 {
                    let v = ctx.read_u32(sp.wrapping_add(i * 4));
                    if (lo..hi).contains(&v) {
                        chain.push(format!("{v:#010x}"));
                    }
                }
                eprintln!(
                    "backtrace f{} t{} {}: lr+stack candidates [{}]",
                    st.cur_frame(),
                    st.current_thread(),
                    nid::name(func_nid),
                    chain.join(" ")
                );
            }
        }
    }

    // Diagnostic (env `VITASLOP_HOSTCALL_WATCH`): every host call that names a watched guest
    // ADDRESS in any of its first four arguments, with the call, its arguments and the site.
    //
    // The question this answers is "what did the guest ever DO to this object", and neither of
    // the two watchpoints can answer it. `VITASLOP_WATCH_STORE` sees guest stores, so it is
    // blind to a host call writing the struct on the guest's behalf; `VITASLOP_HOST_WRITE_WATCH`
    // sees host writes, so it is blind to a call that should have written and did not - which is
    // exactly the interesting case for a struct that is all zeros. Watching the ARGUMENT catches
    // the call either way, including the one that was never made looking absent from a complete
    // list of the ones that were.
    if let Some(watch) = HOSTCALL_WATCH.as_ref() {
        if let Some(hit) = (0..4).map(|i| ctx.arg(i)).find(|a| watch.contains(a)) {
            eprintln!(
                "hostcall watch {hit:#x}: f{} t{} {}({:#x}, {:#x}, {:#x}, {:#x}) lr={:#010x}",
                st.cur_frame(),
                st.current_thread(),
                nid::name(func_nid),
                ctx.arg(0),
                ctx.arg(1),
                ctx.arg(2),
                ctx.arg(3),
                ctx.regs[14],
            );
        }
    }

    // Diagnostic (env `VITASLOP_TRACE_ORDER`): live, globally-ordered timeline of
    // meaningful calls. Filters the lock/unlock and shader-reflection storm so the
    // boot sequence and its flatline-into-spin are legible. Zero cost when unset.
    if TRACE_ORDER.is_some_and(|(lo, hi)| (lo..=hi).contains(&st.cur_frame())) {
        let nm = nid::name(func_nid);
        let noise = nm.contains("LwMutex")
            || nm.contains("LockMutex")
            || nm.contains("UnlockMutex")
            || nm.starts_with("sceGxmProgram")
            || nm.starts_with("sceGxmSet")
            || nm == "sceGxmDraw"
            || nm == "sceKernelGetTLSAddr";
        if !noise {
            let seq = TRACE_ORDER_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let label = if nm == "<unknown>" {
                format!("nid:{func_nid:#010x}")
            } else {
                nm.to_string()
            };
            eprintln!(
                "[ord {seq:>7} f{:<6} t{:<3}] {label}({:#x}, {:#x}, {:#x}, {:#x}) lr={:#010x}",
                st.cur_frame(),
                st.current_thread(),
                ctx.arg(0),
                ctx.arg(1),
                ctx.arg(2),
                ctx.arg(3),
                ctx.regs[14],
            );
        }
    }

    // Diagnostic (`RUST_LOG=vitaslop::ngs=trace`): log every NGS and sceAudioOut
    // call with its first four args and caller, to see exactly how a title feeds AT9
    // data to a voice and where the final mix goes.
    if library_nid == nid::lib::SCE_NGS || library_nid == nid::lib::SCE_AUDIO {
        tracing::trace!(
            target: "vitaslop::ngs",
            name = nid::name(func_nid),
            a0 = format_args!("{:#010x}", ctx.arg(0)),
            a1 = format_args!("{:#010x}", ctx.arg(1)),
            a2 = format_args!("{:#010x}", ctx.arg(2)),
            a3 = format_args!("{:#010x}", ctx.arg(3)),
            lr = format_args!("{:#010x}", ctx.regs[14]),
            "call"
        );
    }
    let outcome = match func_nid {
        // --- lwsync: lightweight mutex / cond (the hottest surface) --------------
        lw_nid::CREATE_LW_MUTEX => cont!(lwsync::create_lw_mutex(ctx, st)),
        lw_nid::CREATE_LW_COND => cont!(lwsync::create_lw_cond(ctx, st)),
        lw_nid::WAIT_LW_COND | lw_nid::WAIT_LW_COND_CB => lwsync::wait_lw_cond(ctx, st),
        lw_nid::SIGNAL_LW_COND => cont!(lwsync::signal_lw_cond(ctx, st, false)),
        // SignalLwCondAll wakes every waiter; SignalLwCondTo targets one thread,
        // approximated by a broadcast (a spurious wake re-checks and re-waits).
        lw_nid::SIGNAL_LW_COND_ALL | lw_nid::SIGNAL_LW_COND_TO => {
            cont!(lwsync::signal_lw_cond(ctx, st, true))
        }
        // A lightweight mutex genuinely blocks on contention and enforces mutual
        // exclusion (keyed by its guest work-area address). The `_CB` lock variant
        // additionally processes pending callbacks - none are queued in this model, so
        // it takes the same path.
        lw_nid::LOCK_LW_MUTEX | lw_nid::LOCK_LW_MUTEX_CB => lwsync::lock_lw_mutex(ctx, st, false),
        lw_nid::TRY_LOCK_LW_MUTEX => lwsync::lock_lw_mutex(ctx, st, true),
        lw_nid::UNLOCK_LW_MUTEX | lw_nid::UNLOCK_LW_MUTEX2 => {
            cont!(lwsync::unlock_lw_mutex(ctx, st))
        }
        lw_nid::DELETE_LW_MUTEX => cont!(lwsync::delete_lw_mutex(ctx, st)),
        // A lightweight cond has no persistent host record beyond its parked waiters
        // (keyed by work address in `wait_lw_cond`/`signal_lw_cond`), so delete is a
        // bare success.
        lw_nid::DELETE_LW_COND => cont!(lwsync::succeed(ctx)),

        // --- sync: heavyweight mutex / sema / cond / event flag -----------------
        sync_nid::CREATE_MUTEX => cont!(sync::create_mutex(ctx, st)),
        // Lock and wait can block under the preemptive scheduler (Block parks).
        // The `CB` spelling shares the handler, for the reason given at the display
        // waits: this engine delivers callbacks at host-call boundaries, so every wait is
        // already a delivery point and the distinction hardware draws does not exist here.
        sync_nid::LOCK_MUTEX | sync_nid::LOCK_MUTEX_CB => sync::lock_mutex(ctx, st, false),
        sync_nid::TRY_LOCK_MUTEX => sync::lock_mutex(ctx, st, true),
        sync_nid::UNLOCK_MUTEX => cont!(sync::unlock_mutex(ctx, st)),
        sync_nid::DELETE_MUTEX => cont!(sync::delete_object(ctx, st)),
        sync_nid::CLOSE_MUTEX => cont!(sync::delete_object(ctx, st)),
        sync_nid::CREATE_SEMA | sync_nid::CREATE_SEMA_16XX => cont!(sync::create_sema(ctx, st)),
        // `sceKernelWaitSemaCB` is the same wait, at which the kernel also delivers
        // the calling thread's pending async callbacks. Callback delivery here is
        // driven by the scheduler (the display queue runs its callbacks on its own
        // serialized path), so there is no separate pump to run at this point - but
        // the WAIT itself is the load-bearing behaviour and must be identical, which
        // is why it shares the handler rather than getting a weaker one.
        sync_nid::WAIT_SEMA | lk_nid::WAIT_SEMA_CB => sync::wait_sema(ctx, st),
        sync_nid::SIGNAL_SEMA => cont!(sync::signal_sema(ctx, st)),
        sync_nid::DELETE_SEMA => cont!(sync::delete_object(ctx, st)),
        sync_nid::CREATE_COND => cont!(sync::create_cond(ctx, st)),
        sync_nid::WAIT_COND => sync::wait_cond(ctx, st),
        sync_nid::SIGNAL_COND => cont!(sync::signal_cond(ctx, st, false)),
        sync_nid::SIGNAL_COND_ALL => cont!(sync::signal_cond(ctx, st, true)),
        sync_nid::DELETE_COND => cont!(sync::delete_object(ctx, st)),
        sync_nid::CREATE_EVENT_FLAG => cont!(sync::create_event_flag(ctx, st)),
        sync_nid::SET_EVENT_FLAG => cont!(sync::set_event_flag(ctx, st)),
        // A real wait: parks under the preemptive scheduler until SetEventFlag
        // satisfies the pattern (or the timeout passes).
        sync_nid::WAIT_EVENT_FLAG | sync_nid::WAIT_EVENT_FLAG_CB => sync::wait_event_flag(ctx, st),
        sync_nid::POLL_EVENT_FLAG => cont!(sync::poll_event_flag(ctx, st)),
        sync_nid::CLEAR_EVENT_FLAG => cont!(sync::clear_event_flag(ctx, st)),
        sync_nid::DELETE_EVENT_FLAG => cont!(sync::delete_object(ctx, st)),
        sync_nid::GET_SYSTEM_TIME_WIDE => cont!(sync::get_system_time_wide(ctx, st)),

        // --- libkernel: clib string/mem, threads, process ----------------------
        lk_nid::CLIB_PRINTF => cont!(libkernel::clib_printf(ctx, st)),
        lk_nid::CLIB_SNPRINTF => cont!(libkernel::clib_snprintf(ctx, st)),
        // memmove shares memcpy's read-then-write impl (tolerates overlap).
        lk_nid::CLIB_MEMCPY | lk_nid::CLIB_MEMMOVE => cont!(libkernel::clib_memcpy(ctx, st)),
        lk_nid::CLIB_MEMSET => cont!(libkernel::clib_memset(ctx, st)),
        lk_nid::CLIB_MEMCMP => cont!(libkernel::clib_memcmp(ctx, st)),
        lk_nid::CLIB_STRNLEN => cont!(libkernel::clib_strnlen(ctx, st)),
        lk_nid::CLIB_STRNCPY => cont!(libkernel::clib_strncpy(ctx, st)),
        lk_nid::CLIB_STRNCMP => cont!(libkernel::clib_strncmp(ctx, st)),
        lk_nid::CLIB_STRCMP => cont!(libkernel::clib_strcmp(ctx, st)),
        lk_nid::CLIB_STRRCHR => cont!(libkernel::clib_strrchr(ctx, st)),
        lk_nid::CLIB_STRNCASECMP => cont!(libkernel::clib_strncasecmp(ctx, st)),
        lk_nid::CLIB_MSPACE_CREATE => cont!(libkernel::clib_mspace_create(ctx, st)),
        lk_nid::CLIB_MSPACE_DESTROY => cont!(libkernel::clib_mspace_destroy(ctx, st)),
        lk_nid::CLIB_MSPACE_MALLOC => cont!(libkernel::clib_mspace_malloc(ctx, st)),
        lk_nid::CLIB_MSPACE_MEMALIGN => cont!(libkernel::clib_mspace_memalign(ctx, st)),
        lk_nid::CLIB_MSPACE_FREE => cont!(libkernel::clib_mspace_free(ctx, st)),
        lk_nid::CREATE_THREAD => cont!(libkernel::create_thread(ctx, st)),
        lk_nid::START_THREAD => libkernel::start_thread(ctx, st),
        // Join can block under the preemptive scheduler.
        // ...CB is the same join with callback delivery; see `WAIT_SEMA_CB` above.
        lk_nid::WAIT_THREAD_END | lk_nid::WAIT_THREAD_END_CB => libkernel::wait_thread_end(ctx, st),
        lk_nid::GET_THREAD_ID => cont!(libkernel::get_thread_id(ctx, st)),
        lk_nid::GET_THREAD_EXIT_STATUS => cont!(libkernel::get_thread_exit_status(ctx, st)),
        lk_nid::GET_TLS_ADDR => cont!(libkernel::get_tls_addr(ctx, st)),
        lk_nid::CREATE_MSG_PIPE => cont!(libkernel::msg_pipe_create(ctx, st)),
        tm_nid::DELETE_MSG_PIPE => cont!(libkernel::msg_pipe_delete(ctx, st)),
        lk_nid::SEND_MSG_PIPE | lk_nid::TRY_SEND_MSG_PIPE => cont!(libkernel::msg_pipe_send(ctx, st)),
        lk_nid::RECEIVE_MSG_PIPE | lk_nid::TRY_RECEIVE_MSG_PIPE => cont!(libkernel::msg_pipe_receive(ctx, st)),
        lk_nid::GET_THREAD_TLS_ADDR => cont!(libkernel::get_thread_tls_addr(ctx, st)),
        lk_nid::GET_RANDOM_NUMBER => cont!(libkernel::get_random_number(ctx, st)),
        lk_nid::GET_PROCESS_TIME => libkernel::get_process_time(ctx, st),
        lk_nid::GET_PROCESS_TIME_WIDE => libkernel::get_process_time_wide(ctx, st),
        lk_nid::EXIT_PROCESS => {
            // r0 (exit code) is left as the guest set it; any exit is a clean stop.
            libkernel::trace_exit(ctx, st);
            SvcOutcome::Halt
        }

        // --- threadmgr: delay, exit, process id --------------------------------
        // A real timed sleep: parks under the preemptive scheduler (see the handler).
        // ...CB is the same sleep with callback delivery; see `WAIT_SEMA_CB` above.
        tm_nid::DELAY_THREAD | tm_nid::DELAY_THREAD_CB => threadmgr::delay_thread(ctx, st),
        // A thread ending itself: just this thread under the preemptive scheduler;
        // a whole-run stop in single-thread-of-control bring-up (only main reaches
        // here there - workers return normally instead).
        tm_nid::EXIT_THREAD | tm_nid::EXIT_DELETE_THREAD => {
            if st.is_preemptive() {
                SvcOutcome::ThreadExit
            } else {
                SvcOutcome::Halt
            }
        }
        tm_nid::DELETE_THREAD => cont!(threadmgr::delete_thread(ctx, st)),
        tm_nid::GET_PROCESS_ID => cont!(threadmgr::get_process_id(ctx, st)),
        tm_nid::GET_THREAD_CURRENT_PRIORITY => {
            cont!(threadmgr::get_thread_current_priority(ctx, st))
        }
        tm_nid::CHANGE_THREAD_CPU_AFFINITY_MASK => {
            cont!(threadmgr::change_thread_cpu_affinity_mask(ctx, st))
        }
        tm_nid::GET_THREAD_CPU_AFFINITY_MASK => {
            cont!(threadmgr::get_thread_cpu_affinity_mask(ctx, st))
        }
        // Closing a semaphore invalidates its id, same as deleting it in this model.
        tm_nid::CLOSE_SEMA => cont!(sync::delete_object(ctx, st)),
        tm_nid::CHANGE_THREAD_VFP_EXCEPTION => cont!(threadmgr::change_thread_vfp_exception(ctx, st)),

        // --- net: BSD sockets, modelled OFFLINE (see `vita::net`) ---------------
        // --- SceMotion: a device AT REST, flat. The two sampling switches are real
        // state because the getter reads them back; the poses are what "at rest" is.
        sv_nid::MOTION_GET_SENSOR_STATE => cont!(services::motion_get_sensor_state(ctx, st)),
        sv_nid::MOTION_GET_MAGNETOMETER_STATE => {
            cont!(services::motion_get_magnetometer_state(ctx, st))
        }
        sv_nid::MOTION_MAGNETOMETER_ON => cont!(services::motion_magnetometer_set(ctx, st, true)),
        sv_nid::MOTION_MAGNETOMETER_OFF => cont!(services::motion_magnetometer_set(ctx, st, false)),
        sv_nid::MOTION_RESET => cont!(services::motion_reset(ctx, st)),
        sv_nid::MOTION_SET_ANGLE_THRESHOLD => cont!(services::motion_set_angle_threshold(ctx, st)),
        sv_nid::MOTION_GET_ANGLE_THRESHOLD => cont!(services::motion_get_angle_threshold(ctx, st)),
        sv_nid::APP_MGR_GET_BUDGET_INFO => cont!(services::app_mgr_get_budget_info(ctx, st)),
        sv_nid::SHARED_FB_OPEN => cont!(services::shared_fb_open(ctx, st)),
        sv_nid::SHARED_FB_BEGIN | sv_nid::SHARED_FB_GET_INFO => cont!(services::shared_fb_info(ctx, st)),
        // The shared framebuffer's End IS the title's present: it is what
        // `sceGxmDisplayQueueAddEntry` is to a title that owns its buffers, so it
        // closes the frame the same way (the scheduler's frame boundary, the clock's
        // per-frame top-up, the recipe's frame count). Without it a SceSharedFb title
        // never flips: it runs unpaced, and a frame-bounded run never ends.
        sv_nid::SHARED_FB_END => {
            services::shared_fb_end(ctx, st);
            SvcOutcome::Flip
        }
        sv_nid::SHARED_FB_CLOSE => cont!(services::shared_fb_close(ctx, st)),
        sv_nid::POWER_SET_ARM_CLOCK => cont!(services::power_set_clock(ctx, st, 0)),
        sv_nid::POWER_SET_BUS_CLOCK => cont!(services::power_set_clock(ctx, st, 1)),
        sv_nid::POWER_SET_GPU_CLOCK => cont!(services::power_set_clock(ctx, st, 2)),
        sv_nid::POWER_SET_GPU_XBAR_CLOCK => cont!(services::power_set_clock(ctx, st, 3)),
        sv_nid::POWER_GET_ARM_CLOCK => cont!(services::power_get_clock(ctx, st, 0)),
        sv_nid::POWER_GET_BUS_CLOCK => cont!(services::power_get_clock(ctx, st, 1)),
        sv_nid::POWER_GET_GPU_CLOCK => cont!(services::power_get_clock(ctx, st, 2)),
        sv_nid::POWER_GET_GPU_XBAR_CLOCK => cont!(services::power_get_clock(ctx, st, 3)),

        // --- SceLibXml (DOM): a real parser over a host-side node arena. See
        // `sce_xml` for how the C++ object layouts were established and why the
        // node ids are global.
        xml_nid::MEM_ALLOCATOR_CTOR => cont!(sce_xml::mem_allocator_ctor(ctx, st)),
        xml_nid::MEM_ALLOCATOR_DTOR => cont!(sce_xml::mem_allocator_dtor(ctx, st)),
        xml_nid::INITIALIZER_CTOR => cont!(sce_xml::initializer_ctor(ctx, st)),
        xml_nid::INITIALIZER_DTOR => cont!(sce_xml::initializer_dtor(ctx, st)),
        xml_nid::INITIALIZER_INITIALIZE => cont!(sce_xml::initializer_initialize(ctx, st)),
        xml_nid::BUILDER_CTOR => cont!(sce_xml::builder_ctor(ctx, st)),
        xml_nid::BUILDER_DTOR => cont!(sce_xml::builder_dtor(ctx, st)),
        xml_nid::BUILDER_INITIALIZE => cont!(sce_xml::builder_initialize(ctx, st)),
        xml_nid::BUILDER_GET_DOCUMENT => cont!(sce_xml::builder_get_document(ctx, st)),
        // The three parser switches share one body; see there for what each would have
        // changed and why none of them changes anything here.
        xml_nid::BUILDER_SET_RESOLVE_ENTITY
        | xml_nid::BUILDER_SET_SKIP_IGNORABLE_TEXT
        | xml_nid::BUILDER_SET_SKIP_IGNORABLE_WHITESPACE => {
            cont!(sce_xml::builder_set_flag(ctx, st))
        }
        xml_nid::BUILDER_PARSE => cont!(sce_xml::builder_parse(ctx, st)),
        xml_nid::DOCUMENT_DTOR => cont!(sce_xml::document_dtor(ctx, st)),
        xml_nid::DOCUMENT_GET_ROOT => cont!(sce_xml::document_get_root(ctx, st)),
        xml_nid::DOCUMENT_GET_FIRST_CHILD => cont!(sce_xml::document_get_first_child(ctx, st)),
        xml_nid::DOCUMENT_GET_SIBLING => cont!(sce_xml::document_get_sibling(ctx, st)),
        xml_nid::DOCUMENT_GET_FIRST_ATTR => cont!(sce_xml::document_get_first_attr(ctx, st)),
        xml_nid::DOCUMENT_GET_NODE_NAME => cont!(sce_xml::document_get_node_name(ctx, st)),
        xml_nid::DOCUMENT_GET_NODE_TYPE => cont!(sce_xml::document_get_node_type(ctx, st)),
        xml_nid::DOCUMENT_GET_TEXT => cont!(sce_xml::document_get_text(ctx, st)),
        xml_nid::NODE_CTOR => cont!(sce_xml::node_ctor(ctx, st)),
        xml_nid::NODE_DTOR => cont!(sce_xml::node_dtor(ctx, st)),
        xml_nid::NODE_GET_NODE_NAME => cont!(sce_xml::node_get_node_name(ctx, st)),
        xml_nid::NODE_GET_NODE_VALUE => cont!(sce_xml::node_get_node_value(ctx, st)),
        xml_nid::STRING_CTOR => cont!(sce_xml::string_ctor(ctx, st)),

        // --- ScePgf: the PSP-compatible font API, on the same engine as ScePvf.
        pgf_nid::NEW_LIB => cont!(pgf::new_lib(ctx, st)),
        pgf_nid::DONE_LIB => cont!(pgf::done_lib(ctx, st)),
        pgf_nid::OPEN => cont!(pgf::open(ctx, st)),
        pgf_nid::OPEN_USER_MEMORY => cont!(pgf::open_user_memory(ctx, st)),
        pgf_nid::CLOSE => cont!(pgf::close(ctx, st)),
        pgf_nid::GET_CHAR_INFO => cont!(pgf::get_char_info(ctx, st)),
        pgf_nid::GET_FONT_INFO => cont!(pgf::get_font_info(ctx, st)),
        pgf_nid::GET_CHAR_GLYPH_IMAGE => cont!(pgf::get_char_glyph_image(ctx, st)),

        // --- SceAudioIn: the microphone, muted. `Input` PARKS for the grain it
        // reports, so a capture loop runs at the input's rate instead of spinning.
        audioin_nid::OPEN_PORT => cont!(audioin::in_open_port(ctx, st)),
        audioin_nid::RELEASE_PORT => cont!(audioin::in_release_port(ctx, st)),
        audioin_nid::INPUT => audioin::in_input(ctx, st),
        audioin_nid::GET_ADOPT => cont!(audioin::in_get_adopt(ctx, st)),
        audioin_nid::GET_STATUS => cont!(audioin::in_get_status(ctx, st)),

        // --- SceLiveAreaUtil: the gate the title's own package declares. -----------
        livearea_nid::GET_FRAME_REVISION => cont!(livearea::get_frame_revision(ctx, st)),
        livearea_nid::GET_FRAME_USER_DATA => cont!(livearea::get_frame_user_data(ctx, st)),

        // --- SceHttp, offline: a real local object graph, and one call that reports
        // the link being down. See `http` for why the send is the one that fails.
        http_nid::CREATE_TEMPLATE => cont!(http::create_template(ctx, st)),
        http_nid::DELETE_TEMPLATE => cont!(http::delete_template(ctx, st)),
        http_nid::CREATE_CONNECTION_WITH_URL => cont!(http::create_connection_with_url(ctx, st)),
        http_nid::DELETE_CONNECTION => cont!(http::delete_connection(ctx, st)),
        http_nid::CREATE_REQUEST_WITH_URL => cont!(http::create_request_with_url(ctx, st)),
        http_nid::DELETE_REQUEST => cont!(http::delete_request(ctx, st)),
        // The timeout slot is named by the ARM, not read from the guest: the three NIDs
        // differ only in which of the three values they set.
        http_nid::SET_CONNECT_TIMEOUT => cont!(http::set_timeout(ctx, st, 0)),
        http_nid::SET_SEND_TIMEOUT => cont!(http::set_timeout(ctx, st, 1)),
        http_nid::SET_RECV_TIMEOUT => cont!(http::set_timeout(ctx, st, 2)),
        http_nid::ADD_REQUEST_HEADER => cont!(http::add_request_header(ctx, st)),
        http_nid::SEND_REQUEST => cont!(http::send_request(ctx, st)),
        http_nid::ABORT_REQUEST => cont!(http::abort_request(ctx, st)),
        http_nid::GET_STATUS_CODE => cont!(http::get_status_code(ctx, st)),
        http_nid::GET_RESPONSE_CONTENT_LENGTH => cont!(http::get_response_content_length(ctx, st)),
        http_nid::READ_DATA => cont!(http::read_data(ctx, st)),
        http_nid::SSL_LOAD_CERT => cont!(http::ssl_load_cert(ctx, st)),
        http_nid::SSL_SET_SSL_CALLBACK => cont!(http::ssl_set_ssl_callback(ctx, st)),
        http_nid::SSL_GET_SSL_ERROR => cont!(http::ssl_get_ssl_error(ctx, st)),

        net_nid::SOCKET => cont!(net::socket(ctx, st)),
        net_nid::SOCKET_CLOSE => cont!(net::socket_close(ctx, st)),
        net_nid::BIND => cont!(net::bind(ctx, st)),
        net_nid::LISTEN => cont!(net::listen(ctx, st)),
        net_nid::ACCEPT => cont!(net::accept(ctx, st)),
        net_nid::CONNECT => cont!(net::connect(ctx, st)),
        net_nid::SEND | net_nid::SENDTO | net_nid::SENDMSG => cont!(net::send(ctx, st)),
        net_nid::RECV | net_nid::RECVFROM => cont!(net::recv(ctx, st)),
        net_nid::SHUTDOWN => cont!(net::shutdown(ctx, st)),
        net_nid::GETSOCKNAME => cont!(net::getsockname(ctx, st)),
        net_nid::GETPEERNAME => cont!(net::getpeername(ctx, st)),
        net_nid::SETSOCKOPT => cont!(net::setsockopt(ctx, st)),
        net_nid::GETSOCKOPT => cont!(net::getsockopt(ctx, st)),
        net_nid::GET_SOCK_INFO => cont!(net::get_sock_info(ctx, st)),
        net_nid::SHOW_NETSTAT => cont!(net::show_netstat(ctx, st)),
        net_nid::HTONL | net_nid::NTOHL => cont!(net::swap32(ctx, st)),
        net_nid::HTONS | net_nid::NTOHS => cont!(net::swap16(ctx, st)),
        net_nid::INET_PTON => cont!(net::inet_pton(ctx, st)),
        net_nid::INET_NTOP => cont!(net::inet_ntop(ctx, st)),
        net_nid::ERRNO_LOC => cont!(net::errno_loc(ctx, st)),
        net_nid::RESOLVER_CREATE => cont!(net::resolver_create(ctx, st)),
        net_nid::RESOLVER_DESTROY => cont!(net::resolver_destroy(ctx, st)),
        net_nid::RESOLVER_START_NTOA | net_nid::RESOLVER_START_ATON => {
            cont!(net::resolver_start(ctx, st))
        }
        net_nid::RESOLVER_GET_ERROR => cont!(net::resolver_get_error(ctx, st)),
        net_nid::EPOLL_CREATE => cont!(net::epoll_create(ctx, st)),
        net_nid::EPOLL_DESTROY => cont!(net::epoll_destroy(ctx, st)),
        net_nid::EPOLL_CONTROL => cont!(net::epoll_control(ctx, st)),
        net_nid::EPOLL_WAIT => cont!(net::epoll_wait(ctx, st)),

        // --- fiber: cooperative user-level threads -------------------------------
        // Run/Switch/ReturnToThread hand the baton over and PARK the caller, so these
        // return their own outcome rather than going through `cont!`.
        fiber_nid::INITIALIZE_IMPL => cont!(fiber::initialize(ctx, st)),
        fiber_nid::INITIALIZE_WITH_INTERNAL_OPTION_IMPL => {
            cont!(fiber::initialize_with_internal_option(ctx, st))
        }
        fiber_nid::RUN => fiber::run(ctx, st),
        fiber_nid::SWITCH => fiber::switch(ctx, st),
        fiber_nid::ATTACH_CONTEXT_AND_SWITCH => fiber::attach_context_and_switch(ctx, st),
        fiber_nid::RETURN_TO_THREAD => fiber::return_to_thread(ctx, st),
        fiber_nid::GET_SELF => cont!(fiber::get_self(ctx, st)),
        fiber_nid::FINALIZE => cont!(fiber::finalize(ctx, st)),
        fiber_nid::GET_INFO => cont!(fiber::get_info(ctx, st)),

        // --- gxm: graphics ------------------------------------------------------
        gxm_nid::INITIALIZE | gxm_nid::VSH_INITIALIZE => cont!(gxm::initialize(ctx, st)),
        gxm_nid::FINISH
        | gxm_nid::PAD_HEARTBEAT
        | gxm_nid::DISPLAY_QUEUE_FINISH
        | gxm_nid::PROGRAM_CHECK
        | gxm_nid::DESTROY_CONTEXT
        | gxm_nid::DESTROY_RENDER_TARGET
        | gxm_nid::SYNC_OBJECT_DESTROY => cont!(gxm::ok(ctx)),
        gxm_nid::MAP_MEMORY => cont!(gxm::map_memory(ctx, st)),
        gxm_nid::DEPTH_STENCIL_SURFACE_INIT => cont!(gxm::depth_stencil_surface_init(ctx, st)),
        // Nothing to tear down for these, but the guest is now free to reuse the
        // program's memory, so the reflected constants cached against its header
        // address must not outlive it.
        gxm_nid::SHADER_PATCHER_DESTROY
        | gxm_nid::SHADER_PATCHER_UNREGISTER_PROGRAM
        | gxm_nid::SHADER_PATCHER_RELEASE_VERTEX_PROGRAM
        | gxm_nid::SHADER_PATCHER_RELEASE_FRAGMENT_PROGRAM => {
            st.invalidate_program_reflection();
            cont!(gxm::ok(ctx))
        }
        // Record the bound fragment program so a draw can reflect its samplers (albedo
        // selection). The direct-draw path binds it here rather than via a precomputed
        // fragment state.
        gxm_nid::SET_FRAGMENT_PROGRAM => cont!(gxm::set_fragment_program(ctx, st)),
        gxm_nid::TERMINATE => {
            ctx.ret(0);
            if st.halt_on_terminate {
                SvcOutcome::Halt
            } else {
                SvcOutcome::Continue
            }
        }
        gxm_nid::MAP_VERTEX_USSE_MEMORY | gxm_nid::MAP_FRAGMENT_USSE_MEMORY => {
            cont!(gxm::map_usse(ctx, st))
        }
        // Split from `SHADER_PATCHER_CREATE`: a shader patcher really is an opaque handle,
        // but a CONTEXT is a guest structure the sticky draw state lives in - see
        // [`gxm::create_context`].
        gxm_nid::CREATE_CONTEXT => gxm::create_context(ctx, st),
        gxm_nid::SHADER_PATCHER_CREATE => cont!(gxm::out_handle(ctx, st, 1)),
        gxm_nid::CREATE_RENDER_TARGET => cont!(gxm::create_render_target(ctx, st)),
        gxm_nid::SYNC_OBJECT_CREATE => cont!(gxm::out_handle(ctx, st, 0)),
        gxm_nid::SHADER_PATCHER_REGISTER_PROGRAM => cont!(gxm::register_program(ctx, st)),
        gxm_nid::SHADER_PATCHER_GET_PROGRAM_FROM_ID => cont!(gxm::get_program_from_id(ctx, st)),
        gxm_nid::PROGRAM_PARAMETER_GET_RESOURCE_INDEX => cont!(gxm::param_get_resource_index(ctx)),
        gxm_nid::PROGRAM_FIND_PARAMETER_BY_NAME => cont!(gxm::find_parameter(ctx, st)),
        gxm_nid::PROGRAM_GET_PARAMETER_COUNT => cont!(gxm::program_get_parameter_count(ctx)),
        gxm_nid::PROGRAM_GET_PARAMETER => cont!(gxm::program_get_parameter(ctx)),
        gxm_nid::PROGRAM_PARAMETER_GET_CATEGORY => cont!(gxm::param_get_category(ctx)),
        gxm_nid::PROGRAM_PARAMETER_GET_TYPE => cont!(gxm::param_get_type(ctx)),
        gxm_nid::PROGRAM_PARAMETER_GET_COMPONENT_COUNT => cont!(gxm::param_get_component_count(ctx)),
        gxm_nid::PROGRAM_PARAMETER_GET_CONTAINER_INDEX => cont!(gxm::param_get_container_index(ctx)),
        gxm_nid::PROGRAM_PARAMETER_GET_ARRAY_SIZE => cont!(gxm::param_get_array_size(ctx)),
        gxm_nid::PROGRAM_PARAMETER_GET_NAME => cont!(gxm::param_get_name(ctx)),
        gxm_nid::COLOR_SURFACE_INIT => cont!(gxm::color_surface_init(ctx, st)),
        gxm_nid::COLOR_SURFACE_INIT_DISABLED => cont!(gxm::color_surface_init_disabled(ctx, st)),
        gxm_nid::SHADER_PATCHER_CREATE_VERTEX_PROGRAM => cont!(gxm::create_vertex_program(ctx, st)),
        gxm_nid::SHADER_PATCHER_CREATE_FRAGMENT_PROGRAM => cont!(gxm::create_fragment_program(ctx, st)),
        gxm_nid::BEGIN_SCENE => cont!(gxm::begin_scene(ctx, st)),
        gxm_nid::END_SCENE => cont!(gxm::end_scene(ctx, st)),
        gxm_nid::SET_VERTEX_PROGRAM => cont!(gxm::set_vertex_program(ctx, st)),
        gxm_nid::RESERVE_VERTEX_DEFAULT_UNIFORM_BUFFER => cont!(gxm::reserve_vertex_uniforms(ctx, st)),
        gxm_nid::RESERVE_FRAGMENT_DEFAULT_UNIFORM_BUFFER => cont!(gxm::reserve_fragment_uniforms(ctx, st)),
        gxm_nid::SET_UNIFORM_DATA_F => cont!(gxm::set_uniform_data_f(ctx, st)),
        gxm_nid::SET_VERTEX_STREAM => cont!(gxm::set_vertex_stream(ctx, st)),
        gxm_nid::SET_FRAGMENT_TEXTURE => cont!(gxm::set_fragment_texture(ctx, st)),
        gxm_nid::TEXTURE_INIT_LINEAR => cont!(gxm::texture_init(ctx, st, gxm::TYPE_LINEAR)),
        gxm_nid::TEXTURE_INIT_LINEAR_STRIDED => {
            cont!(gxm::texture_init(ctx, st, gxm::TYPE_LINEAR_STRIDED))
        }
        gxm_nid::TEXTURE_INIT_SWIZZLED => cont!(gxm::texture_init(ctx, st, gxm::TYPE_SWIZZLED)),
        gxm_nid::TEXTURE_INIT_SWIZZLED_ARBITRARY => {
            cont!(gxm::texture_init(ctx, st, gxm::TYPE_SWIZZLED_ARBITRARY))
        }
        gxm_nid::TEXTURE_INIT_TILED => cont!(gxm::texture_init(ctx, st, gxm::TYPE_TILED)),
        gxm_nid::TEXTURE_SET_DATA => cont!(gxm::texture_set_data(ctx, st)),
        gxm_nid::TEXTURE_SET_FORMAT => cont!(gxm::texture_set_format(ctx, st)),
        gxm_nid::TEXTURE_GET_DATA => cont!(gxm::texture_get_data(ctx)),
        gxm_nid::TEXTURE_GET_WIDTH => cont!(gxm::texture_get_dim(ctx, 12)),
        gxm_nid::TEXTURE_GET_HEIGHT => cont!(gxm::texture_get_dim(ctx, 0)),
        gxm_nid::TEXTURE_GET_FORMAT => cont!(gxm::texture_get_format(ctx, st)),
        // Texture filters: write the min/mag/mip filter into the guest's own control word 0,
        // where the hardware keeps it - so a getter reads it back, a by-value copy of the struct
        // carries it, and the getter can be INLINED into guest code (see `gxm::texword0`).
        gxm_nid::TEXTURE_SET_MIN_FILTER => cont!(gxm::texture_set_min_filter(ctx)),
        gxm_nid::TEXTURE_SET_MAG_FILTER => cont!(gxm::texture_set_mag_filter(ctx)),
        gxm_nid::TEXTURE_SET_MIP_FILTER => cont!(gxm::texture_set_mip_filter(ctx)),
        gxm_nid::TEXTURE_SET_GAMMA_MODE => cont!(gxm::texture_set_gamma_mode(ctx, st)),
        gxm_nid::SET_FRAGMENT_UNIFORM_BUFFER => cont!(gxm::set_uniform_buffer(ctx, "fragment")),
        gxm_nid::SET_VERTEX_UNIFORM_BUFFER => cont!(gxm::set_uniform_buffer(ctx, "vertex")),
        // Texture getters: pure field reads of the guest's control word 0. Every one of these
        // ALSO has an inline form (`gxm::inline_op`), so on a build that inlines its imports the
        // guest never reaches these at all - which is the point, since `GetLodBias` alone was the
        // hottest host call one title makes.
        gxm_nid::TEXTURE_GET_MIPMAP_COUNT_UNSAFE | gxm_nid::TEXTURE_GET_MIPMAP_COUNT => {
            cont!(gxm::texture_get_mipmap_count(ctx))
        }
        gxm_nid::TEXTURE_GET_STRIDE => cont!(gxm::texture_get_stride(ctx, st)),
        gxm_nid::TEXTURE_GET_LOD_BIAS => cont!(gxm::texture_get_lod_bias(ctx)),
        gxm_nid::TEXTURE_GET_U_ADDR_MODE_SAFE => cont!(gxm::texture_get_u_addr_mode(ctx)),
        gxm_nid::TEXTURE_GET_V_ADDR_MODE_SAFE => cont!(gxm::texture_get_v_addr_mode(ctx)),
        gxm_nid::TEXTURE_GET_MIN_FILTER => cont!(gxm::texture_get_min_filter(ctx)),
        gxm_nid::TEXTURE_GET_MAG_FILTER => cont!(gxm::texture_get_mag_filter(ctx)),
        gxm_nid::TEXTURE_GET_GAMMA_MODE => cont!(gxm::texture_get_gamma_mode(ctx)),
        gxm_nid::TEXTURE_INIT_CUBE => cont!(gxm::texture_init(ctx, st, gxm::TYPE_CUBE)),
        // Color-surface getters/setters beyond format.
        gxm_nid::COLOR_SURFACE_GET_DATA => cont!(gxm::color_surface_get_data(ctx, st)),
        gxm_nid::COLOR_SURFACE_GET_STRIDE_IN_PIXELS => {
            cont!(gxm::color_surface_get_stride_in_pixels(ctx, st))
        }
        gxm_nid::COLOR_SURFACE_SET_GAMMA_MODE => cont!(gxm::color_surface_set_gamma_mode(ctx, st)),
        // Render-target sizing + GPU notification region + program reflection.
        gxm_nid::GET_RENDER_TARGET_MEM_SIZE => cont!(gxm::get_render_target_mem_size(ctx, st)),
        gxm_nid::GET_NOTIFICATION_REGION => cont!(gxm::get_notification_region(ctx, st)),
        gxm_nid::PROGRAM_GET_DEFAULT_UNIFORM_BUFFER_SIZE => {
            cont!(gxm::program_get_default_uniform_buffer_size(ctx, st))
        }
        gxm_nid::FRAGMENT_PROGRAM_GET_PASS_TYPE => cont!(gxm::fragment_program_get_pass_type(ctx, st)),
        // Precomputed draws: record the bundle, replay it as a draw on DrawPrecomputed.
        gxm_nid::GET_PRECOMPUTED_DRAW_SIZE => cont!(gxm::get_precomputed_draw_size(ctx, st)),
        gxm_nid::PRECOMPUTED_DRAW_INIT => cont!(gxm::precomputed_draw_init(ctx, st)),
        gxm_nid::PRECOMPUTED_DRAW_SET_VERTEX_STREAM => {
            cont!(gxm::precomputed_draw_set_vertex_stream(ctx, st))
        }
        gxm_nid::PRECOMPUTED_DRAW_SET_PARAMS => cont!(gxm::precomputed_draw_set_params(ctx, st)),
        gxm_nid::PRECOMPUTED_DRAW_SET_PARAMS_INSTANCED => {
            cont!(gxm::precomputed_draw_set_params_instanced(ctx, st))
        }
        gxm_nid::DRAW_PRECOMPUTED => cont!(gxm::draw_precomputed(ctx, st)),
        gxm_nid::GET_PRECOMPUTED_VERTEX_STATE_SIZE => cont!(gxm::get_precomputed_vertex_state_size(ctx, st)),
        gxm_nid::GET_PRECOMPUTED_FRAGMENT_STATE_SIZE => cont!(gxm::get_precomputed_fragment_state_size(ctx, st)),
        gxm_nid::PRECOMPUTED_VERTEX_STATE_INIT => cont!(gxm::precomputed_vertex_state_init(ctx, st)),
        gxm_nid::PRECOMPUTED_FRAGMENT_STATE_INIT => cont!(gxm::precomputed_fragment_state_init(ctx, st)),
        gxm_nid::PRECOMPUTED_VERTEX_STATE_SET_DEFAULT_UNIFORM_BUFFER => {
            cont!(gxm::precomputed_vertex_state_set_default_uniform_buffer(ctx, st))
        }
        gxm_nid::PRECOMPUTED_FRAGMENT_STATE_SET_DEFAULT_UNIFORM_BUFFER => {
            cont!(gxm::precomputed_fragment_state_set_default_uniform_buffer(ctx, st))
        }
        gxm_nid::PRECOMPUTED_VERTEX_STATE_GET_DEFAULT_UNIFORM_BUFFER => {
            cont!(gxm::precomputed_vertex_state_get_default_uniform_buffer(ctx, st))
        }
        gxm_nid::PRECOMPUTED_FRAGMENT_STATE_GET_DEFAULT_UNIFORM_BUFFER => {
            cont!(gxm::precomputed_fragment_state_get_default_uniform_buffer(ctx, st))
        }
        gxm_nid::PRECOMPUTED_VERTEX_STATE_SET_TEXTURE => cont!(gxm::precomputed_vertex_state_set_texture(ctx, st)),
        gxm_nid::PRECOMPUTED_FRAGMENT_STATE_SET_TEXTURE => cont!(gxm::precomputed_fragment_state_set_texture(ctx, st)),
        gxm_nid::SET_PRECOMPUTED_VERTEX_STATE => cont!(gxm::set_precomputed_vertex_state(ctx, st)),
        gxm_nid::SET_PRECOMPUTED_FRAGMENT_STATE => cont!(gxm::set_precomputed_fragment_state(ctx, st)),
        gxm_nid::PRECOMPUTED_DRAW_SET_ALL_VERTEX_STREAMS => {
            cont!(gxm::precomputed_draw_set_all_vertex_streams(ctx, st))
        }
        gxm_nid::PRECOMPUTED_FRAGMENT_STATE_SET_ALL_TEXTURES => {
            cont!(gxm::precomputed_fragment_state_set_all_textures(ctx, st))
        }
        gxm_nid::PRECOMPUTED_VERTEX_STATE_SET_ALL_TEXTURES => {
            cont!(gxm::precomputed_vertex_state_set_all_textures(ctx, st))
        }
        gxm_nid::PRECOMPUTED_VERTEX_STATE_SET_UNIFORM_BUFFER => {
            cont!(gxm::precomputed_state_set_uniform_buffer(ctx, st, "vertex", false))
        }
        gxm_nid::PRECOMPUTED_FRAGMENT_STATE_SET_UNIFORM_BUFFER => {
            cont!(gxm::precomputed_state_set_uniform_buffer(ctx, st, "fragment", false))
        }
        gxm_nid::PRECOMPUTED_VERTEX_STATE_SET_ALL_UNIFORM_BUFFERS => {
            cont!(gxm::precomputed_state_set_uniform_buffer(ctx, st, "vertex", true))
        }
        gxm_nid::PRECOMPUTED_FRAGMENT_STATE_SET_ALL_UNIFORM_BUFFERS => {
            cont!(gxm::precomputed_state_set_uniform_buffer(ctx, st, "fragment", true))
        }
        // Depth/stencil surface: the published struct is written in place, so a copy of
        // it carries its own state (no address-keyed side table).
        gxm_nid::DEPTH_STENCIL_SURFACE_SET_BACKGROUND_DEPTH => {
            cont!(gxm::depth_stencil_surface_set_background_depth(ctx, st))
        }
        gxm_nid::DEPTH_STENCIL_SURFACE_SET_BACKGROUND_STENCIL => {
            cont!(gxm::depth_stencil_surface_set_background_stencil(ctx, st))
        }
        gxm_nid::DEPTH_STENCIL_SURFACE_SET_FORCE_LOAD_MODE => {
            cont!(gxm::depth_stencil_surface_set_force_load_mode(ctx, st))
        }
        gxm_nid::DEPTH_STENCIL_SURFACE_SET_FORCE_STORE_MODE => {
            cont!(gxm::depth_stencil_surface_set_force_store_mode(ctx, st))
        }
        gxm_nid::SET_BACK_DEPTH_WRITE_ENABLE => cont!(gxm::set_back_depth_write_enable(ctx, st)),
        gxm_nid::SET_BACK_POLYGON_MODE => cont!(gxm::set_back_polygon_mode(ctx, st)),
        gxm_nid::SET_VISIBILITY_BUFFER => cont!(gxm::set_visibility_buffer(ctx, st)),
        gxm_nid::SET_FRONT_VISIBILITY_TEST_ENABLE => cont!(gxm::set_front_visibility_test_enable(ctx, st)),
        gxm_nid::SET_FRONT_VISIBILITY_TEST_INDEX => cont!(gxm::set_front_visibility_test_index(ctx, st)),
        gxm_nid::SET_FRONT_VISIBILITY_TEST_OP => cont!(gxm::set_front_visibility_test_op(ctx, st)),
        gxm_nid::UNMAP_MEMORY
        | gxm_nid::UNMAP_VERTEX_USSE_MEMORY
        | gxm_nid::UNMAP_FRAGMENT_USSE_MEMORY => cont!(gxm::unmap_memory(ctx, st)),
        gxm_nid::COLOR_SURFACE_GET_SCALE_MODE => cont!(gxm::color_surface_get_scale_mode(ctx, st)),
        gxm_nid::COLOR_SURFACE_SET_DATA => cont!(gxm::color_surface_set_data(ctx, st)),
        gxm_nid::PROGRAM_GET_TYPE => cont!(gxm::program_get_type(ctx, st)),
        gxm_nid::PROGRAM_GET_SIZE => cont!(gxm::program_get_size(ctx, st)),
        gxm_nid::PROGRAM_FIND_PARAMETER_BY_SEMANTIC => cont!(gxm::find_parameter_by_semantic(ctx, st)),
        gxm_nid::RENDER_TARGET_GET_DRIVER_MEM_BLOCK => {
            cont!(gxm::render_target_get_driver_mem_block(ctx, st))
        }
        gxm_nid::NOTIFICATION_WAIT => cont!(gxm::notification_wait(ctx, st)),
        gxm_nid::SET_VERTEX_TEXTURE => cont!(gxm::set_vertex_texture(ctx, st)),
        gxm_nid::TEXTURE_INIT_CUBE_ARBITRARY => {
            cont!(gxm::texture_init(ctx, st, gxm::TYPE_CUBE_ARBITRARY))
        }
        gxm_nid::TEXTURE_SET_PALETTE => cont!(gxm::texture_set_palette(ctx, st)),
        gxm_nid::TEXTURE_GET_PALETTE => cont!(gxm::texture_get_palette(ctx, st)),
        // Fixed-function pipeline state: record into the sticky render state that is
        // snapshotted per draw (see `capture::RenderState`).
        gxm_nid::SET_CULL_MODE => cont!(gxm::set_cull_mode(ctx, st)),
        gxm_nid::SET_TWO_SIDED_ENABLE => cont!(gxm::set_two_sided_enable(ctx, st)),
        gxm_nid::SET_FRONT_DEPTH_FUNC => cont!(gxm::set_front_depth_func(ctx, st)),
        gxm_nid::SET_FRONT_DEPTH_BIAS => cont!(gxm::set_front_depth_bias(ctx, st)),
        gxm_nid::SET_BACK_DEPTH_FUNC => cont!(gxm::set_back_depth_func(ctx, st)),
        gxm_nid::SET_FRONT_DEPTH_WRITE_ENABLE => cont!(gxm::set_front_depth_write_enable(ctx, st)),
        gxm_nid::SET_FRONT_FRAGMENT_PROGRAM_ENABLE => {
            cont!(gxm::set_front_fragment_program_enable(ctx, st))
        }
        gxm_nid::SET_BACK_FRAGMENT_PROGRAM_ENABLE => {
            cont!(gxm::set_back_fragment_program_enable(ctx, st))
        }
        gxm_nid::SET_FRONT_POINT_LINE_WIDTH => cont!(gxm::set_front_point_line_width(ctx, st)),
        gxm_nid::SET_FRONT_POLYGON_MODE => cont!(gxm::set_front_polygon_mode(ctx, st)),
        gxm_nid::SET_FRONT_STENCIL_REF => cont!(gxm::set_front_stencil_ref(ctx, st)),
        gxm_nid::SET_FRONT_STENCIL_FUNC => cont!(gxm::set_front_stencil_func(ctx, st)),
        gxm_nid::SET_BACK_STENCIL_FUNC => cont!(gxm::set_back_stencil_func(ctx, st)),
        gxm_nid::SET_VIEWPORT => cont!(gxm::set_viewport(ctx, st)),
        gxm_nid::SET_VIEWPORT_ENABLE => cont!(gxm::set_viewport_enable(ctx, st)),
        gxm_nid::SET_REGION_CLIP => cont!(gxm::set_region_clip(ctx, st)),
        gxm_nid::COLOR_SURFACE_GET_FORMAT => cont!(gxm::color_surface_get_format(ctx, st)),
        gxm_nid::COLOR_SURFACE_GET_TYPE => cont!(gxm::color_surface_get_type(ctx, st)),
        gxm_nid::COLOR_SURFACE_SET_CLIP => cont!(gxm::color_surface_set_clip(ctx, st)),
        gxm_nid::TEXTURE_GET_TYPE => cont!(gxm::texture_get_type(ctx, st)),
        gxm_nid::PROGRAM_PARAMETER_GET_SEMANTIC => cont!(gxm::param_get_semantic(ctx, st)),
        gxm_nid::PROGRAM_PARAMETER_GET_SEMANTIC_INDEX => {
            cont!(gxm::param_get_semantic_index(ctx, st))
        }
        // Texture sampler state: record wrap modes / LOD bias per texture (the plain
        // and "safe" variants set the same state; the safe one also validates on HW).
        gxm_nid::TEXTURE_SET_U_ADDR_MODE | gxm_nid::TEXTURE_SET_U_ADDR_MODE_SAFE => {
            cont!(gxm::texture_set_u_addr_mode(ctx))
        }
        gxm_nid::TEXTURE_SET_V_ADDR_MODE | gxm_nid::TEXTURE_SET_V_ADDR_MODE_SAFE => {
            cont!(gxm::texture_set_v_addr_mode(ctx))
        }
        gxm_nid::TEXTURE_SET_LOD_BIAS => cont!(gxm::texture_set_lod_bias(ctx)),
        gxm_nid::DRAW => cont!(gxm::draw(ctx, st)),
        gxm_nid::DRAW_INSTANCED => cont!(gxm::draw_instanced(ctx, st)),
        gxm_nid::DISPLAY_QUEUE_ADD_ENTRY => {
            // The frame is complete and queued to flip; on hardware the caller waits
            // for the flip here. This is the ONE call that ends a display frame, and
            // so the only source of `Flip` - everything else that gives up the CPU
            // yields WITHOUT counting a frame.
            gxm::display_queue_add_entry(ctx, st);
            SvcOutcome::Flip
        }

        // --- iofilemgr: file IO -------------------------------------------------
        io_nid::IO_OPEN => cont!(iofilemgr::io_open(ctx, st)),
        io_nid::IO_CLOSE => cont!(iofilemgr::io_close(ctx, st)),
        // Reads park the caller for their modelled transfer time, so these two arms
        // return the outcome directly instead of forcing `Continue`.
        io_nid::IO_READ => iofilemgr::io_read(ctx, st),
        io_nid::IO_WRITE => cont!(iofilemgr::io_write(ctx, st)),
        io_nid::IO_LSEEK32 => cont!(iofilemgr::io_lseek32(ctx, st)),
        io_nid::IO_LSEEK => cont!(iofilemgr::io_lseek(ctx, st)),
        io_nid::IO_PREAD => iofilemgr::io_pread(ctx, st),
        io_nid::IO_PWRITE => cont!(iofilemgr::io_pwrite(ctx, st)),
        io_nid::IO_GETSTAT => cont!(iofilemgr::io_getstat(ctx, st)),
        io_nid::IO_GETSTAT_BY_FD => cont!(iofilemgr::io_getstat_by_fd(ctx, st)),
        io_nid::IO_CHSTAT_BY_FD => cont!(iofilemgr::io_chstat_by_fd(ctx, st)),
        io_nid::IO_MKDIR => cont!(iofilemgr::io_mkdir(ctx, st)),
        io_nid::IO_REMOVE => cont!(iofilemgr::io_remove(ctx, st)),
        io_nid::IO_DOPEN => cont!(iofilemgr::io_dopen(ctx, st)),
        io_nid::IO_DREAD => cont!(iofilemgr::io_dread(ctx, st)),
        io_nid::IO_DCLOSE => cont!(iofilemgr::io_dclose(ctx, st)),
        io_nid::IO_SYNC_BY_FD => cont!(iofilemgr::io_sync_by_fd(ctx, st)),
        // The same file operations, re-exported by SceLibKernel under its own NIDs.
        lk_nid::IO_RMDIR => cont!(iofilemgr::io_rmdir(ctx, st)),
        lk_nid::IO_RENAME => cont!(iofilemgr::io_rename(ctx, st)),
        lk_nid::IO_CHSTAT => cont!(iofilemgr::io_chstat(ctx, st)),
        lk_nid::IO_SYNC => cont!(iofilemgr::io_sync(ctx, st)),
        // Device control: a real per-command dispatch that stops the run on a command
        // it does not implement rather than inventing an answer.
        lk_nid::IO_DEVCTL => iofilemgr::io_devctl(ctx, st),
        lk_nid::IO_IOCTL => iofilemgr::io_ioctl(ctx, st),

        // --- thread and semaphore introspection, signals, module queries ---------
        lk_nid::GET_THREAD_INFO => cont!(libkernel::get_thread_info(ctx, st)),
        lk_nid::GET_SEMA_INFO => cont!(sync::get_sema_info(ctx, st)),
        sync_nid::OPEN_SEMA => cont!(sync::open_sema(ctx, st)),
        tm_nid::CHANGE_THREAD_PRIORITY => cont!(threadmgr::change_thread_priority(ctx, st)),
        tm_nid::SEND_SIGNAL => cont!(libkernel::send_signal(ctx, st)),
        lk_nid::WAIT_SIGNAL => libkernel::wait_signal(ctx, st),
        lk_nid::GET_PROCESS_TIME_LOW => cont!(libkernel::get_process_time_low(ctx, st)),
        lk_nid::GET_OPEN_PS_ID => cont!(libkernel::get_open_ps_id(ctx, st)),
        lk_nid::GET_MODULE_INFO_BY_ADDR => cont!(libkernel::get_module_info_by_addr(ctx, st)),
        lk_nid::CALL_MODULE_EXIT => libkernel::call_module_exit(ctx, st),
        // The ARM EABI divide-by-zero hooks. The value the division yields is already
        // in r0 (r0:r1 for the long form) and the default handler returns it unchanged,
        // so these must not write a return value at all.
        lk_nid::AEABI_IDIV0 | lk_nid::AEABI_LDIV0 => libkernel::aeabi_div0(ctx, st),

        // --- SceFios2Kernel: the path overlay layer under FIOS2 ------------------
        fios2_nid::OVERLAY_ADD => cont!(fios2::overlay_add(ctx, st)),
        fios2_nid::OVERLAY_ADD_FOR_PROCESS => cont!(fios2::overlay_add_for_process(ctx, st)),
        fios2_nid::OVERLAY_MODIFY => cont!(fios2::overlay_modify(ctx, st)),
        fios2_nid::OVERLAY_MODIFY_FOR_PROCESS => cont!(fios2::overlay_modify_for_process(ctx, st)),
        fios2_nid::OVERLAY_REMOVE => cont!(fios2::overlay_remove(ctx, st)),
        fios2_nid::OVERLAY_REMOVE_FOR_PROCESS => cont!(fios2::overlay_remove_for_process(ctx, st)),
        fios2_nid::OVERLAY_GET_INFO => cont!(fios2::overlay_get_info(ctx, st)),
        fios2_nid::OVERLAY_GET_INFO_FOR_PROCESS => cont!(fios2::overlay_get_info_for_process(ctx, st)),
        fios2_nid::OVERLAY_GET_LIST => cont!(fios2::overlay_get_list(ctx, st)),
        fios2_nid::OVERLAY_RESOLVE_SYNC => cont!(fios2::overlay_resolve_sync(ctx, st)),
        fios2_nid::OVERLAY_RESOLVE_WITH_RANGE_SYNC => {
            cont!(fios2::overlay_resolve_with_range_sync(ctx, st))
        }
        fios2_nid::OVERLAY_GET_RECOMMENDED_SCHEDULER => {
            cont!(fios2::overlay_get_recommended_scheduler(ctx, st))
        }
        fios2_nid::OVERLAY_THREAD_IS_DISABLED => cont!(fios2::overlay_thread_is_disabled(ctx, st)),
        fios2_nid::OVERLAY_THREAD_SET_DISABLED => cont!(fios2::overlay_thread_set_disabled(ctx, st)),
        fios2_nid::DH_OPEN_SYNC => cont!(fios2::dh_open_sync(ctx, st)),
        fios2_nid::DH_READ_SYNC => cont!(fios2::dh_read_sync(ctx, st)),
        fios2_nid::DH_STAT_SYNC => cont!(fios2::dh_stat_sync(ctx, st)),
        fios2_nid::DH_CHSTAT_SYNC => cont!(fios2::dh_chstat_sync(ctx, st)),
        fios2_nid::DH_SYNC_SYNC => cont!(fios2::dh_sync_sync(ctx, st)),
        fios2_nid::DH_CLOSE_SYNC => cont!(fios2::dh_close_sync(ctx, st)),

        // --- SceLibDbg: the title's own assertions and log lines -----------------
        dbg_nid::ASSERTION_HANDLER => cont!(dbg::assertion_handler(ctx, st)),
        dbg_nid::LOGGING_HANDLER => cont!(dbg::logging_handler(ctx, st)),

        // --- process ------------------------------------------------------------
        pm_nid::LIBC_GETTIMEOFDAY => cont!(processmgr::libc_gettimeofday(ctx, st)),
        pm_nid::CALL_ABORT_HANDLER => processmgr::call_abort_handler(ctx, st),

        // --- sysmem: memory blocks ---------------------------------------------
        sm_nid::ALLOC_MEM_BLOCK => cont!(sysmem::alloc_mem_block(ctx, st)),
        sm_nid::GET_MEM_BLOCK_BASE => cont!(sysmem::get_mem_block_base(ctx, st)),
        sm_nid::FREE_MEM_BLOCK => cont!(sysmem::free_mem_block(ctx, st)),
        sm_nid::SET_GPO => cont!(sysmem::set_gpo(ctx, st)),
        sm_nid::FIND_MEM_BLOCK_BY_ADDR => cont!(sysmem::find_mem_block_by_addr(ctx, st)),
        sm_nid::GET_MEM_BLOCK_INFO_BY_ADDR => cont!(sysmem::get_mem_block_info_by_addr(ctx, st)),

        // --- display ------------------------------------------------------------
        display_nid::SET_FRAME_BUF => cont!(display::set_frame_buf(ctx, st)),
        // A real timed vblank wait (parks under the preemptive scheduler).
        //
        // The `CB` spellings share each handler. A `CB` wait additionally runs the
        // calling thread's pending callbacks, and here that is already true of every
        // wait: this engine delivers callbacks at host-call boundaries, which is where
        // a parked thread resumes. So the difference hardware draws - whether THIS wait
        // is a delivery point - does not exist for us, and folding them is exact rather
        // than an approximation.
        display_nid::WAIT_VBLANK_START_MULTI | display_nid::WAIT_VBLANK_START_MULTI_CB => {
            display::wait_vblank_start_multi(ctx, st)
        }
        display_nid::WAIT_VBLANK_START | display_nid::WAIT_VBLANK_START_CB => {
            display::wait_vblank_start(ctx, st)
        }
        display_nid::WAIT_SET_FRAME_BUF | display_nid::WAIT_SET_FRAME_BUF_CB => {
            display::wait_set_frame_buf(ctx, st)
        }
        display_nid::WAIT_SET_FRAME_BUF_MULTI | display_nid::WAIT_SET_FRAME_BUF_MULTI_CB => {
            display::wait_set_frame_buf_multi(ctx, st)
        }
        display_nid::GET_VCOUNT => cont!(display::get_vcount(ctx, st)),

        // --- ctrl: input --------------------------------------------------------
        ctrl_nid::PEEK_BUFFER_POSITIVE => cont!(ctrl::peek_buffer_positive(ctx, st)),
        ctrl_nid::READ_BUFFER_POSITIVE => ctrl::read_buffer_positive(ctx, st),
        ctrl_nid::PEEK_BUFFER_NEGATIVE => cont!(ctrl::peek_buffer_negative(ctx, st)),
        ctrl_nid::READ_BUFFER_NEGATIVE => ctrl::read_buffer_negative(ctx, st),
        ctrl_nid::SET_SAMPLING_MODE => cont!(ctx.ret(0)),

        // --- ngs / audio --------------------------------------------------------
        ngs_nid::SYSTEM_GET_REQUIRED_MEMORY_SIZE => cont!(ngs::system_get_required_memory_size(ctx, st)),
        ngs_nid::SYSTEM_INIT => cont!(ngs::system_init(ctx, st)),
        ngs_nid::RACK_GET_REQUIRED_MEMORY_SIZE => cont!(ngs::rack_get_required_memory_size(ctx, st)),
        ngs_nid::RACK_INIT => cont!(ngs::rack_init(ctx, st)),
        ngs_nid::RACK_GET_VOICE_HANDLE => cont!(ngs::rack_get_voice_handle(ctx, st)),
        ngs_nid::VOICE_GET_STATE_DATA => cont!(ngs::voice_get_state_data(ctx, st)),
        ngs_nid::VOICE_LOCK_PARAMS => cont!(ngs::voice_lock_params(ctx, st)),
        ngs_nid::VOICE_DEF_GET_SIMPLE_ATRAC9
        | ngs_nid::VOICE_DEF_GET_MASTER_BUSS
        | ngs_nid::VOICE_DEF_GET_REVERB_BUSS
        | ngs_nid::VOICE_DEF_GET_EQ_BUSS
        | ngs_nid::VOICE_DEF_GET_SIMPLE_VOICE
        | ngs_nid::VOICE_DEF_GET_MIXER_BUSS
        | ngs_nid::VOICE_DEF_GET_COMPRESSOR_BUSS
        | ngs_nid::VOICE_DEF_GET_DELAY_BUSS
        | ngs_nid::VOICE_DEF_GET_DISTORTION_BUSS
        | ngs_nid::VOICE_DEF_GET_COMPRESSOR_SIDE_CHAIN_BUSS
        | ngs_nid::VOICE_DEF_GET_SCREAM_ATRAC9_VOICE
        | ngs_nid::VOICE_DEF_GET_SCREAM_VOICE
        | ngs_nid::VOICE_DEF_GET_TEMPLATE1
        | ngs_nid::VOICE_DEF_GET_ATRAC9_VOICE => {
            // One blob per definition, keyed by the NID this call arrived on - the pointer
            // is the only thing a rack description says about what it is made of.
            let addr = ngs::voice_def_get_for(st, func_nid);
            ctx.ret(addr);
            SvcOutcome::Continue
        }
        ngs_nid::PATCH_CREATE_ROUTING => cont!(ngs::patch_create_routing(ctx, st)),
        // The remaining NGS calls are state transitions / per-frame pumps that
        // succeed silently: update/flags/release, voice play/keyoff/kill/pause/
        // resume, param unlock, callbacks, bypass, patch info, AT9 details,
        // out-of-range query (0 = in range).
        ngs_nid::SYSTEM_UPDATE => cont!(ngs::system_update(ctx, st)),
        ngs_nid::VOICE_UNLOCK_PARAMS => cont!(ngs::voice_unlock_params(ctx, st)),
        ngs_nid::VOICE_SET_PARAMS_BLOCK => cont!(ngs::voice_set_params_block(ctx, st)),
        // The callbacks a streaming title lives by: the player module's buffer-boundary
        // callback and the voice-finished one. See `ngs::deliver_player_events`.
        ngs_nid::VOICE_SET_MODULE_CALLBACK => cont!(ngs::voice_set_module_callback(ctx, st)),
        ngs_nid::VOICE_SET_FINISHED_CALLBACK => cont!(ngs::voice_set_finished_callback(ctx, st)),
        ngs_nid::SYSTEM_SET_FLAGS
        | ngs_nid::SYSTEM_RELEASE
        | ngs_nid::RACK_RELEASE
        | ngs_nid::VOICE_RESUME
        | ngs_nid::VOICE_BYPASS_MODULE
        | ngs_nid::VOICE_GET_PARAMS_OUT_OF_RANGE
        | ngs_nid::PATCH_GET_INFO
        | ngs_nid::PATCH_REMOVE_ROUTING
        // System lock/unlock guard the mix graph; single-thread-of-control here, so
        // there is no contention - both succeed immediately.
        | ngs_nid::SYSTEM_LOCK
        | ngs_nid::SYSTEM_UNLOCK
        | ngs_nid::AT9_GET_SECTION_DETAILS => cont!(ctx.ret(0)),
        ngs_nid::VOICE_PLAY => cont!(ngs::voice_play(ctx, st)),
        // The routing volumes: without these every voice mixes at unity and the sum
        // clips. See `ngs::voice_patch_set_volume`.
        ngs_nid::VOICE_PATCH_SET_VOLUME => cont!(ngs::voice_patch_set_volume(ctx, st)),
        ngs_nid::VOICE_PATCH_SET_VOLUMES_MATRIX => {
            cont!(ngs::voice_patch_set_volumes_matrix(ctx, st))
        }
        ngs_nid::VOICE_KEY_OFF | ngs_nid::VOICE_KILL | ngs_nid::VOICE_PAUSE => {
            cont!(ngs::voice_stop(ctx, st))
        }
        ngs_nid::VOICE_INIT => cont!(ngs::voice_init(ctx, st)),
        ngs_nid::VOICE_GET_INFO => cont!(ngs::voice_get_info(ctx, st)),
        audio_nid::OUT_OPEN_PORT => cont!(audio::out_open_port(ctx, st)),
        audio_nid::OUT_OUTPUT => audio::out_output(ctx, st),
        audio_nid::OUT_SET_VOLUME => cont!(audio::out_set_volume(ctx, st)),
        audio_nid::OUT_RELEASE_PORT => cont!(audio::out_release_port(ctx, st)),
        audio_nid::OUT_GET_ADOPT => cont!(audio::out_get_adopt(ctx, st)),

        // --- pvf: font library --------------------------------------------------
        pvf_nid::NEW_LIB => cont!(pvf::new_lib(ctx, st)),
        pvf_nid::DONE_LIB => cont!(pvf::done_lib(ctx, st)),
        pvf_nid::OPEN => cont!(pvf::open(ctx, st)),
        pvf_nid::OPEN_USER_FILE => cont!(pvf::open_user_file(ctx, st)),
        pvf_nid::OPEN_USER_MEMORY => cont!(pvf::open_user_memory(ctx, st)),
        pvf_nid::CLOSE => cont!(pvf::close(ctx, st)),
        pvf_nid::SET_EM => cont!(pvf::set_em(ctx, st)),
        pvf_nid::SET_RESOLUTION => cont!(pvf::set_resolution(ctx, st)),
        pvf_nid::SET_CHAR_SIZE => cont!(pvf::set_char_size(ctx, st)),
        pvf_nid::SET_SKEW_VALUE => cont!(pvf::set_skew_value(ctx, st)),
        pvf_nid::IS_ELEMENT => cont!(pvf::is_element(ctx, st)),
        pvf_nid::GET_FONT_INFO => cont!(pvf::get_font_info(ctx, st)),
        pvf_nid::GET_CHAR_INFO => cont!(pvf::get_char_info(ctx, st)),
        pvf_nid::GET_CHAR_IMAGE_RECT => cont!(pvf::get_char_image_rect(ctx, st)),
        pvf_nid::GET_CHAR_GLYPH_IMAGE => cont!(pvf::get_char_glyph_image(ctx, st)),
        pvf_nid::PIXEL_TO_POINT_H => cont!(pvf::pixel_to_point_h(ctx, st)),
        pvf_nid::PIXEL_TO_POINT_V => cont!(pvf::pixel_to_point_v(ctx, st)),

        // --- processmgr: process param, std streams, time ----------------------
        pm_nid::GET_PROCESS_PARAM => cont!(processmgr::get_process_param(ctx, st)),
        pm_nid::GET_STDIN => cont!(processmgr::get_stdin(ctx, st)),
        pm_nid::GET_STDOUT => cont!(processmgr::get_stdout(ctx, st)),
        pm_nid::GET_STDERR => cont!(processmgr::get_stderr(ctx, st)),
        pm_nid::LIBC_TIME => cont!(processmgr::libc_time(ctx, st)),
        pm_nid::LIBC_CLOCK => cont!(processmgr::libc_clock(ctx, st)),
        pm_nid::POWER_TICK => cont!(ctx.ret(0)),

        // --- services: sysmodule / net / http / np / rtc / apputil / touch -----
        sv_nid::SYSMODULE_IS_LOADED => cont!(services::sysmodule_is_loaded(ctx, st)),
        sv_nid::NET_CTL_INET_GET_STATE => cont!(services::netctl_inet_get_state(ctx, st)),
        sv_nid::NET_CTL_INET_GET_INFO => cont!(services::netctl_inet_get_info(ctx, st)),
        sv_nid::NET_CTL_INET_REGISTER_CALLBACK => cont!(services::netctl_register_callback(ctx, st)),
        sv_nid::NET_CTL_CHECK_CALLBACK => cont!(services::net_check_callback(ctx, st)),
        sv_nid::NP_REGISTER_SERVICE_STATE_CALLBACK => {
            cont!(services::np_register_service_state_callback(ctx, st))
        }
        sv_nid::NP_CHECK_CALLBACK => cont!(services::np_check_callback(ctx, st)),
        sv_nid::NP_BASIC_GET_FRIEND_LIST_ENTRY_COUNT => {
            cont!(services::np_basic_get_friend_list_entry_count(ctx, st))
        }
        sv_nid::RTC_GET_CURRENT_CLOCK => cont!(services::rtc_get_current_clock(ctx, st)),
        sv_nid::RTC_SET_TIME64_T => cont!(services::rtc_set_time64_t(ctx, st)),
        // SceAppUtil: the app-event queue (permanently empty offline) and the savedata
        // remove/quota pair, both of which act on the real guest filesystem.
        sv_nid::APPUTIL_RECEIVE_APP_EVENT => cont!(services::apputil_receive_app_event(ctx, st)),
        sv_nid::APPUTIL_APP_EVENT_PARSE_NEAR_GIFT
        | sv_nid::APPUTIL_APP_EVENT_PARSE_NP_BASIC_JOINABLE_PRESENCE
        | sv_nid::APPUTIL_APP_EVENT_PARSE_NP_INVITE_MESSAGE => {
            cont!(services::apputil_app_event_parse(ctx, st))
        }
        sv_nid::APPUTIL_SAVEDATA_DATA_REMOVE => cont!(services::apputil_savedata_data_remove(ctx, st)),
        sv_nid::APPUTIL_SAVEDATA_GET_QUOTA => cont!(services::apputil_savedata_get_quota(ctx, st)),
        sv_nid::NET_CTL_INET_GET_RESULT | sv_nid::NET_CTL_ADHOC_GET_RESULT => {
            cont!(services::net_ctl_get_result(ctx, st))
        }
        sv_nid::APPMGR_RECEIVE_SYSTEM_EVENT => cont!(services::appmgr_receive_system_event(ctx, st)),
        // Ends the run rather than returning: the title asked to be replaced.
        sv_nid::APPMGR_LOAD_EXEC => services::appmgr_load_exec(ctx, st),
        sv_nid::SHUTTER_SOUND_PLAY => cont!(services::shutter_sound_play(ctx, st)),
        sv_nid::PHOTO_EXPORT_FROM_DATA => cont!(services::photo_export_from_data(ctx, st)),
        // SceNp, offline: see the module's own section for why each of these reports
        // signed-out rather than fabricating an account, a ticket or a friend list.
        sv_nid::NP_GET_SERVICE_STATE => cont!(services::np_get_service_state(ctx, st)),
        sv_nid::NP_BASIC_GET_FRIEND_LIST_ENTRIES => {
            cont!(services::np_basic_get_friend_list_entries(ctx, st))
        }
        sv_nid::NP_BASIC_GET_GAME_JOINING_PRESENCE => {
            cont!(services::np_basic_get_game_joining_presence(ctx, st))
        }
        sv_nid::NP_BASIC_SET_IN_GAME_PRESENCE | sv_nid::NP_BASIC_UNREGISTER_HANDLER => {
            cont!(services::np_basic_presence_ok(ctx, st))
        }
        sv_nid::NP_LOOKUP_CREATE_TITLE_CTX => cont!(services::np_lookup_create_title_ctx(ctx, st)),
        sv_nid::NP_LOOKUP_DELETE_REQUEST => cont!(services::np_lookup_delete_request(ctx, st)),
        sv_nid::NP_LOOKUP_USER_PROFILE_ASYNC | sv_nid::NP_LOOKUP_POLL_ASYNC => {
            cont!(services::np_lookup_async(ctx, st))
        }
        sv_nid::NP_AUTH_DESTROY_REQUEST => cont!(services::np_lookup_delete_request(ctx, st)),
        sv_nid::NP_AUTH_CREATE_START_REQUEST
        | sv_nid::NP_AUTH_GET_TICKET
        | sv_nid::NP_AUTH_GET_TICKET_PARAM
        | sv_nid::NP_AUTH_GET_ENTITLEMENT_BY_ID
        | sv_nid::NP_AUTH_GET_ENTITLEMENT_ID_LIST => cont!(services::np_auth_signed_out(ctx, st)),
        sv_nid::NP_ACTIVITY_POST_STATUS => cont!(services::np_activity_post_status(ctx, st)),
        sv_nid::RTC_GET_CURRENT_CLOCK_LOCAL_TIME => {
            cont!(services::rtc_get_current_clock_local_time(ctx, st))
        }
        sv_nid::RTC_GET_CURRENT_TICK => cont!(services::rtc_get_current_tick(ctx, st)),
        sv_nid::RTC_GET_TICK_RESOLUTION => cont!(services::rtc_get_tick_resolution(ctx, st)),
        sv_nid::RTC_CONVERT_UTC_TO_LOCAL_TIME | sv_nid::RTC_CONVERT_LOCAL_TIME_TO_UTC => {
            cont!(services::rtc_convert_time_zone(ctx, st))
        }
        sv_nid::RTC_GET_TICK => cont!(services::rtc_get_tick(ctx, st)),
        sv_nid::RTC_GET_TIME64_T => cont!(services::rtc_get_time64_t(ctx, st)),
        sv_nid::RTC_GET_TIME_T => cont!(services::rtc_get_time_t(ctx, st)),
        sv_nid::RTC_GET_CURRENT_NETWORK_TICK => {
            cont!(services::rtc_get_current_network_tick(ctx, st))
        }
        sv_nid::RTC_SET_TICK => cont!(services::rtc_set_tick(ctx, st)),
        // The sceRtcTickAdd* family, by unit. `true` marks the forms whose count is a
        // 64-bit SceLong64 (an aligned register pair), `false` a plain int.
        sv_nid::RTC_TICK_ADD_TICKS => cont!(services::rtc_tick_add_fixed(ctx, 1, true)),
        sv_nid::RTC_TICK_ADD_MICROSECONDS => cont!(services::rtc_tick_add_fixed(ctx, 1, true)),
        sv_nid::RTC_TICK_ADD_SECONDS => cont!(services::rtc_tick_add_fixed(ctx, 1_000_000, true)),
        sv_nid::RTC_TICK_ADD_MINUTES => cont!(services::rtc_tick_add_fixed(ctx, 60_000_000, true)),
        sv_nid::RTC_TICK_ADD_HOURS => cont!(services::rtc_tick_add_fixed(ctx, 3_600_000_000, false)),
        sv_nid::RTC_TICK_ADD_DAYS => {
            cont!(services::rtc_tick_add_fixed(ctx, 86_400_000_000, false))
        }
        sv_nid::RTC_TICK_ADD_WEEKS => {
            cont!(services::rtc_tick_add_fixed(ctx, 7 * 86_400_000_000, false))
        }
        sv_nid::RTC_TICK_ADD_MONTHS => cont!(services::rtc_tick_add_calendar(ctx, 1)),
        sv_nid::RTC_TICK_ADD_YEARS => cont!(services::rtc_tick_add_calendar(ctx, 12)),
        sv_nid::MOTION_GET_STATE => cont!(services::motion_get_state(ctx, st)),
        // Motion tuning knobs. The modelled device is perfectly still and perfectly
        // level, so a deadband and a tilt correction change nothing about what
        // `sceMotionGetState` reports - but they are accepted, because refusing them
        // would fail a title's sensor setup for a setting that cannot matter here.
        // ...and they are HELD, because each has a getter that reads it back.
        sv_nid::MOTION_SET_DEADBAND => cont!(services::motion_set_tuning(ctx, st, false)),
        sv_nid::MOTION_SET_TILT_CORRECTION => cont!(services::motion_set_tuning(ctx, st, true)),
        sv_nid::MOTION_GET_DEADBAND => cont!(services::motion_get_tuning(ctx, st, false)),
        sv_nid::MOTION_GET_TILT_CORRECTION => cont!(services::motion_get_tuning(ctx, st, true)),
        sv_nid::MOTION_ROTATE_YAW => cont!(services::motion_rotate_yaw(ctx, st)),
        sv_nid::RTC_GET_DAY_OF_WEEK => cont!(services::rtc_get_day_of_week(ctx, st)),
        sv_nid::RTC_FORMAT_RFC3339_LOCAL_TIME => {
            cont!(services::rtc_format_rfc3339_local_time(ctx, st))
        }
        sv_nid::APPUTIL_SYSTEM_PARAM_GET_INT => cont!(services::apputil_system_param_get_int(ctx, st)),
        sv_nid::APPUTIL_APP_PARAM_GET_INT => cont!(services::apputil_app_param_get_int(ctx, st)),
        sv_nid::LIVE_AREA_GET_STATUS => cont!(services::live_area_get_status(ctx, st)),
        sv_nid::APPUTIL_SYSTEM_PARAM_GET_STRING => cont!(services::apputil_system_param_get_string(ctx, st)),
        sv_nid::APPUTIL_DRM_OPEN => cont!(services::apputil_drm_open(ctx, st)),
        sv_nid::APPUTIL_DRM_CLOSE => cont!(services::apputil_drm_close(ctx, st)),
        sv_nid::APPUTIL_SAVEDATA_SLOT_GET_PARAM => {
            cont!(services::apputil_savedata_slot_get_param(ctx, st))
        }
        sv_nid::APPUTIL_SAVEDATA_SLOT_CREATE => {
            cont!(services::apputil_savedata_slot_create(ctx, st))
        }
        sv_nid::APPUTIL_SAVEDATA_SLOT_SET_PARAM => {
            cont!(services::apputil_savedata_slot_set_param(ctx, st))
        }
        sv_nid::APPUTIL_SAVEDATA_SLOT_DELETE => {
            cont!(services::apputil_savedata_slot_delete(ctx, st))
        }
        sv_nid::APPUTIL_SAVEDATA_DATA_SAVE => {
            cont!(services::apputil_savedata_data_save(ctx, st))
        }
        sv_nid::APP_MGR_GET_APP_STATE => cont!(services::app_mgr_get_app_state(ctx, st)),
        sv_nid::APP_MGR_IS_GAME_PROGRAM => cont!(services::app_mgr_is_game_program(ctx, st)),
        sv_nid::FIOS_OVERLAY_GET_RECOMMENDED_SCHEDULER => {
            cont!(services::fios_overlay_get_recommended_scheduler(ctx, st))
        }
        // Offline services with an out-param handle to hand back.
        sv_nid::NETCTL_ADHOC_REGISTER_CALLBACK => {
            cont!(services::netctl_adhoc_register_callback(ctx, st))
        }
        sv_nid::NETCTL_ADHOC_GET_IN_ADDR => cont!(services::netctl_adhoc_get_in_addr(ctx, st)),
        sv_nid::NETCTL_ADHOC_GET_STATE => cont!(services::netctl_adhoc_get_state(ctx, st)),
        sv_nid::NETCTL_ADHOC_GET_PEER_LIST => {
            cont!(services::netctl_adhoc_get_peer_list(ctx, st))
        }
        sv_nid::NETCTL_ADHOC_DISCONNECT => cont!(services::netctl_adhoc_disconnect(ctx, st)),
        sv_nid::ADHOC_MATCHING_INIT => cont!(net::adhoc_matching_init(ctx, st)),
        sv_nid::ADHOC_MATCHING_CREATE => cont!(net::adhoc_matching_create(ctx, st)),
        sv_nid::ADHOC_MATCHING_START => cont!(net::adhoc_matching_set_started(ctx, st, true)),
        sv_nid::ADHOC_MATCHING_STOP => cont!(net::adhoc_matching_set_started(ctx, st, false)),
        sv_nid::ADHOC_MATCHING_DELETE => cont!(net::adhoc_matching_delete(ctx, st)),
        sv_nid::ADHOC_MATCHING_SELECT_TARGET => cont!(net::adhoc_matching_select_target(ctx, st)),
        sv_nid::MP4_OPEN_FILE => cont!(video::mp4_open_file(ctx, st)),
        sv_nid::MP4_START_FILE_STREAMING => cont!(video::mp4_start_file_streaming(ctx, st)),
        sv_nid::MP4_CLOSE_FILE => cont!(video::mp4_close_file(ctx, st)),
        sv_nid::MP4_RELEASE_BUFFER_7B4832FE => cont!(video::mp4_release_buffer(ctx, st)),
        sv_nid::MP4_GET_NEXT_UNIT_8BE0E3D3 => cont!(video::mp4_get_next_unit(ctx, st)),
        sv_nid::MP4_ENABLE_STREAM_609E57AD => cont!(video::mp4_enable_stream(ctx, st)),
        // NOT `cont!`: a unit that is not due yet parks the caller briefly rather than being
        // refused into a spin. See `video::mp4_get_next_unit_info`.
        sv_nid::MP4_GET_NEXT_UNIT => video::mp4_get_next_unit_info(ctx, st),
        sv_nid::MP4_RESET_40351E1A => cont!(video::mp4_reset(ctx, st)),
        // NOT `cont!`: reading a unit is a STORAGE read on the device and is paced like
        // one, or the demuxer outruns the decoder. See `video::mp4_get_next_unit_data`.
        sv_nid::MP4_GET_NEXT_UNIT_DATA => video::mp4_get_next_unit_data(ctx, st),
        // SceVideodec / SceAvcdec: the guest decodes the movie itself.
        vd_nid::VIDEODEC_QUERY_MEM_SIZE => cont!(avcdec::videodec_query_mem_size(ctx, st)),
        vd_nid::VIDEODEC_INIT_LIBRARY_WITH_UNMAP_MEM => {
            cont!(avcdec::videodec_init_library_with_unmap_mem(ctx, st))
        }
        vd_nid::VIDEODEC_TERM_LIBRARY => cont!(avcdec::videodec_term_library(ctx, st)),
        vd_nid::AVCDEC_QUERY_DECODER_MEM_SIZE => {
            cont!(avcdec::avcdec_query_decoder_mem_size(ctx, st))
        }
        vd_nid::AVCDEC_CREATE_DECODER => cont!(avcdec::avcdec_create_decoder(ctx, st)),
        vd_nid::AVCDEC_DELETE_DECODER => cont!(avcdec::avcdec_delete_decoder(ctx, st)),
        // NOT `cont!`: a decode that produced nothing parks its caller, which is what lets
        // a browser's decoder answer at all. See `avcdec::avcdec_decode`.
        vd_nid::AVCDEC_DECODE => avcdec::avcdec_decode(ctx, st),
        vd_nid::AVCDEC_DECODE_STOP => cont!(avcdec::avcdec_decode_stop(ctx, st)),
        vd_nid::AVCDEC_DECODE_FLUSH => cont!(avcdec::avcdec_decode_flush(ctx, st)),
        // SceAudiodec: a movie's sound. Every one of these is synchronous - the decoding
        // itself happened when the demuxer handed the unit over, so nothing here waits.
        ad_nid::GET_CONTEXT_SIZE => cont!(audiodec::audiodec_get_context_size(ctx, st)),
        ad_nid::CREATE_DECODER_EXTERNAL => {
            cont!(audiodec::audiodec_create_decoder_external(ctx, st))
        }
        ad_nid::DECODE => cont!(audiodec::audiodec_decode(ctx, st)),
        // The AT9 family: the title's own stream, decoded synchronously in the call.
        ad_nid::INIT_LIBRARY => cont!(audiodec::audiodec_init_library(ctx, st)),
        ad_nid::TERM_LIBRARY => cont!(audiodec::audiodec_term_library(ctx, st)),
        ad_nid::CREATE_DECODER => cont!(audiodec::audiodec_create_decoder(ctx, st)),
        ad_nid::DELETE_DECODER => cont!(audiodec::audiodec_delete_decoder(ctx, st)),
        ad_nid::CLEAR_CONTEXT => cont!(audiodec::audiodec_clear_context(ctx, st)),
        ad_nid::GET_INTERNAL_ERROR => cont!(audiodec::audiodec_get_internal_error(ctx, st)),
        ad_nid::DELETE_DECODER_EXTERNAL => {
            cont!(audiodec::audiodec_delete_decoder_external(ctx, st))
        }
        vd_nid::CODEC_ENGINE_OPEN_UNMAP_MEM_BLOCK => {
            cont!(avcdec::codec_engine_open_unmap_mem_block(ctx, st))
        }
        vd_nid::CODEC_ENGINE_CLOSE_UNMAP_MEM_BLOCK => {
            cont!(avcdec::codec_engine_close_unmap_mem_block(ctx, st))
        }
        vd_nid::CODEC_ENGINE_ALLOC_MEMORY_FROM_UNMAP_MEM_BLOCK => {
            cont!(avcdec::codec_engine_alloc_memory_from_unmap_mem_block(ctx, st))
        }
        vd_nid::CODEC_ENGINE_FREE_MEMORY_FROM_UNMAP_MEM_BLOCK => {
            cont!(avcdec::codec_engine_free_memory_from_unmap_mem_block(ctx, st))
        }
        sv_nid::NP_TROPHY_CREATE_CONTEXT => cont!(services::np_trophy_create_context(ctx, st)),
        sv_nid::NP_TROPHY_DESTROY_CONTEXT => cont!(services::np_trophy_destroy_context(ctx, st)),
        sv_nid::NP_TROPHY_CREATE_HANDLE => cont!(services::np_trophy_create_handle(ctx, st)),
        sv_nid::NP_TROPHY_GET_GAME_INFO => cont!(services::np_trophy_get_game_info(ctx, st)),
        sv_nid::NP_TROPHY_GET_GAME_ICON => cont!(services::np_trophy_get_game_icon(ctx, st)),
        sv_nid::NP_TROPHY_GET_GROUP_INFO => cont!(services::np_trophy_get_group_info(ctx, st)),
        sv_nid::NP_TROPHY_GET_GROUP_ICON => cont!(services::np_trophy_get_group_icon(ctx, st)),
        sv_nid::NP_TROPHY_GET_TROPHY_INFO => cont!(services::np_trophy_get_trophy_info(ctx, st)),
        sv_nid::NP_TROPHY_GET_TROPHY_ICON => cont!(services::np_trophy_get_trophy_icon(ctx, st)),
        sv_nid::NP_TROPHY_GET_TROPHY_UNLOCK_STATE => cont!(services::np_trophy_get_trophy_unlock_state(ctx, st)),
        sv_nid::NP_TROPHY_UNLOCK_TROPHY => cont!(services::np_trophy_unlock_trophy(ctx, st)),
        // The trophy-setup dialog's result read (zeroed result = OK), like the other
        // dialog GetResult calls.
        sv_nid::NP_TROPHY_SETUP_DIALOG_GET_RESULT => cont!(services::dialog_ok(ctx, st)),
        // SceSystemGesture. No published prototype anywhere - see `vita::gesture` for
        // where each argument shape comes from. The rest of the family is deliberately
        // left to the hard-fail below until its call is observed.
        sv_nid::SYSTEM_GESTURE_INIT_PRIMITIVE_TOUCH_RECOGNIZER => {
            cont!(gesture::init_primitive_touch_recognizer(ctx, st))
        }
        sv_nid::SYSTEM_GESTURE_CREATE_TOUCH_RECOGNIZER => {
            cont!(gesture::create_touch_recognizer(ctx, st))
        }
        sv_nid::SYSTEM_GESTURE_UPDATE_PRIMITIVE_TOUCH_RECOGNIZER => {
            cont!(gesture::update_primitive_touch_recognizer(ctx, st))
        }
        sv_nid::SYSTEM_GESTURE_UPDATE_TOUCH_RECOGNIZER => {
            cont!(gesture::update_touch_recognizer(ctx, st))
        }
        sv_nid::SYSTEM_GESTURE_GET_TOUCH_EVENTS_COUNT => {
            cont!(gesture::get_touch_events_count(ctx, st))
        }
        sv_nid::SYSTEM_GESTURE_GET_TOUCH_EVENT_BY_INDEX => {
            cont!(gesture::get_touch_event_by_index(ctx, st))
        }
        sv_nid::SYSTEM_GESTURE_GET_TOUCH_RECOGNIZER_INFORMATION => {
            cont!(gesture::get_touch_recognizer_information(ctx, st))
        }
        sv_nid::SYSTEM_GESTURE_RESET_TOUCH_RECOGNIZER => {
            cont!(gesture::reset_touch_recognizer(ctx, st))
        }
        sv_nid::SYSTEM_GESTURE_GET_PRIMITIVE_TOUCH_EVENT_BY_PRIMITIVE_ID => {
            cont!(gesture::get_primitive_touch_event_by_primitive_id(ctx, st))
        }
        // SceCamera: no camera is attached to this host, and every entry point says so
        // with the API's own SCE_CAMERA_ERROR_NOT_MOUNTED / _NOT_OPEN. See `vita::camera`.
        // SceJpegEnc: context setup is real; Encode/Csc are left to the hard-fail
        // because there is no honest way to hand back a JPEG that was never encoded.
        sv_nid::JPEG_INIT_MJPEG => cont!(jpeg::init_mjpeg(ctx, st)),
        sv_nid::JPEG_FINISH_MJPEG => cont!(jpeg::finish_mjpeg(ctx, st)),
        sv_nid::JPEGENC_GET_CONTEXT_SIZE => cont!(jpegenc::get_context_size(ctx, st)),
        sv_nid::JPEGENC_INIT => cont!(jpegenc::init(ctx, st)),
        sv_nid::JPEGENC_END => cont!(jpegenc::end(ctx, st)),
        sv_nid::JPEGENC_SET_OUTPUT_ADDR => cont!(jpegenc::set_output_addr(ctx, st)),
        sv_nid::JPEGENC_SET_COMPRESSION_RATIO => cont!(jpegenc::set_compression_ratio(ctx, st)),
        sv_nid::JPEGENC_SET_VALID_REGION => cont!(jpegenc::set_valid_region(ctx, st)),
        sv_nid::CAMERA_OPEN => cont!(camera::open(ctx, st)),
        sv_nid::CAMERA_CLOSE => cont!(camera::close(ctx, st)),
        sv_nid::CAMERA_START => cont!(camera::start(ctx, st)),
        sv_nid::CAMERA_STOP => cont!(camera::stop(ctx, st)),
        sv_nid::CAMERA_READ => cont!(camera::read(ctx, st)),
        sv_nid::CAMERA_GET_REVERSE => cont!(camera::get_reverse(ctx, st)),
        sv_nid::CAMERA_SET_REVERSE => cont!(camera::set_reverse(ctx, st)),
        sv_nid::CAMERA_SET_BACKLIGHT => cont!(camera::set_backlight(ctx, st)),
        sv_nid::CAMERA_SET_WHITE_BALANCE => cont!(camera::set_white_balance(ctx, st)),
        // SceLibLocation: the positioning service, served from the host's own provider
        // through the `World` seam (see `vita::location`).
        sv_nid::LOCATION_OPEN => cont!(location::open(ctx, st)),
        sv_nid::LOCATION_CLOSE => cont!(location::close(ctx, st)),
        sv_nid::LOCATION_REOPEN => cont!(location::reopen(ctx, st)),
        sv_nid::LOCATION_GET_METHOD => cont!(location::get_method(ctx, st)),
        sv_nid::LOCATION_CONFIRM => cont!(location::confirm(ctx, st)),
        sv_nid::LOCATION_CONFIRM_GET_STATUS => cont!(location::confirm_get_status(ctx, st)),
        sv_nid::LOCATION_CONFIRM_GET_RESULT => cont!(location::confirm_get_result(ctx, st)),
        sv_nid::LOCATION_CONFIRM_ABORT => cont!(location::confirm_abort(ctx, st)),
        sv_nid::LOCATION_GET_LOCATION => cont!(location::get_location(ctx, st)),
        sv_nid::LOCATION_GET_LOCATION_WITH_TIMEOUT => {
            cont!(location::get_location_with_timeout(ctx, st))
        }
        sv_nid::LOCATION_CANCEL_GET_LOCATION => cont!(location::cancel_get_location(ctx, st)),
        sv_nid::LOCATION_GET_HEADING => cont!(location::get_heading(ctx, st)),
        sv_nid::LOCATION_GET_PERMISSION => cont!(location::get_permission(ctx, st)),
        sv_nid::LOCATION_DENY_APPLICATION => cont!(location::deny_application(ctx, st)),
        sv_nid::LOCATION_INIT => cont!(location::init(ctx, st)),
        sv_nid::LOCATION_TERM => cont!(location::term(ctx, st)),
        sv_nid::LOCATION_SET_THREAD_PARAMETER => cont!(location::set_thread_parameter(ctx, st)),
        sv_nid::TOUCH_SET_SAMPLING_STATE => cont!(touch::set_sampling_state(ctx, st)),
        sv_nid::TOUCH_GET_SAMPLING_STATE => cont!(touch::get_sampling_state(ctx, st)),
        sv_nid::TOUCH_READ => cont!(touch::read(ctx, st)),
        sv_nid::TOUCH_PEEK => cont!(touch::peek(ctx, st)),
        sv_nid::TOUCH_GET_PANEL_INFO => cont!(touch::get_panel_info(ctx, st)),
        // No online account off-console: identity calls report signed-out so the
        // title takes its offline path instead of dereferencing a null identity.
        // sceNpManagerGetAccountRegion is an account-identity query (account
        // country + language); with no account off-console the faithful signal is
        // signed-out, same as GetNpId, not a fabricated region.
        sv_nid::NP_MANAGER_GET_NP_ID
        | sv_nid::NP_MANAGER_GET_ACCOUNT_REGION
        | sv_nid::NP_MANAGER_GET_CONTENT_RATING_FLAG
        | sv_nid::NP_MANAGER_GET_CHAT_RESTRICTION_FLAG
        | sv_nid::NP_LOOKUP_CREATE_REQUEST
        | sv_nid::NP_MESSAGE_SYNC_MESSAGE
        | sv_nid::NP_TUS_CREATE_REQUEST
        | sv_nid::NP_COMMERCE2_START_EMPTY_STORE_CHECK
        | sv_nid::NP_COMMERCE2_CREATE_SESSION_GET_RESULT
        | sv_nid::NP_SCORE_CREATE_TITLE_CTX
        // SceNpMatching2, the lobby/room surface, and SceNpScore's request surface. Both
        // families hang off a context that has to be created against a live service, and
        // BOTH of those creations already report signed out here - so every call below is
        // reached either with no context at all or with one the title only thinks it has.
        // Reporting the same signed-out cause is what tells it which, where a per-call
        // "bad id" would read as a retryable mistake in its own bookkeeping.
        //
        // The teardown calls are NOT in this group: see the success group below for why
        // destroying something that was never created still succeeds.
        | sv_nid::NP_MATCHING2_CREATE_CONTEXT
        | sv_nid::NP_MATCHING2_CONTEXT_START
        | sv_nid::NP_MATCHING2_REGISTER_CONTEXT_CALLBACK
        | sv_nid::NP_MATCHING2_REGISTER_ROOM_EVENT_CALLBACK
        | sv_nid::NP_MATCHING2_REGISTER_ROOM_MESSAGE_CALLBACK
        | sv_nid::NP_MATCHING2_SET_DEFAULT_REQUEST_OPT_PARAM
        | sv_nid::NP_MATCHING2_GET_SERVER_LOCAL
        | sv_nid::NP_MATCHING2_GET_WORLD_INFO_LIST
        | sv_nid::NP_MATCHING2_SEARCH_ROOM
        | sv_nid::NP_MATCHING2_CREATE_JOIN_ROOM
        | sv_nid::NP_MATCHING2_JOIN_ROOM
        | sv_nid::NP_MATCHING2_SEND_ROOM_MESSAGE
        | sv_nid::NP_SCORE_CREATE_REQUEST
        | sv_nid::NP_SCORE_RECORD_SCORE
        | sv_nid::NP_SCORE_RECORD_SCORE_ASYNC
        | sv_nid::NP_SCORE_GET_RANKING_BY_RANGE
        | sv_nid::NP_SCORE_GET_RANKING_BY_RANGE_ASYNC
        // The async poll: no request was ever accepted, so there is no operation whose
        // completion this could report.
        | sv_nid::NP_SCORE_POLL_ASYNC => {
            cont!(ctx.ret(services::SCE_NP_ERROR_SIGNED_OUT as u32))
        }
        // Everything else here is an init/register that simply succeeds offline.
        sv_nid::NET_INIT
        | sv_nid::NET_CTL_INIT
        | sv_nid::HTTP_INIT
        | sv_nid::SSL_INIT
        | sv_nid::NP_INIT
        | sv_nid::NP_BASIC_INIT
        | sv_nid::NP_BASIC_REGISTER_HANDLER
        // NpBasic per-frame pump: no presence/friend events exist off-console.
        | sv_nid::NP_BASIC_CHECK_CALLBACK
        | sv_nid::FIOS_OVERLAY_GET_LIST
        // Enabling/disabling FIOS overlay resolution for a thread: there are no overlays
        // mounted off-console (the list above is empty), so there is nothing to switch.
        | sv_nid::FIOS_OVERLAY_THREAD_SET_DISABLED
        | sv_nid::ULOBJ_REGISTER_PROTOCOL_REVISION
        | sv_nid::APPUTIL_INIT
        | sv_nid::NP_SCORE_INIT
        // Tearing down the ranking library: there is nothing behind it off-console, so a
        // term has nothing to fail at. Titles reach this from their own no-network cleanup
        // path (which is where `sceNetCtlInetGetInfo` reporting NOT_CONNECTED sends them).
        | sv_nid::NP_SCORE_TERM
        // The requested module is already linked into the image, so a load succeeds.
        | sv_nid::SYSMODULE_LOAD_MODULE
        // SceAppMgr: claim the shared background-music port. On the console this
        // arbitrates against the system's own music player; nothing else is playing
        // here, so the claim is granted. `sceAudioOut` opening a BGM-type port is what
        // actually produces sound (see `vita::audio`), and that is independent of this.
        | sv_nid::APPMGR_ACQUIRE_BGM_PORT
        // ScePerf/Razor: a marker packet for the CPU profiler's timeline. No profiler is
        // attached and there is no capture buffer to append to, so the packet has nowhere
        // to go - which is exactly the retail case the call is written to survive.
        | sv_nid::RAZOR_CPU_WRITE_FIBER_ULT_PKT
        // SceUlobjDbg: the ULT runtime announcing its objects to a debugger, and taking
        // the announcement back. READ OFF THE TWO CALL SITES, which are the only ones in
        // any module here: `libult.suprx` calls the first at the end of building its
        // object array (`f(pool, 1, pool+0x38)`) and the second on teardown, one argument,
        // an object handle read from `[obj+0x38]`. BOTH DISCARD THE RESULT - the register
        // site's `r0` is overwritten by the next instruction and so is the unregister's -
        // so the call cannot be observed to fail, and with no debugger attached there is
        // no object table to announce into. That makes this a genuine nothing-to-do, not a
        // skipped step: the same position as the Razor packet above.
        | sv_nid::ULOBJ_DBG_REGISTER
        | sv_nid::ULOBJ_DBG_UNREGISTER
        | sv_nid::TOUCH_ENABLE_TOUCH_FORCE
        // SceScreenShot: nothing to capture off-console.
        | sv_nid::SCREENSHOT_DISABLE
        | sv_nid::SCREENSHOT_ENABLE
        | sv_nid::SCREENSHOT_SET_PARAM
        | sv_nid::SCREENSHOT_SET_OVERLAY_IMAGE
        // SceNpTrophy init/term and handle lifetime: a handle only scopes an async
        // operation, and every query here completes synchronously, so there is nothing
        // for destroy/abort to cancel.
        | sv_nid::NP_TROPHY_INIT
        | sv_nid::NP_TROPHY_TERM
        | sv_nid::NP_TROPHY_DESTROY_HANDLE
        | sv_nid::NP_TROPHY_ABORT_HANDLE
        | sv_nid::NP_ACTIVITY_INIT
        | sv_nid::NP_AUTH_INIT
        | sv_nid::NP_LOOKUP_INIT
        | sv_nid::NP_TUS_INIT
        | sv_nid::NP_MESSAGE_INIT_WITH_PARAM
        | sv_nid::NP_MESSAGE_TERM
        | sv_nid::NP_MATCHING2_INIT
        // Matching2 / NpScore TEARDOWN, for the same reason the rest of the online stack's
        // teardown succeeds: nothing was created, so there is nothing that can fail to be
        // released, and a title unwinding after its offline path found no service must not
        // be handed an error on the way out.
        | sv_nid::NP_MATCHING2_DESTROY_CONTEXT
        | sv_nid::NP_MATCHING2_CONTEXT_STOP
        | sv_nid::NP_MATCHING2_ABORT_CONTEXT_START
        | sv_nid::NP_SCORE_DELETE_REQUEST
        // Online-stack TEARDOWN. A title that finds itself offline unwinds the whole
        // stack it brought up; terminating a subsystem with no backing service, and
        // unregistering a callback that never fired, genuinely succeed.
        | sv_nid::NP_TERM
        | sv_nid::NP_UNREGISTER_SERVICE_STATE_CALLBACK
        | sv_nid::NP_BASIC_TERM
        | sv_nid::NP_ACTIVITY_TERM
        | sv_nid::NP_AUTH_TERM
        | sv_nid::NP_LOOKUP_TERM
        | sv_nid::NP_LOOKUP_DELETE_TITLE_CTX
        | sv_nid::NP_TUS_TERM
        | sv_nid::NP_TUS_DELETE_TITLE_CTX
        | sv_nid::NP_SCORE_DELETE_TITLE_CTX
        | sv_nid::NP_MATCHING2_TERM
        | sv_nid::HTTP_TERM
        | sv_nid::SSL_TERM
        | sv_nid::NET_TERM
        | sv_nid::NET_CTL_TERM
        | sv_nid::NET_CTL_INET_UNREGISTER_CALLBACK
        | sv_nid::NETCTL_ADHOC_UNREGISTER_CALLBACK
        | sv_nid::SYSMODULE_UNLOAD_MODULE
        | sv_nid::APPUTIL_SHUTDOWN
        | sv_nid::NP_COMMERCE2_INIT
        // SceNpCommerce2 context/request creation: local handle setup that succeeds; the
        // actual store fetch has no server to reach off-console and returns no content.
        | sv_nid::NP_COMMERCE2_CREATE_CTX
        | sv_nid::NP_COMMERCE2_CREATE_SESSION_CREATE_REQ
        | sv_nid::NP_COMMERCE2_CREATE_SESSION_START
        | sv_nid::NP_SNS_FACEBOOK_INIT
        // Device services: motion sampling, ad-hoc power/config. LOCATION_INIT moved
        // out of this group when SceLibLocation got a real implementation - it now has
        // a dedicated arm above, alongside the rest of its family.
        // Motion sampling: the enable is a statement, not a query, and this engine's
        // motion state is the same at-rest pose whether sampling is on or not.
        // MOTION_MAGNETOMETER_ON/OFF moved OUT of this group when
        // `sceMotionGetMagnetometerState` arrived: once something reads the bit back, an
        // accepted no-op stops being harmless and starts being a contradiction.
        | sv_nid::MOTION_START_SAMPLING
        | sv_nid::MOTION_STOP_SAMPLING
        | sv_nid::POWER_SET_CONFIGURATION_MODE
        // Shared dialog config accepted for every family.
        | sv_nid::COMMON_DIALOG_SET_CONFIG_PARAM
        // SceLiveArea: the app's home-screen tile. No home screen exists off-console,
        // so a frame update is an accepted no-op (the async variant has no completion
        // to deliver - there is no LiveArea state that changes).
        | sv_nid::LIVE_AREA_UPDATE_FRAME_ASYNC
        // Unnamed exports absent from every vita-headers revision, serviced as an
        // offline no-op success so they are handled rather than left as gaps.
        | sv_nid::NEAR_UTIL_UNKNOWN_A412E9CA
        | lk_nid::UNKNOWN_023EAA62 => cont!(ctx.ret(0)),

        // --- SceCommonDialog: system dialogs complete instantly offline ---------
        // The NP profile card and the photo picker. Both are pure UI over something
        // this host does not have - an account and a network for one, a photo library
        // for the other - so both open, complete immediately having shown nothing, and
        // report a result of "nothing chosen". That is the same shape every other family
        // here takes, and it is a path the title already handles: a user can always
        // dismiss either dialog without picking anything.
        sv_nid::NP_PROFILE_DIALOG_INIT => {
            cont!(services::dialog_init(ctx, st, services::DialogFamily::NpProfile))
        }
        sv_nid::NP_PROFILE_DIALOG_GET_STATUS => {
            cont!(services::dialog_get_status(ctx, st, services::DialogFamily::NpProfile))
        }
        sv_nid::NP_PROFILE_DIALOG_TERM => {
            cont!(services::dialog_term(ctx, st, services::DialogFamily::NpProfile))
        }
        sv_nid::PHOTO_IMPORT_DIALOG_INIT => {
            cont!(services::dialog_init(ctx, st, services::DialogFamily::PhotoImport))
        }
        sv_nid::PHOTO_IMPORT_DIALOG_GET_STATUS => {
            cont!(services::dialog_get_status(ctx, st, services::DialogFamily::PhotoImport))
        }
        sv_nid::PHOTO_IMPORT_DIALOG_TERM => {
            cont!(services::dialog_term(ctx, st, services::DialogFamily::PhotoImport))
        }
        // Their result reads, and the trophy-setup one, all write a zeroed result -
        // which for each of these families is "completed, nothing selected".
        sv_nid::NP_PROFILE_DIALOG_GET_RESULT | sv_nid::PHOTO_IMPORT_DIALOG_GET_RESULT => {
            cont!(services::dialog_ok(ctx, st))
        }
        // Aborting a dialog that has already completed, and closing one the title put
        // up itself, both genuinely succeed: there is nothing left running to stop.
        sv_nid::NP_PROFILE_DIALOG_ABORT | sv_nid::MSG_DIALOG_CLOSE => cont!(ctx.ret(0)),
        sv_nid::MSG_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::Msg)),
        sv_nid::MSG_DIALOG_GET_STATUS => cont!(services::dialog_get_status(ctx, st, services::DialogFamily::Msg)),
        sv_nid::MSG_DIALOG_TERM => cont!(services::dialog_term(ctx, st, services::DialogFamily::Msg)),
        sv_nid::NET_CHECK_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::NetCheck)),
        sv_nid::NET_CHECK_DIALOG_GET_STATUS => cont!(services::dialog_get_status(ctx, st, services::DialogFamily::NetCheck)),
        sv_nid::NET_CHECK_DIALOG_TERM => cont!(services::dialog_term(ctx, st, services::DialogFamily::NetCheck)),
        sv_nid::SAVEDATA_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::SaveData)),
        sv_nid::SAVEDATA_DIALOG_GET_STATUS | sv_nid::SAVEDATA_DIALOG_GET_SUB_STATUS => {
            cont!(services::dialog_get_status(ctx, st, services::DialogFamily::SaveData))
        }
        sv_nid::SAVEDATA_DIALOG_TERM => cont!(services::dialog_term(ctx, st, services::DialogFamily::SaveData)),
        sv_nid::NP_MESSAGE_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::NpMessage)),
        sv_nid::NP_MESSAGE_DIALOG_GET_STATUS => cont!(services::dialog_get_status(ctx, st, services::DialogFamily::NpMessage)),
        sv_nid::NP_MESSAGE_DIALOG_TERM | sv_nid::NP_MESSAGE_DIALOG_ABORT => {
            cont!(services::dialog_term(ctx, st, services::DialogFamily::NpMessage))
        }
        sv_nid::NP_TROPHY_SETUP_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::NpTrophySetup)),
        sv_nid::NP_TROPHY_SETUP_DIALOG_GET_STATUS => cont!(services::dialog_get_status(ctx, st, services::DialogFamily::NpTrophySetup)),
        sv_nid::NP_TROPHY_SETUP_DIALOG_TERM => cont!(services::dialog_term(ctx, st, services::DialogFamily::NpTrophySetup)),
        sv_nid::STORE_CHECKOUT_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::StoreCheckout)),
        sv_nid::STORE_CHECKOUT_DIALOG_GET_STATUS => cont!(services::dialog_get_status(ctx, st, services::DialogFamily::StoreCheckout)),
        sv_nid::STORE_CHECKOUT_DIALOG_TERM => cont!(services::dialog_term(ctx, st, services::DialogFamily::StoreCheckout)),
        sv_nid::NP_SNS_FACEBOOK_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::NpSnsFacebook)),
        sv_nid::NP_SNS_FACEBOOK_DIALOG_GET_STATUS => cont!(services::dialog_get_status(ctx, st, services::DialogFamily::NpSnsFacebook)),
        sv_nid::IME_DIALOG_INIT => cont!(services::dialog_init(ctx, st, services::DialogFamily::Ime)),
        sv_nid::IME_DIALOG_GET_STATUS => cont!(services::dialog_get_status(ctx, st, services::DialogFamily::Ime)),
        sv_nid::IME_DIALOG_TERM => cont!(services::dialog_term(ctx, st, services::DialogFamily::Ime)),
        // An abort closes the dialog: it must stop reporting FINISHED, or a title that
        // aborts and re-polls sees a completion it cancelled.
        sv_nid::IME_DIALOG_ABORT => cont!(services::dialog_term(ctx, st, services::DialogFamily::Ime)),
        sv_nid::MSG_DIALOG_ABORT => cont!(services::dialog_term(ctx, st, services::DialogFamily::Msg)),
        // The text-entry dialog is the one family whose result is not "zeroed reads as
        // OK": it reports the CLOSE button, because off-console nobody typed anything.
        sv_nid::IME_DIALOG_GET_RESULT => cont!(services::ime_dialog_get_result(ctx, st)),
        // Result reads and per-frame pumping succeed with the caller's (zeroed)
        // result struct untouched; the update pump has no system UI to animate.
        sv_nid::MSG_DIALOG_GET_RESULT => cont!(services::msg_dialog_get_result(ctx, st)),
        sv_nid::COMMON_DIALOG_UPDATE
        | sv_nid::NET_CHECK_DIALOG_GET_RESULT
        | sv_nid::SAVEDATA_DIALOG_GET_RESULT
        | sv_nid::SAVEDATA_DIALOG_CONTINUE
        | sv_nid::SAVEDATA_DIALOG_FINISH
        | sv_nid::SAVEDATA_DIALOG_SUB_CLOSE
        | sv_nid::NP_MESSAGE_DIALOG_GET_RESULT
        | sv_nid::STORE_CHECKOUT_DIALOG_GET_RESULT
        | sv_nid::NP_SNS_FACEBOOK_DIALOG_GET_RESULT_LONG_TOKEN => {
            cont!(services::dialog_ok(ctx, st))
        }

        _ => {
            // No handler for this NID. Do NOT fake a success: a silent `ret(0)` lets
            // the guest continue on a false premise and desync into a spin or memory
            // corruption far from here (the exact failure mode this project keeps
            // hitting). Record it for the report and stop the run loudly, naming the
            // call so the fix is "implement this NID", pinpointed. Every legitimate
            // offline no-op has its own explicit arm above returning 0 deliberately;
            // reaching here means the NID is genuinely unhandled.
            st.capture.note_unimplemented(library_nid, func_nid, nid::name(func_nid));
            let name = nid::name(func_nid);
            // Report the CALL, not just its name. For a library with no published
            // prototype the only source for the signature is the title's own use of
            // it, and this is the one moment that evidence exists: r0-r3 as the guest
            // passed them, the first stack words in case the call takes more than four
            // arguments, and a short dump of whatever each pointer-looking argument
            // points at (a work-area size, a struct's leading fields). Cheap - it runs
            // once, on the way to stopping the run.
            let arg_dump = describe_call_args(ctx);
            // ...and WHO called it. `lr` names only the immediate caller, which for a
            // library reached through a table of one-line thunks is always the thunk -
            // the useful frame is the one above it. See [`guest_return_trail`].
            let trail = guest_return_trail(ctx, 96);
            return SvcOutcome::Fatal(format!(
                "unimplemented NID {name} (lib={library_nid:#010x} nid={func_nid:#010x})                  called by thread {:#x}; implement it (no silent stub)
{arg_dump}                   return trail: {trail}",
                st.current_thread(),
            ));
        }
    };
    // Diagnostic (`RUST_LOG=vitaslop::err=debug`): log any handler that returns an
    // SCE error code (top bit set) - the fastest way to find an HLE call whose
    // failure sends the guest down an unexpected (error/cleanup) path.
    let r = ctx.regs[0];
    if r & 0x8000_0000 != 0 {
        tracing::debug!(
            target: "vitaslop::err",
            thid = st.current_thread(),
            name = nid::name(func_nid),
            nid = format_args!("{func_nid:#010x}"),
            ret = format_args!("{r:#010x}"),
            "error return"
        );
    }
    outcome
}

#[cfg(test)]
mod frame_boundary_tests {
    use super::*;
    use crate::host::VitaState;
    use crate::world::DeterministicWorld;
    use crate::{SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::REG_COUNT;

    /// Dispatch one NID with the given r0..r3 and report the outcome's kind. The
    /// distinction under test is coarse (does this count a display frame or not), so
    /// the outcome is reduced to a name rather than compared structurally.
    fn outcome_of(nid_lib: u32, nid_fn: u32, args: [u32; 4]) -> &'static str {
        let mut regs = [0u32; REG_COUNT];
        regs[..4].copy_from_slice(&args);
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = vec![0u8; 4096];
        let mut st = VitaState::new(0, 4096, Box::new(DeterministicWorld::default()));
        // The frame/yield distinction only exists under the preemptive scheduler; the
        // single-worker model has nothing to yield to and returns Continue throughout.
        st.set_preemptive(true);
        let mut mem = SliceMemory(&mut bytes);
        let mut ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        match dispatch(nid_lib, nid_fn, &mut ctx, &mut st) {
            SvcOutcome::Flip => "Flip",
            SvcOutcome::Reschedule => "Reschedule",
            SvcOutcome::Block => "Block",
            SvcOutcome::Continue => "Continue",
            SvcOutcome::Halt => "Halt",
            SvcOutcome::ThreadExit => "ThreadExit",
            SvcOutcome::Fatal(m) => panic!("dispatch refused the call: {m}"),
        }
    }

    /// >>> EVERY NID [`fast_nid`] ROUTES THROUGH THE NON-SUSPENDING TRAP REALLY CANNOT SUSPEND.
    ///
    /// The browser binds that trap to a plain function: a handler that returned anything but
    /// `Continue` there would end the run. So every name on the list is dispatched here with
    /// zeroed registers and required to CONTINUE - the arm shape (`cont!`) is what admits it,
    /// and this is that shape asserted at the one place a grown parking path would show.
    /// The list is also required to be DISJOINT from the inline forms: a NID with an inline
    /// lowering never reaches either trap, so naming it fast would be a lie the link step
    /// silently drops.
    ///
    /// The blocking primitives the race also makes - a cond wait, a plain lock - are held to
    /// the opposite: not fast, so a copy-paste that widened the list would fail here.
    #[test]
    fn the_fast_nids_only_continue() {
        let fast = [
            gxm_nid::DRAW,
            gxm_nid::DRAW_PRECOMPUTED,
            gxm_nid::BEGIN_SCENE,
            gxm_nid::END_SCENE,
            gxm_nid::SET_VISIBILITY_BUFFER,
            gxm_nid::COLOR_SURFACE_GET_DATA,
            gxm_nid::COLOR_SURFACE_GET_STRIDE_IN_PIXELS,
            gxm_nid::PAD_HEARTBEAT,
            lw_nid::SIGNAL_LW_COND,
            lw_nid::TRY_LOCK_LW_MUTEX,
            sync_nid::UNLOCK_MUTEX,
            sync_nid::SIGNAL_COND,
            lk_nid::CLIB_MSPACE_MALLOC,
            lk_nid::CLIB_MSPACE_MEMALIGN,
            lk_nid::CLIB_MSPACE_FREE,
            lk_nid::GET_TLS_ADDR,
            ngs_nid::VOICE_GET_STATE_DATA,
            ngs_nid::SYSTEM_UPDATE,
            ngs_nid::VOICE_SET_PARAMS_BLOCK,
            pm_nid::POWER_TICK,
            sv_nid::APP_MGR_GET_APP_STATE,
            sv_nid::SYSTEM_GESTURE_UPDATE_TOUCH_RECOGNIZER,
            sv_nid::SYSTEM_GESTURE_GET_TOUCH_EVENTS_COUNT,
            sv_nid::TOUCH_READ,
        ];
        for nid_fn in fast {
            let name = nid::name(nid_fn);
            assert!(fast_nid(nid_fn), "{name} is on the fast list but fast_nid() refuses it");
            assert!(
                inline_op(nid_fn).is_none(),
                "{name} has an inline form, so routing it through a trap is unreachable"
            );
            assert_eq!(
                outcome_of(0, nid_fn, [0, 0, 0, 0]),
                "Continue",
                "{name} is routed through the non-suspending trap but did not CONTINUE"
            );
        }
        for nid_fn in [lw_nid::WAIT_LW_COND, lw_nid::LOCK_LW_MUTEX, tm_nid::DELAY_THREAD] {
            assert!(!fast_nid(nid_fn), "{} can park and must not be fast", nid::name(nid_fn));
        }
    }

    /// >>> EVERY NID [`stub_inline_op`] INLINES REALLY IS A BARE CONSTANT RETURN.
    ///
    /// The inline form emits `r0 = 0` and never reaches the host, so anything else its
    /// handler did simply stops happening - silently, and only in a build that inlines. This
    /// is the admissibility test written down: dispatch each one with poisoned registers over
    /// poisoned memory and require that it CONTINUES, answers 0, and leaves guest memory
    /// exactly as it found it. A stub that grows a body later fails here, in the same commit
    /// that gives it one, rather than being quietly skipped in the browser.
    #[test]
    fn the_inlined_stubs_are_stubs() {
        for nid_fn in [
            ngs_nid::SYSTEM_SET_FLAGS,
            ngs_nid::SYSTEM_RELEASE,
            ngs_nid::RACK_RELEASE,
            ngs_nid::VOICE_RESUME,
            ngs_nid::VOICE_BYPASS_MODULE,
            ngs_nid::VOICE_GET_PARAMS_OUT_OF_RANGE,
            ngs_nid::VOICE_PATCH_SET_VOLUMES_MATRIX,
            ngs_nid::VOICE_PATCH_SET_VOLUME,
            ngs_nid::PATCH_GET_INFO,
            ngs_nid::PATCH_REMOVE_ROUTING,
            ngs_nid::SYSTEM_LOCK,
            ngs_nid::SYSTEM_UNLOCK,
            ngs_nid::AT9_GET_SECTION_DETAILS,
        ] {
            assert!(
                super::stub_inline_op(nid_fn).is_some(),
                "{} is in the test's list but not in stub_inline_op's",
                crate::nid::name(nid_fn),
            );
            let (outcome, r0, wrote) = stub_probe(crate::nid::lib::SCE_NGS, nid_fn);
            assert_eq!(outcome, "Continue", "{} suspended", crate::nid::name(nid_fn));
            assert_eq!(r0, 0, "{} answered something", crate::nid::name(nid_fn));
            assert!(!wrote, "{} wrote guest memory", crate::nid::name(nid_fn));
        }
        let (outcome, r0, wrote) =
            stub_probe(crate::nid::lib::SCE_CTRL, ctrl_nid::SET_SAMPLING_MODE);
        assert_eq!((outcome, r0, wrote), ("Continue", 0, false));
        // `sceKernelSetGPO` is the VOID one: it leaves r0 exactly as the guest passed it,
        // which is why it is inlined as a `Nop` rather than as a constant return.
        let (outcome, r0, wrote) = stub_probe(crate::nid::lib::SCE_SYSMEM, sm_nid::SET_GPO);
        assert_eq!((outcome, r0, wrote), ("Continue", 0x40, false));
        assert!(matches!(
            super::stub_inline_op(sm_nid::SET_GPO),
            None | Some(vitaslop_transpiler::InlineOp::Nop)
        ));
    }

    /// Dispatch one NID with POISONED r0..r3 over POISONED memory, and report
    /// `(outcome, r0, whether any guest byte changed)`. The poison is what makes the last
    /// two mean anything: a handler that leaves r0 alone would pass an `r0 == 0` check on a
    /// zeroed register file for the wrong reason.
    fn stub_probe(nid_lib: u32, nid_fn: u32) -> (&'static str, u32, bool) {
        let mut regs = [0xDEAD_BEEFu32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        // A pointer argument has to be IN RANGE, or a handler that writes through it would
        // decline for that reason rather than because it writes nothing.
        regs[..4].copy_from_slice(&[0x40, 0x80, 0xC0, 0x100]);
        let before = vec![0xA5u8; 4096];
        let mut bytes = before.clone();
        let mut st = VitaState::new(0, 4096, Box::new(DeterministicWorld::default()));
        st.set_preemptive(true);
        let mut mem = SliceMemory(&mut bytes);
        let mut ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        let outcome = match dispatch(nid_lib, nid_fn, &mut ctx, &mut st) {
            SvcOutcome::Continue => "Continue",
            SvcOutcome::Flip => "Flip",
            SvcOutcome::Reschedule => "Reschedule",
            SvcOutcome::Block => "Block",
            SvcOutcome::Halt => "Halt",
            SvcOutcome::ThreadExit => "ThreadExit",
            SvcOutcome::Fatal(m) => panic!("dispatch refused the call: {m}"),
        };
        (outcome, regs[0], bytes != before)
    }

    /// A voluntary yield is NOT a display frame. `sceKernelDelayThread(0)` and
    /// `sceDisplayWaitVblankStartMulti(0)` both mean "give someone else the CPU"
    /// and a worker can spin on either thousands of times between two rendered
    /// frames. Counting them as frames inflates the frame clock by an arbitrary,
    /// title-dependent factor, which silently desynchronises every frame-keyed
    /// input script from the game it is driving (a recipe's button press lands in a
    /// fraction of a game frame and the guest never samples it) and makes every
    /// per-frame timing figure meaningless. Only `sceGxmDisplayQueueAddEntry` -
    /// the guest handing a finished frame to scanout - ends a frame.
    #[test]
    fn only_a_display_queue_entry_counts_as_a_frame_boundary() {
        assert_eq!(
            outcome_of(nid::lib::SCE_THREADMGR, tm_nid::DELAY_THREAD, [0, 0, 0, 0]),
            "Reschedule",
            "delayThread(0) is a plain yield, not a frame"
        );
        assert_eq!(
            outcome_of(nid::lib::SCE_DISPLAY_USER, display_nid::WAIT_VBLANK_START_MULTI, [0, 0, 0, 0]),
            "Reschedule",
            "waitVblankStartMulti(0) asks for no wait at all, so it is not a frame"
        );
        // A real (nonzero) wait still parks the thread on the virtual clock.
        assert_eq!(
            outcome_of(nid::lib::SCE_THREADMGR, tm_nid::DELAY_THREAD, [1000, 0, 0, 0]),
            "Block",
            "a real delay parks the caller"
        );
        assert_eq!(
            outcome_of(nid::lib::SCE_GXM, gxm_nid::DISPLAY_QUEUE_ADD_ENTRY, [0, 0, 0, 0]),
            "Flip",
            "queueing a finished frame for scanout IS the frame boundary"
        );
    }
}
