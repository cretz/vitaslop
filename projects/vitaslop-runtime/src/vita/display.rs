//! SceDisplayUser: the scanout. The cube reaches this only through the display
//! queue callback (deferred), but a direct call is also supported: it records the
//! presented framebuffer address.

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::SvcOutcome;

/// The virtual duration of one display vblank interval at 60 Hz, in microseconds.
pub(crate) const VBLANK_US: u64 = 1_000_000 / 60;

/// int sceDisplayWaitVblankStartMulti(unsigned int vcount)
///
/// Block the caller until the `vcount`th vblank EDGE from now. Preemptive: a REAL
/// timed park until the virtual clock reaches that edge - the same mechanism as
/// `sceKernelDelayThread`, so a frame-pacing loop that waits on vblank yields the CPU
/// to the threads doing work instead of busy-spinning. A `vcount` of
/// 0 is a plain yield - NOT a frame boundary (see [`SvcOutcome::Flip`]); it asks for
/// no wait at all, so a loop doing it spins as fast as the scheduler allows and must
/// not be allowed to advance the display frame count. Single-thread model: nothing to
/// yield to, so it just succeeds (the clock is host-driven). Returns 0.
///
/// **An EDGE, not a duration.** The scanout's vblanks are a free-running 60 Hz
/// heartbeat that this call joins; it does not start a stopwatch. Parking a whole
/// period from wherever the guest happened to call over-waits by half a period on
/// average and never phase-locks - see [`VitaState::vblank_park`].
pub(super) fn wait_vblank_start_multi(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let vcount = ctx.arg(0);
    ctx.ret(0);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    if vcount == 0 {
        return SvcOutcome::Reschedule;
    }
    st.vblank_park(vcount as u64, VBLANK_US);
    SvcOutcome::Block
}

/// int sceDisplayWaitVblankStart(void)
///
/// Wait for exactly one vblank - the no-argument form of
/// [`wait_vblank_start_multi`], and identical to it with a `vcount` of 1. Kept as a
/// separate entry point rather than folded in because it is a DIFFERENT NID and a title
/// that links only this one must not hard-fail; the behaviour is shared so the two cannot
/// diverge. Note this is a wait, NOT a frame boundary: only the display-queue flip counts
/// a frame (see [`SvcOutcome::Flip`]), and letting a vblank wait advance the frame count
/// runs the clock at the rate of the pacing loop rather than the rate of presentation.
pub(super) fn wait_vblank_start(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    ctx.ret(0);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    st.vblank_park(1, VBLANK_US);
    SvcOutcome::Block
}

/// int sceDisplayWaitSetFrameBuf(void)
///
/// Block until the framebuffer queued by the most recent `sceDisplaySetFrameBuf` has
/// been latched, which hardware does at the next vblank. So the wait is until the next
/// vblank EDGE (preemptive), which paces a present-then-wait loop to 60 Hz and yields
/// the CPU, or a plain yield in the single-thread model.
///
/// **This waits whatever `sync` the buffer was set with**, and the attempt to make
/// IMMEDIATE return at once - on the reasoning that a buffer already applied has nothing
/// left to wait for - LIVELOCKED the one retail title that asks for it: it spins on this
/// call, and the run reached frame 3 with 34.3 million thread resumes. What this call
/// waits for is the DISPLAY updating, which is a vblank event either way; `sync` only
/// decides whether the pointer changes mid-scan. See [`VitaState::set_display_sync`].
/// Returns 0.
pub(super) fn wait_set_frame_buf(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    ctx.ret(0);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    st.vblank_park(1, VBLANK_US);
    SvcOutcome::Block
}

/// int sceDisplayGetVcount(void)
///
/// The scanout's vertical-blank counter: how many vblanks the display has produced
/// since boot, free-running at 60 Hz whether or not the guest presents. A title uses
/// it to measure elapsed display time and to detect dropped frames (comparing the
/// vcount delta across its own frame against 1), so it must be derived from the CLOCK
/// rather than from the frame count - counting presented frames would report exactly
/// one vblank per frame no matter how long the frame took, which is precisely the
/// dropped-frame signal being asked for, inverted.
///
/// This NID is also served INLINE, out of a host-mirror word, because a vblank spin
/// calls it tens of thousands of times a frame; [`vcount`] is the one definition both
/// forms compute, so they cannot drift.
#[hostcall]
pub(super) fn get_vcount(st: &mut VitaState) -> u32 {
    vcount(st)
}

/// The vblank counter the display has reached: the virtual clock in units of one
/// vblank period.
///
/// A pure function of the game clock, which is exactly what lets the inline form
/// exist - the clock changes only at a scheduler quantum boundary, a display flip or
/// the idle path, none of which can happen while guest code is running.
pub(crate) fn vcount(st: &VitaState) -> u32 {
    (st.now_us() / VBLANK_US) as u32
}

/// The inline form of a SceDisplay host import, or `None` for the ones with real
/// behaviour. Reached through [`crate::vita::inline_op`], which owns the on/off switch.
///
/// Only the vblank counter qualifies, and it qualifies because it is a pure function of
/// the game clock - see [`crate::vita::mirror`] for the rule and why the two waits
/// below can never join it (they block, which is behaviour, not a read).
pub(crate) fn inline_op(func_nid: u32) -> Option<vitaslop_transpiler::InlineOp> {
    use vitaslop_transpiler::InlineOp::{LoadMirror, LoadMirrorParking};
    match func_nid {
        // >>> THE READ CARRIES A SPIN GUARD, AND THAT IS THE DEFAULT.
        //
        // The ordinary vblank wait is `do { v = sceDisplayGetVcount(); } while (v == last);`
        // and the mirror made each turn of it one `i32.load` - which removed the host calls
        // and left the SPIN. It ends only when the clock reaches the next vblank, and the
        // only thing advancing the clock is the spin's own fuel, so the emulator executes
        // however much guest code the clock model says fits in the rest of the frame.
        // MEASURED in the browser on a retail racer's race: that single function is **26% of
        // all translated guest code**, 1.6 ms of a 16.7 ms frame, the largest guest function
        // in the profile by a factor of four.
        //
        // The guard parks the thread on the next vblank instead - the same wait
        // `sceDisplayWaitVblankStart` performs, reached from the loop that is asking for it.
        // `VITASLOP_VBLANK_PARK=0` is the A/B arm and restores the bare read.
        crate::nid::display::GET_VCOUNT => Some(if vblank_park_spin() {
            LoadMirrorParking {
                slot: crate::vita::mirror::SLOT_VCOUNT,
                budget: crate::vita::mirror::SLOT_SPIN_BUDGET,
            }
        } else {
            LoadMirror { slot: crate::vita::mirror::SLOT_VCOUNT }
        }),
        _ => None,
    }
}

/// Whether an inlined `sceDisplayGetVcount` carries the spin guard (`VITASLOP_VBLANK_PARK`,
/// default ON; `=0` is the A/B arm that restores the bare mirror read).
///
/// VALUE-sensitive, and read through the knob seam rather than the environment because the
/// browser is where the spin costs the most and the browser has no environment.
fn vblank_park_spin() -> bool {
    !matches!(crate::knobs::var("VITASLOP_VBLANK_PARK").as_deref(), Ok("0"))
}

/// int sceDisplaySetFrameBuf(const SceDisplayFrameBuf *pParam, int sync)
/// SceDisplayFrameBuf: { SceSize size; void *base; uint32 pitch; uint32 fmt;
///                       uint32 width; uint32 height; } (0x18 bytes).
///
/// `sync` is a `SceDisplaySetBufSync`: `SCE_DISPLAY_SETBUF_NEXTFRAME` (1) asks for the
/// buffer change to take effect at the next vblank, `SCE_DISPLAY_SETBUF_IMMEDIATE` (0)
/// for it to take effect at once - which tears, and does not pace. That is the whole of
/// the argument: it is NOT a swap interval and cannot ask for a 2-vblank one (the enum
/// has exactly these two values). A title runs at 30 Hz either by taking longer than a
/// vblank to draw, which [`VitaState::pace_flip`]'s grid now models, or by waiting two
/// vblanks itself, which [`wait_vblank_start_multi`] models.
///
/// The value is recorded rather than acted on here because this call does not present:
/// under GXM it is the display-queue CALLBACK, and the flip it describes was submitted
/// by `sceGxmDisplayQueueAddEntry`, which is where the pacing lives.
#[hostcall]
pub(super) fn set_frame_buf(ctx: &mut GuestCtx, st: &mut VitaState, param: Ptr, sync: i32) -> i32 {
    st.set_display_sync(sync as u32);
    let base = ctx.read_u32(param.addr() + 4);
    if base != 0 {
        st.present(base);
    }
    0
}
