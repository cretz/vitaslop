//! SceDisplayUser: the scanout. The cube reaches this only through the display
//! queue callback (deferred), but a direct call is also supported: it records the
//! presented framebuffer address.

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::SvcOutcome;

/// The virtual duration of one display vblank interval at 60 Hz, in microseconds.
const VBLANK_US: u64 = 1_000_000 / 60;

/// int sceDisplayWaitVblankStartMulti(unsigned int vcount)
///
/// Block the caller until `vcount` vblank periods have elapsed. Preemptive: a REAL
/// timed park until the virtual clock reaches `now + vcount * (1/60 s)` - the same
/// mechanism as `sceKernelDelayThread`, so a frame-pacing loop that waits on vblank
/// yields the CPU to the threads doing work instead of busy-spinning. A `vcount` of
/// 0 is a plain yield - NOT a frame boundary (see [`SvcOutcome::Flip`]); it asks for
/// no wait at all, so a loop doing it spins as fast as the scheduler allows and must
/// not be allowed to advance the display frame count. Single-thread model: nothing to
/// yield to, so it just succeeds (the clock is host-driven). Returns 0.
pub(super) fn wait_vblank_start_multi(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let vcount = ctx.arg(0);
    ctx.ret(0);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    if vcount == 0 {
        return SvcOutcome::Reschedule;
    }
    st.sleep_park(vcount as u64 * VBLANK_US);
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
    st.sleep_park(VBLANK_US);
    SvcOutcome::Block
}

/// int sceDisplayWaitSetFrameBuf(void)
///
/// Block until the framebuffer queued by the most recent `sceDisplaySetFrameBuf` has
/// been latched, which hardware does at the next vblank. We apply a set-framebuffer
/// immediately, so the latch is a single vblank away: park for one vblank period
/// (preemptive) so a present-then-wait loop paces to 60 Hz and yields the CPU, or a
/// plain yield in the single-thread model. Returns 0.
pub(super) fn wait_set_frame_buf(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    ctx.ret(0);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    st.sleep_park(VBLANK_US);
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
    use vitaslop_transpiler::InlineOp::LoadMirror;
    match func_nid {
        crate::nid::display::GET_VCOUNT => {
            Some(LoadMirror { slot: crate::vita::mirror::SLOT_VCOUNT })
        }
        _ => None,
    }
}

/// int sceDisplaySetFrameBuf(const SceDisplayFrameBuf *pParam, int sync)
/// SceDisplayFrameBuf: { SceSize size; void *base; uint32 pitch; uint32 fmt;
///                       uint32 width; uint32 height; } (0x18 bytes).
#[hostcall]
pub(super) fn set_frame_buf(ctx: &mut GuestCtx, st: &mut VitaState, param: Ptr, _sync: i32) -> i32 {
    let base = ctx.read_u32(param.addr() + 4);
    if base != 0 {
        st.present(base);
    }
    0
}
