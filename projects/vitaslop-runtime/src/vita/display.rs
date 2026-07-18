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
/// 0 is a plain yield. Single-thread model: nothing to yield to, so it just succeeds
/// (the clock is host-driven). Returns 0.
pub(super) fn wait_vblank_start_multi(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let vcount = ctx.arg(0);
    ctx.ret(0);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    if vcount == 0 {
        return SvcOutcome::Yield;
    }
    st.sleep_park(vcount as u64 * VBLANK_US);
    SvcOutcome::Block
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
