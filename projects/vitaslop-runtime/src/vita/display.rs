//! SceDisplayUser: the scanout. The cube reaches this only through the display
//! queue callback (deferred), but a direct call is also supported: it records the
//! presented framebuffer address.

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::nid::display as nid;
use crate::SvcOutcome;

pub fn try_dispatch(func_nid: u32, ctx: &mut GuestCtx, st: &mut VitaState) -> Option<SvcOutcome> {
    match func_nid {
        nid::SET_FRAME_BUF => set_frame_buf(ctx, st),
        _ => return None,
    }
    Some(SvcOutcome::Continue)
}

/// int sceDisplaySetFrameBuf(const SceDisplayFrameBuf *pParam, int sync)
/// SceDisplayFrameBuf: { SceSize size; void *base; uint32 pitch; uint32 fmt;
///                       uint32 width; uint32 height; } (0x18 bytes).
#[hostcall]
fn set_frame_buf(ctx: &mut GuestCtx, st: &mut VitaState, param: Ptr, _sync: i32) -> i32 {
    let base = ctx.read_u32(param.addr() + 4);
    if base != 0 {
        st.present(base);
    }
    0
}
