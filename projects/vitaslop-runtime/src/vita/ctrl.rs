//! SceCtrl: controller input. The one place the cube reads a non-deterministic
//! external input, so it goes through the World seam (`poll_ctrl`), which a
//! record or replay wrapper can capture.

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::nid::ctrl as nid;
use crate::SvcOutcome;

/// Bytes of SceCtrlData we populate: timeStamp (8) + buttons (4) + the four
/// analog bytes (4). The real struct is larger but the guest reads these fields;
/// we zero this prefix and fill it.
const CTRL_DATA_PREFIX: usize = 16;

pub fn try_dispatch(func_nid: u32, ctx: &mut GuestCtx, st: &mut VitaState) -> Option<SvcOutcome> {
    match func_nid {
        nid::PEEK_BUFFER_POSITIVE => peek_buffer_positive(ctx, st),
        _ => return None,
    }
    Some(SvcOutcome::Continue)
}

/// int sceCtrlPeekBufferPositive(int port, SceCtrlData *pad_data, int count)
#[hostcall]
fn peek_buffer_positive(ctx: &mut GuestCtx, st: &mut VitaState, port: u32, data: Ptr, _count: i32) -> i32 {
    let frame = st.world.poll_ctrl(port);
    let ts = st.world.monotonic_us();

    // SceCtrlData stride is larger than the prefix we write; but each entry's
    // read fields all sit in the prefix, so writing one filled prefix per entry
    // (zeroing the rest is unnecessary as the guest only reads these fields) is
    // faithful for polling. We conservatively write only entry 0's fields, which
    // is what a single-buffer peek reads.
    let mut buf = [0u8; CTRL_DATA_PREFIX];
    buf[0..8].copy_from_slice(&ts.to_le_bytes());
    buf[8..12].copy_from_slice(&frame.buttons.to_le_bytes());
    buf[12] = frame.lx;
    buf[13] = frame.ly;
    buf[14] = frame.rx;
    buf[15] = frame.ry;
    ctx.write_bytes(data.addr(), &buf);

    // Return the number of buffers filled.
    1
}
