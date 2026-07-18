//! SceCtrl: controller input. The one place a title reads a non-deterministic
//! external input, so it goes through the World seam (`poll_ctrl`), which a record
//! or replay wrapper - or a scripted TAS recipe - drives.
//!
//! Titles reach the pad through several entry points with the same `SceCtrlData`
//! payload: `Peek` returns the latest sample immediately, `Read` classically blocks
//! until the next vblank. We serve both from the current [`World`] frame; frame
//! pacing comes from the display flip, so a non-blocking `Read` still advances one
//! input frame per rendered frame. The `Positive`/`Positive2` variants differ only
//! in button remapping we do not model, so they share one filler.

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;

/// Bytes of SceCtrlData we populate: timeStamp (8) + buttons (4) + the four analog
/// bytes lx/ly/rx/ry (4). The real struct is larger but the guest reads these
/// fields; we fill this prefix and leave the rest zeroed.
const CTRL_DATA_PREFIX: usize = 16;

/// Fill one `SceCtrlData` at `data` from the current world frame for `port`, then
/// return the number of buffers reported (always 1 - a single latest sample). When
/// `negative` is set the button mask is inverted (the `*Negative` family reports a
/// pressed button as a cleared bit).
fn fill_ctrl(ctx: &mut GuestCtx, st: &mut VitaState, port: u32, data: u32, negative: bool) -> i32 {
    let frame = st.world.poll_ctrl(port);
    let ts = st.world.monotonic_us();
    let buttons = if negative { !frame.buttons } else { frame.buttons };

    // Diagnostic (`RUST_LOG=vitaslop::input=trace`): log the pad state the guest
    // reads and its caller, to see whether input reaches the code that should act on
    // it and what buttons/analog value it sees. Only non-neutral samples, to stay
    // readable.
    if frame.buttons != 0 || frame.lx != 128 || frame.ly != 128 {
        tracing::trace!(
            target: "vitaslop::input",
            port,
            buttons = format_args!("{:#06x}", frame.buttons),
            lx = frame.lx, ly = frame.ly, rx = frame.rx, ry = frame.ry,
            negative,
            lr = format_args!("{:#010x}", ctx.regs[14]),
            "ctrl"
        );
    }

    let mut buf = [0u8; CTRL_DATA_PREFIX];
    buf[0..8].copy_from_slice(&ts.to_le_bytes());
    buf[8..12].copy_from_slice(&buttons.to_le_bytes());
    buf[12] = frame.lx;
    buf[13] = frame.ly;
    buf[14] = frame.rx;
    buf[15] = frame.ry;
    if data != 0 {
        ctx.write_bytes(data, &buf);
    }
    1
}

/// int sceCtrlPeekBufferPositive(int port, SceCtrlData *pad_data, int count)
#[hostcall]
pub(super) fn peek_buffer_positive(ctx: &mut GuestCtx, st: &mut VitaState, port: u32, data: Ptr, _count: i32) -> i32 {
    fill_ctrl(ctx, st, port, data.addr(), false)
}

/// int sceCtrlReadBufferPositive(int port, SceCtrlData *pad_data, int count)
/// The classically-blocking read. We return the current sample without parking:
/// the render loop's display flip is the frame-pacing yield, so input still
/// advances one frame per rendered frame and other threads still interleave there.
#[hostcall]
pub(super) fn read_buffer_positive(ctx: &mut GuestCtx, st: &mut VitaState, port: u32, data: Ptr, _count: i32) -> i32 {
    fill_ctrl(ctx, st, port, data.addr(), false)
}

/// int sceCtrlPeekBufferNegative(int port, SceCtrlData *pad_data, int count)
#[hostcall]
pub(super) fn peek_buffer_negative(ctx: &mut GuestCtx, st: &mut VitaState, port: u32, data: Ptr, _count: i32) -> i32 {
    fill_ctrl(ctx, st, port, data.addr(), true)
}

/// int sceCtrlReadBufferNegative(int port, SceCtrlData *pad_data, int count)
#[hostcall]
pub(super) fn read_buffer_negative(ctx: &mut GuestCtx, st: &mut VitaState, port: u32, data: Ptr, _count: i32) -> i32 {
    fill_ctrl(ctx, st, port, data.addr(), true)
}
