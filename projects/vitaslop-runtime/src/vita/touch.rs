//! SceTouch: the front and back touch panels.
//!
//! Touch enters through the [`World`](crate::world::World) seam, the same
//! determinism boundary the pad uses ([`World::poll_touch`]). A read reports a
//! well-formed sample carrying whatever points the world has down this frame, or a
//! sample with *no* active touch (`reportNum = 0`) when nothing is pressed. That
//! empty-but-valid sample is deliberately not the old zero-return stub:
//! `sceTouchRead`/`sceTouchPeek` return the number of buffers filled (between 1 and
//! `nBufs`), and a title's per-frame touch update can read a zero return as an error
//! (one observed title logs `touchUpdate::351 ret=0` every frame and wedges its input
//! pipeline on it). Reporting one valid sample keeps that update on its normal path.

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;

/// Size of one `SceTouchData`: timeStamp(8) + status(4) + reportNum(4) +
/// report[8] * sizeof(SceTouchReport=0x10) = 0x90 bytes.
const TOUCH_DATA_SIZE: usize = 0x90;

/// Byte offset of `SceTouchReport[i]` within a `SceTouchData` (past the 16-byte
/// timeStamp/status/reportNum header).
const REPORT_BASE: usize = 16;
/// `sizeof(SceTouchReport)`: id(1) force(1) x(2) y(2) reserved[8] info(2).
const REPORT_SIZE: usize = 0x10;

/// Upper bound on how many history samples we will materialise for one call, so a
/// bogus `nBufs` cannot ask for an unbounded write. The real panel keeps 64 internal
/// samples; a title asks for a handful.
const MAX_BUFS: u32 = 64;

/// Fill `nbufs` back-to-back `SceTouchData` samples at `data` from the world's
/// current touch frame for `port`, each stamped with the current time, and return
/// the count written (at least 1) - the "buffers count" both `sceTouchRead` and
/// `sceTouchPeek` are documented to return.
fn fill_touch(ctx: &mut GuestCtx, st: &mut VitaState, port: u32, data: u32, nbufs: u32) -> i32 {
    let n = nbufs.clamp(1, MAX_BUFS);
    let ts = st.world.monotonic_us();
    let touch = st.world.poll_touch(port);
    let points = touch.active();
    // The panel reports at most report[8]; a world should never exceed that, but clamp
    // defensively so a bad frame cannot overrun the fixed report array.
    let report_count = points.len().min(8);
    if data != 0 {
        for i in 0..n {
            let mut buf = [0u8; TOUCH_DATA_SIZE];
            buf[0..8].copy_from_slice(&ts.to_le_bytes());
            buf[12..16].copy_from_slice(&(report_count as u32).to_le_bytes());
            for (j, p) in points.iter().take(report_count).enumerate() {
                let off = REPORT_BASE + j * REPORT_SIZE;
                buf[off] = p.id;
                buf[off + 1] = p.force;
                buf[off + 2..off + 4].copy_from_slice(&p.x.to_le_bytes());
                buf[off + 4..off + 6].copy_from_slice(&p.y.to_le_bytes());
            }
            ctx.write_bytes(data + i * TOUCH_DATA_SIZE as u32, &buf);
        }
    }
    n as i32
}

/// int sceTouchRead(SceUInt32 port, SceTouchData *pData, SceUInt32 nBufs)
/// The blocking read; served without parking, like the pad read (the display flip
/// is the frame-pacing yield).
#[hostcall]
pub(super) fn read(ctx: &mut GuestCtx, st: &mut VitaState, _port: u32, data: Ptr, nbufs: u32) -> i32 {
    fill_touch(ctx, st, _port, data.addr(), nbufs)
}

/// int sceTouchPeek(SceUInt32 port, SceTouchData *pData, SceUInt32 nBufs)
/// The non-blocking poll; same "no touch" sample.
#[hostcall]
pub(super) fn peek(ctx: &mut GuestCtx, st: &mut VitaState, _port: u32, data: Ptr, nbufs: u32) -> i32 {
    fill_touch(ctx, st, _port, data.addr(), nbufs)
}
