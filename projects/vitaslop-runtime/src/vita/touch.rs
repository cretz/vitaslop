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

/// Size of one `SceTouchPanelInfo`: 8 SceInt16 area/display bounds + minForce/maxForce
/// (u8 each) + reserved[30] = 0x30 bytes.
const PANEL_INFO_SIZE: usize = 0x30;

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
    let ts = st.guest_mono_us();
    let touch = st.world.poll_touch(port);
    let points = touch.active();
    // Diagnostic (`RUST_LOG=vitaslop::input=trace`, `VITASLOP_LOG` in the browser): the
    // touch sample the guest was handed, at the moment it asked for it.
    //
    // The twin of the pad's trace in `ctrl.rs`, and the only way to tell "the scripted
    // tap never reached the guest" from "it reached the guest and the guest ignored it".
    // Those have nothing in common as bugs, and a front end that simply does not respond
    // looks exactly the same either way - which is how a browser run replaying the same
    // recipe as a working native run sat on one menu for fifty thousand frames.
    if tracing::enabled!(target: "vitaslop::input", tracing::Level::TRACE) {
        tracing::trace!(
            target: "vitaslop::input",
            "touch poll port {port}: {} point(s){} at t={ts}us",
            points.len(),
            points
                .iter()
                .map(|p| format!(" ({},{}) force {}", p.x, p.y, p.force))
                .collect::<String>(),
        );
    }
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

/// Front-panel port (SCE_TOUCH_PORT_FRONT). The back panel is port 1.
const PORT_FRONT: u32 = 0;

/// The inclusive maximum active-area Y for `port`. The panels differ only here: X runs
/// 0..1919 on both, the front panel is the full 0..1087 and the smaller back panel stops
/// at 889. Shared with [`super::gesture`], which needs the same extent to spell "the
/// whole panel" for a recognizer created without a rectangle.
pub(super) fn max_active_y(port: u32) -> i16 {
    if port == PORT_FRONT {
        1087
    } else {
        889
    }
}

/// The inclusive maximum active-area X, the same for both panels.
pub(super) const MAX_ACTIVE_X: i16 = 1919;

/// Fill one `SceTouchPanelInfo` (0x30 bytes) for `port`.
///
/// The active-area range we report is deliberately the SAME coordinate space our
/// `sceTouchRead`/`sceTouchPeek` samples emit (front 0..1919 x 0..1087, twice the
/// 960x544 screen; back 0..1919 x 0..889), so a title mapping a touch report through
/// `(coord - minAa) / (maxAa - minAa) * (maxDisp - minDisp)` lands exactly on the
/// screen pixel we intend. The display range is the physical 0..1919 x 0..1087 both
/// panels project onto. Reporting a zeroed struct (the un-implemented fall-through)
/// would give `maxAa - minAa == 0` and a divide-by-zero in that mapping, which is why
/// this is filled explicitly rather than left to the default `ret(0)`.
fn panel_info_bytes(port: u32) -> [u8; PANEL_INFO_SIZE] {
    // Active-area Y extent differs per panel; X and the display extent match.
    let max_aa_y: i16 = max_active_y(port);
    let mut buf = [0u8; PANEL_INFO_SIZE];
    let mut put = |off: usize, v: i16| buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
    put(0x00, 0); // minAaX
    put(0x02, 0); // minAaY
    put(0x04, MAX_ACTIVE_X); // maxAaX
    put(0x06, max_aa_y); // maxAaY
    put(0x08, 0); // minDispX
    put(0x0a, 0); // minDispY
    put(0x0c, 1919); // maxDispX
    put(0x0e, 1087); // maxDispY
    buf[0x10] = 0; // minForce
    buf[0x11] = 128; // maxForce (matches the report force we emit)
    buf
}

/// int sceTouchGetPanelInfo(SceUInt32 port, SceTouchPanelInfo *pPanelInfo)
/// Report the panel's active area / display extent / force range so the title can map
/// raw touch samples to screen coordinates. See [`panel_info_bytes`].
#[hostcall]
pub(super) fn get_panel_info(ctx: &mut GuestCtx, _st: &mut VitaState, port: u32, info: Ptr) -> i32 {
    if !info.is_null() {
        ctx.write_bytes(info.addr(), &panel_info_bytes(port));
    }
    0
}

/// `SCE_TOUCH_ERROR_INVALID_ARG`: a port outside the two panels.
const SCE_TOUCH_ERROR_INVALID_ARG: i32 = 0x8035_0001u32 as i32;

/// int sceTouchSetSamplingState(SceUInt32 port, SceTouchSamplingState state)
///
/// STOP(0) / START(1) for one panel. Recorded so [`get_sampling_state`] reads back what
/// the title set; the panels here always deliver a sample, so this does not gate
/// `sceTouchRead`/`sceTouchPeek` - see [`crate::host::VitaState::touch_sampling`].
#[hostcall]
pub(super) fn set_sampling_state(st: &mut VitaState, port: u32, state: u32) -> i32 {
    match st.touch_sampling.get_mut(port as usize) {
        Some(slot) => {
            *slot = state;
            0
        }
        None => SCE_TOUCH_ERROR_INVALID_ARG,
    }
}

/// int sceTouchGetSamplingState(SceUInt32 port, SceTouchSamplingState *pState)
///
/// The read-back of [`set_sampling_state`]. A port with no panel behind it is an
/// argument error rather than a state, which is the answer the caller can act on.
#[hostcall]
pub(super) fn get_sampling_state(ctx: &mut GuestCtx, st: &mut VitaState, port: u32, out: Ptr) -> i32 {
    // One expression, no early return: a `#[hostcall]` body cannot `return`.
    match st.touch_sampling.get(port as usize) {
        Some(&state) => {
            if !out.is_null() {
                ctx.write_u32(out.addr(), state);
            }
            0
        }
        None => SCE_TOUCH_ERROR_INVALID_ARG,
    }
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
