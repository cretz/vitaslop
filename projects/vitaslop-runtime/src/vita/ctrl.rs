//! SceCtrl: controller input. The one place a title reads a non-deterministic
//! external input, so it goes through the World seam (`poll_ctrl`), which a record
//! or replay wrapper - or a scripted TAS recipe - drives.
//!
//! Titles reach the pad through several entry points with the same `SceCtrlData`
//! payload: `Peek` returns the latest sample immediately, `Read` classically blocks
//! until the next controller sample. We serve both from the current [`World`] frame;
//! `Read` additionally parks the caller on the sampling grid, which is what paces a
//! title whose update loop is not the loop that flips - see [`read_buffer_positive`].
//! The `Positive`/`Positive2` variants differ only in button remapping we do not
//! model, so they share one filler.

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::SvcOutcome;

/// Bytes of SceCtrlData we populate: timeStamp (8) + buttons (4) + the four analog
/// bytes lx/ly/rx/ry (4). The real struct is larger but the guest reads these
/// fields; we fill this prefix and leave the rest zeroed.
const CTRL_DATA_PREFIX: usize = 16;

/// `sizeof(SceCtrlData)`, from `psp2common/ctrl.h` (which asserts it). The STRIDE between
/// consecutive samples when a title asks for more than one, so it has to be the real size
/// and not the prefix above.
const CTRL_DATA_SIZE: u32 = 0x20;

/// The most samples one call will fill, whatever `count` says. The library's own limit is
/// 64 buffers; a larger request is the caller's error and writing what it asked for would
/// scribble over guest memory past a buffer that cannot be that big.
const CTRL_MAX_BUFFERS: u32 = 64;

/// >>> `count` IS THE NUMBER OF SAMPLES OF **HISTORY** THE TITLE IS ASKING FOR, AND
/// >>> FILLING ONLY THE FIRST ONE LEAVES A TITLE READING UNINITIALIZED STACK.
///
/// The pad is sampled once per vblank and the library keeps a ring of past samples; a
/// call with `count = n` copies the newest `n` of them OLDEST FIRST, so `buf[n - 1]` is
/// the current sample. A title that only wants "what is held right now" passes 1 and
/// never notices the difference - which is why filling one buffer worked for three titles
/// and looked correct.
///
/// MEASURED on the golf title, whose whole front end was inert: its input manager reads
/// **32** buffers (`sceCtrlPeekBufferPositive(0, sp+0x208, 0x20)` at `0x812f9a98`), then
/// takes the TIMESTAMP of `buf[31]` - `sp+0x5e8`, exactly `31 * 0x20` bytes in - and
/// searches its own 32-entry timestamp ring for it before it will treat the sample as new.
/// With one buffer filled, `buf[31]` is whatever was on the stack, so every press was read
/// and then discarded: the menu consumed a `cross` (9,701 bytes of guest state moved) and
/// acted on nothing, and a `down` moved not one byte.
///
/// The older samples carry the CURRENT buttons rather than a real history: this engine's
/// world holds one input frame at a time, and inventing a differing past would be
/// inventing input the recipe never gave. What each sample does carry is its own distinct
/// timestamp on the sampling grid, because that is the field a title dedupes on.
fn buffer_count(count: i32) -> u32 {
    (count.max(1) as u32).min(CTRL_MAX_BUFFERS)
}

/// Fill one `SceCtrlData` at `data` from the current world frame for `port`, then
/// return the number of buffers reported (always 1 - a single latest sample). When
/// `negative` is set the button mask is inverted (the `*Negative` family reports a
/// pressed button as a cleared bit).
fn fill_ctrl(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    port: u32,
    data: u32,
    negative: bool,
    count: i32,
) -> i32 {
    let frame = st.world.poll_ctrl(port);
    let ts = st.guest_mono_us();
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

    let n = buffer_count(count);
    if data != 0 {
        for i in 0..n {
            // Oldest first: sample i is (n - 1 - i) vblanks before now, and the last one
            // written is the current sample.
            let age = u64::from(n - 1 - i) * u64::from(super::display::VBLANK_US);
            let mut buf = [0u8; CTRL_DATA_PREFIX];
            buf[0..8].copy_from_slice(&ts.saturating_sub(age).to_le_bytes());
            buf[8..12].copy_from_slice(&buttons.to_le_bytes());
            buf[12] = frame.lx;
            buf[13] = frame.ly;
            buf[14] = frame.rx;
            buf[15] = frame.ry;
            ctx.write_bytes(data + i * CTRL_DATA_SIZE, &buf);
        }
    }
    n as i32
}

/// int sceCtrlPeekBufferPositive(int port, SceCtrlData *pad_data, int count)
#[hostcall]
pub(super) fn peek_buffer_positive(ctx: &mut GuestCtx, st: &mut VitaState, port: u32, data: Ptr, count: i32) -> i32 {
    fill_ctrl(ctx, st, port, data.addr(), false, count)
}

/// int sceCtrlReadBufferPositive(int port, SceCtrlData *pad_data, int count)
///
/// >>> THIS CALL BLOCKS, AND IT IS THE FRAME LIMITER OF EVERY TITLE THAT USES IT.
///
/// `Peek` returns the latest sample; `Read` waits for the NEXT one. The controller is
/// sampled once per vblank, so a loop built on `Read` runs exactly 60 times a second on
/// hardware whatever else it does - which is why it is the standard shape for a Vita main
/// loop, and why returning the current sample immediately is not a near-enough
/// approximation of it.
///
/// The old justification here was "the render loop's display flip is the frame-pacing
/// yield, so input still advances one frame per rendered frame". That holds only when the
/// loop reading input is the loop that flips. MEASURED on a retail racer's race, 100
/// display frames of a headless run: its update loop - `sceCtrlReadBufferPositive`,
/// `sceRtcGetCurrentTick`, `sceMotionGetState` and its two network callbacks, all at
/// exactly the same count - ran **12.7 times per rendered frame**, while
/// `sceGxmBeginScene`/`EndScene`/`Draw` ran exactly once. The update loop is a separate
/// thread from the one that presents, nothing paced it, and it stepped the simulation
/// every time round: the title's own lap timer advanced **12.4x real time**
/// ([[vitaslop-game-clock-runs-12-8x-the-emulated-clock]] measured that ratio without
/// finding what produced it). That is what "the race is way sped up" is.
///
/// It is not a clock-calibration problem, and that was checked before this was changed:
/// the run charges 7.4M guest ARM instructions per displayed frame, which is exactly one
/// frame of a 444 MHz core, so the emulated CPU speed is right and the loop is simply
/// cheap enough to go round a dozen times inside a frame. On hardware it would too. What
/// stops it there is this call.
///
/// So it parks on the sampling grid, exactly as [`super::display::wait_vblank_start`]
/// does: `vblank_park(1, ..)` waits for the first edge STRICTLY after now, so a loop of
/// these phase-locks to 60 Hz instead of over-waiting a full period from wherever it
/// happened to call. The sample is filled and the result written BEFORE parking, because
/// a woken thread resumes inside the call with the registers it parked with.
///
/// The run-to-completion host has no scheduler to park against and keeps the old
/// behaviour; there the display flip really is the only yield there is.
pub(super) fn read_buffer_positive(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    read_blocking(ctx, st, false)
}

/// The blocking half of `sceCtrlRead*`, shared by the Positive and Negative spellings.
/// See [`read_buffer_positive`] for why it parks.
fn read_blocking(ctx: &mut GuestCtx, st: &mut VitaState, negative: bool) -> SvcOutcome {
    let (port, data, count) = (ctx.arg(0), ctx.arg(1), ctx.arg(2) as i32);
    let n = fill_ctrl(ctx, st, port, data, negative, count);
    ctx.ret(n as u32);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    st.vblank_park(1, super::display::VBLANK_US);
    SvcOutcome::Block
}

/// int sceCtrlPeekBufferNegative(int port, SceCtrlData *pad_data, int count)
#[hostcall]
pub(super) fn peek_buffer_negative(ctx: &mut GuestCtx, st: &mut VitaState, port: u32, data: Ptr, count: i32) -> i32 {
    fill_ctrl(ctx, st, port, data.addr(), true, count)
}

/// int sceCtrlReadBufferNegative(int port, SceCtrlData *pad_data, int count)
/// Blocks on the sampling grid exactly as [`read_buffer_positive`] does.
pub(super) fn read_buffer_negative(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    read_blocking(ctx, st, true)
}
