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
fn buffer_count(count: i32) -> u32 {
    (count.max(1) as u32).min(CTRL_MAX_BUFFERS)
}

/// >>> A HISTORY THAT REPEATS THE PRESENT HAS NO EDGES IN IT, AND AN EDGE IS WHAT A MENU
/// >>> CURSOR MOVES ON.
///
/// The older samples used to carry the CURRENT buttons, on the reasoning that this engine's
/// world holds one input frame at a time and inventing a differing past would be inventing
/// input the recipe never gave. The premise is right and the conclusion was not: the past
/// this ring should hold is not invented, it is the input the world ALREADY GAVE, one
/// sample per display frame, and the engine simply was not keeping it.
///
/// MEASURED on PCSE00120's difficulty select, which ignored up and down while `cross` worked
/// at every other screen: a d-pad press moved 9.9 MB of guest heap - the title saw it - and
/// no cursor index changed. A title tests "held now AND NOT held one sample ago" before it
/// moves a cursor, precisely so that holding a direction does not scroll a menu at 60 Hz;
/// with every slot carrying the current buttons that test can never pass, while a press
/// acted on by LEVEL goes through. That is the shape of the defect exactly: one class of
/// press works, the other is read and discarded.
///
/// So this keeps a sample per VBLANK, as the hardware ring does, up to [`CTRL_MAX_BUFFERS`] of
/// them: the pad state is the world's, observed at each read; the vblanks between two reads
/// carry the state held before the later one (see `push`). A title that reads the pad several
/// times inside a vblank is served the same sample each time, exactly as it is on hardware.
#[derive(Default)]
pub(crate) struct CtrlHistory {
    /// One ring per port that has ever been read, newest last. A `Vec` rather than a fixed
    /// array because the port is a guest-supplied word: two ports are the real hardware and
    /// a title asking for a third must not index out of bounds.
    ports: Vec<(u32, std::collections::VecDeque<Sample>)>,
    /// Per port, the vblank of the last BLOCKING read (`sceCtrlReadBuffer*`) - see
    /// [`read_blocking`], which returns only the samples newer than it.
    last_read: Vec<(u32, u64)>,
}

/// One controller sample as it was served: the VBLANK it belongs to, the guest timestamp it
/// carried (the field a title dedupes on), and the pad state itself.
#[derive(Clone, Copy)]
struct Sample {
    vblank: u64,
    ts: u64,
    pad: crate::world::CtrlFrame,
}

impl CtrlHistory {
    /// Record `pad` as `vblank`'s sample for `port`, unless the newest sample is already this
    /// vblank's AND says the same thing.
    ///
    /// >>> THE STAMP IS THE VBLANK, NOT THE DISPLAY FRAME, and the pad state is part of the
    /// >>> test. The first version keyed on the display-frame counter, which only the
    /// > > > preemptive scheduler advances: under the run-to-completion host it never moves, so the
    /// > > > ring froze on its first (neutral) sample and a title reading more than one buffer never
    /// > > > saw a button at all. The conformance cube caught it - it pressed START and the guest ran
    /// > > > on for ever. The vblank counter is a pure function of the clock and advances under every
    /// > > > host; comparing the pad as well means a change is served the moment the world makes it,
    /// > > > whatever the clock is doing.
    fn push(&mut self, port: u32, vblank: u64, ts: u64, pad: crate::world::CtrlFrame) {
        let ring = match self.ports.iter().position(|(p, _)| *p == port) {
            Some(k) => &mut self.ports[k].1,
            None => {
                let cap = CTRL_MAX_BUFFERS as usize;
                self.ports.push((port, std::collections::VecDeque::with_capacity(cap)));
                &mut self.ports.last_mut().expect("just pushed").1
            }
        };
        let cap = CTRL_MAX_BUFFERS as usize;
        let push = |ring: &mut std::collections::VecDeque<Sample>, s: Sample| {
            if ring.len() == cap {
                ring.pop_front();
            }
            ring.push_back(s);
        };
        match ring.back().copied() {
            // >>> ONE SAMPLE PER VBLANK, AS THE HARDWARE RING HOLDS - never two for one vblank,
            // >>> and no vblank missing because nobody read the pad on it.
            //
            // The Vita samples the pad at every vblank whether or not a title reads it, so a
            // title polling every OTHER vblank (a 30 fps menu) that asks for 8 buffers gets 8
            // vblanks, 2 of them new. This ring used to add a sample only per READ: each 30 fps
            // read brought ONE new sample, and a title that takes "the last two" as new saw
            // every press edge twice. MEASURED: Madden reads `count=8` every 2 vblanks from one
            // site (0x8169ae05), and on the desktop browser one d-pad press moved its menu
            // cursor twice unless the key was only tapped (the user, 2026-09-25).
            //
            // The vblanks nobody read carry the state held BEFORE this read (the change is
            // observed here, so it is dated here), each stamped one vblank on.
            Some(last) if vblank == last.vblank => {
                // The same vblank read again after the pad moved: a vblank has one sample, and
                // it is the state as of now.
                if last.pad != pad
                    && let Some(b) = ring.back_mut() {
                        b.pad = pad;
                    }
            }
            Some(last) if vblank > last.vblank => {
                let gap = (vblank - last.vblank - 1).min(cap as u64);
                for k in 1..=gap {
                    let v = vblank - gap - 1 + k;
                    let t = last.ts + (v - last.vblank) * super::display::VBLANK_US;
                    push(ring, Sample { vblank: v, ts: t.min(ts.saturating_sub(1)), pad: last.pad });
                }
                push(ring, Sample { vblank, ts, pad });
            }
            // First sample, or a vblank counter that went BACKWARDS (a new clock): start over.
            _ => {
                ring.clear();
                push(ring, Sample { vblank, ts, pad });
            }
        }
    }

    /// The sample `age` samples back from the newest for `port` (0 = the newest), or `None`
    /// when the ring does not go back that far.
    fn sample(&self, port: u32, age: usize) -> Option<Sample> {
        let (_, ring) = self.ports.iter().find(|(p, _)| *p == port)?;
        ring.len().checked_sub(1 + age).and_then(|i| ring.get(i)).copied()
    }

    /// How many samples the ring holds for `port`.
    fn depth(&self, port: u32) -> usize {
        self.ports.iter().find(|(p, _)| *p == port).map_or(0, |(_, r)| r.len())
    }
}

/// Fill `count` `SceCtrlData` samples at `data` for `port` - the newest of the ring
/// [`CtrlHistory`] keeps, OLDEST FIRST - and return how many were written. When
/// `negative` is set each sample's button mask is inverted (the `*Negative` family
/// reports a pressed button as a cleared bit).
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
            count,
            vblank = super::display::vcount(st),
            lr = format_args!("{:#010x}", ctx.regs[14]),
            "ctrl"
        );
    }

    // The ring advances once per VBLANK - see [`CtrlHistory`], which is the rate the pad is
    // sampled at. Pushed here rather than on a timer because the world is polled here: asking
    // it for a sample the guest never requested would put a poll on the record/replay seam
    // that no title made.
    st.ctrl_history.push(port, u64::from(super::display::vcount(st)), ts, frame);
    let depth = st.ctrl_history.depth(port);

    let n = buffer_count(count);
    if data != 0 {
        // The oldest sample the ring holds, for the slots below it: a title asking for more
        // history than the run has produced gets that sample extended backwards on the vblank
        // grid, with its own distinct timestamp per slot because that is the field a title
        // dedupes on. Extending the OLDEST rather than the newest is what keeps the ring
        // monotone: a run one frame old has one real sample, and the slots below it are the
        // pad as it was before anything was pressed.
        let oldest = st.ctrl_history.sample(port, depth.saturating_sub(1));
        for i in 0..n {
            // Oldest first: slot i is (n - 1 - i) samples before now, and the last one
            // written is the current sample.
            let age = (n - 1 - i) as usize;
            let (s_ts, pad) = match st.ctrl_history.sample(port, age) {
                Some(s) => (s.ts, s.pad),
                None => {
                    let base = oldest.map_or((ts, crate::world::CtrlFrame::default()), |s| (s.ts, s.pad));
                    let short = (age + 1 - depth) as u64;
                    (base.0.saturating_sub(short * super::display::VBLANK_US), base.1)
                }
            };
            let bits = if negative { !pad.buttons } else { pad.buttons };
            let mut buf = [0u8; CTRL_DATA_PREFIX];
            buf[0..8].copy_from_slice(&s_ts.to_le_bytes());
            buf[8..12].copy_from_slice(&bits.to_le_bytes());
            buf[12] = pad.lx;
            buf[13] = pad.ly;
            buf[14] = pad.rx;
            buf[15] = pad.ry;
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
    // >>> A READ RETURNS THE SAMPLES SINCE THE LAST READ, NOT `count` OF HISTORY.
    //
    // `Peek` hands back the newest `count` samples; `Read` hands back only the vblanks sampled
    // since the previous `Read` on that port (at least one - it blocks for the next one
    // otherwise), and RETURNS how many (Vita3K: `min(count, vcount - last_vcount[port])`).
    // Returning `count` told a title that reads every 2-3 vblanks with `count = 8` that the pad
    // had been held for 8 samples each time - MEASURED on Madden's front end (read site lr
    // 0x8169ae05, count 8, every 2-3 vblanks): a held d-pad scrolled 4 rows in 12 frames,
    // its hold-timer running ~3x fast. The slots keep this ring's order (oldest first, newest
    // last), so a title reading `buf[ret - 1]` gets the current sample exactly as before.
    // >>> OPT-IN (`VITASLOP_CTRL_READ_NEW=1`), NOT THE DEFAULT: it did NOT change Madden's
    // held-d-pad rate (ctl26fn vs ctl26fo, team wheel, identical cadence) and it DOES change
    // OlliOlli's pushing recipe (2 of 2 shots differ; READ_ALL restores them byte-identical),
    // with no ground truth for which reading is the hardware's. Default = `count` samples.
    let vblank = u64::from(super::display::vcount(st));
    let count = if read_all() {
        count
    } else {
        let h = &mut st.ctrl_history;
        let new = match h.last_read.iter_mut().find(|(p, _)| *p == port) {
            Some((_, last)) => {
                let n = vblank.saturating_sub(*last);
                *last = vblank;
                n
            }
            None => {
                h.last_read.push((port, vblank));
                u64::from(buffer_count(count))
            }
        };
        new.clamp(1, u64::from(buffer_count(count))) as i32
    };
    let n = fill_ctrl(ctx, st, port, data, negative, count);
    tracing::trace!(target: "vitaslop::input", port, asked = ctx.arg(2), returned = n, vblank, lr = format_args!("{:#010x}", ctx.regs[14]), "ctrl read");
    ctx.ret(n as u32);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    st.vblank_park(1, super::display::VBLANK_US);
    SvcOutcome::Block
}

/// A blocking read returns `count` samples (the default) unless `VITASLOP_CTRL_READ_NEW=1`.
fn read_all() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_CTRL_READ_NEW").as_deref() != Ok("1"))
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

#[cfg(test)]
mod history_tests {
    use super::*;
    use crate::world::CtrlFrame;

    const UP: u32 = 0x0010;

    fn pad(buttons: u32) -> CtrlFrame {
        CtrlFrame { buttons, ..CtrlFrame::default() }
    }

    fn buttons(h: &CtrlHistory, n: usize) -> Vec<u32> {
        (0..n).rev().filter_map(|age| h.sample(0, age)).map(|s| s.pad.buttons).collect()
    }

    #[test]
    fn a_thirty_hz_reader_gets_one_sample_per_vblank_so_an_edge_appears_once() {
        let mut h = CtrlHistory::default();
        let v = super::super::display::VBLANK_US;
        // Read every OTHER vblank: neutral, neutral, then UP is pressed before the read at 6.
        h.push(0, 2, 2 * v, pad(0));
        h.push(0, 4, 4 * v, pad(0));
        h.push(0, 6, 6 * v, pad(UP));
        assert_eq!(h.depth(0), 5, "vblanks 2..=6, one sample each");
        assert_eq!(buttons(&h, 5), vec![0, 0, 0, 0, UP], "vblank 5 holds the state before the read");
        // The next read (vblank 8, still held): the two NEW samples are both UP - "the last
        // two samples" contain no rising edge, which is what stops a second cursor step.
        h.push(0, 8, 8 * v, pad(UP));
        let last_two = buttons(&h, 2);
        assert_eq!(last_two, vec![UP, UP]);
        // Negative control - the old per-READ ring's last two here were [0, UP] again.
        let edges: usize = buttons(&h, h.depth(0)).windows(2).filter(|w| w[0] & UP == 0 && w[1] & UP != 0).count();
        assert_eq!(edges, 1, "one press is one rising edge in the history");
    }

    #[test]
    fn a_vblank_has_one_sample_and_timestamps_stay_increasing() {
        let mut h = CtrlHistory::default();
        let v = super::super::display::VBLANK_US;
        h.push(0, 10, 10 * v, pad(0));
        h.push(0, 10, 10 * v + 5, pad(UP));
        assert_eq!(h.depth(0), 1, "a second read in one vblank replaces, never appends");
        assert_eq!(buttons(&h, 1), vec![UP]);
        h.push(0, 13, 13 * v, pad(0));
        let ts: Vec<u64> = (0..h.depth(0)).rev().filter_map(|a| h.sample(0, a)).map(|s| s.ts).collect();
        assert!(ts.windows(2).all(|w| w[0] < w[1]), "stamps are what titles dedupe on: {ts:?}");
        // A very long gap fills at most a ring's worth.
        h.push(0, 10_000, 10_000 * v, pad(0));
        assert_eq!(h.depth(0), CTRL_MAX_BUFFERS as usize);
    }
}
