//! Frame-keyed scripted input: the [`Timeline`] a [`TimelineWorld`] replays, and the one-line
//! input grammar a live session speaks.
//!
//! # The input line
//! One line is the WHOLE pad and panel state, held until the next line replaces it:
//!
//! - Buttons: `cross`/`x`, `circle`, `square`, `triangle`, `up`, `down`, `left`, `right`,
//!   `start`, `select`, `ltrigger`/`l`, `rtrigger`/`r`, `l1`, `r1`, `l3`, `r3`.
//! - Analog axes `lx=`/`ly=`/`rx=`/`ry=` in `0..255` (128 neutral).
//! - A stick in POLAR form: `lang=<deg>[,<mag>]` and `rang=` set both axes of that stick from a
//!   compass bearing, `0` up-screen and increasing CLOCKWISE (`90` right, `180` down, `270`
//!   left), with an optional magnitude `0..127` that defaults to full deflection - `lang=0` is
//!   exactly `lx=128 ly=1`. A stick is often not two independent axes: a title whose left stick
//!   is an absolute heading cannot be driven diagonally by `lx=` alone. Polar and cartesian
//!   forms may be mixed; the last directive touching an axis wins.
//! - `touch=X,Y`: one finger on the FRONT panel at panel coordinates (1920x1088, so panel =
//!   screen * 2). `back=X,Y`: the same on the REAR panel (1920x890, no screen under it).
//!
//! An empty line releases everything, so a tap is one line and a later empty one.
//!
//! Input is keyed to the frame count the scheduler reports through [`World::set_frame`], not to
//! wall time or poll order, so the same timeline yields the same run however often the guest
//! polls the pad.

use crate::world::{CtrlFrame, TouchFrame, World};

/// One timeline entry: the input state that takes effect at `frame` and holds until the next
/// entry. Public because a live session appends them as it plays (see [`Timeline`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputSegment {
    pub frame: u64,
    pub input: CtrlFrame,
    /// The front panel (port 0).
    pub touch: TouchFrame,
    /// The rear panel (port 1).
    pub back: TouchFrame,
}

/// The frame-keyed, sticky input timeline a [`TimelineWorld`] replays: the segment active at a
/// frame is the last one at or before it.
///
/// APPENDABLE at runtime: a live session ([`SharedTimeline`]) pushes a new segment each time
/// the player changes the input, and the world the guest polls sees it from that frame on.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Timeline {
    segments: Vec<InputSegment>,
}

/// A [`Timeline`] shared between the world the guest polls and whoever drives it. One lock per
/// pad poll (once per display frame) is free at this granularity.
pub type SharedTimeline = std::sync::Arc<std::sync::Mutex<Timeline>>;

impl Timeline {
    /// A timeline over `segments`, frame-sorted (a stable sort keeps authoring order among
    /// entries sharing a frame, so the last one written wins).
    pub fn new(mut segments: Vec<InputSegment>) -> Timeline {
        segments.sort_by_key(|s| s.frame);
        Timeline { segments }
    }

    fn segment_at(&self, frame: u64) -> Option<&InputSegment> {
        self.segments.iter().rev().find(|s| s.frame <= frame)
    }

    /// The pad and front panel in effect at display frame `frame`: the last segment at or before
    /// it, or neutral before any segment starts.
    pub fn at(&self, frame: u64) -> (CtrlFrame, TouchFrame) {
        self.segment_at(frame)
            .map(|s| (s.input, s.touch))
            .unwrap_or_default()
    }

    /// The REAR panel's touch in effect at display frame `frame` - the [`Self::at`] of port 1.
    pub fn back_at(&self, frame: u64) -> TouchFrame {
        self.segment_at(frame).map(|s| s.back).unwrap_or_default()
    }

    /// Append a segment, keeping the timeline frame-sorted. A segment pushed at a frame that
    /// already has one is ordered after it, so the newest write wins - exactly the "I just
    /// pressed a button" semantics a live session needs.
    pub fn push(&mut self, seg: InputSegment) {
        let at = self.segments.partition_point(|s| s.frame <= seg.frame);
        self.segments.insert(at, seg);
    }

    /// Every segment, in frame order.
    pub fn segments(&self) -> &[InputSegment] {
        &self.segments
    }
}

/// The canonical keyword for each button bit, in the order a serialized line lists them (the
/// Vita `SceCtrlButtons` bits, from the MIT vita-headers `psp2common/ctrl.h`). The single source
/// of truth for a round trip: [`button_bit`] also accepts the aliases (`x`, `l`, `r`), which
/// never round-trip out.
pub const BUTTON_NAMES: &[(&str, u32)] = &[
    ("select", 0x0000_0001),
    ("l3", 0x0000_0002),
    ("r3", 0x0000_0004),
    ("start", 0x0000_0008),
    ("up", 0x0000_0010),
    ("right", 0x0000_0020),
    ("down", 0x0000_0040),
    ("left", 0x0000_0080),
    ("ltrigger", 0x0000_0100),
    ("rtrigger", 0x0000_0200),
    ("l1", 0x0000_0400),
    ("r1", 0x0000_0800),
    ("triangle", 0x0000_1000),
    ("circle", 0x0000_2000),
    ("cross", 0x0000_4000),
    ("square", 0x0000_8000),
];

/// The button bit for a keyword or one of its aliases (`x`, `l`, `r`). `None` if unknown.
pub fn button_bit(name: &str) -> Option<u32> {
    let canonical = match name {
        "x" => "cross",
        "l" => "ltrigger",
        "r" => "rtrigger",
        other => other,
    };
    BUTTON_NAMES
        .iter()
        .find(|(n, _)| *n == canonical)
        .map(|(_, bit)| *bit)
}

/// Resolve a polar stick directive's `<deg>[,<mag>]` argument to the `(x, y)` pair the pad
/// carries. `mag` out of `0..=127` is an ERROR rather than a clamp: asking for 200 is a wrong
/// model of the pad, and silently giving 127 would hide it until two headings drove identically.
fn polar_stick(val: &str) -> Result<(u8, u8), String> {
    let (deg_s, mag_s) = match val.split_once(',') {
        Some((d, m)) => (d, Some(m)),
        None => (val, None),
    };
    let deg: f64 = deg_s.trim().parse().map_err(|_| {
        format!("bad stick bearing {deg_s:?} (want degrees, 0 = up-screen, clockwise)")
    })?;
    let mag: f64 = match mag_s {
        Some(m) => m
            .trim()
            .parse()
            .map_err(|_| format!("bad stick magnitude {m:?} (want 0..127)"))?,
        None => 127.0,
    };
    if !(0.0..=127.0).contains(&mag) {
        return Err(format!(
            "stick magnitude {mag} out of range (0..127, 127 = fully deflected)"
        ));
    }
    let rad = deg.to_radians();
    // Screen y grows DOWNWARD in pad units (up-screen is the low end), hence the subtraction on
    // the y axis and not on x.
    let x = (128.0 + mag * rad.sin()).round();
    let y = (128.0 - mag * rad.cos()).round();
    Ok((x.clamp(0.0, 255.0) as u8, y.clamp(0.0, 255.0) as u8))
}

/// Parse one input line (see the module docs) into the pad, the front panel and the rear panel.
pub fn parse_input(line: &str) -> Result<(CtrlFrame, TouchFrame, TouchFrame), String> {
    let mut input = CtrlFrame::default();
    let mut touch = TouchFrame::default();
    let mut back = TouchFrame::default();
    for tok in line.split_whitespace() {
        let tok_lower = tok.to_ascii_lowercase();
        let Some((key, val)) = tok_lower.split_once('=') else {
            input.buttons |=
                button_bit(&tok_lower).ok_or_else(|| format!("unknown button {tok:?}"))?;
            continue;
        };
        match key {
            "touch" | "back" => {
                let (x, y) = val
                    .split_once(',')
                    .ok_or_else(|| format!("{key} expects X,Y panel coords, got {val:?}"))?;
                let px: u16 = x.parse().map_err(|_| format!("bad {key} X {x:?}"))?;
                let py: u16 = y.parse().map_err(|_| format!("bad {key} Y {y:?}"))?;
                if key == "back" {
                    back = TouchFrame::single(px, py);
                } else {
                    touch = TouchFrame::single(px, py);
                }
            }
            "lang" => (input.lx, input.ly) = polar_stick(val)?,
            "rang" => (input.rx, input.ry) = polar_stick(val)?,
            "lx" | "ly" | "rx" | "ry" => {
                let v: u8 = val
                    .parse()
                    .map_err(|_| format!("bad analog value {val:?} (expected 0..255)"))?;
                match key {
                    "lx" => input.lx = v,
                    "ly" => input.ly = v,
                    "rx" => input.rx = v,
                    _ => input.ry = v,
                }
            }
            _ => return Err(format!("unknown analog axis {key:?}")),
        }
    }
    Ok((input, touch, back))
}

/// A [`World`] that replays a [`Timeline`] over a virtual clock. Frame advance comes from the
/// scheduler via [`World::set_frame`].
pub struct TimelineWorld {
    timeline: SharedTimeline,
    frame: u64,
    monotonic_us: u64,
    wall_us: u64,
    rng: u64,
}

/// Microseconds per virtual frame (60Hz), used to advance the clock per frame so a title
/// reading elapsed time still sees monotonic progress.
const FRAME_US: u64 = 16_666;

impl TimelineWorld {
    /// A world over an externally-owned timeline. A live session holds the other end and
    /// appends to it as the run proceeds.
    pub fn from_timeline(timeline: SharedTimeline) -> Self {
        TimelineWorld {
            timeline,
            frame: 0,
            monotonic_us: 0,
            wall_us: 1_500_000_000_000_000,
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// A world over its own fresh timeline of `segments`.
    pub fn new(segments: Vec<InputSegment>) -> Self {
        Self::from_timeline(std::sync::Arc::new(std::sync::Mutex::new(Timeline::new(
            segments,
        ))))
    }

    /// A handle to the timeline this world replays, for a driver that appends to it.
    pub fn timeline(&self) -> SharedTimeline {
        self.timeline.clone()
    }
}

impl World for TimelineWorld {
    fn monotonic_us(&mut self) -> u64 {
        self.monotonic_us
    }
    fn wall_us(&mut self) -> u64 {
        self.wall_us
    }
    fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
        self.timeline.lock().unwrap().at(self.frame).0
    }
    fn poll_touch(&mut self, port: u32) -> TouchFrame {
        let t = self.timeline.lock().unwrap();
        match port {
            0 => t.at(self.frame).1,
            1 => t.back_at(self.frame),
            _ => TouchFrame::default(),
        }
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        // SplitMix64, matching DeterministicWorld: deterministic and cheap.
        for chunk in buf.chunks_mut(8) {
            self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            for (i, b) in chunk.iter_mut().enumerate() {
                *b = (z >> (i * 8)) as u8;
            }
        }
    }
    fn set_frame(&mut self, frame: u64) {
        self.frame = frame;
        // Keep the virtual clock roughly in step with frames so a title polling elapsed time
        // still advances (the preemptive scheduler's own virtual clock drives pacing; this only
        // backstops a title that reads monotonic_us).
        self.monotonic_us = frame.wrapping_mul(FRAME_US);
        self.wall_us = 1_500_000_000_000_000u64.wrapping_add(self.monotonic_us);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(frame: u64, line: &str) -> InputSegment {
        let (input, touch, back) = parse_input(line).unwrap();
        InputSegment {
            frame,
            input,
            touch,
            back,
        }
    }

    #[test]
    fn input_is_sticky_between_segments() {
        let mut w = TimelineWorld::new(vec![seg(30, "cross"), seg(45, "")]);
        w.set_frame(0);
        assert_eq!(
            w.poll_ctrl(0),
            CtrlFrame::default(),
            "neutral before the first segment"
        );
        w.set_frame(30);
        assert_eq!(w.poll_ctrl(0).buttons, 0x4000);
        w.set_frame(44);
        assert_eq!(w.poll_ctrl(0).buttons, 0x4000);
        w.set_frame(45);
        assert_eq!(w.poll_ctrl(0).buttons, 0);
    }

    #[test]
    fn buttons_aliases_and_analog_parse() {
        let (input, _, _) = parse_input("cross down").unwrap();
        assert_eq!(input.buttons, 0x4000 | 0x0040);
        let (input, _, _) = parse_input("x l r").unwrap();
        assert_eq!(input.buttons, 0x4000 | 0x0100 | 0x0200);
        let (input, _, _) = parse_input("lx=0 ly=255").unwrap();
        assert_eq!(
            (input.lx, input.ly, input.rx),
            (0, 255, 128),
            "unset axes stay centred"
        );
        assert!(parse_input("nosuchbutton").is_err());
        assert!(parse_input("lx=300").is_err());
    }

    #[test]
    fn a_later_segment_at_the_same_frame_wins_the_full_state() {
        let mut w = TimelineWorld::new(vec![seg(10, "cross"), seg(10, "circle")]);
        w.set_frame(10);
        assert_eq!(
            w.poll_ctrl(0).buttons,
            0x2000,
            "circle only - the state is replaced"
        );
    }

    /// `touch=` and `back=` script the two panels independently: neither sees the other's finger.
    #[test]
    fn the_front_and_rear_panels_are_scripted_apart() {
        let mut w = TimelineWorld::new(vec![
            seg(10, "back=960,445"),
            seg(20, "touch=1,2 back=3,4"),
            seg(30, ""),
        ]);
        w.set_frame(10);
        let t = w.poll_touch(1);
        assert_eq!((t.count, t.points[0].x, t.points[0].y), (1, 960, 445));
        assert_eq!(w.poll_touch(0).count, 0, "the front panel is untouched");
        w.set_frame(20);
        assert_eq!(
            (w.poll_touch(0).points[0].x, w.poll_touch(1).points[0].x),
            (1, 3)
        );
        w.set_frame(30);
        assert_eq!(
            (w.poll_touch(0).count, w.poll_touch(1).count),
            (0, 0),
            "an empty line lifts both"
        );
        assert!(parse_input("touch=100").is_err());
        assert!(parse_input("back=x,200").is_err());
    }

    #[test]
    fn the_clock_advances_with_the_frame() {
        let mut w = TimelineWorld::new(Vec::new());
        w.set_frame(0);
        assert_eq!(w.monotonic_us(), 0);
        w.set_frame(60);
        assert_eq!(w.monotonic_us(), 60 * FRAME_US);
    }

    /// The live-session model: the world replays a shared timeline its driver extends.
    #[test]
    fn a_timeline_appended_to_at_runtime_takes_effect() {
        let mut w = TimelineWorld::new(Vec::new());
        let timeline = w.timeline();
        w.set_frame(10);
        assert_eq!(w.poll_ctrl(0).buttons, 0);
        timeline.lock().unwrap().push(seg(11, "cross ly=0"));
        w.set_frame(11);
        assert_eq!((w.poll_ctrl(0).buttons, w.poll_ctrl(0).ly), (0x4000, 0));
        timeline.lock().unwrap().push(seg(11, ""));
        assert_eq!(
            w.poll_ctrl(0).buttons,
            0,
            "a push at a frame that has a segment supersedes it"
        );
    }

    #[test]
    fn polar_stick_directives_set_both_axes() {
        // Up-screen is the LOW end of the y axis, so bearing 0 must reach ly=1, not ly=255.
        for (spec, lx, ly) in [
            ("lang=0", 128, 1),
            ("lang=90", 255, 128),
            ("lang=180", 128, 255),
            ("lang=270", 1, 128),
        ] {
            let (input, _, _) = parse_input(spec).unwrap();
            assert_eq!((input.lx, input.ly), (lx, ly), "{spec}");
        }
        let (input, _, _) = parse_input("lang=45").unwrap();
        assert!(input.lx > 128 && input.ly < 128, "45 deg aims up-and-right");
        let (a, _, _) = parse_input("lang=-90").unwrap();
        let (b, _, _) = parse_input("lang=270").unwrap();
        assert_eq!((a.lx, a.ly), (b.lx, b.ly), "bearings wrap");
        let (input, _, _) = parse_input("rang=0").unwrap();
        assert_eq!((input.rx, input.ry, input.lx, input.ly), (128, 1, 128, 128));
        let (input, _, _) = parse_input("lang=90,60").unwrap();
        assert_eq!((input.lx, input.ly), (188, 128), "a partial deflection");
        let (input, _, _) = parse_input("lang=0 ly=200").unwrap();
        assert_eq!((input.lx, input.ly), (128, 200), "the later directive wins");
        assert!(
            parse_input("lang=0,200").is_err(),
            "an unreachable magnitude is an error"
        );
        assert!(parse_input("lang=up").is_err());
    }
}
