//! A deterministic scripted-input world: a naive open-loop "TAS recipe" that drives
//! a title's controller input as a function of the frame count, so a headless run
//! can navigate menus and dialogs reproducibly.
//!
//! The recipe is line based. Each non-blank, non-comment line is
//!
//! ```text
//! <frame>: <directive> <directive> ...
//! ```
//!
//! where `<frame>` is the display-flip index at which this input state takes
//! effect. The state is STICKY: it holds until the next line's frame, so a recipe
//! is a sequence of held-button segments (press at frame A, an empty line at frame
//! B releases). Directives are buttons to hold this segment - `cross`, `circle`,
//! `square`, `triangle`, `up`, `down`, `left`, `right`, `start`, `select`, `l`, `r`
//! - and analog overrides `lx=`, `ly=`, `rx=`, `ry=` in `0..255` (128 is neutral).
//! A line with no directives releases everything and re-centers the sticks. `#`
//! starts a comment.
//!
//! A `touch=X,Y` directive holds one finger on the front panel at panel coordinates
//! `(X, Y)` for this segment (the front panel is 1920x1088, so panel = screen*2). A
//! tap is press-then-release: `touch=` on one segment, then a later segment without
//! it lifts the finger. This mirrors the sticky button model, so menu taps and
//! gameplay buttons script together in one recipe. OlliOlli's front-end is touch
//! driven, so navigating its menus needs these.
//!
//! ```text
//! # dismiss the "not signed in" dialog, then move the menu selection down
//! 0:
//! 30: cross          # hold X from frame 30
//! 45:                # release
//! 90: down           # nudge the selection down
//! 105:
//! ```
//!
//! Input is keyed to the frame count the scheduler reports through
//! [`World::set_frame`], not to wall time or poll order, so the same recipe yields
//! the same run regardless of how often the guest polls the pad. The world still
//! owns a virtual clock (advanced one 60Hz tick per frame) and a seeded PRNG, so it
//! is a drop-in replacement for [`DeterministicWorld`](crate::world::DeterministicWorld)
//! that happens to press buttons.

use crate::world::{CtrlFrame, TouchFrame, World};

/// The Vita `SceCtrlButtons` bit for each directive keyword (from the MIT
/// vita-headers `psp2common/ctrl.h`).
fn button_bit(name: &str) -> Option<u32> {
    Some(match name {
        "select" => 0x0000_0001,
        "l3" => 0x0000_0002,
        "r3" => 0x0000_0004,
        "start" => 0x0000_0008,
        "up" => 0x0000_0010,
        "right" => 0x0000_0020,
        "down" => 0x0000_0040,
        "left" => 0x0000_0080,
        "ltrigger" | "l" => 0x0000_0100,
        "rtrigger" | "r" => 0x0000_0200,
        "l1" => 0x0000_0400,
        "r1" => 0x0000_0800,
        "triangle" => 0x0000_1000,
        "circle" => 0x0000_2000,
        "cross" | "x" => 0x0000_4000,
        "square" => 0x0000_8000,
        _ => return None,
    })
}

/// One parsed recipe entry: the input state that takes effect at `frame` and holds
/// until the next entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Segment {
    frame: u64,
    input: CtrlFrame,
    touch: TouchFrame,
}

/// A parse failure with the 1-based line number and a reason.
#[derive(Debug, PartialEq, Eq)]
pub struct RecipeError {
    pub line: usize,
    pub reason: String,
}

impl std::fmt::Display for RecipeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "recipe line {}: {}", self.line, self.reason)
    }
}

/// Parse a recipe into frame-sorted segments. Empty on an empty input.
fn parse(text: &str) -> Result<Vec<Segment>, RecipeError> {
    let mut segments: Vec<Segment> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line_no = i + 1;
        // Strip a trailing comment and surrounding whitespace.
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (frame_str, rest) = line.split_once(':').ok_or_else(|| RecipeError {
            line: line_no,
            reason: "expected '<frame>: directives'".into(),
        })?;
        let frame: u64 = frame_str.trim().parse().map_err(|_| RecipeError {
            line: line_no,
            reason: format!("bad frame number {:?}", frame_str.trim()),
        })?;
        let mut input = CtrlFrame::default();
        let mut touch = TouchFrame::default();
        for tok in rest.split_whitespace() {
            let tok_lower = tok.to_ascii_lowercase();
            if let Some((key, val)) = tok_lower.split_once('=') {
                if key == "touch" {
                    let (x, y) = val.split_once(',').ok_or_else(|| RecipeError {
                        line: line_no,
                        reason: format!("touch expects X,Y panel coords, got {val:?}"),
                    })?;
                    let px: u16 = x.parse().map_err(|_| RecipeError {
                        line: line_no,
                        reason: format!("bad touch X {x:?}"),
                    })?;
                    let py: u16 = y.parse().map_err(|_| RecipeError {
                        line: line_no,
                        reason: format!("bad touch Y {y:?}"),
                    })?;
                    touch = TouchFrame::single(px, py);
                    continue;
                }
                let v: u8 = val.parse().map_err(|_| RecipeError {
                    line: line_no,
                    reason: format!("bad analog value {val:?} (expected 0..255)"),
                })?;
                match key {
                    "lx" => input.lx = v,
                    "ly" => input.ly = v,
                    "rx" => input.rx = v,
                    "ry" => input.ry = v,
                    _ => {
                        return Err(RecipeError {
                            line: line_no,
                            reason: format!("unknown analog axis {key:?}"),
                        })
                    }
                }
            } else {
                let bit = button_bit(&tok_lower).ok_or_else(|| RecipeError {
                    line: line_no,
                    reason: format!("unknown button {tok:?}"),
                })?;
                input.buttons |= bit;
            }
        }
        segments.push(Segment { frame, input, touch });
    }
    // Sort by frame so lookup is a simple last-entry-at-or-before scan; a stable
    // sort keeps the file order among entries that share a frame (last wins).
    segments.sort_by_key(|s| s.frame);
    Ok(segments)
}

/// A [`World`] that replays a scripted-input recipe over a virtual clock. Frame
/// advance comes from the scheduler via [`World::set_frame`].
pub struct RecipeWorld {
    segments: Vec<Segment>,
    frame: u64,
    monotonic_us: u64,
    wall_us: u64,
    rng: u64,
}

/// Microseconds per virtual frame (60Hz), used to advance the clock per frame so a
/// title reading elapsed time still sees monotonic progress.
const FRAME_US: u64 = 16_666;

impl RecipeWorld {
    /// Parse `text` into a scripted-input world. Fails with the offending line on a
    /// malformed recipe.
    pub fn parse(text: &str) -> Result<Self, RecipeError> {
        Ok(RecipeWorld {
            segments: parse(text)?,
            frame: 0,
            monotonic_us: 0,
            wall_us: 1_500_000_000_000_000,
            rng: 0x9E37_79B9_7F4A_7C15,
        })
    }

    /// The segment active at the current frame: the last one whose frame is at or
    /// before it, or `None` before any segment starts (neutral state).
    fn current(&self) -> Option<&Segment> {
        self.segments.iter().rev().find(|s| s.frame <= self.frame)
    }
}

impl World for RecipeWorld {
    fn monotonic_us(&mut self) -> u64 {
        self.monotonic_us
    }
    fn wall_us(&mut self) -> u64 {
        self.wall_us
    }
    fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
        self.current().map(|s| s.input).unwrap_or_default()
    }
    fn poll_touch(&mut self, port: u32) -> TouchFrame {
        // Only the front panel (port 0) is scripted; the back panel stays untouched.
        if port == 0 {
            self.current().map(|s| s.touch).unwrap_or_default()
        } else {
            TouchFrame::default()
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
        // Keep the virtual clock roughly in step with frames so a title polling
        // elapsed time still advances (the preemptive scheduler's own virtual clock
        // drives pacing; this only backstops a title that reads monotonic_us).
        self.monotonic_us = frame.wrapping_mul(FRAME_US);
        self.wall_us = 1_500_000_000_000_000u64.wrapping_add(self.monotonic_us);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_buttons_and_analog_and_comments() {
        let r = "# header\n0:\n30: cross down  # press\n45: lx=0 ly=255\n";
        let segs = parse(r).unwrap();
        assert_eq!(segs.len(), 3);
        assert_eq!(
            segs[0],
            Segment { frame: 0, input: CtrlFrame::default(), touch: TouchFrame::default() }
        );
        assert_eq!(segs[1].frame, 30);
        assert_eq!(segs[1].input.buttons, 0x4000 | 0x0040); // cross | down
        assert_eq!(segs[2].input.lx, 0);
        assert_eq!(segs[2].input.ly, 255);
        // Analog defaults stay centered where not overridden.
        assert_eq!(segs[2].input.rx, 128);
    }

    #[test]
    fn input_is_sticky_between_segments() {
        let mut w = RecipeWorld::parse("30: cross\n45:\n").unwrap();
        // Before the first segment: neutral.
        w.set_frame(0);
        assert_eq!(w.poll_ctrl(0), CtrlFrame::default());
        // From frame 30 the cross is held...
        w.set_frame(30);
        assert_eq!(w.poll_ctrl(0).buttons, 0x4000);
        w.set_frame(44);
        assert_eq!(w.poll_ctrl(0).buttons, 0x4000);
        // ...and released at 45.
        w.set_frame(45);
        assert_eq!(w.poll_ctrl(0).buttons, 0);
    }

    #[test]
    fn aliases_x_and_l_r() {
        let segs = parse("0: x l r\n").unwrap();
        assert_eq!(segs[0].input.buttons, 0x4000 | 0x0100 | 0x0200);
    }

    #[test]
    fn later_line_at_same_frame_wins_full_state() {
        // Two entries at the same frame: the stable sort keeps file order, and the
        // reverse scan picks the last, so the second line's state is the one used.
        let mut w = RecipeWorld::parse("10: cross\n10: circle\n").unwrap();
        w.set_frame(10);
        assert_eq!(w.poll_ctrl(0).buttons, 0x2000); // circle only (state replaced)
    }

    #[test]
    fn touch_is_sticky_and_lifts_on_release() {
        let mut w = RecipeWorld::parse("10: touch=450,674\n20:\n").unwrap();
        // Before the first segment: no finger.
        w.set_frame(0);
        assert_eq!(w.poll_touch(0).count, 0);
        // Finger down at panel (450,674) from frame 10.
        w.set_frame(10);
        let t = w.poll_touch(0);
        assert_eq!(t.count, 1);
        assert_eq!((t.points[0].x, t.points[0].y), (450, 674));
        // Held through the segment...
        w.set_frame(19);
        assert_eq!(w.poll_touch(0).count, 1);
        // ...and lifted at frame 20.
        w.set_frame(20);
        assert_eq!(w.poll_touch(0).count, 0);
        // The back panel is never scripted.
        w.set_frame(10);
        assert_eq!(w.poll_touch(1).count, 0);
    }

    #[test]
    fn touch_and_buttons_share_a_segment() {
        // A menu tap and a button hold can coexist on one line (mixed scheme).
        let mut w = RecipeWorld::parse("5: cross touch=100,200\n").unwrap();
        w.set_frame(5);
        assert_eq!(w.poll_ctrl(0).buttons, 0x4000);
        assert_eq!(w.poll_touch(0).count, 1);
    }

    #[test]
    fn rejects_malformed_touch() {
        assert_eq!(parse("0: touch=100\n").unwrap_err().line, 1);
        assert_eq!(parse("0: touch=x,200\n").unwrap_err().line, 1);
    }

    #[test]
    fn reports_the_offending_line_on_error() {
        assert_eq!(parse("0:\nnope\n").unwrap_err().line, 2);
        assert_eq!(parse("5: wiggle\n").unwrap_err().line, 1);
        assert_eq!(parse("x: cross\n").unwrap_err().line, 1);
        assert_eq!(parse("0: lx=999\n").unwrap_err().line, 1);
    }

    #[test]
    fn clock_advances_with_frame() {
        let mut w = RecipeWorld::parse("").unwrap();
        w.set_frame(0);
        assert_eq!(w.monotonic_us(), 0);
        w.set_frame(60);
        assert_eq!(w.monotonic_us(), 60 * FRAME_US);
    }
}
