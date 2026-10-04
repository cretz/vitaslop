//! A gamepad drives the SHELL: the library, a title page, the settings, the import screen and,
//! when it is open, the in-game menu - the browser front end's `navpad.js`, same buttons.
//!
//! The d-pad or the left stick moves focus (egui's own spatial focus, fed arrow keys), south
//! activates what is focused, east goes back, start opens the settings (closes the in-game menu).
//! Held directions repeat after [`REPEAT_FIRST`], then every [`REPEAT`].
//!
//! Ownership of the pad is exclusive, as in the browser. While a game runs the pad belongs to
//! the session's [`crate::input::Input`] and this reads nothing; opening the in-game menu hands
//! it here and closing it hands it back. A press already down at a hand-over is ignored until it
//! is released, on both sides: the press that opened the menu does not also pick "Resume", and
//! the press that closed it does not reach the game.
//!
//! # Capturing a binding
//! The remap screens ask for "the next pad button pressed" ([`NavPad::begin_capture`]). While
//! a capture is armed the navigator does not navigate: every standard control is watched, the
//! ones already held when it was armed (the south press that opened it) are skipped until
//! released, and the first fresh press is reported as its Standard Gamepad name. That press
//! then counts as held at a hand-over, so it does not also click whatever has focus.
//!
//! It owns a `Gilrs` of its own (the session's input has another): the shell exists with no
//! session at all, and the two never read the pad at the same time.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use gilrs::{Axis, Button, Gilrs};
use vitaslop_frontend::input::GAMEPAD_CONTROLS;

/// The stick deflection that counts as a direction.
const STICK: f32 = 0.5;
pub const REPEAT_FIRST: Duration = Duration::from_millis(380);
pub const REPEAT: Duration = Duration::from_millis(110);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right,
}

/// What the pad asked of the shell this frame.
#[derive(Default, Debug)]
pub struct NavFrame {
    pub moves: Vec<Dir>,
    pub south: bool,
    pub east: bool,
    pub start: bool,
    /// The control an armed capture caught (see [`NavPad::begin_capture`]).
    pub captured: Option<&'static str>,
}

impl NavFrame {
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty() && !self.south && !self.east && !self.start && self.captured.is_none()
    }
}

/// A pad control as the navigator reads it: a button, or a stick direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Control {
    South,
    East,
    Start,
    Pad(Dir),
    Stick(Dir),
}

impl Control {
    fn dir(self) -> Option<Dir> {
        match self {
            Control::Pad(d) | Control::Stick(d) => Some(d),
            _ => None,
        }
    }
}

pub struct NavPad {
    gilrs: Option<Gilrs>,
    attached: bool,
    /// Down last poll.
    down: HashSet<Control>,
    /// Down at the hand-over: inert until released once.
    ignore: HashSet<Control>,
    repeat: Option<(Dir, Instant)>,
    /// An armed capture, and the controls held when it was armed (inert until released).
    capture: Option<HashSet<&'static str>>,
}

impl NavPad {
    pub fn new() -> NavPad {
        NavPad { gilrs: Gilrs::new().ok(), ..NavPad::inert() }
    }

    /// A navigator that reads no pad - a stand-in while the real one is borrowed.
    pub fn inert() -> NavPad {
        NavPad { gilrs: None, attached: false, down: HashSet::new(), ignore: HashSet::new(), repeat: None, capture: None }
    }

    /// Arm a capture: the next pad button pressed is reported in [`NavFrame::captured`]
    /// instead of navigating. Whatever is held right now does not count.
    pub fn begin_capture(&mut self) {
        self.pump();
        self.capture = Some(self.raw_pressed());
    }

    /// Disarm a capture that was cancelled another way (Esc, the mouse).
    pub fn cancel_capture(&mut self) {
        self.capture = None;
    }

    #[cfg(test)]
    fn capturing(&self) -> bool {
        self.capture.is_some()
    }

    /// One poll of an armed capture over the controls held `now`: the first fresh press, which
    /// also disarms it.
    fn capture_step(&mut self, now: &HashSet<&'static str>) -> Option<&'static str> {
        let held_at_arm = self.capture.as_mut()?;
        held_at_arm.retain(|c| now.contains(c));
        // In the table's order, so two buttons landing in one poll resolve the same way each time.
        let hit = GAMEPAD_CONTROLS.iter().copied().find(|c| now.contains(c) && !held_at_arm.contains(c))?;
        self.capture = None;
        Some(hit)
    }

    /// Take the pad (`true`) or give it up. Taking it ignores whatever is held right now.
    pub fn attach(&mut self, on: bool) {
        if on == self.attached {
            return;
        }
        self.attached = on;
        self.pump();
        self.ignore = if on { self.pressed() } else { HashSet::new() };
        self.down = HashSet::new();
        self.repeat = None;
    }

    /// Drain the pad's events (always - an unread queue only grows) and, when attached, report
    /// what was asked this frame.
    pub fn poll(&mut self) -> NavFrame {
        self.pump();
        if self.capture.is_some() {
            let now = self.raw_pressed();
            let captured = self.capture_step(&now);
            if captured.is_some() {
                // The press that was captured must not also click what has focus.
                self.ignore = self.pressed();
                self.down = HashSet::new();
                self.repeat = None;
            }
            return NavFrame { captured, ..NavFrame::default() };
        }
        if !self.attached {
            return NavFrame::default();
        }
        let now = self.pressed();
        self.step(now, Instant::now())
    }

    /// One poll's worth of edges and repeats from the controls held `now`.
    fn step(&mut self, now: HashSet<Control>, t: Instant) -> NavFrame {
        let mut out = NavFrame::default();
        self.ignore.retain(|c| now.contains(c));
        let mut held_dir = None;
        for &c in &now {
            if self.ignore.contains(&c) {
                continue;
            }
            if let Some(d) = c.dir() {
                held_dir = Some(d);
                if !self.down.contains(&c) {
                    out.moves.push(d);
                    self.repeat = Some((d, t + REPEAT_FIRST));
                }
                continue;
            }
            if self.down.contains(&c) {
                continue;
            }
            match c {
                Control::South => out.south = true,
                Control::East => out.east = true,
                Control::Start => out.start = true,
                _ => {}
            }
        }
        match (held_dir, self.repeat) {
            (Some(d), Some((rd, at))) if d == rd && t >= at => {
                if !out.moves.contains(&d) {
                    out.moves.push(d);
                }
                self.repeat = Some((d, t + REPEAT));
            }
            (None, _) => self.repeat = None,
            _ => {}
        }
        self.down = now;
        out
    }

    fn pump(&mut self) {
        if let Some(g) = self.gilrs.as_mut() {
            while let Some(ev) = g.next_event() {
                g.update(&ev);
            }
        }
    }

    /// Every standard control held, by its settings name.
    fn raw_pressed(&self) -> HashSet<&'static str> {
        let Some(g) = self.gilrs.as_ref() else { return HashSet::new() };
        let Some((_, pad)) = g.gamepads().find(|(_, p)| p.is_connected()) else { return HashSet::new() };
        GAMEPAD_CONTROLS.iter().copied().filter(|c| crate::input::gilrs_button(c).is_some_and(|b| pad.is_pressed(b))).collect()
    }

    fn pressed(&self) -> HashSet<Control> {
        let mut out = HashSet::new();
        let Some(g) = self.gilrs.as_ref() else { return out };
        let Some((_, pad)) = g.gamepads().find(|(_, p)| p.is_connected()) else { return out };
        for (b, c) in [
            (Button::South, Control::South),
            (Button::East, Control::East),
            (Button::Start, Control::Start),
            (Button::DPadUp, Control::Pad(Dir::Up)),
            (Button::DPadDown, Control::Pad(Dir::Down)),
            (Button::DPadLeft, Control::Pad(Dir::Left)),
            (Button::DPadRight, Control::Pad(Dir::Right)),
        ] {
            if pad.is_pressed(b) {
                out.insert(c);
            }
        }
        // gilrs reports stick Y up-positive.
        let (x, y) = (pad.value(Axis::LeftStickX), pad.value(Axis::LeftStickY));
        if y > STICK {
            out.insert(Control::Stick(Dir::Up));
        }
        if y < -STICK {
            out.insert(Control::Stick(Dir::Down));
        }
        if x < -STICK {
            out.insert(Control::Stick(Dir::Left));
        }
        if x > STICK {
            out.insert(Control::Stick(Dir::Right));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pad() -> NavPad {
        NavPad { gilrs: None, attached: true, down: HashSet::new(), ignore: HashSet::new(), repeat: None, capture: None }
    }

    fn held(cs: &[Control]) -> HashSet<Control> {
        cs.iter().copied().collect()
    }

    #[test]
    fn a_held_direction_moves_once_then_repeats() {
        let mut p = pad();
        let t0 = Instant::now();
        let down = held(&[Control::Pad(Dir::Down)]);
        assert_eq!(p.step(down.clone(), t0).moves, vec![Dir::Down]);
        assert!(p.step(down.clone(), t0 + REPEAT_FIRST / 2).moves.is_empty());
        assert_eq!(p.step(down.clone(), t0 + REPEAT_FIRST).moves, vec![Dir::Down]);
        assert!(p.step(down.clone(), t0 + REPEAT_FIRST + REPEAT / 2).moves.is_empty());
        assert_eq!(p.step(down.clone(), t0 + REPEAT_FIRST + REPEAT).moves, vec![Dir::Down]);
        assert!(p.step(HashSet::new(), t0 + REPEAT_FIRST * 3).moves.is_empty());
    }

    #[test]
    fn buttons_fire_on_the_press_edge_only() {
        let mut p = pad();
        let t = Instant::now();
        assert!(p.step(held(&[Control::South]), t).south);
        assert!(!p.step(held(&[Control::South]), t).south);
        assert!(!p.step(HashSet::new(), t).south);
        assert!(p.step(held(&[Control::South]), t).south);
    }

    #[test]
    fn a_capture_skips_the_press_that_armed_it_and_takes_the_next() {
        let mut p = pad();
        p.capture = Some(["south"].into_iter().collect());
        let now = |cs: &[&'static str]| cs.iter().copied().collect::<HashSet<_>>();
        assert_eq!(p.capture_step(&now(&["south"])), None, "still the press that opened the capture");
        assert_eq!(p.capture_step(&now(&[])), None);
        assert!(p.capturing());
        assert_eq!(p.capture_step(&now(&["r2"])), Some("r2"));
        assert!(!p.capturing(), "one capture, one press");
        assert_eq!(p.capture_step(&now(&["east"])), None);
    }

    #[test]
    fn a_press_held_at_the_hand_over_is_inert_until_released() {
        let mut p = pad();
        p.ignore = held(&[Control::South]);
        let t = Instant::now();
        assert!(!p.step(held(&[Control::South]), t).south, "the press that opened the menu");
        assert!(!p.step(HashSet::new(), t).south);
        assert!(p.step(held(&[Control::South]), t).south, "a fresh press after release");
    }
}
