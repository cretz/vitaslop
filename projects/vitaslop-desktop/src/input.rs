//! Live desktop input: keyboard (winit) and gamepad (gilrs) into one [`CtrlFrame`],
//! mapped by the person's settings.
//!
//! The settings name keys by W3C `KeyboardEvent.code` (`KeyZ`, `ArrowUp`) and pad
//! controls by Standard Gamepad position (`south`, `dpad_up`) - the vocabulary the
//! browser front end uses, so one remap serves both. winit's `KeyCode` debug names ARE
//! the W3C codes, and gilrs' buttons map onto the standard positions below.

use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant};

use gilrs::{Axis, Button, Gilrs};
use vitaslop_frontend::input::invert;
use vitaslop_frontend::settings::Settings;
use vitaslop_runtime::CtrlFrame;
use winit::keyboard::KeyCode;

const CENTER: u8 = 128;

/// Start and select held together this long open the shell's in-game menu - the browser's
/// `MENU_HOLD_MS`, and the way in that every pad has (not every pad exposes `home`).
const MENU_HOLD: Duration = Duration::from_millis(1000);

pub struct Input {
    keys: HashSet<String>,
    gilrs: Option<Gilrs>,
    /// `KeyboardEvent.code` -> button bits.
    keymap: BTreeMap<String, u32>,
    /// Standard Gamepad control -> button bits.
    padmap: BTreeMap<String, u32>,
    deadzone: f32,
    /// Pad buttons the game must not see until they are released: the start+select chord once
    /// it opened the menu, `home` when it did, and whatever was held when the menu handed the
    /// pad back (the press that picked "Resume"). See [`Self::hand_back`].
    ignore: HashSet<Button>,
    /// When start and select both went down, while they are both held.
    chord_since: Option<Instant>,
    /// The menu was asked for and not yet taken - see [`Self::take_menu_request`].
    menu_request: bool,
}

impl Input {
    pub fn new(settings: &Settings) -> Self {
        let gilrs = match Gilrs::new() {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("gamepad support unavailable ({e}); keyboard only");
                None
            }
        };
        let mut me = Input {
            keys: HashSet::new(),
            gilrs,
            keymap: BTreeMap::new(),
            padmap: BTreeMap::new(),
            deadzone: 0.14,
            ignore: HashSet::new(),
            chord_since: None,
            menu_request: false,
        };
        me.apply(settings);
        me
    }

    pub fn apply(&mut self, settings: &Settings) {
        self.keymap = invert(&settings.keyboard);
        self.padmap = invert(&settings.gamepad);
        self.deadzone = settings.stick_deadzone;
    }

    pub fn set_key(&mut self, key: KeyCode, pressed: bool) {
        let name = format!("{key:?}");
        if pressed {
            self.keys.insert(name);
        } else {
            self.keys.remove(&name);
        }
    }

    pub fn release_all(&mut self) {
        self.keys.clear();
    }

    /// Drain pending gilrs events so the gamepad state read below is current, and watch for the
    /// pad's way into the shell's menu: `home` (unless a Vita button is mapped to it, which makes
    /// it the game's), or start+select held for [`MENU_HOLD`]. The browser's rule.
    pub fn pump_gamepad(&mut self) {
        let Some(g) = self.gilrs.as_mut() else { return };
        while let Some(ev) = g.next_event() {
            g.update(&ev);
        }
        let Some((_, pad)) = g.gamepads().find(|(_, p)| p.is_connected()) else {
            self.ignore.clear();
            self.chord_since = None;
            return;
        };
        let held: HashSet<Button> = ALL_BUTTONS.iter().copied().filter(|b| pad.is_pressed(*b)).collect();
        self.ignore.retain(|b| held.contains(b));
        if held.contains(&Button::Mode) && !self.ignore.contains(&Button::Mode) && !self.padmap.contains_key("home") {
            self.ignore.insert(Button::Mode);
            self.menu_request = true;
        }
        let chord = [Button::Start, Button::Select];
        if chord.iter().all(|b| held.contains(b) && !self.ignore.contains(b)) {
            let since = *self.chord_since.get_or_insert_with(Instant::now);
            if since.elapsed() >= MENU_HOLD {
                // The game sees both RELEASED from here (not a stuck pair) - see `ctrl_frame`.
                self.ignore.extend(chord);
                self.chord_since = None;
                self.menu_request = true;
            }
        } else {
            self.chord_since = None;
        }
    }

    /// Whether the pad asked for the menu since the last call.
    pub fn take_menu_request(&mut self) -> bool {
        std::mem::take(&mut self.menu_request)
    }

    /// The menu closed and the pad is the game's again: whatever is held now (the press that
    /// closed the menu) is ignored until it is released, so it does not reach the game.
    pub fn hand_back(&mut self) {
        self.keys.clear();
        self.chord_since = None;
        if let Some((_, pad)) = self.gilrs.as_ref().and_then(|g| g.gamepads().find(|(_, p)| p.is_connected())) {
            self.ignore = ALL_BUTTONS.iter().copied().filter(|b| pad.is_pressed(*b)).collect();
        }
    }

    pub fn ctrl_frame(&self) -> CtrlFrame {
        let mut buttons = 0u32;
        for k in &self.keys {
            if let Some(b) = self.keymap.get(k) {
                buttons |= b;
            }
        }
        let mut lx = CENTER;
        let mut ly = CENTER;
        let mut rx = CENTER;
        let mut ry = CENTER;
        if let Some(g) = self.gilrs.as_ref()
            && let Some((_, pad)) = g.gamepads().next() {
                for (control, bits) in &self.padmap {
                    if let Some(b) = gilrs_button(control)
                        && pad.is_pressed(b)
                        && !self.ignore.contains(&b) {
                            buttons |= bits;
                        }
                }
                let (x, y) = dead(pad.value(Axis::LeftStickX), pad.value(Axis::LeftStickY), self.deadzone);
                lx = axis_to_byte(x, false);
                ly = axis_to_byte(y, true);
                let (x, y) = dead(pad.value(Axis::RightStickX), pad.value(Axis::RightStickY), self.deadzone);
                rx = axis_to_byte(x, false);
                ry = axis_to_byte(y, true);
            }
        CtrlFrame { buttons, lx, ly, rx, ry }
    }
}

/// Every pad button [`gilrs_button`] names - what the hand-over reads as held.
const ALL_BUTTONS: [Button; 17] = [
    Button::South,
    Button::East,
    Button::West,
    Button::North,
    Button::LeftTrigger,
    Button::RightTrigger,
    Button::LeftTrigger2,
    Button::RightTrigger2,
    Button::Select,
    Button::Start,
    Button::LeftThumb,
    Button::RightThumb,
    Button::DPadUp,
    Button::DPadDown,
    Button::DPadLeft,
    Button::DPadRight,
    Button::Mode,
];

/// The gilrs button at a Standard Gamepad position.
fn gilrs_button(control: &str) -> Option<Button> {
    Some(match control {
        "south" => Button::South,
        "east" => Button::East,
        "west" => Button::West,
        "north" => Button::North,
        "l1" => Button::LeftTrigger,
        "r1" => Button::RightTrigger,
        "l2" => Button::LeftTrigger2,
        "r2" => Button::RightTrigger2,
        "select" => Button::Select,
        "start" => Button::Start,
        "l3" => Button::LeftThumb,
        "r3" => Button::RightThumb,
        "dpad_up" => Button::DPadUp,
        "dpad_down" => Button::DPadDown,
        "dpad_left" => Button::DPadLeft,
        "dpad_right" => Button::DPadRight,
        "home" => Button::Mode,
        _ => return None,
    })
}

fn dead(x: f32, y: f32, zone: f32) -> (f32, f32) {
    if (x * x + y * y).sqrt() < zone { (0.0, 0.0) } else { (x, y) }
}

/// Map a gilrs axis (-1.0..1.0) to a Vita analog byte (0..255, 128 centered).
/// `invert` flips the sign so gilrs' up-positive Y matches the Vita's top-is-0.
fn axis_to_byte(v: f32, invert: bool) -> u8 {
    let v = if invert { -v } else { v };
    let scaled = (v + 1.0) * 0.5 * 255.0;
    scaled.round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analog_maps_center_and_extremes() {
        assert_eq!(axis_to_byte(0.0, false), CENTER);
        assert_eq!(axis_to_byte(1.0, false), 255);
        assert_eq!(axis_to_byte(-1.0, false), 0);
        assert_eq!(axis_to_byte(1.0, true), 0);
    }

    #[test]
    fn winit_key_names_are_the_w3c_codes_the_settings_use() {
        let s = Settings::default();
        let mut i = Input {
            keys: HashSet::new(),
            gilrs: None,
            keymap: BTreeMap::new(),
            padmap: BTreeMap::new(),
            deadzone: 0.1,
            ignore: HashSet::new(),
            chord_since: None,
            menu_request: false,
        };
        i.apply(&s);
        i.set_key(KeyCode::KeyZ, true);
        i.set_key(KeyCode::ArrowUp, true);
        let f = i.ctrl_frame();
        assert_eq!(f.buttons, 0x4000 | 0x10, "Z is cross and ArrowUp is up by default");
        assert!(vitaslop_frontend::input::GAMEPAD_CONTROLS.iter().all(|c| gilrs_button(c).is_some()));
        assert!(
            vitaslop_frontend::input::GAMEPAD_CONTROLS.iter().all(|c| ALL_BUTTONS.contains(&gilrs_button(c).unwrap())),
            "the hand-over reads every button a control can name"
        );
    }
}
