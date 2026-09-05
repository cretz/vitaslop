//! Live desktop input: keyboard (winit) and gamepad (gilrs) into one [`CtrlFrame`],
//! mapped by the person's settings.
//!
//! The settings name keys by W3C `KeyboardEvent.code` (`KeyZ`, `ArrowUp`) and pad
//! controls by Standard Gamepad position (`south`, `dpad_up`) - the vocabulary the
//! browser front end uses, so one remap serves both. winit's `KeyCode` debug names ARE
//! the W3C codes, and gilrs' buttons map onto the standard positions below.

use std::collections::{BTreeMap, HashSet};

use gilrs::{Axis, Button, Gilrs};
use vitaslop_frontend::input::invert;
use vitaslop_frontend::settings::Settings;
use vitaslop_runtime::CtrlFrame;
use winit::keyboard::KeyCode;

const CENTER: u8 = 128;

pub struct Input {
    keys: HashSet<String>,
    gilrs: Option<Gilrs>,
    /// `KeyboardEvent.code` -> button bits.
    keymap: BTreeMap<String, u32>,
    /// Standard Gamepad control -> button bits.
    padmap: BTreeMap<String, u32>,
    deadzone: f32,
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
        let mut me = Input { keys: HashSet::new(), gilrs, keymap: BTreeMap::new(), padmap: BTreeMap::new(), deadzone: 0.14 };
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

    /// Drain pending gilrs events so the gamepad state read below is current.
    pub fn pump_gamepad(&mut self) {
        if let Some(g) = self.gilrs.as_mut() {
            while let Some(ev) = g.next_event() {
                g.update(&ev);
            }
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
        if let Some(g) = self.gilrs.as_ref() {
            if let Some((_, pad)) = g.gamepads().next() {
                for (control, bits) in &self.padmap {
                    if let Some(b) = gilrs_button(control) {
                        if pad.is_pressed(b) {
                            buttons |= bits;
                        }
                    }
                }
                let (x, y) = dead(pad.value(Axis::LeftStickX), pad.value(Axis::LeftStickY), self.deadzone);
                lx = axis_to_byte(x, false);
                ly = axis_to_byte(y, true);
                let (x, y) = dead(pad.value(Axis::RightStickX), pad.value(Axis::RightStickY), self.deadzone);
                rx = axis_to_byte(x, false);
                ry = axis_to_byte(y, true);
            }
        }
        CtrlFrame { buttons, lx, ly, rx, ry }
    }
}

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
        let mut i = Input { keys: HashSet::new(), gilrs: None, keymap: BTreeMap::new(), padmap: BTreeMap::new(), deadzone: 0.1 };
        i.apply(&s);
        i.set_key(KeyCode::KeyZ, true);
        i.set_key(KeyCode::ArrowUp, true);
        let f = i.ctrl_frame();
        assert_eq!(f.buttons, 0x4000 | 0x10, "Z is cross and ArrowUp is up by default");
        assert!(vitaslop_frontend::input::GAMEPAD_CONTROLS.iter().all(|c| gilrs_button(c).is_some()));
    }
}
