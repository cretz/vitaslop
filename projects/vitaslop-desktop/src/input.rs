//! Real input -> SceCtrl. Collapses the host keyboard (winit) and any connected
//! gamepad (gilrs) into one [`CtrlFrame`] - the exact port-agnostic frame the
//! `World::poll_ctrl` seam hands the guest. Today the guest runs to completion
//! before the window opens, so this frame drives the title readout and playback;
//! once the cooperative scheduler lets the guest yield per frame, the same frame
//! feeds `poll_ctrl` live with no change to this mapping.

use std::collections::HashSet;

use gilrs::{Axis, Button, Gilrs};
use vitaslop_runtime::CtrlFrame;
use winit::keyboard::KeyCode;

// SceCtrlButtons bits (from the Vita headers). The cube reads this bitmask.
pub const SCE_CTRL_SELECT: u32 = 0x0000_0001;
pub const SCE_CTRL_START: u32 = 0x0000_0008;
pub const SCE_CTRL_UP: u32 = 0x0000_0010;
pub const SCE_CTRL_RIGHT: u32 = 0x0000_0020;
pub const SCE_CTRL_DOWN: u32 = 0x0000_0040;
pub const SCE_CTRL_LEFT: u32 = 0x0000_0080;
pub const SCE_CTRL_LTRIGGER: u32 = 0x0000_0100;
pub const SCE_CTRL_RTRIGGER: u32 = 0x0000_0200;
pub const SCE_CTRL_TRIANGLE: u32 = 0x0000_1000;
pub const SCE_CTRL_CIRCLE: u32 = 0x0000_2000;
pub const SCE_CTRL_CROSS: u32 = 0x0000_4000;
pub const SCE_CTRL_SQUARE: u32 = 0x0000_8000;

/// Analog neutral (sticks report 0..255 with 128 centered).
const CENTER: u8 = 128;

/// Live input, updated from window events and polled each frame. Owns the gilrs
/// context so gamepad state is pumped here, next to the keyboard state, and read
/// out as one merged [`CtrlFrame`].
pub struct Input {
    keys: HashSet<KeyCode>,
    gilrs: Option<Gilrs>,
}

impl Input {
    pub fn new() -> Self {
        // gilrs init can fail on a headless box with no input subsystem; a missing
        // gamepad is not fatal, the keyboard still drives the pad.
        let gilrs = match Gilrs::new() {
            Ok(g) => Some(g),
            Err(e) => {
                eprintln!("gamepad support unavailable ({e}); keyboard only");
                None
            }
        };
        Input { keys: HashSet::new(), gilrs }
    }

    /// Record a key press/release from a winit keyboard event.
    pub fn set_key(&mut self, key: KeyCode, pressed: bool) {
        if pressed {
            self.keys.insert(key);
        } else {
            self.keys.remove(&key);
        }
    }

    pub fn key_held(&self, key: KeyCode) -> bool {
        self.keys.contains(&key)
    }

    /// Drain pending gilrs events so the gamepad state read below is current.
    pub fn pump_gamepad(&mut self) {
        if let Some(g) = self.gilrs.as_mut() {
            while let Some(ev) = g.next_event() {
                g.update(&ev);
            }
        }
    }

    /// The merged controller frame this instant: keyboard OR'd with the first
    /// connected gamepad. Keyboard sets digital buttons; the gamepad additionally
    /// supplies the analog sticks (the keyboard has none).
    pub fn ctrl_frame(&self) -> CtrlFrame {
        let mut buttons = self.keyboard_buttons();
        let mut lx = CENTER;
        let mut ly = CENTER;
        let mut rx = CENTER;
        let mut ry = CENTER;

        if let Some(g) = self.gilrs.as_ref() {
            if let Some((_, pad)) = g.gamepads().next() {
                buttons |= gamepad_buttons(&pad);
                lx = axis_to_byte(pad.value(Axis::LeftStickX), false);
                ly = axis_to_byte(pad.value(Axis::LeftStickY), true);
                rx = axis_to_byte(pad.value(Axis::RightStickX), false);
                ry = axis_to_byte(pad.value(Axis::RightStickY), true);
            }
        }

        CtrlFrame { buttons, lx, ly, rx, ry }
    }

    /// The digital buttons from the keyboard. Left column is a keyboard-native
    /// layout; the face buttons follow the common Vita convention.
    fn keyboard_buttons(&self) -> u32 {
        let mut b = 0;
        let mut set = |held: bool, bit: u32| {
            if held {
                b |= bit;
            }
        };
        set(self.key_held(KeyCode::ArrowUp), SCE_CTRL_UP);
        set(self.key_held(KeyCode::ArrowDown), SCE_CTRL_DOWN);
        set(self.key_held(KeyCode::ArrowLeft), SCE_CTRL_LEFT);
        set(self.key_held(KeyCode::ArrowRight), SCE_CTRL_RIGHT);
        set(self.key_held(KeyCode::KeyZ), SCE_CTRL_CROSS);
        set(self.key_held(KeyCode::KeyX), SCE_CTRL_CIRCLE);
        set(self.key_held(KeyCode::KeyA), SCE_CTRL_SQUARE);
        set(self.key_held(KeyCode::KeyS), SCE_CTRL_TRIANGLE);
        set(self.key_held(KeyCode::KeyQ), SCE_CTRL_LTRIGGER);
        set(self.key_held(KeyCode::KeyE), SCE_CTRL_RTRIGGER);
        set(self.key_held(KeyCode::Enter), SCE_CTRL_START);
        set(self.key_held(KeyCode::ShiftRight), SCE_CTRL_SELECT);
        set(self.key_held(KeyCode::ShiftLeft), SCE_CTRL_SELECT);
        b
    }
}

/// The digital buttons from a gamepad this instant.
fn gamepad_buttons(pad: &gilrs::Gamepad) -> u32 {
    let mut b = 0;
    let mut set = |held: bool, bit: u32| {
        if held {
            b |= bit;
        }
    };
    set(pad.is_pressed(Button::DPadUp), SCE_CTRL_UP);
    set(pad.is_pressed(Button::DPadDown), SCE_CTRL_DOWN);
    set(pad.is_pressed(Button::DPadLeft), SCE_CTRL_LEFT);
    set(pad.is_pressed(Button::DPadRight), SCE_CTRL_RIGHT);
    set(pad.is_pressed(Button::South), SCE_CTRL_CROSS);
    set(pad.is_pressed(Button::East), SCE_CTRL_CIRCLE);
    set(pad.is_pressed(Button::West), SCE_CTRL_SQUARE);
    set(pad.is_pressed(Button::North), SCE_CTRL_TRIANGLE);
    set(pad.is_pressed(Button::LeftTrigger) || pad.is_pressed(Button::LeftTrigger2), SCE_CTRL_LTRIGGER);
    set(pad.is_pressed(Button::RightTrigger) || pad.is_pressed(Button::RightTrigger2), SCE_CTRL_RTRIGGER);
    set(pad.is_pressed(Button::Start), SCE_CTRL_START);
    set(pad.is_pressed(Button::Select), SCE_CTRL_SELECT);
    b
}

/// Map a gilrs axis (-1.0..1.0) to a Vita analog byte (0..255, 128 centered).
/// `invert` flips the sign so gilrs' up-positive Y matches the Vita's top-is-0.
/// The full-range map sends -1 -> 0, 0 -> 128 (neutral), +1 -> 255.
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
        // Neutral stick reads as the CtrlFrame default center.
        assert_eq!(axis_to_byte(0.0, false), CENTER);
        assert_eq!(axis_to_byte(0.0, true), CENTER);
        // Full deflection reaches the byte extremes.
        assert_eq!(axis_to_byte(1.0, false), 255);
        assert_eq!(axis_to_byte(-1.0, false), 0);
        // Inversion flips which extreme each direction hits (gilrs Y is up-positive,
        // the Vita's is top-is-0), and stays clamped in range.
        assert_eq!(axis_to_byte(1.0, true), 0);
        assert_eq!(axis_to_byte(-1.0, true), 255);
    }

    #[test]
    fn ctrl_button_bits_are_disjoint() {
        // The bits we OR together must not overlap, or two mapped inputs would
        // alias onto one button.
        let bits = [
            SCE_CTRL_SELECT,
            SCE_CTRL_START,
            SCE_CTRL_UP,
            SCE_CTRL_RIGHT,
            SCE_CTRL_DOWN,
            SCE_CTRL_LEFT,
            SCE_CTRL_LTRIGGER,
            SCE_CTRL_RTRIGGER,
            SCE_CTRL_TRIANGLE,
            SCE_CTRL_CIRCLE,
            SCE_CTRL_CROSS,
            SCE_CTRL_SQUARE,
        ];
        let mut seen = 0u32;
        for b in bits {
            assert_eq!(b.count_ones(), 1, "each SceCtrl button is a single bit");
            assert_eq!(seen & b, 0, "SceCtrl button bits overlap");
            seen |= b;
        }
    }
}
