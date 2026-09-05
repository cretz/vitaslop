//! The Vita's digital buttons, and the default maps from a keyboard and a gamepad.
//!
//! Names, not bits, cross the platform seam: a settings file says `"cross":
//! "KeyZ"`, and each front end resolves the key by its own API. The keyboard
//! vocabulary is the W3C `KeyboardEvent.code` set (`KeyZ`, `ArrowUp`, `ShiftRight`),
//! which the browser hands out directly and winit's `KeyCode` names match. The
//! gamepad vocabulary is the W3C Standard Gamepad layout by POSITION (`south`,
//! `east`, `l1`, `dpad_up`), which the browser's `Gamepad` API indexes and gilrs
//! names the same way - so one map serves both, and a person's remap on the phone
//! is the same remap on the desktop.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Button {
    Up,
    Down,
    Left,
    Right,
    Cross,
    Circle,
    Square,
    Triangle,
    L,
    R,
    Start,
    Select,
}

impl Button {
    pub const ALL: [Button; 12] = [
        Button::Up,
        Button::Down,
        Button::Left,
        Button::Right,
        Button::Cross,
        Button::Circle,
        Button::Square,
        Button::Triangle,
        Button::L,
        Button::R,
        Button::Start,
        Button::Select,
    ];

    /// The settings-file name.
    pub fn name(self) -> &'static str {
        match self {
            Button::Up => "up",
            Button::Down => "down",
            Button::Left => "left",
            Button::Right => "right",
            Button::Cross => "cross",
            Button::Circle => "circle",
            Button::Square => "square",
            Button::Triangle => "triangle",
            Button::L => "l",
            Button::R => "r",
            Button::Start => "start",
            Button::Select => "select",
        }
    }

    pub fn from_name(name: &str) -> Option<Button> {
        Button::ALL.into_iter().find(|b| b.name() == name)
    }

    /// The `SceCtrl` bit.
    pub fn bit(self) -> u32 {
        match self {
            Button::Select => 0x0000_0001,
            Button::Start => 0x0000_0008,
            Button::Up => 0x0000_0010,
            Button::Right => 0x0000_0020,
            Button::Down => 0x0000_0040,
            Button::Left => 0x0000_0080,
            Button::L => 0x0000_0100,
            Button::R => 0x0000_0200,
            Button::Triangle => 0x0000_1000,
            Button::Circle => 0x0000_2000,
            Button::Cross => 0x0000_4000,
            Button::Square => 0x0000_8000,
        }
    }

    /// What a settings screen prints beside the control.
    pub fn label(self) -> &'static str {
        match self {
            Button::Up => "D-pad up",
            Button::Down => "D-pad down",
            Button::Left => "D-pad left",
            Button::Right => "D-pad right",
            Button::Cross => "Cross",
            Button::Circle => "Circle",
            Button::Square => "Square",
            Button::Triangle => "Triangle",
            Button::L => "L",
            Button::R => "R",
            Button::Start => "Start",
            Button::Select => "Select",
        }
    }
}

/// Default keyboard map, by `KeyboardEvent.code`. Faces on ZXAS (the desktop's
/// long-standing layout), with WASD free for nothing - the d-pad is the arrows.
pub fn default_keyboard() -> BTreeMap<String, String> {
    [
        (Button::Up, "ArrowUp"),
        (Button::Down, "ArrowDown"),
        (Button::Left, "ArrowLeft"),
        (Button::Right, "ArrowRight"),
        (Button::Cross, "KeyZ"),
        (Button::Circle, "KeyX"),
        (Button::Square, "KeyA"),
        (Button::Triangle, "KeyS"),
        (Button::L, "KeyQ"),
        (Button::R, "KeyE"),
        (Button::Start, "Enter"),
        (Button::Select, "ShiftRight"),
    ]
    .into_iter()
    .map(|(b, k)| (b.name().to_string(), k.to_string()))
    .collect()
}

/// The Standard Gamepad's controls, by position, in the order the browser indexes
/// them (`Gamepad.buttons[i]`). The index IS the position in this table.
pub const GAMEPAD_CONTROLS: [&str; 17] = [
    "south", "east", "west", "north", "l1", "r1", "l2", "r2", "select", "start", "l3", "r3",
    "dpad_up", "dpad_down", "dpad_left", "dpad_right", "home",
];

/// Default gamepad map: the Vita's face layout is the Standard Gamepad's by
/// position (cross south, circle east, square west, triangle north).
pub fn default_gamepad() -> BTreeMap<String, String> {
    [
        (Button::Up, "dpad_up"),
        (Button::Down, "dpad_down"),
        (Button::Left, "dpad_left"),
        (Button::Right, "dpad_right"),
        (Button::Cross, "south"),
        (Button::Circle, "east"),
        (Button::Square, "west"),
        (Button::Triangle, "north"),
        (Button::L, "l1"),
        (Button::R, "r1"),
        (Button::Start, "start"),
        (Button::Select, "select"),
    ]
    .into_iter()
    .map(|(b, k)| (b.name().to_string(), k.to_string()))
    .collect()
}

/// Invert a `button -> control` map into `control -> bits`, which is the shape a
/// per-event lookup wants. Two buttons on one control both fire.
pub fn invert(map: &BTreeMap<String, String>) -> BTreeMap<String, u32> {
    let mut out: BTreeMap<String, u32> = BTreeMap::new();
    for (button, control) in map {
        if let Some(b) = Button::from_name(button) {
            *out.entry(control.clone()).or_insert(0) |= b.bit();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_are_disjoint_and_every_button_is_mapped_by_default() {
        let mut all = 0;
        for b in Button::ALL {
            assert_eq!(all & b.bit(), 0);
            all |= b.bit();
            assert_eq!(Button::from_name(b.name()), Some(b));
        }
        let kb = default_keyboard();
        let gp = default_gamepad();
        for b in Button::ALL {
            assert!(kb.contains_key(b.name()));
            assert!(GAMEPAD_CONTROLS.contains(&gp[b.name()].as_str()));
        }
        assert_eq!(invert(&kb)["KeyZ"], Button::Cross.bit());
    }
}
