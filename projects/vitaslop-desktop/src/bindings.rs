//! Remapping a control: the capture of the next key or pad button, and where the new binding
//! is kept.
//!
//! The settings page edits a draft and keeps it on Save, as the browser's form does. The
//! in-game menu cannot wait for a Save - a person remaps Cross because the game is unplayable
//! the way it is - so a binding captured there is applied to the running input at once and
//! written straight away. Written WHERE is the one decision here: a change made while playing
//! is a global preference (the browser's rule for every menu setting), EXCEPT a control this
//! title already overrides. That one goes into the title's own record, because writing it to
//! the global record would be a change the title's override hides - the next launch would put
//! the old key back, and the remap would look like it had not saved.

use serde_json::Value;
use vitaslop_frontend::input::Button;

/// Which of a control's two bindings is being captured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Keyboard,
    Gamepad,
}

impl Kind {
    /// The settings record's key for this map.
    pub fn field(self) -> &'static str {
        match self {
            Kind::Keyboard => "keyboard",
            Kind::Gamepad => "gamepad",
        }
    }
}

/// A capture in progress: the next key (or pad button) pressed becomes `button`'s binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capture {
    pub button: Button,
    pub kind: Kind,
    /// Captured for the running game (the in-game menu) rather than the settings draft.
    pub live: bool,
}

/// Where a binding captured in the in-game menu is written - see the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Title,
    Global,
}

/// The record `button`'s `kind` binding belongs in, given the title's settings patch.
pub fn target(title_patch: Option<&Value>, kind: Kind, button: Button) -> Target {
    let overridden = title_patch.and_then(|p| p.get(kind.field())).and_then(|m| m.get(button.name())).is_some_and(|v| !v.is_null());
    if overridden { Target::Title } else { Target::Global }
}

/// Set `button`'s `kind` binding to `value` inside a settings record or patch, creating the
/// map if the record has none.
pub fn set_leaf(record: &mut Value, kind: Kind, button: Button, value: &str) {
    if !record.is_object() {
        *record = Value::Object(Default::default());
    }
    let map = record.as_object_mut().unwrap().entry(kind.field()).or_insert_with(|| Value::Object(Default::default()));
    if !map.is_object() {
        *map = Value::Object(Default::default());
    }
    map.as_object_mut().unwrap().insert(button.name().to_string(), Value::String(value.to_string()));
}

/// Keep a binding captured in the in-game menu for `title_id` - see the module docs.
pub fn save_live(title_id: &str, kind: Kind, button: Button, value: &str) -> std::io::Result<Target> {
    let patch = crate::library::title_patch(title_id);
    let to = target(patch.as_ref(), kind, button);
    match to {
        Target::Title => {
            let mut p = patch.unwrap_or(Value::Null);
            set_leaf(&mut p, kind, button, value);
            crate::library::save_title_patch(title_id, Some(&p))?;
        }
        Target::Global => {
            let mut g = crate::library::effective(None);
            match kind {
                Kind::Keyboard => g.keyboard.insert(button.name().to_string(), value.to_string()),
                Kind::Gamepad => g.gamepad.insert(button.name().to_string(), value.to_string()),
            };
            crate::library::save_global_settings(&g)?;
        }
    }
    Ok(to)
}

/// The settings-file name of a winit key - the W3C `KeyboardEvent.code` the browser stores
/// (`KeyZ`, `ArrowUp`), which is exactly winit's `KeyCode` debug name.
pub fn key_name(code: winit::keyboard::KeyCode) -> String {
    format!("{code:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_remap_goes_where_the_running_binding_came_from() {
        assert_eq!(target(None, Kind::Keyboard, Button::Cross), Target::Global, "no title record at all");
        let patch = json!({ "keyboard": { "cross": "KeyJ" }, "showFps": true });
        assert_eq!(target(Some(&patch), Kind::Keyboard, Button::Cross), Target::Title, "the title overrides this key");
        assert_eq!(target(Some(&patch), Kind::Keyboard, Button::Circle), Target::Global, "a key the title leaves alone");
        assert_eq!(target(Some(&patch), Kind::Gamepad, Button::Cross), Target::Global, "the other map is separate");
        let cleared = json!({ "keyboard": { "cross": null } });
        assert_eq!(target(Some(&cleared), Kind::Keyboard, Button::Cross), Target::Global, "a null leaf is no override");
    }

    #[test]
    fn a_leaf_is_set_without_disturbing_the_rest() {
        let mut v = json!({ "keyboard": { "circle": "KeyX" }, "showFps": true });
        set_leaf(&mut v, Kind::Keyboard, Button::Cross, "KeyJ");
        set_leaf(&mut v, Kind::Gamepad, Button::Start, "home");
        assert_eq!(v, json!({ "keyboard": { "circle": "KeyX", "cross": "KeyJ" }, "gamepad": { "start": "home" }, "showFps": true }));
        let mut empty = Value::Null;
        set_leaf(&mut empty, Kind::Keyboard, Button::L, "KeyQ");
        assert_eq!(empty, json!({ "keyboard": { "l": "KeyQ" } }));
    }

    #[test]
    fn key_names_are_the_codes_the_settings_store() {
        assert_eq!(key_name(winit::keyboard::KeyCode::KeyZ), "KeyZ");
        assert_eq!(key_name(winit::keyboard::KeyCode::ShiftRight), "ShiftRight");
    }
}
