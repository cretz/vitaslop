//! The settings record, its defaults, and the global-plus-per-title merge.
//!
//! One record shape everywhere. The GLOBAL settings are a full record; a TITLE's
//! settings are a JSON patch over it - only the keys the person changed for that
//! title, so a later change to the global default reaches every title that did not
//! override it. [`effective`] is the merge, and it is the only place the rule lives.
//!
//! The record is also what a run is configured FROM: [`Settings::run_knobs`] turns it
//! into the `VITASLOP_*` map the engine reads, so the page and the desktop cannot
//! disagree about which knob a checkbox means.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Where the on-screen pad goes, on a touch device.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PadMode {
    /// Portrait: below the screen. Landscape: over it. The default.
    Auto,
    /// Always drawn over the game.
    Overlay,
    /// Always beside/below the game, which is drawn smaller.
    Beside,
    /// Never shown (a gamepad or keyboard is in use).
    Hidden,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct PadSettings {
    pub mode: PadMode,
    /// 0..1, the overlay's opacity over the game.
    pub opacity: f32,
    /// Size multiplier for the on-screen controls.
    pub scale: f32,
    /// Haptic tick on a press, where the device offers one.
    pub vibrate: bool,
}

impl Default for PadSettings {
    fn default() -> Self {
        PadSettings { mode: PadMode::Auto, opacity: 0.55, scale: 1.0, vibrate: true }
    }
}

/// How the 960x544 panel is fitted to the screen.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Scaling {
    /// Largest size that fits, smooth resample. The default: a phone lays the panel
    /// out at a non-integer ratio, where nearest-neighbour is uneven.
    Fit,
    /// Largest INTEGER multiple that fits, nearest-neighbour. Crisp on a desktop.
    Integer,
    /// Fill the screen, ignoring the aspect ratio.
    Stretch,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct Settings {
    /// Leaving the page or losing the window's focus stops the emulator (not the
    /// title's own pause menu). On, as a device in a pocket would be.
    pub pause_on_blur: bool,
    /// The frame-rate overlay on the game.
    pub show_fps: bool,
    /// Desktop only: the frame rate in the window's title bar.
    pub fps_in_title: bool,
    pub scaling: Scaling,
    pub pad: PadSettings,
    /// Button name -> `KeyboardEvent.code`.
    pub keyboard: BTreeMap<String, String>,
    /// Button name -> Standard Gamepad control (see `input::GAMEPAD_CONTROLS`).
    pub gamepad: BTreeMap<String, String>,
    /// Stick dead zone, 0..1 of full deflection.
    pub stick_deadzone: f32,
    /// Which game-data profile a run saves into. `"default"` is the unnamed one.
    pub profile: String,
    /// Advanced: `VITASLOP_*` knobs merged over the base set, verbatim.
    pub knobs: BTreeMap<String, String>,
    /// Advanced: a scripted-input recipe's text, replayed from the first frame.
    pub recipe: String,
    /// Advanced: run unpaced and unpresented to this display frame first.
    pub fast_forward: u32,
    /// Advanced: time every host call (roughly doubles the frame cost).
    pub debug_capture: bool,
    /// Advanced: mirror the run's notes to the console.
    pub console_notes: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            pause_on_blur: true,
            show_fps: false,
            fps_in_title: false,
            scaling: Scaling::Fit,
            pad: PadSettings::default(),
            keyboard: crate::input::default_keyboard(),
            gamepad: crate::input::default_gamepad(),
            stick_deadzone: 0.14,
            profile: "default".to_string(),
            knobs: BTreeMap::new(),
            recipe: String::new(),
            fast_forward: 0,
            debug_capture: false,
            console_notes: false,
        }
    }
}

/// The knobs every run starts from. These were the launcher's per-title defaults
/// for every run it ever made, so they are the product's defaults.
pub fn base_knobs() -> BTreeMap<String, String> {
    [("VITASLOP_FRAME_TOPUP", "0"), ("VITASLOP_GXP_LIVE", "1"), ("VITASLOP_LOG", "warn")]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

impl Settings {
    pub fn from_value(v: &Value) -> Settings {
        serde_json::from_value(v.clone()).unwrap_or_default()
    }

    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// The `VITASLOP_*` map a run of these settings is configured with: the base set,
    /// then the checkbox-backed knobs, then the person's own knobs LAST so the
    /// advanced box can override anything.
    pub fn run_knobs(&self) -> BTreeMap<String, String> {
        let mut k = base_knobs();
        if self.debug_capture {
            k.insert("VITASLOP_DEBUG_CAPTURE".into(), "1".into());
        }
        if self.fast_forward > 0 {
            k.insert("VITASLOP_BROWSER_FASTFORWARD".into(), self.fast_forward.to_string());
        }
        if !self.pause_on_blur {
            k.insert("VITASLOP_PAUSE_ON_BLUR".into(), "0".into());
        }
        if self.console_notes {
            k.insert("VITASLOP_CONSOLE".into(), "1".into());
        }
        for (name, value) in &self.knobs {
            k.insert(name.clone(), value.clone());
        }
        k
    }
}

/// Deep-merge `patch` over `base`: objects recurse, anything else replaces, and a
/// `null` in the patch removes the key (so a title can clear a global knob).
pub fn merge(base: &Value, patch: &Value) -> Value {
    match (base, patch) {
        (Value::Object(b), Value::Object(p)) => {
            let mut out = b.clone();
            for (k, v) in p {
                if v.is_null() {
                    out.remove(k);
                } else {
                    let merged = match out.get(k) {
                        Some(existing) => merge(existing, v),
                        None => v.clone(),
                    };
                    out.insert(k.clone(), merged);
                }
            }
            Value::Object(out)
        }
        (_, p) => p.clone(),
    }
}

/// The settings a run of one title uses: defaults, then the global record, then the
/// title's patch.
pub fn effective(global: &Value, title_patch: Option<&Value>) -> Settings {
    let mut v = merge(&Settings::default().to_value(), global);
    if let Some(p) = title_patch {
        v = merge(&v, p);
    }
    Settings::from_value(&v)
}

/// Parse a `NAME=VALUE` per line knobs box into a map. Lines that are not a knob are
/// ignored; a name is `VITASLOP_`-prefixed upper-case.
pub fn parse_knobs(text: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some((k, v)) = line.split_once('=') {
            let k = k.trim();
            if !k.is_empty() && k.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_') {
                out.insert(k.to_string(), v.trim().to_string());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_round_trip_and_a_title_patch_wins() {
        let d = Settings::default();
        assert_eq!(Settings::from_value(&d.to_value()), d);
        let global = json!({ "showFps": true, "knobs": { "VITASLOP_X": "1" } });
        let patch = json!({ "showFps": false, "knobs": { "VITASLOP_X": null, "VITASLOP_Y": "2" } });
        let e = effective(&global, Some(&patch));
        assert!(!e.show_fps);
        assert!(e.pause_on_blur, "an untouched default survives both layers");
        assert_eq!(e.knobs.get("VITASLOP_X"), None);
        assert_eq!(e.knobs.get("VITASLOP_Y").map(String::as_str), Some("2"));
        let k = e.run_knobs();
        assert_eq!(k["VITASLOP_GXP_LIVE"], "1");
        assert_eq!(k["VITASLOP_Y"], "2");
    }

    #[test]
    fn an_unknown_field_in_a_stored_record_does_not_reset_everything() {
        let v = json!({ "showFps": true, "somethingFromTheFuture": 3 });
        assert!(Settings::from_value(&v).show_fps);
    }

    #[test]
    fn knob_lines_parse() {
        let k = parse_knobs("VITASLOP_A=1\n  junk\nVITASLOP_B = two words \n");
        assert_eq!(k.len(), 2);
        assert_eq!(k["VITASLOP_B"], "two words");
    }
}
