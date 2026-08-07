//! Process-wide overrides for the `VITASLOP_*` knobs, for a platform that has no
//! environment to read them from.
//!
//! # Why this exists, and why it lives in the LOWEST crate
//! `wasm32-unknown-unknown` has no environment at all: `std::env::var` always returns
//! `NotPresent` and `std::env::set_var` fails outright. So in the browser every knob
//! reads as unset, and there is no way to tell that apart from "the knob is off". That
//! is not a diagnostic inconvenience - it silently changes what the emulator IS. The
//! renderer's master switch `VITASLOP_GXP_LIVE` is read this way, so before this table
//! existed the browser could only ever draw the fixed-function APPROXIMATION, while the
//! desktop oracle it is supposed to match drew the guest's real shaders. Two different
//! renderers, no message, and a browser frame that looked plausible and was wrong.
//!
//! The table lives here rather than in `vitaslop-runtime` because the readers span
//! crates in both directions: `vitaslop-runtime` depends on `vitaslop-platform`, so the
//! renderer in [`crate::gpu`] cannot reach a table owned by the runtime.
//! `vitaslop_runtime::knobs` re-exports these so the public API is unchanged, and keeps
//! the generated `KNOBS.md` index, which needs the whole workspace on disk.
//!
//! # Fail loudly, never partially
//! [`set_override`] PANICS on a name whose reader still calls `std::env::var` directly.
//! A silently-ignored override would leave the caller believing it had configured a run
//! it had not - the exact failure this module exists to prevent. A name earns its place
//! in [`OVERRIDABLE`] only once its reader goes through [`var`] / [`var_os`] / [`flag`].

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Knobs whose readers go through this module, so [`set_override`] can reach them.
///
/// Grouped by what a browser run actually needs:
/// - `VITASLOP_FRAME_TOPUP` - one retail racer never finishes loading without it.
/// - `VITASLOP_GXP_*` - the shader recompiler's master switch and its diagnostics. Without
///   `VITASLOP_GXP_LIVE` the browser renders a different picture than the desktop oracle.
/// - `VITASLOP_BROWSER_*` - knobs only the browser build reads; they have no environment
///   to come from by construction.
pub const OVERRIDABLE: &[&str] = &[
    "VITASLOP_ALLOW_SOFTWARE_GPU",
    "VITASLOP_BROWSER_FASTFORWARD",
    "VITASLOP_BROWSER_FUEL",
    "VITASLOP_BROWSER_HEARTBEAT_MS",
    "VITASLOP_BROWSER_QUANTUM_CALLS",
    "VITASLOP_BROWSER_SUPERSAMPLE",
    "VITASLOP_FRAME_TOPUP",
    // The clock's core model, so a browser run can be A/B'd against native without an
    // environment to set it in.
    "VITASLOP_GUEST_CORES",
    "VITASLOP_GXP_ALLOW_FIXED_FUNCTION",
    "VITASLOP_GXP_DUMP",
    "VITASLOP_GXP_EXCLUDE",
    "VITASLOP_GXP_FORCE",
    "VITASLOP_GXP_KEYS",
    "VITASLOP_GXP_LIVE",
    "VITASLOP_GXP_NEGW",
    "VITASLOP_GXP_NOBLEND",
    "VITASLOP_GXP_NODEPTH",
    "VITASLOP_GXP_ONLY",
    "VITASLOP_GXP_SOLID",
    "VITASLOP_GXP_YFLIP",
    "VITASLOP_GXP_ZFIX",
    "VITASLOP_LOG",
    "VITASLOP_PERF",
    "VITASLOP_TEX_CACHE_MB",
];

/// The override map, consulted by every reader in this module BEFORE the environment.
fn overrides() -> &'static Mutex<HashMap<String, String>> {
    static CELL: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Read a knob: the process-wide override if one is set, else the environment.
pub fn var(name: &str) -> Result<String, std::env::VarError> {
    if let Some(v) = overrides().lock().unwrap_or_else(|e| e.into_inner()).get(name) {
        return Ok(v.clone());
    }
    std::env::var(name)
}

/// Whether a knob is SET at all, regardless of value - the `std::env::var_os(..).is_some()`
/// shape most flags here use.
pub fn var_os(name: &str) -> Option<String> {
    if let Some(v) = overrides().lock().unwrap_or_else(|e| e.into_inner()).get(name) {
        return Some(v.clone());
    }
    std::env::var_os(name).map(|v| v.to_string_lossy().into_owned())
}

/// A presence flag: set (to anything, including the empty string) means on.
///
/// This is the house convention for a boolean knob, and it is deliberately not
/// value-sensitive: `NAME=0` still reads as ON, the same as it does through the shell.
pub fn flag(name: &str) -> bool {
    var_os(name).is_some()
}

/// Set a knob for this process, for a platform where the environment cannot be.
///
/// Panics on a name not in [`OVERRIDABLE`] - see the module docs for why a silently
/// ignored override is worse than a missing one.
pub fn set_override(name: &str, value: &str) {
    assert!(
        OVERRIDABLE.contains(&name),
        "{name} is not overridable - its reader still calls std::env::var directly. \
         Route that reader through vitaslop_platform::knobs and add the name to \
         vitaslop_platform::knobs::OVERRIDABLE."
    );
    overrides()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(name.to_string(), value.to_string());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The list is the contract `set_override` enforces, so a duplicate or an unsorted
    /// entry is a maintenance hazard rather than a style point: both make "is this name
    /// already here?" answerable only by reading every line.
    #[test]
    fn overridable_is_sorted_and_unique() {
        let mut sorted = OVERRIDABLE.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted, OVERRIDABLE);
    }

    #[test]
    fn an_override_is_visible_to_every_reader_shape() {
        set_override("VITASLOP_GXP_LIVE", "1");
        assert_eq!(var("VITASLOP_GXP_LIVE").as_deref(), Ok("1"));
        assert_eq!(var_os("VITASLOP_GXP_LIVE").as_deref(), Some("1"));
        assert!(flag("VITASLOP_GXP_LIVE"));
    }

    /// The empty string is how a shell spells "set but valueless", and the house flag
    /// convention is presence, not truthiness - so it must read as ON.
    #[test]
    fn an_empty_override_still_reads_as_set() {
        set_override("VITASLOP_GXP_SOLID", "");
        assert!(flag("VITASLOP_GXP_SOLID"));
    }

    #[test]
    #[should_panic(expected = "not overridable")]
    fn setting_an_unrouted_knob_panics() {
        set_override("VITASLOP_NOT_ROUTED_ANYWHERE", "1");
    }
}
