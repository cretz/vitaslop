//! Playbooks: Lua scripts that play a title by reacting to it, and the tapes they record.
//!
//! See the crate README for the format, the API and how to run one. In short:
//!
//! - [`engine`] runs one playbook (or replays one tape) against a running title, ticking on
//!   every input frame through the runtime's host-call observer
//!   ([`vitaslop_runtime::observer`]).
//! - [`cue`] is what a playbook can inspect and wait on; [`calls`] counts the host calls a
//!   call cue names, lock-free, from any guest thread.
//! - [`vm`] is the Lua seam: a backend runs a chunk and calls a string function, nothing more,
//!   so the Lua implementation can be swapped. [`lua_mlua`] is the native one.
//! - [`tape`] reads and writes tapes and diffs their timelines.
//!
//! Hosts (the desktop headless runner, the browser) start a run, report each display frame,
//! route the pad through [`engine::input`], write the shots [`engine::take_events`] asks for,
//! and call [`engine::finish`] for the report.
pub mod calls;
pub mod cue;
pub mod engine;
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub mod lua_mlua;
pub mod proto;
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub mod session;
pub mod tape;
pub mod vm;

use std::path::{Path, PathBuf};

/// The crate root, where `playbooks/` lives.
pub fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The tier directory under `playbooks/` for playbooks nobody keeps green: each says when it last
/// played through (`-- last verified: ...`), and a later build is allowed to leave it behind.
pub const BEST_EFFORT_DIR: &str = "best-effort";

/// Every committed playbook, as `(title id, path)`: `playbooks/<TITLE_ID>/*.lua` and
/// `playbooks/best-effort/<TITLE_ID>/*.lua`.
pub fn all_playbooks() -> Vec<(String, PathBuf)> {
    let root = crate_root().join("playbooks");
    let mut out = Vec::new();
    collect_title_dirs(&root, &mut out);
    collect_title_dirs(&root.join(BEST_EFFORT_DIR), &mut out);
    out.sort();
    out
}

fn collect_title_dirs(root: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(dirs) = std::fs::read_dir(root) else { return };
    for d in dirs.flatten() {
        let title = d.file_name().to_string_lossy().to_string();
        if title == BEST_EFFORT_DIR {
            continue;
        }
        let Ok(files) = std::fs::read_dir(d.path()) else { continue };
        for f in files.flatten() {
            if f.path().extension().is_some_and(|e| e == "lua") {
                out.push((title.clone(), f.path()));
            }
        }
    }
}

/// Whether `path` is filed in the best-effort tier.
pub fn is_best_effort(path: &Path) -> bool {
    path.parent().and_then(|p| p.parent()).and_then(|p| p.file_name()).is_some_and(|n| n == BEST_EFFORT_DIR)
}

/// A playbook's name in errors and reports: `<TITLE_ID>/<file>` when it lives under a title
/// directory, else its file name.
pub fn playbook_name(path: &Path) -> String {
    let file = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
    match path.parent().and_then(|p| p.file_name()) {
        Some(dir) => format!("{}/{file}", dir.to_string_lossy()),
        None => file,
    }
}
