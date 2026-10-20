#![cfg(not(target_arch = "wasm32"))]
//! Every committed playbook COMPILES, and is filed under a title-id directory; a best-effort one
//! also says when it last played through.
//!
//! Needs no game: a syntax error in a playbook otherwise surfaces only when someone next runs
//! the title - a boot, a transpile and minutes away.

use vitaslop_playbook::lua_mlua::MluaVm;
use vitaslop_playbook::{all_playbooks, is_best_effort, playbook_name, vm};

#[test]
fn every_committed_playbook_compiles() {
    let all = all_playbooks();
    assert!(!all.is_empty(), "no playbooks under {}", vitaslop_playbook::crate_root().join("playbooks").display());
    let mut bad = Vec::new();
    for (title, path) in &all {
        let name = playbook_name(path);
        if title.len() != 9 || !title.chars().all(|c| c.is_ascii_alphanumeric()) {
            bad.push(format!("{name}: filed under {title:?}, which is not a title id"));
        }
        let src = std::fs::read_to_string(path).unwrap();
        if is_best_effort(path) && !src.lines().any(|l| l.starts_with("-- last verified: ")) {
            bad.push(format!("{name}: a best-effort playbook needs a `-- last verified: <date> <build>` line"));
        }
        if let Err(e) = vm::load(&mut MluaVm::new(), &name, &src) {
            bad.push(format!("{name}: {e}"));
        }
    }
    assert!(bad.is_empty(), "{} playbooks checked:\n  {}", all.len(), bad.join("\n  "));
}
