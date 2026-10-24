//! The README's Titles table names every playbook directory, and only those - so the mapping from
//! an opaque title id to the game it is cannot go stale without this failing.

use std::collections::BTreeSet;

#[test]
fn titles_table_matches_the_playbook_dirs() {
    let root = vitaslop_playbook::crate_root();
    let mut dirs = BTreeSet::new();
    for (prefix, sub) in [("", root.join("playbooks")), ("best-effort/", root.join("playbooks").join("best-effort"))] {
        for e in std::fs::read_dir(&sub).unwrap() {
            let e = e.unwrap();
            let name = e.file_name().to_string_lossy().into_owned();
            if e.file_type().unwrap().is_dir() && name != "best-effort" {
                dirs.insert(format!("{prefix}{name}"));
            }
        }
    }
    let readme = std::fs::read_to_string(root.join("README.md")).unwrap();
    let section = readme.split("\n## Titles").nth(1).expect("README has a ## Titles section");
    let section = section.split("\n## ").next().unwrap();
    let rows: BTreeSet<String> = section
        .lines()
        .filter_map(|l| l.trim().strip_prefix("| `"))
        .filter_map(|l| l.split_once('`').map(|(d, _)| d.to_string()))
        .collect();
    let missing: Vec<_> = dirs.difference(&rows).collect();
    let stale: Vec<_> = rows.difference(&dirs).collect();
    assert!(
        missing.is_empty() && stale.is_empty(),
        "README Titles table out of date - directories with no row: {missing:?}; rows naming no directory: {stale:?}"
    );
}
