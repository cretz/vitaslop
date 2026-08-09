//! The index of `VITASLOP_*` environment knobs - GENERATED from the source that
//! reads them, and checked against it by a test.
//!
//! # Why generated
//! There are well over a hundred of these knobs spread across six crates, and until
//! now the only way to find one was to grep the source, repeatedly, having first
//! guessed what it might be called. A hand-written index would answer that once and
//! then rot: the next knob added would not be in it, and an index that is silently
//! incomplete is worse than none, because you stop grepping and conclude the knob
//! does not exist.
//!
//! So the index is DERIVED. [`scan_sources`] finds every `VITASLOP_*` literal in the
//! workspace and pairs it with the doc comment above the function that reads it (the
//! house style is one small documented helper per knob), and
//! [`index_is_current`](tests::index_is_current) fails if the checked-in `KNOBS.md`
//! does not match what the source says today. Adding a knob therefore *requires*
//! regenerating the index, and the index cannot describe a knob that no longer
//! exists.
//!
//! Regenerate after adding or removing a knob:
//!
//! ```text
//! VITASLOP_BLESS_KNOBS=1 cargo test -p vitaslop-runtime --lib knobs
//! ```

/// The generated knob index, embedded so any binary can print it (`--list-knobs`)
/// without needing the repository on disk.
pub const INDEX: &str = include_str!("../../KNOBS.md");

// The override table itself lives in `vitaslop-platform`, the crate BELOW this one:
// the renderer reads `VITASLOP_GXP_LIVE` from there, and it cannot reach a table owned
// by the runtime. Re-exported here so the name `knobs::set_override` still resolves and
// this module stays the single place to look for anything knob-shaped.
pub use vitaslop_platform::knobs::{flag, log_filter, set_override, var, var_os, OVERRIDABLE};

/// One environment knob: its name, where it is read, and the first line of the doc
/// comment attached to the code that reads it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Knob {
    pub name: String,
    /// Workspace-relative path of the first file that reads it.
    pub file: String,
    pub line: usize,
    /// A one-line summary lifted from the nearest preceding doc comment, or empty
    /// when the read site carries none.
    pub summary: String,
    /// How well this site documents the knob: 2 when its doc comment names the knob
    /// (so it is the definition), 1 when it has any doc comment, 0 when it has none.
    /// Only used to pick between several sites mentioning the same knob.
    score: u8,
}

/// Every `VITASLOP_*` knob read anywhere under `root`, sorted by name.
///
/// A knob is a `VITASLOP_`-prefixed identifier appearing in the source. Prefix
/// fragments (a name built by concatenation, e.g. a `VITASLOP_WATCH_` stem) are kept
/// as-is and marked by their trailing underscore rather than dropped - a reader
/// hunting for the knob needs to know the stem exists.
pub fn scan_sources(root: &std::path::Path) -> Vec<Knob> {
    let mut found: std::collections::BTreeMap<String, Knob> = std::collections::BTreeMap::new();
    for path in rust_sources(root) {
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            // The index describes knobs, not the machinery that indexes them - and the
            // override table names every routed knob as a bare literal, which would
            // otherwise index each one to a line that does not read it.
            if rel.ends_with("vitaslop-runtime/src/knobs.rs")
                || rel.ends_with("vitaslop-platform/src/knobs.rs")
            {
                continue;
            }
            for name in knob_names(line) {
                let (summary, block) = doc_at(&lines, i);
                // A knob is often mentioned in several places (the code that reads
                // it, a host that forwards it, a comment that refers to it). Keep the
                // site whose own documentation actually talks about THIS knob - that
                // is the definition - over one that merely happens to sit under some
                // other doc comment.
                let score = if block.contains(&name) {
                    2
                } else if summary.is_empty() {
                    0
                } else {
                    1
                };
                let cand =
                    Knob { name: name.clone(), file: rel.clone(), line: i + 1, summary, score };
                match found.get(&name) {
                    Some(prev) if prev.score >= score => {}
                    _ => {
                        found.insert(name, cand);
                    }
                }
            }
        }
    }
    found.into_values().collect()
}

/// Every `VITASLOP_*` identifier on one line.
fn knob_names(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while let Some(at) = line[i..].find("VITASLOP_") {
        let start = i + at;
        let mut end = start;
        while end < bytes.len()
            && (bytes[end].is_ascii_uppercase() || bytes[end].is_ascii_digit() || bytes[end] == b'_')
        {
            end += 1;
        }
        out.push(line[start..end].to_string());
        i = end.max(start + 1);
    }
    out
}

/// The doc comment above the item containing line `at`, as
/// `(first sentence, whole block)`. Walks back over code to the nearest `///` block;
/// in this codebase that block is the knob's own documentation.
fn doc_at(lines: &[&str], at: usize) -> (String, String) {
    // Do not wander more than a small item's worth of lines: a match arm deep inside
    // a long function would otherwise inherit a doc comment from far above and
    // describe the wrong thing.
    const MAX_LOOKBACK: usize = 40;
    let lo = at.saturating_sub(MAX_LOOKBACK);
    let mut doc_end = None;
    for i in (lo..=at).rev() {
        if lines[i].trim_start().starts_with("///") {
            doc_end = Some(i);
            break;
        }
    }
    let Some(end) = doc_end else { return (String::new(), String::new()) };
    // Walk up to the start of that contiguous doc block, then take its first line.
    let mut start = end;
    while start > 0 && lines[start - 1].trim_start().starts_with("///") {
        start -= 1;
    }
    let block = lines[start..=end].join(" ");
    let first = lines[start].trim_start().trim_start_matches("///").trim();
    // The doc block's first line often ends mid-sentence; take through the first
    // sentence end if there is one on that line, else the whole line.
    let summary = match first.find(". ") {
        Some(p) => first[..p + 1].to_string(),
        None => first.to_string(),
    };
    (summary, block)
}

/// Every `.rs` file under `root`, excluding build output.
fn rust_sources(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().to_string();
            if p.is_dir() {
                if name != "target" && !name.starts_with('.') {
                    stack.push(p);
                }
            } else if p.extension().map(|x| x == "rs").unwrap_or(false) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Render the knob index as the markdown checked in at `projects/KNOBS.md`.
pub fn render_index(knobs: &[Knob]) -> String {
    let mut s = String::new();
    s.push_str("# VITASLOP_* environment knobs\n\n");
    s.push_str(
        "GENERATED - do not edit by hand. Regenerate with:\n\n\
         ```text\n\
         VITASLOP_BLESS_KNOBS=1 cargo test -p vitaslop-runtime --lib knobs\n\
         ```\n\n\
         Every knob the workspace reads, with the file that reads it and the first\n\
         line of that code's own documentation. A knob read at TRANSPILE time only\n\
         takes effect when the module is built, so it must be set for the whole run,\n\
         not just the frame you care about. Trapping diagnostics can be held inert\n\
         until a chosen display frame with `VITASLOP_ARM_AT_FRAME`, which is what\n\
         makes a first-hit watchpoint usable deep inside a game instead of firing\n\
         during boot.\n\n",
    );
    s.push_str(&format!("{} knobs.\n\n", knobs.len()));
    s.push_str("| knob | read in | what it does |\n|---|---|---|\n");
    for k in knobs {
        let summary = if k.summary.is_empty() { "-" } else { k.summary.as_str() };
        // Pipes would break the table; the summaries are prose, so this is enough.
        s.push_str(&format!(
            "| `{}` | {}:{} | {} |\n",
            k.name,
            k.file,
            k.line,
            summary.replace('|', "/")
        ));
    }
    s
}

/// The workspace root (`projects/`), derived from this crate's manifest directory.
pub fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir has a parent")
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The checked-in index must describe exactly the knobs the source reads today.
    /// This is what stops the index from rotting into a confident lie - and a stale
    /// index is worse than none, because it makes a real knob look nonexistent.
    #[test]
    fn index_is_current() {
        let root = workspace_root();
        let knobs = scan_sources(&root);
        assert!(knobs.len() > 50, "the knob scan found only {} knobs - it is broken", knobs.len());
        let rendered = render_index(&knobs);
        let path = root.join("KNOBS.md");
        if std::env::var("VITASLOP_BLESS_KNOBS").is_ok() {
            std::fs::write(&path, &rendered).expect("write KNOBS.md");
            return;
        }
        let current = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(
            current.replace("\r\n", "\n"),
            rendered,
            "KNOBS.md is out of date. Regenerate it with \
             VITASLOP_BLESS_KNOBS=1 cargo test -p vitaslop-runtime --lib knobs"
        );
    }

    /// A knob whose reader ALREADY goes through this module must be in [`OVERRIDABLE`].
    ///
    /// # This has been forgotten four times, and it fails on the one engine it matters to
    /// The browser has no environment ([[vitaslop-browser-has-no-env]]), so a knob reaches it
    /// only through the override table. Routing a reader through `knobs::var`/`flag` and then
    /// leaving the name out of `OVERRIDABLE` produces the worst possible pairing: the knob looks
    /// browser-ready in every way a reader can check, and `set_override` PANICS on it - so
    /// typing it into the phone's knobs box kills the run on boot with a black canvas and no
    /// output. It happened to the call-site profiler, the inline-imports switch,
    /// `VITASLOP_GXM_NO_MULTISAMPLE`, and then to all four compressed-texture knobs at once -
    /// on a feature whose entire purpose was the phone.
    ///
    /// The condition is exactly checkable: the scanner already knows where each knob is READ,
    /// so a read that names this module is a knob that has earned its place. `EXEMPT` is for the
    /// handful that are routed for a different reason (a shared helper, a desktop-only path) and
    /// each entry needs a stated one - it is not a place to put a knob that was simply
    /// forgotten.
    #[test]
    fn a_knob_routed_through_this_module_is_reachable_from_the_browser() {
        /// Routed, but deliberately not browser-reachable. Add with a REASON or not at all.
        const EXEMPT: &[(&str, &str)] = &[];
        let root = workspace_root();
        let mut missing = Vec::new();
        for k in scan_sources(&root) {
            if OVERRIDABLE.contains(&k.name.as_str())
                || EXEMPT.iter().any(|(n, _)| *n == k.name)
            {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(root.join(&k.file)) else { continue };
            let Some(line) = text.lines().nth(k.line - 1) else { continue };
            // The READ itself, not a mention: `knobs::var("NAME")` on one line. A doc comment
            // that merely says the word "knobs" is not a read, which is why this looks for the
            // call and the name together.
            let routed = ["knobs::var(", "knobs::flag(", "knobs::var_os("]
                .iter()
                .any(|call| line.contains(call))
                && line.contains(&k.name);
            if routed {
                missing.push(format!("{} ({}:{})", k.name, k.file, k.line));
            }
        }
        assert!(
            missing.is_empty(),
            "these knobs are read through vitaslop_platform::knobs but are NOT in OVERRIDABLE, so \
             setting one in the browser PANICS the run on boot:\n  {}",
            missing.join("\n  ")
        );
    }

    #[test]
    fn knob_names_are_read_out_of_a_line() {
        assert_eq!(knob_names("std::env::var(\"VITASLOP_GXP_LIVE\")"), vec!["VITASLOP_GXP_LIVE"]);
        assert_eq!(
            knob_names("// VITASLOP_A2 and VITASLOP_B"),
            vec!["VITASLOP_A2".to_string(), "VITASLOP_B".to_string()]
        );
        assert!(knob_names("nothing here").is_empty());
    }

    #[test]
    fn a_doc_comment_above_the_read_becomes_the_summary() {
        let lines = vec![
            "/// Turn the thing on. More detail here.",
            "/// A second line nobody wants in a table.",
            "fn thing() -> bool {",
            "    std::env::var(\"VITASLOP_THING\").is_ok()",
        ];
        let (summary, block) = doc_at(&lines, 3);
        assert_eq!(summary, "Turn the thing on.");
        assert!(block.contains("second line"), "the whole block is kept for matching");
    }
}
