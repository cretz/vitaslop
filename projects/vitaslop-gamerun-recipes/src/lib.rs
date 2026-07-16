//! The game-run recipe registry: a tiny, engine-free listing that maps a title id to
//! a friendly name and finds the recipe files for a title. This is the only crate that
//! knows specific games exist; it holds no engine code and no game content, just the
//! `games.toml` registry and directory conventions.
//!
//! Layout under this crate:
//! ```text
//! games.toml               # id -> friendly name
//! recipes/<TITLE_ID>/*.recipe
//! ```

use std::path::{Path, PathBuf};

/// One registry entry: a title id and its friendly name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameEntry {
    /// The title id, e.g. `"PCSE00341"` - also the recipes subdir name.
    pub id: String,
    /// A human name, e.g. `"OlliOlli"`.
    pub name: String,
    /// Region tag, if given (e.g. `"PAL"`).
    pub region: Option<String>,
}

/// The crate root (where `games.toml` and `recipes/` live), resolved at compile time.
pub fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The recipes directory (`<crate>/recipes`).
pub fn recipes_root() -> PathBuf {
    crate_root().join("recipes")
}

/// Load and parse `games.toml` from the crate root. Returns the registry entries.
pub fn load_registry() -> Vec<GameEntry> {
    let path = crate_root().join("games.toml");
    let text = std::fs::read_to_string(path).unwrap_or_default();
    parse_registry(&text)
}

/// Parse the minimal `games.toml` subset we use: a series of `[[game]]` blocks, each
/// with `id`, `name`, and optional `region` as quoted `key = "value"` lines. Kept
/// hand-rolled so the crate needs no TOML dependency.
pub fn parse_registry(text: &str) -> Vec<GameEntry> {
    let mut out = Vec::new();
    let mut cur: Option<(Option<String>, Option<String>, Option<String>)> = None;
    let flush = |cur: &mut Option<(Option<String>, Option<String>, Option<String>)>,
                 out: &mut Vec<GameEntry>| {
        if let Some((Some(id), name, region)) = cur.take() {
            out.push(GameEntry { id, name: name.unwrap_or_default(), region });
        }
    };
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line == "[[game]]" {
            flush(&mut cur, &mut out);
            cur = Some((None, None, None));
            continue;
        }
        let Some((key, val)) = line.split_once('=') else { continue };
        let key = key.trim();
        let val = val.trim().trim_matches('"').to_string();
        if let Some(entry) = cur.as_mut() {
            match key {
                "id" => entry.0 = Some(val),
                "name" => entry.1 = Some(val),
                "region" => entry.2 = Some(val),
                _ => {}
            }
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// The friendly name for a title id, if registered.
pub fn friendly_name(id: &str) -> Option<String> {
    load_registry().into_iter().find(|g| g.id == id).map(|g| g.name)
}

/// List the recipe files (`*.recipe`) under `recipes/<id>/`, sorted by name.
pub fn recipes_for(id: &str) -> Vec<PathBuf> {
    list_recipes(&recipes_root().join(id))
}

/// List `*.recipe` files directly under `dir`, sorted by file name.
pub fn list_recipes(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "recipe").unwrap_or(false))
        .collect();
    out.sort();
    out
}

/// Every recipe across every registered title, as `(title_id, recipe_path)`.
pub fn all_recipes() -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for g in load_registry() {
        for p in recipes_for(&g.id) {
            out.push((g.id.clone(), p));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_registry_blocks() {
        let text = "\
# a comment
[[game]]
id = \"PCSE00341\"
name = \"OlliOlli\"
region = \"PAL\"

[[game]]
id = \"ABCD00001\"
name = \"Example\"
";
        let games = parse_registry(text);
        assert_eq!(games.len(), 2);
        assert_eq!(games[0], GameEntry { id: "PCSE00341".into(), name: "OlliOlli".into(), region: Some("PAL".into()) });
        assert_eq!(games[1].id, "ABCD00001");
        assert_eq!(games[1].region, None);
    }

    #[test]
    fn registry_file_loads_and_has_recipes() {
        // The committed registry must parse and every listed title must have at least
        // one recipe file (a listing with no recipes is a mistake).
        let games = load_registry();
        assert!(!games.is_empty(), "games.toml has no entries");
        for g in &games {
            assert!(
                !recipes_for(&g.id).is_empty(),
                "registered title {} has no recipe files under recipes/{}/",
                g.id,
                g.id
            );
        }
    }
}
