//! Every committed recipe PARSES, and its `@game` matches the directory it is filed under.
//!
//! Not `#[ignore]`d and needing no game dump: this is the half of conformance that can run for
//! everyone, and it is the half that catches the mistake actually made. A recipe whose header
//! was edited by hand fails to parse only when someone next runs the title - which is a boot,
//! a transpile and several minutes away - and a recipe filed under the wrong title id is
//! skipped SILENTLY by [`conformance`], which reads exactly like "that recipe passed".

use vitaslop_gamerun_recipes as registry;
use vitaslop_runtime::Recipe;

#[test]
fn every_committed_recipe_parses_and_is_filed_under_its_own_title() {
    let mut seen = 0;
    let mut bad: Vec<String> = Vec::new();
    for (title_id, path) in registry::all_recipes() {
        seen += 1;
        let name = format!("{title_id}/{}", path.file_name().unwrap().to_string_lossy());
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                bad.push(format!("{name}: unreadable: {e}"));
                continue;
            }
        };
        match Recipe::parse(&text) {
            Err(e) => bad.push(format!("{name}: {e}")),
            Ok(r) => {
                // A `@game` that disagrees with the directory is worse than a missing one: the
                // conformance runner filters on it, so the recipe is quietly never run.
                if let Some(g) = &r.meta.game
                    && g != &title_id
                {
                    bad.push(format!("{name}: @game is {g} but it is filed under {title_id}"));
                }
            }
        }
    }
    assert!(seen > 0, "no recipes found under {}", registry::recipes_root().display());
    assert!(bad.is_empty(), "{seen} recipes checked, these are broken:\n  {}", bad.join("\n  "));
}
