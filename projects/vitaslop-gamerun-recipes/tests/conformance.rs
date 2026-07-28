//! Opt-in game-run conformance: replay every committed recipe against a privately
//! supplied game dump and assert the recipe's own assertions pass. Content-free - the
//! recipes embed no game bytes, and the pass criterion is each recipe's declared
//! `@assert`s (a save happened, a value reached a threshold), never pixels.
//!
//! This is `#[ignore]`d and skips unless a game dir is provided, so `cargo test
//! --workspace` stays green for everyone. It only actually runs for whoever holds the
//! private dump. Point it at the extracted app dir for the title whose recipes you
//! want to run:
//!
//!   VITASLOP_GAME_DIR=/path/to/app/<TITLE_ID> \
//!   cargo test --release -p vitaslop-gamerun-recipes --test conformance -- --ignored --nocapture
//!
//! Only recipes whose `@game` matches the dumped title are run (the dir holds one
//! title). A recipe with no `@game` is run against whatever dir is given.

use vitaslop_gamerun_recipes as registry;
use vitaslop_native::{run_recipe, RunOpts};
use vitaslop_runtime::Recipe;

#[test]
#[ignore = "game-run conformance: needs VITASLOP_GAME_DIR (a private dump)"]
fn recipes_pass_their_assertions() {
    let Ok(dir) = std::env::var("VITASLOP_GAME_DIR") else {
        eprintln!("VITASLOP_GAME_DIR not set; skipping (needs a private game dump)");
        return;
    };
    // Which title is dumped? Prefer an explicit id; else infer from the dir's last
    // path component (the extracted app dir is named by title id).
    let want_id = std::env::var("VITASLOP_GAME_ID").ok().or_else(|| {
        std::path::Path::new(&dir).file_name().map(|s| s.to_string_lossy().into_owned())
    });
    eprintln!("game dir {dir} (title {want_id:?})");

    let mut ran = 0;
    let mut failed: Vec<String> = Vec::new();
    for (title_id, path) in registry::all_recipes() {
        let text = std::fs::read_to_string(&path).expect("read recipe");
        let recipe = Recipe::parse(&text).expect("parse recipe");
        // Skip recipes for a different title than the one dumped.
        let recipe_game = recipe.meta.game.clone().unwrap_or_else(|| title_id.clone());
        if let Some(want) = &want_id {
            if &recipe_game != want {
                continue;
            }
        }
        ran += 1;
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        eprintln!("\n=== running {name} (title {recipe_game}) ===");

        let opts = RunOpts { max_frames: 8000, ..RunOpts::default() };
        let report = run_recipe(&dir, &recipe, opts).expect("run recipe");
        eprint!("{}", report.render_text());

        if recipe.asserts.is_empty() {
            failed.push(format!("{name}: recipe has no @assert (nothing to conform to)"));
        } else if !report.all_passed() {
            let fails: Vec<String> = report
                .asserts
                .iter()
                .filter(|a| !a.passed)
                .map(|a| format!("f{} {} ({})", a.frame, a.desc, a.detail))
                .collect();
            failed.push(format!("{name}: {}", fails.join("; ")));
        }
    }

    assert!(ran > 0, "no recipes matched the dumped title {want_id:?}");
    assert!(failed.is_empty(), "recipe assertions failed:\n  {}", failed.join("\n  "));
}
