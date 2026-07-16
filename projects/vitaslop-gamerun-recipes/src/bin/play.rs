//! `play` - the agent-facing recipe loop. Boot a title, replay a recipe, and print a
//! structured report (each assertion's pass/fail with actual-vs-expected, screenshot
//! paths, the determinism signature). Read the report, edit the recipe, run again.
//!
//! Usage:
//!   play --game <extracted-app-dir> --recipe <file.recipe> [options]
//!
//! Options:
//!   --shots <dir>         write @shot screenshots and the watch log here (artifacts
//!                         never go in the repo - point this at a scratch dir)
//!   --max-frames <N>      stop after N display flips (default 4000)
//!   --observe-from <N>    begin per-frame observation at frame N (fast-forward the
//!                         prefix); default auto
//!   --shot-every <N>      also screenshot every N observed frames (overrides the
//!                         recipe's @shot-every); in-game shots let a human assess
//!                         render/gameplay quality across the run
//!   --max-rounds <N>      overall thread-resume cap (default 400000000)
//!
//! Exit code is 0 when every assertion passed, 1 otherwise, so a loop can gate on it.

use std::path::PathBuf;
use std::process::ExitCode;

use vitaslop_native::{run_recipe, RunOpts};
use vitaslop_runtime::Recipe;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut game: Option<String> = None;
    let mut recipe_path: Option<String> = None;
    let mut shots: Option<PathBuf> = None;
    let mut opts = RunOpts::default();

    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        let mut next = || {
            i += 1;
            args.get(i).cloned()
        };
        match a {
            "--game" => game = next(),
            "--recipe" => recipe_path = next(),
            "--shots" => shots = next().map(PathBuf::from),
            "--max-frames" => opts.max_frames = next().and_then(|s| s.parse().ok()).unwrap_or(opts.max_frames),
            "--max-rounds" => opts.max_rounds = next().and_then(|s| s.parse().ok()).unwrap_or(opts.max_rounds),
            "--observe-from" => opts.observe_from = next().and_then(|s| s.parse().ok()),
            "--shot-every" => opts.shot_every = next().and_then(|s| s.parse().ok()),
            "-h" | "--help" => {
                eprintln!("usage: play --game <dir> --recipe <file> [--shots <dir>] [--max-frames N] [--observe-from N]");
                return ExitCode::from(2);
            }
            other => {
                eprintln!("unknown argument {other:?}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let (Some(game), Some(recipe_path)) = (game, recipe_path) else {
        eprintln!("error: --game and --recipe are required");
        return ExitCode::from(2);
    };

    let text = match std::fs::read_to_string(&recipe_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: reading recipe {recipe_path}: {e}");
            return ExitCode::from(2);
        }
    };
    let recipe = match Recipe::parse(&text) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    if let Some(t) = &recipe.meta.title {
        eprintln!("# {t}");
    }
    // Echo the recipe's own notes and open tasks up front - the handoff an author left.
    for n in &recipe.notes {
        eprintln!("# {} f{}: {}", if n.todo { "TODO" } else { "NOTE" }, n.frame, n.text);
    }
    opts.shot_dir = shots.clone();

    let report = match run_recipe(&game, &recipe, opts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: run failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    // The watch log is an artifact: write it to the shots dir if one was given.
    if let (Some(dir), false) = (&shots, report.watch_csv.is_empty()) {
        let _ = std::fs::create_dir_all(dir);
        let p = dir.join("watch.csv");
        if std::fs::write(&p, &report.watch_csv).is_ok() {
            eprintln!("wrote watch log to {}", p.display());
        }
    }

    print!("{}", report.render_text());

    if report.all_passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
