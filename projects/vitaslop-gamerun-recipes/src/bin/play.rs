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
//!   --quantum-fuel <N>    scheduler preemption quantum (default 5000000). Changing it
//!                         changes WHERE threads interleave without changing what any
//!                         of them computes, so it separates a real bug (fails at every
//!                         quantum) from a scheduling-sensitive one (moves or vanishes).
//!
//! `RUST_LOG` selects the engine's `tracing` diagnostics, written to stderr - e.g.
//! `RUST_LOG=vitaslop::input=trace` shows every pad sample the guest actually reads
//! and its caller, which is how you tell "the game ignored my press" from "the game
//! never polled the pad".
//!
//! Exit code is 0 when every assertion passed, 1 otherwise, so a loop can gate on it.

use std::path::PathBuf;
use std::process::ExitCode;

use vitaslop_native::{run_recipe, RunOpts};
use vitaslop_runtime::Recipe;

fn main() -> ExitCode {
    // Surface the engine's `tracing` diagnostics on stderr (the report goes to stdout,
    // so the two stay separable). Without this, RUST_LOG is accepted and ignored.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();
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
            "--quantum-fuel" => {
                opts.quantum_fuel = next().and_then(|s| s.parse().ok()).unwrap_or(opts.quantum_fuel)
            }
            "--observe-from" => opts.observe_from = next().and_then(|s| s.parse().ok()),
            "--shot-every" => opts.shot_every = next().and_then(|s| s.parse().ok()),
            "--list-knobs" => {
                // Every VITASLOP_* diagnostic knob the workspace reads, generated
                // from the source that reads it, so it is never out of date. Beats
                // grepping the tree, which is how these were found until now.
                print!("{}", vitaslop_runtime::knobs::INDEX);
                return ExitCode::SUCCESS;
            }
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
    // Under `VITASLOP_BLOCK_HIST`, follow the report with the per-PC block-entry
    // histogram. Once the hot host calls are inlined away, this is the only thing that
    // says where guest time actually goes, and a recipe run is often the only way to
    // reach the code in question.
    report.dump_block_hist();

    if report.all_passed() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
