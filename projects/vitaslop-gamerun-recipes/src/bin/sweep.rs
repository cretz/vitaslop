//! `sweep` - search a single input-timing parameter fast by bucketing outcomes with
//! the determinism signature.
//!
//! Authoring a frame-precise input (when to press X to land a trick) is a search over
//! one frame number. Running each candidate and eyeballing a screenshot is slow. But
//! the runner already computes a determinism signature over the observable output, and
//! many candidate timings collapse to the SAME signature - the same outcome. So this
//! tool sweeps a button press across a frame range, groups the runs by signature, and
//! writes just one screenshot per DISTINCT outcome. You then eyeball two or three
//! images instead of thirty, and the buckets tell you which frames are equivalent.
//!
//! It works by injecting a press into the recipe TEXT (append `<f>: <button>` then a
//! release), so it needs no special recipe support - any recipe is sweepable.
//!
//! Usage:
//!   sweep --game <dir> --recipe <file> --at <LO>-<HI> [options]
//!
//! Options:
//!   --button <name>    the input to inject (default: cross)
//!   --hold <N>         frames to hold it (default: 1)
//!   --shot-frame <F>   frame to screenshot for each run (default: max-frames - 5)
//!   --shots <dir>      where the per-bucket screenshots go (required to see outcomes)
//!   --max-frames <N>   stop after N flips (default 900)
//!   --observe-from <N> fast-forward the prefix (default auto)

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;

use vitaslop_native::{run_recipe, RunOpts};
use vitaslop_runtime::Recipe;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut game: Option<String> = None;
    let mut recipe_path: Option<String> = None;
    let mut shots: Option<PathBuf> = None;
    let mut range: Option<(u64, u64)> = None;
    let mut button = "cross".to_string();
    let mut hold: u64 = 1;
    let mut shot_frame: Option<u64> = None;
    let mut max_frames: u64 = 900;
    let mut observe_from: Option<u64> = None;

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
            "--at" => {
                range = next().and_then(|s| {
                    let (lo, hi) = s.split_once('-')?;
                    Some((lo.trim().parse().ok()?, hi.trim().parse().ok()?))
                })
            }
            "--button" => button = next().unwrap_or(button),
            "--hold" => hold = next().and_then(|s| s.parse().ok()).unwrap_or(hold),
            "--shot-frame" => shot_frame = next().and_then(|s| s.parse().ok()),
            "--max-frames" => max_frames = next().and_then(|s| s.parse().ok()).unwrap_or(max_frames),
            "--observe-from" => observe_from = next().and_then(|s| s.parse().ok()),
            other => {
                eprintln!("unknown argument {other:?}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let (Some(game), Some(recipe_path), Some((lo, hi))) = (game, recipe_path, range) else {
        eprintln!("usage: sweep --game <dir> --recipe <file> --at <LO>-<HI> [--button cross] [--hold N] [--shots <dir>]");
        return ExitCode::from(2);
    };
    let base = match std::fs::read_to_string(&recipe_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: reading recipe {recipe_path}: {e}");
            return ExitCode::from(2);
        }
    };
    let shot_frame = shot_frame.unwrap_or(max_frames.saturating_sub(5));

    eprintln!(
        "sweep {button} (hold {hold}) over frames {lo}-{hi}, shot at f{shot_frame}, recipe {recipe_path}"
    );

    // Baseline: the recipe with NO injected press, so we can label the "no effect"
    // bucket (any swept frame whose signature equals this changed nothing).
    let baseline_sig = run_variant(&game, &base, None, hold, &button, shot_frame, &shots, max_frames, observe_from, "baseline");

    // Sweep: one run per candidate frame, grouped by outcome signature.
    let mut buckets: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for f in lo..=hi {
        let sig = run_variant(&game, &base, Some(f), hold, &button, shot_frame, &shots, max_frames, observe_from, &format!("f{f:05}"));
        if let Some(sig) = sig {
            buckets.entry(sig).or_default().push(f);
        }
    }

    println!("\n=== sweep outcomes ({} distinct) ===", buckets.len());
    if let Some(bl) = baseline_sig {
        println!("baseline (no press) sig = {bl:#018x}");
    }
    for (sig, frames) in &buckets {
        let tag = if Some(*sig) == baseline_sig { "  (no effect - same as baseline)" } else { "" };
        let repr = frames[0];
        println!(
            "SIG {sig:#018x}  frames {frames:?}  repr shot sweep-f{repr:05}.png{tag}",
        );
    }
    println!(
        "\nEyeball one shot per distinct SIG bucket to find the winning timing. Buckets \
         marked 'no effect' pressed too late (or outside the input window)."
    );
    ExitCode::SUCCESS
}

/// Run one variant (optionally injecting a press at `frame`), returning its signature.
/// Writes a screenshot named by `label` at `shot_frame` when a shots dir is set.
#[allow(clippy::too_many_arguments)]
fn run_variant(
    game: &str,
    base: &str,
    frame: Option<u64>,
    hold: u64,
    button: &str,
    shot_frame: u64,
    shots: &Option<PathBuf>,
    max_frames: u64,
    observe_from: Option<u64>,
    label: &str,
) -> Option<u64> {
    let mut text = base.to_string();
    text.push('\n');
    if let Some(f) = frame {
        // Inject the press and its release.
        text.push_str(&format!("{f}: {button}\n{}:\n", f + hold));
    }
    // A screenshot for this variant, named so buckets map to files.
    if shots.is_some() {
        text.push_str(&format!("{shot_frame}: @shot sweep-{label}\n"));
    }
    let recipe = match Recipe::parse(&text) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  {label}: parse error: {e}");
            return None;
        }
    };
    let opts = RunOpts {
        max_frames,
        observe_from,
        shot_dir: shots.clone(),
        ..RunOpts::default()
    };
    match run_recipe(game, &recipe, opts) {
        Ok(r) => {
            eprintln!("  {label}: sig {:#018x} ({:?})", r.sig, r.run);
            Some(r.sig)
        }
        Err(e) => {
            eprintln!("  {label}: run error: {e}");
            None
        }
    }
}
