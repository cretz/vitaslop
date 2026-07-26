//! `explore` - try many candidate inputs at one decision frame, in PARALLEL, and
//! report which of them actually did something different.
//!
//! # The problem
//! Tuning one input ("when do I press X to land this?", "does the throttle do
//! anything here?") is a search, and every candidate has to re-run the boot prefix
//! to reach the frame under test. Run them one after another and the search costs
//! the prefix times the number of candidates - an afternoon. There is no rewind to
//! avoid that with (a suspended guest thread's state lives partly in a live wasm
//! call stack, which cannot be serialized), but the prefix runs are INDEPENDENT, so
//! they can all run at once. On a 16-core machine a 24-candidate search costs about
//! two prefixes of wall-clock instead of 24.
//!
//! # The second trick: the signature is a free equivalence oracle
//! Most candidates collapse to the same outcome. Every run reports a determinism
//! signature over its observable output, so runs are grouped by it: identical
//! signature, identical outcome, no need to look at both. You end up eyeballing one
//! screenshot per DISTINCT outcome - usually two or three - instead of one per
//! candidate. A bucket equal to the no-input baseline is the loudest result of all:
//! it says that input changed NOTHING, so whatever you are tuning is not the lever.
//!
//! Usage:
//!   explore --game <dir> --at <frame> --variant "<label>=<input>" ... [options]
//!   explore --game <dir> --at <frame> --sweep "cross" --over 1200-1260 [options]
//!
//! Options:
//!   --game <dir>          the extracted app directory (required)
//!   --at <frame>          the decision frame every variant fast-forwards to (required)
//!   --recipe <file>       input prefix replayed on the way to `--at`
//!   --variant "<l>=<in>"  a candidate: a label and a recipe input spec (repeatable).
//!                         Semicolons make it a MANOEUVRE rather than a single held
//!                         input: "40@ry=0 lang=0 ; 30@ry=0 lang=315" accelerates for
//!                         40 frames, then turns for 30. A phase with no `<frames>@`
//!                         runs for --hold.
//!   --sweep "<input>"     with --over LO-HI, one variant per frame in the range
//!   --over LO-HI          the frame range for --sweep
//!   --hold <N>            frames the variant input is held (default 30)
//!   --after <N>           frames to run after releasing it (default 60)
//!   --watch n:type:addr   report this value at the end of every variant (repeatable)
//!   --report "<command>"  run this session command at the end of every variant and
//!                         include its output in the report (repeatable). `--report
//!                         "locate --id <hex>"` turns a bearing search into a table of
//!                         final world positions instead of screenshots to eyeball.
//!   --workers <N>         concurrent worker processes (default: cores - 2)
//!   --shots <dir>         one screenshot per variant lands here
//!   --dump <addr>:<len>   dump that guest region from every variant and diff each
//!                         against the baseline (`--dump all` for the whole guest).
//!                         Because every run replayed the same deterministic prefix,
//!                         a difference is CAUSED by the variant's input and nothing
//!                         else - this is how you find the address of a live value,
//!                         and how you prove an input reaches the game at all.
//!   --dump-dir <dir>      where the dumps go (default: the scratch work dir)
//!   --quantum-fuel <N>    scheduler preemption quantum
//!   --keep                keep the generated per-variant command files for inspection
//!
//! A no-input `baseline` variant is always run, because "the same as doing nothing"
//! is the answer you most need to be able to see.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::time::Instant;

/// One candidate to try at the decision frame.
#[derive(Clone)]
struct Variant {
    label: String,
    /// A recipe input spec (`l ly=0`, `cross`, `touch=450,674`), or empty for the
    /// do-nothing baseline.
    input: String,
    /// The frame to apply it at (`--sweep` moves this; a plain `--variant` uses
    /// `--at`).
    at: u64,
}

/// What one worker reported back.
struct Outcome {
    label: String,
    sig: Option<u64>,
    frame: u64,
    watches: String,
    shot: Option<PathBuf>,
    error: Option<String>,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut game: Option<String> = None;
    let mut recipe: Option<String> = None;
    let mut at: Option<u64> = None;
    let mut hold = 30u64;
    let mut after = 60u64;
    let mut workers = default_workers();
    let mut shots: Option<PathBuf> = None;
    let mut quantum: Option<u64> = None;
    let mut keep = false;
    let mut watches: Vec<String> = Vec::new();
    let mut reports: Vec<String> = Vec::new();
    let mut variants: Vec<Variant> = Vec::new();
    let mut sweep: Option<String> = None;
    let mut over: Option<(u64, u64)> = None;
    let mut dump: Option<String> = None;
    let mut dump_dir: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        let mut next = || {
            i += 1;
            args.get(i).cloned()
        };
        match a {
            "--game" => game = next(),
            "--recipe" => recipe = next(),
            "--at" => at = next().and_then(|s| s.parse().ok()),
            "--hold" => hold = next().and_then(|s| s.parse().ok()).unwrap_or(hold),
            "--after" => after = next().and_then(|s| s.parse().ok()).unwrap_or(after),
            "--workers" => workers = next().and_then(|s| s.parse().ok()).unwrap_or(workers),
            "--shots" => shots = next().map(PathBuf::from),
            "--quantum-fuel" => quantum = next().and_then(|s| s.parse().ok()),
            "--keep" => keep = true,
            "--watch" => {
                if let Some(w) = next() {
                    watches.push(w);
                }
            }
            "--report" => {
                if let Some(c) = next() {
                    reports.push(c);
                }
            }
            "--variant" => {
                let Some(spec) = next() else { continue };
                let (label, input) = match spec.split_once('=') {
                    Some((l, r)) => (l.trim().to_string(), r.trim().to_string()),
                    None => (spec.replace(' ', "_"), spec.clone()),
                };
                variants.push(Variant { label, input, at: 0 });
            }
            "--sweep" => sweep = next(),
            "--dump" => dump = next(),
            "--dump-dir" => dump_dir = next().map(PathBuf::from),
            "--over" => {
                over = next().and_then(|s| {
                    let (lo, hi) = s.split_once('-')?;
                    Some((lo.trim().parse().ok()?, hi.trim().parse().ok()?))
                })
            }
            "-h" | "--help" => {
                eprintln!("usage: explore --game <dir> --at <frame> --variant \"<label>=<input>\" ...");
                return ExitCode::from(2);
            }
            other => {
                eprintln!("unknown argument {other:?}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let (Some(game), Some(at_frame)) = (game, at) else {
        eprintln!("error: --game and --at are required");
        return ExitCode::from(2);
    };
    for v in &mut variants {
        v.at = at_frame;
    }
    // --sweep expands into one variant per frame in the range: the same input tried
    // at each candidate timing.
    if let Some(input) = &sweep {
        let Some((lo, hi)) = over else {
            eprintln!("error: --sweep needs --over LO-HI");
            return ExitCode::from(2);
        };
        for f in lo..=hi {
            variants.push(Variant { label: format!("f{f:05}"), input: input.clone(), at: f });
        }
    }
    if variants.is_empty() {
        eprintln!("error: no variants (use --variant or --sweep)");
        return ExitCode::from(2);
    }
    // The baseline goes first so its signature is known when the table is printed.
    variants.insert(
        0,
        Variant { label: "baseline".into(), input: String::new(), at: at_frame },
    );

    // Validate every phase spec BEFORE booting anything. A worker costs a whole
    // prefix replay, so a typo must not be discovered a minute into the search.
    for v in &variants {
        if let Err(e) = parse_phases(&v.input, hold) {
            eprintln!("error: variant {}: {e}", v.label);
            return ExitCode::from(2);
        }
    }

    let session_bin = match session_binary() {
        Some(p) => p,
        None => {
            eprintln!("error: cannot find the `session` binary next to this executable");
            return ExitCode::FAILURE;
        }
    };
    let work = std::env::temp_dir().join(format!("vitaslop-explore-{}", std::process::id()));
    let dump_root = dump_dir.clone().unwrap_or_else(|| work.clone());
    if let Err(e) = std::fs::create_dir_all(&work).and_then(|_| std::fs::create_dir_all(&dump_root)) {
        eprintln!("error: scratch dir: {e}");
        return ExitCode::FAILURE;
    }

    let workers = workers.max(1);
    eprintln!(
        "exploring {} variants at f{at_frame} (hold {hold}, then {after} frames) on {workers} workers",
        variants.len()
    );
    let started = Instant::now();

    // Run the variants `workers` at a time. Each is its own process: crash-isolated
    // (a variant that traps the guest kills only its own worker) and free of any
    // shared mutable state, which is what makes the parallelism safe.
    let mut outcomes: Vec<Outcome> = Vec::new();
    for chunk in variants.chunks(workers) {
        let mut running = Vec::new();
        for v in chunk {
            let cmd_file = work.join(format!("{}.cmds", v.label));
            let dump_path = dump
                .as_ref()
                .map(|_| dump_root.join(format!("{}.bin", v.label)));
            let script =
                build_script(v, hold, after, &watches, &reports, shots.is_some(), dump.as_deref(), dump_path.as_deref());
            if let Err(e) = std::fs::write(&cmd_file, &script) {
                eprintln!("error: writing {}: {e}", cmd_file.display());
                return ExitCode::FAILURE;
            }
            let mut c = Command::new(&session_bin);
            c.arg("--game").arg(&game).arg("--commands").arg(&cmd_file);
            if let Some(r) = &recipe {
                c.arg("--recipe").arg(r);
            }
            if let Some(s) = &shots {
                c.arg("--shots").arg(s);
            }
            if let Some(q) = quantum {
                c.arg("--quantum-fuel").arg(q.to_string());
            }
            c.stdout(Stdio::piped()).stderr(Stdio::null());
            match c.spawn() {
                Ok(child) => running.push((v.clone(), child)),
                Err(e) => {
                    eprintln!("error: spawning worker for {}: {e}", v.label);
                    return ExitCode::FAILURE;
                }
            }
        }
        for (v, child) in running {
            let out = match child.wait_with_output() {
                Ok(o) => o,
                Err(e) => {
                    outcomes.push(Outcome {
                        label: v.label,
                        sig: None,
                        frame: 0,
                        watches: String::new(),
                        shot: None,
                        error: Some(e.to_string()),
                    });
                    continue;
                }
            };
            let text = String::from_utf8_lossy(&out.stdout).to_string();
            outcomes.push(parse_outcome(&v.label, &text, &reports, shots.as_deref()));
            eprintln!("  {} done", v.label);
        }
    }

    // The dumps outlive the command files: they are the finding, not the scaffolding.
    if !keep && dump.is_none() {
        let _ = std::fs::remove_dir_all(&work);
    }

    // Group by outcome signature. Identical signature = identical observable run.
    let baseline = outcomes.iter().find(|o| o.label == "baseline").and_then(|o| o.sig);
    let mut buckets: BTreeMap<u64, Vec<&Outcome>> = BTreeMap::new();
    let mut broken: Vec<&Outcome> = Vec::new();
    for o in &outcomes {
        match o.sig {
            Some(sig) => buckets.entry(sig).or_default().push(o),
            None => broken.push(o),
        }
    }

    println!("\n=== {} distinct outcomes from {} variants in {:.1}s ===", buckets.len(), outcomes.len(), started.elapsed().as_secs_f64());
    for (sig, group) in &buckets {
        let same_as_baseline = Some(*sig) == baseline;
        let labels: Vec<&str> = group.iter().map(|o| o.label.as_str()).collect();
        let repr = group[0];
        println!(
            "SIG {sig:#018x}  f{}  {}{}",
            repr.frame,
            labels.join(" "),
            if same_as_baseline { "   <- NO EFFECT (identical to doing nothing)" } else { "" }
        );
        if !repr.watches.trim().is_empty() {
            for line in repr.watches.lines() {
                println!("      {line}");
            }
        }
        if let Some(p) = &repr.shot {
            println!("      shot {}", p.display());
        }
    }
    for o in &broken {
        println!("FAIL {:<16} {}", o.label, o.error.as_deref().unwrap_or("no signature reported"));
    }
    if let Some(_) = &dump {
        let base_dump = dump_root.join("baseline.bin");
        let variants: Vec<PathBuf> = outcomes
            .iter()
            .filter(|o| o.label != "baseline")
            .map(|o| dump_root.join(format!("{}.bin", o.label)))
            .filter(|p| p.exists())
            .collect();
        if base_dump.exists() && !variants.is_empty() {
            println!(
                "
memory dumps written; diff them against the baseline with:
  memdiff {} {}",
                base_dump.display(),
                variants.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(" ")
            );
        } else {
            println!("
WARNING: --dump was requested but not every variant produced a dump");
        }
    }

    if buckets.len() == 1 && broken.is_empty() {
        println!(
            "\nEvery variant produced the same outcome. That is a DIAGNOSIS, not a dead end: the \
             parameter you are varying is not the one that matters here. Vary something else \
             (a different button, a different frame, a different game state) before tuning further."
        );
    }
    ExitCode::SUCCESS
}

/// The command script one worker runs: fast-forward, apply the variant, release,
/// settle, then report.
#[allow(clippy::too_many_arguments)]
fn build_script(
    v: &Variant,
    hold: u64,
    after: u64,
    watches: &[String],
    reports: &[String],
    shot: bool,
    dump: Option<&str>,
    dump_path: Option<&Path>,
) -> String {
    let mut s = String::new();
    s.push_str(&format!("# variant {}\n", v.label));
    for w in watches {
        // `name:type:addr` is the compact CLI spelling of a `@watch` declaration.
        let parts: Vec<&str> = w.split(':').collect();
        if parts.len() == 3 {
            s.push_str(&format!("watch {} {} {}\n", parts[0], parts[1], parts[2]));
        }
    }
    // Fast-forward with no per-frame sampling: this is the prefix, not the experiment.
    if v.at > 0 {
        s.push_str(&format!("step {}\n", v.at));
    }
    let phases = parse_phases(&v.input, hold)
        .expect("every variant's phases were validated before any worker was spawned");
    for (input, frames) in phases {
        if !input.is_empty() {
            s.push_str(&format!("input {input}\n"));
        }
        s.push_str(&format!("step {frames}\n"));
    }
    s.push_str("input\n");
    if after > 0 {
        s.push_str(&format!("step {after}\n"));
    }
    if !watches.is_empty() {
        s.push_str("watches\n");
    }
    for c in reports {
        s.push_str(c);
        s.push('\n');
    }
    if shot {
        s.push_str(&format!("shot explore-{}\n", v.label));
    }
    if let (Some(spec), Some(path)) = (dump, dump_path) {
        // `all` means the whole guest region, which the session resolves itself.
        let region = if spec == "all" {
            String::new()
        } else {
            match spec.split_once(':') {
                Some((a, l)) => format!("{a} {l} "),
                None => format!("{spec} 0x100000 "),
            }
        };
        s.push_str(&format!("dump {region}{}\n", path.display()));
    }
    s.push_str("sig\n");
    s
}

/// Split a variant's input spec into the PHASES it plays: `(input, frames)` pairs,
/// applied in order at the decision frame.
///
/// A plain spec is one phase held for `--hold` frames, which is every single-button
/// question ("does the throttle do anything here?"). But the interesting questions in
/// a game with a world in it are not single inputs, they are MANOEUVRES - accelerate,
/// then turn, then straighten - and a search over those cannot be expressed as one
/// held input at one frame. Semicolons make a variant a short script:
///
/// ```text
/// --variant "hook-left=40@ry=0 lang=0 ; 30@ry=0 lang=315 ; 60@ry=0 lang=0"
/// ```
///
/// `<frames>@` sets that phase's duration; a phase without one runs for `--hold`. An
/// empty phase releases everything, so `; 20@ ;` is a deliberate pause. This keeps
/// the parallelism doing the expensive part - one prefix replay per candidate
/// manoeuvre instead of per candidate button.
fn parse_phases(spec: &str, hold: u64) -> Result<Vec<(String, u64)>, String> {
    let mut out = Vec::new();
    for phase in spec.split(';') {
        let (frames, input) = match phase.split_once('@') {
            Some((n, rest)) => {
                let n = n.trim();
                let frames: u64 = n
                    .parse()
                    .map_err(|_| format!("bad phase duration {n:?} in {spec:?} (want <frames>@<input>)"))?;
                if frames == 0 {
                    return Err(format!("phase duration 0 in {spec:?}: a phase the guest never samples is not a test"));
                }
                (frames, rest)
            }
            None => (hold, phase),
        };
        out.push((input.trim().to_string(), frames));
    }
    Ok(out)
}

/// Pull the reported signature, frame, watch values and shot path out of a worker's
/// transcript.
fn parse_outcome(label: &str, text: &str, reported: &[String], shots: Option<&Path>) -> Outcome {
    let mut sig = None;
    let mut frame = 0u64;
    let mut watches = String::new();
    let mut error = None;
    // The session transcript echoes each command as `$ <command>` and then its reply.
    // Capture the reply of every command whose output was asked for - `watches` plus
    // anything named by `--report`.
    let mut capturing = false;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("sig=") {
            sig = u64::from_str_radix(rest.trim().trim_start_matches("0x"), 16).ok();
        }
        if let Some(rest) = line.strip_prefix("frame=") {
            frame = rest.split_whitespace().next().and_then(|s| s.parse().ok()).unwrap_or(frame);
        }
        if let Some(rest) = line.strip_prefix("ERR ") {
            error = Some(rest.to_string());
        }
        if let Some(cmd) = line.strip_prefix("$ ") {
            let cmd = cmd.trim();
            capturing = cmd == "watches" || reported.iter().any(|r| r.trim() == cmd);
            continue;
        }
        if capturing && !line.trim().is_empty() {
            watches.push_str(line);
            watches.push('\n');
        }
    }
    let shot = shots.map(|d| d.join(format!("explore-{label}.png"))).filter(|p| p.exists());
    Outcome { label: label.to_string(), sig, frame, watches, shot, error }
}

/// The `session` executable that sits beside this one (same build, same profile).
fn session_binary() -> Option<PathBuf> {
    let me = std::env::current_exe().ok()?;
    let dir = me.parent()?;
    let name = if cfg!(windows) { "session.exe" } else { "session" };
    let p = dir.join(name);
    p.exists().then_some(p)
}

/// How many workers to run at once by default.
///
/// MEMORY, not cores, is the binding constraint: each worker is a whole booted
/// guest - a 537 MB linear memory for a real title, plus its JIT code - so it costs
/// well over a gigabyte. Defaulting to "cores minus two" would put fourteen of those
/// on a 16-core machine and take twenty gigabytes off whoever is using it. Four is a
/// large speedup over serial and leaves the machine usable; raise it with
/// `--workers` when you know the RAM is there.
fn default_workers() -> usize {
    std::thread::available_parallelism().map(|n| n.get().min(4).max(1)).unwrap_or(2)
}
