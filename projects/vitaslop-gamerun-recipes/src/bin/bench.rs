//! `bench` - measure what a guest frame COSTS, on a chosen screen, and say where the
//! time went.
//!
//! # Why this exists as its own tool
//! A frame rate measured over a whole run is an average of two unrelated workloads:
//! a boot sequence doing heavy one-off work, and whatever screen the title settles
//! on. Worse, a title idling on a menu costs almost nothing per frame, so a number
//! taken there is not a gameplay number at all - the project has been misled by
//! exactly that before. `bench` replays a recipe prefix to a frame you name, throws
//! the prefix away, and then times a fixed window of frames one at a time.
//!
//! It also splits the cost, because "guest CPU" is at least three things with
//! different fixes: translated guest code in the JIT, host calls (`env.import`), and
//! the scheduler's own bookkeeping. With `VITASLOP_PERF=1` the host-call bucket is
//! measured directly and attributed per NID, and the remainder is JIT plus scheduler.
//! Without it, only the frame total is reported and nothing is charged for measuring.
//!
//! Nothing here renders. Rendering has its own split (`VITASLOP_HEADLESS_TIMING` on
//! `vitaslop-desktop --headless`), and mixing them is how a CPU problem gets
//! mistaken for a GPU one.
//!
//! Usage:
//!   bench --game <extracted-app-dir> [--recipe <file.recipe>] --at <frame> [--frames N]
//!
//! Options:
//!   --game <dir>        the extracted app directory (required)
//!   --recipe <file>     input prefix replayed on the way to `--at`, and held through
//!                       the measured window (so a driving benchmark keeps driving)
//!   --at <frame>        frame the measured window starts at (default 0)
//!   --frames <N>        frames to time (default 120)
//!   --quantum-fuel <N>  scheduler preemption quantum (default 5000000). Worth
//!                       sweeping: preemption is billed to every block that runs.
//!   --top <N>           how many NIDs to list in the host-call breakdown (default 15)
//!   --json              also emit one machine-readable line, for a scripted sweep
//!
//! Set `VITASLOP_PERF=1` for the host-call split.

use std::process::ExitCode;

use vitaslop_native::{boot_retail, perf, RunReport, ThreadedScheduler, VitaEnv};
use vitaslop_runtime::{nid, Recipe};

fn main() -> ExitCode {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(vitaslop_runtime::knobs::log_filter()))
        .with_writer(std::io::stderr)
        .try_init();

    let args: Vec<String> = std::env::args().collect();
    let mut game: Option<String> = None;
    let mut recipe_path: Option<String> = None;
    let mut at: u64 = 0;
    let mut frames: u64 = 120;
    let mut quantum_fuel: u64 = 5_000_000;
    let mut top: usize = 15;
    let mut json = false;

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
            "--at" => at = next().and_then(|s| s.parse().ok()).unwrap_or(at),
            "--frames" => frames = next().and_then(|s| s.parse().ok()).unwrap_or(frames),
            "--quantum-fuel" => {
                quantum_fuel = next().and_then(|s| s.parse().ok()).unwrap_or(quantum_fuel)
            }
            "--top" => top = next().and_then(|s| s.parse().ok()).unwrap_or(top),
            "--json" => json = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: bench --game <dir> [--recipe <file>] --at <frame> [--frames N] \
                     [--quantum-fuel N] [--top N] [--json]"
                );
                return ExitCode::from(2);
            }
            other => {
                eprintln!("unknown argument {other:?}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let Some(game) = game else {
        eprintln!("bench: --game <extracted-app-dir> is required");
        return ExitCode::from(2);
    };

    // The recipe's input timeline drives the world exactly as `play` would, so the
    // measured window is the title doing whatever the recipe has it doing. A bench
    // with no recipe measures the title left alone, which for most titles is a menu.
    let recipe = match recipe_path.as_deref() {
        Some(p) => match std::fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|s| {
            Recipe::parse(&s).map_err(|e| format!("{e:?}"))
        }) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("bench: recipe {p}: {e}");
                return ExitCode::from(2);
            }
        },
        None => None,
    };

    let world: Box<dyn vitaslop_runtime::World + Send> = match &recipe {
        Some(r) => Box::new(r.clone().into_world()),
        None => Box::new(vitaslop_runtime::DeterministicWorld::default()),
    };

    let boot = std::time::Instant::now();
    let mut sched = match boot_retail(&game, world, quantum_fuel) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("bench: boot: {e}");
            return ExitCode::FAILURE;
        }
    };
    // One retained scene, as every run does: retaining more is megabytes a frame and
    // would make the measurement describe the benchmark rather than the game.
    sched.host().state.capture.scene_limit = Some(1);
    let build_ms = boot.elapsed().as_secs_f64() * 1000.0;
    println!("bench: booted in {build_ms:.0} ms, fast-forwarding to frame {at}...");

    if at > 0 {
        let prefix = std::time::Instant::now();
        let r = sched.run_frames(at, 400_000_000);
        if !matches!(r, RunReport::FramesReached(_)) {
            eprintln!("bench: prefix stopped short: {r:?} at frame {}", sched.frames());
            return ExitCode::FAILURE;
        }
        println!(
            "bench: prefix {} frames in {:.1} s",
            sched.frames(),
            prefix.elapsed().as_secs_f64()
        );
    }

    // Everything before this point - boot, JIT warm-up, the prefix - is deliberately
    // excluded. The counters are zeroed HERE so the split describes the measured
    // window only.
    perf::reset();
    vitaslop_runtime::perf::reset();
    let rounds_before = sched.rounds_total();
    // The GAME CLOCK and the FUEL over the window, not over the run. Both are the numbers
    // the emulator charges the title for, and both are dominated by boot when read over a
    // whole run - which is the mistake that made one title look seven times faster than
    // another when the two were measuring different things
    // [[vitaslop-compare-like-windows]]. Taken here they describe the screen `--at` names,
    // which is the only place a "this title costs 4x" claim means anything.
    let clock_before = sched.host().state.now_us();
    let (fuel_before, samples_before, _) = sched.fuel_report();
    let mut frame_ms: Vec<f64> = Vec::with_capacity(frames as usize);
    let window = std::time::Instant::now();
    let target = sched.frames() + frames;
    while sched.frames() < target {
        let next = sched.frames() + 1;
        let t = std::time::Instant::now();
        let r = sched.run_frames(next, 4_000_000);
        frame_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        if !matches!(r, RunReport::FramesReached(_)) {
            eprintln!("bench: run stopped at frame {}: {r:?}", sched.frames());
            break;
        }
    }
    let window_s = window.elapsed().as_secs_f64();

    if frame_ms.is_empty() {
        eprintln!("bench: no frames advanced");
        return ExitCode::FAILURE;
    }
    let rounds = sched.rounds_total() - rounds_before;
    let clock_us = sched.host().state.now_us().saturating_sub(clock_before);
    let (fuel_after, samples_after, fuel_max) = sched.fuel_report();
    let win = Window {
        clock_us,
        fuel: fuel_after - fuel_before,
        suspends: samples_after - samples_before,
        fuel_max,
    };
    report(&frame_ms, window_s, rounds, &sched, top, json, &win);
    ExitCode::SUCCESS
}

/// What the emulator CHARGED the guest over the measured window, as opposed to what the
/// window cost the host. `frame_ms` is the host's cost; these are the device's.
struct Window {
    /// Game clock advanced, in microseconds.
    clock_us: u64,
    /// Fuel burned - wasm operators executed by translated guest code.
    fuel: u64,
    /// Suspends the fuel was sampled over.
    suspends: u64,
    /// Largest single burn seen in the RUN. A burn above the preemption interval means the
    /// reading is broken, not that the title is busy, so it is carried as the falsifier.
    fuel_max: u64,
}

/// Percentile of an unsorted sample (nearest rank on the sorted copy).
fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[(((s.len() - 1) as f64) * p).round() as usize]
}

#[allow(clippy::too_many_arguments)]
fn report(
    frame_ms: &[f64],
    window_s: f64,
    rounds: u64,
    sched: &ThreadedScheduler<VitaEnv>,
    top: usize,
    json: bool,
    win: &Window,
) {
    let p50 = pct(frame_ms, 0.50);
    let total_ms = window_s * 1000.0;
    println!(
        "bench: {} frames in {:.2} s - p50 {:.2} ms ({:.1} fps), p95 {:.2} ms, max {:.2} ms, \
         mean {:.2} ms",
        frame_ms.len(),
        window_s,
        p50,
        1000.0 / p50.max(1e-6),
        pct(frame_ms, 0.95),
        pct(frame_ms, 1.0),
        total_ms / frame_ms.len() as f64
    );

    // What the DEVICE was charged, per frame of the measured window. A console frame is
    // 16,667 us, so `clock/frame` divided by that IS the title's clock ratio on this
    // screen - the number every "PCSA00027 costs 4x" claim is about, measured where the
    // claim applies instead of averaged over a boot.
    let n = frame_ms.len() as f64;
    println!(
        "bench: guest clock - {:.1} ms/frame charged ({:.2}x a 60 Hz console frame), \
         fuel {:.1} M/frame over {} suspends (mean {}, max {}, interval {})",
        win.clock_us as f64 / 1000.0 / n,
        win.clock_us as f64 / n / (1_000_000.0 / 60.0),
        win.fuel as f64 / 1e6 / n,
        win.suspends,
        if win.suspends == 0 { 0 } else { win.fuel / win.suspends },
        win.fuel_max,
        vitaslop_runtime::host::QUANTUM_FUEL,
    );

    // What `RenderSceneBuilder::build` did, in the same units the browser reports, so the
    // two engines' build cost is comparable without comparing two machines' clocks.
    println!("bench: {}", vitaslop_runtime::render::take_build_work().line(frame_ms.len() as u64));

    println!(
        "bench: scheduler - {rounds} thread resumes ({:.0} per frame, one per {:.0} host calls)",
        rounds as f64 / frame_ms.len() as f64,
        perf::snapshot().calls as f64 / rounds.max(1) as f64
    );

    let mut host_calls = 0u64;
    let mut import_pct = 0.0;
    if perf::enabled() {
        let s = perf::snapshot();
        host_calls = s.calls;
        let import_ms = s.import_ns as f64 / 1e6;
        let dispatch_ms = s.dispatch_ns as f64 / 1e6;
        let marshal_ms = s.marshal_ns() as f64 / 1e6;
        import_pct = 100.0 * import_ms / total_ms.max(1e-6);
        // The remainder is translated guest code plus the scheduler. It is a
        // SUBTRACTION, not a measurement - said plainly so nobody quotes it as one.
        let rest_ms = total_ms - import_ms;
        println!(
            "bench: host calls - {} calls, {:.0} ms total ({import_pct:.1}% of the window), \
             {:.2} us each",
            s.calls,
            import_ms,
            if s.calls == 0 { 0.0 } else { import_ms * 1000.0 / s.calls as f64 }
        );
        println!(
            "bench:   of which handler {:.0} ms, register marshalling {:.0} ms ({:.1}% of the \
             window - pure host overhead)",
            dispatch_ms,
            marshal_ms,
            100.0 * marshal_ms / total_ms.max(1e-6)
        );
        println!(
            "bench: remainder {:.0} ms ({:.1}%) is translated guest code + scheduler, BY \
             SUBTRACTION - not measured directly",
            rest_ms,
            100.0 * rest_ms / total_ms.max(1e-6)
        );

        // Inside the handler bucket: the phases of the GXM capture path. These are a
        // SUBSET of `handler` above, not a second budget - a phase that nests inside
        // another would be double-counted, so they are kept disjoint by construction.
        let phases: Vec<(&str, f64, u64, u64)> = vitaslop_runtime::perf::Phase::all()
            .iter()
            .map(|p| {
                let (ns, hits, bytes) = vitaslop_runtime::perf::read(*p);
                (p.label(), ns as f64 / 1e6, hits, bytes)
            })
            .filter(|(_, ms, _, _)| *ms > 0.0)
            .collect();
        if !phases.is_empty() {
            let frames = frame_ms.len() as f64;
            println!("bench: capture phases (inside the handler bucket):");
            for (label, ms, hits, bytes) in &phases {
                // Bytes per FRAME, because that is the number a fix has to move: a
                // phase copying tens of megabytes a frame is a volume problem however
                // fast the copy is.
                let vol = if *bytes == 0 {
                    String::new()
                } else {
                    format!(", {:.1} MB/frame", *bytes as f64 / frames / (1024.0 * 1024.0))
                };
                println!(
                    "  {label:<32} {ms:>8.1} ms  {:>5.2}% of the window  over {hits} entries{vol}",
                    100.0 * ms / total_ms.max(1e-6)
                );
            }
        }

        if top > 0 && !s.by_selector.is_empty() {
            let host = sched.host();
            let frames = frame_ms.len() as f64;
            println!("bench: top {top} host calls by total time:");
            println!(
                "  {:<34} {:>10} {:>9} {:>10} {:>9}",
                "nid", "calls", "per frame", "total ms", "us/call"
            );
            for c in s.by_selector.iter().take(top) {
                let name = match host.import_at(c.selector) {
                    Some((_, func_nid)) => {
                        let n = nid::name(func_nid);
                        if n.is_empty() || n == "?" {
                            format!("{func_nid:#010x}")
                        } else {
                            n.to_string()
                        }
                    }
                    None => format!("selector {}", c.selector),
                };
                println!(
                    "  {:<34} {:>10} {:>9.1} {:>10.1} {:>9.2}",
                    name,
                    c.calls,
                    c.calls as f64 / frames,
                    c.ns as f64 / 1e6,
                    c.ns as f64 / 1e3 / c.calls.max(1) as f64
                );
            }
        }
    } else {
        println!("bench: set VITASLOP_PERF=1 for the host-call split (not on by default - two \
                  clock reads per host call would themselves be measurable)");
    }

    if json {
        println!(
            "BENCHJSON {{\"frames\":{},\"p50_ms\":{:.3},\"p95_ms\":{:.3},\"mean_ms\":{:.3},\
             \"window_s\":{:.3},\"host_calls\":{},\"import_pct\":{:.2},\
             \"clock_ms_per_frame\":{:.3},\"clock_ratio\":{:.3},\"fuel_per_frame\":{}}}",
            frame_ms.len(),
            pct(frame_ms, 0.50),
            pct(frame_ms, 0.95),
            total_ms / frame_ms.len() as f64,
            window_s,
            host_calls,
            import_pct,
            win.clock_us as f64 / 1000.0 / n,
            win.clock_us as f64 / n / (1_000_000.0 / 60.0),
            (win.fuel as f64 / n) as u64,
        );
    }
}
