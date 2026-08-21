//! The generic recipe runner: boot a retail title, replay a [`Recipe`]'s input
//! timeline, sample its declared memory watches every frame, evaluate its assertions,
//! grab its screenshots, and return a structured [`RecipeReport`]. This is the harness
//! behind the fast author-run-observe-adjust loop an agent uses to play a game.
//!
//! Nothing here is game specific. Every title-specific fact - where the live state
//! lives, what "level complete" looks like, when to screenshot - is data in the
//! recipe. The runner just interprets it against the running guest, so the same code
//! drives any title for which someone has authored a recipe.

use std::path::PathBuf;

use vitaslop_loader as loader;
use vitaslop_runtime::ingest::pipeline::decrypt_container;
use vitaslop_runtime::ingest::vfs::{DirVfs, Vfs};
use vitaslop_runtime::link::link;
use vitaslop_runtime::recipe::Recipe;
// The evaluator itself lives in `vitaslop-runtime` so the browser reaches the same one;
// see `recipe_eval` for why a wasm32 build could not use this module's own copy.
use vitaslop_runtime::recipe_eval::{AssertOutcome, RecipeEval};

use crate::observe::{signature, write_shot};
use crate::{RunReport, ThreadedScheduler, VitaEnv};

/// Options controlling one recipe run.
pub struct RunOpts {
    /// Stop after this many display flips (bounds a title whose render loop never
    /// returns).
    pub max_frames: u64,
    /// Overall round (thread-resume) cap, a live-lock backstop.
    pub max_rounds: u64,
    /// Per-flip round budget while stepping frame by frame.
    pub per_frame_rounds: u64,
    /// Scheduler preemption quantum in fuel units.
    pub quantum_fuel: u64,
    /// The frame to begin per-frame observation at. Frames before it run in one fast
    /// batch (no watch sampling). `None` = auto: 0 when the recipe declares watches
    /// (so the whole watch log is captured), else just before the first assert/shot.
    pub observe_from: Option<u64>,
    /// Where to write screenshots (`@shot`). `None` = do not render shots.
    pub shot_dir: Option<PathBuf>,
    /// Override the recipe's `@shot-every`: auto-screenshot every N observed frames.
    /// `Some(0)` disables cadence shots even if the recipe sets one; `None` uses the
    /// recipe's value.
    pub shot_every: Option<u64>,
    /// How many completed scenes the capture keeps
    /// ([`Capture::scene_limit`](vitaslop_runtime::capture::Capture::scene_limit)).
    ///
    /// ONE by default, which is all a run reads (`@shot` renders the latest scene,
    /// and the signature folds each scene as it is evicted). Retaining more is
    /// megabytes a frame of per-draw vertex snapshots nobody looks at; a
    /// 4000-frame run of a real 3D title retaining all of them costs gigabytes.
    /// Raise it only for a tool that genuinely inspects a window of past frames.
    pub scene_limit: Option<usize>,
}

impl Default for RunOpts {
    fn default() -> Self {
        RunOpts {
            max_frames: 4000,
            max_rounds: 400_000_000,
            per_frame_rounds: 4_000_000,
            quantum_fuel: 5_000_000,
            observe_from: None,
            shot_dir: None,
            shot_every: None,
            scene_limit: Some(1),
        }
    }
}

/// The outcome of one screenshot request.
#[derive(Clone, Debug)]
pub struct ShotOutcome {
    pub frame: u64,
    pub name: String,
    /// The written path, or `None` if no shot dir was set or no scene was available.
    pub path: Option<PathBuf>,
}

/// The structured result of running a recipe: everything an agent or a test needs to
/// judge the run and decide the next edit.
#[derive(Clone, Debug)]
pub struct RecipeReport {
    /// Frames actually reached.
    pub frames: u64,
    /// The engine's verdict.
    pub run: RunReport,
    /// The determinism signature over the observable output (render stream + egress),
    /// comparable across engines and runs - or `None` on a run that did not fold one.
    ///
    /// `None` is not a failure: folding costs 3.5 MB a frame (8.0% of a race frame) and a
    /// run only pays it when something will read the number - a recipe that declares `@sig`,
    /// or `VITASLOP_SIGNATURE=1`. An `Option` rather than a sentinel because a sentinel
    /// compares unequal and reads like a real hash, which is exactly the confusion
    /// `Capture::signature`'s own refusal exists to avoid.
    pub sig: Option<u64>,
    /// Every assertion's outcome, in recipe order.
    pub asserts: Vec<AssertOutcome>,
    /// Every screenshot's outcome.
    pub shots: Vec<ShotOutcome>,
    /// The per-frame watch log as CSV (empty when no watches were declared).
    pub watch_csv: String,
    /// Who actually got the CPU over the run, when `VITASLOP_CPU_SHARE` is set - see
    /// [`vitaslop_runtime::sched::SchedCore::cpu_share_report`]. Empty otherwise.
    pub cpu_share: String,
    /// The egress ledger, formatted one event per line.
    pub egress: Vec<String>,
}

impl RecipeReport {
    /// True when every assertion passed (an empty assertion set counts as passing).
    pub fn all_passed(&self) -> bool {
        self.asserts.iter().all(|a| a.passed)
    }

    /// A compact, agent-friendly plaintext report: sections of the run's verdict, each
    /// assertion's pass/fail with actual-vs-expected, shot paths, and the signature.
    pub fn render_text(&self) -> String {
        let mut s = String::new();
        let passed = self.asserts.iter().filter(|a| a.passed).count();
        let sig = match self.sig {
            Some(s) => format!("{s:#018x}"),
            None => "(not folded - no reader; set VITASLOP_SIGNATURE=1)".to_string(),
        };
        s.push_str(&format!("RUN {:?} frames={} sig={sig}\n", self.run, self.frames));
        for a in &self.asserts {
            s.push_str(&format!(
                "ASSERT f{:<5} {:<7} {} -> {}\n",
                a.frame,
                if a.passed { "PASS" } else { "FAIL" },
                a.desc,
                a.detail
            ));
        }
        for sh in &self.shots {
            match &sh.path {
                Some(p) => s.push_str(&format!("SHOT   f{:<5} {} -> {}\n", sh.frame, sh.name, p.display())),
                None => s.push_str(&format!("SHOT   f{:<5} {} -> (not written)\n", sh.frame, sh.name)),
            }
        }
        for e in &self.egress {
            s.push_str(&format!("EGRESS {e}\n"));
        }
        s.push_str(&format!("RESULT {passed}/{} asserts passed\n", self.asserts.len()));
        // Under `VITASLOP_DBG_CALLSITES`, end with the hottest `(nid, caller)` pairs.
        // A recipe run is often the ONLY way to reach the code being investigated - a
        // call that needs a touch on screen cannot be provoked by a headless boot probe
        // at all - so the call-site profile has to be readable from here too, not only
        // from the probe and the resident session.
        if std::env::var("VITASLOP_DBG_CALLSITES").is_ok() {
            s.push_str(&vitaslop_runtime::vita::call_sites_report(60));
            s.push('\n');
        }
        // Under `VITASLOP_CPU_SHARE`, who actually got the single baton.
        s.push_str(&self.cpu_share);
        s
    }

    /// Dump the per-PC block-entry histogram gathered under `VITASLOP_BLOCK_HIST`, for
    /// the same reason the call-site report is printed here: once host calls are no
    /// longer the hot path, the only thing that names where guest time goes is this
    /// histogram, and a recipe run is often the only way to reach the code in question.
    /// It goes to stderr (the histogram can be hundreds of lines) rather than into the
    /// report string, and is a no-op when the knob is unset.
    pub fn dump_block_hist(&self) {
        if let Ok(top) = std::env::var("VITASLOP_BLOCK_HIST") {
            crate::dump_block_hist(top.trim().parse().unwrap_or(40));
        }
    }
}

/// Boot a decrypted retail title from its extracted app dir, wired to `world` for
/// input. Shared by the recipe runner, the boot probe, and the desktop front-end.
pub fn boot_retail(
    dir: &str,
    world: Box<dyn vitaslop_runtime::World + Send>,
    quantum_fuel: u64,
) -> Result<ThreadedScheduler<VitaEnv>, String> {
    let game = decrypt_container(&mut DirVfs::new(dir)).map_err(|e| format!("decrypt: {e:?}"))?;
    let modules: Vec<loader::Module> = game
        .modules
        .iter()
        .map(|m| loader::load(&m.elf).map_err(|e| format!("load module: {e:?}")))
        .collect::<Result<_, _>>()?;
    let linked = link(modules).map_err(|e| format!("link: {e:?}"))?;
    let mut env = VitaEnv::new(linked.imports.clone(), linked.base, linked.mem_bytes, world);
    env.state.set_alloc_base(linked.alloc_base);
    env.state.set_process_param(linked.process_param);
    env.state.set_modules(linked.loaded_modules.clone());
    // The TLS template seeds each thread's thread-local block; without it an early
    // thread-local access reads uninitialized memory and traps (MemoryOutOfBounds at
    // boot). The boot probe sets this too - keep the shared helper in parity.
    env.state.set_tls_template(linked.tls_template);
    env.state.set_preemptive(true);
    for path in game.files.list() {
        if let Ok(bytes) = game.files.read(&path) {
            env.state.add_file(&path, bytes);
        }
    }
    let (sched, _stubs) = ThreadedScheduler::from_linked(&linked, env, quantum_fuel)
        .map_err(|e| format!("scheduler: {e:?}"))?;
    Ok(sched)
}

/// Run `recipe` against the title at `game_dir` and return the structured report.
pub fn run_recipe(game_dir: &str, recipe: &Recipe, opts: RunOpts) -> Result<RecipeReport, String> {
    // The world drives input from a clone of the timeline; the metadata (watches,
    // asserts, shots) stays with `recipe` for the runner to interpret.
    let world = recipe.clone().into_world();
    let mut sched = boot_retail(game_dir, Box::new(world), opts.quantum_fuel)?;
    sched.host().state.capture.scene_limit = opts.scene_limit;
    // >>> A RECIPE IS NOT ITSELF A READER OF THE DETERMINISM SIGNATURE, and folding one
    // costs 3.5 MB a frame - MEASURED at 8.0% of a desktop race frame. The only consumer is
    // `RecipeEval::finish`, which uses the number if and only if the recipe DECLARES `@sig`.
    // The BROWSER has been gated on that declaration since 2026-08-19e; this side was not,
    // so every desktop recipe run - every `--headless` render and every `bench` on this
    // machine - has been paying it to produce a hash nothing compared, and every desktop
    // measurement carried 8% of work the shipping engine does not do.
    //
    // `VITASLOP_SIGNATURE=1` asks for one deliberately, which is what a run whose POINT is
    // to learn the signature and bless it into a recipe wants. Same spelling as the browser,
    // so the two engines decide this the same way.
    let want_sig = recipe.meta.sig.is_some() || vitaslop_runtime::knobs::flag("VITASLOP_SIGNATURE");
    sched.host().state.capture.set_signature_wanted(want_sig);

    // Auto-pick the observation start: full log when watching, else just before the
    // first assert/shot so a deep-level run does not step thousands of idle frames.
    let has_watch = !recipe.watches.is_empty();
    let auto_from = if has_watch {
        0
    } else {
        recipe
            .asserts
            .iter()
            .map(|a| a.frame)
            .chain(recipe.shots.iter().map(|s| s.frame))
            .min()
            .unwrap_or(0)
            .saturating_sub(2)
    };
    let observe_from = opts.observe_from.unwrap_or(auto_from).min(opts.max_frames);

    // The recipe's observations - watches, assertions, the shot cadence - are evaluated
    // by the SHARED evaluator in `vitaslop-runtime`, so a browser run of the same recipe
    // reaches the same verdict. A CLI `--shot-every` overrides the recipe's own cadence.
    // A cadence shot is named "<section>-f<frame>" (or "f<frame>" outside any section)
    // so the frames sort and carry context for human review.
    let mut eval = RecipeEval::new(recipe, opts.shot_every);

    let mut last = RunReport::FramesReached(0);
    if observe_from > 0 {
        last = sched.run_frames(observe_from, opts.max_rounds);
    }

    let mut shots_out: Vec<ShotOutcome> = Vec::new();

    // A guest that trapped or halted DURING the fast-forward prefix is finished, and
    // stepping it further resumes an already-completed fiber (a host panic). Report
    // what happened instead: the verdict is the whole point of the run, and it must
    // survive being reached from the batch path as readably as from the stepped one.
    let prefix_ok = matches!(last, RunReport::FramesReached(_));
    while prefix_ok && sched.frames() < opts.max_frames {
        let target = sched.frames() + 1;
        last = sched.run_frames(target, opts.per_frame_rounds);
        let f = sched.frames();

        // Sample the watches and evaluate the assertions due at this frame; the returned
        // names are the screenshots this frame owes. Rendering one is the only part that
        // stays engine-specific - a PNG on disk here, a canvas in the browser.
        let shots = {
            let host = sched.host();
            eval.on_frame(f, &SchedRead(&sched), &host.state.capture)
        };
        for name in shots {
            let path = write_shot(&sched, opts.shot_dir.as_deref(), &name);
            shots_out.push(ShotOutcome { frame: f, name, path });
        }

        // Stop early if the guest finished or trapped (not just a flip).
        if !matches!(last, RunReport::FramesReached(_)) {
            break;
        }
    }

    let frames = sched.frames();

    // Determinism signature + egress ledger from the captured output.
    // The signature is only ASKED FOR when it was folded - calling `signature()` on a run
    // that was not folding returns a sentinel and warns, which is the right refusal but not
    // something to provoke on every ordinary run.
    let (sig, egress) = {
        let host = sched.host();
        let cap = &host.state.capture;
        (
            want_sig.then(|| signature(cap)),
            cap.egress.iter().map(|e| format!("f{:<5} {:?}", e.frame, e.kind)).collect(),
        )
    };

    // Close the run: assertions past the frame actually reached are FAILURES (a run that
    // stalled short has not passed the checks it never ran), and a pinned `@sig` is
    // itself an assertion. A recipe that declares `@sig` always folded, so the number the
    // assertion needs is always there.
    eval.finish(frames, sig.unwrap_or(u64::MAX));

    Ok(RecipeReport {
        frames,
        run: last,
        sig,
        asserts: eval.asserts,
        shots: shots_out,
        watch_csv: eval.watch_csv,
        egress,
        // `VITASLOP_CPU_SHARE`: which threads got the single baton. A title whose
        // background worker runs at a low priority can be starved by a high-priority
        // thread that never blocks, and from outside that is indistinguishable from
        // the title simply being busy.
        cpu_share: if std::env::var("VITASLOP_CPU_SHARE").is_ok() {
            sched.cpu_share_report()
        } else {
            String::new()
        },
    })
}

/// Reads guest memory for the shared recipe evaluator.
///
/// The whole point of the shared evaluator is that a recipe means the same thing on both
/// engines; this is the one thing each has to supply itself, because "read guest memory"
/// is a wasmtime store here and a `SharedArrayBuffer` view in the browser.
struct SchedRead<'a>(&'a ThreadedScheduler<VitaEnv>);

impl vitaslop_runtime::recipe_eval::GuestRead for SchedRead<'_> {
    fn read_into(&self, addr: u32, out: &mut [u8]) -> bool {
        self.0.read_guest_into(addr, out)
    }
}
