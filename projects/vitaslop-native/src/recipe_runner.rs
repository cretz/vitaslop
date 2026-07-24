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
use vitaslop_runtime::capture::EgressKind;
use vitaslop_runtime::ingest::pipeline::decrypt_container;
use vitaslop_runtime::ingest::vfs::{DirVfs, Vfs};
use vitaslop_runtime::link::link;
use vitaslop_runtime::recipe::{
    AssertKind, CmpOp, EgressAssert, FieldMatch, FieldOp, MemAssert, Recipe, WatchDecl,
};
use vitaslop_runtime::render;

use crate::{RunReport, ThreadedScheduler, VitaEnv};

/// Front-panel render size and clear color (the retail titles present at 960x544).
const WIDTH: u32 = 960;
const HEIGHT: u32 = 544;
const CLEAR: [u8; 4] = [0, 0, 0, 255];

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
        }
    }
}

/// The outcome of one assertion.
#[derive(Clone, Debug)]
pub struct AssertOutcome {
    pub frame: u64,
    /// Human-readable statement of what was asserted.
    pub desc: String,
    pub passed: bool,
    /// Actual-vs-expected detail, the feedback an author acts on.
    pub detail: String,
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
    /// The determinism signature over the observable output (render stream + egress).
    /// Comparable across engines and runs.
    pub sig: u64,
    /// Every assertion's outcome, in recipe order.
    pub asserts: Vec<AssertOutcome>,
    /// Every screenshot's outcome.
    pub shots: Vec<ShotOutcome>,
    /// The per-frame watch log as CSV (empty when no watches were declared).
    pub watch_csv: String,
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
        s.push_str(&format!("RUN {:?} frames={} sig={:#018x}\n", self.run, self.frames, self.sig));
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
        s
    }
}

/// Boot a decrypted retail title from its extracted app dir, wired to `world` for
/// input. Shared by the recipe runner, the boot probe, and the desktop front-end.
pub fn boot_retail(
    dir: &str,
    world: Box<dyn vitaslop_runtime::World + Send>,
    quantum_fuel: u64,
) -> Result<ThreadedScheduler<VitaEnv>, String> {
    let game = decrypt_container(&DirVfs::new(dir)).map_err(|e| format!("decrypt: {e:?}"))?;
    let modules: Vec<loader::Module> = game
        .modules
        .iter()
        .map(|m| loader::load(&m.elf).map_err(|e| format!("load module: {e:?}")))
        .collect::<Result<_, _>>()?;
    let linked = link(modules).map_err(|e| format!("link: {e:?}"))?;
    let mut env = VitaEnv::new(linked.imports.clone(), linked.base, linked.mem_bytes, world);
    env.state.set_alloc_base(linked.alloc_base);
    env.state.set_process_param(linked.process_param);
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

    // Screenshot cadence: CLI override wins, else the recipe's @shot-every. A cadence
    // shot is named "<section>-f<frame>" (or "f<frame>" outside any section) so the
    // frames sort and carry context for human review.
    let shot_every = opts.shot_every.or(recipe.meta.shot_every).filter(|&n| n > 0);

    // CSV header: frame plus each watch name.
    let mut csv = String::new();
    if has_watch {
        csv.push_str("frame");
        for w in &recipe.watches {
            csv.push(',');
            csv.push_str(&w.name);
        }
        csv.push('\n');
    }

    let mut last = RunReport::FramesReached(0);
    if observe_from > 0 {
        last = sched.run_frames(observe_from, opts.max_rounds);
    }

    // The freshest sampled value of each watch, by name, for assertion lookup.
    let mut watch_vals: std::collections::HashMap<String, f64> =
        std::collections::HashMap::new();
    let mut asserts_out: Vec<AssertOutcome> = Vec::new();
    let mut shots_out: Vec<ShotOutcome> = Vec::new();

    while sched.frames() < opts.max_frames {
        let target = sched.frames() + 1;
        last = sched.run_frames(target, opts.per_frame_rounds);
        let f = sched.frames();

        // Sample every watch this frame.
        if has_watch {
            csv.push_str(&f.to_string());
            for w in &recipe.watches {
                let v = sample_watch(&sched, w);
                csv.push(',');
                match v {
                    Some(x) => {
                        csv.push_str(&format_f64(x));
                        watch_vals.insert(w.name.clone(), x);
                    }
                    None => csv.push_str("oob"),
                }
            }
            csv.push('\n');
        }

        // Assertions due at this frame.
        for a in recipe.asserts.iter().filter(|a| a.frame == f) {
            asserts_out.push(eval_assert(&sched, a.frame, &a.kind, &watch_vals));
        }

        // Screenshots due at this frame: explicit @shot points...
        for sh in recipe.shots.iter().filter(|s| s.frame == f) {
            let path = write_shot(&sched, &opts.shot_dir, &sh.name);
            shots_out.push(ShotOutcome { frame: f, name: sh.name.clone(), path });
        }
        // ...plus a cadence shot every N frames, named by the active section.
        if let Some(n) = shot_every {
            if f % n == 0 {
                let section = recipe
                    .sections
                    .iter()
                    .rev()
                    .find(|s| s.frame <= f)
                    .map(|s| s.name.as_str());
                let name = match section {
                    Some(sec) => format!("{sec}-f{f:05}"),
                    None => format!("f{f:05}"),
                };
                let path = write_shot(&sched, &opts.shot_dir, &name);
                shots_out.push(ShotOutcome { frame: f, name, path });
            }
        }

        // Stop early if the guest finished or trapped (not just a flip).
        if !matches!(last, RunReport::FramesReached(_)) {
            break;
        }
    }

    let frames = sched.frames();

    // Any assertions/shots past the reached frame never ran: record them as failures
    // so a run that stalled short is not silently "all passed".
    for a in recipe.asserts.iter().filter(|a| a.frame > frames) {
        asserts_out.push(AssertOutcome {
            frame: a.frame,
            desc: describe_assert(&a.kind),
            passed: false,
            detail: format!("frame {} never reached (run stopped at {frames})", a.frame),
        });
    }

    // Determinism signature + egress ledger from the captured output.
    let (sig, egress) = {
        let host = sched.host();
        let cap = &host.state.capture;
        (signature(cap), cap.egress.iter().map(|e| format!("f{:<5} {:?}", e.frame, e.kind)).collect())
    };

    // If the recipe pins an expected signature, that is itself an assertion.
    if let Some(expected) = recipe.meta.sig {
        let passed = expected == sig;
        asserts_out.push(AssertOutcome {
            frame: frames,
            desc: "determinism @sig".to_string(),
            passed,
            detail: format!("expected {expected:#018x}, got {sig:#018x}"),
        });
    }

    Ok(RecipeReport {
        frames,
        run: last,
        sig,
        asserts: asserts_out,
        shots: shots_out,
        watch_csv: csv,
        egress,
    })
}

/// Sample one watched value from current guest memory, widened to `f64`.
fn sample_watch(sched: &ThreadedScheduler<VitaEnv>, w: &WatchDecl) -> Option<f64> {
    let bytes = sched.read_guest(w.addr, w.ty.width());
    if bytes.len() < w.ty.width() {
        return None;
    }
    w.ty.decode(&bytes)
}

/// Evaluate one assertion at `frame` and describe the outcome.
fn eval_assert(
    sched: &ThreadedScheduler<VitaEnv>,
    frame: u64,
    kind: &AssertKind,
    watch_vals: &std::collections::HashMap<String, f64>,
) -> AssertOutcome {
    let desc = describe_assert(kind);
    match kind {
        AssertKind::Mem(m) => eval_mem_assert(frame, m, watch_vals, desc),
        AssertKind::Egress(e) => eval_egress_assert(sched, frame, e, desc),
    }
}

fn eval_mem_assert(
    frame: u64,
    m: &MemAssert,
    watch_vals: &std::collections::HashMap<String, f64>,
    desc: String,
) -> AssertOutcome {
    match watch_vals.get(&m.watch) {
        Some(&actual) => {
            let passed = m.op.eval(actual, m.value, m.tol);
            AssertOutcome {
                frame,
                desc,
                passed,
                detail: format!("actual {}={}", m.watch, format_f64(actual)),
            }
        }
        None => AssertOutcome {
            frame,
            desc,
            passed: false,
            detail: format!("watch {:?} not declared or not sampled", m.watch),
        },
    }
}

fn eval_egress_assert(
    sched: &ThreadedScheduler<VitaEnv>,
    frame: u64,
    e: &EgressAssert,
    desc: String,
) -> AssertOutcome {
    let host = sched.host();
    // An egress event at or before this frame that matches the kind and every field.
    let hit = host
        .state
        .capture
        .egress
        .iter()
        .filter(|ev| ev.frame <= frame)
        .any(|ev| egress_matches(&ev.kind, e));
    AssertOutcome {
        frame,
        desc,
        passed: hit,
        detail: if hit {
            "matched an egress event".to_string()
        } else {
            "no matching egress event".to_string()
        },
    }
}

/// Does egress event `ev` match assertion `want` (kind plus every field matcher)?
fn egress_matches(ev: &EgressKind, want: &EgressAssert) -> bool {
    let kind_ok = match ev {
        EgressKind::SaveWrite { .. } => want.kind == "SaveWrite",
        EgressKind::Trophy { .. } => want.kind == "Trophy",
        EgressKind::ScoreSubmit { .. } => want.kind == "ScoreSubmit",
    };
    if !kind_ok {
        return false;
    }
    want.fields.iter().all(|f| field_matches(ev, f))
}

/// Evaluate one field matcher against an egress event.
fn field_matches(ev: &EgressKind, f: &FieldMatch) -> bool {
    // Resolve the field to either a string or a number, then apply the operator.
    match (ev, f.field.as_str()) {
        (EgressKind::SaveWrite { path, .. }, "path") => str_match(path, f),
        (EgressKind::SaveWrite { ascii, .. }, "ascii") => str_match(ascii, f),
        (EgressKind::SaveWrite { bytes, .. }, "bytes") => num_match(*bytes as f64, f),
        (EgressKind::Trophy { id }, "id") => num_match(*id as f64, f),
        (EgressKind::ScoreSubmit { board, .. }, "board") => num_match(*board as f64, f),
        (EgressKind::ScoreSubmit { score, .. }, "score") => num_match(*score as f64, f),
        // An unknown field for this kind never matches.
        _ => false,
    }
}

fn str_match(actual: &str, f: &FieldMatch) -> bool {
    match f.op {
        FieldOp::Eq => actual == f.value,
        FieldOp::Contains => actual.contains(&f.value),
        // Ordering ops are meaningless on strings.
        _ => false,
    }
}

fn num_match(actual: f64, f: &FieldMatch) -> bool {
    let Ok(want) = f.value.parse::<f64>() else { return false };
    match f.op {
        FieldOp::Eq => actual == want,
        FieldOp::Ge => actual >= want,
        FieldOp::Le => actual <= want,
        FieldOp::Gt => actual > want,
        FieldOp::Lt => actual < want,
        // Substring on a number is meaningless.
        FieldOp::Contains => false,
    }
}

/// A human statement of what an assertion checks.
fn describe_assert(kind: &AssertKind) -> String {
    match kind {
        AssertKind::Mem(m) => {
            let op = match m.op {
                CmpOp::Eq => "==",
                CmpOp::Ne => "!=",
                CmpOp::Lt => "<",
                CmpOp::Le => "<=",
                CmpOp::Gt => ">",
                CmpOp::Ge => ">=",
                CmpOp::Approx => "~",
            };
            if m.tol != 0.0 {
                format!("{} {} {} +-{}", m.watch, op, format_f64(m.value), format_f64(m.tol))
            } else {
                format!("{} {} {}", m.watch, op, format_f64(m.value))
            }
        }
        AssertKind::Egress(e) => {
            let fields: Vec<String> = e.fields.iter().map(|f| format!("{}{:?}{}", f.field, f.op, f.value)).collect();
            format!("egress {} {}", e.kind, fields.join(" "))
        }
    }
}

/// Render the current frame (the last captured scene) to a PNG named `name` in
/// `shot_dir`. Returns the written path, or `None` if no dir or no scene.
fn write_shot(
    sched: &ThreadedScheduler<VitaEnv>,
    shot_dir: &Option<PathBuf>,
    name: &str,
) -> Option<PathBuf> {
    let dir = shot_dir.as_ref()?;
    let scene = {
        let host = sched.host();
        host.state.capture.scenes.last().cloned()
    }?;
    std::fs::create_dir_all(dir).ok()?;
    // Supersample the software shot (VITASLOP_SSAA=N): rasterize at N x native and
    // box-downsample. Antialiases the geometric aliasing of the heavily-tessellated vehicle
    // meshes (dozens of sub-pixel triangles per final pixel, plus coincident-panel z-fighting)
    // that one sample/pixel renders as speckle - a distant 3D vehicle is unreadable at 1x and
    // clean at 2x. A review shot is occasional, so the 4x fill cost of 2x SSAA is immaterial;
    // 2x is the quality default, overridable (1 disables, higher for close scrutiny).
    let ssaa = std::env::var("VITASLOP_SSAA").ok().and_then(|s| s.parse::<u32>().ok()).filter(|&n| n >= 1).unwrap_or(2);
    let fb = render::render_scene_supersampled(&scene, WIDTH, HEIGHT, CLEAR, ssaa);
    let path = dir.join(format!("{name}.png"));
    std::fs::write(&path, fb.to_png()).ok()?;
    Some(path)
}

/// The FNV-1a determinism signature over the observable output (render stream +
/// egress). Identical to the boot probe's, so signatures are comparable across the
/// probe, the runner, and different engines.
fn signature(cap: &vitaslop_runtime::capture::Capture) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for s in &cap.scenes {
        if let Some(c) = &s.color {
            mix(&c.data_addr.to_le_bytes());
            mix(&c.format.to_le_bytes());
        }
        for d in &s.draws {
            mix(&d.vertices);
            mix(&d.indices);
            for u in &d.uniforms {
                mix(&u.to_le_bytes());
            }
        }
    }
    for ev in &cap.egress {
        mix(&ev.frame.to_le_bytes());
        mix(format!("{:?}", ev.kind).as_bytes());
    }
    h
}

/// Format an `f64` compactly: integers without a trailing `.0`.
fn format_f64(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}
