//! A RESIDENT game session: boot a title once, then drive it forward
//! command-by-command for as long as you like - press buttons, step frames, read
//! and search guest memory, grab screenshots - and write the whole thing back out
//! as a committed recipe.
//!
//! # Why this exists
//! Playing a game through the batch runner costs a full replay of the boot prefix
//! per experiment: reaching a decision frame deep in a title is a minute of
//! emulation, and an afternoon of tuning one input is that minute times fifty. The
//! prefix is not the experiment, it is the tax on the experiment. A session pays it
//! ONCE. Every later command starts from where the last one left off, so the loop
//! cost drops to the frames you actually asked for.
//!
//! # The one thing it cannot do, and what replaces it
//! A session runs FORWARD only. The transpiler turns each guest function into a
//! wasm function and each guest call into a wasm call (see
//! [`vitaslop_transpiler::abi`]), so a suspended guest thread's state lives partly
//! in a live wasm call stack, which cannot be serialized and reloaded - there is no
//! rewind, and a state snapshot is not a small piece of work. What recovers the
//! same benefit is running forward in PARALLEL: several sessions reach the branch
//! frame at once (the prefix costs one wall-clock minute total, not N), and each
//! then takes a different input. That is what the explorer does.
//!
//! # A session IS a recipe
//! Input goes through the same frame-keyed [`Timeline`] a recipe replays, and every
//! command that changes the input appends a segment to it. So the played run is
//! literally a recipe under construction: `save` writes it out, and replaying that
//! file reproduces the session exactly. Exploration and the committed artifact are
//! the same object, which is what stops a successful playthrough from being a thing
//! that happened once in someone's terminal.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use vitaslop_runtime::recipe::{
    InputSegment, NoteDecl, Recipe, Section, ShotDecl, SharedTimeline, Timeline, ValType, WatchDecl,
};
use vitaslop_runtime::{CtrlFrame, RecipeWorld, TouchFrame, VitaEnv};

use vitaslop_runtime::render;

use crate::observe;
use crate::observe::{format_f64, sample_watch, signature, write_shot};
use crate::recipe_runner::boot_retail;
use crate::{RunReport, ThreadedScheduler};

/// How a session was configured at boot.
pub struct SessionOpts {
    /// Scheduler preemption quantum in fuel units.
    pub quantum_fuel: u64,
    /// Round (thread-resume) budget for one batch of frames - the live-lock backstop.
    pub max_rounds: u64,
    /// Round budget per frame when stepping frame by frame.
    pub per_frame_rounds: u64,
    /// Where screenshots and logs are written. Never inside the repo.
    pub shot_dir: Option<PathBuf>,
    /// How many completed scenes the capture keeps
    /// ([`Capture::scene_limit`](vitaslop_runtime::capture::Capture::scene_limit)).
    ///
    /// ONE FRAME's worth, because a frame is not one scene. A racing title's race
    /// frame is fifteen scenes and only ONE of them holds the world; the last one
    /// submitted is the composite, so a session that kept a single scene could not see
    /// that title's racetrack at all and `locate` reported nothing while the whole
    /// course was on screen. [`Capture::world_scene`](vitaslop_runtime::capture::Capture::world_scene)
    /// picks the right one out of the retained frame, and this is what leaves it a
    /// frame to pick from.
    ///
    /// Retention is a RING, not growth - the cost is a fixed multiple of one frame's
    /// draws, not a per-frame leak - and eviction still folds each dropped scene into
    /// the determinism signature, so a bounded session and an unbounded one agree.
    pub scene_limit: Option<usize>,
}

/// Scenes retained by default: enough for the deepest multi-pass frame observed on a
/// retail title (fifteen), with headroom, so no title silently loses its world pass.
pub const DEFAULT_SCENE_LIMIT: usize = 24;

impl Default for SessionOpts {
    fn default() -> Self {
        SessionOpts {
            quantum_fuel: 5_000_000,
            max_rounds: 400_000_000,
            per_frame_rounds: 4_000_000,
            shot_dir: None,
            scene_limit: Some(DEFAULT_SCENE_LIMIT),
        }
    }
}

/// A booted title being played interactively.
pub struct Session {
    sched: ThreadedScheduler<VitaEnv>,
    /// The input timeline the guest's world replays, extended as we play.
    timeline: SharedTimeline,
    /// The recipe being authored: the metadata half of the same artifact (watches,
    /// sections, shots, notes). Its segments are refreshed from `timeline` on save.
    recipe: Recipe,
    opts: SessionOpts,
    /// The last verdict the scheduler returned. Once it is not `FramesReached`, the
    /// guest is finished and every further step is refused rather than resuming an
    /// already-completed fiber (which panics inside wasmtime).
    last: RunReport,
    /// Per-frame watch samples, accumulated as CSV.
    watch_csv: String,
    /// Auto-screenshot every N stepped frames, like a recipe's `@shot-every`.
    shot_every: Option<u64>,
    /// The active memory scan, if one has been started.
    scan: Option<Scan>,
    /// Scratch buffer for scan reads, kept across passes so a 256 MiB region is not
    /// reallocated on every predicate.
    scan_scratch: Vec<u8>,
    /// The previous `locate` report, so the next one can say what MOVED. That delta
    /// is how the player object is identified in a scene of hundreds of draws: it is
    /// the one whose world position responds to the pad.
    last_locate: Option<Vec<render::ObjectLoc>>,
    /// The previous `sprites` report - the same idea as `last_locate`, for the 2D path.
    last_sprites: Option<Vec<render::SpriteLoc>>,
    /// Where the capture's coordinate origin has travelled to since the first `locate` of
    /// this session, accumulated from the per-report estimates.
    ///
    /// This is the session's STABLE frame: subtract it from a raw position and the result
    /// is that position measured in the coordinates the first report used, whatever the
    /// origin has done since. An anchor OBJECT gives the same thing exactly rather than
    /// cumulatively, but only while that object stays in view - which a mesh behind a
    /// moving vehicle does not. See [`render::origin_drift`].
    drift_origin: [f32; 3],
    /// How many reports have contributed to `drift_origin`, so a caller can judge how
    /// much accumulated estimate error is in it.
    drift_updates: u32,
}

/// Options shared by `route` and `navigate --plan`: everything that describes the
/// navigation mesh and the destinations, so the two cannot drift apart in how they read
/// the same world.
struct RouteOpts {
    to: Vec<[f32; 2]>,
    from: Option<[f32; 2]>,
    id: Option<u64>,
    frame: FrameRef,
    size: (u32, u32),
    ceiling: Option<f32>,
    /// Largest rise over run that still counts as driveable ground.
    slope: f32,
    /// How far a route keeps from anything blocked, in world units.
    clearance: f32,
    /// How far an endpoint may be nudged to reach driveable ground, in map pixels.
    snap: u32,
    /// Supersample factor for the height field.
    ///
    /// This is a correctness setting, not a quality one. A railing is a few centimetres
    /// thick, which at a metre or two per map pixel is thinner than a pixel: rasterized at
    /// 1x it is hit or missed depending on where the pixel centre lands, so the mask
    /// reports a clear route straight through a fence. `render_map` resolves the height
    /// field by MAXIMUM, so supersampling keeps a thin tall thing and, if anything,
    /// thickens it - which is the safe direction for an obstacle.
    ssaa: u32,
    out: Option<String>,
    mask_out: Option<String>,
}

impl Default for RouteOpts {
    fn default() -> Self {
        RouteOpts {
            to: Vec::new(),
            from: None,
            id: None,
            frame: FrameRef::Stable,
            size: (512, 512),
            ceiling: None,
            slope: 0.5,
            clearance: 3.0,
            snap: 24,
            ssaa: 3,
            out: None,
            mask_out: None,
        }
    }
}

/// The coordinate frame a positional report is expressed in.
#[derive(Clone, Copy, Debug, PartialEq)]
enum FrameRef {
    /// Exactly what the capture said. Comparable only within one frame on a title whose
    /// origin travels.
    Raw,
    /// Measured from where the origin was at this session's first `locate`. Survives
    /// anything leaving the view, at the cost of accumulating estimate error.
    Stable,
    /// Measured from one object's placement: exact, and fails loudly the moment that
    /// object is not in the frame.
    Anchor(u64),
}

/// A progressive memory search: the classic "narrow by behaviour" value finder.
///
/// The problem it solves is the one that blocks every new title: to ASSERT anything
/// about gameplay you need an address, and a title ships no symbols. The technique
/// is to take a baseline of a region, do something in the game, and keep only the
/// slots whose value moved the way the thing you did should have moved it. Two or
/// three rounds of that takes tens of millions of candidates down to a handful, and
/// the survivors are the game's own state.
struct Scan {
    /// Region base and length in bytes.
    addr: u32,
    len: usize,
    /// How each slot is interpreted, and therefore the slot stride.
    ty: ValType,
    /// The region's bytes as of the last pass - what a predicate compares against.
    snapshot: Vec<u8>,
    /// One bit per slot: still a candidate.
    alive: Vec<u64>,
    alive_count: usize,
    /// Passes applied since the baseline (for the report).
    passes: usize,
}

impl Scan {
    fn slots(&self) -> usize {
        self.len / self.ty.width()
    }
    fn slot_addr(&self, i: usize) -> u32 {
        self.addr.wrapping_add((i * self.ty.width()) as u32)
    }
    fn is_alive(&self, i: usize) -> bool {
        self.alive[i / 64] >> (i % 64) & 1 == 1
    }
    fn kill(&mut self, i: usize) {
        self.alive[i / 64] &= !(1u64 << (i % 64));
        self.alive_count -= 1;
    }
}

/// What a scan pass keeps.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ScanPred {
    Changed,
    Unchanged,
    Increased,
    Decreased,
    /// Value equals `v` (within `tol`).
    Eq(f64, f64),
    Ne(f64),
    Gt(f64),
    Lt(f64),
    /// Value lies in `[lo, hi]` - the sanity filter that clears the enormous number
    /// of slots holding pointers, packed pixels or garbage that happen to survive a
    /// behavioural pass.
    Range(f64, f64),
}

impl Session {
    /// Boot `game_dir` and take the first command. `recipe` seeds the run: its input
    /// timeline is replayed as the session steps (so a session can start from an
    /// authored prefix), and its metadata is the header of whatever gets saved.
    pub fn boot(game_dir: &str, recipe: Recipe, opts: SessionOpts) -> Result<Session, String> {
        let timeline = Arc::new(Mutex::new(Timeline::new(recipe.segments().to_vec())));
        let world = RecipeWorld::from_timeline(timeline.clone());
        let sched = boot_retail(game_dir, Box::new(world), opts.quantum_fuel)?;
        sched.host().state.capture.scene_limit = opts.scene_limit;
        let mut s = Session {
            sched,
            timeline,
            shot_every: recipe.meta.shot_every,
            recipe,
            opts,
            last: RunReport::FramesReached(0),
            watch_csv: String::new(),
            scan: None,
            scan_scratch: Vec::new(),
            last_locate: None,
            last_sprites: None,
            drift_origin: [0.0; 3],
            drift_updates: 0,
        };
        s.reset_watch_csv();
        Ok(s)
    }

    /// The display frame the session is currently at.
    pub fn frame(&self) -> u64 {
        self.sched.frames()
    }

    /// True once the guest has stopped for good (trapped, deadlocked, or exited).
    pub fn finished(&self) -> bool {
        !matches!(self.last, RunReport::FramesReached(_))
    }

    /// Execute one command line and return the reply. `Err` is a command-level
    /// failure (bad syntax, out-of-bounds address, guest already finished); the
    /// session stays usable either way.
    pub fn execute(&mut self, line: &str) -> Result<String, String> {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            return Ok(String::new());
        }
        let (cmd, args) = match line.split_once(char::is_whitespace) {
            Some((c, a)) => (c, a.trim()),
            None => (line, ""),
        };
        match cmd {
            "help" => Ok(HELP.to_string()),
            "info" => Ok(self.cmd_info()),
            "frames" => Ok(format!("frame={}", self.frame())),
            "step" => self.cmd_step(args),
            "until" => self.cmd_until(args),
            "input" => self.cmd_input(args),
            "press" => self.cmd_press(args),
            "read" => self.cmd_read(args),
            "peek" => self.cmd_peek(args),
            "poke" => self.cmd_poke(args),
            "dump" => self.cmd_dump(args),
            "watch" => self.cmd_watch(args),
            "unwatch" => self.cmd_unwatch(args),
            "watches" => Ok(self.watch_report()),
            "watchlog" => self.cmd_watchlog(args),
            "scan" => self.cmd_scan(args),
            "shot" => self.cmd_shot(args),
            "shot-every" => self.cmd_shot_every(args),
            "section" => {
                self.recipe.sections.push(Section { frame: self.frame(), name: args.to_string() });
                Ok(format!("section {:?} at f{}", args, self.frame()))
            }
            "note" | "todo" => {
                self.recipe.notes.push(NoteDecl {
                    frame: self.frame(),
                    todo: cmd == "todo",
                    text: args.to_string(),
                });
                Ok(format!("{cmd} recorded at f{}", self.frame()))
            }
            "threads" => Ok(self.sched.host().state.debug_sync_dump()),
            "scene" => Ok(self.cmd_scene(args)),
            "locate" => Ok(self.cmd_locate(args)),
            "map" => self.cmd_map(args),
            "surface" => self.cmd_surface(args),
            "camera" => Ok(self.cmd_camera(args)),
            "calls" => Ok(vitaslop_runtime::vita::call_sites_report(
                args.trim().parse().unwrap_or(30),
            )),
            "sprites" => Ok(self.cmd_sprites(args)),
            "navigate" => self.cmd_navigate(args),
            "route" => self.cmd_route(args),
            "sig" => Ok(format!("sig={:#018x}", self.signature())),
            "egress" => Ok(self.cmd_egress()),
            "stdout" => Ok(self.cmd_console(args, false)),
            "stderr" => Ok(self.cmd_console(args, true)),
            "save" => self.cmd_save(args),
            other => Err(format!("unknown command {other:?} (try `help`)")),
        }
    }

    // --- running ---------------------------------------------------------------

    /// `step [N] [--sample]` - advance N display frames (default 1).
    ///
    /// With watches declared (or `--sample`) each frame is stepped individually so
    /// every frame lands in the watch log; otherwise the whole span runs as one
    /// batch, which is how a session fast-forwards a boot prefix at full speed.
    fn cmd_step(&mut self, args: &str) -> Result<String, String> {
        let mut n = 1u64;
        let mut sample = false;
        for tok in args.split_whitespace() {
            if tok == "--sample" {
                sample = true;
            } else {
                n = tok.parse().map_err(|_| format!("bad frame count {tok:?}"))?;
            }
        }
        self.advance(n, sample)
    }

    /// `until <watch> <op> <value> [--max N] [--tol T]` - step until a watched value
    /// satisfies a condition, or `N` frames pass (default 600). This is what turns
    /// "wait for the race to start" into one command instead of a guessed step count.
    fn cmd_until(&mut self, args: &str) -> Result<String, String> {
        let toks: Vec<&str> = args.split_whitespace().collect();
        if toks.len() < 3 {
            return Err("until <watch> <op> <value> [--max N] [--tol T]".into());
        }
        let name = toks[0];
        let op = vitaslop_runtime::recipe::CmpOp::parse(toks[1])
            .ok_or_else(|| format!("bad op {:?} (== != < <= > >= ~)", toks[1]))?;
        let want: f64 = toks[2].parse().map_err(|_| format!("bad value {:?}", toks[2]))?;
        let mut max = 600u64;
        let mut tol = 0.0f64;
        let mut i = 3;
        while i < toks.len() {
            match toks[i] {
                "--max" => {
                    max = toks
                        .get(i + 1)
                        .and_then(|s| s.parse().ok())
                        .ok_or("--max needs a frame count")?;
                    i += 2;
                }
                "--tol" => {
                    tol = toks
                        .get(i + 1)
                        .and_then(|s| s.parse().ok())
                        .ok_or("--tol needs a number")?;
                    i += 2;
                }
                other => return Err(format!("unknown option {other:?}")),
            }
        }
        let decl = self
            .recipe
            .watches
            .iter()
            .find(|w| w.name == name)
            .cloned()
            .ok_or_else(|| format!("no watch named {name:?}"))?;

        let start = self.frame();
        for _ in 0..max {
            self.advance(1, true)?;
            if self.finished() {
                break;
            }
            if let Some(v) = sample_watch(&self.sched, &decl) {
                if op.eval(v, want, tol) {
                    return Ok(format!(
                        "HIT  f{} after {} frames: {name}={}",
                        self.frame(),
                        self.frame() - start,
                        format_f64(v)
                    ));
                }
            }
        }
        let now = sample_watch(&self.sched, &decl).map(format_f64).unwrap_or("oob".into());
        Ok(format!(
            "MISS f{} after {} frames: {name}={now} never satisfied {} {}",
            self.frame(),
            self.frame() - start,
            op.keyword(),
            format_f64(want)
        ))
    }

    /// Advance `n` frames, sampling watches per frame when asked (or when any watch
    /// is declared). Returns the one-line status an agent reads to decide what next.
    fn advance(&mut self, n: u64, force_sample: bool) -> Result<String, String> {
        if self.finished() {
            return Err(format!("guest has stopped ({:?}); no further frames", self.last));
        }
        // Anything that must happen ONCE PER FRAME forces the stepped path. Missing
        // one here does not fail, it silently produces nothing - a `shot-every` that
        // writes no shots reads as "the renderer is broken", which is a long way from
        // the truth.
        let sample = force_sample || !self.recipe.watches.is_empty() || self.shot_every.is_some();
        let target = self.frame() + n;
        if sample {
            while self.frame() < target {
                let next = self.frame() + 1;
                self.last = self.sched.run_frames(next, self.opts.per_frame_rounds);
                self.sample_frame();
                if self.finished() {
                    break;
                }
            }
        } else {
            self.last = self.sched.run_frames(target, self.opts.max_rounds);
        }
        Ok(self.status())
    }

    /// One line of status: where the run is, what the scheduler said, and the
    /// current value of every watch - the whole observable state in one reply.
    fn status(&self) -> String {
        let mut s = format!("frame={} run={:?}", self.frame(), self.last);
        let w = self.watch_values();
        if !w.is_empty() {
            s.push(' ');
            s.push_str(&w);
        }
        s
    }

    /// Append this frame's watch samples to the CSV log and take any cadence shot.
    fn sample_frame(&mut self) {
        if !self.recipe.watches.is_empty() {
            let f = self.frame();
            let mut row = f.to_string();
            for w in &self.recipe.watches {
                row.push(',');
                match sample_watch(&self.sched, w) {
                    Some(v) => row.push_str(&format_f64(v)),
                    None => row.push_str("oob"),
                }
            }
            row.push('\n');
            self.watch_csv.push_str(&row);
        }
        if let Some(n) = self.shot_every {
            let f = self.frame();
            if n > 0 && f % n == 0 {
                let section = self
                    .recipe
                    .sections
                    .iter()
                    .rev()
                    .find(|s| s.frame <= f)
                    .map(|s| s.name.clone());
                let name = match section {
                    Some(sec) => format!("{sec}-f{f:05}"),
                    None => format!("f{f:05}"),
                };
                self.take_shot(&name);
            }
        }
    }

    // --- input -----------------------------------------------------------------

    /// `input <tokens...>` - set the sticky input state from the current frame on,
    /// in the recipe's own directive grammar. An empty line releases everything.
    fn cmd_input(&mut self, args: &str) -> Result<String, String> {
        let (input, touch) = Recipe::parse_input(args)?;
        let frame = self.frame();
        self.push_input(frame, input, touch);
        Ok(format!("input at f{frame}: {}", if args.is_empty() { "(released)" } else { args }))
    }

    /// `press <tokens...> [--hold N]` - hold an input for N frames (default 8) and
    /// release, stepping those frames.
    ///
    /// The hold is the point. This title samples the pad exactly once per display
    /// frame, so an input that spans fewer frames than the gap between two samples
    /// is simply never observed - which reads on screen as "the game ignored me"
    /// and sends you off tuning timing that was never the problem.
    fn cmd_press(&mut self, args: &str) -> Result<String, String> {
        let mut hold = 8u64;
        let mut tokens: Vec<&str> = Vec::new();
        let mut it = args.split_whitespace().peekable();
        while let Some(tok) = it.next() {
            if tok == "--hold" {
                hold = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or("--hold needs a frame count")?;
            } else {
                tokens.push(tok);
            }
        }
        if hold == 0 {
            return Err("--hold must be at least 1 frame (a 0-frame press is never sampled)".into());
        }
        let spec = tokens.join(" ");
        let (input, touch) = Recipe::parse_input(&spec)?;
        let start = self.frame();
        self.push_input(start, input, touch);
        let status = self.advance(hold, false)?;
        let end = self.frame();
        let (neutral_in, neutral_touch) = (CtrlFrame::default(), TouchFrame::default());
        self.push_input(end, neutral_in, neutral_touch);
        Ok(format!("pressed {spec:?} f{start}..f{end} -> {status}"))
    }

    /// Record an input segment on the shared timeline (what the guest polls) and in
    /// the recipe (what gets saved). The two must never drift, so this is the only
    /// place either is written.
    fn push_input(&mut self, frame: u64, input: CtrlFrame, touch: TouchFrame) {
        let seg = InputSegment { frame, input, touch };
        self.timeline.lock().unwrap().push(seg);
    }

    // --- memory ----------------------------------------------------------------

    /// `read <addr> <type>` - one value out of guest memory.
    fn cmd_read(&mut self, args: &str) -> Result<String, String> {
        let mut it = args.split_whitespace();
        let (Some(addr), Some(ty)) = (it.next(), it.next()) else {
            return Err("read <addr> <type>".into());
        };
        let addr = parse_addr(addr)?;
        let ty = ValType::parse(ty).ok_or_else(|| format!("bad type {ty:?}"))?;
        let mut buf = [0u8; 4];
        if !self.sched.read_guest_into(addr, &mut buf[..ty.width()]) {
            return Err(format!("{addr:#010x} is outside guest memory"));
        }
        let v = ty.decode(&buf[..ty.width()]).ok_or("decode failed")?;
        Ok(format!("{addr:#010x} {} = {}", ty.keyword(), format_f64(v)))
    }

    /// `peek <addr> <len>` - a hex + interpretation dump. Prints each word as hex,
    /// signed integer and float side by side, because when you are staring at a
    /// freshly-found candidate the question is always "which of those is it".
    fn cmd_peek(&mut self, args: &str) -> Result<String, String> {
        let mut it = args.split_whitespace();
        let addr = parse_addr(it.next().ok_or("peek <addr> <len>")?)?;
        let len: usize = it
            .next()
            .map(|s| parse_addr(s).map(|v| v as usize))
            .transpose()?
            .unwrap_or(64);
        let len = len.min(4096);
        let mut buf = vec![0u8; len.next_multiple_of(4)];
        if !self.sched.read_guest_into(addr, &mut buf) {
            return Err(format!("{addr:#010x}+{len:#x} is outside guest memory"));
        }
        let mut s = String::new();
        for (i, w) in buf.chunks_exact(4).enumerate() {
            let raw = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            s.push_str(&format!(
                "{:#010x}  {raw:#010x}  i32={:<12} f32={}\n",
                addr.wrapping_add((i * 4) as u32),
                raw as i32,
                format_f64(f32::from_le_bytes([w[0], w[1], w[2], w[3]]) as f64)
            ));
        }
        Ok(s)
    }

    /// `poke <addr> <type> <value>` - overwrite guest memory. A probe, not a cheat:
    /// writing a candidate address and seeing the car jump is the fastest proof that
    /// the address really is what you think it is.
    fn cmd_poke(&mut self, args: &str) -> Result<String, String> {
        let mut it = args.split_whitespace();
        let (Some(a), Some(t), Some(v)) = (it.next(), it.next(), it.next()) else {
            return Err("poke <addr> <type> <value>".into());
        };
        let addr = parse_addr(a)?;
        let ty = ValType::parse(t).ok_or_else(|| format!("bad type {t:?}"))?;
        let val: f64 = v.parse().map_err(|_| format!("bad value {v:?}"))?;
        let bytes: Vec<u8> = match ty {
            ValType::U8 => vec![val as u8],
            ValType::U16 => (val as u16).to_le_bytes().to_vec(),
            ValType::U32 => (val as u32).to_le_bytes().to_vec(),
            ValType::I32 => (val as i32).to_le_bytes().to_vec(),
            ValType::F32 => (val as f32).to_le_bytes().to_vec(),
        };
        // Refuse a write that would land outside guest memory rather than silently
        // doing nothing - a no-op poke reads as "the address does not matter".
        let mut probe = vec![0u8; bytes.len()];
        if !self.sched.read_guest_into(addr, &mut probe) {
            return Err(format!("{addr:#010x} is outside guest memory"));
        }
        self.sched.write_guest(addr, &bytes);
        Ok(format!("poked {addr:#010x} {} = {}", ty.keyword(), format_f64(val)))
    }

    /// `dump [addr] [len] <file>` - write a region of guest memory to a file.
    ///
    /// The point is DIFFING TWO RUNS. Two runs that replayed the same deterministic
    /// prefix and then took different input have byte-identical memory up to the
    /// moment they diverged, so a diff of their dumps is exactly the state that
    /// input touched - no background churn at all, because both runs experienced the
    /// same amount of it. That is a far sharper instrument than watching one run
    /// change over time, where every timer, animation and allocator cursor moves too
    /// (a scan for "what changed while I held the throttle" turns up thousands of
    /// slots that have nothing to do with the throttle). See the `memdiff` tool.
    fn cmd_dump(&mut self, args: &str) -> Result<String, String> {
        let toks: Vec<&str> = args.split_whitespace().collect();
        let (base, total) = self.sched.guest_region();
        let (addr, len, path) = match toks.len() {
            1 => (base, total, toks[0]),
            3 => (parse_addr(toks[0])?, parse_addr(toks[1])? as usize, toks[2]),
            _ => return Err("dump [<addr> <len>] <file>".into()),
        };
        let mut buf = vec![0u8; len];
        if !self.sched.read_guest_into(addr, &mut buf) {
            return Err(format!("{addr:#010x}+{len:#x} is outside guest memory"));
        }
        let path = PathBuf::from(path);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        // The header names the region, so a diff cannot silently compare dumps of
        // two different address ranges.
        let mut out = Vec::with_capacity(len + 16);
        out.extend_from_slice(b"VDMP");
        out.extend_from_slice(&addr.to_le_bytes());
        out.extend_from_slice(&(len as u64).to_le_bytes());
        out.extend_from_slice(&buf);
        std::fs::write(&path, &out).map_err(|e| format!("write {}: {e}", path.display()))?;
        Ok(format!("dumped {addr:#010x}+{len:#x} at f{} to {}", self.frame(), path.display()))
    }

    // --- watches ---------------------------------------------------------------

    fn cmd_watch(&mut self, args: &str) -> Result<String, String> {
        let mut it = args.split_whitespace();
        let (Some(name), Some(ty), Some(addr)) = (it.next(), it.next(), it.next()) else {
            return Err("watch <name> <type> <addr>".into());
        };
        let ty = ValType::parse(ty).ok_or_else(|| format!("bad type {ty:?}"))?;
        let addr = parse_addr(addr)?;
        self.recipe.watches.retain(|w| w.name != name);
        self.recipe.watches.push(WatchDecl { name: name.to_string(), ty, addr });
        // The CSV header names the watch set, so a changed set starts a new log.
        self.reset_watch_csv();
        Ok(format!("watching {name} {} at {addr:#010x} = {}", ty.keyword(), self.watch_value(name)))
    }

    fn cmd_unwatch(&mut self, args: &str) -> Result<String, String> {
        let name = args.trim();
        let before = self.recipe.watches.len();
        self.recipe.watches.retain(|w| w.name != name);
        if self.recipe.watches.len() == before {
            return Err(format!("no watch named {name:?}"));
        }
        self.reset_watch_csv();
        Ok(format!("dropped watch {name}"))
    }

    fn cmd_watchlog(&mut self, args: &str) -> Result<String, String> {
        let path = self.resolve_out(args.trim(), "watch.csv");
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(&path, &self.watch_csv).map_err(|e| format!("write {}: {e}", path.display()))?;
        let rows = self.watch_csv.lines().count().saturating_sub(1);
        Ok(format!("wrote {rows} rows to {}", path.display()))
    }

    /// Start a fresh CSV with a header naming the current watch set.
    fn reset_watch_csv(&mut self) {
        self.watch_csv.clear();
        if self.recipe.watches.is_empty() {
            return;
        }
        self.watch_csv.push_str("frame");
        for w in &self.recipe.watches {
            self.watch_csv.push(',');
            self.watch_csv.push_str(&w.name);
        }
        self.watch_csv.push('\n');
    }

    fn watch_value(&self, name: &str) -> String {
        match self.recipe.watches.iter().find(|w| w.name == name) {
            Some(w) => sample_watch(&self.sched, w).map(format_f64).unwrap_or("oob".into()),
            None => "?".into(),
        }
    }

    /// `name=value` for every watch, space separated.
    fn watch_values(&self) -> String {
        self.recipe
            .watches
            .iter()
            .map(|w| {
                let v = sample_watch(&self.sched, w).map(format_f64).unwrap_or("oob".into());
                format!("{}={v}", w.name)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn watch_report(&self) -> String {
        if self.recipe.watches.is_empty() {
            return "no watches declared".into();
        }
        let mut s = String::new();
        for w in &self.recipe.watches {
            let v = sample_watch(&self.sched, w).map(format_f64).unwrap_or("oob".into());
            s.push_str(&format!("{:<16} {:<4} {:#010x} = {v}\n", w.name, w.ty.keyword(), w.addr));
        }
        s
    }

    // --- the value finder -------------------------------------------------------

    fn cmd_scan(&mut self, args: &str) -> Result<String, String> {
        let (sub, rest) = match args.split_once(char::is_whitespace) {
            Some((a, b)) => (a, b.trim()),
            None => (args, ""),
        };
        match sub {
            "new" => self.scan_new(rest),
            "list" => self.scan_list(rest),
            "promote" => self.scan_promote(rest),
            "drop" | "reset" => {
                self.scan = None;
                Ok("scan dropped".into())
            }
            "changed" => self.scan_pass(ScanPred::Changed),
            "unchanged" => self.scan_pass(ScanPred::Unchanged),
            "increased" | "inc" => self.scan_pass(ScanPred::Increased),
            "decreased" | "dec" => self.scan_pass(ScanPred::Decreased),
            "eq" => {
                let mut it = rest.split_whitespace();
                let v: f64 = it.next().ok_or("scan eq <value> [+-tol]")?.parse().map_err(|_| "bad value")?;
                let tol: f64 = match it.next() {
                    Some(t) => t.trim_start_matches("+-").parse().map_err(|_| "bad tolerance")?,
                    None => 0.0,
                };
                self.scan_pass(ScanPred::Eq(v, tol))
            }
            "ne" => self.scan_pass(ScanPred::Ne(parse_num(rest)?)),
            "gt" => self.scan_pass(ScanPred::Gt(parse_num(rest)?)),
            "lt" => self.scan_pass(ScanPred::Lt(parse_num(rest)?)),
            "range" => {
                let mut it = rest.split_whitespace();
                let lo = parse_num(it.next().ok_or("scan range <lo> <hi>")?)?;
                let hi = parse_num(it.next().ok_or("scan range <lo> <hi>")?)?;
                self.scan_pass(ScanPred::Range(lo, hi))
            }
            other => Err(format!("unknown scan subcommand {other:?} (see `help`)")),
        }
    }

    /// `scan new <type> [addr len]` - baseline a region. The default region is all
    /// of guest memory, which is the right default: you do not know where the value
    /// lives, and that is the entire point.
    fn scan_new(&mut self, args: &str) -> Result<String, String> {
        let mut it = args.split_whitespace();
        let ty = ValType::parse(it.next().ok_or("scan new <type> [addr len]")?)
            .ok_or("bad type (u8|u16|u32|i32|f32)")?;
        let (base, total) = self.sched.guest_region();
        let addr = match it.next() {
            Some(a) => parse_addr(a)?,
            None => base,
        };
        let len = match it.next() {
            Some(l) => parse_addr(l)? as usize,
            None => total - (addr.wrapping_sub(base) as usize).min(total),
        };
        let width = ty.width();
        let len = len - len % width;
        if len == 0 {
            return Err("empty scan region".into());
        }
        let mut snapshot = vec![0u8; len];
        if !self.sched.read_guest_into(addr, &mut snapshot) {
            return Err(format!("{addr:#010x}+{len:#x} is outside guest memory"));
        }
        let slots = len / width;
        let mut alive = vec![u64::MAX; slots.div_ceil(64)];
        // Zero the padding bits in the last word so `alive_count` cannot drift.
        if slots % 64 != 0 {
            let last = alive.len() - 1;
            alive[last] = (1u64 << (slots % 64)) - 1;
        }
        self.scan_scratch = vec![0u8; len];
        self.scan = Some(Scan { addr, len, ty, snapshot, alive, alive_count: slots, passes: 0 });
        Ok(format!(
            "scan baseline: {slots} candidate {} slots over {addr:#010x}+{len:#x}",
            ty.keyword()
        ))
    }

    /// Apply one predicate, comparing current memory against the previous pass.
    fn scan_pass(&mut self, pred: ScanPred) -> Result<String, String> {
        let Some(scan) = self.scan.as_mut() else {
            return Err("no scan started (run `scan new <type>` first)".into());
        };
        if !self.sched.read_guest_into(scan.addr, &mut self.scan_scratch) {
            return Err("scan region is no longer readable".into());
        }
        let width = scan.ty.width();
        let before = scan.alive_count;
        for i in 0..scan.slots() {
            if !scan.is_alive(i) {
                continue;
            }
            let lo = i * width;
            let old = scan.ty.decode(&scan.snapshot[lo..lo + width]);
            let new = scan.ty.decode(&self.scan_scratch[lo..lo + width]);
            let (Some(old), Some(new)) = (old, new) else {
                scan.kill(i);
                continue;
            };
            // A NaN slot can never satisfy any predicate meaningfully and would
            // otherwise survive "changed" forever (NaN != NaN), so it is dropped.
            let keep = if new.is_nan() {
                false
            } else {
                match pred {
                    ScanPred::Changed => new != old,
                    ScanPred::Unchanged => new == old,
                    ScanPred::Increased => new > old,
                    ScanPred::Decreased => new < old,
                    ScanPred::Eq(v, tol) => (new - v).abs() <= tol,
                    ScanPred::Ne(v) => new != v,
                    ScanPred::Gt(v) => new > v,
                    ScanPred::Lt(v) => new < v,
                    ScanPred::Range(lo, hi) => new >= lo && new <= hi,
                }
            };
            if !keep {
                scan.kill(i);
            }
        }
        scan.snapshot.copy_from_slice(&self.scan_scratch);
        scan.passes += 1;
        let after = scan.alive_count;
        let passes = scan.passes;
        let mut s = format!("scan pass {passes} ({pred:?}): {before} -> {after} candidates");
        if after <= 24 {
            s.push('\n');
            s.push_str(&self.scan_list("")?);
        }
        Ok(s)
    }

    /// `scan list [n]` - the surviving candidates and their current values.
    fn scan_list(&self, args: &str) -> Result<String, String> {
        let Some(scan) = self.scan.as_ref() else { return Err("no scan started".into()) };
        let limit: usize = args.split_whitespace().next().and_then(|s| s.parse().ok()).unwrap_or(24);
        let width = scan.ty.width();
        let mut s = String::new();
        let mut shown = 0;
        for i in 0..scan.slots() {
            if !scan.is_alive(i) {
                continue;
            }
            if shown == limit {
                s.push_str(&format!("... {} more\n", scan.alive_count - shown));
                break;
            }
            let lo = i * width;
            let v = scan.ty.decode(&scan.snapshot[lo..lo + width]).unwrap_or(f64::NAN);
            s.push_str(&format!("{:#010x} = {}\n", scan.slot_addr(i), format_f64(v)));
            shown += 1;
        }
        if shown == 0 {
            s.push_str("no candidates left - the behaviour did not narrow the way you predicted\n");
        }
        Ok(s)
    }

    /// `scan promote <name> <addr>` - turn a candidate into a named watch (and hence
    /// into something a recipe can assert on).
    fn scan_promote(&mut self, args: &str) -> Result<String, String> {
        let Some(scan) = self.scan.as_ref() else { return Err("no scan started".into()) };
        let ty = scan.ty;
        let mut it = args.split_whitespace();
        let (Some(name), Some(addr)) = (it.next(), it.next()) else {
            return Err("scan promote <name> <addr>".into());
        };
        let spec = format!("{name} {} {addr}", ty.keyword());
        self.cmd_watch(&spec)
    }

    // --- output -----------------------------------------------------------------

    fn cmd_shot(&mut self, args: &str) -> Result<String, String> {
        let name = if args.is_empty() { format!("f{:05}", self.frame()) } else { args.to_string() };
        match self.take_shot(&name) {
            Some(p) => Ok(format!("shot {} -> {}", name, p.display())),
            None => Err("no shot written (no --shots dir, or no scene captured yet)".into()),
        }
    }

    fn cmd_shot_every(&mut self, args: &str) -> Result<String, String> {
        let n: u64 = args.trim().parse().map_err(|_| "shot-every <N> (0 disables)")?;
        self.shot_every = if n == 0 { None } else { Some(n) };
        self.recipe.meta.shot_every = self.shot_every;
        Ok(format!("shot-every = {n}"))
    }

    /// Render the current frame and record it in the recipe being authored.
    fn take_shot(&mut self, name: &str) -> Option<PathBuf> {
        let path = write_shot(&self.sched, self.opts.shot_dir.as_deref(), name)?;
        self.recipe.shots.push(ShotDecl { frame: self.frame(), name: name.to_string() });
        Some(path)
    }

    fn signature(&self) -> u64 {
        let host = self.sched.host();
        signature(&host.state.capture)
    }

    /// `stdout [--tail N] [--grep <text>]` (and `stderr`) - what the GUEST has printed.
    ///
    /// Retail titles ship a startling amount of their own diagnostic logging to fd 1/2, and
    /// it is the developer's own account of what the game thinks is happening: which asset
    /// failed, which state machine refused a transition, why a subsystem is retrying. The
    /// engine has been capturing it all along with nothing to read it, which meant guessing
    /// at behaviour the game was willing to explain.
    fn cmd_console(&self, args: &str, err: bool) -> String {
        let mut tail = 80usize;
        let mut needle: Option<String> = None;
        let toks: Vec<&str> = args.split_whitespace().collect();
        let mut i = 0;
        while i < toks.len() {
            match toks[i] {
                "--tail" => {
                    tail = toks.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(tail);
                    i += 1;
                }
                "--grep" => {
                    needle = toks.get(i + 1).map(|s| s.to_string());
                    i += 1;
                }
                _ => {}
            }
            i += 1;
        }
        let host = self.sched.host();
        let raw = if err { &host.state.capture.stderr } else { &host.state.capture.stdout };
        if raw.is_empty() {
            return format!("guest {} is empty", if err { "stderr" } else { "stdout" });
        }
        // Lossy: a title's log can carry non-UTF-8 (a raw pointer or a truncated string),
        // and refusing to show the rest because of one byte would be worse than a
        // replacement character.
        let text = String::from_utf8_lossy(raw);
        let lines: Vec<&str> = text
            .lines()
            .filter(|l| needle.as_ref().is_none_or(|n| l.contains(n.as_str())))
            .collect();
        let shown: Vec<&str> = lines.iter().rev().take(tail).rev().copied().collect();
        format!(
            "guest {} - {} bytes, {} matching line(s), showing last {}\n{}",
            if err { "stderr" } else { "stdout" },
            raw.len(),
            lines.len(),
            shown.len(),
            shown.join("\n")
        )
    }

    fn cmd_egress(&self) -> String {
        let host = self.sched.host();
        let lines: Vec<String> = host
            .state
            .capture
            .egress
            .iter()
            .map(|e| format!("f{:<5} {:?}", e.frame, e.kind))
            .collect();
        if lines.is_empty() {
            "no egress events".into()
        } else {
            lines.join("\n")
        }
    }

    /// `scene` - what the guest last drew.
    ///
    /// The question this answers is "is the game still SIMULATING, or only
    /// redrawing?". A world that is running has a draw list whose contents move
    /// frame to frame; one that is stuck redraws the same geometry forever. Comparing
    /// the per-scene digest across a few frames separates "nothing responds to my
    /// input" from "nothing is happening at all", and those have completely different
    /// causes.
    fn cmd_scene(&self, args: &str) -> String {
        let host = self.sched.host();
        let cap = &host.state.capture;
        // `--passes` reports the frame's whole pass structure and, per pass, the vertex
        // FORMATS its draws use. A title whose world will not map is the case this is for:
        // the observers here decode a vertex through its declared attribute list, so a
        // format that list describes and the decoder does not shows up as geometry at an
        // absurd coordinate rather than as an error, and nothing else in the session says
        // which format it was.
        if args.split_whitespace().any(|a| a == "--passes") {
            let mut out = vec![format!("frame={} retained={}", self.frame(), cap.scenes.len())];
            for (i, s) in cap.frame_scenes().iter().enumerate() {
                let mut fmts: Vec<String> = Vec::new();
                for d in &s.draws {
                    for a in d.attributes.iter() {
                        let f = format!("fmt{}x{}@{}", a.format, a.component_count, a.offset);
                        if !fmts.contains(&f) {
                            fmts.push(f);
                        }
                    }
                }
                fmts.sort();
                // The VIEWPORT the pass's draws were issued under, as the pixel extent it
                // implies (GXM's viewport is offset/scale in pixels, so the width is
                // 2*|xScale|). This is the second, independent statement of how big a pass
                // is, and it is the one to believe when it disagrees with the target: a
                // title may hand `sceGxmBeginScene` a colour surface whose width/height
                // fields are meaningless, and then the surface says 1x1 while the pass
                // really rasterizes the whole screen.
                // Both the implied extent AND the raw offset/scale: a viewport that is the
                // right SIZE but not centred on the target places the pass's image somewhere
                // other than the origin, and a later pass sampling that buffer then needs a
                // bias to find it. Printing only the size hides exactly that case.
                let vp = s.draws.first().map(|d| d.render_state.viewport).map(|v| {
                    format!(
                        "viewport={}x{}@(off {},{} scale {},{})",
                        (2.0 * v[1].abs()) as i64,
                        (2.0 * v[3].abs()) as i64,
                        v[0],
                        v[2],
                        v[1],
                        v[3]
                    )
                });
                out.push(format!(
                    "  pass{i:<2} draws={:<4} world-tris={:<7} target={} {} attrs=[{}]",
                    s.draws.len(),
                    s.world_triangles(),
                    s.color
                        .as_ref()
                        .map(|c| format!("{:#x}:{}x{}", c.data_addr, c.width, c.height))
                        .unwrap_or_else(|| "-".into()),
                    vp.unwrap_or_else(|| "viewport=-".into()),
                    fmts.join(" ")
                ));
            }
            return out.join("\n");
        }
        let Some(scene) = cap.world_scene() else { return "no scene captured yet".into() };
        // A digest over just this scene's draws, so successive frames are comparable
        // (the run signature is cumulative and always differs).
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for d in &scene.draws {
            for b in d.vertices.iter().chain(d.indices.iter()) {
                h ^= *b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            for u in &d.uniforms {
                for b in u.to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
        format!(
            "frame={} scenes={} draws={} digest={h:#018x}",
            self.frame(),
            cap.scenes.len(),
            scene.draws.len()
        )
    }

    /// Locate every object in the current frame and resolve the coordinate frame to
    /// report it in, advancing this session's cumulative origin drift on the way.
    ///
    /// Every positional command goes through here, so there is exactly one place that
    /// knows how a travelling coordinate origin is handled - and no command can quietly
    /// forget to handle it. Returns the RAW objects (the baseline the next call
    /// differences against must be raw), the origin to subtract for the requested frame,
    /// the per-report drift, and the previous report.
    fn locate_frame(
        &mut self,
        want: FrameRef,
    ) -> Result<
        (Vec<render::ObjectLoc>, [f32; 3], Option<render::OriginDrift>, Option<Vec<render::ObjectLoc>>),
        String,
    > {
        let objects = {
            let host = self.sched.host();
            let scene = host.state.capture.world_scene().ok_or("no scene captured yet")?;
            render::locate_scene(scene, observe::WIDTH, observe::HEIGHT)
        };
        let previous = std::mem::replace(&mut self.last_locate, Some(objects.clone()));
        let drift = previous
            .as_deref()
            .and_then(|p| render::origin_drift(p, &objects, 0.05))
            .filter(|d| d.reliable());
        if let Some(d) = drift {
            for c in 0..3 {
                self.drift_origin[c] += d.delta[c];
            }
            self.drift_updates += 1;
        }
        let origin = match want {
            FrameRef::Raw => [0.0; 3],
            FrameRef::Stable => self.drift_origin,
            FrameRef::Anchor(id) => {
                let hits: Vec<&render::ObjectLoc> = objects.iter().filter(|o| o.id == id).collect();
                match hits.len() {
                    1 => hits[0].world,
                    // Both are silent-wrong-answer machines if tolerated: an absent anchor
                    // would report raw coordinates that look anchored, and an ambiguous
                    // one would pick an arbitrary instance of a repeated mesh.
                    0 => {
                        return Err(format!(
                            "anchor id={id:#018x} is not among the {} objects at f{} - it has left \
                             the view. Use the STABLE frame instead, which does not depend on any \
                             one object staying visible",
                            objects.len(),
                            self.frame()
                        ))
                    }
                    n => {
                        return Err(format!(
                            "anchor id={id:#018x} appears {n} times (a repeated mesh) - an anchor \
                             must be unique; pick an id that occurs once"
                        ))
                    }
                }
            }
        };
        Ok((objects, origin, drift, previous))
    }

    /// `locate [--min-tris N] [--top N] [--moving]` - every object in this frame,
    /// where it is in the world and where it lands on screen.
    ///
    /// This is the readout that lets an agent NAVIGATE. Screenshots say where things
    /// are only to a human looking at them; `locate` says it in numbers, so "drive
    /// toward the marker" becomes arithmetic on two positions instead of a judgement
    /// call on a PNG. Which line is the player is a title fact and is found the same
    /// way anything else here is found - hold the throttle and see which world
    /// position moves. `--moving` does that comparison for you against the previous
    /// `locate` in this session.
    fn cmd_locate(&mut self, args: &str) -> String {
        let mut min_tris = 0usize;
        let mut top = usize::MAX;
        let mut moving = false;
        let mut only_id: Option<u64> = None;
        let mut anchor: Option<u64> = None;
        let mut stable = false;
        let toks: Vec<&str> = args.split_whitespace().collect();
        let mut i = 0;
        while i < toks.len() {
            match toks[i] {
                "--id" => {
                    only_id = toks
                        .get(i + 1)
                        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok());
                    i += 1;
                }
                "--anchor" => {
                    anchor = toks
                        .get(i + 1)
                        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok());
                    i += 1;
                }
                "--min-tris" => {
                    min_tris = toks.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(min_tris);
                    i += 1;
                }
                "--top" => {
                    top = toks.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(top);
                    i += 1;
                }
                "--moving" => moving = true,
                "--stable" => stable = true,
                _ => {}
            }
            i += 1;
        }
        // Motion is measured against the previous `locate`, so a caller steps, locates,
        // steps, locates and reads the delta straight out of the report.
        let want = match (anchor, stable) {
            (Some(id), _) => FrameRef::Anchor(id),
            (None, true) => FrameRef::Stable,
            (None, false) => FrameRef::Raw,
        };
        let (objects, origin, drift, previous) = match self.locate_frame(want) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let frame_note = match want {
            FrameRef::Raw => String::new(),
            FrameRef::Stable => format!(
                "positions are in this session's STABLE frame (origin moved \
                 ({:+.2},{:+.2},{:+.2}) over {} reports since the first locate)\n",
                self.drift_origin[0], self.drift_origin[1], self.drift_origin[2],
                self.drift_updates
            ),
            FrameRef::Anchor(id) => format!(
                "anchor id={id:#018x} at raw=({:.2},{:.2},{:.2}); positions below are RELATIVE \
                 to it\n",
                origin[0], origin[1], origin[2]
            ),
        };
        // The origin's own displacement since the previous report, removed from every
        // delta below so `moved` is motion through the WORLD rather than motion relative
        // to a travelling origin.
        let shift = drift.map(|d| d.delta).unwrap_or([0.0; 3]);
        let mut lines = Vec::new();
        let mut shown = 0usize;
        for o in &objects {
            if o.triangles < min_tris || shown >= top {
                continue;
            }
            if only_id.is_some_and(|want| want != o.id) {
                continue;
            }
            // Matched to the previous frame by GEOMETRY id, never by draw index - the draw
            // list is rebuilt every frame - and measured against the drift-corrected
            // expected position, so a row of identical cones is not matched post-to-
            // neighbouring-post. See `render::world_motion`.
            let delta =
                previous.as_ref().and_then(|prev| render::world_motion(prev, o, shift));
            // Drift compensation leaves a little numerical residue on bolted-down
            // geometry (the origin's displacement is estimated, not exact), so `--moving`
            // needs a threshold above that noise rather than a strict inequality.
            const MOVING_MIN: f32 = 0.1;
            if moving && delta.map(|(_, m)| m).unwrap_or(0.0) <= MOVING_MIN {
                continue;
            }
            shown += 1;
            let screen = match (o.screen, o.centroid) {
                (Some(b), Some(c)) => format!(
                    "screen=({:.0},{:.0}) box=({:.0},{:.0})-({:.0},{:.0})",
                    c[0], c[1], b[0], b[1], b[2], b[3]
                ),
                _ => "screen=offscreen".to_string(),
            };
            let dist = o.distance.map(|d| format!(" dist={d:.1}")).unwrap_or_default();
            let heading = o
                .heading
                .map(|h| format!(" heading=({:.1},{:.1})", h[0], h[1]))
                .unwrap_or_default();
            let moved = match delta {
                Some((d, m)) if m > 1e-4 => {
                    format!(" moved={m:.3} d=({:+.3},{:+.3},{:+.3})", d[0], d[1], d[2])
                }
                Some(_) => " moved=0".to_string(),
                None => String::new(),
            };
            lines.push(format!(
                "id={:#018x} draw{:<4} world=({:.2},{:.2},{:.2}){heading} {screen}{dist} tris={}{}{}",
                o.id,
                o.draws[0],
                o.world[0] - origin[0],
                o.world[1] - origin[1],
                o.world[2] - origin[2],
                o.triangles,
                if o.sprites { " sprites" } else { "" },
                moved,
            ));
        }
        // The drift line is printed ALWAYS, not just when it is interesting. A travelling
        // coordinate origin is invisible in the positions themselves - they stay smooth
        // and self-consistent - so the only defence is for every report to state what it
        // removed and how much of the scene agreed.
        let drift_note = match (previous.is_some(), drift) {
            (false, _) => "origin drift: no previous `locate` to measure against\n".to_string(),
            (true, Some(d)) => format!(
                "origin drift since last locate = ({:+.3},{:+.3},{:+.3}) agreed by {}/{} objects; \
                 `moved` below has it REMOVED, so it is motion through the world\n",
                d.delta[0], d.delta[1], d.delta[2], d.agreed, d.matched
            ),
            (true, None) => "origin drift: NO MAJORITY (a scene cut, or most of the frame \
                             genuinely moved) - `moved` is raw and may be measured against a \
                             travelling origin\n"
                .to_string(),
        };
        let header = format!(
            "frame={} objects={} shown={}\n{frame_note}{drift_note}",
            self.frame(),
            objects.len(),
            lines.len()
        );
        if lines.is_empty() {
            return format!("{header}nothing matched the filter");
        }
        format!("{header}{}", lines.join("\n"))
    }

    /// `map [--extent x0,z0,x1,z1] [--size WxH] [--ssaa N] [--grid CxR] [--step F]
    /// [--mark <hex-id>|<x>,<z>] [--out <name>]` - a top-down orthographic map of this
    /// frame's world geometry, as a PNG plus an ASCII height field.
    ///
    /// This exists because `locate` cannot see a route that is PAINTED rather than
    /// placed. A trail drawn into the ground texture has no world matrix and so no
    /// entry in a `locate` report, but it is right there in a bird's-eye render, at a
    /// pixel this command converts back to a world coordinate. The height field is the
    /// other half: the railings and benches a guessed route catches on are a metre of
    /// extra height over the ground, which is a reading rather than a surprise.
    fn cmd_map(&mut self, args: &str) -> Result<String, String> {
        let mut size = (512u32, 512u32);
        let mut grid = (96u32, 48u32);
        let mut ssaa = 2u32;
        let mut step = 1.0f32;
        let mut ceiling: Option<f32> = None;
        let mut anchor: Option<u64> = None;
        let mut stable = false;
        let mut extent: Option<[f32; 4]> = None;
        let mut out: Option<String> = None;
        let mut marks: Vec<String> = Vec::new();
        let toks: Vec<&str> = args.split_whitespace().collect();
        let mut i = 0;
        // A malformed option is an ERROR, never a silently-ignored token: a map whose
        // extent argument did not parse is a picture of the wrong place, and it looks
        // exactly like a picture of the right place.
        while i < toks.len() {
            let val = |i: usize| -> Result<&str, String> {
                toks.get(i + 1).copied().ok_or_else(|| format!("{} needs a value", toks[i]))
            };
            let pair = |s: &str, sep: char| -> Result<(u32, u32), String> {
                let (a, b) = s.split_once(sep).ok_or_else(|| format!("expected N{sep}N, got {s:?}"))?;
                Ok((
                    a.trim().parse().map_err(|_| format!("bad number {a:?}"))?,
                    b.trim().parse().map_err(|_| format!("bad number {b:?}"))?,
                ))
            };
            match toks[i] {
                "--size" => {
                    size = pair(val(i)?, 'x')?;
                    i += 1;
                }
                "--grid" => {
                    grid = pair(val(i)?, 'x')?;
                    i += 1;
                }
                "--ssaa" => {
                    ssaa = val(i)?.parse().map_err(|_| "--ssaa needs a number".to_string())?;
                    i += 1;
                }
                "--step" => {
                    step = val(i)?.parse().map_err(|_| "--step needs a number".to_string())?;
                    i += 1;
                }
                "--ceiling" => {
                    ceiling = Some(val(i)?.parse().map_err(|_| "--ceiling needs a number".to_string())?);
                    i += 1;
                }
                "--out" => {
                    out = Some(val(i)?.to_string());
                    i += 1;
                }
                "--mark" => {
                    marks.push(val(i)?.to_string());
                    i += 1;
                }
                "--anchor" => {
                    let s = val(i)?;
                    anchor = Some(
                        u64::from_str_radix(s.trim_start_matches("0x"), 16)
                            .map_err(|_| format!("bad anchor id {s:?} - want hex"))?,
                    );
                    i += 1;
                }
                "--stable" => stable = true,
                "--extent" => {
                    let v: Vec<f32> = val(i)?
                        .split(',')
                        .map(|p| p.trim().parse::<f32>().map_err(|_| format!("bad number {p:?}")))
                        .collect::<Result<_, _>>()?;
                    if v.len() != 4 {
                        return Err("--extent needs x0,z0,x1,z1".into());
                    }
                    extent = Some([v[0], v[1], v[2], v[3]]);
                    i += 1;
                }
                other => return Err(format!("unknown map option {other:?}")),
            }
            i += 1;
        }

        let scene = {
            let host = self.sched.host();
            host.state.capture.world_scene().cloned().ok_or("no scene captured yet")?
        };
        // The frame every coordinate in this report is measured from - see `locate_frame`,
        // which is the single place that knows how a travelling origin is handled.
        let want = match (anchor, stable) {
            (Some(id), _) => FrameRef::Anchor(id),
            (None, true) => FrameRef::Stable,
            (None, false) => FrameRef::Raw,
        };
        let (objects, origin, _, _) = self.locate_frame(want)?;
        let extent = match extent {
            Some(e) => e,
            // 98% of vertices: a skydome is a few vertices spanning kilometres, and
            // letting it set the extent puts the playable area in four pixels.
            None => {
                let raw = render::world_extent(&scene, 0.98)
                    .ok_or("scene has no 3D geometry to map")?;
                [raw[0] - origin[0], raw[1] - origin[2], raw[2] - origin[0], raw[3] - origin[2]]
            }
        };
        let view = render::MapView { extent, width: size.0.max(1), height: size.1.max(1) };
        let mut map = render::render_map(&scene, view, observe::CLEAR, ssaa, ceiling, origin);
        let ground = map.ground_level(0.25).unwrap_or(0.0);

        // Marks: either an object id from `locate` (the usual case - mark the player) or
        // a literal world x,z (a waypoint being considered).
        let mut mark_lines = Vec::new();
        for m in &marks {
            let (wx, wz, what) = if let Some((a, b)) = m.split_once(',') {
                let x: f32 = a.trim().parse().map_err(|_| format!("bad mark x {a:?}"))?;
                let z: f32 = b.trim().parse().map_err(|_| format!("bad mark z {b:?}"))?;
                (x, z, format!("({x:.2},{z:.2})"))
            } else {
                let id = u64::from_str_radix(m.trim_start_matches("0x"), 16)
                    .map_err(|_| format!("bad mark {m:?} - want a hex object id or x,z"))?;
                let o = objects
                    .iter()
                    .find(|o| o.id == id)
                    .ok_or_else(|| format!("no object with id {id:#018x} in this frame"))?;
                // Into the map's frame, so a marked object's printed coordinates and the
                // extent above are the same kind of number.
                (o.world[0] - origin[0], o.world[2] - origin[2], format!("id={id:#018x}"))
            };
            map.mark(wx, wz, 5, [255, 0, 255]);
            let p = map.view.pixel_of(wx, wz);
            mark_lines.push(format!(
                "mark {what} world=({wx:.2},{wz:.2}) pixel=({:.0},{:.0}) top_y={}",
                p[0],
                p[1],
                map.height_at(wx, wz).map(|h| format!("{h:.2}")).unwrap_or("none".into())
            ));
        }

        let name = out.unwrap_or_else(|| format!("map-f{:05}", self.frame()));
        let path = match self.opts.shot_dir.as_deref() {
            Some(dir) => {
                std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                let p = dir.join(format!("{name}.png"));
                std::fs::write(&p, map.fb.to_png()).map_err(|e| e.to_string())?;
                Some(p)
            }
            None => None,
        };

        let s = map.view.scale();
        let mut report = format!(
            "frame={} extent=x {:.2}..{:.2} z {:.2}..{:.2} size={}x{} scale={:.3},{:.3} world-units/px \
             ground_y={ground:.2}\n\
             up-image is world -Z, right is world +X (bearing 0 = +X, increasing toward -Z)\n{}",
            self.frame(),
            extent[0], extent[2], extent[1], extent[3],
            map.view.width, map.view.height, s[0], s[1],
            match want {
                FrameRef::Anchor(id) => format!(
                    "frame: RELATIVE to anchor id={id:#018x} at raw=({:.2},{:.2},{:.2}) - so these \
                     coordinates are comparable with other frames' maps\n",
                    origin[0], origin[1], origin[2]
                ),
                FrameRef::Stable => format!(
                    "frame: this session's STABLE frame (origin has moved \
                     ({:+.2},{:+.2},{:+.2}) over {} reports)\n",
                    origin[0], origin[1], origin[2], self.drift_updates
                ),
                FrameRef::Raw => "frame: RAW capture coordinates. If this title's origin travels \
                                  with the camera (`locate` prints a drift line saying so) these \
                                  are comparable only WITHIN this frame - pass --stable or \
                                  --anchor <id>\n"
                    .to_string(),
            }
        );
        if let Some(p) = &path {
            report.push_str(&format!("png {}\n", p.display()));
        } else {
            report.push_str("no --shots dir, so no PNG was written (the grid below still applies)\n");
        }
        for l in &mark_lines {
            report.push_str(l);
            report.push('\n');
        }
        // The height distribution, densest bands first. A single band far above
        // everything else means a depth-writing roof or sky is being mapped instead of
        // the floor - pick a `--ceiling` from these numbers rather than guessing one.
        let bins = map.height_bins(2.0);
        let shown: Vec<String> = bins
            .iter()
            .take(6)
            .map(|(h, n)| format!("y{h:.0}:{n}px"))
            .collect();
        report.push_str(&format!(
            "height bands (2.0-unit bins, densest first) {}{}\n",
            shown.join(" "),
            if bins.len() > 6 { format!(" (+{} more)", bins.len() - 6) } else { String::new() }
        ));
        if grid.0 > 0 && grid.1 > 0 {
            report.push_str(&format!(
                "height grid {}x{} step={step:.2}: ' '=unmapped '.'=ground ':'=<={step:.2} \
                 '+'=<={:.2} '#'=higher\n",
                grid.0, grid.1, step * 4.0
            ));
            report.push_str(&map.height_grid(grid.0, grid.1, ground, step));
        }
        Ok(report)
    }

    /// `sprites [--moving] [--top N] [--min-tris N] [--id <hex>] [--textured]` - every 2D
    /// drawn thing in this frame, where it is on screen, and how it moved.
    ///
    /// The 2D counterpart of `locate`, and not a convenience: a 2D title has no
    /// model-to-world matrix, so `locate` reports NOTHING for it at all - which covers a
    /// large part of the library. A sprite's identity here is its texture plus the atlas
    /// region it samples plus its size, because its position is in its vertex data and so
    /// the geometry hash a 3D mesh is identified by changes every time it moves.
    ///
    /// `--moving` has the scene's SCROLL removed, for the same reason `locate`'s does: when
    /// the camera pans, the backdrop moves and the player the camera follows does not.
    fn cmd_sprites(&mut self, args: &str) -> String {
        let mut min_tris = 0usize;
        let mut top = usize::MAX;
        let mut moving = false;
        let mut textured_only = false;
        let mut only_id: Option<u64> = None;
        let toks: Vec<&str> = args.split_whitespace().collect();
        let mut i = 0;
        while i < toks.len() {
            match toks[i] {
                "--id" => {
                    only_id = toks
                        .get(i + 1)
                        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok());
                    i += 1;
                }
                "--min-tris" => {
                    min_tris = toks.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(min_tris);
                    i += 1;
                }
                "--top" => {
                    top = toks.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(top);
                    i += 1;
                }
                "--moving" => moving = true,
                "--textured" => textured_only = true,
                _ => {}
            }
            i += 1;
        }
        let sprites = {
            let host = self.sched.host();
            let Some(scene) = host.state.capture.scenes.last() else {
                return "no scene captured yet".into();
            };
            render::locate_sprites(scene, observe::WIDTH, observe::HEIGHT)
        };
        let previous = std::mem::replace(&mut self.last_sprites, Some(sprites.clone()));
        let drift = previous
            .as_deref()
            .and_then(|p| render::scroll_drift(p, &sprites, 0.75))
            .filter(|d| d.reliable());
        let shift = drift.map(|d| d.delta).unwrap_or([0.0; 3]);
        let mut lines = Vec::new();
        let mut shown = 0usize;
        for s in &sprites {
            if s.triangles < min_tris || shown >= top {
                continue;
            }
            if only_id.is_some_and(|w| w != s.id) || (textured_only && !s.textured) {
                continue;
            }
            let motion = previous.as_ref().and_then(|p| render::sprite_motion(p, s, shift));
            // Above the residue the scroll estimate leaves on a static sprite.
            const MOVING_MIN: f32 = 0.6;
            if moving && motion.map(|(_, m)| m).unwrap_or(0.0) <= MOVING_MIN {
                continue;
            }
            shown += 1;
            let moved = match motion {
                Some((d, m)) if m > 1e-3 => format!(" moved={m:.2} d=({:+.2},{:+.2})", d[0], d[1]),
                Some(_) => " moved=0".to_string(),
                None => String::new(),
            };
            lines.push(format!(
                "id={:#018x} draw{:<4} at=({:.0},{:.0}) box=({:.0},{:.0})-({:.0},{:.0}) \
                 size={:.0}x{:.0} tris={}{}{}",
                s.id, s.draw, s.centroid[0], s.centroid[1],
                s.bbox[0], s.bbox[1], s.bbox[2], s.bbox[3],
                s.size[0], s.size[1], s.triangles,
                if s.textured { "" } else { " untextured" },
                moved,
            ));
        }
        let drift_note = match (previous.is_some(), drift) {
            (false, _) => "scroll: no previous `sprites` to measure against\n".to_string(),
            (true, Some(d)) => format!(
                "scene scrolled ({:+.2},{:+.2}) px since the last `sprites`, agreed by {}/{} \
                 sprites; `moved` below has it REMOVED\n",
                d.delta[0], d.delta[1], d.agreed, d.matched
            ),
            (true, None) => "scroll: NO MAJORITY (a scene change, or most of the frame genuinely \
                             moved) - `moved` is raw screen motion\n"
                .to_string(),
        };
        let header = format!(
            "frame={} sprites={} shown={}\n{drift_note}",
            self.frame(),
            sprites.len(),
            lines.len()
        );
        if lines.is_empty() {
            return format!("{header}nothing matched the filter");
        }
        format!("{header}{}", lines.join("\n"))
    }

    /// `camera` - where the view is and which way it looks, this frame.
    ///
    /// Reconstructed from the world-to-clip matrix the guest drew with, so unlike an
    /// address found by memory diffing it cannot go stale: a title's per-frame matrix
    /// pool moves the vehicle between slots, and the reading from a slot it has left
    /// freezes at a value that looks exactly like a car stopped against a wall.
    /// `--passes` reports EVERY pass of the frame with its own recovered camera, marking
    /// the one `world_scene` selected. This is the instrument that distinguishes a car
    /// that is spinning from a reading that is jumping between two passes: a race frame
    /// carries a rear-view MIRROR pass, whose camera is a legitimate reconstruction of a
    /// view pointing the other way, so a selection that flips between it and the main view
    /// reports a heading that flips 180 degrees while the car drives dead straight.
    fn cmd_camera(&self, args: &str) -> String {
        let host = self.sched.host();
        let fmt = |e: &render::Eye| {
            format!(
                "eye=({:.2},{:.2},{:.2}) dir=({:.4},{:.4},{:.4}) bearing={:.2}",
                e.pos[0], e.pos[1], e.pos[2], e.dir[0], e.dir[1], e.dir[2], e.bearing
            )
        };
        if args.split_whitespace().any(|a| a == "--passes") {
            let frame = host.state.capture.frame_scenes();
            let chosen = host.state.capture.world_scene().map(|s| s as *const _);
            let mut out = vec![format!("frame={} passes={}", self.frame(), frame.len())];
            for (i, s) in frame.iter().enumerate() {
                let mark = if Some(s as *const _) == chosen { "<- world_scene" } else { "" };
                let target = match &s.color {
                    Some(c) => format!("{:#x}:{}x{}", c.data_addr, c.width, c.height),
                    None => "none".into(),
                };
                let cam = match render::scene_eye(s) {
                    Some(e) => fmt(&e),
                    None => "no world-to-clip matrix".into(),
                };
                out.push(format!(
                    "  pass{i:<3} tris={:<7} target={target:<22} {cam} {mark}",
                    s.world_triangles()
                ));
            }
            return out.join("\n");
        }
        let Some(scene) = host.state.capture.world_scene() else {
            return "no scene captured yet".into();
        };
        match render::scene_eye(scene) {
            Some(e) => format!("frame={} {}", self.frame(), fmt(&e)),
            None => format!("frame={} no world-to-clip matrix in this frame", self.frame()),
        }
    }

    /// `surface --at <x>,<z> [--ceiling Y]` - what surface is under that world point.
    /// `surface --tex <hex> --from <x>,<z> --bearing <deg> [--fov F] [--range R] [--top N]`
    ///     - how far that surface reaches in front of you, and where its far edge is.
    ///
    /// # Why
    /// Steering along a ROAD needs to know where the road is, and a height field cannot
    /// say: tarmac and the grass beside it are the same height, so a slope-derived
    /// traversable mask calls the whole valley drivable. The distinction is the MATERIAL,
    /// which the capture already carries per draw. `--at` reads the material the vehicle
    /// is standing on (ask it where the vehicle is certainly legal - on the grid), and
    /// `--tex` then measures that same material ahead, which is an aim point.
    fn cmd_surface(&mut self, args: &str) -> Result<String, String> {
        let mut at: Option<[f32; 2]> = None;
        let mut from: Option<[f32; 2]> = None;
        // A SET, because one surface is rarely one material: a circuit changes tarmac
        // texture between sectors, and a controller that re-identified the road every tick
        // would follow whatever it happened to be standing on the moment it ran wide.
        let mut tex: Vec<u32> = Vec::new();
        let mut bearing = 0f32;
        let mut fov = 100f32;
        let mut range = 400f32;
        let mut top = 8usize;
        let mut ceiling: Option<f32> = None;
        let parts: Vec<&str> = args.split_whitespace().collect();
        let pair = |s: &str| -> Result<[f32; 2], String> {
            let (a, b) = s.split_once(',').ok_or_else(|| format!("expected x,z not {s:?}"))?;
            Ok([a.trim().parse().map_err(|_| "bad x")?, b.trim().parse().map_err(|_| "bad z")?])
        };
        let mut i = 0;
        while i < parts.len() {
            let v = parts.get(i + 1).copied().unwrap_or("");
            match parts[i] {
                "--at" => at = Some(pair(v)?),
                "--from" => from = Some(pair(v)?),
                "--tex" => {
                    for t in v.split(',').filter(|s| !s.is_empty()) {
                        tex.push(
                            u32::from_str_radix(t.trim().trim_start_matches("0x"), 16)
                                .map_err(|_| "bad --tex")?,
                        );
                    }
                }
                "--bearing" => bearing = v.parse().map_err(|_| "bad --bearing")?,
                "--fov" => fov = v.parse().map_err(|_| "bad --fov")?,
                "--range" => range = v.parse().map_err(|_| "bad --range")?,
                "--top" => top = v.parse().map_err(|_| "bad --top")?,
                "--ceiling" => ceiling = Some(v.parse().map_err(|_| "bad --ceiling")?),
                _ => {}
            }
            i += 2;
        }
        let scene = {
            let host = self.sched.host();
            host.state.capture.world_scene().cloned().ok_or("no scene captured yet")?
        };
        let tris = render::surface_tris(&scene);
        if let Some(p) = at {
            let Some((hit, y)) = render::surface_at(&tris, p[0], p[1], ceiling) else {
                return Ok(format!(
                    "frame={} no surface covers ({:.2},{:.2}) among {} triangles",
                    self.frame(),
                    p[0],
                    p[1],
                    tris.len()
                ));
            };
            let same = tris.iter().filter(|t| t.tex == hit.tex).count();
            return Ok(format!(
                "frame={} at=({:.2},{:.2}) y={y:.2} draw={} tex={:#010x} tris_of_tex={same} scene_tris={}",
                self.frame(),
                p[0],
                p[1],
                hit.draw,
                hit.tex,
                tris.len()
            ));
        }
        let Some(from) = from.filter(|_| !tex.is_empty()) else {
            return Err("surface needs --at <x>,<z>, or --tex <hex>[,<hex>...] --from <x>,<z>".into());
        };
        // Forward reach: every triangle of that material whose centroid is inside the cone,
        // ranked by how far ahead it is. The FARTHEST is the aim point - "drive as far along
        // this surface as you can see" needs no track model and no waypoints authored by eye.
        let (sb, cb) = (bearing.to_radians().sin(), bearing.to_radians().cos());
        let mut hits: Vec<(f32, f32, [f32; 3])> = Vec::new();
        for t in tris.iter().filter(|t| tex.contains(&t.tex)) {
            let c = t.centroid();
            // Bearing convention matches `locate` and `lang=`: 0 is world +X, increasing
            // toward world -Z.
            let (dx, dz) = (c[0] - from[0], c[2] - from[1]);
            let dist = (dx * dx + dz * dz).sqrt();
            if dist > range || dist < 1e-3 {
                continue;
            }
            let ahead = (dx * cb + (-dz) * sb) / dist;
            let off = ahead.clamp(-1.0, 1.0).acos().to_degrees();
            if off > fov * 0.5 {
                continue;
            }
            hits.push((dist, off, c));
        }
        if hits.is_empty() {
            return Ok(format!(
                "frame={} tex={tex:02x?} no surface within {range:.0} ahead of ({:.2},{:.2}) bearing {bearing:.1}",
                self.frame(),
                from[0],
                from[1]
            ));
        }
        hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = vec![format!(
            "frame={} tex={tex:02x?} from=({:.2},{:.2}) bearing={bearing:.1} fov={fov:.0} reach={} tris",
            self.frame(),
            from[0],
            from[1],
            hits.len()
        )];
        for (dist, off, c) in hits.iter().take(top) {
            let b = (-(c[2] - from[1])).atan2(c[0] - from[0]).to_degrees();
            out.push(format!(
                "  at=({:.2},{:.2}) y={:.2} dist={dist:.1} bearing={:.1} off={off:.1}",
                c[0],
                c[2],
                c[1],
                if b < 0.0 { b + 360.0 } else { b }
            ));
        }
        Ok(out.join("\n"))
    }

    /// Build a navigation mesh for the current frame: the top-down map, and a
    /// traversability mask over it.
    ///
    /// `slope` and `clearance` are given in WORLD terms - a rise-over-run ratio and a body
    /// width - and converted to the map's pixel scale here. Expressing them in pixels
    /// instead would make every route depend on the resolution the map happened to be
    /// rendered at, which is the kind of hidden coupling that makes a tool untrustworthy.
    fn navmesh(
        &mut self,
        want: FrameRef,
        size: (u32, u32),
        ceiling: Option<f32>,
        slope: f32,
        clearance: f32,
        ssaa: u32,
    ) -> Result<(render::WorldMap, render::Traversable, [f32; 3]), String> {
        let scene = {
            let host = self.sched.host();
            host.state.capture.world_scene().cloned().ok_or("no scene captured yet")?
        };
        let (_, origin, _, _) = self.locate_frame(want)?;
        let raw = render::world_extent(&scene, 0.98).ok_or("scene has no 3D geometry to map")?;
        let extent =
            [raw[0] - origin[0], raw[1] - origin[2], raw[2] - origin[0], raw[3] - origin[2]];
        let view = render::MapView { extent, width: size.0.max(1), height: size.1.max(1) };
        let map = render::render_map(&scene, view, observe::CLEAR, ssaa, ceiling, origin);
        let s = map.view.scale();
        let unit = s[0].max(s[1]).max(1e-6);
        let rise = slope * unit;
        let clear_px = (clearance / unit).round().max(0.0) as u32;
        let mask = render::Traversable::from_map(&map, rise, clear_px);
        Ok((map, mask, origin))
    }

    /// `route --to <x>,<z> [--to ...] [--from <x>,<z> | --id <hex>] [--slope R]
    /// [--clearance W] [--size WxH] [--ceiling Y] [--snap N] [--stable|--anchor <hex>]
    /// [--out <name>]` - a driveable route to each destination in turn, computed from the
    /// frame's own height field.
    ///
    /// Waypoints picked off a map by eye encode only what the eye noticed; every railing,
    /// bench and kerb between two of them is an obstacle the route knows nothing about, and
    /// the vehicle discovers it by driving into it. The height field already holds those
    /// obstacles, so the route is computed rather than guessed. Feed the printed waypoints
    /// to `navigate`, or let `navigate --plan` do both.
    fn cmd_route(&mut self, args: &str) -> Result<String, String> {
        let p = self.parse_route_opts(args)?;
        let (mut map, mask, origin) =
            self.navmesh(p.frame, p.size, p.ceiling, p.slope, p.clearance, p.ssaa)?;
        let start = self.route_start(&p, origin)?;
        let (legs, notes) = self.plan_legs(&map, &mask, start, &p)?;
        let mut out = format!(
            "route from ({:.1},{:.1}) over {} destination(s): {} waypoint(s)\n\
             mask: {:.0}% of the mapped area is driveable (slope <= {:.2}, clearance {:.1} \
             world units = {} px at {:.2} units/px)\n",
            start[0], start[1], p.to.len(), legs.len(),
            mask.open_fraction() * 100.0, p.slope, p.clearance,
            (p.clearance / map.view.scale()[0].max(map.view.scale()[1])).round() as u32,
            map.view.scale()[0].max(map.view.scale()[1]),
        );
        for n in &notes {
            out.push_str(n);
            out.push('\n');
        }
        // Printed as a ready-to-paste `navigate` argument list: the next thing anyone does
        // with a route is drive it.
        out.push_str("--to ");
        out.push_str(
            &legs.iter().map(|w| format!("{:.1},{:.1}", w[0], w[1])).collect::<Vec<_>>().join(" --to "),
        );
        out.push('\n');
        // The mask the route was planned over, as an image. "No route" and "a route
        // straight through a fence" are the same symptom - the mask disagreeing with the
        // world - and looking at it is the only way to tell them apart.
        if let Some(name) = p.mask_out.clone() {
            let dir = self
                .opts
                .shot_dir
                .clone()
                .ok_or("--mask-out needs a --shots directory to write into")?;
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            let path = dir.join(format!("{name}.png"));
            let png = render::rgba_to_png(mask.width, mask.height, &mask.to_rgba());
            std::fs::write(&path, png).map_err(|e| e.to_string())?;
            out.push_str(&format!("mask png {}\n", path.display()));
        }
        // The route drawn on the map it was computed from: the one artifact that shows both
        // the plan and the ground it was planned over.
        if let Some(name) = p.out.clone() {
            let dir = self
                .opts
                .shot_dir
                .clone()
                .ok_or("--out needs a --shots directory to write into")?;
            map.mark(start[0], start[1], 6, [0, 255, 255]);
            for w in &legs {
                map.mark(w[0], w[1], 4, [255, 0, 255]);
            }
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            let path = dir.join(format!("{name}.png"));
            std::fs::write(&path, map.fb.to_png()).map_err(|e| e.to_string())?;
            out.push_str(&format!("png {}\n", path.display()));
        }
        Ok(out)
    }

    /// Parse the route/navmesh options. Unknown options and malformed values are ERRORS: a
    /// route computed with a silently-dropped `--clearance` is a route through a railing.
    fn parse_route_opts(&self, args: &str) -> Result<RouteOpts, String> {
        let mut p = RouteOpts::default();
        let mut stable = false;
        let mut anchor: Option<u64> = None;
        let toks = split_quoted(args);
        let mut i = 0;
        while i < toks.len() {
            let need = |i: usize| -> Result<&str, String> {
                toks.get(i + 1).map(|s| s.as_str()).ok_or_else(|| format!("{} needs a value", toks[i]))
            };
            let num = |s: &str| -> Result<f32, String> {
                s.parse().map_err(|_| format!("bad number {s:?}"))
            };
            let xz = |s: &str| -> Result<[f32; 2], String> {
                let (a, b) = s.split_once(',').ok_or_else(|| format!("expected x,z, got {s:?}"))?;
                Ok([num(a.trim())?, num(b.trim())?])
            };
            let mut consumed = 2;
            match toks[i].as_str() {
                "--to" => p.to.push(xz(need(i)?)?),
                "--from" => p.from = Some(xz(need(i)?)?),
                "--id" => {
                    let s = need(i)?;
                    p.id = Some(
                        u64::from_str_radix(s.trim_start_matches("0x"), 16)
                            .map_err(|_| format!("bad hex id {s:?}"))?,
                    );
                }
                "--slope" => p.slope = num(need(i)?)?,
                "--clearance" => p.clearance = num(need(i)?)?,
                "--ceiling" => p.ceiling = Some(num(need(i)?)?),
                "--snap" => p.snap = need(i)?.parse().map_err(|_| "--snap needs pixels")?,
                "--ssaa" => p.ssaa = need(i)?.parse().map_err(|_| "--ssaa needs a factor")?,
                "--out" => p.out = Some(need(i)?.to_string()),
                "--mask-out" => p.mask_out = Some(need(i)?.to_string()),
                "--size" => {
                    let s = need(i)?;
                    let (a, b) = s.split_once('x').ok_or("--size wants WxH")?;
                    p.size = (
                        a.trim().parse().map_err(|_| "bad width")?,
                        b.trim().parse().map_err(|_| "bad height")?,
                    );
                }
                "--anchor" => {
                    let s = need(i)?;
                    anchor = Some(
                        u64::from_str_radix(s.trim_start_matches("0x"), 16)
                            .map_err(|_| format!("bad hex id {s:?}"))?,
                    );
                }
                "--stable" => {
                    stable = true;
                    consumed = 1;
                }
                "--raw" => {
                    p.frame = FrameRef::Raw;
                    consumed = 1;
                }
                other => return Err(format!("unknown route option {other:?}")),
            }
            i += consumed;
        }
        if let Some(a) = anchor {
            p.frame = FrameRef::Anchor(a);
        } else if stable {
            p.frame = FrameRef::Stable;
        }
        if p.to.is_empty() {
            return Err("at least one --to <x>,<z> is required".into());
        }
        Ok(p)
    }

    fn route_start(&mut self, p: &RouteOpts, origin: [f32; 3]) -> Result<[f32; 2], String> {
        match (p.from, p.id) {
            (Some(f), _) => Ok(f),
            (None, Some(id)) => {
                let (objects, _, _, _) = self.locate_frame(p.frame)?;
                let o = objects
                    .iter()
                    .find(|o| o.id == id)
                    .ok_or_else(|| format!("object {id:#018x} is not in this frame"))?;
                Ok([o.world[0] - origin[0], o.world[2] - origin[2]])
            }
            (None, None) => Err("--from <x>,<z> or --id <hex> is required".into()),
        }
    }

    /// Plan one leg per destination, chaining them. Each leg's failure is REPORTED rather
    /// than skipped - a route that silently omits an unreachable destination looks exactly
    /// like a route that visits it.
    fn plan_legs(
        &self,
        map: &render::WorldMap,
        mask: &render::Traversable,
        start: [f32; 2],
        p: &RouteOpts,
    ) -> Result<(Vec<[f32; 2]>, Vec<String>), String> {
        let mut legs: Vec<[f32; 2]> = Vec::new();
        let mut notes: Vec<String> = Vec::new();
        let mut cur = start;
        for (n, dest) in p.to.iter().enumerate() {
            match render::plan_route(map, mask, cur, *dest, p.snap) {
                Some(leg) => {
                    // The first point of a leg is where we already are.
                    for w in leg.iter().skip(1) {
                        legs.push(*w);
                    }
                    cur = *leg.last().unwrap_or(&cur);
                    notes.push(format!(
                        "leg {n} to ({:.1},{:.1}): {} waypoint(s), ends ({:.1},{:.1})",
                        dest[0], dest[1], leg.len().saturating_sub(1), cur[0], cur[1]
                    ));
                }
                None => {
                    return Err(format!(
                        "leg {n}: no driveable route from ({:.1},{:.1}) to ({:.1},{:.1}). Either \
                         they are genuinely not connected, or the mask is wrong - {:.0}% of the \
                         mapped area is driveable, so try a larger --slope, a smaller \
                         --clearance, or a --ceiling if a roof is being mapped instead of a floor",
                        cur[0], cur[1], dest[0], dest[1], mask.open_fraction() * 100.0
                    ))
                }
            }
        }
        Ok((legs, notes))
    }

    /// `navigate --id <hex> --to <x>,<z> [--to ...] [--anchor <hex>] [--tick N]
    /// [--radius R] [--max-frames N] [--gain G] [--throttle "<tokens>"]
    /// [--reverse "<tokens>"] [--steer <axis>]` - steer a locatable object to a series
    /// of world positions, closing the loop on measured motion.
    ///
    /// # Why this is a command and not a script
    /// Navigating by hand costs one round trip per correction, and a round trip is
    /// minutes: the object drives into a wall long before the second correction lands.
    /// The loop has to run INSIDE the session, next to the thing it is measuring.
    ///
    /// It carries no per-title constant. Which analog axis steers is named by `--steer`,
    /// but WHICH WAY that axis turns is MEASURED from how the heading responded to the
    /// offset applied last tick. Heading itself is taken from measured velocity whenever
    /// the object is moving, because a velocity bearing needs no convention at all - the
    /// mesh's own axes only stand in when it is too slow to have a direction. Both readings
    /// come from `locate`'s drift-compensated motion, so a travelling coordinate origin
    /// cannot masquerade as progress.
    ///
    /// With `--plan`, the `--to` list is treated as DESTINATIONS and the waypoints between
    /// them are computed from the frame's height field (see `cmd_route`) instead of driven
    /// at directly - which is the difference between going around a railing and finding it.
    fn cmd_navigate(&mut self, args: &str) -> Result<String, String> {
        let mut id: Option<u64> = None;
        let mut anchor: Option<u64> = None;
        let mut route: Vec<[f32; 2]> = Vec::new();
        let mut plan = false;
        let mut plan_args: Vec<String> = Vec::new();
        let mut tick = 10u64;
        let mut radius = 12.0f32;
        let mut max_frames = 1200u64;
        let mut gain = 2.0f32;
        let mut throttle = "ry=0".to_string();
        let mut reverse = "ry=255".to_string();
        let mut steer_axis = "lx".to_string();
        let mut reverse_frames = 40u64;
        let mut brake_above = 75.0f32;

        let toks: Vec<String> = split_quoted(args);
        let mut i = 0;
        while i < toks.len() {
            let need = |i: usize| -> Result<&str, String> {
                toks.get(i + 1).map(|s| s.as_str()).ok_or_else(|| format!("{} needs a value", toks[i]))
            };
            let hex = |s: &str| -> Result<u64, String> {
                u64::from_str_radix(s.trim_start_matches("0x"), 16)
                    .map_err(|_| format!("bad hex id {s:?}"))
            };
            let num = |s: &str| -> Result<f32, String> {
                s.parse().map_err(|_| format!("bad number {s:?}"))
            };
            match toks[i].as_str() {
                "--id" => id = Some(hex(need(i)?)?),
                "--anchor" => anchor = Some(hex(need(i)?)?),
                "--to" => {
                    let s = need(i)?;
                    let (a, b) = s.split_once(',').ok_or("--to wants x,z")?;
                    route.push([num(a.trim())?, num(b.trim())?]);
                }
                "--tick" => tick = need(i)?.parse().map_err(|_| "--tick needs frames")?,
                "--radius" => radius = num(need(i)?)?,
                "--max-frames" => {
                    max_frames = need(i)?.parse().map_err(|_| "--max-frames needs frames")?
                }
                "--reverse-frames" => {
                    reverse_frames = need(i)?.parse().map_err(|_| "--reverse-frames needs frames")?
                }
                "--gain" => gain = num(need(i)?)?,
                "--brake-above" => brake_above = num(need(i)?)?,
                // Route-planning options, forwarded verbatim so `route` and
                // `navigate --plan` cannot disagree about what is driveable.
                "--plan" => {
                    plan = true;
                    i += 1;
                    continue;
                }
                "--slope" | "--clearance" | "--ceiling" | "--snap" | "--size" | "--ssaa" => {
                    plan_args.push(toks[i].clone());
                    plan_args.push(need(i)?.to_string());
                }
                "--throttle" => throttle = need(i)?.to_string(),
                "--reverse" => reverse = need(i)?.to_string(),
                "--steer" => steer_axis = need(i)?.to_string(),
                other => return Err(format!("unknown navigate option {other:?}")),
            }
            i += 2;
        }
        let id = id.ok_or("--id <hex> is required (find it with `locate --moving`)")?;
        if route.is_empty() {
            return Err("at least one --to <x>,<z> is required".into());
        }
        if tick == 0 {
            return Err("--tick must be at least 1 frame".into());
        }

        // With --plan, the destinations become a computed obstacle-aware route. Routed
        // through the same option parser and planner `route` uses, so the two can never
        // disagree about what counts as driveable.
        let mut plan_note = String::new();
        if plan {
            let mut spec = plan_args.join(" ");
            for d in &route {
                spec.push_str(&format!(" --to {},{}", d[0], d[1]));
            }
            spec.push_str(&format!(" --id {id:#x}"));
            match anchor {
                Some(a) => spec.push_str(&format!(" --anchor {a:#x}")),
                None => spec.push_str(" --stable"),
            }
            let p = self.parse_route_opts(&spec)?;
            let (map, mask, origin) =
                self.navmesh(p.frame, p.size, p.ceiling, p.slope, p.clearance, p.ssaa)?;
            let start = self.route_start(&p, origin)?;
            let (legs, notes) = self.plan_legs(&map, &mask, start, &p)?;
            plan_note = format!(
                "planned {} waypoint(s) over {} destination(s) from the frame's height field \
                 ({:.0}% driveable)\n{}\n",
                legs.len(),
                p.to.len(),
                mask.open_fraction() * 100.0,
                notes.join("\n")
            );
            route = legs;
            if route.is_empty() {
                return Err("the planner returned no waypoints - already at every \
                            destination?"
                    .into());
            }
        }

        // One reading of where the target object is and how it is moving, in the anchored
        // frame. Every failure here is fatal rather than skipped: navigating on a stale
        // position is worse than not navigating.
        struct Fix {
            pos: [f32; 3],
            motion: [f32; 3],
            speed: f32,
            heading: Option<[f32; 2]>,
        }
        // The STABLE frame by default, not an anchor object: a navigating vehicle drives
        // away from whatever mesh was chosen as an anchor, and the moment that mesh leaves
        // the view an anchored run can only stop. See `locate_frame`.
        let want = match anchor {
            Some(a) => FrameRef::Anchor(a),
            None => FrameRef::Stable,
        };
        let sample = |s: &mut Self, frames: u64| -> Result<Fix, String> {
            let (objects, origin, drift, previous) = s.locate_frame(want)?;
            let hits: Vec<&render::ObjectLoc> = objects.iter().filter(|o| o.id == id).collect();
            let o = match hits.len() {
                1 => hits[0],
                0 => {
                    return Err(format!(
                        "object {id:#018x} is not among the {} objects at f{} - it has left the \
                         view or its mesh changed",
                        objects.len(),
                        s.frame()
                    ))
                }
                n => return Err(format!("object {id:#018x} appears {n} times - not a unique id")),
            };
            let shift = drift.map(|d| d.delta).unwrap_or([0.0; 3]);
            let motion = previous
                .as_ref()
                .and_then(|p| render::world_motion(p, o, shift))
                .map(|(d, _)| d)
                .unwrap_or([0.0; 3]);
            let dist = (motion[0] * motion[0] + motion[2] * motion[2]).sqrt();
            Ok(Fix {
                pos: [o.world[0] - origin[0], o.world[1] - origin[1], o.world[2] - origin[2]],
                motion,
                speed: dist / frames.max(1) as f32,
                heading: o.heading,
            })
        };

        // Bearings in the same convention as `locate`'s heading and the pad's polar stick
        // directive: 0 along world +X, increasing toward world -Z.
        let bearing = |dx: f32, dz: f32| -> f32 { (-dz).atan2(dx).to_degrees() };
        let wrap = |mut a: f32| -> f32 {
            while a > 180.0 {
                a -= 360.0;
            }
            while a < -180.0 {
                a += 360.0;
            }
            a
        };

        let start_frame = self.frame();
        let mut log: Vec<String> = Vec::new();
        let mut sign = if gain < 0.0 { -1.0f32 } else { 1.0 };
        let gain = gain.abs();
        let mut flips = 0u32;
        // Sign calibration state: the offset applied last tick and the heading it was
        // applied from, plus the accumulated votes on which way the axis turns.
        let mut last_steer_off = 0.0f32;
        let mut last_heading: Option<f32> = None;
        let mut sign_votes = 0.0f32;
        let mut stalled = 0u32;
        let mut wp = 0usize;
        let mut reached: Vec<usize> = Vec::new();
        // A calibrated forward axis: which of the mesh's two in-plane axes points the way
        // it travels, learned the first time it is moving fast enough for velocity to be
        // an unambiguous heading.
        let mut fwd_axis: Option<usize> = None;

        self.recipe.sections.push(Section { frame: start_frame, name: "navigate".into() });
        // Prime the motion baseline: the first reading has nothing to difference against.
        let _ = sample(self, tick)?;

        // A tick that cannot be measured ABORTS the run - navigating on a stale position is
        // worse than stopping - but the log gathered so far is the finding and is reported
        // either way. Discarding it on the way out (which the first version of this did)
        // throws away every reading that led up to the failure.
        let mut failure: Option<String> = None;
        macro_rules! tick_try {
            ($e:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(e) => {
                        failure = Some(e);
                        break "ABORTED".to_string();
                    }
                }
            };
        }
        let outcome = loop {
            if self.finished() {
                break format!("guest stopped ({:?})", self.last);
            }
            if self.frame() - start_frame >= max_frames {
                break format!("--max-frames {max_frames} reached");
            }
            let fix = tick_try!(sample(self, tick));
            let target = route[wp];
            let (dx, dz) = (target[0] - fix.pos[0], target[1] - fix.pos[2]);
            let dist = (dx * dx + dz * dz).sqrt();
            if dist <= radius {
                reached.push(wp);
                log.push(format!(
                    "f{:<6} REACHED waypoint {wp} ({:.1},{:.1}) at dist {dist:.1}",
                    self.frame(), target[0], target[1]
                ));
                self.recipe.notes.push(NoteDecl {
                    frame: self.frame(),
                    todo: false,
                    text: format!("reached waypoint {wp} ({:.1},{:.1})", target[0], target[1]),
                });
                wp += 1;
                if wp >= route.len() {
                    break "every waypoint reached".to_string();
                }
                continue;
            }

            // Heading: velocity when there is any, the mesh's calibrated forward axis
            // otherwise. `MIN_SPEED` is per frame, so it is a real motion threshold rather
            // than a tick-length artefact.
            const MIN_SPEED: f32 = 0.15;
            let moving = fix.speed > MIN_SPEED;
            let vel_bearing = bearing(fix.motion[0], fix.motion[2]);
            if moving {
                if let (None, Some(h)) = (fwd_axis, fix.heading) {
                    // Whichever mesh axis agrees with the direction of travel is forward.
                    let d0 = wrap(h[0] - vel_bearing).abs();
                    let d1 = wrap(h[1] - vel_bearing).abs();
                    fwd_axis = Some(if d0 <= d1 { 0 } else { 1 });
                }
            }
            let heading = match (moving, fwd_axis, fix.heading) {
                (true, _, _) => Some(vel_bearing),
                (false, Some(a), Some(h)) => Some(h[a]),
                _ => None,
            };
            let want = bearing(dx, dz);

            // Wedged: throttle is on and nothing is happening. Back out rather than keep
            // pressing into whatever is in the way.
            if !moving {
                stalled += 1;
                if stalled >= 2 {
                    let steer = if sign > 0.0 { 40 } else { 216 };
                    let spec = format!("{reverse} {steer_axis}={steer}");
                    let (input, touch) = tick_try!(Recipe::parse_input(&spec));
                    let f = self.frame();
                    self.push_input(f, input, touch);
                    tick_try!(self.advance(reverse_frames, false));
                    log.push(format!(
                        "f{:<6} WEDGED (speed {:.3}) - reversed {reverse_frames} frames as {spec:?}",
                        f, fix.speed
                    ));
                    stalled = 0;
                    // Re-baseline: the reverse moved things without a reading in between.
                    let _ = tick_try!(sample(self, reverse_frames));
                    continue;
                }
            } else {
                stalled = 0;
            }

            let err = heading.map(|h| wrap(want - h));

            // Sign discovery, by MEASUREMENT rather than inference: how did the heading
            // respond to the offset actually applied last tick? An earlier version instead
            // inverted the sign whenever the error grew while steering, and that is not
            // evidence - a proportional controller chasing a waypoint it is about to pass
            // sees the bearing to it swing wildly, so the heuristic flip-flopped and burned
            // through its own flip budget on a run whose sign was right all along.
            //
            // Votes accumulate, so one noisy tick cannot decide it and a genuinely inverted
            // axis is settled within two.
            if let (Some(h), Some(ph)) = (heading, last_heading) {
                let dh = wrap(h - ph);
                if moving && last_steer_off.abs() > 20.0 && dh.abs() > 4.0 {
                    let before = if sign_votes == 0.0 { 0.0 } else { sign_votes.signum() };
                    sign_votes += (dh * last_steer_off).signum();
                    let after = if sign_votes == 0.0 { 0.0 } else { sign_votes.signum() };
                    if after != 0.0 && after != before {
                        let new_sign = after;
                        if new_sign != sign {
                            log.push(format!(
                                "f{:<6} {steer_axis} MEASURED to turn the other way ({:+.0} deg of \
                                 heading for an offset of {:+.0}) - steering sign flipped",
                                self.frame(), dh, last_steer_off
                            ));
                            flips += 1;
                        }
                        sign = new_sign;
                    }
                }
            }
            last_heading = heading;

            let steer = match err {
                Some(e) => (128.0 + sign * gain * e).round().clamp(1.0, 255.0) as i32,
                None => 128,
            };
            last_steer_off = steer as f32 - 128.0;
            // Too far off course to drive out of: an analog steer is a TURN RATE, so at
            // speed a large error just means a wide arc past the target. Releasing the
            // throttle lets the turn tighten instead. Purely a rate argument, so it holds
            // for any title whose steering works this way.
            let drive = if err.is_some_and(|e| e.abs() > brake_above) {
                String::new()
            } else {
                throttle.clone()
            };
            let spec = format!("{drive} {steer_axis}={steer}");
            let (input, touch) = tick_try!(Recipe::parse_input(&spec));
            let f = self.frame();
            self.push_input(f, input, touch);
            log.push(format!(
                "f{:<6} wp{wp} at ({:.1},{:.1}) dist={dist:6.1} want={want:7.1} head={} err={} \
                 speed={:.2} -> {steer_axis}={steer}",
                f, fix.pos[0], fix.pos[2],
                heading.map(|h| format!("{h:7.1}")).unwrap_or("      ?".into()),
                err.map(|e| format!("{e:+7.1}")).unwrap_or("      ?".into()),
                fix.speed,
            ));
            tick_try!(self.advance(tick, false));
        };

        // Release the controls: leaving the throttle held would carry into whatever the
        // caller does next, which is how a "settled" reading ends up measured mid-drive.
        let f = self.frame();
        let (input, touch) = Recipe::parse_input("")?;
        self.push_input(f, input, touch);

        let final_fix = sample(self, tick).ok();
        let mut report = format!(
            "{plan_note}navigate: {outcome}\nwaypoints reached {}/{} over {} frames \
             (f{start_frame}..f{})\n\
             steering: {steer_axis}, sign {}{} (votes {sign_votes:+.0}, {flips} flip(s) during \
             the run)\n",
            reached.len(),
            route.len(),
            self.frame() - start_frame,
            self.frame(),
            if sign > 0.0 { "+" } else { "-" },
            if sign_votes == 0.0 { " UNMEASURED - the run never turned hard enough to tell" } else { "" },
        );
        if let Some(fx) = final_fix {
            report.push_str(&format!(
                "final position ({:.2},{:.2},{:.2}) speed {:.2}/frame\n",
                fx.pos[0], fx.pos[1], fx.pos[2], fx.speed
            ));
        }
        report.push_str(&log.join("\n"));
        // Loud on the way out, but never empty-handed: the ticks that led to the failure
        // are in the report either way.
        match failure {
            None => Ok(report),
            Some(e) => Err(format!("navigate ABORTED: {e}\n{report}")),
        }
    }

    fn cmd_info(&self) -> String {
        let (base, len) = self.sched.guest_region();
        let scan = match &self.scan {
            Some(s) => format!(
                "{} {} slots alive over {:#010x}+{:#x} after {} passes",
                s.alive_count,
                s.ty.keyword(),
                s.addr,
                s.len,
                s.passes
            ),
            None => "none".into(),
        };
        let (input, touch) = self.timeline.lock().unwrap().at(self.frame());
        // The guest clock beside the frame count, with the RATE it has been running at.
        // A title derives its own timers from this clock, so if it advances faster than
        // the frames are worth then everything the game times - a race clock, a countdown,
        // a timed event - runs fast, and the failure shows up as a game rule (a race that
        // always runs out of time) rather than as anything that looks like a clock bug.
        // A run TOTAL can average out to 1.00x while a stretch inside it runs at 5x, which
        // is why this is reported live rather than only at the end of a run.
        let clock_us = self.sched.host().state.now_us();
        let worth_us = self.frame() * 1_000_000 / 60;
        let rate = if worth_us > 0 { clock_us as f64 / worth_us as f64 } else { 0.0 };
        format!(
            "frame      {}\nrun        {:?}\nguest mem  {base:#010x}+{len:#x}\nheld input buttons={:#06x} lx={} ly={} rx={} ry={} touch={}\nwatches    {}\nscan       {scan}\nshots      {}\nclock      {:.3}s = {rate:.2}x the {:.1}s these frames are worth\nsig        {:#018x}",
            self.frame(),
            self.last,
            input.buttons,
            input.lx,
            input.ly,
            input.rx,
            input.ry,
            touch.count,
            if self.recipe.watches.is_empty() {
                "none".to_string()
            } else {
                self.watch_values()
            },
            self.opts.shot_dir.as_ref().map(|p| p.display().to_string()).unwrap_or("(none)".into()),
            clock_us as f64 / 1e6,
            worth_us as f64 / 1e6,
            self.signature(),
        )
    }

    /// `save <file>` - write the session out as a recipe that replays it exactly.
    ///
    /// This is the command that makes a session worth running: whatever sequence of
    /// presses got the title through a menu or around a lap becomes a committed,
    /// re-runnable artifact instead of a thing that happened once.
    fn cmd_save(&mut self, args: &str) -> Result<String, String> {
        let path = self.resolve_out(args.trim(), "session.recipe");
        let segments = self.timeline.lock().unwrap().segments().to_vec();
        self.recipe.set_segments(segments);
        // Pin the signature the run actually produced, so a replay that diverges
        // fails loudly instead of silently becoming a different playthrough.
        self.recipe.meta.sig = Some(self.signature());
        let text = self.recipe.to_text();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(&path, &text).map_err(|e| format!("write {}: {e}", path.display()))?;
        Ok(format!(
            "saved {} segments through f{} to {}",
            self.recipe.segment_count(),
            self.frame(),
            path.display()
        ))
    }

    /// Resolve an output path: a given path wins, otherwise a default name inside
    /// the shots dir. Never writes into the current directory by accident.
    fn resolve_out(&self, arg: &str, default_name: &str) -> PathBuf {
        if !arg.is_empty() {
            return PathBuf::from(arg);
        }
        match &self.opts.shot_dir {
            Some(d) => d.join(default_name),
            None => PathBuf::from(default_name),
        }
    }
}

/// Split a command's arguments on whitespace, but keep a quoted run together.
///
/// Needed because some options take an INPUT SPEC, which is itself several
/// whitespace-separated tokens (`--throttle "ry=0 l"`). Plain whitespace splitting turns
/// that into an unknown option, and quietly dropping the tail would leave a throttle that
/// half applies. An unterminated quote is an error rather than a silent run to end-of-line.
fn split_quoted(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut any = false;
    for c in s.chars() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => cur.push(c),
            (None, '"') | (None, '\'') => {
                quote = Some(c);
                any = true;
            }
            (None, c) if c.is_whitespace() => {
                if !cur.is_empty() || any {
                    out.push(std::mem::take(&mut cur));
                    any = false;
                }
            }
            (None, c) => cur.push(c),
        }
    }
    if !cur.is_empty() || any {
        out.push(cur);
    }
    out
}

/// Parse a guest address or byte count, hex (`0x...`) or decimal.
fn parse_addr(s: &str) -> Result<u32, String> {
    let s = s.trim();
    let r = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u32::from_str_radix(hex, 16),
        None => s.parse::<u32>(),
    };
    r.map_err(|_| format!("bad address/size {s:?}"))
}

/// Parse a scan operand (decimal, possibly float).
fn parse_num(s: &str) -> Result<f64, String> {
    s.trim().parse().map_err(|_| format!("bad number {:?}", s.trim()))
}

/// A directory-per-session control channel: the way an agent talks to a resident
/// session across separate shell invocations.
///
/// Each shell command an agent runs is its own process, so a session that read
/// commands from its own stdin would be unreachable after the shell that started it
/// returned. Instead both ends share a directory: the client APPENDS a numbered
/// request line to `control` and blocks until the matching reply appears in `log`.
/// Plain append-only files, no OS-specific IPC, and the log doubles as a complete
/// transcript of the playthrough.
pub struct ControlDir {
    root: PathBuf,
}

/// A request read out of the control file.
pub struct Request {
    pub seq: u64,
    pub line: String,
}

impl ControlDir {
    pub fn new(root: impl Into<PathBuf>) -> std::io::Result<ControlDir> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(ControlDir { root })
    }

    pub fn control_path(&self) -> PathBuf {
        self.root.join("control")
    }
    pub fn log_path(&self) -> PathBuf {
        self.root.join("log")
    }
    /// Written when the session is ready, removed when it exits: the client's proof
    /// that someone is listening rather than that it is about to block forever.
    pub fn ready_path(&self) -> PathBuf {
        self.root.join("ready")
    }

    /// Every request appended past `consumed`, and the new consumed count.
    pub fn poll(&self, consumed: u64) -> std::io::Result<(Vec<Request>, u64)> {
        let text = match std::fs::read_to_string(self.control_path()) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), consumed)),
            Err(e) => return Err(e),
        };
        let mut out = Vec::new();
        let mut n = 0u64;
        for line in text.lines() {
            n += 1;
            if n > consumed && !line.trim().is_empty() {
                out.push(Request { seq: n, line: line.to_string() });
            }
        }
        Ok((out, n))
    }

    /// Append a reply block for `seq`. The framing (`<<< n ok|err` ... `>>> n`) is
    /// what lets the client find its own reply in a file several sessions may be
    /// appending to, and tell a finished reply from a half-written one.
    pub fn reply(&self, seq: u64, ok: bool, body: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.log_path())?;
        writeln!(f, "<<< {seq} {}", if ok { "ok" } else { "err" })?;
        for line in body.lines() {
            writeln!(f, "{line}")?;
        }
        writeln!(f, ">>> {seq}")
    }

    /// Append a request line, returning its sequence number.
    pub fn request(&self, line: &str) -> std::io::Result<u64> {
        use std::io::Write;
        let mut f =
            std::fs::OpenOptions::new().create(true).append(true).open(self.control_path())?;
        writeln!(f, "{line}")?;
        let text = std::fs::read_to_string(self.control_path())?;
        Ok(text.lines().count() as u64)
    }

    /// The complete reply for `seq`, if it has been written: `(ok, body)`.
    pub fn read_reply(&self, seq: u64) -> std::io::Result<Option<(bool, String)>> {
        let text = match std::fs::read_to_string(self.log_path()) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        let open = format!("<<< {seq} ");
        let close = format!(">>> {seq}");
        let mut body = String::new();
        let mut ok = None;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix(&open) {
                ok = Some(rest.trim() == "ok");
                body.clear();
                continue;
            }
            if ok.is_some() {
                if line == close {
                    return Ok(Some((ok.unwrap(), body)));
                }
                body.push_str(line);
                body.push('\n');
            }
        }
        Ok(None)
    }
}

/// The command reference, printed by `help`. Kept next to the dispatcher so a new
/// command cannot be added without a line here.
pub const HELP: &str = "\
RUNNING
  step [N] [--sample]        advance N display frames (default 1)
  until <w> <op> <v> [--max N] [--tol T]
                             step until a watch satisfies a condition
  info                       everything about the current state
  frames                     the current frame
INPUT (recipe grammar: cross circle square triangle up down left right start
       select l r, lx=/ly=/rx=/ry= 0..255, touch=X,Y in panel coords = screen*2)
  input <tokens...>          set the sticky input from now on (empty = release)
  press <tokens...> [--hold N]
                             hold for N frames (default 8) then release
MEMORY
  read <addr> <type>         one value (u8|u16|u32|i32|f32)
  dump [<addr> <len>] <file> write a region out, to diff against another run
                             (see `memdiff` - the sharpest way to ask what an
                             input actually touched)
  peek <addr> [len]          hex + i32 + f32 dump
  poke <addr> <type> <value> overwrite a value (to prove an address matters)
  watch <name> <type> <addr> sample every frame into the watch log
  unwatch <name> | watches | watchlog [file]
VALUE FINDER (find the address of a live value with no symbols)
  scan new <type> [addr len] baseline a region (default: all of guest memory)
  scan changed | unchanged | increased | decreased
  scan eq <v> [+-tol] | ne <v> | gt <v> | lt <v> | range <lo> <hi>
  scan list [n] | scan promote <name> <addr> | scan reset
OUTPUT
  shot [name]                render the current frame to a PNG
  shot-every <N>             auto-shot every N stepped frames (0 disables)
  section <name> | note <text> | todo <text>
  scene [--passes]           draw count + a digest of THIS frame's draws, so you
                             can tell a world that is simulating from one that is
                             only redrawing. --passes lists every pass of the frame
                             with its render target, its world-triangle count and
                             the vertex attribute FORMATS its draws declare
  locate [--min-tris N] [--top N] [--moving] [--id <hex>] [--stable|--anchor <hex>]
                             every object in this frame: its world position, its
                             screen box, and how far it moved since the last
                             `locate`. This is how you navigate without eyeballing
                             a PNG - and how you find which object is the player
                             (step, `locate --moving`, see what responds). `id` is
                             the object's geometry, so it is stable frame to frame;
                             `--id` follows just that one.
                             EVERY report states the coordinate ORIGIN's own drift
                             since the last one, because a title may measure its
                             world matrices from a frame that travels with the
                             camera - and then bolted-down scenery appears to move
                             while the player appears not to. `moved` always has
                             that drift removed. `--stable` reports positions in
                             the frame the session's first `locate` used (survives
                             anything leaving the view); `--anchor <id>` measures
                             from one object instead (exact, but fails loudly the
                             moment that object is out of frame).
  sprites [--moving] [--top N] [--min-tris N] [--id <hex>] [--textured]
                             the 2D counterpart of `locate`: every 2D drawn thing,
                             where it is on screen, and how it moved. Needed
                             because a 2D title has no model-to-world matrix, so
                             `locate` reports nothing at all for one. A sprite's id
                             is its texture + the atlas region it samples + its
                             size, since its POSITION is its vertex data. `--moving`
                             has the scene's scroll removed, so a camera pan does
                             not read as the whole backdrop moving.
  map [--extent x0,z0,x1,z1] [--size WxH] [--grid CxR] [--ssaa N] [--step F]
      [--ceiling Y] [--stable|--anchor <hex>] [--mark <hex-id>|<x>,<z>] [--out <name>]
                             a top-down orthographic map of this frame's world
                             geometry: a PNG plus an ASCII height field, with the
                             world-to-pixel scale printed. Use it when the thing
                             you must navigate to is PAINTED rather than placed (a
                             trail in the ground texture has no world matrix, so
                             `locate` cannot see it at all), and to find the
                             railings a guessed route would catch on. Auto extent
                             covers 98% of vertices, so a skydome cannot squash
                             the playable area into four pixels.
  camera                     where the view is and which way it looks, recovered
                             from the frame's own world-to-clip matrix. Use this
                             rather than a position address found by memory
                             diffing: a per-frame matrix pool moves the vehicle
                             between slots, and a stale slot reads exactly like a
                             car stopped against a wall.
  surface --at <x>,<z> [--ceiling Y]
  surface --tex <hex> --from <x>,<z> --bearing <deg> [--fov F] [--range R] [--top N]
                             which MATERIAL is under a world point, and how far that
                             same material reaches ahead of you. Height cannot tell a
                             road from the grass beside it (same height); the texture
                             can. Ask --at where the vehicle is certainly legal, then
                             --tex that answer to get an aim point every tick.
  route --to <x>,<z> [--to ...] [--from <x>,<z>|--id <hex>] [--slope R]
        [--clearance W] [--size WxH] [--ceiling Y] [--snap N] [--out <name>]
                             a DRIVEABLE route to each destination, computed from
                             this frame's height field: A* over the traversable
                             mask, simplified to the turns it actually makes, and
                             printed as a ready-to-paste `navigate --to` list.
                             Waypoints chosen by eye off a map only encode what the
                             eye noticed - the railing between two of them is found
                             by driving into it. `--slope` is rise over run and
                             `--clearance` a body width, both in world units, so a
                             route does not change meaning with map resolution.
  navigate --id <hex> --to <x>,<z> [--to ...] [--plan] [--anchor <hex>] [--tick N]
           [--radius R] [--gain G] [--steer lx] [--throttle \"ry=0\"]
           [--reverse \"ry=255\"] [--max-frames N]
                             steer a locatable object along a route, closing the
                             loop on MEASURED motion: it re-locates every --tick
                             frames, aims at the next waypoint, backs out when it
                             wedges, and prints a per-tick table. Which way the
                             steer axis turns is discovered from the response, not
                             configured. Driving by hand instead costs one round
                             trip per correction, and the object hits a wall first.
                             With --plan the --to list is DESTINATIONS and the way
                             between them is routed around obstacles (see `route`).
  threads                    every sync primitive's owner and waiters - who is
                             blocked, and on what
  stdout [--tail N] [--grep <text>] | stderr [...]
                             what the GUEST printed to fd 1/2. Retail titles log a
                             surprising amount, and it is the developer's own
                             account of what the game thinks is wrong - read it
                             before reverse-engineering the answer.
  sig | egress
  save [file.recipe]         write the played run out as a replayable recipe
  quit";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_dir_round_trips_a_request_and_reply() {
        let dir = std::env::temp_dir().join(format!("vitaslop-ctl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let c = ControlDir::new(&dir).unwrap();
        let seq = c.request("step 10").unwrap();
        assert_eq!(seq, 1);
        assert_eq!(c.read_reply(seq).unwrap(), None, "no reply before the session answers");

        let (reqs, consumed) = c.poll(0).unwrap();
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].line, "step 10");
        assert_eq!(consumed, 1);
        c.reply(seq, true, "frame=10 run=FramesReached(10)").unwrap();
        assert_eq!(
            c.read_reply(seq).unwrap(),
            Some((true, "frame=10 run=FramesReached(10)\n".to_string()))
        );

        // A second request is a fresh sequence number and does not disturb the first.
        let seq2 = c.request("shot here").unwrap();
        assert_eq!(seq2, 2);
        let (reqs, _) = c.poll(consumed).unwrap();
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].seq, 2);
        c.reply(seq2, false, "no shot written").unwrap();
        assert_eq!(c.read_reply(seq2).unwrap().unwrap().0, false);
        assert!(c.read_reply(seq).unwrap().unwrap().0, "the first reply is still intact");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A multi-line reply must survive framing intact - the value finder's candidate
    /// list is the reply an agent most needs to read verbatim.
    #[test]
    fn control_dir_preserves_multiline_replies() {
        let dir = std::env::temp_dir().join(format!("vitaslop-ctl-ml-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let c = ControlDir::new(&dir).unwrap();
        let seq = c.request("scan list").unwrap();
        c.reply(seq, true, "0x81000000 = 1\n0x81000004 = 2\n0x81000008 = 3").unwrap();
        let (ok, body) = c.read_reply(seq).unwrap().unwrap();
        assert!(ok);
        assert_eq!(body.lines().count(), 3);
        assert_eq!(body.lines().last(), Some("0x81000008 = 3"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn addresses_parse_in_hex_and_decimal() {
        assert_eq!(parse_addr("0x81000000").unwrap(), 0x8100_0000);
        assert_eq!(parse_addr("4096").unwrap(), 4096);
        assert!(parse_addr("zz").is_err());
    }
}
