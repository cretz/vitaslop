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
    /// ONE, because that is all anything reads: a screenshot renders the latest
    /// scene, `scene` digests the latest scene, and the determinism signature folds
    /// each scene as it is evicted. Holding more is pure waste, and it is expensive
    /// waste - a scene carries a snapshot of every draw's vertex window, several
    /// megabytes a frame on a real 3D title.
    pub scene_limit: Option<usize>,
}

impl Default for SessionOpts {
    fn default() -> Self {
        SessionOpts {
            quantum_fuel: 5_000_000,
            max_rounds: 400_000_000,
            per_frame_rounds: 4_000_000,
            shot_dir: None,
            scene_limit: Some(1),
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
            "scene" => Ok(self.cmd_scene()),
            "locate" => Ok(self.cmd_locate(args)),
            "sig" => Ok(format!("sig={:#018x}", self.signature())),
            "egress" => Ok(self.cmd_egress()),
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
    fn cmd_scene(&self) -> String {
        let host = self.sched.host();
        let cap = &host.state.capture;
        let Some(scene) = cap.scenes.last() else { return "no scene captured yet".into() };
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
                _ => {}
            }
            i += 1;
        }
        let objects = {
            let host = self.sched.host();
            let Some(scene) = host.state.capture.scenes.last() else {
                return "no scene captured yet".into();
            };
            render::locate_scene(scene, observe::WIDTH, observe::HEIGHT)
        };
        // Movement is measured against the previous `locate`, so a caller steps,
        // locates, steps, locates and reads the delta straight out of the report.
        let previous = std::mem::replace(&mut self.last_locate, Some(objects.clone()));
        let mut lines = Vec::new();
        let mut shown = 0usize;
        for o in &objects {
            if o.triangles < min_tris || shown >= top {
                continue;
            }
            if only_id.is_some_and(|want| want != o.id) {
                continue;
            }
            // Match to the previous frame by GEOMETRY id, never by draw index - the
            // draw list is rebuilt every frame. Several objects can legitimately share
            // one mesh (a row of identical cones), so among same-id candidates take the
            // nearest previous position: over one step nothing outruns its own spacing.
            let delta = previous.as_ref().and_then(|prev| {
                prev.iter()
                    .filter(|p| p.id == o.id)
                    .map(|p| {
                        let d = [
                            o.world[0] - p.world[0],
                            o.world[1] - p.world[1],
                            o.world[2] - p.world[2],
                        ];
                        (d, (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            });
            if moving && delta.map(|(_, m)| m).unwrap_or(0.0) <= 1e-4 {
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
                o.world[0],
                o.world[1],
                o.world[2],
                o.triangles,
                if o.sprites { " sprites" } else { "" },
                moved,
            ));
        }
        if lines.is_empty() {
            return format!(
                "frame={} objects={} - nothing matched the filter",
                self.frame(),
                objects.len()
            );
        }
        format!(
            "frame={} objects={} shown={}\n{}",
            self.frame(),
            objects.len(),
            lines.len(),
            lines.join("\n")
        )
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
        format!(
            "frame      {}\nrun        {:?}\nguest mem  {base:#010x}+{len:#x}\nheld input buttons={:#06x} lx={} ly={} rx={} ry={} touch={}\nwatches    {}\nscan       {scan}\nshots      {}\nsig        {:#018x}",
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
  scene                      draw count + a digest of THIS frame's draws, so you
                             can tell a world that is simulating from one that is
                             only redrawing
  locate [--min-tris N] [--top N] [--moving] [--id <hex>]
                             every object in this frame: its world position, its
                             screen box, and how far it moved since the last
                             `locate`. This is how you navigate without eyeballing
                             a PNG - and how you find which object is the player
                             (step, `locate --moving`, see what responds). `id` is
                             the object's geometry, so it is stable frame to frame;
                             `--id` follows just that one.
  threads                    every sync primitive's owner and waiters - who is
                             blocked, and on what
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
