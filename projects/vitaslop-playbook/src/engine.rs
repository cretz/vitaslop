//! The engine: runs one playbook (or replays one tape) against a running title.
//!
//! # The tick
//! Everything happens on INPUT FRAMES: the first pad read the game makes after each display
//! flip. A frame the game does not read the pad on (a load, a movie) is a lag frame and does
//! not count - which is what makes "hold for 20 frames" mean 20 frames of game logic at any
//! speed. At each tick the engine samples every memory cue, reads every call cue's count,
//! advances the blocking request (a hold, an idle, a wait), resumes the playbook if that
//! request finished, and latches the input every pad read in this display frame returns.
//!
//! # Where the playbook runs
//! Natively the VM is held here and resumed INSIDE the tick, so a request that finishes on
//! input frame N has its successor's input on frame N - exact in every mode. The browser's VM
//! lives in JavaScript on the run worker, which a guest worker's tick cannot call; there the
//! run worker pumps it ([`pump_external`]) once per present, and a request's successor lands
//! on the first input frame after that pump. See the crate README.
//!
//! # Threads
//! The tick runs on whichever guest thread reads the pad, under [`ENGINE`]'s lock; the host's
//! loop calls [`on_display_frame`] from its own. Call counting ([`crate::calls`]) is lock-free.
use std::sync::Mutex;

use vitaslop_runtime::observer::{self, HostCallObserver, ObservedCall};
use vitaslop_runtime::guest_read::GuestRead;
use vitaslop_runtime::{CtrlFrame, TouchFrame};

use crate::calls;
use crate::cue::{Armed, CueDef, CueId, CueState};
use crate::proto::{self, Request};
use crate::tape::{self, Change, Note, Tape};
use crate::vm::{self, LuaVm};

/// The pad reads that define an input frame.
const PAD_READS: &[u32] = &[
    vitaslop_runtime::nid::ctrl::PEEK_BUFFER_POSITIVE,
    vitaslop_runtime::nid::ctrl::PEEK_BUFFER_POSITIVE2,
    vitaslop_runtime::nid::ctrl::READ_BUFFER_POSITIVE,
    vitaslop_runtime::nid::ctrl::PEEK_BUFFER_NEGATIVE,
    vitaslop_runtime::nid::ctrl::READ_BUFFER_NEGATIVE,
];

/// A playbook that makes this many requests in one resume without ever waiting on the game
/// is in a loop that cannot end; the run fails rather than hang the tick.
const MAX_REQUESTS_PER_PUMP: usize = 100_000;

/// See [`Engine::add_cue`].
const MAX_CUES: usize = 4096;

/// What the host does with a run's output.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// A log line, already prefixed with where it happened.
    Log(String),
    /// Write a screenshot of the next rendered frame under this name.
    Shot(String),
}

/// How a run ended (or that it has not).
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Running,
    /// The playbook returned, or the tape ran out.
    Done,
    Failed(String),
}

/// One finished phase, for the report: where it ran and how fast.
#[derive(Clone, Debug)]
pub struct PhaseReport {
    pub name: String,
    pub depth: usize,
    pub input: (u64, u64),
    pub display: (u64, u64),
    pub guest_s: f64,
    pub wall_s: Option<f64>,
}

/// Everything a finished run hands back.
#[derive(Clone, Debug)]
pub struct Report {
    pub name: String,
    pub status: Status,
    pub tape: Tape,
    pub phases: Vec<PhaseReport>,
    pub warnings: Vec<String>,
    pub discovery: Option<String>,
    pub input_frames: u64,
    pub display_frames: u64,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.status == Status::Done
    }

    pub fn render_text(&self) -> String {
        let mut s = String::new();
        let verdict = match &self.status {
            Status::Done => "DONE".to_string(),
            Status::Running => "STOPPED (the run ended before the playbook did)".to_string(),
            Status::Failed(m) => format!("FAILED: {m}"),
        };
        s.push_str(&format!(
            "PLAYBOOK {} {verdict}\n  input frames {} display frames {}\n",
            self.name, self.input_frames, self.display_frames
        ));
        for p in &self.phases {
            let speed = match p.wall_s {
                Some(w) if w > 0.0 => format!("wall {w:.1}s speed {:.0}%", 100.0 * p.guest_s / w),
                _ => String::new(),
            };
            s.push_str(&format!(
                "PHASE {:indent$}{:<24} input {}-{} f{}-f{} guest {:.1}s {speed}\n",
                "",
                p.name,
                p.input.0,
                p.input.1,
                p.display.0,
                p.display.1,
                p.guest_s,
                indent = p.depth * 2
            ));
        }
        for n in &self.tape.notes {
            if let Some(name) = n.text.strip_prefix("split ") {
                s.push_str(&format!("SPLIT {name:<24} input {} f{} {:.3}s\n", n.input, n.display, n.seconds));
            }
        }
        for w in &self.warnings {
            s.push_str(&format!("WARN {w}\n"));
        }
        if let Some(d) = &self.discovery {
            s.push_str(d);
        }
        s
    }
}

enum Block {
    None,
    Hold { mask: u32, until: u64 },
    Idle { until: u64 },
    Wait { armed: Armed, cue: CueId, deadline_us: Option<u64>, soft: bool },
}

struct Phase {
    name: String,
    deadline_us: Option<u64>,
    timeout: Option<f64>,
    input: u64,
    display: u64,
    guest_us: u64,
    wall: Option<f64>,
}

enum Source {
    Script { vm: Option<Box<dyn LuaVm>> },
    Tape(Tape),
}

struct Cues {
    defs: Vec<CueDef>,
    /// The call slot of each call cue.
    slot: Vec<Option<usize>>,
    vals: Vec<Option<f64>>,
    prev: Vec<Option<f64>>,
    changed_at: Vec<u64>,
    changed_now: Vec<bool>,
    /// Whether the cue has been sampled at least once.
    sampled: Vec<bool>,
    /// A pixel cue's count in the last frame the host reported (see [`on_frame_pixels`]),
    /// taken into `vals` at the next input frame like any other state.
    pixels: Vec<Option<f64>>,
    input: u64,
    guest_us: u64,
}

impl Cues {
    /// A cue described in full, children included, so a timeout names what it was waiting
    /// for (`call sceIoOpen(r0 has "title") >= 1`) rather than an id.
    fn label(&self, id: CueId) -> String {
        self.label_depth(id, 0)
    }

    fn label_depth(&self, id: CueId, depth: usize) -> String {
        if depth > 4 {
            return format!("cue#{id}");
        }
        self.defs[id].describe(&|c| self.label_depth(c, depth + 1))
    }
}

impl CueState for Cues {
    fn def(&self, id: CueId) -> &CueDef {
        &self.defs[id]
    }

    fn value(&self, id: CueId) -> Option<f64> {
        match &self.defs[id] {
            CueDef::Call { .. } | CueDef::Mem { .. } | CueDef::Pixels { .. } | CueDef::Locate { .. } | CueDef::Color { .. } => {
                self.vals[id]
            }
            CueDef::Cmp { src, op, value } => self.value(*src).map(|v| op.eval(v, *value) as u8 as f64),
            CueDef::Changed(src) => Some(self.changed_now[*src] as u8 as f64),
            CueDef::Stable(src, n) => {
                self.value(*src).map(|_| (self.input.saturating_sub(self.changed_at[*src]) >= *n) as u8 as f64)
            }
            _ => None,
        }
    }

    fn changed_now(&self, id: CueId) -> bool {
        self.changed_now[id]
    }

    fn input_frame(&self) -> u64 {
        self.input
    }

    fn guest_us(&self) -> u64 {
        self.guest_us
    }
}

struct Engine {
    name: String,
    source: Source,
    cues: Cues,
    traces: Vec<(CueId, String, Option<f64>)>,
    ctrl: CtrlFrame,
    touch: Option<(u16, u16)>,
    /// The rear panel's finger, `back_touch` in a playbook.
    back: Option<(u16, u16)>,
    block: Block,
    /// A reply the playbook is owed: when set, the next pump resumes it.
    pending: Option<String>,
    phases: Vec<Phase>,
    done_phases: Vec<PhaseReport>,
    /// Ticks so far; the current input frame is `ticks - 1`.
    ticks: u64,
    last_tick_display: Option<u64>,
    display: u64,
    status: Status,
    events: Vec<Event>,
    tape: Tape,
    warnings: Vec<String>,
    wall: Option<fn() -> f64>,
    /// Whether this run owns the pad. A discovery-only run records calls and leaves input
    /// to whatever else drives the title.
    drives: bool,
}

static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);

/// How a run starts.
pub struct StartOpts {
    /// The playbook's name in errors and reports (`ABCD00001/example.lua`).
    pub name: String,
    /// Tape header lines describing the run (`title`, `mode`, ...).
    pub header: Vec<(String, String)>,
    /// Record every call and string argument for `--discover`.
    pub discover: bool,
    /// Wall-clock milliseconds, for per-phase speed. `None` = no speed figures.
    pub wall_ms: Option<fn() -> f64>,
    /// False for a discovery-only run: the playbook does not own the pad (see [`input`]).
    pub drives_input: bool,
}

struct Hook;

impl HostCallObserver for Hook {
    fn on_call(&self, call: &ObservedCall<'_>) {
        calls::note(call);
        calls::discover(call);
        if PAD_READS.contains(&call.nid)
            && let Ok(mut g) = ENGINE.lock()
            && let Some(e) = g.as_mut()
        {
            e.on_pad_read(call.mem, call.frame, call.guest_us);
        }
    }
}

static HOOK: Hook = Hook;

fn begin(engine: Engine, discover: bool) -> Result<(), String> {
    observer::install(&HOOK);
    calls::reset();
    *calls::DISCOVERY.lock().map_err(|_| "discovery lock poisoned")? = discover.then(calls::Discovery::default);
    *ENGINE.lock().map_err(|_| "engine lock poisoned")? = Some(engine);
    observer::set_enabled(true);
    Ok(())
}

fn new_engine(name: String, source: Source, header: Vec<(String, String)>, wall: Option<fn() -> f64>) -> Engine {
    Engine {
        name,
        source,
        cues: Cues {
            defs: vec![],
            slot: vec![],
            vals: vec![],
            prev: vec![],
            changed_at: vec![],
            changed_now: vec![],
            sampled: vec![],
            pixels: vec![],
            input: 0,
            guest_us: 0,
        },
        traces: vec![],
        ctrl: CtrlFrame::default(),
        touch: None,
        back: None,
        block: Block::None,
        pending: Some(proto::ok()),
        phases: vec![],
        done_phases: vec![],
        ticks: 0,
        last_tick_display: None,
        display: 0,
        status: Status::Running,
        events: vec![],
        tape: Tape { header, ..Tape::default() },
        warnings: vec![],
        wall,
        drives: true,
    }
}

/// Start a playbook run. `vm` is resumed inside the tick when given (native); `None` leaves
/// the VM with the caller, who loads it ([`vm::load`]) and drives it with [`pump_external`].
pub fn start(opts: StartOpts, source: &str, vm: Option<Box<dyn LuaVm>>) -> Result<(), String> {
    let vm = match vm {
        Some(mut vm) => {
            vm::load(vm.as_mut(), &opts.name, source)?;
            Some(vm)
        }
        None => None,
    };
    let mut header = vec![("playbook".to_string(), opts.name.clone())];
    header.extend(opts.header);
    header.push(("input-frame".into(), "first pad read after each display flip".into()));
    let mut e = new_engine(opts.name, Source::Script { vm }, header, opts.wall_ms);
    e.drives = opts.drives_input;
    begin(e, opts.discover)
}

/// Start replaying `tape`: its input on its input frames, no playbook.
pub fn start_tape(tape: Tape, name: String, discover: bool, wall_ms: Option<fn() -> f64>) -> Result<(), String> {
    let mut header = tape.header.clone();
    header.push(("replay-of".into(), name.clone()));
    begin(new_engine(name, Source::Tape(tape), header, wall_ms), discover)
}

/// Whether a run is in progress (started and not yet finished).
pub fn active() -> bool {
    ENGINE.lock().map(|g| g.is_some()).unwrap_or(false)
}

/// The input every pad read returns this display frame. The playbook drives the built-in pad,
/// the front panel (port 0) and the rear panel (port 1's touch).
pub fn input(port: u32) -> Option<(CtrlFrame, TouchFrame)> {
    let g = ENGINE.lock().ok()?;
    let e = g.as_ref().filter(|e| e.drives)?;
    if port != 0 {
        let back = if port == 1 { e.back.map(|(x, y)| TouchFrame::single(x, y)) } else { None };
        return Some((CtrlFrame::default(), back.unwrap_or_default()));
    }
    Some((e.ctrl, e.touch.map(|(x, y)| TouchFrame::single(x, y)).unwrap_or_default()))
}

/// The host's loop, once per display frame: enforce timeouts that a game which stopped
/// reading the pad would otherwise never reach, and resume an inline VM that is owed a reply.
pub fn on_display_frame(display: u64, guest_us: u64) {
    if let Ok(mut g) = ENGINE.lock()
        && let Some(e) = g.as_mut()
    {
        e.display = display;
        e.cues.guest_us = e.cues.guest_us.max(guest_us);
        e.check_deadlines();
        e.pump_inline();
    }
}

/// Whether the running playbook has declared a pixel cue, so the host must report each
/// presented frame's pixels ([`on_frame_pixels`]). A tape never needs them: it replays inputs.
pub fn wants_pixels() -> bool {
    ENGINE.lock().ok().is_some_and(|g| {
        g.as_ref().is_some_and(|e| {
            matches!(e.source, Source::Script { .. })
                && e.cues.defs.iter().any(|d| matches!(d, CueDef::Pixels { .. } | CueDef::Locate { .. } | CueDef::Color { .. }))
        })
    })
}

/// The presented frame, as `w` x `h` tightly packed RGBA: every pixel cue counts its matches
/// in it. A frame at another size than the 960x544 panel is sampled at the panel's
/// coordinates, scaled.
pub fn on_frame_pixels(w: u32, h: u32, rgba: &[u8]) {
    if w == 0 || h == 0 || rgba.len() < (w * h * 4) as usize {
        return;
    }
    if let Ok(mut g) = ENGINE.lock()
        && let Some(e) = g.as_mut()
    {
        let c = &mut e.cues;
        for i in 0..c.defs.len() {
            match c.defs[i] {
                CueDef::Pixels { x, y, w: rw, h: rh, rgb, tol } => {
                    c.pixels[i] = Some(scan_pixels(w, h, rgba, (x, y, rw, rh), rgb, tol, false) as f64);
                }
                CueDef::Locate { x, y, w: rw, h: rh, rgb, tol } => {
                    let at = scan_pixels(w, h, rgba, (x, y, rw, rh), rgb, tol, true);
                    c.pixels[i] = (at != u32::MAX).then_some(at as f64);
                }
                CueDef::Color { x, y, w: rw, h: rh } => {
                    c.pixels[i] = Some(average_rgb(w, h, rgba, (x, y, rw, rh)) as f64);
                }
                _ => {}
            }
        }
    }
}

/// How many pixels of the panel rectangle `r` match `rgb` within `tol`, in a `w` x `h` frame -
/// or, with `first`, where the first one is (`y * 1024 + x`, panel coordinates), `u32::MAX` for
/// none.
fn scan_pixels(w: u32, h: u32, rgba: &[u8], r: (u32, u32, u32, u32), rgb: [u8; 3], tol: u8, first: bool) -> u32 {
    let (rx, ry, rw, rh) = r;
    let mut n = 0;
    for py in ry..ry + rh {
        let sy = (py as u64 * h as u64 / 544) as u32;
        for px in rx..rx + rw {
            let sx = (px as u64 * w as u64 / 960) as u32;
            let o = ((sy.min(h - 1) * w + sx.min(w - 1)) * 4) as usize;
            if (0..3).all(|k| rgba[o + k].abs_diff(rgb[k]) <= tol) {
                if first {
                    return py * 1024 + px;
                }
                n += 1;
            }
        }
    }
    if first { u32::MAX } else { n }
}

/// The average colour of the panel rectangle `r` in a `w` x `h` frame, packed `0xRRGGBB`.
fn average_rgb(w: u32, h: u32, rgba: &[u8], r: (u32, u32, u32, u32)) -> u32 {
    let (rx, ry, rw, rh) = r;
    let mut sum = [0u64; 3];
    let mut n = 0u64;
    for py in ry..ry + rh {
        let sy = (py as u64 * h as u64 / 544) as u32;
        for px in rx..rx + rw {
            let sx = (px as u64 * w as u64 / 960) as u32;
            let o = ((sy.min(h - 1) * w + sx.min(w - 1)) * 4) as usize;
            for (k, s) in sum.iter_mut().enumerate() {
                *s += u64::from(rgba[o + k]);
            }
            n += 1;
        }
    }
    let n = n.max(1);
    ((sum[0] / n) as u32) << 16 | ((sum[1] / n) as u32) << 8 | (sum[2] / n) as u32
}

/// Drive a VM the engine does not hold (the browser's): resume it until it blocks.
pub fn pump_external(vm: &mut dyn LuaVm) {
    if let Ok(mut g) = ENGINE.lock()
        && let Some(e) = g.as_mut()
    {
        e.pump(vm);
    }
}

/// Take the run's pending output.
pub fn take_events() -> Vec<Event> {
    ENGINE.lock().ok().and_then(|mut g| g.as_mut().map(|e| std::mem::take(&mut e.events))).unwrap_or_default()
}

/// The run's status, or `None` when no run is active.
pub fn status() -> Option<Status> {
    ENGINE.lock().ok().and_then(|g| g.as_ref().map(|e| e.status.clone()))
}

/// End the run: stop observing calls and hand back the report.
pub fn finish() -> Option<Report> {
    observer::set_enabled(false);
    let discovery = calls::DISCOVERY.lock().ok().and_then(|mut d| d.take()).map(|d| d.render());
    let mut e = ENGINE.lock().ok()?.take()?;
    for (i, slot) in e.cues.slot.iter().enumerate() {
        if let Some(s) = slot
            && !calls::resolved(*s)
        {
            e.warnings.push(format!(
                "cue#{i} ({}) never matched a call - check the spelling against `--discover`",
                calls::slot_name(*s)
            ));
        }
    }
    let input_frames = e.ticks;
    e.tape.end = e.ticks.checked_sub(1).map(|i| (i, e.last_tick_display.unwrap_or(e.display)));
    Some(Report {
        name: e.name,
        status: e.status,
        tape: e.tape,
        phases: e.done_phases,
        warnings: e.warnings,
        discovery,
        input_frames,
        display_frames: e.display,
    })
}

impl Engine {
    fn input_frame(&self) -> u64 {
        self.ticks.saturating_sub(1)
    }

    fn seconds(&self) -> f64 {
        self.cues.guest_us as f64 / 1e6
    }

    fn note(&mut self, text: String) {
        let n = Note { input: self.input_frame(), display: self.display, seconds: self.seconds(), text };
        self.events.push(Event::Log(format!("[playbook] input {} f{} {:.3}s {}", n.input, n.display, n.seconds, n.text)));
        self.tape.notes.push(n);
    }

    fn fail(&mut self, msg: String) {
        if self.status != Status::Running {
            return;
        }
        let path: Vec<&str> = self.phases.iter().map(|p| p.name.as_str()).collect();
        let msg = if path.is_empty() { msg } else { format!("{msg} (in phase {})", path.join(" > ")) };
        self.note(format!("FAILED {msg}"));
        self.status = Status::Failed(msg);
        self.block = Block::None;
        self.pending = None;
        self.ctrl = CtrlFrame::default();
        self.touch = None;
        self.back = None;
    }

    fn running(&self) -> bool {
        self.status == Status::Running
    }

    fn on_pad_read(&mut self, mem: &dyn GuestRead, display: u64, guest_us: u64) {
        if self.last_tick_display == Some(display) || !self.running() {
            return;
        }
        self.last_tick_display = Some(display);
        self.display = display;
        self.ticks += 1;
        self.cues.input = self.input_frame();
        self.cues.guest_us = self.cues.guest_us.max(guest_us);
        self.sample(mem);
        if matches!(self.source, Source::Tape(_)) {
            // Taken out for the tick rather than cloned: a long tape is thousands of changes.
            let src = std::mem::replace(&mut self.source, Source::Script { vm: None });
            if let Source::Tape(t) = &src {
                self.replay_tick(t);
            }
            self.source = src;
            return;
        }
        self.trace();
        self.advance_block();
        self.check_deadlines();
        self.pump_inline();
        self.record();
    }

    fn replay_tick(&mut self, t: &Tape) {
        let input = self.input_frame();
        if let Some((end, _)) = t.end
            && input > end
        {
            self.note("tape ended".into());
            self.status = Status::Done;
            self.ctrl = CtrlFrame::default();
            self.touch = None;
            self.back = None;
            return;
        }
        if let Some(c) = t.change_at(input)
            && c.display != self.display
            && self.warnings.len() < 20
        {
            self.warnings.push(format!(
                "tape desync: input frame {input} was f{} when recorded and is f{} now",
                c.display, self.display
            ));
        }
        let (ctrl, touch) = t.input_at(input);
        self.ctrl = ctrl;
        self.touch = (touch.count > 0).then(|| (touch.points[0].x, touch.points[0].y));
        let back = t.back_at(input);
        self.back = (back.count > 0).then(|| (back.points[0].x, back.points[0].y));
        // The recorded run's shots, on the input frames it asked for them.
        let at = t.notes.partition_point(|n| n.input < input);
        for n in t.notes[at..].iter().take_while(|n| n.input == input) {
            if let Some(name) = n.text.strip_prefix("shot ") {
                self.note(format!("shot {name}"));
                self.events.push(Event::Shot(name.to_string()));
            }
        }
        self.record();
    }

    fn sample(&mut self, mem: &dyn GuestRead) {
        let c = &mut self.cues;
        for i in 0..c.defs.len() {
            let v = match &c.defs[i] {
                CueDef::Call { .. } => c.slot[i].map(|s| calls::count(s) as f64),
                CueDef::Mem { ty, base, offs } => read_mem(mem, *ty, *base, offs),
                CueDef::Pixels { .. } | CueDef::Locate { .. } | CueDef::Color { .. } => c.pixels[i],
                _ => continue,
            };
            // A cue's FIRST sample is its starting value, not a change: a memory cue declared
            // mid-tick has had no value until now.
            let first = !c.sampled[i];
            c.sampled[i] = true;
            c.prev[i] = if first { v } else { c.vals[i] };
            c.vals[i] = v;
            c.changed_now[i] = c.prev[i] != v;
            if c.changed_now[i] {
                c.changed_at[i] = c.input;
            }
        }
    }

    fn trace(&mut self) {
        for k in 0..self.traces.len() {
            let (id, label, last) = self.traces[k].clone();
            let v = self.cues.value(id);
            if v != last {
                self.traces[k].2 = v;
                let show = |v: Option<f64>| v.map(|v| format!("{v}")).unwrap_or_else(|| "nil".into());
                self.note(format!("trace {label} {} -> {}", show(last), show(v)));
            }
        }
    }

    fn advance_block(&mut self) {
        let input = self.input_frame();
        let reply = match &mut self.block {
            Block::None => return,
            Block::Hold { mask, until } => {
                if input < *until {
                    return;
                }
                self.ctrl.buttons &= !*mask;
                proto::ok()
            }
            Block::Idle { until } => {
                if input < *until {
                    return;
                }
                proto::ok()
            }
            Block::Wait { armed, .. } => match armed.poll(&self.cues) {
                Some(v) => proto::value(v),
                None => return,
            },
        };
        self.block = Block::None;
        self.pending = Some(reply);
    }

    fn check_deadlines(&mut self) {
        if !self.running() {
            return;
        }
        let now = self.cues.guest_us;
        if let Block::Wait { deadline_us: Some(d), soft, cue, .. } = &self.block
            && now >= *d
        {
            if *soft {
                self.block = Block::None;
                self.pending = Some(proto::value(None));
            } else {
                let what = self.cues.label(*cue);
                self.fail(format!("timed out waiting for {what}"));
                return;
            }
        }
        if let Some(p) = self.phases.iter().find(|p| p.deadline_us.is_some_and(|d| now >= d)) {
            let msg = format!("phase {} ran past its {}s timeout", p.name, p.timeout.unwrap_or(0.0));
            self.fail(msg);
        }
    }

    fn pump_inline(&mut self) {
        if self.pending.is_none() {
            return;
        }
        let taken = match &mut self.source {
            Source::Script { vm } => vm.take(),
            Source::Tape(_) => None,
        };
        if let Some(mut vm) = taken {
            self.pump(vm.as_mut());
            if let Source::Script { vm: slot } = &mut self.source {
                *slot = Some(vm);
            }
        }
    }

    fn pump(&mut self, vm: &mut dyn LuaVm) {
        let mut n = 0usize;
        while let Some(reply) = self.pending.take() {
            if !self.running() {
                return;
            }
            n += 1;
            if n > MAX_REQUESTS_PER_PUMP {
                self.fail(format!(
                    "the playbook made {MAX_REQUESTS_PER_PUMP} requests without waiting on the game - a loop with no wait, idle or hold in it never ends"
                ));
                return;
            }
            let text = match vm::resume(vm, &reply) {
                Ok(t) => t,
                Err(e) => {
                    self.fail(format!("the Lua VM failed: {e}"));
                    return;
                }
            };
            match proto::parse(&text) {
                Ok(req) => self.handle(req),
                Err(e) => self.pending = Some(proto::error(&e)),
            }
        }
    }

    fn mask_of(names: &[String]) -> Result<u32, String> {
        names.iter().try_fold(0u32, |m, n| Ok(m | tape::button_bit(n).ok_or_else(|| format!("unknown button {n:?}"))?))
    }

    fn deadline(&self, timeout: Option<f64>) -> Option<u64> {
        let explicit = timeout.map(|t| self.cues.guest_us + (t * 1e6) as u64);
        self.phases.iter().filter_map(|p| p.deadline_us).chain(explicit).min()
    }

    fn add_cue(&mut self, def: CueDef) -> Result<CueId, String> {
        let n = self.cues.defs.len();
        // A playbook declares tens of cues. Thousands means one is declared inside a loop
        // (`until try_wait(c:happened(), 1)` makes a new condition every time round), which
        // grows the per-frame sampling without bound - say so rather than slow down.
        if n >= MAX_CUES {
            return Err(format!("{MAX_CUES} cues declared - declare cues (and their `:happened()` etc.) once, outside loops"));
        }
        for c in def.children() {
            if c >= n {
                return Err(format!("cue#{c} does not exist"));
            }
        }
        let is_state = |c: CueId| {
            matches!(
                self.cues.defs[c],
                CueDef::Call { .. } | CueDef::Mem { .. } | CueDef::Pixels { .. } | CueDef::Locate { .. } | CueDef::Color { .. }
            )
        };
        match &def {
            CueDef::Cmp { src, .. } | CueDef::Changed(src) | CueDef::Stable(src, _) if !is_state(*src) => {
                return Err(format!("{} needs a call or memory cue, not {}", def.describe(&|c| format!("cue#{c}")), self.cues.label(*src)));
            }
            _ => {}
        }
        let slot = match &def {
            CueDef::Call { name, filter } => Some(calls::add(name, filter.clone())?),
            _ => None,
        };
        let c = &mut self.cues;
        c.defs.push(def);
        c.slot.push(slot);
        c.vals.push(slot.map(|s| calls::count(s) as f64));
        c.prev.push(None);
        c.changed_at.push(c.input);
        c.changed_now.push(false);
        c.sampled.push(false);
        c.pixels.push(None);
        Ok(n)
    }

    fn handle(&mut self, req: Request) {
        let reply = match req {
            Request::Cue(def) => match self.add_cue(def) {
                Ok(id) => proto::id(id),
                Err(e) => proto::error(&e),
            },
            Request::Value(id) if id < self.cues.defs.len() => proto::value(self.cues.value(id)),
            Request::Value(id) | Request::Trace { cue: id, .. } | Request::Check { cue: id, .. } | Request::Wait { cue: id, .. }
                if id >= self.cues.defs.len() =>
            {
                proto::error(&format!("cue#{id} does not exist"))
            }
            Request::Value(_) => unreachable!(),
            Request::Trace { cue, label } => {
                let v = self.cues.value(cue);
                self.note(format!("trace {label} = {}", v.map(|v| v.to_string()).unwrap_or_else(|| "nil".into())));
                self.traces.push((cue, label, v));
                proto::ok()
            }
            Request::Phase { name, timeout } => {
                self.note(format!("phase {name}"));
                self.phases.push(Phase {
                    deadline_us: timeout.map(|t| self.cues.guest_us + (t * 1e6) as u64),
                    timeout,
                    name,
                    input: self.input_frame(),
                    display: self.display,
                    guest_us: self.cues.guest_us,
                    wall: self.wall.map(|w| w()),
                });
                proto::ok()
            }
            Request::EndPhase { name } => match self.phases.pop() {
                Some(p) if p.name == name => {
                    let wall_s = match (p.wall, self.wall) {
                        (Some(a), Some(w)) => Some((w() - a) / 1000.0),
                        _ => None,
                    };
                    let guest_s = (self.cues.guest_us - p.guest_us) as f64 / 1e6;
                    self.note(format!("end {name} ({guest_s:.1}s)"));
                    self.done_phases.push(PhaseReport {
                        name,
                        depth: self.phases.len(),
                        input: (p.input, self.input_frame()),
                        display: (p.display, self.display),
                        guest_s,
                        wall_s,
                    });
                    proto::ok()
                }
                other => proto::error(&format!("phase {name} ended while {:?} was open", other.map(|p| p.name))),
            },
            Request::Wait { cue, timeout, soft } => {
                let Some(deadline_us) = self.deadline(timeout) else {
                    let reply = proto::error(
                        "a wait needs a timeout: pass { timeout = <seconds> } or put it inside a phase that has one",
                    );
                    self.pending = Some(reply);
                    return;
                };
                let mut armed = Armed::arm(&self.cues, cue);
                match armed.poll(&self.cues) {
                    Some(v) => proto::value(v),
                    None => {
                        self.block = Block::Wait { armed, cue, deadline_us: Some(deadline_us), soft };
                        return;
                    }
                }
            }
            Request::Press(names) => match Self::mask_of(&names) {
                Ok(m) => {
                    self.ctrl.buttons |= m;
                    proto::ok()
                }
                Err(e) => proto::error(&e),
            },
            Request::Release(names) => match Self::mask_of(&names) {
                Ok(_) if names.is_empty() => {
                    self.ctrl.buttons = 0;
                    proto::ok()
                }
                Ok(m) => {
                    self.ctrl.buttons &= !m;
                    proto::ok()
                }
                Err(e) => proto::error(&e),
            },
            Request::Hold { buttons, frames } => match Self::mask_of(&buttons) {
                Ok(_) if frames == 0 => proto::ok(),
                Ok(mask) => {
                    self.ctrl.buttons |= mask;
                    self.block = Block::Hold { mask, until: self.input_frame() + frames };
                    return;
                }
                Err(e) => proto::error(&e),
            },
            Request::Idle(0) => proto::ok(),
            Request::Idle(frames) => {
                self.block = Block::Idle { until: self.input_frame() + frames };
                return;
            }
            Request::Stick { right, x, y } => {
                if right {
                    (self.ctrl.rx, self.ctrl.ry) = (x, y);
                } else {
                    (self.ctrl.lx, self.ctrl.ly) = (x, y);
                }
                proto::ok()
            }
            Request::Touch { back, at } => {
                if back {
                    self.back = at;
                } else {
                    self.touch = at;
                }
                proto::ok()
            }
            Request::Shot(name) => {
                // On the timeline too, so a replay of the tape asks for the same shot on the
                // same input frame - the byte comparison a tape exists for.
                self.note(format!("shot {name}"));
                self.events.push(Event::Shot(name));
                proto::ok()
            }
            Request::Log(text) => {
                self.note(format!("log {text}"));
                proto::ok()
            }
            Request::Split(name) => {
                self.note(format!("split {name}"));
                proto::ok()
            }
            Request::Check { cue, msg } => {
                if self.cues.value(cue).is_some_and(|v| v != 0.0) {
                    proto::ok()
                } else {
                    let what = self.cues.label(cue);
                    self.fail(format!("check failed: {what}{}", msg.map(|m| format!(" - {m}")).unwrap_or_default()));
                    return;
                }
            }
            Request::Now => proto::now(self.input_frame(), self.display, self.seconds()),
            Request::Done => {
                self.note("done".into());
                self.status = Status::Done;
                return;
            }
            Request::Error(msg) => {
                self.fail(msg);
                return;
            }
        };
        self.pending = Some(reply);
    }

    fn record(&mut self) {
        let change =
            Change { input: self.input_frame(), display: self.display, ctrl: self.ctrl, touch: self.touch, back: self.back };
        let same = self
            .tape
            .changes
            .last()
            .is_some_and(|c| c.ctrl == change.ctrl && c.touch == change.touch && c.back == change.back);
        if !same {
            self.tape.changes.push(change);
        }
    }
}

fn read_mem(mem: &dyn GuestRead, ty: crate::cue::MemTy, base: u32, offs: &[u32]) -> Option<f64> {
    let mut addr = base;
    for off in offs {
        let mut p = [0u8; 4];
        if !mem.read_into(addr, &mut p) {
            return None;
        }
        addr = u32::from_le_bytes(p).wrapping_add(*off);
    }
    let mut b = [0u8; 4];
    let w = ty.width();
    if !mem.read_into(addr, &mut b[..w]) {
        return None;
    }
    ty.decode(&b[..w])
}

#[cfg(all(test, feature = "native", not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::lua_mlua::MluaVm;

    /// `cue.color`'s value is the rectangle's average, packed `0xRRGGBB`, at panel coordinates
    /// whatever the frame's own size.
    #[test]
    fn a_colour_cue_averages_its_rectangle() {
        // A 2x1 frame: left pixel (10, 20, 30), right pixel (30, 40, 50).
        let rgba = [10, 20, 30, 255, 30, 40, 50, 255];
        assert_eq!(average_rgb(2, 1, &rgba, (0, 0, 960, 544)), 0x0014_1e28);
        assert_eq!(average_rgb(2, 1, &rgba, (480, 0, 480, 544)), 0x001e_2832);
    }

    /// The engine is process-wide; its tests take turns.
    static SERIAL: Mutex<()> = Mutex::new(());

    /// A guest whose memory is one u32 at `0x8100_0000` and one pointer at `0x8100_0010`.
    struct Mem {
        value: u32,
    }

    impl GuestRead for Mem {
        fn read_into(&self, addr: u32, out: &mut [u8]) -> bool {
            let word = match addr {
                0x8100_0000 => self.value,
                0x8100_0010 => 0x8100_0000,
                _ => return false,
            };
            out.copy_from_slice(&word.to_le_bytes()[..out.len()]);
            true
        }
    }

    const PAD: u32 = vitaslop_runtime::nid::ctrl::PEEK_BUFFER_POSITIVE;

    /// One display frame of a fake game: optionally make `call`, then read the pad.
    fn frame(display: u64, mem: &Mem, call: Option<u32>) -> CtrlFrame {
        let regs = [0u32; 16];
        let us = display * 16_666;
        if let Some(nid) = call {
            HOOK.on_call(&ObservedCall { nid, regs: &regs, mem, frame: display, guest_us: us });
        }
        HOOK.on_call(&ObservedCall { nid: PAD, regs: &regs, mem, frame: display, guest_us: us });
        // A second read in the same display frame is the same input frame.
        HOOK.on_call(&ObservedCall { nid: PAD, regs: &regs, mem, frame: display, guest_us: us });
        on_display_frame(display, us);
        input(0).unwrap().0
    }

    fn opts() -> StartOpts {
        StartOpts { name: "test.lua".into(), header: vec![], discover: false, wall_ms: None, drives_input: true }
    }

    const CROSS: u32 = 0x4000;
    const START: u32 = 0x0008;

    #[test]
    fn a_playbook_waits_on_a_call_then_a_value_and_records_a_tape() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let src = r#"
            local pad = cue.call("sceCtrlPeekBufferPositive")
            local open = cue.call("sceIoOpen")
            local v = cue.u32(ptr(0x81000010, 0))
            phase("menu", { timeout = 60 }, function()
              wait(open)
              split("opened")
              tap("start", 2, 1)
              repeat tap("cross", 1, 1) until v:value() == 3
              hold("cross", 2)
            end)
        "#;
        start(opts(), src, Some(Box::new(MluaVm::new()))).unwrap();
        let mut mem = Mem { value: 0 };
        let open = vitaslop_runtime::nid::iofilemgr::IO_OPEN;
        let mut seen = Vec::new();
        for f in 1..=40u64 {
            if f == 20 {
                mem.value = 3;
            }
            seen.push(frame(f, &mem, (f == 5).then_some(open)).buttons);
            if status() != Some(Status::Running) {
                break;
            }
        }
        let report = finish().unwrap();
        assert_eq!(report.status, Status::Done, "{}", report.render_text());
        // The open on display frame 5 is seen at that frame's pad read: start goes down there
        // for two input frames, then is up for one.
        assert_eq!(&seen[4..7], &[START, START, 0], "{seen:?}");
        assert!(seen.contains(&CROSS));
        let text = report.tape.render();
        let back = Tape::parse(&text).unwrap();
        assert_eq!(back.changes, report.tape.changes, "{text}");
        assert!(report.tape.notes.iter().any(|n| n.text == "split opened" && n.display == 5), "{text}");
    }

    #[test]
    fn a_wait_past_its_phase_timeout_fails_and_names_the_cue_and_phase() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let src = r#"
            phase("boot", { timeout = 0.2 }, function()
              wait(cue.call("sceNeverCalled"))
            end)
        "#;
        start(opts(), src, Some(Box::new(MluaVm::new()))).unwrap();
        let mem = Mem { value: 0 };
        for f in 1..=30u64 {
            frame(f, &mem, None);
        }
        let report = finish().unwrap();
        match &report.status {
            Status::Failed(m) => {
                assert!(m.contains("timed out waiting for call sceNeverCalled"), "{m}");
                assert!(m.contains("in phase boot"), "{m}");
            }
            other => panic!("{other:?}"),
        }
        assert!(report.warnings.iter().any(|w| w.contains("sceNeverCalled")), "{:?}", report.warnings);
    }

    #[test]
    fn a_wait_with_no_timeout_anywhere_is_an_error_at_its_line() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        start(opts(), "\n\nwait(cue.frames(3))\n", Some(Box::new(MluaVm::new()))).unwrap();
        frame(1, &Mem { value: 0 }, None);
        let report = finish().unwrap();
        match &report.status {
            Status::Failed(m) => assert!(m.contains("test.lua:3:") && m.contains("needs a timeout"), "{m}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_tape_replays_the_recorded_input_on_the_same_input_frames() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let src = r#"
            phase("p", { timeout = 60 }, function()
              idle(3); hold("start", 2); shot("mid"); idle(2); press("cross"); idle(4)
            end)
        "#;
        start(opts(), src, Some(Box::new(MluaVm::new()))).unwrap();
        let mem = Mem { value: 0 };
        let recorded: Vec<u32> = (1..=14u64).map(|f| frame(f, &mem, None).buttons).collect();
        let shots = |evs: Vec<Event>| evs.into_iter().filter(|e| matches!(e, Event::Shot(_))).collect::<Vec<_>>();
        let recorded_shots = shots(take_events());
        let report = finish().unwrap();
        assert_eq!(report.status, Status::Done);
        start_tape(Tape::parse(&report.tape.render()).unwrap(), "t".into(), false, None).unwrap();
        let replayed: Vec<u32> = (1..=14u64).map(|f| frame(f, &mem, None).buttons).collect();
        let replayed_shots = shots(take_events());
        let r = finish().unwrap();
        assert_eq!(recorded[..12], replayed[..12], "{}", report.tape.render());
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        assert_eq!(recorded_shots, vec![Event::Shot("mid".into())]);
        assert_eq!(replayed_shots, recorded_shots, "a replay asks for the recorded run's shots");
        let mid = |r: &Report| r.tape.notes.iter().find(|n| n.text == "shot mid").map(|n| n.input);
        assert_eq!(mid(&r), mid(&report), "...on the same input frame");
    }
}

#[cfg(test)]
mod pixel_tests {
    use super::scan_pixels;

    /// A frame of `w` x `h` black pixels with a white `s` x `s` block at panel `(bx, by)`.
    fn frame(w: u32, h: u32, bx: u32, by: u32, s: u32) -> Vec<u8> {
        let mut f = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let (px, py) = (x * 960 / w, y * 544 / h);
                if px >= bx && px < bx + s && py >= by && py < by + s {
                    let o = ((y * w + x) * 4) as usize;
                    f[o..o + 4].copy_from_slice(&[255, 255, 255, 255]);
                }
            }
        }
        f
    }

    #[test]
    fn a_box_counts_the_pixels_of_a_colour_and_locate_finds_the_first() {
        let f = frame(960, 544, 100, 200, 12);
        let white = [255, 255, 255];
        assert_eq!(scan_pixels(960, 544, &f, (90, 190, 20, 20), white, 30, false), 100, "10x10 of the block lies inside");
        assert_eq!(scan_pixels(960, 544, &f, (0, 0, 50, 50), white, 30, false), 0);
        assert_eq!(scan_pixels(960, 544, &f, (0, 150, 300, 100), white, 30, true), 200 * 1024 + 100);
        assert_eq!(scan_pixels(960, 544, &f, (0, 0, 50, 50), white, 30, true), u32::MAX, "no match");
        assert_eq!(scan_pixels(960, 544, &f, (90, 190, 20, 20), [250, 250, 250], 4, false), 0, "outside the tolerance");
    }

    #[test]
    fn a_frame_at_another_size_is_read_at_panel_coordinates() {
        // A 480x272 frame: each of its pixels covers 2x2 of the panel.
        let f = frame(480, 272, 100, 200, 12);
        assert_eq!(scan_pixels(480, 272, &f, (100, 200, 12, 12), [255, 255, 255], 30, false), 144);
    }
}
