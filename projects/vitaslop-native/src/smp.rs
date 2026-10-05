//! >>> PARALLEL GUEST THREADS on the desktop (`VITASLOP_SMP=1`): the guest's threads run AT ONCE
//! on several OS threads over the one shared memory, instead of one at a time.
//!
//! This is the browser's parallel scheduler (`vitaslop-web/src/smp.rs`) on native threads, and
//! its POLICY is that one, restated rather than reinvented: placement by the title's CPU affinity
//! mask (the Vita's cores are bits 16..18, a thread is bound to one worker for life), strict
//! priority with round-robin inside a level per worker, the spin cooldown, a better-priority wake
//! preempting its worker, the wake token for a wake that races ahead of its block, a timed-out
//! thread yielding to the ones already waiting, the idle clock jump taken only when NOTHING can
//! run anywhere, the frame GATE, real-time (wall-parked) threads running behind a shut gate, and
//! the game clock floored at the wall. Where this file and that one disagree, that one is the
//! measured reference - every rule there carries the run that justified it.
//!
//! What is NOT here, because native does not need it: forwarding. The browser forwards the calls
//! whose handlers need its run worker's JavaScript (decoders, location, save data); every handler
//! here is plain Rust and runs on the worker that made the call, under the host lock. The
//! small-target completion (`VitaState::complete_scene_now`) runs inside its host call too, on
//! the window's renderer behind its own lock.
//!
//! # Determinism
//! OFF unless asked for. A parallel run is not reproducible - which thread reaches a lock first
//! depends on the host's timing - and every recipe, capsule and bisection depends on the
//! one-at-a-time engine being exactly that. The desktop window turns it on through the same
//! `VITASLOP_SMP` knob the browser's settings set; headless runs leave it off.
//!
//! # The present
//! Stop-the-world: [`SmpRun::run_frames`] returns once the target frame has flipped AND every
//! worker has reached a switch point, so the frame's scenes are complete and nothing writes guest
//! memory while the presenter reads it - except a real-time thread (the audio output), which keeps
//! running behind the shut gate as the device's audio does, and touches no render target.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use vitaslop_runtime::sched::{FiberEnd, GuestThread, RunReport, Stop, ThreadHandle, ThreadStep};
use vitaslop_runtime::{ImportDispatch, VitaEnv};
use vitaslop_transpiler::abi;

use crate::threaded::{SmpSpawn, WasmtimeEngine, WasmtimeThread};

/// Whether this run asked for parallel guest threads. Read once: the transpile, the link and the
/// scheduler must agree for the whole run.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| vitaslop_runtime::knobs::flag("VITASLOP_SMP"))
}

/// Guest workers: `VITASLOP_SMP_WORKERS`, default 3 - the Vita gives a title three cores, and
/// its threads' affinity masks name them. The browser's rule, the browser's range.
pub fn worker_count() -> usize {
    vitaslop_runtime::knobs::var("VITASLOP_SMP_WORKERS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| (1..=8).contains(&n))
        .unwrap_or(3)
}

/// Present frame N while the guest workers build N+1: the gate stands ONE frame ahead of the
/// present, and the presenter takes only the flipped frame's scenes
/// (`Capture::take_scenes_through_flip`). ON by default, as in the browser - whose measurement
/// is the reason (the run worker's frame cost went to ~0 there); `VITASLOP_SMP_OVERLAP=0` makes
/// every present wait for every worker to reach a switch point (stop-the-world). MEASURED here
/// without it (MLB window, 13,500 frames): 52.5 fps with the three workers 10%, 12% and 3% busy -
/// the guest frame and the present ran back to back on the window's one thread.
pub fn overlap() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| vitaslop_runtime::knobs::var("VITASLOP_SMP_OVERLAP").as_deref().map(str::trim) != Ok("0"))
}

/// A SLOW PRESENT DOES NOT FREEZE THE GAME. While the presenter is inside a present (marked by
/// [`LatePresent`]), a helper opens the frame gate one frame at a time for as long as the guest's
/// clock is behind the wall time the present has taken, so the guest runs on in real time - its
/// sound mixer above all - and the next `run_frames` takes the newest frame. On the Vita a late
/// frame blocks only the thread that waits for the display; the rest of the game keeps time.
/// MEASURED without it (a fighting title's round start, window, a 238 ms present of first-draw
/// pipeline builds): game time moved 34 ms, the speakers went dry 179 ms. Only a present that
/// outlasts the guest's one frame of lead is touched. `VITASLOP_SMP_LATE_PRESENT=0` is the arm.
pub fn late_present() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| overlap() && vitaslop_runtime::knobs::var("VITASLOP_SMP_LATE_PRESENT").as_deref().map(str::trim) != Ok("0"))
}

/// How far past the frame being presented the late-present helper lets the guest run. A bound,
/// not a pace: the pace is the wall clock. 1.5 s at 60 fps - a browser's first-draw pipeline
/// waits reach 1.2 s - and under the frames `flip_clock` remembers ([`FLIP_CLOCK_KEEP`]), so
/// the catch-up's game time is always known. The scenes of all but the last
/// [`LATE_KEEP_FRAMES`] are retired as they flip (`Capture::untaken_frame_limit`).
const LATE_AHEAD_MAX: u64 = 90;

/// A present only counts as LATE once it has run this long; the gate is then opened for the
/// whole of it. An ordinary present is never touched: on a title whose own frames run slower
/// than real time, every present finds the guest "behind" and opening there skipped a frame in
/// twenty while the catch-up's charge held the guest back - MEASURED (an action title's
/// prologue, window): underrun 0.14 -> 1.09 s, 383 frames not shown. A first-draw pipeline
/// build, what this is for, is hundreds of ms.
const LATE_AFTER: Duration = Duration::from_millis(50);

/// Frames whose flip clock is remembered - see [`LATE_AHEAD_MAX`].
const FLIP_CLOCK_KEEP: usize = 128;

/// Flipped-but-untaken frames that keep their scenes through a late present: the catch-up
/// renders these (a frame may sample the one before) and shows the newest.
const LATE_KEEP_FRAMES: usize = 4;

/// The presenter's side of [`late_present`]: `begin` before a present, `end` after it.
#[derive(Clone)]
pub struct LatePresent {
    sh: Arc<Shared>,
}

impl LatePresent {
    pub fn begin(&self) {
        let clock0 = self.sh.lock_host().clock_us();
        *self.sh.late.at.lock().unwrap() = Some((Instant::now(), clock0));
        self.sh.late.cv.notify_all();
    }

    pub fn end(&self) {
        *self.sh.late.at.lock().unwrap() = None;
    }
}

/// The present in progress - `(wall, guest clock)` when it began, `None` outside one - and the
/// gate openings the helper has made.
#[derive(Default)]
struct LateState {
    at: Mutex<Option<(Instant, u64)>>,
    cv: Condvar,
    opened: AtomicU64,
}

/// The helper: asleep outside a present; inside one, every millisecond, opens the gate one
/// frame when the guest is held at it and its clock is behind the present's wall time.
fn late_loop(sh: &Shared) {
    loop {
        let began = {
            let mut at = sh.late.at.lock().unwrap();
            loop {
                if sh.stop.load(Ordering::SeqCst) {
                    return;
                }
                if let Some(a) = *at {
                    break a;
                }
                at = sh.late.cv.wait_timeout(at, Duration::from_millis(50)).unwrap().0;
            }
        };
        let (wall0, clock0) = began;
        while !sh.stop.load(Ordering::SeqCst) && *sh.late.at.lock().unwrap() == Some(began) {
            if wall0.elapsed() < LATE_AFTER {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            let opened = {
                let mut st = sh.state.lock().unwrap();
                let held = st.frames >= st.gate && st.verdict.is_none() && !st.paused;
                if held && st.gate < st.consumed + 1 + LATE_AHEAD_MAX {
                    let mut host = sh.lock_host();
                    if host.clock_us() < clock0 + wall0.elapsed().as_micros() as u64 {
                        // As `run_frames` does when it opens the gate: the hold so far is closed
                        // off on the wall floor rather than counted when the guest moves again.
                        if wall_floor() {
                            host.state.wall_floor_tick(abs_ms(), true);
                        }
                        st.gate += 1;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            };
            if opened {
                sh.late.opened.fetch_add(1, Ordering::Relaxed);
                sh.ring_all();
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// The flip's geometry and texture-set reads on a RESOLVER thread of their own, applied by the
/// presenter before it takes the frame - the browser's `VITASLOP_SMP_ASYNC_RESOLVE`, same knob,
/// same default (on whenever [`overlap`] is). Without it every `sceGxmDraw` whose texture set
/// changed snapshots and compares the textures inline on the drawing guest thread. MEASURED
/// (Uncharted ruins, window, VITASLOP_PERF from f5500): `sceGxmDraw` 39 ms of a ~40 ms main
/// thread per flip, all of it the fragment-set MISS decode (7.8 ms each, 5 a frame); game clock
/// 80-84% of the wall where the browser, which runs this resolver, holds 100%.
pub fn async_resolve() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| overlap() && vitaslop_runtime::knobs::var("VITASLOP_SMP_ASYNC_RESOLVE").as_deref().map(str::trim) != Ok("0"))
}

/// A clock every worker reads the same way, in ms - the one handed to the runtime as its HOST
/// WALL CLOCK (`host::set_host_wall_clock`), which wall parks (the audio output's pacing) and the
/// wall floor are measured on.
pub fn abs_ms() -> f64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// The game clock is floored at the wall - see `VitaState::wall_floor_tick`. ON by default under
/// SMP, as in the browser; `VITASLOP_CLOCK_WALL_FLOOR=0` is the arm back.
fn wall_floor() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| vitaslop_runtime::knobs::var("VITASLOP_CLOCK_WALL_FLOOR").as_deref().map(str::trim) != Ok("0"))
}

/// A thread released by TIME yields to the threads already runnable on its worker - see
/// `drain` and the browser's measurement behind it. `VITASLOP_SMP_TIMEOUT_YIELDS=0`.
fn timeout_yields() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| vitaslop_runtime::knobs::var("VITASLOP_SMP_TIMEOUT_YIELDS").as_deref().map(str::trim) != Ok("0"))
}

/// Real-time (wall-parked) threads run behind a shut gate. `VITASLOP_SMP_RT_THROUGH_GATE=0`.
fn rt_through_gate() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| vitaslop_runtime::knobs::var("VITASLOP_SMP_RT_THROUGH_GATE").as_deref().map(str::trim) != Ok("0"))
}

/// `VITASLOP_SMP_FIBER_NEAR=1`: a fiber's backing thread is placed on its runner's worker. OFF by
/// default, as in the browser (see the open phone A/B there).
fn fiber_near_on() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| vitaslop_runtime::knobs::var("VITASLOP_SMP_FIBER_NEAR").as_deref().map(str::trim) == Ok("1"))
}

/// `VITASLOP_SMP_UNPIN_STARVED=1`: the browser's opt-in starvation unpin. OFF by default there and
/// here (it breaks Dead or Alive 5 Plus's movie - see the browser's note).
fn unpin_starved_on() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| vitaslop_runtime::knobs::var("VITASLOP_SMP_UNPIN_STARVED").as_deref().map(str::trim) == Ok("1"))
}

/// `VITASLOP_SMP_PLACE` - the browser's placement modes, `apart` the default. See there for what
/// each measured.
#[derive(Clone, Copy, PartialEq)]
enum Place {
    Load,
    Spread,
    Apart,
    Busy,
    Demand,
}

fn place_mode() -> Place {
    static V: OnceLock<Place> = OnceLock::new();
    *V.get_or_init(|| match vitaslop_runtime::knobs::var("VITASLOP_SMP_PLACE").as_deref().map(str::trim) {
        Ok("spread") => Place::Spread,
        Ok("load") => Place::Load,
        Ok("busy") => Place::Busy,
        Ok("demand") => Place::Demand,
        _ => Place::Apart,
    })
}

const YOUNG_DEMAND: f64 = 0.05;
const NAMED_CORE_SHARE: f64 = 0.5;
const IDLE_SHARE: f64 = 0.02;
const IDLE_GRACE_MS: f64 = 1000.0;

/// One worker's doorbell. Rung by bumping `seq` and notifying; a worker reads `seq` BEFORE it
/// decides it has nothing to do and waits only while it is unchanged, so a ring that lands in
/// between is never lost - the browser's futex word, as a condition variable.
struct Bell {
    seq: Mutex<u64>,
    cv: Condvar,
}

impl Bell {
    fn new() -> Bell {
        Bell { seq: Mutex::new(0), cv: Condvar::new() }
    }

    fn seen(&self) -> u64 {
        *self.seq.lock().unwrap()
    }

    fn ring(&self) {
        *self.seq.lock().unwrap() += 1;
        self.cv.notify_all();
    }

    fn wait(&self, seen: u64, timeout: Duration) {
        let g = self.seq.lock().unwrap();
        let _ = self.cv.wait_timeout_while(g, timeout, |s| *s == seen);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SState {
    Runnable,
    Running,
    Blocked,
    Finished(u32),
}

/// How to start a thread that has not run yet.
struct Birth {
    entries: Vec<u32>,
    r: [u32; 4],
    sp: u32,
}

struct Slot {
    thid: i32,
    priority: i32,
    home: usize,
    state: SState,
    cooled: bool,
    /// Waits on REAL time (woken from a wall park - the audio output thread): runs while the
    /// frame gate is shut.
    rt: bool,
    fuel_seen: u64,
    arm_seen: u64,
    birth: Option<Birth>,
    /// Finished while its instance still lives on its worker: that worker drops it.
    release: bool,
    born: f64,
    life_run_ms: f64,
    /// Per-thread words, written before each resume - see `threaded::SmpThread::words`.
    words: Arc<[AtomicU32; 2]>,
}

struct State {
    threads: Vec<Slot>,
    /// Slots not finished, ascending.
    live: Vec<usize>,
    wake_tokens: HashSet<i32>,
    /// Threads TIME released whose next wake is `cooled`.
    yield_next: HashSet<i32>,
    cursor: Vec<usize>,
    frames: u64,
    /// Guest threads run while `frames < gate`.
    gate: u64,
    running: usize,
    verdict: Option<RunReport>,
    /// Guest instructions retired per worker, decayed each frame - placement's idea of busy.
    load: Vec<f64>,
    running_on: Vec<Option<usize>>,
    fuel_total: u64,
    fuel_samples: u64,
    fuel_max: u64,
    arm_total: u64,
    idle_jumps: u64,
    resumes: u64,
    preempt_gate: u64,
    preempt_prio: u64,
    /// `SmpRun::consumed`, readable by the late-present helper: the frame being presented,
    /// which its opening of the gate is bounded from.
    consumed: u64,
    /// The presenter is writing guest memory the guest may also be writing (a render target's
    /// write-back): no worker picks a thread until it clears. See [`SmpRun::pause_guest`].
    paused: bool,
    /// The game clock at each recent flip, `(frame, us)`, newest last - so an overlapped present
    /// is paced by the game time ITS frame took, not by whatever the clock did while the next
    /// frame was already running. See [`SmpRun::frame_advance_us`].
    flip_clock: std::collections::VecDeque<(u64, u64)>,
}

/// Everything the workers share.
struct Shared {
    host: Arc<Mutex<VitaEnv>>,
    engine: Arc<WasmtimeEngine<VitaEnv>>,
    workers: usize,
    state: Mutex<State>,
    /// One doorbell per worker, index 0 = the presenter (`run_frames`'s caller).
    bells: Box<[Bell]>,
    /// Runnable threads per worker, kept current under the state lock.
    runnable: Arc<[AtomicUsize]>,
    stop: AtomicBool,
    /// Linear address of the mirror block, if the module has one.
    mirror: Option<usize>,
    /// Layout offset of the mirror block (what the module's back-edge check is pointed at).
    mirror_off: Option<u64>,
    /// Per-worker busy time, for the report.
    busy_us: Box<[AtomicU64]>,
    /// A worker that panicked, set without taking any lock.
    failed: OnceLock<String>,
    /// The present in progress, for [`late_present`].
    late: LateState,
}

impl Shared {
    fn ring(&self, w: usize) {
        if let Some(b) = self.bells.get(w) {
            b.ring();
        }
    }

    fn ring_all(&self) {
        for b in self.bells.iter() {
            b.ring();
        }
    }

    /// Worker `w`'s PREEMPT word - `abi::PREEMPT_SLOT_BASE + w` in the mirror block.
    fn preempt(&self, w: usize) -> Option<&AtomicI32> {
        let a = self.mirror? + 4 * (abi::PREEMPT_SLOT_BASE as usize + w);
        // SAFETY: inside the mirror block of the shared memory, which outlives this run; aligned;
        // only ever accessed as an i32.
        Some(unsafe { &*(a as *const AtomicI32) })
    }

    fn preempt_worker(&self, w: usize) {
        if let Some(p) = self.preempt(w) {
            p.store(1, Ordering::SeqCst);
        }
    }

    fn mirror_word(&self, slot: u32) -> Option<usize> {
        Some(self.mirror? + 4 * slot as usize)
    }

    fn lock_host(&self) -> MutexGuard<'_, VitaEnv> {
        self.host.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// >>> PUBLISH THE GLOBAL WORDS a parallel build reads inline - the vblank count and SA bank as
/// plain words, the process clock and RTC tick as one atomic 64-bit word each. Returns
/// `(current thread, reported thread id)`. The browser's `publish_words`.
fn publish_words(sh: &Shared, st: &vitaslop_runtime::host::VitaState) -> (u32, u32) {
    use vitaslop_runtime::vita::mirror as m;
    let s = m::snapshot(st);
    // SAFETY (all four): aligned words inside the mirror block, which lives for the whole run;
    // written atomically, read by the guest with plain or atomic loads.
    unsafe {
        if let Some(a) = sh.mirror_word(m::SLOT_VCOUNT) {
            (*(a as *const AtomicI32)).store(s[m::SLOT_VCOUNT as usize] as i32, Ordering::Relaxed);
        }
        if let Some(a) = sh.mirror_word(m::SLOT_SA_BANK) {
            (*(a as *const AtomicI32)).store(s[m::SLOT_SA_BANK as usize] as i32, Ordering::Relaxed);
        }
        let pair = |lo: u32| u64::from(s[lo as usize]) | (u64::from(s[lo as usize + 1]) << 32);
        if let Some(a) = sh.mirror_word(abi::SMP_CLOCK64_SLOT) {
            (*(a as *const AtomicU64)).store(pair(m::SLOT_CLOCK_LO), Ordering::Relaxed);
        }
        if let Some(a) = sh.mirror_word(abi::SMP_RTC64_SLOT) {
            (*(a as *const AtomicU64)).store(pair(m::SLOT_RTC_LO), Ordering::Relaxed);
        }
    }
    (s[m::SLOT_CURRENT_THREAD as usize], s[m::SLOT_THREAD_ID as usize])
}

/// Guest words through the shared memory, for the host's deferred settles - with a REAL atomic
/// compare-and-swap, which the one-at-a-time engine's read-then-write default cannot give while
/// other workers run.
struct Words<'a>(&'a WasmtimeEngine<VitaEnv>);

impl vitaslop_runtime::host::GuestWords for Words<'_> {
    fn word(&self, addr: u32) -> u32 {
        let mut b = [0u8; 4];
        if self.0.read_guest_into(addr, &mut b) { u32::from_le_bytes(b) } else { 0 }
    }
    fn set_word(&mut self, addr: u32, value: u32) {
        self.0.write_guest(addr, &value.to_le_bytes());
    }
    fn cas_word(&mut self, addr: u32, expect: u32, new: u32) -> u32 {
        match self.0.guest_atomic_u32(addr) {
            Some(a) => match a.compare_exchange(expect, new, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(v) | Err(v) => v,
            },
            None => {
                let cur = self.word(addr);
                if cur == expect {
                    self.set_word(addr, new);
                }
                cur
            }
        }
    }
}

impl State {
    fn new(workers: usize) -> State {
        State {
            threads: Vec::new(),
            live: Vec::new(),
            wake_tokens: HashSet::new(),
            yield_next: HashSet::new(),
            cursor: vec![0; workers + 1],
            frames: 0,
            gate: 0,
            running: 0,
            verdict: None,
            load: vec![0.0; workers + 1],
            running_on: vec![None; workers + 1],
            fuel_total: 0,
            fuel_samples: 0,
            fuel_max: 0,
            arm_total: 0,
            idle_jumps: 0,
            resumes: 0,
            preempt_gate: 0,
            consumed: 0,
            preempt_prio: 0,
            paused: false,
            flip_clock: std::collections::VecDeque::new(),
        }
    }

    /// A thread just became runnable: a WORSE-priority thread running on its worker gives up the
    /// CPU now - the SceKernel rule.
    fn preempt_for(&mut self, sh: &Shared, idx: usize) {
        let w = self.threads[idx].home;
        if let Some(r) = self.running_on[w]
            && self.threads[idx].priority < self.threads[r].priority
        {
            self.preempt_prio += 1;
            sh.preempt_worker(w);
        }
    }

    fn publish_runnable(&self, sh: &Shared) {
        let mut n = vec![0usize; sh.workers + 1];
        for &i in &self.live {
            if self.threads[i].state == SState::Runnable {
                n[self.threads[i].home] += 1;
            }
        }
        for (w, c) in n.into_iter().enumerate() {
            sh.runnable[w].store(c, Ordering::Relaxed);
            // ...and in the guest's own word, which the inline elided yield reads.
            let slot = abi::PREEMPT_SLOT_BASE + abi::SMP_RUNNABLE_SLOT_OFFSET + w as u32;
            if let Some(a) = sh.mirror_word(slot) {
                // SAFETY: an aligned word in the mirror block.
                unsafe { (*(a as *const AtomicI32)).store(c as i32, Ordering::Relaxed) };
            }
        }
    }

    /// The worker a new thread is bound to - the browser's `State::place`, rule for rule.
    fn place(&self, workers: usize, mask: i32, priority: i32) -> usize {
        let mut cands: Vec<usize> = if workers >= 3 {
            (1..=workers).filter(|&w| mask & (1 << (16 + (w - 1) % 3)) != 0).collect()
        } else {
            (0..3).filter(|k| mask & (1 << (16 + k)) != 0).map(|k| 1 + (k as usize) % workers).collect()
        };
        cands.sort_unstable();
        cands.dedup();
        if cands.is_empty() || mask & 0x0007_0000 == 0x0007_0000 {
            cands = (1..=workers).collect();
        }
        if unpin_starved_on()
            && cands.len() < workers
            && cands.iter().all(|&w| {
                self.live
                    .iter()
                    .map(|&i| &self.threads[i])
                    .any(|t| t.home == w && !matches!(t.state, SState::Finished(_)) && t.priority < priority)
            })
        {
            let away: Vec<usize> = (1..=workers).filter(|w| !cands.contains(w)).collect();
            if !away.is_empty() {
                cands = away;
            }
        }
        let named = cands.clone();
        if place_mode() == Place::Demand {
            cands = (1..=workers).collect();
        }
        if matches!(place_mode(), Place::Apart | Place::Busy | Place::Demand) && cands.len() > 1 {
            let main_home =
                self.threads.iter().find(|t| t.thid == vitaslop_runtime::host::MAIN_THID).map(|t| t.home);
            if let Some(m) = main_home {
                cands.retain(|&w| w != m);
            }
        }
        let now = abs_ms();
        let counts = |t: &Slot| {
            place_mode() != Place::Busy || !matches!(t.state, SState::Blocked | SState::Finished(_)) || {
                let age = now - t.born;
                age >= IDLE_GRACE_MS && t.life_run_ms >= IDLE_SHARE * age
            }
        };
        let live_on =
            |w: usize| self.live.iter().filter(|&&i| self.threads[i].home == w && counts(&self.threads[i])).count();
        let by_load = |a: usize, b: usize| self.load[a].partial_cmp(&self.load[b]).unwrap_or(std::cmp::Ordering::Equal);
        let _ = YOUNG_DEMAND;
        *cands
            .iter()
            .min_by(|&&a, &&b| {
                if place_mode() == Place::Demand {
                    let score = |w: usize| self.load[w] * if named.contains(&w) { NAMED_CORE_SHARE } else { 1.0 };
                    score(a)
                        .partial_cmp(&score(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(named.contains(&b).cmp(&named.contains(&a)))
                        .then(live_on(a).cmp(&live_on(b)))
                } else if matches!(place_mode(), Place::Spread | Place::Apart | Place::Busy) {
                    live_on(a).cmp(&live_on(b)).then(by_load(a, b))
                } else {
                    by_load(a, b).then(live_on(a).cmp(&live_on(b)))
                }
            })
            .unwrap_or(&1)
    }

    fn add_thread(&mut self, sh: &Shared, thid: i32, priority: i32, home: usize, birth: Birth) {
        let idx = self.threads.len();
        self.threads.push(Slot {
            thid,
            priority,
            home,
            state: SState::Runnable,
            cooled: false,
            rt: false,
            fuel_seen: 0,
            arm_seen: 0,
            birth: Some(birth),
            release: false,
            born: abs_ms(),
            life_run_ms: 0.0,
            words: Arc::new([AtomicU32::new(0), AtomicU32::new(0)]),
        });
        self.live.push(idx);
        tracing::info!(target: "vitaslop::smp", "thread {thid:#x} (prio {priority:#x}) bound to worker {home}");
        self.preempt_for(sh, idx);
        sh.ring(home);
    }

    /// Strict priority among worker `w`'s runnable threads, round-robin inside a level,
    /// spin-cooled threads stepping aside.
    fn pick(&mut self, w: usize, rt_only: bool) -> Option<usize> {
        let mine: Vec<usize> = self
            .live
            .iter()
            .copied()
            .filter(|&i| self.threads[i].home == w && self.threads[i].state == SState::Runnable)
            .filter(|&i| !rt_only || self.threads[i].rt)
            .collect();
        if mine.is_empty() {
            return None;
        }
        if mine.iter().all(|&i| self.threads[i].cooled) {
            for &i in &mine {
                self.threads[i].cooled = false;
            }
        }
        let best = mine.iter().filter(|&&i| !self.threads[i].cooled).map(|&i| self.threads[i].priority).min()?;
        let cursor = self.cursor[w];
        let start = mine.partition_point(|&i| i < cursor);
        let idx = mine[start..]
            .iter()
            .chain(&mine[..start])
            .copied()
            .find(|&i| !self.threads[i].cooled && self.threads[i].priority == best)?;
        self.cursor[w] = idx + 1;
        self.threads[idx].state = SState::Running;
        self.running += 1;
        self.running_on[w] = Some(idx);
        Some(idx)
    }

    fn retire(&mut self, idx: usize) {
        if let Ok(at) = self.live.binary_search(&idx) {
            self.live.remove(at);
        }
    }

    fn block_or_consume_token(&mut self, idx: usize) {
        let thid = self.threads[idx].thid;
        if self.wake_tokens.remove(&thid) {
            self.threads[idx].state = SState::Runnable;
            self.threads[idx].cooled = false;
        } else {
            self.threads[idx].state = SState::Blocked;
        }
    }

    /// Charge the game clock for the guest work a resume did, with the runnable count taken over
    /// EVERY worker (running threads included).
    fn charge(&mut self, sh: &Shared, idx: usize, fuel: Option<u64>, arm: Option<u64>) {
        let Some(total) = fuel else { return };
        let burned = total.saturating_sub(self.threads[idx].fuel_seen);
        self.threads[idx].fuel_seen = total;
        let retired = arm.map(|t| {
            let d = t.saturating_sub(self.threads[idx].arm_seen);
            self.threads[idx].arm_seen = t;
            d
        });
        if burned == 0 && retired.unwrap_or(0) == 0 {
            return;
        }
        let home = self.threads[idx].home;
        self.load[home] += retired.unwrap_or(burned) as f64;
        let runnable = 1 + self
            .live
            .iter()
            .filter(|&&i| i != idx && matches!(self.threads[i].state, SState::Runnable | SState::Running))
            .count();
        self.fuel_total = self.fuel_total.saturating_add(burned);
        self.fuel_samples += 1;
        self.fuel_max = self.fuel_max.max(burned);
        self.arm_total = self.arm_total.saturating_add(retired.unwrap_or(0));
        let mut host = sh.lock_host();
        host.on_guest_work(runnable, burned, retired);
        publish_words(sh, &host.state);
    }

    /// One display frame ended on thread `idx`.
    fn flip(&mut self, sh: &Shared, idx: usize) {
        self.block_or_consume_token(idx);
        self.frames += 1;
        for l in self.load.iter_mut() {
            *l *= 0.9;
        }
        vitaslop_runtime::sched::set_current_frame(self.frames);
        sh.engine.note_frame(self.frames);
        {
            let mut host = sh.lock_host();
            host.on_frame_boundary(self.frames);
            publish_words(sh, &host.state);
            self.flip_clock.push_back((self.frames, host.clock_us()));
            if self.flip_clock.len() > FLIP_CLOCK_KEEP {
                self.flip_clock.pop_front();
            }
        }
        // The gate just closed: every running thread must reach a switch point before the
        // presenter may read the frame, so ask them now rather than waiting for their fuel.
        if self.frames >= self.gate {
            self.preempt_gate += 1;
            for w in 1..=sh.workers {
                if self.running_on[w].is_some() {
                    sh.preempt_worker(w);
                }
            }
        }
        sh.ring(0);
    }

    fn finish(&mut self, sh: &Shared, idx: usize, end: FiberEnd) {
        let thid = self.threads[idx].thid;
        if self.threads[idx].state == SState::Running {
            self.running -= 1;
            let home = self.threads[idx].home;
            self.running_on[home] = None;
        }
        match end {
            FiberEnd::Returned(code) | FiberEnd::ThreadExit(code) => {
                self.threads[idx].state = SState::Finished(code);
                self.threads[idx].release = true;
                self.retire(idx);
                sh.lock_host().set_thread_exit(thid, code);
            }
            FiberEnd::ProcessHalt(code) => {
                for t in self.threads.iter_mut() {
                    if !matches!(t.state, SState::Finished(_)) {
                        t.state = SState::Finished(code);
                        t.release = true;
                    }
                }
                self.live.clear();
                self.verdict = Some(RunReport::Finished(code));
            }
            FiberEnd::Error(e) => {
                self.threads[idx].state = SState::Finished(0);
                self.verdict = Some(RunReport::Error(format!("thread {thid:#x}: {e}")));
            }
        }
        sh.ring_all();
    }

    /// Fold a finished resume of `idx` back in.
    fn fold(&mut self, sh: &Shared, idx: usize, step: ThreadStep, fuel: Option<u64>, arm: Option<u64>, preempted: bool) {
        self.charge(sh, idx, fuel, arm);
        match step {
            ThreadStep::Finished(end) => self.finish(sh, idx, end),
            ThreadStep::Suspended(stop) => {
                self.running -= 1;
                let home = self.threads[idx].home;
                self.running_on[home] = None;
                match stop {
                    // Taken off the CPU by another worker, not by spinning through its own
                    // quantum: still runnable, and NOT cooled.
                    Stop::Quantum if preempted => self.threads[idx].state = SState::Runnable,
                    Stop::Blocked => self.block_or_consume_token(idx),
                    Stop::Flip => self.flip(sh, idx),
                    Stop::Quantum => {
                        self.threads[idx].state = SState::Runnable;
                        self.threads[idx].cooled = true;
                    }
                }
            }
        }
    }

    /// Start the threads the host spawned and wake the ones it released.
    fn drain(&mut self, sh: &Shared) {
        let (spawns, wakes, stat_writes, masks, rt_woke, timed_out) = {
            let mut host = sh.lock_host();
            let held = self.frames >= self.gate;
            if wall_floor() && host.state.wall_floor_tick(abs_ms(), held) {
                publish_words(sh, &host.state);
            }
            let rt_woke = if host.state.has_wall_parks() {
                host.state.expire_wall_parks((abs_ms() * 1000.0) as u64)
            } else {
                Vec::new()
            };
            let mut words = Words(&sh.engine);
            host.resolve_deferred(&mut words);
            let spawns = host.take_spawns();
            let masks: Vec<(i32, Option<i32>)> =
                spawns.iter().map(|s| (host.state.thread_cpu_affinity(s.thid), host.state.spawn_near(s.thid))).collect();
            let _ = host.state.take_wake_causes();
            let timed_out = host.state.take_timeout_wakes();
            (spawns, host.take_wakes(), host.take_stat_writes(), masks, rt_woke, timed_out)
        };
        for thid in rt_woke {
            if let Some(i) = self.live.iter().copied().find(|&i| self.threads[i].thid == thid) {
                self.threads[i].rt = true;
            }
        }
        for (addr, value) in stat_writes {
            sh.engine.write_guest(addr, &value.to_le_bytes());
        }
        for (sp, (mask, near)) in spawns.into_iter().zip(masks) {
            let near_home = near
                .filter(|_| fiber_near_on())
                .and_then(|r| self.live.iter().map(|&i| &self.threads[i]).find(|t| t.thid == r).map(|t| t.home));
            let home = near_home.unwrap_or_else(|| self.place(sh.workers, mask, sp.priority));
            let birth = Birth { entries: vec![sp.entry], r: [sp.arg_len, sp.arg_ptr, sp.r2, sp.r3], sp: sp.stack_top };
            self.add_thread(sh, sp.thid, sp.priority, home, birth);
        }
        if timeout_yields() {
            self.yield_next.extend(timed_out);
        }
        for thid in wakes {
            let found = self
                .live
                .iter()
                .copied()
                .find(|&i| self.threads[i].thid == thid && self.threads[i].state == SState::Blocked);
            match found {
                Some(i) => {
                    let yields = self.yield_next.remove(&thid);
                    self.threads[i].state = SState::Runnable;
                    self.threads[i].cooled = yields;
                    if !yields {
                        self.preempt_for(sh, i);
                    }
                    sh.ring(self.threads[i].home);
                }
                None => {
                    if self.live.iter().any(|&i| self.threads[i].thid == thid) {
                        self.wake_tokens.insert(thid);
                    }
                }
            }
        }
        self.publish_runnable(sh);
    }

    fn globally_idle(&self) -> bool {
        self.running == 0 && !self.live.iter().any(|&i| self.threads[i].state == SState::Runnable)
    }

    /// Nothing can run anywhere: deliver what is owed, then complete the nearer of the next
    /// transfer and the next timed wait, or report a deadlock.
    fn idle_step(&mut self, sh: &Shared) {
        self.drain(sh);
        if !self.globally_idle() {
            return;
        }
        let blocked: Vec<i32> = self
            .live
            .iter()
            .filter(|&&i| self.threads[i].state == SState::Blocked)
            .map(|&i| self.threads[i].thid)
            .collect();
        if blocked.is_empty() {
            let main = self.threads.first().and_then(|t| match t.state {
                SState::Finished(c) => Some(c),
                _ => None,
            });
            self.verdict = Some(RunReport::Finished(main.unwrap_or(0)));
            sh.ring_all();
            return;
        }
        self.idle_jumps += 1;
        let mut host = sh.lock_host();
        let (deadline, io, now) = (host.earliest_deadline(), host.earliest_io_remaining_us(), host.clock_us());
        if vitaslop_runtime::sched::storage_completes_first(io, deadline, now) && host.release_earliest_io() {
            drop(host);
            self.drain(sh);
            return;
        }
        match deadline {
            Some(t) => {
                host.charge_io_idle(t.saturating_sub(now));
                host.advance_time_to(t);
                publish_words(sh, &host.state);
                drop(host);
                self.drain(sh);
            }
            // Only wall parks left: the guest is WAITING on real time, not stuck.
            None if host.state.has_wall_parks() => {}
            None => {
                let dump = host.sync_dump();
                drop(host);
                tracing::error!(
                    target: "vitaslop::smp",
                    "SMP DEADLOCK: {} thread(s) blocked with no timeout and no pending I/O: {blocked:x?}\n{dump}",
                    blocked.len()
                );
                self.verdict = Some(RunReport::Deadlock(blocked));
                sh.ring_all();
            }
        }
    }
}

/// One guest worker's loop: take a job (a thread of its own to run, or the machine's idle step),
/// run it, fold it back - the browser's `helper_loop`, with a condition variable for its futex.
fn helper_loop(sh: &Arc<Shared>, w: usize) {
    let mut local: std::collections::HashMap<usize, WasmtimeThread> = std::collections::HashMap::new();
    enum Job {
        Run(usize, Option<Birth>, i32, i32, Arc<[AtomicU32; 2]>),
        Again,
        Wait,
    }
    loop {
        if sh.stop.load(Ordering::SeqCst) {
            break;
        }
        let bell = sh.bells[w].seen();
        let (job, releases) = {
            let mut st = sh.state.lock().unwrap();
            let releases: Vec<usize> = local.keys().copied().filter(|&i| st.threads[i].release).collect();
            let gated = st.frames >= st.gate;
            let job = if gated && st.verdict.is_none() && !st.paused && rt_through_gate()
                && let Some(i) = st.pick(w, true)
            {
                let t = &mut st.threads[i];
                Job::Run(i, t.birth.take(), t.thid, t.priority, t.words.clone())
            } else if st.verdict.is_some() || gated || st.paused {
                Job::Wait
            } else if let Some(i) = st.pick(w, false) {
                st.publish_runnable(sh);
                let t = &mut st.threads[i];
                Job::Run(i, t.birth.take(), t.thid, t.priority, t.words.clone())
            } else if st.globally_idle() {
                st.idle_step(sh);
                Job::Again
            } else {
                Job::Wait
            };
            (job, releases)
        };
        for i in releases {
            local.remove(&i);
        }
        match job {
            Job::Again => {}
            Job::Wait => {
                // Woken for the earliest WALL park too: nothing else expires one while the guest
                // is idle and the presenter is presenting. 50 ms is only a backstop - every state
                // change that can give this worker something to do rings it.
                let timeout = match vitaslop_runtime::host::next_wall_park_us() {
                    Some(t) => {
                        let now = (abs_ms() * 1000.0) as u64;
                        Duration::from_micros(t.saturating_sub(now).clamp(100, 50_000))
                    }
                    None => Duration::from_millis(50),
                };
                sh.bells[w].wait(bell, timeout);
                if vitaslop_runtime::host::next_wall_park_us().is_some_and(|t| t <= (abs_ms() * 1000.0) as u64) {
                    sh.state.lock().unwrap().drain(sh);
                }
            }
            Job::Run(i, birth, thid, priority, words) => {
                if let Some(b) = birth {
                    let spawn = SmpSpawn {
                        worker: w,
                        preempt_off: sh.mirror_off.map_or(0, |m| (m + 4 * u64::from(abi::PREEMPT_SLOT_BASE + w as u32)) as u32),
                        preempt_ptr: sh.preempt(w).map_or(0, |p| p as *const AtomicI32 as usize),
                        runnable: sh.runnable.clone(),
                        words: words.clone(),
                    };
                    match sh.engine.instantiate_thread_seq(thid, b.entries, b.r[0], b.r[1], b.r[2], b.r[3], b.sp, priority, Some(spawn)) {
                        Ok(t) => {
                            local.insert(i, t);
                        }
                        Err(e) => {
                            tracing::warn!(target: "vitaslop::smp", "thread {thid:#x} could not start: {e:?}");
                            let mut st = sh.state.lock().unwrap();
                            st.running -= 1;
                            st.running_on[w] = None;
                            st.threads[i].state = SState::Finished(0);
                            st.retire(i);
                            sh.lock_host().set_thread_exit(thid, 0);
                            st.drain(sh);
                            continue;
                        }
                    }
                }
                let Some(mut t) = local.remove(&i) else {
                    let mut st = sh.state.lock().unwrap();
                    st.verdict = Some(RunReport::Error(format!("SMP: thread {thid:#x} was picked on worker {w}, which does not hold it")));
                    sh.ring_all();
                    continue;
                };
                // Make it the host's current thread and hand it this resume's words.
                {
                    let mut host = sh.lock_host();
                    host.set_current_thread(thid);
                    let (cur, tid) = publish_words(sh, &host.state);
                    words[0].store(cur, Ordering::Relaxed);
                    words[1].store(tid, Ordering::Relaxed);
                }
                // A preempt request left over from before this pick was for the thread running
                // then, not for this one.
                if let Some(p) = sh.preempt(w) {
                    p.store(0, Ordering::SeqCst);
                }
                let t0 = Instant::now();
                let step = t.resume();
                let ran = t0.elapsed();
                sh.busy_us[w].fetch_add(ran.as_micros() as u64, Ordering::Relaxed);
                let preempted = sh.preempt(w).is_some_and(|p| p.swap(0, Ordering::SeqCst) != 0)
                    && matches!(step, ThreadStep::Suspended(Stop::Quantum));
                let fuel = t.fuel_used();
                let arm = t.arm_retired();
                let finished = matches!(step, ThreadStep::Finished(_));
                {
                    let mut st = sh.state.lock().unwrap();
                    st.resumes += 1;
                    st.threads[i].life_run_ms += ran.as_secs_f64() * 1000.0;
                    st.fold(sh, i, step, fuel, arm, preempted);
                    st.drain(sh);
                    if st.paused && st.running == 0 {
                        sh.ring(0);
                    }
                }
                if !finished {
                    local.insert(i, t);
                }
            }
        }
    }
}

/// The newest parallel run's number. The runtime's resolver and wall-park switches are process
/// globals, and a run REPLACED by another (`sceAppMgrLoadExec`) is dropped AFTER its successor
/// started - MEASURED: Uncharted's executable ran with the resolver detached and every texture
/// set proven inline on the drawing thread, because its launcher's drop cleared the switches the
/// new run had just set. A run clears them only while it is still the newest.
static RUN_GEN: AtomicU64 = AtomicU64::new(0);

/// The presenter's handle on a parallel run.
pub struct SmpRun {
    sh: Arc<Shared>,
    joins: Vec<std::thread::JoinHandle<()>>,
    /// The resolver thread under [`async_resolve`].
    resolver: Option<std::thread::JoinHandle<()>>,
    /// This run's number in [`RUN_GEN`] - see there.
    generation: u64,
    /// The frame taken for presenting. Under [`overlap`] the guest may already be a frame past
    /// it, and everything the caller keys on a frame number - its next target above all - means
    /// the frame being PRESENTED, which is this. MEASURED without it: each `run_frames` asked for
    /// one past the guest's own count, moved the gate two frames, and an MLB window ran at
    /// 93.9 fps with its game clock at 160% of the wall.
    consumed: u64,
}

impl SmpRun {
    /// Stand up the workers over `engine` and bind the main thread (which runs `entries` in
    /// sequence - the libraries' constructors, then the executable's) by the title's own request.
    pub(crate) fn start(
        engine: WasmtimeEngine<VitaEnv>,
        entries: Vec<u32>,
        args: [u32; 4],
        main_sp: u32,
    ) -> Result<SmpRun, crate::RunError> {
        let workers = worker_count();
        let generation = RUN_GEN.fetch_add(1, Ordering::SeqCst) + 1;
        let host = engine.host_handle();
        let mirror_off = engine.mirror_off();
        let mirror = mirror_off.and_then(|off| engine.linear_address(off));
        if mirror.is_none() {
            return Err(crate::RunError::Wasm("a parallel run needs the module's mirror block (preempt words), and this module has none".into()));
        }
        // The runtime measures wall parks and the wall floor on this clock, and this engine serves
        // the parks (an idle worker wakes for the earliest one).
        vitaslop_runtime::host::set_host_wall_clock(abs_ms);
        vitaslop_runtime::host::set_wall_parks_served(true);
        // Guest threads run while the presenter writes render targets back: check the whole
        // region, not its first row (see the function) - as the browser's parallel run does.
        vitaslop_runtime::rtt_writeback::set_whole_region_guard(true);
        if let Some(base) = vitaslop_runtime::sched::GuestEngine::mirror_base(&engine) {
            host.lock().unwrap_or_else(|e| e.into_inner()).set_mirror_base(base);
        }
        let sh = Arc::new(Shared {
            host,
            engine: Arc::new(engine),
            workers,
            state: Mutex::new(State::new(workers)),
            bells: (0..=workers).map(|_| Bell::new()).collect(),
            runnable: (0..=workers).map(|_| AtomicUsize::new(0)).collect(),
            stop: AtomicBool::new(false),
            mirror,
            mirror_off,
            busy_us: (0..=workers).map(|_| AtomicU64::new(0)).collect(),
            failed: OnceLock::new(),
            late: LateState::default(),
        });
        if late_present() {
            sh.lock_host().state.capture.untaken_frame_limit = Some(LATE_KEEP_FRAMES);
        }
        {
            // The title's own request for its main thread (`SceProcessParam`): priority and the
            // core it is pinned to, so the threads it creates with mask 0 - which inherit that pin
            // - land beside it.
            let (main_prio, main_mask) = {
                let h = sh.lock_host();
                (h.state.main_thread_priority(), h.state.main_affinity())
            };
            let mut st = sh.state.lock().unwrap();
            let home = st.place(workers, main_mask, main_prio);
            st.add_thread(
                &sh,
                vitaslop_runtime::host::MAIN_THID,
                main_prio,
                home,
                Birth { entries: entries.iter().map(|e| e & !1).collect(), r: args, sp: main_sp },
            );
            st.publish_runnable(&sh);
        }
        // >>> THE RESOLVER, under the async resolve: a flip only MOVES its frame's draw
        // descriptions into a job, this thread reads the geometry and proves the texture sets
        // over the shared region, and the presenter applies the result (`resolve_flipped`). It
        // is attached and started BEFORE any guest worker: the runtime decides once, at the
        // first draw, whether draws defer their texture sets to it (`defer_textures`), and
        // MEASURED with it started after the workers, that first draw had already happened and
        // every set was still proven inline (wun8: unchanged 39 ms of `sceGxmDraw` a frame). A
        // job queued before its loop starts waits in the queue.
        let resolver = if async_resolve() {
            vitaslop_runtime::host::set_async_flip_resolve(true);
            vitaslop_runtime::host::set_resolver_attached(true);
            let sh = sh.clone();
            let j = std::thread::Builder::new()
                .name("vitaslop-resolver".into())
                .spawn(move || {
                    let handles = sh.lock_host().resolver_handles();
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        sh.engine.with_guest_view(|mem, base| vitaslop_runtime::host::run_resolver(handles, mem, base))
                    }));
                    if let Err(p) = r {
                        vitaslop_runtime::host::set_resolver_attached(false);
                        let why = p
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "a panic".into());
                        let _ = sh.failed.set(format!("SMP resolver died: {why}"));
                        sh.ring_all();
                    }
                })
                .map_err(|e| crate::RunError::Wasm(format!("spawn SMP resolver: {e}")))?;
            Some(j)
        } else {
            None
        };
        let mut joins = Vec::new();
        for w in 1..=workers {
            let sh = sh.clone();
            let j = std::thread::Builder::new()
                .name(format!("vitaslop-core{w}"))
                // A guest thread's wasm stack lives on its fiber; this is only the worker's own.
                .stack_size(8 << 20)
                .spawn(move || {
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| helper_loop(&sh, w)));
                    if let Err(p) = r {
                        let why = p
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "a panic".into());
                        let _ = sh.failed.set(format!("SMP worker {w} died: {why}"));
                        sh.ring_all();
                    }
                })
                .map_err(|e| crate::RunError::Wasm(format!("spawn SMP worker {w}: {e}")))?;
            joins.push(j);
        }
        if late_present() {
            let sh = sh.clone();
            let j = std::thread::Builder::new()
                .name("vitaslop-late-present".into())
                .spawn(move || late_loop(&sh))
                .map_err(|e| crate::RunError::Wasm(format!("spawn SMP late-present helper: {e}")))?;
            joins.push(j);
        }
        tracing::info!(target: "vitaslop::status", "SMP run: {workers} guest worker(s) - this run is NOT deterministic");
        Ok(SmpRun { sh, joins, resolver, generation, consumed: 0 })
    }

    pub fn host(&self) -> MutexGuard<'_, VitaEnv> {
        self.sh.lock_host()
    }

    pub fn engine(&self) -> &WasmtimeEngine<VitaEnv> {
        &self.sh.engine
    }

    /// The frame the caller is at: the last one taken for presenting under [`overlap`], else the
    /// last one flipped (the same thing, since nothing runs past it there). The browser's rule.
    pub fn frames(&self) -> u64 {
        if overlap() {
            self.consumed
        } else {
            self.sh.state.lock().unwrap().frames
        }
    }

    /// The game time frame `frame` took: the clock at its flip minus the clock at the flip
    /// before it. `None` when either is no longer remembered. The browser's rule.
    pub fn frame_advance_us(&self, frame: u64) -> Option<u64> {
        self.frame_span_us(frame.checked_sub(1)?, frame)
    }

    /// The game time from frame `from`'s flip to frame `to`'s - several frames when a late
    /// present's catch-up takes them at once. `None` when either is no longer remembered.
    pub fn frame_span_us(&self, from: u64, to: u64) -> Option<u64> {
        let st = self.sh.state.lock().unwrap();
        let at = |f: u64| st.flip_clock.iter().find(|(n, _)| *n == f).map(|(_, us)| *us);
        Some(at(to)?.saturating_sub(at(from)?))
    }

    /// The presenter's handle for [`late_present`]; `None` when it is off.
    pub fn late_handle(&self) -> Option<LatePresent> {
        late_present().then(|| LatePresent { sh: self.sh.clone() })
    }

    /// Gate openings the late-present helper made - frames the guest ran during slow presents.
    pub fn late_opened(&self) -> u64 {
        self.sh.late.opened.load(Ordering::Relaxed)
    }

    pub fn resumes(&self) -> u64 {
        self.sh.state.lock().unwrap().resumes
    }

    pub fn census(&self) -> (usize, usize) {
        let st = self.sh.state.lock().unwrap();
        let live = st.live.len();
        (live, st.threads.len() - live)
    }

    pub fn fuel_report(&self) -> (u64, u64, u64) {
        let st = self.sh.state.lock().unwrap();
        (st.fuel_total, st.fuel_samples, st.fuel_max)
    }

    pub fn arm_report(&self) -> u64 {
        self.sh.state.lock().unwrap().arm_total
    }

    /// Per-worker busy time, threads and preemptions - one line for the end-of-run report.
    pub fn report(&self) -> String {
        let st = self.sh.state.lock().unwrap();
        let busy: Vec<String> = (1..=self.sh.workers)
            .map(|w| {
                let on = st.live.iter().filter(|&&i| st.threads[i].home == w).count();
                format!("w{w} {:.1} s busy, {on} live thread(s)", self.sh.busy_us[w].load(Ordering::Relaxed) as f64 / 1e6)
            })
            .collect();
        let (calls, waited) = crate::threaded::HOST_CALL_WAIT.read();
        format!(
            "SMP {} worker(s): {} | {} resumes, {} idle jumps, preempts gate {} prio {} | {} host calls waited {:.1} s for the host | {}",
            self.sh.workers,
            busy.join("; "),
            st.resumes,
            st.idle_jumps,
            st.preempt_gate,
            st.preempt_prio,
            calls,
            waited,
            vitaslop_runtime::host::deferral_report()
        )
    }

    /// Stop every guest worker at a switch point - for the presenter's write into guest memory
    /// the guest may also be writing (a render target's write-back). Returns once none is
    /// running; [`Self::resume_guest`] lets them go again. The browser's `pause_guest`.
    pub fn pause_guest(&mut self) {
        {
            let mut st = self.sh.state.lock().unwrap();
            st.paused = true;
            for w in 1..=self.sh.workers {
                if st.running_on[w].is_some() {
                    self.sh.preempt_worker(w);
                }
            }
        }
        loop {
            let bell = self.sh.bells[0].seen();
            if self.sh.state.lock().unwrap().running == 0 || self.sh.failed.get().is_some() {
                return;
            }
            self.sh.bells[0].wait(bell, Duration::from_millis(4));
        }
    }

    pub fn resume_guest(&mut self) {
        self.sh.state.lock().unwrap().paused = false;
        self.sh.ring_all();
    }

    /// The presenter's half of the async resolve: wait for the resolver's reads of every flip so
    /// far and apply them, before the flipped frame's scenes are taken. The browser's
    /// `resolve_flipped_geometry`: a boundary the flip left (no resolver attached) is read here,
    /// without the host lock; taking and applying are under it, and both are cheap.
    fn resolve_flipped(&self) {
        let (taken, post) = {
            let mut h = self.sh.lock_host();
            (h.take_flip_resolve_job(), h.async_resolve_handle())
        };
        if let Some(job) = taken {
            self.sh.engine.with_guest_view(|mem, base| vitaslop_runtime::host::compute_flip_resolve_job(job, mem, base));
        }
        post.wait_idle();
        self.sh.engine.with_guest_view(|mem, base| self.sh.lock_host().apply_posted_resolve(mem, base));
    }

    /// Run the guest to `target` display frames: open the gate, take the idle clock steps, and
    /// return once the target frame has flipped - and, unless [`overlap`]ped, once every worker
    /// has reached a switch point. Overlapped, the gate stands one frame AHEAD of the target:
    /// the guest builds the next frame while this one is presented, and stops (preempted at
    /// that flip) only if it gets a whole frame ahead.
    pub fn run_frames(&mut self, target: u64) -> RunReport {
        let ahead = u64::from(overlap());
        {
            let mut st = self.sh.state.lock().unwrap();
            // The hold ends here: close it off on the wall floor so its first tick after the gate
            // opens does not count the hold (see `VitaState::wall_floor_tick`).
            if st.frames >= st.gate && wall_floor() {
                self.sh.lock_host().state.wall_floor_tick(abs_ms(), true);
            }
            st.gate = st.gate.max(target.saturating_add(ahead));
        }
        self.sh.ring_all();
        loop {
            let bell = self.sh.bells[0].seen();
            if let Some(why) = self.sh.failed.get() {
                return RunReport::Error(why.clone());
            }
            {
                let mut st = self.sh.state.lock().unwrap();
                if let Some(v) = st.verdict.clone() {
                    return v;
                }
                if ahead == 1 && st.frames >= target {
                    // A late present let the guest run on: take the NEWEST flipped frame (the
                    // caller takes its scenes and those before it together) and stand the gate
                    // one frame past it, as always.
                    let newest = st.frames.max(target);
                    self.consumed = newest;
                    st.consumed = newest;
                    if newest > target {
                        st.gate = st.gate.max(newest + 1);
                        self.sh.ring_all();
                    }
                    drop(st);
                    if async_resolve() {
                        self.resolve_flipped();
                    }
                    return RunReport::FramesReached(self.consumed);
                }
                if st.frames >= target && st.running == 0 {
                    return RunReport::FramesReached(st.frames);
                }
                if st.frames < st.gate && st.globally_idle() {
                    st.idle_step(&self.sh);
                    continue;
                }
            }
            self.sh.bells[0].wait(bell, Duration::from_millis(4));
        }
    }
}

impl Drop for SmpRun {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::SeqCst);
        self.sh.ring_all();
        for j in self.joins.drain(..) {
            let _ = j.join();
        }
        let newest = RUN_GEN.load(Ordering::SeqCst) == self.generation;
        if let Some(j) = self.resolver.take() {
            if newest {
                vitaslop_runtime::host::set_resolver_attached(false);
            }
            self.sh.lock_host().async_resolve_handle().stop();
            let _ = j.join();
            if newest {
                vitaslop_runtime::host::set_async_flip_resolve(false);
            }
        }
        if newest {
            vitaslop_runtime::host::set_wall_parks_served(false);
        }
    }
}

impl crate::threaded::SmpOps<VitaEnv> for SmpRun {
    fn host(&self) -> MutexGuard<'_, VitaEnv> {
        self.sh.lock_host()
    }
    fn engine_read_guest(&self, addr: u32, len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        if self.sh.engine.read_guest_into(addr, &mut out) { out } else { Vec::new() }
    }
    fn engine_read_guest_into(&self, addr: u32, buf: &mut [u8]) -> bool {
        self.sh.engine.read_guest_into(addr, buf)
    }
    fn engine_write_guest(&self, addr: u32, bytes: &[u8]) {
        self.sh.engine.write_guest(addr, bytes)
    }
    fn guest_region(&self) -> (u32, usize) {
        self.sh.engine.region()
    }
    fn with_host_words_dyn(&mut self, f: &mut dyn FnMut(&mut VitaEnv, &mut dyn vitaslop_runtime::host::GuestWords)) {
        let mut host = self.sh.lock_host();
        let mut words = Words(&self.sh.engine);
        f(&mut host, &mut words);
    }
    fn frames(&self) -> u64 {
        SmpRun::frames(self)
    }
    fn resumes(&self) -> u64 {
        SmpRun::resumes(self)
    }
    fn census(&self) -> (usize, usize) {
        SmpRun::census(self)
    }
    fn fuel_report(&self) -> (u64, u64, u64) {
        SmpRun::fuel_report(self)
    }
    fn arm_report(&self) -> u64 {
        SmpRun::arm_report(self)
    }
    fn report(&self) -> String {
        SmpRun::report(self)
    }
    fn run_frames(&mut self, target: u64) -> RunReport {
        SmpRun::run_frames(self, target)
    }
    fn pause_guest(&mut self) {
        SmpRun::pause_guest(self)
    }
    fn frame_advance_us(&self, frame: u64) -> Option<u64> {
        SmpRun::frame_advance_us(self, frame)
    }
    fn frame_span_us(&self, from: u64, to: u64) -> Option<u64> {
        SmpRun::frame_span_us(self, from, to)
    }
    fn late_handle(&self) -> Option<LatePresent> {
        SmpRun::late_handle(self)
    }
    fn late_opened(&self) -> u64 {
        SmpRun::late_opened(self)
    }
    fn resume_guest(&mut self) {
        SmpRun::resume_guest(self)
    }
}
