//! >>> PARALLEL GUEST THREADS (`VITASLOP_SMP=1`): the guest's threads run AT ONCE on several
//! Web Workers over one shared memory, instead of one at a time on the run worker.
//!
//! OFF by default, and off is not a degraded mode - it is the DETERMINISTIC engine: one guest
//! thread at a time, switched only at host calls, so the same inputs replay the same run. Every
//! recipe, capsule and bisection depends on that, and a parallel run cannot give it (which
//! thread reaches a lock first depends on the host's timing). So this is a run-time choice, and
//! the single-worker path does not change when it exists.
//!
//! # The shape
//! - **The run worker (W0)** keeps everything that is JavaScript or GPU: the `VitaEnv` it built,
//!   WebGPU and the present, the storage reader, WebCodecs, the event loop. It runs NO guest
//!   thread, so it is always free to answer the others.
//! - **Guest workers (W1..Wn)**, one per Vita core by default, each instantiate this same bundle
//!   over the same shared linear memory - so the guest region, the host and this module's
//!   statics are the same objects everywhere - and run guest threads on their own JSPI stacks.
//! - A guest thread is BOUND to one worker when it first runs. Its suspended stack is a JS
//!   promise in that worker's heap and cannot move. The binding follows the thread's CPU
//!   affinity mask where it names cores (the Vita's are bits 16..18), else the least-loaded
//!   worker.
//! - A host call runs on the worker that made it, under the host mutex - GXM recording, the
//!   kernel's sync objects and the clock are plain Rust state. The calls whose handlers need W0's
//!   JavaScript (`vita::smp_forwarded` - the video and audio DECODERS, location, and the rarer
//!   services) are FORWARDED: the thread suspends, W0 dispatches the call, and the registers come
//!   back to be written before the stack resumes. Audio OUTPUT and FILE READS are not: each guest
//!   worker holds its own JS view of the audio ring and of the title's storage ring, both
//!   SharedArrayBuffers. A forward waits for whatever W0 is doing - usually a present - so a
//!   per-frame one serialises the whole frame (see `smp_forwarded`).
//! - **The frame GATE.** Guest threads run only while the live loop wants a frame
//!   (`frames < gate`); the flip that reaches it closes the gate, and W0 presents once every
//!   worker has reached a switch point. The present reads guest memory (deferred geometry,
//!   textures, render-target write-back), and a guest still running would be writing it.
//!
//! # What is the same policy as the one-baton scheduler
//! Strict priority with round-robin inside a level (per worker - each is a core), the spin
//! cooldown, the wake token for a wake that races ahead of its block, the idle clock jump taken
//! only when NOTHING can run anywhere, storage/deadline ordering, the frame boundary and its
//! input step. They are restated here rather than shared with `SchedCore`, because that type
//! owns its threads and this one cannot: a thread's JavaScript lives on one worker and its
//! scheduling state is shared by all of them.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use js_sys::{Object, Reflect, WebAssembly};
use vitaslop_runtime::sched::{FiberEnd, GuestEngine, RunReport, Stop, ThreadHandle, ThreadStep};
use vitaslop_runtime::{ImportDispatch, Reentry, SvcOutcome, VitaEnv};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::browser_sched::{
    self, BrowserEngine, BrowserThread, EarlyCompleter, ForwardReq, Host, RegPatch, SmpHooks,
};

/// Whether this run asked for parallel guest threads. Read once: the transpile and the
/// scheduler must agree for the whole run, and a knob flipped between them would hand an SMP
/// scheduler a one-baton module (or the reverse).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| vitaslop_runtime::knobs::flag("VITASLOP_SMP"))
}

/// Present frame N while the guest workers build N+1 - the gate runs ONE frame ahead of the
/// present, and the run worker takes only the flipped frame's scenes
/// (`Capture::take_scenes_through_flip`). ON by default under SMP; `VITASLOP_SMP_OVERLAP=0`
/// makes every present wait for every worker to reach a switch point first (stop-the-world).
/// See `SmpRun::run_frames`.
///
/// # Why it is the default
/// MEASURED (`pf25b`, desktop Chrome, CPU ms of run-worker wall per frame): Madden one worker
/// 13.9-16.1, SMP stop-the-world 11.6-13.0, overlapped ~0 (the run worker slept 27 of a 32.7 ms
/// period); MLB one worker 8.5-9.0, SMP stop-the-world 9.7-10.7 (SLOWER), overlapped 0.0-0.2
/// (slept 12.5 of 17.1). Sweep `sw25o`: all 8 gameplay recipes reach their end frame with
/// correct pictures and DOA5's @assert passes. It is safe because the present reads no live
/// guest memory: geometry is snapshotted at the guest's GPU wait or flip, textures per scene.
pub fn overlap() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(vitaslop_runtime::knobs::var("VITASLOP_SMP_OVERLAP").as_deref().map(str::trim), Ok("0")))
}

/// A sync point's resolve-only park is settled on the guest's own worker - see
/// `BrowserEngine::settle_sync_resolve`. `VITASLOP_SYNC_RESOLVE_ON_WORKER=0` is the arm back.
/// See `Slot::rt`.
fn rt_through_gate() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| vitaslop_runtime::knobs::var("VITASLOP_SMP_RT_THROUGH_GATE").as_deref().map(str::trim) != Ok("0"))
}

fn sync_resolve_on_worker() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| vitaslop_runtime::knobs::var("VITASLOP_SYNC_RESOLVE_ON_WORKER").as_deref().map(str::trim) != Ok("0"))
}

/// Pause every guest worker for a small-target completion's whole render + readback
/// (`VITASLOP_SMP_EARLY_PAUSE_ALL=1`), or - the DEFAULT since 27c - only for its write into guest
/// memory. On hardware the GPU renders and writes that target while every other core keeps
/// running; pausing them all froze the audio thread for 30-60 ms per batch. MEASURED on the
/// phone (MLB pitches): 027 vs 026, speed 79.7% vs 73.3%, audio underrun 21% vs 28%.
fn early_pause_all() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| vitaslop_runtime::knobs::var("VITASLOP_SMP_EARLY_PAUSE_ALL").as_deref().map(str::trim) == Ok("1"))
}

/// The game clock is floored at the wall - see `VitaState::wall_floor_tick`. ON by default under
/// SMP; `VITASLOP_CLOCK_WALL_FLOOR=0` is the arm back.
///
/// Not below `VITASLOP_BROWSER_FASTFORWARD`: a fast-forward is not real time by definition, and
/// a frame-numbered recipe written against the unfloored clock must land where it was written
/// on a device slower than real time too (the phone's MLB menu walk ended in another scene).
fn wall_floor(frame: u64) -> bool {
    static ON: OnceLock<(bool, u64)> = OnceLock::new();
    let (on, from) = *ON.get_or_init(|| {
        let on = vitaslop_runtime::knobs::var("VITASLOP_CLOCK_WALL_FLOOR").as_deref().map(str::trim) != Ok("0");
        let from = vitaslop_runtime::knobs::var("VITASLOP_BROWSER_FASTFORWARD")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0);
        (on, from)
    });
    on && frame >= from
}

/// The flip's geometry resolve runs on a RESOLVER worker of its own - see `smp_resolver_main`
/// and `VitaState::resolve_at_flip`. ON by default under SMP (overlapped); the phone proxy
/// measured it at -23% per guest frame (px27a 44-47 ms -> px27c 34.6 ms), because the read left
/// the game's serial main -> render chain. `VITASLOP_SMP_ASYNC_RESOLVE=0` is the arm back.
/// `VITASLOP_GUEST_PROF=<from frame>:<frames>` (SMP only): WHICH GUEST CODE each worker runs,
/// measured on the device. The module stores the executing guest function's address into a
/// per-worker slot (`abi::SMP_PROF_SLOT_OFFSET`), each worker zeroes it when a resume ends, and a
/// SAMPLER worker reads the slots every ~0.25 ms over the window and prints `guestprof` lines:
/// per worker, the share of wall time in each guest function (0 = not running guest code).
///
/// The V8 profiler is unavailable in Chrome's workers, so without this the only function-level
/// view of guest code is a desktop proxy - and the phone is 4-5x slower in ways that do not
/// scale uniformly.
pub(crate) fn guest_prof() -> Option<(u64, u64)> {
    static SPEC: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *SPEC.get_or_init(|| {
        let v = vitaslop_runtime::knobs::var("VITASLOP_GUEST_PROF").ok()?;
        let (a, b) = v.trim().split_once(':')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    })
}

/// Worker `w`'s guest-function slot - see [`guest_prof`] and `abi::SMP_PROF_SLOT_OFFSET`.
fn prof_slot(sh: &Shared, w: usize) -> Option<&AtomicI32> {
    use vitaslop_transpiler::abi::{PREEMPT_SLOT_BASE, SMP_PROF_SLOT_OFFSET, SMP_PROF_SLOT_STRIDE};
    let g = sh.geometry;
    let slot = PREEMPT_SLOT_BASE + SMP_PROF_SLOT_OFFSET + SMP_PROF_SLOT_STRIDE * w as u32;
    let addr = u64::from(g.host_off) + g.mirror_off? + 4 * u64::from(slot);
    // SAFETY: inside the reserved mirror block (checked against its size by the ABI constants);
    // aligned, only ever accessed as an i32.
    Some(unsafe { &*(addr as usize as *const AtomicI32) })
}

/// The tag a worker's slot carries while one of its threads is in a HOST CALL: the selector
/// under this marker, so the report can name our own time per call apart from the guest's.
pub(crate) const PROF_HOST_TAG: u32 = 0xf000_0000;

/// Tag worker `w`'s slot with host-call `selector`, returning what it held - see
/// [`PROF_HOST_TAG`]. `None` when not profiling.
pub(crate) fn prof_host_enter(w: usize, selector: u32) -> Option<i32> {
    guest_prof()?;
    let s = prof_slot(SHARED.get()?, w)?;
    Some(s.swap((PROF_HOST_TAG | selector) as i32, Ordering::Relaxed))
}

/// Undo [`prof_host_enter`].
pub(crate) fn prof_host_exit(w: usize, prev: Option<i32>) {
    if let (Some(prev), Some(sh)) = (prev, SHARED.get()) {
        if let Some(s) = prof_slot(sh, w) {
            s.store(prev, Ordering::Relaxed);
        }
    }
}

/// Entry point of the SAMPLER worker (`role: "sampler"`) - see [`guest_prof`]. Returns the
/// report once the window has passed (or the run stops).
#[wasm_bindgen]
pub fn smp_sampler_main() -> Result<String, JsValue> {
    crate::logging::install_panic_hook();
    let sh = SHARED.get().cloned().ok_or_else(|| JsValue::from_str("sampler started before the run was set up"))?;
    let Some((from, frames)) = guest_prof() else { return Ok(String::new()) };
    let workers = sh.workers;
    let mut counts: Vec<HashMap<u32, u64>> = vec![HashMap::new(); workers + 1];
    let mut total = 0u64;
    let t0 = abs_ms();
    loop {
        if sh.stop.load(Ordering::SeqCst) {
            break;
        }
        let f = vitaslop_runtime::sched::current_frame();
        if f >= from + frames {
            break;
        }
        if f >= from {
            for w in 1..=workers {
                if let Some(s) = prof_slot(&sh, w) {
                    *counts[w].entry(s.load(Ordering::Relaxed) as u32).or_default() += 1;
                }
            }
            total += 1;
        }
        // A timed wait on a word nobody rings: ~0.25 ms, without spinning a core.
        let quiet = AtomicI32::new(0);
        bell_wait(&quiet, 0, 250_000);
    }
    let mut out = Vec::new();
    out.push(format!("guestprof window f{from}+{frames}: {total} samples over {:.0} ms", abs_ms() - t0));
    let host = lock_host(&sh.host);
    let label = |a: u32| -> String {
        if a & PROF_HOST_TAG == PROF_HOST_TAG {
            match host.import_at(a & !PROF_HOST_TAG) {
                Some((_, nid)) => format!("host:{}", vitaslop_runtime::nid::name(nid)),
                None => format!("host:sel{}", a & !PROF_HOST_TAG),
            }
        } else if a == 0 {
            "idle".to_string()
        } else {
            format!("{a:#010x}")
        }
    };
    for w in 1..=workers {
        let mut v: Vec<(u32, u64)> = counts[w].iter().map(|(a, n)| (*a, *n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        let host_share: u64 = v.iter().filter(|(a, _)| a & PROF_HOST_TAG == PROF_HOST_TAG).map(|(_, n)| n).sum();
        let body: Vec<String> = std::iter::once(format!("ALL-HOST={:.2}", 100.0 * host_share as f64 / total.max(1) as f64))
            .chain(v.iter().take(60).map(|(a, n)| format!("{}={:.2}", label(*a), 100.0 * *n as f64 / total.max(1) as f64)))
            .collect();
        out.push(format!("guestprof w{w} {}", body.join(" ")));
    }
    Ok(out.join("\n"))
}

fn async_resolve() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        overlap() && vitaslop_runtime::knobs::var("VITASLOP_SMP_ASYNC_RESOLVE").as_deref().map(str::trim) != Ok("0")
    })
}

/// `VITASLOP_SMP_GUEST_SLOW`: the extra multiple of each guest slice to spin - see its use in
/// `helper_loop`. 0 (unset) = off.
fn guest_slow() -> f64 {
    static V: OnceLock<f64> = OnceLock::new();
    *V.get_or_init(|| {
        vitaslop_runtime::knobs::var("VITASLOP_SMP_GUEST_SLOW").ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0.0)
    })
}

/// How many guest workers a parallel run uses: `VITASLOP_SMP_WORKERS`, default 3 - the Vita
/// gives a title three cores, and its threads' affinity masks name them.
pub fn worker_count() -> usize {
    vitaslop_runtime::knobs::var("VITASLOP_SMP_WORKERS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| (1..=8).contains(&n))
        .unwrap_or(3)
}

// ---------------------------------------------------------------------------------------
// The host lock, counted.
// ---------------------------------------------------------------------------------------

/// Host-lock acquisitions and how many of them found the lock HELD. On the one-baton engine
/// the second number is zero by construction; under SMP it is the one figure that says
/// whether the big host lock is what the workers are queueing on.
static LOCK_TAKEN: AtomicU64 = AtomicU64::new(0);
static LOCK_CONTENDED: AtomicU64 = AtomicU64::new(0);

/// Lock the host, counting contention. A `try_lock` first costs one compare-and-swap either
/// way, and it is what lets the panel say how often a worker waited here.
pub(crate) fn lock_host(host: &Host) -> MutexGuard<'_, VitaEnv> {
    // Counted per thread and flushed in batches of 64: every worker takes this lock on every
    // host call, and one shared counter bumped from all of them bounces a cache line between
    // cores. The panel's share lags by under a batch per worker.
    thread_local! {
        static TAKEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }
    TAKEN.with(|t| {
        let n = t.get() + 1;
        t.set(n);
        if n.is_multiple_of(64) {
            LOCK_TAKEN.fetch_add(64, Ordering::Relaxed);
        }
    });
    match host.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::WouldBlock) => {
            LOCK_CONTENDED.fetch_add(1, Ordering::Relaxed);
            let t0 = now_ms();
            let g = host.lock().unwrap();
            // Blocked time, so `VITASLOP_SMP_GUEST_SLOW` scales guest WORK and not waiting.
            LOCK_WAIT_MS.with(|w| w.set(w.get() + (now_ms() - t0)));
            g
        }
        Err(std::sync::TryLockError::Poisoned(p)) => {
            panic!("the host lock is poisoned - a host call panicked on another worker: {p}")
        }
    }
}

// >>> WHAT EACH THREAD CALLS, AND HOW OFTEN IT RUNS OUT OF FUEL - per worker, merged into
// `Shared::tallies` every so often. The one-baton engine's per-NID panel is fed from
// thread-locals of the one worker that ran everything; under SMP that worker runs nothing, so
// without this the question "what is this spinning thread doing" has no answer at all.
thread_local! {
    /// This worker's time spent blocked on the host lock (see `lock_host`).
    static LOCK_WAIT_MS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    // FxHash, not SipHash: `CALLS` is touched on EVERY host call.
    static CALLS: RefCell<vitaslop_platform::fasthash::FxHashMap<(i32, u32), u64>> = RefCell::new(Default::default());
    static FUEL: RefCell<vitaslop_platform::fasthash::FxHashMap<i32, u64>> = RefCell::new(Default::default());
}

/// Count one host call by `thid` (SMP workers only - see [`CALLS`]).
pub(crate) fn note_call(thid: i32, selector: u32) {
    CALLS.with(|c| *c.borrow_mut().entry((thid, selector)).or_default() += 1);
}

/// Count one software-fuel yield by `thid` - a thread that ran a whole quantum of guest loop
/// without a host call ending it.
pub(crate) fn note_fuel(thid: i32) {
    FUEL.with(|c| *c.borrow_mut().entry(thid).or_default() += 1);
}

/// Merge this worker's tallies into the shared ones.
fn publish_tallies(sh: &Shared) {
    let calls = CALLS.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let fuel = FUEL.with(|c| std::mem::take(&mut *c.borrow_mut()));
    let mut t = sh.tallies.lock().unwrap();
    for (k, v) in calls {
        *t.0.entry(k).or_default() += v;
    }
    for (k, v) in fuel {
        *t.1.entry(k).or_default() += v;
    }
}

/// The SMP run of this process, once W0 has stood it up. A static because the guest workers
/// reach it through the shared memory every worker's instance of this bundle sees.
static SHARED: OnceLock<Arc<Shared>> = OnceLock::new();

/// A guest worker that DIED (a panic, an uncaught error). Set without taking any lock -
/// under `panic = "abort"` a worker can die holding the state or host lock, and the run
/// worker must still be able to end the run and say why rather than wait on it for ever.
static FAILED: OnceLock<String> = OnceLock::new();

/// How many threads OTHER than the caller's own could run on worker `w` right now - what a
/// `sceKernelDelayThread(0)` asks. Zero outside an SMP run.
pub(crate) fn runnable_others(w: usize) -> usize {
    SHARED
        .get()
        .and_then(|s| s.runnable.get(w))
        .map_or(0, |n| n.load(Ordering::Relaxed))
}

// ---------------------------------------------------------------------------------------
// Shared state.
// ---------------------------------------------------------------------------------------

/// Everything the workers share. Lives in shared memory; reached through [`SHARED`].
pub(crate) struct Shared {
    host: Host,
    /// Guest workers (W1..=workers). W0 is the run worker and hosts no guest thread.
    workers: usize,
    state: Mutex<State>,
    /// One doorbell word per worker, index 0 = W0. Rung by incrementing and notifying; a
    /// worker reads it BEFORE deciding it has nothing to do and waits on that value, so a ring
    /// that lands in between is never lost.
    bells: Box<[AtomicI32]>,
    /// When each bell was last rung, as `abs_ms` f64 bits: a sleeping worker's wait is judged
    /// by when it was RUNG, not when it woke - see `helper_loop`'s spin history.
    rung_at: Box<[AtomicU64]>,
    /// Runnable threads per worker, kept current under the state lock (see [`runnable_others`]).
    runnable: Box<[AtomicUsize]>,
    stop: AtomicBool,
    stats: Box<[WorkerStats]>,
    geometry: Geometry,
    owner_only: Vec<bool>,
    /// By SELECTOR: which `owner_only` calls are routed per CALL (`vita::smp_forward_per_call`).
    per_call: Vec<bool>,
    /// `(host calls by (thid, selector), fuel yields by thid)` since the last panel line.
    tallies: Mutex<(HashMap<(i32, u32), u64>, HashMap<i32, u64>)>,
}

/// Per-worker wall-clock accounting, in microseconds of that worker's own clock (each worker's
/// `performance.now()` has its own origin, so only DURATIONS cross workers).
#[derive(Default)]
struct WorkerStats {
    busy_us: AtomicU64,
    wait_us: AtomicU64,
    resumes: AtomicU64,
    /// Resumes that ended by burning a whole quantum (fuel or a yield) rather than blocking:
    /// a thread SPINNING on this worker, which on one core would have been descheduled.
    quanta: AtomicU64,
    /// Waits that polled before sleeping ([`spin_cap_us`]), and how many of those the poll
    /// CAUGHT (the bell rang inside it, so no futex sleep and no wake-up latency).
    spins: AtomicU64,
    spin_caught: AtomicU64,
}

/// `VITASLOP_SMP_SPIN_US`: how long a guest worker with nothing to run polls its doorbell
/// before sleeping on it. A futex sleep costs a wake-up latency every time another worker
/// hands this one a thread; polling trades a little CPU for that latency. Default 0.
fn spin_us() -> f64 {
    static V: OnceLock<f64> = OnceLock::new();
    *V.get_or_init(|| {
        vitaslop_runtime::knobs::var("VITASLOP_SMP_SPIN_US")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .unwrap_or(0.0)
    })
}

/// >>> ADAPTIVE SPIN: the ceiling, in microseconds, on how long a guest worker polls its doorbell
/// before sleeping WHEN ITS RECENT MID-FRAME WAITS SAY A HAND-OFF IS COMING (`VITASLOP_SMP_SPIN_CAP_US`,
/// default 400; 0 turns adaptive spinning off). `VITASLOP_SMP_SPIN_US` (a fixed spin on every
/// wait) still wins when it is set.
///
/// MEASURED on the phone (2026-09-25, MLB, SMP): a thread made runnable by another worker waited
/// for its sleeping worker to wake - `queued` 4.42 ms/f for the main thread and 6.70 ms/f for
/// the Kinematics thread against 0.29 and 1.02 on the desktop, i.e. a futex wake there costs
/// ~15x what it does here, and those waits sit on the frame's chain. A worker that polls for
/// a few hundred microseconds picks the hand-off up at once; a worker whose waits are long
/// (the storage threads' worker, idle for whole frames) keeps sleeping and burns nothing.
fn spin_cap_us() -> f64 {
    static V: OnceLock<f64> = OnceLock::new();
    *V.get_or_init(|| {
        vitaslop_runtime::knobs::var("VITASLOP_SMP_SPIN_CAP_US")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .unwrap_or(400.0)
            .max(0.0)
    })
}

/// Whether `VITASLOP_SMP_SPIN_US` was set at all - it then replaces the adaptive rule.
fn spin_fixed() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| vitaslop_runtime::knobs::var("VITASLOP_SMP_SPIN_US").is_ok())
}

/// `VITASLOP_SMP_PLACE=spread`: bind a new thread whose mask allows several workers to the one
/// with the FEWEST live threads (then the least recent work), instead of the least recent work
/// first. The load figure does not move between the spawns of one burst, so load-first sends a
/// whole burst to one worker - MEASURED on MLB (`pf25b`): 21 threads on w2 beside its render
/// thread, w3 holding 3 and busy 3%.
fn place_spread() -> bool {
    matches!(place_mode(), Place::Spread | Place::Apart)
}

/// `VITASLOP_SMP_PLACE=apart` (THE DEFAULT; `load` = the old least-recent-work rule): `spread`, but a thread whose mask allows another worker is never
/// bound beside the MAIN thread - the title's heaviest thread keeps a worker to itself, and the
/// rest share the others by count. MEASURED (MLB desktop, `sp25a`): under the default, 20
/// threads (the render thread and the per-frame Kinematics thread among them) shared w2 at 70%
/// busy while w3 held three storage threads at 4%, and Kinematics waited 1.3 ms/f RUNNABLE
/// behind its neighbours - 6.7 ms/f on the phone.
/// Capacity (UNPACED presented/s, desktop, one build): MLB load 78-90 -> apart 89-94 (`sp25a`/
/// `sp25c`); Madden load 68-73 -> apart 75-77 (`sp25d`/`sp25e`).
#[derive(Clone, Copy, PartialEq)]
enum Place {
    Load,
    Spread,
    Apart,
}

fn place_mode() -> Place {
    static V: OnceLock<Place> = OnceLock::new();
    *V.get_or_init(|| match vitaslop_runtime::knobs::var("VITASLOP_SMP_PLACE").as_deref().map(str::trim) {
        Ok("spread") => Place::Spread,
        Ok("load") => Place::Load,
        _ => Place::Apart,
    })
}

// ---------------------------------------------------------------------------------------
// The timeline (`VITASLOP_SMP_TRACE=<from frame>:<frames>`).
// ---------------------------------------------------------------------------------------

/// >>> WHO RAN WHERE, WHEN, AND WHY IT STOPPED - for `frames` frames from `from`, then dumped to
/// the console as `smptrace` lines. The per-thread totals (`run`/`queued`) cannot say what a
/// frame's critical path is: MLB's threads summed to its whole period with no worker busier than
/// 55%, which is a dependency chain, and only an ordered record of the spans shows the chain.
///
/// Events, all stamped with [`abs_ms`] (one clock across workers):
///   `R w thid t0 t1 stop` - a resume on worker `w`; stop = q(uantum) p(reempted) b(locked)
///                           f(lip) F(orward) x (finished)
///   `W thid t by`         - `thid` made runnable at `t` by worker `by` (0 = the run worker)
///   `J t`                 - an idle clock jump
///   `V frame t`           - a flip
#[derive(Clone, Copy)]
enum TraceEv {
    /// `sel`: the host-call selector a `b`locked run parked in (0 = unknown).
    Run { w: u8, thid: i32, t0: f64, t1: f64, stop: u8, sel: u32 },
    /// `wby`/`line`: the thread whose host call queued the wake and the `host.rs` line that did
    /// (0 = not recorded).
    Wake { thid: i32, t: f64, by: u8, wby: i32, line: u32 },
    Jump { t: f64 },
    Flip { frame: u64, t: f64, gate: u64 },
    /// `run_frames(target)` opened the gate to `gate`.
    Gate { target: u64, gate: u64, t: f64 },
    /// Worker `w` found nothing to run mid-frame while the machine was NOT globally idle: the
    /// counts that kept it from taking an idle step (running, runnable, forwarding, queued
    /// forwards, queued early completions). Emitted when the counts change.
    Stuck { w: u8, t: f64, c: [u16; 5] },
    /// A span of the RUN WORKER's own: r(un_frames), e(arly batch), F(orward), P(resent); inside
    /// run_frames: z (asleep on its bell), v (serving), l (waiting for the state lock), i (idle clock step).
    W0 { kind: u8, t0: f64, t1: f64 },
}

/// Record a span of the run worker's (see [`TraceEv::W0`]); `lib.rs` records the present.
pub fn trace_w0(kind: u8, t0_ms: f64, t1_ms: f64) {
    if trace().is_none() {
        return;
    }
    let frame = TRACE_FRAME.load(Ordering::Relaxed);
    // `t0_ms`/`t1_ms` are this worker's `performance.now()`; the trace clock adds the origin.
    let o = abs_ms() - now_ms();
    trace_ev(frame, TraceEv::W0 { kind, t0: t0_ms + o, t1: t1_ms + o });
}

/// A [`trace_w0`] span from construction to drop - for a region with early exits or braces
/// a closure cannot wrap.
pub struct W0Span(u8, f64);

impl W0Span {
    pub fn new(kind: u8) -> Self {
        W0Span(kind, now_ms())
    }
}

impl Drop for W0Span {
    fn drop(&mut self) {
        trace_w0(self.0, self.1, now_ms());
    }
}

/// [`trace_w0`] for times taken with `Date.now()` (epoch ms) - the clock `abs_ms` is on.
pub fn trace_w0_abs(kind: u8, t0: f64, t1: f64) {
    if trace().is_none() {
        return;
    }
    let frame = TRACE_FRAME.load(Ordering::Relaxed);
    trace_ev(frame, TraceEv::W0 { kind, t0, t1 });
}

/// The flip count, mirrored for the tracer so recording a W0 span never takes the state lock
/// (which the guest workers contend - measured stalling W0 ~5 ms after a flip).
static TRACE_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Per-selector host-call cost on the guest workers inside the traced window: (calls, ms waiting
/// for the host lock, ms in the handler). Shared memory, so every worker adds to one table.
static TRACE_CALLS: Mutex<Option<HashMap<u32, (u64, f64, f64)>>> = Mutex::new(None);

/// Record one host call's cost if the traced window is open - see [`TRACE_CALLS`].
pub fn trace_call(selector: u32, lock_wait_ms: f64, handler_ms: f64) {
    let Some(t) = trace() else { return };
    let f = TRACE_FRAME.load(Ordering::Relaxed);
    {
        let t = t.lock().unwrap();
        if f < t.from || f >= t.until {
            return;
        }
    }
    let mut g = TRACE_CALLS.lock().unwrap();
    let e = g.get_or_insert_with(HashMap::new).entry(selector).or_insert((0, 0.0, 0.0));
    e.0 += 1;
    e.1 += lock_wait_ms;
    e.2 += handler_ms;
}

struct Trace {
    from: u64,
    until: u64,
    evs: Vec<TraceEv>,
    dumped: bool,
}

static TRACE: OnceLock<Option<Mutex<Trace>>> = OnceLock::new();

/// Whether `VITASLOP_SMP_TRACE` is armed at all - for a caller that would otherwise take
/// clock readings only the trace consumes.
pub(crate) fn tracing() -> bool {
    trace().is_some()
}

fn trace() -> Option<&'static Mutex<Trace>> {
    TRACE
        .get_or_init(|| {
            let v = vitaslop_runtime::knobs::var("VITASLOP_SMP_TRACE").ok()?;
            let (a, b) = v.trim().split_once(':')?;
            let from: u64 = a.trim().parse().ok()?;
            let n: u64 = b.trim().parse().ok()?;
            Some(Mutex::new(Trace { from, until: from + n, evs: Vec::new(), dumped: false }))
        })
        .as_ref()
}

thread_local! {
    /// The last `Stuck` counts this worker traced, so only a change is recorded.
    static STUCK_LAST: std::cell::Cell<[u16; 5]> = const { std::cell::Cell::new([u16::MAX; 5]) };
}

thread_local! {
    /// The selector of the host call this worker's thread last parked in - see [`note_block`].
    static LAST_BLOCK_SEL: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Note that the running thread is parking in host call `selector` (tracing only).
pub fn note_block(selector: u32) {
    LAST_BLOCK_SEL.with(|c| c.set(selector));
}

thread_local! {
    /// This worker's index, for the wake events it records.
    static TRACE_WORKER: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Record `ev` if the traced window is open at `frame`.
fn trace_ev(frame: u64, ev: TraceEv) {
    if let Some(t) = trace() {
        let mut t = t.lock().unwrap();
        if frame >= t.from && frame < t.until {
            t.evs.push(ev);
        }
    }
}

/// Once the window has closed, print it (run worker only) and stop.
fn trace_dump(frame: u64) {
    let Some(t) = trace() else { return };
    let evs = {
        let mut t = t.lock().unwrap();
        if t.dumped || frame < t.until {
            return;
        }
        t.dumped = true;
        std::mem::take(&mut t.evs)
    };
    let base = evs
        .iter()
        .map(|e| match *e {
            TraceEv::Run { t0, .. } | TraceEv::W0 { t0, .. } => t0,
            TraceEv::Wake { t, .. } | TraceEv::Jump { t } | TraceEv::Flip { t, .. } | TraceEv::Gate { t, .. } | TraceEv::Stuck { t, .. } => t,
        })
        .fold(f64::INFINITY, f64::min);
    let us = |t: f64| ((t - base) * 1000.0).round() as i64;
    let evs_sels: Vec<u32> = evs.iter().filter_map(|e| match *e { TraceEv::Run { sel, .. } if sel != 0 => Some(sel), _ => None }).collect();
    let lines: Vec<String> = evs
        .iter()
        .map(|e| match *e {
            TraceEv::Run { w, thid, t0, t1, stop, sel } => {
                format!("R {w} {thid:#x} {} {} {} {sel}", us(t0), us(t1), stop as char)
            }
            TraceEv::Wake { thid, t, by, wby, line } => format!("W {thid:#x} {} {by} {wby:#x} {line}", us(t)),
            TraceEv::Jump { t } => format!("J {}", us(t)),
            TraceEv::Flip { frame, t, gate } => format!("V {frame} {} {gate}", us(t)),
            TraceEv::Gate { target, gate, t } => format!("G {target} {gate} {}", us(t)),
            TraceEv::Stuck { w, t, c } => format!("K {w} {} {} {} {} {} {}", us(t), c[0], c[1], c[2], c[3], c[4]),
            TraceEv::W0 { kind, t0, t1 } => format!("S {} {} {}", kind as char, us(t0), us(t1)),
        })
        .collect();
    for chunk in lines.chunks(400) {
        web_sys::console::log_1(&JsValue::from_str(&format!("smptrace {}", chunk.join(";"))));
    }
    // The host calls the window made, by selector, named: `smpcall <name> <calls> <lock ms> <handler ms>`.
    if let Some(calls) = TRACE_CALLS.lock().unwrap().take()
        && let Some(sh) = SHARED.get()
    {
        let host = lock_host(&sh.host);
        let mut rows: Vec<(u32, (u64, f64, f64))> = calls.into_iter().collect();
        rows.sort_by(|a, b| (b.1 .1 + b.1 .2).total_cmp(&(a.1 .1 + a.1 .2)));
        let named: Vec<String> = rows
            .iter()
            .take(40)
            .map(|(sel, (n, lw, hd))| {
                let name = host
                    .import_at(*sel)
                    .map(|(_, nid)| {
                        let n = vitaslop_runtime::nid::name(nid);
                        if n.is_empty() || n == "?" { format!("{nid:#010x}") } else { n.to_string() }
                    })
                    .unwrap_or_else(|| format!("sel{sel}"));
                format!("{name} {n} {lw:.2} {hd:.2}")
            })
            .collect();
        drop(host);
        web_sys::console::log_1(&JsValue::from_str(&format!("smptrace CALLS {}", named.join(";"))));
        // Names for the selectors the `R ... b <sel>` events carry.
        let host = lock_host(&sh.host);
        let mut sels: Vec<u32> = evs_sels.clone();
        sels.sort_unstable();
        sels.dedup();
        let names: Vec<String> = sels
            .iter()
            .map(|&sel| {
                let n = host.import_at(sel).map(|(_, nid)| vitaslop_runtime::nid::name(nid).to_string()).unwrap_or_default();
                format!("{sel}={n}")
            })
            .collect();
        drop(host);
        web_sys::console::log_1(&JsValue::from_str(&format!("smptrace SELS {}", names.join(";"))));
    }
    web_sys::console::log_1(&JsValue::from_str(&format!("smptrace END {} events", lines.len())));
}

/// A worker `error` event as text. >>> IT IS NOT ALWAYS AN `ErrorEvent`: a worker whose script
/// fails to load or instantiate gets a plain `Event` with no `message`, and reading the typed
/// getters on it threw inside wasm-bindgen ("Cannot read properties of undefined (reading
/// 'length')") - on the phone, that crash was the only thing the run reported, and it hid the
/// failure it was meant to describe. Every field is read by name and may be absent.
fn describe_error_event(e: &JsValue) -> String {
    let get = |k: &str| Reflect::get(e, &JsValue::from_str(k)).ok().filter(|v| !v.is_undefined() && !v.is_null());
    let s = |k: &str| get(k).map(|v| v.as_string().unwrap_or_else(|| format!("{v:?}")));
    let n = |k: &str| get(k).and_then(|v| v.as_f64()).map(|v| v as i64);
    let stack = get("error").and_then(|er| Reflect::get(&er, &JsValue::from_str("stack")).ok()).and_then(|v| v.as_string());
    format!(
        "{} [event type {}{}]{}",
        s("message").unwrap_or_else(|| "(no message: the worker's script did not load or instantiate)".into()),
        s("type").unwrap_or_else(|| "?".into()),
        match (s("filename"), n("lineno"), n("colno")) {
            (Some(f), l, c) if !f.is_empty() => format!(" at {f}:{}:{}", l.unwrap_or(0), c.unwrap_or(0)),
            _ => String::new(),
        },
        stack.map(|st| format!("
{st}")).unwrap_or_default()
    )
}

/// Where the guest lives in the shared memory, as the transpiled module was built for it.
#[derive(Clone, Copy)]
struct Geometry {
    base: u32,
    mem_pages: u32,
    mirror_off: Option<u64>,
    dirty_off: Option<u64>,
    host_off: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SState {
    Runnable,
    /// Executing on a worker right now.
    Running,
    Blocked,
    /// Parked on a host call the run worker is dispatching for it.
    Forwarding,
    Finished(u32),
}

/// How to start a thread that has not run yet: the entries in order (the main thread's
/// `module_init`s then the eboot entry; one entry for a spawned thread), r0..r3, and sp.
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
    /// Waits on REAL time (it has been woken from a wall park - the audio output thread): it
    /// runs while the frame gate is shut, as the device's audio keeps running whatever the
    /// picture is doing. `VITASLOP_SMP_RT_THROUGH_GATE=0` is the arm back.
    rt: bool,
    picks: u64,
    quanta: u64,
    retired_total: u64,
    fuel_seen: u64,
    arm_seen: u64,
    birth: Option<Birth>,
    forward: Option<ForwardReq>,
    patch: Option<RegPatch>,
    /// Finished while its engine state still lives on its worker: that worker drops it.
    release: bool,
    /// When another worker made this thread runnable ([`abs_ms`]), until its worker picks it;
    /// 0 when the last transition was its own. See [`Handoffs`].
    ready_at: f64,
    /// When it last became runnable by ANY path (its own quantum end included), and the wall
    /// clock it has spent since - runnable but not running (`queued_ms`, with the waits of 5 ms
    /// or more counted in `queued_long`) and running (`run_ms`). Beside the frame count they say
    /// whether a thread on a frame's critical path is short of CPU or waiting on its worker.
    since: f64,
    queued_ms: f64,
    queued_long: u64,
    run_ms: f64,
}

struct State {
    threads: Vec<Slot>,
    /// Slots not finished, ascending - every pass walks this, never `threads`.
    live: Vec<usize>,
    wake_tokens: HashSet<i32>,
    cursor: Vec<usize>,
    frames: u64,
    /// Guest threads run while `frames < gate`.
    gate: u64,
    running: usize,
    forwards: VecDeque<usize>,
    early: VecDeque<(i32, usize, usize)>,
    /// Early batches (and resolve-only sync parks) taken off `early` and still being served:
    /// their thread is parked on HOST work, not on game time, so the machine is not idle -
    /// an idle jump here raced the clock ahead by the whole readback (Hot Shots: 561 s of
    /// game time over one 3.3 s phone readback, 0.01x sound).
    early_serving: u32,
    verdict: Option<RunReport>,
    runnable_hist: Vec<u64>,
    fuel_total: u64,
    fuel_samples: u64,
    fuel_max: u64,
    arm_total: u64,
    fuel_unreported: u64,
    fuel_idle: u64,
    /// Guest instructions retired per worker, decayed at every frame - the placement's idea
    /// of how busy a worker is.
    load: Vec<f64>,
    forwarded: u64,
    /// Forwarded calls by import selector since the last panel line (drained there).
    forwarded_by: HashMap<u32, u64>,
    idle_jumps: u64,
    handoffs: Handoffs,
    /// Run-worker time per phase of `run_frames`, ms of its own clock: serving forwards,
    /// completing early batches, and the STOP-THE-WORLD tail (target flipped, waiting for every
    /// worker to reach a switch point).
    w0_forward_ms: f64,
    w0_early_ms: f64,
    w0_tail_ms: f64,
    /// The slot running on each worker right now, for the priority preemption test.
    running_on: Vec<Option<usize>>,
    /// Preemptions asked for, by cause: the frame gate closing, a better-priority wake.
    preempt_gate: u64,
    preempt_prio: u64,
    /// The virtual clock at each recent flip, `(frame, us)`, newest last - so an overlapped
    /// present can charge its pacing with the game time ITS frame took, not whatever the clock
    /// did while the next frame was already running.
    flip_clock: VecDeque<(u64, u64)>,
    /// The run worker is writing guest memory the guest may also be writing (a render-target
    /// write-back): no worker picks a thread until it clears. See [`SmpRun::pause_guest`].
    paused: bool,
}

/// Wake every worker blocked on `bell`.
#[cfg(target_feature = "atomics")]
fn bell_notify(bell: &AtomicI32) {
    // SAFETY: the bell is an aligned i32 in this module's shared linear memory.
    unsafe {
        std::arch::wasm32::memory_atomic_notify(bell.as_ptr(), u32::MAX);
    }
}

/// Block this worker while `bell` still reads `seen`, for at most `timeout_ns`. Only guest
/// workers call it: the run worker must keep its event loop and waits with [`wait_async`].
#[cfg(target_feature = "atomics")]
fn bell_wait(bell: &AtomicI32, seen: i32, timeout_ns: i64) {
    // SAFETY: as above; a dedicated worker may block.
    unsafe {
        std::arch::wasm32::memory_atomic_wait32(bell.as_ptr(), seen, timeout_ns);
    }
}

// The single-threaded bundle has no shared memory and no parallel run (`SmpRun::start` refuses
// it), so these are never reached there; they exist so the module still compiles.
#[cfg(not(target_feature = "atomics"))]
fn bell_notify(_bell: &AtomicI32) {}
#[cfg(not(target_feature = "atomics"))]
fn bell_wait(_bell: &AtomicI32, _seen: i32, _timeout_ns: i64) {}

impl Shared {
    fn ring(&self, w: usize) {
        if let Some(b) = self.bells.get(w) {
            if let Some(r) = self.rung_at.get(w) {
                r.store(abs_ms().to_bits(), Ordering::Relaxed);
            }
            b.fetch_add(1, Ordering::SeqCst);
            bell_notify(b);
        }
    }

    fn ring_all(&self) {
        for w in 0..=self.workers {
            self.ring(w);
        }
    }

    /// Worker `w`'s PREEMPT word: a non-zero value makes the thread running there yield at its
    /// next loop back edge (or host call). It sits in the host-mirror page at
    /// `abi::PREEMPT_SLOT_BASE + w`, which the guest reaches through its layout offset and this
    /// module through the linear address.
    fn preempt(&self, w: usize) -> Option<&AtomicI32> {
        let g = self.geometry;
        let off = g.mirror_off? + 4 * u64::from(vitaslop_transpiler::abi::PREEMPT_SLOT_BASE + w as u32);
        let addr = u64::from(g.host_off) + off;
        // SAFETY: inside the guest layout this module reserved and never frees; aligned; only
        // ever accessed as an i32 (atomically here, by a plain load in the guest).
        Some(unsafe { &*(addr as usize as *const AtomicI32) })
    }

    /// Ask worker `w` for its CPU back.
    fn preempt_worker(&self, w: usize) {
        if let Some(p) = self.preempt(w) {
            p.store(1, Ordering::SeqCst);
        }
    }
}

impl State {
    fn new(workers: usize) -> State {
        State {
            threads: Vec::new(),
            live: Vec::new(),
            wake_tokens: HashSet::new(),
            cursor: vec![0; workers + 1],
            frames: 0,
            gate: 0,
            running: 0,
            forwards: VecDeque::new(),
            early: VecDeque::new(),
            early_serving: 0,
            verdict: None,
            runnable_hist: Vec::new(),
            fuel_total: 0,
            fuel_samples: 0,
            fuel_max: 0,
            arm_total: 0,
            fuel_unreported: 0,
            fuel_idle: 0,
            load: vec![0.0; workers + 1],
            forwarded: 0,
            forwarded_by: HashMap::new(),
            idle_jumps: 0,
            handoffs: Handoffs::default(),
            w0_forward_ms: 0.0,
            w0_early_ms: 0.0,
            w0_tail_ms: 0.0,
            running_on: vec![None; workers + 1],
            preempt_gate: 0,
            preempt_prio: 0,
            flip_clock: VecDeque::new(),
            paused: false,
        }
    }

    /// Thread `idx` just became runnable: if the thread running on its worker is of a WORSE
    /// priority (a larger number), ask that worker for the CPU - the SceKernel rule that a
    /// woken higher-priority thread takes the core now, not at the running one's next block.
    fn preempt_for(&mut self, sh: &Shared, idx: usize) {
        let w = self.threads[idx].home;
        if let Some(r) = self.running_on[w] {
            if self.threads[idx].priority < self.threads[r].priority {
                self.preempt_prio += 1;
                sh.preempt_worker(w);
            }
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
            // ...and in the guest's own word for it, which the inline elided yield reads
            // (`InlineOp::SmpDelayYield`).
            let slot = vitaslop_transpiler::abi::PREEMPT_SLOT_BASE
                + vitaslop_transpiler::abi::SMP_RUNNABLE_SLOT_OFFSET
                + w as u32;
            if let Some(a) = mirror_word(sh, slot) {
                // SAFETY: an aligned word in the mirror page of the reserved layout.
                unsafe { (*(a as *const AtomicI32)).store(c as i32, Ordering::Relaxed) };
            }
        }
    }

    /// The worker a new thread is bound to. The Vita's user cores are affinity bits 16..18;
    /// a mask naming some of them picks among those, any other mask (including "all") picks
    /// among every worker. Among the candidates: the least recent guest work, then the fewest
    /// live threads - a thread spawned now is most likely to share a core with the ones that
    /// are busy NOW.
    fn place(&self, workers: usize, mask: i32) -> usize {
        let mut cands: Vec<usize> = (0..3)
            .filter(|k| mask & (1 << (16 + k)) != 0)
            .map(|k| 1 + (k as usize) % workers)
            .collect();
        cands.sort_unstable();
        cands.dedup();
        // No core named, or all three ("any core"): every worker is a candidate.
        if cands.is_empty() || mask & 0x0007_0000 == 0x0007_0000 {
            cands = (1..=workers).collect();
        }
        if place_mode() == Place::Apart && cands.len() > 1 {
            let main_home = self
                .threads
                .iter()
                .find(|t| t.thid == vitaslop_runtime::host::MAIN_THID)
                .map(|t| t.home);
            if let Some(m) = main_home {
                cands.retain(|&w| w != m);
            }
        }
        let live_on = |w: usize| self.live.iter().filter(|&&i| self.threads[i].home == w).count();
        let by_load = |a: usize, b: usize| {
            self.load[a].partial_cmp(&self.load[b]).unwrap_or(std::cmp::Ordering::Equal)
        };
        *cands
            .iter()
            .min_by(|&&a, &&b| {
                if place_spread() {
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
            picks: 0,
            quanta: 0,
            retired_total: 0,
            fuel_seen: 0,
            arm_seen: 0,
            birth: Some(birth),
            forward: None,
            patch: None,
            release: false,
            ready_at: abs_ms(),
            since: abs_ms(),
            queued_ms: 0.0,
            queued_long: 0,
            run_ms: 0.0,
        });
        self.live.push(idx);
        tracing::info!(
            target: "vitaslop::smp",
            "thread {thid:#x} (prio {priority:#x}) bound to worker {home}"
        );
        self.preempt_for(sh, idx);
        sh.ring(home);
    }

    /// The next thread worker `w` should run: strict priority among ITS runnable threads,
    /// round-robin inside a level, spin-cooled threads stepping aside - the one-baton pick,
    /// per core.
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
        let best = mine
            .iter()
            .filter(|&&i| !self.threads[i].cooled)
            .map(|&i| self.threads[i].priority)
            .min()?;
        let cursor = self.cursor[w];
        let start = mine.partition_point(|&i| i < cursor);
        let idx = mine[start..]
            .iter()
            .chain(&mine[..start])
            .copied()
            .find(|&i| !self.threads[i].cooled && self.threads[i].priority == best)?;
        self.cursor[w] = idx + 1;
        let t = &mut self.threads[idx];
        if t.ready_at > 0.0 {
            let ms = abs_ms() - t.ready_at;
            t.ready_at = 0.0;
            self.handoffs.note(ms.max(0.0));
        }
        let t = &mut self.threads[idx];
        if t.since > 0.0 {
            let q = (abs_ms() - t.since).max(0.0);
            t.queued_ms += q;
            if q >= 5.0 {
                t.queued_long += 1;
            }
        }
        t.picks += 1;
        t.state = SState::Running;
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
                        self.threads[idx].since = abs_ms();
            self.threads[idx].cooled = false;
        } else {
            self.threads[idx].state = SState::Blocked;
        }
    }

    /// Charge the game clock for the guest work a resume did - `SchedCore::charge_guest_work`,
    /// with the runnable count taken over EVERY worker (running threads included), which is
    /// what the clock divides by: up to three of them retired their work at the same time.
    fn charge(&mut self, sh: &Shared, idx: usize, fuel: Option<u64>, arm: Option<u64>) {
        let Some(total) = fuel else {
            self.fuel_unreported += 1;
            return;
        };
        let burned = total.saturating_sub(self.threads[idx].fuel_seen);
        self.threads[idx].fuel_seen = total;
        let retired = arm.map(|t| {
            let d = t.saturating_sub(self.threads[idx].arm_seen);
            self.threads[idx].arm_seen = t;
            self.threads[idx].retired_total = self.threads[idx].retired_total.saturating_add(d);
            d
        });
        if burned == 0 && retired.unwrap_or(0) == 0 {
            self.fuel_idle += 1;
            return;
        }
        let home = self.threads[idx].home;
        self.load[home] += retired.unwrap_or(burned) as f64;
        // The thread that just stopped counts: it was running when this work was retired.
        let runnable = 1 + self
            .live
            .iter()
            .filter(|&&i| i != idx && matches!(self.threads[i].state, SState::Runnable | SState::Running))
            .count();
        if self.runnable_hist.len() <= runnable {
            self.runnable_hist.resize(runnable + 1, 0);
        }
        self.runnable_hist[runnable] += 1;
        self.fuel_total = self.fuel_total.saturating_add(burned);
        self.fuel_samples += 1;
        self.fuel_max = self.fuel_max.max(burned);
        self.arm_total = self.arm_total.saturating_add(retired.unwrap_or(0));
        let mut host = lock_host(&sh.host);
        host.on_guest_work(runnable, burned, retired);
        publish_words(sh, &host.state);
    }

    /// One display frame ended on thread `idx` (which blocks until the next vblank, as it does
    /// on hardware - see `SchedCore::on_suspended`).
    fn flip(&mut self, sh: &Shared, idx: usize) {
        self.block_or_consume_token(idx);
        self.frames += 1;
        TRACE_FRAME.store(self.frames, Ordering::Relaxed);
        trace_ev(self.frames, TraceEv::Flip { frame: self.frames, t: abs_ms(), gate: self.gate });
        if Some(self.frames) == vitaslop_runtime::sched::cpu_share_from() {
            for t in self.threads.iter_mut() {
                t.picks = 0;
                t.quanta = 0;
                t.retired_total = 0;
                t.queued_ms = 0.0;
                t.queued_long = 0;
                t.run_ms = 0.0;
            }
            for h in self.runnable_hist.iter_mut() {
                *h = 0;
            }
        }
        for l in self.load.iter_mut() {
            *l *= 0.9;
        }
        vitaslop_runtime::sched::set_current_frame(self.frames);
        {
            // `B` (on a guest worker, under the state lock): the flip's host bookkeeping.
            let tb = abs_ms();
            let mut host = lock_host(&sh.host);
            let tbl = abs_ms();
            host.on_frame_boundary(self.frames);
            let tbf = abs_ms();
            publish_words(sh, &host.state);
            let tbp = abs_ms();
            trace_ev(self.frames, TraceEv::W0 { kind: b'h', t0: tb, t1: tbl });
            trace_ev(self.frames, TraceEv::W0 { kind: b'B', t0: tbl, t1: tbf });
            trace_ev(self.frames, TraceEv::W0 { kind: b'U', t0: tbf, t1: tbp });
            self.flip_clock.push_back((self.frames, host.clock_us()));
            if self.flip_clock.len() > 16 {
                self.flip_clock.pop_front();
            }
        }
        // The gate just closed: every thread still running has to reach a switch point
        // before the run worker may present, so ask them all now rather than waiting for
        // their fuel to run out.
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
        let (kind, code) = match &end {
            FiberEnd::Returned(c) => ("returned from its entry", *c),
            FiberEnd::ThreadExit(c) => ("sceKernelExitThread", *c),
            FiberEnd::ProcessHalt(c) => ("halted the process", *c),
            FiberEnd::Error(_) => ("TRAPPED", 0),
        };
        tracing::info!(
            target: "vitaslop::thread",
            "thread {thid:#x} FINISHED at frame {} on worker {}: {kind} (code {code:#x})",
            self.frames,
            self.threads[idx].home,
        );
        match end {
            FiberEnd::Returned(code) | FiberEnd::ThreadExit(code) => {
                self.threads[idx].state = SState::Finished(code);
                self.threads[idx].release = true;
                self.retire(idx);
                lock_host(&sh.host).set_thread_exit(thid, code);
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

    /// Fold a finished resume of `idx` back in. `forward` and `early` are what the thread's
    /// import closure parked during it.
    #[allow(clippy::too_many_arguments)]
    fn fold(
        &mut self,
        sh: &Shared,
        idx: usize,
        step: ThreadStep,
        fuel: Option<u64>,
        arm: Option<u64>,
        forward: Option<ForwardReq>,
        early: Option<(i32, usize, usize)>,
        preempted: bool,
    ) {
        self.charge(sh, idx, fuel, arm);
        if let Some(e) = early {
            self.early.push_back(e);
            sh.ring(0);
        }
        match step {
            ThreadStep::Finished(end) => self.finish(sh, idx, end),
            ThreadStep::Suspended(stop) => {
                self.running -= 1;
                let home = self.threads[idx].home;
                self.running_on[home] = None;
                match stop {
                    // Taken off the CPU by another worker (the gate, a better-priority wake),
                    // not by spinning through its own quantum: still runnable, and NOT cooled -
                    // the cooldown is for a thread that burned a whole quantum doing nothing.
                    Stop::Quantum if preempted => {
                        self.threads[idx].state = SState::Runnable;
                        self.threads[idx].since = abs_ms();
                    }
                    Stop::Blocked if forward.is_some() => {
                        if let Some(f) = &forward {
                            *self.forwarded_by.entry(f.selector).or_default() += 1;
                        }
                        self.threads[idx].state = SState::Forwarding;
                        self.threads[idx].forward = forward;
                        self.forwards.push_back(idx);
                        self.forwarded += 1;
                        sh.ring(0);
                    }
                    Stop::Blocked => self.block_or_consume_token(idx),
                    Stop::Flip => self.flip(sh, idx),
                    Stop::Quantum => {
                        self.threads[idx].state = SState::Runnable;
                        self.threads[idx].since = abs_ms();
                        self.threads[idx].cooled = true;
                        self.threads[idx].quanta += 1;
                    }
                }
            }
        }
    }

    /// Start the threads the host spawned and wake the ones it released - `SchedCore::drain`.
    fn drain(&mut self, sh: &Shared, engine: &BrowserEngine) {
        let (spawns, wakes, stat_writes, masks, causes, rt_woke) = {
            let mut host = lock_host(&sh.host);
            let held = self.frames >= self.gate || self.paused;
            if wall_floor(self.frames) && host.state.wall_floor_tick(abs_ms(), held) {
                publish_words(sh, &host.state);
            }
            let rt_woke = if host.state.has_wall_parks() {
                host.state.expire_wall_parks((abs_ms() * 1000.0) as u64)
            } else {
                Vec::new()
            };
            let mut words = Words(engine);
            host.resolve_deferred(&mut words);
            let spawns: Vec<Reentry> = host.take_spawns();
            let masks: Vec<i32> =
                spawns.iter().map(|s| host.state.thread_cpu_affinity(s.thid)).collect();
            host.state.trace_wakes = trace().is_some();
            let causes = host.state.take_wake_causes();
            (spawns, host.take_wakes(), host.take_stat_writes(), masks, causes, rt_woke)
        };
        for thid in rt_woke {
            if let Some(i) = self.live.iter().copied().find(|&i| self.threads[i].thid == thid) {
                self.threads[i].rt = true;
            }
        }
        for (addr, value) in stat_writes {
            engine.write_guest(addr, &value.to_le_bytes());
        }
        for (sp, mask) in spawns.into_iter().zip(masks) {
            let home = self.place(sh.workers, mask);
            let birth = Birth { entries: vec![sp.entry], r: [sp.arg_len, sp.arg_ptr, sp.r2, sp.r3], sp: sp.stack_top };
            self.add_thread(sh, sp.thid, sp.priority, home, birth);
        }
        for thid in wakes {
            let cause = causes.iter().find(|c| c.0 == thid).map_or((0, 0), |c| (c.1, c.2));
            let found = self
                .live
                .iter()
                .copied()
                .find(|&i| self.threads[i].thid == thid && self.threads[i].state == SState::Blocked);
            match found {
                Some(i) => {
                    self.threads[i].state = SState::Runnable;
                    self.threads[i].since = abs_ms();
                    self.threads[i].cooled = false;
                    self.threads[i].ready_at = abs_ms();
                    if tracing() {
                        trace_ev(self.frames, TraceEv::Wake { thid, t: abs_ms(), by: TRACE_WORKER.with(|c| c.get()), wby: cause.0, line: cause.1 });
                    }
                    self.preempt_for(sh, i);
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

    /// Nothing can run anywhere and nothing is in flight: the only way forward is time.
    fn globally_idle(&self) -> bool {
        self.running == 0
            && self.forwards.is_empty()
            && self.early.is_empty()
            && self.early_serving == 0
            && !self.live.iter().any(|&i| {
                matches!(self.threads[i].state, SState::Runnable | SState::Forwarding)
            })
    }

    /// `SchedCore::handle_idle` for the whole machine: deliver what is owed, then complete the
    /// nearer of the next transfer and the next timed wait, or report a deadlock.
    fn idle_step(&mut self, sh: &Shared, engine: &BrowserEngine) {
        self.drain(sh, engine);
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
        if tracing() {
            trace_ev(self.frames, TraceEv::Jump { t: abs_ms() });
        }
        let mut host = lock_host(&sh.host);
        let (deadline, io, now) = (host.earliest_deadline(), host.earliest_io_remaining_us(), host.clock_us());
        if vitaslop_runtime::sched::storage_completes_first(io, deadline, now) && host.release_earliest_io() {
            drop(host);
            self.drain(sh, engine);
            return;
        }
        match deadline {
            Some(t) => {
                host.charge_io_idle(t.saturating_sub(now));
                host.advance_time_to(t);
                publish_words(sh, &host.state);
                drop(host);
                self.drain(sh, engine);
            }
            // Only wall parks left (`VitaState::wall_park`): the guest is WAITING on real time,
            // not stuck - an idle worker wakes for it.
            None if host.state.has_wall_parks() => {}
            None => {
                let orphans: Vec<i32> =
                    blocked.iter().copied().filter(|&t| !host.thread_has_wait_record(t)).collect();
                let dump = host.sync_dump();
                drop(host);
                web_sys::console::error_1(&JsValue::from_str(&format!(
                    "SMP DEADLOCK: {} thread(s) blocked with no timeout and no pending I/O: \
                     {blocked:x?} (no wait record: {orphans:x?})\n{dump}",
                    blocked.len()
                )));
                self.verdict = Some(RunReport::Deadlock(blocked));
                sh.ring_all();
            }
        }
    }
}

/// Guest words through an engine's view of the shared region, for the host's deferred settles.
struct Words<'a>(&'a BrowserEngine);

impl vitaslop_runtime::host::GuestWords for Words<'_> {
    fn word(&self, addr: u32) -> u32 {
        let mut b = [0u8; 4];
        if self.0.read_mem(addr, &mut b) { u32::from_le_bytes(b) } else { 0 }
    }
    fn set_word(&mut self, addr: u32, value: u32) {
        self.0.write_guest(addr, &value.to_le_bytes());
    }
}

/// Linear address of host-mirror slot `slot` (SMP words live at the top of the block).
fn mirror_word(sh: &Shared, slot: u32) -> Option<usize> {
    let g = sh.geometry;
    Some((u64::from(g.host_off) + g.mirror_off? + u64::from(slot) * 4) as usize)
}

/// >>> PUBLISH THE GLOBAL WORDS a parallel build reads inline: the vblank count and the SA bank
/// as plain words, the process clock and RTC tick as one atomic 64-bit word each
/// (`InlineOp::LoadClock64`). Returns `(current thread, reported thread id)` as the snapshot
/// computed them, for the caller's instance globals (`InlineOp::ThreadWord`).
///
/// Called at every resume AND wherever the virtual clock moves (a charge, an idle jump, a
/// flip): under SMP the clock advances while other workers' threads run, and a poller reading
/// it inline must see that without waiting for its own next resume - that is how a pacing loop
/// on one worker notices the time another worker's work paid for.
fn publish_words(sh: &Shared, st: &vitaslop_runtime::host::VitaState) -> (u32, u32) {
    use vitaslop_runtime::vita::mirror as m;
    use vitaslop_transpiler::abi;
    let s = m::snapshot(st);
    // SAFETY (all four): aligned words inside the reserved guest layout's mirror page, which
    // lives for the whole run; written atomically, read by the guest with plain or atomic loads.
    unsafe {
        if let Some(a) = mirror_word(sh, m::SLOT_VCOUNT) {
            (*(a as *const AtomicI32)).store(s[m::SLOT_VCOUNT as usize] as i32, Ordering::Relaxed);
        }
        if let Some(a) = mirror_word(sh, m::SLOT_SA_BANK) {
            (*(a as *const AtomicI32)).store(s[m::SLOT_SA_BANK as usize] as i32, Ordering::Relaxed);
        }
        let pair = |lo: u32| u64::from(s[lo as usize]) | (u64::from(s[lo as usize + 1]) << 32);
        if let Some(a) = mirror_word(sh, abi::SMP_CLOCK64_SLOT) {
            (*(a as *const AtomicU64)).store(pair(m::SLOT_CLOCK_LO), Ordering::Relaxed);
        }
        if let Some(a) = mirror_word(sh, abi::SMP_RTC64_SLOT) {
            (*(a as *const AtomicU64)).store(pair(m::SLOT_RTC_LO), Ordering::Relaxed);
        }
    }
    (s[m::SLOT_CURRENT_THREAD as usize], s[m::SLOT_THREAD_ID as usize])
}

/// Before `thid` resumes on this worker: make it the host's current thread, publish the
/// global words, and hand back its per-thread words.
fn prepare_resume(sh: &Shared, thid: i32) -> (u32, u32) {
    let mut host = lock_host(&sh.host);
    host.set_current_thread(thid);
    publish_words(sh, &host.state)
}

fn now_ms() -> f64 {
    browser_sched::perf_clock()
}

/// A clock every worker reads the SAME way: `performance.timeOrigin + performance.now()`.
/// Each worker's `now()` counts from its own start, so a time stamped on one worker and read on
/// another is only comparable after adding the origin - which is what a HANDOFF latency (made
/// runnable here, picked up there) needs.
pub(crate) fn abs_ms() -> f64 {
    thread_local! {
        static ORIGIN: f64 = Reflect::get(&js_sys::global(), &"performance".into())
            .ok()
            .and_then(|p| Reflect::get(&p, &"timeOrigin".into()).ok())
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
    }
    ORIGIN.with(|o| *o) + now_ms()
}

/// Where cross-worker handoffs spend their time: a thread made runnable by ANOTHER worker (a
/// wake from a host call there, a forwarded call W0 finished) until its own worker runs it.
/// This is the latency a parallel run adds that the one-baton engine does not have at all, and
/// it sits on the critical path of every producer/consumer chain in a frame.
#[derive(Default, Clone, Copy)]
struct Handoffs {
    n: u64,
    sum_ms: f64,
    max_ms: f64,
    /// `< 0.05 ms`, `< 0.2`, `< 1`, `< 5`, `>= 5`.
    buckets: [u64; 5],
}

impl Handoffs {
    fn note(&mut self, ms: f64) {
        self.n += 1;
        self.sum_ms += ms;
        self.max_ms = self.max_ms.max(ms);
        let b = match ms {
            m if m < 0.05 => 0,
            m if m < 0.2 => 1,
            m if m < 1.0 => 2,
            m if m < 5.0 => 3,
            _ => 4,
        };
        self.buckets[b] += 1;
    }
}

// ---------------------------------------------------------------------------------------
// The guest worker.
// ---------------------------------------------------------------------------------------

/// Entry point of a guest worker (`web/smp-worker.js`), after the bundle is initialised over
/// the shared memory. Runs until the run ends.
#[wasm_bindgen]
pub async fn smp_helper_main(
    worker: u32,
    guest_module: JsValue,
    audio_ring: JsValue,
    storage: JsValue,
) -> Result<(), JsValue> {
    crate::logging::install_panic_hook();
    let w = worker as usize;
    TRACE_WORKER.with(|c| c.set(worker as u8));
    let sh = SHARED
        .get()
        .cloned()
        .ok_or_else(|| JsValue::from_str("SMP worker started before the run worker set the run up"))?;
    if !audio_ring.is_undefined() && !audio_ring.is_null() {
        let _ = crate::audio::install_ring(&audio_ring);
    }
    // The run worker only stops forwarding file calls when it sent a storage ring, so a worker
    // that was sent one and cannot use it must fail here, not at its first read.
    if !storage.is_undefined() && !storage.is_null() && !crate::opfs::install_worker_reader(&storage) {
        return Err(JsValue::from_str("SMP worker: the storage reader it was sent is not a reader"));
    }
    let module: WebAssembly::Module = guest_module.dyn_into()?;
    let g = sh.geometry;
    let engine = BrowserEngine::attach(
        module,
        sh.host.clone(),
        g.base,
        g.mem_pages,
        g.mirror_off,
        g.dirty_off,
        g.host_off,
        SmpHooks {
            owner_only: sh.owner_only.clone(),
            per_call: sh.per_call.clone(),
            worker: w,
            preempt_off: g.mirror_off.map(|m| m + 4 * u64::from(vitaslop_transpiler::abi::PREEMPT_SLOT_BASE + w as u32)),
            preempt_ptr: sh.preempt(w).map(|p| p.as_ptr() as usize),
        },
    )?;
    helper_loop(&sh, &engine, w).await;
    Ok(())
}

/// Entry point of the RESOLVER worker (`web/smp-worker.js` with `role: "resolver"`), under
/// `VITASLOP_SMP_ASYNC_RESOLVE`: it reads each flip's deferred geometry so neither the guest's
/// render thread nor the run worker does. Returns when the run stops.
#[wasm_bindgen]
pub fn smp_resolver_main() -> Result<(), JsValue> {
    crate::logging::install_panic_hook();
    let sh = SHARED
        .get()
        .cloned()
        .ok_or_else(|| JsValue::from_str("SMP resolver started before the run worker set the run up"))?;
    let g = sh.geometry;
    crate::browser_sched::run_resolver_worker(&sh.host, g.base, g.mem_pages, g.dirty_off, g.host_off);
    Ok(())
}

async fn helper_loop(sh: &Arc<Shared>, engine: &BrowserEngine, w: usize) {
    let local: RefCell<HashMap<usize, BrowserThread>> = RefCell::new(HashMap::new());
    let stats = &sh.stats[w];
    let mut history = vitaslop_runtime::sched::SpinHistory::new();
    loop {
        if sh.stop.load(Ordering::SeqCst) {
            break;
        }
        let bell = sh.bells[w].load(Ordering::SeqCst);
        // Take a job: a thread of ours to run, or an idle clock step for the whole machine.
        enum Job {
            Run(usize, Option<Birth>, Option<RegPatch>, i32, i32),
            Again,
            /// Nothing to run. `true` = MID-FRAME (the gate is open), where a wait is a hand-off
            /// from another worker and [`vitaslop_runtime::sched::SpinHistory`] decides whether to poll for it.
            Wait(bool),
        }
        let (job, releases) = {
            let mut st = sh.state.lock().unwrap();
            let releases: Vec<usize> = local
                .borrow()
                .keys()
                .copied()
                .filter(|&i| st.threads[i].release)
                .collect();
            let gated = st.frames >= st.gate;
            let job = if gated && st.verdict.is_none() && !st.paused && rt_through_gate() && let Some(i) = st.pick(w, true) {
                // A real-time thread keeps running behind a shut gate - see `Slot::rt`.
                let t = &mut st.threads[i];
                Job::Run(i, t.birth.take(), t.patch.take(), t.thid, t.priority)
            } else if st.verdict.is_some() || gated || st.paused {
                Job::Wait(false)
            } else if let Some(i) = st.pick(w, false) {
                st.publish_runnable(sh);
                let t = &mut st.threads[i];
                Job::Run(i, t.birth.take(), t.patch.take(), t.thid, t.priority)
            } else if st.globally_idle() {
                st.idle_step(sh, engine);
                Job::Again
            } else {
                if trace().is_some() {
                    let mut c = [st.running as u16, 0, 0, st.forwards.len() as u16, st.early.len() as u16];
                    for &i in &st.live {
                        match st.threads[i].state {
                            SState::Runnable => c[1] += 1,
                            SState::Forwarding => c[2] += 1,
                            _ => {}
                        }
                    }
                    STUCK_LAST.with(|l| {
                        if l.get() != c {
                            l.set(c);
                            trace_ev(st.frames, TraceEv::Stuck { w: w as u8, t: abs_ms(), c });
                        }
                    });
                }
                Job::Wait(true)
            };
            (job, releases)
        };
        for i in releases {
            if let Some(mut t) = local.borrow_mut().remove(&i) {
                t.release();
            }
        }
        match job {
            Job::Wait(mid_frame) => {
                let t0 = now_ms();
                let cap = spin_cap_us() / 1000.0;
                let spin = if spin_fixed() {
                    spin_us() / 1000.0
                } else if mid_frame && cap > 0.0 && history.should_spin() {
                    cap
                } else {
                    0.0
                };
                let mut rung = false;
                if spin > 0.0 {
                    stats.spins.fetch_add(1, Ordering::Relaxed);
                    while now_ms() - t0 < spin {
                        if sh.bells[w].load(Ordering::SeqCst) != bell {
                            rung = true;
                            break;
                        }
                        std::hint::spin_loop();
                    }
                }
                // 50 ms is only a backstop - every state change that can give this worker
                // something to do rings it.
                if rung {
                    stats.spin_caught.fetch_add(1, Ordering::Relaxed);
                } else {
                    // Woken for the earliest WALL park too (`VitaState::wall_park`): nothing else
                    // expires one while the guest is idle and the run worker presents.
                    let timeout_ns = match vitaslop_runtime::host::next_wall_park_us() {
                        Some(t) => {
                            let now = (abs_ms() * 1000.0) as u64;
                            (t.saturating_sub(now) * 1000).clamp(100_000, 50_000_000) as i64
                        }
                        None => 50_000_000,
                    };
                    bell_wait(&sh.bells[w], bell, timeout_ns);
                    if vitaslop_runtime::host::next_wall_park_us().is_some_and(|t| t <= (abs_ms() * 1000.0) as u64) {
                        sh.state.lock().unwrap().drain(sh, engine);
                    }
                }
                let waited = now_ms() - t0;
                if mid_frame {
                    // >>> JUDGED BY WHEN THE BELL RANG, NOT WHEN THIS WORKER WOKE. A futex wake
                    // costs what the device charges for it, and counting it here made every
                    // wait on the phone look long, so the history never said "spin" - MEASURED
                    // `spun 0 caught 0` on all three workers (2026-09-25, MLB) while the
                    // render thread queued 7.4 ms/f and Kinematics 11.5 ms/f. The hand-off
                    // distance is sleep-start to ring; the wake latency is what spinning removes.
                    let rung = f64::from_bits(sh.rung_at[w].load(Ordering::Relaxed));
                    let a0 = abs_ms() - waited;
                    let to_ring = if rung >= a0 { rung - a0 } else { waited };
                    history.record(cap > 0.0 && to_ring <= cap);
                }
                stats.wait_us.fetch_add((waited * 1000.0) as u64, Ordering::Relaxed);
            }
            Job::Again => {}
            Job::Run(i, birth, patch, thid, priority) => {
                if let Some(b) = birth {
                    match engine.make_thread(thid, &b.entries, b.r[0], b.r[1], b.r[2], b.r[3], b.sp, priority) {
                        Ok(t) => {
                            local.borrow_mut().insert(i, t);
                        }
                        Err(e) => {
                            tracing::warn!(target: "vitaslop::smp", "thread {thid:#x} could not start: {e:?}");
                            let mut st = sh.state.lock().unwrap();
                            st.running -= 1;
                            st.running_on[w] = None;
                            st.threads[i].state = SState::Finished(0);
                            st.retire(i);
                            lock_host(&sh.host).set_thread_exit(thid, 0);
                            st.drain(sh, engine);
                            continue;
                        }
                    }
                }
                let mut t = match local.borrow_mut().remove(&i) {
                    Some(t) => t,
                    None => {
                        let mut st = sh.state.lock().unwrap();
                        st.verdict = Some(RunReport::Error(format!(
                            "SMP: thread {thid:#x} was picked on worker {w}, which does not hold it"
                        )));
                        sh.ring_all();
                        continue;
                    }
                };
                if let Some(p) = patch {
                    t.set_patch(p);
                }
                let (cur, tid) = prepare_resume(sh, thid);
                t.set_smp_words(cur, tid);
                // A preempt request left over from before this pick is for the thread that was
                // running then, not for this one.
                if let Some(p) = sh.preempt(w) {
                    p.store(0, Ordering::SeqCst);
                }
                LOCK_WAIT_MS.with(|w| w.set(0.0));
                let t0 = now_ms();
                let step = browser_sched::resume(&mut t).await;
                if guest_prof().is_some() {
                    if let Some(s) = prof_slot(sh, w) {
                        s.store(0, Ordering::Relaxed);
                    }
                }
                // `VITASLOP_SMP_GUEST_SLOW=<x>`: spin x times the slice's own length after it,
                // before the thread is folded back - a guest worker that is (1+x) times slower,
                // as a phone's is. The desktop's guest is 4-5x faster than the phone's, so how
                // the scheduler overlaps guest, render and GPU THERE cannot be read here without
                // it. A measurement rig only; unset = no cost.
                let slow = guest_slow();
                if slow > 0.0 {
                    let waited = LOCK_WAIT_MS.with(|w| w.replace(0.0)) + vitaslop_runtime::perf::take_excluded_ms();
                    let until = now_ms() + ((now_ms() - t0) - waited).max(0.0) * slow;
                    while now_ms() < until {
                        std::hint::spin_loop();
                    }
                }
                let ran_ms = now_ms() - t0;
                stats.busy_us.fetch_add((ran_ms * 1000.0) as u64, Ordering::Relaxed);
                stats.resumes.fetch_add(1, Ordering::Relaxed);
                // Whether the quantum stop (if that is what this was) came from another
                // worker's request rather than this thread's own fuel.
                let preempted = sh.preempt(w).is_some_and(|p| p.swap(0, Ordering::SeqCst) != 0)
                    && matches!(step, ThreadStep::Suspended(Stop::Quantum));
                if matches!(step, ThreadStep::Suspended(Stop::Quantum)) && !preempted {
                    stats.quanta.fetch_add(1, Ordering::Relaxed);
                }
                let fuel = t.fuel_used();
                let arm = t.arm_retired();
                let forward = t.take_forward();
                let mut early = t.take_early();
                // A resolve-only park (empty batch) is settled right here - see
                // `BrowserEngine::settle_sync_resolve`. Folded as a plain block first, so the
                // wake below lands on a blocked thread (or as its token).
                let settle = match early {
                    Some((thid_e, s, n)) if s == n && sync_resolve_on_worker() => {
                        early = None;
                        Some(thid_e)
                    }
                    _ => None,
                };
                let finished = matches!(step, ThreadStep::Finished(_));
                {
                    let mut st = sh.state.lock().unwrap();
                    st.threads[i].run_ms += ran_ms;
                    if trace().is_some() {
                        let stop = match &step {
                            ThreadStep::Finished(_) => b'x',
                            ThreadStep::Suspended(Stop::Quantum) if preempted => b'p',
                            ThreadStep::Suspended(Stop::Quantum) => b'q',
                            ThreadStep::Suspended(Stop::Blocked) if forward.is_some() => b'F',
                            ThreadStep::Suspended(Stop::Blocked) => b'b',
                            ThreadStep::Suspended(Stop::Flip) => b'f',
                        };
                        let t1 = abs_ms();
                        let sel = if stop == b'b' { LAST_BLOCK_SEL.with(|c| c.replace(0)) } else { 0 };
                        trace_ev(st.frames, TraceEv::Run { w: w as u8, thid, t0: t1 - ran_ms, t1, stop, sel });
                    }
                    st.fold(sh, i, step, fuel, arm, forward, early, preempted);
                    if settle.is_some() {
                        st.early_serving += 1;
                    }
                    if settle.is_none() {
                        st.drain(sh, engine);
                    }
                    if st.paused && st.running == 0 {
                        sh.ring(0);
                    }
                }
                if let Some(thid_e) = settle {
                    engine.settle_sync_resolve(thid_e);
                    let mut st = sh.state.lock().unwrap();
                    st.early_serving -= 1;
                    st.drain(sh, engine);
                }
                if finished {
                    t.release();
                } else {
                    local.borrow_mut().insert(i, t);
                }
                if stats.resumes.load(Ordering::Relaxed) % 64 == 0 {
                    publish_tallies(sh);
                }
            }
        }
    }
    for (_, mut t) in local.borrow_mut().drain() {
        t.release();
    }
}

// ---------------------------------------------------------------------------------------
// The run worker's side.
// ---------------------------------------------------------------------------------------

/// W0's handle on the parallel run: its own engine (for guest memory and forwarded calls - it
/// runs no thread), the shared state, and the worker objects.
pub struct SmpRun {
    sh: Arc<Shared>,
    engine: BrowserEngine,
    /// Frames the run worker has taken for presenting. Under [`overlap`] the guest may already
    /// be a frame past this; everything the live loop keys on a frame number (its next target,
    /// its shots, its recipe cues) means the frame being PRESENTED, which is this.
    consumed: u64,
    _workers: Vec<web_sys::Worker>,
    _listeners: Vec<Closure<dyn FnMut(web_sys::MessageEvent)>>,
    /// Kept alive for the same reason: a guest worker that fails to START reports it here.
    _error_listeners: Vec<Closure<dyn FnMut(JsValue)>>,
}

impl SmpRun {
    /// Stand the parallel run up: bind the main thread (its `entries` in order) to a worker,
    /// start the guest workers over this bundle's shared memory, and wait until every one of
    /// them is running its loop.
    pub async fn start(
        engine: BrowserEngine,
        mem_pages: u32,
        dirty_off: Option<u64>,
        host_off: u32,
        entries: &[u32],
        main_sp: u32,
        audio_ring: &JsValue,
    ) -> Result<SmpRun, JsValue> {
        if !crate::host_memory_is_shared() {
            return Err(JsValue::from_str(
                "VITASLOP_SMP needs the wasm-threads bundle (a shared memory); this one is \
                 single-threaded - rebuild without --no-threads",
            ));
        }
        let workers = worker_count();
        let host = engine.host().clone();
        // Guest threads run while the run worker writes render targets back: check the whole
        // region, not its first row (see the function).
        vitaslop_runtime::rtt_writeback::set_whole_region_guard(true);
        // The flip's geometry resolve on the presenter instead of the flipping thread.
        vitaslop_runtime::host::set_async_flip_resolve(async_resolve());
        // The mirror block's address, told to the host once - `SchedCore::new` does the same.
        if let Some(base) = engine.mirror_base() {
            lock_host(&host).set_mirror_base(base);
        }
        // The title's storage ring, if its reader has one: every guest worker then reads files
        // itself (`opfs::install_worker_reader`) and the file families are not forwarded.
        let storage = crate::opfs::storage_share();
        let owner_only: Vec<bool> = {
            let h = lock_host(&host);
            (0u32..)
                .map_while(|sel| {
                    h.import_at(sel)
                        .map(|(_, nid)| vitaslop_runtime::vita::smp_forwarded(nid, storage.is_some()))
                })
                .collect()
        };
        let per_call: Vec<bool> = {
            let h = lock_host(&host);
            (0u32..)
                .map_while(|sel| h.import_at(sel).map(|(_, nid)| vitaslop_runtime::vita::smp_forward_per_call(nid)))
                .collect()
        };
        let sh = Arc::new(Shared {
            host,
            workers,
            state: Mutex::new(State::new(workers)),
            bells: (0..=workers).map(|_| AtomicI32::new(0)).collect(),
            rung_at: (0..=workers).map(|_| AtomicU64::new(0)).collect(),
            runnable: (0..=workers).map(|_| AtomicUsize::new(0)).collect(),
            stop: AtomicBool::new(false),
            stats: (0..=workers).map(|_| WorkerStats::default()).collect(),
            geometry: Geometry {
                base: engine.base(),
                mem_pages,
                mirror_off: engine.mirror_off(),
                dirty_off,
                host_off,
            },
            owner_only,
            per_call,
            tallies: Mutex::new((HashMap::new(), HashMap::new())),
        });
        {
            let mut st = sh.state.lock().unwrap();
            let home = st.place(workers, 0);
            st.add_thread(
                &sh,
                vitaslop_runtime::host::MAIN_THID,
                vitaslop_runtime::host::DEFAULT_THREAD_PRIORITY,
                home,
                Birth { entries: entries.iter().map(|e| e & !1).collect(), r: [0; 4], sp: main_sp },
            );
            st.publish_runnable(&sh);
        }
        SHARED
            .set(sh.clone())
            .map_err(|_| JsValue::from_str("an SMP run is already set up in this worker"))?;

        // The guest workers. Each posts `ready` once its loop is running; the run starts when
        // all of them have, so the first frame is not raced against a worker still booting.
        let mut worker_objs = Vec::new();
        let mut listeners = Vec::new();
        let mut error_listeners = Vec::new();
        let mut ready = Vec::new();
        // One guest worker, started: its handles and the promise its `ready` settles.
        let spawn = |w: usize| -> Result<(web_sys::Worker, Closure<dyn FnMut(web_sys::MessageEvent)>, Closure<dyn FnMut(JsValue)>, js_sys::Promise), JsValue> {
            let opts = web_sys::WorkerOptions::new();
            opts.set_type(web_sys::WorkerType::Module);
            opts.set_name(&format!("vitaslop-core{w}"));
            let worker = web_sys::Worker::new_with_options("./smp-worker.js", &opts)?;
            // >>> A WORKER THAT FAILS BEFORE `ready` MUST FAIL THE RUN. The start used to await
            // `ready` only, so a guest worker whose script did not load, or whose `init` threw,
            // left the page on "starting..." forever with nothing said (the first phone SMP run,
            // 2026-09-25). Every early failure now REJECTS the wait with its own text.
            let (tx, fail, rx) = futures_oneshot_or_fail();
            let tx = Rc::new(RefCell::new(Some((tx, fail))));
            let fail_start = {
                let tx = tx.clone();
                move |text: &str| {
                    if let Some((_, fail)) = tx.borrow_mut().take() {
                        let _ = fail.call1(&JsValue::UNDEFINED, &JsValue::from_str(text));
                    }
                }
            };
            let on_error = {
                let fail_start = fail_start.clone();
                Closure::wrap(Box::new(move |e: JsValue| {
                    // Handled HERE (retried below): not also re-raised on this worker's global
                    // `error`, which worker.js reports as the run's death.
                    if let Some(pd) = Reflect::get(&e, &"preventDefault".into()).ok().and_then(|f| f.dyn_into::<js_sys::Function>().ok()) {
                        let _ = pd.call0(&e);
                    }
                    let text = format!("[smp worker {w}] FAILED TO START: {}", describe_error_event(&e));
                    web_sys::console::error_1(&JsValue::from_str(&text));
                    fail_start(&text);
                }) as Box<dyn FnMut(JsValue)>)
            };
            worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));
            let on_msg = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
                let d = e.data();
                let kind = Reflect::get(&d, &JsValue::from_str("type")).ok().and_then(|v| v.as_string());
                let msg = Reflect::get(&d, &JsValue::from_str("message")).ok().and_then(|v| v.as_string());
                match kind.as_deref() {
                    Some("ready") => {
                        if let Some((tx, _)) = tx.borrow_mut().take() {
                            let _ = tx.call0(&JsValue::UNDEFINED);
                        }
                    }
                    Some("panic") => {
                        let text = format!("[smp worker {w}] PANIC: {}", msg.unwrap_or_default());
                        web_sys::console::error_1(&JsValue::from_str(&text));
                        fail_start(&text);
                        let _ = FAILED.set(text.clone());
                        if let Some(sh) = SHARED.get() {
                            sh.ring(0);
                        }
                        // Forward to the page's panic sink, as a panic here would be.
                        if let Ok(f) = Reflect::get(&js_sys::global(), &JsValue::from_str("__vitaslopPanic")) {
                            if let Ok(f) = f.dyn_into::<js_sys::Function>() {
                                let _ = f.call1(&JsValue::UNDEFINED, &JsValue::from_str(&text));
                            }
                        }
                    }
                    // A guest worker's CPU profile (`VITASLOP_JS_PROFILE`, smp-worker.js): raw to
                    // the console, where the page's worker forwards `jsprofile` lines as notes.
                    Some("jsprofile") => {
                        web_sys::console::log_1(&JsValue::from_str(&msg.unwrap_or_default()));
                    }
                    Some("error") => {
                        let text = format!("[smp worker {w}] DIED: {}", msg.unwrap_or_default());
                        web_sys::console::error_1(&JsValue::from_str(&text));
                        fail_start(&text);
                        let _ = FAILED.set(text);
                        if let Some(sh) = SHARED.get() {
                            sh.ring(0);
                        }
                    }
                    _ => {
                        crate::logging::note(&format!(
                            "[smp worker {w}] {}",
                            msg.unwrap_or_else(|| format!("{d:?}"))
                        ));
                    }
                }
            }) as Box<dyn FnMut(web_sys::MessageEvent)>);
            worker.set_onmessage(Some(on_msg.as_ref().unchecked_ref()));
            let init = Object::new();
            Reflect::set(&init, &"worker".into(), &JsValue::from_f64(w as f64))?;
            Reflect::set(&init, &"module".into(), &wasm_bindgen::module())?;
            Reflect::set(&init, &"memory".into(), &wasm_bindgen::memory())?;
            Reflect::set(&init, &"guest".into(), engine.module())?;
            Reflect::set(&init, &"audioRing".into(), audio_ring)?;
            if let Some(s) = &storage {
                Reflect::set(&init, &"storage".into(), s)?;
            }
            worker.post_message(&init)?;
            Ok((worker, on_msg, on_error, rx))
        };
        for w in 1..=workers {
            let (worker, on_msg, on_error, rx) = spawn(w)?;
            worker_objs.push(worker);
            listeners.push(on_msg);
            error_listeners.push(on_error);
            // A JsFuture NOW, not at the await: it attaches the handlers, so a worker that
            // fails while an earlier one is still being awaited is not an UNHANDLED rejection
            // (which worker.js reports as fatal before the retry below ever sees it).
            ready.push(JsFuture::from(rx));
        }
        // >>> A WORKER WHOSE SCRIPT DID NOT LOAD IS STARTED AGAIN. On the phone (PowerVR,
        // Chrome 153) roughly one run in three lost one guest worker before any of its code
        // ran - an `error` event with no message, the module graph not loaded - and the run
        // died at f0. A fresh Worker for the same core has always come up; the attempt is noted
        // so a device that needs it is visible in the dump.
        for (i, rx) in ready.into_iter().enumerate() {
            let w = i + 1;
            let mut rx = rx;
            let mut attempt = 0;
            loop {
                match (&mut rx).await {
                    Ok(_) => break,
                    Err(e) if attempt < 3 => {
                        attempt += 1;
                        let why = e.as_string().unwrap_or_else(|| format!("{e:?}"));
                        crate::logging::note(&format!("[smp worker {w}] start attempt {attempt} failed ({why}); starting it again"));
                        worker_objs[i].terminate();
                        let (worker, on_msg, on_error, r) = spawn(w)?;
                        worker_objs[i] = worker;
                        listeners[i] = on_msg;
                        error_listeners[i] = on_error;
                        rx = JsFuture::from(r);
                    }
                    Err(e) => return Err(e),
                }
            }
        }
        // >>> THE RESOLVER WORKER, under the async resolve: a core of its own for the flip's
        // geometry read, which otherwise sits on the game's serial main -> render chain (the
        // phone proxy put it at ~40% of the render thread). It posts no `ready`: it is attached
        // below only once its loop can be reached, and until then a flip leaves the boundary
        // for the run worker exactly as without it.
        if async_resolve() {
            let opts = web_sys::WorkerOptions::new();
            opts.set_type(web_sys::WorkerType::Module);
            opts.set_name("vitaslop-resolver");
            let worker = web_sys::Worker::new_with_options("./smp-worker.js", &opts)?;
            let on_msg = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
                let d = e.data();
                let kind = Reflect::get(&d, &JsValue::from_str("type")).ok().and_then(|v| v.as_string());
                let msg = Reflect::get(&d, &JsValue::from_str("message")).ok().and_then(|v| v.as_string());
                match kind.as_deref() {
                    Some("ready") => vitaslop_runtime::host::set_resolver_attached(true),
                    Some("panic") | Some("error") => {
                        // A dead resolver would hold every sync point for ever: fail loudly.
                        vitaslop_runtime::host::set_resolver_attached(false);
                        let text = format!("[smp resolver] DIED: {}", msg.unwrap_or_default());
                        web_sys::console::error_1(&JsValue::from_str(&text));
                        let _ = FAILED.set(text);
                        if let Some(sh) = SHARED.get() {
                            sh.ring(0);
                        }
                    }
                    _ => crate::logging::note(&format!("[smp resolver] {}", msg.unwrap_or_else(|| format!("{d:?}")))),
                }
            }) as Box<dyn FnMut(web_sys::MessageEvent)>);
            worker.set_onmessage(Some(on_msg.as_ref().unchecked_ref()));
            let init = Object::new();
            Reflect::set(&init, &"role".into(), &"resolver".into())?;
            Reflect::set(&init, &"module".into(), &wasm_bindgen::module())?;
            Reflect::set(&init, &"memory".into(), &wasm_bindgen::memory())?;
            worker.post_message(&init)?;
            worker_objs.push(worker);
            listeners.push(on_msg);
        }
        // The guest-function SAMPLER, only under `VITASLOP_GUEST_PROF` - see [`guest_prof`].
        if guest_prof().is_some() {
            let opts = web_sys::WorkerOptions::new();
            opts.set_type(web_sys::WorkerType::Module);
            opts.set_name("vitaslop-sampler");
            let worker = web_sys::Worker::new_with_options("./smp-worker.js", &opts)?;
            let on_msg = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
                let d = e.data();
                let msg = Reflect::get(&d, &JsValue::from_str("message")).ok().and_then(|v| v.as_string()).unwrap_or_default();
                // One console line per report line, in THIS worker, whose console the page
                // forwards (`guestprof` prefix).
                for line in msg.split('\n').filter(|l| l.starts_with("guestprof")) {
                    web_sys::console::log_1(&JsValue::from_str(line));
                }
            }) as Box<dyn FnMut(web_sys::MessageEvent)>);
            worker.set_onmessage(Some(on_msg.as_ref().unchecked_ref()));
            let init = Object::new();
            Reflect::set(&init, &"role".into(), &"sampler".into())?;
            Reflect::set(&init, &"module".into(), &wasm_bindgen::module())?;
            Reflect::set(&init, &"memory".into(), &wasm_bindgen::memory())?;
            worker.post_message(&init)?;
            worker_objs.push(worker);
            listeners.push(on_msg);
        }
        tracing::info!(
            target: "vitaslop::smp",
            "SMP run: {workers} guest worker(s) up; the run worker hosts no guest thread"
        );
        // This engine expires wall parks (drain + idle workers) - `VitaState::wall_park`.
        vitaslop_runtime::host::set_wall_parks_served(true);
        Ok(SmpRun {
            sh,
            engine,
            consumed: 0,
            _workers: worker_objs,
            _listeners: listeners,
            _error_listeners: error_listeners,
        })
    }

    /// The game time frame `frame` took: the virtual clock at its flip minus the clock at the
    /// flip before it. `None` when either flip is no longer remembered.
    pub fn frame_advance_us(&self, frame: u64) -> Option<u64> {
        let st = self.sh.state.lock().unwrap();
        let at = |f: u64| st.flip_clock.iter().find(|(n, _)| *n == f).map(|(_, us)| *us);
        Some(at(frame)?.saturating_sub(at(frame.checked_sub(1)?)?))
    }

    pub fn host(&self) -> &Host {
        &self.sh.host
    }

    /// The frame the live loop is at: the last one taken for presenting under [`overlap`],
    /// else the last one flipped (the same thing, since nothing runs past it there).
    pub fn frames(&self) -> u64 {
        if overlap() {
            self.consumed
        } else {
            self.sh.state.lock().unwrap().frames
        }
    }

    pub fn read_guest(&self, addr: u32, out: &mut [u8]) -> bool {
        self.engine.read_mem(addr, out)
    }

    pub fn write_guest(&self, addr: u32, bytes: &[u8]) {
        self.engine.write_guest(addr, bytes)
    }

    pub fn thread_census(&self) -> (usize, usize) {
        let st = self.sh.state.lock().unwrap();
        (st.live.len(), st.threads.len() - st.live.len())
    }

    pub fn fuel_report(&self) -> (u64, u64, u64) {
        let st = self.sh.state.lock().unwrap();
        (st.fuel_total, st.fuel_samples, st.fuel_max)
    }

    pub fn unbilled_report(&self) -> (u64, u64) {
        let st = self.sh.state.lock().unwrap();
        (st.fuel_unreported, st.fuel_idle)
    }

    /// One line for the panel: per worker, its share of wall clock busy in guest code, its
    /// resumes and the threads bound to it; then the host lock's contention and the forwards.
    pub fn report(&self, since: &mut SmpSnapshot) -> String {
        use std::fmt::Write;
        let mut st = self.sh.state.lock().unwrap();
        let forwarded_by = std::mem::take(&mut st.forwarded_by);
        let st = st;
        let mut s = format!("SMP {} worker(s):", self.sh.workers);
        let mut snap = SmpSnapshot::default();
        for w in 1..=self.sh.workers {
            let busy = self.sh.stats[w].busy_us.load(Ordering::Relaxed);
            let wait = self.sh.stats[w].wait_us.load(Ordering::Relaxed);
            let res = self.sh.stats[w].resumes.load(Ordering::Relaxed);
            let qua = self.sh.stats[w].quanta.load(Ordering::Relaxed);
            let (b0, w0, r0) = since.per.get(w - 1).copied().unwrap_or((0, 0, 0));
            let q0 = since.quanta.get(w - 1).copied().unwrap_or(0);
            let sp = self.sh.stats[w].spins.load(Ordering::Relaxed);
            let sc = self.sh.stats[w].spin_caught.load(Ordering::Relaxed);
            let (sp0, sc0) = since.spins.get(w - 1).copied().unwrap_or((0, 0));
            snap.spins.push((sp, sc));
            let (db, dw, dr) = (busy - b0, wait - w0, res - r0);
            snap.quanta.push(qua);
            let total = (db + dw).max(1);
            let threads: Vec<String> = st
                .live
                .iter()
                .filter(|&&i| st.threads[i].home == w)
                .map(|&i| format!("{:#x}", st.threads[i].thid))
                .collect();
            let _ = write!(
                s,
                " | w{w} busy {:.0}% ({dr} resumes, {} whole-quantum, spun {} caught {}) [{}]",
                db as f64 * 100.0 / total as f64,
                qua - q0,
                sp - sp0,
                sc - sc0,
                threads.join(" ")
            );
            snap.per.push((busy, wait, res));
        }
        let taken = LOCK_TAKEN.load(Ordering::Relaxed);
        let cont = LOCK_CONTENDED.load(Ordering::Relaxed);
        let (dt, dc) = (taken - since.lock.0, cont - since.lock.1);
        let _ = write!(
            s,
            " | host lock {dc} of {dt} contended ({:.1}%) | forwarded {} | idle jumps {}",
            dc as f64 * 100.0 / dt.max(1) as f64,
            st.forwarded - since.forwarded,
            st.idle_jumps - since.idle_jumps,
        );
        {
            let mut v: Vec<(u32, u64)> = forwarded_by.into_iter().collect();
            v.sort_by_key(|x| std::cmp::Reverse(x.1));
            let host = lock_host(&self.sh.host);
            let names: Vec<String> = v
                .iter()
                .take(6)
                .map(|(sel, n)| {
                    let name = host
                        .import_at(*sel)
                        .map_or_else(|| format!("sel{sel}"), |(_, nid)| vitaslop_runtime::nid::name(nid).to_string());
                    format!("{name} x{n}")
                })
                .collect();
            let _ = write!(s, " [{}]", names.join(", "));
        }
        let h = st.handoffs;
        let p = since.handoffs;
        let n = h.n - p.n;
        let _ = write!(
            s,
            " | handoffs {n}, mean {:.3} ms, max-ever {:.2} ms, <50us {} <200us {} <1ms {} <5ms {} >=5ms {}",
            (h.sum_ms - p.sum_ms) / n.max(1) as f64,
            h.max_ms,
            h.buckets[0] - p.buckets[0],
            h.buckets[1] - p.buckets[1],
            h.buckets[2] - p.buckets[2],
            h.buckets[3] - p.buckets[3],
            h.buckets[4] - p.buckets[4],
        );
        let _ = write!(
            s,
            " | run worker: forwards {:.1} ms, early {:.1} ms, stop-the-world tail {:.1} ms",
            st.w0_forward_ms - since.w0.0,
            st.w0_early_ms - since.w0.1,
            st.w0_tail_ms - since.w0.2,
        );
        let _ = write!(
            s,
            " | preempts: gate {} prio {}",
            st.preempt_gate - since.preempts.0,
            st.preempt_prio - since.preempts.1,
        );
        snap.preempts = (st.preempt_gate, st.preempt_prio);
        // The busiest threads over the run (or since VITASLOP_CPU_SHARE_FROM), with where they
        // live and how often they spun: the table that says which thread a worker's busy time
        // actually was, and whether it was work or a spin.
        let total: u64 = st.threads.iter().map(|t| t.retired_total).sum::<u64>().max(1);
        let mut rows: Vec<&Slot> = st.threads.iter().filter(|t| t.retired_total > 0).collect();
        rows.sort_by_key(|t| std::cmp::Reverse(t.retired_total));
        // Per frame since the counters were last cleared (`VITASLOP_CPU_SHARE_FROM`, else the
        // run): wall ms RUNNING and ms RUNNABLE-BUT-WAITING for its worker.
        let per = st.frames.saturating_sub(vitaslop_runtime::sched::cpu_share_from().unwrap_or(0)).max(1) as f64;
        let _ = write!(s, " | threads by work:");
        for t in rows.iter().take(8) {
            let _ = write!(
                s,
                " {:#x}@w{} p{:#x} {:.1}% ({} picks, {} spun; run {:.2} queued {:.2} ms/f, {} waits>=5ms)",
                t.thid,
                t.home,
                t.priority,
                t.retired_total as f64 * 100.0 / total as f64,
                t.picks,
                t.quanta,
                t.run_ms / per,
                t.queued_ms / per,
                t.queued_long
            );
        }
        // What the busiest callers call, this window: per thread, its fuel yields and its top
        // three NIDs. Drained, so each line is a window.
        {
            let (calls, fuel) = std::mem::take(&mut *self.sh.tallies.lock().unwrap());
            let mut per: HashMap<i32, Vec<(u32, u64)>> = HashMap::new();
            for ((thid, sel), n) in calls {
                per.entry(thid).or_default().push((sel, n));
            }
            let mut threads: Vec<(i32, u64, u64)> = per
                .iter()
                .map(|(t, v)| (*t, v.iter().map(|x| x.1).sum::<u64>(), fuel.get(t).copied().unwrap_or(0)))
                .collect();
            for (t, f) in &fuel {
                if !per.contains_key(t) {
                    threads.push((*t, 0, *f));
                }
            }
            threads.sort_by_key(|x| std::cmp::Reverse(x.1 + x.2 * 1000));
            let host = lock_host(&self.sh.host);
            let _ = write!(s, " | calls by thread:");
            for (thid, total, fy) in threads.iter().take(8) {
                let mut v = per.get(thid).cloned().unwrap_or_default();
                v.sort_by_key(|x| std::cmp::Reverse(x.1));
                let top: Vec<String> = v
                    .iter()
                    .take(3)
                    .map(|(sel, n)| {
                        let name = host
                            .import_at(*sel)
                            .map_or_else(|| format!("sel{sel}"), |(_, nid)| vitaslop_runtime::nid::name(nid).to_string());
                        format!("{name} x{n}")
                    })
                    .collect();
                let _ = write!(s, " {thid:#x}: {total} calls, {fy} fuel-yields [{}];", top.join(", "));
            }
        }
        snap.lock = (taken, cont);
        snap.forwarded = st.forwarded;
        snap.idle_jumps = st.idle_jumps;
        snap.handoffs = h;
        snap.w0 = (st.w0_forward_ms, st.w0_early_ms, st.w0_tail_ms);
        *since = snap;
        s
    }

    /// The presenter's half of the async flip resolve (`VITASLOP_SMP_ASYNC_RESOLVE=1`): read
    /// the flipped frame's geometry HERE, on the run worker, instead of on the guest render
    /// thread at its flip - which is the guest's critical path on a phone (ovl26b: the frame
    /// is main THEN render, and the render thread's flip-time resolve is ~9 ms of it there).
    /// Called before the live loop takes the frame's scenes, which leaves pending ones behind.
    pub fn resolve_flipped(&self) {
        if async_resolve() {
            self.engine.resolve_flipped_geometry();
        }
    }

    /// Run the guest to `target` display frames: open the gate, serve the workers' forwards and
    /// completions, take the idle clock steps, and return once the target frame has flipped AND
    /// every worker has reached a switch point (see the module docs for why the present waits).
    pub async fn run_frames(
        &mut self,
        target: u64,
        progress: &mut dyn FnMut(u64),
        completer: Option<&mut (dyn EarlyCompleter + 'static)>,
    ) -> RunReport {
        let t0 = now_ms();
        let r = self.run_frames_inner(target, progress, completer).await;
        trace_w0(b'r', t0, now_ms());
        r
    }

    async fn run_frames_inner(
        &mut self,
        target: u64,
        progress: &mut dyn FnMut(u64),
        mut completer: Option<&mut (dyn EarlyCompleter + 'static)>,
    ) -> RunReport {
        // Under `overlap` the gate stands ONE frame ahead of the target: the guest may build the
        // next frame while this one is presented, and stops (preempted at that flip) only if it
        // gets a whole frame ahead of the present.
        let ahead = u64::from(overlap());
        let (flipped, gate_now) = {
            let mut st = self.sh.state.lock().unwrap();
            // The hold ends here: close it off on the wall floor so its first tick after the
            // gate opens does not count the hold (see `VitaState::wall_floor_tick`).
            if (st.frames >= st.gate || st.paused) && wall_floor(st.frames) {
                lock_host(&self.sh.host).state.wall_floor_tick(abs_ms(), true);
            }
            st.gate = st.gate.max(target + ahead);
            (st.frames, st.gate)
        };
        // `G target gate t`: where W0 opened the gate.
        trace_ev(flipped, TraceEv::Gate { target, gate: gate_now, t: abs_ms() });
        trace_dump(flipped);
        self.sh.ring_all();
        let mut rounds = 0u64;
        // When the target frame was first seen flipped: from there to the return is the
        // stop-the-world tail.
        let mut reached_at: Option<f64> = None;
        loop {
            rounds += 1;
            if rounds % 64 == 0 {
                progress(rounds);
            }
            let bell = self.sh.bells[0].load(Ordering::SeqCst);
            if let Some(why) = FAILED.get() {
                return RunReport::Error(why.clone());
            }
            // Forwarded host calls, in arrival order, and the small-target completions a thread
            // is parked on.
            let tv = now_ms();
            self.serve_pending(completer.as_mut().map(|c| &mut **c)).await;
            trace_w0(b'v', tv, now_ms());
            // `l` (waiting for the state lock) and `i` (an idle clock step) are TIMED under the
            // lock and EMITTED after it drops: `trace_w0` takes this same lock for the frame number,
            // and calling it while `st` is held deadlocked the run at f0.
            let tl = now_ms();
            let mut idle_span: Option<(f64, f64)> = None;
            let got_lock_at;
            {
                let mut st = self.sh.state.lock().unwrap();
                got_lock_at = now_ms();
                if let Some(v) = st.verdict.clone() {
                    return v;
                }
                if st.frames >= target && reached_at.is_none() {
                    reached_at = Some(now_ms());
                }
                // Overlapped: the frame has flipped, so its scenes are complete - present it
                // now, whatever the workers are doing on the next one.
                if ahead == 1 && st.frames >= target {
                    self.consumed = target;
                    return RunReport::FramesReached(target);
                }
                if st.frames >= target && st.running == 0 && st.forwards.is_empty() {
                    if let Some(t) = reached_at {
                        st.w0_tail_ms += now_ms() - t;
                    }
                    return RunReport::FramesReached(st.frames);
                }
                if st.frames < st.gate && st.globally_idle() {
                    let ti = now_ms();
                    st.idle_step(&self.sh, &self.engine);
                    idle_span = Some((ti, now_ms()));
                }
            }
            trace_w0(b'l', tl, got_lock_at);
            if let Some((a, b)) = idle_span {
                trace_w0(b'i', a, b);
                continue;
            }
            // `z`: this worker asleep on its bell; the `V` inside one and the span's end are the
            // flip-to-notice latency.
            let tz = now_ms();
            wait_async(&self.sh.bells[0], bell, 4.0).await;
            trace_w0(b'z', tz, now_ms());
        }
    }

    /// The run worker's pacing wait under SMP: `ms` of wall clock spent SERVING the guest
    /// workers - their forwarded calls and small-target completions - while the event loop
    /// turns, instead of one blocking sleep.
    ///
    /// # Why the one-worker sleep is wrong here
    /// The live loop sleeps between frames with `Atomics.wait`, which blocks this thread: no
    /// task runs, so no GPU work-done callback, no decoder output, no message. On the one-worker
    /// engine nothing else is running then, and the next `run_frames` awaits plenty. Under SMP
    /// the guest workers keep running (always under [`overlap`], and on their gate's tail
    /// otherwise), and they need this worker: a forwarded call parks its thread until it is
    /// served, and the present's own GPU callbacks go stale. MEASURED with the overlapped
    /// present (`ov25c`): the GPU-lag rule declined 74 of ~80 presents, because every
    /// work-done callback waited behind a blocking sleep.
    pub async fn serve_for(&mut self, ms: f64, mut completer: Option<&mut (dyn EarlyCompleter + 'static)>) {
        let end = now_ms() + ms.max(0.0);
        loop {
            let bell = self.sh.bells[0].load(Ordering::SeqCst);
            self.serve_pending(completer.as_mut().map(|c| &mut **c)).await;
            let left = end - now_ms();
            if left <= 0.0 {
                break;
            }
            wait_async(&self.sh.bells[0], bell, left.min(4.0)).await;
        }
    }

    /// Serve every forwarded call and early completion queued right now.
    async fn serve_pending(&mut self, mut completer: Option<&mut (dyn EarlyCompleter + 'static)>) {
        loop {
            let next = {
                let mut st = self.sh.state.lock().unwrap();
                st.forwards.pop_front().map(|i| {
                    let t = &mut st.threads[i];
                    (i, t.thid, t.forward.take())
                })
            };
            let Some((i, thid, Some(req))) = next else { break };
            let t0 = now_ms();
            self.serve_forward(i, thid, req);
            self.sh.state.lock().unwrap().w0_forward_ms += now_ms() - t0;
            trace_w0(b'F', t0, now_ms());
        }
        loop {
            let next = {
                let mut st = self.sh.state.lock().unwrap();
                let e = st.early.pop_front();
                if e.is_some() {
                    st.early_serving += 1;
                }
                e
            };
            let Some((thid, start, n)) = next else { break };
            let t0 = now_ms();
            // A sync point may have handed its geometry to the resolver worker and parked its
            // thread (see `VitaState::queue_sync_resolve`): wait for that read WITHOUT the host
            // lock, then apply it under the lock, before anything renders or the thread wakes.
            self.engine.resolve_flipped_geometry();
            let host = self.sh.host.clone();
            // Paused for the whole of it unless `VITASLOP_SMP_EARLY_PAUSE_ALL=0` (see
            // `early_pause_all`), which renders and reads back with the guest running.
            let pause_all = early_pause_all();
            if pause_all {
                self.pause_guest().await;
            }
            let rendered =
                browser_sched::render_early_batch(&host, completer.as_mut().map(|c| &mut **c), start, n).await;
            // The completion WRITES the rendered pixels into guest memory - with the guest
            // stopped, as the one-worker engine always is here (see `pause_guest`).
            if !pause_all {
                self.pause_guest().await;
            }
            browser_sched::apply_early_batch(&host, &self.engine, thid, start, n, rendered);
            self.resume_guest();
            {
                let mut st = self.sh.state.lock().unwrap();
                st.early_serving -= 1;
                st.drain(&self.sh, &self.engine);
                st.w0_early_ms += now_ms() - t0;
            }
            trace_w0(b'e', t0, now_ms());
        }
    }

    /// Dispatch a forwarded host call here, on the worker that owns its JavaScript, and put the
    /// thread back where the outcome says - `SchedCore::on_suspended` for a call that ran here.
    fn serve_forward(&self, i: usize, thid: i32, req: ForwardReq) {
        let mut regs = req.regs;
        let mut vfp = req.vfp;
        let (outcome, early) = self.engine.dispatch_forwarded(thid, req.selector, &mut regs, &mut vfp);
        let mut st = self.sh.state.lock().unwrap();
        st.threads[i].patch = Some(RegPatch { before: (req.regs, req.vfp), regs, vfp });
        if let Some(e) = early {
            st.early.push_back(e);
        }
        let home = st.threads[i].home;
        match outcome {
            SvcOutcome::Continue | SvcOutcome::Reschedule => {
                st.threads[i].state = SState::Runnable;
                st.threads[i].since = abs_ms();
                st.threads[i].ready_at = abs_ms();
                if tracing() {
                    trace_ev(st.frames, TraceEv::Wake { thid, t: abs_ms(), by: 0, wby: thid, line: 0 });
                }
                st.preempt_for(&self.sh, i);
            }
            SvcOutcome::Block => st.block_or_consume_token(i),
            SvcOutcome::Flip => st.flip(&self.sh, i),
            // The thread's stack stays parked for ever; its worker drops the instance.
            SvcOutcome::ThreadExit => st.finish(&self.sh, i, FiberEnd::ThreadExit(regs[0])),
            SvcOutcome::Halt => st.finish(&self.sh, i, FiberEnd::ProcessHalt(regs[0])),
            SvcOutcome::Fatal(m) => st.finish(&self.sh, i, FiberEnd::Error(m)),
        }
        st.drain(&self.sh, &self.engine);
        self.sh.ring(home);
    }

    /// >>> STOP THE GUEST WORKERS, for a write into guest memory the guest could be writing at
    /// the same time: every running thread is asked for its CPU (the preempt words) and no
    /// worker picks another until [`Self::resume_guest`]. Returns once none is running.
    ///
    /// Used around render-target WRITE-BACKS, which are the run worker writing pixels into
    /// memory a title may have recycled. The whole-region check that refuses a recycled target
    /// and the write that follows it must not have a guest store between them; on the one-worker
    /// engine nothing runs then by construction, and this restores that for the few writes a
    /// frame that need it - the rest of the frame keeps its parallelism.
    pub async fn pause_guest(&mut self) {
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
            let bell = self.sh.bells[0].load(Ordering::SeqCst);
            if self.sh.state.lock().unwrap().running == 0 {
                return;
            }
            if FAILED.get().is_some() {
                return;
            }
            wait_async(&self.sh.bells[0], bell, 4.0).await;
        }
    }

    /// Let the guest workers run again after [`Self::pause_guest`].
    pub fn resume_guest(&mut self) {
        self.sh.state.lock().unwrap().paused = false;
        self.sh.ring_all();
    }

    /// End the run: every worker leaves its loop.
    pub fn stop(&self) {
        self.sh.stop.store(true, Ordering::SeqCst);
        vitaslop_runtime::host::set_resolver_attached(false);
        lock_host(&self.sh.host).async_resolve_handle().stop();
        self.sh.ring_all();
    }
}

impl Drop for SmpRun {
    fn drop(&mut self) {
        vitaslop_runtime::host::set_wall_parks_served(false);
        self.stop();
        for w in &self._workers {
            w.terminate();
        }
    }
}

/// The counters as of the last panel line, so the next one reports a window.
#[derive(Default)]
pub struct SmpSnapshot {
    per: Vec<(u64, u64, u64)>,
    quanta: Vec<u64>,
    spins: Vec<(u64, u64)>,
    lock: (u64, u64),
    forwarded: u64,
    idle_jumps: u64,
    handoffs: Handoffs,
    w0: (f64, f64, f64),
    preempts: (u64, u64),
}

/// A promise and the function that resolves it.
fn futures_oneshot() -> (js_sys::Function, js_sys::Promise) {
    let (resolve, _, p) = futures_oneshot_or_fail();
    (resolve, p)
}

/// [`futures_oneshot`] that can also FAIL: `(resolve, reject, promise)`.
fn futures_oneshot_or_fail() -> (js_sys::Function, js_sys::Function, js_sys::Promise) {
    let mut ends = None;
    let p = js_sys::Promise::new(&mut |res, rej| ends = Some((res, rej)));
    let (res, rej) = ends.expect("the executor runs synchronously");
    (res, rej, p)
}

/// Wait on `bell` (while it still reads `seen`) WITHOUT blocking the event loop: the run worker
/// has WebCodecs callbacks and GPU promises to deliver while the guest workers run. Falls back
/// to one event-loop turn where `Atomics.waitAsync` is missing.
thread_local! {
    /// Wall ms this (run) worker has spent inside [`wait_async`] - i.e. YIELDED to its event
    /// loop, where GPU callbacks are delivered. Added to the live loop's own tick sleeps by
    /// `worker_yielded_ms` (lib.rs), which ages GPU submits for the lag bound.
    static YIELDED_MS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// See [`YIELDED_MS`].
///
/// >>> WITHOUT THIS THE GPU LAG BOUND NEVER FIRES UNDER SMP. The run worker spends most of a
/// frame awaiting the guest's flip inside `run_frames` - yielded, callbacks flowing - but only
/// the live loop's between-frame sleep was counted, 0.6 ms a frame on the phone (2026-09-25,
/// MLB, period 45.6 ms). A submit then aged ~0.6 ms a frame against a 150 ms bound, the bound
/// never bound, and work-done latency and the render-target write-back age both reached
/// ~780 ms - old enough to wash MLB's auto-exposure.
pub(crate) fn yielded_ms() -> f64 {
    YIELDED_MS.with(|y| y.get())
}

async fn wait_async(bell: &AtomicI32, seen: i32, timeout_ms: f64) {
    let t0 = now_ms();
    wait_async_inner(bell, seen, timeout_ms).await;
    let dt = (now_ms() - t0).max(0.0);
    YIELDED_MS.with(|y| y.set(y.get() + dt));
}

async fn wait_async_inner(bell: &AtomicI32, seen: i32, timeout_ms: f64) {
    thread_local! {
        static VIEW: RefCell<Option<(js_sys::Int32Array, u32)>> = const { RefCell::new(None) };
        static WAIT_ASYNC: Option<js_sys::Function> = Reflect::get(&js_sys::global(), &"Atomics".into())
            .ok()
            .and_then(|a| Reflect::get(&a, &"waitAsync".into()).ok())
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok());
    }
    let Some(f) = WAIT_ASYNC.with(|f| f.clone()) else {
        browser_sched::event_loop_turn().await;
        return;
    };
    // The view is rebuilt when the memory has grown: a grown shared memory hands out a new,
    // longer buffer object, and an old view would not reach a bell allocated past its end.
    let buffer = wasm_bindgen::memory().unchecked_into::<WebAssembly::Memory>().buffer();
    let len = js_sys::Reflect::get(&buffer, &"byteLength".into()).ok().and_then(|v| v.as_f64()).unwrap_or(0.0) as u32;
    let view = VIEW.with(|v| {
        let mut v = v.borrow_mut();
        if v.as_ref().is_none_or(|(_, l)| *l != len) {
            *v = Some((js_sys::Int32Array::new(&buffer), len));
        }
        v.as_ref().unwrap().0.clone()
    });
    let index = (bell.as_ptr() as usize / 4) as u32;
    let args = js_sys::Array::of4(&view, &JsValue::from(index), &JsValue::from(seen), &JsValue::from(timeout_ms));
    let Ok(res) = f.apply(&JsValue::NULL, &args) else {
        browser_sched::event_loop_turn().await;
        return;
    };
    let is_async = Reflect::get(&res, &"async".into()).ok().and_then(|v| v.as_bool()).unwrap_or(false);
    if is_async {
        if let Ok(p) = Reflect::get(&res, &"value".into()) {
            let _ = JsFuture::from(js_sys::Promise::from(p)).await;
        }
    } else {
        // Already changed (or timed out at once): still give the event loop its turn, since a
        // decoder callback may be what the run is waiting for.
        browser_sched::event_loop_turn().await;
    }
}

