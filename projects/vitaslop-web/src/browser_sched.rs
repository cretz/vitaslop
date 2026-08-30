//! The browser preemptive scheduler: the JSPI implementation of the engine-agnostic
//! [`SchedCore`](vitaslop_runtime::sched::SchedCore). It is the browser twin of
//! `vitaslop-native`'s wasmtime `ThreadedScheduler`, and shares the exact scheduling
//! policy - only the "resume a guest thread to its next switch point" primitive is
//! reimplemented here on the browser's own WebAssembly engine.
//!
//! # One worker, instance-per-thread, one shared memory
//! Every guest thread is its own `WebAssembly.Instance` (its ARM register file lives
//! in wasm globals, which are per-instance, so each thread's registers are naturally
//! private), and all instances import ONE shared linear memory (the transpiler emits
//! `env.memory` when `import_memory` is set) - one guest address space, private
//! registers, exactly the native model. Everything runs on one thread: the single
//! owner of [`VitaEnv`], the scheduler, and all guest instances. Because the host
//! (`VitaEnv`) lives here too, a guest host call needs no cross-thread hop.
//!
//! # JSPI is how a mid-stack thread suspends
//! A guest thread blocks deep inside its wasm call stack (inside game logic that
//! called a blocking kernel primitive). To switch away we must suspend that stack.
//! The browser has no wasmtime fibers, so we use **JSPI**: `env.import` is a
//! `WebAssembly.Suspending` function and each thread's entry is called through
//! `WebAssembly.promising`. A host call that must block returns a *pending Promise*,
//! which suspends the guest stack and returns control to the async scheduler loop; a
//! host call that continues returns a plain value, which does NOT suspend (so the
//! common case stays cheap). Resuming a suspended thread resolves that Promise.
//!
//! Because a JSPI resume is inherently asynchronous (it unwinds to the event loop),
//! the run loop here is `async` and cannot reuse the synchronous
//! [`Scheduler`](vitaslop_runtime::sched::Scheduler) loop - but it composes the same
//! [`SchedCore`] helpers (priority pick, frame counting, spawn/wake drain,
//! deadlock/timed-wait, verdict), so the discipline is identical to native.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use js_sys::{Array, Function, Object, Promise, Reflect, Uint8Array, WebAssembly};
use vitaslop_runtime::sched::{
    FiberEnd, GuestEngine, IdleStep, RunReport, SchedCore, Stop, ThreadHandle, ThreadStep,
};
use vitaslop_runtime::{GuestMemory, ImportDispatch, Reentry, SvcOutcome, VitaEnv, VFP_ARG_COUNT};
use vitaslop_transpiler::abi;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

/// Host-call accounting for the live run: how many the guest has made, and how long they
/// took inside the host.
///
/// # Why the browser needs this and native does not
/// A browser frame is one opaque `await`: while a guest entry runs, nothing on the Rust
/// side gets control except this import closure, so it is the ONLY place that can say
/// what a long frame is spending its time on. Without it the two candidate explanations -
/// translated guest code being slow, and the host-call seam being slow - produce exactly
/// the same observation (a frame counter that does not move while Chrome burns CPU), and
/// they need opposite fixes. The counter is a load and an add per call; the clock reads
/// are two boundary crossings, which is noise beside a call that crosses the boundary
/// dozens of times anyway.
/// # Why the TIMING half is opt-in
/// `performance.now()` is a call out to JS, and the split needs four of them per host
/// call. At half a million host calls a second that is not a rounding error - measured
/// here, it was most of what remained after the marshalling fix, so leaving it on would
/// have made the fix look smaller than it is. Native's `perf` module makes the same
/// choice for the same reason. The CALL COUNT stays unconditional: it is one add, and it
/// is what tells a stuck frame from a busy one.
mod hostcalls {
    use std::cell::Cell;

    thread_local! {
        static CALLS: Cell<u64> = const { Cell::new(0) };
        static MS: Cell<f64> = const { Cell::new(0.0) };
        /// Of [`MS`], the part spent in the handler rather than around it.
        static DISPATCH_MS: Cell<f64> = const { Cell::new(0.0) };
        /// Whether the live loop currently has per-call timing SAMPLED on. One cell shared by
        /// the reader and the setter below - two `thread_local!` blocks declaring the same name
        /// would be two different cells, and the setter would write one nothing reads.
        static TIMING_SAMPLED: Cell<bool> = const { Cell::new(false) };
    }

    /// Whether to time each host call right now.
    ///
    /// # Why this is togglable rather than read once
    /// The knob (`VITASLOP_PERF`) pins it on for a whole run, and the timing costs about 4 us
    /// per call on a phone - roughly doubling a frame. That leaves a user of the live page with
    /// two bad options: watch at full speed and learn nothing, or profile at half speed and
    /// watch nothing. Neither is what anyone wants, and asking them to pick is the wrong
    /// question.
    ///
    /// So the live loop SAMPLES it: on for one perf window in every few, off the rest, with the
    /// numbers labelled as sampled. The average cost is the fraction of windows it runs in, and
    /// the reading is the same reading. The knob still forces it permanently on, for a harness
    /// that wants every call timed.
    pub fn timing_enabled() -> bool {
        thread_local! {
            static FORCED: bool = vitaslop_runtime::knobs::flag("VITASLOP_PERF");
        }
        FORCED.with(|on| *on) || TIMING_SAMPLED.with(|s| s.get())
    }

    /// Turn per-call timing on or off for the next stretch of the run. See `timing_enabled`.
    pub fn set_timing_sampling(on: bool) {
        TIMING_SAMPLED.with(|s| s.set(on));
    }

    /// Host calls one guest thread may make before the browser preempts it
    /// (`VITASLOP_BROWSER_QUANTUM_CALLS`, 0 disables preemption entirely).
    ///
    /// # This is a FAIRNESS backstop, NOT the game clock's driver
    /// It used to be both, and that was the bug. A preemption advances the virtual game
    /// clock (`SchedCore::on_quantum` -> `charge_cpu_quantum`), so whichever mechanism
    /// preempts most sets the rate at which game time passes. Host-call count measures
    /// host-call DENSITY, which differs by an order of magnitude between a menu and a
    /// race, so a clock driven by it is calibrated for one screen and wrong on the rest.
    ///
    /// The previous default of 1,300 came from matching native's clock over this title's
    /// LOADER - and the comment here already warned that "a ratio measured on one
    /// workload does not transfer to another", which is exactly what then happened: on
    /// the front end the same setting ran the clock 5.1x slow and stranded a frame-keyed
    /// recipe on a timed screen for 14,000 frames.
    ///
    /// The clock is now driven by [`super::fuel_interval`], which counts guest WORK the
    /// way native's engine fuel does. This stays as a liveness backstop for the one case
    /// fuel cannot see - a thread that makes host calls in a straight line with no loop
    /// back edge - so it is set far above the rate at which fuel fires and contributes
    /// almost nothing to the clock.
    pub fn quantum_calls() -> u64 {
        thread_local! {
            static N: u64 = match vitaslop_runtime::knobs::var("VITASLOP_BROWSER_QUANTUM_CALLS") {
                // A backstop while fuel drives the clock - but if fuel is switched OFF
                // this becomes the ONLY preemption again, and a 50,000-call quantum is
                // far too coarse to schedule or to time with. Fall back to the old
                // value in that case rather than silently making `FUEL=0` a much worse
                // run than it used to be.
                Err(_) if super::fuel_interval() == 0 => 1_300,
                Err(_) => 50_000,
                Ok(v) => v.parse().unwrap_or_else(|_| {
                    panic!("VITASLOP_BROWSER_QUANTUM_CALLS={v} is not a call count")
                }),
            };
        }
        N.with(|n| *n)
    }

    thread_local! {
        static SINCE_YIELD: Cell<u64> = const { Cell::new(0) };
        /// Preemptions actually taken. Reported, because "I added preemption" and "the
        /// guest is being preempted" are different claims and only the second is useful.
        static QUANTA: Cell<u64> = const { Cell::new(0) };
        /// Calls per import selector, so a runaway frame names the NID it is spinning on
        /// the way native's `perf` module does. Selectors are dense loader indices.
        static PER_SELECTOR: std::cell::RefCell<Vec<u64>> =
            std::cell::RefCell::new(vec![0; MAX_SELECTOR]);
        /// Milliseconds per import selector, filled only while per-call timing is on.
        ///
        /// # Why a COUNT by NID was not enough
        /// The panel could already say which NID is called most, and the desktop `bench`
        /// could say which one COSTS most - but the desktop is not the machine whose
        /// numbers matter, and "the crossing costs many times a native one" was an
        /// assertion this project carried from an old measurement without reproducing it.
        /// A count cannot settle it: two NIDs called equally often can differ by an order
        /// of magnitude in what they do. This is the browser's own per-NID cost.
        static PER_SELECTOR_MS: std::cell::RefCell<Vec<f64>> =
            std::cell::RefCell::new(vec![0.0; MAX_SELECTOR]);
    }

    /// Highest selector tracked per-NID; a real title imports a few hundred. Calls above
    /// it still reach the totals, only their attribution is dropped.
    const MAX_SELECTOR: usize = 4096;

    /// Preemptions taken so far.
    pub fn quanta() -> u64 {
        QUANTA.with(|c| c.get())
    }

    thread_local! {
        /// Preemptions taken because a thread ran out of SOFTWARE FUEL, as opposed to
        /// because it made its quota of host calls. Counted separately because the two
        /// answer different questions: a run whose fuel yields are zero has no guest
        /// loop spinning without host calls, and a run where they dominate has one.
        static FUEL_YIELDS: Cell<u64> = const { Cell::new(0) };
    }

    /// Count one fuel preemption.
    pub fn note_fuel_yield() {
        FUEL_YIELDS.with(|c| c.set(c.get() + 1));
    }

    /// Fuel preemptions taken so far.
    pub fn fuel_yields() -> u64 {
        FUEL_YIELDS.with(|c| c.get())
    }

    thread_local! {
        /// The RAW software-fuel counter as `fuel_used` last read it, and the smallest such
        /// reading. A preemption fires only once the counter has reached `fuel_interval()`, so
        /// a small reading here is the one thing that separates "this frame executed an
        /// enormous amount of guest work" from "this frame preempted on a counter that is not
        /// tracking work" - and those are opposite bugs that look identical from the outside.
        static RAW_FUEL: std::cell::Cell<(i64, i64)> = const { std::cell::Cell::new((0, i64::MAX)) };
    }

    /// Record one raw counter reading.
    pub fn note_raw_fuel(now: i64) {
        RAW_FUEL.with(|c| {
            let (_, lo) = c.get();
            c.set((now, lo.min(now)));
        });
    }

    /// `(the last raw counter reading, the smallest one seen)`.
    pub fn raw_fuel() -> (i64, i64) {
        RAW_FUEL.with(|c| c.get())
    }

    thread_local! {
        /// Calls per GUEST thread id. Every guest thread runs on this one worker, so
        /// "which thread is burning the calls" is not answerable from the totals - and
        /// that is exactly the question when one thread spins while another never runs.
        static PER_THID: std::cell::RefCell<std::collections::BTreeMap<i32, u64>> =
            std::cell::RefCell::new(std::collections::BTreeMap::new());
    }

    /// Count one call against `selector`, made by guest thread `thid`, and - when per-call
    /// timing is on - charge it `ms`.
    pub fn note_selector(selector: u32, thid: i32, ms: f64) {
        PER_SELECTOR.with(|v| {
            if let Some(slot) = v.borrow_mut().get_mut(selector as usize) {
                *slot += 1;
            }
        });
        if ms != 0.0 {
            PER_SELECTOR_MS.with(|v| {
                if let Some(slot) = v.borrow_mut().get_mut(selector as usize) {
                    *slot += ms;
                }
            });
        }
        PER_THID.with(|m| *m.borrow_mut().entry(thid).or_insert(0) += 1);
    }

    /// The `n` costliest selectors as `(selector, calls, ms)`, descending by ms. Empty
    /// when nothing was timed, which is the honest answer for a run without debug capture
    /// rather than a list of zeroes that reads like "these calls are free".
    pub fn top_selectors_by_ms(n: usize) -> Vec<(u32, u64, f64)> {
        let calls = PER_SELECTOR.with(|v| v.borrow().clone());
        PER_SELECTOR_MS.with(|v| {
            let ms = v.borrow();
            let mut all: Vec<(u32, u64, f64)> = ms
                .iter()
                .enumerate()
                .filter(|&(_, &m)| m > 0.0)
                .map(|(i, &m)| (i as u32, calls[i], m))
                .collect();
            all.sort_unstable_by(|a, b| b.2.total_cmp(&a.2));
            all.truncate(n);
            all
        })
    }

    /// Calls per guest thread, descending.
    pub fn by_thread() -> Vec<(i32, u64)> {
        PER_THID.with(|m| {
            let mut all: Vec<(i32, u64)> = m.borrow().iter().map(|(&t, &c)| (t, c)).collect();
            all.sort_unstable_by_key(|&(_, c)| std::cmp::Reverse(c));
            all
        })
    }

    /// The `n` most-called selectors, descending.
    pub fn top_selectors(n: usize) -> Vec<(u32, u64)> {
        PER_SELECTOR.with(|v| {
            let mut all: Vec<(u32, u64)> = v
                .borrow()
                .iter()
                .enumerate()
                .filter(|(_, c)| **c > 0)
                .map(|(i, c)| (i as u32, *c))
                .collect();
            all.sort_unstable_by_key(|&(_, c)| std::cmp::Reverse(c));
            all.truncate(n);
            all
        })
    }

    /// Count one host call against the current thread's quantum, and say whether it just
    /// expired. See [`super::preempt_note`] for why this exists.
    pub fn quantum_expired() -> bool {
        let limit = quantum_calls();
        if limit == 0 {
            return false;
        }
        let n = SINCE_YIELD.with(|c| {
            c.set(c.get() + 1);
            c.get()
        });
        if n < limit {
            return false;
        }
        reset_quantum();
        QUANTA.with(|c| c.set(c.get() + 1));
        true
    }

    /// Start a fresh quantum - called whenever the guest actually leaves the CPU, so a
    /// thread that just blocked is not preempted the instant it resumes.
    pub fn reset_quantum() {
        SINCE_YIELD.with(|c| c.set(0));
    }

    /// How many host calls between progress lines. A guest frame makes thousands, so
    /// this reports a few times a second on a healthy run and is the only sign of life
    /// during a frame that takes minutes.
    const REPORT_EVERY: u64 = 250_000;

    thread_local! {
        static LAST_REPORT_MS: Cell<f64> = const { Cell::new(0.0) };
        static LAST_REPORT_CALLS: Cell<u64> = const { Cell::new(0) };
    }

    /// Fold one completed host call into the totals, and speak up periodically.
    /// `ms` is the whole import closure; `dispatch_ms` is the handler within it.
    /// Returns true when this call completed a reporting window, so the caller - which
    /// holds the host and can turn a selector into a NID name - emits the breakdown.
    pub fn record(ms: f64, dispatch_ms: f64) -> bool {
        let calls = CALLS.with(|c| {
            c.set(c.get() + 1);
            c.get()
        });
        let total_ms = MS.with(|m| {
            m.set(m.get() + ms);
            m.get()
        });
        let dispatch_total = DISPATCH_MS.with(|m| {
            m.set(m.get() + dispatch_ms);
            m.get()
        });
        if calls % REPORT_EVERY != 0 {
            return false;
        }
        if !timing_enabled() {
            // Untimed: the count and the preemption count are all there is, and they are
            // still the difference between "this frame is grinding", "this frame is hung"
            // and "this frame is never being let go of".
            tracing::info!(
                target: "vitaslop::perf",
                "hostcalls: {calls} total, {} preemptions ({} on fuel)",
                quanta(), fuel_yields()
            );
            return true;
        }
        let wall = now();
        let (prev_wall, prev_calls) =
            (LAST_REPORT_MS.with(|c| c.get()), LAST_REPORT_CALLS.with(|c| c.get()));
        LAST_REPORT_MS.with(|c| c.set(wall));
        LAST_REPORT_CALLS.with(|c| c.set(calls));
        if prev_wall == 0.0 {
            return true;
        }
        let dt = wall - prev_wall;
        let rate = if dt > 0.0 { (calls - prev_calls) as f64 * 1000.0 / dt } else { 0.0 };
        let per_call_us = if calls > 0 { total_ms * 1000.0 / calls as f64 } else { 0.0 };
        let marshal_ms = total_ms - dispatch_total;
        tracing::info!(
            target: "vitaslop::perf",
            "hostcalls: {calls} total, {rate:.0}/s over the last {dt:.0} ms, \
             {total_ms:.0} ms cumulative ({per_call_us:.2} us/call) = \
             {dispatch_total:.0} ms handler + {marshal_ms:.0} ms register marshalling, \
             {} preemptions ({} on fuel)", quanta(), fuel_yields()
        );
        true
    }

    /// Total host calls made and milliseconds spent in them since the run started.
    pub fn totals() -> (u64, f64) {
        (CALLS.with(|c| c.get()), MS.with(|m| m.get()))
    }

    /// `(calls, total ms, handler ms, marshalling ms)` since the run started.
    ///
    /// The same split `record` logs, as a VALUE. The log line reaches a filter that has to be
    /// configured to `vitaslop::perf=info` to see it, which on the live page means editing a
    /// knob box to answer the single most load-bearing question about host-call cost - whether
    /// the time is in the handler doing the guest's work, or around it copying a register file
    /// the handler mostly does not read. That question deserves to be on the screen, not behind
    /// a log filter.
    pub fn split() -> (u64, f64, f64, f64) {
        let calls = CALLS.with(|c| c.get());
        let total = MS.with(|m| m.get());
        let handler = DISPATCH_MS.with(|m| m.get());
        (calls, total, handler, total - handler)
    }

    thread_local! {
        /// JSPI SUSPENSIONS: every time a guest stack parked on a pending Promise.
        static SUSPENDS: Cell<u64> = const { Cell::new(0) };
        /// JSPI STACK STARTS: every `WebAssembly.promising` call, each of which makes V8
        /// allocate a fresh wasm stack.
        static STACK_STARTS: Cell<u64> = const { Cell::new(0) };
        /// Stacks parked on a NEVER-resolving Promise. These can never be reclaimed: the
        /// stack stays suspended and everything it holds stays live, for the rest of the
        /// run.
        static ABANDONED: Cell<u64> = const { Cell::new(0) };
    }

    /// Count one guest-stack suspension.
    pub fn note_suspend() {
        SUSPENDS.with(|c| c.set(c.get() + 1));
    }

    /// Count one fresh JSPI stack (a `promising` call).
    pub fn note_stack_start() {
        STACK_STARTS.with(|c| c.set(c.get() + 1));
    }

    /// Count one stack parked forever on a never-resolving Promise.
    pub fn note_abandoned_stack() {
        ABANDONED.with(|c| c.set(c.get() + 1));
    }

    thread_local! {
        /// Finished threads whose engine state was handed back.
        static RELEASED: Cell<u64> = const { Cell::new(0) };
    }

    /// Count one finished thread whose module instance was released.
    pub fn note_thread_released() {
        RELEASED.with(|c| c.set(c.get() + 1));
    }

    /// Finished threads released so far. Reported next to the stack starts, because the
    /// pair is the whole claim: instances created versus instances given back. One
    /// number alone cannot show a leak closing.
    pub fn released() -> u64 {
        RELEASED.with(|c| c.get())
    }

    thread_local! {
        /// Module instances actually INSTANTIATED, pooled on release, and taken back out
        /// of the pool. The three together are what say whether the pool is working: a
        /// title creating a guest thread per frame should show `created` going flat while
        /// `reused` climbs with the frames. `created` alone cannot distinguish a pool
        /// that is being used from one that is always empty.
        static INSTANCES: Cell<(u64, u64, u64)> = const { Cell::new((0, 0, 0)) };
    }

    /// Count one `WebAssembly.Instance` of the transpiled title.
    pub fn note_instance_created() {
        INSTANCES.with(|c| {
            let (a, b, d) = c.get();
            c.set((a + 1, b, d));
        });
    }

    /// Count one instance handed back to the pool by a finished thread.
    pub fn note_instance_pooled() {
        INSTANCES.with(|c| {
            let (a, b, d) = c.get();
            c.set((a, b + 1, d));
        });
    }

    /// Count one instance taken from the pool for a new thread.
    pub fn note_instance_reused() {
        INSTANCES.with(|c| {
            let (a, b, d) = c.get();
            c.set((a, b, d + 1));
        });
    }

    /// `(instantiated, pooled, reused)` so far.
    pub fn instance_stats() -> (u64, u64, u64) {
        INSTANCES.with(|c| c.get())
    }

    /// `(suspensions, stack starts, abandoned stacks)` so far.
    ///
    /// These three are reported per frame because a process that grows by a fixed amount
    /// every frame is either allocating a fixed NUMBER of something per frame or a fixed
    /// SIZE per unit of work, and only a count divides the growth into a per-item cost.
    /// A JSPI stack is the largest single allocation the scheduler can make - megabytes -
    /// so "how many were started and how many can never be freed" is the first question,
    /// and it was not answerable at all before these existed.
    pub fn stack_stats() -> (u64, u64, u64) {
        (
            SUSPENDS.with(|c| c.get()),
            STACK_STARTS.with(|c| c.get()),
            ABANDONED.with(|c| c.get()),
        )
    }

    /// The `performance` clock of whichever global we run in (a Worker has no `window`).
    pub fn now() -> f64 {
        thread_local! {
            static PERF: Option<web_sys::Performance> = js_sys::Reflect::get(
                &js_sys::global(),
                &wasm_bindgen::JsValue::from_str("performance"),
            )
            .ok()
            .and_then(|p| wasm_bindgen::JsCast::dyn_into::<web_sys::Performance>(p).ok());
        }
        PERF.with(|p| p.as_ref().map(|p| p.now()).unwrap_or(0.0))
    }
}

/// The monotonic millisecond clock this engine measures with, as a plain `fn` the
/// engine-agnostic runtime can hold. Handed to `vitaslop_runtime::perf::set_clock` at
/// startup: without it the runtime's phase timers are silently inert on `wasm32` (there is
/// no `Instant` there), which is why the browser could report a frame total and nothing
/// inside it.
pub fn perf_clock() -> f64 {
    hostcalls::now()
}

/// Total host calls and milliseconds spent in them since the run started. Published in
/// the per-frame status so a slow frame names its own cause.
pub fn host_call_totals() -> (u64, f64) {
    hostcalls::totals()
}

/// `(calls, total ms, handler ms, marshalling ms)` since the run started. See
/// [`hostcalls::split`] - the split is only meaningful over calls that were actually timed.
/// The `n` costliest host calls as `(selector, calls, ms)`, descending by total time.
///
/// Only meaningful with per-call timing on (debug capture, or `VITASLOP_PERF`), and empty
/// without it. See [`hostcalls::top_selectors_by_ms`].
pub fn host_calls_by_ms(n: usize) -> Vec<(u32, u64, f64)> {
    hostcalls::top_selectors_by_ms(n)
}

pub fn host_call_split() -> (u64, f64, f64, f64) {
    hostcalls::split()
}

/// Turn per-call host-call TIMING on or off for the next stretch of the run, so the live loop
/// can sample it instead of asking a user to choose between watching and profiling. See
/// [`hostcalls::timing_enabled`].
pub fn set_host_call_timing(on: bool) {
    hostcalls::set_timing_sampling(on);
}

/// `(suspensions, JSPI stack starts, abandoned stacks, released threads)` so far. See
/// [`hostcalls::stack_stats`] for why a per-frame count is what makes a per-frame process
/// growth attributable.
pub fn stack_stats() -> (u64, u64, u64, u64) {
    let (susp, starts, abandoned) = hostcalls::stack_stats();
    (susp, starts, abandoned, hostcalls::released())
}

/// `(instances instantiated, pooled, reused)` so far - see
/// [`hostcalls::instance_stats`].
pub fn instance_stats() -> (u64, u64, u64) {
    hostcalls::instance_stats()
}

/// `(preemptions, of which on software fuel)` so far.
///
/// Reported per frame because these are what ADVANCE THE GAME CLOCK - every preemption
/// charges `charge_cpu_quantum` a flat `QUANTUM_CPU_US` - so the clock's rate is exactly
/// their rate, and the split says which trigger set it. A host-call preemption measures
/// host-call density (which varies by screen) and a fuel preemption measures guest
/// execution (which does not); a clock driven by the first is calibrated for whatever
/// screen the calibration was taken on and wrong everywhere else.
pub fn preemption_stats() -> (u64, u64) {
    (hostcalls::quanta(), hostcalls::fuel_yields())
}

/// `(the last raw software-fuel reading, the smallest one seen)` - see
/// [`hostcalls::note_raw_fuel`].
pub fn raw_fuel_stats() -> (i64, i64) {
    hostcalls::raw_fuel()
}

/// Announce, once, HOW this run preempts a guest thread - because it is not how native
/// does it, and the difference is visible in every timing and every determinism
/// signature.
///
/// # The browser has no fuel, and that is not a detail
/// Native runs the guest on wasmtime with `fuel_async_yield_interval`, so a thread is
/// interrupted after a fixed amount of EXECUTION whatever it is doing. That interrupt is
/// also what advances the virtual game clock (`SchedCore::on_quantum` ->
/// `charge_cpu_quantum`). The browser's WebAssembly engine offers no such thing: a guest
/// thread here can only leave the CPU at a host call.
///
/// Left alone, that is not "slightly less preemptive" - it is a livelock. This title's
/// loader busy-polls `sceKernelGetProcessTime` waiting for time to pass, and time only
/// passes when a thread is preempted, so nothing ever ended the poll. Measured: native
/// spent 107,362 host calls on such a frame; the browser passed 16,700,000 on the same
/// one and was still going.
///
/// So the browser preempts on HOST CALLS, at a count calibrated to the resume rate
/// native's fuel produces. Every busy-wait that asks the host something is reached by
/// that.
///
/// # A host-call quantum is not enough on its own
/// What it does NOT reach is a guest loop that makes no host call at all. That was long
/// assumed not to occur; it does. Measured on this title: the browser reached display
/// flip 2 and then burned 100% CPU indefinitely with a completely FLAT host-call count
/// and zero scheduler rounds, while native ran the same boot to flip 45 in 4.3 s. The
/// loop spins on a word another thread writes, so it can only ever end if the scheduler
/// is allowed to run something else.
///
/// The second mechanism closes that: the transpiler emits a SOFTWARE FUEL counter on
/// guest loop back edges (`vitaslop_transpiler::emit::set_fuel_interval`), which the
/// browser turns on and native leaves off because wasmtime gives it real fuel. Both
/// mechanisms preempt the same way and both advance the clock; they differ only in what
/// they can see, so a run reports its yields split by cause.
fn preempt_note() {
    use std::sync::OnceLock;
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let n = hostcalls::quantum_calls();
        let fuel = fuel_interval();
        if n == 0 && fuel == 0 {
            tracing::warn!(
                target: "vitaslop::sched",
                "browser preemption is DISABLED on BOTH mechanisms \
                 (VITASLOP_BROWSER_QUANTUM_CALLS=0, VITASLOP_BROWSER_FUEL=0): a guest thread \
                 will run until it blocks of its own accord, a busy-wait on the virtual clock \
                 cannot terminate, and a guest loop that makes no host call will hang the tab"
            );
        } else if fuel == 0 {
            tracing::warn!(
                target: "vitaslop::sched",
                "browser preemption: every {n} host calls, and SOFTWARE FUEL IS OFF \
                 (VITASLOP_BROWSER_FUEL=0) - a guest loop that makes no host call is not \
                 interruptible in this run and will hang the tab if one is reached"
            );
        } else {
            tracing::info!(
                target: "vitaslop::sched",
                "browser preemption: every {fuel} units of software fuel (the engine has \
                 no fuel counter of its own, so the module counts itself by wasmtime's \
                 rule - this is native's own quantum in native's own unit), with a {n} \
                 host-call backstop. Fuel is the clock's driver; the host-call count is \
                 only a liveness net."
            );
        }
    });
}

/// Guest work a thread may execute before the browser preempts it, in WASMTIME FUEL UNITS
/// (`VITASLOP_BROWSER_FUEL`, 0 disables software fuel entirely). Accounted per wasm basic
/// block as the module is emitted, tested on loop back edges - see
/// `vitaslop_transpiler::emit::Body` and `emit_fuel_check`.
///
/// # It is the same UNIT as native's, and it is no longer the clock's RATE
/// The transpiler reproduces wasmtime's own accounting - its operator cost table and its
/// flush points - so a unit here is a unit there.
///
/// It used to be the clock's rate as well, because a preemption advanced the clock by a flat
/// `QUANTUM_CPU_US`. That tied this number to native's preemption interval - they agree, both
/// being [`QUANTUM_FUEL`](vitaslop_runtime::host::QUANTUM_FUEL) - but it also meant the clock
/// measured how OFTEN a thread was preempted rather than how much it executed. The clock is now
/// charged per unit of FUEL BURNED (`SchedCore::charge_guest_work`), so the two engines agree
/// whatever their intervals are, and this is free to be chosen for what it actually trades off:
/// preemption granularity against the cost of the check.
///
/// It is deliberately the DOMINANT mechanism: the host-call quantum
/// ([`hostcalls::quantum_calls`]) measures host-call DENSITY, which varies by an order of
/// magnitude between a menu and a race, so a clock driven by it is calibrated for
/// whichever screen the calibration was taken on and wrong on every other. That model
/// measured 17.37 s of game clock where native had 88.688 s - **5.1x slow** - and it
/// stranded a frame-keyed recipe on a timed screen for 14,000 frames while native was two
/// screens further on.
///
/// # Why fitting a constant never worked
/// Every earlier model measured a PROXY for guest work, and a proxy's ratio to the real
/// thing moves with the workload, so the error moved too and no scalar tracked it:
///
/// | IR-statement interval | ~f10,300 | ~f29,300 | ~f40,800 |
/// |-----------------------|----------|----------|----------|
/// | 405,000               | **+1%**  | +27%     | +65%     |
/// | 428,000               | -12%     | -12%     | -12%     |
/// | 440,000               | -14%     | -15%     | -23%     |
///
/// A value perfect at one frame and 65% out at another is worse everywhere it is not
/// being measured. **That the error moved with the workload rather than scaling is the
/// evidence that the UNIT was wrong rather than the number** - so the unit was replaced,
/// twice, rather than the constant being re-fitted again.
///
/// # It is not linear, so never tune it by ratio
/// A faster clock changes how much guest work happens per frame, so this is a feedback
/// loop: at the old unit, below roughly 400,000 the run BIFURCATED - the title cleared a
/// load in far fewer frames and ended up AHEAD of native. Re-calibrate by running the
/// whole curve against native at several frames, never by scaling a single point.
///
/// Read once, before the transpile that bakes it into the module - it is an emit-time
/// property of the code, not something a running thread can be re-tuned with.
pub fn fuel_interval() -> u32 {
    thread_local! {
        static N: u32 = match vitaslop_runtime::knobs::var("VITASLOP_BROWSER_FUEL") {
            // Native's `fuel_async_yield_interval`, in the unit both now count.
            Err(_) => 5_000_000,
            Ok(v) => v.parse().unwrap_or_else(|_| {
                panic!("VITASLOP_BROWSER_FUEL={v} is not a fuel count")
            }),
        };
    }
    N.with(|n| *n)
}

/// The shared host: a single `VitaEnv` behind an `Arc<Mutex>` (single-threaded here,
/// so the lock never contends - it just satisfies `SchedCore`'s bound, which mirrors
/// native's `Send` host).
type Host = Arc<Mutex<VitaEnv>>;

/// A `Uint8Array` view over the shared linear memory, rebased so guest address `A` is
/// byte `A - base`.
///
/// # The view is built ONCE, and that is a load-bearing property
/// This used to call `Uint8Array::new(&self.mem.buffer())` inside every `read`, `write`
/// and `len` - allocating a fresh JS typed array per ACCESS, not per host call. A host
/// call is made of guest-memory accesses (every pointer argument, every struct field,
/// every string), so a four-byte `read_u32` cost a `.buffer` getter, a `Uint8Array`
/// allocation, a `subarray` allocation and a copy. That is what made a browser guest
/// frame take minutes against 55 ms native.
///
/// Caching is sound because this memory CANNOT change identity: it is created with
/// `initial == maximum` (see `BrowserEngine::new`), so it never grows, and its buffer is
/// a `SharedArrayBuffer`, which never detaches. A growable memory would need the view
/// rebuilt on grow - there isn't one, and if one ever appears the fixed size is asserted
/// at construction rather than silently worked around here.
#[derive(Clone)]
struct SharedView {
    bytes: Uint8Array,
    /// A `Uint32Array` and a `Uint16Array` over the SAME buffer, from offset 0, so a
    /// rebased byte offset indexes as `off >> 2` / `off >> 1`.
    ///
    /// # These are the scalar path, and they exist because a block read is not one
    /// `read(off, &mut [0u8; 4])` here is `subarray` - a boundary crossing that ALLOCATES
    /// a JS typed array - plus `copy_to`, a second crossing. Through these it is one
    /// `get_index`, no allocation. Every host handler that reads a guest struct field, a
    /// pointer or a flag pays that, tens of thousands of times a presented frame
    /// ([[vitaslop-count-calls-not-bytes-across-the-guest-boundary]] counted ~22,000 on a
    /// racing title's on-track frame AFTER the four biggest per-word readers had been
    /// bulk-converted).
    ///
    /// Sound for the same reason the byte view is built once: the memory is created
    /// `initial == maximum`, so it never grows and its `SharedArrayBuffer` never detaches.
    /// Only ALIGNED accesses go through them - the caller's offset is checked, and a
    /// misaligned one falls back to the byte path, because a typed array cannot express it.
    words: js_sys::Uint32Array,
    halves: js_sys::Uint16Array,
    /// Byte offset of the GUEST-STORE DIRTY BLOCK this module was emitted with, or
    /// `None` when it was built without one. The block is the epoch byte followed by
    /// one stamp byte per 4 KB page - see `vitaslop_transpiler::emit::emit_dirty_mark`,
    /// which writes it.
    dirty_off: Option<u64>,
}

impl SharedView {
    /// Absolute byte offset of page `p`'s stamp.
    fn stamp_at(&self, block: u64, page: usize) -> u32 {
        (block + transpiler_dirty_map_off() + page as u64) as u32
    }

    /// Stamp every page `[off, off + len)` touches with the current epoch.
    ///
    /// # This is not an extra - it is the other half of the map
    /// The transpiler stamps what the GUEST stores, and a host call writes guest memory
    /// too: a file read, a `memcpy` NID, a GXM transfer. Those writes are invisible to
    /// translated code [[vitaslop-host-write-watch]], so a map that only the guest wrote
    /// would report a texture the host had just overwritten as untouched - a silent
    /// stale texture, the exact bug the compare exists to prevent.
    /// Pages of the whole linear memory, which is what the map covers.
    fn pages(&self) -> usize {
        (self.bytes.length() as usize) >> vitaslop_transpiler::DIRTY_SHIFT
    }

    fn stamp_written(&self, off: usize, len: usize) {
        let (Some(block), true) = (self.dirty_off, len > 0) else { return };
        let epoch = self.bytes.get_index((block + vitaslop_transpiler::DIRTY_EPOCH_OFF) as u32);
        let shift = vitaslop_transpiler::DIRTY_SHIFT;
        let first = off >> shift;
        let last = (off + len - 1) >> shift;
        // Nearly every host write is a pointer argument or a struct field and lands in
        // ONE page; a single `set_index` is one boundary crossing where `fill` is one
        // crossing plus a range, and crossings are what host calls are made of
        // [[vitaslop-browser-host-call-cost]].
        if first == last {
            self.bytes.set_index(self.stamp_at(block, first), epoch);
        } else {
            self.bytes.fill(epoch, self.stamp_at(block, first), self.stamp_at(block, last + 1));
        }
    }
}

impl GuestMemory for SharedView {
    fn len(&self) -> usize {
        self.bytes.length() as usize
    }
    fn read(&self, off: usize, buf: &mut [u8]) {
        // SAFETY: the view borrows this module's linear memory for the duration of the call
        // below. `copy_range` is synchronous JS that cannot grow it, and nothing else holds a
        // reference to `buf`.
        unsafe {
            let dst = js_sys::Uint8Array::view_mut_raw(buf.as_mut_ptr(), buf.len());
            copy_range(&self.bytes, off as u32, &dst);
        }
    }
    fn write(&mut self, off: usize, bytes: &[u8]) {
        SharedView::write_at(self, off, bytes)
    }
    fn write_u32(&mut self, off: usize, v: u32) {
        SharedView::write_word(self, off, v)
    }
}

// >>> AND THE SAME IMPL FOR A BORROW, WHICH IS WHAT A HOST CALL USES.
//
// `GuestMemory` is handed to `dispatch` as `&mut dyn`, so the browser used to CLONE the view
// per call (`rt.view()`). The view is three `js_sys` typed-array handles, and cloning a
// `JsValue`-backed handle is a call into the wasm-bindgen heap table - three crossings to make
// it and three more to drop it, SIX per host call, ~630 times a frame on a race, for a
// structure that is immutable and already lives on the thread
// ([[vitaslop-count-calls-not-bytes-across-the-guest-boundary]]).
//
// Nothing in the impl mutates Rust state - every method reaches through a handle into the
// `SharedArrayBuffer` - so a `&SharedView` is as capable as an owned one, and the call site
// passes `&mut &rt.view`.
impl GuestMemory for &'_ SharedView {
    fn len(&self) -> usize {
        (**self).len()
    }
    fn read(&self, off: usize, buf: &mut [u8]) {
        (**self).read(off, buf)
    }
    fn write(&mut self, off: usize, bytes: &[u8]) {
        SharedView::write_at(self, off, bytes)
    }
    fn read_u32(&self, off: usize) -> u32 {
        (**self).read_u32(off)
    }
    fn read_u16(&self, off: usize) -> u16 {
        (**self).read_u16(off)
    }
    fn write_u32(&mut self, off: usize, v: u32) {
        SharedView::write_word(self, off, v)
    }
    fn dirty_since(&self, off: usize, len: usize, stamp: u8) -> Option<bool> {
        (**self).dirty_since(off, len, stamp)
    }
    fn dirty_runs_since(
        &self,
        off: usize,
        len: usize,
        stamp: u8,
        out: &mut Vec<(usize, usize)>,
    ) -> Option<()> {
        (**self).dirty_runs_since(off, len, stamp, out)
    }
    fn rebase_dirty_epoch(&self, floor: u8) -> Option<u8> {
        (**self).rebase_dirty_epoch(floor)
    }
    fn bump_dirty_epoch(&self) -> Option<(u8, bool)> {
        (**self).bump_dirty_epoch()
    }
    fn dirty_epoch(&self) -> Option<u8> {
        (**self).dirty_epoch()
    }
    fn borrow(&self, off: usize, len: usize) -> Option<&[u8]> {
        (**self).borrow(off, len)
    }
}

impl SharedView {
    fn write_at(&self, off: usize, bytes: &[u8]) {
        // SAFETY: as in `read`.
        unsafe {
            let src = js_sys::Uint8Array::view(bytes);
            write_range(&self.bytes, off as u32, &src);
        }
        self.stamp_written(off, bytes.len());
    }

    // >>> THE SCALAR PATH: ONE CROSSING, NO ALLOCATION. See `SharedView::words`.
    fn read_u32(&self, off: usize) -> u32 {
        if off & 3 != 0 {
            let mut b = [0u8; 4];
            self.read(off, &mut b);
            return u32::from_le_bytes(b);
        }
        self.words.get_index((off >> 2) as u32)
    }

    fn read_u16(&self, off: usize) -> u16 {
        if off & 1 != 0 {
            let mut b = [0u8; 2];
            self.read(off, &mut b);
            return u16::from_le_bytes(b);
        }
        self.halves.get_index((off >> 1) as u32)
    }

    fn write_word(&self, off: usize, v: u32) {
        if off & 3 != 0 {
            self.write_at(off, &v.to_le_bytes());
            return;
        }
        self.words.set_index((off >> 2) as u32, v);
        // The other half of the map, exactly as `write` owes it - a host write the guest
        // cannot see stamped would let a texture snapshot report memory the host had just
        // overwritten as untouched. See `stamp_written`.
        self.stamp_written(off, 4);
    }

    fn dirty_since(&self, off: usize, len: usize, stamp: u8) -> Option<bool> {
        let block = self.dirty_off?;
        if len == 0 {
            return Some(false);
        }
        let shift = vitaslop_transpiler::DIRTY_SHIFT;
        // One page BELOW the range as well: a store is stamped against the page it
        // STARTS in, and an 8-byte store starting in the page below can reach into
        // this one. See `GuestMemory::dirty_since`.
        let first = (off >> shift).saturating_sub(1);
        // The map covers every page of the memory, and a texture lives in the guest
        // region below it, so this clamp should never bite; it is here so that a bad
        // length reads a short range rather than the bytes above the map.
        let last = ((off + len - 1) >> shift).min(self.pages());
        // >>> THE SCAN RUNS WHERE THE MAP IS, so this is ONE crossing and no buffer at all.
        //
        // It used to copy the page range into a reused `Vec` - a `subarray` (a crossing that
        // allocates a JS typed array), a `copy_to` (a second crossing) and a `resize` that
        // zeroes the buffer first - and then test the bytes in Rust. Every one of those is
        // work for an answer that is one bit, and the copy could not stop early at the first
        // dirty page the way the loop on the JS side does.
        Some(any_ge(
            &self.bytes,
            self.stamp_at(block, first),
            self.stamp_at(block, last + 1),
            stamp,
        ))
    }

    /// See [`GuestMemory::dirty_runs_since`]. ONE crossing: the page map for this range is a
    /// byte per 4 KB, so a 0.75 MB texture asks about 192 bytes - against re-reading the
    /// texture itself, which is what the answer avoids.
    fn dirty_runs_since(
        &self,
        off: usize,
        len: usize,
        stamp: u8,
        out: &mut Vec<(usize, usize)>,
    ) -> Option<()> {
        let block = self.dirty_off?;
        if len == 0 {
            return Some(());
        }
        let shift = vitaslop_transpiler::DIRTY_SHIFT;
        let page_bytes = 1usize << shift;
        let first = off >> shift;
        let last = (off + len - 1) >> shift;
        if last >= self.pages() {
            return None;
        }
        // The page BELOW as well - a store stamped against it can reach into the first page
        // of this range. It is read as part of the same crossing and folded into page 0.
        let below = first.saturating_sub(1);
        let map = self.stamp_at(block, below);
        let end = map + (last - below + 1) as u32;
        if end > self.bytes.length() {
            return None;
        }
        let mut pages = vec![0u8; (end - map) as usize];
        self.bytes.subarray(map, end).copy_to(&mut pages);
        // `pages[0]` is the page below when there is one, so the range's own pages start at
        // `skip`, and the overhang makes page 0 of the range dirty if that one is.
        let skip = (first - below) as usize;
        let overhang = skip == 1 && pages[0] >= stamp;
        let mut run: Option<(usize, usize)> = None;
        for (i, &p) in pages[skip..].iter().enumerate() {
            let dirty = p >= stamp || (i == 0 && overhang);
            if !dirty {
                if let Some(r) = run.take() {
                    out.push(r);
                }
                continue;
            }
            // Page i of the range, clipped to the range itself: the first page starts part
            // way in and the last one ends part way through.
            let page_start = ((first + i) << shift).max(off) - off;
            let page_end = (((first + i) << shift) + page_bytes).min(off + len) - off;
            match run.as_mut() {
                Some(r) => r.1 = page_end,
                None => run = Some((page_start, page_end)),
            }
        }
        if let Some(r) = run {
            out.push(r);
        }
        Some(())
    }

    fn dirty_epoch(&self) -> Option<u8> {
        let block = self.dirty_off?;
        Some(self.bytes.get_index((block + vitaslop_transpiler::DIRTY_EPOCH_OFF) as u32))
    }

    /// See [`GuestMemory::rebase_dirty_epoch`]. The map crosses the boundary TWICE - once out,
    /// once back - which is two crossings and 131 KB against the 53 MB of texture a wrap makes
    /// the host re-read.
    fn rebase_dirty_epoch(&self, floor: u8) -> Option<u8> {
        let block = self.dirty_off?;
        let map = self.stamp_at(block, 0);
        // `pages() + 1` for the same reason `bump_dirty_epoch` uses it: the map covers every
        // page of the memory and the last one is partial.
        let end = self.bytes.length().min(map + self.pages() as u32 + 1);
        let mut pages = vec![0u8; (end - map) as usize];
        self.bytes.subarray(map, end).copy_to(&mut pages);
        for p in pages.iter_mut() {
            *p = if *p >= floor { *p - floor + 1 } else { 0 };
        }
        self.bytes.subarray(map, end).copy_from(&pages);
        let epoch_at = (block + vitaslop_transpiler::DIRTY_EPOCH_OFF) as u32;
        let cur = self.bytes.get_index(epoch_at);
        let next = if cur >= floor { cur - floor + 1 } else { 1 };
        self.bytes.set_index(epoch_at, next);
        Some(next)
    }

    fn bump_dirty_epoch(&self) -> Option<(u8, bool)> {
        let block = self.dirty_off?;
        let epoch_at = (block + vitaslop_transpiler::DIRTY_EPOCH_OFF) as u32;
        let next = self.bytes.get_index(epoch_at).wrapping_add(1);
        // The epoch is one byte and it is compared with `>=`, so it may not wrap
        // silently: a stamp of 250 would suddenly read as "later than" a store at 3.
        // Zeroing the map and restarting at 1 makes every existing stamp strictly
        // greater than every map entry, i.e. "no store since" - which would be a LIE,
        // so the caller is told and drops them. Rare by construction: the epoch only
        // advances when a texture's bytes are actually established.
        if next == 0 || next == u8::MAX {
            let map = self.stamp_at(block, 0);
            let pages = (self.len() >> vitaslop_transpiler::DIRTY_SHIFT) as u32 + 1;
            let end = self.bytes.length().min(map + pages);
            self.bytes.fill(0, map, end);
            self.bytes.set_index(epoch_at, 1);
            return Some((1, true));
        }
        self.bytes.set_index(epoch_at, next);
        Some((next, false))
    }
}

/// Byte offset of the page map within the dirty block. A thin wrapper so the two uses
/// above read as offsets rather than as arithmetic on a transpiler constant.
fn transpiler_dirty_map_off() -> u64 {
    vitaslop_transpiler::DIRTY_MAP_OFF
}

// Batched register marshalling, in JS.
//
// # Why a JS helper and not a Rust loop
// A host call has to move the guest's whole register file out of the instance's wasm
// globals and back. Done from Rust that is `Global::value()` per register - and each one
// is a call out to the wasm-bindgen glue, a JS value pushed into the shared heap table,
// an `as_f64` back across, and a drop: four boundary crossings for four bytes. Times 32
// registers, times two directions, that is over a hundred crossings per host call, and a
// retail title makes millions of host calls per loading frame. It measured 3.3 us per
// call, nearly all of it here.
//
// The loop itself is trivial - it is the BOUNDARY that costs. So the loop moves to the
// side of the boundary where the globals live, and the whole file crosses once, as one
// `Float64Array`. `inline_js` (rather than `new Function`) so the helper is an ordinary
// ES module wasm-bindgen emits, and a page with a strict CSP can still run it.
//
// # ...and why the buffer is a `Uint32Array` VIEW over the Rust array
// The register file is 32 `u32`s. Carrying it as `f64` needed a staging `Float64Array` owned
// by JS, a `copy_to`/`copy_from` of 256 bytes to get it into or out of Rust, and a 32-lane
// conversion loop on each side - so the "one crossing" was really two plus a copy plus a
// convert, each way. A `Uint32Array` VIEW over the Rust buffer is the same memory, so the JS
// loop writes straight into the array the caller reads: ONE crossing, no copy, no conversion.
//
// `Global.value` on an `i32` global is a JS number and stores into a `Uint32Array` under
// `ToUint32`, which is bit-exact both ways - the same round trip the `f64` form relied on.
#[wasm_bindgen(inline_js = "
export function save_some(globals, idxs, out) {
  for (let k = 0; k < idxs.length; k++) { const i = idxs[k]; out[i] = globals[i].value; }
}
")]
extern "C" {
    /// Copy just the globals named by `idxs` into `out`, at their own indices.
    fn save_some(globals: &Array, idxs: &js_sys::Uint32Array, out: &js_sys::Uint32Array);
}

/// >>> THE REGISTERS A HOST CALL CAN ACTUALLY REACH.
///
/// AAPCS puts the arguments in r0..r3 and the return in r0; the stack arguments and the
/// call-site attribution need sp and lr; a fault report wants pc; r12 is the intra-procedure
/// scratch. **Every live `GuestCtx::regs` index in the workspace is one of these** - the only
/// wider read is the thread-exit log, which says so where it prints.
///
/// The other 25 lanes of the file were read out of `WebAssembly.Global` getters and written
/// straight back unchanged, ~630 times a frame on a race. They are r4..r11 (callee-saved, so a
/// host call must not touch them) and the VFP argument file (which only a handler with a FLOAT
/// parameter reads, and the GXM/kernel calls a race makes have none).
const NARROW_REGS: [u32; 7] = [0, 1, 2, 3, abi::SP as u32, 14, 15];


// >>> BULK GUEST-MEMORY MOVES AND THE DIRTY SCAN, EACH AS ONE CROSSING.
//
// `subarray(a, b).copy_to(buf)` reads as one operation and is two: `subarray` is a crossing
// that ALLOCATES a JS typed array, and `copy_to` is a second crossing that copies through it.
// Same for the write direction, and the dirty-page scan paid a third cost on top - it copied a
// range of the map into a Rust buffer (which it first had to zero) only to test each byte.
//
// A race frame makes ~2,300 bulk guest reads and ~800 dirty queries, and the boundary is what
// this engine is billed in ([[vitaslop-count-calls-not-bytes-across-the-guest-boundary]]) - so
// each of these is one call now, with the loop or the copy on the side of the boundary the
// memory lives on. `any_ge` also EARLY-EXITS, which the copy-then-scan could not.
#[wasm_bindgen(inline_js = "
export function copy_range(src, off, dst) {
  dst.set(src.subarray(off, off + dst.length));
}
export function write_range(dst, off, src) {
  dst.set(src, off);
}
export function any_ge(u8, from, to, stamp) {
  for (let i = from; i < to; i++) if (u8[i] >= stamp) return true;
  return false;
}
")]
extern "C" {
    /// `dst[..] = src[off .. off + dst.len()]`.
    fn copy_range(src: &js_sys::Uint8Array, off: u32, dst: &js_sys::Uint8Array);
    /// `dst[off .. off + src.len()] = src[..]`.
    fn write_range(dst: &js_sys::Uint8Array, off: u32, src: &js_sys::Uint8Array);
    /// Whether any byte of `u8[from..to]` is `>= stamp`.
    fn any_ge(u8: &js_sys::Uint8Array, from: u32, to: u32, stamp: u8) -> bool;
}

/// A guest instance's mutable state the host reaches during a call: its 16 ARM
/// register globals, its VFP single-precision argument globals, and the shared memory.
struct ThreadRt {
    /// Every register global, ARM first then the VFP argument file - the same order as
    /// [`ThreadRt::file`]. For the few single-register accesses (a park's resume code, an
    /// entry's seed values) and for the write-back of a call that changed one or two lanes.
    /// Bulk transfer goes through [`ThreadRt::file`].
    regs: Vec<WebAssembly::Global>,
    /// The same globals as one JS array - the 16 ARM registers followed by the VFP
    /// argument registers - so the whole file marshals in a single boundary crossing
    /// through [`save_some`], and back one lane at a time through [`ThreadRt::set_reg`].
    file: Array,
    /// [`NARROW_REGS`] as a `Uint32Array`, built once per thread so the per-call read does
    /// not allocate one.
    narrow: js_sys::Uint32Array,
    /// The whole shared memory as one typed array, built once. See [`SharedView`] for
    /// why caching it is both sound and load-bearing.
    view: SharedView,
    base: u32,
}

/// Slots in [`ThreadRt::file`]: the ARM registers, then the VFP argument registers.
const FILE_LEN: usize = abi::REG_COUNT + VFP_ARG_COUNT;

impl ThreadRt {
    /// The whole register file, in one crossing each way.
    ///
    /// A register value is a `u32` carried as an `f64`, which is exact (an `f64` holds
    /// every integer up to 2^53), so nothing is lost in the round trip - the per-register
    /// path this replaced used the same representation.
    fn read_file(&self) -> ([u32; abi::REG_COUNT], [u32; VFP_ARG_COUNT]) {
        let mut buf = [0u32; FILE_LEN];
        // SAFETY: the view borrows this module's linear memory for the duration of the call
        // below. `save_some` is synchronous JS that allocates nothing, so the memory cannot
        // grow and the view cannot be detached while it is live, and nothing else holds a
        // reference to `buf`.
        //
        // Only [`NARROW_REGS`] are read; every other lane stays zero and is written back only
        // if the handler CHANGED it, which the write-back's diff already decides.
        unsafe {
            let view = js_sys::Uint32Array::view_mut_raw(buf.as_mut_ptr(), FILE_LEN);
            save_some(&self.file, &self.narrow, &view);
        }
        let mut regs = [0u32; abi::REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        regs.copy_from_slice(&buf[..abi::REG_COUNT]);
        vfp.copy_from_slice(&buf[abi::REG_COUNT..]);
        (regs, vfp)
    }

    /// Write back only the lanes the handler actually CHANGED.
    ///
    /// >>> THE FULL WRITE WAS 32 `WebAssembly.Global` SETTERS FOR AN AVERAGE OF ONE CHANGED
    /// REGISTER. A host call takes its arguments in r0..r3 and returns in r0, so almost every
    /// one of them leaves 31 of the 32 lanes exactly as it found them - and writing those back
    /// is a JS setter each, on a path a retail race takes ~630 times a frame
    /// ([[vitaslop-browser-host-call-cost]] is why the crossing, not the handler, is what a
    /// count-based win buys back here).
    ///
    /// Comparing 32 words in Rust is free next to one boundary crossing, so the mask decides
    /// the shape: nothing changed, write nothing; one or two lanes, set them individually and
    /// send only those - which is also the only CORRECT shape once the read is narrow, since
    /// the lanes the read skipped are zero here.
    fn write_file_changed(
        &self,
        before: &([u32; abi::REG_COUNT], [u32; VFP_ARG_COUNT]),
        regs: &[u32; abi::REG_COUNT],
        vfp: &[u32; VFP_ARG_COUNT],
    ) {
        let mut mask: u32 = 0;
        for (i, r) in regs.iter().enumerate() {
            if *r != before.0[i] {
                mask |= 1 << i;
            }
        }
        for (i, v) in vfp.iter().enumerate() {
            if *v != before.1[i] {
                mask |= 1 << (abi::REG_COUNT + i);
            }
        }
        // >>> ALWAYS PER-LANE, NEVER THE BULK PATH.
        //
        // The bulk write sends all 32 lanes, and since the READ is narrow ([`NARROW_REGS`])
        // the lanes it did not read are ZERO here - so falling back to it for a call that
        // changed three registers would write zero over r4..r11 and the whole VFP file, which
        // the guest is entitled to find exactly as it left them. A host call changes one or
        // two lanes (the return, occasionally a 64-bit pair), so this loop is the fast path as
        // well as the only correct one.
        if mask != 0 {
            for i in 0..FILE_LEN {
                if mask & (1 << i) != 0 {
                    let v = if i < abi::REG_COUNT { regs[i] } else { vfp[i - abi::REG_COUNT] };
                    self.set_reg(i, v);
                }
            }
        }
    }


    fn set_reg(&self, i: usize, v: u32) {
        self.regs[i].set_value(&JsValue::from_f64(v as f64));
    }
    fn read_reg(&self, i: usize) -> u32 {
        self.regs[i].value().as_f64().unwrap_or(0.0) as i64 as u32
    }
    fn view(&self) -> SharedView {
        self.view.clone()
    }
}

/// One guest thread on the browser engine: its own instance (its register file), the
/// `promising`-wrapped entries it runs in sequence, and the two one-slot channels JSPI
/// uses to hand control back and forth with the scheduler loop.
///
/// The main thread runs several entries in load order (a linked title's `module_init`s
/// then the eboot entry, each on the same instance with a fresh stack) - the browser
/// twin of native's `instantiate_thread_seq`. A spawned worker is just one entry.
pub struct BrowserThread {
    thid: i32,
    priority: i32,
    /// Everything the ENGINE holds for this thread, dropped by
    /// [`release`](ThreadHandle::release) the moment the scheduler records the thread as
    /// finished. `None` afterwards: a finished thread is never picked again, so only its
    /// identity above is still read.
    ///
    /// This is `Option` rather than plain fields precisely so the release is total. A
    /// guest thread here is one `WebAssembly.Instance` of the ENTIRE transpiled title
    /// plus a JSPI stack - about 7 MB - and the pieces reference each other (the entry
    /// functions hold the instance, the import closure holds the register file), so
    /// clearing them one at a time leaves whichever one was forgotten pinning the rest.
    engine: Option<ThreadEngine>,
    /// The shared host, so the un-park path can claim any return code owed to this
    /// thread (a timed wait that expired -> WAIT_TIMEOUT) and write it into r0 before
    /// the guest stack resumes. Native does this inside its import closure after the
    /// block await; the browser has no such re-entry, so it applies it here.
    host: Host,
    /// The engine's shared instance pool, so a finished thread can give its instance
    /// back instead of dropping it - see [`release`](ThreadHandle::release).
    pool: InstancePool,
}

/// Instances whose guest thread has finished, reset and ready to run another. Shared
/// between the engine (which takes from it) and every live thread (which gives back).
type InstancePool = Rc<RefCell<Vec<ThreadEngine>>>;

/// Whether a finished thread's module instance may be REUSED by the next thread
/// (`VITASLOP_BROWSER_INSTANCE_POOL=0` to disable).
///
/// On by default: without it a title that creates a guest thread per frame instantiates the
/// whole transpiled module per frame, and the browser is killed for it (see
/// [`ThreadHandle::release`]). It is a knob because it is the kind of change that can only be
/// cleared by A/B - a reused instance is supposed to be indistinguishable from a fresh one,
/// and the one-line way to test that claim on any title is to turn the reuse off and see
/// whether the behaviour moves.
fn instance_pool_enabled() -> bool {
    thread_local! {
        static ON: bool = !matches!(
            vitaslop_runtime::knobs::var("VITASLOP_BROWSER_INSTANCE_POOL").as_deref(),
            Ok("0")
        );
    }
    ON.with(|v| *v)
}

/// The per-thread engine state: one module instance and the JSPI machinery that drives
/// it. Held by [`BrowserThread`] only while the thread is live.
struct ThreadEngine {
    rt: Rc<ThreadRt>,
    /// The same `ThreadRt` as the import closure sees it. Cleared on release, so the
    /// closure stops pinning the instance's register file.
    rt_cell: Rc<RefCell<Option<Rc<ThreadRt>>>>,
    /// One `promising`-wrapped entry per address, run in order.
    entries: Vec<Function>,
    /// Index of the entry currently running (or about to start).
    entry_idx: usize,
    /// Whether the current entry has been started (its `promising` call made). Cleared
    /// when advancing to the next entry.
    entry_started: bool,
    sp: u32,
    r0: u32,
    r1: u32,
    r2: u32,
    /// The resolver for the *current* resume's step Promise. The import closure (on a
    /// block/yield) or an entry's completion fills it with the encoded event; the
    /// scheduler awaits the matching Promise. Reset each loop turn.
    signal: Rc<RefCell<Option<Function>>>,
    /// The resolver of the Promise a suspended thread is parked on; the scheduler calls
    /// it to un-park (resume) the thread.
    cont: Rc<RefCell<Option<Function>>>,
    /// The import closure must outlive every call the instance can make into it.
    _import: Closure<dyn FnMut(i32) -> JsValue>,
    /// This instance's SOFTWARE FUEL counter (`abi::FUEL_EXPORT`), or `None` in a build
    /// with fuel switched off. Read to price this thread's guest work - see
    /// [`BrowserThread::fuel_used`].
    fuel: Option<WebAssembly::Global>,
    /// The baseline the NEXT reading is differenced against, and the total burned. The
    /// counter itself cannot be the answer: the guest clears it after every yield, so only
    /// the differences between readings accumulate to guest work.
    fuel_last: i64,
    /// The last RAW reading. A counter is a value, not an event: the scheduler may read it
    /// twice without the guest having run in between, and billing that twice is a game clock
    /// that runs fast for no reason.
    fuel_raw: i64,
    fuel_total: u64,
    /// The GUEST-INSTRUCTION half of the same `work` global, at the last reading, and the
    /// total retired. That half counts UP and is never reset by the guest, so a reading is
    /// a running total and the delta between readings is what this resume retired.
    arm_last: u64,
    arm_total: u64,
    /// The `work` global's two halves as they read at THIS suspension, or `None` when the
    /// thread has run since the last reading.
    ///
    /// # Why the reading is cached and not simply taken twice
    /// `fuel_used` and `arm_retired` are the two halves of one 64-bit global and are called
    /// back to back by `SchedCore::charge_guest_work`, once per scheduler round. Each call
    /// was a `Global::value()` (a boundary crossing) plus a `BigInt` construction and a
    /// conversion back to `i64` - and a BigInt round trip is the dearest way JS has of
    /// moving 64 bits. The guest cannot run between the two calls, so the second reading is
    /// the first by construction. Cleared in `resume`, which is the only place the guest
    /// runs.
    work_read: Option<(u64, u64)>,
    /// The instance's exports, kept so a REUSED instance can look up its next entry's
    /// function and its `tp` global without instantiating anything.
    exports: Object,
    /// The instance's `reset` export (`abi::RESET_EXPORT`), called before the instance is
    /// handed to another guest thread.
    reset: Function,
    /// Which guest thread this instance is currently running. A CELL, not a captured
    /// value, because the import closure is built once per INSTANCE and an instance now
    /// outlives the thread it was made for - the closure has to report the thread that is
    /// running now, not the one it was created for.
    thid: Rc<Cell<i32>>,
    /// Set when this instance's guest stack parked on a never-resolving Promise (a
    /// `sceKernelExitThread`, a process halt, or a fatal call). Such an instance is
    /// DROPPED rather than pooled: a parked JSPI stack still references it, and while
    /// that stack can never resume, reusing the instance underneath it would mean a
    /// suspended frame and a live thread sharing one register file.
    abandoned: Rc<Cell<bool>>,
}

impl BrowserThread {
    /// Read the packed `work` global once and split it into
    /// `(operators since the last yield, guest instructions retired)`.
    ///
    /// # Why both halves come from ONE read
    /// They live in one i64 global (see `abi::WORK_GLOBAL`) so the emitted code can advance
    /// both with a single `i64.add`, which is what makes billing the clock in guest
    /// instructions cost no extra code. Reading them separately would also cost two
    /// boundary crossings per suspend for a value that is already in hand.
    ///
    /// An i64 global crosses into JS as a BigInt, which `as_f64` does not accept - hence
    /// the explicit BigInt conversion rather than the numeric path the i32 counter used.
    /// # The global is a BIT PATTERN, and reading it as a MAGNITUDE stopped the clock
    /// `WORK_INSTR_SHIFT` is 32, so the instruction half owns bits 32..64 - and the moment a
    /// thread retires 2^31 guest instructions, bit 63 is set and the i64 is NEGATIVE.
    /// `u64::try_from` REFUSES a negative BigInt, so this returned `None`, `fuel_used` returned
    /// `None`, and `charge_guest_work` billed that thread nothing from then on. The game clock
    /// simply stopped advancing for it, so whatever it was waiting on never arrived and it spun
    /// - **3,164 suspensions in one frame, 986 ms of wall time, and not one of them billed**
    /// (a retail boot, frame 174, at a guest clock of 4.87 s, which is where a long-lived
    /// thread crosses 2^31 retired instructions).
    ///
    /// It is read as `i64` and reinterpreted, because the value was never a number: it is two
    /// counters packed into one word so the emitted code can advance both with a single add.
    fn read_work(&mut self) -> Option<(u64, u64)> {
        if let Some(cached) = self.engine.as_ref()?.work_read {
            return Some(cached);
        }
        let raw = self.engine.as_ref()?.fuel.as_ref()?.value();
        // `unchecked_into`, not `BigInt::new`: the global is declared `i64`, so its value IS
        // a BigInt and `BigInt(x)` would be a JS function call to convert it to itself.
        let packed: js_sys::BigInt = raw.unchecked_into();
        let bits = i64::try_from(packed).ok()? as u64;
        let (ops, instructions) = abi::split_work(bits);
        let out = (u64::from(ops), u64::from(instructions));
        self.engine.as_mut()?.work_read = Some(out);
        Some(out)
    }
}

impl ThreadHandle for BrowserThread {
    fn thid(&self) -> i32 {
        self.thid
    }
    fn priority(&self) -> i32 {
        self.priority
    }

    /// Guest work this thread has executed, in the SAME UNIT native reports.
    ///
    /// The transpiler's software fuel reproduces wasmtime's own operator cost table and
    /// flush points, so a unit here is a unit there - which is what lets one game-clock
    /// calibration serve both engines. It also removes a divergence the per-preemption
    /// charge had: the two engines preempt at different intervals (this one every
    /// [`fuel_interval`] units, native every `QUANTUM_FUEL`), and while the clock was
    /// charged per preemption that ratio WAS a clock error. Charged per unit of fuel, the
    /// interval is free to differ.
    ///
    /// The counter runs DOWN from a full interval and is reloaded after each yield, so the
    /// reading is differenced rather than used directly. A reading ABOVE the last one means
    /// a reload happened in between: the thread burned the rest of the old interval and
    /// then the part of the new one it has already spent.
    ///
    /// # The reading is taken BEFORE the reload, and that is what has to be carried
    /// The emitted fuel check calls the host and only THEN reloads the counter (see
    /// `emit_fuel_check`), so at a fuel yield this reads a spent counter - zero or a little
    /// below it. Recording that as the new baseline makes the NEXT yield difference to
    /// nothing, and a thread that yields on fuel over and over - a spin - is then billed
    /// **zero** for every interval it burns.
    ///
    /// That is not a small error. Measured on a retail title in the browser: 7,953 of 8,000
    /// scheduler rounds were fuel yields by one thread, each burning a full five-million
    /// interval, while the game clock stood still at 3.454 s and the storage transfer the
    /// thread was spinning on could never complete. The run stopped at frame 2 for ever.
    ///
    /// So a spent counter records the baseline the guest is ABOUT to restore - **ZERO**, which
    /// is what the emitted clear leaves - rather than the spent value.
    ///
    /// # MEASURED, and recording `now` instead cost a title a 990 ms frame
    /// The old code recorded `now` and relied on the NEXT reading being smaller to detect the
    /// clear. At a fuel yield the reading is always `interval + overshoot`, and a tight loop's
    /// overshoot is CONSTANT - so the difference is exactly zero, every time, for ever.
    /// Measured on a retail boot in the browser: frame 174 makes **3,176 fuel yields
    /// and bills 1.28 MB of fuel** - 403 per yield against a 5,000,000 interval - and produces
    /// **16 non-zero samples out of 3,176 suspends**, because `charge_guest_work` skips a
    /// suspend that burned nothing. The game clock therefore advanced exactly as much over that
    /// frame as over its quiet neighbours while the frame took 990 ms of wall time.
    fn fuel_used(&mut self) -> Option<u64> {
        let interval = i64::from(fuel_interval());
        let (ops, _) = self.read_work()?;
        let engine = self.engine.as_mut()?;
        let now = ops as i64;
        // Nothing has run since the last reading, so there is nothing to bill. Without this the
        // clear-aware baseline below would re-bill the same spent counter on a second read.
        if now == engine.fuel_raw {
            return Some(engine.fuel_total);
        }
        hostcalls::note_raw_fuel(now);
        let burned = (now - engine.fuel_last).max(0);
        engine.fuel_raw = now;
        // The emitted check calls the host only once the counter has REACHED the interval, and
        // the guest clears it immediately after that call returns (see `emit_fuel_check`), so a
        // reading at or above the interval is a spent counter whose successor starts from zero.
        engine.fuel_last = if now >= interval { 0 } else { now };
        engine.fuel_total = engine.fuel_total.saturating_add(burned as u64);
        Some(engine.fuel_total)
    }

    /// Guest ARM instructions retired, from the high half of the same `work` global.
    ///
    /// Simpler than the operator half: it only counts UP and is never cleared, so the
    /// reading IS a running total and there is no wrap or reload case to reason about.
    fn arm_retired(&mut self) -> Option<u64> {
        let (_, instructions) = self.read_work()?;
        let engine = self.engine.as_mut()?;
        // The half is 32 bits WIDE, so it wraps - at 2^32 retired instructions, which a
        // long-lived thread reaches inside ten seconds of guest time. `saturating_sub` turned
        // that wrap into a zero and then kept returning zeros, which is the emulated CPU clock
        // quietly stopping. Difference it modulo the field instead, which is what a counter of
        // fixed width means.
        const WRAP: u64 = 1 << abi::WORK_INSTR_SHIFT;
        let retired = instructions.wrapping_sub(engine.arm_last) % WRAP;
        engine.arm_last = instructions;
        engine.arm_total = engine.arm_total.saturating_add(retired);
        Some(engine.arm_total)
    }

    /// Hand this thread's instance back: to the engine's POOL if it can be reused, else
    /// dropped outright.
    ///
    /// # Why pooling is not an optimisation here
    /// One instance is one funcref table with an entry per translated function - 106,572
    /// of them on a measured retail title - allocated and eagerly initialized by every
    /// `WebAssembly.Instance` call. This title creates a guest thread PER FRAME (measured
    /// on both engines: one created and one finished, every frame, from frame 1), so
    /// instantiating per thread hands the browser's GC a fresh copy of that table sixty
    /// times a second. The renderer went 875 MB -> 2.19 GB in five seconds and was killed
    /// at frame 22, with the emulator's own wasm heap FLAT at 44 MB throughout - the
    /// growth was never the guest's memory, it was the instances around it. Native does
    /// not care: a wasmtime instance's table is cheap and its store is dropped at once.
    ///
    /// A pooled instance is made indistinguishable from a fresh one by the module's own
    /// `reset` export, which is the only thing that CAN do it completely - see
    /// [`abi::RESET_EXPORT`].
    fn release(&mut self) {
        let Some(mut engine) = self.engine.take() else { return };
        // Break the closure -> register-file reference before dropping or pooling, so
        // nothing the instance can still be reached through outlives this call.
        *engine.rt_cell.borrow_mut() = None;
        engine.signal.borrow_mut().take();
        engine.cont.borrow_mut().take();
        hostcalls::note_thread_released();
        if engine.abandoned.get() || !instance_pool_enabled() {
            drop(engine);
            return;
        }
        // Reset INSIDE the module: it clears the whole ARM and VFP/NEON file, the
        // diagnostic latches, `tp` and the fuel counter. Doing it on release rather than
        // on checkout means an instance is never sitting in the pool holding a finished
        // thread's register values.
        if engine.reset.call0(&JsValue::UNDEFINED).is_err() {
            // A reset that failed leaves an instance nobody can characterise; drop it
            // rather than lend it to the next thread.
            drop(engine);
            return;
        }
        engine.entries.clear();
        engine.entry_idx = 0;
        engine.entry_started = false;
        // ZERO, because that is what the instance's `reset` export leaves in the counter -
        // the same rule `fuel_used` follows after a clear.
        engine.fuel_last = 0;
        engine.fuel_raw = 0;
        engine.fuel_total = 0;
        // The instance's `reset` export zeroes the counter itself; this is the HOST's
        // matching baseline. Leaving it would make the next thread's first delta the whole
        // of the previous thread's total, which is a game clock that jumps by hours.
        engine.arm_last = 0;
        engine.arm_total = 0;
        engine.work_read = None;
        hostcalls::note_instance_pooled();
        self.pool.borrow_mut().push(engine);
    }
}

/// One raw event a running guest entry reports over the JS step channel. Distinct
/// from [`ThreadStep`] because the main thread runs several entries in sequence (the
/// linked title's `module_init`s then the eboot entry): a bare [`Returned`](Ev::Returned)
/// or [`ThreadExit`](Ev::ThreadExit) on a non-final entry advances to the next entry
/// rather than ending the thread. [`resume`] folds these into a `ThreadStep`.
enum Ev {
    /// The entry suspended at a switch point (host call blocked / flipped / preempted).
    Suspend(Stop),
    /// The entry returned normally; the value is r0.
    Returned(u32),
    /// The entry called `sceKernelExitThread`; the value is r0.
    ThreadExit(u32),
    /// A host call halted the whole process; the value is r0.
    Halt(u32),
    /// The entry trapped.
    Error(String),
}

/// Encode an event as a small JS array `[tag, a, b]` for the step channel.
fn encode(ev: &Ev) -> JsValue {
    let f = JsValue::from_f64;
    match ev {
        Ev::Suspend(stop) => {
            let code = match stop {
                Stop::Quantum => 0.0,
                Stop::Blocked => 1.0,
                Stop::Flip => 2.0,
            };
            Array::of2(&f(0.0), &f(code)).into()
        }
        Ev::Returned(c) => Array::of2(&f(1.0), &f(*c as f64)).into(),
        Ev::ThreadExit(c) => Array::of2(&f(2.0), &f(*c as f64)).into(),
        Ev::Halt(c) => Array::of2(&f(3.0), &f(*c as f64)).into(),
        Ev::Error(m) => Array::of3(&f(4.0), &f(0.0), &JsValue::from_str(m)).into(),
    }
}

/// Decode an event the JS channel resolved with.
fn decode(val: &JsValue) -> Ev {
    let arr: Array = val.clone().into();
    let a = arr.get(1).as_f64().unwrap_or(0.0) as u32;
    match arr.get(0).as_f64().unwrap_or(4.0) as u32 {
        0 => Ev::Suspend(match a {
            1 => Stop::Blocked,
            2 => Stop::Flip,
            _ => Stop::Quantum,
        }),
        1 => Ev::Returned(a),
        2 => Ev::ThreadExit(a),
        3 => Ev::Halt(a),
        _ => Ev::Error(arr.get(2).as_string().unwrap_or_default()),
    }
}

/// Deliver `ev` to the current resume's awaiting Promise (one-shot: takes the
/// resolver so a later stray call is a no-op).
fn deliver(signal: &Rc<RefCell<Option<Function>>>, ev: &Ev) {
    if let Some(res) = signal.borrow_mut().take() {
        let _ = res.call1(&JsValue::UNDEFINED, &encode(ev));
    }
}

/// The browser execution engine: the transpiled module, the one shared memory every
/// instance imports, the shared host, and the JSPI primitives. Implements
/// [`GuestEngine`] so it stands up threads for [`SchedCore`].
pub struct BrowserEngine {
    module: WebAssembly::Module,
    shared_mem: WebAssembly::Memory,
    /// The one cached typed-array view over `shared_mem`, handed to every thread and to
    /// every host call. See [`SharedView`].
    view: SharedView,
    host: Host,
    base: u32,
    /// `WebAssembly.promising` (not in the wasm-bindgen bindings; fetched by name).
    promising: Function,
    /// `WebAssembly.Suspending` constructor.
    suspending: Function,
    /// Shared, non-suspending env stubs (`env.svc`, `env.dispatch_miss`), kept alive.
    _svc: Closure<dyn FnMut(i32)>,
    svc_fn: JsValue,
    _dispatch_miss: Closure<dyn FnMut(i32, i32)>,
    dispatch_miss_fn: JsValue,
    /// Linear-memory offset of the host-mirror block, when this build inlined any read
    /// of it (`vitaslop_transpiler::Artifact::mirror_off`). The scheduler refreshes it
    /// before every resume.
    mirror_off: Option<u64>,
    /// Instances given back by finished threads, ready to run another - see
    /// [`ThreadHandle::release`] for why this exists at all.
    pool: InstancePool,
}

impl BrowserEngine {
    /// Stand up one guest thread: an instance importing the shared memory and a
    /// `Suspending` host-call trap, with each of `entries` wrapped by `promising` to be
    /// run in sequence. `(r0, r1)` seed only the first entry.
    ///
    /// The instance comes from the POOL when a finished thread has left one there, and is
    /// instantiated only when the pool is empty. A pooled instance was reset by the module
    /// itself on release, so the two paths are indistinguishable to the guest.
    fn make_thread(
        &self,
        thid: i32,
        entries: &[u32],
        r0: u32,
        r1: u32,
        r2: u32,
        sp: u32,
        priority: i32,
    ) -> Result<BrowserThread, JsValue> {
        let mut engine = match self.pool.borrow_mut().pop() {
            Some(e) => {
                hostcalls::note_instance_reused();
                e
            }
            None => self.new_instance()?,
        };
        // Whoever built it, the instance is now this thread's: the import closure reports
        // the thread through this cell, and the register file is live again.
        engine.thid.set(thid);
        *engine.rt_cell.borrow_mut() = Some(engine.rt.clone());

        // This thread's thread-local storage, mirroring native `instantiate_thread_seq`:
        // allocate the private block whose base is the thread pointer (TPIDRURO), copy
        // the template's initialized `.tdata` head into it (the `.tbss` tail is already
        // zero), and seed the instance's per-thread `tp` global before any entry runs
        // (a `MRC p15,0,Rt,c13,c0,3` reads it). No guest code is running yet, so the
        // shared-memory copy is safe. A title with no TLS template yields tp == 0, and
        // `reset` has already put the global back to 0 for a reused instance.
        let (tp, tls_src, tls_len) = self.host.lock().unwrap().thread_tls_base(thid);
        if tp != 0 {
            if tls_len != 0 {
                let view = &self.view.bytes;
                let src = tls_src.wrapping_sub(self.base);
                let dst = tp.wrapping_sub(self.base);
                let head = view.subarray(src, src + tls_len).to_vec();
                view.subarray(dst, dst + tls_len).copy_from(&head);
            }
            let tp_global = Reflect::get(&engine.exports, &JsValue::from_str(abi::TP_EXPORT))?
                .dyn_into::<WebAssembly::Global>()?;
            tp_global.set_value(&JsValue::from(tp));
        }

        // One `promising` wrapper per entry, in load order. Per THREAD, not per instance:
        // a reused instance runs a different entry, and each wrapper is what allocates
        // the JSPI stack the entry runs on.
        for &entry in entries {
            let entry_fn =
                Reflect::get(&engine.exports, &JsValue::from_str(&abi::func_export(entry & !1)))?
                    .dyn_into::<Function>()?;
            engine.entries.push(
                self.promising.call1(&JsValue::UNDEFINED, &entry_fn)?.dyn_into::<Function>()?,
            );
        }
        engine.sp = sp;
        engine.r0 = r0;
        engine.r1 = r1;
        engine.r2 = r2;

        Ok(BrowserThread {
            thid,
            priority,
            host: self.host.clone(),
            pool: self.pool.clone(),
            engine: Some(engine),
        })
    }

    /// Instantiate the module for one guest thread: its own register file (the globals),
    /// its own import closure, and the two one-slot channels JSPI drives it through.
    /// Everything here is per-INSTANCE and survives into the pool; the per-THREAD arming
    /// is in [`make_thread`].
    fn new_instance(&self) -> Result<ThreadEngine, JsValue> {
        let signal: Rc<RefCell<Option<Function>>> = Rc::new(RefCell::new(None));
        let cont: Rc<RefCell<Option<Function>>> = Rc::new(RefCell::new(None));
        // The import closure needs the instance's globals, which only exist after
        // instantiation - the chicken-and-egg the runtime cell resolves (imports fire
        // only during execution, by when the cell is filled).
        let rt_cell: Rc<RefCell<Option<Rc<ThreadRt>>>> = Rc::new(RefCell::new(None));
        // The thread this instance is running, and whether its stack has been abandoned.
        // Cells rather than captured values because the instance outlives the thread.
        let thid_cell = Rc::new(Cell::new(0i32));
        let abandoned = Rc::new(Cell::new(false));

        let import_closure = {
            let host = self.host.clone();
            let rt_cell = rt_cell.clone();
            let signal = signal.clone();
            let cont = cont.clone();
            let thid_cell = thid_cell.clone();
            let abandoned = abandoned.clone();
            Closure::wrap(Box::new(move |selector: i32| -> JsValue {
                let thid = thid_cell.get();
                // A software fuel point (see `vitaslop_transpiler::emit::set_fuel_interval`)
                // is not a host call and must not be billed as one: it carries no
                // arguments and no return value, so marshalling the register file for it
                // - which is 91% of what a host call costs here - would be pure waste on
                // the one path added specifically to make spinning cheap. Counting it
                // would also put a synthetic entry at the top of every per-NID profile
                // and reset the host-call quantum, which measures a different thing.
                if selector as u32 == vitaslop_transpiler::abi::FUEL_SELECTOR {
                    hostcalls::note_fuel_yield();
                    // The thread is leaving the CPU, so its host-call quantum starts
                    // fresh - the same rule every other suspension here follows. Without
                    // it a thread that fuel-yielded would carry a nearly-spent host-call
                    // budget back in and preempt again almost immediately, billing the
                    // clock twice for one switch.
                    hostcalls::reset_quantum();
                    return suspend(&signal, &cont, Stop::Quantum);
                }
                let timed = hostcalls::timing_enabled();
                let clock = || if timed { hostcalls::now() } else { 0.0 };
                let t0 = clock();
                let rt = rt_cell.borrow().as_ref().expect("rt set before first call").clone();
                let (mut regs, mut vfp) = rt.read_file();
                // What the guest handed in, so the write-back can send back only what moved.
                let before = (regs, vfp);
                let d0 = clock();
                let outcome = {
                    // BORROWED, not cloned - see `impl GuestMemory for &SharedView`.
                    let mut mem: &SharedView = &rt.view;
                    let mut host = host.lock().unwrap();
                    host.set_current_thread(thid);
                    host.dispatch(selector as u32, &mut regs, &mut vfp, &mut mem, rt.base)
                };
                let d1 = clock();
                rt.write_file_changed(&before, &regs, &vfp);
                // Split the call the way native's `perf` module does: the handler versus
                // everything around it. The difference is register MARSHALLING - work the
                // guest never asked for - and the two need completely different fixes.
                let total_ms = clock() - t0;
                hostcalls::note_selector(selector as u32, thid, total_ms);
                if hostcalls::record(total_ms, d1 - d0) {
                    // Name the NIDs the guest is spending its calls on. This is the
                    // browser twin of what native's per-selector `perf` breakdown does,
                    // and it is the difference between "millions of host calls" and
                    // "millions of calls to one specific NID", which are different bugs.
                    let host = host.lock().unwrap();
                    let top: Vec<String> = hostcalls::top_selectors(5)
                        .into_iter()
                        .map(|(sel, n)| match host.import_at(sel) {
                            Some((_, func_nid)) => {
                                format!("{} x{n}", vitaslop_runtime::nid::name(func_nid))
                            }
                            None => format!("selector {sel} x{n}"),
                        })
                        .collect();
                    tracing::info!(target: "vitaslop::perf", "hostcalls by nid: {}", top.join(", "));
                    let threads: Vec<String> = hostcalls::by_thread()
                        .into_iter()
                        .take(6)
                        .map(|(t, n)| format!("thid {t:#x} x{n}"))
                        .collect();
                    tracing::info!(
                        target: "vitaslop::perf",
                        "hostcalls by thread: {}", threads.join(", ")
                    );
                    // The decisive question when a frame will not end is whether the
                    // thread that should finish it is PARKED (and on what) or merely not
                    // being picked. Only this dump answers it, and a browser run has no
                    // other way to ask - there is no session command to type here.
                    tracing::debug!(
                        target: "vitaslop::sched",
                        "stall dump:\n{}", host.state.debug_sync_dump()
                    );
                }
                // Anything but a plain return means the guest is leaving the CPU anyway,
                // so its quantum starts fresh.
                if !matches!(outcome, SvcOutcome::Continue) {
                    hostcalls::reset_quantum();
                }
                match outcome {
                    // Plain return: the guest continues without suspending (the cheap,
                    // common path - most host calls just return a value)... unless this
                    // thread has used its quantum, in which case preempt it here. This is
                    // the browser's stand-in for native's fuel interrupt; see
                    // [`preempt_note`] for why the run cannot work without one.
                    SvcOutcome::Continue if hostcalls::quantum_expired() => {
                        suspend(&signal, &cont, Stop::Quantum)
                    }
                    SvcOutcome::Continue => JsValue::UNDEFINED,
                    // A switch point: tell the scheduler why we stopped, then return a
                    // pending Promise so the guest stack suspends until it resolves.
                    SvcOutcome::Reschedule => suspend(&signal, &cont, Stop::Quantum),
                    SvcOutcome::Block => suspend(&signal, &cont, Stop::Blocked),
                    SvcOutcome::Flip => suspend(&signal, &cont, Stop::Flip),
                    // The thread (or process) ends here: report the event and park on a
                    // never-resolving Promise (this stack is abandoned - on a thread exit
                    // the scheduler may still start the thread's next entry on a fresh
                    // stack; on a halt the run is over).
                    SvcOutcome::ThreadExit => {
                        abandoned.set(true);
                        deliver(&signal, &Ev::ThreadExit(regs[0]));
                        never()
                    }
                    SvcOutcome::Halt => {
                        abandoned.set(true);
                        deliver(&signal, &Ev::Halt(regs[0]));
                        never()
                    }
                    // Unfaithful call (e.g. unimplemented NID): stop the run loudly as
                    // an error rather than fake a success (which would desync the guest).
                    SvcOutcome::Fatal(msg) => {
                        abandoned.set(true);
                        deliver(&signal, &Ev::Error(msg));
                        never()
                    }
                }
            }) as Box<dyn FnMut(i32) -> JsValue>)
        };

        // env.import wrapped as Suspending; env.memory the shared memory; env.svc /
        // env.dispatch_miss the shared non-suspending stubs.
        let suspending_import = Reflect::construct(
            &self.suspending,
            &Array::of1(import_closure.as_ref().unchecked_ref()),
        )?;
        let env = Object::new();
        Reflect::set(&env, &JsValue::from_str(abi::MEMORY_EXPORT), &self.shared_mem)?;
        Reflect::set(&env, &JsValue::from_str(abi::IMPORT_NAME), &suspending_import)?;
        Reflect::set(&env, &JsValue::from_str(abi::SVC_NAME), &self.svc_fn)?;
        Reflect::set(&env, &JsValue::from_str(abi::DISPATCH_MISS_NAME), &self.dispatch_miss_fn)?;
        let imports = Object::new();
        Reflect::set(&imports, &JsValue::from_str(abi::IMPORT_MODULE), &env)?;

        let instance = WebAssembly::Instance::new(&self.module, &imports)?;
        hostcalls::note_instance_created();
        let exports = instance.exports();

        let regs = read_globals(&exports, |i| abi::reg_export(i), abi::REG_COUNT)?;
        let vfp = read_globals(&exports, |i| abi::vfp_s_export(i as u8), VFP_ARG_COUNT)?;
        // The same globals as one JS array, ARM registers first, so a host call marshals
        // the whole file in one crossing rather than one per register.
        let file = Array::new();
        for g in regs.iter().chain(vfp.iter()) {
            file.push(g);
        }
        // One flat list in the same order as `file`, so a lane index means the same thing to
        // the bulk path and to the single-lane path.
        let globals: Vec<WebAssembly::Global> = regs.into_iter().chain(vfp).collect();
        let narrow = js_sys::Uint32Array::new_with_length(NARROW_REGS.len() as u32);
        narrow.copy_from(&NARROW_REGS);
        let rt = Rc::new(ThreadRt {
            regs: globals,
            file,
            narrow,
            view: self.view.clone(),
            base: self.base,
        });

        // The module's own reset (see `abi::RESET_EXPORT`), resolved once here so
        // releasing a thread is a single call and cannot fail on a lookup.
        let reset = Reflect::get(&exports, &JsValue::from_str(abi::RESET_EXPORT))?
            .dyn_into::<Function>()?;

        Ok(ThreadEngine {
            rt,
            rt_cell,
            entries: Vec::new(),
            entry_idx: 0,
            entry_started: false,
            sp: 0,
            r0: 0,
            r1: 0,
            r2: 0,
            signal,
            cont,
            _import: import_closure,
            // Absent in a build with `VITASLOP_BROWSER_FUEL=0`, which is exactly the
            // build that has no fuel to report; the clock then falls back to advancing
            // on flips and idles alone, as it did before fuel existed.
            fuel: Reflect::get(&exports, &JsValue::from_str(abi::FUEL_EXPORT))
                .ok()
                .and_then(|g| g.dyn_into::<WebAssembly::Global>().ok()),
            fuel_last: 0,
            fuel_raw: 0,
            fuel_total: 0,
            arm_last: 0,
            work_read: None,
            arm_total: 0,
            exports,
            reset,
            thid: thid_cell,
            abandoned,
        })
    }
}

impl GuestEngine for BrowserEngine {
    type Thread = BrowserThread;

    fn spawn(&mut self, r: &Reentry) -> Result<BrowserThread, ()> {
        self.make_thread(r.thid, &[r.entry], r.arg_len, r.arg_ptr, r.r2, r.stack_top, r.priority)
            .map_err(|_| ())
    }

    fn write_mem(&mut self, addr: u32, bytes: &[u8]) {
        let off = addr.wrapping_sub(self.base) as usize;
        let view = &self.view.bytes;
        if off + bytes.len() <= view.length() as usize {
            view.subarray(off as u32, (off + bytes.len()) as u32).copy_from(bytes);
            // A scheduler-side write is a host write like any other - see
            // `SharedView::stamp_written`.
            self.view.stamp_written(off, bytes.len());
        }
    }

    fn read_mem(&self, addr: u32, out: &mut [u8]) -> bool {
        let off = addr.wrapping_sub(self.base) as usize;
        let view = &self.view.bytes;
        if off.checked_add(out.len()).is_none_or(|end| end > view.length() as usize) {
            return false;
        }
        view.subarray(off as u32, (off + out.len()) as u32).copy_to(out);
        true
    }

    fn mirror_base(&self) -> Option<u32> {
        // The block sits above the guest region, so its guest address is the rebase
        // origin plus the offset - the same convention `write_mem` undoes.
        self.mirror_off.map(|off| self.base.wrapping_add(off as u32))
    }
}

/// Build the pending-Promise a suspended thread parks on, and signal the scheduler
/// with the stop reason. Returned from the import closure so the guest stack suspends.
fn suspend(
    signal: &Rc<RefCell<Option<Function>>>,
    cont: &Rc<RefCell<Option<Function>>>,
    stop: Stop,
) -> JsValue {
    let cont = cont.clone();
    let park = Promise::new(&mut |resolve, _reject| {
        *cont.borrow_mut() = Some(resolve);
    });
    hostcalls::note_suspend();
    deliver(signal, &Ev::Suspend(stop));
    park.into()
}

/// >>> GIVE THE HOST'S EVENT LOOP A TURN. THE IDLE PATH HAD NO WAY TO.
///
/// A resume awaits a Promise that JSPI resolves, and a resolved Promise is a MICROTASK - the
/// microtask checkpoint drains without ever running a task. So the whole of this scheduler
/// loop, including the idle path, executes inside one task: `handle_idle` is synchronous, an
/// idle round `continue`s, and the worker can spin for the entire length of a frame without
/// the event loop turning over once.
///
/// Which is fine until the emulator is waiting for something only the event loop can deliver.
/// `VideoDecoder` hands its pictures back through an output callback, and that callback is a
/// TASK. So the movie thread parks for 2 ms of guest time, the scheduler finds nothing
/// runnable, spins the idle path until the park expires, resumes the thread, asks the decoder
/// again - and the decoder has still not been given a single moment in which to answer.
///
/// MEASURED: **120 access units submitted and no pictures at all** on a device, and 91 ms to
/// the first picture on this machine. The park was added to fix exactly this and could not,
/// because parking a guest THREAD is not the same as yielding the WORKER.
///
/// A `MessageChannel` message is the cheapest real task there is - unlike `setTimeout(0)`,
/// which a worker clamps to 4 ms ([[vitaslop-worker-settimeout-is-clamped]]).
async fn event_loop_turn() {
    EVENT_LOOP_TURNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    thread_local! {
        static CHANNEL: Option<web_sys::MessageChannel> = web_sys::MessageChannel::new().ok();
    }
    let promise = Promise::new(&mut |resolve, _reject| {
        let posted = CHANNEL.with(|ch| {
            let Some(ch) = ch else { return false };
            let cb = Closure::once_into_js(move |_: JsValue| {
                let _ = resolve.call0(&JsValue::UNDEFINED);
            });
            ch.port1().set_onmessage(Some(cb.as_ref().unchecked_ref()));
            ch.port2().post_message(&JsValue::NULL).is_ok()
        });
        if !posted {
            // No channel is not a thing a worker does, but resolving immediately degrades to
            // the old always-spin behaviour rather than hanging the run.
            let cb = Closure::once_into_js(move |_: JsValue| {});
            let _ = js_sys::Function::from(cb).call0(&JsValue::UNDEFINED);
        }
    });
    let _ = JsFuture::from(promise).await;
}

/// How many turns the worker's event loop has been given from INSIDE a frame.
///
/// The tick loop returns to the event loop once per tick whatever happens, so a host reply
/// that needs a task always gets at least that. This counts the EXTRA turns the idle path
/// hands out - and a run where it reads zero is a run where a callback-driven decoder was
/// offered exactly one chance per displayed frame to answer, which for a 30 fps movie on a
/// 60 Hz display is already the whole of its budget.
static EVENT_LOOP_TURNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many turns the idle path has handed the event loop so far this run.
pub fn event_loop_turns() -> u64 {
    EVENT_LOOP_TURNS.load(std::sync::atomic::Ordering::Relaxed)
}

/// How many CONSECUTIVE idle rounds may pass before the loop hands the event loop a turn.
///
/// Not every idle round: a turn is a real task dispatch, and the idle path can run thousands
/// of rounds in a frame while the virtual clock walks to the next deadline - paying a task
/// for each would make idling cost more than running. Not never, either, for the reason
/// above. A run of idle rounds means the emulator has nothing to do, which is exactly when
/// an outstanding host reply is the thing it is waiting for, and the counter resets the
/// moment any thread becomes runnable - so a busy frame never reaches it at all.
const IDLE_ROUNDS_PER_EVENT_LOOP_TURN: u64 = 64;

/// A Promise that never resolves - a finished thread's stack parks here forever (it is
/// never resumed), the browser analog of a fiber that has returned.
fn never() -> JsValue {
    hostcalls::note_abandoned_stack();
    Promise::new(&mut |_resolve, _reject| {}).into()
}

/// Fetch `n` exported globals named `name(0)..name(n-1)`.
fn read_globals(
    exports: &JsValue,
    name: impl Fn(usize) -> String,
    n: usize,
) -> Result<Vec<WebAssembly::Global>, JsValue> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(
            Reflect::get(exports, &JsValue::from_str(&name(i)))?
                .dyn_into::<WebAssembly::Global>()?,
        );
    }
    Ok(v)
}

/// Resume `t` to its next scheduler-visible switch point (or its end), asynchronously.
/// A suspend returns immediately; a bare entry return / thread-exit on a NON-final
/// entry does not return to the scheduler - it starts the next entry (a fresh stack)
/// and keeps going, so the whole `module_init` sequence runs as one uninterrupted main
/// thread (matching native's `instantiate_thread_seq`). Only a suspend, a halt, a trap,
/// or the final entry ending yields a [`ThreadStep`].
async fn resume(t: &mut BrowserThread) -> ThreadStep {
    let thid = t.thid;
    let host = t.host.clone();
    // A released thread is a finished one, and `pick_next` never returns a finished
    // thread - so reaching here without engine state is a scheduler bug, not an input.
    let t = t.engine.as_mut().expect("resume of a released (finished) thread");
    // The guest is about to run, so the cached `work` reading stops being current. See
    // `work_read`.
    t.work_read = None;
    loop {
        // A fresh step channel for this turn; the import closure or the entry's
        // completion fills its resolver.
        let mut resolver = None;
        let step_promise = Promise::new(&mut |res, _rej| resolver = Some(res));
        *t.signal.borrow_mut() = Some(resolver.expect("Promise executor runs synchronously"));

        if !t.entry_started {
            t.entry_started = true;
            // Each entry starts on a fresh stack; only the first carries args.
            t.rt.set_reg(abi::SP, t.sp);
            t.rt.set_reg(0, if t.entry_idx == 0 { t.r0 } else { 0 });
            t.rt.set_reg(1, if t.entry_idx == 0 { t.r1 } else { 0 });
            t.rt.set_reg(2, if t.entry_idx == 0 { t.r2 } else { 0 });
            hostcalls::note_stack_start();
            // Which thread started a stack, and which of its entries. A count alone says
            // stacks are being created; it cannot say whether that is a handful of guest
            // threads each running a long entry sequence once, or one thread starting
            // entries over and over - and those are a normal run and a leak respectively.
            tracing::debug!(
                target: "vitaslop::sched",
                "jspi stack start: thid {thid:#x} entry {}/{}",
                t.entry_idx,
                t.entries.len()
            );
            let done: Promise = match t.entries[t.entry_idx].call0(&JsValue::UNDEFINED) {
                Ok(p) => p.unchecked_into(),
                Err(e) => return ThreadStep::Finished(FiberEnd::Error(format!("start: {e:?}"))),
            };
            // When this entry returns (with no final host call) or traps, deliver the
            // event through whatever step channel is current at that time.
            let sig_ok = t.signal.clone();
            let rt = t.rt.clone();
            let on_ok = Closure::once(Box::new(move |_v: JsValue| {
                deliver(&sig_ok, &Ev::Returned(rt.read_reg(0)));
            }) as Box<dyn FnOnce(JsValue)>);
            let sig_err = t.signal.clone();
            let on_err = Closure::once(Box::new(move |e: JsValue| {
                let msg = e.as_string().unwrap_or_else(|| format!("{e:?}"));
                deliver(&sig_err, &Ev::Error(msg));
            }) as Box<dyn FnOnce(JsValue)>);
            let _ = done.then2(&on_ok, &on_err);
            on_ok.forget();
            on_err.forget();
        } else if t.cont.borrow().is_none() {
            // NEITHER branch: the entry is already running and nothing is parked, so this
            // resume starts nothing and un-parks nothing. It can only be waiting for an
            // event some earlier turn already consumed - a scheduler-state bug, not guest
            // behaviour - and it is invisible from outside because the thread burns no
            // fuel and makes no host call while it happens.
            tracing::warn!(
                target: "vitaslop::sched",
                "resume of thid {thid:#x} with no parked continuation and its entry \
                 already started: nothing to run"
            );
        } else if let Some(res) = t.cont.borrow_mut().take() {
            // A timed wait that expired owes this thread a return code other than the
            // 0 it parked with (a WAIT_TIMEOUT); write it into r0 before the guest
            // stack resumes. A signal wake has no code and keeps r0 = 0. (Native does
            // the equivalent inside its import closure after the block await.)
            if let Some(code) = host.lock().unwrap().take_resume_code(thid) {
                t.rt.set_reg(0, code);
            }
            // Un-park: resolving the parked Promise resumes the suspended guest stack.
            let _ = res.call0(&JsValue::UNDEFINED);
        }

        let ev = match JsFuture::from(step_promise).await {
            Ok(v) => decode(&v),
            Err(e) => return ThreadStep::Finished(FiberEnd::Error(format!("resume: {e:?}"))),
        };

        let last = t.entry_idx + 1 >= t.entries.len();
        match ev {
            Ev::Suspend(stop) => return ThreadStep::Suspended(stop),
            Ev::Halt(c) => return ThreadStep::Finished(FiberEnd::ProcessHalt(c)),
            Ev::Error(m) => return ThreadStep::Finished(FiberEnd::Error(m)),
            Ev::Returned(c) if last => return ThreadStep::Finished(FiberEnd::Returned(c)),
            Ev::ThreadExit(c) if last => return ThreadStep::Finished(FiberEnd::ThreadExit(c)),
            // A non-final entry ended (returned or called ExitThread): advance to the
            // next one on a fresh stack, without returning to the scheduler.
            Ev::Returned(_) | Ev::ThreadExit(_) => {
                t.entry_idx += 1;
                t.entry_started = false;
            }
        }
    }
}

/// How often [`run_frames`] reports its round count.
///
/// Sized against the observed round rate, not guessed: with preemption at ~1,300 host
/// calls a long frame retires a few thousand rounds a second, so a 100,000-round window
/// would have said nothing for minutes - which is exactly the silence this report exists
/// to break.
/// >>> AND SIZED SMALL, BECAUSE A LONG FRAME IS NOT ALWAYS A BUSY ONE.
///
/// The reasoning above assumes a frame that grinds retires rounds while it grinds. A frame
/// that blocks - one host call reading tens of megabytes off storage, one driver call
/// compiling a pipeline - retires almost NONE, so a window of thousands never fires and the
/// page shows the last frame it FINISHED for as long as the block lasts. That is exactly the
/// report this exists to prevent, and it is the shape a user describes as "it never went to
/// frame 2".
///
/// The callback is cheap to reach (its first act is a rate-limited emit, ten a second), so the
/// cost of a small window is one clock read per 64 rounds - unmeasurable next to a round, which
/// is a resume of translated guest code.
const PROGRESS_ROUNDS: u64 = 64;

/// How often the heavier per-frame report - the game clock, the I/O waiters and the hottest
/// NIDs - is built. Coarse on purpose: it locks the host and walks a histogram, which answers
/// "what is it spinning ON" and is only a question worth asking of a frame that IS spinning.
const LONG_FRAME_ROUNDS: u64 = 2_000;

/// The browser preemptive run loop: the async twin of native's
/// `Scheduler::run_frames`, composing the shared [`SchedCore`]. Runs until the process
/// halts, all threads finish, the run deadlocks, or `max_frames`/`max_rounds` is hit.
///
/// `progress` is called every [`PROGRESS_ROUNDS`] scheduler rounds with the count so far.
/// A single guest frame can be MILLIONS of rounds - the first one runs every
/// `module_init` and the whole eboot entry before the title's first display flip - and
/// until this existed such a frame was completely silent: the page reported the frame
/// number it had last FINISHED, so a healthy run grinding through a heavy frame and a
/// hung one printed the identical line for minutes. The round count is what tells them
/// apart, and its rate is the only direct read on how fast the browser executes guest
/// code at all.
pub async fn run_frames(
    core: &mut SchedCore<BrowserEngine, VitaEnv>,
    max_frames: u64,
    max_rounds: u64,
    progress: &mut dyn FnMut(u64),
) -> RunReport {
    let mut rounds = 0u64;
    // What the rounds ARE: idle-path turns (nothing runnable), and resumes split by why
    // the thread stopped. A round count alone cannot tell a scheduler spinning on its own
    // idle path from a guest being resumed and immediately blocking again, and those have
    // nothing in common but the symptom.
    let (mut idle_rounds, mut n_quantum, mut n_blocked, mut n_flip) = (0u64, 0u64, 0u64, 0u64);
    // Idle rounds since the last time a thread was runnable - see `event_loop_turn`.
    let mut consecutive_idle = 0u64;
    // >>> A LONG WORK SLICE HANDS THE EVENT LOOP A TURN ON WALL CLOCK, RUNNABLE OR NOT.
    //
    // The idle-path turns above fire only when nothing is runnable - and a heavy frame is
    // exactly when something always is. On the device that shipped this: a boot frame ran
    // 1,077 ms and a course-load frame 261 ms inside ONE task, so for that whole stretch
    // the worker's event loop never turned - the video decoder's output callback (a TASK)
    // could not deliver a picture, the input port's messages queued, and the page read as
    // hung. The movie panel's `7 still owed` and the user's "browser hangs during loads"
    // are the same starvation seen from two sides.
    //
    // This is deliberately GENERAL, not a boot special case: ANY slice that runs longer
    // than the budget yields once, wherever it happens - boot, course load, a pathological
    // in-round frame. The clock is read every TURN_CHECK_ROUNDS rounds rather than every
    // round because `performance.now()` is a JS call and a busy frame runs tens of
    // thousands of rounds; at 64 that is a few hundred reads across the longest frame ever
    // seen. The turn itself is the same MessageChannel task the idle path uses - real
    // enough to run queued tasks, cheap enough to take twenty times in a long load.
    const TURN_CHECK_ROUNDS: u64 = 64;
    const SLICE_BUDGET_MS: f64 = 20.0;
    let mut slice_start = perf_clock();
    loop {
        if rounds >= max_rounds {
            return RunReport::RoundLimit;
        }
        rounds += 1;
        if rounds % TURN_CHECK_ROUNDS == 0 && perf_clock() - slice_start > SLICE_BUDGET_MS {
            event_loop_turn().await;
            slice_start = perf_clock();
        }
        // `rounds == 1` as well as the window: a frame that blocks on its FIRST call would
        // otherwise say nothing at all, and "frame N in progress: 1 round" against a frozen
        // clock is the whole diagnosis.
        if rounds == 1 || rounds % PROGRESS_ROUNDS == 0 {
            progress(rounds);
        }
        // The HEAVY half of the report keeps the coarse window: it takes the host lock and
        // builds a selector histogram, which is not something to do every 64 rounds. The
        // cheap half above is what a blocked frame needs; this is what a SPINNING one does.
        if rounds % LONG_FRAME_ROUNDS == 0 {
            // What a long frame is actually DOING, unconditionally: the game clock (a
            // frame that grinds with a FROZEN clock is a livelock, one that grinds with a
            // moving clock is just slow, and those need opposite fixes), and the NIDs the
            // rounds are going into. The per-selector histogram already existed and
            // nothing ever printed it, so "which call is it spinning on" - the one
            // question a runaway frame asks - had no answer in a browser run.
            let (clock_us, io_waiters) = {
                let host = core.host().lock().unwrap();
                (host.state.now_us(), host.state.has_io_waiters())
            };
            let top: Vec<String> = hostcalls::top_selectors(4)
                .into_iter()
                .map(|(sel, n)| format!("sel {sel}: {n}"))
                .collect();
            tracing::info!(
                target: "vitaslop::sched",
                "long frame: {rounds} rounds ({idle_rounds} idle, resumes: {n_quantum} \
                 quantum / {n_blocked} blocked / {n_flip} flip), clock {:.3}s, \
                 io_waiters={io_waiters}, hottest calls [{}]",
                clock_us as f64 / 1e6,
                top.join(", "),
            );
        }

        // >>> THE BROWSER'S OWN SCHEDULER WORK, WHICH WAS NEVER TIMED HERE.
        //
        // `Phase::SchedOverhead` is charged by the NATIVE scheduler's `run_frames`, and the
        // browser drives `SchedCore` from this loop instead - so on the engine that ships,
        // the scheduler's share of a frame was simply absent from every report. On the
        // desktop the same phase is **11.2% of a frame on a retail title**, and that title switches
        // threads 230 times a frame, so "absent" is not the same as "small".
        //
        // Timed around the PICK and the DRAIN, never around the resume: resuming runs the
        // guest, and timing that would measure the guest. Same rule the native loop states.
        let pick = vitaslop_runtime::perf::scope(vitaslop_runtime::perf::Phase::SchedOverhead);
        let picked = core.pick_next();
        drop(pick);
        let Some(idx) = picked else {
            idle_rounds += 1;
            consecutive_idle += 1;
            let idle = vitaslop_runtime::perf::scope(vitaslop_runtime::perf::Phase::SchedIdle);
            let step = core.handle_idle();
            drop(idle);
            match step {
                IdleStep::Done(report) => return report,
                IdleStep::Continue => {
                    // Nothing is runnable, so let the host deliver whatever it owes us before
                    // asking again. See `event_loop_turn`.
                    //
                    // >>> AND WHEN A DECODER OWES PICTURES, SIXTY-FOUR ROUNDS IS A CEILING IT
                    // >>> MAY NEVER REACH.
                    //
                    // The flat threshold assumes a long idle stretch is what an outstanding
                    // host reply looks like. On a title with many live threads it is not:
                    // MEASURED through the shipping page, a title's front end read `event loop:
                    // 0 extra turns from the idle path` for a whole run - the scheduler never
                    // saw sixty-four CONSECUTIVE idle rounds, because something always became
                    // runnable first - while its movie read `10 access units submitted, 0
                    // pictures delivered, 100% of calls empty`. The decoder's entire budget was
                    // the one turn the tick gives per displayed frame, and delivering a picture
                    // takes two (the output callback, then the async copy out of the frame).
                    //
                    // So while pictures are owed, the turns come on a POWER-OF-TWO cadence:
                    // immediately when idling starts, which is exactly when the decoder needs
                    // one, and thinning out to the old rate over a long stretch. That bounds
                    // the extra turns to ~log2(rounds) per idle run, which is what keeps the
                    // "a task per idle round costs more than running" objection from applying.
                    let owed = vitaslop_runtime::vita::avcdec::pictures_owed() > 0;
                    let turn = if owed {
                        consecutive_idle.is_power_of_two()
                    } else {
                        consecutive_idle % IDLE_ROUNDS_PER_EVENT_LOOP_TURN == 0
                    };
                    if turn {
                        event_loop_turn().await;
                        // The slice clock measures time since the event loop last turned,
                        // whatever caused the turn.
                        slice_start = perf_clock();
                    }
                    continue;
                }
            }
        };
        consecutive_idle = 0;

        // A resume is the only AWAIT in this loop, so a resume that never comes back stops
        // the run with no other trace: the round counter stops, no fuel is burned, no host
        // call is made, and the frame never completes - which reads exactly like a quiet
        // deadlock but is not one. This names the thread that went in, so the last line
        // before the silence is the answer (`vitaslop::sched=trace`).
        tracing::trace!(
            target: "vitaslop::sched",
            "resume thid={:#x} round={rounds}",
            core.thread_mut(idx).thid(),
        );
        let step = resume(core.thread_mut(idx)).await;
        if let ThreadStep::Suspended(stop) = &step {
            match stop {
                Stop::Quantum => n_quantum += 1,
                Stop::Blocked => n_blocked += 1,
                Stop::Flip => n_flip += 1,
            }
        }
        // The post-resume bookkeeping is scheduler work too, and it is where the guest
        // clock is charged and the wake queue is applied.
        let book = vitaslop_runtime::perf::scope(vitaslop_runtime::perf::Phase::SchedBook);
        let done = match step {
            ThreadStep::Finished(end) => core.on_finished(idx, end),
            ThreadStep::Suspended(stop) => core.on_suspended(idx, stop, max_frames),
        };
        if let Some(report) = done {
            return report;
        }
        // A host call in this resume may have started threads or woken parked ones.
        core.drain();
        drop(book);
    }
}

/// Compile a transpiled module asynchronously. `WebAssembly.Module::new` (sync) is
/// disallowed on the main thread for modules over 8 MB - a real title easily exceeds
/// that - so use async `WebAssembly.compile`, which the caller (already async) awaits.
pub async fn compile_module(wasm: &[u8]) -> Result<WebAssembly::Module, JsValue> {
    let promise = WebAssembly::compile(&Uint8Array::from(wasm).into());
    JsFuture::from(promise).await?.dyn_into::<WebAssembly::Module>()
}

/// The JSPI primitives and a fresh shared memory, ready to build a [`SchedCore`].
pub struct BrowserSched {
    pub core: SchedCore<BrowserEngine, VitaEnv>,
    pub host: Host,
}

impl BrowserSched {
    /// Stand up a preemptive run of `wasm` (the transpiler's `import_memory` module for
    /// a guest loaded at `base`, sized `mem_pages`), seeding `image` into a fresh shared
    /// memory and the main thread ready to run from `entry`. `env` is the single-owner
    /// host every thread dispatches its NID calls to.
    pub fn new(
        module: WebAssembly::Module,
        image: &[u8],
        base: u32,
        mem_pages: u32,
        mirror_off: Option<u64>,
        dirty_off: Option<u64>,
        entry: u32,
        main_sp: u32,
        env: VitaEnv,
    ) -> Result<BrowserSched, JsValue> {
        let (engine, host) =
            build_engine(module, image, base, mem_pages, mirror_off, dirty_off, env)?;
        let main = engine.make_thread(
            0,
            &[entry & !1],
            0,
            0,
            0,
            main_sp,
            vitaslop_runtime::host::DEFAULT_THREAD_PRIORITY,
        )?;
        let core = SchedCore::new(engine, host.clone(), main);
        Ok(BrowserSched { core, host })
    }

    /// Stand up a preemptive run whose main thread runs `entries` in sequence (a linked
    /// title's `module_init`s in load order, then the eboot entry) on one instance - the
    /// browser twin of native's `ThreadedScheduler::from_linked`. `env` should already
    /// have its alloc base / process param / preemptive flag set and its guest files
    /// preloaded.
    pub fn from_linked(
        module: WebAssembly::Module,
        image: &[u8],
        base: u32,
        mem_pages: u32,
        mirror_off: Option<u64>,
        dirty_off: Option<u64>,
        entries: &[u32],
        main_sp: u32,
        env: VitaEnv,
    ) -> Result<BrowserSched, JsValue> {
        let (engine, host) =
            build_engine(module, image, base, mem_pages, mirror_off, dirty_off, env)?;
        let main = engine.make_thread(
            0,
            entries,
            0,
            0,
            0,
            main_sp,
            vitaslop_runtime::host::DEFAULT_THREAD_PRIORITY,
        )?;
        let core = SchedCore::new(engine, host.clone(), main);
        Ok(BrowserSched { core, host })
    }
}

/// Build the browser engine (JSPI primitives, module, a fresh seeded shared memory) and
/// the shared host - the common setup both [`BrowserSched`] constructors share.
fn build_engine(
    module: WebAssembly::Module,
    image: &[u8],
    base: u32,
    mem_pages: u32,
    mirror_off: Option<u64>,
    dirty_off: Option<u64>,
    env: VitaEnv,
) -> Result<(BrowserEngine, Host), JsValue> {
    {
        let wasm_global =
            Reflect::get(&js_sys::global(), &JsValue::from_str("WebAssembly"))?;
        let promising = Reflect::get(&wasm_global, &JsValue::from_str("promising"))?
            .dyn_into::<Function>()
            .map_err(|_| JsValue::from_str("WebAssembly.promising missing (needs JSPI)"))?;
        let suspending = Reflect::get(&wasm_global, &JsValue::from_str("Suspending"))?
            .dyn_into::<Function>()
            .map_err(|_| JsValue::from_str("WebAssembly.Suspending missing (needs JSPI)"))?;

        // One shared memory of exactly the transpiler's declared size, imported into
        // every instance. A shared memory needs a maximum and a cross-origin-isolated
        // page (COOP/COEP).
        let desc = Object::new();
        Reflect::set(&desc, &JsValue::from_str("initial"), &JsValue::from_f64(mem_pages as f64))?;
        Reflect::set(&desc, &JsValue::from_str("maximum"), &JsValue::from_f64(mem_pages as f64))?;
        Reflect::set(&desc, &JsValue::from_str("shared"), &JsValue::TRUE)?;
        let shared_mem = WebAssembly::Memory::new(&desc)?;
        // The one view over it, for the life of the run. Sound because `initial ==
        // maximum` above makes this memory non-growable and its SharedArrayBuffer never
        // detaches - see [`SharedView`] for why rebuilding it per access was the whole
        // browser performance problem.
        let buffer = shared_mem.buffer();
        let view = SharedView {
            bytes: Uint8Array::new(&buffer),
            // Over the SAME buffer from offset 0, so a rebased byte offset indexes as
            // `off >> 2` / `off >> 1`. Built once, for the reason the byte view is.
            words: js_sys::Uint32Array::new(&buffer),
            halves: js_sys::Uint16Array::new(&buffer),
            dirty_off,
        };
        // Seed the image at offset 0.
        view.bytes.subarray(0, image.len() as u32).copy_from(image);

        // Shared non-suspending env stubs. svc is unused on the Vita path; a dispatch
        // miss (an indirect call to an untranslated target) throws a clear error.
        let svc = Closure::wrap(Box::new(|_sel: i32| {}) as Box<dyn FnMut(i32)>);
        let svc_fn: JsValue = svc.as_ref().clone();
        let dispatch_miss = Closure::wrap(Box::new(|target: i32, caller: i32| -> () {
            let msg = format!(
                "indirect dispatch to unknown target {:#010x} from f_{:x}",
                target as u32, caller as u32
            );
            wasm_bindgen::throw_str(&msg)
        }) as Box<dyn FnMut(i32, i32)>);
        let dispatch_miss_fn: JsValue = dispatch_miss.as_ref().clone();

        // Say how this run preempts before any guest code runs, so the one behavioural
        // difference from native is on the record rather than inferred from timings.
        preempt_note();

        let host: Host = Arc::new(Mutex::new(env));
        let engine = BrowserEngine {
            module,
            shared_mem,
            view,
            host: host.clone(),
            base,
            promising,
            suspending,
            _svc: svc,
            svc_fn,
            _dispatch_miss: dispatch_miss,
            dispatch_miss_fn,
            mirror_off,
            pool: Rc::new(RefCell::new(Vec::new())),
        };

        Ok((engine, host))
    }
}

/// Guest address of each transpiled function, indexed by wasm function index minus
/// `abi::IMPORT_FUNC_COUNT`. Recorded once, when the module is built.
static FUNC_ADDRS: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();

/// Record the emitted module's function table so a trap backtrace can name guest code.
pub fn record_function_addresses(addrs: Vec<u32>) {
    let _ = FUNC_ADDRS.set(addrs);
}

/// Rewrite `wasm-function[N]` in a V8 stack to name the GUEST function it is.
///
/// # Why the browser needs its own copy of this
/// The native engine prints `<wasm function N>` and this prints `wasm-function[N]`, and the
/// mapping is the same arithmetic in both: `funcs[N - IMPORT_FUNC_COUNT]`, the imports
/// occupying the low indices. What differs is who reads it. A native backtrace is read by
/// someone with the module on disk; a browser one is read off a phone screen, pasted into a
/// chat window, by someone who cannot resolve a module index at all.
///
/// MEASURED on the device: a guest fault came back as ten frames of bare indices
/// (`wasm-function[5362]`, `[12889]` repeating). Named, the same stack says the fault is in
/// guest `0x81134030` and that the repeating frame is the INDIRECT-CALL DISPATCHER - i.e.
/// a guest routine recursing through function pointers, which is a description of the bug.
pub fn name_guest_frames(s: &str) -> String {
    let Some(addrs) = FUNC_ADDRS.get() else { return s.to_string() };
    const MARK: &str = "wasm-function[";
    let mut out = String::with_capacity(s.len() + 32);
    let mut rest = s;
    while let Some(at) = rest.find(MARK) {
        let (head, tail) = rest.split_at(at + MARK.len());
        out.push_str(head);
        let end = tail.find(']').unwrap_or(tail.len());
        let (num, after) = tail.split_at(end);
        out.push_str(num);
        if let Ok(widx) = num.trim().parse::<usize>() {
            match widx
                .checked_sub(vitaslop_transpiler::abi::IMPORT_FUNC_COUNT as usize)
                .and_then(|i| addrs.get(i))
            {
                Some(a) => out.push_str(&format!("={a:#010x}")),
                // The dispatcher and `reset` are emitted above every guest function. Saying
                // so is as useful as an address: a stack that ALTERNATES with the dispatcher
                // is an indirect-call chain.
                None => out.push_str("=dispatcher"),
            }
        }
        rest = after;
    }
    out.push_str(rest);
    out
}
