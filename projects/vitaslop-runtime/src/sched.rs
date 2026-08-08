//! The engine-agnostic preemptive scheduler policy: many guest threads sharing one
//! guest address space, run one-at-a-time and switched at their blocking points.
//!
//! This is the *policy* half of the preemptive model - the part that is identical
//! whatever runs the guest code underneath. Native uses wasmtime async fibers; the
//! browser uses Web Workers over a shared `SharedArrayBuffer`. Both drive this same
//! loop through two small traits:
//!
//! - [`GuestThread`] is one suspendable guest thread. [`resume`](GuestThread::resume)
//!   runs it (servicing however many non-blocking host calls it makes) until it hits
//!   a switch point - a block, a frame flip, or a preemption slice - or finishes.
//! - [`GuestEngine`] stands up a new thread ([`spawn`](GuestEngine::spawn)) and
//!   writes shared guest memory ([`write_mem`](GuestEngine::write_mem)).
//!
//! The scheduling discipline - strict priority with round-robin within a level,
//! deadlock detection with a virtual-clock jump for timed waits, and frame counting
//! at each display flip - lives here and is shared verbatim. See the native
//! `threaded.rs` (the wasmtime engine) and the browser scheduler for the two
//! [`GuestEngine`] implementations.
//!
//! # Why cooperative and single-guest-at-a-time
//! Only one guest thread runs at any instant; a thread yields control only at a host
//! call (a blocking primitive, or a preemption slice). Because no two guest threads
//! ever touch memory truly concurrently, the shared memory needs no atomics for
//! correctness, and scheduling stays deterministic - the same inputs drive the same
//! interleaving. That invariant is the engine's job to uphold (natively a single OS
//! thread; in the browser a run baton), and the policy here relies on it: a
//! [`resume`](GuestThread::resume) call runs exactly one thread to its next switch
//! point with no other thread live.

use std::sync::{Arc, Mutex};

use crate::host::{ImportDispatch, Reentry};

/// Backstop on scheduler rounds in [`Scheduler::run`], so a runaway or live-locking
/// guest cannot spin forever. A round is one thread resume.
pub const MAX_ROUNDS: u64 = 100_000_000;

/// Why a thread's [`resume`](GuestThread::resume) returned without the thread
/// finishing: the switch point it suspended at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stop {
    /// A preemption slice: no host call blocked, the thread is still runnable and
    /// simply used up its quantum. Also how a voluntary yield that is NOT a frame
    /// boundary arrives (`SvcOutcome::Reschedule`) - both mean "still runnable, let
    /// someone else go", and both earn the spin cooldown.
    Quantum,
    /// The thread hit a blocking primitive and must be parked until woken.
    Blocked,
    /// The thread ended one DISPLAY FRAME (it queued a finished frame for scanout).
    /// The only stop that advances the frame count; see [`crate::SvcOutcome::Flip`]
    /// for why nothing else may.
    Flip,
}

/// How a thread's guest execution finished.
pub enum FiberEnd {
    /// The guest entry returned normally; the value is its r0 (the thread's exit
    /// code).
    Returned(u32),
    /// The thread called `sceKernelExitThread` (or equivalent): it ends, but the
    /// process continues. The value is its r0.
    ThreadExit(u32),
    /// The thread called `sceKernelExitProcess`: the whole run stops. The value is
    /// the process exit code (r0).
    ProcessHalt(u32),
    /// The guest trapped for a reason that was not a clean halt/exit.
    Error(String),
}

/// The result of one [`resume`](GuestThread::resume): the thread suspended at a
/// switch point, or it finished.
pub enum ThreadStep {
    Suspended(Stop),
    Finished(FiberEnd),
}

/// A scheduled thread's lifecycle state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ThreadState {
    /// On the run queue; the scheduler may resume it.
    Runnable,
    /// Parked at a blocking primitive; not resumed until a wake makes it runnable.
    Blocked,
    /// Done. The value is its exit code.
    Finished(u32),
}

/// The verdict of a [`Scheduler::run`] / [`Scheduler::run_frames`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunReport {
    /// Every thread finished cleanly (or the process exited). The value is the
    /// process exit code (the halting thread's r0, or the main thread's).
    Finished(u32),
    /// No thread is runnable but some remain blocked: a deadlock (e.g. a wait with
    /// no possible signaller). The value lists the still-blocked thread ids.
    Deadlock(Vec<i32>),
    /// A thread trapped. The run cannot continue.
    Error(String),
    /// The round budget was exhausted (a live-lock backstop).
    RoundLimit,
    /// The requested number of frame boundaries was reached; the run was stopped
    /// deliberately (used to bound a title whose render loop never returns). The
    /// value is the number of frames observed.
    FramesReached(u64),
}

/// The identity a scheduled thread must expose regardless of engine: its SceUID and
/// its priority. The scheduler needs only this to run the priority/round-robin
/// discipline; how the thread is actually resumed differs by engine (a sync
/// [`GuestThread::resume`] natively, an async loop in the browser), so that is kept
/// off this trait.
pub trait ThreadHandle {
    /// The thread's SceUID (its exit code becomes a joiner's result). The main thread
    /// is 0 by convention.
    fn thid(&self) -> i32;
    /// SceKernel priority (lower number = higher priority).
    fn priority(&self) -> i32;

    /// Drop everything the ENGINE holds for this thread, keeping only its identity.
    ///
    /// Called once the scheduler has recorded the thread as finished. Nothing resumes a
    /// finished thread - [`SchedCore::pick_next`] skips it - so the instance, its entry
    /// functions and its stack are dead weight from that moment on. Only `thid` and
    /// `priority` are still read, by the joins and the sync dumps.
    ///
    /// The default is a no-op, which is right for an engine whose per-thread state is
    /// cheap. It is NOT right for the browser, where a guest thread is a whole
    /// `WebAssembly.Instance` of the entire transpiled title: measured at about 7 MB
    /// each, so a title that spawns a few hundred short-lived threads over a few hundred
    /// frames grows the renderer by a gigabyte and is killed. The slot itself stays -
    /// indices are stable for the whole run and the exit code has to outlive the thread.
    fn release(&mut self) {}

    /// Total engine FUEL this thread has burned since it started, or `None` from an
    /// engine with no fuel accounting.
    ///
    /// Fuel is the only measure of guest work that is exactly proportional to executed
    /// wasm and identical on both engines, which is what makes it the thing to charge the
    /// game clock for. Host wall time would be neither - it varies with the machine, so it
    /// would break determinism.
    ///
    /// It must be CUMULATIVE rather than per-resume: the scheduler differences it, and a
    /// per-resume figure would have to be reset by whoever read it, which is exactly the
    /// bookkeeping that goes wrong when a resume ends on a path nobody remembered.
    fn fuel_used(&mut self) -> Option<u64> {
        None
    }
}

/// A guest thread the scheduler can resume *synchronously* to its next switch point.
/// Native wraps a wasmtime async fiber that a single `poll` drives to the next
/// `.await`. The browser cannot implement this (JSPI resumption is asynchronous - it
/// unwinds to the event loop), so the browser drives [`SchedCore`] from its own async
/// loop and does NOT implement this trait.
pub trait GuestThread: ThreadHandle {
    /// Run the thread until it suspends at a switch point or finishes. Any number of
    /// non-blocking host calls may be serviced within one resume; it returns only at
    /// a block, a frame flip, a preemption slice, or the thread's end.
    fn resume(&mut self) -> ThreadStep;
}

/// Stands up guest threads and reaches shared guest memory for the scheduler. The
/// one object that must cross the per-thread boundary (each thread having its own
/// register file) is the shared memory, which the engine owns.
pub trait GuestEngine {
    /// The concrete thread type this engine produces. It must expose its identity
    /// ([`ThreadHandle`]); a synchronous engine additionally implements
    /// [`GuestThread`] so it can be driven by [`Scheduler`].
    type Thread: ThreadHandle;
    /// Instantiate a new guest thread for `reentry` (entry address, args, own stack,
    /// thid, priority), ready to resume from its entry. `Err` if the entry was not
    /// translated (the caller records the thread as immediately finished so a join
    /// does not hang).
    fn spawn(&mut self, reentry: &Reentry) -> Result<Self::Thread, ()>;
    /// Write `bytes` into shared guest memory at guest address `addr`. Out-of-range
    /// writes are dropped. Used to deliver exit codes owed to a woken joiner's `stat`
    /// out-parameter while no thread is live.
    fn write_mem(&mut self, addr: u32, bytes: &[u8]);

    /// Fill `out` from shared guest memory at guest address `addr`. False if the range is
    /// not mapped.
    ///
    /// The symmetric partner of [`write_mem`](Self::write_mem), and what lets a recipe's
    /// `@watch` be sampled by the SHARED evaluator on either engine. It returns a bool
    /// rather than zero-filling because "this address is not mapped" and "this address
    /// holds zero" are different findings and only one of them means the recipe is wrong.
    fn read_mem(&self, addr: u32, out: &mut [u8]) -> bool;

    /// Called once per display frame boundary, with the new frame count. The default
    /// ignores it; an engine that supports frame-armed diagnostics
    /// (`VITASLOP_ARM_AT_FRAME`) uses it to arm them at the requested frame, which is
    /// what lets a first-hit trap fire deep inside a run instead of during boot.
    fn on_frame(&mut self, _frames: u64) {}

    /// Guest address of the HOST MIRROR block, when this build's inline imports read
    /// one (`vitaslop_transpiler::InlineOp::LoadMirror`). `None` when the transpiled
    /// module reserved no block, which is every build that inlines nothing.
    ///
    /// An engine that transpiles a module with a mirror block MUST report it here.
    /// Failing to is not a lost optimisation, it is a guest reading a word nobody ever
    /// writes - so [`SchedCore::new`] checks that the block is actually being filled
    /// and refuses to start otherwise.
    fn mirror_base(&self) -> Option<u32> {
        None
    }
}

/// One entry in the live thread table: the engine's thread plus the policy's state.
struct Slot<T> {
    thread: T,
    state: ThreadState,
    /// Anti-starvation cooldown. Set when this thread was preempted by exhausting a
    /// full quantum WITHOUT ever blocking - i.e. it is CPU-bound (a busy spin), not
    /// cooperatively yielding. A cooled thread steps aside so every other runnable
    /// thread gets a turn before it runs again (see [`SchedCore::pick_next`]). This
    /// models real multicore hardware, where a lower-priority worker on another core
    /// keeps making progress while a higher-priority thread spins - without it, a
    /// spin-wait on a lower-priority worker's result deadlocks the single baton.
    /// Only [`Stop::Quantum`] sets it, so purely cooperative threads (which yield via
    /// a host call long before a full quantum) are never cooled and their interleave
    /// is unchanged.
    cooled: bool,
    /// How many times this thread has been picked to run, and of those how many ended
    /// by burning a WHOLE quantum rather than blocking or flipping. Together they are
    /// this thread's share of the single baton - see [`SchedCore::cpu_share_report`].
    picks: u64,
    quanta: u64,
    /// Cumulative [`GuestThread::fuel_used`] as of this thread's last suspend, so the
    /// next one can be charged for the DIFFERENCE - the work this resume actually did.
    fuel_seen: u64,
}

impl<T> Slot<T> {
    fn new(thread: T, state: ThreadState) -> Slot<T> {
        Slot { thread, state, cooled: false, picks: 0, quanta: 0, fuel_seen: 0 }
    }
}

/// What to do when no thread is runnable (see [`SchedCore::handle_idle`]).
pub enum IdleStep {
    /// The run is over; report the verdict.
    Done(RunReport),
    /// A timed wait was advanced (virtual clock jumped, expiries woken); the loop
    /// should pick again.
    Continue,
}

/// The engine-agnostic scheduler *core*: the live thread table, the shared host, and
/// every scheduling decision that does not depend on HOW a thread is resumed. Both
/// the synchronous [`Scheduler`] (native) and the browser's asynchronous loop compose
/// it - each owns only the tiny resume step, and defers priority, frame counting,
/// spawn/wake draining, deadlock/timed-wait, and the verdict to these methods.
pub struct SchedCore<E: GuestEngine, H: ImportDispatch> {
    engine: E,
    host: Arc<Mutex<H>>,
    threads: Vec<Slot<E::Thread>>,
    /// Round-robin cursor: the index after the last thread resumed.
    cursor: usize,
    /// Frame boundaries (display flips) observed so far.
    frames: u64,
    /// How many threads were RUNNABLE at each quantum boundary, bucketed by count -
    /// `runnable_hist[n]` is the number of quanta at which `n` threads were ready.
    /// Index 0 is unused (the thread that just stopped is itself runnable).
    ///
    /// This is the measurement behind the core model: the clock divides a quantum's
    /// wall time by `min(runnable, GUEST_CORES)`, and whether that matters at all is
    /// entirely a question of this distribution. A run that is almost always at 1 has
    /// nothing to divide - which is a real answer, and not one to assume either way.
    /// See [`SchedCore::runnable_report`].
    runnable_hist: Vec<u64>,
    /// Fuel accounting totals - see [`SchedCore::fuel_report`].
    fuel_total: u64,
    fuel_samples: u64,
    fuel_max: u64,
}

impl<E: GuestEngine, H: ImportDispatch> SchedCore<E, H> {
    /// A core seeded with its `main` thread. `host` is shared with the engine (its
    /// clones ride into each thread), so pass the same `Arc` the engine was built with.
    pub fn new(engine: E, host: Arc<Mutex<H>>, main: E::Thread) -> Self {
        let mut core = SchedCore {
            engine,
            host,
            threads: vec![Slot::new(main, ThreadState::Runnable)],
            cursor: 0,
            frames: 0,
            runnable_hist: Vec::new(),
            fuel_total: 0,
            fuel_samples: 0,
            fuel_max: 0,
        };
        // If this build inlined any host-mirror read, the host must actually be filling
        // the block. Check it once, here, while the failure is still one line from its
        // cause: an unfilled mirror means every inlined read returns a word that never
        // changes, and for the clock that is a vblank spin that can never be satisfied -
        // a livelock thousands of frames away with nothing pointing back here.
        if core.engine.mirror_base().is_some() {
            let written = core.refresh_mirror();
            assert!(
                written > 0,
                "this build inlines host-mirror reads, but the host writes no mirror slots \
                 (ImportDispatch::refresh_mirror); the guest would read a word that never \
                 changes",
            );
        }
        core
    }

    /// Bring the host-mirror block in guest memory up to date, and report how many
    /// slots the host wrote. A no-op when this build reserved no block.
    ///
    /// Called before every resume: that is the whole contract that makes an inlined
    /// mirror read equal to the host call it replaced (see
    /// `vitaslop_transpiler::InlineOp::LoadMirror`).
    fn refresh_mirror(&mut self) -> usize {
        let Some(base) = self.engine.mirror_base() else {
            return 0;
        };
        // Split borrows: the engine writes memory, the host supplies the values.
        let engine = &mut self.engine;
        self.host.lock().unwrap().refresh_mirror(&mut |slot, value| {
            engine.write_mem(base.wrapping_add(slot * 4), &value.to_le_bytes());
        })
    }

    /// Borrow the engine (e.g. to read guest memory or the shared host after a run).
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// Who actually got the CPU: one line per thread, most-scheduled first, with its
    /// share of all resumes and how many of its turns ended by burning a WHOLE quantum
    /// (a thread that never blocks) rather than blocking or flipping.
    ///
    /// This exists because the console has SEVERAL CPU cores and this scheduler has ONE
    /// baton. On hardware a high-priority thread that busy-waits occupies one core while
    /// a low-priority background worker keeps running on another; here the busy-wait
    /// takes the whole machine, and a title whose loader runs at a low priority can be
    /// starved into taking tens of thousands of frames to do a few seconds of work. That
    /// failure looks exactly like "loading is slow" from the outside, and nothing else
    /// in the system distinguishes it from a title that is genuinely busy - so the share
    /// has to be measurable.
    ///
    /// `quanta/picks` is the tell: a thread whose turns nearly all end in `Quantum` is
    /// spinning, not working through host calls.
    pub fn cpu_share_report(&self) -> String {
        use std::fmt::Write;
        let total: u64 = self.threads.iter().map(|t| t.picks).sum();
        let mut rows: Vec<(i32, i32, u64, u64, ThreadState)> = self
            .threads
            .iter()
            .map(|t| (t.thread.thid(), t.thread.priority(), t.picks, t.quanta, t.state))
            .collect();
        rows.sort_by(|a, b| b.2.cmp(&a.2));
        let mut s = format!("--- scheduler CPU share: {total} resumes over {} threads ---\n", rows.len());
        for (thid, prio, picks, quanta, state) in rows {
            let pct = if total == 0 { 0.0 } else { picks as f64 * 100.0 / total as f64 };
            let _ = writeln!(
                s,
                "  thid={thid:#x} prio={prio:#x} {pct:6.2}%  picks={picks} whole-quanta={quanta} {state:?}"
            );
        }
        s
    }

    /// How many threads were ready to run at each quantum boundary, as a distribution.
    ///
    /// The companion to [`cpu_share_report`](Self::cpu_share_report), and the thing that
    /// says whether the one-baton scheduler is misrepresenting a MULTI-CORE device or
    /// faithfully representing a single busy thread. `picks` and `resumes` cannot answer
    /// it: a title can resume eighteen threads a frame and still never have two ready at
    /// the same instant, and in that case there is no parallelism for the clock to
    /// divide by. The mean is the number the clock actually divides by, capped at the
    /// device's core count.
    pub fn runnable_report(&self, cores: usize) -> String {
        use std::fmt::Write;
        let total: u64 = self.runnable_hist.iter().sum();
        if total == 0 {
            return "--- runnable at a quantum: no quanta observed ---\n".to_string();
        }
        let weighted: u64 =
            self.runnable_hist.iter().enumerate().map(|(n, c)| n as u64 * c).sum();
        let capped: u64 = self
            .runnable_hist
            .iter()
            .enumerate()
            .map(|(n, c)| (n.min(cores)) as u64 * c)
            .sum();
        let mut s = format!(
            "--- runnable threads at a quantum boundary: {total} quanta, mean {:.2}, \
             mean capped at {cores} cores {:.2} (the clock divisor) ---\n",
            weighted as f64 / total as f64,
            capped as f64 / total as f64,
        );
        for (n, count) in self.runnable_hist.iter().enumerate() {
            if *count == 0 {
                continue;
            }
            let _ = writeln!(
                s,
                "  {n:3} runnable: {count:10} quanta ({:5.2}%)",
                *count as f64 * 100.0 / total as f64
            );
        }
        s
    }

    /// The shared host handle (the browser loop locks it directly to dispatch).
    pub fn host(&self) -> &Arc<Mutex<H>> {
        &self.host
    }

    /// Frame boundaries observed so far.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// `(live, finished)` guest threads: how many the title is still running, and how
    /// many have ended and are held only as a slot with an exit code.
    ///
    /// Reported per frame in the browser because "threads created" and "threads still
    /// alive" are different numbers, and a run whose process grows steadily is asking
    /// which of the two is climbing. A title that spawns and joins short-lived workers
    /// looks identical to one that leaks them if only the creation count is visible -
    /// which is exactly how a per-thread engine allocation went unattributed.
    pub fn thread_census(&self) -> (usize, usize) {
        let finished =
            self.threads.iter().filter(|t| matches!(t.state, ThreadState::Finished(_))).count();
        (self.threads.len() - finished, finished)
    }

    /// Read shared guest memory at guest address `addr`; false if it is not mapped.
    /// The seam a recipe's `@watch` is sampled through on either engine.
    pub fn read_guest(&self, addr: u32, out: &mut [u8]) -> bool {
        self.engine.read_mem(addr, out)
    }

    /// Mutable access to thread `idx`, for an engine whose resume is driven externally
    /// (the browser async loop resumes `thread_mut(idx)` itself, then folds the result
    /// back with [`on_suspended`](Self::on_suspended)/[`on_finished`](Self::on_finished)).
    pub fn thread_mut(&mut self, idx: usize) -> &mut E::Thread {
        &mut self.threads[idx].thread
    }

    /// The next thread to run, advancing the round-robin cursor past it: the
    /// highest-priority runnable thread (lowest priority number), round-robin among
    /// threads sharing that priority. This is the real SceKernel discipline - a strict
    /// priority scheduler with round-robin within a level - and the ordering titles
    /// rely on (a higher-priority worker started by a lower-priority thread runs to its
    /// first block before the starter continues). `None` if nothing is runnable.
    pub fn pick_next(&mut self) -> Option<usize> {
        // Strict priority, but only among threads not on the spin cooldown: a CPU-bound
        // thread that just burned a whole quantum steps aside so peers get a turn. When
        // every runnable thread is cooled (all spinning), the cycle is complete - clear
        // the cooldowns and start a fresh round. This keeps strict priority for the
        // normal case (cooperative threads never cool) while guaranteeing every runnable
        // thread makes progress, the way separate cores do on real hardware.
        if !self.threads.iter().any(|t| t.state == ThreadState::Runnable && !t.cooled) {
            for t in self.threads.iter_mut() {
                t.cooled = false;
            }
        }
        let n = self.threads.len();
        let runnable = |t: &Slot<E::Thread>| t.state == ThreadState::Runnable && !t.cooled;
        let best = self.threads.iter().filter(|t| runnable(t)).map(|t| t.thread.priority()).min()?;
        let idx = (0..n)
            .map(|k| (self.cursor + k) % n)
            .find(|&i| runnable(&self.threads[i]) && self.threads[i].thread.priority() == best)?;
        self.cursor = idx + 1;
        self.threads[idx].picks += 1;
        // Guest code is about to run, so the host-mirror block has to be current. This
        // is the one place both schedulers (native and browser) pass through on their
        // way to a resume, which is why the refresh lives here rather than in either
        // loop - a resume path that skipped it would serve a stale clock.
        self.refresh_mirror();
        Some(idx)
    }

    /// A resumed thread (thread `idx`) suspended at `stop`. Updates its state and, for
    /// a frame flip, counts the frame and advances frame-keyed input; returns
    /// `Some(report)` if the frame budget `max_frames` was reached.
    pub fn on_suspended(&mut self, idx: usize, stop: Stop, max_frames: u64) -> Option<RunReport> {
        // Charge the game clock for the guest work this resume actually did, WHATEVER it
        // stopped for. This is the whole correction: `Stop::Quantum` means either "burned a
        // whole preemption quantum" or "yielded voluntarily after almost nothing" - the
        // engine sets it as the DEFAULT before polling - and billing both a full quantum
        // while billing `Stop::Blocked` nothing is two errors in opposite directions, each
        // workload-dependent. Measured before this: the game clock ran 1.08x on one title
        // and 4.34x on another, same build, same day. No flat per-quantum constant can fit
        // both, because a quantum is not a unit of work. Fuel is.
        self.charge_guest_work(idx);
        match stop {
            Stop::Blocked => {
                self.threads[idx].state = ThreadState::Blocked;
                None
            }
            Stop::Flip => {
                // A display frame ended. The thread BLOCKS: on hardware this call waits
                // for the scanout to latch the frame, and `VitaState::pace_flip` has
                // already parked it until the next vblank. Leaving it runnable here is
                // what let a title flip as fast as it could draw - 216 fps on a 60 Hz
                // panel - which stretched every frame-keyed timeline by that factor.
                //
                // The park may be zero (the guest is already late), in which case the
                // sleep waiter fires on the next scheduler pass and this costs one
                // reschedule - which is also what hardware does at a frame boundary.
                self.threads[idx].state = ThreadState::Blocked;
                // Count the frame and advance any frame-keyed input (a scripted TAS
                // recipe) in lockstep with the render loop.
                self.frames += 1;
                self.host.lock().unwrap().on_frame_boundary(self.frames);
                // Let the engine arm anything keyed to a frame (see
                // [`GuestEngine::on_frame`]); a no-op in an ordinary build.
                self.engine.on_frame(self.frames);
                (self.frames >= max_frames).then_some(RunReport::FramesReached(self.frames))
            }
            // Either a preemption slice (the thread used a whole quantum without
            // blocking, so it is CPU-bound) or a voluntary non-frame yield. Both mean
            // the thread keeps running but should step aside: put it on the spin
            // cooldown so peers run before it does again (anti-starvation; see
            // [`Slot::cooled`]).
            Stop::Quantum => {
                self.threads[idx].cooled = true;
                self.threads[idx].quanta += 1;
                None
            }
        }
    }

    /// Charge the host's work-tracking clocks for the fuel thread `idx` burned since its
    /// last suspend.
    ///
    /// # Why the clock must advance for guest CPU at all
    /// Without it the game clock advances only on a display flip or on the scheduler's
    /// nothing-is-runnable idle path, so a guest busy-wait ON THE CLOCK ITSELF - the
    /// `do { v = sceDisplayGetVcount(); } while (v == last);` vblank spin that is ordinary,
    /// correct guest code - can never be satisfied, and two such threads livelock the title.
    ///
    /// # Why FUEL and not a per-quantum constant
    /// Fuel is deterministic and exactly proportional to executed wasm on both engines, so
    /// the same guest work costs the same game time whichever engine ran it. A per-suspend
    /// constant instead measures how OFTEN a title suspends, which is a property of its
    /// host-call density and its blocking pattern, not of its workload.
    ///
    /// # Why the runnable count divides it
    /// The device has three CPUs for the game, so up to three threads would have retired
    /// this work AT ONCE and only one span of wall time would have passed. `cooled` is an
    /// anti-starvation nudge inside our one-baton rotation and says nothing about whether a
    /// thread could run on hardware, so it is not consulted; the thread that just stopped is
    /// still counted, since it was running.
    fn charge_guest_work(&mut self, idx: usize) {
        let Some(total) = self.threads[idx].thread.fuel_used() else { return };
        // Saturating, not wrapping: an engine that reports a cumulative counter which ever
        // goes backwards is one whose accounting is wrong, and a huge bogus charge would
        // jump the game clock by hours rather than showing up as a small drift.
        let burned = total.saturating_sub(self.threads[idx].fuel_seen);
        self.threads[idx].fuel_seen = total;
        if burned == 0 {
            return;
        }
        let runnable = self.threads.iter().filter(|t| t.state == ThreadState::Runnable).count();
        if self.runnable_hist.len() <= runnable {
            self.runnable_hist.resize(runnable + 1, 0);
        }
        self.runnable_hist[runnable] += 1;
        // The fuel accounting's own totals, which are what say whether a clock built on them
        // is measuring guest work or measuring a bug. A per-suspend burn above the preemption
        // interval is impossible (the engine preempts AT it), so seeing one means the reading
        // is wrong rather than the title being busy - and that distinction cannot be made
        // from the clock alone, because a wrong clock looks exactly like a slow title.
        self.fuel_total = self.fuel_total.saturating_add(burned);
        self.fuel_samples += 1;
        self.fuel_max = self.fuel_max.max(burned);
        self.host.lock().unwrap().on_guest_work(runnable, burned);
    }

    /// `(total fuel burned, samples, largest single burn)`.
    ///
    /// Reported unconditionally at the end of a headless run, for the same reason every
    /// approximation here reports itself: the largest single burn is the one number that can
    /// FALSIFY the clock's calibration, and it is worthless behind a flag nobody sets when the
    /// surprising timing is already in front of them.
    pub fn fuel_report(&self) -> (u64, u64, u64) {
        (self.fuel_total, self.fuel_samples, self.fuel_max)
    }

    /// A resumed thread (thread `idx`) finished with `end`; returns `Some(report)` if
    /// the whole run must stop (a process halt or a trap).
    pub fn on_finished(&mut self, idx: usize, end: FiberEnd) -> Option<RunReport> {
        let thid = self.threads[idx].thread.thid();
        // A thread ending is reported UNCONDITIONALLY, on both engines, in one place.
        // It used to be silent, and that is how "the browser's threads all finish, one
        // per frame" could only be seen as a live/finished COUNT in a per-frame telemetry
        // line - a count says a thread went, never which one or why. The two engines run
        // the same guest, so a divergence here is only legible if both say the same thing
        // in the same words.
        let (kind, code) = match &end {
            FiberEnd::Returned(c) => ("returned from its entry", *c),
            FiberEnd::ThreadExit(c) => ("sceKernelExitThread", *c),
            FiberEnd::ProcessHalt(c) => ("halted the process", *c),
            FiberEnd::Error(_) => ("TRAPPED", 0),
        };
        let finished = self
            .threads
            .iter()
            .filter(|t| matches!(t.state, ThreadState::Finished(_)))
            .count();
        tracing::info!(
            target: "vitaslop::thread",
            "thread {thid:#x} FINISHED at frame {}: {kind} (code {code:#x}) - {} of {} \
             threads finished",
            self.frames,
            finished + 1,
            self.threads.len(),
        );
        match end {
            FiberEnd::Returned(code) | FiberEnd::ThreadExit(code) => {
                self.threads[idx].state = ThreadState::Finished(code);
                // Hand the engine's per-thread state back the moment the thread is
                // recorded as finished. See [`ThreadHandle::release`]: on the browser
                // this is a whole module instance, and holding one per thread for the
                // rest of the run is a leak measured in gigabytes.
                self.threads[idx].thread.release();
                // Tell the host this thread ended, so any sibling waiting on it can be
                // woken (the wake is drained right after, by the caller).
                self.host.lock().unwrap().set_thread_exit(thid, code);
                None
            }
            FiberEnd::ProcessHalt(code) => {
                self.threads[idx].state = ThreadState::Finished(code);
                // The process is over: mark every other live thread finished too.
                for t in self.threads.iter_mut() {
                    if let ThreadState::Runnable | ThreadState::Blocked = t.state {
                        t.state = ThreadState::Finished(code);
                    }
                }
                // Every thread is finished now, so every engine allocation can go.
                for t in self.threads.iter_mut() {
                    t.thread.release();
                }
                Some(RunReport::Finished(code))
            }
            FiberEnd::Error(e) => Some(RunReport::Error(format!("thread {thid:#x}: {e}"))),
        }
    }

    /// Called when [`pick_next`](Self::pick_next) found nothing runnable. Either the
    /// run is finished/deadlocked ([`IdleStep::Done`]), or a timed wait was advanced
    /// and the loop should pick again ([`IdleStep::Continue`]). Before declaring a
    /// deadlock, honor any timed wait: jump the virtual clock to the earliest deadline
    /// and wake what expires - how a frame-pacing / timed condition wait makes progress
    /// without a real signaller. Only purely infinite waits with no timer deadlock.
    pub fn handle_idle(&mut self) -> IdleStep {
        let blocked: Vec<i32> = self
            .threads
            .iter()
            .filter(|t| t.state == ThreadState::Blocked)
            .map(|t| t.thread.thid())
            .collect();
        if blocked.is_empty() {
            return IdleStep::Done(RunReport::Finished(self.main_exit_code()));
        }
        let deadline = self.host.lock().unwrap().earliest_deadline();
        // An outstanding storage transfer comes FIRST, before any timed wait. Nothing is
        // runnable, so the modelled device is the only thing left with work to do, and
        // completing it is the next thing that happens - no game time passes while it
        // does. Jumping the clock to a pending timeout instead is wrong twice over: it
        // makes a thread waiting on a zero-length delay spin forever without the clock
        // moving (the livelock that kept the I/O model switched off), and, worse, when the
        // nearest timeout is far away it LEAPS the game clock over a load. A title that
        // paces its simulation off the wall clock but caps how far it will step per frame
        // then runs its clock fast and its world slow - which showed up as a car crawling
        // at 16 mph while its race timer counted five times too quickly.
        if self.host.lock().unwrap().release_earliest_io() {
            self.drain();
            return IdleStep::Continue;
        }
        match deadline {
            Some(t) => {
                // Diagnostic (VITASLOP_CLOCK_TRACE=<us>): report every idle jump at least
                // that large. A title derives its own timers from this clock, so a jump
                // bigger than a frame is game time passing that the rendered frames never
                // accounted for - the game's race clock runs ahead of its own simulation,
                // and the symptom is a rule (a race that always times out), not anything
                // that looks like a clock bug. The size and frequency of these jumps is the
                // only thing that says whether the leap is the cause or a bystander.
                if let Ok(min) = std::env::var("VITASLOP_CLOCK_TRACE") {
                    let min: u64 = min.parse().unwrap_or(16_667);
                    let now = self.host.lock().unwrap().clock_us();
                    if t.saturating_sub(now) >= min {
                        eprintln!(
                            "CLOCKJUMP +{}us (now {}us, {} blocked)",
                            t - now,
                            now,
                            blocked.len()
                        );
                    }
                }
                self.host.lock().unwrap().advance_time_to(t);
                self.drain();
                IdleStep::Continue
            }
            None => IdleStep::Done(RunReport::Deadlock(blocked)),
        }
    }

    /// Start any threads the last host call requested, and wake any it released. Call
    /// after every resume and after a clock advance.
    pub fn drain(&mut self) {
        let (spawns, wakes, stat_writes) = {
            let mut host = self.host.lock().unwrap();
            (host.take_spawns(), host.take_wakes(), host.take_stat_writes())
        };
        // Deliver any exit codes owed to a blocked `sceKernelWaitThreadEnd` joiner's
        // `stat` out-parameter. Done here (no thread runs during a drain) before the
        // woken joiner resumes.
        for (addr, value) in stat_writes {
            self.engine.write_mem(addr, &value.to_le_bytes());
        }
        for sp in spawns {
            let spawned = crate::perf::time(crate::perf::Phase::ThreadSpawn, || {
                self.engine.spawn(&sp)
            });
            match spawned {
                Ok(thread) => self.threads.push(Slot::new(thread, ThreadState::Runnable)),
                // A spawn whose entry was not translated: record it as finished with
                // code 0 so a later join does not hang.
                Err(()) => self.host.lock().unwrap().set_thread_exit(sp.thid, 0),
            }
        }
        for thid in wakes {
            if let Some(t) = self
                .threads
                .iter_mut()
                .find(|t| t.thread.thid() == thid && t.state == ThreadState::Blocked)
            {
                t.state = ThreadState::Runnable;
                // A freshly woken thread is immediately eligible: it did cooperative
                // work (it blocked), so it must not inherit a stale spin cooldown.
                t.cooled = false;
            }
        }
    }

    /// The process exit code: the main thread's finished code if it has one, else 0.
    pub fn main_exit_code(&self) -> u32 {
        self.threads
            .first()
            .and_then(|t| match t.state {
                ThreadState::Finished(c) => Some(c),
                _ => None,
            })
            .unwrap_or(0)
    }
}

impl<E, H> SchedCore<E, H>
where
    E: GuestEngine,
    E::Thread: GuestThread,
    H: ImportDispatch,
{
    /// Resume thread `idx` synchronously and fold its step back into the table,
    /// returning `Some(report)` if the run must stop. Only available when the engine's
    /// threads implement the synchronous [`GuestThread`] (native); the browser's async
    /// loop calls [`on_suspended`](Self::on_suspended)/[`on_finished`](Self::on_finished)
    /// itself after awaiting its own resume.
    fn resume_sync(&mut self, idx: usize, max_frames: u64) -> Option<RunReport> {
        match self.threads[idx].thread.resume() {
            ThreadStep::Finished(end) => self.on_finished(idx, end),
            ThreadStep::Suspended(stop) => self.on_suspended(idx, stop, max_frames),
        }
    }
}

/// A preemptive multi-thread guest run driven *synchronously* - the native front over
/// [`SchedCore`], for an engine whose threads implement [`GuestThread`]. The loop here
/// is deliberately tiny; all the discipline lives in `SchedCore` and is shared with
/// the browser's async loop.
pub struct Scheduler<E: GuestEngine, H: ImportDispatch> {
    core: SchedCore<E, H>,
    /// Thread resumes over the scheduler's whole life. One round is a pick plus a
    /// fiber switch plus a drain, so rounds-per-frame is the scheduler's own share of
    /// a frame - the part of "guest CPU" that is not the guest at all.
    rounds_total: u64,
}

impl<E, H> Scheduler<E, H>
where
    E: GuestEngine,
    E::Thread: GuestThread,
    H: ImportDispatch,
{
    /// A scheduler seeded with its `main` thread, ready to run.
    pub fn new(engine: E, host: Arc<Mutex<H>>, main: E::Thread) -> Self {
        Scheduler { core: SchedCore::new(engine, host, main), rounds_total: 0 }
    }

    /// Thread resumes so far. See [`rounds_total`](Self::rounds_total)'s field docs.
    pub fn rounds_total(&self) -> u64 {
        self.rounds_total
    }

    /// Borrow the engine (e.g. to read guest memory or the shared host after a run).
    pub fn engine(&self) -> &E {
        self.core.engine()
    }

    /// Display frame boundaries (flips) observed so far. A live front-end steps one
    /// frame at a time by calling `run_frames(frames() + 1, ..)` each redraw.
    pub fn frames(&self) -> u64 {
        self.core.frames()
    }

    /// Who actually got the CPU - see [`SchedCore::cpu_share_report`].
    pub fn cpu_share_report(&self) -> String {
        self.core.cpu_share_report()
    }

    /// How much of the device's parallelism the run actually used - see
    /// [`SchedCore::runnable_report`].
    pub fn runnable_report(&self, cores: usize) -> String {
        self.core.runnable_report(cores)
    }

    /// The fuel accounting's own totals - see [`SchedCore::fuel_report`].
    pub fn fuel_report(&self) -> (u64, u64, u64) {
        self.core.fuel_report()
    }

    /// Run cooperatively until the process halts, every thread finishes, or the run
    /// deadlocks / errors. Returns the verdict.
    pub fn run(&mut self) -> RunReport {
        self.run_frames(u64::MAX, MAX_ROUNDS)
    }

    /// Like [`run`](Self::run) but stop after `max_frames` frame boundaries (display
    /// flips) even if threads are still running - the way to bound a real title whose
    /// render loop never returns and capture a fixed number of frames. `max_rounds`
    /// caps the number of thread resumes so a busy-waiting guest cannot run unbounded.
    pub fn run_frames(&mut self, max_frames: u64, max_rounds: u64) -> RunReport {
        let mut rounds = 0u64;
        loop {
            if rounds >= max_rounds {
                self.rounds_total += rounds;
                return RunReport::RoundLimit;
            }
            rounds += 1;
            self.rounds_total += 1;

            // The scheduler's own work is timed; the RESUME between them is not,
            // because resuming runs the guest (see `Phase::SchedOverhead`).
            let pick = crate::perf::scope(crate::perf::Phase::SchedOverhead);
            let Some(idx) = self.core.pick_next() else {
                let step = self.core.handle_idle();
                drop(pick);
                match step {
                    IdleStep::Done(report) => return report,
                    IdleStep::Continue => continue,
                }
            };
            drop(pick);

            if let Some(report) = self.core.resume_sync(idx, max_frames) {
                return report;
            }
            // A host call in this resume may have asked to start threads or woken
            // parked ones; act on both before the next round.
            let drain = crate::perf::scope(crate::perf::Phase::SchedOverhead);
            self.core.drain();
            drop(drain);
        }
    }
}
