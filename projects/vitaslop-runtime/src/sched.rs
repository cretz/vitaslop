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

    /// Total GUEST ARM INSTRUCTIONS this thread has retired since it started, from the
    /// emitted per-block counter (`abi::ARM_COUNT_GLOBAL`), or `None` from an engine that
    /// does not carry one.
    ///
    /// # Why the clock is billed in this and not in fuel
    /// Fuel counts executed WASM OPERATORS. That is the right unit for PREEMPTION - it
    /// bounds real work on both engines - and the wrong unit for the emulated CPU clock,
    /// because the number of operators a guest instruction becomes is a property of THIS
    /// TRANSPILER's codegen. Billing the clock in operators means every codegen improvement
    /// silently speeds the emulated console up: three changes in one session cut executed
    /// operators 28% for identical guest work, which would have made the emulated Vita
    /// 1.39x faster with nothing in a run to notice it by, and it took a hand-tuned
    /// compensation constant to undo. Guest instructions do not move when the codegen does.
    ///
    /// Cumulative, for the same reason [`Self::fuel_used`] is: the scheduler differences it.
    fn arm_retired(&mut self) -> Option<u64> {
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

/// Guest WORDS through an engine, for the scheduler's own use between resumes.
///
/// [`crate::host::GuestCtx`] is the ordinary way to touch guest memory, but it exists only
/// while a host call is in flight. The scheduler sometimes has to settle guest-resident
/// state with no call in flight at all - see [`crate::host::ImportDispatch::resolve_deferred`]
/// - and the engine's byte read/write is all that is available there.
///
/// An unmapped address reads zero and swallows its write, which matches what a host call
/// would do with the same address.
struct EngineWords<'a, E: GuestEngine>(&'a mut E);

impl<E: GuestEngine> crate::host::GuestWords for EngineWords<'_, E> {
    fn word(&self, addr: u32) -> u32 {
        let mut b = [0u8; 4];
        if self.0.read_mem(addr, &mut b) {
            u32::from_le_bytes(b)
        } else {
            0
        }
    }
    fn set_word(&mut self, addr: u32, value: u32) {
        self.0.write_mem(addr, &value.to_le_bytes());
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
    /// Cumulative [`ThreadHandle::arm_retired`] as of this thread's last suspend. The
    /// GAME CLOCK is charged for the difference; `fuel_seen` above now only feeds the fuel
    /// report and the preemption accounting.
    arm_seen: u64,
}

impl<T> Slot<T> {
    fn new(thread: T, state: ThreadState) -> Slot<T> {
        Slot { thread, state, cooled: false, picks: 0, quanta: 0, fuel_seen: 0, arm_seen: 0 }
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
    /// Wakes that arrived for a thread which was not blocked YET.
    ///
    /// # The lost wakeup this exists to stop
    /// A wake is applied by matching a thread id against the threads currently in
    /// [`ThreadState::Blocked`]. A wake for a thread in any other state used to be dropped
    /// on the floor - and there is a real ordering in which that happens every time: a host
    /// call parks the caller (`pace_flip` registers a sleep waiter with a deadline that has
    /// ALREADY passed, because the guest is late), the expiry runs and pushes the wake while
    /// the thread is still RUNNABLE, the wake is discarded, and only then does the scheduler
    /// mark the thread blocked. Nothing is left to wake it.
    ///
    /// MEASURED: a retail headless run ends at frame 1 with eight "blocked" threads, and
    /// the eighth is its `Graphics::RenderThread`, parked on nothing - no semaphore, no
    /// lwcond, no sleep waiter. It was written up in the notes as a frame-clock trap and
    /// worked around with a knob for several sessions.
    ///
    /// So a wake with nobody to deliver it to is REMEMBERED, and the next block by that
    /// thread consumes the token instead of parking. That is the same contract a condition
    /// variable's wakeup token has, and for the same reason.
    wake_tokens: std::collections::HashSet<i32>,
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
    /// Total GUEST ARM INSTRUCTIONS retired, summed over every charged suspension - see
    /// [`SchedCore::arm_report`]. This is the numerator of the only ABSOLUTE throughput
    /// number the project has: retired instructions divided by wall time is our effective
    /// guest MIPS, which a relative operator-count A/B can never produce.
    arm_total: u64,
    /// Suspensions the clock was NOT charged for, split by why: the engine reported no fuel
    /// counter at all, or it reported that the resume did no work.
    ///
    /// A suspension that bills nothing is invisible in every total - it does not move the
    /// clock, and it does not even count as a SAMPLE - so a thread suspending thousands of
    /// times for nothing looks exactly like a quiet frame from the outside. It is not: each
    /// one is a full JSPI suspend and resume, and on the browser that is the most expensive
    /// thing a scheduler round can do.
    fuel_unreported: u64,
    fuel_idle: u64,
}

impl<E: GuestEngine, H: ImportDispatch> SchedCore<E, H> {
    /// A core seeded with its `main` thread. `host` is shared with the engine (its
    /// clones ride into each thread), so pass the same `Arc` the engine was built with.
    pub fn new(engine: E, host: Arc<Mutex<H>>, main: E::Thread) -> Self {
        let mut core = SchedCore {
            wake_tokens: std::collections::HashSet::new(),
            engine,
            host,
            threads: vec![Slot::new(main, ThreadState::Runnable)],
            cursor: 0,
            frames: 0,
            runnable_hist: Vec::new(),
            fuel_total: 0,
            fuel_samples: 0,
            fuel_max: 0,
            arm_total: 0,
            fuel_unreported: 0,
            fuel_idle: 0,
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
        // >>> GATHER, THEN WRITE ONCE. This runs before EVERY resume, and a resume is the
        // scheduler's unit of work - MEASURED at 544-789 rounds a frame on one title.
        //
        // Writing the slots one at a time was one `write_mem` per slot, and on the browser
        // a `write_mem` is TWO JS calls (`subarray` to make a view, then `copy_from`) plus
        // a dirty stamp. Six slots by ~700 rounds is about 8,400 boundary crossings a frame
        // to move twenty-four bytes. The bytes were never the cost; the calls were - the
        // same thing that made `texture_bindings` 13.7% of a browser frame.
        //
        // The block is contiguous from slot 0 by construction (`vita::mirror::snapshot`
        // fills every slot), so the gathered bytes are one range. A host that ever writes a
        // SPARSE set falls back to per-slot writes rather than having the gaps clobbered
        // with zeros - that would be a silent wrong answer, and this is a cache the guest
        // reads as truth.
        const CAP_SLOTS: usize = 64;
        let mut buf = [0u8; CAP_SLOTS * 4];
        let mut seen = [false; CAP_SLOTS];
        let mut top = 0usize;
        let mut overflow: Vec<(u32, u32)> = Vec::new();
        let written = self.host.lock().unwrap().refresh_mirror(&mut |slot, value| {
            let s = slot as usize;
            if s < CAP_SLOTS {
                buf[s * 4..s * 4 + 4].copy_from_slice(&value.to_le_bytes());
                seen[s] = true;
                top = top.max(s + 1);
            } else {
                // A block this big means the "only values that cannot change while guest
                // code runs" rule was abandoned; carry it correctly and let the size show.
                overflow.push((slot, value));
            }
        });
        if seen[..top].iter().all(|&s| s) {
            self.engine.write_mem(base, &buf[..top * 4]);
        } else {
            for (s, _) in seen[..top].iter().enumerate().filter(|(_, set)| **set) {
                let at = s * 4;
                self.engine
                    .write_mem(base.wrapping_add(s as u32 * 4), &buf[at..at + 4]);
            }
        }
        for (slot, value) in overflow {
            self.engine.write_mem(base.wrapping_add(slot * 4), &value.to_le_bytes());
        }
        written
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

    /// Run `f` against the host WITH guest memory in hand, between resumes.
    ///
    /// The accessor for anything that has to reach guest-resident state from outside a host
    /// call - a lightweight mutex handed over, a diagnostic that signals a cond from the
    /// harness. Locking [`host`](Self::host) alone is not enough for those: the state they
    /// touch lives in the guest, and a host that writes it nowhere leaves a woken thread
    /// believing it holds a mutex the work area says is free.
    pub fn with_host_words<R>(&mut self, f: impl FnOnce(&mut H, &mut dyn crate::host::GuestWords) -> R) -> R {
        let mut words = EngineWords(&mut self.engine);
        let mut host = self.host.lock().unwrap();
        f(&mut host, &mut words)
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

    /// `VITASLOP_SCHED_CORES=<n>`: cap the baton to the top `n` runnable PRIORITIES, as the
    /// console's core count would. `None` (the default) keeps the current discipline.
    ///
    /// A DIAGNOSTIC, deliberately not a default - see the knob's entry in
    /// `vitaslop_platform::knobs::OVERRIDABLE` for the livelock it can cause. Read once.
    fn sched_cores() -> Option<usize> {
        use std::sync::OnceLock;
        static CELL: OnceLock<Option<usize>> = OnceLock::new();
        *CELL.get_or_init(|| {
            vitaslop_platform::knobs::var("VITASLOP_SCHED_CORES")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
        })
    }

    /// `VITASLOP_SCHED_RR=1`: round-robin every runnable thread, ignoring priority. A
    /// DIAGNOSTIC for the ordering question only - see the use site in [`Self::pick_next`].
    fn sched_rr() -> bool {
        use std::sync::OnceLock;
        static CELL: OnceLock<bool> = OnceLock::new();
        *CELL.get_or_init(|| {
            vitaslop_platform::knobs::var("VITASLOP_SCHED_RR").is_ok_and(|s| s.trim() == "1")
        })
    }

    /// `VITASLOP_SCHED_TRACE=<from>-<to>` (display frames, inclusive): print one line per
    /// scheduling transition inside that window - every pick, every block WITH WHAT IT IS
    /// WAITING ON, every wake, and every idle clock jump, each stamped with the frame and
    /// the virtual clock.
    ///
    /// # Why the scheduler needs its own timeline
    /// A guest-side block trace answers "in what order did these functions run". It cannot
    /// answer WHY that was the order, and for a defect that turns on two threads' relative
    /// progress inside one frame, the why is the whole question: a thread that runs late
    /// because it was descheduled and a thread that runs late because it was parked on a
    /// semaphore until another thread signalled it are the same picture from the guest's
    /// side and different bugs. Reconstructing this by A/B-ing scheduler policies costs a
    /// three-minute run per guess and confounds every guess with the timing it perturbs.
    fn sched_trace_window() -> &'static Option<(u64, u64)> {
        use std::sync::OnceLock;
        static CELL: OnceLock<Option<(u64, u64)>> = OnceLock::new();
        CELL.get_or_init(|| {
            let s = vitaslop_platform::knobs::var("VITASLOP_SCHED_TRACE").ok()?;
            let (from, to) = s.split_once('-').unwrap_or_else(|| {
                panic!("VITASLOP_SCHED_TRACE: {s:?} is not <from>-<to> (decimal frames)")
            });
            match (from.trim().parse::<u64>(), to.trim().parse::<u64>()) {
                (Ok(from), Ok(to)) if from <= to => Some((from, to)),
                _ => panic!("VITASLOP_SCHED_TRACE: {s:?} is not a valid frame window"),
            }
        })
    }

    /// Emit one [`sched_trace_window`] line, if the run is inside the window.
    fn sched_trace(&self, what: &str) {
        let Some((from, to)) = *Self::sched_trace_window() else { return };
        if self.frames < from || self.frames > to {
            return;
        }
        let now = self.host.lock().unwrap().clock_us();
        eprintln!("[sched] frame={} us={now} {what}", self.frames);
    }

    /// Is the scheduler timeline live right now? Lets a caller skip building a line's
    /// text (which locks the host) when nothing would print it.
    fn sched_trace_on(&self) -> bool {
        match *Self::sched_trace_window() {
            Some((from, to)) => self.frames >= from && self.frames <= to,
            None => false,
        }
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
        let n = self.threads.len();
        // DIAGNOSTIC (`VITASLOP_SCHED_CORES=<n>`): the cooldown below eventually admits EVERY
        // runnable thread regardless of priority, so a low-priority thread can hold the baton in
        // a quantum the hardware would never have given it - measured at 13.2% of quanta with
        // more than three runnable threads on one title. This cap answers "does a bug depend on
        // that". Unset, it costs nothing and changes nothing.
        //
        // **The cap is on THREADS, not on priority LEVELS**, and getting that wrong wastes a
        // whole experiment: a three-core console runs three THREADS at once, while capping
        // distinct priorities is no restriction at all when several threads share one. This
        // title has FOUR threads at `0xa0`, and a priority-level cap reproduced the uncapped
        // schedule to the digit - 483,268 resumes and identical per-thread picks.
        //
        // Ties are broken by the round-robin cursor, so which of several equal-priority threads
        // is the one left out ROTATES, rather than one being starved for the whole run.
        let admitted: Option<Vec<usize>> = Self::sched_cores().map(|cores| {
            let mut order: Vec<usize> = (0..n)
                .map(|k| (self.cursor + k) % n)
                .filter(|&i| self.threads[i].state == ThreadState::Runnable)
                .collect();
            order.sort_by_key(|&i| self.threads[i].thread.priority());
            order.truncate(cores);
            order
        });
        // The cooldown clear has to look at the ADMITTED set, not at every runnable thread: a
        // capped-out thread that is uncooled would otherwise hold the clear off for ever, and
        // every admitted thread being cooled would then leave nothing pickable at all.
        let admitted_uncooled = (0..n).any(|i| {
            self.threads[i].state == ThreadState::Runnable
                && !self.threads[i].cooled
                && admitted.as_ref().is_none_or(|a| a.contains(&i))
        });
        if !admitted_uncooled {
            for t in self.threads.iter_mut() {
                t.cooled = false;
            }
        }
        let runnable = |i: usize, t: &Slot<E::Thread>| {
            t.state == ThreadState::Runnable
                && !t.cooled
                && admitted.as_ref().is_none_or(|a| a.contains(&i))
        };
        // EXPERIMENT (`VITASLOP_SCHED_RR=1`): ignore priority entirely and round-robin every
        // runnable thread. Not a model of anything - it exists to answer whether a defect
        // depends on strict priority placing a whole frame of the top thread's work ahead of
        // the first instruction of a lower-priority thread woken by the SAME event.
        let best = if Self::sched_rr() {
            (0..n).filter(|&i| runnable(i, &self.threads[i])).map(|i| self.threads[i].thread.priority()).min()?;
            i32::MIN
        } else {
            (0..n)
                .filter(|&i| runnable(i, &self.threads[i]))
                .map(|i| self.threads[i].thread.priority())
                .min()?
        };
        let idx = (0..n).map(|k| (self.cursor + k) % n).find(|&i| {
            runnable(i, &self.threads[i])
                && (best == i32::MIN || self.threads[i].thread.priority() == best)
        })?;
        self.cursor = idx + 1;
        self.threads[idx].picks += 1;
        if self.sched_trace_on() {
            let runnable_now: Vec<i32> = (0..n)
                .filter(|&i| self.threads[i].state == ThreadState::Runnable)
                .map(|i| self.threads[i].thread.thid())
                .collect();
            self.sched_trace(&format!(
                "PICK t{:#x} prio={:#x} (runnable {runnable_now:x?})",
                self.threads[idx].thread.thid(),
                self.threads[idx].thread.priority()
            ));
        }
        // The host's idea of the current thread is now a PROPERTY OF THE PICK, not of the
        // next dispatch. It is mirrored into guest memory below, for the inlined
        // lightweight-mutex take, and a resumed thread must read its own id from the first
        // instruction - not the previous thread's until it happens to call the host.
        let thid = self.threads[idx].thread.thid();
        self.host.lock().unwrap().set_current_thread(thid);
        // Guest code is about to run, so the host-mirror block has to be current. This
        // is the one place both schedulers (native and browser) pass through on their
        // way to a resume, which is why the refresh lives here rather than in either
        // loop - a resume path that skipped it would serve a stale clock.
        {
            let _m = crate::perf::scope(crate::perf::Phase::SchedMirror);
            self.refresh_mirror();
        }
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
                self.block_or_consume_token(idx);
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
                self.block_or_consume_token(idx);
                // Count the frame and advance any frame-keyed input (a scripted TAS
                // recipe) in lockstep with the render loop.
                self.frames += 1;
                {
                    let _f = crate::perf::scope(crate::perf::Phase::FrameBoundary);
                    self.host.lock().unwrap().on_frame_boundary(self.frames);
                }
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
        let Some(total) = self.threads[idx].thread.fuel_used() else {
            self.fuel_unreported += 1;
            return;
        };
        // Saturating, not wrapping: an engine that reports a cumulative counter which ever
        // goes backwards is one whose accounting is wrong, and a huge bogus charge would
        // jump the game clock by hours rather than showing up as a small drift.
        let burned = total.saturating_sub(self.threads[idx].fuel_seen);
        self.threads[idx].fuel_seen = total;
        // The GUEST INSTRUCTIONS this resume retired, which is what the emulated CPU clock
        // is billed in. Differenced exactly as fuel is. An engine with no such counter
        // falls back to fuel below, which is the pre-2026-08-16 behaviour.
        let retired = match self.threads[idx].thread.arm_retired() {
            Some(t) => {
                let d = t.saturating_sub(self.threads[idx].arm_seen);
                self.threads[idx].arm_seen = t;
                Some(d)
            }
            None => None,
        };
        if burned == 0 && retired.unwrap_or(0) == 0 {
            self.fuel_idle += 1;
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
        self.arm_total = self.arm_total.saturating_add(retired.unwrap_or(0));
        self.host.lock().unwrap().on_guest_work(runnable, burned, retired);
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

    /// Total GUEST ARM INSTRUCTIONS retired so far, or 0 from an engine with no such
    /// counter. Cumulative, so a caller measuring a window differences it.
    pub fn arm_report(&self) -> u64 {
        self.arm_total
    }

    /// `(suspensions the engine reported no fuel for, suspensions that did no work)` - the two
    /// ways a scheduler round can cost a full suspend and bill the clock nothing. See the
    /// fields for why a silent one is worse than a slow one.
    pub fn unbilled_report(&self) -> (u64, u64) {
        (self.fuel_unreported, self.fuel_idle)
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
        // >>> DELIVER WHAT IS ALREADY OWED BEFORE CONCLUDING NOTHING CAN RUN.
        //
        // A wake is produced by the host (a timer expiring, a signal, an I/O completion)
        // and APPLIED by `drain`. Those are separate steps, and a wake produced after the
        // last drain sits in the host's pending list until the next one. Reaching this
        // function without draining therefore asks "is anything runnable?" while holding
        // the very wake that would make something runnable - and answering "no" to that is
        // a deadlock report for a run that was fine.
        //
        // MEASURED on a retail title, which stops at frame 1 and has been written up as a
        // frame-clock trap for several sessions. The order is exactly:
        //   PARK thid=0x11a us=714      (pace_flip parks the render thread)
        //   PARK-EXPIRE thid=0x11a      (the deadline passes; a wake is pushed)
        //   BLOCK thid=0x11a            (only NOW is the thread marked blocked)
        //   ...deadlock, with that wake still undelivered.
        // The thread is parked on nothing, no timer names it, and the run ends one frame
        // into a title that was about to render its title screen.
        self.drain();
        if self.threads.iter().any(|t| t.state == ThreadState::Runnable) {
            return IdleStep::Continue;
        }
        let blocked: Vec<i32> = self
            .threads
            .iter()
            .filter(|t| t.state == ThreadState::Blocked)
            .map(|t| t.thread.thid())
            .collect();
        if blocked.is_empty() {
            return IdleStep::Done(RunReport::Finished(self.main_exit_code()));
        }
        let (deadline, io_remaining, now) = {
            let h = self.host.lock().unwrap();
            (h.earliest_deadline(), h.earliest_io_remaining_us(), h.clock_us())
        };
        // >>> AN OUTSTANDING TRANSFER AND A PENDING TIMED WAIT ARE HONOURED IN TIME ORDER.
        //
        // Both are things that complete on their own while no guest code runs, and both
        // are counted in microseconds at the same rate - one display frame of game time
        // buys one display frame of storage time - so the next thing that happens is
        // simply whichever of the two is nearer.
        //
        // The rule used to be "the transfer comes FIRST, before any timed wait", on the
        // reasoning that nothing the guest can observe is lost because no guest code can
        // run until something completes. What that misses is the DISPLAY: the render
        // thread's vblank wait is a timed wait, so a transfer always beat the next flip
        // however large it was modelled to be, and the storage clock only advances on
        // flips and quanta - so its size bought nothing. A 250x bandwidth sweep on a
        // retail racer's course load moved the modelled cost of the read and left the
        // frame it landed on unchanged to the digit, which is the measurement that says
        // the model was not reaching the guest at all.
        //
        // The two hazards the old ordering was written against both survive this: a
        // zero-length poll cannot livelock the storage clock, because a spinning thread is
        // charged through `charge_io_quantum` and an idle jump now through
        // `charge_io_idle` - the clock the original livelock had no sources for at all;
        // and the game clock can no longer LEAP over a load, because a far-away timeout
        // does not win against a nearer transfer.
        if storage_completes_first(io_remaining, deadline, now)
            && self.host.lock().unwrap().release_earliest_io()
        {
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
                if self.sched_trace_on() {
                    let now = self.host.lock().unwrap().clock_us();
                    self.sched_trace(&format!("IDLE-JUMP to {t} (+{}us)", t.saturating_sub(now)));
                }
                {
                    let mut h = self.host.lock().unwrap();
                    // The device was reading across this idle interval too. Without this
                    // the storage clock advances only on flips and quanta, so a title
                    // waiting on a load - which is idle by definition - never pays the
                    // transfer down and the model's own units stop meaning anything.
                    h.charge_io_idle(t.saturating_sub(now));
                    h.advance_time_to(t);
                }
                self.drain();
                IdleStep::Continue
            }
            None => {
                // >>> A DEADLOCK IS THE ONE MOMENT THE WAIT STATE IS THE WHOLE ANSWER, and
                // it used to be the one moment nothing printed it. `Deadlock([..])` names the
                // blocked THREADS and not what any of them is waiting ON, so the report is a
                // list of numbers - and a run that stops this way looks, from the outside,
                // exactly like a clock pathology. MEASURED cost: a retail headless run
                // has stopped at frame 1 for several sessions, was written up in the notes as a
                // frame-clock trap, and was worked around with `VITASLOP_FRAME_TOPUP=0` rather
                // than diagnosed. It is a deadlock of eight threads.
                //
                // `debug_sync_dump` already builds exactly the right text and was only ever
                // reachable from a debugger command and a browser stall path.
                // Split the blocked threads by whether the host is actually waiting on
                // them. A thread with NO wait record cannot be woken by anything - that is
                // a lost wakeup in this emulator, not a guest deadlock, and calling it a
                // deadlock has sent several sessions looking at the guest's locking.
                let (dump, orphans) = {
                    let h = self.host.lock().unwrap();
                    let orphans: Vec<i32> =
                        blocked.iter().copied().filter(|&t| !h.thread_has_wait_record(t)).collect();
                    (h.sync_dump(), orphans)
                };
                if orphans.is_empty() {
                    eprintln!(
                        "DEADLOCK: {} thread(s) blocked with no timeout and no pending I/O,                          so nothing can wake them: {blocked:?}
{dump}",
                        blocked.len()
                    );
                } else {
                    eprintln!(
                        "LOST WAKEUP (reported as a deadlock): {} of {} blocked thread(s) have                          NO wait record in the host - nothing is waiting on them, so no signal,                          timeout or I/O completion can ever name them. This is an emulator bug,                          not a guest deadlock. Orphaned: {orphans:x?}; all blocked:                          {blocked:?}
{dump}",
                        orphans.len(),
                        blocked.len()
                    );
                }
                IdleStep::Done(RunReport::Deadlock(blocked))
            }
        }
    }

    /// Start any threads the last host call requested, and wake any it released. Call
    /// after every resume and after a clock advance.
    pub fn drain(&mut self) {
        // FIRST, and before the wakes are taken: settle anything the host decided where it
        // had no guest memory to decide it with (a lightweight mutex owed to a cond wait
        // that timed out - see `ImportDispatch::resolve_deferred`). It can push wakes of
        // its own, so it has to run ahead of the take below or they would sit until the
        // next drain, one whole scheduling round late.
        self.with_host_words(|host, words| host.resolve_deferred(words));
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
                Ok(thread) => {
                    self.sched_trace(&format!(
                        "SPAWN t{:#x} prio={:#x} entry={:#010x}",
                        thread.thid(),
                        thread.priority(),
                        sp.entry
                    ));
                    self.threads.push(Slot::new(thread, ThreadState::Runnable))
                }
                // A spawn whose entry was not translated: record it as finished with
                // code 0 so a later join does not hang.
                Err(()) => self.host.lock().unwrap().set_thread_exit(sp.thid, 0),
            }
        }
        for thid in wakes {
            match self
                .threads
                .iter_mut()
                .find(|t| t.thread.thid() == thid && t.state == ThreadState::Blocked)
            {
                Some(t) => {
                    t.state = ThreadState::Runnable;
                    // A freshly woken thread is immediately eligible: it did cooperative
                    // work (it blocked), so it must not inherit a stale spin cooldown.
                    t.cooled = false;
                    let prio = t.thread.priority();
                    self.sched_trace(&format!("WAKE  t{thid:#x} prio={prio:#x}"));
                }
                // Not blocked yet - the wake raced ahead of the block. Keep it; the block
                // that follows will consume it instead of parking. See `wake_tokens`.
                None => {
                    if self.threads.iter().any(|t| {
                        t.thread.thid() == thid && !matches!(t.state, ThreadState::Finished(_))
                    }) {
                        self.wake_tokens.insert(thid);
                    }
                }
            }
        }
    }

    /// Park thread `idx`, UNLESS a wake for it already arrived while it was still
    /// runnable - in which case consume that token and leave it eligible.
    ///
    /// Blocking on a wait that has already been satisfied is the lost wakeup this whole
    /// mechanism exists to prevent; see [`SchedCore::wake_tokens`] for the ordering that
    /// produces it and the run it stopped dead.
    fn block_or_consume_token(&mut self, idx: usize) {
        let thid = self.threads[idx].thread.thid();
        if self.wake_tokens.remove(&thid) {
            self.threads[idx].state = ThreadState::Runnable;
            self.threads[idx].cooled = false;
            self.sched_trace(&format!("BLOCK t{thid:#x} SKIPPED (wake token already owed)"));
            return;
        }
        self.threads[idx].state = ThreadState::Blocked;
        if self.sched_trace_on() {
            let why = self.host.lock().unwrap().thread_wait_reason(thid);
            self.sched_trace(&format!("BLOCK t{thid:#x} on {why}"));
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

    /// The scheduling core, for the few things that need both the host and the engine at
    /// once (see [`SchedCore::with_host_words`]).
    pub fn core_mut(&mut self) -> &mut SchedCore<E, H> {
        &mut self.core
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

    /// Cumulative retired guest ARM instructions - see [`SchedCore::arm_report`].
    pub fn arm_report(&self) -> u64 {
        self.core.arm_report()
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

/// Whether the modelled storage device completes its earliest outstanding transfer
/// BEFORE the next timed wait expires, given the game clock reads `now_us`.
///
/// Both quantities are microseconds and both advance at the same rate through an idle
/// interval, so this is a plain comparison of what happens next. It is a free function
/// so the rule can be tested without a scheduler: it is the whole of the ordering that a
/// retail racer's course load depends on, and it used to be the constant `true`.
///
/// A deadline that has already passed expired before any transfer can complete, so it
/// goes first. That cannot starve the device: a woken poller burning its quantum charges
/// the storage clock, as does the idle jump itself.
fn storage_completes_first(
    io_remaining_us: Option<u64>,
    next_deadline_us: Option<u64>,
    now_us: u64,
) -> bool {
    match (io_remaining_us, next_deadline_us) {
        (None, _) => false,
        (Some(_), None) => true,
        (Some(io), Some(t)) => io <= t.saturating_sub(now_us),
    }
}

#[cfg(test)]
mod idle_order_tests {
    use super::storage_completes_first;

    const FRAME_US: u64 = 1_000_000 / 60;

    #[test]
    fn nothing_in_flight_never_takes_the_storage_branch() {
        assert!(!storage_completes_first(None, Some(1_000), 0));
        assert!(!storage_completes_first(None, None, 0));
    }

    #[test]
    fn a_transfer_with_no_timed_wait_against_it_completes() {
        assert!(storage_completes_first(Some(40_000), None, 0));
    }

    #[test]
    fn the_vblank_beats_a_transfer_that_is_worth_more_than_a_frame() {
        // The defect this exists to prevent: a 2 MB read modelled at ~38 ms used to
        // complete ahead of a vblank 3 ms away, so it cost the guest no display frames
        // at all and its modelled size meant nothing.
        assert!(!storage_completes_first(Some(38_286), Some(3_000), 0));
        // One frame of it paid down still leaves more than a frame owed, so the next
        // vblank goes first again; after two, what is left fits inside a frame and the
        // transfer completes. That is 2 MB costing three display frames, which is the
        // whole point of modelling it.
        assert!(!storage_completes_first(Some(38_286 - FRAME_US), Some(FRAME_US), 0));
        assert!(storage_completes_first(Some(38_286 - 2 * FRAME_US), Some(FRAME_US), 0));
    }

    #[test]
    fn a_short_transfer_beats_a_distant_timeout() {
        // The other half of the rule, and the reason the game clock cannot LEAP a load:
        // a timeout a second away does not get to run before a 200 us read completes.
        assert!(storage_completes_first(Some(200), Some(1_000_000), 0));
    }

    #[test]
    fn an_already_expired_deadline_runs_before_any_transfer() {
        // A zero-length poll re-armed every iteration names a deadline behind the clock.
        // It expired before the transfer will complete, so it goes first - and that
        // cannot starve the device the way it once could, because the woken thread
        // burning its quantum charges the storage clock (`charge_io_quantum`) and an idle
        // jump charges it too (`charge_io_idle`). The livelock this ordering was
        // originally written against was a storage clock with no such sources at all.
        assert!(!storage_completes_first(Some(5_000), Some(100), 200));
    }

    #[test]
    fn the_comparison_is_against_the_clock_now_not_the_raw_deadline() {
        // 5 ms of transfer left against a deadline 1 ms away: the deadline is at
        // 101_000 on a clock reading 100_000, so the transfer must NOT win.
        assert!(!storage_completes_first(Some(5_000), Some(101_000), 100_000));
    }
}
