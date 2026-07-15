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
    /// simply used up its quantum.
    Quantum,
    /// The thread hit a blocking primitive and must be parked until woken.
    Blocked,
    /// The thread reached a frame boundary (display flip).
    Yielded,
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
}

/// One entry in the live thread table: the engine's thread plus the policy's state.
struct Slot<T> {
    thread: T,
    state: ThreadState,
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
}

impl<E: GuestEngine, H: ImportDispatch> SchedCore<E, H> {
    /// A core seeded with its `main` thread. `host` is shared with the engine (its
    /// clones ride into each thread), so pass the same `Arc` the engine was built with.
    pub fn new(engine: E, host: Arc<Mutex<H>>, main: E::Thread) -> Self {
        SchedCore {
            engine,
            host,
            threads: vec![Slot { thread: main, state: ThreadState::Runnable }],
            cursor: 0,
            frames: 0,
        }
    }

    /// Borrow the engine (e.g. to read guest memory or the shared host after a run).
    pub fn engine(&self) -> &E {
        &self.engine
    }

    /// The shared host handle (the browser loop locks it directly to dispatch).
    pub fn host(&self) -> &Arc<Mutex<H>> {
        &self.host
    }

    /// Frame boundaries observed so far.
    pub fn frames(&self) -> u64 {
        self.frames
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
        let n = self.threads.len();
        let best = self
            .threads
            .iter()
            .filter(|t| t.state == ThreadState::Runnable)
            .map(|t| t.thread.priority())
            .min()?;
        let idx = (0..n).map(|k| (self.cursor + k) % n).find(|&i| {
            self.threads[i].state == ThreadState::Runnable && self.threads[i].thread.priority() == best
        })?;
        self.cursor = idx + 1;
        Some(idx)
    }

    /// A resumed thread (thread `idx`) suspended at `stop`. Updates its state and, for
    /// a frame flip, counts the frame and advances frame-keyed input; returns
    /// `Some(report)` if the frame budget `max_frames` was reached.
    pub fn on_suspended(&mut self, idx: usize, stop: Stop, max_frames: u64) -> Option<RunReport> {
        match stop {
            Stop::Blocked => {
                self.threads[idx].state = ThreadState::Blocked;
                None
            }
            Stop::Yielded => {
                // A frame boundary (display flip): the thread stays runnable, but count
                // it toward the frame budget and advance any frame-keyed input (a
                // scripted TAS recipe) in lockstep with the render loop.
                self.frames += 1;
                self.host.lock().unwrap().on_frame_boundary(self.frames);
                (self.frames >= max_frames).then_some(RunReport::FramesReached(self.frames))
            }
            // A preemption slice: still runnable, not a frame.
            Stop::Quantum => None,
        }
    }

    /// A resumed thread (thread `idx`) finished with `end`; returns `Some(report)` if
    /// the whole run must stop (a process halt or a trap).
    pub fn on_finished(&mut self, idx: usize, end: FiberEnd) -> Option<RunReport> {
        let thid = self.threads[idx].thread.thid();
        match end {
            FiberEnd::Returned(code) | FiberEnd::ThreadExit(code) => {
                self.threads[idx].state = ThreadState::Finished(code);
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
        match deadline {
            Some(t) => {
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
            match self.engine.spawn(&sp) {
                Ok(thread) => self.threads.push(Slot { thread, state: ThreadState::Runnable }),
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
}

impl<E, H> Scheduler<E, H>
where
    E: GuestEngine,
    E::Thread: GuestThread,
    H: ImportDispatch,
{
    /// A scheduler seeded with its `main` thread, ready to run.
    pub fn new(engine: E, host: Arc<Mutex<H>>, main: E::Thread) -> Self {
        Scheduler { core: SchedCore::new(engine, host, main) }
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
                return RunReport::RoundLimit;
            }
            rounds += 1;

            let Some(idx) = self.core.pick_next() else {
                match self.core.handle_idle() {
                    IdleStep::Done(report) => return report,
                    IdleStep::Continue => continue,
                }
            };

            if let Some(report) = self.core.resume_sync(idx, max_frames) {
                return report;
            }
            // A host call in this resume may have asked to start threads or woken
            // parked ones; act on both before the next round.
            self.core.drain();
        }
    }
}
