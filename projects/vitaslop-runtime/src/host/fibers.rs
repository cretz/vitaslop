//! SceFiber: the fibers a title has initialised, and which thread is running each chain.
//!
//! A fiber is cooperative: exactly one member of a fiber chain runs at a time, and a switch is
//! explicit. That is precisely what the preemptive scheduler already provides through
//! park/wake, so a fiber is backed by an ordinary guest thread that is runnable ONLY while it
//! holds the baton.
//!
//! [`FiberTable`] owns the fiber records and the per-runner state a chain needs, and makes the
//! fiber-level decisions - which call is allowed, what a dispatch has to do, which fiber is
//! abandoned. Carrying a decision out - creating, starting, waking or retiring the backing
//! thread, binding its storage, choosing its priority - is scheduler work and stays in
//! `VitaState` (`state_threads.rs`).

use std::collections::HashMap;

/// The stack pointer a fiber starts at: the top of its context buffer, 8-byte aligned
/// as AAPCS requires at a public call. A fiber with no context yields 0 here: it runs on
/// its runner's stack instead, which `fiber_dispatch` supplies.
pub(super) fn fiber_stack_top(context_addr: u32, context_size: u32) -> u32 {
    if context_addr == 0 {
        0
    } else {
        context_addr.wrapping_add(context_size) & !0xF
    }
}

/// One `SceFiber` the guest has initialised.
///
/// Nothing about the guest's own stack changes: `Reentry` already carries an explicit
/// `stack_top`, so the fiber's thread runs on the context buffer the guest supplied, exactly
/// as the hardware does.
///
/// Modelling a switched-away fiber by nesting a re-entry cannot work - its stack has
/// to survive with live frames on it until it is switched back to - and that survival
/// is exactly what a parked scheduler thread gives for free.
struct FiberRec {
    /// The guest `SceFiber*`. This IS the fiber's identity: every API call names it,
    /// and the guest may hold it anywhere, so nothing about a fiber is keyed on
    /// anything else.
    addr: u32,
    name: String,
    entry: u32,
    arg_on_initialize: u32,
    /// The guest-supplied context (the fiber's stack) as `(base, size)`. Both zero for a
    /// CONTEXT-LESS fiber, which runs on its runner's own stack and is restarted at its
    /// entry every time it is run (see `VitaState::fiber_restart_later`) - unless
    /// `_sceFiberAttachContextAndSwitch` gives it one.
    context: (u32, u32),
    /// The scheduler thread backing this fiber. Created at initialize, STARTED only
    /// on the first run/switch: a fiber that is never run must never execute.
    thid: i32,
    started: bool,
    /// Set by `sceFiberFinalize`; the record is kept so a later call naming a
    /// finalized fiber is refused rather than silently resurrecting it.
    finalized: bool,
    /// The THREAD that ran this fiber's chain - what `sceFiberReturnToThread`
    /// returns to. 0 when the fiber is not currently running.
    runner: i32,
    /// Where to write the value this fiber is handed when it is next resumed: the
    /// `argOnRun` out-pointer of the `sceFiberSwitch`/`ReturnToThread` it is parked
    /// in. 0 when it has none (or has not run yet, where the value arrives in r1).
    resume_out: u32,
}

/// What a dispatch - giving fiber `i` the baton - requires of the scheduler.
pub(super) enum Dispatch {
    /// The fiber is parked mid-function: wake `thid`, writing `arg_on_run` through `out`
    /// first when it is non-zero.
    Wake { thid: i32, out: u32 },
    /// The fiber has never run (or was abandoned): start `thid` at `entry(arg0, argOnRun)` on
    /// `stack_top`.
    Start {
        thid: i32,
        entry: u32,
        arg0: u32,
        stack_top: u32,
    },
}

/// Every fiber and the per-runner state of the chains. See the module docs.
#[derive(Default)]
pub(super) struct FiberTable {
    /// Every `SceFiber` the guest has initialised.
    fibers: Vec<FiberRec>,
    /// Per THREAD that called `sceFiberRun`, the `argOnRun` out-pointer to fill when
    /// the fiber chain it started returns to it. Keyed by thread id, not by fiber,
    /// because it belongs to the run call the thread is parked in.
    run_out: HashMap<i32, u32>,
    /// Per THREAD that called `sceFiberRun`, its stack pointer at that call: where a
    /// CONTEXT-LESS fiber of its chain runs (see [`FiberRec::context`]). Kept for the
    /// chain's whole life, because a context-less fiber switched to mid-chain starts at
    /// this same base, not on the stack of the fiber that switched to it.
    run_sp: HashMap<i32, u32>,
    /// Fiber threads retired while PARKED (a fiber with a context, finalized or
    /// re-initialised while suspended) and therefore ENDED by the runtime rather than by the
    /// guest - see `VitaState::retire_fiber_thread`. Counted for the stall dump: a title that
    /// finalizes suspended fibers does so per job, and this is the only place that rate is
    /// visible.
    terminated: u64,
}

impl FiberTable {
    /// The live (not finalized) fiber at `addr`.
    pub(super) fn index(&self, addr: u32) -> Option<usize> {
        self.fibers
            .iter()
            .position(|f| f.addr == addr && !f.finalized)
    }

    /// The fiber thread `thid` is executing, if it is executing one.
    pub(super) fn running_on(&self, thid: i32) -> Option<usize> {
        self.fibers
            .iter()
            .position(|f| f.thid == thid && f.started && !f.finalized)
    }

    pub(super) fn addr(&self, i: usize) -> u32 {
        self.fibers[i].addr
    }

    /// The thread a title's own code running on `thid` would call itself - see
    /// `VitaState::logical_thread`.
    pub(super) fn logical_thread(&self, thid: i32) -> i32 {
        match self
            .fibers
            .iter()
            .find(|f| f.thid == thid && f.started && !f.finalized)
        {
            Some(f) if f.runner != 0 => self.logical_thread(f.runner),
            _ => thid,
        }
    }

    /// Whether fiber `i` is the one running a chain right now (it holds the baton).
    pub(super) fn is_running(&self, i: usize) -> bool {
        self.fibers[i].runner != 0
    }

    /// Whether fiber `i` is running a chain AND has started - the state a re-initialise refuses.
    pub(super) fn is_live_and_running(&self, i: usize) -> bool {
        self.fibers[i].started && self.fibers[i].runner != 0
    }

    /// Remove fiber `i`'s record, returning its backing thread for the caller to retire.
    pub(super) fn remove(&mut self, i: usize) -> i32 {
        self.fibers.swap_remove(i).thid
    }

    /// Drop the finalized records at `addr` - a new initialise there supersedes them.
    pub(super) fn drop_finalized(&mut self, addr: u32) {
        self.fibers.retain(|f| !(f.addr == addr && f.finalized));
    }

    /// Record a freshly initialised fiber backed by the (unstarted) thread `thid`.
    pub(super) fn add(
        &mut self,
        addr: u32,
        name: String,
        entry: u32,
        arg_on_initialize: u32,
        context: (u32, u32),
        thid: i32,
    ) {
        self.fibers.push(FiberRec {
            addr,
            name,
            entry: entry & !1,
            arg_on_initialize,
            context,
            thid,
            started: false,
            finalized: false,
            runner: 0,
            resume_out: 0,
        });
    }

    /// `(name, entry)` of fiber `i`, for a fresh backing thread.
    pub(super) fn name_and_entry(&self, i: usize) -> (String, u32) {
        (self.fibers[i].name.clone(), self.fibers[i].entry)
    }

    /// Re-arm fiber `i` on the fresh, unstarted thread `thid` - it restarts at its entry.
    pub(super) fn restart_on(&mut self, i: usize, thid: i32) {
        let f = &mut self.fibers[i];
        f.thid = thid;
        f.started = false;
        f.runner = 0;
        f.resume_out = 0;
    }

    /// Record thread `runner`'s `sceFiberRun`: its stack pointer and its `argOnRun` out-pointer.
    pub(super) fn note_run(&mut self, runner: i32, caller_sp: u32, out: u32) {
        self.run_sp.insert(runner, caller_sp);
        self.run_out.insert(runner, out);
    }

    /// Give fiber `i` the baton for `runner`, and say what the scheduler must do.
    pub(super) fn dispatch(&mut self, i: usize, runner: i32) -> Dispatch {
        self.fibers[i].runner = runner;
        if self.fibers[i].started {
            return Dispatch::Wake {
                thid: self.fibers[i].thid,
                out: self.fibers[i].resume_out,
            };
        }
        self.fibers[i].started = true;
        let f = &self.fibers[i];
        // A context-less fiber runs on its runner's own stack, just below where the
        // runner stood at its `sceFiberRun` - the runner is parked in that call, so the
        // space below is free for as long as the chain lasts.
        let stack_top = if f.context.0 != 0 {
            fiber_stack_top(f.context.0, f.context.1)
        } else {
            self.run_sp.get(&runner).map_or(0, |sp| sp & !0xF)
        };
        Dispatch::Start {
            thid: f.thid,
            entry: f.entry,
            arg0: f.arg_on_initialize,
            stack_top,
        }
    }

    /// Fiber `from` hands the baton on (a switch or a return), parking with `out` as the
    /// pointer its next resume writes through. Returns the runner it held the baton for.
    pub(super) fn hand_off(&mut self, from: usize, out: u32) -> i32 {
        let runner = self.fibers[from].runner;
        self.fibers[from].runner = 0;
        self.fibers[from].resume_out = out;
        runner
    }

    /// Whether fiber `i` has no context - such a fiber is abandoned when it leaves, not parked.
    pub(super) fn context_less(&self, i: usize) -> bool {
        self.fibers[i].context.0 == 0
    }

    /// Whether fiber `i` has started - a context cannot be attached under live frames.
    pub(super) fn started(&self, i: usize) -> bool {
        self.fibers[i].started
    }

    pub(super) fn attach_context(&mut self, i: usize, context: (u32, u32)) {
        self.fibers[i].context = context;
    }

    /// The chain thread `runner` started has returned to it: its `argOnRun` out-pointer (0 for
    /// none), forgetting the run's state.
    pub(super) fn end_run(&mut self, runner: i32) -> u32 {
        self.run_sp.remove(&runner);
        self.run_out.remove(&runner).unwrap_or(0)
    }

    /// Mark fiber `i` finalized, returning its backing thread for the caller to retire.
    pub(super) fn finalize(&mut self, i: usize) -> i32 {
        self.fibers[i].finalized = true;
        self.fibers[i].thid
    }

    /// Count one parked fiber thread ended by the runtime.
    pub(super) fn note_terminated(&mut self) {
        self.terminated += 1;
    }

    pub(super) fn terminated(&self) -> u64 {
        self.terminated
    }

    /// `(thid, context, started, finalized, runner)` of every record at `addr`, finalized
    /// ones included, for the refusal report.
    pub(super) fn records_at(&self, addr: u32) -> Vec<(i32, (u32, u32), bool, bool, i32)> {
        self.fibers
            .iter()
            .filter(|f| f.addr == addr)
            .map(|f| (f.thid, f.context, f.started, f.finalized, f.runner))
            .collect()
    }

    /// `(entry, argOnInitialize, context base, context size, name)` of the live fiber at `addr`.
    pub(super) fn info(&self, addr: u32) -> Option<(u32, u32, u32, u32, &str)> {
        let f = &self.fibers[self.index(addr)?];
        Some((
            f.entry,
            f.arg_on_initialize,
            f.context.0,
            f.context.1,
            f.name.as_str(),
        ))
    }

    /// The backing thread of fiber `i`, for the tests.
    #[cfg(test)]
    pub(super) fn thid(&self, i: usize) -> i32 {
        self.fibers[i].thid
    }

    /// The backing thread of the first live fiber, for the tests that follow one through
    /// its restarts.
    #[cfg(test)]
    pub(super) fn first_live_thid(&self) -> Option<i32> {
        self.fibers.iter().find(|f| !f.finalized).map(|f| f.thid)
    }

    /// How many records the table holds, finalized ones included.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.fibers.len()
    }

    /// `(started, runner, thid)` of record `i`, for the tests.
    #[cfg(test)]
    pub(super) fn state(&self, i: usize) -> (bool, i32, i32) {
        let f = &self.fibers[i];
        (f.started, f.runner, f.thid)
    }
}
