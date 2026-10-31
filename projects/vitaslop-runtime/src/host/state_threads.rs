use super::*;

impl VitaState {
    /// Enable preemptive multithreading: blocking primitives now actually park the
    /// calling thread instead of succeeding uncontended. Set once, before the run,
    /// by the [`ThreadedScheduler`](vitaslop_native::ThreadedScheduler); the single-
    /// worker hosts leave it off and keep the run-to-completion behavior.
    /// (Scheduler: `vitaslop_native::ThreadedScheduler`.)
    pub fn set_preemptive(&mut self, on: bool) {
        self.preemptive = on;
    }

    /// True when running under the preemptive scheduler.
    pub fn is_preemptive(&self) -> bool {
        self.preemptive
    }

    /// The id of the thread currently executing a host call (preemptive mode). Set
    /// by the scheduler before each dispatch via `ImportDispatch::set_current_thread`.
    pub fn current_thread(&self) -> i32 {
        self.current
    }

    /// Record which thread is running (scheduler hook).
    pub fn set_current(&mut self, thid: i32) {
        self.current = thid;
    }
    /// Take the threads a host call asked to start (scheduler hook).
    pub fn take_spawns(&mut self) -> Vec<Reentry> {
        std::mem::take(&mut self.pending_spawns)
    }

    /// Make a parked thread runnable from OUTSIDE a host call - the frontend's async
    /// completion of a scene (`complete_scene_async`) releases the thread that ended it.
    pub fn wake_thread(&mut self, thid: i32) {
        push_wake!(self, thid);
    }

    /// Take the recorded wake causes - see `push_wake!`.
    pub fn take_wake_causes(&mut self) -> Vec<(i32, i32, &'static str)> {
        std::mem::take(&mut self.wake_causes)
    }

    /// Take the parked threads a host call just made runnable (scheduler hook).
    pub fn take_wakes(&mut self) -> Vec<i32> {
        std::mem::take(&mut self.pending_wakes)
    }

    /// Take the PARKED threads the runtime has ended since the last call (scheduler hook):
    /// each is to be dropped without being resumed, as if it had ended with `ThreadExit` -
    /// see `retire_fiber_thread`. Their exit is already recorded here.
    pub fn take_terminations(&mut self) -> Vec<i32> {
        std::mem::take(&mut self.pending_terminations)
    }

    /// Take the threads that TIME released since the last call - a timed wait that expired or
    /// a sleep that ended - as opposed to another thread's signal. A scheduler may let such a
    /// thread yield to ones already waiting for the CPU: when the clock is pushed forward in
    /// steps (the wall-clock floor), a short timed wait can expire as it begins, and a poller
    /// that never actually waits starves every lower-priority thread on its core. On the
    /// console a 1 ms wait hands the core over for 1 ms.
    pub fn take_timeout_wakes(&mut self) -> Vec<i32> {
        std::mem::take(&mut self.timeout_wakes)
    }

    /// How many live thread records is so many that the guest must be in a retry loop.
    ///
    /// No real title holds thousands of threads at once; a title that reaches this is
    /// creating and abandoning them. Left unchecked, each one takes a stack off a bump
    /// allocator that cannot free, so the run ends in a heap exhaustion whose crash site is
    /// unrelated to the cause - an out-of-bounds access with a stack pointer near zero, in
    /// whichever thread happened to start last. Naming the entry point that is repeating
    /// turns that into a one-line diagnosis. Measured against a real case: a trophy-info
    /// thread respawning on an error it could not act on reached ~74,000.
    pub(super) const RUNAWAY_THREAD_LIMIT: usize = 2048;

    /// Create a thread: allocate its own stack and record it, returning its
    /// SceUID. The entry's Thumb bit is cleared so it names the transpiled export.
    pub fn create_thread(
        &mut self,
        entry: u32,
        stack_size: u32,
        priority: i32,
        attr: u32,
        cpu_affinity: i32,
    ) -> i32 {
        if self.threads.len() >= Self::RUNAWAY_THREAD_LIMIT {
            // Report ONCE, with the entry that dominates, then keep going: the heap
            // exhaustion below will stop the run loudly anyway, and this is the line that
            // explains it.
            if !self.runaway_threads_reported {
                self.runaway_threads_reported = true;
                let mut counts: std::collections::HashMap<u32, usize> =
                    std::collections::HashMap::new();
                for t in &self.threads {
                    *counts.entry(t.entry).or_insert(0) += 1;
                }
                let worst = counts.iter().max_by_key(|(_, n)| **n);
                tracing::error!(
                    target: "vitaslop::err",
                    live = self.threads.len(),
                    entry = format_args!("{entry:#x}"),
                    repeated_entry = format_args!("{:#x}", worst.map(|(e, _)| *e).unwrap_or(0)),
                    repeated_count = worst.map(|(_, n)| *n).unwrap_or(0),
                    "RUNAWAY thread creation - the guest is spawning threads it never \
                     reaps, which will exhaust the guest heap and crash somewhere \
                     unrelated. Almost always a host call returning something the guest \
                     treats as retryable: look at what the repeated entry point calls."
                );
            }
        }
        let size = stack_size.max(0x1000);
        // A deleted thread's stack is deliberately NOT recycled here. It looks like free
        // memory - the guest allocator is a bump allocator with no free, so recycling is
        // tempting - but "the guest deleted the thread" and "the host has finished with
        // that fiber" are different events: an exited thread's stack can still be live
        // under the scheduler when the guest deletes its record. Handing it to a new
        // thread then corrupts two threads' stacks at once, which surfaces as an
        // out-of-bounds access with a nonsense stack pointer nowhere near the cause.
        // Leaking it is the safe direction; `free_stacks` records the leak so it is
        // measurable rather than invisible.
        let stack = self.galloc(size, 16);
        // The stack grows down from an 8-byte-aligned top (AAPCS at a public call).
        let stack_top = stack.wrapping_add(size) & !0xF;
        self.guest_stacks.note(stack, stack.wrapping_add(size));
        let uid = self.next_uid;
        self.next_uid += 1;
        // Resolve the sentinel/relative priority to a concrete value now, so every
        // scheduler comparison against it (and against the main thread's default)
        // is meaningful.
        let priority = resolve_priority(priority);
        // >>> MASK 0 INHERITS THE CREATOR'S MASK (`SCE_KERNEL_THREAD_CPU_AFFINITY_MASK_DEFAULT`:
        // "Inherit calling thread affinity mask", vitasdk). It used to be stored as 0 and read
        // as "any core", so a title that pins its main thread and lets its helpers inherit
        // the pin had them scattered over every worker. MEASURED (phone, a fighting title,
        // 10-02): main asks for core 0 (`SceProcessParam`) and creates its movie player's
        // controller, decoders and demuxer with mask 0; spread over three workers the
        // controller's buffering check ran BESIDE the demuxer and found its queues empty 9-17
        // ms into playback, declared buffering and the movie never resumed (black screen); on
        // one core, strict priority runs the demuxer (142) before the controller (149).
        let cpu_affinity = if cpu_affinity == 0 {
            if self.current == MAIN_THID {
                self.main_affinity
            } else {
                self.threads
                    .iter()
                    .find(|t| t.uid == self.current)
                    .map_or(0, |t| t.cpu_affinity)
            }
        } else {
            cpu_affinity
        };
        tracing::trace!(
            target: "vitaslop::thread",
            uid,
            entry = format_args!("{:#x}", entry & !1),
            priority = format_args!("{priority:#x}"),
            main_priority = format_args!("{DEFAULT_THREAD_PRIORITY:#x}"),
            "create"
        );
        self.threads.push(ThreadRec {
            uid,
            name: String::new(),
            entry: entry & !1,
            stack_top,
            stack: (stack, size),
            started: false,
            exit_code: None,
            priority,
            init_priority: priority,
            attr,
            cpu_affinity,
            signals: 0,
        });
        uid
    }

    // --- SceFiber -----------------------------------------------------------
    //
    // See `host/fibers.rs` for why a fiber is backed by a scheduler thread. Everything
    // below is bookkeeping around one invariant: within a fiber chain exactly ONE of
    // {the running thread, one fiber} is runnable, and every transition hands the
    // baton over in a single step (wake the target, park the caller).

    /// `SCE_FIBER_ERROR_*`, from vitasdk `fiber.h`.
    pub const FIBER_ERROR_NULL: i32 = 0x8059_0001u32 as i32;
    pub const FIBER_ERROR_INVALID: i32 = 0x8059_0004u32 as i32;
    pub const FIBER_ERROR_PERMISSION: i32 = 0x8059_0005u32 as i32;
    pub const FIBER_ERROR_STATE: i32 = 0x8059_0006u32 as i32;

    /// The fiber the CURRENT thread is executing, if it is executing one.
    #[inline]
    pub(super) fn current_fiber_index(&self) -> Option<usize> {
        self.fibers.running_on(self.current)
    }

    /// The guest `SceFiber*` the current thread is running, or 0 for a plain thread.
    /// This is `sceFiberGetSelf`.
    pub fn current_fiber(&self) -> u32 {
        self.current_fiber_index()
            .map(|i| self.fibers.addr(i))
            .unwrap_or(0)
    }

    /// The thread a title's own code would call itself: for a fiber's backing thread,
    /// the THREAD that ran the chain, because on hardware a fiber runs on its runner's
    /// thread and shares its identity and its thread-local storage. Everything keyed by
    /// "which thread am I" - `sceKernelGetThreadId`, `sceKernelGetTLSAddr` - has to ask
    /// this rather than the raw scheduler id, or a fiber-based job system sees a
    /// different worker identity (and a different TLS block) each time it switches.
    /// The thread a newly spawned `thid` should be placed BESIDE, if any: a fiber's first real
    /// runner (see `fiber_dispatch`). `None` for an ordinary thread.
    pub fn spawn_near(&self, thid: i32) -> Option<i32> {
        self.spawn_near
            .iter()
            .rev()
            .find(|(t, _)| *t == thid)
            .map(|(_, r)| *r)
    }

    #[inline]
    pub fn logical_thread(&self, thid: i32) -> i32 {
        self.fibers.logical_thread(thid)
    }

    /// `_sceFiberInitializeImpl(fiber, name, entry, argOnInitialize, addrContext,
    /// sizeContext, params)`: record the fiber and create - but do not start - the
    /// thread that will run it.
    ///
    /// `addrContext` may be null (a fiber initialised without a stack), in which case
    /// `_sceFiberAttachContextAndSwitch` supplies one before it first runs.
    #[allow(clippy::too_many_arguments)]
    pub fn fiber_initialize(
        &mut self,
        addr: u32,
        name: String,
        entry: u32,
        arg_on_initialize: u32,
        context_addr: u32,
        context_size: u32,
    ) -> i32 {
        if addr == 0 || entry == 0 {
            return Self::FIBER_ERROR_NULL;
        }
        // Re-initialising a live fiber would strand its parked thread on a stack the
        // guest is about to reuse. That is a guest error the kernel refuses, so refuse
        // it here rather than leaving a thread parked forever.
        if let Some(i) = self.fibers.index(addr) {
            if self.fibers.is_live_and_running(i) {
                return Self::FIBER_ERROR_STATE;
            }
            let old = self.fibers.remove(i);
            self.retire_fiber_thread(old);
        }
        // The finalized records at this address are superseded by the one made here.
        self.fibers.drop_finalized(addr);
        // Its stack is a guest buffer, and SP-relative stores into it are unmarked - see
        // `GuestStacks`.
        if context_addr != 0 {
            self.guest_stacks
                .note(context_addr, context_addr.wrapping_add(context_size));
        }
        let uid = self.new_fiber_thread(&name, entry, fiber_stack_top(context_addr, context_size));
        self.fibers.add(
            addr,
            name,
            entry,
            arg_on_initialize,
            (context_addr, context_size),
            uid,
        );
        0
    }

    /// The (not yet started) scheduler thread that backs a fiber. The fiber runs on a stack
    /// the GUEST owns - its context, or for a context-less fiber its runner's - so the record
    /// owns none: `stack: (0, 0)` marks that there is nothing to reclaim.
    pub(super) fn new_fiber_thread(&mut self, name: &str, entry: u32, stack_top: u32) -> i32 {
        let uid = self.next_uid;
        self.next_uid += 1;
        self.threads.push(ThreadRec {
            uid,
            name: name.to_string(),
            entry: entry & !1,
            stack_top,
            stack: (0, 0),
            started: false,
            exit_code: None,
            priority: DEFAULT_THREAD_PRIORITY,
            init_priority: DEFAULT_THREAD_PRIORITY,
            attr: 0,
            cpu_affinity: 0,
            signals: 0,
        });
        uid
    }

    /// >>> A RETIRED FIBER'S BACKING THREAD IS DROPPED, NOT LEFT IN THE THREAD TABLE.
    ///
    /// A fiber record is retired by `sceFiberFinalize` and by re-initialising the same
    /// `SceFiber`; a context-less fiber is also re-armed with a fresh, unstarted thread every
    /// time it leaves (`fiber_restart_later`). A title that initialises a fiber per job and
    /// finalizes it afterwards therefore retires one thread per job, and leaving those records
    /// behind grew the table by one per fiber per frame. MEASURED: one title's first mission
    /// reached 64,307 thread records in 3,900 frames, every lookup a linear walk over them,
    /// the frame time climbing until the stall watchdog fired. A thread that never started has
    /// no scheduler state, so its record is all there is to drop; one that already ended has
    /// none left either.
    ///
    /// >>> A PARKED ONE IS ENDED HERE, AND THE SCHEDULER DROPS IT. A fiber with a context
    /// that switched away or returned to its thread is parked in that svc, and finalizing it
    /// (or re-initialising its `SceFiber`) leaves nothing that can ever wake it: the record
    /// that named it is gone. Leaving it parked kept one scheduler thread - an instance, a
    /// stack, a slot every pass walks - per finalized fiber for the rest of the run. MEASURED
    /// (headless, the title above): 64,020 live threads, 63,999 of them fiber threads
    /// stranded this way, by frame 3,900, every frame slower than the last. So its end is
    /// recorded here as a `ThreadExit` would record it (`set_thread_exit`, code 0 - nothing
    /// joins a fiber thread, and it runs no EXIT handlers because on the hardware it was never
    /// a thread) and it is queued in `pending_terminations` for the scheduler to drop without
    /// resuming. On the hardware the suspended context is simply forgotten; this is the same
    /// thing done to the thread that stands in for it.
    pub(super) fn retire_fiber_thread(&mut self, thid: i32) {
        let Some(i) = self.threads.iter().position(|t| t.uid == thid) else {
            return;
        };
        if self.threads[i].started && self.threads[i].exit_code.is_none() {
            // A fiber svc parks WITHOUT a waiter record, so a parked fiber thread is in no
            // primitive's list. One that is would be a fiber thread blocked in a kernel wait,
            // which this model never produces - said loudly rather than left to a stale entry
            // whose wake would name a thread that no longer exists.
            let wait = self.thread_wait_state(thid);
            if wait != "RUNNABLE" {
                tracing::warn!(
                    target: "vitaslop::sched",
                    "fiber thread {thid:#x} ended by a finalize while {wait}; its wait entry \
                     is left to name a thread the scheduler has dropped"
                );
            }
            self.set_thread_exit(thid, 0);
            self.pending_terminations.push(thid);
            self.fibers.note_terminated();
            // Anything queued FOR it would be delivered to a thread that is gone: a wake the
            // scheduler has not drained yet, a timed wait's resume code, a pending guest call.
            self.pending_wakes.retain(|&t| t != thid);
            self.wake_causes.retain(|&(t, _, _)| t != thid);
            self.timeout_wakes.retain(|&t| t != thid);
            self.pending_resume_codes.retain(|&(t, _)| t != thid);
            self.guest_calls.forget_thread(thid);
            self.sleep_waiters.retain(|&(t, _)| t != thid);
            self.signal_waiters.retain(|&t| t != thid);
            self.join_waiters.retain(|&(t, _, _)| t != thid);
            self.sync.forget_lwcond_waiter(thid);
            self.storage.forget(thid);
        }
        // The record, and the per-thread rows only an instantiation reads: a retired fiber's
        // thread is never instantiated again, and each row left behind is one more entry in a
        // list walked linearly per spawn - the same growth as the records themselves.
        self.threads.swap_remove(i);
        self.spawn_near.retain(|&(t, _)| t != thid);
        self.tls.forget_thread(thid);
    }

    /// >>> A CONTEXT-LESS FIBER THAT LEAVES IS ABANDONED, AND RESTARTS AT ITS ENTRY.
    ///
    /// A fiber initialised with no context runs on its runner's stack, and the library saves
    /// nothing of it when it hands the baton on (`ReturnToThread`, or a `Switch` away): its
    /// frames are gone, and the next Run or Switch to it begins at `entry` again. So its
    /// backing thread ENDS here (the caller returns `ThreadExit`) and the record gets a
    /// fresh, unstarted one. Parking it instead - what this model did - left a thread to
    /// resume mid-function on a stack the runner has since reused.
    pub(super) fn fiber_restart_later(&mut self, i: usize) {
        let (name, entry) = self.fibers.name_and_entry(i);
        let uid = self.new_fiber_thread(&name, entry, 0);
        self.fibers.restart_on(i, uid);
    }

    /// Give a fiber the baton: start its thread on first use, or wake it and deliver
    /// `arg_on_run` through the out-pointer of the call it is parked in.
    pub(super) fn fiber_dispatch(&mut self, i: usize, runner: i32, arg_on_run: u32) {
        let (thid, entry, arg0, stack_top) = match self.fibers.dispatch(i, runner) {
            Dispatch::Wake { thid, out } => {
                if out != 0 {
                    self.pending_stat_writes.push((out, arg_on_run));
                }
                push_wake!(self, thid);
                return;
            }
            Dispatch::Start {
                thid,
                entry,
                arg0,
                stack_top,
            } => (thid, entry, arg0, stack_top),
        };
        // A fiber shares its runner's thread-local storage, because on hardware it IS
        // that thread. Bind the block before the scheduler instantiates the fiber (it
        // reads the base then), so a `__thread` variable the fiber touches is the same
        // storage the runner sees.
        let runner_tls = self.ensure_tls_block(self.logical_thread(runner));
        if runner_tls != 0 {
            self.tls.bind(thid, runner_tls);
        }
        // >>> A FIBER RUNS ON ITS RUNNER'S CORE AT ITS RUNNER'S PRIORITY - it is user-mode
        // context switching inside that thread, not a thread of its own. The default priority
        // here, and a placement anywhere, made every Run/Switch a CROSS-WORKER handoff under
        // SMP. MEASURED (phone, a fighting title's fight, SMP trace job 528): main (prio 0x56,
        // core 0) blocked in sceFiberRun 7.7 ms a frame while its fibers ran on another worker,
        // the frame chain came to 27 ms and the title paced itself at 30 fps (60 on a Vita).
        let logical_runner = self.logical_thread(runner);
        let runner_priority = if fiber_runner_priority() {
            self.threads
                .iter()
                .find(|t| t.uid == logical_runner)
                .map_or(
                    if logical_runner == MAIN_THID {
                        self.main_priority
                    } else {
                        DEFAULT_THREAD_PRIORITY
                    },
                    |t| t.priority,
                )
        } else {
            DEFAULT_THREAD_PRIORITY
        };
        let runner_mask = self.thread_cpu_affinity(logical_runner);
        if let Some(t) = self.threads.iter_mut().find(|t| t.uid == thid) {
            t.started = true;
            t.stack_top = stack_top;
            t.priority = runner_priority;
            t.init_priority = runner_priority;
            if runner_mask > 0 && fiber_runner_priority() {
                t.cpu_affinity = runner_mask;
            }
        }
        self.spawn_near.push((thid, logical_runner));
        self.pending_spawns.push(Reentry {
            entry,
            // `void entry(SceUInt32 argOnInitialize, SceUInt32 argOnRun)`.
            arg_len: arg0,
            arg_ptr: arg_on_run,
            r2: 0,
            r3: 0,
            stack_top,
            thid,
            priority: runner_priority,
        });
    }

    /// `sceFiberRun(fiber, argOnRunTo, argOnRun)` called by a plain THREAD: hand the
    /// fiber the baton and park the caller until the chain returns to it. Returns
    /// `Ok(())` when the caller must park, or the errno to return instead.
    pub fn fiber_run(
        &mut self,
        addr: u32,
        arg_on_run_to: u32,
        arg_on_run_out: u32,
        caller_sp: u32,
    ) -> Result<(), i32> {
        if self.current_fiber_index().is_some() {
            // A fiber must Switch, not Run: running nests a second chain on one thread,
            // which the baton model (and the hardware) does not have.
            return Err(Self::FIBER_ERROR_PERMISSION);
        }
        let Some(i) = self.fibers.index(addr) else {
            return Err(Self::FIBER_ERROR_INVALID);
        };
        if self.fibers.is_running(i) {
            return Err(Self::FIBER_ERROR_STATE);
        }
        // A fiber with no context is run, not refused: it runs on this thread's stack - see
        // `fiber_restart_later`.
        let runner = self.current;
        self.fibers.note_run(runner, caller_sp, arg_on_run_out);
        self.fiber_dispatch(i, runner, arg_on_run_to);
        Ok(())
    }

    /// `sceFiberSwitch(fiber, argOnRunTo, argOnRun)` called by a FIBER: pass the baton
    /// sideways, keeping the same runner. `context` optionally attaches a stack first
    /// (`_sceFiberAttachContextAndSwitch`).
    pub fn fiber_switch(
        &mut self,
        addr: u32,
        arg_on_run_to: u32,
        arg_on_run_out: u32,
        context: Option<(u32, u32)>,
    ) -> Result<bool, i32> {
        let Some(from) = self.current_fiber_index() else {
            return Err(Self::FIBER_ERROR_PERMISSION);
        };
        let Some(to) = self.fibers.index(addr) else {
            return Err(Self::FIBER_ERROR_INVALID);
        };
        if to == from || self.fibers.is_running(to) {
            return Err(Self::FIBER_ERROR_STATE);
        }
        if let Some(c) = context {
            // Attaching a context to a fiber that is already running on one would move
            // the stack out from under live frames.
            if self.fibers.started(to) {
                return Err(Self::FIBER_ERROR_STATE);
            }
            self.fibers.attach_context(to, c);
        }
        // A context-less target starts at its entry on the chain's thread stack (see
        // `fiber_dispatch`); it is not refused.
        let runner = self.fibers.hand_off(from, arg_on_run_out);
        self.fiber_dispatch(to, runner, arg_on_run_to);
        // `true`: the fiber that switched away had no context, so it is abandoned rather
        // than parked, and the caller's thread ends - see `fiber_restart_later`.
        let abandoned = self.fibers.context_less(from);
        if abandoned {
            self.fiber_restart_later(from);
        }
        Ok(abandoned)
    }

    /// `sceFiberReturnToThread(argOnReturn, argOnRun)`: give the baton back to the
    /// thread that ran this chain and park the fiber where it stands.
    pub fn fiber_return_to_thread(
        &mut self,
        arg_on_return: u32,
        arg_on_run_out: u32,
    ) -> Result<bool, i32> {
        let Some(i) = self.current_fiber_index() else {
            return Err(Self::FIBER_ERROR_PERMISSION);
        };
        if !self.fibers.is_running(i) {
            return Err(Self::FIBER_ERROR_STATE);
        }
        let runner = self.fibers.hand_off(i, arg_on_run_out);
        let out = self.fibers.end_run(runner);
        if out != 0 {
            self.pending_stat_writes.push((out, arg_on_return));
        }
        push_wake!(self, runner);
        // `true`: a context-less fiber is abandoned, not parked - see `fiber_restart_later`.
        let abandoned = self.fibers.context_less(i);
        if abandoned {
            self.fiber_restart_later(i);
        }
        Ok(abandoned)
    }

    /// `sceFiberFinalize(fiber)`: retire a fiber that is not running.
    pub fn fiber_finalize(&mut self, addr: u32) -> i32 {
        let Some(i) = self.fibers.index(addr) else {
            return Self::FIBER_ERROR_INVALID;
        };
        // Only the fiber that is RUNNING is refused.
        if self.fibers.is_running(i) {
            return Self::FIBER_ERROR_STATE;
        }
        // >>> A SUSPENDED FIBER IS FINALIZED, NOT REFUSED. One that has switched away or
        // returned to its thread mid-function holds only a saved resume point, and finalizing
        // it drops that point and leaves its context memory to the guest - which then commonly
        // re-initialises the same `SceFiber` for the next job. Refusing it (a STATE error
        // this model used to return) is a failure the hardware does not have, and a title's
        // fiber pool does not check for one. Its parked backing thread is ENDED, not left
        // parked: nothing could name it once the record is retired (see `retire_fiber_thread`).
        let thid = self.fibers.finalize(i);
        self.retire_fiber_thread(thid);
        0
    }

    /// Every record this model holds for the fiber at `addr` - finalized ones included, because
    /// a stale record at a reused address is exactly what a refusal needs to show - as one line:
    /// its thread, whether that has started and finished, and the runner it thinks it has.
    pub fn fiber_state_line(&self, addr: u32) -> String {
        let recs: Vec<String> = self
            .fibers
            .records_at(addr)
            .into_iter()
            .map(|(thid, context, started, finalized, runner)| {
                format!(
                    "[thid {} context {:#x}/{:#x} started {} finished {} finalized {} runner {}]",
                    thid,
                    context.0,
                    context.1,
                    started,
                    self.thread_finished(thid),
                    finalized,
                    runner
                )
            })
            .collect();
        if recs.is_empty() { "no record".to_string() } else { recs.join(" ") }
    }

    /// `sceFiberGetInfo(fiber, SceFiberInfo *out)`: entry, argOnInitialize, context and
    /// name, at the offsets vitasdk's `SceFiberInfo` publishes.
    #[inline]
    pub fn fiber_info(&self, addr: u32) -> Option<(u32, u32, u32, u32, &str)> {
        self.fibers.info(addr)
    }

    /// int sceKernelChangeThreadPriority(SceUID thid, int priority): retarget a
    /// thread's scheduler priority. `thid` 0 means the calling thread. Returns the
    /// PREVIOUS priority on success (what the kernel returns, and what a title saves
    /// to restore after a temporary boost), or an error for an unknown id.
    ///
    /// This genuinely moves the thread in the run order - the scheduler picks the
    /// highest-priority runnable thread - so a title raising a loader above the main
    /// thread gets the ordering it asked for.
    ///
    /// `priority` 0 is the CALLER's current priority; one in `0x100000E0..=0x1000011F` is
    /// relative to the default; anything else must be an absolute user priority in `64..=191`
    /// or the change is refused with ILLEGAL_PRIORITY. Returns the previous priority.
    pub fn change_thread_priority(&mut self, thid: i32, priority: i32) -> Result<i32, u32> {
        let thid = if thid == 0 { self.current } else { thid };
        let priority = match priority {
            0 => self.current_priority(),
            0x1000_00E0..=0x1000_011F => resolve_priority(priority),
            64..=191 => priority,
            _ => return Err(SCE_KERNEL_ERROR_ILLEGAL_PRIORITY),
        };
        match self.threads.iter_mut().find(|t| t.uid == thid) {
            Some(t) => {
                let previous = t.priority;
                t.priority = priority;
                Ok(previous)
            }
            // The INITIAL thread has no create record; its priority lives beside the table
            // (`current_priority` reads it from there too).
            None if thid == MAIN_THID => Ok(std::mem::replace(&mut self.main_priority, priority)),
            None => Err(SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID),
        }
    }

    /// Everything `sceKernelGetThreadInfo` reports that this kernel models:
    /// `(name, attr, status, entry, stack base, stack size, init priority, current
    /// priority, cpu affinity, exit status)`. `thid` 0 means the calling thread.
    pub fn thread_info(&self, thid: i32) -> Option<ThreadInfo<'_>> {
        let thid = if thid == 0 { self.current } else { thid };
        let t = self.threads.iter().find(|t| t.uid == thid)?;
        Some(ThreadInfo {
            name: t.name.as_str(),
            attr: t.attr,
            status: thread_status(t, self.current),
            entry: t.entry,
            stack_base: t.stack.0,
            stack_size: t.stack.1 as i32,
            init_priority: t.init_priority,
            current_priority: t.priority,
            cpu_affinity: t.cpu_affinity,
            exit_status: t.exit_code.unwrap_or(0) as i32,
        })
    }

    /// int sceKernelSendSignal(SceUID thid): deliver one signal to a thread, waking it
    /// if it is parked in `sceKernelWaitSignal`. Counted, so a send that arrives before
    /// the wait is not lost - that ordering is the whole point of the primitive.
    ///
    /// >>> A LATCH OF ONE, AND THE INITIAL THREAD CAN BE SIGNALLED TOO. The SDK documents the
    /// primitive as "a semaphore with limit 1": a second send before the first is consumed is
    /// refused with `SCE_KERNEL_ERROR_ALREADY_SENT`, it is not counted. And the INITIAL thread
    /// ([`MAIN_THID`]) has no create record here, so a send to it used to fail as an unknown
    /// thread and the signal was lost - MEASURED on a 2011 adventure title: its save thread
    /// signalled the main thread, got `0x80028021`, and every thread of the process then waited
    /// on a signal for ever at frame 0.
    pub fn send_signal(&mut self, thid: i32) -> Result<(), u32> {
        let slot: &mut u32 = if thid == MAIN_THID {
            &mut self.main_signals
        } else {
            match self.threads.iter_mut().find(|t| t.uid == thid) {
                Some(t) => &mut t.signals,
                None => return Err(SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID),
            }
        };
        if *slot > 0 {
            return Err(SCE_KERNEL_ERROR_ALREADY_SENT);
        }
        *slot = 1;
        if self.preemptive && self.signal_waiters.contains(&thid) {
            self.signal_waiters.retain(|w| *w != thid);
            self.take_signal(thid);
            push_wake!(self, thid);
        }
        Ok(())
    }

    /// Consume the pending signal of a thread, if it has one.
    pub fn take_signal(&mut self, thid: i32) -> bool {
        let slot: Option<&mut u32> = if thid == MAIN_THID {
            Some(&mut self.main_signals)
        } else {
            self.threads
                .iter_mut()
                .find(|t| t.uid == thid)
                .map(|t| &mut t.signals)
        };
        match slot {
            Some(n) if *n > 0 => {
                *n -= 1;
                true
            }
            _ => false,
        }
    }

    /// Park the current thread until a signal is sent to it (`sceKernelWaitSignal`).
    pub fn signal_block(&mut self) {
        self.signal_waiters.push(self.current);
    }

    /// Record the guest's own name for a thread (from `sceKernelCreateThread`).
    pub fn set_thread_name(&mut self, thid: i32, name: &str) {
        if let Some(t) = self.threads.iter_mut().find(|t| t.uid == thid) {
            t.name = name.to_string();
        }
    }

    /// int sceKernelDeleteThread(SceUID thid): drop a DORMANT thread's record and hand its
    /// stack back for reuse.
    ///
    /// Deleting a thread that is still running is refused with `SCE_KERNEL_ERROR_NOT_DORMANT`,
    /// as the kernel does - accepting it would invalidate the id of a thread the scheduler
    /// is about to resume. A thread that was created and never started is dormant too, which
    /// is why [`ThreadRec::started`] exists: neither it nor a running thread has an exit code.
    pub fn delete_thread(&mut self, thid: i32) -> Result<(), u32> {
        let Some(i) = self.threads.iter().position(|t| t.uid == thid) else {
            return Err(SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID);
        };
        let t = &self.threads[i];
        if t.started && t.exit_code.is_none() {
            return Err(SCE_KERNEL_ERROR_NOT_DORMANT);
        }
        let rec = self.threads.swap_remove(i);
        self.free_stacks.push(rec.stack);
        tracing::trace!(target: "vitaslop::thread", uid = thid, "delete");
        Ok(())
    }

    /// Record a thread's requested CPU affinity mask (`sceKernelChangeThreadCpuAffinityMask`).
    ///
    /// Stored, not obeyed: this scheduler interleaves guest threads on one baton, so there
    /// is no per-core placement to honour. What matters is that the getter agrees with the
    /// setter - a title that sets an affinity and reads it back must not be contradicted.
    /// The `cpu_affinity` field is the same one `sceKernelCreateThread` fills and
    /// `sceKernelGetThreadInfo` already reports, so all three now tell one story.
    pub fn set_thread_cpu_affinity(&mut self, thid: i32, mask: i32) -> i32 {
        // 0 addresses the CALLING thread throughout the threadmgr API.
        let target = if thid == 0 { self.current } else { thid };
        match self.threads.iter_mut().find(|t| t.uid == target) {
            Some(t) => {
                t.cpu_affinity = mask;
                0
            }
            None => SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID as i32,
        }
    }

    /// A thread's CPU affinity mask, or a negative error. See
    /// [`Self::set_thread_cpu_affinity`].
    pub fn thread_cpu_affinity(&self, thid: i32) -> i32 {
        let target = if thid == 0 { self.current } else { thid };
        if target == MAIN_THID && !self.threads.iter().any(|t| t.uid == MAIN_THID) {
            return if self.main_affinity == 0 {
                crate::vita::threadmgr::CPU_MASK_USER_ALL
            } else {
                self.main_affinity
            };
        }
        match self.threads.iter().find(|t| t.uid == target) {
            // A thread created with 0 ("inherit") has never named a set, and the kernel
            // answers with the real one it may run on rather than the sentinel.
            Some(t) if t.cpu_affinity == 0 => crate::vita::threadmgr::CPU_MASK_USER_ALL,
            Some(t) => t.cpu_affinity,
            None => SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID as i32,
        }
    }

    /// The priority the current thread is running at (lower = higher priority).
    pub fn current_priority(&self) -> i32 {
        self.threads.iter().find(|t| t.uid == self.current).map_or(
            if self.current == MAIN_THID {
                self.main_priority
            } else {
                DEFAULT_THREAD_PRIORITY
            },
            |t| t.priority,
        )
    }

    /// Start a thread. In the single-thread model this raises a *synchronous*
    /// re-entry (the engine host runs the entry to completion before the guest sees
    /// anything past the start call). Under the preemptive scheduler it instead
    /// queues a *spawn*: the entry becomes its own concurrent fiber, and the
    /// starting thread keeps running.
    /// Returns true if the started thread outranks the current one (lower priority
    /// number), so the caller should reschedule at once - the real kernel preempts
    /// the starter and runs the higher-priority thread until it blocks. This is the
    /// ordering guarantee titles rely on when a worker must initialize (e.g. create
    /// and store a semaphore the starter then waits on) before the starter proceeds.
    ///
    /// >>> A RUNNING THREAD IS NOT STARTED AGAIN, AND A RESTARTED ONE IS RUNNING AGAIN.
    ///
    /// Both used to be missing. The exit code of a thread's FIRST run was never cleared, so
    /// `sceKernelWaitThreadEnd` on every later run returned at once, while the thread was still
    /// going - and the next `sceKernelStartThread` then launched a second run of the SAME thread
    /// on the SAME stack while the first was still using it. MEASURED on the phone (a baseball title, two
    /// runs of ~30): a per-frame job worker (entry 0x8104ecfe) whose command cursor lives in its
    /// 12 KB stack frame read garbage commands and called a null handler - `GUEST FAULT` at
    /// f3978 and f4321, `pc=0`, callee-saved registers clobbered. The kernel keeps a started
    /// thread RUNNING until it ends and refuses to start it again with `NOT_DORMANT`
    /// (the same rule [`Self::delete_thread`] applies).
    pub fn start_thread(&mut self, thid: i32, arg_len: u32, arg_ptr: u32) -> Result<bool, u32> {
        let Some(t) = self.threads.iter_mut().find(|t| t.uid == thid) else {
            return Err(SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID);
        };
        if t.started && t.exit_code.is_none() {
            tracing::debug!(target: "vitaslop::thread", thid, "sceKernelStartThread on a running thread refused (NOT_DORMANT)");
            return Err(SCE_KERNEL_ERROR_NOT_DORMANT);
        }
        t.started = true;
        t.exit_code = None;
        let new_priority = t.priority;
        let req = Reentry {
            entry: t.entry,
            arg_len,
            arg_ptr,
            r2: 0,
            r3: 0,
            stack_top: t.stack_top,
            thid,
            priority: new_priority,
        };
        // A new run of the thread, so its next end owes the EXIT handlers again.
        self.thread_events.forget_exit(thid);
        if self.preemptive {
            self.pending_spawns.push(req);
            Ok(new_priority < self.current_priority())
        } else {
            self.pending_reentry = Some(req);
            Ok(false)
        }
    }

    /// The exit code of a finished thread, if it has run.
    pub fn thread_exit_code(&self, thid: i32) -> Option<u32> {
        self.threads
            .iter()
            .find(|t| t.uid == thid)
            .and_then(|t| t.exit_code)
    }

    /// Whether `thid` is STARTED and has not ended - the state `sceKernelStartThread` and
    /// `sceKernelDeleteThread` refuse with `NOT_DORMANT`.
    pub fn thread_running(&self, thid: i32) -> bool {
        self.threads
            .iter()
            .any(|t| t.uid == thid && t.started && t.exit_code.is_none())
    }

    /// Whether the thread with id `thid` has finished.
    pub fn thread_finished(&self, thid: i32) -> bool {
        self.threads
            .iter()
            .any(|t| t.uid == thid && t.exit_code.is_some())
    }

    /// Take the pending thread-run request, if any (drained by the engine host).
    pub fn take_reentry(&mut self) -> Option<Reentry> {
        self.pending_reentry.take()
    }

    /// >>> CALL A GUEST FUNCTION FROM INSIDE A HOST CALL, ON THE CALLER'S THREAD.
    ///
    /// Returns the call's result when this dispatch has already had it run, and `None` when
    /// it has not: the handler must then RETURN at once (any value - the register file is
    /// restored for it) and the engine runs `call`, then dispatches the same host call
    /// again. `capture` names a guest region `(addr, len)` to copy the moment the call
    /// returns, into [`GuestCallResult::bytes`] - a read callback's buffer, which a later
    /// call may reuse before the replay gets back to it.
    ///
    /// # The replay contract
    /// A dispatch that makes N guest calls runs N+1 times, each run answering one more call
    /// from the record. So up to its LAST guest call a handler must be a pure function of
    /// its arguments, the guest memory it reads and the results it is handed: anything it
    /// changes before then (host state, guest memory, a counter) happens once per replay.
    /// Do the reading first and the changing after - which is also the order a library
    /// that can fail half-way has to keep anyway.
    ///
    /// # Why replay, and not a suspended host function
    /// A host call holds `&mut VitaState` for its whole run, and the guest function it waits
    /// on makes host calls of its own, which need the same state - so the handler cannot be
    /// suspended mid-body holding it. Re-running it is the one shape that needs no second
    /// copy of the state and no unsafe aliasing, and every engine can drive it: native nests a
    /// wasm call, the browser nests a JSPI stack.
    pub fn call_guest(
        &mut self,
        call: GuestCall,
        capture: Option<(u32, u32)>,
    ) -> Option<GuestCallResult> {
        self.guest_calls.call(self.current, call, capture)
    }

    /// The engine's movie I/O buffer in guest memory (`size` bytes, the first request fixes
    /// it): where a title's read function puts a movie's bytes when the title gave SceMp4
    /// no allocator of its own. One per run - the heap it comes from is a bump allocator
    /// with no free, and only one movie is open at a time.
    pub(crate) fn mp4_io_scratch(&mut self, size: u32) -> u32 {
        if self.mp4_io_scratch == 0 {
            self.mp4_io_scratch = self.galloc(size, 64);
        }
        self.mp4_io_scratch
    }

    /// Whether the running dispatch has asked for a guest call it does not have yet - a
    /// helper deep in a handler returns a neutral value, and the handler checks this before
    /// reading that value as a failure.
    #[inline]
    pub fn guest_call_pending(&self) -> bool {
        self.guest_calls.pending(self.current)
    }

    /// Before a dispatch: pick the record it replays against - the innermost frame of this
    /// thread when that frame's call has just returned, none otherwise (a new call, or a host
    /// call the guest function itself is making).
    #[inline]
    pub(crate) fn guest_call_begin(&mut self) {
        self.guest_calls.begin(self.current)
    }

    /// After a dispatch: whether it is waiting on a guest call ([`SvcOutcome::CallGuest`]).
    /// A dispatch that finished drops its record.
    #[inline]
    pub(crate) fn guest_call_end(&mut self) -> bool {
        self.guest_calls.end(self.current)
    }

    /// The guest call thread `thid`'s last dispatch is waiting on.
    #[inline]
    pub fn take_guest_call(&mut self, thid: i32) -> Option<GuestCall> {
        self.guest_calls.take(thid)
    }

    /// The guest call thread `thid` was waiting on has returned: record its result (and the
    /// capture it asked for) so the replay that follows finds it.
    #[inline]
    pub fn guest_call_returned(
        &mut self,
        thid: i32,
        r0: u32,
        r1: u32,
        mem: &dyn GuestMemory,
        base: u32,
    ) {
        self.guest_calls.returned(thid, r0, r1, mem, base)
    }

    /// Record a `sceKernelRegisterThreadEventHandler` registration and return its id.
    pub fn register_thread_event_handler(
        &mut self,
        target: i32,
        mask: u32,
        handler: u32,
        common: u32,
    ) -> i32 {
        let uid = self.new_uid();
        self.thread_events
            .register(uid, target, mask, handler, common);
        uid
    }

    /// Whether `thid` is a thread of the title's: one it created (`sceKernelCreateThread`),
    /// or the initial thread ([`MAIN_THID`], which has no create record). The host's own
    /// delivery threads (a kernel callback's, a display-queue callback's) have neither.
    pub fn thread_exists(&self, thid: i32) -> bool {
        thid == MAIN_THID || self.threads.iter().any(|t| t.uid == thid)
    }

    /// Thread `thid` is starting one of its entries; `last` when it is its final one. Only
    /// the initial thread has more than one (each module's `module_start`, the executable's
    /// last), and only its final entry's end is the thread's end.
    #[inline]
    pub fn note_thread_entry(&mut self, thid: i32, last: bool) {
        self.thread_events.note_entry(thid, last)
    }

    /// Drop a registration; false when `uid` names none.
    #[inline]
    pub fn unregister_thread_event_handler(&mut self, uid: i32) -> bool {
        self.thread_events.unregister(uid)
    }

    /// Whether this dispatch is a REPLAY of a host call that already asked for a guest call
    /// (see [`Self::call_guest`]) - read before the handler makes any guest call of its own.
    #[inline]
    pub fn guest_call_replaying(&self) -> bool {
        self.guest_calls.replaying()
    }

    /// Claim thread `thid`'s END for its EXIT handlers: true the first time, false after.
    ///
    /// The handlers run once per thread whichever way it ends - `sceKernelExitThread`, or
    /// returning from its entry - and a handler that itself calls `sceKernelExitThread`
    /// must end the thread rather than run the handlers again. False, claiming nothing, while
    /// the initial thread is in an early `module_start` (see [`Self::note_thread_entry`]).
    #[inline]
    pub fn claim_thread_exit_events(&mut self, thid: i32) -> bool {
        self.thread_events.claim_exit(thid)
    }

    /// The EXIT handler calls owed thread `thid` ending with `code`, in registration order:
    /// `handler(SCE_KERNEL_THREAD_EVENT_TYPE_EXIT, thid, code, common)`, to be run ON that
    /// thread (a C library's handler frees the ending thread's own TLS).
    ///
    /// Only the title's threads ([`Self::thread_exists`]) get them: the host's own delivery
    /// threads (a kernel callback's, a display-queue callback's) are not threads it knows about. Pure,
    /// so a host call can rebuild the list on each replay.
    ///
    /// The handler's id rides as a fifth (stack) argument the handler never reads, so two
    /// registrations of the same function with the same `common` are still two distinct
    /// calls - [`Self::call_guest`] matches a replayed call by its arguments.
    pub fn thread_exit_event_calls(&self, thid: i32, code: u32) -> Vec<GuestCall> {
        if !self.thread_exists(thid) || self.kcallbacks.is_delivery_thread(thid) {
            return Vec::new();
        }
        self.thread_events.exit_calls(thid, code)
    }

    /// Record a finished thread's return value (set by the engine host after the
    /// thread ends). In preemptive mode this also wakes any thread parked joining
    /// it (`sceKernelWaitThreadEnd`).
    pub fn set_thread_exit(&mut self, thid: i32, code: u32) {
        let flip = self.flip_pacer.flip_count();
        if let Some(t) = self.threads.iter_mut().find(|t| t.uid == thid) {
            // A thread ENDING is reported (`RUST_LOG=vitaslop::thread=debug`) for the
            // same reason its creation is: a title whose game logic has quietly gone
            // away looks exactly like one that is running and drawing, because the
            // render thread keeps resubmitting the last command buffer. The only
            // record of the difference is this line and the frame it happened on.
            tracing::debug!(
                target: "vitaslop::thread",
                thid,
                name = %t.name,
                code = format_args!("{code:#x}"),
                flip,
                "threadEnded"
            );
            t.exit_code = Some(code);
        }
        // A callback's delivery thread ended: a NON-ZERO return deletes the callback, which
        // is the kernel's contract for a one-shot handler.
        self.kcallbacks.delivery_ended(thid, code);
        if self.preemptive {
            let mut i = 0;
            while i < self.join_waiters.len() {
                if self.join_waiters[i].1 == thid {
                    let (waiter, _, stat) = self.join_waiters.remove(i);
                    // Deliver the exit code to the joiner's `stat` out-parameter now
                    // that it is known; the wait handler cannot write it at wake time.
                    tracing::debug!(
                        target: "vitaslop::exit",
                        target_thid = format_args!("{thid:#x}"),
                        code = format_args!("{code:#x}"),
                        waking = format_args!("{waiter:#x}"),
                        stat = format_args!("{stat:#x}"),
                        "join delivered"
                    );
                    if stat != 0 {
                        self.pending_stat_writes.push((stat, code));
                    }
                    push_wake!(self, waiter);
                } else {
                    i += 1;
                }
            }
            // A finished display callback unblocks the next queued one (the
            // display queue runs its callbacks serially, in submission order).
            if self.display_callbacks.finished(thid) {
                self.pump_display_callback();
                for waiter in self.display_callbacks.take_idle_waiters() {
                    push_wake!(self, waiter);
                }
            }
            // ...and a finished NGS callback starts the next one, in event order.
            if self.ngs_callbacks.finished(thid) {
                self.pump_ngs_callback();
            }
        }
    }

    /// Park the current thread joining `target`, unless it has already finished.
    /// Returns true if `target` is already done (the caller continues) or false if
    /// the caller was parked (return [`SvcOutcome::Block`]). `stat` is the joiner's
    /// `int *` out-parameter (0 = NULL), written with the target's exit code when the
    /// join completes (at wake time, via [`take_stat_writes`](Self::take_stat_writes)).
    pub fn join_block(&mut self, target: i32, stat: u32) -> bool {
        if self.thread_finished(target) {
            return true;
        }
        // >>> A JOIN ON A THREAD THAT CANNOT END IS A PERMANENT PARK, AND IT IS SILENT.
        //
        // The waiter list is drained only by `thread_exited`, so a target that no thread
        // will ever exit under - the caller itself, or an id that names no thread at all -
        // parks the caller for the rest of the process. Nothing reports it: the scheduler's
        // deadlock check does not fire while any OTHER thread is still runnable (a sound
        // thread on its own timer is enough), so the run does not fail, it just stops
        // producing frames. MEASURED on one title: the main thread parked joining thid 0
        // and the run spun for forty minutes with no output but sound calls.
        //
        // Both shapes are refused rather than parked on, because the kernel refuses them
        // too - a thread id it does not know is an error return, not a wait. The caller
        // continues with the exit code 0 that `wait_thread_end` writes, which is the same
        // thing it would see for a thread that had already finished.
        if target == self.current {
            tracing::warn!(
                target: "vitaslop::sched",
                thid = format_args!("{target:#x}"),
                "sceKernelWaitThreadEnd: a thread asked to JOIN ITSELF. That can never complete, so it is refused rather than parked on. The id the guest passed is almost certainly not the one it meant."
            );
            return true;
        }
        // >>> A DORMANT THREAD HAS ALREADY ENDED, as far as a join is concerned. A thread the
        // guest created and never STARTED is in `SCE_THREAD_DORMANT`, which is the same state a
        // finished one returns to, and the kernel's wait completes immediately for it: there is
        // no run to wait for. Parking instead is the third shape of the same silent permanent
        // park the two arms below refuse, and it has the same symptom - the run keeps its
        // clock and its sound and simply stops producing frames.
        //
        // MEASURED on a baseball title loading a game: `main` joins `KineCore_0`, a worker its
        // job system creates ahead of the load and starts only when work arrives. The join
        // never completed, a sleep loop on another thread then burned the round budget - 28.5
        // MILLION delay jumps in one frame - and the run ended on the live-lock backstop with
        // the loading spinner still on screen.
        if let Some(t) = self.threads.iter().find(|t| t.uid == target)
            && !t.started
        {
            tracing::debug!(
                target: "vitaslop::sched",
                thid = format_args!("{target:#x}"),
                "sceKernelWaitThreadEnd on a DORMANT thread (created, never started) - it has no run to end, so the wait completes at once"
            );
            return true;
        }
        if !self.threads.iter().any(|t| t.uid == target) && target != MAIN_THID {
            tracing::warn!(
                target: "vitaslop::sched",
                thid = format_args!("{target:#x}"),
                "sceKernelWaitThreadEnd: JOIN ON AN UNKNOWN THREAD ID - no thread with this id was ever created, so nothing can ever end it. Refused rather than parked on. If the id looks plausible, the create that should have recorded it is the bug."
            );
            return true;
        }
        self.join_waiters.push((self.current, target, stat));
        false
    }

    /// Take the queued joiner `stat` writes (`(stat_ptr, exit_code)`) so the scheduler
    /// can apply them to guest memory before the woken joiners resume. Drained after
    /// each dispatch alongside spawns and wakes.
    pub fn take_stat_writes(&mut self) -> Vec<(u32, u32)> {
        std::mem::take(&mut self.pending_stat_writes)
    }
}