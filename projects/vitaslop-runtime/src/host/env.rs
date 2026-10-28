use super::*;

/// The Vita host environment: the NID import table plus per-run state. Implements
/// [`ImportDispatch`] so an engine host (native wasmtime or the browser) can
/// route the `env.import` trap here.
pub struct VitaEnv {
    /// (library_nid, func_nid) per dense import index, in loader order.
    pub(super) imports: Vec<(u32, u32)>,
    pub state: VitaState,
    /// The cross-engine host-call digest for the frame in progress - see [`NidDigest`].
    /// Inert (and untouched) unless `VITASLOP_NID_DIGEST` asked for it.
    pub(super) nid_digest: NidDigest,
}

impl VitaEnv {
    /// The presenting worker's half of [`VitaState::resolve_at_flip`]: read the geometry the
    /// last flip left, through `mem`, before the flipped frame's scenes are taken.
    pub fn resolve_flipped_geometry(&mut self, mem: &mut dyn GuestMemory, base: u32) {
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let ctx = GuestCtx::new(&mut regs, &mut vfp, mem, base);
        self.state.resolve_flip_boundary(&ctx);
    }

    /// Take the flip's resolve job, if one is owed - see [`VitaState::take_flip_resolve_job`].
    /// The caller computes it with [`compute_flip_resolve_job`] WITHOUT this host locked, then
    /// calls [`Self::apply_posted_resolve`].
    pub fn take_flip_resolve_job(&mut self) -> Option<FlipResolveJob> {
        self.state
            .take_flip_resolve_job()
            .map(|(job, post, snaps)| FlipResolveJob { job, post, snaps })
    }

    /// See [`VitaState::apply_posted_resolve`]. `mem`/`base` are guest memory, which a
    /// deferred texture set the resolver could not prove is built from.
    pub fn apply_posted_resolve(&mut self, mem: &mut dyn GuestMemory, base: u32) {
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let ctx = GuestCtx::new(&mut regs, &mut vfp, mem, base);
        self.state.apply_posted_resolve(&ctx);
    }

    /// The hand-off a presenting worker waits on (see [`AsyncResolve::wait_idle`]) before it
    /// applies the posted jobs.
    pub fn async_resolve_handle(&self) -> Arc<AsyncResolve> {
        self.state.async_resolve.clone()
    }

    /// Handles for a resolver worker's [`run_resolver`].
    pub fn resolver_handles(&self) -> ResolverHandles {
        ResolverHandles {
            post: self.state.async_resolve.clone(),
            snaps: self.state.texture_snapshots.clone(),
        }
    }

    /// Build an environment for a module whose imports are `(library_nid,
    /// func_nid)` in dense index order, over guest memory `[base, base+mem_bytes)`.
    pub fn new(
        imports: Vec<(u32, u32)>,
        base: u32,
        mem_bytes: u32,
        world: Box<dyn World + Send>,
    ) -> Self {
        VitaEnv {
            imports,
            state: VitaState::new(base, mem_bytes, world),
            nid_digest: NidDigest::default(),
        }
    }

    /// Convenience constructor with the default deterministic world.
    pub fn with_default_world(imports: Vec<(u32, u32)>, base: u32, mem_bytes: u32) -> Self {
        VitaEnv::new(
            imports,
            base,
            mem_bytes,
            Box::new(DeterministicWorld::default()),
        )
    }

    /// The `(library_nid, func_nid)` pair a dense import index dispatches to, or
    /// `None` past the end of the table. A profiler counts host calls by the index
    /// the emitted wasm passes, so this is what turns those counts back into names.
    pub fn import_at(&self, index: u32) -> Option<(u32, u32)> {
        self.imports.get(index as usize).copied()
    }

    /// Per dense import index, whether that call's handler reads or writes the VFP argument
    /// file ([`crate::vita::float_abi`]) - what a frontend that marshals only the core
    /// registers must widen its read for. Indexed like [`Self::import_at`].
    pub fn import_float_abi(&self) -> Vec<bool> {
        self.imports.iter().map(|&(_, func_nid)| crate::vita::float_abi(func_nid)).collect()
    }

    /// For a call [`vita::smp_forward_per_call`] names: whether THIS call touches no
    /// JavaScript and may run on the guest worker that made it. Ask with the lock held and
    /// dispatch under the same hold. False for anything else, so a caller that asks about the
    /// wrong call forwards it - the safe direction.
    pub fn smp_call_is_self_contained(
        &self,
        index: u32,
        regs: &[u32; REG_COUNT],
        mem: &dyn GuestMemory,
        base: u32,
    ) -> bool {
        match self.import_at(index) {
            Some((_, nid)) if vita::smp_forward_per_call(nid) => {
                vita::audiodec::decode_is_self_contained(&self.state, mem, base, regs[0])
            }
            _ => false,
        }
    }
}

/// The seam an engine host implements to deliver an ARM `svc` trap to a host
/// convention (e.g. the Linux-EABI the ARM conformance corpus uses). The parallel
/// of [`ImportDispatch`] for the `env.svc` import instead of `env.import`. The
/// host passes the `svc` immediate, the register file, and rebased guest memory;
/// any register the handler changes is written back.
pub trait SvcDispatch {
    fn svc(
        &mut self,
        imm: u32,
        regs: &mut [u32; REG_COUNT],
        mem: &mut dyn GuestMemory,
        base: u32,
    ) -> SvcOutcome;
}

/// Drive a `svc` handler behind a shared handle, so a caller can keep a clone to
/// read accumulated state (e.g. captured output) back after the run.
impl<T: SvcDispatch> SvcDispatch for std::rc::Rc<std::cell::RefCell<T>> {
    fn svc(
        &mut self,
        imm: u32,
        regs: &mut [u32; REG_COUNT],
        mem: &mut dyn GuestMemory,
        base: u32,
    ) -> SvcOutcome {
        self.borrow_mut().svc(imm, regs, mem, base)
    }
}

// --- the CROSS-ENGINE host-call digest -------------------------------------------
//
// >>> WHY: "the browser diverges from the desktop" is the hardest question this project asks,
// > > > and until now nothing could answer it directly. Both engines run the SAME translated
// > > > module against the SAME host crate, so if they disagree the disagreement entered
// > > > through a host call - a file read that returned different bytes, a clock that advanced
// > > > differently, a thread that was scheduled in another order. A picture cannot say which,
// > > > a block trace covers one function, and a whole-run signature says only "different".
//
// `VITASLOP_NID_DIGEST=1` folds every host call the guest makes - its NID, the calling thread
// and its four argument registers - into a per-frame digest, printed at each frame boundary.
// Two runs' digest columns line up frame by frame, so the FIRST frame that differs is the
// frame the divergence entered, and it is one `diff` away on any pair of engines.
//
// `VITASLOP_NID_DIGEST=<from>-<to>` additionally lists every call in that frame window, with
// its arguments, which names the single call once the window is one frame wide.
//
// The digest is FNV-1a over little-endian words: no dependency, no allocation, and the same
// arithmetic on every target, which a hasher chosen by the platform's RandomState would not be.
pub(super) static NID_DIGEST_SPEC: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();

/// `VITASLOP_THREAD_TRAILS_AT=<frame>` - see its use in the host-call dispatch.
pub(super) fn thread_trails_at() -> Option<u64> {
    static AT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *AT.get_or_init(|| {
        crate::knobs::var("VITASLOP_THREAD_TRAILS_AT")
            .ok()
            .and_then(|v| v.trim().parse().ok())
    })
}

/// `VITASLOP_THREAD_TRAILS_NID=<hex>` - see its use in the host-call dispatch.
pub(super) fn thread_trails_nid() -> Option<u32> {
    static NID: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *NID.get_or_init(|| {
        crate::knobs::var("VITASLOP_THREAD_TRAILS_NID")
            .ok()
            .and_then(|v| u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok())
    })
}

/// Whether this trail key (a thread, or a thread+chain fold) is new. Capped, so a scan that
/// keeps picking up stale stack slots cannot flood the log.
pub(super) fn thread_trail_first(key: u64) -> bool {
    static SEEN: std::sync::Mutex<Vec<u64>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.contains(&key) || seen.len() >= 400 {
        return false;
    }
    seen.push(key);
    true
}

/// Is the digest on, and over which frame window should each call be LISTED?
/// `None` = off. `Some((0, 0))` = digest only (the `=1` spelling).
pub(super) fn nid_digest_spec() -> Option<(u64, u64)> {
    *NID_DIGEST_SPEC.get_or_init(|| {
        let raw = crate::knobs::var("VITASLOP_NID_DIGEST").ok()?;
        let raw = raw.trim();
        if raw.is_empty() || raw == "0" {
            return None;
        }
        if raw == "1" {
            return Some((0, 0));
        }
        let (a, b) = raw.split_once('-')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    })
}

/// The running host-call digest, per thread and for the run as a whole.
///
/// # Why PER THREAD and not one stream
/// The obvious digest - fold every call in the order it happens - is useless across engines,
/// and would have been worse than useless: two engines preempt at different points, so the
/// INTERLEAVING of calls from different threads legitimately differs on the first busy frame.
/// A digest that folds the interleaving in reports a divergence in every run, including two
/// correct ones, and an instrument that always fires cannot be used to find anything.
///
/// Each thread's own call sequence is the invariant part: given the same host answers, a
/// guest thread makes the same calls with the same arguments in the same order whatever else
/// is running. So the digest is per thread and CUMULATIVE, and the frame line is a snapshot
/// of every thread's running value. The first frame at which a given thread's column differs
/// is the frame that thread's behaviour diverged, and the `nid call` listing for that one
/// frame names the call.
#[derive(Default)]
pub(crate) struct NidDigest {
    /// `thid -> (cumulative FNV-1a hash, cumulative call count)`, ordered so two runs'
    /// lines are directly comparable as text.
    threads: std::collections::BTreeMap<i32, (u64, u64)>,
    /// Calls folded this frame, for the frame line's own count.
    frame_calls: u64,
}

impl NidDigest {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    fn fold(hash: &mut u64, word: u32) {
        for b in word.to_le_bytes() {
            *hash ^= u64::from(b);
            *hash = hash.wrapping_mul(Self::FNV_PRIME);
        }
    }

    /// Fold one host call into its thread's digest, and list it when the frame is inside the
    /// requested window.
    fn record(&mut self, frame: u64, func_nid: u32, thid: i32, regs: &[u32; REG_COUNT]) {
        let entry = self.threads.entry(thid).or_insert((Self::FNV_OFFSET, 0));
        entry.1 += 1;
        Self::fold(&mut entry.0, func_nid);
        for r in &regs[..4] {
            Self::fold(&mut entry.0, *r);
        }
        self.frame_calls += 1;
        let n = entry.1;
        if let Some((from, to)) = nid_digest_spec()
            && (from, to) != (0, 0)
            && frame >= from
            && frame <= to
        {
            tracing::info!(
                target: "vitaslop::status",
                "nid call: frame={frame} thid={thid:#x} #{n} nid={func_nid:#010x} {} r0={:#010x} r1={:#010x} r2={:#010x} r3={:#010x} lr={:#010x} sp={:#010x}",
                crate::nid::name(func_nid),
                regs[0], regs[1], regs[2], regs[3], regs[14], regs[13],
            );
        }
    }

    /// Print the per-thread snapshot at a frame boundary. A frame in which no thread called
    /// anything prints nothing, so an idle stretch adds no lines for two runs to disagree
    /// about for no reason.
    fn end_frame(&mut self, frame: u64) {
        if self.frame_calls == 0 {
            return;
        }
        let cols: Vec<String> = self
            .threads
            .iter()
            .map(|(thid, (hash, calls))| format!("t{thid:#x}={hash:016x}/{calls}"))
            .collect();
        tracing::info!(
            target: "vitaslop::status",
            "nid digest: frame={frame} calls={} {}",
            self.frame_calls,
            cols.join(" "),
        );
        self.frame_calls = 0;
    }
}

/// The seam an engine host implements to deliver a NID import trap to the Vita
/// host. Engine-agnostic: the host passes the raw register file and rebased guest
/// memory, and any register the handler changes is written back.
pub trait ImportDispatch {
    /// The priority the MAIN thread starts at - what the title's `SceProcessParam` asks for
    /// (`VitaState::main_thread_priority`); the default for a host that has no title.
    fn main_thread_priority(&self) -> i32 {
        DEFAULT_THREAD_PRIORITY
    }

    fn dispatch(
        &mut self,
        index: u32,
        regs: &mut [u32; REG_COUNT],
        vfp: &mut [u32; VFP_ARG_COUNT],
        mem: &mut dyn GuestMemory,
        base: u32,
    ) -> SvcOutcome;

    /// Take a synchronous guest re-entry the last dispatch raised (a thread
    /// start), if any. The engine host calls this after each dispatch and runs
    /// the returned entry, then reports the result via [`set_thread_exit`]. A host
    /// with no engine to re-enter (default) never has one.
    ///
    /// [`set_thread_exit`]: ImportDispatch::set_thread_exit
    fn take_reentry(&mut self) -> Option<Reentry> {
        None
    }

    /// Record a re-entered thread's return value.
    fn set_thread_exit(&mut self, _thid: i32, _code: u32) {}

    /// The guest call thread `thid`'s last dispatch returned [`SvcOutcome::CallGuest`] for.
    /// The engine runs it (see that variant), then reports through
    /// [`guest_call_returned`](Self::guest_call_returned). A host that never asks for one
    /// (default) has none.
    fn take_guest_call(&mut self, _thid: i32) -> Option<GuestCall> {
        None
    }

    /// Thread `thid` RETURNED from its entry with `code`: the guest calls to run on it,
    /// in order, before it counts as ended (its EXIT event handlers - see
    /// [`VitaState::thread_exit_event_calls`]). Claims the thread's end, so a second ask
    /// returns nothing. A host with no handlers (default) has none.
    fn thread_return_calls(&mut self, _thid: i32, _code: u32) -> Vec<GuestCall> {
        Vec::new()
    }

    /// Thread `thid` is starting one of its entries, `last` when it is its final one (see
    /// [`VitaState::note_thread_entry`]). Default no-op.
    fn thread_entry_starting(&mut self, _thid: i32, _last: bool) {}

    /// Record the r0/r1 a guest call left, before the engine re-dispatches the host call
    /// that asked for it.
    fn guest_call_returned(
        &mut self,
        _thid: i32,
        _r0: u32,
        _r1: u32,
        _mem: &dyn GuestMemory,
        _base: u32,
    ) {
    }

    /// The thread the host takes to be running - for an engine that keeps no thread ids of
    /// its own (the run-to-completion `Vm`) and has to name one to the calls above.
    fn running_thread(&self) -> i32 {
        0
    }

    // --- preemptive scheduler hooks (default no-ops) -----------------------
    //
    // The single-worker `Vm`/`Scheduler` never call these; only the preemptive
    // `ThreadedScheduler` does. They let a shared host and the scheduler agree on
    // which thread is running, which new threads to spawn as their own fibers, and
    // which parked threads a signal just made runnable - the state the scheduler
    // cannot see (it lives in the host's sync objects) and the host cannot act on
    // (only the scheduler owns the fibers).

    /// Tell the host which guest thread is running, so a blocking primitive knows whom to
    /// park and a wake knows whom to release.
    ///
    /// The scheduler sets this when it PICKS a thread, before anything of that thread's is
    /// refreshed or resumed, and again before each dispatch. The first of those is the one
    /// that matters and it was missing: the current thread is mirrored into guest memory
    /// for the inlined lightweight-mutex take ([`crate::vita::mirror::SLOT_CURRENT_THREAD`]),
    /// so a build that only set it at dispatch time would refresh the block with the
    /// PREVIOUS thread's id and let a resumed thread take mutexes in its name until it
    /// happened to make a host call.
    fn set_current_thread(&mut self, _thid: i32) {}

    /// How many OTHER threads are RUNNABLE at the pick that is about to resume this one.
    ///
    /// The host cannot see the scheduler's runnable set, and one question needs it: whether a
    /// `sceKernelDelayThread(0)` - "give someone else the CPU" - has anyone to give it to. See
    /// `VitaState::yield_would_repick_this_thread`.
    fn note_runnable_others(&mut self, _n: usize) {}

    /// Settle any host state that was decided where no guest memory was reachable, and
    /// report how many items were settled.
    ///
    /// Called by the scheduler at the top of every drain - after a resume and after a clock
    /// advance, and always before a woken thread can run. Today this is the
    /// lightweight-mutex handoff owed to a cond wait that timed out; see
    /// [`VitaState::resolve_deferred_lwmutex`] for why that one cannot be a plain queued
    /// write. The default settles nothing, which is correct for a host with no such state.
    fn resolve_deferred(&mut self, _words: &mut dyn GuestWords) -> usize {
        0
    }

    /// The thread pointer (`TPIDRURO`) for thread `thid`: the base of its private
    /// thread-local-storage block, allocated on first request. The engine reads it
    /// when it instantiates the thread and seeds the guest `tp` register, so a
    /// `MRC p15,0,Rt,c13,c0,3` reads a per-thread pointer. Also returns the TLS init
    /// image `(source, len)` to copy into the block's `.tdata` head (`len == 0` for a
    /// pure-`.tbss` template, and for a block that is already live - a fiber's backing
    /// thread shares its runner's). Default `(0, 0, 0)`: a host with no TLS model.
    fn thread_tls_base(&mut self, _thid: i32) -> (u32, u32, u32) {
        (0, 0, 0)
    }

    /// Take the threads the last dispatch asked to *start* (each becomes its own
    /// fiber sharing the guest address space), if any. The preemptive counterpart
    /// of [`take_reentry`](Self::take_reentry): instead of running the entry
    /// synchronously to completion, the scheduler creates a concurrent thread.
    fn take_spawns(&mut self) -> Vec<Reentry> {
        Vec::new()
    }

    /// Take the parked threads the last dispatch just made runnable (a signal, an
    /// unlock, an event-flag set that satisfied a waiter). The scheduler drains
    /// this after each dispatch and moves those threads back onto the run queue;
    /// each resumes inside its blocking call, which then returns success.
    fn take_wakes(&mut self) -> Vec<i32> {
        Vec::new()
    }

    /// Take the PARKED threads the runtime has ended (see
    /// [`VitaState::take_terminations`]). The scheduler drains this at every point it drains
    /// [`take_wakes`](Self::take_wakes) and DROPS each thread without resuming it, exactly as
    /// it would one that ended with `ThreadExit`: its engine state (stack, instance, future)
    /// is released and it is never picked again. The host has already recorded the end, so
    /// the scheduler must not report it back through `set_thread_exit`.
    fn take_terminations(&mut self) -> Vec<i32> {
        Vec::new()
    }

    /// Take the guest memory writes (`(addr, value)`) the last dispatch queued for
    /// woken joiners - the exit code owed to a `sceKernelWaitThreadEnd` `stat`
    /// out-parameter. The scheduler applies these to guest memory before the woken
    /// joiner resumes (the wait handler cannot write them at wake time).
    fn take_stat_writes(&mut self) -> Vec<(u32, u32)> {
        Vec::new()
    }

    /// Human-readable state of every thread and sync primitive, for a scheduler that
    /// wants to report a stall from inside its own loop.
    ///
    /// It lives on the trait rather than being reached through a concrete host because
    /// both schedulers are generic over the host, and a stall that happens on one engine
    /// and not the other can only be diagnosed by comparing the two dumps - which
    /// requires both to be able to produce one. The default is empty for a host with no
    /// sync state to report.
    fn sync_dump(&self) -> String {
        String::new()
    }

    /// [`sync_dump`](Self::sync_dump) with a reader for GUEST memory.
    ///
    /// A lightweight mutex keeps its owner and recursion count in the guest's own work area
    /// (`vita::lwwork`), so the plain dump can say who is PARKED on one but not who HOLDS it
    /// - and on a stall that is the whole question. The caller supplies the reader because
    /// only it can reach linear memory from outside a host call.
    fn sync_dump_with(&self, _read_word: &dyn Fn(u32) -> u32) -> String {
        self.sync_dump()
    }

    /// Whether the host has a WAIT RECORD for `thid` - it is parked in some waiter queue.
    ///
    /// # Why a scheduler needs to ask
    /// A thread the SCHEDULER has marked blocked but that the HOST is not waiting on cannot
    /// be woken by anything: no signal, no timeout, no I/O completion names it. That is a
    /// LOST WAKEUP - a bug in the emulator - and it is not the same thing as a deadlock,
    /// where the waits are real and simply cannot be satisfied. Reported as one for years:
    /// a retail headless run stops at frame 1 with eight "blocked" threads, and the eighth
    /// is its render thread, parked by nothing at all.
    ///
    /// The default is `true` (assume a record exists) so a host that cannot answer never
    /// turns a real deadlock into a spurious lost-wakeup report.
    fn thread_has_wait_record(&self, _thid: i32) -> bool {
        true
    }

    /// What `thid` is waiting ON, in one short phrase ("RUNNABLE" if nothing).
    ///
    /// The whole-machine [`sync_dump`](Self::sync_dump) answers this for every thread at
    /// one instant; this answers it for ONE thread at the instant it parks, which is what
    /// a scheduling TIMELINE needs. A timeline of picks and blocks without the reason says
    /// only that a thread stopped running - and "it was descheduled" and "it is waiting on
    /// a semaphore the other thread signals once a frame" are the two readings such a
    /// timeline exists to separate.
    fn thread_wait_reason(&self, _thid: i32) -> String {
        String::new()
    }

    /// The earliest pending timed-wait deadline (virtual microseconds), if a thread
    /// is parked on a timed wait. When no thread is runnable, the scheduler jumps the
    /// clock to this instead of declaring a deadlock (a busy loop's timed wait).
    fn earliest_deadline(&self) -> Option<u64> {
        None
    }

    /// Advance the virtual clock to `to_us`, waking any timed waits that expire; the
    /// woken thread ids then surface through [`take_wakes`](Self::take_wakes).
    fn advance_time_to(&mut self, _to_us: u64) {}

    /// The virtual clock now, in microseconds. The scheduler compares it against
    /// [`earliest_deadline`](Self::earliest_deadline) to tell a wait that will buy time
    /// from one that will not (a zero-length delay re-armed every iteration).
    fn clock_us(&self) -> u64 {
        0
    }

    /// Nothing is runnable and no timed wait completes sooner: complete the earliest
    /// outstanding modelled storage transfer, waking its reader. Returns whether anything
    /// was released. See [`VitaState::release_earliest_io`].
    fn release_earliest_io(&mut self) -> bool {
        false
    }

    /// Storage time the earliest outstanding transfer still owes, for the idle path to
    /// compare against the next timed wait. See [`VitaState::earliest_io_remaining_us`].
    fn earliest_io_remaining_us(&self) -> Option<u64> {
        None
    }

    /// Credit the modelled storage device for an idle interval the scheduler just jumped
    /// the game clock over. See [`VitaState::charge_io_idle`].
    fn charge_io_idle(&mut self, _us: u64) {}

    /// A thread just suspended having burned `fuel` units of guest execution. Lets the
    /// host charge clocks that track executed work rather than rendered frames - the game
    /// clock and the modelled storage clock. The default ignores it.
    ///
    /// This is called for EVERY suspend, not only a preemption: a thread that blocks or
    /// flips did guest work on the way there and must be billed for it, and a thread that
    /// yielded immediately must not be billed for a quantum it never used.
    ///
    /// `runnable` is how many threads were ready to run at that moment, INCLUDING the
    /// one that just stopped. The scheduler runs one at a time; the device does not, so
    /// this is what lets the host divide the WALL time among the cores that would have
    /// been executing it in parallel. See [`VitaEnv::on_guest_work`].
    fn on_guest_work(&mut self, _runnable: usize, _fuel: u64, _retired: Option<u64>) {}

    /// The `r0` value owed to thread `thid` as it resumes from a block (a timed wait
    /// that expired, returning `SCE_KERNEL_ERROR_WAIT_TIMEOUT`), if any. The engine
    /// calls this at the point the woken thread resumes and, on `Some`, overwrites its
    /// `r0`; `None` (a signal wake) leaves the pre-park return value (0) in place.
    fn take_resume_code(&mut self, _thid: i32) -> Option<u32> {
        None
    }

    /// A display frame just flipped (`frame` flips so far). Lets a frame-keyed input
    /// source - a scripted input timeline - advance in lockstep with the render loop.
    /// The default ignores it.
    fn on_frame_boundary(&mut self, _frame: u64) {}

    /// Write the current value of every HOST MIRROR slot through `write`, which stores
    /// one word at a slot index. Returns how many slots were written.
    ///
    /// Called by the scheduler immediately before guest code resumes, which is what
    /// makes an inlined read of the block exactly equal to the host call it replaces -
    /// see [`crate::vita::mirror`] for the rule about what may live there. The default
    /// writes nothing and returns 0, which is correct for a host with no mirrored
    /// values and is DETECTED (not tolerated) for one that needs them.
    fn refresh_mirror(&mut self, _write: &mut dyn FnMut(u32, u32)) -> usize {
        0
    }

    /// The GUEST address of the host-mirror block, told to the host once the scheduler knows
    /// it. Only a host that keeps shared STATE behind the mirrored slots needs it - the
    /// kernel mutex table does ([`crate::vita::kmutex`]) - and the default ignores it.
    fn set_mirror_base(&mut self, _addr: u32) {}
}

impl ImportDispatch for VitaEnv {
    fn main_thread_priority(&self) -> i32 {
        self.state.main_thread_priority()
    }

    fn dispatch(
        &mut self,
        index: u32,
        regs: &mut [u32; REG_COUNT],
        vfp: &mut [u32; VFP_ARG_COUNT],
        mem: &mut dyn GuestMemory,
        base: u32,
    ) -> SvcOutcome {
        // The transpiler's software fuel point, not a NID: this thread has used its
        // quantum and is asking to be rescheduled. Intercepted before the import table
        // because it is not IN the import table - it is a reserved selector above every
        // real index (see `vitaslop_transpiler::abi::FUEL_SELECTOR`).
        //
        // Deliberately NOT recorded as a call: it is host bookkeeping the guest never
        // asked for, and folding it into the per-NID histogram would put a synthetic
        // entry at the top of every profile.
        if index == vitaslop_transpiler::abi::FUEL_SELECTOR {
            return SvcOutcome::Reschedule;
        }
        // >>> THE VBLANK SPIN, PARKED. Also not a NID, and not recorded as a call.
        //
        // The emitted `sceDisplayGetVcount` has taken `mirror::SPIN_BUDGET` reads inside one
        // resume. The mirror cannot change while guest code runs, so every one of those reads
        // returned the same word: this thread is in a loop it cannot leave until the clock
        // moves, and while it is runnable the scheduler will not move it - the loop's own
        // fuel is what drags the clock to the next vblank. So do what the loop is asking for
        // and park it there, which is the same wait `sceDisplayWaitVblankStart` performs.
        //
        // The budget needs no reset here: the block is refreshed before EVERY resume, which
        // is the same contract that makes the mirror readable at all, so the thread comes
        // back with a full budget by construction. A host that broke that contract would
        // leave the counter running negative and simply stop parking - the behaviour this
        // guard replaced, which is the right way for it to fail.
        if index == vitaslop_transpiler::abi::VBLANK_PARK_SELECTOR {
            // Same guard the real vblank wait carries: in the SINGLE-THREAD model there is
            // nothing to yield to and the clock is host-driven, so a park would block on a
            // wake nothing can deliver. There the spin is left exactly as it was.
            if !self.state.is_preemptive() {
                return SvcOutcome::Continue;
            }
            self.state.note_vblank_spin_park();
            self.state.vblank_park(1, vita::display::VBLANK_US);
            return SvcOutcome::Block;
        }
        let (library_nid, func_nid) = self.imports.get(index as usize).copied().unwrap_or((0, 0));
        self.state.capture.record_call(func_nid, self.state.current);
        // A playbook's cues and its pad-poll tick - see `crate::observer`.
        if crate::observer::enabled() {
            let frame = self.state.cur_frame();
            let guest_us = self.state.guest_mono_us();
            crate::observer::notify(&crate::observer::ObservedCall {
                nid: func_nid,
                regs: &regs[..],
                mem: &MemRead { mem: &*mem, base },
                frame,
                guest_us,
            });
        }
        // The cross-engine divergence instrument, off unless asked for. Placed HERE - after
        // the selector resolves to a NID and before the handler runs - because that is the
        // one point both engines share and the arguments are still the guest's own.
        if nid_digest_spec().is_some() {
            let frame = self.state.cur_frame();
            self.nid_digest
                .record(frame, func_nid, self.state.current, regs);
        }
        if crate::call_table::spec().is_some() {
            crate::call_table::record(
                self.state.cur_frame(),
                func_nid,
                self.state.current,
                regs[14],
                &regs[..13],
            );
        }
        // Diagnostic (`RUST_LOG=vitaslop::display=trace`): EVERY host call the guest's
        // display-queue callback makes, in order. GXM runs that callback once per queued
        // entry and the queue is bounded by `displayQueueMaxPendingCount`, so whatever the
        // callback WAITS on is the ceiling on a title's flip rate - and nothing else in the
        // engine names those calls.
        if self.state.display_callback_thread() == Some(self.state.current) {
            tracing::trace!(
                target: "vitaslop::display",
                thid = self.state.current,
                nid = format_args!("{func_nid:#010x}"),
                name = crate::nid::name(func_nid),
                us = self.state.now_us(),
                "display callback host call"
            );
        }
        // `VITASLOP_THREAD_TRAILS_AT=<frame>`: from that frame on, each thread's FIRST host call
        // prints the plausible saved return addresses on its stack - where in the title's own
        // code every thread is sitting. A stall in which every thread is idle or polling makes
        // no call that differs from a run that does not stall; the trails are what differ.
        // `VITASLOP_THREAD_TRAILS_NID=<hex>` narrows it to calls of that one function, still once
        // per thread - the chain at the moment a thread does the thing a stalled run never does.
        if let Some(at) = thread_trails_at()
            && self.state.cur_frame() >= at
            && thread_trails_nid().is_none_or(|n| n == func_nid)
        {
            let ctx = GuestCtx::new(regs, vfp, mem, base);
            let trail = vita::guest_return_trail(&ctx, 512);
            // Per thread without a NID; per DISTINCT CHAIN with one - every different reason a
            // thread makes that call, which is what tells two runs' paths apart.
            let key = if thread_trails_nid().is_some() {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(&(self.state.current, &trail), &mut h);
                std::hash::Hasher::finish(&h)
            } else {
                self.state.current as u64
            };
            if thread_trail_first(key) {
                tracing::warn!(
                    target: "vitaslop::warning",
                    thread = format_args!("{:#x}", self.state.current).to_string(),
                    frame = self.state.cur_frame(),
                    call = crate::nid::name(func_nid),
                    lr = format_args!("{:#010x}", ctx.regs[14]).to_string(),
                    pc_trail = %trail,
                    "THREAD TRAIL"
                );
            }
        }
        self.state.guest_call_begin();
        // Any handler may stop to wait on a guest call, and must then leave the register
        // file exactly as the guest handed it over - the engine re-dispatches from it. See
        // `VitaState::call_guest`. (A 128-byte copy; the call around it costs far more.)
        let saved = (*regs, *vfp);
        let out = {
            let mut ctx = GuestCtx::new(regs, vfp, mem, base);
            vita::dispatch(library_nid, func_nid, &mut ctx, &mut self.state)
        };
        if self.state.guest_call_end() {
            (*regs, *vfp) = saved;
            return SvcOutcome::CallGuest;
        }
        if crate::call_table::spec().is_some() {
            crate::call_table::record_ret(
                self.state.cur_frame(),
                func_nid,
                self.state.current,
                saved.0[14],
                regs[0],
            );
        }
        out
    }

    fn take_reentry(&mut self) -> Option<Reentry> {
        self.state.take_reentry()
    }

    fn take_guest_call(&mut self, thid: i32) -> Option<GuestCall> {
        self.state.take_guest_call(thid)
    }

    fn thread_return_calls(&mut self, thid: i32, code: u32) -> Vec<GuestCall> {
        if self.state.claim_thread_exit_events(thid) {
            self.state.thread_exit_event_calls(thid, code)
        } else {
            Vec::new()
        }
    }

    fn thread_entry_starting(&mut self, thid: i32, last: bool) {
        self.state.note_thread_entry(thid, last);
    }

    fn guest_call_returned(
        &mut self,
        thid: i32,
        r0: u32,
        r1: u32,
        mem: &dyn GuestMemory,
        base: u32,
    ) {
        self.state.guest_call_returned(thid, r0, r1, mem, base);
    }

    fn running_thread(&self) -> i32 {
        self.state.current_thread()
    }

    fn set_thread_exit(&mut self, thid: i32, code: u32) {
        self.state.set_thread_exit(thid, code);
    }

    fn set_current_thread(&mut self, thid: i32) {
        self.state.set_current(thid);
    }

    fn note_runnable_others(&mut self, n: usize) {
        self.state.note_runnable_others(n);
    }

    fn resolve_deferred(&mut self, words: &mut dyn GuestWords) -> usize {
        self.state.resolve_deferred_lwmutex(words)
    }

    fn thread_tls_base(&mut self, thid: i32) -> (u32, u32, u32) {
        self.state.tls_block_to_instantiate(thid)
    }

    fn take_spawns(&mut self) -> Vec<Reentry> {
        self.state.take_spawns()
    }

    fn take_wakes(&mut self) -> Vec<i32> {
        self.state.take_wakes()
    }

    fn take_terminations(&mut self) -> Vec<i32> {
        self.state.take_terminations()
    }

    fn take_stat_writes(&mut self) -> Vec<(u32, u32)> {
        self.state.take_stat_writes()
    }

    fn sync_dump(&self) -> String {
        self.state.debug_sync_dump()
    }

    fn sync_dump_with(&self, read_word: &dyn Fn(u32) -> u32) -> String {
        self.state.debug_sync_dump_with(Some(read_word))
    }

    fn thread_has_wait_record(&self, thid: i32) -> bool {
        self.state.thread_wait_state(thid) != "RUNNABLE"
    }

    fn thread_wait_reason(&self, thid: i32) -> String {
        self.state.thread_wait_state(thid)
    }

    fn earliest_deadline(&self) -> Option<u64> {
        self.state.earliest_lwcond_deadline()
    }

    fn advance_time_to(&mut self, to_us: u64) {
        self.state.advance_time_to(to_us);
    }

    fn clock_us(&self) -> u64 {
        self.state.now_us()
    }

    fn release_earliest_io(&mut self) -> bool {
        self.state.release_earliest_io()
    }

    fn earliest_io_remaining_us(&self) -> Option<u64> {
        self.state.earliest_io_remaining_us()
    }

    fn charge_io_idle(&mut self, us: u64) {
        self.state.charge_io_idle(us);
    }

    /// # A quantum of guest EXECUTION is not a quantum of WALL time
    /// The scheduler has one baton: it resumes a thread, that thread retires a quantum
    /// of guest code, and the clock is charged for it. The Vita has three CPUs for the
    /// game, so three threads retire their quanta AT ONCE and only ONE quantum of wall
    /// time passes while they do. Charging the full quantum per thread makes the game
    /// clock run at the total rate of every runnable thread, which turns a guest thread
    /// SPINNING - waiting on nothing, retiring instructions - into game time it never
    /// costs on hardware.
    ///
    /// Measured on this title's event load: the load costs ~230 s of GAME clock against
    /// **54 s of wall clock** - the frames themselves run at 2.83 ms each - and `bench`
    /// reports 18 thread resumes a frame, "one per 0 host calls". Threads burning whole
    /// quanta in guest code, waiting on nothing, and the clock billing every one of them
    /// as if it had a machine to itself.
    ///
    /// So the charge is the quantum divided by the cores that would have been busy:
    /// `runnable` of them, capped at [`GUEST_CORES`], and never fewer than one (a lone
    /// runnable thread really does have the machine to itself, and the other two cores
    /// are idle). The STORAGE clock divides by exactly the same figure and for exactly
    /// the same reason - it charges for elapsed wall time too, and the device does not
    /// get three times the bandwidth because three threads are running.
    fn on_guest_work(&mut self, runnable: usize, fuel: u64, retired: Option<u64>) {
        let cores = runnable.clamp(1, guest_cores()) as u64;
        // >>> A QUANTUM OF GUEST WORK, IN THE UNIT THE ENGINE CAN REPORT.
        //
        // GUEST INSTRUCTIONS when the engine carries the emitted per-block counter, which
        // is the whole point: an ARM instruction is the same amount of guest work whatever
        // this transpiler's codegen turns it into, so the emulated console's speed stops
        // moving every time the codegen improves. Fuel (wasm operators) is the fallback for
        // an engine with no such counter, and is exactly the pre-2026-08-16 behaviour.
        let (work, quantum) = match retired {
            Some(r) => (r, QUANTUM_ARM),
            None => (fuel, QUANTUM_FUEL),
        };
        // A quantum's worth of fuel costs a quantum's worth of time, so a thread that
        // really does burn a whole preemption slice is charged exactly what it always was.
        // What changes is everything either side of that: a voluntary yield after a
        // hundred instructions now costs a hundred instructions, and a thread that blocks
        // is billed for the work it did before blocking instead of for nothing.
        // Divide ONCE, carrying the remainder (see `VitaState::charge_rem`). Dividing
        // twice - by `QUANTUM_FUEL` and then by `cores` - truncated a small burn to zero
        // microseconds and stopped the clock outright for a title that yields in short
        // bursts, which is a livelock and not a rounding error.
        let den = quantum.saturating_mul(cores);
        let charge = |num: u64, rem: &mut u64| -> u64 {
            let total = rem.saturating_add(num);
            *rem = total % den;
            total / den
        };
        let (mut cpu_rem, mut io_rem) = self.state.charge_rem;
        let io_us = charge(QUANTUM_IO_US.saturating_mul(work), &mut io_rem);
        let cpu_us = charge(quantum_cpu_us().saturating_mul(work), &mut cpu_rem);
        self.state.charge_rem = (cpu_rem, io_rem);
        if self.state.has_io_waiters() {
            self.state.charge_io_quantum(io_us);
        }
        self.state.charge_cpu_quantum(cpu_us);
    }

    fn take_resume_code(&mut self, thid: i32) -> Option<u32> {
        self.state.take_resume_code(thid)
    }

    fn set_mirror_base(&mut self, addr: u32) {
        self.state
            .set_mutex_table(addr.wrapping_add(vita::mirror::SLOT_MUTEX_TABLE * 4));
        self.state.set_mirror_block_base(addr);
    }

    fn refresh_mirror(&mut self, write: &mut dyn FnMut(u32, u32)) -> usize {
        let words = vita::mirror::snapshot(&self.state);
        for (slot, value) in words.iter().enumerate() {
            write(slot as u32, *value);
        }
        self.state
            .note_yield_free_published(words[vita::mirror::SLOT_YIELD_FREE as usize] != 0);
        words.len()
    }

    fn on_frame_boundary(&mut self, frame: u64) {
        if nid_digest_spec().is_some() {
            self.nid_digest.end_frame(frame);
        }
        // Re-arm the per-frame texture comparison cadence - see `TextureSnapshots`.
        // Never WAIT here - see `VitaState::snaps_frames_pending`.
        match self.state.texture_snapshots.try_lock() {
            Ok(mut g) => {
                crate::perf::snaps_taken(crate::site!());
                for _ in 0..self
                    .state
                    .snaps_frames_pending
                    .swap(0, std::sync::atomic::Ordering::AcqRel)
                {
                    g.begin_frame();
                }
                g.begin_frame();
            }
            Err(_) => {
                self.state
                    .snaps_frames_pending
                    .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            }
        }
        self.state.world.set_frame(frame);
        self.state.set_cur_frame(frame);
        // Close the capture's frame so an observer can ask for the scenes THIS frame was
        // built from rather than for the last scene submitted, which on a multi-pass
        // title is the composite and holds no world geometry at all.
        self.state.capture.end_frame();
        self.state.end_frame_raw_targets();
        // Advance the virtual clock by one full 60 Hz frame per display flip: a
        // rendered frame represents ~16.6 ms of wall time passing, so the game's
        // animation and dialog timers progress at the right rate. `advance_time_to`
        // wakes every timed wait (cond timeout, sleep, audio grain park) whose
        // deadline falls within the frame, so those still fire - just at frame
        // granularity. This advance is essential: while a title renders continuously
        // its main thread never blocks, so the scheduler's "nothing runnable -> jump
        // the clock" idle path never runs and only this drives wall time.
        //
        // It must NOT be clamped down to the earliest pending deadline: a thread that
        // perpetually re-waits with a near-zero timeout (a 1 us cond poll) would then
        // pin the per-frame advance to ~1 us and freeze the game clock, so every
        // time-based state (a dialog's fade-in / input-enable timer) would never
        // elapse and the title would sit frozen and input-inert.
        //
        // It is a TOP-UP, not an unconditional addition: `charge_cpu_quantum` has
        // already advanced the clock for the guest CPU work this frame took, and adding
        // a whole frame on top of that would run the game clock fast (the failure mode
        // in `vitaslop-race-clock-5x`).
        const FRAME_US: u64 = 1_000_000 / 60;
        self.state.advance_time_frame(FRAME_US);
        // The modelled storage device gets one frame of progress per rendered frame,
        // net of anything the quantum charges already contributed.
        self.state.advance_io_frame(FRAME_US);
    }
}

/// The preemption quantum, in units of engine FUEL. Both engines must use this as their
/// preemption granularity, and it is the denominator every per-quantum constant below is
/// expressed against: a suspend that burned this much fuel costs exactly one quantum's
/// worth of clock, and a suspend that burned a tenth of it costs a tenth.
///
/// It lives here rather than in an engine because it is a UNIT, not a policy. The two
/// engines measure fuel the same way - it is proportional to executed wasm on both - so
/// the same guest work must cost the same game time whichever one ran it, and that only
/// holds while both divide by the same number.
///
/// **5,000,000 is what every real consumer preempts at** - the desktop's retail runner, the
/// resident session, the headless runner, and the browser's software-fuel interval (whose own
/// knob is documented where it is read, not here: naming it in this comment makes the generated
/// knob index attribute it to a constant rather than to the code that reads it). A 1,000,000
/// default also exists on two constructors nothing production uses, and taking the calibration
/// from THAT one made the game clock five times fast; the run that caught it reported a single
/// suspend burning 5,000,000 against a stated interval of 1,000,000, which is impossible when
/// the engine preempts at the interval. The headless run's `fuel ... max` line exists for
/// exactly that check - a max above the interval is a broken reading, not a busy title.
pub const QUANTUM_FUEL: u64 = 5_000_000;

/// Storage-clock time charged for one [`QUANTUM_ARM`] of guest execution, in microseconds.
///
/// # It is the SAME elapsed time the game clock charges, and that is not a coincidence
/// The storage clock ([`StorageClock`]) is a second clock in real microseconds:
/// a transfer parks until `io_us` reaches a deadline derived from the device's bandwidth. So
/// "how much time passed while the guest burned a quantum" has exactly one answer, and both
/// clocks must charge it. Two clocks measuring the same elapsed time at different rates means
/// the emulated console's storage and its CPU disagree about how long a second is - which
/// shows up as loads that complete at the wrong point in a title's own timeline, with nothing
/// in a run to say so.
///
/// It was an independent 2000 us until 2026-08-17b, when the game clock stopped being fitted
/// and the two drifted apart by 1.75x. The exact figure only matters when a title spins in
/// guest code waiting for a load instead of rendering, since a rendering title's storage
/// progress is pinned to its frames by [`VitaState::advance_io_frame`] - but that case is
/// exactly a loading screen, which is where every title spends its first minute.
pub(super) const QUANTUM_IO_US: u64 = QUANTUM_CPU_US;

/// The device's CPU clock. The Vita's application processor is a quad-core ARM
/// Cortex-A9 MPCore; 444 MHz is the clock a game runs at (the SoC can be driven higher,
/// but a title's own `sceKernelGetProcessTime` rate and every published figure for this
/// console are at 444).
pub(super) const GUEST_CPU_HZ: u64 = 444_000_000;
/// Guest ARM instructions the device retires per 1000 CYCLES - IPC in thousandths.
///
/// # This is the one judgement in the clock, and it is a claim about the DEVICE
/// Everything else in the emulated CPU's speed is now measured: the guest's instruction
/// count comes from the emitted counter and the clock rate is 444 MHz. What remains is
/// how many of those instructions a real A9 retires per cycle, and that cannot be read off
/// a datasheet for game code.
///
/// **1.000 is the model taken: one instruction per cycle.** The Cortex-A9 is a
/// dual-issue, out-of-order core, so its ceiling is 2.0 and this is well inside what the
/// hardware can sustain rather than an overstatement of it. It is also the simplest
/// statement that can be made about the device - a 444 MHz core retiring an instruction a
/// cycle - which matters for a constant that no measurement on this machine can settle.
///
/// A slower model is one edit away and its consequences are exactly proportional: 0.75
/// gives 1521 us against this 1141, i.e. an emulated console a third slower. If a real
/// device measurement ever becomes available (a title's own frame pacing against the
/// console's, taken on hardware), this is the constant it lands in.
pub(super) const GUEST_IPC_MILLI: u64 = 1_000;

/// Game-clock time charged for one [`QUANTUM_ARM`] of guest execution, in microseconds.
/// See [`VitaState::charge_cpu_quantum`] for why the game clock must advance for CPU work
/// at all.
///
/// # >>> DERIVED FROM THE DEVICE, NOT FITTED TO A RUN
/// `QUANTUM_ARM` guest instructions, a 444 MHz core, and an instruction per cycle. That is
/// the whole derivation, and every term in it is either measured or a stated claim about
/// the hardware. Nothing here is tuned to make a particular title behave.
///
/// **What it replaced, so it is not reintroduced: a hand-fitted 2780.** The clock used to
/// be billed in wasm OPERATORS, so the emulated console's speed was `fuel rate / code
/// expansion` - and the expansion is a property of this transpiler's codegen. Every
/// codegen improvement therefore made the emulated Vita faster, silently, and had to be
/// undone by hand: one session cut executed operators 28% and moved this constant 2000 ->
/// 2780 to put the speed back. That compensation could only ever be right for the
/// instruction mix it was measured on (the static factor said 1.391 where the race window
/// said 1.316), it had to be redone after every codegen change, and it left the emulated
/// console running at 182 MIPS - **2.4x slower than the device it models**.
///
/// It also had a correctness cost that was measured before it was removed. One title's
/// null-pointer dereference at a fixed frame moved with NOTHING except this constant: a
/// 33x sweep of I/O latency and a 4x sweep of the preemption quantum did not shift the
/// crash by a single frame, while 2000 and 4000 both made it disappear and only the fitted
/// 2780 reproduced it. A fitted timing constant does not merely mismodel the device, it
/// puts the guest into interleavings the device never produces.
///
/// **This is a rate, not a per-suspend price.** It was the latter until the game clock ran
/// 1.08x on one title and 4.34x on another with the same build on the same day, which is
/// the signature of billing a fixed amount for a variable thing. Charging per unit of
/// guest work is what makes one constant able to fit both.
///
/// The integer microsecond truncates 1140.96 to 1140, so the emulated core runs 0.08%
/// fast. That is three orders of magnitude inside the uncertainty on
/// [`GUEST_IPC_MILLI`], and the alternative is a finer time unit than the clock's own
/// consumers use.
///
/// Override for an experiment with `VITASLOP_QUANTUM_CPU_US`; 0 restores the old model
/// (a game clock that moves only on a flip or a scheduler idle) for an A/B.
pub(super) const QUANTUM_CPU_US: u64 = QUANTUM_ARM * 1_000 * 1_000_000 / (GUEST_CPU_HZ * GUEST_IPC_MILLI);

/// GUEST ARM INSTRUCTIONS in one quantum of guest work - the unit [`QUANTUM_CPU_US`] and
/// [`QUANTUM_IO_US`] are priced against, on any engine that carries the emitted per-block
/// instruction counter (`abi::ARM_COUNT_GLOBAL`).
///
/// # It is a UNIT, and only the ratio to [`QUANTUM_CPU_US`] means anything
/// The pair states one rate: `QUANTUM_ARM` guest instructions cost `QUANTUM_CPU_US`
/// microseconds of game time, so the emulated core retires 444 million instructions per
/// emulated second. Scaling both by the same factor changes nothing.
///
/// The particular value is historical and harmless: it is `QUANTUM_FUEL / 9.87`, the code
/// expansion measured on the day the clock stopped being billed in wasm OPERATORS, chosen
/// so that changeover was exactly speed-neutral. What it is NOT any more is a speed - see
/// [`QUANTUM_CPU_US`], which now derives that from the device.
///
/// # Both engines carry the counter this is measured in
/// The browser has emitted the per-block instruction count since 2026-08-16 and native
/// since 2026-08-17 (`threaded::WasmtimeThread::arm_retired`). So the two engines charge
/// identical guest work identically, and neither of them moves when the codegen does -
/// which is what makes a codegen A/B measurable at all, on either engine.
pub(super) const QUANTUM_ARM: u64 = 506_586;

/// [`QUANTUM_CPU_US`], overridable per-run by `VITASLOP_QUANTUM_CPU_US` (the knob exists
/// to take the calibration measurement and to A/B it; 0 restores the pre-calibration
/// behaviour of a game clock that only moves on a flip or an idle).
pub(super) fn quantum_cpu_us() -> u64 {
    static CELL: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_QUANTUM_CPU_US")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(QUANTUM_CPU_US)
    })
}

/// CPU cores a Vita gives a GAME. The console has four Cortex-A9 cores; the system
/// software reserves one, so a title's threads run three at a time. See
/// [`VitaEnv::on_quantum`], which is the only thing this number means to us: how much
/// wall time one scheduler quantum of guest execution is worth.
///
/// It is a property of the DEVICE, not a tuning constant. Override it with
/// `VITASLOP_GUEST_CORES` only to A/B the model; 1 restores the one-baton clock.
pub(super) const GUEST_CORES: usize = 3;

/// [`GUEST_CORES`], overridable per-run by `VITASLOP_GUEST_CORES` for an A/B.
pub fn guest_cores() -> usize {
    static CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_GUEST_CORES")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(GUEST_CORES)
    })
}

/// `VITASLOP_FRAME_TOPUP=1`: top the game clock up to a full frame at a display flip.
/// **OFF by default, and it should stay off** - see [`VitaState::advance_time_frame`].
///
/// It is superseded by [`VitaState::pace_flip`], which parks the flipping thread until
/// the scanout can latch. The two cannot both be on: the top-up runs at the frame
/// boundary, AFTER the flip's park, so it would add a second frame of clock on top of
/// the one the guest just waited out and run the game clock at double rate. Turning it
/// on also silently disables pacing - the top-up leaves the clock past the vblank floor,
/// so the next flip never has anything to wait for.
///
/// Kept as a knob only so the old behaviour can be measured against the new one.
pub(super) fn frame_topup() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::var("VITASLOP_FRAME_TOPUP").as_deref() == Ok("1"))
}

/// A shared handle to a `VitaEnv`: attach one clone as the engine's import
/// environment and keep another to read the capture back after the run. Single
/// threaded (the guest CPU worker), so `Rc`/`RefCell` are the right tools.
impl ImportDispatch for std::rc::Rc<std::cell::RefCell<VitaEnv>> {
    fn dispatch(
        &mut self,
        index: u32,
        regs: &mut [u32; REG_COUNT],
        vfp: &mut [u32; VFP_ARG_COUNT],
        mem: &mut dyn GuestMemory,
        base: u32,
    ) -> SvcOutcome {
        self.borrow_mut().dispatch(index, regs, vfp, mem, base)
    }

    fn take_reentry(&mut self) -> Option<Reentry> {
        self.borrow_mut().take_reentry()
    }

    fn set_thread_exit(&mut self, thid: i32, code: u32) {
        self.borrow_mut().set_thread_exit(thid, code);
    }

    fn take_guest_call(&mut self, thid: i32) -> Option<GuestCall> {
        self.borrow_mut().take_guest_call(thid)
    }

    fn thread_return_calls(&mut self, thid: i32, code: u32) -> Vec<GuestCall> {
        self.borrow_mut().thread_return_calls(thid, code)
    }

    fn thread_entry_starting(&mut self, thid: i32, last: bool) {
        self.borrow_mut().thread_entry_starting(thid, last);
    }

    fn guest_call_returned(
        &mut self,
        thid: i32,
        r0: u32,
        r1: u32,
        mem: &dyn GuestMemory,
        base: u32,
    ) {
        self.borrow_mut()
            .guest_call_returned(thid, r0, r1, mem, base);
    }

    fn running_thread(&self) -> i32 {
        self.borrow().running_thread()
    }

    fn set_current_thread(&mut self, thid: i32) {
        self.borrow_mut().set_current_thread(thid);
    }

    fn note_runnable_others(&mut self, n: usize) {
        self.borrow_mut().note_runnable_others(n);
    }

    fn resolve_deferred(&mut self, words: &mut dyn GuestWords) -> usize {
        self.borrow_mut().resolve_deferred(words)
    }

    fn thread_tls_base(&mut self, thid: i32) -> (u32, u32, u32) {
        self.borrow_mut().thread_tls_base(thid)
    }

    fn take_spawns(&mut self) -> Vec<Reentry> {
        self.borrow_mut().take_spawns()
    }

    fn take_wakes(&mut self) -> Vec<i32> {
        self.borrow_mut().take_wakes()
    }

    fn take_terminations(&mut self) -> Vec<i32> {
        self.borrow_mut().take_terminations()
    }

    fn take_stat_writes(&mut self) -> Vec<(u32, u32)> {
        self.borrow_mut().take_stat_writes()
    }

    fn earliest_deadline(&self) -> Option<u64> {
        self.borrow().earliest_deadline()
    }

    fn advance_time_to(&mut self, to_us: u64) {
        self.borrow_mut().advance_time_to(to_us);
    }

    fn clock_us(&self) -> u64 {
        self.borrow().clock_us()
    }

    fn release_earliest_io(&mut self) -> bool {
        self.borrow_mut().release_earliest_io()
    }

    fn earliest_io_remaining_us(&self) -> Option<u64> {
        self.borrow().earliest_io_remaining_us()
    }

    fn charge_io_idle(&mut self, us: u64) {
        self.borrow_mut().charge_io_idle(us);
    }

    fn on_guest_work(&mut self, runnable: usize, fuel: u64, retired: Option<u64>) {
        self.borrow_mut().on_guest_work(runnable, fuel, retired);
    }

    fn take_resume_code(&mut self, thid: i32) -> Option<u32> {
        self.borrow_mut().take_resume_code(thid)
    }
}
