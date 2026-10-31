//! The kernel's synchronization objects - semaphores, mutexes, lightweight mutexes, condition
//! variables, lightweight condition variables, reader/writer locks, event flags, simple events
//! and message pipes - with every thread parked on them.
//!
//! [`SyncObjects`] OWNS that state: its fields are private to this file, so nothing else in the
//! host can reach past these methods into a waiter list. What an operation needs from the rest
//! of the host - the clock, the calling thread, and the queues a release feeds - it is handed
//! as a [`Sched`], built by `sched!` from `VitaState`'s own fields. The `VitaState` methods the
//! NID handlers call are thin forwarders in `state_sync.rs`.
use super::*;

/// What a sync operation needs from the scheduler: the clock, the calling thread, and the
/// queues a release feeds. Built per call by `sched!` from `VitaState`'s fields - borrows only,
/// so building one costs a few pointer copies.
pub(super) struct Sched<'a> {
    pub(super) now: u64,
    pub(super) current: i32,
    pub(super) wakes: &'a mut Vec<i32>,
    pub(super) causes: &'a mut Vec<(i32, i32, &'static str)>,
    pub(super) trace: bool,
    pub(super) resume_codes: &'a mut Vec<(i32, u32)>,
    pub(super) stat_writes: &'a mut Vec<(u32, u32)>,
}

impl Sched<'_> {
    /// Make `thid` runnable; with wake tracing on, note who released it and from where - the
    /// same record `push_wake!` keeps for the rest of the host.
    #[inline]
    fn wake(&mut self, thid: i32, site: &'static str) {
        if self.trace {
            self.causes.push((thid, self.current, site));
        }
        self.wakes.push(thid);
    }

    /// Owe `thid` an `r0` other than the pre-park one when it resumes.
    #[inline]
    fn resume_code(&mut self, thid: i32, code: u32) {
        self.resume_codes.push((thid, code));
    }

    /// Queue a guest word write to apply before the woken thread resumes.
    #[inline]
    fn stat_write(&mut self, addr: u32, value: u32) {
        self.stat_writes.push((addr, value));
    }
}

/// [`Sched::wake`] at the call's own `file:line`.
macro_rules! wake {
    ($cx:expr, $thid:expr) => {
        $cx.wake($thid, *crate::site!())
    };
}

/// A live semaphore. The count is what the primitive runs on; the rest is what
/// `sceKernelGetSemaInfo` reports and what `sceKernelOpenSema` matches on, so it is
/// recorded at create time rather than reconstructed. `init`/`max` are the values the
/// guest asked for, not clamps we apply - the count is floored at 0 by the waiters and
/// the kernel does not police `max` on signal either.
struct SemaRec {
    uid: i32,
    /// The name passed to `sceKernelCreateSema`. Named semaphores are what
    /// `sceKernelOpenSema` resolves, so this is a lookup key, not just a label.
    name: String,
    attr: u32,
    init: i32,
    max: i32,
    count: i32,
}

/// A recursive mutex's state (preemptive mode only; the single-thread model needs none).
/// `waiters` are the threads parked in `sceKernelLockMutex`, in FIFO order.
///
/// # Ownership lives in the KERNEL MUTEX TABLE, not here - usually
/// Identity, owner and recursion depth are four words in the table
/// [`crate::vita::kmutex`] lays out inside the host-mirror block, so the emitted
/// `InlineOp::KernelMutexLock` and the handler read ONE copy. `owner`/`count` below are the
/// fallback home for the mutexes that have no entry: a run whose module inlined nothing
/// reserves no block, and a uid that collides with a live entry is refused one at create. Which
/// home applies is decided in one place, [`SyncObjects::mutex_in_table`], never guessed - two
/// homes that could both be live would be exactly the drift this table exists to end.
struct MutexRec {
    uid: i32,
    /// What the guest created it with, for `sceKernelGetMutexInfo` to hand back.
    name: String,
    attr: u32,
    init: i32,
    /// Owner while this mutex has no table entry; `None` if free. Unused otherwise.
    owner: Option<i32>,
    /// Recursion depth while this mutex has no table entry. Unused otherwise.
    count: i32,
    waiters: Vec<i32>,
}

/// What the HOST still keeps for a lightweight mutex: the threads parked on it, in FIFO
/// order, and the work-area address that names it.
///
/// The ownership itself - identity, owner, recursion count - is NOT here. It lives in the
/// guest's own work area ([`crate::vita::lwwork`]), which is where the hardware keeps it
/// and what makes the uncontended take inlinable. Only the parked QUEUE stays host-side,
/// because a list of thread ids in arrival order is not something guest memory holds
/// usefully; its LENGTH is published into the work area so guest code can tell the case it
/// may serve itself from the case only the host can.
struct LwMutexRec {
    work: u32,
    /// Threads parked on this mutex in FIFO order, each with the virtual-clock deadline of
    /// its TIMED lock (`None` = wait forever).
    ///
    /// The deadline is not decoration: `sceKernelLockLwMutex(work, count, pTimeout)` takes a
    /// timeout, and a title that passes one has a path for the lock FAILING. Parking such a
    /// caller forever turns a lock it expected to give up on into a hang - measured on a
    /// retail title whose display thread holds this mutex while it idles, so the timed lock
    /// its main thread takes to post work is the only way either of them ever moves again.
    waiters: Vec<LwMutexWaiter>,
    /// What `sceKernelCreateLwMutex` was given - `(name, attr, initCount)` - for
    /// `sceKernelGetLwMutexInfo` to hand back. Empty/zero for an ADOPTED work area.
    meta: (String, u32, i32),
}

/// One thread parked on a lightweight mutex - see [`LwMutexRec::waiters`].
#[derive(Clone, Copy, Debug)]
struct LwMutexWaiter {
    thid: i32,
    deadline: Option<u64>,
}

/// A condition variable's state (preemptive mode only). `mutex` is the associated
/// mutex it releases on wait and re-acquires on wake; `waiters` are the threads
/// parked in `sceKernelWaitCond` in FIFO order, each with an optional virtual-clock
/// `deadline` for a timed wait (`None` = wait forever). A condition variable has no
/// memory: a signal with no waiter is lost.
struct CondRec {
    uid: i32,
    mutex: i32,
    waiters: Vec<CondWaiter>,
}

/// A thread parked in `sceKernelWaitCond`: which thread and an optional virtual-clock
/// deadline (`Some` for a timed wait, woken by a signal or when the clock reaches it;
/// `None` for an infinite wait). A timed-out cond wait still re-acquires the mutex
/// before it resumes (see [`VitaState::advance_time_to`]).
struct CondWaiter {
    thid: i32,
    deadline: Option<u64>,
}

/// A thread parked in `sceKernelWaitSema`: which semaphore, which thread, how many
/// signals it still needs, and an optional virtual-clock `deadline` for a timed wait
/// (`None` = wait forever). It is released (and the count consumed) when a signal
/// makes `need` available, or woken with `SCE_KERNEL_ERROR_WAIT_TIMEOUT` when the
/// deadline passes.
struct SemaWaiter {
    uid: i32,
    thid: i32,
    need: i32,
    deadline: Option<u64>,
    /// Clock and its sources `(clock, quanta, wall, idle)` when the wait began - so a wait that
    /// TIMES OUT can say what produced the guest time it waited through. See
    /// [`timed_out_waits`].
    began: (u64, u64, u64, u64),
}

/// A reader/writer lock (`sceKernelCreateRWLock`). Host-resident: there is no inline fast
/// path for it, so every take and release is a host call and this record is the one home of
/// its state.
struct RwLockRec {
    uid: i32,
    name: String,
    attr: u32,
    /// The thread holding it for WRITING and its recursion depth.
    writer: Option<i32>,
    write_count: u32,
    /// The threads holding it for READING, each with its own recursion depth.
    readers: Vec<(i32, u32)>,
    /// Parked lockers, FIFO. See [`SyncObjects::rwlock_grant`] for the order they are let in.
    waiters: Vec<RwWaiter>,
}

/// A thread parked in `sceKernelLockReadRWLock` / `sceKernelLockWriteRWLock`.
struct RwWaiter {
    thid: i32,
    write: bool,
    deadline: Option<u64>,
}

/// What a reader/writer lock operation answers, before the guest-facing error code is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RwLockResult {
    /// Taken (or released) now.
    Done,
    /// The lock is held in a way that excludes this caller: park it (or, for a try-lock, refuse).
    Busy,
    /// No reader/writer lock has this uid.
    Unknown,
    /// A NON-recursive lock re-taken by a thread that already holds it.
    Recursive,
    /// A release by a thread that does not hold it in that mode.
    NotOwned,
}

/// A thread parked in `sceKernelReceiveMsgPipe` until the pipe holds what it asked for.
/// `full`: the FULL mode wants all `size` bytes; otherwise (ASAP) any bytes will do.
struct MsgPipeWaiter {
    uid: i32,
    thid: i32,
    buf: u32,
    size: u32,
    full: bool,
    result: u32,
    deadline: Option<u64>,
}

/// A thread parked in `sceKernelWaitEventFlag`: which flag, which thread, the bit
/// pattern and wait mode it asked for, the guest address of its `outBits`
/// out-parameter (0 = NULL), and an optional virtual-clock deadline for a timed
/// wait. Released when a `sceKernelSetEventFlag` satisfies the pattern (the match
/// pattern is then written through `out_addr` via the pending stat-write channel)
/// or the deadline passes.
struct EvfWaiter {
    uid: i32,
    thid: i32,
    bits: u32,
    mode: u32,
    out_addr: u32,
    deadline: Option<u64>,
    /// A SIMPLE EVENT wait's `SceUInt64 *pUserData`: the event's user data is written there
    /// at the match. Zero for an event-flag wait.
    user_data_addr: u32,
}

/// Every kernel synchronization object and every thread parked on one. See the module docs.
#[derive(Default)]
pub(super) struct SyncObjects {
    // Synchronization objects. Bring-up model: one thread of control (workers run
    // synchronously to completion), so nothing ever actually blocks; a semaphore's
    // count and an event flag's bit pattern are still tracked so their observable
    // state is faithful for single-thread use (guarding data, wait-then-read).
    semaphores: Vec<SemaRec>,
    event_flags: Vec<(i32, u32)>,
    /// The guest's own name for each event flag, for REPORTS only (see
    /// [`create_event_flag`](VitaState::create_event_flag)). Kept beside `event_flags` rather
    /// than in it so no waiter/set path pays for a string it never reads.
    event_flag_names: std::collections::BTreeMap<i32, String>,
    /// Each event flag's creation `(attr, initPattern)`, for `sceKernelGetEventFlagInfo`. Kept
    /// beside `event_flags` for the same reason as the names.
    event_flag_params: std::collections::BTreeMap<i32, (u32, u32)>,
    mutexes: Vec<MutexRec>,
    rwlocks: Vec<RwLockRec>,
    lwmutexes: Vec<LwMutexRec>,
    /// Lightweight-mutex handoffs decided where no guest memory was reachable, settled by
    /// [`resolve_deferred_lwmutex`](VitaState::resolve_deferred_lwmutex) before the next
    /// resume. `(work area, thread to give it to)`.
    pending_lwmutex_acquires: Vec<(u32, i32)>,
    /// The same deferral for HEAVYWEIGHT mutexes: `(mutex uid, thread)` pairs a timed-out
    /// `sceKernelWaitCond` owes, settled by
    /// [`resolve_deferred_lwmutex`](VitaState::resolve_deferred_lwmutex) before the next
    /// resume, because deciding them reads the kernel mutex table in guest memory.
    pending_mutex_acquires: Vec<(i32, i32)>,
    conds: Vec<CondRec>,
    sema_waiters: Vec<SemaWaiter>,
    evf_waiters: Vec<EvfWaiter>,
    /// Guest address of the KERNEL MUTEX TABLE - the run of words behind the mirrored slots
    /// in the host-mirror block, laid out by [`crate::vita::kmutex`]. Zero when this build
    /// reserved no block (nothing inlined a mirror read), and then every kernel mutex keeps
    /// its ownership on the host exactly as it did before the table existed.
    mutex_table: u32,
    /// Message pipes by uid: a byte FIFO each.
    msg_pipes: std::collections::BTreeMap<i32, std::collections::VecDeque<u8>>,
    /// Simple events: `uid -> (attr, user data)`. The pattern and the waiters are the event
    /// flag's own (`event_flags`, `evf_waiters`) - see [`VitaState::create_simple_event`].
    simple_events: std::collections::HashMap<i32, (u32, u64)>,
    /// Receivers parked on a message pipe, FIFO. See [`MsgPipeWaiter`].
    msgpipe_waiters: Vec<MsgPipeWaiter>,
    /// Threads parked in `sceKernelWaitLwCond`, as `(thread id, cond work address,
    /// wake deadline)`. `deadline` is `Some(virtual_us)` for a timed wait (woken by
    /// a signal or when the clock reaches it) or `None` for an infinite wait (only a
    /// signal wakes it). Keyed by the cond's guest work pointer, since lightweight
    /// objects have no kernel handle. See the scheduler's idle-time advance.
    lwcond_waiters: Vec<(i32, u32, Option<u64>)>,
    /// Each lightweight condition variable's associated lightweight mutex, as `(cond
    /// work address, mutex work address)`, recorded at `sceKernelCreateLwCond` (which
    /// is handed the `SceKernelLwMutexWork*` it binds to). `sceKernelWaitLwCond`
    /// atomically releases this mutex as it parks and re-acquires it before the woken
    /// thread runs - exactly like the heavyweight cond/mutex pair. Without it a thread
    /// would hold the mutex across the wait and deadlock any thread that then locks it.
    lwcond_mutex: Vec<(u32, u32)>,
}

impl SyncObjects {
    /// `SCE_KERNEL_RW_LOCK_ATTR_RECURSIVE`. vitasdk names no RW-lock attribute bits; this is the
    /// mutex family's `SCE_KERNEL_MUTEX_ATTR_RECURSIVE`, which the kernel's attribute words share.
    const RWLOCK_ATTR_RECURSIVE: u32 = 0x02;

    /// SceEventFlagWaitTypes: OR (any requested bit) vs the default AND (all of
    /// them), plus the two clear-on-match ops.
    const EVF_WAITOR: u32 = 1;
    const EVF_WAITCLEAR: u32 = 2;
    const EVF_WAITCLEAR_PAT: u32 = 4;

    /// `attr` bit of a simple event read as AUTO-RESET. A fighting title creates its task events with
    /// exactly this bit and uses each as a one-shot wake between a producer and one waiter.
    pub(super) const SIMPLE_EVENT_AUTO_RESET: u32 = 0x100;

    // --- semaphores ---

    pub(super) fn create_sema(&mut self, uid: i32, name: &str, attr: u32, init: i32, max: i32) {
        self.semaphores.push(SemaRec {
            uid,
            name: name.to_string(),
            attr,
            init,
            max,
            count: init,
        });
    }

    pub(super) fn sema_exists(&self, uid: i32) -> bool {
        self.semaphores.iter().any(|s| s.uid == uid)
    }

    pub(super) fn sema_by_name(&self, name: &str) -> Option<i32> {
        self.semaphores
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.uid)
    }

    pub(super) fn sema_info(&self, uid: i32) -> Option<(&str, u32, i32, i32, i32, i32)> {
        let s = self.semaphores.iter().find(|s| s.uid == uid)?;
        let waiting = self.sema_waiters.iter().filter(|w| w.uid == uid).count() as i32;
        Some((s.name.as_str(), s.attr, s.init, s.count, s.max, waiting))
    }

    pub(super) fn sema_signal_overflows(&self, uid: i32, n: i32) -> Option<bool> {
        let s = self.semaphores.iter().find(|s| s.uid == uid)?;
        Some(s.count.saturating_add(n) > s.max)
    }

    pub(super) fn sema_try_acquire(&mut self, uid: i32, need: i32) -> bool {
        if let Some(s) = self.semaphores.iter_mut().find(|s| s.uid == uid)
            && s.count >= need
        {
            s.count -= need;
            return true;
        }
        false
    }

    pub(super) fn sema_wait(&mut self, uid: i32, n: i32) {
        if let Some(s) = self.semaphores.iter_mut().find(|s| s.uid == uid) {
            s.count = (s.count - n).max(0);
        }
    }

    pub(super) fn sema_signal(&mut self, uid: i32, n: i32) {
        if let Some(s) = self.semaphores.iter_mut().find(|s| s.uid == uid) {
            s.count += n;
        }
    }

    /// Park `thid` for `need` signals on `uid`; `began` is the clock and its sources now.
    pub(super) fn sema_block(
        &mut self,
        uid: i32,
        thid: i32,
        need: i32,
        deadline: Option<u64>,
        began: (u64, u64, u64, u64),
    ) {
        self.sema_waiters.push(SemaWaiter {
            uid,
            thid,
            need,
            deadline,
            began,
        });
    }

    pub(super) fn sema_signal_wake(&mut self, cx: &mut Sched, uid: i32, n: i32) {
        self.sema_signal(uid, n);
        loop {
            let count = self
                .semaphores
                .iter()
                .find(|s| s.uid == uid)
                .map(|s| s.count)
                .unwrap_or(0);
            // The first waiter on this semaphore whose need the count can meet.
            let next = self
                .sema_waiters
                .iter()
                .position(|w| w.uid == uid && w.need <= count);
            let Some(idx) = next else { break };
            let w = self.sema_waiters.remove(idx);
            self.sema_signal(uid, -w.need); // consume the count for the woken waiter
            wake!(cx, w.thid);
        }
    }

    pub(super) fn sema_cancel(&mut self, cx: &mut Sched, uid: i32, set_count: i32) -> Option<i32> {
        let init = self.semaphores.iter().find(|s| s.uid == uid)?.init;
        let (waiters, need): (i32, i32) = self
            .sema_waiters
            .iter()
            .filter(|w| w.uid == uid)
            .fold((0, 0), |(n, need), w| (n + 1, need + w.need));
        if waiters > 0 {
            self.sema_signal_wake(cx, uid, need);
        }
        let s = self.semaphores.iter_mut().find(|s| s.uid == uid)?;
        s.count = if set_count < 0 { init } else { set_count };
        Some(waiters)
    }

    // --- heavyweight mutexes ---

    pub(super) fn set_mutex_table(&mut self, addr: u32) {
        self.mutex_table = addr;
    }

    pub(super) fn mutex_table(&self) -> u32 {
        self.mutex_table
    }

    /// Where mutex `uid`'s ownership lives: `true` when the KERNEL MUTEX TABLE holds it,
    /// `false` when it is host-resident in its [`MutexRec`].
    ///
    /// Asked at the top of every operation and threaded through rather than re-derived: the two
    /// homes must never both be written for one mutex, and the way to guarantee that is for one
    /// question to decide it.
    fn mutex_in_table(&self, w: &dyn GuestWords, uid: i32) -> bool {
        crate::vita::kmutex::owns_entry(w, self.mutex_table, uid)
    }

    /// The owning thread of `uid`, or `None` while it is free.
    fn mutex_owner(&self, w: &dyn GuestWords, uid: i32) -> Option<i32> {
        if self.mutex_in_table(w, uid) {
            return crate::vita::kmutex::owner(w, self.mutex_table, uid);
        }
        self.mutexes
            .iter()
            .find(|m| m.uid == uid)
            .and_then(|m| m.owner)
    }

    /// Take `uid` for `thid`: name the owner and bump the recursion depth.
    fn mutex_take(&mut self, w: &mut dyn GuestWords, uid: i32, thid: i32) {
        use crate::vita::kmutex;
        if self.mutex_in_table(w, uid) {
            let n = kmutex::count(w, self.mutex_table, uid);
            kmutex::set_owner(w, self.mutex_table, uid, thid);
            kmutex::set_count(w, self.mutex_table, uid, n + 1);
            return;
        }
        if let Some(m) = self.mutexes.iter_mut().find(|m| m.uid == uid) {
            m.owner = Some(thid);
            m.count += 1;
        }
    }

    /// Publish `uid`'s parked-waiter COUNT into its table entry, so the emitted form can see
    /// that somebody is waiting and stay off its fast path. A no-op for a host-resident mutex.
    fn mutex_publish_waiters(&mut self, w: &mut dyn GuestWords, uid: i32) {
        if !self.mutex_in_table(w, uid) {
            return;
        }
        let n = self
            .mutexes
            .iter()
            .find(|m| m.uid == uid)
            .map_or(0, |m| m.waiters.len());
        crate::vita::kmutex::set_waiters(w, self.mutex_table, uid, n);
    }

    pub(super) fn create_mutex(
        &mut self,
        w: &mut dyn GuestWords,
        uid: i32,
        name: &str,
        attr: u32,
        init: i32,
    ) {
        crate::vita::kmutex::claim(w, self.mutex_table, uid);
        self.mutexes.push(MutexRec {
            uid,
            name: name.to_string(),
            attr,
            init,
            owner: None,
            count: 0,
            waiters: Vec::new(),
        });
    }

    pub(super) fn mutex_info(
        &self,
        w: &dyn GuestWords,
        uid: i32,
    ) -> Option<(String, u32, i32, i32, i32, i32)> {
        let m = self.mutexes.iter().find(|m| m.uid == uid)?;
        let count = if self.mutex_in_table(w, uid) {
            crate::vita::kmutex::count(w, self.mutex_table, uid)
        } else {
            m.count
        };
        let owner = self.mutex_owner(w, uid).unwrap_or(0);
        Some((
            m.name.clone(),
            m.attr,
            m.init,
            count,
            owner,
            m.waiters.len() as i32,
        ))
    }

    pub(super) fn mutex_lock(&mut self, w: &mut dyn GuestWords, uid: i32, cur: i32) -> bool {
        if !self.mutexes.iter().any(|m| m.uid == uid) {
            // Unknown mutex: treat as uncontended success.
            return true;
        }
        match self.mutex_owner(w, uid) {
            None => {
                self.mutex_take(w, uid, cur);
                true
            }
            Some(o) if o == cur => {
                self.mutex_take(w, uid, cur);
                true
            }
            Some(_) => {
                if let Some(m) = self.mutexes.iter_mut().find(|m| m.uid == uid) {
                    m.waiters.push(cur);
                }
                self.mutex_publish_waiters(w, uid);
                false
            }
        }
    }

    pub(super) fn mutex_contended(&self, w: &dyn GuestWords, uid: i32, cur: i32) -> bool {
        if !self.mutexes.iter().any(|m| m.uid == uid) {
            return false;
        }
        matches!(self.mutex_owner(w, uid), Some(o) if o != cur)
    }

    pub(super) fn mutex_unlock(&mut self, cx: &mut Sched, w: &mut dyn GuestWords, uid: i32) {
        use crate::vita::kmutex;
        let mut wake = None;
        let in_table = self.mutex_in_table(w, uid);
        if self.mutexes.iter().any(|m| m.uid == uid) {
            let count = if in_table {
                let n = kmutex::count(w, self.mutex_table, uid);
                let n = if n > 0 { n - 1 } else { 0 };
                kmutex::set_count(w, self.mutex_table, uid, n);
                n
            } else {
                let m = self
                    .mutexes
                    .iter_mut()
                    .find(|m| m.uid == uid)
                    .expect("just checked");
                if m.count > 0 {
                    m.count -= 1;
                }
                m.count
            };
            if count == 0 {
                let next = self
                    .mutexes
                    .iter_mut()
                    .find(|m| m.uid == uid)
                    .and_then(|m| (!m.waiters.is_empty()).then(|| m.waiters.remove(0)));
                match next {
                    // A table entry needs no "free" write: `count == 0` IS free, which is why
                    // the owner word is meaningful only beside a non-zero count.
                    None => {
                        if !in_table && let Some(m) = self.mutexes.iter_mut().find(|m| m.uid == uid)
                        {
                            m.owner = None;
                        }
                    }
                    Some(next) => {
                        self.mutex_take(w, uid, next);
                        self.mutex_publish_waiters(w, uid);
                        wake = Some(next);
                    }
                }
            }
        }
        if let Some(next) = wake {
            // Traced HERE and not at the lock site, because this is where a parked thread
            // actually becomes the owner. `lock_mutex` only ever sees the immediate acquire, so
            // counting acquisitions there undercounts every handoff - which on a two-thread
            // producer/consumer mutex means the log shows thousands of unlocks against a
            // handful of takes and reads like a thread unlocking something it never held.
            tracing::trace!(target: "vitaslop::sema", id = uid, thread = next, "mutex GRANTED on wake");
            wake!(cx, next);
        }
    }

    /// Acquire mutex `uid` on behalf of thread `thid` (not the current thread).
    /// Used when a condition-variable signal transfers a waiter to its mutex: if
    /// the mutex is free the thread takes it and is woken now; otherwise it joins
    /// the mutex wait queue and is woken when the owner unlocks. An unknown mutex
    /// just wakes the thread.
    fn mutex_acquire_for(&mut self, cx: &mut Sched, w: &mut dyn GuestWords, uid: i32, thid: i32) {
        let mut wake = true;
        if self.mutexes.iter().any(|m| m.uid == uid) {
            match self.mutex_owner(w, uid) {
                None => self.mutex_take(w, uid, thid),
                Some(o) if o == thid => self.mutex_take(w, uid, thid),
                Some(_) => {
                    if let Some(m) = self.mutexes.iter_mut().find(|m| m.uid == uid) {
                        m.waiters.push(thid);
                    }
                    self.mutex_publish_waiters(w, uid);
                    wake = false; // woken later, when the owner unlocks
                }
            }
        }
        if wake {
            wake!(cx, thid);
        }
    }

    // --- reader/writer locks ---

    pub(super) fn create_rwlock(&mut self, uid: i32, name: &str, attr: u32) {
        self.rwlocks.push(RwLockRec {
            uid,
            name: name.to_string(),
            attr,
            writer: None,
            write_count: 0,
            readers: Vec::new(),
            waiters: Vec::new(),
        });
    }

    pub(super) fn delete_rwlock(&mut self, cx: &mut Sched, uid: i32) -> bool {
        let Some(idx) = self.rwlocks.iter().position(|l| l.uid == uid) else {
            return false;
        };
        let l = self.rwlocks.remove(idx);
        for w in l.waiters {
            wake!(cx, w.thid);
            cx.resume_code(w.thid, SCE_KERNEL_ERROR_WAIT_DELETE);
        }
        true
    }

    pub(super) fn rwlock_try(&mut self, uid: i32, write: bool, cur: i32) -> RwLockResult {
        let Some(l) = self.rwlocks.iter_mut().find(|l| l.uid == uid) else {
            return RwLockResult::Unknown;
        };
        let recursive = l.attr & Self::RWLOCK_ATTR_RECURSIVE != 0;
        let holds_write = l.writer == Some(cur);
        let holds_read = l.readers.iter().any(|&(t, _)| t == cur);
        if (holds_write || holds_read) && !recursive {
            return RwLockResult::Recursive;
        }
        if write {
            let others_read = l.readers.iter().any(|&(t, _)| t != cur);
            if holds_write
                || (l.writer.is_none() && !others_read && (holds_read || l.waiters.is_empty()))
            {
                l.writer = Some(cur);
                l.write_count += 1;
                return RwLockResult::Done;
            }
            return RwLockResult::Busy;
        }
        let writer_queued = l.waiters.iter().any(|w| w.write);
        if holds_write || holds_read || (l.writer.is_none() && !writer_queued) {
            match l.readers.iter_mut().find(|(t, _)| *t == cur) {
                Some(r) => r.1 += 1,
                None => l.readers.push((cur, 1)),
            }
            return RwLockResult::Done;
        }
        RwLockResult::Busy
    }

    pub(super) fn rwlock_info(&self, uid: i32) -> Option<(&str, u32, u32, i32, u32, u32)> {
        let l = self.rwlocks.iter().find(|l| l.uid == uid)?;
        let count = match l.writer {
            Some(_) => l.write_count,
            None => l.readers.iter().map(|&(_, n)| n).sum(),
        };
        let write_waiters = l.waiters.iter().filter(|w| w.write).count() as u32;
        let read_waiters = l.waiters.len() as u32 - write_waiters;
        Some((
            l.name.as_str(),
            l.attr,
            count,
            l.writer.unwrap_or(0),
            read_waiters,
            write_waiters,
        ))
    }

    pub(super) fn rwlock_block(&mut self, uid: i32, write: bool, thid: i32, deadline: Option<u64>) {
        if let Some(l) = self.rwlocks.iter_mut().find(|l| l.uid == uid) {
            l.waiters.push(RwWaiter {
                thid,
                write,
                deadline,
            });
        }
    }

    pub(super) fn rwlock_unlock(&mut self, cx: &mut Sched, uid: i32, write: bool) -> RwLockResult {
        let cur = cx.current;
        let Some(l) = self.rwlocks.iter_mut().find(|l| l.uid == uid) else {
            return RwLockResult::Unknown;
        };
        if write {
            if l.writer != Some(cur) {
                return RwLockResult::NotOwned;
            }
            l.write_count -= 1;
            if l.write_count == 0 {
                l.writer = None;
            }
        } else {
            let Some(i) = l.readers.iter().position(|&(t, _)| t == cur) else {
                return RwLockResult::NotOwned;
            };
            l.readers[i].1 -= 1;
            if l.readers[i].1 == 0 {
                l.readers.remove(i);
            }
        }
        self.rwlock_grant(cx, uid);
        RwLockResult::Done
    }

    /// Let parked lockers into reader/writer lock `uid` in FIFO order: the head writer once
    /// nobody holds the lock, or the run of readers at the head while nobody writes. Stops at
    /// the first waiter that cannot get in, so a queued writer is never overtaken by a reader
    /// that arrived after it.
    fn rwlock_grant(&mut self, cx: &mut Sched, uid: i32) {
        let mut woken = Vec::new();
        if let Some(l) = self.rwlocks.iter_mut().find(|l| l.uid == uid) {
            while let Some(w) = l.waiters.first() {
                if l.writer.is_some() {
                    break;
                }
                if w.write {
                    if !l.readers.is_empty() {
                        break;
                    }
                    let w = l.waiters.remove(0);
                    l.writer = Some(w.thid);
                    l.write_count = 1;
                    woken.push(w.thid);
                    break;
                }
                let w = l.waiters.remove(0);
                match l.readers.iter_mut().find(|(t, _)| *t == w.thid) {
                    Some(r) => r.1 += 1,
                    None => l.readers.push((w.thid, 1)),
                }
                woken.push(w.thid);
            }
        }
        for thid in woken {
            tracing::trace!(target: "vitaslop::sema", id = uid, thread = thid, "rwlock GRANTED on wake");
            wake!(cx, thid);
        }
    }

    // --- lightweight mutexes, whose state lives in the GUEST WORK AREA -------------
    //
    // A lightweight mutex has no kernel handle. Identity, owner and recursion count are
    // four words of the caller's own `SceKernelLwMutexWork` (see `crate::vita::lwwork`),
    // which is where the hardware keeps them and what lets the uncontended take be
    // emitted straight into guest code. Everything below is the CONTENDED half plus the
    // parked queue - the part a userspace CAS cannot do, and the part that on the device
    // is a syscall too.
    //
    // These still mirror the heavyweight `mutex_lock`/`mutex_contended`/`mutex_unlock`
    // in behaviour, so a `sceKernelLockLwMutex` genuinely blocks on contention and enforces
    // mutual exclusion.

    /// The record for the lightweight mutex at guest work address `work`, created (with no
    /// waiters) on first use - a title may lock a work area it zero-initialized itself
    /// without a distinct create call.
    fn lwmutex_rec(&mut self, work: u32) -> &mut LwMutexRec {
        if !self.lwmutexes.iter().any(|m| m.work == work) {
            self.lwmutexes.push(LwMutexRec {
                work,
                waiters: Vec::new(),
                meta: Default::default(),
            });
        }
        self.lwmutexes
            .iter_mut()
            .find(|m| m.work == work)
            .expect("just inserted")
    }

    pub(super) fn lwmutex_register(&mut self, w: &mut dyn GuestWords, work: u32) {
        let _ = self.lwmutex_rec(work);
        lwwork::init(w, work);
    }

    pub(super) fn lwmutex_set_meta(&mut self, work: u32, name: String, attr: u32, init: i32) {
        self.lwmutex_rec(work).meta = (name, attr, init);
    }

    pub(super) fn lwmutex_meta(&self, work: u32) -> Option<(String, u32, i32, usize)> {
        self.lwmutexes
            .iter()
            .find(|m| m.work == work)
            .map(|m| (m.meta.0.clone(), m.meta.1, m.meta.2, m.waiters.len()))
    }

    pub(super) fn lwmutex_is_known(&self, work: u32) -> bool {
        self.lwmutexes.iter().any(|m| m.work == work)
    }

    /// Whether any thread is parked on the lightweight mutex at `work`.
    fn lwmutex_queued(&self, work: u32) -> bool {
        self.lwmutexes
            .iter()
            .any(|m| m.work == work && !m.waiters.is_empty())
    }

    pub(super) fn lwmutex_lock(
        &mut self,
        w: &mut dyn GuestWords,
        work: u32,
        cur: i32,
        deadline: Option<u64>,
    ) -> bool {
        if lwwork::fast_lock(w, work, cur, 1) {
            return true;
        }
        if self.lwmutex_take_for(w, work, cur) {
            return true;
        }
        let m = self.lwmutex_rec(work);
        m.waiters.push(LwMutexWaiter {
            thid: cur,
            deadline,
        });
        let parked = m.waiters.len();
        lwwork::set_waiters(w, work, parked);
        false
    }

    /// Take the lightweight mutex at `work` for `thid` if it is free or already `thid`'s, or
    /// else mark it [`lwwork::CONTENDED`] so its owner's release comes to the host - in which
    /// case the caller parks `thid`. Returns whether it was taken.
    ///
    /// >>> A LOOP, because under the parallel run the owner releases and recurses INLINE on
    /// another worker while this runs (`InlineOp::LwMutexLockSmp`): every decision is a
    /// compare-and-swap from the word just read, and a lost swap means "decide again". The
    /// park in particular happens only if the mark succeeded from a HELD value - parking on a
    /// mutex released in between would wait for a release that already happened. On a
    /// single-baton backing every swap succeeds and this is the old one-pass decision.
    ///
    /// Taking a free mutex past parked threads (`queued`) is the host's barge, as before.
    fn lwmutex_take_for(&mut self, w: &mut dyn GuestWords, work: u32, thid: i32) -> bool {
        let queued = self.lwmutex_queued(work);
        loop {
            let raw = lwwork::count_word(w, work);
            let held = raw & !lwwork::CONTENDED;
            if held == 0 {
                if lwwork::take_free(w, work, thid, queued) {
                    return true;
                }
                continue;
            }
            if lwwork::owner(w, work) == thid {
                // `thid` is not running guest code (it is this call's caller, or parked), and
                // only the owner changes a held count - a plain store is exact.
                lwwork::set_held(w, work, held + 1, queued);
                return true;
            }
            if raw & lwwork::CONTENDED != 0 || lwwork::mark_contended(w, work, raw) {
                return false;
            }
        }
    }

    pub(super) fn lwmutex_try_lock(&mut self, w: &mut dyn GuestWords, work: u32, cur: i32) -> bool {
        if lwwork::fast_lock(w, work, cur, 1) {
            return true;
        }
        let queued = self.lwmutex_queued(work);
        loop {
            let raw = lwwork::count_word(w, work);
            let held = raw & !lwwork::CONTENDED;
            if held == 0 {
                if lwwork::take_free(w, work, cur, queued) {
                    return true;
                }
                continue;
            }
            if lwwork::owner(w, work) == cur {
                lwwork::set_held(w, work, held + 1, queued);
                return true;
            }
            return false;
        }
    }

    pub(super) fn lwmutex_unlock(&mut self, cx: &mut Sched, w: &mut dyn GuestWords, work: u32) {
        let held = lwwork::count(w, work);
        if held == 0 {
            // An unlock with nothing held. The old host-side record silently ignored this
            // and so does this, but say so: it is either a title bug or a mutex we failed
            // to resolve, and both are worth seeing when a deadlock is being read.
            tracing::debug!(
                target: "vitaslop::sema",
                work = format_args!("{work:#010x}"),
                thread = format_args!("{:#x}", cx.current),
                "sceKernelUnlockLwMutex on a mutex nothing holds"
            );
            return;
        }
        let remaining = held - 1;
        if remaining != 0 {
            let queued = self.lwmutex_queued(work);
            lwwork::set_held(w, work, remaining, queued);
            return;
        }
        // Fully released. Hand it straight to the next parked thread rather than freeing
        // it - a woken waiter that had to race for it could lose to a barging inline take
        // and park again, forever.
        let (next, still_queued) = {
            let m = self.lwmutex_rec(work);
            let next = (!m.waiters.is_empty()).then(|| m.waiters.remove(0).thid);
            lwwork::set_waiters(w, work, m.waiters.len());
            (next, !m.waiters.is_empty())
        };
        match next {
            Some(thid) => {
                lwwork::hand_to(w, work, thid, still_queued);
                wake!(cx, thid);
            }
            None => lwwork::release(w, work),
        }
    }

    pub(super) fn lwmutex_delete(&mut self, w: &mut dyn GuestWords, work: u32) {
        lwwork::clear(w, work);
        self.lwmutexes.retain(|m| m.work != work);
    }

    /// Acquire the lightweight mutex at `work` on behalf of thread `thid` (not the
    /// current thread). Used when a lightweight-cond signal/timeout transfers a waiter
    /// back to its mutex: if free the thread takes it and is woken now, else it queues
    /// behind the owner and is woken when the owner unlocks. Work-keyed twin of
    /// [`mutex_acquire_for`](Self::mutex_acquire_for).
    fn lwmutex_acquire_for(
        &mut self,
        cx: &mut Sched,
        w: &mut dyn GuestWords,
        work: u32,
        thid: i32,
    ) {
        if self.lwmutex_take_for(w, work, thid) {
            wake!(cx, thid);
            return;
        }
        let m = self.lwmutex_rec(work);
        // A cond waiter handed back to its mutex waits for the OWNER, not for a clock: its
        // own timeout already expired (that is why it is here), so it carries none.
        m.waiters.push(LwMutexWaiter {
            thid,
            deadline: None,
        });
        let parked = m.waiters.len();
        lwwork::set_waiters(w, work, parked);
        // Woken later, when the owner unlocks.
    }

    pub(super) fn resolve_deferred_lwmutex(
        &mut self,
        cx: &mut Sched,
        w: &mut dyn GuestWords,
    ) -> usize {
        let queued = std::mem::take(&mut self.pending_lwmutex_acquires);
        for &(work, thid) in &queued {
            self.lwmutex_acquire_for(cx, w, work, thid);
        }
        let heavy = std::mem::take(&mut self.pending_mutex_acquires);
        for &(uid, thid) in &heavy {
            self.mutex_acquire_for(cx, w, uid, thid);
        }
        queued.len() + heavy.len()
    }

    // --- condition variables ---

    pub(super) fn create_cond(&mut self, uid: i32, mutex_uid: i32) {
        self.conds.push(CondRec {
            uid,
            mutex: mutex_uid,
            waiters: Vec::new(),
        });
    }

    pub(super) fn cond_exists(&self, uid: i32) -> bool {
        self.conds.iter().any(|c| c.uid == uid)
    }

    pub(super) fn cond_wait(
        &mut self,
        cx: &mut Sched,
        w: &mut dyn GuestWords,
        uid: i32,
        deadline: Option<u64>,
    ) {
        let Some(mutex) = self.conds.iter().find(|c| c.uid == uid).map(|c| c.mutex) else {
            return;
        };
        self.mutex_unlock(cx, w, mutex);
        let cur = cx.current;
        if let Some(c) = self.conds.iter_mut().find(|c| c.uid == uid) {
            c.waiters.push(CondWaiter {
                thid: cur,
                deadline,
            });
        }
    }

    pub(super) fn cond_signal(
        &mut self,
        cx: &mut Sched,
        w: &mut dyn GuestWords,
        uid: i32,
        all: bool,
    ) {
        let (mutex, woken) = {
            let Some(c) = self.conds.iter_mut().find(|c| c.uid == uid) else {
                return;
            };
            let woken: Vec<i32> = if all {
                std::mem::take(&mut c.waiters)
                    .into_iter()
                    .map(|w| w.thid)
                    .collect()
            } else if c.waiters.is_empty() {
                Vec::new()
            } else {
                vec![c.waiters.remove(0).thid]
            };
            (c.mutex, woken)
        };
        for thid in woken {
            self.mutex_acquire_for(cx, w, mutex, thid);
        }
    }

    // --- lightweight condition variables ---

    pub(super) fn lwcond_bind_mutex(&mut self, cond_work: u32, mutex_work: u32) {
        self.lwcond_mutex.retain(|&(c, _)| c != cond_work);
        self.lwcond_mutex.push((cond_work, mutex_work));
    }

    /// The lightweight mutex work address bound to lightweight cond `cond_work`, if
    /// its `CreateLwCond` was seen.
    pub(super) fn lwcond_mutex_of(&self, cond_work: u32) -> Option<u32> {
        self.lwcond_mutex
            .iter()
            .find(|&&(c, _)| c == cond_work)
            .map(|&(_, m)| m)
    }

    pub(super) fn lwcond_wait(
        &mut self,
        cx: &mut Sched,
        w: &mut dyn GuestWords,
        work: u32,
        deadline: Option<u64>,
    ) -> bool {
        let Some(mutex_work) = self.lwcond_mutex_of(work) else {
            return false;
        };
        self.lwmutex_unlock(cx, w, mutex_work);
        self.lwcond_waiters.push((cx.current, work, deadline));
        true
    }

    pub(super) fn lwcond_signal(
        &mut self,
        cx: &mut Sched,
        w: &mut dyn GuestWords,
        work: u32,
        all: bool,
    ) {
        let mut woke_one = false;
        let mut woken: Vec<i32> = Vec::new();
        self.lwcond_waiters.retain(|&(thid, w, _)| {
            if w == work && (all || !woke_one) {
                woken.push(thid);
                woke_one = true;
                false
            } else {
                true
            }
        });
        match self.lwcond_mutex_of(work) {
            Some(mutex_work) => {
                for thid in woken {
                    self.lwmutex_acquire_for(cx, w, mutex_work, thid);
                }
            }
            // No bound mutex recorded (a bare cond): just make the waiters runnable.
            None => {
                for t in woken {
                    wake!(cx, t);
                }
            }
        }
    }

    /// Drop `thid` from the lightweight-cond wait list - its thread is gone.
    pub(super) fn forget_lwcond_waiter(&mut self, thid: i32) {
        self.lwcond_waiters.retain(|&(t, _, _)| t != thid);
    }

    // --- event flags and simple events ---

    pub(super) fn create_event_flag(&mut self, uid: i32, init: u32) {
        self.event_flags.push((uid, init));
    }

    pub(super) fn name_event_flag(&mut self, uid: i32, name: &str) {
        if !name.is_empty() {
            self.event_flag_names.insert(uid, name.to_string());
        }
    }

    pub(super) fn record_event_flag_params(&mut self, uid: i32, attr: u32, init: u32) {
        self.event_flag_params.insert(uid, (attr, init));
    }

    pub(super) fn event_flag_info(&self, uid: i32) -> Option<(u32, u32, u32, u32)> {
        let &(_, current) = self.event_flags.iter().find(|(u, _)| *u == uid)?;
        let (attr, init) = self.event_flag_params.get(&uid).copied().unwrap_or((0, 0));
        let waiters = self.evf_waiters.iter().filter(|w| w.uid == uid).count() as u32;
        Some((attr, init, current, waiters))
    }

    pub(super) fn event_flag_name(&self, uid: i32) -> &str {
        self.event_flag_names
            .get(&uid)
            .map(String::as_str)
            .unwrap_or("")
    }

    pub(super) fn event_set(&mut self, uid: i32, bits: u32) {
        if let Some((_, p)) = self.event_flags.iter_mut().find(|(u, _)| *u == uid) {
            *p |= bits;
        }
    }

    /// Whether `pattern` satisfies a wait for `bits` under `mode`.
    fn evf_satisfied(pattern: u32, bits: u32, mode: u32) -> bool {
        if mode & Self::EVF_WAITOR != 0 {
            pattern & bits != 0
        } else {
            pattern & bits == bits
        }
    }

    pub(super) fn evf_exists(&self, uid: i32) -> bool {
        self.event_flags.iter().any(|(u, _)| *u == uid)
    }

    pub(super) fn evf_try_wait(&mut self, uid: i32, bits: u32, mode: u32) -> Option<u32> {
        let (_, p) = self.event_flags.iter_mut().find(|(u, _)| *u == uid)?;
        if !Self::evf_satisfied(*p, bits, mode) {
            return None;
        }
        let at_match = *p;
        if mode & Self::EVF_WAITCLEAR != 0 {
            *p = 0;
        } else if mode & Self::EVF_WAITCLEAR_PAT != 0 {
            *p &= !bits;
        }
        Some(at_match)
    }

    /// Park `thid` on event flag `uid`. `user_data_addr` is non-zero only for a simple event.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evf_block(
        &mut self,
        uid: i32,
        thid: i32,
        bits: u32,
        mode: u32,
        out_addr: u32,
        deadline: Option<u64>,
        user_data_addr: u32,
    ) {
        self.evf_waiters.push(EvfWaiter {
            uid,
            thid,
            bits,
            mode,
            out_addr,
            deadline,
            user_data_addr,
        });
    }

    pub(super) fn event_set_wake(&mut self, cx: &mut Sched, uid: i32, bits: u32) {
        self.event_set(uid, bits);
        loop {
            let pattern = self.event_pattern(uid);
            let next = self
                .evf_waiters
                .iter()
                .position(|w| w.uid == uid && Self::evf_satisfied(pattern, w.bits, w.mode));
            let Some(idx) = next else { break };
            let w = self.evf_waiters.remove(idx);
            let at_match = self
                .evf_try_wait(uid, w.bits, w.mode)
                .expect("matched waiter satisfies its own condition");
            if w.out_addr != 0 {
                cx.stat_write(w.out_addr, at_match);
            }
            if w.user_data_addr != 0 {
                let d = self.simple_events.get(&uid).map_or(0, |e| e.1);
                cx.stat_write(w.user_data_addr, d as u32);
                cx.stat_write(w.user_data_addr + 4, (d >> 32) as u32);
            }
            wake!(cx, w.thid);
        }
    }

    pub(super) fn event_clear(&mut self, uid: i32, bits: u32) {
        if let Some((_, p)) = self.event_flags.iter_mut().find(|(u, _)| *u == uid) {
            *p &= bits;
        }
    }

    pub(super) fn event_pattern(&self, uid: i32) -> u32 {
        self.event_flags
            .iter()
            .find(|(u, _)| *u == uid)
            .map(|(_, p)| *p)
            .unwrap_or(0)
    }

    pub(super) fn create_simple_event(&mut self, uid: i32, attr: u32, init: u32) {
        self.create_event_flag(uid, init);
        self.simple_events.insert(uid, (attr, 0));
    }

    pub(super) fn simple_event_mode(&self, uid: i32) -> Option<u32> {
        self.simple_events.get(&uid).map(|(attr, _)| {
            Self::EVF_WAITOR
                | if attr & Self::SIMPLE_EVENT_AUTO_RESET != 0 {
                    Self::EVF_WAITCLEAR
                } else {
                    0
                }
        })
    }

    pub(super) fn set_simple_event_data(&mut self, uid: i32, data: u64) {
        if let Some(e) = self.simple_events.get_mut(&uid) {
            e.1 = data;
        }
    }

    pub(super) fn simple_event_data(&self, uid: i32) -> u64 {
        self.simple_events.get(&uid).map_or(0, |e| e.1)
    }

    pub(super) fn delete_simple_event(&mut self, uid: i32) -> bool {
        if self.simple_events.remove(&uid).is_none() {
            return false;
        }
        self.event_flags.retain(|(u, _)| *u != uid);
        true
    }

    // --- message pipes ---

    pub(super) fn msg_pipe_create(&mut self, uid: i32) {
        self.msg_pipes
            .insert(uid, std::collections::VecDeque::new());
    }

    pub(super) fn msg_pipe_remove(&mut self, uid: i32) -> bool {
        self.msg_pipes.remove(&uid).is_some()
    }

    pub(super) fn msg_pipe(&self, uid: i32) -> Option<&std::collections::VecDeque<u8>> {
        self.msg_pipes.get(&uid)
    }

    pub(super) fn msg_pipe_mut(&mut self, uid: i32) -> Option<&mut std::collections::VecDeque<u8>> {
        self.msg_pipes.get_mut(&uid)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn msgpipe_block(
        &mut self,
        uid: i32,
        thid: i32,
        buf: u32,
        size: u32,
        full: bool,
        result: u32,
        deadline: Option<u64>,
    ) {
        self.msgpipe_waiters.push(MsgPipeWaiter {
            uid,
            thid,
            buf,
            size,
            full,
            result,
            deadline,
        });
    }

    pub(super) fn msgpipe_release(&mut self, cx: &mut Sched, uid: i32, code: u32) -> u32 {
        let mut n = 0;
        let mut i = 0;
        while i < self.msgpipe_waiters.len() {
            if self.msgpipe_waiters[i].uid == uid {
                let w = self.msgpipe_waiters.remove(i);
                wake!(cx, w.thid);
                cx.resume_code(w.thid, code);
                n += 1;
            } else {
                i += 1;
            }
        }
        n
    }

    pub(super) fn msgpipe_service(&mut self, cx: &mut Sched, ctx: &mut GuestCtx, uid: i32) {
        loop {
            let held = self.msg_pipes.get(&uid).map_or(0, |p| p.len());
            let Some(idx) = self.msgpipe_waiters.iter().position(|x| x.uid == uid) else {
                break;
            };
            let x = &self.msgpipe_waiters[idx];
            let ok = if x.full {
                held >= x.size as usize
            } else {
                held > 0
            };
            if !ok {
                break;
            }
            let x = self.msgpipe_waiters.remove(idx);
            let n = (x.size as usize).min(held);
            let bytes: Vec<u8> = self
                .msg_pipes
                .get_mut(&uid)
                .expect("held > 0")
                .drain(..n)
                .collect();
            ctx.write_bytes(x.buf, &bytes);
            if x.result != 0 {
                ctx.write_u32(x.result, n as u32);
            }
            wake!(cx, x.thid);
        }
    }

    // --- time: deadlines and expiry ---

    /// The earliest pending timed-wake deadline among the sync waits, with who waits on it -
    /// `sleepers` (the host's pure sleeps) slotted in right after the lightweight-cond waits.
    /// That position is not cosmetic: `min_by_key` keeps the FIRST of equal deadlines, and only
    /// one wait may be credited for an idle jump, so the order decides the attribution.
    pub(super) fn earliest_deadline_owner(
        &self,
        sleepers: impl Iterator<Item = (u64, IdleOwner)>,
    ) -> Option<(u64, IdleOwner)> {
        let own = |kind: IdleKind, thid: i32| IdleOwner { kind, thid };
        let lw = self
            .lwcond_waiters
            .iter()
            .filter_map(|&(t, _, d)| d.map(|d| (d, own(IdleKind::LwCond, t))));
        let ev = self
            .evf_waiters
            .iter()
            .filter_map(|w| w.deadline.map(|d| (d, own(IdleKind::EventFlag, w.thid))));
        let sem = self
            .sema_waiters
            .iter()
            .filter_map(|w| w.deadline.map(|d| (d, own(IdleKind::Sema, w.thid))));
        let cnd = self.conds.iter().flat_map(|c| {
            c.waiters
                .iter()
                .filter_map(|w| w.deadline.map(|d| (d, own(IdleKind::Cond, w.thid))))
        });
        // A timed lightweight-mutex lock is a real deadline like any other: omitted here, the
        // scheduler would refuse to jump the clock for it and a run where every other thread
        // is parked would report a DEADLOCK instead of letting the lock time out.
        let lwm = self.lwmutexes.iter().flat_map(|m| {
            m.waiters
                .iter()
                .filter_map(|x| x.deadline.map(|d| (d, own(IdleKind::LwMutex, x.thid))))
        });
        // `min_by_key` on the deadline alone: two waits can share a deadline and either is a
        // correct answer for the JUMP, but only one may be credited for the time or the
        // attribution double-counts.
        let mpp = self
            .msgpipe_waiters
            .iter()
            .filter_map(|w| w.deadline.map(|d| (d, own(IdleKind::MsgPipe, w.thid))));
        let rw = self.rwlocks.iter().flat_map(|l| {
            l.waiters
                .iter()
                .filter_map(|x| x.deadline.map(|d| (d, own(IdleKind::RwLock, x.thid))))
        });
        lw.chain(sleepers)
            .chain(ev)
            .chain(sem)
            .chain(cnd)
            .chain(lwm)
            .chain(mpp)
            .chain(rw)
            .min_by_key(|&(d, _)| d)
    }

    /// The FIRST half of the timed-wait expiry at `cx.now`: lightweight conds, lightweight
    /// mutexes and reader/writer locks. The host's pure sleeps expire between this and
    /// [`expire_waits`](Self::expire_waits), which keeps the wake order the scheduler has
    /// always seen.
    pub(super) fn expire_locks(&mut self, cx: &mut Sched, timeout_wakes: &mut Vec<i32>) {
        let now = cx.now;
        // Timed lightweight-cond waits: like the heavyweight cond, a timed-out
        // WaitLwCond re-acquires its bound mutex before resuming, so collect the
        // expirees, then hand each to its mutex (or wake directly if none is bound).
        let mut expired_lw: Vec<(i32, u32)> = Vec::new(); // (thid, cond work address)
        self.lwcond_waiters
            .retain(|&(thid, work, deadline)| match deadline {
                Some(d) if d <= now => {
                    expired_lw.push((thid, work));
                    false
                }
                _ => true,
            });
        for (thid, work) in expired_lw {
            timeout_wakes.push(thid);
            cx.resume_code(thid, SCE_KERNEL_ERROR_WAIT_TIMEOUT);
            match self.lwcond_mutex_of(work) {
                // No guest memory is reachable here (see `resolve_deferred_lwmutex`), and
                // deciding this needs to READ the work area, so queue the intent and let
                // the scheduler settle it before anything resumes.
                Some(mutex_work) => self.pending_lwmutex_acquires.push((mutex_work, thid)),
                None => wake!(cx, thid),
            }
        }
        // Timed lightweight-MUTEX locks whose deadline passed. Unlike a cond wait there is
        // nothing to re-acquire - the whole point is that the lock was never taken - so the
        // thread is simply woken carrying WAIT_TIMEOUT, which is what
        // `sceKernelLockLwMutex` returns when its timeout expires. The parked COUNT in the
        // work area is refreshed by the next unlock; it is a hint for the inline fast path
        // (non-zero means "go to the host"), and leaving it high only costs a crossing.
        for m in self.lwmutexes.iter_mut() {
            m.waiters.retain(|x| match x.deadline {
                Some(d) if d <= now => {
                    wake!(cx, x.thid);
                    cx.resume_code(x.thid, SCE_KERNEL_ERROR_WAIT_TIMEOUT);
                    false
                }
                _ => true,
            });
        }
        // Timed reader/writer lock waits: like the lightweight mutex, nothing was taken, so
        // the thread is woken carrying WAIT_TIMEOUT. Its leaving the queue can let readers
        // queued behind it in (a parked WRITER holds back later readers), so each lock that
        // lost a waiter is re-granted.
        let mut rw_touched = Vec::new();
        for l in self.rwlocks.iter_mut() {
            let before = l.waiters.len();
            l.waiters.retain(|x| match x.deadline {
                Some(d) if d <= now => {
                    wake!(cx, x.thid);
                    cx.resume_code(x.thid, SCE_KERNEL_ERROR_WAIT_TIMEOUT);
                    false
                }
                _ => true,
            });
            if l.waiters.len() != before {
                rw_touched.push(l.uid);
            }
        }
        for uid in rw_touched {
            self.rwlock_grant(cx, uid);
        }
    }

    /// The SECOND half of the timed-wait expiry: semaphores, heavyweight conds, message pipes
    /// and event flags. `clock` is `(quanta, wall floor, idle jumps)` now and `frame` the
    /// display frame, for the line a timed-out semaphore wait leaves in [`timed_out_waits`].
    pub(super) fn expire_waits(
        &mut self,
        cx: &mut Sched,
        timeout_wakes: &mut Vec<i32>,
        clock: (u64, u64, u64),
        frame: u64,
    ) {
        let now = cx.now;
        // Timed semaphore waits whose deadline passed: wake with WAIT_TIMEOUT. The
        // count is untouched (nothing was available to consume).
        let (q, wl, idl) = clock;
        self.sema_waiters.retain(|w| match w.deadline {
            Some(d) if d <= now => {
                wake!(cx, w.thid);
                cx.resume_code(w.thid, SCE_KERNEL_ERROR_WAIT_TIMEOUT);
                note_timed_out_wait(format!(
                    "f{frame} sema {:#x} thread {:#x}: timeout {} us expired after {} us of guest time = quanta {} + wall floor {} + idle jumps {} (+ other {})",
                    w.uid,
                    w.thid,
                    d.saturating_sub(w.began.0),
                    now.saturating_sub(w.began.0),
                    q.saturating_sub(w.began.1),
                    wl.saturating_sub(w.began.2),
                    idl.saturating_sub(w.began.3),
                    now.saturating_sub(w.began.0)
                        .saturating_sub(q.saturating_sub(w.began.1) + wl.saturating_sub(w.began.2) + idl.saturating_sub(w.began.3)),
                ));
                false
            }
            _ => true,
        });
        // Timed heavyweight cond waits whose deadline passed: a timed-out WaitCond must
        // re-acquire its mutex before returning, so collect the expirees first, then
        // hand each to its mutex (taken now if free, else queued behind the owner). The
        // WAIT_TIMEOUT code is delivered whenever the thread ultimately resumes.
        let mut expired_cond: Vec<(i32, i32)> = Vec::new(); // (thid, mutex uid)
        for c in self.conds.iter_mut() {
            let mutex = c.mutex;
            c.waiters.retain(|w| match w.deadline {
                Some(d) if d <= now => {
                    expired_cond.push((w.thid, mutex));
                    false
                }
                _ => true,
            });
        }
        for (thid, mutex) in expired_cond {
            cx.resume_code(thid, SCE_KERNEL_ERROR_WAIT_TIMEOUT);
            // No guest memory is reachable here and re-acquiring now needs to READ the mutex
            // table, so the intent is queued and the scheduler settles it before anything
            // resumes - the same deferral `pending_lwmutex_acquires` already makes, and for
            // the same reason. See `resolve_deferred_lwmutex`.
            self.pending_mutex_acquires.push((mutex, thid));
            timeout_wakes.push(thid);
        }
        // Timed event flag waits whose deadline passed: wake with WAIT_TIMEOUT and the
        // CURRENT pattern written through outBits (the caller reads the pattern back
        // and re-checks; see `vita::sync::wait_event_flag`).
        // Timed message-pipe receives: wake with WAIT_TIMEOUT and nothing received.
        let mut expired_mpp = Vec::new();
        self.msgpipe_waiters.retain(|w| match w.deadline {
            Some(d) if d <= now => {
                expired_mpp.push((w.thid, w.result));
                false
            }
            _ => true,
        });
        for (thid, result) in expired_mpp {
            if result != 0 {
                cx.stat_write(result, 0);
            }
            wake!(cx, thid);
            cx.resume_code(thid, SCE_KERNEL_ERROR_WAIT_TIMEOUT);
        }
        let patterns = &self.event_flags;
        self.evf_waiters.retain(|w| match w.deadline {
            Some(d) if d <= now => {
                if w.out_addr != 0 {
                    let p = patterns
                        .iter()
                        .find(|(u, _)| *u == w.uid)
                        .map(|(_, p)| *p)
                        .unwrap_or(0);
                    cx.stat_write(w.out_addr, p);
                }
                wake!(cx, w.thid);
                cx.resume_code(w.thid, SCE_KERNEL_ERROR_WAIT_TIMEOUT);
                false
            }
            _ => true,
        });
    }

    // --- reports ---

    /// What thread `thid` is parked on among the sync objects, as one line - `None` when it
    /// is in none of their waiter lists. The order is the stall dump's, which reports the
    /// first list a thread appears in.
    pub(super) fn wait_state(&self, thid: i32) -> Option<String> {
        for m in &self.lwmutexes {
            if m.waiters.iter().any(|x| x.thid == thid) {
                // No owner here on purpose: it lives in the guest's work area now, and
                // this dump has no guest memory. Naming the address is enough to go and
                // read it (`VITASLOP_PEEK=<work>:16`), and inventing a host-side echo of
                // it would print a value the inline take never updated - a wrong answer
                // in the one report that exists to explain a deadlock.
                return Some(format!(
                    "blocked on lwmutex work={:#010x} ({} parked; owner is in the work area)",
                    m.work,
                    m.waiters.len()
                ));
            }
        }
        for m in &self.mutexes {
            if m.waiters.contains(&thid) {
                return Some(format!(
                    "blocked on mutex uid={:#x} (owner={:?})",
                    m.uid, m.owner
                ));
            }
        }
        for l in &self.rwlocks {
            if let Some(w) = l.waiters.iter().find(|w| w.thid == thid) {
                return Some(format!(
                    "blocked on rwlock uid={:#x} {:?} for {} (writer={:?} readers={:x?})",
                    l.uid,
                    l.name,
                    if w.write { "WRITE" } else { "READ" },
                    l.writer,
                    l.readers
                ));
            }
        }
        if let Some(w) = self.sema_waiters.iter().find(|w| w.thid == thid) {
            // The semaphore's own state beside the wait: a waiter parked while the count already
            // covers its need is a LOST WAKE-UP, and the line alone could not say so.
            let state = self
                .sema_info(w.uid)
                .map(|(name, _, _, count, max, waiting)| {
                    format!(" ({name:?} count={count} max={max} waiting={waiting})")
                })
                .unwrap_or_else(|| " (NO SUCH SEMAPHORE)".into());
            return Some(format!(
                "blocked on sema uid={:#x} need={}{state}",
                w.uid, w.need
            ));
        }
        if let Some(w) = self.msgpipe_waiters.iter().find(|w| w.thid == thid) {
            return Some(format!(
                "blocked on msgpipe uid={:#x} want={} ({}) holding={}",
                w.uid,
                w.size,
                if w.full { "FULL" } else { "ASAP" },
                self.msg_pipes.get(&w.uid).map_or(0, |p| p.len())
            ));
        }
        if let Some(w) = self.evf_waiters.iter().find(|w| w.thid == thid) {
            let pattern = self
                .event_flags
                .iter()
                .find(|(u, _)| *u == w.uid)
                .map(|(_, p)| *p)
                .unwrap_or(0);
            return Some(format!(
                "blocked on eventflag uid={:#x}{} want={:#x} mode={:#x} have={:#x}{}",
                w.uid,
                match self.event_flag_name(w.uid) {
                    "" => String::new(),
                    n => format!(" {n:?}"),
                },
                w.bits,
                w.mode,
                pattern,
                match w.deadline {
                    Some(d) => format!(" deadline_us={d}"),
                    None => String::new(),
                }
            ));
        }
        for c in &self.conds {
            if c.waiters.iter().any(|w| w.thid == thid) {
                return Some(format!(
                    "blocked on cond uid={:#x} (mutex uid={:#x})",
                    c.uid, c.mutex
                ));
            }
        }
        if let Some((_, work, deadline)) = self.lwcond_waiters.iter().find(|(t, _, _)| *t == thid) {
            return Some(format!(
                "blocked on lwcond work={work:#010x} deadline={deadline:?}"
            ));
        }
        None
    }

    /// Append every sync object's ownership and waiter state to the stall dump `s`. With
    /// `read_word`, the lightweight mutexes' guest-resident owner and count are printed too.
    pub(super) fn dump_into(&self, s: &mut String, read_word: Option<&dyn Fn(u32) -> u32>) {
        use std::fmt::Write;
        let _ = writeln!(s, "lwmutexes ({}):", self.lwmutexes.len());
        for m in &self.lwmutexes {
            // Owner and count are guest-resident (`vita::lwwork`). WITH a reader they are
            // printed here, which is the line that names the thread holding a mutex a stalled
            // one is parked on; without, `VITASLOP_PEEK=<work>:16` reads them in that order.
            //
            // The owner is only meaningful while the COUNT is non-zero - thid 0 is the main
            // thread, so no owner value can spell "nobody" - which is why both are printed
            // and a count of zero says FREE outright rather than leaving a stale owner to be
            // read as the holder.
            match read_word {
                Some(read) => {
                    let (owner, count) = (
                        read(m.work + crate::vita::lwwork::off::OWNER),
                        read(m.work + crate::vita::lwwork::off::COUNT),
                    );
                    let held = if count == 0 {
                        "FREE".to_string()
                    } else {
                        format!("held by thid={owner:#x} count={count}")
                    };
                    let _ = writeln!(s, "  work={:#010x} {held} parked={:x?}", m.work, m.waiters);
                }
                None => {
                    let _ = writeln!(s, "  work={:#010x} parked={:x?}", m.work, m.waiters);
                }
            }
        }
        let _ = writeln!(s, "mutexes ({}):", self.mutexes.len());
        for m in &self.mutexes {
            let _ = writeln!(
                s,
                "  uid={:#x} owner={:?} count={} waiters={:x?}",
                m.uid, m.owner, m.count, m.waiters
            );
        }
        let _ = writeln!(s, "rwlocks ({}):", self.rwlocks.len());
        for l in &self.rwlocks {
            let waiters: Vec<(i32, bool)> = l.waiters.iter().map(|w| (w.thid, w.write)).collect();
            let _ = writeln!(
                s,
                "  uid={:#x} name={:?} writer={:?}x{} readers={:x?} waiters(thid,write)={waiters:x?}",
                l.uid, l.name, l.writer, l.write_count, l.readers
            );
        }
        let _ = writeln!(s, "semaphores ({}):", self.semaphores.len());
        for sem in &self.semaphores {
            let waiters: Vec<(i32, i32)> = self
                .sema_waiters
                .iter()
                .filter(|w| w.uid == sem.uid)
                .map(|w| (w.thid, w.need))
                .collect();
            let _ = writeln!(
                s,
                "  uid={:#x} name={:?} count={} waiters(thid,need)={waiters:x?}",
                sem.uid, sem.name, sem.count
            );
        }
        // Event flags were missing from this dump entirely, and they are what a title's
        // own worker threads actually park on - so a stall could show every printed
        // list empty while several threads were blocked.
        let _ = writeln!(s, "event flags ({}):", self.event_flags.len());
        for (uid, pattern) in &self.event_flags {
            let waiters: Vec<(i32, u32)> = self
                .evf_waiters
                .iter()
                .filter(|w| w.uid == *uid)
                .map(|w| (w.thid, w.bits))
                .collect();
            if !waiters.is_empty() {
                let _ = writeln!(
                    s,
                    "  uid={uid:#x} pattern={pattern:#x} waiters(thid,want)={waiters:x?}"
                );
            }
        }
        let _ = writeln!(s, "cond waiters:");
        for c in &self.conds {
            let w: Vec<i32> = c.waiters.iter().map(|x| x.thid).collect();
            if !w.is_empty() {
                let _ = writeln!(
                    s,
                    "  cond uid={:#x} mutex={:#x} waiters={:x?}",
                    c.uid, c.mutex, w
                );
            }
        }
        let _ = writeln!(
            s,
            "lwcond waiters (thid, cond_work, deadline): {:x?}",
            self.lwcond_waiters
        );
        let _ = writeln!(
            s,
            "lwcond->mutex bindings (cond_work, mutex_work): {:x?}",
            self.lwcond_mutex
        );
    }
}
