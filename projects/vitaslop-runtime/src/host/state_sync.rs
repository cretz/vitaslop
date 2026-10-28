use super::*;

// The kernel synchronization API the NID handlers call. The state and the logic are
// [`SyncObjects`]'s (`host/sync.rs`); every method here supplies what that type cannot see -
// the clock, the calling thread, a fresh uid, the scheduler's queues - and forwards.
impl VitaState {
    // --- semaphores ---

    /// Create a semaphore, returning its SceUID. `name`, `attr` and `max` are kept
    /// because they are observable: `sceKernelOpenSema` resolves the name and
    /// `sceKernelGetSemaInfo` reports all three.
    pub fn create_sema(&mut self, name: &str, attr: u32, init: i32, max: i32) -> i32 {
        let uid = self.new_uid();
        self.sync.create_sema(uid, name, attr, init, max);
        uid
    }

    /// Whether a semaphore with this uid currently exists.
    #[inline]
    pub fn sema_exists(&self, uid: i32) -> bool {
        self.sync.sema_exists(uid)
    }

    /// The SceUID of the live semaphore created with this name, if any.
    /// `sceKernelOpenSema` resolves a name to the existing object rather than making
    /// a new one, so this is the lookup behind it. First match wins, which matches a
    /// kernel that refuses a duplicate openable name.
    #[inline]
    pub fn sema_by_name(&self, name: &str) -> Option<i32> {
        self.sync.sema_by_name(name)
    }

    /// Everything `sceKernelGetSemaInfo` reports: `(name, attr, init, current, max,
    /// waiting threads)`. The waiter count is derived from the park queue rather than
    /// stored, so it cannot drift from the threads actually blocked.
    #[inline]
    pub fn sema_info(&self, uid: i32) -> Option<(&str, u32, i32, i32, i32, i32)> {
        self.sync.sema_info(uid)
    }

    /// Whether signalling `uid` by `n` would push its count past the MAXIMUM it was created
    /// with - `None` when no such semaphore exists.
    ///
    /// >>> A SEMAPHORE HAS A CEILING AND THE KERNEL ENFORCES IT.
    ///
    /// `sceKernelCreateSema` takes a `maxCount`, and a signal that would exceed it is REFUSED
    /// with `SCE_KERNEL_ERROR_SEMA_OVF`, leaving the count where it was. This engine used to
    /// add unconditionally, which turns a BINARY semaphore (`max = 1`, the shape a handshake
    /// between a producer and one consumer uses) into a counter: the consumer's next wait is
    /// then satisfied by a count the device would never have handed it, so it runs a second
    /// time on data that arrived once. That is not a stall, it is a DESYNC, and it surfaces
    /// wherever the two ends next expect to agree.
    #[inline]
    pub fn sema_signal_overflows(&self, uid: i32, n: i32) -> Option<bool> {
        self.sync.sema_signal_overflows(uid, n)
    }

    /// Try to take `need` from semaphore `uid` without blocking. Returns true if
    /// the count was available (and consumed), false otherwise.
    #[inline]
    pub fn sema_try_acquire(&mut self, uid: i32, need: i32) -> bool {
        self.sync.sema_try_acquire(uid, need)
    }

    /// Wait on a semaphore: take `n` from its count (never blocks in the single-
    /// thread model; the count floors at 0).
    #[inline]
    pub fn sema_wait(&mut self, uid: i32, n: i32) {
        self.sync.sema_wait(uid, n)
    }

    /// Signal a semaphore: add `n` to its count.
    #[inline]
    pub fn sema_signal(&mut self, uid: i32, n: i32) {
        self.sync.sema_signal(uid, n)
    }

    /// `sceKernelCancelSema`: release the waiters (by signalling their count, see
    /// `sync::cancel_sema`), then set the count. `None` when `uid` is not a semaphore.
    /// Returns how many waiters there were.
    pub fn sema_cancel(&mut self, uid: i32, set_count: i32) -> Option<i32> {
        let mut cx = sched!(self);
        self.sync.sema_cancel(&mut cx, uid, set_count)
    }

    /// Park the current thread waiting for `need` signals on semaphore `uid`.
    /// `timeout_us` of 0 is an infinite wait; non-zero arms a deadline at
    /// `now + timeout_us`, after which the wait is woken with
    /// `SCE_KERNEL_ERROR_WAIT_TIMEOUT` even if no signal arrives.
    pub fn sema_block(&mut self, uid: i32, need: i32, timeout_us: u32) {
        let deadline = (timeout_us != 0).then(|| self.virtual_us + timeout_us as u64);
        let began = (
            self.virtual_us,
            self.clock_from_quanta_us,
            self.clock_from_wall_us,
            self.clock_from_idle_us,
        );
        self.sync
            .sema_block(uid, self.current, need, deadline, began);
    }

    /// Signal semaphore `uid` by `n`, then release every parked waiter the new
    /// count can satisfy (in FIFO order), consuming the count for each. Preemptive
    /// counterpart of [`sema_signal`](Self::sema_signal).
    pub fn sema_signal_wake(&mut self, uid: i32, n: i32) {
        let mut cx = sched!(self);
        self.sync.sema_signal_wake(&mut cx, uid, n)
    }

    // --- heavyweight mutexes ---

    /// Publish where the KERNEL MUTEX TABLE lives, once the scheduler knows the host-mirror
    /// block's guest address. Before this the table address is zero, which
    /// [`crate::vita::kmutex::owns_entry`] reads as "no table" - so a mutex created earlier is
    /// host-resident and stays so, which is correct rather than merely safe.
    #[inline]
    pub fn set_mutex_table(&mut self, addr: u32) {
        self.sync.set_mutex_table(addr)
    }

    /// Where the KERNEL MUTEX TABLE lives, for the diagnostics that dump it.
    #[inline]
    pub fn mutex_table(&self) -> u32 {
        self.sync.mutex_table()
    }

    /// Create a recursive mutex, recording its state (preemptive ownership
    /// tracking), and return its SceUID.
    ///
    /// The uid claims its table entry here when that entry is free. A COLLISION - another live
    /// mutex already owns the entry this uid indexes - leaves the newcomer host-resident for
    /// its whole life, which is exactly what every kernel mutex was before the table existed.
    /// Refused rather than shared, because two mutexes in one entry would each read the
    /// other's owner.
    pub fn create_mutex(
        &mut self,
        w: &mut dyn GuestWords,
        name: &str,
        attr: u32,
        init: i32,
    ) -> i32 {
        let uid = self.new_uid();
        self.sync.create_mutex(w, uid, name, attr, init);
        uid
    }

    /// What `sceKernelGetMutexInfo` reports for `uid`: `(name, attr, init count, current
    /// count, owner thread or 0, waiting threads)`, from whichever home the mutex's ownership
    /// lives in. `None` for an unknown id.
    #[inline]
    pub fn mutex_info(
        &self,
        w: &dyn GuestWords,
        uid: i32,
    ) -> Option<(String, u32, i32, i32, i32, i32)> {
        self.sync.mutex_info(w, uid)
    }

    /// Lock mutex `uid` for the current thread. Returns true if acquired (free, or
    /// already held by this thread - recursive), false if the caller was parked
    /// behind the current owner (return [`SvcOutcome::Block`]).
    #[inline]
    pub fn mutex_lock(&mut self, w: &mut dyn GuestWords, uid: i32) -> bool {
        self.sync.mutex_lock(w, uid, self.current)
    }

    /// Whether locking mutex `uid` right now would contend (another thread owns
    /// it). Used by `sceKernelTryLockMutex`, which fails rather than blocks.
    #[inline]
    pub fn mutex_contended(&self, w: &dyn GuestWords, uid: i32) -> bool {
        self.sync.mutex_contended(w, uid, self.current)
    }

    /// Unlock mutex `uid`. When the recursion count reaches zero, hand ownership to
    /// the next parked waiter (FIFO) and wake it.
    pub fn mutex_unlock(&mut self, w: &mut dyn GuestWords, uid: i32) {
        let mut cx = sched!(self);
        self.sync.mutex_unlock(&mut cx, w, uid)
    }

    // --- reader/writer locks ---

    /// Create a reader/writer lock, free, and return its SceUID.
    pub fn create_rwlock(&mut self, name: &str, attr: u32) -> i32 {
        let uid = self.new_uid();
        self.sync.create_rwlock(uid, name, attr);
        uid
    }

    /// Destroy reader/writer lock `uid`. Its parked lockers are woken with
    /// `SCE_KERNEL_ERROR_WAIT_DELETE` - the object they waited on is gone and nothing else
    /// would ever release them. `false` for an unknown uid.
    pub fn delete_rwlock(&mut self, uid: i32) -> bool {
        let mut cx = sched!(self);
        self.sync.delete_rwlock(&mut cx, uid)
    }

    /// Take reader/writer lock `uid` for the current thread, for reading or writing, if it can
    /// be taken NOW. Never parks - [`Self::rwlock_block`] does that on [`RwLockResult::Busy`].
    ///
    /// A reader gets in while no OTHER thread writes and no writer is queued (a queued writer
    /// holds later readers back, or a steady stream of readers starves it). A thread that
    /// already holds the lock re-takes it in either mode when the lock is RECURSIVE - including
    /// a read under its own write, and a read while a writer is queued, which would otherwise
    /// deadlock the thread on itself. A writer gets in while nobody else holds it at all.
    #[inline]
    pub fn rwlock_try(&mut self, uid: i32, write: bool) -> RwLockResult {
        self.sync.rwlock_try(uid, write, self.current)
    }

    /// What `sceKernelGetRWLockInfo` reports for lock `uid`, all live state: `(name, attr,
    /// lockCount, writeOwnerId, numReadWaitThreads, numWriteWaitThreads)`. The count is the
    /// writer's recursion depth while one holds it, else every reader's depth summed (0 when
    /// free); the owner is 0 unless a thread holds it for writing. `None` for an unknown uid.
    #[inline]
    pub fn rwlock_info(&self, uid: i32) -> Option<(&str, u32, u32, i32, u32, u32)> {
        self.sync.rwlock_info(uid)
    }

    /// Park the current thread on reader/writer lock `uid` (`timeout_us` `None` = forever).
    pub fn rwlock_block(&mut self, uid: i32, write: bool, timeout_us: Option<u32>) {
        let deadline = timeout_us.map(|t| self.virtual_us + u64::from(t));
        self.sync.rwlock_block(uid, write, self.current, deadline)
    }

    /// Release one level of the current thread's hold on reader/writer lock `uid`, then let in
    /// whoever the release unblocks.
    pub fn rwlock_unlock(&mut self, uid: i32, write: bool) -> RwLockResult {
        let mut cx = sched!(self);
        self.sync.rwlock_unlock(&mut cx, uid, write)
    }

    // --- lightweight mutexes, whose state lives in the GUEST WORK AREA ---

    /// Register the lightweight mutex at `work` and lay its state out there
    /// (`sceKernelCreateLwMutex`). Idempotent as a record; the work area is re-initialized,
    /// which is what a re-create means.
    #[inline]
    pub fn lwmutex_register(&mut self, w: &mut dyn GuestWords, work: u32) {
        self.sync.lwmutex_register(w, work)
    }

    /// Record the create arguments of the lightweight mutex at `work`. See the record's `meta`.
    #[inline]
    pub fn lwmutex_set_meta(&mut self, work: u32, name: String, attr: u32, init: i32) {
        self.sync.lwmutex_set_meta(work, name, attr, init)
    }

    /// `(name, attr, initCount, parked waiters)` of a KNOWN lightweight mutex.
    #[inline]
    pub fn lwmutex_meta(&self, work: u32) -> Option<(String, u32, i32, usize)> {
        self.sync.lwmutex_meta(work)
    }

    /// Whether `work` is a lightweight mutex we have a record for (created, or already
    /// locked at least once). Used to resolve a possibly-copied work pointer.
    #[inline]
    pub fn lwmutex_is_known(&self, work: u32) -> bool {
        self.sync.lwmutex_is_known(work)
    }

    /// Adopt a work area no create was ever seen for, so it can be locked - and, from the
    /// next lock on, be taken inline.
    ///
    /// A zero identity stamp is unambiguous: a byte COPY of a created mutex carries the
    /// ORIGINAL's address there, so the only work area that can read zero is one no
    /// `sceKernelCreateLwMutex` ever touched. Statically-initialized mutexes are ordinary
    /// in libc, and this is the same lazy-by-address behaviour the host record has always
    /// had - just written down where the guest can see it.
    #[inline]
    pub fn lwmutex_adopt(&mut self, w: &mut dyn GuestWords, work: u32) {
        self.sync.lwmutex_register(w, work)
    }

    /// Lock the lightweight mutex at `work` for the current thread. Returns true if
    /// acquired (free, or already held by this thread - recursive), false if the caller
    /// was parked behind the owner (return [`SvcOutcome::Block`]).
    ///
    /// The uncontended answer comes from [`lwwork::fast_lock`], which is the same function
    /// the inline form is compiled from - so the two paths cannot decide differently about
    /// the same four words.
    #[inline]
    pub fn lwmutex_lock(
        &mut self,
        w: &mut dyn GuestWords,
        work: u32,
        timeout_us: Option<u32>,
    ) -> bool {
        let deadline = timeout_us.map(|us| self.virtual_us + us as u64);
        self.sync.lwmutex_lock(w, work, self.current, deadline)
    }

    /// Whether locking the lightweight mutex at `work` now would contend (another
    /// thread owns it). A snapshot: under the parallel run it can change the moment it is
    /// read, so a decision that acts on it is [`Self::lwmutex_try_lock`] instead.
    #[inline]
    pub fn lwmutex_contended(&self, w: &dyn GuestWords, work: u32) -> bool {
        lwwork::count(w, work) != 0 && lwwork::owner(w, work) != self.current
    }

    /// `sceKernelTryLockLwMutex`: take the lightweight mutex at `work` for the current thread
    /// if that needs no wait. ONE decision, never a park - a check followed by a separate take
    /// would let an inline take on another worker land in between and queue a caller that
    /// was promised it would not wait.
    #[inline]
    pub fn lwmutex_try_lock(&mut self, w: &mut dyn GuestWords, work: u32) -> bool {
        self.sync.lwmutex_try_lock(w, work, self.current)
    }

    /// Unlock the lightweight mutex at `work`. On full release, hand ownership to the
    /// next parked waiter (FIFO) and wake it.
    pub fn lwmutex_unlock(&mut self, w: &mut dyn GuestWords, work: u32) {
        let mut cx = sched!(self);
        self.sync.lwmutex_unlock(&mut cx, w, work)
    }

    /// Forget the lightweight mutex at `work` (`sceKernelDeleteLwMutex`), clearing the
    /// identity stamp so a guest that kept the pointer cannot take the stale work area
    /// inline.
    #[inline]
    pub fn lwmutex_delete(&mut self, w: &mut dyn GuestWords, work: u32) {
        self.sync.lwmutex_delete(w, work)
    }

    /// Resolve the deferred lightweight-mutex handoffs queued where no guest memory was
    /// reachable, and report how many were applied.
    ///
    /// A lightweight-cond wait that TIMES OUT must re-acquire its bound mutex before it
    /// runs, and that expiry is decided in [`advance_time_to`](Self::advance_time_to) - on
    /// the scheduler's idle path, with no host call in flight and so no [`GuestCtx`]
    /// anywhere. The decision needs to READ the work area (is it free?), which is why it
    /// cannot simply be queued as a write. So it is queued as an INTENT and settled here,
    /// by the scheduler, before anything is resumed.
    ///
    /// That is exact rather than nearly exact: no guest code runs between the expiry and
    /// this drain, so there is no moment at which a guest could observe the difference.
    pub fn resolve_deferred_lwmutex(&mut self, w: &mut dyn GuestWords) -> usize {
        let mut cx = sched!(self);
        self.sync.resolve_deferred_lwmutex(&mut cx, w)
    }

    // --- condition variables (preemptive mode) ---

    /// Create a condition variable bound to mutex `mutex_uid`.
    pub fn create_cond(&mut self, mutex_uid: i32) -> i32 {
        let uid = self.new_uid();
        self.sync.create_cond(uid, mutex_uid);
        uid
    }

    /// Whether a condition variable with this uid exists. The companion of
    /// [`Self::sema_exists`] and [`Self::evf_exists`]: `cond_wait` silently does NOTHING for an
    /// unknown uid while its caller parks the thread regardless, so without this the wait is a
    /// guaranteed permanent stall with no waiter recorded anywhere - invisible even to the
    /// watchdog's dump, which lists waiters.
    #[inline]
    pub fn cond_exists(&self, uid: i32) -> bool {
        self.sync.cond_exists(uid)
    }

    /// `sceKernelWaitCond`: release the associated mutex (handing it to any waiter)
    /// and park the current thread on the condition variable. The caller returns
    /// [`SvcOutcome::Block`]; on wake it has re-acquired the mutex (see
    /// [`cond_signal`](Self::cond_signal)) and the wait returns 0. `timeout_us` of 0
    /// is an infinite wait; non-zero arms a deadline after which the wait times out
    /// (still re-acquiring the mutex) and returns `SCE_KERNEL_ERROR_WAIT_TIMEOUT`.
    pub fn cond_wait(&mut self, w: &mut dyn GuestWords, uid: i32, timeout_us: u32) {
        let deadline = (timeout_us != 0).then(|| self.virtual_us + timeout_us as u64);
        let mut cx = sched!(self);
        self.sync.cond_wait(&mut cx, w, uid, deadline)
    }

    /// `sceKernelSignalCond`/`SignalCondAll`: wake one (or all) parked waiter. Each
    /// woken thread must re-acquire the condition's mutex before it runs, so it is
    /// handed to the mutex (taken now if free, else queued behind the owner).
    pub fn cond_signal(&mut self, w: &mut dyn GuestWords, uid: i32, all: bool) {
        let mut cx = sched!(self);
        self.sync.cond_signal(&mut cx, w, uid, all)
    }

    // --- lightweight condition variables ---

    /// Record a lightweight condition variable's associated lightweight mutex at
    /// `sceKernelCreateLwCond(cond_work, .., mutex_work, ..)`, so a later
    /// `sceKernelWaitLwCond` on `cond_work` knows which mutex to release and re-acquire.
    #[inline]
    pub fn lwcond_bind_mutex(&mut self, cond_work: u32, mutex_work: u32) {
        self.sync.lwcond_bind_mutex(cond_work, mutex_work)
    }

    /// Whether `cond_work` is a lightweight cond we recorded at `sceKernelCreateLwCond`
    /// (every created cond binds a mutex, so a recorded binding is its existence).
    #[inline]
    pub fn lwcond_is_known(&self, cond_work: u32) -> bool {
        self.sync.lwcond_mutex_of(cond_work).is_some()
    }

    /// Park the current thread in `sceKernelWaitLwCond` on cond `work`, atomically
    /// releasing its bound lightweight mutex (handing it to any waiter) - exactly like
    /// the heavyweight [`cond_wait`](Self::cond_wait). `timeout_us` of 0 is an infinite
    /// wait (only a signal wakes it); non-zero sets a deadline at `now + timeout_us`.
    /// On wake (signal or timeout) the thread re-acquires the mutex before it runs.
    ///
    /// Returns `false` if `work` is not a cond we recorded a `CreateLwCond` for: the
    /// kernel rejects an unknown lightweight cond (`SCE_KERNEL_ERROR_UNKNOWN_LW_COND_ID`)
    /// rather than releasing a mutex or blocking, so the caller must surface that error
    /// and the thread must NOT park. A cond with no binding reaching here means its
    /// creation was never observed upstream (the real bug lives there); faithful
    /// behavior here refuses to invent a binding.
    #[must_use]
    pub fn lwcond_wait(&mut self, w: &mut dyn GuestWords, work: u32, timeout_us: u32) -> bool {
        let deadline = (timeout_us != 0).then(|| self.virtual_us + timeout_us as u64);
        let mut cx = sched!(self);
        self.sync.lwcond_wait(&mut cx, w, work, deadline)
    }

    /// `sceKernelSignalLwCond`/`SignalLwCondAll`: wake one (or all) threads parked on
    /// cond `work`. Each woken thread must re-acquire the cond's bound lightweight mutex
    /// before it runs (taken now if free, else queued behind the owner), mirroring the
    /// heavyweight [`cond_signal`](Self::cond_signal).
    pub fn lwcond_signal(&mut self, w: &mut dyn GuestWords, work: u32, all: bool) {
        let mut cx = sched!(self);
        self.sync.lwcond_signal(&mut cx, w, work, all)
    }

    // --- event flags ---

    /// Create an event flag with an initial bit pattern, returning its SceUID.
    pub fn create_event_flag(&mut self, init: u32) -> i32 {
        let uid = self.new_uid();
        self.sync.create_event_flag(uid, init);
        uid
    }

    /// Record the guest's name for an event flag, so every report that names the uid can
    /// name the subsystem too. Empty names are not stored.
    #[inline]
    pub fn name_event_flag(&mut self, uid: i32, name: &str) {
        self.sync.name_event_flag(uid, name)
    }

    /// Record an event flag's creation attributes, read back by [`Self::event_flag_info`].
    #[inline]
    pub fn record_event_flag_params(&mut self, uid: i32, attr: u32, init: u32) {
        self.sync.record_event_flag_params(uid, attr, init)
    }

    /// `(attr, initPattern, currentPattern, waiting threads)` of an event flag, or `None` for
    /// a uid that names none.
    #[inline]
    pub fn event_flag_info(&self, uid: i32) -> Option<(u32, u32, u32, u32)> {
        self.sync.event_flag_info(uid)
    }

    /// The guest's name for an event flag, or `""`.
    #[inline]
    pub fn event_flag_name(&self, uid: i32) -> &str {
        self.sync.event_flag_name(uid)
    }

    /// Set (OR in) bits on an event flag.
    #[inline]
    pub fn event_set(&mut self, uid: i32, bits: u32) {
        self.sync.event_set(uid, bits)
    }

    /// Whether an event flag with this uid exists. The companion of [`Self::sema_exists`],
    /// and it exists for the same reason: a wait on a uid that names nothing must be an
    /// ERROR, not a park, and `evf_try_wait` returning `None` cannot tell the two apart.
    #[inline]
    pub fn evf_exists(&self, uid: i32) -> bool {
        self.sync.evf_exists(uid)
    }

    /// Try to satisfy an event flag wait without blocking. On a match, applies the
    /// mode's clear op and returns the pattern AT the match (what `outBits` should
    /// report); `None` means the caller must park ([`Self::evf_block`]).
    #[inline]
    pub fn evf_try_wait(&mut self, uid: i32, bits: u32, mode: u32) -> Option<u32> {
        self.sync.evf_try_wait(uid, bits, mode)
    }

    /// Park the current thread on event flag `uid` until `bits` is satisfied under
    /// `mode` (or `timeout_us` passes; 0 = wait forever). `out_addr` is the guest
    /// `outBits` pointer the wake will write the match pattern through.
    pub fn evf_block(&mut self, uid: i32, bits: u32, mode: u32, out_addr: u32, timeout_us: u32) {
        let deadline = (timeout_us != 0).then(|| self.virtual_us + timeout_us as u64);
        self.sync
            .evf_block(uid, self.current, bits, mode, out_addr, deadline, 0)
    }

    /// Set bits on an event flag, then release every parked waiter the new pattern
    /// satisfies (in FIFO order; each match applies its clear op before the next
    /// waiter is evaluated, matching kernel release order). The preemptive
    /// counterpart of [`event_set`](Self::event_set).
    pub fn event_set_wake(&mut self, uid: i32, bits: u32) {
        let mut cx = sched!(self);
        self.sync.event_set_wake(&mut cx, uid, bits)
    }

    /// Clear an event flag's bits: keep only the bits also set in `bits` (the
    /// sceKernelClearEventFlag semantics, `pattern &= bits`).
    #[inline]
    pub fn event_clear(&mut self, uid: i32, bits: u32) {
        self.sync.event_clear(uid, bits)
    }

    /// The current bit pattern of an event flag.
    #[inline]
    pub fn event_pattern(&self, uid: i32) -> u32 {
        self.sync.event_pattern(uid)
    }

    // --- simple events ---

    /// `sceKernelCreateSimpleEvent`: an event flag whose waits are OR waits, which with
    /// [`Self::SIMPLE_EVENT_AUTO_RESET`] in `attr` also CLEAR the pattern when satisfied (so a
    /// set releases one waiter). It carries 64-bit user data from the last set.
    pub fn create_simple_event(&mut self, attr: u32, init: u32) -> i32 {
        let uid = self.new_uid();
        self.sync.create_simple_event(uid, attr, init);
        uid
    }

    /// `attr` bit of a simple event read as AUTO-RESET. A fighting title creates its task events with
    /// exactly this bit and uses each as a one-shot wake between a producer and one waiter.
    pub const SIMPLE_EVENT_AUTO_RESET: u32 = SyncObjects::SIMPLE_EVENT_AUTO_RESET;

    /// The event-flag wait mode a simple event's wait uses, or `None` if `uid` is not one.
    #[inline]
    pub fn simple_event_mode(&self, uid: i32) -> Option<u32> {
        self.sync.simple_event_mode(uid)
    }

    /// Record a simple event's user data (`sceKernelSetEvent`'s third argument).
    #[inline]
    pub fn set_simple_event_data(&mut self, uid: i32, data: u64) {
        self.sync.set_simple_event_data(uid, data)
    }

    #[inline]
    pub fn simple_event_data(&self, uid: i32) -> u64 {
        self.sync.simple_event_data(uid)
    }

    /// `sceKernelDeleteSimpleEvent`. False if `uid` is not a simple event.
    #[inline]
    pub fn delete_simple_event(&mut self, uid: i32) -> bool {
        self.sync.delete_simple_event(uid)
    }

    /// Park the current thread on a simple event - [`Self::evf_block`] plus the user-data
    /// out pointer.
    pub fn simple_event_block(
        &mut self,
        uid: i32,
        bits: u32,
        mode: u32,
        out_addr: u32,
        user_data_addr: u32,
        timeout_us: u32,
    ) {
        let deadline = (timeout_us != 0).then(|| self.virtual_us + timeout_us as u64);
        self.sync.evf_block(
            uid,
            self.current,
            bits,
            mode,
            out_addr,
            deadline,
            user_data_addr,
        )
    }

    // --- message pipes ---

    /// Create the (empty) byte FIFO of message pipe `uid`.
    #[inline]
    pub fn msg_pipe_create(&mut self, uid: i32) {
        self.sync.msg_pipe_create(uid)
    }

    /// Destroy message pipe `uid`'s FIFO. Whether it existed.
    #[inline]
    pub fn msg_pipe_remove(&mut self, uid: i32) -> bool {
        self.sync.msg_pipe_remove(uid)
    }

    /// The bytes message pipe `uid` holds, or `None` if it does not exist.
    #[inline]
    pub fn msg_pipe(&self, uid: i32) -> Option<&std::collections::VecDeque<u8>> {
        self.sync.msg_pipe(uid)
    }

    /// Message pipe `uid`'s FIFO, to send into or receive from.
    #[inline]
    pub fn msg_pipe_mut(&mut self, uid: i32) -> Option<&mut std::collections::VecDeque<u8>> {
        self.sync.msg_pipe_mut(uid)
    }

    /// Park the current thread in a message-pipe receive.
    pub fn msgpipe_block(
        &mut self,
        uid: i32,
        buf: u32,
        size: u32,
        full: bool,
        result: u32,
        timeout_us: Option<u32>,
    ) {
        let deadline = timeout_us.map(|t| self.virtual_us + t as u64);
        self.sync
            .msgpipe_block(uid, self.current, buf, size, full, result, deadline)
    }

    /// Release every receiver parked on pipe `uid` with `code` as its wait's result (the
    /// pipe was cancelled or deleted under it), returning how many there were. Nothing is
    /// written to their buffers - the wait did not complete.
    pub fn msgpipe_release(&mut self, uid: i32, code: u32) -> u32 {
        let mut cx = sched!(self);
        self.sync.msgpipe_release(&mut cx, uid, code)
    }

    /// After bytes arrive on pipe `uid`: hand them to parked receivers in FIFO order, each
    /// only when the pipe can satisfy its mode, and wake it. The receive buffer is written
    /// here, straight into guest memory - the receiver is parked, so nothing races it.
    pub fn msgpipe_service(&mut self, ctx: &mut GuestCtx, uid: i32) {
        let mut cx = sched!(self);
        self.sync.msgpipe_service(&mut cx, ctx, uid)
    }
}
