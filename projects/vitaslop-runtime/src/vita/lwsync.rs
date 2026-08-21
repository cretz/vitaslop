//! Lightweight mutexes and condition variables (SceLibKernel LwMutex/LwCond).
//!
//! Unlike the heavyweight primitives, a lightweight object's state lives in a
//! caller-provided work area (`SceKernelLwMutexWork` / `SceKernelLwCondWork`, both
//! 32 bytes) rather than a kernel handle - libc embeds them directly in its own
//! structures.
//!
//! The MUTEX takes that literally: its identity, owner and recursion count are four words
//! of the guest's own work area ([`super::lwwork`]), which is where the device keeps them
//! and what lets the uncontended take be emitted straight into guest code with no boundary
//! crossing at all. The handlers below are the contended half - parking, waking, and
//! resolving a work area the fast path would not serve.
//!
//! The COND keeps its waiter list host-side (a list of parked thread ids is not something
//! guest memory holds usefully), identified by the id stamped in its own work area.

use crate::host::{GuestCtx, VitaState};
use crate::vita::lwwork;
use crate::SvcOutcome;

/// `SCE_KERNEL_ERROR_LW_MUTEX_FAILED_TO_OWN`: `sceKernelTryLockLwMutex` on a
/// lightweight mutex another thread holds returns this rather than blocking.
const ERR_LW_MUTEX_FAILED_TO_OWN: u32 = 0x8002_8185;

/// `SCE_KERNEL_ERROR_UNKNOWN_LW_COND_ID`: `sceKernelWaitLwCond`/`SignalLwCond` on a
/// lightweight cond that was never created (no `CreateLwCond` observed). The kernel
/// rejects it with this code; it does not block or touch any mutex.
const ERR_UNKNOWN_LW_COND_ID: u32 = 0x8002_81C1;

/// Size of `SceKernelLwCondWork` (`SceInt32 data[4]`), in bytes. Distinct from the
/// [`lwwork::WORK_SIZE`] - a cond work area is half as large, and zeroing the mutex size
/// over it would overrun into whatever the caller placed immediately after.
pub(super) const LW_COND_WORK_SIZE: usize = 16;

/// Resolve a lightweight-cond work pointer to the canonical cond recorded at
/// `sceKernelCreateLwCond`. The kernel identifies a cond by an id stored *inside* its
/// work area, so a caller may legitimately wait or signal on a byte *copy* of the work
/// area at a different address - e.g. a C++ condition-variable wrapper that stages its
/// embedded `SceKernelLwCondWork` on the stack and passes `&copy` to the kernel. A
/// pointer that is itself a known cond resolves directly; otherwise read the identity
/// word the copy carries (written by `create_lw_cond`) and resolve that. `None` means
/// neither the pointer nor its identity word names a created cond.
fn resolve_cond(ctx: &GuestCtx, st: &VitaState, work: u32) -> Option<u32> {
    if st.lwcond_is_known(work) {
        return Some(work);
    }
    let carried = ctx.read_u32(work);
    st.lwcond_is_known(carried).then_some(carried)
}

/// int sceKernelWaitLwCond(SceKernelLwCondWork *work, SceUInt32 *ppTimeout)
/// Preemptive: release the cond's bound lightweight mutex and park the calling thread
/// so the scheduler can run the thread that will signal it; a non-null timeout also
/// arms a deadline wake. In the single-thread model nothing contends, so the wait is
/// immediate success.
pub(super) fn wait_lw_cond(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let work = ctx.arg(0);
    let timeout_ptr = ctx.arg(1);
    let timeout = if timeout_ptr != 0 { ctx.read_u32(timeout_ptr) } else { 0 };
    sample_wait(st, work, timeout_ptr, timeout);
    if !st.is_preemptive() {
        // Single-thread model: uncontended, immediate success.
        ctx.ret(0);
        return SvcOutcome::Continue;
    }
    // Resolve the (possibly copied) work area to its canonical created cond. A pointer
    // that names no created cond is genuinely unknown: the kernel rejects it with
    // SCE_KERNEL_ERROR_UNKNOWN_LW_COND_ID without releasing a mutex or blocking.
    let Some(canonical) = resolve_cond(ctx, st, work) else {
        tracing::warn!(
            target: "vitaslop::sema",
            thread = format_args!("0x{:x}", st.current_thread()),
            cond = format_args!("0x{work:08x}"),
            "sceKernelWaitLwCond on an unknown lightweight cond -> UNKNOWN_LW_COND_ID"
        );
        ctx.ret(ERR_UNKNOWN_LW_COND_ID);
        return SvcOutcome::Continue;
    };
    let parked = st.lwcond_wait(ctx, canonical, timeout);
    debug_assert!(parked, "canonical cond is known, so the wait must park");
    // A satisfied/woken wait returns 0.
    ctx.ret(0);
    SvcOutcome::Block
}

/// int sceKernelSignalLwCond(SceKernelLwCondWork *work) and friends.
pub(super) fn signal_lw_cond(ctx: &mut GuestCtx, st: &mut VitaState, all: bool) {
    let work = ctx.arg(0);
    if st.is_preemptive() {
        // Resolve a possibly-copied work area to the canonical cond so waiters parked
        // via the original (or another copy) are woken.
        if let Some(canonical) = resolve_cond(ctx, st, work) {
            st.lwcond_signal(ctx, canonical, all);
        }
    }
    ctx.ret(0);
}

/// Diagnostic: capture the first few `sceKernelWaitLwCond(work, *timeout)` calls -
/// the cond work pointer and the timeout (0 = infinite wait). Distinguishes a
/// producer/consumer wait from a timed delay loop. Capped, then free.
pub(super) fn sample_wait(st: &mut VitaState, work: u32, timeout_ptr: u32, timeout: u32) {
    if st.capture.lwcond_wait_samples.len() < 8 {
        st.capture.lwcond_wait_samples.push((work, timeout_ptr, timeout));
    }
}

/// int sceKernelCreateLwMutex(SceKernelLwMutexWork *pWork, const char *pName,
///     unsigned int attr, int initCount, const ...OptParam *pOpt)
///
/// Register the lightweight mutex at its work-area address as the canonical object and lay
/// its state out there: free, unowned, nothing parked, stamped with its own address as an
/// identity id. Like [`create_lw_cond`], the stamp is what lets a lock/unlock on a byte
/// *copy* of the work area resolve back to this mutex (see [`resolve_mutex`]) - the real
/// kernel keeps an id inside the work area, and a caller may stage the struct elsewhere and
/// operate on the copy.
///
/// `initCount` (a mutex created already-held) is not modeled; every mutex starts free.
///
/// The rest of the 32-byte work area is zeroed as it always was: it is the guest's storage
/// and we own all of it, so leaving whatever the caller happened to have there would make
/// an uninitialised read look like state.
pub(super) fn create_lw_mutex(ctx: &mut GuestCtx, st: &mut VitaState) {
    let work = ctx.arg(0);
    ctx.write_bytes(work, &[0u8; lwwork::WORK_SIZE as usize]);
    st.lwmutex_register(ctx, work);
    ctx.ret(0);
}

/// Resolve a lightweight-mutex work pointer to the canonical mutex, ADOPTING a work area no
/// create was ever seen for.
///
/// Three cases, and the identity word in the work area separates them:
/// - it names ITSELF: this is the canonical mutex, use it;
/// - it names another known mutex: this is a byte COPY staged elsewhere (a C++ wrapper
///   putting its embedded work struct on the stack), so operate on the original;
/// - it is zero: no create ever touched this work area. A statically-initialized mutex is
///   ordinary in libc, so adopt it - stamp it as its own canonical mutex, which is the same
///   lazy-by-address behaviour the host record always had, plus the one write that lets
///   every later lock be taken inline instead of crossing.
///
/// A fourth shape exists and is NOT adopted: an id that names a mutex we have no record of.
/// That is a copy of something deleted, or of a work area whose stamp was overwritten, and
/// the honest thing is to leave it alone and operate by address - stamping it would make
/// the copy a mutex in its own right, which is precisely the split this resolution exists
/// to prevent. It is reported once per address because it should not happen.
fn resolve_mutex(ctx: &mut GuestCtx, st: &mut VitaState, work: u32) -> u32 {
    if lwwork::is_mutex(ctx, work) {
        return work;
    }
    let carried = lwwork::carried_id(ctx, work);
    if carried == 0 {
        if work != 0 {
            st.lwmutex_adopt(ctx, work);
        }
        return work;
    }
    if st.lwmutex_is_known(carried) {
        return carried;
    }
    tracing::debug!(
        target: "vitaslop::sema",
        work = format_args!("{work:#010x}"),
        carried = format_args!("{carried:#010x}"),
        "a lightweight-mutex work area names a mutex with no record - operating by address"
    );
    work
}

/// int sceKernelCreateLwCond(SceKernelLwCondWork *pWork, const char *pName,
///     unsigned int attr, SceKernelLwMutexWork *pLwMutex, const ...OptParam *pOpt)
///
/// Record which lightweight mutex (arg 3) this cond binds to, so `sceKernelWaitLwCond`
/// releases and re-acquires that mutex atomically (without the binding a waiter would
/// hold the mutex across the wait and deadlock any locker). Then initialize the work
/// area: zero it, and stamp its own address into the first word as an identity id. The
/// real kernel stores an id inside the work area; a caller may wait/signal on a byte
/// copy of it (a wrapper staging the struct on the stack), and [`resolve_cond`] uses
/// this id to map the copy back to the canonical cond.
pub(super) fn create_lw_cond(ctx: &mut GuestCtx, st: &mut VitaState) {
    let cond_work = ctx.arg(0);
    let mutex_work = ctx.arg(3);
    st.lwcond_bind_mutex(cond_work, mutex_work);
    ctx.write_bytes(cond_work, &[0u8; LW_COND_WORK_SIZE]);
    ctx.write_u32(cond_work, cond_work);
    ctx.ret(0);
}

/// The shared lock/unlock/wait/signal/delete outcome: success.
pub(super) fn succeed(ctx: &mut GuestCtx) {
    ctx.ret(0);
}

/// The inline form of a lightweight-sync host import: the two halves of an UNCONTENDED
/// lock, emitted straight into guest code. `None` for everything else. Reached through
/// [`crate::vita::inline_op`], which owns the global on/off switch.
///
/// # Why a lock, of all things, may be inlined
/// Because on the device it is not a system call. `sceKernelLockLwMutex` is a userspace
/// function that compare-and-swaps the caller's own work area and enters the kernel only on
/// CONTENTION - which is exactly the split the emitted guard makes. Inlining it is not an
/// approximation of the call, it is the shape the call has; the fallback arm is the syscall
/// the hardware would also have made.
///
/// # What must be true for this to be safe, and where each part lives
/// - The state is in the work area, not on the host: [`super::lwwork`].
/// - The emitted predicate is the same predicate the handler applies:
///   [`lwwork::fast_lock`] / [`lwwork::fast_unlock`] are what the handler calls first, and
///   the emitted code is held against them by an execution test.
/// - Every case the fast path refuses reaches this file, including the ones that report.
///
/// # The trade
/// An inlined call never reaches the host, so it does not appear in the host-call trace or
/// the call histogram. For this pair that is the whole point of the change, but it means a
/// run investigating lock traffic wants `VITASLOP_NO_INLINE_IMPORTS=1`.
pub(crate) fn inline_op(func_nid: u32) -> Option<vitaslop_transpiler::InlineOp> {
    use crate::nid::lwsync as lw;
    use crate::vita::mirror::SLOT_CURRENT_THREAD;
    use vitaslop_transpiler::InlineOp::{LwMutexLock, LwMutexUnlock};
    let layout = lwwork::layout();
    Some(match func_nid {
        // Both spellings of each, exactly as the dispatch routes both to one handler. The
        // `...CB` lock differs on hardware by also delivering the caller's pending
        // callbacks; we model no callback delivery for either, so inlining one and not the
        // other would make two spellings of one call behave differently - which is a worse
        // answer than the one they already share.
        lw::LOCK_LW_MUTEX | lw::LOCK_LW_MUTEX_CB => {
            LwMutexLock { layout, thread_slot: SLOT_CURRENT_THREAD }
        }
        lw::UNLOCK_LW_MUTEX | lw::UNLOCK_LW_MUTEX2 => {
            LwMutexUnlock { layout, thread_slot: SLOT_CURRENT_THREAD }
        }
        _ => return None,
    })
}

/// Why the rest of the lightweight-sync calls are NOT inlined. Kept as code rather than a
/// comment so a future reader adding one has to answer the same question, and so
/// `only_the_uncontended_lock_is_inlined` can walk the list.
///
/// The theme is that a lock is the ONLY member of this family whose common case is a
/// userspace compare-and-swap. Everything else here parks a thread, wakes one, or allocates
/// an identity - and an inlined call never reaches the host, so whatever it was going to do
/// simply stops happening, silently.
#[cfg(test)]
const NOT_INLINABLE: &[(u32, &str)] = &[
    (crate::nid::lwsync::TRY_LOCK_LW_MUTEX, "returns an error code the work area cannot spell"),
    (crate::nid::lwsync::CREATE_LW_MUTEX, "stamps an identity and registers a record"),
    (crate::nid::lwsync::DELETE_LW_MUTEX, "drops the parked queue"),
    (crate::nid::lwsync::WAIT_LW_COND, "parks the caller - that IS the behaviour"),
    (crate::nid::lwsync::SIGNAL_LW_COND, "wakes a parked thread, which only the host can do"),
];

/// int sceKernelLockLwMutex(SceKernelLwMutexWork *pWork, int lockCount,
///     unsigned int *pTimeout), and (with `try_lock`) sceKernelTryLockLwMutex.
///
/// Acquire the lightweight mutex if free or already held by this thread (recursive), else
/// the plain lock parks the caller ([`SvcOutcome::Block`]) while the try-lock returns
/// `ERR_LW_MUTEX_FAILED_TO_OWN` without blocking.
///
/// # This is the SLOW half
/// The uncontended take is emitted straight into guest code (`InlineOp::LwMutexLock`) and
/// never arrives here at all. What does arrive is a contended take, a work area that is a
/// copy or was never created, a mutex with a thread parked on it, or a `lockCount` other
/// than one - and every one of those is a case only the host can settle. That split is the
/// device's own: on hardware this call is userspace until it contends.
///
/// The `lockCount`/`pTimeout` arguments follow the heavyweight mutex's handling: a single
/// acquisition, and the timeout is not yet modeled for either mutex kind. The inline form
/// refuses any count but one for exactly that reason, so the two paths agree wherever both
/// can run.
pub(super) fn lock_lw_mutex(ctx: &mut GuestCtx, st: &mut VitaState, try_lock: bool) -> SvcOutcome {
    let work = resolve_mutex(ctx, st, ctx.arg(0));
    if try_lock && st.lwmutex_contended(ctx, work) {
        ctx.ret(ERR_LW_MUTEX_FAILED_TO_OWN);
        return SvcOutcome::Continue;
    }
    // Success returns 0 whether acquired now or after a wake by the releasing thread.
    ctx.ret(0);
    let acquired = st.lwmutex_lock(ctx, work);
    // >>> WHAT THIS TRACE CAN AND CANNOT SEE. Only the SLOW half arrives here, so an
    // uncontended take - the common case, emitted as `InlineOp::LwMutexLock` straight into guest
    // code - produces NO trace line at all. An empty lwmutex log therefore means "never
    // contended", NOT "never taken", and reading it the second way is how a lock gets wrongly
    // ruled out. To see every take, watch the WORK AREA in guest memory instead
    // (`VITASLOP_WATCH_STORE=<work addr>`, which now covers vector stores too).
    tracing::trace!(
        target: "vitaslop::sema",
        work = format_args!("{work:#010x}").to_string(),
        thread = st.current_thread(),
        lr = format_args!("{:#010x}", ctx.regs[14]).to_string(),
        acquired,
        "lwmutex lock (CONTENDED path only)"
    );
    if acquired || !st.is_preemptive() {
        // The single-thread model has nobody to wake a parked caller, so it cannot park
        // one. It also cannot reach here: with one thread of control every lock is free or
        // recursive. Refusing to block is the belt to that argument's braces.
        SvcOutcome::Continue
    } else {
        SvcOutcome::Block
    }
}

/// int sceKernelUnlockLwMutex(SceKernelLwMutexWork *pWork, int unlockCount) and the
/// `...2` variant: release the lightweight mutex, waking the next parked waiter.
pub(super) fn unlock_lw_mutex(ctx: &mut GuestCtx, st: &mut VitaState) {
    let work = resolve_mutex(ctx, st, ctx.arg(0));
    st.lwmutex_unlock(ctx, work);
    ctx.ret(0);
}

/// int sceKernelDeleteLwMutex(SceKernelLwMutexWork *pWork): clear the work area's identity
/// stamp and forget the parked queue.
pub(super) fn delete_lw_mutex(ctx: &mut GuestCtx, st: &mut VitaState) {
    let work = resolve_mutex(ctx, st, ctx.arg(0));
    st.lwmutex_delete(ctx, work);
    ctx.ret(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nid::lwsync as lw;
    use crate::world::DeterministicWorld;
    use crate::{SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::REG_COUNT;
    use vitaslop_transpiler::InlineOp;

    /// A state and a guest image, with the registers a call would arrive with.
    fn with<R>(f: impl FnOnce(&mut GuestCtx, &mut VitaState) -> R) -> R {
        let mut st = VitaState::new(0, 0x4000, Box::new(DeterministicWorld::default()));
        st.set_preemptive(true);
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = vec![0u8; 0x4000];
        let mut mem = SliceMemory(&mut bytes);
        let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        f(&mut ctx, &mut st)
    }

    const WORK: u32 = 0x800;

    /// Every inlined NID must reach the SAME decision the handler reaches, from the same
    /// work area. This is the definition side of the proof; the emitted code is held to
    /// `lwwork`'s fast paths by an execution test in `vitaslop-native`, and this holds the
    /// handler to them too - so all three agree by transitivity rather than by inspection.
    #[test]
    fn the_handler_takes_exactly_what_the_inline_form_takes() {
        for nid in [lw::LOCK_LW_MUTEX, lw::LOCK_LW_MUTEX_CB] {
            assert!(matches!(inline_op(nid), Some(InlineOp::LwMutexLock { .. })));
            with(|ctx, st| {
                st.set_current(3);
                create(ctx, st, WORK);
                // A free mutex: the inline form would take it, and so must the handler,
                // leaving the same words behind.
                assert!(lwwork::fast_lock(ctx, WORK, 3, 1), "the definition takes it");
                // Put it back and let the handler do it instead.
                lwwork::init(ctx, WORK);
                ctx.regs[0] = WORK;
                ctx.regs[1] = 1;
                assert!(matches!(lock_lw_mutex(ctx, st, false), SvcOutcome::Continue), "uncontended");
                assert_eq!(ctx.regs[0], 0, "success");
                assert_eq!(lwwork::count(ctx, WORK), 1);
                assert_eq!(lwwork::owner(ctx, WORK), 3);
            });
        }
        for nid in [lw::UNLOCK_LW_MUTEX, lw::UNLOCK_LW_MUTEX2] {
            assert!(matches!(inline_op(nid), Some(InlineOp::LwMutexUnlock { .. })));
        }
    }

    /// `sceKernelCreateLwMutex` must leave a work area the fast path recognises, or every
    /// lock the title makes crosses to the host and the whole change buys nothing. That is
    /// a silent failure - correct, just slow - so it is asserted rather than assumed.
    #[test]
    fn a_created_mutex_is_takeable_inline() {
        with(|ctx, st| {
            create(ctx, st, WORK);
            assert!(lwwork::is_mutex(ctx, WORK), "create must stamp the identity");
            assert!(lwwork::fast_lock(ctx, WORK, 1, 1), "and leave it free");
        });
    }

    /// A work area no create ever touched is ADOPTED on its first host-side lock, so the
    /// second lock is inline. Statically-initialized lightweight mutexes are ordinary in
    /// libc, and a title that uses one would otherwise never see the fast path at all.
    #[test]
    fn a_never_created_work_area_is_adopted_on_first_use() {
        with(|ctx, st| {
            st.set_current(1);
            assert!(!lwwork::is_mutex(ctx, WORK), "zeroed memory is not yet a mutex");
            assert!(!lwwork::fast_lock(ctx, WORK, 1, 1), "so the fast path refuses it");
            ctx.regs[0] = WORK;
            ctx.regs[1] = 1;
            assert!(matches!(lock_lw_mutex(ctx, st, false), SvcOutcome::Continue), "uncontended");
            assert!(lwwork::is_mutex(ctx, WORK), "the handler adopted it");
            assert_eq!(lwwork::count(ctx, WORK), 1, "and took it");
            // Released, the next take is inline.
            ctx.regs[0] = WORK;
            unlock_lw_mutex(ctx, st);
            assert!(lwwork::fast_lock(ctx, WORK, 1, 1));
        });
    }

    /// A byte COPY of a created work area must resolve to the ORIGINAL. This is the
    /// reference-semantics bug the whole layout exists to avoid: a wrapper that stages its
    /// embedded work struct elsewhere and locks the copy must not get a second, independent
    /// mutex - two threads would then hold what the title believes is one lock.
    #[test]
    fn a_copy_of_a_work_area_locks_the_original() {
        with(|ctx, st| {
            let copy = WORK + 0x100;
            st.set_current(1);
            create(ctx, st, WORK);
            for i in 0..lwwork::BYTES / 4 {
                let v = ctx.read_u32(WORK + i * 4);
                ctx.write_u32(copy + i * 4, v);
            }
            assert!(!lwwork::fast_lock(ctx, copy, 1, 1), "the fast path will not serve a copy");
            ctx.regs[0] = copy;
            ctx.regs[1] = 1;
            assert!(matches!(lock_lw_mutex(ctx, st, false), SvcOutcome::Continue), "uncontended");
            assert_eq!(lwwork::count(ctx, WORK), 1, "the ORIGINAL is the one that got locked");
            assert_eq!(lwwork::count(ctx, copy), 0, "and the copy is untouched");
        });
    }

    /// Deleting a mutex must make its work area untakeable, or a guest that kept the
    /// pointer keeps locking a mutex nothing tracks - inline, so the host never sees it.
    #[test]
    fn a_deleted_mutex_cannot_be_taken_inline() {
        with(|ctx, st| {
            create(ctx, st, WORK);
            ctx.regs[0] = WORK;
            delete_lw_mutex(ctx, st);
            assert!(!lwwork::is_mutex(ctx, WORK));
            assert!(!lwwork::fast_lock(ctx, WORK, 1, 1));
        });
    }

    /// Nothing else in this family may be inlined, and each entry says why. An inlined call
    /// never reaches the host, so for any of these the behaviour would simply stop
    /// happening - silently, at the one call the title needed it at.
    #[test]
    fn only_the_uncontended_lock_is_inlined() {
        for &(nid, why) in NOT_INLINABLE {
            assert!(
                inline_op(nid).is_none(),
                "{} must stay a host call: {why}",
                crate::nid::name(nid)
            );
        }
    }

    /// Run `sceKernelCreateLwMutex(work, ...)` the way the dispatch would.
    fn create(ctx: &mut GuestCtx, st: &mut VitaState, work: u32) {
        ctx.regs[0] = work;
        create_lw_mutex(ctx, st);
    }
}
