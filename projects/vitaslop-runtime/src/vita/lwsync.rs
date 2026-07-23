//! Lightweight mutexes and condition variables (SceLibKernel LwMutex/LwCond).
//!
//! Unlike the heavyweight primitives, a lightweight object's state lives in a
//! caller-provided work area (`SceKernelLwMutexWork` / `SceKernelLwCondWork`, both
//! 32 bytes) rather than a kernel handle - libc embeds them directly in its own
//! structures. The bring-up model matches [`crate::vita::sync`]: with one thread of
//! control nothing ever contends, so a lock or a wait is unconditional success.
//! Create zero-initializes the work area to the "unlocked, no owner" state; the
//! preemptive scheduler will give these real blocking semantics alongside the
//! heavyweight ones.

use crate::host::{GuestCtx, VitaState};
use crate::SvcOutcome;

/// `SCE_KERNEL_ERROR_LW_MUTEX_FAILED_TO_OWN`: `sceKernelTryLockLwMutex` on a
/// lightweight mutex another thread holds returns this rather than blocking.
const ERR_LW_MUTEX_FAILED_TO_OWN: u32 = 0x8002_8185;

/// `SCE_KERNEL_ERROR_UNKNOWN_LW_COND_ID`: `sceKernelWaitLwCond`/`SignalLwCond` on a
/// lightweight cond that was never created (no `CreateLwCond` observed). The kernel
/// rejects it with this code; it does not block or touch any mutex.
const ERR_UNKNOWN_LW_COND_ID: u32 = 0x8002_81C1;

/// Size of `SceKernelLwMutexWork` (`SceInt64 data[4]`), in bytes.
pub(super) const LW_MUTEX_WORK_SIZE: usize = 32;
/// Size of `SceKernelLwCondWork` (`SceInt32 data[4]`), in bytes. Distinct from the
/// mutex work size - a cond work area is half as large, and zeroing the mutex size
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
    let parked = st.lwcond_wait(canonical, timeout);
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
            st.lwcond_signal(canonical, all);
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
/// Register the lightweight mutex at its work-area address as the canonical object, then
/// initialize the work area: zero it and stamp its own address into the first word as an
/// identity id. Like [`create_lw_cond`], this lets a lock/unlock on a byte *copy* of the
/// work area resolve back to this mutex (see [`resolve_mutex`]) - the real kernel keeps an
/// id inside the work area, and a caller may stage the struct elsewhere and operate on the
/// copy. `initCount` (a mutex created already-held) is not modeled; every mutex starts free.
pub(super) fn create_lw_mutex(ctx: &mut GuestCtx, st: &mut VitaState) {
    let work = ctx.arg(0);
    if st.is_preemptive() {
        st.lwmutex_register(work);
    }
    ctx.write_bytes(work, &[0u8; LW_MUTEX_WORK_SIZE]);
    ctx.write_u32(work, work);
    ctx.ret(0);
}

/// Resolve a lightweight-mutex work pointer to the canonical mutex registered at
/// `sceKernelCreateLwMutex`. A pointer that is itself a known mutex resolves directly;
/// otherwise the caller may hold a byte copy of the work area, so read the identity word
/// it carries (written by [`create_lw_mutex`]) and use that if it names a known mutex.
/// Falls back to the pointer itself for a never-created (e.g. statically-initialized)
/// mutex, preserving the lazy-by-address behavior for the uncopied case.
fn resolve_mutex(ctx: &GuestCtx, st: &VitaState, work: u32) -> u32 {
    if st.lwmutex_is_known(work) {
        return work;
    }
    let carried = ctx.read_u32(work);
    if st.lwmutex_is_known(carried) {
        carried
    } else {
        work
    }
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

/// int sceKernelLockLwMutex(SceKernelLwMutexWork *pWork, int lockCount,
///     unsigned int *pTimeout), and (with `try_lock`) sceKernelTryLockLwMutex.
///
/// Single-thread model: uncontended, always succeeds. Preemptive: acquire the
/// lightweight mutex (keyed by its work-area address) if free or already held by this
/// thread (recursive), else the plain lock parks the caller ([`SvcOutcome::Block`])
/// while the try-lock returns `ERR_LW_MUTEX_FAILED_TO_OWN` without blocking. The
/// `lockCount`/`pTimeout` arguments follow the heavyweight mutex's handling (a single
/// acquisition; the timeout is not yet modeled for either mutex kind).
pub(super) fn lock_lw_mutex(ctx: &mut GuestCtx, st: &mut VitaState, try_lock: bool) -> SvcOutcome {
    let work = ctx.arg(0);
    if !st.is_preemptive() {
        ctx.ret(0);
        return SvcOutcome::Continue;
    }
    let work = resolve_mutex(ctx, st, work);
    if try_lock && st.lwmutex_contended(work) {
        ctx.ret(ERR_LW_MUTEX_FAILED_TO_OWN);
        return SvcOutcome::Continue;
    }
    // Success returns 0 whether acquired now or after a wake by the releasing thread.
    ctx.ret(0);
    if st.lwmutex_lock(work) {
        SvcOutcome::Continue
    } else {
        SvcOutcome::Block
    }
}

/// int sceKernelUnlockLwMutex(SceKernelLwMutexWork *pWork, int unlockCount) and the
/// `...2` variant: release the lightweight mutex, waking the next parked waiter.
pub(super) fn unlock_lw_mutex(ctx: &mut GuestCtx, st: &mut VitaState) {
    let work = ctx.arg(0);
    if st.is_preemptive() {
        let work = resolve_mutex(ctx, st, work);
        st.lwmutex_unlock(work);
    }
    ctx.ret(0);
}

/// int sceKernelDeleteLwMutex(SceKernelLwMutexWork *pWork): forget its host state.
pub(super) fn delete_lw_mutex(ctx: &mut GuestCtx, st: &mut VitaState) {
    let work = ctx.arg(0);
    if st.is_preemptive() {
        let work = resolve_mutex(ctx, st, work);
        st.lwmutex_delete(work);
    }
    ctx.ret(0);
}
