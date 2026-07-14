//! Synchronization primitives (mutex, semaphore, event flag) and system time.
//!
//! Bring-up model: one thread of control, with workers run synchronously to
//! completion (see the re-entry seam in `host.rs`). Nothing ever actually
//! contends, so a mutex lock/unlock is unconditional success - which is exactly
//! correct for the common single-thread use (guarding data touched by the main
//! thread and a synchronously-run worker). A semaphore's count and an event
//! flag's bit pattern are still tracked so wait-then-read is observable and
//! faithful. Cross-thread blocking semantics arrive with the preemptive
//! multi-thread scheduler (see the runtime README concurrency model).

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::nid::sync as nid;
use crate::SvcOutcome;

/// A `sceKernelTryLockMutex` failure when another thread owns the mutex (the
/// try-lock returns an error rather than blocking). Approximate value; the guest
/// only checks nonzero.
const ERR_MUTEX_FAILED_TO_OWN: u32 = 0x8002_8082;

pub fn try_dispatch(func_nid: u32, ctx: &mut GuestCtx, st: &mut VitaState) -> Option<SvcOutcome> {
    match func_nid {
        nid::CREATE_MUTEX => create_mutex(ctx, st),
        // Lock and wait can block under the preemptive scheduler, so they return
        // the outcome directly (Block parks the calling thread).
        nid::LOCK_MUTEX => return Some(lock_mutex(ctx, st, false)),
        nid::TRY_LOCK_MUTEX => return Some(lock_mutex(ctx, st, true)),
        nid::UNLOCK_MUTEX => unlock_mutex(ctx, st),
        nid::DELETE_MUTEX => delete_object(ctx, st),
        nid::CREATE_SEMA => create_sema(ctx, st),
        nid::WAIT_SEMA => return Some(wait_sema(ctx, st)),
        nid::SIGNAL_SEMA => signal_sema(ctx, st),
        nid::DELETE_SEMA => delete_object(ctx, st),
        // Condition variables. Wait can block; signal wakes and hands off the
        // mutex. Single-thread mode: wait returns immediately, signal is a no-op.
        nid::CREATE_COND => create_cond(ctx, st),
        nid::WAIT_COND => return Some(wait_cond(ctx, st)),
        nid::SIGNAL_COND => signal_cond(ctx, st, false),
        nid::SIGNAL_COND_ALL => signal_cond(ctx, st, true),
        nid::DELETE_COND => delete_object(ctx, st),
        nid::CREATE_EVENT_FLAG => create_event_flag(ctx, st),
        nid::SET_EVENT_FLAG => set_event_flag(ctx, st),
        nid::WAIT_EVENT_FLAG => wait_event_flag(ctx, st),
        nid::CLEAR_EVENT_FLAG => clear_event_flag(ctx, st),
        nid::DELETE_EVENT_FLAG => delete_object(ctx, st),
        nid::GET_SYSTEM_TIME_WIDE => get_system_time_wide(ctx, st),
        _ => return None,
    }
    Some(SvcOutcome::Continue)
}

// --- mutex ---

/// SceUID sceKernelCreateMutex(const char *name, SceUInt attr, int initCount,
///     SceKernelMutexOptParam *option)
#[hostcall]
fn create_mutex(st: &mut VitaState, _name: Ptr, _attr: u32, _init: i32, _opt: Ptr) -> i32 {
    // Recording ownership state is harmless single-thread (lock/unlock take the
    // immediate path there) and necessary for preemptive blocking.
    st.create_mutex()
}

/// int sceKernelLockMutex(SceUID mutexid, int lockCount, unsigned int *timeout),
/// and (with `try_lock`) int sceKernelTryLockMutex(SceUID, int).
///
/// Single-thread model: uncontended, always succeeds. Preemptive: acquire if free
/// or already held by this thread (recursive), else the plain lock parks the caller
/// ([`SvcOutcome::Block`]) while the try-lock returns an error without blocking.
fn lock_mutex(ctx: &mut GuestCtx, st: &mut VitaState, try_lock: bool) -> SvcOutcome {
    let id = ctx.arg(0) as i32;
    if !st.is_preemptive() {
        ctx.ret(0);
        return SvcOutcome::Continue;
    }
    if try_lock && st.mutex_contended(id) {
        ctx.ret(ERR_MUTEX_FAILED_TO_OWN);
        return SvcOutcome::Continue;
    }
    // The return value on success is 0, whether acquired now or after a wake.
    ctx.ret(0);
    if st.mutex_lock(id) {
        SvcOutcome::Continue
    } else {
        SvcOutcome::Block
    }
}

/// int sceKernelUnlockMutex(SceUID mutexid, int unlockCount)
#[hostcall]
fn unlock_mutex(st: &mut VitaState, id: i32, _count: i32) -> i32 {
    if st.is_preemptive() {
        st.mutex_unlock(id);
    }
    0
}

// --- semaphore ---

/// SceUID sceKernelCreateSema(const char *name, SceUInt attr, int initVal,
///     int maxVal, SceKernelSemaOptParam *option)
#[hostcall]
fn create_sema(st: &mut VitaState, _name: Ptr, _attr: u32, init: i32, _max: i32, _opt: Ptr) -> i32 {
    st.create_sema(init)
}

/// int sceKernelWaitSema(SceUID semaid, int signal, unsigned int *timeout)
///
/// Single-thread model: take `signal` from the count (floored, never blocks).
/// Preemptive: take it if available, else park the caller until a signal delivers
/// it ([`SvcOutcome::Block`]); the return value is 0 either way.
fn wait_sema(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let id = ctx.arg(0) as i32;
    let signal = ctx.arg(1) as i32;
    ctx.ret(0);
    if !st.is_preemptive() {
        st.sema_wait(id, signal);
        return SvcOutcome::Continue;
    }
    if st.sema_try_acquire(id, signal) {
        SvcOutcome::Continue
    } else {
        st.sema_block(id, signal);
        SvcOutcome::Block
    }
}

/// int sceKernelSignalSema(SceUID semaid, int signal)
#[hostcall]
fn signal_sema(st: &mut VitaState, id: i32, signal: i32) -> i32 {
    if st.is_preemptive() {
        st.sema_signal_wake(id, signal);
    } else {
        st.sema_signal(id, signal);
    }
    0
}

// --- condition variable ---

/// SceUID sceKernelCreateCond(const char *name, SceUInt attr, SceUID mutexId,
///     const SceKernelCondOptParam *option)
#[hostcall]
fn create_cond(st: &mut VitaState, _name: Ptr, _attr: u32, mutex: i32, _opt: Ptr) -> i32 {
    st.create_cond(mutex)
}

/// int sceKernelWaitCond(SceUID condId, unsigned int *timeout)
///
/// Single-thread model: nothing else can signal, so the wait returns immediately
/// (the mutex stays held) - correct for the degenerate single-thread use.
/// Preemptive: release the mutex and park the caller until a signal delivers it
/// back with the mutex re-acquired ([`SvcOutcome::Block`]).
fn wait_cond(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let id = ctx.arg(0) as i32;
    ctx.ret(0);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    st.cond_wait(id);
    SvcOutcome::Block
}

/// int sceKernelSignalCond(SceUID condId) / sceKernelSignalCondAll(SceUID condId)
fn signal_cond(ctx: &mut GuestCtx, st: &mut VitaState, all: bool) {
    let id = ctx.arg(0) as i32;
    if st.is_preemptive() {
        st.cond_signal(id, all);
    }
    ctx.ret(0);
}

// --- event flag ---

/// SceUID sceKernelCreateEventFlag(const char *name, int attr, int initPattern,
///     SceKernelEventFlagOptParam *opt)
#[hostcall]
fn create_event_flag(st: &mut VitaState, _name: Ptr, _attr: u32, init: u32, _opt: Ptr) -> i32 {
    st.create_event_flag(init)
}

/// int sceKernelSetEventFlag(SceUID evid, unsigned int bitPattern)
#[hostcall]
fn set_event_flag(st: &mut VitaState, id: i32, bits: u32) -> i32 {
    st.event_set(id, bits);
    0
}

/// int sceKernelClearEventFlag(SceUID evid, unsigned int bitPattern)
#[hostcall]
fn clear_event_flag(st: &mut VitaState, id: i32, bits: u32) -> i32 {
    st.event_clear(id, bits);
    0
}

/// int sceKernelWaitEventFlag(SceUID evid, unsigned int bits, unsigned int wait,
///     unsigned int *outBits, SceUInt *timeout)
/// Uncontended: the requested bits are assumed already set (the main thread set
/// them), so it succeeds and reports the current pattern through `outBits`.
#[hostcall]
fn wait_event_flag(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: i32,
    _bits: u32,
    _wait: u32,
    out: Ptr,
    _timeout: Ptr,
) -> i32 {
    let pattern = st.event_pattern(id);
    if !out.is_null() {
        ctx.write_u32(out.addr(), pattern);
    }
    0
}

// --- delete (shared: no teardown needed for these lightweight handles) ---

/// int sceKernelDelete{Mutex,Sema,EventFlag}(SceUID id) - all succeed.
#[hostcall]
fn delete_object(_st: &mut VitaState, _id: i32) -> i32 {
    0
}

// --- time ---

/// SceUInt64 sceKernelGetSystemTimeWide(void)
/// A 64-bit return goes in r0 (low) and r1 (high), so this is hand-written rather
/// than `#[hostcall]` (whose value returns are 32-bit). Time is the virtual
/// monotonic clock, so it never goes backward.
fn get_system_time_wide(ctx: &mut GuestCtx, st: &mut VitaState) {
    let t = st.now_us();
    ctx.regs[0] = t as u32;
    ctx.regs[1] = (t >> 32) as u32;
}
