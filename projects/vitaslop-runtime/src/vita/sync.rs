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
use crate::SvcOutcome;

/// A `sceKernelTryLockMutex` failure when another thread owns the mutex (the
/// try-lock returns an error rather than blocking). Approximate value; the guest
/// only checks nonzero.
const ERR_MUTEX_FAILED_TO_OWN: u32 = 0x8002_8082;

/// `SCE_KERNEL_ERROR_UNKNOWN_SEMA_ID`: waiting on / signalling a semaphore id that
/// names no live semaphore. The kernel returns this at once instead of blocking.
const SCE_KERNEL_ERROR_UNKNOWN_SEMA_ID: u32 = 0x8002_8101;

// --- mutex ---

/// SceUID sceKernelCreateMutex(const char *name, SceUInt attr, int initCount,
///     SceKernelMutexOptParam *option)
#[hostcall]
pub(super) fn create_mutex(st: &mut VitaState, _name: Ptr, attr: u32, _init: i32, _opt: Ptr) -> i32 {
    // Recording ownership state is harmless single-thread (lock/unlock take the
    // immediate path there) and necessary for preemptive blocking.
    let id = st.create_mutex();
    // `attr` carries the waiter discipline (TH_FIFO 0x0000 vs TH_PRIO 0x2000) and the
    // recursion/ceiling bits. It is traced rather than dropped because "which discipline did
    // the guest ask for" is the first question any thread-ordering investigation asks, and a
    // silently discarded argument reads as "the guest never said".
    tracing::trace!(
        target: "vitaslop::sema",
        id, attr = format_args!("{attr:#x}"), thread = st.current_thread(), "mutex create"
    );
    id
}

/// int sceKernelLockMutex(SceUID mutexid, int lockCount, unsigned int *timeout),
/// and (with `try_lock`) int sceKernelTryLockMutex(SceUID, int).
///
/// Single-thread model: uncontended, always succeeds. Preemptive: acquire if free
/// or already held by this thread (recursive), else the plain lock parks the caller
/// ([`SvcOutcome::Block`]) while the try-lock returns an error without blocking.
pub(super) fn lock_mutex(ctx: &mut GuestCtx, st: &mut VitaState, try_lock: bool) -> SvcOutcome {
    let id = ctx.arg(0) as i32;
    if !st.is_preemptive() {
        ctx.ret(0);
        return SvcOutcome::Continue;
    }
    // `lr` is the whole point of tracing this: "which mutex" is rarely the question, "which
    // guest code took it, and did the other thread take the same one" always is. Traced on the
    // `vitaslop::sema` target alongside the semaphores and event flags, which carried every
    // other blocking primitive EXCEPT the mutexes - so a run asking what serialises two threads
    // saw everything but the answer.
    let lr = format_args!("{:#010x}", ctx.regs[14]).to_string();
    if try_lock && st.mutex_contended(id) {
        tracing::trace!(
            target: "vitaslop::sema",
            id, thread = st.current_thread(), lr, "mutex TRYLOCK refused"
        );
        ctx.ret(ERR_MUTEX_FAILED_TO_OWN);
        return SvcOutcome::Continue;
    }
    // The return value on success is 0, whether acquired now or after a wake.
    ctx.ret(0);
    if st.mutex_lock(id) {
        tracing::trace!(
            target: "vitaslop::sema",
            id, thread = st.current_thread(), lr, try_lock, "mutex lock acquired"
        );
        SvcOutcome::Continue
    } else {
        tracing::trace!(
            target: "vitaslop::sema",
            id, thread = st.current_thread(), lr, "mutex lock BLOCK"
        );
        SvcOutcome::Block
    }
}

/// int sceKernelUnlockMutex(SceUID mutexid, int unlockCount)
#[hostcall]
pub(super) fn unlock_mutex(st: &mut VitaState, id: i32, _count: i32) -> i32 {
    if st.is_preemptive() {
        tracing::trace!(target: "vitaslop::sema", id, thread = st.current_thread(), "mutex unlock");
        st.mutex_unlock(id);
    }
    0
}

// --- semaphore ---

/// SceUID sceKernelCreateSema(const char *name, SceUInt attr, int initVal,
///     int maxVal, SceKernelSemaOptParam *option)
#[hostcall]
pub(super) fn create_sema(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    name: Ptr,
    attr: u32,
    init: i32,
    max: i32,
    _opt: Ptr,
) -> i32 {
    let name = if name.addr() == 0 { String::new() } else { super::iofilemgr::read_cstr(ctx, name.addr()) };
    let id = st.create_sema(&name, attr, init, max);
    tracing::trace!(target: "vitaslop::sema", id, init, max, name, thread = st.current_thread(), "create");
    id
}

/// SceUID sceKernelOpenSema(const char *name)
///
/// Resolve an existing openable semaphore by name; it does NOT create one. Two
/// modules sharing a semaphore do it this way - one creates it named, the other opens
/// it - so returning a fresh object here would hand them two independent counters that
/// look right and synchronize nothing. A name that names nothing is an error.
#[hostcall]
pub(super) fn open_sema(ctx: &mut GuestCtx, st: &mut VitaState, name: Ptr) -> i32 {
    let name = if name.addr() == 0 { String::new() } else { super::iofilemgr::read_cstr(ctx, name.addr()) };
    match st.sema_by_name(&name) {
        Some(uid) => uid,
        None => SCE_KERNEL_ERROR_UNKNOWN_SEMA_ID as i32,
    }
}

/// Byte size of a guest `SceKernelSemaInfo`, and the offset of each field after the
/// 32-byte `name` array. Layout (vitasdk `psp2common/kernel/threadmgr.h`, asserted
/// there at 0x3C): size, semaId, name[32], attr, initCount, currentCount, maxCount,
/// numWaitThreads.
const SEMA_INFO_SIZE: u32 = 0x3C;
const SEMA_INFO_NAME: u32 = 8;
const SEMA_INFO_ATTR: u32 = 40;

/// int sceKernelGetSemaInfo(SceUID semaId, SceKernelSemaInfo *info)
///
/// The caller sets `info->size` before the call and the kernel fills the rest. Every
/// field is real state here (the name and attr from create, the count from the live
/// primitive, the waiter count from the park queue), so a title that reports or asserts
/// on a semaphore sees what the semaphore is actually doing.
#[hostcall]
pub(super) fn get_sema_info(ctx: &mut GuestCtx, st: &mut VitaState, sema_id: i32, info: Ptr) -> i32 {
    // Expression-bodied throughout: `#[hostcall]` inlines this into a `()` wrapper, so
    // an early `return` would leave the wrapper rather than this handler.
    match st.sema_info(sema_id) {
        Some((name, attr, init, current, max, waiting)) if info.addr() != 0 => {
            let addr = info.addr();
            let mut name_buf = [0u8; 32];
            let bytes = name.as_bytes();
            let n = bytes.len().min(31); // keep the terminating NUL
            name_buf[..n].copy_from_slice(&bytes[..n]);

            ctx.write_u32(addr, SEMA_INFO_SIZE);
            ctx.write_u32(addr + 4, sema_id as u32);
            ctx.write_bytes(addr + SEMA_INFO_NAME, &name_buf);
            ctx.write_u32(addr + SEMA_INFO_ATTR, attr);
            ctx.write_u32(addr + SEMA_INFO_ATTR + 4, init as u32);
            ctx.write_u32(addr + SEMA_INFO_ATTR + 8, current as u32);
            ctx.write_u32(addr + SEMA_INFO_ATTR + 12, max as u32);
            ctx.write_u32(addr + SEMA_INFO_ATTR + 16, waiting as u32);
            0
        }
        _ => SCE_KERNEL_ERROR_UNKNOWN_SEMA_ID as i32,
    }
}

/// int sceKernelWaitSema(SceUID semaid, int signal, unsigned int *timeout)
///
/// Single-thread model: take `signal` from the count (floored, never blocks).
/// Preemptive: take it if available, else park the caller until a signal delivers
/// it ([`SvcOutcome::Block`]) or the timeout passes. A satisfied wait (now or by a
/// later signal) returns 0; a `*timeout`-armed wait that expires first returns
/// `SCE_KERNEL_ERROR_WAIT_TIMEOUT`, delivered at wake through the resume-code channel
/// (the return value is set to 0 before parking, since a woken thread resumes with
/// the registers it parked with).
pub(super) fn wait_sema(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let id = ctx.arg(0) as i32;
    let signal = ctx.arg(1) as i32;
    let timeout_ptr = ctx.arg(2);
    let timeout_us = if timeout_ptr != 0 { ctx.read_u32(timeout_ptr) } else { 0 };
    // A wait on a semaphore that does not exist is not a wait at all: the real kernel
    // rejects it immediately with SCE_KERNEL_ERROR_UNKNOWN_SEMA_ID rather than parking
    // the caller forever (id 0 - an uninitialized `SceUID` - is the common case). A
    // title that reaches such a wait is on an error/cleanup path and expects the error
    // back so it can carry on, so returning it is what the hardware does.
    if st.is_preemptive() && !st.sema_exists(id) {
        ctx.ret(SCE_KERNEL_ERROR_UNKNOWN_SEMA_ID);
        return SvcOutcome::Continue;
    }
    ctx.ret(0);
    if !st.is_preemptive() {
        st.sema_wait(id, signal);
        return SvcOutcome::Continue;
    }
    if st.sema_try_acquire(id, signal) {
        tracing::trace!(target: "vitaslop::sema", id, n = signal, thread = st.current_thread(), "wait acquired");
        SvcOutcome::Continue
    } else {
        tracing::trace!(
            target: "vitaslop::sema",
            id, n = signal, thread = st.current_thread(), lr = format_args!("{:#010x}", ctx.regs[14]),
            "wait BLOCK"
        );
        st.sema_block(id, signal, timeout_us);
        SvcOutcome::Block
    }
}

/// int sceKernelSignalSema(SceUID semaid, int signal)
#[hostcall]
pub(super) fn signal_sema(st: &mut VitaState, id: i32, signal: i32) -> i32 {
    tracing::trace!(target: "vitaslop::sema", id, n = signal, thread = st.current_thread(), "signal");
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
pub(super) fn create_cond(st: &mut VitaState, _name: Ptr, _attr: u32, mutex: i32, _opt: Ptr) -> i32 {
    st.create_cond(mutex)
}

/// int sceKernelWaitCond(SceUID condId, unsigned int *timeout)
///
/// Single-thread model: nothing else can signal, so the wait returns immediately
/// (the mutex stays held) - correct for the degenerate single-thread use.
/// Preemptive: release the mutex and park the caller until a signal delivers it
/// back with the mutex re-acquired ([`SvcOutcome::Block`]), or the timeout passes -
/// on which the caller still re-acquires the mutex but the wait returns
/// `SCE_KERNEL_ERROR_WAIT_TIMEOUT` (via the resume-code channel). A null timeout
/// waits forever.
pub(super) fn wait_cond(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let id = ctx.arg(0) as i32;
    let timeout_ptr = ctx.arg(1);
    let timeout_us = if timeout_ptr != 0 { ctx.read_u32(timeout_ptr) } else { 0 };
    ctx.ret(0);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    tracing::trace!(target: "vitaslop::sema", cond = id, timeout_us, thread = st.current_thread(), lr = format_args!("{:#010x}", ctx.regs[14]), "cond WAIT");
    st.cond_wait(id, timeout_us);
    SvcOutcome::Block
}

/// int sceKernelSignalCond(SceUID condId) / sceKernelSignalCondAll(SceUID condId)
pub(super) fn signal_cond(ctx: &mut GuestCtx, st: &mut VitaState, all: bool) {
    let id = ctx.arg(0) as i32;
    tracing::trace!(target: "vitaslop::sema", cond = id, all, thread = st.current_thread(), lr = format_args!("{:#010x}", ctx.regs[14]), "cond SIGNAL");
    if st.is_preemptive() {
        st.cond_signal(id, all);
    }
    ctx.ret(0);
}

// --- event flag ---

/// SceUID sceKernelCreateEventFlag(const char *name, int attr, int initPattern,
///     SceKernelEventFlagOptParam *opt)
///
/// The NAME is read only to be REPORTED. A wait report that says a thread is parked on
/// "eventflag uid=0x118" names a host-assigned integer the guest never chose and nothing
/// in the title can be searched for; the guest's own name for it ("NU::Geom::Request", and
/// so on) is the difference between a uid and a subsystem. `sceKernelCreateSema` has
/// always recorded its name for exactly this reason - the event flag was the one primitive
/// that dropped it.
#[hostcall]
pub(super) fn create_event_flag(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    name: Ptr,
    _attr: u32,
    init: u32,
    _opt: Ptr,
) -> i32 {
    let name =
        if name.addr() == 0 { String::new() } else { super::iofilemgr::read_cstr(ctx, name.addr()) };
    let id = st.create_event_flag(init);
    st.name_event_flag(id, &name);
    tracing::trace!(
        target: "vitaslop::sema",
        id, init, name, thread = st.current_thread(),
        lr = format_args!("{:#010x}", ctx.regs[14]).to_string(),
        "evf create"
    );
    id
}

/// int sceKernelSetEventFlag(SceUID evid, unsigned int bitPattern)
/// Preemptive: also releases any parked waiters the new pattern satisfies.
#[hostcall]
pub(super) fn set_event_flag(ctx: &mut GuestCtx, st: &mut VitaState, id: i32, bits: u32) -> i32 {
    // `lr` is what makes a set/wait pair searchable: a flag that is WAITED on and never SET is
    // either a subsystem the title does not use or a signal this engine is failing to deliver,
    // and the two are indistinguishable from the uid alone. The setter's call site names the
    // guest code, which is the only way to read the difference out of the binary.
    tracing::trace!(
        target: "vitaslop::sema",
        id, bits, thread = st.current_thread(),
        lr = format_args!("{:#010x}", ctx.regs[14]).to_string(),
        "evf set"
    );
    if st.is_preemptive() {
        st.event_set_wake(id, bits);
    } else {
        st.event_set(id, bits);
    }
    0
}

/// int sceKernelClearEventFlag(SceUID evid, unsigned int bitPattern)
#[hostcall]
pub(super) fn clear_event_flag(st: &mut VitaState, id: i32, bits: u32) -> i32 {
    st.event_clear(id, bits);
    0
}

/// int sceKernelWaitEventFlag(SceUID evid, unsigned int bits, unsigned int wait,
///     unsigned int *outBits, SceUInt *timeout)
///
/// Single-thread model: report the current pattern and succeed (workers ran
/// synchronously, so whatever would set the bits already ran).
///
/// Preemptive: a REAL wait. If the pattern already satisfies `bits` under the
/// wait mode, apply the mode's clear op and return; otherwise PARK the caller
/// until a `sceKernelSetEventFlag` satisfies it (the match pattern is delivered
/// to `outBits` at wake through the scheduler's stat-write channel) or the
/// timeout passes. A stub that returns success without blocking makes every
/// waiter a busy-spin - tens of millions of no-op host calls that starve the
/// threads doing real work.
///
/// The return value is fixed at 0 before parking (a woken thread resumes inside
/// the call with the registers it parked with), so a timed-out wait also reads
/// as success with the then-current pattern in `outBits`; a caller distinguishes
/// by re-checking its condition, which is exactly what the wait-in-a-loop shape
/// that uses timeouts does.
pub(super) fn wait_event_flag(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let id = ctx.arg(0) as i32;
    let bits = ctx.arg(1);
    let mode = ctx.arg(2);
    let out = ctx.arg(3);
    let timeout_ptr = ctx.arg(4);
    ctx.ret(0);
    if !st.is_preemptive() {
        let pattern = st.event_pattern(id);
        if out != 0 {
            ctx.write_u32(out, pattern);
        }
        return SvcOutcome::Continue;
    }
    match st.evf_try_wait(id, bits, mode) {
        Some(at_match) => {
            tracing::trace!(target: "vitaslop::sema", id, bits, mode, thread = st.current_thread(), "evf wait satisfied");
            if out != 0 {
                ctx.write_u32(out, at_match);
            }
            SvcOutcome::Continue
        }
        None => {
            let timeout_us = if timeout_ptr != 0 { ctx.read_u32(timeout_ptr) } else { 0 };
            tracing::trace!(
                target: "vitaslop::sema",
                id, bits, mode, timeout_us, thread = st.current_thread(),
                lr = format_args!("{:#010x}", ctx.regs[14]).to_string(),
                "evf wait BLOCK"
            );
            st.evf_block(id, bits, mode, out, timeout_us);
            SvcOutcome::Block
        }
    }
}

/// `SCE_KERNEL_ERROR_EVF_COND` (`psp2/kernel/error.h`): a POLL whose condition the current
/// pattern does not satisfy. This is the whole difference between poll and wait - the wait
/// parks, the poll says "not yet" - so it must be a real error rather than a success with a
/// stale pattern, which a caller polling in a loop cannot tell from the bits being set.
const SCE_KERNEL_ERROR_EVF_COND: u32 = 0x8002_80E3;

/// int sceKernelPollEventFlag(int evid, unsigned int bits, unsigned int wait,
///     unsigned int *outBits)
///
/// The non-blocking `sceKernelWaitEventFlag`: identical test, identical clear-on-match
/// side effect, but it reports `SCE_KERNEL_ERROR_EVF_COND` instead of parking when the
/// pattern does not satisfy the condition.
///
/// It shares [`VitaState::evf_try_wait`] with the wait, deliberately: the wait mode's
/// AND/OR semantics and its CLEAR_ALL / CLEAR_PAT side effects are the subtle part, and a
/// second copy of them here would be a second thing to get wrong. `outBits` receives the
/// pattern AT MATCH (before the mode's clear), which is what the waiting path delivers too.
///
/// In the non-preemptive single-thread model there is nothing that could set the bits
/// later, so - exactly as [`wait_event_flag`] argues - whatever would have set them has
/// already run, and the current pattern is reported as a success.
#[hostcall]
pub(super) fn poll_event_flag(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: i32,
    bits: u32,
    mode: u32,
    out: Ptr,
) -> i32 {
    do_poll_event_flag(ctx, st, id, bits, mode, out.addr())
}

/// See [`poll_event_flag`]. Split out because a `#[hostcall]` body cannot early-return.
fn do_poll_event_flag(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: i32,
    bits: u32,
    mode: u32,
    out: u32,
) -> i32 {
    if !st.is_preemptive() {
        let pattern = st.event_pattern(id);
        if out != 0 {
            ctx.write_u32(out, pattern);
        }
        return 0;
    }
    match st.evf_try_wait(id, bits, mode) {
        Some(at_match) => {
            tracing::trace!(
                target: "vitaslop::sema",
                id, bits, mode, thread = st.current_thread(),
                "evf poll satisfied"
            );
            if out != 0 {
                ctx.write_u32(out, at_match);
            }
            0
        }
        None => {
            // The out-parameter is deliberately NOT written on a failed poll: there is no
            // matched pattern to report, and writing the current one would hand a caller
            // that ignores the return code a pattern that never satisfied its condition.
            tracing::trace!(
                target: "vitaslop::sema",
                id, bits, mode, thread = st.current_thread(),
                "evf poll NOT satisfied"
            );
            SCE_KERNEL_ERROR_EVF_COND as i32
        }
    }
}

// --- delete (shared: no teardown needed for these lightweight handles) ---

/// int sceKernelDelete{Mutex,Sema,EventFlag}(SceUID id), and sceKernelCloseMutex - all
/// succeed. See `nid::sync::CLOSE_MUTEX` for why close and delete share this.
#[hostcall]
pub(super) fn delete_object(_st: &mut VitaState, _id: i32) -> i32 {
    0
}

// --- time ---

/// SceUInt64 sceKernelGetSystemTimeWide(void)
/// A 64-bit return goes in r0 (low) and r1 (high), so this is hand-written rather
/// than `#[hostcall]` (whose value returns are 32-bit). Time is the virtual
/// monotonic clock, so it never goes backward.
pub(super) fn get_system_time_wide(ctx: &mut GuestCtx, st: &mut VitaState) {
    let t = st.now_us();
    ctx.regs[0] = t as u32;
    ctx.regs[1] = (t >> 32) as u32;
}
