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

use crate::host::{GuestCtx, VitaState, SCE_KERNEL_ERROR_WAIT_TIMEOUT};
use crate::hostcall;
use crate::SvcOutcome;

/// A `sceKernelTryLockMutex` failure when another thread owns the mutex (the
/// try-lock returns an error rather than blocking). Approximate value; the guest
/// only checks nonzero.
const ERR_MUTEX_FAILED_TO_OWN: u32 = 0x8002_8082;

/// `SCE_KERNEL_ERROR_UNKNOWN_SEMA_ID`: waiting on / signalling a semaphore id that
/// names no live semaphore. The kernel returns this at once instead of blocking.
const SCE_KERNEL_ERROR_UNKNOWN_SEMA_ID: u32 = 0x8002_8101;
/// `SCE_KERNEL_ERROR_UNKNOWN_MUTEX_ID`: the mutex sibling, from the same error table.
const SCE_KERNEL_ERROR_UNKNOWN_MUTEX_ID: u32 = 0x8002_8141;

/// `SCE_KERNEL_ERROR_UNKNOWN_COND_ID` (`psp2/kernel/error.h`): a wait on a condition-variable
/// uid that names no condition variable.
const SCE_KERNEL_ERROR_UNKNOWN_COND_ID: u32 = 0x8002_81A1;
/// `SCE_KERNEL_ERROR_SEMA_OVF`: a `sceKernelSignalSema` whose count would exceed the
/// semaphore's `maxCount`. From vitasdk `psp2/kernel/error.h`.
const SCE_KERNEL_ERROR_SEMA_OVF: u32 = 0x8002_8103;

// --- mutex ---

/// SceUID sceKernelCreateMutex(const char *name, SceUInt attr, int initCount,
///     SceKernelMutexOptParam *option)
#[hostcall]
pub(super) fn create_mutex(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    name: Ptr,
    attr: u32,
    init: i32,
    _opt: Ptr,
) -> i32 {
    // Recording ownership state is harmless single-thread (lock/unlock take the
    // immediate path there) and necessary for preemptive blocking.
    // `ctx` is here for the KERNEL MUTEX TABLE, not for an argument: the mutex claims its
    // entry in guest memory at create ([`crate::vita::kmutex`]).
    // The name, attributes and initial count are kept for `sceKernelGetMutexInfo`.
    let name = if name.addr() == 0 { String::new() } else { super::iofilemgr::read_cstr(ctx, name.addr()) };
    let id = st.create_mutex(ctx, &name, attr, init);
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
    if try_lock && st.mutex_contended(ctx, id) {
        tracing::trace!(
            target: "vitaslop::sema",
            id, thread = st.current_thread(), lr, "mutex TRYLOCK refused"
        );
        ctx.ret(ERR_MUTEX_FAILED_TO_OWN);
        return SvcOutcome::Continue;
    }
    // The return value on success is 0, whether acquired now or after a wake.
    ctx.ret(0);
    if st.mutex_lock(ctx, id) {
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
pub(super) fn unlock_mutex(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: i32,
    _count: i32,
) -> i32 {
    if st.is_preemptive() {
        tracing::trace!(target: "vitaslop::sema", id, thread = st.current_thread(), "mutex unlock");
        st.mutex_unlock(ctx, id);
    }
    0
}

/// Which of this module's NIDs the transpiler may emit INLINE.
///
/// Only the uncontended take and release of a KERNEL MUTEX, and only under the preemptive
/// scheduler, because that is the only run with a host-mirror block to hold the table and the
/// only one where a mutex has more than one thread to contend with. See
/// [`vitaslop_transpiler::InlineOp::KernelMutexLock`] for the state machine and
/// [`crate::vita::kmutex`] for where the state lives.
///
/// `sceKernelTryLockMutex` is deliberately NOT here: its refusal is a return value the handler
/// defines, and the count argument it does not take makes it a different call, not the same one
/// with a different name.
pub(crate) fn inline_op(
    func_nid: u32,
    preemptive: bool,
) -> Option<vitaslop_transpiler::InlineOp> {
    use crate::nid::sync as sy;
    use crate::vita::kmutex;
    use crate::vita::mirror::{SLOT_CURRENT_THREAD, SLOT_MUTEX_TABLE};
    if !preemptive {
        return None;
    }
    let (layout, thread_slot, table_slot, entries) =
        (kmutex::layout(), SLOT_CURRENT_THREAD, SLOT_MUTEX_TABLE, kmutex::ENTRIES);
    match func_nid {
        sy::LOCK_MUTEX => Some(vitaslop_transpiler::InlineOp::KernelMutexLock {
            layout,
            thread_slot,
            table_slot,
            entries,
        }),
        sy::UNLOCK_MUTEX => Some(vitaslop_transpiler::InlineOp::KernelMutexUnlock {
            layout,
            thread_slot,
            table_slot,
            entries,
        }),
        _ => None,
    }
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

/// Layout of `SceKernelMutexInfo`: `{ size, mutexId, name[32], attr, initCount, currentCount,
/// currentOwnerId, numWaitThreads }` - 0x3C bytes.
const MUTEX_INFO_SIZE: u32 = 0x3C;
const MUTEX_INFO_NAME: u32 = 8;
const MUTEX_INFO_ATTR: u32 = 40;

/// int sceKernelGetMutexInfo(SceUID mutexid, SceKernelMutexInfo *info)
///
/// The same shape as [`get_sema_info`]: the mutex's creation parameters plus its live
/// ownership - current recursion count, owning thread (0 when free) and the number of threads
/// parked on it. A sports title's engine reads this at boot.
#[hostcall]
pub(super) fn get_mutex_info(ctx: &mut GuestCtx, st: &mut VitaState, mutex_id: i32, info: Ptr) -> i32 {
    // Expression-bodied, for the reason `get_sema_info` states.
    match st.mutex_info(ctx, mutex_id) {
        Some((name, attr, init, current, owner, waiting)) if info.addr() != 0 => {
            let addr = info.addr();
            let mut name_buf = [0u8; 32];
            let bytes = name.as_bytes();
            let n = bytes.len().min(31);
            name_buf[..n].copy_from_slice(&bytes[..n]);
            ctx.write_u32(addr, MUTEX_INFO_SIZE);
            ctx.write_u32(addr + 4, mutex_id as u32);
            ctx.write_bytes(addr + MUTEX_INFO_NAME, &name_buf);
            ctx.write_u32(addr + MUTEX_INFO_ATTR, attr);
            ctx.write_u32(addr + MUTEX_INFO_ATTR + 4, init as u32);
            ctx.write_u32(addr + MUTEX_INFO_ATTR + 8, current as u32);
            ctx.write_u32(addr + MUTEX_INFO_ATTR + 12, owner as u32);
            ctx.write_u32(addr + MUTEX_INFO_ATTR + 16, waiting as u32);
            0
        }
        _ => SCE_KERNEL_ERROR_UNKNOWN_MUTEX_ID as i32,
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
    // A NULL POINTER AND A POINTER TO ZERO ARE DIFFERENT CALLS. Null is "park until
    // signalled"; a pointer to 0 is "do not wait at all". Carried as an `Option` for exactly
    // that reason - the old `u32` could not tell them apart, and `sema_block` reads a zero as
    // INFINITE, so every polling caller was parked forever. See
    // [`report_zero_timeout_poll`], and `lock_lw_mutex` for the same fix on the mutex.
    let timeout_us = (timeout_ptr != 0).then(|| ctx.read_u32(timeout_ptr));
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
    } else if timeout_us == Some(0) {
        // A POLL the count could not satisfy. The deadline is "now", so this must answer
        // exactly what a deadline that expires answers (`VitaState::advance_time_to`).
        report_zero_timeout_poll("sema", id, st.current_thread());
        ctx.ret(SCE_KERNEL_ERROR_WAIT_TIMEOUT);
        SvcOutcome::Continue
    } else {
        tracing::trace!(
            target: "vitaslop::sema",
            id, n = signal, thread = st.current_thread(), lr = format_args!("{:#010x}", ctx.regs[14]),
            "wait BLOCK"
        );
        if sema_trail_watched(id) {
            tracing::warn!(
                target: "vitaslop::warning",
                id = format_args!("{id:#x}").to_string(), thread = st.current_thread(),
                lr = format_args!("{:#010x}", ctx.regs[14]).to_string(),
                trail = %super::guest_return_trail(ctx, 192),
                "SEMA TRAIL: a thread PARKED on a watched semaphore; the trail is the plausible saved return addresses on its stack, innermost first"
            );
        }
        st.sema_block(id, signal, timeout_us.unwrap_or(0));
        SvcOutcome::Block
    }
}

/// A wait whose `*timeout` is ZERO is a POLL, and this says so the first time each
/// primitive does it.
///
/// # Why this is a report and not a trace
/// The value a guest passes here separates two calls that look identical at the ABI: a NULL
/// `pTimeout` means "park until signalled", and a pointer to the value 0 means "do not wait
/// at all - tell me now if you cannot". This engine used to collapse both onto an infinite
/// park, so every caller of the second form was parked FOREVER on a primitive nothing was
/// ever going to signal. MEASURED on a retail sports title: its `LoopSimulation` polled an
/// asset manager's semaphore through `EA::Thread::Semaphore::Wait(kTimeoutImmediate)` - the
/// eboot carries both EA constants, `0xFFFFFFFFFFFFFFFF` at 140 sites and ZERO at the six in
/// that subsystem - and never came back; the stall surfaced 160 frames later as a display
/// frame that never ended.
///
/// So the population this line names is exactly the population whose behaviour CHANGED. A
/// guest that polls in a loop is perfectly normal and needs no fix; a guest that polls a
/// primitive we would previously have parked on is the evidence that this fix reaches it, and
/// a run that never prints one is a run where the fix is not being exercised. ONE LINE PER
/// PRIMITIVE - a poll in a loop polls every frame, and a per-call warning would bury the
/// finding it exists to make [[vitaslop-a-diagnostic-can-bury-the-findings]].
pub(super) fn report_zero_timeout_poll(primitive: &str, id: i32, thread: i32) {
    // Fact space: this call site's own tag XOR the primitive it names. Deliberately NOT keyed
    // on the uid - the fact is "this primitive gets polled at all", and keying on the uid
    // would print a line per object for a title that polls a pool of them.
    let mut key = 0x7a17_0011_0000_0000_u64;
    for b in primitive.as_bytes() {
        key = key.rotate_left(8) ^ u64::from(*b);
    }
    if !crate::rtt_writeback::report_once(key) {
        return;
    }
    tracing::warn!(
        target: "vitaslop::warning",
        primitive,
        id = format_args!("{id:#x}").to_string(),
        thread,
        "a guest POLLED this primitive - *timeout == 0, which means DO NOT WAIT - and the \
         condition was not available, so it is answered with WAIT_TIMEOUT and NOT parked. \
         This engine used to read a pointer-to-zero the same way as a NULL pointer and park \
         such a caller forever; this line names the population that fix reaches."
    );
}

/// TEMPORARY DIAGNOSTIC. `VITASLOP_SEMA_TRAIL=<uid>[,<uid>...]` (hex with `0x`, else
/// decimal): when a thread PARKS on one of these semaphores, print the plausible saved
/// return addresses on its stack. A `sceKernelWaitSema` names only the thin EA wrapper in
/// `lr`; the subsystem that decided to wait is several frames up, and a park that never
/// ends has no other witness - the thread makes no further host calls, so a call-site
/// tally sees it as an absence.
fn sema_trail_watched(id: i32) -> bool {
    static WATCH: std::sync::OnceLock<Vec<i32>> = std::sync::OnceLock::new();
    let watch = WATCH.get_or_init(|| {
        std::env::var("VITASLOP_SEMA_TRAIL")
            .unwrap_or_default()
            .split(',')
            .filter_map(|s| {
                let s = s.trim();
                match s.strip_prefix("0x") {
                    Some(hex) => i32::from_str_radix(hex, 16).ok(),
                    None if s.is_empty() => None,
                    None => s.parse::<i32>().ok(),
                }
            })
            .collect()
    });
    watch.contains(&id)
}

/// Whether this semaphore's overflow refusal has not been reported yet. One line per
/// semaphore: a handshake that hits its ceiling hits it every frame, and a per-call warning
/// would bury the finding it is there to make [[a-diagnostic-can-bury-the-findings]].
fn sema_ovf_first_time(id: i32) -> bool {
    static SEEN: std::sync::Mutex<Vec<i32>> = std::sync::Mutex::new(Vec::new());
    let Ok(mut seen) = SEEN.lock() else { return false };
    if seen.contains(&id) {
        return false;
    }
    seen.push(id);
    true
}

/// Say ONCE per uid that a thread waited on an event flag that does not exist.
///
/// A WARNING, not a status line: the answer this path gives CHANGED from "park forever" to an
/// error, so a title whose behaviour moves when it starts hearing the truth is one we owe a
/// look [[vitaslop-a-warning-means-we-owe-a-fix]]. It is also the one line that names the uid,
/// and a uid of 0 is the whole tell - an uninitialised `SceUID` means the CREATE never ran or
/// never reached this thread, which is the bug upstream of the wait.
fn report_wait_on_unknown_evf(id: i32, bits: u32, mode: u32) {
    static SEEN: std::sync::Mutex<Vec<i32>> = std::sync::Mutex::new(Vec::new());
    let Ok(mut seen) = SEEN.lock() else { return };
    if seen.contains(&id) {
        return;
    }
    seen.push(id);
    tracing::warn!(
        target: "vitaslop::warning",
        id = format_args!("{id:#x}").to_string(),
        bits = format_args!("{bits:#x}").to_string(),
        mode = format_args!("{mode:#x}").to_string(),
        "sceKernelWaitEventFlag on a uid that names no event flag - UNKNOWN_EVF_ID. The device \
         refuses this immediately; parking on it is a permanent stall whose symptom surfaces \
         hundreds of frames later somewhere else. A uid of 0 means the CREATE never happened."
    );
}

/// int sceKernelSignalSema(SceUID semaid, int signal)
///
/// A signal that would push the count past the semaphore's `maxCount` is REFUSED with
/// `SCE_KERNEL_ERROR_SEMA_OVF` and changes nothing - see
/// [`VitaState::sema_signal_overflows`] for why silently exceeding it is a desync rather than
/// a harmless slack. Reported the first time it happens per semaphore, because a title whose
/// handshake relies on the ceiling is relying on a refusal this engine never used to give.
#[hostcall]
pub(super) fn signal_sema(st: &mut VitaState, id: i32, signal: i32) -> i32 {
    tracing::trace!(target: "vitaslop::sema", id, n = signal, thread = st.current_thread(), "signal");
    match st.sema_signal_overflows(id, signal) {
        None => {
            // A signal naming no live semaphore. The kernel refuses it, and this used to answer
            // 0 - "delivered" - which is a lie a title can build on. Reported once per uid
            // because the answer CHANGED, and a title whose picture moves when it starts
            // hearing the truth is a title we owe a look.
            if sema_ovf_first_time(id) {
                tracing::warn!(
                    target: "vitaslop::warning",
                    id = format_args!("{id:#x}").to_string(), thread = st.current_thread(),
                    "sceKernelSignalSema on a uid that names no semaphore - UNKNOWN_SEMA_ID"
                );
            }
            SCE_KERNEL_ERROR_UNKNOWN_SEMA_ID as i32
        }
        Some(true) => {
            if sema_ovf_first_time(id) {
                let (count, max) = st.sema_info(id).map(|i| (i.3, i.4)).unwrap_or((0, 0));
                tracing::warn!(
                    target: "vitaslop::warning",
                    id = format_args!("{id:#x}").to_string(), signal, count, max,
                    "sceKernelSignalSema REFUSED (SEMA_OVF): the count is already at the maximum \
                     this semaphore was created with, so the device refuses it too"
                );
            }
            SCE_KERNEL_ERROR_SEMA_OVF as i32
        }
        Some(false) => {
            if st.is_preemptive() {
                st.sema_signal_wake(id, signal);
            } else {
                st.sema_signal(id, signal);
            }
            0
        }
    }
}

/// int sceKernelCancelSema(SceUID semaId, int setCount, int *numWaitThreads)
///
/// Resets the count - a negative `setCount` means the initial count - and releases every
/// thread waiting on it with `SCE_KERNEL_ERROR_WAIT_CANCEL`. The release is done by
/// signalling the waiters' worth of count first (the same wake path `sceKernelSignalSema`
/// takes), then setting the count; a waiter that wakes this way returns 0 rather than the
/// cancel error, which is the one deviation and is REPORTED when it happens.
#[hostcall]
pub(super) fn cancel_sema(ctx: &mut GuestCtx, st: &mut VitaState, id: i32, set_count: i32, out: Ptr) -> i32 {
    match st.sema_cancel(id, set_count) {
        Some(waiters) => {
            if waiters > 0 {
                tracing::warn!(
                    target: "vitaslop::warning",
                    id = format_args!("{id:#x}").to_string(), waiters,
                    "sceKernelCancelSema released waiters by SIGNALLING them - they return 0, not SCE_KERNEL_ERROR_WAIT_CANCEL"
                );
            }
            if !out.is_null() {
                ctx.write_u32(out.addr(), waiters as u32);
            }
            0
        }
        None => uid_not_found(st, id, "sceKernelCancelSema"),
    }
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
    // Null is "wait forever"; a pointer to 0 is "do not wait" - see
    // [`report_zero_timeout_poll`] for why collapsing the two was a permanent park.
    let timeout_us = (timeout_ptr != 0).then(|| ctx.read_u32(timeout_ptr));
    // A wait on a condition variable that does not exist parks the caller with NO WAITER
    // RECORDED - `cond_wait` returns early and this function blocks anyway - so nothing can
    // ever wake it and nothing can even see it waiting. See
    // [[vitaslop-a-wait-on-a-uid-that-names-nothing-must-be-an-error]].
    if st.is_preemptive() && !st.cond_exists(id) {
        report_wait_on_unknown_cond(id);
        ctx.ret(SCE_KERNEL_ERROR_UNKNOWN_COND_ID);
        return SvcOutcome::Continue;
    }
    ctx.ret(0);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    if timeout_us == Some(0) {
        // A POLL. A timed-out cond wait re-acquires its mutex before returning, so a wait
        // that never RELEASED it is already in the right state: answer WAIT_TIMEOUT without
        // calling `cond_wait`, which is the call that would have dropped the mutex.
        report_zero_timeout_poll("cond", id, st.current_thread());
        ctx.ret(SCE_KERNEL_ERROR_WAIT_TIMEOUT);
        return SvcOutcome::Continue;
    }
    tracing::trace!(target: "vitaslop::sema", cond = id, timeout_us, thread = st.current_thread(), lr = format_args!("{:#010x}", ctx.regs[14]), "cond WAIT");
    st.cond_wait(ctx, id, timeout_us.unwrap_or(0));
    SvcOutcome::Block
}

/// Say ONCE per uid that a thread waited on a condition variable that does not exist.
fn report_wait_on_unknown_cond(id: i32) {
    static SEEN: std::sync::Mutex<Vec<i32>> = std::sync::Mutex::new(Vec::new());
    let Ok(mut seen) = SEEN.lock() else { return };
    if seen.contains(&id) {
        return;
    }
    seen.push(id);
    tracing::warn!(
        target: "vitaslop::warning",
        id = format_args!("{id:#x}").to_string(),
        "sceKernelWaitCond on a uid that names no condition variable - UNKNOWN_COND_ID. Parking \
         on it records no waiter anywhere, so nothing can wake it and nothing can see it waiting - \
         not even the stall watchdog. A uid of 0 means the CREATE never happened."
    );
}

/// int sceKernelSignalCond(SceUID condId) / sceKernelSignalCondAll(SceUID condId)
pub(super) fn signal_cond(ctx: &mut GuestCtx, st: &mut VitaState, all: bool) {
    let id = ctx.arg(0) as i32;
    tracing::trace!(target: "vitaslop::sema", cond = id, all, thread = st.current_thread(), lr = format_args!("{:#010x}", ctx.regs[14]), "cond SIGNAL");
    if st.is_preemptive() {
        st.cond_signal(ctx, id, all);
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
    // A wait on an event flag that does not exist is not a wait - see
    // [`SCE_KERNEL_ERROR_UNKNOWN_EVF_ID`], which is where the measurement is.
    if st.is_preemptive() && !st.evf_exists(id) {
        report_wait_on_unknown_evf(id, bits, mode);
        ctx.ret(SCE_KERNEL_ERROR_UNKNOWN_EVF_ID);
        return SvcOutcome::Continue;
    }
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
            // Null is "wait forever"; a pointer to 0 is "do not wait" - see
            // [`report_zero_timeout_poll`].
            let timeout_us = (timeout_ptr != 0).then(|| ctx.read_u32(timeout_ptr));
            if timeout_us == Some(0) {
                // A POLL whose bits are not set. NOT `SCE_KERNEL_ERROR_EVF_COND`: that is
                // `sceKernelPollEventFlag`'s own answer. This is a WAIT whose deadline is
                // NOW, so it must agree with the deadline path in `advance_time_to`, which
                // wakes an expired evf waiter with WAIT_TIMEOUT.
                report_zero_timeout_poll("evf", id, st.current_thread());
                ctx.ret(SCE_KERNEL_ERROR_WAIT_TIMEOUT);
                return SvcOutcome::Continue;
            }
            tracing::trace!(
                target: "vitaslop::sema",
                id, bits, mode, timeout_us, thread = st.current_thread(),
                lr = format_args!("{:#010x}", ctx.regs[14]).to_string(),
                "evf wait BLOCK"
            );
            st.evf_block(id, bits, mode, out, timeout_us.unwrap_or(0));
            SvcOutcome::Block
        }
    }
}

/// `SCE_KERNEL_ERROR_EVF_COND` (`psp2/kernel/error.h`): a POLL whose condition the current
/// pattern does not satisfy. This is the whole difference between poll and wait - the wait
/// parks, the poll says "not yet" - so it must be a real error rather than a success with a
/// stale pattern, which a caller polling in a loop cannot tell from the bits being set.
const SCE_KERNEL_ERROR_EVF_COND: u32 = 0x8002_80E3;

/// `SCE_KERNEL_ERROR_UNKNOWN_EVF_ID` (`psp2/kernel/error.h`): a wait on an event-flag uid that
/// names no event flag.
///
/// >>> THIS IS THE DIFFERENCE BETWEEN AN ERROR AND A PERMANENT STALL, AND IT WAS MEASURED.
/// PCSE00084's loader thread calls `sceKernelWaitEventFlag(0x0, 1, 4, 0)` - uid **0**, an
/// uninitialised `SceUID` - exactly once, and parked there for the rest of the run. The title
/// screen then kept queueing UI commands with nothing draining them until it overran its own
/// 256 KB command buffer and faulted 400 frames later, which is a very long way from the
/// instruction that actually went wrong. The real kernel rejects the wait immediately and the
/// caller carries on down its error path. `wait_sema` already does this for semaphores and
/// names id 0 as the common case; the event-flag path did not.
const SCE_KERNEL_ERROR_UNKNOWN_EVF_ID: u32 = 0x8002_80E1;

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

/// Timer errors, from `psp2/kernel/error.h`:
///   `SCE_KERNEL_ERROR_TIMER_COUNTING` - start on a timer already counting
///   `SCE_KERNEL_ERROR_TIMER_STOPPED`  - stop on a timer that is not counting
///   `SCE_KERNEL_ERROR_UNKNOWN_TIMER_ID` - a uid that is not a timer
const SCE_KERNEL_ERROR_TIMER_COUNTING: u32 = 0x8002_7303;
const SCE_KERNEL_ERROR_TIMER_STOPPED: u32 = 0x8002_7304;
const SCE_KERNEL_ERROR_UNKNOWN_TIMER_ID: u32 = 0x8002_8241;

// --- virtual timers ---
//
// A Vita timer is a stopwatch, not an alarm: create it, start it, read it back. The
// titles seen so far use it exactly that way (the one that brought this family in
// creates one called "System Debug Timer" and reads it for its own profiling), so the
// counting half is implemented and the EVENT half - `sceKernelSetTimerEvent`, which
// wakes a thread on a schedule - deliberately is not. Nothing imports it, and a stub
// that accepted one would be a timer that silently never fires.
//
// The count is DERIVED from `st.now_us()` rather than ticked, so it is exact under a
// scheduler that does not run in real time. See `VitaState::timer_time_us`.
//
// vitasdk publishes these NIDs with no header; prototypes are from the henkaku wiki.

/// SceUID sceKernelCreateTimer(const char *name, SceUInt32 attr,
///     const SceKernelTimerOptParam *opt)
///
/// Created STOPPED with a count of zero: `sceKernelStartTimer` is a separate call.
#[hostcall]
pub(super) fn create_timer(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    name: Ptr,
    attr: u32,
    _opt: Ptr,
) -> i32 {
    let name =
        if name.addr() == 0 { String::new() } else { super::iofilemgr::read_cstr(ctx, name.addr()) };
    let id = st.create_timer(&name, attr);
    tracing::trace!(
        target: "vitaslop::sema",
        id, attr, name, thread = st.current_thread(),
        "timer create"
    );
    id
}

/// SceUID sceKernelOpenTimer(const char *name)
///
/// Resolves an EXISTING timer by name, the way `sceKernelOpenSema` does. A name that
/// was never created is the guest asking for something that is not there, so it gets
/// the not-found error rather than a fresh timer it would then read as zero forever.
#[hostcall]
pub(super) fn open_timer(ctx: &mut GuestCtx, st: &mut VitaState, name: Ptr) -> i32 {
    let name =
        if name.addr() == 0 { String::new() } else { super::iofilemgr::read_cstr(ctx, name.addr()) };
    match st.timer_by_name(&name) {
        Some(id) => id,
        None => {
            tracing::warn!(
                target: "vitaslop::warning",
                name, thread = st.current_thread(),
                "sceKernelOpenTimer: no timer with this name exists"
            );
            SCE_KERNEL_ERROR_UNKNOWN_TIMER_ID as i32
        }
    }
}

/// int sceKernelStartTimer(SceUID timerId)
///
/// Zero if it started, and the "already counting" error if it was running - which is
/// REPORTED rather than treated as a restart, because a silent restart would throw away
/// however long the timer had already counted.
#[hostcall]
pub(super) fn start_timer(_ctx: &mut GuestCtx, st: &mut VitaState, id: i32) -> i32 {
    match st.timer_start(id) {
        Some(false) => 0,
        Some(true) => SCE_KERNEL_ERROR_TIMER_COUNTING as i32,
        None => uid_not_found(st, id, "sceKernelStartTimer"),
    }
}

/// int sceKernelStopTimer(SceUID timerId)
#[hostcall]
pub(super) fn stop_timer(_ctx: &mut GuestCtx, st: &mut VitaState, id: i32) -> i32 {
    match st.timer_stop(id) {
        Some(true) => 0,
        Some(false) => SCE_KERNEL_ERROR_TIMER_STOPPED as i32,
        None => uid_not_found(st, id, "sceKernelStopTimer"),
    }
}

/// int sceKernelGetTimerTime(SceUID timerId, SceUInt64 *time)
///
/// The count in microseconds. The pointer is optional in the sense that a title may
/// pass null and take the value from the return, which is why the write is guarded.
#[hostcall]
pub(super) fn get_timer_time(ctx: &mut GuestCtx, st: &mut VitaState, id: i32, out: Ptr) -> i32 {
    // One expression, no early return: a `#[hostcall]` body cannot return early - the
    // macro wraps it and the `return` would leave through the wrapper's `()`.
    match st.timer_time_us(id) {
        Some(us) => {
            if out.addr() != 0 {
                ctx.write_u32(out.addr(), us as u32);
                ctx.write_u32(out.addr() + 4, (us >> 32) as u32);
            }
            0
        }
        None => uid_not_found(st, id, "sceKernelGetTimerTime"),
    }
}

/// SceUInt64 sceKernelGetTimerTimeWide(SceUID timerId)
///
/// The same count as `sceKernelGetTimerTime`, RETURNED in r0:r1 (a 64-bit return, which
/// `#[hostcall]` cannot express, so it is marshalled by hand like
/// `sceRtcGetAccumulativeTime`). The unknown-uid case has no error channel - the return IS
/// the value - so it is reported and reads as zero, which is what a stopped timer reads.
pub(super) fn get_timer_time_wide(ctx: &mut GuestCtx, st: &mut VitaState) {
    let id = ctx.arg(0) as i32;
    let us = match st.timer_time_us(id) {
        Some(us) => us,
        None => {
            uid_not_found(st, id, "sceKernelGetTimerTimeWide");
            0
        }
    };
    ctx.regs[0] = us as u32;
    ctx.regs[1] = (us >> 32) as u32;
}

/// int sceKernelDeleteTimer(SceUID timerId)
#[hostcall]
pub(super) fn delete_timer(_ctx: &mut GuestCtx, st: &mut VitaState, id: i32) -> i32 {
    if st.timer_delete(id) { 0 } else { uid_not_found(st, id, "sceKernelDeleteTimer") }
}

/// The one report for a timer call naming a uid that does not exist. It is a WARNING:
/// either the title opened a timer we failed to create, or it is using a uid from a
/// family we have not implemented, and both are ours to fix.
fn uid_not_found(st: &VitaState, id: i32, what: &str) -> i32 {
    tracing::warn!(
        target: "vitaslop::warning",
        id = format_args!("{id:#x}").to_string(), thread = st.current_thread(),
        "{what}: no timer with this uid"
    );
    SCE_KERNEL_ERROR_UNKNOWN_TIMER_ID as i32
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

#[cfg(test)]
mod zero_timeout_tests {
    use super::*;
    use crate::world::DeterministicWorld;
    use crate::{SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::{REG_COUNT, SP};

    /// A preemptive state and a guest image, with the registers a call arrives with.
    fn with<R>(f: impl FnOnce(&mut GuestCtx, &mut VitaState) -> R) -> R {
        let mut st = VitaState::new(0, 0x4000, Box::new(DeterministicWorld::default()));
        st.set_preemptive(true);
        st.set_current(1);
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = vec![0u8; 0x4000];
        let mut mem = SliceMemory(&mut bytes);
        let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        f(&mut ctx, &mut st)
    }

    /// Guest address of the `SceUInt32` a polling caller points `pTimeout` at.
    const TIMEOUT_PTR: u32 = 0x900;
    /// Guest address of `sceKernelWaitEventFlag`'s `outBits`.
    const OUT_PTR: u32 = 0x910;
    /// A stack pointer for the one call whose `pTimeout` is a STACK argument.
    const SP_ADDR: u32 = 0xA00;

    /// >>> THE WHOLE POINT. A POINTER TO ZERO IS NOT A NULL POINTER.
    ///
    /// Null `pTimeout` means "park until signalled"; a pointer to the value 0 means "do not
    /// wait at all". This engine read a zero as infinite and parked every polling caller
    /// forever - MEASURED as a retail sports title's `LoopSimulation` polling an asset
    /// manager's semaphore through `EA::Thread::Semaphore::Wait(kTimeoutImmediate)` and never
    /// coming back.
    ///
    /// Asserted on the OUTCOME and the return value, not on a log line: a diagnostic can be
    /// right while the behaviour is wrong, and it is the `SvcOutcome::Block` that is the park.
    #[test]
    fn a_semaphore_poll_times_out_and_does_not_park() {
        with(|ctx, st| {
            let id = st.create_sema("poll", 0, 0, 1);
            ctx.write_u32(TIMEOUT_PTR, 0);
            ctx.regs[0] = id as u32;
            ctx.regs[1] = 1;
            ctx.regs[2] = TIMEOUT_PTR;
            let out = wait_sema(ctx, st);
            assert!(matches!(out, SvcOutcome::Continue), "a poll must NOT park");
            assert_eq!(ctx.regs[0], SCE_KERNEL_ERROR_WAIT_TIMEOUT, "and must say why");

            // The count being available is still a plain success through the same path.
            st.sema_signal(id, 1);
            ctx.regs[0] = id as u32;
            ctx.regs[1] = 1;
            ctx.regs[2] = TIMEOUT_PTR;
            assert!(matches!(wait_sema(ctx, st), SvcOutcome::Continue));
            assert_eq!(ctx.regs[0], 0, "an available count is taken, not refused");
        });
    }

    /// The NEGATIVE CONTROL for all four, and the reason the fix is an `Option` rather than a
    /// second comparison against 0: a null `pTimeout` must still park forever. Without this
    /// the fix could pass its own tests by never blocking at all.
    #[test]
    fn a_null_timeout_still_parks_forever() {
        with(|ctx, st| {
            let id = st.create_sema("wait", 0, 0, 1);
            ctx.regs[0] = id as u32;
            ctx.regs[1] = 1;
            ctx.regs[2] = 0; // NULL - wait forever
            assert!(matches!(wait_sema(ctx, st), SvcOutcome::Block), "null still parks");
        });
    }

    /// A cond poll must not park - and must not RELEASE THE MUTEX either. A timed-out cond
    /// wait re-acquires its mutex before returning, so the state a poll leaves behind has to
    /// be the state it arrived with; dropping the mutex and handing back a timeout would be a
    /// silent unlock the caller never asked for.
    #[test]
    fn a_cond_poll_times_out_without_releasing_the_mutex() {
        with(|ctx, st| {
            let mutex = st.create_mutex(ctx, "m", 0, 0);
            let id = st.create_cond(mutex);
            assert!(st.mutex_lock(ctx, mutex), "the caller holds it going in");
            ctx.write_u32(TIMEOUT_PTR, 0);
            ctx.regs[0] = id as u32;
            ctx.regs[1] = TIMEOUT_PTR;
            let out = wait_cond(ctx, st);
            assert!(matches!(out, SvcOutcome::Continue), "a poll must NOT park");
            assert_eq!(ctx.regs[0], SCE_KERNEL_ERROR_WAIT_TIMEOUT);
            let owner = st.mutex_info(ctx, mutex).expect("a live mutex").4;
            assert_eq!(owner, 1, "the mutex is still ours - a poll must not unlock it");
        });
    }

    /// An event-flag poll whose bits are not set times out rather than parking. NOT
    /// `SCE_KERNEL_ERROR_EVF_COND`: that is `sceKernelPollEventFlag`'s own answer, and this is
    /// a WAIT whose deadline is now, so it must agree with the deadline path.
    #[test]
    fn an_event_flag_poll_times_out_and_does_not_park() {
        with(|ctx, st| {
            let id = st.create_event_flag(0);
            ctx.write_u32(TIMEOUT_PTR, 0);
            ctx.regs[0] = id as u32;
            ctx.regs[1] = 1; // waiting on bit 0, which is clear
            ctx.regs[2] = 0;
            ctx.regs[3] = OUT_PTR;
            // `pTimeout` is argument FIVE, so AAPCS puts it on the STACK, not in r4 -
            // `GuestCtx::arg(4)` reads `[sp]`. Getting this wrong is what the first run of
            // this test caught, and it would have silently tested nothing.
            ctx.regs[SP] = SP_ADDR;
            ctx.write_u32(SP_ADDR, TIMEOUT_PTR);
            let out = wait_event_flag(ctx, st);
            assert!(matches!(out, SvcOutcome::Continue), "a poll must NOT park");
            assert_eq!(ctx.regs[0], SCE_KERNEL_ERROR_WAIT_TIMEOUT);

            // A pattern that IS satisfied still succeeds through the same call.
            st.event_set_wake(id, 1);
            ctx.regs[0] = id as u32;
            ctx.regs[1] = 1;
            ctx.regs[2] = 0;
            ctx.regs[3] = OUT_PTR;
            ctx.regs[SP] = SP_ADDR;
            ctx.write_u32(SP_ADDR, TIMEOUT_PTR);
            assert!(matches!(wait_event_flag(ctx, st), SvcOutcome::Continue));
            assert_eq!(ctx.regs[0], 0, "a satisfied wait is not a timeout");
        });
    }
}
