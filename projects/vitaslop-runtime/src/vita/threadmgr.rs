//! SceThreadmgr: the thread-manager primitives not wrapped by SceLibKernel. The
//! user-facing create/start/wait wrappers live in `libkernel`; what remains here
//! are the direct primitives a program can also call.

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::SvcOutcome;

/// SceUID sceKernelGetProcessId(void)
/// A single process, so a fixed nonzero id is faithful and stable.
#[hostcall]
pub(super) fn get_process_id(_st: &mut VitaState) -> i32 {
    0x1000
}

/// int sceKernelGetThreadCurrentPriority(void)
/// The scheduler priority of the calling thread (lower value = higher priority).
/// A title reads this to spawn a worker at a relative priority or to briefly raise
/// its own; returning the real running priority keeps that arithmetic correct.
#[hostcall]
pub(super) fn get_thread_current_priority(st: &mut VitaState) -> i32 {
    st.current_priority()
}

/// int sceKernelDeleteThread(SceUID thid)
/// Delete a DORMANT thread: drop its record so its SceUID stops resolving, and give its
/// stack back for reuse. A running thread cannot be deleted (`NOT_DORMANT`) - accepting
/// that would invalidate the id of a thread the scheduler is about to resume, which is a
/// corruption rather than an error. See [`VitaState::delete_thread`].
#[hostcall]
pub(super) fn delete_thread(st: &mut VitaState, thid: i32) -> i32 {
    match st.delete_thread(thid) {
        Ok(()) => 0,
        Err(e) => e as i32,
    }
}

/// int sceKernelChangeThreadVfpException(int clearMask, int setMask)
///
/// Selects which VFP/NEON floating-point exceptions (invalid, div-by-zero, overflow,
/// underflow, inexact, input-denormal) trap for the calling thread by clearing then
/// setting bits in its FPSCR exception-enable field. We evaluate every float and NEON
/// op with standard non-trapping IEEE semantics (no host trap is ever raised), so this
/// only records intent - it never changes a numeric result or control flow. Accepted
/// with success. Returns 0.
#[hostcall]
pub(super) fn change_thread_vfp_exception(_clear_mask: i32, _set_mask: i32) -> i32 {
    0
}

/// int sceKernelDelayThread(SceUInt delay)
///
/// Preemptive: a REAL timed sleep - park the caller until the virtual clock
/// reaches `now + delay` ([`VitaState::sleep_park`]); the scheduler wakes it on a
/// display flip's clock advance, or jumps the clock straight to the deadline when
/// nothing else is runnable. A no-op "just succeed" here turns every
/// delay-then-poll loop into a full-speed busy spin (millions of host calls
/// starving the threads doing real work).
///
/// Single-thread model: still a no-op success (workers run synchronously, so
/// there is nothing to yield to and the clock is host-driven).
pub(super) fn delay_thread(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let delay_us = ctx.arg(0);
    ctx.ret(0);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    // A zero/one-us delay is "give someone else the CPU", not a real sleep - and not
    // a display frame either (see [`SvcOutcome::Flip`]). A worker polling in a
    // delay(0) loop hits this thousands of times per rendered frame.
    if delay_us <= 1 {
        return SvcOutcome::Reschedule;
    }
    st.sleep_park(delay_us as u64);
    SvcOutcome::Block
}
