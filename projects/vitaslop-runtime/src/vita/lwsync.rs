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

/// Size of `SceKernelLwMutexWork` (`SceInt64 data[4]`), in bytes.
pub(super) const LW_MUTEX_WORK_SIZE: usize = 32;
/// Size of `SceKernelLwCondWork` (`SceInt32 data[4]`), in bytes. Distinct from the
/// mutex work size - a cond work area is half as large, and zeroing the mutex size
/// over it would overrun into whatever the caller placed immediately after.
pub(super) const LW_COND_WORK_SIZE: usize = 16;

/// int sceKernelWaitLwCond(SceKernelLwCondWork *work, SceUInt32 *ppTimeout)
/// Preemptive: park the calling thread on the cond so the scheduler can run the
/// thread that will signal it; a non-null timeout also arms a deadline wake. In the
/// single-thread model nothing contends, so the wait is immediate success.
pub(super) fn wait_lw_cond(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let work = ctx.arg(0);
    let timeout_ptr = ctx.arg(1);
    let timeout = if timeout_ptr != 0 { ctx.read_u32(timeout_ptr) } else { 0 };
    sample_wait(st, work, timeout_ptr, timeout);
    // The wait returns 0 whether satisfied now or after a wake.
    ctx.ret(0);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    st.lwcond_wait(work, timeout);
    SvcOutcome::Block
}

/// int sceKernelSignalLwCond(SceKernelLwCondWork *work) and friends.
pub(super) fn signal_lw_cond(ctx: &mut GuestCtx, st: &mut VitaState, all: bool) {
    let work = ctx.arg(0);
    if st.is_preemptive() {
        st.lwcond_signal(work, all);
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

/// Create a lightweight object: zero its caller-provided work area (arg 0 is the
/// work pointer for both `CreateLwMutex` and `CreateLwCond`) to the "unlocked, no
/// owner" initial state, then report success. `size` must be the exact work-area
/// size for the primitive - over-zeroing corrupts whatever the caller placed after
/// the work area (a cond work area is only 16 bytes, half the mutex's 32).
pub(super) fn init_work(ctx: &mut GuestCtx, size: usize) {
    let work = ctx.arg(0);
    ctx.write_bytes(work, &[0u8; LW_MUTEX_WORK_SIZE][..size]);
    ctx.ret(0);
}

/// The shared lock/unlock/wait/signal/delete outcome: success.
pub(super) fn succeed(ctx: &mut GuestCtx) {
    ctx.ret(0);
}
