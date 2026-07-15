//! SceThreadmgr: the thread-manager primitives not wrapped by SceLibKernel. The
//! user-facing create/start/wait wrappers live in `libkernel`; what remains here
//! are the direct primitives a program can also call.

use crate::hostcall;

/// SceUID sceKernelGetProcessId(void)
/// A single process, so a fixed nonzero id is faithful and stable.
#[hostcall]
pub(super) fn get_process_id(_st: &mut VitaState) -> i32 {
    0x1000
}

/// int sceKernelDelayThread(SceUInt delay)
/// Time is virtual and host-driven, so a delay does not actually sleep; it just
/// succeeds. (The monotonic clock advances at the host's chosen cadence, not from
/// guest sleeps, which keeps runs deterministic.)
#[hostcall]
pub(super) fn delay_thread(_st: &mut VitaState, _delay_us: u32) -> i32 {
    0
}
