//! SceThreadmgr: the thread-manager primitives not wrapped by SceLibKernel. The
//! user-facing create/start/wait wrappers live in `libkernel`; what remains here
//! are the direct primitives a program can also call.

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::nid::threadmgr as nid;
use crate::SvcOutcome;

pub fn try_dispatch(func_nid: u32, ctx: &mut GuestCtx, st: &mut VitaState) -> Option<SvcOutcome> {
    match func_nid {
        nid::DELAY_THREAD => {
            delay_thread(ctx, st);
            Some(SvcOutcome::Continue)
        }
        // A thread ending itself. Under the preemptive scheduler this ends just
        // this thread ([`SvcOutcome::ThreadExit`]); in the single-thread-of-control
        // bring-up the only thread that reaches here is main's (workers return
        // normally instead), so it is a clean whole-run stop.
        nid::EXIT_THREAD | nid::EXIT_DELETE_THREAD => Some(if st.is_preemptive() {
            SvcOutcome::ThreadExit
        } else {
            SvcOutcome::Halt
        }),
        _ => None,
    }
}

/// int sceKernelDelayThread(SceUInt delay)
/// Time is virtual and host-driven, so a delay does not actually sleep; it just
/// succeeds. (The monotonic clock advances at the host's chosen cadence, not from
/// guest sleeps, which keeps runs deterministic.)
#[hostcall]
fn delay_thread(_st: &mut VitaState, _delay_us: u32) -> i32 {
    0
}
