//! Vita host-call implementations, grouped by module (one file per Sce* module,
//! mirroring the vita-headers layout). Each module exposes `try_dispatch`, which
//! returns `Some(outcome)` if it handles the function NID. The top-level
//! `dispatch` tries them in turn and records anything unhandled.

pub mod cfmt;
pub mod ctrl;
pub mod display;
pub mod gxm;
pub mod iofilemgr;
pub mod libkernel;
pub mod lwsync;
pub mod processmgr;
pub mod services;
pub mod sync;
pub mod sysmem;
pub mod threadmgr;

use crate::host::{GuestCtx, VitaState};
use crate::{nid, SvcOutcome};

/// Route a NID call to the module that implements it. Function NIDs are globally
/// unique, so trying modules in turn cannot misroute; `library_nid` is only for
/// logging. Unimplemented calls are recorded and return 0 so the run continues
/// and the gap is visible in the capture.
///
/// Modules are probed hottest-first (each `try_dispatch` is one jump-table match
/// returning `None` on a miss). A real title's dominant host calls are the
/// synchronization primitives - `sceKernelWaitLwCond`/`SignalLwCond`/`LockLwMutex`
/// and `UnlockMutex` ran ~4.3M times each in an OlliOlli boot - so `lwsync`,
/// `sync`, and the `libkernel` clock calls lead, ahead of the graphics and setup
/// modules that fire a few thousand times total. Ordering is a pure performance
/// choice (correctness is order-independent), worth it at tens of millions of
/// calls per frame budget.
pub fn dispatch(
    library_nid: u32,
    func_nid: u32,
    ctx: &mut GuestCtx,
    st: &mut VitaState,
) -> SvcOutcome {
    if let Some(outcome) = lwsync::try_dispatch(func_nid, ctx, st) {
        return outcome;
    }
    if let Some(outcome) = sync::try_dispatch(func_nid, ctx, st) {
        return outcome;
    }
    if let Some(outcome) = libkernel::try_dispatch(func_nid, ctx, st) {
        return outcome;
    }
    if let Some(outcome) = threadmgr::try_dispatch(func_nid, ctx, st) {
        return outcome;
    }
    if let Some(outcome) = gxm::try_dispatch(func_nid, ctx, st) {
        return outcome;
    }
    if let Some(outcome) = iofilemgr::try_dispatch(func_nid, ctx, st) {
        return outcome;
    }
    if let Some(outcome) = sysmem::try_dispatch(func_nid, ctx, st) {
        return outcome;
    }
    if let Some(outcome) = display::try_dispatch(func_nid, ctx, st) {
        return outcome;
    }
    if let Some(outcome) = ctrl::try_dispatch(func_nid, ctx, st) {
        return outcome;
    }
    if let Some(outcome) = processmgr::try_dispatch(func_nid, ctx, st) {
        return outcome;
    }
    if let Some(outcome) = services::try_dispatch(func_nid, ctx, st) {
        return outcome;
    }
    st.capture.note_unimplemented(library_nid, func_nid, nid::name(func_nid));
    ctx.ret(0);
    SvcOutcome::Continue
}
