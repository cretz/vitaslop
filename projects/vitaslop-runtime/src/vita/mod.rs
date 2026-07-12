//! Vita host-call implementations, grouped by module (one file per Sce* module,
//! mirroring the vita-headers layout). Each module exposes `try_dispatch`, which
//! returns `Some(outcome)` if it handles the function NID. The top-level
//! `dispatch` tries them in turn and records anything unhandled.

pub mod ctrl;
pub mod display;
pub mod gxm;
pub mod sysmem;

use crate::host::{GuestCtx, VitaState};
use crate::{nid, SvcOutcome};

/// Route a NID call to the module that implements it. Function NIDs are globally
/// unique, so trying modules in turn cannot misroute; `library_nid` is only for
/// logging. Unimplemented calls are recorded and return 0 so the run continues
/// and the gap is visible in the capture.
pub fn dispatch(
    library_nid: u32,
    func_nid: u32,
    ctx: &mut GuestCtx,
    st: &mut VitaState,
) -> SvcOutcome {
    if let Some(outcome) = gxm::try_dispatch(func_nid, ctx, st) {
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
    st.capture.note_unimplemented(library_nid, func_nid, nid::name(func_nid));
    ctx.ret(0);
    SvcOutcome::Continue
}
