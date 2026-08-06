//! SceFiber: cooperative user-level threads with guest-supplied stacks.
//!
//! A fiber is a coroutine the title schedules itself: `sceFiberRun` enters one from a
//! thread, `sceFiberSwitch` hands control from one fiber straight to another, and
//! `sceFiberReturnToThread` gives control back to the thread that started the chain.
//! Only one member of a chain runs at a time and every transfer is explicit, so a
//! fiber is not a thread in the concurrency sense - but it IS a separate stack whose
//! frames stay live while it is switched away.
//!
//! That last property is the whole design constraint, and it is why these are backed
//! by the existing preemptive scheduler rather than by nesting host re-entries: a
//! switched-away fiber's stack must survive with live frames on it, which is exactly
//! what a parked scheduler thread already gives. See [`crate::host::VitaState`]'s
//! fiber section for the baton invariant.
//!
//! Two calls in this surface are NOT published anywhere:
//! `_sceFiberInitializeWithInternalOptionImpl` and `_sceFiberAttachContextAndSwitch`.
//! Both are read from their documented siblings - see their doc comments - and both
//! fail loudly rather than quietly if that reading is wrong.

use crate::host::GuestCtx;
use crate::hostcall;
use crate::SvcOutcome;

use super::iofilemgr::read_cstr;

/// Offsets in the guest `SceFiberInfo` (vitasdk `fiber.h`): `{ +0x00 entry, +0x04
/// argOnInitialize, +0x08 addrContext, +0x0c sizeContext, +0x10 name[32] }`.
const INFO_ENTRY: u32 = 0x00;
const INFO_ARG_ON_INITIALIZE: u32 = 0x04;
const INFO_ADDR_CONTEXT: u32 = 0x08;
const INFO_SIZE_CONTEXT: u32 = 0x0c;
const INFO_NAME: u32 = 0x10;
const INFO_NAME_LEN: usize = 32;

/// SceInt32 _sceFiberInitializeImpl(SceFiber *fiber, char *name, SceFiberEntry *entry,
///     SceUInt32 argOnInitialize, void *addrContext, SceSize sizeContext,
///     SceFiberOptParam *params)
///
/// Records the fiber and creates - but does not start - the thread that will run it.
/// Nothing executes until the first `sceFiberRun`/`Switch`, which is the whole point of
/// a fiber: initialising one is not running one.
///
/// `addrContext` may be null; such a fiber gets its stack from
/// `_sceFiberAttachContextAndSwitch` at its first switch.
pub(super) fn initialize(ctx: &mut GuestCtx, st: &mut crate::host::VitaState) {
    let fiber = ctx.arg(0);
    let name_ptr = ctx.arg(1);
    let entry = ctx.arg(2);
    let arg_on_initialize = ctx.arg(3);
    let context_addr = ctx.arg(4);
    let context_size = ctx.arg(5);
    let name = if name_ptr == 0 { String::new() } else { read_cstr(ctx, name_ptr) };
    tracing::debug!(
        target: "vitaslop::thread",
        fiber = format_args!("{fiber:#x}"),
        name = %name,
        entry = format_args!("{entry:#x}"),
        context = format_args!("{context_addr:#x}"),
        size = context_size,
        "fiber initialize"
    );
    let r = st.fiber_initialize(fiber, name, entry, arg_on_initialize, context_addr, context_size);
    ctx.ret(r as u32);
}

/// SceInt32 _sceFiberInitializeWithInternalOptionImpl(...)
///
/// UNPUBLISHED, and routed to [`initialize`] deliberately. Its documented sibling
/// `_sceFiberInitializeImpl` already ends in an options pointer, so the "internal
/// option" variant can only add to the tail of that argument list - the six arguments
/// this surface reads (fiber, name, entry, argOnInitialize, addrContext, sizeContext)
/// occupy the same positions either way. What the extra internal option selects is not
/// knowable from any clean source; it is not consulted, and nothing here silently
/// depends on it, because every fiber behaviour this models is fixed by the other six.
pub(super) fn initialize_with_internal_option(ctx: &mut GuestCtx, st: &mut crate::host::VitaState) {
    initialize(ctx, st);
}

/// SceInt32 sceFiberRun(SceFiber *fiber, SceUInt32 argOnRunTo, SceUInt32 *argOnRun)
///
/// Enter `fiber` from the calling THREAD and block until the chain returns here. The
/// value `sceFiberReturnToThread` passes lands in `*argOnRun` before this returns - it
/// is queued as a pending write rather than written now, because this handler does not
/// run again at wake time.
pub(super) fn run(ctx: &mut GuestCtx, st: &mut crate::host::VitaState) -> SvcOutcome {
    let fiber = ctx.arg(0);
    let arg_on_run_to = ctx.arg(1);
    let arg_on_run = ctx.arg(2);
    match st.fiber_run(fiber, arg_on_run_to, arg_on_run) {
        Ok(()) => {
            ctx.ret(0);
            SvcOutcome::Block
        }
        Err(e) => {
            ctx.ret(e as u32);
            SvcOutcome::Continue
        }
    }
}

/// SceInt32 sceFiberSwitch(SceFiber *fiber, SceUInt32 argOnRunTo, SceUInt32 *argOnRun)
///
/// Hand control from the running fiber straight to another one, without unwinding to
/// the thread. Called from anything but a fiber this is a permission error, exactly as
/// the kernel reports it - the alternative, silently treating it as a run, would nest a
/// second chain on one thread and there is no such state.
pub(super) fn switch(ctx: &mut GuestCtx, st: &mut crate::host::VitaState) -> SvcOutcome {
    switch_inner(ctx, st, None)
}

/// SceInt32 _sceFiberAttachContextAndSwitch(SceFiber *fiber, void *addrContext,
///     SceSize sizeContext, SceUInt32 argOnRunTo, SceUInt32 *argOnRun)
///
/// UNPUBLISHED. Read from its two documented halves: the name says "attach a context,
/// then switch", `_sceFiberInitializeImpl` already accepts a null context (so a fiber
/// legitimately exists without a stack until something supplies one), and
/// `sceFiberSwitch`'s own three arguments must still be passed. That gives the argument
/// order below - the context pair inserted after the fiber, ahead of the switch
/// arguments, which is how the sibling `sceFiberOptParam`-free calls are shaped.
///
/// If that reading is wrong the fiber runs on a stack that is not its context, and the
/// failure is a bounds trap on the first push rather than silent corruption, because a
/// null or unaligned context is refused outright.
pub(super) fn attach_context_and_switch(ctx: &mut GuestCtx, st: &mut crate::host::VitaState) -> SvcOutcome {
    let context = (ctx.arg(1), ctx.arg(2));
    switch_inner(ctx, st, Some(context))
}

fn switch_inner(
    ctx: &mut GuestCtx,
    st: &mut crate::host::VitaState,
    context: Option<(u32, u32)>,
) -> SvcOutcome {
    let fiber = ctx.arg(0);
    // The switch arguments sit after the context pair in the attach variant.
    let (arg_on_run_to, arg_on_run) = match context {
        Some(_) => (ctx.arg(3), ctx.arg(4)),
        None => (ctx.arg(1), ctx.arg(2)),
    };
    match st.fiber_switch(fiber, arg_on_run_to, arg_on_run, context) {
        Ok(()) => {
            ctx.ret(0);
            SvcOutcome::Block
        }
        Err(e) => {
            ctx.ret(e as u32);
            SvcOutcome::Continue
        }
    }
}

/// SceInt32 sceFiberReturnToThread(SceUInt32 argOnReturn, SceUInt32 *argOnRun)
///
/// Give control back to the thread that ran this chain. The fiber parks exactly where
/// it stands, with its stack intact, and resumes here when it is next run or switched
/// to - at which point `*argOnRun` holds the value that transfer carried.
pub(super) fn return_to_thread(ctx: &mut GuestCtx, st: &mut crate::host::VitaState) -> SvcOutcome {
    let arg_on_return = ctx.arg(0);
    let arg_on_run = ctx.arg(1);
    match st.fiber_return_to_thread(arg_on_return, arg_on_run) {
        Ok(()) => {
            ctx.ret(0);
            SvcOutcome::Block
        }
        Err(e) => {
            ctx.ret(e as u32);
            SvcOutcome::Continue
        }
    }
}

/// SceInt32 sceFiberGetSelf(SceFiber **fiber)
///
/// The fiber the caller is running, or null on a plain thread. vitasdk types the
/// argument as `SceFiber *`, but the call returns an `SceInt32` status and has nowhere
/// else to put its answer, so it is an out-pointer.
#[hostcall]
pub(super) fn get_self(ctx: &mut GuestCtx, st: &mut VitaState, out: Ptr) -> i32 {
    if out.is_null() {
        crate::host::VitaState::FIBER_ERROR_NULL
    } else {
        ctx.write_u32(out.addr(), st.current_fiber());
        0
    }
}

/// SceInt32 sceFiberFinalize(SceFiber *fiber)
#[hostcall]
pub(super) fn finalize(st: &mut VitaState, fiber: u32) -> i32 {
    st.fiber_finalize(fiber)
}

/// SceInt32 sceFiberGetInfo(SceFiber *fiber, SceFiberInfo *fiberInfo)
///
/// The whole struct is written (name buffer included, zero-padded) so a caller never
/// reads its own uninitialised stack back as fiber configuration.
#[hostcall]
pub(super) fn get_info(ctx: &mut GuestCtx, st: &mut VitaState, fiber: u32, out: Ptr) -> i32 {
    if out.is_null() {
        return_null_error()
    } else {
        match st.fiber_info(fiber) {
            Some((entry, arg_on_initialize, context_addr, context_size, name)) => {
                let out = out.addr();
                ctx.write_u32(out + INFO_ENTRY, entry);
                ctx.write_u32(out + INFO_ARG_ON_INITIALIZE, arg_on_initialize);
                ctx.write_u32(out + INFO_ADDR_CONTEXT, context_addr);
                ctx.write_u32(out + INFO_SIZE_CONTEXT, context_size);
                let mut buf = [0u8; INFO_NAME_LEN];
                let n = name.len().min(INFO_NAME_LEN - 1);
                buf[..n].copy_from_slice(&name.as_bytes()[..n]);
                ctx.write_bytes(out + INFO_NAME, &buf);
                0
            }
            None => crate::host::VitaState::FIBER_ERROR_INVALID,
        }
    }
}

/// `#[hostcall]` bodies cannot use an early `return`, so the null-pointer errno is a
/// call rather than a statement.
fn return_null_error() -> i32 {
    crate::host::VitaState::FIBER_ERROR_NULL
}
