//! SceProcessmgr: process parameters and the standard IO handles the libc crt
//! reads while starting up.
//!
//! `sceKernelGetProcessParam` is the important one: libc's `module_start` calls it
//! to find the main module's `SceProcessParam` (the "PSP2" block), then follows its
//! `SceLibcParam` pointer to size the heap. The linker locates that block in the
//! image ([`crate::link::LinkedProgram::process_param`]); the host holds its address
//! and hands it back here. Returning 0 would make the crt dereference NULL.

use crate::host::{GuestCtx, VitaState, FD_STDERR, FD_STDOUT};
use crate::hostcall;
use crate::nid::processmgr as nid;
use crate::SvcOutcome;

/// The fd reported for stdin. Reads from it hit the empty console input.
const FD_STDIN: i32 = 0;

pub fn try_dispatch(func_nid: u32, ctx: &mut GuestCtx, st: &mut VitaState) -> Option<SvcOutcome> {
    match func_nid {
        nid::GET_PROCESS_PARAM => get_process_param(ctx, st),
        nid::GET_STDIN => get_stdin(ctx, st),
        nid::GET_STDOUT => get_stdout(ctx, st),
        nid::GET_STDERR => get_stderr(ctx, st),
        nid::LIBC_TIME => libc_time(ctx, st),
        _ => return None,
    }
    Some(SvcOutcome::Continue)
}

/// SceKernelProcessParam *sceKernelGetProcessParam(void)
/// Returns the main module's `SceProcessParam` address (0 if the title has none).
#[hostcall]
fn get_process_param(st: &mut VitaState) -> u32 {
    st.process_param()
}

/// int sceKernelGetStdin(void) / Stdout / Stderr - the standard IO fds. These map
/// onto the same fds the virtual filesystem and console capture already use.
#[hostcall]
fn get_stdin(_st: &mut VitaState) -> i32 {
    FD_STDIN
}

#[hostcall]
fn get_stdout(_st: &mut VitaState) -> i32 {
    FD_STDOUT
}

#[hostcall]
fn get_stderr(_st: &mut VitaState) -> i32 {
    FD_STDERR
}

/// time_t sceKernelLibcTime(time_t *tloc)
/// Seconds since the Unix epoch from the host wall clock (0 under the deterministic
/// world), written through `tloc` when non-NULL and also returned.
#[hostcall]
fn libc_time(ctx: &mut GuestCtx, st: &mut VitaState, tloc: Ptr) -> i32 {
    let secs = (st.world.wall_us() / 1_000_000) as i32;
    if !tloc.is_null() {
        ctx.write_u32(tloc.addr(), secs as u32);
    }
    secs
}
