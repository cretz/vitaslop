//! SceProcessmgr: process parameters and the standard IO handles the libc crt
//! reads while starting up.
//!
//! `sceKernelGetProcessParam` is the important one: libc's `module_start` calls it
//! to find the main module's `SceProcessParam` (the "PSP2" block), then follows its
//! `SceLibcParam` pointer to size the heap. The linker locates that block in the
//! image ([`crate::link::LinkedProgram::process_param`]); the host holds its address
//! and hands it back here. Returning 0 would make the crt dereference NULL.

use crate::host::{FD_STDERR, FD_STDOUT};
use crate::hostcall;

/// The fd reported for stdin. Reads from it hit the empty console input.
const FD_STDIN: i32 = 0;

/// SceKernelProcessParam *sceKernelGetProcessParam(void)
/// Returns the main module's `SceProcessParam` address (0 if the title has none).
#[hostcall]
pub(super) fn get_process_param(st: &mut VitaState) -> u32 {
    st.process_param()
}

/// int sceKernelGetStdin(void) / Stdout / Stderr - the standard IO fds. These map
/// onto the same fds the virtual filesystem and console capture already use.
#[hostcall]
pub(super) fn get_stdin(_st: &mut VitaState) -> i32 {
    FD_STDIN
}

#[hostcall]
pub(super) fn get_stdout(_st: &mut VitaState) -> i32 {
    FD_STDOUT
}

#[hostcall]
pub(super) fn get_stderr(_st: &mut VitaState) -> i32 {
    FD_STDERR
}

/// time_t sceKernelLibcTime(time_t *tloc)
/// Seconds since the Unix epoch from the host wall clock (0 under the deterministic
/// world), written through `tloc` when non-NULL and also returned.
#[hostcall]
pub(super) fn libc_time(ctx: &mut GuestCtx, st: &mut VitaState, tloc: Ptr) -> i32 {
    let secs = (st.world.wall_us() / 1_000_000) as i32;
    if !tloc.is_null() {
        ctx.write_u32(tloc.addr(), secs as u32);
    }
    secs
}

/// clock_t sceKernelLibcClock(void)
/// CPU time consumed by the process in CLOCKS_PER_SEC (1 MHz on the Vita's newlib) -
/// the virtual monotonic clock is the faithful stand-in. A title polls this every
/// frame for its delta-time, so a constant 0 freezes its animations.
#[hostcall]
pub(super) fn libc_clock(st: &mut VitaState) -> u32 {
    st.world.monotonic_us() as u32
}
