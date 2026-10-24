//! SceProcessmgr: process parameters and the standard IO handles the libc crt
//! reads while starting up.
//!
//! `sceKernelGetProcessParam` is the important one: libc's `module_start` calls it
//! to find the main module's `SceProcessParam` (the "PSP2" block), then follows its
//! `SceLibcParam` pointer to size the heap. The linker locates that block in the
//! image ([`crate::link::LinkedProgram::process_param`]); the host holds its address
//! and hands it back here. Returning 0 would make the crt dereference NULL.

use crate::SvcOutcome;
use crate::host::{FD_STDERR, FD_STDOUT, GuestCtx, VitaState};
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
    let secs = (st.guest_wall_us() / 1_000_000) as i32;
    if !tloc.is_null() {
        ctx.write_u32(tloc.addr(), secs as u32);
    }
    secs
}

/// The kernel's broken-down time: the nine `int` fields of `struct tm` (`sec min hour mday
/// mon(0-11) year wday(0=Sunday) yday isdst`) - but with the FULL year, not `year - 1900`.
/// MEASURED in the SceLibc module titles ship: its `gmtime_r` subtracts 1900 from the year the
/// kernel call wrote (only when it is at least 1900), and its `mktime` adds 1900 before the
/// call and takes it back off after.
fn write_kernel_tm(ctx: &mut GuestCtx, out: u32, secs: i64) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (year, month, day) = super::services::civil_from_days(days);
    let yday = days - super::services::days_from_civil(year, 1, 1);
    let fields = [
        rem % 60,
        rem / 60 % 60,
        rem / 3_600,
        day,
        month - 1,
        year,
        (days + 4).rem_euclid(7),
        yday,
        0,
    ];
    for (i, v) in fields.iter().enumerate() {
        ctx.write_u32(out + 4 * i as u32, *v as i32 as u32);
    }
}

/// struct tm *sceKernelLibcGmtime_r(const time_t *t, struct tm *out) and its
/// `sceKernelLibcLocaltime_r` twin: the 32-bit seconds at `t` broken down into a kernel
/// `struct tm` ([`write_kernel_tm`]), returning `out`; NULL for a NULL pointer. Local time IS
/// UTC here - the console's zone is UTC with no DST rule, the same answer
/// `sceKernelLibcGettimeofday` gives - so the two are one function.
#[hostcall]
pub(super) fn libc_gmtime_r(ctx: &mut GuestCtx, t: Ptr, out: Ptr) -> u32 {
    if t.is_null() || out.is_null() {
        0
    } else {
        let secs = i64::from(ctx.read_u32(t.addr()));
        write_kernel_tm(ctx, out.addr(), secs);
        out.addr()
    }
}

/// int sceKernelLibcMktime(struct tm *tm, time_t *out, SceUInt64 *out64): the inverse of
/// [`libc_gmtime_r`], over the same full-year kernel `struct tm`. Fields may be out of range
/// (a month of 13, a day of 0) and are normalised as C's `mktime` does; the normalised fields,
/// with `wday` and `yday` filled in, are written back, the seconds land in `*out` (32-bit) and
/// `*out64`, and the answer is 0. That shape is MEASURED from the SceLibc module titles ship:
/// its `mktime` passes a stack copy with the year plus 1900 and two out pointers, treats a
/// negative answer as failure (-1), and returns the 32-bit result. A time the 32-bit result
/// cannot hold, and a NULL `tm`, are INVALID_ARGUMENT.
#[hostcall]
pub(super) fn libc_mktime(ctx: &mut GuestCtx, tm: Ptr, out: Ptr, out64: Ptr) -> i32 {
    const SCE_KERNEL_ERROR_INVALID_ARGUMENT: u32 = 0x8002_0005;
    if tm.is_null() {
        SCE_KERNEL_ERROR_INVALID_ARGUMENT as i32
    } else {
        let field = |i: u32| i64::from(ctx.read_u32(tm.addr() + 4 * i) as i32);
        let (sec, min, hour, mday, mon, year) =
            (field(0), field(1), field(2), field(3), field(4), field(5));
        let year = year + mon.div_euclid(12);
        let days = super::services::days_from_civil(year, mon.rem_euclid(12) + 1, 1) + mday - 1;
        let secs = days * 86_400 + hour * 3_600 + min * 60 + sec;
        if !(0..=i64::from(u32::MAX)).contains(&secs) {
            SCE_KERNEL_ERROR_INVALID_ARGUMENT as i32
        } else {
            write_kernel_tm(ctx, tm.addr(), secs);
            if !out.is_null() {
                ctx.write_u32(out.addr(), secs as u32);
            }
            if !out64.is_null() {
                ctx.write_u32(out64.addr(), secs as u32);
                ctx.write_u32(out64.addr() + 4, (secs >> 32) as u32);
            }
            0
        }
    }
}

/// int sceKernelLibcGettimeofday(struct timeval *tv, struct timezone *tz)
///
/// Wall-clock time as a `{ time_t tv_sec; suseconds_t tv_usec; }` pair, both 32-bit
/// here. Taken from the SAME world wall clock `sceKernelLibcTime` reads, so the two
/// cannot disagree about what time it is. `tz` is filled with zeroes when asked for:
/// the console reports UTC with no DST rule, and leaving the caller's struct
/// untouched would let it read its own stack as a timezone offset.
#[hostcall]
pub(super) fn libc_gettimeofday(ctx: &mut GuestCtx, st: &mut VitaState, tv: Ptr, tz: Ptr) -> i32 {
    let now_us = st.guest_wall_us();
    if !tv.is_null() {
        ctx.write_u32(tv.addr(), (now_us / 1_000_000) as u32);
        ctx.write_u32(tv.addr() + 4, (now_us % 1_000_000) as u32);
    }
    if !tz.is_null() {
        ctx.write_u32(tz.addr(), 0); // tz_minuteswest
        ctx.write_u32(tz.addr() + 4, 0); // tz_dsttime
    }
    0
}

/// int sceKernelCallAbortHandler(...)
///
/// Invoke the process's registered abort handler - the path libc's `abort()` takes
/// after flushing, and the last thing that runs before the process is killed. It does
/// not return on hardware, so it must not return here either: the run ends loudly with
/// the abort named, rather than the guest carrying on past its own abort() and
/// crashing later somewhere unrelated.
pub(super) fn call_abort_handler(_ctx: &mut GuestCtx, _st: &mut VitaState) -> SvcOutcome {
    SvcOutcome::Fatal(
        "sceKernelCallAbortHandler: the guest called abort() - the title itself decided \
         it cannot continue. Look at what it printed or asserted just before this."
            .into(),
    )
}

/// clock_t sceKernelLibcClock(void)
/// CPU time consumed by the process in CLOCKS_PER_SEC (1 MHz on the Vita's newlib) -
/// the virtual monotonic clock is the faithful stand-in. A title polls this every
/// frame for its delta-time, so a constant 0 freezes its animations.
#[hostcall]
pub(super) fn libc_clock(st: &mut VitaState) -> u32 {
    st.guest_mono_us() as u32
}
