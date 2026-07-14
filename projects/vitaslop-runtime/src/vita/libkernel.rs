//! SceLibKernel: the user-facing clib (string/memory/print) plus process and
//! thread control. This is the module a plain C program leans on the most.
//!
//! Two shapes of handler live here:
//!   - Pure clib string/memory calls use `#[hostcall]`: a typed signature with
//!     the AAPCS marshalling generated (same as the sysmem handlers).
//!   - The `printf` family is VARIADIC, which the fixed-signature macro cannot
//!     express, so those are written by hand. They drive the shared C formatter
//!     ([`crate::vita::cfmt`]), which walks the variadic tail out of the core
//!     registers and stack (never the VFP file - variadic floats are promoted to
//!     double and passed in the integer sequence per AAPCS).
//!
//! `sceKernelExitProcess` is control flow, not a value call: it returns
//! [`SvcOutcome::Halt`] so the driver unwinds and stops the run cleanly (the
//! blob-free replacement for the cube's "halt on sceGxmTerminate" hack).

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::nid::libkernel as nid;
use crate::vita::cfmt;
use crate::SvcOutcome;

/// Bound on an unbounded string read (strcmp), so a missing NUL cannot make us
/// scan the whole guest address space.
const MAX_STR: usize = 4096;

/// The SceUID reported for the implicit main thread (bring-up: one thread of
/// control, plus synchronously-run workers).
const MAIN_THREAD_ID: i32 = 0x40;

pub fn try_dispatch(func_nid: u32, ctx: &mut GuestCtx, st: &mut VitaState) -> Option<SvcOutcome> {
    match func_nid {
        // Variadic print family (hand-written).
        nid::CLIB_PRINTF => clib_printf(ctx, st),
        nid::CLIB_SNPRINTF => clib_snprintf(ctx, st),
        // Pure clib memory/string (macro-marshalled). memmove shares memcpy's
        // read-then-write impl, which already tolerates overlap.
        nid::CLIB_MEMCPY | nid::CLIB_MEMMOVE => clib_memcpy(ctx, st),
        nid::CLIB_MEMSET => clib_memset(ctx, st),
        nid::CLIB_MEMCMP => clib_memcmp(ctx, st),
        nid::CLIB_STRNLEN => clib_strnlen(ctx, st),
        nid::CLIB_STRNCPY => clib_strncpy(ctx, st),
        nid::CLIB_STRNCMP => clib_strncmp(ctx, st),
        nid::CLIB_STRCMP => clib_strcmp(ctx, st),
        // Threads. create/start record state; the actual worker run happens either
        // synchronously in the engine host (single-thread re-entry) or as a spawned
        // fiber (preemptive), depending on the mode (see host.rs).
        nid::CREATE_THREAD => create_thread(ctx, st),
        nid::START_THREAD => start_thread(ctx, st),
        // Join can block under the preemptive scheduler, so it returns the outcome.
        nid::WAIT_THREAD_END => return Some(wait_thread_end(ctx, st)),
        nid::GET_THREAD_ID => get_thread_id(ctx, st),
        nid::GET_THREAD_EXIT_STATUS => get_thread_exit_status(ctx, st),
        // Thread-local storage: a per-thread pointer slot keyed by an integer.
        nid::GET_TLS_ADDR => get_tls_addr(ctx, st),
        // 64-bit process runtime in microseconds (r0 low, r1 high).
        nid::GET_PROCESS_TIME_WIDE => return Some(get_process_time_wide(ctx, st)),
        // Process control: unwind the run. r0 (the exit code) is left as the
        // guest set it; the host treats any exit as a clean stop.
        nid::EXIT_PROCESS => {
            trace_exit(ctx, st);
            return Some(SvcOutcome::Halt);
        }
        _ => return None,
    }
    Some(SvcOutcome::Continue)
}

/// Diagnostic (env-gated by `VITASLOP_TRACE_EXIT`): when the guest calls
/// `sceKernelExitProcess`, dump the exit code, the immediate caller (LR), and a
/// window of the stack top. Return addresses saved on the stack (even words that
/// point into the code image) reveal the call chain that decided to quit, so the
/// deciding function can be disassembled. Zero cost when the env var is unset.
fn trace_exit(ctx: &mut GuestCtx, st: &VitaState) {
    if std::env::var("VITASLOP_TRACE_EXIT").is_err() {
        return;
    }
    // The last serviced calls, tagged by thread, so the exiting (main) thread's final
    // decisions are legible apart from any worker's interleaved calls.
    let trace = &st.capture.trace;
    let thids = &st.capture.trace_thid;
    // Optionally write the WHOLE thread-tagged trace to a file (VITASLOP_TRACE_FILE) so
    // the pre-exit decision region can be examined, not just the exit-machinery tail.
    if let Ok(path) = std::env::var("VITASLOP_TRACE_FILE") {
        let mut out = String::new();
        for i in 0..trace.len() {
            let thid = thids.get(i).copied().unwrap_or(0);
            out.push_str(&format!("{i} t{thid:#x} {}\n", crate::nid::name(trace[i])));
        }
        let _ = std::fs::write(&path, out);
    }
    let start = trace.len().saturating_sub(30);
    eprintln!("[exit] last {} calls (idx thid name):", trace.len() - start);
    for i in start..trace.len() {
        let thid = thids.get(i).copied().unwrap_or(0);
        eprintln!("[exit]   {i} t{thid:#x} {}", crate::nid::name(trace[i]));
    }
    let r0 = ctx.regs[0];
    let lr = ctx.regs[14];
    let sp = ctx.regs[13];
    eprintln!("[exit] code={r0:#x} (r0={} signed) lr={lr:#010x} sp={sp:#010x}", r0 as i32);
    eprintln!("[exit] r0..r12: {:08x?}", &ctx.regs[0..13]);
    // Print stack words; flag any that fall inside the loaded code image (a plausible
    // return address, Thumb or ARM) so the manual backtrace is quick. The image spans
    // [base, base + ~5 MiB); use a generous 8 MiB window to stay title-agnostic.
    let base = ctx.base;
    let code_end = base.wrapping_add(0x0080_0000);
    for i in 0..48u32 {
        let a = sp.wrapping_add(i * 4);
        let v = ctx.read_u32(a);
        let tag = if v >= base && v < code_end { "  <- code?" } else { "" };
        eprintln!("[exit]   sp+{:<3} {a:#010x}: {v:#010x}{tag}", i * 4);
    }
}

/// SceUID sceKernelCreateThread(const char *name, SceKernelThreadEntry entry,
///     int initPriority, SceSize stackSize, SceUInt attr, int cpuAffinityMask,
///     const SceKernelThreadOptParam *option)
/// The last three args sit past r3 on the stack; `#[hostcall]` reads them there.
#[hostcall]
fn create_thread(
    st: &mut VitaState,
    _name: Ptr,
    entry: Ptr,
    _prio: i32,
    stack_size: u32,
    _attr: u32,
    _cpu: i32,
    _opt: Ptr,
) -> i32 {
    st.create_thread(entry.addr(), stack_size)
}

/// int sceKernelStartThread(SceUID thid, SceSize arglen, void *argp)
/// Under the preemptive scheduler the worker runs later, not synchronously, so the
/// argument block must be *snapshotted now*: the kernel copies `arglen` bytes from
/// `argp` to the new thread before it runs, and callers rely on that - `argp` is
/// almost always a stack temporary in the caller's frame that is reused (overwritten)
/// long before the worker reads it. Copy it into a stable heap buffer and hand the
/// worker that copy; without this the worker reads garbage for its argument.
#[hostcall]
fn start_thread(ctx: &mut GuestCtx, st: &mut VitaState, thid: i32, arglen: u32, argp: Ptr) -> i32 {
    let arg_ptr = if arglen > 0 && argp.addr() != 0 {
        let bytes = ctx.read_bytes(argp.addr(), arglen as usize);
        let buf = st.galloc(arglen, 8);
        ctx.write_bytes(buf, &bytes);
        buf
    } else {
        argp.addr()
    };
    st.start_thread(thid, arglen, arg_ptr);
    0
}

/// int sceKernelWaitThreadEnd(SceUID thid, int *stat, SceUInt *timeout)
///
/// Single-thread model: the worker already ran (synchronously at start), so its
/// exit code is available immediately - write it through `stat` and succeed.
/// Preemptive: if the target is still running, park the caller
/// ([`SvcOutcome::Block`]) until it ends; the wake means it has finished. NOTE:
/// `stat` is written only when the target had already finished at the call - the
/// blocked path cannot write it at wake time (the handler does not re-run and has
/// no memory access there), so a caller needing the code across a real wait should
/// read it another way. Callers that pass NULL (the common case) are unaffected.
fn wait_thread_end(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let thid = ctx.arg(0) as i32;
    let stat = ctx.arg(1);
    ctx.ret(0);
    if st.is_preemptive() && !st.join_block(thid, stat) {
        // Parked; the target is still running. `join_block` recorded the waiter and
        // its `stat` pointer, so the exit code is delivered when the target ends (the
        // handler cannot re-run at wake time - see `VitaState::take_stat_writes`).
        return SvcOutcome::Block;
    }
    // Either single-thread (worker already ran) or the target was already finished.
    let code = st.thread_exit_code(thid).unwrap_or(0);
    if stat != 0 {
        ctx.write_u32(stat, code);
    }
    SvcOutcome::Continue
}

/// SceUID sceKernelGetThreadId(void)
/// Reports the running thread's id: the scheduler's `current` under preemption, or
/// a fixed main-thread id in the single-thread-of-control bring-up.
#[hostcall]
fn get_thread_id(st: &mut VitaState) -> i32 {
    if st.is_preemptive() {
        st.current_thread()
    } else {
        MAIN_THREAD_ID
    }
}

/// int sceKernelGetThreadExitStatus(SceUID thid, int *pExitStatus)
/// Writes the thread's exit code and succeeds; a never-finished or unknown thread
/// reports 0. (Single-thread: workers already ran to completion by the time this is
/// asked.)
#[hostcall]
fn get_thread_exit_status(ctx: &mut GuestCtx, st: &mut VitaState, thid: i32, out: Ptr) -> i32 {
    let code = st.thread_exit_code(thid).unwrap_or(0);
    if !out.is_null() {
        ctx.write_u32(out.addr(), code);
    }
    0
}

/// SceUInt64 sceKernelGetProcessTimeWide(void)
/// The 64-bit process-runtime clock in microseconds. Returned in r0 (low)/r1 (high),
/// so it is hand-written rather than `#[hostcall]`. Uses the virtual monotonic clock.
fn get_process_time_wide(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    // The virtual monotonic clock the scheduler advances (jumping over idle waits),
    // so a timed wait loop reads real elapsed time instead of a frozen value.
    let t = st.now_us();
    ctx.regs[0] = t as u32;
    ctx.regs[1] = (t >> 32) as u32;
    SvcOutcome::Continue
}

/// void *sceKernelGetTLSAddr(int key)
/// Returns this thread's storage slot for `key`, a stable zero-initialized pointer
/// slot (see [`VitaState::tls_addr`]).
#[hostcall]
fn get_tls_addr(st: &mut VitaState, key: u32) -> u32 {
    st.tls_addr(key)
}

/// int sceClibPrintf(const char *fmt, ...)
/// Formats to the debug console. Returns the number of bytes produced.
fn clib_printf(ctx: &mut GuestCtx, st: &mut VitaState) {
    let fmt_addr = ctx.arg(0);
    let mut out = Vec::new();
    // The format string is word 0; the variadic tail begins at word 1.
    cfmt::format_into(&mut out, ctx, fmt_addr, 1);
    let n = out.len() as u32;
    st.write_stdout(&out);
    ctx.ret(n);
}

/// int sceClibSnprintf(char *dst, SceSize dst_max, const char *fmt, ...)
/// Writes at most dst_max-1 bytes plus a NUL. Returns the length that would have
/// been written had the buffer been unbounded (C99 semantics).
fn clib_snprintf(ctx: &mut GuestCtx, _st: &mut VitaState) {
    let dst = ctx.arg(0);
    let dst_max = ctx.arg(1);
    let fmt_addr = ctx.arg(2);
    let mut out = Vec::new();
    // dst, dst_max, fmt are words 0..2; the variadic tail begins at word 3.
    cfmt::format_into(&mut out, ctx, fmt_addr, 3);
    let full_len = out.len();
    if dst_max > 0 {
        let n = (dst_max as usize - 1).min(full_len);
        let mut written = out[..n].to_vec();
        written.push(0); // NUL terminator
        ctx.write_bytes(dst, &written);
    }
    ctx.ret(full_len as u32);
}

/// void *sceClibMemcpy(void *dst, const void *src, SceSize len)
#[hostcall]
fn clib_memcpy(ctx: &mut GuestCtx, dst: Ptr, src: Ptr, len: u32) -> Ptr {
    let bytes = ctx.read_bytes(src.addr(), len as usize);
    ctx.write_bytes(dst.addr(), &bytes);
    dst
}

/// void *sceClibMemset(void *dst, int ch, SceSize len)
#[hostcall]
fn clib_memset(ctx: &mut GuestCtx, dst: Ptr, ch: i32, len: u32) -> Ptr {
    let fill = vec![ch as u8; len as usize];
    ctx.write_bytes(dst.addr(), &fill);
    dst
}

/// int sceClibMemcmp(const void *a, const void *b, SceSize len)
/// Expression-bodied (no early `return`): `#[hostcall]` inlines the body into a
/// `()` wrapper, so a `return` would escape the wrapper, not this handler.
#[hostcall]
fn clib_memcmp(ctx: &mut GuestCtx, a: Ptr, b: Ptr, len: u32) -> i32 {
    let x = ctx.read_bytes(a.addr(), len as usize);
    let y = ctx.read_bytes(b.addr(), len as usize);
    x.iter()
        .zip(y.iter())
        .find(|(p, q)| p != q)
        .map(|(p, q)| *p as i32 - *q as i32)
        .unwrap_or(0)
}

/// SceSize sceClibStrnlen(const char *s, SceSize maxlen)
#[hostcall]
fn clib_strnlen(ctx: &mut GuestCtx, s: Ptr, maxlen: u32) -> u32 {
    let bytes = ctx.read_bytes(s.addr(), maxlen as usize);
    bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len()) as u32
}

/// char *sceClibStrncpy(char *dst, const char *src, SceSize len)
/// Copies up to len bytes, zero-filling the remainder if src is shorter.
#[hostcall]
fn clib_strncpy(ctx: &mut GuestCtx, dst: Ptr, src: Ptr, len: u32) -> Ptr {
    let src_bytes = ctx.read_bytes(src.addr(), len as usize);
    let nul = src_bytes.iter().position(|&b| b == 0).unwrap_or(src_bytes.len());
    let mut out = vec![0u8; len as usize];
    out[..nul].copy_from_slice(&src_bytes[..nul]);
    ctx.write_bytes(dst.addr(), &out);
    dst
}

/// int sceClibStrncmp(const char *a, const char *b, SceSize len)
#[hostcall]
fn clib_strncmp(ctx: &mut GuestCtx, a: Ptr, b: Ptr, len: u32) -> i32 {
    let x = ctx.read_bytes(a.addr(), len as usize);
    let y = ctx.read_bytes(b.addr(), len as usize);
    let mut result = 0i32;
    for i in 0..len as usize {
        let p = x.get(i).copied().unwrap_or(0);
        let q = y.get(i).copied().unwrap_or(0);
        if p != q {
            result = p as i32 - q as i32;
            break;
        }
        if p == 0 {
            break;
        }
    }
    result
}

/// int sceClibStrcmp(const char *a, const char *b)
#[hostcall]
fn clib_strcmp(ctx: &mut GuestCtx, a: Ptr, b: Ptr) -> i32 {
    let x = read_cstr_bytes(ctx, a.addr());
    let y = read_cstr_bytes(ctx, b.addr());
    let mut result = 0i32;
    for i in 0..=x.len().max(y.len()) {
        let p = x.get(i).copied().unwrap_or(0);
        let q = y.get(i).copied().unwrap_or(0);
        if p != q {
            result = p as i32 - q as i32;
            break;
        }
        if p == 0 {
            break;
        }
    }
    result
}

/// Read a bounded NUL-terminated byte string from guest memory.
fn read_cstr_bytes(ctx: &GuestCtx, addr: u32) -> Vec<u8> {
    let bytes = ctx.read_bytes(addr, MAX_STR);
    let n = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    bytes[..n].to_vec()
}
