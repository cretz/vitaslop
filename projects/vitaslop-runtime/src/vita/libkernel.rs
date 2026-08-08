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
use crate::Ptr;
use crate::vita::cfmt;
use crate::SvcOutcome;

/// Bound on an unbounded string read (strcmp), so a missing NUL cannot make us
/// scan the whole guest address space.
const MAX_STR: usize = 4096;

/// The SceUID reported for the implicit main thread (bring-up: one thread of
/// control, plus synchronously-run workers).
const MAIN_THREAD_ID: i32 = 0x40;

/// Diagnostic (`RUST_LOG=vitaslop::exit=debug`): when the guest calls
/// `sceKernelExitProcess`, dump the exit code, the immediate caller (LR), and a
/// window of the stack top. Return addresses saved on the stack (even words that
/// point into the code image) reveal the call chain that decided to quit, so the
/// deciding function can be disassembled. Zero cost when the log target is off.
pub(super) fn trace_exit(ctx: &mut GuestCtx, st: &VitaState) {
    if !tracing::enabled!(target: "vitaslop::exit", tracing::Level::DEBUG) {
        return;
    }
    macro_rules! exit_log {
        ($($arg:tt)*) => { tracing::debug!(target: "vitaslop::exit", $($arg)*) };
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
    exit_log!("last {} calls (idx thid name):", trace.len() - start);
    for i in start..trace.len() {
        let thid = thids.get(i).copied().unwrap_or(0);
        exit_log!("  {i} t{thid:#x} {}", crate::nid::name(trace[i]));
    }
    let r0 = ctx.regs[0];
    let lr = ctx.regs[14];
    let sp = ctx.regs[13];
    exit_log!("code={r0:#x} (r0={} signed) lr={lr:#010x} sp={sp:#010x}", r0 as i32);
    exit_log!("r0..r12: {:08x?}", &ctx.regs[0..13]);
    // Print stack words; flag any that fall inside the loaded code image (a plausible
    // return address, Thumb or ARM) so the manual backtrace is quick. The image spans
    // [base, base + ~5 MiB); use a generous 8 MiB window to stay title-agnostic.
    let base = ctx.base;
    let code_end = base.wrapping_add(0x0080_0000);
    for i in 0..48u32 {
        let a = sp.wrapping_add(i * 4);
        let v = ctx.read_u32(a);
        let tag = if v >= base && v < code_end { "  <- code?" } else { "" };
        exit_log!("  sp+{:<3} {a:#010x}: {v:#010x}{tag}", i * 4);
    }
}

/// SceUID sceKernelCreateThread(const char *name, SceKernelThreadEntry entry,
///     int initPriority, SceSize stackSize, SceUInt attr, int cpuAffinityMask,
///     const SceKernelThreadOptParam *option)
/// The last three args sit past r3 on the stack; `#[hostcall]` reads them there.
#[hostcall]
pub(super) fn create_thread(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    name: Ptr,
    entry: Ptr,
    prio: i32,
    stack_size: u32,
    attr: u32,
    cpu: i32,
    _opt: Ptr,
) -> i32 {
    let thid = st.create_thread(entry.addr(), stack_size, prio, attr, cpu);
    // Thread names are the fastest way to identify a worker's purpose when diagnosing
    // a boot stall (RUST_LOG=vitaslop::thread=debug): a pure-poll thread's name
    // ("Online", "Sync", ...) names the subsystem the title is waiting on.
    if !name.is_null() {
        let raw = ctx.read_bytes(name.addr(), 32);
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        let nm = String::from_utf8_lossy(&raw[..end]);
        tracing::debug!(target: "vitaslop::thread", thid, entry = format_args!("{:#010x}", entry.addr()), prio, name = %nm, flip = st.flip_count(), "createThread");
        st.set_thread_name(thid, &nm);
    }
    thid
}

/// int sceKernelStartThread(SceUID thid, SceSize arglen, void *argp)
/// Under the preemptive scheduler the worker runs later, not synchronously, so the
/// argument block must be *snapshotted now*: the kernel copies `arglen` bytes from
/// `argp` to the new thread before it runs, and callers rely on that - `argp` is
/// almost always a stack temporary in the caller's frame that is reused (overwritten)
/// long before the worker reads it. Copy it into a stable heap buffer and hand the
/// worker that copy; without this the worker reads garbage for its argument.
pub(super) fn start_thread(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let thid = ctx.arg(0) as i32;
    let arglen = ctx.arg(1);
    let argp = ctx.arg(2);
    let arg_ptr = if arglen > 0 && argp != 0 {
        let bytes = ctx.read_bytes(argp, arglen as usize);
        let buf = st.galloc(arglen, 8);
        ctx.write_bytes(buf, &bytes);
        buf
    } else {
        argp
    };
    let preempt = st.start_thread(thid, arglen, arg_ptr);
    ctx.ret(0);
    // The real kernel switches to the just-started thread immediately when it
    // outranks us, running it until it blocks before we continue. Reschedule so the
    // scheduler picks the highest-priority runnable thread (the new one).
    if preempt {
        SvcOutcome::Reschedule
    } else {
        SvcOutcome::Continue
    }
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
pub(super) fn wait_thread_end(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
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
pub(super) fn get_thread_id(st: &mut VitaState) -> i32 {
    if st.is_preemptive() {
        // A fiber reports the thread that ran it, because on hardware it executes on
        // that thread - only its stack differs. A job system that keys per-worker state
        // off the thread id would otherwise see a new worker at every switch.
        st.logical_thread(st.current_thread())
    } else {
        MAIN_THREAD_ID
    }
}

/// int sceKernelGetThreadExitStatus(SceUID thid, int *pExitStatus)
/// Writes the thread's exit code and succeeds; a never-finished or unknown thread
/// reports 0. (Single-thread: workers already ran to completion by the time this is
/// asked.)
#[hostcall]
pub(super) fn get_thread_exit_status(ctx: &mut GuestCtx, st: &mut VitaState, thid: i32, out: Ptr) -> i32 {
    let code = st.thread_exit_code(thid).unwrap_or(0);
    if !out.is_null() {
        ctx.write_u32(out.addr(), code);
    }
    0
}

/// The inline form of a SceLibKernel host import, or `None` for the overwhelming
/// majority that have real behaviour. Reached through [`crate::vita::inline_op`], which
/// owns the global on/off switch.
///
/// Only the process-clock family qualifies, and it qualifies for exactly the reason the
/// [`mirror`](crate::vita::mirror) module doc gives: the value is a pure function of the
/// virtual clock, and the virtual clock advances only in the scheduler, with no guest
/// thread live. So a word refreshed before every resume is not an approximation of the
/// call - it is the answer the call would have given, for as long as the guest can
/// observe it.
///
/// The three spellings differ only in where the answer goes, which is a calling
/// convention and not behaviour:
///   - `sceKernelGetProcessTime(SceKernelSysClock *)` writes it through a pointer,
///   - `...Wide()` returns it in the r0/r1 pair,
///   - `...Low()` returns its low half in r0.
///
/// **Every entry here duplicates a handler below, and the two must agree.** The
/// `mirror_matches_its_handlers` test holds them to that.
pub(crate) fn inline_op(func_nid: u32) -> Option<vitaslop_transpiler::InlineOp> {
    use crate::nid::libkernel as lk;
    use crate::vita::mirror::SLOT_CLOCK_LO;
    use vitaslop_transpiler::InlineOp::{LoadMirror, LoadMirrorPair, StoreMirrorPair};
    Some(match func_nid {
        lk::GET_PROCESS_TIME => StoreMirrorPair { slot: SLOT_CLOCK_LO },
        lk::GET_PROCESS_TIME_WIDE => LoadMirrorPair { slot: SLOT_CLOCK_LO },
        lk::GET_PROCESS_TIME_LOW => LoadMirror { slot: SLOT_CLOCK_LO },
        _ => return None,
    })
}

/// SceUInt64 sceKernelGetProcessTimeWide(void)
/// The 64-bit process-runtime clock in microseconds. Returned in r0 (low)/r1 (high),
/// so it is hand-written rather than `#[hostcall]`. Uses the virtual monotonic clock.
pub(super) fn get_process_time(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    // sceKernelGetProcessTime(SceKernelSysClock *pClock): write the 64-bit virtual
    // monotonic process time (microseconds) to *pClock and return 0. Same clock the
    // wide form returns in registers.
    let t = st.now_us();
    let ptr = ctx.regs[0];
    ctx.write_u32(ptr, t as u32);
    ctx.write_u32(ptr.wrapping_add(4), (t >> 32) as u32);
    ctx.regs[0] = 0;
    SvcOutcome::Continue
}

pub(super) fn get_process_time_wide(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    // The virtual monotonic clock the scheduler advances (jumping over idle waits),
    // so a timed wait loop reads real elapsed time instead of a frozen value.
    let t = st.now_us();
    ctx.regs[0] = t as u32;
    ctx.regs[1] = (t >> 32) as u32;
    SvcOutcome::Continue
}

/// Byte size of a guest `SceKernelThreadInfo` and the offsets past its 32-byte
/// `name` array. Layout from vitasdk `psp2common/kernel/threadmgr.h`, which asserts
/// the total at 0x80: size, processId, name[32], attr, status, entry, stack,
/// stackSize, initPriority, currentPriority, initCpuAffinityMask,
/// currentCpuAffinityMask, currentCpuId, lastExecutedCpuId, waitType, waitId,
/// exitStatus, runClocks (64-bit), intrPreemptCount, threadPreemptCount,
/// threadReleaseCount, changeCpuCount, ...
const THREAD_INFO_SIZE: u32 = 0x80;
const THREAD_INFO_NAME: u32 = 8;
const THREAD_INFO_ATTR: u32 = 40;

/// The one CPU this scheduler runs guest threads on. Reported for both `currentCpuId`
/// and `lastExecutedCpuId`: there is a single baton, so a thread has only ever run
/// here, and saying so is more truthful than leaving the fields at a value that names
/// a core the thread never touched.
const ONLY_CPU_ID: u32 = 0;

/// int sceKernelGetThreadInfo(SceUID thid, SceKernelThreadInfo *info)
///
/// Fill the guest's thread-info struct. `thid` 0 is the calling thread. Everything
/// this kernel actually models is real (name, attributes, run status, entry, stack,
/// both priorities, affinity, exit status); the counters it does not model
/// (preemption/release tallies, wait type/id, accumulated run clocks) are written as
/// zero rather than left holding whatever the guest buffer had, so a title reading
/// them sees a defined value.
#[hostcall]
pub(super) fn get_thread_info(ctx: &mut GuestCtx, st: &mut VitaState, thid: i32, info: Ptr) -> i32 {
    match st.thread_info(thid) {
        Some(t) if info.addr() != 0 => {
            let addr = info.addr();
            let mut name = [0u8; 32];
            let bytes = t.name.as_bytes();
            let n = bytes.len().min(31); // keep the terminating NUL
            name[..n].copy_from_slice(&bytes[..n]);

            ctx.write_u32(addr, THREAD_INFO_SIZE);
            ctx.write_u32(addr + 4, PROCESS_ID as u32);
            ctx.write_bytes(addr + THREAD_INFO_NAME, &name);
            let f = |i: u32| addr + THREAD_INFO_ATTR + i * 4;
            ctx.write_u32(f(0), t.attr);
            ctx.write_u32(f(1), t.status);
            ctx.write_u32(f(2), t.entry);
            ctx.write_u32(f(3), t.stack_base);
            ctx.write_u32(f(4), t.stack_size as u32);
            ctx.write_u32(f(5), t.init_priority as u32);
            ctx.write_u32(f(6), t.current_priority as u32);
            ctx.write_u32(f(7), t.cpu_affinity as u32);
            ctx.write_u32(f(8), t.cpu_affinity as u32);
            ctx.write_u32(f(9), ONLY_CPU_ID); // currentCpuId
            ctx.write_u32(f(10), ONLY_CPU_ID); // lastExecutedCpuId
            ctx.write_u32(f(11), 0); // waitType
            ctx.write_u32(f(12), 0); // waitId
            ctx.write_u32(f(13), t.exit_status as u32);
            // runClocks (64-bit) then the four preempt/release/changeCpu counters and
            // the trailing notify-callback fields: not modelled, defined as zero.
            for i in 14..(THREAD_INFO_SIZE - THREAD_INFO_ATTR) / 4 {
                ctx.write_u32(f(i), 0);
            }
            0
        }
        _ => SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID as i32,
    }
}

/// The single process this kernel runs, matching `sceKernelGetProcessId`.
const PROCESS_ID: i32 = 0x1000;

/// `SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID`.
const SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID: u32 = 0x8002_8021;

/// int sceKernelSendSignal(SceUID threadId)
///
/// Deliver one signal to a thread. Counted rather than latched, so a producer that
/// signals twice before the consumer runs does not lose one - a lost signal here is
/// exactly the hazard the primitive exists to avoid.
#[hostcall]
pub(super) fn send_signal(st: &mut VitaState, thid: i32) -> i32 {
    match st.send_signal(thid) {
        Ok(()) => 0,
        Err(e) => e as i32,
    }
}

/// int sceKernelWaitSignal(SceUInt32 unk0, SceUInt32 delay, SceUInt32 *timeout)
///
/// Consume one signal sent to the calling thread, parking until one arrives. The
/// first argument is unused by the kernel's own callers and the delay/timeout pair
/// arms a deadline exactly as the other timed waits do.
///
/// Under the single-thread bring-up there is no sibling to signal us, so an
/// unsatisfied wait cannot be made to succeed by waiting; it returns at once, which
/// is the same choice the other primitives make there.
pub(super) fn wait_signal(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    ctx.ret(0);
    let thid = st.current_thread();
    if st.take_signal(thid) || !st.is_preemptive() {
        SvcOutcome::Continue
    } else {
        st.signal_block();
        SvcOutcome::Block
    }
}

/// SceUInt32 sceKernelGetProcessTimeLow(void)
///
/// The low 32 bits of the same microsecond process clock the wide form returns. It is
/// a separate export rather than a convenience wrapper because callers use it for
/// cheap interval timing, and it must come off the SAME clock as the wide one or two
/// timings of the same interval disagree.
#[hostcall]
pub(super) fn get_process_time_low(st: &mut VitaState) -> u32 {
    st.now_us() as u32
}

/// The 16-byte OpenPSID this console reports. It is a per-console identifier, so
/// there is no "correct" value to reproduce - what matters is that it is STABLE
/// (a title keying save data or a local profile on it must see the same bytes on
/// every boot) and constant across runs, which determinism requires anyway. Titles
/// also reject an all-zero id as "not provisioned", so it is not zero.
const OPEN_PS_ID: [u8; 16] = [
    0x76, 0x69, 0x74, 0x61, 0x73, 0x6C, 0x6F, 0x70, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
];

/// int sceKernelGetOpenPsId(SceKernelOpenPsId *id)
#[hostcall]
pub(super) fn get_open_ps_id(ctx: &mut GuestCtx, out: Ptr) -> i32 {
    if out.is_null() {
        SCE_KERNEL_ERROR_INVALID_ARGUMENT as i32
    } else {
        ctx.write_bytes(out.addr(), &OPEN_PS_ID);
        0
    }
}

/// `SCE_KERNEL_ERROR_INVALID_ARGUMENT`.
const SCE_KERNEL_ERROR_INVALID_ARGUMENT: u32 = 0x8002_0005;
/// `SCE_KERNEL_ERROR_MODULEMGR_NOENT`: no loaded module covers the address.
const SCE_KERNEL_ERROR_MODULEMGR_NOENT: u32 = 0x8002_D082;

/// Guest `SceKernelModuleInfo` layout (vitasdk `psp2common/kernel/modulemgr.h`,
/// asserted there at 0x1B8): size, modid, modattr(u16), modver[2], module_name[28],
/// unk28, start/stop/exit entries, exidx/extab bounds, tlsInit + two sizes,
/// path[256], segments[4] (0x18 each), state.
const MODULE_INFO_SIZE: u32 = 0x1B8;
const MODULE_INFO_NAME: u32 = 12;
const MODULE_INFO_START_ENTRY: u32 = 44;
const MODULE_INFO_PATH: u32 = 76;
const MODULE_INFO_SEGMENTS: u32 = MODULE_INFO_PATH + 256;
const SEGMENT_INFO_SIZE: u32 = 0x18;
/// `SceKernelModuleInfo::segments` is a fixed array of four.
const MODULE_INFO_MAX_SEGMENTS: usize = 4;
/// `SCE_KERNEL_MODULE_STATE_STARTED`: every module in the image was started before
/// the guest's main entry ran, which is the only state one can be observed in here.
const MODULE_STATE_STARTED: u32 = 1 | 4 | 8;

/// int sceKernelGetModuleInfoByAddr(const void *addr, SceKernelModuleInfo *info)
///
/// Identify the module a code or data address belongs to and describe it. This is how
/// a title turns a return address into "which module faulted" for its own crash
/// reporting, so the answer has to be the real module and the real segment bounds -
/// the linker records where each one was actually placed (see
/// [`crate::link::LoadedModule`]).
///
/// An address in no module (the heap, a thread stack, a memory block) is not an error
/// case to paper over: the kernel reports NOENT and so do we.
#[hostcall]
pub(super) fn get_module_info_by_addr(ctx: &mut GuestCtx, st: &mut VitaState, addr: Ptr, info: Ptr) -> i32 {
    match st.module_by_addr(addr.addr()) {
        Some((uid, m)) if info.addr() != 0 => {
            let base = info.addr();
            // The name field is 28 bytes and the path 256; both are NUL-terminated, and
            // the whole struct is written (not just the fields we know) so the guest
            // never reads its own stale buffer back as module state.
            let mut buf = vec![0u8; MODULE_INFO_SIZE as usize];
            let put = |buf: &mut [u8], off: u32, v: u32| {
                buf[off as usize..off as usize + 4].copy_from_slice(&v.to_le_bytes());
            };
            let put_str = |buf: &mut [u8], off: u32, cap: usize, s: &str| {
                let b = s.as_bytes();
                let n = b.len().min(cap - 1); // keep the terminating NUL
                buf[off as usize..off as usize + n].copy_from_slice(&b[..n]);
            };
            put(&mut buf, 0, MODULE_INFO_SIZE);
            put(&mut buf, 4, uid as u32);
            put_str(&mut buf, MODULE_INFO_NAME, 28, &m.name);
            put(&mut buf, MODULE_INFO_START_ENTRY, m.entry);
            // The module was loaded from the title's own directory; report the name as
            // its path, which is what a module loaded by the process shows.
            put_str(&mut buf, MODULE_INFO_PATH, 256, &m.name);
            for (i, &(vaddr, mem_size, file_size, exec, write)) in
                m.segments.iter().take(MODULE_INFO_MAX_SEGMENTS).enumerate()
            {
                let seg = MODULE_INFO_SEGMENTS + i as u32 * SEGMENT_INFO_SIZE;
                // `perms` is rwx in the low bits, in the usual ELF order.
                let perms = 4 | u32::from(write) << 1 | u32::from(exec);
                put(&mut buf, seg, SEGMENT_INFO_SIZE);
                put(&mut buf, seg + 4, perms);
                put(&mut buf, seg + 8, vaddr);
                put(&mut buf, seg + 12, mem_size);
                put(&mut buf, seg + 16, file_size);
            }
            put(&mut buf, MODULE_INFO_SIZE - 4, MODULE_STATE_STARTED);
            ctx.write_bytes(base, &buf);
            0
        }
        Some(_) => SCE_KERNEL_ERROR_INVALID_ARGUMENT as i32,
        None => SCE_KERNEL_ERROR_MODULEMGR_NOENT as i32,
    }
}

/// int sceKernelCallModuleExit(...)
///
/// Run the process's registered module-exit handlers. On hardware this is part of
/// shutting the process down: the kernel walks the loaded modules calling each
/// `module_stop`, and the process is torn down afterwards. Ending the run is exactly
/// what this engine's process teardown does, so it is reported as a clean halt rather
/// than allowed to return to a guest that expects never to be resumed.
pub(super) fn call_module_exit(ctx: &mut GuestCtx, _st: &mut VitaState) -> SvcOutcome {
    ctx.ret(0);
    tracing::info!(target: "vitaslop::exit", "sceKernelCallModuleExit - process teardown");
    SvcOutcome::Halt
}

/// void __sce_aeabi_idiv0(int return_value) / long long __sce_aeabi_ldiv0(long long)
///
/// The ARM EABI division-by-zero hook: the compiler's `__aeabi_idiv`/`__aeabi_ldiv`
/// call it when the divisor is zero, passing the value the division will yield, and
/// the default implementation RETURNS THAT VALUE UNCHANGED (a hook exists so a program
/// can raise SIGFPE instead; the SCE runtime's does not). The value is already in
/// r0 (and r0:r1 for the long form) on entry, so returning is the whole behaviour -
/// and it must not be clobbered, which is why this writes no return value at all.
pub(super) fn aeabi_div0(_ctx: &mut GuestCtx, _st: &mut VitaState) -> SvcOutcome {
    SvcOutcome::Continue
}

/// void *sceKernelGetTLSAddr(int key)
/// Returns this thread's storage slot for `key`, a stable zero-initialized pointer
/// slot (see [`VitaState::tls_addr`]).
#[hostcall]
pub(super) fn get_tls_addr(st: &mut VitaState, key: u32) -> u32 {
    st.tls_addr(key)
}

/// int sceClibPrintf(const char *fmt, ...)
/// Formats to the debug console. Returns the number of bytes produced.
pub(super) fn clib_printf(ctx: &mut GuestCtx, st: &mut VitaState) {
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
pub(super) fn clib_snprintf(ctx: &mut GuestCtx, _st: &mut VitaState) {
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
pub(super) fn clib_memcpy(ctx: &mut GuestCtx, dst: Ptr, src: Ptr, len: u32) -> Ptr {
    let bytes = ctx.read_bytes(src.addr(), len as usize);
    ctx.write_bytes(dst.addr(), &bytes);
    dst
}

/// void *sceClibMemset(void *dst, int ch, SceSize len)
#[hostcall]
pub(super) fn clib_memset(ctx: &mut GuestCtx, dst: Ptr, ch: i32, len: u32) -> Ptr {
    let fill = vec![ch as u8; len as usize];
    ctx.write_bytes(dst.addr(), &fill);
    dst
}

/// int sceClibMemcmp(const void *a, const void *b, SceSize len)
/// Expression-bodied (no early `return`): `#[hostcall]` inlines the body into a
/// `()` wrapper, so a `return` would escape the wrapper, not this handler.
#[hostcall]
pub(super) fn clib_memcmp(ctx: &mut GuestCtx, a: Ptr, b: Ptr, len: u32) -> i32 {
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
pub(super) fn clib_strnlen(ctx: &mut GuestCtx, s: Ptr, maxlen: u32) -> u32 {
    let bytes = ctx.read_bytes(s.addr(), maxlen as usize);
    bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len()) as u32
}

/// char *sceClibStrncpy(char *dst, const char *src, SceSize len)
/// Copies up to len bytes, zero-filling the remainder if src is shorter.
#[hostcall]
pub(super) fn clib_strncpy(ctx: &mut GuestCtx, dst: Ptr, src: Ptr, len: u32) -> Ptr {
    let src_bytes = ctx.read_bytes(src.addr(), len as usize);
    let nul = src_bytes.iter().position(|&b| b == 0).unwrap_or(src_bytes.len());
    let mut out = vec![0u8; len as usize];
    out[..nul].copy_from_slice(&src_bytes[..nul]);
    ctx.write_bytes(dst.addr(), &out);
    dst
}

/// int sceClibStrncmp(const char *a, const char *b, SceSize len)
#[hostcall]
pub(super) fn clib_strncmp(ctx: &mut GuestCtx, a: Ptr, b: Ptr, len: u32) -> i32 {
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
pub(super) fn clib_strcmp(ctx: &mut GuestCtx, a: Ptr, b: Ptr) -> i32 {
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

/// char *sceClibStrrchr(const char *s, int c)
/// The LAST occurrence of `c` in `s`, or NULL. Expression-bodied (no early `return`):
/// `#[hostcall]` inlines the body into a `()` wrapper.
#[hostcall]
pub(super) fn clib_strrchr(ctx: &mut GuestCtx, s: Ptr, c: i32) -> Ptr {
    let bytes = read_cstr_bytes(ctx, s.addr());
    let needle = c as u8;
    match needle {
        // Searching for the terminator finds it, as C specifies.
        0 => Ptr(s.addr() + bytes.len() as u32),
        _ => match bytes.iter().rposition(|&b| b == needle) {
            Some(i) => Ptr(s.addr() + i as u32),
            None => Ptr(0),
        },
    }
}

/// int sceClibStrncasecmp(const char *a, const char *b, SceSize n)
/// Compare at most `n` bytes, case-insensitively for ASCII. Like the other clib compares
/// the result is the difference of the first differing (folded) bytes.
#[hostcall]
pub(super) fn clib_strncasecmp(ctx: &mut GuestCtx, a: Ptr, b: Ptr, n: u32) -> i32 {
    let x = read_cstr_bytes(ctx, a.addr());
    let y = read_cstr_bytes(ctx, b.addr());
    let mut result = 0;
    for i in 0..n as usize {
        let p = x.get(i).copied().unwrap_or(0).to_ascii_lowercase();
        let q = y.get(i).copied().unwrap_or(0).to_ascii_lowercase();
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

// ---------------------------------------------------------------------------
// sceClibMspace*: a general allocator over a block of the title's OWN memory.
//
// A title that manages its own heap hands the system a block and asks for an allocator
// over it. The allocator is real (see [`crate::mspace`]) and lives entirely inside the
// block the guest provided, so nothing here can collide with anything else in the guest
// address space - the failure mode `vitaslop-galloc-vs-main-stack` records.
// ---------------------------------------------------------------------------

/// void *sceClibMspaceCreate(void *base, SceSize capacity, SceUInt32 flag)
/// Create a memory space over the caller's block and return its handle. The handle is
/// the block's own base address, exactly as hardware's returns it, so a title that
/// stores and passes the handle around always resolves to the same space.
///
/// A degenerate request (null base, or a capacity too small to allocate anything from)
/// returns NULL, which is what a caller checks for.
#[hostcall]
pub(super) fn clib_mspace_create(_ctx: &mut GuestCtx, st: &mut VitaState, base: Ptr, capacity: u32, _flag: u32) -> Ptr {
    match st.mspaces.create(base.addr(), capacity) {
        Some(handle) => {
            tracing::debug!(
                target: "vitaslop::cb",
                base = format_args!("{:#x}", base.addr()),
                capacity,
                "sceClibMspaceCreate"
            );
            Ptr(handle)
        }
        None => {
            tracing::warn!(
                target: "vitaslop::err",
                base = format_args!("{:#x}", base.addr()),
                capacity,
                "sceClibMspaceCreate: degenerate space -> NULL"
            );
            Ptr(0)
        }
    }
}

/// void sceClibMspaceDestroy(void *msp)
/// Drop a space. The guest owns the memory it was built over and reclaims it itself, so
/// there is nothing to release here beyond the bookkeeping.
#[hostcall]
pub(super) fn clib_mspace_destroy(_ctx: &mut GuestCtx, st: &mut VitaState, msp: Ptr) {
    if !st.mspaces.destroy(msp.addr()) {
        tracing::warn!(
            target: "vitaslop::err",
            msp = format_args!("{:#x}", msp.addr()),
            "sceClibMspaceDestroy: no such memory space"
        );
    }
}

/// void *sceClibMspaceMalloc(void *msp, SceSize size)
/// Allocate from a space. Returns NULL when the space is exhausted - the honest result,
/// and one the caller is written to handle. An unknown space is reported: it means a
/// handle the guest holds is not one we created, which is a bug worth seeing rather than
/// an allocation to invent.
#[hostcall]
pub(super) fn clib_mspace_malloc(_ctx: &mut GuestCtx, st: &mut VitaState, msp: Ptr, size: u32) -> Ptr {
    Ptr(mspace_alloc(st, msp, size, 0, "sceClibMspaceMalloc"))
}

/// void *sceClibMspaceMemalign(void *msp, SceSize alignment, SceSize size)
#[hostcall]
pub(super) fn clib_mspace_memalign(_ctx: &mut GuestCtx, st: &mut VitaState, msp: Ptr, alignment: u32, size: u32) -> Ptr {
    Ptr(mspace_alloc(st, msp, size, alignment, "sceClibMspaceMemalign"))
}

fn mspace_alloc(st: &mut VitaState, msp: Ptr, size: u32, align: u32, what: &str) -> u32 {
    let Some(space) = st.mspaces.get_mut(msp.addr()) else {
        tracing::warn!(
            target: "vitaslop::err",
            msp = format_args!("{:#x}", msp.addr()),
            "{what}: no such memory space -> NULL"
        );
        return 0;
    };
    match space.alloc(size, align) {
        Some(p) => p,
        None => {
            let (used, capacity) = (space.used_bytes(), space.capacity());
            tracing::warn!(
                target: "vitaslop::err",
                msp = format_args!("{:#x}", msp.addr()),
                size,
                align,
                used,
                capacity,
                "{what}: memory space exhausted -> NULL"
            );
            0
        }
    }
}

/// void sceClibMspaceFree(void *msp, void *ptr)
/// Free a pointer back to its space. Freeing NULL is a no-op, as it is in C; freeing
/// anything the space did not hand out is reported rather than absorbed, because a double
/// free or a foreign pointer is a real defect that silence would hide until the heap
/// misbehaves somewhere unrelated.
#[hostcall]
pub(super) fn clib_mspace_free(_ctx: &mut GuestCtx, st: &mut VitaState, msp: Ptr, ptr: Ptr) {
    if ptr.is_null() {
        return;
    }
    let freed = st.mspaces.get_mut(msp.addr()).is_some_and(|s| s.free(ptr.addr()));
    if !freed {
        tracing::warn!(
            target: "vitaslop::err",
            msp = format_args!("{:#x}", msp.addr()),
            ptr = format_args!("{:#x}", ptr.addr()),
            "sceClibMspaceFree: pointer is not a live allocation of this space"
        );
    }
}

/// Read a bounded NUL-terminated byte string from guest memory.
pub(super) fn read_cstr_bytes(ctx: &GuestCtx, addr: u32) -> Vec<u8> {
    let bytes = ctx.read_bytes(addr, MAX_STR);
    let n = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    bytes[..n].to_vec()
}
