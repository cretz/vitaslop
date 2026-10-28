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

use crate::Ptr;
use crate::SvcOutcome;
use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::vita::cfmt;

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
    exit_log!(
        "code={r0:#x} (r0={} signed) lr={lr:#010x} sp={sp:#010x}",
        r0 as i32
    );
    // >>> r4..r12 ARE NOT MARSHALLED INTO A HOST CALL, so they are not printed as if they
    // were. The browser reads only the registers AAPCS lets a host call reach (arguments,
    // return, sp, lr, pc - see `NARROW_REGS`), and the callee-saved half stays in the guest's
    // own globals where the call is required to leave it. Printing zeros for them here would
    // read as a guest that had zeroed its saved registers, which is a different fault
    // entirely. `r1..r3` are argument lanes and ARE current.
    exit_log!(
        "r1..r3: {:08x?} (r4..r12 are not marshalled into a host call and are not shown)",
        &ctx.regs[1..4]
    );
    // Print stack words; flag any that fall inside the loaded code image (a plausible
    // return address, Thumb or ARM) so the manual backtrace is quick. The image spans
    // [base, base + ~5 MiB); use a generous 8 MiB window to stay title-agnostic.
    let base = ctx.base;
    let code_end = base.wrapping_add(0x0080_0000);
    for i in 0..48u32 {
        let a = sp.wrapping_add(i * 4);
        let v = ctx.read_u32(a);
        let tag = if v >= base && v < code_end {
            "  <- code?"
        } else {
            ""
        };
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
        tracing::debug!(target: "vitaslop::thread", thid, entry = format_args!("{:#010x}", entry.addr()), prio, cpu = format_args!("{cpu:#x}"), name = %nm, flip = st.flip_count(), "createThread");
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
    // Refused before the argument block is copied: a running thread is not started again
    // (see `VitaState::start_thread`), and the copy would be a leaked allocation per refusal.
    if st.thread_running(thid) {
        ctx.ret(crate::host::SCE_KERNEL_ERROR_NOT_DORMANT);
        return SvcOutcome::Continue;
    }
    let arg_ptr = if arglen > 0 && argp != 0 {
        let bytes = ctx.read_bytes(argp, arglen as usize);
        let buf = st.galloc(arglen, 8);
        ctx.write_bytes(buf, &bytes);
        buf
    } else {
        argp
    };
    let preempt = match st.start_thread(thid, arglen, arg_ptr) {
        Ok(p) => p,
        Err(e) => {
            ctx.ret(e);
            return SvcOutcome::Continue;
        }
    };
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

/// The id `sceKernelGetThreadId` reports, as ONE definition.
///
/// Two things compute this now - the handler below and the host-mirror snapshot that backs
/// its inlined form - and a second implementation of a value is exactly how the inline and
/// host answers drift apart with nothing in the system to notice. `mirror_matches_its_handlers`
/// holds them to this function.
pub(crate) fn thread_id(st: &VitaState) -> i32 {
    if st.is_preemptive() {
        // A fiber reports the thread that ran it, because on hardware it executes on
        // that thread - only its stack differs. A job system that keys per-worker state
        // off the thread id would otherwise see a new worker at every switch.
        st.logical_thread(st.current_thread())
    } else {
        MAIN_THREAD_ID
    }
}

/// SceUID sceKernelGetThreadId(void)
/// Reports the running thread's id: the scheduler's `current` under preemption, or
/// a fixed main-thread id in the single-thread-of-control bring-up.
#[hostcall]
pub(super) fn get_thread_id(st: &mut VitaState) -> i32 {
    thread_id(st)
}

/// int sceKernelGetThreadExitStatus(SceUID thid, int *pExitStatus)
/// Writes the thread's exit code and succeeds; a never-finished or unknown thread
/// reports 0. (Single-thread: workers already ran to completion by the time this is
/// asked.)
#[hostcall]
pub(super) fn get_thread_exit_status(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    thid: i32,
    out: Ptr,
) -> i32 {
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
/// Two families qualify, on two different arguments.
///
/// **The process-clock family** qualifies for exactly the reason the
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
/// **The three `sceClibMem*` bulk primitives** qualify on the opposite ground: they touch no
/// host value at all. Each is a pure function of guest memory, with no handle to resolve, no
/// kernel object behind it and no `VitaState` field anywhere in its handler - which is what
/// makes a call inlinable ([[vitaslop-guest-state-is-what-makes-a-call-inlinable]]). They
/// earn their place by COUNT: 508,181 crossings of a real title's 3.85 M, 13% of every host
/// call it makes, with 403,687 `sceClibMemcmp`s from a SINGLE call site.
///
/// The other `sceClib*` functions are deliberately NOT here. `strncpy` and `strnlen` scan for
/// a NUL and then behave differently on either side of it, and `snprintf` formats - none is a
/// bulk move, and each would be a new emitted form for a call count three orders of magnitude
/// smaller. Their handlers stay the definition.
///
/// **Every entry here duplicates a handler below, and the two must agree.** The
/// `mirror_matches_its_handlers` test holds the clock family to that; the bulk family is held
/// by `clib_memcmp` calling the shared [`vitaslop_transpiler::mem_compare`] the emitted loop
/// is asserted against, and by the execution tests in
/// `vitaslop-native/tests/inline_imports.rs`.
pub(crate) fn inline_op(func_nid: u32) -> Option<vitaslop_transpiler::InlineOp> {
    use crate::nid::libkernel as lk;
    use crate::vita::mirror::{SLOT_CLOCK_LO, SLOT_THREAD_ID};
    use vitaslop_transpiler::InlineOp::{
        LoadMirror, LoadMirrorPair, MemCompare, MemCopy, MemFill, StoreMirrorPair,
    };
    Some(match func_nid {
        lk::CLIB_MEMCPY => MemCopy,
        lk::CLIB_MEMSET => MemFill,
        lk::CLIB_MEMCMP => MemCompare,
        lk::GET_PROCESS_TIME => StoreMirrorPair {
            slot: SLOT_CLOCK_LO,
        },
        lk::GET_PROCESS_TIME_WIDE => LoadMirrorPair {
            slot: SLOT_CLOCK_LO,
        },
        lk::GET_PROCESS_TIME_LOW => LoadMirror {
            slot: SLOT_CLOCK_LO,
        },
        // `sceKernelGetThreadId` takes no argument, touches nothing, and returns a value the
        // guest cannot change: which thread the host handed the baton to. Same terms as the
        // clock, and it earns its place by COUNT - on a real title's front end it is the
        // most-called host function of the whole run at 1.64 million crossings, 1.51 million
        // of them from ONE call site, for a one-word read. See [`SLOT_THREAD_ID`] for why it
        // reads its own slot and not the scheduler's `current`.
        lk::GET_THREAD_ID => LoadMirror {
            slot: SLOT_THREAD_ID,
        },
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

/// SceInt32 sceKernelGetRemoteProcessTime(SceUID processId, SceKernelSysClock *pClock): the
/// process time of a process named by id. There is one process (`sceKernelGetProcessId`), so
/// it is [`get_process_time`] for that id and UNKNOWN_UID for any other.
pub(super) fn get_remote_process_time(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    const SCE_KERNEL_ERROR_UNKNOWN_UID: u32 = 0x8002_8001;
    let (pid, clock) = (ctx.regs[0] as i32, ctx.regs[1]);
    if pid != super::threadmgr::PROCESS_ID && pid != 0 {
        ctx.regs[0] = SCE_KERNEL_ERROR_UNKNOWN_UID;
    } else {
        let t = st.now_us();
        ctx.write_u32(clock, t as u32);
        ctx.write_u32(clock.wrapping_add(4), (t >> 32) as u32);
        ctx.regs[0] = 0;
    }
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
pub(super) fn get_module_info_by_addr(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    addr: Ptr,
    info: Ptr,
) -> i32 {
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
/// Returns this thread's storage slot for `key`, word `key` of a stable zero-initialized
/// per-thread array (see [`VitaState::tls_addr_of`]).
#[hostcall]
pub(super) fn get_tls_addr(st: &mut VitaState, key: u32) -> u32 {
    tls_slot_or_null(st.tls_addr(key), "sceKernelGetTLSAddr", 0, key)
}

/// void *sceKernelGetThreadTLSAddr(SceUID thid, int key)
#[hostcall]
pub(super) fn get_thread_tls_addr(st: &mut VitaState, thid: i32, key: u32) -> u32 {
    tls_slot_or_null(
        st.tls_addr_of(thid, key),
        "sceKernelGetThreadTLSAddr",
        thid,
        key,
    )
}

/// A key with no slot gets NULL, which is the kernel's answer to a bad key - said LOUDLY, once,
/// because a title that dereferences it faults somewhere that no longer names the call. Not a
/// FATAL: both calls stay on the fast `cont!` path (they are among the hottest a title makes).
fn tls_slot_or_null(slot: Option<u32>, name: &str, thid: i32, key: u32) -> u32 {
    slot.unwrap_or_else(|| {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::error!(
                target: "vitaslop::err",
                "{name}(thid={thid:#x}, key={key:#x}): NO SLOT - keys run 0..={:#x} (or guest memory ran                  out allocating the thread's slot array). Returned NULL; a fault that follows is this call",
                crate::host::TLS_KEY_MAX
            );
        }
        0
    })
}

/// int sceKernelGetRandomNumber(void *buf, SceSize size)
/// Fills `buf` from the world's random source - the same one every other random
/// read goes through, so a recorded run replays the same bytes.
#[hostcall]
pub(super) fn get_random_number(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    buf: Ptr,
    size: u32,
) -> i32 {
    if buf.addr() == 0 && size > 0 {
        -1
    } else {
        let mut bytes = vec![0u8; size.min(64 * 1024) as usize];
        st.world.fill_random(&mut bytes);
        ctx.write_bytes(buf.addr(), &bytes);
        0
    }
}

/// `SCE_KERNEL_MSG_PIPE_MODE_*` bits of a send/receive's mode word, as a fighting title's own call sites
/// use them (a FULL receive of one 4-byte message that dereferences what it got with no
/// error check; a send of mode `0x11` from a path that must not stall). The low bit asks for
/// the WHOLE size (FULL) rather than whatever is there (ASAP), and `0x10` is DONT_WAIT.
const MPP_MODE_FULL: u32 = 0x01;
const MPP_MODE_DONT_WAIT: u32 = 0x10;

/// SceUID sceKernelCreateMsgPipe(const char *name, int type, int attr, unsigned int bufSize, void *opt)
/// A byte FIFO between threads. Receives BLOCK (see [`msg_pipe_receive`]); sends always
/// complete at once, because the buffer is not bounded here - a sender that relies on a full
/// pipe parking it is not modelled, and would show up as a pipe that only ever grows.
pub(super) fn msg_pipe_create(ctx: &mut GuestCtx, st: &mut VitaState) {
    let uid = st.new_uid();
    st.msg_pipe_create(uid);
    ctx.ret(uid as u32);
}

/// int sceKernelDeleteMsgPipe(SceUID uid)
///
/// A receiver still parked on the pipe is woken with `SCE_KERNEL_ERROR_WAIT_DELETE`: the
/// object it waited on is gone and nothing else would ever release it, as for every other
/// deleted waitable here.
pub(super) fn msg_pipe_delete(ctx: &mut GuestCtx, st: &mut VitaState) {
    let uid = ctx.arg(0) as i32;
    ctx.ret(if st.msg_pipe_remove(uid) {
        st.msgpipe_release(uid, crate::host::SCE_KERNEL_ERROR_WAIT_DELETE);
        0
    } else {
        0x8002_0005
    });
}

/// `SCE_KERNEL_BACKTRACE_MODE_DONT_EXCEED` (henkaku wiki, SceKernelModulemgr): stop at the end
/// of the buffer and return SCE_OK instead of the stack depth.
const BACKTRACE_MODE_DONT_EXCEED: i32 = 0x2;
/// `SceKernelCallFrame`: `{ sp, pc }`, 8 bytes.
const CALL_FRAME_SIZE: u32 = 8;

/// SceInt32 sceKernelBacktraceSelf(SceKernelCallFrame *pCallFrameBuffer, SceSize numBytesBuffer,
///                                 SceUInt32 *pNumReturn, SceInt32 mode)
///
/// The henkaku wiki's prototype: walk the calling thread's stack into `{ sp, pc }` frames, put
/// how many were filled in `*pNumReturn`, and return the call-stack depth - or SCE_OK under
/// DONT_EXCEED. The one title seen calling it prints the frames' `pc` words into a crash report
/// on its fatal out-of-memory path, after checking the return is positive.
///
/// >>> ONE FRAME, BECAUSE ONE FRAME IS ALL THAT IS KNOWN. The console unwinds through each
/// module's ARM exception-index tables; this engine runs the guest as recompiled code whose
/// frames the host cannot walk, and parses no unwind tables. What it DOES know exactly is the
/// innermost frame: the caller's stack pointer and the address the call returns to (`sp` and
/// `lr` at the call). So that frame is reported, and the depth is reported as 1 - a short
/// stack, never a frame read off a guessed stack word. (`guest_return_trail` is that guess,
/// and it stays a diagnostic.)
pub(super) fn backtrace_self(ctx: &mut GuestCtx, _st: &mut VitaState) {
    let (buf, size, pnum, mode) = (ctx.arg(0), ctx.arg(1), ctx.arg(2), ctx.arg(3) as i32);
    if buf == 0 || pnum == 0 {
        ctx.ret(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        return;
    }
    let fits = size >= CALL_FRAME_SIZE;
    if fits {
        let (sp, lr) = (ctx.regs[13], ctx.regs[14]);
        ctx.write_u32(buf, sp);
        ctx.write_u32(buf + 4, lr);
    }
    ctx.write_u32(pnum, fits as u32);
    ctx.ret(if mode & BACKTRACE_MODE_DONT_EXCEED != 0 {
        0
    } else {
        1
    });
}

/// `SCE_KERNEL_ERROR_WAIT_CANCEL` (`psp2/kernel/error.h`): the wait was cancelled.
const SCE_KERNEL_ERROR_WAIT_CANCEL: u32 = 0x8002_8007;

/// int sceKernelCancelMsgPipe(SceUID uid, int *psend, int *precv)
///
/// Releases every thread waiting on the pipe with `SCE_KERNEL_ERROR_WAIT_CANCEL` and reports
/// how many were waiting to send and to receive (`psp2/kernel/threadmgr/msgpipe.h`; either
/// pointer may be NULL). Senders never park here - the buffer is unbounded, see
/// [`msg_pipe_create`] - so the send count is a true 0, and the receive count is the receivers
/// actually woken. The pipe's buffered bytes are discarded, which is what a cancel does to the
/// PSP kernel's message pipe this one descends from: the pipe is left as freshly created.
pub(super) fn msg_pipe_cancel(ctx: &mut GuestCtx, st: &mut VitaState) {
    let uid = ctx.arg(0) as i32;
    let (psend, precv) = (ctx.arg(1), ctx.arg(2));
    let code = match st.msg_pipe_mut(uid) {
        None => 0x8002_0005,
        Some(pipe) => {
            pipe.clear();
            let woken = st.msgpipe_release(uid, SCE_KERNEL_ERROR_WAIT_CANCEL);
            if psend != 0 {
                ctx.write_u32(psend, 0);
            }
            if precv != 0 {
                ctx.write_u32(precv, woken);
            }
            0
        }
    };
    ctx.ret(code);
}

/// int sceKernelSendMsgPipe(SceUID uid, const void *buf, unsigned int size, int mode, unsigned int *result, unsigned int *timeout)
///
/// Appends the bytes, then hands them to any receiver parked on the pipe.
pub(super) fn msg_pipe_send(ctx: &mut GuestCtx, st: &mut VitaState) {
    let uid = ctx.arg(0) as i32;
    let (buf, size, result) = (ctx.arg(1), ctx.arg(2), ctx.arg(4));
    let Some(pipe) = st.msg_pipe_mut(uid) else {
        ctx.ret(0x8002_0005);
        return;
    };
    let bytes = ctx.read_bytes(buf, size as usize);
    pipe.extend(bytes);
    if result != 0 {
        ctx.write_u32(result, size);
    }
    st.msgpipe_service(ctx, uid);
    ctx.ret(0);
}

/// int sceKernelReceiveMsgPipe(SceUID uid, void *buf, unsigned int size, int mode, unsigned int *result, unsigned int *timeout)
/// int sceKernelTryReceiveMsgPipe(SceUID uid, void *buf, unsigned int size, int mode, unsigned int *result)
///
/// Satisfied at once when the pipe holds what the mode asks for (FULL: `size` bytes; ASAP:
/// any). Otherwise the TRY form, a DONT_WAIT mode, or the single-thread model report
/// WAIT_TIMEOUT with nothing received; the blocking form PARKS until a send satisfies it
/// (or its `*timeout`, in microseconds, passes).
pub(super) fn msg_pipe_receive(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    try_form: bool,
) -> SvcOutcome {
    let uid = ctx.arg(0) as i32;
    let (buf, size, mode, result) = (ctx.arg(1), ctx.arg(2), ctx.arg(3), ctx.arg(4));
    let timeout_ptr = if try_form { 0 } else { ctx.arg(5) };
    let full = mode & MPP_MODE_FULL != 0;
    let Some(pipe) = st.msg_pipe_mut(uid) else {
        ctx.ret(0x8002_0005);
        return SvcOutcome::Continue;
    };
    let ready = if full {
        pipe.len() >= size as usize
    } else {
        !pipe.is_empty()
    };
    if ready {
        let n = (size as usize).min(pipe.len());
        let bytes: Vec<u8> = pipe.drain(..n).collect();
        ctx.write_bytes(buf, &bytes);
        if result != 0 {
            ctx.write_u32(result, n as u32);
        }
        ctx.ret(0);
        return SvcOutcome::Continue;
    }
    let timeout = (timeout_ptr != 0).then(|| ctx.read_u32(timeout_ptr));
    if try_form || mode & MPP_MODE_DONT_WAIT != 0 || !st.is_preemptive() || timeout == Some(0) {
        if result != 0 {
            ctx.write_u32(result, 0);
        }
        ctx.ret(0x8002_8005); // SCE_KERNEL_ERROR_WAIT_TIMEOUT
        return SvcOutcome::Continue;
    }
    st.msgpipe_block(uid, buf, size, full, result, timeout);
    ctx.ret(0);
    SvcOutcome::Block
}

/// The read-modify-write each `sceKernelAtomic<Op><8|16|32>` performs (psp2/kernel/cpu.h).
/// `XAndGet` returns the NEW value, `GetAndX` the OLD one; `Clear` is `& !value`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AtomicOp {
    Set,
    CompareAndSet,
    AddAndGet,
    SubAndGet,
    AndAndGet,
    OrAndGet,
    XorAndGet,
    ClearAndGet,
    GetAndSet,
    GetAndAdd,
    GetAndSub,
    GetAndAnd,
    GetAndOr,
    GetAndXor,
    GetAndClear,
    ClearMask,
    AddUnless,
}

/// `sceKernelAtomic<op><bits>(store, value[, third])` for 8-, 16- and 32-bit stores.
///
/// One compare-and-swap loop on the aligned word holding the store, for every width: under
/// SMP guest code on other workers writes the same word concurrently (its own `ldrex`/
/// `strex`, or a neighbouring byte), and [`GuestWords::cas_word`] is the one primitive both
/// engines make atomic against it - the same one the lightweight mutex races the guest on. A
/// narrow store changes only its own bytes of the word, and a concurrent write to the rest
/// fails the swap and retries rather than being overwritten.
///
/// Signed results (`SceInt8`/`SceInt16`) come back sign-extended, as AAPCS returns them;
/// `AddUnless` returns `SceBool` - true when it added, i.e. the store did not hold `cmpv`.
/// A NULL or misaligned store stops the run by name: the console faults there, and a
/// handler that quietly read zero would hand the title a lock it does not hold.
///
/// (The 64-bit members need a 64-bit swap neither engine's memory offers yet, and
/// `DecIfPositive` does not say whether it returns the old value or the decremented one;
/// both stay unimplemented rather than guessed.)
pub(super) fn atomic(ctx: &mut GuestCtx, op: AtomicOp, bits: u32) -> SvcOutcome {
    use crate::host::GuestWords;
    let (addr, value, third) = (ctx.arg(0), ctx.arg(1), ctx.arg(2));
    let size = bits / 8;
    if addr == 0 || addr % size != 0 {
        return SvcOutcome::Fatal(format!(
            "sceKernelAtomic{op:?}{bits}: store {addr:#010x} is NULL or not {size}-byte aligned - the console \
             faults here (no silent stub)"
        ));
    }
    let width_mask: u32 = if bits == 32 {
        u32::MAX
    } else {
        (1 << bits) - 1
    };
    let shift = (addr & 3) * 8;
    let field = width_mask << shift;
    let word_addr = addr & !3;
    let (v, c) = (value & width_mask, third & width_mask);
    let (old, ret) = loop {
        let cur = ctx.word(word_addr);
        let old = (cur & field) >> shift;
        let new = match op {
            AtomicOp::Set | AtomicOp::GetAndSet => Some(v),
            AtomicOp::CompareAndSet => (old == v).then_some(c),
            AtomicOp::AddAndGet | AtomicOp::GetAndAdd => Some(old.wrapping_add(v)),
            AtomicOp::SubAndGet | AtomicOp::GetAndSub => Some(old.wrapping_sub(v)),
            AtomicOp::AndAndGet | AtomicOp::GetAndAnd => Some(old & v),
            AtomicOp::OrAndGet | AtomicOp::GetAndOr => Some(old | v),
            AtomicOp::XorAndGet | AtomicOp::GetAndXor => Some(old ^ v),
            AtomicOp::ClearAndGet | AtomicOp::GetAndClear | AtomicOp::ClearMask => Some(old & !v),
            AtomicOp::AddUnless => (old != c).then_some(old.wrapping_add(v)),
        }
        .map(|n| n & width_mask);
        let Some(new) = new else { break (old, old) };
        if ctx.cas_word(word_addr, cur, (cur & !field) | (new << shift)) == cur {
            break (old, new);
        }
    };
    let sign_extend = |x: u32| ((x << (32 - bits)) as i32 >> (32 - bits)) as u32;
    match op {
        AtomicOp::Set | AtomicOp::ClearMask => {}
        AtomicOp::AddUnless => ctx.ret((old != c) as u32),
        AtomicOp::CompareAndSet
        | AtomicOp::GetAndSet
        | AtomicOp::GetAndAdd
        | AtomicOp::GetAndSub
        | AtomicOp::GetAndAnd
        | AtomicOp::GetAndOr
        | AtomicOp::GetAndXor
        | AtomicOp::GetAndClear => ctx.ret(sign_extend(old)),
        _ => ctx.ret(sign_extend(ret)),
    }
    SvcOutcome::Continue
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

/// int sceClibVsnprintf(char *dst, SceSize dst_max, const char *fmt, va_list ap)
///
/// The `va_list` spelling of [`clib_snprintf`], and the one a title's OWN logging wrapper
/// calls: the wrapper takes `(...)`, and hands the list on. Same C99 semantics - at most
/// `dst_max - 1` bytes plus a NUL, and the return is the length that WOULD have been written
/// - and the same formatter, so a message cannot format differently depending on which of
/// the two a title happened to route it through.
///
/// On ARM EABI a `va_list` is a pointer into the caller's argument area, which is what `ap`
/// is here; [`cfmt::format_into_va`] walks it.
pub(super) fn clib_vsnprintf(ctx: &mut GuestCtx, _st: &mut VitaState) {
    let dst = ctx.arg(0);
    let dst_max = ctx.arg(1);
    let fmt_addr = ctx.arg(2);
    let ap = ctx.arg(3);
    let mut out = Vec::new();
    cfmt::format_into_va(&mut out, ctx, fmt_addr, ap);
    let full_len = out.len();
    if dst_max > 0 {
        let n = (dst_max as usize - 1).min(full_len);
        let mut written = out[..n].to_vec();
        written.push(0); // NUL terminator
        ctx.write_bytes(dst, &written);
    }
    ctx.ret(full_len as u32);
}

/// int sceClibVprintf(const char *fmt, va_list ap)
///
/// [`clib_printf`] over a caller-built list, in the same relationship [`clib_vsnprintf`] has
/// to [`clib_snprintf`]. Registered alongside it because a title with one logging wrapper
/// usually has both, and discovering the second one boot later is the cost this whole
/// exercise exists to avoid.
pub(super) fn clib_vprintf(ctx: &mut GuestCtx, st: &mut VitaState) {
    let fmt_addr = ctx.arg(0);
    let ap = ctx.arg(1);
    let mut out = Vec::new();
    cfmt::format_into_va(&mut out, ctx, fmt_addr, ap);
    let n = out.len() as u32;
    st.write_stdout(&out);
    ctx.ret(n);
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
///
/// The answer comes from [`vitaslop_transpiler::mem_compare`] rather than from a loop
/// written here, because this call has an INLINE FORM ([`inline_op`]) whose emitted loop has
/// to compute the same thing. Two hand-written loops in two languages agreeing today is not
/// the same as their agreeing after either is touched, and the disagreement would be silent:
/// the inline form serves every ordinary call and the handler only the ones near the end of
/// guest memory, so a divergence would surface as a rare wrong comparison and nothing else.
///
/// Expression-bodied (no early `return`): `#[hostcall]` inlines the body into a
/// `()` wrapper, so a `return` would escape the wrapper, not this handler.
#[hostcall]
pub(super) fn clib_memcmp(ctx: &mut GuestCtx, a: Ptr, b: Ptr, len: u32) -> i32 {
    // `read_bytes` CLAMPS at the end of guest memory and returns short, so a pointer with
    // fewer than `len` bytes in front of it compares only what is there. That is this
    // handler's own behaviour and the inline form does not reproduce it - it hands exactly
    // that case back here, which is why the two can differ about it without diverging.
    let x = ctx.read_bytes(a.addr(), len as usize);
    let y = ctx.read_bytes(b.addr(), len as usize);
    vitaslop_transpiler::mem_compare(&x, &y)
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
    let nul = src_bytes
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(src_bytes.len());
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
pub(super) fn clib_mspace_create(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    base: Ptr,
    capacity: u32,
    _flag: u32,
) -> Ptr {
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
pub(super) fn clib_mspace_malloc(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    msp: Ptr,
    size: u32,
) -> Ptr {
    Ptr(mspace_alloc(st, msp, size, 0, "sceClibMspaceMalloc"))
}

/// void *sceClibMspaceMemalign(void *msp, SceSize alignment, SceSize size)
#[hostcall]
pub(super) fn clib_mspace_memalign(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    msp: Ptr,
    alignment: u32,
    size: u32,
) -> Ptr {
    Ptr(mspace_alloc(
        st,
        msp,
        size,
        alignment,
        "sceClibMspaceMemalign",
    ))
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
            // >>> REPORTED ONCE PER SPACE, NOT ONCE PER FAILURE.
            //
            // A full mspace returning NULL is a NORMAL outcome that the guest handles: a
            // title that queues work into a fixed pool allocates until the pool is full and
            // retries, which is its own back-pressure. Logging every one of those at `warn`
            // produced thousands of identical lines on a device - enough that the on-screen
            // diagnostics panel reported "855 earlier DISTINCT line(s) dropped" and pushed
            // every real finding out of the one report a person can actually read.
            //
            // The first failure is worth saying: it is where a pool that is too small, or a
            // leak, first shows. After that the COUNT is the only new information, and it
            // is folded into the same line at the end of a run rather than repeated.
            space.note_failure();
            let (used, capacity) = (space.used_bytes(), space.capacity());
            let first = st.mspace_exhausted.insert(msp.addr());
            if first {
                // STATUS, not a warning: the text below says it - this is the guest's own
                // pool and a NULL is an outcome it handles. A title that fills a pool is
                // running as designed, and the panel is where the count is read.
                tracing::info!(
                    target: "vitaslop::status",
                    msp = format_args!("{:#x}", msp.addr()),
                    size,
                    align,
                    used,
                    capacity,
                    "{what}: memory space exhausted -> NULL. This is the guest's OWN pool and a NULL is an outcome it handles, so this says it ONCE per space; a run that does it constantly is a pool the guest is filling faster than it drains, which shows as guest CPU, not as an error."
                );
            }
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
    let freed = st
        .mspaces
        .get_mut(msp.addr())
        .is_some_and(|s| s.free(ptr.addr()));
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

/// void __stack_chk_fail(void)
/// A stack-protector epilogue found its frame's canary overwritten: the guest smashed its own
/// stack (or we corrupted it). The device terminates the process, so the run stops here, naming
/// the function whose check failed - `lr` is the return address inside it.
pub(super) fn stack_chk_fail(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    SvcOutcome::Fatal(format!(
        "__stack_chk_fail: thread {:#x} found its stack canary overwritten in the function \
         returning to {:#010x} - a buffer overran a stack frame there",
        st.current_thread(),
        ctx.regs[14],
    ))
}

#[cfg(test)]
mod atomic_tests {
    //! `sceKernelAtomic*` against guest memory: the value each form returns (new vs old,
    //! sign-extended), and that a narrow store leaves the rest of its word alone.
    use super::*;
    use crate::{SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::REG_COUNT;

    const WORD: u32 = 0x100;

    /// Run one op on a fresh context over `bytes`; returns (r0, outcome was Continue).
    fn op(bytes: &mut [u8], o: AtomicOp, bits: u32, args: [u32; 3]) -> (u32, bool) {
        let mut regs = [0u32; REG_COUNT];
        regs[..3].copy_from_slice(&args);
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut mem = SliceMemory(bytes);
        let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        let out = atomic(&mut ctx, o, bits);
        (ctx.regs[0], matches!(out, SvcOutcome::Continue))
    }

    fn word(bytes: &[u8]) -> u32 {
        u32::from_le_bytes(bytes[WORD as usize..WORD as usize + 4].try_into().unwrap())
    }

    fn set_word(bytes: &mut [u8], v: u32) {
        bytes[WORD as usize..WORD as usize + 4].copy_from_slice(&v.to_le_bytes());
    }

    #[test]
    fn thirty_two_bit_forms_return_new_or_old_as_named() {
        let mut b = vec![0u8; 0x200];
        set_word(&mut b, 10);
        assert_eq!(op(&mut b, AtomicOp::AddAndGet, 32, [WORD, 5, 0]).0, 15);
        assert_eq!(op(&mut b, AtomicOp::GetAndSub, 32, [WORD, 3, 0]).0, 15);
        assert_eq!(word(&b), 12);
        assert_eq!(op(&mut b, AtomicOp::GetAndClear, 32, [WORD, 4, 0]).0, 12);
        assert_eq!(word(&b), 8);
        assert_eq!(op(&mut b, AtomicOp::OrAndGet, 32, [WORD, 3, 0]).0, 11);
        assert_eq!(op(&mut b, AtomicOp::XorAndGet, 32, [WORD, 1, 0]).0, 10);
        assert_eq!(
            op(&mut b, AtomicOp::GetAndSet, 32, [WORD, 0xFFFF_FFFF, 0]).0,
            10
        );
        assert_eq!(op(&mut b, AtomicOp::AndAndGet, 32, [WORD, 0xF0, 0]).0, 0xF0);
    }

    #[test]
    fn compare_and_set_and_add_unless_store_only_on_their_condition() {
        let mut b = vec![0u8; 0x200];
        set_word(&mut b, 7);
        // A mismatch returns the current value and stores nothing.
        assert_eq!(op(&mut b, AtomicOp::CompareAndSet, 32, [WORD, 6, 99]).0, 7);
        assert_eq!(word(&b), 7);
        assert_eq!(op(&mut b, AtomicOp::CompareAndSet, 32, [WORD, 7, 99]).0, 7);
        assert_eq!(word(&b), 99);
        // AddUnless: false (nothing added) when the store holds `cmpv`.
        assert_eq!(op(&mut b, AtomicOp::AddUnless, 32, [WORD, 1, 99]).0, 0);
        assert_eq!(word(&b), 99);
        assert_eq!(op(&mut b, AtomicOp::AddUnless, 32, [WORD, 1, 5]).0, 1);
        assert_eq!(word(&b), 100);
    }

    #[test]
    fn narrow_stores_touch_only_their_bytes_and_sign_extend() {
        let mut b = vec![0u8; 0x200];
        set_word(&mut b, 0x1122_3344);
        // Byte 2 (0x22) + 0x70 = 0x92, a negative SceInt8: returned sign-extended.
        assert_eq!(
            op(&mut b, AtomicOp::AddAndGet, 8, [WORD + 2, 0x70, 0]).0,
            0xFFFF_FF92
        );
        assert_eq!(word(&b), 0x1192_3344);
        // The high halfword, wrapping: 0x1192 - 0x1193 = 0xFFFF = -1.
        assert_eq!(
            op(&mut b, AtomicOp::SubAndGet, 16, [WORD + 2, 0x1193, 0]).0,
            0xFFFF_FFFF
        );
        assert_eq!(word(&b), 0xFFFF_3344);
        // The low byte's CompareAndSet compares only its own bits.
        assert_eq!(
            op(&mut b, AtomicOp::CompareAndSet, 8, [WORD, 0x44, 0x55]).0,
            0x44
        );
        assert_eq!(word(&b), 0xFFFF_3355);
        // ClearMask is void: it changes the store and leaves r0 as the call left it.
        op(&mut b, AtomicOp::ClearMask, 16, [WORD, 0x3300, 0]);
        assert_eq!(word(&b), 0xFFFF_0055);
    }

    #[test]
    fn a_null_or_misaligned_store_stops_the_run() {
        let mut b = vec![0u8; 0x200];
        assert!(!op(&mut b, AtomicOp::AddAndGet, 32, [0, 1, 0]).1);
        assert!(!op(&mut b, AtomicOp::AddAndGet, 32, [WORD + 2, 1, 0]).1);
        assert!(!op(&mut b, AtomicOp::AddAndGet, 16, [WORD + 1, 1, 0]).1);
        assert!(op(&mut b, AtomicOp::AddAndGet, 8, [WORD + 3, 1, 0]).1);
    }
}

#[cfg(test)]
mod msg_pipe_tests {
    use super::*;
    use crate::world::DeterministicWorld;
    use crate::{SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::REG_COUNT;

    const COUNTS: u32 = 0x100;

    fn call(
        st: &mut VitaState,
        bytes: &mut [u8],
        f: fn(&mut GuestCtx, &mut VitaState),
        args: &[u32],
    ) -> u32 {
        let mut regs = [0u32; REG_COUNT];
        regs[..args.len()].copy_from_slice(args);
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut mem = SliceMemory(bytes);
        let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        f(&mut ctx, st);
        ctx.regs[0]
    }

    /// The backtrace is the caller's own frame - `sp` and the return address - counted, with
    /// the depth (or SCE_OK under DONT_EXCEED) returned; a buffer too small for a frame gets
    /// none.
    #[test]
    fn backtrace_self_reports_the_callers_frame() {
        let mut st = VitaState::new(0, 0x4000, Box::new(DeterministicWorld::default()));
        let mut b = vec![0u8; 0x400];
        let mut run = |b: &mut [u8], args: [u32; 4]| {
            let mut regs = [0u32; REG_COUNT];
            regs[..4].copy_from_slice(&args);
            regs[13] = 0x3F0;
            regs[14] = 0x8100_1235;
            let mut vfp = [0u32; VFP_ARG_COUNT];
            let mut mem = SliceMemory(b);
            let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            backtrace_self(&mut ctx, &mut st);
            ctx.regs[0]
        };
        let w = |b: &[u8], a: usize| u32::from_le_bytes(b[a..a + 4].try_into().unwrap());
        assert_eq!(run(&mut b, [0x100, 0x100, 0x200, 0]), 1);
        assert_eq!((w(&b, 0x100), w(&b, 0x104), w(&b, 0x200)), (0x3F0, 0x8100_1235, 1));
        assert_eq!(run(&mut b, [0x100, 0x100, 0x200, 2]), 0, "DONT_EXCEED returns SCE_OK");
        assert_eq!(run(&mut b, [0x100, 4, 0x200, 0]), 1);
        assert_eq!(w(&b, 0x200), 0, "no room for a frame");
        assert_eq!(run(&mut b, [0, 0x100, 0x200, 0]), SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }

    /// A cancel wakes each parked receiver with WAIT_CANCEL, counts them (and the senders,
    /// which never park), and empties the pipe; a delete wakes the rest with WAIT_DELETE.
    #[test]
    fn cancel_releases_receivers_and_empties_the_pipe() {
        let mut st = VitaState::new(0, 0x4000, Box::new(DeterministicWorld::default()));
        st.set_preemptive(true);
        let mut b = vec![0xAAu8; 0x400];
        let uid = call(&mut st, &mut b, msg_pipe_create, &[0, 0, 0, 0x100]) as i32;
        st.msg_pipe_mut(uid).unwrap().extend([1u8, 2]);
        st.set_current(7);
        st.msgpipe_block(uid, 0x200, 4, true, 0, None);
        st.set_current(8);
        st.msgpipe_block(uid, 0x200, 4, true, 0, None);
        st.take_wakes();

        assert_eq!(
            call(&mut st, &mut b, msg_pipe_cancel, &[uid as u32, COUNTS, COUNTS + 4]),
            0
        );
        assert_eq!(u32::from_le_bytes(b[0x100..0x104].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(b[0x104..0x108].try_into().unwrap()), 2);
        assert_eq!(st.take_wakes(), vec![7, 8]);
        assert_eq!(st.take_resume_code(7), Some(SCE_KERNEL_ERROR_WAIT_CANCEL));
        assert_eq!(st.take_resume_code(8), Some(SCE_KERNEL_ERROR_WAIT_CANCEL));
        assert!(st.msg_pipe(uid).unwrap().is_empty());
        assert_eq!(b[0x200], 0xAA, "a cancelled receive writes nothing");

        // NULL count pointers are allowed; nothing is waiting now.
        assert_eq!(call(&mut st, &mut b, msg_pipe_cancel, &[uid as u32, 0, 0]), 0);
        assert!(st.take_wakes().is_empty());

        st.set_current(9);
        st.msgpipe_block(uid, 0x200, 4, true, 0, None);
        assert_eq!(call(&mut st, &mut b, msg_pipe_delete, &[uid as u32]), 0);
        assert_eq!(
            st.take_resume_code(9),
            Some(crate::host::SCE_KERNEL_ERROR_WAIT_DELETE)
        );
        assert_eq!(
            call(&mut st, &mut b, msg_pipe_cancel, &[uid as u32, 0, 0]),
            0x8002_0005
        );
    }
}
