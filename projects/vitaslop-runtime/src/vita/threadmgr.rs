//! SceThreadmgr: the thread-manager primitives not wrapped by SceLibKernel. The
//! user-facing create/start/wait wrappers live in `libkernel`; what remains here
//! are the direct primitives a program can also call.

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::SvcOutcome;

/// SceUID sceKernelGetProcessId(void)
/// A single process, so a fixed nonzero id is faithful and stable.
#[hostcall]
pub(super) fn get_process_id(_st: &mut VitaState) -> i32 {
    0x1000
}

/// int sceKernelGetThreadCurrentPriority(void)
/// The scheduler priority of the calling thread (lower value = higher priority).
/// A title reads this to spawn a worker at a relative priority or to briefly raise
/// its own; returning the real running priority keeps that arithmetic correct.
#[hostcall]
pub(super) fn get_thread_current_priority(st: &mut VitaState) -> i32 {
    st.current_priority()
}

/// The CPU affinity mask a thread gets when it has never asked for one.
///
/// `SCE_KERNEL_CPU_MASK_USER_ALL` in `psp2/kernel/threadmgr/thread.h` terms: the three
/// cores a title's threads may run on (the console has four Cortex-A9 cores and the system
/// software reserves one - the same fact [`crate::host::guest_cores`] encodes). 0 is
/// deliberately NOT the default: it means "inherit", and reporting it would tell a title
/// that queried an untouched thread that its affinity is unset when the kernel would name
/// the real set.
pub(crate) const CPU_MASK_USER_ALL: i32 = 0x0007_0000;

/// int sceKernelChangeThreadCpuAffinityMask(SceUID thid, int mask)
///
/// Records the requested mask so [`get_thread_cpu_affinity_mask`] reports it back, and
/// changes NOTHING about where the thread runs.
///
/// # Why recording it is the faithful answer and pinning would not be
/// This emulator runs guest threads on ONE baton, interleaved cooperatively - that is the
/// whole scheduler. There is no per-core placement to honour, so a mask cannot be obeyed;
/// the question is only what a title is TOLD. A title sets affinity to spread work and
/// then, commonly, reads it back to confirm - so a setter that silently discards the value
/// makes the getter contradict it, which is a disagreement the guest can see. Accepting and
/// remembering it is consistent, and the scheduling difference from the console is already
/// modelled where it belongs: `guest_cores` divides the clock charge by how many threads
/// would really have been running at once.
#[hostcall]
pub(super) fn change_thread_cpu_affinity_mask(st: &mut VitaState, thid: i32, mask: i32) -> i32 {
    st.set_thread_cpu_affinity(thid, mask)
}

/// int sceKernelGetThreadCpuAffinityMask(SceUID thid)
///
/// Returns the mask itself (a non-negative value) or a negative error, which is why it is
/// an `i32` return and not an out-parameter.
#[hostcall]
pub(super) fn get_thread_cpu_affinity_mask(st: &mut VitaState, thid: i32) -> i32 {
    st.thread_cpu_affinity(thid)
}

/// int sceKernelDeleteThread(SceUID thid)
/// Delete a DORMANT thread: drop its record so its SceUID stops resolving, and give its
/// stack back for reuse. A running thread cannot be deleted (`NOT_DORMANT`) - accepting
/// that would invalidate the id of a thread the scheduler is about to resume, which is a
/// corruption rather than an error. See [`VitaState::delete_thread`].
#[hostcall]
pub(super) fn delete_thread(st: &mut VitaState, thid: i32) -> i32 {
    match st.delete_thread(thid) {
        Ok(()) => 0,
        Err(e) => e as i32,
    }
}

/// int sceKernelChangeThreadVfpException(int clearMask, int setMask)
///
/// Selects which VFP/NEON floating-point exceptions (invalid, div-by-zero, overflow,
/// underflow, inexact, input-denormal) trap for the calling thread by clearing then
/// setting bits in its FPSCR exception-enable field. We evaluate every float and NEON
/// op with standard non-trapping IEEE semantics (no host trap is ever raised), so this
/// only records intent - it never changes a numeric result or control flow. Accepted
/// with success. Returns 0.
#[hostcall]
pub(super) fn change_thread_vfp_exception(_clear_mask: i32, _set_mask: i32) -> i32 {
    0
}

/// int sceKernelCheckCallback(void)
///
/// Run the CALLING thread's pending kernel callbacks and report how many ran. A title puts
/// it in its main loop so that callbacks it registered with `sceKernelCreateCallback` -
/// power/exit notifications, its own `sceKernelNotifyCallback` posts - are delivered at a
/// point of its choosing rather than inside an arbitrary wait.
///
/// **ZERO here is the complete answer, not a stub.** A callback can only exist if it was
/// created, and `sceKernelCreateCallback` has no handler in this engine at all: a title that
/// makes one gets the unimplemented-NID hard-fail naming it, by the same rule as every other
/// unhandled call. So a title that reaches HERE has no callback object, nothing can be
/// pending for it, and "none ran" is what the kernel would report. There is no silent gap
/// behind this: the moment a title actually uses the callback surface, the run stops and
/// says so.
///
/// The service-state pumps (`sceNpCheckCallback`, `sceNetCtlCheckCallback`) are a different
/// mechanism with their own registration and their own deliveries - see `vita::services`.
#[hostcall]
pub(super) fn check_callback(_st: &mut VitaState) -> i32 {
    0
}

/// int sceKernelChangeThreadPriority(SceUID thid, int priority)
///
/// Retarget a thread's scheduler priority; `thid` 0 is the calling thread. Returns
/// the PREVIOUS priority, which is what a title saves so it can restore after a
/// temporary boost. This genuinely moves the thread in the run order - the scheduler
/// picks the highest-priority runnable thread - so a title raising a loader above the
/// main thread gets the ordering it asked for rather than a value it can read back
/// and no change in behaviour.
#[hostcall]
pub(super) fn change_thread_priority(st: &mut VitaState, thid: i32, priority: i32) -> i32 {
    match st.change_thread_priority(thid, priority) {
        Ok(previous) => previous,
        Err(e) => e as i32,
    }
}

/// The inline form of `sceKernelDelayThread`: a yield (`delay <= 1`) with nobody to yield to
/// is answered in guest code from the host-mirror block, and everything else reaches
/// [`delay_thread`] unchanged. Both spellings, as the dispatch routes both to one handler.
/// See [`vitaslop_transpiler::InlineOp::DelayYield`] for the argument and the measurement.
pub(crate) fn inline_op(func_nid: u32) -> Option<vitaslop_transpiler::InlineOp> {
    use crate::vita::mirror::{SLOT_ELIDE_RUN, SLOT_YIELD_FREE};
    match func_nid {
        crate::vita::tm_nid::DELAY_THREAD | crate::vita::tm_nid::DELAY_THREAD_CB => {
            Some(vitaslop_transpiler::InlineOp::DelayYield {
                free_slot: SLOT_YIELD_FREE,
                run_slot: SLOT_ELIDE_RUN,
                cap: VitaState::ELIDE_RUN_CAP,
            })
        }
        _ => None,
    }
}

/// int sceKernelDelayThread(SceUInt delay)
///
/// Preemptive: a REAL timed sleep - park the caller until the virtual clock
/// reaches `now + delay` ([`VitaState::sleep_park`]); the scheduler wakes it on a
/// display flip's clock advance, or jumps the clock straight to the deadline when
/// nothing else is runnable. A no-op "just succeed" here turns every
/// delay-then-poll loop into a full-speed busy spin (millions of host calls
/// starving the threads doing real work).
///
/// Single-thread model: still a no-op success (workers run synchronously, so
/// there is nothing to yield to and the clock is host-driven).
pub(super) fn delay_thread(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let delay_us = ctx.arg(0);
    ctx.ret(0);
    if !st.is_preemptive() {
        return SvcOutcome::Continue;
    }
    // >>> THE CENSUS RUNS FIRST, BECAUSE THE CASE IT COULD NOT SEE IS THE COMMON ONE.
    //
    // It used to sit below the early return, so every `delay(0)` and `delay(1)` was invisible
    // to it - and those are the ones a polling loop makes. On a football title the call-site
    // profiler put 10.4 MILLION calls at one address and the delay census listed that address
    // NOWHERE, which reads as "that site does not sleep" rather than "this instrument cannot
    // see it" [[vitaslop-instrument-failure-imitating-its-subject]].
    delay_census(delay_us, ctx.regs[14]);
    // A zero/one-us delay is "give someone else the CPU", not a real sleep - and not
    // a display frame either (see [`SvcOutcome::Flip`]). A worker polling in a
    // delay(0) loop hits this thousands of times per rendered frame.
    //
    // >>> AND WITH NOBODY TO GIVE IT TO, THE KERNEL RETURNS TO THE CALLER. Suspending anyway
    // costs a full fiber suspend and resume for a scheduler round that re-picks this same
    // thread - 2,377 of them a frame on that title, against 4 on another. See
    // [`VitaState::yield_would_repick_this_thread`] for why the answer is exact.
    if delay_us <= 1 {
        // >>> A YIELD THAT REACHED THE HOST FROM THE INLINE FORM AT ITS CAP IS DUE A SUSPEND.
        //
        // The emitted `DelayYield` elides up to `ELIDE_RUN_CAP` yields a slice in guest code and
        // falls through here for the next one. Eliding THAT one through the handler's own
        // (still-zero) count would spend another whole cap of crossings before the suspend
        // the cap exists to force. MEASURED on a football title's browser frame: 5.0 million
        // host-elided yields remained after the inline form landed, one crossing each, for
        // exactly this reason. Reading the run word costs one guest-memory read on this path
        // only, which is one per cap rather than one per yield.
        if st.inline_elide_run_at_cap(ctx) {
            return SvcOutcome::Reschedule;
        }
        if st.yield_would_repick_this_thread() {
            return SvcOutcome::Continue;
        }
        return SvcOutcome::Reschedule;
    }
    st.sleep_park(delay_us as u64);
    SvcOutcome::Block
}

/// Diagnostic (`VITASLOP_DELAY_CENSUS=1`): every `sceKernelDelayThread` tallied by (call site,
/// requested microseconds), printed by [`dump_delay_census`] at the end of a run.
///
/// A polling thread's cost is `iterations x crossings`, and the iteration count is decided by
/// how long it asked to sleep - which is the ONE number the call-site profiler cannot show.
/// Without it, "this loop runs 8,000 times a second" reads the same whether the title asked for
/// a 125 us sleep or asked for a millisecond and this scheduler woke it eight times too often.
/// Those are opposite defects with opposite fixes.
static DELAY_HIST: std::sync::Mutex<Option<std::collections::BTreeMap<(u32, u32), u64>>> =
    std::sync::Mutex::new(None);

fn delay_census(delay_us: u32, lr: u32) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| crate::knobs::flag("VITASLOP_DELAY_CENSUS")) {
        return;
    }
    let mut g = DELAY_HIST.lock().unwrap_or_else(|e| e.into_inner());
    *g.get_or_insert_with(Default::default).entry((lr, delay_us)).or_insert(0) += 1;
}

/// Print what [`delay_census`] gathered, most-called first. A no-op unless the knob is set.
pub fn dump_delay_census(top: usize) {
    let g = DELAY_HIST.lock().unwrap_or_else(|e| e.into_inner());
    let Some(hist) = g.as_ref() else { return };
    let mut rows: Vec<_> = hist.iter().map(|(&(lr, us), &n)| (lr, us, n)).collect();
    rows.sort_by_key(|&(_, _, n)| std::cmp::Reverse(n));
    eprintln!("--- sceKernelDelayThread by (call site, requested us): count ---");
    for (lr, us, n) in rows.into_iter().take(top) {
        eprintln!("  {n:>10}  {us:>8} us @ lr={lr:#010x}");
    }
}
