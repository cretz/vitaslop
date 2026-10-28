//! The preemptive block/wake bookkeeping in `VitaState`, exercised directly
//! (no wasm engine): a parked thread and its release for each primitive. The
//! engine-level proof that these outcomes drive real fibers lives in
//! `vitaslop-native/tests/threaded_scheduler.rs`.
use super::*;
use crate::world::DeterministicWorld;

fn state() -> VitaState {
    let mut st = VitaState::new(0x1000, 0x10000, Box::new(DeterministicWorld::default()));
    st.set_preemptive(true);
    st
}

/// A grain's NGS callbacks run as guest calls on the CALLING thread, one per replay of the
/// dispatch, each on its own info slot; the last one returning ends the replay.
#[test]
fn a_grains_ngs_callbacks_run_on_the_calling_thread_in_order() {
    const AUDIO: i32 = 0x40;
    let mut st = state();
    st.set_current(AUDIO);
    let mut regs = [0u32; REG_COUNT];
    let mut vfp = [0u32; VFP_ARG_COUNT];
    let mut bytes = vec![0u8; 0x10000];
    let mut mem = SliceMemory(&mut bytes);
    let calls = [(0x8100_0001, [1u8; 0x1c]), (0x8100_0001, [2u8; 0x1c])];
    {
        let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0x1000);
        assert!(st.ngs_sync_stash(&mut ctx, &calls));
    }
    assert!(st.ngs_sync_replaying());
    let mut ran = Vec::new();
    loop {
        st.guest_call_begin();
        let done = st.ngs_sync_run();
        let pending = st.guest_call_end();
        if done {
            assert!(!pending);
            break;
        }
        assert!(pending, "a run that is not done is waiting on a guest call");
        let call = st.take_guest_call(AUDIO).expect("the call being run");
        assert_eq!(call.entry, 0x8100_0001);
        ran.push(call.args[0]);
        st.guest_call_returned(AUDIO, 0, 0, &mem, 0x1000);
    }
    assert_eq!(ran.len(), 2, "both callbacks ran");
    assert_ne!(ran[0], ran[1], "each on its own info slot");
    assert!(!st.ngs_sync_replaying(), "the stash is gone");
    assert!(st.ngs_sync_run(), "nothing left to run");
}

/// The INITIAL thread has no create record and must still be signallable, and a signal is a
/// latch of one: a second send before the wait consumes it is refused, not counted.
#[test]
fn the_initial_thread_takes_a_signal_and_a_signal_is_a_latch_of_one() {
    let mut st = state();
    assert_eq!(
        st.send_signal(MAIN_THID),
        Ok(()),
        "the initial thread is a known thread"
    );
    assert_eq!(
        st.send_signal(MAIN_THID),
        Err(SCE_KERNEL_ERROR_ALREADY_SENT),
        "unconsumed: refused"
    );
    assert!(st.take_signal(MAIN_THID));
    assert!(!st.take_signal(MAIN_THID), "one send, one take");
    assert_eq!(
        st.send_signal(MAIN_THID),
        Ok(()),
        "consumed, so it can be sent again"
    );
    assert_eq!(
        st.send_signal(0x7777),
        Err(SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID)
    );
}

/// Guest memory for the tests that touch guest-resident state, as a sparse word map -
/// a lightweight mutex lives in the guest's own work area now, so a test of one needs
/// somewhere for it to live.
#[derive(Default)]
pub(super) struct Words(std::collections::BTreeMap<u32, u32>);

impl GuestWords for Words {
    fn word(&self, addr: u32) -> u32 {
        self.0.get(&addr).copied().unwrap_or(0)
    }
    fn set_word(&mut self, addr: u32, value: u32) {
        self.0.insert(addr, value);
    }
}

/// A work area laid out as a created lightweight mutex, the way
/// `sceKernelCreateLwMutex` leaves it.
fn created_lwmutex(st: &mut VitaState, w: &mut Words, work: u32) {
    st.lwmutex_register(w, work);
}

/// The MOST RECENT park a thread registered, in microseconds from now. The last one,
/// not the first: `sleep_waiters` accumulates within a test, and the first entry is
/// whatever an earlier step parked on.
fn parked_us(st: &VitaState, thid: i32) -> u64 {
    let (_, deadline) = st
        .sleep_waiters
        .iter()
        .rev()
        .find(|(t, _)| *t == thid)
        .copied()
        .expect("thread is not parked");
    deadline.saturating_sub(st.now_us())
}

/// The LATCH times a run of flips schedules, which is what the vblank floor is about -
/// they land on the grid and never closer together than one period, whatever the guest
/// does in between.
fn latches(st: &VitaState) -> Vec<u64> {
    st.flip_pacer.latches()
}

#[test]
fn a_flip_latches_on_a_vblank_edge_not_a_period_after_the_request() {
    const FRAME_US: u64 = 1_000_000 / 60;
    let mut st = state();
    st.set_current(1);
    // A frame whose work took 5 ms latches at the vblank, 11.67 ms away - NOT a
    // whole period later. That is what phase-locks a cheap frame to 60 Hz instead
    // of letting it drift by its own render time every frame.
    st.advance_time_to(5_000);
    st.pace_flip(FRAME_US);
    assert_eq!(latches(&st), vec![FRAME_US], "the latch is ON the edge");
    // A frame that overruns its period MISSES the vblank and waits for the next
    // one, so it costs two whole periods - the console's 60/30 quantisation, not a
    // continuum. Here the guest lands 1 ms past the 2nd edge and latches at the 3rd.
    st.advance_time_to(2 * FRAME_US + 1_000);
    st.pace_flip(FRAME_US);
    assert_eq!(*latches(&st).last().expect("a latch"), 3 * FRAME_US);
}

/// The queue is as DEEP as the guest declared, and the caller parks only when it is
/// full - see `pace_flip` for the display period that a forced depth of one put into
/// every frame of a self-pacing title.
#[test]
fn a_flip_parks_only_when_the_display_queue_is_full() {
    const FRAME_US: u64 = 1_000_000 / 60;
    let mut st = state();
    st.set_current(1);
    st.set_display_queue_depth(3);
    // Start off the grid, so "the next vblank" is a real wait and not the zero-length
    // one a clock sitting exactly on an edge produces.
    st.advance_time_to(1_000);
    // Three frames may be in flight, so the first three queue without waiting - and
    // each one still reserves its own vblank, so the LATCHES are a period apart.
    for i in 1..=3u64 {
        assert_eq!(
            st.pace_flip(FRAME_US),
            0,
            "frame {i} fits in a queue of three"
        );
        assert_eq!(*latches(&st).last().expect("a latch"), i * FRAME_US);
    }
    // The fourth is one too many: it waits for the OLDEST to reach the panel, which
    // is the first frame's latch, and no longer.
    assert_eq!(
        st.pace_flip(FRAME_US),
        FRAME_US - 1_000,
        "the queue is full"
    );
    assert_eq!(parked_us(&st, 1), FRAME_US - 1_000);
    // A guest that never declares a depth gets one: ONE frame in flight, so the second
    // flip waits for the first to latch - and not, as this used to, for the vblank
    // after that.
    let mut one = state();
    one.set_current(1);
    one.advance_time_to(1_000);
    assert_eq!(one.pace_flip(FRAME_US), 0);
    assert_eq!(one.pace_flip(FRAME_US), FRAME_US - 1_000);
}

#[test]
fn the_setbuf_sync_mode_is_recorded_and_changes_no_pacing() {
    const FRAME_US: u64 = 1_000_000 / 60;
    let mut st = state();
    st.set_current(1);
    assert_eq!(
        st.display_sync(),
        VitaState::SETBUF_NEXTFRAME,
        "the safe default"
    );
    st.advance_time_to(5_000);
    st.pace_flip(FRAME_US);
    // IMMEDIATE decides whether the scanout's pointer changes MID-SCAN, which tears.
    // It is not a swap interval and it does not let a title present faster, so the
    // vblank floor is unchanged by it. Both readings that said otherwise were tried
    // and both broke the one retail title that asks for it - see `set_display_sync`.
    st.set_display_sync(VitaState::SETBUF_IMMEDIATE);
    assert_eq!(st.display_sync(), VitaState::SETBUF_IMMEDIATE, "recorded");
    // The second flip is reserved for the vblank after the first, and asking for
    // IMMEDIATE does not release it: the floor is the panel, not the mode. With a
    // queue of one the caller waits for the FIRST frame to reach the panel, and the
    // second is then scheduled for the vblank after that.
    st.advance_time_to(10_000);
    assert_eq!(st.pace_flip(FRAME_US), FRAME_US - 10_000);
    assert_eq!(*latches(&st).last().expect("a latch"), 2 * FRAME_US);
}

/// A display wait that follows a PRESENT has nothing left to wait for in this engine -
/// the frame reached the panel inside `sceGxmDisplayQueueAddEntry` and `pace_flip` has
/// already held the queueing thread to the panel's rate. Parking it a second time is
/// the whole display period that made one 30 fps title's clock run 2.99x its flips.
#[test]
fn a_present_satisfies_exactly_one_display_wait() {
    let mut st = state();
    assert!(!st.take_present_since_wait(), "nothing presented yet");
    st.present(0x1000);
    assert!(
        st.take_present_since_wait(),
        "the wait after a present is satisfied"
    );
    assert!(!st.take_present_since_wait(), "and only the FIRST one is");
}

#[test]
fn a_vblank_wait_joins_the_heartbeat_and_never_returns_instantly() {
    const FRAME_US: u64 = 1_000_000 / 60;
    let mut st = state();
    st.set_current(1);
    // Called 5 ms into a frame, one vblank away is 11.67 ms - not 16.67.
    st.advance_time_to(5_000);
    assert_eq!(st.vblank_park(1, FRAME_US), FRAME_US - 5_000);
    // n counts EDGES, so two vblanks from the same point is one more period.
    assert_eq!(st.vblank_park(2, FRAME_US), 2 * FRAME_US - 5_000);
    // Exactly ON an edge the wait is for the NEXT one. A full period, never zero:
    // a loop of instantly-returning vblank waits would spin without ever advancing
    // the clock, which is the shape that livelocks this scheduler.
    st.advance_time_to(4 * FRAME_US);
    assert_eq!(st.vblank_park(1, FRAME_US), FRAME_US);
}

#[test]
fn storage_park_is_paid_by_frames_not_by_the_game_clock() {
    const FRAME_US: u64 = 1_000_000 / 60;
    let mut st = state();
    // Thread 1 pays for a transfer worth two and a bit frames.
    st.set_current(1);
    st.io_park(2 * FRAME_US + 10);
    assert!(st.take_wakes().is_empty());
    // The game clock running does NOT pay for it: a transfer is the device's time,
    // and coupling the two is what livelocked (the clock only moves on flips, and
    // the title is waiting on this very load).
    st.advance_time_to(st.now_us() + 10_000_000);
    assert!(st.take_wakes().is_empty());
    // Rendered frames do. Two are not enough; the third completes it.
    st.advance_io_frame(FRAME_US);
    st.advance_io_frame(FRAME_US);
    assert!(st.take_wakes().is_empty());
    st.advance_io_frame(FRAME_US);
    assert_eq!(st.take_wakes(), vec![1]);
}

#[test]
fn quantum_charges_are_netted_out_of_the_next_frame() {
    const FRAME_US: u64 = 1_000_000 / 60;
    let mut st = state();
    st.set_current(1);
    st.io_park(FRAME_US);
    // A whole frame's worth of storage time charged by quanta must not be charged
    // AGAIN by the flip that follows - a rendering title advances exactly one frame
    // of storage time per frame however its quanta fell.
    st.charge_io_quantum(FRAME_US / 2);
    st.charge_io_quantum(FRAME_US / 2);
    assert_eq!(st.take_wakes(), vec![1]);
    st.set_current(2);
    st.io_park(1);
    st.advance_io_frame(FRAME_US);
    assert!(
        st.take_wakes().is_empty(),
        "the flip owed nothing, so nothing completed"
    );
}

#[test]
fn the_idle_path_completes_the_earliest_transfer() {
    let mut st = state();
    st.set_current(1);
    st.io_park(5_000);
    st.set_current(2);
    st.io_park(1_000);
    // Nothing else can run, so the device is the only thing with work left: the
    // earliest transfer completes, and only that one.
    assert!(st.release_earliest_io());
    assert_eq!(st.take_wakes(), vec![2]);
    assert!(st.has_io_waiters());
    assert!(st.release_earliest_io());
    assert_eq!(st.take_wakes(), vec![1]);
    assert!(!st.release_earliest_io(), "nothing outstanding");
}

/// The half a guest-visible conformance case cannot reach: what a timer does when
/// the clock MOVES. `vita_timer` runs a real velf through the loader, but that corpus
/// has no scheduler and nothing charges the clock, so it can only assert that a
/// running timer never goes backward. Here the clock is driven directly.
#[test]
fn timer_counts_only_while_running() {
    let mut st = state();
    let t = st.create_timer("System Debug Timer", 0);

    // Created stopped: the clock may run, the timer does not.
    st.charge_cpu_quantum(1_000);
    assert_eq!(
        st.timer_time_us(t),
        Some(0),
        "a timer counts nothing before it is started"
    );

    assert_eq!(st.timer_start(t), Some(false), "not already running");
    st.charge_cpu_quantum(3_000);
    assert_eq!(st.timer_time_us(t), Some(3_000));

    // A second start must NOT restart it: the caller is told, and the count stands.
    assert_eq!(st.timer_start(t), Some(true), "already running");
    assert_eq!(
        st.timer_time_us(t),
        Some(3_000),
        "a double start must not lose the count"
    );

    // Stopped: what it counted is banked, and further clock does not reach it.
    assert_eq!(st.timer_stop(t), Some(true));
    st.charge_cpu_quantum(5_000);
    assert_eq!(
        st.timer_time_us(t),
        Some(3_000),
        "a stopped timer does not keep counting"
    );
    assert_eq!(st.timer_stop(t), Some(false), "already stopped");

    // Restarting resumes from the banked count rather than from zero.
    assert_eq!(st.timer_start(t), Some(false));
    st.charge_cpu_quantum(2_000);
    assert_eq!(
        st.timer_time_us(t),
        Some(5_000),
        "a restart resumes, it does not reset"
    );

    // Set replaces the count and keeps it running from there.
    assert_eq!(
        st.timer_set_time_us(t, 100),
        Some(5_000),
        "returns what it was"
    );
    st.charge_cpu_quantum(1_000);
    assert_eq!(st.timer_time_us(t), Some(1_100));

    // Name lookup is what `sceKernelOpenTimer` resolves.
    assert_eq!(st.timer_by_name("System Debug Timer"), Some(t));
    assert_eq!(st.timer_by_name("no such timer"), None);
    assert_eq!(st.timer_name(t), "System Debug Timer");
    assert!(st.timer_running(t));

    // A second timer is independent of the first.
    let other = st.create_timer("other", 0);
    assert_eq!(st.timer_start(other), Some(false));
    st.charge_cpu_quantum(500);
    assert_eq!(st.timer_time_us(other), Some(500));
    assert_eq!(st.timer_time_us(t), Some(1_600));

    assert!(st.timer_delete(t));
    assert_eq!(st.timer_time_us(t), None, "a deleted uid is gone");
    assert!(!st.timer_delete(t), "deleting twice is not a delete");
    assert_eq!(
        st.timer_time_us(other),
        Some(500),
        "an unrelated timer is untouched"
    );
}

#[test]
fn semaphore_parks_then_a_signal_wakes_and_consumes() {
    let mut st = state();
    let sem = st.create_sema("test", 0, 0, i32::MAX);
    // Thread 1 wants 1 but the count is 0: it cannot acquire and parks.
    st.set_current(1);
    assert!(!st.sema_try_acquire(sem, 1));
    st.sema_block(sem, 1, 0);
    assert!(st.take_wakes().is_empty());
    // Thread 2 signals: thread 1 is released and the token is consumed for it.
    st.set_current(2);
    st.sema_signal_wake(sem, 1);
    assert_eq!(st.take_wakes(), vec![1]);
    // The count was consumed by the wake, so a fresh acquire would block again.
    assert!(!st.sema_try_acquire(sem, 1));
}

#[test]
fn event_flag_parks_then_a_set_wakes_with_the_match_pattern() {
    let mut st = state();
    let ev = st.create_event_flag(0);
    // Thread 1 waits (AND) for bits 0x3: unsatisfied, parks.
    st.set_current(1);
    assert_eq!(st.evf_try_wait(ev, 0x3, 0), None);
    st.evf_block(ev, 0x3, 0, 0x2000, 0);
    assert!(st.take_wakes().is_empty());
    // Setting only bit 0 does not satisfy an AND wait for 0x3.
    st.set_current(2);
    st.event_set_wake(ev, 0x1);
    assert!(st.take_wakes().is_empty());
    // Completing the pattern wakes thread 1 and delivers 0x3 to its outBits.
    st.event_set_wake(ev, 0x2);
    assert_eq!(st.take_wakes(), vec![1]);
    assert_eq!(st.take_stat_writes(), vec![(0x2000, 0x3)]);
}

#[test]
fn event_flag_wait_modes_or_and_clear() {
    let mut st = state();
    let ev = st.create_event_flag(0b110);
    // OR: any requested bit satisfies; no clear op leaves the pattern intact.
    assert_eq!(st.evf_try_wait(ev, 0b010, 1), Some(0b110));
    assert_eq!(st.event_pattern(ev), 0b110);
    // WAITCLEAR_PAT clears only the requested bits at match.
    assert_eq!(st.evf_try_wait(ev, 0b100, 1 | 4), Some(0b110));
    assert_eq!(st.event_pattern(ev), 0b010);
    // WAITCLEAR zeroes the whole pattern at match.
    assert_eq!(st.evf_try_wait(ev, 0b010, 1 | 2), Some(0b010));
    assert_eq!(st.event_pattern(ev), 0);
    // AND on a partial pattern does not match.
    st.event_set(ev, 0b01);
    assert_eq!(st.evf_try_wait(ev, 0b11, 0), None);
}

#[test]
fn timed_event_flag_wait_wakes_at_its_deadline() {
    let mut st = state();
    let ev = st.create_event_flag(0);
    st.set_current(1);
    st.evf_block(ev, 0x1, 0, 0x3000, 500);
    assert_eq!(st.earliest_lwcond_deadline(), Some(500));
    // Short of the deadline: still parked.
    st.advance_time_to(499);
    assert!(st.take_wakes().is_empty());
    // At the deadline: woken with the (still unset) current pattern delivered.
    st.advance_time_to(500);
    assert_eq!(st.take_wakes(), vec![1]);
    assert_eq!(st.take_stat_writes(), vec![(0x3000, 0)]);
    // The expired waiter is gone: a later set wakes no one.
    st.event_set_wake(ev, 0x1);
    assert!(st.take_wakes().is_empty());
}

#[test]
fn multiple_semaphore_waiters_release_fifo() {
    let mut st = state();
    let sem = st.create_sema("test", 0, 0, i32::MAX);
    // Threads 1, 2, 3 park in order, each needing one permit.
    for t in [1, 2, 3] {
        st.set_current(t);
        assert!(!st.sema_try_acquire(sem, 1));
        st.sema_block(sem, 1, 0);
    }
    // Three single posts release them in FIFO (block) order, one permit each.
    st.set_current(4);
    st.sema_signal_wake(sem, 1);
    st.sema_signal_wake(sem, 1);
    st.sema_signal_wake(sem, 1);
    assert_eq!(st.take_wakes(), vec![1, 2, 3]);
}

#[test]
fn semaphore_need_greater_than_one_waits_for_accumulation() {
    let mut st = state();
    let sem = st.create_sema("test", 0, 0, i32::MAX);
    st.set_current(1);
    st.sema_block(sem, 3, 0); // needs 3
    st.set_current(2);
    st.sema_signal_wake(sem, 2); // count 2 < 3: not released
    assert!(st.take_wakes().is_empty());
    st.sema_signal_wake(sem, 1); // count 3: released, consuming all 3
    assert_eq!(st.take_wakes(), vec![1]);
    assert!(
        !st.sema_try_acquire(sem, 1),
        "the wake consumed the whole count"
    );
}

#[test]
fn timed_semaphore_wait_times_out_with_wait_timeout_code() {
    let mut st = state();
    let sem = st.create_sema("test", 0, 0, i32::MAX);
    st.set_current(1);
    assert!(!st.sema_try_acquire(sem, 1));
    st.sema_block(sem, 1, 500);
    // The semaphore deadline participates in the scheduler's clock jump.
    assert_eq!(st.earliest_lwcond_deadline(), Some(500));
    // Short of the deadline: still parked, nothing owed.
    st.advance_time_to(499);
    assert!(st.take_wakes().is_empty());
    assert_eq!(st.take_resume_code(1), None);
    // At the deadline: woken and owed SCE_KERNEL_ERROR_WAIT_TIMEOUT for its r0.
    st.advance_time_to(500);
    assert_eq!(st.take_wakes(), vec![1]);
    assert_eq!(st.take_resume_code(1), Some(SCE_KERNEL_ERROR_WAIT_TIMEOUT));
    assert_eq!(st.take_resume_code(1), None, "the code is drained once");
}

#[test]
fn timed_semaphore_wait_satisfied_before_deadline_returns_zero() {
    let mut st = state();
    let sem = st.create_sema("test", 0, 0, i32::MAX);
    st.set_current(1);
    st.sema_block(sem, 1, 500);
    // A signal before the deadline releases it with NO resume code (r0 stays 0).
    st.set_current(2);
    st.sema_signal_wake(sem, 1);
    assert_eq!(st.take_wakes(), vec![1]);
    assert_eq!(st.take_resume_code(1), None);
    // Its deadline no longer holds the scheduler's clock.
    assert_eq!(st.earliest_lwcond_deadline(), None);
}

#[test]
fn timed_cond_wait_times_out_reacquires_free_mutex_and_returns_code() {
    let mut st = state();
    // The KERNEL MUTEX TABLE lives in guest memory; these tests supply a stub for it.
    let kw = &mut Words::default();
    let m = st.create_mutex(kw, "test", 0, 0);
    let cv = st.create_cond(m);
    // Thread 1 holds the mutex, then waits with a 500 us timeout: the wait releases
    // the mutex, parks, and registers a deadline.
    st.set_current(1);
    assert!(st.mutex_lock(kw, m));
    st.cond_wait(kw, cv, 500);
    assert_eq!(st.earliest_lwcond_deadline(), Some(500));
    assert!(st.take_wakes().is_empty());
    // No signaller comes: at the deadline the wait times out, re-acquires the (now
    // free) mutex, is woken, and is owed WAIT_TIMEOUT.
    //
    // The re-acquisition is DEFERRED, exactly as the lightweight cond's is and for the
    // same reason: it reads the kernel mutex table in guest memory, and `advance_time_to`
    // has none. Asserted rather than skipped - a wake appearing here would mean the
    // handoff had been guessed at instead of read.
    st.advance_time_to(500);
    assert!(
        st.take_wakes().is_empty(),
        "the handoff is deferred, so the wake is too"
    );
    assert_eq!(st.resolve_deferred_lwmutex(kw), 1);
    assert_eq!(st.take_wakes(), vec![1]);
    assert_eq!(st.take_resume_code(1), Some(SCE_KERNEL_ERROR_WAIT_TIMEOUT));
    st.set_current(2);
    assert!(
        st.mutex_contended(kw, m),
        "the timed-out cond wait re-acquired its mutex"
    );
}

#[test]
fn timed_cond_wait_timeout_queues_behind_a_held_mutex() {
    let mut st = state();
    // The KERNEL MUTEX TABLE lives in guest memory; these tests supply a stub for it.
    let kw = &mut Words::default();
    let m = st.create_mutex(kw, "test", 0, 0);
    let cv = st.create_cond(m);
    // Thread 1 waits with a timeout (releasing the mutex as it parks).
    st.set_current(1);
    assert!(st.mutex_lock(kw, m));
    st.cond_wait(kw, cv, 500);
    // Thread 2 grabs the mutex before the deadline.
    st.set_current(2);
    assert!(st.mutex_lock(kw, m));
    // At the deadline the wait times out, but the mutex is held: thread 1 queues
    // behind thread 2 (not woken yet) while its WAIT_TIMEOUT code is already owed.
    st.advance_time_to(500);
    assert!(
        st.take_wakes().is_empty(),
        "the timed-out waiter queues behind the owner"
    );
    // ...and the queueing itself is deferred to where guest memory is reachable, so it
    // has to be settled before the unlock below can find a waiter to hand the mutex to.
    assert_eq!(st.resolve_deferred_lwmutex(kw), 1);
    assert!(
        st.take_wakes().is_empty(),
        "still behind the owner after it is settled"
    );
    // Thread 2 unlocks: thread 1 finally gets the mutex, is woken, and the owed
    // timeout code is delivered when it resumes.
    st.mutex_unlock(kw, m);
    assert_eq!(st.take_wakes(), vec![1]);
    assert_eq!(st.take_resume_code(1), Some(SCE_KERNEL_ERROR_WAIT_TIMEOUT));
}

#[test]
fn timed_lwcond_wait_times_out_with_code_but_a_signal_returns_zero() {
    let mut st = state();
    // A timed lightweight-cond wait that expires is owed WAIT_TIMEOUT. A cond
    // always has a bound mutex (set at CreateLwCond) - bind one so the wait parks.
    let w = &mut Words::default();
    created_lwmutex(&mut st, w, 0x9100);
    st.lwcond_bind_mutex(0x9000, 0x9100);
    st.set_current(1);
    assert!(st.lwcond_wait(w, 0x9000, 250)); // work-area address, 250 us timeout
    assert_eq!(st.earliest_lwcond_deadline(), Some(250));
    st.advance_time_to(250);
    // The expiry owes thread 1 its mutex back, and `advance_time_to` has no guest
    // memory to hand it over in - so nothing is woken until the scheduler settles it.
    // Asserted rather than skipped: a wake that appeared here would mean the handoff
    // had been guessed at instead of read.
    assert!(
        st.take_wakes().is_empty(),
        "the handoff is deferred, so the wake is too"
    );
    assert_eq!(st.resolve_deferred_lwmutex(w), 1);
    assert_eq!(st.take_wakes(), vec![1]);
    assert_eq!(st.take_resume_code(1), Some(SCE_KERNEL_ERROR_WAIT_TIMEOUT));
    // A timed wait released by a signal instead returns 0 (no resume code).
    st.set_current(2);
    assert!(st.lwcond_wait(w, 0x9000, 250));
    st.set_current(3);
    st.lwcond_signal(w, 0x9000, false);
    assert_eq!(st.take_wakes(), vec![2]);
    assert_eq!(st.take_resume_code(2), None);
}

#[test]
fn mutex_hands_ownership_to_the_next_waiter_on_unlock() {
    let mut st = state();
    // The KERNEL MUTEX TABLE lives in guest memory; these tests supply a stub for it.
    let kw = &mut Words::default();
    let m = st.create_mutex(kw, "test", 0, 0);
    // Thread 1 acquires; thread 2 contends and parks.
    st.set_current(1);
    assert!(st.mutex_lock(kw, m));
    st.set_current(2);
    assert!(st.mutex_contended(kw, m));
    assert!(!st.mutex_lock(kw, m));
    assert!(st.take_wakes().is_empty());
    // Thread 1 unlocks: ownership passes to thread 2, which is woken.
    st.set_current(1);
    st.mutex_unlock(kw, m);
    assert_eq!(st.take_wakes(), vec![2]);
    // Thread 2 now owns it, so thread 3 would contend.
    st.set_current(3);
    assert!(st.mutex_contended(kw, m));
}

#[test]
fn lightweight_mutex_blocks_and_hands_over_keyed_by_work_address() {
    let mut st = state();
    let w = &mut Words::default();
    let work = 0x8000; // guest work-area address (no kernel handle)
    created_lwmutex(&mut st, w, work);
    created_lwmutex(&mut st, w, 0x9000);
    // Thread 1 locks the lightweight mutex; thread 2 contends and parks.
    st.set_current(1);
    assert!(st.lwmutex_lock(w, work, None));
    st.set_current(2);
    assert!(st.lwmutex_contended(w, work));
    assert!(
        !st.lwmutex_lock(w, work, None),
        "contender must block, not silently succeed"
    );
    assert!(st.take_wakes().is_empty());
    // ...and the work area SAYS a thread is parked, which is what keeps the inline
    // fast path off a mutex only the host can now release correctly.
    assert_eq!(w.word(work + lwwork::off::WAITERS), 1);
    // Thread 1 unlocks: ownership passes to thread 2, which is woken.
    st.set_current(1);
    st.lwmutex_unlock(w, work);
    assert_eq!(st.take_wakes(), vec![2]);
    assert_eq!(
        w.word(work + lwwork::off::WAITERS),
        0,
        "the queue drained with the handoff"
    );
    assert_eq!(lwwork::owner(w, work), 2, "and thread 2 holds it now");
    // A different work address is an independent lock (thread 3 takes it freely).
    st.set_current(3);
    assert!(st.lwmutex_lock(w, 0x9000, None));
    assert!(
        st.lwmutex_contended(w, work),
        "the first lock is still held by thread 2"
    );
}

#[test]
fn lightweight_mutex_is_recursive_for_the_owner() {
    let mut st = state();
    let w = &mut Words::default();
    let work = 0x8000;
    created_lwmutex(&mut st, w, work);
    st.set_current(1);
    assert!(st.lwmutex_lock(w, work, None)); // count 1
    assert!(st.lwmutex_lock(w, work, None)); // count 2 (recursive, same owner)
    st.lwmutex_unlock(w, work); // count 1, still owned
    st.set_current(2);
    assert!(
        st.lwmutex_contended(w, work),
        "still held after one of two unlocks"
    );
    st.set_current(1);
    st.lwmutex_unlock(w, work); // count 0, released
    st.set_current(2);
    assert!(
        !st.lwmutex_contended(w, work),
        "free after the matching unlock"
    );
}

#[test]
fn lightweight_cond_wait_releases_and_reacquires_its_bound_mutex() {
    let mut st = state();
    let w = &mut Words::default();
    let mutex_work = 0x8000;
    let cond_work = 0x8100;
    created_lwmutex(&mut st, w, mutex_work);
    st.lwcond_bind_mutex(cond_work, mutex_work);
    // Thread 1 holds the lwmutex, then waits on the lwcond: the wait releases the
    // mutex (so a sibling can take it) and parks thread 1.
    st.set_current(1);
    assert!(st.lwmutex_lock(w, mutex_work, None));
    assert!(st.lwcond_wait(w, cond_work, 0));
    assert!(st.take_wakes().is_empty(), "waiter is parked, not runnable");
    st.set_current(2);
    assert!(
        !st.lwmutex_contended(w, mutex_work),
        "the lwmutex was released by the wait"
    );
    // Thread 2 takes the lwmutex, then signals: the waiter must re-acquire the
    // mutex first, so it queues behind thread 2 (not woken yet).
    assert!(st.lwmutex_lock(w, mutex_work, None));
    st.lwcond_signal(w, cond_work, false);
    assert!(
        st.take_wakes().is_empty(),
        "waiter must re-acquire the lwmutex first"
    );
    // Thread 2 unlocks: ownership passes to the waiter, which is finally woken.
    st.lwmutex_unlock(w, mutex_work);
    assert_eq!(st.take_wakes(), vec![1]);
    st.set_current(3);
    assert!(
        st.lwmutex_contended(w, mutex_work),
        "the woken waiter re-acquired the lwmutex"
    );
}

#[test]
fn mutex_is_recursive_for_the_owner() {
    let mut st = state();
    // The KERNEL MUTEX TABLE lives in guest memory; these tests supply a stub for it.
    let kw = &mut Words::default();
    let m = st.create_mutex(kw, "test", 0, 0);
    st.set_current(1);
    assert!(st.mutex_lock(kw, m)); // count 1
    assert!(st.mutex_lock(kw, m)); // count 2 (recursive, same owner)
    st.mutex_unlock(kw, m); // count 1, still owned
    st.set_current(2);
    assert!(
        st.mutex_contended(kw, m),
        "still held after one of two unlocks"
    );
    st.set_current(1);
    st.mutex_unlock(kw, m); // count 0, released
    st.set_current(2);
    assert!(!st.mutex_contended(kw, m), "free after the matching unlock");
}

#[test]
fn cond_wait_releases_the_mutex_and_signal_hands_it_back() {
    let mut st = state();
    // The KERNEL MUTEX TABLE lives in guest memory; these tests supply a stub for it.
    let kw = &mut Words::default();
    let m = st.create_mutex(kw, "test", 0, 0);
    let cv = st.create_cond(m);

    // Thread 1 holds the mutex, then waits on the condition: the wait releases
    // the mutex (so another thread can take it) and parks thread 1.
    st.set_current(1);
    assert!(st.mutex_lock(kw, m));
    st.cond_wait(kw, cv, 0);
    assert!(st.take_wakes().is_empty(), "waiter is parked, not runnable");
    st.set_current(2);
    assert!(!st.mutex_contended(kw, m), "mutex was released by the wait");

    // Thread 2 takes the mutex, then signals: the waiter cannot run yet (thread
    // 2 still owns the mutex), so it is queued behind the owner, not woken.
    assert!(st.mutex_lock(kw, m));
    st.cond_signal(kw, cv, false);
    assert!(
        st.take_wakes().is_empty(),
        "waiter must re-acquire the mutex first"
    );

    // Thread 2 unlocks: ownership passes to the waiter, which is finally woken.
    st.mutex_unlock(kw, m);
    assert_eq!(st.take_wakes(), vec![1]);
    // Thread 1 now owns the mutex again (the wait re-acquired it).
    st.set_current(3);
    assert!(st.mutex_contended(kw, m));
}

#[test]
fn cond_signal_with_free_mutex_wakes_immediately() {
    let mut st = state();
    // The KERNEL MUTEX TABLE lives in guest memory; these tests supply a stub for it.
    let kw = &mut Words::default();
    let m = st.create_mutex(kw, "test", 0, 0);
    let cv = st.create_cond(m);
    st.set_current(1);
    assert!(st.mutex_lock(kw, m));
    st.cond_wait(kw, cv, 0); // releases m, parks thread 1
    // The signaller does not hold the mutex, so the woken waiter takes it now.
    st.set_current(2);
    st.cond_signal(kw, cv, false);
    assert_eq!(st.take_wakes(), vec![1]);
    st.set_current(3);
    assert!(
        st.mutex_contended(kw, m),
        "waiter re-acquired the free mutex"
    );
}

#[test]
fn cond_signal_all_wakes_every_waiter() {
    let mut st = state();
    // The KERNEL MUTEX TABLE lives in guest memory; these tests supply a stub for it.
    let kw = &mut Words::default();
    let m = st.create_mutex(kw, "test", 0, 0);
    let cv = st.create_cond(m);
    // Two threads wait (neither holds the mutex at wait's end - a plain park).
    for t in [1, 2] {
        st.set_current(t);
        st.mutex_lock(kw, m);
        st.cond_wait(kw, cv, 0);
        st.mutex_unlock(kw, m); // no-op: wait already released it
    }
    st.set_current(3);
    st.cond_signal(kw, cv, true);
    // All waiters are released from the condition, but the mutex serializes
    // them: only the first can hold it and wake now; the second queues behind
    // it and is woken when the first unlocks.
    assert_eq!(st.take_wakes(), vec![1]);
    st.set_current(1);
    st.mutex_unlock(kw, m);
    assert_eq!(st.take_wakes(), vec![2]);
}

/// `cpuAffinityMask` 0 is "inherit calling thread affinity mask": a thread the main thread
/// creates with 0 gets main's mask (from the title's `SceProcessParam`), a grandchild
/// created with 0 gets its creator's, and a named mask is kept as named.
#[test]
fn a_zero_affinity_mask_inherits_the_creators() {
    let mut st = state();
    st.set_main_thread_request((Some(0x56), Some(0x10000)));
    assert_eq!(st.main_thread_priority(), 0x56);
    st.set_current(MAIN_THID);
    assert_eq!(
        st.thread_cpu_affinity(0),
        0x10000,
        "main reports its requested mask"
    );
    let child = st.create_thread(0x2000, 0x1000, 0x8e, 0, 0);
    assert_eq!(
        st.thread_cpu_affinity(child),
        0x10000,
        "mask 0 inherits main's pin"
    );
    let named = st.create_thread(0x2000, 0x1000, 0x8e, 0, 0x40000);
    assert_eq!(
        st.thread_cpu_affinity(named),
        0x40000,
        "a named mask is kept"
    );
    st.set_current(named);
    let grandchild = st.create_thread(0x2000, 0x1000, 0x8e, 0, 0);
    assert_eq!(
        st.thread_cpu_affinity(grandchild),
        0x40000,
        "inherits its own creator's"
    );
    let mut plain = state();
    plain.set_current(MAIN_THID);
    let t = plain.create_thread(0x2000, 0x1000, 0x8e, 0, 0);
    assert_eq!(
        plain.thread_cpu_affinity(t),
        crate::vita::threadmgr::CPU_MASK_USER_ALL,
        "no request anywhere: every user core, as before"
    );
}

#[test]
fn join_parks_until_the_target_thread_exits() {
    let mut st = state();
    let worker = st.create_thread(0x2000, 0x1000, DEFAULT_THREAD_PRIORITY, 0, 0);
    // STARTED, because a join on a DORMANT thread completes at once (see `join_block`) -
    // it is the same state a finished thread returns to and there is no run to wait for.
    // The park this test is about is the one on a thread that is actually RUNNING.
    st.start_thread(worker, 0, 0).unwrap();
    // Main joins the not-yet-finished worker and parks, passing a
    // `stat` out-parameter at guest address 0x5000.
    st.set_current(MAIN_THID);
    assert!(!st.join_block(worker, 0x5000));
    assert!(st.take_wakes().is_empty());
    assert!(
        st.take_stat_writes().is_empty(),
        "no exit code to deliver yet"
    );
    // The worker ends: the joiner is woken, the exit code is recorded, AND queued
    // for delivery to the joiner's `stat` pointer (the wait handler cannot write
    // it at wake time).
    st.set_thread_exit(worker, 7);
    assert_eq!(st.take_wakes(), vec![MAIN_THID]);
    assert_eq!(st.thread_exit_code(worker), Some(7));
    assert_eq!(st.take_stat_writes(), vec![(0x5000, 7)]);
    // A join after the fact does not park (and a NULL stat queues no write).
    assert!(st.join_block(worker, 0));
    assert!(st.take_stat_writes().is_empty());
}

/// A restarted thread is RUNNING again: a join parks on the new run instead of returning the
/// old run's exit code, and a start while it runs is refused - see `start_thread`.
#[test]
fn a_restarted_thread_is_running_again_and_cannot_be_started_twice() {
    let mut st = state();
    let worker = st.create_thread(0x2000, 0x1000, DEFAULT_THREAD_PRIORITY, 0, 0);
    st.start_thread(worker, 0, 0).unwrap();
    assert_eq!(
        st.start_thread(worker, 0, 0),
        Err(SCE_KERNEL_ERROR_NOT_DORMANT)
    );
    st.set_thread_exit(worker, 7);
    let _ = st.take_spawns();
    // Restart: the old exit code is gone and a join waits for this run.
    st.start_thread(worker, 0, 0).unwrap();
    assert_eq!(st.thread_exit_code(worker), None);
    assert!(st.thread_running(worker));
    st.set_current(MAIN_THID);
    assert!(
        !st.join_block(worker, 0),
        "a join on the restarted thread parks"
    );
    st.set_thread_exit(worker, 9);
    assert_eq!(st.take_wakes(), vec![MAIN_THID]);
    assert!(!st.thread_running(worker));
}

/// A host call that makes two guest calls runs three times: each replay answers the
/// calls already made from the record, a guest call's OWN host calls get records of their
/// own (they must not consume or clear the outer one), and a finished call leaves none.
#[test]
fn a_host_call_replays_its_guest_calls_from_the_record() {
    let mut st = state();
    st.set_current(MAIN_THID);
    let mem = SliceMemory(&mut []);
    // The "handler": two guest calls, then the sum of their results.
    let handler = |st: &mut VitaState| -> Option<u32> {
        let a = st.call_guest(
            GuestCall {
                entry: 0x100,
                args: vec![1],
            },
            None,
        )?;
        let b = st.call_guest(
            GuestCall {
                entry: 0x200,
                args: vec![a.r0],
            },
            None,
        )?;
        Some(a.r0 + b.r0)
    };
    let dispatch = |st: &mut VitaState| {
        st.guest_call_begin();
        let r = handler(st);
        (r, st.guest_call_end())
    };
    assert_eq!(dispatch(&mut st), (None, true));
    assert_eq!(st.take_guest_call(MAIN_THID).map(|c| c.entry), Some(0x100));
    // The guest call makes a host call of its own, which asks for nothing.
    st.guest_call_begin();
    assert!(!st.guest_call_end());
    st.guest_call_returned(MAIN_THID, 5, 0, &mem, 0);
    assert_eq!(dispatch(&mut st), (None, true));
    assert_eq!(
        st.take_guest_call(MAIN_THID),
        Some(GuestCall {
            entry: 0x200,
            args: vec![5]
        })
    );
    // A nested host call that ITSELF calls back gets its own record and finishes in it.
    let nested = |st: &mut VitaState| {
        st.guest_call_begin();
        let r = st.call_guest(
            GuestCall {
                entry: 0x300,
                args: vec![],
            },
            None,
        );
        (r.map(|r| r.r0), st.guest_call_end())
    };
    assert_eq!(nested(&mut st), (None, true));
    assert_eq!(st.take_guest_call(MAIN_THID).map(|c| c.entry), Some(0x300));
    st.guest_call_returned(MAIN_THID, 99, 0, &mem, 0);
    assert_eq!(nested(&mut st), (Some(99), false));
    // Back to the outer call's second guest call.
    st.guest_call_returned(MAIN_THID, 7, 0, &mem, 0);
    assert_eq!(dispatch(&mut st), (Some(12), false));
    assert!(
        st.guest_calls.is_empty(),
        "a finished call leaves no record"
    );
}

/// Arguments past the fourth go on a fresh, 8-byte-aligned frame below the caller's sp.
#[test]
fn a_guest_call_seeds_registers_and_stacks_the_rest() {
    let mut bytes = vec![0u8; 0x1000];
    let mut regs = [0u32; REG_COUNT];
    regs[SP] = 0x0ffc;
    let call = GuestCall {
        entry: 0x8100_0001,
        args: vec![10, 11, 12, 13, 14],
    };
    call.seed(&mut regs, &mut SliceMemory(&mut bytes), 0);
    assert_eq!(&regs[..4], &[10, 11, 12, 13]);
    assert_eq!(regs[SP] % 8, 0);
    assert!(regs[SP] + 4 <= 0x0ffc - GUEST_CALL_GAP);
    let at = regs[SP] as usize;
    assert_eq!(
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()),
        14
    );
}

#[test]
fn start_thread_queues_a_spawn_not_a_synchronous_reentry() {
    let mut st = state();
    let worker = st.create_thread(0x2000, 0x1000, DEFAULT_THREAD_PRIORITY, 0, 0);
    st.start_thread(worker, 4, 0x1234).unwrap();
    assert!(
        st.take_reentry().is_none(),
        "preemptive start does not re-enter synchronously"
    );
    let spawns = st.take_spawns();
    assert_eq!(spawns.len(), 1);
    assert_eq!(spawns[0].thid, worker);
    assert_eq!(spawns[0].entry, 0x2000);
    assert_eq!(spawns[0].arg_len, 4);
    assert_eq!(spawns[0].arg_ptr, 0x1234);
}