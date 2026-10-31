//! SceFiber's baton bookkeeping: what a fiber runs on, and what leaving it does.
use super::*;
use crate::world::DeterministicWorld;

const ENTRY: u32 = 0x8100_0b47;

fn state() -> VitaState {
    VitaState::new(0x1000, 0x10000, Box::new(DeterministicWorld::default()))
}

fn stack_top_of(st: &VitaState, thid: i32) -> u32 {
    st.threads.iter().find(|t| t.uid == thid).map(|t| t.stack_top).unwrap()
}

/// A fiber initialised with NO context runs - on its runner's stack, just below where the
/// runner stood at `sceFiberRun` - and when it returns to the thread it is abandoned: its
/// backing thread ends, and the next run starts a FRESH one at the entry, on the same base.
#[test]
fn a_context_less_fiber_runs_on_the_runner_stack_and_restarts_at_its_entry() {
    let mut st = state();
    let fiber = 0x8300_0000;
    assert_eq!(st.fiber_initialize(fiber, "q".into(), ENTRY, fiber, 0, 0), 0);
    let runner = st.current;
    assert_eq!(st.fiber_run(fiber, 7, 0, 0x8100_1234), Ok(()));
    let first = st.fibers.thid(0);
    assert!(st.fibers.started(0));
    assert_eq!(stack_top_of(&st, first), 0x8100_1230, "below the runner's sp, aligned");

    st.current = first;
    assert_eq!(st.fiber_return_to_thread(1, 0), Ok(true), "abandoned, not parked");
    let (started, runner_now, thid) = st.fibers.state(0);
    assert!(!started && runner_now == 0 && thid != first, "a fresh, unstarted thread");

    st.current = runner;
    assert_eq!(st.fiber_run(fiber, 8, 0, 0x8100_1234), Ok(()));
    assert_eq!(stack_top_of(&st, st.fibers.thid(0)), 0x8100_1230);
}

/// A fiber WITH a context parks when it returns to its thread (it can be resumed
/// mid-function), and finalizing it while it is suspended succeeds - only a RUNNING fiber
/// is refused - after which its address can be initialised afresh.
#[test]
fn a_suspended_fiber_with_a_context_parks_and_can_be_finalized() {
    let mut st = state();
    let fiber = 0x8300_0100;
    assert_eq!(st.fiber_initialize(fiber, "c".into(), ENTRY, 0, 0x8400_0000, 0x4000), 0);
    assert_eq!(st.fiber_run(fiber, 0, 0, 0x8100_1234), Ok(()));
    let thid = st.fibers.thid(0);
    assert_eq!(stack_top_of(&st, thid), 0x8400_4000, "the top of its own context");
    assert_eq!(st.fiber_finalize(fiber), VitaState::FIBER_ERROR_STATE, "it is running");

    st.current = thid;
    assert_eq!(st.fiber_return_to_thread(0, 0), Ok(false), "parked, resumable");
    assert_eq!(st.fibers.thid(0), thid);
    assert_eq!(st.fiber_finalize(fiber), 0, "a suspended fiber is finalized");
    assert_eq!(st.fibers.terminated(), 1, "its parked thread is ended");
    assert_eq!(st.fiber_initialize(fiber, "c".into(), ENTRY, 0, 0x8400_0000, 0x4000), 0);
}

/// Finalizing a PARKED fiber ends its backing thread: the scheduler is handed the thid to
/// drop (once), the thread table no longer holds it, nothing queued for it survives, and
/// the same `SceFiber` re-initialises and runs on a fresh thread. Re-initialising a parked
/// fiber without a finalize ends its thread the same way. These parked threads used to be
/// left blocked for ever - one per finalized fiber per frame.
#[test]
fn finalizing_a_parked_fiber_terminates_its_thread() {
    let mut st = state();
    let fiber = 0x8300_0400;
    let runner = st.current;
    let base = st.threads.len();
    assert_eq!(st.fiber_initialize(fiber, "p".into(), ENTRY, 0, 0x8400_0000, 0x4000), 0);
    assert_eq!(st.fiber_run(fiber, 0, 0, 0x8100_1234), Ok(()));
    let parked = st.fibers.thid(0);
    assert_eq!(st.take_spawns().len(), 1);
    st.current = parked;
    assert_eq!(st.fiber_return_to_thread(0, 0), Ok(false), "parked, resumable");
    assert_eq!(st.take_wakes(), vec![runner]);
    // A wake for the parked thread that the scheduler has not drained yet must not
    // outlive it.
    st.pending_wakes.push(parked);

    st.current = runner;
    assert_eq!(st.fiber_finalize(fiber), 0);
    assert_eq!(st.take_terminations(), vec![parked], "handed to the scheduler to drop");
    assert!(st.take_terminations().is_empty(), "once");
    assert!(st.take_wakes().is_empty(), "no wake survives for a dropped thread");
    assert!(!st.threads.iter().any(|t| t.uid == parked), "its record is gone");
    assert_eq!(st.threads.len(), base, "nothing else is left behind");
    assert_eq!(st.spawn_near(parked), None);
    assert_eq!(st.fibers.terminated(), 1);

    // The same SceFiber, again: a fresh thread, which runs.
    assert_eq!(st.fiber_initialize(fiber, "p".into(), ENTRY, 0, 0x8400_0000, 0x4000), 0);
    assert_eq!(st.fiber_run(fiber, 0, 0, 0x8100_1234), Ok(()));
    let again = st.fibers.first_live_thid().unwrap();
    assert_ne!(again, parked);
    assert_eq!(st.take_spawns().len(), 1);

    // Re-initialising a parked fiber (no finalize) ends its thread too.
    st.current = again;
    assert_eq!(st.fiber_return_to_thread(0, 0), Ok(false));
    st.current = runner;
    assert_eq!(st.fiber_initialize(fiber, "p".into(), ENTRY, 0, 0x8400_0000, 0x4000), 0);
    assert_eq!(st.take_terminations(), vec![again]);
    assert_eq!(st.fibers.terminated(), 2);
    let live = st.threads.iter().filter(|t| t.exit_code.is_none()).count();
    assert_eq!(live, base + 1, "only the new, unstarted thread");
}

/// A fiber initialised per job and finalized after it - run or not, with a context or
/// without - leaves no thread record behind, and no fiber record but the live one. Each
/// retired record used to keep its thread, one per fiber per frame.
#[test]
fn a_fiber_per_job_leaves_no_thread_records_behind() {
    let mut st = state();
    let fiber = 0x8300_0300;
    let runner = st.current;
    let base = st.threads.len();
    for job in 0..50 {
        assert_eq!(st.fiber_initialize(fiber, "job".into(), ENTRY, job, 0, 0), 0);
        if job % 2 == 0 {
            st.current = runner;
            assert_eq!(st.fiber_run(fiber, 0, 0, 0x8100_1234), Ok(()));
            let thid = st.fibers.first_live_thid().unwrap();
            st.current = thid;
            assert_eq!(st.fiber_return_to_thread(0, 0), Ok(true));
            // The scheduler ends the abandoned thread.
            st.set_thread_exit(thid, 0);
            st.current = runner;
        }
        assert_eq!(st.fiber_finalize(fiber), 0);
    }
    let live = st.threads.iter().filter(|t| t.exit_code.is_none()).count();
    assert_eq!(live, base, "no unstarted thread survives its fiber");
    assert!(st.fibers.len() <= 1, "one finalized record per address at most");
    assert_eq!(st.fibers.terminated(), 0, "no fiber here was parked");
}

/// A fiber's backing thread is instantiated on its runner's TLS block, which the runner
/// has been writing since it started: the instantiation must NOT seed `.tdata` into it
/// again (it reset a worker's scratch-stack pointers to the template's zeros). A plain
/// thread still gets its own block, seeded once.
#[test]
fn a_fiber_shares_its_runners_tls_block_without_reseeding_it() {
    let mut st = state();
    st.set_tls_template((0x8100_2000, 0x18, 0xb8));
    let runner = st.current;
    let (tp, src, len) = st.tls_block_to_instantiate(runner);
    assert!(tp != 0);
    assert_eq!((src, len), (0x8100_2000, 0x18), "a fresh block is seeded");

    let fiber = 0x8300_0200;
    assert_eq!(st.fiber_initialize(fiber, "j".into(), ENTRY, 0, 0x8400_0000, 0x4000), 0);
    assert_eq!(st.fiber_run(fiber, 0, 0, 0x8100_1234), Ok(()));
    let backing = st.fibers.thid(0);
    assert_eq!(
        st.tls_block_to_instantiate(backing),
        (tp, 0x8100_2000, 0),
        "the runner's live block, not re-seeded"
    );
    assert_eq!(st.tls_block_to_instantiate(runner).2, 0, "seeded at most once");

    let other = runner + 0x100;
    let (tp2, _, len2) = st.tls_block_to_instantiate(other);
    assert!(tp2 != 0 && tp2 != tp, "a plain thread has its own block");
    assert_eq!(len2, 0x18);
}
