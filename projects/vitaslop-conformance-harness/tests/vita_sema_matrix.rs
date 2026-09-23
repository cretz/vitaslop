//! End-to-end preemptive SEMAPHORE coverage across threads on the native
//! `ThreadedScheduler`, over real Vita velfs (see `vitaslop-conformance-suite-vita`).
//!
//! Each case is a clean-room multi-threaded C program whose captured stdout is a
//! deterministic function of the strict-priority/round-robin scheduler, so the exact
//! output pins the semantic:
//!   - `semafifo`    - three parked waiters released FIFO by three single posts.
//!   - `semacount`   - counting accumulation: a need-3 wait is not released by a
//!                     partial post, and the leftover count satisfies a later wait.
//!   - `sematimeout` - a timed wait times out with SCE_KERNEL_ERROR_WAIT_TIMEOUT,
//!                     while satisfied timed/untimed waits return 0.
//!   - `semapoll`    - a ZERO timeout is a POLL, not a park: an unsatisfied one answers
//!                     WAIT_TIMEOUT at once and a satisfied one returns 0, for both the
//!                     semaphore and the event flag. NOT covered by `sematimeout`, which
//!                     passes whether a zero timeout polls or parks for ever.
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_sema_matrix

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, RunReport, ThreadedScheduler, VitaEnv};

const SEMAFIFO: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/semafifo-src/semafifo.velf");
const SEMACOUNT: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/semacount-src/semacount.velf");
const SEMATIMEOUT: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/sematimeout-src/sematimeout.velf");
const SEMAPOLL: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/semapoll-src/semapoll.velf");

/// Load `velf`, run it to completion on the preemptive scheduler, and return the
/// verdict plus the captured stdout. Panics on any unimplemented NID so a missing
/// host call is a loud failure rather than a silently wrong string.
fn run(velf: &[u8]) -> (RunReport, String) {
    let m = loader::load(velf).expect("load velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    let mut env = VitaEnv::new(
        imports,
        inputs.base,
        inputs.mem_bytes,
        Box::new(DeterministicWorld::default()),
    );
    env.state.set_preemptive(true);

    let mut sched = ThreadedScheduler::new(
        &inputs.code,
        inputs.base,
        inputs.thumb_entry,
        &inputs.entries,
        &inputs.externs,
        inputs.mem_bytes,
        env,
    )
    .expect("build threaded scheduler");

    let report = sched.run();
    let host = sched.host();
    let cap = &host.state.capture;
    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    let output = String::from_utf8_lossy(&cap.stdout).into_owned();
    (report, output)
}

#[test]
fn semaphore_multi_waiter_fifo_release() {
    let (report, output) = run(SEMAFIFO);
    assert_eq!(report, RunReport::Finished(0), "process should exit cleanly");
    // 'p' (producer), then the three waiters released in FIFO order, then main.
    assert_eq!(output, "p123M");
}

#[test]
fn semaphore_counting_accumulation() {
    let (report, output) = run(SEMACOUNT);
    assert_eq!(report, RunReport::Finished(0), "process should exit cleanly");
    // 'g' (giver); the need-3 wait releases only after 2+1 accumulate ('X'), and the
    // leftover 2 satisfies the need-2 wait ('Y'); then main.
    assert_eq!(output, "gXYM");
}

#[test]
fn semaphore_timed_wait_times_out_and_returns_code() {
    let (report, output) = run(SEMATIMEOUT);
    assert_eq!(report, RunReport::Finished(0), "process should exit cleanly");
    // 'T' = the unsatisfied timed wait returned SCE_KERNEL_ERROR_WAIT_TIMEOUT; 'S' =
    // a satisfied timed wait returned 0; 'i' = a satisfied untimed wait returned 0.
    assert_eq!(output, "TSiM");
}

/// >>> A ZERO TIMEOUT IS A POLL, NOT A PARK - and this is the case the suite was missing.
///
/// `sematimeout` above proves a NON-ZERO timeout expires and that satisfied timed/untimed
/// waits return 0. Every one of those passes whether a zero timeout polls or parks for ever,
/// because none of them ever passes a pointer to the value zero. A NULL `pTimeout` and a
/// pointer to 0 are different calls; this engine read them as the same one and parked a
/// retail title's asset-load thread permanently on a semaphore it had merely POLLED, and the
/// whole suite stayed green through it.
///
/// The guest is single-threaded on purpose: a poll must not park, so if one does, the only
/// thread is parked and the verdict is not `Finished` - a deadlock rather than a wrong
/// string. So this asserts BOTH, and the report assertion is the one that catches a
/// regression to the old behaviour.
#[test]
fn a_zero_timeout_is_a_poll_and_never_parks() {
    let (report, output) = run(SEMAPOLL);
    assert_eq!(report, RunReport::Finished(0), "a poll must not park - this is the deadlock check");
    // 'T'/'U' = the unsatisfied semaphore/event-flag POLLS returned WAIT_TIMEOUT at once;
    // 'S'/'V' = the same polls, now satisfied, returned 0 (a poll is not a refusal); 'M' =
    // main reached the end.
    assert_eq!(output, "TUSVM");
}
