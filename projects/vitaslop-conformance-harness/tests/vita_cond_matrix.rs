//! End-to-end preemptive CONDITION-VARIABLE coverage across threads on the native
//! `ThreadedScheduler`, over real Vita velfs (see `vitaslop-conformance-suite-vita`).
//!
//! Each case is a clean-room multi-threaded C program whose captured stdout is a
//! deterministic function of the strict-priority/round-robin scheduler:
//!   - `condall`  - sceKernelSignalCondAll wakes EVERY parked waiter, each
//!                  re-acquiring the shared mutex before it runs.
//!   - `condlost` - a signal delivered with no waiter parked is LOST (a condition
//!                  variable has no memory); a later signal releases the waiter.
//!   - `prodcons` - a bounded-buffer producer/consumer handshake over a mutex and
//!                  two condition variables, proving cross-thread wait/signal with no
//!                  lost wakeups.
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_cond_matrix

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, RunReport, ThreadedScheduler, VitaEnv};

const CONDALL: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/condall-src/condall.velf");
const CONDLOST: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/condlost-src/condlost.velf");
const PRODCONS: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/prodcons-src/prodcons.velf");

/// Load `velf`, run it to completion on the preemptive scheduler, and return the
/// verdict plus the captured stdout. Panics on any unimplemented NID.
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
fn condition_variable_broadcast_wakes_all() {
    let (report, output) = run(CONDALL);
    assert_eq!(report, RunReport::Finished(0), "process should exit cleanly");
    // 'B' (broadcaster), then all three waiters woken in mutex-handoff order, main.
    assert_eq!(output, "B123M");
}

#[test]
fn condition_variable_signal_with_no_waiter_is_lost() {
    let (report, output) = run(CONDLOST);
    assert_eq!(report, RunReport::Finished(0), "process should exit cleanly");
    // 'L' (poster's early signal, lost), 'r' (releaser's later signal wakes the
    // waiter), 'w' (waiter), 'M'. The waiter waking on 'r' - not on the earlier
    // lost signal - is the proof.
    assert_eq!(output, "LrwM");
}

#[test]
fn bounded_buffer_producer_consumer() {
    let (report, output) = run(PRODCONS);
    assert_eq!(report, RunReport::Finished(0), "process should exit cleanly");
    // Strictly alternating single-slot handshake feeds 1,2,3 in order, then main.
    assert_eq!(output, "123M");
}
