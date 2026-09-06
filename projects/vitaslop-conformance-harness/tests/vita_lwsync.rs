//! End-to-end preemptive LIGHTWEIGHT-MUTEX coverage on the native `ThreadedScheduler`,
//! over a real Vita velf (see `vitaslop-conformance-suite-vita/lwmutex-src`).
//!
//! A lightweight mutex keeps its state in the caller's work area (no kernel handle),
//! but must still block a contender and enforce mutual exclusion. `lwmutex` holds the
//! lock across a block in one thread while another contends, so the captured order
//! "AaBM" is reachable only if the contender genuinely blocked and ownership was
//! handed over on unlock - an "always succeeds" stub instead yields "ABaM".
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_lwsync

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, RunReport, ThreadedScheduler, VitaEnv};

const LWMUTEX: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/lwmutex-src/lwmutex.velf");
const LWCOND: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/lwcond-src/lwcond.velf");
const LWCONDCOPY: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/lwcondcopy-src/lwcondcopy.velf");

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
fn lightweight_mutex_blocks_a_contender_and_hands_over_on_unlock() {
    let (report, output) = run(LWMUTEX);
    assert_eq!(report, RunReport::Finished(0), "process should exit cleanly");
    assert_eq!(output, "AaBM");
}

#[test]
fn lightweight_cond_wait_releases_and_reacquires_its_mutex() {
    let (report, output) = run(LWCOND);
    assert_eq!(report, RunReport::Finished(0), "process should exit cleanly");
    // "BAM": the waiter parked in WaitLwCond (releasing the lwmutex), the signaller
    // ran and signalled, and the waiter re-acquired the lwmutex to finish.
    assert_eq!(output, "BAM");
}

#[test]
fn lightweight_cond_waited_on_a_copy_of_its_work_area_resolves_to_the_same_cond() {
    // The waiter waits on a BYTE COPY of the cond work area (as a C++ condvar wrapper
    // that stages its embedded LwCondWork on the stack does) while the signaller
    // signals the ORIGINAL. The identity stored in the work area must resolve the copy
    // to the same cond: the copy-wait releases the lwmutex and parks, and the signal on
    // the original wakes it. "BAM" is only reachable if that resolution works - a retail
    // 3D title's deadlock was this exact pattern failing (copy read as an unknown cond).
    let (report, output) = run(LWCONDCOPY);
    assert_eq!(report, RunReport::Finished(0), "process should exit cleanly");
    assert_eq!(output, "BAM");
}
