//! End-to-end preemptive EVENT-FLAG rendezvous on the native `ThreadedScheduler`,
//! over a real Vita velf (see `vitaslop-conformance-suite-vita/evfjoin-src`).
//!
//! `evfjoin` has one thread wait on an event flag for all three worker bits under
//! WAITAND (a barrier): only the third set completes the pattern and releases the
//! waiter, which reads 0x7 back through outBits. The exact output "abcD" proves the
//! AND semantics (no early release on a partial pattern) and that a set from one
//! thread releases a waiter parked by another.
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_evf

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, RunReport, ThreadedScheduler, VitaEnv};

const EVFJOIN: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/evfjoin-src/evfjoin.velf");

#[test]
fn event_flag_waitand_barrier() {
    let m = loader::load(EVFJOIN).expect("load evfjoin.velf");
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
    assert_eq!(report, RunReport::Finished(0), "process should exit cleanly");

    let host = sched.host();
    let cap = &host.state.capture;
    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    let output = String::from_utf8_lossy(&cap.stdout);
    assert_eq!(output, "abcD");
}
