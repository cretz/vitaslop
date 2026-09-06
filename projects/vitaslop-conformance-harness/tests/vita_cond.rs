//! End-to-end preemptive condition variables: a real Vita velf whose worker locks
//! a mutex and blocks in sceKernelWaitCond, released only when main signals the
//! condition (which hands the mutex back to the woken worker). Run on the native
//! `ThreadedScheduler`. The captured order proves the wait genuinely released the
//! mutex and blocked, rather than the single-thread bring-up's run-to-completion.
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_cond

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, RunReport, ThreadedScheduler, VitaEnv};

const COND: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/cond-src/cond.velf");

/// "BAM": the worker parks in sceKernelWaitCond (mutex released); main runs, prints
/// 'B', signals; the worker re-acquires the mutex, wakes, and prints 'A'; main
/// (parked joining) then prints 'M'. A synchronous run-to-completion would print
/// 'A' first (the worker's wait could not block), giving "ABM" - so "BAM" is only
/// reachable if the wait truly released the mutex and a sibling's signal woke it.
const EXPECTED: &str = "BAM";

#[test]
fn preemptive_condition_variable_handshake() {
    let m = loader::load(COND).expect("load cond.velf");
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
        &inputs.entries, // main; the worker is discovered as a code pointer
        &inputs.externs,
        inputs.mem_bytes,
        env,
    )
    .expect("build threaded scheduler");

    let report = sched.run();
    assert_eq!(report, RunReport::Finished(0), "process should exit cleanly");

    let host = sched.host();
    let cap = &host.state.capture;
    let output = String::from_utf8_lossy(&cap.stdout);
    eprintln!("---output---\n{output}\n------------");
    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert_eq!(output, EXPECTED);
}
