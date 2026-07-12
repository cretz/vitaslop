//! End-to-end preemptive multithreading: a real Vita velf whose worker BLOCKS in
//! sceKernelWaitSema until a sibling thread signals it, run on the native
//! `ThreadedScheduler` (each guest thread its own instance over one shared linear
//! memory). The captured order proves genuine blocking and cross-thread wakeups
//! rather than the single-thread bring-up's synchronous run-to-completion.
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_mt

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, RunReport, ThreadedScheduler, VitaEnv};

const MT: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/mt-src/mt.velf");

/// "BAM": the signaller prints 'B' and posts the semaphore; the waiter (parked in
/// sceKernelWaitSema) then wakes and prints 'A'; main (parked joining both) then
/// prints 'M'. A synchronous run-to-completion would instead print 'A' first (the
/// waiter's empty-semaphore wait could not block), giving "ABM" - so "BAM" is only
/// reachable if the wait truly blocked and a sibling thread woke it.
const EXPECTED: &str = "BAM";

#[test]
fn preemptive_semaphore_handshake() {
    let m = loader::load(MT).expect("load mt.velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    let mut env =
        VitaEnv::new(imports, inputs.base, inputs.mem_bytes, Box::new(DeterministicWorld::default()));
    // Turn on real blocking: waits park the calling thread and are woken by a
    // sibling's signal/unlock/thread-end instead of succeeding uncontended.
    env.state.set_preemptive(true);

    let mut sched = ThreadedScheduler::new(
        &inputs.code,
        inputs.base,
        inputs.thumb_entry,
        &inputs.entries, // main; the two workers are discovered as code pointers
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
