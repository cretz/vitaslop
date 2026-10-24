//! Service odds and ends, end to end: a real Vita velf checks the controller sampling mode
//! (the previous mode answered, an out-of-range one refused), the console model, and SceAppUtil
//! safe memory (byte counts, zeroes where never written, neighbours untouched, ranges refused).
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_svc2

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, RunReport, ThreadedScheduler, VitaEnv};

const SVC2: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/svc2-src/svc2.velf");

/// See svc2.c.
const EXPECTED: &str = "ctrl 0 0 0 2 80340001 1\n\
model 10000 0\n\
safe 16 4 16 0000a0a1a2a311111111a8a9aaabacad 80100600 80100600 16";

#[test]
fn service_odds_and_ends_answer_as_the_console_does() {
    let m = loader::load(SVC2).expect("load svc2.velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    let mut env =
        VitaEnv::new(imports, inputs.base, inputs.mem_bytes, Box::new(DeterministicWorld::default()));
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
    let output = String::from_utf8_lossy(&cap.stdout);
    eprintln!("---output---\n{output}\n------------");
    assert_eq!(report, RunReport::Finished(0), "process should exit cleanly");
    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert_eq!(output, EXPECTED);
}
