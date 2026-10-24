//! Kernel odds and ends, end to end: a real Vita velf checks the two thread-priority spellings
//! (0 vs the previous priority, the caller's-own and relative forms, ILLEGAL_PRIORITY),
//! `sceKernelPollSema`, the default affinity mask, and the libc time helpers' full-year
//! `struct tm` with `sceKernelLibcMktime`'s normalisation.
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_kernel2

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, RunReport, ThreadedScheduler, VitaEnv};

const KERNEL2: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/kernel2-src/kernel2.velf");

/// See kernel2.c.
const EXPECTED: &str = "prio 160 0 100 100 165 165 80028023 80028023 165\n\
sema 0 80028102 0 0 8002000f 80028101\n\
affinity 70000\n\
gm 2012-1-29 13:45:30 w3 y59 d0\n\
local 2012-1-29 13:45:30 w3 y59 d0\n\
mk 0 1328140801 1328140801 2012-1-2 0:0:1 w4 y32 d0";

#[test]
fn kernel_odds_and_ends_answer_as_the_kernel_does() {
    let m = loader::load(KERNEL2).expect("load kernel2.velf");
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
