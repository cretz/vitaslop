//! Thread EXIT event handlers end to end: a real Vita velf registers a handler for every
//! thread of the process and checks it runs ON each ending thread - one that returns from
//! its entry and one that calls `sceKernelExitThread` - with the exit code, before the
//! thread counts as ended, and not after it is unregistered. Run on the native
//! `ThreadedScheduler`, the engine whose fibers deliver the returning thread's handlers.
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_tevent

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, RunReport, ThreadedScheduler, VitaEnv};

const TEVENT: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/tevent-src/tevent.velf");

/// See tevent.c: `a`/`b`/`c` from the three workers, `x<code>s` from the handler on the
/// worker it was called for (the restarted one twice), `M4` = main saw the handler called four times.
const EXPECTED: &str = "ax5sbx7sax5sax5scM4";

#[test]
fn exit_handlers_run_on_the_ending_thread() {
    let m = loader::load(TEVENT).expect("load tevent.velf");
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
