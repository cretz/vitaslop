//! GXM depth-stencil surfaces, end to end: a real Vita velf checks init's stride checks, what a
//! fresh surface reads back through every getter, that each setter is read back on the original
//! and on a by-value copy, and what a disabled surface reports.
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_gxmds

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, RunReport, ThreadedScheduler, VitaEnv};

const GXMDS: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/gxmds-src/gxmds.velf");

/// See gxmds.c: `f` format, `st` stride in samples, `d` background depth x1000, `s` background
/// stencil, `m` background mask, `l`/`w` force load/store, `e` enabled.
const EXPECTED: &str = "init 805b0003 805b0003 0\n\
fresh f1266000 st960 d1000 s0 m1 l0 w0 e1\n\
set f1266000 st960 d250 s90 m0 l2 w4 e1\n\
copy f1266000 st960 d250 s90 m0 l2 w4 e1\n\
linear fcc000 st32 d1000 s0 m1 l0 w0 e1\n\
off f0 st32 d1000 s0 m1 l0 w0 e0";

#[test]
fn depth_stencil_state_lives_in_the_surface() {
    let m = loader::load(GXMDS).expect("load gxmds.velf");
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
