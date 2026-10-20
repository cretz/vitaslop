//! GXM texture control words, end to end: a real Vita velf checks that a new texture has no
//! LOD bias, that a `LINEAR_STRIDED` texture's stride survives a by-value copy of the struct,
//! that two strided views of one surface keep their own pitches, and that an unaligned stride
//! is refused.
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_gxmtex

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, RunReport, ThreadedScheduler, VitaEnv};

const GXMTEX: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/gxmtex-src/gxmtex.velf");

/// See gxmtex.c: `l`/`t` = init results, `b` = LOD bias read back, the bare numbers are
/// strides, `e` = the refused init's error.
const EXPECTED: &str = "l0 b31 b40 t0 3968 3968 t0 3840 3968 e805b0005";

#[test]
fn texture_state_lives_in_the_control_words() {
    let m = loader::load(GXMTEX).expect("load gxmtex.velf");
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
