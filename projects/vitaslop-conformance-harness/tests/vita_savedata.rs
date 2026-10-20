//! AppUtil save slots written by a data save, end to end: a real Vita velf saves a file
//! through `sceAppUtilSaveDataDataSave` with and without a slot param and reads the slot back
//! with `sceAppUtilSaveDataSlotGetParam` - a save with a param creates or updates its slot, one
//! without writes only the file - then reads the file back through the savedata mount.
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_savedata

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, RunReport, ThreadedScheduler, VitaEnv};

const SAVEDATA: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/savedata-src/savedata.velf");

/// See savedata.c: `n` = SLOT_NOT_FOUND, `s<ret>` = a data save, `p<userParam>` = the slot
/// read back, `r<bytes>` = the file read back.
const EXPECTED: &str = "i0 n s0 n s0 p42 s0 p43 r3";

#[test]
fn a_data_save_with_a_param_writes_its_slot() {
    let m = loader::load(SAVEDATA).expect("load savedata.velf");
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
