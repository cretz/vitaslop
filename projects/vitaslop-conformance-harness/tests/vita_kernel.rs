//! End-to-end coverage for the SceLibKernel/SceThreadmgr synchronization and
//! timing primitives: a mutex, a semaphore, an event flag (set a pattern, wait,
//! read it back), and the wide system clock. Each call's observable result is
//! printed and asserted. Run with:
//!   cargo test -p vitaslop-conformance-harness --test vita_kernel

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, HostAbi, VitaEnv, Vm};

const KERNEL: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/kernel-src/kernel.velf");

/// The deterministic transcript kernel.c prints. Lock/wait succeed (return 0),
/// handles are valid, the event-flag pattern round-trips (set 0x5, read 0x5), and
/// the clock is monotonic.
const EXPECTED: &str = "\
mutex: id_ok=1 lock=0 unlock=0
sema: id_ok=1 wait=0 signal=0
eventflag: id_ok=1 pattern=0x5
time: monotonic=1
";

#[test]
fn kernel_sync_and_time() {
    let m = loader::load(KERNEL).expect("load kernel.velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    let env = VitaEnv::new(imports, inputs.base, inputs.mem_bytes, Box::new(DeterministicWorld::default()));
    let env = Rc::new(RefCell::new(env));

    let mut vm = Vm::new(
        &inputs.code,
        inputs.base,
        inputs.thumb_entry,
        &inputs.entries,
        &inputs.externs,
        inputs.mem_bytes,
        &HostAbi::default(),
    )
    .expect("instantiate kernel");
    vm.set_import_env(Box::new(env.clone()));

    vm.call(m.entry & !1).expect("run kernel main");

    let env = env.borrow();
    let cap = &env.state.capture;
    let output = String::from_utf8_lossy(&cap.stdout);
    eprintln!("---output---\n{output}------------");

    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert_eq!(output, EXPECTED);
}
