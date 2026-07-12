//! End-to-end CPU-core coverage: 64-bit widening multiply (UMULL/SMULL), count-
//! leading-zeros (CLZ), 64-bit add/sub, and shifts/rotate, each result printed
//! and asserted. This is the demand-driven transpiler-growth probe - it exists to
//! surface and certify the integer instructions real code emits. Run with:
//!   cargo test -p vitaslop-conformance-harness --test vita_compute

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, HostAbi, VitaEnv, Vm};

const COMPUTE: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/compute-src/compute.velf");

const EXPECTED: &str = "\
umull: 18446744065119617025
smull: -3000000
clz: 31 15 0
wide: sum=8589934593 dif=8589934588
shift: shl=591751040 shr=19088743 ror8=0x78123456
";

#[test]
fn compute_results() {
    let m = loader::load(COMPUTE).expect("load compute.velf");
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
    .expect("instantiate compute");
    vm.set_import_env(Box::new(env.clone()));

    vm.call(m.entry & !1).expect("run compute main");

    let env = env.borrow();
    let cap = &env.state.capture;
    let output = String::from_utf8_lossy(&cap.stdout);
    eprintln!("---output---\n{output}------------");

    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert_eq!(output, EXPECTED);
}
