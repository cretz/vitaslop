//! End-to-end CPU-core coverage, part 2: byte reverse (REV/REV16), sign/zero
//! extension (SXTB/UXTB/SXTH), multiply-accumulate (MLA), and bitfield extract
//! (UBFX) - the bit/byte-manipulation instructions ordinary C emits constantly.
//! Demand-driven transpiler-growth probe. Run with:
//!   cargo test -p vitaslop-conformance-harness --test vita_compute2

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, HostAbi, VitaEnv, Vm};

const COMPUTE2: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/compute2-src/compute2.velf");

const EXPECTED: &str = "\
rev: 0x44332211
rev16: 0xcdab
sxtb: -101
uxtb: 155
sxth: -101
mla: 142
ubfx: 43
";

#[test]
fn compute2_results() {
    let m = loader::load(COMPUTE2).expect("load compute2.velf");
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
    .expect("instantiate compute2");
    vm.set_import_env(Box::new(env.clone()));

    vm.call(m.entry & !1).expect("run compute2 main");

    let env = env.borrow();
    let cap = &env.state.capture;
    let output = String::from_utf8_lossy(&cap.stdout);
    eprintln!("---output---\n{output}------------");

    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert_eq!(output, EXPECTED);
}
