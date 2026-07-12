//! End-to-end NEON auto-vectorization coverage: gcc -O2 vectorizes array
//! reductions into NEON data-processing (vmovl / vaddw / vadd.i / vpadd / vabdl /
//! vabal / vpadal), which the transpiler lifts to wasm 128-bit SIMD. Each array is
//! >= 16 elements so the vector body actually runs; the printed sums prove the
//! lift is numerically correct. Run with:
//!   cargo test -p vitaslop-conformance-harness --test vita_neon

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, HostAbi, VitaEnv, Vm};

const NEON: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/neon-src/neon.velf");

// bsum = 1+..+32 = 528; ssum = sum(-16..15) = -16; isum = 7*496 - 3200 = 272;
// sad = sum|2i-5|, i=0..31 = 850.
const EXPECTED: &str = "\
bsum=528
ssum=-16
isum=272
sad=850
";

#[test]
fn neon_results() {
    let m = loader::load(NEON).expect("load neon.velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    let env = VitaEnv::new(
        imports,
        inputs.base,
        inputs.mem_bytes,
        Box::new(DeterministicWorld::default()),
    );
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
    .expect("instantiate neon");
    vm.set_import_env(Box::new(env.clone()));

    vm.call(m.entry & !1).expect("run neon main");

    let env = env.borrow();
    let cap = &env.state.capture;
    let output = String::from_utf8_lossy(&cap.stdout);
    eprintln!("---output---\n{output}------------");

    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert_eq!(output, EXPECTED);
}
