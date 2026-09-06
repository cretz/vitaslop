//! Whole-cube transpile milestone: discover and lower the entire transitive
//! call closure from the cube's real entry point (SceModuleInfo::module_start),
//! including `main()` and all its VFP/NEON leaf math. `Vm::new` transpiles,
//! validates, and instantiates the whole module, so a clean construction proves
//! the transpiler covers every instruction the cube reaches - the gate for GXM
//! capture.

use vitaslop_loader as loader;
use vitaslop_native::{HostAbi, Vm};

const CUBE: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.velf");

#[test]
fn cube_transpiles_and_instantiates_from_entry() {
    let m = loader::load(CUBE).expect("load cube.velf");
    let inputs = m.program_inputs();

    let vm = Vm::new(
        &inputs.code,
        inputs.base,
        inputs.thumb_entry,
        &[m.entry & !1],
        &inputs.externs,
        inputs.mem_bytes,
        &HostAbi::default(),
    );

    match vm {
        Ok(_) => {}
        Err(e) => panic!("whole-cube transpile/instantiate failed: {e:?}"),
    }
}
