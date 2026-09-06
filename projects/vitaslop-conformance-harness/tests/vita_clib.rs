//! End-to-end coverage for the SceLibKernel clib memory/string host calls. Where
//! vita_hello exercises the variadic printf path, this drives sceClibMemcpy,
//! Memset, Memcmp, Strnlen, Strncpy, Strcmp, Strncmp and Snprintf on real guest
//! memory and asserts the observable results (printed through the already-trusted
//! sceClibPrintf). Run with:
//!   cargo test -p vitaslop-conformance-harness --test vita_clib

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, HostAbi, VitaEnv, Vm};

const CLIB: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/clib-src/clib.velf");

/// The deterministic transcript clib.c prints. Each line is the observable result
/// of one clib host call on real guest memory.
const EXPECTED: &str = "\
memset: AAAAAAAA
memcpy: copied text
memcmp: neg=-1 eq=0
strnlen: 5 3
strncpy: [hi] pad=0
strcmp: 0 -1
strncmp: -1
snprintf: [n=7 s=ok] ret=8
snprintf trunc: [123] ret=4
";

#[test]
fn clib_calls_produce_correct_results() {
    let m = loader::load(CLIB).expect("load clib.velf");
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
    .expect("instantiate clib");
    vm.set_import_env(Box::new(env.clone()));

    vm.call(m.entry & !1).expect("run clib main");

    let env = env.borrow();
    let cap = &env.state.capture;
    let output = String::from_utf8_lossy(&cap.stdout);
    eprintln!("---output---\n{output}------------");

    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert_eq!(output, EXPECTED);
}
