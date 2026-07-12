//! End-to-end coverage for the SceIoFilemgr file-IO host module. Drives
//! sceIoWrite-to-stdout, sceIoOpen/Read/Getstat on a preloaded asset, and
//! create/write then reopen/lseek/read-back on a guest-produced file, asserting
//! the printed transcript plus the bytes the guest actually wrote. Run with:
//!   cargo test -p vitaslop-conformance-harness --test vita_io

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, HostAbi, VitaEnv, Vm};

const IO: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/io-src/io.velf");

/// The deterministic transcript io.c prints (interleaving raw fd-1 writes and
/// sceClibPrintf, both of which land in the captured stdout in call order).
const EXPECTED: &str = "\
hello io
open asset: ok=1
read: n=10 sum=55
getstat: ret=0 size=10
open write: ok=1
seek: pos=4 tail=[slop]
missing: failed=1
";

#[test]
fn io_calls_produce_correct_results() {
    let m = loader::load(IO).expect("load io.velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    let mut env = VitaEnv::new(
        imports,
        inputs.base,
        inputs.mem_bytes,
        Box::new(DeterministicWorld::default()),
    );
    // Preload the read-only asset the guest opens: bytes 1..=10 (sum 55).
    env.state.add_file("app0:/asset.bin", (1u8..=10).collect());
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
    .expect("instantiate io");
    vm.set_import_env(Box::new(env.clone()));

    vm.call(m.entry & !1).expect("run io main");

    let env = env.borrow();
    let cap = &env.state.capture;
    let output = String::from_utf8_lossy(&cap.stdout);
    eprintln!("---output---\n{output}------------");

    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert_eq!(output, EXPECTED);

    // The guest created and wrote this file through the IO host module.
    assert_eq!(env.state.file_bytes("ux0:/data/out.bin"), Some(&b"vitaslop"[..]));
}
