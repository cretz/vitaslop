//! End-to-end: load the hello velf, transpile+execute its CPU code through the
//! real Vita host, and assert the captured debug-console output. This is the
//! blob-free "it printed" signal - one rung up from the arm/hello svc case: the
//! guest reaches the host through the Vita's real NID import mechanism, and the
//! print goes through a VARIADIC host call (sceClibPrintf) whose argument walk
//! and formatter are exercised here against real arm-vita-eabi-gcc output.
//!
//! The guest exits cleanly via sceKernelExitProcess (which halts the run), so -
//! unlike the cube - no input scripting or terminate hack is needed. Run with:
//!   cargo test -p vitaslop-conformance-harness --test vita_hello

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, HostAbi, VitaEnv, Vm};

const HELLO: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/hello-src/hello.velf");

/// Exactly what hello.c's six sceClibPrintf calls produce. This is the golden:
/// standard C printf semantics for every conversion the program uses.
const EXPECTED: &str = "\
Hello, world
int=-42 uint=42 hex=beef HEX=BEEF oct=100
char=! str=vitaslop pct=%
width=[   42] zero=[00042] left=[42   ] plus=[+42]
ptr=0x81000000 six=1,2,3,4,5,6
float=1.500000 half=0.250000 neg=-3.500000
";

#[test]
fn hello_runs_and_prints() {
    let m = loader::load(HELLO).expect("load hello.velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    // A plain deterministic world: hello observes no external input.
    let world = Box::new(DeterministicWorld::default());
    let env = VitaEnv::new(imports, inputs.base, inputs.mem_bytes, world);
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
    .expect("instantiate hello");
    vm.set_import_env(Box::new(env.clone()));

    // sceKernelExitProcess halts the run; a clean halt is Ok.
    vm.call(m.entry & !1).expect("run hello main");

    let env = env.borrow();
    let cap = &env.state.capture;

    let output = String::from_utf8_lossy(&cap.stdout);
    eprintln!(
        "calls={} bytes={} unimplemented={:?}\n---output---\n{}------------",
        cap.call_count,
        cap.stdout.len(),
        cap.unimplemented,
        output
    );

    // Every imported NID hello used is implemented.
    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);

    // The formatted output matches standard C printf byte for byte. This is the
    // real oracle for the AAPCS variadic argument walk (core registers then
    // stack, doubles promoted and 8-byte aligned).
    assert_eq!(output, EXPECTED);
}
