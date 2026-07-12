//! End-to-end: a program that creates a thread, starts it with an argument,
//! waits for it, and reads its return value. This proves two hard mechanisms at
//! once:
//!   - code-pointer discovery: `worker` is address-taken (passed to
//!     sceKernelCreateThread), never directly called, so the transpiler must find
//!     it via the movw/movt materialization, not the direct-call closure, and
//!   - guest re-entry: the host runs the worker's own guest code (its printf
//!     appears, and its return value flows back through sceKernelWaitThreadEnd).
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_thread

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, HostAbi, VitaEnv, Vm};

const THREAD: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/thread-src/thread.velf");

/// The interleaving the create/start/wait pattern produces: main creates and
/// starts the thread, the worker runs synchronously at start (printing its line
/// with the argument 14), and main then collects the return value 14*3 = 42.
const EXPECTED: &str = "\
main: creating thread
main: thid ok=1
worker: got 14
main: worker returned 42
";

#[test]
fn thread_create_start_wait() {
    let m = loader::load(THREAD).expect("load thread.velf");
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
    .expect("instantiate thread");
    vm.set_import_env(Box::new(env.clone()));

    vm.call(m.entry & !1).expect("run thread main");

    let env = env.borrow();
    let cap = &env.state.capture;
    let output = String::from_utf8_lossy(&cap.stdout);
    eprintln!("---output---\n{output}------------");

    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert_eq!(output, EXPECTED);
}
