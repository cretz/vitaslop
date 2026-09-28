//! End-to-end coverage for the SceLibKernel/SceThreadmgr virtual TIMER family:
//! create, open by name, start, stop, and read the count back. Run with:
//!   cargo test -p vitaslop-conformance-harness --test vita_timer
//!
//! It asserts SEMANTICS, never microsecond values, because a timer that returns a
//! plausible number is indistinguishable from one that is right: zero until
//! started, an error on a second start rather than a silent restart, advancing
//! while running, stable and non-zero once stopped, and an error for a uid that is
//! not a timer. See `../../vitaslop-conformance-suite-vita/timer-src/timer.c`.

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, HostAbi, VitaEnv, Vm};

const TIMER: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/timer-src/timer.velf");

/// The deterministic transcript kernel.c prints. Lock/wait succeed (return 0),
/// handles are valid, the event-flag pattern round-trips (set 0x5, read 0x5), and
/// the clock is monotonic.
const EXPECTED: &str = "\
create: id_ok=1 get=0 zero_before_start=1
start: first=0 second_is_error=1
running: never_backward=1 work=1
open: same_id=1
stop: ret=0 stable=1 kept_count=1
restop: is_error=1
unknown: is_error=1
delete: ret=0 gone=1 work=1
";

#[test]
fn timer_create_start_read_stop() {
    let m = loader::load(TIMER).expect("load timer.velf");
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
    .expect("instantiate timer");
    vm.set_import_env(Box::new(env.clone()));

    vm.call(m.entry & !1).expect("run timer main");

    let env = env.borrow();
    let cap = &env.state.capture;
    let output = String::from_utf8_lossy(&cap.stdout);
    eprintln!("---output---\n{output}------------");

    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert_eq!(output, EXPECTED);
}
