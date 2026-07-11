//! Runtime experiment (ignored by default): actually execute the cube from its
//! entry point and record the sequence of NID host-import calls main() makes.
//! This validates the CPU path end to end (not just transpile) and surfaces the
//! first GXM host-call demand. Run with:
//!   cargo test -p vitaslop-conformance-harness --test cube_run -- --ignored --nocapture

use std::cell::RefCell;

use vitaslop_loader as loader;
use vitaslop_native::{DEFAULT_MEM_BYTES, HostAbi, Vm};
use vitaslop_native::SvcOutcome;

const CUBE: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.velf");

thread_local! {
    // Recorded (import index, r0..r3) per call, and a bump allocator cursor.
    static CALLS: RefCell<Vec<(u32, [u32; 4])>> = const { RefCell::new(Vec::new()) };
    static BUMP: RefCell<u32> = const { RefCell::new(0) };
}

const CALL_LIMIT: usize = 400;
const BASE: u32 = 0x8100_0000;

fn recording_import(
    selector: u32,
    regs: &mut [u32; 16],
    _mem: &mut [u8],
    _base: u32,
    _out: &mut Vec<u8>,
) -> SvcOutcome {
    let n = CALLS.with(|c| {
        let mut c = c.borrow_mut();
        c.push((selector, [regs[0], regs[1], regs[2], regs[3]]));
        c.len()
    });
    if n >= CALL_LIMIT {
        return SvcOutcome::Halt;
    }
    // Return a fresh non-null bump pointer as r0 so allocation/handle-returning
    // NIDs give the program something usable to proceed with.
    let ptr = BUMP.with(|b| {
        let mut b = b.borrow_mut();
        if *b == 0 {
            *b = BASE + 0x0300_0000; // 48 MB in: past image + heap, below stack
        }
        let p = *b;
        *b += 0x0004_0000; // 256 KB per allocation
        p
    });
    regs[0] = ptr;
    SvcOutcome::Continue
}

#[test]
#[ignore]
fn cube_run_records_nid_calls() {
    let m = loader::load(CUBE).expect("load cube.velf");
    let inputs = m.program_inputs();

    let abi = HostAbi { noreturn_svc: &[], svc: recording_import, import: recording_import };
    let mut vm = Vm::new(
        &inputs.code,
        inputs.base,
        inputs.thumb_entry,
        &[m.entry & !1],
        &inputs.externs,
        DEFAULT_MEM_BYTES,
        &abi,
    )
    .expect("instantiate cube");

    let result = vm.call(m.entry & !1);

    let calls = CALLS.with(|c| c.borrow().clone());
    eprintln!("=== main() made {} NID calls before {} ===", calls.len(),
        match &result { Ok(()) => "clean halt/return".to_string(), Err(e) => format!("{e:?}") });

    // Map import index -> (library_nid, func_nid) for labeling.
    for (i, (idx, args)) in calls.iter().enumerate().take(60) {
        let imp = m.imports.get(*idx as usize);
        match imp {
            Some(imp) => eprintln!(
                "{i:3}: import[{idx}] lib={:08x} nid={:08x} args=[{:08x} {:08x} {:08x} {:08x}]",
                imp.library_nid, imp.func_nid, args[0], args[1], args[2], args[3]
            ),
            None => eprintln!("{i:3}: import[{idx}] (out of range)"),
        }
    }
}
