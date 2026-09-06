//! Host-import milestone: run the cube's `gpu_alloc` helper end to end. It has
//! everything the pure-integer functions lack: two IT blocks (`itete`/`it`),
//! three `blx` calls into host NID stubs (sceKernelAllocMemBlock,
//! sceKernelGetMemBlockBase, sceGxmMapMemory), conditional `blt` error checks,
//! sp-relative stack slots, and a `pop {..,pc}` return. A recording host returns
//! success for each import; success is observable because `gpu_alloc` writes the
//! returned alloc UID to its out-pointer only on the all-succeeded path.

use std::cell::RefCell;

use vitaslop_loader as loader;
use vitaslop_native::{abi, DEFAULT_MEM_BYTES, HostAbi, SvcOutcome, Vm};

const CUBE: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.velf");

const GPU_ALLOC: u32 = 0x8100_08c0;
/// The UID the fake sceKernelAllocMemBlock returns; gpu_alloc propagates it.
const FAKE_UID: u32 = 0x0000_1000;

thread_local! {
    /// Import indices seen by the recording host, in call order.
    static CALLS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// Records the import call and returns success (a positive r0), so every `blt`
/// error check in gpu_alloc falls through to the success path.
fn recording_import(
    selector: u32,
    regs: &mut [u32; abi::REG_COUNT],
    _mem: &mut [u8],
    _base: u32,
    _out: &mut Vec<u8>,
) -> SvcOutcome {
    CALLS.with(|c| c.borrow_mut().push(selector));
    regs[0] = FAKE_UID; // positive -> passes `subs; blt` / `cmp; blt`
    SvcOutcome::Continue
}

#[test]
fn cube_gpu_alloc_runs_through_host_imports() {
    let m = loader::load(CUBE).expect("load cube.velf");
    let inputs = m.program_inputs();
    let abi_host = HostAbi { import: recording_import, ..Default::default() };

    let mut vm = Vm::new(
        &inputs.code,
        inputs.base,
        true,
        &[GPU_ALLOC],
        &inputs.externs,
        DEFAULT_MEM_BYTES,
        &abi_host,
    )
    .expect("instantiate gpu_alloc");

    // gpu_alloc(size=r0, ?, type=r2, out=r3): write the UID to *out.
    let out_ptr = inputs.base + 0x0090_0000;
    vm.write_mem(out_ptr, &0xDEAD_BEEFu32.to_le_bytes()).unwrap();
    vm.set_reg(0, 0x0002_0000); // size
    vm.set_reg(1, 0); // (matched against r4; either branch of the itete is fine)
    vm.set_reg(2, 1); // type
    vm.set_reg(3, out_ptr); // out pointer
    vm.call(GPU_ALLOC).expect("run gpu_alloc");

    // All three NID stubs must have been dispatched as host imports.
    let calls = CALLS.with(|c| c.borrow().clone());
    assert_eq!(calls.len(), 3, "expected 3 host imports, got {calls:?}");

    // Success path ran: the alloc UID landed in *out (fail path never writes it).
    let written = u32::from_le_bytes(vm.read_mem(out_ptr, 4).unwrap().try_into().unwrap());
    assert_eq!(written, FAKE_UID, "gpu_alloc wrote the alloc UID to *out");
}
