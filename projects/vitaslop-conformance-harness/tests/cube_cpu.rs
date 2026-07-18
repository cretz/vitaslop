//! End-to-end CPU milestone: run the cube's own Thumb-2 `memcpy` and `memset`
//! through the transpiler and verify the bytes they move in guest memory. These
//! are the simplest complete functions in the cube (no VFP, no host calls), so
//! they exercise the whole integer + memory + CFG + flags core together: cbz,
//! subs/cmp with flags, pre-index writeback loads/stores, a backward `bne` loop,
//! and `bx lr` returning through the wasm call stack.

use vitaslop_loader as loader;
use vitaslop_native::{DEFAULT_MEM_BYTES, HostAbi, Vm};

const CUBE: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.velf");

// Addresses from the cube disassembly.
const MEMCPY: u32 = 0x8100_092c;
const MEMSET: u32 = 0x8100_0948;

fn cube_vm(entries: &[u32]) -> Vm {
    let m = loader::load(CUBE).expect("load cube.velf");
    let inputs = m.program_inputs();
    Vm::new(
        &inputs.code,
        inputs.base,
        true, // Thumb
        entries,
        &inputs.externs,
        DEFAULT_MEM_BYTES,
        &HostAbi::default(),
    )
    .expect("instantiate cube module")
}

#[test]
fn cube_memcpy_copies_bytes() {
    let mut vm = cube_vm(&[MEMCPY]);
    let base = 0x8100_0000;
    let src = base + 0x0080_0000;
    let dst = base + 0x0081_0000;
    let data: Vec<u8> = (0..97u32).map(|i| (i.wrapping_mul(37).wrapping_add(11)) as u8).collect();

    vm.write_mem(src, &data).unwrap();
    // Poison the destination so a short/over copy is visible.
    vm.write_mem(dst, &vec![0xEE; data.len() + 4]).unwrap();
    vm.set_reg(0, dst);
    vm.set_reg(1, src);
    vm.set_reg(2, data.len() as u32);
    vm.call(MEMCPY).expect("run memcpy");

    assert_eq!(vm.read_mem(dst, data.len()).unwrap(), data, "copied bytes");
    // The byte just past the copy must be untouched.
    assert_eq!(vm.read_mem(dst + data.len() as u32, 1).unwrap(), [0xEE], "no overrun");
}

#[test]
fn cube_memcpy_zero_length_is_noop() {
    let mut vm = cube_vm(&[MEMCPY]);
    let base = 0x8100_0000;
    let dst = base + 0x0081_0000;
    vm.write_mem(dst, &[0x11, 0x22, 0x33]).unwrap();
    vm.set_reg(0, dst);
    vm.set_reg(1, base + 0x0080_0000);
    vm.set_reg(2, 0);
    vm.call(MEMCPY).expect("run memcpy len=0");
    assert_eq!(vm.read_mem(dst, 3).unwrap(), [0x11, 0x22, 0x33]);
}

#[test]
fn cube_memset_fills_bytes() {
    let mut vm = cube_vm(&[MEMSET]);
    let base = 0x8100_0000;
    let dst = base + 0x0082_0000;
    let len = 73u32;
    vm.write_mem(dst, &vec![0x00; (len + 2) as usize]).unwrap();
    vm.set_reg(0, dst);
    vm.set_reg(1, 0xAB);
    vm.set_reg(2, len);
    vm.call(MEMSET).expect("run memset");

    assert_eq!(vm.read_mem(dst, len as usize).unwrap(), vec![0xAB; len as usize], "filled");
    assert_eq!(vm.read_mem(dst + len, 1).unwrap(), [0x00], "no overrun");
}
