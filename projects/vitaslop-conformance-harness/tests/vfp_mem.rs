//! VFP memory milestone: exercise every VFP load/store form the cube uses, plus
//! the S/D register aliasing that makes them correct. A hand-assembled Thumb leaf
//! runs:
//!   vldr/vstr of a low double D1 (aliased over S2:S3),
//!   an `vadd s2` that must show through D1 (aliasing),
//!   vldr/vstr of an upper double D16 (its own i64 global, no alias),
//!   vld1/vst1 (NEON) with writeback,
//!   vpush/vpop of D4 around a clobber (stack save/restore),
//!   and a pc-relative literal `vldr s20, .Lc`.

use vitaslop_native::{DEFAULT_MEM_BYTES, HostAbi, Vm};

const CODE: &[u8] = &[
    0x90, 0xed, 0x00, 0x1b, 0x81, 0xed, 0x00, 0x1b, 0x31, 0xee, 0x01, 0x1a, 0x81, 0xed, 0x02, 0x1b,
    0xd0, 0xed, 0x00, 0x0b, 0xc1, 0xed, 0x04, 0x0b, 0x20, 0xf9, 0x8f, 0x07, 0x01, 0xf9, 0x9d, 0x07,
    0x2d, 0xed, 0x02, 0x4b, 0xb0, 0xee, 0x40, 0x4a, 0xbd, 0xec, 0x02, 0x4b, 0x82, 0xed, 0x00, 0x4b,
    0x9f, 0xed, 0x03, 0xaa, 0x82, 0xed, 0x02, 0xaa, 0x70, 0x47, 0x00, 0xbf, 0xaf, 0xf3, 0x00, 0x80,
    0x00, 0x00, 0x2a, 0x42,
];

const BASE: u32 = 0x8100_0000;
const R0: u32 = BASE + 0x1_0000;
const R1: u32 = BASE + 0x2_0000;
const R2: u32 = BASE + 0x3_0000;

fn read_f32(vm: &mut Vm, addr: u32) -> f32 {
    let b = vm.read_mem(addr, 4).unwrap();
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[test]
fn vfp_memory_and_aliasing() {
    let mut vm =
        Vm::new(CODE, BASE, true, &[BASE], &[], DEFAULT_MEM_BYTES, &HostAbi::default())
            .expect("instantiate VFP mem module");

    // Input: two f32 packed as a double at [r0].
    vm.write_mem(R0, &1.5_f32.to_le_bytes()).unwrap();
    vm.write_mem(R0 + 4, &2.5_f32.to_le_bytes()).unwrap();
    vm.set_reg(0, R0);
    vm.set_reg(1, R1);
    vm.set_reg(2, R2);

    vm.call(BASE).expect("run vfp mem");

    // D1 roundtrip: [r1] == input.
    assert_eq!(read_f32(&mut vm, R1), 1.5, "vstr d1 low");
    assert_eq!(read_f32(&mut vm, R1 + 4), 2.5, "vstr d1 high");
    // After `vadd s2,s2,s2` the low half of D1 (= S2) is 3.0 - aliasing must show.
    assert_eq!(read_f32(&mut vm, R1 + 8), 3.0, "aliasing: s2 write visible in d1 low");
    assert_eq!(read_f32(&mut vm, R1 + 12), 2.5, "d1 high unchanged");
    // D16 (upper double, no single alias) roundtrip.
    assert_eq!(read_f32(&mut vm, R1 + 16), 1.5, "vstr d16 low");
    assert_eq!(read_f32(&mut vm, R1 + 20), 2.5, "vstr d16 high");
    // D4 restored to its pre-clobber value (0.0) by vpush/vpop across `vmov s8`.
    assert_eq!(read_f32(&mut vm, R2), 0.0, "vpop restores d4 low (s8)");
    assert_eq!(read_f32(&mut vm, R2 + 4), 0.0, "vpop restores d4 high (s9)");
    // pc-relative literal load.
    assert_eq!(read_f32(&mut vm, R2 + 8), 42.5, "pc-relative vldr literal");

    // Register-level checks too.
    assert_eq!(vm.get_s(2), 3.0, "s2 after vadd");
    assert_eq!(vm.get_s(0), 1.5, "s0 from vld1 d0 low");
    assert_eq!(vm.get_s(1), 2.5, "s1 from vld1 d0 high");
    assert_eq!(vm.get_s(20), 42.5, "s20 literal");
    // vld1 wrote back r1 += 8.
    assert_eq!(vm.get_reg(1), R1 + 8, "vst1 writeback");
}
