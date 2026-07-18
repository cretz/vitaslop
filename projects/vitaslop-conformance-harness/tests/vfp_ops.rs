//! VFP (floating-point) milestone: run a hand-assembled Thumb-2 VFP leaf function
//! through the transpiler and check the results against IEEE f32 math. This
//! exercises the whole VFP path added for the cube's `main()`: single-precision
//! arithmetic (vadd/vsub/vmul/vdiv), non-fused multiply-accumulate (vmla/vmls),
//! vneg/vabs/vsqrt, register move, float<->int convert, and the compare + `vmrs
//! APSR_nzcv` bridge from FP flags into the integer condition flags.
//!
//! Source is a hand-assembled Thumb leaf, assembled with
//! `arm-none-eabi-as -mthumb -mcpu=cortex-a9 -mfpu=neon`.

use vitaslop_native::{DEFAULT_MEM_BYTES, HostAbi, Vm};

// vadd s4,s0,s1 / vsub s5 / vmul s6 / vdiv s7 / vmov s8,s2 / vmla s8,s0,s1 /
// vmov s9,s2 / vmls s9,s0,s1 / vneg s10,s0 / vabs s11,s5 / vsqrt s12,s0 /
// vcvt.s32.f32 s13,s0 / vcvt.f32.s32 s14,s3 / vcmp s0,s1 / vmrs APSR_nzcv / bx lr
const CODE: &[u8] = &[
    0x30, 0xee, 0x20, 0x2a, 0x70, 0xee, 0x60, 0x2a, 0x20, 0xee, 0x20, 0x3a, 0xc0, 0xee, 0x20, 0x3a,
    0xb0, 0xee, 0x41, 0x4a, 0x00, 0xee, 0x20, 0x4a, 0xf0, 0xee, 0x41, 0x4a, 0x40, 0xee, 0x60, 0x4a,
    0xb1, 0xee, 0x40, 0x5a, 0xf0, 0xee, 0xe2, 0x5a, 0xb1, 0xee, 0xc0, 0x6a, 0xfd, 0xee, 0xc0, 0x6a,
    0xb8, 0xee, 0xe1, 0x7a, 0xb4, 0xee, 0x60, 0x0a, 0xf1, 0xee, 0x10, 0xfa, 0x70, 0x47,
];

const BASE: u32 = 0x8100_0000;

fn new_vm() -> Vm {
    Vm::new(CODE, BASE, true, &[BASE], &[], DEFAULT_MEM_BYTES, &HostAbi::default())
        .expect("instantiate VFP module")
}

#[test]
fn vfp_single_precision_ops() {
    let mut vm = new_vm();
    vm.set_s(0, 3.0);
    vm.set_s(1, 2.0);
    vm.set_s(2, 10.0);
    // s3 holds a raw *integer* (5) for vcvt.f32.s32.
    vm.set_s(3, f32::from_bits(5));

    vm.call(BASE).expect("run vfp");

    assert_eq!(vm.get_s(4), 5.0, "vadd");
    assert_eq!(vm.get_s(5), 1.0, "vsub");
    assert_eq!(vm.get_s(6), 6.0, "vmul");
    assert_eq!(vm.get_s(7), 1.5, "vdiv");
    assert_eq!(vm.get_s(8), 16.0, "vmla: 10 + 3*2");
    assert_eq!(vm.get_s(9), 4.0, "vmls: 10 - 3*2");
    assert_eq!(vm.get_s(10), -3.0, "vneg");
    assert_eq!(vm.get_s(11), 1.0, "vabs |3-2|... |1|");
    assert_eq!(vm.get_s(12), 3.0_f32.sqrt(), "vsqrt");
    assert_eq!(vm.get_s_bits(13), 3, "vcvt.s32.f32: (int)3.0");
    assert_eq!(vm.get_s(14), 5.0, "vcvt.f32.s32: (float)5");

    // vcmp 3.0 > 2.0 then vmrs APSR_nzcv: greater-than sets C, clears N/Z/V.
    let f = vm.flags();
    assert!(!f.n && !f.z && f.c && !f.v, "flags after vcmp greater: {:?}", (f.n, f.z, f.c, f.v));
}

#[test]
fn vfp_compare_less_and_equal() {
    // Less-than: N set, C clear. Equal: Z and C set. Re-run with fresh seeds.
    let mut vm = new_vm();
    vm.set_s(0, 1.0);
    vm.set_s(1, 2.0);
    vm.set_s(2, 0.0);
    vm.set_s(3, f32::from_bits(0));
    vm.call(BASE).expect("run vfp less");
    let f = vm.flags();
    assert!(f.n && !f.z && !f.c && !f.v, "flags after vcmp less: {:?}", (f.n, f.z, f.c, f.v));

    let mut vm = new_vm();
    vm.set_s(0, 4.0);
    vm.set_s(1, 4.0);
    vm.set_s(2, 0.0);
    vm.set_s(3, f32::from_bits(0));
    vm.call(BASE).expect("run vfp equal");
    let f = vm.flags();
    assert!(!f.n && f.z && f.c && !f.v, "flags after vcmp equal: {:?}", (f.n, f.z, f.c, f.v));
}
