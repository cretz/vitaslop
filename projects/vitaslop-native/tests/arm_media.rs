//! The A32 (ARM-state) MEDIA instructions and the NEON `vswp`/`vcnt`, end to end: decode,
//! lift, run.
//!
//! The whole A32 media class used to decode as "incomplete" - Thumb-2 had every one of these
//! and ARM state had none - so any ARM-state function using one lifted with a trap at that
//! instruction. MEASURED on a retail title's hand-written ARM/NEON routines (`ubfx`, `vcnt`,
//! `vswp`), reached as soon as a new game started.
//!
//! `pkhbt`/`pkhtb`, `usad8`/`usada8` and the dual (`smuad`/`smlad`/`smusd`/`smlsd`, and their
//! `x` forms) and most-significant (`smmul`/`smmla`/`smmls`, and their `r` forms) multiplies
//! are exercised in BOTH states against a reference written from the architecture's
//! pseudocode. The Thumb decoder used to drop the `x`/`r` bit and hand back the plain form -
//! a silently different result - and rejected `usad8` while accepting its invalid neighbour.
//! Each case is one instruction followed by `bx lr`.

use vitaslop_native::{HostAbi, Vm, DEFAULT_MEM_BYTES};

const BASE: u32 = 0x10000;
const BX_LR: u32 = 0xe12f_ff1e;

/// Run one ARM-state instruction with r0..r3 seeded, returning r0.
fn arm1(word: u32, regs: [u32; 4]) -> u32 {
    let code: Vec<u8> = [word, BX_LR].iter().flat_map(|w| w.to_le_bytes()).collect();
    let abi = HostAbi::default();
    let mut vm = Vm::new(&code, BASE, false, &[BASE], &[], DEFAULT_MEM_BYTES, &abi).expect("build vm");
    for (i, v) in regs.iter().enumerate() {
        vm.set_reg(i, *v);
    }
    vm.call(BASE).expect("run");
    vm.get_reg(0)
}

#[test]
fn bitfield_extract_insert_clear() {
    // ubfx r0, r1, #31, #1 / #27, #4 (the retail title's two encodings, re-targeted to r0/r1)
    assert_eq!(arm1(0xe7e0_0fd1, [0, 0x8000_0000, 0, 0]), 1);
    assert_eq!(arm1(0xe7e3_0dd1, [0, 0xd800_0000, 0, 0]), 0xb);
    // sbfx r0, r1, #4, #8
    assert_eq!(arm1(0xe7a7_0251, [0, 0x0000_0f80, 0, 0]), 0xffff_fff8);
    assert_eq!(arm1(0xe7a7_0251, [0, 0x0000_0700, 0, 0]), 0x70);
    // bfi r0, r1, #8, #4
    assert_eq!(arm1(0xe7cb_0411, [0xffff_ffff, 0x5, 0, 0]), 0xffff_f5ff);
    // bfc r0, #0, #4
    assert_eq!(arm1(0xe7c3_001f, [0x1234_5678, 0, 0, 0]), 0x1234_5670);
}

#[test]
fn extend_and_reverse() {
    assert_eq!(arm1(0xe6ef_0071, [0, 0x1234_56f8, 0, 0]), 0xf8); // uxtb r0, r1
    assert_eq!(arm1(0xe6af_0071, [0, 0x1234_56f8, 0, 0]), 0xffff_fff8); // sxtb r0, r1
    assert_eq!(arm1(0xe6ff_0471, [0, 0x1234_56f8, 0, 0]), 0x3456); // uxth r0, r1, ror #8
    assert_eq!(arm1(0xe6bf_0f31, [0, 0x1234_5678, 0, 0]), 0x7856_3412); // rev
    assert_eq!(arm1(0xe6bf_0fb1, [0, 0x1234_5678, 0, 0]), 0x3412_7856); // rev16
    assert_eq!(arm1(0xe6ff_0f31, [0, 0x0000_0001, 0, 0]), 0x8000_0000); // rbit
    assert_eq!(arm1(0xe6ff_0fb1, [0, 0x0000_80ff, 0, 0]), 0xffff_ff80); // revsh
}

#[test]
fn saturate_and_parallel_add() {
    // ssat r0, #8, r1 / usat r0, #8, r1
    assert_eq!(arm1(0xe6a7_0011, [0, 300, 0, 0]), 127);
    assert_eq!(arm1(0xe6a7_0011, [0, (-300i32) as u32, 0, 0]), (-128i32) as u32);
    assert_eq!(arm1(0xe6e8_0011, [0, 300, 0, 0]), 255);
    assert_eq!(arm1(0xe6e8_0011, [0, (-5i32) as u32, 0, 0]), 0);
    // uadd8 r0, r1, r2 (sets GE, which `sel` then reads)
    assert_eq!(arm1(0xe651_0f92, [0, 0xff01_7f80, 0x0101_0180, 0]), 0x0002_8000);
}

/// Run one ARM-state NEON instruction with d0/d1 seeded, returning d0..d2.
fn neon1(word: u32, d0: u64, d1: u64) -> [u64; 3] {
    let code: Vec<u8> = [word, BX_LR].iter().flat_map(|w| w.to_le_bytes()).collect();
    let abi = HostAbi::default();
    let mut vm = Vm::new(&code, BASE, false, &[BASE], &[], DEFAULT_MEM_BYTES, &abi).expect("build vm");
    for (i, w) in [d0 as u32, (d0 >> 32) as u32, d1 as u32, (d1 >> 32) as u32].iter().enumerate() {
        vm.set_s(i as u8, f32::from_bits(*w));
    }
    vm.call(BASE).expect("run");
    std::array::from_fn(|d| u64::from(vm.get_s_bits(2 * d as u8)) | (u64::from(vm.get_s_bits(2 * d as u8 + 1)) << 32))
}

#[test]
fn vswp_exchanges_whole_registers() {
    // vswp d0, d1
    let r = neon1(0xf3b2_0001, 0x1111_2222_3333_4444, 0xaaaa_bbbb_cccc_dddd);
    assert_eq!((r[0], r[1]), (0xaaaa_bbbb_cccc_dddd, 0x1111_2222_3333_4444));
}

#[test]
fn vcnt_counts_bits_per_byte() {
    // vcnt.8 d2, d0
    let r = neon1(0xf3b0_2500, u64::from_le_bytes([0x00, 0xff, 0x0f, 0x81, 0x7e, 0x01, 0x10, 0xaa]), 0);
    assert_eq!(r[2], u64::from_le_bytes([0, 8, 4, 2, 6, 1, 1, 4]));
}

#[test]
fn vclz_and_vcls_count_per_element() {
    // vclz.i32 d2, d0
    let r = neon1(0xf3b8_2480, 0xffff_ffff_0001_0000, 0);
    assert_eq!(r[2], (0u64 << 32) | 15);
    assert_eq!(neon1(0xf3b8_2480, 0, 0)[2], (32u64 << 32) | 32);
    // vclz.i8 d2, d0
    let r = neon1(0xf3b0_2480, u64::from_le_bytes([0x00, 0x01, 0x80, 0x10, 0xff, 0x7f, 0x02, 0x40]), 0);
    assert_eq!(r[2], u64::from_le_bytes([8, 7, 0, 3, 0, 1, 6, 1]));
    // vcls.s16 d2, d0: sign bits after the sign bit that equal it
    let lanes = [0x0000u16, 0xffff, 0x0001, 0xc000];
    let d0 = lanes.iter().enumerate().fold(0u64, |a, (i, l)| a | (u64::from(*l) << (16 * i)));
    let r = neon1(0xf3b4_2400, d0, 0);
    let want = [15u16, 15, 14, 1].iter().enumerate().fold(0u64, |a, (i, l)| a | (u64::from(*l) << (16 * i)));
    assert_eq!(r[2], want);
}

/// Run one Thumb-2 (32-bit) instruction with r0..r3 seeded, returning r0.
fn t32(hw1: u16, hw2: u16, regs: [u32; 4]) -> u32 {
    let code: Vec<u8> = [hw1, hw2, 0x4770].iter().flat_map(|h| h.to_le_bytes()).collect();
    let abi = HostAbi::default();
    let mut vm = Vm::new(&code, BASE, true, &[BASE], &[], DEFAULT_MEM_BYTES, &abi).expect("build vm");
    for (i, v) in regs.iter().enumerate() {
        vm.set_reg(i, *v);
    }
    vm.call(BASE).expect("run");
    vm.get_reg(0)
}

// The reference, straight from the pseudocode: 16-bit halves as signed integers, 64-bit
// arithmetic, the result's low 32 bits (dual) or bits 63:32 (most-significant word).
fn lo(v: u32) -> i64 {
    i64::from(v as u16 as i16)
}
fn hi(v: u32) -> i64 {
    i64::from((v >> 16) as u16 as i16)
}
fn dual(n: u32, m: u32, x: bool, sub: bool, a: Option<u32>) -> u32 {
    let m = if x { m.rotate_right(16) } else { m };
    let p = if sub { lo(n) * lo(m) - hi(n) * hi(m) } else { lo(n) * lo(m) + hi(n) * hi(m) };
    (p + a.map_or(0, |a| i64::from(a as i32))) as u32
}
fn msw(n: u32, m: u32, a: Option<u32>, sub: bool, round: bool) -> u32 {
    let p = i64::from(n as i32).wrapping_mul(i64::from(m as i32));
    let acc = a.map_or(0i64, |a| (i64::from(a as i32)) << 32);
    let s = if sub { acc.wrapping_sub(p) } else { acc.wrapping_add(p) };
    let s = if round { s.wrapping_add(0x8000_0000) } else { s };
    (s >> 32) as u32
}
fn usad(n: u32, m: u32, a: u32) -> u32 {
    (0..4).map(|k| ((n >> (8 * k)) as u8).abs_diff((m >> (8 * k)) as u8) as u32).sum::<u32>().wrapping_add(a)
}

/// Operand sets that reach the corners: the halves' extremes (so the exchange and the sign
/// extension both matter), a product whose low word rounds up, and bytes both ways round.
const CASES: [[u32; 4]; 4] = [
    [0, 0x7fff_8000, 0x8001_0003, 0x0000_1000],
    [0, 0x1234_5678, 0x9abc_def0, 0xfedc_ba98],
    [0, 0xffff_ffff, 0x8000_0000, 0x8000_0000],
    [0, 0x0180_ff01, 0xff01_0180, 0x7fff_ffff],
];

#[test]
fn dual_multiplies_both_states() {
    // (a32 word, t32 halfwords, x, sub, accumulate) with rd=r0, rn=r1, rm=r2, ra=r3.
    let forms: [(u32, [u16; 2], bool, bool, bool); 8] = [
        (0xe700_f211, [0xfb21, 0xf002], false, false, false), // smuad
        (0xe700_f231, [0xfb21, 0xf012], true, false, false),  // smuadx
        (0xe700_3211, [0xfb21, 0x3002], false, false, true),  // smlad
        (0xe700_3231, [0xfb21, 0x3012], true, false, true),   // smladx
        (0xe700_f251, [0xfb41, 0xf002], false, true, false),  // smusd
        (0xe700_f271, [0xfb41, 0xf012], true, true, false),   // smusdx
        (0xe700_3251, [0xfb41, 0x3002], false, true, true),   // smlsd
        (0xe700_3271, [0xfb41, 0x3012], true, true, true),    // smlsdx
    ];
    for (a32, [h1, h2], x, sub, acc) in forms {
        for regs in CASES {
            let want = dual(regs[1], regs[2], x, sub, acc.then_some(regs[3]));
            assert_eq!(arm1(a32, regs), want, "a32 {a32:#010x} {regs:x?}");
            assert_eq!(t32(h1, h2, regs), want, "t32 {h1:04x} {h2:04x} {regs:x?}");
        }
    }
}

#[test]
fn most_significant_word_multiplies_both_states() {
    // (a32 word, t32 halfwords, accumulate, sub, round)
    let forms: [(u32, [u16; 2], bool, bool, bool); 6] = [
        (0xe750_f211, [0xfb51, 0xf002], false, false, false), // smmul
        (0xe750_f231, [0xfb51, 0xf012], false, false, true),  // smmulr
        (0xe750_3211, [0xfb51, 0x3002], true, false, false),  // smmla
        (0xe750_3231, [0xfb51, 0x3012], true, false, true),   // smmlar
        (0xe750_32d1, [0xfb61, 0x3002], true, true, false),   // smmls
        (0xe750_32f1, [0xfb61, 0x3012], true, true, true),    // smmlsr
    ];
    for (a32, [h1, h2], acc, sub, round) in forms {
        for regs in CASES {
            let want = msw(regs[1], regs[2], acc.then_some(regs[3]), sub, round);
            assert_eq!(arm1(a32, regs), want, "a32 {a32:#010x} {regs:x?}");
            assert_eq!(t32(h1, h2, regs), want, "t32 {h1:04x} {h2:04x} {regs:x?}");
        }
    }
}

#[test]
fn sum_of_absolute_differences_both_states() {
    for regs in CASES {
        assert_eq!(arm1(0xe780_f211, regs), usad(regs[1], regs[2], 0), "usad8 a32 {regs:x?}");
        assert_eq!(t32(0xfb71, 0xf002, regs), usad(regs[1], regs[2], 0), "usad8 t32 {regs:x?}");
        assert_eq!(arm1(0xe780_3211, regs), usad(regs[1], regs[2], regs[3]), "usada8 a32 {regs:x?}");
        assert_eq!(t32(0xfb71, 0x3002, regs), usad(regs[1], regs[2], regs[3]), "usada8 t32 {regs:x?}");
    }
}

#[test]
fn pack_halfwords_both_states() {
    let (n, m) = (0x1111_2222u32, 0x8765_4321u32);
    let regs = [0, n, m, 0];
    // pkhbt r0, r1, r2, lsl #8: bottom from rn, top from rm << 8.
    let bt = (n & 0xffff) | ((m << 8) & 0xffff_0000);
    assert_eq!(arm1(0xe681_0412, regs), bt);
    assert_eq!(t32(0xeac1, 0x2002, regs), bt);
    // pkhtb r0, r1, r2, asr #16: top from rn, bottom from rm >> 16 (arithmetic).
    let tb = (n & 0xffff_0000) | ((((m as i32) >> 16) as u32) & 0xffff);
    assert_eq!(arm1(0xe681_0852, regs), tb);
    assert_eq!(t32(0xeac1, 0x4022, regs), tb);
    // pkhtb r0, r1, r2, asr #32 (imm5 = 0): the bottom half is rm's sign, repeated.
    let tb32 = (n & 0xffff_0000) | 0xffff;
    assert_eq!(arm1(0xe681_0052, regs), tb32);
}

/// `lsr #32` / `asr #32` in a Thumb-2 shifted-register operand: the decoder expands the
/// encoded `0` to the literal 32, and a wasm shift by 32 is a shift by NOTHING.
#[test]
fn thumb2_shift_by_32_is_not_a_shift_by_0() {
    let regs = [0, 5, 0x8000_0001, 0];
    // add.w r0, r1, r2, lsr #32  ->  5 + 0
    assert_eq!(t32(0xeb01, 0x0012, regs), 5);
    // add.w r0, r1, r2, asr #32  ->  5 + (-1)
    assert_eq!(t32(0xeb01, 0x0022, regs), 4);
}
