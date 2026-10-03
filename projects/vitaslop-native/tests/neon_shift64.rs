//! `vshl.u64` / `vshl.s64` by REGISTER: a 64-bit lane shifted by the signed low byte of the
//! matching lane of the amount register - see `transpiler::emit::emit_shift_reg_64`.
//!
//! It was refused as unsupported, which made the whole block trap; a retail title's
//! bit-stream reader (`vshl.u64 d1, d0, d1`) hit it as soon as a new game started. Every
//! expectation here is hand-computed from the architecture's definition, including the
//! cases a 32-bit lane walk would get wrong (a shift that crosses the word boundary) and the
//! out-of-range amounts.

use vitaslop_native::{HostAbi, Vm, DEFAULT_MEM_BYTES};

const BASE: u32 = 0x10000;

/// Run `vshl.{u,s}64 d1, d0, d1; bx lr` with d0 = `x` and d1 = `amount`, returning d1.
fn vshl64(signed: bool, x: u64, amount: u64) -> u64 {
    // The instruction as the title encodes it: `31 ff 00 14` is `vshl.u64 d1, d0, d1`; the
    // U bit is bit 12 of the first halfword, so `ef31` is the signed form.
    let first: u16 = if signed { 0xef31 } else { 0xff31 };
    let code: Vec<u8> = [first, 0x1400, 0x4770].iter().flat_map(|h| h.to_le_bytes()).collect();
    let abi = HostAbi::default();
    let mut vm = Vm::new(&code, BASE, true, &[BASE], &[], DEFAULT_MEM_BYTES, &abi).expect("build vm");
    let words = [x as u32, (x >> 32) as u32, amount as u32, (amount >> 32) as u32];
    for (n, w) in words.iter().enumerate() {
        vm.set_s(n as u8, f32::from_bits(*w));
    }
    vm.call(BASE).expect("run");
    // The source is left alone.
    assert_eq!(vm.get_s_bits(0), x as u32);
    assert_eq!(vm.get_s_bits(1), (x >> 32) as u32);
    u64::from(vm.get_s_bits(2)) | (u64::from(vm.get_s_bits(3)) << 32)
}

/// An amount is the SIGNED LOW BYTE of its lane; the rest of the lane is ignored.
fn amt(n: i8) -> u64 {
    0xdead_beef_0000_0000 | u64::from(n as u8)
}

#[test]
fn unsigned_left_and_right() {
    let x = 0x8000_0000_0000_0001;
    assert_eq!(vshl64(false, x, amt(4)), 0x10);
    assert_eq!(vshl64(false, x, amt(-4)), 0x0800_0000_0000_0000);
    assert_eq!(vshl64(false, x, amt(0)), x);
}

#[test]
fn a_shift_crosses_the_word_boundary() {
    assert_eq!(vshl64(false, 1, amt(36)), 1 << 36);
    assert_eq!(vshl64(false, 0x0000_0010_0000_0000, amt(-33)), 0x8);
    assert_eq!(vshl64(false, 0xffff_ffff, amt(63)), 0x8000_0000_0000_0000);
}

#[test]
fn sixty_four_or_more_clears_or_fills_with_the_sign() {
    let x = 0x8000_0000_0000_0001;
    assert_eq!(vshl64(false, x, amt(64)), 0);
    assert_eq!(vshl64(false, x, amt(-64)), 0);
    assert_eq!(vshl64(false, x, amt(127)), 0);
    assert_eq!(vshl64(true, x, amt(-64)), u64::MAX);
    assert_eq!(vshl64(true, x, amt(-128)), u64::MAX);
    assert_eq!(vshl64(true, 0x7000_0000_0000_0000, amt(-100)), 0);
}

#[test]
fn signed_right_is_arithmetic() {
    let x = 0x8000_0000_0000_0001;
    assert_eq!(vshl64(true, x, amt(-4)), 0xf800_0000_0000_0000);
    assert_eq!(vshl64(true, x, amt(4)), 0x10);
    assert_eq!(vshl64(true, 0x4000_0000_0000_0000, amt(-62)), 1);
}
