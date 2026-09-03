//! The low NEON bank held in wasm LOCALS across a run of vector statements must compute
//! exactly what the same code computes with every operand gathered from and scattered to
//! its scalar globals - see `transpiler::emit::NqState`.
//!
//! # Why this test exists rather than trusting the corpus
//! The ARM conformance corpus is one NEON instruction per case, and the compiled NEON
//! program in the conformance harness is gcc's auto-vectorised output in the UPPER bank.
//! The cache only does anything across a RUN of low-bank vector statements, and it is
//! only wrong at the edges of one: where a scalar VFP op, a single-register load or
//! store, a call or a loop back edge reaches the same registers through the globals.
//! Those shapes are exactly what a per-instruction corpus never contains, so it passes
//! with the cache right or wrong.
//!
//! # What is checked
//! Every program runs twice in one process - cache off, then on - and BOTH arms' full
//! low-bank register file (S0..S31), core registers and a memory window are compared
//! with each other AND against a hand-computed expectation. Comparing the arms catches a
//! cache bug; the hand-computed value catches the case where both arms are wrong
//! together, which a pure differential test would call a pass.

use vitaslop_native::{HostAbi, Vm, DEFAULT_MEM_BYTES};
use vitaslop_transpiler as transpiler;

const BASE: u32 = 0x10000;
/// A data window the programs load from and store to, well past the code.
const DATA: u32 = BASE + 0x1000;

/// One arm's observable state after a run.
#[derive(Debug, PartialEq, Eq)]
struct State {
    s: [u32; 32],
    r: [u32; 13],
    mem: Vec<u8>,
}

/// Build and run `code` (Thumb, at `BASE`) once with the cache as `on`, seeding S0..S7
/// with `q01` (Q0 then Q1), core registers with `regs` and the data window with `data`.
fn arm(code: &[u8], on: bool, q01: [f32; 8], regs: &[(usize, u32)], data: &[f32]) -> State {
    transpiler::set_neon_cache(on);
    let abi = HostAbi::default();
    let mut vm =
        Vm::new(code, BASE, true, &[BASE], &[], DEFAULT_MEM_BYTES, &abi).expect("build vm");
    for (n, v) in q01.iter().enumerate() {
        vm.set_s(n as u8, *v);
    }
    for &(i, v) in regs {
        vm.set_reg(i, v);
    }
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    vm.write_mem(DATA, &bytes).expect("seed data");
    vm.call(BASE).expect("run");
    let s = std::array::from_fn(|n| vm.get_s_bits(n as u8));
    let r = std::array::from_fn(|i| vm.get_reg(i));
    let mem = vm.read_mem(DATA, 16).expect("read data");
    State { s, r, mem }
}

/// Run both arms, assert they agree exactly, and return the common state.
fn both_arms(code: &[u8], q01: [f32; 8], regs: &[(usize, u32)], data: &[f32]) -> State {
    let plain = arm(code, false, q01, regs, data);
    let cached = arm(code, true, q01, regs, data);
    // Leave the thread as we found it, so test order cannot change another test's build.
    transpiler::set_neon_cache(true);
    assert_eq!(
        plain, cached,
        "the cached arm diverged from the uncached one\nplain  {plain:x?}\ncached {cached:x?}"
    );
    plain
}

/// The four f32 lanes of low-bank quad `k` in `st`.
fn q(st: &State, k: usize) -> [f32; 4] {
    std::array::from_fn(|i| f32::from_bits(st.s[4 * k + i]))
}

/// Thumb-2 halfwords to little-endian bytes.
fn thumb(halfwords: &[u16]) -> Vec<u8> {
    halfwords.iter().flat_map(|h| h.to_le_bytes()).collect()
}

const Q0: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
const Q1: [f32; 4] = [0.5, 0.25, 2.0, 8.0];
const SEED: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 0.5, 0.25, 2.0, 8.0];
const ONES: u32 = 0xffff_ffff;

/// The case the cache exists for: a straight run of low-bank quad, double and lane
/// operations, with nothing between them, ending in moves to the core registers.
#[test]
fn a_straight_run_of_low_bank_vector_ops() {
    let code = thumb(&[
        0xef00, 0x0d42, // vadd.f32 q0, q0, q1
        0xff00, 0x2d52, // vmul.f32 q1, q0, q1
        0xef21, 0x0d02, // vsub.f32 d0, d1, d2
        0xefb0, 0x2442, // vext.8   q1, q0, q1, #4
        0xff00, 0x3d01, // vpadd.f32 d3, d0, d1
        0xffbc, 0x4c41, // vdup.32  q2, d1[1]
        0xff24, 0x6e40, // vcgt.f32 q3, q2, q0
        0xec51, 0x0b10, // vmov     r0, r1, d0
        0xec5b, 0xab16, // vmov     r10, r11, d6
        0x4770, // bx lr
    ]);
    let st = both_arms(&code, SEED, &[], &[]);
    // q0 = Q0 + Q1 = [1.5, 2.25, 5, 12]; q1 = q0 * Q1 = [0.75, 0.5625, 10, 96];
    // d0 = d1 - d2 = [4.25, 11.4375]; q1 = bytes 4.. of q0:q1 = [11.4375, 5, 12, 0.75];
    // d3 = [d0.0 + d0.1, d1.0 + d1.1] = [15.6875, 17]; q2 = d1[1] = 12 everywhere;
    // q3 = q2 > q0 lanewise.
    assert_eq!(q(&st, 0), [4.25, 11.4375, 5.0, 12.0]);
    assert_eq!(q(&st, 1), [11.4375, 5.0, 15.6875, 17.0]);
    assert_eq!(q(&st, 2), [12.0; 4]);
    assert_eq!(&st.s[12..16], &[ONES, ONES, ONES, 0]);
    assert_eq!((st.r[0], st.r[1]), (4.25f32.to_bits(), 11.4375f32.to_bits()));
    assert_eq!((st.r[10], st.r[11]), (ONES, ONES));
}

/// The edges of a run: a scalar VFP op and single-register loads and stores reach the
/// same registers through the globals, so every one of them has to see the vector work
/// before it and be seen by the vector work after it. Double-register memory ops and a
/// single-lane load stay inside the run.
#[test]
fn scalar_ops_and_single_register_memory_see_the_vector_state() {
    let code = thumb(&[
        0xef00, 0x0d42, // vadd.f32 q0, q0, q1        q0 = [1.5, 2.25, 5, 12]
        0xee70, 0x0a81, // vadd.f32 s1, s1, s2        s1 = 7.25
        0xedc8, 0x1a00, // vstr     s3, [r8]           mem[0] = 12
        0xff00, 0x2d52, // vmul.f32 q1, q0, q1        q1 = [0.75, 1.8125, 10, 96]
        0xed98, 0x1a01, // vldr     s2, [r8, #4]       s2 = 200
        0xff00, 0x3d01, // vpadd.f32 d3, d0, d1       d3 = [8.75, 212]
        0xed98, 0x2b00, // vldr     d2, [r8]           d2 = [12, 200]
        0xed88, 0x1b02, // vstr     d1, [r8, #8]       mem[8..16] = [200, 12]
        0xf9a8, 0x088f, // vld1.32  {d0[1]}, [r8]      s1 = 12
        0x4770, // bx lr
    ]);
    let st = both_arms(&code, SEED, &[(8, DATA)], &[100.0, 200.0, 300.0, 400.0]);
    assert_eq!(q(&st, 0), [1.5, 12.0, 200.0, 12.0]);
    assert_eq!(q(&st, 1), [12.0, 200.0, 8.75, 212.0]);
    let mem: Vec<f32> =
        st.mem.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(mem, [12.0, 200.0, 200.0, 12.0]);
}

/// A run that ends in a CALL. The callee reaches the same registers, so what the caller
/// holds has to be written back before the call and re-read after it.
#[test]
fn a_call_sees_and_is_seen_by_the_run_around_it() {
    let code = thumb(&[
        0xef00, 0x0d42, // vadd.f32 q0, q0, q1        q0 = [1.5, 2.25, 5, 12]
        0xf000, 0xf803, // bl       callee (+10)
        0xff00, 0x0d52, // vmul.f32 q0, q0, q1        q0 = callee's q0 * Q1
        0x4770, // bx lr
        // callee at BASE + 0xe:
        0xef00, 0x0d40, // vadd.f32 q0, q0, q0        doubles q0
        0x4770, // bx lr
    ]);
    let st = both_arms(&code, SEED, &[], &[]);
    assert_eq!(q(&st, 0), [1.5, 1.125, 20.0, 192.0]);
    assert_eq!(q(&st, 1), Q1);
}

/// A loop: the back edge leaves the block, so the run's write-back is what the next
/// iteration reads.
#[test]
fn a_loop_carries_the_quad_across_its_back_edge() {
    let code = thumb(&[
        0xef00, 0x0d42, // loop: vadd.f32 q0, q0, q1
        0x3801, // subs r0, #1
        0xd1fb, // bne loop
        0xec51, 0x0b10, // vmov r0, r1, d0
        0x4770, // bx lr
    ]);
    let st = both_arms(&code, SEED, &[(0, 3)], &[]);
    let want: [f32; 4] = std::array::from_fn(|i| Q0[i] + 3.0 * Q1[i]);
    assert_eq!(q(&st, 0), want);
    assert_eq!((st.r[0], st.r[1]), (want[0].to_bits(), want[1].to_bits()));
}

/// The two banks mixed in one run: an upper-bank quad written from the low bank, a
/// low-bank double written from an upper-bank double, and a pairwise add over the result.
#[test]
fn upper_and_lower_banks_mix_in_one_run() {
    let code = thumb(&[
        0xef00, 0x0d42, // vadd.f32 q0, q0, q1        q0 = [1.5, 2.25, 5, 12]
        0xef40, 0x0d40, // vadd.f32 q8, q0, q0        q8 = [3, 4.5, 10, 24]
        0xef20, 0x1d82, // vsub.f32 d1, d16, d2       d1 = [2.5, 4.25]
        0xff00, 0x3d01, // vpadd.f32 d3, d0, d1       d3 = [3.75, 6.75]
        0x4770, // bx lr
    ]);
    let st = both_arms(&code, SEED, &[], &[]);
    assert_eq!(q(&st, 0), [1.5, 2.25, 2.5, 4.25]);
    assert_eq!(q(&st, 1), [0.5, 0.25, 3.75, 6.75]);
}
