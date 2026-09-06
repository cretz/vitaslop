//! The ARM register file held in wasm LOCALS must compute exactly what the same code
//! computes with it on the globals - see `transpiler::promote`.
//!
//! # Why this test exists rather than trusting the corpus
//! The ARM conformance corpus is one instruction per case. Promotion only fires inside a
//! straight-line RUN of three or more accesses to the same register, so the corpus
//! exercises approximately none of it and passes either way: it is a test that cannot
//! fail for this feature, which is worse than no test at all. (Confirmed: the corpus
//! passes with promotion on and off, and it did so before the emitter was correct.)
//!
//! # What is checked
//! Every program here is run twice in one process - promotion off, then on - and BOTH
//! arms' full register files and NZCV are compared against each other AND against a
//! hand-computed expectation. Comparing the arms catches a promotion bug; the
//! hand-computed value catches the case where both arms are wrong together, which a
//! pure differential test would call a pass.
//!
//! The programs are chosen for the places the write-back cache can be wrong:
//! straight-line reuse (the case it exists for), a value that must survive a CALL, a
//! value written on only one side of a predicated instruction, and a loop whose back
//! edge carries a register across the dispatch join.

use vitaslop_native::{run, HostAbi, RunResult};
use vitaslop_transpiler as transpiler;

const BASE: u32 = 0x10000;

/// Run one program with promotion off and on, assert the two agree exactly, and return
/// the (common) result.
///
/// The knob is a thread-local override precisely so both arms can be built here; an
/// environment variable latched once per process could not express this test.
fn both_arms(code: &[u8], thumb: bool, in_regs: &[(usize, u32)]) -> RunResult {
    let abi = HostAbi::default();

    transpiler::set_promote_registers(false);
    let plain = run(code, BASE, BASE, thumb, &[], in_regs, &abi).expect("run unpromoted");

    transpiler::set_promote_registers(true);
    let promoted = run(code, BASE, BASE, thumb, &[], in_regs, &abi).expect("run promoted");

    // Leave the thread as we found it, so test order cannot change another test's build.
    transpiler::set_promote_registers(false);

    assert_eq!(
        plain.regs, promoted.regs,
        "promoted register file diverged: unpromoted {:x?} promoted {:x?}",
        plain.regs, promoted.regs
    );
    assert_eq!(
        (plain.flags.n, plain.flags.z, plain.flags.c, plain.flags.v),
        (promoted.flags.n, promoted.flags.z, promoted.flags.c, promoted.flags.v),
        "promoted NZCV diverged"
    );
    plain
}

/// The case promotion exists for: one register read and written over and over with no
/// branch between, so the cache holds it for the whole run and writes it back once.
#[test]
fn a_straight_run_of_reuse_computes_the_same_value() {
    // add r0,r0,r1 x6 ; bx lr.  r0 = r0 + 6*r1.
    const ADD: [u8; 4] = [0x01, 0x00, 0x80, 0xe0]; // add r0, r0, r1
    const BX_LR: [u8; 4] = [0x1e, 0xff, 0x2f, 0xe1];
    let mut code = Vec::new();
    for _ in 0..6 {
        code.extend_from_slice(&ADD);
    }
    code.extend_from_slice(&BX_LR);

    let r = both_arms(&code, false, &[(0, 100), (1, 7)]);
    assert_eq!(r.regs[0], 100 + 6 * 7, "r0 = r0 + 6*r1");
    assert_eq!(r.regs[1], 7, "r1 untouched");
}

/// A run that ends in a CALL. The callee reaches the same register globals, so every
/// value the cache is holding has to be written back BEFORE the call and re-read after
/// it - this is the sync point that is not negotiable, and the one a cache that only
/// flushed at branches would get wrong.
#[test]
fn a_value_survives_a_call_that_rewrites_it() {
    // caller at 0x10000:
    //   add r0,r0,r1      ; build a value in a run
    //   add r0,r0,r1
    //   add r0,r0,r1
    //   bl  callee        ; callee adds 1000 to r0 - it must see the built value
    //   add r0,r0,r1      ; and the caller must see the callee's
    //   bx  lr
    // callee at 0x10018:
    //   add r0,r0,r2      ; r2 seeded with 1000
    //   bx  lr
    let code: [u8; 32] = [
        0x01, 0x00, 0x80, 0xe0, // add r0, r0, r1
        0x01, 0x00, 0x80, 0xe0, // add r0, r0, r1
        0x01, 0x00, 0x80, 0xe0, // add r0, r0, r1
        0x01, 0x00, 0x00, 0xeb, // bl  +4 insns -> 0x10018
        0x01, 0x00, 0x80, 0xe0, // add r0, r0, r1
        0x1e, 0xff, 0x2f, 0xe1, // bx  lr
        0x02, 0x00, 0x80, 0xe0, // callee: add r0, r0, r2
        0x1e, 0xff, 0x2f, 0xe1, // bx  lr
    ];

    let r = both_arms(&code, false, &[(0, 5), (1, 2), (2, 1000)]);
    assert_eq!(r.regs[0], 5 + 2 + 2 + 2 + 1000 + 2, "the call must see and be seen");
}

/// After a call, the cache must RELOAD - not merely have written back.
///
/// # Why this test exists as well as the one above
/// Write-back and invalidation are two separate obligations and only the first is
/// obvious. A cache that flushed correctly but then kept trusting its local across the
/// call passes every other test in this file: the run AFTER a call usually touches the
/// register too few times to be promoted at all, so the access falls back to the global
/// and the stale local is never read. (Measured, by making exactly that change: all five
/// other tests passed.)
///
/// So this program is built to defeat that masking. Both the run before the call and the
/// run after it use r0 heavily enough to be promoted, and the run after it READS r0
/// first - which is the only shape where a missing invalidation is observable.
#[test]
fn after_a_call_a_promoted_register_is_reloaded_not_remembered() {
    // caller: 3x add r0,r0,r1 ; bl callee ; 3x add r0,r0,r1 ; bx lr
    // callee: add r0,r0,r2 ; bx lr        (r2 = 1000)
    let code: [u8; 40] = [
        0x01, 0x00, 0x80, 0xe0, // add r0, r0, r1
        0x01, 0x00, 0x80, 0xe0, // add r0, r0, r1
        0x01, 0x00, 0x80, 0xe0, // add r0, r0, r1
        0x03, 0x00, 0x00, 0xeb, // bl  +6 insns -> 0x10020
        0x01, 0x00, 0x80, 0xe0, // add r0, r0, r1
        0x01, 0x00, 0x80, 0xe0, // add r0, r0, r1
        0x01, 0x00, 0x80, 0xe0, // add r0, r0, r1
        0x1e, 0xff, 0x2f, 0xe1, // bx  lr
        0x02, 0x00, 0x80, 0xe0, // callee: add r0, r0, r2
        0x1e, 0xff, 0x2f, 0xe1, // bx  lr
    ];

    let r = both_arms(&code, false, &[(0, 5), (1, 2), (2, 1000)]);
    // A cache that remembered r0 across the call would drop the callee's +1000.
    assert_eq!(r.regs[0], 5 + 3 * 2 + 1000 + 3 * 2, "the callee's write must be reloaded");
}

/// The same obligation at a control-flow JOIN. A register promoted inside a predicated
/// body has a local that the not-taken path never wrote, so the code after the `end` must
/// go back to the global rather than trust it.
///
/// Consecutive same-condition predicated instructions fold into ONE guard body, which is
/// what makes the body long enough for its r0 to be promoted - and therefore what makes
/// this test able to fail at all.
#[test]
fn after_a_predicated_body_a_promoted_register_is_reloaded() {
    // cmp r1,#0 ; addne r0,r0,#7 x3 ; add r0,r0,#1 x3 ; bx lr
    let code: [u8; 32] = [
        0x00, 0x00, 0x51, 0xe3, // cmp   r1, #0
        0x07, 0x00, 0x80, 0x12, // addne r0, r0, #7
        0x07, 0x00, 0x80, 0x12, // addne r0, r0, #7
        0x07, 0x00, 0x80, 0x12, // addne r0, r0, #7
        0x01, 0x00, 0x80, 0xe2, // add   r0, r0, #1
        0x01, 0x00, 0x80, 0xe2, // add   r0, r0, #1
        0x01, 0x00, 0x80, 0xe2, // add   r0, r0, #1
        0x1e, 0xff, 0x2f, 0xe1, // bx    lr
    ];

    let taken = both_arms(&code, false, &[(0, 10), (1, 1)]);
    assert_eq!(taken.regs[0], 10 + 3 * 7 + 3, "predicate true");

    // The one that catches a cache trusting a local the not-taken path never wrote.
    let not_taken = both_arms(&code, false, &[(0, 10), (1, 0)]);
    assert_eq!(not_taken.regs[0], 10 + 3, "predicate false: r0 comes from the global");
}

/// A register written on only ONE side of a predicated instruction. The `if`/`end` this
/// lowers to is a control-flow join: a cache that carried a value across it would hand
/// the merge a local that the not-taken path never wrote.
#[test]
fn a_predicated_write_is_not_carried_across_the_join() {
    // cmp r1,#0 ; addne r0,r0,#7 ; add r0,r0,#1 ; add r0,r0,#1 ; bx lr
    let code: [u8; 20] = [
        0x00, 0x00, 0x51, 0xe3, // cmp  r1, #0
        0x07, 0x00, 0x80, 0x12, // addne r0, r0, #7
        0x01, 0x00, 0x80, 0xe2, // add  r0, r0, #1
        0x01, 0x00, 0x80, 0xe2, // add  r0, r0, #1
        0x1e, 0xff, 0x2f, 0xe1, // bx   lr
    ];

    let taken = both_arms(&code, false, &[(0, 10), (1, 1)]);
    assert_eq!(taken.regs[0], 10 + 7 + 1 + 1, "predicate true: the addne runs");

    let not_taken = both_arms(&code, false, &[(0, 10), (1, 0)]);
    assert_eq!(not_taken.regs[0], 10 + 1 + 1, "predicate false: the addne does not");
}

/// A loop. Its back edge re-enters through the function's dispatch `br_table`, which is
/// a join every block in the function can arrive at - so no cached value may survive it,
/// and the counter and accumulator have to be correct on every turn.
#[test]
fn a_loop_carries_its_registers_across_the_back_edge() {
    // loop: add r0,r0,r1 ; add r0,r0,r1 ; subs r2,r2,#1 ; bne loop ; bx lr
    let code: [u8; 20] = [
        0x01, 0x00, 0x80, 0xe0, // add  r0, r0, r1
        0x01, 0x00, 0x80, 0xe0, // add  r0, r0, r1
        0x01, 0x20, 0x52, 0xe2, // subs r2, r2, #1
        0xfb, 0xff, 0xff, 0x1a, // bne  -> start
        0x1e, 0xff, 0x2f, 0xe1, // bx   lr
    ];

    let r = both_arms(&code, false, &[(0, 0), (1, 3), (2, 5)]);
    assert_eq!(r.regs[0], 2 * 3 * 5, "five turns of two adds of r1");
    assert_eq!(r.regs[2], 0, "the counter ran out");
    assert!(r.flags.z, "subs to zero sets Z");
}

/// Memory traffic interleaved with register reuse. Guest stores go to linear memory and
/// the register file is in globals or locals, so the two cannot alias - but a cache that
/// mis-ordered a write-back against a load would still show up here.
#[test]
fn stores_and_loads_interleave_with_a_promoted_run() {
    // str r1,[r0] ; ldr r2,[r0] ; add r2,r2,r1 ; add r2,r2,r1 ; str r2,[r0,#4]
    // ldr r3,[r0,#4] ; add r3,r3,r2 ; bx lr
    let code: [u8; 32] = [
        0x00, 0x10, 0x80, 0xe5, // str r1, [r0]
        0x00, 0x20, 0x90, 0xe5, // ldr r2, [r0]
        0x01, 0x20, 0x82, 0xe0, // add r2, r2, r1
        0x01, 0x20, 0x82, 0xe0, // add r2, r2, r1
        0x04, 0x20, 0x80, 0xe5, // str r2, [r0, #4]
        0x04, 0x30, 0x90, 0xe5, // ldr r3, [r0, #4]
        0x02, 0x30, 0x83, 0xe0, // add r3, r3, r2
        0x1e, 0xff, 0x2f, 0xe1, // bx  lr
    ];

    // r0 points at scratch guest memory well clear of the code.
    let addr = BASE + 0x1000;
    let r = both_arms(&code, false, &[(0, addr), (1, 11)]);
    assert_eq!(r.regs[2], 11 + 11 + 11, "r2 = [r0] + 2*r1");
    assert_eq!(r.regs[3], 33 + 33, "r3 = [r0+4] + r2");
}
