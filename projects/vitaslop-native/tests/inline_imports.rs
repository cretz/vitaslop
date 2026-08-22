//! Execution test for the INLINE IMPORT forms: does the emitted wasm actually compute
//! what [`InlineOp`] says it computes?
//!
//! # The leg this closes
//! An inline import is checked in two places and, until this file, only two:
//!
//!   1. `InlineOp::eval` against the real host handler (`inline_ops_match_their_handlers`
//!      and `mirror_matches_its_handlers`, in the runtime crate). That proves the
//!      DEFINITION is right.
//!   2. `wasmparser::validate` on the emitted module. That proves the code is well
//!      FORMED.
//!
//! Neither proves the emitted code computes the definition. The instruction sequences in
//! `emit_import` are written by hand, one per form, and a wrong shift, a swapped guard
//! arm or a mistyped store offset produces a module that validates perfectly and answers
//! wrongly - in guest code that never traps, on a call the host never sees. So each form
//! is run here through the real engine and compared against `eval`.
//!
//! # Both arms, deliberately
//! Every guarded form is exercised on BOTH sides of its guard, because the fallback is
//! the arm nothing else would notice: an inline form that quietly answers the
//! out-of-range or clamped case itself is exactly the bug the guard exists to prevent,
//! and it is invisible from the outside (the answer is plausible, and no host call is
//! made to contradict it). The mock handler records every crossing, so "did this reach
//! the host" is asserted, not assumed.

use vitaslop_native::{HostAbi, Vm};
use vitaslop_runtime::SvcOutcome;
use vitaslop_transpiler::abi::{self, REG_COUNT};
use vitaslop_transpiler::{Extern, InlineImport, InlineOp, Program};

const BASE: u32 = 0x1_0000;
const MEM_BYTES: u32 = 0x10_0000;
/// The one guest function: `bl <stub>`, then a structural return.
const ENTRY: u32 = BASE;
/// The import stub's address. Never executed - the `bl` to it becomes the host call (or
/// its inline form), so it only has to be an address the externs table names.
const STUB: u32 = 0x2_0000;

/// Where the test puts the guest structure an inline form reads through r0.
const PTR: u32 = BASE + 0x1000;
/// Where the test puts the buffer a storing form writes through r0.
const OUT_PTR: u32 = BASE + 0x2000;
/// The r0 the mock handler returns, seeded per test. This is how a fallback is told
/// apart from an inline answer by VALUE as well as by the crossing record.
const HANDLER_ANSWER: u32 = BASE + 0x3000;

/// `bl <target>` (ARM). The transpiler resolves the target as `addr + (off << 2)` with
/// the PC+8 prefetch bias already folded in, so the raw `imm24` is
/// `(target - addr - 8) >> 2`.
fn bl(addr: u32, target: u32) -> u32 {
    let imm24 = ((target.wrapping_sub(addr).wrapping_sub(8)) >> 2) & 0x00FF_FFFF;
    0xEB00_0000 | imm24
}

/// `bx lr` (ARM): a structural return in the transpiler.
const BX_LR: u32 = 0xE12F_FF1E;

fn build_code() -> Vec<u8> {
    let mut code = vec![0u8; 8];
    code[0..4].copy_from_slice(&bl(ENTRY, STUB).to_le_bytes());
    code[4..8].copy_from_slice(&BX_LR.to_le_bytes());
    code
}

/// The mock host import: record the crossing in the output buffer, then answer with the
/// word the test seeded at [`HANDLER_ANSWER`].
///
/// It records rather than computes because what matters here is not what the handler
/// says - the runtime's own tests already hold the handler and `eval` together - but
/// WHETHER the boundary was crossed at all.
fn host_import(
    _selector: u32,
    regs: &mut [u32; REG_COUNT],
    mem: &mut [u8],
    base: u32,
    out: &mut Vec<u8>,
) -> SvcOutcome {
    out.push(b'H');
    let off = (HANDLER_ANSWER - base) as usize;
    regs[0] = u32::from_le_bytes(mem[off..off + 4].try_into().expect("4 bytes"));
    SvcOutcome::Continue
}

fn noop_svc(
    _selector: u32,
    _regs: &mut [u32; REG_COUNT],
    _mem: &mut [u8],
    _base: u32,
    _out: &mut Vec<u8>,
) -> SvcOutcome {
    SvcOutcome::Continue
}

/// A VM whose single import lowers to `op`, with the handler's answer seeded.
fn vm_with(op: InlineOp) -> Vm {
    let code = build_code();
    let externs = [Extern { addr: STUB, import: 0 }];
    let inline = [InlineImport { import: 0, op }];
    let abi = HostAbi { noreturn_svc: &[], svc: noop_svc, import: host_import };
    let mut vm = Vm::from_program(
        &Program {
            code: &code,
            base: BASE,
            thumb: false,
            entries: &[ENTRY],
            arm_entries: &[],
            externs: &externs,
            redirects: &[],
            inline_imports: &inline,
            noreturn_svc: &[],
            mem_bytes: MEM_BYTES,
            discover_code_pointers: false,
            import_memory: false,
        },
        &abi,
    )
    .expect("the module builds and instantiates");
    vm.write_mem(HANDLER_ANSWER, &HANDLER_SENTINEL.to_le_bytes()).expect("seed the handler answer");
    vm
}

/// The value the mock handler returns. Deliberately not a plausible result of any op
/// under test, so an assertion cannot pass by coincidence.
const HANDLER_SENTINEL: u32 = 0xFEED_FACE;

/// Write the host-mirror block this module reserved. The test plays the host's side of
/// the [`InlineOp::LoadMirror`] contract - see [`Vm::mirror_off`].
fn write_mirror(vm: &mut Vm, words: &[u32]) {
    let off = vm.mirror_off().expect("a mirror op reserves the block");
    let addr = BASE + off as u32;
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    vm.write_mem(addr, &bytes).expect("the mirror block is inside linear memory");
}

/// Run the guest and report whether the host was reached.
fn run(vm: &mut Vm) -> bool {
    vm.call(ENTRY).expect("the guest returns");
    !vm.output().is_empty()
}

#[test]
fn load_shift_mask_computes_eval_inline() {
    let op = InlineOp::LoadShiftMask { offset: 4, shift: 8, mask: 0xf, plus: 0 };
    let word = 0xABCD_1234u32;
    let mut vm = vm_with(op);
    vm.write_mem(PTR + 4, &word.to_le_bytes()).expect("seed the struct");
    vm.set_reg(0, PTR);
    let crossed = run(&mut vm);
    assert!(!crossed, "an in-range pointer must not reach the host");
    assert_eq!(vm.get_reg(0), op.eval(word), "the emitted code must compute eval");
}

#[test]
fn a_null_pointer_falls_back_to_the_handler() {
    // The guard's whole purpose: r0 - base wraps to a huge value, so the load would be
    // out of range, and the handler keeps defining what a null pointer means.
    let op = InlineOp::LoadShiftMask { offset: 4, shift: 8, mask: 0xf, plus: 0 };
    let mut vm = vm_with(op);
    vm.set_reg(0, 0);
    let crossed = run(&mut vm);
    assert!(crossed, "a null pointer must reach the host");
    assert_eq!(vm.get_reg(0), HANDLER_SENTINEL, "the handler's answer must survive");
}

#[test]
fn load_scaled_shifts_inline_below_the_cap() {
    let op = InlineOp::LoadScaled { offset: 0x64, max: 4096, shl: 2 };
    let word = 100u32;
    assert!(!op.falls_back(word), "this case is the inline one");
    let mut vm = vm_with(op);
    vm.write_mem(PTR + 0x64, &word.to_le_bytes()).expect("seed the header");
    vm.set_reg(0, PTR);
    let crossed = run(&mut vm);
    assert!(!crossed, "an unclamped value must not reach the host");
    assert_eq!(vm.get_reg(0), op.eval(word), "400 = 100 * 4");
}

#[test]
fn load_scaled_hands_the_clamped_case_to_the_handler() {
    // The arm that matters. Inline, `word << shl` would be 20000; the handler's
    // `min(4096) * 4` is 16384. Answering inline here would be plausible and wrong, and
    // nothing downstream would flag it - so the guard must route it out.
    let op = InlineOp::LoadScaled { offset: 0x64, max: 4096, shl: 2 };
    let word = 5000u32;
    assert!(op.falls_back(word), "this case is the fallback one");
    let mut vm = vm_with(op);
    vm.write_mem(PTR + 0x64, &word.to_le_bytes()).expect("seed the header");
    vm.set_reg(0, PTR);
    let crossed = run(&mut vm);
    assert!(crossed, "a clamped value must reach the host");
    assert_eq!(vm.get_reg(0), HANDLER_SENTINEL, "the handler, not the shift, answers");
}

#[test]
fn load_scaled_splits_exactly_at_the_cap() {
    // The boundary itself, both sides. `max` is the last value the inline form may
    // answer; `max + 1` is the first it may not. An off-by-one here is a wrong answer
    // for exactly one input, which is the kind of thing a sampled test misses.
    let op = InlineOp::LoadScaled { offset: 0x64, max: 4096, shl: 2 };
    for (word, expect_crossing) in [(4096u32, false), (4097u32, true)] {
        let mut vm = vm_with(op);
        vm.write_mem(PTR + 0x64, &word.to_le_bytes()).expect("seed the header");
        vm.set_reg(0, PTR);
        let crossed = run(&mut vm);
        assert_eq!(crossed, expect_crossing, "at word={word}");
        assert_eq!(op.falls_back(word), expect_crossing, "eval's predicate agrees at {word}");
        if !expect_crossing {
            assert_eq!(vm.get_reg(0), op.eval(word), "at word={word}");
        }
    }
}

#[test]
fn load_mirror_reads_the_block() {
    let op = InlineOp::LoadMirror { slot: 0 };
    let mut vm = vm_with(op);
    write_mirror(&mut vm, &[0x1234_5678]);
    let crossed = run(&mut vm);
    assert!(!crossed, "a mirror read never reaches the host");
    assert_eq!(vm.get_reg(0), op.eval(0x1234_5678));
}

/// The constant-return form answers in r0 and never crosses the boundary. Both halves
/// matter and they are different claims: the value proves the guest sees what the handler
/// would have returned, and the crossing count proves the call is GONE - which is the whole
/// point of inlining a handler that computes nothing.
#[test]
fn ret_const_answers_without_reaching_the_host() {
    let op = InlineOp::RetConst { value: 0 };
    let mut vm = vm_with(op);
    // A register the emitted code must overwrite, so "r0 happened to be 0" cannot pass.
    vm.set_reg(0, 0xDEAD_BEEF);
    let crossed = run(&mut vm);
    assert!(!crossed, "a constant return never reaches the host");
    assert_eq!(vm.get_reg(0), op.eval(0), "r0 is the constant");
}

/// ...and the constant is whatever the caller named, not a hard-wired zero.
#[test]
fn ret_const_carries_its_own_value() {
    let op = InlineOp::RetConst { value: 0x0000_2A2A };
    let mut vm = vm_with(op);
    let crossed = run(&mut vm);
    assert!(!crossed, "a constant return never reaches the host");
    assert_eq!(vm.get_reg(0), 0x0000_2A2A);
}

/// The VOID twin emits no code at all, so r0 still holds what the caller passed. That is
/// the whole difference from `RetConst { value: 0 }`, and it is the difference between
/// reproducing a void handler and quietly changing what it answers.
#[test]
fn nop_leaves_r0_alone_and_never_reaches_the_host() {
    let mut vm = vm_with(InlineOp::Nop);
    vm.set_reg(0, 0xDEAD_BEEF);
    let crossed = run(&mut vm);
    assert!(!crossed, "a void no-op never reaches the host");
    assert_eq!(vm.get_reg(0), 0xDEAD_BEEF, "r0 is untouched");
}

#[test]
fn load_mirror_pair_fills_the_return_pair() {
    let op = InlineOp::LoadMirrorPair { slot: 0 };
    let mut vm = vm_with(op);
    write_mirror(&mut vm, &[0xAAAA_BBBB, 0xCCCC_DDDD]);
    let crossed = run(&mut vm);
    assert!(!crossed, "a mirror read never reaches the host");
    assert_eq!(vm.get_reg(0), 0xAAAA_BBBB, "r0 is the LOW word");
    assert_eq!(vm.get_reg(1), 0xCCCC_DDDD, "r1 is the HIGH word");
}

#[test]
fn store_mirror_pair_writes_through_the_pointer() {
    let op = InlineOp::StoreMirrorPair { slot: 0 };
    let mut vm = vm_with(op);
    write_mirror(&mut vm, &[0x0000_1111, 0x0000_2222]);
    vm.set_reg(0, OUT_PTR);
    let crossed = run(&mut vm);
    assert!(!crossed, "an in-range pointer must not reach the host");
    let got = vm.read_mem(OUT_PTR, 8).expect("read the out-parameter");
    assert_eq!(
        u32::from_le_bytes(got[0..4].try_into().unwrap()),
        0x0000_1111,
        "the low word lands at the pointer"
    );
    assert_eq!(
        u32::from_le_bytes(got[4..8].try_into().unwrap()),
        0x0000_2222,
        "the high word lands four bytes on"
    );
    assert_eq!(vm.get_reg(0), 0, "the call returns success");
}

#[test]
fn store_mirror_pair_falls_back_on_a_null_pointer() {
    let op = InlineOp::StoreMirrorPair { slot: 0 };
    let mut vm = vm_with(op);
    write_mirror(&mut vm, &[0x0000_1111, 0x0000_2222]);
    vm.set_reg(0, 0);
    let crossed = run(&mut vm);
    assert!(crossed, "a null out-parameter must reach the host");
    assert_eq!(vm.get_reg(0), HANDLER_SENTINEL);
}

// --- The argument-storing forms ---------------------------------------------------
//
// These are the setters, not the getters, and the failure they can hide is worse: a
// getter that answers wrongly at least answers, and a wrong value often shows up
// downstream. A setter that writes the WRONG OFFSET writes a real word into a real
// structure - the guest reads back a plausible number from the field next door and
// renders a picture that is subtly wrong with nothing anywhere to report. So the word
// under test is surrounded by SENTINELS, and every test asserts they survived.

/// Fill `[OUT_PTR, OUT_PTR + words * 4)` with a distinctive pattern, so a store landing at
/// the wrong offset is visible as a changed sentinel rather than as nothing at all.
fn seed_sentinels(vm: &mut Vm, words: u32) {
    for i in 0..words {
        let bytes = (SENTINEL_BASE | i).to_le_bytes();
        vm.write_mem(OUT_PTR + i * 4, &bytes).expect("seed the sentinel");
    }
}

/// Assert every word of the seeded region still holds its sentinel, except `written`
/// (a word index) which must hold `value`.
fn assert_only_wrote(vm: &mut Vm, words: u32, written: Option<(u32, u32)>) {
    for i in 0..words {
        let got = vm.read_mem(OUT_PTR + i * 4, 4).expect("read back");
        let got = u32::from_le_bytes(got[0..4].try_into().expect("4 bytes"));
        match written {
            Some((w, value)) if w == i => assert_eq!(got, value, "word {i} is the one written"),
            _ => assert_eq!(got, SENTINEL_BASE | i, "word {i} must be UNTOUCHED"),
        }
    }
}

/// The sentinel pattern's high bits; the low bits carry the word index, so a store that
/// lands one slot over is caught by value and not only by position.
const SENTINEL_BASE: u32 = 0x5EED_0000;

/// The value the guest asks the setter to store. Not a plausible sentinel.
const STORED: u32 = 0xC0FF_EE01;

#[test]
fn store_arg_writes_r1_through_the_pointer() {
    let op = InlineOp::StoreArg { offset: 8 };
    let mut vm = vm_with(op);
    seed_sentinels(&mut vm, 6);
    vm.set_reg(0, OUT_PTR);
    vm.set_reg(1, STORED);
    let crossed = run(&mut vm);
    assert!(!crossed, "an in-range pointer must not reach the host");
    assert_only_wrote(&mut vm, 6, Some((2, STORED)));
    assert_eq!(vm.get_reg(0), op.eval(0), "the call returns the handler's success code");
}

/// Offset zero is its own case: the emitted `i32.store` carries the offset as an immediate,
/// and a form whose immediate is zero would still pass a test that only ever used one
/// non-zero offset if the immediate were dropped entirely.
#[test]
fn store_arg_honours_an_offset_of_zero() {
    let mut vm = vm_with(InlineOp::StoreArg { offset: 0 });
    seed_sentinels(&mut vm, 4);
    vm.set_reg(0, OUT_PTR);
    vm.set_reg(1, STORED);
    assert!(!run(&mut vm));
    assert_only_wrote(&mut vm, 4, Some((0, STORED)));
}

#[test]
fn store_arg_falls_back_on_a_null_pointer() {
    // The arm nothing else notices. Inline, a null pointer would store to `0 - base`,
    // which wraps to an address near the top of linear memory - a real write to a real
    // page, silently corrupting whatever lives there. The guard must route it out.
    let op = InlineOp::StoreArg { offset: 8 };
    let mut vm = vm_with(op);
    seed_sentinels(&mut vm, 6);
    vm.set_reg(0, 0);
    vm.set_reg(1, STORED);
    let crossed = run(&mut vm);
    assert!(crossed, "a null pointer must reach the host");
    assert_only_wrote(&mut vm, 6, None);
    assert_eq!(vm.get_reg(0), HANDLER_SENTINEL, "the handler's answer must survive");
}

#[test]
fn store_arg_indexed_writes_the_slot_the_index_names() {
    let op = InlineOp::StoreArgIndexed { offset: 4, count: 4 };
    for index in 0..4u32 {
        let mut vm = vm_with(op);
        seed_sentinels(&mut vm, 6);
        vm.set_reg(0, OUT_PTR);
        vm.set_reg(1, index);
        vm.set_reg(2, STORED);
        let crossed = run(&mut vm);
        assert!(!crossed, "index {index} is in range and must not reach the host");
        // offset 4 = word 1, plus `index` words.
        assert_only_wrote(&mut vm, 6, Some((1 + index, STORED)));
        assert_eq!(vm.get_reg(0), 0, "the call returns success");
    }
}

/// The index bound, both sides of it. `count - 1` is the last index the inline form may
/// serve and `count` is the first it may not - and the one it may not is exactly the one
/// that would write past the end of the array, over whatever field follows.
#[test]
fn store_arg_indexed_splits_exactly_at_its_bound() {
    let op = InlineOp::StoreArgIndexed { offset: 4, count: 4 };
    for (index, expect_crossing) in [(3u32, false), (4, true), (99, true)] {
        let mut vm = vm_with(op);
        seed_sentinels(&mut vm, 6);
        vm.set_reg(0, OUT_PTR);
        vm.set_reg(1, index);
        vm.set_reg(2, STORED);
        let crossed = run(&mut vm);
        assert_eq!(crossed, expect_crossing, "at index={index}");
        assert_eq!(op.falls_back_on_index(index), expect_crossing, "the predicate agrees");
        if expect_crossing {
            assert_only_wrote(&mut vm, 6, None);
            assert_eq!(vm.get_reg(0), HANDLER_SENTINEL, "the handler answers at index={index}");
        } else {
            assert_only_wrote(&mut vm, 6, Some((1 + index, STORED)));
        }
    }
}

#[test]
fn store_arg_indexed_falls_back_on_a_null_pointer() {
    let op = InlineOp::StoreArgIndexed { offset: 4, count: 4 };
    let mut vm = vm_with(op);
    seed_sentinels(&mut vm, 6);
    vm.set_reg(0, 0);
    vm.set_reg(1, 1);
    vm.set_reg(2, STORED);
    assert!(run(&mut vm), "a null pointer must reach the host");
    assert_only_wrote(&mut vm, 6, None);
    assert_eq!(vm.get_reg(0), HANDLER_SENTINEL);
}

/// The six raw bit patterns the VFP-run tests store: a normal, a negative, a signalling
/// NaN pattern, a denormal, a large value and negative zero. Bit-exactness or nothing -
/// a form that round-tripped these through a float operation would quiet the NaN.
const VFP_BITS: [u32; 6] =
    [0x3f80_0000, 0xbf00_0000, 0x7fa0_0001, 0x0000_0001, 0x4479_c000, 0x8000_0000];

#[test]
fn store_vfp_run_writes_the_argument_registers_bits() {
    let op = InlineOp::StoreVfpRun { offset: 8, count: 6 };
    let mut vm = vm_with(op);
    seed_sentinels(&mut vm, 10);
    vm.set_reg(0, OUT_PTR);
    for (i, &b) in VFP_BITS.iter().enumerate() {
        vm.set_s(i as u8, f32::from_bits(b));
    }
    let crossed = run(&mut vm);
    assert!(!crossed, "an in-range pointer must not reach the host");
    for (i, &b) in VFP_BITS.iter().enumerate() {
        let got = vm.read_mem(OUT_PTR + 8 + i as u32 * 4, 4).expect("read back");
        let got = u32::from_le_bytes(got[0..4].try_into().expect("4 bytes"));
        assert_eq!(got, b, "run word {i} must be the raw bits of s{i}");
    }
    // The words around the run keep their sentinels: 8 bytes before, and beyond word 7.
    for i in [0u32, 1, 8, 9] {
        let got = vm.read_mem(OUT_PTR + i * 4, 4).expect("read back");
        let got = u32::from_le_bytes(got[0..4].try_into().expect("4 bytes"));
        assert_eq!(got, SENTINEL_BASE | i, "word {i} must be UNTOUCHED");
    }
    assert_eq!(vm.get_reg(0), 0, "the call returns the handler's success code");
}

#[test]
fn store_vfp_run_falls_back_on_a_null_pointer() {
    let mut vm = vm_with(InlineOp::StoreVfpRun { offset: 8, count: 6 });
    seed_sentinels(&mut vm, 10);
    vm.set_reg(0, 0);
    for (i, &b) in VFP_BITS.iter().enumerate() {
        vm.set_s(i as u8, f32::from_bits(b));
    }
    assert!(run(&mut vm), "a null pointer must reach the host");
    assert_only_wrote(&mut vm, 10, None);
    assert_eq!(vm.get_reg(0), HANDLER_SENTINEL, "the handler's answer must survive");
}

/// Where the arg-run tests park the guest stack: far from the output buffer, so a store
/// that confused the two would land on a sentinel.
const SP_PTR: u32 = BASE + 0x4000;

/// The five argument words an arg-run stores: r1..r3, then two AAPCS stack words.
const RUN_ARGS: [u32; 5] = [0xAAAA_0001, 0xBBBB_0002, 0xCCCC_0003, 0xDDDD_0004, 0xEEEE_0005];

fn seed_arg_run(vm: &mut Vm, sp: u32) {
    seed_sentinels(vm, 8);
    vm.set_reg(0, OUT_PTR);
    vm.set_reg(1, RUN_ARGS[0]);
    vm.set_reg(2, RUN_ARGS[1]);
    vm.set_reg(3, RUN_ARGS[2]);
    vm.set_reg(13, sp);
    let stack: Vec<u8> = RUN_ARGS[3..].iter().flat_map(|w| w.to_le_bytes()).collect();
    vm.write_mem(SP_PTR, &stack).expect("seed the stack words");
}

#[test]
fn store_arg_run_writes_registers_then_stack_words() {
    let op = InlineOp::StoreArgRun { offset: 4, count: 5 };
    let mut vm = vm_with(op);
    seed_arg_run(&mut vm, SP_PTR);
    let crossed = run(&mut vm);
    assert!(!crossed, "in-range pointers must not reach the host");
    for (i, &v) in RUN_ARGS.iter().enumerate() {
        let got = vm.read_mem(OUT_PTR + 4 + i as u32 * 4, 4).expect("read back");
        let got = u32::from_le_bytes(got[0..4].try_into().expect("4 bytes"));
        assert_eq!(got, v, "run word {i} must be argument {} as passed", i + 1);
    }
    for i in [0u32, 6, 7] {
        let got = vm.read_mem(OUT_PTR + i * 4, 4).expect("read back");
        let got = u32::from_le_bytes(got[0..4].try_into().expect("4 bytes"));
        assert_eq!(got, SENTINEL_BASE | i, "word {i} must be UNTOUCHED");
    }
    assert_eq!(vm.get_reg(0), 0, "the call returns the handler's success code");
}

#[test]
fn store_arg_run_falls_back_on_a_null_pointer() {
    let mut vm = vm_with(InlineOp::StoreArgRun { offset: 4, count: 5 });
    seed_arg_run(&mut vm, SP_PTR);
    vm.set_reg(0, 0);
    assert!(run(&mut vm), "a null pointer must reach the host");
    assert_only_wrote(&mut vm, 8, None);
    assert_eq!(vm.get_reg(0), HANDLER_SENTINEL);
}

/// A run that reads the guest stack must guard sp as a POINTER of its own: a garbage sp
/// would otherwise become a wild load in emitted code, where the handler's `read_u32`
/// defines the case. sp = 0 rebases to a wrap far past memory, which is exactly the
/// garbage the guard exists for.
#[test]
fn store_arg_run_falls_back_on_a_garbage_sp() {
    let mut vm = vm_with(InlineOp::StoreArgRun { offset: 4, count: 5 });
    seed_arg_run(&mut vm, 0);
    assert!(run(&mut vm), "a garbage sp must reach the host");
    assert_only_wrote(&mut vm, 8, None);
    assert_eq!(vm.get_reg(0), HANDLER_SENTINEL);
}

/// ...and a run that FITS in registers must not care what sp holds: three argument words
/// come from r1..r3, no stack word is read, and the sp guard must not have been emitted.
#[test]
fn store_arg_run_without_stack_words_ignores_sp() {
    let mut vm = vm_with(InlineOp::StoreArgRun { offset: 4, count: 3 });
    seed_arg_run(&mut vm, 0);
    let crossed = run(&mut vm);
    assert!(!crossed, "a register-only run must not consult sp");
    for (i, &v) in RUN_ARGS[..3].iter().enumerate() {
        let got = vm.read_mem(OUT_PTR + 4 + i as u32 * 4, 4).expect("read back");
        let got = u32::from_le_bytes(got[0..4].try_into().expect("4 bytes"));
        assert_eq!(got, v, "run word {i} must be argument {} as passed", i + 1);
    }
    assert_eq!(vm.get_reg(0), 0);
}

/// A small synthetic [`BindStateLayout`]: a 3-word copy, magic checks on both structures,
/// and the program-handle store on. The runtime's own layout is pinned against its
/// handlers in the runtime crate; this test proves the EMITTER computes any layout.
fn bind_layout() -> vitaslop_transpiler::BindStateLayout {
    vitaslop_transpiler::BindStateLayout {
        ctx_magic_at: 0,
        ctx_magic: 0xC0DE_C7A0,
        st_magic_at: 0,
        st_magic: 0x57A7_E001,
        st_block_at: 4,
        st_buf_at: 8,
        st_size_at: 12,
        st_header_at: 16,
        st_handle_at: 20,
        ctx_record: 4,
        copy_dst: 16,
        copy_bytes: 12,
        ctx_prog: 28,
        has_prog: true,
    }
}

/// Where the bind test parks the state struct and its arrays block.
const ST_PTR: u32 = BASE + 0x5000;
const BLK_PTR: u32 = BASE + 0x6000;

fn seed_bind(vm: &mut Vm, ctx_magic: u32, st_magic: u32) {
    let l = bind_layout();
    // The context: magic, then sentinels over everything the bind writes (words 1..8).
    seed_sentinels(vm, 8);
    vm.write_mem(OUT_PTR, &ctx_magic.to_le_bytes()).expect("ctx magic");
    // The state struct: magic, block, buf/size/header/handle.
    let st: [u32; 6] = [st_magic, BLK_PTR, 0xB0F0_0001, 0x0512_E000, 0x0EAD_E400, 0xAA55_0001];
    let bytes: Vec<u8> = st.iter().flat_map(|w| w.to_le_bytes()).collect();
    vm.write_mem(ST_PTR, &bytes).expect("state struct");
    // The arrays block: three distinctive words.
    let blk: [u32; 3] = [0x0B10_C001, 0x0B10_C002, 0x0B10_C003];
    let bytes: Vec<u8> = blk.iter().flat_map(|w| w.to_le_bytes()).collect();
    vm.write_mem(BLK_PTR, &bytes).expect("arrays block");
    vm.set_reg(0, OUT_PTR);
    vm.set_reg(1, ST_PTR);
    let _ = l;
}

#[test]
fn bind_state_copies_the_block_record_and_program() {
    let l = bind_layout();
    let mut vm = vm_with(InlineOp::BindPrecomputedState { layout: l });
    seed_bind(&mut vm, l.ctx_magic, l.st_magic);
    let crossed = run(&mut vm);
    assert!(!crossed, "both magics hold - the inline arm must serve this");
    let word = |vm: &mut Vm, at: u32| {
        let b = vm.read_mem(OUT_PTR + at, 4).expect("read back");
        u32::from_le_bytes(b[0..4].try_into().expect("4 bytes"))
    };
    // The record: buf, size, header from the struct - all three words, so a store that
    // lands one slot over cannot pass.
    assert_eq!(word(&mut vm, 4), 0xB0F0_0001, "record: buffer");
    assert_eq!(word(&mut vm, 8), 0x0512_E000, "record: size");
    assert_eq!(word(&mut vm, 12), 0x0EAD_E400, "record: header");
    // The copy: the block's three words at copy_dst.
    assert_eq!(word(&mut vm, 16), 0x0B10_C001, "copy word 0");
    assert_eq!(word(&mut vm, 20), 0x0B10_C002, "copy word 1");
    assert_eq!(word(&mut vm, 24), 0x0B10_C003, "copy word 2");
    // The program handle.
    assert_eq!(word(&mut vm, 28), 0xAA55_0001, "the program handle lands at ctx_prog");
    assert_eq!(vm.get_reg(0), 0, "the call returns the handler's success code");
}

#[test]
fn bind_state_falls_back_when_either_magic_is_wrong() {
    let l = bind_layout();
    for (cm, sm) in [(0u32, l.st_magic), (l.ctx_magic, 0)] {
        let mut vm = vm_with(InlineOp::BindPrecomputedState { layout: l });
        seed_bind(&mut vm, cm, sm);
        assert!(run(&mut vm), "a wrong magic must reach the host (ctx={cm:#x} st={sm:#x})");
        // Words 1..8 keep their sentinels - nothing of the bind may land.
        for i in 1..8u32 {
            let b = vm.read_mem(OUT_PTR + i * 4, 4).expect("read back");
            let got = u32::from_le_bytes(b[0..4].try_into().expect("4 bytes"));
            assert_eq!(got, SENTINEL_BASE | i, "word {i} must be untouched");
        }
        assert_eq!(vm.get_reg(0), HANDLER_SENTINEL);
    }
}

#[test]
fn bind_state_falls_back_on_a_null_context() {
    let l = bind_layout();
    let mut vm = vm_with(InlineOp::BindPrecomputedState { layout: l });
    seed_bind(&mut vm, l.ctx_magic, l.st_magic);
    vm.set_reg(0, 0);
    assert!(run(&mut vm), "a null context must reach the host");
    assert_eq!(vm.get_reg(0), HANDLER_SENTINEL);
}

/// A NULL STATE is the unbind, and on a real title it is most of the traffic - so it is
/// served inline. For a fragment-shaped layout (`has_prog`) the handler's null arm does
/// NOTHING but return success; the emitted arm must match, touching no context word.
#[test]
fn bind_state_null_state_is_a_pure_success_for_the_fragment_shape() {
    let l = bind_layout();
    assert!(l.has_prog, "the fixture layout is the fragment shape");
    let mut vm = vm_with(InlineOp::BindPrecomputedState { layout: l });
    seed_bind(&mut vm, l.ctx_magic, l.st_magic);
    vm.set_reg(1, 0);
    let crossed = run(&mut vm);
    assert!(!crossed, "the null unbind must not reach the host");
    for i in 1..8u32 {
        let b = vm.read_mem(OUT_PTR + i * 4, 4).expect("read back");
        let got = u32::from_le_bytes(b[0..4].try_into().expect("4 bytes"));
        assert_eq!(got, SENTINEL_BASE | i, "word {i}: a fragment null bind touches nothing");
    }
    assert_eq!(vm.get_reg(0), 0, "the call returns the handler's success code");
}

/// The vertex shape's null arm ZEROES the table and the record - behind the context
/// magic, whose failing side runs the handler.
#[test]
fn bind_state_null_state_zeroes_the_vertex_shape() {
    let l = vitaslop_transpiler::BindStateLayout { has_prog: false, ..bind_layout() };
    // Magic holds: the copy region (words 4..7) and record (words 1..3) go to zero, the
    // rest keeps its sentinels.
    let mut vm = vm_with(InlineOp::BindPrecomputedState { layout: l });
    seed_bind(&mut vm, l.ctx_magic, l.st_magic);
    vm.set_reg(1, 0);
    assert!(!run(&mut vm), "the null unbind must not reach the host");
    for i in 1..8u32 {
        let b = vm.read_mem(OUT_PTR + i * 4, 4).expect("read back");
        let got = u32::from_le_bytes(b[0..4].try_into().expect("4 bytes"));
        if (1..=6).contains(&i) {
            assert_eq!(got, 0, "word {i} is table or record and must be ZERO");
        } else {
            assert_eq!(got, SENTINEL_BASE | i, "word {i} must be untouched");
        }
    }
    assert_eq!(vm.get_reg(0), 0);
    // Magic broken: the handler owns the case (it is the side that reports no-context).
    let mut vm = vm_with(InlineOp::BindPrecomputedState { layout: l });
    seed_bind(&mut vm, 0, l.st_magic);
    vm.set_reg(1, 0);
    assert!(run(&mut vm), "a null bind through an unstamped context must reach the host");
    assert_eq!(vm.get_reg(0), HANDLER_SENTINEL);
}

/// An indexed form's POINTER guard must be computed against its LAST element, not its
/// first. A pointer that leaves room for element 0 but not element `count - 1` passes a
/// first-element bound and then stores past the end of linear memory on a high index -
/// which the engine traps rather than reports, so it looks like an engine bug.
#[test]
fn an_indexed_forms_pointer_bound_covers_its_last_element() {
    let op = InlineOp::StoreArgIndexed { offset: 4, count: 4 };
    // The last address the whole array fits at, and the first one it does not.
    let last_fits = BASE + MEM_BYTES - (4 + 3 * 4) - 4;
    for (ptr, expect_crossing) in [(last_fits, false), (last_fits + 4, true)] {
        let mut vm = vm_with(op);
        vm.set_reg(0, ptr);
        vm.set_reg(1, 3);
        vm.set_reg(2, STORED);
        let crossed = run(&mut vm);
        assert_eq!(crossed, expect_crossing, "at ptr={ptr:#x}");
        if !expect_crossing {
            let got = vm.read_mem(ptr + 4 + 3 * 4, 4).expect("the last element is in memory");
            assert_eq!(u32::from_le_bytes(got[0..4].try_into().unwrap()), STORED);
        }
    }
}

// --- The lightweight-mutex forms --------------------------------------------------
//
// These are the only forms that both READ and WRITE, and the only ones whose guard is a
// predicate over several words rather than a range check. Two things can go wrong that
// nothing else would catch:
//
//   - A wrong term in the predicate takes a lock that should have gone to the host. There
//     is no error and no crossing; two threads simply hold the same mutex, and what shows
//     up is corrupted data somewhere else entirely, frames later.
//   - A wrong OFFSET writes a real number into a real word of the work area. The mutex
//     then behaves like a different mutex, and again nothing reports it.
//
// So every case is run through the real engine, checked against `lwwork`'s own definition
// of the same decision, AND checked word for word against a host-side replay of it. The
// work area is surrounded by sentinels so a store one slot over is visible.

use vitaslop_runtime::host::GuestWords;
use vitaslop_runtime::vita::lwwork;

/// Where the test puts the mutex work area. Word 0 of the sentinel block, so
/// [`assert_only_wrote`]'s neighbours bracket it.
const WORK: u32 = OUT_PTR;
/// The thread the mirror says is running, and one that is not.
const CUR: i32 = 7;
const OTHER: i32 = 9;

/// A sparse word map, for replaying a case against [`lwwork`] outside the engine.
#[derive(Default)]
struct Words(std::collections::BTreeMap<u32, u32>);

impl GuestWords for Words {
    fn word(&self, addr: u32) -> u32 {
        self.0.get(&addr).copied().unwrap_or(0)
    }
    fn set_word(&mut self, addr: u32, value: u32) {
        self.0.insert(addr, value);
    }
}

/// The four state words of a work area, in layout order.
#[derive(Clone, Copy, Debug)]
struct State {
    id: u32,
    owner: i32,
    count: u32,
    waiters: u32,
}

impl State {
    /// A created, free mutex at [`WORK`].
    fn free() -> State {
        State { id: WORK, owner: 0, count: 0, waiters: 0 }
    }
    fn held_by(thid: i32, count: u32) -> State {
        State { id: WORK, owner: thid, count, waiters: 0 }
    }
    fn write(self, w: &mut dyn GuestWords, at: u32) {
        w.set_word(at + lwwork::off::ID, self.id);
        w.set_word(at + lwwork::off::OWNER, self.owner as u32);
        w.set_word(at + lwwork::off::COUNT, self.count);
        w.set_word(at + lwwork::off::WAITERS, self.waiters);
    }
    fn read(w: &dyn GuestWords, at: u32) -> State {
        State {
            id: w.word(at + lwwork::off::ID),
            owner: w.word(at + lwwork::off::OWNER) as i32,
            count: w.word(at + lwwork::off::COUNT),
            waiters: w.word(at + lwwork::off::WAITERS),
        }
    }
    /// Seed the work area in the VM's own memory.
    fn write_vm(self, vm: &mut Vm, at: u32) {
        let mut words = Words::default();
        self.write(&mut words, at);
        for (&addr, &value) in &words.0 {
            vm.write_mem(addr, &value.to_le_bytes()).expect("in range");
        }
    }
    /// Read it back out of the VM.
    fn read_vm(vm: &mut Vm, at: u32) -> State {
        let mut words = Words::default();
        for off in [lwwork::off::ID, lwwork::off::OWNER, lwwork::off::COUNT, lwwork::off::WAITERS] {
            words.set_word(at + off, vm_word(vm, at + off));
        }
        State::read(&words, at)
    }
}

/// One word of the VM's guest memory. `Vm::read_mem` needs `&mut`, so this cannot be a
/// [`GuestWords`] impl - hence the free function.
fn vm_word(vm: &mut Vm, addr: u32) -> u32 {
    let b = vm.read_mem(addr, 4).expect("in range");
    u32::from_le_bytes(b[0..4].try_into().expect("4 bytes"))
}

/// Run one lock-or-unlock case through the emitted code and hold it to [`lwwork`].
///
/// Returns nothing because every assertion is here: what the guest sees in r0, whether the
/// host was crossed, and every word of the work area including its neighbours.
fn check_lw(lock: bool, before: State, ptr: u32, count_arg: u32, thid: i32) {
    let op = if lock {
        InlineOp::LwMutexLock { layout: lwwork::layout(), thread_slot: 3 }
    } else {
        InlineOp::LwMutexUnlock { layout: lwwork::layout(), thread_slot: 3 }
    };
    // What the definition says should happen, replayed on a plain word map.
    let mut expect = Words::default();
    before.write(&mut expect, WORK);
    let taken = if lock {
        lwwork::fast_lock(&mut expect, WORK, thid, count_arg)
    } else {
        lwwork::fast_unlock(&mut expect, WORK, thid, count_arg)
    };
    // ...and a pointer the guard rejects is the host's case whatever the words say.
    let in_range = ptr == WORK;
    let taken = taken && in_range;

    let mut vm = vm_with(op);
    seed_sentinels(&mut vm, 6);
    before.write_vm(&mut vm, WORK);
    write_mirror(&mut vm, &[0, 0, 0, thid as u32]);
    vm.set_reg(0, ptr);
    vm.set_reg(1, count_arg);
    let crossed = run(&mut vm);

    let what = format!("{} {before:?} ptr={ptr:#x} n={count_arg} thid={thid}", if lock { "lock" } else { "unlock" });
    assert_eq!(crossed, !taken, "crossing disagrees with lwwork: {what}");
    let after = State::read_vm(&mut vm, WORK);
    if taken {
        assert_eq!(vm.get_reg(0), 0, "a served call returns success: {what}");
        let want = State::read(&expect, WORK);
        assert_eq!(after.id, want.id, "id: {what}");
        assert_eq!(after.owner, want.owner, "owner: {what}");
        assert_eq!(after.count, want.count, "count: {what}");
        assert_eq!(after.waiters, want.waiters, "waiters: {what}");
    } else {
        assert_eq!(vm.get_reg(0), HANDLER_SENTINEL, "the handler answers: {what}");
        // A refused call must leave the work area EXACTLY as it found it. A form that
        // wrote first and checked after would pass every other assertion here.
        assert_eq!(after.id, before.id, "id: {what}");
        assert_eq!(after.owner, before.owner, "owner: {what}");
        assert_eq!(after.count, before.count, "count: {what}");
        assert_eq!(after.waiters, before.waiters, "waiters: {what}");
    }
    // Words 4 and 5 are past the state; nothing may reach them.
    for i in 4..6u32 {
        let got = vm.read_mem(WORK + i * 4, 4).expect("read back");
        let got = u32::from_le_bytes(got[0..4].try_into().expect("4 bytes"));
        assert_eq!(got, SENTINEL_BASE | i, "word {i} past the state must be UNTOUCHED: {what}");
    }
}

/// The take, over every shape of work area, against the definition.
#[test]
fn lw_mutex_lock_matches_lwwork_on_every_arm() {
    for &before in &[
        // Served inline: free, free with a stale owner, and the owner recursing.
        State::free(),
        State { owner: OTHER, ..State::free() },
        State::held_by(CUR, 1),
        State::held_by(CUR, 5),
        // The host's: held by somebody else, a parked waiter, and a work area that is a
        // COPY (its id names the original) or was never created (id zero).
        State::held_by(OTHER, 1),
        State { waiters: 1, ..State::free() },
        State { id: WORK + 0x100, ..State::free() },
        State { id: 0, ..State::free() },
    ] {
        check_lw(true, before, WORK, 1, CUR);
    }
}

#[test]
fn lw_mutex_unlock_matches_lwwork_on_every_arm() {
    for &before in &[
        // Served inline: the owner releasing, once and one of several.
        State::held_by(CUR, 1),
        State::held_by(CUR, 3),
        // The host's: nothing held, held by somebody else, a parked waiter to hand it to,
        // a copy, a work area never created.
        State::free(),
        // A DOUBLE unlock by the last owner, which is the case the owner word alone
        // cannot see: releasing leaves `owner` stale, so this reads as "mine" and is
        // refused only by the count. Inline, `count - 1` would wrap it to four billion
        // and the mutex would be permanently, invisibly held.
        State::held_by(CUR, 0),
        State::held_by(OTHER, 1),
        State { waiters: 1, ..State::held_by(CUR, 1) },
        State { id: WORK + 0x100, ..State::held_by(CUR, 1) },
        State { id: 0, ..State::held_by(CUR, 1) },
    ] {
        check_lw(false, before, WORK, 1, CUR);
    }
}

/// The lock/unlock COUNT argument. Only 1 is served; the handler defines the rest,
/// including the illegal zero. Folding a multi-count acquire into `count += 1` would
/// under-count the recursion and release the mutex while the guest still believed it held
/// it - a data race with no error anywhere.
#[test]
fn a_count_argument_other_than_one_reaches_the_handler() {
    for n in [0u32, 2, 3, 0xFFFF_FFFF] {
        check_lw(true, State::free(), WORK, n, CUR);
        check_lw(false, State::held_by(CUR, 4), WORK, n, CUR);
    }
}

/// THREAD ZERO is the main thread by convention, so it is a real owner and not a sentinel.
/// This is the case an implementation that spelled "free" as `owner == 0` gets wrong: the
/// main thread would find every mutex it holds indistinguishable from a free one, recurse
/// where it should have counted, and release early.
#[test]
fn the_main_thread_is_a_real_owner_not_a_free_marker() {
    check_lw(true, State::held_by(0, 1), WORK, 1, 0); // recursive take, count 1 -> 2
    check_lw(false, State::held_by(0, 1), WORK, 1, 0); // release, count 1 -> 0
    // ...and thread 0 must NOT be able to take a mutex thread 9 holds, however its owner
    // word reads.
    check_lw(true, State::held_by(OTHER, 1), WORK, 1, 0);
    check_lw(false, State::held_by(OTHER, 1), WORK, 1, 0);
}

/// The pointer guard, on the arm that matters. Inline, a null work pointer would read and
/// WRITE at `0 - base`, which wraps near the top of linear memory: a real store into a real
/// page, and the guest would believe it holds a lock that lives in somebody else's data.
#[test]
fn a_null_work_pointer_reaches_the_handler() {
    check_lw(true, State::free(), 0, 1, CUR);
    check_lw(false, State::held_by(CUR, 1), 0, 1, CUR);
}

/// The pointer bound must cover the LAST word of the layout. A bound computed for the id
/// word alone would admit a pointer with room for one word and then read three past the end
/// of guest memory - which the engine traps, so it presents as an engine bug rather than as
/// this.
#[test]
fn the_pointer_bound_covers_the_whole_layout() {
    let op = InlineOp::LwMutexLock { layout: lwwork::layout(), thread_slot: 3 };
    let last_fits = BASE + MEM_BYTES - lwwork::BYTES;
    for (ptr, expect_crossing) in [(last_fits, false), (last_fits + 4, true)] {
        let mut vm = vm_with(op);
        write_mirror(&mut vm, &[0, 0, 0, CUR as u32]);
        // A canonical, free mutex right at the boundary.
        State { id: ptr, owner: 0, count: 0, waiters: 0 }.write_vm(&mut vm, ptr);
        vm.set_reg(0, ptr);
        vm.set_reg(1, 1);
        let crossed = run(&mut vm);
        assert_eq!(crossed, expect_crossing, "at ptr={ptr:#x}");
        if !expect_crossing {
            assert_eq!(vm_word(&mut vm, ptr + lwwork::off::COUNT), 1, "it was taken");
        }
    }
}

/// A lock form must read the CURRENT THREAD from the mirror slot it names, not from a
/// fixed one. Two slots holding different ids, one op each: an emitter that ignored
/// `thread_slot` would take both locks for the same thread and pass every test above.
#[test]
fn a_lock_form_reads_the_thread_slot_it_names() {
    for (slot, thid) in [(2u32, 11i32), (3, 22)] {
        let op = InlineOp::LwMutexLock { layout: lwwork::layout(), thread_slot: slot };
        let mut vm = vm_with(op);
        // Every slot holds a DIFFERENT id, so reading the wrong one is visible.
        write_mirror(&mut vm, &[99, 98, 11, 22]);
        State::free().write_vm(&mut vm, WORK);
        vm.set_reg(0, WORK);
        vm.set_reg(1, 1);
        assert!(!run(&mut vm), "a free canonical mutex is taken inline");
        assert_eq!(
            vm_word(&mut vm, WORK + lwwork::off::OWNER) as i32,
            thid,
            "slot {slot} names thread {thid}"
        );
    }
}

/// The mirror block must be sized from the TOP slot a pair form touches. Sized from the
/// base slot, a pair's high word would read the page above the block - which for a clock
/// is a garbage timestamp, not a fault, so nothing would report it.
#[test]
fn a_pair_form_reserves_both_of_its_slots() {
    let base_only = InlineOp::LoadMirror { slot: 3 };
    let pair = InlineOp::LoadMirrorPair { slot: 3 };
    assert_eq!(base_only.top_mirror_slot(), Some(3));
    assert_eq!(pair.top_mirror_slot(), Some(4), "a pair reaches one slot further");
    assert_eq!(pair.mirror_slot(), Some(3), "...but still BASES at its own slot");
}

// --- the bias, and the field store ----------------------------------------------------

/// The `plus` bias must be ADDED after the shift and mask, not folded into either. The case
/// that separates the two: a field of all ones plus one, which carries out of the field.
#[test]
fn load_shift_mask_adds_its_bias_after_the_mask() {
    let op = InlineOp::LoadShiftMask { offset: 4, shift: 12, mask: 0xfff, plus: 1 };
    for word in [0x00FF_F000u32, 0xFFFF_FFFF, 0x0000_0000, 0xABCD_1234] {
        let mut vm = vm_with(op);
        vm.write_mem(PTR + 4, &word.to_le_bytes()).expect("seed the struct");
        vm.set_reg(0, PTR);
        assert!(!run(&mut vm), "an in-range pointer must not reach the host");
        assert_eq!(vm.get_reg(0), op.eval(word), "word {word:#x}");
    }
}

/// A bias of zero must emit NO add. Not a style point: every inline form that existed before
/// the bias did carries `plus: 0`, and a build that emitted a `+ 0` for each of them would
/// change the fuel every guest call to them costs - which moves the browser's clock and every
/// preemption point with it, for nothing.
#[test]
fn a_zero_bias_is_the_unbiased_form_exactly() {
    let op = InlineOp::LoadShiftMask { offset: 4, shift: 8, mask: 0xf, plus: 0 };
    let word = 0xABCD_1234u32;
    let mut vm = vm_with(op);
    vm.write_mem(PTR + 4, &word.to_le_bytes()).expect("seed the struct");
    vm.set_reg(0, PTR);
    assert!(!run(&mut vm));
    assert_eq!(vm.get_reg(0), (word >> 8) & 0xf, "the answer is the field itself");
    assert_eq!(op.eval(word), (word >> 8) & 0xf, "...and `eval` agrees");
}

/// A field store must leave every bit outside its field exactly as it found it. This is the
/// whole reason the form exists - the word it writes packs eight independent settings, and a
/// whole-word store would clear seven of them with nothing to report it.
#[test]
fn store_arg_field_rewrites_only_its_field() {
    // magFilter's shape: two bits at 12.
    let op = InlineOp::StoreArgField { offset: 4, shift: 12, mask: 0x3 };
    // A word with EVERY other bit set, so any bit the form clears is visible.
    const BEFORE: u32 = 0xFFFF_FFFF;
    for value in [0u32, 1, 2, 3, 0xFFFF_FFFF] {
        let mut vm = vm_with(op);
        seed_sentinels(&mut vm, 6);
        vm.write_mem(OUT_PTR + 4, &BEFORE.to_le_bytes()).expect("seed the word");
        vm.set_reg(0, OUT_PTR);
        vm.set_reg(1, value);
        assert!(!run(&mut vm), "an in-range pointer must not reach the host");
        let got = vm.read_mem(OUT_PTR + 4, 4).expect("read back");
        let got = u32::from_le_bytes(got[0..4].try_into().expect("4 bytes"));
        assert_eq!(got, (BEFORE & !(0x3 << 12)) | ((value & 0x3) << 12), "value {value:#x}");
        assert_eq!(vm.get_reg(0), 0, "a void setter returns the success code");
        // ...and not one word either side of it moved.
        assert_only_wrote(&mut vm, 6, Some((1, got)));
    }
}

/// The same, starting from a word of all ZEROES, so a form that OR-ed without clearing first
/// passes the test above (every bit was already set) and fails here.
#[test]
fn store_arg_field_clears_before_it_writes() {
    let op = InlineOp::StoreArgField { offset: 0, shift: 6, mask: 0x7 };
    let mut vm = vm_with(op);
    vm.write_mem(OUT_PTR, &0xFFFF_FFFFu32.to_le_bytes()).expect("seed the word");
    vm.set_reg(0, OUT_PTR);
    vm.set_reg(1, 0);
    assert!(!run(&mut vm));
    let got = vm.read_mem(OUT_PTR, 4).expect("read back");
    let got = u32::from_le_bytes(got[0..4].try_into().expect("4 bytes"));
    assert_eq!(got, 0xFFFF_FFFF & !(0x7 << 6), "writing zero must CLEAR the field");
}

/// A field store READS the word before writing it, so an out-of-range pointer would load
/// garbage and store it back - which is worse than the plain store's failure, not better.
#[test]
fn store_arg_field_falls_back_on_a_null_pointer() {
    let op = InlineOp::StoreArgField { offset: 4, shift: 12, mask: 0x3 };
    let mut vm = vm_with(op);
    seed_sentinels(&mut vm, 6);
    vm.set_reg(0, 0);
    vm.set_reg(1, 2);
    assert!(run(&mut vm), "a null pointer must reach the host");
    assert_only_wrote(&mut vm, 6, None);
    assert_eq!(vm.get_reg(0), HANDLER_SENTINEL, "the handler's answer must survive");
}

// --- the bulk forms -------------------------------------------------------------------
//
// These are the only forms whose reach is chosen by the GUEST, so they are the only ones
// whose guard is arithmetic rather than a constant, and the only ones that can walk off the
// end of linear memory by being handed a large enough length with a perfectly ordinary
// pointer. Both facts are tested below, on both arms.

/// Where a bulk test puts its source buffer. Far enough from [`OUT_PTR`] that an overrun in
/// either direction lands on sentinels rather than on the other buffer.
const SRC_PTR: u32 = BASE + 0x4000;

/// A pattern that is not a sentinel, not zero and not constant along its length, so a copy
/// that transposes, truncates or repeats bytes is visible in the result.
fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i as u8).wrapping_mul(37).wrapping_add(0x5b)).collect()
}

/// The last address at which `len` bytes still fit in guest memory, rebased to a pointer.
/// The bulk guard's boundary is exactly here, and a form that computed it from the pointer
/// alone would admit the address one past it.
fn last_fitting_ptr(len: u32) -> u32 {
    BASE + MEM_BYTES - len
}

#[test]
fn mem_copy_moves_the_bytes_and_returns_the_destination() {
    let src = pattern(300);
    let mut vm = vm_with(InlineOp::MemCopy);
    seed_sentinels(&mut vm, 128);
    vm.write_mem(SRC_PTR, &src).expect("seed the source");
    vm.set_reg(0, OUT_PTR);
    vm.set_reg(1, SRC_PTR);
    vm.set_reg(2, src.len() as u32);
    assert!(!run(&mut vm), "an in-range copy must not reach the host");
    assert_eq!(vm.read_mem(OUT_PTR, src.len()).expect("read back"), src);
    assert_eq!(vm.get_reg(0), OUT_PTR, "memcpy returns its destination");
    // ...and stopped exactly there. A length the emitted code rounded up would overwrite
    // the sentinel that follows, which nothing else in the assertion above would notice.
    let past = vm.read_mem(OUT_PTR + src.len() as u32, 4).expect("read back");
    let past = u32::from_le_bytes(past[0..4].try_into().expect("4 bytes"));
    assert_eq!(past, SENTINEL_BASE | 75, "the byte after the copy is UNTOUCHED");
}

/// A zero length is a legal call, not an error, and it is the case a range computed as
/// `addr + len - 1` gets catastrophically wrong: the subtraction wraps, and a form that
/// stamped or filled that range would touch the whole of memory.
#[test]
fn mem_copy_of_zero_bytes_writes_nothing() {
    let mut vm = vm_with(InlineOp::MemCopy);
    seed_sentinels(&mut vm, 8);
    vm.write_mem(SRC_PTR, &pattern(64)).expect("seed the source");
    vm.set_reg(0, OUT_PTR);
    vm.set_reg(1, SRC_PTR);
    vm.set_reg(2, 0);
    assert!(!run(&mut vm), "a zero-length copy is still in range");
    assert_only_wrote(&mut vm, 8, None);
    assert_eq!(vm.get_reg(0), OUT_PTR);
}

/// An OVERLAPPING copy. The handler reads the whole source before writing, so it moves the
/// ORIGINAL bytes; `memory.copy` is specified the same way. A form built on a
/// forward byte loop would agree with it on every non-overlapping case and disagree here.
#[test]
fn mem_copy_moves_overlapping_bytes_as_the_handler_does() {
    let src = pattern(64);
    let mut vm = vm_with(InlineOp::MemCopy);
    vm.write_mem(OUT_PTR, &src).expect("seed the buffer");
    // Copy forward over itself by 8 bytes: dst[i] = src[i - 8].
    vm.set_reg(0, OUT_PTR + 8);
    vm.set_reg(1, OUT_PTR);
    vm.set_reg(2, 56);
    assert!(!run(&mut vm));
    let got = vm.read_mem(OUT_PTR, 64).expect("read back");
    let mut want = src.clone();
    want.copy_within(0..56, 8);
    assert_eq!(got, want, "the copy must see the ORIGINAL source bytes");
}

/// The length is part of the bound, and this is the case that proves it: a pointer that is
/// perfectly ordinary on its own, with a length that reaches past the end of memory. A guard
/// on the pointers alone admits it, and the access then traps in the engine rather than
/// reaching the handler - which reads as a transpiler bug, not as a guest one.
#[test]
fn mem_copy_falls_back_when_the_length_runs_off_the_end() {
    for (dst, src_ptr, len, expect_crossing) in [
        // The last length that fits at this pointer, and the first that does not.
        (last_fitting_ptr(64), SRC_PTR, 64u32, false),
        (last_fitting_ptr(64), SRC_PTR, 65, true),
        // The same boundary on the SOURCE, with an unimpeachable destination.
        (OUT_PTR, last_fitting_ptr(64), 64, false),
        (OUT_PTR, last_fitting_ptr(64), 65, true),
        // A length larger than memory itself, which is what makes `mem_bytes - len` wrap.
        (OUT_PTR, SRC_PTR, MEM_BYTES + 1, true),
        (OUT_PTR, SRC_PTR, u32::MAX, true),
        // ...and the null pointer, on each side in turn.
        (0, SRC_PTR, 16, true),
        (OUT_PTR, 0, 16, true),
    ] {
        let mut vm = vm_with(InlineOp::MemCopy);
        vm.set_reg(0, dst);
        vm.set_reg(1, src_ptr);
        vm.set_reg(2, len);
        assert_eq!(
            run(&mut vm),
            expect_crossing,
            "dst={dst:#x} src={src_ptr:#x} len={len}"
        );
    }
}

#[test]
fn mem_fill_writes_the_low_byte_of_r1() {
    // A value whose low byte is what must land and whose high bytes must not: a form that
    // stored the whole word, or masked the wrong end, fails here and nowhere else.
    let mut vm = vm_with(InlineOp::MemFill);
    seed_sentinels(&mut vm, 16);
    vm.set_reg(0, OUT_PTR);
    vm.set_reg(1, 0xDEAD_BE5A);
    vm.set_reg(2, 20);
    assert!(!run(&mut vm), "an in-range fill must not reach the host");
    assert_eq!(vm.read_mem(OUT_PTR, 20).expect("read back"), vec![0x5Au8; 20]);
    assert_eq!(vm.get_reg(0), OUT_PTR, "memset returns its destination");
    // Byte 20 onward is still the sentinel's third byte, so the fill stopped where asked.
    let tail = vm.read_mem(OUT_PTR + 20, 4).expect("read back");
    let tail = u32::from_le_bytes(tail[0..4].try_into().expect("4 bytes"));
    assert_eq!(tail, SENTINEL_BASE | 5, "the word after the fill is UNTOUCHED");
}

#[test]
fn mem_fill_falls_back_when_the_length_runs_off_the_end() {
    for (dst, len, expect_crossing) in [
        (last_fitting_ptr(32), 32u32, false),
        (last_fitting_ptr(32), 33, true),
        (OUT_PTR, MEM_BYTES + 1, true),
        (0, 16, true),
    ] {
        let mut vm = vm_with(InlineOp::MemFill);
        vm.set_reg(0, dst);
        vm.set_reg(1, 0);
        vm.set_reg(2, len);
        assert_eq!(run(&mut vm), expect_crossing, "dst={dst:#x} len={len}");
    }
}

/// The emitted loop against [`vitaslop_transpiler::mem_compare`], which is the same function
/// the host handler calls - so this is the definition, not a second opinion of it.
///
/// The cases are chosen to separate the ways a byte loop goes wrong: a difference in the
/// FIRST byte (an off-by-one entry test skips it), in the LAST (an off-by-one exit test
/// skips it), a difference past the length (it must NOT be seen), and a pair whose
/// difference is negative one way and positive the other, which a sign-extended load would
/// get backwards.
#[test]
fn mem_compare_computes_the_shared_definition() {
    let cases: [(&[u8], &[u8]); 8] = [
        (b"abc", b"abc"),
        (b"abc", b"abd"),
        (b"abc", b"abb"),
        (b"xbc", b"abc"),
        (b"abc", b"xbc"),
        (b"", b""),
        (&[0x01, 0x02, 0xff], &[0x01, 0x02, 0x01]),
        (&[0x01, 0x02, 0x01], &[0x01, 0x02, 0xff]),
    ];
    for (a, b) in cases {
        let mut vm = vm_with(InlineOp::MemCompare);
        vm.write_mem(OUT_PTR, a).expect("seed a");
        vm.write_mem(SRC_PTR, b).expect("seed b");
        vm.set_reg(0, OUT_PTR);
        vm.set_reg(1, SRC_PTR);
        vm.set_reg(2, a.len() as u32);
        assert!(!run(&mut vm), "an in-range compare must not reach the host");
        assert_eq!(
            vm.get_reg(0) as i32,
            vitaslop_transpiler::mem_compare(a, b),
            "comparing {a:?} against {b:?}"
        );
    }
}

/// A difference PAST the length must not be reported. This is the exit test, and getting it
/// wrong produces an answer that is correct on almost every input a title supplies - two
/// buffers that differ somewhere are the common case, and the extra byte the loop read is
/// usually equal too.
#[test]
fn mem_compare_stops_at_the_length() {
    let mut vm = vm_with(InlineOp::MemCompare);
    vm.write_mem(OUT_PTR, &[1, 2, 3, 4]).expect("seed a");
    vm.write_mem(SRC_PTR, &[1, 2, 3, 99]).expect("seed b");
    vm.set_reg(0, OUT_PTR);
    vm.set_reg(1, SRC_PTR);
    vm.set_reg(2, 3);
    assert!(!run(&mut vm));
    assert_eq!(vm.get_reg(0), 0, "the fourth byte is past the length and must not be read");
}

/// A compare writes nothing. Obvious, and worth an assertion anyway: it shares its guard and
/// its locals with two forms that DO write, and the shared code is where a stray store
/// would come from.
#[test]
fn mem_compare_writes_nothing() {
    let mut vm = vm_with(InlineOp::MemCompare);
    seed_sentinels(&mut vm, 8);
    vm.write_mem(SRC_PTR, &pattern(32)).expect("seed b");
    vm.set_reg(0, OUT_PTR);
    vm.set_reg(1, SRC_PTR);
    vm.set_reg(2, 32);
    assert!(!run(&mut vm));
    assert_only_wrote(&mut vm, 8, None);
}

#[test]
fn mem_compare_falls_back_when_the_length_runs_off_the_end() {
    for (a, b, len, expect_crossing) in [
        (last_fitting_ptr(16), SRC_PTR, 16u32, false),
        (last_fitting_ptr(16), SRC_PTR, 17, true),
        (OUT_PTR, last_fitting_ptr(16), 16, false),
        (OUT_PTR, last_fitting_ptr(16), 17, true),
        (OUT_PTR, SRC_PTR, MEM_BYTES + 1, true),
        (0, SRC_PTR, 8, true),
        (OUT_PTR, 0, 8, true),
    ] {
        let mut vm = vm_with(InlineOp::MemCompare);
        vm.set_reg(0, a);
        vm.set_reg(1, b);
        vm.set_reg(2, len);
        assert_eq!(run(&mut vm), expect_crossing, "a={a:#x} b={b:#x} len={len}");
        if expect_crossing {
            assert_eq!(vm.get_reg(0), HANDLER_SENTINEL, "the handler's answer survives");
        }
    }
}

// --- the dirty stamp a bulk WRITE owes -------------------------------------------------

/// Build a VM for `op` with guest-store tracking ON, and hand back the guest address of the
/// dirty block.
///
/// Tracking is a per-thread emit-time setting, so it is turned on around the build and off
/// again immediately: leaving it on would change every module a later test on this thread
/// emits, and the symptom would be a byte-count assertion failing in a test that never
/// mentions dirty pages.
fn vm_with_dirty(op: InlineOp) -> (Vm, u32) {
    vitaslop_transpiler::set_dirty_tracking(true);
    let vm = vm_with(op);
    vitaslop_transpiler::set_dirty_tracking(false);
    let off = vm.dirty_off().expect("a tracked build reserves the dirty block");
    (vm, BASE + off as u32)
}

/// Read page `page`'s stamp out of the map.
fn stamp_of(vm: &mut Vm, block: u32, page: u32) -> u8 {
    let at = block + vitaslop_transpiler::DIRTY_MAP_OFF as u32 + page;
    vm.read_mem(at, 1).expect("the map is inside linear memory")[0]
}

/// Seed the epoch the guest will stamp with, and clear the pages the test looks at.
fn seed_epoch(vm: &mut Vm, block: u32, epoch: u8, pages: std::ops::Range<u32>) {
    vm.write_mem(block + vitaslop_transpiler::DIRTY_EPOCH_OFF as u32, &[epoch])
        .expect("seed the epoch");
    for p in pages {
        let at = block + vitaslop_transpiler::DIRTY_MAP_OFF as u32 + p;
        vm.write_mem(at, &[0]).expect("clear the stamp");
    }
}

/// A bulk write must stamp EVERY page it touches, not just the one it starts in.
///
/// This is the assertion the whole `emit_dirty_range` helper exists for, and the failure it
/// catches is silent by construction: a copy that stamps only its first page leaves the host
/// believing the rest of a texture is exactly as it last read it, so the frame draws from
/// bytes the guest has since replaced, with nothing reported anywhere
/// ([[vitaslop-guest-store-stamps]]).
#[test]
fn a_bulk_copy_stamps_every_page_it_spans() {
    const EPOCH: u8 = 0x2a;
    let shift = vitaslop_transpiler::DIRTY_SHIFT;
    // A destination part-way into a page, long enough to span three of them, so a form that
    // stamped the first page, the last page, or a page count off by one all fail here.
    let dst = OUT_PTR;
    let len = (2 << shift) + 100;
    let first = (dst - BASE) >> shift;
    let last = (dst - BASE + len - 1) >> shift;
    assert_eq!(last - first, 2, "the fixture must really span three pages");

    let (mut vm, block) = vm_with_dirty(InlineOp::MemCopy);
    seed_epoch(&mut vm, block, EPOCH, first.saturating_sub(1)..last + 2);
    vm.write_mem(SRC_PTR, &pattern(len as usize)).expect("seed the source");
    vm.set_reg(0, dst);
    vm.set_reg(1, SRC_PTR);
    vm.set_reg(2, len);
    assert!(!run(&mut vm), "an in-range copy is still served inline when tracking is on");
    for page in first..=last {
        assert_eq!(stamp_of(&mut vm, block, page), EPOCH, "page {page} is in the copy");
    }
    // ...and stops. An over-wide stamp is only a lost optimisation, but it is also the
    // signature of a page count computed from the wrong end.
    assert_eq!(stamp_of(&mut vm, block, last + 1), 0, "the page after the copy is untouched");
}

#[test]
fn a_bulk_fill_stamps_every_page_it_spans() {
    const EPOCH: u8 = 0x71;
    let shift = vitaslop_transpiler::DIRTY_SHIFT;
    let len = (1 << shift) + 1;
    let first = (OUT_PTR - BASE) >> shift;
    let last = (OUT_PTR - BASE + len - 1) >> shift;
    let (mut vm, block) = vm_with_dirty(InlineOp::MemFill);
    seed_epoch(&mut vm, block, EPOCH, first..last + 2);
    vm.set_reg(0, OUT_PTR);
    vm.set_reg(1, 0xff);
    vm.set_reg(2, len);
    assert!(!run(&mut vm));
    for page in first..=last {
        assert_eq!(stamp_of(&mut vm, block, page), EPOCH, "page {page} is in the fill");
    }
    assert_eq!(stamp_of(&mut vm, block, last + 1), 0, "the page after the fill is untouched");
}

/// A ZERO-length write stamps nothing. `last = addr + len - 1` underflows here, and a range
/// computed without the guard would ask `memory.fill` for four billion bytes - which traps,
/// turning a legal `memcpy(d, s, 0)` into a dead guest.
#[test]
fn a_zero_length_bulk_write_stamps_nothing() {
    const EPOCH: u8 = 0x5c;
    let first = (OUT_PTR - BASE) >> vitaslop_transpiler::DIRTY_SHIFT;
    let (mut vm, block) = vm_with_dirty(InlineOp::MemFill);
    seed_epoch(&mut vm, block, EPOCH, first..first + 2);
    vm.set_reg(0, OUT_PTR);
    vm.set_reg(1, 0xff);
    vm.set_reg(2, 0);
    assert!(!run(&mut vm), "a zero-length fill is in range and must not trap");
    assert_eq!(stamp_of(&mut vm, block, first), 0, "no page was written, so none is stamped");
}

/// A compare writes nothing and therefore stamps nothing. A stamp here is not merely
/// wasteful: it would report a texture as overwritten every time a title compared it,
/// which is a re-upload of every read-only texture on every frame that looks at one.
#[test]
fn a_bulk_compare_stamps_nothing() {
    const EPOCH: u8 = 0x13;
    let first = (OUT_PTR - BASE) >> vitaslop_transpiler::DIRTY_SHIFT;
    let (mut vm, block) = vm_with_dirty(InlineOp::MemCompare);
    seed_epoch(&mut vm, block, EPOCH, first..first + 2);
    vm.write_mem(OUT_PTR, &pattern(64)).expect("seed a");
    vm.write_mem(SRC_PTR, &pattern(64)).expect("seed b");
    vm.set_reg(0, OUT_PTR);
    vm.set_reg(1, SRC_PTR);
    vm.set_reg(2, 64);
    assert!(!run(&mut vm));
    assert_eq!(vm.get_reg(0), 0, "identical buffers compare equal");
    assert_eq!(stamp_of(&mut vm, block, first), 0, "a read stamps nothing");
}

// --- InlineOp::ReserveUniformBuffer ---------------------------------------------------
//
// The form that HANDS THE GUEST AN ADDRESS rather than answering a question, so its
// failures are a class the others do not have: a wrong bump gives two draws the same
// buffer (one object's uniforms on another) and a bump that escapes the ring gives the
// guest a pointer into someone else's memory. Both are silent. Hence an execution test
// per arm rather than only a layout assertion.

/// The context block the reserve bumps, and the program handle it reads its size from.
/// Placed away from [`OUT_PTR`], which here is the `void **uniformBuffer` out-parameter.
const CTX_PTR: u32 = BASE + 0x4000;
const PROG_PTR: u32 = BASE + 0x5000;
/// The ring the bump hands blocks out of.
const RING: u32 = BASE + 0x8000;
const RING_BYTES: u32 = 0x1000;

const CTX_MAGIC: u32 = 0x5658_4354;
const PROG_MAGIC: u32 = 0x5658_5047;
/// What one reserve asks for, and what the handle says to record for it. Deliberately
/// different numbers, because they are different fields and a form that confused them
/// would pass a test that used one value for both.
const U_SIZE: u32 = 0x30;
const U_ALLOC: u32 = 0x100;

/// The layout the runtime passes for the VERTEX stage, spelled out here rather than
/// imported: this test is about what the EMITTER does with a layout, and reading the
/// runtime's own constants would make it agree with itself.
fn reserve_layout() -> vitaslop_transpiler::UniformRingLayout {
    vitaslop_transpiler::UniformRingLayout {
        ctx_magic_at: 0x00,
        ctx_magic: CTX_MAGIC,
        ctx_program: 0x88,
        ctx_ring_base: 0x10,
        ctx_ring_end: 0x14,
        ctx_ring_cursor: 0x18,
        record: 0x20,
        prog_magic_at: 0x00,
        prog_magic: PROG_MAGIC,
        prog_size: 0x04,
        prog_alloc: 0x08,
        prog_header: 0x0c,
        align: 16,
    }
}

const PROG_HEADER: u32 = 0x8123_4567;

/// A VM with a stamped context block (ring attached, cursor at `cursor`) and a stamped
/// program handle bound into it.
fn vm_with_reserve(cursor: u32) -> Vm {
    let l = reserve_layout();
    let mut vm = vm_with(InlineOp::ReserveUniformBuffer { layout: l });
    let mut w = |addr: u32, v: u32| vm.write_mem(addr, &v.to_le_bytes()).expect("seed");
    w(CTX_PTR + l.ctx_magic_at, CTX_MAGIC);
    w(CTX_PTR + l.ctx_program, PROG_PTR);
    w(CTX_PTR + l.ctx_ring_base, RING);
    w(CTX_PTR + l.ctx_ring_end, RING + RING_BYTES);
    w(CTX_PTR + l.ctx_ring_cursor, cursor);
    for k in 0..3 {
        w(CTX_PTR + l.record + k * 4, SENTINEL_BASE | k);
    }
    w(PROG_PTR + l.prog_magic_at, PROG_MAGIC);
    w(PROG_PTR + l.prog_size, U_SIZE);
    w(PROG_PTR + l.prog_alloc, U_ALLOC);
    w(PROG_PTR + l.prog_header, PROG_HEADER);
    vm.write_mem(OUT_PTR, &SENTINEL_BASE.to_le_bytes()).expect("seed the out-parameter");
    vm
}

fn word(vm: &mut Vm, addr: u32) -> u32 {
    let b = vm.read_mem(addr, 4).expect("read back");
    u32::from_le_bytes(b[0..4].try_into().expect("4 bytes"))
}

/// The ordinary case: bump the cursor, hand the block back, record all three words.
#[test]
fn reserve_bumps_the_ring_and_records_the_binding() {
    let l = reserve_layout();
    let mut vm = vm_with_reserve(RING);
    vm.set_reg(0, CTX_PTR);
    vm.set_reg(1, OUT_PTR);
    assert!(!run(&mut vm), "a fully seeded context must not reach the host");
    assert_eq!(word(&mut vm, OUT_PTR), RING, "the guest is handed the block");
    assert_eq!(word(&mut vm, CTX_PTR + l.ctx_ring_cursor), RING + U_ALLOC, "the cursor advances");
    assert_eq!(word(&mut vm, CTX_PTR + l.record), RING, "...the record names it");
    assert_eq!(word(&mut vm, CTX_PTR + l.record + 4), U_SIZE, "...at its RECORDED size");
    assert_eq!(word(&mut vm, CTX_PTR + l.record + 8), PROG_HEADER, "...for its program");
    assert_eq!(vm.get_reg(0), 0, "the call returns the handler's success code");
}

/// Two reserves in a row must not alias, and the second must start ALIGNED. This is the
/// failure that puts one object's model matrix on another, and the one a single-call test
/// cannot see at all.
#[test]
fn two_reserves_hand_out_distinct_aligned_blocks() {
    let l = reserve_layout();
    // A cursor deliberately off an alignment boundary, which is what an unaligned size
    // would leave behind.
    let mut vm = vm_with_reserve(RING + 4);
    vm.set_reg(0, CTX_PTR);
    vm.set_reg(1, OUT_PTR);
    assert!(!run(&mut vm));
    let first = word(&mut vm, OUT_PTR);
    assert_eq!(first, RING + 16, "the block starts at the next 16-byte boundary");
    vm.set_reg(0, CTX_PTR);
    vm.set_reg(1, OUT_PTR);
    assert!(!run(&mut vm));
    let second = word(&mut vm, OUT_PTR);
    assert_eq!(second, first + U_ALLOC, "the second block starts past the first");
    assert_eq!(second % l.align, 0, "and is aligned");
    assert!(second - first >= U_SIZE, "the two do not overlap");
}

/// A reserve that would leave the ring is the handler's: it WRAPS and reports the
/// aliasing, and inlining that would make a real fidelity loss silent.
#[test]
fn reserve_hands_an_overrun_to_the_handler() {
    let l = reserve_layout();
    let mut vm = vm_with_reserve(RING + RING_BYTES - U_ALLOC + 16);
    vm.set_reg(0, CTX_PTR);
    vm.set_reg(1, OUT_PTR);
    assert!(run(&mut vm), "a reserve that does not fit must reach the host");
    assert_eq!(word(&mut vm, OUT_PTR), SENTINEL_BASE, "and must write nothing itself");
    assert_eq!(word(&mut vm, CTX_PTR + l.record), SENTINEL_BASE, "...including the record");
}

/// The last block that FITS must still be handed out inline. The boundary is where an
/// off-by-one lives, and either side of it is a different program.
#[test]
fn reserve_splits_exactly_at_the_end_of_the_ring() {
    let mut vm = vm_with_reserve(RING + RING_BYTES - U_ALLOC);
    vm.set_reg(0, CTX_PTR);
    vm.set_reg(1, OUT_PTR);
    assert!(!run(&mut vm), "a block ending exactly at the end of the ring fits");
    assert_eq!(word(&mut vm, OUT_PTR), RING + RING_BYTES - U_ALLOC);
}

/// Each way the seeding can be wrong, and the assertion that ALL of them cross. Every one
/// is a case the handler defines: nothing bound, a pointer we did not stamp, a context
/// with no ring yet.
#[test]
fn reserve_falls_back_on_every_unstamped_case() {
    let l = reserve_layout();
    let cases: [(&str, u32, u32); 5] = [
        ("no bound program", l.ctx_program, 0),
        ("a handle we did not create", l.ctx_program, PROG_PTR + 0x40),
        ("a block that is not a context", l.ctx_magic_at, CTX_MAGIC ^ 1),
        ("no ring attached", l.ctx_ring_base, 0),
        ("a cursor below the ring", l.ctx_ring_cursor, RING - 0x100),
    ];
    for (what, at, value) in cases {
        let mut vm = vm_with_reserve(RING);
        vm.write_mem(CTX_PTR + at, &value.to_le_bytes()).expect("break one word");
        vm.set_reg(0, CTX_PTR);
        vm.set_reg(1, OUT_PTR);
        assert!(run(&mut vm), "{what} must reach the host");
        assert_eq!(word(&mut vm, OUT_PTR), SENTINEL_BASE, "{what} must write nothing");
        assert_eq!(
            word(&mut vm, CTX_PTR + l.ctx_ring_cursor),
            if at == l.ctx_ring_cursor { value } else { RING },
            "{what} must not move the cursor"
        );
    }
}

/// The two pointer arms nothing else notices. A null context would rebase to an address
/// near the top of linear memory - a real page - and be read as a ring; a null
/// out-parameter would have the block stored there.
#[test]
fn reserve_falls_back_on_a_null_pointer_either_side() {
    for (r0, r1) in [(0, OUT_PTR), (CTX_PTR, 0)] {
        let mut vm = vm_with_reserve(RING);
        vm.set_reg(0, r0);
        vm.set_reg(1, r1);
        assert!(run(&mut vm), "a null pointer must reach the host");
        assert_eq!(
            word(&mut vm, CTX_PTR + reserve_layout().ctx_ring_cursor),
            RING,
            "and must not have bumped anything on the way"
        );
    }
}

// --- InlineOp::SetUniformData ----------------------------------------------------------
//
// The form that writes the bytes a SHADER READS, into two places at once, from a pointer
// the caller left on the STACK. Its failure modes are a wrong picture rather than a missing
// one, and none of them would trip an assertion anywhere else - hence an execution test per
// arm.

/// The parameter record, the uniform buffer the guest names, the SA bank, and the stack
/// slot holding the fifth argument. All far enough apart that a copy landing in the wrong
/// one is visible as a sentinel that did not change.
const PARAM_REC: u32 = BASE + 0x6000;
const UBUF: u32 = BASE + 0x9000;
const SA_BANK: u32 = BASE + 0xA000;
const STACK: u32 = BASE + 0xC000;
const SRC: u32 = BASE + 0xD000;

const MAX_REGS: u32 = 64;
const BANK_DATA: u32 = 4;
/// The parameter's `resource_index`: the register the write starts at. Non-zero, because a
/// form that ignored it would pass every test that used zero.
const RES_INDEX: u32 = 3;

fn uniform_data_layout() -> vitaslop_transpiler::UniformDataLayout {
    vitaslop_transpiler::UniformDataLayout {
        bank_slot: 0,
        bank_len_at: 0,
        bank_data_at: BANK_DATA,
        param_packed_at: 4,
        type_shift: 4,
        type_mask: 0xf,
        f16_type: 1,
        param_index_at: 12,
        max_regs: MAX_REGS,
    }
}

/// A VM with a parameter record of component type `type_bits`, `count` floats staged at
/// [`SRC`], and the stack slot pointing at them.
fn vm_with_uniform_data(type_bits: u32, count: u32) -> Vm {
    let mut vm = vm_with(InlineOp::SetUniformData { layout: uniform_data_layout() });
    seed_uniform_data(&mut vm, type_bits, count);
    vm
}

/// The same fixture on a build that TRACKS guest stores, for the dirty-map assertion.
fn vm_with_dirty_uniform_data() -> (Vm, u32) {
    let (mut vm, block) = vm_with_dirty(InlineOp::SetUniformData { layout: uniform_data_layout() });
    seed_uniform_data(&mut vm, 0, 8);
    (vm, block)
}

fn seed_uniform_data(vm: &mut Vm, type_bits: u32, count: u32) {
    let l = uniform_data_layout();
    write_mirror(vm, &[SA_BANK]);
    let mut w = |addr: u32, v: u32| vm.write_mem(addr, &v.to_le_bytes()).expect("seed");
    w(PARAM_REC + l.param_packed_at, type_bits << l.type_shift);
    w(PARAM_REC + l.param_index_at, RES_INDEX);
    w(STACK, SRC);
    for i in 0..count {
        w(SRC + i * 4, UNIFORM_VALUES | i);
    }
    // Sentinels across both destinations, so a copy at the wrong offset shows up as a word
    // that changed when it should not have.
    for i in 0..MAX_REGS {
        w(UBUF + i * 4, SENTINEL_BASE | i);
        w(SA_BANK + BANK_DATA + i * 4, SENTINEL_BASE | i);
    }
    w(SA_BANK, 0);
    vm.set_reg(13, STACK);
}

/// The float bit patterns the guest hands over. Distinctive, and each carries its own index.
const UNIFORM_VALUES: u32 = 0x4B10_0000;

/// The ordinary case: the same bytes land in the buffer at the parameter's register, in the
/// bank at the same register, and the bank's high-water mark rises to cover them.
#[test]
fn set_uniform_data_writes_the_buffer_and_the_bank() {
    let l = uniform_data_layout();
    const COUNT: u32 = 4;
    const OFFSET: u32 = 2;
    let mut vm = vm_with_uniform_data(0, COUNT);
    vm.set_reg(0, UBUF);
    vm.set_reg(1, PARAM_REC);
    vm.set_reg(2, OFFSET);
    vm.set_reg(3, COUNT);
    assert!(!run(&mut vm), "a readable F32 parameter must not reach the host");
    let at = RES_INDEX + OFFSET;
    for i in 0..MAX_REGS {
        let want = if (at..at + COUNT).contains(&i) {
            UNIFORM_VALUES | (i - at)
        } else {
            SENTINEL_BASE | i
        };
        assert_eq!(word(&mut vm, UBUF + i * 4), want, "buffer register {i}");
        assert_eq!(word(&mut vm, SA_BANK + BANK_DATA + i * 4), want, "bank register {i}");
    }
    assert_eq!(word(&mut vm, SA_BANK + l.bank_len_at), at + COUNT, "the high-water mark covers it");
    assert_eq!(vm.get_reg(0), 0, "the call returns the handler's success code");
}

/// The high-water mark rises and never falls: two calls setting different uniforms must
/// leave BOTH readable, which is what the guest's own buffer does.
#[test]
fn set_uniform_data_raises_the_high_water_mark_but_never_lowers_it() {
    let l = uniform_data_layout();
    let mut vm = vm_with_uniform_data(0, 8);
    // A far write first...
    vm.set_reg(0, UBUF);
    vm.set_reg(1, PARAM_REC);
    vm.set_reg(2, 20);
    vm.set_reg(3, 4);
    assert!(!run(&mut vm));
    assert_eq!(word(&mut vm, SA_BANK + l.bank_len_at), RES_INDEX + 24);
    // ...then a nearer one, which must not shrink the mark.
    vm.set_reg(0, UBUF);
    vm.set_reg(1, PARAM_REC);
    vm.set_reg(2, 0);
    vm.set_reg(3, 2);
    assert!(!run(&mut vm));
    assert_eq!(word(&mut vm, SA_BANK + l.bank_len_at), RES_INDEX + 24, "the mark holds");
    assert_eq!(word(&mut vm, SA_BANK + BANK_DATA + RES_INDEX * 4), UNIFORM_VALUES, "and both writes are there");
    assert_eq!(word(&mut vm, SA_BANK + BANK_DATA + (RES_INDEX + 20) * 4), UNIFORM_VALUES);
}

/// A count of zero writes nothing and still succeeds - `memory.copy` of zero bytes is legal
/// and the handler accepts it, so the two must agree rather than one of them refusing.
#[test]
fn set_uniform_data_of_no_components_writes_nothing() {
    let l = uniform_data_layout();
    let mut vm = vm_with_uniform_data(0, 0);
    vm.set_reg(0, UBUF);
    vm.set_reg(1, PARAM_REC);
    vm.set_reg(2, 0);
    vm.set_reg(3, 0);
    assert!(!run(&mut vm));
    assert_eq!(word(&mut vm, SA_BANK + l.bank_len_at), RES_INDEX, "the mark covers an empty write");
    for i in 0..MAX_REGS {
        assert_eq!(word(&mut vm, UBUF + i * 4), SENTINEL_BASE | i, "buffer register {i} untouched");
    }
}

/// Every arm that belongs to the handler. Each one is a case the handler DEFINES - an F16
/// parameter packs two components per register, a clamped or absurd index is the handler's
/// clamp, a write past the ceiling is dropped from the bank but not from the buffer - and
/// each must leave both destinations alone here.
#[test]
fn set_uniform_data_falls_back_on_every_case_the_handler_defines() {
    let l = uniform_data_layout();
    // (what, r1, r2, r3, and an optional (address, value) to break first)
    let cases: [(&str, u32, u32, u32, Option<(u32, u32)>); 6] = [
        ("an F16 parameter", PARAM_REC, 0, 4, Some((PARAM_REC + l.param_packed_at, 1 << l.type_shift))),
        ("a null parameter record", 0, 0, 4, None),
        ("a NEGATIVE resource_index", PARAM_REC, 0, 4, Some((PARAM_REC + l.param_index_at, 0xFFFF_FFFF))),
        ("a resource_index past the ceiling", PARAM_REC, 0, 4, Some((PARAM_REC + l.param_index_at, MAX_REGS + 1))),
        ("a write that ends past the ceiling", PARAM_REC, MAX_REGS - 4, 4, None),
        ("no bank at all", PARAM_REC, 0, 4, None),
    ];
    for (what, r1, r2, r3, poke) in cases {
        let mut vm = vm_with_uniform_data(0, 8);
        if what == "no bank at all" {
            write_mirror(&mut vm, &[0]);
        }
        if let Some((addr, value)) = poke {
            vm.write_mem(addr, &value.to_le_bytes()).expect("break one word");
        }
        vm.set_reg(0, UBUF);
        vm.set_reg(1, r1);
        vm.set_reg(2, r2);
        vm.set_reg(3, r3);
        assert!(run(&mut vm), "{what} must reach the host");
        for i in 0..MAX_REGS {
            assert_eq!(word(&mut vm, UBUF + i * 4), SENTINEL_BASE | i, "{what}: buffer register {i}");
        }
        assert_eq!(word(&mut vm, SA_BANK + l.bank_len_at), 0, "{what}: the mark did not move");
    }
}

/// The three pointer arms nothing else notices: a null buffer, a source pointer off the end
/// of memory, and a stack pointer that cannot even be read for the fifth argument.
#[test]
fn set_uniform_data_falls_back_on_a_bad_pointer() {
    for (what, r0, src, sp) in [
        ("a null uniform buffer", 0, SRC, STACK),
        ("a source past the end of memory", UBUF, BASE + MEM_BYTES - 4, STACK),
        ("a null stack pointer", UBUF, SRC, 0),
    ] {
        let mut vm = vm_with_uniform_data(0, 4);
        vm.write_mem(STACK, &src.to_le_bytes()).expect("stage the fifth argument");
        vm.set_reg(13, sp);
        vm.set_reg(0, r0);
        vm.set_reg(1, PARAM_REC);
        vm.set_reg(2, 0);
        vm.set_reg(3, 4);
        assert!(run(&mut vm), "{what} must reach the host");
        assert_eq!(word(&mut vm, SA_BANK + BANK_DATA), SENTINEL_BASE, "{what} wrote nothing");
    }
}

/// The write STAMPS the dirty map over the buffer it wrote, and only over that.
///
/// A uniform buffer is wherever the guest put it, so this form has no argument for skipping
/// the stamp - the same position `sceClibMemcpy` is in. An unstamped write is a page the
/// host believes it has already uploaded, drawn from bytes the guest has since changed.
#[test]
fn set_uniform_data_stamps_the_buffer_it_wrote() {
    const EPOCH: u8 = 0x21;
    let l = uniform_data_layout();
    let first = (UBUF - BASE) >> vitaslop_transpiler::DIRTY_SHIFT;
    let (mut vm, block) = vm_with_dirty_uniform_data();
    let vm = &mut vm;
    seed_epoch(vm, block, EPOCH, first..first + 2);
    vm.set_reg(0, UBUF);
    vm.set_reg(1, PARAM_REC);
    vm.set_reg(2, 0);
    vm.set_reg(3, 4);
    assert!(!run(vm));
    assert_eq!(stamp_of(vm, block, first), EPOCH, "the page the buffer is on is stamped");
    assert_eq!(word(vm, SA_BANK + l.bank_len_at), RES_INDEX + 4, "and the write happened");
}

/// Guard against the registers being read back from somewhere other than the globals the
/// emitted code writes - if `set_reg`/`get_reg` did not round-trip, every assertion above
/// would be vacuous.
#[test]
fn the_register_accessors_round_trip() {
    let mut vm = vm_with(InlineOp::LoadMirror { slot: 0 });
    write_mirror(&mut vm, &[0]);
    vm.set_reg(1, 0x5A5A_5A5A);
    assert_eq!(vm.get_reg(1), 0x5A5A_5A5A);
    assert!(abi::REG_COUNT > 1, "r1 exists");
}
