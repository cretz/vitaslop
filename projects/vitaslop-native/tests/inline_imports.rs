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
    let op = InlineOp::LoadShiftMask { offset: 4, shift: 8, mask: 0xf };
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
    let op = InlineOp::LoadShiftMask { offset: 4, shift: 8, mask: 0xf };
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
