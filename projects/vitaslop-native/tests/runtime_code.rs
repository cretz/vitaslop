//! Code reached only through an address the guest COMPUTES at run time: no call names it, no
//! constant materializes it, no pointer in the image points at it - so static discovery never
//! translates it, and the dispatcher's miss translates it then (`vitaslop_native::runtime_code`).
//!
//! The image is hand-assembled ARM. Its entry branches to `base + r1`, with `r1` set by the test,
//! which is as invisible to discovery as an interpreter's opcode handler is. The last case is the
//! interpreter's other habit: two handlers that tail-jump to each other's (computed) addresses a
//! million times without ever returning, which only runs on a bounded wasm stack if a guest tail
//! jump is a wasm tail call.

use vitaslop_native::{HostAbi, Vm, DEFAULT_MEM_BYTES};

const BASE: u32 = 0x8100_0000;

/// The image: little-endian ARM words at their offsets, Thumb halfwords where noted.
fn image() -> Vec<u8> {
    let mut code = vec![0u8; 0x100];
    let mut word = |off: usize, w: u32| code[off..off + 4].copy_from_slice(&w.to_le_bytes());
    // entry: r0 = base + r1, then call it and return.
    word(0x00, 0xe24f_2008); // sub r2, pc, #8     ; r2 = base
    word(0x04, 0xe082_0001); // add r0, r2, r1
    word(0x08, 0xe52d_e004); // str lr, [sp, #-4]!
    word(0x0c, 0xe12f_ff30); // blx r0
    word(0x10, 0xe49d_f004); // ldr pc, [sp], #4
    // +0x40, ARM: return 42.
    word(0x40, 0xe3a0_002a); // mov r0, #42
    word(0x44, 0xe12f_ff1e); // bx lr
    // +0x60, ARM handler A: count r3 down; at zero return, else jump to r4.
    word(0x60, 0xe253_3001); // subs r3, r3, #1
    word(0x64, 0x012f_ff1e); // bxeq lr
    word(0x68, 0xe12f_ff14); // bx r4
    // +0x70, ARM handler B: jump to r5.
    word(0x70, 0xe12f_ff15); // bx r5
    // +0x50, Thumb: return 7.
    code[0x50..0x52].copy_from_slice(&0x2007u16.to_le_bytes()); // movs r0, #7
    code[0x52..0x54].copy_from_slice(&0x4770u16.to_le_bytes()); // bx lr
    code
}

fn vm() -> Vm {
    Vm::new(&image(), BASE, false, &[BASE], &[], DEFAULT_MEM_BYTES, &HostAbi::default()).expect("transpile")
}

#[test]
fn a_computed_arm_target_is_translated_when_reached() {
    let mut vm = vm();
    assert_eq!(vm.runtime_stats().entries, 0, "nothing is translated before it is reached");
    vm.set_reg(1, 0x40);
    vm.call(BASE).expect("run");
    assert_eq!(vm.get_reg(0), 42);
    let stats = vm.runtime_stats();
    assert_eq!(stats.entries - stats.swept, 1, "one target asked for; the rest of the stretch is guessed");
    // Reached again, it is the same translation: the dispatcher finds it in the entry table.
    vm.set_reg(1, 0x40);
    vm.set_reg(0, 0);
    vm.call(BASE).expect("run again");
    assert_eq!(vm.get_reg(0), 42);
    assert_eq!(vm.runtime_stats().modules, 1);
}

#[test]
fn a_computed_thumb_target_is_lifted_in_the_mode_its_address_names() {
    let mut vm = vm();
    vm.set_reg(1, 0x51);
    vm.call(BASE).expect("run");
    assert_eq!(vm.get_reg(0), 7);
}

#[test]
fn handlers_that_tail_jump_to_each_other_run_on_a_bounded_stack() {
    let mut vm = vm();
    vm.set_reg(1, 0x60);
    vm.set_reg(3, 1_000_000);
    vm.set_reg(4, BASE + 0x70);
    vm.set_reg(5, BASE + 0x60);
    vm.call(BASE).expect("a million tail jumps");
    assert_eq!(vm.get_reg(3), 0);
    // Handler B sits in the stretch A was found in, so it came with A: one miss, one module.
    let stats = vm.runtime_stats();
    assert_eq!(stats.modules, 1, "{stats:?}");
    assert_eq!(stats.entries - stats.swept, 1, "only A was asked for: {stats:?}");
}

#[test]
fn without_tail_calls_the_same_chain_exhausts_the_stack() {
    // The control for the case above: the chain is what needs them.
    vitaslop_transpiler::set_tail_calls(false);
    let mut vm = vm();
    vitaslop_transpiler::set_tail_calls(true);
    vm.set_reg(1, 0x60);
    vm.set_reg(3, 1_000_000);
    vm.set_reg(4, BASE + 0x70);
    vm.set_reg(5, BASE + 0x60);
    let err = vm.call(BASE).expect_err("a million nested calls");
    assert!(format!("{err:?}").contains("StackOverflow"), "{err:?}");
}

static TRACED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn count_traces(
    selector: u32,
    _regs: &mut [u32; vitaslop_native::abi::REG_COUNT],
    _mem: &mut [u8],
    _base: u32,
    _out: &mut Vec<u8>,
) -> vitaslop_runtime::SvcOutcome {
    if selector == BASE + 0x40 {
        TRACED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    vitaslop_runtime::SvcOutcome::Continue
}

#[test]
fn the_block_tracer_reaches_code_translated_at_run_time() {
    // A diagnostic that goes silent on runtime-translated code reads as "this never ran".
    vitaslop_transpiler::set_trace_blocks(vec![(BASE + 0x40, BASE + 0x44)]);
    let abi = HostAbi { svc: count_traces, ..HostAbi::default() };
    let mut vm = Vm::new(&image(), BASE, false, &[BASE], &[], DEFAULT_MEM_BYTES, &abi).expect("transpile");
    vitaslop_transpiler::set_trace_blocks(Vec::new());
    vm.set_reg(1, 0x40);
    vm.call(BASE).expect("run");
    assert_eq!(vm.get_reg(0), 42);
    assert_eq!(TRACED.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn a_target_outside_the_image_still_fails_loudly() {
    let mut vm = vm();
    vm.set_reg(1, 0x0100_0000);
    let err = vm.call(BASE).expect_err("no code there");
    assert!(format!("{err:?}").contains("not in the image"), "{err:?}");
}
