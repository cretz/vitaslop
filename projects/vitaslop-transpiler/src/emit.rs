//! WASM backend: lower each [`ir::Func`] to one wasm function and assemble the
//! module.
//!
//! # Control flow
//! A function's basic blocks become a **dispatch loop**: a `loop` wrapping a
//! `br_table` on a `$bb` (current-block) local, with each block's code emitted in
//! ascending address order. Straight-line fall-through between adjacent blocks
//! needs no branch (control flows through the block boundary); only real jumps
//! and loop back-edges pay a `br` back to the dispatch. Direct calls are wasm
//! `call`s, returns are wasm `return`s - so the wasm call stack mirrors the guest
//! call stack and stays out of the dispatch machinery. This keeps hot loops
//! entirely in-function (no host round-trips) while handling arbitrary intra-
//! function control flow correctly. A relooper that recovers structured loops/
//! ifs for even better codegen can replace the dispatch loop later without
//! touching lowering.
//!
//! # Memory
//! Guest addresses are rebased: linear offset = guest address - `base` (see
//! [`crate::abi`]). Every load/store subtracts `base` before touching memory.
//!
//! # Flags
//! N,Z,C,V are computed eagerly into their globals by the flag statements, using
//! an i64 widening for an always-correct unsigned carry. Lazy flags (compute a
//! condition only where consumed) are a later optimization; the IR seam
//! ([`ir::Stmt::FlagsAdd`]/[`ir::Stmt::FlagsLogic`]) already isolates the choice.

use std::collections::BTreeMap;

use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, ExportKind, ExportSection, Function, FunctionSection,
    GlobalSection, GlobalType, ImportSection, Instruction as W, MemArg, MemorySection, MemoryType,
    Module, TypeSection, ValType,
};

use crate::abi;
use crate::ir::{BinOp, Block, ConditionCode, Func, MemSize, Stmt, Term, Value};

/// Function indices of the two host imports (imports come first).
const SVC_FUNC: u32 = 0;
const IMPORT_FUNC: u32 = 1;
/// Number of imported functions before the guest functions.
pub(crate) const IMPORT_FUNCS: u32 = 2;

// Scratch locals used by flag computation. Local 0 is `$bb`.
const L_BB: u32 = 0;
const L_T0: u32 = 1;
const L_T1: u32 = 2;
const L_T2: u32 = 3;
/// Fourth i32 scratch: the carry-in of add/sub-family flag computation (an
/// immediate, or the C flag for adc/sbc).
const L_T3: u32 = 4;
const L_I32_COUNT: u32 = 5;
/// i64 scratch, used to split/merge a double register across its two aliased
/// single-register halves. Index follows the i32 locals.
const L_D64: u32 = L_I32_COUNT;
/// Two `v128` scratch locals, used by NEON emission to hold a quad register for
/// read-modify-write (writing one D lane of an upper-bank quad) and to stage the
/// two operands of the ops that read each twice (`vabd`/`vabdl`). Follow the i64
/// scratch.
const L_V128A: u32 = L_D64 + 1;
const L_V128B: u32 = L_D64 + 2;

/// Assemble the full wasm module for `funcs`. `func_index` maps a guest function
/// address to its wasm function index. `mem_bytes` sizes the exported linear
/// memory; `base` is the guest image base for the address rebase.
pub fn emit_module(
    funcs: &[Func],
    func_index: &BTreeMap<u32, u32>,
    base: u32,
    mem_bytes: u32,
    import_memory: bool,
) -> Vec<u8> {
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], []); // svc / import: (i32) -> ()
    let host_ty = 0;
    types.ty().function([], []); // guest function: () -> ()
    let func_ty = 1;

    let pages = (mem_bytes as u64).div_ceil(abi::PAGE_SIZE as u64).max(1);
    // Preemptive multithreading (the native `ThreadedScheduler`) runs each guest
    // thread as its own instance so their register globals stay independent, but
    // they must share one address space. wasm gives us exactly one tool for that:
    // a `shared` linear memory imported into every instance. When `import_memory`
    // is set the module imports `env.memory` (shared, fixed size) instead of
    // defining its own; single-instance hosts leave it off and get the original
    // self-contained module unchanged. The memory index is 0 either way (it is the
    // only memory), so every load/store and the `memory` export are identical.
    let mut imports = ImportSection::new();
    imports.import(abi::IMPORT_MODULE, abi::SVC_NAME, wasm_encoder::EntityType::Function(host_ty));
    imports.import(abi::IMPORT_MODULE, abi::IMPORT_NAME, wasm_encoder::EntityType::Function(host_ty));
    if import_memory {
        // A shared memory must declare a maximum; the guest never grows memory, so
        // pin max == min at the provisioned size.
        imports.import(
            abi::IMPORT_MODULE,
            abi::MEMORY_EXPORT,
            wasm_encoder::EntityType::Memory(MemoryType {
                minimum: pages,
                maximum: Some(pages),
                memory64: false,
                shared: true,
                page_size_log2: None,
            }),
        );
    }

    let mut function_section = FunctionSection::new();
    for _ in funcs {
        function_section.function(func_ty);
    }
    // The indirect-call dispatcher (the last defined function): `(i32) -> ()`,
    // the same shape as the host imports (type 0). It maps a runtime guest
    // function-pointer to the matching translated function - see `emit_dispatch`.
    function_section.function(host_ty);

    let mut mems = MemorySection::new();
    if !import_memory {
        mems.memory(MemoryType {
            minimum: pages,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
    }

    // Globals, in ABI order: 16 registers + 4 integer flags (i32), then the VFP/NEON
    // register file - 32 single-precision S registers (raw-bit i32) and 8 upper
    // quad registers Q8..Q15 (v128) - then 4 FP condition flags (i32).
    let mut globals = GlobalSection::new();
    let i32_global =
        GlobalType { val_type: ValType::I32, mutable: true, shared: false };
    let v128_global =
        GlobalType { val_type: ValType::V128, mutable: true, shared: false };
    for _ in 0..abi::GLOBAL_COUNT {
        globals.global(i32_global, &ConstExpr::i32_const(0));
    }
    for _ in 0..abi::VFP_S_COUNT {
        globals.global(i32_global, &ConstExpr::i32_const(0));
    }
    for _ in 0..abi::VFP_Q_HI_COUNT {
        globals.global(v128_global, &ConstExpr::v128_const(0));
    }
    for _ in 0..abi::FP_FLAG_COUNT {
        globals.global(i32_global, &ConstExpr::i32_const(0));
    }

    let mut exports = ExportSection::new();
    exports.export(abi::MEMORY_EXPORT, ExportKind::Memory, 0);
    for i in 0..abi::REG_COUNT {
        exports.export(&abi::reg_export(i), ExportKind::Global, abi::reg_global(i));
    }
    for f in [abi::Flag::N, abi::Flag::Z, abi::Flag::C, abi::Flag::V] {
        exports.export(abi::flag_export(f), ExportKind::Global, abi::flag_global(f));
    }
    // Export the VFP/NEON register file so the host can seed/read FP state (tests,
    // GXM capture). S0..S31 (i32), Q8..Q15 (v128), and the FP flags. The host only
    // marshals the low-bank S registers; the v128 quads are exported for
    // completeness (JS cannot read them, but native hosts and tools can).
    for n in 0..abi::VFP_S_COUNT as u8 {
        exports.export(&abi::vfp_s_export(n), ExportKind::Global, abi::vfp_s_global(n));
    }
    for q in abi::VFP_Q_HI_FIRST as u8..(abi::VFP_Q_HI_FIRST + abi::VFP_Q_HI_COUNT) as u8 {
        exports.export(&abi::vfp_qhi_export(q), ExportKind::Global, abi::vfp_qhi_global(q));
    }
    for f in [abi::Flag::N, abi::Flag::Z, abi::Flag::C, abi::Flag::V] {
        exports.export(abi::fp_flag_export(f), ExportKind::Global, abi::fp_flag_global(f));
    }

    let mut code = CodeSection::new();
    for (i, func) in funcs.iter().enumerate() {
        let idx = IMPORT_FUNCS + i as u32;
        exports.export(&abi::func_export(func.addr), ExportKind::Func, idx);
        code.function(&emit_func(func, func_index, base));
    }
    code.function(&emit_dispatch(func_index));

    let mut module = Module::new();
    module
        .section(&types)
        .section(&imports)
        .section(&function_section)
        .section(&mems)
        .section(&globals)
        .section(&exports)
        .section(&code);
    module.finish()
}

/// Emit the indirect-call dispatcher: `(addr: i32) -> ()`. It masks the Thumb
/// bit off the runtime function-pointer and, for each translated function,
/// compares the address and directly calls the match. Every guest function is
/// `() -> ()` (state lives in globals), so a direct `call` after the match is all
/// that is needed - no wasm table or `call_indirect`. An unresolved target hits
/// `unreachable` (a real bug: an indirect callee we never discovered). The chain
/// is linear in the function count; a binary search is a later optimization.
fn emit_dispatch(func_index: &BTreeMap<u32, u32>) -> Function {
    let mut f = Function::new([]);
    // addr &= ~1  (clear the Thumb bit; function addresses are even).
    f.instruction(&W::LocalGet(0));
    f.instruction(&W::I32Const(!1));
    f.instruction(&W::I32And);
    f.instruction(&W::LocalSet(0));
    for (&addr, &idx) in func_index {
        f.instruction(&W::LocalGet(0));
        f.instruction(&W::I32Const(addr as i32));
        f.instruction(&W::I32Eq);
        f.instruction(&W::If(BlockType::Empty));
        f.instruction(&W::Call(idx));
        f.instruction(&W::Return);
        f.instruction(&W::End);
    }
    f.instruction(&W::Unreachable);
    f.instruction(&W::End);
    f
}

/// Emit one guest function as a wasm function: a dispatch loop over its blocks.
fn emit_func(func: &Func, func_index: &BTreeMap<u32, u32>, base: u32) -> Function {
    // Locals: $bb + i32 scratch temps (flag computation), then one i64 scratch
    // (double-register split/merge) and one v128 scratch (NEON quad staging).
    let mut f = Function::new([
        (L_I32_COUNT, ValType::I32),
        (1, ValType::I64),
        (2, ValType::V128),
    ]);

    let n = func.blocks.len() as u32;

    // Single-block functions need no dispatch machinery.
    if n == 1 {
        emit_block(&mut f, &func.blocks[0], func, func_index, base, 0);
        f.instruction(&W::End);
        return f;
    }

    // block $exit ; loop $loop ; block $B{n-1} ... block $B0 ; br_table ...
    f.instruction(&W::Block(BlockType::Empty)); // $exit
    f.instruction(&W::Loop(BlockType::Empty)); // $loop
    for _ in 0..n {
        f.instruction(&W::Block(BlockType::Empty));
    }
    // At the innermost point, $B0 is depth 0 .. $B{n-1} is depth n-1.
    f.instruction(&W::LocalGet(L_BB));
    let targets: Vec<u32> = (0..n).collect();
    f.instruction(&W::BrTable(targets.into(), n /* default -> $loop */));
    // Close $B0 (its body follows), then $B1, ...
    for (k, block) in func.blocks.iter().enumerate() {
        f.instruction(&W::End); // closes $B{k}
        emit_block(&mut f, block, func, func_index, base, n - 1 - k as u32);
    }
    f.instruction(&W::End); // loop
    f.instruction(&W::End); // $exit block
    f.instruction(&W::End); // function body
    f
}

/// Emit one basic block's statements and terminator. `loop_depth` is the wasm
/// branch depth from within this block's code to the enclosing dispatch `loop`.
fn emit_block(
    f: &mut Function,
    block: &Block,
    func: &Func,
    func_index: &BTreeMap<u32, u32>,
    base: u32,
    loop_depth: u32,
) {
    for stmt in &block.stmts {
        emit_stmt(f, stmt, func_index, base);
    }
    emit_term(f, &block.term, func, loop_depth);
}

/// Re-dispatch to the block at `target` address: set `$bb`, branch to the loop.
/// `extra` accounts for any `if`/`block` frames open between here and the loop.
fn goto(f: &mut Function, func: &Func, target: u32, loop_depth: u32, extra: u32) {
    let idx = func
        .block_index(target)
        .unwrap_or_else(|| panic!("branch target {target:#x} is not a block in f_{:x}", func.addr))
        as i32;
    f.instruction(&W::I32Const(idx));
    f.instruction(&W::LocalSet(L_BB));
    f.instruction(&W::Br(loop_depth + extra));
}

fn emit_term(f: &mut Function, term: &Term, func: &Func, loop_depth: u32) {
    match term {
        Term::Fallthrough => {} // flow into the next block's code
        Term::Return | Term::Halt => {
            f.instruction(&W::Return);
        }
        Term::Jump(target) => {
            goto(f, func, *target, loop_depth, 0);
        }
        Term::Branch { cond, taken } => {
            emit_cond(f, *cond);
            f.instruction(&W::If(BlockType::Empty));
            goto(f, func, *taken, loop_depth, 1); // +1 for the `if` frame
            f.instruction(&W::End);
        }
        Term::BranchZero { reg, nonzero, taken } => {
            f.instruction(&W::GlobalGet(abi::reg_global(*reg as usize)));
            f.instruction(&W::I32Eqz); // reg == 0
            if *nonzero {
                f.instruction(&W::I32Eqz); // reg != 0
            }
            f.instruction(&W::If(BlockType::Empty));
            goto(f, func, *taken, loop_depth, 1);
            f.instruction(&W::End);
        }
    }
}

/// Push a 0/1 i32 for `cond` computed from the flag globals.
fn emit_cond(f: &mut Function, cond: ConditionCode) {
    use ConditionCode::*;
    fn get(f: &mut Function, flag: abi::Flag) {
        f.instruction(&W::GlobalGet(abi::flag_global(flag)));
    }
    fn eqz(f: &mut Function) {
        f.instruction(&W::I32Eqz);
    }
    match cond {
        EQ => get(f, abi::Flag::Z),
        NE => { get(f, abi::Flag::Z); eqz(f); }
        HS => get(f, abi::Flag::C),
        LO => { get(f, abi::Flag::C); eqz(f); }
        MI => get(f, abi::Flag::N),
        PL => { get(f, abi::Flag::N); eqz(f); }
        VS => get(f, abi::Flag::V),
        VC => { get(f, abi::Flag::V); eqz(f); }
        HI => {
            // C && !Z
            get(f, abi::Flag::C);
            get(f, abi::Flag::Z);
            eqz(f);
            f.instruction(&W::I32And);
        }
        LS => {
            // !C || Z  ==  !(C && !Z)
            get(f, abi::Flag::C);
            get(f, abi::Flag::Z);
            eqz(f);
            f.instruction(&W::I32And);
            eqz(f);
        }
        GE => { get(f, abi::Flag::N); get(f, abi::Flag::V); f.instruction(&W::I32Eq); }
        LT => { get(f, abi::Flag::N); get(f, abi::Flag::V); f.instruction(&W::I32Ne); }
        GT => {
            // !Z && (N == V)
            get(f, abi::Flag::N);
            get(f, abi::Flag::V);
            f.instruction(&W::I32Eq);
            get(f, abi::Flag::Z);
            eqz(f);
            f.instruction(&W::I32And);
        }
        LE => {
            // Z || (N != V)
            get(f, abi::Flag::N);
            get(f, abi::Flag::V);
            f.instruction(&W::I32Ne);
            get(f, abi::Flag::Z);
            f.instruction(&W::I32Or);
        }
        AL => { f.instruction(&W::I32Const(1)); }
    }
}

fn emit_stmt(
    f: &mut Function,
    stmt: &Stmt,
    func_index: &BTreeMap<u32, u32>,
    base: u32,
) {
    match stmt {
        Stmt::SetReg(r, v) => {
            emit_value(f, v, base);
            f.instruction(&W::GlobalSet(abi::reg_global(*r as usize)));
        }
        Stmt::Store { addr, data, size } => {
            emit_addr(f, addr, base);
            emit_value(f, data, base);
            f.instruction(&store_op(*size));
        }
        Stmt::FlagsAdd { a, b, cin } => emit_flags_add(f, a, b, cin, base),
        Stmt::FlagsLogic { value, carry } => {
            emit_value(f, value, base);
            f.instruction(&W::LocalTee(L_T0));
            f.instruction(&W::I32Eqz);
            f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::Z)));
            f.instruction(&W::LocalGet(L_T0));
            f.instruction(&W::I32Const(31));
            f.instruction(&W::I32ShrU);
            f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::N)));
            if let Some(c) = carry {
                emit_value(f, c, base);
                f.instruction(&W::I32Const(1));
                f.instruction(&W::I32And);
                f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::C)));
            }
        }
        Stmt::Svc(imm) => {
            f.instruction(&W::I32Const(*imm as i32));
            f.instruction(&W::Call(SVC_FUNC));
        }
        Stmt::Import(index) => {
            f.instruction(&W::I32Const(*index as i32));
            f.instruction(&W::Call(IMPORT_FUNC));
        }
        Stmt::MulLong { rdlo, rdhi, rn, rm, signed } => {
            // Extend both operands to i64 (sign- or zero-extend), multiply, and
            // split the 64-bit product into its low and high 32-bit halves. The
            // full product is computed before either register is written, so an
            // operand that aliases a destination reads its old value first.
            let extend = |f: &mut Function| {
                if *signed {
                    f.instruction(&W::I64ExtendI32S);
                } else {
                    f.instruction(&W::I64ExtendI32U);
                }
            };
            emit_value(f, rn, base);
            extend(f);
            emit_value(f, rm, base);
            extend(f);
            f.instruction(&W::I64Mul);
            f.instruction(&W::LocalTee(L_D64)); // full product, kept for the high half
            f.instruction(&W::I32WrapI64); // low 32 bits
            f.instruction(&W::GlobalSet(abi::reg_global(*rdlo as usize)));
            f.instruction(&W::LocalGet(L_D64));
            f.instruction(&W::I64Const(32));
            f.instruction(&W::I64ShrU);
            f.instruction(&W::I32WrapI64); // high 32 bits
            f.instruction(&W::GlobalSet(abi::reg_global(*rdhi as usize)));
        }
        Stmt::Call { target } => {
            let idx = *func_index.get(target).expect("callee index");
            f.instruction(&W::Call(idx));
        }
        Stmt::CallIndirect { addr } => {
            // Push the runtime target address and call the module dispatcher,
            // which maps it to the matching translated function. The dispatcher is
            // the last defined function: IMPORT_FUNCS + one index per guest func.
            let dispatch = IMPORT_FUNCS + func_index.len() as u32;
            emit_value(f, addr, base);
            f.instruction(&W::Call(dispatch));
        }
        Stmt::Guard(cond, body) => {
            emit_cond(f, *cond);
            f.instruction(&W::If(BlockType::Empty));
            for s in body {
                emit_stmt(f, s, func_index, base);
            }
            f.instruction(&W::End);
        }
        Stmt::Vfp(op) => emit_vfp(f, op),
        Stmt::VfpMem { reg, addr, load } => emit_vfp_mem(f, *reg, addr, *load, base),
        Stmt::Neon(op) => emit_neon(f, op),
    }
}

// --- VFP / floating-point emission ---------------------------------------

/// Push S`n` interpreted as an f32.
fn get_s_f32(f: &mut Function, n: u8) {
    f.instruction(&W::GlobalGet(abi::vfp_s_global(n)));
    f.instruction(&W::F32ReinterpretI32);
}

/// Store the f32 on the stack into S`n` (as raw bits).
fn set_s_f32(f: &mut Function, n: u8) {
    f.instruction(&W::I32ReinterpretF32);
    f.instruction(&W::GlobalSet(abi::vfp_s_global(n)));
}

/// Push the raw 64 bits of D`n` as an i64. Low bank (n < 16): merge the two S
/// halves. Upper bank (n >= 16): extract `i64x2` lane `n & 1` of the quad `q(n/2)`.
fn get_d_bits(f: &mut Function, n: u8) {
    if (n as usize) < abi::VFP_D_HI_FIRST {
        let lo = 2 * n;
        let hi = 2 * n + 1;
        f.instruction(&W::GlobalGet(abi::vfp_s_global(hi)));
        f.instruction(&W::I64ExtendI32U);
        f.instruction(&W::I64Const(32));
        f.instruction(&W::I64Shl);
        f.instruction(&W::GlobalGet(abi::vfp_s_global(lo)));
        f.instruction(&W::I64ExtendI32U);
        f.instruction(&W::I64Or);
    } else {
        f.instruction(&W::GlobalGet(abi::vfp_qhi_global(n / 2)));
        f.instruction(&W::I64x2ExtractLane(n & 1));
    }
}

/// Store the raw i64 on the stack into D`n`. Low bank (n < 16): split into the two
/// S halves. Upper bank (n >= 16): replace `i64x2` lane `n & 1` of quad `q(n/2)`,
/// leaving the sibling D untouched. Uses the i64 scratch local.
fn set_d_bits(f: &mut Function, n: u8) {
    if (n as usize) < abi::VFP_D_HI_FIRST {
        let lo = 2 * n;
        let hi = 2 * n + 1;
        f.instruction(&W::LocalSet(L_D64));
        f.instruction(&W::LocalGet(L_D64));
        f.instruction(&W::I32WrapI64);
        f.instruction(&W::GlobalSet(abi::vfp_s_global(lo)));
        f.instruction(&W::LocalGet(L_D64));
        f.instruction(&W::I64Const(32));
        f.instruction(&W::I64ShrU);
        f.instruction(&W::I32WrapI64);
        f.instruction(&W::GlobalSet(abi::vfp_s_global(hi)));
    } else {
        // Read-modify-write the quad so the sibling D lane survives.
        let q = abi::vfp_qhi_global(n / 2);
        f.instruction(&W::LocalSet(L_D64));
        f.instruction(&W::GlobalGet(q));
        f.instruction(&W::LocalGet(L_D64));
        f.instruction(&W::I64x2ReplaceLane(n & 1));
        f.instruction(&W::GlobalSet(q));
    }
}

fn fbinop(op: crate::ir::FBinOp) -> W<'static> {
    use crate::ir::FBinOp::*;
    match op {
        Add => W::F32Add,
        Sub => W::F32Sub,
        Mul => W::F32Mul,
        Div => W::F32Div,
    }
}

fn emit_vfp(f: &mut Function, op: &crate::ir::VfpOp) {
    use crate::ir::VfpOp::*;
    match op {
        Bin32 { op, rd, rn, rm } => {
            get_s_f32(f, *rn);
            get_s_f32(f, *rm);
            f.instruction(&fbinop(*op));
            set_s_f32(f, *rd);
        }
        MulAcc32 { rd, rn, rm, sub } => {
            // rd = rd +/- (rn * rm), non-fused.
            get_s_f32(f, *rd);
            get_s_f32(f, *rn);
            get_s_f32(f, *rm);
            f.instruction(&W::F32Mul);
            f.instruction(if *sub { &W::F32Sub } else { &W::F32Add });
            set_s_f32(f, *rd);
        }
        Neg32 { rd, rm } => {
            get_s_f32(f, *rm);
            f.instruction(&W::F32Neg);
            set_s_f32(f, *rd);
        }
        Abs32 { rd, rm } => {
            get_s_f32(f, *rm);
            f.instruction(&W::F32Abs);
            set_s_f32(f, *rd);
        }
        Sqrt32 { rd, rm } => {
            get_s_f32(f, *rm);
            f.instruction(&W::F32Sqrt);
            set_s_f32(f, *rd);
        }
        Mov32 { rd, rm } => {
            // Raw bit copy (no float interpretation).
            f.instruction(&W::GlobalGet(abi::vfp_s_global(*rm)));
            f.instruction(&W::GlobalSet(abi::vfp_s_global(*rd)));
        }
        ScalarToCore { rt, s } => {
            f.instruction(&W::GlobalGet(abi::vfp_s_global(*s)));
            f.instruction(&W::GlobalSet(abi::reg_global(*rt as usize)));
        }
        CoreToScalar { s, rt } => {
            f.instruction(&W::GlobalGet(abi::reg_global(*rt as usize)));
            f.instruction(&W::GlobalSet(abi::vfp_s_global(*s)));
        }
        SetImmS { s, bits } => {
            f.instruction(&W::I32Const(*bits as i32));
            f.instruction(&W::GlobalSet(abi::vfp_s_global(*s)));
        }
        SetImmD { d, lo, hi } => {
            if (*d as usize) < abi::VFP_D_HI_FIRST {
                // Aliased low double: write the two S halves directly.
                f.instruction(&W::I32Const(*lo as i32));
                f.instruction(&W::GlobalSet(abi::vfp_s_global(2 * d)));
                f.instruction(&W::I32Const(*hi as i32));
                f.instruction(&W::GlobalSet(abi::vfp_s_global(2 * d + 1)));
            } else {
                // Upper-bank double: an i64 lane of the quad (set_d_bits does the RMW).
                let bits = ((*hi as u64) << 32) | (*lo as u64);
                f.instruction(&W::I64Const(bits as i64));
                set_d_bits(f, *d);
            }
        }
        Cmp32 { rn, rm } => emit_vfp_cmp(f, *rn, *rm),
        MrsNzcv => {
            // Copy FP flags -> integer NZCV.
            for flag in [abi::Flag::N, abi::Flag::Z, abi::Flag::C, abi::Flag::V] {
                f.instruction(&W::GlobalGet(abi::fp_flag_global(flag)));
                f.instruction(&W::GlobalSet(abi::flag_global(flag)));
            }
        }
        CvtToInt { rd, rm, signed } => {
            // Round toward zero, saturating (matches ARM vcvt.{s,u}32.f32).
            get_s_f32(f, *rm);
            f.instruction(if *signed { &W::I32TruncSatF32S } else { &W::I32TruncSatF32U });
            f.instruction(&W::GlobalSet(abi::vfp_s_global(*rd)));
        }
        CvtFromInt { rd, rm, signed } => {
            f.instruction(&W::GlobalGet(abi::vfp_s_global(*rm)));
            f.instruction(if *signed { &W::F32ConvertI32S } else { &W::F32ConvertI32U });
            set_s_f32(f, *rd);
        }
    }
}

/// Set the FP condition flags (FPSCR N,Z,C,V) from comparing S`rn` against S`rm`
/// (or +0.0 when `rm` is `None`). N=less, Z=equal, C=not-less, V=unordered.
fn emit_vfp_cmp(f: &mut Function, rn: u8, rm: Option<u8>) {
    let push_b = |f: &mut Function| match rm {
        Some(m) => get_s_f32(f, m),
        None => {
            f.instruction(&W::F32Const(0.0f32.into()));
        }
    };
    // N = (a < b)
    get_s_f32(f, rn);
    push_b(f);
    f.instruction(&W::F32Lt);
    f.instruction(&W::GlobalSet(abi::fp_flag_global(abi::Flag::N)));
    // Z = (a == b)
    get_s_f32(f, rn);
    push_b(f);
    f.instruction(&W::F32Eq);
    f.instruction(&W::GlobalSet(abi::fp_flag_global(abi::Flag::Z)));
    // C = !(a < b)
    get_s_f32(f, rn);
    push_b(f);
    f.instruction(&W::F32Lt);
    f.instruction(&W::I32Eqz);
    f.instruction(&W::GlobalSet(abi::fp_flag_global(abi::Flag::C)));
    // V = unordered = (a != a) | (b != b)
    get_s_f32(f, rn);
    get_s_f32(f, rn);
    f.instruction(&W::F32Ne);
    push_b(f);
    push_b(f);
    f.instruction(&W::F32Ne);
    f.instruction(&W::I32Or);
    f.instruction(&W::GlobalSet(abi::fp_flag_global(abi::Flag::V)));
}

/// One VFP register <-> memory transfer. S = 4-byte raw i32; D = 8-byte raw i64.
fn emit_vfp_mem(f: &mut Function, reg: crate::ir::VfpReg, addr: &Value, load: bool, base: u32) {
    use crate::ir::VfpReg::*;
    match reg {
        S(n) => {
            if load {
                emit_addr(f, addr, base);
                f.instruction(&W::I32Load(mem_arg()));
                f.instruction(&W::GlobalSet(abi::vfp_s_global(n)));
            } else {
                emit_addr(f, addr, base);
                f.instruction(&W::GlobalGet(abi::vfp_s_global(n)));
                f.instruction(&W::I32Store(mem_arg()));
            }
        }
        D(n) => {
            if load {
                emit_addr(f, addr, base);
                f.instruction(&W::I64Load(mem_arg()));
                set_d_bits(f, n);
            } else {
                emit_addr(f, addr, base);
                get_d_bits(f, n);
                f.instruction(&W::I64Store(mem_arg()));
            }
        }
    }
}

// --- NEON data-processing emission ----------------------------------------
//
// Every NEON operand is materialized as a wasm `v128`; a `D` register uses the low
// 64 bits. The register banks are handled by `neon_get`/`neon_set` (see the model
// in `abi`): the upper bank Q8..Q15 are `v128` globals accessed directly, the low
// bank is assembled from / scattered to the `s` globals. The widening family maps
// onto the wasm `extend_low` / `extmul_low` / `extadd_pairwise` primitives, which
// widen the low 64 bits of a `v128` - exactly the NEON long/wide semantics.

/// Push NEON register `reg` as a `v128` (a `D` lands in the low 64 bits).
fn neon_get(f: &mut Function, reg: crate::ir::NeonReg) {
    use crate::ir::NeonReg::*;
    match reg {
        Q(k) => {
            if (k as usize) >= abi::VFP_Q_HI_FIRST {
                f.instruction(&W::GlobalGet(abi::vfp_qhi_global(k)));
            } else {
                // Low bank: assemble the quad from its four aliased S registers.
                let s = 4 * k;
                f.instruction(&W::GlobalGet(abi::vfp_s_global(s)));
                f.instruction(&W::I32x4Splat);
                f.instruction(&W::GlobalGet(abi::vfp_s_global(s + 1)));
                f.instruction(&W::I32x4ReplaceLane(1));
                f.instruction(&W::GlobalGet(abi::vfp_s_global(s + 2)));
                f.instruction(&W::I32x4ReplaceLane(2));
                f.instruction(&W::GlobalGet(abi::vfp_s_global(s + 3)));
                f.instruction(&W::I32x4ReplaceLane(3));
            }
        }
        D(n) => {
            get_d_bits(f, n);
            f.instruction(&W::I64x2Splat);
        }
    }
}

/// Store the `v128` on the stack into NEON register `reg`. A `Q` writes all 128
/// bits; a `D` writes only the low 64 (leaving the sibling half of an upper-bank
/// quad intact). Uses the `L_V128A` scratch to fan a low-bank quad out to its S
/// halves (safe: any staged operands are already consumed by this point).
fn neon_set(f: &mut Function, reg: crate::ir::NeonReg) {
    use crate::ir::NeonReg::*;
    match reg {
        Q(k) => {
            if (k as usize) >= abi::VFP_Q_HI_FIRST {
                f.instruction(&W::GlobalSet(abi::vfp_qhi_global(k)));
            } else {
                let s = 4 * k;
                f.instruction(&W::LocalSet(L_V128A));
                for lane in 0..4u8 {
                    f.instruction(&W::LocalGet(L_V128A));
                    f.instruction(&W::I32x4ExtractLane(lane));
                    f.instruction(&W::GlobalSet(abi::vfp_s_global(s + lane)));
                }
            }
        }
        D(n) => {
            f.instruction(&W::I64x2ExtractLane(0));
            set_d_bits(f, n);
        }
    }
}

/// Lanewise `i{bits}x{n}.add`.
fn simd_add(bits: u8) -> W<'static> {
    match bits {
        8 => W::I8x16Add,
        16 => W::I16x8Add,
        32 => W::I32x4Add,
        64 => W::I64x2Add,
        _ => unreachable!("neon add width {bits}"),
    }
}

/// Lanewise `i{bits}x{n}.sub`.
fn simd_sub(bits: u8) -> W<'static> {
    match bits {
        8 => W::I8x16Sub,
        16 => W::I16x8Sub,
        32 => W::I32x4Sub,
        64 => W::I64x2Sub,
        _ => unreachable!("neon sub width {bits}"),
    }
}

/// Lanewise `i{bits}x{n}.mul` (undefined for 8-bit: wasm has no `i8x16.mul`).
fn simd_mul(bits: u8) -> W<'static> {
    match bits {
        16 => W::I16x8Mul,
        32 => W::I32x4Mul,
        64 => W::I64x2Mul,
        _ => unreachable!("neon mul width {bits}"),
    }
}

/// Lanewise signed/unsigned min.
fn simd_min(bits: u8, signed: bool) -> W<'static> {
    match (bits, signed) {
        (8, true) => W::I8x16MinS,
        (8, false) => W::I8x16MinU,
        (16, true) => W::I16x8MinS,
        (16, false) => W::I16x8MinU,
        (32, true) => W::I32x4MinS,
        (32, false) => W::I32x4MinU,
        _ => unreachable!("neon min width {bits}"),
    }
}

/// Lanewise signed/unsigned max.
fn simd_max(bits: u8, signed: bool) -> W<'static> {
    match (bits, signed) {
        (8, true) => W::I8x16MaxS,
        (8, false) => W::I8x16MaxU,
        (16, true) => W::I16x8MaxS,
        (16, false) => W::I16x8MaxU,
        (32, true) => W::I32x4MaxS,
        (32, false) => W::I32x4MaxU,
        _ => unreachable!("neon max width {bits}"),
    }
}

/// Lanewise integer abs / neg.
fn simd_abs(bits: u8) -> W<'static> {
    match bits {
        8 => W::I8x16Abs,
        16 => W::I16x8Abs,
        32 => W::I32x4Abs,
        64 => W::I64x2Abs,
        _ => unreachable!("neon abs width {bits}"),
    }
}
fn simd_neg(bits: u8) -> W<'static> {
    match bits {
        8 => W::I8x16Neg,
        16 => W::I16x8Neg,
        32 => W::I32x4Neg,
        64 => W::I64x2Neg,
        _ => unreachable!("neon neg width {bits}"),
    }
}

/// Widen the low half of a `v128` from `bits`-wide elements to `2*bits` (signed or
/// unsigned zero/sign extension).
fn simd_extend_low(bits: u8, signed: bool) -> W<'static> {
    match (bits, signed) {
        (8, true) => W::I16x8ExtendLowI8x16S,
        (8, false) => W::I16x8ExtendLowI8x16U,
        (16, true) => W::I32x4ExtendLowI16x8S,
        (16, false) => W::I32x4ExtendLowI16x8U,
        (32, true) => W::I64x2ExtendLowI32x4S,
        (32, false) => W::I64x2ExtendLowI32x4U,
        _ => unreachable!("neon extend width {bits}"),
    }
}

/// Widen-and-multiply the low halves of two `v128`s (`bits` -> `2*bits`).
fn simd_extmul_low(bits: u8, signed: bool) -> W<'static> {
    match (bits, signed) {
        (8, true) => W::I16x8ExtMulLowI8x16S,
        (8, false) => W::I16x8ExtMulLowI8x16U,
        (16, true) => W::I32x4ExtMulLowI16x8S,
        (16, false) => W::I32x4ExtMulLowI16x8U,
        (32, true) => W::I64x2ExtMulLowI32x4S,
        (32, false) => W::I64x2ExtMulLowI32x4U,
        _ => unreachable!("neon extmul width {bits}"),
    }
}

/// Pairwise-add adjacent `bits`-wide elements, widening to `2*bits` (8->16 or
/// 16->32 only; wasm has no 32->64 form).
fn simd_extadd_pairwise(bits: u8, signed: bool) -> W<'static> {
    match (bits, signed) {
        (8, true) => W::I16x8ExtAddPairwiseI8x16S,
        (8, false) => W::I16x8ExtAddPairwiseI8x16U,
        (16, true) => W::I32x4ExtAddPairwiseI16x8S,
        (16, false) => W::I32x4ExtAddPairwiseI16x8U,
        _ => unreachable!("neon extadd-pairwise width {bits}"),
    }
}

fn emit_neon(f: &mut Function, op: &crate::ir::NeonStmt) {
    use crate::ir::NeonStmt::*;
    match op {
        Bin { op: bop, ty, dst, a, b } => {
            use crate::ir::NeonBin::*;
            match bop {
                Add | Sub | Mul => {
                    neon_get(f, *a);
                    neon_get(f, *b);
                    f.instruction(&match bop {
                        Add => simd_add(ty.bits),
                        Sub => simd_sub(ty.bits),
                        _ => simd_mul(ty.bits),
                    });
                    neon_set(f, *dst);
                }
                Max | Min => {
                    neon_get(f, *a);
                    neon_get(f, *b);
                    f.instruction(&if matches!(bop, Max) {
                        simd_max(ty.bits, ty.signed)
                    } else {
                        simd_min(ty.bits, ty.signed)
                    });
                    neon_set(f, *dst);
                }
                Abd => {
                    // |a - b| = max(a, b) - min(a, b), avoiding the overflow of a
                    // straight sub-then-abs. Both operands are read twice.
                    neon_get(f, *a);
                    f.instruction(&W::LocalSet(L_V128A));
                    neon_get(f, *b);
                    f.instruction(&W::LocalSet(L_V128B));
                    f.instruction(&W::LocalGet(L_V128A));
                    f.instruction(&W::LocalGet(L_V128B));
                    f.instruction(&simd_max(ty.bits, ty.signed));
                    f.instruction(&W::LocalGet(L_V128A));
                    f.instruction(&W::LocalGet(L_V128B));
                    f.instruction(&simd_min(ty.bits, ty.signed));
                    f.instruction(&simd_sub(ty.bits));
                    neon_set(f, *dst);
                }
            }
        }
        MulAcc { ty, dst, a, b, sub } => {
            neon_get(f, *dst);
            neon_get(f, *a);
            neon_get(f, *b);
            f.instruction(&simd_mul(ty.bits));
            f.instruction(&if *sub { simd_sub(ty.bits) } else { simd_add(ty.bits) });
            neon_set(f, *dst);
        }
        PairAdd { ty, dst, a, b } => {
            // No wasm horizontal add: gather the even and odd source elements of the
            // (a : b) concatenation with two shuffles, then add. `a` is source bytes
            // 0..16, `b` is 16..32.
            let ebytes = (ty.bits / 8) as usize;
            let cnt = 8 / ebytes; // elements per D register
            let mut xmask = [0u8; 16];
            let mut ymask = [0u8; 16];
            for k in 0..cnt {
                let (src_base, within) =
                    if k < cnt / 2 { (0usize, k) } else { (16usize, k - cnt / 2) };
                let even = src_base + (2 * within) * ebytes;
                let odd = src_base + (2 * within + 1) * ebytes;
                for j in 0..ebytes {
                    xmask[k * ebytes + j] = (even + j) as u8;
                    ymask[k * ebytes + j] = (odd + j) as u8;
                }
            }
            neon_get(f, *a);
            f.instruction(&W::LocalSet(L_V128A));
            neon_get(f, *b);
            f.instruction(&W::LocalSet(L_V128B));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128B));
            f.instruction(&W::I8x16Shuffle(xmask));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128B));
            f.instruction(&W::I8x16Shuffle(ymask));
            f.instruction(&simd_add(ty.bits));
            neon_set(f, *dst);
        }
        Widen { ty, dst, a } => {
            neon_get(f, *a);
            f.instruction(&simd_extend_low(ty.bits, ty.signed));
            neon_set(f, *dst);
        }
        WideAddSub { sub, wide, ty, dst, a, b } => {
            // Wide form: `a` is already a Q of 2*bits elements; long form: widen `a`.
            neon_get(f, *a);
            if !wide {
                f.instruction(&simd_extend_low(ty.bits, ty.signed));
            }
            neon_get(f, *b);
            f.instruction(&simd_extend_low(ty.bits, ty.signed));
            f.instruction(&if *sub { simd_sub(ty.bits * 2) } else { simd_add(ty.bits * 2) });
            neon_set(f, *dst);
        }
        WideMul { acc, sub, ty, dst, a, b } => {
            if *acc {
                neon_get(f, *dst);
            }
            neon_get(f, *a);
            neon_get(f, *b);
            f.instruction(&simd_extmul_low(ty.bits, ty.signed));
            if *acc {
                f.instruction(&if *sub { simd_sub(ty.bits * 2) } else { simd_add(ty.bits * 2) });
            }
            neon_set(f, *dst);
        }
        WideAbd { acc, ty, dst, a, b } => {
            let w = ty.bits * 2;
            neon_get(f, *a);
            f.instruction(&simd_extend_low(ty.bits, ty.signed));
            f.instruction(&W::LocalSet(L_V128A));
            neon_get(f, *b);
            f.instruction(&simd_extend_low(ty.bits, ty.signed));
            f.instruction(&W::LocalSet(L_V128B));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128B));
            f.instruction(&simd_max(w, ty.signed));
            f.instruction(&W::LocalGet(L_V128A));
            f.instruction(&W::LocalGet(L_V128B));
            f.instruction(&simd_min(w, ty.signed));
            f.instruction(&simd_sub(w));
            if *acc {
                f.instruction(&W::LocalSet(L_V128A));
                neon_get(f, *dst);
                f.instruction(&W::LocalGet(L_V128A));
                f.instruction(&simd_add(w));
            }
            neon_set(f, *dst);
        }
        PairLong { acc, ty, dst, a } => {
            neon_get(f, *a);
            f.instruction(&simd_extadd_pairwise(ty.bits, ty.signed));
            if *acc {
                f.instruction(&W::LocalSet(L_V128A));
                neon_get(f, *dst);
                f.instruction(&W::LocalGet(L_V128A));
                f.instruction(&simd_add(ty.bits * 2));
            }
            neon_set(f, *dst);
        }
        Unary { neg, ty, dst, a } => {
            neon_get(f, *a);
            f.instruction(&if ty.float {
                if *neg { W::F32x4Neg } else { W::F32x4Abs }
            } else if *neg {
                simd_neg(ty.bits)
            } else {
                simd_abs(ty.bits)
            });
            neon_set(f, *dst);
        }
        MovImm { ty, dst, imm } => {
            let mut bytes = [0u8; 16];
            match ty.bits {
                8 => bytes = [*imm as u8; 16],
                16 => {
                    let h = (*imm as u16).to_le_bytes();
                    for i in 0..8 {
                        bytes[2 * i] = h[0];
                        bytes[2 * i + 1] = h[1];
                    }
                }
                32 => {
                    let w = imm.to_le_bytes();
                    for i in 0..4 {
                        bytes[4 * i..4 * i + 4].copy_from_slice(&w);
                    }
                }
                _ => unreachable!("neon vmov.i width {}", ty.bits),
            }
            f.instruction(&W::V128Const(i128::from_le_bytes(bytes)));
            neon_set(f, *dst);
        }
    }
}

/// N,Z,C,V for `a + b + cin`. `cin` is a runtime value (0 or 1) - an immediate for
/// add/sub/cmp, the C flag for adc/sbc. Uses i64 for an always-correct unsigned
/// carry. `cin` is emitted once into a local, since a flag read must not be
/// duplicated if it ever had a cost.
fn emit_flags_add(f: &mut Function, a: &Value, b: &Value, cin: &Value, base: u32) {
    emit_value(f, a, base);
    f.instruction(&W::LocalSet(L_T0)); // a
    emit_value(f, b, base);
    f.instruction(&W::LocalSet(L_T1)); // b
    emit_value(f, cin, base);
    f.instruction(&W::LocalSet(L_T3)); // cin
    // res = a + b + cin
    f.instruction(&W::LocalGet(L_T0));
    f.instruction(&W::LocalGet(L_T1));
    f.instruction(&W::I32Add);
    f.instruction(&W::LocalGet(L_T3));
    f.instruction(&W::I32Add);
    f.instruction(&W::LocalSet(L_T2)); // res
    // Z = res == 0
    f.instruction(&W::LocalGet(L_T2));
    f.instruction(&W::I32Eqz);
    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::Z)));
    // N = res >> 31
    f.instruction(&W::LocalGet(L_T2));
    f.instruction(&W::I32Const(31));
    f.instruction(&W::I32ShrU);
    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::N)));
    // C = (a_u64 + b_u64 + cin) >> 32
    f.instruction(&W::LocalGet(L_T0));
    f.instruction(&W::I64ExtendI32U);
    f.instruction(&W::LocalGet(L_T1));
    f.instruction(&W::I64ExtendI32U);
    f.instruction(&W::I64Add);
    f.instruction(&W::LocalGet(L_T3));
    f.instruction(&W::I64ExtendI32U);
    f.instruction(&W::I64Add);
    f.instruction(&W::I64Const(32));
    f.instruction(&W::I64ShrU);
    f.instruction(&W::I32WrapI64);
    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::C)));
    // V = (~(a^b) & (a^res)) >> 31
    f.instruction(&W::LocalGet(L_T0));
    f.instruction(&W::LocalGet(L_T1));
    f.instruction(&W::I32Xor);
    f.instruction(&W::I32Const(-1));
    f.instruction(&W::I32Xor);
    f.instruction(&W::LocalGet(L_T0));
    f.instruction(&W::LocalGet(L_T2));
    f.instruction(&W::I32Xor);
    f.instruction(&W::I32And);
    f.instruction(&W::I32Const(31));
    f.instruction(&W::I32ShrU);
    f.instruction(&W::GlobalSet(abi::flag_global(abi::Flag::V)));
}

/// Emit a guest address as a linear-memory offset (guest addr - base).
fn emit_addr(f: &mut Function, addr: &Value, base: u32) {
    emit_value(f, addr, base);
    f.instruction(&W::I32Const(base as i32));
    f.instruction(&W::I32Sub);
}

fn emit_value(f: &mut Function, v: &Value, base: u32) {
    match v {
        Value::Imm(x) => {
            f.instruction(&W::I32Const(*x as i32));
        }
        Value::Reg(r) => {
            f.instruction(&W::GlobalGet(abi::reg_global(*r as usize)));
        }
        Value::Not(a) => {
            emit_value(f, a, base);
            f.instruction(&W::I32Const(-1));
            f.instruction(&W::I32Xor);
        }
        Value::Flag(flag) => {
            // Flag globals hold 0 or 1, so a plain read is the flag's value.
            f.instruction(&W::GlobalGet(abi::flag_global(*flag)));
        }
        Value::Clz(a) => {
            emit_value(f, a, base);
            f.instruction(&W::I32Clz);
        }
        Value::Bin(op, a, b) => {
            emit_value(f, a, base);
            emit_value(f, b, base);
            f.instruction(&binop(*op));
        }
        Value::Load { addr, size, signed } => {
            emit_addr(f, addr, base);
            f.instruction(&load_op(*size, *signed));
        }
    }
}

fn binop(op: BinOp) -> W<'static> {
    match op {
        BinOp::Add => W::I32Add,
        BinOp::Sub => W::I32Sub,
        BinOp::And => W::I32And,
        BinOp::Or => W::I32Or,
        BinOp::Xor => W::I32Xor,
        BinOp::Shl => W::I32Shl,
        BinOp::Lsr => W::I32ShrU,
        BinOp::Asr => W::I32ShrS,
        BinOp::Mul => W::I32Mul,
    }
}

fn mem_arg() -> MemArg {
    MemArg { offset: 0, align: 0, memory_index: 0 }
}

fn load_op(size: MemSize, signed: bool) -> W<'static> {
    match (size, signed) {
        (MemSize::Byte, false) => W::I32Load8U(mem_arg()),
        (MemSize::Byte, true) => W::I32Load8S(mem_arg()),
        (MemSize::Half, false) => W::I32Load16U(mem_arg()),
        (MemSize::Half, true) => W::I32Load16S(mem_arg()),
        (MemSize::Word, _) => W::I32Load(mem_arg()),
    }
}

fn store_op(size: MemSize) -> W<'static> {
    match size {
        MemSize::Byte => W::I32Store8(mem_arg()),
        MemSize::Half => W::I32Store16(mem_arg()),
        MemSize::Word => W::I32Store(mem_arg()),
    }
}
