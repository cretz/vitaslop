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
const L_I32_COUNT: u32 = 4;

/// Assemble the full wasm module for `funcs`. `func_index` maps a guest function
/// address to its wasm function index. `mem_bytes` sizes the exported linear
/// memory; `base` is the guest image base for the address rebase.
pub fn emit_module(
    funcs: &[Func],
    func_index: &BTreeMap<u32, u32>,
    base: u32,
    mem_bytes: u32,
) -> Vec<u8> {
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], []); // svc / import: (i32) -> ()
    let host_ty = 0;
    types.ty().function([], []); // guest function: () -> ()
    let func_ty = 1;

    let mut imports = ImportSection::new();
    imports.import(abi::IMPORT_MODULE, abi::SVC_NAME, wasm_encoder::EntityType::Function(host_ty));
    imports.import(abi::IMPORT_MODULE, abi::IMPORT_NAME, wasm_encoder::EntityType::Function(host_ty));

    let mut function_section = FunctionSection::new();
    for _ in funcs {
        function_section.function(func_ty);
    }

    let pages = (mem_bytes as u64).div_ceil(abi::PAGE_SIZE as u64).max(1);
    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: pages,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    // 16 register globals + 4 flag globals, all mutable i32.
    let mut globals = GlobalSection::new();
    for _ in 0..abi::GLOBAL_COUNT {
        globals.global(
            GlobalType { val_type: ValType::I32, mutable: true, shared: false },
            &ConstExpr::i32_const(0),
        );
    }

    let mut exports = ExportSection::new();
    exports.export(abi::MEMORY_EXPORT, ExportKind::Memory, 0);
    for i in 0..abi::REG_COUNT {
        exports.export(&abi::reg_export(i), ExportKind::Global, abi::reg_global(i));
    }
    for f in [abi::Flag::N, abi::Flag::Z, abi::Flag::C, abi::Flag::V] {
        exports.export(abi::flag_export(f), ExportKind::Global, abi::flag_global(f));
    }

    let mut code = CodeSection::new();
    for (i, func) in funcs.iter().enumerate() {
        let idx = IMPORT_FUNCS + i as u32;
        exports.export(&abi::func_export(func.addr), ExportKind::Func, idx);
        code.function(&emit_func(func, func_index, base));
    }

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

/// Emit one guest function as a wasm function: a dispatch loop over its blocks.
fn emit_func(func: &Func, func_index: &BTreeMap<u32, u32>, base: u32) -> Function {
    // Locals: $bb + three i32 scratch temps (used by flag computation).
    let mut f = Function::new([(L_I32_COUNT, ValType::I32)]);

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
        Stmt::FlagsAdd { a, b, cin } => emit_flags_add(f, a, b, *cin, base),
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
        Stmt::Call { target } => {
            let idx = *func_index.get(target).expect("callee index");
            f.instruction(&W::Call(idx));
        }
        Stmt::Guard(cond, body) => {
            emit_cond(f, *cond);
            f.instruction(&W::If(BlockType::Empty));
            for s in body {
                emit_stmt(f, s, func_index, base);
            }
            f.instruction(&W::End);
        }
    }
}

/// N,Z,C,V for `a + b + cin`. Uses i64 for an always-correct unsigned carry.
fn emit_flags_add(f: &mut Function, a: &Value, b: &Value, cin: u32, base: u32) {
    emit_value(f, a, base);
    f.instruction(&W::LocalSet(L_T0)); // a
    emit_value(f, b, base);
    f.instruction(&W::LocalSet(L_T1)); // b
    // res = a + b + cin
    f.instruction(&W::LocalGet(L_T0));
    f.instruction(&W::LocalGet(L_T1));
    f.instruction(&W::I32Add);
    f.instruction(&W::I32Const(cin as i32));
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
    f.instruction(&W::I64Const(cin as i64));
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
