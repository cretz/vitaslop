//! Vita-agnostic ARMv7-A + Thumb-2 + NEON + VFP to WASM transpiler.
//!
//! Input is a code image, a set of entry-point addresses, and
//! relocation/provenance facts ([`Program`]). Output is raw WASM plus the
//! guest-address-to-export dispatch map ([`Artifact`]). Nothing Vita-specific
//! belongs in this crate.
//!
//! The pipeline is decode (yaxpeax) -> a small IR ([`Stmt`]/[`Value`]) that we
//! optimize over (e.g. const-folding `adr`) -> emit (wasm-encoder). Today only
//! a handful of instructions are lifted; the IR seam is what lets that grow
//! into a real optimizing transpiler rather than a 1:1 emitter.

pub mod abi;

use wasm_encoder::{
    CodeSection, ConstExpr, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    GlobalSection, GlobalType, ImportSection, Instruction, MemorySection, MemoryType, Module,
    TypeSection, ValType,
};
use yaxpeax_arch::{Decoder, U8Reader};
use yaxpeax_arm::armv7::{InstDecoder, Opcode, Operand};

/// Function index of the imported `svc` trap. Imports come first, so it is 0.
const SVC_FUNC: u32 = 0;

/// A code image to transpile: the ARM blob, where it loads, which addresses to
/// start decoding from, and how guest imports wire to host functions.
pub struct Program<'a> {
    /// The ARM code/data image.
    pub code: &'a [u8],
    /// Guest address the image loads at.
    pub base: u32,
    /// Entry points to transpile (each becomes a block).
    pub entries: &'a [u32],
    /// Extern (imported-function) wiring. Unused until NID imports land.
    pub externs: &'a [Extern],
    /// Syscall numbers (guest r7) that do not return, so a `svc` with a
    /// statically-known one of them ends a block. Supplied by the host so the
    /// transpiler stays free of any particular syscall convention.
    pub noreturn_svc: &'a [u32],
}

/// A guest address that dispatches to a host import (the Vita NID mechanism,
/// later). Present so the public shape is stable; not consumed yet.
pub struct Extern {
    pub addr: u32,
    pub import: u32,
}

/// The transpiler output: the WASM blob plus the dispatch map the runtime needs
/// to call guest addresses.
pub struct Artifact {
    /// The emitted WASM module bytes.
    pub wasm: Vec<u8>,
    /// One entry per transpiled block: guest address and its WASM export name.
    pub blocks: Vec<Block>,
}

/// A transpiled block: the guest address it starts at and the WASM function
/// exported for it.
pub struct Block {
    pub addr: u32,
    pub export: String,
}

/// Why transpilation failed.
#[derive(Debug)]
pub enum Error {
    /// The decoder could not decode the bytes at `addr`.
    Decode { addr: u32 },
    /// A decoded instruction is not lifted yet.
    Unsupported { addr: u32, opcode: Opcode },
    /// An operand had an unexpected shape for its opcode.
    Operand { addr: u32 },
}

/// A computed value in the IR. Only what the current instruction set needs;
/// grows toward a full SSA value graph.
enum Value {
    Const(u32),
    Reg(u8),
    Add(Box<Value>, Box<Value>),
}

/// One lowered effect of a guest instruction.
enum Stmt {
    /// `r[reg] = value`
    SetReg { reg: u8, value: Value },
}

/// What ends a segment.
enum Boundary {
    /// A host `svc`. Promoted registers are flushed to their globals, the trap
    /// runs, then (unless it was EXIT) the next segment reloads and continues.
    Svc(u32),
    /// The block ran off the end of the image: nothing more to run.
    FellThrough,
}

/// A boundary-free run within a block: its statements, the registers worth
/// caching in locals for just this run, and the boundary that ends it. Register
/// caching lives at this granularity because every boundary spills the locals to
/// their globals (the host observes and can change registers there), so a local
/// only earns its keep within a single segment.
struct Segment {
    stmts: Vec<Stmt>,
    /// Registers promoted to a local for this segment, ascending.
    promoted: Vec<u8>,
    boundary: Boundary,
}

/// A decoded/lifted block: its entry address and its segments. The block is the
/// wasm function (and owns the local slots); segments are the caching spans.
struct Lowered {
    addr: u32,
    segments: Vec<Segment>,
}

/// Transpile `program` into a WASM module and its dispatch map.
pub fn transpile(program: &Program) -> Result<Artifact, Error> {
    let decoder = InstDecoder::default();
    let mut lowered = Vec::new();
    for &entry in program.entries {
        lowered.push(lift_block(
            program.code,
            program.base,
            entry,
            &decoder,
            program.noreturn_svc,
        )?);
    }
    let wasm = emit(&lowered, program.base, program.code.len());
    let blocks = lowered
        .iter()
        .map(|b| Block {
            addr: b.addr,
            export: abi::block_export(b.addr),
        })
        .collect();
    Ok(Artifact { wasm, blocks })
}

/// Decode and lift a single straight-line block, starting at `entry`. Stops at
/// a noreturn `svc` (exit) or when it runs off the end of the image.
fn lift_block(
    code: &[u8],
    base: u32,
    entry: u32,
    decoder: &InstDecoder,
    noreturn_svc: &[u32],
) -> Result<Lowered, Error> {
    let mut segments = Vec::new();
    let mut stmts = Vec::new();
    // Per-segment access tally (reset at each boundary by `close_segment`). We
    // decide promotion per segment because every boundary spills the locals to
    // their globals anyway, so only density *between* boundaries earns a local.
    let mut seg_uses = [0u32; abi::REG_COUNT];
    // Known-constant register values, tracked across the whole block. Lets us
    // recognize a noreturn `svc` (r7 = an exit syscall) and end the block there,
    // rather than decoding whatever data follows it.
    let mut regvals = [None; abi::REG_COUNT];
    let mut addr = entry;
    loop {
        let off = addr.wrapping_sub(base) as usize;
        if off + 4 > code.len() {
            // Ran past the image: close the trailing segment and stop.
            segments.push(close_segment(
                &mut stmts,
                &mut seg_uses,
                Boundary::FellThrough,
            ));
            break;
        }
        let mut reader = U8Reader::new(&code[off..]);
        let inst = decoder
            .decode(&mut reader)
            .map_err(|_| Error::Decode { addr })?;
        match inst.opcode {
            // adr rd, label  ->  rd = pc + imm (pc = addr + 8). Const-folded.
            Opcode::ADR => {
                let rd = reg(inst.operands[0], addr)?;
                let imm = imm32(inst.operands[2], addr)?;
                let value = addr.wrapping_add(8).wrapping_add(imm);
                seg_uses[rd as usize] += 1;
                regvals[rd as usize] = Some(value);
                stmts.push(Stmt::SetReg {
                    reg: rd,
                    value: Value::Const(value),
                });
            }
            // mov rd, #imm
            Opcode::MOV if matches!(inst.operands[1], Operand::Imm32(_)) => {
                let rd = reg(inst.operands[0], addr)?;
                let imm = imm32(inst.operands[1], addr)?;
                seg_uses[rd as usize] += 1;
                regvals[rd as usize] = Some(imm);
                stmts.push(Stmt::SetReg {
                    reg: rd,
                    value: Value::Const(imm),
                });
            }
            // add{s} rd, rn, rm  ->  rd = rn + rm (flags not modeled yet).
            Opcode::ADD => {
                let rd = reg(inst.operands[0], addr)?;
                let rn = reg(inst.operands[1], addr)?;
                let rm = reg(inst.operands[2], addr)?;
                seg_uses[rn as usize] += 1;
                seg_uses[rm as usize] += 1;
                seg_uses[rd as usize] += 1;
                regvals[rd as usize] = None;
                stmts.push(Stmt::SetReg {
                    reg: rd,
                    value: Value::Add(Box::new(Value::Reg(rn)), Box::new(Value::Reg(rm))),
                });
            }
            // svc #imm: a host call, and a segment boundary.
            Opcode::SVC => {
                let imm = imm32(inst.operands[0], addr)?;
                segments.push(close_segment(&mut stmts, &mut seg_uses, Boundary::Svc(imm)));
                // A syscall whose number (r7) is statically a noreturn one ends
                // the block; otherwise execution continues into the next segment.
                if regvals[7].is_some_and(|nr| noreturn_svc.contains(&nr)) {
                    break;
                }
            }
            opcode => return Err(Error::Unsupported { addr, opcode }),
        }
        addr = addr.wrapping_add(4); // ARM fixed instruction width
    }
    Ok(Lowered {
        addr: entry,
        segments,
    })
}

/// Close a segment: take its statements, decide which registers were hot enough
/// within it to cache in a local (accessed past the threshold, since each spills
/// at the boundary regardless), reset the tally, and pair them with the boundary
/// that ends the segment.
fn close_segment(
    stmts: &mut Vec<Stmt>,
    seg_uses: &mut [u32; abi::REG_COUNT],
    boundary: Boundary,
) -> Segment {
    let promoted = (0..abi::REG_COUNT)
        .filter(|&r| seg_uses[r] > abi::LOCAL_PROMOTION_THRESHOLD)
        .map(|r| r as u8)
        .collect();
    seg_uses.fill(0);
    Segment {
        stmts: std::mem::take(stmts),
        promoted,
        boundary,
    }
}

fn reg(op: Operand, addr: u32) -> Result<u8, Error> {
    match op {
        Operand::Reg(r) => Ok(r.number()),
        _ => Err(Error::Operand { addr }),
    }
}

fn imm32(op: Operand, addr: u32) -> Result<u32, Error> {
    match op {
        Operand::Imm32(v) => Ok(v),
        Operand::Imm12(v) => Ok(v as u32),
        _ => Err(Error::Operand { addr }),
    }
}

/// Emit the WASM module for the lowered blocks (see [`abi`] for the layout).
fn emit(blocks: &[Lowered], base: u32, code_len: usize) -> Vec<u8> {
    // Types: svc import `(i32) -> ()`, block `() -> i32`.
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32], []);
    types.ty().function([], [ValType::I32]);
    let svc_type = 0;
    let block_type = 1;

    let mut imports = ImportSection::new();
    imports.import(
        abi::SVC_MODULE,
        abi::SVC_NAME,
        EntityType::Function(svc_type),
    );

    let mut funcs = FunctionSection::new();
    for _ in blocks {
        funcs.function(block_type);
    }

    // Enough linear-memory pages to cover the guest image at its load base.
    let end = base as u64 + code_len as u64;
    let pages = end.div_ceil(abi::PAGE_SIZE as u64).max(1);
    let mut mems = MemorySection::new();
    mems.memory(MemoryType {
        minimum: pages,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    // The register file: 16 mutable i32 globals, r0..r15, exported by name.
    let mut globals = GlobalSection::new();
    for _ in 0..abi::REG_COUNT {
        globals.global(
            GlobalType {
                val_type: ValType::I32,
                mutable: true,
                shared: false,
            },
            &ConstExpr::i32_const(0),
        );
    }

    let mut exports = ExportSection::new();
    exports.export(abi::MEMORY_EXPORT, ExportKind::Memory, 0);
    for i in 0..abi::REG_COUNT {
        exports.export(&abi::reg_export(i), ExportKind::Global, abi::reg_global(i));
    }

    let mut code = CodeSection::new();
    for (i, block) in blocks.iter().enumerate() {
        // Imported funcs occupy the low indices; block funcs follow.
        let func_index = SVC_FUNC + 1 + i as u32;
        exports.export(&abi::block_export(block.addr), ExportKind::Func, func_index);
        code.function(&emit_block(block));
    }

    let mut module = Module::new();
    module
        .section(&types)
        .section(&imports)
        .section(&funcs)
        .section(&mems)
        .section(&globals)
        .section(&exports)
        .section(&code);
    module.finish()
}

/// Per-segment register caching: which of this segment's registers live in a
/// wasm local, and the block-wide local slot each uses.
struct RegCache<'a> {
    /// `local_of[r]` is the local slot for register `r` if it is cached in this
    /// segment, else `None` (accessed through its global).
    local_of: [Option<u32>; abi::REG_COUNT],
    /// This segment's promoted registers, ascending.
    promoted: &'a [u8],
}

impl RegCache<'_> {
    /// Push the current value of register `r`.
    fn get(&self, f: &mut Function, r: u8) {
        match self.local_of[r as usize] {
            Some(l) => f.instruction(&Instruction::LocalGet(l)),
            None => f.instruction(&Instruction::GlobalGet(abi::reg_global(r as usize))),
        };
    }

    /// Store the top of stack into register `r`.
    fn set(&self, f: &mut Function, r: u8) {
        match self.local_of[r as usize] {
            Some(l) => f.instruction(&Instruction::LocalSet(l)),
            None => f.instruction(&Instruction::GlobalSet(abi::reg_global(r as usize))),
        };
    }

    /// Load this segment's promoted registers from their globals into locals.
    fn load(&self, f: &mut Function) {
        for &r in self.promoted {
            f.instruction(&Instruction::GlobalGet(abi::reg_global(r as usize)));
            f.instruction(&Instruction::LocalSet(self.local_of[r as usize].unwrap()));
        }
    }

    /// Flush this segment's promoted registers back to their globals at the
    /// boundary, so the host and the next segment see current registers.
    fn flush(&self, f: &mut Function) {
        for &r in self.promoted {
            f.instruction(&Instruction::LocalGet(self.local_of[r as usize].unwrap()));
            f.instruction(&Instruction::GlobalSet(abi::reg_global(r as usize)));
        }
    }
}

fn emit_block(block: &Lowered) -> Function {
    // A wasm local is declared once per function, so give each register promoted
    // in *any* segment a block-wide slot; a segment that does not promote it just
    // leaves the slot untouched. Slots are keyed by register, so a register
    // promoted in several segments reuses the same slot (reloaded each time).
    let mut slot_of = [None; abi::REG_COUNT];
    let mut slots = 0u32;
    for seg in &block.segments {
        for &r in &seg.promoted {
            if slot_of[r as usize].is_none() {
                slot_of[r as usize] = Some(slots);
                slots += 1;
            }
        }
    }
    let locals = if slots == 0 {
        Vec::new()
    } else {
        vec![(slots, ValType::I32)]
    };
    let mut f = Function::new(locals);

    for seg in &block.segments {
        // Cache only this segment's promoted registers (into their block-wide
        // slots); everything else goes straight through the globals.
        let mut local_of = [None; abi::REG_COUNT];
        for &r in &seg.promoted {
            local_of[r as usize] = slot_of[r as usize];
        }
        let cache = RegCache {
            local_of,
            promoted: &seg.promoted,
        };

        cache.load(&mut f);
        for stmt in &seg.stmts {
            match stmt {
                Stmt::SetReg { reg, value } => {
                    emit_value(&mut f, &cache, value);
                    cache.set(&mut f, *reg);
                }
            }
        }
        cache.flush(&mut f);

        if let Boundary::Svc(imm) = seg.boundary {
            f.instruction(&Instruction::I32Const(imm as i32));
            f.instruction(&Instruction::Call(SVC_FUNC));
        }
    }

    f.instruction(&Instruction::I32Const(abi::HALT));
    f.instruction(&Instruction::End);
    f
}

fn emit_value(f: &mut Function, cache: &RegCache, value: &Value) {
    match value {
        Value::Const(v) => {
            f.instruction(&Instruction::I32Const(*v as i32));
        }
        Value::Reg(src) => cache.get(f, *src),
        Value::Add(a, b) => {
            emit_value(f, cache, a);
            emit_value(f, cache, b);
            f.instruction(&Instruction::I32Add);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpiles_hello() {
        let code: [u8; 16] = [
            0x08, 0x00, 0x8f, 0xe2, // adr r0, msg
            0x0d, 0x10, 0xa0, 0xe3, // mov r1, #13
            0x00, 0x00, 0x00, 0xef, // svc #0
            0x01, 0x00, 0x00, 0xef, // svc #1
        ];
        let artifact = transpile(&Program {
            code: &code,
            base: 0x10000,
            entries: &[0x10000],
            externs: &[],
            noreturn_svc: &[],
        })
        .expect("transpile");
        assert!(!artifact.wasm.is_empty());
        assert_eq!(artifact.blocks.len(), 1);
        assert_eq!(artifact.blocks[0].addr, 0x10000);
        assert_eq!(artifact.blocks[0].export, "b_10000");
    }
}
