//! Decode + lower: turn a code image into per-function IR ([`ir::Func`]).
//!
//! Discovery is reachability-driven, not a linear sweep: starting from a
//! function entry we follow only real control-flow edges (fall-through, taken/
//! not-taken branches), so inline literal pools and the padding after an
//! unconditional branch are never mistaken for instructions - nothing branches
//! into them. Direct `bl`/`blx` targets are recorded as separate functions to
//! discover; they are never pulled into the caller's body. Thumb `IT` state is
//! tracked along fall-through edges so predicated instructions (which yaxpeax
//! decodes as unconditional) get their real condition.

use std::collections::{BTreeMap, BTreeSet};

use yaxpeax_arch::{Decoder, U8Reader};
use yaxpeax_arm::armv7::{
    ConditionCode, InstDecoder, Instruction, NeonOp, Opcode, Operand, Reg, RegShift, RegShiftStyle,
    SIMDDataType, SIMDElement, SIMDSizeCode, ShiftStyle, VfpType,
};

use crate::Error;
use crate::ir::{
    Block, BinOp, ElemLane, FBinOp, FlagMask, Func, MemSize, NeonBin, NeonReg, NeonStmt, NeonType,
    Stmt, Term, Value, VfpOp, VfpReg,
};

/// What lowering records for a flag statement's observable set.
///
/// Lowering sees one instruction at a time and cannot know which flags a later block
/// reads, so it claims all of them. [`crate::flags::annotate`] runs over the finished
/// function and narrows each one. Keeping the pessimistic value HERE rather than as an
/// implicit default is deliberate: a build that skips the liveness pass emits exactly the
/// code it emitted before the pass existed.
const ALL_FLAGS: FlagMask = FlagMask::ALL;

/// A resolved import: the guest stub address a `bl`/`blx` targets maps to a
/// dense host-import index.
///
/// `redirects` handles inter-module calls in a multi-module link: a call to one
/// module's import stub is rewritten to a *direct* call to the exporting module's
/// function (both live in the same unified wasm module), so an inter-module call
/// is a plain guest-to-guest call with no host round-trip. Redirection is applied
/// to a branch target before the host-import lookup, so a redirected stub is
/// never also treated as a host import.
/// A redirect also carries the exporting function's instruction set, because the
/// stub in between is not what runs: a Thumb caller reaches an ARM veneer that the
/// linker has already resolved away, so neither the caller's mode nor the branch
/// that got there says anything about the callee's.
pub struct Imports<'a> {
    map: &'a BTreeMap<u32, u32>,
    redirects: &'a BTreeMap<u32, (u32, bool)>,
}

impl<'a> Imports<'a> {
    pub fn new(map: &'a BTreeMap<u32, u32>, redirects: &'a BTreeMap<u32, (u32, bool)>) -> Self {
        Imports { map, redirects }
    }
    fn get(&self, addr: u32) -> Option<u32> {
        self.map.get(&addr).copied()
    }
    /// Rewrite an inter-module import stub address to the real callee address it
    /// resolves to; a non-redirected address is returned unchanged.
    fn resolve(&self, addr: u32) -> u32 {
        self.redirects.get(&addr).map(|&(t, _)| t).unwrap_or(addr)
    }
    /// Resolve a call target AND the mode to decode it in. `mode` is what the branch
    /// itself selects (the caller's mode for `bl`, the other one for `blx`); it is
    /// used only when the target is not a redirect, since a redirect's callee mode
    /// comes from the export, not from the call site.
    fn resolve_call(&self, addr: u32, mode: bool) -> (u32, bool) {
        self.redirects.get(&addr).copied().unwrap_or((addr, mode))
    }
}

/// Result of discovering one function: its IR, the guest addresses of the direct
/// callees found in it, and any code pointers it materializes (address-taken
/// functions - e.g. a thread entry passed to sceKernelCreateThread - which the
/// direct-call closure alone would never reach).
pub struct Discovered {
    pub func: Func,
    /// Addresses where an instruction failed to DECODE and the lenient build isolated it
    /// into a trapping block (empty in the strict build, which fails the whole function
    /// instead).
    ///
    /// These are collected because they are the gaps the strict diagnostic pass CANNOT
    /// see. That pass walks from the entry points and follows DIRECT calls, so a function
    /// reached only through a vtable - which is where a C++ engine keeps its hot
    /// per-object work - is never visited by it, and its decode gaps go unlisted while
    /// the runtime lifts the function around them. That blind spot is expensive: two
    /// undecodable NEON instructions in one title's audio mixer truncated its inner loop,
    /// and the only visible symptom was a null dereference several calls away.
    pub trap_leaders: Vec<u32>,
    /// Direct callees as `(address, thumb)`: the mode to decode each in. A `blx
    /// <label>` interworks, so a callee's mode is NOT always the caller's.
    pub callees: Vec<(u32, bool)>,
    /// Thumb code pointers this function materializes (odd `movw`/`movt`
    /// constants, and odd targets of a register-indirect `blx`/`bx`).
    pub code_pointers: Vec<u32>,
    /// ARM-mode code pointers: even constants used as the target of a
    /// register-indirect `blx`/`bx` (a Thumb function reaching an ARM helper via
    /// `blx reg`). Seeded as tentative ARM entries, dropped if they fail to decode.
    pub arm_code_pointers: Vec<u32>,
}

/// Per-register tracked immediate constants along a straight run, so a `movt`
/// completing a `movw` can be recognized as a full 32-bit value. Index is the
/// register number; `None` means "not a known constant here". `[7]` doubles as
/// the r7 tracking a noreturn `svc` needs.
type RegConsts = [Option<u32>; 16];

/// Upper bound on the instructions in one discovered function. Real ARM functions
/// are far smaller (the largest seen in a shipping title is a few thousand); exceeding this
/// means discovery is walking data as code, so the "function" is rejected.
const MAX_FUNC_INSNS: usize = 16384;

/// Decode one instruction at guest `addr`, returning it and its byte length.
fn decode_at(
    decoder: &InstDecoder,
    code: &[u8],
    base: u32,
    addr: u32,
    thumb: bool,
) -> Result<(Instruction, u32), Error> {
    let off = addr.wrapping_sub(base) as usize;
    let end = off.checked_add(if thumb { 2 } else { 4 });
    if end.is_none_or(|e| e > code.len()) {
        return Err(Error::Decode { addr });
    }
    let mut reader = U8Reader::new(&code[off..]);
    let inst = decoder
        .decode(&mut reader)
        .map_err(|_| Error::Decode { addr })?;
    let len = if !thumb || inst.wide { 4 } else { 2 };
    Ok((inst, len))
}

/// The control-flow shape of a decoded instruction, used to find successors.
enum Flow {
    /// Continue to `addr + len` only.
    Seq,
    /// A call: continue to `addr + len` (the callee returns). `guest` is the
    /// callee address and the instruction-set mode to decode it in (true = Thumb)
    /// if it is translated code, else it is a host import. The mode is carried
    /// because `blx <label>` INTERWORKS - the callee is decoded in the opposite
    /// mode to the caller - and a callee decoded in the wrong mode does not fail,
    /// it silently becomes different instructions.
    Call { guest: Option<(u32, bool)> },
    /// Unconditional branch: successor is `target` only.
    Jump(u32),
    /// An unconditional branch that leaves this function to another translated
    /// function - a tail call (`return other(...)`). `target` is discovered as its
    /// own callee, run as a call, then this function returns to its caller (lr is
    /// unchanged, so the callee's return unwinds past us). It is NOT pulled into
    /// this body. Distinguished from [`Jump`] (a local branch) by [`is_tail_call`].
    /// Carries the callee's mode like [`Call`]; a plain `b` never interworks, so it
    /// is always the caller's own mode.
    TailCall(u32, bool),
    /// Conditional/zero branch: successors are `target` and `addr + len`.
    Fork(u32),
    /// Returns to caller; no successors.
    Return,
    /// Stops without returning; no successors.
    Halt,
}

/// Forward reach of a local (in-function) unconditional branch. A branch beyond
/// this from the function entry cannot be a local label - no real function is this
/// large (the largest in a shipping title is a few thousand instructions, well
/// under 32 KiB) - so it is a tail call into another function. Kept comfortably
/// below the cross-function hops that motivate this (library tail calls in a
/// statically-linked image land tens of KiB away) yet above any real function.
const TAIL_CALL_FORWARD: u32 = 0x8000;

/// True if an unconditional `b`/`b.w` to `target` from a function entered at
/// `entry` is a tail call into a different function rather than a local branch. A
/// local label always lies at or after the entry and within the function body; a
/// target before the entry, or far past it, belongs to another function. This is
/// what lets a `b.w` to a library routine (common in a statically-linked image,
/// where calls are not import stubs) be run as a call instead of dragging the
/// callee - and its own tail-call chain - into this function's body.
fn is_tail_call(target: u32, entry: u32) -> bool {
    target < entry || target.wrapping_sub(entry) >= TAIL_CALL_FORWARD
}

/// The pc-relative target of a branch instruction, if it has one. yaxpeax measures
/// the offset from the instruction address (already folding in the pipeline pc bias),
/// so the target is `addr + 2*off` for a Thumb branch (halfwords) and `addr + 4*off`
/// for an ARM branch (words) - NOT `pc + off`; adding the `pc+4`/`pc+8` bias again
/// would land the target 4/8 bytes past the real callee. `blx <label>` INTERWORKS: from
/// Thumb it switches to ARM and word-aligns the pc (`Align(PC,4)`), so from a
/// non-word-aligned address its target is rounded down to the next word - otherwise it
/// lands 2 bytes past the callee. From ARM it switches to Thumb, whose targets are
/// halfword granular, so the same rounding there would land 2 bytes BEFORE the callee.
fn branch_target(inst: &Instruction, addr: u32, thumb: bool) -> Option<u32> {
    for op in &inst.operands {
        match op {
            Operand::BranchThumbOffset(off) => {
                let t = addr.wrapping_add((2 * off) as u32);
                return Some(if inst.opcode == Opcode::BLX && thumb { t & !3 } else { t });
            }
            Operand::BranchOffset(off) => {
                // ARM branch: yaxpeax's `off` is in words from `addr`, pipeline bias
                // already included, so the target is `addr + 4*off`.
                return Some(addr.wrapping_add((4 * off) as u32));
            }
            _ => {}
        }
    }
    None
}

/// True if this ldm/pop register list writes the pc (bit 15) - i.e. it returns.
fn writes_pc(inst: &Instruction) -> bool {
    inst.operands.iter().any(|op| match op {
        Operand::RegList(mask) => mask & (1 << 15) != 0,
        _ => false,
    })
}

/// Classify an instruction's control flow.
fn flow(
    inst: &Instruction,
    addr: u32,
    len: u32,
    thumb: bool,
    entry: u32,
    imports: &Imports,
    r7: Option<u32>,
    noreturn_svc: &[u32],
) -> Flow {
    match inst.opcode {
        // A plain `b` never interworks, so the mode it selects is the caller's own -
        // unless the target resolves through a redirect, whose export names its own.
        Opcode::B => match branch_target(inst, addr, thumb).map(|t| imports.resolve_call(t, thumb)) {
            // `b .` (branch to self) is an idle spin: treat as Halt so we do not
            // emit an infinite wasm loop the host cannot leave.
            Some((t, _)) if t == addr => Flow::Halt,
            // An unconditional branch to an import stub/veneer is a tail call
            // (`return memset(...)`): it transfers out of the function, so it has
            // no in-function successor and adds no callee - lowering runs the
            // import then returns.
            Some((t, _)) if inst.condition == ConditionCode::AL && imports.get(t).is_some() => {
                Flow::Return
            }
            // An unconditional branch out of this function to another translated
            // function is a tail call: run it as a call and return, rather than
            // inlining the callee (and its own tail-call chain) into this body -
            // which is what makes statically-linked `b.w library_fn` explode the
            // function span. A near forward branch is a local label - inline it.
            Some((t, t_thumb)) if inst.condition == ConditionCode::AL && is_tail_call(t, entry) => {
                Flow::TailCall(t, t_thumb)
            }
            Some((t, _)) if inst.condition == ConditionCode::AL => Flow::Jump(t),
            Some((t, _)) => Flow::Fork(t),
            None => Flow::Halt,
        },
        Opcode::CBZ | Opcode::CBNZ => match branch_target(inst, addr, thumb) {
            Some(t) => Flow::Fork(t),
            None => Flow::Halt,
        },
        // `bl <label>` stays in the caller's instruction set; `blx <label>` switches to
        // the other one. Decoding an ARM callee as Thumb does not fail - it decodes into
        // an entirely different, plausible-looking function - so the mode has to travel
        // with the callee address rather than being inherited from the caller. A target
        // that resolves through an inter-module redirect takes the EXPORT's mode instead:
        // the veneer the branch actually names has been resolved away.
        Opcode::BL | Opcode::BLX => match branch_target(inst, addr, thumb)
            .map(|t| imports.resolve_call(t, if inst.opcode == Opcode::BLX { !thumb } else { thumb }))
        {
            Some((t, _)) if imports.get(t).is_some() => Flow::Call { guest: None },
            Some((t, t_thumb)) => Flow::Call { guest: Some((t, t_thumb)) },
            // A register-target `blx rN` is an indirect call through a function
            // pointer: it returns here, so continue to the fall-through, but the
            // target is not a statically-known callee (the dispatcher resolves it
            // at runtime).
            None if matches!(inst.operands[0], Operand::Reg(_)) => Flow::Call { guest: None },
            None => Flow::Halt,
        },
        Opcode::BX => match inst.operands[0] {
            // `bx pc`: the interworking veneer (`bx pc; nop; <ARM code>`) that a
            // homebrew toolchain emits to enter ARM from Thumb. The target is the pc
            // VALUE - Align(addr+4, 4) in Thumb, addr+8 in ARM - and it is ARM code
            // (a pc value has bit 0 clear). Decoding on into the `nop` and the ARM
            // words as Thumb would make plausible garbage, so this is a tail call to
            // a separate function in ARM mode, not a fall-through.
            Operand::Reg(r) if r.number() == 15 => {
                let pc = if thumb { addr.wrapping_add(4) & !3 } else { addr.wrapping_add(8) };
                Flow::TailCall(pc, false)
            }
            // `bx lr` returns; `bx rN` is an indirect tail call (dispatch to the
            // target, then return). Either way there is no in-function successor.
            Operand::Reg(_) => Flow::Return,
            _ => Flow::Halt,
        },
        Opcode::POP | Opcode::LDM(..) if writes_pc(inst) => Flow::Return,
        // A breakpoint traps (abort path); it has no in-function successor, so
        // decoding must not run past it into the literal pool that often follows.
        Opcode::BKPT => Flow::Halt,
        Opcode::SVC => {
            if r7.is_some_and(|n| noreturn_svc.contains(&n)) {
                Flow::Halt
            } else {
                Flow::Seq
            }
        }
        // `ldr pc, [...]` / `mov pc, rN`: a return (`ldr pc, [sp], #4` is how gcc pops a
        // lone lr; `mov pc, lr` is the ARMv4 return) or an indirect jump through
        // memory (a thunk, a pc-relative jump table). Either way there is no
        // in-function successor; the lowering picks return vs dispatch.
        Opcode::LDR | Opcode::MOV if regnum(&inst.operands[0]) == Some(15) => Flow::Return,
        _ => {
            let _ = len;
            Flow::Seq
        }
    }
}

/// Track per-register immediate constants across a straight run and return the
/// updated state. Two jobs, one pass:
///   - r7 tracking a noreturn `svc` (r7 = an exit syscall) can end decoding
///     before the data that follows it (formerly `track_r7`), and
///   - `movw`/`movt` materialization of a full 32-bit value, so an address-taken
///     code pointer can be recognized. When `discover_pointers` is set and a
///     completed value has bit 0 set (a Thumb function pointer, as opposed to a
///     data pointer, which is even) and lands inside the code image, it is
///     recorded in `code_pointers`.
///
/// Conservative: any register written by an instruction we do not model as a
/// constant is cleared. Register `n` is deemed written when it is the first
/// operand (matching the old r7 rule, so r7's noreturn behavior is unchanged).
fn track_regs(
    inst: &Instruction,
    addr: u32,
    mut regs: RegConsts,
    in_code: &impl Fn(u32) -> bool,
    discover_pointers: bool,
    code_pointers: &mut BTreeSet<u32>,
) -> RegConsts {
    let dst = inst.operands.first().and_then(regnum);
    match inst.opcode {
        // Constant arithmetic, which is how HAND-WRITTEN assembly forms a computed jump
        // target: there is no literal pool and no relocation to notice, just `add rd,
        // pc, #imm` to get a code address and then `add rd, rd, #stride` to step it,
        // with a `blx`/`bx rd` at the end. An assembly MD5 in this title's libc walks
        // its round blocks exactly so:
        //
        // ```text
        //   add r12, pc, #77       ; r12 = the first round helper
        //   ...  blx r12           ; call it
        //   add r12, r12, #16      ; step to the next block
        //   bx  r12                ; and enter that
        // ```
        //
        // Neither target is reachable by any direct branch, so without following the
        // arithmetic they are never discovered and the run dies on an indirect dispatch
        // to an address that is perfectly correct but has no entry. Tracking the step
        // makes discovery cascade: each block entered this way contains the `add` that
        // reveals the next.
        //
        // Bit 0 is the Thumb bit, as for a `movw`/`movt` pointer: the assembler folds
        // it into the constant, so `pc + 77` = `...349` addresses the code at `...348`.
        Opcode::ADD => {
            if let Some(rd) = dst {
                let pc =
                    if inst.thumb { addr.wrapping_add(4) & !3 } else { addr.wrapping_add(8) };
                // The addend's constant value: pc itself when the operand is r15,
                // otherwise whatever that register is tracked as holding.
                let tracked = |op: Option<&Operand>| -> Option<u32> {
                    match op {
                        Some(Operand::Reg(r)) if r.number() == 15 => Some(pc),
                        Some(Operand::Reg(r)) => regs[r.number() as usize],
                        _ => None,
                    }
                };
                let v = match (&inst.operands[1], &inst.operands[2]) {
                    // add rd, rn, #imm - including `rn == pc`.
                    (_, op2 @ (Operand::Imm32(_) | Operand::Imm(_) | Operand::Imm12(_))) => {
                        tracked(Some(&inst.operands[1]))
                            .zip(imm(op2))
                            .map(|(a, k)| a.wrapping_add(k))
                    }
                    // add rd, #imm (two-operand form: rd is also the source).
                    (op1 @ (Operand::Imm32(_) | Operand::Imm(_) | Operand::Imm12(_)), _) => {
                        regs[rd as usize].zip(imm(op1)).map(|(a, k)| a.wrapping_add(k))
                    }
                    _ => None,
                };
                if let Some(v) = v {
                    if discover_pointers && v & 1 == 1 && in_code(v & !1) {
                        code_pointers.insert(v & !1);
                    }
                }
                regs[rd as usize] = v;
            }
        }
        // movw / mov rd, #imm: rd becomes the immediate (or unknown if not one).
        // A register source (`mov rd, rn`) yields None, clearing rd.
        Opcode::MOV => {
            if let Some(rd) = dst {
                regs[rd as usize] = inst.operands.get(1).and_then(imm);
            }
        }
        // movt rd, #hi: set the top halfword. If rd held a low halfword, this
        // completes a 32-bit value - the moment an address-taken pointer appears.
        Opcode::MOVT => {
            if let (Some(rd), Some(hi)) = (dst, inst.operands.get(1).and_then(imm)) {
                let rd = rd as usize;
                regs[rd] = regs[rd].map(|lo| {
                    let v = (lo & 0x0000_FFFF) | (hi << 16);
                    if discover_pointers && v & 1 == 1 && in_code(v & !1) {
                        code_pointers.insert(v & !1);
                    }
                    v
                });
            }
        }
        // Opcodes whose FIRST operand is a SOURCE, not a destination. They write no
        // register, so the blanket clear below would throw away a constant that is
        // still live afterwards.
        //
        // `blx rN` is the one that matters: a computed jump CALLS THROUGH the very
        // register it is stepping, so clearing it at the call broke the chain between
        // `add r12, pc, #imm` and the `add r12, r12, #16` that walks to the next block.
        // (A call may of course clobber a caller-saved register for real; this tracker
        // does not model calls at all, and a wrong constant can only ever produce a
        // tentative code pointer, which is bounds-checked and dropped if it does not
        // decode.) The compares and stores are the same mistake, just less costly.
        Opcode::BX
        | Opcode::BLX
        | Opcode::CMP
        | Opcode::CMN
        | Opcode::TST
        | Opcode::TEQ
        | Opcode::STR
        | Opcode::STRB
        | Opcode::STRH
        | Opcode::PUSH => {}
        // Anything else that writes a register clears its tracked constant.
        _ => {
            if let Some(rd) = dst {
                regs[rd as usize] = None;
            }
        }
    }
    regs
}

/// ITSTATE advance (ARM ARM `ITAdvance`): shift the low 5 bits left, keeping the
/// top 3 condition bits. Returns the next ITSTATE (0 = out of an IT block).
fn it_advance(itstate: u8) -> u8 {
    if itstate & 0b111 == 0 {
        0
    } else {
        (itstate & 0b1110_0000) | ((itstate << 1) & 0b0001_1111)
    }
}

/// The condition an instruction runs under, given the current ITSTATE.
fn it_condition(itstate: u8) -> ConditionCode {
    cond_from_u8((itstate >> 4) & 0xF)
}

/// Map a 4-bit ARM condition field to yaxpeax's `ConditionCode` (same order).
fn cond_from_u8(c: u8) -> ConditionCode {
    use ConditionCode::*;
    [EQ, NE, HS, LO, MI, PL, VS, VC, HI, LS, GE, LT, GT, LE, AL, AL][c as usize & 0xF]
}

/// A statically-recovered `tbb`/`tbh` jump table: the index register, the resolved
/// target block addresses (`targets[i]` for index `i`), and the out-of-range
/// default block when the range-check branch is recognized.
struct SwitchInfo {
    index: u8,
    targets: Vec<u32>,
    default: Option<u32>,
    /// `targets[(reg + bias) >> shift]`; `(0, 0)` for a `tbb`/`tbh`, whose register IS the
    /// index. An ARM `add pc, pc, rN` (see [`arm_add_pc_switch`]) indexes by a byte
    /// offset over a window, so it needs both.
    bias: i32,
    shift: u32,
}

/// The window an ARM `add pc, pc, rN` may land in: `pc + 8 + rN` for `rN` in
/// `[-ADD_PC_BACK, ADD_PC_FWD)`. Every word in ARM code is an instruction boundary,
/// so each is a valid case body; the ones the register never names are simply
/// never taken. newlib's memcpy (every vitasdk homebrew) enters its unrolled copy
/// loop this way: `rsb r3, r3, #0x34 ; add pc, pc, r3`.
const ADD_PC_BACK: u32 = 64;
const ADD_PC_FWD: u32 = 192;

/// `add pc, pc, rN` in ARM mode, unconditional: the switch over its window.
fn arm_add_pc_switch(inst: &Instruction, addr: u32, thumb: bool, code_len: u32, base: u32) -> Option<SwitchInfo> {
    if thumb || inst.opcode != Opcode::ADD || inst.condition != ConditionCode::AL {
        return None;
    }
    let (Some(Operand::Reg(rd)), Some(Operand::Reg(rn)), Some(Operand::Reg(rm))) =
        (inst.operands.first(), inst.operands.get(1), inst.operands.get(2))
    else {
        return None;
    };
    if rd.number() != 15 || rn.number() != 15 || rm.number() == 15 {
        return None;
    }
    let pc = addr.wrapping_add(8);
    // The window is clamped to the code image on both sides: below `base` there is
    // no instruction to name (a function at the very start of an image), and past
    // its end there is none either.
    let start = pc.wrapping_sub(ADD_PC_BACK).max(base);
    let mut targets = Vec::new();
    let mut t = start;
    while t < pc.wrapping_add(ADD_PC_FWD) && t.wrapping_sub(base) < code_len {
        targets.push(t);
        t = t.wrapping_add(4);
    }
    if targets.is_empty() {
        return None;
    }
    Some(SwitchInfo { index: rm.number(), targets, default: None, bias: pc.wrapping_sub(start) as i32, shift: 2 })
}

/// libgcc's integer-division ladder (`__aeabi_uidiv`/`__aeabi_idiv`, `lib1funcs.S`
/// `ARM_DIV_BODY` on a core without hardware divide - every vitasdk homebrew) enters
/// its unrolled shift-subtract steps with a computed jump:
///
/// ```text
///   adr  r2, 1f                ; the first 16-byte step
///   add  r3, r2, r3, lsl #4    ; step index (0..31) scaled by the step size
///   mov  r2, #0                ; (unrelated instructions may sit in between)
///   mov  pc, r3
/// ```
///
/// Recognised at the `mov pc` from the decoded stream: walk back to the writer of the
/// jump register (an `add rd, rn, rm, lsl #s`), then to the writer of its base (`adr`
/// or `add rn, pc, #imm`), skipping instructions that write neither. The switch keys
/// on the JUMP REGISTER's final value - `targets[(rd - ladder) >> s]` - because the
/// index register is usually `rd` itself and is gone by the jump. Targets are the
/// ladder's entries for the 32 steps a 32-bit divide can need, clamped to the image.
/// Without this the jump has no static target and the divide misses in the dispatcher
/// (before the `mov pc` lowering existed it fell through, and `n / es` came back
/// undivided - newlib's qsort then recursed over garbage).
fn mov_pc_ladder_switch(
    inst: &Instruction,
    addr: u32,
    decoded: &BTreeMap<u32, (Instruction, u32, ConditionCode, bool)>,
    code_len: u32,
    base: u32,
) -> Option<SwitchInfo> {
    if inst.opcode != Opcode::MOV || inst.condition != ConditionCode::AL {
        return None;
    }
    let (Operand::Reg(pc), Operand::Reg(rd)) = (&inst.operands[0], &inst.operands[1]) else {
        return None;
    };
    if pc.number() != 15 || rd.number() == 15 {
        return None;
    }
    let rd = rd.number();
    // The unconditional instruction ending exactly at `at`.
    let ending_at = |at: u32| {
        decoded
            .range(..at)
            .next_back()
            .filter(|(a, (_, len, cond, in_it))| {
                a.wrapping_add(*len) == at && !*in_it && *cond == ConditionCode::AL
            })
            .map(|(a, (i, _, _, _))| (*a, i))
    };
    let writes = |i: &Instruction, r: u8| regnum(&i.operands[0]) == Some(r);
    // The most recent writer of `r`, walking back at most `max` instructions.
    let writer_of = |from: u32, r: u8, max: usize| -> Option<(u32, &Instruction)> {
        let mut cur = from;
        for _ in 0..max {
            let (a, i) = ending_at(cur)?;
            if writes(i, r) {
                return Some((a, i));
            }
            cur = a;
        }
        None
    };
    let (add_addr, add) = writer_of(addr, rd, 4)?;
    if add.opcode != Opcode::ADD {
        return None;
    }
    let (Operand::Reg(arn), Operand::RegShift(rs)) = (&add.operands[1], &add.operands[2]) else {
        return None;
    };
    let RegShiftStyle::RegImm(sh) = rs.into_shift() else { return None };
    if sh.stype() != ShiftStyle::LSL {
        return None;
    }
    let shift = sh.imm() as u32;
    let (adr_addr, adr) = writer_of(add_addr, arn.number(), 4)?;
    let pc_of = |a: u32| if adr.thumb { a.wrapping_add(4) & !3 } else { a.wrapping_add(8) };
    let ladder = match (adr.opcode, &adr.operands[1], &adr.operands[2]) {
        (Opcode::ADR, op, _) => pc_of(adr_addr).wrapping_add(imm(op)?),
        (Opcode::ADD, Operand::Reg(p), op) if p.number() == 15 => {
            pc_of(adr_addr).wrapping_add(imm(op)?)
        }
        _ => return None,
    };
    let mut targets = Vec::new();
    for k in 0..32u32 {
        let t = ladder.wrapping_add(k << shift);
        if t.wrapping_sub(base) >= code_len {
            break;
        }
        targets.push(t);
    }
    if targets.is_empty() {
        return None;
    }
    Some(SwitchInfo {
        index: rd,
        targets,
        default: None,
        bias: 0u32.wrapping_sub(ladder) as i32,
        shift,
    })
}

/// The index register of a pc-relative `tbb`/`tbh`. Returns `None` (leaving the
/// instruction unlowered) for the register-base form, whose table lives at a
/// runtime address we cannot read statically.
fn switch_index_reg(inst: &Instruction) -> Option<u8> {
    match &inst.operands[0] {
        // `tbb [pc, Rm]`
        Operand::RegDerefPreindexReg(base, index, _, _) if base.number() == 15 => {
            Some(index.number())
        }
        // `tbh [pc, Rm, lsl #1]`
        Operand::RegDerefPreindexRegShift(base, rs, _, _) if base.number() == 15 => {
            match rs.into_shift() {
                RegShiftStyle::RegImm(s) => Some(s.shiftee().number()),
                RegShiftStyle::RegReg(s) => Some(s.shiftee().number()),
            }
        }
        _ => None,
    }
}

/// The logical negation of an ARM condition code (the condition that holds on the
/// fall-through when the branch is not taken).
fn negate_cond(c: ConditionCode) -> ConditionCode {
    use ConditionCode::*;
    match c {
        EQ => NE, NE => EQ,
        HS => LO, LO => HS,
        HI => LS, LS => HI,
        GE => LT, LT => GE,
        GT => LE, LE => GT,
        MI => PL, PL => MI,
        VS => VC, VC => VS,
        AL => AL,
    }
}

/// The value of a data-processing source operand as a constant: an immediate, or a
/// register whose materialized constant is known in `regs`.
fn const_operand(op: &Operand, regs: &RegConsts) -> Option<u32> {
    if let Some(v) = imm(op) {
        return Some(v);
    }
    regnum(op).and_then(|r| regs[r as usize])
}

/// Whether control can reach `target` from `from` by walking the decoded stream:
/// fall-through, unconditional branches, and BOTH sides of a conditional one. Calls are
/// not followed (they come back), and a table branch stops the walk (where it goes is the
/// very thing being recovered).
///
/// This is what tells a range check's in-range side from its out-of-range side. Position
/// cannot: a guard's taken target routinely lands between the guard and the table branch
/// while leading somewhere else entirely - a chain of `cmp`/`b` guards selecting among
/// SEVERAL tables is exactly that shape, and reading the wrong side either loses the
/// bound or, worse, sizes the table wrong.
fn reaches(
    decoded: &BTreeMap<u32, (Instruction, u32, ConditionCode, bool)>,
    from: u32,
    target: u32,
) -> bool {
    // Bounded so recovery stays linear-ish on a pathological CFG. A range check sits
    // within a few blocks of the table it guards.
    const BUDGET: usize = 256;
    let mut seen = BTreeSet::new();
    let mut stack = vec![from];
    let mut steps = 0;
    while let Some(mut pc) = stack.pop() {
        loop {
            if pc == target {
                return true;
            }
            steps += 1;
            if steps > BUDGET || !seen.insert(pc) {
                break;
            }
            let Some((ins, len, cond, _)) = decoded.get(&pc) else { break };
            if matches!(ins.opcode, Opcode::TBB | Opcode::TBH) {
                break;
            }
            if ins.opcode == Opcode::B {
                let Some(t) = branch_target(ins, pc, true) else { break };
                if ins.condition == ConditionCode::AL && *cond == ConditionCode::AL {
                    pc = t;
                    continue;
                }
                stack.push(t);
            }
            pc = pc.wrapping_add(*len);
        }
    }
    false
}

/// One step of the `idx = switch + k` chain: given an instruction that DEFINES
/// `reg`, report which register it derived that value from and the constant it
/// added.
///
/// Returns `(reg, 0)` - the register unchanged, no step - when the definition is not
/// a constant adjustment or a register copy. That is the chain's terminator: the
/// value being indexed by IS the switch variable at that point, and walking further
/// back would attribute unrelated arithmetic to the rebase.
fn adjustment_step(ins: &Instruction, reg: u8, before: &RegConsts) -> (u8, u32) {
    match ins.opcode {
        // `idx = a +/- b`, where the label base folds into `k`. Both the two-operand
        // (`add rd,#imm` == `rd = rd +/- imm`) and three-operand (`add rd,rn,rm`)
        // forms appear; a source register carrying a tracked constant is the base,
        // the other is the switch variable.
        Opcode::ADD | Opcode::SUB => {
            let neg = ins.opcode == Opcode::SUB;
            let signed = |kk: u32| if neg { kk.wrapping_neg() } else { kk };
            let o1 = &ins.operands[1];
            let o2 = &ins.operands[2];
            if matches!(o2, Operand::Nothing) {
                // Two-operand: `rd = rd (op) o1`. The switch value is the destination
                // (also the implicit first source), so the chain continues on `reg`.
                if let Some(kk) = imm(o1) {
                    (reg, signed(kk))
                } else if let Some(kk) = regnum(o1).and_then(|r| before[r as usize]) {
                    (reg, signed(kk))
                } else {
                    (reg, 0)
                }
            } else if let (Some(rs), Some(kk)) = (regnum(o1), const_operand(o2, before)) {
                (rs, signed(kk))
            } else if let (Some(kk), Some(rs)) = (const_operand(o1, before), regnum(o2)) {
                // `rd = imm (op) rs`: only `imm + rs` keeps `rs` a straight index;
                // `imm - rs` negates the index, which we do not model.
                if neg { (reg, 0) } else { (rs, kk) }
            } else {
                (reg, 0)
            }
        }
        Opcode::MOV => regnum(&ins.operands[1]).map_or((reg, 0), |rs| (rs, 0)),
        _ => (reg, 0),
    }
}

/// Recover the entry count and out-of-range default of a `tbb`/`tbh` switch from
/// the compiler's range check.
///
/// GCC lowers a `switch` to a range check guarding a table branch, but in more
/// shapes than the textbook `cmp idx,#k ; bhi default ; tbh`:
///  - the value compared can be the raw switch variable, while the table is
///    indexed by a *rebased* copy `idx = switch + k` (an `add`/`sub` folding the
///    case-label base into the index, so table entry `i` means `switch = i - k`);
///  - the bound can be a register materialized by `movw`/`movt`, not an immediate;
///  - the guard can branch *to* the table setup on the in-range condition
///    (`... ; ble .Lin ; b .Ldefault ; .Lin: ... ; tbh`), the reverse polarity of
///    the textbook form, and it need not sit immediately before the table branch.
///
/// The recovery walks a bounded window before the table branch tracking `movw`/
/// `movt` constants (so register bounds resolve), finds how the index register was
/// produced from the switch variable (`idx = switch + k`), then picks the range
/// check closest to the table branch whose in-range side is an upper bound on the
/// switch value and whose branch actually steers control toward the table. The
/// entry count follows from `bound + k` (+1 when the bound is inclusive). Returns
/// `(count, default)`; `count` is `None` when no such guard is recognized (the
/// caller then falls back to reading the table's own extent).
fn recover_switch_bound(
    decoded: &BTreeMap<u32, (Instruction, u32, ConditionCode, bool)>,
    tb_addr: u32,
    index_reg: u8,
) -> (Option<u32>, Option<u32>) {
    use ConditionCode::*;
    // A window before the table branch, with the tracked constant state *before*
    // each instruction (so a `cmp`/`add` against a `movw`/`movt` register resolves).
    const WINDOW: u32 = 0x100;
    let mut regs: RegConsts = [None; 16];
    let mut throwaway = BTreeSet::new();
    let mut snaps: Vec<(u32, &Instruction, u32, RegConsts)> = Vec::new();
    for (&a, (ins, len, _, _)) in decoded.range(tb_addr.wrapping_sub(WINDOW)..tb_addr) {
        snaps.push((a, ins, *len, regs));
        regs = track_regs(ins, a, regs, &|_| false, false, &mut throwaway);
    }

    // How the index register was produced from the switch variable: `idx = switch + k`.
    //
    // This is a CHAIN, not a single step, and following only one step is an
    // off-by-`k` waiting to happen. A compiler normalising a switch whose labels do
    // not start at zero emits the rebase in as many instructions as the constants
    // need - a real title has `subs r0,#1 ; sub.w r0,r0,#0x1000 ; tbh`, two
    // adjustments in a row - and taking only the nearest one recovers `k = -0x1000`
    // instead of `-0x1001`. The count then comes out exactly one too high, the extra
    // "entry" is whatever instruction follows the table, and the switch is rejected
    // as unresolvable (its bogus last target lands far outside the function). The
    // whole function then fails to lift, which is how this surfaced: a trapping stub
    // reached during boot.
    //
    // So walk back through consecutive definitions, accumulating `k`, until a
    // definition is reached that is not a constant adjustment or a register copy.
    // The walk is bounded by the snapshot window, and each step must define the
    // register the previous step read, so it cannot loop.
    let (switch_reg, k) = {
        let mut reg = index_reg;
        let mut k: u32 = 0;
        let mut from = snaps.len();
        loop {
            let Some(i) = snaps[..from]
                .iter()
                .rposition(|(_, ins, _, _)| ins.operands.first().and_then(regnum) == Some(reg))
            else {
                break;
            };
            let (_, ins, _, before) = &snaps[i];
            let (next_reg, step) = adjustment_step(ins, reg, before);
            k = k.wrapping_add(step);
            // A definition that is not an adjustment reports itself as the same
            // register with no step, which is where the chain ends.
            if next_reg == reg && step == 0 {
                break;
            }
            reg = next_reg;
            from = i;
        }
        (reg, k)
    };

    // Pick the range check closest to the table branch whose in-range side is an
    // upper bound on the switch variable. A `cmp switch, bound` paired with the next
    // conditional branch; the branch's taken target tells us which side is in-range.
    let mut best: Option<(u32, u32, Option<u32>)> = None; // (cmp_addr, count, default)
    for (i, (cmp_addr, cmp, _, before)) in snaps.iter().enumerate() {
        if cmp.opcode != Opcode::CMP || regnum(&cmp.operands[0]) != Some(switch_reg) {
            continue;
        }
        let Some(bound) = const_operand(&cmp.operands[1], before) else { continue };
        // The conditional branch this compare feeds: the next branch in the window.
        let Some((br_addr, br, br_len, _)) =
            snaps[i + 1..].iter().find(|(_, ins, _, _)| ins.opcode == Opcode::B).copied()
        else {
            continue;
        };
        if br.condition == AL {
            continue;
        }
        let Some(gt) = branch_target(br, br_addr, true) else { continue };
        // In-range is the side that actually steers toward the table branch, decided by
        // walking the CFG from each side. When exactly one side reaches the table that
        // settles it; when the walk is inconclusive (a diamond that rejoins, or a budget
        // cut-off) fall back to position - the taken branch steers toward the table when
        // its target lands in the setup between the guard and the table.
        let fall = br_addr.wrapping_add(br_len);
        let in_range_taken = match (reaches(decoded, gt, tb_addr), reaches(decoded, fall, tb_addr)) {
            (true, false) => true,
            (false, true) => false,
            _ => gt > br_addr && gt <= tb_addr,
        };
        let effective = if in_range_taken { br.condition } else { negate_cond(br.condition) };
        // Only an upper bound on the switch value fixes the entry count.
        let max_index = match effective {
            LE | LS => bound.wrapping_add(k),            // switch <= bound  (inclusive)
            LT | LO => bound.wrapping_add(k).wrapping_sub(1), // switch <  bound  (exclusive)
            _ => continue,                                // a lower bound: not the count
        };
        let count = max_index.wrapping_add(1);
        if count == 0 || count > 1024 {
            continue; // implausible: a wrong pairing (wrapped) - reject
        }
        let default = if in_range_taken { br_addr.wrapping_add(br_len) } else { gt };
        // Closest compare to the table branch wins (the innermost cluster's guard).
        if best.is_none_or(|(a, _, _)| *cmp_addr > a) {
            if switch_why(tb_addr) {
                eprintln!(
                    "  guard @{cmp_addr:#x}: cmp r{switch_reg}, {bound:#x}  br @{br_addr:#x} \
                     cond={:?} target={gt:#x} fall={fall:#x} in_range_taken={in_range_taken} \
                     effective={effective:?} k={k:#x} -> count={count} default={default:#x}",
                    br.condition
                );
            }
            best = Some((*cmp_addr, count, Some(default)));
        }
    }
    match best {
        Some((_, count, default)) => (Some(count), default),
        None => (None, None),
    }
}

/// Read a `tbb`/`tbh` jump table and resolve its targets. The table sits inline
/// right after the instruction (base `pc = tb_addr + 4`); each entry is a byte
/// (`tbb`) or halfword (`tbh`) offset, and the target is `pc + 2*entry`.
///
/// The entry count comes from the compiler's range check when recognizable
/// ([`recover_switch_bound`]); otherwise it is inferred from the table's own
/// extent - the table runs up to the nearest case body, so the smallest offset
/// bounds the entry count (`count <= min_entry` for `tbh`, `<= 2*min_entry` for
/// `tbb`). Returns `None` (leaving the branch unlowered, a clean transpile
/// failure rather than wrong code) if the table cannot be resolved in bounds.
fn resolve_switch(
    code: &[u8],
    base: u32,
    tb_addr: u32,
    inst: &Instruction,
    decoded: &BTreeMap<u32, (Instruction, u32, ConditionCode, bool)>,
    leaders: &BTreeSet<u32>,
) -> Option<SwitchInfo> {
    // Diagnostic (`VITASLOP_SWITCH_WHY=<hex tb address>`, or `all`): explain why a
    // table branch did or did not resolve. An unresolved `tbb`/`tbh` fails its whole
    // function, and the report only says "Unsupported TBH" - which names the
    // instruction but not which of the three independent bounds recoveries gave up,
    // so the next step is a guess. This prints the actual count each one produced.
    let why = switch_why(tb_addr);
    let index = match switch_index_reg(inst) {
        Some(i) => i,
        None => {
            if why {
                eprintln!(
                    "switch @{tb_addr:#x}: UNRESOLVED - operand form is not a pc-relative \
                     table ({:?}), so the table address is only known at runtime",
                    inst.operands[0]
                );
            }
            return None;
        }
    };
    let is_tbh = inst.opcode == Opcode::TBH;
    let pc = tb_addr.wrapping_add(4);
    let esize = if is_tbh { 2u32 } else { 1 };
    let read = |i: u32| -> Option<u32> {
        let off = pc.wrapping_add(i * esize).wrapping_sub(base) as usize;
        if is_tbh {
            let b = code.get(off..off + 2)?;
            Some(u16::from_le_bytes([b[0], b[1]]) as u32)
        } else {
            code.get(off).map(|&b| b as u32)
        }
    };

    let (cmp_count, default) = recover_switch_bound(decoded, tb_addr, index);
    // The table ABUTS the next code leader: a jump table is inline data, so the first
    // known instruction address after it ends it exactly. This is the bound for a switch
    // whose range check is NOT local - a compiler lowering a large sparse switch as a
    // binary search over sub-tables establishes the bound in an enclosing comparison many
    // instructions earlier, so `recover_switch_bound` finds nothing, and the table-extent
    // heuristic below fails too because it assumes the nearest case body follows the table
    // (here the bodies are kilobytes away, so the smallest entry bounds nothing and the
    // scan runs to its runaway guard).
    //
    // The abutment is self-checking, which is what makes it safe to trust: the leader must
    // land on an exact entry boundary from the table base. A leader inside the table (which
    // would mean we mistook table data for code) almost never satisfies that, and a count
    // that is wrong anyway still has to pass the per-target validation below.
    let abut_count = leaders
        .range(pc.wrapping_add(1)..)
        .next()
        .map(|&l| l.wrapping_sub(pc))
        .filter(|span| *span % esize == 0)
        .map(|span| span / esize)
        .filter(|c| *c > 0 && *c <= 1024)
        // CONSISTENCY, and this is load-bearing: a leader that merely happens to follow the
        // table does not prove the table reaches it. The table's own entries have to agree.
        // Every entry is a forward offset to a case body, so the table cannot contain more
        // entries than the smallest offset allows - entry `min` would otherwise sit at or
        // past the first case body. Reading code or unrelated data as entries almost always
        // violates that, so this rejects an over-long count instead of seeding far-flung
        // garbage leaders into the function. Without it, resolving tables that had
        // previously been left alone broke a title at frame 0.
        .filter(|&c| {
            let mut min_entry = u32::MAX;
            for i in 0..c {
                match read(i) {
                    Some(v) => min_entry = min_entry.min(v),
                    None => return false,
                }
            }
            let limit = if is_tbh { min_entry } else { 2 * min_entry };
            c <= limit
        });
    if why {
        eprintln!(
            "switch @{tb_addr:#x}: {} index=r{index} table={:#x} cmp_count={cmp_count:?} \
             abut_count={abut_count:?} default={default:?} next_leader={:?}",
            if is_tbh { "tbh" } else { "tbb" },
            pc,
            leaders.range(pc.wrapping_add(1)..).next().map(|l| format!("{l:#x}")),
        );
    }
    let count = match cmp_count.or(abut_count) {
        Some(c) => c,
        None => {
            // Infer from the table extent: grow the entry list until the next index
            // would reach the nearest case body.
            let mut entries: Vec<u32> = Vec::new();
            loop {
                let i = entries.len() as u32;
                if let Some(&m) = entries.iter().min() {
                    let limit = if is_tbh { m } else { 2 * m };
                    if i >= limit {
                        break;
                    }
                }
                if i >= 1024 {
                    return None; // runaway: not a table we understand
                }
                match read(i) {
                    Some(v) => entries.push(v),
                    None => break,
                }
            }
            entries.len() as u32
        }
    };
    if count == 0 || count > 1024 {
        if why {
            eprintln!("switch @{tb_addr:#x}: UNRESOLVED - entry count {count} is out of range");
        }
        return None;
    }

    // Case bodies of one switch are local to their function; a real target lands
    // within a function's span of the table. A target further than that means the
    // count is wrong (we read code/data past the table's end as entries) - reject
    // the whole switch rather than seed far-flung leaders that would drag unrelated
    // code into this function (and trip the runaway span guard). Defense in depth:
    // the range-check recovery already bounds the count, this guards a misfire.
    const MAX_REACH: u32 = 0x1_0000; // 64 KiB, matches the discovery span guard
    let mut targets = Vec::with_capacity(count as usize);
    for i in 0..count {
        let Some(entry) = read(i) else {
            if why {
                eprintln!("switch @{tb_addr:#x}: UNRESOLVED - entry {i} of {count} is off the image");
            }
            return None;
        };
        let target = pc.wrapping_add(2 * entry);
        if target.wrapping_sub(base) as usize >= code.len() {
            if why {
                eprintln!(
                    "switch @{tb_addr:#x}: UNRESOLVED - entry {i} ({entry:#x}) targets \
                     {target:#x}, outside the image: the entry count is wrong"
                );
            }
            return None; // target outside the image: bound is wrong, bail cleanly
        }
        if target.wrapping_sub(tb_addr).min(tb_addr.wrapping_sub(target)) > MAX_REACH {
            if why {
                eprintln!(
                    "switch @{tb_addr:#x}: UNRESOLVED - entry {i} ({entry:#x}) targets \
                     {target:#x}, more than {MAX_REACH:#x} from the table: the count is wrong"
                );
            }
            return None; // implausibly far from the table: bound is wrong, bail cleanly
        }
        targets.push(target);
    }
    if why {
        eprintln!("switch @{tb_addr:#x}: RESOLVED {count} targets, default={default:?}");
    }
    Some(SwitchInfo { index, targets, default, bias: 0, shift: 0 })
}

/// Whether the table-branch diagnostic is on for this address
/// (`VITASLOP_SWITCH_WHY=<hex address>` or `all`). Parsed once.
fn switch_why(tb_addr: u32) -> bool {
    use std::sync::OnceLock;
    static SPEC: OnceLock<Option<Option<u32>>> = OnceLock::new();
    match SPEC.get_or_init(|| {
        std::env::var("VITASLOP_SWITCH_WHY").ok().map(|s| {
            let s = s.trim();
            if s.eq_ignore_ascii_case("all") {
                None
            } else {
                u32::from_str_radix(s.trim_start_matches("0x"), 16).ok()
            }
        })
    }) {
        None => false,
        Some(None) => true,
        Some(Some(a)) => *a == tb_addr,
    }
}

/// Discover and lower the function at `entry`.
pub fn discover(
    code: &[u8],
    base: u32,
    entry: u32,
    thumb: bool,
    imports: &Imports,
    noreturn_svc: &[u32],
    discover_pointers: bool,
    isolate: bool,
) -> Result<Discovered, Error> {
    let decoder = InstDecoder::default().with_thumb_mode(thumb);

    // Pass 1: reachability. Decode every reachable instruction, recording its
    // decoded form + length + applied (IT) condition, and collect leaders,
    // callees, and the terminating instructions.
    // Per address: the decoded instruction, its length, the applied condition,
    // and whether it sits inside an IT block (where flag-setting is suppressed).
    let mut decoded: BTreeMap<u32, (Instruction, u32, ConditionCode, bool)> = BTreeMap::new();
    let mut leaders: BTreeSet<u32> = BTreeSet::new();
    // Direct callees as (address, Thumb-mode-to-decode-in). The mode is per callee, not
    // per caller: `blx <label>` interworks.
    let mut callees: BTreeSet<(u32, bool)> = BTreeSet::new();
    // Address-taken code pointers materialized in this function (thread entries,
    // callbacks). Collected via `movw`/`movt` tracking; processed as tentative
    // entries by the caller.
    let mut code_pointers: BTreeSet<u32> = BTreeSet::new();
    // Even code pointers reached via a register-indirect `blx`/`bx` (ARM-mode
    // helpers a Thumb function calls). Kept separate so the caller seeds them as
    // tentative ARM entries rather than Thumb.
    let mut arm_code_pointers: BTreeSet<u32> = BTreeSet::new();
    // Register-indirect `blx`/`bx` sites whose tracked target constant is a known
    // host-import stub: the compiler routed a host import (e.g. `memset`) through a
    // function pointer. Keyed by the call-site address so pass 2 lowers it as the
    // import instead of dispatching to the stub's `mvn r0,#0; bx lr` placeholder
    // (which would silently return -1 and do nothing).
    let mut indirect_imports: BTreeMap<u32, u32> = BTreeMap::new();
    // Register-indirect `blx`/`bx` sites whose tracked target is an inter-module
    // import stub that the linker resolved to a guest export (a Redirect). Keyed by
    // call-site address -> resolved guest target, so pass 2 lowers a direct call to
    // the real routine instead of dispatching to the unresolved-stub placeholder.
    let mut indirect_redirects: BTreeMap<u32, u32> = BTreeMap::new();
    // Statically-recovered `tbb`/`tbh` jump tables, keyed by the branch address.
    // Filled in pass 1 (where the whole decoded stream is available to read the
    // range check) and consumed in pass 2 to build the computed-jump terminator.
    let mut switches: BTreeMap<u32, SwitchInfo> = BTreeMap::new();
    // Leaders whose first instruction failed to decode (a speculative target that ran
    // into data / an unlifted op). Pass 2 turns each into a single trapping block, so
    // one bad target does not stub the whole function. See the decode-failure arm.
    let mut trap_leaders: BTreeSet<u32> = BTreeSet::new();
    // Worklist carries IT state and the tracked register constants along fall-
    // through (a fresh, all-unknown set at every branch target, which may have
    // multiple predecessors).
    let init: RegConsts = [None; 16];
    let mut work: Vec<(u32, u8, RegConsts)> = vec![(entry, 0, init)];
    // The low halves of CONDITIONAL `movw`s, by register: (condition, value). An
    // `itete` block interleaves two movw/movt pairs into one register
    // (`movweq r2, #lo1 ; movwne r2, #lo2 ; movteq r2, #hi ; movtne r2, #hi` - FreeType
    // picking a rasteriser callback), and the linear tracker only ever completes the
    // second pair. Pairing each `movt` with the `movw` of ITS condition seeds both.
    // Reset at every leader: an IT block never spans one.
    let mut cond_lo: Vec<(u8, ConditionCode, u32)> = Vec::new();
    leaders.insert(entry);
    // Table branches whose extent could not be bounded on first sight; retried after the
    // walk settles, when `leaders` is complete (see the retry loop below).
    let mut pending_switches: BTreeSet<u32> = BTreeSet::new();

    // A guest address is decodable only if it lies within the code image.
    let in_bounds = |addr: u32| {
        let off = addr.wrapping_sub(base) as usize;
        off < code.len()
    };

    // The walk runs to a fixpoint with the table-branch retry below: resolving a table
    // reveals new case bodies to decode, and decoding them can supply the leader that
    // bounds the NEXT table.
    loop {
    while let Some((addr, itstate, regs)) = work.pop() {
        if !in_bounds(addr) {
            continue;
        }
        if decoded.contains_key(&addr) {
            continue;
        }
        // Runaway guard: no real function is this large. Hitting the cap means we
        // are decoding data as code - almost always a mis-identified code pointer
        // (a `movw`/`movt` constant that happens to point into the image) whose
        // straight-line "instructions" never reach a terminator. Reject the whole
        // function: as a tentative code pointer it is silently dropped; as a hard
        // callee it surfaces as a failure rather than a multi-megabyte wasm body.
        if decoded.len() >= MAX_FUNC_INSNS {
            return Err(Error::Decode { addr });
        }
        let (inst, len) = match decode_at(&decoder, code, base, addr, thumb) {
            Ok(v) => v,
            // A decode failure at the entry is a genuine whole-function failure (the
            // caller stubs it). Anywhere else it is a block reached only speculatively
            // - a heuristically-recovered branch/switch target that ran into data or
            // an unlifted instruction. Isolate it: record a trap leader (pass 2 emits
            // a trapping block) and stop following this path, so the rest of the
            // function still lifts instead of the whole thing becoming a stub.
            Err(e) => {
                // Strict callers (the diagnostic report, the abort-on-error build)
                // surface every gap. The lenient runtime build isolates a non-entry
                // failure to a trap block so one speculatively-reached bad target does
                // not stub the whole function.
                if !isolate || addr == entry {
                    return Err(e);
                }
                trap_leaders.insert(addr);
                leaders.insert(addr);
                continue;
            }
        };

        // Applied condition: an explicit branch condition wins; otherwise the IT
        // condition if we are in a block; otherwise unconditional.
        let applied = if inst.condition != ConditionCode::AL {
            inst.condition
        } else if itstate != 0 && inst.opcode != Opcode::IT {
            it_condition(itstate)
        } else {
            ConditionCode::AL
        };
        let in_it = itstate != 0 && inst.opcode != Opcode::IT;
        decoded.insert(addr, (inst.clone(), len, applied, in_it));

        // Next IT state for the fall-through successor.
        let next_it = if inst.opcode == Opcode::IT {
            it_state_from(&inst)
        } else if itstate != 0 {
            it_advance(itstate)
        } else {
            0
        };
        let next_regs =
            track_regs(&inst, addr, regs, &in_bounds, discover_pointers, &mut code_pointers);
        let next = addr.wrapping_add(len);
        if leaders.contains(&addr) {
            cond_lo.clear();
        }
        if applied != ConditionCode::AL {
            if let (Some(rd), Some(v)) = (regnum(&inst.operands[0]), inst.operands.get(1).and_then(imm)) {
                match inst.opcode {
                    Opcode::MOV => cond_lo.push((rd, applied, v)),
                    Opcode::MOVT => {
                        if let Some(&(_, _, lo)) =
                            cond_lo.iter().rev().find(|(r, c, _)| *r == rd && *c == applied)
                        {
                            let full = (lo & 0xFFFF) | (v << 16);
                            if discover_pointers && full & 1 == 1 && in_bounds(full & !1) {
                                code_pointers.insert(full & !1);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // A register-indirect `blx`/`bx` whose target register holds a tracked
        // `movw`/`movt` constant is a definite code pointer: its low bit selects
        // the mode the call switches to. An odd target is Thumb; an even target is
        // an ARM-mode helper (which the odd-only `movw`/`movt` scan would miss).
        // Seeds discovery so the runtime dispatcher can resolve the target instead
        // of trapping on it. `bx lr`/returns carry no tracked constant, so skip.
        if discover_pointers && matches!(inst.opcode, Opcode::BLX | Opcode::BX) {
            if let Some(v) = inst.operands.first().and_then(regnum).and_then(|rn| regs[rn as usize]) {
                // A register-indirect call whose tracked target is a host-import
                // stub is a host-import call reached through a function pointer
                // (a compiler routing e.g. `memset` through one thunk). Record the
                // site so pass 2 lowers it as the import, and do NOT seed the stub
                // placeholder as a lifted function - dispatching to that placeholder
                // silently returns -1 and skips the call's effect (e.g. a hash table
                // never gets its 0xffffffff sentinels, so a probe loops forever).
                let s = v & !1;
                let (resolved, resolved_thumb) = imports.resolve_call(s, v & 1 == 1);
                if let Some(idx) = imports.get(s) {
                    // Host import reached through a function pointer.
                    indirect_imports.insert(addr, idx);
                } else if resolved != s {
                    // Inter-module import stub resolved to a guest export: call the
                    // real routine directly and seed it for discovery, instead of
                    // dispatching to the stub's placeholder.
                    indirect_redirects.insert(addr, resolved);
                    // The mode is the EXPORT's, not the tracked pointer's: the pointer
                    // names the ARM import stub, whose Thumb bit says nothing about the
                    // function the linker resolved it to.
                    callees.insert((resolved, resolved_thumb));
                } else if in_bounds(s) {
                    if v & 1 == 1 {
                        code_pointers.insert(s);
                    } else {
                        arm_code_pointers.insert(v);
                    }
                }
            }
        }

        // Table-branch (`tbb`/`tbh`) switch dispatch: resolve the inline jump table
        // now, while the whole decoded stream (the range check that bounds it) is in
        // hand, and seed every case body as a leader. The bytes right after the
        // instruction are the table itself, never code, so this path never falls
        // through - resolved or not. An unresolved table is reported per-function in
        // pass 2 rather than marching the decoder into table data.
        if let Some(info) = arm_add_pc_switch(&inst, addr, thumb, code.len() as u32, base)
            .or_else(|| mov_pc_ladder_switch(&inst, addr, &decoded, code.len() as u32, base))
        {
            for &t in &info.targets {
                leaders.insert(t);
                work.push((t, 0, init));
            }
            switches.insert(addr, info);
            continue;
        }
        if matches!(inst.opcode, Opcode::TBB | Opcode::TBH) {
            match resolve_switch(code, base, addr, &inst, &decoded, &leaders) {
                Some(info) => {
                    for &t in &info.targets {
                        leaders.insert(t);
                        work.push((t, 0, init));
                    }
                    if let Some(d) = info.default {
                        leaders.insert(d);
                        work.push((d, 0, init));
                    }
                    switches.insert(addr, info);
                }
                // Not resolvable YET. The abutment bound needs the leader that follows
                // the table, which a sibling branch may not have contributed at this
                // point in the walk - so remember it and retry once the walk settles,
                // rather than giving up on an order-dependent snapshot.
                None => {
                    pending_switches.insert(addr);
                }
            }
            continue;
        }

        // A terminator under a condition (`bxeq lr`, an `it`-guarded `bx lr`/`pop {pc}`,
        // `beq` inside an IT block) has a fall-through successor as well: the
        // instruction after it is a leader, entered when the condition fails.
        let mut fl = flow(&inst, addr, len, thumb, entry, imports, regs[7], noreturn_svc);
        if applied != ConditionCode::AL {
            match fl {
                Flow::Return | Flow::Halt | Flow::TailCall(..) => {
                    leaders.insert(next);
                    work.push((next, next_it, next_regs));
                }
                Flow::Jump(t) => fl = Flow::Fork(t),
                _ => {}
            }
        }
        match fl {
            Flow::Seq => work.push((next, next_it, next_regs)),
            Flow::Call { guest } => {
                if let Some((t, t_thumb)) = guest {
                    callees.insert((t, t_thumb));
                }
                work.push((next, next_it, next_regs));
            }
            // A tail call records the callee (a separate function to discover) and
            // terminates this path: no fall-through, and the target is not a leader
            // in this function's body.
            Flow::TailCall(t, t_thumb) => {
                callees.insert((t, t_thumb));
            }
            Flow::Jump(t) => {
                leaders.insert(t);
                work.push((t, 0, init));
            }
            Flow::Fork(t) => {
                leaders.insert(t);
                leaders.insert(next);
                work.push((t, 0, init));
                work.push((next, next_it, next_regs));
            }
            Flow::Return | Flow::Halt => {}
        }
    }

        // Retry the table branches that could not be bounded on first sight.
        //
        // The abutment bound reads the leader that FOLLOWS the table, and that leader is
        // often contributed by a sibling comparison the walk had not reached yet: a large
        // sparse switch compiles to a binary search whose sub-tables are interleaved with
        // the compare chain that targets the code after each one. Retrying once the walk
        // settles removes that order dependence. Each round either resolves a pending table
        // (and loops to decode its case bodies) or stops, so this terminates.
        if pending_switches.is_empty() {
            break;
        }
        let mut progressed = false;
        for addr in core::mem::take(&mut pending_switches) {
            let Some((inst, _, _, _)) = decoded.get(&addr) else { continue };
            let inst = inst.clone();
            match resolve_switch(code, base, addr, &inst, &decoded, &leaders) {
                Some(info) => {
                    for &t in &info.targets {
                        leaders.insert(t);
                        work.push((t, 0, init));
                    }
                    if let Some(d) = info.default {
                        leaders.insert(d);
                        work.push((d, 0, init));
                    }
                    switches.insert(addr, info);
                    progressed = true;
                }
                None => {
                    pending_switches.insert(addr);
                }
            }
        }
        if !progressed {
            break; // Nothing further is resolvable; pass 2 reports these exactly as before.
        }
    }

    // Pass 1b: constant propagation ACROSS branches, whose only output is more tentative
    // code pointers.
    //
    // Pass 1 deliberately restarts constant tracking at every branch target, because a
    // target can have several predecessors and it uses those constants for decisions
    // that must not be wrong (resolving an indirect call to an import, spotting a
    // noreturn `svc`). That is the right rule there and it stays.
    //
    // But it also breaks a chain this title needs: a hand-written assembly MD5 steps a
    // code pointer with `add r12, r12, #16` and enters it with `bx r12`, and the loop
    // that does the stepping is re-entered BY A BRANCH from each block it reaches. So
    // every block after the first arrives with r12 forgotten, and the walk finds one
    // more round block per pass instead of all of them.
    //
    // This pass propagates the same tracking to a fixpoint WITH A PROPER MERGE at joins
    // (a register keeps a value only where every path into a block agrees on it), so it
    // cannot invent a constant that some path contradicts. It feeds nothing but
    // `code_pointers`, which are TENTATIVE - bounds-checked, decoded speculatively, and
    // dropped if they do not lift - so even a wrong one costs nothing but the attempt.
    if discover_pointers {
        // Meet: keep only the values both sides agree on.
        let meet = |a: &RegConsts, b: &RegConsts| -> RegConsts {
            let mut out = [None; 16];
            for i in 0..16 {
                if a[i] == b[i] {
                    out[i] = a[i];
                }
            }
            out
        };
        let mut at: BTreeMap<u32, RegConsts> = BTreeMap::new();
        at.insert(entry, init);
        let work: Vec<u32> = vec![entry];
        // Instructions decoded by this pass alone, for addresses pass 1 did not put in
        // `decoded`. A computed-jump chain runs THROUGH a shared loop that pass 1 leaves
        // by an outside branch, which it lowers as a tail call - correct for execution,
        // but it ends the walk, and with it the chain. Following it here costs nothing
        // at run time because this pass produces only tentative entries.
        let mut extra: BTreeMap<u32, (Instruction, u32)> = BTreeMap::new();
        // How far outside the function this pass will follow such a branch. A computed
        // jump chain is one hand-written routine's internal structure, so it is local;
        // the window keeps a stray branch from walking the pass across the image.
        const CHAIN_WINDOW: u32 = 0x1000;
        let near = |a: u32| a.wrapping_sub(entry) <= CHAIN_WINDOW
            || entry.wrapping_sub(a) <= CHAIN_WINDOW;
        let mut work: Vec<u32> = work;
        // Each register can only lose its value, so a block's state can change at most
        // 16 times and the fixpoint terminates; the cap is a belt-and-braces guard.
        let mut steps = 0usize;
        while let Some(addr) = work.pop() {
            steps += 1;
            if steps > MAX_FUNC_INSNS * 4 {
                break;
            }
            if !decoded.contains_key(&addr) && !extra.contains_key(&addr) {
                match decode_at(&decoder, code, base, addr, thumb) {
                    Ok(v) => {
                        extra.insert(addr, v);
                    }
                    Err(_) => continue,
                }
            }
            let (inst, len) = match decoded.get(&addr) {
                Some((i, l, _, _)) => (i.clone(), *l),
                None => extra[&addr].clone(),
            };
            let (inst, len) = (&inst, &len);
            let regs = at[&addr];
            let next_regs =
                track_regs(inst, addr, regs, &in_bounds, true, &mut code_pointers);
            // A register-indirect call/branch through a tracked constant is an entry,
            // exactly as in pass 1 - but here the constant may have come from across a
            // branch.
            if matches!(inst.opcode, Opcode::BLX | Opcode::BX) {
                if let Some(v) =
                    inst.operands.first().and_then(regnum).and_then(|rn| regs[rn as usize])
                {
                    let s = v & !1;
                    if imports.get(s).is_none() && in_bounds(s) {
                        if v & 1 == 1 {
                            code_pointers.insert(s);
                        } else {
                            arm_code_pointers.insert(v);
                        }
                    }
                }
            }
            let mut go = |t: u32, r: RegConsts, work: &mut Vec<u32>| {
                if !decoded.contains_key(&t) && !(in_bounds(t) && near(t)) {
                    return;
                }
                let merged = match at.get(&t) {
                    Some(prev) => meet(prev, &r),
                    None => r,
                };
                if at.get(&t) != Some(&merged) {
                    at.insert(t, merged);
                    work.push(t);
                }
            };
            let next = addr.wrapping_add(*len);
            match flow(inst, addr, *len, thumb, entry, imports, regs[7], noreturn_svc) {
                Flow::Seq | Flow::Call { .. } => go(next, next_regs, &mut work),
                Flow::Jump(t) => go(t, next_regs, &mut work),
                Flow::Fork(t) => {
                    go(t, next_regs, &mut work);
                    go(next, next_regs, &mut work);
                }
                Flow::TailCall(t, _) => go(t, next_regs, &mut work),
                Flow::Return | Flow::Halt => {}
            }
        }
    }

    // Pass 2: build blocks. Each leader starts a block that runs until a
    // terminating instruction or the address just before the next leader.
    let mut blocks = Vec::new();
    for &addr in &leaders {
        // A leader whose first instruction failed to decode becomes a lone trapping
        // block, so a branch/switch that targets it stays well-formed (the target is
        // a real block) while executing it faults loudly.
        if trap_leaders.contains(&addr) {
            // A trap block runs no guest instructions, so it is billed nothing.
            blocks.push(Block { addr, stmts: Vec::new(), term: Term::Unreachable, arm_count: 0 });
            continue;
        }
        // A leader can be out of bounds (a branch target past the image); such a
        // block is unreachable in practice - skip it.
        if !decoded.contains_key(&addr) {
            continue;
        }
        let mut cursor = addr;
        let mut stmts = Vec::new();
        // Guest instructions lifted into this block - the emulator's unit of guest work.
        let mut arm_count = 0u32;
        let term = loop {
            let Some((inst, len, applied, in_it)) = decoded.get(&cursor) else {
                // >>> RUNNING STRAIGHT INTO AN INSTRUCTION THAT DID NOT DECODE MUST TRAP,
                // >>> NOT RETURN. This is the single most expensive bug shape this file
                // >>> has produced, so it is spelled out.
                //
                // Pass 1 records an undecodable address as a TRAP LEADER and a block is
                // built for it above (`Term::Unreachable`), on the reasoning that a
                // branch to it stays well-formed and faults when taken. But the common
                // way to reach one is not a branch - it is the PRECEDING INSTRUCTION
                // falling through. That path used to land here, find nothing in
                // `decoded`, and end the block with `Term::Halt`, which emits a plain
                // `return`. So the trap block was never jumped to at all, and the real
                // behaviour was: run the block's valid prefix, then RETURN OUT OF THE
                // FUNCTION with half its work done and its registers half-updated.
                //
                // MEASURED (2026-08-24): two undecodable NEON ops in one title's audio
                // mixer cut its unrolled inner loop after four instructions. The mixer
                // returned early every grain, and the failure surfaced as a null
                // dereference several calls away, in a different function, with nothing
                // anywhere naming the cause. It cost most of a session.
                //
                // A gap reached by fallthrough is exactly as unrunnable as one reached by
                // a branch, so it gets the same answer. `Term::Halt` is kept for the
                // OTHER way to get here - walking off the end of the decoded region -
                // because that is not a decode gap and has no trap leader.
                if trap_leaders.contains(&cursor) {
                    break Term::Unreachable;
                }
                // Fell through to code that was never decoded (off image or
                // unreachable): stop the function here.
                break Term::Halt;
            };
            // A table branch ends the block with a computed jump built from the table
            // recovered in pass 1. An unresolved one is a clean per-function failure.
            if matches!(inst.opcode, Opcode::TBB | Opcode::TBH) || arm_add_pc_switch(inst, cursor, thumb, 0, 0).is_some() || switches.contains_key(&cursor) {
                match switches.get(&cursor) {
                    Some(info) => {
                        arm_count += 1;
                        let index = if info.bias == 0 && info.shift == 0 {
                            Value::Reg(info.index)
                        } else {
                            bin(
                                BinOp::Lsr,
                                bin(BinOp::Add, Value::Reg(info.index), Value::Imm(info.bias as u32)),
                                Value::Imm(info.shift),
                            )
                        };
                        break Term::Switch {
                            index,
                            targets: info.targets.clone(),
                            default: info.default,
                        };
                    }
                    // An unresolved table branch cannot be lowered. Strict: a clean
                    // per-function failure. Lenient: trap here so the rest lifts.
                    None if isolate => break Term::Unreachable,
                    None => return Err(Error::Unsupported { addr: cursor, opcode: inst.opcode }),
                }
            }
            let (mut effects, term) =
                match lower_insn(inst, cursor, *len, *applied, *in_it, thumb, entry, imports, &indirect_imports, &indirect_redirects) {
                    Ok(v) => v,
                    // An unlifted instruction (e.g. `udf`, or a NEON op not yet
                    // covered). Strict callers report it; the lenient build runs the
                    // block's valid prefix then traps, isolating the gap to this block
                    // instead of stubbing the whole function.
                    Err(_) if isolate => break Term::Unreachable,
                    Err(e) => return Err(e),
                };
            // The conditional form of a terminator: its effects run under the
            // condition and the transfer itself becomes the conditional one.
            let term = match term {
                Some(t) if *applied != ConditionCode::AL => {
                    if !effects.is_empty() {
                        effects = vec![Stmt::Guard(*applied, std::mem::take(&mut effects))];
                    }
                    Some(match t {
                        Term::Return => Term::ReturnIf { cond: *applied },
                        Term::Jump(t) => Term::Branch { cond: *applied, taken: t },
                        // A conditional halt: the (guarded) effect is what halts.
                        Term::Halt => Term::Fallthrough,
                        other => other,
                    })
                }
                t => t,
            };
            stmts.append(&mut effects);
            arm_count += 1;
            cursor = cursor.wrapping_add(*len);
            if let Some(t) = term {
                break t;
            }
            // A block also ends when the next instruction begins another block.
            if leaders.contains(&cursor) && decoded.contains_key(&cursor) {
                break Term::Fallthrough;
            }
        };
        blocks.push(Block { addr, stmts, term, arm_count });
    }
    blocks.sort_by_key(|b| b.addr);

    // Repair, rather than reject, a function with an intra-function branch to an address
    // that never became a block (a leader pass 1 could not decode to). Every such target
    // gets a trapping block of its own, which makes the function well-formed by
    // construction: the paths that DID lift run, and only the one that reaches the gap
    // faults, at the exact instruction.
    //
    // The lenient build does this because the alternative is worse than it looks:
    // dropping the function leaves nothing at its address, so every DYNAMIC dispatch to
    // it becomes a miss - and a real function reached only through a function pointer
    // (a C++ virtual, a registered callback) then takes the whole title down over a gap
    // it may never have executed. Strict callers keep the old behaviour: the caller sees
    // a malformed tentative function and drops it.
    //
    // This runs BEFORE the runaway guard below, so a repaired block that lands far away -
    // which means the branch target was garbage, not a gap - widens the span and gets the
    // whole discovery rejected, exactly as an over-read switch table does.
    if isolate {
        let mut missing: BTreeSet<u32> = BTreeSet::new();
        let is_block = |a: u32, bs: &[Block]| bs.iter().any(|b| b.addr == a);
        for b in &blocks {
            match &b.term {
                Term::Jump(t) | Term::Branch { taken: t, .. } | Term::BranchZero { taken: t, .. } => {
                    if !is_block(*t, &blocks) {
                        missing.insert(*t);
                    }
                }
                Term::Switch { targets, default, .. } => {
                    for t in targets.iter().chain(default.iter()) {
                        if !is_block(*t, &blocks) {
                            missing.insert(*t);
                        }
                    }
                }
                Term::Fallthrough
                | Term::Return
                | Term::ReturnIf { .. }
                | Term::Halt
                | Term::Unreachable => {}
            }
        }
        if !missing.is_empty() {
            blocks.extend(
                missing.iter().map(|&addr| Block {
                    addr,
                    stmts: Vec::new(),
                    term: Term::Unreachable,
                    arm_count: 0,
                }),
            );
            blocks.sort_by_key(|b| b.addr);
        }
    }

    // Runaway guard: a real function's blocks cluster tightly, but a mis-recovered
    // computed jump (a `tbh` whose bound we could not read from the range check, so
    // the extent fallback over-reads the table into data) seeds "case" leaders
    // scattered across the image, dragging unrelated code into this function as one
    // enormous, sometimes self-looping blob. When the block span is implausibly wide,
    // the discovery is not a single function: reject it. Strict callers see the
    // failure; the lenient build stubs it cleanly (a loud trap on entry) rather than
    // emitting - and running - the garbage. Keyed on span, not block count, so a
    // legitimately large but contiguous function is unaffected.
    const MAX_FUNC_SPAN: u32 = 0x1_0000; // 64 KiB - larger than any real function
    if let (Some(first), Some(last)) = (blocks.first(), blocks.last()) {
        if last.addr.wrapping_sub(first.addr) > MAX_FUNC_SPAN {
            return Err(Error::Decode { addr: last.addr });
        }
    }

    // The function is complete, so its flag liveness is knowable. Annotating HERE rather
    // than at each of the three `discover` call sites is what makes it unmissable: a
    // build path that forgot the pass would silently emit the slow code and pass every
    // test, which is the shape of defect this project keeps meeting.
    let mut func = Func { addr: entry, thumb, blocks, stub: false };
    // Fold runs of same-condition predication (a Thumb `IT` block) into one guard BEFORE
    // liveness runs, so the analysis sees the shape the emitter will actually emit.
    for b in &mut func.blocks {
        crate::flags::merge_guards(&mut b.stmts);
    }
    crate::flags::annotate(&mut func);
    // And with liveness settled, a flag statement whose four flags are ALL dead and whose
    // sum nothing reads back is not a cheaper statement - it is no statement at all.
    crate::flags::drop_dead_flag_adds(&mut func);

    Ok(Discovered {
        func,
        trap_leaders: trap_leaders.into_iter().collect(),
        callees: callees.into_iter().collect(),
        code_pointers: code_pointers.into_iter().collect(),
        arm_code_pointers: arm_code_pointers.into_iter().collect(),
    })
}

/// Build the initial ITSTATE byte from an `IT` instruction's `firstcond`/`mask`.
fn it_state_from(inst: &Instruction) -> u8 {
    let firstcond = match inst.operands[0] {
        Operand::Imm32(v) => v as u8,
        _ => 0,
    };
    let mask = match inst.operands[1] {
        Operand::Imm32(v) => v as u8,
        _ => 0,
    };
    ((firstcond & 0xF) << 4) | (mask & 0xF)
}

// --- instruction lowering -------------------------------------------------

fn regnum(op: &Operand) -> Option<u8> {
    match op {
        Operand::Reg(r) => Some(r.number()),
        Operand::RegWBack(r, _) => Some(r.number()),
        _ => None,
    }
}

/// The coprocessor number of a `CReg` operand (`c0`..`c15`), or None.
fn cregnum(op: &Operand) -> Option<u8> {
    match op {
        Operand::CReg(c) => Some(c.number()),
        _ => None,
    }
}

/// Whether a coprocessor register-move (`MRC`/`MCR`) names the ARM thread-ID
/// register: `p15, 0, Rt, c13, c0, {2,3}`. `c13,c0,opc2=2` is TPIDRURW and
/// `opc2=3` TPIDRURO - both the user-mode thread pointer the C runtime uses for
/// thread-local storage. `crn`/`crm` are the `CReg` operands; `coproc`/`opc1`/
/// `opc2` come from the decoded opcode.
fn is_thread_pointer_reg(coproc: u8, opc1: u8, crn: &Operand, crm: &Operand, opc2: u8) -> bool {
    coproc == 15
        && opc1 == 0
        && cregnum(crn) == Some(13)
        && cregnum(crm) == Some(0)
        && (opc2 == 2 || opc2 == 3)
}

fn imm(op: &Operand) -> Option<u32> {
    match op {
        Operand::Imm32(v) | Operand::Imm(v) => Some(*v),
        Operand::Imm12(v) => Some(*v as u32),
        _ => None,
    }
}

/// A data-processing operand: a register read, an immediate, or a shifted
/// register (the Thumb-2 wide `add.w`/`mov.w`/... forms carry an explicit shift,
/// often `lsl #0`).
fn operand_value(op: &Operand, pc: u32) -> Option<Value> {
    match op {
        // Reading r15 as a data-processing SOURCE yields the ISA's pc constant, not the
        // register file's r15 (which the transpiler does not maintain per-instruction -
        // it would read 0). `lower_addr` already folds it this way for a memory operand
        // and `ADR` for its own form; a hand-written `add r12, pc, #imm` followed by
        // `blx r12` - the computed-jump idiom in an assembly MD5 - reaches it HERE, and
        // without this fold it dispatched to the bare immediate.
        Operand::Reg(r) if r.number() == 15 => Some(Value::Imm(pc)),
        Operand::Reg(r) => Some(Value::Reg(r.number())),
        Operand::Imm32(v) | Operand::Imm(v) => Some(Value::Imm(*v)),
        Operand::Imm12(v) => Some(Value::Imm(*v as u32)),
        Operand::RegShift(rs) => shift_operand(rs),
        _ => None,
    }
}

/// Lower a shifted-register operand into its value. Immediate shifts fold to a
/// single wasm shift (with the ARM `#0` conventions: LSL #0 is the bare register,
/// LSR/ASR #0 mean shift by 32, ROR #0 is RRX). Register-controlled shifts mask
/// the amount to a byte; amounts >= 32 are left to wasm's mod-32 shift for now
/// (exact wide-shift semantics are a later refinement). `RRX` (needs the carry
/// flag) is not modeled yet.
fn shift_operand(rs: &RegShift) -> Option<Value> {
    match rs.into_shift() {
        RegShiftStyle::RegImm(s) => {
            let base = Value::Reg(s.shiftee().number());
            let n = s.imm() as u32;
            Some(match s.stype() {
                ShiftStyle::LSL => {
                    if n == 0 {
                        base
                    } else {
                        bin(BinOp::Shl, base, Value::Imm(n))
                    }
                }
                ShiftStyle::LSR => {
                    if n == 0 {
                        Value::Imm(0) // LSR #32
                    } else {
                        bin(BinOp::Lsr, base, Value::Imm(n))
                    }
                }
                ShiftStyle::ASR => {
                    // ASR #0 means ASR #32: fill with the sign bit.
                    bin(BinOp::Asr, base, Value::Imm(if n == 0 { 31 } else { n }))
                }
                ShiftStyle::ROR => {
                    if n == 0 {
                        return None; // RRX (rotate through carry) not modeled yet
                    }
                    bin(
                        BinOp::Or,
                        bin(BinOp::Lsr, base.clone(), Value::Imm(n)),
                        bin(BinOp::Shl, base, Value::Imm(32 - n)),
                    )
                }
            })
        }
        RegShiftStyle::RegReg(s) => {
            let base = Value::Reg(s.shiftee().number());
            let amt = bin(BinOp::And, Value::Reg(s.shifter().number()), Value::Imm(0xff));
            let op = match s.stype() {
                ShiftStyle::LSL => BinOp::Shl,
                ShiftStyle::LSR => BinOp::Lsr,
                ShiftStyle::ASR => BinOp::Asr,
                ShiftStyle::ROR => return None,
            };
            Some(bin(op, base, amt))
        }
    }
}

/// Decode a data-processing instruction's `(rd, rn, op2)`. Handles the 3-operand
/// form `op rd, rn, op2` and the 2-operand form `op rd, op2` (where rd is rn).
fn dataproc(inst: &Instruction, pc: u32) -> Option<(u8, Value, Value)> {
    let rd = regnum(&inst.operands[0])?;
    if matches!(inst.operands[2], Operand::Nothing) {
        let op2 = operand_value(&inst.operands[1], pc)?;
        Some((rd, Value::Reg(rd), op2))
    } else {
        let rn = operand_value(&inst.operands[1], pc)?;
        let op2 = operand_value(&inst.operands[2], pc)?;
        Some((rd, rn, op2))
    }
}

fn bin(op: BinOp, a: Value, b: Value) -> Value {
    Value::Bin(op, Box::new(a), Box::new(b))
}

/// The value tree for a full 32-bit byte reverse (ARM `rev`): move each byte to
/// the mirrored position. `x` is read four times (a pure register read, so this
/// is free).
fn byte_reverse(x: Value) -> Value {
    let b0 = bin(BinOp::And, bin(BinOp::Lsr, x.clone(), Value::Imm(24)), Value::Imm(0x0000_00FF));
    let b1 = bin(BinOp::And, bin(BinOp::Lsr, x.clone(), Value::Imm(8)), Value::Imm(0x0000_FF00));
    let b2 = bin(BinOp::And, bin(BinOp::Shl, x.clone(), Value::Imm(8)), Value::Imm(0x00FF_0000));
    let b3 = bin(BinOp::And, bin(BinOp::Shl, x, Value::Imm(24)), Value::Imm(0xFF00_0000));
    bin(BinOp::Or, bin(BinOp::Or, b0, b1), bin(BinOp::Or, b2, b3))
}

/// Lower a memory addressing operand into the effective address, plus any
/// base-register writeback statements (ordered relative to the access: pre-index
/// writeback happens before the access uses the base, post-index after).
struct Addr {
    /// Statements to run *before* the memory access (pre-index writeback).
    pre: Vec<Stmt>,
    /// The effective address expression.
    addr: Value,
    /// Statements to run *after* the memory access (post-index writeback).
    post: Vec<Stmt>,
}

fn lower_addr(op: &Operand, iaddr: u32, thumb: bool) -> Option<Addr> {
    let signed_off = |off: u16, add: bool| -> Value {
        let v = off as u32;
        Value::Imm(if add { v } else { v.wrapping_neg() })
    };
    // The pc-relative literal form `[pc, #off]`: the base register r15 is not
    // maintained per-instruction (pc is a wasm global the transpiler does not
    // update on every step), so fold it to the constant pc the ISA defines -
    // `Align(addr+4, 4)` in Thumb, `addr+8` in ARM. Matches `adr` and vldr/vstr.
    let base_value = |b: u8| -> Value {
        if b == 15 {
            let pc = if thumb { iaddr.wrapping_add(4) & !3 } else { iaddr.wrapping_add(8) };
            Value::Imm(pc)
        } else {
            Value::Reg(b)
        }
    };
    match op {
        Operand::RegDeref(base) => Some(Addr {
            pre: vec![],
            addr: base_value(base.number()),
            post: vec![],
        }),
        Operand::RegDerefPreindexOffset(base, off, add, wback) => {
            let b = base.number();
            // Const-fold a pc base so the literal address is correct.
            if b == 15 {
                return Some(Addr {
                    pre: vec![],
                    addr: bin(BinOp::Add, base_value(b), signed_off(*off, *add)),
                    post: vec![],
                });
            }
            let ea = bin(BinOp::Add, Value::Reg(b), signed_off(*off, *add));
            if *wback {
                // Writeback sets base = ea; the access then uses the new base.
                Some(Addr {
                    pre: vec![Stmt::SetReg(b, ea)],
                    addr: Value::Reg(b),
                    post: vec![],
                })
            } else {
                Some(Addr { pre: vec![], addr: ea, post: vec![] })
            }
        }
        Operand::RegDerefPostindexOffset(base, off, add, _wback) => {
            // Post-index: access uses the old base, then base += offset.
            let b = base.number();
            Some(Addr {
                pre: vec![],
                addr: Value::Reg(b),
                post: vec![Stmt::SetReg(
                    b,
                    bin(BinOp::Add, Value::Reg(b), signed_off(*off, *add)),
                )],
            })
        }
        // Register offset (optionally shifted): `[Rn, Rm{, shift}]{!}`.
        Operand::RegDerefPreindexReg(base, index, add, wback) => {
            reg_offset_addr(base.number(), Value::Reg(index.number()), *add, *wback, false)
        }
        Operand::RegDerefPreindexRegShift(base, rs, add, wback) => {
            reg_offset_addr(base.number(), shift_operand(rs)?, *add, *wback, false)
        }
        Operand::RegDerefPostindexReg(base, index, add, _wback) => {
            reg_offset_addr(base.number(), Value::Reg(index.number()), *add, true, true)
        }
        Operand::RegDerefPostindexRegShift(base, rs, add, _wback) => {
            reg_offset_addr(base.number(), shift_operand(rs)?, *add, true, true)
        }
        _ => None,
    }
}

/// Build an [`Addr`] for a register-offset addressing mode: effective address is
/// `base +/- offset`. `post` selects post-index (access at the old base, then
/// writeback); otherwise it is pre-index with optional writeback.
fn reg_offset_addr(base: u8, offset: Value, add: bool, wback: bool, post: bool) -> Option<Addr> {
    let ea = bin(if add { BinOp::Add } else { BinOp::Sub }, Value::Reg(base), offset);
    if post {
        Some(Addr { pre: vec![], addr: Value::Reg(base), post: vec![Stmt::SetReg(base, ea)] })
    } else if wback {
        Some(Addr { pre: vec![Stmt::SetReg(base, ea)], addr: Value::Reg(base), post: vec![] })
    } else {
        Some(Addr { pre: vec![], addr: ea, post: vec![] })
    }
}

/// Lower one instruction into its statements and, if it terminates the block,
/// the terminator. `cond` is the applied (possibly IT) condition; when it is not
/// `AL` the data effects are wrapped in a `Guard`.
fn lower_insn(
    inst: &Instruction,
    addr: u32,
    len: u32,
    cond: ConditionCode,
    in_it: bool,
    thumb: bool,
    entry: u32,
    imports: &Imports,
    indirect_imports: &BTreeMap<u32, u32>,
    indirect_redirects: &BTreeMap<u32, u32>,
) -> Result<(Vec<Stmt>, Option<Term>), Error> {
    use Opcode::*;
    let err = || Error::Operand { addr };
    let ops = &inst.operands;

    // Control-flow terminators first (these are never predicated in our corpus
    // except conditional branches, which carry their own condition).
    match inst.opcode {
        B => {
            let target = imports.resolve(branch_target(inst, addr, thumb).ok_or_else(err)?);
            if target == addr {
                return Ok((vec![], Some(Term::Halt)));
            }
            if cond == ConditionCode::AL {
                // Tail call to an import stub/veneer: run the import, then return
                // to our caller (lr already holds the caller's return address).
                if let Some(index) = imports.get(target) {
                    return Ok((vec![Stmt::Import(index)], Some(Term::Return)));
                }
                // Tail call to another translated function: call it and return.
                // lr is left untouched, so the callee returns straight to our
                // caller. Must match the pass-1 classification in `flow` exactly,
                // or the block's terminator disagrees with its successor set.
                if is_tail_call(target, entry) {
                    return Ok((vec![Stmt::Call { target }], Some(Term::Return)));
                }
                return Ok((vec![], Some(Term::Jump(target))));
            }
            return Ok((vec![], Some(Term::Branch { cond, taken: target })));
        }
        CBZ | CBNZ => {
            let reg = regnum(&ops[0]).ok_or_else(err)?;
            let target = branch_target(inst, addr, thumb).ok_or_else(err)?;
            return Ok((
                vec![],
                Some(Term::BranchZero {
                    reg,
                    nonzero: inst.opcode == CBNZ,
                    taken: target,
                }),
            ));
        }
        // `ldr pc, [sp], #4`: the return of a function that pushed only lr - the
        // address loaded IS our caller, so it is a plain return (the load itself is
        // skipped, as `pop {pc}` skips it). Any other `ldr pc, [...]` is an indirect
        // jump through memory: dispatch to the loaded target, then return to our
        // caller, exactly like `bx rN`.
        LDR if regnum(&ops[0]) == Some(15) => {
            let a = lower_addr(&ops[1], addr, thumb).ok_or_else(err)?;
            let pops_sp = matches!(
                &ops[1],
                Operand::RegDerefPostindexOffset(b, _, _, _) if b.number() == 13
            );
            let mut stmts = a.pre;
            if pops_sp {
                stmts.extend(a.post);
                return Ok((stmts, Some(Term::Return)));
            }
            stmts.push(Stmt::SetReg(
                15,
                Value::Load { addr: Box::new(a.addr), size: MemSize::Word, signed: false },
            ));
            stmts.extend(a.post);
            stmts.push(Stmt::CallIndirect { addr: Value::Reg(15), set_lr: None });
            return Ok((stmts, Some(Term::Return)));
        }
        // `mov pc, lr` returns; `mov pc, rN` is an indirect tail call.
        MOV if regnum(&ops[0]) == Some(15) => {
            return match ops[1] {
                Operand::Reg(r) if r.number() == 14 => Ok((vec![], Some(Term::Return))),
                Operand::Reg(r) => Ok((
                    vec![Stmt::CallIndirect { addr: Value::Reg(r.number()), set_lr: None }],
                    Some(Term::Return),
                )),
                _ => Err(err()),
            };
        }
        BX => {
            return match ops[0] {
                Operand::Reg(r) if r.number() == 14 => Ok((vec![], Some(Term::Return))),
                // `bx pc`: a direct tail call to the ARM code at the pc value (see
                // `flow`); the callee returns to OUR caller, so call then return.
                Operand::Reg(r) if r.number() == 15 => {
                    let pc = if thumb { addr.wrapping_add(4) & !3 } else { addr.wrapping_add(8) };
                    Ok((vec![Stmt::Call { target: pc }], Some(Term::Return)))
                }
                // `bx rN` tail-calls through a function pointer: dispatch to the
                // runtime target, then return to our caller (lr is unchanged, so
                // the callee's own return unwinds past us correctly). If the target
                // is a known import stub (tracked constant), tail-call the host
                // import directly instead of dispatching to its placeholder.
                Operand::Reg(r) => Ok((
                    match (indirect_imports.get(&addr), indirect_redirects.get(&addr)) {
                        (Some(&index), _) => vec![Stmt::Import(index)],
                        (_, Some(&target)) => vec![Stmt::Call { target }],
                        _ => vec![Stmt::CallIndirect { addr: Value::Reg(r.number()), set_lr: None }],
                    },
                    Some(Term::Return),
                )),
                _ => Err(err()),
            };
        }
        BL | BLX => {
            let ret = addr.wrapping_add(len);
            // bl/blx set lr = return address (Thumb bit set in Thumb state).
            let lr = if thumb { ret | 1 } else { ret };
            let stmts = match branch_target(inst, addr, thumb).map(|t| imports.resolve(t)) {
                // Direct target: set lr, then call (the target is a constant, so the
                // order is safe).
                Some(target) => match imports.get(target) {
                    Some(index) => vec![Stmt::SetReg(14, Value::Imm(lr)), Stmt::Import(index)],
                    None => vec![Stmt::SetReg(14, Value::Imm(lr)), Stmt::Call { target }],
                },
                // Register-target `blx rN`: indirect call through a function pointer,
                // resolved at runtime by the dispatcher. The CallIndirect sets lr
                // itself, AFTER reading the target, so `blx lr` still works. If the
                // target is a known import stub (tracked constant), lower it as the
                // host import (set lr, then call) rather than dispatching to the
                // stub's return-(-1) placeholder.
                None => match ops[0] {
                    Operand::Reg(r) => match (indirect_imports.get(&addr), indirect_redirects.get(&addr)) {
                        (Some(&index), _) => vec![Stmt::SetReg(14, Value::Imm(lr)), Stmt::Import(index)],
                        (_, Some(&target)) => vec![Stmt::SetReg(14, Value::Imm(lr)), Stmt::Call { target }],
                        _ => vec![Stmt::CallIndirect { addr: Value::Reg(r.number()), set_lr: Some(lr) }],
                    },
                    _ => return Err(err()),
                },
            };
            return Ok((stmts, None));
        }
        POP | LDM(..) if writes_pc(inst) => {
            let mut stmts = lower_ldm(inst, addr)?;
            // The popped pc becomes the return; we do not write r15.
            let _ = &mut stmts;
            return Ok((stmts, Some(Term::Return)));
        }
        SVC => {
            let n = imm(&ops[0]).unwrap_or(0);
            return Ok((vec![Stmt::Svc(n)], None));
        }
        // A breakpoint is a debug trap the compiler plants on an unreachable /
        // abort path (`__builtin_trap`, a failed noreturn guard). Reaching one is
        // a fault, so it ends the run here; the block terminates and the rest of
        // the function still translates.
        BKPT => {
            return Ok((vec![], Some(Term::Halt)));
        }
        _ => {}
    }

    // Non-control effects. Build the effect list, then wrap in a Guard if
    // predicated.
    let effects = lower_effects(inst, addr, in_it)?;
    if cond == ConditionCode::AL || effects.is_empty() {
        Ok((effects, None))
    } else {
        Ok((vec![Stmt::Guard(cond, effects)], None))
    }
}

/// Lower the (non-control) data/memory effects of an instruction. Inside an IT
/// block the S bit is suppressed for data-processing ops (ARM: only cmp/cmn/tst,
/// which set flags unconditionally, still do).
fn lower_effects(inst: &Instruction, addr: u32, in_it: bool) -> Result<Vec<Stmt>, Error> {
    use Opcode::*;
    let err = || Error::Operand { addr };
    let ops = &inst.operands;
    let sets_flags = inst.s && !in_it;
    // The value an instruction reads when it uses r15 as a source: `Align(addr+4, 4)` in
    // Thumb, `addr+8` in ARM. Const per instruction, so it folds; see `operand_value`.
    let pc_const = if inst.thumb { addr.wrapping_add(4) & !3 } else { addr.wrapping_add(8) };

    let mut out = Vec::new();
    match inst.opcode {
        NOP | IT | HINT => {}

        // Memory barriers and cache preload hints have no effect on the guest's
        // observable state in our memory model (one guest CPU worker, sequential
        // consistency at host-call sync points), so they lower to nothing.
        DMB | DSB | ISB | PLD | PLI => {}

        // Coprocessor register move: the only coprocessor a user-mode Vita title
        // touches is the thread-ID register, `MRC/MCR p15, 0, Rt, c13, c0, {2,3}`
        // (TPIDRURW / TPIDRURO). `MRC` reads the per-thread pointer into `Rt`; `MCR`
        // writes it. The compiler emits the read to reach ELF thread-local storage
        // (`__thread` variables at `tp + offset`). Any other coprocessor access is a
        // privileged / system operation a user title never issues; leave it a gap.
        MRC(coproc, opc1, opc2, _) => {
            let rt = regnum(&ops[0]).ok_or_else(err)?;
            if is_thread_pointer_reg(coproc, opc1, &ops[1], &ops[2], opc2) {
                out.push(Stmt::SetReg(rt, Value::ThreadPtr));
            } else {
                return Err(Error::Unsupported { addr, opcode: inst.opcode });
            }
        }
        MCR(coproc, opc1, opc2, _) => {
            let rt = regnum(&ops[0]).ok_or_else(err)?;
            if is_thread_pointer_reg(coproc, opc1, &ops[1], &ops[2], opc2) {
                out.push(Stmt::SetThreadPtr(Value::Reg(rt)));
            } else {
                return Err(Error::Unsupported { addr, opcode: inst.opcode });
            }
        }

        ADR => {
            // adr rd, label => rd = pc-relative constant. pc is addr+8 in ARM,
            // (addr+4) word-aligned in Thumb. Const-folded.
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let disp = ops.iter().find_map(imm).ok_or_else(err)?;
            let pc = if inst.thumb {
                addr.wrapping_add(4) & !3
            } else {
                addr.wrapping_add(8)
            };
            out.push(Stmt::SetReg(rd, Value::Imm(pc.wrapping_add(disp))));
        }

        MOV => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let src = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            // Flags before the write: the value expression reads original regs.
            if sets_flags {
                let carry = match (&src, &ops[1]) {
                    (Value::Imm(v), _) => modified_imm_carry(*v, inst.thumb),
                    // ARM `movs rd, rm, <shift> #n` (the A32 spelling of `lsls`/`lsrs`/
                    // `asrs`/`rors`): C is the shifter carry-out. newlib's memcpy tail
                    // reads it (`lsls r2, r2, #31 ; ldrhhs ...`).
                    (_, Operand::RegShift(rs)) => reg_shift_carry(rs),
                    _ => None,
                };
                out.push(Stmt::FlagsLogic { value: src.clone(), carry, live: ALL_FLAGS });
            }
            out.push(Stmt::SetReg(rd, src));
        }
        MOVT => {
            // movt rd, #imm16: set the top halfword, keep the low halfword.
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let hi = imm(&ops[1]).ok_or_else(err)?;
            let value = bin(
                BinOp::Or,
                bin(BinOp::And, Value::Reg(rd), Value::Imm(0x0000_FFFF)),
                Value::Imm(hi << 16),
            );
            out.push(Stmt::SetReg(rd, value));
        }
        MVN => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let src = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            // MVN's carry-out comes from the raw immediate expansion, not the ~value;
            // a shifted register contributes its shifter carry-out.
            let carry = match (&src, &ops[1]) {
                (Value::Imm(v), _) => modified_imm_carry(*v, inst.thumb),
                (_, Operand::RegShift(rs)) => reg_shift_carry(rs),
                _ => None,
            };
            let value = Value::Not(Box::new(src));
            if sets_flags {
                out.push(Stmt::FlagsLogic { value: value.clone(), carry, live: ALL_FLAGS });
            }
            out.push(Stmt::SetReg(rd, value));
        }

        UADD8 => {
            // Byte-wise unsigned add setting the per-byte GE flags (consumed by
            // `sel`). Operands: rd, rn, rm.
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rn = regnum(&ops[1]).ok_or_else(err)?;
            let rm = regnum(&ops[2]).ok_or_else(err)?;
            out.push(Stmt::Uadd8 { rd, rn, rm });
        }
        SEL => {
            // Byte-wise select by the GE flags a prior parallel add/sub set.
            // Operands: rd, rn, rm.
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rn = regnum(&ops[1]).ok_or_else(err)?;
            let rm = regnum(&ops[2]).ok_or_else(err)?;
            out.push(Stmt::Sel { rd, rn, rm });
        }

        ADD => {
            let (rd, rn, op2) = dataproc(inst, pc_const).ok_or_else(err)?;
            if sets_flags {
                out.push(Stmt::FlagsAdd {
                    a: rn.clone(),
                    b: op2.clone(),
                    cin: Value::Imm(0),
                    live: ALL_FLAGS,
                });
            }
            out.push(Stmt::SetReg(rd, bin(BinOp::Add, rn, op2)));
        }
        SUB => {
            let (rd, rn, op2) = dataproc(inst, pc_const).ok_or_else(err)?;
            if sets_flags {
                out.push(Stmt::FlagsAdd {
                    a: rn.clone(),
                    b: Value::Not(Box::new(op2.clone())),
                    cin: Value::Imm(1),
                    live: ALL_FLAGS,
                });
            }
            out.push(Stmt::SetReg(rd, bin(BinOp::Sub, rn, op2)));
        }
        // adc rd, rn, op2 => rd = rn + op2 + C. The carry-in is the runtime C flag.
        // When it also sets flags, `FlagsAdd` computes the sum (and overwrites C with
        // the carry-out), so the result must read that already-computed sum rather
        // than re-adding `Flag(C)` - which would now be the carry-out, not the
        // carry-in (and `rd` may alias `rn`, so there is no re-reading it either).
        ADC => {
            let (rd, rn, op2) = dataproc(inst, pc_const).ok_or_else(err)?;
            if sets_flags {
                out.push(Stmt::FlagsAdd {
                    a: rn,
                    b: op2,
                    cin: Value::Flag(crate::abi::Flag::C),
                    live: ALL_FLAGS,
                });
                out.push(Stmt::SetReg(rd, Value::CarryAddResult));
            } else {
                let sum =
                    bin(BinOp::Add, bin(BinOp::Add, rn, op2), Value::Flag(crate::abi::Flag::C));
                out.push(Stmt::SetReg(rd, sum));
            }
        }
        // sbc rd, rn, op2 => rd = rn - op2 - NOT(C) = rn + ~op2 + C.
        SBC => {
            let (rd, rn, op2) = dataproc(inst, pc_const).ok_or_else(err)?;
            let not_op2 = Value::Not(Box::new(op2));
            if sets_flags {
                out.push(Stmt::FlagsAdd {
                    a: rn,
                    b: not_op2,
                    cin: Value::Flag(crate::abi::Flag::C),
                    live: ALL_FLAGS,
                });
                out.push(Stmt::SetReg(rd, Value::CarryAddResult));
            } else {
                let diff = bin(
                    BinOp::Add,
                    bin(BinOp::Add, rn, not_op2),
                    Value::Flag(crate::abi::Flag::C),
                );
                out.push(Stmt::SetReg(rd, diff));
            }
        }
        RSB => {
            // rsb rd, rn, op2 => rd = op2 - rn.
            let (rd, rn, op2) = dataproc(inst, pc_const).ok_or_else(err)?;
            if sets_flags {
                out.push(Stmt::FlagsAdd {
                    a: op2.clone(),
                    b: Value::Not(Box::new(rn.clone())),
                    cin: Value::Imm(1),
                    live: ALL_FLAGS,
                });
            }
            out.push(Stmt::SetReg(rd, bin(BinOp::Sub, op2, rn)));
        }
        CMP => {
            let a = operand_value(&ops[0], pc_const).ok_or_else(err)?;
            let b = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            out.push(Stmt::FlagsAdd {
                a,
                b: Value::Not(Box::new(b)),
                cin: Value::Imm(1),
                live: ALL_FLAGS,
            });
        }
        CMN => {
            let a = operand_value(&ops[0], pc_const).ok_or_else(err)?;
            let b = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            out.push(Stmt::FlagsAdd { a, b, cin: Value::Imm(0), live: ALL_FLAGS });
        }

        AND | BIC | ORR | ORN | EOR | TST | TEQ => {
            let (rd, rn, op2) = dataproc(inst, pc_const).ok_or_else(err)?;
            // A flag-setting logical op with a rotate-form immediate updates C from the
            // immediate expansion (ThumbExpandImm_C). Read it from the *raw* immediate,
            // before BIC/ORN complement the operand below.
            // A shifted-register second operand (`ands r6, r3, r0, asr #32` - newlib's
            // qsort selects its pivot on the C this leaves) carries the shifter's
            // carry-out instead.
            let op2_operand =
                if matches!(ops[2], Operand::Nothing) { &ops[1] } else { &ops[2] };
            let imm_carry = match (&op2, op2_operand) {
                (Value::Imm(v), _) => modified_imm_carry(*v, inst.thumb),
                (_, Operand::RegShift(rs)) => reg_shift_carry(rs),
                _ => None,
            };
            // BIC/ORN take the bitwise complement of the second operand.
            let op2 = if matches!(inst.opcode, BIC | ORN) {
                Value::Not(Box::new(op2))
            } else {
                op2
            };
            let binop = match inst.opcode {
                ORR | ORN => BinOp::Or,
                EOR | TEQ => BinOp::Xor,
                _ => BinOp::And, // AND, BIC, TST
            };
            let result = bin(binop, rn, op2);
            // `tst`/`teq` set flags only (libgcc's signed divide fixes its result's
            // sign with `teq ip, r0 ; rsbmi r0, r0, #0`).
            if matches!(inst.opcode, TST | TEQ) {
                out.push(Stmt::FlagsLogic { value: result, carry: imm_carry, live: ALL_FLAGS });
            } else {
                if sets_flags {
                    out.push(Stmt::FlagsLogic {
                        value: result.clone(),
                        carry: imm_carry,
                        live: ALL_FLAGS,
                    });
                }
                out.push(Stmt::SetReg(rd, result));
            }
        }

        LSL | LSR | ASR => {
            let (rd, rn, sh) = dataproc(inst, pc_const).ok_or_else(err)?;
            if let Value::Imm(_) = sh {
                // Immediate-amount shift: the amount is known at lowering, so wasm's
                // masked shift and the constant-folded `shift_carry` are already exact.
                let binop = match inst.opcode {
                    LSL => BinOp::Shl,
                    LSR => BinOp::Lsr,
                    _ => BinOp::Asr,
                };
                let result = bin(binop, rn.clone(), sh.clone());
                if sets_flags {
                    let carry = shift_carry(inst.opcode, &rn, &sh);
                    out.push(Stmt::FlagsLogic {
                        value: result.clone(),
                        carry,
                        live: ALL_FLAGS,
                    });
                }
                out.push(Stmt::SetReg(rd, result));
            } else {
                // Register-amount shift: the exact ARM result AND carry-out depend on
                // the runtime amount (`Rm[7:0]`), which wasm's mod-32 shift and a
                // constant carry expression cannot model. Emit the dedicated exact form.
                use crate::ir::ShiftKind;
                let kind = match inst.opcode {
                    LSL => ShiftKind::Lsl,
                    LSR => ShiftKind::Lsr,
                    _ => ShiftKind::Asr,
                };
                out.push(Stmt::ShiftRegFlags {
                    kind,
                    rd,
                    rn,
                    amount: sh,
                    set_flags: sets_flags,
                    live: ALL_FLAGS,
                });
            }
        }
        // ror rd, rn, rm/#imm: rotate right. Amount is masked mod 32 (wasm rotr,
        // matching ARM's register-rotate masking); ROR's carry-out is bit 31 of
        // the result when it sets flags.
        ROR => {
            let (rd, rn, sh) = dataproc(inst, pc_const).ok_or_else(err)?;
            let result = bin(BinOp::Ror, rn, sh);
            if sets_flags {
                let carry = bin(BinOp::And, bin(BinOp::Lsr, result.clone(), Value::Imm(31)), Value::Imm(1));
                out.push(Stmt::FlagsLogic {
                    value: result.clone(),
                    carry: Some(carry),
                    live: ALL_FLAGS,
                });
            }
            out.push(Stmt::SetReg(rd, result));
        }
        // bfi rd, rn, #lsb, #width: insert the low `width` bits of rn into rd at
        // bit `lsb`, leaving the rest of rd unchanged.
        BFI => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rn = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            let lsb = imm(&ops[2]).ok_or_else(err)?;
            let width = imm(&ops[3]).ok_or_else(err)?;
            let field = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
            let mask = field.wrapping_shl(lsb);
            let inserted = bin(
                BinOp::And,
                bin(BinOp::Shl, rn, Value::Imm(lsb)),
                Value::Imm(mask),
            );
            let kept = bin(BinOp::And, Value::Reg(rd), Value::Imm(!mask));
            out.push(Stmt::SetReg(rd, bin(BinOp::Or, kept, inserted)));
        }
        // bfc rd, #lsb, #width: clear the `width` bits of rd at bit `lsb`.
        BFC => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let lsb = imm(&ops[1]).ok_or_else(err)?;
            let width = imm(&ops[2]).ok_or_else(err)?;
            let field = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
            let mask = field.wrapping_shl(lsb);
            out.push(Stmt::SetReg(rd, bin(BinOp::And, Value::Reg(rd), Value::Imm(!mask))));
        }

        MUL => {
            let (rd, rn, op2) = dataproc(inst, pc_const).ok_or_else(err)?;
            let result = bin(BinOp::Mul, rn, op2);
            if sets_flags {
                out.push(Stmt::FlagsLogic { value: result.clone(), carry: None, live: ALL_FLAGS });
            }
            out.push(Stmt::SetReg(rd, result));
        }
        // umull/smull rdlo, rdhi, rn, rm: {rdhi:rdlo} = rn * rm (64-bit). The S
        // form (flag-setting) is not emitted by the compilers we target; ignore
        // flags here (the widening product's N/Z would need the 64-bit value).
        UMULL | SMULL | UMLAL | SMLAL => {
            let rdlo = regnum(&ops[0]).ok_or_else(err)?;
            let rdhi = regnum(&ops[1]).ok_or_else(err)?;
            let rn = operand_value(&ops[2], pc_const).ok_or_else(err)?;
            let rm = operand_value(&ops[3], pc_const).ok_or_else(err)?;
            out.push(Stmt::MulLong {
                rdlo,
                rdhi,
                rn,
                rm,
                signed: matches!(inst.opcode, SMULL | SMLAL),
                accumulate: matches!(inst.opcode, UMLAL | SMLAL),
            });
        }
        // clz rd, rm.
        CLZ => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rm = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            out.push(Stmt::SetReg(rd, Value::Clz(Box::new(rm))));
        }
        // rbit rd, rm: reverse the 32 bits of rm.
        RBIT => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rm = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            out.push(Stmt::Rbit { rd, rm });
        }
        // Signed halfword multiplies (DSP): rd = sext16(rn.half) * sext16(rm.half),
        // plus ra for the accumulating form. The Q flag these can set is not modelled
        // (nothing here reads it). smulw/smlaw: rd = (rn * sext16(rm.half)) >> 16.
        SMUL(n_top, m_top) | SMLA(n_top, m_top) => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rn = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            let rm = operand_value(&ops[2], pc_const).ok_or_else(err)?;
            let half = |v: Value, top: bool| {
                if top {
                    bin(BinOp::Asr, v, Value::Imm(16))
                } else {
                    bin(BinOp::Asr, bin(BinOp::Shl, v, Value::Imm(16)), Value::Imm(16))
                }
            };
            let prod = bin(BinOp::Mul, half(rn, n_top), half(rm, m_top));
            let result = if matches!(inst.opcode, SMLA(..)) {
                let ra = operand_value(&ops[3], pc_const).ok_or_else(err)?;
                bin(BinOp::Add, prod, ra)
            } else {
                prod
            };
            out.push(Stmt::SetReg(rd, result));
        }
        SMULW(m_top) | SMLAW(m_top) => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rn = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            let rm = operand_value(&ops[2], pc_const).ok_or_else(err)?;
            let half = if m_top {
                bin(BinOp::Asr, rm, Value::Imm(16))
            } else {
                bin(BinOp::Asr, bin(BinOp::Shl, rm, Value::Imm(16)), Value::Imm(16))
            };
            // Bits 47:16 of the 48-bit product, in 32-bit arithmetic: with rn split
            // as hi*2^16 + lo (hi signed, lo unsigned 16-bit), (rn*h) >> 16 is
            // hi*h + ((lo*h) >> 16), and lo*h fits an i32 (|h| < 2^15, lo < 2^16).
            let hi = bin(BinOp::Asr, rn.clone(), Value::Imm(16));
            let lo = bin(BinOp::And, rn, Value::Imm(0xFFFF));
            let word = bin(
                BinOp::Add,
                bin(BinOp::Mul, hi, half.clone()),
                bin(BinOp::Asr, bin(BinOp::Mul, lo, half), Value::Imm(16)),
            );
            let result = if matches!(inst.opcode, SMLAW(..)) {
                let ra = operand_value(&ops[3], pc_const).ok_or_else(err)?;
                bin(BinOp::Add, word, ra)
            } else {
                word
            };
            out.push(Stmt::SetReg(rd, result));
        }
        // mla rd, rn, rm, ra => rd = rn*rm + ra; mls => rd = ra - rn*rm.
        MLA | MLS => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rn = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            let rm = operand_value(&ops[2], pc_const).ok_or_else(err)?;
            let ra = operand_value(&ops[3], pc_const).ok_or_else(err)?;
            let prod = bin(BinOp::Mul, rn, rm);
            let result = if inst.opcode == MLS {
                bin(BinOp::Sub, ra, prod)
            } else {
                bin(BinOp::Add, prod, ra)
            };
            out.push(Stmt::SetReg(rd, result));
        }
        // rev rd, rm: reverse all four bytes.
        REV => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rm = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            out.push(Stmt::SetReg(rd, byte_reverse(rm)));
        }
        // rev16 rd, rm: reverse the bytes within each halfword.
        REV16 => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rm = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            // ((rm >> 8) & 0x00FF00FF) | ((rm << 8) & 0xFF00FF00)
            let hi = bin(BinOp::And, bin(BinOp::Lsr, rm.clone(), Value::Imm(8)), Value::Imm(0x00FF_00FF));
            let lo = bin(BinOp::And, bin(BinOp::Shl, rm, Value::Imm(8)), Value::Imm(0xFF00_FF00));
            out.push(Stmt::SetReg(rd, bin(BinOp::Or, hi, lo)));
        }
        REVSH => {
            // Byte-swap the low halfword, then sign-extend it (a big-endian int16 read:
            // FreeType's TrueType table parsers).
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rm = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            let hi = bin(BinOp::And, bin(BinOp::Lsr, rm.clone(), Value::Imm(8)), Value::Imm(0xFF));
            let lo = bin(BinOp::And, bin(BinOp::Shl, rm, Value::Imm(8)), Value::Imm(0xFF00));
            let swapped = bin(BinOp::Or, hi, lo);
            let sext = bin(BinOp::Asr, bin(BinOp::Shl, swapped, Value::Imm(16)), Value::Imm(16));
            out.push(Stmt::SetReg(rd, sext));
        }
        // Sign/zero extend a byte or halfword. An optional `ROR #rot` (rot in
        // {8,16,24}, carried by yaxpeax as a trailing Imm32 operand) rotates the
        // source right BEFORE the byte/half is extracted - the endian-swap idiom
        // `uxtb rd, rm, ror #16` relies on it. Dropping the rotate corrupts every
        // byte-swapped word (e.g. a big-endian table count read back as the wrong
        // value), so honour it here.
        SXTB | UXTB | SXTH | UXTH => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let mut rm = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            if let Some(rot) = imm(&ops[2]).filter(|&r| r != 0) {
                rm = bin(BinOp::Ror, rm, Value::Imm(rot));
            }
            let result = match inst.opcode {
                UXTB => bin(BinOp::And, rm, Value::Imm(0xFF)),
                UXTH => bin(BinOp::And, rm, Value::Imm(0xFFFF)),
                // Sign-extend by shifting the field to the top then arithmetic-
                // shifting back down.
                SXTB => bin(BinOp::Asr, bin(BinOp::Shl, rm, Value::Imm(24)), Value::Imm(24)),
                _ => bin(BinOp::Asr, bin(BinOp::Shl, rm, Value::Imm(16)), Value::Imm(16)),
            };
            out.push(Stmt::SetReg(rd, result));
        }
        // Extend-and-add: `Xtab rd, rn, rm{, ror #rot}` => rd = rn + extend(rm's
        // low byte/half), with the same optional pre-rotate (trailing Imm32 in
        // ops[3]).
        SXTAB | UXTAB | SXTAH | UXTAH => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rn = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            let mut rm = operand_value(&ops[2], pc_const).ok_or_else(err)?;
            if let Some(rot) = imm(&ops[3]).filter(|&r| r != 0) {
                rm = bin(BinOp::Ror, rm, Value::Imm(rot));
            }
            let ext = match inst.opcode {
                UXTAB => bin(BinOp::And, rm, Value::Imm(0xFF)),
                UXTAH => bin(BinOp::And, rm, Value::Imm(0xFFFF)),
                SXTAB => bin(BinOp::Asr, bin(BinOp::Shl, rm, Value::Imm(24)), Value::Imm(24)),
                _ => bin(BinOp::Asr, bin(BinOp::Shl, rm, Value::Imm(16)), Value::Imm(16)),
            };
            out.push(Stmt::SetReg(rd, bin(BinOp::Add, rn, ext)));
        }
        // Bitfield extract: ubfx rd, rn, #lsb, #width (zero-extended); sbfx
        // (sign-extended).
        SSAT | USAT => {
            // `ssat Rd, #n, Rm{, shift}` / `usat Rd, #n, Rm{, shift}`: clamp the (already
            // shifted) source into an n-bit signed or unsigned range. The source is read as
            // SIGNED in both cases - `usat` of a negative value is 0, not a huge unsigned.
            //
            // Emitted BRANCHLESS, because this IR has no select and no min/max, and because
            // the whole point of the instruction in a codec is that it is one cheap step in
            // an inner loop. The two identities used:
            //
            //   * "does it fit" is `sign_extend_n(x) == x`, and a difference `d` becomes an
            //     all-ones mask with `(d | -d) >> 31` (arithmetic), 0 exactly when d == 0.
            //   * the clamped value for `ssat` is `(x >> 31) ^ max`: 0 ^ max = max for a
            //     positive x, and ~max = min for a negative one. No compare needed.
            //
            // **The Q flag is NOT set.** ARM makes saturation sticky in APSR.Q, and this
            // engine models only N/Z/C/V, so a title that reads Q back (via `mrs`) would
            // see it clear. That is a real gap rather than a silent approximation of the
            // RESULT, which is exact; nothing observed so far reads Q, and modelling it
            // means a fifth flag global.
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let n = imm(&ops[1]).ok_or_else(err)?;
            let x = operand_value(&ops[2], pc_const).ok_or_else(err)?;
            // `(a | (0 - a)) >> 31` - all ones iff `a` is nonzero.
            let nonzero_mask = |a: Value| {
                bin(
                    BinOp::Asr,
                    bin(BinOp::Or, a.clone(), bin(BinOp::Sub, Value::Imm(0), a)),
                    Value::Imm(31),
                )
            };
            let result = if inst.opcode == SSAT {
                // n is 1..=32 for ssat; the range is [-(2^(n-1)), 2^(n-1) - 1].
                let spare = 32u32.saturating_sub(n);
                let max = if n >= 32 { i32::MAX as u32 } else { (1u32 << (n - 1)) - 1 };
                // sign_extend_n(x): up to the top and arithmetically back down.
                let sat = if spare == 0 {
                    x.clone()
                } else {
                    bin(
                        BinOp::Asr,
                        bin(BinOp::Shl, x.clone(), Value::Imm(spare)),
                        Value::Imm(spare),
                    )
                };
                let over = nonzero_mask(bin(BinOp::Xor, sat, x.clone()));
                let clamped = bin(BinOp::Xor, bin(BinOp::Asr, x.clone(), Value::Imm(31)), Value::Imm(max));
                bin(
                    BinOp::Or,
                    bin(BinOp::And, x, Value::Not(Box::new(over.clone()))),
                    bin(BinOp::And, clamped, over),
                )
            } else {
                // n is 0..=31 for usat; the range is [0, 2^n - 1]. A negative source clamps
                // to 0 and anything with a bit at or above n clamps to the maximum, so the
                // two conditions are independent and are applied as two masks.
                let max = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
                let neg = bin(BinOp::Asr, x.clone(), Value::Imm(31));
                let over = nonzero_mask(bin(BinOp::Lsr, x.clone(), Value::Imm(n.min(31))));
                let picked = bin(
                    BinOp::Or,
                    bin(BinOp::And, x, Value::Not(Box::new(over.clone()))),
                    bin(BinOp::And, Value::Imm(max), over),
                );
                bin(BinOp::And, picked, Value::Not(Box::new(neg)))
            };
            out.push(Stmt::SetReg(rd, result));
        }

        UBFX | SBFX => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rn = operand_value(&ops[1], pc_const).ok_or_else(err)?;
            let lsb = imm(&ops[2]).ok_or_else(err)?;
            let width = imm(&ops[3]).ok_or_else(err)?;
            let result = if inst.opcode == UBFX {
                let mask = if width >= 32 { !0 } else { (1u32 << width) - 1 };
                bin(BinOp::And, bin(BinOp::Lsr, rn, Value::Imm(lsb)), Value::Imm(mask))
            } else {
                // Shift the field to the top (bit 31 = msb of field), then
                // arithmetic-shift down by 32-width to sign-extend.
                let top = 32u32.saturating_sub(lsb + width);
                bin(BinOp::Asr, bin(BinOp::Shl, rn, Value::Imm(top)), Value::Imm(32 - width))
            };
            out.push(Stmt::SetReg(rd, result));
        }

        LDR | LDRB | LDRH | LDRSB | LDRSH => {
            let rt = regnum(&ops[0]).ok_or_else(err)?;
            let a = lower_addr(&ops[1], addr, inst.thumb).ok_or_else(err)?;
            let (size, signed) = load_kind(inst.opcode);
            out.extend(a.pre);
            out.push(Stmt::SetReg(
                rt,
                Value::Load { addr: Box::new(a.addr), size, signed },
            ));
            out.extend(a.post);
        }
        STR | STRB | STRH => {
            let rt = regnum(&ops[0]).ok_or_else(err)?;
            let a = lower_addr(&ops[1], addr, inst.thumb).ok_or_else(err)?;
            let size = store_size(inst.opcode);
            out.extend(a.pre);
            out.push(Stmt::Store { addr: a.addr, data: Value::Reg(rt), size });
            out.extend(a.post);
        }
        LDRD => {
            let rt = regnum(&ops[0]).ok_or_else(err)?;
            let rt2 = regnum(&ops[1]).ok_or_else(err)?;
            let a = lower_addr(&ops[2], addr, inst.thumb).ok_or_else(err)?;
            out.extend(a.pre);
            emit_load_pair(&mut out, rt, rt2, a.addr);
            out.extend(a.post);
        }
        STRD => {
            let rt = regnum(&ops[0]).ok_or_else(err)?;
            let rt2 = regnum(&ops[1]).ok_or_else(err)?;
            let a = lower_addr(&ops[2], addr, inst.thumb).ok_or_else(err)?;
            out.extend(a.pre);
            out.push(Stmt::Store {
                addr: a.addr.clone(),
                data: Value::Reg(rt),
                size: MemSize::Word,
            });
            out.push(Stmt::Store {
                addr: bin(BinOp::Add, a.addr, Value::Imm(4)),
                data: Value::Reg(rt2),
                size: MemSize::Word,
            });
            out.extend(a.post);
        }

        // Load-exclusive: with one guest CPU worker there is no contending core,
        // so an exclusive load is a plain load. (When preemptive multi-threading
        // lands, these need a real exclusive monitor; single-thread bring-up is
        // faithful as an ordinary load.)
        LDREX | LDREXB | LDREXH => {
            let rt = regnum(&ops[0]).ok_or_else(err)?;
            let a = lower_addr(&ops[1], addr, inst.thumb).ok_or_else(err)?;
            let size = match inst.opcode {
                LDREXB => MemSize::Byte,
                LDREXH => MemSize::Half,
                _ => MemSize::Word,
            };
            out.extend(a.pre);
            out.push(Stmt::SetReg(rt, Value::Load { addr: Box::new(a.addr), size, signed: false }));
            out.extend(a.post);
        }
        // Load-exclusive doubleword: two plain 32-bit loads (single guest core).
        LDREXD => {
            let rt = regnum(&ops[0]).ok_or_else(err)?;
            let rt2 = regnum(&ops[1]).ok_or_else(err)?;
            let a = lower_addr(&ops[2], addr, inst.thumb).ok_or_else(err)?;
            out.extend(a.pre);
            emit_load_pair(&mut out, rt, rt2, a.addr);
            out.extend(a.post);
        }
        // Store-exclusive doubleword: store both words, report success (0).
        STREXD => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rt = regnum(&ops[1]).ok_or_else(err)?;
            let rt2 = regnum(&ops[2]).ok_or_else(err)?;
            let a = lower_addr(&ops[3], addr, inst.thumb).ok_or_else(err)?;
            out.extend(a.pre);
            out.push(Stmt::Store { addr: a.addr.clone(), data: Value::Reg(rt), size: MemSize::Word });
            out.push(Stmt::Store {
                addr: bin(BinOp::Add, a.addr, Value::Imm(4)),
                data: Value::Reg(rt2),
                size: MemSize::Word,
            });
            out.push(Stmt::SetReg(rd, Value::Imm(0)));
            out.extend(a.post);
        }
        // Store-exclusive: the store always succeeds (no contention), so it writes
        // the value and reports success (0) in the status register.
        STREX | STREXB | STREXH => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rt = regnum(&ops[1]).ok_or_else(err)?;
            let a = lower_addr(&ops[2], addr, inst.thumb).ok_or_else(err)?;
            let size = match inst.opcode {
                STREXB => MemSize::Byte,
                STREXH => MemSize::Half,
                _ => MemSize::Word,
            };
            out.extend(a.pre);
            out.push(Stmt::Store { addr: a.addr, data: Value::Reg(rt), size });
            out.push(Stmt::SetReg(rd, Value::Imm(0)));
            out.extend(a.post);
        }

        PUSH => out.extend(lower_push(inst, addr)?),
        POP | LDM(..) => out.extend(lower_ldm(inst, addr)?),
        STM(..) => out.extend(lower_stm(inst, addr)?),

        // --- VFP / floating-point ---------------------------------------
        VADD | VSUB | VMUL | VDIV => {
            let op = match inst.opcode {
                VADD => FBinOp::Add,
                VSUB => FBinOp::Sub,
                VMUL => FBinOp::Mul,
                _ => FBinOp::Div,
            };
            if is_double(&ops[0]) {
                let rd = d_num(&ops[0]).ok_or_else(err)?;
                let rn = d_num(&ops[1]).ok_or_else(err)?;
                let rm = d_num(&ops[2]).ok_or_else(err)?;
                out.push(Stmt::Vfp(VfpOp::Bin64 { op, rd, rn, rm }));
            } else {
                let rd = s_num(&ops[0]).ok_or_else(err)?;
                let rn = s_num(&ops[1]).ok_or_else(err)?;
                let rm = s_num(&ops[2]).ok_or_else(err)?;
                out.push(Stmt::Vfp(VfpOp::Bin32 { op, rd, rn, rm }));
            }
        }
        // Multiply-accumulate family: vmla/vmls and their negated forms
        // vnmla/vnmls (which negate the accumulator first).
        VMLA | VMLS | VNMLA | VNMLS => {
            let sub = matches!(inst.opcode, VMLS | VNMLA);
            let neg = matches!(inst.opcode, VNMLA | VNMLS);
            if is_double(&ops[0]) {
                let rd = d_num(&ops[0]).ok_or_else(err)?;
                let rn = d_num(&ops[1]).ok_or_else(err)?;
                let rm = d_num(&ops[2]).ok_or_else(err)?;
                out.push(Stmt::Vfp(VfpOp::MulAcc64 { rd, rn, rm, sub, neg }));
            } else {
                let rd = s_num(&ops[0]).ok_or_else(err)?;
                let rn = s_num(&ops[1]).ok_or_else(err)?;
                let rm = s_num(&ops[2]).ok_or_else(err)?;
                out.push(Stmt::Vfp(VfpOp::MulAcc32 { rd, rn, rm, sub, neg }));
            }
        }
        VNMUL => {
            if is_double(&ops[0]) {
                let rd = d_num(&ops[0]).ok_or_else(err)?;
                let rn = d_num(&ops[1]).ok_or_else(err)?;
                let rm = d_num(&ops[2]).ok_or_else(err)?;
                out.push(Stmt::Vfp(VfpOp::NegMul64 { rd, rn, rm }));
            } else {
                let rd = s_num(&ops[0]).ok_or_else(err)?;
                let rn = s_num(&ops[1]).ok_or_else(err)?;
                let rm = s_num(&ops[2]).ok_or_else(err)?;
                out.push(Stmt::Vfp(VfpOp::NegMul32 { rd, rn, rm }));
            }
        }
        VNEG => {
            if is_double(&ops[0]) {
                let rd = d_num(&ops[0]).ok_or_else(err)?;
                let rm = d_num(&ops[1]).ok_or_else(err)?;
                out.push(Stmt::Vfp(VfpOp::Neg64 { rd, rm }));
            } else {
                let rd = s_num(&ops[0]).ok_or_else(err)?;
                let rm = s_num(&ops[1]).ok_or_else(err)?;
                out.push(Stmt::Vfp(VfpOp::Neg32 { rd, rm }));
            }
        }
        VABS => {
            if is_double(&ops[0]) {
                let rd = d_num(&ops[0]).ok_or_else(err)?;
                let rm = d_num(&ops[1]).ok_or_else(err)?;
                out.push(Stmt::Vfp(VfpOp::Abs64 { rd, rm }));
            } else {
                let rd = s_num(&ops[0]).ok_or_else(err)?;
                let rm = s_num(&ops[1]).ok_or_else(err)?;
                out.push(Stmt::Vfp(VfpOp::Abs32 { rd, rm }));
            }
        }
        VSQRT => {
            if is_double(&ops[0]) {
                let rd = d_num(&ops[0]).ok_or_else(err)?;
                let rm = d_num(&ops[1]).ok_or_else(err)?;
                out.push(Stmt::Vfp(VfpOp::Sqrt64 { rd, rm }));
            } else {
                let rd = s_num(&ops[0]).ok_or_else(err)?;
                let rm = s_num(&ops[1]).ok_or_else(err)?;
                out.push(Stmt::Vfp(VfpOp::Sqrt32 { rd, rm }));
            }
        }
        VMOV => {
            match (&ops[0], &ops[1]) {
                // Immediate move. A single-precision destination is the VFP
                // `vmov.f32 sd, #imm` (one 32-bit Imm); a D/Q destination is either
                // a VFP `vmov.f64` or a NEON `vmov.iN` modified-immediate, both a
                // 64-bit constant per D register (Imm low, Imm high).
                (Operand::SIMDReg(dst), Operand::Imm(lo)) => {
                    if dst.size == SIMDSizeCode::S {
                        out.push(Stmt::Vfp(VfpOp::SetImmS { s: dst.num, bits: *lo }));
                    } else {
                        let hi = match ops[2] {
                            Operand::Imm(h) => h,
                            _ => 0,
                        };
                        for d in simd_imm_dregs(dst) {
                            out.push(Stmt::Vfp(VfpOp::SetImmD { d, lo: *lo, hi }));
                        }
                    }
                }
                // Core <- scalar: `vmov Rt, Sn` (e.g. extracting a reduced NEON/VFP
                // lane into a GP register).
                (Operand::Reg(rt), Operand::SIMDReg(s)) if s.size == SIMDSizeCode::S => {
                    out.push(Stmt::Vfp(VfpOp::ScalarToCore { rt: rt.number(), s: s.num }));
                }
                // Scalar <- core: `vmov Sn, Rt`.
                (Operand::SIMDReg(s), Operand::Reg(rt)) if s.size == SIMDSizeCode::S => {
                    out.push(Stmt::Vfp(VfpOp::CoreToScalar { s: s.num, rt: rt.number() }));
                }
                // Double -> core pair: `vmov Rt, Rt2, Dm` (D low half -> rt, high -> rt2).
                (Operand::Reg(rt), Operand::Reg(rt2)) if d_num(&ops[2]).is_some() => {
                    let d = d_num(&ops[2]).ok_or_else(err)?;
                    out.push(Stmt::Vfp(VfpOp::DoubleToCore {
                        rt: rt.number(),
                        rt2: rt2.number(),
                        d,
                    }));
                }
                // Core pair -> double: `vmov Dm, Rt, Rt2`.
                (Operand::SIMDReg(dst), Operand::Reg(rt)) if dst.size == SIMDSizeCode::D => {
                    let rt2 = regnum(&ops[2]).ok_or_else(err)?;
                    out.push(Stmt::Vfp(VfpOp::CoreToDouble {
                        d: dst.num,
                        rt: rt.number(),
                        rt2,
                    }));
                }
                // Double register move (raw 64-bit copy): `vmov Dd, Dm`.
                (Operand::SIMDReg(dst), Operand::SIMDReg(src))
                    if dst.size == SIMDSizeCode::D && src.size == SIMDSizeCode::D =>
                {
                    out.push(Stmt::Vfp(VfpOp::Mov64 { rd: dst.num, rm: src.num }));
                }
                // Register move between two single-precision regs (raw bit copy).
                _ => {
                    let rd = s_num(&ops[0]).ok_or_else(err)?;
                    let rm = s_num(&ops[1]).ok_or_else(err)?;
                    out.push(Stmt::Vfp(VfpOp::Mov32 { rd, rm }));
                }
            }
        }
        VCMP(_) => {
            if is_double(&ops[0]) {
                let rn = d_num(&ops[0]).ok_or_else(err)?;
                let rm = match &ops[1] {
                    Operand::SIMDReg(r) if r.size == SIMDSizeCode::D => Some(r.num),
                    Operand::Imm(0) => None,
                    _ => return Err(err()),
                };
                out.push(Stmt::Vfp(VfpOp::Cmp64 { rn, rm }));
            } else {
                let rn = s_num(&ops[0]).ok_or_else(err)?;
                // Second operand is either a register or `#0.0` (Imm(0)).
                let rm = match &ops[1] {
                    Operand::SIMDReg(r) if r.size == SIMDSizeCode::S => Some(r.num),
                    Operand::Imm(0) => None,
                    _ => return Err(err()),
                };
                out.push(Stmt::Vfp(VfpOp::Cmp32 { rn, rm }));
            }
        }
        VMRS => {
            // `vmrs APSR_nzcv, fpscr` (Rt == 15) moves the FP flags into NZCV;
            // `vmrs Rt, fpscr` reads the FPSCR into a core register (we synthesize
            // it from the tracked NZCV flags, other bits zero - rounding/exception
            // state is not modeled, matching the fixed round-to-nearest engine).
            match regnum(&ops[0]) {
                Some(15) => out.push(Stmt::Vfp(VfpOp::MrsNzcv)),
                Some(rt) => out.push(Stmt::Vfp(VfpOp::MrsFpscr { rt })),
                None => return Err(Error::Unsupported { addr, opcode: inst.opcode }),
            }
        }
        // `vmsr fpscr, Rt`: the only observable FPSCR state we model is the NZCV
        // flags (set by vcmp, read back by vmrs), so writing rounding / flush-to-
        // zero / exception control is a no-op on the fixed-mode engine.
        VMSR => {}
        VCVT { to, from, .. } => {
            use VfpType::{F32, F64, S32, U32};
            match (to, from) {
                // Single precision: f32 <-> 32-bit integer (S registers).
                (S32, F32) => {
                    let (rd, rm) = (s_num(&ops[0]).ok_or_else(err)?, s_num(&ops[1]).ok_or_else(err)?);
                    out.push(Stmt::Vfp(VfpOp::CvtToInt { rd, rm, signed: true }));
                }
                (U32, F32) => {
                    let (rd, rm) = (s_num(&ops[0]).ok_or_else(err)?, s_num(&ops[1]).ok_or_else(err)?);
                    out.push(Stmt::Vfp(VfpOp::CvtToInt { rd, rm, signed: false }));
                }
                (F32, S32) => {
                    let (rd, rm) = (s_num(&ops[0]).ok_or_else(err)?, s_num(&ops[1]).ok_or_else(err)?);
                    out.push(Stmt::Vfp(VfpOp::CvtFromInt { rd, rm, signed: true }));
                }
                (F32, U32) => {
                    let (rd, rm) = (s_num(&ops[0]).ok_or_else(err)?, s_num(&ops[1]).ok_or_else(err)?);
                    out.push(Stmt::Vfp(VfpOp::CvtFromInt { rd, rm, signed: false }));
                }
                // Double precision: f64 (D) <-> 32-bit integer (S).
                (F64, S32) | (F64, U32) => {
                    let (d, s) = (d_num(&ops[0]).ok_or_else(err)?, s_num(&ops[1]).ok_or_else(err)?);
                    out.push(Stmt::Vfp(VfpOp::CvtF64FromInt { d, s, signed: from == S32 }));
                }
                (S32, F64) | (U32, F64) => {
                    let (s, d) = (s_num(&ops[0]).ok_or_else(err)?, d_num(&ops[1]).ok_or_else(err)?);
                    out.push(Stmt::Vfp(VfpOp::CvtIntFromF64 { s, d, signed: to == S32 }));
                }
                // Single <-> double width conversion.
                (F64, F32) => {
                    let (d, s) = (d_num(&ops[0]).ok_or_else(err)?, s_num(&ops[1]).ok_or_else(err)?);
                    out.push(Stmt::Vfp(VfpOp::CvtF64FromF32 { d, s }));
                }
                (F32, F64) => {
                    let (s, d) = (s_num(&ops[0]).ok_or_else(err)?, d_num(&ops[1]).ok_or_else(err)?);
                    out.push(Stmt::Vfp(VfpOp::CvtF32FromF64 { s, d }));
                }
                _ => return Err(Error::Unsupported { addr, opcode: inst.opcode }),
            }
        }
        VCVTHalf { to_half, top } => {
            let (sd, sm) = (s_num(&ops[0]).ok_or_else(err)?, s_num(&ops[1]).ok_or_else(err)?);
            if to_half {
                // f32 -> f16 (round-to-nearest-even) is a separate, more involved
                // conversion; defer it to a safe trapping stub until needed.
                return Err(Error::Unsupported { addr, opcode: inst.opcode });
            }
            out.push(Stmt::Vfp(VfpOp::CvtF32FromHalf { sd, sm, top }));
        }
        VLDR | VSTR => {
            let (size, num) = simd(&ops[0]).ok_or_else(err)?;
            let a = vfp_single_addr(&ops[1], addr, inst.thumb).ok_or_else(err)?;
            let reg = vfp_reg(size, num).ok_or_else(err)?;
            out.push(Stmt::VfpMem { reg, addr: a, load: inst.opcode == VLDR });
        }
        VLDM(inc) | VSTM(inc) => {
            let base = ldm_base(inst);
            let wback = ldm_wback(inst);
            let (size, num, count) = simd_list(inst).ok_or_else(err)?;
            let load = matches!(inst.opcode, VLDM(_));
            out.extend(vfp_multi(size, num, count, base, !inc, wback, None, load)?);
        }
        VPUSH => {
            let (size, num, count) = simd_list(inst).ok_or_else(err)?;
            out.extend(vfp_multi(size, num, count, 13, true, true, None, false)?);
        }
        VPOP => {
            let (size, num, count) = simd_list(inst).ok_or_else(err)?;
            out.extend(vfp_multi(size, num, count, 13, false, true, None, true)?);
        }
        VLDN(n, dt) | VSTN(n, dt) => {
            let load = matches!(inst.opcode, VLDN(..));
            let (first, count, stride, element) = simd_points(inst).ok_or_else(err)?;
            let (base, wback, post) = simd_deref(inst).ok_or_else(err)?;
            let esize = neon_type(dt).bits;
            out.extend(neon_structure_mem(
                n, first, count, stride, element, esize, base, wback, post, load, addr,
            )?);
        }
        // vdup.N broadcast: from a core register (`vdup Qd/Dd, Rt`) or from one lane
        // of a D source (the scalar form `vdup Qd/Dd, Dm[x]`), told apart by ops[1].
        VDUP(dt) => {
            let dst = neon_reg(&ops[0]).ok_or_else(err)?;
            match &ops[1] {
                Operand::SIMDRegLane(src, lane) if src.size == SIMDSizeCode::D => {
                    out.push(Stmt::Neon(NeonStmt::DupLane {
                        esize: neon_type(dt).bits,
                        dst,
                        src: src.num,
                        lane: *lane,
                    }));
                }
                _ => {
                    let rt = regnum(&ops[1]).ok_or_else(err)?;
                    out.push(Stmt::Neon(NeonStmt::DupCore { ty: neon_type(dt), dst, rt }));
                }
            }
        }
        // vmov between a core register and one lane of a D register, in either
        // direction, told apart by which operand is the `Dn[x]` lane. Lane->core
        // (`vmov Rt, Dn[x]`) has ops = [Reg, SIMDRegLane] and sign/zero-extends the
        // 8/16-bit sub-word per `dt`; core->lane (`vmov Dn[x], Rt`) has ops =
        // [SIMDRegLane, Reg] and `dt` is always `Any(size)`.
        VMOVLane(dt) => {
            let nt = neon_type(dt);
            match (&ops[0], &ops[1]) {
                (Operand::Reg(rt), Operand::SIMDRegLane(src, lane)) if src.size == SIMDSizeCode::D => {
                    out.push(Stmt::Neon(NeonStmt::MovLane {
                        to_core: true,
                        bits: nt.bits,
                        signed: nt.signed,
                        dreg: src.num,
                        lane: *lane,
                        rt: rt.number(),
                    }));
                }
                (Operand::SIMDRegLane(dst, lane), Operand::Reg(rt)) if dst.size == SIMDSizeCode::D => {
                    out.push(Stmt::Neon(NeonStmt::MovLane {
                        to_core: false,
                        bits: nt.bits,
                        signed: false,
                        dreg: dst.num,
                        lane: *lane,
                        rt: rt.number(),
                    }));
                }
                _ => return Err(Error::Unsupported { addr, opcode: inst.opcode }),
            }
        }
        Neon { op, dt } => {
            out.push(Stmt::Neon(lower_neon(op, dt, ops).ok_or_else(err)?));
        }

        // `uadd16 Rd, Rn, Rm`: independent unsigned add of the two 16-bit halfwords
        // (each wraps modulo 2^16). Rd[15:0] = Rn[15:0]+Rm[15:0], Rd[31:16] =
        // Rn[31:16]+Rm[31:16]. The APSR.GE bits it also sets are not modeled (nothing
        // in the engine reads them - there is no `sel`), matching how GE is elided
        // everywhere else.
        UADD16 => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rn = Value::Reg(regnum(&ops[1]).ok_or_else(err)?);
            let rm = Value::Reg(regnum(&ops[2]).ok_or_else(err)?);
            let bin = |op, a, b| Value::Bin(op, Box::new(a), Box::new(b));
            let mask = |v| bin(BinOp::And, v, Value::Imm(0xffff));
            // low halfword sum, truncated to 16 bits.
            let lo = mask(bin(BinOp::Add, mask(rn.clone()), mask(rm.clone())));
            // high halfword sum: shift each source down, add, truncate, shift back up.
            let hi_sum = mask(bin(
                BinOp::Add,
                bin(BinOp::Lsr, rn, Value::Imm(16)),
                bin(BinOp::Lsr, rm, Value::Imm(16)),
            ));
            let hi = bin(BinOp::Shl, hi_sum, Value::Imm(16));
            out.push(Stmt::SetReg(rd, bin(BinOp::Or, lo, hi)));
        }

        _ => return Err(Error::Unsupported { addr, opcode: inst.opcode }),
    }
    Ok(out)
}

fn load_kind(op: Opcode) -> (MemSize, bool) {
    match op {
        Opcode::LDRB => (MemSize::Byte, false),
        Opcode::LDRSB => (MemSize::Byte, true),
        Opcode::LDRH => (MemSize::Half, false),
        Opcode::LDRSH => (MemSize::Half, true),
        _ => (MemSize::Word, false),
    }
}

fn store_size(op: Opcode) -> MemSize {
    match op {
        Opcode::STRB => MemSize::Byte,
        Opcode::STRH => MemSize::Half,
        _ => MemSize::Word,
    }
}

/// Carry-out of the modified-immediate expansion (`ThumbExpandImm_C` in Thumb,
/// `ARMExpandImm_C` in ARM) for a flag-setting logical op with an immediate operand.
/// ARM leaves C unchanged when the value fits a zero-extended byte (rotation 0) and
/// otherwise sets C = result<31>; Thumb additionally leaves C unchanged for the three
/// byte-replication forms (0x00XY00XY, 0xXY00XY00, 0xXYXYXYXY). Because the rotate form
/// yields the value verbatim, its carry-out is just bit 31 of the expanded immediate.
/// Returns `None` when the encoding leaves C unchanged, `Some(bit31)` otherwise.
fn modified_imm_carry(value: u32, thumb: bool) -> Option<Value> {
    if value <= 0xff {
        return None; // zero-extended byte (Thumb 00-form / ARM rotation 0): C unchanged
    }
    if thumb {
        let b = value & 0xff;
        if value == (b << 16) | b            // 0x00XY00XY
            || value == (b << 24) | (b << 8) // 0xXY00XY00
            || value == (b << 24) | (b << 16) | (b << 8) | b // 0xXYXYXYXY
        {
            return None;
        }
    }
    Some(Value::Imm(value >> 31))
}

/// The shifter carry-out of a shifted-register operand with an immediate amount
/// (`rm, lsl #n` and friends); `None` when the amount is a register or the shift
/// leaves C unchanged (`lsl #0`).
fn reg_shift_carry(rs: &RegShift) -> Option<Value> {
    let RegShiftStyle::RegImm(s) = rs.into_shift() else { return None };
    let base = Value::Reg(s.shiftee().number());
    let n = s.imm() as u32;
    match s.stype() {
        ShiftStyle::LSL => shift_carry(Opcode::LSL, &base, &Value::Imm(n)),
        ShiftStyle::LSR => shift_carry(Opcode::LSR, &base, &Value::Imm(n)),
        ShiftStyle::ASR => shift_carry(Opcode::ASR, &base, &Value::Imm(n)),
        // `ror #0` is `rrx`: C takes the bit shifted out, bit 0. `ror #n` carries the
        // result's top bit, which is bit `n-1` of the source.
        ShiftStyle::ROR => {
            let k = if n == 0 { 0 } else { n - 1 };
            Some(bin(BinOp::And, bin(BinOp::Lsr, base, Value::Imm(k)), Value::Imm(1)))
        }
    }
}

/// The shifter carry-out for a flag-setting immediate shift.
fn shift_carry(op: Opcode, rn: &Value, sh: &Value) -> Option<Value> {
    // Only the immediate-shift form has a statically simple carry; register
    // shifts are left to a later, exact model. Return None (C unchanged) if the
    // amount is not a constant.
    let n = match sh {
        Value::Imm(v) => *v & 0xFF,
        _ => return None,
    };
    match op {
        Opcode::LSL if n == 0 => None,
        Opcode::LSL => Some(bin(BinOp::And, bin(BinOp::Lsr, rn.clone(), Value::Imm(32 - n)), Value::Imm(1))),
        Opcode::LSR => {
            let n = if n == 0 { 32 } else { n };
            Some(bin(BinOp::And, bin(BinOp::Lsr, rn.clone(), Value::Imm(n - 1)), Value::Imm(1)))
        }
        Opcode::ASR => {
            let n = if n == 0 { 32 } else { n };
            Some(bin(BinOp::And, bin(BinOp::Asr, rn.clone(), Value::Imm(n - 1)), Value::Imm(1)))
        }
        _ => None,
    }
}

/// The registers named by an ldm/stm/push/pop register list, ascending.
fn reglist(inst: &Instruction) -> Vec<u8> {
    for op in &inst.operands {
        if let Operand::RegList(mask) = op {
            return (0..16u8).filter(|i| mask & (1 << i) != 0).collect();
        }
    }
    Vec::new()
}

/// `push {list}`: sp -= 4*n; store each register at ascending addresses.
fn lower_push(inst: &Instruction, _addr: u32) -> Result<Vec<Stmt>, Error> {
    let regs = reglist(inst);
    let n = regs.len() as u32;
    let mut out = vec![Stmt::SetReg(
        13,
        bin(BinOp::Sub, Value::Reg(13), Value::Imm(4 * n)),
    )];
    for (i, r) in regs.iter().enumerate() {
        out.push(Stmt::Store {
            addr: bin(BinOp::Add, Value::Reg(13), Value::Imm(4 * i as u32)),
            data: Value::Reg(*r),
            size: MemSize::Word,
        });
    }
    Ok(out)
}

/// The address of the FIRST (lowest-numbered) register's slot, relative to the base,
/// for a block transfer of `n` registers in mode `(add, pre)` - the U and P bits.
///
/// All four ARM modes put the lowest-numbered register at the lowest address; they
/// differ only in where the block sits relative to the base:
///
/// | mode | add | pre | first slot   | writeback |
/// |------|-----|-----|--------------|-----------|
/// | IA   | 1   | 0   | base         | base + 4n |
/// | IB   | 1   | 1   | base + 4     | base + 4n |
/// | DA   | 0   | 0   | base - 4n + 4| base - 4n |
/// | DB   | 0   | 1   | base - 4n    | base - 4n |
///
/// Ignoring these bits (treating everything as IA) is not a rounding error: a
/// `stmdb rN!, {..}` then writes its block ABOVE the base instead of below it,
/// scribbling over whatever lives there - a caller's saved registers, typically -
/// and moves the base the wrong way afterwards.
fn block_first_offset(add: bool, pre: bool, n: u32) -> i32 {
    match (add, pre) {
        (true, false) => 0,
        (true, true) => 4,
        (false, false) => -(4 * n as i32) + 4,
        (false, true) => -(4 * n as i32),
    }
}

/// The base's post-transfer value for a writeback block transfer of `n` registers.
fn block_writeback(base: u8, add: bool, n: u32) -> Stmt {
    let delta = Value::Imm(4 * n);
    let op = if add { BinOp::Add } else { BinOp::Sub };
    Stmt::SetReg(base, bin(op, Value::Reg(base), delta))
}

/// Address of the `i`-th register's slot: `base + first + 4i`, folded into an
/// unsigned add because guest addresses wrap modulo 2^32.
fn block_slot_addr(base: u8, first: i32, i: usize) -> Value {
    let off = (first + 4 * i as i32) as u32;
    bin(BinOp::Add, Value::Reg(base), Value::Imm(off))
}

/// The `(add, pre)` addressing mode of a block transfer. `push`/`pop` are the
/// dedicated `stmdb sp!` / `ldmia sp!` forms and carry no flags of their own:
/// `pop` is increment-after, and `push` is lowered separately by [`lower_push`].
fn ldm_mode(inst: &Instruction) -> (bool, bool) {
    match inst.opcode {
        Opcode::POP | Opcode::PUSH => (true, false),
        Opcode::LDM(add, pre, _, _) | Opcode::STM(add, pre, _, _) => (add, pre),
        // Not a block transfer; the caller only reaches here for one, so keep the
        // common mode rather than inventing a failure path.
        _ => (true, false),
    }
}

/// `pop {list}` / `ldm{ia,ib,da,db} rn(!), {list}`: load each register from its
/// slot, then writeback. When the list has pc, the pc slot is consumed (sp advances)
/// but not written - the caller turns that into a return.
fn lower_ldm(inst: &Instruction, _addr: u32) -> Result<Vec<Stmt>, Error> {
    let regs = reglist(inst);
    let base = ldm_base(inst);
    let wback = ldm_wback(inst);
    let (add, pre) = ldm_mode(inst);
    let first = block_first_offset(add, pre, regs.len() as u32);
    // If the base register is itself in the list, ARM loads every word from the
    // ORIGINAL base and the loaded value wins (writeback is then not permitted). A
    // naive in-order lowering would overwrite the base with an early word and then
    // address the remaining words off that garbage (underflowing linear memory ->
    // MemoryOutOfBounds). Defer the base's own load to last so every other element's
    // address still sees the original base.
    let base_in_list = regs.iter().any(|&r| r == base);
    let mut out = Vec::new();
    let load_at = |i: usize, r: u8| {
        Stmt::SetReg(
            r,
            Value::Load {
                addr: Box::new(block_slot_addr(base, first, i)),
                size: MemSize::Word,
                signed: false,
            },
        )
    };
    for (i, r) in regs.iter().enumerate() {
        if *r == 15 {
            continue; // pc -> return, handled by terminator
        }
        if base_in_list && *r == base {
            continue; // deferred below so it does not clobber the base early
        }
        out.push(load_at(i, *r));
    }
    if base_in_list {
        let i = regs.iter().position(|&r| r == base).unwrap();
        out.push(load_at(i, base));
    } else if wback {
        // Writeback only when the base is not loaded (base-in-list forbids it).
        out.push(block_writeback(base, add, regs.len() as u32));
    }
    Ok(out)
}

/// True if evaluating `v` reads guest register `r` (so writing `r` first would
/// change `v`). Used to order multi-register loads so the base register, when it is
/// also a destination, is written last.
fn value_uses_reg(v: &Value, r: u8) -> bool {
    match v {
        Value::Reg(x) => *x == r,
        Value::Imm(_) | Value::Flag(_) | Value::CarryAddResult | Value::ThreadPtr => false,
        Value::Not(a) | Value::Clz(a) => value_uses_reg(a, r),
        Value::Bin(_, a, b) => value_uses_reg(a, r) || value_uses_reg(b, r),
        Value::Load { addr, .. } => value_uses_reg(addr, r),
    }
}

/// Emit a two-word load: `rt` <- [addr], `rt2` <- [addr + 4]. When the address's
/// base register is `rt` (e.g. `ldrd r6, r7, [r6, #8]`), the base is written last so
/// both element addresses are computed from the original base. ARM defines a
/// no-writeback doubleword load with the base among the destinations to read both
/// words from the original base; writing `rt` first would clobber it and make the
/// second address garbage (a latent MemoryOutOfBounds).
fn emit_load_pair(out: &mut Vec<Stmt>, rt: u8, rt2: u8, addr: Value) {
    let lo = Value::Load { addr: Box::new(addr.clone()), size: MemSize::Word, signed: false };
    let hi = Value::Load {
        addr: Box::new(bin(BinOp::Add, addr.clone(), Value::Imm(4))),
        size: MemSize::Word,
        signed: false,
    };
    if rt != rt2 && value_uses_reg(&addr, rt) {
        out.push(Stmt::SetReg(rt2, hi));
        out.push(Stmt::SetReg(rt, lo));
    } else {
        out.push(Stmt::SetReg(rt, lo));
        out.push(Stmt::SetReg(rt2, hi));
    }
}

/// `stm{ia,ib,da,db} rn(!), {list}`: store each register into its slot, then
/// writeback. See [`block_first_offset`] for why the U/P bits are load-bearing.
fn lower_stm(inst: &Instruction, _addr: u32) -> Result<Vec<Stmt>, Error> {
    let regs = reglist(inst);
    let base = ldm_base(inst);
    let wback = ldm_wback(inst);
    let (add, pre) = ldm_mode(inst);
    let first = block_first_offset(add, pre, regs.len() as u32);
    let mut out = Vec::new();
    for (i, r) in regs.iter().enumerate() {
        out.push(Stmt::Store {
            addr: block_slot_addr(base, first, i),
            data: Value::Reg(*r),
            size: MemSize::Word,
        });
    }
    if wback {
        out.push(block_writeback(base, add, regs.len() as u32));
    }
    Ok(out)
}

/// The base register of an ldm/stm; `pop`/`push` are implicitly sp (r13).
fn ldm_base(inst: &Instruction) -> u8 {
    match inst.opcode {
        Opcode::POP | Opcode::PUSH => 13,
        _ => inst
            .operands
            .iter()
            .find_map(|o| match o {
                Operand::RegWBack(r, _) => Some(r.number()),
                _ => None,
            })
            .unwrap_or(13),
    }
}

/// Whether an ldm/stm writes back its base; `pop`/`push` always do (sp).
fn ldm_wback(inst: &Instruction) -> bool {
    match inst.opcode {
        Opcode::POP | Opcode::PUSH => true,
        _ => inst.operands.iter().any(|o| matches!(o, Operand::RegWBack(_, true))),
    }
}

// --- VFP operand helpers --------------------------------------------------

/// A SIMD register operand as `(width, number)`.
fn simd(op: &Operand) -> Option<(SIMDSizeCode, u8)> {
    match op {
        Operand::SIMDReg(r) => Some((r.size, r.num)),
        _ => None,
    }
}

/// A single-precision (`S`) register operand's number.
fn s_num(op: &Operand) -> Option<u8> {
    match op {
        Operand::SIMDReg(r) if r.size == SIMDSizeCode::S => Some(r.num),
        _ => None,
    }
}

/// A double-precision (`D`) register operand's number.
fn d_num(op: &Operand) -> Option<u8> {
    match op {
        Operand::SIMDReg(r) if r.size == SIMDSizeCode::D => Some(r.num),
        _ => None,
    }
}

/// True if the VFP data-processing operand at `ops[0]` is a double-precision (`D`)
/// register - the discriminator between the f32 and f64 forms of vadd/vmul/vcmp/...
fn is_double(op: &Operand) -> bool {
    matches!(op, Operand::SIMDReg(r) if r.size == SIMDSizeCode::D)
}

/// The constituent double-register numbers of a VFP register targeted by a NEON
/// modified-immediate: a `Q` register is two consecutive `D`s; a `D` is itself.
fn simd_imm_dregs(r: &yaxpeax_arm::armv7::SIMDReg) -> Vec<u8> {
    match r.size {
        SIMDSizeCode::Q => vec![r.num * 2, r.num * 2 + 1],
        _ => vec![r.num],
    }
}

/// Build the IR register reference for a `(width, number)` VFP register.
fn vfp_reg(size: SIMDSizeCode, num: u8) -> Option<VfpReg> {
    match size {
        SIMDSizeCode::S => Some(VfpReg::S(num)),
        SIMDSizeCode::D => Some(VfpReg::D(num)),
        SIMDSizeCode::Q => None,
    }
}

/// Bytes occupied by one register of the given width in a memory transfer.
fn reg_bytes(size: SIMDSizeCode) -> u32 {
    match size {
        SIMDSizeCode::S => 4,
        SIMDSizeCode::D => 8,
        SIMDSizeCode::Q => 16,
    }
}

/// The `{first, ...}` register list of a vldm/vstm/vpush/vpop/vld1/vst1 as
/// `(width, first number, count)`.
fn simd_list(inst: &Instruction) -> Option<(SIMDSizeCode, u8, u8)> {
    inst.operands.iter().find_map(|op| match op {
        Operand::SIMDRegList(r, count) => Some((r.size, r.num, *count)),
        _ => None,
    })
}

/// The NEON structure/element register list of a vldN/vstN as
/// `(first D-register number, register count, register stride, element addressing)`.
fn simd_points(inst: &Instruction) -> Option<(u8, u8, u8, SIMDElement)> {
    inst.operands.iter().find_map(|op| match op {
        Operand::SIMDRegPoints { first, count, stride, element } => {
            Some((first.num, *count, *stride, *element))
        }
        _ => None,
    })
}

/// Lower a NEON structure/element load or store (`vld1`-`vld4`/`vst1`-`vst4`).
///
/// The one-structure forms (`vld1`/`vst1`, `n == 1`) lower directly and correctly:
/// a whole-register list is a contiguous D-register memory transfer; a single-lane
/// form is one masked element transfer into/out of a lane; an all-lanes form
/// broadcasts one loaded element across every lane of each destination register.
/// The multi-structure forms (`vld2`-`vld4`, `n >= 2`) deinterleave across registers
/// and are deferred: returning `Unsupported` makes the containing function a trapping
/// stub (a loud, safe failure) rather than risking a silent mis-deinterleave. The
/// decode is capstone-certified, so these never mis-decode - only await their lift.
#[allow(clippy::too_many_arguments)]
fn neon_structure_mem(
    n: u8,
    first: u8,
    count: u8,
    stride: u8,
    element: SIMDElement,
    esize: u8,
    base: u8,
    wback: bool,
    post: Option<u8>,
    load: bool,
    addr: u32,
) -> Result<Vec<Stmt>, Error> {
    let unsupported = || Error::Unsupported { addr, opcode: Opcode::VLDN(n, SIMDDataType::Any(esize)) };
    let eb = (esize / 8) as u32;
    let at = |off: u32| {
        if off == 0 { Value::Reg(base) } else { bin(BinOp::Add, Value::Reg(base), Value::Imm(off)) }
    };
    if n >= 2 {
        // vld2/vld3/vld4 and vst2/vst3/vst4: `n` interleaved structures. Every form
        // is a fixed set of single-element transfers at known offsets, so it lowers
        // to `ElemMem` lane moves - exact, if not fast (these run in image decoders
        // and format converters, not per draw).
        let mut out = Vec::new();
        match element {
            SIMDElement::Whole => {
                // Multiple structures: `count` registers, in `passes` groups of `n`.
                // Register `k` of structure element `e` in pass `r` is
                // `first + k*inc + r` (ARM ARM VLDn multiple: inc is 2 for the
                // four-register vld2 and the `{d0,d2,...}` lists, else 1); lane `e`
                // of it comes from memory element `(r*lanes + e)*n + k`.
                if count % n != 0 || esize == 0 || 64 % esize as u32 != 0 {
                    return Err(unsupported());
                }
                let passes = count / n;
                let inc = if passes > 1 { passes } else { stride };
                let lanes = 64 / esize;
                for r in 0..passes {
                    for e in 0..lanes {
                        for k in 0..n {
                            let d = first + k * inc + r;
                            let off = ((r as u32 * lanes as u32 + e as u32) * n as u32 + k as u32) * eb;
                            out.push(Stmt::Neon(NeonStmt::ElemMem {
                                d,
                                esize,
                                lane: ElemLane::One(e),
                                addr: at(off),
                                load,
                            }));
                        }
                    }
                }
                if let Some(rm) = post {
                    out.push(Stmt::SetReg(base, bin(BinOp::Add, Value::Reg(base), Value::Reg(rm))));
                } else if wback {
                    out.push(Stmt::SetReg(
                        base,
                        bin(BinOp::Add, Value::Reg(base), Value::Imm(count as u32 * 8)),
                    ));
                }
            }
            SIMDElement::Lane(idx) => {
                for k in 0..n {
                    out.push(Stmt::Neon(NeonStmt::ElemMem {
                        d: first + k * stride,
                        esize,
                        lane: ElemLane::One(idx),
                        addr: at(k as u32 * eb),
                        load,
                    }));
                }
                out.extend(elem_writeback(base, (n as u32 * eb) as u8, wback, post));
            }
            SIMDElement::AllLanes => {
                if !load {
                    return Err(unsupported());
                }
                for k in 0..n {
                    out.push(Stmt::Neon(NeonStmt::ElemMem {
                        d: first + k * stride,
                        esize,
                        lane: ElemLane::All,
                        addr: at(k as u32 * eb),
                        load: true,
                    }));
                }
                out.extend(elem_writeback(base, (n as u32 * eb) as u8, wback, post));
            }
        }
        return Ok(out);
    }
    if stride != 1 {
        return Err(unsupported());
    }
    match element {
        SIMDElement::Whole => {
            // vld1/vst1 multiple: `count` contiguous doubleword transfers.
            vfp_multi(SIMDSizeCode::D, first, count, base, false, wback, post, load)
        }
        SIMDElement::Lane(idx) => {
            // vld1/vst1 single element to one lane: one masked element transfer.
            let mut out = vec![Stmt::Neon(NeonStmt::ElemMem {
                d: first,
                esize,
                lane: ElemLane::One(idx),
                addr: Value::Reg(base),
                load,
            })];
            out.extend(elem_writeback(base, esize / 8, wback, post));
            Ok(out)
        }
        SIMDElement::AllLanes => {
            // vld1 single element to all lanes: broadcast the one loaded element into
            // every lane of each destination register (store-to-all-lanes has no
            // encoding, so this is load-only).
            if !load {
                return Err(unsupported());
            }
            let mut out = Vec::new();
            for i in 0..count {
                out.push(Stmt::Neon(NeonStmt::ElemMem {
                    d: first + i,
                    esize,
                    lane: ElemLane::All,
                    addr: Value::Reg(base),
                    load: true,
                }));
            }
            out.extend(elem_writeback(base, esize / 8, wback, post));
            Ok(out)
        }
    }
}

/// The post-transfer base update for a single-element NEON load/store: `bytes` is
/// the number of bytes the transfer moved (one element). `!` adds that constant;
/// the `, Rm` form adds a register.
fn elem_writeback(base: u8, bytes: u8, wback: bool, post: Option<u8>) -> Vec<Stmt> {
    if let Some(rm) = post {
        vec![Stmt::SetReg(base, bin(BinOp::Add, Value::Reg(base), Value::Reg(rm)))]
    } else if wback {
        vec![Stmt::SetReg(base, bin(BinOp::Add, Value::Reg(base), Value::Imm(bytes as u32)))]
    } else {
        Vec::new()
    }
}

/// The `[Rn]{!}` / `[Rn], Rm` addressing of a vld1/vst1 as
/// `(base, writeback, post-index register)`.
fn simd_deref(inst: &Instruction) -> Option<(u8, bool, Option<u8>)> {
    inst.operands.iter().find_map(|op| match op {
        Operand::SIMDDeref { base, wback, post, .. } => {
            Some((base.number(), *wback, post.map(|r| r.number())))
        }
        _ => None,
    })
}

// --- NEON data-processing lowering ----------------------------------------

/// The IR register reference for a NEON `Q`/`D` register operand (an `S` operand is
/// not a NEON data-processing operand and yields `None`).
fn neon_reg(op: &Operand) -> Option<NeonReg> {
    match op {
        Operand::SIMDReg(r) => match r.size {
            SIMDSizeCode::Q => Some(NeonReg::Q(r.num)),
            SIMDSizeCode::D => Some(NeonReg::D(r.num)),
            SIMDSizeCode::S => None,
        },
        _ => None,
    }
}

/// The element data type of a NEON operation (size, signedness, float-ness).
fn neon_type(dt: SIMDDataType) -> NeonType {
    match dt {
        SIMDDataType::Signed(b) => NeonType { bits: b, signed: true, float: false },
        SIMDDataType::Unsigned(b) => NeonType { bits: b, signed: false, float: false },
        SIMDDataType::Int(b) => NeonType { bits: b, signed: false, float: false },
        SIMDDataType::Float(b) => NeonType { bits: b, signed: false, float: true },
        SIMDDataType::Any(b) => NeonType { bits: b, signed: false, float: false },
        SIMDDataType::Poly(b) => NeonType { bits: b, signed: false, float: false },
    }
}

/// Lower a decoded NEON data-processing instruction to a [`NeonStmt`]. Returns
/// `None` (surfaced as an `Unsupported` error) for operand shapes the emitter
/// cannot map. The operand order matches the ARM encoding: `[dst, a, b]` (or
/// `[dst, a]` / `[dst, #imm]`).
fn lower_neon(op: NeonOp, dt: SIMDDataType, ops: &[Operand]) -> Option<NeonStmt> {
    let ty = neon_type(dt);
    let r = |i: usize| neon_reg(&ops[i]);
    use NeonOp::*;
    let st = match op {
        VADD => NeonStmt::Bin { op: NeonBin::Add, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VSUB => NeonStmt::Bin { op: NeonBin::Sub, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        // The multiply family has a "by scalar" form where the second factor is one broadcast lane
        // `Dm[x]` (ops[2] is a `SIMDRegLane`); tell it apart from the 3-same register form here.
        VMUL => match scalar_lane(&ops[2]) {
            Some((src, lane)) => NeonStmt::MulScalar { ty, dst: r(0)?, a: r(1)?, src, lane, acc: false, sub: false },
            None => NeonStmt::Bin { op: NeonBin::Mul, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        },
        VQADD => NeonStmt::Bin { op: NeonBin::QAdd, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VQSUB => NeonStmt::Bin { op: NeonBin::QSub, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VHADD => NeonStmt::Bin { op: NeonBin::HAdd, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VHSUB => NeonStmt::Bin { op: NeonBin::HSub, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VRHADD => NeonStmt::Bin { op: NeonBin::RHAdd, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VMVN => NeonStmt::Not { dst: r(0)?, src: r(1)? },
        VQABS => NeonStmt::SatAbsNeg { ty, neg: false, dst: r(0)?, src: r(1)? },
        VQNEG => NeonStmt::SatAbsNeg { ty, neg: true, dst: r(0)?, src: r(1)? },
        VMAX => NeonStmt::Bin { op: NeonBin::Max, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VMIN => NeonStmt::Bin { op: NeonBin::Min, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VABD => NeonStmt::Bin { op: NeonBin::Abd, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VMLA => match scalar_lane(&ops[2]) {
            Some((src, lane)) => NeonStmt::MulScalar { ty, dst: r(0)?, a: r(1)?, src, lane, acc: true, sub: false },
            None => NeonStmt::MulAcc { ty, dst: r(0)?, a: r(1)?, b: r(2)?, sub: false },
        },
        VMLS => match scalar_lane(&ops[2]) {
            Some((src, lane)) => NeonStmt::MulScalar { ty, dst: r(0)?, a: r(1)?, src, lane, acc: true, sub: true },
            None => NeonStmt::MulAcc { ty, dst: r(0)?, a: r(1)?, b: r(2)?, sub: true },
        },
        VPADD => NeonStmt::PairAdd { ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VMOVL => NeonStmt::Widen { ty, dst: r(0)?, a: r(1)? },
        VADDL => NeonStmt::WideAddSub { sub: false, wide: false, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VADDW => NeonStmt::WideAddSub { sub: false, wide: true, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VSUBL => NeonStmt::WideAddSub { sub: true, wide: false, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VSUBW => NeonStmt::WideAddSub { sub: true, wide: true, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VMULL => NeonStmt::WideMul { acc: false, sub: false, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VMLAL => NeonStmt::WideMul { acc: true, sub: false, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VMLSL => NeonStmt::WideMul { acc: true, sub: true, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VABDL => NeonStmt::WideAbd { acc: false, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VABAL => NeonStmt::WideAbd { acc: true, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VPADDL => NeonStmt::PairLong { acc: false, ty, dst: r(0)?, a: r(1)? },
        VPADAL => NeonStmt::PairLong { acc: true, ty, dst: r(0)?, a: r(1)? },
        VABS => NeonStmt::Unary { neg: false, ty, dst: r(0)?, a: r(1)? },
        VNEG => NeonStmt::Unary { neg: true, ty, dst: r(0)?, a: r(1)? },
        VMOVI => {
            // `vmov.i64` carries its 64-bit-per-lane value as low/high immediates; the
            // narrower `vmov.iN` forms carry a single per-element value in ops[1].
            if ty.bits == 64 {
                let (lo, hi) = match (&ops[1], &ops[2]) {
                    (Operand::Imm(lo), Operand::Imm(hi)) => (*lo, *hi),
                    _ => return None,
                };
                NeonStmt::MovImm64 { dst: r(0)?, val: (lo as u64) | ((hi as u64) << 32) }
            } else {
                let imm = match ops[1] {
                    Operand::Imm(v) => v,
                    _ => return None,
                };
                NeonStmt::MovImm { ty, dst: r(0)?, imm }
            }
        }
        VAND => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::And, dst: r(0)?, a: r(1)?, b: r(2)? },
        VORR => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Or, dst: r(0)?, a: r(1)?, b: r(2)? },
        VEOR => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Xor, dst: r(0)?, a: r(1)?, b: r(2)? },
        VBIC => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Bic, dst: r(0)?, a: r(1)?, b: r(2)? },
        VORN => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Orn, dst: r(0)?, a: r(1)?, b: r(2)? },
        VBSL => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Bsl, dst: r(0)?, a: r(1)?, b: r(2)? },
        VBIT => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Bit, dst: r(0)?, a: r(1)?, b: r(2)? },
        VBIF => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Bif, dst: r(0)?, a: r(1)?, b: r(2)? },
        // VSHL is both an immediate shift (`vshl.iN Qd, Qm, #n`, the two-registers-and-shift form)
        // and a per-lane variable shift (`vshl Qd, Qm, Qn`, the three-registers form). The register
        // form has a register third operand rather than an immediate.
        VSHL if !matches!(ops[2], Operand::Imm(_)) => {
            NeonStmt::ShiftReg { sat: false, ty, dst: r(0)?, src: r(1)?, amt: r(2)? }
        }
        VQSHL => NeonStmt::ShiftReg { sat: true, ty, dst: r(0)?, src: r(1)?, amt: r(2)? },
        VSHR | VSRA | VSHL | VSLI | VSRI | VRSHR | VRSRA => {
            use crate::ir::NeonShift;
            let sop = match op {
                VSHR => NeonShift::Shr,
                VSRA => NeonShift::Sra,
                VSHL => NeonShift::Shl,
                VSLI => NeonShift::Sli,
                VRSHR => NeonShift::Rshr,
                VRSRA => NeonShift::Rsra,
                _ => NeonShift::Sri,
            };
            let amount = match ops[2] {
                Operand::Imm(v) => v as u8,
                _ => return None,
            };
            NeonStmt::ShiftImm { op: sop, ty, dst: r(0)?, src: r(1)?, amount }
        }
        VEXT => {
            // The immediate is in element units of `ty.bits`; recover the byte offset.
            let elem = match ops[3] {
                Operand::Imm(v) => v,
                _ => return None,
            };
            let byte_off = (elem * (ty.bits as u32) / 8) as u8;
            NeonStmt::Ext { dst: r(0)?, a: r(1)?, b: r(2)?, byte_off }
        }
        VCVTFtoI => NeonStmt::CvtFloatInt { to_int: true, signed: ty.signed, dst: r(0)?, src: r(1)? },
        VCVTItoF => NeonStmt::CvtFloatInt { to_int: false, signed: ty.signed, dst: r(0)?, src: r(1)? },
        // VCEQ/VCGT/VCGE take either a register second operand (`a <rel> b`) or a `#0`
        // immediate (`a <rel> 0`, the two-registers-misc form). VCLE/VCLT exist only as the
        // compare-against-`#0` form.
        VCEQ | VCGT | VCGE => {
            let rel = match op {
                VCEQ => crate::ir::NeonCmp::Eq,
                VCGT => crate::ir::NeonCmp::Gt,
                _ => crate::ir::NeonCmp::Ge,
            };
            if matches!(ops[2], Operand::Imm(_)) {
                NeonStmt::CmpZero { op: rel, ty, dst: r(0)?, src: r(1)? }
            } else {
                NeonStmt::Cmp { op: rel, ty, dst: r(0)?, a: r(1)?, b: r(2)? }
            }
        }
        VCLE => NeonStmt::CmpZero { op: crate::ir::NeonCmp::Le, ty, dst: r(0)?, src: r(1)? },
        VCLT => NeonStmt::CmpZero { op: crate::ir::NeonCmp::Lt, ty, dst: r(0)?, src: r(1)? },
        VACGE => NeonStmt::CmpAbs { ge: true, dst: r(0)?, a: r(1)?, b: r(2)? },
        VACGT => NeonStmt::CmpAbs { ge: false, dst: r(0)?, a: r(1)?, b: r(2)? },
        VPMAX => NeonStmt::PairMinMax { ty, min: false, dst: r(0)?, a: r(1)?, b: r(2)? },
        VPMIN => NeonStmt::PairMinMax { ty, min: true, dst: r(0)?, a: r(1)?, b: r(2)? },
        VREV16 => NeonStmt::Rev { esize: ty.bits, container: 16, dst: r(0)?, src: r(1)? },
        VREV32 => NeonStmt::Rev { esize: ty.bits, container: 32, dst: r(0)?, src: r(1)? },
        VREV64 => NeonStmt::Rev { esize: ty.bits, container: 64, dst: r(0)?, src: r(1)? },
        VRECPE => NeonStmt::RecipEstimate { sqrt: false, dst: r(0)?, src: r(1)? },
        VRSQRTE => NeonStmt::RecipEstimate { sqrt: true, dst: r(0)?, src: r(1)? },
        VRECPS => NeonStmt::RecipStep { sqrt: false, dst: r(0)?, a: r(1)?, b: r(2)? },
        VRSQRTS => NeonStmt::RecipStep { sqrt: true, dst: r(0)?, a: r(1)?, b: r(2)? },
        // The permutes read and write both register operands (ops[0], ops[1]).
        VTRN => NeonStmt::Permute { op: crate::ir::PermuteOp::Trn, esize: ty.bits, a: r(0)?, b: r(1)? },
        VZIP => NeonStmt::Permute { op: crate::ir::PermuteOp::Zip, esize: ty.bits, a: r(0)?, b: r(1)? },
        VUZP => NeonStmt::Permute { op: crate::ir::PermuteOp::Uzp, esize: ty.bits, a: r(0)?, b: r(1)? },
        VTST => NeonStmt::Test { ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        // VMOVN result element is half the encoded (source) size.
        VMOVN => NeonStmt::Narrow { esize: ty.bits / 2, dst: r(0)?, src: r(1)? },
        // The narrowing right shifts. `dt` names the SOURCE element and its signedness,
        // so the result element is half it - the same convention as VMOVN.
        VSHRN | VRSHRN | VQSHRN | VQRSHRN | VQSHRUN | VQRSHRUN => {
            let Operand::Imm(shift) = ops[2] else { return None };
            let sat = match op {
                VSHRN | VRSHRN => None,
                // vqshrun narrows a SIGNED source into an UNSIGNED result; the plain
                // vqshrn keeps whatever signedness the source has.
                VQSHRUN | VQRSHRUN => Some((true, false)),
                _ => Some((ty.signed, ty.signed)),
            };
            NeonStmt::NarrowShift {
                esize: ty.bits / 2,
                dst: r(0)?,
                src: r(1)?,
                shift: shift as u8,
                round: matches!(op, VRSHRN | VQRSHRN | VQRSHRUN),
                sat,
            }
        }
        // VSHLL's `dt` names the SOURCE element, and the result is twice it.
        VSHLL => {
            let Operand::Imm(shift) = ops[2] else { return None };
            NeonStmt::WidenShift {
                esize: ty.bits,
                dst: r(0)?,
                src: r(1)?,
                shift: shift as u8,
                signed: ty.signed,
            }
        }
        // The saturating narrows, same shape as VMOVN. `dt` names the SOURCE element and
        // its signedness; the destination's signedness is the opcode's.
        VQMOVN => NeonStmt::NarrowSat {
            esize: ty.bits / 2,
            dst: r(0)?,
            src: r(1)?,
            src_signed: ty.signed,
            dst_signed: ty.signed,
        },
        VQMOVUN => NeonStmt::NarrowSat {
            esize: ty.bits / 2,
            dst: r(0)?,
            src: r(1)?,
            src_signed: true,
            dst_signed: false,
        },
        // `vtbl`/`vtbx`: ops are [Dd, {Dn..} run, Dm]. The table arrives as the same
        // register-run operand a `vld1` list uses, so it is read with `simd_points`'
        // shape rather than as a plain register.
        VTBL | VTBX => {
            let (first, count, stride, element) = match ops[1] {
                Operand::SIMDRegPoints { first, count, stride, element } => {
                    (first.num, count, stride, element)
                }
                _ => return None,
            };
            // A strided or element-wise run is not a table; refuse rather than read it
            // as one (the encoding cannot produce either, so this is a decoder check).
            if stride != 1 || !matches!(element, SIMDElement::Whole) || count == 0 || count > 4 {
                return None;
            }
            let (NeonReg::D(dst), NeonReg::D(index)) = (r(0)?, r(2)?) else { return None };
            NeonStmt::TableLookup { dst, table: first, len: count, index, extend: op == VTBX }
        }
        // VSWP is decoded but not lifted yet (it would land here as a permute swap).
        VSWP => return None,
    };
    neon_emittable(&st).then_some(st)
}

/// If `op` is a scalar lane operand `Dm[x]` (the "by scalar" second factor), return the source D
/// register number and lane index; otherwise `None` (a plain register or immediate).
fn scalar_lane(op: &Operand) -> Option<(u8, u8)> {
    match op {
        Operand::SIMDRegLane(src, lane) if src.size == SIMDSizeCode::D => Some((src.num, *lane)),
        _ => None,
    }
}

/// Whether the emitter has a wasm-SIMD sequence for this NEON statement. A few width
/// combinations have no direct wasm primitive (no `i8x16.mul`, no 64-bit lanewise
/// min/max, no 32->64 pairwise widen); those are reported as `Unsupported` at lift
/// rather than reached at emit. gcc's auto-vectorizer does not emit them.
fn neon_emittable(s: &NeonStmt) -> bool {
    // wasm SIMD has lanewise float arithmetic only for f32x4 and f64x2; NEON's F16
    // vector float has no wasm primitive. Integer widths are checked per op below.
    if let NeonStmt::Bin { ty, .. } | NeonStmt::MulAcc { ty, .. } | NeonStmt::Unary { ty, .. } = s {
        if ty.float && ty.bits != 32 && ty.bits != 64 {
            return false;
        }
    }
    match s {
        NeonStmt::Bin { op, ty, .. } => match op {
            // wasm has no `i8x16.mul` (but f32x4.mul is fine).
            NeonBin::Mul => ty.float || ty.bits != 8,
            // float min/max map straight to f32x4.min/max; wasm has no 64-bit
            // lanewise integer min/max.
            NeonBin::Max | NeonBin::Min => ty.float || ty.bits != 64,
            // `abd` (|a-b|) is emitted from integer min/max/sub only.
            NeonBin::Abd => !ty.float && ty.bits != 64,
            NeonBin::Add | NeonBin::Sub => true,
            // wasm's saturating add/sub exist for 8- and 16-bit lanes only.
            NeonBin::QAdd | NeonBin::QSub => !ty.float && (ty.bits == 8 || ty.bits == 16),
            // The halving forms are emitted through a widening extend, which covers
            // 8/16/32-bit sources.
            NeonBin::HAdd | NeonBin::HSub | NeonBin::RHAdd => !ty.float && ty.bits != 64,
        },
        // `vpadd` gathers even/odd lanes with shuffles then adds; the float form (f32x4.add)
        // is emittable at 32-bit, the F16 form has no wasm primitive.
        NeonStmt::PairAdd { ty, .. } => !ty.float || ty.bits == 32,
        // The pairwise min/max gather the same way; float is f32 only, and wasm has no
        // 64-bit lanewise integer min/max (the encoding does not produce one either).
        NeonStmt::PairMinMax { ty, .. } => {
            if ty.float {
                ty.bits == 32
            } else {
                ty.bits != 64
            }
        }
        NeonStmt::MulAcc { ty, .. } => ty.float || ty.bits != 8,
        // by-scalar multiply is decoded only for 16/32-bit elements (f32 or integer), all of which
        // have a wasm lane-multiply; the 8-bit form does not exist in this encoding class.
        NeonStmt::MulScalar { ty, .. } => ty.float || ty.bits != 8,
        // `extadd_pairwise` widens only 8->16 and 16->32.
        NeonStmt::PairLong { ty, .. } => ty.bits == 8 || ty.bits == 16,
        // The saturating abs/negate need a lanewise compare and negate, which wasm has for
        // 8/16/32-bit lanes; the encoding does not produce a 64-bit element anyway.
        NeonStmt::SatAbsNeg { ty, .. } => ty.bits != 64,
        // The saturating narrows use wasm's `narrow` pair, which exists only for
        // 16->8 and 32->16. A 64->32 `vqmovn` (a 32-bit result element) has no wasm
        // primitive, so it is refused at lift and reported rather than approximated.
        NeonStmt::NarrowSat { esize, .. } => *esize == 8 || *esize == 16,
        // The narrowing shifts share that `narrow` pair when they saturate, and the
        // shift itself needs a lanewise shift of the 2*esize source - which wasm has for
        // 16/32/64-bit lanes. So an 8-bit result (16-bit source) up to a 32-bit result
        // (64-bit source) is fine for the truncating forms, while a saturating 32-bit
        // result has no `narrow` and is refused.
        NeonStmt::NarrowShift { esize, sat, .. } => match sat {
            Some(_) => *esize == 8 || *esize == 16,
            None => *esize == 8 || *esize == 16 || *esize == 32,
        },
        // VSHLL widens with wasm's `extend_low`, which covers 8->16, 16->32 and 32->64.
        NeonStmt::WidenShift { esize, .. } => *esize == 8 || *esize == 16 || *esize == 32,
        // the widened `|a-b|` needs 16/32-bit min/max, so the source is 8/16-bit.
        NeonStmt::WideAbd { ty, .. } => ty.bits == 8 || ty.bits == 16,
        // Float compares are f32x4 only; wasm has no unsigned 64-bit lane compare, so
        // a 64-bit unsigned ordered compare has no direct primitive.
        NeonStmt::Cmp { op, ty, .. } => {
            if ty.float {
                ty.bits == 32
            } else {
                !(ty.bits == 64 && !ty.signed && !matches!(op, crate::ir::NeonCmp::Eq))
            }
        }
        // Compare-against-zero: f32 lanes only for the float form; the integer form is 8/16/32
        // (the decoder rejects the 64-bit element case).
        NeonStmt::CmpZero { ty, .. } => {
            if ty.float {
                ty.bits == 32
            } else {
                ty.bits != 64
            }
        }
        // Per-lane variable shift is emitted lane-by-lane over i32; the 8/16/32-bit unsaturated
        // form is supported. The saturating VQSHL and the 64-bit form lift as unsupported.
        NeonStmt::ShiftReg { sat, ty, .. } => !sat && ty.bits != 64,
        _ => true,
    }
}

/// The effective address of a single-register vldr/vstr. Handles the pc-relative
/// literal form by constant-folding (pc = `Align(addr+4,4)` in Thumb, `addr+8` in
/// ARM), matching how `adr` and integer literal loads resolve the pc.
fn vfp_single_addr(op: &Operand, iaddr: u32, thumb: bool) -> Option<Value> {
    let pc_const = |disp: u32| {
        let pc = if thumb { iaddr.wrapping_add(4) & !3 } else { iaddr.wrapping_add(8) };
        Value::Imm(pc.wrapping_add(disp))
    };
    match op {
        Operand::RegDeref(base) => {
            if base.number() == 15 {
                Some(pc_const(0))
            } else {
                Some(Value::Reg(base.number()))
            }
        }
        Operand::RegDerefPreindexOffset(base, off, add, _wback) => {
            let disp = if *add { *off as u32 } else { (*off as u32).wrapping_neg() };
            if base.number() == 15 {
                Some(pc_const(disp))
            } else {
                Some(bin(BinOp::Add, Value::Reg(base.number()), Value::Imm(disp)))
            }
        }
        _ => None,
    }
}

/// Lower a multi-register VFP memory transfer (vldm/vstm/vpush/vpop/vld1/vst1):
/// `count` consecutive registers starting at `num`, each `reg_bytes` wide, at
/// ascending addresses from `base` (decrement-before when `db`), with optional
/// base writeback. `post` (a register added to the base afterward) overrides the
/// fixed writeback delta when present.
#[allow(clippy::too_many_arguments)]
fn vfp_multi(
    size: SIMDSizeCode,
    num: u8,
    count: u8,
    base: u8,
    db: bool,
    wback: bool,
    post: Option<u8>,
    load: bool,
) -> Result<Vec<Stmt>, Error> {
    let bytes = reg_bytes(size);
    let total = count as u32 * bytes;
    // Start address: decrement-before subtracts the whole transfer up front.
    let start = if db {
        bin(BinOp::Sub, Value::Reg(base), Value::Imm(total))
    } else {
        Value::Reg(base)
    };
    let mut out = Vec::new();
    for i in 0..count {
        let addr = if i == 0 {
            start.clone()
        } else {
            bin(BinOp::Add, start.clone(), Value::Imm(i as u32 * bytes))
        };
        let reg = vfp_reg(size, num + i).ok_or(Error::Operand { addr: 0 })?;
        out.push(Stmt::VfpMem { reg, addr, load });
    }
    // Writeback runs after all transfers, so every transfer sees the old base.
    if let Some(rm) = post {
        out.push(Stmt::SetReg(base, bin(BinOp::Add, Value::Reg(base), Value::Reg(rm))));
    } else if wback {
        let newbase = if db {
            bin(BinOp::Sub, Value::Reg(base), Value::Imm(total))
        } else {
            bin(BinOp::Add, Value::Reg(base), Value::Imm(total))
        };
        out.push(Stmt::SetReg(base, newbase));
    }
    Ok(out)
}

/// Unused import guard for `Reg` (kept for future indirect-branch handling).
#[allow(dead_code)]
fn _reg_marker(_: Reg) {}

#[cfg(test)]
mod block_transfer_tests {
    use super::*;

    /// Decode one ARM-mode word and lower it.
    fn lower_arm_word(word: u32) -> Vec<Stmt> {
        let decoder = InstDecoder::default().with_thumb_mode(false);
        let bytes = word.to_le_bytes();
        let mut reader = U8Reader::new(&bytes);
        let inst = decoder.decode(&mut reader).expect("decodes");
        match inst.opcode {
            Opcode::STM(..) => lower_stm(&inst, 0).expect("lowers"),
            Opcode::LDM(..) => lower_ldm(&inst, 0).expect("lowers"),
            other => panic!("expected a block transfer, got {other:?}"),
        }
    }

    /// The constant offset a `Stmt::Store`'s `base + imm` address adds.
    fn store_offsets(stmts: &[Stmt]) -> Vec<u32> {
        stmts
            .iter()
            .filter_map(|s| match s {
                Stmt::Store { addr: Value::Bin(BinOp::Add, _, imm), .. } => match **imm {
                    Value::Imm(v) => Some(v),
                    _ => None,
                },
                _ => None,
            })
            .collect()
    }

    /// A block transfer's U (increment) and P (pre-index) bits decide WHERE the block
    /// sits relative to the base. Lowering everything as increment-after - which this
    /// did until 2026-07-25 - makes `stmdb rN!, {..}` write its registers ABOVE the
    /// base instead of below it, scribbling over whatever lives there (a caller's
    /// saved registers, typically) and then moving the base the wrong way. The bug is
    /// silent: the store succeeds, and the damage only surfaces later as a wrong value
    /// restored from a stack slot nobody appears to have touched.
    #[test]
    fn block_transfer_honours_the_increment_and_preindex_bits() {
        // The four modes for a 2-register block, per ARM ARM A5.4.
        assert_eq!(block_first_offset(true, false, 2), 0, "IA starts at the base");
        assert_eq!(block_first_offset(true, true, 2), 4, "IB starts one word above");
        assert_eq!(block_first_offset(false, false, 2), -4, "DA ends AT the base");
        assert_eq!(block_first_offset(false, true, 2), -8, "DB ends one word below");

        // Real encodings: stmia/stmdb r0!, {r4, r5} (cond=AL, Rn=r0, list=0x0030).
        // stmia = 0xe8a00030, stmdb = 0xe9200030.
        let ia = lower_arm_word(0xe8a0_0030);
        assert_eq!(store_offsets(&ia), vec![0, 4], "stmia writes at base, base+4");
        let db = lower_arm_word(0xe920_0030);
        assert_eq!(
            store_offsets(&db),
            vec![(-8i32) as u32, (-4i32) as u32],
            "stmdb writes BELOW the base, at base-8 and base-4"
        );

        // Writeback follows the same direction: +4n up, -4n down.
        // BinOp has no PartialEq, so name the direction rather than compare values.
        let wb = |stmts: &[Stmt]| -> Option<(&'static str, u32)> {
            stmts.iter().rev().find_map(|s| match s {
                Stmt::SetReg(0, Value::Bin(op, _, imm)) => {
                    let dir = match op {
                        BinOp::Add => "add",
                        BinOp::Sub => "sub",
                        _ => "other",
                    };
                    match **imm {
                        Value::Imm(v) => Some((dir, v)),
                        _ => None,
                    }
                }
                _ => None,
            })
        };
        assert_eq!(wb(&ia), Some(("add", 8)), "stmia! advances the base");
        assert_eq!(wb(&db), Some(("sub", 8)), "stmdb! retreats the base");

        // ldmdb r0!, {r4, r5} = 0xe930_0030 loads from the same slots it would store to.
        let ldb = lower_arm_word(0xe930_0030);
        let loads: Vec<u32> = ldb
            .iter()
            .filter_map(|s| match s {
                Stmt::SetReg(_, Value::Load { addr, .. }) => match &**addr {
                    Value::Bin(BinOp::Add, _, imm) => match **imm {
                        Value::Imm(v) => Some(v),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            })
            .collect();
        assert_eq!(loads, vec![(-8i32) as u32, (-4i32) as u32], "ldmdb reads below the base");
    }
}

#[cfg(test)]
mod pc_source_operand_tests {
    use super::*;

    /// Lower one Thumb instruction (given as its raw little-endian bytes) at `addr`.
    fn lower_thumb(bytes: &[u8], addr: u32) -> Vec<Stmt> {
        let decoder = InstDecoder::default().with_thumb_mode(true);
        let mut reader = U8Reader::new(bytes);
        let inst = decoder.decode(&mut reader).expect("decodes");
        lower_effects(&inst, addr, false).expect("lowers")
    }

    /// Reading `r15` as a data-processing SOURCE has to produce the ISA's pc constant.
    /// The transpiler keeps no live r15 in the register file, so an unfolded read yields
    /// 0 and the instruction computes a small integer instead of a code address.
    ///
    /// The encoding here is the real one this was found on: `f2 0f 4d 0c`
    /// (`add r12, pc, #77`) at `0x813432fa`, the head of a hand-written assembly MD5
    /// that builds a jump target with it and then does `blx r12`. Unfolded, r12 became
    /// `0x4d` and the run died on "indirect dispatch to unknown target 0x4c" - far from
    /// anything that looked like an addressing bug.
    #[test]
    fn reading_pc_as_a_source_operand_folds_to_the_isa_pc() {
        let stmts = lower_thumb(&[0x0f, 0xf2, 0x4d, 0x0c], 0x8134_32fa);
        // Thumb pc is Align(addr + 4, 4) = 0x813432fc, so r12 = 0x813432fc + 77.
        let want = ((0x8134_32fau32.wrapping_add(4)) & !3).wrapping_add(77);
        let got = stmts.iter().find_map(|s| match s {
            // add rd, pc, #imm folds to a single constant, or to Imm + Imm.
            Stmt::SetReg(12, Value::Imm(v)) => Some(*v),
            Stmt::SetReg(12, Value::Bin(BinOp::Add, a, b)) => match (&**a, &**b) {
                (Value::Imm(x), Value::Imm(y)) => Some(x.wrapping_add(*y)),
                _ => None,
            },
            _ => None,
        });
        assert_eq!(got, Some(want), "add r12, pc, #77 must compute {want:#x}, not a bare 77");
        assert_ne!(got, Some(77), "an unfolded pc read gives the immediate alone");
    }
}

