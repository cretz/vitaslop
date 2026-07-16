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
    Block, BinOp, ElemLane, FBinOp, Func, MemSize, NeonBin, NeonReg, NeonStmt, NeonType, Stmt, Term,
    Value, VfpOp, VfpReg,
};

/// A resolved import: the guest stub address a `bl`/`blx` targets maps to a
/// dense host-import index.
///
/// `redirects` handles inter-module calls in a multi-module link: a call to one
/// module's import stub is rewritten to a *direct* call to the exporting module's
/// function (both live in the same unified wasm module), so an inter-module call
/// is a plain guest-to-guest call with no host round-trip. Redirection is applied
/// to a branch target before the host-import lookup, so a redirected stub is
/// never also treated as a host import.
pub struct Imports<'a> {
    map: &'a BTreeMap<u32, u32>,
    redirects: &'a BTreeMap<u32, u32>,
}

impl<'a> Imports<'a> {
    pub fn new(map: &'a BTreeMap<u32, u32>, redirects: &'a BTreeMap<u32, u32>) -> Self {
        Imports { map, redirects }
    }
    fn get(&self, addr: u32) -> Option<u32> {
        self.map.get(&addr).copied()
    }
    /// Rewrite an inter-module import stub address to the real callee address it
    /// resolves to; a non-redirected address is returned unchanged.
    fn resolve(&self, addr: u32) -> u32 {
        self.redirects.get(&addr).copied().unwrap_or(addr)
    }
}

/// Result of discovering one function: its IR, the guest addresses of the direct
/// callees found in it, and any code pointers it materializes (address-taken
/// functions - e.g. a thread entry passed to sceKernelCreateThread - which the
/// direct-call closure alone would never reach).
pub struct Discovered {
    pub func: Func,
    pub callees: Vec<u32>,
    pub code_pointers: Vec<u32>,
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
    /// callee address if it is translated code, else it is a host import.
    Call { guest: Option<u32> },
    /// Unconditional branch: successor is `target` only.
    Jump(u32),
    /// Conditional/zero branch: successors are `target` and `addr + len`.
    Fork(u32),
    /// Returns to caller; no successors.
    Return,
    /// Stops without returning; no successors.
    Halt,
}

/// The pc-relative target of a branch instruction, if it has one. yaxpeax measures
/// the offset from the instruction address (already folding in the pipeline pc bias),
/// so the target is `addr + 2*off` for a Thumb branch (halfwords) and `addr + 4*off`
/// for an ARM branch (words) - NOT `pc + off`; adding the `pc+4`/`pc+8` bias again
/// would land the target 4/8 bytes past the real callee. `blx` switches to ARM and
/// word-aligns the pc (`Align(PC,4)`), so from a non-word-aligned address its target
/// is rounded down to the next word - otherwise it lands 2 bytes past the callee.
fn branch_target(inst: &Instruction, addr: u32, _thumb: bool) -> Option<u32> {
    for op in &inst.operands {
        match op {
            Operand::BranchThumbOffset(off) => {
                let t = addr.wrapping_add((2 * off) as u32);
                return Some(if inst.opcode == Opcode::BLX { t & !3 } else { t });
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
    imports: &Imports,
    r7: Option<u32>,
    noreturn_svc: &[u32],
) -> Flow {
    match inst.opcode {
        Opcode::B => match branch_target(inst, addr, thumb).map(|t| imports.resolve(t)) {
            // `b .` (branch to self) is an idle spin: treat as Halt so we do not
            // emit an infinite wasm loop the host cannot leave.
            Some(t) if t == addr => Flow::Halt,
            // An unconditional branch to an import stub/veneer is a tail call
            // (`return memset(...)`): it transfers out of the function, so it has
            // no in-function successor and adds no callee - lowering runs the
            // import then returns.
            Some(t) if inst.condition == ConditionCode::AL && imports.get(t).is_some() => {
                Flow::Return
            }
            Some(t) if inst.condition == ConditionCode::AL => Flow::Jump(t),
            Some(t) => Flow::Fork(t),
            None => Flow::Halt,
        },
        Opcode::CBZ | Opcode::CBNZ => match branch_target(inst, addr, thumb) {
            Some(t) => Flow::Fork(t),
            None => Flow::Halt,
        },
        Opcode::BL | Opcode::BLX => match branch_target(inst, addr, thumb).map(|t| imports.resolve(t)) {
            Some(t) if imports.get(t).is_some() => Flow::Call { guest: None },
            Some(t) => Flow::Call { guest: Some(t) },
            // A register-target `blx rN` is an indirect call through a function
            // pointer: it returns here, so continue to the fall-through, but the
            // target is not a statically-known callee (the dispatcher resolves it
            // at runtime).
            None if matches!(inst.operands[0], Operand::Reg(_)) => Flow::Call { guest: None },
            None => Flow::Halt,
        },
        Opcode::BX => match inst.operands[0] {
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
        _ => {
            // A write to pc via mov/ldr etc. would be an indirect jump; not yet
            // handled. Everything else falls through.
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
    mut regs: RegConsts,
    in_code: &impl Fn(u32) -> bool,
    discover_pointers: bool,
    code_pointers: &mut BTreeSet<u32>,
) -> RegConsts {
    let dst = inst.operands.first().and_then(regnum);
    match inst.opcode {
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
        regs = track_regs(ins, regs, &|_| false, false, &mut throwaway);
    }

    // How the index register was produced from the switch variable: `idx = switch + k`.
    // The nearest writer of the index register wins; an `add`/`sub` folds the label
    // base into `k`, a `mov` copies (k = 0), and anything else (a load, arithmetic we
    // do not model) means the index *is* the switch variable (k = 0).
    let (switch_reg, k) = {
        let def = snaps.iter().rev().find(|(_, ins, _, _)| {
            ins.operands.first().and_then(regnum) == Some(index_reg)
        });
        match def {
            Some((_, ins, _, before)) => match ins.opcode {
                // `idx = a +/- b`, where the label base folds into `k`. Both the
                // two-operand (`add rd,#imm` == `rd = rd +/- imm`) and three-operand
                // (`add rd,rn,rm`) forms appear; a source register carrying a tracked
                // constant is the base, the other is the switch variable.
                Opcode::ADD | Opcode::SUB => {
                    let neg = ins.opcode == Opcode::SUB;
                    let signed = |kk: u32| if neg { kk.wrapping_neg() } else { kk };
                    let o1 = &ins.operands[1];
                    let o2 = &ins.operands[2];
                    if matches!(o2, Operand::Nothing) {
                        // Two-operand: `rd = rd (op) o1`. The switch value is the
                        // destination (also the implicit first source).
                        if let Some(kk) = imm(o1) {
                            (index_reg, signed(kk))
                        } else if let Some(kk) = regnum(o1).and_then(|r| before[r as usize]) {
                            (index_reg, signed(kk))
                        } else {
                            (index_reg, 0)
                        }
                    } else if let (Some(rs), Some(kk)) = (regnum(o1), const_operand(o2, before)) {
                        (rs, signed(kk))
                    } else if let (Some(kk), Some(rs)) = (const_operand(o1, before), regnum(o2)) {
                        // `rd = imm (op) rs`: only `imm + rs` keeps `rs` a straight
                        // index; `imm - rs` negates the index, which we do not model.
                        if neg { (index_reg, 0) } else { (rs, kk) }
                    } else {
                        (index_reg, 0)
                    }
                }
                Opcode::MOV => regnum(&ins.operands[1]).map_or((index_reg, 0), |rs| (rs, 0)),
                _ => (index_reg, 0),
            },
            None => (index_reg, 0),
        }
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
        // In-range is the side that steers toward the table branch. The taken branch
        // does so when its target lands in the setup between the guard and the table.
        let in_range_taken = gt > br_addr && gt <= tb_addr;
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
) -> Option<SwitchInfo> {
    let index = switch_index_reg(inst)?;
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
    let count = match cmp_count {
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
        let target = pc.wrapping_add(2 * read(i)?);
        if target.wrapping_sub(base) as usize >= code.len() {
            return None; // target outside the image: bound is wrong, bail cleanly
        }
        if target.wrapping_sub(tb_addr).min(tb_addr.wrapping_sub(target)) > MAX_REACH {
            return None; // implausibly far from the table: bound is wrong, bail cleanly
        }
        targets.push(target);
    }
    Some(SwitchInfo { index, targets, default })
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
    let mut callees: BTreeSet<u32> = BTreeSet::new();
    // Address-taken code pointers materialized in this function (thread entries,
    // callbacks). Collected via `movw`/`movt` tracking; processed as tentative
    // entries by the caller.
    let mut code_pointers: BTreeSet<u32> = BTreeSet::new();
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
    leaders.insert(entry);

    // A guest address is decodable only if it lies within the code image.
    let in_bounds = |addr: u32| {
        let off = addr.wrapping_sub(base) as usize;
        off < code.len()
    };

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
        let next_regs = track_regs(&inst, regs, &in_bounds, discover_pointers, &mut code_pointers);
        let next = addr.wrapping_add(len);

        // Table-branch (`tbb`/`tbh`) switch dispatch: resolve the inline jump table
        // now, while the whole decoded stream (the range check that bounds it) is in
        // hand, and seed every case body as a leader. The bytes right after the
        // instruction are the table itself, never code, so this path never falls
        // through - resolved or not. An unresolved table is reported per-function in
        // pass 2 rather than marching the decoder into table data.
        if matches!(inst.opcode, Opcode::TBB | Opcode::TBH) {
            if let Some(info) = resolve_switch(code, base, addr, &inst, &decoded) {
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
            continue;
        }

        match flow(&inst, addr, len, thumb, imports, regs[7], noreturn_svc) {
            Flow::Seq => work.push((next, next_it, next_regs)),
            Flow::Call { guest } => {
                if let Some(t) = guest {
                    callees.insert(t);
                }
                work.push((next, next_it, next_regs));
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

    // Pass 2: build blocks. Each leader starts a block that runs until a
    // terminating instruction or the address just before the next leader.
    let mut blocks = Vec::new();
    for &addr in &leaders {
        // A leader whose first instruction failed to decode becomes a lone trapping
        // block, so a branch/switch that targets it stays well-formed (the target is
        // a real block) while executing it faults loudly.
        if trap_leaders.contains(&addr) {
            blocks.push(Block { addr, stmts: Vec::new(), term: Term::Unreachable });
            continue;
        }
        // A leader can be out of bounds (a branch target past the image); such a
        // block is unreachable in practice - skip it.
        if !decoded.contains_key(&addr) {
            continue;
        }
        let mut cursor = addr;
        let mut stmts = Vec::new();
        let term = loop {
            let Some((inst, len, applied, in_it)) = decoded.get(&cursor) else {
                // Fell through to code that was never decoded (off image or
                // unreachable): stop the function here.
                break Term::Halt;
            };
            // A table branch ends the block with a computed jump built from the table
            // recovered in pass 1. An unresolved one is a clean per-function failure.
            if matches!(inst.opcode, Opcode::TBB | Opcode::TBH) {
                match switches.get(&cursor) {
                    Some(info) => {
                        break Term::Switch {
                            index: Value::Reg(info.index),
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
                match lower_insn(inst, cursor, *len, *applied, *in_it, thumb, imports) {
                    Ok(v) => v,
                    // An unlifted instruction (e.g. `udf`, or a NEON op not yet
                    // covered). Strict callers report it; the lenient build runs the
                    // block's valid prefix then traps, isolating the gap to this block
                    // instead of stubbing the whole function.
                    Err(_) if isolate => break Term::Unreachable,
                    Err(e) => return Err(e),
                };
            stmts.append(&mut effects);
            cursor = cursor.wrapping_add(*len);
            if let Some(t) = term {
                break t;
            }
            // A block also ends when the next instruction begins another block.
            if leaders.contains(&cursor) && decoded.contains_key(&cursor) {
                break Term::Fallthrough;
            }
        };
        blocks.push(Block { addr, stmts, term });
    }
    blocks.sort_by_key(|b| b.addr);

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

    Ok(Discovered {
        func: Func { addr: entry, thumb, blocks, stub: false },
        callees: callees.into_iter().collect(),
        code_pointers: code_pointers.into_iter().collect(),
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
fn operand_value(op: &Operand) -> Option<Value> {
    match op {
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
fn dataproc(inst: &Instruction) -> Option<(u8, Value, Value)> {
    let rd = regnum(&inst.operands[0])?;
    if matches!(inst.operands[2], Operand::Nothing) {
        let op2 = operand_value(&inst.operands[1])?;
        Some((rd, Value::Reg(rd), op2))
    } else {
        let rn = operand_value(&inst.operands[1])?;
        let op2 = operand_value(&inst.operands[2])?;
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
    imports: &Imports,
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
        BX => {
            return match ops[0] {
                Operand::Reg(r) if r.number() == 14 => Ok((vec![], Some(Term::Return))),
                // `bx rN` tail-calls through a function pointer: dispatch to the
                // runtime target, then return to our caller (lr is unchanged, so
                // the callee's own return unwinds past us correctly).
                Operand::Reg(r) => Ok((
                    vec![Stmt::CallIndirect { addr: Value::Reg(r.number()), set_lr: None }],
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
                // itself, AFTER reading the target, so `blx lr` still works.
                None => match ops[0] {
                    Operand::Reg(r) => {
                        vec![Stmt::CallIndirect { addr: Value::Reg(r.number()), set_lr: Some(lr) }]
                    }
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

    let mut out = Vec::new();
    match inst.opcode {
        NOP | IT | HINT => {}

        // Memory barriers and cache preload hints have no effect on the guest's
        // observable state in our memory model (one guest CPU worker, sequential
        // consistency at host-call sync points), so they lower to nothing.
        DMB | DSB | ISB | PLD | PLI => {}

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
            let src = operand_value(&ops[1]).ok_or_else(err)?;
            // Flags before the write: the value expression reads original regs.
            if sets_flags {
                out.push(Stmt::FlagsLogic { value: src.clone(), carry: None });
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
            let src = operand_value(&ops[1]).ok_or_else(err)?;
            let value = Value::Not(Box::new(src));
            if sets_flags {
                out.push(Stmt::FlagsLogic { value: value.clone(), carry: None });
            }
            out.push(Stmt::SetReg(rd, value));
        }

        ADD => {
            let (rd, rn, op2) = dataproc(inst).ok_or_else(err)?;
            if sets_flags {
                out.push(Stmt::FlagsAdd { a: rn.clone(), b: op2.clone(), cin: Value::Imm(0) });
            }
            out.push(Stmt::SetReg(rd, bin(BinOp::Add, rn, op2)));
        }
        SUB => {
            let (rd, rn, op2) = dataproc(inst).ok_or_else(err)?;
            if sets_flags {
                out.push(Stmt::FlagsAdd {
                    a: rn.clone(),
                    b: Value::Not(Box::new(op2.clone())),
                    cin: Value::Imm(1),
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
            let (rd, rn, op2) = dataproc(inst).ok_or_else(err)?;
            if sets_flags {
                out.push(Stmt::FlagsAdd { a: rn, b: op2, cin: Value::Flag(crate::abi::Flag::C) });
                out.push(Stmt::SetReg(rd, Value::CarryAddResult));
            } else {
                let sum =
                    bin(BinOp::Add, bin(BinOp::Add, rn, op2), Value::Flag(crate::abi::Flag::C));
                out.push(Stmt::SetReg(rd, sum));
            }
        }
        // sbc rd, rn, op2 => rd = rn - op2 - NOT(C) = rn + ~op2 + C.
        SBC => {
            let (rd, rn, op2) = dataproc(inst).ok_or_else(err)?;
            let not_op2 = Value::Not(Box::new(op2));
            if sets_flags {
                out.push(Stmt::FlagsAdd { a: rn, b: not_op2, cin: Value::Flag(crate::abi::Flag::C) });
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
            let (rd, rn, op2) = dataproc(inst).ok_or_else(err)?;
            if sets_flags {
                out.push(Stmt::FlagsAdd {
                    a: op2.clone(),
                    b: Value::Not(Box::new(rn.clone())),
                    cin: Value::Imm(1),
                });
            }
            out.push(Stmt::SetReg(rd, bin(BinOp::Sub, op2, rn)));
        }
        CMP => {
            let a = operand_value(&ops[0]).ok_or_else(err)?;
            let b = operand_value(&ops[1]).ok_or_else(err)?;
            out.push(Stmt::FlagsAdd { a, b: Value::Not(Box::new(b)), cin: Value::Imm(1) });
        }
        CMN => {
            let a = operand_value(&ops[0]).ok_or_else(err)?;
            let b = operand_value(&ops[1]).ok_or_else(err)?;
            out.push(Stmt::FlagsAdd { a, b, cin: Value::Imm(0) });
        }

        AND | BIC | ORR | ORN | EOR | TST => {
            let (rd, rn, op2) = dataproc(inst).ok_or_else(err)?;
            // BIC/ORN take the bitwise complement of the second operand.
            let op2 = if matches!(inst.opcode, BIC | ORN) {
                Value::Not(Box::new(op2))
            } else {
                op2
            };
            let binop = match inst.opcode {
                ORR | ORN => BinOp::Or,
                EOR => BinOp::Xor,
                _ => BinOp::And, // AND, BIC, TST
            };
            let result = bin(binop, rn, op2);
            if inst.opcode == TST {
                out.push(Stmt::FlagsLogic { value: result, carry: None });
            } else {
                if sets_flags {
                    out.push(Stmt::FlagsLogic { value: result.clone(), carry: None });
                }
                out.push(Stmt::SetReg(rd, result));
            }
        }

        LSL | LSR | ASR => {
            let (rd, rn, sh) = dataproc(inst).ok_or_else(err)?;
            let binop = match inst.opcode {
                LSL => BinOp::Shl,
                LSR => BinOp::Lsr,
                _ => BinOp::Asr,
            };
            let result = bin(binop, rn.clone(), sh.clone());
            if sets_flags {
                let carry = shift_carry(inst.opcode, &rn, &sh);
                out.push(Stmt::FlagsLogic { value: result.clone(), carry });
            }
            out.push(Stmt::SetReg(rd, result));
        }
        // ror rd, rn, rm/#imm: rotate right. Amount is masked mod 32 (wasm rotr,
        // matching ARM's register-rotate masking); ROR's carry-out is bit 31 of
        // the result when it sets flags.
        ROR => {
            let (rd, rn, sh) = dataproc(inst).ok_or_else(err)?;
            let result = bin(BinOp::Ror, rn, sh);
            if sets_flags {
                let carry = bin(BinOp::And, bin(BinOp::Lsr, result.clone(), Value::Imm(31)), Value::Imm(1));
                out.push(Stmt::FlagsLogic { value: result.clone(), carry: Some(carry) });
            }
            out.push(Stmt::SetReg(rd, result));
        }
        // bfi rd, rn, #lsb, #width: insert the low `width` bits of rn into rd at
        // bit `lsb`, leaving the rest of rd unchanged.
        BFI => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rn = operand_value(&ops[1]).ok_or_else(err)?;
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
            let (rd, rn, op2) = dataproc(inst).ok_or_else(err)?;
            let result = bin(BinOp::Mul, rn, op2);
            if sets_flags {
                out.push(Stmt::FlagsLogic { value: result.clone(), carry: None });
            }
            out.push(Stmt::SetReg(rd, result));
        }
        // umull/smull rdlo, rdhi, rn, rm: {rdhi:rdlo} = rn * rm (64-bit). The S
        // form (flag-setting) is not emitted by the compilers we target; ignore
        // flags here (the widening product's N/Z would need the 64-bit value).
        UMULL | SMULL => {
            let rdlo = regnum(&ops[0]).ok_or_else(err)?;
            let rdhi = regnum(&ops[1]).ok_or_else(err)?;
            let rn = operand_value(&ops[2]).ok_or_else(err)?;
            let rm = operand_value(&ops[3]).ok_or_else(err)?;
            out.push(Stmt::MulLong { rdlo, rdhi, rn, rm, signed: inst.opcode == SMULL });
        }
        // clz rd, rm.
        CLZ => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rm = operand_value(&ops[1]).ok_or_else(err)?;
            out.push(Stmt::SetReg(rd, Value::Clz(Box::new(rm))));
        }
        // rbit rd, rm: reverse the 32 bits of rm.
        RBIT => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rm = operand_value(&ops[1]).ok_or_else(err)?;
            out.push(Stmt::Rbit { rd, rm });
        }
        // mla rd, rn, rm, ra => rd = rn*rm + ra; mls => rd = ra - rn*rm.
        MLA | MLS => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rn = operand_value(&ops[1]).ok_or_else(err)?;
            let rm = operand_value(&ops[2]).ok_or_else(err)?;
            let ra = operand_value(&ops[3]).ok_or_else(err)?;
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
            let rm = operand_value(&ops[1]).ok_or_else(err)?;
            out.push(Stmt::SetReg(rd, byte_reverse(rm)));
        }
        // rev16 rd, rm: reverse the bytes within each halfword.
        REV16 => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rm = operand_value(&ops[1]).ok_or_else(err)?;
            // ((rm >> 8) & 0x00FF00FF) | ((rm << 8) & 0xFF00FF00)
            let hi = bin(BinOp::And, bin(BinOp::Lsr, rm.clone(), Value::Imm(8)), Value::Imm(0x00FF_00FF));
            let lo = bin(BinOp::And, bin(BinOp::Shl, rm, Value::Imm(8)), Value::Imm(0xFF00_FF00));
            out.push(Stmt::SetReg(rd, bin(BinOp::Or, hi, lo)));
        }
        // Sign/zero extend a byte or halfword (the plain form, no pre-rotate).
        SXTB | UXTB | SXTH | UXTH => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rm = operand_value(&ops[1]).ok_or_else(err)?;
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
        // Extend-and-add: `Xtab rd, rn, rm` => rd = rn + extend(rm's low byte/half).
        SXTAB | UXTAB | SXTAH | UXTAH => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rn = operand_value(&ops[1]).ok_or_else(err)?;
            let rm = operand_value(&ops[2]).ok_or_else(err)?;
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
        UBFX | SBFX => {
            let rd = regnum(&ops[0]).ok_or_else(err)?;
            let rn = operand_value(&ops[1]).ok_or_else(err)?;
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
        // vdup.N Qd/Dd, Rt: broadcast a core register across every element. (The
        // lane-source form `vdup Dd, Dm[x]` is a different operand shape, not yet
        // needed.)
        VDUP(dt) => {
            let dst = neon_reg(&ops[0]).ok_or_else(err)?;
            let rt = regnum(&ops[1]).ok_or_else(err)?;
            out.push(Stmt::Neon(NeonStmt::DupCore { ty: neon_type(dt), dst, rt }));
        }
        Neon { op, dt } => {
            out.push(Stmt::Neon(lower_neon(op, dt, ops).ok_or_else(err)?));
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

/// `pop {list}` / `ldmia rn(!), {list}`: load each register ascending, then
/// writeback. When the list has pc, the pc slot is consumed (sp advances) but
/// not written - the caller turns that into a return.
fn lower_ldm(inst: &Instruction, _addr: u32) -> Result<Vec<Stmt>, Error> {
    let regs = reglist(inst);
    let base = ldm_base(inst);
    let wback = ldm_wback(inst);
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
                addr: Box::new(bin(BinOp::Add, Value::Reg(base), Value::Imm(4 * i as u32))),
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
        out.push(Stmt::SetReg(
            base,
            bin(BinOp::Add, Value::Reg(base), Value::Imm(4 * regs.len() as u32)),
        ));
    }
    Ok(out)
}

/// True if evaluating `v` reads guest register `r` (so writing `r` first would
/// change `v`). Used to order multi-register loads so the base register, when it is
/// also a destination, is written last.
fn value_uses_reg(v: &Value, r: u8) -> bool {
    match v {
        Value::Reg(x) => *x == r,
        Value::Imm(_) | Value::Flag(_) | Value::CarryAddResult => false,
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

/// `stmia rn(!), {list}`: store each register ascending, then writeback.
fn lower_stm(inst: &Instruction, _addr: u32) -> Result<Vec<Stmt>, Error> {
    let regs = reglist(inst);
    let base = ldm_base(inst);
    let wback = ldm_wback(inst);
    let mut out = Vec::new();
    for (i, r) in regs.iter().enumerate() {
        out.push(Stmt::Store {
            addr: bin(BinOp::Add, Value::Reg(base), Value::Imm(4 * i as u32)),
            data: Value::Reg(*r),
            size: MemSize::Word,
        });
    }
    if wback {
        out.push(Stmt::SetReg(
            base,
            bin(BinOp::Add, Value::Reg(base), Value::Imm(4 * regs.len() as u32)),
        ));
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
    if n != 1 || stride != 1 {
        // vld2/vld3/vld4 (and any strided list) need deinterleaving - deferred.
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
        VMUL => NeonStmt::Bin { op: NeonBin::Mul, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VMAX => NeonStmt::Bin { op: NeonBin::Max, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VMIN => NeonStmt::Bin { op: NeonBin::Min, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VABD => NeonStmt::Bin { op: NeonBin::Abd, ty, dst: r(0)?, a: r(1)?, b: r(2)? },
        VMLA => NeonStmt::MulAcc { ty, dst: r(0)?, a: r(1)?, b: r(2)?, sub: false },
        VMLS => NeonStmt::MulAcc { ty, dst: r(0)?, a: r(1)?, b: r(2)?, sub: true },
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
            let imm = match ops[1] {
                Operand::Imm(v) => v,
                _ => return None,
            };
            NeonStmt::MovImm { ty, dst: r(0)?, imm }
        }
        VAND => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::And, dst: r(0)?, a: r(1)?, b: r(2)? },
        VORR => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Or, dst: r(0)?, a: r(1)?, b: r(2)? },
        VEOR => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Xor, dst: r(0)?, a: r(1)?, b: r(2)? },
        VBIC => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Bic, dst: r(0)?, a: r(1)?, b: r(2)? },
        VORN => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Orn, dst: r(0)?, a: r(1)?, b: r(2)? },
        VBSL => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Bsl, dst: r(0)?, a: r(1)?, b: r(2)? },
        VBIT => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Bit, dst: r(0)?, a: r(1)?, b: r(2)? },
        VBIF => NeonStmt::Bitwise { op: crate::ir::NeonBitwise::Bif, dst: r(0)?, a: r(1)?, b: r(2)? },
    };
    neon_emittable(&st).then_some(st)
}

/// Whether the emitter has a wasm-SIMD sequence for this NEON statement. A few width
/// combinations have no direct wasm primitive (no `i8x16.mul`, no 64-bit lanewise
/// min/max, no 32->64 pairwise widen); those are reported as `Unsupported` at lift
/// rather than reached at emit. gcc's auto-vectorizer does not emit them.
fn neon_emittable(s: &NeonStmt) -> bool {
    match s {
        NeonStmt::Bin { op, ty, .. } => match op {
            // wasm has no `i8x16.mul`.
            NeonBin::Mul => ty.bits != 8,
            // wasm has no 64-bit lanewise min/max (`abd` uses them).
            NeonBin::Max | NeonBin::Min | NeonBin::Abd => ty.bits != 64,
            NeonBin::Add | NeonBin::Sub => true,
        },
        NeonStmt::MulAcc { ty, .. } => ty.bits != 8,
        // `extadd_pairwise` widens only 8->16 and 16->32.
        NeonStmt::PairLong { ty, .. } => ty.bits == 8 || ty.bits == 16,
        // the widened `|a-b|` needs 16/32-bit min/max, so the source is 8/16-bit.
        NeonStmt::WideAbd { ty, .. } => ty.bits == 8 || ty.bits == 16,
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
