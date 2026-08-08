//! Reference interpreter for the emittable USSE arithmetic core.
//!
//! This evaluates the SAME operations the WGSL emitter emits ([`crate::wgsl`]) - mad, mul,
//! add, min, max, frc, dot, and inline CNST6 constants - against a register file of `f32`
//! lanes, producing the resulting register values. It exists for two reasons:
//!
//! 1. **Semantic validation.** The emitter turns each op into a WGSL string; this turns the
//!    same op into a number. Unit tests here check the numbers against hand-computed values,
//!    so the *meaning* the emitter claims (not just its syntax) is pinned down.
//! 2. **Behavioral-oracle foundation.** Establishing the UNDOCUMENTED groups (flow / tex /
//!    pack / mov / transcendentals) needs a reference that can partially-evaluate a shader
//!    from captured inputs and constrain the unknown ops by the values their consumers
//!    require. That reference is this interpreter, extended as each op becomes established.
//!
//! It is deliberately faithful to the scalarised register-file model: every operand reads
//! `bank[base + lane]` per channel. Screen-space derivatives (`dsx`/`dsy`) evaluate to 0 in
//! a single-pixel context (they need a pixel quad); every other emittable op is exact. It
//! never interprets an unestablished op - it returns [`InterpError`] naming it, mirroring
//! the emitter's hard-fail contract so the interpreter can never fabricate a value.

use crate::ir::{Bank, Instr, Op, Operand, Predicate, Shader};
use crate::wgsl::cnst6_value;

/// A register file of 32-bit lanes for one shader invocation. Banks are addressed by scalar
/// lane index (the same `base + lane` model the decoder/emitter use). `pa`/`sa` are the
/// interpolated varyings / uniform buffer (shader inputs); `r`/`o`/`i` start zeroed and are
/// written by the program.
#[derive(Debug, Clone, Default)]
pub struct RegFile {
    pub r: Vec<f32>,
    pub pa: Vec<f32>,
    pub sa: Vec<f32>,
    pub o: Vec<f32>,
    pub i: Vec<f32>,
    /// Predicate registers p0..p3 (written by test ops, gate predicated instructions).
    pub p: [bool; 4],
    /// The INDEX register file, for register-INDIRECT operands. Two registers, because the
    /// extension row names exactly two indexed banks (INDEXED1 -> i0, INDEXED2 -> i1).
    pub idx: [i32; 2],
}

impl RegFile {
    /// A register file with each bank pre-sized to `lanes` zeroed lanes.
    pub fn with_lanes(lanes: usize) -> RegFile {
        RegFile {
            r: vec![0.0; lanes],
            pa: vec![0.0; lanes],
            sa: vec![0.0; lanes],
            o: vec![0.0; lanes],
            i: vec![0.0; lanes],
            p: [false; 4],
            idx: [0; 2],
        }
    }

    fn bank_mut(&mut self, bank: Bank) -> Option<&mut Vec<f32>> {
        Some(match bank {
            Bank::Temp => &mut self.r,
            Bank::PrimaryAttr => &mut self.pa,
            Bank::SecondaryAttr => &mut self.sa,
            Bank::Output => &mut self.o,
            Bank::Internal => &mut self.i,
            // Constant is materialised inline; Global is a hardware register the interpreter
            // has no state for (its value is pipeline state, not register-file storage).
            // Indexed/Index are register-INDIRECT addressing and the index register file: the
            // interpreter is a straight-line evaluator with no index state, so an operand that
            // needs one has no value here rather than a fabricated one.
            Bank::Constant | Bank::Immediate | Bank::Global | Bank::Indexed | Bank::Index
            | Bank::Raw(_) => return None,
        })
    }

    fn bank(&self, bank: Bank) -> Option<&Vec<f32>> {
        Some(match bank {
            Bank::Temp => &self.r,
            Bank::PrimaryAttr => &self.pa,
            Bank::SecondaryAttr => &self.sa,
            Bank::Output => &self.o,
            Bank::Internal => &self.i,
            // Constant is materialised inline; Global is a hardware register the interpreter
            // has no state for (its value is pipeline state, not register-file storage).
            // Indexed/Index are register-INDIRECT addressing and the index register file: the
            // interpreter is a straight-line evaluator with no index state, so an operand that
            // needs one has no value here rather than a fabricated one.
            Bank::Constant | Bank::Immediate | Bank::Global | Bank::Indexed | Bank::Index
            | Bank::Raw(_) => return None,
        })
    }
}

/// Why interpretation stopped. Mirrors the emitter's hard-fail contract: an op or operand
/// the reference does not model is never given a fabricated value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpError {
    /// An op not wired for interpretation (an unestablished / non-arithmetic op). Names it.
    UnsupportedOp { index: usize, op: &'static str },
    /// A blocked instruction (the decoder flagged an unmodeled feature).
    Blocked { index: usize, reason: &'static str },
    /// An operand referenced a bank/lane the register file does not provide.
    OutOfRange { index: usize },
}

impl core::fmt::Display for InterpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InterpError::UnsupportedOp { index, op } => {
                write!(f, "interp: instruction #{index} op '{op}' not modeled")
            }
            InterpError::Blocked { index, reason } => {
                write!(f, "interp: instruction #{index} blocked ({reason})")
            }
            InterpError::OutOfRange { index } => {
                write!(f, "interp: instruction #{index} operand out of range")
            }
        }
    }
}

/// Read one source channel `c` of `op` from `regs`, applying swizzle + abs/neg (or the
/// inline constant). `None` on an unmapped bank / out-of-range lane.
fn read_channel(regs: &RegFile, op: &Operand, c: usize) -> Option<f32> {
    // A register-INDIRECT operand resolves through the index register file to a real bank,
    // exactly as the emitter's `indexed_element` does. Modelling it here is what lets the
    // interpreter run a program that indexes a uniform ARRAY - the idiom a vector-canvas
    // vertex program is built out of, and the reason such programs used to be unmeasurable.
    let mut v = if matches!(op.bank, Bank::Indexed) {
        let bank = regs.bank(crate::ir::indexed_sub_bank(op.index))?;
        let reg = if op.bank_sel == 0 { 0 } else { 1 };
        let offset = crate::ir::indexed_offset(op.index) as i32 + c as i32;
        let e = (regs.idx[reg] + offset).max(0) as usize;
        *bank.get(e.min(bank.len().saturating_sub(1)))?
    } else if matches!(op.bank, Bank::Constant) {
        cnst6_value(op.index)
    } else if matches!(op.bank, Bank::Immediate) {
        // A scalar literal (spec A.7): the operand's number IS the value, and the swizzle's
        // constant selectors still apply, exactly as the emitter reads it.
        match op.swizzle[c] {
            4 => 0.0,
            5 => 1.0,
            6 => 2.0,
            7 => 0.5,
            _ => op.index as f32,
        }
    } else {
        let sel = op.swizzle[c];
        match sel {
            0..=3 => *regs.bank(op.bank)?.get(op.index as usize + sel as usize)?,
            4 => 0.0,
            5 => 1.0,
            6 => 2.0,
            7 => 0.5,
            _ => return None,
        }
    };
    if op.abs {
        v = v.abs();
    }
    if op.neg {
        v = -v;
    }
    Some(v)
}

/// Compute the scalar value an emittable op produces for written channel `c` (dot broadcasts
/// the same scalar to every channel). Returns the op mnemonic on an unmodeled op.
fn eval_channel(regs: &RegFile, instr: &Instr, c: usize) -> Result<f32, &'static str> {
    let s = |n: usize, ch: usize| instr.srcs.get(n).and_then(|o| read_channel(regs, o, ch)).ok_or("operand");
    Ok(match instr.op {
        Op::Mul => s(0, c)? * s(1, c)?,
        Op::Add => s(0, c)? + s(1, c)?,
        Op::Min => s(0, c)?.min(s(1, c)?),
        Op::Max => s(0, c)?.max(s(1, c)?),
        Op::Frc => s(0, c)?.fract(),
        // Screen-space derivatives require a pixel quad; 0 in a single-pixel reference.
        Op::Dsx | Op::Dsy => 0.0,
        Op::Mad => s(0, c)? * s(1, c)? + s(2, c)?,
        Op::Dot { components } => {
            let n = (components as usize).clamp(1, 4);
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += s(0, k)? * s(1, k)?;
            }
            acc
        }
        // Group 0x30 transcendentals (base-2) and the 0x38 move; the source already
        // broadcasts its selected component, so channel `c` uses `s(0, c)`.
        Op::Rcp => 1.0 / s(0, c)?,
        Op::Rsq => 1.0 / s(0, c)?.sqrt(),
        Op::Log => s(0, c)?.log2(),
        Op::Exp => s(0, c)?.exp2(),
        // Move and float<->float pack are swizzled copies.
        Op::Mov | Op::Pack { .. } => s(0, c)?,
        // Conditional move (VMOVC): test src0 (srcs[2]) against zero, pick src1 (srcs[0]) when
        // it holds else src2 (srcs[1]) - the same select the emitter produces.
        Op::Cmov { test } => {
            use crate::ir::CompareMethod::*;
            let t = s(2, c)?;
            let cond = match test {
                EqZero => t == 0.0,
                NeZero => t != 0.0,
                LtZero => t < 0.0,
                LteZero => t <= 0.0,
            };
            if cond { s(0, c)? } else { s(1, c)? }
        }
        // Integer bitwise/shift on the lane bit pattern (channel 0 only). A 16-bit lane
        // operates on the low half and WRAPS there, so the mask is part of the result and not
        // a tidy-up: a left shift that overflows 16 bits keeps different bits than one that
        // overflows 32.
        Op::Bitwise { kind, imm, lane_bits } => {
            use crate::ir::BitwiseKind::*;
            let mask: u32 = if lane_bits >= 32 { u32::MAX } else { (1u32 << lane_bits) - 1 };
            let shift_mask = lane_bits as u32 - 1;
            let a = s(0, c)?.to_bits() & mask;
            let b = match imm {
                Some(v) => v,
                None => s(1, c)?.to_bits(),
            } & mask;
            let r = match kind {
                And => a & b,
                Or => a | b,
                Xor => a ^ b,
                Shl => a << (b & shift_mask),
                Shr => a >> (b & shift_mask),
                // Arithmetic shift is over the LANE's sign bit, so a narrow lane is sign-
                // extended to 32 first and re-masked after.
                Asr => {
                    let signed = if lane_bits >= 32 {
                        a as i32
                    } else {
                        ((a << (32 - lane_bits)) as i32) >> (32 - lane_bits)
                    };
                    (signed >> (b & shift_mask)) as u32
                }
            };
            f32::from_bits(r & mask)
        }
        // A truncating float->integer convert whose result is the integer's BIT PATTERN in the
        // lane - which is what the integer ops above then read. Matching `emit_pack_to_int`,
        // including its clamp: the source can be a NaN or a huge float, and an unclamped
        // conversion of either is undefined rather than merely wrong.
        Op::PackToInt { bits, signed, .. } => {
            let f = s(0, c)?;
            let lane_mask: u32 = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
            let raw = if signed {
                f.trunc().clamp(-2_147_483_000.0, 2_147_483_000.0) as i32 as u32
            } else {
                f.trunc().clamp(0.0, 4_294_967_000.0) as u32
            };
            f32::from_bits(raw & lane_mask)
        }
        // A format convert between float widths is value-preserving; this register file holds
        // one f32 per lane and carries no packing, so it is the identity here.
        Op::Pack { .. } => s(0, c)?,
        _ => return Err("unmodeled"),
    })
}

/// Interpret a shader against `regs` in place, writing every op's masked destination lanes.
/// Hard-fails (leaving `regs` partially updated) on the first op it does not model, naming
/// it - the reference never fabricates a value for an unestablished op.
pub fn run(shader: &Shader, regs: &mut RegFile) -> Result<(), InterpError> {
    run_watching_for_nan(shader, regs).map(|_| ())
}

/// How the interpreter obtains a texture sample: `(unit, coordinate) -> RGBA`.
///
/// `None` from the callback means "this unit is not available", which BLOCKS the run rather
/// than substituting a value - the interpreter's whole contract is that it never fabricates
/// one. The default fetcher returns `None` for every unit, so a caller that does not supply
/// textures behaves exactly as before this existed.
pub type TexFetch<'a> = &'a dyn Fn(u8, [f32; 4]) -> Option<[f32; 4]>;

/// The fetcher used when a caller supplies none: no unit is available.
fn no_textures(_unit: u8, _coord: [f32; 4]) -> Option<[f32; 4]> {
    None
}

/// Where a shader first produced a value that is not finite.
#[derive(Debug, Clone)]
pub struct NanSite {
    /// Index into the shader's (unrolled) instruction stream.
    pub index: usize,
    pub op: &'static str,
    /// The destination the non-finite value landed in, as `bank[index]`.
    pub dest: String,
    /// The channel that went bad and the value it took.
    pub channel: usize,
    pub value: f32,
    /// The source register values that produced it, in operand order.
    pub sources: Vec<String>,
}

/// Interpret like [`run`], additionally returning the FIRST instruction to write a non-finite
/// value (NaN or an infinity) into a destination whose inputs were all finite.
///
/// A vertex program whose clip position comes out NaN draws nothing at all, and the frame is
/// then indistinguishable from a shader that painted black, a depth rejection, or a draw that
/// was never submitted. Every one of those was ruled out by a separate whole-title replay on a
/// title whose track surface had gone missing; the instruction that actually did it is one
/// interpreted run away, and only if something reports it. Infinities count as well as NaNs
/// because the usual route to a NaN is `0 * inf`, and the infinity is the earlier and more
/// diagnostic event - a reciprocal of a uniform the guest left zero.
pub fn run_watching_for_nan(
    shader: &Shader,
    regs: &mut RegFile,
) -> Result<Option<NanSite>, InterpError> {
    run_watching_for_nan_with_textures(shader, regs, &no_textures)
}

/// [`run_watching_for_nan`], with texture sampling available.
///
/// A vertex program that SAMPLES builds its geometry out of what it reads, so without this the
/// interpreter cannot run one at all - and everything that depends on interpreting a vertex
/// program goes blind on exactly those draws: the clip-`w` sign measurement, the depth fit, the
/// NaN hunt. On this title that blindness is not academic - the campaign map's whole body is
/// one such draw.
pub fn run_watching_for_nan_with_textures(
    shader: &Shader,
    regs: &mut RegFile,
    tex: TexFetch<'_>,
) -> Result<Option<NanSite>, InterpError> {
    let mut site: Option<NanSite> = None;
    for (index, instr) in shader.instrs.iter().enumerate() {
        if let Some(reason) = instr.blocked {
            return Err(InterpError::Blocked { index, reason });
        }
        if !instr.op.is_emittable() {
            return Err(InterpError::UnsupportedOp { index, op: instr.op.mnemonic() });
        }
        // A no-op (phase declaration / NOP) has no register effect.
        if matches!(instr.op, Op::Nop) {
            continue;
        }
        // A predicated instruction executes only when its predicate register holds.
        match instr.pred {
            Predicate::Always => {}
            Predicate::IfP(n) if regs.p[(n & 3) as usize] => {}
            Predicate::IfNotP(n) if !regs.p[(n & 3) as usize] => {}
            Predicate::IfP(_) | Predicate::IfNotP(_) => continue,
            Predicate::Raw(_) => return Err(InterpError::Blocked { index, reason: "unresolved predicate encoding" }),
        }
        let Some(dest) = instr.dest.as_ref() else {
            return Err(InterpError::OutOfRange { index });
        };
        // Loading an INDEX register is a write to the index file, not to a bank, so it too
        // sits outside the per-channel evaluator. The source lane holds an integer BIT
        // PATTERN (a truncating convert followed by a 16-bit shift produced it), and the
        // interpreter's lanes are f32 - so it is read through the same `& 0xffff` the emitter
        // applies, over the lane's bits rather than its float value.
        if let Op::LoadIndex { addend } = instr.op {
            let s1 = instr.srcs.first().ok_or(InterpError::OutOfRange { index })?;
            let bank = regs.bank(s1.bank).ok_or(InterpError::OutOfRange { index })?;
            let raw = bank.get(s1.index as usize).ok_or(InterpError::OutOfRange { index })?.to_bits();
            regs.idx[(dest.index & 1) as usize] = (raw & 0xffff) as i32 + addend;
            continue;
        }
        // A texture sample produces four components from ONE fetch, so it cannot go through
        // the per-channel evaluator below. The result lands in `dest.index + 0..4`, matching
        // the decoder's direct destination rule.
        if let Op::Tex { unit, coords, .. } = instr.op {
            // The result goes to four CONSECUTIVE lanes whatever the precision. On the hardware
            // an F16 sample lands as two packed pairs, but this register file has no packing
            // ANYWHERE - `read_channel` reads `index + selector` for every operand regardless
            // of width - so writing four lanes is the consistent thing rather than a special
            // case, and a half-precision sample is no less modelled here than a half-precision
            // multiply already is.
            let src = instr.srcs.first().ok_or(InterpError::OutOfRange { index })?;
            let mut coord = [0.0f32; 4];
            for k in 0..(coords as usize).clamp(1, 4) {
                coord[k] = read_channel(regs, src, k).ok_or(InterpError::OutOfRange { index })?;
            }
            let rgba = tex(unit, coord)
                .ok_or(InterpError::UnsupportedOp { index, op: "tex (no texture bound for this unit)" })?;
            let base = dest.index as usize;
            let bank = regs.bank_mut(dest.bank).ok_or(InterpError::OutOfRange { index })?;
            for (k, v) in rgba.iter().enumerate() {
                match bank.get_mut(base + k) {
                    Some(slot) => *slot = *v,
                    None => return Err(InterpError::OutOfRange { index }),
                }
            }
            continue;
        }
        // Compute every masked channel from the CURRENT register state first, so an in-place
        // op that reads and writes the same register uses pre-write inputs (USSE semantics).
        let mut out = [0.0f32; 4];
        for c in 0..4 {
            if instr.write_mask[c] {
                out[c] = eval_channel(regs, instr, c).map_err(|op| InterpError::UnsupportedOp { index, op })?;
            }
        }
        // Read the sources BEFORE the write, so the report shows what went in - an in-place op
        // would otherwise print its own result back as its input.
        if site.is_none() {
            if let Some(c) = (0..4).find(|&c| instr.write_mask[c] && !out[c].is_finite()) {
                let sources = instr
                    .srcs
                    .iter()
                    .map(|s| {
                        let vals: Vec<String> = (0..4)
                            .map(|k| match regs.bank(s.bank).and_then(|b| b.get(s.index as usize + k)) {
                                Some(v) => format!("{v}"),
                                None => "-".to_string(),
                            })
                            .collect();
                        format!("{:?}[{}]={:?}", s.bank, s.index, vals)
                    })
                    .collect();
                site = Some(NanSite {
                    index,
                    op: instr.op.mnemonic(),
                    dest: format!("{:?}[{}]", dest.bank, dest.index),
                    channel: c,
                    value: out[c],
                    sources,
                });
            }
        }
        let base = dest.index as usize;
        let bank = regs.bank_mut(dest.bank).ok_or(InterpError::OutOfRange { index })?;
        for c in 0..4 {
            if instr.write_mask[c] {
                *bank.get_mut(base + c).ok_or(InterpError::OutOfRange { index })? = out[c];
            }
        }
    }
    Ok(site)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::ProgramKind;
    use crate::ir::{Bank, Instr, Op, Operand, Predicate};

    fn instr(op: Op, dest: Operand, srcs: Vec<Operand>) -> Instr {
        Instr {
            op,
            pred: Predicate::Always,
            dest: Some(dest),
            write_mask: [true; 4],
            srcs,
            half_precision: false,
            raw: 0,
            group: 0,
            blocked: None,
        }
    }

    fn shader(instrs: Vec<Instr>) -> Shader {
        Shader { kind: ProgramKind::Fragment, instrs }
    }

    #[test]
    fn mad_computes_a_times_b_plus_c() {
        // r[0..4] = pa[0..4] * sa[0..4] + r[4..8], per channel.
        let mut regs = RegFile::with_lanes(8);
        regs.pa[0..4].copy_from_slice(&[2.0, 3.0, 4.0, 5.0]);
        regs.sa[0..4].copy_from_slice(&[10.0, 10.0, 10.0, 10.0]);
        regs.r[4..8].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let d = Operand::plain(Bank::Temp, 0, 0);
        let a = Operand::plain(Bank::PrimaryAttr, 0, 2);
        let b = Operand::plain(Bank::SecondaryAttr, 0, 3);
        let cc = Operand::plain(Bank::Temp, 4, 0);
        run(&shader(vec![instr(Op::Mad, d, vec![a, b, cc])]), &mut regs).unwrap();
        assert_eq!(&regs.r[0..4], &[21.0, 31.0, 41.0, 51.0]);
    }

    #[test]
    fn dot_broadcasts_scalar_and_constant_resolves() {
        // o[0] = dot3(pa.xyz, const 1.0) = pa.x + pa.y + pa.z.
        let mut regs = RegFile::with_lanes(4);
        regs.pa[0..3].copy_from_slice(&[1.0, 2.0, 3.0]);
        let d = Operand::plain(Bank::Output, 0, 1);
        let a = Operand::plain(Bank::PrimaryAttr, 0, 2);
        let k = Operand::plain(Bank::Constant, 2, 0); // CNST6[2] = 1.0
        let mut ins = instr(Op::Dot { components: 3 }, d, vec![a, k]);
        ins.write_mask = [true, false, false, false];
        run(&shader(vec![ins]), &mut regs).unwrap();
        assert_eq!(regs.o[0], 6.0);
    }

    #[test]
    fn min_max_frc_and_negate() {
        let mut regs = RegFile::with_lanes(4);
        regs.r[0] = 3.5;
        regs.r[1] = -2.0;
        // r[2] = min(r[0], 1.0const) ; using a negate modifier on src.
        let d = Operand::plain(Bank::Temp, 2, 0);
        let a = Operand::plain(Bank::Temp, 0, 0);
        let one = Operand::plain(Bank::Constant, 2, 0); // 1.0
        let mut mn = instr(Op::Min, d, vec![a, one]);
        mn.write_mask = [true, false, false, false];
        run(&shader(vec![mn]), &mut regs).unwrap();
        assert_eq!(regs.r[2], 1.0);
        // frc(3.5) = 0.5.
        let df = Operand::plain(Bank::Temp, 3, 0);
        let mut fr = instr(Op::Frc, df, vec![Operand::plain(Bank::Temp, 0, 0)]);
        fr.write_mask = [true, false, false, false];
        run(&shader(vec![fr]), &mut regs).unwrap();
        assert_eq!(regs.r[3], 0.5);
    }

    #[test]
    fn in_place_op_uses_pre_write_inputs() {
        // r[0] = r[0] + r[1] where dest and src0 are the same register: must use old r[0].
        let mut regs = RegFile::with_lanes(4);
        regs.r[0] = 5.0;
        regs.r[1] = 7.0;
        let d = Operand::plain(Bank::Temp, 0, 0);
        let a = Operand::plain(Bank::Temp, 0, 0);
        let b = Operand::plain(Bank::Temp, 1, 0);
        let mut add = instr(Op::Add, d, vec![a, b]);
        add.write_mask = [true, false, false, false];
        run(&shader(vec![add]), &mut regs).unwrap();
        assert_eq!(regs.r[0], 12.0);
    }

    #[test]
    fn transcendentals_and_move() {
        // rsq(4)=0.5, exp2(3)=8, log2(8)=3, rcp(2)=0.5, mov copies.
        let mut regs = RegFile::with_lanes(4);
        regs.r[0] = 4.0;
        let d = |n| Operand::plain(Bank::Temp, n, 0);
        let src = |n| Operand::plain(Bank::Temp, n, 0);
        let scalar = |op, dn, sn| {
            let mut i = instr(op, d(dn), vec![src(sn)]);
            i.write_mask = [true, false, false, false];
            i
        };
        run(&shader(vec![scalar(Op::Rsq, 1, 0)]), &mut regs).unwrap();
        assert_eq!(regs.r[1], 0.5);
        regs.r[0] = 3.0;
        run(&shader(vec![scalar(Op::Exp, 2, 0)]), &mut regs).unwrap();
        assert_eq!(regs.r[2], 8.0);
        run(&shader(vec![scalar(Op::Log, 3, 2)]), &mut regs).unwrap();
        assert_eq!(regs.r[3], 3.0);
        run(&shader(vec![scalar(Op::Mov, 0, 3)]), &mut regs).unwrap();
        assert_eq!(regs.r[0], 3.0);
    }

    #[test]
    fn unmodeled_op_hard_fails_naming_it() {
        let mut regs = RegFile::with_lanes(4);
        let d = Operand::plain(Bank::Temp, 0, 0);
        let ins = instr(Op::Illegal, d, vec![Operand::plain(Bank::Temp, 1, 0)]);
        let err = run(&shader(vec![ins]), &mut regs).unwrap_err();
        assert!(matches!(err, InterpError::UnsupportedOp { op: "illegal", .. }));
    }

    #[test]
    fn predicated_instruction_gates_on_predicate() {
        use crate::ir::Predicate;
        let mut regs = RegFile::with_lanes(8);
        regs.r[2] = 3.0;
        regs.r[4] = 5.0;
        let mut ins = instr(Op::Add, Operand::plain(Bank::Output, 0, 1),
            vec![Operand::plain(Bank::Temp, 2, 0), Operand::plain(Bank::Temp, 4, 0)]);
        ins.pred = Predicate::IfP(1);
        ins.write_mask = [true, false, false, false];
        // p1 false -> the write is skipped, o[0] stays 0.
        run(&shader(vec![ins.clone()]), &mut regs).unwrap();
        assert_eq!(regs.o[0], 0.0);
        // p1 true -> the write executes.
        regs.p[1] = true;
        run(&shader(vec![ins]), &mut regs).unwrap();
        assert_eq!(regs.o[0], 8.0);
    }

    #[test]
    fn cmov_selects_on_zero_test() {
        use crate::ir::CompareMethod;
        // r[2] = (r[3] < 0) ? r[0] : r[1], per channel. src order [src1(true), src2(false), src0(test)].
        let mut regs = RegFile::with_lanes(4);
        regs.r[0] = 10.0; // true value
        regs.r[1] = 20.0; // false value
        regs.r[3] = -1.0; // test < 0 -> true -> pick r[0]
        let d = Operand::plain(Bank::Temp, 2, 0);
        let mut ins = instr(
            Op::Cmov { test: CompareMethod::LtZero },
            d,
            vec![
                Operand::plain(Bank::Temp, 0, 0),
                Operand::plain(Bank::Temp, 1, 0),
                Operand::plain(Bank::Temp, 3, 0),
            ],
        );
        ins.write_mask = [true, false, false, false];
        run(&shader(vec![ins.clone()]), &mut regs).unwrap();
        assert_eq!(regs.r[2], 10.0);
        // Flip the test to positive -> pick the false value r[1].
        regs.r[3] = 5.0;
        run(&shader(vec![ins]), &mut regs).unwrap();
        assert_eq!(regs.r[2], 20.0);
    }
}
