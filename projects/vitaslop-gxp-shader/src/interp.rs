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

use crate::ir::{Bank, Instr, Op, Operand, Predicate, Shader, TestReduce, TexLod};
use crate::fold::f32_to_f16_bits_saturating;
use crate::wgsl::{bank_prec, cnst6_channel_value, f16_bits_to_f32, Prec};

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
    /// Did the program execute a [`Op::Kill`]? A fragment that kills ends with no colour
    /// written, so this is part of its result and not a detail of the walk - a translation that
    /// dropped the discard would leave every register agreeing and still paint a pixel the
    /// hardware does not.
    pub killed: bool,
    /// The per-fragment FACING flag the run is given, or `None` when the caller supplies none -
    /// in which case a program that reads `GLOBAL[16]` is refused rather than answered with a
    /// fabricated side.
    ///
    /// It is an INPUT, like `pa` and `sa`, not state the program computes: the hardware hands it
    /// to the fragment. Both case wrappers pin it to `true` (a compute dispatch has no facing,
    /// and the render rig draws one triangle), so a case harness pins the reference to the same
    /// value - and the back-facing arm of such a program is then not exercised by anything,
    /// which is a coverage statement the harness has to make rather than hide.
    pub facing: Option<bool>,
    /// The depth [`Op::DepthF`] wrote, in the GUEST's window encoding, or `None` when the
    /// program never wrote one.
    ///
    /// RAW, not remapped: which forward map the renderer applies on the way to
    /// `@builtin(frag_depth)` is a property of the DRAW (`gxp_depth_to_window` chooses from a
    /// per-draw uniform), so applying one here would put the renderer's state into the
    /// reference's answer. The caller that knows which map is in force applies it.
    pub frag_depth: Option<f32>,
    /// The PA lane the render rig puts a SCREEN RAMP on, or `None` for a rig without one.
    ///
    /// A reference that runs one fragment has no quad, so a derivative is 0 here - exact on a rig
    /// whose every pixel reads the same register file, and useless as an oracle: `dpdx` for
    /// `dpdy`, or a derivative of the wrong register, agrees with it. The rig that sets this adds
    /// `(pos.x - 0.5) * CASE_RAMP_DX + (pos.y - 0.5) * CASE_RAMP_DY` to that one lane, which is
    /// ZERO at the real pixel (so no value this reference reads changes) and gives the lane an
    /// exact screen gradient across the quad. [`Op::Dsx`]/[`Op::Dsy`] of a DIRECT read of that
    /// lane then has a known answer; of anything else it is still 0.
    pub ramp: Option<usize>,
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
            killed: false,
            facing: None,
            frag_depth: None,
            ramp: None,
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

/// >>> THE REGISTER FILE IS A FILE OF 32-BIT WORDS, AND ITS PRECISION IS THE INSTRUCTION'S.
///
/// A `RegFile` lane is declared `f32` but what it really holds is a 32-bit WORD - exactly like
/// the `array<u32, N>` the emitter's module declares - and the instruction reading it decides
/// how to divide that word up: one F32, two packed F16s, or four unorm bytes. This reads lane
/// selector `sel` of the register at `base` at precision `prec`, addressing registers the same
/// way [`crate::wgsl::read_lane`] does, because it must be the same addressing.
///
/// >>> AND UNTIL THIS EXISTED THE REFERENCE HAD NO AUTHORITY OVER TWO THIRDS OF THE CORPUS.
/// Every lane was read as one whole `f32`, so for an F16 instruction the reference and the
/// shipped shader were not modelling the same register file at all: where the shader reads two
/// halves of a word, the reference read the word's bit pattern as a single float - a denormal
/// or a huge number, never the value. 433 of 670 corpus programs carry half-precision
/// instructions, and that is where the F16 varying packing and the F16 lighting lanes live.
fn read_reg_lane(regs: &RegFile, bank: Bank, base: u32, sel: u32, prec: Prec) -> Option<f32> {
    let lanes = regs.bank(bank)?;
    Some(match prec {
        Prec::F32 => *lanes.get((base + sel) as usize)?,
        // Two halves per word: the selector's high bits pick the register, its low bit the half.
        Prec::F16 => {
            let word = lanes.get((base + (sel >> 1)) as usize)?.to_bits();
            let half = if sel & 1 == 1 { (word >> 16) as u16 } else { word as u16 };
            f16_bits_to_f32(half)
        }
        // All four channels live in ONE register, so the selector picks a BYTE and never a
        // neighbouring register the way the two float widths do.
        Prec::Fx8 => {
            let word = lanes.get(base as usize)?.to_bits();
            ((word >> (8 * sel)) & 0xff) as f32 / 255.0
        }
    })
}

/// Write `value` into channel `c` of the destination register at `base`, at precision `prec` -
/// the inverse of [`read_reg_lane`], and the same addressing the emitter's store path uses.
///
/// The narrow widths are READ-MODIFY-WRITE: an F16 channel is half of a word whose other half
/// belongs to a different channel, and a byte channel is one quarter of one. Writing a whole
/// lane for either would destroy the neighbouring channels - which is the same class of defect
/// as a write mask that claims lanes it does not own.
fn write_reg_lane(lanes: &mut [f32], base: u32, c: u32, prec: Prec, value: f32) -> Option<()> {
    match prec {
        Prec::F32 => *lanes.get_mut((base + c) as usize)? = value,
        Prec::F16 => {
            let slot = lanes.get_mut((base + (c >> 1)) as usize)?;
            let word = slot.to_bits();
            // SATURATING, because that is what the store does: an overflow to infinity would
            // propagate into a NaN at the next subtraction and poison the whole program.
            let half = f32_to_f16_bits_saturating(value) as u32;
            let merged = if c & 1 == 1 { (word & 0x0000_ffff) | (half << 16) } else { (word & 0xffff_0000) | half };
            *slot = f32::from_bits(merged);
        }
        Prec::Fx8 => {
            let slot = lanes.get_mut(base as usize)?;
            let word = slot.to_bits();
            let byte = ((value.clamp(0.0, 1.0) * 255.0) + 0.5) as u32 & 0xff;
            let shift = 8 * c;
            *slot = f32::from_bits((word & !(0xffu32 << shift)) | (byte << shift));
        }
    }
    Some(())
}

/// Read one source channel `c` of `op` from `regs`, applying swizzle + abs/neg (or the
/// inline constant). `None` on an unmapped bank / out-of-range lane.
///
/// Precision-free entry point, for the integer/pointer/coordinate readers below: each of those
/// reads a whole 32-bit word by design (an address, a raw lane), which is [`Prec::F32`]'s
/// addressing. Arithmetic goes through [`read_channel_prec`] instead.
fn read_channel(regs: &RegFile, op: &Operand, c: usize) -> Option<f32> {
    read_channel_prec(regs, op, c, Prec::F32)
}

/// [`read_channel`], told the PRECISION the reading instruction addresses its sources at - what
/// divides each 32-bit register word into channels, and which view of the hardware constant
/// table applies. See [`read_reg_lane`] and [`cnst6_channel_value`].
fn read_channel_prec(regs: &RegFile, op: &Operand, c: usize, prec: Prec) -> Option<f32> {
    let half = matches!(prec, Prec::F16);
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
        // The CHANNEL's swizzle selector chooses the table, not just the lane: selectors 4..7
        // are the inline constants and selector 1 is the second F32 bank. Reading the index
        // alone made this the interpreter's single largest source of wrong numbers.
        cnst6_channel_value(op.index, op.swizzle[c], half, matches!(prec, Prec::Fx8))?
    } else if matches!(op.bank, Bank::Global) {
        // >>> THE ONE ESTABLISHED HARDWARE GLOBAL: the per-fragment FACING flag, read as
        // `GLOBAL[16] & 1`. The emitter spells it `select(0u, 1u, gxp_front_facing)`, so it is a
        // RAW BIT PATTERN in an integer context - which is how this register file carries it,
        // like every other raw lane here.
        //
        // Any other global index is pipeline state the interpreter holds nothing for, and it
        // stays refused: a fabricated value would be a number the hardware never produced, and
        // the emitter hard-fails on those too rather than guessing.
        if op.index != crate::wgsl::GLOBAL_FACING {
            return None;
        }
        f32::from_bits(u32::from(regs.facing?))
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
            0..=3 => read_reg_lane(regs, op.bank, op.index as u32, sel as u32, bank_prec(op.bank, prec))?,
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
    // The precision an instruction READS at is not always the one it WRITES at - a format
    // convert exists precisely so the two differ - so the source precision comes from the
    // emitter's own rule rather than from `half_precision` directly.
    let sp = Prec::src_of(instr);
    let s = |n: usize, ch: usize| {
        instr.srcs.get(n).and_then(|o| read_channel_prec(regs, o, ch, sp)).ok_or("operand")
    };
    Ok(match instr.op {
        Op::Mul => s(0, c)? * s(1, c)?,
        Op::Add => s(0, c)? + s(1, c)?,
        Op::Min => s(0, c)?.min(s(1, c)?),
        Op::Max => s(0, c)?.max(s(1, c)?),
        // >>> `x - floor(x)`, NOT Rust's `f32::fract`. Rust's is `x - trunc(x)`, which KEEPS THE
        // SIGN: `(-3.7).fract()` is `-0.7`, where WGSL's `fract` - what the emitter emits and
        // what the GPU runs - gives `0.3`. For any negative operand the two differ by exactly
        // 1.0, and a fractional part is normally the range reduction in front of a polynomial,
        // so that 1.0 is squared and scaled into a completely different number. The corpus-wide
        // differential found it on seven 99-instruction mk vertex programs whose outputs were
        // ~10^4 times the GPU's.
        Op::Frc => {
            let v = s(0, c)?;
            v - v.floor()
        }
        // Screen-space derivatives require a pixel quad; 0 in a single-pixel reference - except
        // of the rig's RAMP lane, read directly at full precision, whose gradient the rig
        // states (see `RegFile::ramp`). A negated read differentiates to the negated step.
        Op::Dsx | Op::Dsy => {
            let src = instr.srcs.first().ok_or("operand")?;
            let sel = src.swizzle[c] as usize;
            let hit = regs.ramp.is_some_and(|lane| {
                src.bank == Bank::PrimaryAttr
                    && !instr.half_precision
                    && !src.abs
                    && sel < 4
                    && src.index as usize + sel == lane
            });
            let step = if !hit {
                0.0
            } else if matches!(instr.op, Op::Dsx) {
                crate::wgsl::CASE_RAMP_DX
            } else {
                crate::wgsl::CASE_RAMP_DY
            };
            // `x - x` is +0 in round-to-nearest whatever the sign of `x`, so a negated read of a
            // lane with no gradient is still +0 - only a real step takes the sign.
            if src.neg && hit { -step } else { step }
        }
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
        // Move and float<->float pack are swizzled copies. A format convert between float
        // widths is value-preserving, and this register file holds one f32 per lane and
        // carries no packing, so `Pack` is the identity here.
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
        // Group 0x15 IMAD32, scalar on channel 0: the lane holds an integer's BIT PATTERN, the
        // same representation the bitwise ops above read and write.
        //
        // An IMMEDIATE source is read as the INTEGER its number spells, not through
        // `read_channel` - that helper yields a literal as an f32 (`48.0`), whose bit pattern is
        // not 48. The emitter materialises the same literal as `48u`, so reading it any other
        // way here would make this reference disagree with the code that ships, which is the one
        // thing an oracle may never do.
        Op::IntMad { signed, bits, src0_high, src1_high } => {
            if bits != 32 {
                return Err("imad (only the 32-bit width is established)");
            }
            let raw = |n: usize| -> Result<u32, &'static str> {
                let o = instr.srcs.get(n).ok_or("operand")?;
                if matches!(o.bank, Bank::Immediate) {
                    return Ok(o.index as u32);
                }
                Ok(read_channel(regs, o, 0).ok_or("operand")?.to_bits())
            };
            let (a0, b0, d) = (raw(0)?, raw(1)?, raw(2)?);
            // `src1_high` picks a half of src1 the same way - see the decoder's note. A clear
            // bit reads the whole register, matching the emitter exactly.
            let b = match (signed, src1_high) {
                (_, false) => b0,
                (true, true) => ((b0 as i32) >> 16) as u32,
                (false, true) => b0 >> 16,
            };
            // src0 is one HALF of a packed pair - the same widening the emitter does, so this
            // oracle and the code that ships agree about which value the multiply sees.
            let a = match (signed, src0_high) {
                (true, true) => ((a0 as i32) >> 16) as u32,
                (true, false) => (((a0 as i32) << 16) >> 16) as u32,
                (false, true) => a0 >> 16,
                (false, false) => a0 & 0xffff,
            };
            let r = if signed {
                ((a as i32).wrapping_mul(b as i32).wrapping_add(d as i32)) as u32
            } else {
                a.wrapping_mul(b).wrapping_add(d)
            };
            f32::from_bits(r)
        }
        // Group 0x1a, ONE STEP of a 32-bit integer multiply-add. Same lane representation as
        // the sibling group above - the lane holds the integer's bit pattern - and the same
        // rule for an inline literal. `high_half` picks which 16-bit half of src0 the
        // multiplier sees and whether its product is shifted back up; the two steps of a pair
        // sum to the whole product in wrapping 32-bit arithmetic, which is what the emitter
        // writes and what this must therefore agree with.
        Op::IntMadStep { signed, high_half } => {
            // >>> THE SIGNED FORM NEEDED NO EXTRA FACT, AND THE EMITTER SAYS SO.
            //
            // This refused with "only the unsigned form is established" - 13 corpus programs -
            // while `emit_int_mad_step` decodes `signed` and DELIBERATELY IGNORES IT, because
            // two's-complement multiplication agrees on the low 32 bits for signed and
            // unsigned operands alike and the `<< 16` of the high step discards everything
            // above bit 15 anyway. The wrapping arithmetic below is already exactly what that
            // emitter writes for both, so refusing one of them withheld an oracle from
            // programs it could always have judged.
            let _ = signed;
            let raw = |n: usize| -> Result<u32, &'static str> {
                let o = instr.srcs.get(n).ok_or("operand")?;
                if matches!(o.bank, Bank::Immediate) {
                    return Ok(o.index as u32);
                }
                Ok(read_channel(regs, o, 0).ok_or("operand")?.to_bits())
            };
            let (a, b, d) = (raw(0)?, raw(1)?, raw(2)?);
            let part = if high_half {
                (a >> 16).wrapping_mul(b) << 16
            } else {
                (a & 0xffff).wrapping_mul(b)
            };
            f32::from_bits(part.wrapping_add(d))
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
        // >>> THE INTEGER->FLOAT WIDEN IS MODELLED NOW, and it is the last of the family whose
        // >>> refusal was about the old register file rather than about the instruction.
        //
        // It refused because "this file holds one f32 per lane with nowhere to put the other
        // half", and added that widening the whole lane instead would make the oracle agree
        // with an emitter that had the half selection wrong. The first half is stale - a lane
        // is a 32-bit word here and `Prec::F16` has been splitting one for several sessions -
        // and the second is answered rather than ignored: the element is addressed by the
        // operand's OWN component selector across `32 / bits` elements per register, which is
        // the decoder's swizzle rule, the rule `Op::PackIntCopy` above moves by, and the rule
        // `store_raw_half` writes by. A value this recompiler stored through that path reads
        // back as itself, which is the property that makes the pair checkable at all.
        //
        // The remaining exposure is honest and worth naming: both sides take that addressing
        // from the same decoder, so a WRONG selector rule would agree with itself here. The
        // AUTHORED conformance suite is what can break that tie, not this file.
        Op::PackFromInt { bits, signed } => {
            let bits = bits as u32;
            if bits != 16 && bits != 8 {
                return Err("unpack.int at a width other than 8 or 16");
            }
            let s1 = instr.srcs.first().ok_or("operand")?;
            let per_reg = 32 / bits;
            let sel = s1.swizzle[c] as u32;
            let word = regs
                .bank(s1.bank)
                .and_then(|b| b.get((s1.index as u32 + sel / per_reg) as usize))
                .ok_or("operand")?
                .to_bits();
            let lo = (sel % per_reg) * bits;
            let raw = (word >> lo) & ((1u32 << bits) - 1);
            // Sign extension by the same shift pair `emit_int_mad` uses on a packed operand -
            // those two already agree on how a narrow element is widened, and a third rule
            // here would be a third answer.
            let v = if signed {
                ((raw << (32 - bits)) as i32 >> (32 - bits)) as f32
            } else {
                raw as f32
            };
            let v = if s1.neg { -v } else { v };
            if s1.abs {
                v.abs()
            } else {
                v
            }
        }
        // >>> THE 8-BIT COMBINER IS MODELLED NOW, and as with the unorm convert beside it what
        // >>> changed is the REGISTER FILE, not the instruction.
        //
        // It used to refuse with "this register file is one f32 per lane, not four bytes", and
        // that was true when written. [`read_reg_lane`]/[`write_reg_lane`] have carried a
        // `Prec::Fx8` view since - four unsigned-normalised bytes in one register - and this
        // family is precisely what that view is FOR: unlike `cmov.u8`, whose bytes are a raw
        // index, these bytes are a colour, so `byte / 255` in and `round(v * 255)` out is the
        // value the pipeline computes with.
        //
        // The two terms, per channel, are `coeff * src` with the RGB and ALPHA operations
        // independent over the same pair - see [`Op::Sop2`]. The coefficient is the SELECTOR,
        // not the operand, which is the reading that took several sessions to settle: `Zero`
        // complemented is the coefficient 1, making that term a plain copy of its source. An
        // alpha selector broadcasts channel 3.
        Op::Sop2 { color, alpha, f1, f1_complement, f2, f2_complement } => {
            use crate::ir::{SopFactor, SopOp};
            let (s1, s2) = (instr.srcs.first().ok_or("operand")?, instr.srcs.get(1).ok_or("operand")?);
            let fx = |o: &Operand, ch: usize| -> Result<f32, &'static str> {
                read_channel_prec(regs, o, ch, Prec::Fx8).ok_or("operand")
            };
            let coeff = |f: SopFactor, complement: bool| -> Result<f32, &'static str> {
                let base = match f {
                    SopFactor::Zero => 0.0,
                    SopFactor::Src1Color => fx(s1, c)?,
                    SopFactor::Src1Alpha => fx(s1, 3)?,
                    SopFactor::Src2Color => fx(s2, c)?,
                    SopFactor::Src2Alpha => fx(s2, 3)?,
                };
                Ok(if complement { 1.0 - base } else { base })
            };
            let t1 = coeff(f1, f1_complement)? * fx(s1, c)?;
            let t2 = coeff(f2, f2_complement)? * fx(s2, c)?;
            match if c == 3 { alpha } else { color } {
                SopOp::Add => t1 + t2,
                SopOp::Sub => t1 - t2,
                SopOp::Min => t1.min(t2),
                SopOp::Max => t1.max(t2),
            }
        }
        // >>> THE NORMALIZED U8 CONVERT AND THE 8-BIT MOVE ARE MODELLED NOW, AND WHAT CHANGED
        // >>> IS THE REGISTER FILE, NOT THIS INSTRUCTION.
        //
        // Both used to refuse with "this register file is one f32 per lane, not four bytes",
        // and that was true when it was written: there was nowhere to put four bytes, and the
        // note added that interpreting the convert as the IDENTITY would make the oracle agree
        // with an emitter that had the scaling wrong. Both halves of that reasoning are now
        // stale, and the second is the interesting one.
        //
        // [`read_reg_lane`] and [`write_reg_lane`] carry a `Prec::Fx8` view - four
        // unsigned-normalised bytes in one register - and [`Prec::of`] / [`Prec::src_of`]
        // already route exactly these two operations through it, in the correct direction each.
        // So the NORMALIZATION is not the identity here: it happens in the lane accessors, as
        // `byte / 255` on the way in and `round(value * 255)` on the way out, written in this
        // file and shared with the emitter only through the rule they both implement - which is
        // what an oracle needs. The per-channel VALUE is then genuinely a copy, because the
        // whole operation is the change of storage view.
        //
        // The two directions are already distinguished for us: `to_unorm8` picks which end is
        // the byte register, and the precision rule reads it. There is nothing left for this
        // arm to decide.
        //
        // MEASURED: 19 of the corpus's 104 interpreter refusals were `pack.unorm8` and 3 were
        // `mov.fx8` - together a fifth of the shader code that ships with no oracle at all.
        Op::PackUnorm8 { .. } | Op::CopyFx8 => s(0, c)?,
        // A memory load reads GUEST MEMORY, which this register-file model does not hold; a
        // fabricated value here would defeat the oracle's whole purpose.
        Op::MemLoad { .. } => return Err("ldmem (resolved before the per-channel evaluator)"),
        // A TEST with write-back also stores its raw ALU result; the predicate itself was
        // written before this point. Only the float families reach here - the raw-lane ones
        // have no float result to store and the emitter refuses them too.
        Op::Test { alu, .. } => {
            // The 8-bit family's write-back stores its BYTE difference, read and written
            // through the unorm view - the same split `emit_test`'s write-back makes with its
            // own `wp`. `Prec::of` says Fx8 for this instruction, so the store lands in the
            // right byte of the right register without this arm having to say so again.
            if matches!(alu, crate::ir::TestAlu::Fx8Sub) {
                let a = read_channel_prec(regs, instr.srcs.first().ok_or("operand")?, c, Prec::Fx8)
                    .ok_or("operand")?;
                let b = read_channel_prec(regs, instr.srcs.get(1).ok_or("operand")?, c, Prec::Fx8)
                    .ok_or("operand")?;
                return Ok(a - b);
            }
            let (a, b) = (s(0, c)?, s(1, c)?);
            match alu {
                crate::ir::TestAlu::Add => a + b,
                crate::ir::TestAlu::Sub => a - b,
                crate::ir::TestAlu::Mul => a * b,
                _ => return Err("vtst write-back on a raw-lane family"),
            }
        }
        // VTSTMSK: the same compare, written out as one value per channel - NUMERIC for the
        // float families, a raw bit-pattern mask for the unsigned 16-bit integer one.
        Op::TestMask { alu, cmp } => {
            use crate::ir::{TestAlu, TestCmp};
            // >>> THE UNSIGNED 16-BIT FORM WRITES BITS, NOT A NUMBER.
            //
            // This register file holds `f32`, and its raw-lane readers already work in the
            // lane's BITS (`to_bits`), so the mask is stored the same way round: the lane ends
            // up holding the pattern 0x0000FFFF, which is what a later raw-lane read of it
            // sees. Writing 65535.0 instead would agree with nothing - not with the device,
            // not with the emitted WGSL, which uses `store_raw` for exactly this reason.
            if matches!(alu, TestAlu::IntSub16U) {
                // Channel x only; the decoder's write mask says so, and any other channel of
                // this destination is not written by this instruction at all.
                if c != 0 {
                    return Err("vtstmsk u16 writes channel x only");
                }
                let raw = |i: usize| -> Result<u32, &'static str> {
                    Ok(s(i, 0)?.to_bits())
                };
                let held = (raw(0)? & 0xffff) == (raw(1)? & 0xffff);
                let held = match cmp {
                    TestCmp::Eq => held,
                    TestCmp::Ne => !held,
                    // An ordered relation would need the compare's signedness pinned, which
                    // no source establishes - the decoder refuses these before here.
                    _ => return Err("vtstmsk u16 with an ordered relation"),
                };
                return Ok(f32::from_bits(if held { 0xffff } else { 0 }));
            }
            let (a, b) = (s(0, c)?, s(1, c)?);
            let v = match alu {
                TestAlu::Add => a + b,
                TestAlu::Sub => a - b,
                TestAlu::Mul => a * b,
                _ => return Err("vtstmsk on a raw-lane family"),
            };
            let held = match cmp {
                TestCmp::Eq => v == 0.0,
                TestCmp::Ne => v != 0.0,
                TestCmp::Lt => v < 0.0,
                TestCmp::Le => v <= 0.0,
                TestCmp::Gt => v > 0.0,
                TestCmp::Ge => v >= 0.0,
            };
            if held {
                1.0
            } else {
                0.0
            }
        }
        // NAME THE OP. A placeholder reason is an uninstrumented path: a census of refusals that
        // says "unmodeled" 49 times tells a reader nothing about what to implement next, while
        // the mnemonic turns the same census into a ranked work list.
        _ => return Err(instr.op.mnemonic()),
    })
}

/// Interpret a shader against `regs` in place, writing every op's masked destination lanes.
/// Hard-fails (leaving `regs` partially updated) on the first op it does not model, naming
/// it - the reference never fabricates a value for an unestablished op.
pub fn run(shader: &Shader, regs: &mut RegFile) -> Result<(), InterpError> {
    run_watching_for_nan(shader, regs).map(|_| ())
}

/// How the interpreter obtains a texture sample: `(unit, coordinate, lod) -> RGBA`.
///
/// `None` from the callback means "this unit is not available", which BLOCKS the run rather
/// than substituting a value - the interpreter's whole contract is that it never fabricates
/// one. The default fetcher returns `None` for every unit, so a caller that does not supply
/// textures behaves exactly as before this existed.
///
/// The LOD argument is what the instruction says about the mip level. A real texture's answer
/// depends on it; a caller modelling one mip may ignore it. It is passed rather than dropped
/// because the differential's stand-in folds it into the result - otherwise nothing could tell
/// a `textureSampleBias` from a `textureSampleLevel`, or see the level read from the wrong
/// register.
pub type TexFetch<'a> = &'a dyn Fn(u8, [f32; 4], TexLodArg) -> Option<[f32; 4]>;

/// A sample's LOD operand, as the instruction supplies it: the mode, and the F32 channels of
/// `src2` that mode reads - channel 0 for a bias or a level, channels 0..3 (`ddx.xy`, `ddy.xy`)
/// for a 2D gradient. Channels the mode does not read are zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TexLodArg {
    pub mode: TexLod,
    pub args: [f32; 4],
}

impl TexLodArg {
    /// An implicit-LOD sample: no operand.
    pub const IMPLICIT: TexLodArg = TexLodArg { mode: TexLod::Implicit, args: [0.0; 4] };

    /// How many channels of `src2` this mode reads.
    pub fn channels(mode: TexLod) -> usize {
        match mode {
            TexLod::Implicit => 0,
            TexLod::Bias | TexLod::Level => 1,
            TexLod::Gradient => 4,
        }
    }
}

/// The fetcher used when a caller supplies none: no unit is available.
fn no_textures(_unit: u8, _coord: [f32; 4], _lod: TexLodArg) -> Option<[f32; 4]> {
    None
}

/// How the interpreter obtains a texture GATHER: `(unit, uv) -> (four texels, two fractions)`.
///
/// The four values are the 2x2 footprint in the platform's order - the texels at `(x0,y1)`,
/// `(x1,y1)`, `(x1,y0)`, `(x0,y0)` - and the fractions are `fract(uv * dims - 0.5)`, the pair a
/// bilinear filter of that same footprint would weight with.
///
/// >>> WHY THE FOOTPRINT IS THE ENVIRONMENT'S AND NOT THIS FILE'S. Which texels a coordinate
/// names depends on the texture's SIZE and the sampler's addressing, which are properties of
/// the bound unit, not of the instruction. What [`Op::TexGather`] means - four texels into four
/// consecutive registers and four bilinear coefficients into the two above them, paired in
/// reverse - is the ISA fact, and that is what this file models. The same split the ordinary
/// [`TexFetch`] makes.
pub type GatherFetch<'a> = &'a dyn Fn(u8, [f32; 2]) -> Option<([f32; 4], [f32; 2])>;

/// The texture environment a run is given: an ordinary sample, and optionally a gather.
///
/// A caller that supplies no gather leaves [`Op::TexGather`] uninterpretable, naming itself,
/// rather than fabricating a footprint out of four copies of one filtered sample - which would
/// make a shadow filter look as though it worked.
pub struct TexEnv<'a> {
    pub sample: TexFetch<'a>,
    pub gather: Option<GatherFetch<'a>>,
}

impl<'a> TexEnv<'a> {
    /// The environment for a caller that has only an ordinary fetcher.
    pub fn sampling(sample: TexFetch<'a>) -> TexEnv<'a> {
        TexEnv { sample, gather: None }
    }
}

/// How the interpreter reads GUEST MEMORY for a 0xE8 load: `address -> the 32-bit word there`.
///
/// The caller supplies the same WINDOWS the shader is bound (see
/// `vitaslop_gxp_shader::module::MemWindow`), so an address inside one reads the bytes the draw
/// carried and an address outside every one reads ZERO - which is what the emitted
/// `gxp_mem_word` helper does, and an oracle may not disagree with the code that ships.
///
/// A caller that supplies NONE leaves a program with memory loads uninterpretable, naming
/// itself, rather than reading zeroes it was never given: those zeroes would be a fabricated
/// vertex position, and the whole point of this interpreter is that it never fabricates one.
pub type MemFetch<'a> = &'a dyn Fn(u32) -> u32;

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
    run_watching_for_nan_with_env(shader, regs, tex, None)
}

/// [`run_watching_for_nan_with_textures`], with the draw's GUEST-MEMORY WINDOWS available too.
///
/// A vertex program that chases a bound uniform buffer with 0xE8 loads builds its geometry out
/// of what it reads, so without this the interpreter cannot run one at all - and everything
/// that depends on interpreting a vertex program goes blind on exactly those draws: the clip-`w`
/// sign measurement, the depth fit, the NaN hunt. On a golf title that blindness decided a whole
/// 960x544 world pass's projection from the two 2D overlay quads that DID interpret, left the
/// negative-projection correction off, and WebGPU clipped every one of the world's 241 draws
/// away - a black frame with no fallback and nothing in any log.
pub fn run_watching_for_nan_with_env(
    shader: &Shader,
    regs: &mut RegFile,
    tex: TexFetch<'_>,
    mem: Option<MemFetch<'_>>,
) -> Result<Option<NanSite>, InterpError> {
    run_traced(shader, regs, tex, mem, &mut |_, _| {})
}

/// [`run_watching_for_nan_with_env`], calling `observe(index, regs)` immediately BEFORE each
/// instruction the walk executes.
///
/// # Why the hook is here and not a second walk
/// The differential can say two register files disagree at the END of a 183-instruction
/// program. It cannot say WHERE, and a divergence nobody can locate is one nobody can fix -
/// the remaining rows in the corpus remainder are 63, 115 and 183 instructions long and were
/// being read by eye. A second interpreter written to answer that question would be a second
/// reading of the ISA and would disagree with this one somewhere, which is the mistake this
/// file's own notes keep recording. So it is the SAME walk with an observer.
///
/// The state is reported BEFORE the instruction runs, because that is the only point every
/// path through this loop passes through - a dozen `continue`s leave by different doors. So a
/// trace entry that differs names the instruction whose INPUT already differed, and the
/// culprit is the most recent executed instruction before it.
pub fn run_traced(
    shader: &Shader,
    regs: &mut RegFile,
    tex: TexFetch<'_>,
    mem: Option<MemFetch<'_>>,
    observe: &mut dyn FnMut(usize, &RegFile),
) -> Result<Option<NanSite>, InterpError> {
    run_traced_env(shader, regs, &TexEnv::sampling(tex), mem, observe)
}

/// [`run_traced`], with the full texture ENVIRONMENT - which is what a [`Op::TexGather`] needs.
///
/// Every other entry point in this file delegates here with no gather, so a caller that has not
/// got one behaves exactly as it did before gather was modelled.
pub fn run_traced_env(
    shader: &Shader,
    regs: &mut RegFile,
    env: &TexEnv<'_>,
    mem: Option<MemFetch<'_>>,
    observe: &mut dyn FnMut(usize, &RegFile),
) -> Result<Option<NanSite>, InterpError> {
    let tex = env.sample;
    let mut site: Option<NanSite> = None;
    let mut index = 0usize;
    // A BOUNDED walk. The stream carries real loops (a golf title's world programs run a
    // four-light loop around a memory load), so a straight-line walk would run each body once
    // and a PC-driven one can in principle not terminate at all - on a register file the caller
    // filled from a live draw, which is exactly where a wrong loop bound would come from. The
    // budget is far past any shipped program's dynamic length and turns a runaway into a named
    // failure instead of a hung render thread.
    let mut steps = 0u32;
    const MAX_STEPS: u32 = 1 << 20;
    while let Some(instr) = shader.instrs.get(index) {
        steps += 1;
        observe(index, regs);
        if steps > MAX_STEPS {
            return Err(InterpError::UnsupportedOp {
                index,
                op: "a loop that did not terminate within the interpreter's step budget",
            });
        }
        if let Some(reason) = instr.blocked {
            return Err(InterpError::Blocked { index, reason });
        }
        if !instr.op.is_emittable() {
            return Err(InterpError::UnsupportedOp { index, op: instr.op.mnemonic() });
        }
        // A no-op (phase declaration / NOP) has no register effect.
        if matches!(instr.op, Op::Nop) {
            index += 1;
            continue;
        }
        // A predicated instruction executes only when its predicate register holds.
        match instr.pred {
            Predicate::Always => {}
            Predicate::IfP(n) if regs.p[(n & 3) as usize] => {}
            Predicate::IfNotP(n) if !regs.p[(n & 3) as usize] => {}
            Predicate::IfP(_) | Predicate::IfNotP(_) => {
                index += 1;
                continue;
            }
            Predicate::Raw(_) => return Err(InterpError::Blocked { index, reason: "unresolved predicate encoding" }),
        }
        // BRANCHES ARE FOLLOWED, so the reference executes the control flow the emitted shader
        // executes. A straight-line walk is not a cheaper approximation of this - it runs the
        // bodies of skipped `if`s and runs a loop once - and the values it leaves behind are
        // then a program the frame never ran.
        //
        // `rel` is already in the UNROLLED stream's numbering (`remap_branch_targets`), and the
        // predicate above has already decided whether this branch is taken at all.
        if let Op::Branch { rel } = instr.op {
            let target = index as i64 + rel as i64;
            if target < 0 {
                return Err(InterpError::OutOfRange { index });
            }
            index = target as usize;
            continue;
        }
        // The TEST group writes a PREDICATE, not a register, so it comes before the destination
        // is required - and a predicate-only VTST has no destination at all. Without this the
        // interpreter refused every program with a conditional in it, which on this corpus is
        // every world vertex program of a golf title.
        if let Op::Test { alu, cmp, reduce, pdst, write_back } = instr.op {
            let bools = test_channels(regs, instr, alu, cmp, index)?;
            regs.p[(pdst & 3) as usize] = match reduce {
                TestReduce::Channel(c) => bools[(c as usize).min(3)],
                TestReduce::AndAll => bools.iter().all(|&b| b),
                TestReduce::OrAll => bools.iter().any(|&b| b),
            };
            if !write_back {
                index += 1;
                continue;
            }
        }
        // >>> KILL ENDS THE FRAGMENT, and it has no destination - so like the TEST group it
        // comes before one is required.
        //
        // The walk STOPS. A discarded pixel writes no colour, and whatever the register file
        // would have held afterwards is unobservable: there is no pixel to hold it. Continuing
        // would make the reference's answer depend on instructions the hardware's pixel never
        // completes, and the emitted module (whose `discard` demotes the invocation, so its
        // stores stop landing) would then disagree with it on every later write.
        if matches!(instr.op, Op::Kill) {
            regs.killed = true;
            return Ok(site);
        }
        // >>> DEPTHF REPLACES THE FRAGMENT'S DEPTH with a scalar the program computed, in the
        // GUEST's window encoding. It is recorded RAW - see `RegFile::frag_depth` for why the
        // renderer's forward map is not applied here.
        if matches!(instr.op, Op::DepthF) {
            let src = instr.srcs.first().ok_or(InterpError::OutOfRange { index })?;
            let v = read_channel_prec(regs, src, 0, Prec::of(instr))
                .ok_or(InterpError::OutOfRange { index })?;
            regs.frag_depth = Some(v);
            index += 1;
            continue;
        }
        let Some(dest) = instr.dest.as_ref() else {
            return Err(InterpError::OutOfRange { index });
        };
        // A 32-bit LITERAL LOAD writes the raw bit pattern into the destination's first lane,
        // with no per-channel evaluation and no float interpretation - exactly what the
        // emitter's `store_raw` does. This register file already carries bit patterns in its
        // `f32` lanes (every raw-lane reader works in `to_bits`), so the literal is stored the
        // same way round and a later raw read of it sees the pattern the shader sees.
        if let Op::Limm { value } = instr.op {
            let bank = regs.bank_mut(dest.bank).ok_or(InterpError::OutOfRange { index })?;
            let slot = bank.get_mut(dest.index as usize).ok_or(InterpError::OutOfRange { index })?;
            *slot = f32::from_bits(value);
            index += 1;
            continue;
        }
        // Loading an INDEX register is a write to the index file, not to a bank, so it too
        // sits outside the per-channel evaluator. The source lane holds an integer BIT
        // PATTERN (a truncating convert followed by a 16-bit shift produced it), and the
        // interpreter's lanes are f32 - so it is read through the same `& 0xffff` the emitter
        // applies, over the lane's bits rather than its float value.
        if let Op::LoadIndex { addend, to_index, stride } = instr.op {
            let s1 = instr.srcs.first().ok_or(InterpError::OutOfRange { index })?;
            let bank = regs.bank(s1.bank).ok_or(InterpError::OutOfRange { index })?;
            let raw = bank.get(s1.index as usize).ok_or(InterpError::OutOfRange { index })?.to_bits();
            // The same PAIR scale the emitter applies - see `wgsl::emit_load_index`. An
            // interpreter that indexes in single registers is measuring a different program
            // than the one the GPU runs, which is the whole failure mode this file exists to
            // avoid.
            if to_index {
                regs.idx[(dest.index & 1) as usize] =
                    ((raw & 0xffff) as i32 + addend) * crate::module::index_register_scale();
            } else {
                // The ORDINARY-REGISTER form: no index file, no pair scale - see
                // `Op::LoadIndex`. The sum is an integer bit pattern like every other value
                // this interpreter carries in an f32 lane.
                let bank = regs.bank_mut(dest.bank).ok_or(InterpError::OutOfRange { index })?;
                let slot = bank.get_mut(dest.index as usize).ok_or(InterpError::OutOfRange { index })?;
                *slot = f32::from_bits(
                    ((raw & 0xffff) as i32
                        * crate::link::index_load_multiplier().unwrap_or(stride as i32)
                        + addend) as u32,
                );
            }
            index += 1;
            continue;
        }
        // A texture sample produces four components from ONE fetch, so it cannot go through
        // the per-channel evaluator below. The result lands in `dest.index + 0..4`, matching
        // the decoder's direct destination rule.
        // A memory load reads `elements` consecutive guest WORDS through the draw's bound
        // windows into consecutive destination registers, exactly as `emit_mem_load` does. The
        // pointer register and the loaded values are raw 32-bit lanes, so both go through the
        // lane's bit pattern rather than through a float view.
        if let Op::MemLoad { elements, offset_bytes } = instr.op {
            let Some(mem) = mem else {
                return Err(InterpError::UnsupportedOp {
                    index,
                    op: "ldmem (no guest-memory window supplied to this interpretation)",
                });
            };
            let src = instr.srcs.first().ok_or(InterpError::OutOfRange { index })?;
            // >>> AN ADDRESS OPERAND IS A RAW LANE, NOT A SWIZZLED CHANNEL. The emitted load
            // reads `bank[src0.index]` exactly - no swizzle selector, no abs/neg - so reading it
            // through `read_channel` addressed lane `index + swizzle[0]` instead and, for any
            // operand whose first selector is not x, fetched a DIFFERENT REGISTER as the
            // pointer. The two then resolved different guest addresses from the same program.
            let raw_lane = |o: &Operand| -> Option<u32> {
                Some(regs.bank(o.bank)?.get(o.index as usize)?.to_bits())
            };
            let ptr_bits = raw_lane(src).ok_or(InterpError::OutOfRange { index })?;
            // The instruction's REGISTER-supplied byte offsets, which the decoder puts in
            // `srcs` after the pointer, added exactly as the emitted `gxp_a<n>` expression adds
            // them. Leaving them out is not a small inaccuracy: an indexed read of a bone
            // matrix or a uniform array would land on element ZERO every time, so a mesh whose
            // vertices are nowhere would interpret as a mesh that is fine.
            // The register offset is 16 bits wide - see `wgsl::emit_mem_load`, which is the
            // code that ships and which this reference may never disagree with.
            let narrow = crate::link::arm_on(crate::link::MEM_OFFSET16_ARM);
            let mut addr = ptr_bits.wrapping_add(offset_bytes);
            for o in instr.srcs.iter().skip(1) {
                // Raw lanes too, for the same reason: the emitted expression adds
                // `bank[o.index]`, not a swizzled channel of it.
                let v = raw_lane(o).ok_or(InterpError::OutOfRange { index })?;
                let v = if narrow { v & 0xffff } else { v };
                addr = addr.wrapping_add(v);
            }
            let base = dest.index as usize;
            let bank = regs.bank_mut(dest.bank).ok_or(InterpError::OutOfRange { index })?;
            for k in 0..elements as u32 {
                let word = mem(addr.wrapping_add(k * 4));
                match bank.get_mut(base + k as usize) {
                    Some(slot) => *slot = f32::from_bits(word),
                    None => return Err(InterpError::OutOfRange { index }),
                }
            }
            index += 1;
            continue;
        }
        // >>> A GATHER WRITES SIX REGISTERS, and the destination extent is part of what the
        // opcode means: `dest + 0..3` are the four texels of the 2x2 footprint at the
        // instruction's result precision, and `dest + 4..5` hold four F16 bilinear
        // coefficients, two to a register.
        //
        // The footprint itself comes from the ENVIRONMENT (see [`GatherFetch`]), because which
        // texels a coordinate names is a property of the bound texture and not of the
        // instruction. A caller with no gather fetcher is refused by name rather than handed
        // four copies of one filtered sample, which would make a shadow filter look as though
        // it worked.
        //
        // >>> THE COEFFICIENT PAIRING IS THE ONE THING THE INSTRUCTION DOES NOT STATE, and the
        // corpus's only consumer settles it: a golf title's shadow filter reduces the gathered
        // comparisons with `dot(mask.wzyx, coeff.xyzw)`, pairing coefficient `k` with gathered
        // texel `3 - k`. So coefficient `k` is the bilinear weight of texel `3 - k`, and with
        // the footprint in the platform's `(x0,y1) (x1,y1) (x1,y0) (x0,y0)` order that makes
        // coefficient 0 the weight of `(x0,y0)`.
        if let Op::TexGather { unit, coords, coord_half } = instr.op {
            if coords != 2 {
                return Err(InterpError::UnsupportedOp {
                    index,
                    op: "tex.gather4 with a coordinate that is not 2D",
                });
            }
            let Some(gather) = env.gather else {
                return Err(InterpError::UnsupportedOp {
                    index,
                    op: "tex.gather4 (no texel-level fetch supplied to this interpretation)",
                });
            };
            let src = instr.srcs.first().ok_or(InterpError::OutOfRange { index })?;
            let cp = if coord_half { Prec::F16 } else { Prec::F32 };
            let uv = [
                read_channel_prec(regs, src, 0, cp).ok_or(InterpError::OutOfRange { index })?,
                read_channel_prec(regs, src, 1, cp).ok_or(InterpError::OutOfRange { index })?,
            ];
            let (texels, frac) = gather(unit, uv).ok_or(InterpError::UnsupportedOp {
                index,
                op: "tex.gather4 (no texture bound for this unit)",
            })?;
            // The emitter stores the four texels at FULL precision whatever the instruction's
            // own result precision says (`emit_tex_gather` passes `Prec::F32`), so the
            // reference does too - a disagreement about that would be about the emitter's
            // choice, not about the footprint this arm is here to check.
            let dest_bank = dest.bank;
            let base = dest.index as u32;
            {
                let bank = regs.bank_mut(dest_bank).ok_or(InterpError::OutOfRange { index })?;
                for (k, v) in texels.iter().enumerate() {
                    write_reg_lane(bank, base, k as u32, bank_prec(dest_bank, Prec::F32), *v)
                        .ok_or(InterpError::OutOfRange { index })?;
                }
            }
            let (fx, fy) = (frac[0], frac[1]);
            let weight = |k: usize| {
                // The bilinear weight of the footprint's texel `k`, in the platform's order:
                // index 0 is `(x0,y1)`, 1 is `(x1,y1)`, 2 is `(x1,y0)`, 3 is `(x0,y0)`.
                let (wx, wy) = match k {
                    0 => (1.0 - fx, fy),
                    1 => (fx, fy),
                    2 => (fx, 1.0 - fy),
                    _ => (1.0 - fx, 1.0 - fy),
                };
                wx * wy
            };
            let coeff_base = base.checked_add(4).ok_or(InterpError::OutOfRange { index })?;
            let bank = regs.bank_mut(dest_bank).ok_or(InterpError::OutOfRange { index })?;
            for c in 0..4u32 {
                write_reg_lane(
                    bank,
                    coeff_base,
                    c,
                    bank_prec(dest_bank, Prec::F16),
                    weight(3 - c as usize),
                )
                .ok_or(InterpError::OutOfRange { index })?;
            }
            index += 1;
            continue;
        }
        if let Op::Tex { unit, coords, coord_half, lod } = instr.op {
            // >>> THE SAMPLED RGBA LANDS AT THE INSTRUCTION'S OWN PRECISION, like every other
            // write. This used to store four consecutive whole lanes on the stated grounds that
            // "this register file has no packing ANYWHERE" - true when it was written, false
            // since the file gained the packed F16/byte views. An F16 sample lands as TWO packed
            // pairs, and storing it as four whole lanes overwrote the neighbouring register and
            // put every channel in the wrong half.
            //
            // The COORDINATE carries its own precision, independent of the result's - a shader
            // routinely computes an F16 UV and asks for an F32 result - which is the same split
            // `emit_tex` makes from the same two decoded fields.
            let src = instr.srcs.first().ok_or(InterpError::OutOfRange { index })?;
            let cp = if coord_half { Prec::F16 } else { Prec::F32 };
            let mut coord = [0.0f32; 4];
            for k in 0..(coords as usize).clamp(1, 4) {
                coord[k] = read_channel_prec(regs, src, k, cp).ok_or(InterpError::OutOfRange { index })?;
            }
            // The LOD operand is `src2`, read at F32 - the width `emit_tex` reads it at.
            let mut lod_arg = TexLodArg { mode: lod, args: [0.0; 4] };
            if TexLodArg::channels(lod) > 0 {
                let l = instr.srcs.get(1).ok_or(InterpError::OutOfRange { index })?;
                for k in 0..TexLodArg::channels(lod) {
                    lod_arg.args[k] =
                        read_channel_prec(regs, l, k, Prec::F32).ok_or(InterpError::OutOfRange { index })?;
                }
            }
            let rgba = tex(unit, coord, lod_arg)
                .ok_or(InterpError::UnsupportedOp { index, op: "tex (no texture bound for this unit)" })?;
            // All four channels are stored, write mask or not - exactly as `emit_tex` does.
            let dp = bank_prec(dest.bank, Prec::of(instr));
            let base = dest.index as u32;
            let bank = regs.bank_mut(dest.bank).ok_or(InterpError::OutOfRange { index })?;
            for (k, v) in rgba.iter().enumerate() {
                write_reg_lane(bank, base, k as u32, dp, *v).ok_or(InterpError::OutOfRange { index })?;
            }
            index += 1;
            continue;
        }
        // >>> AN EQUAL-WIDTH INTEGER REPACK MOVES BIT PATTERNS, and this register file holds
        // >>> them: a lane is a 32-bit WORD (every raw-lane reader here works in `to_bits`).
        //
        // It refused with "this register file cannot hold a packed 16-bit half pair", and that
        // was never quite the situation - two 16-bit halves fit in a 32-bit word exactly as
        // four bytes do, and the F16 view has been splitting one for several sessions. What is
        // true is that neither end CONVERTS: `emit_pack_int_copy` moves the element unsigned,
        // because at equal widths the pattern does not depend on the labels and sign-extending
        // here would spill into the partner half's bits and destroy a value the shader reads
        // back. So the reference moves it unsigned too.
        //
        // The SWIZZLE selects an element and elements cross registers: four 16-bit halves span
        // two registers, four bytes span one, which is the `elems_per_reg` arithmetic below and
        // the same rule `store_raw_half`/`store_raw_byte` write by.
        //
        // MEASURED: 31 of the corpus's 82 interpreter refusals, the single largest group.
        if let Op::PackIntCopy { bits } = instr.op {
            let bits = bits as u32;
            if bits != 16 && bits != 8 {
                return Err(InterpError::UnsupportedOp {
                    index,
                    op: "pack.int.copy at a width other than 8 or 16",
                });
            }
            let s1 = instr.srcs.first().ok_or(InterpError::OutOfRange { index })?;
            let per_reg = 32 / bits;
            let mask = (1u32 << bits) - 1;
            let mut elems = [0u32; 4];
            for c in 0..4 {
                if !instr.write_mask[c] {
                    continue;
                }
                let sel = s1.swizzle[c] as u32;
                let word = regs
                    .bank(s1.bank)
                    .and_then(|b| b.get((s1.index as u32 + sel / per_reg) as usize))
                    .ok_or(InterpError::OutOfRange { index })?
                    .to_bits();
                elems[c] = (word >> ((sel % per_reg) * bits)) & mask;
            }
            let base = dest.index as u32;
            let bank = regs.bank_mut(dest.bank).ok_or(InterpError::OutOfRange { index })?;
            for c in 0..4u32 {
                if !instr.write_mask[c as usize] {
                    continue;
                }
                let (reg_off, sh) = crate::ir::packed_dest_slot(bits, c);
                let slot =
                    bank.get_mut((base + reg_off) as usize).ok_or(InterpError::OutOfRange { index })?;
                let word = slot.to_bits();
                *slot = f32::from_bits((word & !(mask << sh)) | (elems[c as usize] << sh));
            }
            index += 1;
            continue;
        }
        // >>> A BYTE-SELECT MOVES RAW BYTES, so it cannot ride the per-channel float evaluator
        // below - the value it carries is a bone index, not a number, and `byte / 255` is not
        // it.
        //
        // This is why the refusal census's "the same Fx8 routing" reading of `cmov.u8` was
        // wrong: `Prec::Fx8` is the UNORM view (`byte / 255` in, `round(v * 255)` out), which
        // is right for the colour combiner and would quantise an index to nothing. What this
        // instruction needs is the RAW byte of the lane's bit pattern, which is exactly what
        // the emitter's `store_raw_byte` writes and `raw_elem_expr` reads.
        //
        // The TEST's reading is shared with the emitter through `module::cmov_u8_tests_each_byte`
        // rather than transcribed: whether the form tests one byte or four is not visible in
        // any captured word (every one carries the full mask), so a reference that chose for
        // itself would disagree with the shipped shader on exactly the programs where it
        // matters - a source whose four bytes differ.
        if let Op::CmovU8 { test } = instr.op {
            use crate::ir::CompareMethod;
            let s1 = instr.srcs.first().ok_or(InterpError::OutOfRange { index })?;
            let s2 = instr.srcs.get(1).ok_or(InterpError::OutOfRange { index })?;
            let s0 = instr.srcs.get(2).ok_or(InterpError::OutOfRange { index })?;
            // The operand's OWN register, byte `c` of it - no swizzle selector and no abs/neg,
            // the same addressing the emitted statement uses.
            let byte = |o: &Operand, c: u32| -> Option<u32> {
                let word = regs.bank(o.bank)?.get(o.index as usize)?.to_bits();
                Some((word >> (8 * c)) & 0xff)
            };
            let per_byte = crate::module::cmov_u8_tests_each_byte();
            let mut bytes = [0u32; 4];
            for c in 0..4u32 {
                if !instr.write_mask[c as usize] {
                    continue;
                }
                let t = byte(s0, if per_byte { c } else { 0 })
                    .ok_or(InterpError::OutOfRange { index })?;
                // `LtZero`/`LteZero` read that byte SIGNED, which is what makes the comparison
                // mean anything - an unsigned byte is never below zero.
                let signed = t as i8 as i32;
                let held = match test {
                    CompareMethod::EqZero => t == 0,
                    CompareMethod::NeZero => t != 0,
                    CompareMethod::LtZero => signed < 0,
                    CompareMethod::LteZero => signed <= 0,
                };
                let pick = if held { s1 } else { s2 };
                bytes[c as usize] = byte(pick, c).ok_or(InterpError::OutOfRange { index })?;
            }
            let bank = regs.bank_mut(dest.bank).ok_or(InterpError::OutOfRange { index })?;
            let slot = bank.get_mut(dest.index as usize).ok_or(InterpError::OutOfRange { index })?;
            let mut word = slot.to_bits();
            for c in 0..4u32 {
                if instr.write_mask[c as usize] {
                    word = (word & !(0xffu32 << (8 * c))) | (bytes[c as usize] << (8 * c));
                }
            }
            *slot = f32::from_bits(word);
            index += 1;
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
        if site.is_none()
            && let Some(c) = (0..4).find(|&c| instr.write_mask[c] && !out[c].is_finite()) {
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
        // The destination's precision decides how a channel maps onto the register file - two
        // F16 channels share one word, four byte channels share one - and the INTERNAL bank is
        // four unpacked 32-bit lanes whatever precision the instruction runs at.
        let dp = bank_prec(dest.bank, Prec::of(instr));
        let base = dest.index as u32;
        // >>> AN INSTRUCTION WHOSE DESTINATION IS NOT A FLOAT CANNOT BE PLACED BY `Prec`.
        // A float->integer convert leaves a bit PATTERN in the lane, and `Prec::of` answers
        // `F32` for it - one whole register per channel - where the shipped shader writes a
        // 16-bit result into HALF a register and an 8-bit one into a quarter. Both sides now
        // read `Instr::dest_raw_packed_bits`, which is where that rule is stated and why.
        //
        // NO EXCEPTION FOR THE INTERNAL BANK, deliberately: `bank_prec` makes one for the FLOAT
        // views, but `store_raw_half`/`store_raw_byte` do not, and no corpus program writes a
        // packed integer there (0 of 180 instances - every destination is a PA, an SA or a
        // temp). An exception here would be a behaviour nothing ships and nothing can check.
        let raw_bits = instr.dest_raw_packed_bits();
        let bank = regs.bank_mut(dest.bank).ok_or(InterpError::OutOfRange { index })?;
        for c in 0..4 {
            if !instr.write_mask[c] {
                continue;
            }
            match raw_bits {
                Some(bits) => {
                    let mask = (1u32 << bits) - 1;
                    let (reg_off, sh) = crate::ir::packed_dest_slot(bits, c as u32);
                    let slot = bank
                        .get_mut((base + reg_off) as usize)
                        .ok_or(InterpError::OutOfRange { index })?;
                    let word = slot.to_bits();
                    *slot = f32::from_bits(
                        (word & !(mask << sh)) | ((out[c].to_bits() & mask) << sh),
                    );
                }
                None => write_reg_lane(bank, base, c as u32, dp, out[c])
                    .ok_or(InterpError::OutOfRange { index })?,
            }
        }
        index += 1;
    }
    Ok(site)
}

/// The four per-channel booleans a TEST instruction produces: `alu(src1, src2)` compared
/// against zero, exactly as [`crate::wgsl`]'s `emit_test` writes it.
///
/// The FLOAT families read their operands as lanes and the two RAW-LANE families (bitwise AND
/// and the integer subtract) read them as the lane's bit pattern - the same split the emitter
/// makes, for the same reason: an integer flag register read as an `f32` is a denormal, and
/// comparing that against zero answers a different question.
///
/// The 8-BIT family reads its operands as four unorm BYTES of one register, through
/// [`Prec::Fx8`] - the view this file gained after the arm was written, which is why it says so
/// instead of refusing. The u16 MASK family is refused here and is not a coverage gap: only
/// VTSTMSK's decoder produces that ALU, so a VTST can never carry it.
fn test_channels(
    regs: &RegFile,
    instr: &Instr,
    alu: crate::ir::TestAlu,
    cmp: crate::ir::TestCmp,
    index: usize,
) -> Result<[bool; 4], InterpError> {
    use crate::ir::{TestAlu, TestCmp};
    let (s1, s2) = (
        instr.srcs.first().ok_or(InterpError::OutOfRange { index })?,
        instr.srcs.get(1).ok_or(InterpError::OutOfRange { index })?,
    );
    // An immediate's number IS the integer it spells, which is how the emitter materialises it.
    let raw = |o: &Operand| -> Result<u32, InterpError> {
        if matches!(o.bank, Bank::Immediate) {
            return Ok(o.index as u32);
        }
        Ok(read_channel(regs, o, 0).ok_or(InterpError::OutOfRange { index })?.to_bits())
    };
    let mut out = [false; 4];
    for (c, slot) in out.iter_mut().enumerate() {
        *slot = match alu {
            // The BITWISE family, on the raw lane. Its operands are UNSIGNED 32-bit (both spec
            // sources say so), so the relations are unsigned ones - which is also what the
            // emitter has always written, comparing against a `0u` literal. Reading them as
            // signed here would make the oracle disagree with the shader it is checking on
            // exactly the values that have the top bit set, which is the whole population a
            // shift-left by 31 produces.
            TestAlu::BitAnd | TestAlu::BitShl => {
                let (a, b) = (raw(s1)?, raw(s2)?);
                let v = match alu {
                    TestAlu::BitAnd => a & b,
                    // The decoder refuses any amount that is not an immediate below 32.
                    _ => a << b,
                };
                match cmp {
                    TestCmp::Eq => v == 0,
                    TestCmp::Ne => v != 0,
                    TestCmp::Lt => false,
                    TestCmp::Le => v == 0,
                    TestCmp::Gt => v > 0,
                    TestCmp::Ge => true,
                }
            }
            TestAlu::IntSub => {
                let v = (raw(s1)? as i32).wrapping_sub(raw(s2)? as i32);
                match cmp {
                    TestCmp::Eq => v == 0,
                    TestCmp::Ne => v != 0,
                    TestCmp::Lt => v < 0,
                    TestCmp::Le => v <= 0,
                    TestCmp::Gt => v > 0,
                    TestCmp::Ge => v >= 0,
                }
            }
            // >>> THE 8-BIT FAMILY IS MODELLED NOW. It used to refuse with "this register file
            // has no byte packing", which was true when written and has not been since
            // [`read_reg_lane`] gained the `Prec::Fx8` view. The emitter reads exactly these
            // two operands at that precision and subtracts (see `emit_test`, where taking the
            // instruction's own precision instead would read a flag register's 0x00000001 as
            // an f32 denormal, compare it equal to zero, and turn an alpha test into a no-op
            // that draws every cut-out texel) - so the reference reads them the same way.
            TestAlu::Fx8Sub => {
                let a = read_channel_prec(regs, s1, c, Prec::Fx8)
                    .ok_or(InterpError::OutOfRange { index })?;
                let b = read_channel_prec(regs, s2, c, Prec::Fx8)
                    .ok_or(InterpError::OutOfRange { index })?;
                let v = a - b;
                match cmp {
                    TestCmp::Eq => v == 0.0,
                    TestCmp::Ne => v != 0.0,
                    TestCmp::Lt => v < 0.0,
                    TestCmp::Le => v <= 0.0,
                    TestCmp::Gt => v > 0.0,
                    TestCmp::Ge => v >= 0.0,
                }
            }
            // Only VTSTMSK's decoder produces this; VTST's own ALU table has no path to it.
            TestAlu::IntSub16U => {
                return Err(InterpError::UnsupportedOp {
                    index,
                    op: "vtst with the u16 mask family (only VTSTMSK decodes that ALU)",
                })
            }
            // >>> AT THE INSTRUCTION'S OWN PRECISION, which is what `emit_test` reads them at
            // >>> (`Prec::of(instr)`) and what this arm did NOT do.
            //
            // `read_channel` is `read_channel_prec(.., Prec::F32)` - hardwired - so a VTST in
            // the F16 pipeline had its operands read as whole 32-bit floats here and as HALVES
            // by the shader. Those are not nearby numbers: the same word read the other way is
            // a different value entirely, so the predicate came out differently, and a
            // predicate decides which arm of a branch runs. Everything downstream of it then
            // disagrees.
            //
            // MEASURED: this ONE default was the first divergence in **five** of the corpus
            // remainder's programs, across THREE titles (`cw-rr-corpus__frag_8668a340` and
            // `__frag_86689800`, `atlas-bin__frag_843f1518` and `__vert_84315b14`,
            // `mlb-corpus__frag_843f3d24`) - every one of them located by the per-instruction
            // trace naming `vtst`, and none of them findable by reading a hundred instructions.
            //
            // The `Fx8Sub` arm above already took its precision explicitly and says why; the
            // FLOAT family was left on the default beside it.
            TestAlu::Add | TestAlu::Sub | TestAlu::Mul => {
                let tp = Prec::of(instr);
                let a = read_channel_prec(regs, s1, c, tp)
                    .ok_or(InterpError::OutOfRange { index })?;
                let b = read_channel_prec(regs, s2, c, tp)
                    .ok_or(InterpError::OutOfRange { index })?;
                let v = match alu {
                    TestAlu::Add => a + b,
                    TestAlu::Sub => a - b,
                    _ => a * b,
                };
                match cmp {
                    TestCmp::Eq => v == 0.0,
                    TestCmp::Ne => v != 0.0,
                    TestCmp::Lt => v < 0.0,
                    TestCmp::Le => v <= 0.0,
                    TestCmp::Gt => v > 0.0,
                    TestCmp::Ge => v >= 0.0,
                }
            }
        };
    }
    Ok(out)
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

    /// Group 0x15 IMAD32 over the lane's INTEGER bit pattern, with an inline literal for the
    /// multiplier - the exact shape of the instruction that established the group:
    /// `pa[2] = pa[2] * 48 + sa[24]`.
    ///
    /// The literal is the point of the test. A `Bank::Immediate` operand read through the
    /// ordinary float path yields `48.0`, whose bit pattern is 0x42400000, and an integer
    /// multiply by that is not wrong by a little - it is wrong by nine orders of magnitude.
    #[test]
    fn int_mad_multiplies_a_bit_pattern_by_an_inline_literal() {
        let mut regs = RegFile::with_lanes(32);
        regs.pa[2] = f32::from_bits(7);
        regs.sa[24] = f32::from_bits(1000);
        let d = Operand::plain(Bank::PrimaryAttr, 2, 2);
        let a = Operand::plain(Bank::PrimaryAttr, 2, 2);
        let b = Operand::plain(Bank::Immediate, 48, 2);
        let cc = Operand::plain(Bank::SecondaryAttr, 24, 3);
        let mut i = instr(Op::IntMad { signed: true, bits: 32, src0_high: false, src1_high: false }, d, vec![a, b, cc]);
        // The group is scalar and carries no write mask.
        i.write_mask = [true, false, false, false];
        run(&shader(vec![i]), &mut regs).unwrap();
        assert_eq!(regs.pa[2].to_bits(), 7 * 48 + 1000);
    }

    /// A width the decoder does not establish must hard-fail here too, rather than be
    /// interpreted at 32 bits and quietly disagree with a shader that never gets emitted.
    #[test]
    fn int_mad_refuses_an_unestablished_width() {
        let mut regs = RegFile::with_lanes(8);
        let d = Operand::plain(Bank::Temp, 0, 0);
        let a = Operand::plain(Bank::Temp, 1, 0);
        let b = Operand::plain(Bank::Temp, 2, 0);
        let cc = Operand::plain(Bank::Temp, 3, 0);
        let i = instr(Op::IntMad { signed: false, bits: 16, src0_high: false, src1_high: false }, d, vec![a, b, cc]);
        assert!(run(&shader(vec![i]), &mut regs).is_err(), "a 16-bit imad must hard-fail");
    }

    /// A constant operand's value is PER CHANNEL, and the channel's swizzle selector chooses
    /// the table: selector Y (1) reads the second F32 constant bank, every other lane selector
    /// reads the first. So a `.xyzw` constant operand is NOT a broadcast of one value.
    ///
    /// >>> THIS TEST USED TO ASSERT THE BROADCAST, AND THAT IS WHAT WAS WRONG. It expected
    /// `dot3(pa.xyz, CNST6[2])` = `1+2+3` = 6.0, because the interpreter resolved a constant
    /// from the operand's INDEX alone and returned `1.0` on all three channels. The emitter -
    /// the code that actually runs on the GPU - has always read the selector, so on the Y
    /// channel it materialises bank 1's entry (0.0 here) and the dot is `1*1 + 2*0 + 3*1` =
    /// 4.0. The corpus-wide execution differential found the disagreement on a real vertex
    /// program; the interpreter has been brought to the emitter's reading.
    ///
    /// The ISA fact itself rests on the EMITTER's reading of the constant-bank selection, which
    /// no authored shader has yet confirmed against hardware. It ships, so the reference must
    /// match it; confirming it is the conformance app's job, not this test's.
    #[test]
    fn a_constant_operands_value_depends_on_the_channels_selector() {
        // o[0] = dot3(pa.xyz, const[2].xyz) = pa.x*bank0[2] + pa.y*bank1[2] + pa.z*bank0[2].
        let mut regs = RegFile::with_lanes(4);
        regs.pa[0..3].copy_from_slice(&[1.0, 2.0, 3.0]);
        let d = Operand::plain(Bank::Output, 0, 1);
        let a = Operand::plain(Bank::PrimaryAttr, 0, 2);
        let k = Operand::plain(Bank::Constant, 2, 0);
        let mut ins = instr(Op::Dot { components: 3 }, d, vec![a, k]);
        ins.write_mask = [true, false, false, false];
        run(&shader(vec![ins]), &mut regs).unwrap();
        let bank0 = crate::wgsl::cnst6_channel_value(2, 0, false, false).unwrap();
        let bank1 = crate::wgsl::cnst6_channel_value(2, 1, false, false).unwrap();
        assert_eq!(regs.o[0], 1.0 * bank0 + 2.0 * bank1 + 3.0 * bank0);
        assert_eq!(regs.o[0], 4.0, "bank 0 entry 2 is 1.0 and bank 1 entry 2 is 0.0");
    }

    /// The four INLINE constants (swizzle selectors 4..7) are a property of the selector and
    /// are the same in every precision - the one part of the constant bank that IS a broadcast
    /// when every channel selects the same one.
    #[test]
    fn the_inline_constant_selectors_are_precision_independent() {
        for (sel, want) in [(4u8, 0.0f32), (5, 1.0), (6, 2.0), (7, 0.5)] {
            for half in [false, true] {
                assert_eq!(crate::wgsl::cnst6_channel_value(0, sel, half, false), Some(want));
                // ...including in the 8-bit view, which has no constant TABLE but does have these.
                assert_eq!(crate::wgsl::cnst6_channel_value(0, sel, half, true), Some(want));
            }
        }
        // A table read in the 8-bit view is refused rather than substituted from a float bank.
        assert_eq!(crate::wgsl::cnst6_channel_value(2, 0, false, true), None);
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

    /// An instruction with no destination - what `kill` and `depthf` are.
    fn no_dest(op: Op, srcs: Vec<Operand>) -> Instr {
        Instr {
            op,
            pred: Predicate::Always,
            dest: None,
            write_mask: [true; 4],
            srcs,
            half_precision: false,
            raw: 0,
            group: 0,
            blocked: None,
        }
    }

    /// >>> A KILL ENDS THE WALK, AND THE INSTRUCTIONS AFTER IT DO NOT RUN.
    ///
    /// A discarded fragment writes no colour, so whatever the register file would have held
    /// afterwards is unobservable - there is no pixel to hold it. The emitted module's `discard`
    /// demotes the invocation and its stores stop landing, so a reference that kept walking
    /// would disagree with the GPU on every later write and report it as a defect.
    ///
    /// PROVEN DISCRIMINATING by the second half: with the walk continuing, `r[1]` would be 7.
    #[test]
    fn a_kill_ends_the_fragment_and_the_rest_of_the_program() {
        let mut regs = RegFile::with_lanes(8);
        regs.pa[0] = 7.0;
        let d0 = Operand::plain(Bank::Temp, 0, 0);
        let d1 = Operand::plain(Bank::Temp, 1, 0);
        let src = Operand::plain(Bank::PrimaryAttr, 0, 0);
        let sh = shader(vec![
            instr(Op::Mov, d0, vec![src]),
            no_dest(Op::Kill, vec![]),
            instr(Op::Mov, d1, vec![src]),
        ]);
        run(&sh, &mut regs).unwrap();
        assert!(regs.killed, "the kill was not recorded");
        assert_eq!(regs.r[0], 7.0, "what ran before the kill still ran");
        assert_eq!(regs.r[1], 0.0, "an instruction after the kill must not have run");
    }

    /// A PREDICATED kill that does not fire leaves the fragment alive and the program running.
    /// The predicate gate is shared with every other instruction, so this is checking that the
    /// kill arm sits AFTER it rather than before.
    #[test]
    fn a_predicated_kill_that_does_not_fire_leaves_the_fragment_alive() {
        let mut regs = RegFile::with_lanes(8);
        regs.pa[0] = 7.0;
        let src = Operand::plain(Bank::PrimaryAttr, 0, 0);
        let mut k = no_dest(Op::Kill, vec![]);
        k.pred = Predicate::IfP(0);
        let sh = shader(vec![k, instr(Op::Mov, Operand::plain(Bank::Temp, 1, 0), vec![src])]);
        // p[0] is false, so the kill is skipped.
        run(&sh, &mut regs).unwrap();
        assert!(!regs.killed);
        assert_eq!(regs.r[1], 7.0);
    }

    /// >>> A WRITTEN DEPTH IS RECORDED RAW, in the guest's own window encoding.
    ///
    /// Which forward map the renderer applies on the way to `@builtin(frag_depth)` is chosen
    /// per DRAW from a uniform (`gxp_depth_to_window` has four arms), so applying one here would
    /// put the renderer's state into the reference's answer for a program that never saw it.
    /// The caller that knows which map is in force applies it.
    #[test]
    fn a_written_depth_is_recorded_without_the_renderers_remap() {
        let mut regs = RegFile::with_lanes(8);
        regs.pa[2] = 17.5;
        assert_eq!(regs.frag_depth, None, "a program that writes no depth reports none");
        let sh = shader(vec![no_dest(Op::DepthF, vec![Operand::plain(Bank::PrimaryAttr, 2, 0)])]);
        run(&sh, &mut regs).unwrap();
        // NOT clamped into [0,1]: that clamp belongs to one of the four forward maps.
        assert_eq!(regs.frag_depth, Some(17.5));
    }

    /// >>> A GATHER WRITES SIX REGISTERS, and the destination extent is part of what the opcode
    /// >>> means - which is why it is its own operation rather than a flag on `Op::Tex`.
    ///
    /// Four texels at `dest + 0..3` at FULL precision, then four bilinear coefficients packed
    /// two to a register at `dest + 4..5` - and coefficient `k` is the weight of texel `3 - k`,
    /// which is the one part of the instruction the encoding does not state. The corpus's only
    /// consumer settles it: a shadow filter reduces the gathered comparisons with
    /// `dot(mask.wzyx, coeff.xyzw)`, so the coefficient order is the REVERSE of the gather's.
    ///
    /// The footprint comes from the environment because which texels a coordinate names is a
    /// property of the bound texture; what this checks is the ISA half.
    #[test]
    fn a_gather_writes_four_texels_then_four_reversed_coefficients() {
        let mut regs = RegFile::with_lanes(16);
        regs.pa[0] = 0.25;
        regs.pa[1] = 0.75;
        let texels = [0.125f32, 0.25, 0.5, 0.75];
        let frac = [0.25f32, 0.75];
        let seen = std::cell::Cell::new(None);
        let gather = |unit: u8, uv: [f32; 2]| {
            seen.set(Some((unit, uv)));
            Some((texels, frac))
        };
        let env = TexEnv { sample: &no_textures, gather: Some(&gather) };
        let d = Operand::plain(Bank::Temp, 0, 0);
        let sh = shader(vec![instr(
            Op::TexGather { unit: 5, coords: 2, coord_half: false },
            d,
            vec![Operand::plain(Bank::PrimaryAttr, 0, 0)],
        )]);
        run_traced_env(&sh, &mut regs, &env, None, &mut |_, _| {}).unwrap();

        // The coordinate reached the environment as the program's own two channels.
        assert_eq!(seen.get(), Some((5u8, [0.25, 0.75])));
        assert_eq!(&regs.r[0..4], &texels, "the four texels land in four consecutive registers");
        // The coefficients are F16 PAIRS: channel c is half `c & 1` of register `4 + (c >> 1)`.
        let (fx, fy) = (frac[0], frac[1]);
        let want = [
            (1.0 - fx) * (1.0 - fy), // the weight of texel 3, at (x0,y0)
            fx * (1.0 - fy),         // texel 2, at (x1,y0)
            fx * fy,                 // texel 1, at (x1,y1)
            (1.0 - fx) * fy,         // texel 0, at (x0,y1)
        ];
        for c in 0..4usize {
            let word = regs.r[4 + (c >> 1)].to_bits();
            let half = if c & 1 == 1 { word >> 16 } else { word & 0xffff };
            let got = crate::wgsl::f16_bits_to_f32(half as u16);
            assert_eq!(
                got,
                crate::wgsl::f16_bits_to_f32(crate::fold::f32_to_f16_bits(want[c])),
                "coefficient {c} must be the bilinear weight of texel {}",
                3 - c
            );
        }
    }

    /// >>> THE FACING FLAG IS AN INPUT, AND EVERY OTHER GLOBAL STAYS REFUSED.
    ///
    /// The emitter spells `GLOBAL[16]` as `select(0u, 1u, gxp_front_facing)` - a RAW BIT PATTERN
    /// in an integer context - so the reference has to carry it the same way, in the bit view
    /// this register file's lanes already use for every other raw value. Reading it as a FLOAT
    /// `1.0` instead would put `0x3f800000` where the shader has `1`, and the `& 1` that always
    /// follows would then take the wrong answer on a front-facing fragment.
    ///
    /// The refusal for a caller that supplied no facing, and for every other index, is the
    /// contract this whole file is built on: it never fabricates a value it was not given.
    #[test]
    fn the_facing_global_is_a_raw_bit_and_every_other_global_is_refused() {
        let d = Operand::plain(Bank::Temp, 0, 0);
        let g = Operand::plain(Bank::Global, crate::wgsl::GLOBAL_FACING, 0);
        // `x | 0` at 32 lane bits is this ISA's whole-word copy, so it moves the pattern.
        let copy = Op::Bitwise { kind: crate::ir::BitwiseKind::Or, imm: Some(0), lane_bits: 32 };

        for facing in [true, false] {
            let mut regs = RegFile::with_lanes(8);
            regs.facing = Some(facing);
            run(&shader(vec![instr(copy, d, vec![g])]), &mut regs).unwrap();
            assert_eq!(
                regs.r[0].to_bits(),
                u32::from(facing),
                "facing must arrive as the integer {} and not as a float",
                u32::from(facing)
            );
        }

        // No facing supplied: refused, not answered.
        let mut regs = RegFile::with_lanes(8);
        assert!(run(&shader(vec![instr(copy, d, vec![g])]), &mut regs).is_err());

        // A different global index: refused even with a facing in hand.
        let mut regs = RegFile::with_lanes(8);
        regs.facing = Some(true);
        let other = Operand::plain(Bank::Global, 3, 0);
        assert!(run(&shader(vec![instr(copy, d, vec![other])]), &mut regs).is_err());
    }

    /// A gather with no texel-level fetcher is REFUSED BY NAME rather than answered with four
    /// copies of one filtered sample - which would make a shadow filter look as though it
    /// worked. The reference never fabricates a value for something it has not been given.
    #[test]
    fn a_gather_with_no_fetcher_is_refused_rather_than_approximated() {
        let mut regs = RegFile::with_lanes(16);
        let sh = shader(vec![instr(
            Op::TexGather { unit: 0, coords: 2, coord_half: false },
            Operand::plain(Bank::Temp, 0, 0),
            vec![Operand::plain(Bank::PrimaryAttr, 0, 0)],
        )]);
        let err = run(&sh, &mut regs).unwrap_err();
        match err {
            InterpError::UnsupportedOp { op, .. } => {
                assert!(op.contains("gather4"), "the refusal must name the op, got {op}");
            }
            other => panic!("expected a named refusal, got {other:?}"),
        }
    }
}
