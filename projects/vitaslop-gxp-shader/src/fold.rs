//! Constant folding over a FRAGMENT program's own instruction stream, with ONE draw's uniform
//! bytes as the knowns, to answer a question no static analysis of the shader can: does this
//! draw's colour output equal the DESTINATION it was handed, for every pixel it covers?
//!
//! # Why the question is worth a whole evaluator
//! A title whose blending is ALU rather than fixed-function carries ONE generic-blend program
//! and configures it per draw out of a uniform buffer - so the same pair is a source-over, a
//! multiply, or NOTHING AT ALL depending on 64 bytes of guest memory. MEASURED on a baseball
//! title's world pass: pair `6357447cb96d21b1` is submitted EIGHTEEN times in one frame, each a
//! full-screen quad, **3,064,899 of the pass's 3,963,212 samples - 77% of all the fill in the
//! frame's biggest pass** - and at the measured frame every source coefficient in its window is
//! zero and its destination coefficient is one. Eighteen full-screen no-ops. A phone that is
//! GPU-bound at 84% with a 12.45 ms world pass cannot afford them, and nothing about the SHADER
//! says they are no-ops: the bytes say it, and the bytes are per draw.
//!
//! # What makes this sound rather than a guess about parameter names
//! Nothing here reads a parameter NAME (`UFP_SRC_Ck` and its family are this title's own words
//! and mean nothing to the next one). The evaluator runs the same operations
//! [`crate::wgsl`] emits, over the same register-file model, and folds what it can:
//!
//!   * The register file is UNTYPED 32-BIT STORAGE, tracked at BYTE granularity with each byte
//!     either known or not. That is not a detail - it is the whole reason this can be trusted
//!     where [`crate::interp`] cannot. The reference interpreter holds one `f32` per lane and
//!     reads channel `c` at `index + sel` for every precision, which is wrong for an F16
//!     instruction (four halves live in a register PAIR) and wrong for an 8-bit one (four bytes
//!     live in ONE register). A fragment corpus is 70-90% F16, so the interpreter would fold
//!     this title's blend out of the wrong registers and answer confidently
//!     [[vitaslop-the-interpreter-cannot-model-f16-packing]]. Here every read and write goes
//!     through [`Prec`] exactly as `wgsl::read_lane` / `wgsl::store_stmt` do.
//!   * An unknown input stays unknown through every operation EXCEPT where an operand's own
//!     value makes the result independent of it: `0 * x`, `0 + x`, a `dot` against a zero
//!     vector, a `select` whose two arms are the same known value. That is what a disabled
//!     blend term IS, and it is the only kind of reasoning this needs.
//!   * Anything not modelled - a texture sample, a derivative, an unestablished op, a
//!     predicated instruction, a branch - yields UNKNOWN, and an unknown anywhere in the
//!     colour means the answer is "cannot prove it" and the draw is left exactly alone.
//!
//! # The one assumption, stated
//! `0 * x == 0` and `x * 0 == 0` are taken for an unknown `x`, which is false for `NaN` and
//! for the infinities. Every unknown reaching a coefficient multiply here is a sampled texel
//! or an interpolated vertex attribute; a title whose fragment inputs are NaN has a garbage
//! frame either way. Nothing else in this file depends on the value of an unknown.

use crate::ir::{Bank, Instr, Op, Operand, Predicate};
use crate::module::{ColorOutput, ColorPrecision};
use crate::wgsl::{bank_prec, f16_bits_to_f32, Prec};

/// One 32-bit register, tracked per BYTE: `Some(b)` is a byte this draw's data determines,
/// `None` one it does not. Byte granularity because the three register views the ISA has -
/// an F32 lane, an F16 half pair, four unsigned-normalised bytes - all land on byte
/// boundaries, so one representation serves all three exactly and a partial write (an F16
/// instruction writing one half, an 8-bit one writing one byte) keeps what it did not touch.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
struct Word([Option<u8>; 4]);

impl Word {
    const UNKNOWN: Word = Word([None; 4]);

    fn from_bits(bits: u32) -> Word {
        let b = bits.to_le_bytes();
        Word([Some(b[0]), Some(b[1]), Some(b[2]), Some(b[3])])
    }

    fn bits(&self) -> Option<u32> {
        let mut out = [0u8; 4];
        for (i, b) in self.0.iter().enumerate() {
            out[i] = (*b)?;
        }
        Some(u32::from_le_bytes(out))
    }

    /// Half `h` (0 = low, 1 = high) as a 16-bit pattern, if both of its bytes are known.
    fn half_bits(&self, h: usize) -> Option<u16> {
        let (lo, hi) = (self.0[h * 2]?, self.0[h * 2 + 1]?);
        Some(u16::from(lo) | (u16::from(hi) << 8))
    }

    fn set_half_bits(&mut self, h: usize, bits: Option<u16>) {
        let (lo, hi) = match bits {
            Some(b) => (Some((b & 0xff) as u8), Some((b >> 8) as u8)),
            None => (None, None),
        };
        self.0[h * 2] = lo;
        self.0[h * 2 + 1] = hi;
    }
}

/// The abstract value of one channel: a number this draw's data pins down, or nothing.
type Val = Option<f32>;

/// What ONE DRAW supplies to the fragment stage, as the renderer already holds it.
pub struct DrawUniforms<'a> {
    /// The fragment default uniform buffer, 4 bytes per SA register from register 0 - the
    /// bytes the module binds as `fs_sa`.
    pub frag_sa: &'a [u8],
    /// Each bound fragment memory window as `(guest base address, bytes)`, in the order
    /// [`crate::module::resolve_mem_windows`] returns them - the order the `gxp_fmem` binding
    /// lays them out, so index `k` here is window `k` there.
    pub windows: &'a [(u32, &'a [u8])],
}

impl DrawUniforms<'_> {
    /// The 32-bit word at guest address `addr`, if some bound window holds it whole.
    fn load(&self, addr: u32) -> Option<u32> {
        for (base, bytes) in self.windows {
            let Some(off) = addr.checked_sub(*base) else { continue };
            let off = off as usize;
            if off + 4 <= bytes.len() {
                return Some(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
            }
        }
        None
    }
}

/// The abstract register file: the five banks the emitter subscripts, as untyped words.
struct State {
    r: Vec<Word>,
    pa: Vec<Word>,
    sa: Vec<Word>,
    o: Vec<Word>,
    i: Vec<Word>,
}

/// How many registers each bank is given. The emitted module sizes its banks to what the
/// program reads and writes; this is simply above any index a shader can name with a 7-bit
/// register field, so an operand is never out of range and never wraps into another bank.
const BANK_LANES: usize = 256;

impl State {
    /// Every bank UNKNOWN. Deliberately not zero: the emitted module's `var` banks are
    /// zero-initialised by WGSL, but a shader that reads a register nothing wrote is the
    /// uninitialised-scratch case the linker judges elsewhere, and folding a zero in here
    /// would let this file PROVE an identity out of a register the hardware never filled.
    fn new() -> State {
        State {
            r: vec![Word::UNKNOWN; BANK_LANES],
            pa: vec![Word::UNKNOWN; BANK_LANES],
            sa: vec![Word::UNKNOWN; BANK_LANES],
            o: vec![Word::UNKNOWN; BANK_LANES],
            i: vec![Word::UNKNOWN; BANK_LANES],
        }
    }

    fn bank(&self, b: Bank) -> Option<&Vec<Word>> {
        Some(match b {
            Bank::Temp => &self.r,
            Bank::PrimaryAttr => &self.pa,
            Bank::SecondaryAttr => &self.sa,
            Bank::Output => &self.o,
            Bank::Internal => &self.i,
            _ => return None,
        })
    }

    fn bank_mut(&mut self, b: Bank) -> Option<&mut Vec<Word>> {
        Some(match b {
            Bank::Temp => &mut self.r,
            Bank::PrimaryAttr => &mut self.pa,
            Bank::SecondaryAttr => &mut self.sa,
            Bank::Output => &mut self.o,
            Bank::Internal => &mut self.i,
            _ => return None,
        })
    }

    fn word(&self, b: Bank, reg: usize) -> Word {
        self.bank(b).and_then(|v| v.get(reg)).copied().unwrap_or(Word::UNKNOWN)
    }

    fn set_word(&mut self, b: Bank, reg: usize, w: Word) {
        if let Some(slot) = self.bank_mut(b).and_then(|v| v.get_mut(reg)) {
            *slot = w;
        }
    }
}

/// Read source channel `c` of `op`, mirroring `wgsl::src_channel` read for read.
fn read_channel(st: &State, op: &Operand, c: usize, prec: Prec) -> Val {
    let sel = *op.swizzle.get(c)?;
    // The swizzle CONSTANTS are the same four values at every precision and in every bank.
    let mut v = match sel {
        4 => 0.0,
        5 => 1.0,
        6 => 2.0,
        7 => 0.5,
        0..=3 => match op.bank {
            // An inline scalar literal: the operand's NUMBER is the value (spec A.7), which is
            // what the emitter writes out, and every lane selector reads that one scalar.
            Bank::Immediate => f32::from(op.index),
            Bank::Temp | Bank::PrimaryAttr | Bank::SecondaryAttr | Bank::Output | Bank::Internal => {
                let sel = u32::from(sel);
                let base = u32::from(op.index);
                match bank_prec(op.bank, prec) {
                    Prec::F32 => f32::from_bits(st.word(op.bank, (base + sel) as usize).bits()?),
                    Prec::F16 => f16_bits_to_f32(
                        st.word(op.bank, (base + (sel >> 1)) as usize).half_bits((sel & 1) as usize)?,
                    ),
                    // Four unsigned-normalised bytes in ONE register - the selector picks a
                    // byte and never a neighbouring register.
                    Prec::Fx8 => {
                        f32::from(st.word(op.bank, base as usize).0[sel as usize]?) / 255.0
                    }
                }
            }
            // The hardware constant table, a register-indirect read, a global: each has a
            // value, and none of them has one this file needs. Unknown is always safe.
            _ => return None,
        },
        _ => return None,
    };
    if op.abs {
        v = v.abs();
    }
    if op.neg {
        v = -v;
    }
    Some(v)
}

/// [`f32_to_f16_bits`], but SATURATING: a finite value too large for binary16 becomes the
/// largest finite binary16 of the same sign rather than an infinity.
///
/// >>> THIS IS WHAT A STORE INTO AN F16 HALF ACTUALLY DOES, and modelling it as an overflow to
/// infinity is not a harmless approximation - an infinity propagates, and the next subtraction
/// turns it into a NaN, so ONE overflowed store poisons everything downstream of it. The
/// corpus-wide execution differential saw exactly that: 200 diverging lanes where the reference
/// held an infinity or a NaN and the GPU held a finite number, and the GPU's number was
/// repeatedly **65504** - the largest finite binary16 - which is saturation, not overflow.
///
/// [`f32_to_f16_bits`] is deliberately left alone: its own note already records that the
/// hardware saturates and that its caller proves nothing about an overflowing value. This is
/// the variant for callers that MODEL a store, and it is a thin wrapper rather than a second
/// rounding implementation, because a second implementation is how the interpreter and the
/// emitter came to disagree about the constant bank in the first place.
pub fn f32_to_f16_bits_saturating(v: f32) -> u16 {
    let bits = f32_to_f16_bits(v);
    // Only a FINITE input that overflowed saturates; a genuine infinity stays one, and a NaN
    // (which carries a non-zero payload) is left exactly as it is.
    if v.is_finite() && bits & 0x7fff == 0x7c00 {
        return (bits & 0x8000) | 0x7bff;
    }
    bits
}

/// Round `v` to IEEE-754 binary16, returning the bit pattern - the rounding a store into an
/// F16 half performs, and the inverse of [`f16_bits_to_f32`].
///
/// Public because the reference interpreter stores into F16 halves too, and a SECOND rounding
/// implementation is exactly how the interpreter and the emitter came to disagree about the
/// constant bank. Note that this OVERFLOWS TO INFINITY; a caller modelling a real store wants
/// [`f32_to_f16_bits_saturating`].
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;
    // NaN keeps a non-zero payload (a NaN that rounded to an infinity would change the kind of
    // value it is); an infinity stays one.
    if exp == 0xff {
        return sign | 0x7c00 | if frac != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        // Overflow rounds to infinity, which is what the hardware's saturating store does not
        // do - but an overflowing value is not one this file proves anything about.
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal (or zero): shift the implicit one back in and round to nearest even.
        if e < -10 {
            return sign;
        }
        let m = frac | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut q = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        if rem > half || (rem == half && q & 1 == 1) {
            q += 1;
        }
        return sign | q as u16;
    }
    // Normal: ten fraction bits, round to nearest even.
    let mut m = frac >> 13;
    let rem = frac & 0x1fff;
    let mut e = e as u32;
    if rem > 0x1000 || (rem == 0x1000 && m & 1 == 1) {
        m += 1;
        if m == 0x400 {
            m = 0;
            e += 1;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | m as u16
}

/// Store `v` into destination channel `c`, mirroring `wgsl::store_stmt` write for write.
fn store_channel(st: &mut State, op: &Operand, c: usize, v: Val, prec: Prec) {
    let base = u32::from(op.index);
    match bank_prec(op.bank, prec) {
        Prec::F32 => {
            let w = match v {
                Some(x) => Word::from_bits(x.to_bits()),
                None => Word::UNKNOWN,
            };
            st.set_word(op.bank, (base + c as u32) as usize, w);
        }
        Prec::F16 => {
            let reg = (base + (c as u32 >> 1)) as usize;
            let mut w = st.word(op.bank, reg);
            w.set_half_bits(c & 1, v.map(f32_to_f16_bits));
            st.set_word(op.bank, reg, w);
        }
        Prec::Fx8 => {
            let reg = base as usize;
            let mut w = st.word(op.bank, reg);
            w.0[c] = v.map(|x| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            st.set_word(op.bank, reg, w);
        }
    }
}

/// `a * b`, folded: zero from EITHER side makes the product zero whatever the other side is.
/// This is the one place an unknown is allowed to vanish, and it is the mechanism a disabled
/// blend term is made of - see the module's stated assumption.
fn mul(a: Val, b: Val) -> Val {
    match (a, b) {
        (Some(x), Some(y)) => Some(x * y),
        (Some(z), None) | (None, Some(z)) if z == 0.0 => Some(0.0),
        _ => None,
    }
}

/// `a + b`, folded: a known ZERO passes the other side through unchanged. (`-0.0 + x == x`
/// for every `x` including zero, so the sign of the zero does not have to be inspected.)
fn add(a: Val, b: Val) -> Val {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + y),
        (Some(0.0), None) | (None, Some(0.0)) => None,
        _ => None,
    }
}

/// The value one emittable instruction produces for written channel `c`, or `None` when this
/// evaluator cannot pin it down. The arms are the emitter's own semantics; every op not named
/// here is unknown by construction rather than by omission - a wrong value would be a shader
/// this file claims to understand and does not.
fn eval_channel(st: &State, instr: &Instr, c: usize) -> Val {
    let sp = Prec::src_of(instr);
    let s = |n: usize, ch: usize| -> Val {
        instr.srcs.get(n).and_then(|o| read_channel(st, o, ch, sp))
    };
    match instr.op {
        Op::Mul => mul(s(0, c), s(1, c)),
        Op::Add => add(s(0, c), s(1, c)),
        Op::Mad => add(mul(s(0, c), s(1, c)), s(2, c)),
        Op::Mov | Op::Pack { .. } => s(0, c),
        Op::Min => Some(s(0, c)?.min(s(1, c)?)),
        Op::Max => Some(s(0, c)?.max(s(1, c)?)),
        Op::Frc => Some(s(0, c)?.fract()),
        Op::Dot { components } => {
            let n = usize::from(components).clamp(1, 4);
            let mut acc = Some(0.0f32);
            for k in 0..n {
                acc = add(acc, mul(s(0, k), s(1, k)));
            }
            acc
        }
        Op::Rcp => Some(1.0 / s(0, c)?),
        Op::Rsq => Some(1.0 / s(0, c)?.sqrt()),
        Op::Log => Some(s(0, c)?.log2()),
        Op::Exp => Some(s(0, c)?.exp2()),
        // A derivative of a value that is the SAME at every pixel is zero, and this evaluator's
        // knowns are exactly the per-draw uniforms - constant across the quad by construction.
        Op::Dsx | Op::Dsy => s(0, c).map(|_| 0.0),
        // The select the emitter emits. A test on a known value picks its arm; two arms that
        // are the same known value need no test at all.
        Op::Cmov { test } => {
            use crate::ir::CompareMethod::*;
            let t = s(2, c);
            match t {
                Some(t) => {
                    let cond = match test {
                        EqZero => t == 0.0,
                        NeZero => t != 0.0,
                        LtZero => t < 0.0,
                        LteZero => t <= 0.0,
                    };
                    if cond { s(0, c) } else { s(1, c) }
                }
                None => match (s(0, c), s(1, c)) {
                    (Some(a), Some(b)) if a.to_bits() == b.to_bits() => Some(a),
                    _ => None,
                },
            }
        }
        // Everything else - a texture sample, the 8-bit combiner, the integer families, a
        // memory load (handled by the caller, which writes a SPAN rather than four channels),
        // an unestablished op - is not folded.
        _ => None,
    }
}

/// Run one instruction, writing what it determines and UNKNOWN where it does not.
///
/// Returns `false` when the instruction is one this evaluator must not step over at all - a
/// predicated instruction (whose execution depends on a predicate register), or a branch
/// (whose control flow it does not model). The caller then abandons the whole question.
fn step(
    st: &mut State,
    u: &DrawUniforms,
    instr: &Instr,
    at: usize,
    trace: &mut Option<Vec<(usize, String)>>,
) -> bool {
    if instr.pred != Predicate::Always {
        return false;
    }
    if matches!(instr.op, Op::Branch { .. }) {
        return false;
    }
    if matches!(instr.op, Op::Nop) {
        return true;
    }
    let Some(dest) = instr.dest.as_ref() else {
        // An instruction with no register destination writes a predicate or nothing. A test
        // writes predicates, which only a predicated instruction reads - and one of those ends
        // the walk above - so there is nothing to record here.
        return true;
    };
    // A memory load fills `elements` CONSECUTIVE registers from one guest address, which is
    // not a four-channel write and cannot be described by the write mask (see `Op::MemLoad`).
    if let Op::MemLoad { elements, offset_bytes } = instr.op {
        // The address register holds a guest BYTE POINTER as a raw 32-bit value, so it is read
        // as BITS and not as a float: the pointer was seeded from the window's own base.
        let ptr = instr
            .srcs
            .first()
            .filter(|s| (0..=3).contains(&s.swizzle[0]))
            .and_then(|s| st.word(s.bank, (u32::from(s.index) + u32::from(s.swizzle[0])) as usize).bits());
        for k in 0..u32::from(elements) {
            let w = ptr
                .and_then(|p| p.checked_add(offset_bytes)?.checked_add(k * 4))
                .and_then(|a| u.load(a))
                .map_or(Word::UNKNOWN, Word::from_bits);
            st.set_word(dest.bank, (u32::from(dest.index) + k) as usize, w);
        }
        return true;
    }
    // >>> A LOAD-IMMEDIATE IS A WHOLE-WORD WRITE OF A KNOWN CONSTANT, and leaving it
    // >>> unmodelled threw that constant away.
    //
    // `Op::Limm` is the one 0xF8 member with a destination, and the emitter stores it with
    // `store_raw(dest, 0, ..)` - ONE raw 32-bit lane, no swizzle and no precision view, exactly
    // like the bitwise op below. Falling through to `eval_channel` instead applied a PRECISION
    // VIEW to a value that has none (the corpus uses one LIMM as an integer sentinel and
    // another as a float bit pattern) and, finding no rule for the opcode, wrote UNKNOWN - so
    // a program that loads a literal and multiplies by it lost the literal and everything
    // downstream of it. That is the same shape as `Bitwise { Or, imm: 0 }` below, which was
    // measured to be the FIRST and ONLY blocker in one eligible program.
    //
    // Exact, and it can only ever ADD proofs: the value is in the instruction.
    if let Op::Limm { value } = instr.op {
        st.set_word(dest.bank, dest.index as usize, Word::from_bits(value));
        return true;
    }
    // >>> THE INTEGER BITWISE OP IS A WHOLE-WORD OP, and modelling it is what lets this fold
    // >>> see through the emitter's own register MOVES.
    //
    // MEASURED, by the census that went looking for the instructions that stop this fold:
    // `Bitwise { kind: Or, imm: Some(0), lane_bits: 32 }` appears in EVERY eligible program in
    // every captured corpus, and in one of them it is the FIRST and ONLY thing that stops it.
    // `x | 0` is `x` - the shader is moving a raw 32-bit lane - and leaving it unmodelled threw
    // away a value the fold had already pinned down, poisoning everything downstream. That is
    // why the census's top rows are `Mad` and `Mul`: most of those are arithmetic on a value
    // this instruction had just discarded, not arithmetic the fold genuinely cannot do.
    //
    // It is modelled HERE and not in `eval_channel` because it is not a four-channel write:
    // `wgsl::emit_bitwise` reads `bank[index]` as ONE raw word (no swizzle, no precision view)
    // and assigns ONE raw word to `dest_bank[dest.index]`. Mirrored operation for operation,
    // including the lane mask, which applies to BOTH operands and to the result.
    //
    // A `Bank::Constant` or register-indirect operand stays UNKNOWN, exactly as `read_channel`
    // leaves it: this fold may only ever lose a proof, never invent one.
    if let Op::Bitwise { kind, imm, lane_bits } = instr.op {
        use crate::ir::BitwiseKind::*;
        let mask: u32 = if lane_bits >= 32 { u32::MAX } else { (1u32 << lane_bits) - 1 };
        let shift_mask = u32::from(lane_bits).wrapping_sub(1);
        let raw = |o: &Operand| -> Option<u32> {
            match o.bank {
                Bank::Temp
                | Bank::PrimaryAttr
                | Bank::SecondaryAttr
                | Bank::Output
                | Bank::Internal => st.word(o.bank, o.index as usize).bits().map(|v| v & mask),
                _ => None,
            }
        };
        let a = instr.srcs.first().and_then(&raw);
        let b = match imm {
            Some(v) => Some(v & mask),
            None => instr.srcs.get(1).and_then(&raw),
        };
        let out = match (a, b) {
            (Some(a), Some(b)) => Some(
                match kind {
                    And => a & b,
                    Or => a | b,
                    Xor => a ^ b,
                    Shl => a << (b & shift_mask),
                    Shr => a >> (b & shift_mask),
                    // Arithmetic shift is over the LANE's sign bit: a narrow lane is shifted up
                    // to 32 bits first so the sign fill comes from the right place, exactly as
                    // the emitted expression does.
                    Asr if lane_bits >= 32 => ((a as i32) >> (b & 31)) as u32,
                    Asr => {
                        let up = 32 - u32::from(lane_bits);
                        ((((a << up) as i32) >> up) as u32) >> (b & shift_mask)
                    }
                } & mask,
            ),
            _ => None,
        };
        if out.is_none()
            && let Some(t) = trace.as_mut()
        {
            t.push((at, format!("{:?}", instr.op)));
        }
        st.set_word(dest.bank, dest.index as usize, out.map_or(Word::UNKNOWN, Word::from_bits));
        return true;
    }
    let prec = Prec::of(instr);
    // Every channel is READ before any is written, exactly as the hardware does and as the
    // emitter's staged stores do - otherwise an instruction whose destination overlaps its
    // source computes something no hardware would.
    let mut vals = [None; 4];
    for c in 0..4 {
        if instr.write_mask[c] {
            vals[c] = eval_channel(st, instr, c);
        }
    }
    // A channel the mask selects and the evaluator could not pin down is the fold's gap, and
    // where a trace is being taken it is recorded HERE - at the instruction that produced it,
    // which is the only place that knows.
    if let Some(t) = trace.as_mut()
        && (0..4).any(|c| instr.write_mask[c] && vals[c].is_none())
    {
        t.push((at, format!("{:?}", instr.op)));
    }
    for c in 0..4 {
        if instr.write_mask[c] {
            store_channel(st, dest, c, vals[c], prec);
        }
    }
    true
}

/// Seed the OUTPUT bank with a constant destination colour, in the colour's own register
/// layout - the abstract mirror of `module::dest_seed`, which is what each arm of a
/// dual-source body does before it runs.
fn seed_dest(st: &mut State, precision: ColorPrecision, d: f32) {
    match precision {
        ColorPrecision::F32 => {
            for c in 0..4 {
                st.set_word(Bank::Output, c, Word::from_bits(d.to_bits()));
            }
        }
        ColorPrecision::F16 => {
            let h = f32_to_f16_bits(d);
            let packed = u32::from(h) | (u32::from(h) << 16);
            st.set_word(Bank::Output, 0, Word::from_bits(packed));
            st.set_word(Bank::Output, 1, Word::from_bits(packed));
        }
        ColorPrecision::Fx8 => {
            let b = (d.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            st.set_word(Bank::Output, 0, Word([Some(b); 4]));
        }
    }
}

/// The four colour components the module would return, read out of the colour registers in
/// the layout `link::color_return_expr` reads them - the exact inverse of [`seed_dest`].
fn read_color(st: &State, color: ColorOutput, precision: ColorPrecision) -> [Val; 4] {
    let (bank, base) = match color {
        ColorOutput::NativeO0 => (Bank::Output, 0usize),
        ColorOutput::NonNativePa(b) => (Bank::PrimaryAttr, b as usize),
    };
    let mut out = [None; 4];
    match precision {
        ColorPrecision::F32 => {
            for c in 0..4 {
                out[c] = st.word(bank, base + c).bits().map(f32::from_bits);
            }
        }
        ColorPrecision::F16 => {
            for c in 0..4 {
                out[c] = st.word(bank, base + (c >> 1)).half_bits(c & 1).map(f16_bits_to_f32);
            }
        }
        ColorPrecision::Fx8 => {
            let w = st.word(bank, base);
            for c in 0..4 {
                out[c] = w.0[c].map(|b| f32::from(b) / 255.0);
            }
        }
    }
    out
}

/// Seed the SA file with everything the module binds into it: the draw's default uniform
/// buffer, the container literals its prologue reads, and each memory window's base address
/// in the register the window is reached through.
fn seed_sa(st: &mut State, u: &DrawUniforms, uniform_regs: u32, literals: &[(u32, u32)], bases: &[u32]) {
    for reg in 0..uniform_regs as usize {
        let off = reg * 4;
        if off + 4 <= u.frag_sa.len() {
            let bits = u32::from_le_bytes(u.frag_sa[off..off + 4].try_into().unwrap());
            st.set_word(Bank::SecondaryAttr, reg, Word::from_bits(bits));
        }
    }
    for &(reg, bits) in literals {
        st.set_word(Bank::SecondaryAttr, reg as usize, Word::from_bits(bits));
    }
    // The window's base ADDRESS, which is what the module writes into this register
    // (`sa[k] = gxp_fmem[0u].x`) and what the program's own loads add their offsets to.
    for (k, &reg) in bases.iter().enumerate() {
        if let Some((base, _)) = u.windows.get(k) {
            st.set_word(Bank::SecondaryAttr, reg as usize, Word::from_bits(*base));
        }
    }
}

/// Whether THIS DRAW's fragment colour is, for every pixel, exactly the destination colour it
/// was handed - so writing it changes no pixel and the colour write can be dropped entirely.
///
/// `false` whenever that cannot be PROVEN, which includes every program this evaluator does not
/// fully fold. The proof:
///
///   1. the program must read the destination and be LINEAR in it
///      ([`crate::module::dest_is_linear`]), so `out = G + dst * F` with `G`, `F` independent
///      of `dst` - which is the same fact the dual-source lowering rests on;
///   2. the body is evaluated twice from this draw's own bytes, at `dst = 0` and `dst = 1`,
///      exactly as `link::emit_dual_split_tail` evaluates it on the GPU;
///   3. the answer is yes when `G` folds to zero in all four channels and `F` to one - i.e.
///      `out = dst` identically.
///
/// Step 1 is what makes two sample points a proof rather than two spot checks, and it is also
/// why this holds whichever path the draw takes: a dual-source pipeline computes
/// `G + dst * (F - G)` and an attachment copy computes the body over the real texel, and
/// linearity makes those the same function.
pub fn fragment_colour_is_destination(bytes: &[u8], u: &DrawUniforms) -> bool {
    // G == 0 and F == 1, every channel, or this draw paints something.
    matches!(fragment_colour_terms(bytes, u), Some((g, f))
        if (0..4).all(|c| g[c] == Some(0.0) && f[c] == Some(1.0)))
}

/// The two terms of `out = G + dst * F` as THIS DRAW's bytes determine them, channel by channel,
/// with `None` for a channel the fold cannot pin down - the evidence behind
/// [`fragment_colour_is_destination`], and what a report prints when the answer is no.
///
/// `None` for a program the fold refuses outright (it does not read the destination, is not
/// linear in it, or carries control flow this does not model).
pub fn fragment_colour_terms(bytes: &[u8], u: &DrawUniforms) -> Option<([Val; 4], [Val; 4])> {
    fragment_colour_terms_traced(bytes, u, &mut None)
}

/// >>> EVERYTHING ABOUT A FOLD THAT DEPENDS ON THE BLOB AND NOT ON THE DRAW, prepared once.
///
/// # Why this type exists
/// The fold is asked PER DRAW, because a generic-blend pair is a no-op in one draw and a
/// composite in the next - that is the whole finding it rests on. But the work in front of the
/// per-draw evaluation was not per-draw at all: `recompile_fragment` DECODES the USSE stream
/// and EMITS the complete WGSL body (tens of kilobytes of text), and `plan_bindings`,
/// `with_secondary`, `secondary_attr_init` and `resolve_mem_windows` each walk the program
/// again. None of it can change with a draw's uniform bytes.
///
/// MEASURED, with the census that made this visible (`EncodeWork::fold_asks`): on a baseball
/// title's world pass the byte-keyed memo in front of the fold takes **141 asks a frame and
/// misses 19 of them**, and the hash those asks pay is **128 bytes each** - 0.02 MB a frame.
/// So the hash was never the cost, and the two reverted attempts to make it cheaper were both
/// aimed at the wrong half. Nineteen full recompiles a frame were, and this removes them
/// without changing a single answer: the cached values are a pure function of the blob.
pub struct FoldProgram {
    /// The stream as it RUNS - secondary (which fills the SA file) then primary.
    shader: crate::ir::Shader,
    plan: crate::module::BindingPlan,
    literals: Vec<(u32, u32)>,
    bases: Vec<u32>,
    sa_extent: u32,
    /// Index of the first instruction that reads the destination: the prefix runs once and the
    /// tail runs twice from the state it leaves.
    cut: usize,
}

impl FoldProgram {
    /// Prepare a blob for folding, or `None` for a program no draw of which can ever be an
    /// identity (it does not read the destination, or is not linear in it). That second answer
    /// is exactly [`fragment_can_be_identity`], so a caller holding a `FoldProgram` has already
    /// paid for both questions.
    pub fn prepare(bytes: &[u8]) -> Option<FoldProgram> {
        let rc = crate::recompile_fragment(bytes).ok()?;
        if !crate::module::declares_dest_color(&rc.shader) {
            return None;
        }
        let shader = crate::module::with_secondary(&rc.program, &rc.shader);
        if crate::module::dest_is_linear_or_why(&shader).is_err() {
            return None;
        }
        let plan = crate::module::plan_bindings(&rc.shader, rc.program.sa_carried_extent(), |o| {
            rc.program.sampler_is_cube(o as u32)
        });
        let literals = crate::link::secondary_attr_init(&rc.shader, &rc.program).ok()?;
        // >>> THE WINDOWS COME FROM `resolve_mem_windows`, NOT FROM THE BINDING PLAN.
        // `plan_bindings` is built from the SHADER alone and says so: it cannot resolve a window
        // without the program's containers and parameter table, so it carries an EMPTY list that
        // the link fills in later. Reading the plan's list here left the pointer register
        // unseeded, every coefficient load unknown, and the fold answered `?` on all four
        // channels of both terms for a draw whose window plainly held the identity - which is
        // what the report's `?` is for. This is the same call, in the same ORDER, that
        // `fragment_dual_source_plan` resolves its gates through, so window `k` here is window
        // `k` in the draw's own list.
        let bases: Vec<u32> =
            crate::module::resolve_mem_windows(&rc.program, &rc.shader).ok()?.iter().map(|w| w.base_sa).collect();
        let cut = crate::module::first_dest_reader(&shader).unwrap_or(0);
        Some(FoldProgram { shader, plan, literals, bases, sa_extent: rc.program.sa_carried_extent(), cut })
    }
}

fn fragment_colour_terms_traced(
    bytes: &[u8],
    u: &DrawUniforms,
    trace: &mut Option<Vec<(usize, String)>>,
) -> Option<([Val; 4], [Val; 4])> {
    fragment_colour_terms_prepared(&FoldProgram::prepare(bytes)?, u, trace)
}

/// The per-DRAW half: the same evaluation, against a blob prepared once.
pub fn fragment_colour_terms_prepared(
    fp: &FoldProgram,
    u: &DrawUniforms,
    trace: &mut Option<Vec<(usize, String)>>,
) -> Option<([Val; 4], [Val; 4])> {
    let (plan, shader, literals, bases) = (&fp.plan, &fp.shader, &fp.literals, &fp.bases);
    // A window the draw did not bind is a window whose bytes are not known, and every load
    // through it folds to unknown - which can only lose the proof, never fake one.
    let mut st = State::new();
    seed_sa(&mut st, u, fp.sa_extent, literals, bases);
    // The prefix - everything before the first read of the destination - runs ONCE, and the
    // tail runs twice from the state it leaves. Same split the emitter makes, for the same
    // reason: the prefix cannot depend on a value it has not read.
    let cut = fp.cut;
    for (at, instr) in shader.instrs[..cut].iter().enumerate() {
        if !step(&mut st, u, instr, at, trace) {
            if let Some(t) = trace.as_mut() {
                t.push((at, format!("ABANDONED at {:?}", instr.op)));
            }
            return None;
        }
    }
    let mut arms = [[None; 4]; 2];
    for (arm, d) in [0.0f32, 1.0].iter().enumerate() {
        let mut arm_st = State {
            r: st.r.clone(),
            pa: st.pa.clone(),
            sa: st.sa.clone(),
            o: st.o.clone(),
            i: st.i.clone(),
        };
        seed_dest(&mut arm_st, plan.color_precision, *d);
        for (k, instr) in shader.instrs[cut..].iter().enumerate() {
            // Only the FIRST arm is traced: the second runs the identical instruction stream
            // from an identical state but for the destination seed, so tracing it would print
            // every gap twice and say nothing new.
            let t = &mut (if arm == 0 { trace.take() } else { None });
            let ok = step(&mut arm_st, u, instr, cut + k, t);
            if arm == 0 {
                *trace = t.take();
            }
            if !ok {
                if let Some(t) = trace.as_mut() {
                    t.push((cut + k, format!("ABANDONED at {:?}", instr.op)));
                }
                return None;
            }
        }
        arms[arm] = read_color(&arm_st, plan.color, plan.color_precision);
    }
    Some((arms[0], arms[1]))
}

/// Whether this program could EVER be elided, whatever a draw's bytes say - the question worth
/// asking once per blob rather than once per draw. A program that does not read the destination
/// or is not linear in it can never write the destination back unchanged, so a draw of it is
/// never worth folding.
pub fn fragment_can_be_identity(bytes: &[u8]) -> bool {
    // >>> ONE definition of the question. This used to repeat `prepare`'s first three steps,
    // and a caller that asked both paid the decode twice; worse, the two could drift.
    FoldProgram::prepare(bytes).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_word_tracks_its_bytes_and_halves() {
        let w = Word::from_bits(0x3c00_0000);
        assert_eq!(w.bits(), Some(0x3c00_0000));
        assert_eq!(w.half_bits(0), Some(0x0000));
        assert_eq!(w.half_bits(1), Some(0x3c00));
        let mut p = Word::UNKNOWN;
        assert_eq!(p.bits(), None);
        p.set_half_bits(1, Some(0x3c00));
        // HALF known is not WORD known: an F32 read of a register whose halves were written
        // one at a time must not come back as a number.
        assert_eq!(p.bits(), None);
        assert_eq!(p.half_bits(1), Some(0x3c00));
        assert_eq!(p.half_bits(0), None);
    }

    /// The f16 rounding is the inverse of the decode the emitter's own reads go through, so a
    /// round trip is the test that matters - a wrong rounding here would let a coefficient of
    /// almost-one fold to exactly one and elide a draw that paints.
    #[test]
    fn f16_rounding_round_trips_every_representable_half() {
        for bits in 0u16..=u16::MAX {
            let exp = (bits >> 10) & 0x1f;
            if exp == 0x1f {
                continue; // inf/NaN: not a number this file reasons about
            }
            let v = f16_bits_to_f32(bits);
            assert_eq!(f32_to_f16_bits(v), bits, "half {bits:#06x} = {v} did not round trip");
        }
    }

    #[test]
    fn f16_rounding_is_to_nearest_even_and_saturates_to_infinity() {
        assert_eq!(f32_to_f16_bits(0.0), 0x0000);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-1.0), 0xbc00);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        // Exactly halfway between two halves: 1.0 + 2^-11 ties to even, i.e. down to 1.0.
        assert_eq!(f32_to_f16_bits(1.0 + 0.000_488_281_25), 0x3c00);
        // ...and one step past the tie rounds up.
        assert_eq!(f32_to_f16_bits(1.0 + 0.000_6), 0x3c01);
        // A value just under one must NOT become one.
        assert_ne!(f32_to_f16_bits(0.999), 0x3c00);
        assert_eq!(f32_to_f16_bits(70000.0), 0x7c00);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        // The smallest normal and the smallest subnormal.
        assert_eq!(f32_to_f16_bits(6.103_515_6e-5), 0x0400);
        assert_eq!(f32_to_f16_bits(5.960_464_5e-8), 0x0001);
    }

    #[test]
    fn a_zero_annihilates_an_unknown_and_a_zero_add_passes_it_through() {
        assert_eq!(mul(Some(0.0), None), Some(0.0));
        assert_eq!(mul(None, Some(0.0)), Some(0.0));
        assert_eq!(mul(Some(1.0), None), None);
        assert_eq!(mul(None, None), None);
        assert_eq!(add(Some(0.0), None), None);
        assert_eq!(add(Some(1.0), None), None);
        assert_eq!(add(Some(2.0), Some(3.0)), Some(5.0));
    }

    /// One instruction, built by hand, so the whole-word bitwise model is checked against the
    /// emitter's own semantics rather than against a corpus that happens to agree.
    fn bitwise(kind: crate::ir::BitwiseKind, imm: Option<u32>, lane_bits: u8, src1: bool) -> Instr {
        let mut srcs = vec![Operand::plain(Bank::Temp, 1, 0)];
        if src1 {
            srcs.push(Operand::plain(Bank::Temp, 2, 0));
        }
        Instr {
            op: Op::Bitwise { kind, imm, lane_bits },
            pred: Predicate::Always,
            dest: Some(Operand::plain(Bank::Temp, 0, 0)),
            write_mask: [true, false, false, false],
            srcs,
            half_precision: false,
            raw: 0,
            group: 0x50,
            blocked: None,
        }
    }

    /// `x | 0` is `x`, and until this was modelled it was `unknown` - which is what stopped the
    /// colour-no-op fold on EVERY eligible program in every captured corpus. The emitter moves a
    /// raw 32-bit lane with it, so a fold that cannot see through it cannot see a blend
    /// coefficient the shader merely copied.
    #[test]
    fn an_or_with_zero_is_the_move_it_is_and_carries_the_value_through() {
        let u = DrawUniforms { frag_sa: &[], windows: &[] };
        let mut st = State::new();
        st.set_word(Bank::Temp, 1, Word::from_bits(0x3f80_0000));
        let mut trace = None;
        assert!(step(&mut st, &u, &bitwise(crate::ir::BitwiseKind::Or, Some(0), 32, false), 0, &mut trace));
        assert_eq!(st.word(Bank::Temp, 0).bits(), Some(0x3f80_0000));
    }

    #[test]
    fn the_bitwise_kinds_fold_to_what_the_emitted_expression_computes() {
        use crate::ir::BitwiseKind::*;
        let u = DrawUniforms { frag_sa: &[], windows: &[] };
        let run = |kind, imm, lane_bits, a: u32, b: Option<u32>| -> Option<u32> {
            let mut st = State::new();
            st.set_word(Bank::Temp, 1, Word::from_bits(a));
            if let Some(b) = b {
                st.set_word(Bank::Temp, 2, Word::from_bits(b));
            }
            let mut trace = None;
            assert!(step(&mut st, &u, &bitwise(kind, imm, lane_bits, b.is_some()), 0, &mut trace));
            st.word(Bank::Temp, 0).bits()
        };
        assert_eq!(run(And, Some(0x00ff_00ff), 32, 0x1234_5678, None), Some(0x0034_0078));
        assert_eq!(run(Xor, None, 32, 0xffff_ffff, Some(0x0f0f_0f0f)), Some(0xf0f0_f0f0));
        assert_eq!(run(Shl, Some(4), 32, 0x0000_0001, None), Some(0x0000_0010));
        assert_eq!(run(Shr, Some(4), 32, 0x0000_00f0, None), Some(0x0000_000f));
        // ASR is over the LANE's sign bit: in a 16-bit lane 0x8000 IS -32768, and -32768 >> 4
        // is -2048, which is 0xf800 in that lane - not the 0x0f80 a logical shift would give.
        // (Above the lane the emitted expression keeps shifting an i32 while this shifts a u32;
        // the two differ only in bits at or above `lane_bits`, which the lane mask discards, so
        // they agree on every bit either one keeps.)
        assert_eq!(run(Asr, Some(4), 16, 0x0000_8000, None), Some(0x0000_f800));
        assert_eq!(run(Asr, Some(4), 32, 0x8000_0000, None), Some(0xf800_0000));
        // A 16-bit lane masks BOTH operands and the result, so the high half never survives.
        assert_eq!(run(Or, Some(0), 16, 0xdead_beef, None), Some(0x0000_beef));
    }

    /// An operand this evaluator has no value for must leave the destination UNKNOWN, never a
    /// zero: a fold may only ever fail to prove an identity, never invent one.
    #[test]
    fn a_bitwise_over_an_unknown_operand_writes_unknown() {
        let u = DrawUniforms { frag_sa: &[], windows: &[] };
        let mut st = State::new();
        let mut trace = None;
        assert!(step(&mut st, &u, &bitwise(crate::ir::BitwiseKind::Or, Some(0), 32, false), 0, &mut trace));
        assert_eq!(st.word(Bank::Temp, 0).bits(), None);
        // ...and a CONSTANT-bank operand is unknown here exactly as it is in `read_channel`.
        let mut instr = bitwise(crate::ir::BitwiseKind::Or, Some(0), 32, false);
        instr.srcs[0] = Operand::plain(Bank::Constant, 2, 0);
        let mut st = State::new();
        assert!(step(&mut st, &u, &instr, 0, &mut trace));
        assert_eq!(st.word(Bank::Temp, 0).bits(), None);
    }
}
