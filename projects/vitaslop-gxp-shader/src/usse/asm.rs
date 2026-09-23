//! A USSE **assembler**: the inverse of [`super::decode`], built over the decoder's own
//! field tables and swizzle tables.
//!
//! # Why this exists
//!
//! Everything this crate checks is checked against blobs a title happened to ship. That has
//! two consequences that no amount of corpus work removes:
//!
//! * **A feature nothing in the corpus uses is unchecked.** The corpus is an accident of what
//!   seven games' compilers emitted; it is not a specification.
//! * **A corpus blob has no stated INTENT.** The differential harness
//!   (`tests/execcases.rs`) can say the emitter and the interpreter agree, and that is worth a
//!   great deal - but where both are wrong in the same way, they agree. Its own header says so.
//!
//! An assembled program fixes both. The test says what the program MEANS before anything runs
//! it, so a divergence names which side is wrong instead of only that one of them is; and a
//! feature can be exercised because we chose to, not because a shipped title chose to.
//!
//! # The encoding cannot drift from the decoder, by construction
//!
//! Three rules, and together they are the whole safety argument:
//!
//! 1. **Fields are written through the decoder's own tables.** [`set_field`] is the exact
//!    inverse of [`super::decode::field`] and walks the same `&[Field]` slice, so a field that
//!    moves moves for both directions at once.
//! 2. **Table-encoded operands are SEARCHED, not restated.** A swizzle or a write mask that the
//!    encoding expresses through a lookup table is found by asking the decoder's own table
//!    function for every field value and taking the one that produces the requested operand.
//!    There is no second transcription of any table here to fall out of step.
//! 3. **Every instruction is round-tripped before it is returned.** [`verify`] decodes the word
//!    it just built and compares the result against the request; a mismatch is
//!    [`AsmError::RoundTrip`], never a silently-wrong word. An assembler that can emit a word
//!    meaning something other than what was asked is worse than no assembler, because the test
//!    built on it would then assert the wrong thing confidently.
//!
//! What this does NOT establish is fidelity to the real SGX543 - an assembled word is still our
//! reading of the ISA, and a misread reaches the encoder and the decoder alike. It establishes
//! that the shader the GPU runs computes what the program was WRITTEN to mean, which is the
//! link that has been breaking.

use crate::ir::{Bank, Op, Operand, Predicate, TestAlu, TestCmp, TestReduce, TexLod};

use super::decode::{self, mask_table_08_for, mask_table_mad_for, rswz2_mad_for, rswz2_op2_for};

/// Why an instruction could not be assembled. Every variant is a refusal to emit a word that
/// would not mean what was asked - the encoding simply cannot express the request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AsmError {
    /// The named field is not in this group's table (a typo, or the wrong group).
    NoSuchField(&'static str),
    /// The value does not fit the field's width.
    FieldTooWide { field: &'static str, width: u8, value: u32 },
    /// A destination bank this group's 2-bit RSI2 selector cannot name (only Temp, Output and
    /// PrimaryAttr are encodable; the fourth selector value is index mode).
    DestBank(Bank),
    /// A source bank this group cannot name. Group 0x00's `op1` is one bit wide, for instance,
    /// so it reaches Temp and PrimaryAttr and nothing else.
    SrcBank { which: u8, bank: Bank },
    /// A register index this group's field cannot address. The six-bit fields address the bank
    /// in DOUBLE-register units, so an odd index has no encoding at all.
    RegIndex { bank: Bank, index: u8 },
    /// The requested swizzle is not in the table this operand is encoded through.
    Swizzle { which: u8, want: [u8; 4] },
    /// The requested write mask is not one this group's mask encoding can produce. In a
    /// 32-bit group-0x00 mad, for one, channel 3 has no mask bit.
    WriteMask([bool; 4]),
    /// The assembled word did not decode back to the request. Always a defect in this module.
    RoundTrip { what: &'static str },
    /// An operation this assembler has no encoder for.
    UnsupportedOp(&'static str),
}

impl std::fmt::Display for AsmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsmError::NoSuchField(n) => write!(f, "no field named `{n}` in this group"),
            AsmError::FieldTooWide { field, width, value } => {
                write!(f, "value {value} does not fit the {width}-bit field `{field}`")
            }
            AsmError::DestBank(b) => write!(f, "{b:?} is not an encodable destination bank here"),
            AsmError::SrcBank { which, bank } => {
                write!(f, "{bank:?} is not an encodable bank for source {which} here")
            }
            AsmError::RegIndex { bank, index } => write!(
                f,
                "{bank:?}[{index}] has no encoding: this field addresses the bank in \
                 double-register units, so the index must be even"
            ),
            AsmError::Swizzle { which, want } => {
                write!(f, "source {which} cannot be given the swizzle {want:?} in this group")
            }
            AsmError::WriteMask(m) => write!(f, "this group cannot write exactly the mask {m:?}"),
            AsmError::RoundTrip { what } => {
                write!(f, "ASSEMBLER DEFECT: the word did not decode back to its request ({what})")
            }
            AsmError::UnsupportedOp(op) => write!(f, "no encoder for `{op}`"),
        }
    }
}

impl std::error::Error for AsmError {}

/// Write `value` into the named field of `word`, the exact inverse of
/// [`super::decode::field`]: both walk the same MSB-first `&[Field]` table, so neither can
/// place a field the other does not.
///
/// Fast-fails on an absent field and on a value the field cannot hold, rather than truncating -
/// a silently-truncated register number is a program addressing the wrong register.
pub fn set_field(
    word: &mut u32,
    table: &[(&'static str, u8)],
    name: &'static str,
    value: u32,
) -> Result<(), AsmError> {
    let mut pos = 32u8;
    for &(fname, width) in table {
        pos -= width;
        if fname == name {
            let mask = if width == 32 { u32::MAX } else { (1u32 << width) - 1 };
            if value & !mask != 0 {
                return Err(AsmError::FieldTooWide { field: name, width, value });
            }
            *word &= !(mask << pos);
            *word |= (value & mask) << pos;
            return Ok(());
        }
    }
    Err(AsmError::NoSuchField(name))
}

/// A source operand, in the forms the ALU groups can encode.
///
/// The swizzle is stated as the four [`Operand::swizzle`] selectors you want (0..3 = x/y/z/w,
/// 4..7 = the inline constants), not as a field value: which field value produces it is the
/// encoder's problem, and where a group's table cannot produce it at all the answer is
/// [`AsmError::Swizzle`] rather than a near miss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Src {
    /// An ordinary register read.
    Reg { bank: Bank, index: u8, swizzle: [u8; 4], abs: bool, neg: bool },
    /// An inline hardware constant, selected from the CNST6 table. **The value is per
    /// channel**: the swizzle selector chooses the bank each channel reads, which is the
    /// defect the differential found in the reference first.
    Const { sel: u8, swizzle: [u8; 4], abs: bool, neg: bool },
    /// An inline integer literal: the operand's own six-bit number IS the value.
    Imm { value: u8, swizzle: [u8; 4], abs: bool, neg: bool },
    /// Register-INDIRECT: `bank[i<which> + offset]`, where `number`'s top two bits select the
    /// sub-bank and its low five are the offset. This is what an indexed uniform ARRAY read
    /// compiles to, and it is the operand form a football title's crowd sprites turn on.
    Indexed { number: u8, which: u8, swizzle: [u8; 4], abs: bool, neg: bool },
}

impl Src {
    /// `bank[index].xyzw`, unmodified.
    pub fn reg(bank: Bank, index: u8) -> Src {
        Src::Reg { bank, index, swizzle: [0, 1, 2, 3], abs: false, neg: false }
    }

    /// The CNST6 constant `sel`, broadcast (every channel reads the selector's own bank).
    pub fn cnst(sel: u8) -> Src {
        Src::Const { sel, swizzle: [0, 1, 2, 3], abs: false, neg: false }
    }

    /// The inline literal `value`.
    pub fn imm(value: u8) -> Src {
        Src::Imm { value, swizzle: [0, 1, 2, 3], abs: false, neg: false }
    }

    /// `sub_bank[i<which> + offset].xyzw`, the register-indirect read.
    pub fn indexed(sub_bank: Bank, offset: u8, which: u8) -> Src {
        let top = match sub_bank {
            Bank::Temp => 0u8,
            Bank::Output => 1,
            Bank::PrimaryAttr => 2,
            _ => 3, // SecondaryAttr - the uniform bank an indexed array read reaches.
        };
        Src::Indexed {
            number: (top << 5) | (offset & 0x1f),
            which,
            swizzle: [0, 1, 2, 3],
            abs: false,
            neg: false,
        }
    }

    /// This source with the given per-channel swizzle selectors.
    pub fn swz(mut self, s: [u8; 4]) -> Src {
        match &mut self {
            Src::Reg { swizzle, .. }
            | Src::Const { swizzle, .. }
            | Src::Imm { swizzle, .. }
            | Src::Indexed { swizzle, .. } => *swizzle = s,
        }
        self
    }

    /// This source, negated.
    pub fn negated(mut self) -> Src {
        match &mut self {
            Src::Reg { neg, .. }
            | Src::Const { neg, .. }
            | Src::Imm { neg, .. }
            | Src::Indexed { neg, .. } => *neg = true,
        }
        self
    }

    /// This source, with the absolute-value modifier.
    pub fn absolute(mut self) -> Src {
        match &mut self {
            Src::Reg { abs, .. }
            | Src::Const { abs, .. }
            | Src::Imm { abs, .. }
            | Src::Indexed { abs, .. } => *abs = true,
        }
        self
    }

    fn swizzle(&self) -> [u8; 4] {
        match *self {
            Src::Reg { swizzle, .. }
            | Src::Const { swizzle, .. }
            | Src::Imm { swizzle, .. }
            | Src::Indexed { swizzle, .. } => swizzle,
        }
    }

    fn mods(&self) -> (bool, bool) {
        match *self {
            Src::Reg { abs, neg, .. }
            | Src::Const { abs, neg, .. }
            | Src::Imm { abs, neg, .. }
            | Src::Indexed { abs, neg, .. } => (abs, neg),
        }
    }

    /// The `(alt_opt, opt, op)` field triple this source encodes to for a SIX-bit operand
    /// field. `alt_opt` selects the exotic row; `opt` is the two-bit bank/mode selector; `op`
    /// is the register or value number.
    fn fields6(&self, which: u8) -> Result<(u32, u32, u32), AsmError> {
        match *self {
            Src::Reg { bank, index, .. } => {
                let sel = match bank {
                    Bank::Temp => 0u32,
                    Bank::Output => 1,
                    Bank::PrimaryAttr => 2,
                    Bank::SecondaryAttr => 3,
                    // An INTERNAL register is the reserved top of the Temp selector's range,
                    // not a bank selector of its own.
                    Bank::Internal => {
                        let n = index / 4;
                        if index % 4 != 0 || n > 3 {
                            return Err(AsmError::RegIndex { bank, index });
                        }
                        return Ok((0, 0, 60 + u32::from(n)));
                    }
                    other => return Err(AsmError::SrcBank { which, bank: other }),
                };
                if index % 2 != 0 {
                    return Err(AsmError::RegIndex { bank, index });
                }
                let n = u32::from(index) / 2;
                if n >= 60 {
                    // 60..63 are the reserved internal-register encodings, so a plain register
                    // there has no encoding rather than a colliding one.
                    return Err(AsmError::RegIndex { bank, index });
                }
                Ok((0, sel, n))
            }
            // The exotic rows, per the operand table: 00 index1, 01 constant, 10 immediate,
            // 11 index2.
            Src::Const { sel, .. } => Ok((1, 0b01, u32::from(sel) & 0x3f)),
            Src::Imm { value, .. } => Ok((1, 0b10, u32::from(value) & 0x3f)),
            Src::Indexed { number, which: w, .. } => {
                // The indexed number is carried in the same six-bit field the register rows
                // use, so it travels DOUBLED exactly as a register number does.
                if number % 2 != 0 {
                    return Err(AsmError::RegIndex { bank: Bank::Indexed, index: number });
                }
                Ok((1, if w == 0 { 0b00 } else { 0b11 }, u32::from(number) / 2))
            }
        }
    }
}

/// The destination of an assembled instruction: a bank the group's RSI2 selector can name and
/// a register index its field can address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dest {
    pub bank: Bank,
    pub index: u8,
}

impl Dest {
    pub fn new(bank: Bank, index: u8) -> Dest {
        Dest { bank, index }
    }

    /// `(opt0, op0)` for a SIX-bit destination field.
    fn fields6(&self) -> Result<(u32, u32), AsmError> {
        let sel = match self.bank {
            Bank::Temp => 0u32,
            Bank::Output => 1,
            Bank::PrimaryAttr => 2,
            Bank::Internal => {
                let n = self.index / 4;
                if !self.index.is_multiple_of(4) || n > 3 {
                    return Err(AsmError::RegIndex { bank: self.bank, index: self.index });
                }
                return Ok((0, 60 + u32::from(n)));
            }
            other => return Err(AsmError::DestBank(other)),
        };
        if !self.index.is_multiple_of(2) {
            return Err(AsmError::RegIndex { bank: self.bank, index: self.index });
        }
        let n = u32::from(self.index) / 2;
        if n >= 60 {
            return Err(AsmError::RegIndex { bank: self.bank, index: self.index });
        }
        Ok((sel, n))
    }
}

/// Find the `(swz_alt, op_swz)` field pair whose decoded RSWZ2 swizzle is exactly `want`, by
/// asking the DECODER's own table for every field value.
///
/// Searching rather than transcribing is the point: there is no copy of the swizzle table in
/// this module to drift from the one the decoder reads.
fn find_rswz2_mad(which: u8, half: bool, want: [u8; 4]) -> Result<(u32, u32), AsmError> {
    for alt in 0..2u32 {
        for swz in 0..4u32 {
            if rswz2_mad_for(which, half, alt, swz) == want {
                return Ok((alt, swz));
            }
        }
    }
    Err(AsmError::Swizzle { which, want })
}

/// The same search for the 0x08/0x10 groups' operand-2 table.
fn find_rswz2_op2(want: [u8; 4]) -> Result<(u32, u32), AsmError> {
    for alt in 0..4u32 {
        for swz in 0..4u32 {
            if rswz2_op2_for(alt, swz) == want {
                return Ok((alt, swz));
            }
        }
    }
    Err(AsmError::Swizzle { which: 2, want })
}

/// The `(m3, m2, m1, en)` field values producing exactly `want` in the 0x08/0x10 groups, found
/// the same way.
fn find_mask_08(want: [bool; 4]) -> Result<(u32, u32, u32, u32), AsmError> {
    for raw in 0..16u32 {
        let (m3, m2, m1, en) = ((raw >> 3) & 1, (raw >> 2) & 1, (raw >> 1) & 1, raw & 1);
        if mask_table_08_for(m3, m2, m1, en) == want {
            return Ok((m3, m2, m1, en));
        }
    }
    Err(AsmError::WriteMask(want))
}

/// The `(swz_mask16, swz_mask32, swz_en)` values producing exactly `want` in group 0x00.
///
/// Not every mask exists here - a 32-bit mad has no bit for channel 3 - and the refusal is
/// the honest answer. See [`mask_table_mad_for`] for why the three bits are one per-lane
/// bitmask rather than two tables.
fn find_mask_mad(half: bool, want: [bool; 4]) -> Result<(u32, u32, u32), AsmError> {
    for raw in 0..8u32 {
        let (m16, m32, en) = ((raw >> 2) & 1, (raw >> 1) & 1, raw & 1);
        if mask_table_mad_for(half, m16, m32, en) == want {
            return Ok((m16, m32, en));
        }
    }
    Err(AsmError::WriteMask(want))
}

/// The operand the decoder should produce for `src` - what [`verify`] compares against.
fn expected_operand(src: &Src) -> Operand {
    let (abs, neg) = src.mods();
    let (bank, index, bank_sel) = match *src {
        Src::Reg { bank, index, .. } => {
            let sel = match bank {
                Bank::Temp | Bank::Internal => 0u8,
                Bank::Output => 1,
                Bank::PrimaryAttr => 2,
                _ => 3,
            };
            (bank, index, sel)
        }
        Src::Const { sel, .. } => (Bank::Constant, sel, 0b01),
        Src::Imm { value, .. } => (Bank::Immediate, value, 0b10),
        Src::Indexed { number, which, .. } => {
            (Bank::Indexed, number, if which == 0 { 0b00 } else { 0b11 })
        }
    };
    Operand { bank, index, bank_sel, swizzle: src.swizzle(), abs, neg }
}

/// Decode the assembled word and require it to be the instruction that was requested.
///
/// This is the assembler's whole correctness argument at the level of one word: whatever the
/// encoders got right or wrong about field placement, a word that does not read back as the
/// request never leaves this module.
fn verify(
    word: u64,
    op: Op,
    dest: Option<Dest>,
    mask: [bool; 4],
    srcs: &[Src],
    half: bool,
) -> Result<u64, AsmError> {
    let got = decode::decode(word);
    if got.op != op {
        return Err(AsmError::RoundTrip { what: "operation" });
    }
    if got.half_precision != half {
        return Err(AsmError::RoundTrip { what: "precision" });
    }
    if got.pred != Predicate::Always {
        return Err(AsmError::RoundTrip { what: "predicate" });
    }
    if let Some(blocked) = got.blocked {
        // A BLOCKED instruction is one the emitter will refuse, so assembling one would build
        // a case that can never run. Name it here rather than at the far end of the pipeline.
        let _ = blocked;
        return Err(AsmError::RoundTrip { what: "the decoder blocked the assembled word" });
    }
    if let Some(d) = dest {
        match got.dest {
            Some(g) if g.bank == d.bank && g.index == d.index => {}
            _ => return Err(AsmError::RoundTrip { what: "destination" }),
        }
    }
    if got.write_mask != mask {
        return Err(AsmError::RoundTrip { what: "write mask" });
    }
    if got.srcs.len() != srcs.len() {
        return Err(AsmError::RoundTrip { what: "source count" });
    }
    for (g, want) in got.srcs.iter().zip(srcs) {
        if *g != expected_operand(want) {
            return Err(AsmError::RoundTrip { what: "source operand" });
        }
    }
    Ok(word)
}

/// Assemble a group-0x08/0x10 vector-ALU instruction: `dest = op(src1, src2)`.
///
/// `half` selects the F16 pipeline (opcode1 0x02) over the F32 one (0x01). `op` must be one of
/// the eight this group's three-bit `opcode2` names.
pub fn alu(
    op: Op,
    half: bool,
    dest: Dest,
    mask: [bool; 4],
    src1: Src,
    src2: Src,
) -> Result<u64, AsmError> {
    let opcode2 = match op {
        Op::Mul => 0u32,
        Op::Add => 1,
        Op::Frc => 2,
        Op::Dsx => 3,
        Op::Dsy => 4,
        Op::Min => 5,
        Op::Max => 6,
        // The ALU group's own four-channel DOT - the only DOT the F16 pipeline has; group 0x18
        // is 32-bit.
        Op::Dot { components: 4 } => 7,
        _ => return Err(AsmError::UnsupportedOp("not a group-0x08 ALU operation")),
    };
    let (high, low) = group_tables("grp08_alu");
    let (mut hi, mut lo) = (0u32, 0u32);

    set_field(&mut hi, high, "opcode1", if half { 0x02 } else { 0x01 })?;
    set_field(&mut hi, high, "predicate", 0)?;
    set_field(&mut lo, low, "opcode2", opcode2)?;

    let (opt0, op0) = dest.fields6()?;
    set_field(&mut hi, high, "opt0", opt0)?;
    set_field(&mut lo, low, "op0", op0)?;
    set_field(&mut hi, high, "alt_opt0", 0)?;

    // src1: a PRECISE per-channel swizzle (three selector bits per channel), split across the
    // two halves of the word.
    let (alt1, opt1, n1) = src1.fields6(1)?;
    set_field(&mut hi, high, "alt_opt1", alt1)?;
    set_field(&mut lo, low, "opt1", opt1)?;
    set_field(&mut lo, low, "op1", n1)?;
    let s1 = src1.swizzle();
    set_field(&mut lo, low, "op1_swz_c0", u32::from(s1[0]))?;
    set_field(&mut lo, low, "op1_swz_c1", u32::from(s1[1]))?;
    set_field(&mut hi, high, "op1_swz_c2x", u32::from(s1[2]) >> 1)?;
    set_field(&mut lo, low, "op1_swz_c20", u32::from(s1[2]) & 1)?;
    set_field(&mut hi, high, "op1_swz_c3x", u32::from(s1[3]) >> 1)?;
    set_field(&mut hi, high, "op1_swz_c30", u32::from(s1[3]) & 1)?;
    let (abs1, neg1) = src1.mods();
    set_field(&mut hi, high, "abs_op1", u32::from(abs1))?;
    set_field(&mut hi, high, "neg_op1", u32::from(neg1))?;

    // src2: a TABLE swizzle, and no negate modifier exists for it in this group.
    let (alt2, opt2, n2) = src2.fields6(2)?;
    set_field(&mut hi, high, "alt_opt2", alt2)?;
    set_field(&mut lo, low, "opt2", opt2)?;
    set_field(&mut lo, low, "op2", n2)?;
    let (swz_alt2, op2_swz) = find_rswz2_op2(src2.swizzle())?;
    set_field(&mut hi, high, "swz_alt_op2", swz_alt2)?;
    set_field(&mut hi, high, "op2_swz", op2_swz)?;
    let (abs2, neg2) = src2.mods();
    if neg2 {
        return Err(AsmError::UnsupportedOp("group 0x08 has no negate modifier for src2"));
    }
    set_field(&mut hi, high, "abs_op2", u32::from(abs2))?;

    let (m3, m2, m1, en) = find_mask_08(mask)?;
    set_field(&mut hi, high, "swz_mask3", m3)?;
    set_field(&mut hi, high, "swz_mask2", m2)?;
    set_field(&mut hi, high, "swz_mask1", m1)?;
    set_field(&mut hi, high, "swz_en", en)?;

    let word = (u64::from(hi) << 32) | u64::from(lo);
    verify(word, op, Some(dest), mask, &[src1, src2], half)
}

/// Assemble a group-0x00 multiply-add: `dest = src1 * src2 + src3`.
///
/// `src1`'s bank field is ONE bit wide here, so it reaches Temp and PrimaryAttr only, and it
/// carries an absolute modifier but no negate - both are properties of the encoding, and both
/// are refused rather than approximated.
pub fn mad(
    half: bool,
    dest: Dest,
    mask: [bool; 4],
    src1: Src,
    src2: Src,
    src3: Src,
) -> Result<u64, AsmError> {
    let (high, low) = group_tables("grp00_mad");
    let (mut hi, mut lo) = (0u32, 0u32);

    set_field(&mut hi, high, "opcode1", 0x00)?;
    set_field(&mut hi, high, "data_format", u32::from(half))?;
    set_field(&mut hi, high, "predicate", 0)?;

    let (opt0, op0) = dest.fields6()?;
    set_field(&mut hi, high, "opt0", opt0)?;
    set_field(&mut lo, low, "op0", op0)?;
    set_field(&mut hi, high, "alt_opt0", 0)?;

    // src1: one bank bit (0 = Temp, 1 = PrimaryAttr), a table swizzle, abs but no negate.
    let opt1 = match src1 {
        Src::Reg { bank: Bank::Temp, .. } => 0u32,
        Src::Reg { bank: Bank::PrimaryAttr, .. } => 1,
        Src::Reg { bank, .. } => return Err(AsmError::SrcBank { which: 1, bank }),
        _ => {
            return Err(AsmError::UnsupportedOp(
                "group 0x00 src1 has no exotic-operand row: it is a plain register",
            ))
        }
    };
    let (_, _, n1) = src1.fields6(1)?;
    set_field(&mut hi, high, "opt1", opt1)?;
    set_field(&mut lo, low, "op1", n1)?;
    let (swz_alt1, op1_swz) = find_rswz2_mad(1, half, src1.swizzle())?;
    set_field(&mut hi, high, "swz_alt_op1", swz_alt1)?;
    set_field(&mut lo, low, "op1_swz", op1_swz)?;
    let (abs1, neg1) = src1.mods();
    if neg1 {
        return Err(AsmError::UnsupportedOp("group 0x00 has no negate modifier for src1"));
    }
    set_field(&mut hi, high, "abs_op1", u32::from(abs1))?;

    for (which, src, alt_name, opt_name, op_name, swz_alt_name, swz_name, abs_name, neg_name) in [
        (2u8, src2, "alt_opt2", "opt2", "op2", "swz_alt_op2", "op2_swz", "abs_op2", "neg_op2"),
        (3u8, src3, "alt_opt3", "opt3", "op3", "swz_alt_op3", "op3_swz", "abs_op3", "neg_op3"),
    ] {
        let (alt, opt, n) = src.fields6(which)?;
        set_field(&mut hi, high, alt_name, alt)?;
        set_field(&mut lo, low, opt_name, opt)?;
        set_field(&mut lo, low, op_name, n)?;
        let (swz_alt, swz) = find_rswz2_mad(which, half, src.swizzle())?;
        set_field(&mut hi, high, swz_alt_name, swz_alt)?;
        // op3's swizzle field is in the HIGH word and op2's is in the LOW word: the two are
        // not symmetric in this group, so each is written where its own table puts it.
        if which == 2 {
            set_field(&mut lo, low, swz_name, swz)?;
        } else {
            set_field(&mut hi, high, swz_name, swz)?;
        }
        let (abs, neg) = src.mods();
        set_field(&mut hi, high, abs_name, u32::from(abs))?;
        set_field(&mut hi, high, neg_name, u32::from(neg))?;
    }

    let (m16, m32, en) = find_mask_mad(half, mask)?;
    set_field(&mut hi, high, "swz_mask16", m16)?;
    set_field(&mut hi, high, "swz_mask32", m32)?;
    set_field(&mut hi, high, "swz_en", en)?;

    let word = (u64::from(hi) << 32) | u64::from(lo);
    verify(word, Op::Mad, Some(dest), mask, &[src1, src2, src3], half)
}

/// [`alu`], PREDICATED: the write happens only where `pred` holds.
///
/// The group's `predicate` field is the 3-bit ExtVecPredicate - `P0..P2` and their NEGATIONS,
/// and no `P3` - which is not the table the other groups use, and reading one for the other
/// silently inverts a condition. So the value is SEARCHED against the decoder rather than
/// restated: whichever field value decodes to `pred` is the one written, and a predicate the
/// field cannot express is refused.
///
/// The rest of the word must decode exactly as the unpredicated one did - the predicate is the
/// only thing this is allowed to change.
pub fn alu_pred(
    op: Op,
    half: bool,
    pred: Predicate,
    dest: Dest,
    mask: [bool; 4],
    src1: Src,
    src2: Src,
) -> Result<u64, AsmError> {
    let plain = alu(op, half, dest, mask, src1, src2)?;
    let (high, _) = group_tables("grp08_alu");
    let want = decode::decode(plain);
    for raw in 0..8u32 {
        let mut hi = (plain >> 32) as u32;
        set_field(&mut hi, high, "predicate", raw)?;
        let word = (u64::from(hi) << 32) | (plain & 0xffff_ffff);
        let got = decode::decode(word);
        if got.pred == pred && got.blocked.is_none() {
            if got.op != want.op || got.dest != want.dest || got.srcs != want.srcs || got.write_mask != want.write_mask {
                return Err(AsmError::RoundTrip { what: "setting the predicate moved another field" });
            }
            return Ok(word);
        }
    }
    Err(AsmError::UnsupportedOp("this predicate has no ExtVecPredicate encoding"))
}

/// `dest.mask = src` as an ADD OF ZERO. The vector-ALU group has no move opcode, and this is
/// the shape the encoding actually permits.
///
/// **Which operand carries the zero is forced by the encoding, and getting it the wrong way
/// round does not assemble at all.** Source 1 of this group has a free per-channel swizzle -
/// three selector bits per channel, so selectors 4..7 reach the inline constants 0.0/1.0/2.0/
/// 0.5 - while source 2's swizzle comes from a sixteen-entry TABLE that holds lane patterns and
/// one trailing `1`. So the inline zero can only be source 1, and the value being moved can
/// only be source 2. `add` is commutative, so the operation is the move either way; the
/// encoding is not.
///
/// A move written this way also exercises the per-channel constant selector on every case that
/// uses it - the first defect the corpus differential found in the reference.
pub fn mov(half: bool, dest: Dest, mask: [bool; 4], src: Src) -> Result<u64, AsmError> {
    alu(Op::Add, half, dest, mask, Src::cnst(0).swz([4, 4, 4, 4]), src)
}

/// The four-channel swizzle a group-0x00 mad can give an F32 operand, from its own tables:
/// `xy`, padded. The F32 mad is a TWO-LANE operation (its write mask reaches lanes 0..2 and its
/// operand tables hold two-lane patterns), so a caller asking for `xyzw` there is asking for
/// something the encoding does not have.
pub const MAD_F32_XY: [u8; 4] = [0, 1, 0, 0];

/// The numeric format a [`pack`] operand is stored in. These are the encoding's own format
/// numbers, so a case can say exactly which conversion it means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackFmt {
    U8,
    S8,
    U16,
    S16,
    F16,
    F32,
}

impl PackFmt {
    fn bits(self) -> u64 {
        match self {
            PackFmt::U8 => 0,
            PackFmt::S8 => 1,
            PackFmt::U16 => 3,
            PackFmt::S16 => 4,
            PackFmt::F16 => 5,
            PackFmt::F32 => 6,
        }
    }
}

/// Assemble a group-0x40 PACK: a format conversion `dest = convert(src)`, per channel.
///
/// This is the instruction a fragment program spends most of its life in - real titles are
/// 70-90% F16, and every value that crosses between the F32 and F16 pipelines crosses here. It
/// is also where a wrong number is least visible: a conversion that reads the wrong half of a
/// packed register yields a denormal or a huge number rather than an error.
///
/// `scale` selects the NORMALIZED conversion (integer 0..max against float 0..1) rather than a
/// plain numeric cast; the two are different instructions sharing an opcode, and a case that
/// means one must not be able to assemble the other by accident.
///
/// **The destination is R7 (a direct register number) and the source is R6 (double-register).**
/// They are not the same numbering, and that asymmetry is the encoding's, not this function's -
/// which is why the destination accepts an odd register and the source does not.
pub fn pack(
    dest: Dest,
    dest_fmt: PackFmt,
    src_bank: Bank,
    src_reg: u8,
    src_fmt: PackFmt,
    swizzle: [u8; 4],
    mask: [bool; 4],
    scale: bool,
) -> Result<u64, AsmError> {
    // >>> AN INTERNAL DESTINATION IS THE RESERVED TOP OF THE **TEMP** SELECTOR'S RANGE, so its
    // selector is Temp's and only its NUMBER says it is internal - exactly as `Dest::fields6`
    // reads it for the six-bit groups. This once fell through to the catch-all and refused every
    // internal destination while the number field below went to the trouble of encoding one; the
    // encodable-space sweep is what noticed, because no hand-written case had ever asked.
    let dest_sel = match dest.bank {
        Bank::Temp | Bank::Internal => 0u64,
        Bank::Output => 1,
        Bank::PrimaryAttr => 2,
        other => return Err(AsmError::DestBank(other)),
    };
    // The destination number is DIRECT here (seven bits), with the top four values reserved for
    // the internal registers - so an internal destination is that reserved range, not a bank.
    let dest_n = match dest.bank {
        Bank::Internal => {
            let n = dest.index / 4;
            if !dest.index.is_multiple_of(4) || n > 3 {
                return Err(AsmError::RegIndex { bank: dest.bank, index: dest.index });
            }
            124 + u64::from(n)
        }
        _ => {
            if dest.index >= 124 {
                return Err(AsmError::RegIndex { bank: dest.bank, index: dest.index });
            }
            u64::from(dest.index)
        }
    };
    let src_sel = match src_bank {
        Bank::Temp => 0u64,
        Bank::Output => 1,
        Bank::PrimaryAttr => 2,
        Bank::SecondaryAttr => 3,
        other => return Err(AsmError::SrcBank { which: 1, bank: other }),
    };
    if !src_reg.is_multiple_of(2) || src_reg / 2 >= 60 {
        return Err(AsmError::RegIndex { bank: src_bank, index: src_reg });
    }
    if swizzle.iter().any(|&s| s > 3) {
        // The component selectors here are two bits (plus one high bit for comp0): they name a
        // component of the source vector and cannot name an inline constant.
        return Err(AsmError::Swizzle { which: 1, want: swizzle });
    }

    let mut word: u64 = 0x08 << 59;
    word |= src_fmt.bits() << 41;
    word |= dest_fmt.bits() << 38;
    word |= u64::from(scale) << 18;
    word |= dest_sel << 32;
    word |= dest_n << 21;
    word |= src_sel << 30;
    word |= u64::from(src_reg / 2) << 8;
    // >>> AN F32 SOURCE IS A REGISTER PAIR PER FIELD, and components 2 and 3 come from the
    // SECOND field (`src2_n`, 6:1, bank 29:28) - see `decode_grp_pack`. The compiler sets it to
    // the NEXT pair for a plain vec4, which is the vector this function means. It used to leave
    // the field zero, so any F32 source reading `.z`/`.w` read them from `r[0]` instead, and the
    // round trip below compared the first source only and never noticed.
    if src_fmt == PackFmt::F32 {
        if src_reg / 2 + 1 >= 64 {
            return Err(AsmError::RegIndex { bank: src_bank, index: src_reg });
        }
        word |= src_sel << 28;
        word |= u64::from(src_reg / 2 + 1) << 1;
    }
    // Component selectors: comp0's low bit is bit 0; its HIGH bit is at whichever position the
    // decoder reads it from for this source format, which is a fact of the format and not of
    // this function - `decode::pack_comp0_high_bit` is the single statement of it, and asking
    // it rather than restating it is the rule the whole assembler is built on. Comps 1..3 have
    // their own two-bit fields.
    let c0 = u64::from(swizzle[0]);
    word |= c0 & 1;
    word |= ((c0 >> 1) & 1) << decode::pack_comp0_high_bit(src_fmt.bits() as u32);
    word |= u64::from(swizzle[1]) << 16;
    word |= u64::from(swizzle[2]) << 14;
    word |= u64::from(swizzle[3]) << 19;
    let mask_bits = (0..4).fold(0u64, |acc, c| acc | (u64::from(mask[c]) << c));
    word |= mask_bits << 34;

    let got = decode::decode(word);
    if got.blocked.is_some() {
        return Err(AsmError::RoundTrip { what: "the decoder blocked the assembled pack" });
    }
    if got.write_mask != mask {
        return Err(AsmError::RoundTrip { what: "pack write mask" });
    }
    match got.dest {
        Some(d) if d.bank == dest.bank && d.index == dest.index => {}
        _ => return Err(AsmError::RoundTrip { what: "pack destination" }),
    }
    // ONE source: the decoder adds a second only when the pair is NOT contiguous, and a word that
    // reads its upper components from anywhere but `src + 2` is not the vec4 asked for.
    match got.srcs.as_slice() {
        [s] if s.bank == src_bank && s.index == src_reg && s.swizzle == swizzle => {}
        _ => return Err(AsmError::RoundTrip { what: "pack source" }),
    }
    Ok(word)
}

/// Assemble a group-0x09 VTST: `p<pdst> = (alu(src1, src2) cmp 0)`, the FLOAT families only.
///
/// >>> WHY THIS GROUP, AND WHY NOW. A hardwired `Prec::F32` in the reference's float-TEST arm
/// was the FIRST divergence in five corpus programs across three titles: a VTST in the F16
/// pipeline had its operands read as one 32-bit float by the oracle and as two HALVES by the
/// shader, and the result is a PREDICATE - one bit that picks which arm of a branch runs. A unit
/// test pins the reference's rule, but an AUTHORED case could not be written at all, because
/// this group was not assemblable. A family that has produced a real defect and cannot be
/// written a case for is the worst combination there is.
///
/// # Only the FLOAT families, deliberately
/// `Add`/`Sub`/`Mul` at either width are the arms whose `(alu_sel, alu_op)` numbering the
/// decoder states outright. The integer and bitwise families' numbering is partly INFERRED -
/// the decoder's own comments say which parts and what would refute them - so assembling one
/// here would be this file asserting a fact it does not have. Refused by name instead, which is
/// how three of this assembler's real ISA facts were learned.
///
/// The PRECISION BIT IS INVERTED against the obvious reading: `prec == 0` is HALF
/// (`decode_grp_test` reads `half_precision = prec == 0`), so full precision SETS bit 47. That
/// is exactly the kind of detail a second transcription gets backwards, which is why the
/// round-trip below is not optional.
pub fn vtst(
    alu: TestAlu,
    cmp: TestCmp,
    reduce: TestReduce,
    pdst: u8,
    half: bool,
    s1_bank: Bank,
    s1_reg: u8,
    s2_bank: Bank,
    s2_reg: u8,
) -> Result<u64, AsmError> {
    // The FLOAT families only - see the note above.
    let (alu_sel, alu_op) = match alu {
        TestAlu::Add => (0u64, 2u64),
        TestAlu::Mul => (0, 13),
        TestAlu::Sub => (0, 14),
        _ => return Err(AsmError::RoundTrip { what: "vtst: only the FLOAT ALU families are assembled" }),
    };
    // `(sign_test, zero_test, combiner)`, the three fields the decoder folds into one relation.
    let (sign_test, zero_test, combiner) = match cmp {
        TestCmp::Eq => (0u64, 1u64, 0u64),
        TestCmp::Ne => (0, 2, 0),
        TestCmp::Lt => (1, 0, 0),
        TestCmp::Gt => (2, 0, 0),
        TestCmp::Le => (1, 1, 0),
        TestCmp::Ge => (2, 1, 0),
    };
    let chan_cc = match reduce {
        TestReduce::Channel(c) if c < 4 => u64::from(c),
        TestReduce::AndAll => 4,
        TestReduce::OrAll => 5,
        TestReduce::Channel(_) => {
            return Err(AsmError::RoundTrip { what: "vtst: a channel reduction names 0..3" })
        }
    };
    if pdst > 3 {
        return Err(AsmError::RoundTrip { what: "vtst: there are four predicate registers" });
    }
    // A float ALU's source number is DOUBLE-REGISTER scaled, so an odd register cannot be
    // named - the same refusal the other groups make rather than rounding it.
    let src_sel = |b: Bank| -> Result<u64, AsmError> {
        match b {
            Bank::Temp => Ok(0),
            Bank::Output => Ok(1),
            Bank::PrimaryAttr => Ok(2),
            Bank::SecondaryAttr => Ok(3),
            other => Err(AsmError::SrcBank { which: 1, bank: other }),
        }
    };
    let field = |reg: u8, bank: Bank| -> Result<u64, AsmError> {
        if !reg.is_multiple_of(2) || reg / 2 >= 124 {
            return Err(AsmError::RegIndex { bank, index: reg });
        }
        Ok(u64::from(reg / 2))
    };
    let (b1, b2) = (src_sel(s1_bank)?, src_sel(s2_bank)?);
    let (n1, n2) = (field(s1_reg, s1_bank)?, field(s2_reg, s2_bank)?);
    if n1 > 0x7f || n2 > 0x7f {
        return Err(AsmError::RegIndex { bank: s1_bank, index: s1_reg });
    }

    let mut word: u64 = 0x09 << 59;
    // `prec == 0` is HALF.
    word |= u64::from(!half) << 47;
    word |= sign_test << 42;
    word |= zero_test << 40;
    word |= combiner << 39;
    word |= chan_cc << 36;
    word |= u64::from(pdst) << 34;
    word |= b1 << 30;
    word |= b2 << 28;
    word |= alu_sel << 18;
    word |= alu_op << 14;
    word |= n1 << 7;
    word |= n2;

    // >>> AND IT IS DECODED BACK. Every field above is a second transcription of the decoder's
    // reading, and the inverted precision bit is exactly the sort that comes out backwards.
    let got = decode::decode(word);
    if got.blocked.is_some() {
        return Err(AsmError::RoundTrip { what: "the decoder blocked the assembled vtst" });
    }
    match got.op {
        Op::Test { alu: a, cmp: c, reduce: r, pdst: pd, .. }
            if a == alu && c == cmp && r == reduce && pd == pdst => {}
        _ => return Err(AsmError::RoundTrip { what: "vtst operation fields" }),
    }
    if got.half_precision != half {
        return Err(AsmError::RoundTrip { what: "vtst precision" });
    }
    match (got.srcs.first(), got.srcs.get(1)) {
        (Some(a), Some(b))
            if a.bank == s1_bank && a.index == s1_reg && b.bank == s2_bank && b.index == s2_reg => {}
        _ => return Err(AsmError::RoundTrip { what: "vtst sources" }),
    }
    Ok(word)
}

/// Assemble the shipped FACING TEST: `p<pdst> = (GLOBAL[global] & literal) != 0`, VTST's
/// bitwise-AND family.
///
/// This is the whole of the corpus's GLOBAL use - `GLOBAL[16] & 1`, the per-fragment facing
/// bit - so it is assembled as that idiom rather than as a general bitwise test: `src1` is the
/// extension row's SPECIAL bank with the GLOBAL discriminator (`0x40`) set, and `src2` the inline
/// literal row, whose 7-bit number is the value. The family's operands are not doubled.
pub fn vtst_global_bit(pdst: u8, global: u8, literal: u8) -> Result<u64, AsmError> {
    if pdst > 3 || global > 0x3f || literal > 0x7f {
        return Err(AsmError::UnsupportedOp("vtst_global_bit: p0..p3, GLOBAL[0..63], a 7-bit literal"));
    }
    let mut w = 0u64;
    put(&mut w, 63, 59, 0x09, "opcode")?;
    put(&mut w, 49, 49, 1, "src1_ext")?;
    put(&mut w, 48, 48, 1, "src2_ext")?;
    put(&mut w, 41, 40, 2, "zero_test")?;
    put(&mut w, 35, 34, u64::from(pdst), "pdst")?;
    put(&mut w, 31, 30, 1, "src1_sel")?;
    put(&mut w, 29, 28, 2, "src2_sel")?;
    put(&mut w, 19, 18, 3, "alu_sel")?;
    put(&mut w, 13, 7, 0x40 | u64::from(global), "src1_n")?;
    put(&mut w, 6, 0, u64::from(literal), "src2_n")?;
    let got = decode::decode(w);
    if got.blocked.is_some() {
        return Err(AsmError::RoundTrip { what: "the decoder blocked the assembled facing test" });
    }
    match got.op {
        Op::Test { alu: TestAlu::BitAnd, cmp: TestCmp::Ne, reduce: TestReduce::Channel(0), pdst: p, write_back: false }
            if p == pdst => {}
        _ => return Err(AsmError::RoundTrip { what: "facing test operation" }),
    }
    match (got.srcs.first(), got.srcs.get(1)) {
        (Some(a), Some(b))
            if a.bank == Bank::Global && a.index == global && b.bank == Bank::Immediate && b.index == literal => {}
        _ => return Err(AsmError::RoundTrip { what: "facing test operands" }),
    }
    Ok(w)
}

/// Assemble a group-0xE0 TEXTURE SAMPLE: `dest.xyzw = sample(unit, coords)`.
///
/// The sampled RGBA lands in the destination's FOUR channels whatever the coordinate count -
/// the destination extent is part of what the opcode means, and a translation that wrote fewer
/// would leave the channels after it holding whatever was there.
///
/// `coords` is 1 or 2 components, read from `coord_reg` onward. This is the implicit-LOD,
/// ordinary-sample form (`sb_mode` 0, `lod_mode` 0) - the one a fragment shader spends; the
/// gather and explicit-level variants are different destination layouts and are not assembled
/// here.
///
/// **`sampler_ordinal` IS NOT THE TEXTURE UNIT.** The instruction names an SA register - the
/// DOUBLE of this ordinal - and which unit's control words live there is stated by the
/// container's texture-control table, not by the instruction. A program that samples without an
/// entry in that table is refused by name, which is why a caller must write one; this function
/// cannot check it, because a bare word has no container.
pub fn tex(
    dest: Dest,
    sampler_ordinal: u8,
    coord_bank: Bank,
    coord_reg: u8,
    coords: u8,
) -> Result<u64, AsmError> {
    tex_lod(dest, sampler_ordinal, coord_bank, coord_reg, coords, TexLod::Implicit, None)
}

/// [`tex`], with the mip level supplied the way `lod_mode` (41:40) says: a BIAS or a LEVEL
/// reads one F32 scalar from `lod_src`, a GRADIENT reads `ddx.xy` then `ddy.xy` from its four
/// channels (2D only - the decoder blocks any other gradient). `lod_src` must be `None` exactly
/// when `lod` is [`TexLod::Implicit`].
///
/// The `src2` field is the group's double-register one - bank at [29:28], number at [6:0] - so an
/// odd register is refused, like every other doubled field here.
pub fn tex_lod(
    dest: Dest,
    sampler_ordinal: u8,
    coord_bank: Bank,
    coord_reg: u8,
    coords: u8,
    lod: TexLod,
    lod_src: Option<(Bank, u8)>,
) -> Result<u64, AsmError> {
    tex_typed(dest, sampler_ordinal, coord_bank, coord_reg, coords, false, false, lod, lod_src)
}

/// [`tex_lod`] with the two type fields free: `coord_half` is `src0_type` (36:35, F16 = 1) - the
/// coordinate read as F16 halves - and `result_half` is `fconv_type` (47:46, F16 = 2) - the RGBA
/// stored as two packed pairs. They are INDEPENDENT in the encoding: a shader routinely computes
/// an F16 UV and asks for an F32 result, and reading either at the wrong width samples or stores
/// garbage.
#[allow(clippy::too_many_arguments)]
pub fn tex_typed(
    dest: Dest,
    sampler_ordinal: u8,
    coord_bank: Bank,
    coord_reg: u8,
    coords: u8,
    coord_half: bool,
    result_half: bool,
    lod: TexLod,
    lod_src: Option<(Bank, u8)>,
) -> Result<u64, AsmError> {
    if !(1..=2).contains(&coords) {
        return Err(AsmError::UnsupportedOp("a sample coordinate is one or two components here"));
    }
    if matches!(lod, TexLod::Implicit) != lod_src.is_none() {
        return Err(AsmError::UnsupportedOp("an implicit sample takes no LOD operand, and every other form takes one"));
    }
    if matches!(lod, TexLod::Gradient) && coords != 2 {
        return Err(AsmError::UnsupportedOp("a gradient sample is 2D only"));
    }
    let lod_mode = match lod {
        TexLod::Implicit => 0u64,
        TexLod::Bias => 1,
        TexLod::Level => 2,
        TexLod::Gradient => 3,
    };
    let (src2_sel, src2_n) = match lod_src {
        None => (0u64, 0u64),
        Some((bank, index)) => {
            let sel = match bank {
                Bank::Temp => 0u64,
                Bank::Output => 1,
                Bank::PrimaryAttr => 2,
                Bank::SecondaryAttr => 3,
                other => return Err(AsmError::SrcBank { which: 2, bank: other }),
            };
            // Doubled, and the top four field values of the Temp row are the internal registers.
            if !index.is_multiple_of(2) || (sel == 0 && index / 2 >= 124) {
                return Err(AsmError::RegIndex { bank, index });
            }
            (sel, u64::from(index / 2))
        }
    };
    // The destination bank is a single bit: PrimaryAttr when set, Temp when clear.
    let dest_use_pa = match dest.bank {
        Bank::Temp => 0u64,
        Bank::PrimaryAttr => 1,
        other => return Err(AsmError::DestBank(other)),
    };
    if dest.index >= 124 {
        return Err(AsmError::RegIndex { bank: dest.bank, index: dest.index });
    }
    let coord_bank_bit = match coord_bank {
        Bank::Temp => 0u64,
        Bank::PrimaryAttr => 1,
        other => return Err(AsmError::SrcBank { which: 0, bank: other }),
    };
    // The coordinate field is DOUBLE-register scaled (the decoder reads `reg_index`), so an odd
    // register has no encoding. This encoder used to write the register number undoubled, and
    // the round trip did not check the coordinate - every caller named register 0, where the
    // two readings coincide, so nothing noticed.
    if !coord_reg.is_multiple_of(2) || (coord_bank == Bank::Temp && coord_reg / 2 >= 124) {
        return Err(AsmError::RegIndex { bank: coord_bank, index: coord_reg });
    }

    let mut word: u64 = 0x1c << 59;
    word |= u64::from(coords - 1) << 42; // `dim`, base-0
    word |= lod_mode << 40;
    word |= u64::from(coord_reg / 2) << 14;
    word |= coord_bank_bit << 34;
    word |= u64::from(sampler_ordinal) << 7;
    word |= u64::from(dest.index) << 21;
    word |= dest_use_pa << 39;
    word |= src2_sel << 28;
    word |= src2_n;
    word |= u64::from(coord_half) << 35;
    word |= if result_half { 2u64 << 46 } else { 0 };

    let got = decode::decode(word);
    if got.blocked.is_some() {
        return Err(AsmError::RoundTrip { what: "the decoder blocked the assembled sample" });
    }
    match got.op {
        // A bare word carries the RAW ordinal in `unit`; the container's texture-control table
        // is what turns it into a texture unit, and `decode_shader` does that with the program
        // in hand. So this checks the field, which is all a bare word can state.
        Op::Tex { unit: u, coords: c, lod: l, coord_half: ch }
            if u == sampler_ordinal && c == coords && l == lod && ch == coord_half => {}
        _ => return Err(AsmError::RoundTrip { what: "sample ordinal, coordinate count, coordinate type or LOD mode" }),
    }
    if got.half_precision != result_half {
        return Err(AsmError::RoundTrip { what: "sample result type" });
    }
    match got.dest {
        Some(d) if d.bank == dest.bank && d.index == dest.index => {}
        _ => return Err(AsmError::RoundTrip { what: "sample destination" }),
    }
    match got.srcs.first() {
        Some(s) if s.bank == coord_bank && s.index == coord_reg => {}
        _ => return Err(AsmError::RoundTrip { what: "sample coordinate register" }),
    }
    match (lod_src, got.srcs.get(1)) {
        (None, None) => {}
        (Some((b, i)), Some(s)) if s.bank == b && s.index == i => {}
        _ => return Err(AsmError::RoundTrip { what: "sample LOD operand" }),
    }
    Ok(word)
}

/// Assemble a group-0xE0 GATHER (`sb_mode` 3): the 2x2 footprint's four texels of ONE
/// component into `dest..dest+4` at full precision, and the four F16 bilinear coefficients into
/// the two registers after them. It is the implicit-LOD 2D sample word with the sub-behaviour
/// field set, which is all the decoder distinguishes it by.
pub fn tex_gather(dest: Dest, sampler_ordinal: u8, coord_bank: Bank, coord_reg: u8) -> Result<u64, AsmError> {
    let base = tex_lod(dest, sampler_ordinal, coord_bank, coord_reg, 2, TexLod::Implicit, None)?;
    let word = base | (0b11u64 << 37);
    let got = decode::decode(word);
    if got.blocked.is_some() {
        return Err(AsmError::RoundTrip { what: "the decoder blocked the assembled gather" });
    }
    match got.op {
        Op::TexGather { unit, coords: 2, .. } if unit == sampler_ordinal => {}
        _ => return Err(AsmError::RoundTrip { what: "gather operation" }),
    }
    match (got.dest, got.srcs.first()) {
        (Some(d), Some(s))
            if d.bank == dest.bank && d.index == dest.index && s.bank == coord_bank && s.index == coord_reg => {}
        _ => return Err(AsmError::RoundTrip { what: "gather operands" }),
    }
    Ok(word)
}

/// Assemble a group-0x14 INDEX-REGISTER LOAD: `i0 = int(src) + addend`.
///
/// This is the instruction a shader emits before reading a uniform ARRAY at a computed index,
/// and it is the one a football title's stadium crowd turns on: six-vertex sprites carrying a
/// corner number, a corner table in the literals, and an indexed read to pick the row.
///
/// # Why this encoder is a TEMPLATE and not a field layout
///
/// The ISA reference this project works from does not carry group 0x14 at all - it is listed
/// among its own open questions - so the corpus is the entire authority, and what the corpus
/// establishes is ONE encoding plus the handful of fields measured to vary inside it. Building
/// a word from scratch here would mean inventing values for bits nothing has ever explained.
/// So the word starts as the established one and only the proven fields move, and the decoder's
/// own `i16mad_is_load_index` is what confirms the result is still that encoding - by way of
/// [`verify`], which refuses a blocked instruction.
///
/// `src_bank` is Temp or PrimaryAttr (the group's one-bit source bank table). The source
/// register is read as an INTEGER from the low 16 bits of the register's bit pattern, which is
/// what the shader that computed the index left there.
pub fn load_index(src_bank: Bank, src_reg: u8, addend: u8) -> Result<u64, AsmError> {
    let (template, variable) = decode::i16mad_load_index_template();
    let bank_bit = match src_bank {
        Bank::Temp => 0u64,
        Bank::PrimaryAttr => 1,
        other => return Err(AsmError::SrcBank { which: 0, bank: other }),
    };
    if src_reg > 0x3f {
        return Err(AsmError::RegIndex { bank: src_bank, index: src_reg });
    }
    if addend > 0x7f {
        return Err(AsmError::FieldTooWide { field: "addend", width: 7, value: u32::from(addend) });
    }
    // Start from the established word, clear every field the corpus proved may vary, then set
    // the ones this instruction means - so any bit outside the mask keeps the template's value
    // and the decoder still recognises the encoding.
    let mut word = template & !variable;
    word |= u64::from(src_reg) << 14;
    word |= u64::from(addend);
    word |= bank_bit << 34;
    // `(b8, b51)` is ONE field naming the DESTINATION: (0, 1) writes the INDEX REGISTER, which
    // is the form an indexed uniform read consumes. (1, 0) writes an ordinary register instead
    // and is a different instruction wearing the same opcode.
    word |= 1 << 51;

    let got = decode::decode(word);
    match got.op {
        Op::LoadIndex { addend: a, to_index: true, .. } if a == i32::from(addend) => {}
        _ => return Err(AsmError::RoundTrip { what: "index-load operation or addend" }),
    }
    if got.blocked.is_some() {
        return Err(AsmError::RoundTrip { what: "the decoder did not recognise the index-load encoding" });
    }
    match got.dest {
        Some(d) if d.bank == Bank::Index && d.index == 0 => {}
        _ => return Err(AsmError::RoundTrip { what: "index-load destination" }),
    }
    match got.srcs.first() {
        Some(s) if s.bank == src_bank && s.index == src_reg => {}
        _ => return Err(AsmError::RoundTrip { what: "index-load source" }),
    }
    Ok(word)
}

/// Assemble a **KILL** (fragment discard), optionally predicated.
///
/// >>> WHY THIS IS HERE AT ALL. 29 corpus blobs carry a `kill` and every one of them is an alpha
/// test, so what the render rig's discard path has ever been asked is one shape of one idiom.
/// An authored case states the INTENT before anything runs - "this program discards, and these
/// registers hold these values when it does" - so a divergence names which side is wrong rather
/// than only that one of them is.
///
/// The predicate is KILL's OWN 2-bit field at [42:41], not the group's ExtPredicate slot, and
/// its ordering is the one the corpus settles: `IfNotP(1)` is what an alpha test encodes, which
/// under the rival reading would discard exactly the texels that PASSED. Only the four the field
/// can express are encodable; `IfP(1)` and `IfNotP(2..)` have no encoding here and are refused
/// rather than approximated.
pub fn kill(pred: Predicate) -> Result<u64, AsmError> {
    let (template, variable) = decode::kill_template();
    let bits = match pred {
        Predicate::Always => 0u64,
        Predicate::IfNotP(0) => 1,
        Predicate::IfNotP(1) => 2,
        Predicate::IfP(0) => 3,
        _ => return Err(AsmError::UnsupportedOp("a predicate KILL's 2-bit field cannot express")),
    };
    let word = (template & !variable) | (bits << 41);

    let got = decode::decode(word);
    if got.op != Op::Kill {
        return Err(AsmError::RoundTrip { what: "kill operation" });
    }
    if got.blocked.is_some() {
        return Err(AsmError::RoundTrip { what: "the decoder did not recognise the kill encoding" });
    }
    if got.pred != pred {
        return Err(AsmError::RoundTrip { what: "kill predicate" });
    }
    Ok(word)
}

/// Assemble a **DEPTHF**: replace the interpolated fragment depth with a scalar register.
///
/// >>> NOT ONE CORPUS PROGRAM WRITES A FRAGMENT DEPTH. `Op::DepthF`'s reference arm and the
/// render rig's depth capture are built and pinned by a Rust test, and the GPU half of that path
/// is checked by nothing at all - which is the largest single hole left in the render rig, and
/// the only one no amount of corpus work can close.
///
/// The source is a RAW LANE at [20:14], read direct rather than double-register scaled: scaling
/// belongs to the float data types and this group carries no data-type field to select one. The
/// bank comes from the selector at [36] and its extension at [51], which between them reach all
/// four of Temp, PrimaryAttr, Output and SecondaryAttr.
///
/// The value is in the GUEST's depth space, because that is the only space a shader can compute
/// one in; turning it into whatever the pipeline rasterises is the emitter's job.
pub fn depthf(src_bank: Bank, src_reg: u8) -> Result<u64, AsmError> {
    let (template, variable) = decode::depthf_template();
    // Spec A.2, src0 bank: ext=0 -> 0 TEMP / 1 PRIMATTR; ext=1 -> 0 OUTPUT / 1 SECATTR.
    let (ext, sel) = match src_bank {
        Bank::Temp => (0u64, 0u64),
        Bank::PrimaryAttr => (0, 1),
        Bank::Output => (1, 0),
        Bank::SecondaryAttr => (1, 1),
        other => return Err(AsmError::SrcBank { which: 0, bank: other }),
    };
    // >>> THE TOP FOUR TEMP NUMBERS NAME AN INTERNAL REGISTER, NOT A TEMPORARY. The decoder
    // reads 124..=127 in the `r` bank as i0..i3, so assembling `Temp[124]` would produce a word
    // that means something else - refused, rather than emitted and round-trip-failed, so the
    // error names the cause.
    if src_reg > 0x7f || (matches!(src_bank, Bank::Temp) && src_reg >= 124) {
        return Err(AsmError::RegIndex { bank: src_bank, index: src_reg });
    }
    let word = (template & !variable) | (ext << 51) | (sel << 36) | (u64::from(src_reg) << 14);

    let got = decode::decode(word);
    if got.op != Op::DepthF {
        return Err(AsmError::RoundTrip { what: "depthf operation" });
    }
    if got.blocked.is_some() {
        return Err(AsmError::RoundTrip {
            what: "the decoder did not recognise the depthf encoding",
        });
    }
    match got.srcs.first() {
        Some(s) if s.bank == src_bank && s.index == src_reg => {}
        _ => return Err(AsmError::RoundTrip { what: "depthf source" }),
    }
    Ok(word)
}

/// The `(swz_alt_op2, op2_swz)` values producing exactly `want` in the 0x18 DOT's operand-2
/// table, found by asking the decoder's own table for every field value.
///
/// The right table depends on `c3_en`, which is not part of the swizzle - see
/// [`decode::rswz2_dot_op2_for`].
fn find_rswz2_dot_op2(c3_en: bool, want: [u8; 4]) -> Result<(u32, u32), AsmError> {
    for alt in 0..4u32 {
        for swz in 0..4u32 {
            if decode::rswz2_dot_op2_for(c3_en, alt, swz) == want {
                return Ok((alt, swz));
            }
        }
    }
    Err(AsmError::Swizzle { which: 2, want })
}

/// Assemble a group-0x18 **DOT**: `dest = dot(src1, internal)` over 3 or 4 channels.
///
/// >>> THIS IS THE FIRST ARITHMETIC A VERTEX PROGRAM DOES, and until now it could not be
/// >>> written. A retail title's world vertex program is one repeating DOT - a 4x4 transform
/// into clip position - and the decoder's note on [`decode::repeat_extra_iterations`] records
/// what reading it wrong costs: emitted once, the program produces a single scalar where a whole
/// clip position belongs and the title renders a BLACK FRAME. That reading rests on a corpus
/// census and a destination-closure argument, and on no case that states what the instruction
/// means before running it. This is what lets one be written.
///
/// # The operand shapes are the encoding's, not a simplification
///
/// * `src1` is an ordinary R6 register with a PRECISE per-channel swizzle (three selector bits
///   each), and carries both `abs` and `neg`.
/// * `src2` is ALWAYS an internal register `i0..i3` with a TABLE swizzle, and has `abs` but no
///   negate - the field does not exist. Asking for one is refused rather than dropped.
/// * `components` picks the 3- or 4-channel form, and it also picks which operand-2 swizzle
///   TABLE is in force, so a swizzle expressible at four channels may have no encoding at three.
///
/// `extra_iterations` is the repeat count: `0` is the plain single execution (which the encoding
/// writes as bit 47 set and a zero count, not as a zero field - a zero field is the one value no
/// corpus contains and the decoder blocks it). A non-zero count walks the destination one
/// CHANNEL per iteration, so the mask must name exactly one channel, and this refuses any other
/// - the same limit the decoder states, asserted from the writing side.
pub fn dot(
    components: u8,
    dest: Dest,
    mask: [bool; 4],
    src1: Src,
    src2: Src,
    extra_iterations: u32,
) -> Result<u64, AsmError> {
    let c3_en = match components {
        3 => false,
        4 => true,
        _ => return Err(AsmError::UnsupportedOp("a DOT is over three or four channels")),
    };
    let (high, low) = group_tables("grp18_dot");
    let (mut hi, mut lo) = (0u32, 0u32);

    set_field(&mut hi, high, "opcode1", 0x03)?;
    set_field(&mut hi, high, "opcode2", 0)?;
    set_field(&mut hi, high, "predicate", 0)?;
    set_field(&mut hi, high, "c3_en", u32::from(c3_en))?;

    let (opt0, op0) = dest.fields6()?;
    set_field(&mut hi, high, "opt0", opt0)?;
    set_field(&mut lo, low, "op0", op0)?;
    set_field(&mut hi, high, "alt_opt0", 0)?;

    let (alt1, opt1, n1) = src1.fields6(1)?;
    set_field(&mut hi, high, "alt_opt1", alt1)?;
    set_field(&mut lo, low, "opt1", opt1)?;
    set_field(&mut lo, low, "op1", n1)?;
    let s1 = src1.swizzle();
    for (n, name) in ["op1_swz_c0", "op1_swz_c1", "op1_swz_c2", "op1_swz_c3"].iter().enumerate() {
        set_field(&mut lo, low, name, u32::from(s1[n]))?;
    }
    let (abs1, neg1) = src1.mods();
    set_field(&mut hi, high, "abs_op1", u32::from(abs1))?;
    set_field(&mut hi, high, "neg_op1", u32::from(neg1))?;

    // src2 is an INTERNAL register and nothing else: its field is a two-bit `op2i` naming
    // i0..i3, so there is no bank selector to get wrong and no other bank to reach.
    let i2 = match src2 {
        Src::Reg { bank: Bank::Internal, index, .. } if index.is_multiple_of(4) && index / 4 < 4 => {
            u32::from(index / 4)
        }
        Src::Reg { bank, index, .. } => return Err(AsmError::RegIndex { bank, index }),
        _ => {
            return Err(AsmError::UnsupportedOp(
                "a DOT's second operand is an internal register i0..i3 and has no exotic row",
            ))
        }
    };
    set_field(&mut lo, low, "op2i", i2)?;
    let (swz_alt2, op2_swz) = find_rswz2_dot_op2(c3_en, src2.swizzle())?;
    set_field(&mut lo, low, "swz_alt_op2", swz_alt2)?;
    set_field(&mut lo, low, "op2_swz", op2_swz)?;
    let (abs2, neg2) = src2.mods();
    if neg2 {
        return Err(AsmError::UnsupportedOp("the 0x18 DOT has no negate modifier for op2"));
    }
    // >>> AND NO ABSOLUTE MODIFIER EITHER, because bit 46 is already spoken for. The field
    // tables name it `abs_op2`; [`decode::repeat_extra_iterations`] reads it as the MIDDLE BIT
    // of this group's repeat count at 46:44. A word cannot mean both, the corpus census that
    // settled the repeat reading never saw the bit set, and a modifier that silently multiplies
    // a matrix transform's iteration count by four is not a modifier worth having. Refused by
    // name - see the sweep that pins the collision.
    if abs2 {
        return Err(AsmError::UnsupportedOp(
            "the 0x18 DOT's `abs_op2` bit is the repeat count's middle bit - a word means one              or the other, and this assembler will not emit one that means both",
        ));
    }

    let (m3, m2, m1, en) = find_mask_08(mask)?;
    set_field(&mut hi, high, "swz_mask3", m3)?;
    set_field(&mut hi, high, "swz_mask2", m2)?;
    set_field(&mut hi, high, "swz_mask1", m1)?;
    set_field(&mut hi, high, "swz_en", en)?;

    // >>> THE REPEAT COUNT IS BITS 46:44, AND BIT 47 RIDES ALONGSIDE IT SET. The decoder's own
    // reading, and the field tables name those four bits `unk7 / abs_op2 / swz_en_strange1 /
    // swz_en_strange0` - so `abs_op2` above is bit 46, the middle bit of the count, and setting
    // the count after it is not an ordering accident. Writing the count through the three
    // separate names it is spelled under keeps the fields where the decoder reads them.
    if extra_iterations > 3 {
        return Err(AsmError::UnsupportedOp(
            "a DOT repeat count above three is outside the corpus census the reading rests on",
        ));
    }
    if extra_iterations > 0 && mask.iter().filter(|m| **m).count() != 1 {
        return Err(AsmError::UnsupportedOp(
            "a repeating DOT steps its destination one CHANNEL per iteration, so its mask must              name exactly one channel",
        ));
    }
    set_field(&mut hi, high, "unk7", 1)?;
    set_field(&mut hi, high, "abs_op2", (extra_iterations >> 2) & 1)?;
    set_field(&mut hi, high, "swz_en_strange1", (extra_iterations >> 1) & 1)?;
    set_field(&mut hi, high, "swz_en_strange0", extra_iterations & 1)?;

    let word = (u64::from(hi) << 32) | u64::from(lo);
    if decode::repeat_extra_iterations(word) != Some(extra_iterations) {
        return Err(AsmError::RoundTrip { what: "dot repeat count" });
    }
    // >>> THE ROUND TRIP IS SPELLED OUT HERE RATHER THAN DELEGATED TO [`verify`], for one
    // reason: op2's `bank_sel` is its INTERNAL-REGISTER NUMBER, not a bank selector. The decoder
    // fills it from the two-bit `op2i` field - i1 comes back with `bank_sel == 1` - while
    // `expected_operand` derives a selector from the bank, which for an internal register is 0.
    // Comparing the two would fail on every correct word, so this compares what the field
    // actually means: the bank, the index, the swizzle and the modifiers.
    let got = decode::decode(word);
    if got.op != (Op::Dot { components }) {
        return Err(AsmError::RoundTrip { what: "dot operation" });
    }
    if got.blocked.is_some() {
        return Err(AsmError::RoundTrip { what: "the decoder blocked the assembled dot" });
    }
    if got.pred != Predicate::Always || got.half_precision {
        return Err(AsmError::RoundTrip { what: "dot predicate or precision" });
    }
    match got.dest {
        Some(d) if d.bank == dest.bank && d.index == dest.index => {}
        _ => return Err(AsmError::RoundTrip { what: "dot destination" }),
    }
    if got.write_mask != mask {
        return Err(AsmError::RoundTrip { what: "dot write mask" });
    }
    if got.srcs.len() != 2 {
        return Err(AsmError::RoundTrip { what: "dot source count" });
    }
    if got.srcs[0] != expected_operand(&src1) {
        return Err(AsmError::RoundTrip { what: "dot op1" });
    }
    let want2 = expected_operand(&src2);
    let s2 = &got.srcs[1];
    if (s2.bank, s2.index, s2.swizzle, s2.abs, s2.neg)
        != (want2.bank, want2.index, want2.swizzle, want2.abs, want2.neg)
    {
        return Err(AsmError::RoundTrip { what: "dot op2" });
    }
    Ok(word)
}

/// >>> THE SAME INSTRUCTION, SET TO REPEAT `extra` MORE TIMES - by SEARCHING for the field
/// >>> values the decoder reads back as that count, never by restating where the field is.
///
/// # Why this is a search and not an offset
///
/// Every USSE group puts its repeat count in a different place and some have none at all:
/// [`decode::repeat_extra_iterations`] is four screens of per-group argument about exactly that,
/// including two groups where the bits a count would occupy are an operand swizzle, one where
/// they are a write mask, and one where bit 47 rides alongside the count instead of being the
/// top of it. Transcribing any of that here would be a second copy of the hardest-won reading
/// in the decoder, and the first divergence between the two would be silent.
///
/// So this asks. It tries every value the four candidate bits can hold and keeps the one the
/// decoder reports as the requested count - the same discipline [`find_rswz2_op2`] and
/// [`find_mask_08`] follow for the swizzle and mask tables.
///
/// # And the rest of the instruction must survive it
///
/// In group 0x08 those same bits ARE `src2`'s swizzle. A group with no repeat count would
/// otherwise come back from here as a word that still "reads as count 0" while meaning a
/// different operand - so the whole decoded instruction is compared, and only the count may
/// differ. A group that cannot express the request is refused by name.
pub fn with_repeat_count(word: u64, extra: u32) -> Result<u64, AsmError> {
    let before = decode::decode(word);
    if before.blocked.is_some() {
        return Err(AsmError::RoundTrip { what: "the word to repeat is itself blocked" });
    }
    let same_instruction = |got: &crate::ir::Instr| {
        got.op == before.op
            && got.pred == before.pred
            && got.dest == before.dest
            && got.write_mask == before.write_mask
            && got.srcs == before.srcs
            && got.half_precision == before.half_precision
    };
    for raw in 0..16u64 {
        let cand = (word & !(0xfu64 << 44)) | (raw << 44);
        if decode::repeat_extra_iterations(cand) != Some(extra) {
            continue;
        }
        let got = decode::decode(cand);
        if got.blocked.is_some() || !same_instruction(&got) {
            continue;
        }
        return Ok(cand);
    }
    Err(AsmError::UnsupportedOp(
        "this group has no repeat count that reaches the requested number of iterations",
    ))
}

/// Assemble an SMLSI: the per-operand repeat state (`[dest, src0, src1, src2]`) every later
/// repeating instruction steps its operands by, until the next SMLSI.
///
/// Built on the corpus's own opening word - a football title's skin program starts
/// `0xfa10000009010101` - with the four bytes and four mode bits replaced, and refused unless
/// [`decode::decode_smlsi`] reads the request back and [`decode::is_smlsi`] still recognises it.
pub fn smlsi(state: [decode::SmlsiSlot; 4]) -> Result<u64, AsmError> {
    let mut word: u64 = 0xfa10_0000_0000_0000;
    for (k, slot) in state.iter().enumerate() {
        let (mode, byte) = match *slot {
            decode::SmlsiSlot::Increment(n) => (0u64, n as u8),
            decode::SmlsiSlot::Swizzle(b) => (1u64, b),
        };
        word |= u64::from(byte) << (8 * (3 - k));
        word |= mode << (35 - k);
    }
    if !decode::is_smlsi(word) || decode::decode_smlsi(word) != state {
        return Err(AsmError::RoundTrip { what: "smlsi state" });
    }
    Ok(word)
}

/// Write `value` into bits `msb..=lsb` of `word`, refusing a value the span cannot hold.
///
/// The groups below have no named field table in the decoder - it reads them with `bits(word,
/// msb, lsb)` - so their encoders restate the positions. That is a SECOND TRANSCRIPTION, which is
/// exactly why every one of them ends in [`expect_decodes_to`]: a position restated wrong decodes
/// to a different instruction and never leaves this module.
fn put(word: &mut u64, msb: u32, lsb: u32, value: u64, name: &'static str) -> Result<(), AsmError> {
    let width = msb - lsb + 1;
    let mask = (1u64 << width) - 1;
    if value & !mask != 0 {
        return Err(AsmError::FieldTooWide { field: name, width: width as u8, value: value as u32 });
    }
    *word = (*word & !(mask << lsb)) | (value << lsb);
    Ok(())
}

/// One operand as a request states it, for [`expect_decodes_to`]: bank, register, swizzle and
/// the two modifiers. The decoder's `bank_sel` is the raw selector and not part of the meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Want {
    pub bank: Bank,
    pub index: u8,
    pub swizzle: [u8; 4],
    pub abs: bool,
    pub neg: bool,
}

impl Want {
    pub fn reg(bank: Bank, index: u8) -> Want {
        Want { bank, index, swizzle: [0, 1, 2, 3], abs: false, neg: false }
    }
    pub fn swz(mut self, s: [u8; 4]) -> Want {
        self.swizzle = s;
        self
    }
}

/// Decode `word` and require it to be exactly the request: the operation, an unconditional or
/// the stated predicate, the destination, the write mask, every source, the precision, and not
/// blocked.
#[allow(clippy::too_many_arguments)]
fn expect_decodes_to(
    word: u64,
    op: Op,
    pred: Predicate,
    dest: Option<(Bank, u8)>,
    mask: [bool; 4],
    srcs: &[Want],
    half: bool,
) -> Result<u64, AsmError> {
    let got = decode::decode(word);
    if got.blocked.is_some() {
        return Err(AsmError::RoundTrip { what: "the decoder blocked the assembled word" });
    }
    if got.op != op {
        return Err(AsmError::RoundTrip { what: "operation" });
    }
    if got.pred != pred {
        return Err(AsmError::RoundTrip { what: "predicate" });
    }
    if got.half_precision != half {
        return Err(AsmError::RoundTrip { what: "precision" });
    }
    if got.dest.map(|d| (d.bank, d.index)) != dest {
        return Err(AsmError::RoundTrip { what: "destination" });
    }
    if got.write_mask != mask {
        return Err(AsmError::RoundTrip { what: "write mask" });
    }
    if got.srcs.len() != srcs.len() {
        return Err(AsmError::RoundTrip { what: "source count" });
    }
    for (g, w) in got.srcs.iter().zip(srcs) {
        if g.bank != w.bank || g.index != w.index || g.swizzle != w.swizzle || g.abs != w.abs || g.neg != w.neg {
            return Err(AsmError::RoundTrip { what: "source operand" });
        }
    }
    Ok(word)
}

/// A DOUBLE-register seven-bit operand number, with 124..127 of the Temp row reserved for the
/// internal registers: `(selector, field)`. Used by groups 0x30 (both operands) and 0xE0.
fn r7_double(bank: Bank, index: u8, dest: bool) -> Result<(u64, u64), AsmError> {
    let sel = match bank {
        Bank::Temp | Bank::Internal => 0u64,
        Bank::Output => 1,
        Bank::PrimaryAttr => 2,
        Bank::SecondaryAttr if !dest => 3,
        other => {
            return Err(if dest { AsmError::DestBank(other) } else { AsmError::SrcBank { which: 1, bank: other } })
        }
    };
    if bank == Bank::Internal {
        if !index.is_multiple_of(4) || index / 4 > 3 {
            return Err(AsmError::RegIndex { bank, index });
        }
        return Ok((0, 124 + u64::from(index / 4)));
    }
    if !index.is_multiple_of(2) || (sel == 0 && index / 2 >= 124) {
        return Err(AsmError::RegIndex { bank, index });
    }
    Ok((sel, u64::from(index / 2)))
}

/// The same for a SIX-bit double-register field, whose reserved internal range is 60..63.
fn r6_double(bank: Bank, index: u8, dest: bool) -> Result<(u64, u64), AsmError> {
    let (sel, n) = r7_double(bank, index, dest)?;
    if bank == Bank::Internal {
        return Ok((0, n - 124 + 60));
    }
    if n >= 60 {
        return Err(AsmError::RegIndex { bank, index });
    }
    Ok((sel, n))
}

/// Assemble a group-0x30 VCOMP: `dest.mask = f(src.comp)` for `f` one of `rcp`, `rsq`, `log2`,
/// `exp2` - a SCALAR operation on one selected source component, BROADCAST to every channel the
/// four-bit mask names.
///
/// Both operand numbers are double-register (the decoder's note establishes it by def-use on a
/// real normalize), so odd registers are refused. `half` is the F16 pipeline for both the source
/// and the destination; the two type fields are independent in the encoding but no case here
/// needs them to differ.
#[allow(clippy::too_many_arguments)]
pub fn vcomp(
    op: Op,
    half: bool,
    dest: Dest,
    mask: [bool; 4],
    src_bank: Bank,
    src_reg: u8,
    comp: u8,
    abs: bool,
    neg: bool,
) -> Result<u64, AsmError> {
    let op2 = match op {
        Op::Rcp => 0u64,
        Op::Rsq => 1,
        Op::Log => 2,
        Op::Exp => 3,
        _ => return Err(AsmError::UnsupportedOp("not a group-0x30 VCOMP operation")),
    };
    if comp > 3 {
        return Err(AsmError::Swizzle { which: 1, want: [comp; 4] });
    }
    let (dsel, dn) = r7_double(dest.bank, dest.index, true)?;
    let (ssel, sn) = r7_double(src_bank, src_reg, false)?;
    let mut w = 0u64;
    put(&mut w, 63, 59, 0x06, "opcode")?;
    put(&mut w, 42, 41, op2, "op2")?;
    put(&mut w, 54, 53, u64::from(half), "dest_type")?;
    put(&mut w, 40, 39, u64::from(half), "src_type")?;
    put(&mut w, 33, 32, dsel, "dest_sel")?;
    put(&mut w, 27, 21, dn, "dest_n")?;
    put(&mut w, 31, 30, ssel, "src_sel")?;
    put(&mut w, 13, 7, sn, "src_n")?;
    put(&mut w, 38, 37, u64::from(abs) << 1 | u64::from(neg), "src_mod")?;
    put(&mut w, 36, 35, u64::from(comp), "src_comp")?;
    put(&mut w, 3, 0, (0..4).fold(0u64, |a, c| a | u64::from(mask[c]) << c), "mask")?;
    let src = Want { bank: src_bank, index: src_reg, swizzle: [comp; 4], abs, neg };
    expect_decodes_to(w, op, Predicate::Always, Some((dest.bank, dest.index)), mask, &[src], half)
}

/// Assemble a group-0xF8 LIMM: `dest = value`, one whole 32-bit lane holding a RAW pattern.
///
/// The immediate is split `[20:0]` low, `[40:36]`, `[48:44]`, and bit 54 as the TOP bit - the
/// top bit sits in the group's `opcat_extra` position, which is why a LIMM whose value is
/// negative as an integer is the interesting one. The destination number is DIRECT (seven bits,
/// undoubled), so an odd register is expressible.
pub fn limm(dest: Dest, value: u32) -> Result<u64, AsmError> {
    let sel = match dest.bank {
        Bank::Temp => 0u64,
        Bank::Output => 1,
        Bank::PrimaryAttr => 2,
        other => return Err(AsmError::DestBank(other)),
    };
    if dest.index >= 124 {
        return Err(AsmError::RegIndex { bank: dest.bank, index: dest.index });
    }
    let v = u64::from(value);
    let mut w = 0u64;
    put(&mut w, 63, 59, 0x1f, "opcode")?;
    put(&mut w, 58, 56, 0b100, "op2")?;
    put(&mut w, 53, 52, 0b10, "opcat")?;
    put(&mut w, 33, 32, sel, "dest_sel")?;
    put(&mut w, 27, 21, u64::from(dest.index), "dest_n")?;
    put(&mut w, 20, 0, v & 0x1f_ffff, "imm_low")?;
    put(&mut w, 40, 36, (v >> 21) & 0x1f, "imm_mid")?;
    put(&mut w, 48, 44, (v >> 26) & 0x1f, "imm_high")?;
    put(&mut w, 54, 54, v >> 31, "imm_top")?;
    expect_decodes_to(
        w,
        Op::Limm { value },
        Predicate::Always,
        Some((dest.bank, dest.index)),
        [true, false, false, false],
        &[],
        false,
    )
}

/// Assemble a group-0xF8 BRANCH: `if pred { pc += rel }`, `rel` counted in instruction words
/// from the branch itself (so `rel = 2` skips exactly one instruction).
///
/// A backward displacement is two's-complement in the 20-bit field with `br_type` (bit 38) set;
/// a forward one is the raw value. The condition is the group's ExtPredicate slot at [58:56],
/// SEARCHED against the decoder like every table here. A zero displacement is refused: the
/// decoder reads the unconditional one as the universal prologue no-op and the conditional one
/// as not a branch at all.
pub fn br(pred: Predicate, rel: i32) -> Result<u64, AsmError> {
    if rel == 0 || !(-(1 << 19)..(1 << 19)).contains(&rel) {
        return Err(AsmError::UnsupportedOp("a branch displacement is non-zero and fits 20 signed bits"));
    }
    let mut w = 0u64;
    put(&mut w, 63, 59, 0x1f, "opcode")?;
    put(&mut w, 38, 38, u64::from(rel < 0), "br_type")?;
    put(&mut w, 19, 0, (rel as u32 as u64) & 0xf_ffff, "br_off")?;
    for raw in 0..8u64 {
        let mut cand = w;
        put(&mut cand, 58, 56, raw, "predicate")?;
        if let Ok(word) = expect_decodes_to(cand, Op::Branch { rel }, pred, None, [false; 4], &[], false) {
            return Ok(word);
        }
    }
    Err(AsmError::UnsupportedOp("this predicate has no branch encoding"))
}

/// An integer operand: a register (its number DIRECT - the integer groups do not double) or an
/// inline literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntSrc {
    Reg(Bank, u8),
    /// The literal. Its width is the group's: 16 bits in VBW, 7 in the multiply-add groups.
    Imm(u32),
}

/// A DIRECT seven-bit register operand `(selector, field)` on the RS2 bank table, the internal
/// registers at 124..127 of the Temp row.
fn r7_direct(bank: Bank, index: u8, dest: bool) -> Result<(u64, u64), AsmError> {
    let sel = match bank {
        Bank::Temp | Bank::Internal => 0u64,
        Bank::Output => 1,
        Bank::PrimaryAttr => 2,
        Bank::SecondaryAttr if !dest => 3,
        other => {
            return Err(if dest { AsmError::DestBank(other) } else { AsmError::SrcBank { which: 1, bank: other } })
        }
    };
    if bank == Bank::Internal {
        if !index.is_multiple_of(4) || index / 4 > 3 {
            return Err(AsmError::RegIndex { bank, index });
        }
        return Ok((0, 124 + u64::from(index / 4)));
    }
    if index >= 124 || (sel != 0 && index > 127) {
        return Err(AsmError::RegIndex { bank, index });
    }
    Ok((sel, u64::from(index)))
}

/// Assemble a group-0x50 VBW: `dest = src1 OP src2` on the lane's BIT PATTERN, scalar.
///
/// `lane16` is the 16-bit form: the low half of each operand, the result masked to 16 bits. A
/// `src2` literal is 16 bits, assembled from the operand's own 7-bit field plus `[20:14]` and
/// `[37:36]`; no rotate and no invert is written, so the literal is the value. Only `src2` may
/// be the literal here - the first-source literal form is the decoder's to fold, and a shift of
/// an immediate is refused by it.
pub fn bitwise(
    kind: crate::ir::BitwiseKind,
    lane16: bool,
    dest: Dest,
    src1: (Bank, u8),
    src2: IntSrc,
) -> Result<u64, AsmError> {
    use crate::ir::BitwiseKind::*;
    let (opcode, op2) = match kind {
        And => (0x0au64, 0u64),
        Or => (0x0a, 1),
        Xor => (0x0b, 0),
        Shl => (0x0c, 0),
        Shr => (0x0d, 0),
        Asr => (0x0d, 1),
    };
    let (dsel, dn) = r7_direct(dest.bank, dest.index, true)?;
    let (s1sel, s1n) = r7_direct(src1.0, src1.1, false)?;
    let mut w = 0u64;
    put(&mut w, 63, 59, opcode, "opcode")?;
    put(&mut w, 35, 35, op2, "op2")?;
    put(&mut w, 34, 34, u64::from(lane16), "width16")?;
    put(&mut w, 33, 32, dsel, "dest_sel")?;
    put(&mut w, 27, 21, dn, "dest_n")?;
    put(&mut w, 31, 30, s1sel, "src1_sel")?;
    put(&mut w, 13, 7, s1n, "src1_n")?;
    let (imm, srcs) = match src2 {
        IntSrc::Reg(b, i) => {
            let (s2sel, s2n) = r7_direct(b, i, false)?;
            put(&mut w, 29, 28, s2sel, "src2_sel")?;
            put(&mut w, 6, 0, s2n, "src2_n")?;
            (None, vec![Want::reg(src1.0, src1.1), Want::reg(b, i)])
        }
        IntSrc::Imm(v) => {
            if v > 0xffff {
                return Err(AsmError::FieldTooWide { field: "vbw immediate", width: 16, value: v });
            }
            let v64 = u64::from(v);
            put(&mut w, 48, 48, 1, "src2_ext")?;
            put(&mut w, 29, 28, 2, "src2_sel")?;
            put(&mut w, 6, 0, v64 & 0x7f, "imm_low")?;
            put(&mut w, 20, 14, (v64 >> 7) & 0x7f, "imm_mid")?;
            put(&mut w, 37, 36, v64 >> 14, "imm_high")?;
            (Some(v & if lane16 { 0xffff } else { u32::MAX }), vec![Want::reg(src1.0, src1.1)])
        }
    };
    let op = Op::Bitwise { kind, imm, lane_bits: if lane16 { 16 } else { 32 } };
    expect_decodes_to(w, op, Predicate::Always, Some((dest.bank, dest.index)), [true, false, false, false], &srcs, false)
}

/// Assemble a group-0x78 VTSTMSK in the FLOAT families: `dest.c = (alu(src1, src2).c cmp 0) ?
/// 1.0 : 0.0` for all four channels, the NUMERIC mask form - the one the decoder establishes for
/// a float family. `neg1` is `test_flag_2`, the source-1 negate.
///
/// The destination is DIRECT (the ordinary seven-bit field) while the float sources are DOUBLED,
/// exactly as the sibling VTST reads them.
#[allow(clippy::too_many_arguments)]
pub fn vtstmsk(
    alu: TestAlu,
    cmp: TestCmp,
    half: bool,
    dest: Dest,
    src1: (Bank, u8),
    neg1: bool,
    src2: (Bank, u8),
) -> Result<u64, AsmError> {
    let alu_op = match alu {
        TestAlu::Add => 2u64,
        TestAlu::Mul => 13,
        TestAlu::Sub => 14,
        _ => return Err(AsmError::UnsupportedOp("vtstmsk: only the FLOAT families are assembled")),
    };
    let (sign_test, zero_test) = match cmp {
        TestCmp::Eq => (0u64, 1u64),
        TestCmp::Ne => (0, 2),
        TestCmp::Lt => (1, 0),
        TestCmp::Gt => (2, 0),
        TestCmp::Le => (1, 1),
        TestCmp::Ge => (2, 1),
    };
    let (dsel, dn) = r7_direct(dest.bank, dest.index, true)?;
    let (s1sel, s1n) = r7_double(src1.0, src1.1, false)?;
    let (s2sel, s2n) = r7_double(src2.0, src2.1, false)?;
    let mut w = 0u64;
    put(&mut w, 63, 59, 0x0f, "opcode")?;
    put(&mut w, 50, 50, u64::from(neg1), "src1_neg")?;
    put(&mut w, 47, 47, u64::from(!half), "prec")?;
    put(&mut w, 43, 42, sign_test, "sign_test")?;
    put(&mut w, 41, 40, zero_test, "zero_test")?;
    put(&mut w, 37, 36, 2, "mask_type")?;
    put(&mut w, 33, 32, dsel, "dest_sel")?;
    put(&mut w, 31, 30, s1sel, "src1_sel")?;
    put(&mut w, 29, 28, s2sel, "src2_sel")?;
    put(&mut w, 27, 21, dn, "dest_n")?;
    put(&mut w, 20, 20, 1, "test_wben")?;
    put(&mut w, 17, 14, alu_op, "alu_op")?;
    put(&mut w, 13, 7, s1n, "src1_n")?;
    put(&mut w, 6, 0, s2n, "src2_n")?;
    let mut s1 = Want::reg(src1.0, src1.1);
    s1.neg = neg1;
    expect_decodes_to(
        w,
        Op::TestMask { alu, cmp },
        Predicate::Always,
        Some((dest.bank, dest.index)),
        [true; 4],
        &[s1, Want::reg(src2.0, src2.1)],
        half,
    )
}

/// The `(ext, sel, field)` of an IMAD32/IMAD-step `src1`/`src2`: a register on the RS2 table, or
/// the inline literal (extension row, selector 2), whose 7-bit number is the value.
fn imad_operand(src: IntSrc) -> Result<(u64, u64, u64, Want), AsmError> {
    match src {
        IntSrc::Reg(b, i) => {
            let (sel, n) = r7_direct(b, i, false)?;
            Ok((0, sel, n, Want::reg(b, i)))
        }
        IntSrc::Imm(v) => {
            if v > 0x7f {
                return Err(AsmError::FieldTooWide { field: "imad immediate", width: 7, value: v });
            }
            Ok((1, 2, u64::from(v), Want::reg(Bank::Immediate, v as u8)))
        }
    }
}

/// Assemble a group-0x15 IMAD32: `dest = half(src0) * src1' + src2` on 32-bit lanes, where
/// `half(src0)` is the low or high 16 bits (`src0_high`) sign- or zero-extended (`signed`), and
/// `src1'` is the whole register or, with `src1_high`, its high half. Every number is DIRECT.
/// `src0` is Temp or PrimaryAttr (a one-bit selector).
#[allow(clippy::too_many_arguments)]
pub fn imad(
    signed: bool,
    src0_high: bool,
    src1_high: bool,
    dest: Dest,
    src0: (Bank, u8),
    src1: IntSrc,
    src2: IntSrc,
) -> Result<u64, AsmError> {
    let (dsel, dn) = r7_direct(dest.bank, dest.index, true)?;
    let s0sel = match src0.0 {
        Bank::Temp => 0u64,
        Bank::PrimaryAttr => 1,
        other => return Err(AsmError::SrcBank { which: 0, bank: other }),
    };
    if src0.1 >= 124 {
        return Err(AsmError::RegIndex { bank: src0.0, index: src0.1 });
    }
    let (e1, sel1, n1, w1) = imad_operand(src1)?;
    let (e2, sel2, n2, w2) = imad_operand(src2)?;
    let mut w = 0u64;
    put(&mut w, 63, 59, 0x15, "opcode")?;
    put(&mut w, 56, 56, u64::from(src0_high), "src0_high")?;
    put(&mut w, 53, 53, u64::from(src1_high), "src1_high")?;
    put(&mut w, 49, 49, e1, "src1_ext")?;
    put(&mut w, 48, 48, e2, "src2_ext")?;
    put(&mut w, 43, 43, u64::from(signed), "signed")?;
    put(&mut w, 39, 38, 2, "src2_type")?;
    put(&mut w, 34, 34, s0sel, "src0_bank")?;
    put(&mut w, 33, 32, dsel, "dest_bank")?;
    put(&mut w, 31, 30, sel1, "src1_bank")?;
    put(&mut w, 29, 28, sel2, "src2_bank")?;
    put(&mut w, 27, 21, dn, "dest_n")?;
    put(&mut w, 20, 14, u64::from(src0.1), "src0_n")?;
    put(&mut w, 13, 7, n1, "src1_n")?;
    put(&mut w, 6, 0, n2, "src2_n")?;
    expect_decodes_to(
        w,
        Op::IntMad { signed, bits: 32, src0_high, src1_high },
        Predicate::Always,
        Some((dest.bank, dest.index)),
        [true, false, false, false],
        &[Want::reg(src0.0, src0.1), w1, w2],
        false,
    )
}

/// Assemble ONE STEP of a group-0x1a 32-bit multiply-add: `high` selects `((src0 >> 16) * src1)
/// << 16 + src2` over `(src0 & 0xffff) * src1 + src2`.
///
/// A step is only ever emitted as HALF OF A PAIR - the low step then the high one, same `src0`
/// and `src1`, the second's `src2` the first's destination - and the decoder's pairing check
/// blocks anything else. So a caller writes both; this assembles one word. `src0` may be any of
/// the four banks (this group has the extension bit group 0x15 lacks).
pub fn imad_step(
    high: bool,
    signed: bool,
    dest: Dest,
    src0: (Bank, u8),
    src1: IntSrc,
    src2: IntSrc,
) -> Result<u64, AsmError> {
    let (dsel, dn) = r7_direct(dest.bank, dest.index, true)?;
    let (s0ext, s0sel) = match src0.0 {
        Bank::Temp => (0u64, 0u64),
        Bank::PrimaryAttr => (0, 1),
        Bank::Output => (1, 0),
        Bank::SecondaryAttr => (1, 1),
        other => return Err(AsmError::SrcBank { which: 0, bank: other }),
    };
    if src0.1 >= 124 {
        return Err(AsmError::RegIndex { bank: src0.0, index: src0.1 });
    }
    let (e1, sel1, n1, w1) = imad_operand(src1)?;
    let (e2, sel2, n2, w2) = imad_operand(src2)?;
    let mut w = 0u64;
    put(&mut w, 63, 59, 0x1a, "opcode")?;
    put(&mut w, 52, 52, u64::from(high), "sn")?;
    put(&mut w, 49, 49, e1, "src1_ext")?;
    put(&mut w, 48, 48, e2, "src2_ext")?;
    put(&mut w, 47, 47, s0ext, "src0_ext")?;
    put(&mut w, 41, 41, u64::from(signed), "signed")?;
    put(&mut w, 34, 34, s0sel, "src0_bank")?;
    put(&mut w, 33, 32, dsel, "dest_bank")?;
    put(&mut w, 31, 30, sel1, "src1_bank")?;
    put(&mut w, 29, 28, sel2, "src2_bank")?;
    put(&mut w, 27, 21, dn, "dest_n")?;
    put(&mut w, 20, 14, u64::from(src0.1), "src0_n")?;
    put(&mut w, 13, 7, n1, "src1_n")?;
    put(&mut w, 6, 0, n2, "src2_n")?;
    expect_decodes_to(
        w,
        Op::IntMadStep { signed, high_half: high },
        Predicate::Always,
        Some((dest.bank, dest.index)),
        [true, false, false, false],
        &[Want::reg(src0.0, src0.1), w1, w2],
        false,
    )
}

/// Assemble the ORDINARY-REGISTER form of group 0x14: `dest = low16(src) * stride + addend`,
/// where `stride` is not in the word at all - the decoder reads it off the PROGRAM as one more
/// than the largest addend any such load carries (rows per block; see
/// `resolve_index_load_stride`). So a case states the stride it expects from the addends it
/// wrote, which is the claim.
///
/// Built from the shipped template like [`load_index`], with the `(b8, b51)` field at `(1, 0)`
/// and the destination in `[25:21]` under bank bit 33 (Temp / PrimaryAttr).
pub fn index_add(dest: Dest, src_bank: Bank, src_reg: u8, addend: u8) -> Result<u64, AsmError> {
    let (template, variable) = decode::i16mad_load_index_template();
    let bank_bit = |b: Bank, which: u8| match b {
        Bank::Temp => Ok(0u64),
        Bank::PrimaryAttr => Ok(1),
        other => Err(AsmError::SrcBank { which, bank: other }),
    };
    let (sb, db) = (bank_bit(src_bank, 1)?, bank_bit(dest.bank, 0)?);
    if src_reg > 0x3f {
        return Err(AsmError::RegIndex { bank: src_bank, index: src_reg });
    }
    if dest.index > 0x1f {
        return Err(AsmError::RegIndex { bank: dest.bank, index: dest.index });
    }
    if addend > 0x7f {
        return Err(AsmError::FieldTooWide { field: "addend", width: 7, value: u32::from(addend) });
    }
    let mut word = template & !variable;
    word |= u64::from(src_reg) << 14;
    word |= u64::from(addend);
    word |= sb << 34;
    word |= db << 33;
    word |= u64::from(dest.index) << 21;
    word |= 1 << 8;
    let got = decode::decode(word);
    if got.blocked.is_some() {
        return Err(AsmError::RoundTrip { what: "the decoder did not recognise the index-add encoding" });
    }
    match got.op {
        Op::LoadIndex { addend: a, to_index: false, .. } if a == i32::from(addend) => {}
        _ => return Err(AsmError::RoundTrip { what: "index-add operation or addend" }),
    }
    match (got.dest, got.srcs.first()) {
        (Some(d), Some(s))
            if d.bank == dest.bank && d.index == dest.index && s.bank == src_bank && s.index == src_reg => {}
        _ => return Err(AsmError::RoundTrip { what: "index-add operands" }),
    }
    Ok(word)
}

/// Assemble a group-0x90 SOP2M, the 8-bit sum-of-products combiner: per channel,
/// `op(coeff1 * src1, coeff2 * src2)` with the COLOUR op on channels 0..2 and the ALPHA op on
/// channel 3, every value a unorm byte. Operand numbers are DIRECT (an 8-bit type is never
/// doubled). The write mask is stored ROTATED - alpha in bit 0 - and is searched rather than
/// rotated here, so the one easy-to-invert field is the decoder's to state.
#[allow(clippy::too_many_arguments)]
pub fn sop2(
    color: crate::ir::SopOp,
    alpha: crate::ir::SopOp,
    f1: crate::ir::SopFactor,
    f1_complement: bool,
    f2: crate::ir::SopFactor,
    f2_complement: bool,
    dest: Dest,
    mask: [bool; 4],
    src1: (Bank, u8),
    src2: (Bank, u8),
) -> Result<u64, AsmError> {
    use crate::ir::{SopFactor, SopOp};
    let op = |o: SopOp| match o {
        SopOp::Add => 0u64,
        SopOp::Sub => 1,
        SopOp::Min => 2,
        SopOp::Max => 3,
    };
    let sel = |f: SopFactor| match f {
        SopFactor::Zero => 0u64,
        SopFactor::Src1Color => 2,
        SopFactor::Src1Alpha => 3,
        SopFactor::Src2Color => 6,
        SopFactor::Src2Alpha => 7,
    };
    let (dsel, dn) = r7_direct(dest.bank, dest.index, true)?;
    let (s1sel, s1n) = r7_direct(src1.0, src1.1, false)?;
    let (s2sel, s2n) = r7_direct(src2.0, src2.1, false)?;
    let mut w = 0u64;
    put(&mut w, 63, 59, 0x12, "opcode")?;
    put(&mut w, 56, 56, u64::from(f1_complement), "mod1")?;
    put(&mut w, 53, 52, op(color), "cop")?;
    put(&mut w, 47, 47, u64::from(f2_complement), "mod2")?;
    put(&mut w, 42, 41, op(alpha), "aop")?;
    put(&mut w, 40, 38, sel(f1), "sel1")?;
    put(&mut w, 37, 35, sel(f2), "sel2")?;
    put(&mut w, 33, 32, dsel, "dest_bank")?;
    put(&mut w, 31, 30, s1sel, "src1_bank")?;
    put(&mut w, 29, 28, s2sel, "src2_bank")?;
    put(&mut w, 27, 21, dn, "dest_n")?;
    put(&mut w, 13, 7, s1n, "src1_n")?;
    put(&mut w, 6, 0, s2n, "src2_n")?;
    let want_op = Op::Sop2 { color, alpha, f1, f1_complement, f2, f2_complement };
    let srcs = [Want::reg(src1.0, src1.1), Want::reg(src2.0, src2.1)];
    for raw in 0..16u64 {
        let mut cand = w;
        put(&mut cand, 46, 43, raw, "wmask")?;
        if let Ok(word) =
            expect_decodes_to(cand, want_op, Predicate::Always, Some((dest.bank, dest.index)), mask, &srcs, false)
        {
            return Ok(word);
        }
    }
    Err(AsmError::WriteMask(mask))
}

/// Assemble a group-0x80 whole-register BYTE COPY, `dest = src` in the four-unorm-byte view -
/// the SOP2 form a fragment epilogue ends with.
///
/// The decoder reads this group through ONE established shape and pins every coefficient and
/// op field to its observed value, because what those fields select is not established. So this
/// builds exactly that shape (`sel2` 0, `aop` 0) and varies only the operands the def-use chain
/// that established it reads. `src2` names the OUTPUT bank, the slot the chain says is not the
/// source.
pub fn copy_fx8(dest: Dest, src: (Bank, u8)) -> Result<u64, AsmError> {
    let (dsel, dn) = r7_direct(dest.bank, dest.index, true)?;
    let (s1sel, s1n) = r7_direct(src.0, src.1, false)?;
    let mut w = 0u64;
    put(&mut w, 63, 59, 0x10, "opcode")?;
    put(&mut w, 53, 52, 1, "cop")?;
    put(&mut w, 47, 47, 1, "mod2")?;
    put(&mut w, 40, 38, 3, "sel1")?;
    put(&mut w, 33, 32, dsel, "dest_bank")?;
    put(&mut w, 31, 30, s1sel, "src1_bank")?;
    put(&mut w, 29, 28, 1, "src2_bank")?;
    put(&mut w, 27, 21, dn, "dest_n")?;
    put(&mut w, 13, 7, s1n, "src1_n")?;
    expect_decodes_to(
        w,
        Op::CopyFx8,
        Predicate::Always,
        Some((dest.bank, dest.index)),
        [true; 4],
        &[Want::reg(src.0, src.1)],
        false,
    )
}

/// Assemble a group-0xE8 MEMORY LOAD: `elements` consecutive 32-bit guest words from `src0 +
/// offset1 + offset2` into `dest..dest+elements`.
///
/// `src0` is the byte POINTER, on the four-bank src0 table (a selector bit plus an extension
/// bit). An offset is an inline literal counting ELEMENTS (scaled by four here) or a register
/// holding BYTES - different units, which is the spec's own address sum. Every number is DIRECT.
/// Only the census variant is expressible: unpredicated, `mode`/`addr_mode` 0, 32-bit elements.
pub fn ldmem(
    dest: Dest,
    elements: u8,
    src0: (Bank, u8),
    offset1: IntSrc,
    offset2: IntSrc,
) -> Result<u64, AsmError> {
    if !(1..=16).contains(&elements) {
        return Err(AsmError::FieldTooWide { field: "elements", width: 4, value: u32::from(elements) });
    }
    let dest_bit = match dest.bank {
        Bank::Temp => 0u64,
        Bank::PrimaryAttr => 1,
        other => return Err(AsmError::DestBank(other)),
    };
    if dest.index > 0x7f || (dest.bank == Bank::Temp && u32::from(dest.index) + u32::from(elements) > 124) {
        return Err(AsmError::RegIndex { bank: dest.bank, index: dest.index });
    }
    let (s0ext, s0sel) = match src0.0 {
        Bank::Temp => (0u64, 0u64),
        Bank::PrimaryAttr => (0, 1),
        Bank::Output => (1, 0),
        Bank::SecondaryAttr => (1, 1),
        other => return Err(AsmError::SrcBank { which: 0, bank: other }),
    };
    if src0.1 > 0x7f {
        return Err(AsmError::RegIndex { bank: src0.0, index: src0.1 });
    }
    let mut w = 0u64;
    put(&mut w, 63, 59, 0x1d, "opcode")?;
    put(&mut w, 47, 44, u64::from(elements - 1), "mask_count")?;
    put(&mut w, 39, 39, dest_bit, "dest_bank")?;
    put(&mut w, 27, 21, u64::from(dest.index), "dest_n")?;
    put(&mut w, 50, 50, s0ext, "src0_ext")?;
    put(&mut w, 34, 34, s0sel, "src0_bank")?;
    put(&mut w, 20, 14, u64::from(src0.1), "src0_n")?;
    let mut imm = 0u32;
    let mut regs = vec![Want::reg(src0.0, src0.1)];
    for (src, (ext_bit, sel_hi, n_hi)) in [(offset1, (49u32, 31u32, 13u32)), (offset2, (48, 29, 6))] {
        match src {
            IntSrc::Imm(v) => {
                if v > 0x7f {
                    return Err(AsmError::FieldTooWide { field: "ldmem offset", width: 7, value: v });
                }
                imm += v;
                put(&mut w, ext_bit, ext_bit, 1, "offset_ext")?;
                put(&mut w, sel_hi, sel_hi - 1, 2, "offset_sel")?;
                put(&mut w, n_hi, n_hi - 6, u64::from(v), "offset_n")?;
            }
            IntSrc::Reg(b, i) => {
                let (sel, n) = r7_direct(b, i, false)?;
                put(&mut w, sel_hi, sel_hi - 1, sel, "offset_sel")?;
                put(&mut w, n_hi, n_hi - 6, n, "offset_n")?;
                regs.push(Want::reg(b, i));
            }
        }
    }
    let got = decode::decode(w);
    if got.blocked.is_some() {
        return Err(AsmError::RoundTrip { what: "the decoder blocked the assembled memory load" });
    }
    match got.op {
        Op::MemLoad { elements: e, offset_bytes } if e == elements && offset_bytes == imm * 4 => {}
        _ => return Err(AsmError::RoundTrip { what: "memory-load element count or offset" }),
    }
    if got.dest.map(|d| (d.bank, d.index)) != Some((dest.bank, dest.index)) {
        return Err(AsmError::RoundTrip { what: "memory-load destination" });
    }
    if got.srcs.len() != regs.len()
        || got.srcs.iter().zip(&regs).any(|(g, w)| g.bank != w.bank || g.index != w.index)
    {
        return Err(AsmError::RoundTrip { what: "memory-load pointer or offset registers" });
    }
    Ok(w)
}

/// The group-0x38 move forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveKind {
    /// `dest = src1`, swizzled.
    Mov,
    /// `dest.c = test(src0.c) ? src1.c : src2.c`, per FLOAT channel.
    Cmov(crate::ir::CompareMethod),
    /// The same select per BYTE of one register, every operand read undoubled.
    CmovU8(crate::ir::CompareMethod),
}

/// Assemble a group-0x38 VMOV / VMOVC / VMOVCU8.
///
/// `src1` carries the swizzle (searched in the decoder's vec4-standard table), and a conditional
/// form applies it to `src2` too - the encoding has one swizzle field, so a request for different
/// swizzles on the two is not expressible. `src0` (the TEST value) is read unswizzled, its bank a
/// single bit (Temp or PrimaryAttr).
///
/// The FLOAT forms' operands are double-register (R6); the BYTE form's are the register number
/// itself - the decoder's note gives the three closures that establish it. `half` selects F16
/// over F32 for the float forms and is ignored by the byte form, which reads no float view.
#[allow(clippy::too_many_arguments)]
pub fn vmov(
    kind: MoveKind,
    half: bool,
    dest: Dest,
    mask: [bool; 4],
    src1: (Bank, u8),
    swizzle: [u8; 4],
    src2: Option<(Bank, u8)>,
    src0: Option<(Bank, u8)>,
) -> Result<u64, AsmError> {
    use crate::ir::CompareMethod;
    let byte = matches!(kind, MoveKind::CmovU8(_));
    let (move_type, test) = match kind {
        MoveKind::Mov => (0u64, None),
        MoveKind::Cmov(t) => (1, Some(t)),
        MoveKind::CmovU8(t) => (2, Some(t)),
    };
    if test.is_some() != (src2.is_some() && src0.is_some()) || (test.is_none() && (src2.is_some() || src0.is_some())) {
        return Err(AsmError::UnsupportedOp("a conditional move takes src2 and src0, and a plain move neither"));
    }
    // A byte operand is its register number; a float one is doubled.
    let operand = |bank: Bank, index: u8, is_dest: bool| -> Result<(u64, u64), AsmError> {
        if byte {
            let sel = match bank {
                Bank::Temp => 0u64,
                Bank::Output => 1,
                Bank::PrimaryAttr => 2,
                Bank::SecondaryAttr if !is_dest => 3,
                other => {
                    return Err(if is_dest { AsmError::DestBank(other) } else { AsmError::SrcBank { which: 1, bank: other } })
                }
            };
            if index > 0x3f {
                return Err(AsmError::RegIndex { bank, index });
            }
            Ok((sel, u64::from(index)))
        } else {
            r6_double(bank, index, is_dest)
        }
    };
    let mut w = 0u64;
    put(&mut w, 63, 59, 0x07, "opcode")?;
    put(&mut w, 47, 46, move_type, "move_type")?;
    put(&mut w, 42, 40, if byte { 6 } else if half { 4 } else { 5 }, "data_type")?;
    let (dsel, dn) = operand(dest.bank, dest.index, true)?;
    put(&mut w, 33, 32, dsel, "dest_sel")?;
    put(&mut w, 23, 18, dn, "dest_n")?;
    let (s1sel, s1n) = operand(src1.0, src1.1, false)?;
    put(&mut w, 31, 30, s1sel, "src1_sel")?;
    put(&mut w, 11, 6, s1n, "src1_n")?;
    if let (Some(t), Some(s2), Some(s0)) = (test, src2, src0) {
        let t = match t {
            CompareMethod::EqZero => 0u64,
            CompareMethod::NeZero => 1,
            CompareMethod::LtZero => 2,
            CompareMethod::LteZero => 3,
        };
        put(&mut w, 54, 54, t >> 1, "test_hi")?;
        put(&mut w, 39, 39, t & 1, "test_lo")?;
        let (s2sel, s2n) = operand(s2.0, s2.1, false)?;
        put(&mut w, 29, 28, s2sel, "src2_sel")?;
        put(&mut w, 5, 0, s2n, "src2_n")?;
        let s0sel = match s0.0 {
            Bank::Temp => 0u64,
            Bank::PrimaryAttr => 1,
            other => return Err(AsmError::SrcBank { which: 0, bank: other }),
        };
        put(&mut w, 34, 34, s0sel, "src0_sel")?;
        let s0n = if byte {
            u64::from(s0.1)
        } else {
            if !s0.1.is_multiple_of(2) || s0.1 / 2 >= 60 {
                return Err(AsmError::RegIndex { bank: s0.0, index: s0.1 });
            }
            u64::from(s0.1 / 2)
        };
        put(&mut w, 17, 12, s0n, "src0_n")?;
    }
    // The swizzle and the mask are table lookups, so both are SEARCHED against the decoder.
    let op = match kind {
        MoveKind::Mov => Op::Mov,
        MoveKind::Cmov(t) => Op::Cmov { test: t },
        MoveKind::CmovU8(t) => Op::CmovU8 { test: t },
    };
    let sw = if byte { [0, 1, 2, 3] } else { swizzle };
    let mut srcs = vec![Want::reg(src1.0, src1.1).swz(sw)];
    if let (Some(s2), Some(s0)) = (src2, src0) {
        srcs.push(Want::reg(s2.0, s2.1).swz(sw));
        srcs.push(Want::reg(s0.0, s0.1));
    }
    for sraw in 0..16u64 {
        if byte && sraw != 0 {
            break;
        }
        for mraw in 0..16u64 {
            let mut cand = w;
            put(&mut cand, 38, 35, sraw, "src0_swiz")?;
            put(&mut cand, 27, 24, mraw, "mask")?;
            if let Ok(word) =
                expect_decodes_to(cand, op, Predicate::Always, Some((dest.bank, dest.index)), mask, &srcs, half && !byte)
            {
                return Ok(word);
            }
        }
    }
    Err(AsmError::RoundTrip { what: "no swizzle and mask field pair decodes to this move" })
}

/// The decoder's field tables for a group, by the name it registers them under.
///
/// Panics for an unknown name: the names are compile-time constants in this module, so a miss
/// is a typo in code, not a runtime condition to handle.
fn group_tables(name: &str) -> (&'static [(&'static str, u8)], &'static [(&'static str, u8)]) {
    let (_, high, low) = decode::GROUP_TABLES
        .iter()
        .find(|(n, _, _)| *n == name)
        .unwrap_or_else(|| panic!("no decoder field table named `{name}`"));
    (high, low)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`set_field`] must be the exact inverse of [`decode::field`] for EVERY field of EVERY
    /// group table, at every value the field can hold. This is the property the whole module
    /// rests on, so it is checked exhaustively rather than sampled.
    #[test]
    fn set_field_is_the_exact_inverse_of_field_on_every_group_table() {
        for (group, high, low) in decode::GROUP_TABLES {
            for table in [high, low] {
                for &(name, width) in table.iter() {
                    let limit = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
                    for value in [0, 1, limit / 2, limit] {
                        let mut word = 0u32;
                        set_field(&mut word, table, name, value)
                            .unwrap_or_else(|e| panic!("{group}.{name}: {e}"));
                        assert_eq!(
                            decode::field(word, table, name),
                            value,
                            "{group}.{name} did not read back"
                        );
                        // And it must disturb NOTHING else: a field written into a word that is
                        // otherwise all ones must leave every other field at its maximum.
                        let mut busy = u32::MAX;
                        set_field(&mut busy, table, name, value).unwrap();
                        for &(other, owidth) in table.iter() {
                            if other == name {
                                continue;
                            }
                            let omax = if owidth >= 32 { u32::MAX } else { (1u32 << owidth) - 1 };
                            assert_eq!(
                                decode::field(busy, table, other),
                                omax,
                                "{group}.{name} overwrote {other}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A value too wide for its field is REFUSED, not truncated. A truncated register number
    /// is a program reading a register nobody wrote, and it would look like a decode defect
    /// for as long as it took to find.
    #[test]
    fn a_value_too_wide_for_its_field_is_refused() {
        let (high, _) = group_tables("grp08_alu");
        assert_eq!(
            set_field(&mut 0, high, "opcode1", 32),
            Err(AsmError::FieldTooWide { field: "opcode1", width: 5, value: 32 })
        );
        assert_eq!(set_field(&mut 0, high, "not_a_field", 0), Err(AsmError::NoSuchField("not_a_field")));
    }

    // =========================================================================================
    // >>> ENUMERATE THE ENCODABLE SPACE INSTEAD OF COLLECTING IT.
    //
    // The corpus is an accident of what seven titles' compilers emitted. A write mask, a swizzle
    // or a modifier that none of them happened to produce is a hole nothing in this project
    // touches - and it is emitted the first time an eighth title ships. The ALU groups are the
    // ones every program is mostly made of, so they are where the holes cost most.
    //
    // Each sweep below asserts TWO things, and the second is the one that finds defects:
    //
    //   1. every value the encoding accepts DECODES BACK to what was asked, and
    //   2. WHICH values it accepts does not depend on a field that has nothing to do with it.
    //
    // A refusal that tracks the precision, or the operation, or which source it is applied to,
    // reads as an encoding fact and is almost always two transcriptions disagreeing. The pack
    // and test groups' sweeps already assert exactly this, and each found something.
    // =========================================================================================

    /// The sixteen write masks, against every ALU operation and both precisions.
    ///
    /// The MASK is table-encoded (`find_mask_08` searches the decoder's own table), so which of
    /// the sixteen are expressible is a property of that table and nothing else - certainly not
    /// of the arithmetic the instruction does.
    #[test]
    fn every_write_mask_the_alu_group_accepts_is_the_same_set_for_every_operation() {
        let ops = [Op::Mul, Op::Add, Op::Frc, Op::Dsx, Op::Dsy, Op::Min, Op::Max];
        let mut reference: Option<(Op, bool, Vec<[bool; 4]>)> = None;
        let mut checked = 0usize;
        for op in ops {
            for half in [false, true] {
                let mut accepted = Vec::new();
                for bits in 0u8..16 {
                    let mask = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0];
                    let Ok(word) = alu(
                        op,
                        half,
                        Dest::new(Bank::Temp, 4),
                        mask,
                        Src::reg(Bank::PrimaryAttr, 2),
                        Src::reg(Bank::SecondaryAttr, 8),
                    ) else {
                        continue;
                    };
                    let got = decode::decode(word);
                    assert_eq!(got.op, op, "{op:?} half={half} mask={mask:?}: {word:#018x}");
                    assert_eq!(got.half_precision, half, "{op:?} mask={mask:?}");
                    assert_eq!(got.write_mask, mask, "{op:?} half={half}: mask did not read back");
                    accepted.push(mask);
                    checked += 1;
                }
                match &reference {
                    None => reference = Some((op, half, accepted)),
                    Some((rop, rhalf, want)) => assert_eq!(
                        &accepted, want,
                        "the accepted write masks differ between {rop:?} half={rhalf} and \
                         {op:?} half={half} - a mask's existence must not depend on either"
                    ),
                }
            }
        }
        let (_, _, accepted) = reference.expect("at least one arm ran");
        println!("  ALU write masks: {} of 16 encodable", accepted.len());
        assert!(
            accepted.len() >= 8,
            "only {} of 16 write masks are encodable in the ALU group: {accepted:?}",
            accepted.len()
        );
        assert!(checked >= 100, "only {checked} masked ALU words round-tripped - the sweep is empty");
    }

    /// Every three-bit swizzle quadruple `src1` can carry - all 4,096 of them, constants
    /// included - and every one `src2`'s TABLE can express.
    ///
    /// The two sources are encoded completely differently and that is the whole point of
    /// sweeping both: `src1` carries a PRECISE per-channel selector split across four fields in
    /// two halves of the word, while `src2` is a table lookup. A selector field that was
    /// assembled into the wrong half would decode as a different channel, which no
    /// corpus-derived test can see unless some title happened to emit that exact swizzle.
    #[test]
    fn every_swizzle_the_alu_sources_can_express_round_trips() {
        let mut precise = 0usize;
        for a in 0..8u8 {
            for b in 0..8u8 {
                for c in 0..8u8 {
                    for d in 0..8u8 {
                        let want = [a, b, c, d];
                        let Ok(word) = alu(
                            Op::Add,
                            false,
                            Dest::new(Bank::Temp, 4),
                            [true; 4],
                            Src::reg(Bank::PrimaryAttr, 2).swz(want),
                            Src::reg(Bank::SecondaryAttr, 8),
                        ) else {
                            continue;
                        };
                        let got = decode::decode(word);
                        assert_eq!(
                            got.srcs[0].swizzle, want,
                            "src1 swizzle {want:?} came back as {:?}: {word:#018x}",
                            got.srcs[0].swizzle
                        );
                        // And it must not have disturbed the OTHER source, which shares the word.
                        assert_eq!(got.srcs[1].index, 8, "src1's swizzle moved src2: {word:#018x}");
                        precise += 1;
                    }
                }
            }
        }
        assert_eq!(precise, 4096, "src1's swizzle is a free 3-bit selector per channel");

        // `src2`'s table expresses a SUBSET, and which subset must be the same whichever
        // operation asks for it.
        let table_for = |op: Op| -> Vec<[u8; 4]> {
            let mut v = Vec::new();
            for a in 0..8u8 {
                for b in 0..8u8 {
                    for c in 0..8u8 {
                        for d in 0..8u8 {
                            let want = [a, b, c, d];
                            if alu(
                                op,
                                false,
                                Dest::new(Bank::Temp, 4),
                                [true; 4],
                                Src::reg(Bank::PrimaryAttr, 2),
                                Src::reg(Bank::SecondaryAttr, 8).swz(want),
                            )
                            .is_ok()
                            {
                                v.push(want);
                            }
                        }
                    }
                }
            }
            v
        };
        let base = table_for(Op::Add);
        assert!(!base.is_empty(), "src2's swizzle table expresses nothing at all");
        // The size of each space, printed rather than only asserted: "the table expresses some"
        // is not a measurement, and the next person to widen either encoding needs the before.
        println!(
            "  ALU swizzles: src1 {precise} of 4096 (a free 3-bit selector per channel), src2 {} of 4096 (a table lookup)",
            base.len()
        );
        for op in [Op::Mul, Op::Min, Op::Max, Op::Frc] {
            assert_eq!(
                table_for(op),
                base,
                "src2's expressible swizzles must not depend on the OPERATION ({op:?})"
            );
        }
        // Each one that IS expressible must come back exactly.
        for want in &base {
            let word = alu(
                Op::Add,
                false,
                Dest::new(Bank::Temp, 4),
                [true; 4],
                Src::reg(Bank::PrimaryAttr, 2),
                Src::reg(Bank::SecondaryAttr, 8).swz(*want),
            )
            .unwrap();
            assert_eq!(decode::decode(word).srcs[1].swizzle, *want, "{want:?}: {word:#018x}");
        }
    }

    /// The source MODIFIERS, on every operation and both precisions.
    ///
    /// `abs` and `neg` are single bits, and which source carries which is an asymmetry of the
    /// encoding rather than of the arithmetic: group 0x08's `src2` has no negate at all. That is
    /// a fact worth pinning in both directions - a `neg` silently DROPPED is a sign error in
    /// every program that uses it, and a `neg` silently accepted on a source that has no bit for
    /// it is the same error wearing a success.
    #[test]
    fn every_source_modifier_the_alu_group_has_round_trips_and_the_rest_are_refused() {
        let mut checked = 0usize;
        for op in [Op::Mul, Op::Add, Op::Min, Op::Max] {
            for half in [false, true] {
                for (abs, neg) in [(false, false), (true, false), (false, true), (true, true)] {
                    let mut s1 = Src::reg(Bank::PrimaryAttr, 2);
                    if abs {
                        s1 = s1.absolute();
                    }
                    if neg {
                        s1 = s1.negated();
                    }
                    let word = alu(
                        op,
                        half,
                        Dest::new(Bank::Temp, 4),
                        [true; 4],
                        s1,
                        Src::reg(Bank::SecondaryAttr, 8),
                    )
                    .unwrap_or_else(|e| panic!("{op:?} half={half} abs={abs} neg={neg}: {e}"));
                    let got = decode::decode(word);
                    assert_eq!(
                        (got.srcs[0].abs, got.srcs[0].neg),
                        (abs, neg),
                        "{op:?} half={half}: src1 modifiers did not read back: {word:#018x}"
                    );
                    checked += 1;

                    // src2 carries ABS and no NEGATE - both halves asserted, because a refusal
                    // nobody checks is indistinguishable from an encoder that forgot the field.
                    let mut s2 = Src::reg(Bank::SecondaryAttr, 8);
                    if abs {
                        s2 = s2.absolute();
                    }
                    if neg {
                        s2 = s2.negated();
                    }
                    let out = alu(
                        op,
                        half,
                        Dest::new(Bank::Temp, 4),
                        [true; 4],
                        Src::reg(Bank::PrimaryAttr, 2),
                        s2,
                    );
                    if neg {
                        assert!(out.is_err(), "{op:?}: group 0x08's src2 has no negate modifier");
                    } else {
                        let got = decode::decode(out.unwrap_or_else(|e| panic!("{op:?}: {e}")));
                        assert_eq!(got.srcs[1].abs, abs, "{op:?}: src2's abs did not read back");
                        assert!(!got.srcs[1].neg, "{op:?}: src2 decoded a negate it cannot carry");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked >= 40, "only {checked} modifier combinations ran - the sweep is empty");
    }

    /// The MAD group's sixteen write masks, at each precision, swept with THAT precision's own
    /// operand form - and the two answers are DIFFERENT SHAPES, not one shape with a flag.
    ///
    /// >>> THE SWEEP CORRECTED ITS OWN PREMISE TWICE, which is the whole argument for sweeping.
    ///
    /// First it fed both precisions plain `.xyzw` operands, and the F32 arm refused all sixteen
    /// masks - for a reason that has nothing to do with masks. The F32 mad is a TWO-LANE
    /// operation whose operand tables hold two-lane patterns ([`MAD_F32_XY`]); a sweep that fed
    /// both the same operand reports the operand table's refusal as a fact about the mask.
    ///
    /// Then it expected sixteen masks at F16 and found FOUR. `mask_table_mad`'s own note says
    /// why, and this is the assembler side of it: the three bits are one bitmask over the
    /// destination's REGISTER LANES, and at 16 bits a lane is a register holding TWO f16
    /// channels. So the mask moves in pairs - `----`, `xy--`, `--zw`, `xyzw` - and a caller
    /// asking for `x---` there is asking for something the encoding does not have.
    ///
    /// Both directions are asserted at both precisions, because "the encoding cannot do this"
    /// and "the encoder forgot to write the field" look identical from one side.
    #[test]
    fn the_mad_groups_write_mask_is_three_lanes_at_f32_and_register_pairs_at_f16() {
        // The 32-bit arm's third lane is an A/B ARM (`VITASLOP_GXP_MAD_MASK16=0` is the old
        // two-table reading), so the sweep says which arm it measured rather than depending on
        // an environment it does not name.
        let lane2 = std::env::var("VITASLOP_GXP_MAD_MASK16").as_deref() != Ok("0");
        let mut accepted_at: Vec<Vec<[bool; 4]>> = Vec::new();
        for half in [false, true] {
            let swz = if half { [0, 1, 2, 3] } else { MAD_F32_XY };
            let mut accepted = Vec::new();
            for bits in 0u8..16 {
                let mask = [bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0];
                let Ok(word) = mad(
                    half,
                    Dest::new(Bank::Temp, 0),
                    mask,
                    Src::reg(Bank::PrimaryAttr, 0).swz(swz),
                    Src::reg(Bank::SecondaryAttr, 4).swz(swz),
                    Src::reg(Bank::Temp, 2).swz(swz),
                ) else {
                    continue;
                };
                let got = decode::decode(word);
                assert_eq!(got.op, Op::Mad, "half={half} mask={mask:?}: {word:#018x}");
                assert_eq!(got.half_precision, half, "half={half} mask={mask:?}");
                assert_eq!(got.write_mask, mask, "half={half}: the mask did not read back");
                accepted.push(mask);
            }
            accepted_at.push(accepted);
        }
        let (full, halfm) = (&accepted_at[0], &accepted_at[1]);

        // >>> AT 32 BITS: three per-channel lanes, and channel 3 has no mask bit at all.
        assert!(
            full.iter().all(|m| !m[3]),
            "an F32 mad encoded a write to channel 3, which has no mask bit: {full:?}"
        );
        let want_full = if lane2 { 8 } else { 4 };
        assert_eq!(
            full.len(),
            want_full,
            "the F32 mad should reach every mask over its {} lane(s): {full:?}",
            if lane2 { 3 } else { 2 }
        );

        // >>> AT 16 BITS: two lanes, each a REGISTER, so the mask moves in channel PAIRS.
        assert_eq!(
            halfm,
            &vec![
                [false, false, false, false],
                [true, true, false, false],
                [false, false, true, true],
                [true, true, true, true],
            ],
            "the F16 mad's mask is a bitmask over REGISTER lanes, two f16 channels each"
        );
        // Stated the other way too: no F16 mask splits a pair.
        assert!(
            halfm.iter().all(|m| m[0] == m[1] && m[2] == m[3]),
            "an F16 mad encoded a mask that writes half a register: {halfm:?}"
        );
    }

    /// Every ALU operation, both precisions, both operand orders, every write mask - assembled
    /// and decoded back. `alu` verifies internally, so reaching `unwrap` IS the assertion; the
    /// explicit checks below state what the test is about.
    #[test]
    fn every_alu_operation_round_trips_at_both_precisions() {
        let ops = [Op::Mul, Op::Add, Op::Frc, Op::Min, Op::Max];
        for op in ops {
            for half in [false, true] {
                for mask in [[true, false, false, false], [true, true, true, true], [false, true, false, true]] {
                    let word = alu(
                        op,
                        half,
                        Dest::new(Bank::Temp, 4),
                        mask,
                        Src::reg(Bank::PrimaryAttr, 2),
                        Src::reg(Bank::SecondaryAttr, 8),
                    )
                    .unwrap_or_else(|e| panic!("{op:?} half={half} mask={mask:?}: {e}"));
                    let got = decode::decode(word);
                    assert_eq!(got.op, op);
                    assert_eq!(got.half_precision, half);
                    assert_eq!(got.write_mask, mask);
                    assert_eq!(got.dest.unwrap().bank, Bank::Temp);
                    assert_eq!(got.dest.unwrap().index, 4);
                    assert_eq!(got.srcs[0].bank, Bank::PrimaryAttr);
                    assert_eq!(got.srcs[0].index, 2);
                    assert_eq!(got.srcs[1].bank, Bank::SecondaryAttr);
                    assert_eq!(got.srcs[1].index, 8);
                }
            }
        }
    }

    /// The mad group, including the write mask whose reading
    /// [`mask_table_mad_for`] documents: in 32-bit mode three lanes are reachable and channel 3
    /// is not, and the assembler says so with a refusal rather than an approximation.
    #[test]
    fn the_mad_group_round_trips_and_refuses_the_mask_it_cannot_encode() {
        let word = mad(
            false,
            Dest::new(Bank::Output, 0),
            [true, true, true, false],
            Src::reg(Bank::PrimaryAttr, 0).swz(MAD_F32_XY),
            Src::reg(Bank::SecondaryAttr, 4).swz(MAD_F32_XY),
            Src::reg(Bank::Temp, 2).swz(MAD_F32_XY),
        )
        .expect("a three-lane f32 mad is encodable");
        let got = decode::decode(word);
        assert_eq!(got.op, Op::Mad);
        assert_eq!(got.write_mask, [true, true, true, false]);
        assert_eq!(got.srcs.len(), 3);

        assert_eq!(
            mad(
                false,
                Dest::new(Bank::Output, 0),
                [true, true, true, true],
                Src::reg(Bank::PrimaryAttr, 0).swz(MAD_F32_XY),
                Src::reg(Bank::SecondaryAttr, 4).swz(MAD_F32_XY),
                Src::reg(Bank::Temp, 2).swz(MAD_F32_XY),
            ),
            Err(AsmError::WriteMask([true, true, true, true])),
            "a 32-bit mad has no mask bit for channel 3"
        );
    }

    /// The F16 mad is the FOUR-lane form of the same group - its operand tables carry
    /// four-channel patterns and its mask covers all four channels as two register pairs. The
    /// two precisions are not the same instruction with a flag, and the assembler expresses
    /// the difference by refusing each what the other has.
    #[test]
    fn the_f16_mad_reaches_four_channels_where_the_f32_mad_reaches_two() {
        let word = mad(
            true,
            Dest::new(Bank::Temp, 0),
            [true, true, true, true],
            Src::reg(Bank::PrimaryAttr, 0),
            Src::reg(Bank::SecondaryAttr, 4),
            Src::reg(Bank::Temp, 2),
        )
        .expect("a four-channel f16 mad is encodable");
        let got = decode::decode(word);
        assert!(got.half_precision);
        assert_eq!(got.write_mask, [true, true, true, true]);
        assert_eq!(got.srcs[0].swizzle, [0, 1, 2, 3]);

        assert_eq!(
            mad(
                false,
                Dest::new(Bank::Temp, 0),
                [true, false, false, false],
                Src::reg(Bank::PrimaryAttr, 0),
                Src::reg(Bank::SecondaryAttr, 4),
                Src::reg(Bank::Temp, 2),
            ),
            Err(AsmError::Swizzle { which: 1, want: [0, 1, 2, 3] }),
            "the F32 mad's operand tables hold two-lane patterns only"
        );
    }

    /// An odd register index has NO encoding in a six-bit double-register field, and an
    /// assembler that quietly rounded one would write a program addressing its neighbour.
    #[test]
    fn an_odd_register_index_is_refused_rather_than_rounded() {
        assert_eq!(
            alu(
                Op::Add,
                false,
                Dest::new(Bank::Temp, 5),
                [true, false, false, false],
                Src::reg(Bank::Temp, 0),
                Src::reg(Bank::Temp, 0),
            ),
            Err(AsmError::RegIndex { bank: Bank::Temp, index: 5 })
        );
    }

    /// The inline-constant and inline-immediate operand rows, and the per-channel selector
    /// that the corpus differential found the reference reading wrong.
    ///
    /// The constant sits on source 1 because that is the only operand of this group with a free
    /// per-channel swizzle, and the inline constants ARE swizzle selectors 4..7. Source 2's
    /// table cannot name them - so this test is also the statement of where a constant can go.
    #[test]
    fn the_constant_and_immediate_operand_rows_round_trip() {
        let word = alu(
            Op::Mul,
            false,
            Dest::new(Bank::Temp, 0),
            [true, true, false, false],
            Src::cnst(0).swz([5, 5, 5, 5]),
            Src::reg(Bank::PrimaryAttr, 0),
        )
        .expect("a multiply by the inline constant 1.0");
        let got = decode::decode(word);
        assert_eq!(got.srcs[0].bank, Bank::Constant);
        assert_eq!(got.srcs[0].swizzle, [5, 5, 5, 5]);

        assert_eq!(
            alu(
                Op::Mul,
                false,
                Dest::new(Bank::Temp, 0),
                [true, false, false, false],
                Src::reg(Bank::PrimaryAttr, 0),
                Src::cnst(0).swz([5, 5, 5, 5]),
            ),
            Err(AsmError::Swizzle { which: 2, want: [5, 5, 5, 5] }),
            "source 2's swizzle table cannot name an inline constant selector"
        );

        let word = alu(
            Op::Add,
            false,
            Dest::new(Bank::Temp, 0),
            [true, false, false, false],
            Src::imm(7),
            Src::reg(Bank::Temp, 0),
        )
        .expect("an add of the inline literal 7");
        let got = decode::decode(word);
        assert_eq!(got.srcs[0].bank, Bank::Immediate);
        assert_eq!(got.srcs[0].index, 7);
    }

    /// The register-INDIRECT source row: `sa[i0 + offset]`, the operand form an indexed uniform
    /// array read compiles to and the one a football title's crowd sprites turn on.
    #[test]
    fn the_indexed_source_row_round_trips_with_its_sub_bank_and_offset() {
        let word = alu(
            Op::Add,
            false,
            Dest::new(Bank::Temp, 0),
            [true, true, true, false],
            Src::reg(Bank::Temp, 2),
            Src::indexed(Bank::SecondaryAttr, 14, 0),
        )
        .expect("an add reading sa[i0 + 14]");
        let got = decode::decode(word);
        assert_eq!(got.srcs[1].bank, Bank::Indexed);
        assert_eq!(crate::ir::indexed_sub_bank(got.srcs[1].index), Bank::SecondaryAttr);
        assert_eq!(crate::ir::indexed_offset(got.srcs[1].index), 14);
    }

    /// The PACK group, in both float directions, round-tripped through the decoder. F32 -> F16
    /// is the conversion a fragment program spends most of its life in.
    #[test]
    fn the_pack_group_round_trips_in_both_float_directions() {
        let narrow = pack(
            Dest::new(Bank::Temp, 8),
            PackFmt::F16,
            Bank::PrimaryAttr,
            0,
            PackFmt::F32,
            [0, 1, 2, 3],
            [true; 4],
            false,
        )
        .expect("f32 -> f16");
        let got = decode::decode(narrow);
        assert_eq!(got.op, Op::Pack { src_half: false });
        assert!(got.half_precision, "the destination is the F16 view");
        assert_eq!(got.dest.unwrap().index, 8);
        assert_eq!(got.write_mask, [true; 4]);

        let widen = pack(
            Dest::new(Bank::Output, 3),
            PackFmt::F32,
            Bank::Temp,
            8,
            PackFmt::F16,
            [0, 1, 0, 0],
            [true, true, false, false],
            false,
        )
        .expect("f16 -> f32");
        let got = decode::decode(widen);
        assert_eq!(got.op, Op::Pack { src_half: true });
        assert!(!got.half_precision);
        // The DESTINATION number is direct here, so an odd register is addressable - which the
        // six-bit source field cannot do. The two ends of one instruction are numbered
        // differently, and this is the assertion that says so.
        assert_eq!(got.dest.unwrap().index, 3);
    }

    /// The NORMALIZED conversion is a different instruction from the plain cast, and the
    /// `scale` bit is what tells them apart. A case that meant one must not assemble the other.
    #[test]
    fn the_normalised_and_plain_integer_conversions_are_different_instructions() {
        let plain = pack(
            Dest::new(Bank::Temp, 0),
            PackFmt::U8,
            Bank::PrimaryAttr,
            0,
            PackFmt::F32,
            [0, 1, 2, 3],
            [true; 4],
            false,
        )
        .expect("a truncating float -> u8 cast");
        assert_eq!(
            decode::decode(plain).op,
            Op::PackToInt { bits: 8, signed: false, src_half: false }
        );

        let normalised = pack(
            Dest::new(Bank::Temp, 0),
            PackFmt::U8,
            Bank::PrimaryAttr,
            0,
            PackFmt::F32,
            [0, 1, 2, 3],
            [true; 4],
            true,
        )
        .expect("a normalised float -> unorm8 conversion");
        assert_eq!(
            decode::decode(normalised).op,
            Op::PackUnorm8 { to_unorm8: true, float_half: false }
        );
    }

    /// The texture sample: the unit, the coordinate count and the FOUR-channel destination.
    #[test]
    fn a_texture_sample_round_trips_with_its_unit_and_coordinate_count() {
        for coords in [1u8, 2] {
            for ordinal in [0u8, 1, 12] {
                let word = tex(Dest::new(Bank::Temp, 12), ordinal, Bank::PrimaryAttr, 4, coords)
                    .unwrap_or_else(|e| panic!("ordinal {ordinal}, {coords} coords: {e}"));
                let got = decode::decode(word);
                match got.op {
                    Op::Tex { unit: u, coords: c, .. } => {
                        // Bare, `unit` is the raw sampler ordinal: the container has not been
                        // consulted, because a word on its own has no container.
                        assert_eq!(u, ordinal);
                        assert_eq!(c, coords);
                    }
                    other => panic!("not a sample: {other:?}"),
                }
                assert_eq!(got.dest.unwrap().index, 12);
                assert_eq!(
                    got.write_mask,
                    [true; 4],
                    "a sample writes four channels whatever its coordinate count"
                );
            }
        }
    }

    /// The index-register load, and the pairing that gives it meaning: the load sets `i0` and
    /// the instruction after it reads `sa[i0 + offset]`.
    #[test]
    fn an_index_load_and_the_indexed_read_that_consumes_it_round_trip() {
        let load = load_index(Bank::Temp, 3, 5).expect("i0 = int(r[3]) + 5");
        let got = decode::decode(load);
        assert_eq!(got.op, Op::LoadIndex { addend: 5, to_index: true, stride: 1 });
        assert_eq!(got.dest.unwrap().bank, Bank::Index);
        assert_eq!(got.srcs[0].bank, Bank::Temp);
        assert_eq!(got.srcs[0].index, 3);
        assert_eq!(got.blocked, None, "the assembled word is the established encoding");

        // The source's other bank, which is the single bit a football title's crowd word
        // differs from the seven pinned ones in.
        let from_pa = load_index(Bank::PrimaryAttr, 3, 5).expect("i0 = int(pa[3]) + 5");
        assert_eq!(decode::decode(from_pa).srcs[0].bank, Bank::PrimaryAttr);
        assert_ne!(from_pa, load, "the source bank is bit 34 and it moved");
    }

    /// A bank the index load's ONE-BIT source field cannot name is refused rather than folded
    /// onto one it can.
    #[test]
    fn the_index_load_refuses_a_source_bank_it_cannot_encode() {
        assert_eq!(
            load_index(Bank::SecondaryAttr, 0, 0),
            Err(AsmError::SrcBank { which: 0, bank: Bank::SecondaryAttr })
        );
    }

    /// A swizzle the group's table cannot express is a REFUSAL. Group 0x08's operand-2 table
    /// holds sixteen patterns and no more; asking for a seventeenth must not silently land on
    /// the nearest one.
    #[test]
    fn a_swizzle_outside_the_groups_table_is_refused() {
        // `wzyx` (a full reverse) is in neither operand-2 table.
        let err = alu(
            Op::Add,
            false,
            Dest::new(Bank::Temp, 0),
            [true, false, false, false],
            Src::reg(Bank::Temp, 0),
            Src::reg(Bank::Temp, 2).swz([3, 2, 1, 0]),
        );
        assert_eq!(err, Err(AsmError::Swizzle { which: 2, want: [3, 2, 1, 0] }));
    }

    /// The assembled move is a real move: `o[0..3] = pa[2..5]`, as an add of the inline zero.
    #[test]
    fn a_move_assembled_as_an_add_of_zero_reads_back_as_that_move() {
        let word = mov(
            false,
            Dest::new(Bank::Output, 0),
            [true, true, true, true],
            Src::reg(Bank::PrimaryAttr, 2),
        )
        .expect("a four-lane move");
        let got = decode::decode(word);
        assert_eq!(got.op, Op::Add);
        assert_eq!(got.srcs[0].bank, Bank::Constant);
        assert_eq!(got.srcs[0].swizzle, [4, 4, 4, 4], "the addend is the inline 0.0");
        assert_eq!(got.srcs[1].bank, Bank::PrimaryAttr);
        assert_eq!(got.srcs[1].index, 2);
        assert_eq!(got.write_mask, [true, true, true, true]);
    }

    /// >>> EVERY PACK SELECTOR, AT EVERY SOURCE FORMAT, IN BOTH DIRECTIONS.
    ///
    /// The assembler's round-trip check is the thing that makes it safe to write a case by
    /// hand, and it can only check words somebody asked for. Nobody had ever asked for `comp0`
    /// of 2 or 3 at a 16-bit source format, and the assembler could not encode one: it put
    /// comp0's high bit at bit 1 for every non-F32 format while the decoder had moved format 3
    /// to bit 7, so every such word came back with the bit lost and the round-trip REFUSED it.
    ///
    /// A refusal reads as "the encoding cannot express this" - which is how three of the
    /// assembler's real ISA facts were learned - and here it was not an encoding fact at all,
    /// it was two transcriptions of one rule disagreeing. Both now read
    /// `decode::pack_comp0_high_bit`, and this sweeps the whole selector space so a future
    /// disagreement cannot hide in the corner nobody writes cases for.
    ///
    /// This is the shape [[the enumeration item]] argues for generally: where a per-case number
    /// cannot be hand-written, state a PROPERTY - here, that a word decodes back to the
    /// operand it was asked for - and sweep the space the decoder's own field tables define.
    #[test]
    fn every_pack_selector_round_trips_at_every_source_format() {
        use PackFmt::*;
        let mut checked = 0usize;
        for src_fmt in [U8, S8, U16, S16, F16, F32] {
            for dest_fmt in [U8, S8, U16, S16, F16, F32] {
                // A pack needs one side to be a FLOAT or the widths to match; the assembler
                // refuses the rest on its own, and a refusal there is not this test's subject.
                for c0 in 0..4u8 {
                    for c1 in 0..4u8 {
                        let sw = [c0, c1, (c0 + 2) % 4, (c1 + 1) % 4];
                        let Ok(word) = pack(
                            Dest::new(Bank::Temp, 0),
                            dest_fmt,
                            Bank::PrimaryAttr,
                            0,
                            src_fmt,
                            sw,
                            [true; 4],
                            false,
                        ) else {
                            continue;
                        };
                        let got = decode::decode(word);
                        let s = got.srcs.first().expect("a pack has a source");
                        assert_eq!(
                            s.swizzle, sw,
                            "src_fmt {src_fmt:?} dest_fmt {dest_fmt:?} selector {sw:?}                              decoded back as {:?} from {word:#018x}",
                            s.swizzle
                        );
                        checked += 1;
                    }
                }
            }
        }
        // The sweep has to BE a sweep: a refusal everywhere would make every assertion vacuous
        // and the test would pass having checked nothing.
        assert!(checked >= 200, "only {checked} pack words round-tripped - the sweep is empty");
    }

    /// >>> THE PACK GROUP'S REMAINING SPACE: the full 256-selector square, every write mask,
    /// >>> both conversion kinds, and the FORMAT MATRIX printed as a matrix.
    ///
    /// The sweep above walks 16 of the 256 selector quadruples at each format pair, which was
    /// enough to catch a selector written into the wrong field but not enough to state the
    /// space. This states it, and it states the thing no test here said at all: WHICH of the 36
    /// source/destination format pairs have an encoding.
    ///
    /// >>> THAT MATRIX IS WHY THIS IS WORTH A SWEEP RATHER THAN CASES. Real fragment programs
    /// are 70-90% F16 and every value crossing between the pipelines crosses here, so a pair
    /// that silently has no encoding is a conversion some title does that this crate would
    /// refuse - and a pair that encodes when it should not is a word meaning a different
    /// conversion from the one asked for. Both are invisible until someone counts.
    ///
    /// The comp0 selector's HIGH bit moves with the source format (`pack_comp0_high_bit`), so
    /// the square is swept at EVERY format rather than sampled at one: a selector of 0..1 would
    /// never exercise that bit, and 2..3 exercise nothing else.
    #[test]
    fn the_pack_groups_selector_square_mask_and_format_matrix_are_swept() {
        use PackFmt::*;
        const FMTS: [PackFmt; 6] = [U8, S8, U16, S16, F16, F32];
        let name = |f: PackFmt| match f {
            U8 => "u8",
            S8 => "s8",
            U16 => "u16",
            S16 => "s16",
            F16 => "f16",
            F32 => "f32",
        };

        // >>> THE WHOLE 256-SELECTOR SQUARE, at every format pair that encodes at all.
        let (mut selectors, mut pairs) = (0usize, 0usize);
        let mut matrix: Vec<String> = Vec::new();
        for src_fmt in FMTS {
            let mut row = String::new();
            for dest_fmt in FMTS {
                let mut here = 0usize;
                for raw in 0..256u16 {
                    let sw = [
                        (raw & 3) as u8,
                        ((raw >> 2) & 3) as u8,
                        ((raw >> 4) & 3) as u8,
                        ((raw >> 6) & 3) as u8,
                    ];
                    let Ok(word) = pack(
                        Dest::new(Bank::Temp, 0),
                        dest_fmt,
                        Bank::PrimaryAttr,
                        0,
                        src_fmt,
                        sw,
                        [true; 4],
                        false,
                    ) else {
                        continue;
                    };
                    let got = decode::decode(word);
                    let s = got.srcs.first().expect("a pack has a source");
                    assert_eq!(
                        s.swizzle,
                        sw,
                        "{} -> {}: selector {sw:?} read back as {:?} from {word:#018x}",
                        name(src_fmt),
                        name(dest_fmt),
                        s.swizzle
                    );
                    here += 1;
                }
                row.push_str(&format!("{here:>5}"));
                selectors += here;
                if here > 0 {
                    pairs += 1;
                    assert_eq!(
                        here,
                        256,
                        "{} -> {} encodes SOME selectors but not all - a per-channel selector \
                         that depends on the value it selects is not a selector",
                        name(src_fmt),
                        name(dest_fmt)
                    );
                }
            }
            matrix.push(format!("    {:>4} |{row}", name(src_fmt)));
        }
        println!(
            "  0x40 PACK  format matrix (selectors encodable, src down / dest across)\n         |{}",
            FMTS.iter().map(|f| format!("{:>5}", name(*f))).collect::<String>()
        );
        for row in &matrix {
            println!("{row}");
        }
        println!("             {pairs} of 36 format pairs encode; {selectors} selector words swept");
        assert!(pairs >= 12, "only {pairs} of 36 format pairs encode - the group is barely reachable");

        // >>> EVERY WRITE MASK, and it must be the SAME SET at every format pair that encodes.
        // The mask is four plain bits at [37:34] and has nothing to do with the conversion, so
        // a set that varied would mean the mask field is being read as something else somewhere.
        let masks_for = |src_fmt: PackFmt, dest_fmt: PackFmt| -> Vec<[bool; 4]> {
            (0u8..16)
                .map(|b| [b & 1 != 0, b & 2 != 0, b & 4 != 0, b & 8 != 0])
                .filter(|m| {
                    pack(
                        Dest::new(Bank::Temp, 0),
                        dest_fmt,
                        Bank::PrimaryAttr,
                        0,
                        src_fmt,
                        [0, 1, 2, 3],
                        *m,
                        false,
                    )
                    .is_ok()
                })
                .collect()
        };
        let mut mask_reference: Option<(PackFmt, PackFmt, Vec<[bool; 4]>)> = None;
        for src_fmt in FMTS {
            for dest_fmt in FMTS {
                let got = masks_for(src_fmt, dest_fmt);
                if got.is_empty() {
                    continue;
                }
                match &mask_reference {
                    None => mask_reference = Some((src_fmt, dest_fmt, got)),
                    Some((rs, rd, want)) => assert_eq!(
                        &got,
                        want,
                        "the encodable write masks differ between {} -> {} and {} -> {}",
                        name(*rs),
                        name(*rd),
                        name(src_fmt),
                        name(dest_fmt)
                    ),
                }
            }
        }
        let (_, _, masks) = mask_reference.expect("at least one format pair encodes");
        println!("             write masks   {} of 16, the same set at every format pair", masks.len());
        assert_eq!(masks.len(), 16, "the pack mask is four plain bits: {masks:?}");

        // >>> AND THE NORMALISED CONVERSION IS A DIFFERENT INSTRUCTION, not a flag on this one.
        // `scale` selects integer-0..max against float-0..1; a case that meant one must not be
        // able to assemble the other by accident, so both arms are swept and required to differ
        // in the decoded OPERATION rather than only in a bit.
        //
        // >>> ONLY WHERE ONE SIDE IS AN INTEGER, and the other half of that is asserted too.
        // MEASURED, and it corrected the first version of this sweep: `f16 -> f16` decodes
        // identically with the flag set and clear, which is right - there is no 0..1 range to
        // normalise against when neither side is an integer, so the bit has nothing to select.
        // A sweep that demanded a difference everywhere would have reported that as a defect.
        let is_int = |f: PackFmt| matches!(f, U8 | S8 | U16 | S16);
        let mut scaled_pairs = 0usize;
        let mut float_pairs_ignoring_the_flag = 0usize;
        for src_fmt in FMTS {
            for dest_fmt in FMTS {
                if is_int(src_fmt) == is_int(dest_fmt) {
                    // Neither a normalisation nor an integer-to-integer one: the flag selects
                    // nothing, and the word must not change behind it.
                    let plain = pack(
                        Dest::new(Bank::Temp, 0),
                        dest_fmt,
                        Bank::PrimaryAttr,
                        0,
                        src_fmt,
                        [0, 1, 2, 3],
                        [true; 4],
                        false,
                    );
                    let scaled = pack(
                        Dest::new(Bank::Temp, 0),
                        dest_fmt,
                        Bank::PrimaryAttr,
                        0,
                        src_fmt,
                        [0, 1, 2, 3],
                        [true; 4],
                        true,
                    );
                    if let (Ok(p), Ok(s)) = (plain, scaled) {
                        assert_eq!(
                            decode::decode(p).op,
                            decode::decode(s).op,
                            "{} -> {}: neither side is an integer, so there is no 0..1 range for \
                             the normalise flag to select and it must change nothing",
                            name(src_fmt),
                            name(dest_fmt)
                        );
                        float_pairs_ignoring_the_flag += 1;
                    }
                    continue;
                }
                let plain = pack(
                    Dest::new(Bank::Temp, 0),
                    dest_fmt,
                    Bank::PrimaryAttr,
                    0,
                    src_fmt,
                    [0, 1, 2, 3],
                    [true; 4],
                    false,
                );
                let scaled = pack(
                    Dest::new(Bank::Temp, 0),
                    dest_fmt,
                    Bank::PrimaryAttr,
                    0,
                    src_fmt,
                    [0, 1, 2, 3],
                    [true; 4],
                    true,
                );
                let (Ok(plain), Ok(scaled)) = (plain, scaled) else { continue };
                assert_ne!(
                    decode::decode(plain).op,
                    decode::decode(scaled).op,
                    "{} -> {}: the normalised conversion decodes as the SAME operation as the \
                     plain one, so nothing distinguishes them",
                    name(src_fmt),
                    name(dest_fmt)
                );
                scaled_pairs += 1;
            }
        }
        println!(
            "             {scaled_pairs} pair(s) have a distinct NORMALISED conversion; \
             {float_pairs_ignoring_the_flag} same-kind pair(s) correctly ignore the flag"
        );
        assert!(scaled_pairs > 0, "no format pair distinguishes the normalised conversion");
        assert!(
            float_pairs_ignoring_the_flag > 0,
            "the other half of the claim was never exercised"
        );
    }

    /// >>> THE PACK GROUP'S TWO REGISTER NUMBERINGS ARE NOT THE SAME NUMBERING, and the
    /// >>> asymmetry is the encoding's.
    ///
    /// The destination is a DIRECT seven-bit number whose top four values name the internal
    /// registers; the source is an R6 double-register that only reaches even numbers. So the
    /// destination accepts an odd register and the source does not, and a reader that applied
    /// one rule to both would be off by a factor of two on every pack in every fragment program
    /// - which is 70-90% of the instructions a real one runs.
    #[test]
    fn a_packs_destination_is_direct_where_its_source_is_doubled() {
        use PackFmt::*;
        // ODD destinations exist, and read back as themselves.
        for n in [1u8, 3, 17, 123] {
            let word = pack(
                Dest::new(Bank::Temp, n),
                F16,
                Bank::PrimaryAttr,
                0,
                F32,
                [0, 1, 2, 3],
                [true; 4],
                false,
            )
            .unwrap_or_else(|e| panic!("destination r{n}: {e:?}"));
            let d = decode::decode(word).dest.expect("a pack writes something");
            assert_eq!((d.bank, d.index), (Bank::Temp, n), "{word:#018x}");
        }
        // 124..=127 are the internal registers' reserved range, so an ordinary register there
        // has no encoding rather than a colliding one.
        for n in [124u8, 125, 127] {
            assert!(
                pack(
                    Dest::new(Bank::Temp, n),
                    F16,
                    Bank::PrimaryAttr,
                    0,
                    F32,
                    [0, 1, 2, 3],
                    [true; 4],
                    false,
                )
                .is_err(),
                "r{n} is the internal-register range, not a temporary"
            );
        }
        // And the internal registers ARE reachable there, four lanes apart.
        for n in 0..4u8 {
            let word = pack(
                Dest::new(Bank::Internal, n * 4),
                F16,
                Bank::PrimaryAttr,
                0,
                F32,
                [0, 1, 2, 3],
                [true; 4],
                false,
            )
            .unwrap_or_else(|e| panic!("destination i{n}: {e:?}"));
            let d = decode::decode(word).dest.expect("a pack writes something");
            assert_eq!((d.bank, d.index), (Bank::Internal, n * 4));
        }
        // The SOURCE is doubled: odd numbers have no encoding at all.
        for n in [1u8, 3, 41] {
            assert!(
                pack(
                    Dest::new(Bank::Temp, 0),
                    F16,
                    Bank::PrimaryAttr,
                    n,
                    F32,
                    [0, 1, 2, 3],
                    [true; 4],
                    false,
                )
                .is_err(),
                "a pack source is a double-register, so r{n} has no encoding"
            );
        }
        let mut even = 0usize;
        for n in (0u8..118).step_by(2) {
            let Ok(word) = pack(
                Dest::new(Bank::Temp, 0),
                F16,
                Bank::PrimaryAttr,
                n,
                F32,
                [0, 1, 2, 3],
                [true; 4],
                false,
            ) else {
                continue;
            };
            let s = decode::decode(word).srcs.first().copied().expect("a pack has a source");
            assert_eq!((s.bank, s.index), (Bank::PrimaryAttr, n), "{word:#018x}");
            even += 1;
        }
        println!("  0x40 PACK  source registers {even} even numbers round-trip; every odd one is refused");
        assert!(even >= 32, "only {even} even source registers encode");
    }

    /// >>> EVERY FLOAT VTST ROUND-TRIPS, AT BOTH WIDTHS AND EVERY RELATION - which is the sweep
    /// >>> the group that produced a real defect had never had.
    ///
    /// A hardwired `Prec::F32` in the reference's float-TEST arm was the first divergence in
    /// five corpus programs across three titles, and no case could be authored for it because
    /// this group was not assemblable. The PRECISION bit is inverted (`prec == 0` is HALF), and
    /// an inverted bit that only one width exercises is exactly what a single hand-written case
    /// would have missed.
    #[test]
    fn every_float_vtst_round_trips_at_both_widths() {
        let mut checked = 0usize;
        let mut split = 0usize;
        for alu in [TestAlu::Add, TestAlu::Sub, TestAlu::Mul] {
            for cmp in [TestCmp::Eq, TestCmp::Ne, TestCmp::Lt, TestCmp::Le, TestCmp::Gt, TestCmp::Ge] {
                for reduce in [
                    TestReduce::Channel(0),
                    TestReduce::Channel(3),
                    TestReduce::AndAll,
                    TestReduce::OrAll,
                ] {
                    let mut accepted = [false; 2];
                    for (i, half) in [true, false].into_iter().enumerate() {
                        match vtst(alu, cmp, reduce, 2, half, Bank::SecondaryAttr, 4, Bank::PrimaryAttr, 6) {
                            Ok(word) => {
                                accepted[i] = true;
                                let got = decode::decode(word);
                                assert_eq!(got.half_precision, half, "width {half}: {word:#018x}");
                                match got.op {
                                    Op::Test { alu: a, cmp: c, reduce: r, pdst, .. } => {
                                        assert_eq!((a, c, r, pdst), (alu, cmp, reduce, 2), "{word:#018x}");
                                    }
                                    other => panic!("not a test: {other:?} from {word:#018x}"),
                                }
                                checked += 1;
                            }
                            Err(_) => accepted[i] = false,
                        }
                    }
                    // >>> AND WHETHER A VTST EXISTS MUST NOT DEPEND ON ITS WIDTH. The same
                    // property the pack group's sweep asserts about its selectors, for the same
                    // reason: a refusal that tracks one field reads as an encoding fact and is
                    // usually two transcriptions disagreeing.
                    if accepted[0] != accepted[1] {
                        split += 1;
                        println!("  {alu:?} {cmp:?} {reduce:?}: half={} full={}", accepted[0], accepted[1]);
                    }
                }
            }
        }
        assert_eq!(split, 0, "a vtst's existence must not depend on its PRECISION (listed above)");
        assert!(checked >= 100, "only {checked} vtst words round-tripped - the sweep is empty");
    }

    /// >>> EVERY KILL THIS ENCODING CAN EXPRESS, ROUND-TRIPPED - and every predicate it cannot
    /// >>> REFUSED rather than approximated.
    ///
    /// The four the 2-bit field names are not the four a caller might expect: there is no
    /// `IfP(1)`, and `IfNotP(1)` sits where a plain `ShortPredicate` reading would put `P1`.
    /// That ordering is what a football title's alpha test settles - under the rival reading the
    /// discard would erase exactly the texels that PASSED - so an assembler that quietly mapped
    /// an unencodable predicate onto a neighbour would be writing the inverse of an alpha test.
    #[test]
    fn every_encodable_kill_predicate_round_trips_and_the_rest_are_refused() {
        for pred in [Predicate::Always, Predicate::IfNotP(0), Predicate::IfNotP(1), Predicate::IfP(0)] {
            let word = kill(pred).unwrap_or_else(|e| panic!("{pred:?}: {e}"));
            let got = decode::decode(word);
            assert_eq!(got.op, Op::Kill, "{word:#018x}");
            assert_eq!(got.pred, pred, "{word:#018x}");
            assert!(got.blocked.is_none(), "{pred:?} decoded blocked: {:?}", got.blocked);
            assert!(got.dest.is_none(), "a kill writes no register");
        }
        for pred in [Predicate::IfP(1), Predicate::IfP(2), Predicate::IfNotP(2), Predicate::Raw(3)] {
            assert!(kill(pred).is_err(), "{pred:?} has no encoding here and must be refused");
        }
    }

    /// >>> EVERY BANK AND EVERY REGISTER A DEPTHF CAN NAME, ROUND-TRIPPED.
    ///
    /// The register field is read DIRECT rather than double-register scaled, and the corpus
    /// cannot tell the two readings apart - its single DEPTHF names register 0, where they
    /// agree. So the sweep is the evidence: under a doubled reading every ODD register here
    /// would come back as its neighbour, and half of these assertions would fail.
    ///
    /// >>> AND WHETHER A DEPTHF EXISTS MUST NOT DEPEND ON WHICH BANK IT NAMES. The same property
    /// the pack and test groups assert about their own selectors, for the same reason: a refusal
    /// that tracks one field reads as an encoding fact and is usually two transcriptions
    /// disagreeing.
    #[test]
    fn every_depthf_source_round_trips_and_its_register_is_not_doubled() {
        let mut checked = 0usize;
        let mut accepted_per_bank = Vec::new();
        for bank in [Bank::Temp, Bank::PrimaryAttr, Bank::Output, Bank::SecondaryAttr] {
            let mut accepted = 0usize;
            for reg in [0u8, 1, 2, 7, 63, 64, 100, 123] {
                let Ok(word) = depthf(bank, reg) else { continue };
                accepted += 1;
                let got = decode::decode(word);
                assert_eq!(got.op, Op::DepthF, "{bank:?}[{reg}]: {word:#018x}");
                assert!(got.blocked.is_none(), "{bank:?}[{reg}] blocked: {:?}", got.blocked);
                let src = got.srcs.first().expect("a depthf has one source");
                assert_eq!((src.bank, src.index), (bank, reg), "{word:#018x} named the wrong register");
                checked += 1;
            }
            accepted_per_bank.push((bank, accepted));
        }
        assert!(checked >= 28, "only {checked} depthf words round-tripped - the sweep is empty");
        let first = accepted_per_bank[0].1;
        for (bank, n) in &accepted_per_bank {
            assert_eq!(*n, first, "a depthf's existence must not depend on its BANK ({bank:?})");
        }

        // The top four TEMP numbers name an internal register, not a temporary, so they are
        // refused rather than encoded into a word that means something else.
        for reg in 124..=127u8 {
            assert!(depthf(Bank::Temp, reg).is_err(), "Temp[{reg}] is i{} and has no depthf encoding", (reg - 124));
        }
        assert!(depthf(Bank::Internal, 0).is_err(), "the index bank cannot hold a depth");
    }

    /// >>> THE TWO FRAGMENT-PIPELINE WORDS MUST NOT COLLIDE WITH EACH OTHER OR WITH A NO-OP.
    ///
    /// Both are members of group 0xF8, whose last arm is a documented CATCH-ALL that classifies
    /// as `Op::Nop` - so an encoder that got a discriminant bit wrong would produce a word that
    /// decodes perfectly, means nothing, and passes every round-trip that only asked whether the
    /// decode succeeded. This asks the discriminating question instead.
    /// >>> THE 0x18 DOT'S WHOLE ENCODABLE SPACE: both channel counts, both operand-2 tables,
    /// >>> every `src1` selector and every write mask.
    ///
    /// The numbers are PRINTED, not only asserted. "The table expresses some" is not a
    /// measurement, and the two operand-2 tables here are different SIZES from one another -
    /// a fact about the encoding that only a sweep can state.
    #[test]
    fn the_dot_groups_encodable_space_is_two_tables_and_a_repeat_count() {
        // src1's per-channel selector: three free bits per channel, as in the ALU group.
        let mut precise = 0usize;
        for a in 0..8u8 {
            for b in 0..8u8 {
                for c in 0..8u8 {
                    for d in 0..8u8 {
                        let want = [a, b, c, d];
                        let Ok(word) = dot(
                            4,
                            Dest::new(Bank::Temp, 4),
                            [true; 4],
                            Src::reg(Bank::PrimaryAttr, 2).swz(want),
                            Src::reg(Bank::Internal, 4),
                            0,
                        ) else {
                            continue;
                        };
                        let got = decode::decode(word);
                        assert_eq!(got.srcs[0].swizzle, want, "src1 swizzle: {word:#018x}");
                        // And it must not have disturbed the operand that shares the word.
                        assert_eq!(
                            (got.srcs[1].bank, got.srcs[1].index),
                            (Bank::Internal, 4),
                            "src1's swizzle moved op2: {word:#018x}"
                        );
                        precise += 1;
                    }
                }
            }
        }
        assert_eq!(precise, 4096, "src1's swizzle is a free 3-bit selector per channel");

        // >>> AND OPERAND 2 HAS TWO TABLES, chosen by the CHANNEL COUNT rather than by any
        // swizzle field - so a swizzle that exists at four channels can be unencodable at three,
        // and an assembler that searched one table for both would emit a wrong word silently.
        let table_for = |components: u8| -> Vec<[u8; 4]> {
            let mut v = Vec::new();
            for a in 0..9u8 {
                for b in 0..9u8 {
                    for c in 0..9u8 {
                        for d in 0..9u8 {
                            let want = [a, b, c, d];
                            if dot(
                                components,
                                Dest::new(Bank::Temp, 4),
                                [true; 4],
                                Src::reg(Bank::PrimaryAttr, 2),
                                Src::reg(Bank::Internal, 4).swz(want),
                                0,
                            )
                            .is_ok()
                            {
                                v.push(want);
                            }
                        }
                    }
                }
            }
            v
        };
        let t3 = table_for(3);
        let t4 = table_for(4);
        println!("  0x18 DOT   src1 swizzle  {precise} of 4096 (a free 3-bit selector per channel)");
        println!(
            "             op2 swizzle   {} at 3 channels, {} at 4 - TWO tables, selected by the \
             channel count and not by any swizzle field",
            t3.len(),
            t4.len()
        );
        assert!(!t3.is_empty() && !t4.is_empty(), "neither operand-2 table expresses anything");
        assert_ne!(t3, t4, "the two tables are the encoding's own, and they are not the same");
        // Every one that IS expressible must come back exactly, at its own channel count.
        for (components, table) in [(3u8, &t3), (4u8, &t4)] {
            for want in table {
                let word = dot(
                    components,
                    Dest::new(Bank::Temp, 4),
                    [true; 4],
                    Src::reg(Bank::PrimaryAttr, 2),
                    Src::reg(Bank::Internal, 4).swz(*want),
                    0,
                )
                .unwrap();
                let got = decode::decode(word);
                assert_eq!(got.srcs[1].swizzle, *want, "{components}ch {want:?}: {word:#018x}");
                assert_eq!(got.op, Op::Dot { components }, "{word:#018x}");
            }
        }

        // The write masks, and they must be the SAME SET at both channel counts: the mask comes
        // from the shared 0x08 table and has nothing to do with how many channels are reduced.
        //
        // >>> AND OP2 IS GIVEN A SWIZZLE ITS OWN TABLE HAS, because the two tables do not share
        // one. `xyzw` - what `Src::reg` defaults to - exists at four channels and NOT at three,
        // so a mask sweep that held the operand fixed across both counts would measure the
        // swizzle's absence and report it as "no mask encodes at three channels".
        let masks_for = |components: u8, op2: [u8; 4]| -> Vec<[bool; 4]> {
            (0u8..16)
                .map(|b| [b & 1 != 0, b & 2 != 0, b & 4 != 0, b & 8 != 0])
                .filter(|m| {
                    dot(
                        components,
                        Dest::new(Bank::Temp, 4),
                        *m,
                        Src::reg(Bank::PrimaryAttr, 2),
                        Src::reg(Bank::Internal, 4).swz(op2),
                        0,
                    )
                    .is_ok()
                })
                .collect()
        };
        let m3 = masks_for(3, t3[0]);
        assert_eq!(
            m3,
            masks_for(4, t4[0]),
            "a DOT's write mask does not depend on its channel count"
        );
        println!("             write masks   {} of 16", m3.len());
        assert!(m3.len() >= 8, "only {} of 16 masks encode: {m3:?}", m3.len());
    }

    /// >>> THE REPEAT COUNT, WHICH IS THE DIFFERENCE BETWEEN A MATRIX TRANSFORM AND A BLACK
    /// >>> FRAME - and the one bit of it that is spelled twice.
    ///
    /// [`decode::repeat_extra_iterations`]'s note records what reading this wrong costs: a
    /// retail title's world vertex program is ONE repeating DOT, and emitted once it produces a
    /// single scalar where a whole clip position belongs. That reading rested on a corpus census
    /// and a destination-closure argument. This is the other direction - a word WRITTEN to
    /// repeat, which no census can supply.
    #[test]
    fn a_dots_repeat_count_round_trips_and_its_unobserved_values_are_refused() {
        // A repeating DOT walks its destination one CHANNEL per iteration, so its mask names
        // exactly one - the decoder's own limit, asserted here from the writing side.
        for extra in 0..=3u32 {
            let word = dot(
                4,
                Dest::new(Bank::Output, 0),
                [true, false, false, false],
                Src::reg(Bank::SecondaryAttr, 0),
                Src::reg(Bank::Internal, 0),
                extra,
            )
            .unwrap_or_else(|e| panic!("a DOT repeating {extra} extra time(s): {e:?}"));
            assert_eq!(
                decode::repeat_extra_iterations(word),
                Some(extra),
                "repeat {extra}: {word:#018x}"
            );
            let got = decode::decode(word);
            assert!(got.blocked.is_none(), "repeat {extra}: {:?}", got.blocked);
            assert_eq!(got.op, Op::Dot { components: 4 });
        }

        // A MULTI-CHANNEL mask with a repeat is refused: the per-iteration channel step is
        // established only for the single-channel form, and a four-lane transform written into
        // a four-channel mask would stomp what its earlier iterations just wrote.
        for extra in 1..=3u32 {
            assert!(
                dot(
                    4,
                    Dest::new(Bank::Output, 0),
                    [true, true, false, false],
                    Src::reg(Bank::SecondaryAttr, 0),
                    Src::reg(Bank::Internal, 0),
                    extra,
                )
                .is_err(),
                "a repeating DOT with a two-channel mask must be refused (extra={extra})"
            );
        }
        // And a count past the census is refused rather than encoded into bit 46.
        assert!(
            dot(
                4,
                Dest::new(Bank::Output, 0),
                [true, false, false, false],
                Src::reg(Bank::SecondaryAttr, 0),
                Src::reg(Bank::Internal, 0),
                4,
            )
            .is_err(),
            "a repeat count of four would set bit 46, which is spelled `abs_op2` as well"
        );

        // >>> BIT 46 IS SPELLED TWICE AND THE CORPUS NEVER SETS IT. Of 1,884 corpus DOTs the
        // field 47:44 takes six values - 0x8 (1,645), 0x2 (130), 0x1 (52), 0x3 (49), 0x9 (7),
        // 0xa (1) - and bit 46 is clear in every one. So no shipped program distinguishes "abs
        // on op2" from "repeat count 4..7", and a decoder that answered would be guessing.
        // BOTH directions are pinned: the decoder blocks such a word, this refuses to write one.
        assert!(
            dot(
                4,
                Dest::new(Bank::Temp, 4),
                [true; 4],
                Src::reg(Bank::PrimaryAttr, 2),
                Src::reg(Bank::Internal, 4).absolute(),
                0,
            )
            .is_err(),
            "an `abs` on the DOT's op2 sets the repeat count's middle bit - refuse it"
        );
        let plain = dot(
            4,
            Dest::new(Bank::Temp, 4),
            [true; 4],
            Src::reg(Bank::PrimaryAttr, 2),
            Src::reg(Bank::Internal, 4),
            0,
        )
        .unwrap();
        assert!(decode::decode(plain).blocked.is_none(), "the plain word is fine");
        let bit46 = plain | (1u64 << 46);
        assert_eq!(decode::repeat_extra_iterations(bit46), None, "bit 46 has no settled reading");
        assert!(
            decode::decode(bit46).blocked.is_some(),
            "a word whose bit 46 is set means two things, so the decoder must not pick one"
        );
    }

    /// The DOT's second operand is an internal register and NOTHING ELSE - its field is two bits
    /// naming `i0..i3`, so there is no bank selector to get wrong and no other bank to reach.
    #[test]
    fn a_dots_second_operand_refuses_every_bank_but_the_internal_one() {
        for bank in [Bank::Temp, Bank::Output, Bank::PrimaryAttr, Bank::SecondaryAttr] {
            assert!(
                dot(
                    4,
                    Dest::new(Bank::Temp, 4),
                    [true; 4],
                    Src::reg(Bank::PrimaryAttr, 2),
                    Src::reg(bank, 2),
                    0,
                )
                .is_err(),
                "{bank:?} has no encoding as a DOT's op2"
            );
        }
        // The four that DO exist, and their decoded base lanes - i0..i3 are four lanes apart.
        for n in 0..4u8 {
            let word = dot(
                4,
                Dest::new(Bank::Temp, 4),
                [true; 4],
                Src::reg(Bank::PrimaryAttr, 2),
                Src::reg(Bank::Internal, n * 4),
                0,
            )
            .unwrap_or_else(|e| panic!("i{n}: {e:?}"));
            let got = decode::decode(word);
            assert_eq!((got.srcs[1].bank, got.srcs[1].index), (Bank::Internal, n * 4));
        }
        // An internal register that is not a multiple of four is not an internal register.
        assert!(dot(
            4,
            Dest::new(Bank::Temp, 4),
            [true; 4],
            Src::reg(Bank::PrimaryAttr, 2),
            Src::reg(Bank::Internal, 2),
            0,
        )
        .is_err());
        // And no channel count other than three or four exists.
        for components in [0u8, 1, 2, 5] {
            assert!(dot(
                components,
                Dest::new(Bank::Temp, 4),
                [true; 4],
                Src::reg(Bank::PrimaryAttr, 2),
                Src::reg(Bank::Internal, 0),
                0,
            )
            .is_err());
        }
    }

    /// >>> THE REPEAT FORMS, ACROSS EVERY GROUP THIS ASSEMBLER CAN WRITE - which counts each
    /// >>> group can express, and which cannot express one at all.
    ///
    /// A repeat count is the single most expensive field in this ISA to read wrong, and it fails
    /// in the quiet direction: a count read as zero drops iterations, so the program computes a
    /// prefix of what it meant and writes plausible numbers into the lanes it did reach.
    /// `decode::repeat_extra_iterations`'s note records a title rendering a BLACK FRAME from
    /// exactly that on the DOT group.
    ///
    /// The corpus can only ever report which counts some compiler happened to emit. This asks
    /// each group for every count and prints what it answered, so a group that silently lost its
    /// field - or gained one where the bits are an operand - says so here.
    #[test]
    fn every_group_this_assembler_writes_reports_which_repeat_counts_it_can_express() {
        use PackFmt::*;
        let alu_word = alu(
            Op::Add,
            false,
            Dest::new(Bank::Temp, 4),
            [true; 4],
            Src::reg(Bank::PrimaryAttr, 2),
            Src::reg(Bank::SecondaryAttr, 8),
        )
        .unwrap();
        let pack_word = pack(
            Dest::new(Bank::Temp, 0),
            F16,
            Bank::PrimaryAttr,
            0,
            F32,
            [0, 1, 2, 3],
            [true; 4],
            false,
        )
        .unwrap();
        let vtst_word = vtst(
            TestAlu::Add,
            TestCmp::Lt,
            TestReduce::Channel(0),
            0,
            false,
            Bank::PrimaryAttr,
            0,
            Bank::PrimaryAttr,
            2,
        )
        .unwrap();
        let dot_word = dot(
            4,
            Dest::new(Bank::Output, 0),
            [true, false, false, false],
            Src::reg(Bank::SecondaryAttr, 0),
            Src::reg(Bank::Internal, 0),
            0,
        )
        .unwrap();

        let mut rows: Vec<(&str, Vec<u32>)> = Vec::new();
        for (name, word) in
            [("0x08 ALU", alu_word), ("0x40 PACK", pack_word), ("0x48 VTST", vtst_word), ("0x18 DOT", dot_word)]
        {
            let mut reached = Vec::new();
            for extra in 0..16u32 {
                let Ok(w) = with_repeat_count(word, extra) else { continue };
                // The count must READ BACK, and the instruction must still be the one asked for.
                assert_eq!(
                    decode::repeat_extra_iterations(w),
                    Some(extra),
                    "{name}: repeat {extra} did not read back from {w:#018x}"
                );
                let (a, b) = (decode::decode(word), decode::decode(w));
                assert_eq!(a.op, b.op, "{name}: setting a repeat changed the OPERATION");
                assert_eq!(a.srcs, b.srcs, "{name}: setting a repeat moved an OPERAND");
                assert_eq!(a.write_mask, b.write_mask, "{name}: setting a repeat changed the MASK");
                reached.push(extra);
            }
            println!("  repeat counts expressible - {name:<10} {reached:?}");
            rows.push((name, reached));
        }

        // >>> EVERY GROUP REACHES ZERO, because "runs once" is what an ordinary instruction is,
        // and a group that could not express it would mean this assembler has been writing words
        // whose count field says something it did not intend.
        for (name, reached) in &rows {
            assert!(reached.contains(&0), "{name} cannot express 'runs once'");
        }
        // And the groups that HAVE a count reach more than zero, while the ALU group - whose
        // bits 47:44 are `src2`'s swizzle, not a count - reaches nothing else. That asymmetry is
        // the whole point of sweeping rather than assuming, and it is asserted in BOTH
        // directions: a count silently appearing in group 0x08 would be an operand being
        // overwritten by something that thinks it is a repeat.
        let reached_of = |want: &str| {
            rows.iter().find(|(n, _)| *n == want).map(|(_, r)| r.clone()).unwrap_or_default()
        };
        assert_eq!(reached_of("0x08 ALU"), vec![0], "group 0x08's 47:44 is a swizzle, not a count");
        for group in ["0x40 PACK", "0x48 VTST", "0x18 DOT"] {
            assert!(
                reached_of(group).len() > 1,
                "{group} has a documented repeat count and expressed only {:?}",
                reached_of(group)
            );
        }
        // A repeat the group cannot reach is refused rather than silently truncated - the whole
        // failure mode this field has.
        assert!(
            with_repeat_count(alu_word, 3).is_err(),
            "group 0x08 has no repeat count, so asking for three iterations must be refused"
        );
    }

    #[test]
    fn the_fragment_pipeline_words_are_distinguishable_from_a_flow_no_op() {
        let k = kill(Predicate::Always).unwrap();
        let d = depthf(Bank::Temp, 0).unwrap();
        assert_ne!(k, d, "a kill and a depth write must not assemble to the same word");
        assert_eq!(decode::decode(k).op, Op::Kill);
        assert_eq!(decode::decode(d).op, Op::DepthF);
        // And a predicated kill is still a kill, not a branch - the group's ExtPredicate slot
        // is where a branch reads its condition, and a kill must not have written into it.
        assert_eq!(decode::decode(kill(Predicate::IfNotP(1)).unwrap()).op, Op::Kill);
    }

    /// The families whose `(alu_sel, alu_op)` numbering the decoder marks INFERRED are refused
    /// rather than guessed at - the assembler may not assert a fact this project does not hold.
    #[test]
    fn a_vtst_in_an_unestablished_alu_family_is_refused() {
        for alu in [TestAlu::BitAnd, TestAlu::BitShl, TestAlu::Fx8Sub, TestAlu::IntSub, TestAlu::IntSub16U] {
            assert!(
                vtst(alu, TestCmp::Ne, TestReduce::Channel(0), 0, false, Bank::SecondaryAttr, 0, Bank::SecondaryAttr, 2)
                    .is_err(),
                "{alu:?} must be refused, not encoded on a guess"
            );
        }
    }

    /// An ODD source register cannot be named by a float VTST - the field is double-register
    /// scaled - so it is refused rather than silently rounded to its neighbour.
    #[test]
    fn a_vtst_refuses_an_odd_source_register() {
        assert!(
            vtst(TestAlu::Sub, TestCmp::Ge, TestReduce::Channel(0), 0, true, Bank::SecondaryAttr, 3, Bank::SecondaryAttr, 0)
                .is_err(),
            "an odd register is not expressible and must not be rounded"
        );
    }
}
