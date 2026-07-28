//! USSE (SGX543) instruction decoder.
//!
//! The bitfield tables below are transcribed verbatim from psvgxp `grammar.json` (an
//! independent RE of the encoding). They are MSB-first: the first `(name, width)` entry
//! of each word occupies the high bits. Each 32-bit word's widths sum to 32. This is the
//! *structural* decode and it is entirely fact-driven - `field(word, TABLE, "op0")`
//! returns exactly the bits the hardware uses.
//!
//! The *operation* layer is fact-driven from the public SGX543 USSE instruction encoding:
//! each group's sub-opcode, operand banks, swizzle/mask tables, and semantics are known
//! facts about the hardware. An operation that is not yet wired for emit stays classified
//! but is a HARD FAILURE at emit time, not a silent fallback - the error names the exact
//! opcode to implement next (an opcode grind, like the NID grind). Anything whose exact
//! operand layout is not established is flagged `blocked` so it hard-fails rather than
//! risk an inexact translation. No guess is ever emitted, so a wrong translation can never
//! paint a pixel.

use crate::ir::{Bank, CompareMethod, Instr, Op, Operand, Predicate, TestAlu, TestCmp, TestReduce, TexLod};

/// A field descriptor: name and bit width, MSB-first within its 32-bit word.
type Field = (&'static str, u8);

/// The eight top-level opcode groups, keyed by `opcode1` (high-word bits 63:59). Only
/// the groups psvgxp documented (and which appear in real blobs) have tables; the rest
/// decode to `Unsupported` with their group recorded.
pub struct GroupTable {
    pub name: &'static str,
    pub high: &'static [Field],
    pub low: &'static [Field],
}

// ---- Group 0x00 (high word 0x00000000..0x08000000) ----
const G00_HIGH: &[Field] = &[
    ("opcode1", 5), ("data_format", 1), ("predicate", 2), ("unk4", 1), ("unk3", 1),
    ("swz_alt_op1", 1), ("unk2", 1), ("alt_opt0", 1), ("abs_op1", 1), ("alt_opt2", 1),
    ("alt_opt3", 1), ("swz_alt_op3", 1), ("op3_swz", 2), ("swz_alt_op2", 1), ("unk1", 1),
    ("unk0", 1), ("swz_mask16", 1), ("swz_mask32", 1), ("swz_en", 1), ("abs_op2", 1),
    ("neg_op2", 1), ("abs_op3", 1), ("neg_op3", 1), ("opt1", 1), ("opt0", 2),
];
const G00_LOW: &[Field] = &[
    ("opt2", 2), ("opt3", 2), ("op0", 6), ("op2_swz", 2), ("op1_swz", 2),
    ("op1", 6), ("op2", 6), ("op3", 6),
];

// ---- Group 0x08 (== 0x10 layout) ----
const G08_HIGH: &[Field] = &[
    ("opcode1", 5), ("predicate", 3), ("unk2", 1), ("op1_swz_c3x", 2), ("unk1", 1),
    ("alt_opt0", 1), ("op1_swz_c30", 1), ("alt_opt1", 1), ("alt_opt2", 1),
    ("swz_alt_op2", 2), ("op2_swz", 2), ("unk0", 1), ("swz_mask3", 1), ("swz_mask2", 1),
    ("swz_mask1", 1), ("swz_en", 1), ("abs_op1", 1), ("neg_op1", 1), ("abs_op2", 1),
    ("op1_swz_c2x", 2), ("opt0", 2),
];
const G08_LOW: &[Field] = &[
    ("opt1", 2), ("opt2", 2), ("op0", 6), ("op1_swz_c20", 1), ("op1_swz_c1", 3),
    ("op1_swz_c0", 3), ("opcode2", 3), ("op1", 6), ("op2", 6),
];

// ---- Group 0x18 DOT ----
const G18_DOT_HIGH: &[Field] = &[
    ("opcode1", 5), ("predicate", 3), ("unk12", 1), ("unk11", 1), ("opcode2", 1),
    ("c3_en", 1), ("alt_opt0", 1), ("unk9", 1), ("alt_opt1", 1), ("unk8", 1), ("unk7", 1),
    ("abs_op2", 1), ("swz_en_strange1", 1), ("swz_en_strange0", 1), ("unk6", 1),
    ("swz_mask3", 1), ("swz_mask2", 1), ("swz_mask1", 1), ("swz_en", 1), ("neg_op1", 1),
    ("abs_op1", 1), ("unk5", 1), ("unk4", 1), ("unk3", 1), ("opt0", 2),
];
const G18_DOT_LOW: &[Field] = &[
    ("opt1", 2), ("op2i", 2), ("op0", 6), ("swz_alt_op2", 2), ("op2_swz", 2),
    ("op1_swz_c3", 3), ("op1_swz_c2", 3), ("op1_swz_c1", 3), ("op1_swz_c0", 3), ("op1", 6),
];

// ---- Group 0x18 MAD ----
const G18_MAD_HIGH: &[Field] = &[
    ("opcode1", 5), ("predicate", 3), ("unk6", 1), ("swz_alt_op3_2", 1), ("opcode2", 1),
    ("unk5", 1), ("alt_opt0", 1), ("unk4", 1), ("alt_opt1", 1), ("unk3", 1), ("unk2", 1),
    ("abs_op2", 1), ("op0_strange1", 1), ("op0_strange0", 1), ("unk1", 1), ("swz_mask3", 1),
    ("swz_mask2", 1), ("swz_mask1", 1), ("swz_en", 1), ("neg_op1", 1), ("abs_op1", 1),
    ("neg_op3", 1), ("abs_op3", 1), ("swz_alt_op2_2", 1), ("opt0", 2),
];
const G18_MAD_LOW: &[Field] = &[
    ("opt1", 2), ("op2i", 2), ("op0", 6), ("swz_alt_op2_x", 2), ("op2_swz", 2),
    ("swz_alt_op3_x", 2), ("op3_swz", 2), ("op3i", 2), ("unk0", 1), ("swz_alt_op1", 3),
    ("op1_swz", 2), ("op1", 6),
];

// ---- Group 0x30 (LD / SMP / special) ----
const G30_HIGH: &[Field] = &[
    ("opcode1", 5), ("predicate", 3), ("unk9", 1), ("data_format", 2), ("unk8", 1),
    ("alt_opt0", 1), ("unk7", 1), ("alt_opt1", 1), ("unk6", 1), ("unk5", 5), ("opcode2", 2),
    ("modifier1", 1), ("modifier0", 1), ("abs_op1", 1), ("neg_op1", 1), ("mask2_op1", 1),
    ("mask1_op1", 1), ("unk4", 1), ("opt0", 2),
];
const G30_LOW: &[Field] = &[
    ("opt1", 2), ("unk3", 2), ("op0", 7), ("unk2", 5), ("unk1", 2), ("op1", 7),
    ("unk0", 5), ("mask1_op0", 1), ("swz_en", 1),
];

// ---- Group 0x38 (branch / flow / emit) ----
const G38_HIGH: &[Field] = &[
    ("opcode1", 5), ("predicate", 3), ("unk3", 1), ("cond0", 1), ("unk2", 2), ("alt_opt0", 1),
    ("unk1", 1), ("alt_opt2", 1), ("alt_opt3", 1), ("opcode2", 2), ("unk0", 3),
    ("data_format", 3), ("cond1", 1), ("op23_swz", 3), ("opt0", 4),
];
const G38_LOW: &[Field] = &[
    ("opt2", 2), ("opt3", 2), ("swz_mask3", 1), ("swz_mask2", 1), ("swz_mask1", 1),
    ("swz_en", 1), ("op0", 6), ("op1", 6), ("op2", 6), ("op3", 6),
];

/// The transcribed bitfield tables for the documented groups, as (name, high, low).
/// Groups 0x08/0x10 (`grp08_alu`) are fully wired into [`decode`]; the rest are kept as
/// validated ISA facts to wire operand decode for next (0x00 mad, 0x18 dot, 0x30
/// transcendentals, 0x38 mov). Referenced here so they stay live + tested.
pub const GROUP_TABLES: &[(&str, &[Field], &[Field])] = &[
    ("grp00_mad", G00_HIGH, G00_LOW),
    ("grp08_alu", G08_HIGH, G08_LOW),
    ("grp18_dot", G18_DOT_HIGH, G18_DOT_LOW),
    ("grp18_mad", G18_MAD_HIGH, G18_MAD_LOW),
    ("grp30", G30_HIGH, G30_LOW),
    ("grp38", G38_HIGH, G38_LOW),
];

/// Extract a named field from a 32-bit word given its MSB-first table. Returns 0 if the
/// field is absent (callers treat absence as "not encoded in this group").
pub fn field(word: u32, table: &[Field], name: &str) -> u32 {
    let mut pos = 32u8; // bit just past the MSB of the next field
    for &(fname, width) in table {
        pos -= width;
        if fname == name {
            let mask = if width == 32 { u32::MAX } else { (1u32 << width) - 1 };
            return (word >> pos) & mask;
        }
    }
    0
}

/// The high and low 32-bit halves of a 64-bit instruction word.
#[inline]
fn halves(word: u64) -> (u32, u32) {
    ((word >> 32) as u32, word as u32)
}

/// `opcode1` (top-level group) of an instruction word - a fact for every instruction.
#[inline]
pub fn opcode1(word: u64) -> u8 {
    ((word >> 59) & 0x1f) as u8
}

/// Extract the inclusive bit range `[msb..=lsb]` of a 64-bit instruction word as an
/// unsigned value. Bit 63 is the MSB. Used to decode the groups whose fields are given as
/// absolute bit positions in the ISA reference (0x30/0x38/0x40/0x50/0xE0/0xF8) rather than
/// as the MSB-first per-half tables the earlier groups use - direct extraction is less
/// error-prone than reconstructing contiguous field tables with filler for these.
#[inline]
pub fn bits(word: u64, msb: u32, lsb: u32) -> u32 {
    debug_assert!(msb >= lsb && msb < 64);
    let width = msb - lsb + 1;
    let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
    ((word >> lsb) & mask) as u32
}

/// The "vec4 standard" table-indexed swizzle (a 4-bit index selects a whole 4-channel
/// pattern), in the [`Operand::swizzle`] selector encoding (0..3 = x,y,z,w; 5 = const 1.0).
/// Used by VMOV (0x38) double-register source operands. Entry 15 sets channel 3 to the
/// constant 1.0. These are ISA facts (the SGX543 register swizzle table).
const VEC4_STD_SWIZZLE: [[u8; 4]; 16] = [
    [0, 0, 0, 0], // XXXX
    [1, 1, 1, 1], // YYYY
    [2, 2, 2, 2], // ZZZZ
    [3, 3, 3, 3], // WWWW
    [0, 1, 2, 3], // XYZW
    [1, 2, 3, 3], // YZWW
    [0, 1, 2, 2], // XYZZ
    [0, 0, 1, 2], // XXYZ
    [0, 1, 0, 1], // XYXY
    [0, 1, 3, 2], // XYWZ
    [2, 0, 1, 3], // ZXYW
    [2, 3, 2, 3], // ZWZW
    [1, 2, 0, 2], // YZXZ
    [0, 0, 1, 1], // XXYY
    [0, 2, 3, 3], // XZWW
    [0, 1, 2, 5], // XYZ1  (channel 3 = constant 1.0)
];

/// A source modifier field (2 bits): 0 none, 1 negate, 2 absolute, 3 negate+absolute. A
/// fact shared across groups. Returns `(abs, neg)`.
#[inline]
fn src_mod2(field_val: u32) -> (bool, bool) {
    match field_val & 3 {
        0 => (false, false),
        1 => (false, true),
        2 => (true, false),
        _ => (true, true),
    }
}

/// A direct 4-bit destination write mask (bit i => channel i written).
#[inline]
fn write_mask4(field_val: u32) -> [bool; 4] {
    [field_val & 1 != 0, field_val & 2 != 0, field_val & 4 != 0, field_val & 8 != 0]
}

// The spec's A.6 destination write-mask transform (F16 mask bits expanding to channel pairs,
// F32 masks truncating to the low two channels) is deliberately NOT applied to the 0x08/0x10
// vector-ALU masks. MEASURED 2026-07-24: wiring it in changes no whole-corpus dataflow
// corroboration (temp 195/265, output 5/8, internal 660/663 either way), so there is no positive
// evidence for it here, and group-0x38 moves already decode within the two-channel form the
// transform would produce - suggesting the tables this decoder uses encode it where it applies.
// It was once suspected of causing the vertex-to-fragment varying mismatch; that turned out to
// be a linker-model bug (the stages are matched by USAGE, see `link::plan_varyings`) and the
// masks were never implicated. Do NOT wire it in to make one shader's numbers match.

/// Decode one 64-bit USSE instruction into the IR: classify the operation (a fact from
/// the henkaku SGX543 opcode map) and, for the groups whose operand encoding is exactly
/// wired, decode operands / banks / swizzles / write-mask. When the operation is known
/// but this specific instruction carries a feature not yet wired EXACTLY (an exotic
/// operand mode, a predicate, an unimplemented group's operands), `blocked` is set so the
/// emitter hard-fails naming it rather than emit an inexact translation.
pub fn decode(word: u64) -> Instr {
    let op1 = opcode1(word);
    let (hi, lo) = halves(word);
    match op1 {
        0x00 => decode_grp_mad(word, hi, lo),
        0x01 | 0x02 => decode_grp_alu(word, hi, lo, op1),
        0x03 => decode_grp_18(word, hi, lo),
        0x06 => decode_grp_30(word, hi, lo),
        0x07 => decode_grp_38(word, hi, lo),
        0x08 => decode_grp_pack(word),
        0x09 => decode_grp_test(word),
        0x0a | 0x0b | 0x0c | 0x0d => decode_grp_bitwise(word, op1),
        0x1c => decode_grp_tex(word),
        0x1f => decode_grp_flow(word),
        _ => classified_stub(word, op1, hi, lo),
    }
}


/// Decode a group-0x48 VTST (test -> predicate). The field layout is a fact from the SGX543
/// ISA reference's TEST group, cross-checked between a field-level decoder layout and the
/// hardware bit-definition macros (see the distilled TEST-group spec, section T.1):
///
/// | field | bits | | field | bits |
/// |---|---|---|---|---|
/// | `pred` | 58:56 | | `chan_cc` | 38:36 |
/// | `src1_neg` | 50 | | `pdst_n` | 35:34 |
/// | `src1_ext` / `src2_ext` | 49 / 48 | | `dest_bank` | 33:32 |
/// | `prec` | 47 | | `src1_bank` / `src2_bank` | 31:30 / 29:28 |
/// | `sign_test` | 43:42 | | `dest_n` | 27:21 |
/// | `zero_test` | 41:40 | | `test_wben` | 20 |
/// | `test_crcomb_and` | 39 | | `alu_sel` / `alu_op` | 19:18 / 17:14 |
/// | | | | `src1_n` / `src2_n` | 13:7 / 6:0 |
///
/// The instruction computes `r = alu(src1', src2)` per channel (with `src1'` negated when
/// `src1_neg`), forms a per-channel boolean by testing `r` against ZERO, reduces the four
/// booleans per `chan_cc`, and writes the single bit to predicate `p[pdst_n]`. With
/// `test_wben` it also writes `r` to the destination register.
///
/// The three test fields collapse to one comparison here: `sign_test` (0 none, 1 negative,
/// 2 positive) and `zero_test` (0 none, 1 zero, 2 notzero) combined by `test_crcomb_and`
/// (1 = AND, 0 = OR) enumerate exactly the six relations of [`TestCmp`]. Combinations that do
/// not (both sub-tests disabled, a reserved value, or a contradictory pair) are BLOCKED rather
/// than approximated.
///
/// Operands use the shared decode with the extension row ([`ext_source`]); float ALU types are
/// double-register scaled and the bitwise family is not, exactly as elsewhere.
fn decode_grp_test(word: u64) -> Instr {
    let mut blocked: Option<&'static str> = None;
    let predicate_raw = bits(word, 58, 56);
    let pred = ext_predicate(predicate_raw);
    if matches!(pred, Predicate::Raw(_)) {
        blocked = blocked.or(Some("PN (per-instance) predicate depends on repeat state - not modeled"));
    }

    // The ALU whose result is tested: family from `alu_sel`, width from `prec`, operation
    // from `alu_op`. Only FLOAT (both widths) and the BITWISE AND are modelled - the integer
    // families' op numbering is not established, so they block instead of guessing.
    let alu_sel = bits(word, 19, 18);
    let alu_op = bits(word, 17, 14);
    let prec = bits(word, 47, 47);
    let (alu, half_precision, float_ty) = match (alu_sel, alu_op) {
        (0, 2) => (TestAlu::Add, prec == 0, true),
        (0, 13) => (TestAlu::Mul, prec == 0, true),
        (0, 14) => (TestAlu::Sub, prec == 0, true),
        (3, 0) => (TestAlu::BitAnd, false, false),
        _ => {
            blocked = blocked.or(Some("0x48 VTST ALU family/op not modeled"));
            (TestAlu::Sub, prec == 0, true)
        }
    };

    // sign_test x zero_test x combiner -> one relation against zero.
    let cmp = match (bits(word, 43, 42), bits(word, 41, 40), bits(word, 39, 39)) {
        (0, 1, _) => TestCmp::Eq,
        (0, 2, _) => TestCmp::Ne,
        (1, 0, _) => TestCmp::Lt,
        (2, 0, _) => TestCmp::Gt,
        (1, 1, 0) => TestCmp::Le,
        (2, 1, 0) => TestCmp::Ge,
        (1, 2, 1) => TestCmp::Lt, // negative AND non-zero is just negative
        (2, 2, 1) => TestCmp::Gt,
        _ => {
            blocked = blocked.or(Some("0x48 VTST sign/zero sub-test combination not modeled"));
            TestCmp::Ne
        }
    };

    let reduce = match bits(word, 38, 36) {
        c @ 0..=3 => TestReduce::Channel(c as u8),
        4 => TestReduce::AndAll,
        5 => TestReduce::OrAll,
        _ => {
            // 6/7 are the per-channel / paired forms, whose vector meaning is not confirmed.
            blocked = blocked.or(Some("0x48 VTST chan_cc per-channel/paired reduction not modeled"));
            TestReduce::Channel(0)
        }
    };
    let pdst = bits(word, 35, 34) as u8;
    let write_back = bits(word, 20, 20) != 0;

    // Sources. A float ALU type is double-register scaled; the bitwise family is not, and the
    // SPECIAL/IMMEDIATE extension banks are never scaled either.
    let mut source = |bank_sel: u8, field_val: u32, ext: bool| -> Operand {
        if ext {
            // The IMMEDIATE row is assembled per group, so the shared [`ext_source`] leaves it
            // to the caller. The TEST layout carries no extended-immediate assembly fields, so
            // the operand's own 7-bit number IS the literal (spec T.5b step 6) - the form the
            // corpus's flag-bit tests use.
            if bank_sel & 3 == 2 {
                return Operand::plain(Bank::Immediate, (field_val & 0x7f) as u8, bank_sel);
            }
            match ext_source(bank_sel, field_val) {
                Ok(o) => return o,
                Err(why) => {
                    blocked = blocked.or(Some(why));
                    return Operand::plain(Bank::Temp, 0, bank_sel);
                }
            }
        }
        let (bank, index) = if float_ty {
            source_bank_index(bank_sel, field_val, 124, |v| (v as u8).wrapping_mul(2))
        } else {
            r7_source_bank_index(bank_sel, field_val)
        };
        Operand::plain(bank, index, bank_sel)
    };
    let mut s1 = source(bits(word, 31, 30) as u8, bits(word, 13, 7), bits(word, 49, 49) != 0);
    s1.neg = bits(word, 50, 50) != 0;
    let mut s2 = source(bits(word, 29, 28) as u8, bits(word, 6, 0), bits(word, 48, 48) != 0);
    // `src2_vscomp` (bit 46) consumes src2 as a BROADCAST SCALAR rather than as a vector.
    // The broadcast channel is 0, corroborated by the parameter table on the real alpha test:
    // its src2 is `Alphathreshold`, a ONE-component F16 uniform, so the only channel that
    // holds it is 0 - while the reduction picks channel 3 (alpha) on the src1 side. Reading
    // src2 per channel instead lands on the next register entirely, which makes the threshold
    // whatever happens to follow the uniform and the cutout stop matching the texture.
    if bits(word, 46, 46) != 0 {
        s2.swizzle = [0; 4];
    }

    // The general-register destination is only written when `test_wben` is set; otherwise the
    // dest fields are inert and the instruction is predicate-only.
    let dest = if write_back {
        let dest_sel = bits(word, 33, 32) as u8;
        match r7_dest_bank_index(dest_sel, bits(word, 27, 21)) {
            Some((b, i)) => Some(Operand::plain(b, i, dest_sel)),
            None => {
                blocked = blocked.or(Some("0x48 VTST write-back destination in index mode"));
                None
            }
        }
    } else {
        None
    };

    Instr {
        op: Op::Test { alu, cmp, reduce, pdst, write_back },
        pred,
        dest,
        write_mask: [write_back, write_back, write_back, write_back],
        srcs: vec![s1, s2],
        half_precision,
        raw: word,
        group: 0x09,
        blocked,
    }
}

/// Bank from the 2-bit RS2 selector (henkaku, FACT): 0=r(temp) 1=o(output) 2=pa 3=sa.
fn bank_rs2(sel: u8) -> Bank {
    match sel & 3 {
        0 => Bank::Temp,
        1 => Bank::Output,
        2 => Bank::PrimaryAttr,
        _ => Bank::SecondaryAttr,
    }
}

/// Bank from the 2-bit RSI2 selector used by destination op0 (henkaku, FACT): 0=r 1=o
/// 2=pa 3=index-mode. Returns `None` for index mode (an exotic addressing mode this
/// decoder does not yet resolve - the caller blocks emit).
fn bank_rsi2(sel: u8) -> Option<Bank> {
    match sel & 3 {
        0 => Some(Bank::Temp),
        1 => Some(Bank::Output),
        2 => Some(Bank::PrimaryAttr),
        _ => None,
    }
}

/// Register index from a SIX-bit (R6) register field value: `index = value * 2` (henkaku,
/// FACT). A 6-bit field can only name 64 values, so it addresses the 128-register bank in
/// double-register units - which is why an F32 vec4 at field `f` occupies registers
/// `2f..2f+3`. Do NOT use this for a SEVEN-bit (R7) field: see [`r7_reg_index`].
#[inline]
fn reg_index(field_val: u32) -> u8 {
    (field_val as u8).wrapping_mul(2)
}

/// Register index from a SEVEN-bit (R7) register field value: the field IS the register
/// number, with no double-register scaling.
///
/// ESTABLISHED (not assumed) from three independent facts, each of which the doubling rule
/// contradicts:
/// 1. A 7-bit field spans 0..127 - exactly the 128-register bank. Doubling it would address
///    0..254, off the end of the file. The reserved internal-register encodings sit at
///    124..127 (the top FOUR of 128), and this decoder already tests the raw R7 field against
///    that range - a test that only means anything for a direct register number.
/// 2. Corpus check on a real vertex blob whose `primary_reg_count` is 20: its two group-0x50
///    bitwise moves carry R7 source fields 18 and 19. Direct, they read the last two
///    components of the declared `VertexColour1@pa16x4` attribute; doubled, they read pa36 and
///    pa38 - registers the program does not allocate.
/// 3. Same two instructions' R7 destination fields are 8 and 9. Direct, they fill output lanes
///    8 and 9, closing the only hole in that program's written output lanes and making the
///    total match the varyings block's own output-lane count exactly; doubled, they land on
///    lanes 16 and 18, which other instructions already write, and leave lane 9 unwritten.
#[inline]
fn r7_reg_index(field_val: u32) -> u8 {
    field_val as u8
}

/// An internal register i0..i3 is 128-bit = 4 consecutive 32-bit lanes; the emitter's
/// `i[]` array is laid out as `i[n*4 + lane]`. This maps an internal-register number
/// (0..3) to its scalar base index.
#[inline]
fn internal_base(n: u32) -> u8 {
    ((n & 3) * 4) as u8
}

/// Decode a SOURCE operand's bank + scalar base index from an RS2 selector and a register
/// field, detecting the reserved internal-register encodings. Per henkaku "Register
/// Encoding - r", the top 4 index values of the `r` bank are reserved to select internal
/// registers i0..i3 rather than temporaries: `reserved_lo` is 60 for a 6-bit field (R6)
/// and 124 for a 7-bit field (R7). Other banks (o/pa/sa) have no reserved range.
fn source_bank_index(
    bank_sel: u8,
    field_val: u32,
    reserved_lo: u32,
    index: fn(u32) -> u8,
) -> (Bank, u8) {
    let bank = bank_rs2(bank_sel);
    if matches!(bank, Bank::Temp) && field_val >= reserved_lo && field_val <= reserved_lo + 3 {
        (Bank::Internal, internal_base(field_val - reserved_lo))
    } else {
        (bank, index(field_val))
    }
}

/// R6 (6-bit) source operand: double-register field, reserved internal range 60..63.
fn r6_source_bank_index(bank_sel: u8, field_val: u32) -> (Bank, u8) {
    source_bank_index(bank_sel, field_val, 60, reg_index)
}

/// R7 (7-bit) source operand: DIRECT register number ([`r7_reg_index`]), reserved internal
/// range 124..127.
fn r7_source_bank_index(bank_sel: u8, field_val: u32) -> (Bank, u8) {
    source_bank_index(bank_sel, field_val, 124, r7_reg_index)
}

/// An R6 source operand in its plain (non-`alt_opt`) addressing mode, with no swizzle or
/// modifiers applied yet. Goes through [`r6_source_bank_index`] so a field in the reserved
/// 60..63 range names the INTERNAL register it selects rather than temporaries r120..r127 -
/// registers no program allocates, which a shader would then read as zero.
fn r6_plain_source(bank_sel: u8, field_val: u32) -> Operand {
    let (bank, index) = r6_source_bank_index(bank_sel, field_val);
    Operand::plain(bank, index, bank_sel)
}

/// Resolve a source operand whose `alt_opt<N>` bit is set (an exotic addressing mode). Per
/// the henkaku "Operand N" table, `(alt_opt=1, opt)` selects: 00 index1 (RIO6), 01 constant
/// (CNST6), 10 immediate (IMM6), 11 index2 (RIO6). Only the CONSTANT sub-mode is resolvable
/// from clean facts (the CNST6 table), so it yields an inline constant operand; the index/
/// immediate modes need the RIO6/IMM6 addressing this decoder does not model, so they return
/// `None` and the caller blocks emit. `op_field` is the operand's register/value field.
fn exotic_source(opt_sel: u8, op_field: u32, swizzle: [u8; 4], abs: bool, neg: bool) -> Option<Operand> {
    if opt_sel & 3 == 0b01 {
        // Constant mode: the 6-bit field is the CNST6 selector. The operand's swizzle is
        // still live - per spec A.7 it chooses which hardware constant BANK each channel
        // reads (F32: bank 1 for a Y selector, else bank 0; F16: one of four banks by
        // selector) - so it must be carried through, not dropped.
        let mut o = Operand::plain(Bank::Constant, (op_field & 0x3f) as u8, opt_sel);
        o.swizzle = swizzle;
        o.abs = abs;
        o.neg = neg;
        Some(o)
    } else {
        None
    }
}

/// Resolve a SOURCE operand whose bank-EXTENSION bit is set, for the groups that carry a
/// separate `src*_ext` bit next to the 2-bit bank selector (the shared operand decode, spec
/// Core A.2 / TEST T.5b). The extension row is: 0 = INDEXED1, 1 = SPECIAL, 2 = IMMEDIATE,
/// 3 = INDEXED2.
///
/// SPECIAL then splits on bit `0x40` of the RAW register field: set selects the GLOBAL
/// (special hardware register) bank, clear selects FPCONSTANT - a hardware constant-table
/// lookup, the same CNST6 table [`exotic_source`] reads. Neither SPECIAL nor IMMEDIATE is ever
/// double-register scaled.
///
/// The FPCONSTANT and GLOBAL cases yield an operand - both are structural decodes, and which
/// register a GLOBAL operand names is as much a fact as which constant an FPCONSTANT one does.
/// What a GLOBAL register CONTAINS is a separate question, and it is settled (or hard-failed,
/// per index) by the emitter, not here. The remaining rows return `Err(reason)`: the two
/// indexed modes need RIO6 addressing, and IMMEDIATE is assembled from group-specific fields so
/// the groups that allow it resolve it themselves before reaching here. The caller blocks emit
/// naming the reason rather than substituting a guess.
fn ext_source(bank_sel: u8, field_val: u32) -> Result<Operand, &'static str> {
    match bank_sel & 3 {
        1 if field_val & 0x40 == 0 => {
            Ok(Operand::plain(Bank::Constant, (field_val & 0x3f) as u8, bank_sel))
        }
        1 => Ok(Operand::plain(Bank::Global, (field_val & 0x3f) as u8, bank_sel)),
        2 => Err("extended bank IMMEDIATE not modeled for this group"),
        _ => Err("extended bank INDEXED (RIO6 addressing) not modeled"),
    }
}

/// Resolve the 3-bit **ExtPredicate** field (bits 58:56) that gates whether an instruction
/// executes. Table (spec Part D): 0 NONE, 1 P0, 2 P1, 3 P2, 4 P3, 5 NEGP0, 6 NEGP1, 7 PN.
///
/// Value 7 (PN, the per-instance / negated-last form) depends on repeat state this decoder
/// does not model, so it stays [`Predicate::Raw`] and the caller blocks emit.
///
/// Used by the groups whose predicate field is the full ExtPredicate: 0x30, 0x38, 0x40, 0x48,
/// 0x50, 0xE0. The vector-ALU groups use a DIFFERENT table - see [`ext_vec_predicate`].
fn ext_predicate(raw: u32) -> Predicate {
    match raw {
        0 => Predicate::Always,
        1..=4 => Predicate::IfP((raw - 1) as u8),
        5 => Predicate::IfNotP(0),
        6 => Predicate::IfNotP(1),
        _ => Predicate::Raw(raw as u8),
    }
}

/// Resolve the 3-bit **ExtVecPredicate** field used by the vector-ALU groups (0x00 VMAD,
/// 0x08/0x10 V32NMAD/V16NMAD, 0x18 VDP). Table (spec Part D): 0 NONE, 1 P0, 2 P1, 3 P2,
/// 4 NEGP0, 5 NEGP1, 6 NEGP2, 7 PN.
///
/// It differs from [`ext_predicate`] exactly at 4..6 - there is no P3, and 4/5/6 are the
/// NEGATED P0/P1/P2. Reading one table for the other silently inverts a condition, so the
/// group decides which to call. PN (7) stays raw and blocks.
fn ext_vec_predicate(raw: u32) -> Predicate {
    match raw {
        0 => Predicate::Always,
        1..=3 => Predicate::IfP((raw - 1) as u8),
        4..=6 => Predicate::IfNotP((raw - 4) as u8),
        _ => Predicate::Raw(raw as u8),
    }
}

/// Resolve the 2-bit **ShortPredicate** field, the narrow form group 0x00 (VMAD) carries:
/// 0 NONE, 1 P0, 2 P1, 3 NEGP0. Every encoding is resolvable, so this never blocks.
fn short_predicate(raw: u32) -> Predicate {
    match raw & 3 {
        0 => Predicate::Always,
        1 => Predicate::IfP(0),
        2 => Predicate::IfP(1),
        _ => Predicate::IfNotP(0),
    }
}

/// Decode an R6 DESTINATION operand (op0) bank + scalar base index via RSI2, detecting
/// the reserved internal-register range for the `r` bank. Returns `None` for RSI2
/// index-mode (an exotic addressing mode the caller must block).
fn r6_dest_bank_index(bank_sel: u8, field_val: u32) -> Option<(Bank, u8)> {
    match bank_rsi2(bank_sel)? {
        Bank::Temp if (60..=63).contains(&field_val) => {
            Some((Bank::Internal, internal_base(field_val - 60)))
        }
        b => Some((b, reg_index(field_val))),
    }
}

/// Decode an R7 DESTINATION operand (the `bits(27,21)` field shared by groups 0x30/0x40/0x50)
/// bank + register index via RSI2. The field is a DIRECT register number ([`r7_reg_index`])
/// and its top four values name the internal registers. Returns `None` for RSI2 index-mode
/// (an exotic addressing mode the caller must block).
fn r7_dest_bank_index(bank_sel: u8, field_val: u32) -> Option<(Bank, u8)> {
    match bank_rsi2(bank_sel)? {
        Bank::Temp if (124..=127).contains(&field_val) => {
            Some((Bank::Internal, internal_base(field_val - 124)))
        }
        b => Some((b, r7_reg_index(field_val))),
    }
}

/// opcode2 (3 bits) -> the operation, for groups 0x08 (f32) / 0x10 (f16). Order per the
/// henkaku map: 0 mul, 1 add, 2 frc, 3 dsx, 4 dsy, 5 min, 6 max, 7 dot.
fn alu_op(opcode2: u32) -> Op {
    match opcode2 & 7 {
        0 => Op::Mul,
        1 => Op::Add,
        2 => Op::Frc,
        3 => Op::Dsx,
        4 => Op::Dsy,
        5 => Op::Min,
        6 => Op::Max,
        _ => Op::Dot { components: 4 },
    }
}

/// The destination write mask for the 0x08/0x10 groups, from the henkaku masking truth
/// table indexed by (swz_mask3, swz_mask2, swz_mask1, swz_en). `x` (masked) reads as
/// not-written.
fn mask_table_08(m3: u32, m2: u32, m1: u32, en: u32) -> [bool; 4] {
    // 16 rows, MSB-first index = m3 m2 m1 en. Values are (ch0,ch1,ch2,ch3) written flags.
    const T: [[bool; 4]; 16] = [
        [false, false, false, false], // 0000
        [true, false, false, false],  // 0001
        [false, true, false, false],  // 0010 (x100)
        [true, true, false, false],   // 0011
        [false, false, true, false],  // 0100 (xx10)
        [true, false, true, false],   // 0101 (1x10)
        [false, true, true, false],   // 0110 (x110)
        [true, true, true, false],    // 0111
        [false, false, false, true],  // 1000 (xxx1)
        [true, false, false, true],   // 1001 (1xx1)
        [false, true, false, true],   // 1010 (x1x1)
        [true, true, false, true],    // 1011 (11x1)
        [false, false, true, true],   // 1100 (xx11)
        [true, false, true, true],    // 1101 (1x11)
        [false, true, true, true],    // 1110 (x111)
        [true, true, true, true],     // 1111
    ];
    let idx = ((m3 & 1) << 3) | ((m2 & 1) << 2) | ((m1 & 1) << 1) | (en & 1);
    T[idx as usize]
}

/// The operand-2 RSWZ2 swizzle for the 0x08/0x10 groups, from the henkaku table indexed
/// by (swz_alt_op2 << 2 | op2_swz). Each entry is four component selectors in the
/// [`Operand::swizzle`] encoding (0..3 = x,y,z,w; 5 = constant 1.0).
fn rswz2_op2(swz_alt_op2: u32, op2_swz: u32) -> [u8; 4] {
    // x=0 y=1 z=2 w=3 ; '1' constant -> 5 (matches RSWZ3 constant-1 selector).
    const T: [[u8; 4]; 16] = [
        [0, 0, 0, 0], // xxxx
        [1, 1, 1, 1], // yyyy
        [2, 2, 2, 2], // zzzz
        [3, 3, 3, 3], // wwww
        [0, 1, 2, 3], // xyzw
        [1, 2, 3, 3], // yzww
        [0, 1, 2, 2], // xyzz
        [0, 0, 1, 2], // xxyz
        [0, 1, 0, 1], // xyxy
        [0, 1, 3, 2], // xywz
        [2, 0, 1, 3], // zxyw
        [2, 3, 2, 3], // zwzw
        [1, 2, 0, 2], // yzxz
        [0, 0, 1, 1], // xxyy
        [0, 2, 3, 3], // xzww
        [0, 1, 2, 5], // xyz1
    ];
    let idx = ((swz_alt_op2 & 3) << 2) | (op2_swz & 3);
    T[idx as usize]
}

/// Decode a groups-0x08/0x10 ALU instruction (mul/add/frc/dsx/dsy/min/max/dot) exactly.
fn decode_grp_alu(word: u64, hi: u32, lo: u32, op1: u8) -> Instr {
    let high = G08_HIGH;
    let low = G08_LOW;
    let op = alu_op(field(lo, low, "opcode2"));
    let half_precision = op1 == 0x02;
    let predicate_raw = field(hi, high, "predicate");
    let mut blocked: Option<&'static str> = None;

    // Predication conditionally writes; defer predicated emit (block, keep classified).
    if matches!(ext_vec_predicate(predicate_raw), Predicate::Raw(_)) {
        blocked = blocked.or(Some("PN (per-instance) predicate depends on repeat state - not modeled"));
    }

    // Destination (op0): RSI2 bank + reg index*2. alt_opt0 selects an exotic mode.
    let op0_sel = field(hi, high, "opt0") as u8;
    let alt_opt0 = field(hi, high, "alt_opt0");
    let dest_bank = bank_rsi2(op0_sel);
    if alt_opt0 != 0 || dest_bank.is_none() {
        blocked = blocked.or(Some("dest operand in index/exotic mode"));
    }
    let (dest_bank, dest_index) = r6_dest_bank_index(op0_sel, field(lo, low, "op0"))
        .unwrap_or((Bank::Temp, reg_index(field(lo, low, "op0"))));
    let dest = Some(Operand::plain(dest_bank, dest_index, op0_sel));

    // Source 1 (op1): RS2 bank, reg*2, precise per-channel RSWZ3 swizzle, abs/neg mods; or
    // an inline constant when alt_opt1 selects constant mode.
    let op1_sel = field(lo, low, "opt1") as u8;
    let (abs1, neg1) = (field(hi, high, "abs_op1") != 0, field(hi, high, "neg_op1") != 0);
    let op1_swizzle = {
        let c0 = field(lo, low, "op1_swz_c0") as u8;
        let c1 = field(lo, low, "op1_swz_c1") as u8;
        let c2 = (((field(hi, high, "op1_swz_c2x")) << 1) | field(lo, low, "op1_swz_c20")) as u8;
        let c3 = (((field(hi, high, "op1_swz_c3x")) << 1) | field(hi, high, "op1_swz_c30")) as u8;
        [c0, c1, c2, c3]
    };
    let s1 = if field(hi, high, "alt_opt1") != 0 {
        match exotic_source(op1_sel, field(lo, low, "op1"), op1_swizzle, abs1, neg1) {
            Some(o) => o,
            None => {
                blocked = blocked.or(Some("op1 in index/immediate mode"));
                r6_plain_source(op1_sel, field(lo, low, "op1"))
            }
        }
    } else {
        let mut s1 = r6_plain_source(op1_sel, field(lo, low, "op1"));
        s1.swizzle = op1_swizzle;
        s1.abs = abs1;
        s1.neg = neg1;
        s1
    };

    // Source 2 (op2): RS2 bank, reg*2, RSWZ2 table swizzle, abs mod; or an inline constant
    // when alt_opt2 selects constant mode.
    let op2_sel = field(lo, low, "opt2") as u8;
    let abs2 = field(hi, high, "abs_op2") != 0;
    let s2 = if field(hi, high, "alt_opt2") != 0 {
        match exotic_source(
            op2_sel,
            field(lo, low, "op2"),
            rswz2_op2(field(hi, high, "swz_alt_op2"), field(hi, high, "op2_swz")),
            abs2,
            false,
        ) {
            Some(o) => o,
            None => {
                blocked = blocked.or(Some("op2 in index/immediate mode"));
                r6_plain_source(op2_sel, field(lo, low, "op2"))
            }
        }
    } else {
        let mut s2 = r6_plain_source(op2_sel, field(lo, low, "op2"));
        s2.swizzle = rswz2_op2(field(hi, high, "swz_alt_op2"), field(hi, high, "op2_swz"));
        s2.abs = abs2;
        s2
    };

    // Destination write mask - read from the encoding for EVERY operation in this group,
    // the in-group `dot` included. A scalar dot broadcast to all four destination channels
    // (what an earlier "dot has no masking" special case assumed) silently destroys whatever
    // the untouched channels held, and this title relies on those channels: its fragments keep
    // a per-vertex scale in channel 3 of the same register a dot writes, then read it back
    // several instructions later. The mask decodes to channel 0 alone on every in-group dot in
    // the corpus, which the def-use chain independently confirms - each one is immediately
    // followed by an `rsq`/`min` reading channel 0 and nothing ever reads the other three.
    let write_mask = mask_table_08(
        field(hi, high, "swz_mask3"),
        field(hi, high, "swz_mask2"),
        field(hi, high, "swz_mask1"),
        field(hi, high, "swz_en"),
    );

    Instr {
        op,
        pred: ext_vec_predicate(predicate_raw),
        dest,
        write_mask,
        srcs: vec![s1, s2],
        half_precision,
        raw: word,
        group: op1,
        blocked,
    }
}

/// Parse a short swizzle string ("xy", "xyzw", "yzxw") into the [`Operand::swizzle`]
/// lane-selector encoding, padding unused channels with 0. Only lane chars occur in the
/// mad RSWZ2 tables (no constants).
fn swz_str(s: &str) -> [u8; 4] {
    let mut out = [0u8; 4];
    for (i, ch) in s.chars().take(4).enumerate() {
        out[i] = match ch {
            'x' => 0,
            'y' => 1,
            'z' => 2,
            _ => 3, // 'w'
        };
    }
    out
}

/// The RSWZ2 swizzle for a mad-group source operand, from the henkaku per-operand tables
/// indexed by (swz_alt << 2 | op_swz). f32 mode uses 2-lane swizzles, f16 mode 4-lane.
/// `which` is 1/2/3 for op1/op2/op3 (each has its own table).
fn rswz2_mad(which: u8, half: bool, swz_alt: u32, op_swz: u32) -> [u8; 4] {
    let idx = (((swz_alt & 1) << 2) | (op_swz & 3)) as usize;
    let f32t: [[&str; 8]; 3] = [
        ["xx", "yy", "zz", "ww", "xy", "yz", "xy", "zw"], // op1
        ["xx", "yy", "zz", "ww", "xy", "xy", "yy", "wy"], // op2
        ["xx", "yy", "zz", "ww", "xy", "xz", "xx", "xy"], // op3
    ];
    let f16t: [[&str; 8]; 3] = [
        ["xxxx", "yyyy", "zzzz", "wwww", "xyzw", "yzxw", "xyww", "zwxy"], // op1
        ["xxxx", "yyyy", "zzzz", "wwww", "xyzw", "xyyz", "yyww", "wyzw"], // op2
        ["xxxx", "yyyy", "zzzz", "wwww", "xyzw", "xzww", "xxyz", "xyzz"], // op3
    ];
    let row = (which as usize).saturating_sub(1).min(2);
    swz_str(if half { f16t[row][idx] } else { f32t[row][idx] })
}

/// The mad-group destination write mask. f32 mode writes 2 channels (swz_mask32, swz_en);
/// f16 mode writes 4 (swz_mask16, swz_en). Both from the henkaku masking tables.
fn mask_table_mad(half: bool, swz_mask16: u32, swz_mask32: u32, swz_en: u32) -> [bool; 4] {
    if half {
        // (swz_mask16, swz_en) -> ch0..3. x(masked) = not written.
        match ((swz_mask16 & 1) << 1) | (swz_en & 1) {
            0b00 => [false, false, false, false],
            0b01 => [true, true, false, false],
            0b10 => [false, false, true, true],
            _ => [true, true, true, true],
        }
    } else {
        // (swz_mask32, swz_en) -> ch0, ch1; ch2/ch3 unused in f32 mode.
        match ((swz_mask32 & 1) << 1) | (swz_en & 1) {
            0b00 => [false, false, false, false],
            0b01 => [true, false, false, false],
            0b10 => [false, true, false, false],
            _ => [true, true, false, false],
        }
    }
}

/// Decode a group-0x00 `mad` (multiply-add) instruction: `op0 = op1 * op2 + op3`.
fn decode_grp_mad(word: u64, hi: u32, lo: u32) -> Instr {
    let high = G00_HIGH;
    let low = G00_LOW;
    let half_precision = field(hi, high, "data_format") != 0;
    let predicate_raw = field(hi, high, "predicate"); // 2-bit ShortPredicate in this group
    let mut blocked: Option<&'static str> = None;
    // Every 2-bit ShortPredicate encoding resolves, so this group never blocks on predication.

    // Destination op0: RSI2 bank + reg*2; alt_opt0 selects an exotic mode.
    let op0_sel = field(hi, high, "opt0") as u8;
    let dest_bank = bank_rsi2(op0_sel);
    if field(hi, high, "alt_opt0") != 0 || dest_bank.is_none() {
        blocked = blocked.or(Some("dest operand in index/exotic mode"));
    }
    let (dest_bank, dest_index) = r6_dest_bank_index(op0_sel, field(lo, low, "op0"))
        .unwrap_or((Bank::Temp, reg_index(field(lo, low, "op0"))));
    let dest = Some(Operand::plain(dest_bank, dest_index, op0_sel));

    // op1: 1-bit bank (opt1: 0=r, 1=pa), RSWZ2 swizzle, abs (no negate for op1 in mad).
    let op1_pa = field(hi, high, "opt1") != 0;
    let mut s1 = r6_plain_source(if op1_pa { 2 } else { 0 }, field(lo, low, "op1"));
    s1.swizzle = rswz2_mad(1, half_precision, field(hi, high, "swz_alt_op1"), field(lo, low, "op1_swz"));
    s1.abs = field(hi, high, "abs_op1") != 0;

    // op2 / op3: RS2 bank + reg*2, RSWZ2 swizzle, abs+neg; or an inline constant when
    // alt_opt selects constant mode (index/immediate modes block).
    let op2_sel = field(lo, low, "opt2") as u8;
    let (abs2, neg2) = (field(hi, high, "abs_op2") != 0, field(hi, high, "neg_op2") != 0);
    let s2 = if field(hi, high, "alt_opt2") != 0 {
        match exotic_source(
            op2_sel,
            field(lo, low, "op2"),
            rswz2_mad(2, half_precision, field(hi, high, "swz_alt_op2"), field(lo, low, "op2_swz")),
            abs2,
            neg2,
        ) {
            Some(o) => o,
            None => {
                blocked = blocked.or(Some("op2 in index/immediate mode"));
                r6_plain_source(op2_sel, field(lo, low, "op2"))
            }
        }
    } else {
        let mut s2 = r6_plain_source(op2_sel, field(lo, low, "op2"));
        s2.swizzle = rswz2_mad(2, half_precision, field(hi, high, "swz_alt_op2"), field(lo, low, "op2_swz"));
        s2.abs = abs2;
        s2.neg = neg2;
        s2
    };

    let op3_sel = field(lo, low, "opt3") as u8;
    let (abs3, neg3) = (field(hi, high, "abs_op3") != 0, field(hi, high, "neg_op3") != 0);
    let s3 = if field(hi, high, "alt_opt3") != 0 {
        match exotic_source(
            op3_sel,
            field(lo, low, "op3"),
            rswz2_mad(3, half_precision, field(hi, high, "swz_alt_op3"), field(hi, high, "op3_swz")),
            abs3,
            neg3,
        ) {
            Some(o) => o,
            None => {
                blocked = blocked.or(Some("op3 in index/immediate mode"));
                r6_plain_source(op3_sel, field(lo, low, "op3"))
            }
        }
    } else {
        let mut s3 = r6_plain_source(op3_sel, field(lo, low, "op3"));
        s3.swizzle = rswz2_mad(3, half_precision, field(hi, high, "swz_alt_op3"), field(hi, high, "op3_swz"));
        s3.abs = abs3;
        s3.neg = neg3;
        s3
    };

    let write_mask = mask_table_mad(
        half_precision,
        field(hi, high, "swz_mask16"),
        field(hi, high, "swz_mask32"),
        field(hi, high, "swz_en"),
    );

    Instr {
        op: Op::Mad,
        pred: short_predicate(predicate_raw),
        dest,
        write_mask,
        srcs: vec![s1, s2, s3],
        half_precision,
        raw: word,
        group: 0x00,
        blocked,
    }
}

/// Parse a swizzle string that may contain constant selectors ('0','1') as well as lane
/// chars into the [`Operand::swizzle`] encoding: x/y/z/w -> 0/1/2/3, '0' -> 4, '1' -> 5.
/// Unused channels pad with 0 (never read past the operand's component count).
fn swz_str_const(s: &str) -> [u8; 4] {
    let mut out = [0u8; 4];
    for (i, ch) in s.chars().take(4).enumerate() {
        out[i] = match ch {
            'x' => 0,
            'y' => 1,
            'z' => 2,
            'w' => 3,
            '0' => 4,
            '1' => 5,
            _ => 0,
        };
    }
    out
}

/// The operand-2 swizzle for the 0x18 dot.f32 group, from the henkaku "Swizzles - operand
/// 2" tables indexed by (swz_alt_op2 << 2 | op2_swz). dot.f32 has a 3-channel and a
/// 4-channel table selected by `c3_en` (channel-3 enable). Constant '1' -> selector 5.
fn rswz2_dot_op2(c3_en: bool, swz_alt_op2: u32, op2_swz: u32) -> [u8; 4] {
    // 3-channel table (c3_en = 0).
    const T3: [&str; 16] = [
        "xxx", "yyy", "zzz", "www", "xyz", "yzw", "xxy", "xyx",
        "yyx", "yyz", "zxy", "xzy", "yzx", "zyx", "zzy", "xy1",
    ];
    // 4-channel table (c3_en = 1).
    const T4: [&str; 16] = [
        "xxxx", "yyyy", "zzzz", "wwww", "xyzw", "yzww", "xyzz", "xxyz",
        "xyxy", "xywz", "zxyw", "zwzw", "yzxz", "xxyy", "xzww", "xyz1",
    ];
    let idx = (((swz_alt_op2 & 3) << 2) | (op2_swz & 3)) as usize;
    swz_str_const(if c3_en { T4[idx] } else { T3[idx] })
}

/// Decode a group-0x18 instruction. Its 1-bit `opcode2` (a fact) splits the group into
/// dot.f32 (0) and mad.f32 (1); both read the same bit position (high-word bit 21).
fn decode_grp_18(word: u64, hi: u32, lo: u32) -> Instr {
    if field(hi, G18_DOT_HIGH, "opcode2") == 0 {
        decode_grp_18_dot(word, hi, lo)
    } else {
        decode_grp_18_mad(word, hi, lo)
    }
}

/// Decode a group-0x18 dot.f32: `op0 = dot(op1, op2)` over 3 or 4 channels (`c3_en`).
/// op1 is an R6/RS2 register with an explicit per-channel RSWZ3 swizzle; op2 is always an
/// internal register (RI2) with a table-driven RSWZ2 swizzle. The result is a scalar
/// written to the masked channels of op0 (standard masking table). Every field is a
/// henkaku fact; exotic operand modes and the single-channel-override strange bits are
/// blocked so the emitter hard-fails rather than mis-translate.
fn decode_grp_18_dot(word: u64, hi: u32, lo: u32) -> Instr {
    let high = G18_DOT_HIGH;
    let low = G18_DOT_LOW;
    let mut blocked: Option<&'static str> = None;

    let predicate_raw = field(hi, high, "predicate");
    if matches!(ext_vec_predicate(predicate_raw), Predicate::Raw(_)) {
        blocked = blocked.or(Some("PN (per-instance) predicate depends on repeat state - not modeled"));
    }
    // The strange bits force a single-channel masking override we do not model exactly.
    if field(hi, high, "swz_en_strange0") != 0 || field(hi, high, "swz_en_strange1") != 0 {
        blocked = blocked.or(Some("dot swz_en_strange single-channel override not modeled"));
    }

    let c3_en = field(hi, high, "c3_en") != 0;
    let components: u8 = if c3_en { 4 } else { 3 };

    // Destination op0: RSI2 bank + R6 index (with internal reserved range). alt_opt0 = exotic.
    let op0_sel = field(hi, high, "opt0") as u8;
    if field(hi, high, "alt_opt0") != 0 {
        blocked = blocked.or(Some("dest operand in index/exotic mode"));
    }
    let (dbank, didx) = match r6_dest_bank_index(op0_sel, field(lo, low, "op0")) {
        Some(v) => v,
        None => {
            blocked = blocked.or(Some("dest operand in index/exotic mode"));
            (Bank::Temp, 0)
        }
    };
    let dest = Some(Operand::plain(dbank, didx, op0_sel));

    // Source 1 (op1): R6/RS2 register, explicit per-channel RSWZ3 swizzle (c0..c3), abs+neg.
    let op1_sel = field(lo, low, "opt1") as u8;
    if field(hi, high, "alt_opt1") != 0 {
        blocked = blocked.or(Some("op1 in index/exotic mode"));
    }
    let (b1, i1) = r6_source_bank_index(op1_sel, field(lo, low, "op1"));
    let c0 = field(lo, low, "op1_swz_c0") as u8;
    let c1 = field(lo, low, "op1_swz_c1") as u8;
    let c2 = field(lo, low, "op1_swz_c2") as u8;
    let c3 = field(lo, low, "op1_swz_c3") as u8;
    let mut s1 = Operand::plain(b1, i1, op1_sel);
    s1.swizzle = [c0, c1, c2, c3];
    s1.abs = field(hi, high, "abs_op1") != 0;
    s1.neg = field(hi, high, "neg_op1") != 0;

    // Source 2 (op2): always an internal register i0..i3 (RI2), table-driven RSWZ2 swizzle,
    // abs modifier (no negate field for op2 in dot).
    let op2i = field(lo, low, "op2i");
    let mut s2 = Operand::plain(Bank::Internal, internal_base(op2i), op2i as u8);
    s2.swizzle = rswz2_dot_op2(c3_en, field(lo, low, "swz_alt_op2"), field(lo, low, "op2_swz"));
    s2.abs = field(hi, high, "abs_op2") != 0;

    let write_mask = mask_table_08(
        field(hi, high, "swz_mask3"),
        field(hi, high, "swz_mask2"),
        field(hi, high, "swz_mask1"),
        field(hi, high, "swz_en"),
    );

    Instr {
        op: Op::Dot { components },
        pred: ext_vec_predicate(predicate_raw),
        dest,
        write_mask,
        srcs: vec![s1, s2],
        half_precision: false,
        raw: word,
        group: 0x03,
        blocked,
    }
}

/// The shared 32-entry per-operand swizzle table for the 0x18 mad.f32 group (henkaku
/// "Swizzles - operand 1/2/3" - operands 1, 2 and 3 all index the SAME table). Selector
/// encoding matches [`Operand::swizzle`]: 0..3 = x,y,z,w lanes; 4 = 0.0; 5 = 1.0; 6 = 2.0;
/// 7 = 0.5; 8 = the undocumented `h` value (a sentinel the decoder blocks on rather than
/// guess). The 5-bit index is built from the operand's swizzle control fields.
const MAD18_SWZ: [[u8; 4]; 32] = [
    [0, 0, 0, 0], // xxxx
    [1, 1, 1, 0], // yyyx
    [2, 2, 2, 0], // zzzx
    [3, 3, 3, 0], // wwwx
    [0, 1, 2, 0], // xyzx
    [1, 2, 3, 0], // yzwx
    [0, 0, 1, 0], // xxyx
    [0, 1, 0, 0], // xyxx
    [1, 1, 0, 0], // yyxx
    [1, 1, 2, 0], // yyzx
    [2, 0, 1, 0], // zxyx
    [0, 2, 1, 0], // xzyx
    [1, 2, 0, 0], // yzxx
    [2, 1, 0, 0], // zyxx
    [2, 2, 1, 0], // zzyx
    [0, 1, 5, 0], // xy1x
    [0, 1, 1, 0], // xyyx
    [1, 0, 1, 0], // yxyx
    [0, 0, 2, 0], // xxzx
    [1, 0, 0, 0], // yxxx
    [0, 1, 4, 0], // xy0x
    [0, 5, 4, 0], // x10x
    [4, 4, 4, 0], // 000x
    [5, 5, 5, 0], // 111x
    [8, 8, 8, 0], // hhhx (h undocumented -> sentinel 8, blocked)
    [6, 6, 6, 0], // 222x
    [0, 4, 4, 0], // x00x
    [7, 7, 7, 7], // {0.5, 0.5, 0.5, 0.5}
    [7, 7, 7, 7],
    [7, 7, 7, 7],
    [7, 7, 7, 7],
    [7, 7, 7, 7],
];

/// The sentinel selector for the undocumented `h` swizzle value in [`MAD18_SWZ`].
const SWZ_UNKNOWN: u8 = 8;

/// Decode a group-0x18 mad.f32: `op0 = op1 * op2 + op3`. op1 is an R6/RS2 register; op2 and
/// op3 are internal registers (RI2, i0..i3), each with a swizzle from the shared 0x18 mad
/// table. Modifiers abs/neg per operand; standard masking table. All fields are henkaku
/// facts. The `op0_strange` dest adjustment and the `h` swizzle selector are undocumented
/// exactly, so an instruction using either is blocked (hard-fail) rather than guessed.
fn decode_grp_18_mad(word: u64, hi: u32, lo: u32) -> Instr {
    let high = G18_MAD_HIGH;
    let low = G18_MAD_LOW;
    let mut blocked: Option<&'static str> = None;

    let predicate_raw = field(hi, high, "predicate");
    if matches!(ext_vec_predicate(predicate_raw), Predicate::Raw(_)) {
        blocked = blocked.or(Some("PN (per-instance) predicate depends on repeat state - not modeled"));
    }
    // op0_strange0/1 adjust the destination register index in a way henkaku documents only
    // vaguely ("sums with op0, adds 2"); block rather than mis-address the destination.
    if field(hi, high, "op0_strange0") != 0 || field(hi, high, "op0_strange1") != 0 {
        blocked = blocked.or(Some("0x18 mad op0_strange dest adjustment undocumented"));
    }

    // Destination op0: RSI2 bank + R6 index (with internal reserved range).
    let op0_sel = field(hi, high, "opt0") as u8;
    if field(hi, high, "alt_opt0") != 0 {
        blocked = blocked.or(Some("dest operand in index/exotic mode"));
    }
    let (dbank, didx) = match r6_dest_bank_index(op0_sel, field(lo, low, "op0")) {
        Some(v) => v,
        None => {
            blocked = blocked.or(Some("dest operand in index/exotic mode"));
            (Bank::Temp, 0)
        }
    };
    let dest = Some(Operand::plain(dbank, didx, op0_sel));

    // op1: R6/RS2 register, swizzle from the shared table (idx = swz_alt_op1<<2 | op1_swz).
    //
    // With `alt_opt1` set the operand is on the bank-EXTENSION row - the SAME row the shared
    // operand decode resolves through [`ext_source`], so this is the established reading and
    // not a new guess: SPECIAL yields a hardware constant (FPCONSTANT) or a GLOBAL register,
    // and the two INDEXED rows plus IMMEDIATE stay unmodeled and block by name.
    //
    // It must NOT go through [`r6_source_bank_index`]: an extension-row operand is never
    // double-register scaled, so that path would report constant 1 as register 2 - a wrong
    // operand that decodes cleanly and reads like an ordinary register.
    let op1_sel = field(lo, low, "opt1") as u8;
    let op1_field = field(lo, low, "op1");
    let mut s1 = if field(hi, high, "alt_opt1") != 0 {
        match ext_source(op1_sel, op1_field) {
            Ok(o) => o,
            Err(why) => {
                blocked = blocked.or(Some(why));
                Operand::plain(Bank::Temp, 0, op1_sel)
            }
        }
    } else {
        let (b1, i1) = r6_source_bank_index(op1_sel, op1_field);
        Operand::plain(b1, i1, op1_sel)
    };
    s1.swizzle = MAD18_SWZ[(((field(lo, low, "swz_alt_op1") & 7) << 2) | (field(lo, low, "op1_swz") & 3)) as usize];
    s1.abs = field(hi, high, "abs_op1") != 0;
    s1.neg = field(hi, high, "neg_op1") != 0;

    // op2: internal register i0..i3 (op2i), swizzle idx = swz_alt_op2_2<<4 | swz_alt_op2_x<<2 | op2_swz.
    let op2i = field(lo, low, "op2i");
    let mut s2 = Operand::plain(Bank::Internal, internal_base(op2i), op2i as u8);
    s2.swizzle = MAD18_SWZ[(((field(hi, high, "swz_alt_op2_2") & 1) << 4)
        | ((field(lo, low, "swz_alt_op2_x") & 3) << 2)
        | (field(lo, low, "op2_swz") & 3)) as usize];
    s2.abs = field(hi, high, "abs_op2") != 0;

    // op3: internal register i0..i3 (op3i), swizzle idx = swz_alt_op3_2<<4 | swz_alt_op3_x<<2 | op3_swz.
    let op3i = field(lo, low, "op3i");
    let mut s3 = Operand::plain(Bank::Internal, internal_base(op3i), op3i as u8);
    s3.swizzle = MAD18_SWZ[(((field(hi, high, "swz_alt_op3_2") & 1) << 4)
        | ((field(lo, low, "swz_alt_op3_x") & 3) << 2)
        | (field(lo, low, "op3_swz") & 3)) as usize];
    s3.abs = field(hi, high, "abs_op3") != 0;
    s3.neg = field(hi, high, "neg_op3") != 0;

    // Any operand resolving to the undocumented `h` selector cannot be translated exactly.
    if [&s1, &s2, &s3].iter().any(|s| s.swizzle.contains(&SWZ_UNKNOWN)) {
        blocked = blocked.or(Some("0x18 mad 'h' swizzle selector undocumented"));
    }

    let write_mask = mask_table_08(
        field(hi, high, "swz_mask3"),
        field(hi, high, "swz_mask2"),
        field(hi, high, "swz_mask1"),
        field(hi, high, "swz_en"),
    );

    Instr {
        op: Op::Mad,
        pred: ext_vec_predicate(predicate_raw),
        dest,
        write_mask,
        srcs: vec![s1, s2, s3],
        half_precision: false,
        raw: word,
        group: 0x03,
        blocked,
    }
}

/// Decode a group-0x30 unary transcendental (rcp/rsq/log/exp): `op0 = f(op1.comp)`
/// broadcast to the masked destination channels. The complete operand encoding is a fact
/// from the SGX543 ISA reference (group VCOMP):
/// `op2` (bits 42:41) selects rcp/rsq/log/exp; the single source component is picked by
/// `src_comp` (36:35); banks/numbers/modifiers/write-mask are exact. The op is scalar - it
/// reads one source channel, applies the function, and writes the result to every channel
/// the 4-bit write mask (bits 3:0) selects.
///
/// Blocked (hard-fail, never guessed): a predicated instruction, a C10-typed operand (the
/// 10-bit packed pipeline is not modeled), an extended-bank operand (immediate/special/
/// indexed), or an index-mode destination. F16/F32 both emit in the shared scalar-lane
/// register model (the same convention the 0x08/0x10 groups already use).
fn decode_grp_30(word: u64, _hi: u32, _lo: u32) -> Instr {
    let op = match bits(word, 42, 41) {
        0 => Op::Rcp,
        1 => Op::Rsq,
        2 => Op::Log,
        _ => Op::Exp,
    };
    let predicate_raw = bits(word, 58, 56);
    let mut blocked: Option<&'static str> = None;
    if matches!(ext_predicate(predicate_raw), Predicate::Raw(_)) {
        blocked = blocked.or(Some("PN (per-instance) predicate depends on repeat state - not modeled"));
    }
    // Data formats: 0=F32 1=F16 2=C10 3=reserved. Block C10/reserved (unmodeled packing).
    let dest_type = bits(word, 54, 53);
    let src_type = bits(word, 40, 39);
    if dest_type >= 2 || src_type >= 2 {
        blocked = blocked.or(Some("0x30 C10/reserved operand type not modeled"));
    }
    // Extended banks (immediate/special/indexed) are not modeled for this group.
    if bits(word, 51, 51) != 0 || bits(word, 49, 49) != 0 {
        blocked = blocked.or(Some("0x30 extended-bank operand (immediate/special/indexed) not modeled"));
    }

    // Destination op0: 2-bit bank (0=temp 1=output 2=pa 3=indexed) + R7 number, with the
    // reserved top-of-temp internal range.
    // Group 0x30's operand fields sit at the same bit positions as 0x40's and 0x50's but are
    // DOUBLE-REGISTER, not direct. Established by def-use on a real fragment: a normalize reads
    // `dot -> PA[4]` back through `rsq`, and the rsq's fields are 2 - so field 2 addresses
    // register 4. Do not "unify" this with the neighbouring groups; the ISA encodes each group's
    // operands its own way, and the only safe rule is to decide per group on evidence.
    let op0_sel = bits(word, 33, 32) as u8;
    let dest_n = bits(word, 27, 21);
    let dest = match r6_dest_bank_index(op0_sel, dest_n) {
        Some((b, i)) => Some(Operand::plain(b, i, op0_sel)),
        None => {
            blocked = blocked.or(Some("dest operand in index mode"));
            None
        }
    };

    // Source op1: 2-bit bank + R7 number (with internal range), 2-bit modifier, broadcast of
    // the single selected source component to every channel.
    let op1_sel = bits(word, 31, 30) as u8;
    let (b1, i1) = r6_source_bank_index(op1_sel, bits(word, 13, 7));
    let (abs1, neg1) = src_mod2(bits(word, 38, 37));
    let comp = bits(word, 36, 35) as u8; // 0=x 1=y 2=z 3=w
    let mut s1 = Operand::plain(b1, i1, op1_sel);
    s1.swizzle = [comp, comp, comp, comp];
    s1.abs = abs1;
    s1.neg = neg1;

    Instr {
        op,
        pred: ext_predicate(predicate_raw),
        dest,
        write_mask: write_mask4(bits(word, 3, 0)),
        srcs: vec![s1],
        half_precision: dest_type == 1,
        raw: word,
        group: 0x06,
        blocked,
    }
}

/// Decode a group-0x38 VMOV (move / conditional-move). The complete operand encoding is a
/// fact from the SGX543 ISA reference (group VMOV):
/// `move_type` (bits 47:46) selects unconditional VMOV (0) vs conditional VMOVC (1) /
/// VMOVCU8 (2); the source-1 swizzle (in double-register mode) is the vec4-standard table
/// indexed by `src0_swiz` (38:35); the destination write mask is the direct 4-bit field
/// (27:24). An unconditional VMOV is `dest = src1` (a swizzled per-channel copy). A VMOVC is
/// `dest.c = compare(src0.c, 0) ? src1.c : src2.c`, the compare method (spec B.1b) being
/// `(test_bit_2 << 1) | test_bit_1` (bits 54 and 39): 0 EQ, 1 NE, 2 LT, 3 LTE against zero.
///
/// Blocked (hard-fail, never guessed): a predicated instruction, VMOVCU8 (`move_type==2`, its
/// test reads the source as UINT8 which the scalar-float register model cannot represent) and
/// the reserved `move_type==3`, a non-float data type (INT8/16/32, UINT8/16 - the scalar-lane
/// register model holds floats) or C10, an extended-bank operand on any read operand, or an
/// index-mode destination.
fn decode_grp_38(word: u64, _hi: u32, _lo: u32) -> Instr {
    let move_type = bits(word, 47, 46);
    let conditional = move_type == 1; // VMOVC (float conditional select); the only wired form
    let predicate_raw = bits(word, 58, 56);
    let mut blocked: Option<&'static str> = None;
    if matches!(ext_predicate(predicate_raw), Predicate::Raw(_)) {
        blocked = blocked.or(Some("PN (per-instance) predicate depends on repeat state - not modeled"));
    }
    if move_type == 2 {
        blocked = blocked.or(Some("0x38 VMOVCU8 (UINT8-test conditional move) not modeled in float register file"));
    }
    if move_type == 3 {
        blocked = blocked.or(Some("0x38 reserved move_type 3"));
    }
    // Data type (bits 42:40): 0=INT8 1=INT16 2=INT32 3=C10 4=F16 5=F32 6=UINT8 7=UINT16.
    // Only the float types F16/F32 map onto the scalar-lane register model; block the rest.
    let data_type = bits(word, 42, 40);
    let is_float = data_type == 4 || data_type == 5;
    if !is_float {
        blocked = blocked.or(Some("0x38 non-float (integer/C10) move not modeled in scalar-lane register file"));
    }
    // Extended-bank rows for the DESTINATION (bit 51) and, in the conditional form, src0
    // (bit 50) select index/immediate/indexed addressing this decoder does not model - block
    // them (a constant destination is meaningless; src0's ext modes are unmodeled). The SRC1
    // (bit 49) and SRC2 (bit 48) extension bits are handled at their own operand decodes
    // below, where the resolvable modes are decoded rather than blocked wholesale. In the
    // unconditional form bit 50 = `end` and bit 48 is unused, so they are ignored.
    let ext_hit = bits(word, 51, 51) != 0 || (conditional && bits(word, 50, 50) != 0);
    if ext_hit {
        blocked = blocked.or(Some("0x38 extended-bank operand (immediate/special/indexed) not modeled"));
    }

    // Destination op0: 2-bit bank + R6 number (with reserved top-of-temp internal range).
    let op0_sel = bits(word, 33, 32) as u8;
    let dest = match r6_dest_bank_index(op0_sel, bits(word, 23, 18)) {
        Some((b, idx)) => Some(Operand::plain(b, idx, op0_sel)),
        None => {
            blocked = blocked.or(Some("dest operand in index mode"));
            None
        }
    };

    // Source 1: normally a 2-bit bank (RS2) + R6 number, with the double-register (float) mode
    // swizzle from the vec4-standard table (src0_swiz). When alt_opt1 (bit 49, the src1
    // extended-bank helper) is set, `opt1` (src1_sel) instead selects an addressing mode per
    // the SGX543 operand-N table (henkaku wiki, FACT): 00 = index1 (RIO6), 01 = CNST6 constant,
    // 10 = immediate (IMM6), 11 = index2 (RIO6). Only the constant mode is resolvable from clean
    // facts (the CNST6 table, exactly as the group 0x00 constant operand is handled); the RIO6/
    // IMM6 addressing modes are not modeled and stay blocked.
    let src1_sel = bits(word, 31, 30) as u8;
    let src1_ext = bits(word, 49, 49) != 0;
    let s1 = if src1_ext {
        if src1_sel & 3 == 0b01 {
            // CNST6 constant: op1 (bits 11:6) is the 6-bit constant selector; the operand's
            // swizzle still selects the constant bank per channel, so it is carried through.
            let mut o = Operand::plain(Bank::Constant, (bits(word, 11, 6) & 0x3f) as u8, src1_sel);
            if is_float {
                o.swizzle = VEC4_STD_SWIZZLE[bits(word, 38, 35) as usize];
            }
            o
        } else {
            blocked = blocked.or(Some("0x38 src1 extended bank (index1/immediate/index2) not modeled"));
            Operand::plain(Bank::Temp, 0, src1_sel) // placeholder; the instruction is blocked
        }
    } else {
        let (b1, i1) = r6_source_bank_index(src1_sel, bits(word, 11, 6));
        let mut o = Operand::plain(b1, i1, src1_sel);
        if is_float {
            o.swizzle = VEC4_STD_SWIZZLE[bits(word, 38, 35) as usize];
        }
        o
    };

    let (op, srcs) = if conditional {
        // VMOVC: dest.c = compare(src0.c, 0) ? src1.c : src2.c.
        let test = match (bits(word, 54, 54) << 1) | bits(word, 39, 39) {
            0 => CompareMethod::EqZero,
            1 => CompareMethod::NeZero,
            2 => CompareMethod::LtZero,
            _ => CompareMethod::LteZero,
        };
        // src0 (test value): 1-bit bank (bit 34) + reserved top-of-temp internal range.
        // ext (bit 50) already forces a block above, so only the ext=0 row is decoded here:
        // sel 0 => TEMP, sel 1 => PRIMATTR (A.2 src0 table).
        let src0_sel = bits(word, 34, 34);
        let src0_n = bits(word, 17, 12);
        let src0_bank = if src0_sel == 0 { Bank::Temp } else { Bank::PrimaryAttr };
        let (b0, i0) = if matches!(src0_bank, Bank::Temp) && (60..=63).contains(&src0_n) {
            (Bank::Internal, internal_base(src0_n - 60))
        } else {
            (src0_bank, reg_index(src0_n))
        };
        let mut s0 = Operand::plain(b0, i0, src0_sel as u8);
        // src0 uses src1's swizzle when src0_comp_sel (bit 53) is set, else identity.
        if bits(word, 53, 53) != 0 {
            s0.swizzle = s1.swizzle;
        }
        // src2 (false value): 2-bit bank (29:28) + R6 number (5:0), swizzle copied from src1.
        // With the extension bit (48) set the selector names the extension row instead; here
        // that is the hardware constant bank (a conditional move whose "else" arm is a
        // constant). Note the field is only SIX bits, so its `0x40` GLOBAL discriminator can
        // never be set and SPECIAL always resolves to FPCONSTANT in this group.
        let src2_sel = bits(word, 29, 28) as u8;
        let mut s2 = if bits(word, 48, 48) != 0 {
            match ext_source(src2_sel, bits(word, 5, 0)) {
                Ok(o) => o,
                Err(why) => {
                    blocked = blocked.or(Some(why));
                    Operand::plain(Bank::Temp, 0, src2_sel)
                }
            }
        } else {
            let (b2, i2) = r6_source_bank_index(src2_sel, bits(word, 5, 0));
            Operand::plain(b2, i2, src2_sel)
        };
        s2.swizzle = s1.swizzle;
        (Op::Cmov { test }, vec![s1, s2, s0])
    } else {
        (Op::Mov, vec![s1])
    };

    Instr {
        op,
        pred: ext_predicate(predicate_raw),
        dest,
        write_mask: write_mask4(bits(word, 27, 24)),
        srcs,
        half_precision: data_type == 4,
        raw: word,
        group: 0x07,
        blocked,
    }
}

/// Decode a group-0x50 VBW (integer bitwise / shift). The encoding is a fact from the
/// SGX543 ISA reference (group VBW): the operation is
/// `op1` (bits 61:59) + `op2` (bit 35) - AND/OR (010), XOR (011), SHL/ROL (100), SHR/ASR
/// (101); it is a scalar op on channel 0 only. Source 2 may be a register or an inline
/// immediate `src2_n | (src2_sel<<7) | (src2_exth<<14)`, and in both cases a `src2_rot`
/// rotate-left (bits 42:38) and `src2_invert` (bit 43) apply.
///
/// Emitted for: any AND/OR/XOR/SHL/SHR/ASR whose source 2 is a plain register (no rotate /
/// invert) OR an immediate (rotate/invert folded in at decode). ROL (rotate) and the
/// register-source-with-rotate/invert cases are decoded but hard-fail BLOCKED (named) until
/// wired. `op1`/`op2` come straight from the ISA; the operation on the 32-bit lane bit
/// pattern is exact.
fn decode_grp_bitwise(word: u64, op1: u8) -> Instr {
    use crate::ir::BitwiseKind::*;
    let op2 = bits(word, 35, 35);
    let kind = match (op1 & 0b111, op2) {
        (0b010, 0) => And,
        (0b010, _) => Or,
        (0b011, _) => Xor,
        (0b100, 0) => Shl,
        (0b101, 0) => Shr,
        (0b101, _) => Asr,
        // (0b100, 1) => ROL - rotate; not wired (handled below by leaving kind and blocking).
        _ => And,
    };
    let is_rol = (op1 & 0b111) == 0b100 && op2 == 1;

    let predicate_raw = bits(word, 58, 56);
    let mut blocked: Option<&'static str> = None;
    if matches!(ext_predicate(predicate_raw), Predicate::Raw(_)) {
        blocked = blocked.or(Some("PN (per-instance) predicate depends on repeat state - not modeled"));
    }
    if is_rol {
        blocked = blocked.or(Some("0x50 ROL (rotate-left) not yet wired"));
    }

    let width16 = bits(word, 34, 34) != 0;
    if width16 {
        // 16-bit-lane bitwise needs a result mask the f32-lane emit does not yet carry.
        blocked = blocked.or(Some("0x50 16-bit-lane bitwise not yet wired"));
    }
    let lane_mask: u32 = if width16 { 0xFFFF } else { 0xFFFF_FFFF };
    let rot = bits(word, 42, 38) & if width16 { 0xF } else { 0x1F };
    let invert = bits(word, 43, 43) != 0;

    // Destination: 2-bit bank + R7 number (with reserved top-of-temp internal range).
    let dest_sel = bits(word, 33, 32) as u8;
    let dest_n = bits(word, 27, 21);
    let dest = match r7_dest_bank_index(dest_sel, dest_n) {
        Some((b, i)) => Some(Operand::plain(b, i, dest_sel)),
        None => {
            blocked = blocked.or(Some("dest operand in index mode"));
            None
        }
    };

    // Source 1: 2-bit bank + R7 number, channel 0 only. With the extension bit set the
    // selector names the extension row instead (FPCONSTANT / GLOBAL / immediate / indexed);
    // a bitwise move of a hardware constant is the case that occurs here.
    let src1_sel = bits(word, 31, 30) as u8;
    let s1 = if bits(word, 49, 49) != 0 {
        match ext_source(src1_sel, bits(word, 13, 7)) {
            Ok(o) => o,
            Err(why) => {
                blocked = blocked.or(Some(why));
                Operand::plain(Bank::Temp, 0, src1_sel)
            }
        }
    } else {
        let (b1, i1) = r7_source_bank_index(src1_sel, bits(word, 13, 7));
        Operand::plain(b1, i1, src1_sel)
    };

    // Source 2: an immediate (ext + IMMEDIATE bank) folds rotate/invert at decode; otherwise
    // a plain register (rotate/invert on a register operand are not wired -> block).
    let src2_sel = bits(word, 29, 28);
    let src2_ext = bits(word, 48, 48) != 0;
    let mut srcs = vec![s1];
    let imm = if src2_ext && src2_sel == 2 {
        let raw_imm = bits(word, 6, 0) | (bits(word, 20, 14) << 7) | (bits(word, 37, 36) << 14);
        let mut v = raw_imm & lane_mask;
        if rot != 0 {
            v = ((v << rot) | (v >> ((if width16 { 16 } else { 32 }) - rot))) & lane_mask;
        }
        if invert {
            v = !v & lane_mask;
        }
        Some(v)
    } else {
        if src2_ext {
            blocked = blocked.or(Some("0x50 src2 extended bank (special/indexed) not modeled"));
        }
        if rot != 0 || invert {
            blocked = blocked.or(Some("0x50 register source-2 with rotate/invert not yet wired"));
        }
        let (b2, i2) = r7_source_bank_index(src2_sel as u8, bits(word, 6, 0));
        srcs.push(Operand::plain(b2, i2, src2_sel as u8));
        None
    };

    Instr {
        op: Op::Bitwise { kind, imm },
        pred: ext_predicate(predicate_raw),
        dest,
        write_mask: [true, false, false, false],
        srcs,
        half_precision: false,
        raw: word,
        group: op1,
        blocked,
    }
}

/// Decode a group-0x40 VPCK (pack / unpack / format-convert). The encoding is a fact from
/// the SGX543 ISA reference (group VPCK): `src_fmt`
/// (bits 43:41) and `dest_fmt` (40:38) pick source/dest formats (0=U8 1=S8 2=O8 3=U16
/// 4=S16 5=F16 6=F32 7=C10); four 2-bit component selectors form the source swizzle; the
/// direct 4-bit `dest_mask` (37:34) selects written channels.
///
/// Only the value-preserving FLOAT<->FLOAT subset (F16/F32 on both sides) is emitted: it is a
/// swizzled copy `dest = src1` (no cycling for float->float, per the ISA note) that preserves
/// the NUMBER while changing its STORAGE WIDTH. The two formats are independent fields, so the
/// source is read at `src_fmt`'s precision and the destination written at `dest_fmt`'s -
/// carried as `Op::Pack { src_half }` next to the instruction's `half_precision`. Collapsing
/// them into one precision makes an F16->F32 unpack read its source as a 32-bit float, which
/// yields a denormal rather than the value. The integer<->float NORMALIZED conversions (the `scale`
/// bit path) and the C10/O8 packed formats change the stored value and depend on a packed
/// representation this model does not carry, so they are decoded but hard-fail BLOCKED
/// (named) rather than guessed. src1 is double-register (float source), so the scalarised
/// contiguous lanes naturally span the adjacent register a two-source gather would use.
fn decode_grp_pack(word: u64) -> Instr {
    let src_fmt = bits(word, 43, 41);
    let dest_fmt = bits(word, 40, 38);
    let predicate_raw = bits(word, 58, 56);
    let mut blocked: Option<&'static str> = None;
    if matches!(ext_predicate(predicate_raw), Predicate::Raw(_)) {
        blocked = blocked.or(Some("PN (per-instance) predicate depends on repeat state - not modeled"));
    }
    // Float formats are F16 (5) and F32 (6). Only float<->float is a value-preserving copy.
    let is_float = |f: u32| f == 5 || f == 6;
    if !(is_float(src_fmt) && is_float(dest_fmt)) {
        blocked = blocked.or(Some("0x40 pack non-float<->float conversion (int-normalize / C10 / O8) not modeled"));
    }
    if bits(word, 51, 51) != 0 || bits(word, 49, 49) != 0 {
        blocked = blocked.or(Some("0x40 pack extended-bank operand (immediate/special/indexed) not modeled"));
    }

    // Destination: 2-bit bank + R7 number (with reserved top-of-temp internal range).
    let dest_sel = bits(word, 33, 32) as u8;
    let dest_n = bits(word, 27, 21);
    let dest = match r7_dest_bank_index(dest_sel, dest_n) {
        Some((b, i)) => Some(Operand::plain(b, i, dest_sel)),
        None => {
            blocked = blocked.or(Some("dest operand in index mode"));
            None
        }
    };

    // Source 1 (float source => double-register, direct number). The component selectors
    // form the source swizzle (no cycling for float->float). comp0's high bit is comp0_sel_
    // bit1 (bit 7) for an F32 source, else the low bit of src2_n (bit 1).
    let src1_sel = bits(word, 31, 30) as u8;
    // SIX-bit field: bit 7 is the F32 source's comp0 selector high bit, so this operand is R6
    // (double-register), unlike the R7 destination above. Cross-checked on a real fragment
    // whose F32->F16 fog pack reads field 4 -> register 8, exactly the pa_base the container's
    // Fog interpolant descriptor accumulates to.
    let (b1, i1) = r6_source_bank_index(src1_sel, bits(word, 13, 8));
    let comp0_hi = if src_fmt == 6 { bits(word, 7, 7) } else { bits(word, 1, 1) };
    let c0 = ((comp0_hi << 1) | bits(word, 0, 0)) as u8;
    let c1 = bits(word, 17, 16) as u8;
    let c2 = bits(word, 15, 14) as u8;
    let c3 = bits(word, 20, 19) as u8;
    let mut s1 = Operand::plain(b1, i1, src1_sel);
    s1.swizzle = [c0, c1, c2, c3];

    Instr {
        op: Op::Pack { src_half: src_fmt == 5 },
        pred: ext_predicate(predicate_raw),
        dest,
        write_mask: write_mask4(bits(word, 37, 34)),
        srcs: vec![s1],
        half_precision: dest_fmt == 5,
        raw: word,
        group: 0x08,
        blocked,
    }
}

/// Decode a group-0xE0 texture sample (SMP). The encoding is a fact from the SGX543 ISA
/// reference (group 0xE0): the coordinate is
/// `src0` (number bits 20:14, 1-bit bank 34 + ext 50), the sampler unit is `src1` (number
/// bits 13:7), the destination is `dest_n` (27:21) in the primary-attr bank when
/// `dest_use_pa` (bit 39) else temp, `dim` (43:42, base-0) gives the coordinate component
/// count, and `lod_mode` (41:40) / `sb_mode` (38:37) select the sample variant. The sampled
/// RGBA is written to the destination's four channels.
///
/// This wires the common case a fragment shader uses: a normal (`sb_mode==0`), implicit-LOD
/// (`lod_mode==0`), 1D/2D sample. Blocked (hard-fail, named): predication, the gather/info
/// sub-behaviours, explicit-LOD/bias/gradient, 3D/cube coordinates (need a matching sampler
/// type at bind time), and a C10 coordinate. The sampler unit is the raw `src1` number,
/// cross-checked against the reflected sampler parameter table at pipeline-build time.
fn decode_grp_tex(word: u64) -> Instr {
    let mut blocked: Option<&'static str> = None;
    let predicate_raw = bits(word, 58, 56);
    if matches!(ext_predicate(predicate_raw), Predicate::Raw(_)) {
        blocked = blocked.or(Some("PN (per-instance) predicate depends on repeat state - not modeled"));
    }
    if bits(word, 38, 37) != 0 {
        blocked = blocked.or(Some("0xE0 tex gather4/info sub-behaviour not yet wired"));
    }
    // `lod_mode` (E0.4) selects where the mip level comes from. Bias and explicit-LOD each
    // read one scalar from src2 and map onto a WGSL sample variant exactly; the GRADIENT form
    // reads two derivative VECTORS whose component split this decoder has no corpus evidence
    // for, so it stays blocked.
    let lod = match bits(word, 41, 40) {
        0 => TexLod::Implicit,
        1 => TexLod::Bias,
        2 => TexLod::Level,
        _ => {
            blocked = blocked.or(Some("0xE0 tex gradient (lod_mode 3) not yet wired"));
            TexLod::Implicit
        }
    };
    // dim is base-0: 0 => 1D (padded to 2D), 1 => 2D, 2 => 3D/cube (3 coords), 3 => reserved.
    let dim_field = bits(word, 43, 42);
    let coords = match dim_field {
        0 => 1u8,
        1 => 2,
        2 => 3,
        _ => {
            blocked = blocked.or(Some("0xE0 tex reserved dim field 3 (unexpected)"));
            2
        }
    };
    // Coordinate data type (`src0_type`, E0.2): 0 = F32, 1 = F16, 2 = C10. The coordinate is
    // usually computed by the F16 ALU pipeline, so reading it as F32 would sample garbage.
    let coord_half = match bits(word, 36, 35) {
        1 => true,
        0 => false,
        _ => {
            blocked = blocked.or(Some("0xE0 tex C10 coordinate type not modeled"));
            false
        }
    };

    // Coordinate operand src0: 1-bit bank selector + extension. ext=0 -> temp/pa; ext=1 ->
    // output/sa. Temp numbers in the reserved top range alias the internal registers.
    let src0_ext = bits(word, 50, 50) != 0;
    let src0_sel = bits(word, 34, 34);
    let coord_bank = match (src0_ext, src0_sel) {
        (false, 0) => Bank::Temp,
        (false, _) => Bank::PrimaryAttr,
        (true, 0) => Bank::Output,
        (true, _) => Bank::SecondaryAttr,
    };
    let src0_n = bits(word, 20, 14);
    let (cbank, cidx) = if matches!(coord_bank, Bank::Temp) && (124..=127).contains(&src0_n) {
        (Bank::Internal, internal_base(src0_n - 124))
    } else {
        (coord_bank, reg_index(src0_n))
    };
    let coord = Operand::plain(cbank, cidx, src0_sel as u8);

    // Sampler unit = src1 register number. Under the shared double-register rule the bound
    // texture's control words live at SA register `2 * src1_n`, which the container's
    // texture-control table resolves to a GXM texture unit (see `Program::sampler_unit_at`).
    let unit = bits(word, 13, 7) as u8;
    // Result data type (`fconv_type`, E0.5): 0/3 = F32, 2 = F16, 1 = "the bound texture's
    // component type" - not knowable from the instruction alone, so it hard-blocks rather
    // than guessing a width that would mis-address the destination register pair.
    let result_f16 = match bits(word, 47, 46) {
        2 => true,
        0 | 3 => false,
        _ => {
            blocked = blocked.or(Some("0xE0 tex result format 1 (texture-derived type) not resolvable"));
            true
        }
    };
    let dest_pa = bits(word, 39, 39) != 0;
    let dest_bank = if dest_pa { Bank::PrimaryAttr } else { Bank::Temp };
    let dest_n = bits(word, 27, 21);
    // The SMP destination is NOT double-scaled the way an ALU operand is: an F16 4-component
    // result lands in the register PAIR `dest_n, dest_n+1`. Established by def-use chains in
    // real fragment blobs (every albedo/ambient/fog sample resolves only under this rule);
    // see the SA-bank layout notes distilled alongside the ISA reference.
    let dest_reg = if result_f16 { dest_n } else { reg_index(dest_n) as u32 };
    let (dbank, didx) = if matches!(dest_bank, Bank::Temp) && (124..=127).contains(&dest_n) {
        (Bank::Internal, internal_base(dest_n - 124))
    } else {
        (dest_bank, dest_reg as u8)
    };

    // src2 carries the bias / explicit level when `lod_mode` asks for one. Like every other
    // SMP operand it is double-register scaled (spec E0.2). It is read as F32: in the corpus
    // the instruction immediately before each explicit-LOD sample is a `mov .f32` writing
    // exactly this register, so the level is produced and consumed at 32-bit width.
    let mut tex_srcs = vec![coord];
    if !matches!(lod, TexLod::Implicit) {
        let src2_sel = bits(word, 29, 28) as u8;
        let (b2, i2) = source_bank_index(src2_sel, bits(word, 6, 0), 124, reg_index);
        tex_srcs.push(Operand::plain(b2, i2, src2_sel));
    }

    Instr {
        op: Op::Tex { unit, coords, coord_half, lod },
        pred: ext_predicate(predicate_raw),
        dest: Some(Operand::plain(dbank, didx, 0)),
        write_mask: [true; 4],
        srcs: tex_srcs,
        half_precision: result_f16,
        raw: word,
        group: 0x1c,
        blocked,
    }
}

/// Decode a group-0xF8 complex-flow instruction. Member selection is a fact from the
/// SGX543 ISA reference (group 0xF8): the
/// operation category `opcat` (bits 53:52), `opcat_extra` (bit 54), and the secondary
/// `op2` (bits 58:56) pick among PHAS / NOP / BR / SMLSI / SMBO / KILL / LIMM / DEPTHF.
///
/// Only the two members that are *provably* free of any effect on the register file the
/// emitter models are wired as no-ops: PHAS (the mandatory phase-declaration header every
/// program begins with - it sets phase metadata, no data) and NOP. Everything else is
/// hard-fail BLOCKED naming the member, rather than silently ignored: SMLSI/SMBO set
/// per-operand repeat / base-offset state that would change how LATER instructions address
/// registers, so treating them as no-ops could mis-address and paint a wrong pixel; BR is
/// control flow; KILL/DEPTHF/LIMM have real data effects (discard / depth write / immediate
/// load) not yet plumbed. Each is wired as it becomes needed by a real shader.
/// Whether `word` provably executes EXACTLY ONCE, i.e. carries no repeat.
///
/// This is what decides whether an SMLSI ahead of it can reach anything: SMLSI only sets the
/// per-operand increment/swizzle state that a REPEATED instruction steps its registers by
/// (spec F8.8 - "metadata for repeated instructions; emits nothing directly"). An instruction
/// that runs once never consults it. See [`slots_repeats_consult`].
///
/// `None` means "cannot be established for this group" and must be treated as "might repeat".
/// This is [`repeat_extra_iterations`] asked as a yes/no - deliberately the SAME answer, since
/// both questions turn on the single fact of where (or whether) a group encodes a repeat count,
/// and two readings of that could only disagree by one of them being wrong.
///
/// It once answered the 0x00/0x18 vector-MAD groups more strictly, demanding every `unk*` bit
/// of the operand grammar be zero on the reasoning that a repeat count might hide among them.
/// The corpus refutes that: those groups' `unk` bits are set all over the captured programs
/// (bit 55 - the position the ISA gives `sync` in every group that documents it - is set on
/// essentially every MAD), while the whole corpus's vertex programs write EXACTLY their
/// declared output interface under the reading that a MAD executes once. Had those bits been a
/// repeat count, a matrix-transform MAD would re-execute five or nine times and stomp lanes far
/// outside the declared interface, which is precisely what
/// `vertex_written_lanes_close_against_declared_total` would report. The strictness was
/// therefore not buying safety - it was only refusing SMLSIs whose repeats are elsewhere.
fn executes_once(word: u64) -> Option<bool> {
    repeat_extra_iterations(word).map(|extra| extra == 0)
}

/// How many EXTRA times `word` re-executes after its first execution (0 = runs once), or
/// `None` when this group's repeat encoding is not established and the answer therefore cannot
/// be stated. A caller that cannot handle `None` must block the instruction rather than assume
/// zero: assuming zero is what silently drops the later iterations of a repeating instruction.
///
/// The field map is the one [`executes_once`] documents, kept here as the single place the
/// per-group repeat encoding is written down so the two can never disagree.
pub fn repeat_extra_iterations(word: u64) -> Option<u32> {
    match opcode1(word) {
        // Established repeat_count fields.
        0x06 | 0x08 | 0x0a..=0x0d => Some(bits(word, 47, 44) as u32), // 0x30, 0x40, 0x50 family
        0x07 | 0x09 | 0x0f => Some(bits(word, 45, 44) as u32),        // 0x38 VMOV, 0x48 VTST
        // No repeat_count field exists in these layouts.
        0x01 | 0x02 => Some(0), // 0x08/0x10 V32NMAD/V16NMAD - 47:44 is src2_swiz
        0x1c => Some(0),        // 0xE0 SMP
        0x1f => Some(0),        // 0xF8 complex flow
        // 0x00/0x18 vector MAD/DP: no repeat_count field either. Every group whose grammar is
        // documented places `repeat_count` at bits 47:44, and in this group's own field table
        // those four bits are `abs_op2` plus the `op0_strange`/`swz_en_strange` pair - named
        // operand-modifier bits the decoder already reads, and already blocks on where their
        // effect is undocumented. A repeat count cannot also live there. The remaining `unk`
        // bits in this group are scattered singles, not a contiguous field at any position a
        // repeat count occupies elsewhere.
        0x00 | 0x03 => Some(0),
        _ => None,
    }
}

/// True when `word` is a group-0xF8 SMLSI (the discriminant [`decode_grp_flow`] matches on).
pub(crate) fn is_smlsi(word: u64) -> bool {
    opcode1(word) == 0x1f && bits(word, 58, 56) == 0b010 && bits(word, 53, 52) == 0b01
}

/// What one SMLSI leaves for a single hardware operand slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SmlsiSlot {
    /// The operand advances by `n` of its OWN widths per repeat iteration - the unit is the
    /// operand width, not the register word, which is what the corpus measurement below shows.
    Increment(u8),
    /// The operand does not advance by a register count; the byte selects the channels each
    /// iteration reads instead. Not modeled.
    Swizzle(u8),
}

/// Decode an SMLSI word into the per-operand repeat state it sets, indexed by hardware operand
/// slot: `[dest, src0, src1, src2]` - the operand order every group's own field table uses.
///
/// The layout is four 8-bit values in bits[31:0] (slot `k` at bits `8k+7 : 8k`) and four mode
/// bits at bits[35:32] (slot `k` at bit `32+k`), which is the spec's "four inc-mode bits, four
/// 8-bit increments in the low 32 bits" read as the 36-bit field [35:0] it must be - the
/// sentence describes 36 bits and the increments alone are 32 of them.
///
/// MEASURED against the corpus, on both ends of the only idiom that uses it. Across two
/// unrelated titles, eight vertex programs open with `SMLSI; VMOV(repeat N)` copying vertex
/// attributes straight to the output bank, and in every one of them the DEFAULT stepping (one
/// operand width per iteration) is what closes:
///
///  * on the destination side, one program's three iterations of `Output[8] <- PA[4]` land its
///    last write exactly on the `TexCoord(0)` varying the container declares at output lane 12,
///    and the program's writes then fill lanes 0..13 of a declared 14-lane interface with no
///    gap. A stride of one word would never reach lane 12 (the varying would go uninterpolated);
///    a stride of four would run two lanes past the declared interface.
///  * on the source side, another program's four iterations of `Output[4] <- PA[4]` consume
///    exactly `PA[4..11]` - its `in_texCoord` and `in_colour` attributes, the whole declared
///    12-register attribute set with nothing left over. Under a non-advancing source every
///    iteration would re-read `in_texCoord.xy` and `in_colour` would be dead, yet the container
///    declares it as a fed vertex attribute and no other instruction in that program reads it.
///
/// In all four distinct SMLSI words the corpus contains, the dest / src1 / src2 slots carry
/// increment 1 - the default - and only the src0 slot varies (0x01, 0x0e, 0x4e, 0xf6, with its
/// mode bit set in two of them). Every repeat those words govern is an UNCONDITIONAL VMOV,
/// which reads src1 alone and never src0, so that byte is a don't-care the compiler left
/// uninitialised. That is why [`slots_repeats_consult`] asks which slots are actually read
/// rather than demanding the whole word be default.
///
/// Bit 50 also varies (set on the words that restore the default, clear on the ones that open
/// the attribute copy). It sits where group 0x38 documents `end`, and it is not part of either
/// field above; the register-addressing model does not read it.
pub(crate) fn decode_smlsi(word: u64) -> [SmlsiSlot; 4] {
    std::array::from_fn(|k| {
        let value = ((word >> (8 * k)) & 0xff) as u8;
        if (word >> (32 + k)) & 1 == 0 {
            SmlsiSlot::Increment(value)
        } else {
            SmlsiSlot::Swizzle(value)
        }
    })
}

/// The hardware operand slots `word` READS or WRITES, indexed as [`decode_smlsi`] indexes them,
/// or `None` when that is not established for this opcode group.
///
/// Only a repeating instruction consults SMLSI state at all, and it consults it only for the
/// slots it actually uses - so this is what decides whether a non-default byte in an SMLSI can
/// reach anything. `None` must be read as "every slot", never as "no slot".
fn slots_used(word: u64) -> Option<[bool; 4]> {
    match opcode1(word) {
        // 0x38 VMOV. `move_type` (47:46) 0 is the unconditional form, `dest = src1` - it has no
        // src0 and no src2 operand at all (see `decode_grp_38`, where those fields are decoded
        // only under the conditional form). Any other move_type is a conditional select reading
        // all three sources.
        0x07 if bits(word, 47, 46) == 0 => Some([true, false, true, false]),
        0x07 => Some([true, true, true, true]),
        _ => None,
    }
}

/// The union of hardware operand slots that any instruction in `code` which CAN repeat will
/// consult - i.e. exactly the slots an SMLSI's state can still reach.
///
/// An instruction established as single-execution never advances anything, so it contributes
/// nothing. One whose repeat encoding (or operand grammar) is not established contributes every
/// slot, because being wrong here silently mis-addresses an operand.
pub(crate) fn slots_repeats_consult(code: &[u64]) -> [bool; 4] {
    let mut needed = [false; 4];
    for &w in code {
        if executes_once(w) == Some(true) {
            continue;
        }
        let used = slots_used(w).unwrap_or([true; 4]);
        for (n, u) in needed.iter_mut().zip(used) {
            *n |= u;
        }
    }
    needed
}

fn decode_grp_flow(word: u64) -> Instr {
    let op2 = bits(word, 58, 56);
    let opcat = bits(word, 53, 52);
    let opcat_extra = bits(word, 54, 54);

    // Identify the member (order matches the ISA reference: specific members before the
    // broad catch-all). Returns the classified op and, for a no-op member, `None` blocked.
    let (op, blocked): (Op, Option<&'static str>) = if op2 == 0b010 && bits(word, 54, 52) == 0b100 {
        (Op::Nop, None) // PHAS - phase declaration header, no data effect
    } else if opcat_extra == 0 && opcat == 0 && bits(word, 42, 40) == 0b101 {
        (Op::Nop, None) // NOP
    } else if op2 == 0b010 && opcat == 0b01 {
        (Op::Todo("flow smlsi (repeat-state) not modeled"), Some("0xF8 SMLSI repeat/swizzle state not modeled - would mis-address later instructions"))
    } else if op2 == 0b011 && opcat == 0b01 {
        (Op::Todo("flow smbo (base-offset state) not modeled"), Some("0xF8 SMBO base-offset state not modeled - would mis-address later instructions"))
    } else if op2 == 0b001 && opcat == 0b11 {
        (Op::Kill, None)
    } else if op2 == 0b100 && opcat == 0b10 {
        (Op::Todo("flow limm load-immediate"), Some("0xF8 LIMM (load 32-bit immediate) not yet wired"))
    } else if op2 == 0b011 && opcat == 0b11 {
        (Op::Todo("flow depthf depth-write"), Some("0xF8 DEPTHF (fragment depth write) not yet wired"))
    } else if opcat_extra == 0 && opcat == 0 {
        // BR (branch) family. `br_op` (bits[40:39]) selects the member within it: 0 is a plain
        // branch, and the reference's return shape carries 2. The word EVERY captured vertex
        // program emits at instruction #1 (0xf800094000000000, immediately after the PHAS
        // prologue) carries br_op = 2 with a ZERO displacement and no predicate, so under either
        // reading it cannot move the program counter across any instruction - it neither skips
        // forward over code nor loops backward over it - and a linear decode is semantically
        // unaffected. That is validated across 13 vertex blobs: they compute o0..o3 clip
        // position + varyings after it, and the title renders. It stays a structural no-op.
        //
        // br_op == 0 with a displacement is genuine control flow, which
        // [`crate::wgsl::emit_body`] reconstructs as structured WGSL. `save_link` (bit 41) marks
        // a branch-with-link (a call), whose matching return is not modelled, so it blocks.
        let br_off = bits(word, 19, 0);
        let br_op = bits(word, 40, 39);
        let save_link = bits(word, 41, 41) == 1;
        if br_off == 0 && op2 == 0 {
            (Op::Nop, None)
        } else if br_op != 0 {
            (Op::Todo("flow branch (non-branch br_op)"), Some("0xF8 BR with a displacement and a non-branch br_op is not modelled"))
        } else if save_link {
            (Op::Todo("flow branch-with-link"), Some("0xF8 BR save_link (branch-with-link / call) not modelled"))
        } else {
            // The offset is a 20-bit count of 64-bit instruction words relative to the branch's
            // own program offset. It is two's-complement signed only when `br_type` (bit 38) is
            // set AND the field's own sign bit (bit 19) is set; otherwise it is the raw
            // non-negative value (spec F8.2).
            let br_type = bits(word, 38, 38) == 1;
            let signed = br_type && (br_off & (1 << 19)) != 0;
            let rel = if signed { (br_off as i32) - (1 << 20) } else { br_off as i32 };
            if rel == 0 {
                // A PREDICATED zero-displacement branch: under the offset convention above its
                // target is the branch itself, i.e. an infinite loop whenever the predicate
                // holds, which no shader compiler emits. That makes the word evidence that the
                // convention is wrong for it rather than a case to translate, so it blocks.
                // (The UNCONDITIONAL zero-displacement word is handled above: it is the
                // universal prologue, and is a no-op under either convention.)
                (Op::Todo("flow branch (zero displacement)"), Some("0xF8 BR with a zero displacement and a predicate is not a structurable skip"))
            } else {
                (Op::Branch { rel }, None)
            }
        }
    } else {
        // SPEC catch-all (its discriminant fixes only the top 5 bits; op2/opcat are
        // don't-cares). It is a documented NO-OP: it writes no register/predicate/output and
        // does not touch the o0/pa0 colour path. In real fragment shaders it appears once,
        // at the end (e.g. the word 0xf920000000000000), so it is safe to emit nothing.
        (Op::Nop, None)
    };

    // KILL carries its OWN 2-bit predicate at bits[42:41] rather than the group's usual
    // ExtPredicate slot, and the reference describes two conflicting orderings for it:
    // the analyzer's (0 NONE, 1 NEGP0, 2 NEGP1, 3 P0) and a plain ShortPredicate
    // (0 NONE, 1 P0, 2 P1, 3 NEGP0). The corpus settles it. Every KILL in it is preceded,
    // two instructions earlier, by a VTST that writes p1 with `albedo.a - threshold >= 0`
    // ("the texel passes the alpha test") and carries this field as 2. A discard must run when
    // that test FAILS, so 2 is NEGP1 - the analyzer ordering. Under the ShortPredicate reading
    // it would be P1, i.e. discard exactly the texels that passed, which erases the surface
    // and keeps the transparent parts: the inverse of an alpha test.
    let pred = if matches!(op, Op::Kill) {
        match bits(word, 42, 41) {
            0 => Predicate::Always,
            1 => Predicate::IfNotP(0),
            2 => Predicate::IfNotP(1),
            _ => Predicate::IfP(0),
        }
    } else if matches!(op, Op::Branch { .. }) {
        // A branch reads the group's normal ExtPredicate slot (bits[58:56]) as its CONDITION,
        // and unlike an ALU op it cannot be treated as unpredicated: the predicate IS the
        // control flow, so an unresolved encoding (PN) must reach the emitter as `Raw` and
        // block there rather than silently becoming "always taken".
        ext_predicate(op2 as u32)
    } else {
        Predicate::Always
    };

    Instr {
        op,
        pred,
        dest: None,
        write_mask: [false; 4],
        srcs: Vec::new(),
        half_precision: false,
        raw: word,
        group: 0x1f,
        blocked,
    }
}

/// Classify an instruction whose operand decode is not yet wired: set its operation from
/// the ISA opcode map (a fact) but leave operands empty so the emitter hard-fails naming
/// the op. `hi`/`lo` are used only where a sub-opcode is needed to name the op.
fn classified_stub(word: u64, op1: u8, _hi: u32, lo: u32) -> Instr {
    let op = match op1 {
        0x00 => Op::Mad,                       // 0x00 mad (handled in decode)
        0x01 | 0x02 => alu_op(field(lo, G08_LOW, "opcode2")),
        // 0x03 (0x18 dot/mad) is handled directly in decode() with full operand decode.
        0x04 => Op::Todo("grp20 mad/dot/add/mul/subfl/exp/mov/log/rsq/rcp"),
        0x05 => Op::Todo("grp28 mad/dot/mul/add/mov/rsq/rsp"),
        // 0x06 (0x30 rcp/rsq/log/exp) and 0x07 (0x38 mov/cmov) are handled in decode() with
        // structural operand decode; their channel mask/swizzle tables remain unestablished.
        0x08 => Op::Todo("pack"),              // 0x40
        0x0a => Op::Todo("and.u32"),           // 0x50
        0x0b => Op::Todo("xor.u32"),
        0x0c => Op::Todo("shl.u32"),
        0x0d => Op::Todo("shr.u32"),
        0x0e => Op::Todo("rlp.u32"),
        0x10 => Op::Todo("add.fx8"),
        0x11 => Op::Todo("add/sub.fx8"),
        0x12 => Op::Todo("add/sub/min/max.fx8"),
        0x13 => Op::Todo("mad.u8"),
        0x14 | 0x15 | 0x1a => Op::Todo("mad (integer group)"),
        0x19 => Op::Todo("mad.u8"),
        0x1c => Op::Todo("tex"),               // 0xE0
        0x1d => Op::Todo("lda32/ldl32/ldt32"), // 0xE8
        0x1e => Op::Todo("sta32/stl32/stt32"), // 0xF0
        0x1f => Op::Todo("flow (0xF8 complex)"),
        // 0x09/0x0f are the TEST group (VTST / VTSTMSK): a compare that writes a predicate
        // (VTST) or a per-channel mask (VTSTMSK). Fully decoded from clean facts, but the
        // real captured VTST tests a GLOBAL special hardware register (`p0 = (GLOBAL[16] &
        // imm) != 0`), which the WGSL register-file model does not model - so it stays a
        // named Todo (hard-fail) until GLOBAL register semantics are established, rather than
        // guessing a value. See the distilled VTST/VTSTMSK spec notes.
        0x09 => Op::Todo("vtst (test->predicate; reads GLOBAL special reg, not modeled)"),
        0x0f => Op::Todo("vtstmsk (test->mask)"),
        // 0x16, 0x17, 0x18, 0x1b: documented illegal groups.
        0x16 | 0x17 | 0x18 | 0x1b => Op::Illegal,
        _ => Op::Unsupported { group: op1 },
    };
    Instr {
        op,
        pred: Predicate::Always,
        dest: None,
        write_mask: [true; 4],
        srcs: Vec::new(),
        half_precision: false,
        raw: word,
        group: op1,
        // Operands are not decoded for stub groups, so even an emittable op (e.g. 0x18
        // dot) must be blocked until its operand layout is wired - the emitter then hard-
        // fails naming this, rather than emitting with missing operands.
        blocked: Some("operand decode not yet wired for this group"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inverse of [`field`]: pack named values into a 32-bit word MSB-first per `table`.
    /// Used to build synthetic instruction words with known fields, so the decoder can be
    /// checked against the grammar layout exactly.
    fn encode(table: &[Field], vals: &[(&str, u32)]) -> u32 {
        // Verify the table tiles a full 32 bits (a grammar invariant).
        let sum: u8 = table.iter().map(|&(_, w)| w).sum();
        assert_eq!(sum, 32, "table widths must sum to 32");
        let mut word = 0u32;
        let mut pos = 32u8;
        for &(fname, width) in table {
            pos -= width;
            if let Some(&(_, v)) = vals.iter().find(|&&(n, _)| n == fname) {
                let mask = if width == 32 { u32::MAX } else { (1u32 << width) - 1 };
                word |= (v & mask) << pos;
            }
        }
        word
    }

    fn word(hi: u32, lo: u32) -> u64 {
        ((hi as u64) << 32) | lo as u64
    }

    #[test]
    fn field_extraction_is_inverse_of_encode() {
        // Round-trip a set of fields through the 0x08 layout.
        let hi = encode(
            G08_HIGH,
            &[("opcode1", 0x01), ("predicate", 0b101), ("swz_en", 1), ("neg_op1", 1)],
        );
        assert_eq!(field(hi, G08_HIGH, "opcode1"), 0x01);
        assert_eq!(field(hi, G08_HIGH, "predicate"), 0b101);
        assert_eq!(field(hi, G08_HIGH, "swz_en"), 1);
        assert_eq!(field(hi, G08_HIGH, "neg_op1"), 1);
        assert_eq!(field(hi, G08_HIGH, "abs_op1"), 0);

        let lo = encode(G08_LOW, &[("op0", 0x2a), ("op1", 0x11), ("op2", 0x07), ("opcode2", 0b011)]);
        assert_eq!(field(lo, G08_LOW, "op0"), 0x2a);
        assert_eq!(field(lo, G08_LOW, "op1"), 0x11);
        assert_eq!(field(lo, G08_LOW, "op2"), 0x07);
        assert_eq!(field(lo, G08_LOW, "opcode2"), 0b011);
    }

    #[test]
    fn opcode1_is_top_five_bits() {
        // opcode1 occupies instruction bits 63:59.
        assert_eq!(opcode1(word(0x0100_0000, 0)), 0x00);
        assert_eq!(opcode1(word(0b0001_1000 << 24, 0)), 0x03); // 0x18 group
        assert_eq!(opcode1(word(0b0011_1000 << 24, 0)), 0x07); // 0x38 group
        assert_eq!(opcode1(word(0xffff_ffff, 0xffff_ffff)), 0x1f);
    }

    #[test]
    fn all_group_tables_tile_32_bits() {
        for (name, high, low) in GROUP_TABLES {
            let sh: u8 = high.iter().map(|&(_, w)| w).sum();
            let sl: u8 = low.iter().map(|&(_, w)| w).sum();
            assert_eq!(sh, 32, "{name} high table must tile 32 bits");
            assert_eq!(sl, 32, "{name} low table must tile 32 bits");
        }
    }

    #[test]
    fn decodes_alu_op_operands_banks_and_index() {
        // 0x10-group (f16) instruction: opcode2=1 (add), op0/op1/op2 registers, banks via
        // opt fields. Register index = field*2 (henkaku fact); banks via RS2/RSI2.
        let hi = encode(
            G08_HIGH,
            &[("opcode1", 0x02), ("predicate", 0), ("opt0", 0b10) /* pa */],
        );
        let lo = encode(
            G08_LOW,
            &[("opcode2", 1), ("op0", 0x0c), ("op1", 0x05), ("op2", 0x21),
              ("opt1", 0b00) /* r */, ("opt2", 0b11) /* sa */],
        );
        let ins = decode(word(hi, lo));
        assert_eq!(ins.group, 0x02);
        assert_eq!(ins.op, Op::Add);
        assert!(ins.half_precision, "0x10 group is f16");
        assert_eq!(ins.pred, Predicate::Always);
        // dest: bank pa (RSI2 0b10), index 0x0c*2 = 0x18.
        let d = ins.dest.unwrap();
        assert_eq!(d.bank, Bank::PrimaryAttr);
        assert_eq!(d.index, 0x18);
        // src1: bank r (temp), index 0x05*2 = 0x0a. src2: bank sa, index 0x21*2 = 0x42.
        assert_eq!(ins.srcs.len(), 2);
        assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::Temp, 0x0a));
        assert_eq!((ins.srcs[1].bank, ins.srcs[1].index), (Bank::SecondaryAttr, 0x42));
        assert!(ins.is_supported());
    }

    #[test]
    fn alu_opcode2_maps_all_eight_ops() {
        let want = [Op::Mul, Op::Add, Op::Frc, Op::Dsx, Op::Dsy, Op::Min, Op::Max, Op::Dot { components: 4 }];
        for (opc2, exp) in want.iter().enumerate() {
            let hi = encode(G08_HIGH, &[("opcode1", 0x01)]);
            let lo = encode(G08_LOW, &[("opcode2", opc2 as u32)]);
            assert_eq!(decode(word(hi, lo)).op, *exp, "opcode2={opc2}");
        }
    }

    #[test]
    fn alu_swizzle_and_mask_decode() {
        // op1 per-channel RSWZ3 = y,z,w,x ; op2 RSWZ2 = xyzw ; dest mask = all (11 1 1).
        let hi = encode(
            G08_HIGH,
            &[("opcode1", 0x01), ("swz_mask3", 1), ("swz_mask2", 1), ("swz_mask1", 1), ("swz_en", 1),
              ("op1_swz_c2x", 0b01) /* c2 hi bits -> with c20 below */,
              ("op1_swz_c3x", 0b01), ("op1_swz_c30", 1),
              ("swz_alt_op2", 0b01), ("op2_swz", 0b00) /* -> xyzw */,
              ("abs_op1", 1)],
        );
        let lo = encode(
            G08_LOW,
            &[("opcode2", 0), ("op1_swz_c0", 1) /* y */, ("op1_swz_c1", 2) /* z */,
              ("op1_swz_c20", 1) /* c2 = 01<<1|1 = 3 -> w */],
        );
        let ins = decode(word(hi, lo));
        // c3 = c3x(01)<<1 | c30(1) = 0b011 = 3 -> w. So op1 swizzle = [y,z,w,w]? c0=1,c1=2,c2=3,c3=3.
        assert_eq!(ins.srcs[0].swizzle, [1, 2, 3, 3]);
        assert!(ins.srcs[0].abs);
        assert_eq!(ins.srcs[1].swizzle, [0, 1, 2, 3]); // op2 xyzw
        assert_eq!(ins.write_mask, [true, true, true, true]);
    }

    #[test]
    fn decodes_real_vmovc_conditional_move() {
        // The real VMOVC word that blocked frag_82ee8c30 at instr #12: F32 conditional move,
        // compare method LT_ZERO (test_bit_2=1, test_bit_1=0), unpredicated, no extended bank.
        let ins = decode(0x38404506a1001006);
        assert_eq!(ins.group, 0x07);
        assert_eq!(ins.op, Op::Cmov { test: crate::ir::CompareMethod::LtZero });
        assert!(!ins.half_precision, "move_data_type 5 = F32");
        assert_eq!(ins.pred, Predicate::Always);
        assert_eq!(ins.blocked, None, "a plain float VMOVC must not be blocked");
        assert!(ins.is_supported());
        // VMOVC carries three sources: src1 (true), src2 (false), src0 (test).
        assert_eq!(ins.srcs.len(), 3);
    }

    #[test]
    fn vmovcu8_and_predicated_move_still_block() {
        // move_type 2 (VMOVCU8) cannot be modeled in the float register file -> blocked.
        let cu8 = decode(0x38408506a1001006); // same word, move_type bits 47:46 -> 10
        assert!(cu8.blocked.is_some(), "VMOVCU8 must hard-fail, not emit");
        // A predicated unconditional VMOV still blocks (predicate model not yet wired).
        let pred = decode(0x39000000_00000000);
        assert!(pred.blocked.is_some());
    }

    #[test]
    fn vmov_src1_extended_constant_decodes_and_index_blocks() {
        // An unconditional VMOV whose src1 is in extended-bank mode (alt_opt1 = bit 49) with
        // opt1 = 01 selects a CNST6 constant (SGX543 operand-N table): op1 (bits 11:6) is the
        // constant selector. Build a synthetic word: opcode1 0x38, data_type F32 (5 @ 42:40),
        // dest = output reg (opt0 = 01 @ 33:32, op0 = 9 @ 23:18 -> o[18]), write channel 0.
        let base: u64 = (0x07 << 59)   // opcode1 0x38 (top 5 bits)
            | (5 << 40)                // data_type F32
            | (1 << 32)                // opt0 = 01 -> output bank (dest)
            | (9 << 18)                // op0 dest field 9 -> reg_index 18
            | (1 << 24)                // write_mask channel 0
            | (2 << 6)                 // op1 field = 2 (CNST6 selector 2 = 1.0)
            | (1 << 49); // alt_opt1: src1 extended-bank
        // opt1 = 01 (bit 30 set) -> CNST6 constant, resolvable.
        let cst = decode(base | (1 << 30));
        assert_eq!(cst.op, Op::Mov);
        assert_eq!(cst.blocked, None, "src1 ext-constant must decode, not block");
        assert_eq!(cst.srcs.len(), 1);
        assert_eq!(cst.srcs[0].bank, Bank::Constant);
        assert_eq!(cst.srcs[0].index, 2);
        assert_eq!((cst.dest.unwrap().bank, cst.dest.unwrap().index), (Bank::Output, 18));
        // opt1 = 00 (index1 mode) is a real RIO6 addressing mode not modeled -> still blocked.
        let idx = decode(base); // bits 31:30 = 00
        assert!(idx.blocked.is_some(), "src1 ext index1 mode must stay blocked");
    }

    #[test]
    fn decodes_mad_operands_banks_mask() {
        // f16 mad (data_format=1 -> 4 channels): op0 dest r, op1 pa, op2 sa, op3 r.
        let hi = encode(
            G00_HIGH,
            &[("opcode1", 0), ("data_format", 1), ("swz_mask16", 1), ("swz_en", 1),
              ("opt1", 1) /* op1 = pa */, ("opt0", 0) /* dest = r */, ("abs_op2", 1)],
        );
        let lo = encode(
            G00_LOW,
            &[("op0", 2), ("op1", 3), ("op2", 4), ("op3", 5),
              ("opt2", 0b11) /* sa */, ("opt3", 0b00) /* r */],
        );
        let ins = decode(word(hi, lo));
        assert_eq!(ins.op, Op::Mad);
        assert!(ins.half_precision);
        assert_eq!(ins.write_mask, [true, true, true, true]);
        assert_eq!((ins.dest.unwrap().bank, ins.dest.unwrap().index), (Bank::Temp, 4));
        assert_eq!(ins.srcs.len(), 3);
        assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::PrimaryAttr, 6)); // op1 pa
        assert_eq!((ins.srcs[1].bank, ins.srcs[1].index), (Bank::SecondaryAttr, 8)); // op2 sa
        assert!(ins.srcs[1].abs);
        assert_eq!((ins.srcs[2].bank, ins.srcs[2].index), (Bank::Temp, 10)); // op3 r
        assert!(ins.is_supported());
    }

    #[test]
    fn mad_f32_writes_two_channels() {
        // f32 mad (data_format=0): full write is 2 channels (swz_mask32=1, swz_en=1).
        let hi = encode(G00_HIGH, &[("opcode1", 0), ("data_format", 0), ("swz_mask32", 1), ("swz_en", 1)]);
        let lo = encode(G00_LOW, &[("op0", 1), ("op1", 1), ("op2", 1), ("op3", 1)]);
        let ins = decode(word(hi, lo));
        assert!(!ins.half_precision);
        assert_eq!(ins.write_mask, [true, true, false, false]);
    }

    #[test]
    fn decodes_18_dot_operands_internal_src_and_swizzle() {
        // 0x18 dot.f32, c3_en=1 (4 channels): op0 dest r (opt0=00) reg 5 -> r10;
        // op1 src pa (opt1=10) reg 3 -> pa6 with explicit RSWZ3 x,y,z,w; op2 internal
        // register i1 (op2i=1) with 4ch swizzle table[0] = xxxx; full write mask.
        let hi = encode(
            G18_DOT_HIGH,
            &[("opcode1", 0x03), ("opcode2", 0), ("c3_en", 1), ("predicate", 0),
              ("swz_mask3", 1), ("swz_mask2", 1), ("swz_mask1", 1), ("swz_en", 1),
              ("opt0", 0b00) /* dest r */],
        );
        let lo = encode(
            G18_DOT_LOW,
            &[("op0", 5), ("opt1", 0b10) /* pa */, ("op1", 3), ("op2i", 1),
              ("op1_swz_c0", 0), ("op1_swz_c1", 1), ("op1_swz_c2", 2), ("op1_swz_c3", 3),
              ("swz_alt_op2", 0), ("op2_swz", 0)],
        );
        let ins = decode(word(hi, lo));
        assert_eq!(ins.group, 0x03);
        assert_eq!(ins.op, Op::Dot { components: 4 });
        assert!(ins.blocked.is_none(), "plain dot must not be blocked: {:?}", ins.blocked);
        let d = ins.dest.unwrap();
        assert_eq!((d.bank, d.index), (Bank::Temp, 10));
        assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::PrimaryAttr, 6));
        assert_eq!(ins.srcs[0].swizzle, [0, 1, 2, 3]);
        // op2 = internal i1 -> base lane 4; xxxx swizzle.
        assert_eq!((ins.srcs[1].bank, ins.srcs[1].index), (Bank::Internal, 4));
        assert_eq!(ins.srcs[1].swizzle, [0, 0, 0, 0]);
        assert_eq!(ins.write_mask, [true, true, true, true]);
    }

    #[test]
    fn dot_18_three_channel_and_reserved_internal_src() {
        // c3_en=0 -> 3 channels, op2 swizzle from the 3ch table[4] = xyz.
        let hi = encode(G18_DOT_HIGH, &[("opcode1", 0x03), ("opcode2", 0), ("c3_en", 0)]);
        let lo = encode(
            G18_DOT_LOW,
            &[("op0", 1), ("opt1", 0b00) /* r */, ("op1", 60) /* reserved -> i0 */,
              ("op2i", 2), ("swz_alt_op2", 1), ("op2_swz", 0) /* idx 4 -> xyz */],
        );
        let ins = decode(word(hi, lo));
        assert_eq!(ins.op, Op::Dot { components: 3 });
        // op1 field 60 in the r bank is the reserved i0 encoding.
        assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::Internal, 0));
        // op2 = i2 -> base 8; 3ch table swz_alt=1,op2_swz=0 -> index 4 = xyz.
        assert_eq!((ins.srcs[1].bank, ins.srcs[1].index), (Bank::Internal, 8));
        assert_eq!(&ins.srcs[1].swizzle[..3], &[0, 1, 2]);
    }

    #[test]
    fn decodes_18_mad_operands_two_internal_srcs() {
        // 0x18 mad.f32 (opcode2=1): op0 dest r reg 4 -> r8; op1 pa reg 3 -> pa6 (xxxx);
        // op2 = internal i1; op3 = internal i2; full write mask.
        let hi = encode(
            G18_MAD_HIGH,
            &[("opcode1", 0x03), ("opcode2", 1), ("opt0", 0b00) /* dest r */,
              ("swz_mask3", 1), ("swz_mask2", 1), ("swz_mask1", 1), ("swz_en", 1),
              ("neg_op1", 1)],
        );
        let lo = encode(
            G18_MAD_LOW,
            &[("op0", 4), ("opt1", 0b10) /* pa */, ("op1", 3), ("op2i", 1), ("op3i", 2)],
        );
        let ins = decode(word(hi, lo));
        assert_eq!(ins.op, Op::Mad);
        assert!(ins.blocked.is_none(), "plain 0x18 mad must not be blocked: {:?}", ins.blocked);
        assert!(ins.is_supported());
        assert_eq!((ins.dest.unwrap().bank, ins.dest.unwrap().index), (Bank::Temp, 8));
        assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::PrimaryAttr, 6));
        assert!(ins.srcs[0].neg);
        assert_eq!((ins.srcs[1].bank, ins.srcs[1].index), (Bank::Internal, 4)); // i1
        assert_eq!((ins.srcs[2].bank, ins.srcs[2].index), (Bank::Internal, 8)); // i2
        assert_eq!(ins.write_mask, [true, true, true, true]);
    }

    #[test]
    fn mad_18_blocks_on_strange_and_h_swizzle() {
        // op0_strange set -> blocked (undocumented dest adjustment).
        let hi = encode(G18_MAD_HIGH, &[("opcode1", 0x03), ("opcode2", 1), ("op0_strange0", 1)]);
        assert!(decode(word(hi, 0)).blocked.is_some());
        // op1 swizzle resolving to the `h` table entry (index 24 = swz_alt_op1=0b110,
        // op1_swz=0b00) -> blocked.
        let hi2 = encode(G18_MAD_HIGH, &[("opcode1", 0x03), ("opcode2", 1)]);
        let lo2 = encode(G18_MAD_LOW, &[("swz_alt_op1", 0b110), ("op1_swz", 0)]);
        let ins = decode(word(hi2, lo2));
        assert_eq!(ins.op, Op::Mad);
        assert!(ins.blocked.is_some(), "h-swizzle must block");
    }

    #[test]
    fn alu_op2_constant_mode_decodes_and_is_emittable() {
        // 0x08 mul with alt_opt2=1, opt2=01 -> op2 is a CNST6 constant (index from op2
        // field). op2 field = 2 -> CNST6[2] = 1.0. Must not block; must be emittable.
        let hi = encode(G08_HIGH, &[("opcode1", 0x01), ("alt_opt2", 1)]);
        let lo = encode(G08_LOW, &[("opcode2", 0), ("op2", 2), ("opt2", 0b01) /* const mode */]);
        let ins = decode(word(hi, lo));
        assert_eq!(ins.op, Op::Mul);
        assert_eq!(ins.srcs[1].bank, Bank::Constant);
        assert_eq!(ins.srcs[1].index, 2);
        assert!(ins.blocked.is_none(), "constant mode should not block: {:?}", ins.blocked);
        assert!(ins.is_supported());
    }

    #[test]
    fn alu_op2_index_mode_still_blocks() {
        // alt_opt2=1 with opt2=00 (index1 mode) is not resolvable -> block.
        let hi = encode(G08_HIGH, &[("opcode1", 0x01), ("alt_opt2", 1)]);
        let lo = encode(G08_LOW, &[("opcode2", 0), ("opt2", 0b00)]);
        assert!(decode(word(hi, lo)).blocked.is_some());
    }

    /// Build a 64-bit word by OR-ing `(value, msb, lsb)` fields at absolute bit positions -
    /// the encoding the 0x30+ decoders read (mirrors [`bits`]).
    fn word_bits(fields: &[(u64, u32, u32)]) -> u64 {
        let mut w = 0u64;
        for &(v, msb, lsb) in fields {
            let width = msb - lsb + 1;
            let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
            w |= (v & mask) << lsb;
        }
        w
    }

    #[test]
    fn decodes_30_transcendental_and_emits() {
        // 0x30 rsq (op2=1): dest pa (sel=10) field 3 -> pa6; src pa (sel=10) field 5 -> pa10.
        // Group 0x30's fields are double-register (unlike 0x40's and 0x50's, which sit at the
        // same bit positions but are direct). src_comp=x, abs modifier; full write mask.
        let w = word_bits(&[
            (0x06, 63, 59), // opcode1
            (1, 42, 41),    // op2 = rsq
            (0b10, 33, 32), // dest bank = pa
            (3, 27, 21),    // dest_n
            (0b10, 31, 30), // src1 bank = pa
            (5, 13, 7),     // src1_n
            (0b10, 38, 37), // src1_mod = absolute
            (0, 36, 35),    // src_comp = x
            (0b1111, 3, 0), // write mask = xyzw
        ]);
        let ins = decode(w);
        assert_eq!(ins.op, Op::Rsq);
        assert_eq!(ins.group, 0x06);
        assert_eq!((ins.dest.unwrap().bank, ins.dest.unwrap().index), (Bank::PrimaryAttr, 6));
        assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::PrimaryAttr, 10));
        assert_eq!(ins.srcs[0].swizzle, [0, 0, 0, 0]); // component x broadcast
        assert!(ins.srcs[0].abs);
        assert_eq!(ins.write_mask, [true, true, true, true]);
        assert!(ins.blocked.is_none() && ins.is_supported(), "plain 0x30 must emit: {:?}", ins.blocked);
    }

    /// Group 0x30 reads the full 3-bit ExtPredicate: 1..4 select p0..p3, 5/6 are the negated
    /// p0/p1, and only 7 (PN, whose meaning depends on repeat state) is unresolvable and
    /// blocks. The two tables differ at 4..6, so a group reading the wrong one silently
    /// INVERTS a condition - which is why the resolution is per group and pinned here.
    #[test]
    fn decode_30_resolves_ext_predicate_and_blocks_only_pn() {
        let at = |v: u64| decode(word_bits(&[(0x06, 63, 59), (v, 58, 56)]));
        assert_eq!(at(0).pred, Predicate::Always);
        assert_eq!(at(1).pred, Predicate::IfP(0));
        assert_eq!(at(4).pred, Predicate::IfP(3));
        assert_eq!(at(5).pred, Predicate::IfNotP(0));
        assert_eq!(at(6).pred, Predicate::IfNotP(1));
        for v in 0..=6 {
            assert!(at(v).blocked.is_none(), "ExtPredicate {v} must not block");
        }
        assert!(at(7).blocked.is_some_and(|b| b.contains("PN")));
        // C10 source type (src_type=2) -> blocked (unchanged).
        let c10 = word_bits(&[(0x06, 63, 59), (2, 40, 39)]);
        assert!(decode(c10).blocked.is_some());
    }

    /// The vector-ALU groups read ExtVecPredicate, where 4/5/6 are NEGATED p0/p1/p2 and there
    /// is no p3 - the exact opposite reading of 4 from [`ext_predicate`].
    #[test]
    fn vector_alu_groups_read_the_vector_predicate_table() {
        let at = |v: u64| decode(word_bits(&[(0x03, 63, 59), (v, 58, 56)]));
        assert_eq!(at(3).pred, Predicate::IfP(2));
        assert_eq!(at(4).pred, Predicate::IfNotP(0), "vector table: 4 is NEGP0, not P3");
        assert_eq!(at(6).pred, Predicate::IfNotP(2));
        assert!(at(7).blocked.is_some_and(|b| b.contains("PN")));
    }

    #[test]
    fn decodes_38_mov_and_emits() {
        // 0x38 VMOV (move_type=0, F32 data type=5): dest output (sel=1) reg 4 -> o8; src1 sa
        // (sel=11) reg 6 -> sa12, src0_swiz=4 (XYZW); full write mask. Fully emittable.
        let w = word_bits(&[
            (0x07, 63, 59), // opcode1
            (0, 47, 46),    // move_type = VMOV (unconditional)
            (5, 42, 40),    // data type = F32
            (1, 33, 32),    // dest bank = output
            (4, 23, 18),    // dest_n
            (0b11, 31, 30), // src1 bank = sa
            (6, 11, 6),     // src1_n
            (4, 38, 35),    // src0_swiz index 4 = XYZW
            (0b1111, 27, 24), // write mask
        ]);
        let ins = decode(w);
        assert_eq!(ins.op, Op::Mov);
        assert_eq!((ins.dest.unwrap().bank, ins.dest.unwrap().index), (Bank::Output, 8));
        assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::SecondaryAttr, 12));
        assert_eq!(ins.srcs[0].swizzle, [0, 1, 2, 3]);
        assert_eq!(ins.write_mask, [true, true, true, true]);
        assert!(ins.blocked.is_none() && ins.is_supported(), "plain 0x38 mov must emit: {:?}", ins.blocked);
    }

    #[test]
    fn decode_38_conditional_move_and_integer() {
        // Float conditional move (move_type=1, F32) now emits a Cmov (not blocked).
        let cmov = word_bits(&[(0x07, 63, 59), (1, 47, 46), (5, 42, 40)]);
        let ci = decode(cmov);
        assert!(matches!(ci.op, Op::Cmov { .. }), "float VMOVC classifies as Cmov");
        assert!(ci.blocked.is_none() && ci.is_supported(), "plain float VMOVC must emit");
        // VMOVCU8 (move_type=2) is still blocked (UINT8 test not modeled in a float file).
        let cu8 = word_bits(&[(0x07, 63, 59), (2, 47, 46), (5, 42, 40)]);
        assert!(decode(cu8).blocked.is_some());
        // Integer data type (INT32=2) -> blocked (scalar-lane file holds floats).
        let int_mov = word_bits(&[(0x07, 63, 59), (0, 47, 46), (2, 42, 40)]);
        assert!(decode(int_mov).blocked.is_some());
    }

    #[test]
    fn flow_prologue_phas_decodes_to_nop() {
        // The universal shader prologue word is a PHAS (phase declaration): a no-op for
        // codegen, so it must not block whole-shader emit.
        let ins = decode(0xfa44070000000000);
        assert_eq!(ins.group, 0x1f);
        assert_eq!(ins.op, Op::Nop);
        assert!(ins.blocked.is_none(), "PHAS prologue must be a no-op, not blocked");
        assert!(ins.is_supported());
    }

    #[test]
    fn flow_state_and_branch_ops_block_named() {
        // SMLSI (op2=010, opcat=01) sets repeat state -> must block (not silently no-op).
        let smlsi = word_bits(&[(0x1f, 63, 59), (0b010, 58, 56), (0b01, 53, 52)]);
        let ins = decode(smlsi);
        assert!(ins.blocked.is_some() && ins.blocked.unwrap().contains("SMLSI"));
        // A branch-with-link (save_link, bit 41) is a CALL: its matching return is not
        // modelled, so it blocks naming itself rather than translating as a plain skip.
        let call = word_bits(&[
            (0x1f, 63, 59),
            (0b001, 58, 56),
            (0, 54, 54),
            (0b00, 53, 52),
            (1, 41, 41),
            (4, 19, 0),
        ]);
        let ins = decode(call);
        assert!(ins.blocked.is_some_and(|r| r.contains("save_link")), "{:?}", ins.blocked);
    }

    #[test]
    fn zero_offset_unconditional_branch_is_nop() {
        // The exact word every captured vertex program emits at instruction #1: an
        // unconditional (pred=0), zero-displacement branch, right after PHAS. It cannot move
        // the PC across any instruction, so it decodes to a structural no-op (unblocked).
        let ins = decode(0xf800094000000000);
        assert_eq!(ins.op, Op::Nop);
        assert!(ins.blocked.is_none(), "zero-offset unconditional branch must not block");

        // A branch with a NONZERO displacement is genuine control flow: it decodes to a
        // BRANCH carrying its instruction-word delta, and the emitter's structuring pass
        // decides whether that delta is expressible.
        let taken = word_bits(&[(0x1f, 63, 59), (0, 58, 56), (0, 54, 54), (0b00, 53, 52), (5, 19, 0)]);
        let ti = decode(taken);
        assert_eq!(ti.op, Op::Branch { rel: 5 });
        assert_eq!(ti.pred, Predicate::Always);
        assert!(ti.blocked.is_none(), "a plain forward branch must not block: {:?}", ti.blocked);
        // A CONDITIONAL zero-offset branch stays blocked: its target is the branch itself.
        let cond = word_bits(&[(0x1f, 63, 59), (0b001, 58, 56), (0, 54, 54), (0b00, 53, 52)]);
        assert!(decode(cond).blocked.is_some(), "conditional zero-offset branch must block");
    }

    /// The offset is signed ONLY when `br_type` (bit 38) is set AND the field's own sign bit
    /// (bit 19) is set (spec F8.2). Both halves matter: without br_type a large offset is a
    /// legitimate long forward jump, and reading it as negative would turn a forward skip into
    /// a backward loop.
    #[test]
    fn branch_offset_is_signed_only_with_br_type_and_sign_bit() {
        let neg = word_bits(&[
            (0x1f, 63, 59), (0b101, 58, 56), (0, 54, 54), (0b00, 53, 52), (1, 38, 38),
            ((1 << 20) - 6, 19, 0),
        ]);
        let ins = decode(neg);
        assert_eq!(ins.op, Op::Branch { rel: -6 });
        assert_eq!(ins.pred, Predicate::IfNotP(0));
        // Same bit pattern in the offset, br_type CLEAR: a raw non-negative long jump.
        let far = word_bits(&[
            (0x1f, 63, 59), (0b101, 58, 56), (0, 54, 54), (0b00, 53, 52),
            ((1 << 20) - 6, 19, 0),
        ]);
        assert_eq!(decode(far).op, Op::Branch { rel: (1 << 20) - 6 });
    }

    /// The two branch words a retail title's menu shaders actually encode, decoded end to end:
    /// a fragment `if p0` skipping 3 words and a vertex `if !p0` skipping 7. Both carry
    /// br_type, a zero `br_op` and no link, so both are plain forward conditional skips.
    #[test]
    fn corpus_branch_words_decode_as_forward_conditional_skips() {
        let frag = decode(0xf900004000000003);
        assert_eq!(frag.op, Op::Branch { rel: 3 });
        assert_eq!(frag.pred, Predicate::IfP(0));
        assert!(frag.blocked.is_none());
        let vert = decode(0xfd00004000000007);
        assert_eq!(vert.op, Op::Branch { rel: 7 });
        assert_eq!(vert.pred, Predicate::IfNotP(0));
        assert!(vert.blocked.is_none());
        // The universal prologue word carries br_op = 2, not 0, and no displacement: still a
        // structural no-op, and specifically NOT reclassified as a branch by this work.
        assert_eq!(decode(0xf800094000000000).op, Op::Nop);
    }

    #[test]
    fn tex_decodes_sampler_coord_and_dest() {
        // 0xE0 tex: 2D (dim field 1), implicit LOD, normal sample. coord src0 = pa
        // (sel=1,ext=0) reg 4 -> pa8; sampler src1 = 5; dest_use_pa=0 -> temp reg 3 -> r6.
        let w = word_bits(&[
            (0x1c, 63, 59), // opcode1 (tex)
            (1, 43, 42),    // dim = 2D
            (0, 41, 40),    // implicit LOD
            (0, 38, 37),    // normal sample
            (0, 50, 50),    // src0 ext = 0
            (1, 34, 34),    // src0 bank sel = 1 -> pa
            (4, 20, 14),    // src0_n (coord)
            (5, 13, 7),     // src1_n (sampler unit)
            (0, 39, 39),    // dest_use_pa = 0 -> temp
            (3, 27, 21),    // dest_n
        ]);
        let ins = decode(w);
        assert_eq!(ins.group, 0x1c);
        assert_eq!(ins.op, Op::Tex { unit: 5, coords: 2, coord_half: false, lod: TexLod::Implicit });
        assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::PrimaryAttr, 8));
        assert_eq!((ins.dest.unwrap().bank, ins.dest.unwrap().index), (Bank::Temp, 6));
        assert!(ins.blocked.is_none() && ins.is_supported(), "plain 2D tex must emit: {:?}", ins.blocked);
    }

    #[test]
    fn pack_float_to_float_is_swizzled_copy() {
        // VPCK F16<-F32 (dest_fmt=5, src_fmt=6): dest is R7 (direct) temp reg 3 -> r3; src1 is
        // R6 (double-register, bit 7 holds the F32 comp0 selector) pa field 2 -> pa4;
        // component selectors -> swizzle; full mask. Emits as a value-preserving copy.
        let w = word_bits(&[
            (0x08, 63, 59), // opcode1 (pack)
            (6, 43, 41),    // src_fmt = F32
            (5, 40, 38),    // dest_fmt = F16
            (0, 33, 32),    // dest bank = temp
            (3, 27, 21),    // dest_n
            (0b10, 31, 30), // src1 bank = pa
            (2, 13, 8),     // src1_n
            (0b1111, 37, 34), // dest mask
            // comp selectors: c0=0(x), c1=1(y), c2=2(z), c3=3(w)
            (1, 17, 16),    // comp_sel_1 = y
            (2, 15, 14),    // comp_sel_2 = z
            (3, 20, 19),    // comp_sel_3 = w
        ]);
        let ins = decode(w);
        assert_eq!(ins.op, Op::Pack { src_half: false });
        assert_eq!((ins.dest.unwrap().bank, ins.dest.unwrap().index), (Bank::Temp, 3));
        assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::PrimaryAttr, 4));
        assert_eq!(ins.srcs[0].swizzle, [0, 1, 2, 3]);
        assert!(ins.blocked.is_none() && ins.is_supported(), "float<->float pack must emit: {:?}", ins.blocked);
        // The two formats are INDEPENDENT fields: this one reads F32 and writes F16, so the
        // source must be read as whole registers and the destination written as packed halves.
        assert!(!ins.source_half_precision(), "src_fmt=F32 reads whole registers");
        assert!(ins.half_precision, "dest_fmt=F16 writes packed halves");
    }

    #[test]
    fn pack_f32_from_f16_reads_its_source_as_halves_not_whole_registers() {
        // The mirror case (VPCK F32<-F16, dest_fmt=6 src_fmt=5) is the one the terrain vertex
        // program uses to widen its computed fog term into an F32 output lane. Reading that
        // source at F32 would take a register holding two F16 halves and interpret the pair as
        // one 32-bit float - a denormal, not the value - so the source width must come from
        // `src_fmt` alone. Note src_fmt is NOT F32 here, so comp0's high selector bit is taken
        // from bit 1 rather than bit 7 (spec B.2 operand layout).
        let w = word_bits(&[
            (0x08, 63, 59),   // opcode1 (pack)
            (5, 43, 41),      // src_fmt = F16
            (6, 40, 38),      // dest_fmt = F32
            (0, 33, 32),      // dest bank = temp
            (3, 27, 21),      // dest_n -> r3
            (0b10, 31, 30),   // src1 bank = pa
            (0, 13, 8),       // src1_n
            (0b0001, 37, 34), // dest mask = x only
        ]);
        let ins = decode(w);
        assert_eq!(ins.op, Op::Pack { src_half: true });
        assert!(ins.source_half_precision(), "src_fmt=F16 reads packed halves");
        assert!(!ins.half_precision, "dest_fmt=F32 writes whole registers");
        let body = crate::wgsl::emit_body(&crate::ir::Shader {
            kind: crate::container::ProgramKind::Vertex,
            instrs: vec![ins],
        })
        .expect("float<->float pack emits");
        assert!(body.contains("unpack2x16float(pa[0])[0]"), "{body}");
        assert!(body.contains("r[3] = bitcast<u32>("), "{body}");
    }

    #[test]
    fn pack_int_conversion_blocks() {
        // VPCK U8<-F32 (dest_fmt=0) changes the value (normalize) -> blocked.
        let w = word_bits(&[(0x08, 63, 59), (6, 43, 41), (0, 40, 38)]);
        assert!(decode(w).blocked.is_some());
    }

    #[test]
    fn bitwise_and_with_immediate_folds_and_emits() {
        // VBW AND (op1=010, op2=0), 32-bit lane, dest temp reg 2, src1 temp reg 3 - both R7
        // (7-bit) fields, so both are direct register numbers. src2 immediate 0x00FF (via
        // IMMEDIATE bank ext), no rotate/invert. Emittable.
        use crate::ir::BitwiseKind;
        let w = word_bits(&[
            (0x0a, 63, 59), // opcode1 (AND/OR group, op1=010)
            (0, 35, 35),    // op2 = 0 -> AND
            (0, 34, 34),    // 32-bit lane
            (0, 33, 32),    // dest bank = temp
            (2, 27, 21),    // dest_n
            (0, 31, 30),    // src1 bank = temp
            (3, 13, 7),     // src1_n
            (1, 48, 48),    // src2 ext = 1
            (2, 29, 28),    // src2 bank sel = 2 -> IMMEDIATE
            (0x7f, 6, 0),   // src2_n = low 7 bits of imm
            (1, 20, 14),    // src2_sel = next 7 bits -> imm = 0x7f | (1<<7) = 0xFF
        ]);
        let ins = decode(w);
        assert_eq!(ins.op, Op::Bitwise { kind: BitwiseKind::And, imm: Some(0xFF) });
        assert_eq!((ins.dest.unwrap().bank, ins.dest.unwrap().index), (Bank::Temp, 2));
        assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::Temp, 3));
        assert_eq!(ins.write_mask, [true, false, false, false]);
        assert!(ins.blocked.is_none() && ins.is_supported(), "imm AND must emit: {:?}", ins.blocked);
    }

    /// Pins the R7-is-direct rule to the REAL instruction words that established it. These are
    /// literal words lifted from captured vertex programs (content-free: only the encoding is
    /// asserted, no game data is embedded beyond the 64-bit words themselves).
    ///
    /// `FOG_PACK` is byte-identical across every captured vertex program that declares a fog
    /// varying: an F16->F32 VPCK into the varyings block's reserved fog lane. Its dest field is
    /// 4. Direct, that is output lane 4 - the reserved slot the container's own output-lane
    /// total accounts for (4 position + 2 reserved + the vo2 texcoord widths), written by
    /// nothing else. Doubled, it would be lane 8, which a texcoord move already writes.
    ///
    /// The two `BITWISE` words come from one vertex whose `primary_reg_count` is 20. Their R7
    /// source fields are 18 and 19 - direct, the z/w components of that program's declared
    /// `VertexColour1@pa16x4`; doubled, pa36/pa38, which the program does not allocate. Their
    /// R7 dest fields 8 and 9 fill the only gap in that program's written output lanes.
    /// EVERY R6 operand path resolves the reserved 60..63 field range to an INTERNAL register,
    /// not to temporaries r120..r127.
    ///
    /// A 6-bit register field addresses in double-register units, so its top four values would
    /// name r120/r122/r124/r126 - registers no captured program allocates (their temp counts are
    /// far lower), which the emitter would read as zero. The ISA reserves those four encodings
    /// for the internal registers instead, and the decoder already applied that rule on the
    /// paths that go through `r6_source_bank_index`; the plain (non-`alt_opt`) source paths, the
    /// R6 destination paths and the `mad` op1 path did not, so the same field decoded two
    /// different ways depending on which operand slot it sat in.
    ///
    /// Pinned by three consecutive real instructions from one vertex program. The `mad` writes
    /// its result to a reserved-field destination, and the next two instructions read that same
    /// field as their op1 - a def-use chain that only closes when both ends name the internal
    /// register. Under the old decode the destination was r120 and the reads were r120 too, so
    /// the chain looked fine in isolation while every internal-register value computed by the
    /// program's real `pack`/`mad` sequence (which the correct path already decoded as `i0`) was
    /// silently dropped on the floor.
    /// The real VTST + KILL pair that forms this title's ALPHA TEST, decoded end to end. The
    /// two instructions sit two apart in four captured fragment programs:
    ///
    /// ```text
    /// #2  tex   T6 <- albedo
    /// #4  vtst  p1 = (T6.w - sa6.w) >= 0
    /// #5  kill  (predicated)
    /// ```
    ///
    /// This pins every field the test group needs at once - the F16 FLOAT family with
    /// `alu_op` VSUB, the POSITIVE+ZERO/OR sub-test pair collapsing to `>=`, `chan_cc`
    /// SELECT3 picking the ALPHA channel, `pdst_n` naming p1, and the double-register scaling
    /// of both sources - because any one of them being wrong makes the pair stop reading as
    /// an alpha test.
    #[test]
    fn real_vtst_and_kill_decode_as_an_alpha_test() {
        const VTST: u64 = 0x4888493530038183;
        const KILL: u64 = 0xf9300406f000070e;

        let t = decode(VTST);
        assert!(t.blocked.is_none(), "{:?}", t.blocked);
        assert_eq!(
            t.op,
            Op::Test {
                alu: TestAlu::Sub,
                cmp: TestCmp::Ge,
                reduce: TestReduce::Channel(3),
                pdst: 1,
                write_back: false,
            }
        );
        assert!(t.half_precision, "prec=0 in the FLOAT family is F16");
        assert_eq!((t.srcs[0].bank, t.srcs[0].index), (Bank::Temp, 6), "src1 field 3, doubled");
        assert_eq!((t.srcs[1].bank, t.srcs[1].index), (Bank::SecondaryAttr, 6), "src2 field 3, doubled");
        assert_eq!(
            t.srcs[1].swizzle,
            [0; 4],
            "src2_vscomp broadcasts the scalar threshold from channel 0, while the reduction              reads src1's channel 3"
        );
        assert!(t.dest.is_none(), "test_wben is clear, so the register destination is inert");

        let k = decode(KILL);
        assert_eq!(k.op, Op::Kill);
        assert!(k.blocked.is_none());
        assert_eq!(
            k.pred,
            Predicate::IfNotP(1),
            "kill runs when the alpha test FAILS - the analyzer ordering of bits[42:41]"
        );
    }

    /// A test whose ALU result feeds the destination too (`test_wben`), and the ANDALL/ORALL
    /// reductions, are decoded rather than silently dropped. Built synthetically because the
    /// corpus only encodes the predicate-only SELECT form.
    #[test]
    fn vtst_write_back_and_all_channel_reductions_decode() {
        // FLOAT/VADD, zero_test=NOTZERO, chan_cc=ANDALL, pdst=2, test_wben=1, dest pa4.
        let w = word_bits(&[
            (0x09, 63, 59),
            (2, 41, 40),  // zero_test = NOTZERO
            (4, 38, 36),  // chan_cc = ANDALL
            (2, 35, 34),  // pdst_n = p2
            (2, 33, 32),  // dest bank = pa
            (4, 27, 21),  // dest_n
            (1, 20, 20),  // test_wben
            (0, 19, 18),  // alu_sel = FLOAT
            (2, 17, 14),  // alu_op = VADD
        ]);
        let ins = decode(w);
        assert!(ins.blocked.is_none(), "{:?}", ins.blocked);
        assert_eq!(
            ins.op,
            Op::Test {
                alu: TestAlu::Add,
                cmp: TestCmp::Ne,
                reduce: TestReduce::AndAll,
                pdst: 2,
                write_back: true,
            }
        );
        assert_eq!(ins.dest.map(|d| (d.bank, d.index)), Some((Bank::PrimaryAttr, 4)));
        assert_eq!(ins.write_mask, [true; 4]);
    }

    /// Group 0x50's `src1_ext` bit selects the shared operand decode's EXTENSION row, not a
    /// different register in the ordinary banks. On the real word below (the only 0x50 with
    /// the bit set in the corpus) the selector is 1 = SPECIAL and the raw register field is 4,
    /// whose `0x40` bit is CLEAR - so the operand is FPCONSTANT[4], the hardware constant
    /// table entry for 2.0f, and the instruction is `pa8 = const4 | 0` (an OR with an
    /// assembled immediate of 0), i.e. a 32-bit move of that constant.
    ///
    /// The two things this pins are that the extension row is consulted at all (before the
    /// fix the field was read as an ordinary bank and the instruction blocked), and that
    /// SPECIAL is NOT double-register scaled - a doubled 4 would select table entry 8, a
    /// different constant.
    #[test]
    fn group_50_extended_src1_resolves_the_hardware_constant_bank() {
        const MOV_CONST: u64 = 0x5083000a61000200;
        let ins = decode(MOV_CONST);
        assert!(ins.blocked.is_none(), "extended-bank src1 must now decode: {:?}", ins.blocked);
        assert!(matches!(ins.op, Op::Bitwise { kind: crate::ir::BitwiseKind::Or, imm: Some(0) }));
        assert_eq!(
            (ins.srcs[0].bank, ins.srcs[0].index),
            (Bank::Constant, 4),
            "src1_ext + selector 1 + field 4 is FPCONSTANT[4], undoubled"
        );
        assert_eq!(ins.dest.as_ref().map(|d| (d.bank, d.index)), Some((Bank::PrimaryAttr, 8)));
    }

    /// The SMLSI decode and the slots-a-repeat-consults rule, pinned in BOTH directions on the
    /// real corpus.
    ///
    /// The captured vertex program that carries SMLSI genuinely REPEATS: its `mov` (group 0x38)
    /// encodes `repeat_count = 3` in bits 45:44, so that one instruction executes four times
    /// with the stepping SMLSI configures. Reading the disassembly as though every instruction
    /// ran once - which is how the surrounding code looks - gets this wrong. What makes that
    /// program's SMLSI inert anyway is narrower and is the fact pinned here: the repeat is an
    /// UNCONDITIONAL VMOV, which reads src1 and nothing else, and every slot other than src0
    /// carries increment 1.
    #[test]
    fn smlsi_slots_a_repeat_consults_decide_whether_it_is_inert() {
        // vert_82c14da0's prologue: SMLSI, then a VMOV with repeat_count 3.
        const SMLSI: u64 = 0xfa10000201014e01;
        const MOV_REPEAT_3: u64 = 0x3880352183080080;
        assert_eq!(bits(MOV_REPEAT_3, 45, 44), 3, "the VMOV really does encode a repeat");
        assert_eq!(
            decode_smlsi(SMLSI),
            [
                SmlsiSlot::Increment(1),  // dest
                SmlsiSlot::Swizzle(0x4e), // src0 - the one slot that varies across the corpus
                SmlsiSlot::Increment(1),  // src1
                SmlsiSlot::Increment(1),  // src2
            ]
        );
        // An unconditional VMOV is `dest = src1`: it has no src0 and no src2 operand, so the
        // one non-default byte in that word cannot reach anything it does.
        assert_eq!(slots_repeats_consult(&[SMLSI, MOV_REPEAT_3]), [true, false, true, false]);

        // Turn the same repeat into the CONDITIONAL form (move_type 1, bits 47:46), which reads
        // src0 as its test - now the swizzle-mode byte is live and the program must stay blocked.
        const MOV_COND_REPEAT: u64 = MOV_REPEAT_3 | (1 << 46);
        assert_eq!(bits(MOV_COND_REPEAT, 45, 44), 3, "still a repeat");
        assert_eq!(slots_repeats_consult(&[SMLSI, MOV_COND_REPEAT]), [true; 4]);

        // The same SMLSI with only non-repeating instructions after it consults nothing at all
        // (real word, instruction #12 of the same program).
        const MOV_ONCE: u64 = 0x38800d0902000f40;
        assert_eq!(bits(MOV_ONCE, 45, 44), 0);
        assert_eq!(slots_repeats_consult(&[SMLSI, MOV_ONCE]), [false; 4]);

        // A group whose repeat encoding is not established leaves the answer unknown, which
        // must read as "might repeat, through every operand" rather than as "safe".
        const UNESTABLISHED_GROUP: u64 = 0xa000_0000_0000_0000; // opcode1 = 0x14, no table
        assert_eq!(opcode1(UNESTABLISHED_GROUP), 0x14);
        assert_eq!(slots_repeats_consult(&[SMLSI, UNESTABLISHED_GROUP]), [true; 4]);

        // The other three distinct SMLSI words the corpus contains, all default on every slot a
        // VMOV repeat reads. The src0 byte is the only one that ever differs.
        for w in [0xfa10000201010e01u64, 0xfa14000001010101, 0xfa1400000101f601] {
            assert!(is_smlsi(w));
            let state = decode_smlsi(w);
            assert_eq!(
                [state[0], state[2], state[3]],
                [SmlsiSlot::Increment(1); 3],
                "{w:#018x} must be the default step on dest/src1/src2"
            );
        }
    }

    /// The SPECIAL row's GLOBAL half DECODES (which register it names is as structural as
    /// which constant its FPCONSTANT half names); what a GLOBAL register CONTAINS is settled
    /// per index by the emitter, so decode leaves `blocked` clear here and the emitter
    /// hard-fails on any index it has not established. The indexed modes still block at
    /// decode - they need RIO6 addressing, so their operand cannot even be named.
    #[test]
    fn group_50_extended_src1_names_the_global_register_and_blocks_indexed_banks() {
        // Same word with the src1 field's 0x40 bit set (field 4 -> 68): SPECIAL -> GLOBAL.
        const MOV_GLOBAL: u64 = 0x5083000a61002200;
        let g = decode(MOV_GLOBAL);
        assert_eq!(bits(MOV_GLOBAL, 13, 7) & 0x40, 0x40, "test word must select the GLOBAL half");
        assert!(g.blocked.is_none(), "GLOBAL is a structural decode: {:?}", g.blocked);
        assert_eq!(
            (g.srcs[0].bank, g.srcs[0].index),
            (Bank::Global, 4),
            "the 0x40 discriminator is cleared from the index, leaving GLOBAL[4]"
        );
        // Selector 0 (bits 31:30 cleared) with the extension bit still set is INDEXED1.
        const MOV_INDEXED: u64 = 0x5083000a21000200;
        assert!(decode(MOV_INDEXED).blocked.is_some_and(|b| b.contains("INDEXED")));
    }

    #[test]
    fn reserved_r6_fields_name_internal_registers_in_every_operand_slot() {
        // dst = i0, sources i1 (reserved field 61) and the constant bank.
        const MUL_TO_I0: u64 = 0x08c11f889f240041;
        // mad: op1 is the reserved field 60 -> i0, read right after the mul above wrote it.
        const MAD_FROM_I0: u64 = 0x00800882f003c65f;
        const MAD_FROM_I0_AGAIN: u64 = 0x00800882e007c6c0;

        let mul = decode(MUL_TO_I0);
        assert_eq!(mul.dest.as_ref().map(|d| (d.bank, d.index)), Some((Bank::Internal, 0)));
        for word in [MAD_FROM_I0, MAD_FROM_I0_AGAIN] {
            let ins = decode(word);
            assert_eq!(
                (ins.srcs[0].bank, ins.srcs[0].index),
                (Bank::Internal, 0),
                "{word:#018x}: mad op1 must resolve the reserved field to an internal register"
            );
            assert!(ins.blocked.is_none(), "{word:#018x} should decode cleanly");
        }
    }

    #[test]
    fn r7_operand_fields_are_direct_register_numbers() {
        const FOG_PACK: u64 = 0x40810b85a0800000;
        const BITWISE_O9: u64 = 0x50c10009a1200980;
        const BITWISE_O8: u64 = 0x50c10009a1000900;

        let fog = decode(FOG_PACK);
        // This real instruction is a VPCK F32<-F16 (src_fmt=5, dest_fmt=6): it WIDENS the F16
        // fog term the preceding half-precision instructions computed into an F32 output lane.
        assert_eq!(fog.op, Op::Pack { src_half: true });
        assert!(!fog.half_precision, "the destination lane is F32");
        let d = fog.dest.unwrap();
        assert_eq!((d.bank, d.index), (Bank::Output, 4), "fog packs into the reserved lane 4");

        for (word, out_lane, pa_reg) in [(BITWISE_O9, 9u8, 19u8), (BITWISE_O8, 8, 18)] {
            let ins = decode(word);
            let d = ins.dest.unwrap();
            assert_eq!((d.bank, d.index), (Bank::Output, out_lane));
            assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::PrimaryAttr, pa_reg));
        }
    }

    /// The counterpart to the test above, and the reason it must not be generalised: group 0x30
    /// puts its operands at the SAME bit positions as 0x40/0x50 but addresses them in
    /// double-register units.
    ///
    /// These three words are consecutive instructions from a real fragment program, and together
    /// they are a vector normalize: `dot` writes the squared length, `rsq` takes its reciprocal
    /// square root in place, and `mul` scales the vector by the result. The `mul` reads register
    /// 4, and the `dot` writes register 4, so the `rsq` between them must also address register
    /// 4 - and its own fields are 2. Under a direct reading the rsq would touch register 2, and
    /// the normalize would silently compute from an unrelated register.
    #[test]
    fn group_30_operand_fields_are_double_register() {
        const DOT_TO_PA4: u64 = 0x1040f086a0847000;
        const RSQ_IN_PLACE: u64 = 0x3020028280400101;
        const MUL_BY_PA4: u64 = 0x10240384a0040002;

        let dot = decode(DOT_TO_PA4);
        assert_eq!((dot.dest.unwrap().bank, dot.dest.unwrap().index), (Bank::PrimaryAttr, 4));
        let mul = decode(MUL_BY_PA4);
        assert!(mul.srcs.iter().any(|s| s.bank == Bank::PrimaryAttr && s.index == 4));

        let rsq = decode(RSQ_IN_PLACE);
        assert_eq!(rsq.op, Op::Rsq);
        assert_eq!((rsq.dest.unwrap().bank, rsq.dest.unwrap().index), (Bank::PrimaryAttr, 4));
        assert_eq!((rsq.srcs[0].bank, rsq.srcs[0].index), (Bank::PrimaryAttr, 4));
    }

    #[test]
    fn bitwise_shr_with_register_source_emits() {
        // VBW SHR (op1=101, op2=0), src2 a plain register (no rotate/invert). Emittable.
        use crate::ir::BitwiseKind;
        let w = word_bits(&[
            (0x0d, 63, 59), // opcode1 (SHR/ASR group, op1=101)
            (0, 35, 35),    // op2 = 0 -> SHR
            (1, 27, 21),    // dest_n
            (2, 13, 7),     // src1_n
            (3, 6, 0),      // src2_n (register)
        ]);
        let ins = decode(w);
        assert_eq!(ins.op, Op::Bitwise { kind: BitwiseKind::Shr, imm: None });
        assert_eq!(ins.srcs.len(), 2);
        assert!(ins.blocked.is_none() && ins.is_supported());
    }

    #[test]
    fn flow_spec_catchall_is_nop() {
        // The real end-of-fragment SPEC word (special=0, category=0b10) is a documented
        // no-op: it must not block whole-shader emit.
        let ins = decode(0xf920000000000000);
        assert_eq!(ins.group, 0x1f);
        assert_eq!(ins.op, Op::Nop);
        assert!(ins.blocked.is_none() && ins.is_supported(), "SPEC no-op must emit");
    }

    #[test]
    fn tex_3d_cube_gives_three_coords_and_emits() {
        // dim field 2 => 3D/cube, 3 coordinate components; emittable (bound-texture type is a
        // bind-time fact).
        let d3 = word_bits(&[(0x1c, 63, 59), (2, 43, 42), (7, 13, 7)]);
        let ins = decode(d3);
        assert_eq!(ins.op, Op::Tex { unit: 7, coords: 3, coord_half: false, lod: TexLod::Implicit });
        assert!(ins.blocked.is_none() && ins.is_supported());
    }

    /// `lod_mode` selects the sample variant (spec E0.4). Bias and explicit-LOD each read one
    /// scalar from src2 and map exactly onto a WGSL sample function, so they decode; only the
    /// GRADIENT form - whose two derivative vectors have no corpus evidence for their component
    /// split - stays blocked, along with the gather/info sub-behaviours.
    #[test]
    fn tex_lod_modes_decode_except_gradient() {
        // sb_mode != 0 (gather/info) -> blocked.
        let gather = word_bits(&[(0x1c, 63, 59), (1, 38, 37)]);
        assert!(decode(gather).blocked.is_some());

        // The real explicit-LOD sample from the livery fragment: lod_mode 2, and src2 field 4
        // doubled to primary-attribute register 8 - the register the `mov .f32` immediately
        // before it writes, which is what pins the operand and its 32-bit width.
        let level = decode(0xe0808a00e09f0584);
        assert!(level.blocked.is_none(), "{:?}", level.blocked);
        assert!(matches!(level.op, Op::Tex { lod: TexLod::Level, .. }));
        assert_eq!(
            (level.srcs[1].bank, level.srcs[1].index),
            (Bank::PrimaryAttr, 8),
            "the LOD operand is src2, double-register scaled like every other SMP operand"
        );

        let bias = decode(word_bits(&[(0x1c, 63, 59), (1, 41, 40)]));
        assert!(matches!(bias.op, Op::Tex { lod: TexLod::Bias, .. }));
        assert_eq!(bias.srcs.len(), 2, "bias reads its scalar from src2");

        // Gradient stays blocked, naming itself.
        let grad = decode(word_bits(&[(0x1c, 63, 59), (3, 41, 40)]));
        assert!(grad.blocked.is_some_and(|b| b.contains("gradient")));
    }

    #[test]
    fn stub_groups_classified_and_lossless() {
        // Every documented group classifies its operation (a fact) and preserves the raw
        // word, even where operand decode is not yet wired. Loads (0xE8) remain a stub.
        let load = decode(word(0xE800_0000 | 0x1234, 0xabcd_ef01));
        assert_eq!(load.group, 0x1d);
        assert_eq!(load.op, Op::Todo("lda32/ldl32/ldt32"));
        assert_eq!(load.raw, word(0xE800_0000 | 0x1234, 0xabcd_ef01));
        assert!(!load.is_supported(), "loads not wired for emit yet");
        assert!(load.is_classified(), "load operation is known");

        // The TEST group (0x48 = opcode1 0x09 = VTST) decodes to a real test operation. This
        // all-zero word has both sub-tests disabled, which names no relation against zero, so
        // it classifies but blocks rather than picking one.
        let vtst = decode(word(0x4800_0000, 0));
        assert_eq!(vtst.group, 0x09);
        assert!(matches!(vtst.op, Op::Test { .. }), "0x09 is the VTST test op");
        assert!(vtst.is_classified());
        assert!(vtst.blocked.is_some_and(|b| b.contains("0x48 VTST")));

        // A documented-illegal group (0xB0 = opcode1 0x16).
        let bad = decode(word(0xB000_0000, 0));
        assert_eq!(bad.op, Op::Illegal);
        assert!(!bad.is_classified());
    }
}
