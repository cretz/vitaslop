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

use crate::ir::{
    Bank, CompareMethod, Instr, Op, Operand, Predicate, SopFactor, SopOp, TestAlu, TestCmp,
    TestReduce, TexLod,
};

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
// F32 masks truncating to the low two channels) is NOT applied to the 0x08/0x10 vector-ALU
// masks, and that is now PROVEN rather than merely unevidenced. Group 0x02 (V16NMAD - the F16
// half of that pair) uses the FULL four-bit mask range in the corpus, bits 1 and 3 included,
// thousands of times in each of three unrelated titles. An encoding that emits `0b0010` for an
// F16 destination cannot be one where only bits 0 and 2 are meaningful.
// (The earlier note here recorded only that wiring it in changed no dataflow corroboration -
// temp 195/265, output 5/8, internal 660/663 either way. True, and much weaker than the above.)
// It was once suspected of causing the vertex-to-fragment varying mismatch; that turned out to
// be a linker-model bug (the stages are matched by USAGE, see `link::plan_varyings`) and the
// masks were never implicated. Do NOT wire it in to make one shader's numbers match.
//
// The same note used to add "and group-0x38 moves already decode within the two-channel form the
// transform would produce". That part was WRONG and is now [`write_mask_f16`]: a group-0x38 F16
// move with raw mask `0b0001` writes ONE half-lane under the raw reading and the channel PAIR
// under A.6, and the corpus settles it in favour of the pair - see that function.

/// The A.6 destination write-mask transform, for the case the corpus supports: an **F16**
/// destination in a **GPR** bank, where each meaningful raw bit covers a channel PAIR.
///
/// The spec states the transform in four cases; this implements the two that decide anything
/// here. FPINTERNAL keeps the raw mask (the internal registers are not half-packed), an F16
/// destination in an exotic bank keeps the raw mask, and the F32 case is NOT implemented -
/// the spec's `mask & 0b0011` there would turn every four-component F32 write (a clip position,
/// for one) into a two-component one, which the whole corpus refutes immediately.
///
/// MEASURED, and this is what turned a "deliberately not applied" note into code. Under the raw
/// reading a title's composite vertex program copies its screen-space UV OFFSET with
/// `mov.f16 sa[8].x, sa[0].zw` - one half-lane - and its primary program then reads `sa[8].xy`,
/// so the V offset is a register nothing wrote. The pair expansion makes the move write exactly
/// the two halves the read consumes, and the def-use closes. Corpus-wide, the same change takes
/// the number of vertex programs that fail to write exactly their declared output lanes from
/// 18/191 to fewer, while leaving two other titles' corpora clean.
#[inline]
fn write_mask_f16(raw: u32, dest: Option<&Operand>, is_f16: bool) -> [bool; 4] {
    let gpr = matches!(
        dest.map(|d| d.bank),
        Some(Bank::PrimaryAttr | Bank::SecondaryAttr | Bank::Output | Bank::Temp)
    );
    if !is_f16 || !gpr {
        return write_mask4(raw);
    }
    let mut out = 0u32;
    if raw & 0b0001 != 0 {
        out |= 0b0011;
    }
    if raw & 0b0100 != 0 {
        out |= 0b1100;
    }
    write_mask4(out)
}

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
        0x0f => decode_grp_test_mask(word),
        0x0a | 0x0b | 0x0c | 0x0d => decode_grp_bitwise(word, op1),
        0x10 => decode_grp_sop2(word),
        0x12 => decode_grp_sop2m(word),
        0x14 => decode_grp_i16mad(word),
        0x15 => decode_grp_imad32(word),
        0x1a => decode_grp_imad32_step(word),
        0x1c => decode_grp_tex(word),
        0x1d => decode_grp_mem_load(word),
        0x1f => decode_grp_flow(word),
        _ => classified_stub(word, op1, hi, lo),
    }
}


/// Decode a group-0x80 SOP2 (opcode1 0x10), the write-mask-less sibling of
/// [`decode_grp_sop2m`] - **as the FRAGMENT-EPILOGUE form only**, which is the one this
/// corpus establishes and the only one it can.
///
/// # What is established, and by what
/// The OPERAND grammar is SOP2M's, and the def-use chain confirms it rather than assuming it.
/// Five fragment programs from one title end in the identical two-instruction epilogue:
///
/// ```text
///   pack.unorm8  pa[0] <- pa[0]     (the F16 colour converted to four bytes)
///   <this>                          (raw 0x809080d990000000, or ...c19... in one program)
/// ```
///
/// Decoded through SOP2M's operand fields this instruction reads `src1 = pa[0]` - exactly the
/// register the pack before it wrote - and writes `dest = o[0]`, the OUTPUT bank, which is
/// where a native fragment colour goes. Two independent fields landing on the two registers
/// the neighbouring instructions name is the same closure that settled SOP2M.
///
/// # What is NOT established, and why this still emits something
/// The COEFFICIENT and colour/alpha-op fields are not. SOP2M spends bits 46:43 on its write
/// mask and this group does not, so its freed bits carry something unknown and the fields
/// around them cannot be assumed to sit where SOP2M puts them - `stub_reason` said exactly
/// this. Read through SOP2M's table these words come out as
/// `dest = a*src1 - (1-a)*src2`, and **`src2` names `o[0]`, a register not one of the five
/// programs ever writes**. A compiler does not emit a term over its own uninitialised output
/// register five times, so whatever the fields mean, the second term's coefficient is zero
/// here and the instruction is a COPY of the packed colour into the output bank. That is what
/// is emitted ([`Op::CopyFx8`]).
///
/// So the guard below pins every field that is not established and lets the operand fields -
/// the ones the chain confirms - vary freely. A group-0x80 word that differs anywhere else is
/// a form this evidence says nothing about and it BLOCKS, naming itself. Emitting a copy for
/// one of those would be the confident wrong picture this whole family's refusal exists to
/// avoid.
///
/// # A second title carries the same idiom, one bit apart - AND ITS OWN CHAIN NOW SETTLES IT
/// Another title's single group-0x80 word is `0x809082dd90000000`. Field by field it is the
/// word above with ONE difference, 42:41 (0 -> 1), the position SOP2M reads as its alpha op;
/// the operands are identical - `dest = o[0]`, `src1 = pa[0]`, `src2 = o[0]`. Two unrelated
/// titles emitting one epilogue that differs in one field is what a shared compiler back end
/// looks like, and this guard names each pinned field separately so exactly one of them can be
/// widened when the evidence arrives.
///
/// It has. That program is TWO instructions - `Nop`, then this - with a PDS-prefetched sample
/// in `pa[0]` and a `g_TexSampler` on unit 0, so it is the same epilogue with the pack absent
/// because the prefetch already delivers bytes. **Its `o[0]` is written by nothing at all**,
/// which is the same closure that settled the first title's word and is stronger here: with
/// only a `Nop` before it there is no instruction that COULD have written it. So whatever
/// 42:41 selects, its term stands over an uninitialised output register, and the instruction
/// is a copy.
///
/// What that argument does NOT establish, and it is worth naming because it is where a wrong
/// picture would come from: if 42:41 turned out to select an alpha op that reads no source at
/// all (a "write 1.0" would be the obvious one), a copy carries the texel's alpha where the
/// hardware writes an opaque one. The oracle for that is the title's own frame, where its
/// alpha-blended sprites either composite or do not.
///
/// # A SWAPPED shape, settled by two byte-identical programs
/// `0x8190002160040000` is the same epilogue with the two source slots the other way round:
/// `src1` names `o[0]` and `src2` names `pa[0]`, where the words above have it `src1 = pa[0]`,
/// `src2 = o[0]`. Its `mod1`/`mod2`/`sel1`/`sel2` all differ, so nothing about the coefficient
/// fields carries over and it is pinned separately.
///
/// **What settles it is a PAIR of programs, not a chain.** One title registers this shader
/// twice: `frag_81a7f590` and `frag_81a7f798`, 216 bytes each, and they differ in exactly two
/// places - the container's header HASH, and this one instruction word. Same parameter table,
/// same varyings block, same `Nop`, same `pack.unorm8 pa[0] <- pa[0]`. Both are created with a
/// **NULL `blendInfo`** (measured: `gxm blend: fragment program 0x81a7f590 ... blends=false`
/// and the same line for `0x81a7f798`), so the difference is not the ROP's either. Two
/// encodings of one program, and the first of them is already read as a copy of the packed
/// colour into the colour register.
///
/// So the source is taken from the slot that does NOT name the output bank - which is the same
/// closure as above, stated over the operand slots instead of over one of them. Every other
/// field of this shape is pinned to the single observed encoding, `20:14 = 16` included,
/// because what it selects is not established either.
///
/// The risk, named: if the differing coefficient fields make this variant scale or premultiply
/// what it copies, a copy is wrong by exactly that factor. It would show as the title's 2D UI
/// being uniformly too bright or too dark against the same shader's other variant, which is on
/// screen beside it.
fn decode_grp_sop2(word: u64) -> Instr {
    let mut blocked: Option<&'static str> = None;
    // Every field the epilogue evidence does NOT establish, pinned to the value it has in the
    // two observed encodings. `sel2` (37:35) is the one that differs between them.
    let established = bits(word, 58, 57) == 0        // no predicate
        && bits(word, 56, 56) == 0                   // mod1
        && bits(word, 53, 52) == 1                   // "cop" under SOP2M's reading
        && bits(word, 51, 51) == 0                   // no destination bank extension
        && bits(word, 49, 49) == 0                   // no src1 bank extension
        && bits(word, 48, 48) == 0                   // no src2 bank extension
        && bits(word, 47, 47) == 1                   // mod2
        && bits(word, 46, 43) == 0                   // the bits SOP2M spends on its write mask
        && matches!(bits(word, 42, 41), 0 | 1)       // "aop" - the two observed values
        && bits(word, 40, 38) == 3                   // "sel1"
        && matches!(bits(word, 37, 35), 0 | 3); // "sel2" - the two observed values
    // >>> THE SAME EPILOGUE WITH ITS TWO SOURCE SLOTS THE OTHER WAY ROUND. See the SWAPPED
    // section of this function's docs: one title carries `frag_81a7f590` and `frag_81a7f798`,
    // 216 bytes each, byte-identical but for the header hash and THIS WORD - so the two
    // programs compute the same thing and the first of them is already read as a copy. Every
    // field of this shape is pinned to the one observed encoding, and the SOURCE is taken from
    // the other operand slot.
    let swapped = bits(word, 58, 57) == 0            // no predicate
        && bits(word, 56, 56) == 1                   // mod1 - set here, clear in the shape above
        && bits(word, 53, 52) == 1                   // "cop", as above
        && bits(word, 51, 51) == 0                   // no destination bank extension
        && bits(word, 49, 49) == 0                   // no src1 bank extension
        && bits(word, 48, 48) == 0                   // no src2 bank extension
        && bits(word, 47, 47) == 0                   // mod2 - clear here, set in the shape above
        && bits(word, 46, 43) == 0                   // the bits SOP2M spends on its write mask
        && bits(word, 42, 41) == 0                   // "aop"
        && bits(word, 40, 38) == 0                   // "sel1"
        && bits(word, 37, 35) == 4                   // "sel2"
        && bits(word, 29, 28) == 2                   // src2 bank select - the PA bank
        // The src2 upper field, whatever it selects. 16 is the pair-established encoding;
        // 20 (bit 16 set as well) is `0x8190002160050000`, the epilogue of one title's lit,
        // fogged world material (`velvet_fragment_cdb40b4ac29bee79` in its corpus), which
        // computes its own alpha into `pa[0].w` two instructions earlier and packs the whole
        // register to unorm8 before this word - so the copy reading is the one that keeps
        // what the program computed. Which selector bit 16 is has NOT been established: if it
        // scales or drops the alpha, those materials come out uniformly wrong in alpha, and the
        // frame it first appears in is the title's first 3D scene. Refusing it stops that
        // title at its first scene instead.
        && matches!(bits(word, 20, 14), 16 | 20);
    if !established && !swapped {
        blocked = blocked.or(Some(
            "0x80 SOP2 in a form outside the fragment epilogue this corpus establishes - its \
             coefficient and op fields are not read, only pinned (see `decode_grp_sop2`)",
        ));
    }

    // The operand grammar, which the chain above confirms. The SOURCE is whichever slot does
    // NOT name the output bank: that is the whole content of the def-use closure this group is
    // read through, and it is what makes the swapped shape the same instruction rather than a
    // second reading of it.
    let (s1_bank, s1_index) = r7_source_bank_index(bits(word, 31, 30) as u8, bits(word, 13, 7));
    let src1 = if swapped {
        let sel = bits(word, 29, 28) as u8;
        let (bank, index) = r7_source_bank_index(sel, bits(word, 6, 0));
        Operand::plain(bank, index, sel)
    } else {
        Operand::plain(s1_bank, s1_index, bits(word, 31, 30) as u8)
    };
    let dest_sel = bits(word, 33, 32) as u8;
    let dest = match r7_dest_bank_index(dest_sel, bits(word, 27, 21)) {
        Some((b, i)) => Some(Operand::plain(b, i, dest_sel)),
        None => {
            blocked = blocked.or(Some("0x80 SOP2 destination in index mode"));
            None
        }
    };

    Instr {
        op: Op::CopyFx8,
        // The predicate field is pinned to 0 by the guard above, which is the unpredicated
        // encoding SOP2M reads the same way.
        pred: short_predicate(bits(word, 58, 57)),
        dest,
        // No write mask field: the instruction writes the whole register, all four bytes.
        write_mask: [true, true, true, true],
        srcs: vec![src1],
        // Neither side is a float view - both are the packed-byte register. `half_precision`
        // describes a float precision and means nothing here; the emitter reads
        // `Op::CopyFx8` instead.
        half_precision: false,
        raw: word,
        group: 0x80,
        blocked,
    }
}

/// Decode a group-0x90 SOP2M, the 8-BIT FIXED-POINT SUM-OF-PRODUCTS combiner (opcode1 0x12).
///
/// | field | bits | | field | bits |
/// |---|---|---|---|---|
/// | `pred` (2-bit short form) | 58:57 | | `wmask` | 46:43 |
/// | `mod1` | 56 | | `aop` (alpha op) | 42:41 |
/// | `cop` (colour op) | 53:52 | | `sel1` / `sel2` | 40:38 / 37:35 |
/// | `destbankext` | 51 | | `destbank` | 33:32 |
/// | `src1bankext` / `src2bankext` | 49 / 48 | | `src1bank` / `src2bank` | 31:30 / 29:28 |
/// | `mod2` | 47 | | `destnum` | 27:21 |
/// | | | | `src1num` / `src2num` | 13:7 / 6:0 |
///
/// # The write mask is ROTATED, and it is the one field that is easy to get silently wrong
/// The raw field carries ALPHA in bit 0 and RGB in bits 3:1. Rotating alpha up to bit 3 gives
/// the ordinary `[x, y, z, w]` order every other group uses. Reading it unrotated writes the
/// wrong channel with no other symptom.
///
/// # What is NOT modelled, and blocks rather than guessing
/// Selector values 1, 4 and 5 (the reference material establishes 0, 2, 3, 6 and 7 only), a
/// destination bank extension (no corpus program sets it), and the `sel`/`mod` combination is
/// otherwise taken exactly as documented. [[vitaslop-fx8-family-is-a-wall]] recorded this
/// whole family as unreadable from four sources; what changed is the OPERAND grammar and the
/// fact that the selector picks a COEFFICIENT rather than an operand - see [`SopFactor`].
///
/// # The corpus check that says the layout is right
/// Five captured fragment programs carry this instruction, always in the alpha-test macro,
/// and in every one the destination this decode produces is EXACTLY the register the next two
/// instructions (an OR that zeroes it under the inverse predicate, and a VTST that compares it
/// against zero) write and read. Two different destinations across the five programs, both
/// matching. A wrong bank or number field would have to be wrong in the same direction as an
/// independently-decoded neighbour to produce that.
fn decode_grp_sop2m(word: u64) -> Instr {
    let mut blocked: Option<&'static str> = None;
    let pred = short_predicate(bits(word, 58, 57));

    let sop_op = |v: u32| match v {
        0 => SopOp::Add,
        1 => SopOp::Sub,
        2 => SopOp::Min,
        _ => SopOp::Max,
    };
    let color = sop_op(bits(word, 53, 52));
    let alpha = sop_op(bits(word, 42, 41));

    let mut factor = |v: u32| match v {
        0 => SopFactor::Zero,
        2 => SopFactor::Src1Color,
        3 => SopFactor::Src1Alpha,
        6 => SopFactor::Src2Color,
        7 => SopFactor::Src2Alpha,
        _ => {
            // 1, 4 and 5 are absent from every source consulted. A combiner coefficient that
            // is guessed wrong scales a whole term, so this blocks.
            blocked = blocked.or(Some("0x90 SOP2M source selector 1/4/5 not established"));
            SopFactor::Zero
        }
    };
    let f1 = factor(bits(word, 40, 38));
    let f2 = factor(bits(word, 37, 35));

    // Sources. An 8-bit type is never double-register scaled, so the operand's number is a
    // direct register index, exactly as the bitwise family's is.
    let mut source = |bank_sel: u8, field_val: u32, ext: bool| -> Operand {
        if ext {
            // The IMMEDIATE row is assembled per group and this layout carries no extended
            // immediate fields, so the operand's own number IS the literal - the same rule the
            // TEST group states.
            if bank_sel & 3 == 2 {
                return Operand::plain(Bank::Immediate, (field_val & 0x7f) as u8, bank_sel);
            }
            match ext_source(bank_sel, field_val, true) {
                Ok(o) => return o,
                Err(why) => {
                    blocked = blocked.or(Some(why));
                    return Operand::plain(Bank::Temp, 0, bank_sel);
                }
            }
        }
        let (bank, index) = r7_source_bank_index(bank_sel, field_val);
        Operand::plain(bank, index, bank_sel)
    };
    let s1 = source(bits(word, 31, 30) as u8, bits(word, 13, 7), bits(word, 49, 49) != 0);
    let s2 = source(bits(word, 29, 28) as u8, bits(word, 6, 0), bits(word, 48, 48) != 0);

    // Destination. The extension bit selects a different bank row that no corpus program uses,
    // so it is reported rather than resolved from an unchecked table.
    if bits(word, 51, 51) != 0 {
        blocked = blocked.or(Some("0x90 SOP2M destination bank extension not established"));
    }
    let dest_sel = bits(word, 33, 32) as u8;
    let dest = match r7_dest_bank_index(dest_sel, bits(word, 27, 21)) {
        Some((b, i)) => Some(Operand::plain(b, i, dest_sel)),
        None => {
            blocked = blocked.or(Some("0x90 SOP2M destination in index mode"));
            None
        }
    };

    // ALPHA IN BIT 0, RGB IN BITS 3:1 - rotate alpha to the top so the mask is [x,y,z,w].
    let raw_mask = bits(word, 46, 43);
    let rotated = ((raw_mask & 0b1110) >> 1) | ((raw_mask & 1) << 3);
    let write_mask = [
        rotated & 1 != 0,
        rotated & 2 != 0,
        rotated & 4 != 0,
        rotated & 8 != 0,
    ];

    Instr {
        op: Op::Sop2 {
            color,
            alpha,
            f1,
            f1_complement: bits(word, 56, 56) != 0,
            f2,
            f2_complement: bits(word, 47, 47) != 0,
        },
        pred,
        dest,
        write_mask,
        srcs: vec![s1, s2],
        // The register file this instruction sees is four 8-bit channels, which is neither of
        // the two float widths `half_precision` selects between. The emitter reads the data
        // type off the OP, so this stays false rather than claiming an F16 read.
        half_precision: false,
        raw: word,
        group: 0x12,
        blocked,
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
        // The BITWISE family's SHIFT LEFT. Three routes agree and none of them is a guess:
        //
        // * `usse-spec-test-ops.md` gives the family's ordering as 0=AND, 1=OR, 2=XOR, 3=SHL,
        //   and marks 4..7 INFERRED.
        // * `usse-spec-test-mask-forms.md` reaches the same table from a different source and
        //   CORRECTS only 4..7 (ROL=5, a gap at 6, ASR=7). The two disagree about nothing at
        //   or below 3, so the value in hand does not depend on which of them is right.
        // * The IDIOM refutes the only competing reading that could be constructed - that the
        //   TEST group reuses the separate VBW group's `op1` numbering, where 3 is XOR. The
        //   two corpus instances (`frag_7089f16e34be693f` #31 and #34) are the two SIDES of a
        //   facing select: a VTSTMSK writes `0x0000FFFF`-or-zero from `GLOBAL[16]` (EQ for one,
        //   NE for the other), the very next instruction tests THAT register against the inline
        //   immediate 31, and a `mov IfP(0)` follows. Under XOR both tests are `(mask ^ 31) > 0`
        //   - unconditionally TRUE for either mask value - so both movs would always fire, the
        //   second would always win, and the two complementary VTSTMSKs above them would be
        //   dead code the compiler emitted for nothing. Under SHL each test is `(mask << 31)`,
        //   which keeps bit 0 of the mask and makes exactly one side fire. That is the same
        //   two-sided facing select the `(1, 10)` arm below documents for other titles.
        //
        // The shift amount is required to be an inline immediate below 32 (checked once the
        // operands are decoded, further down). WGSL leaves a shift of 32 or more indeterminate
        // and no source we hold states what the device does there, so that case is REFUSED
        // rather than masked into something plausible.
        //
        // WHAT WOULD REFUTE THIS, named here rather than left for a later reader to notice: an
        // instance whose src1 is not a value this program's own VTSTMSK just wrote, or whose
        // shift amount is not 31 - either one puts the op back in a domain where SHL and XOR
        // give different answers and the idiom no longer decides between them.
        (3, 3) => (TestAlu::BitShl, false, false),
        // The 8-BIT family (alu_sel 2). Its operands are read as four 8-bit unsigned-
        // normalised channels whatever `prec` says - the precision bit selects between the two
        // FLOAT widths and has no meaning once the family is an integer one. This is the arm
        // the corpus's alpha-test macro takes, always as the subtract that turns a comparison
        // into a test against zero.
        (2, 8) => (TestAlu::Fx8Sub, false, false),
        // The INT16 family (`alu_sel` 1) at its two HIGH op numbers. The reference states that
        // the integer families "reuse the family's integer add/sub/mul/etc. op numbering (e.g.
        // a signed 32-bit subtract at the high alu_op values)", and 14 is VSUB in the FLOAT
        // family's own table - so the two high values are the family's subtract pair. Which of
        // the two is which (signed/unsigned, or the two widths `prec` also selects between) is
        // NOT established and is deliberately not claimed: both are emitted as an integer
        // subtract, which is the same expression either way.
        //
        // Every occurrence in the corpus is the same idiom - a bitwise AND of a uniform's
        // branch bits, then this test for NOT-ZERO against the program's own literal-zero
        // register - where the subtract's sign and operand order cannot change the answer at
        // all. A future case where they could is one this comment is here to make visible.
        (1, 14 | 15) => (TestAlu::IntSub, false, false),
        // The INT16 family at op 10, which is where a THIRD title's whole lit-material family
        // sits - `frag_81f43190` and fourteen siblings, the shaders that draw its terrain,
        // characters and props. Every one of them is the same two-instruction idiom and nothing
        // else:
        //
        //   vtst  <- GLOBAL[16], SA[k]   cmp EQ   -> p0     ; then a `mov IfNotP(0)`
        //   vtst  <- GLOBAL[16], SA[k]   cmp NE   -> p0     ; then a `mov IfNotP(0)`
        //
        // - a two-sided select on the FACING bit. Two facts close it inside that domain without
        // establishing the integer family's op numbering, which no source we hold states:
        //
        // * `GLOBAL[16]` is the front-facing bit and the emitter materialises it as `select(0u,
        //   1u, front_facing)`, so src1 is 0 or 1 and NEVER negative.
        // * `SA[k]` is the program's own literal ZERO (`frag_81f43190`: `literal SA[39] =
        //   0x00000000`). The operand prints as `SA[78]` under a float decode only because that
        //   decode doubles the register number; this family is not double-register.
        //
        // On {0,1} against zero, `a - b`, `a + b` and `max(a, b)` all test the same predicate,
        // and 10 is VMAX in the FLOAT family's own numbering that the reference says the integer
        // families reuse - so every reading that numbering allows gives the same answer here.
        // What would REFUTE it: an instance of `(1, 10)` whose src2 is not the literal zero, or
        // whose src1 is not the facing bit. That is a case this arm must not silently absorb, so
        // it is named here rather than left to a future reader to notice.
        //
        // Leaving it blocked is not the safe option it looks like. The pair then falls back to
        // the fixed-function approximation, which paints this title's world FLAT - a black
        // course at address and a flat green one at the ball - with no error anywhere.
        (1, 10) => (TestAlu::IntSub, false, false),
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
            match ext_source(bank_sel, field_val, false) {
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

    // A SHIFT's amount has to be known to be in range. WGSL leaves a shift of 32 or more
    // indeterminate, and what the device does there is not established by either spec source,
    // so anything but an inline immediate below 32 is refused instead of masked.
    if matches!(alu, TestAlu::BitShl)
        && !matches!(s2.bank, Bank::Immediate if (s2.index as u32) < 32)
    {
        blocked = blocked.or(Some(
            "0x48 VTST shift-left whose amount is not an inline immediate below 32 - a shift of \
             32 or more is indeterminate in WGSL and the device's behaviour is not established",
        ));
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

/// Decode a group-0x78 VTSTMSK (test -> per-channel MASK). The layout is the TEST group's,
/// with two slots swapped, and it is a fact from the same distilled TEST-group spec that
/// established [`decode_grp_test`] (section T.2):
///
/// | field | bits | | field | bits |
/// |---|---|---|---|---|
/// | `pred` | 58:56 | | `tst_mask_type` | 37:36 |
/// | `test_flag_2` | 50 | | `dest_bank` | 33:32 |
/// | `src1_ext` / `src2_ext` | 49 / 48 | | `src1_bank` / `src2_bank` | 31:30 / 29:28 |
/// | `prec` | 47 | | `dest_n` | 27:21 |
/// | `src2_vscomp` | 46 | | `test_wben` | 20 |
/// | `rpt_count` | 45:44 | | `alu_sel` / `alu_op` | 19:18 / 17:14 |
/// | `sign_test` / `zero_test` | 43:42 / 41:40 | | `src1_n` / `src2_n` | 13:7 / 6:0 |
/// | `test_crcomb_and` | 39 | | | |
///
/// It computes the same `alu(src1, src2)` per channel and the same test against zero as its
/// VTST sibling, and differs only in where the four booleans go: VTST reduces them to one
/// predicate bit, this writes one value per channel into a general register. There is no
/// `chan_cc` and no `pdst_n`.
///
/// # What the corpus establishes, and what is therefore refused
/// Group 0x78 occurs THREE times in the whole corpus and every occurrence is the same word - a
/// shadow filter's per-texel depth comparison in three of a golf title's world fragment
/// programs. So the encoding this decodes is exactly one point in the space, and every field
/// value away from it is BLOCKED rather than extrapolated:
///
/// * `tst_mask_type` other than NUMERIC. The numeric form writes `1.0` / `0.0` per channel,
///   which is what the consumer needs (the very next instruction dots the mask against the
///   sample's four bilinear coefficients, i.e. averages the four comparisons). The 8-bit and
///   precision-width mask forms write a BIT PATTERN whose width rule is not established, and a
///   wrong one there is a shadow that is either always on or always off.
/// * `test_flag_2` (bit 50) set. VTST puts `src1_neg` there and this group does not; what the
///   flag feeds is explicitly open in the reference.
/// * `test_wben` clear. With no predicate destination and the write-back disabled the
///   instruction would write nothing at all, which is not a thing a compiler emits - so a clear
///   bit means the field means something this does not model.
///
/// Note what `test_wben` does NOT decide here: the register write itself. Both readings of the
/// bit that survive - "enable the destination write" and VTST's "ALSO write the raw ALU result"
/// - agree that the destination is written when it is set, and the corpus's consumer requires
/// the MASK to be what lands there.
fn decode_grp_test_mask(word: u64) -> Instr {
    let mut blocked: Option<&'static str> = None;
    let predicate_raw = bits(word, 58, 56);
    let pred = ext_predicate(predicate_raw);
    if matches!(pred, Predicate::Raw(_)) {
        blocked = blocked.or(Some("PN (per-instance) predicate depends on repeat state - not modeled"));
    }
    // The mask FORM. Checked against the ALU family below, because what this field selects is
    // the FORMAT OF THE WRITTEN VALUE and the format's width comes from the family - see the
    // `mask_type` gate after the ALU table.
    let mask_type = bits(word, 37, 36);
    if bits(word, 50, 50) != 0 {
        blocked = blocked.or(Some("0x78 VTSTMSK test_flag_2 set - what it feeds is not established"));
    }
    if bits(word, 20, 20) == 0 {
        blocked = blocked.or(Some(
            "0x78 VTSTMSK with test_wben clear: the instruction would write neither a predicate \
             nor a register",
        ));
    }

    // The ALU whose result is tested, and the relation it is tested by: the same two tables
    // VTST reads, from the same fields.
    let alu_sel = bits(word, 19, 18);
    let alu_op = bits(word, 17, 14);
    let prec = bits(word, 47, 47);
    let (alu, half_precision, float_ty) = match (alu_sel, alu_op) {
        (0, 2) => (TestAlu::Add, prec == 0, true),
        (0, 13) => (TestAlu::Mul, prec == 0, true),
        (0, 14) => (TestAlu::Sub, prec == 0, true),
        // >>> THE 16/32-BIT INTEGER FAMILY'S UNSIGNED 16-BIT SUBTRACT.
        //
        // This is the one non-float VTSTMSK that has ever been seen: the word
        // `0x7802019271f6a839`, which panicked a user's run several holes into a round. Every
        // part of it is now sourced rather than guessed - see the `mask_type` gate below for
        // the written value, and note the two things that make it translatable at all:
        //
        //   * `float_ty = false`, so the operands are NOT doubled. That was the one open
        //     contradiction between the two clean-room passes, and the CORPUS settled it: 12
        //     decisive cases over 4 programs and two unrelated titles where a non-float TEST
        //     operand stays inside its program's declared register file only if it is not
        //     doubled, and none the other way (`test_group_operand_numbering_evidence`).
        //   * The operation only has to be right about ADD vs SUB vs MUL, because the relation
        //     is `== 0` and `a - b == 0` iff `a == b` at any width and either signedness. Op 10
        //     being the unsigned 16-bit subtract is corroborated by the vendor's own hardware
        //     header, which pins indices 14/15 of this family as the 32-bit subtracts and so
        //     anchors the table numerically.
        //
        // `half_precision` is false: this family's operands are raw lanes, not F16 halves, and
        // the emitter must not put them through a float view [[TestAlu::IntSub16U]].
        //
        // >>> THIS IS THE MASK-WRITING SIBLING OF THE VTST `(1, 10)` ARM ABOVE, and the two
        // should be read together. That arm decodes the SAME operation with the same operand
        // shape - `GLOBAL[16]` against an SA register, EQ then NE - for a third title's lit
        // materials, and it reached `float_ty = false` independently, by observing that the
        // register number only looks like `SA[78]` under a float decode BECAUSE that decode
        // doubles. The corpus census now says the same thing from a third direction. Three
        // independent routes to "this family is not double-register" is what makes the operand
        // numbering settled rather than merely chosen.
        //
        // >>> WHAT WOULD REFUTE THIS ARM, named here rather than left for a reader to notice,
        // exactly as the VTST arm names its own: an instance whose src2 is a value wider than
        // 16 bits AND not the program's literal zero. The VTST arm rests on src2 being that
        // zero, where subtract, add and max all test the same predicate; this arm does NOT
        // need that, because the vendor's own header anchors the family's table numerically
        // (indices 14/15 are its 32-bit subtracts, added later on the vector core), which makes
        // index 10 the unsigned 16-bit subtract at table level. But if such an instance ever
        // appears and renders wrong, the op numbering is where to look first - and note that
        // the VTST arm models the same op at 32 bits, which is harmless in ITS {0,1}-against-
        // zero domain and would not be if that domain widened.
        (1, 10) => (TestAlu::IntSub16U, false, false),
        _ => {
            blocked = blocked.or(Some("0x78 VTSTMSK ALU family/op not modeled"));
            (TestAlu::Sub, prec == 0, true)
        }
    };
    // >>> WHAT THE MASK FORM WRITES, AND WHY ONLY THESE TWO COMBINATIONS ARE ALLOWED.
    //
    // The field is a FORMAT SELECTOR for the written value - a family of "all-ones of some
    // width" forms plus a numeric one - which is the vendor's own naming and refutes the
    // earlier reading that it merely chose a channel count. The two available readings agree
    // on the numbers for some combinations and DIVERGE for others, so only the combinations
    // where they coincide can be translated; a divergence is a shadow or a facing term that is
    // silently always-on or always-off.
    //
    //   * NUMERIC (2) with a FLOAT family: 1.0 / 0.0. The corpus's three words, and what the
    //     consumer there requires (the next instruction dots the mask against four bilinear
    //     coefficients, i.e. averages the comparisons).
    //   * PRECISION-MASK (1) with the UNSIGNED 16-BIT family: 0xFFFF / 0x0000. "A mask at the
    //     ALU precision" (vendor naming) and "all-ones at the type width" (the emulator rule)
    //     are DIFFERENT RULES that land on the SAME PAIR OF NUMBERS here. That is what makes
    //     this translatable without knowing which rule is the true one - the answer does not
    //     depend on the choice.
    //
    // Still refused, because there the two rules give different numbers: the 8-bit-mask form
    // (0) with a 16- or 32-bit type (0xFF against 0xFFFF/0xFFFFFFFF), the numeric form (2) with
    // an INTEGER family (all-ones against 1), and every signed integer type (one reading writes
    // maximum-positive and nothing corroborates it). Value 3 is named "reserved".
    let mask_ok = match (mask_type, alu) {
        (2, TestAlu::Add | TestAlu::Sub | TestAlu::Mul) => true,
        (1, TestAlu::IntSub16U) => true,
        _ => false,
    };
    if !mask_ok {
        blocked = blocked.or(Some(
            "0x78 VTSTMSK: this (mask type, ALU family) pair is one where the two readings of \
             what a mask writes DISAGREE - only numeric-with-float and precision-with-u16 are \
             established",
        ));
    }
    let cmp = match (bits(word, 43, 42), bits(word, 41, 40), bits(word, 39, 39)) {
        (0, 1, _) => TestCmp::Eq,
        (0, 2, _) => TestCmp::Ne,
        (1, 0, _) => TestCmp::Lt,
        (2, 0, _) => TestCmp::Gt,
        (1, 1, 0) => TestCmp::Le,
        (2, 1, 0) => TestCmp::Ge,
        (1, 2, 1) => TestCmp::Lt,
        (2, 2, 1) => TestCmp::Gt,
        _ => {
            blocked = blocked.or(Some("0x78 VTSTMSK sign/zero sub-test combination not modeled"));
            TestCmp::Ne
        }
    };

    let mut source = |bank_sel: u8, field_val: u32, ext: bool| -> Operand {
        if ext {
            if bank_sel & 3 == 2 {
                return Operand::plain(Bank::Immediate, (field_val & 0x7f) as u8, bank_sel);
            }
            match ext_source(bank_sel, field_val, false) {
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
    let s1 = source(bits(word, 31, 30) as u8, bits(word, 13, 7), bits(word, 49, 49) != 0);
    let mut s2 = source(bits(word, 29, 28) as u8, bits(word, 6, 0), bits(word, 48, 48) != 0);
    if bits(word, 46, 46) != 0 {
        s2.swizzle = [0; 4];
    }

    let dest_sel = bits(word, 33, 32) as u8;
    let dest = match r7_dest_bank_index(dest_sel, bits(word, 27, 21)) {
        Some((b, i)) => Some(Operand::plain(b, i, dest_sel)),
        None => {
            blocked = blocked.or(Some("0x78 VTSTMSK destination in index mode"));
            None
        }
    };

    Instr {
        op: Op::TestMask { alu, cmp },
        pred,
        dest,
        // >>> THE CHANNEL COUNT IS DERIVED FROM THE ALU FAMILY, NOT FROM THE MASK TYPE.
        //
        // The group carries no write-mask field, so the count comes from elsewhere, and the
        // reference is explicit about where: four channels for the FLOAT family (or a dot
        // product), ONE - channel x only - for every non-float family. That is also the only
        // reading that makes sense of each: the float form's consumer dots the mask against
        // four bilinear coefficients and needs four distinguishable comparisons, while the
        // integer form's operands are whole raw lanes with no per-channel content to compare,
        // so four channels would be four copies of one answer written over four registers the
        // instruction was never meant to touch.
        write_mask: if float_ty { [true; 4] } else { [true, false, false, false] },
        srcs: vec![s1, s2],
        half_precision,
        raw: word,
        group: 0x0f,
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

/// A SEVEN-bit source field whose register number is DOUBLE-REGISTER scaled (group 0x30's
/// `src1_n` at bits 13:7, `reg_bits = 8`).
///
/// The two properties are independent and were conflated here: the reserved internal-register
/// encodings are the top four values a FIELD can hold, so a seven-bit field reserves 124..127
/// whatever its number is then scaled by, while the doubling belongs to the register numbering.
/// Reading this group through the six-bit reserved range instead let field 124 - internal
/// register i0 - decode as `Temp[248]`, which the secondary program's bank rule then forces to
/// `SecondaryAttr[248]`, a register nothing writes.
///
/// MEASURED on a racing title's track vertex program, where def-use closes only under this
/// reading: the instruction before packs `SA[36..39]` into internal register 0, this one reads
/// component w of field 124 and writes `SA[38]` channel 1, and the instruction after is the
/// identical idiom on a plainly-addressed register (`SA[35] = 1 / SA[35]`). Under the old
/// reading the pair became `SA[39] = 1 / SA[251]` - a reciprocal of zero, whose infinity
/// multiplied the clip position and took the whole track surface out of the frame.
fn r7_double_source_bank_index(bank_sel: u8, field_val: u32) -> (Bank, u8) {
    source_bank_index(bank_sel, field_val, 124, reg_index)
}

/// The DESTINATION counterpart of [`r7_double_source_bank_index`]: a seven-bit field (bits
/// 27:21) reserving 124..127 for the internal registers, with a double-register number.
fn r7_double_dest_bank_index(bank_sel: u8, field_val: u32) -> Option<(Bank, u8)> {
    match bank_rsi2(bank_sel)? {
        Bank::Temp if (124..=127).contains(&field_val) => {
            Some((Bank::Internal, internal_base(field_val - 124)))
        }
        b => Some((b, reg_index(field_val))),
    }
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
/// (CNST6), 10 immediate (IMM6), 11 index2 (RIO6).
///
/// CONSTANT yields an inline constant operand from the CNST6 table. IMMEDIATE yields an inline
/// LITERAL: spec A.7 states it plainly - "the operand's `num` field IS the literal value
/// (zero-extended); when loaded, it yields a scalar constant equal to `num`". The larger
/// immediates the same paragraph describes are assembled from EXTRA per-instruction fields and
/// belong to the instructions that have them (VBW's `src2_n | src2_sel<<7 | src2_exth<<14`,
/// LIMM's three fields); the groups reaching here carry no such fields, so the 6-bit number is
/// the whole value. The swizzle is still live and is the ordinary one - selectors 0..3 read the
/// scalar, 4..7 are the constants 0.0/1.0/2.0/0.5 - so a `(5, 5, 0, 0)` swizzle over the
/// immediate 0 is the vector `(1, 1, 0, 0)`, which is what a title's blocked instruction wanted.
///
/// The two INDEXED modes need RIO6 register-indirect addressing this decoder does not model, so
/// they still return `None` and the caller blocks emit. `op_field` is the operand's
/// register/value field.
fn exotic_source(opt_sel: u8, op_field: u32, swizzle: [u8; 4], abs: bool, neg: bool) -> Option<Operand> {
    if opt_sel & 3 == 0b10 {
        // Immediate mode: the 6-bit field IS the literal.
        let mut o = Operand::plain(Bank::Immediate, (op_field & 0x3f) as u8, opt_sel);
        o.swizzle = swizzle;
        o.abs = abs;
        o.neg = neg;
        return Some(o);
    }
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
fn ext_source(bank_sel: u8, field_val: u32, seven_bit_number: bool) -> Result<Operand, &'static str> {
    match bank_sel & 3 {
        1 if field_val & 0x40 == 0 => {
            Ok(Operand::plain(Bank::Constant, (field_val & 0x3f) as u8, bank_sel))
        }
        1 => Ok(Operand::plain(Bank::Global, (field_val & 0x3f) as u8, bank_sel)),
        2 => Err("extended bank IMMEDIATE not modeled for this group"),
        // INDEXED1 (selector 0) / INDEXED2 (selector 3): register-INDIRECT addressing. The
        // whole 7-bit number is carried through - `indexed_sub_bank` and `indexed_offset` split
        // it - and `bank_sel` records WHICH index register, so both halves reach the emitter.
        //
        // Only a group whose number field really is 7 bits may produce this; a 6-bit field
        // would put the sub-bank selector in the wrong place and silently address another bank.
        // That is why the split is not applied here but at use, against the group's own field.
        _ if !seven_bit_number => {
            Err("extended bank INDEXED needs a 7-bit number and this group encodes 6")
        }
        _ => Ok(Operand {
            bank: Bank::Indexed,
            index: (field_val & 0x7f) as u8,
            bank_sel,
            swizzle: [0, 1, 2, 3],
            abs: false,
            neg: false,
        }),
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
    // The `swz_en_strange` pair is this group's REPEAT COUNT - see
    // [`repeat_extra_iterations`], which is where the census behind that reading lives. It is
    // consumed there, not here; what stays here is the one shape the unroller cannot express:
    // a repeated DP steps its destination by one CHANNEL per iteration, so the mask it starts
    // from has to name exactly one channel. Every observed occurrence does (the reference
    // calls the bits a "single channel" override for that reason), and anything else is a
    // shape this model has not seen and must not guess at.
    let dot_repeat = dot_repeat_extra(word);
    
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
    match dot_repeat {
        None => {
            blocked = blocked.or(Some(
                "dot repeat field 47:44 outside the census (only 8 = once, and 1..3 = that                  many extra iterations, are observed)",
            ))
        }
        Some(n) if n > 0 && write_mask.iter().filter(|w| **w).count() != 1 => {
            blocked = blocked.or(Some(
                "a repeating dot whose destination mask names other than ONE channel - the                  per-iteration channel step is only established for the single-channel form",
            ))
        }
        Some(_) => {}
    }

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

/// The VEC3 form of the table-indexed ("vec34") swizzle scheme, promoted to four channels by
/// appending X in slot 3 - standard patterns at index 0..15, extended at 16..31. Selector
/// encoding matches [`Operand::swizzle`]: 0..3 = x,y,z,w lanes; 4 = 0.0; 5 = 1.0; 6 = 2.0;
/// 7 = 0.5; 8 = the undocumented `h` value (a sentinel the decoder blocks on rather than
/// guess).
///
/// No decode path reads it today - the 0x18 MAD takes [`MAD18_SWZ_VEC4`] and the 0x18 DOT's
/// operands have their own tables - but it is what the vec4 table has to be read AGAINST:
/// the two halves differ only in channel 3, which is the whole of the bug that came of
/// picking the wrong one, and a reader checking that fix needs both in front of them.
#[allow(dead_code)]
const MAD18_SWZ_VEC3: [[u8; 4]; 32] = [
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

/// The sentinel selector for the undocumented `h` swizzle value in [`MAD18_SWZ_VEC3`].
const SWZ_UNKNOWN: u8 = 8;

/// The VEC4 form of the same table-indexed ("vec34") swizzle scheme: standard patterns at
/// index 0..15, the extended set at 16..31.
///
/// **VMAD (0x18 with the present bit set) indexes THIS table, and VDP the vec3 one above.**
/// The ISA reference says so outright - the swizzle type is "vec3 when op2=0 else vec4", and
/// `op2` is the very bit that tells a dot from a mad. The two differ only in channel 3: every
/// vec3 pattern is promoted to vec4 by appending X, so reading a mad through the vec3 table
/// silently rewrites its w. That is not cosmetic on a world mesh: `XYZW`-times-a-matrix-row
/// becomes `XYZX`, so the object-to-clip transform accumulates NOTHING into clip w from the y
/// and z rows, every vertex of the mesh collapses toward one point, and the draw covers no
/// pixels at all.
const MAD18_SWZ_VEC4: [[u8; 4]; 32] = [
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
    [0, 1, 2, 5], // xyz1 (channel 3 = constant 1.0)
    // The EXTENDED half is the vec3 one, and that is a MEASUREMENT, not a transcription: the
    // same object-to-clip idiom that forces the standard half to be vec4 ends with
    // `o.xy = SA[12..13] * <index 23> + i[4..5]`, the translation column, which is only a
    // translation if index 23 multiplies by ONE. Index 23 is `111` in the vec3 extended table
    // and `zzww` in the vec4 one, and `SA[12] * position.z` is not a transform. The vec4
    // extended list carries no constant patterns at all, so a shader could not express
    // "times one" through it.
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
        match ext_source(op1_sel, op1_field, false) {
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
    s1.swizzle =
        MAD18_SWZ_VEC4[(((field(lo, low, "swz_alt_op1") & 7) << 2) | (field(lo, low, "op1_swz") & 3)) as usize];
    s1.abs = field(hi, high, "abs_op1") != 0;
    s1.neg = field(hi, high, "neg_op1") != 0;

    // op2: internal register i0..i3 (op2i), swizzle idx = swz_alt_op2_2<<4 | swz_alt_op2_x<<2 | op2_swz.
    let op2i = field(lo, low, "op2i");
    let mut s2 = Operand::plain(Bank::Internal, internal_base(op2i), op2i as u8);
    s2.swizzle = MAD18_SWZ_VEC4[(((field(hi, high, "swz_alt_op2_2") & 1) << 4)
        | ((field(lo, low, "swz_alt_op2_x") & 3) << 2)
        | (field(lo, low, "op2_swz") & 3)) as usize];
    s2.abs = field(hi, high, "abs_op2") != 0;

    // op3: internal register i0..i3 (op3i), swizzle idx = swz_alt_op3_2<<4 | swz_alt_op3_x<<2 | op3_swz.
    let op3i = field(lo, low, "op3i");
    let mut s3 = Operand::plain(Bank::Internal, internal_base(op3i), op3i as u8);
    s3.swizzle = MAD18_SWZ_VEC4[(((field(hi, high, "swz_alt_op3_2") & 1) << 4)
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
    let dest = match r7_double_dest_bank_index(op0_sel, dest_n) {
        Some((b, i)) => Some(Operand::plain(b, i, op0_sel)),
        None => {
            blocked = blocked.or(Some("dest operand in index mode"));
            None
        }
    };

    // Source op1: 2-bit bank + a SEVEN-bit field (13:7) whose number is double-register scaled,
    // reserving 124..127 for the internal registers; 2-bit modifier; broadcast of the single
    // selected source component to every channel.
    let op1_sel = bits(word, 31, 30) as u8;
    let (b1, i1) = r7_double_source_bank_index(op1_sel, bits(word, 13, 7));
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
            match ext_source(src2_sel, bits(word, 5, 0), false) {
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
        write_mask: write_mask_f16(bits(word, 27, 24), dest.as_ref(), data_type == 4),
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

    // A 16-bit-lane VBW operates on the low half of the lane and masks its result to 16 bits.
    // The lane width reaches the emitter on the op itself rather than being applied here,
    // because it changes the RESULT (an overflowing shift wraps at 16 bits, not 32) and the
    // emitter is the only place that knows the destination.
    let width16 = bits(word, 34, 34) != 0;
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

    // >>> EITHER SOURCE CAN BE THE IMMEDIATE, and the low bits come from that source's OWN
    // register field.
    //
    // The 16-bit immediate is assembled from three fields: the naming operand's 7-bit register
    // number, a 7-bit field at 20:14, and two bits at 37:36. So `src2 = #imm` takes its low
    // seven from `src2_n` (6:0) and `src1 = #imm` takes them from `src1_n` (13:7) - which is
    // what makes it possible for the OTHER operand to still be a register, as it is in the case
    // that found this: `and dest, #8, sa[80]` in three vertex programs of a retail title, where
    // 8 is a one-bit mask and SA register 80 is exactly where that program's container table
    // puts its `g_BranchBits` uniform buffer. Two independent readings landing on one
    // instruction. The 20:14 and 37:36 halves are shared and read the same way either way.
    let assemble_imm = |low: u32| -> u32 {
        let raw_imm = low | (bits(word, 20, 14) << 7) | (bits(word, 37, 36) << 14);
        let mut v = raw_imm & lane_mask;
        if rot != 0 {
            v = ((v << rot) | (v >> ((if width16 { 16 } else { 32 }) - rot))) & lane_mask;
        }
        if invert {
            v = !v & lane_mask;
        }
        v
    };
    let src1_sel = bits(word, 31, 30) as u8;
    let src1_ext = bits(word, 49, 49) != 0;
    let src1_imm = (src1_ext && src1_sel & 3 == 2).then(|| assemble_imm(bits(word, 13, 7)));
    let s1 = if src1_ext && src1_imm.is_none() {
        match ext_source(src1_sel, bits(word, 13, 7), true) {
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
    let src2_imm = src2_ext && src2_sel == 2;
    if src1_imm.is_some() && src2_imm {
        blocked = blocked.or(Some("0x50 with BOTH sources immediate - the two would assemble                                    different values out of one shared field pair"));
    }
    // The emitter puts `srcs[0]` on the LEFT and the immediate on the right, so a `src1`
    // immediate can only be expressed by swapping - which is exact for the commutative members
    // and wrong for a shift. A shift whose SHIFTED value is the immediate is not observed and
    // is blocked rather than silently reversed.
    if src1_imm.is_some() && !matches!(kind, And | Or | Xor) {
        blocked = blocked.or(Some("0x50 non-commutative member (shift/rotate) with an                                    IMMEDIATE first source - operand order not established"));
    }
    let mut srcs = Vec::new();
    let imm = if let Some(v) = src1_imm {
        // The register operand becomes the left-hand side; the immediate is the right.
        if src2_ext {
            blocked = blocked.or(Some("0x50 src2 extended bank (special/indexed) not modeled"));
        }
        let (b2, i2) = r7_source_bank_index(src2_sel as u8, bits(word, 6, 0));
        srcs.push(Operand::plain(b2, i2, src2_sel as u8));
        Some(v)
    } else if src2_imm {
        srcs.push(s1);
        Some(assemble_imm(bits(word, 6, 0)))
    } else {
        srcs.push(s1);
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
        op: Op::Bitwise { kind, imm, lane_bits: if width16 { 16 } else { 32 } },
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

/// The one group-0x14 (I16MAD) encoding this corpus establishes, with the register-number
/// field [17:14] masked out.
///
/// Every bit outside [17:14] is CONSTANT across every occurrence of the group in three titles,
/// so the corpus can say nothing about what those bits mean. Matching the whole word is
/// therefore not paranoia, it is the exact limit of the evidence: a different group-0x14
/// instruction is a different instruction, and must hard-fail rather than be decoded by a rule
/// that was only ever fitted to this one.
const I16MAD_LOAD_INDEX_WORD: u64 = 0xa08b_0946_a020_0088;

/// How much the [`I16MAD_LOAD_INDEX_WORD`] encoding adds to its source on the way into the
/// index register.
///
/// This is NOT decoded - no bit of the instruction varies, so there is nothing to decode it
/// from. It is fixed by ARITHMETIC CLOSURE against the container's own parameter table, and the
/// closure is exact. In the one shader that uses it, the six source registers hold `2*k` for
/// `k = 6*offsetIndices.x + p`, the indexed read that follows is `SA[i0 + 14 + r]` over two
/// repeat iterations, and the parameter table puts `sampleOffsets` (F32[2] x 36) at resource
/// index 22 - independently corroborated by `worldViewProj` at 0, `texCoord2offset` at 16 and
/// `posOffset` at 18, each confirmed against the instruction that reads it. `2*k + 8 + 14`
/// is `SA[22 + 2k]` and `SA[23 + 2k]`, which is exactly `sampleOffsets[k].xy`. No other addend
/// lands on the array at all.
///
/// The corpus cannot say whether the 8 belongs to this instruction or to the indexed
/// addressing; it does not matter, because the two only ever appear together, and the pairing
/// is what is measured.
const I16MAD_LOAD_INDEX_ADDEND: i32 = 8;

/// Decode a group-0x14 (I16MAD) instruction.
///
/// The ISA reference this project works from does not carry this group's encoding - it is
/// listed among its own open questions - so the only authority is the corpus, and the corpus is
/// narrow: six occurrences, one program, three titles, and the only bits that ever vary are
/// [17:14]. What those six instructions DO is settled instead by what surrounds them: each
/// loads one of six computed array indices, and each is immediately followed by a
/// register-indirect read that uses index register 0. See [`I16MAD_LOAD_INDEX_ADDEND`].
fn decode_grp_i16mad(word: u64) -> Instr {
    const REG_FIELD: u64 = 0xf << 14;
    let mut blocked = None;
    if word & !REG_FIELD != I16MAD_LOAD_INDEX_WORD {
        blocked = Some(
            "0x14 I16MAD: only the one encoding the corpus establishes (an index-register load) \
             is modeled - this is a different one, and its fields are not decodable from the \
             corpus",
        );
    }
    let src_n = bits(word, 17, 14) as u8;
    Instr {
        op: Op::LoadIndex { addend: I16MAD_LOAD_INDEX_ADDEND },
        pred: Predicate::Always,
        // Index register 0: the read that consumes it is INDEXED1, whose index register is i0.
        dest: Some(Operand::plain(Bank::Index, 0, 0)),
        write_mask: [true, false, false, false],
        srcs: vec![Operand::plain(Bank::PrimaryAttr, src_n, 2)],
        half_precision: false,
        raw: word,
        group: 0x14,
        blocked,
    }
}

/// Decode a group-0x15 IMAD32, the 32-BIT INTEGER MULTIPLY-ADD: `dest = src0 * src1 + src2`.
///
/// # The layout, and what makes it a decode rather than a reading
/// Every one of the 64 bits is accounted for, and the four reserved-zero groups (bit 47, bits
/// 41:40, bits 37:35) are checked rather than ignored - a word that sets one is a different
/// encoding and is BLOCKED, not decoded through this table.
///
/// ```text
///   63:59 opcode1 = 0x15   58:57 pred (short)     56 src0_high   54 nosched
///   53 src1_high  52 src2_high  51 dest_ext  50 end  49 src1_ext  48 src2_ext
///   47 =0  46:44 repeat  43 signed  42 saturate  41:40 =0  39:38 src2_type
///   37:35 =0  34 src0_bank  33:32 dest_bank  31:30 src1_bank  29:28 src2_bank
///   27:21 dest_n  20:14 src0_n  13:7 src1_n  6:0 src2_n        (all numbers DIRECT)
/// ```
///
/// **No operand here is double-register scaled and there is no write mask or swizzle** - the
/// instruction is scalar on the selected word. Scaling a number that is already direct reads
/// the wrong register entirely, which is the failure mode
/// [`r7_double_source_bank_index`] documents from the other direction.
///
/// # Why the bank tables are corroborated rather than assumed
/// The destination bank field goes through [`bank_rsi2`] and the src1/src2 fields through
/// [`bank_rs2`] - the tables this decoder already carried for unrelated groups, established
/// from a different source. They agree, field for field, with the layout this group was
/// extracted under. `src0` is the exception: it is a ONE-bit selector with its own table
/// (0 = TEMP, 1 = PRIMATTR) and no extension bit at all.
///
/// # What is decoded and what is blocked
/// Blocked, each naming itself, because the evidence establishes the field but not its
/// meaning: SATURATION (the clamping rule is not established, and a wrong clamp is a silently
/// wrong number), a `src2_type` other than 32-bit (the 16-bit forms exist but which of the
/// three remaining values means what does not), a REPEAT count above zero (which operands a
/// repeat increments is not established for this group), and a destination bank EXTENSION.
///
/// # The IMMEDIATE source
/// `src1` with its extension bit set and bank selector 2 is an inline literal, and the literal
/// is the 7-bit `src1_n` zero-extended - the same assembly rule the TEST group uses for its own
/// immediate. It is resolved here rather than in [`ext_source`] because how a literal is put
/// together is group-specific, which is exactly why that helper refuses it.
fn decode_grp_imad32(word: u64) -> Instr {
    let mut blocked = None;
    let mut block = |reason: &'static str| {
        if blocked.is_none() {
            blocked = Some(reason);
        }
    };

    // The reserved bits are part of the evidence: the layout closes only because they are zero
    // on every word it was checked against, so a word that sets one is outside it.
    if bits(word, 47, 47) != 0 || bits(word, 41, 40) != 0 || bits(word, 37, 35) != 0 {
        block("0x15 IMAD32: a bit this group's layout requires to be zero is set, so this is a \
               different encoding and its fields are not established");
    }
    if bits(word, 42, 42) != 0 {
        block("0x15 IMAD32: the saturating form - the clamping rule is not established");
    }
    if bits(word, 46, 44) != 0 {
        block("0x15 IMAD32: a repeat count above zero - which operands a repeat increments is \
               not established for this group");
    }
    let signed = bits(word, 43, 43) != 0;
    if bits(word, 39, 38) != 2 {
        block("0x15 IMAD32: a 16-bit source width - the group encodes one, but which of the \
               three non-32-bit selector values means what is not established");
    }

    // Destination: 2-bit bank + a 1-bit extension. Only the unextended row is established here.
    let dest_ext = bits(word, 51, 51) != 0;
    let dest = if dest_ext {
        block("0x15 IMAD32: an extended destination bank");
        None
    } else {
        match r7_dest_bank_index(bits(word, 33, 32) as u8, bits(word, 27, 21)) {
            Some((bank, index)) => Some(Operand::plain(bank, index, bits(word, 33, 32) as u8)),
            None => {
                block("0x15 IMAD32: an index-mode destination");
                None
            }
        }
    };

    // src0: a ONE-bit bank selector with its own table and no extension bit.
    let src0_sel = bits(word, 34, 34) as u8;
    let src0_bank = if src0_sel == 0 { Bank::Temp } else { Bank::PrimaryAttr };
    let src0 = Operand::plain(src0_bank, r7_reg_index(bits(word, 20, 14)), src0_sel);

    // src1 / src2: a 2-bit bank selector plus an extension bit each.
    let mut source = |ext: bool, sel: u8, field: u32| -> Operand {
        if !ext {
            let (bank, index) = r7_source_bank_index(sel, field);
            return Operand::plain(bank, index, sel);
        }
        if sel & 3 == 2 {
            // IMMEDIATE: the 7-bit number IS the literal, zero-extended.
            return Operand::plain(Bank::Immediate, (field & 0x7f) as u8, sel);
        }
        match ext_source(sel, field, true) {
            Ok(o) => o,
            Err(reason) => {
                block(reason);
                Operand::plain(Bank::Temp, 0, sel)
            }
        }
    };
    let src1 = source(bits(word, 49, 49) != 0, bits(word, 31, 30) as u8, bits(word, 13, 7));
    let src2 = source(bits(word, 48, 48) != 0, bits(word, 29, 28) as u8, bits(word, 6, 0));

    if dest.is_none() {
        // A blocked destination still has to produce a well-formed instruction for the
        // listing; the block above is what stops it being emitted.
        return Instr {
            op: Op::IntMad { signed, bits: 32 },
            pred: short_predicate(bits(word, 58, 57)),
            dest: None,
            write_mask: [true, false, false, false],
            srcs: vec![src0, src1, src2],
            half_precision: false,
            raw: word,
            group: 0x15,
            blocked,
        };
    }

    Instr {
        op: Op::IntMad { signed, bits: 32 },
        pred: short_predicate(bits(word, 58, 57)),
        dest,
        // Scalar: this group carries no write mask, so it writes the one word it names.
        write_mask: [true, false, false, false],
        srcs: vec![src0, src1, src2],
        half_precision: false,
        raw: word,
        group: 0x15,
        blocked,
    }
}

/// The bank a group-0x1a `src0` names: a ONE-bit selector at 34 widened by this group's own
/// extension bit at 47.
///
/// | ext | sel 0 | sel 1 |
/// |---|---|---|
/// | 0 | TEMP | PRIMATTR |
/// | 1 | OUTPUT | SECATTR |
///
/// Group 0x15 has the unextended row and no extension bit at all (its bit 47 is a reserved
/// zero); 0x1a is the only member of the integer multiply-add family that carries one, which is
/// what lets its `src0` reach the SECATTR bank where a uniform POINTER lives.
fn imad_step_src0_bank(ext: bool, sel: u32) -> Bank {
    match (ext, sel & 1) {
        (false, 0) => Bank::Temp,
        (false, _) => Bank::PrimaryAttr,
        (true, 0) => Bank::Output,
        (true, _) => Bank::SecondaryAttr,
    }
}

/// Decode a group-0x1a instruction: ONE STEP of a 32-bit integer multiply-add.
///
/// # The layout
/// ```text
///   63:59 opcode1 = 0x1a   58:56 pred (extended, 3 bits)   55 don't care   54 nosched
///   53:52 sn (step selector)  51 dest_ext  50 end  49 src1_ext  48 src2_ext  47 src0_ext
///   46:44 repeat  43:42 =0  41 signed  40 neg_src1  39 neg_src2  38:35 =0
///   34 src0_bank  33:32 dest_bank  31:30 src1_bank  29:28 src2_bank
///   27:21 dest_n  20:14 src0_n  13:7 src1_n  6:0 src2_n        (all numbers DIRECT)
/// ```
/// Two independent references agree on it: the opcode-map wiki's own encoding table for this
/// group (which gives the 5-bit opcode, the 3-bit predicate at 58:56, "bit 53 set produces an
/// invalid instruction", the step selector at 52 with its two values `s0`/`s1`, and the
/// signedness at 41), and the distilled integer-multiply-add field spec, which additionally
/// names the bank/number fields and the extension bits. Every bit of every word in the corpus
/// is accounted for by it, with all four reserved-zero groups reading zero.
///
/// # What `sn` selects, and why this reading rather than its rivals
/// Group 0x1a always appears as an ADJACENT PAIR - `sn = 0` then `sn = 1` - and never alone:
/// nine pairs, eighteen words, three vertex programs. In every pair the two words carry the
/// SAME `src0` and `src1`, and the second's `src2` is the first's DESTINATION. Read as the two
/// halves of a 16x32 multiplier building a 32x32 product,
///
/// ```text
///   sn = 0:  dest = (src0 & 0xffff) * src1 + src2
///   sn = 1:  dest = ((src0 >> 16) * src1) << 16 + src2
/// ```
///
/// the pair sums to exactly `src0 * src1 + src2`, and every instruction in it does real work.
///
/// Two rival readings were considered and one of them is REFUTED outright:
///
/// * "`sn` selects the low / HIGH 32 bits of a 64-bit product" is refuted by the corpus. One
///   pair's two words share a destination that the very next instruction uses as a memory
///   ADDRESS, so the second write cannot be the (zero) high word; another pair's `sn = 1`
///   writes the loop counter its own `sn = 0` read, so a high word there would zero the counter
///   and the loop would never terminate.
/// * "`sn = 1` simply delivers `src2` to the destination" survives arithmetically - it gives the
///   same final value on every word here - but it cannot explain the encoding the compiler
///   chose. Where the pair's destination is ALSO its `src0` (the loop-counter increment), the
///   first step writes a scratch register and only the second writes the counter; where the
///   destination and `src0` are different registers, both steps write the destination directly.
///   That is precisely the constraint the two-halves reading imposes - step 1 still needs the
///   unmodified `src0`, so step 0 may not clobber it - and it is no constraint at all under the
///   "deliver src2" reading, which would also make one of the two words in every pair a no-op
///   the compiler had no reason to emit.
///
/// The residual ambiguity is confined to the value the FIRST step leaves behind, and the
/// pairing check in [`crate::usse`] is what keeps it confined: a group-0x1a instruction that is
/// not part of a well-formed pair is BLOCKED, so the only shape this decoder ever emits is the
/// one on which every surviving reading agrees.
///
/// # What is blocked
/// Named individually, because each is a field this corpus establishes the position of and not
/// the meaning of: a SIGNED step (which half carries the sign is not established, and the
/// corpus is entirely unsigned), a negated source, an index-mode destination, `sn` values 2 and
/// 3 (the wiki calls a set bit 53 an invalid instruction), and any word that sets one of the
/// reserved-zero groups.
fn decode_grp_imad32_step(word: u64) -> Instr {
    let mut blocked = None;
    let mut block = |reason: &'static str| {
        if blocked.is_none() {
            blocked = Some(reason);
        }
    };

    if bits(word, 43, 42) != 0 || bits(word, 38, 35) != 0 {
        block(
            "0x1a IMAD32-STEP: a bit this group's layout requires to be zero is set, so this is \
             a different encoding and its fields are not established",
        );
    }
    if bits(word, 53, 53) != 0 {
        block("0x1a IMAD32-STEP: bit 53 is set - the reference calls that an invalid instruction");
    }
    if bits(word, 46, 44) != 0 {
        block(
            "0x1a IMAD32-STEP: a repeat count above zero - which operands a repeat increments is \
             not established for this group",
        );
    }
    let signed = bits(word, 41, 41) != 0;
    if signed {
        block(
            "0x1a IMAD32-STEP: the SIGNED form - which of the two steps carries the sign of src0 \
             is not established, and no word in the corpus is signed",
        );
    }
    if bits(word, 40, 40) != 0 || bits(word, 39, 39) != 0 {
        block(
            "0x1a IMAD32-STEP: a negated source operand - the field is established but its \
             interaction with the half-selected step is not",
        );
    }
    let high_half = bits(word, 52, 52) != 0;

    // Destination: a 2-bit bank plus its extension. Only the unextended row is established
    // here, exactly as in the sibling group.
    let dest = if bits(word, 51, 51) != 0 {
        block("0x1a IMAD32-STEP: an extended destination bank");
        None
    } else {
        match r7_dest_bank_index(bits(word, 33, 32) as u8, bits(word, 27, 21)) {
            Some((bank, index)) => Some(Operand::plain(bank, index, bits(word, 33, 32) as u8)),
            None => {
                block("0x1a IMAD32-STEP: an index-mode destination");
                None
            }
        }
    };

    // src0: a one-bit bank selector WITH an extension bit, which group 0x15 does not have.
    let src0_sel = bits(word, 34, 34);
    let src0_bank = imad_step_src0_bank(bits(word, 47, 47) != 0, src0_sel);
    let src0_n = bits(word, 20, 14);
    let src0 = if matches!(src0_bank, Bank::Temp) && (124..=127).contains(&src0_n) {
        Operand::plain(Bank::Internal, internal_base(src0_n - 124), src0_sel as u8)
    } else {
        Operand::plain(src0_bank, r7_reg_index(src0_n), src0_sel as u8)
    };

    // src1 / src2: a 2-bit bank selector plus an extension bit each, the same rows the sibling
    // group reads, including the inline-literal one.
    let mut source = |ext: bool, sel: u8, field: u32| -> Operand {
        if !ext {
            let (bank, index) = r7_source_bank_index(sel, field);
            return Operand::plain(bank, index, sel);
        }
        if sel & 3 == 2 {
            return Operand::plain(Bank::Immediate, (field & 0x7f) as u8, sel);
        }
        match ext_source(sel, field, true) {
            Ok(o) => o,
            Err(reason) => {
                block(reason);
                Operand::plain(Bank::Temp, 0, sel)
            }
        }
    };
    let src1 = source(bits(word, 49, 49) != 0, bits(word, 31, 30) as u8, bits(word, 13, 7));
    let src2 = source(bits(word, 48, 48) != 0, bits(word, 29, 28) as u8, bits(word, 6, 0));

    Instr {
        op: Op::IntMadStep { signed, high_half },
        pred: ext_predicate(bits(word, 58, 56)),
        dest,
        // Scalar: this group carries no write mask, so it writes the one word it names.
        write_mask: [true, false, false, false],
        srcs: vec![src0, src1, src2],
        half_precision: false,
        raw: word,
        group: 0x1a,
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
    // The integer destination formats, as (width, signed): U8, S8, U16, S16. O8 (2) and C10 (7)
    // are packed representations this model does not carry and stay blocked.
    let int_dest = match dest_fmt {
        0 => Some((8u8, false)),
        1 => Some((8u8, true)),
        3 => Some((16u8, false)),
        4 => Some((16u8, true)),
        _ => None,
    };
    // `scale` (bit 18) selects the NORMALIZED conversion - integer 0..max mapped to float 0..1
    // and back. Clear, it is a plain truncating numeric cast, which is what a shader computing
    // an ARRAY INDEX in float and then indexing with it uses.
    let scale = bits(word, 18, 18) != 0;
    let float_to_int = is_float(src_fmt) && int_dest.is_some() && !scale;
    // The NORMALIZED U8 conversions, both directions. U8 is format 0, and it is the one
    // normalized width this model can represent exactly: a register read and written as four
    // `byte/255` channels is `Prec::Fx8`, which already exists for the SOP2M combiner. So
    // these are emitted rather than blocked - see `Op::PackUnorm8`. The other normalized
    // widths (S8/U16/S16) have no such representation and stay blocked below.
    let unorm8_from_float = scale && is_float(src_fmt) && dest_fmt == 0;
    let unorm8_to_float = scale && src_fmt == 0 && is_float(dest_fmt);
    if !(is_float(src_fmt) && is_float(dest_fmt)) && !float_to_int && !unorm8_from_float && !unorm8_to_float {
        blocked = blocked.or(Some("0x40 pack non-float<->float conversion (int-normalize / C10 / O8) not modeled"));
    }
    // A DESTINATION in an extended bank stays blocked: the extension row for a destination is
    // SECATTR/SPECIAL/INDEX/INDEXED2, none of which this group's emit models.
    if bits(word, 51, 51) != 0 {
        blocked = blocked.or(Some("0x40 pack extended-bank DESTINATION (secattr/special/indexed) not modeled"));
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
    //
    // >>> ...UNLESS ITS BANK-EXTENSION BIT IS SET, in which case the row is
    // INDEXED1/SPECIAL/IMMEDIATE/INDEXED2 and the number is NOT doubled (spec A.3). This used
    // to block the whole instruction, and what it was blocking is a real and common operand: a
    // retail title's fragment SECONDARY programs read `SPECIAL` field 15 with bit 0x40 clear,
    // i.e. FPCONSTANT[15]. Read through the ordinary row that came out as `sa[30]` - a register
    // three times past the 10 its container declares - and the linker refused the pair for
    // reading past its uniform buffer. MEASURED: three programs of that title's corpus read
    // exactly that operand, with uniform buffers of 10, 6 and 10 registers, which is the
    // signature of a CONSTANT rather than an offset into anything.
    let src1_ext = bits(word, 49, 49) != 0;
    let (b1, i1) = if src1_ext {
        // The extended row's own number, undoubled - `ext_source` splits SPECIAL into
        // FPCONSTANT/GLOBAL on bit 0x40 exactly as every other group's operands do.
        match ext_source(src1_sel, bits(word, 13, 8), false) {
            Ok(o) => (o.bank, o.index),
            Err(why) => {
                blocked = blocked.or(Some(why));
                (Bank::Temp, 0)
            }
        }
    } else {
        r6_source_bank_index(src1_sel, bits(word, 13, 8))
    };
    let comp0_hi = if src_fmt == 6 { bits(word, 7, 7) } else { bits(word, 1, 1) };
    let c0 = ((comp0_hi << 1) | bits(word, 0, 0)) as u8;
    let c1 = bits(word, 17, 16) as u8;
    let c2 = bits(word, 15, 14) as u8;
    let c3 = bits(word, 20, 19) as u8;
    let mut s1 = Operand::plain(b1, i1, src1_sel);
    s1.swizzle = [c0, c1, c2, c3];

    Instr {
        op: if unorm8_from_float {
            Op::PackUnorm8 { to_unorm8: true, float_half: src_fmt == 5 }
        } else if unorm8_to_float {
            Op::PackUnorm8 { to_unorm8: false, float_half: dest_fmt == 5 }
        } else {
            match int_dest {
                Some((bits_, signed)) if float_to_int => {
                    Op::PackToInt { bits: bits_, signed, src_half: src_fmt == 5 }
                }
                _ => Op::Pack { src_half: src_fmt == 5 },
            }
        },
        pred: ext_predicate(predicate_raw),
        dest,
        write_mask: write_mask4(bits(word, 37, 34)),
        srcs: vec![s1],
        // An integer destination is not the F16 pipeline whatever its width, so the
        // half-precision flag (which selects the DESTINATION's float view) is only meaningful
        // for a float destination.
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
    // `sb_mode` (E0.6) picks the sample SUB-BEHAVIOUR. 0 is the ordinary sample and 3 is the
    // gather-with-coefficients form below; 1 (a bare gather4) and 2 (the u-frac/v-frac/lod
    // query) occur nowhere in the corpus, so their destination layouts have nothing to close
    // against and they stay blocked, each naming itself.
    let sb_mode = bits(word, 38, 37);
    match sb_mode {
        0 | 3 => {}
        1 => blocked = blocked.or(Some("0xE0 tex sb_mode 1 (gather4 without coefficients)")),
        _ => blocked = blocked.or(Some("0xE0 tex sb_mode 2 (texture info / query)")),
    }
    // `lod_mode` (E0.4) selects where the mip level comes from. Bias and explicit-LOD each
    // read one scalar from src2 and map onto a WGSL sample variant exactly; the GRADIENT form
    // reads two derivative VECTORS from src2, packed one after the other.
    //
    // The component split is spec E0.4, which is in that document's CONFIRMED list along with
    // the rest of the `lod_mode` table: "for 2D the first 2 components are ddx and the next 2
    // are ddy; for 3D the first 3 components are ddx and a SECOND REGISTER's 3 components are
    // ddy". Only the 2D case is wired: it reads four components of one register and needs no
    // decision about where the second register is. The 3D case does, and that is exactly the
    // thing there is no evidence for, so it stays blocked.
    let lod_mode = bits(word, 41, 40);
    let lod = match lod_mode {
        0 => TexLod::Implicit,
        1 => TexLod::Bias,
        2 => TexLod::Level,
        _ => TexLod::Gradient,
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
    // A GRADIENT sample is wired for 2D only - see the `lod_mode` note above. With three
    // coordinates the two derivative vectors are three components each and the second one
    // begins in a register this decoder would have to GUESS at; a wrong guess samples the
    // wrong mip of the right texture, which looks like a texture-filtering bug and not like a
    // decode bug.
    if matches!(lod, TexLod::Gradient) && coords != 2 {
        blocked = blocked.or(Some(
            "0xE0 tex gradient with a non-2D coordinate: the second derivative vector's \
             register is not established - wire this case and re-run",
        ));
    }
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

    // Sampler unit = src1 register number, DOUBLED: the bound texture's control words live at
    // SA register `2 * src1_n`, which the container's texture-control table resolves to a GXM
    // texture unit (see `Program::sampler_unit_at`).
    //
    // This used to rest on "the shared double-register rule" - an ANALOGY, and the same analogy
    // that was wrong for this instruction's DESTINATION (see `dest_reg` below). It is now
    // measured instead. Over three titles' corpora, `2 * src1_n` lands on a declared texture
    // 166 times out of 167, while `src1_n`, `src1_n + dubuf` and `2 * src1_n + dubuf` land on
    // one 4, 37 and 8 times. The doubling is also exercised on FOUR DISTINCT table slots inside
    // a single program (`frag_82f324c0` samples units 11, 15, 0 and 1 at SA 14/18/22/26), which
    // is what retires the old worry that it was only ever right by coincidence on a first slot.
    // Test: `how_a_smp_sampler_field_addresses_the_texture_control_table`.
    //
    // The one program that disagrees is `frag_866a1840`, whose table places its only texture at
    // an ODD SA register - see `Program::texture_control_base_is_addressable`. A double-register
    // field cannot name an odd register at all, so that is a property of that blob and not a
    // rule this one is missing; it stays BLOCKED rather than resolved by an invented offset.
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
    // The SMP destination is NOT double-scaled the way an ALU operand is, at EITHER precision:
    // `dest_n` is the register the result starts at, and it occupies as many registers as its
    // width needs - two for an F16 pair, four for F32.
    //
    // The F16 half was established by def-use chains in real fragment blobs (every
    // albedo/ambient/fog sample resolves only under this rule). The F32 half was decoded as
    // DOUBLED by analogy with the ALU operand rule, and that was wrong; the corpus settles it
    // (`full_precision_sample_destination_closure`). A title's vector-canvas VERTEX program
    // samples with `dest_n = 4`: it writes temps {0..3, 8..11} under the doubled reading, then
    // READS {0..3, 4, 5, 7} - so the sample lands where nothing reads it and three reads name
    // registers the program never writes. Direct, every read closes against a write. The
    // shipped compiler does not emit programs that read undefined temporaries.
    //
    // Nothing else in three titles' corpora discriminates: every other full-precision sample
    // uses `dest_n = 0`, where the two readings are the same register.
    let dest_reg = dest_n;
    let (dbank, didx) = if matches!(dest_bank, Bank::Temp) && (124..=127).contains(&dest_n) {
        (Bank::Internal, internal_base(dest_n - 124))
    } else {
        (dest_bank, dest_reg as u8)
    };

    // src2 carries the bias / explicit level - or, for a gradient sample, BOTH derivative
    // vectors - when `lod_mode` asks for one. Like every other SMP operand it is
    // double-register scaled (spec E0.2). It is read as F32: in the corpus the instruction
    // immediately before each explicit-LOD sample is a `mov .f32` writing exactly this
    // register, so the level is produced and consumed at 32-bit width.
    let mut tex_srcs = vec![coord];
    if !matches!(lod, TexLod::Implicit) {
        let src2_sel = bits(word, 29, 28) as u8;
        let (b2, i2) = source_bank_index(src2_sel, bits(word, 6, 0), 124, reg_index);
        tex_srcs.push(Operand::plain(b2, i2, src2_sel));
    }

    // The GATHER form is a different operation with a different destination extent, so it is a
    // different op. Everything the corpus does not establish about it is refused here rather
    // than folded into the ordinary sample: a gather is 2D only (the footprint is a 2x2 of
    // texels, and what a 1D or 3D one gathers is not stated), its result is the full-precision
    // one every occurrence asks for, and it reads no LOD operand.
    if sb_mode == 3 {
        if coords != 2 {
            blocked = blocked.or(Some(
                "0xE0 tex gather4 with a non-2D coordinate: the gathered footprint is not \
                 established for that dimensionality",
            ));
        }
        if !matches!(lod, TexLod::Implicit) {
            blocked = blocked.or(Some("0xE0 tex gather4 with an explicit LOD / bias / gradient"));
        }
        if result_f16 {
            blocked = blocked.or(Some(
                "0xE0 tex gather4 with an F16 result: where the four F16 coefficients land \
                 relative to a half-width gather is not established",
            ));
        }
        if matches!(dbank, Bank::Internal) {
            blocked = blocked.or(Some(
                "0xE0 tex gather4 into an internal register: the six-register result does not \
                 fit one",
            ));
        }
        return Instr {
            op: Op::TexGather { unit, coords, coord_half },
            pred: ext_predicate(predicate_raw),
            dest: Some(Operand::plain(dbank, didx, 0)),
            write_mask: [true; 4],
            srcs: tex_srcs,
            half_precision: result_f16,
            raw: word,
            group: 0x1c,
            blocked,
        };
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
/// How many EXTRA times `word` re-executes after its first execution (0 = runs once), or
/// `None` when this group's repeat encoding is not established and the answer therefore cannot
/// be stated. A caller that cannot handle `None` must block the instruction rather than assume
/// zero: assuming zero is what silently drops the later iterations of a repeating instruction.
///
/// This is the single place the per-group repeat encoding is written down.
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
/// Extra iterations of a group-0x18 DOT, from its bits 47:44 - or `None` when the field holds
/// a value this corpus has never shown and the answer therefore cannot be stated.
///
/// # The field the reference names `unk7 / abs_op2 / swz_en_strange1 / swz_en_strange0`
/// Bits 47:44 are where every USSE group with a documented `repeat_count` puts it. This
/// group's own field table names them otherwise, and the reference says of the two low bits
/// only that they "force override swizzle masking with single channel" - a description of an
/// EFFECT with no encoding behind it.
///
/// # The census that settles it (766 blobs, six titles)
/// Over every 0x18 DOT in the corpus the field takes exactly four values: `0x8` (365
/// occurrences), `0x1` (3), `0x2` (6), `0x3` (5). Never `0x0`, and never any value with bit 47
/// set *and* a low bit set. That perfect anti-correlation is what says bit 47 is part of the
/// same field rather than an independent flag: an orthogonal bit would show `0x9`/`0xa`/`0xb`
/// somewhere in 379 samples. So `0x8` is the "runs once" encoding and `1..3` are extra
/// iterations - and reading the four bits as a plain count instead would make 365 instructions
/// repeat eight times, which the destination-closure oracle would report immediately.
///
/// # What a repeated DP does, and why it is not the generic operand step
/// A DP writes a SCALAR. Repeating it walks the destination one CHANNEL per iteration (which
/// is precisely the "single channel" the reference saw) while the vector source advances by a
/// whole vector, so `n = 3` over a 4-channel DP is a 4x4 matrix transform written into the four
/// lanes of clip position - the idiom a vertex program's very first arithmetic is. One retail
/// title's world vertex programs contain exactly one DOT each, with `n = 3`, sourced from a
/// `WorldViewProjection` uniform declared `F32[4]` x 4 at register 0, writing the four position
/// lanes nothing else in the program writes. Read as running once, those programs emit a single
/// scalar for a whole clip position and the title renders a BLACK FRAME.
fn dot_repeat_extra(word: u64) -> Option<u32> {
    match bits(word, 47, 44) {
        0x8 => Some(0),
        n @ 1..=3 => Some(n as u32),
        _ => None,
    }
}

pub fn repeat_extra_iterations(word: u64) -> Option<u32> {
    match opcode1(word) {
        // Established repeat_count fields.
        0x06 | 0x08 | 0x0a..=0x0d => Some(bits(word, 47, 44) as u32), // 0x30, 0x40, 0x50 family
        0x07 | 0x09 | 0x0f => Some(bits(word, 45, 44) as u32),        // 0x38 VMOV, 0x48 VTST
        // No repeat_count field exists in these layouts.
        0x01 | 0x02 => Some(0), // 0x08/0x10 V32NMAD/V16NMAD - 47:44 is src2_swiz
        0x1c => Some(0),        // 0xE0 SMP
        0x1f => Some(0),        // 0xF8 complex flow
        // 0xE8/0xF0 memory access: the 64-bit layout tiles exactly with no repeat_count
        // field anywhere in it - `mask_count` at 47:44 is the ELEMENT count, which is the
        // instruction's own multi-register transfer, not a repeat. See the distilled
        // memory-access spec notes (the widths sum to 64 with no hole).
        0x1d | 0x1e => Some(0),
        // 0x00 vector MAD: no repeat_count field. The remaining `unk` bits in that group are
        // scattered singles, not a contiguous field at any position a repeat count occupies
        // elsewhere, and its whole corpus closes against the declared output interface under
        // the reading that a MAD executes once.
        0x00 => Some(0),
        // 0x18: the MAD form has no repeat field either, for the same reason and with the same
        // corroboration. The DOT form's bits 47:44 ARE one - see [`dot_repeat_extra`] for the
        // census, and note that reading the MAD form's identically-positioned bits the same way
        // would repeat several hundred instructions that demonstrably run once.
        0x03 if bits(word, 53, 53) == 1 => Some(0),
        0x03 => dot_repeat_extra(word),
        // 0x80 SOP2, in the fragment-epilogue form `decode_grp_sop2` establishes and only
        // there: that form pins bits 46:43 to zero, and bit 47 is the second complement flag
        // its sibling 0x90 establishes at the same position. So the four bits where a repeat
        // count lives elsewhere are `1000` with three of them accounted for as zero and the
        // fourth as a field the sibling group corroborates - there is no count to read. The
        // corpus agrees from the other side: this instruction is the LAST one in all five
        // programs that carry it, and a repeat would step its destination past the colour
        // register the hardware emits, into registers nothing reads. Any other group-0x80
        // word is a form this says nothing about, and unknown means blocked.
        //
        // The SWAPPED shape reads `00000` here instead, because bit 47 - that same complement
        // flag - is clear in it. The rest of the argument is unchanged and the corpus makes it
        // the same way: the one program carrying it ends on this instruction, and it is the
        // byte-for-byte twin of a program that ends on the `10000` word. Anything outside the
        // two shapes `decode_grp_sop2` pins still blocks there, so admitting `00000` here does
        // not admit a word on its own.
        0x10 if matches!(bits(word, 47, 43), 0b10000 | 0b00000) => Some(0),
        // 0x90 SOP2M: no repeat_count field either, and the field table accounts for every bit
        // - the four bits at 47:44, where every group that HAS a repeat count puts it, are the
        // second complement bit plus the top three bits of the write mask. Both are read by
        // `decode_grp_sop2m` and both are corroborated by the corpus: the mask names exactly
        // the channel the following test reads, and the complement is what turns the zero
        // coefficient into the copy the macro needs. A repeat count cannot also live there.
        0x12 => Some(0),
        // 0x14 I16MAD: the reference does not carry this group's layout, so there is no
        // documented repeat field to read. What IS known is that every occurrence in the corpus
        // is one fixed word outside its register-number field, so for that word the answer is
        // "no repeat" - it either repeats every time or never, and its neighbours show it
        // executing once. Any OTHER group-0x14 word is a different instruction whose repeat
        // encoding is unknown, and unknown means blocked, not zero: a dropped iteration is
        // exactly the invisible failure this pass exists to prevent.
        0x14 if word & !(0xf << 14) == I16MAD_LOAD_INDEX_WORD => Some(0),
        // 0x15 IMAD32: a THREE-bit repeat count at 46:44, and bit 47 is a reserved zero rather
        // than the top of it. Reading it as the usual four bits at 47:44 would be harmless on
        // every word whose bit 47 is clear and would double the count on any word where it is
        // not - and `decode_grp_imad32` blocks such a word anyway, because a set bit 47 means
        // the whole layout is a different one. The field's POSITION is established; what a
        // repeat ITERATES over in this group is not, so a non-zero count is `None` (blocked)
        // rather than an unroll count - the same stance `decode_grp_imad32` takes.
        0x15 if bits(word, 46, 44) == 0 => Some(0),
        // 0x1a IMAD32-STEP: the same three-bit repeat count at 46:44, with bit 47 taken by
        // this group's own src0 bank EXTENSION rather than by a reserved zero or the top of the
        // count. Same stance as 0x15: the position is established, what an iteration steps is
        // not, so a non-zero count blocks instead of unrolling.
        0x1a if bits(word, 46, 44) == 0 => Some(0),
        _ => None,
    }
}

/// True when `word` is a group-0xF8 SMLSI (the discriminant [`decode_grp_flow`] matches on).
pub fn is_smlsi(word: u64) -> bool {
    opcode1(word) == 0x1f && bits(word, 58, 56) == 0b010 && bits(word, 53, 52) == 0b01
}

/// What one SMLSI leaves for a single hardware operand slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmlsiSlot {
    /// The operand's encoded register FIELD advances by `n` per repeat iteration, so its
    /// register index advances by `n * stride` where the stride is the field's own scaling
    /// ([`RepeatOperand::stride`]). SIGNED: the corpus's SMLSI bytes include 0xff, 0xfe, 0xfd,
    /// 0xfc and 0xf8, which as unsigned increments would step an operand 248 registers past
    /// every bank in the machine, and as two's-complement are the small backward steps
    /// -1, -2, -3, -4 and -8 that a program walking a matrix or an attribute block in reverse
    /// asks for. `Increment(1)` is the default - the state in force before any SMLSI.
    Increment(i8),
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
/// MEASURED against the corpus, on both ends of the idiom that uses it. Across three unrelated
/// titles, vertex programs open with `SMLSI; VMOV(repeat N)` copying vertex attributes straight
/// to the output bank, and in every one of them the DEFAULT stepping (increment 1, which for
/// those six-bit operand fields is two registers) is what closes:
///
///  * on the destination side, one program's three iterations of `Output[8] <- PA[4]` land its
///    last write exactly on the `TexCoord(0)` varying the container declares at output lane 12,
///    and the program's writes then fill lanes 0..13 of a declared 14-lane interface with no
///    gap. A stride of one register would never reach lane 12 (the varying would go
///    uninterpolated); a stride of four would run two lanes past the declared interface.
///  * on the source side, another program's four iterations of `Output[4] <- PA[4]` consume
///    exactly `PA[4..11]` - its `in_texCoord` and `in_colour` attributes, the whole declared
///    12-register attribute set with nothing left over. Under a non-advancing source every
///    iteration would re-read `in_texCoord.xy` and `in_colour` would be dead, yet the container
///    declares it as a fed vertex attribute and no other instruction in that program reads it.
///
/// A slot an instruction does not have is a DON'T CARE, and the corpus is full of them: the
/// bytes for src0 and src2 vary freely (0x01, 0x0e, 0x21, 0x2c, 0x38, 0x4e, 0xf6, 0xf8, 0xff)
/// across words whose repeat is an unconditional VMOV, which reads src1 alone. That is why
/// [`repeat_operands`] names the slots an instruction actually occupies rather than the state
/// being required to be default everywhere.
///
/// Bit 50 also varies (set on the words that restore the default, clear on the ones that open
/// the attribute copy). It sits where group 0x38 documents `end`, and it is not part of either
/// field above; the register-addressing model does not read it.
pub fn decode_smlsi(word: u64) -> [SmlsiSlot; 4] {
    std::array::from_fn(|k| {
        let value = ((word >> (8 * k)) & 0xff) as u8;
        if (word >> (32 + k)) & 1 == 0 {
            SmlsiSlot::Increment(value as i8)
        } else {
            SmlsiSlot::Swizzle(value)
        }
    })
}

/// The state a repeat consults before any SMLSI has run: every operand advances by one field
/// step per iteration. This is what [`unroll_repeats`](crate::usse::unroll_repeats) applied
/// unconditionally before SMLSI was modelled, and it is what the measurements above pin.
pub(crate) const DEFAULT_REPEAT_STATE: [SmlsiSlot; 4] = [SmlsiSlot::Increment(1); 4];

/// One operand of a repeating instruction, as the repeat machinery sees it: which hardware
/// operand slot it occupies (so the right SMLSI byte governs it) and how far its REGISTER INDEX
/// advances for an increment of one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RepeatOperand {
    /// Slot, indexed exactly as [`decode_smlsi`] indexes its result: 0 dest, 1 src0, 2 src1,
    /// 3 src2. Ignored when `moe` is false.
    pub slot: usize,
    /// Whether this operand's advance is governed by the MOE state at all. False means the
    /// advance is INTRINSIC to the instruction - it steps by `stride` every iteration whatever
    /// the SMLSI last programmed. See the DP destination in [`repeat_operands`].
    pub moe: bool,
    /// Registers advanced per unit of increment. This is NOT a free parameter: an operand's
    /// register field is either SIX bits, which the hardware scales by two ([`reg_index`]), or
    /// SEVEN bits, which it uses directly ([`r7_reg_index`]). A repeat steps the encoded FIELD,
    /// so a six-bit field's register index moves two at a time and a seven-bit field's moves one.
    pub stride: u32,
}

/// Where each operand of `word` sits for repeat purposes: the destination, then each source in
/// the order [`decode`] puts them in `Instr::srcs`. `None` when this group's operand grammar is
/// not established, which must BLOCK the instruction rather than default to anything.
///
/// This is only ever consulted for an instruction that actually repeats, so it needs to cover
/// exactly the groups a repeat is encodable in and observed for. It reproduces, from the field
/// widths the decoder already uses, the two per-group "repeat multiplier" statements the ISA
/// reference makes explicitly - group 0x40's `(dest,src1,src2) = (1,2,2)` for a float source and
/// group 0x50's `all 1` - which is the cross-check that the six-vs-seven-bit reading is the rule
/// behind those numbers rather than a coincidence.
///
/// The SLOT numbering - that an SMLSI's four bytes are `[dest, src0, src1, src2]`, so an
/// instruction with no src0 (an unconditional VMOV, a VPCK, a VBW) skips the second byte - is
/// the reference's own order for the sibling SMBO ("four 12-bit base offsets (dest/src0/src1/
/// src2)"), and it is MEASURED against the alternative. Renumbering the slots so the bytes read
/// `[dest, src1, src2, src0]` instead - which is tempting, because it repairs three vertex
/// programs that otherwise read a PA register no attribute declares - loses far more than it
/// fixes across three titles' corpora: 312 -> 310 standalone recompiles on one, 64 -> 59 on
/// another, and 79 -> 78 on a corpus that is otherwise CLEAN under this order. A reading that
/// breaks a corpus with nothing left to explain is the wrong reading.
pub(crate) fn repeat_operands(word: u64) -> Option<Vec<RepeatOperand>> {
    let op = |slot, stride| RepeatOperand { slot, stride, moe: true };
    // An operand whose per-iteration advance is the instruction's own, not the MOE's.
    let fixed = |stride| RepeatOperand { slot: 0, stride, moe: false };
    match opcode1(word) {
        // 0x38 VMOV (`decode_grp_38`). Every operand is a SIX-bit field: dest `dest_n` (23:18),
        // src1 `src1_n` (11:6), src2 `src2_n` (5:0), src0 `src0_n` (17:12) - so all stride 2.
        // `move_type` (47:46) 0 is the unconditional form `dest = src1`, which has no src0 and
        // no src2 operand at all; any other form is the conditional select, whose sources are
        // pushed in the order (src1, src2, src0).
        0x07 if bits(word, 47, 46) == 0 => Some(vec![op(0, 2), op(2, 2)]),
        0x07 => Some(vec![op(0, 2), op(2, 2), op(3, 2), op(1, 2)]),
        // 0x40 VPCK (`decode_grp_pack`). The destination is a SEVEN-bit field `dest_n` (27:21)
        // and the single source is a SIX-bit field (13:8) - strides (1, 2), the reference's own
        // repeat multipliers.
        //
        // The source's SMLSI slot is 1 (src0), MEASURED, and it is NOT the slot the sibling VMOV
        // uses. Two independent populations of repeating VPCKs in one title's corpus say so:
        //
        //   dest 46 src 62, smlsi [dest,src0,src1,src2] = [ 3,  3,  1, -4]   (6 blobs)
        //   dest 46 src  0, smlsi [dest,src0,src1,src2] = [ 2,  2, -4, -4]   (2 blobs)
        //
        // In both, the compiler set slot 0 and slot 1 to the SAME increment and left the others
        // unrelated - the signature of a one-source instruction whose two live operands are dest
        // and src0. It also closes arithmetically: this VPCK writes packed F16 halves from F32
        // sources, so one iteration's dest advance in HALVES must equal its source advance in
        // FLOATS. Slot 1 gives (3 regs = 6 halves, 3*2 = 6 floats) and (2 regs = 4 halves,
        // 2*2 = 4 floats); slot 2 gives 6 halves from 2 floats, and 4 halves from a source
        // stepping to register -16, which is not a register at all.
        0x08 => Some(vec![op(0, 1), op(1, 2)]),
        // 0x50 VBW family (`decode_grp_bitwise`): dest `dest_n` (27:21), src1 (13:7) and src2
        // (6:0) are all SEVEN-bit fields - "repeat multipliers all 1". An immediate src2 is
        // folded into the op at decode and is not an IR source, so the list is dest, src1, and
        // src2 only when it is a register; a shorter `srcs` simply consumes fewer entries.
        0x0a..=0x0d => Some(vec![op(0, 1), op(2, 1), op(3, 1)]),
        // 0x18 DOT (`decode_grp_18_dot`), and it is the one group whose repeat is not a plain
        // register walk - see [`dot_repeat_extra`]:
        //
        // * the DESTINATION advances one LANE per iteration, because a DP writes a scalar into
        //   one channel and the next iteration writes the next channel. In this IR a masked
        //   destination writes lane `index + channel`, so a single-channel mask plus a stride
        //   of ONE lane expresses that exactly (`decode_grp_18_dot` refuses any other mask).
        //
        //   >>> AND THAT ADVANCE IS INTRINSIC, NOT MOE-GOVERNED. It used to be routed through
        //   the SMLSI destination byte, which is invisible while the state is its default
        //   `Increment(1)` - every repeating DP in four corpora runs under exactly that - and
        //   wrong the moment a program programs the state. A golf lit-material vertex program
        //   opens `SMLSI [7, 7, 1, 1]` and then transforms its position with a three-iteration
        //   DP: the source byte 1 walks the matrix rows correctly (stride 4 -> sa[0], sa[4],
        //   sa[8]) while the destination byte 7 sent the three results to `o[0]`, `o[7]` and
        //   `o[14]` - two varying lanes - leaving clip `y` and `z` unwritten, which is what
        //   `[o0 -1 -2 o3]` was. Under the intrinsic advance they land in `o[0..3]` beside the
        //   `w` the next instruction copies, and the program's clip position closes. The
        //   channel walk is the DP's own; the MOE has nothing to step it with.
        // * op1, the R6 vector source, advances by a whole VECTOR - `c3_en` channels - which is
        //   what makes four iterations read the four rows of a 4x4 matrix laid out at
        //   consecutive registers.
        // * op2 is an INTERNAL register (RI2, a 2-bit selector, not a register field). It is
        //   the vector being transformed and it does not move; a 2-bit selector has nowhere to
        //   step to in any case.
        0x03 if bits(word, 53, 53) == 0 => {
            // >>> THE VECTOR SOURCE STEPS A WHOLE FOUR-REGISTER SLOT, NOT `c3_en` CHANNELS.
            // This used to advance by the number of channels the DP SUMS (3 or 4 from bit 52),
            // which is right for a 4-component DP and wrong for a 3-component one - and a
            // 3-component DP over a matrix is exactly how a program transforms a NORMAL.
            // PCSA00009's world material opens with a three-iteration 3-component DP whose
            // source is `PrimaryAttr[28]`; its rows are the `modelWorldX/Y/Z` attributes, which
            // the container declares at lanes 28, 32 and 36 - FOUR apart, because each is a
            // declared float4. Stepping by three read `pa[28]`, `pa[31]`, `pa[34]`: row 0
            // correct, row 1 straddling `modelWorldX.w`, row 2 straddling `modelWorldY.zw`.
            // The world normal came out wrong, `dot(normal, g.Light.position)` came out
            // non-positive over the whole course, `max(N.L, 0)` killed the diffuse term and the
            // terrain rendered BLACK. Stride 4 fits both widths at once: the 4-component DPs
            // that walk a 4x4 matrix at consecutive registers are unchanged, because their
            // rows are four apart too. `c3_en` says how many channels are SUMMED; it does not
            // say how far the next iteration's operand slot begins.
            let _ = bits(word, 52, 52);
            Some(vec![fixed(1), op(2, 4), op(3, 0)])
        }
        _ => None,
    }
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
        // DEPTHF (spec F8.7): write `src0` - a scalar - into the fragment depth output.
        //
        // `two_sided` (bit 39) and `feedback` (bits 38:37) select variants whose extra
        // behaviour the reference does not resolve, and neither is a detail that can be
        // ignored: a depth WRITE is what the depth test then compares against, so a variant
        // silently treated as the plain form would sort the whole surface wrongly. Both are
        // zero on the only DEPTHF this corpus contains, so anything else blocks.
        let mut d = None;
        if bits(word, 39, 39) != 0 {
            d = Some("0xF8 DEPTHF two-sided variant not modeled");
        } else if bits(word, 38, 37) != 0 {
            d = Some("0xF8 DEPTHF feedback mode not modeled");
        }
        (Op::DepthF, d)
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

    // DEPTHF is the one member of this group with a data operand: `src0` (bank bit 36 + ext
    // bit 51, number bits 20:14) holds the depth. The number is read DIRECT, not
    // double-register scaled: scaling belongs to the float data types (spec A.3), and this
    // group carries no data-type field at all to select one. The corpus cannot separate the
    // two readings on its own - its single DEPTHF names register 0, where they agree - so the
    // argument has to come from the encoding, and it does.
    let srcs = if matches!(op, Op::DepthF) {
        let sel = bits(word, 36, 36);
        let ext = bits(word, 51, 51);
        // Spec A.2, src0 bank: ext=0 -> 0 TEMP / 1 PRIMATTR; ext=1 -> 0 OUTPUT / 1 SECATTR.
        let bank = match (ext, sel) {
            (0, 0) => Bank::Temp,
            (0, _) => Bank::PrimaryAttr,
            (_, 0) => Bank::Output,
            (_, _) => Bank::SecondaryAttr,
        };
        let n = bits(word, 20, 14);
        let (b, i) = if matches!(bank, Bank::Temp) && (124..=127).contains(&n) {
            (Bank::Internal, internal_base(n - 124))
        } else {
            (bank, r7_reg_index(n))
        };
        vec![Operand::plain(b, i, sel as u8)]
    } else {
        Vec::new()
    };

    Instr {
        op,
        pred,
        dest: None,
        write_mask: [false; 4],
        srcs,
        half_precision: false,
        raw: word,
        group: 0x1f,
        blocked,
    }
}

/// Decode a group-0xE8 MEMORY LOAD (opcode1 0x1d) - the load half of the 0x1d/0x1e
/// memory-access format (see the distilled memory-access spec notes for the closed bit
/// table; the store half, opcode1 0x1e, stays a classified stub).
///
/// | field | bits | | field | bits |
/// |---|---|---|---|---|
/// | group const `111` | 63:61 | | `mask_count` (elements - 1) | 47:44 |
/// | direction (`01` load) | 60:59 | | `addr_mode` / `mode` | 43:42 / 41:40 |
/// | `pred` | 58:56 | | `dest_bank` (1-bit: TEMP/PA) | 39 |
/// | `skipinv` / `nosched` | 55 / 54 | | `range_enable` | 38 |
/// | `moe_expand` / `sync_start` | 53 / 52 | | `data_type` / `inc_dec` | 37:36 / 35 |
/// | `cache_ext` | 51 | | `src0_bank` (1-bit) | 34 |
/// | src0/1/2 bank ext | 50 / 49 / 48 | | `cache_bypass12` / `drc_sel` | 33 / 32 |
/// | `src1_bank` / `src2_bank` | 31:30 / 29:28 | | dest / src0 / src1 / src2 n | 27:21 / 20:14 / 13:7 / 6:0 |
///
/// The effective byte address is `src0 + src1 + src2`, where an IMMEDIATE offset counts
/// ELEMENTS (scaled by the element size here), while a register-supplied offset would
/// already be in bytes. No operand is double-register scaled.
///
/// # What is modelled, and the census that scopes it
/// ONLY the variant every shipped instruction of this family uses (the corpus census over
/// every captured blob): `mode = 0, addr_mode = 0`, 32-bit elements, unconditional, source
/// pointer in the PA bank, both offsets IMMEDIATE. `mode`/`addr_mode` are the format's
/// address-space selector in some arrangement that remains unestablished - but a field that
/// is ZERO in every shipped instance does not need its other values established to decode
/// the zero case. Every departure from the census blocks BY NAME below; none is guessed.
fn decode_grp_mem_load(word: u64) -> Instr {
    let mut blocked: Option<&'static str> = None;
    // First reason wins, matching the convention every other group decoder uses.
    let set = |b: &mut Option<&'static str>, why: &'static str| *b = b.or(Some(why));

    // The predicate field's table for this group is not established; every shipped
    // instance is 0, and 0 is `Always` in every established table, so only 0 passes.
    if bits(word, 58, 56) != 0 {
        set(&mut blocked, "0xE8 memory-load predicate table not established");
    }
    // `moe_expand` (bit 53) puts the instruction's operands under the MULTIPLE-OPERAND
    // EXPANSION state - the per-iteration register/address stepping an SMLSI programs, which
    // this decoder already models as [`SmlsiSlot`] with a default of `Increment(1)`.
    //
    // >>> WHY THE SET CASE IS ALLOWED FOR A SINGLE ELEMENT, and only for it. The census over
    // every captured corpus (`usse_memory_group_field_census`, 322 memory-access instructions
    // across six titles) splits perfectly complementarily:
    //
    //   moe_expand = 1 -> elements = 1, always (4 instances, one golf lit-material vertex
    //                     program in four variants), and NO SMLSI has executed earlier in any
    //                     of those programs, so the state in force is the default.
    //   moe_expand = 0 -> elements 2, 3, 4, 8 or 12 - never 1 (318 instances).
    //
    // Expansion is a stepping rule between ITERATIONS. With one element there is no second
    // iteration to step to, and the first is at step zero under every reading of the field, so
    // on this domain the bit cannot change which address is read or which register is written.
    // What the bit means when a burst is expanded stays unestablished and stays blocked - and
    // the second half of the guard, that the MOE state is still its default, is enforced in
    // `unroll_repeats`, which is where the state is actually walked.
    if bits(word, 53, 53) != 0 && bits(word, 47, 44) != 0 {
        set(&mut blocked, "0xE8 memory-load moe_expand over a MULTI-ELEMENT burst not established");
    }
    if bits(word, 52, 52) != 0 {
        set(&mut blocked, "0xE8 memory-load sync_start semantics not established");
    }
    if bits(word, 43, 42) != 0 {
        set(&mut blocked, "0xE8 memory-load addr_mode value not established (only 0 is)");
    }
    if bits(word, 41, 40) != 0 {
        set(&mut blocked, "0xE8 memory-load mode value not established (only 0 is)");
    }
    if bits(word, 38, 38) != 0 {
        set(&mut blocked, "0xE8 memory-load range_enable semantics not established");
    }
    if bits(word, 37, 36) != 0 {
        set(&mut blocked, "0xE8 memory-load non-32-bit data_type not established");
    }
    if bits(word, 35, 35) != 0 {
        set(&mut blocked, "0xE8 memory-load auto-increment (inc_dec) not established");
    }
    // `skipinv`/`nosched` are pipeline hints with no data-path effect, `cache_ext`/
    // `cache_bypass12` steer the cache, and `drc_sel` names which dependent-read counter
    // tracks the (asynchronous) access - none of them changes WHAT is loaded, and this
    // model completes every load synchronously, which is exact or stronger. All ignored.

    let elements = (bits(word, 47, 44) + 1) as u8;

    // src0, the byte pointer: 1-bit bank at 34, extension at 50 (spec A.2 row). Two banks are
    // in the census, and they are the two ways a program gets hold of a buffer address:
    //
    // * PRIMATTR - per-vertex data, or per-vertex-derived, as in the skinning idiom that
    //   computes the pointer into a PA register first.
    // * SECONDARY-ATTR - the buffer's bound address as the DRIVER left it, in the SA register
    //   the container's +0x78 binding table names. That is the shape
    //   [`crate::module::resolve_mem_window`] was written around (it REQUIRES an instruction to
    //   read exactly that register, and `link` initialises it from the bound window's own
    //   header), so blocking it here refused the one form the rest of the pipeline is built to
    //   serve. Three vertex programs of a retail title load through it.
    //
    // TEMP and OUTPUT stay out: neither is observed, and a pointer read out of a register the
    // program computed for something else is not a thing to guess at.
    let src0 = {
        let (ext, sel) = (bits(word, 50, 50), bits(word, 34, 34));
        let bank = match (ext, sel) {
            (0, 0) => Bank::Temp,
            (0, _) => Bank::PrimaryAttr,
            (_, 0) => Bank::Output,
            (_, _) => Bank::SecondaryAttr,
        };
        if !matches!(bank, Bank::PrimaryAttr | Bank::SecondaryAttr) {
            set(&mut blocked, "0xE8 memory-load src0 outside the PA/SA banks not in the census");
        }
        Operand::plain(bank, r7_reg_index(bits(word, 20, 14)), sel as u8)
    };

    // src1/src2, the offsets: 2-bit bank + extension, the shared row. The census holds
    // IMMEDIATE only, where the 7-bit number is a count of ELEMENTS scaled by the element
    // size (32-bit here). A register-supplied byte offset is spec'd but unobserved.
    let mut imm_elements = 0u32;
    for (bank_hi, bank_lo, ext_bit, n_hi, n_lo) in [(31u32, 30u32, 49u32, 13u32, 7u32), (29, 28, 48, 6, 0)] {
        let (sel, ext) = (bits(word, bank_hi, bank_lo), bits(word, ext_bit, ext_bit));
        if ext == 1 && sel == 2 {
            imm_elements += bits(word, n_hi, n_lo);
        } else {
            set(&mut blocked, "0xE8 memory-load register-supplied offset not in the census");
        }
    }
    let offset_bytes = imm_elements * 4;

    // Destination: a 1-bit bank (TEMP / PRIMATTR), 7-bit direct number, `elements`
    // CONSECUTIVE registers from it. The reserved internal-register encodings (TEMP
    // 124..127) select internal registers, which no shipped load targets.
    let dest_n = bits(word, 27, 21);
    let dest_bank = if bits(word, 39, 39) == 0 { Bank::Temp } else { Bank::PrimaryAttr };
    if dest_bank == Bank::Temp && dest_n + elements as u32 > 124 {
        set(&mut blocked, "0xE8 memory-load destination reaches the reserved TEMP range");
    }
    let dest = Operand::plain(dest_bank, r7_reg_index(dest_n), bits(word, 39, 39) as u8);

    Instr {
        op: Op::MemLoad { elements, offset_bytes },
        pred: Predicate::Always,
        dest: Some(dest),
        // NOT meaningful for this op: the written span is `elements` consecutive registers
        // (up to 16), which four channel bits cannot carry. Consumers read `elements`.
        write_mask: [true; 4],
        srcs: vec![src0],
        half_precision: false,
        raw: word,
        group: 0x1d,
        blocked,
    }
}

/// Classify an instruction whose operand decode is not yet wired: set its operation from
/// the ISA opcode map (a fact) but leave operands empty so the emitter hard-fails naming
/// the op. `hi`/`lo` are used only where a sub-opcode is needed to name the op.
/// Why one stub group is blocked, where saying more than "not yet wired" would send the next
/// reader somewhere useful.
fn stub_reason(op1: u8) -> &'static str {
    match op1 {
        // The last group in the whole captured corpus that stops a shader recompiling, and it
        // is worth knowing WHY it is not simply the next grind item.
        //
        // 0x80 is SOP2, the 8-bit sum-of-products combiner WITHOUT a write mask; its sibling
        // 0x90 (SOP2M, "+ wmask") is fully decoded here. The two cannot share a table: SOP2M
        // spends bits 46:43 on the mask and SOP2 does not, so SOP2's freed bits carry
        // something else and one word cannot say what. And the corpus carries exactly ONE
        // group-0x80 word, in ONE program - a two-instruction fragment (a PHAS and this),
        // whose neighbours therefore constrain nothing. Reading it through SOP2M's table
        // yields a defensible-looking `dest = sub(a * src1, (1-a) * src2)` with no way to
        // check any of it, and a guessed combiner COEFFICIENT scales a whole term.
        //
        // What would settle it: a second program carrying the group (so the varying bits name
        // the fields), or an operand-level reference for the SOP family - the distilled core
        // spec lists SOP2/SOP2M/SOP3 as explicitly out of its scope.
        0x10 => "0x80 SOP2 (sum-of-products, no write mask): its operand layout differs from \
                 the decoded 0x90 SOP2M exactly where that group puts its write mask, and the \
                 corpus carries ONE word of it in ONE program - see `stub_reason`",
        _ => "operand decode not yet wired for this group",
    }
}

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
        0x14 | 0x15 => Op::Todo("mad (integer group)"),
        0x19 => Op::Todo("mad.u8"),
        0x1c => Op::Todo("tex"),               // 0xE0
        // 0x1d (0xE8 loads) is handled in decode() with full operand decode for the
        // established variant; only the STORE half of the memory-access format remains a stub.
        0x1e => Op::Todo("sta32/stl32/stt32"), // 0xF0
        0x1f => Op::Todo("flow (0xF8 complex)"),
        // 0x09/0x0f are the TEST group (VTST / VTSTMSK): a compare that writes a predicate
        // (VTST) or a per-channel mask (VTSTMSK). Fully decoded from clean facts, but the
        // real captured VTST tests a GLOBAL special hardware register (`p0 = (GLOBAL[16] &
        // imm) != 0`), which the WGSL register-file model does not model - so it stays a
        // named Todo (hard-fail) until GLOBAL register semantics are established, rather than
        // guessing a value. See the distilled VTST/VTSTMSK spec notes.
        0x09 => Op::Todo("vtst (test->predicate; reads GLOBAL special reg, not modeled)"),
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
        // fails naming this, rather than emitting with missing operands. A group whose
        // specific wall is worth naming says so instead of using the generic line.
        blocked: Some(stub_reason(op1)),
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

    /// The group-0x1a IMAD32-STEP layout, checked against the two shapes the corpus carries -
    /// a golf title's world vertex programs, where every one of its eighteen words is one of
    /// these four.
    ///
    /// SHAPE 1 is the ADDRESS computation: `pa[0] += sa[107]`, done as the two halves of a
    /// 16x32 multiplier with the immediate 1. Its `src0` reaches the SECATTR bank through this
    /// group's own extension bit, which is where the driver put the bound buffer's pointer.
    ///
    /// SHAPE 2 is the LOOP-COUNTER increment: `pa[17] += 1`, whose two steps must use a scratch
    /// register because the destination is also `src0`.
    #[test]
    fn group_1a_imad32_step_closes_on_the_corpus_words() {
        // Shape 1, step 0 and step 1: identical but for the step selector.
        for (word, high) in [(0xd092_8006_a01a_c080u64, true), (0xd082_8006_a01a_c080, false)] {
            let instr = decode(word);
            assert_eq!(instr.group, 0x1a);
            assert_eq!(instr.op, Op::IntMadStep { signed: false, high_half: high });
            assert_eq!(instr.pred, Predicate::Always);
            assert_eq!(instr.blocked, None, "the layout must close with nothing blocked");
            let dest = instr.dest.expect("a decoded destination");
            assert_eq!((dest.bank, dest.index), (Bank::PrimaryAttr, 0));
            // src0 is the driver-placed POINTER, in SECATTR - reachable only through bit 47.
            assert_eq!((instr.srcs[0].bank, instr.srcs[0].index), (Bank::SecondaryAttr, 107));
            assert_eq!((instr.srcs[1].bank, instr.srcs[1].index), (Bank::Immediate, 1));
            assert_eq!((instr.srcs[2].bank, instr.srcs[2].index), (Bank::PrimaryAttr, 0));
            assert_eq!(instr.write_mask, [true, false, false, false]);
            assert_eq!(repeat_extra_iterations(word), Some(0));
        }
        // Shape 2: the counter increment. Step 0 writes a scratch and adds the literal 1;
        // step 1 writes the counter and adds the scratch.
        let lo = decode(0xd083_0006_a004_4081);
        assert_eq!(lo.op, Op::IntMadStep { signed: false, high_half: false });
        assert_eq!(lo.blocked, None);
        assert_eq!(lo.dest.map(|d| (d.bank, d.index)), Some((Bank::PrimaryAttr, 0)));
        assert_eq!((lo.srcs[0].bank, lo.srcs[0].index), (Bank::PrimaryAttr, 17));
        assert_eq!((lo.srcs[1].bank, lo.srcs[1].index), (Bank::Immediate, 1));
        assert_eq!((lo.srcs[2].bank, lo.srcs[2].index), (Bank::Immediate, 1));
        let hi = decode(0xd092_0006_a224_4080);
        assert_eq!(hi.op, Op::IntMadStep { signed: false, high_half: true });
        assert_eq!(hi.blocked, None);
        assert_eq!(hi.dest.map(|d| (d.bank, d.index)), Some((Bank::PrimaryAttr, 17)));
        assert_eq!((hi.srcs[0].bank, hi.srcs[0].index), (Bank::PrimaryAttr, 17));
        assert_eq!((hi.srcs[1].bank, hi.srcs[1].index), (Bank::Immediate, 1));
        assert_eq!((hi.srcs[2].bank, hi.srcs[2].index), (Bank::PrimaryAttr, 0));
    }

    /// Every group-0x1a form whose FIELD the corpus establishes but whose MEANING it does not
    /// must hard-fail rather than decode through a table fitted to the unsigned, unrepeated,
    /// unnegated one.
    #[test]
    fn group_1a_blocks_what_it_cannot_establish() {
        let base: u64 = 0xd082_8006_a01a_c080;
        for (name, word) in [
            ("reserved bits 43:42", base | 1 << 42),
            ("reserved bits 38:35", base | 1 << 35),
            ("sn value 2 (bit 53)", base | 1 << 53),
            ("a repeat count", base | 1 << 44),
            ("the signed form", base | 1 << 41),
            ("a negated src1", base | 1 << 40),
            ("a negated src2", base | 1 << 39),
            ("an extended destination bank", base | 1 << 51),
        ] {
            assert!(
                decode(word).blocked.is_some(),
                "{name} must block rather than decode through this layout"
            );
        }
    }

    /// The group-0x78 VTSTMSK layout, checked against the ONE word the whole corpus carries -
    /// the per-texel depth comparison of a golf title's shadow filter, which appears in three
    /// of its world fragment programs and nowhere else.
    ///
    /// The reading is `i0[c] = (r[c] - i0[c]) > 0 ? 1 : 0` for all four channels: the four
    /// gathered shadow-map depths against the reference depth a `mov` broadcast into `i0`,
    /// written back as a numeric mask the next instruction dots against the sample's bilinear
    /// coefficients.
    #[test]
    fn group_78_vtstmsk_closes_on_the_word_that_established_it() {
        let instr = decode(0x78c0_8aa0_0f93_807c);
        assert_eq!(instr.group, 0x0f);
        assert_eq!(instr.op, Op::TestMask { alu: TestAlu::Sub, cmp: TestCmp::Gt });
        assert_eq!(instr.pred, Predicate::Always);
        assert_eq!(instr.blocked, None, "the layout must close with nothing blocked");
        // Destination and src2 are the SAME internal register: the reference depth goes in and
        // the mask comes back out.
        let dest = instr.dest.expect("a decoded destination");
        assert_eq!((dest.bank, dest.index), (Bank::Internal, 0));
        assert_eq!((instr.srcs[0].bank, instr.srcs[0].index), (Bank::Temp, 0));
        assert_eq!((instr.srcs[1].bank, instr.srcs[1].index), (Bank::Internal, 0));
        // A mask is per-channel, so all four are written.
        assert_eq!(instr.write_mask, [true; 4]);
        assert_eq!(repeat_extra_iterations(0x78c0_8aa0_0f93_807c), Some(0));
    }

    /// The VTSTMSK forms the corpus does not establish stay blocked - the two bit-pattern mask
    /// types, the second test flag, and a write-back-disabled word that would write nothing.
    #[test]
    fn group_78_vtstmsk_blocks_what_it_cannot_establish() {
        let base: u64 = 0x78c0_8aa0_0f93_807c;
        for (name, word) in [
            ("the 8-bit mask type", base & !(0b11 << 36)),
            ("the precision mask type", (base & !(0b11 << 36)) | (1 << 36)),
            ("test_flag_2", base | 1 << 50),
            ("test_wben clear", base & !(1 << 20)),
        ] {
            assert!(decode(word).blocked.is_some(), "{name} must block");
        }
    }

    /// The group-0xE0 GATHER form, on the word a golf title's shadow filter uses: a 2D
    /// implicit-LOD gather into `r0`, whose four F16 bilinear coefficients follow at `r4`.
    #[test]
    fn group_e0_gather4_decodes_the_corpus_word() {
        let instr = decode(0xe001_c464_e001_0580);
        assert_eq!(instr.group, 0x1c);
        // The sampler ORDINAL is what the raw word carries; `decode_shader` resolves it to a
        // GXM texture unit against the container's own table.
        assert_eq!(instr.op, Op::TexGather { unit: 11, coords: 2, coord_half: false });
        assert_eq!(instr.blocked, None, "the gather form must decode");
        assert_eq!(instr.dest.map(|d| (d.bank, d.index)), Some((Bank::Temp, 0)));
        assert!(!instr.half_precision, "this title's gather asks for the F32 result");
    }

    /// The gather sub-forms with no corpus instance, and the gather features whose layout is
    /// not established, stay blocked by name.
    #[test]
    fn group_e0_blocks_the_gather_forms_it_cannot_establish() {
        let base: u64 = 0xe001_c464_e001_0580;
        // sb_mode 1 (a bare gather4) and 2 (the texture-info query).
        let sb = |v: u64| (base & !(0b11 << 37)) | (v << 37);
        assert!(decode(sb(1)).blocked.is_some(), "sb_mode 1 must block");
        assert!(decode(sb(2)).blocked.is_some(), "sb_mode 2 must block");
        // A gather with an explicit LOD, and one with a non-2D coordinate.
        assert!(decode(base | (2 << 40)).blocked.is_some(), "an explicit-LOD gather must block");
        assert!(
            decode((base & !(0b11 << 42)) | (2 << 42)).blocked.is_some(),
            "a 3D gather must block"
        );
        // An F16 result: where the coefficients land beside a half-width gather is not known.
        assert!(
            decode((base & !(0b11 << 46)) | (2 << 46)).blocked.is_some(),
            "an F16-result gather must block"
        );
    }
    /// The group-0x15 IMAD32 layout, checked against the one real word that established it -
    /// instruction #3 of a shipped vertex program, at code byte 0x18.
    ///
    /// The reading is `pa[2] = pa[2] * 48 + sa[24]`, signed, unpredicated, one iteration. What
    /// makes it a decode rather than a plausible story is that it CLOSES: all four reserved-zero
    /// groups read zero, every register number is inside its 7-bit field, the width selector
    /// holds its one defined value, and the arithmetic is the shape a vertex program's
    /// matrix-palette address computation has - an index scaled by a 48-byte stride (three vec4
    /// rows) offset by a uniform base. Nothing in the word is left over.
    #[test]
    fn group_15_imad32_closes_on_the_word_that_established_it() {
        let instr = decode(0xa882_0886_b040_9818);
        assert_eq!(instr.group, 0x15);
        assert_eq!(instr.op, Op::IntMad { signed: true, bits: 32 });
        assert_eq!(instr.pred, Predicate::Always);
        assert_eq!(instr.blocked, None, "the layout must close with nothing blocked");
        let dest = instr.dest.expect("a decoded destination");
        assert_eq!((dest.bank, dest.index), (Bank::PrimaryAttr, 2));
        assert_eq!((instr.srcs[0].bank, instr.srcs[0].index), (Bank::PrimaryAttr, 2));
        // src1 is an IMMEDIATE literal, not a register: its bank-extension bit is set and its
        // bank selector names the immediate row, so the 7-bit number IS the value.
        assert_eq!((instr.srcs[1].bank, instr.srcs[1].index), (Bank::Immediate, 48));
        assert_eq!((instr.srcs[2].bank, instr.srcs[2].index), (Bank::SecondaryAttr, 24));
        // Scalar: the group carries no write mask, so exactly one channel is written.
        assert_eq!(instr.write_mask, [true, false, false, false]);
        // And the repeat encoding agrees: three bits at 46:44, reading zero here.
        assert_eq!(repeat_extra_iterations(0xa882_0886_b040_9818), Some(0));
    }

    /// A group-0x15 word that sets a bit the layout requires to be zero is a DIFFERENT encoding,
    /// and must hard-fail rather than be decoded through a table that was never fitted to it.
    /// Same for the forms whose fields are established but whose meaning is not - saturation,
    /// a narrower width, and a non-zero repeat.
    #[test]
    fn group_15_imad32_blocks_what_it_cannot_establish() {
        let base: u64 = 0xa882_0886_b040_9818;
        for (name, word) in [
            ("reserved bit 47", base | 1 << 47),
            ("reserved bits 41:40", base | 1 << 40),
            ("reserved bits 37:35", base | 1 << 35),
            ("saturation", base | 1 << 42),
            ("a repeat count", base | 1 << 44),
        ] {
            assert!(
                decode(word).blocked.is_some(),
                "{name} must block rather than decode through this layout"
            );
        }
        // The width selector: 2 is the established 32-bit value, so clearing it to a 16-bit
        // form must block.
        let narrow = (base & !(0b11 << 38)) | (1 << 38);
        assert!(decode(narrow).blocked.is_some(), "a 16-bit width must block");
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
            // `unk7` is the top bit of the repeat field at 47:44 and its "runs once" value is
            // 1 - see `dot_repeat_extra`. Every real DOT in the corpus sets it.
            &[("opcode1", 0x03), ("opcode2", 0), ("c3_en", 1), ("predicate", 0), ("unk7", 1),
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

    /// A DOT whose repeat field says four iterations is a 4x4 matrix transform: the four
    /// executions write four consecutive destination LANES from four consecutive source
    /// VECTORS, with the internal-register operand standing still. This is the real word out
    /// of a retail title's world vertex program - `WorldViewProjection * position` into clip
    /// position - and the whole title renders black if it is emitted once. See
    /// [`dot_repeat_extra`].
    #[test]
    fn a_repeating_dot_walks_lanes_and_vectors() {
        let word = 0x18903081c011a200u64;
        let ins = decode(word);
        assert_eq!(ins.op, Op::Dot { components: 4 });
        assert!(ins.blocked.is_none(), "{:?}", ins.blocked);
        assert_eq!(repeat_extra_iterations(word), Some(3));
        // Single-channel destination mask: the per-iteration step is a channel, so the form
        // is only modelled when exactly one channel is named.
        assert_eq!(ins.write_mask, [true, false, false, false]);
        let d = ins.dest.unwrap();
        assert_eq!((d.bank, d.index), (Bank::Output, 0));
        assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::SecondaryAttr, 0));
        assert_eq!((ins.srcs[1].bank, ins.srcs[1].index), (Bank::Internal, 0));
        // dest one lane, op1 one four-component vector, the internal operand still.
        let ops = repeat_operands(word).expect("dot repeat operands");
        assert_eq!(ops.len(), 3);
        assert_eq!((ops[0].slot, ops[0].stride), (0, 1));
        assert_eq!((ops[1].slot, ops[1].stride), (2, 4));
        assert_eq!((ops[2].slot, ops[2].stride), (3, 0));
    }

    /// The value the census never shows must BLOCK rather than pick a plausible count.
    #[test]
    fn a_dot_repeat_field_outside_the_census_blocks() {
        // Bit 47 set AND a low bit set - never observed in 379 real DOTs.
        let word = 0x18903081c011a200u64 | (1 << 47);
        assert_eq!(repeat_extra_iterations(word), None);
        assert!(decode(word).blocked.is_some());
    }

    #[test]
    fn dot_18_three_channel_and_reserved_internal_src() {
        // c3_en=0 -> 3 channels, op2 swizzle from the 3ch table[4] = xyz.
        let hi = encode(G18_DOT_HIGH, &[("opcode1", 0x03), ("opcode2", 0), ("c3_en", 0), ("unk7", 1)]);
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
    fn mad_18_blocks_on_strange_dest_and_swizzles_through_the_vec4_table() {
        // op0_strange set -> blocked (undocumented dest adjustment).
        let hi = encode(G18_MAD_HIGH, &[("opcode1", 0x03), ("opcode2", 1), ("op0_strange0", 1)]);
        assert!(decode(word(hi, 0)).blocked.is_some());
        // The MAD indexes the VEC4 half of the vec34 scheme, not the vec3 half. Index 4 is
        // the one that matters most - it is the plain `xyzw` an object-to-clip transform uses,
        // and reading it through the vec3 table gives `xyzx`, which drops the y and z rows'
        // contribution to clip w and collapses the mesh.
        let hi4 = encode(G18_MAD_HIGH, &[("opcode1", 0x03), ("opcode2", 1)]);
        let lo4 = encode(G18_MAD_LOW, &[("swz_alt_op1", 0b001), ("op1_swz", 0)]);
        let ins = decode(word(hi4, lo4));
        assert_eq!(ins.op, Op::Mad);
        assert_eq!(ins.srcs[0].swizzle, [0, 1, 2, 3], "index 4 is xyzw in the vec4 table");
        // Index 24, the vec3 table's undocumented `h` entry, is an ordinary lane pattern here,
        // so a mad that lands on it decodes rather than blocking.
        // Index 23 stays the constant `111` of the extended half - the translation column of
        // every object-to-clip transform multiplies by it.
        let lo23 = encode(G18_MAD_LOW, &[("swz_alt_op1", 0b101), ("op1_swz", 0b11)]);
        assert_eq!(decode(word(hi4, lo23)).srcs[0].swizzle, [5, 5, 5, 0]);
        // The extended half still carries the undocumented `h` entry, which blocks.
        let lo24 = encode(G18_MAD_LOW, &[("swz_alt_op1", 0b110), ("op1_swz", 0)]);
        assert!(decode(word(hi4, lo24)).blocked.is_some(), "h-swizzle must block");
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
        // (sel=1,ext=0) reg 4 -> pa8; sampler src1 = 5; dest_use_pa=0 -> temp `dest_n` 3 -> r3.
        //
        // The DESTINATION is direct at either precision - it is not double-register scaled the
        // way an ALU operand is (see `decode_grp_tex`, and the corpus closure test that settles
        // it). This assertion said r6 while that was decoded as `2*dest_n`.
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
        assert_eq!((ins.dest.unwrap().bank, ins.dest.unwrap().index), (Bank::Temp, 3));
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

    /// A float->integer VPCK is a TRUNCATING cast when `scale` is clear and a NORMALIZE when it
    /// is set, and the two differ by a factor of the format's range - so they decode to
    /// DIFFERENT operations. The unscaled form is what a shader computing an array INDEX in
    /// float emits before indexing with it; the scaled U8 form is a fragment epilogue writing
    /// an 8-bit surface, and it is emittable because `Prec::Fx8` already carries the packed
    /// representation. The scaled forms of the OTHER widths do not have one and stay blocked.
    #[test]
    fn pack_float_to_int_converts_unscaled_and_normalizes_only_into_u8() {
        // VPCK U8<-F32 (src_fmt=6, dest_fmt=0), scale (bit 18) clear: a truncating cast.
        let w = word_bits(&[(0x08, 63, 59), (6, 43, 41), (0, 40, 38)]);
        let ins = decode(w);
        assert_eq!(ins.op, Op::PackToInt { bits: 8, signed: false, src_half: false });
        assert!(ins.blocked.is_none(), "unscaled float->int converts: {:?}", ins.blocked);
        // The same word with `scale` set is the NORMALIZED conversion - a different number by a
        // factor of 255, and a different operation.
        let scaled = word_bits(&[(0x08, 63, 59), (6, 43, 41), (0, 40, 38), (1, 18, 18)]);
        let ins = decode(scaled);
        assert_eq!(ins.op, Op::PackUnorm8 { to_unorm8: true, float_half: false });
        assert!(ins.blocked.is_none(), "normalized U8 converts: {:?}", ins.blocked);
        // ...and the other direction, U8 -> float, is the same conversion run backwards.
        let back = word_bits(&[(0x08, 63, 59), (0, 43, 41), (5, 40, 38), (1, 18, 18)]);
        assert_eq!(decode(back).op, Op::PackUnorm8 { to_unorm8: false, float_half: true });
        // A normalized S16 has no packed representation in this register model, so it stays
        // blocked where the U8 form no longer does.
        let s16_norm = word_bits(&[(0x08, 63, 59), (6, 43, 41), (4, 40, 38), (1, 18, 18)]);
        assert!(decode(s16_norm).blocked.is_some(), "normalized S16 must stay blocked");
        // S16<-F32 is the width the one shader that needs this uses, and it is SIGNED.
        let s16 = word_bits(&[(0x08, 63, 59), (6, 43, 41), (4, 40, 38)]);
        assert_eq!(decode(s16).op, Op::PackToInt { bits: 16, signed: true, src_half: false });
        // C10 (7) on the destination is a packed representation this model does not carry.
        let c10 = word_bits(&[(0x08, 63, 59), (6, 43, 41), (7, 40, 38)]);
        assert!(decode(c10).blocked.is_some(), "C10 destination must stay blocked");
    }

    /// Group 0x80 SOP2 in the fragment-epilogue form, on the two words a title's five
    /// fragment programs actually carry, and the refusal that keeps every other form out.
    ///
    /// The words are captured, not constructed: both end a program whose previous
    /// instruction is `pack.unorm8 pa[0] <- pa[0]`, and this decode has to name `pa[0]` as the
    /// source and the OUTPUT bank as the destination for that chain to close. See
    /// [`decode_grp_sop2`].
    #[test]
    fn sop2_decodes_the_fragment_epilogue_and_refuses_every_other_form() {
        // The third word is a SECOND title's, and it is here because that program's own chain
        // settles it: two instructions, `Nop` then this, a PDS-prefetched sample in `pa[0]`,
        // and NOTHING that writes `o[0]` - so the term SOP2M's reading puts over `src2 = o[0]`
        // stands over a register the program cannot have written. It differs from the first
        // word at 42:41 alone, which is why that field is the one this decode widened.
        for word in [0x8090_80d9_9000_0000u64, 0x8090_80c1_9000_0000, 0x8090_82dd_9000_0000] {
            let ins = decode(word);
            assert_eq!(ins.op, Op::CopyFx8, "{word:#x}");
            assert!(ins.blocked.is_none(), "{word:#x}: {:?}", ins.blocked);
            assert_eq!(ins.write_mask, [true; 4], "no write mask field: all four bytes");
            let src = ins.srcs.first().expect("one source");
            assert_eq!((src.bank, src.index), (Bank::PrimaryAttr, 0), "src is what the pack wrote");
            let dest = ins.dest.as_ref().expect("a destination");
            assert_eq!((dest.bank, dest.index), (Bank::Output, 0), "dest is the colour register");
        }
        // The SWAPPED shape: `src1` names the output register and `src2` the packed colour, so
        // the source has to come from the other slot. Its twin `frag_81a7f590` carries the word
        // above, byte-identical program otherwise - see [`decode_grp_sop2`].
        let swapped = decode(0x8190_0021_6004_0000u64);
        assert_eq!(swapped.op, Op::CopyFx8);
        assert!(swapped.blocked.is_none(), "{:?}", swapped.blocked);
        let src = swapped.srcs.first().expect("one source");
        assert_eq!(
            (src.bank, src.index),
            (Bank::PrimaryAttr, 0),
            "the source is the slot that is not the output register"
        );
        let dest = swapped.dest.as_ref().expect("a destination");
        assert_eq!((dest.bank, dest.index), (Bank::Output, 0));
        // ...and the shape is pinned: the same word with `mod2` set is a form nothing
        // establishes and must block rather than emit a copy.
        assert!(decode(0x8190_0021_6004_0000u64 | (1 << 47)).blocked.is_some());

        // One bit outside the operand fields - here the "cop" position the epilogue pins to 1
        // - is a form nothing establishes, and it must refuse rather than emit a copy.
        let other = 0x8090_80d9_9000_0000u64 & !(1 << 52);
        assert!(decode(other).blocked.is_some(), "an unestablished 0x80 form must block");
        // ...and so must one whose repeat-count position is not the pinned `1000`.
        let repeated = 0x8090_80d9_9000_0000u64 | (1 << 44);
        assert!(
            crate::usse::decode::repeat_extra_iterations(repeated).is_none(),
            "an unpinned repeat field must block rather than assume no repeat"
        );
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
        assert_eq!(ins.op, Op::Bitwise { kind: BitwiseKind::And, imm: Some(0xFF), lane_bits: 32 });
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

    /// The five-instruction ALPHA-TEST MACRO that group 0x90 (SOP2M) and the INT8 test family
    /// exist for, decoded end to end from one real fragment program:
    ///
    /// ```text
    /// #3  vtst  p0 = (t0.x - sa8) <= 0        sa8 is the reference, 0.01
    /// #4  sop2  t0.x = 1 * sa9                sa9 is the literal 1 - the "discard" flag
    /// #5  or    t0.x = sa7 | 0     IF NOT p0  sa7 is the literal 0 - the "keep" flag
    /// #6  vtst  p1 = (t0.x - sa7) == 0        in the 8-BIT family
    /// #7  kill                     IF NOT p1
    /// ```
    ///
    /// This one chain pins every field at once, which is why it is worth more than any of the
    /// assertions below taken separately:
    ///
    /// * The SOP2M destination must be the register the OR writes and the VTST reads. Those
    ///   two are decoded by unrelated code paths (the bitwise group and the test group), so a
    ///   wrong bank or number field here cannot accidentally agree with them.
    /// * The write mask must name the channel `chan_cc` reduces on. It is the one field whose
    ///   raw encoding is ROTATED, and reading it unrotated names a different channel.
    /// * The coefficient must come out as ONE, not zero. The selector picks a FACTOR that
    ///   multiplies the named source, so `Zero` plus the complement bit is a copy of sa9 -
    ///   whereas reading the selector as the OPERAND makes this instruction a constant, leaves
    ///   sa9 unread, and makes the macro discard every pixel of every draw.
    /// * The 8-bit test must read its operands as BYTES. sa9 is the bit pattern 0x00000001,
    ///   which as an f32 is a denormal that compares equal to zero - so an F32 read of the same
    ///   registers turns the whole macro into a no-op that draws every cut-out texel opaque.
    #[test]
    fn the_real_alpha_test_macro_decodes_end_to_end() {
        // #4 and #6 of one program; #5 and #7 are decoded by groups already covered.
        const SOP2M: u64 = 0x91811000e0000480;
        const VTST8: u64 = 0x48880185300a0007;

        let s = decode(SOP2M);
        assert!(s.blocked.is_none(), "{:?}", s.blocked);
        assert_eq!(
            s.op,
            Op::Sop2 {
                color: SopOp::Add,
                alpha: SopOp::Add,
                f1: SopFactor::Zero,
                f1_complement: true,
                f2: SopFactor::Zero,
                f2_complement: false,
            },
            "coefficient 1 - 0 = 1 on the first term, 0 on the second: a copy of src1"
        );
        assert_eq!(s.pred, Predicate::Always, "the flag is set unconditionally");
        let d = s.dest.expect("SOP2M writes a register");
        assert_eq!((d.bank, d.index), (Bank::Temp, 0), "the register the OR and the VTST use");
        assert_eq!(
            s.write_mask,
            [true, false, false, false],
            "raw mask 0b0010 rotates to 0b0001 - channel 0, which is what chan_cc reduces on"
        );
        assert_eq!(
            (s.srcs[0].bank, s.srcs[0].index),
            (Bank::SecondaryAttr, 9),
            "src1 is the literal 1, and it is READ - the selector is a coefficient, not the operand"
        );
        assert_eq!(
            (s.srcs[1].bank, s.srcs[1].index),
            (Bank::Immediate, 0),
            "src2 is an inline immediate, multiplied by a zero coefficient"
        );

        let t = decode(VTST8);
        assert!(t.blocked.is_none(), "{:?}", t.blocked);
        assert_eq!(
            t.op,
            Op::Test {
                alu: TestAlu::Fx8Sub,
                cmp: TestCmp::Eq,
                reduce: TestReduce::Channel(0),
                pdst: 1,
                write_back: false,
            }
        );
        assert!(!t.half_precision, "the 8-bit family ignores the float precision bit");
        assert_eq!(
            (t.srcs[0].bank, t.srcs[0].index),
            (Bank::Temp, 0),
            "src1 is the flag register the SOP2M above wrote"
        );
        assert_eq!(
            (t.srcs[1].bank, t.srcs[1].index),
            (Bank::SecondaryAttr, 7),
            "src2 is the literal 0, NOT double-register scaled: an 8-bit operand is a direct              register number, and scaling it would read sa14 instead"
        );
    }

    /// The SECOND program's pair, which is the same macro on a different register. It is here
    /// because agreement across two destinations is what makes the destination-field decode a
    /// measurement rather than a coincidence: both words differ from the pair above in exactly
    /// the destination bank and number, and both still name the register their own VTST reads.
    #[test]
    fn the_alpha_test_macro_agrees_on_a_second_destination() {
        let s = decode(0x91811002e0200480);
        assert!(s.blocked.is_none(), "{:?}", s.blocked);
        let d = s.dest.expect("SOP2M writes a register");
        assert_eq!((d.bank, d.index), (Bank::PrimaryAttr, 1));
        assert_eq!((s.srcs[0].bank, s.srcs[0].index), (Bank::SecondaryAttr, 9));

        let t = decode(0x48880185b00a0087);
        assert!(t.blocked.is_none(), "{:?}", t.blocked);
        assert_eq!(
            (t.srcs[0].bank, t.srcs[0].index),
            (Bank::PrimaryAttr, 1),
            "the test reads exactly what the combiner wrote"
        );
    }

    /// A selector value the reference material does not establish must BLOCK. A combiner
    /// coefficient scales a whole term, so a guessed one is a wrong picture with no other
    /// symptom - the failure mode this family was left unwired for in the first place.
    #[test]
    fn an_unestablished_sop2m_selector_blocks() {
        // The same word with sel1 = 1 (bits 40:38), which no source consulted defines.
        let word = 0x91811000e0000480 | (1u64 << 38);
        let s = decode(word);
        assert!(s.blocked.is_some(), "selector 1 is not established and must not be guessed");
    }

    /// The one real DEPTHF in the corpus, and the two instructions around it that say what it
    /// writes:
    ///
    /// ```text
    /// #1  add    r0.x = sa0 (kDepthBias) + pa6 (the POSITION interpolant's z)
    /// #2  depthf r0
    /// ```
    ///
    /// It is also the instruction that PROVED a fragment's `Position` interpolant is the
    /// window coordinate rather than the interpolated clip position: a depth write only
    /// type-checks against a value already in depth-buffer space, and clip `z` is not.
    #[test]
    fn real_depthf_writes_the_temp_the_previous_add_filled() {
        const ADD: u64 = 0x08a40084e0041003;
        const DEPTHF: u64 = 0xfb300000f0000183;

        let a = decode(ADD);
        assert!(a.blocked.is_none(), "{:?}", a.blocked);
        let d = a.dest.expect("the add has a register destination");
        assert_eq!((d.bank, d.index), (Bank::Temp, 0));

        let f = decode(DEPTHF);
        assert_eq!(f.op, Op::DepthF);
        assert!(f.blocked.is_none(), "{:?}", f.blocked);
        assert!(f.dest.is_none(), "a depth write has no REGISTER destination");
        assert_eq!(
            (f.srcs[0].bank, f.srcs[0].index),
            (Bank::Temp, 0),
            "src0 (bank bit 36 + ext bit 51, number bits 20:14) names the temp the add wrote"
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

    /// The 0x40 PACK group reads its extended source row too, and what it was blocking is the
    /// operand a retail title's fragment SECONDARY programs are built out of.
    ///
    /// `0x40830b5e60014f01` is `pack.f16 sa[0] <- <src1>`, with `src1_sel = 1` and
    /// `src1_ext` SET, so the row is SPECIAL; the raw field is 15 and its `0x40` bit is clear,
    /// so the operand is FPCONSTANT[15] - undoubled, because SPECIAL and IMMEDIATE never are.
    ///
    /// Read through the ORDINARY row it came out `OUTPUT[15]`, doubled to 30 and then forced to
    /// SECATTR by the secondary-program remap: `sa[30]`, against a container that declares TEN
    /// uniform registers. The linker refused the pair for reading past its uniform buffer, and
    /// the pair is that title's WORLD material. The corpus says the same thing from the other
    /// side: three of its programs read exactly this operand, with uniform buffers of 10, 6 and
    /// 10 registers - a fixed register number under three different layouts is a CONSTANT.
    #[test]
    fn pack_extended_src1_is_the_hardware_constant_bank_not_a_doubled_register() {
        const PACK_CONST: u64 = 0x40830b5e60014f01;
        let ins = decode(PACK_CONST);
        assert!(ins.blocked.is_none(), "extended-bank src1 must decode: {:?}", ins.blocked);
        assert_eq!(ins.op, Op::Pack { src_half: true });
        assert_eq!(
            (ins.srcs[0].bank, ins.srcs[0].index),
            (Bank::Constant, 15),
            "src1_ext + selector 1 + field 15 is FPCONSTANT[15], undoubled"
        );
        // The ordinary row is unchanged: clear the extension bit and the same field is the
        // doubled register the group has always read.
        let plain = decode(PACK_CONST & !(1 << 49));
        assert_eq!((plain.srcs[0].bank, plain.srcs[0].index), (Bank::Output, 30));
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
        assert!(matches!(ins.op, Op::Bitwise { kind: crate::ir::BitwiseKind::Or, imm: Some(0), lane_bits: 32 }));
        assert_eq!(
            (ins.srcs[0].bank, ins.srcs[0].index),
            (Bank::Constant, 4),
            "src1_ext + selector 1 + field 4 is FPCONSTANT[4], undoubled"
        );
        assert_eq!(ins.dest.as_ref().map(|d| (d.bank, d.index)), Some((Bank::PrimaryAttr, 8)));
    }

    /// The SMLSI decode, pinned on real corpus words - including the SIGN of the increment.
    ///
    /// The captured vertex program that carries SMLSI genuinely REPEATS: its `mov` (group 0x38)
    /// encodes `repeat_count = 3` in bits 45:44, so that one instruction executes four times
    /// with the stepping SMLSI configures. Reading the disassembly as though every instruction
    /// ran once - which is how the surrounding code looks - gets this wrong.
    #[test]
    fn smlsi_decodes_its_per_slot_stepping_including_negative_increments() {
        // vert_82c14da0's prologue: SMLSI, then a VMOV with repeat_count 3.
        const SMLSI: u64 = 0xfa10000201014e01;
        const MOV_REPEAT_3: u64 = 0x3880352183080080;
        assert_eq!(bits(MOV_REPEAT_3, 45, 44), 3, "the VMOV really does encode a repeat");
        assert_eq!(
            decode_smlsi(SMLSI),
            [
                SmlsiSlot::Increment(1),  // dest
                SmlsiSlot::Swizzle(0x4e), // src0 - a slot an unconditional VMOV does not have
                SmlsiSlot::Increment(1),  // src1
                SmlsiSlot::Increment(1),  // src2
            ]
        );

        // A racing title's track programs, where the high bytes are BACKWARD steps rather than
        // 248-register forward leaps. 0xfa140000ff01ff01 sets src0 and src2 to -1;
        // 0xfa1000000601f801 sets src0 to -8 and src2 to +6.
        assert_eq!(
            decode_smlsi(0xfa140000ff01ff01),
            [SmlsiSlot::Increment(1), SmlsiSlot::Increment(-1), SmlsiSlot::Increment(1), SmlsiSlot::Increment(-1)]
        );
        assert_eq!(
            decode_smlsi(0xfa1000000601f801),
            [SmlsiSlot::Increment(1), SmlsiSlot::Increment(-8), SmlsiSlot::Increment(1), SmlsiSlot::Increment(6)]
        );

        // The mode bits at [35:32] put a slot in swizzle mode, and they are per-slot: this word
        // is swizzle on src2 (bit 35) and increments elsewhere.
        assert_eq!(
            decode_smlsi(0xfa10000838010201),
            [SmlsiSlot::Increment(1), SmlsiSlot::Increment(2), SmlsiSlot::Increment(1), SmlsiSlot::Swizzle(0x38)]
        );

        for w in [0xfa10000201010e01u64, 0xfa14000001010101, 0xfa1400000101f601] {
            assert!(is_smlsi(w));
        }
    }

    /// Which SMLSI byte governs each operand, and what one unit of it moves.
    ///
    /// The strides are not a tuning parameter: they restate the operand field WIDTHS this
    /// decoder already uses. Getting one wrong steps a repeated instruction onto the wrong
    /// register, which produces a shader that compiles and paints the wrong thing.
    #[test]
    fn repeat_operand_slots_and_strides_follow_the_operand_field_widths() {
        // 0x38 VMOV, unconditional (move_type 0): dest and src1 only, both six-bit fields.
        const MOV_REPEAT_3: u64 = 0x3880352183080080;
        assert_eq!(bits(MOV_REPEAT_3, 47, 46), 0, "unconditional form");
        assert_eq!(
            repeat_operands(MOV_REPEAT_3),
            Some(vec![
                RepeatOperand { slot: 0, stride: 2, moe: true },
                RepeatOperand { slot: 2, stride: 2, moe: true },
            ])
        );
        // The conditional form gains src2 and src0, in the order `decode_grp_38` pushes them.
        let cond = MOV_REPEAT_3 | (1 << 46);
        assert_eq!(
            repeat_operands(cond),
            Some(vec![
                RepeatOperand { slot: 0, stride: 2, moe: true },
                RepeatOperand { slot: 2, stride: 2, moe: true },
                RepeatOperand { slot: 3, stride: 2, moe: true },
                RepeatOperand { slot: 1, stride: 2, moe: true },
            ])
        );

        // 0x40 VPCK: a SEVEN-bit destination and a SIX-bit source - the reference's own
        // "(dest,src1,src2) = (1,2,2) for a float source", recovered from the field widths.
        //
        // Its source's SMLSI SLOT is 1, not the 2 the sibling VMOV above uses. Measured on the
        // corpus; the reasoning is at the match arm. The two groups differing here is the whole
        // point of the assertion - a slot is a property of a group's field table, not a global.
        let vpck = 0x40u64 << 56;
        assert_eq!(opcode1(vpck), 0x08);
        assert_eq!(
            repeat_operands(vpck),
            Some(vec![
                RepeatOperand { slot: 0, stride: 1, moe: true },
                RepeatOperand { slot: 1, stride: 2, moe: true },
            ])
        );

        // 0x50 VBW: every operand is a seven-bit field - the reference's "all 1".
        let vbw = 0x50u64 << 56;
        assert_eq!(opcode1(vbw), 0x0a);
        assert!(repeat_operands(vbw).unwrap().iter().all(|o| o.stride == 1));

        // 0x18 DP: the destination's channel walk is the instruction's OWN, so it carries
        // `moe: false` and steps one lane whatever the SMLSI programmed - see the match arm.
        // The vector source IS MOE-governed, and its stride is a whole vector.
        let dp = (0x18u64 << 56) | (1 << 52) | (2 << 44);
        assert_eq!(opcode1(dp), 0x03);
        assert_eq!(bits(dp, 53, 53), 0, "the DOT form, not the MAD form");
        assert_eq!(
            repeat_operands(dp),
            Some(vec![
                RepeatOperand { slot: 0, stride: 1, moe: false },
                RepeatOperand { slot: 2, stride: 4, moe: true },
                RepeatOperand { slot: 3, stride: 0, moe: true },
            ])
        );

        // A group with no established operand grammar must answer "unknown", never a default.
        assert_eq!(repeat_operands(0xa000_0000_0000_0000), None);
    }

    /// The SPECIAL row's GLOBAL half DECODES (which register it names is as structural as
    /// which constant its FPCONSTANT half names); what a GLOBAL register CONTAINS is settled
    /// per index by the emitter, so decode leaves `blocked` clear here and the emitter
    /// hard-fails on any index it has not established. The INDEXED modes name a sub-bank and an
    /// offset out of the same 7-bit field, which only a group whose number field really is
    /// seven bits wide may split - so this group resolves them and a six-bit group does not.
    #[test]
    fn group_50_extended_src1_names_the_global_register_and_resolves_indexed_banks() {
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
        let ix = decode(MOV_INDEXED);
        assert!(ix.blocked.is_none(), "INDEXED1 resolves in a 7-bit group: {:?}", ix.blocked);
        assert_eq!(ix.srcs[0].bank, Bank::Indexed);
        assert_eq!(ix.srcs[0].bank_sel, 0, "selector 0 is INDEXED1, which uses index register 0");

        // The real instruction from a retail vertex program: number 110 = 0b1101110 splits into
        // sub-bank 3 (SECATTR) and offset 14, which is what makes `sa[i0 + 14]` the element.
        const REAL_INDEXED_READ: u64 = 0x5083100a20003700;
        let r = decode(REAL_INDEXED_READ);
        assert!(r.blocked.is_none(), "{:?}", r.blocked);
        assert_eq!(r.srcs[0].bank, Bank::Indexed);
        assert_eq!(r.srcs[0].index, 110);
        assert_eq!(crate::ir::indexed_sub_bank(r.srcs[0].index), Bank::SecondaryAttr);
        assert_eq!(crate::ir::indexed_offset(r.srcs[0].index), 14);
        // It repeats once, so it walks TWO consecutive elements - which is how one instruction
        // reads a whole two-component array entry.
        assert_eq!(repeat_extra_iterations(REAL_INDEXED_READ), Some(1));
    }

    /// Group 0x14 (I16MAD) has no published layout, so the ONE encoding the corpus establishes
    /// decodes and every other word of the group hard-fails. That asymmetry is the point: the
    /// corpus can say what this instruction is, and cannot say what a different one would be.
    #[test]
    fn i16mad_decodes_only_the_index_load_the_corpus_establishes() {
        // The six real words differ only in bits [17:14], the source register.
        for (word, reg) in [
            (0xa08b_0946_a022_0088u64, 8u8),
            (0xa08b_0946_a021_8088, 6),
            (0xa08b_0946_a022_c088, 11),
        ] {
            let ins = decode(word);
            assert!(ins.blocked.is_none(), "{word:#018x}: {:?}", ins.blocked);
            assert_eq!(ins.op, Op::LoadIndex { addend: I16MAD_LOAD_INDEX_ADDEND });
            assert_eq!(ins.dest.unwrap().bank, Bank::Index);
            assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::PrimaryAttr, reg));
            assert_eq!(repeat_extra_iterations(word), Some(0));
        }
        // Any other group-0x14 word is a different instruction whose fields are not decodable
        // from this corpus - including its repeat encoding, which is why it must not be zero.
        let other = 0xa08b_0946_a022_0089u64;
        assert!(decode(other).blocked.is_some(), "an unestablished 0x14 encoding must block");
        assert_eq!(repeat_extra_iterations(other), None);
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
        assert_eq!(ins.op, Op::Bitwise { kind: BitwiseKind::Shr, imm: None, lane_bits: 32 });
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
    /// scalar from src2 and map exactly onto a WGSL sample function; the GRADIENT form reads
    /// both derivative vectors from src2 and decodes for a 2D coordinate, where the split is
    /// spec E0.4's "first 2 components ddx, next 2 ddy" and no register has to be guessed at.
    /// A gradient with a 1D or 3D coordinate, and the gather/info sub-behaviours, stay blocked.
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

        // The real 2D gradient sample this title draws its terrain with (dim 1 = 2D,
        // lod_mode 3). It decodes, and it reads its derivatives from src2 like every other
        // extra operand.
        let grad = decode(0xe080878ce0408101);
        assert!(grad.blocked.is_none(), "{:?}", grad.blocked);
        assert!(matches!(grad.op, Op::Tex { coords: 2, lod: TexLod::Gradient, .. }));
        assert_eq!(grad.srcs.len(), 2, "gradient reads both derivative vectors from src2");

        // A gradient with a 3D coordinate still blocks: its second derivative vector starts in
        // a register this decoder has no evidence for, and guessing samples the wrong mip.
        let grad3d = word_bits(&[(0x1c, 63, 59), (3, 41, 40), (2, 43, 42)]);
        assert!(decode(grad3d).blocked.is_some_and(|b| b.contains("gradient")));
    }

    #[test]
    fn stub_groups_classified_and_lossless() {
        // Every documented group classifies its operation (a fact) and preserves the raw
        // word, even where operand decode is not yet wired. A load (0xE8) whose
        // mode/addr_mode is outside the established zero variant decodes but BLOCKS by name.
        let load = decode(word(0xE800_0000 | 0x1234, 0xabcd_ef01));
        assert_eq!(load.group, 0x1d);
        assert!(matches!(load.op, Op::MemLoad { .. }));
        assert_eq!(load.raw, word(0xE800_0000 | 0x1234, 0xabcd_ef01));
        assert!(!load.is_supported(), "a non-zero mode selector is not established");
        assert!(load.blocked.is_some_and(|b| b.contains("mode value not established")));
        assert!(load.is_classified(), "load operation is known");

        // Stores (0xF0) remain a stub.
        let store = decode(word(0xF000_0000, 0));
        assert_eq!(store.group, 0x1e);
        assert_eq!(store.op, Op::Todo("sta32/stl32/stt32"));
        assert!(!store.is_supported(), "stores not wired for emit yet");

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

    /// The four memory loads of the shipped skinning vertex program that established this
    /// group's decodable variant (the census: every 0x1d/0x1e instruction in every captured
    /// blob is `mode = 0, addr_mode = 0`, 32-bit, PA-bank pointer, immediate offsets).
    /// Field-by-field expectations from the closed bit table, checked against the idiom the
    /// surrounding code spells out: a 12-word (4x3 matrix) fetch through the pointer in
    /// pa[3], and three 4-word row fetches through pa[2] at element offsets 0 / 4 / 8.
    #[test]
    fn memory_load_established_variant_decodes() {
        let cases: [(u64, u8, u32, Bank, u8, u8); 4] = [
            // raw, elements, offset_bytes, dest bank, dest reg, src0 pa reg
            (0xe883b004a000c000, 12, 0, Bank::Temp, 0, 3),
            (0xe8833084a0808000, 4, 0, Bank::PrimaryAttr, 4, 2),
            (0xe8833084a0808200, 4, 16, Bank::PrimaryAttr, 4, 2),
            (0xe8833004a0808400, 4, 32, Bank::Temp, 4, 2),
        ];
        for (raw, elements, offset_bytes, dbank, dreg, s0reg) in cases {
            let i = decode(raw);
            assert_eq!(i.group, 0x1d, "{raw:#018x}");
            assert_eq!(i.blocked, None, "{raw:#018x}");
            assert_eq!(i.op, Op::MemLoad { elements, offset_bytes }, "{raw:#018x}");
            let d = i.dest.expect("load has a register destination");
            assert_eq!((d.bank, d.index), (dbank, dreg), "{raw:#018x}");
            assert_eq!(i.srcs.len(), 1);
            assert_eq!((i.srcs[0].bank, i.srcs[0].index), (Bank::PrimaryAttr, s0reg), "{raw:#018x}");
            assert_eq!(i.pred, Predicate::Always);
            // The group has no repeat field - the element count IS the transfer.
            assert_eq!(repeat_extra_iterations(raw), Some(0));
        }

        // Departures from the census block BY NAME rather than decode loosely: flip
        // addr_mode, the data type, and the src0 bank on the first real word.
        let base = 0xe883b004a000c000u64;
        let addr = base | (1 << 42);
        assert!(decode(addr).blocked.is_some_and(|b| b.contains("addr_mode")));
        let half = base | (1 << 36);
        assert!(decode(half).blocked.is_some_and(|b| b.contains("data_type")));
        // An SA pointer is the second census shape (the driver's own buffer address, in the
        // register the +0x78 binding table names) and DECODES.
        let sa_ptr = base | (1 << 50); // ext=1, sel=1 -> SECATTR pointer
        let sa_i = decode(sa_ptr);
        assert_eq!(sa_i.blocked, None, "{sa_ptr:#018x}");
        assert_eq!(sa_i.srcs[0].bank, Bank::SecondaryAttr);
        // A TEMP pointer is in neither, and still blocks by name.
        let temp_ptr = base & !(1 << 34); // ext=0, sel=0 -> TEMP pointer
        assert!(decode(temp_ptr).blocked.is_some_and(|b| b.contains("PA/SA banks")));
        let reg_off = base & !(1 << 49); // src1 ext cleared -> register offset row
        assert!(decode(reg_off).blocked.is_some_and(|b| b.contains("register-supplied")));
    }
}
