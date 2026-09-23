//! The intermediate representation the USSE decoder produces and the WGSL emitter
//! consumes.
//!
//! Two layers, deliberately separated by what is a *fact* vs a *hypothesis*:
//!
//! * [`RawInstr`] - the faithful structural decode. Every field comes straight from the
//!   psvgxp `grammar.json` bit layout for the instruction's group, so it is a fact:
//!   opcode fields, per-operand register index + bank + swizzle + abs/neg modifiers,
//!   predicate, destination write mask. This never guesses.
//!
//! * [`Op`] - the interpreted operation. Only variants whose meaning is an established
//!   clean-room fact are ever produced; anything else is [`Op::Unsupported`]. The WGSL
//!   emitter HARD-FAILS (naming the opcode) on an unsupported op rather than guess or
//!   silently degrade, so a wrong translation can never paint a pixel.

use crate::container::ProgramKind;

/// A USSE register bank (the 2-bit operand bank selector). The bank *names* are facts
/// (psdevwiki / gxm reflection): PA = primary attributes = the interpolated fragment
/// inputs / vertex iterators; SA = secondary attributes = the default uniform buffer /
/// constants; Temp = general scratch; Output = the result registers feeding the pixel
/// back end; Internal = the SGX internal/index registers.
///
/// The numeric value->bank *mapping* is not fully pinned from clean facts, so the
/// decoder records the raw 2-bit selector alongside a best-effort classification and
/// the emitter only trusts banks it can corroborate against the parameter table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bank {
    Temp,
    PrimaryAttr,
    Output,
    SecondaryAttr,
    Internal,
    /// Not a register bank: an inline float constant selected from the CNST6 table (the
    /// operand's `index` holds the 6-bit CNST6 selector). Used when an operand is in
    /// constant mode (`alt_opt` set + the constant sub-mode). The emitter materialises the
    /// exact 32-bit value; it carries no swizzle meaning (a scalar broadcast).
    Constant,
    /// A SPECIAL hardware register ("GLOBAL" bank): pipeline state the shader reads but no
    /// program writes, selected by the extension row when the raw register field carries the
    /// `0x40` discriminator (`index` holds the remaining 6-bit selector).
    ///
    /// Decoding it is a fact; giving it a VALUE is not. The emitter hard-fails on every index
    /// whose meaning has not been established, naming the index, so a GLOBAL read can never
    /// silently become a zero.
    Global,
    /// Not a register bank: an inline INTEGER literal assembled by the instruction's own
    /// group (`index` holds the value). The extension row names IMMEDIATE, but how the
    /// literal is assembled is group-specific, so only the groups that establish it produce
    /// this - for the TEST group it is the 7-bit `src2_n`, zero-extended (spec T.5b step 6).
    Immediate,
    /// Register-INDIRECT addressing (the extension row's INDEXED1 / INDEXED2 banks): the
    /// element read is `bank[index_register + offset]`, where the bank and the offset come
    /// from the operand's own 7-bit number (bits[6:5] select TEMP/OUTPUT/PRIMATTR/SECATTR,
    /// bits[4:0] are the offset) and the index register is `i0` for INDEXED1, `i1` for
    /// INDEXED2. [`Operand::index`] holds the raw 7-bit number; [`Operand::bank_sel`] holds
    /// which index register, so both halves survive to the emitter.
    ///
    /// This is what a shader that indexes a uniform ARRAY by a value it computed compiles to,
    /// and it is the only operand form whose address is not known until the shader runs.
    Indexed,
    /// The INDEX register file (`i0`, `i1`) itself, as a DESTINATION. Only the integer-MAD
    /// groups write it; nothing reads it except [`Bank::Indexed`] addressing.
    Index,
    /// A bank selector value not yet mapped to a named bank.
    Raw(u8),
}

/// Which sub-bank an [`Bank::Indexed`] operand's 7-bit number names (bits [6:5]).
pub fn indexed_sub_bank(number: u8) -> Bank {
    match (number >> 5) & 3 {
        0 => Bank::Temp,
        1 => Bank::Output,
        2 => Bank::PrimaryAttr,
        _ => Bank::SecondaryAttr,
    }
}

/// The additive offset an [`Bank::Indexed`] operand's 7-bit number carries (bits [4:0]).
pub fn indexed_offset(number: u8) -> u32 {
    (number & 0x1f) as u32
}

/// One decoded source or destination operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operand {
    pub bank: Bank,
    /// Register index within the bank (6 or 7 bits depending on group).
    pub index: u8,
    /// The raw 2-bit bank selector as encoded, preserved so the oracle harness can
    /// correlate it with the parameter table without losing information.
    pub bank_sel: u8,
    /// Per-component swizzle selectors (x,y,z,w). Values are the raw 3-bit (or 2-bit,
    /// zero-extended) selector fields; [`Swizzle`] interprets them.
    pub swizzle: [u8; 4],
    pub abs: bool,
    pub neg: bool,
}

impl Operand {
    /// A plain, unmodified `.xyzw` operand from a bank/index (used by tests + emit).
    pub fn plain(bank: Bank, index: u8, bank_sel: u8) -> Operand {
        Operand { bank, index, bank_sel, swizzle: [0, 1, 2, 3], abs: false, neg: false }
    }
}

/// Predication state of an instruction (the small predicate field). `Always` executes
/// unconditionally; the others gate on a predicate register the flow ops set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Predicate {
    Always,
    /// Execute if predicate register `n` is set.
    IfP(u8),
    /// Execute if predicate register `n` is clear.
    IfNotP(u8),
    /// A predicate encoding not yet classified (carried raw).
    Raw(u8),
}

/// The specific integer bitwise/shift operation of a [`Op::Bitwise`] (group 0x50). AND/OR/
/// XOR are bitwise; SHL is logical left shift; SHR is logical (zero-fill) right shift; ASR
/// is arithmetic (sign-fill) right shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitwiseKind {
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Asr,
}

/// The zero-test a conditional move (VMOVC, group 0x38) applies to its test source per
/// channel, from the SGX543 spec compare-method table (B.1b). The test is always against
/// the constant 0; the selected value is `src1` when the test is true, else `src2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareMethod {
    /// src0 == 0
    EqZero,
    /// src0 != 0
    NeZero,
    /// src0 < 0
    LtZero,
    /// src0 <= 0
    LteZero,
}

/// The ALU operation a test instruction (VTST, group 0x48) evaluates before comparing the
/// result against zero. Chosen by the encoding's `alu_sel` family plus `alu_op` (spec tables
/// T-2a/T-2b). Only the members the corpus actually encodes are modelled; anything else is
/// decoded but blocked, so an unmodelled family can never be silently mistranslated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestAlu {
    /// `src1' + src2` (VADD).
    Add,
    /// `src1' - src2` (VSUB) - the form a two-operand relational compare uses.
    Sub,
    /// `src1' * src2` (VMUL).
    Mul,
    /// `src1 & src2` on the raw 32-bit lane (the BITWISE family's AND).
    BitAnd,
    /// `src1 << src2` on the raw 32-bit lane (the BITWISE family's SHIFT LEFT, `alu_op` 3).
    ///
    /// A distinct member rather than a flag on [`Self::BitAnd`] for the reason the whole family
    /// is split up: the operation decides what the tested value IS. The one idiom that encodes
    /// it shifts a `0x0000FFFF`-or-zero mask left by 31, which keeps only the mask's bit 0 -
    /// reading that as an AND would test five bits instead of one.
    ///
    /// The family's operands are UNSIGNED 32-bit, so a result of `0x80000000` is a large
    /// positive number and not a negative one. Both spec sources state that, and it is the
    /// difference between this test passing and failing.
    BitShl,
    /// `src1' - src2` in the 8-BIT fixed-point pipeline (the INT8 family's FPSUB8), where a
    /// register holds four 8-bit unsigned-normalised channels rather than one float.
    ///
    /// It is a distinct member rather than a flag on [`Self::Sub`] because the ALU family
    /// changes how the OPERANDS ARE READ, not just what is done to them: reading an 8-bit
    /// lane as an F32 register turns the byte pattern 0x00000001 into a denormal, which
    /// compares equal to zero and silently disables the alpha test the corpus uses this for.
    Fx8Sub,
    /// `src1' - src2` in the INTEGER pipeline (the INT16 family), on the raw 32-bit lane bit
    /// pattern rather than through a float view.
    ///
    /// It is a distinct member for the same reason [`Self::Fx8Sub`] is: the ALU FAMILY decides
    /// how the operands are READ. The instruction that needs it tests the result of a bitwise
    /// AND - a small integer mask - and reading `0x00000008` as an f32 makes it a denormal.
    /// That happens to survive a `!= 0` test and would not survive a `< 0` one, which is
    /// exactly the kind of near-miss this codebase refuses to leave in.
    IntSub,
    /// `src1 - src2` as UNSIGNED 16-BIT integers (the 16/32-bit integer family's `alu_op` 10),
    /// on the raw lane's low half.
    ///
    /// Distinct from [`Self::IntSub`], which is the same family's 32-BIT subtract, because the
    /// WIDTH changes the answer and this codebase does not round that off: the device compares
    /// only the low 16 bits, so two lanes that differ ONLY above bit 15 are equal to the
    /// hardware and not equal to a 32-bit compare. That is a silent wrong answer in exactly the
    /// direction a facing or stencil-style test would take.
    IntSub16U,
}

/// One term's coefficient in the 8-bit sum-of-products combiner ([`Op::Sop2`]). The field
/// selects a FACTOR that multiplies the term's own source register - it does not select the
/// operand. That distinction is the whole instruction: with the factor `Zero` and the term's
/// complement bit set, the coefficient becomes `1 - 0 = 1` and the term is a plain copy of
/// its source, which is exactly how the corpus's alpha-test macro moves a register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SopFactor {
    /// Constant 0 in every channel.
    Zero,
    /// Source 1's own channel value.
    Src1Color,
    /// Source 1's alpha, broadcast to every channel.
    Src1Alpha,
    /// Source 2's own channel value.
    Src2Color,
    /// Source 2's alpha, broadcast to every channel.
    Src2Alpha,
}

/// The per-channel operation the 8-bit combiner applies to its two terms ([`Op::Sop2`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SopOp {
    Add,
    Sub,
    Min,
    Max,
}

/// The per-channel boolean a test instruction forms from its ALU result `r` against zero.
/// Assembled at decode time from `sign_test` (STST), `zero_test` (ZTST) and the AND/OR
/// combiner, so the emitter sees one comparison rather than three fields (spec T.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestCmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// How a test instruction reduces its four per-channel booleans into the single bit written
/// to the destination predicate register (spec table T-4, `chan_cc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestReduce {
    /// Take channel `n`'s boolean (SELECT0..SELECT3).
    Channel(u8),
    /// AND of all four channels (ANDALL).
    AndAll,
    /// OR of all four channels (ORALL).
    OrAll,
}

/// How a texture sample (group 0xE0 SMP) supplies its mip level, from the encoding's
/// `lod_mode` field (spec E0.4). `Bias` and `Level` read a scalar from src2, `Gradient` reads
/// two derivative vectors from it, and `Implicit` reads nothing and lets the hardware derive
/// the level from the coordinate derivatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexLod {
    /// lod_mode 0 - hardware-derived level (`textureSample`).
    Implicit,
    /// lod_mode 1 - src2 is added to the derived level (`textureSampleBias`).
    Bias,
    /// lod_mode 2 - src2 IS the level (`textureSampleLevel`).
    Level,
    /// lod_mode 3 - src2 carries the explicit derivatives (`textureSampleGrad`). For a 2D
    /// sample the first two components are `ddx` and the next two are `ddy` (spec E0.4).
    Gradient,
}

/// The interpreted operation, classified from the henkaku SGX543 opcode map (a fact for
/// every documented instruction). Being *classified* is separate from being *emittable*:
/// [`crate::wgsl`] translates the ops it has fully wired and hard-fails (naming the op) on
/// the rest, so classification can be complete while emit coverage grows incrementally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Multiply-add: `dest = src1 * src2 + src3` (group 0x00, and the multi-op groups).
    Mad,
    /// `dest = src1 * src2` (group 0x08/0x10 opcode2 0).
    Mul,
    /// `dest = src1 + src2` (opcode2 1).
    Add,
    /// Fractional part `dest = fract(src1)` (opcode2 2).
    Frc,
    /// Screen-space partial derivatives (opcode2 3/4) -> WGSL `dpdx`/`dpdy`.
    Dsx,
    Dsy,
    /// `dest = min(src1, src2)` / `max(...)` (opcode2 5/6).
    Min,
    Max,
    /// Dot product (opcode2 7 in 0x08/0x10; the whole 0x18 group). `components` = 3 or 4.
    Dot { components: u8 },
    /// Reciprocal / reciprocal-sqrt / log2 / exp2 (group 0x30, unary).
    Rcp,
    Rsq,
    Log,
    Exp,
    /// Move (group 0x38, unconditional VMOV). `dest = src1` (a swizzled per-channel copy).
    Mov,
    /// Conditional move (group 0x38, VMOVC). Per written channel:
    /// `dest.c = test(src0.c) ? src1.c : src2.c`, where `test` is `CompareMethod` against 0
    /// (equivalent to WGSL `select(src2.c, src1.c, cond)`). Source order in `srcs` is
    /// `[src1 (true value), src2 (false value), src0 (test value)]`.
    Cmov { test: CompareMethod },
    /// A no-operation: an instruction that carries no data effect on the register file the
    /// emitter models (a group 0xF8 phase declaration `PHAS` or an explicit `NOP`). It is a
    /// FACT that these produce no arithmetic/data result; the emitter emits nothing for them,
    /// so a shader that is otherwise fully wired is not blocked by its mandatory phase header.
    Nop,
    /// Integer bitwise / shift (group 0x50, VBW): a scalar (channel 0) operation on the
    /// 32-bit lane bit patterns. `imm` is the assembled inline source-2 constant when source
    /// 2 is an immediate (already rotated/inverted at decode); otherwise source 2 is a
    /// register (`srcs[1]`). Emitted via `bitcast<u32>`.
    Bitwise { kind: BitwiseKind, imm: Option<u32>, lane_bits: u8 },
    /// The 8-BIT FIXED-POINT SUM-OF-PRODUCTS combiner (group 0x12, "SOP2M"): a register is
    /// four 8-bit unsigned-normalised channels, and the instruction computes
    ///
    /// ```text
    ///   term1  = coeff(f1, complement1) * src1
    ///   term2  = coeff(f2, complement2) * src2
    ///   rgb    = color(term1.rgb, term2.rgb)
    ///   alpha  = alpha(term1.a,   term2.a)
    /// ```
    ///
    /// The RGB channels and the alpha channel carry INDEPENDENT operations over the SAME two
    /// terms, which is what makes this one instruction rather than two. See [`SopFactor`] for
    /// why the coefficient is not the operand.
    Sop2 {
        color: SopOp,
        alpha: SopOp,
        f1: SopFactor,
        /// `f1` is used as `1 - f1` (a one's complement of the COEFFICIENT, not of the source).
        f1_complement: bool,
        f2: SopFactor,
        f2_complement: bool,
    },
    /// Format pack/convert (group 0x40, VPCK). A float<->float repack (F16<->F32) preserves
    /// the NUMBER while changing its STORAGE width, so it is emitted like a move - but the
    /// source and destination are read and written at their own precisions, which is the whole
    /// point of the instruction: `src_half` is the source format (VPCK's `src_fmt`), while the
    /// instruction's `half_precision` carries the destination format. The integer<->float
    /// normalized conversions and the C10/O8 packed formats change the numeric value and are
    /// decoded but blocked (their exact layout is not established).
    Pack { src_half: bool },
    /// VPCK converting a FLOAT source to an INTEGER destination with `scale` clear - a
    /// truncating numeric cast, not a normalize. `bits` is the destination width (8, 16 or 32)
    /// and `signed` its signedness; the result is stored as the integer's two's-complement bit
    /// pattern in the destination lane, which is the same representation the integer groups
    /// (VBW, the integer MADs) read and write. The normalized (`scale` set) and C10/O8 forms
    /// stay blocked - they change the value by a factor this does not model.
    PackToInt { bits: u8, signed: bool, src_half: bool },
    /// VPCK converting a 16-BIT INTEGER source to a FLOAT destination with `scale` clear - the
    /// exact mirror of [`Op::PackToInt`], and the same truncating-cast reading: a widening
    /// integer-to-float convert changes no value, so there is nothing here to guess once the
    /// direction is decoded.
    ///
    /// The source is HALF a register, addressed exactly as [`Op::PackToInt`]'s 16-bit
    /// destination writes it (`crate::wgsl::Dest::store_raw_half`) and exactly as the
    /// group-0x15 IMAD32s read it back through their own `src0_high` bit - so this reads the
    /// pairs those two already agree on rather than introducing a third packing.
    ///
    /// `bits` is the SOURCE width, 16 or 8.
    ///
    /// >>> THE 8-BIT WIDTH IS THE SAME RULE, NOT A SECOND ONE. A component selector always
    /// names the c-th element of the source's stream starting at the operand's own register,
    /// and the only thing the width changes is how many of them a register holds: four 16-bit
    /// halves span TWO registers (`index + (sel >> 1)`, half `sel & 1`), four 8-bit BYTES fit
    /// in ONE (`index`, byte `sel`). A football title's fragment secondaries establish it
    /// directly - four consecutive `U8 -> F16` packs read selectors 0,1,2,3 off ONE `sa[36]`
    /// and the primary stream then dots the four results against one sampled RGBA, which is
    /// the title's own `EBR`/`EBG`/`EBB`/`EBA` parameters, declared `Uniform U8 comps 1` and
    /// four to a register. A per-register reading would have made all four the same byte.
    ///
    /// This is a QUARTER of a register, which is the reason the width was refused - and the
    /// model does carry that quarter: [`Self::source_packed_bytes`] already describes an
    /// operand as four channels in one register for the `fx8` ops, and every span computation
    /// asks it.
    PackFromInt { bits: u8, signed: bool },
    /// LIMM (group 0xF8, `op2 = 100`, `opcat = 10`): `dest = <32-bit immediate>`.
    ///
    /// The value is a RAW 32-bit pattern, typed by whoever reads it - the corpus carries it
    /// used both as an integer (`0x7FFFFFFF` selected into an INT32 conditional move) and as a
    /// float (`0xCF000000`, i.e. `-2^31`, as the identity a running maximum starts from). So it
    /// is stored through `Dest::store_raw` with no view applied, exactly as `Op::PackToInt`'s
    /// result is.
    ///
    /// # The layout, and what would refute it
    /// DESTINATION: bank `[33:32]` (the 2-bit selector every other group uses) plus number
    /// `[27:21]`, UNDOUBLED - the ordinary seven-bit destination. Established by liveness over
    /// every LIMM in a football title's corpus: a brute-force sweep of every 7-bit window in
    /// the word, scored by whether the register it names is READ before anything overwrites it,
    /// leaves `[27:21]` undoubled well clear of every rival (`limm_layout_candidates_by_liveness`).
    /// Two closures pin it directly, one instruction apart in both cases:
    ///   * `#45 LIMM number 5` then `#46 pa[0] = IntMad(pa[0], 16, pa[5])`. Five is ODD, so no
    ///     doubled field can name it.
    ///   * `#18 LIMM number 7` then `#19 VMOVCU8 ... ? pa[7] : pa[12]`, where `pa[12]` is the
    ///     register the two `PackToInt16` instructions just above wrote. The whole triple only
    ///     lines up with both operands undoubled.
    ///
    /// IMMEDIATE: three fields, `[20:0]` low, `[40:36]` next, `[48:44]` next, and bit 54 as the
    /// TOP bit - which is what the ISA note means when it says the value is assembled from
    /// three fields at positions that collide with the opcode discriminant. It is not fitted:
    /// the corpus's three distinct immediates all come out canonical under it and under no
    /// other split tried - `0x7FFFFFFF` (INT_MAX, selected into an integer move), `0xCF000000`
    /// (`-2^31` as a float, broadcast to four registers and then used as the floor of a `>=`
    /// running maximum), and `0x00010000` (65,536, added as a byte offset into a bound buffer
    /// one instruction before the load that reads it).
    ///
    /// WHAT WOULD REFUTE IT: an immediate whose top bit is set where the value is plainly
    /// meant to be positive, or any LIMM whose destination under this reading is written again
    /// before it is read.
    Limm { value: u32 },
    /// VMOVCU8 (group 0x38, `move_type = 2`): a BYTE-WISE conditional move.
    ///
    /// `dest.byte[c] = test(src0.byte[c]) ? src1.byte[c] : src2.byte[c]` for each byte the
    /// write mask names - the four mask bits are the four bytes of ONE register, not four
    /// registers, and every operand is a raw lane read undoubled. See `decode_grp_38` for the
    /// three one-instruction-apart closures in a football title's skinned vertex programs that
    /// establish the channel count and the numbering, and why neither depends on the data type
    /// field (nothing here is read through a float view).
    ///
    /// The sources are ordered `[src1, src2, src0]`, the same order [`Op::Cmov`] uses.
    CmovU8 { test: CompareMethod },
    /// VPCK whose SOURCE and DESTINATION are both 16-BIT INTEGERS with `scale` clear - a
    /// same-width integer copy, which converts nothing at all. There is no numeric reading to
    /// establish here and nothing to invent: the bits of the source half ARE the bits of the
    /// destination half, and both ends already exist - the source is read exactly as
    /// [`Op::PackFromInt`] reads one and the destination written exactly as [`Op::PackToInt`]'s
    /// 16-bit form writes one (`crate::wgsl::Dest::store_raw_half`), so this introduces no
    /// third packing.
    ///
    /// Signedness is not carried because at equal widths it cannot matter: the two's-complement
    /// pattern is the same whichever way each end is labelled, and the sign is applied by
    /// whoever later WIDENS the half ([`Op::PackFromInt`], the group-0x15 IMAD32s), not here.
    ///
    /// `bits` is that shared width - 16, or 8 for the BYTE form, which is the same statement
    /// one element down: the source byte's bits are the destination byte's bits, read as
    /// [`Op::PackFromInt`]'s 8-bit source reads one and written as [`Op::PackToInt`]'s 8-bit
    /// destination writes one.
    PackIntCopy { bits: u8 },
    /// VPCK converting between a FLOAT and a U8 with `scale` SET - the NORMALIZED
    /// conversion, where the byte range 0..255 maps onto 0.0..1.0. This is how a fragment
    /// program that computes in F16 writes an 8-bit-per-channel surface, and how it reads
    /// one back.
    ///
    /// It is emittable, where the other normalized widths are not, because the packed U8
    /// representation is one this model ALREADY carries: [`crate::wgsl::Prec::Fx8`] reads a
    /// register as four `byte/255` channels for the SOP2M combiner, and its store is the
    /// same rounded `clamp(v,0,1)*255` this conversion performs. So there is nothing to
    /// invent - the two directions are that precision on one side and the float precision
    /// on the other. `to_unorm8` says which way; `float_half` is the FLOAT side's
    /// precision, whichever side that is.
    ///
    /// S8/U16/S16 normalized stay blocked: no packed representation for them exists here,
    /// and inventing one would put a wrong value in a register that reads back plausibly.
    PackUnorm8 { to_unorm8: bool, float_half: bool },
    /// A whole-register copy in the FOUR-BYTE view: `dest = src1`, all four unorm8 channels.
    /// This is the group-0x80 SOP2 form a fragment epilogue ends with - see
    /// `decode_grp_sop2`, which explains what the corpus establishes about it and what it
    /// pins rather than reads.
    CopyFx8,
    /// Integer multiply-add (group 0x15, IMAD32): `dest = half(src0) * src1 + src2`, scalar,
    /// on the 32-bit lane read as an integer of the given signedness. `bits` is the operand
    /// width; only 32 is decoded, because the narrower selector values are encoded but not
    /// established (the decoder blocks them by name rather than guessing).
    ///
    /// `src0_high` is the group's own bit 56, and it selects which 16-bit HALF of `src0` feeds
    /// the multiplier - the same shape the sibling group 0x1a spells as
    /// [`Op::IntMadStep`]. It is not a refinement: `src0` in this group is ALWAYS half a packed
    /// pair. Over five titles' corpora every one of the 122 IMAD32s reads a register a 16-bit
    /// PACK wrote, and 42 of them set this bit, so reading the whole 32-bit register makes the
    /// pairs collapse - a four-bone skinned mesh fetches two matrices, each twice.
    ///
    /// This shares no encoding with [`Op::LoadIndex`]'s group 0x14 despite the neighbouring
    /// opcode: the two groups carry different field layouts, and reading one through the
    /// other's table is how a "similar" group silently addresses the wrong registers.
    /// `src1_high` is the sibling selector at bit 53, and it picks the 16-bit half of `src1`
    /// the multiplier sees, exactly as `src0_high` does for `src0`. It was zero on every word
    /// of five titles' corpora and so was refused by name; a sixth title's SKINNED vertex
    /// programs set it, and refusing it dropped every one of them.
    IntMad { signed: bool, bits: u8, src0_high: bool, src1_high: bool },
    /// One STEP of a 32-bit integer multiply-add (group 0x1a, the second 32-bit form):
    /// `dest = half(src0) * src1 + src2`, where `high_half` selects which 16-bit half of
    /// `src0` feeds the multiplier and whether its product is shifted back up:
    ///
    /// ```text
    ///   high_half = false:  dest = (src0 & 0xffff) * src1 + src2
    ///   high_half = true:   dest = ((src0 >> 16) * src1) << 16 + src2
    /// ```
    ///
    /// The two steps therefore SUM to the full `src0 * src1 + src2` in 32-bit wrapping
    /// arithmetic, which is how the hardware's 16x32 multiplier builds a 32x32 product. See
    /// the decoder for why this reading rather than one of its rivals, and for the pairing
    /// rule that makes the result independent of the remaining ambiguity.
    IntMadStep { signed: bool, high_half: bool },
    /// Load an INDEX register (group 0x14, I16MAD, in the one encoding the corpus establishes):
    /// `i[dest] = src + addend`, as a 16-bit integer.
    ///
    /// The corpus is the whole authority for this and it is narrow: group 0x14 occurs in ONE
    /// program across three titles, six times, and the only bits that ever vary are [17:14],
    /// the source register number. Every other bit is constant, so the encoding establishes a
    /// register and nothing else. `addend` is fixed instead by ARITHMETIC CLOSURE against the
    /// container's own parameter table - see the decoder - and any group-0x14 word that is not
    /// this exact encoding must hard-fail rather than inherit that assumption.
    /// `to_index` says WHERE the sum goes, and the two destinations are different instructions
    /// wearing one opcode. Set: the INDEX REGISTER, for a later register-indirect read - the
    /// form the particle title establishes and the one this doc paragraph describes. Clear: an
    /// ORDINARY REGISTER named by the word itself, which is the form a football title's
    /// skinning uses - `LoadIndex ; IntMad ; MemLoad`, where the IMAD32 reads that register as
    /// its `src0` and turns the blend index into a byte offset into the matrix palette. The
    /// two are told apart by the `(b8, b51)` flag; see `decode_grp_i16mad`.
    /// `stride` is how far apart two consecutive INDEX values' blocks of rows are, in rows.
    /// Meaningful only when `to_index` is clear; see `resolve_index_load_stride`, which reads it
    /// off the program. 1 leaves the value `src + addend`, which is what an unresolved stream
    /// decodes to.
    LoadIndex { addend: i32, to_index: bool, stride: u8 },
    /// Texture sample (group 0xE0). `unit` is the GXM texture unit, which
    /// [`crate::usse::decode_shader`] resolves from the instruction's raw sampler-register
    /// field through the container's texture-control table; `coords` is the number of
    /// coordinate components (1 or 2 - the coordinate vector is `srcs[0]`, read
    /// `bank[base + 0..coords]`). The sampled RGBA is written to the destination's four
    /// channels. Implicit-LOD normal samples only; the gather/info/bias/gradient/3D variants
    /// are decoded but blocked until wired.
    /// `lod` selects the sample variant; for `Bias`/`Level` the scalar level operand is
    /// `srcs[1]`.
    Tex { unit: u8, coords: u8, coord_half: bool, lod: TexLod },
    /// GATHER4 with bilinear coefficients (group 0xE0 SMP, `sb_mode == 3`): one 2x2 texel
    /// footprint of the bound texture, plus the two fractional weights that a bilinear filter
    /// of the same footprint would use.
    ///
    /// It writes SIX registers where an ordinary sample writes four - `dest + 0..3` are the
    /// four gathered texels at the instruction's result precision, and `dest + 4..5` hold four
    /// F16 coefficients - which is why it is its own operation rather than a flag on
    /// [`Op::Tex`]: the destination extent is part of what the opcode means.
    ///
    /// Only the ONE-component form is decoded (the decoder checks the bound sampler and blocks
    /// otherwise), so the four gathered values are four texels of a single channel.
    TexGather { unit: u8, coords: u8, coord_half: bool },
    /// Test -> predicate (VTST, group 0x48): evaluate `alu(src1', src2)` per channel, compare
    /// the result against zero with `cmp`, reduce the four booleans with `reduce`, and write
    /// the single bit to predicate register `pdst`. `write_back` mirrors the encoding's
    /// `test_wben`: when set the raw ALU result is ALSO written to the destination register,
    /// so the instruction doubles as an ALU op. Sources are `[src1, src2]`.
    Test { alu: TestAlu, cmp: TestCmp, reduce: TestReduce, pdst: u8, write_back: bool },
    /// Test -> per-channel MASK (VTSTMSK, group 0x78): the same `alu(src1, src2)` compared
    /// against zero as [`Op::Test`], but instead of reducing the four booleans into one
    /// predicate bit it writes ONE VALUE PER CHANNEL into a general register.
    ///
    /// Only the NUMERIC mask form is decoded - each channel becomes `1.0` or `0.0` at the ALU's
    /// own precision - because that is the only `tst_mask_type` the corpus carries and the
    /// other two encode a bit pattern whose width rule is not established.
    TestMask { alu: TestAlu, cmp: TestCmp },
    /// Fragment discard (group 0xF8 KILL). Ends the fragment with no colour written; the
    /// emitter maps it to WGSL `discard`.
    Kill,
    /// Fragment DEPTH write (group 0xF8 DEPTHF, spec F8.7): `srcs[0]` is a scalar depth that
    /// replaces the interpolated one, and the whole shader becomes depth-replacing.
    ///
    /// The value is in the GUEST's depth space - the same encoding `gxp_guest_depth` produces
    /// and a fragment's `POSITION.z` reads - because that is the only space a shader can
    /// compute one in. Converting it to whatever depth the pipeline actually rasterises is the
    /// emitter's job, not this decode's.
    DepthF,
    /// Conditional or unconditional BRANCH (group 0xF8 BR). `rel` is the target expressed as a
    /// signed instruction-word delta from the branch's OWN index, so `target = index + rel`
    /// (spec F8.2 - the offset is a count of 64-bit words relative to the branch's own program
    /// offset). The instruction's [`Instr::pred`] is the branch CONDITION: the branch is taken
    /// when it holds, so the words it jumps over execute when it does NOT.
    ///
    /// This op never reaches the per-instruction emitter. [`crate::wgsl::emit_body`] consumes it
    /// structurally, turning a forward branch into a WGSL `if` around the range it skips, and
    /// hard-fails on any shape that is not a properly nested forward skip (a backward branch is a
    /// loop, a branch out of an enclosing range is irreducible, and a branch-with-link is a call
    /// - none are reconstructed yet). `rel` is rewritten by
    /// [`crate::usse::decode_shader`] when repeat-unrolling renumbers the instruction stream, so
    /// it is always a delta in the CURRENT stream.
    Branch { rel: i32 },
    /// MEMORY LOAD (group 0xE8, opcode1 0x1d) in the ONE variant the corpus establishes:
    /// `mode = 0, addr_mode = 0`, 32-bit elements, unconditional. Reads `elements`
    /// consecutive 32-bit words of GUEST MEMORY starting at byte address
    /// `srcs[0] + offset_bytes` (the operand's register holds a guest BYTE POINTER, and
    /// `offset_bytes` folds the instruction's immediate src1/src2 element offsets, already
    /// scaled by the element size), into `elements` consecutive destination registers from
    /// `dest`.
    ///
    /// `elements` runs 1..=16, so [`Instr::write_mask`] CANNOT describe the written span -
    /// it is left all-true and every consumer of this op must use `elements` instead. The
    /// decoder only produces this op for the established variant (PA-bank pointer,
    /// immediate offsets, load direction); every other combination stays blocked by name.
    ///
    /// The emitter translates it against the draw's bound MEMORY WINDOW - a storage of the
    /// addressed guest bytes the host uploads per draw (see `module::MemWindow`) - because
    /// WGSL has no raw pointers. A shader whose window cannot be established hard-fails at
    /// link time rather than reading fabricated bytes.
    MemLoad { elements: u8, offset_bytes: u32 },
    /// A documented operation that is not yet wired for WGSL emit (tex, pack, the u32
    /// bitwise ops, fx8/u8 integer ops, loads/stores, complex flow). Carries a static
    /// mnemonic so an emit attempt hard-fails naming exactly what to implement next. This
    /// is a FACT (the op is known) that is simply not translated yet - not a guess.
    Todo(&'static str),
    /// A group the ISA documents as containing only illegal instructions, or an operand
    /// in an exotic mode (index/constant/immediate) this decoder does not yet resolve.
    Illegal,
    /// An instruction word in an undocumented group (should not occur in valid shaders).
    Unsupported { group: u8 },
}

impl Op {
    /// Whether [`crate::wgsl`] has this operation fully wired for emit. Classification
    /// (knowing what the op is) is a fact for far more ops than this; emit grows as each
    /// is wired + tested. (Transcendentals / mov / tex / flow are classified but not yet
    /// emitted - their operand layouts are the next grind items.)
    pub fn is_emittable(self) -> bool {
        matches!(
            self,
            Op::Mad | Op::Mul | Op::Add | Op::Frc | Op::Dsx | Op::Dsy | Op::Min | Op::Max
                | Op::Dot { .. }
                | Op::Rcp | Op::Rsq | Op::Log | Op::Exp | Op::Mov | Op::Cmov { .. }
                | Op::Nop | Op::Tex { .. } | Op::TexGather { .. }
                | Op::Pack { .. } | Op::PackToInt { .. } | Op::PackFromInt { .. }
                | Op::PackIntCopy { .. }
                | Op::Limm { .. }
                | Op::CmovU8 { .. }
                | Op::PackUnorm8 { .. } | Op::CopyFx8
                | Op::Bitwise { .. }
                | Op::Sop2 { .. }
                | Op::IntMad { .. }
                | Op::IntMadStep { .. }
                | Op::LoadIndex { .. }
                | Op::Test { .. } | Op::TestMask { .. } | Op::Kill | Op::DepthF
                | Op::MemLoad { .. }
                // A branch is translated by the emitter's STRUCTURING pass rather than by
                // `emit_instr`, so it counts as wired here. Reaching `emit_instr` with one is a
                // bug in that pass and hard-fails there, naming itself.
                | Op::Branch { .. }
        )
    }

    /// Whether the operation is known (classified from the ISA), regardless of emit.
    pub fn is_classified(self) -> bool {
        !matches!(self, Op::Unsupported { .. } | Op::Illegal)
    }

    /// A stable short mnemonic for the operation, used in diagnostics and the grind error
    /// messages that name what to implement next.
    pub fn mnemonic(self) -> &'static str {
        match self {
            Op::Mad => "mad",
            Op::Mul => "mul",
            Op::Add => "add",
            Op::Frc => "frc",
            Op::Dsx => "dsx",
            Op::Dsy => "dsy",
            Op::Min => "min",
            Op::Max => "max",
            Op::Dot { .. } => "dot",
            Op::Rcp => "rcp",
            Op::Rsq => "rsq",
            Op::Log => "log",
            Op::Exp => "exp",
            Op::Mov => "mov",
            Op::Cmov { .. } => "cmov",
            Op::Nop => "nop",
            Op::Tex { .. } => "tex",
            Op::TexGather { .. } => "tex.gather4",
            Op::Pack { .. } => "pack",
            Op::PackToInt { .. } => "pack.int",
            Op::PackFromInt { .. } => "unpack.int",
            Op::PackIntCopy { .. } => "pack.int.copy",
            Op::Limm { .. } => "limm",
            Op::CmovU8 { .. } => "cmov.u8",
            Op::PackUnorm8 { to_unorm8, .. } => {
                if to_unorm8 {
                    "pack.unorm8"
                } else {
                    "unpack.unorm8"
                }
            }
            Op::CopyFx8 => "mov.fx8",
            Op::IntMad { .. } => "imad",
            Op::IntMadStep { high_half, .. } => {
                if high_half {
                    "imad.step1"
                } else {
                    "imad.step0"
                }
            }
            Op::LoadIndex { to_index, .. } => {
                if to_index {
                    "loadidx"
                } else {
                    "idxadd"
                }
            }
            Op::MemLoad { .. } => "ldmem",
            Op::Bitwise { .. } => "bitwise",
            Op::Sop2 { .. } => "sop2.fx8",
            Op::Test { .. } => "vtst",
            Op::TestMask { .. } => "vtstmsk",
            Op::Kill => "kill",
            Op::DepthF => "depthf",
            Op::Branch { .. } => "br",
            Op::Todo(name) => name,
            Op::Illegal => "illegal",
            Op::Unsupported { .. } => "unsupported",
        }
    }
}

/// A single decoded + interpreted instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instr {
    pub op: Op,
    pub pred: Predicate,
    /// Destination operand (where applicable). `None` for ops with an implicit or no
    /// register destination.
    pub dest: Option<Operand>,
    /// Destination component write mask (x,y,z,w).
    pub write_mask: [bool; 4],
    /// Source operands in operation order.
    pub srcs: Vec<Operand>,
    /// Whether this instruction operates in the F16/C10 pipeline (`data_format` bit)
    /// rather than F32.
    pub half_precision: bool,
    /// The original 64-bit instruction word, kept for diagnostics + the oracle harness.
    pub raw: u64,
    /// The top-level opcode group (`opcode1`), a fact used for histograms/coverage.
    pub group: u8,
    /// Set when the operation is known but this specific instruction carries an operand
    /// feature the decoder does not yet translate EXACTLY (an exotic operand mode, a
    /// swizzle table not yet transcribed, a partial destination mask). The op stays
    /// classified, but the emitter hard-fails naming this reason rather than risk an
    /// inexact translation. `None` = fully translatable.
    pub blocked: Option<&'static str>,
}

/// >>> WHERE CHANNEL `c` OF A PACKED-ELEMENT DESTINATION LANDS: the register offset from the
/// >>> destination's base register, and the bit shift within that register.
///
/// `bits` is the element width - 16 for a half pair, 8 for four bytes - and the rule is the
/// obvious one: `32 / bits` elements per register, channels in order, low element first.
///
/// # Why this is a function and not four lines written out four times
/// It WAS four lines written out four times, and the arithmetic is small enough that nobody
/// looked at it twice: the emitter's `store_raw_half` and `store_raw_byte`, the reference's
/// generic raw store, and the reference's `PackIntCopy` arm each carried its own copy. The
/// session that found `PackToInt` placing a 16-bit result one whole register per channel
/// cleared 69 corpus cases and stated in its own notes that the placement now had ONE
/// statement - which was true of the width and not of the arithmetic. Three copies were still
/// there, and an authored case for the partial write mask (the shape every corpus
/// `PackIntCopy` actually carries, and which no case covered) is what made that visible.
///
/// A rule with one statement cannot drift; a rule with four copies drifts silently, because
/// three of them agreeing is indistinguishable from all four being right.
pub fn packed_dest_slot(bits: u32, c: u32) -> (u32, u32) {
    debug_assert!(bits == 8 || bits == 16, "only the 8- and 16-bit packed widths land this way");
    let per_reg = 32 / bits;
    (c / per_reg, (c % per_reg) * bits)
}

impl Instr {
    /// True when the emitter can translate this instruction to WGSL today (operation is
    /// wired AND nothing about this instance is blocked).
    pub fn is_supported(&self) -> bool {
        self.op.is_emittable() && self.blocked.is_none()
    }

    /// The precision the instruction READS its source operands at, as a "is F16" flag. This is
    /// `half_precision` for every operation except a format convert ([`Op::Pack`]), which
    /// exists precisely to move a value between storage widths - so anything walking source
    /// operands (the emitter, the PA/SA read maps that decide the varying and uniform
    /// interfaces) must ask for this rather than assume one precision per instruction.
    pub fn source_half_precision(&self) -> bool {
        match self.op {
            Op::Pack { src_half } => src_half,
            // A float->integer convert reads its source at the SOURCE format's precision,
            // exactly like the float->float form - the instruction's own `half_precision`
            // describes the destination, which here is not a float at all.
            Op::PackToInt { src_half, .. } => src_half,
            // The INTEGER->float convert reads a 16-bit half pair, which spans registers the
            // same way an F16 operand does - so the read maps that size a varying or a uniform
            // from this flag get the right span by reporting the packed width, not the
            // destination float's.
            // The 8-bit width is neither of the two this flag can say, and it is not a half
            // pair: `source_packed_bytes` is what describes it, and reporting F16 here would
            // size the read two registers wide where it is one.
            Op::PackFromInt { bits, .. } => bits == 16,
            // Both ends are 16-bit halves, so the packed width is the right span at both ends -
            // the same reason the integer->float convert reports it. The byte form is packed
            // bytes at both ends instead, and answers below.
            Op::PackIntCopy { bits } => bits == 16,
            // The normalized U8 convert reads its source at the FLOAT precision only when the
            // float is the source; in the other direction the source is the packed byte
            // register, whose four channels live in one word and are read through
            // `Prec::Fx8`, so this flag does not describe it at all. Reporting the float
            // precision there would make the read maps size a varying or uniform as if it
            // held halves.
            Op::PackUnorm8 { to_unorm8, float_half } => to_unorm8 && float_half,
            _ => self.half_precision,
        }
    }

    /// True when this instruction reads its source as PACKED BYTES - four channels in ONE
    /// register, not one register per channel and not a half pair.
    ///
    /// [`Self::source_half_precision`] answers a two-way question (F32 or F16) and there is a
    /// third width. The emitter has always known it - `wgsl::Prec::src_of` returns `Prec::Fx8`
    /// for exactly these ops - but every place that computes which REGISTERS an operand spans
    /// asked only the two-way question, and so read a four-channel `fx8` operand at `pa[0]` as
    /// spanning `pa[0..4)`.
    ///
    /// That is four times too wide, and it is not cosmetic: the linker refuses a fragment that
    /// reads past its declared PA allocation, and a title's two-instruction passthrough
    /// (`Nop`, then `CopyFx8 o[0] <- pa[0]`) declares ONE primary register. It was refused for
    /// reading `pa[1]`, a register nothing in it names. Another title's version of the same
    /// shader survived only because it happened to allocate four.
    pub fn source_packed_bytes(&self) -> bool {
        matches!(
            self.op,
            Op::CopyFx8
                | Op::PackUnorm8 { to_unorm8: false, .. }
                // The 8-bit VPCK widths are the same shape: four selectable BYTES in one
                // register. Their `f32(byte)` is a numeric cast rather than `fx8`'s
                // `byte/255`, but that is the VALUE, and this asks about the SPAN.
                | Op::PackFromInt { bits: 8, .. }
                | Op::PackIntCopy { bits: 8 }
        )
    }

    /// The width in bits of the RAW element this instruction writes to each destination
    /// CHANNEL, or `None` when a channel is a whole 32-bit lane.
    ///
    /// This is the DESTINATION half of the question [`Self::source_packed_bytes`] asks about a
    /// source, and it exists because [`crate::wgsl::Prec`] cannot answer it: `Prec` describes a
    /// FLOAT view, and a float->integer convert's destination is not a float at all. Reading
    /// `half_precision` there returns `Prec::F32`, which places channel `c` in register
    /// `index + c` - a whole register each.
    ///
    /// That is wrong, and the emitter has always known it: `emit_pack_to_int` writes a 16-bit
    /// result through `store_raw_half` (`index + c/2`, half `c & 1`) and an 8-bit one through
    /// `store_raw_byte` (`index`, byte `c`), because a whole-register store put a skinned
    /// mesh's four blend indices in four registers where its four bone fetches look in two.
    /// The reference interpreter's generic store path had no such rule and so disagreed with
    /// the shipped shader on every program that reads a packed integer back - MEASURED as 22 of
    /// the 33 corpus programs carrying an equal-width 16-bit repack, which is the instruction
    /// that reads one back.
    ///
    /// Both of those callers read THIS function, so the placement has one statement. Note that
    /// `Prec::F16` is not the answer for a 16-bit integer: its store re-encodes the value as an
    /// f16 float, and what a pack leaves in the lane is a bit PATTERN.
    pub fn dest_raw_packed_bits(&self) -> Option<u32> {
        match self.op {
            Op::PackToInt { bits: 16, .. } => Some(16),
            Op::PackToInt { bits: 8, .. } => Some(8),
            // The equal-width integer repacks write the same shapes, and say so HERE so that
            // "which width does this instruction place at" has one answer too. They never
            // reach the reference's generic store - each has its own arm, which reads this -
            // so naming them changes no behaviour and removes a place to disagree.
            Op::PackIntCopy { bits: 16 } => Some(16),
            Op::PackIntCopy { bits: 8 } => Some(8),
            // A byte-select moves one RAW BYTE per channel, four to a register.
            Op::CmovU8 { .. } => Some(8),
            _ => None,
        }
    }

    /// True when the instruction's operation is known from the ISA (may not be emittable
    /// yet). Useful for coverage reporting - decode/classify is far ahead of emit.
    pub fn is_classified(&self) -> bool {
        self.op.is_classified()
    }

    /// The channels this instruction reads from its sources, mirroring the emitter's read
    /// model ([`crate::wgsl`]): a dot reads a fixed component prefix, a texture sample reads
    /// its coordinate prefix, a memory load reads ONE scalar address lane, a predicate-only
    /// test reads the channels its reduction consults, and every other op reads a source
    /// channel only where it writes the destination channel.
    ///
    /// This is the ONE answer. It used to be copied into the linker, the module extent scan
    /// and the corpus checks, and the copies drifted: the extent scan's lacked the test cases
    /// and a corpus check used the raw write mask, which is how 48 vertex programs read as
    /// reading past their declared attributes when none of them does.
    pub fn read_channels(&self) -> [bool; 4] {
        match self.op {
            Op::Dot { components } => {
                let n = (components as usize).clamp(1, 4);
                [0 < n, 1 < n, 2 < n, 3 < n]
            }
            Op::Tex { coords, .. } | Op::TexGather { coords, .. } => {
                let n = (coords as usize).clamp(1, 4);
                [0 < n, 1 < n, 2 < n, 3 < n]
            }
            // A memory load's only source is a scalar ADDRESS - one lane, whatever its
            // destination spans. Its write mask is explicitly not meaningful (the written span
            // is `elements` consecutive registers), so taking the mask as the read count claims
            // the three registers ABOVE the pointer are read too. That is how a pointer sitting
            // near the top of the SA bank made a program look like it read past its buffer.
            Op::MemLoad { .. } => [true, false, false, false],
            // A PREDICATE-ONLY test (`write_back = false`) has an all-false write mask, and
            // taking the mask as the read set therefore says it reads NOTHING. It reads two
            // operands and compares them; what it does not do is write a register. The channels
            // are the ones the REDUCTION consults: one for `Channel(n)`, all four for an AND/OR
            // over the vector.
            Op::Test { reduce: TestReduce::Channel(n), .. } => {
                let n = (n as usize).min(3);
                [n == 0, n == 1, n == 2, n == 3]
            }
            Op::Test { .. } | Op::TestMask { .. } => [true; 4],
            _ => self.write_mask,
        }
    }

    /// The register `src`'s channel `c` reads, and which 16-bit HALVES of it, at this
    /// instruction's own source precision - or `None` when the channel names no register
    /// (a swizzle constant, selector >= 4).
    ///
    /// An F32 channel reads register `index + selector`; the four F16 channels share a
    /// register PAIR (`index + selector/2`, one half each); a packed-byte operand keeps all
    /// four channels in ONE register. The SELECTOR is what addresses, never the channel
    /// ordinal - a source swizzled `[0,0,0,0]` reads one register four times, not four
    /// consecutive ones.
    pub fn source_register(&self, src: &Operand, c: usize) -> Option<(u32, std::ops::Range<usize>)> {
        let sel = *src.swizzle.get(c)? as usize;
        if sel > 3 {
            return None; // a swizzle constant reads no register
        }
        Some(if self.source_packed_bytes() {
            (u32::from(src.index), 0..2)
        } else if self.source_half_precision() {
            (u32::from(src.index) + (sel >> 1) as u32, (sel & 1)..(sel & 1) + 1)
        } else {
            (u32::from(src.index) + sel as u32, 0..2)
        })
    }
}

/// A fully decoded shader: its kind and instruction list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shader {
    pub kind: ProgramKind,
    pub instrs: Vec<Instr>,
}

impl Shader {
    /// The number of instructions the emitter can translate to WGSL today.
    pub fn supported_count(&self) -> usize {
        self.instrs.iter().filter(|i| i.is_supported()).count()
    }

    /// The number of instructions whose operation is known from the ISA (classified),
    /// whether or not emit is wired yet.
    pub fn classified_count(&self) -> usize {
        self.instrs.iter().filter(|i| i.is_classified()).count()
    }

    /// True when every instruction is emittable - the precondition for emitting WGSL.
    pub fn fully_supported(&self) -> bool {
        !self.instrs.is_empty() && self.instrs.iter().all(Instr::is_supported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instr(op: Op, write_mask: [bool; 4], srcs: Vec<Operand>, half: bool) -> Instr {
        Instr {
            op,
            pred: Predicate::Always,
            dest: Some(Operand::plain(Bank::Output, 0, 1)),
            write_mask,
            srcs,
            half_precision: half,
            raw: 0,
            group: 0,
            blocked: None,
        }
    }

    /// A source channel is addressed by its SWIZZLE SELECTOR, never by its channel ordinal.
    ///
    /// This is the rule a corpus check got wrong by adding the ordinal, which made a source
    /// swizzled `[0,0,0,0]` near the top of the bank look like it spanned four registers and
    /// reported 43 vertex programs as reading past attributes they never leave. The emitter has
    /// always addressed by selector (`wgsl::read_lane`), so the check was the half that was
    /// wrong - and a broadcast operand is the commonest shape there is.
    #[test]
    fn a_source_channel_is_addressed_by_its_selector_not_its_ordinal() {
        let mut broadcast = Operand::plain(Bank::PrimaryAttr, 10, 2);
        broadcast.swizzle = [0, 0, 0, 0];
        let i = instr(Op::Mad, [true, true, true, false], vec![broadcast], false);
        for c in 0..3 {
            assert_eq!(
                i.source_register(&i.srcs[0], c).map(|(r, _)| r),
                Some(10),
                "channel {c} of a broadcast reads pa[10], not pa[10 + {c}]"
            );
        }
    }

    /// The three widths span differently, and the span is what decides whether a program reads
    /// past its declared registers: an F32 channel is one register per selector, four F16
    /// channels share a register PAIR, and a packed-byte operand is ONE register for all four.
    #[test]
    fn each_source_width_spans_its_own_number_of_registers() {
        let plain = Operand::plain(Bank::PrimaryAttr, 4, 2);

        let f32_ = instr(Op::Mad, [true; 4], vec![plain], false);
        let regs: Vec<_> =
            (0..4).filter_map(|c| f32_.source_register(&f32_.srcs[0], c).map(|(r, _)| r)).collect();
        assert_eq!(regs, vec![4, 5, 6, 7], "an F32 operand is one register per selector");

        let f16 = instr(Op::Mad, [true; 4], vec![plain], true);
        let regs: Vec<_> =
            (0..4).filter_map(|c| f16.source_register(&f16.srcs[0], c).map(|(r, _)| r)).collect();
        assert_eq!(regs, vec![4, 4, 5, 5], "four F16 channels share a register pair");

        let fx8 = instr(Op::CopyFx8, [true; 4], vec![plain], false);
        let regs: Vec<_> =
            (0..4).filter_map(|c| fx8.source_register(&fx8.srcs[0], c).map(|(r, _)| r)).collect();
        assert_eq!(regs, vec![4, 4, 4, 4], "four bytes of ONE register");
    }

    /// A swizzle CONSTANT (selector 4..7 - the 0.0/1.0/2.0/0.5 literals) names no register, so
    /// it must not contribute a read. Counting it would charge the operand's base register to
    /// a channel that never touches the register file.
    #[test]
    fn a_swizzle_constant_reads_no_register() {
        let mut lit = Operand::plain(Bank::PrimaryAttr, 4, 2);
        lit.swizzle = [5, 4, 2, 7];
        let i = instr(Op::Mad, [true; 4], vec![lit], false);
        let regs: Vec<_> =
            (0..4).map(|c| i.source_register(&i.srcs[0], c).map(|(r, _)| r)).collect();
        assert_eq!(regs, vec![None, None, Some(6), None]);
    }

    /// A predicate-only test writes no channel, and its read set is the channels its REDUCTION
    /// consults - not its (empty) write mask. Taking the mask says a comparison reads nothing,
    /// which is how the registers a test reads became invisible to the code that has to route
    /// them.
    #[test]
    fn a_predicate_only_test_reads_the_channels_its_reduction_consults() {
        let src = Operand::plain(Bank::PrimaryAttr, 0, 2);
        let chan = instr(
            Op::Test {
                alu: TestAlu::Sub,
                cmp: TestCmp::Ne,
                reduce: TestReduce::Channel(2),
                pdst: 0,
                write_back: false,
            },
            [false; 4],
            vec![src],
            false,
        );
        assert_eq!(chan.read_channels(), [false, false, true, false]);

        let vector = instr(
            Op::Test {
                alu: TestAlu::Sub,
                cmp: TestCmp::Ne,
                reduce: TestReduce::AndAll,
                pdst: 0,
                write_back: false,
            },
            [false; 4],
            vec![src],
            false,
        );
        assert_eq!(vector.read_channels(), [true; 4]);
    }

    /// A memory load's one source is a scalar ADDRESS, whatever its destination spans. Reading
    /// the write mask instead claims the three registers ABOVE the pointer are read too.
    #[test]
    fn a_memory_load_reads_one_address_lane() {
        let i = instr(
            Op::MemLoad { elements: 3, offset_bytes: 16 },
            [true; 4],
            vec![Operand::plain(Bank::PrimaryAttr, 4, 1)],
            false,
        );
        assert_eq!(i.read_channels(), [true, false, false, false]);
    }
}
