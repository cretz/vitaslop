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
/// `lod_mode` field (spec E0.4). `Bias` and `Level` read a scalar from src2; `Implicit` reads
/// nothing and lets the hardware derive the level from the coordinate derivatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexLod {
    /// lod_mode 0 - hardware-derived level (`textureSample`).
    Implicit,
    /// lod_mode 1 - src2 is added to the derived level (`textureSampleBias`).
    Bias,
    /// lod_mode 2 - src2 IS the level (`textureSampleLevel`).
    Level,
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
    /// Load an INDEX register (group 0x14, I16MAD, in the one encoding the corpus establishes):
    /// `i[dest] = src + addend`, as a 16-bit integer.
    ///
    /// The corpus is the whole authority for this and it is narrow: group 0x14 occurs in ONE
    /// program across three titles, six times, and the only bits that ever vary are [17:14],
    /// the source register number. Every other bit is constant, so the encoding establishes a
    /// register and nothing else. `addend` is fixed instead by ARITHMETIC CLOSURE against the
    /// container's own parameter table - see the decoder - and any group-0x14 word that is not
    /// this exact encoding must hard-fail rather than inherit that assumption.
    LoadIndex { addend: i32 },
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
    /// Test -> predicate (VTST, group 0x48): evaluate `alu(src1', src2)` per channel, compare
    /// the result against zero with `cmp`, reduce the four booleans with `reduce`, and write
    /// the single bit to predicate register `pdst`. `write_back` mirrors the encoding's
    /// `test_wben`: when set the raw ALU result is ALSO written to the destination register,
    /// so the instruction doubles as an ALU op. Sources are `[src1, src2]`.
    Test { alu: TestAlu, cmp: TestCmp, reduce: TestReduce, pdst: u8, write_back: bool },
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
                | Op::Nop | Op::Tex { .. }
                | Op::Pack { .. } | Op::PackToInt { .. } | Op::Bitwise { .. }
                | Op::LoadIndex { .. }
                | Op::Test { .. } | Op::Kill | Op::DepthF
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
            Op::Pack { .. } => "pack",
            Op::PackToInt { .. } => "pack.int",
            Op::LoadIndex { .. } => "loadidx",
            Op::Bitwise { .. } => "bitwise",
            Op::Test { .. } => "vtst",
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
            _ => self.half_precision,
        }
    }

    /// True when the instruction's operation is known from the ISA (may not be emittable
    /// yet). Useful for coverage reporting - decode/classify is far ahead of emit.
    pub fn is_classified(&self) -> bool {
        self.op.is_classified()
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
