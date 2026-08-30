//! IR -> WGSL emitter.
//!
//! USSE registers are 32-bit scalars; a vector operand at register base B reads the
//! consecutive registers B, B+1, B+2, B+3 selected per output channel by the swizzle
//! (register index itself is `field * 2`, so B is even - see the decoder). To translate
//! faithfully the emitter therefore SCALARISES: it emits one WGSL statement per written
//! destination channel, reading `bank[base + lane]` for each source channel. This avoids
//! assuming a `vec4` aliasing that would silently mis-map the register file.
//!
//! The register banks map to WGSL arrays the pipeline builder binds:
//!   r[] temporaries (local), pa[] primary attributes (interpolated varyings), sa[]
//!   secondary attributes (default uniform buffer), o[] outputs (fragment result).
//!
//! The emitter is strict: it HARD-FAILS with [`EmitError`] the moment it meets an op it
//! has not wired or an instruction the decoder flagged `blocked`, naming exactly what to
//! implement next (opcode grind). It never emits an approximation or silently degrades.

use core::fmt::Write as _;

use crate::container::ProgramKind;
use crate::ir::{
    Bank, BitwiseKind, CompareMethod, Instr, Op, Operand, Predicate, Shader, SopFactor, SopOp,
    TestAlu, TestCmp, TestReduce, TexLod,
};

/// Why WGSL emission hard-failed. Each variant pinpoints what to implement next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    /// An instruction whose operation is not yet wired for emit. Names the instruction
    /// index, byte offset, opcode group, raw word, and the operation mnemonic.
    UnsupportedOp { index: usize, byte_offset: usize, op: &'static str, group: u8, raw: u64 },
    /// The decoder classified the operation but flagged this specific instruction as
    /// carrying a feature not yet translated exactly (exotic operand mode, predicate, a
    /// group whose operands are not decoded). Names the reason.
    Blocked { index: usize, byte_offset: usize, reason: &'static str, raw: u64 },
    /// The shader decoded to zero instructions.
    Empty,
    /// A wired op referenced an operand the emitter cannot express (an unmapped bank, or a
    /// missing source). Names the instruction.
    UnmappedOperand { index: usize, raw: u64 },
    /// A source read an internal register (i0..i3) lane that no earlier instruction in the
    /// USSE stream wrote. Internal registers are not bound by the pipeline builder, so
    /// their pre-shader (iterator/PDS) contents are unmodeled - emitting a read of one
    /// would translate garbage. Hard-fail rather than guess. Names the instruction + lane.
    UndefinedInternal { index: usize, byte_offset: usize, lane: u8, raw: u64 },
    /// A source read a SPECIAL/GLOBAL hardware register whose contents have not been
    /// established. Names the GLOBAL index so the next one to appear says what to go and
    /// establish, rather than reading as a generic unmapped-bank failure.
    UnmodeledGlobal { index: usize, byte_offset: usize, global: u8, raw: u64 },
}

impl core::fmt::Display for EmitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EmitError::UnsupportedOp { index, byte_offset, op, group, raw } => write!(
                f,
                "unsupported USSE op '{op}': instruction #{index} at code byte {byte_offset:#x}, \
                 opcode1 group {group:#04x}, raw {raw:#018x} - wire this op's emit \
                 from the SGX543 USSE ISA facts and re-run",
            ),
            EmitError::Blocked { index, byte_offset, reason, raw } => write!(
                f,
                "blocked USSE instruction #{index} at code byte {byte_offset:#x} (raw {raw:#018x}): \
                 {reason} - wire this case and re-run",
            ),
            EmitError::Empty => write!(f, "empty USSE code stream"),
            EmitError::UnmappedOperand { index, raw } => write!(
                f,
                "instruction #{index} (raw {raw:#018x}) references an unmapped register bank/operand",
            ),
            EmitError::UndefinedInternal { index, byte_offset, lane, raw } => write!(
                f,
                "USSE instruction #{index} at code byte {byte_offset:#x} (raw {raw:#018x}) reads \
                 internal register lane i[{lane}] that no earlier instruction wrote - the \
                 iterator/PDS pre-load of internal registers is not modeled; wire that before \
                 emitting this shader",
            ),
            EmitError::UnmodeledGlobal { index, byte_offset, global, raw } => write!(
                f,
                "USSE instruction #{index} at code byte {byte_offset:#x} (raw {raw:#018x}) reads \
                 SPECIAL/GLOBAL hardware register {global} - its contents are not established; \
                 establish what GLOBAL[{global}] holds and wire it, do not substitute a value",
            ),
        }
    }
}

/// Number of internal-register scalar lanes the pipeline exposes: i0..i3, four lanes each.
const INTERNAL_LANES: usize = 16;

/// The WGSL name the fragment module binds `@builtin(front_facing)` to. The emitter
/// references it whenever it translates the established `GLOBAL[16]` facing test, so a module
/// builder that emits such a body MUST declare this (see [`FRONT_FACING_DECL`]).
pub const FRONT_FACING_VAR: &str = "gxp_front_facing";

/// The declaration a fragment module emits to bind [`FRONT_FACING_VAR`] from its entry point.
pub const FRONT_FACING_DECL: &str = "  let gxp_front_facing: bool = in.front_facing;\n";

/// The GLOBAL (SPECIAL hardware register) index whose meaning is established: the per-fragment
/// facing flag, read as `GLOBAL[16] & 1`.
const GLOBAL_FACING: u8 = 16;

/// The WGSL `u32` expression for a read of an established GLOBAL hardware register, or `None`
/// when this register's contents are not established (the caller then hard-fails naming it).
///
/// **`GLOBAL[16]` bit 0 is the per-fragment FACING flag.** No clean source names the GLOBAL
/// registers, so this is an inference from the corpus; here is the whole argument, and the
/// scope is deliberately narrow enough that it cannot quietly apply to anything it was not
/// derived from:
///
/// * The decode is a fact, confirmed field by field against the TEST-group layout: the
///   instruction is `p0 = ((GLOBAL[16] & 1) != 0)`.
/// * There are exactly THREE GLOBAL reads in the whole captured corpus. All three are
///   `GLOBAL[16]`, all three are this same test, and all three are in FRAGMENT programs
///   (`global_special_register_reads` in the oracle prints them).
/// * In all three, `p0` selects between two SA registers that the program's own SECONDARY
///   program sets, by byte-identical instruction pairs, to exactly `+1.0` and `-1.0`
///   (`mov SA[a] <- FPCONSTANT 1.0`, then `mul SA[b] <- -SA[a] * 1.0`). `p0` set picks `-1.0`.
/// * That selected value is packed to F16 and run through two complementary conditional moves
///   (`LteZero` / `LtZero`) and a subtract, which is exactly `sign(x)` - so the shader has
///   computed `+1` or `-1` from the predicate and nothing else.
/// * It multiplies an interpolated 3-vector which is then normalized and used as a cube-map
///   sampling direction and as the operand of dot products with the directional light
///   direction. That vector is the shading NORMAL.
///
/// Flipping the sign of a shading normal per fragment, keyed on one bit of a register no
/// program writes, is two-sided lighting; the only per-fragment hardware boolean that idiom
/// keys on is facing.
///
/// **The POLARITY is measured, not reasoned**, and it is the one part of this that is tied to
/// the pipeline rather than to the shader. `select(0u, 1u, front_facing)` is what renders the
/// car-body liveries correctly; the opposite sense paints those bodies pure black (the flipped
/// normal drives `saturate(N.L)` to zero on every visible surface), which is how it was
/// decided. Note WGSL's `front_facing` is defined against the pipeline's `front_face` winding,
/// which here is wgpu's default (CCW is front) with no culling, because the guest's own
/// winding/cull state is not yet wired into this path. **If that is ever wired, re-measure this
/// polarity at the same time** - the two are one setting, not two.
///
/// So: bit 0 set selects the flipped normal, and under the current winding configuration that
/// is `front_facing`. Whether the hardware's own name for that bit is "front" or "back" is not
/// settled here, and nothing depends on which word is used.
fn global_u32_expr(op: &Operand, kind: ProgramKind) -> Option<String> {
    // Fragment-only: `front_facing` exists per fragment and nowhere else. A vertex program
    // reading GLOBAL[16] would be a different register file and must hard-fail, not inherit
    // this reading.
    if op.index != GLOBAL_FACING || kind != ProgramKind::Fragment {
        return None;
    }
    Some(format!("select(0u, 1u, {FRONT_FACING_VAR})"))
}

/// The WGSL array prefix for a register bank, or `None` for a bank the emitter cannot
/// express as an indexed array (which is a hard failure at the call site). `Constant` is
/// not an array bank - it is materialised inline by [`src_channel`], so it returns `None`
/// here and callers must handle it before reaching this.
fn bank_prefix(bank: Bank) -> Option<&'static str> {
    Some(match bank {
        Bank::Temp => "r",
        Bank::Output => "o",
        Bank::PrimaryAttr => "pa",
        Bank::SecondaryAttr => "sa",
        Bank::Internal => "i",
        // Constant / Immediate are materialised inline; Global is pipeline state with no
        // register-file storage; Indexed and Index are ADDRESSING, not a bank - an Indexed
        // operand resolves through `indexed_sub_bank` to a real bank at use, and the index
        // register file has its own name. None of them has a plain `bank[n]` spelling.
        Bank::Constant | Bank::Immediate | Bank::Global | Bank::Indexed | Bank::Index
        | Bank::Raw(_) => return None,
    })
}

/// The CNST6 f32-mode (bank 0) constant table, as exact 32-bit IEEE-754 bit patterns
/// (henkaku SGX543 "Constants"). A constant operand's 6-bit selector indexes this. The
/// emitter materialises `bitcast<f32>(bitsu)` so the value is EXACT - including the packed
/// f16-pair and NaN entries - never a decimal approximation.
const CNST6_F32_BANK0: [u32; 64] = [
    0x0000_0000, 0x0000_0000, 0x3F80_0000, 0x3F80_0000, 0x4000_0000, 0x4100_0000, 0x4200_0000, 0x4300_0000,
    0x4400_0000, 0x4500_0000, 0x4600_0000, 0x4700_0000, 0x3F00_0000, 0x3E00_0000, 0x3D00_0000, 0x3C00_0000,
    0x3B00_0000, 0x3A00_0000, 0x3900_0000, 0x3800_0000, 0x402D_F854, 0x3FB5_04F3, 0x4049_0FDB, 0x3F49_0FDB,
    0x40C9_0FDB, 0x41C9_0FDB, 0x3780_0000, 0x3780_0080, 0x35D0_0D01, 0x3988_8889, 0x3CAA_AAAB, 0x3F00_0000,
    0x0000_0000, 0x0000_0000, 0x3C00_3C00, 0x4400_4000, 0x5400_5000, 0x6400_6000, 0x7400_7000, 0x3400_3800,
    0x2400_2800, 0x1400_1800, 0x0400_0800, 0x35E2_416F, 0x39A8_3DA8, 0x3E48_4248, 0x4A48_4648, 0x0000_0000,
    0x0000_0000, 0x3000_2555, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
    0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0x7FFF_7FFF, 0x7FFF_7FFF, 0x7FFF_7FFF, 0x7FFF_7FFF,
];

/// The CNST6 f32-mode bank 1, read when a channel's swizzle selects Y (spec A.7/A.9).
const CNST6_F32_BANK1: [u32; 64] = [
    0x0000_0000, 0x3F80_0000, 0x0000_0000, 0x3F80_0000, 0x4080_0000, 0x4180_0000, 0x4280_0000, 0x4380_0000,
    0x4480_0000, 0x4580_0000, 0x4680_0000, 0x4780_0000, 0x3E80_0000, 0x3D80_0000, 0x3C80_0000, 0x3B80_0000,
    0x3A80_0000, 0x3980_0000, 0x3880_0000, 0x3780_0000, 0x3EBC_5AB2, 0x3F35_04F3, 0x3FC9_0FDB, 0x3EC9_0FDB,
    0x4149_0FDB, 0x0000_0000, 0x3800_0000, 0x3800_0100, 0x37B6_0B61, 0x3B2A_AAAB, 0x3E00_0000, 0x3F80_0000,
    0x3C00_0000, 0x0000_0000, 0x3C00_3C00, 0x4C00_4800, 0x5C00_5800, 0x6C00_6800, 0x0000_7800, 0x2C00_3000,
    0x1C00_2000, 0x0C00_1000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x3648_3A48, 0x0000_4E48, 0x0000_0000,
    0x1955_0C44, 0x3C00_3800, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
    0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000,
];

/// The four CNST6 F16-mode constant banks (spec A.9), as exact IEEE-754 half bit patterns in
/// the low 16 bits of each entry. For an F16 operand the channel's swizzle selector chooses
/// the bank (X=0, Y=1, Z=2, W=3) and the 6-bit CNST6 selector indexes it. Stored as bit
/// patterns so the NaN entries stay exact.
const CNST6_F16: [[u32; 64]; 4] = [
    // bank 0
    [
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        0x0000, 0x0000, 0x0000, 0x0000, 0x3800, 0x0000, 0x0000, 0x0000,
        0x3C00, 0x0000, 0x0000, 0x0000, 0xF854, 0x04F3, 0x0FDB, 0x0FDB,
        0x0FDB, 0x0FDB, 0x0000, 0x0080, 0x0D01, 0x8889, 0xAAAB, 0x0000,
        0x0000, 0x0000, 0x3C00, 0x4000, 0x5000, 0x6000, 0x7000, 0x3800,
        0x2800, 0x1800, 0x0800, 0x416F, 0x3DA8, 0x4248, 0x4648, 0x0000,
        0x0000, 0x2555, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF,
    ],
    // bank 1
    [
        0x0000, 0x0000, 0x3F80, 0x3F80, 0x4000, 0x4100, 0x4200, 0x4300,
        0x4400, 0x4500, 0x4600, 0x4700, 0x3F00, 0x3E00, 0x3D00, 0x3C00,
        0x3B00, 0x3A00, 0x3900, 0x3800, 0x402D, 0x3FB5, 0x4049, 0x3F49,
        0x40C9, 0x41C9, 0x3780, 0x3780, 0x35D0, 0x3988, 0x3CAA, 0x3F00,
        0x0000, 0x0000, 0x3C00, 0x4400, 0x5400, 0x6400, 0x7400, 0x3400,
        0x2400, 0x1400, 0x0400, 0x35E2, 0x39A8, 0x3E48, 0x4A48, 0x0000,
        0x0000, 0x3000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0x7FFF, 0x7FFF, 0x7FFF, 0x7FFF,
    ],
    // bank 2
    [
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        0x0000, 0x0000, 0x0000, 0x0000, 0x5AB2, 0x04F3, 0x0FDB, 0x0FDB,
        0x0FDB, 0x0000, 0x0000, 0x0100, 0x0B61, 0xAAAB, 0x0000, 0x0000,
        0x0000, 0x0000, 0x3C00, 0x4800, 0x5800, 0x6800, 0x7800, 0x3000,
        0x2000, 0x1000, 0x0000, 0x0000, 0x0000, 0x3A48, 0x4E48, 0x0000,
        0x0C44, 0x3800, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    ],
    // bank 3
    [
        0x0000, 0x3F80, 0x0000, 0x3F80, 0x4080, 0x4180, 0x4280, 0x4380,
        0x4480, 0x4580, 0x4680, 0x4780, 0x3E80, 0x3D80, 0x3C80, 0x3B80,
        0x3A80, 0x3980, 0x3880, 0x3780, 0x3EBC, 0x3F35, 0x3FC9, 0x3EC9,
        0x4149, 0x0000, 0x3800, 0x3800, 0x37B6, 0x3B2A, 0x3E00, 0x3F80,
        0x3C00, 0x0000, 0x3C00, 0x4C00, 0x5C00, 0x6C00, 0x0000, 0x2C00,
        0x1C00, 0x0C00, 0x0000, 0x0000, 0x0000, 0x3648, 0x0000, 0x0000,
        0x1955, 0x3C00, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
    ],
];

/// The exact `f32` value of CNST6 selector `sel` (bank 0, f32 mode) - the reference
/// interpreter's counterpart to the emitter's `bitcast<f32>` materialisation.
pub fn cnst6_value(sel: u8) -> f32 {
    f32::from_bits(CNST6_F32_BANK0[(sel & 0x3f) as usize])
}

/// The exact value of an F16-mode CNST6 constant: bank `sel` (the channel's swizzle
/// selector, 0..3), entry `index`. The reference interpreter's counterpart to the F16 arm of
/// [`src_channel`].
pub fn cnst6_f16_value(sel: u8, index: u8) -> f32 {
    f16_bits_to_f32(CNST6_F16[(sel & 3) as usize][(index & 0x3f) as usize] as u16)
}

/// Decode an IEEE-754 binary16 bit pattern to `f32` (exact - every half is representable).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = match exp {
        // Zero / subnormal: scale the fraction by 2^-24 in f32 terms.
        0 if frac == 0 => sign,
        0 => return f32::from_bits(sign | 0x3380_0000).mul_add(frac as f32, 0.0) * if sign != 0 { 1.0 } else { 1.0 },
        // Inf / NaN keep their payload in the top fraction bits.
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 112) << 23) | (frac << 13),
    };
    f32::from_bits(out)
}

/// The element width an instruction addresses its operands with. The USSE unified store is
/// an array of 32-bit registers either way; the precision decides how a channel maps onto it
/// (a fact, see the distilled SA-bank layout notes, section 4):
///
/// * `F32` - channel `c` is the whole 32-bit register `base + c`.
/// * `F16` - channel `c` is half `c & 1` of register `base + (c >> 1)`, so four channels
///   occupy the register PAIR `base, base+1`.
///
/// Fragment programs are 70-90% F16 on real titles, so treating everything as F32 (the
/// obvious-looking model) reads every uniform and varying from the wrong place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prec {
    F32,
    F16,
    /// Four 8-bit UNSIGNED-NORMALISED channels in ONE register: channel `c` is byte `c`,
    /// valued `byte / 255`. This is the register view the 8-bit families (the SOP2M combiner
    /// and the INT8 test ALU) see, and it is not a narrower float - reading such a register
    /// through [`Prec::F32`] reinterprets the byte pattern as an f32, which turns the corpus's
    /// alpha-test flag of 0x00000001 into a denormal indistinguishable from zero.
    Fx8,
}

impl Prec {
    /// The precision an instruction WRITES its destination at (`half_precision`).
    fn of(instr: &Instr) -> Prec {
        Prec::from_half(instr.half_precision)
    }

    /// The precision an instruction READS its sources at. For every operation this is the
    /// same as the destination's - except a format convert ([`Op::Pack`], VPCK), whose whole
    /// purpose is that the two differ: an F16->F32 unpack read at F32 would take a register
    /// holding two halves and interpret it as one 32-bit float, i.e. a denormal instead of the
    /// value. (A texture sample also carries independent coordinate/result precisions, but it
    /// gets them from its own decoded fields - see [`emit_tex`].)
    fn src_of(instr: &Instr) -> Prec {
        Prec::from_half(instr.source_half_precision())
    }

    fn from_half(half: bool) -> Prec {
        if half {
            Prec::F16
        } else {
            Prec::F32
        }
    }
}

/// The precision an operand in `bank` is really addressed at.
///
/// Every ordinary bank stores what the instruction's precision says: an F16 instruction reads
/// and writes a register as two packed halves, an F32 one as a single float. The INTERNAL
/// registers do not - they are the pipeline's UNPACKED accumulators, four 32-bit lanes each,
/// whatever precision the instruction that touches them runs at.
///
/// MEASURED, on the shadow filter of a golf title's three world fragment programs, where the
/// def-use closes only under this reading. One `mov.f32` broadcasts the reference depth into
/// all four lanes of `i0`; a VTSTMSK (an F32 test) compares the four gathered depths against
/// those lanes and writes its four-channel mask back into `i0`; and the very next instruction
/// is an **F16** `dot4` that reads `i0` with the swizzle `[3,2,1,0]` against the sample's four
/// bilinear coefficients. Under the packed reading that dot reads `i0`'s four selectors as
/// `i[1].hi, i[1].lo, i[0].hi, i[0].lo` - two registers holding F32 bit patterns, read as four
/// halves - and the instruction before and the instruction after both address the same register
/// as four floats. Four distinct mask values cannot live in two packed registers that an F32
/// test wrote.
///
/// This is also what the emitter's own undefined-internal-lane guard has always assumed: it
/// marks and checks lane `index + selector`, the four-lane layout, with no precision term in
/// it. The two were simply inconsistent, and the guard is the half that was right.
fn bank_prec(bank: Bank, prec: Prec) -> Prec {
    if matches!(bank, Bank::Internal) {
        Prec::F32
    } else {
        prec
    }
}

/// The WGSL rvalue reading channel-selector `sel` (0..3) of the register file at `base`.
fn read_lane(prefix: &str, base: u32, sel: u32, prec: Prec) -> String {
    match prec {
        Prec::F32 => format!("bitcast<f32>({prefix}[{}])", base + sel),
        Prec::F16 => format!("unpack2x16float({prefix}[{}])[{}]", base + (sel >> 1), sel & 1),
        // All four channels live in ONE register, so the selector picks a BYTE and never a
        // neighbouring register the way the two float widths do.
        Prec::Fx8 => format!("unpack4x8unorm({prefix}[{base}])[{sel}]"),
    }
}

/// The WGSL expression for source operand channel `c`: a register-file read for a lane
/// selector, or a float literal for a swizzle constant, with abs/neg modifiers applied.
fn src_channel(op: &Operand, c: usize, prec: Prec) -> Option<String> {
    // A constant operand materialises the exact hardware constant-table value for this
    // channel. Which table is a fact of the operand's data type and the channel's swizzle
    // selector (spec A.7): F32 reads bank 1 when the channel selects Y and bank 0 otherwise;
    // F16 reads one of four banks chosen by the selector directly.
    if matches!(op.bank, Bank::Constant) {
        let sel = op.swizzle[c];
        let mut e = match (prec, sel) {
            (_, 4) => "0.0".to_string(),
            (_, 5) => "1.0".to_string(),
            (_, 6) => "2.0".to_string(),
            (_, 7) => "0.5".to_string(),
            (Prec::F32, 1) => format!("bitcast<f32>({:#010x}u)", CNST6_F32_BANK1[(op.index & 0x3f) as usize]),
            (Prec::F32, _) => format!("bitcast<f32>({:#010x}u)", CNST6_F32_BANK0[(op.index & 0x3f) as usize]),
            (Prec::F16, _) => {
                let bits = CNST6_F16[(sel & 3) as usize][(op.index & 0x3f) as usize];
                format!("unpack2x16float({bits:#010x}u)[0]")
            }
            // The hardware constant table has an F32 and an F16 view and no established 8-bit
            // one. Refusing is exact: no corpus program reads the constant bank from an 8-bit
            // instruction, and picking either float table would silently substitute a value.
            (Prec::Fx8, _) => return None,
        };
        if op.abs {
            e = format!("abs({e})");
        }
        if op.neg {
            e = format!("(-{e})");
        }
        return Some(e);
    }
    // An inline IMMEDIATE is a scalar literal, not a register file: spec A.7 says the operand's
    // number IS the value, "typed per the operand's DataType", so it is the NUMBER `num` and not
    // a bit pattern to reinterpret. Every channel whose selector names a lane reads that same
    // scalar - there are no other lanes to read - and selectors 4..7 are the ordinary swizzle
    // constants, which is how one operand supplies a mixed vector like `(1, 1, num, num)`.
    if matches!(op.bank, Bank::Immediate) {
        let mut e = match op.swizzle[c] {
            4 => "0.0".to_string(),
            5 => "1.0".to_string(),
            6 => "2.0".to_string(),
            7 => "0.5".to_string(),
            _ => format!("{:?}", op.index as f32),
        };
        if op.abs {
            e = format!("abs({e})");
        }
        if op.neg {
            e = format!("(-{e})");
        }
        return Some(e);
    }
    let prefix = bank_prefix(op.bank)?;
    let sel = op.swizzle[c];
    let prec = bank_prec(op.bank, prec);
    let mut e = match sel {
        0..=3 => read_lane(prefix, op.index as u32, sel as u32, prec),
        4 => "0.0".to_string(),
        5 => "1.0".to_string(),
        6 => "2.0".to_string(),
        7 => "0.5".to_string(),
        _ => return None,
    };
    if op.abs {
        e = format!("abs({e})");
    }
    if op.neg {
        e = format!("(-{e})");
    }
    Some(e)
}

/// The statement storing `expr` (an f32 rvalue) into destination channel `c`. An F32 channel
/// overwrites a whole register; an F16 channel is a read-modify-write of one half, so the
/// paired channel keeps its value - exactly how the hardware packs two halves per register.
fn store_stmt(op: &Operand, c: usize, expr: &str, prec: Prec) -> Option<String> {
    let prefix = bank_prefix(op.bank)?;
    let prec = bank_prec(op.bank, prec);
    Some(match prec {
        Prec::F32 => {
            format!("  {prefix}[{}] = bitcast<u32>({expr});\n", op.index as u32 + c as u32)
        }
        Prec::F16 => {
            let reg = op.index as u32 + (c as u32 >> 1);
            if c & 1 == 0 {
                format!(
                    "  {prefix}[{reg}] = ({prefix}[{reg}] & 0xffff0000u) | (pack2x16float(vec2<f32>({expr}, 0.0)) & 0x0000ffffu);\n"
                )
            } else {
                format!(
                    "  {prefix}[{reg}] = ({prefix}[{reg}] & 0x0000ffffu) | (pack2x16float(vec2<f32>(0.0, {expr})) & 0xffff0000u);\n"
                )
            }
        }
        // One BYTE of one register, read-modify-write so the other three channels keep their
        // bytes. Rounded, not truncated: the value is a `byte/255` unorm coming back the way
        // it went out, and truncating loses the last representable step on every round trip.
        Prec::Fx8 => {
            let reg = op.index as u32;
            let shift = 8 * c as u32;
            let keep = !(0xffu32 << shift);
            format!(
                "  {prefix}[{reg}] = ({prefix}[{reg}] & {keep:#010x}u) | \
                 (u32(clamp({expr}, 0.0, 1.0) * 255.0 + 0.5) << {shift}u);\n"
            )
        }
    })
}

/// The statement sink one instruction emits into.
///
/// A USSE instruction reads ALL of its sources before it writes ANY of its destination
/// channels. The emitter scalarises a vector instruction into one statement per channel, and
/// those statements run in order - so when the destination register range overlaps a source
/// register range, a channel written early is visible to a channel emitted later, and the
/// instruction computes something the hardware never would.
///
/// MEASURED, on a title's display composite: `mul pa[2].xyz <- pa[2].zzz, pa[2].xxx` is the
/// last step of a Reinhard tonemap, `L * (1/(1+L))`. Emitted straight, channel x overwrote
/// `pa[2].x` (the `1/(1+L)` term) before channels y and z read it, so green and blue came out
/// multiplied by an extra factor of `L` while red was correct. On screen that is a frame that
/// is too dark and too RED, with nothing anywhere to say a shader was miscompiled - and red
/// being exactly right is what makes it read as a colour-space problem rather than a bug.
///
/// So when `stage` is set, every store is held back: the value goes into a `let` first, and
/// the stores are flushed only once the whole instruction has been read. `stage` is off for
/// the common non-aliasing instruction, where deferring would only make the emitted WGSL
/// harder to read for no change in meaning.
struct Dest<'a> {
    body: &'a mut String,
    /// Held-back store statements, in channel order. Always empty when `stage` is false.
    deferred: Vec<String>,
    stage: bool,
}

impl std::fmt::Write for Dest<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.body.write_str(s)
    }
}

impl Dest<'_> {
    /// Emit (or stage) the store of `expr` into destination channel `c`.
    fn store(&mut self, op: &Operand, c: usize, expr: &str, prec: Prec) -> Option<()> {
        self.store_in_slot(op, c, c, expr, prec)
    }

    /// [`Self::store`] with the staging temporary named by `slot` rather than by the channel.
    ///
    /// Every ordinary instruction writes ONE destination operand, so naming the temporary after
    /// the channel gives four distinct names and the enclosing block keeps them off every other
    /// instruction's. A GATHER breaks that: it writes the four texels AND, four registers
    /// higher, the four bilinear coefficients - eight stores from one instruction. Both groups
    /// asked for `g0..g3` in one block, which is a WGSL `redefinition of g0` and therefore a
    /// pipeline the device REFUSES, dropping every draw that uses it. It only bit when the
    /// gather's destination happened to alias its coordinate operand (staging is off otherwise),
    /// which is why one title's shadow filter compiled and another title's did not.
    fn store_in_slot(
        &mut self,
        op: &Operand,
        c: usize,
        slot: usize,
        expr: &str,
        prec: Prec,
    ) -> Option<()> {
        if !self.stage {
            self.body.push_str(&store_stmt(op, c, expr, prec)?);
            return Some(());
        }
        let tmp = format!("g{slot}");
        let stmt = store_stmt(op, c, &tmp, prec)?;
        let _ = writeln!(self.body, "  let {tmp} = {expr};");
        self.deferred.push(stmt);
        Some(())
    }

    /// Store a lane whose expression is already the RAW 32-bit pattern, not a float.
    ///
    /// An integer result has no float view to go through: `store` bitcasts an f32 (or packs an
    /// f16 half), and putting an integer through either would reinterpret its bits. The only
    /// producer is [`emit_pack_to_int`], whose whole purpose is to leave an integer in the lane
    /// for the integer groups to read.
    fn store_raw(&mut self, op: &Operand, c: usize, expr: &str) -> Option<()> {
        let prefix = bank_prefix(op.bank)?;
        let stmt = format!("  {prefix}[{}] = {expr};\n", op.index as u32 + c as u32);
        if !self.stage {
            self.body.push_str(&stmt);
            return Some(());
        }
        let tmp = format!("g{c}");
        let _ = writeln!(self.body, "  let {tmp} = {expr};");
        self.deferred.push(format!("  {prefix}[{}] = {tmp};\n", op.index as u32 + c as u32));
        Some(())
    }

    /// Apply every held-back store. Called once the instruction has read everything it reads.
    fn flush(&mut self) {
        for stmt in std::mem::take(&mut self.deferred) {
            self.body.push_str(&stmt);
        }
    }
}

/// Whether this instruction's destination shares a register with any of its sources, so the
/// emitted statements must read before they write (see [`Dest`]).
///
/// Deliberately conservative: it compares BANK and register index within the four-register
/// span an operand can address, without modelling which channels each end actually touches.
/// A false positive costs two extra lines of generated WGSL; a false negative is a silent
/// miscompile, and this is exactly the kind of analysis where being clever is how one gets in.
fn dest_aliases_source(instr: &Instr) -> bool {
    let Some(dest) = instr.dest.as_ref() else { return false };
    instr.srcs.iter().any(|s| {
        s.bank == dest.bank
            && (s.index as i32 - dest.index as i32).abs() < OPERAND_REGISTER_SPAN
    })
}

/// How many consecutive registers one operand can name: a four-channel F32 vector.
const OPERAND_REGISTER_SPAN: i32 = 4;

/// Emit the `VITASLOP_GXP_PROBE=<bank><idx>@<instr>` snapshot, if this is that instruction.
///
/// Copying the register into locals - rather than having the return expression read the bank
/// array at the end - is the whole point: every interesting intermediate in a lit material is
/// written again further down, so the end value answers a different question than the one asked.
fn emit_probe_snapshot(body: &mut String, index: usize, depth: usize) {
    let Some(spec) = crate::module::probe_spec() else { return };
    if spec.at != Some(index) {
        return;
    }
    let pad = "  ".repeat(depth);
    let (bank, i) = (spec.bank.as_str(), spec.index);
    let _ = writeln!(
        body,
        "{pad}_probe0 = {bank}[{i}]; _probe1 = {bank}[{}]; _probe2 = {bank}[{}];          _probe3 = {bank}[{}];",
        i + 1,
        i + 2,
        i + 3
    );
}

/// Emit a WGSL body from a fully-supported IR - the historical fragment entry point. The
/// USSE arithmetic core is identical for vertex and fragment programs, so this simply
/// delegates to [`emit_body`]; only the surrounding module I/O wrapper differs by kind.
pub fn emit_fragment(shader: &Shader) -> Result<String, EmitError> {
    emit_body(shader)
}

/// Emit the scalarised register-file statement body for a shader of EITHER kind. On success
/// returns the statements of a `fn ..._main(...)` the module builder wraps (register banks
/// bound as arrays). Vertex and fragment programs share this exactly - the difference is
/// only which banks are inputs (fragment: pa varyings; vertex: pa attributes) and how the
/// outputs are surfaced (fragment: o0/pa0 colour; vertex: o position + varyings), which the
/// module wrapper handles, not the body.
pub fn emit_body(shader: &Shader) -> Result<String, EmitError> {
    if shader.instrs.is_empty() {
        return Err(EmitError::Empty);
    }
    let mut body = String::new();
    // Track which internal-register lanes (i0..i3 x 4) an earlier instruction has written, so a
    // read of an unwritten internal lane in a FRAGMENT program hard-fails instead of translating
    // garbage: fragment internal registers can be pre-loaded by the texture-coordinate iterators
    // / PDS, which this model does not carry, so an unwritten read is genuinely unmodeled input.
    // A VERTEX program has no such preload - its internal registers are pure zero-initialised
    // scratch - so an unwritten read there is a defined 0.0 (a benign over-read of a padding lane
    // the fragment stage ignores, e.g. moving a computed vec3's absent w into an unused output
    // lane); the guard would wrongly reject those, so it applies to fragment programs only.
    let guard_internal_reads = shader.kind == ProgramKind::Fragment;
    let mut internal_written = [false; INTERNAL_LANES];
    emit_range(
        &mut body,
        shader,
        0,
        shader.instrs.len(),
        shader.instrs.len(),
        guard_internal_reads,
        &mut internal_written,
        None,
        1,
    )?;
    Ok(body)
}

/// Emit instructions `[start, end)`, turning USSE branches into structured WGSL.
///
/// A USSE branch is taken when its predicate holds, so the words it jumps OVER are exactly the
/// ones that execute when the predicate does not - which is a WGSL `if` on the negated
/// condition around the range `[branch+1, target)`. Ranges nest, so this recurses, and `end`
/// bounds how far a nested branch may jump: a target past the enclosing range is a jump out of
/// a block, which no `if` can express.
///
/// `exit` is the instruction index control reaches when THIS range simply runs off its end -
/// which for a then-arm is the if/else MERGE point, not `end`. It is what makes an early exit
/// expressible. A compiler emits
///
///   i:   br c        -> M      (skip the rest of this arm)
///   ...                        the rest of the arm
///   end:                       (the arm's last word, the jump over the else-arm)
///   ...                        the else-arm
///   M:                         the merge
///
/// and a branch to `M` from inside the arm is a jump out of the enclosing block by index, yet
/// it means exactly "stop executing this arm" - because running off the arm's end arrives at
/// `M` anyway. So a target equal to `exit` is rewritten to `end` and structures as an ordinary
/// skip. Anything else past `end` is a genuine jump out and still hard-fails: this rewrite is
/// an identity on the control flow, not a guess about where a branch meant to go.
///
/// Everything that is NOT a properly nested forward skip hard-fails naming itself. A BACKWARD
/// branch is a loop, and a loop cannot be reconstructed by skipping ranges - emitting its body
/// straight-line would run it exactly once, which is a plausible-looking wrong picture rather
/// than a failure. That is the class of silent error this recompiler refuses to make.
///
/// `internal_written` is carried through a conditional block as a UNION rather than being
/// discarded at its end. The guard it feeds asks "does this program ever write the lane it is
/// reading", because an internal lane no instruction writes is a PDS/iterator preload this model
/// does not carry; it is not a path-sensitive definite-assignment analysis. A write under a
/// branch answers that question, so intersecting at the join would reject shaders that are fine.
///
/// `depth` is only the indentation of the generated WGSL.
#[allow(clippy::too_many_arguments)]
fn emit_range(
    body: &mut String,
    shader: &Shader,
    start: usize,
    end: usize,
    exit: usize,
    guard_internal_reads: bool,
    internal_written: &mut [bool; INTERNAL_LANES],
    open_loop: Option<usize>,
    depth: usize,
) -> Result<(), EmitError> {
    let mut index = start;
    while index < end {
        // A loop is recognised by its BACK EDGE, and the instruction the back edge lands on is
        // this one - so the check belongs here, before the instruction is emitted as ordinary
        // straight-line code.
        // `open_loop` names the back edge of the loop this range is already the body of, and a
        // back edge points at its own head - so finding it here means `index` is that head and
        // the loop is already open. Anything else is a new loop starting here.
        match back_edge_to(shader, index).filter(|&tail| Some(tail) != open_loop) {
            Some(tail) if tail >= end => {
                return Err(EmitError::Blocked {
                    index: tail,
                    byte_offset: tail * 8,
                    reason: "0xF8 BR: a loop body extends past its enclosing block",
                    raw: shader.instrs[tail].raw,
                });
            }
            Some(tail) => {
                emit_loop(
                    body,
                    shader,
                    index,
                    tail,
                    guard_internal_reads,
                    internal_written,
                    depth,
                )?;
                index = tail + 1;
                continue;
            }
            None => {}
        }
        let instr = &shader.instrs[index];
        let byte_offset = index * 8;
        if let Some(reason) = instr.blocked {
            return Err(EmitError::Blocked { index, byte_offset, reason, raw: instr.raw });
        }
        let Op::Branch { rel } = instr.op else {
            if guard_internal_reads {
                check_internal_reads(instr, index, byte_offset, internal_written)?;
            }
            emit_instr(body, instr, index, byte_offset, shader.kind)?;
            emit_probe_snapshot(body, index, depth);
            record_internal_writes(instr, internal_written);
            index += 1;
            continue;
        };
        let blocked = |reason| Err(EmitError::Blocked { index, byte_offset, reason, raw: instr.raw });
        let target = index as i64 + rel as i64;
        if target <= index as i64 {
            return blocked("0xF8 BR jumps backward - a USSE loop is not reconstructed");
        }
        // A forward branch to the instruction after the innermost loop's back edge is a BREAK.
        // It is only reachable here when it leaves the current range - every range inside a loop
        // body ends at or before the back edge - so this never re-reads a target the ordinary
        // skip already expresses.
        if open_loop.is_some_and(|tail| target as usize == tail + 1) && target > end as i64 {
            // The branch's own predicate is the condition under which it is TAKEN, which is the
            // condition under which the loop is left - the opposite polarity from the skip form
            // below, where the guarded range is what runs when the branch is not taken.
            let taken = match instr.pred {
                Predicate::Always => None,
                Predicate::IfP(n) => Some(format!("p[{n}]")),
                Predicate::IfNotP(n) => Some(format!("!p[{n}]")),
                Predicate::Raw(_) => {
                    return blocked("0xF8 BR carries an unresolved predicate encoding")
                }
            };
            let pad = "  ".repeat(depth);
            match taken {
                Some(c) => {
                    let _ = writeln!(body, "{pad}if ({c}) {{ break; }}");
                    index += 1;
                }
                None => {
                    // Unconditional: everything after it in this range is unreachable, so
                    // emitting nothing for it is exact rather than a dropped instruction.
                    let _ = writeln!(body, "{pad}break;");
                    index = end;
                }
            }
            continue;
        }
        // A branch to this range's own exit point stops the range, which `end` already is.
        // See the `exit` note above: this is a re-indexing of the same control flow, not a
        // reinterpretation of it.
        let early_exit = target > end as i64 && target == exit as i64;
        if target > end as i64 && !early_exit {
            return blocked("0xF8 BR jumps out of its enclosing block - not structurable");
        }
        let target = if early_exit { end } else { target as usize };
        // The skipped range is what runs when the branch is NOT taken. An UNCONDITIONAL branch
        // therefore always skips it: the range is unreachable and emitting nothing for it is
        // exact. (This is the shape a compiler emits for the `else` arm's jump over the `then`
        // arm's tail, so it is not an oddity.)
        let cond = match instr.pred {
            Predicate::Always => None,
            Predicate::IfP(n) => Some(format!("!p[{n}]")),
            Predicate::IfNotP(n) => Some(format!("p[{n}]")),
            Predicate::Raw(_) => {
                return blocked("0xF8 BR carries an unresolved predicate encoding")
            }
        };
        let conditional = cond.is_some();
        let pad = "  ".repeat(depth);
        // IF/ELSE. When the last word of the skipped range is itself an UNCONDITIONAL forward
        // branch past `target`, that word is not part of the guarded body - it is the `then`
        // arm's jump over the `else` arm, which is exactly how a compiler lays an if/else out:
        //
        //   i:   br cond -> T        (skip the then-arm)
        //   i+1..T-2:                the then-arm
        //   T-1: br       -> E       (jump over the else-arm)
        //   T..E-1:                  the else-arm
        //   E:                       the merge point
        //
        // Recovering it matters beyond tidiness: without it the inner branch reads as a jump out
        // of its enclosing block and the whole pair falls back to fixed-function. Both of a
        // retail title's menu fragment programs are this shape.
        // An early exit has no else-arm to recover: its `target - 1` is just the last word of
        // the arm being cut short, not a compiler's jump over an alternative.
        //
        // THE MERGE CAN LIE OUTSIDE THIS RANGE, and refusing that is what blocked a whole
        // title's world. An if / else-if CHAIN compiles to arms that each end in a jump to the
        // chain's ONE merge point, so the inner arms' jumps target a word past their own
        // enclosing range's end:
        //
        //   28: br !p -> 64     30: br p -> 58     32: br p -> 51     34: br p -> 44
        //   36: br !p -> 64   37..42: arm A   43: br -> 64
        //   44..49: arm B     50: br -> 64
        //   51..56: arm C     57: br -> 64      58..63: arm D      64: the merge
        //
        // `e` is that merge and it is this range's own `exit`, which is the same statement as
        // "control leaves this range and arrives there" - so the arm is structurable after all.
        // What the range can CONTAIN still stops at `end`, so the else-arm's text is clamped to
        // it while its exit stays the true merge: ending the range IS the jump. Without this the
        // chain read as an arm jumping out of its block, the pair fell back to fixed-function,
        // and the title's terrain, characters and props were painted flat.
        let else_arm = (!early_exit && target > index + 1)
            .then(|| &shader.instrs[target - 1])
            .and_then(|last| match (last.op, last.pred) {
                (Op::Branch { rel: r }, Predicate::Always) => {
                    let e = (target - 1) as i64 + r as i64;
                    (e > target as i64
                        && (e <= end as i64 || e == exit as i64)
                        && last.blocked.is_none())
                    .then_some(e as usize)
                }
                _ => None,
            });
        // Where the else-arm's TEXT stops (never past this range), as against where control
        // goes when it runs off that text (`else_arm`, the merge).
        let else_end = else_arm.map(|e| e.min(end));
        match cond {
            // An unconditional branch always skips its range: that range is unreachable and
            // emitting nothing for it is exact, not a dropped instruction.
            None => {}
            Some(c) => {
                let then_end = if else_arm.is_some() { target - 1 } else { target };
                // Where the then-arm arrives when it runs off its end: the merge if this is an
                // if/else, otherwise the branch target - and, when the target was clamped as an
                // early exit, this whole range's own exit.
                let then_exit =
                    else_arm.unwrap_or(if early_exit { exit } else { target });
                let _ = writeln!(body, "{pad}if ({c}) {{");
                emit_range(
                    body,
                    shader,
                    index + 1,
                    then_end,
                    then_exit,
                    guard_internal_reads,
                    internal_written,
                    open_loop,
                    depth + 1,
                )?;
                match else_arm {
                    None => {
                        let _ = writeln!(body, "{pad}}}");
                    }
                    Some(e) => {
                        let _ = writeln!(body, "{pad}}} else {{");
                        emit_range(
                            body,
                            shader,
                            target,
                            else_end.unwrap_or(e),
                            e,
                            guard_internal_reads,
                            internal_written,
                            open_loop,
                            depth + 1,
                        )?;
                        let _ = writeln!(body, "{pad}}}");
                    }
                }
            }
        }
        // An unconditional branch consumes only its own skip; a conditional one that recovered
        // an else-arm has emitted through to the merge point.
        index = match (conditional, else_end) {
            (true, Some(e)) => e,
            _ => target,
        };
    }
    Ok(())
}

/// The back edge of a loop whose HEAD is `head`: the index of a branch that jumps back to
/// exactly `head`.
///
/// The search covers the WHOLE instruction stream rather than the range being emitted, because
/// a back edge that lands outside that range still makes `head` a loop head - one whose body
/// leaves the enclosing block, which is irreducible. Finding it here is what lets the caller
/// say so; searching only the range would meet the same branch later as a bare backward jump
/// and report the wrong cause.
///
/// When more than one branch targets `head` the LAST is taken as the back edge, so a `continue`
/// earlier in the body falls inside the loop region rather than cutting it short. [`emit_loop`]
/// then checks that what it found really is a single-entry, single-exit region, and hard-fails
/// if it is not - this only proposes the region.
fn back_edge_to(shader: &Shader, head: usize) -> Option<usize> {
    (head..shader.instrs.len()).rev().find(|&t| {
        matches!(shader.instrs[t].op, Op::Branch { rel } if t as i64 + rel as i64 == head as i64)
    })
}

/// Emit `[head, tail]` - a USSE loop whose back edge is the branch at `tail` - as a WGSL `loop`.
///
/// # What the hardware does and what this writes
/// The compiler lays a loop out as a body ending in a branch back to its first word, with the
/// exit as a forward branch out of the body:
///
/// ```text
///   head:   the test that computes the loop condition
///   head+1: br !cond -> tail+1        (leave)
///   ...     the body
///   tail:   br       -> head          (go round again)
///   tail+1: the instruction after the loop
/// ```
///
/// which is exactly a WGSL `loop { ... }` whose exit branches become `break`. Nothing is
/// reordered and no condition is re-derived: the body is emitted by the same [`emit_range`]
/// that emits straight-line code, with `open_loop` set to this back edge so a branch to the
/// instruction after it becomes the `break` it already is.
///
/// A CONDITIONAL back edge (`br cond -> head`) means "go round again if cond", so falling out
/// of the WGSL body must break when it does not hold - the negated form, written after the
/// body.
///
/// # What is checked, and why each check is not optional
/// A `loop` is only equivalent to the original control flow if the region is single-entry and
/// its only way out is the exit. All three are verified over the WHOLE instruction stream
/// rather than the enclosing range, because a branch from outside the range can reach into it
/// just as easily as one inside:
///
///  * exactly ONE branch in the region jumps backward, the back edge itself. A second one is a
///    second loop sharing this body, which a single `loop` cannot express.
///  * every branch in the region targets `[head, tail + 1]`. A jump anywhere else leaves the
///    loop for somewhere that is not its exit, which `break` does not mean.
///  * no branch from OUTSIDE the region targets STRICTLY INSIDE it. A jump into the middle of
///    a loop body is a second entry, and a `loop` has one.
///
/// Every failure hard-fails naming itself rather than emitting the body straight-line - running
/// a loop once is the plausible-looking wrong picture this recompiler exists to refuse.
fn emit_loop(
    body: &mut String,
    shader: &Shader,
    head: usize,
    tail: usize,
    guard_internal_reads: bool,
    internal_written: &mut [bool; INTERNAL_LANES],
    depth: usize,
) -> Result<(), EmitError> {
    let back = &shader.instrs[tail];
    let blocked = |reason| {
        Err(EmitError::Blocked { index: tail, byte_offset: tail * 8, reason, raw: back.raw })
    };
    if let Some(reason) = back.blocked {
        return Err(EmitError::Blocked { index: tail, byte_offset: tail * 8, reason, raw: back.raw });
    }
    for (at, instr) in shader.instrs.iter().enumerate() {
        let Op::Branch { rel } = instr.op else { continue };
        let target = at as i64 + rel as i64;
        if (head..=tail).contains(&at) {
            if target <= at as i64 && at != tail {
                return blocked("0xF8 BR: a second backward branch inside a loop body");
            }
            if target < head as i64 || target > tail as i64 + 1 {
                return blocked("0xF8 BR: a loop body branches somewhere that is neither inside \
                                the loop nor its exit");
            }
        } else if target > head as i64 && target <= tail as i64 {
            return blocked("0xF8 BR: a branch from outside jumps into the middle of a loop body");
        }
    }
    // The condition under which the back edge is TAKEN - i.e. the loop goes round again - so
    // the WGSL body breaks on its negation.
    let repeat = match back.pred {
        Predicate::Always => None,
        Predicate::IfP(n) => Some(format!("!p[{n}]")),
        Predicate::IfNotP(n) => Some(format!("p[{n}]")),
        Predicate::Raw(_) => return blocked("0xF8 BR carries an unresolved predicate encoding"),
    };
    let pad = "  ".repeat(depth);
    let _ = writeln!(body, "{pad}loop {{");
    emit_range(
        body,
        shader,
        head,
        tail,
        tail,
        guard_internal_reads,
        internal_written,
        Some(tail),
        depth + 1,
    )?;
    if let Some(c) = repeat {
        let _ = writeln!(body, "{pad}  if ({c}) {{ break; }}");
    }
    let _ = writeln!(body, "{pad}}}");
    Ok(())
}

/// The source channels an instruction actually reads: a dot sums channels `0..components`
/// regardless of the destination mask; every other wired op reads source channel `c` only
/// where it writes destination channel `c`.
fn read_channels(instr: &Instr) -> [bool; 4] {
    match instr.op {
        Op::Dot { components } => {
            let n = (components as usize).clamp(1, 4);
            [0 < n, 1 < n, 2 < n, 3 < n]
        }
        // A texture sample reads only its coordinate components (not the full write mask,
        // which covers the 4-channel RESULT), so the internal-read guard checks only those.
        Op::Tex { coords, .. } | Op::TexGather { coords, .. } => {
            let n = (coords as usize).clamp(1, 4);
            [0 < n, 1 < n, 2 < n, 3 < n]
        }
        // A memory load's only source is a scalar ADDRESS - one lane, whatever its
        // destination spans. Its write mask is explicitly not meaningful (the written span is
        // `elements` consecutive registers), so taking the mask as the read count claims the
        // three registers ABOVE the pointer are read too. That is how a pointer sitting near
        // the top of the SA bank made a program look like it read past its uniform buffer.
        Op::MemLoad { .. } => [true, false, false, false],
        _ => instr.write_mask,
    }
}

/// Hard-fail if any source reads an internal-register lane not yet written in-stream.
fn check_internal_reads(
    instr: &Instr,
    index: usize,
    byte_offset: usize,
    written: &[bool; INTERNAL_LANES],
) -> Result<(), EmitError> {
    let read = read_channels(instr);
    for src in &instr.srcs {
        if !matches!(src.bank, Bank::Internal) {
            continue;
        }
        for c in 0..4 {
            if !read[c] {
                continue;
            }
            let sel = src.swizzle[c];
            if sel > 3 {
                continue; // a swizzle constant reads no register lane
            }
            let lane = src.index as usize + sel as usize;
            if lane >= INTERNAL_LANES || !written[lane] {
                return Err(EmitError::UndefinedInternal {
                    index,
                    byte_offset,
                    lane: lane.min(u8::MAX as usize) as u8,
                    raw: instr.raw,
                });
            }
        }
    }
    Ok(())
}

/// Mark the internal-register lanes this instruction writes (an internal destination whose
/// masked channels become defined for later reads).
fn record_internal_writes(instr: &Instr, written: &mut [bool; INTERNAL_LANES]) {
    let Some(dest) = instr.dest.as_ref() else { return };
    if !matches!(dest.bank, Bank::Internal) {
        return;
    }
    for c in 0..4 {
        if instr.write_mask[c] {
            let lane = dest.index as usize + c;
            if lane < INTERNAL_LANES {
                written[lane] = true;
            }
        }
    }
}

/// The number of 32-bit registers each bank exposes in a wrapped module. The decoder scales
/// register indices by 2 (R7 reaches 254) and a swizzle can add up to 3 more, so the arrays
/// must hold at least 258 registers; 512 leaves headroom.
pub const BANK_REGS: usize = 512;

/// Wrap an emitted [`emit_fragment`] body into a complete, self-contained WGSL fragment
/// module: the register banks declared as private scalar arrays (the USSE register-file
/// model), the body as the function's statements, and the output register lanes returned as
/// the fragment colour. This is a STANDALONE, compilable module - used to validate that what
/// the emitter produces is real WGSL (see the naga test), and the skeleton the renderer's
/// pipeline builder will later bind pa/sa/samplers into. `pa`/`sa` are inputs the real
/// builder binds; here they are zeroed private storage so the module compiles in isolation.
pub fn wrap_module(body: &str, tex_units: &[TexBinding], kind: ProgramKind) -> String {
    let mut m = String::new();
    // The pipeline depth state, unconditionally: a body containing a DEPTHF reads it, and this
    // wrapper exists to validate ANY emittable body in isolation, so leaving it out would make
    // exactly the ops this file is meant to pin unvalidatable. It costs nothing here - the
    // wrapper is never a shipped pipeline.
    m.push_str(crate::link::GXP_DEPTH_DECL);
    // Each sampled unit needs a bound texture + sampler (referenced as `t{u}`/`s{u}` by
    // `emit_tex`). Group 0 / running bindings; the real pipeline builder assigns the same
    // names to the draw's bound textures (and its actual type - cube/3d for 3-coord samples).
    // Declared before the private register banks. Here a 3-coord sample validates as 3D.
    for (i, b) in tex_units.iter().enumerate() {
        let (tb, sb) = (i as u32 * 2, i as u32 * 2 + 1);
        let ty = if b.coords >= 3 { "texture_3d<f32>" } else { "texture_2d<f32>" };
        let (tex, samp) = sampler_names(kind, b.unit);
        let _ = writeln!(m, "@group(0) @binding({tb}) var {tex}: {ty};");
        let _ = writeln!(m, "@group(0) @binding({sb}) var {samp}: sampler;");
    }
    for bank in ["r", "pa", "sa", "o", "i"] {
        let _ = writeln!(m, "var<private> {bank}: array<u32, {BANK_REGS}>;");
    }
    // Predicate registers p0..p3, written by the test (VTST) ops and read by predicated
    // instructions. Four booleans, zero-initialised (a predicate is false until a test sets it).
    let _ = writeln!(m, "var<private> p: array<bool, 4>;");
    // The INDEX register file, for register-INDIRECT operands. Two registers, because the
    // extension row names exactly two indexed banks (INDEXED1 -> i0, INDEXED2 -> i1).
    let _ = writeln!(m, "var<private> idx: array<i32, 2>;");
    // `front_facing` is declared unconditionally - see the note in `link::build_linked_module`.
    let _ = writeln!(
        m,
        "\nstruct FsIn {{ @builtin(front_facing) front_facing: bool, @builtin(position) frag_coord: vec4<f32> }};"
    );
    let _ = writeln!(
        m,
        "\nstruct FsOut {{\n  @location(0) color: vec4<f32>,\n  @builtin(frag_depth) depth: f32,\n}};"
    );
    let _ = writeln!(m, "\n@fragment\nfn fs_main(in: FsIn) -> FsOut {{");
    m.push_str(FRONT_FACING_DECL);
    let _ = writeln!(m, "  let gxp_interp_depth = in.frag_coord.z;");
    let _ = writeln!(m, "  var gxp_frag_depth: f32 = gxp_interp_depth;");
    m.push_str(body);
    let _ = writeln!(
        m,
        "  return FsOut(vec4<f32>(bitcast<f32>(o[0]), bitcast<f32>(o[1]), bitcast<f32>(o[2]), bitcast<f32>(o[3])), gxp_frag_depth);\n}}"
    );
    m
}

/// Wrap an emitted [`emit_body`] into a complete, self-contained WGSL VERTEX module: the
/// register banks as private scalar arrays, the body as the function statements, the clip
/// position (`o0..o3`) returned as `@builtin(position)`, and `varying_vec4s` interpolant
/// outputs (`o[4..]` grouped four lanes per `@location`). Standalone (pa/sa are zeroed
/// private storage) so the emitted vertex body validates as real WGSL in isolation - the
/// counterpart to [`wrap_module`] for the fragment side. The real pipeline builder binds pa
/// from the vertex attributes and sa from the uniform buffer instead of zeroing them.
pub fn wrap_vertex_module(body: &str, varying_vec4s: u32) -> String {
    let mut m = String::new();
    // A body with 0xE8 memory loads references the draw's memory window; here it is zeroed
    // private storage (like pa/sa), sized minimally - this wrapper only validates syntax and
    // typing, never runs.
    if body.contains("gxp_mem") {
        // A stand-in for the real binding, in the ONE-window shape (header vec4 + 16 bytes)
        // so a body emitted for a program with a window is still a complete module here.
        let _ = writeln!(m, "var<private> gxp_mem: array<vec4<u32>, 2>;");
        let _ = m.write_str(&crate::module::mem_window_helper(&[crate::module::MemWindow {
            buffer_index: 0,
            bytes: 16,
            base_sa: 0,
            base_offset: 0,
        }]));
    }
    for bank in ["r", "pa", "sa", "o", "i"] {
        let _ = writeln!(m, "var<private> {bank}: array<u32, {BANK_REGS}>;");
    }
    let _ = writeln!(m, "var<private> p: array<bool, 4>;");
    // The INDEX register file, for register-INDIRECT operands. Two registers, because the
    // extension row names exactly two indexed banks (INDEXED1 -> i0, INDEXED2 -> i1).
    let _ = writeln!(m, "var<private> idx: array<i32, 2>;");
    // Output struct: clip position builtin + one vec4 per varying location.
    let _ = writeln!(m, "\nstruct VsOut {{");
    let _ = writeln!(m, "  @builtin(position) position: vec4<f32>,");
    for j in 0..varying_vec4s {
        let _ = writeln!(m, "  @location({j}) v{j}: vec4<f32>,");
    }
    let _ = writeln!(m, "}};");
    let _ = writeln!(m, "\n@vertex\nfn vs_main() -> VsOut {{");
    m.push_str(body);
    let f = |reg: u32| format!("bitcast<f32>(o[{reg}])");
    let _ = writeln!(m, "  var out: VsOut;");
    let _ = writeln!(m, "  out.position = vec4<f32>({}, {}, {}, {});", f(0), f(1), f(2), f(3));
    for j in 0..varying_vec4s {
        let b = 4 + j * 4;
        let _ = writeln!(
            m,
            "  out.v{j} = vec4<f32>({}, {}, {}, {});",
            f(b),
            f(b + 1),
            f(b + 2),
            f(b + 3)
        );
    }
    let _ = writeln!(m, "  return out;\n}}");
    m
}

fn emit_instr(
    body: &mut String,
    instr: &Instr,
    index: usize,
    byte_offset: usize,
    kind: ProgramKind,
) -> Result<(), EmitError> {
    // Reject an op the emitter has not wired before touching operands, so the error names
    // the op (what to implement next) rather than a missing-operand symptom.
    if !instr.op.is_emittable() {
        return Err(EmitError::UnsupportedOp {
            index,
            byte_offset,
            op: op_name(instr.op),
            group: instr.group,
            raw: instr.raw,
        });
    }
    // A no-op (phase declaration / NOP) has no destination and produces no statement.
    if matches!(instr.op, Op::Nop) {
        return Ok(());
    }
    // A branch is consumed by [`emit_range`]'s structuring, never emitted per-instruction.
    // Reaching here means that pass has a hole, and translating the branch as nothing would
    // silently run a skipped range - so say so instead.
    if matches!(instr.op, Op::Branch { .. }) {
        return Err(EmitError::Blocked {
            index,
            byte_offset,
            reason: "0xF8 BR reached the per-instruction emitter (branch structuring missed it)",
            raw: instr.raw,
        });
    }
    // A GLOBAL (SPECIAL hardware register) operand is decoded structurally but has no value
    // until its index's meaning is established. Report it by INDEX, ahead of the generic
    // unmapped-operand path, so the failure says which register to go and establish. The one
    // established register ([`global_u32_expr`]) is exempt, and only inside the operations that
    // read it as RAW BITS - the bitwise test and the integer one, which share the emitter's
    // raw-u32 operand path. Any other op reading even that register is outside what the corpus
    // establishes, so it still hard-fails by index.
    //
    // The integer arm is what a third title's lit materials use: `vtst <- GLOBAL[16], SA[zero]`
    // with EQ and then NE, the two-sided select on the facing bit (see the `(1, 10)` arm in the
    // VTST decoder). It reads the same 0-or-1 the bitwise form does, through the same
    // expression; refusing it here would leave those fifteen shaders on the fixed-function
    // fallback, which is what painted this title's whole world flat.
    let global_ok = matches!(instr.op, Op::Test { alu: TestAlu::BitAnd | TestAlu::IntSub, .. });
    if let Some(g) = instr
        .srcs
        .iter()
        .chain(instr.dest.iter())
        .find(|o| matches!(o.bank, Bank::Global) && !(global_ok && global_u32_expr(o, kind).is_some()))
    {
        return Err(EmitError::UnmodeledGlobal { index, byte_offset, global: g.index, raw: instr.raw });
    }
    let unmapped = || EmitError::UnmappedOperand { index, raw: instr.raw };
    let mask = instr.write_mask;

    // Emit the instruction's statements into a local buffer first, so a predicated
    // instruction can wrap them in an `if` on its predicate register (the writes execute
    // only when the predicate a VTST set holds). Unpredicated instructions append directly.
    //
    // The buffer goes through a [`Dest`], which is what enforces the hardware's read-all-then-
    // write-all ordering when this instruction's destination aliases one of its sources.
    let mut stmts = String::new();
    let staged = dest_aliases_source(instr);
    let mut sink = Dest { body: &mut stmts, deferred: Vec::new(), stage: staged };
    let s = &mut sink;
    // The two ops with no mandatory register destination: a predicate-only test writes just
    // `p[n]`, and a discard writes nothing at all.
    if let Op::Test { alu, cmp, reduce, pdst, write_back } = instr.op {
        emit_test(s, instr, instr.dest.as_ref(), alu, cmp, reduce, pdst, write_back, kind)
            .ok_or_else(unmapped)?;
        s.flush();
        return finish_predicated(body, instr, &block(&stmts, staged), index);
    }
    if let Op::TestMask { alu, cmp } = instr.op {
        let dest = instr.dest.as_ref().ok_or_else(unmapped)?;
        emit_test_mask(s, instr, dest, alu, cmp).ok_or_else(unmapped)?;
        s.flush();
        return finish_predicated(body, instr, &block(&stmts, staged), index);
    }
    if matches!(instr.op, Op::Kill) {
        return finish_predicated(body, instr, "  discard;\n", index);
    }
    // DEPTHF: replace the fragment's depth with a scalar the shader computed. The value is in
    // the GUEST's depth encoding (it is built out of `POSITION.z`, which `gxp_window_position`
    // delivers in exactly that space), so it goes through the inverse of the pipeline's own
    // clip-depth remap on the way to `@builtin(frag_depth)` - otherwise a written depth and an
    // interpolated one would be two different quantities in one depth buffer.
    //
    // Only a FRAGMENT program has a depth to write; a vertex program carrying this word is
    // not something the ISA describes, so it hard-fails rather than emitting a store to a
    // variable that stage does not have.
    if matches!(instr.op, Op::DepthF) {
        if !matches!(kind, ProgramKind::Fragment) {
            return Err(EmitError::UnmappedOperand { index, raw: instr.raw });
        }
        let src = instr.srcs.first().ok_or_else(unmapped)?;
        let e = src_channel(src, 0, Prec::of(instr)).ok_or_else(unmapped)?;
        let stmt = format!("  gxp_frag_depth = gxp_depth_to_window({e}, gxp_interp_depth);\n");
        return finish_predicated(body, instr, &stmt, index);
    }
    let dest = instr.dest.as_ref().ok_or_else(unmapped)?;
    let r = match instr.op {
        Op::Mul => emit_binop(s, instr, dest, mask, "*", index).ok_or_else(unmapped),
        Op::Add => emit_binop(s, instr, dest, mask, "+", index).ok_or_else(unmapped),
        Op::Min => emit_func2(s, instr, dest, mask, "min", index).ok_or_else(unmapped),
        Op::Max => emit_func2(s, instr, dest, mask, "max", index).ok_or_else(unmapped),
        Op::Frc => emit_func1(s, instr, dest, mask, "fract", index).ok_or_else(unmapped),
        Op::Dsx => emit_func1(s, instr, dest, mask, "dpdx", index).ok_or_else(unmapped),
        Op::Dsy => emit_func1(s, instr, dest, mask, "dpdy", index).ok_or_else(unmapped),
        Op::Mad => emit_mad(s, instr, dest, mask).ok_or_else(unmapped),
        Op::Dot { components } => emit_dot(s, instr, dest, mask, components).ok_or_else(unmapped),
        // Unary transcendentals (group 0x30) - the source broadcasts its single selected
        // component, so each written channel gets the same scalar function applied. rcp/rsq/
        // log/exp map to WGSL's native reciprocal / inverse-sqrt / log2 / exp2 (the SGX USSE
        // transcendentals are base-2). VMOV (0x38) is a swizzled per-channel copy.
        Op::Rcp => emit_unary(s, instr, dest, mask, &|a| format!("(1.0 / {a})")).ok_or_else(unmapped),
        Op::Rsq => emit_unary(s, instr, dest, mask, &|a| format!("inverseSqrt({a})")).ok_or_else(unmapped),
        Op::Log => emit_unary(s, instr, dest, mask, &|a| format!("log2({a})")).ok_or_else(unmapped),
        Op::Exp => emit_unary(s, instr, dest, mask, &|a| format!("exp2({a})")).ok_or_else(unmapped),
        // A move and a float<->float pack are both swizzled copies in the f32 register model.
        Op::Mov | Op::Pack { .. } => {
            emit_unary(s, instr, dest, mask, &|a| a.to_string()).ok_or_else(unmapped)
        }
        Op::Cmov { test } => emit_cmov(s, instr, dest, mask, test).ok_or_else(unmapped),
        Op::TexGather { unit, coords, coord_half } => {
            emit_tex_gather(s, instr, dest, unit, coords, coord_half, index, kind)
                .ok_or_else(unmapped)
        }
        Op::Tex { unit, coords, coord_half, lod } => {
            emit_tex(s, instr, dest, unit, coords, coord_half, lod, index, kind).ok_or_else(unmapped)
        }
        Op::Bitwise { kind, imm, lane_bits } => {
            emit_bitwise(s, instr, dest, kind, imm, lane_bits).ok_or_else(unmapped)
        }
        Op::PackToInt { bits, signed, .. } => {
            emit_pack_to_int(s, instr, dest, mask, bits, signed).ok_or_else(unmapped)
        }
        Op::IntMad { signed, bits } => emit_int_mad(s, instr, dest, signed, bits).ok_or_else(unmapped),
        Op::IntMadStep { signed, high_half } => {
            emit_int_mad_step(s, instr, dest, signed, high_half).ok_or_else(unmapped)
        }
        // MEMORY LOAD: `elements` consecutive 32-bit guest words from byte address
        // `src0 + offset_bytes` into consecutive destination registers. WGSL has no raw
        // pointers, so the read goes through the draw's bound MEMORY WINDOW: a uniform
        // array of the addressed guest bytes whose vec4 0 lane x holds the window's own
        // guest base address (see `module::MemWindow`). Only a VERTEX stage declares that
        // binding - no fragment program in the census loads memory - so a fragment body
        // carrying one hard-fails here instead of referencing an undeclared name.
        Op::MemLoad { elements, offset_bytes } => {
            if !matches!(kind, ProgramKind::Vertex) {
                return Err(EmitError::Blocked {
                    index,
                    byte_offset,
                    reason: "0xE8 memory load in a FRAGMENT program - the memory window \
                             binding is only established for the vertex stage",
                    raw: instr.raw,
                });
            }
            emit_mem_load(s, instr, dest, elements, offset_bytes, index).ok_or_else(unmapped)
        }
        Op::LoadIndex { addend } => emit_load_index(s, instr, dest, addend).ok_or_else(unmapped),
        Op::Sop2 { color, alpha, f1, f1_complement, f2, f2_complement } => {
            emit_sop2(s, instr, dest, mask, color, alpha, (f1, f1_complement), (f2, f2_complement))
                .ok_or_else(unmapped)
        }
        other => Err(EmitError::UnsupportedOp {
            index,
            byte_offset,
            op: op_name(other),
            group: instr.group,
            raw: instr.raw,
        }),
    };
    r?;
    s.flush();
    finish_predicated(body, instr, &block(&stmts, staged), index)
}

/// Wrap `stmts` in a WGSL block when the instruction staged its stores, so the `let`
/// temporaries [`Dest`] introduces are scoped to that one instruction and can never collide
/// with another's - including across the secondary and primary streams, which are emitted
/// separately and concatenated into one function.
fn block(stmts: &str, staged: bool) -> String {
    if !staged {
        return stmts.to_string();
    }
    format!("  {{\n{stmts}  }}\n")
}

/// `dest.c = (src1.c OP src2.c)` for each written channel.
fn emit_binop(body: &mut Dest, instr: &Instr, dest: &Operand, mask: [bool; 4], op: &str, _i: usize) -> Option<()> {
    let (s1, s2) = (instr.srcs.first()?, instr.srcs.get(1)?);
    let p = Prec::of(instr);
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let e = format!("({} {op} {})", src_channel(s1, c, p)?, src_channel(s2, c, p)?);
        body.store(dest, c, &e, p)?;
    }
    Some(())
}

/// `dest.c = FN(src1.c, src2.c)` for each written channel (min/max).
fn emit_func2(body: &mut Dest, instr: &Instr, dest: &Operand, mask: [bool; 4], func: &str, _i: usize) -> Option<()> {
    let (s1, s2) = (instr.srcs.first()?, instr.srcs.get(1)?);
    let p = Prec::of(instr);
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let e = format!("{func}({}, {})", src_channel(s1, c, p)?, src_channel(s2, c, p)?);
        body.store(dest, c, &e, p)?;
    }
    Some(())
}

/// `dest.c = FN(src1.c)` for each written channel (fract/dpdx/dpdy).
fn emit_func1(body: &mut Dest, instr: &Instr, dest: &Operand, mask: [bool; 4], func: &str, _i: usize) -> Option<()> {
    let s1 = instr.srcs.first()?;
    let p = Prec::of(instr);
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let e = format!("{func}({})", src_channel(s1, c, p)?);
        body.store(dest, c, &e, p)?;
    }
    Some(())
}

/// `dest.c = WRAP(src1.c)` for each written channel, where `wrap` builds the WGSL rvalue
/// from the source channel expression. Covers the transcendentals (rcp/rsq/log2/exp2) and a
/// plain move (`wrap` = identity), which do not fit the fixed `FN(x)` shape of `emit_func1`.
fn emit_unary(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    mask: [bool; 4],
    wrap: &dyn Fn(&str) -> String,
) -> Option<()> {
    let s1 = instr.srcs.first()?;
    // A format convert reads its source at one width and writes its destination at another.
    let (sp, p) = (Prec::src_of(instr), Prec::of(instr));
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let e = wrap(&src_channel(s1, c, sp)?);
        body.store(dest, c, &e, p)?;
    }
    Some(())
}

/// `dest.c = (int)src1.c`, a TRUNCATING float->integer convert (VPCK with `scale` clear).
///
/// The result is stored as the integer's two's-complement bit pattern in the destination lane,
/// which is the representation the integer groups read: the shader that needs this computes an
/// array index in float, converts it here, doubles it with a 16-bit-lane shift, and hands it to
/// the index register. Writing a float there instead would make the shift operate on an
/// exponent.
fn emit_pack_to_int(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    mask: [bool; 4],
    bits: u8,
    signed: bool,
) -> Option<()> {
    let s1 = instr.srcs.first()?;
    let sp = Prec::src_of(instr);
    let lane_mask: u32 = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let f = src_channel(s1, c, sp)?;
        // `trunc` before the cast, not `i32()` alone: WGSL's float->int conversion truncates
        // toward zero already, but saying so keeps the rounding explicit next to the mask, and
        // the clamp keeps a NaN or a huge float from being an undefined conversion.
        let conv = if signed {
            format!("bitcast<u32>(i32(clamp(trunc({f}), -2147483000.0, 2147483000.0)))")
        } else {
            format!("u32(clamp(trunc({f}), 0.0, 4294967000.0))")
        };
        let e = if lane_mask == u32::MAX { conv } else { format!("({conv} & {lane_mask:#x}u)") };
        body.store_raw(dest, c, &e)?;
    }
    Some(())
}

/// `idx[n] = src + addend` - load an index register for later register-INDIRECT addressing.
///
/// The source lane holds an integer bit pattern (it was produced by [`emit_pack_to_int`] and a
/// 16-bit shift), so it is read raw rather than through a float view.
fn emit_load_index(body: &mut Dest, instr: &Instr, dest: &Operand, addend: i32) -> Option<()> {
    let s1 = instr.srcs.first()?;
    let reg = dest.index.min(1) as u32;
    let bank = bank_prefix(s1.bank)?;
    writeln!(
        body,
        "  idx[{reg}] = i32({bank}[{}] & 0xffffu) + {addend}i;",
        s1.index as u32
    )
    .ok();
    Some(())
}

/// `elements` consecutive guest words from byte address `src0 + offset_bytes` into
/// consecutive destination registers, through the draw's bound MEMORY WINDOW.
///
/// The windows are bound as one `gxp_mem: array<vec4<u32>, N>` and resolved by ADDRESS
/// through the `gxp_mem_word` helper the module wrapper emits - see
/// [`crate::module::mem_window_helper`], which is where the layout and the address dispatch
/// are documented. The pointer register and the loaded values are raw 32-bit lanes (the
/// pointer was computed by the integer pipeline; the data's type is whatever the guest
/// stored), so everything here reads and writes the register file WITHOUT a float view.
///
/// A byte address that is not 4-aligned truncates to its containing word; the host refuses a
/// window (dropping the draw, reported) if its BASE is misaligned, and every in-shader offset
/// is a multiple of the 4-byte element size.
fn emit_mem_load(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    elements: u8,
    offset_bytes: u32,
    index: usize,
) -> Option<()> {
    let src0 = instr.srcs.first()?;
    let ptr_bank = bank_prefix(src0.bank)?;
    let dest_bank = bank_prefix(dest.bank)?;
    // The GUEST address of the first element. Named per instruction so two loads in one
    // (unbraced) function body cannot collide.
    writeln!(
        body,
        "  let gxp_a{index}: u32 = {ptr_bank}[{}] + {offset_bytes}u;",
        src0.index as u32
    )
    .ok()?;
    for k in 0..elements as u32 {
        writeln!(
            body,
            "  {dest_bank}[{}] = gxp_mem_word(gxp_a{index} + {}u);",
            dest.index as u32 + k,
            k * 4
        )
        .ok()?;
    }
    Some(())
}

/// `dest.c = src1.c * src2.c + src3.c` (multiply-add).
fn emit_mad(body: &mut Dest, instr: &Instr, dest: &Operand, mask: [bool; 4]) -> Option<()> {
    let (s1, s2, s3) = (instr.srcs.first()?, instr.srcs.get(1)?, instr.srcs.get(2)?);
    let p = Prec::of(instr);
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let e = format!(
            "{} * {} + {}",
            src_channel(s1, c, p)?,
            src_channel(s2, c, p)?,
            src_channel(s3, c, p)?
        );
        body.store(dest, c, &e, p)?;
    }
    Some(())
}

/// Conditional move (VMOVC): `dest.c = select(src2.c, src1.c, test(src0.c, 0))` per written
/// channel. `srcs` is `[src1 (true), src2 (false), src0 (test)]`; the WGSL `select(f, t,
/// cond)` returns `t` when `cond` is true, matching "src1 when the compare holds".
fn emit_cmov(body: &mut Dest, instr: &Instr, dest: &Operand, mask: [bool; 4], test: CompareMethod) -> Option<()> {
    let (s1, s2, s0) = (instr.srcs.first()?, instr.srcs.get(1)?, instr.srcs.get(2)?);
    let p = Prec::of(instr);
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let cond = compare_zero_expr(&src_channel(s0, c, p)?, test);
        let e = format!("select({}, {}, {cond})", src_channel(s2, c, p)?, src_channel(s1, c, p)?);
        body.store(dest, c, &e, p)?;
    }
    Some(())
}

/// The WGSL boolean expression testing scalar `a` against zero per the VMOVC compare method.
fn compare_zero_expr(a: &str, test: CompareMethod) -> String {
    let op = match test {
        CompareMethod::EqZero => "==",
        CompareMethod::NeZero => "!=",
        CompareMethod::LtZero => "<",
        CompareMethod::LteZero => "<=",
    };
    format!("({a} {op} 0.0)")
}



/// Append an instruction's emitted statements to `body`, gated on its predicate register.
/// `Raw` should not reach emit - the decoder either resolves a predicate to
/// Always/IfP/IfNotP or blocks the instruction - so a leftover raw predicate is a hard
/// failure rather than a dropped condition.
fn finish_predicated(
    body: &mut String,
    instr: &Instr,
    stmts: &str,
    index: usize,
) -> Result<(), EmitError> {
    match instr.pred {
        Predicate::Always => body.push_str(stmts),
        Predicate::IfP(n) => {
            writeln!(body, "  if (p[{n}]) {{").ok();
            body.push_str(stmts);
            writeln!(body, "  }}").ok();
        }
        Predicate::IfNotP(n) => {
            writeln!(body, "  if (!p[{n}]) {{").ok();
            body.push_str(stmts);
            writeln!(body, "  }}").ok();
        }
        Predicate::Raw(_) => {
            return Err(EmitError::Blocked {
                index,
                byte_offset: index * 8,
                reason: "unresolved predicate encoding reached emit",
                raw: instr.raw,
            })
        }
    }
    Ok(())
}

/// Emit a test -> predicate (VTST, group 0x48): evaluate the ALU per channel, compare each
/// result against zero, reduce the four booleans, and assign the single bit to `p[pdst]`.
///
/// The bitwise family works on the raw 32-bit lane, so its result is compared as an INTEGER
/// against zero; the float families compare as floats at the instruction's precision. With
/// `write_back` the raw ALU result is also stored to the destination, exactly as the encoding
/// says - so the instruction can double as an ALU op rather than silently losing that write.
fn emit_test(
    body: &mut Dest,
    instr: &Instr,
    dest: Option<&Operand>,
    alu: TestAlu,
    cmp: TestCmp,
    reduce: TestReduce,
    pdst: u8,
    write_back: bool,
    kind: ProgramKind,
) -> Option<()> {
    let (s1, s2) = (instr.srcs.first()?, instr.srcs.get(1)?);
    let p = Prec::of(instr);
    let op = match cmp {
        TestCmp::Eq => "==",
        TestCmp::Ne => "!=",
        TestCmp::Lt => "<",
        TestCmp::Le => "<=",
        TestCmp::Gt => ">",
        TestCmp::Ge => ">=",
    };
    // Which channels the reduction actually needs: a SELECT reads one, ANDALL/ORALL read all
    // four. Evaluating only those keeps the emitted body proportional to the work the
    // hardware's reduction observes.
    let channels: Vec<usize> = match reduce {
        TestReduce::Channel(c) => vec![(c as usize).min(3)],
        _ => (0..4).collect(),
    };
    // The BITWISE family tests the raw 32-bit lane, so its operands are read as u32 (not
    // through the float channel reader) and the comparison against zero is integer. Its banks
    // differ too - an inline immediate and a hardware register only ever appear here - so it
    // is resolved BEFORE the float path touches the operands.
    let raw = |o: &Operand| -> Option<String> {
        if matches!(o.bank, Bank::Constant) {
            let bank = if o.swizzle[0] == 1 { &CNST6_F32_BANK1 } else { &CNST6_F32_BANK0 };
            return Some(format!("{:#010x}u", bank[(o.index & 0x3f) as usize]));
        }
        // An inline integer literal the group assembled (the flag-bit mask).
        if matches!(o.bank, Bank::Immediate) {
            return Some(format!("{}u", o.index as u32));
        }
        // A hardware register: only the established ones materialise, and `emit_instr` has
        // already hard-failed on any other, so `None` here is a belt-and-braces refusal
        // rather than the reporting path.
        if matches!(o.bank, Bank::Global) {
            return global_u32_expr(o, kind);
        }
        Some(format!("{}[{}]", bank_prefix(o.bank)?, o.index as u32))
    };
    let mut bools = Vec::with_capacity(channels.len());
    for &c in &channels {
        if matches!(alu, TestAlu::BitAnd) {
            // The AND must be parenthesised: WGSL binds the equality operators TIGHTER than
            // `&`, so `a & b != 0u` parses as `a & (b != 0u)` and fails to validate (u32 vs
            // bool). Do not "simplify" these parentheses away.
            bools.push(format!("(({} & {}) {op} 0u)", raw(s1)?, raw(s2)?));
            continue;
        }
        // The INTEGER family reads its operands as the raw 32-bit lane, signed, exactly as the
        // 8-bit family reads its as four unorm bytes - see `TestAlu::IntSub`. Parenthesised for
        // the same precedence reason the bitwise arm is.
        if matches!(alu, TestAlu::IntSub) {
            bools.push(format!(
                "((bitcast<i32>({}) - bitcast<i32>({})) {op} 0)",
                raw(s1)?,
                raw(s2)?
            ));
            continue;
        }
        // The 8-bit family reads its operands as four unorm BYTES of one register, not as a
        // float lane. Taking the instruction's own precision here instead would read the flag
        // register 0x00000001 as an f32 denormal, compare it equal to zero, and turn the alpha
        // test it gates into a no-op that draws every cut-out texel.
        let p = if matches!(alu, TestAlu::Fx8Sub) { Prec::Fx8 } else { p };
        let (a, b) = (src_channel(s1, c, p)?, src_channel(s2, c, p)?);
        let value = match alu {
            TestAlu::Add => format!("({a} + {b})"),
            TestAlu::Sub | TestAlu::Fx8Sub => format!("({a} - {b})"),
            TestAlu::Mul => format!("({a} * {b})"),
            // Resolved above - the raw-lane paths never reach here.
            TestAlu::BitAnd | TestAlu::IntSub => {
                unreachable!("raw-lane test resolved before the float path")
            }
        };
        bools.push(format!("({value} {op} 0.0)"));
    }
    let expr = match reduce {
        TestReduce::Channel(_) => bools.remove(0),
        TestReduce::AndAll => bools.join(" && "),
        TestReduce::OrAll => bools.join(" || "),
    };
    writeln!(body, "  p[{pdst}] = {expr};").ok();

    // `test_wben`: the ALU result also lands in the destination register, on every channel the
    // write mask selects.
    if write_back {
        let dest = dest?;
        for c in 0..4 {
            if !instr.write_mask[c] {
                continue;
            }
            let wp = if matches!(alu, TestAlu::Fx8Sub) { Prec::Fx8 } else { p };
            let (a, b) = (src_channel(s1, c, wp)?, src_channel(s2, c, wp)?);
            let value = match alu {
                TestAlu::Add => format!("({a} + {b})"),
                TestAlu::Sub | TestAlu::Fx8Sub => format!("({a} - {b})"),
                TestAlu::Mul => format!("({a} * {b})"),
                // A raw-lane write-back is not modelled in the float store path; the corpus
                // has no such instruction, so refusing is exact rather than restrictive.
                TestAlu::BitAnd | TestAlu::IntSub => return None,
            };
            body.store(dest, c, &value, wp)?;
        }
    }
    Some(())
}


/// GATHER4 with bilinear coefficients ([`Op::TexGather`], group 0xE0 `sb_mode == 3`).
///
/// The instruction produces SIX registers from one 2x2 texel footprint:
///
/// ```text
///   dest + 0 .. 3   the four gathered texels, in the platform's own gather order
///   dest + 4 .. 5   four F16 bilinear coefficients, packed two per register
/// ```
///
/// # Where the coefficient base comes from
/// The reference states only that the coefficients follow "at `dest.num + component_size`".
/// The corpus fixes the number: the golf title's shadow filter gathers a ONE-component depth
/// map into `r0` and the instruction two later dots the coefficients out of `r4` - so the four
/// gathered texels occupy four registers and the coefficients start after them. That is the
/// only sampler width this is decoded for (see `decode_shader`), so the two candidate readings
/// of "component_size" cannot disagree here.
///
/// # Which coefficient weights which texel
/// `textureGather` returns the footprint in the platform's order - the texels at `(x0,y1)`,
/// `(x1,y1)`, `(x1,y0)`, `(x0,y0)` relative to the same 2x2 a bilinear filter would take - and
/// `fract(uv * dims - 0.5)` is that filter's own pair of fractions, so the four bilinear
/// weights are determined once the pairing is.
///
/// The pairing is the one thing the instruction does not state, and the corpus's only consumer
/// is what settles it: the shadow filter reduces the gathered comparisons with
/// `dot(mask.wzyx, coeff.xyzw)`, i.e. it pairs coefficient `k` with gathered texel `3 - k`. A
/// compiler that emitted that swizzle knew the hardware's coefficient order is the REVERSE of
/// its gather order, so the coefficients are written in reverse footprint order below. Both
/// halves are emitted here from the same footprint, so the pairing is self-consistent whatever
/// absolute order the platform's gather uses - what the swizzle fixes is which weight goes with
/// which texel, and getting that wrong would mis-weight the filter within a single texel rather
/// than change what it samples.
#[allow(clippy::too_many_arguments)]
fn emit_tex_gather(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    unit: u8,
    coords: u8,
    coord_half: bool,
    index: usize,
    kind: ProgramKind,
) -> Option<()> {
    // The decoder refuses every other dimensionality; this keeps the emitter honest if that
    // ever changes without a footprint rule to go with it.
    if coords != 2 {
        return None;
    }
    let (tex, samp) = sampler_names(kind, unit);
    let coord = instr.srcs.first()?;
    let cp = if coord_half { Prec::F16 } else { Prec::F32 };
    let (cx, cy) = (src_channel(coord, 0, cp)?, src_channel(coord, 1, cp)?);
    let uv = format!("_guv{index}");
    let g = format!("_g{index}");
    let f = format!("_gf{index}");
    writeln!(body, "  let {uv} = vec2<f32>({cx}, {cy});").ok();
    writeln!(body, "  let {g} = textureGather(0u, {tex}, {samp}, {uv});").ok();
    writeln!(
        body,
        "  let {f} = fract({uv} * vec2<f32>(textureDimensions({tex}, 0u)) - vec2<f32>(0.5));"
    )
    .ok();
    const COMP: [&str; 4] = ["x", "y", "z", "w"];
    for c in 0..4 {
        body.store(dest, c, &format!("{g}.{}", COMP[c]), Prec::F32)?;
    }
    // The coefficients live four registers past the gathered texels, and they are F16 - two to
    // a register - which is how four of them fit in the two the consumer reads as one vec4.
    let coeff = Operand::plain(dest.bank, dest.index.checked_add(4)?, dest.bank_sel);
    let weights = [
        format!("((1.0 - {f}.x) * (1.0 - {f}.y))"),
        format!("({f}.x * (1.0 - {f}.y))"),
        format!("({f}.x * {f}.y)"),
        format!("((1.0 - {f}.x) * {f}.y)"),
    ];
    for (c, w) in weights.iter().enumerate() {
        // Slots 4..8: this is the SECOND group of stores from one instruction, and the first
        // four already hold `g0..g3` - see `Dest::store_in_slot`.
        body.store_in_slot(&coeff, c, c + 4, w, Prec::F16)?;
    }
    Some(())
}

/// VTSTMSK ([`Op::TestMask`]): the same per-channel `alu(src1, src2)` compared against zero as
/// [`emit_test`], written out as one NUMERIC value per channel instead of reduced to a
/// predicate bit.
///
/// Only the float ALU families reach here - the decoder blocks the integer and bitwise ones,
/// which have no corpus instance in this group - so every channel is a float comparison and the
/// mask value is `1.0` or `0.0` at the instruction's own precision.
fn emit_test_mask(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    alu: TestAlu,
    cmp: TestCmp,
) -> Option<()> {
    let (s1, s2) = (instr.srcs.first()?, instr.srcs.get(1)?);
    let p = Prec::of(instr);
    let op = match cmp {
        TestCmp::Eq => "==",
        TestCmp::Ne => "!=",
        TestCmp::Lt => "<",
        TestCmp::Le => "<=",
        TestCmp::Gt => ">",
        TestCmp::Ge => ">=",
    };
    for c in 0..4 {
        let (a, b) = (src_channel(s1, c, p)?, src_channel(s2, c, p)?);
        let value = match alu {
            TestAlu::Add => format!("({a} + {b})"),
            TestAlu::Sub => format!("({a} - {b})"),
            TestAlu::Mul => format!("({a} * {b})"),
            // The decoder does not produce these for this group; refusing keeps the emitter
            // from inventing a raw-lane mask if that ever changes.
            TestAlu::Fx8Sub | TestAlu::BitAnd | TestAlu::IntSub => return None,
        };
        body.store(dest, c, &format!("select(0.0, 1.0, ({value} {op} 0.0))"), p)?;
    }
    Some(())
}

/// Dot product: a scalar `src1 . src2` over `components` channels, broadcast to every
/// written destination channel.
fn emit_dot(body: &mut Dest, instr: &Instr, dest: &Operand, mask: [bool; 4], components: u8) -> Option<()> {
    let (s1, s2) = (instr.srcs.first()?, instr.srcs.get(1)?);
    let p = Prec::of(instr);
    let n = (components as usize).clamp(1, 4);
    let mut terms = Vec::new();
    for c in 0..n {
        terms.push(format!("{} * {}", src_channel(s1, c, p)?, src_channel(s2, c, p)?));
    }
    let expr = format!("({})", terms.join(" + "));
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        body.store(dest, c, &expr, p)?;
    }
    Some(())
}

/// Texture sample: `dest.xyzw = textureSample(t{unit}, s{unit}, coord)`. The coordinate is
/// `srcs[0]`, read as `coords` components (1D pads Y to 0); the bound texture+sampler are
/// the module-scope `t{unit}`/`s{unit}` bindings the pipeline builder wires. The sampled
/// RGBA is written to the destination's four channels.
#[allow(clippy::too_many_arguments)]
fn emit_tex(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    unit: u8,
    coords: u8,
    coord_half: bool,
    lod: TexLod,
    index: usize,
    kind: ProgramKind,
) -> Option<()> {
    let (tex, samp) = sampler_names(kind, unit);
    let coord = instr.srcs.first()?;
    // The coordinate and the result carry INDEPENDENT precisions (`src0_type` vs
    // `fconv_type`): a shader routinely computes an F16 UV and asks for an F16 result, but
    // either can be F32, and reading the coordinate at the wrong width samples garbage.
    let cp = if coord_half { Prec::F16 } else { Prec::F32 };
    let dp = Prec::of(instr);
    let cx = src_channel(coord, 0, cp)?;
    let tmp = format!("_tex{index}");
    // 3-component samples (3D/cube) pass a vec3 direction/coordinate; 1D/2D pass a vec2
    // (1D pads Y to 0). The bound-texture type (2d/3d/cube) is chosen at pipeline-build.
    // The mip level, when the encoding supplies one. It is a scalar read at F32 (see the
    // decoder's note) and selects the WGSL sample variant.
    let (func, extra) = match lod {
        TexLod::Implicit => ("textureSample", String::new()),
        TexLod::Bias => ("textureSampleBias", format!(", {}", src_channel(instr.srcs.get(1)?, 0, Prec::F32)?)),
        TexLod::Level => ("textureSampleLevel", format!(", {}", src_channel(instr.srcs.get(1)?, 0, Prec::F32)?)),
        // The gradient form supplies BOTH derivatives from src2, packed one after the other:
        // for a 2D sample components 0,1 are ddx and 2,3 are ddy (spec E0.4). They are
        // vectors of the same arity as the coordinate, so a 3D sample would take 3 and 3 -
        // the decoder blocks that case rather than guess where the second vector starts.
        TexLod::Gradient => {
            let g = instr.srcs.get(1)?;
            let (ddx0, ddx1) = (src_channel(g, 0, Prec::F32)?, src_channel(g, 1, Prec::F32)?);
            let (ddy0, ddy1) = (src_channel(g, 2, Prec::F32)?, src_channel(g, 3, Prec::F32)?);
            (
                "textureSampleGrad",
                format!(", vec2<f32>({ddx0}, {ddx1}), vec2<f32>({ddy0}, {ddy1})"),
            )
        }
    };
    if coords >= 3 {
        let cy = src_channel(coord, 1, cp)?;
        let cz = src_channel(coord, 2, cp)?;
        writeln!(body, "  let {tmp} = {func}({tex}, {samp}, vec3<f32>({cx}, {cy}, {cz}){extra});").ok();
    } else {
        let cy = if coords >= 2 { src_channel(coord, 1, cp)? } else { "0.0".to_string() };
        writeln!(body, "  let {tmp} = {func}({tex}, {samp}, vec2<f32>({cx}, {cy}){extra});").ok();
    }
    const COMP: [&str; 4] = ["x", "y", "z", "w"];
    for c in 0..4 {
        body.store(dest, c, &format!("{tmp}.{}", COMP[c]), dp)?;
    }
    Some(())
}

/// The 8-bit sum-of-products combiner ([`Op::Sop2`]), one statement per written channel:
///
/// ```text
///   dest.c = op_c( coeff1.c * src1.c , coeff2.c * src2.c )
/// ```
///
/// where `op_c` is the COLOUR operation for channels 0..2 and the ALPHA operation for channel
/// 3, and each coefficient comes from its selector (optionally one's-complemented). Everything
/// is read and written through [`Prec::Fx8`], so the arithmetic happens on `byte / 255` values
/// and lands back in the right byte of the destination register.
///
/// # The selector is a coefficient, and that is the whole instruction
/// `Zero` with the complement bit set is the coefficient 1, which makes the term a plain copy
/// of its source register. Reading the selector as "the operand is zero" instead makes this
/// instruction a constant and leaves the register it names doing nothing, which is how the
/// family read as unusable for several sessions. See [`SopFactor`].
#[allow(clippy::too_many_arguments)]
fn emit_sop2(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    mask: [bool; 4],
    color: SopOp,
    alpha: SopOp,
    (f1, f1_complement): (SopFactor, bool),
    (f2, f2_complement): (SopFactor, bool),
) -> Option<()> {
    let (s1, s2) = (instr.srcs.first()?, instr.srcs.get(1)?);
    // The coefficient for channel `c`. An ALPHA selector broadcasts channel 3, which is what
    // makes "modulate by the source's alpha" one instruction.
    let coeff = |f: SopFactor, complement: bool, c: usize| -> Option<String> {
        let base = match f {
            SopFactor::Zero => "0.0".to_string(),
            SopFactor::Src1Color => src_channel(s1, c, Prec::Fx8)?,
            SopFactor::Src1Alpha => src_channel(s1, 3, Prec::Fx8)?,
            SopFactor::Src2Color => src_channel(s2, c, Prec::Fx8)?,
            SopFactor::Src2Alpha => src_channel(s2, 3, Prec::Fx8)?,
        };
        Some(if complement { format!("(1.0 - {base})") } else { base })
    };
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let t1 = format!("({} * {})", coeff(f1, f1_complement, c)?, src_channel(s1, c, Prec::Fx8)?);
        let t2 = format!("({} * {})", coeff(f2, f2_complement, c)?, src_channel(s2, c, Prec::Fx8)?);
        let op = if c == 3 { alpha } else { color };
        let expr = match op {
            SopOp::Add => format!("({t1} + {t2})"),
            SopOp::Sub => format!("({t1} - {t2})"),
            SopOp::Min => format!("min({t1}, {t2})"),
            SopOp::Max => format!("max({t1}, {t2})"),
        };
        body.store(dest, c, &expr, Prec::Fx8)?;
    }
    Some(())
}

/// Integer bitwise / shift on channel 0 only, operating on the 32-bit lane bit pattern:
/// `dest.x = bitcast<f32>(bitcast<u32>(src1.x) OP b)`, where `b` is the inline immediate or
/// `bitcast<u32>(src2.x)`. Shift amounts are masked to 31; ASR uses a signed shift.
/// Emit a group-0x15 IMAD32: `dest = src0 * src1 + src2`, scalar, over the 32-bit lane read as
/// an integer.
///
/// The register file is `array<u32>`, so this is the natural view and no bitcast is needed for
/// the UNSIGNED form. The signed form goes through `i32` for the multiply and the add - the two
/// differ on overflow, and WGSL defines both as wrapping, so the signedness has to be honoured
/// rather than let a `u32` multiply stand in.
///
/// Only channel 0 is written: the group carries no write mask, and the instruction is scalar.
fn emit_int_mad(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    signed: bool,
    bits: u8,
) -> Option<()> {
    // The decoder only produces 32 today and blocks the narrower widths by name; this keeps the
    // emitter honest if that ever changes without the emitter being taught the masking.
    if bits != 32 {
        return None;
    }
    // Each source is read as a raw lane. An IMMEDIATE source is materialised inline - it has no
    // register-file storage - which is the one case `bank_prefix` cannot spell.
    let raw = |o: &Operand| -> Option<String> {
        if matches!(o.bank, Bank::Immediate) {
            return Some(format!("{}u", o.index as u32));
        }
        if matches!(o.bank, Bank::Indexed) {
            return indexed_element(o, 0);
        }
        Some(format!("{}[{}]", bank_prefix(o.bank)?, o.index as u32))
    };
    let a = raw(instr.srcs.first()?)?;
    let b = raw(instr.srcs.get(1)?)?;
    let c = raw(instr.srcs.get(2)?)?;
    let expr = if signed {
        format!("bitcast<u32>(bitcast<i32>({a}) * bitcast<i32>({b}) + bitcast<i32>({c}))")
    } else {
        format!("({a} * {b} + {c})")
    };
    writeln!(body, "  {}[{}] = {};", bank_prefix(dest.bank)?, dest.index as u32, expr).ok();
    Some(())
}

/// One STEP of the group-0x1a 32-bit integer multiply-add: `dest = half(src0) * src1 + src2`.
///
/// All three operands and the result are 32-bit lanes read as unsigned bit patterns, and WGSL's
/// `u32` arithmetic wraps, which is what makes the two steps sum to the whole product: the low
/// step keeps the bits below 2^32 and the high step's `<< 16` drops exactly the ones the 32-bit
/// result never had. Writing this as a widening 64-bit multiply would keep bits the hardware
/// discards.
fn emit_int_mad_step(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    signed: bool,
    high_half: bool,
) -> Option<()> {
    // The decoder blocks the signed form by name; this keeps the emitter honest if that ever
    // changes without the sign rule for the two halves being established first.
    if signed {
        return None;
    }
    let raw = |o: &Operand| -> Option<String> {
        if matches!(o.bank, Bank::Immediate) {
            return Some(format!("{}u", o.index as u32));
        }
        if matches!(o.bank, Bank::Indexed) {
            return indexed_element(o, 0);
        }
        Some(format!("{}[{}]", bank_prefix(o.bank)?, o.index as u32))
    };
    let a = raw(instr.srcs.first()?)?;
    let b = raw(instr.srcs.get(1)?)?;
    let c = raw(instr.srcs.get(2)?)?;
    let expr = if high_half {
        format!("((({a} >> 16u) * {b}) << 16u) + {c}")
    } else {
        format!("(({a} & 0xffffu) * {b}) + {c}")
    };
    writeln!(body, "  {}[{}] = {};", bank_prefix(dest.bank)?, dest.index as u32, expr).ok();
    Some(())
}

fn emit_bitwise(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    kind: BitwiseKind,
    imm: Option<u32>,
    lane_bits: u8,
) -> Option<()> {
    use BitwiseKind::*;
    // A 16-bit lane operates on the low half and WRAPS there. The mask is part of the result,
    // not a tidy-up: a left shift that overflows 16 bits keeps different bits than one that
    // overflows 32, and this instruction's whole job in the shader that needs it is to double
    // a small integer index.
    let mask: u32 = if lane_bits >= 32 { u32::MAX } else { (1u32 << lane_bits) - 1 };
    let shift_mask = lane_bits as u32 - 1;
    // VBW is an integer op on the 32-bit lane bit pattern, so both operands are read as raw
    // registers rather than through a float precision view.
    let raw = |o: &Operand| -> Option<String> {
        // A hardware-constant source (the extended SPECIAL bank resolving to FPCONSTANT)
        // contributes its 32-bit table entry verbatim: VBW is a bit-pattern op, so the entry
        // is used as stored rather than through a float view. Channel 0 is the only channel a
        // scalar VBW reads, and its swizzle selector picks the bank exactly as elsewhere.
        if matches!(o.bank, Bank::Constant) {
            let bank = if o.swizzle[0] == 1 { &CNST6_F32_BANK1 } else { &CNST6_F32_BANK0 };
            return Some(format!("{:#010x}u", bank[(o.index & 0x3f) as usize]));
        }
        // A register-INDIRECT source: the element is only known at run time, so it spells out
        // the address rather than a constant index. See [`indexed_element`].
        if matches!(o.bank, crate::ir::Bank::Indexed) {
            return indexed_element(o, 0);
        }
        Some(format!("{}[{}]", bank_prefix(o.bank)?, o.index as u32))
    };
    let masked = |e: String| -> String {
        if mask == u32::MAX { e } else { format!("({e} & {mask:#x}u)") }
    };
    let a = masked(raw(instr.srcs.first()?)?);
    let b = masked(match imm {
        Some(v) => format!("{v}u"),
        None => raw(instr.srcs.get(1)?)?,
    });
    let expr = match kind {
        And => format!("({a} & {b})"),
        Or => format!("({a} | {b})"),
        Xor => format!("({a} ^ {b})"),
        Shl => format!("({a} << ({b} & {shift_mask}u))"),
        Shr => format!("({a} >> ({b} & {shift_mask}u))"),
        // Arithmetic shift is over the LANE's sign bit, so a narrow lane is sign-extended to
        // 32 first and re-masked by the write below.
        Asr if lane_bits >= 32 => format!("bitcast<u32>(bitcast<i32>({a}) >> ({b} & 31u))"),
        Asr => {
            let up = 32 - lane_bits as u32;
            format!("bitcast<u32>((bitcast<i32>({a} << {up}u) >> {up}u) >> ({b} & {shift_mask}u))")
        }
    };
    writeln!(body, "  {}[{}] = {};", bank_prefix(dest.bank)?, dest.index as u32, masked(expr)).ok();
    Some(())
}

/// The WGSL expression for one element of a register-INDIRECT ([`Bank::Indexed`]) operand,
/// `iteration` steps past its base.
///
/// The operand's own 7-bit number carries the bank and an additive offset; the index register
/// supplies the rest and is only known at run time. `iteration` is the repeat step - a repeated
/// instruction walks consecutive elements, which is how one instruction reads a whole
/// two-component array entry.
///
/// The index is clamped to the bank's size. A dynamic index is the one operand form that can
/// address outside the register file at all, and WGSL's behaviour for an out-of-bounds dynamic
/// index is not something to leave to chance in a shader that samples a texture with the result.
fn indexed_element(o: &Operand, iteration: u32) -> Option<String> {
    let bank = bank_prefix(crate::ir::indexed_sub_bank(o.index))?;
    let offset = crate::ir::indexed_offset(o.index) + iteration;
    // `bank_sel` records which index register the extension row named: INDEXED1 -> i0,
    // INDEXED2 -> i1.
    let reg = if o.bank_sel == 0 { 0 } else { 1 };
    Some(format!(
        "{bank}[min(u32(max(idx[{reg}] + {offset}i, 0i)), {}u)]",
        BANK_REGS - 1
    ))
}

/// A texture/sampler binding the emitted body references: the sampler `unit` (the SMP
/// operand's sampler ordinal, which the container resolves to a GXM texture unit) and the
/// number of coordinate components sampled. The pipeline builder binds the draw's bound
/// texture+sampler to the module-scope `t{unit}`/`s{unit}` names (matching [`emit_tex`]).
///
/// `cube` distinguishes the two three-coordinate cases, which need different WGSL texture
/// types and different bound view dimensions: a CUBE map's three coordinates are a direction,
/// a 3D texture's are a volume position. It comes from the container's own sampler flag, not
/// from the coordinate count - see [`crate::container::Program::sampler_is_cube`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TexBinding {
    pub unit: u8,
    pub coords: u8,
    pub cube: bool,
}

impl TexBinding {
    /// The WGSL texture type this binding must be declared as.
    pub fn wgsl_type(&self) -> &'static str {
        match (self.coords >= 3, self.cube) {
            (true, true) => "texture_cube<f32>",
            (true, false) => "texture_3d<f32>",
            _ => "texture_2d<f32>",
        }
    }
}

/// The module-scope WGSL names for one stage's texture + sampler at `unit`.
///
/// The two stages have INDEPENDENT sampler unit numbering, so a linked module can hold a
/// vertex `unit 0` and a fragment `unit 0` that are different textures. They must therefore be
/// different identifiers, or the vertex fetch silently reads the fragment's texture - and a
/// vertex fetch builds GEOMETRY, so that is not a shading error, it is the wrong mesh.
pub fn sampler_names(kind: ProgramKind, unit: u8) -> (String, String) {
    match kind {
        ProgramKind::Vertex => (format!("vt{unit}"), format!("vs{unit}")),
        _ => (format!("t{unit}"), format!("s{unit}")),
    }
}

/// The distinct sampler bindings a shader references, ascending by unit. Deduplicated; if a
/// unit is sampled with more than one coordinate count the larger is reported (the binding
/// must satisfy every sample of that unit).
///
/// `is_cube` answers, for a GXM texture unit, whether the container declares that sampler a
/// CUBE map. It is a callback because the parameter table lives in the container while this
/// walk only sees the decoded instruction stream.
pub fn tex_units(shader: &Shader, is_cube: impl Fn(u8) -> bool) -> Vec<TexBinding> {
    let mut out: Vec<TexBinding> = Vec::new();
    for i in &shader.instrs {
        let sampled = match i.op {
            Op::Tex { unit, coords, .. } | Op::TexGather { unit, coords, .. } => {
                Some((unit, coords))
            }
            _ => None,
        };
        if let Some((unit, coords)) = sampled {
            match out.iter_mut().find(|b| b.unit == unit) {
                Some(b) => b.coords = b.coords.max(coords),
                None => out.push(TexBinding { unit, coords, cube: is_cube(unit) }),
            }
        }
    }
    out.sort_unstable_by_key(|b| b.unit);
    out
}

/// A stable mnemonic for an op, for error messages naming what to wire next.
fn op_name(op: Op) -> &'static str {
    op.mnemonic()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::ProgramKind;
    use crate::ir::{Bank, Instr, Predicate};

    fn instr(op: Op, dest: Option<Operand>, srcs: Vec<Operand>) -> Instr {
        Instr {
            op,
            pred: Predicate::Always,
            dest,
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

    /// An F32 register read as the emitter writes it (the register file is raw `u32`).
    fn rd(bank: &str, reg: u32) -> String {
        format!("bitcast<f32>({bank}[{reg}])")
    }

    /// An F32 register store statement as the emitter writes it.
    fn st(bank: &str, reg: u32, expr: &str) -> String {
        format!("{bank}[{reg}] = bitcast<u32>({expr});")
    }

    /// A GATHER whose destination ALIASES its coordinate operand is emitted through the
    /// staging path, and that path names its temporary after the CHANNEL - which gave the four
    /// texel stores and the four coefficient stores the same four names in one block. WGSL
    /// rejects the redefinition, wgpu then refuses the pipeline, and the renderer DROPS every
    /// draw that pair ever makes: a whole shadow-filtered material family vanishes from the
    /// frame over a name collision. Two of a retail title's pairs did exactly that.
    ///
    /// Pinned by asserting no `let` name is declared twice, rather than by naming `g4`: what
    /// has to hold is uniqueness, not a particular spelling.
    #[test]
    fn a_gather_that_aliases_its_coordinate_declares_each_temporary_once() {
        // Destination within OPERAND_REGISTER_SPAN of the source, which is what turns staging
        // on - the condition the shipping failure needed and the plain test above does not meet.
        let d = Operand::plain(Bank::PrimaryAttr, 6, 2);
        let coord = Operand::plain(Bank::PrimaryAttr, 4, 2);
        let wgsl = emit_fragment(&shader(vec![instr(
            Op::TexGather { unit: 3, coords: 2, coord_half: false },
            Some(d),
            vec![coord],
        )]))
        .unwrap();
        assert!(wgsl.contains("let g0 ="), "the staging path must be the one under test:
{wgsl}");
        let mut seen = std::collections::BTreeSet::new();
        for line in wgsl.lines() {
            let t = line.trim();
            let Some(rest) = t.strip_prefix("let ") else { continue };
            let name = rest.split(|c: char| !c.is_ascii_alphanumeric() && c != '_').next().unwrap();
            assert!(seen.insert(name.to_string()), "`{name}` is declared twice:
{wgsl}");
        }
        // And all eight stores still happen: four texels and four coefficients.
        assert_eq!(wgsl.matches("let g").count(), 8, "got:
{wgsl}");
    }

    /// GATHER4 writes SIX registers from one footprint: four gathered texels at `dest + 0..3`
    /// and four F16 bilinear coefficients at `dest + 4..5`. The COEFFICIENT ORDER is the
    /// reverse of the gather order, which is what the only consumer in the corpus asks for -
    /// it reduces the two with `dot(gathered.wzyx, coeff.xyzw)` - so `coeff[k]` must be the
    /// weight of `gathered[3 - k]`. Getting that backwards mis-weights the filter inside a
    /// texel with nothing to say so, which is why it is pinned here.
    #[test]
    fn gather4_writes_its_texels_then_its_reversed_coefficients() {
        let d = Operand::plain(Bank::Temp, 0, 0);
        let coord = Operand::plain(Bank::PrimaryAttr, 8, 2);
        let wgsl = emit_fragment(&shader(vec![instr(
            Op::TexGather { unit: 3, coords: 2, coord_half: false },
            Some(d),
            vec![coord],
        )]))
        .unwrap();
        assert!(wgsl.contains("textureGather(0u, t3, s3,"), "got:\n{wgsl}");
        // The fractional position of the same 2x2 a bilinear filter would take.
        assert!(wgsl.contains("fract("), "the coefficients need the bilinear fractions:\n{wgsl}");
        assert!(wgsl.contains("textureDimensions(t3, 0u)"), "got:\n{wgsl}");
        for (c, lane) in ["x", "y", "z", "w"].iter().enumerate() {
            assert!(
                wgsl.contains(&format!("r[{c}] = bitcast<u32>(_g0.{lane});")),
                "gathered texel {c} must land at r[{c}]:\n{wgsl}"
            );
        }
        // coeff[0] weights gathered[3] = the (x0,y0) texel, coeff[3] weights gathered[0].
        assert!(wgsl.contains("(1.0 - _gf0.x) * (1.0 - _gf0.y)"), "got:\n{wgsl}");
        assert!(wgsl.contains("(1.0 - _gf0.x) * _gf0.y"), "got:\n{wgsl}");
        // Four F16 coefficients occupy TWO registers past the four gathered texels.
        assert!(wgsl.contains("r[4] = (r[4] &"), "coefficients start at dest + 4:\n{wgsl}");
        assert!(wgsl.contains("r[5] = (r[5] &"), "coefficients span two registers:\n{wgsl}");
        assert!(!wgsl.contains("r[6]"), "a gather writes six registers, not more:\n{wgsl}");
    }

    /// VTSTMSK writes ONE VALUE PER CHANNEL rather than reducing to a predicate bit, and the
    /// numeric mask is `1.0` where the test holds.
    #[test]
    fn vtstmsk_writes_a_numeric_mask_on_every_channel() {
        let d = Operand::plain(Bank::Internal, 0, 0);
        let a = Operand::plain(Bank::Temp, 0, 0);
        let b = Operand::plain(Bank::Internal, 0, 0);
        // The real program broadcasts a reference value into i0 first, and the fragment
        // internal-read guard requires it: an unwritten internal lane is unmodelled input.
        let wgsl = emit_fragment(&shader(vec![
            instr(Op::Mov, Some(d), vec![Operand::plain(Bank::PrimaryAttr, 4, 2)]),
            instr(Op::TestMask { alu: TestAlu::Sub, cmp: TestCmp::Gt }, Some(d), vec![a, b]),
        ]))
        .unwrap();
        assert!(!wgsl.contains("p[0] ="), "a mask writes no predicate:\n{wgsl}");
        for c in 0..4u32 {
            assert!(
                wgsl.contains(&format!("i[{c}] = bitcast<u32>(g{c});")),
                "channel {c} must be written:\n{wgsl}"
            );
        }
        assert!(wgsl.contains("select(0.0, 1.0,"), "the mask is numeric:\n{wgsl}");
        // The internal registers are four F32 lanes whatever the precision, so a four-channel
        // mask lands on i[0..3] rather than on two packed registers.
        assert!(wgsl.contains("i[3] ="), "the fourth channel needs a fourth lane:\n{wgsl}");
    }
    #[test]
    fn emits_scalarised_mul_over_channels() {
        // o[base..] = r[..] * sa[..], full mask -> 4 statements, one per channel.
        let d = Operand::plain(Bank::Output, 0, 1);
        let a = Operand::plain(Bank::Temp, 4, 0);
        let b = Operand::plain(Bank::SecondaryAttr, 8, 3);
        let wgsl = emit_fragment(&shader(vec![instr(Op::Mul, Some(d), vec![a, b])])).unwrap();
        assert!(wgsl.contains(&st("o", 0, &format!("({} * {})", rd("r", 4), rd("sa", 8)))), "got:\n{wgsl}");
        assert!(wgsl.contains(&st("o", 3, &format!("({} * {})", rd("r", 7), rd("sa", 11)))), "got:\n{wgsl}");
    }

    /// The 8-bit combiner emits its whole term structure, reads its sources as BYTES, and
    /// writes one byte back without disturbing the other three.
    ///
    /// The byte view is the part worth a test of its own: the flag this instruction writes in
    /// the real corpus is the bit pattern 0x00000001, which read as an f32 is a denormal that
    /// compares equal to zero. An F32 read here would emit WGSL that validates, runs, and
    /// silently disables the alpha test the flag gates.
    #[test]
    fn emits_the_eight_bit_combiner_as_bytes() {
        let d = Operand::plain(Bank::Temp, 0, 0);
        let a = Operand::plain(Bank::SecondaryAttr, 9, 3);
        let b = Operand::plain(Bank::Temp, 4, 0);
        let mut ins = instr(
            Op::Sop2 {
                color: SopOp::Add,
                alpha: SopOp::Add,
                f1: SopFactor::Zero,
                f1_complement: true,
                f2: SopFactor::Zero,
                f2_complement: false,
            },
            Some(d),
            vec![a, b],
        );
        ins.write_mask = [true, false, false, false];
        let wgsl = emit_fragment(&shader(vec![ins])).unwrap();
        assert!(
            wgsl.contains("(1.0 - 0.0) * unpack4x8unorm(sa[9])[0]"),
            "the complemented zero coefficient multiplies src1, making the term a copy:\n{wgsl}"
        );
        assert!(
            wgsl.contains("0.0 * unpack4x8unorm(r[4])[0]"),
            "the second term is scaled to nothing but is still READ:\n{wgsl}"
        );
        assert!(
            wgsl.contains("r[0] = (r[0] & 0xffffff00u) |"),
            "channel 0 is byte 0, and the other three bytes survive the write:\n{wgsl}"
        );
        assert!(!wgsl.contains("r[1] ="), "one register, not four:\n{wgsl}");
    }

    /// The 8-bit TEST reads its operands as bytes too, and from the register the combiner
    /// wrote. Same reason as above: an F32 read of the flag register compares a denormal
    /// against zero and reports equal.
    #[test]
    fn emits_the_eight_bit_test_as_bytes() {
        let a = Operand::plain(Bank::Temp, 0, 0);
        let b = Operand::plain(Bank::SecondaryAttr, 7, 3);
        let ins = instr(
            Op::Test {
                alu: TestAlu::Fx8Sub,
                cmp: TestCmp::Eq,
                reduce: TestReduce::Channel(0),
                pdst: 1,
                write_back: false,
            },
            None,
            vec![a, b],
        );
        let wgsl = emit_fragment(&shader(vec![ins])).unwrap();
        assert!(
            wgsl.contains("p[1] = ((unpack4x8unorm(r[0])[0] - unpack4x8unorm(sa[7])[0]) == 0.0)"),
            "got:\n{wgsl}"
        );
    }

    #[test]
    fn honours_partial_write_mask() {
        // Registers far enough apart that the destination cannot alias a source - this is a
        // test about the write MASK, and an aliasing destination would also (correctly) stage
        // the stores through temporaries, which is a different property with its own test.
        let d = Operand::plain(Bank::Temp, 2, 0);
        let a = Operand::plain(Bank::Temp, 8, 0);
        let b = Operand::plain(Bank::Temp, 12, 0);
        let mut ins = instr(Op::Add, Some(d), vec![a, b]);
        ins.write_mask = [true, false, true, false];
        let wgsl = emit_fragment(&shader(vec![ins])).unwrap();
        assert!(wgsl.contains(&st("r", 2, &format!("({} + {})", rd("r", 8), rd("r", 12)))), "got:\n{wgsl}");
        assert!(wgsl.contains(&st("r", 4, &format!("({} + {})", rd("r", 10), rd("r", 14)))), "got:\n{wgsl}"); // channel 2
        assert!(!wgsl.contains("r[3] ="), "channel 1 masked out:\n{wgsl}");
        assert!(!wgsl.contains("r[5] ="), "channel 3 masked out:\n{wgsl}");
    }

    #[test]
    fn swizzle_lanes_and_constants_and_mods() {
        let d = Operand::plain(Bank::Output, 0, 1);
        let mut a = Operand::plain(Bank::Temp, 10, 0);
        a.swizzle = [1, 1, 5, 3]; // y, y, const 1.0, w
        a.neg = true;
        let b = Operand::plain(Bank::Temp, 20, 0);
        let mut ins = instr(Op::Add, Some(d), vec![a, b]);
        ins.write_mask = [true, false, true, false];
        let wgsl = emit_fragment(&shader(vec![ins])).unwrap();
        assert!(wgsl.contains(&st("o", 0, &format!("((-{}) + {})", rd("r", 11), rd("r", 20)))), "got:\n{wgsl}"); // ch0: -r[10+1]
        assert!(wgsl.contains(&st("o", 2, &format!("((-1.0) + {})", rd("r", 22)))), "got:\n{wgsl}"); // ch2: const 1.0
    }

    #[test]
    fn emits_min_max_frc_dpdx_and_dot() {
        // Non-aliasing registers, for the same reason as `honours_partial_write_mask`.
        let d = || Some(Operand::plain(Bank::Temp, 0, 0));
        let a = Operand::plain(Bank::Temp, 8, 0);
        let b = Operand::plain(Bank::Temp, 12, 0);
        let mn = emit_fragment(&shader(vec![instr(Op::Min, d(), vec![a, b])])).unwrap();
        assert!(mn.contains(&st("r", 0, &format!("min({}, {})", rd("r", 8), rd("r", 12)))), "got:\n{mn}");
        let fr = emit_fragment(&shader(vec![instr(Op::Frc, d(), vec![a])])).unwrap();
        assert!(fr.contains(&st("r", 0, &format!("fract({})", rd("r", 8)))), "got:\n{fr}");
        let dx = emit_fragment(&shader(vec![instr(Op::Dsx, d(), vec![a])])).unwrap();
        assert!(dx.contains(&st("r", 0, &format!("dpdx({})", rd("r", 8)))), "got:\n{dx}");
        let dt = emit_fragment(&shader(vec![instr(Op::Dot { components: 4 }, d(), vec![a, b])])).unwrap();
        assert!(dt.contains(&st("r", 0, &format!("({})", (0..4).map(|c| format!("{} * {}", rd("r", 8 + c), rd("r", 12 + c))).collect::<Vec<_>>().join(" + ")))), "got:\n{dt}");
    }

    /// An instruction whose destination shares registers with a source must read every source
    /// BEFORE it writes any channel, because that is what the hardware does.
    ///
    /// The shape here is the one that was miscompiling a title's display composite: the last
    /// step of a Reinhard tonemap, `dest.xyz = src.zzz * src.xxx`, with `dest` and both sources
    /// the same register pair. Emitted as three independent statements, channel x overwrote the
    /// `1/(1+L)` term before channels y and z read it, so green and blue picked up an extra
    /// factor of the luminance while red stayed correct - a frame too dark and too red, with a
    /// perfectly correct-looking shader.
    #[test]
    fn an_instruction_whose_dest_aliases_a_source_reads_before_it_writes() {
        let d = Operand::plain(Bank::Temp, 2, 0);
        let mut zzz = Operand::plain(Bank::Temp, 2, 0);
        zzz.swizzle = [2, 2, 2, 2];
        let mut xxx = Operand::plain(Bank::Temp, 2, 0);
        xxx.swizzle = [0, 0, 0, 0];
        let mut ins = instr(Op::Mul, Some(d), vec![zzz, xxx]);
        ins.write_mask = [true, true, true, false];
        let wgsl = emit_fragment(&shader(vec![ins])).unwrap();
        // Every read is a `let` ahead of every store, so no store can be observed by a later
        // channel of the same instruction.
        let first_store = wgsl.find("r[2] = ").expect("a store");
        for c in 0..3 {
            let read = wgsl.find(&format!("let g{c} = ")).unwrap_or_else(|| panic!("channel {c} staged:\n{wgsl}"));
            assert!(read < first_store, "channel {c} is read after a store:\n{wgsl}");
        }
        // And the temporaries are block-scoped, so two such instructions cannot collide.
        assert!(wgsl.contains("  {\n"), "staged stores are wrapped in a block:\n{wgsl}");
    }

    /// The complement: an ordinary instruction that cannot alias keeps the direct, readable
    /// one-statement-per-channel form. Staging everything would be correct too, and would make
    /// every emitted module harder to read for no gain.
    #[test]
    fn an_instruction_that_cannot_alias_stores_directly() {
        let d = Operand::plain(Bank::Temp, 0, 0);
        let a = Operand::plain(Bank::Temp, 8, 0);
        let b = Operand::plain(Bank::Temp, 12, 0);
        let wgsl = emit_fragment(&shader(vec![instr(Op::Mul, Some(d), vec![a, b])])).unwrap();
        assert!(!wgsl.contains("let g0 ="), "no staging needed:\n{wgsl}");
    }

    #[test]
    fn unsupported_op_hard_fails_naming_it() {
        let bad = instr(Op::Todo("tex"), None, vec![]);
        let err = emit_fragment(&shader(vec![bad])).unwrap_err();
        match &err {
            EmitError::UnsupportedOp { op, .. } => assert_eq!(*op, "tex"),
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
        assert!(err.to_string().contains("tex"));
    }

    /// The one established GLOBAL register, on the REAL captured word. The whole instruction
    /// decodes to `p0 = ((GLOBAL[16] & 1) != 0)` and emits as the per-fragment facing bit, so
    /// the predicated move after it selects the flipped normal exactly on the faces the guest
    /// shades two-sided. The polarity here is the one that renders the car liveries; see
    /// [`global_u32_expr`] for why it is measured rather than reasoned.
    #[test]
    fn global16_bit0_is_the_facing_bit() {
        // frag_82d27fb0 #2 (and byte-identically in frag_82ed89c0); the `skipinv` variant
        // 0x488b... in frag_82d1bd50 differs only in bit 55 and must translate the same.
        for raw in [0x480b_0281_600c_2801u64, 0x488b_0281_600c_2801u64] {
            let ins = crate::usse::decode(raw);
            assert!(ins.blocked.is_none(), "{raw:#018x} must decode: {:?}", ins.blocked);
            assert!(
                matches!(ins.op, Op::Test { alu: TestAlu::BitAnd, cmp: TestCmp::Ne, reduce: crate::ir::TestReduce::Channel(0), pdst: 0, write_back: false }),
                "{raw:#018x} decoded as {:?}",
                ins.op
            );
            assert_eq!((ins.srcs[0].bank, ins.srcs[0].index), (Bank::Global, 16));
            assert_eq!((ins.srcs[1].bank, ins.srcs[1].index), (Bank::Immediate, 1));

            let wgsl = emit_fragment(&shader(vec![ins])).unwrap();
            assert!(
                wgsl.contains("p[0] = ((select(0u, 1u, gxp_front_facing) & 1u) != 0u);"),
                "got:\n{wgsl}"
            );
        }
    }

    /// The facing reading is scoped to the one register it was derived from, in the one stage
    /// that has a facing bit, read by the one operation that reads it as raw bits. Everything
    /// else hard-fails naming the GLOBAL index rather than inheriting a value.
    #[test]
    fn any_other_global_read_hard_fails_naming_its_index() {
        let test_of = |global: u8| {
            instr(
                Op::Test { alu: TestAlu::BitAnd, cmp: TestCmp::Ne, reduce: crate::ir::TestReduce::Channel(0), pdst: 0, write_back: false },
                None,
                vec![Operand::plain(Bank::Global, global, 1), Operand::plain(Bank::Immediate, 1, 2)],
            )
        };
        // A different GLOBAL index, same instruction shape.
        match emit_fragment(&shader(vec![test_of(17)])).unwrap_err() {
            EmitError::UnmodeledGlobal { global, .. } => assert_eq!(global, 17),
            other => panic!("expected UnmodeledGlobal, got {other:?}"),
        }
        // GLOBAL[16] in a VERTEX program: there is no per-fragment facing bit there.
        let vsh = Shader { kind: ProgramKind::Vertex, instrs: vec![test_of(16)] };
        match emit_body(&vsh).unwrap_err() {
            EmitError::UnmodeledGlobal { global, .. } => assert_eq!(global, 16),
            other => panic!("expected UnmodeledGlobal, got {other:?}"),
        }
        // GLOBAL[16] read by an ordinary float op rather than a bitwise test.
        let mov = instr(Op::Mov, Some(Operand::plain(Bank::Temp, 0, 0)), vec![Operand::plain(Bank::Global, 16, 1)]);
        match emit_fragment(&shader(vec![mov])).unwrap_err() {
            EmitError::UnmodeledGlobal { global, .. } => assert_eq!(global, 16),
            other => panic!("expected UnmodeledGlobal, got {other:?}"),
        }
    }

    #[test]
    fn blocked_instruction_hard_fails_naming_reason() {
        let mut ins = instr(Op::Mul, Some(Operand::plain(Bank::Temp, 0, 0)),
            vec![Operand::plain(Bank::Temp, 2, 0), Operand::plain(Bank::Temp, 4, 0)]);
        ins.blocked = Some("predicated instruction not yet wired");
        ins.raw = 0xdead_beef;
        let err = emit_fragment(&shader(vec![ins])).unwrap_err();
        assert!(matches!(err, EmitError::Blocked { .. }));
        assert!(err.to_string().contains("predicated"));
    }

    #[test]
    fn dot_reading_undefined_internal_hard_fails() {
        // A dot whose op2 is internal register i0, with nothing writing i0 first, must
        // hard-fail rather than translate an unmodeled iterator pre-load.
        let d = Operand::plain(Bank::Temp, 0, 0);
        let a = Operand::plain(Bank::PrimaryAttr, 4, 2);
        let i = Operand::plain(Bank::Internal, 0, 0); // i0, xxxx
        let err = emit_fragment(&shader(vec![instr(Op::Dot { components: 4 }, Some(d), vec![a, i])]))
            .unwrap_err();
        match err {
            EmitError::UndefinedInternal { lane, .. } => assert_eq!(lane, 0),
            other => panic!("expected UndefinedInternal, got {other:?}"),
        }
    }

    #[test]
    fn vertex_reading_undefined_internal_emits_as_zero_scratch() {
        // The SAME shape that hard-fails for a fragment program (reading an unwritten internal
        // lane) is allowed for a VERTEX program: vertex internal registers are zero-initialised
        // scratch (no iterator/PDS preload), so the read is a defined 0.0, not unmodeled input.
        let d = Operand::plain(Bank::Output, 8, 1);
        let mut src = Operand::plain(Bank::Internal, 0, 0);
        src.swizzle = [2, 3, 2, 3]; // reads i[2], i[3] - neither written in-stream
        let mut ins = instr(Op::Mov, Some(d), vec![src]);
        ins.write_mask = [true, true, false, false];
        let vsh = Shader { kind: ProgramKind::Vertex, instrs: vec![ins] };
        let wgsl = emit_fragment(&vsh).expect("vertex unwritten-internal read must emit");
        assert!(wgsl.contains(&st("o", 8, &rd("i", 2))), "got:\n{wgsl}");
        assert!(wgsl.contains(&st("o", 9, &rd("i", 3))), "got:\n{wgsl}");
    }

    #[test]
    fn dot_reading_in_stream_defined_internal_emits() {
        // Write i0 (internal dest) then dot from it: the read is defined, so it emits.
        let mut wr = instr(
            Op::Add,
            Some(Operand::plain(Bank::Internal, 0, 0)),
            vec![Operand::plain(Bank::PrimaryAttr, 2, 2), Operand::plain(Bank::PrimaryAttr, 4, 2)],
        );
        wr.write_mask = [true, true, true, true];
        let dt = instr(
            Op::Dot { components: 4 },
            Some(Operand::plain(Bank::Temp, 0, 0)),
            vec![Operand::plain(Bank::Temp, 8, 0), Operand::plain(Bank::Internal, 0, 0)],
        );
        let wgsl = emit_fragment(&shader(vec![wr, dt])).unwrap();
        assert!(wgsl.contains("i[0] ="), "internal write emitted:\n{wgsl}");
        assert!(wgsl.contains(&format!("{} * {}", rd("r", 8), rd("i", 0))), "dot reads i[0]:\n{wgsl}");
    }

    #[test]
    fn predicated_instruction_wraps_in_if() {
        // An instruction predicated on p1 wraps its writes in `if (p[1]) { ... }`; a negated
        // predicate uses `if (!p[n])`.
        let d = Operand::plain(Bank::Output, 0, 1);
        let a = Operand::plain(Bank::Temp, 2, 0);
        let b = Operand::plain(Bank::Temp, 4, 0);
        let mut ins = instr(Op::Add, Some(d), vec![a, b]);
        ins.pred = Predicate::IfP(1);
        ins.write_mask = [true, false, false, false];
        let wgsl = emit_fragment(&shader(vec![ins])).unwrap();
        assert!(wgsl.contains("if (p[1]) {"), "got:\n{wgsl}");
        assert!(wgsl.contains(&st("o", 0, &format!("({} + {})", rd("r", 2), rd("r", 4)))), "got:\n{wgsl}");

        let mut neg = instr(Op::Add, Some(Operand::plain(Bank::Output, 0, 1)),
            vec![Operand::plain(Bank::Temp, 2, 0), Operand::plain(Bank::Temp, 4, 0)]);
        neg.pred = Predicate::IfNotP(0);
        let w2 = emit_fragment(&shader(vec![neg])).unwrap();
        assert!(w2.contains("if (!p[0]) {"), "got:\n{w2}");
    }

    #[test]
    fn emits_conditional_move_as_select() {
        use crate::ir::CompareMethod;
        // o[0] = select(r[6], r[2], (r[4] < 0.0)) for the first channel; src order is
        // [src1(true), src2(false), src0(test)].
        let d = Operand::plain(Bank::Output, 0, 1);
        let s1 = Operand::plain(Bank::Temp, 2, 0);
        let s2 = Operand::plain(Bank::Temp, 6, 0);
        let s0 = Operand::plain(Bank::Temp, 4, 0);
        let mut ins = instr(Op::Cmov { test: CompareMethod::LtZero }, Some(d), vec![s1, s2, s0]);
        ins.write_mask = [true, false, false, false];
        let wgsl = emit_fragment(&shader(vec![ins])).unwrap();
        assert!(wgsl.contains(&st("o", 0, &format!("select({}, {}, ({} < 0.0))", rd("r", 6), rd("r", 2), rd("r", 4)))), "got:\n{wgsl}");
    }

    #[test]
    fn constant_operand_selects_bank_per_channel() {
        // A constant operand's swizzle still selects which hardware constant BANK each
        // channel reads (spec A.7): in F32 mode a Y selector reads bank 1 and every other
        // selector reads bank 0. CNST6 index 2 is 1.0 in bank 0 but 0.0 in bank 1.
        let d = Operand::plain(Bank::Output, 0, 1);
        let a = Operand::plain(Bank::Temp, 4, 0);
        let mut k = Operand::plain(Bank::Constant, 2, 0);
        k.neg = true;
        let wgsl = emit_fragment(&shader(vec![instr(Op::Mul, Some(d), vec![a, k])])).unwrap();
        assert!(
            wgsl.contains(&st("o", 0, &format!("({} * (-bitcast<f32>(0x3f800000u)))", rd("r", 4)))),
            "channel 0 (X selector) reads bank 0:
{wgsl}"
        );
        assert!(
            wgsl.contains(&st("o", 1, &format!("({} * (-bitcast<f32>(0x00000000u)))", rd("r", 5)))),
            "channel 1 (Y selector) reads bank 1:
{wgsl}"
        );
    }

    #[test]
    fn f16_instruction_addresses_half_lanes_of_a_register_pair() {
        // The register file is 32-bit registers either way; an F16 operand's four channels
        // are the two halves of registers base and base+1, so channel 2 is the LOW half of
        // base+1 - not register base+2 as the F32 view would have it.
        let d = Operand::plain(Bank::Temp, 0, 0);
        let a = Operand::plain(Bank::SecondaryAttr, 6, 3);
        let b = Operand::plain(Bank::PrimaryAttr, 4, 2);
        let mut ins = instr(Op::Mul, Some(d), vec![a, b]);
        ins.half_precision = true;
        ins.write_mask = [true, true, true, false];
        let wgsl = emit_fragment(&shader(vec![ins])).unwrap();
        assert!(wgsl.contains("unpack2x16float(sa[6])[0] * unpack2x16float(pa[4])[0]"), "got:
{wgsl}");
        assert!(wgsl.contains("unpack2x16float(sa[6])[1] * unpack2x16float(pa[4])[1]"), "got:
{wgsl}");
        assert!(wgsl.contains("unpack2x16float(sa[7])[0] * unpack2x16float(pa[5])[0]"), "got:
{wgsl}");
        // Channel 0 writes the LOW half of r[0], preserving the high half.
        assert!(wgsl.contains("r[0] = (r[0] & 0xffff0000u) |"), "got:
{wgsl}");
        // Channel 1 writes the HIGH half of the same register.
        assert!(wgsl.contains("r[0] = (r[0] & 0x0000ffffu) |"), "got:
{wgsl}");
        // Channel 2 moves on to r[1]; channel 3 is masked out.
        assert!(wgsl.contains("r[1] = (r[1] & 0xffff0000u) |"), "got:
{wgsl}");
        assert!(!wgsl.contains("r[1] = (r[1] & 0x0000ffffu) |"), "channel 3 masked:
{wgsl}");
        // No F32-width access anywhere in an all-F16 instruction.
        assert!(!wgsl.contains("bitcast<f32>(sa["), "got:
{wgsl}");
    }

    #[test]
    fn f16_constant_reads_the_swizzle_selected_half_bank() {
        // In F16 mode the four constant banks are chosen by the channel's swizzle selector.
        // CNST6 index 15 is 0.0 in bank 0 but 1.0h (0x3c00) in bank 1.
        let d = Operand::plain(Bank::Temp, 0, 0);
        let a = Operand::plain(Bank::Temp, 2, 0);
        let k = Operand::plain(Bank::Constant, 15, 0);
        let mut ins = instr(Op::Mul, Some(d), vec![a, k]);
        ins.half_precision = true;
        ins.write_mask = [true, true, false, false];
        let wgsl = emit_fragment(&shader(vec![ins])).unwrap();
        assert!(wgsl.contains("unpack2x16float(0x00000000u)[0]"), "channel 0 = bank 0:
{wgsl}");
        assert!(wgsl.contains("unpack2x16float(0x00003c00u)[0]"), "channel 1 = bank 1:
{wgsl}");
    }

    #[test]
    fn empty_declines() {
        assert_eq!(emit_fragment(&shader(vec![])).unwrap_err(), EmitError::Empty);
    }

    /// A branch that is TAKEN when its predicate holds skips the words after it, so the range
    /// it skips runs when the predicate does NOT hold - the emitted `if` must carry the NEGATED
    /// condition. Getting this backwards runs exactly the wrong arm, which is why it is pinned.
    fn mov(dest: u8) -> Instr {
        instr(Op::Mov, Some(Operand::plain(Bank::Temp, dest, 0)), vec![Operand::plain(Bank::Temp, 100, 0)])
    }

    fn branch(rel: i32, pred: Predicate) -> Instr {
        let mut b = instr(Op::Branch { rel }, None, vec![]);
        b.pred = pred;
        b.write_mask = [false; 4];
        b
    }

    #[test]
    fn forward_branch_becomes_an_if_on_the_negated_condition() {
        // 0: br if p0 -> 3      (skips instructions 1..2)
        // 1: mov r0
        // 2: mov r2
        // 3: mov r4
        // Destination bases are 4 apart so each `mov`'s four written lanes are disjoint and a
        // lane names exactly one instruction.
        let wgsl = emit_fragment(&shader(vec![branch(3, Predicate::IfP(0)), mov(0), mov(4), mov(8)]))
            .unwrap();
        assert!(wgsl.contains("if (!p[0]) {"), "got:\n{wgsl}");
        let inside = wgsl.split("if (!p[0]) {").nth(1).unwrap();
        let (guarded, after) = inside.split_once("}").unwrap();
        assert!(guarded.contains("r[0] ="), "the skipped range is the guarded body:\n{wgsl}");
        assert!(guarded.contains("r[4] ="), "the skipped range is the guarded body:\n{wgsl}");
        assert!(!guarded.contains("r[8] ="), "the branch target is NOT guarded:\n{wgsl}");
        assert!(after.contains("r[8] ="), "the branch target is emitted after:\n{wgsl}");
    }

    /// A branch predicated on p0 being CLEAR guards its range on p0 being SET.
    #[test]
    fn negated_predicate_branch_inverts_the_same_way() {
        let wgsl =
            emit_fragment(&shader(vec![branch(2, Predicate::IfNotP(1)), mov(0), mov(2)])).unwrap();
        assert!(wgsl.contains("if (p[1]) {"), "got:\n{wgsl}");
    }

    /// An UNCONDITIONAL forward branch always skips its range, so that range is unreachable and
    /// emitting nothing for it is exact - not a dropped instruction.
    #[test]
    fn unconditional_forward_branch_drops_the_unreachable_range() {
        let wgsl =
            emit_fragment(&shader(vec![branch(2, Predicate::Always), mov(0), mov(2)])).unwrap();
        assert!(!wgsl.contains("r[0] ="), "skipped range must not be emitted:\n{wgsl}");
        assert!(wgsl.contains("r[2] ="), "the target must be emitted:\n{wgsl}");
    }

    /// A BACKWARD branch is a loop, and its CONDITION is the condition to go round AGAIN - so
    /// the WGSL body breaks on the negation. Getting that polarity backwards runs the loop
    /// exactly once or never, which is the plausible-looking wrong picture rather than a
    /// failure, so it is pinned.
    #[test]
    fn backward_branch_becomes_a_loop_breaking_on_the_negated_condition() {
        // 0: mov r0
        // 1: mov r2
        // 2: br if p0 -> 0      (go round again while p0)
        let wgsl =
            emit_fragment(&shader(vec![mov(0), mov(2), branch(-2, Predicate::IfP(0))])).unwrap();
        assert!(wgsl.contains("loop {"), "got:\n{wgsl}");
        let inside = wgsl.split("loop {").nth(1).unwrap();
        assert!(inside.contains("r[0] ="), "the body is inside the loop:\n{wgsl}");
        assert!(inside.contains("r[2] ="), "the body is inside the loop:\n{wgsl}");
        assert!(inside.contains("if (!p[0]) { break; }"), "got:\n{wgsl}");
    }

    /// The shape a compiler actually emits: a guarded exit at the top of the body and an
    /// UNCONDITIONAL back edge at the bottom. The exit branch becomes a `break` on the
    /// condition under which it is TAKEN - the opposite polarity from a forward skip, because
    /// what it guards is leaving the loop rather than a range that runs when it is not taken.
    #[test]
    fn a_loop_exit_branch_becomes_a_break_on_the_taken_condition() {
        // 0: br if p0 -> 4      (leave)
        // 1: mov r0
        // 2: mov r2
        // 3: br       -> 0      (go round again)
        // 4: mov r4
        let wgsl = emit_fragment(&shader(vec![
            branch(4, Predicate::IfP(0)),
            mov(0),
            mov(2),
            branch(-3, Predicate::Always),
            mov(4),
        ]))
        .unwrap();
        let (before, inside) = wgsl.split_once("loop {").expect(&format!("got:\n{wgsl}"));
        assert!(!before.contains("r[0] ="), "the body belongs to the loop:\n{wgsl}");
        assert!(inside.contains("if (p[0]) { break; }"), "got:\n{wgsl}");
        assert!(inside.contains("r[0] ="), "got:\n{wgsl}");
        // An unconditional back edge repeats by falling off the end of the WGSL body, so there
        // is no trailing break to write.
        assert!(!inside.contains("if (!p[0]) { break; }"), "got:\n{wgsl}");
        let after = inside.rsplit_once("}").unwrap().1;
        assert!(after.contains("r[4] ="), "the instruction after the loop follows it:\n{wgsl}");
    }

    /// A second backward branch inside a loop body is a second loop sharing that body, which one
    /// `loop` cannot express - so it hard-fails rather than emitting one of the two.
    #[test]
    fn a_second_backward_branch_inside_a_loop_hard_fails() {
        // 0: mov r0
        // 1: br if p0 -> 0      (an inner back edge to the same head)
        // 2: mov r2
        // 3: br if p1 -> 0
        let err = emit_fragment(&shader(vec![
            mov(0),
            branch(-1, Predicate::IfP(0)),
            mov(2),
            branch(-3, Predicate::IfP(1)),
        ]))
        .unwrap_err();
        match err {
            EmitError::Blocked { reason, .. } => {
                assert!(reason.contains("second backward branch"), "{reason}");
            }
            other => panic!("expected Blocked, got {other:?}"),
        }
    }

    /// A jump into the MIDDLE of a loop body is a second entry, and a `loop` has one - so the
    /// region is not a loop and must not be emitted as one.
    #[test]
    fn a_branch_into_a_loop_body_hard_fails() {
        // 0: mov r0             <- loop head
        // 1: mov r2
        // 2: br       -> 0      (the back edge)
        // 3: br if p0 -> 1      (a second entry, into the middle)
        let err = emit_fragment(&shader(vec![
            mov(0),
            mov(2),
            branch(-2, Predicate::Always),
            branch(-2, Predicate::IfP(0)),
        ]))
        .unwrap_err();
        match err {
            EmitError::Blocked { reason, .. } => {
                assert!(reason.contains("into the middle of a loop body"), "{reason}");
            }
            other => panic!("expected Blocked, got {other:?}"),
        }
    }

    /// A branch out of a loop body to somewhere that is NOT the instruction after the back edge
    /// is not a `break`, and there is no other statement that means it.
    #[test]
    fn a_loop_body_branch_past_the_exit_hard_fails() {
        // 0: br if p0 -> 4      (past the loop's own exit at 3)
        // 1: mov r0
        // 2: br       -> 0
        // 3: mov r2
        // 4: mov r4
        let err = emit_fragment(&shader(vec![
            branch(4, Predicate::IfP(0)),
            mov(0),
            branch(-2, Predicate::Always),
            mov(2),
            mov(4),
        ]))
        .unwrap_err();
        match err {
            EmitError::Blocked { reason, .. } => {
                assert!(
                    reason.contains("neither inside the loop nor its exit"),
                    "{reason}"
                );
            }
            other => panic!("expected Blocked, got {other:?}"),
        }
    }

    /// A branch whose target leaves the block an enclosing branch opened is irreducible - no
    /// nest of `if`s expresses it - so it blocks rather than being silently clamped.
    #[test]
    fn branch_out_of_an_enclosing_block_hard_fails() {
        // 0: br if p0 -> 3 (opens the block [1,3))
        // 1: br if p1 -> 4 (would leave it)
        let err = emit_fragment(&shader(vec![
            branch(3, Predicate::IfP(0)),
            branch(3, Predicate::IfP(1)),
            mov(0),
            mov(2),
        ]))
        .unwrap_err();
        match err {
            EmitError::Blocked { reason, index, .. } => {
                assert_eq!(index, 1);
                assert!(reason.contains("out of its enclosing block"), "{reason}");
            }
            other => panic!("expected Blocked, got {other:?}"),
        }
    }

    /// A conditional skip whose range ENDS in an unconditional jump past the target is an
    /// if/else: the trailing jump is the then-arm's hop over the else-arm, not part of the
    /// guarded body. Without recovering it the inner branch reads as a jump out of its
    /// enclosing block and the whole pair falls back - this is the exact shape of both of a
    /// retail title's menu fragment programs (`br #6 -> 10`, `br #9 -> 13 of 13`).
    #[test]
    fn conditional_skip_ending_in_an_unconditional_jump_is_an_if_else() {
        // 0: br if p0 -> 4 ; 1: mov r0 ; 2: mov r4 ; 3: br -> 6 ; 4: mov r8 ; 5: mov r12
        let wgsl = emit_fragment(&shader(vec![
            branch(4, Predicate::IfP(0)),
            mov(0),
            mov(4),
            branch(3, Predicate::Always),
            mov(8),
            mov(12),
        ]))
        .unwrap();
        assert!(wgsl.contains("} else {"), "an else arm must be emitted:\n{wgsl}");
        let (then_arm, rest) = wgsl.split_once("} else {").unwrap();
        assert!(then_arm.contains("r[0] =") && then_arm.contains("r[4] ="), "then arm:\n{wgsl}");
        assert!(!then_arm.contains("r[8] ="), "else arm must not be in the then arm:\n{wgsl}");
        assert!(rest.contains("r[8] =") && rest.contains("r[12] ="), "else arm:\n{wgsl}");
    }

    /// Nested skips nest as blocks, and the inner one's range stays inside the outer one's.
    #[test]
    fn nested_forward_branches_nest() {
        // 0: br if p0 -> 4 ; 1: br if p1 -> 3 ; 2: mov r0 ; 3: mov r2 ; 4: mov r4
        let wgsl = emit_fragment(&shader(vec![
            branch(4, Predicate::IfP(0)),
            branch(2, Predicate::IfP(1)),
            mov(0),
            mov(2),
            mov(4),
        ]))
        .unwrap();
        let outer = wgsl.split("if (!p[0]) {").nth(1).unwrap();
        let inner = outer.split("if (!p[1]) {").nth(1).unwrap();
        assert!(inner.starts_with(|_c: char| true) && inner.contains("r[0] ="), "got:\n{wgsl}");
    }
}
