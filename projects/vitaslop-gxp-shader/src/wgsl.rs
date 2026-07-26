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
use crate::ir::{Bank, BitwiseKind, CompareMethod, Instr, Op, Operand, Predicate, Shader, TestAlu, TestCmp, TestReduce, TexLod};

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
        Bank::Constant | Bank::Immediate | Bank::Global | Bank::Raw(_) => return None,
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

/// The WGSL rvalue reading channel-selector `sel` (0..3) of the register file at `base`.
fn read_lane(prefix: &str, base: u32, sel: u32, prec: Prec) -> String {
    match prec {
        Prec::F32 => format!("bitcast<f32>({prefix}[{}])", base + sel),
        Prec::F16 => format!("unpack2x16float({prefix}[{}])[{}]", base + (sel >> 1), sel & 1),
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

/// Emit the store of `expr` (an f32 rvalue) into destination channel `c`. An F32 channel
/// overwrites a whole register; an F16 channel is a read-modify-write of one half, so the
/// paired channel keeps its value - exactly how the hardware packs two halves per register.
fn store_channel(body: &mut String, op: &Operand, c: usize, expr: &str, prec: Prec) -> Option<()> {
    let prefix = bank_prefix(op.bank)?;
    match prec {
        Prec::F32 => {
            writeln!(body, "  {prefix}[{}] = bitcast<u32>({expr});", op.index as u32 + c as u32).ok();
        }
        Prec::F16 => {
            let reg = op.index as u32 + (c as u32 >> 1);
            if c & 1 == 0 {
                writeln!(
                    body,
                    "  {prefix}[{reg}] = ({prefix}[{reg}] & 0xffff0000u) | (pack2x16float(vec2<f32>({expr}, 0.0)) & 0x0000ffffu);"
                )
                .ok();
            } else {
                writeln!(
                    body,
                    "  {prefix}[{reg}] = ({prefix}[{reg}] & 0x0000ffffu) | (pack2x16float(vec2<f32>(0.0, {expr})) & 0xffff0000u);"
                )
                .ok();
            }
        }
    }
    Some(())
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
    for (index, instr) in shader.instrs.iter().enumerate() {
        let byte_offset = index * 8;
        if let Some(reason) = instr.blocked {
            return Err(EmitError::Blocked { index, byte_offset, reason, raw: instr.raw });
        }
        if guard_internal_reads {
            check_internal_reads(instr, index, byte_offset, &internal_written)?;
        }
        emit_instr(&mut body, instr, index, byte_offset, shader.kind)?;
        record_internal_writes(instr, &mut internal_written);
    }
    Ok(body)
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
        Op::Tex { coords, .. } => {
            let n = (coords as usize).clamp(1, 4);
            [0 < n, 1 < n, 2 < n, 3 < n]
        }
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
pub fn wrap_module(body: &str, tex_units: &[TexBinding]) -> String {
    let mut m = String::new();
    // Each sampled unit needs a bound texture + sampler (referenced as `t{u}`/`s{u}` by
    // `emit_tex`). Group 0 / running bindings; the real pipeline builder assigns the same
    // names to the draw's bound textures (and its actual type - cube/3d for 3-coord samples).
    // Declared before the private register banks. Here a 3-coord sample validates as 3D.
    for (i, b) in tex_units.iter().enumerate() {
        let (tb, sb) = (i as u32 * 2, i as u32 * 2 + 1);
        let ty = if b.coords >= 3 { "texture_3d<f32>" } else { "texture_2d<f32>" };
        let _ = writeln!(m, "@group(0) @binding({tb}) var t{}: {ty};", b.unit);
        let _ = writeln!(m, "@group(0) @binding({sb}) var s{}: sampler;", b.unit);
    }
    for bank in ["r", "pa", "sa", "o", "i"] {
        let _ = writeln!(m, "var<private> {bank}: array<u32, {BANK_REGS}>;");
    }
    // Predicate registers p0..p3, written by the test (VTST) ops and read by predicated
    // instructions. Four booleans, zero-initialised (a predicate is false until a test sets it).
    let _ = writeln!(m, "var<private> p: array<bool, 4>;");
    // `front_facing` is declared unconditionally - see the note in `link::build_linked_module`.
    let _ = writeln!(m, "\nstruct FsIn {{ @builtin(front_facing) front_facing: bool }};");
    let _ = writeln!(m, "\n@fragment\nfn fs_main(in: FsIn) -> @location(0) vec4<f32> {{");
    m.push_str(FRONT_FACING_DECL);
    m.push_str(body);
    let _ = writeln!(
        m,
        "  return vec4<f32>(bitcast<f32>(o[0]), bitcast<f32>(o[1]), bitcast<f32>(o[2]), bitcast<f32>(o[3]));\n}}"
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
    for bank in ["r", "pa", "sa", "o", "i"] {
        let _ = writeln!(m, "var<private> {bank}: array<u32, {BANK_REGS}>;");
    }
    let _ = writeln!(m, "var<private> p: array<bool, 4>;");
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
    // A GLOBAL (SPECIAL hardware register) operand is decoded structurally but has no value
    // until its index's meaning is established. Report it by INDEX, ahead of the generic
    // unmapped-operand path, so the failure says which register to go and establish. The one
    // established register ([`global_u32_expr`]) is exempt, and only inside the one operation
    // that reads it as raw bits - a bitwise test. Any other op reading even that register is
    // outside what the corpus establishes, so it still hard-fails by index.
    let global_ok = matches!(instr.op, Op::Test { alu: TestAlu::BitAnd, .. });
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
    let mut stmts = String::new();
    let s = &mut stmts;
    // The two ops with no mandatory register destination: a predicate-only test writes just
    // `p[n]`, and a discard writes nothing at all.
    if let Op::Test { alu, cmp, reduce, pdst, write_back } = instr.op {
        emit_test(s, instr, instr.dest.as_ref(), alu, cmp, reduce, pdst, write_back, kind)
            .ok_or_else(unmapped)?;
        return finish_predicated(body, instr, &stmts, index);
    }
    if matches!(instr.op, Op::Kill) {
        stmts.push_str("  discard;
");
        return finish_predicated(body, instr, &stmts, index);
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
        Op::Tex { unit, coords, coord_half, lod } => {
            emit_tex(s, instr, dest, unit, coords, coord_half, lod, index).ok_or_else(unmapped)
        }
        Op::Bitwise { kind, imm } => emit_bitwise(s, instr, dest, kind, imm).ok_or_else(unmapped),
        other => Err(EmitError::UnsupportedOp {
            index,
            byte_offset,
            op: op_name(other),
            group: instr.group,
            raw: instr.raw,
        }),
    };
    r?;
    finish_predicated(body, instr, &stmts, index)
}

/// `dest.c = (src1.c OP src2.c)` for each written channel.
fn emit_binop(body: &mut String, instr: &Instr, dest: &Operand, mask: [bool; 4], op: &str, _i: usize) -> Option<()> {
    let (s1, s2) = (instr.srcs.first()?, instr.srcs.get(1)?);
    let p = Prec::of(instr);
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let e = format!("({} {op} {})", src_channel(s1, c, p)?, src_channel(s2, c, p)?);
        store_channel(body, dest, c, &e, p)?;
    }
    Some(())
}

/// `dest.c = FN(src1.c, src2.c)` for each written channel (min/max).
fn emit_func2(body: &mut String, instr: &Instr, dest: &Operand, mask: [bool; 4], func: &str, _i: usize) -> Option<()> {
    let (s1, s2) = (instr.srcs.first()?, instr.srcs.get(1)?);
    let p = Prec::of(instr);
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let e = format!("{func}({}, {})", src_channel(s1, c, p)?, src_channel(s2, c, p)?);
        store_channel(body, dest, c, &e, p)?;
    }
    Some(())
}

/// `dest.c = FN(src1.c)` for each written channel (fract/dpdx/dpdy).
fn emit_func1(body: &mut String, instr: &Instr, dest: &Operand, mask: [bool; 4], func: &str, _i: usize) -> Option<()> {
    let s1 = instr.srcs.first()?;
    let p = Prec::of(instr);
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let e = format!("{func}({})", src_channel(s1, c, p)?);
        store_channel(body, dest, c, &e, p)?;
    }
    Some(())
}

/// `dest.c = WRAP(src1.c)` for each written channel, where `wrap` builds the WGSL rvalue
/// from the source channel expression. Covers the transcendentals (rcp/rsq/log2/exp2) and a
/// plain move (`wrap` = identity), which do not fit the fixed `FN(x)` shape of `emit_func1`.
fn emit_unary(
    body: &mut String,
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
        store_channel(body, dest, c, &e, p)?;
    }
    Some(())
}

/// `dest.c = src1.c * src2.c + src3.c` (multiply-add).
fn emit_mad(body: &mut String, instr: &Instr, dest: &Operand, mask: [bool; 4]) -> Option<()> {
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
        store_channel(body, dest, c, &e, p)?;
    }
    Some(())
}

/// Conditional move (VMOVC): `dest.c = select(src2.c, src1.c, test(src0.c, 0))` per written
/// channel. `srcs` is `[src1 (true), src2 (false), src0 (test)]`; the WGSL `select(f, t,
/// cond)` returns `t` when `cond` is true, matching "src1 when the compare holds".
fn emit_cmov(body: &mut String, instr: &Instr, dest: &Operand, mask: [bool; 4], test: CompareMethod) -> Option<()> {
    let (s1, s2, s0) = (instr.srcs.first()?, instr.srcs.get(1)?, instr.srcs.get(2)?);
    let p = Prec::of(instr);
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let cond = compare_zero_expr(&src_channel(s0, c, p)?, test);
        let e = format!("select({}, {}, {cond})", src_channel(s2, c, p)?, src_channel(s1, c, p)?);
        store_channel(body, dest, c, &e, p)?;
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
    body: &mut String,
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
        let (a, b) = (src_channel(s1, c, p)?, src_channel(s2, c, p)?);
        let value = match alu {
            TestAlu::Add => format!("({a} + {b})"),
            TestAlu::Sub => format!("({a} - {b})"),
            TestAlu::Mul => format!("({a} * {b})"),
            // Resolved above - the raw-lane path never reaches here.
            TestAlu::BitAnd => unreachable!("bitwise test resolved before the float path"),
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
            let (a, b) = (src_channel(s1, c, p)?, src_channel(s2, c, p)?);
            let value = match alu {
                TestAlu::Add => format!("({a} + {b})"),
                TestAlu::Sub => format!("({a} - {b})"),
                TestAlu::Mul => format!("({a} * {b})"),
                // A bitwise write-back is not modelled in the float store path; the corpus has
                // no such instruction, so refusing is exact rather than restrictive.
                TestAlu::BitAnd => return None,
            };
            store_channel(body, dest, c, &value, p)?;
        }
    }
    Some(())
}

/// Dot product: a scalar `src1 . src2` over `components` channels, broadcast to every
/// written destination channel.
fn emit_dot(body: &mut String, instr: &Instr, dest: &Operand, mask: [bool; 4], components: u8) -> Option<()> {
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
        store_channel(body, dest, c, &expr, p)?;
    }
    Some(())
}

/// Texture sample: `dest.xyzw = textureSample(t{unit}, s{unit}, coord)`. The coordinate is
/// `srcs[0]`, read as `coords` components (1D pads Y to 0); the bound texture+sampler are
/// the module-scope `t{unit}`/`s{unit}` bindings the pipeline builder wires. The sampled
/// RGBA is written to the destination's four channels.
#[allow(clippy::too_many_arguments)]
fn emit_tex(
    body: &mut String,
    instr: &Instr,
    dest: &Operand,
    unit: u8,
    coords: u8,
    coord_half: bool,
    lod: TexLod,
    index: usize,
) -> Option<()> {
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
    };
    if coords >= 3 {
        let cy = src_channel(coord, 1, cp)?;
        let cz = src_channel(coord, 2, cp)?;
        writeln!(body, "  let {tmp} = {func}(t{unit}, s{unit}, vec3<f32>({cx}, {cy}, {cz}){extra});").ok();
    } else {
        let cy = if coords >= 2 { src_channel(coord, 1, cp)? } else { "0.0".to_string() };
        writeln!(body, "  let {tmp} = {func}(t{unit}, s{unit}, vec2<f32>({cx}, {cy}){extra});").ok();
    }
    const COMP: [&str; 4] = ["x", "y", "z", "w"];
    for c in 0..4 {
        store_channel(body, dest, c, &format!("{tmp}.{}", COMP[c]), dp)?;
    }
    Some(())
}

/// Integer bitwise / shift on channel 0 only, operating on the 32-bit lane bit pattern:
/// `dest.x = bitcast<f32>(bitcast<u32>(src1.x) OP b)`, where `b` is the inline immediate or
/// `bitcast<u32>(src2.x)`. Shift amounts are masked to 31; ASR uses a signed shift.
fn emit_bitwise(body: &mut String, instr: &Instr, dest: &Operand, kind: BitwiseKind, imm: Option<u32>) -> Option<()> {
    use BitwiseKind::*;
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
        Some(format!("{}[{}]", bank_prefix(o.bank)?, o.index as u32))
    };
    let a = raw(instr.srcs.first()?)?;
    let b = match imm {
        Some(v) => format!("{v}u"),
        None => raw(instr.srcs.get(1)?)?,
    };
    let expr = match kind {
        And => format!("({a} & {b})"),
        Or => format!("({a} | {b})"),
        Xor => format!("({a} ^ {b})"),
        Shl => format!("({a} << ({b} & 31u))"),
        Shr => format!("({a} >> ({b} & 31u))"),
        Asr => format!("bitcast<u32>(bitcast<i32>({a}) >> ({b} & 31u))"),
    };
    writeln!(body, "  {}[{}] = {expr};", bank_prefix(dest.bank)?, dest.index as u32).ok();
    Some(())
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
        if let Op::Tex { unit, coords, .. } = i.op {
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

    #[test]
    fn honours_partial_write_mask() {
        let d = Operand::plain(Bank::Temp, 2, 0);
        let a = Operand::plain(Bank::Temp, 4, 0);
        let b = Operand::plain(Bank::Temp, 6, 0);
        let mut ins = instr(Op::Add, Some(d), vec![a, b]);
        ins.write_mask = [true, false, true, false];
        let wgsl = emit_fragment(&shader(vec![ins])).unwrap();
        assert!(wgsl.contains(&st("r", 2, &format!("({} + {})", rd("r", 4), rd("r", 6)))), "got:\n{wgsl}");
        assert!(wgsl.contains(&st("r", 4, &format!("({} + {})", rd("r", 6), rd("r", 8)))), "got:\n{wgsl}"); // channel 2
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
        let d = || Some(Operand::plain(Bank::Temp, 0, 0));
        let a = Operand::plain(Bank::Temp, 2, 0);
        let b = Operand::plain(Bank::Temp, 4, 0);
        let mn = emit_fragment(&shader(vec![instr(Op::Min, d(), vec![a, b])])).unwrap();
        assert!(mn.contains(&st("r", 0, &format!("min({}, {})", rd("r", 2), rd("r", 4)))), "got:\n{mn}");
        let fr = emit_fragment(&shader(vec![instr(Op::Frc, d(), vec![a])])).unwrap();
        assert!(fr.contains(&st("r", 0, &format!("fract({})", rd("r", 2)))), "got:\n{fr}");
        let dx = emit_fragment(&shader(vec![instr(Op::Dsx, d(), vec![a])])).unwrap();
        assert!(dx.contains(&st("r", 0, &format!("dpdx({})", rd("r", 2)))), "got:\n{dx}");
        let dt = emit_fragment(&shader(vec![instr(Op::Dot { components: 4 }, d(), vec![a, b])])).unwrap();
        assert!(dt.contains(&st("r", 0, &format!("({})", (0..4).map(|c| format!("{} * {}", rd("r", 2 + c), rd("r", 4 + c))).collect::<Vec<_>>().join(" + ")))), "got:\n{dt}");
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
}
