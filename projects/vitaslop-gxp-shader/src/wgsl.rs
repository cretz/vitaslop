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
///
/// Public because the REFERENCE has to answer for the same register, and the index is the whole
/// of what "this global is the one we know" means - a second copy of the number in `interp` is a
/// second place for it to be wrong.
pub const GLOBAL_FACING: u8 = 16;

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

/// The value the constant arm of [`src_channel`] materialises for ONE channel: the operand's
/// `index` selects the table entry, the channel's SWIZZLE SELECTOR selects which table (and
/// carries the four inline constants), and `half` picks the F16 view.
///
/// >>> THIS EXISTS BECAUSE THE TWO HALVES DISAGREED, AND ONLY ONE OF THEM WAS RIGHT.
///
/// The emitter read the selector. The reference interpreter called [`cnst6_value`] with the
/// operand's index alone - so for every constant operand whose channel selects anything but
/// bank 0 it returned a DIFFERENT NUMBER than the shader the GPU runs. The corpus-wide
/// execution differential (`tests/execcases.rs`) found it on a four-instruction vertex program
/// where the emitter reads `1.0` and the reference read `0.0`, turning `pos * 1.0` into
/// `pos * 0.0` - a clip position of zero, i.e. a collapsed mesh.
///
/// That is not only a test-oracle fault: `interp` decides the per-pass DEPTH FIT and the clip-`w`
/// sign that sets the negative-projection correction, so a constant it read wrong is a
/// projection decided wrong on the shipped path.
///
/// `None` for the 8-bit view, which has no established constant table - refusing is exact, and
/// it mirrors the emitter, which refuses the same case rather than pick a float table.
pub fn cnst6_channel_value(index: u8, sel: u8, half: bool, fx8: bool) -> Option<f32> {
    Some(match (half, sel) {
        // The four INLINE constants are a property of the selector alone and are available in
        // every precision, the 8-bit view included - which is why the refusal below comes after
        // them, exactly as it does in `src_channel`.
        (_, 4) => 0.0,
        (_, 5) => 1.0,
        (_, 6) => 2.0,
        (_, 7) => 0.5,
        _ if fx8 => return None,
        (true, _) => cnst6_f16_value(sel, index),
        (false, 1) => f32::from_bits(CNST6_F32_BANK1[(index & 0x3f) as usize]),
        (false, _) => cnst6_value(index),
    })
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
        // `sign` is already in the bit pattern handed to `from_bits`, so the value carries it.
        // (There was a trailing `* if sign != 0 { 1.0 } else { 1.0 }` here, which multiplied by
        // one whichever way it went.)
        0 => return f32::from_bits(sign | 0x3380_0000).mul_add(frac as f32, 0.0),
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
    pub fn of(instr: &Instr) -> Prec {
        // The normalized U8 convert writes PACKED BYTES when that is the direction it runs
        // in - four channels in one register, not one float per lane. `half_precision`
        // cannot say so: it describes a float destination, and here there is not one.
        if let Op::PackUnorm8 { to_unorm8: true, .. } | Op::CopyFx8 = instr.op {
            return Prec::Fx8;
        }
        // The 8-bit COMBINER writes packed unorm bytes for the same reason, and `emit_sop2`
        // has always passed `Prec::Fx8` by hand - so this is the two agreeing rather than a
        // change of behaviour. It is load-bearing for anything that asks an instruction what
        // it writes WITHOUT going through that emitter: the reference interpreter's generic
        // store path, and the constant fold's unknown-lane marking, both of which were
        // addressing a sop2 destination as whole float lanes - claiming three registers it
        // never writes while leaving stale known values in the bytes it does.
        //
        // MEASURED EMISSION-NEUTRAL, because the fold can move generated code: the corpus WGSL
        // hash over all 1,151 blobs is BYTE-IDENTICAL with this arm and with
        // `from_half(half_precision)` in its place. Nothing in the emitter asks `Prec::of`
        // about these two ops (its own call sites are per-op, and the two generic ones are for
        // `DepthF` and the derivatives), and the fold has no folded value to lose where it
        // never modelled the op. Re-run that hash if either of those changes.
        if let Op::Sop2 { .. } = instr.op {
            return Prec::Fx8;
        }
        // A VTST on the 8-BIT ALU reads and writes bytes, which `emit_test` already says with
        // its own local override (`wp`). Same rule as `Op::Sop2` above: the general answer must
        // match the emitter's hand-written one, or anything that asks the instruction directly
        // addresses its destination as whole float lanes. VTSTMSK is deliberately NOT here -
        // the decoder does not produce the 8-bit ALU for that group and `emit_test_mask`
        // refuses it, so naming it would claim a form that does not exist.
        if let Op::Test { alu: crate::ir::TestAlu::Fx8Sub, .. } = instr.op {
            return Prec::Fx8;
        }
        Prec::from_half(instr.half_precision)
    }

    /// The precision an instruction READS its sources at. For every operation this is the
    /// same as the destination's - except a format convert ([`Op::Pack`], VPCK), whose whole
    /// purpose is that the two differ: an F16->F32 unpack read at F32 would take a register
    /// holding two halves and interpret it as one 32-bit float, i.e. a denormal instead of the
    /// value. (A texture sample also carries independent coordinate/result precisions, but it
    /// gets them from its own decoded fields - see [`emit_tex`].)
    pub fn src_of(instr: &Instr) -> Prec {
        // ...and it READS packed bytes in the other direction, for the same reason. The 8-bit
        // combiner reads them at BOTH ends: its sources and its destination are the same
        // four-bytes-in-one-register view.
        if let Op::PackUnorm8 { to_unorm8: false, .. } | Op::CopyFx8 | Op::Sop2 { .. } = instr.op {
            return Prec::Fx8;
        }
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
pub fn bank_prec(bank: Bank, prec: Prec) -> Prec {
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
    // A register-INDIRECT operand: its base is only known at run time, so a lane selector
    // spells out `bank[idx + offset + sel]` rather than a constant index. That is the same
    // arithmetic `read_lane` does for a plain F32 operand - one register per lane - with the
    // index register in front of it, which is what makes a vector read through `i0` (a matrix
    // row a program looked up) come out as the four consecutive registers it is.
    //
    // F32 only: the corpus has no F16 or 8-bit instruction reading this row, and the two narrow
    // views pack several lanes into ONE register, so their lane-to-register map is a different
    // question this has no evidence for. It returns None and the caller hard-fails.
    if matches!(op.bank, Bank::Indexed) {
        let sel = op.swizzle[c];
        let mut e = match (bank_prec(op.bank, prec), sel) {
            (Prec::F32, 0..=3) => format!("bitcast<f32>({})", indexed_element(op, sel as u32)?),
            (_, 4) => "0.0".to_string(),
            (_, 5) => "1.0".to_string(),
            (_, 6) => "2.0".to_string(),
            (_, 7) => "0.5".to_string(),
            _ => return None,
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
        Prec::F16 => half_stmt(prefix, op.index as u32 + (c as u32 >> 1), c & 1 == 1, expr, false),
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

    /// [`Self::store_raw`] for a lane that is HALF a register: lane `c` is half `c & 1` of
    /// register `index + (c >> 1)`, the same packing [`Prec::F16`] uses, but read-modify-writing
    /// a raw 16-bit pattern rather than packing a float. The paired half keeps its value, which
    /// is the whole point - the two halves are two different values the shader will read back
    /// separately.
    fn store_raw_half(&mut self, op: &Operand, c: usize, expr: &str) -> Option<()> {
        let prefix = bank_prefix(op.bank)?;
        // The placement is `ir::packed_dest_slot`, not a fifth copy of `c >> 1` and `c & 1`.
        let (reg_off, shift) = crate::ir::packed_dest_slot(16, c as u32);
        let reg = op.index as u32 + reg_off;
        let stmt = |v: &str| half_stmt(prefix, reg, shift == 16, v, true);
        if !self.stage {
            self.body.push_str(&stmt(expr));
            return Some(());
        }
        let tmp = format!("g{c}");
        let _ = writeln!(self.body, "  let {tmp} = {expr};");
        self.deferred.push(stmt(&tmp));
        Some(())
    }

    /// [`Self::store_raw_half`] one element down: lane `c` is BYTE `c` of register `index`.
    ///
    /// Four bytes fit in one register where four halves need two, so there is no register
    /// stride here - the whole four-channel span is `index` alone. That is the same rule
    /// [`emit_pack_from_int`]'s 8-bit source reads by, and the reason both of them answer
    /// [`crate::ir::Instr::source_packed_bytes`] rather than the two-way precision question.
    fn store_raw_byte(&mut self, op: &Operand, c: usize, expr: &str) -> Option<()> {
        let prefix = bank_prefix(op.bank)?;
        // As in `store_raw_half`: the placement is stated once, in `ir::packed_dest_slot`.
        let (reg_off, sh) = crate::ir::packed_dest_slot(8, c as u32);
        let reg = op.index as u32 + reg_off;
        let keep = !(0xffu32 << sh);
        let stmt = |v: &str| {
            format!("  {prefix}[{reg}] = ({prefix}[{reg}] & {keep:#010x}u) | (({v} & 0xffu) << {sh}u);
")
        };
        if !self.stage {
            self.body.push_str(&stmt(expr));
            return Some(());
        }
        let tmp = format!("g{c}");
        let _ = writeln!(self.body, "  let {tmp} = {expr};");
        self.deferred.push(stmt(&tmp));
        Some(())
    }

    /// Apply every held-back store. Called once the instruction has read everything it reads.
    fn flush(&mut self) {
        for stmt in std::mem::take(&mut self.deferred) {
            self.body.push_str(&stmt);
        }
    }
}

/// [`Op::CmovU8`]: `dest.byte[c] = test(src0.byte[c]) ? src1.byte[c] : src2.byte[c]`.
///
/// Every read is a raw byte of the operand's own register - the same addressing
/// [`Dest::store_raw_byte`] writes by - so nothing here goes through a float or half view.
fn emit_cmov_u8(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    mask: [bool; 4],
    test: CompareMethod,
) -> Option<()> {
    let s1 = instr.srcs.first()?;
    let s2 = instr.srcs.get(1)?;
    let s0 = instr.srcs.get(2)?;
    // >>> ONE TEST, ON ONE BYTE - not four, and the corpus cannot tell the two apart.
    //
    // "U8" names how the TEST reads its operand: as an unsigned byte. Whether the instruction
    // then tests each of the four bytes SEPARATELY or tests one and moves the masked bytes on
    // its answer is not visible in any captured word, because every one of them carries the
    // full mask `0b1111` - and with a full mask the two readings differ only where src0's four
    // bytes DISAGREE. They agree everywhere else, which is why this takes the reading that
    // cannot produce a value neither source holds: a per-byte test over a source whose bytes
    // differ SPLICES the two sources together, and the value this instruction feeds is a bone
    // index that is then multiplied and used as a memory offset, where a spliced index
    // addresses neither matrix.
    //
    // `VITASLOP_GXP_CMOVU8=byte` is the ARM BACK to the per-byte test, so both readings are
    // reachable from ONE build [[vitaslop-browser-ab-needs-a-negative-control]].
    let per_byte = crate::module::cmov_u8_tests_each_byte();
    let elem = |o: &Operand, lo: u32, signed: bool| -> Option<String> {
        Some(raw_elem_expr(&format!("{}[{}]", bank_prefix(o.bank)?, o.index as u32), lo, 8, signed))
    };
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let lo = c as u32 * 8;
        let test_lo = if per_byte { lo } else { 0 };
        // The test is on the UNSIGNED byte, which is what the form is named for. `LtZero` and
        // `LteZero` therefore need the SIGNED view of that byte, so the comparison means what
        // it says rather than always failing.
        let cond = match test {
            CompareMethod::EqZero => format!("({} == 0u)", elem(s0, test_lo, false)?),
            CompareMethod::NeZero => format!("({} != 0u)", elem(s0, test_lo, false)?),
            CompareMethod::LtZero => format!("({} < 0i)", elem(s0, test_lo, true)?),
            CompareMethod::LteZero => format!("({} <= 0i)", elem(s0, test_lo, true)?),
        };
        let e = format!(
            "select({}, {}, {cond})",
            elem(s2, lo, false)?,
            elem(s1, lo, false)?,
            cond = cond
        );
        body.store_raw_byte(dest, c, &e)?;
    }
    Some(())
}

/// >>> EVERY f32 -> f16 NARROWING THIS CRATE EMITS GOES THROUGH ONE FUNCTION, AND THE MODULE
/// >>> PREAMBLE - NOT THE BODY - DECIDES HOW IT ROUNDS.
///
/// # The defect this exists to fix
/// Every half-precision store used to be spelled `pack2x16float(...)` inline. That builtin
/// lowers to SPIR-V `PackHalf2x16`, **whose rounding mode the language does not specify**, and
/// the device this project is developed on TRUNCATES: `probe-f16round.mjs`, 48 chosen inputs
/// sitting at known places between two representable halves, matched round-toward-zero 48/48
/// and round-to-nearest-even 28/48. The CPU reference rounds to nearest even (checked
/// exhaustively against a third-party oracle over 674,872 values), so every f16 store the
/// recompiler emitted disagreed with it by up to one ULP **in a direction that biases**, and a
/// chain of them drifts. Fragment programs on real titles are 70-90% F16, so this was
/// essentially all fragment arithmetic.
///
/// MEASURED, by making the reference truncate to match: corpus divergences **126 -> 21**, exact
/// **607 -> 834**, 105 cleared and 0 new. That measurement is what says the mode is the cause;
/// the reference was reverted, because adopting an implementation-defined mode as the spec is
/// not a fix.
///
/// It is also a PORTABILITY defect and not only an accuracy one: two devices may lower the same
/// builtin two different ways, so the same title renders differently on each and a desktop
/// number cannot predict the phone's.
///
/// # Why the body calls a function instead of spelling the conversion
/// The emitted body is HASHED AND CACHED - the hash is the identity of a recompiled program -
/// so it must not depend on the device. The body therefore calls [`HALF_LO_FN`]/[`HALF_HI_FN`]/
/// [`HALF_PK_FN`] and the MODULE PREAMBLE, assembled after the adapter is known, supplies the
/// definitions. Both definitions round to nearest even; which one is chosen is a speed
/// question, never a numeric one, and [`half_helper_text`]'s two arms must agree bit for bit.
pub const HALF_LO_FN: &str = "gxp_hlo";

/// The high-half store helper. See [`HALF_LO_FN`].
pub const HALF_HI_FN: &str = "gxp_hhi";

/// The WHOLE-register store helper: both halves at once, the shape [`fold_halves`] produces.
/// See [`HALF_LO_FN`].
pub const HALF_PK_FN: &str = "gxp_hpk";

/// The f32 -> f16 narrowing itself, as a 16-bit pattern in the low half of a `u32`. Every other
/// helper is written in terms of this one, so the rounding mode is stated in ONE place.
///
/// The four arms' WGSL lives in `src/f16rounding/*.wgsl` rather than in Rust string literals,
/// because `probe-f16round.mjs` runs the SHIPPED text on a real device - and a probe that
/// carried its own copy would be measuring a transcription, which is the one thing a probe
/// must not do. Rust `include_str!`s the same bytes the probe reads.
pub const HALF_BITS_FN: &str = "gxp_f16b";

/// The f16 ROUND TRIP - narrow and widen again, leaving an `f32` holding a value the 16-bit
/// register can hold. What [`crate::link`]'s unpacked half-register home rounds through.
pub const HALF_QUANT_FN: &str = "gxp_hq";

/// Whether the device this build talks to offers `shader-f16`, so the preamble may use the
/// language's own narrowing instead of the portable bit arithmetic. Set by the renderer from
/// the adapter ([`set_native_f16`]); OFF until it says so, because `enable f16;` does not
/// compile on a device that was not created with the feature - and OFF is the arm that is
/// correct everywhere.
static NATIVE_F16: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Set by the renderer once it knows the device. See [`NATIVE_F16`].
pub fn set_native_f16(on: bool) {
    NATIVE_F16.store(on, core::sync::atomic::Ordering::Relaxed);
}

/// Whether the preamble will use the native narrowing. See [`NATIVE_F16`].
pub fn native_f16() -> bool {
    NATIVE_F16.load(core::sync::atomic::Ordering::Relaxed)
}

/// The portable round-to-nearest-even narrowing: no extension, correct on every device.
///
/// Written as the cases the format actually has rather than as bit tricks, because this is the
/// one function whose wrongness would be invisible - it would round *nearly* right and the
/// residue would read as a different defect.
///
/// * `0x477ff000` is the exact TIE that rounds up out of the f16 range (its 10-bit significand
///   is odd, so ties-to-even carries out of the exponent), which is why the overflow test is
///   `>=`. A finite value that overflows **SATURATES to 65504** rather than becoming an
///   infinity: that is what the guest's hardware does, what the CPU reference models
///   ([`crate::fold::f32_to_f16_bits_saturating`]) and what the corpus differential measured on
///   the device - 200 diverging lanes where the reference held an infinity or a NaN and the GPU
///   repeatedly held 65504. An infinity that arrives as one stays one.
/// * `0x33000000` is 2^-25, the tie between zero and the smallest subnormal half; its
///   significand is even, so it rounds to ZERO and the underflow test is `<`.
/// * A carry out of the rounded significand is left to propagate into the exponent field on its
///   own - which is what turns the largest subnormal into the smallest normal, with no case of
///   its own. It can never carry past 65504, because the saturation test already returned.
const HALF_HELPERS_PORTABLE: &str = include_str!("f16rounding/portable.wgsl");

/// The same helpers over the language's own `f16`, for a device that has the feature.
///
/// The pair form is ONE conversion instruction here rather than two narrowings and a shift,
/// which is why [`fold_halves`] is worth as much as it is: a four-channel 16-bit store is one
/// of these.
///
/// `f16(v)` is the VALUE conversion, a different operation from `pack2x16float` in the language
/// and measured round-to-nearest-even 48/48 by `probe-f16round.mjs` on the same inputs the pack
/// builtin truncated. That is a MEASUREMENT of one device, not a guarantee from the spec, which
/// is why the probe is a CI job and why [`HALF_HELPERS_PORTABLE`] is the default.
///
/// What the language does NOT leave open is that a conversion of an out-of-range value gives an
/// INDETERMINATE result - so the saturation the hardware performs cannot be left to it, and
/// `gxp_f16c` clamps first. `0x477fe000` is 65504, the largest finite half; a value between it
/// and the tie rounds to 65504 anyway, so clamping there changes nothing a round would not.
/// The test is on the BIT PATTERN rather than on `abs(v)` so that a NaN - which no comparison
/// answers usefully - falls through untouched instead of being clamped into a number.
///
/// >>> AND THE BITS COME OUT THROUGH `pack2x16float`, WHICH IS THE TRUNCATING BUILTIN THIS
/// >>> WHOLE CHANGE EXISTS TO STOP USING. That is not a contradiction, it is the point: its
/// argument has ALREADY been rounded to a value f16 holds exactly, and every rounding mode
/// agrees on a value that needs no rounding. So the mode stops mattering and the pack becomes a
/// pure bit move.
///
/// The obvious spelling - `bitcast<u32>(vec2<f16>(f16(lo), f16(hi)))` - is what the probe uses
/// and Tint accepts it, but **naga rejects it**: it reads the bitcast as componentwise and
/// types the result `vec2<u16>`, so the mask that follows is a vector-scalar `&` and the module
/// fails validation. The desktop is naga and the browser is Tint, so a form only one of them
/// takes is a module that builds here and refuses there. `every_f16_rounding_arm_parses_and_validates`
/// is what caught it, and is why all three arms are validated on every machine rather than only
/// the one this adapter happens to choose.
const HALF_HELPERS_NATIVE: &str = include_str!("f16rounding/native.wgsl");

/// The two helpers written in terms of [`HALF_BITS_FN`], the same in both arms.
const HALF_HELPERS_COMMON: &str = include_str!("f16rounding/common.wgsl");

/// The NEGATIVE CONTROL arm: `pack2x16float` under the same helper names, which is bit for bit
/// what every build before this one emitted. See [`crate::link::F16_ROUND_ARM`].
const HALF_HELPERS_PACK: &str = include_str!("f16rounding/pack.wgsl");

/// The helper definitions a module needs, in the arm this run is on.
///
/// All five are emitted together whenever any is called. WGSL has no dead-function warning and
/// a backend drops what nothing calls, so splitting them per call site would buy nothing and
/// would be five more ways for the text to disagree with itself.
fn half_helper_text() -> (String, bool) {
    let (narrow, enable) = match (crate::link::arm(crate::link::F16_ROUND_ARM), native_f16()) {
        (Some("0"), _) => (HALF_HELPERS_PACK, false),
        // `native` / `portable` FORCE an arm, which is what lets the case harnesses check both
        // on one device: the two must agree bit for bit, and only a run of each can say so.
        (Some("native"), _) => (HALF_HELPERS_NATIVE, true),
        (Some("portable"), _) => (HALF_HELPERS_PORTABLE, false),
        (_, true) => (HALF_HELPERS_NATIVE, true),
        (_, false) => (HALF_HELPERS_PORTABLE, false),
    };
    (format!("{narrow}{HALF_HELPERS_COMMON}"), enable)
}

/// Whether this run rounds f16 stores to nearest even - false only under the negative-control
/// arm. Reported by the renderer, because "which rounding mode did this picture use" is not
/// answerable from a screenshot [[vitaslop-a-device-dump-must-name-its-own-build]].
pub fn f16_round_to_nearest() -> bool {
    crate::link::arm(crate::link::F16_ROUND_ARM) != Some("0")
}

/// Whether `module` calls any of the f16 helpers, and therefore needs their definitions.
fn calls_half_helpers(module: &str) -> bool {
    [HALF_LO_FN, HALF_HI_FN, HALF_PK_FN, HALF_QUANT_FN, HALF_BITS_FN]
        .iter()
        .any(|f| module.contains(&format!("{f}(")))
}

/// Give an assembled module the f16 helper definitions its body calls, and - in the native arm -
/// the `enable f16;` that lets them compile.
///
/// # Why the definitions go after the directives and the `enable` goes at byte zero
/// A WGSL module's `enable`/`requires`/`diagnostic` directives must all precede every
/// declaration. A dual-source fragment pair already carries `enable dual_source_blending;`, so
/// inserting a FUNCTION at byte zero would put that directive after a declaration, the device
/// would refuse the pipeline, and the frame would go black - which is exactly what happened
/// when the rounding helper was first added [[vitaslop-a-diagnostic-at-debug-is-a-diagnostic-that-does-not-exist]].
/// So the `enable` goes first and the functions go after whatever directives are there.
///
/// Idempotent: a module that already carries the definitions is returned unchanged, so a
/// builder that wraps another builder's output cannot emit them twice.
pub fn add_half_helpers(module: String) -> String {
    if !calls_half_helpers(&module) || module.contains(&format!("fn {HALF_BITS_FN}(")) {
        return module;
    }
    let mut out = module;
    let (text, needs_enable) = half_helper_text();
    out.insert_str(crate::link::directives_end(&out), &text);
    if needs_enable {
        out.insert_str(0, "enable f16;\n");
    }
    out
}

/// The read-modify-write of ONE half of a 16-bit-packed register - the form a half with no
/// partner beside it needs. See [`fold_halves`] for the pair.
fn half_stmt(prefix: &str, reg: u32, high: bool, expr: &str, raw: bool) -> String {
    match (high, raw) {
        (false, false) => {
            format!("  {prefix}[{reg}] = {HALF_LO_FN}({prefix}[{reg}], {expr});\n")
        }
        (true, false) => {
            format!("  {prefix}[{reg}] = {HALF_HI_FN}({prefix}[{reg}], {expr});\n")
        }
        (false, true) => format!(
            "  {prefix}[{reg}] = ({prefix}[{reg}] & 0xffff0000u) | ({expr} & 0x0000ffffu);\n"
        ),
        (true, true) => format!(
            "  {prefix}[{reg}] = ({prefix}[{reg}] & 0x0000ffffu) | (({expr} & 0x0000ffffu) << 16u);\n"
        ),
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
    Ok(strip_split_markers(&emit_body_marked(shader)?))
}

/// A WGSL comment naming a TOP-LEVEL instruction boundary in an emitted body, so a module
/// builder can cut the body there. See the emission in [`emit_range`].
/// The prefix must not be a prefix of `link::BANKS_MARKER` (`  //@@GXP_REGISTER_BANKS`):
/// [`strip_split_markers`] drops every line that starts with it, and when the two shared
/// `  //@@` it dropped the bank DECLARATIONS too - a module whose every stage then referred to
/// register arrays that did not exist, which the device refuses, which drops every draw.
pub const SPLIT_MARKER: &str = "  //@@GXP_SPLIT ";

/// The marker line for instruction `index`, as [`emit_body_marked`] writes it - INCLUDING the
/// trailing newline, so a caller searching for it cuts between whole lines.
pub fn split_marker(index: usize) -> String {
    format!("{SPLIT_MARKER}{index}\n")
}

/// Remove every [`SPLIT_MARKER`] line.
///
/// Pre-sized, because this runs on EVERY recompile and the result is within a marker line or two
/// of the input: collecting into a `String` from an iterator of `&str` gets no useful size hint,
/// so it grew from zero and copied the whole text about a dozen times on the way. `emit_body` is
/// 94% of this crate's per-program CPU cost (p50 110 us, max 1.4 ms over a 1,151-blob corpus,
/// `where_the_shader_pipeline_spends_its_cpu_time_per_program`), and it builds the text once and
/// then rebuilds it here - so the copying in this four-line function is a real share of it.
pub fn strip_split_markers(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    for line in body.lines() {
        if line.starts_with(SPLIT_MARKER) {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// [`emit_body`] with the top-level instruction boundaries still marked.
pub fn emit_body_marked(shader: &Shader) -> Result<String, EmitError> {
    if shader.instrs.is_empty() {
        return Err(EmitError::Empty);
    }
    // Pre-sized from the instruction count rather than grown from zero. Over a 1,151-blob
    // corpus the emitted bodies average about 9 KB, which is roughly a hundred bytes an
    // instruction; a `String` growing from nothing to that copies the whole text a dozen times.
    // The figure is a HINT and nothing depends on it - a program that emits more simply grows
    // once or twice, exactly as before.
    let mut body = String::with_capacity(shader.instrs.len() * 128);
    // >>> THE DERIVATIVE PRELUDE, and why the function needs one.
    //
    // `dpdx`/`dpdy` may only be called from UNIFORM control flow, and a program is entitled to
    // put one inside a branch. The call therefore has to leave the branch, and there is nowhere
    // to put it but the top of the function - so it is emitted here and the branch body reads
    // the temporary. See the hoist in `emit_instr` for the guard that makes that exact.
    let mut prelude = String::new();
    // Track which internal-register lanes (i0..i3 x 4) an earlier instruction has written, so a
    // read of an unwritten internal lane in a FRAGMENT program hard-fails instead of translating
    // garbage: fragment internal registers can be pre-loaded by the texture-coordinate iterators
    // / PDS, which this model does not carry, so an unwritten read is genuinely unmodeled input.
    // A VERTEX program has no such preload - its internal registers are pure zero-initialised
    // scratch - so an unwritten read there is a defined 0.0 (a benign over-read of a padding lane
    // the fragment stage ignores, e.g. moving a computed vec3's absent w into an unused output
    // lane); the guard would wrongly reject those, so it applies to fragment programs only.
    //
    // >>> AND ONLY WHERE THE VALUE IS LIVE. The guard as first written refused on the read
    // ALONE, and that is what dropped fifteen of Madden's twenty-one unrecompilable fragment
    // blobs - a dropped pair means its mesh is ABSENT from the frame. MEASURED over every
    // captured corpus (`undefined_internal_reads_that_are_actually_live`): 57 fragment reads of
    // an unwritten internal lane, and **not one of them is live** - every single one is a
    // channel the program computes and then throws away. The idiom is ordinary: an F32 op
    // writes lane 0 of an internal register, and the F16 conditional move that consumes it
    // carries a two-channel write mask, so the guard counts a read of lane 1 whose destination
    // channel nothing downstream reads (e.g. `frag_9227e300` #11-#15, where the surviving
    // multiply takes `.x` from both operands).
    //
    // So the refusal stands exactly where it earns its keep - an unmodeled pre-load that
    // REACHES THE OUTPUT is still a hard failure, naming the lane - and a dead over-read
    // translates as the zero the vertex path has always given it. This is the same lesson the
    // SA bank already taught: refusing an unwritten scratch register dropped 39 of 40 pairs,
    // and zero was faithful.
    let live = live_instructions(shader);
    let guard_internal_reads: &[bool] =
        if shader.kind == ProgramKind::Fragment { &live } else { &[] };
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
        &mut prelude,
        None,
        &[],
    )?;
    Ok(format!("{prelude}{body}"))
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
/// One enclosing conditional region of the body being emitted, innermost LAST.
///
/// Threaded so a DERIVATIVE can be given a uniform gap where it stands - see
/// [`uniform_gap`]. `cond_start` alone says a derivative is inside a block; this says what the
/// block IS, which is what closing and re-entering it needs.
#[derive(Clone, Debug)]
struct Enclosing {
    /// The WGSL condition the region's `if` tests, exactly as it was written. Always a read of
    /// a predicate register (`p[n]` / `!p[n]`), so re-testing it is free of side effects - the
    /// gap still checks that nothing has WRITTEN that register since.
    cond: String,
    /// This region is the ELSE arm of that `if`, so re-entering it tests the negation.
    else_arm: bool,
    /// The indentation of the region's own `if` line.
    pad: String,
    /// A `loop`, which cannot be closed and re-entered at all: leaving it would run its
    /// remaining iterations outside the loop. See [`uniform_gap`].
    is_loop: bool,
}

/// Close every enclosing region so control flow is UNIFORM, and re-enter them all - the text
/// either side of a derivative that cannot be hoisted out of its block.
///
/// # Why a gap rather than a hoist
/// `dpdx`/`dpdy` difference a value across the rasteriser's 2x2 quad, so WGSL requires them in
/// uniform control flow. The hoist above handles the common case by computing the derivative
/// ABOVE the block - which is exact only while nothing in the block has rewritten the register
/// it reads. A football title's three world fragment programs compute the value they then
/// difference INSIDE the branch, and for those the hoist reads a different number; they were
/// refused, and a refused pair's mesh is absent from the frame.
///
/// Closing the block, calling the builtin, and re-entering is not a workaround for the
/// restriction - it is what the hardware does. The USSE does not predicate the DIFFERENCING at
/// all: a predicated `dsx` computes the quad derivative from the register file as it stands and
/// only the WRITE-BACK is conditional. In the emitted WGSL the register banks are function-scope
/// `var`s, so in the gap every lane of the quad holds exactly what the hardware's register file
/// would - the branch's value for the lanes that took it, the older value for the lanes that did
/// not - and differencing them there is the same operation. A hoist, by contrast, would compute
/// it as if every lane had taken the branch.
///
/// Refuses two shapes rather than emitting something else:
///   * an enclosing LOOP - closing it would run the rest of its iterations outside it;
///   * a PREDICATE REGISTER written inside the region before this point, which would make the
///     re-entry test a different condition than the one that let control in.
fn uniform_gap(
    shader: &Shader,
    enclosing: &[Enclosing],
    from: usize,
    index: usize,
    at: usize,
) -> Result<(String, String), EmitError> {
    let blocked = |reason| {
        Err(EmitError::Blocked { index: at, byte_offset: at * 8, reason, raw: shader.instrs[at].raw })
    };
    if enclosing.is_empty() {
        return blocked("a derivative reported as inside a block with no enclosing region to                         cut a uniform gap into");
    }
    if enclosing.iter().any(|e| e.is_loop) {
        return blocked("a derivative inside a LOOP whose source register the loop body writes -                         a loop cannot be closed and re-entered to reach uniform control flow");
    }
    // A write to ANY predicate register in the region counts: the conditions of the enclosing
    // ifs are predicate reads, and re-testing one the region has rewritten would admit a
    // different set of lanes to the rest of the block.
    let hi = index.min(shader.instrs.len());
    if shader.instrs[from.min(hi)..hi]
        .iter()
        .any(|i| matches!(i.op, Op::Test { .. } | Op::TestMask { .. }))
    {
        return blocked("a derivative inside a branch that rewrites a predicate register before                         it - the block cannot be re-entered on the same condition");
    }
    let mut close = String::new();
    let mut reopen = String::new();
    for e in enclosing.iter().rev() {
        let _ = writeln!(close, "{}}}", e.pad);
    }
    for e in enclosing {
        let c = if e.else_arm { format!("!({})", e.cond) } else { e.cond.clone() };
        let _ = writeln!(reopen, "{}if ({c}) {{", e.pad);
    }
    Ok((close, reopen))
}

#[allow(clippy::too_many_arguments)]
fn emit_range(
    body: &mut String,
    shader: &Shader,
    start: usize,
    end: usize,
    exit: usize,
    guard_internal_reads: &[bool],
    internal_written: &mut [bool; INTERNAL_LANES],
    open_loop: Option<usize>,
    depth: usize,
    prelude: &mut String,
    // `cond_start`: the instruction at which the OUTERMOST enclosing conditional region begins,
    // or `None` at the top level. A derivative hoisted out of a block is placed immediately
    // above that region, so this is where the "was the source written in between" check starts.
    cond_start: Option<usize>,
    // The conditional regions this range is INSIDE, outermost first - what a derivative that
    // cannot be hoisted needs in order to close them and re-enter. See [`uniform_gap`].
    enclosing: &[Enclosing],
) -> Result<(), EmitError> {
    let mut index = start;
    while index < end {
        // A TOP-LEVEL instruction boundary, named by index. It is a comment, so it costs the
        // emitted shader nothing, and [`strip_split_markers`] removes it for every caller but
        // the one that wants it - the dual-source module builder, which cuts the body in two
        // here (see [`split_marker`]). Only depth 1 is marked because only a top-level boundary
        // is a place a body can be CUT: a position inside an `if` or a loop would put the brace
        // on one side of the cut and its match on the other.
        if depth == 1 {
            let _ = writeln!(body, "{SPLIT_MARKER}{index}");
        }
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
                    prelude,
                    cond_start,
                    enclosing,
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
            // `guard_internal_reads` is empty for a vertex program (no guard at all) and
            // otherwise the per-instruction liveness mask: an undefined read whose result
            // reaches nothing is translated, not refused. See `emit_body_marked`.
            if guard_internal_reads.get(index).copied().unwrap_or(false) {
                check_internal_reads(instr, index, byte_offset, internal_written)?;
            }
            emit_instr(body, instr, index, byte_offset, shader.kind, shader, prelude, cond_start, enclosing)?;
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
                // >>> THE ARMS ARE BUFFERED so anything they HOIST can be written ABOVE the
                // `if`. A derivative may only be called from uniform control flow, and the
                // first uniform point outside this block is the line before it - see the
                // hoist in `emit_instr`. A block nested inside another one hoists all the way
                // out to the outermost, which is where `cond_start` points.
                let mut arms = String::new();
                let mut block_prelude = String::new();
                let inner_start = cond_start.or(Some(index + 1));
                {
                    let inner: &mut String =
                        if cond_start.is_some() { &mut *prelude } else { &mut block_prelude };
                    let _ = writeln!(arms, "{pad}if ({c}) {{");
                    // This range's own regions plus the arm being emitted, so an instruction
                    // inside it knows every block it would have to leave.
                    let mut inner_then: Vec<Enclosing> = enclosing.to_vec();
                    inner_then.push(Enclosing {
                        cond: c.clone(),
                        else_arm: false,
                        pad: pad.clone(),
                        is_loop: false,
                    });
                    let mut inner_else: Vec<Enclosing> = enclosing.to_vec();
                    inner_else.push(Enclosing {
                        cond: c.clone(),
                        else_arm: true,
                        pad: pad.clone(),
                        is_loop: false,
                    });
                    emit_range(
                        &mut arms,
                        shader,
                        index + 1,
                        then_end,
                        then_exit,
                        guard_internal_reads,
                        internal_written,
                        open_loop,
                        depth + 1,
                        inner,
                        inner_start,
                        &inner_then,
                    )?;
                    match else_arm {
                        None => {
                            let _ = writeln!(arms, "{pad}}}");
                        }
                        Some(e) => {
                            let _ = writeln!(arms, "{pad}}} else {{");
                            emit_range(
                                &mut arms,
                                shader,
                                target,
                                else_end.unwrap_or(e),
                                e,
                                guard_internal_reads,
                                internal_written,
                                open_loop,
                                depth + 1,
                                inner,
                                inner_start,
                                &inner_else,
                            )?;
                            let _ = writeln!(arms, "{pad}}}");
                        }
                    }
                }
                body.push_str(&block_prelude);
                body.push_str(&arms);
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
#[allow(clippy::too_many_arguments)]
fn emit_loop(
    body: &mut String,
    shader: &Shader,
    head: usize,
    tail: usize,
    guard_internal_reads: &[bool],
    internal_written: &mut [bool; INTERNAL_LANES],
    depth: usize,
    prelude: &mut String,
    cond_start: Option<usize>,
    enclosing: &[Enclosing],
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
    // Buffered for the reason the branch arms are: a derivative hoisted out of the body has to
    // land ABOVE the loop, where control flow is still uniform.
    let mut arms = String::new();
    let mut block_prelude = String::new();
    let inner_start = cond_start.or(Some(head));
    // A LOOP is an enclosing region a derivative cannot cut a uniform gap into: closing it
    // would run the rest of its iterations outside it. Recorded as one so the refusal names
    // the loop rather than emitting something the hardware does not do.
    let inner_encl: Vec<Enclosing> = enclosing
        .iter()
        .cloned()
        .chain([Enclosing { cond: String::new(), else_arm: false, pad: pad.clone(), is_loop: true }])
        .collect();
    {
        let inner: &mut String =
            if cond_start.is_some() { &mut *prelude } else { &mut block_prelude };
        let _ = writeln!(arms, "{pad}loop {{");
        emit_range(
            &mut arms,
            shader,
            head,
            tail,
            tail,
            guard_internal_reads,
            internal_written,
            Some(tail),
            depth + 1,
            inner,
            inner_start,
            &inner_encl,
        )?;
    }
    body.push_str(&block_prelude);
    body.push_str(&arms);
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

/// Which instructions of `shader` produce a value that REACHES ITS OUTPUT, by a backward walk
/// over the register file. Used by the undefined-internal-lane guard, which must fire on an
/// unmodeled input that is really consumed and stay silent on one the program throws away.
///
/// The walk is deliberately conservative in three places:
///
/// * A shader carrying a BACKWARD branch - a loop - is reported entirely live. The walk is a
///   single backward pass, which is exact only while every control-flow edge goes FORWARD (the
///   linear order is then a topological order of the CFG, so every reader is visited before
///   what it reads). A back edge breaks that: a read ABOVE a write can be reached AFTER it.
/// * A CONDITIONAL instruction's write does not KILL the lane it writes. A write that may not
///   execute does not redefine anything on the path where it is skipped, so an earlier write to
///   the same lane can still be the value a later read sees - and calling that earlier write
///   dead would silence a refusal that is owed. (This is the same "on EVERY path" distinction
///   `link::conditionally_executed` was written for.)
/// * An instruction with no destination, and any write to the OUTPUT bank, is live by
///   definition - the output IS the result, and a sideways effect this model does not name
///   must not be optimised away on the strength of not being named.
///
/// # Why "any branch at all is live" was not good enough
/// That was the original rule, and it was fine while every program it had to judge was
/// branch-free. It is not fine now: three of a football title's world fragment programs carry a
/// derivative inside a branch AND an undefined internal-lane read, and the blanket rule made the
/// second one live by fiat - so the pair stayed dropped for a reason nothing had measured. A
/// forward-branching program gets the ordinary answer; only a loop keeps the blanket one.
///
/// This decides only whether to REFUSE. It never removes an instruction: every instruction is
/// still emitted, so a wrong answer here cannot change what the shader computes.
fn live_instructions(shader: &Shader) -> Vec<bool> {
    let n = shader.instrs.len();
    let backward_branch = shader.instrs.iter().enumerate().any(|(i, instr)| {
        matches!(instr.op, Op::Branch { rel } if i as i64 + rel as i64 <= i as i64)
    });
    if backward_branch {
        return vec![true; n];
    }
    let conditional = crate::link::conditionally_executed(shader);
    // A register lane, keyed by bank and by `index + channel` - the same flat addressing the
    // emitter reads and writes the banks with.
    let key = |b: Bank, idx: u32, c: usize| -> (u8, u32) {
        let d = match b {
            Bank::Temp => 0u8,
            Bank::PrimaryAttr => 1,
            Bank::SecondaryAttr => 2,
            Bank::Internal => 3,
            Bank::Output => 4,
            _ => 5,
        };
        (d, idx + c as u32)
    };
    let mut wanted: std::collections::BTreeSet<(u8, u32)> = Default::default();
    let mut live = vec![false; n];
    for i in (0..n).rev() {
        let instr = &shader.instrs[i];
        if matches!(instr.op, Op::Nop) {
            continue;
        }
        let mut is_live = false;
        match instr.dest.as_ref() {
            // No destination named: this model cannot say what it produces, so it stays.
            None => is_live = true,
            Some(d) => {
                if matches!(d.bank, Bank::Output) {
                    is_live = true;
                }
                for c in 0..4 {
                    if instr.write_mask[c] && wanted.contains(&key(d.bank, u32::from(d.index), c)) {
                        is_live = true;
                    }
                }
                // Redefined here, so reads BELOW this point no longer keep the lane live
                // above it - but ONLY if this write happens on every path. A conditional write
                // leaves the older value in place wherever it is skipped.
                if is_live && !conditional.get(i).copied().unwrap_or(true) {
                    for c in 0..4 {
                        if instr.write_mask[c] {
                            wanted.remove(&key(d.bank, u32::from(d.index), c));
                        }
                    }
                }
            }
        }
        live[i] = is_live;
        if !is_live {
            continue;
        }
        for src in &instr.srcs {
            for c in 0..4 {
                if !instr.write_mask[c] {
                    continue;
                }
                let sel = src.swizzle[c];
                if sel > 3 {
                    continue; // a swizzle constant reads no register lane
                }
                wanted.insert(key(src.bank, u32::from(src.index), sel as usize));
            }
        }
    }
    live
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
    add_half_helpers(m)
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
    add_half_helpers(m)
}

/// The number of `u32` lanes one bank occupies in a [`wrap_compute_module`] case buffer.
pub const CASE_BANK_LANES: usize = BANK_REGS;

/// Rewrite every texture SAMPLE in `body` into a call on a constant-valued stand-in, returning
/// the rewritten body and the sampler units it replaced (ascending, deduplicated).
///
/// # Why a sample can be replaced by a constant and the check still means something
///
/// A texture unit bound to a texture whose every texel holds the SAME value returns that value
/// for any coordinate, at any mip level, under any filter and any wrap mode. So a constant
/// stand-in is not an approximation of that configuration - it is exactly it, and the reference
/// interpreter can be handed the identical constant through its own fetcher. Every piece of
/// arithmetic the sampled value feeds is then checked, across 230 corpus programs that had no
/// execution case at all because a compute dispatch cannot call `textureSample` (it needs the
/// implicit derivatives only a fragment stage has).
///
/// >>> AND WHAT IT DOES NOT CHECK, WHICH A READER MUST NOT FORGET. The coordinate is DISCARDED,
/// so the UV arithmetic that selected the texel is not verified - only everything downstream of
/// the fetch. A defect that computes the wrong UV is invisible here and needs a rig that varies
/// the texture with position (a real binding in a fragment stage), which this is not.
///
/// The parse is exact rather than heuristic: [`emit_tex`] writes one statement per sample, of
/// the form `let _texN = FUNC(tex, samp, vecK<f32>(...)EXTRA);`, and the coordinate always
/// begins at a `vec2<f32>(` or `vec3<f32>(`. Anything that does not match that shape is left
/// alone and the caller excludes the program rather than emit a module that samples a binding
/// this wrapper never declared.
fn substitute_constant_samples(body: &str, kind: ProgramKind) -> (String, Vec<u8>) {
    let mut units: Vec<u8> = Vec::new();
    let mut out = String::with_capacity(body.len());
    for line in body.lines() {
        let rewritten = (|| {
            let eq = line.find("= texture")?;
            let call = &line[eq + 2..];
            let open = call.find('(')?;
            let func = &call[..open];
            if !matches!(
                func,
                "textureSample" | "textureSampleBias" | "textureSampleLevel" | "textureSampleGrad"
            ) {
                return None;
            }
            // The first argument is the texture binding, whose name carries the unit.
            let args = &call[open + 1..];
            let first = args.split(',').next()?.trim();
            let prefix = match kind {
                ProgramKind::Vertex => "vt",
                _ => "t",
            };
            let unit: u8 = first.strip_prefix(prefix)?.parse().ok()?;
            if !units.contains(&unit) {
                units.push(unit);
            }
            // >>> AND THE COORDINATE COMES WITH IT, when the varying stand-in is armed. The
            // constant one discards it, which is precisely the gap: a program that computes the
            // WRONG UV samples a constant texture and gets the right answer.
            if case_tex_varies() {
                let coord = coord_arg(args)?;
                // >>> AND SO DOES THE LOD OPERAND, which a one-mip rig could not otherwise see at
                // all - see `case_texture_value_at`. It is everything after the coordinate: one
                // scalar for a bias or a level, two `vec2` derivatives for a gradient, which a
                // `vec4` constructor takes as they stand.
                let mode = match func {
                    "textureSampleBias" => TexLod::Bias,
                    "textureSampleLevel" => TexLod::Level,
                    "textureSampleGrad" => TexLod::Gradient,
                    _ => return Some(format!("{}= gxp_case_tex({unit}u, {coord});", &line[..eq])),
                };
                let (_, end, _) = coord_arg_span(args)?;
                let extra = args[end..].trim().strip_prefix(',')?.trim().strip_suffix(");")?.trim();
                let lod = if mode == TexLod::Gradient {
                    format!("vec4<f32>({extra})")
                } else {
                    format!("vec4<f32>({extra}, 0.0, 0.0, 0.0)")
                };
                return Some(format!(
                    "{}= gxp_case_texl({unit}u, {coord}, {}u, {lod});",
                    &line[..eq],
                    case_tex_lod_tag(mode)
                ));
            }
            Some(format!("{}= gxp_case_tex({unit}u);", &line[..eq]))
        })();
        out.push_str(rewritten.as_deref().unwrap_or(line));
        out.push('\n');
    }
    units.sort_unstable();
    (out, units)
}

/// The COORDINATE argument of an emitted sample call, as WGSL text: the second comma-separated
/// argument, taken by balanced parentheses so a nested call inside it cannot end it early.
///
/// [`emit_tex`] always writes the coordinate as a `vecK<f32>(...)` constructor, and a sample may
/// carry a bias or level AFTER it - so the argument cannot be found by splitting on commas, and
/// the closing parenthesis cannot be found by searching for the first `)`.
///
/// Normalised to three components, because a 2-coord sample and a 3-coord one must reach the
/// same stand-in: the extra component is an exact literal zero, so it costs the comparison
/// nothing.
fn coord_arg(args: &str) -> Option<String> {
    let (start, end, comps) = coord_arg_span(args)?;
    let text = &args[start..end];
    Some(if comps == 3 {
        text.to_string()
    } else {
        format!("vec3<f32>({text}, 0.0)")
    })
}

/// Where the coordinate argument BEGINS and ENDS within `args`, and how many components its
/// constructor names. One statement of the rule, read by [`coord_arg`] (which normalises the
/// text) and by [`rewrite_body_for_render`] (which splices around it and must not disturb a
/// trailing bias, level or gradient argument).
fn coord_arg_span(args: &str) -> Option<(usize, usize, u8)> {
    // Past the texture and the sampler, both plain identifiers.
    let mut at = args.find(',')? + 1;
    at += args[at..].find(',')? + 1;
    at += args[at..].len() - args[at..].trim_start().len();
    let rest = &args[at..];
    let comps = if rest.starts_with("vec2<f32>(") {
        2
    } else if rest.starts_with("vec3<f32>(") {
        3
    } else {
        return None;
    };
    // The constructor's own parentheses, balanced, so a nested call inside the coordinate
    // cannot end it early.
    let ctor = &rest["vec2<f32>".len()..];
    let mut depth = 0i32;
    for (i, b) in ctor.as_bytes().iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some((at, at + "vec2<f32>".len() + i + 1, comps));
                }
            }
            _ => {}
        }
    }
    None
}

/// >>> DOES THE STAND-IN TEXTURE VARY WITH THE COORDINATE?
///
/// ON by default; `VITASLOP_GXP_CASE_TEX=const` restores the constant stand-in, which is what
/// every measurement before 2026-09-21c used.
///
/// # Why it is the default, measured rather than argued
/// It strictly covers more - it is the only thing in the rig that can see a program computing
/// the WRONG UV - and it was run BESIDE the constant arm in one session before being made the
/// default, because it changes the value every sample feeds and a stand-in change that moved
/// the divergence count would have to be attributed before it could be trusted. It did not:
/// **const 861 exact / 150 in tolerance / 18 diverged, vary 860 / 151 / 18, with the SAME 18
/// names - none new, none fixed.** 160 of the 1,029 modules sample something, so that is 160
/// programs whose UV arithmetic is now in the comparison and was not before.
///
/// It costs about four times the GPU minute (71 s against 266 s): the coordinate expression is
/// inlined at every sample site, so the modules Tint compiles are bigger.
pub fn case_tex_varies() -> bool {
    crate::link::arm(crate::link::CASE_TEX_ARM) != Some("const")
}

/// The stand-in texture's RGBA at a coordinate, for BOTH sides of the differential.
///
/// # Why this is EXACT on both sides rather than approximately equal
/// Every operation here is either a copy, a comparison against a literal, or a multiply/add of
/// two f32 values - all of which IEEE-754 fixes exactly, so the interpreter and the emitted
/// module compute the same bits without either transcribing the other's numbers. Nothing here
/// is a texture fetch: there is no filtering, no wrap mode and no mip selection to agree about,
/// which is the whole reason a stand-in can be exact where a real binding could not.
///
/// The coordinate is CLAMPED into `[-4, 4]` and scaled by a quarter before it is added to the
/// unit's constant, for the same reason the seeded register file is tame: a coordinate computed
/// from a seeded register can be 1e38 or a denormal, and a stand-in that fed that downstream
/// would report IEEE corner cases as translation defects. The clamp is `min`/`max` against
/// literals, which is exact and leaves a NaN as one of the two literals on both sides.
///
/// Channel 3 keeps the PURE constant for an implicit-LOD sample, so a sample whose coordinate is
/// garbage still carries one channel the comparison can read.
///
/// # The LOD operand, and why it is folded in here
/// Both rigs pin ONE mip, so at a real texture `textureSample`, `textureSampleBias` and
/// `textureSampleLevel` return the same texel and no case could tell which builtin was emitted,
/// or see a level read from the wrong register. So the stand-in answers the question itself:
///
/// ```text
///   ch3 += 0.0625 * mode + tame(lod[0])     mode = 0 implicit, 1 bias, 2 level, 3 gradient
///   ch2 += tame(lod[1])   ch1 += tame(lod[2])   ch0 += tame(lod[3])
/// ```
///
/// where `lod` is the channels of `src2` the mode reads (one for a bias or a level; `ddx.xy`,
/// `ddy.xy` for a gradient) and zero elsewhere. An implicit sample adds exact zeros, so its value
/// is bit-identical to the plain stand-in. Every term is a clamp against literals times a power of
/// two, so a fused multiply-add computes the same bits as the unfused form on either side.
///
/// What this gives up is the SAMPLER's reading of the level - which mip a bias of 0.5 selects.
/// That is the hardware's arithmetic, not the translation's; what the recompiler decides (which
/// builtin, which register, which channel in which slot) is what this checks.
pub fn case_texture_value_at(unit: u8, coord: [f32; 3], lod: crate::interp::TexLodArg) -> [f32; 4] {
    let k = case_texture_value(unit);
    // NOT `clamp`, and this is not a style choice: `f32::clamp` PROPAGATES a NaN while
    // `max().min()` returns a bound - and WGSL's `min(max(...))`, which the emitted helper
    // spells, returns a bound too. Taking clippy's suggestion here would make the reference
    // disagree with the module on exactly the coordinate no comparison can read.
    #[allow(clippy::manual_clamp)]
    let tame = |v: f32| v.max(-4.0).min(4.0) * 0.25;
    let l = lod.args.map(tame);
    let km = k[3] + 0.0625 * case_tex_lod_tag(lod.mode) as f32;
    [
        k[0] + tame(coord[0]) + l[3],
        k[1] + tame(coord[1]) + l[2],
        k[2] + tame(coord[2]) + l[1],
        km + l[0],
    ]
}

/// The stand-in's tag for a LOD mode - the number the WGSL twin receives as `mode`.
pub fn case_tex_lod_tag(mode: TexLod) -> u32 {
    match mode {
        TexLod::Implicit => 0,
        TexLod::Bias => 1,
        TexLod::Level => 2,
        TexLod::Gradient => 3,
    }
}

/// The constant RGBA the stand-in texture at `unit` returns, for BOTH sides of the differential.
///
/// A plain function of the unit so the interpreter's fetcher and the emitted module agree
/// without either transcribing the other's numbers. The values are tame and distinct per unit
/// and per channel, for the same reason the seeded register file is: a divergence should mean
/// a translation defect, not an IEEE corner case.
pub fn case_texture_value(unit: u8) -> [f32; 4] {
    let u = unit as f32;
    [
        0.125 + 0.0625 * u,
        0.375 - 0.03125 * u,
        0.625 + 0.015625 * u,
        0.5 + 0.0078125 * u,
    ]
}

/// Wrap an emitted body into a COMPUTE module that runs the program once over a register file
/// supplied in a storage buffer and writes the resulting register file back out.
///
/// # Why this exists
///
/// [`wrap_module`] and [`wrap_vertex_module`] prove an emitted body COMPILES. They cannot say
/// it COMPUTES THE RIGHT NUMBERS - and every graphics defect this project has chased one title
/// at a time was a wrong number, not a refused module. This wrapper closes that gap: the same
/// body, run on the real GPU through the real browser shader compiler, against the same inputs
/// [`crate::interp`] evaluates on the CPU. Two independent implementations of the same USSE
/// semantics (a string emitter and a numeric evaluator, written separately) disagreeing is a
/// defect in one of them, and it surfaces over the WHOLE corpus in one run rather than when
/// some title happens to draw the affected pair.
///
/// The buffer layout is flat and fixed so the runner needs no per-case metadata:
///
/// * input  `gxp_case_in`  - `pa[0..N]` then `sa[0..N]`, raw 32-bit lane bits.
/// * output `gxp_case_out` - `r[0..N]`, `o[0..N]`, `i[0..N]` then `pa[0..N]`, raw 32-bit bits,
///
/// >>> AND `pa` IS AN OUTPUT BANK, WHICH IS NOT A CURIOSITY. A fragment program's colour does
/// not have to land in `o`: `ColorOutput::NonNativePa` is a real shape and the module builder
/// reads the result out of `pa` for it. While this wrapper wrote only three banks, such a
/// program's ENTIRE effect was invisible - MEASURED as **176 of 1,025 cases whose expectation
/// was an all-zero register file**, 174 of them fragment programs, most of them one to four
/// instructions. They passed, and they checked nothing.
///
/// `pa` starts SEEDED rather than zero, so its baseline is the input rather than zero and the
/// case carries the lanes the program CHANGED. See `execcases::changed_pairs`.
///
/// with `N` = [`CASE_BANK_LANES`]. Lanes are BITS, not floats, because the register file is a
/// union of float and integer views (the bitwise ops read the integer one) and a comparison
/// that went through an `f32` would lose exactly the NaN payloads a pack/unpack bug produces.
///
/// The wrapper declares the pipeline lets a body may reference (`gxp_front_facing`, the depth
/// pair) as plain locals: a compute dispatch has no facing and no fragment depth, so a program
/// that genuinely depends on either is not a case this harness can judge, and the caller
/// excludes it rather than comparing against a substituted value.
pub fn wrap_compute_module(body: &str) -> String {
    wrap_compute_module_for(body, ProgramKind::Fragment, &[]).0
}

/// [`wrap_compute_module`], told the program's KIND - which names its sampler bindings - and the
/// guest-memory WINDOWS its 0xE8 loads read.
///
/// The windows' BYTES are not passed and are not baked in: they travel as DATA in the
/// `gxp_mem` storage binding this declares, uploaded by the runner from the caller's own slice
/// - so they exist in exactly one place and the reference interpreter is handed that same
/// slice rather than a second generator that would have to agree with this one. Baking them in
/// as literals is what this used to do, and it cost a 14-second corpus run 162 seconds in
/// Tint's compile (384 `vec4` assignments for one 6 KB window). The `mem_words` parameter
/// outlived that change by a session, unused, as a warning.
///
/// Returns the module and the sampler units whose samples were replaced by the constant stand-in
/// (see [`substitute_constant_samples`]).
pub fn wrap_compute_module_for(
    body: &str,
    kind: ProgramKind,
    mem_windows: &[crate::module::MemWindow],
) -> (String, Vec<u8>) {
    wrap_compute_module_facing(body, kind, mem_windows, true)
}

/// >>> [`wrap_compute_module_for`] WITH THE FACING FLAG THE CASE DECLARES.
///
/// `GLOBAL[16]` is the per-fragment FACING bit, and both rigs PIN it rather than reading the
/// real `@builtin(front_facing)`: neither side models a rasteriser, so a real facing would put
/// a value in the comparison that the reference cannot compute. Pinned, it is an ordinary case,
/// and 18 corpus programs read it.
///
/// >>> BUT EVERY ONE OF THOSE 18 RAN FRONT-FACING ONLY, so whichever way each program's facing
/// test branches, the other arm was never executed by anything. A SECOND case at `false` runs
/// it - and it is a second case rather than a second draw because the value is a pinned constant
/// on both sides, so there is nothing a draw would add that a constant does not already say.
///
/// What this still does NOT check is that the SHIPPED module wires `@builtin(front_facing)`
/// through correctly, because neither rig reads the builtin at all. That was true before this
/// existed and is unchanged by it.
pub fn wrap_compute_module_facing(
    body: &str,
    kind: ProgramKind,
    mem_windows: &[crate::module::MemWindow],
    facing: bool,
) -> (String, Vec<u8>) {
    let (body, units) = substitute_constant_samples(body, kind);
    let mut m = String::new();
    let n = CASE_BANK_LANES;
    // A program with 0xE8 loads resolves an ADDRESS through the bound windows. The helper is
    // the module builder's own, so the address arithmetic under test is the shipped one.
    // WGSL has ONE global namespace, so a linked pair that loads memory in both stages needs two
    // helpers with two names - and the fragment side's is `gxp_fmem`. The body decides which it
    // calls, so the binding is named after what the body actually references rather than after
    // the program kind: getting that wrong emits a helper nothing calls and leaves the call
    // unresolved, which fails the whole command buffer.
    let mem_binding = if body.contains("gxp_fmem_word") { "gxp_fmem" } else { "gxp_mem" };
    if !mem_windows.is_empty() {
        // >>> BOUND, NOT BAKED. Writing the window's words in as literals put 384 `vec4`
        // assignments at the top of a module for a 6 KB window, and Tint's compile of those took
        // a 14-second corpus run to 162 - with single batches at 42 and 48 seconds. The words
        // are the same either way, so they travel as data.
        let _ = writeln!(
            m,
            "@group(0) @binding(2) var<storage, read> {mem_binding}: array<vec4<u32>>;"
        );
        m.push_str(&crate::module::mem_window_helper_named(mem_windows, mem_binding));
    }
    // The stand-in sampler: one constant per unit, the same values [`case_texture_value`] hands
    // the reference interpreter.
    if !units.is_empty() {
        let varies = case_tex_varies();
        let sig = if varies { "unit: u32, c: vec3<f32>" } else { "unit: u32" };
        // The tame coordinate, computed ONCE and spelled exactly as
        // `case_texture_value_at` computes it - min/max against literals, then a quarter.
        let _ = writeln!(m, "fn gxp_case_tex({sig}) -> vec4<f32> {{");
        if varies {
            let _ = writeln!(
                m,
                "  let t = min(max(c, vec3<f32>(-4.0)), vec3<f32>(4.0)) * 0.25;"
            );
        }
        let _ = writeln!(m, "  switch unit {{");
        for &u in &units {
            let v = case_texture_value(u);
            let body = if varies {
                format!(
                    "vec4<f32>({:?} + t.x, {:?} + t.y, {:?} + t.z, {:?})",
                    v[0], v[1], v[2], v[3]
                )
            } else {
                format!("vec4<f32>({:?}, {:?}, {:?}, {:?})", v[0], v[1], v[2], v[3])
            };
            let _ = writeln!(m, "    case {u}u: {{ return {body}; }}");
        }
        let _ = writeln!(m, "    default: {{ return vec4<f32>(0.0); }}");
        let _ = writeln!(m, "  }}\n}}");
        // The LOD-carrying form, spelled term for term as `case_texture_value_at` adds them.
        if varies {
            let _ = writeln!(
                m,
                "fn gxp_case_texl(unit: u32, c: vec3<f32>, mode: u32, lod: vec4<f32>) -> vec4<f32> {{\n\
                 \x20 let b = gxp_case_tex(unit, c);\n\
                 \x20 let l = min(max(lod, vec4<f32>(-4.0)), vec4<f32>(4.0)) * 0.25;\n\
                 \x20 let km = b.w + 0.0625 * f32(mode);\n\
                 \x20 return vec4<f32>(b.x + l.w, b.y + l.z, b.z + l.y, km + l.x);\n\
                 }}"
            );
        }
    }
    let _ = writeln!(m, "@group(0) @binding(0) var<storage, read> gxp_case_in: array<u32>;");
    let _ = writeln!(
        m,
        "@group(0) @binding(1) var<storage, read_write> gxp_case_out: array<u32>;"
    );
    for bank in ["r", "pa", "sa", "o", "i"] {
        let _ = writeln!(m, "var<private> {bank}: array<u32, {BANK_REGS}>;");
    }
    let _ = writeln!(m, "var<private> p: array<bool, 4>;");
    let _ = writeln!(m, "var<private> idx: array<i32, 2>;");
    let _ = writeln!(m, "\n@compute @workgroup_size(1)\nfn cs_main() {{");
    let _ = writeln!(
        m,
        "  for (var n: u32 = 0u; n < {n}u; n = n + 1u) {{ pa[n] = gxp_case_in[n]; sa[n] = gxp_case_in[{n}u + n]; }}"
    );
    // The bound window's bytes, then its guest base address into the SA register the driver
    // places it in - the same two steps, in the same order, the shipped module builder emits.
    // The base must land AFTER the seeded register load or the seed would overwrite it.
    for (i, win) in mem_windows.iter().enumerate() {
        let _ = writeln!(m, "  sa[{}] = {mem_binding}[{i}u].x;", win.base_sa);
    }
    // A body emitted for a FRAGMENT program may read either of these. Neither exists in a
    // compute dispatch; they are declared so such a body still COMPILES here (the caller
    // excludes any program whose result depends on one - see the module doc).
    let _ = writeln!(m, "  let gxp_front_facing: bool = {facing};");
    let _ = writeln!(m, "  let gxp_interp_depth: f32 = 0.0;");
    let _ = writeln!(m, "  var gxp_frag_depth: f32 = gxp_interp_depth;");
    m.push_str(&body);
    let _ = writeln!(
        m,
        "  for (var n: u32 = 0u; n < {n}u; n = n + 1u) {{ gxp_case_out[n] = r[n]; gxp_case_out[{n}u + n] = o[n]; gxp_case_out[{}u + n] = i[n]; gxp_case_out[{}u + n] = pa[n]; }}",
        n * 2,
        n * 3
    );
    // Keep the declared-but-unread locals live: WGSL does not warn, but a future emitter change
    // that stops reading them must not silently turn this wrapper into a different program.
    let _ = writeln!(m, "  if (gxp_front_facing && gxp_frag_depth < -1.0e30) {{ gxp_case_out[0] = 1u; }}");
    let _ = writeln!(m, "}}");
    (add_half_helpers(m), units)
}

// =====================================================================================
// >>> THE FRAGMENT-STAGE CASE RIG
//
// Everything above runs an emitted body in a COMPUTE dispatch, which is what lets one
// differential cover a thousand programs cheaply. Three families of instruction cannot be run
// there at all, and they were excluded BY NAME rather than checked - 93 of the corpus's 1,151
// blobs, every one of them shader code that ships:
//
//  * `tex.gather4` (56 blobs). Its whole point is that four NEIGHBOURING TEXELS differ, so no
//    constant-valued or coordinate-valued stand-in can represent it: it needs a real texture.
//  * `dsx`/`dsy` (8 blobs). A screen-space derivative differences a value across the
//    rasteriser's 2x2 quad, which a compute dispatch has not got.
//  * `kill` and `depthf` (29 blobs). Discard and fragment depth are pipeline state.
//
// A RENDER pipeline has all three. This wrapper puts the same emitted body in a fragment
// entry point, drawing one triangle over a 1x1 target so exactly one invocation writes the
// register file back.
//
// >>> WHAT A REAL BINDING CAN AND CANNOT CHECK, because the difference decides the whole design.
//
// A sampler's result is NOT a pure function of the coordinate at the precision this
// differential compares at. Hardware quantises the texture coordinate to a few subtexel bits
// before it selects a texel, and WebGPU's own specification permits an implementation to
// APPROXIMATE the level-of-detail computation. So a rig that fed arbitrary coordinates to a
// filtered, mipped sampler and held the GPU to a CPU model of it would report the device's
// permitted freedom as a translation defect - on a corpus whose coordinates come from a seeded
// register file and therefore land on texel boundaries by chance.
//
// This rig pins the ONE configuration that is exact, and says plainly what that leaves out:
//
//  * NEAREST filtering, CLAMP-TO-EDGE addressing, ONE mip level. The sampled value is then the
//    texel at `floor(uv * size)` with the index clamped - an exact integer selection with no
//    interpolation, no wrap arithmetic and no level to choose.
//  * the coordinate is QUANTISED TO A TEXEL POSITION by [`CASE_UV_FNS`] before it reaches the
//    sampler, on both sides, so the selection is a quarter of a texel away from any boundary
//    the hardware's subtexel rounding could tip. Without it a coordinate that landed on a
//    boundary would flip a texel and the difference would be reported as a defect.
//  * FILTERING, WRAP MODES and MIP SELECTION are consequently NOT checked by this rig, and
//    cannot be by any rig that compares bit patterns. They are also not what this differential
//    is for: they are sampler state the runtime sets, not arithmetic the recompiler emits.
//    What IS checked is everything the recompiler decides - which unit, which coordinate
//    components in which order, the gather's footprint and its bilinear coefficients, and where
//    the four returned channels land.
//
// >>> AND THE UV SENSITIVITY IS COARSER HERE THAN ON THE STAND-IN, which is why the stand-in
// stays. Quantising to one of `CASE_TEX_SIZE` positions per axis means a UV error smaller than
// one texel is invisible, while [`case_texture_value_at`] varies continuously and sees every
// bit. So an ordinary sampling program stays on the compute rig, and only a program that needs
// a fragment stage comes here.
// =====================================================================================

/// The width and height of every stand-in texture the render rig binds, in texels.
///
/// A power of two, so `(i + k) / CASE_TEX_SIZE` and the sampler's own `uv * size` are EXACT
/// float operations and the coordinate the reference quantises to is the coordinate the
/// hardware receives, bit for bit.
pub const CASE_TEX_SIZE: u32 = 64;

/// How many sampler units the rig's texture set covers. The GXM unit numbering is 4 bits.
pub const CASE_TEX_UNITS: usize = 16;

/// One texel of the render rig's stand-in texture set, as its four stored bytes.
///
/// >>> NEIGHBOURING TEXELS MUST DIFFER, which is the one property a gather needs and the one
/// property a stand-in cannot have. An avalanche of `(unit, x, y)` gives it without any
/// structure a program could accidentally satisfy.
///
/// The bytes travel to the runner as a FILE (`casetex.bin`), not as a second generator to keep
/// in step - the same discipline the guest-memory windows follow.
/// `layer` is 0 for the flat 2D texture of a unit and 1..=6 for the six faces of its CUBE, so
/// one avalanche covers both sets and no two of them can collide.
pub fn case_texel(unit: u8, layer: u32, x: u32, y: u32) -> [u8; 4] {
    let mut h: u32 = 2166136261;
    for b in [u32::from(unit), layer, x, y] {
        h = (h ^ b).wrapping_mul(16777619);
        h ^= h >> 13;
    }
    h.to_le_bytes()
}

/// How many layers the texel file carries per unit: the flat texture and the cube's six faces.
pub const CASE_TEX_LAYERS: u32 = 7;

/// Every unit's texels, in the layout the runner uploads: the whole FLAT set first (unit-major,
/// row-major RGBA8), then the whole CUBE set (unit-major, then face 0..5, then row-major).
///
/// Two blocks rather than seven interleaved layers per unit, because the runner uploads them as
/// two different kinds of texture and a contiguous block per kind is one `subarray` each.
pub fn case_tex_bytes() -> Vec<u8> {
    let n = CASE_TEX_SIZE;
    let per = (n * n * 4) as usize;
    let mut out = Vec::with_capacity(CASE_TEX_UNITS * per * CASE_TEX_LAYERS as usize);
    let plane = |unit: usize, layer: u32, out: &mut Vec<u8>| {
        for y in 0..n {
            for x in 0..n {
                out.extend_from_slice(&case_texel(unit as u8, layer, x, y));
            }
        }
    };
    for unit in 0..CASE_TEX_UNITS {
        plane(unit, 0, &mut out);
    }
    // A cube's six faces are contiguous PER UNIT, so the runner uploads one unit's whole cube in
    // a single `writeTexture` of depth 6 rather than six of depth 1.
    for unit in 0..CASE_TEX_UNITS {
        for face in 0..6 {
            plane(unit, face + 1, &mut out);
        }
    }
    out
}

/// The texel index one coordinate component names, and the fraction left over inside it.
///
/// The Rust twin of `gxp_case_ti` in [`CASE_UV_FNS`]; every step is a copy, a comparison
/// against a literal, a multiply by a power of two or a `floor`, all of which IEEE-754 fixes
/// exactly, so the two compute the same bits.
///
/// A NaN is mapped to zero EXPLICITLY rather than left to `min`/`max`, whose result WGSL leaves
/// indeterminate when an operand is NaN - and a coordinate computed from a seeded register is
/// a NaN often enough for that to decide cases.
pub fn case_tex_index_frac(c: f32) -> (u32, f32) {
    let s = if c.is_nan() { 0.0 } else { c };
    #[allow(clippy::manual_clamp)]
    let t = (s.max(-4.0).min(4.0) + 4.0) * 0.125;
    let fi = t * CASE_TEX_SIZE as f32;
    let fl = fi.floor();
    let i = fl.min((CASE_TEX_SIZE - 1) as f32);
    (i as u32, fi - fl)
}

/// The texel at `(x, y)` of `unit` as the sampler delivers it: each stored byte divided by 255.
///
/// The unorm decode is `byte / 255`, correctly rounded, which is what a WebGPU `rgba8unorm`
/// fetch produces and what an f32 division of two exact values produces - the same number, not
/// two numbers within a tolerance.
pub fn case_texel_value(unit: u8, layer: u32, x: u32, y: u32) -> [f32; 4] {
    let t = case_texel(unit, layer, x.min(CASE_TEX_SIZE - 1), y.min(CASE_TEX_SIZE - 1));
    [
        f32::from(t[0]) / 255.0,
        f32::from(t[1]) / 255.0,
        f32::from(t[2]) / 255.0,
        f32::from(t[3]) / 255.0,
    ]
}

/// What a NEAREST sample of the rig's texture returns for a shader coordinate - the reference's
/// side of every `Op::Tex` in a render case.
pub fn case_render_sample(unit: u8, coord: [f32; 4]) -> [f32; 4] {
    let (x, _) = case_tex_index_frac(coord[0]);
    let (y, _) = case_tex_index_frac(coord[1]);
    case_texel_value(unit, 0, x, y)
}

/// What a NEAREST sample of the rig's CUBE texture returns for a shader coordinate.
///
/// >>> A CUBE COORDINATE IS A DIRECTION, and which face it names is the hardware's decision -
/// so the rig does not hand it an arbitrary one. `gxp_case_uv3c` turns the program's three
/// components into a FACE and a texel position, then builds the direction that selects exactly
/// that face and that texel, and this computes the same pair directly. The reference therefore
/// never has to model the major-axis selection or the face's `u = 0.5 * (sc / |ma| + 1)`: the
/// direction it constructs makes both exact, with the major component exactly +-1 and the other
/// two at most `1 - 1/size` - so no tie is possible and the recovered `u` is the one that went
/// in, bit for bit (every value is a small multiple of `1/size`, which f32 holds exactly).
///
/// What this gives up is the face-selection ARITHMETIC itself, which is the hardware's and not
/// this translation's. What it keeps is everything the recompiler decides: that all three
/// components reach the sampler, in order, and that the four channels land where they should.
pub fn case_render_sample_cube(unit: u8, coord: [f32; 4]) -> [f32; 4] {
    let (x, _) = case_tex_index_frac(coord[0]);
    let (y, _) = case_tex_index_frac(coord[1]);
    let (z, _) = case_tex_index_frac(coord[2]);
    case_texel_value(unit, z % 6 + 1, x, y)
}

/// The quantised GATHER coordinate, as the shader's `gxp_case_uv2g` computes it. The footprint
/// below is derived from this same uv rather than from a second reading of the rule.
fn case_gather_uv(c: f32) -> f32 {
    let (i, f) = case_tex_index_frac(c);
    (i as f32 + 0.75 + f * 0.5) * (1.0 / CASE_TEX_SIZE as f32)
}

/// What `textureGather` returns for a shader coordinate, and the two bilinear fractions the
/// instruction's coefficients are built from.
///
/// >>> THE FOOTPRINT AND THE FRACTIONS COME FROM ONE QUANTISED uv, and the quantiser
/// deliberately lands it a quarter of a texel from the boundary `floor(uv * size - 0.5)` turns
/// on - the gather's own rounding - while leaving the FRACTION free to move over `[0.25, 0.75]`
/// so the bilinear coefficients still vary with the program's coordinate. A quantiser that
/// pinned the sample to a texel CENTRE (which is right for a nearest sample) would make every
/// coefficient the constant 0.75 and check nothing.
///
/// The returned order is the platform's: the texels at `(x0,y1)`, `(x1,y1)`, `(x1,y0)`,
/// `(x0,y0)`. `emit_tex_gather` states the same order and pairs coefficient `k` with texel
/// `3 - k`; both are read from that one statement of the rule.
pub fn case_render_gather(unit: u8, coord: [f32; 2]) -> ([f32; 4], [f32; 2]) {
    let n = CASE_TEX_SIZE as f32;
    let (ux, uy) = (case_gather_uv(coord[0]), case_gather_uv(coord[1]));
    // The emitted statement is `fract(uv * vec2<f32>(textureDimensions(t, 0u)) - vec2<f32>(0.5))`,
    // and the footprint the sampler takes is the same expression's floor. Spelled in that order
    // here so the two sides round identically.
    let (sx, sy) = (ux * n - 0.5, uy * n - 0.5);
    let (bx, by) = (sx.floor(), sy.floor());
    let (fx, fy) = (sx - bx, sy - by);
    let at = |x: f32, y: f32| {
        #[allow(clippy::manual_clamp)]
        let cl = |v: f32| v.max(0.0).min((CASE_TEX_SIZE - 1) as f32) as u32;
        // Gather reads ONE component, and `emit_tex_gather` asks for component 0.
        case_texel_value(unit, 0, cl(x), cl(y))[0]
    };
    (
        [
            at(bx, by + 1.0),
            at(bx + 1.0, by + 1.0),
            at(bx + 1.0, by),
            at(bx, by),
        ],
        [fx, fy],
    )
}

/// The coordinate quantisers the render rig's modules carry - the WGSL twins of
/// [`case_tex_index_frac`] and `case_gather_uv`. `CASE_TEX_SIZEf` is substituted with the
/// texture size as a float literal, so the constant exists once.
pub const CASE_UV_FNS: &str = "
fn gxp_case_ti(c: f32) -> vec2<f32> {
  let s = select(c, 0.0, c != c);
  let t = (min(max(s, -4.0), 4.0) + 4.0) * 0.125;
  let fi = t * CASE_TEX_SIZEf;
  let fl = floor(fi);
  return vec2<f32>(min(fl, CASE_TEX_SIZEf - 1.0), fi - fl);
}
fn gxp_case_uv2(c: vec2<f32>) -> vec2<f32> {
  let a = gxp_case_ti(c.x);
  let b = gxp_case_ti(c.y);
  return vec2<f32>((a.x + 0.5) * (1.0 / CASE_TEX_SIZEf), (b.x + 0.5) * (1.0 / CASE_TEX_SIZEf));
}
fn gxp_case_uv2g(c: vec2<f32>) -> vec2<f32> {
  let a = gxp_case_ti(c.x);
  let b = gxp_case_ti(c.y);
  return vec2<f32>((a.x + 0.75 + a.y * 0.5) * (1.0 / CASE_TEX_SIZEf),
                   (b.x + 0.75 + b.y * 0.5) * (1.0 / CASE_TEX_SIZEf));
}
fn gxp_case_uv3c(c: vec3<f32>) -> vec3<f32> {
  let a = gxp_case_ti(c.x);
  let b = gxp_case_ti(c.y);
  let f = gxp_case_ti(c.z);
  let sc = 2.0 * ((a.x + 0.5) * (1.0 / CASE_TEX_SIZEf)) - 1.0;
  let tc = 2.0 * ((b.x + 0.5) * (1.0 / CASE_TEX_SIZEf)) - 1.0;
  switch (u32(f.x) % 6u) {
CASE_CUBE_SWITCH  }
}
";

/// One term of a cube face's direction: a signed `1`, `sc` or `tc`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CubeTerm {
    One,
    Sc,
    Tc,
}

/// >>> THE INVERSE OF THE CUBE MAPPING, STATED ONCE. For each face, the direction whose major
/// axis selects that face and whose face coordinates are exactly `(sc, tc)`.
///
/// The forward rule - which face a direction names, and the `sc`/`tc` it yields - is the
/// hardware's, and is the same in every graphics API:
///
/// ```text
///   +X: sc = -z  tc = -y  ma = x      -X: sc = +z  tc = -y  ma = x
///   +Y: sc = +x  tc = +z  ma = y      -Y: sc = +x  tc = -z  ma = y
///   +Z: sc = +x  tc = -y  ma = z      -Z: sc = -x  tc = -y  ma = z
///   u = 0.5 * (sc / |ma| + 1)         v = 0.5 * (tc / |ma| + 1)
/// ```
///
/// This table is that rule solved for the direction, with `|ma| = 1`. The WGSL helper is
/// GENERATED from it and `the_cube_direction_selects_the_face_and_texel_it_names` checks it
/// against the forward rule written out independently - so there is one statement of the
/// inverse and it is tested, rather than two transcriptions that agree until they do not.
const CASE_CUBE_FACES: [[(i8, CubeTerm); 3]; 6] = [
    [(1, CubeTerm::One), (-1, CubeTerm::Tc), (-1, CubeTerm::Sc)],
    [(-1, CubeTerm::One), (-1, CubeTerm::Tc), (1, CubeTerm::Sc)],
    [(1, CubeTerm::Sc), (1, CubeTerm::One), (1, CubeTerm::Tc)],
    [(1, CubeTerm::Sc), (-1, CubeTerm::One), (-1, CubeTerm::Tc)],
    [(1, CubeTerm::Sc), (-1, CubeTerm::Tc), (1, CubeTerm::One)],
    [(-1, CubeTerm::Sc), (-1, CubeTerm::Tc), (-1, CubeTerm::One)],
];

/// The direction that selects `face` at face coordinates `(sc, tc)` - the Rust twin of the
/// generated `gxp_case_uv3c` switch, read from the same table.
pub fn case_cube_direction(face: usize, sc: f32, tc: f32) -> [f32; 3] {
    let mut d = [0.0f32; 3];
    for (k, (sign, term)) in CASE_CUBE_FACES[face % 6].iter().enumerate() {
        let v = match term {
            CubeTerm::One => 1.0,
            CubeTerm::Sc => sc,
            CubeTerm::Tc => tc,
        };
        d[k] = f32::from(*sign) * v;
    }
    d
}

/// The `switch` arms of `gxp_case_uv3c`, generated from [`CASE_CUBE_FACES`].
fn case_cube_switch() -> String {
    let mut out = String::new();
    for (f, face) in CASE_CUBE_FACES.iter().enumerate() {
        let terms: Vec<String> = face
            .iter()
            .map(|(sign, term)| {
                let neg = if *sign < 0 { "-" } else { "" };
                match term {
                    CubeTerm::One => format!("{neg}1.0"),
                    CubeTerm::Sc => format!("{neg}sc"),
                    CubeTerm::Tc => format!("{neg}tc"),
                }
            })
            .collect();
        let label = if f == 5 { "default".to_string() } else { format!("case {f}u") };
        let _ = writeln!(out, "    {label}: {{ return vec3<f32>({}); }}", terms.join(", "));
    }
    out
}

/// [`CASE_UV_FNS`] with the texture size substituted in.
pub fn case_uv_fns() -> String {
    CASE_UV_FNS
        .replace("CASE_CUBE_SWITCH", &case_cube_switch())
        .replace("CASE_TEX_SIZEf", &format!("{:?}", CASE_TEX_SIZE as f32))
}

/// What a render case's wrapper had to rewrite, so the caller can require that it rewrote
/// EVERYTHING the program contains rather than trust a textual parse.
///
/// A sample left unrewritten is not a compile error - it is a real `textureSample` on an
/// unquantised coordinate, which would pass through boundary rounding and report the device as
/// wrong. The counts are checked against the instruction stream by the case writer.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RenderRewrites {
    pub samples: usize,
    pub gathers: usize,
    pub kills: usize,
}

/// Wrap an emitted FRAGMENT body into a RENDER module whose fragment entry runs it once and
/// writes the register file back, with REAL texture bindings.
///
/// See the section comment above for what this rig can and cannot check. The output buffer
/// extends the compute rig's four banks with two words:
///
/// ```text
///   [4N]     1 if the program executed a `kill`, else 0
///   [4N + 1] the fragment depth, as bits
/// ```
///
/// >>> `kill` IS MODELLED, NOT EXECUTED, AND THAT IS THE ONLY HONEST CHOICE. WGSL's `discard`
/// demotes the invocation to a helper, and a helper's stores DO NOT LAND - so a case whose
/// program discards would read its output buffer back untouched and compare as "the GPU
/// computed zero" on every lane. The wrapper rewrites the discard into "record the kill, write
/// the register file, leave the shader", which is the state the hardware's pixel ends with, and
/// the reference's `Op::Kill` arm ends its walk at the same instruction.
///
/// >>> THE DEPTH REMAP IS PINNED TO ITS DEFAULT ARM. `gxp_depth_to_window` in a shipped module
/// chooses between four forward maps from a per-draw uniform the renderer fills; which one is
/// in force is a property of the DRAW, not of the translation, so the rig defines the helper as
/// the default (`range.w >= 2.5`) arm - a clamp - and the reference applies the same clamp.
/// What is under test is the value the program computed, not which map the renderer picked.
///
/// Returns `None` when a sampled unit is not a plain 2D float texture: a cube, a 3D or a raw
/// integer binding needs a texture set and a coordinate quantiser this rig does not have, and
/// emitting the module anyway would bind something else's texture. The caller skips the blob
/// and the skip is counted.
pub fn wrap_render_case_module_for(
    body: &str,
    kind: ProgramKind,
    mem_windows: &[crate::module::MemWindow],
    units: &[TexBinding],
) -> Result<(String, RenderRewrites), &'static str> {
    wrap_render_case_module_facing(body, kind, mem_windows, units, true)
}

/// [`wrap_render_case_module_for`] with the facing flag the case declares - see
/// [`wrap_compute_module_facing`], which makes the same substitution for the same reason.
pub fn wrap_render_case_module_facing(
    body: &str,
    kind: ProgramKind,
    mem_windows: &[crate::module::MemWindow],
    units: &[TexBinding],
    facing: bool,
) -> Result<(String, RenderRewrites), &'static str> {
    wrap_render_case_module_ramped(body, kind, mem_windows, units, facing, false)
}

/// The PA lane the render rig's optional SCREEN RAMP rides on - high enough that no program
/// under test reads it by accident, low enough that a doubled six-bit operand field can name it.
pub const CASE_RAMP_LANE: usize = 100;
/// The ramp's step per pixel in x and y. Powers of two far below the seeded lane's own
/// magnitude, so `x + step` stays in `x`'s binade and is EXACT, and the quad's difference is
/// exactly the step (Sterbenz) whether the device takes a coarse or a fine derivative.
pub const CASE_RAMP_DX: f32 = 1.0 / 1024.0;
pub const CASE_RAMP_DY: f32 = 1.0 / 2048.0;

/// [`wrap_render_case_module_facing`], optionally with the SCREEN RAMP on [`CASE_RAMP_LANE`] -
/// the one input that varies across the quad, so a derivative has something to measure. See
/// `interp::RegFile::ramp` for what the reference does with it. The ramp is zero at the real
/// pixel (`pos = (0.5, 0.5)`), so no value the program reads there changes.
pub fn wrap_render_case_module_ramped(
    body: &str,
    kind: ProgramKind,
    mem_windows: &[crate::module::MemWindow],
    units: &[TexBinding],
    facing: bool,
    ramp: bool,
) -> Result<(String, RenderRewrites), &'static str> {
    // NAME THE KIND. A census that says "an unsupported texture" 35 times tells a reader
    // nothing about what to build next, while naming the dimensionality turns the same census
    // into a ranked work list - the same rule the interpreter's refusals follow.
    if let Some(b) = units.iter().find(|b| b.raw || (b.coords >= 3 && !b.cube)) {
        return Err(if b.raw { "a raw 64-bit integer texture" } else { "a 3D texture" });
    }
    let (body, rewrites) = rewrite_body_for_render(body, kind);
    let n = CASE_BANK_LANES;
    let mut m = String::new();

    let mem_binding = if body.contains("gxp_fmem_word") { "gxp_fmem" } else { "gxp_mem" };
    if !mem_windows.is_empty() {
        let _ = writeln!(
            m,
            "@group(0) @binding(2) var<storage, read> {mem_binding}: array<vec4<u32>>;"
        );
        m.push_str(&crate::module::mem_window_helper_named(mem_windows, mem_binding));
    }
    let _ = writeln!(m, "@group(0) @binding(0) var<storage, read> gxp_case_in: array<u32>;");
    let _ = writeln!(
        m,
        "@group(0) @binding(1) var<storage, read_write> gxp_case_out: array<u32>;"
    );
    // Textures from binding 4 up, leaving 2 to the guest-memory window and 3 to the trace
    // buffer the compute rig uses - one binding numbering across both rigs, so a reader of the
    // runner does not have to hold two.
    for (k, b) in units.iter().enumerate() {
        let (tex, samp) = sampler_names(kind, b.unit);
        // The binding's TYPE is the one the shipped pipeline builder would declare
        // ([`TexBinding::wgsl_type`]), because the body was emitted against it: a cube sampled
        // with three components must be a `texture_cube` or the call does not type-check.
        let _ = writeln!(m, "@group(0) @binding({}) var {tex}: {};", 4 + 2 * k, b.wgsl_type());
        let _ = writeln!(m, "@group(0) @binding({}) var {samp}: sampler;", 5 + 2 * k);
    }
    for bank in ["r", "pa", "sa", "o", "i"] {
        let _ = writeln!(m, "var<private> {bank}: array<u32, {BANK_REGS}>;");
    }
    let _ = writeln!(m, "var<private> p: array<bool, 4>;");
    let _ = writeln!(m, "var<private> idx: array<i32, 2>;");
    // The two fragment-stage outputs. `gxp_frag_depth` is module scope rather than a local
    // because the emitted body assigns to it by name and the epilogue below must read it.
    let _ = writeln!(m, "var<private> gxp_killed: u32 = 0u;");
    let _ = writeln!(m, "var<private> gxp_frag_depth: f32 = 0.0;");
    if !units.is_empty() {
        m.push_str(&case_uv_fns());
    }
    // See the doc above: the default forward map, so the value under test is the one the
    // program computed.
    m.push_str(
        "fn gxp_depth_to_window(d: f32, interpolated: f32) -> f32 { return clamp(d, 0.0, 1.0); }\n",
    );
    let _ = writeln!(m, "fn gxp_case_store() {{");
    let _ = writeln!(
        m,
        "  for (var n: u32 = 0u; n < {n}u; n = n + 1u) {{ gxp_case_out[n] = r[n]; gxp_case_out[{n}u + n] = o[n]; gxp_case_out[{}u + n] = i[n]; gxp_case_out[{}u + n] = pa[n]; }}",
        n * 2,
        n * 3
    );
    let _ = writeln!(m, "  gxp_case_out[{}u] = gxp_killed;", n * 4);
    let _ = writeln!(m, "  gxp_case_out[{}u] = bitcast<u32>(gxp_frag_depth);", n * 4 + 1);
    let _ = writeln!(m, "}}");
    // ONE triangle over the whole clip square. The target is 1x1, so exactly one invocation is
    // real and the other three lanes of its quad are helpers - whose stores do not land, which
    // is what keeps the single output buffer unraced.
    m.push_str(
        "@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {\n  \
         var xy = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));\n  \
         return vec4<f32>(xy[vi], 0.0, 1.0);\n}\n",
    );
    if ramp {
        let _ = writeln!(m, "@fragment fn fs_main(@builtin(position) gxp_pos: vec4<f32>) -> @location(0) vec4<f32> {{");
    } else {
        let _ = writeln!(m, "@fragment fn fs_main() -> @location(0) vec4<f32> {{");
    }
    let _ = writeln!(
        m,
        "  for (var n: u32 = 0u; n < {n}u; n = n + 1u) {{ pa[n] = gxp_case_in[n]; sa[n] = gxp_case_in[{n}u + n]; }}"
    );
    if ramp {
        let _ = writeln!(
            m,
            "  pa[{CASE_RAMP_LANE}] = bitcast<u32>(bitcast<f32>(pa[{CASE_RAMP_LANE}]) + (gxp_pos.x - 0.5) * {:?} + (gxp_pos.y - 0.5) * {:?});",
            CASE_RAMP_DX, CASE_RAMP_DY
        );
    }
    for (i, win) in mem_windows.iter().enumerate() {
        let _ = writeln!(m, "  sa[{}] = {mem_binding}[{i}u].x;", win.base_sa);
    }
    // >>> FACING AND INTERPOLATED DEPTH ARE PINNED, exactly as the compute rig pins them, and
    // for the same reason: NEITHER SIDE MODELS THEM. The reference has no facing input and no
    // interpolated depth, so reading the real `@builtin(front_facing)` here would put a value
    // in the comparison that one side cannot compute. A program whose result depends on either
    // is still not a case this harness can judge.
    let _ = writeln!(m, "  let gxp_front_facing: bool = {facing};");
    let _ = writeln!(m, "  let gxp_interp_depth: f32 = 0.0;");
    let _ = writeln!(m, "  gxp_frag_depth = gxp_interp_depth;");
    m.push_str(&body);
    let _ = writeln!(m, "  gxp_case_store();");
    let _ = writeln!(m, "  return vec4<f32>(0.0);");
    let _ = writeln!(m, "}}");
    Ok((add_half_helpers(m), rewrites))
}

/// Quantise every sample coordinate and turn every `discard` into a recorded kill.
///
/// The parse is exact rather than heuristic, like [`substitute_constant_samples`]: `emit_tex`
/// writes one `FUNC(tex, samp, vecK<f32>(...)EXTRA)` per sample and `emit_tex_gather` writes
/// one `let _guvN = vec2<f32>(...);`, and anything not of that shape is left alone and NOT
/// counted, so the caller can refuse the blob rather than ship a module that samples an
/// unquantised coordinate.
fn rewrite_body_for_render(body: &str, kind: ProgramKind) -> (String, RenderRewrites) {
    let mut n = RenderRewrites::default();
    let mut out = String::with_capacity(body.len());
    let prefix = match kind {
        ProgramKind::Vertex => "vt",
        _ => "t",
    };
    for line in body.lines() {
        // A gather's coordinate is bound to its own `let` and read twice - once by the gather
        // and once by the coefficients - so quantising it at the binding covers both.
        if line.trim_start().starts_with("let _guv")
            && let Some(eq) = line.find("= vec2<f32>(")
        {
            let (head, tail) = line.split_at(eq + 1);
            let _ = writeln!(out, "{head} gxp_case_uv2g({});", tail.trim().trim_end_matches(';'));
            n.gathers += 1;
            continue;
        }
        if line.trim() == "discard;" {
            out.push_str("  { gxp_killed = 1u; gxp_case_store(); return vec4<f32>(0.0); }\n");
            n.kills += 1;
            continue;
        }
        let rewritten = (|| {
            let eq = line.find("= texture")?;
            let call = &line[eq + 2..];
            let open = call.find('(')?;
            let func = &call[..open];
            if !matches!(
                func,
                "textureSample" | "textureSampleBias" | "textureSampleLevel" | "textureSampleGrad"
            ) {
                return None;
            }
            let args = &call[open + 1..];
            // The binding must be one this rig declared, or the rewrite would wrap a
            // coordinate for a texture nothing bound.
            let tex = args.split(',').next()?.trim();
            let _unit: u8 = tex.strip_prefix(prefix)?.parse().ok()?;
            let samp = args.split(',').nth(1)?.trim();
            let (start, end, comps) = coord_arg_span(args)?;
            // Two components go to the flat texture's quantiser and three to the cube's, which
            // is exactly the split the binding's own type makes - a 3D texture is refused
            // before this runs, so a three-component coordinate here is a cube direction.
            let quant = if comps == 3 { "gxp_case_uv3c" } else { "gxp_case_uv2" };
            let coord = &args[start..end];
            // >>> AN IMPLICIT OR BIASED SAMPLE BECOMES AN EXPLICIT LEVEL-0 ONE, AND NOT BECAUSE
            // >>> IT IS CONVENIENT.
            //
            // `textureSample` and `textureSampleBias` take the quad's derivatives, so WGSL
            // requires them in UNIFORM control flow - and this rig's prologue makes every
            // predicate non-uniform where the shipped module's is not. In the module the
            // product builds, `sa` is a UNIFORM BUFFER, so `if (!p[0])` on an `sa` value is
            // provably uniform and Tint accepts the sample inside it. Here `sa` is a
            // `var<private>` array filled per invocation from a storage buffer, so the same
            // condition is "possibly non-uniform" and Tint refuses the same sample.
            //
            // MEASURED, and the measurement is the whole point: `tintcheck` over all **596
            // shipped modules reports 0 failures**, while three of these cases were refused. So
            // the refusal was the RIG's, and "fixing" the emitter's hoisting would have been a
            // change to the product to satisfy an artefact of its test harness.
            //
            // The rig binds ONE mip level, so level 0 is the only level any of the four sample
            // forms can return - the substitution changes no value. What it gives up is the LOD
            // argument's plumbing, which a single-mip rig could not check either way, and the
            // UNIFORMITY property, which this rig cannot check at all and `tintcheck` over the
            // shipped modules can and does.
            if matches!(func, "textureSample" | "textureSampleBias") {
                return Some(format!(
                    "{}= textureSampleLevel({tex}, {samp}, {quant}({coord}), 0.0);",
                    &line[..eq]
                ));
            }
            let at = eq + 2 + open + 1;
            Some(format!(
                "{}{quant}({}){}",
                &line[..at + start],
                coord,
                &line[at + end..]
            ))
        })();
        match rewritten {
            Some(r) => {
                n.samples += 1;
                out.push_str(&r);
            }
            None => out.push_str(line),
        }
        out.push('\n');
    }
    (out, n)
}

/// The FNV-1a checksum of the whole register file, as a WGSL function and as a Rust one that
/// must agree with it bit for bit.
///
/// >>> WHY A CHECKSUM AND NOT THE REGISTERS THEMSELVES. A trace that carried the file at every
/// instruction would be `instrs * 4 * 512` words - about a megabyte for a 115-instruction
/// program, and a loop of 2,048 stores emitted per instruction, which is a module Tint spends
/// real time on. One word per checkpoint answers the question this instrument is for ("where do
/// the two sides first differ?"), and once the step is known the ordinary differential names
/// the lanes.
///
/// It hashes BIT PATTERNS in a fixed order with integer arithmetic only, so there is nothing
/// for the two implementations to round differently.
///
/// >>> AND IT COVERS THE PREDICATE AND INDEX REGISTERS, WHICH THE FIRST VERSION DID NOT.
/// A state a trace omits is a state whose disagreement the trace cannot see - and it does not
/// merely miss it, it MISLOCATES it. MEASURED on `cw-rr-corpus__frag_8668a340`: the two sides
/// disagreed about `p[0]`, set by the TEST at instruction 1, and the checksum covering only the
/// four register banks agreed at instruction 2 and first differed at 3 - so the trace named the
/// predicated MOVE, which is downstream and innocent. A predicate is one bit and it decides
/// which of two arms runs, so it is the most consequential state in the file per bit.
pub const GXP_TRACE_CK: &str = "
fn gxp_ck() -> u32 {
  var h: u32 = 2166136261u;
  for (var n: u32 = 0u; n < GXP_LANESu; n = n + 1u) {
    h = (h ^ r[n]) * 16777619u;
    h = (h ^ o[n]) * 16777619u;
    h = (h ^ i[n]) * 16777619u;
    h = (h ^ pa[n]) * 16777619u;
  }
  for (var k: u32 = 0u; k < 4u; k = k + 1u) {
    h = (h ^ select(0u, 1u, p[k])) * 16777619u;
  }
  for (var k: u32 = 0u; k < 2u; k = k + 1u) {
    h = (h ^ bitcast<u32>(idx[k])) * 16777619u;
  }
  return h;
}
";

/// The Rust twin of `gxp_ck`. The lane ORDER, the bank order and the trailing predicate and
/// index registers are all part of the agreement.
pub fn trace_checksum(
    r: &[f32],
    o: &[f32],
    i: &[f32],
    pa: &[f32],
    p: &[bool],
    idx: &[i32],
) -> u32 {
    let mut h: u32 = 2166136261;
    let at = |v: &[f32], n: usize| v.get(n).map(|x| x.to_bits()).unwrap_or(0);
    for n in 0..CASE_BANK_LANES {
        for bank in [r, o, i, pa] {
            h = (h ^ at(bank, n)).wrapping_mul(16777619);
        }
    }
    for k in 0..4 {
        h = (h ^ u32::from(p.get(k).copied().unwrap_or(false))).wrapping_mul(16777619);
    }
    for k in 0..2 {
        h = (h ^ idx.get(k).copied().unwrap_or(0) as u32).wrapping_mul(16777619);
    }
    h
}

/// Wrap a MARKED body into a compute module that records a register-file checksum at every
/// top-level instruction boundary, keyed by instruction index.
///
/// # What this is for
/// [`wrap_compute_module_for`] says WHETHER two register files agree at the end. On a
/// 183-instruction program that is not enough to fix anything: the disagreement has to be
/// located first, and reading a hundred instructions by eye is the guess-and-check this
/// project's notes keep recording as the expensive way.
///
/// # Keyed by INDEX, not by a step counter, and last write wins
/// The reference follows branches, so its execution ORDER is not the emitted order and a step
/// counter would put the two traces out of phase at the first taken branch. Indexing by the
/// instruction means both sides record "the state on last arriving at instruction k", which is
/// the same statement on both and needs no agreement about control flow. In a loop both record
/// the final iteration.
///
/// Only DEPTH-1 boundaries carry a marker ([`emit_range`]), so an instruction inside an `if`
/// or a loop body has no checkpoint of its own and the trace narrows to the enclosing top-level
/// region. That is a limit worth knowing, not a defect: it is where a body can be cut.
pub fn wrap_trace_module_for(
    marked_body: &str,
    kind: ProgramKind,
    mem_windows: &[crate::module::MemWindow],
) -> (String, Vec<u8>, Vec<usize>) {
    // The checkpoints, in place of the markers, and the indices they name.
    let mut indices = Vec::new();
    let mut body = String::with_capacity(marked_body.len());
    for line in marked_body.lines() {
        if let Some(rest) = line.strip_prefix(SPLIT_MARKER)
            && let Ok(index) = rest.trim().parse::<usize>()
        {
            indices.push(index);
            let _ = writeln!(body, "  gxp_trace[{index}u] = gxp_ck();");
            continue;
        }
        body.push_str(line);
        body.push('\n');
    }
    // One past the last instruction: the FINAL state, which is what the ordinary differential
    // compares. Without it a program whose only disagreement is in its last instruction traces
    // as identical the whole way.
    let end = indices.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let _ = writeln!(body, "  gxp_trace[{end}u] = gxp_ck();");
    indices.push(end);

    let (module, units) = wrap_compute_module_for(&body, kind, mem_windows);
    // The buffer and the checksum go in ahead of the entry point, after the directives.
    let decl = format!(
        "@group(0) @binding(3) var<storage, read_write> gxp_trace: array<u32>;\n{}",
        GXP_TRACE_CK.replace("GXP_LANES", &CASE_BANK_LANES.to_string())
    );
    let mut out = module;
    out.insert_str(crate::link::directives_end(&out), &decl);
    (out, units, indices)
}

#[allow(clippy::too_many_arguments)]
fn emit_instr(
    body: &mut String,
    instr: &Instr,
    index: usize,
    byte_offset: usize,
    kind: ProgramKind,
    shader: &Shader,
    prelude: &mut String,
    cond_start: Option<usize>,
    // The conditional regions this instruction sits inside, outermost first - see
    // [`uniform_gap`], which a derivative uses to reach uniform control flow where it stands.
    enclosing: &[Enclosing],
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
    //
    // VTSTMSK's unsigned-16-bit form is admitted for the SAME reason and reads the same
    // register through the same raw-u32 path. It is the mask-writing sibling of the `vtst`
    // above - `GLOBAL[16]` against an SA register, EQ - and it is the instruction that panicked
    // a user's run several holes into a round.
    let global_ok = matches!(
        instr.op,
        Op::Test { alu: TestAlu::BitAnd | TestAlu::BitShl | TestAlu::IntSub, .. }
            | Op::TestMask { alu: TestAlu::IntSub16U, .. }
    );
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
        emit_test_mask(s, instr, dest, alu, cmp, kind).ok_or_else(unmapped)?;
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
    // >>> A DERIVATIVE MAY NOT SIT INSIDE A PREDICATE'S `if`, AND ON HARDWARE IT DOES NOT.
    //
    // `dpdx`/`dpdy` difference a value across the rasteriser's 2x2 QUAD, so they need every
    // lane of the quad to have executed them - which is why WGSL requires them in UNIFORM
    // control flow, and why Tint REJECTS one inside an `if` on a per-pixel predicate:
    // `error: 'dpdy' must only be called from uniform control flow`. naga accepts it, so the
    // desktop ran this program for months and the browser could not compile it at all - the
    // pipeline was invalid, its command buffer with it, and the run eventually trapped
    // [[vitaslop-tint-rejects-what-naga-accepts]].
    //
    // The USSE does not predicate the DIFFERENCING either: a predicated `dsx`/`dsy` computes
    // the quad derivative regardless and only the WRITE-BACK is conditional. So hoisting the
    // call above the `if` and predicating only the store is both what the hardware does and
    // what WGSL requires - it is a correctness fix that happens to also be a portability one.
    //
    // TWO PLACES to put the hoisted call, because there are two shapes of conditional here:
    //
    //  * the instruction's OWN predicate, whose `if` this function emits a line later. Lifting
    //    the call just above that `if` is enough, and keeps it reading the register values it
    //    would have read.
    //  * an enclosing BRANCH the structuring turned into an `if`/`loop`. Nothing inside that
    //    block is in uniform control flow, so the call has to leave the block entirely - and
    //    the only place guaranteed to be uniform is the top of the function. That is exact ONLY
    //    if nothing before this instruction has written the registers it reads, so that is
    //    CHECKED, and a program that fails the check is refused by name rather than hoisted to
    //    a different value.
    let hoisted = if matches!(instr.op, Op::Dsx | Op::Dsy)
        && (cond_start.is_some() || !matches!(instr.pred, Predicate::Always))
    {
        let func = if matches!(instr.op, Op::Dsx) { "dpdx" } else { "dpdy" };
        let s1 = instr.srcs.first().ok_or_else(unmapped)?;
        // >>> WHERE THE CALL GOES, and there are two places because there are two facts.
        //
        // Nothing has rewritten the source since the region began: the call is HOISTED above
        // the region, where control flow is uniform and the register still holds the same
        // value. That is the common case and the cheap one.
        //
        // The region ITSELF computes the value being differenced: no point above the region
        // holds it, so the region is CLOSED here, the builtin called in the gap, and the region
        // re-entered - see [`uniform_gap`] for why that is what the hardware does rather than a
        // way around the WGSL rule. This case used to be refused outright, and a refused pair's
        // mesh is absent from the frame: it is three of a football title's world programs.
        let gap = match cond_start {
            Some(from) if writes_between(shader, from, index, s1) => {
                Some(uniform_gap(shader, enclosing, from, index, index)?)
            }
            _ => None,
        };
        let p = Prec::of(instr);
        let mut names: [Option<String>; 4] = [None, None, None, None];
        // The gap's close/re-enter brackets the WHOLE set of channel calls, so a four-channel
        // derivative cuts one gap and not four.
        let mut gap_calls = String::new();
        for (c, name) in names.iter_mut().enumerate() {
            if !mask[c] {
                continue;
            }
            let e = src_channel(s1, c, p).ok_or_else(unmapped)?;
            let n = format!("gxp_deriv{index}_{c}");
            let out = match (&gap, cond_start) {
                (Some(_), _) => &mut gap_calls,
                (None, Some(_)) => &mut *prelude,
                (None, None) => &mut *body,
            };
            let _ = writeln!(out, "  let {n} = {func}({e});");
            *name = Some(n);
        }
        if let Some((close, reopen)) = gap {
            body.push_str(&close);
            body.push_str(&gap_calls);
            body.push_str(&reopen);
        }
        Some(names)
    } else {
        None
    };
    let r = match instr.op {
        Op::Mul => emit_binop(s, instr, dest, mask, "*", index).ok_or_else(unmapped),
        Op::Add => emit_binop(s, instr, dest, mask, "+", index).ok_or_else(unmapped),
        Op::Min => emit_func2(s, instr, dest, mask, "min", index).ok_or_else(unmapped),
        Op::Max => emit_func2(s, instr, dest, mask, "max", index).ok_or_else(unmapped),
        Op::Frc => emit_func1(s, instr, dest, mask, "fract", index).ok_or_else(unmapped),
        // The call itself may already be hoisted above this instruction's predicate `if` -
        // see `hoisted` above. When it is, the store reads the temporary rather than
        // calling the builtin again (calling it twice would put one back inside the `if`).
        Op::Dsx | Op::Dsy => {
            let func = if matches!(instr.op, Op::Dsx) { "dpdx" } else { "dpdy" };
            match &hoisted {
                Some(names) => emit_stored(s, instr, dest, mask, names).ok_or_else(unmapped),
                None => emit_func1(s, instr, dest, mask, func, index).ok_or_else(unmapped),
            }
        }
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
        // A move and a float<->float pack are both swizzled copies in the f32 register model,
        // and so is the NORMALIZED U8 convert: the byte<->float scaling is not written here,
        // it is what `Prec::Fx8`'s own read and store already do (`unpack4x8unorm` one way,
        // a rounded `clamp(v,0,1)*255` byte insert the other). `Prec::of`/`src_of` put that
        // precision on whichever side the conversion runs into, so the copy is the whole
        // instruction.
        Op::Mov | Op::Pack { .. } | Op::PackUnorm8 { .. } | Op::CopyFx8 => {
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
        Op::PackFromInt { bits, signed } => {
            emit_pack_from_int(s, instr, dest, mask, bits, signed).ok_or_else(unmapped)
        }
        Op::PackIntCopy { bits } => {
            emit_pack_int_copy(s, instr, dest, mask, bits).ok_or_else(unmapped)
        }
        // The BYTE-WISE conditional move: one register, four independently selected bytes,
        // every operand a raw lane. There is no float view anywhere in it, which is why the
        // "cannot be represented in a float register file" refusal did not apply.
        Op::CmovU8 { test } => emit_cmov_u8(s, instr, dest, mask, test).ok_or_else(unmapped),
        // A LIMM's value is a RAW pattern - the corpus uses one as an integer sentinel and
        // another as a float - so it is stored with no view applied, like an integer pack's.
        Op::Limm { value } => s
            .store_raw(dest, 0, &format!("{value:#010x}u"))
            .ok_or_else(unmapped),
        Op::IntMad { signed, bits, src0_high, src1_high } => {
            emit_int_mad(s, instr, dest, signed, bits, src0_high, src1_high).ok_or_else(unmapped)
        }
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
            emit_mem_load(s, instr, dest, elements, offset_bytes, index, kind).ok_or_else(unmapped)
        }
        Op::LoadIndex { addend, to_index, stride } => {
            emit_load_index(s, instr, dest, addend, to_index, stride).ok_or_else(unmapped)
        }
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
    let (stmts, hoisted) = cse_unpacks(&fold_halves(stmts));
    if !staged && !hoisted {
        return stmts;
    }
    format!("  {{\n{stmts}  }}\n")
}

/// >>> AN INSTRUCTION THAT WRITES BOTH HALVES OF A 16-BIT REGISTER WRITES IT ONCE.
///
/// A half-register store has to be a read-modify-write, because the other half is a different
/// value the shader reads back separately. Emitted per channel, the commonest instruction in a
/// 16-bit fragment program - one writing `.xy` or `.xyzw` - spends two `pack2x16float` calls,
/// two masks and two ORs per register where ONE pack says the same thing: the pair overwrites
/// the whole register, so there is nothing to preserve and nothing to read back.
///
/// The fold is exact: [`HALF_PK_FN`] of a pair is by definition the low half [`HALF_LO_FN`]
/// writes beside the high half [`HALF_HI_FN`] writes - the two narrowings are independent and
/// the helpers are defined in terms of the same one.
///
/// >>> IT FOLDS ONLY TWO ADJACENT LINES, AND THAT IS WHAT MAKES IT SAFE.
///
/// Pairing the two halves anywhere in the instruction - which is the tempting version, since
/// the stores are right there in [`Dest`] - moves a store past whatever sits between them, and
/// what sits between them can be a READ of the same register: [`dest_aliases_source`] only
/// looks four registers either side of the destination, and a repeated instruction reaches
/// further than that. MEASURED: pairing structurally changed mlb's frame on 31% of its pixels.
/// Adjacent lines cannot have anything between them, so there is nothing to move past.
///
/// It is worth this care because a 16-bit program pays the pack/unpack emulation on EVERY
/// fragment: mlb's world-family blend is 54 packs and 118 unpacks per evaluation over three
/// million samples a frame, which a desktop GPU shrugs off and a phone's does not
/// [[phone-gpu-has-four-times-the-headroom]].
fn fold_halves(stmts: &str) -> String {
    // `  X[n] = gxp_hlo(X[n], LO);` followed by the HIGH half of the same register.
    let lo_open = format!(" = {HALF_LO_FN}(");
    let hi_open = format!(" = {HALF_HI_FN}(");
    let mut out = String::with_capacity(stmts.len());
    let mut lines = stmts.lines().peekable();
    while let Some(line) = lines.next() {
        let folded = (|| {
            // `  X[n] = gxp_hlo(X[n], ` - the same register on both sides, spelled the same way.
            let (dest, rest) = line.strip_prefix("  ")?.split_once(lo_open.as_str())?;
            let lo_body = rest.strip_prefix(dest)?.strip_prefix(", ")?.strip_suffix(");")?;
            let next = lines.peek()?;
            let (next_dest, next_rest) = next.strip_prefix("  ")?.split_once(hi_open.as_str())?;
            if next_dest != dest {
                return None;
            }
            let hi_body = next_rest.strip_prefix(dest)?.strip_prefix(", ")?.strip_suffix(");")?;
            Some(format!("  {dest} = {HALF_PK_FN}({lo_body}, {hi_body});"))
        })();
        match folded {
            Some(one) => {
                out.push_str(&one);
                out.push('\n');
                lines.next(); // the high half is folded into the line just written
            }
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}

/// Hoist a repeated `unpack2x16float(bank[n])` in ONE instruction into a single `let`.
///
/// A 16-bit source operand reads one HALF of a register, so a four-channel instruction over
/// two registers spells `unpack2x16float(pa[2])` four times for two distinct registers, and a
/// three-source instruction spells each of its sources' registers four times over. mlb's
/// world-family blend emits 118 unpacks per evaluation where 40-odd registers are read; the
/// rest is the same call again. A desktop compiler folds them and the run never notices; the
/// phone's does not, and this shader is three million fragments a frame
/// [[phone-gpu-has-four-times-the-headroom]].
///
/// The rule is conservative and that is what makes it exact: a register is hoisted only when
/// NOTHING in this instruction assigns it. `Dest` holds every store to the end of the
/// instruction, but a handful of emitters (the integer groups) write a whole register inline,
/// so "no assignment anywhere in the instruction" is the one test that covers both without
/// having to know which emitter ran.
fn cse_unpacks(stmts: &str) -> (String, bool) {
    const CALL: &str = "unpack2x16float(";
    // Distinct `bank[n]` arguments and how often each is unpacked.
    let mut seen: Vec<(String, usize)> = Vec::new();
    let mut rest = stmts;
    while let Some(at) = rest.find(CALL) {
        rest = &rest[at + CALL.len()..];
        let Some(close) = rest.find(')') else { break };
        let arg = &rest[..close];
        // Only a plain register read. A constant (`0x00003c00u`) is folded by any compiler and
        // an indexed or computed argument is not a stable name to key on.
        if !is_register_read(arg) {
            continue;
        }
        match seen.iter_mut().find(|(a, _)| a == arg) {
            Some((_, n)) => *n += 1,
            None => seen.push((arg.to_string(), 1)),
        }
    }
    let mut out = stmts.to_string();
    let mut lets = String::new();
    for (arg, n) in seen {
        if n < 2 || out.contains(&format!("{arg} =")) {
            continue;
        }
        let name = format!("u_{}", arg.replace(['[', ']'], ""));
        let _ = writeln!(lets, "  let {name} = unpack2x16float({arg});");
        out = out.replace(&format!("{CALL}{arg})"), &name);
    }
    if lets.is_empty() {
        return (out, false);
    }
    (format!("{lets}{out}"), true)
}

/// Whether `arg` is exactly a `bank[n]` register read - the only argument [`cse_unpacks`]
/// keys on.
fn is_register_read(arg: &str) -> bool {
    let Some((bank, idx)) = arg.split_once('[') else { return false };
    let Some(idx) = idx.strip_suffix(']') else { return false };
    matches!(bank, "r" | "o" | "pa" | "sa" | "i")
        && !idx.is_empty()
        && idx.bytes().all(|b| b.is_ascii_digit())
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

/// Does any instruction in `[from, index)` write a register `src` reads?
///
/// The question a hoist has to answer. The temporary is computed immediately ABOVE the
/// outermost enclosing block, so the hoist is exact exactly when nothing between that point
/// and the instruction changes the register it reads. Deliberately coarse - any write to the
/// same bank and register number counts, whatever the channel, and an INDEXED write counts
/// wherever it might land. A coarse "no" is safe; a coarse "yes" only costs a refusal that
/// names itself.
fn writes_between(shader: &Shader, from: usize, index: usize, src: &Operand) -> bool {
    let hi = index.min(shader.instrs.len());
    shader.instrs[from.min(hi)..hi].iter().any(|i| {
        i.dest.as_ref().is_some_and(|d| {
            if matches!(d.bank, Bank::Indexed) {
                return true;
            }
            d.bank == src.bank && d.index == src.index
        })
    })
}

/// `dest.c = <precomputed name for c>` for each written channel: the store half of an
/// instruction whose VALUE was computed before this statement. Used by the derivative ops,
/// whose builtin must be called outside any predicate `if` - see `hoisted` in `emit_instr`.
fn emit_stored(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    mask: [bool; 4],
    names: &[Option<String>; 4],
) -> Option<()> {
    let p = Prec::of(instr);
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        body.store(dest, c, names[c].as_ref()?, p)?;
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
        // >>> A 16-BIT RESULT IS HALF A REGISTER, NOT A WHOLE ONE. Two lanes share one
        // register, exactly as an F16 pair does, and the group-0x15 IMAD32s that read these
        // values back address them by (register, half) through their own `src0_high` bit. A
        // whole-register store put a skinned mesh's four blend indices in four registers where
        // its four bone fetches look in two, so half the fetches read a register nothing had
        // written and the other half were taken twice.
        // >>> AND AN 8-BIT RESULT IS A QUARTER OF ONE, for the same reason and by the same
        // rule: the four channels of an 8-bit VPCK are the four BYTES of the operand's own
        // register, which is how `emit_pack_from_int` reads one back. A whole-register store
        // here would put channel 1 in register `index + 1`, where its reader looks in byte 1
        // of `index`.
        // >>> THE PLACEMENT RULE IS `Instr::dest_raw_packed_bits`, NOT A SECOND `match` HERE.
        // The reference interpreter's generic store path reads the same function, because it
        // had no rule at all and placed a 16-bit result one whole register per channel - the
        // defect this comment describes, on the oracle's side, found by the GPU differential
        // months after the emitter got it right.
        match instr.dest_raw_packed_bits() {
            Some(16) => body.store_raw_half(dest, c, &e)?,
            Some(8) => body.store_raw_byte(dest, c, &e)?,
            _ => body.store_raw(dest, c, &e)?,
        }
    }
    Some(())
}

/// The RAW bit pattern of one packed element of `reg`: `width` bits starting at bit `lo`,
/// sign-extended to 32 when `signed`.
///
/// A shift by zero and a mask that covers the whole remainder are both omitted, so the 16-bit
/// forms read exactly as they were spelled out when this was two hand-written branches.
fn raw_elem_expr(reg: &str, lo: u32, width: u32, signed: bool) -> String {
    if signed {
        let up = 32 - width - lo;
        let down = 32 - width;
        if up == 0 {
            format!("(bitcast<i32>({reg}) >> {down}u)")
        } else {
            format!("((bitcast<i32>({reg}) << {up}u) >> {down}u)")
        }
    } else {
        let m = (1u64 << width) - 1;
        match (lo, lo + width) {
            (0, _) => format!("({reg} & {m:#x}u)"),
            (_, 32) => format!("({reg} >> {lo}u)"),
            _ => format!("(({reg} >> {lo}u) & {m:#x}u)"),
        }
    }
}

/// The mirror of [`emit_pack_to_int`]: widen a 16-bit integer half to the destination float.
///
/// The SOURCE half is addressed exactly as `Dest::store_raw_half` writes one - half `sel & 1` of
/// register `index + (sel >> 1)`, where `sel` is the operand's own component selector - so a
/// value this recompiler stored through that path reads back as itself. The sign extension is
/// the same shift pair `emit_int_mad` uses on its packed operand, for the same reason: those two
/// already agree on how a 16-bit half is widened, and a third rule here would be a third answer.
fn emit_pack_from_int(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    mask: [bool; 4],
    bits: u8,
    signed: bool,
) -> Option<()> {
    debug_assert!(bits == 16 || bits == 8, "only the 16- and 8-bit widths decode to this op");
    let s1 = instr.srcs.first()?;
    let dp = Prec::of(instr);
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let sel = s1.swizzle[c] as u32;
        let elems_per_reg = 32 / bits as u32;
        let reg = format!("{}[{}]", bank_prefix(s1.bank)?, s1.index as u32 + sel / elems_per_reg);
        let lo = (sel % elems_per_reg) * bits as u32;
        let half = raw_elem_expr(&reg, lo, bits as u32, signed);
        let e = format!("f32({half})");
        let e = if s1.neg { format!("(-{e})") } else { e };
        let e = if s1.abs { format!("abs({e})") } else { e };
        body.store(dest, c, &e, dp)?;
    }
    Some(())
}

/// A 16-bit integer VPCK whose destination is 16-bit too: a swizzled copy of HALVES.
///
/// Neither end converts, so this is the source reader of [`emit_pack_from_int`] and the
/// destination writer of [`emit_pack_to_int`]'s 16-bit branch with no arithmetic between them.
/// The bits are moved unsigned because at equal widths the pattern does not depend on the
/// labels: a sign is applied by whoever WIDENS a half later, and applying one here would
/// sign-extend into the partner half's bits and destroy a value the shader reads back.
fn emit_pack_int_copy(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    mask: [bool; 4],
    bits: u8,
) -> Option<()> {
    debug_assert!(bits == 16 || bits == 8, "only the equal-width int copies decode to this op");
    let s1 = instr.srcs.first()?;
    let elems_per_reg = 32 / bits as u32;
    for c in 0..4 {
        if !mask[c] {
            continue;
        }
        let sel = s1.swizzle[c] as u32;
        let reg = format!("{}[{}]", bank_prefix(s1.bank)?, s1.index as u32 + sel / elems_per_reg);
        let lo = (sel % elems_per_reg) * bits as u32;
        let elem = raw_elem_expr(&reg, lo, bits as u32, false);
        if bits == 16 {
            body.store_raw_half(dest, c, &elem)?;
        } else {
            body.store_raw_byte(dest, c, &elem)?;
        }
    }
    Some(())
}

/// `idx[n] = src + addend` - load an index register for later register-INDIRECT addressing.
///
/// The source lane holds an integer bit pattern (it was produced by [`emit_pack_to_int`] and a
/// 16-bit shift), so it is read raw rather than through a float view.
fn emit_load_index(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    addend: i32,
    to_index: bool,
    stride: u8,
) -> Option<()> {
    let s1 = instr.srcs.first()?;
    let bank = bank_prefix(s1.bank)?;
    // The form whose sum goes to an ORDINARY REGISTER - see `Op::LoadIndex`. It addresses
    // nothing indirectly, so the index register's PAIR SCALE below does not apply: the
    // consuming IMAD32 multiplies this by the palette's own row stride.
    if !to_index {
        let dbank = bank_prefix(dest.bank)?;
        let mul = crate::link::index_load_multiplier().unwrap_or(stride as i32);
        let scaled = if mul == 1 {
            format!("i32({bank}[{}] & 0xffffu)", s1.index as u32)
        } else {
            format!("i32({bank}[{}] & 0xffffu) * {mul}i", s1.index as u32)
        };
        writeln!(body, "  {dbank}[{}] = bitcast<u32>({scaled} + {addend}i);", dest.index as u32)
            .ok();
        return Some(());
    }
    let reg = dest.index.min(1) as u32;
    // >>> THE INDEX REGISTER COUNTS PAIRS OF REGISTERS, NOT REGISTERS.
    //
    // `idx = (src + addend) * 2`. Read as single registers, one title's particle-streak
    // program indexes its corner table at `src*2 + 21 + 14` = SA 35..45 - which is the TAIL of
    // the default uniform container, then the DATA container's uniform-buffer POINTERS. A
    // pointer used as a texcoord weight is a texture coordinate in the millions, and the draw
    // came out as a full-screen neon moiré (`1a8667f6685d5f47`, capsule 177 of
    // `caps869b`) [[vitaslop-a-region-clip-outlives-its-scene]].
    //
    // The scale is not fitted, it is the only one that closes. That program's literal block
    // holds EIGHT one-hot `vec4`s at SA 56..88 - the corner selection table for a quad. Under
    // `*2` the two indexed dots read SA `56 + 4c` and `72 + 4c` for corner `c`, which is
    // (umin,vmin) (umin,vmax) (umax,vmax) (umax,vmin): a quad's winding, using the table
    // exactly once end to end with no overrun. A search over every corner step and every base
    // in the bank returns that solution and NO other.
    writeln!(
        body,
        "  idx[{reg}] = (i32({bank}[{}] & 0xffffu) + {addend}i) * {}i;",
        s1.index as u32,
        crate::module::index_register_scale()
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
/// The name of the guest-memory-window binding a given STAGE reads through. One WGSL module
/// carries both when a pair loads memory in both stages, and they are different buffers -
/// bound from different GXM uniform-buffer tables - so they cannot share a name.
pub fn mem_binding_name(kind: ProgramKind) -> &'static str {
    match kind {
        ProgramKind::Vertex => "gxp_mem",
        _ => "gxp_fmem",
    }
}

fn emit_mem_load(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    elements: u8,
    offset_bytes: u32,
    index: usize,
    kind: ProgramKind,
) -> Option<()> {
    let src0 = instr.srcs.first()?;
    let ptr_bank = bank_prefix(src0.bank)?;
    let dest_bank = bank_prefix(dest.bank)?;
    // The GUEST address of the first element, in its OWN BLOCK.
    //
    // The name carries the instruction index, which is unique within a stream and NOT
    // across streams: the secondary and primary programs are emitted separately, each
    // numbered from zero, and concatenated into one function (see `block`). A program that
    // loads from BOTH - which is what a title doing skinning on top of a pointer-chased
    // uniform buffer does - then declares `gxp_a8` twice in one body, the module fails to
    // parse with "redefinition of gxp_a8", wgpu refuses the pipeline, and every draw using
    // it is dropped. So the defect surfaces as missing geometry and says nothing about
    // names. The `let` is read only by the stores right below it, so a block scopes it
    // with nothing else to change.
    // Any REGISTER-supplied byte offsets the instruction carries, added to the pointer. The
    // decoder puts them in `srcs` after the pointer; they hold an integer BYTE displacement
    // the guest's own integer pipeline computed (`index * stride + base`), so they are read
    // raw, exactly as the pointer is, with no float view.
    // >>> A REGISTER OFFSET IS 16 BITS WIDE. This is the whole of one title's particle
    // >>> corruption, and the population it is decided against is ONE program shape.
    //
    // THE FINDING. A SubUV particle program computes `pa[8] = int16(SubUVIndices.x) * 16 +
    // sa[88]` and loads `mem[sa[38] + pa[8]]`, where `sa[88]` is a CONTAINER LITERAL holding
    // `0x00010000`. Its window is `SubUVExtents`, declared 32 float4s = 512 bytes, indexed
    // 0..32 - so every displacement the buffer can want is `0..496` and the literal's
    // `0x10000` is one bit ABOVE that range. Read full-width, a run reported **192 of 208
    // guest-memory reads landing in NO bound window and reading ZERO**, every one missing by
    // exactly `0x10000`, and the particle quads it feeds painted the saturated masses and the
    // screen-length streaks that stood as this title's open picture defect for sessions.
    //
    // >>> THE GUEST'S OWN MEMORY SETTLES IT, and it was asked rather than reasoned about.
    // `VITASLOP_GXP_MEM_PEEK=10000` prints the words at a window's base and at base+0x10000:
    // at the BASE they are `SubUVExtents` exactly (0.2930, 0.6680, 0.3047, 0.8789 - UV extents
    // in 0..1); at +0x10000 every word is ZERO, on every particle program in the title. The
    // data is where this renderer puts the window, so the addend cannot be a byte displacement.
    //
    // >>> AND THE OPERAND IS NOT SHARED WITH ANYTHING ELSE. `mem_load_register_offsets_and_
    // what_computes_them` (tests/corpus.rs) enumerates every 0xE8 load carrying a register
    // offset across every captured corpus: **there is exactly ONE**, this one. The golf title's
    // programs - the ones that do 32-bit address arithmetic in an IMAD32 - put the whole
    // ADDRESS in `src0` (`mem[pa[7]]`, base and all), not in an offset, so their arithmetic
    // never passes through here and cannot be narrowed by this. A 2026-09-07 note recorded the
    // narrowing as REFUTED because "some other program's offset legitimately exceeds 65535";
    // no such program exists in any corpus, and that reading is retired.
    //
    // Kept as an arm (`VITASLOP_GXP_MEM_OFFSET16=0` restores the full-width read) because the
    // rule rests on one program shape plus the guest's memory rather than on a published field
    // width, and an A/B that needs a rebuild is one nobody takes.
    let narrow = crate::link::arm_on(crate::link::MEM_OFFSET16_ARM);
    let mut reg_offsets = String::new();
    for o in instr.srcs.iter().skip(1) {
        let bank = bank_prefix(o.bank)?;
        let idx = o.index as u32;
        if narrow {
            write!(reg_offsets, " + ({bank}[{idx}] & 0xffffu)").ok()?;
        } else {
            write!(reg_offsets, " + {bank}[{idx}]").ok()?;
        }
    }
    writeln!(body, "  {{").ok()?;
    writeln!(
        body,
        "    let gxp_a{index}: u32 = {ptr_bank}[{}] + {offset_bytes}u{reg_offsets};",
        src0.index as u32
    )
    .ok()?;
    for k in 0..elements as u32 {
        writeln!(
            body,
            "    {dest_bank}[{}] = {}_word(gxp_a{index} + {}u);",
            dest.index as u32 + k,
            mem_binding_name(kind),
            k * 4
        )
        .ok()?;
    }
    writeln!(body, "  }}").ok()?;
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
        // SHIFT LEFT, the same raw unsigned 32-bit lane. The decoder has already refused any
        // amount that is not an inline immediate below 32, so this shift is always in WGSL's
        // defined range - there is no clamp here because there is nothing to clamp. The
        // comparison is against `0u` for the same reason the AND above is: the family's
        // operands are UNSIGNED, so `0x80000000` is a large positive number. Comparing it as
        // a signed integer instead flips exactly the case the corpus encodes.
        if matches!(alu, TestAlu::BitShl) {
            bools.push(format!("(({} << {}) {op} 0u)", raw(s1)?, raw(s2)?));
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
            TestAlu::BitAnd | TestAlu::BitShl | TestAlu::IntSub => {
                unreachable!("raw-lane test resolved before the float path")
            }
            // VTSTMSK's decoder is the only producer of this family; VTST cannot reach it.
            TestAlu::IntSub16U => return None,
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
                TestAlu::BitAnd | TestAlu::BitShl | TestAlu::IntSub | TestAlu::IntSub16U => {
                    return None
                }
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
/// Two forms reach here and the decoder refuses every other, because they are the only two
/// whose written value is agreed by both readings of the mask field (see `decode_grp_test_mask`):
///
/// * the FLOAT families with the numeric form - four channels of `1.0` / `0.0`;
/// * the UNSIGNED 16-BIT integer family with the precision-mask form - ONE channel of
///   `0xFFFF` / `0x0000`, written as a raw lane.
///
/// The integer form goes nowhere near the float channel reader. Its operands are whole raw
/// lanes, and putting one through a float view would read a small integer as a denormal - the
/// same near-miss [`TestAlu::Fx8Sub`] and [`TestAlu::IntSub`] exist to avoid.
fn emit_test_mask(
    body: &mut Dest,
    instr: &Instr,
    dest: &Operand,
    alu: TestAlu,
    cmp: TestCmp,
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
    // >>> THE UNSIGNED 16-BIT INTEGER FORM: ONE CHANNEL, A RAW LANE, `0xFFFF` OR `0x0000`.
    //
    // The comparison is done on the low 16 bits of each raw lane, which is what makes the
    // width load-bearing: two lanes differing only above bit 15 are EQUAL to the device and
    // would not be to a 32-bit compare. The subtract is expressed as the equality it is - the
    // relation is `== 0` and `a - b == 0` iff `a == b` - so unsigned wraparound cannot enter.
    if matches!(alu, TestAlu::IntSub16U) {
        // Only the `== 0` / `!= 0` relations are established for this form: an ordered
        // relation would additionally need the SIGNEDNESS of the compare pinned, and the one
        // reading that describes signed integer masks is uncorroborated.
        let eq = match cmp {
            TestCmp::Eq => "==",
            TestCmp::Ne => "!=",
            _ => return None,
        };
        let raw = raw_lane_expr(s1, kind)?;
        let raw2 = raw_lane_expr(s2, kind)?;
        // Channel x only - the decoder's write mask says so and the reference derives the
        // count from the ALU family. `store_raw` because the lane holds an integer bit
        // pattern: going through the float store would bitcast it and change the bits.
        body.store_raw(
            dest,
            0,
            &format!("select(0u, 0xffffu, ((({raw}) & 0xffffu) {eq} (({raw2}) & 0xffffu)))"),
        )?;
        return Some(());
    }
    for c in 0..4 {
        let (a, b) = (src_channel(s1, c, p)?, src_channel(s2, c, p)?);
        let value = match alu {
            TestAlu::Add => format!("({a} + {b})"),
            TestAlu::Sub => format!("({a} - {b})"),
            TestAlu::Mul => format!("({a} * {b})"),
            // The decoder does not produce these for this group; refusing keeps the emitter
            // from inventing a raw-lane mask if that ever changes.
            TestAlu::Fx8Sub
            | TestAlu::BitAnd
            | TestAlu::BitShl
            | TestAlu::IntSub
            | TestAlu::IntSub16U => return None,
        };
        body.store(dest, c, &format!("select(0.0, 1.0, ({value} {op} 0.0))"), p)?;
    }
    Some(())
}

/// One operand read as its RAW 32-bit lane, for the integer/bitwise test families.
///
/// Shared by [`emit_test`] and [`emit_test_mask`] so the two cannot drift: the banks that only
/// ever appear on a raw-lane operand - an assembled inline immediate, a constant-bank entry,
/// and the hardware GLOBAL registers - each need their own spelling, and a family that read
/// one of them through the float path would read a small integer as a denormal.
fn raw_lane_expr(o: &Operand, kind: ProgramKind) -> Option<String> {
    if matches!(o.bank, Bank::Constant) {
        let bank = if o.swizzle[0] == 1 { &CNST6_F32_BANK1 } else { &CNST6_F32_BANK0 };
        return Some(format!("{:#010x}u", bank[(o.index & 0x3f) as usize]));
    }
    if matches!(o.bank, Bank::Immediate) {
        return Some(format!("{}u", o.index as u32));
    }
    if matches!(o.bank, Bank::Global) {
        return global_u32_expr(o, kind);
    }
    Some(format!("{}[{}]", bank_prefix(o.bank)?, o.index as u32))
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
    src0_high: bool,
    src1_high: bool,
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
        // A hardware-constant source contributes its 32-bit table entry VERBATIM, the same way
        // the VBW emitter reads one: these groups are bit-pattern ops, so the entry is used as
        // stored rather than through a float view, and the channel-0 swizzle selector picks the
        // bank exactly as it does everywhere else.
        if matches!(o.bank, Bank::Constant) {
            let bank = if o.swizzle[0] == 1 { &CNST6_F32_BANK1 } else { &CNST6_F32_BANK0 };
            return Some(format!("{:#010x}u", bank[(o.index & 0x3f) as usize]));
        }
        if matches!(o.bank, Bank::Indexed) {
            return indexed_element(o, 0);
        }
        Some(format!("{}[{}]", bank_prefix(o.bank)?, o.index as u32))
    };
    // >>> src0 IS ONE HALF OF A PACKED PAIR - see `Op::IntMad`. `src0_high` picks which, and
    // the half is widened to 32 bits before the multiply. The widening follows the
    // instruction's own `signed` flag; every value any corpus program puts through here is a
    // small non-negative array index, so the two widenings agree on all of them and this
    // corpus cannot separate them - the flag is the only statement available and it is used
    // rather than assumed away.
    let a0 = raw(instr.srcs.first()?)?;
    let a = match (signed, src0_high) {
        // A shift into the sign bit and an ARITHMETIC shift back is the sign extension; the
        // masks are the zero extension.
        (true, true) => format!("bitcast<u32>(bitcast<i32>({a0}) >> 16u)"),
        (true, false) => format!("bitcast<u32>((bitcast<i32>({a0}) << 16u) >> 16u)"),
        (false, true) => format!("({a0} >> 16u)"),
        (false, false) => format!("({a0} & 0xffffu)"),
    };
    // `src1_high` selects a 16-bit half of src1 the same way, and for the same reason: the
    // packed pair the multiplier reads a half of is wherever the compiler put it. A CLEAR bit
    // reads the whole register, which is what every program that recompiled before this bit
    // was decoded did - see the decoder's note.
    let b0 = raw(instr.srcs.get(1)?)?;
    let b = match (signed, src1_high) {
        (_, false) => b0,
        (true, true) => format!("bitcast<u32>(bitcast<i32>({b0}) >> 16u)"),
        (false, true) => format!("({b0} >> 16u)"),
    };
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
    // >>> `signed` IS DECODED AND DELIBERATELY UNUSED HERE.
    //
    // The expression below is exactly `src0 * src1 + src2` modulo 2^32 - the `<< 16u` discards
    // what the high product carries above bit 15 - and two's-complement multiplication agrees on
    // the low 32 bits for signed and unsigned operands alike, so ONE emission is correct for
    // both. `usse::validate_imad_step_pairs` is what makes the PAIR's value the only observable;
    // see `decode_grp_imad32_step` for that and for the corpus closure behind it.
    let _ = signed;
    let raw = |o: &Operand| -> Option<String> {
        if matches!(o.bank, Bank::Immediate) {
            return Some(format!("{}u", o.index as u32));
        }
        // A hardware-constant source contributes its 32-bit table entry VERBATIM, the same way
        // the VBW emitter reads one: these groups are bit-pattern ops, so the entry is used as
        // stored rather than through a float view, and the channel-0 swizzle selector picks the
        // bank exactly as it does everywhere else.
        if matches!(o.bank, Bank::Constant) {
            let bank = if o.swizzle[0] == 1 { &CNST6_F32_BANK1 } else { &CNST6_F32_BANK0 };
            return Some(format!("{:#010x}u", bank[(o.index & 0x3f) as usize]));
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
    /// The bound texture is a 64-bit RAW format whose texel the sampler hands over as its two
    /// 32-bit words UNCONVERTED, so the binding is a `texture_2d<u32>` read with `textureLoad`
    /// and the shader's registers receive the words as stored. See
    /// [`crate::link::LinkOptions::raw_units`].
    pub raw: bool,
}

impl TexBinding {
    /// The WGSL texture type this binding must be declared as.
    pub fn wgsl_type(&self) -> &'static str {
        match (self.raw, self.coords >= 3, self.cube) {
            (true, ..) => "texture_2d<u32>",
            (false, true, true) => "texture_cube<f32>",
            (false, true, false) => "texture_3d<f32>",
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
                None => out.push(TexBinding { unit, coords, cube: is_cube(unit), raw: false }),
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

    /// >>> THE EMITTER'S CONSTANT AND THE REFERENCE'S CONSTANT ARE ONE FACT, AND THEY DRIFTED.
    ///
    /// [`src_channel`] builds a WGSL literal for a constant operand channel;
    /// [`cnst6_channel_value`] returns the number the interpreter uses for the same channel.
    /// They were written separately and disagreed for every selector but bank 0 - `pos * 1.0`
    /// against `pos * 0.0` on a real vertex program, which is a collapsed mesh.
    ///
    /// This evaluates the emitted LITERAL and requires it to equal the value function, over
    /// every index and selector in both precisions. It is deliberately a parse of the emitted
    /// text rather than a shared call, because the emitted text is what the GPU runs: a change
    /// that alters the string without altering the value function fails here.
    #[test]
    fn the_emitted_constant_and_the_reference_constant_are_the_same_number() {
        /// Evaluate the exact literal forms the constant arm of `src_channel` produces.
        fn eval(lit: &str) -> f32 {
            if let Some(hex) = lit.strip_prefix("bitcast<f32>(").and_then(|s| s.strip_suffix("u)")) {
                return f32::from_bits(u32::from_str_radix(hex.trim_start_matches("0x"), 16).unwrap());
            }
            if let Some(hex) = lit
                .strip_prefix("unpack2x16float(")
                .and_then(|s| s.strip_suffix("u)[0]"))
            {
                let bits = u32::from_str_radix(hex.trim_start_matches("0x"), 16).unwrap();
                return f16_bits_to_f32(bits as u16);
            }
            lit.parse().unwrap_or_else(|_| panic!("unrecognised constant literal `{lit}`"))
        }

        for index in 0u8..64 {
            for sel in 0u8..8 {
                for (prec, half) in [(Prec::F32, false), (Prec::F16, true)] {
                    let mut op = Operand::plain(Bank::Constant, index, 0);
                    op.swizzle = [sel; 4];
                    let emitted = src_channel(&op, 0, prec).expect("the float views always emit");
                    let reference =
                        cnst6_channel_value(index, sel, half, false).expect("the float views always have a value");
                    let e = eval(&emitted);
                    assert_eq!(
                        e.to_bits(),
                        reference.to_bits(),
                        "constant index {index} selector {sel} half={half}: emitter `{emitted}` = {e}, reference {reference}"
                    );
                }
                // The 8-bit view has no established constant TABLE and both sides must refuse a
                // table read rather than substitute a float bank's entry - but the four inline
                // constants (selectors 4..7) are a property of the selector alone and both
                // sides do supply those. The agreement has to hold in either direction.
                let mut op = Operand::plain(Bank::Constant, index, 0);
                op.swizzle = [sel; 4];
                let emitted = src_channel(&op, 0, Prec::Fx8);
                let reference = cnst6_channel_value(index, sel, false, true);
                assert_eq!(
                    emitted.is_some(),
                    reference.is_some(),
                    "8-bit constant index {index} selector {sel}: emitter {emitted:?}, reference {reference:?}"
                );
                if let (Some(e), Some(r)) = (emitted, reference) {
                    assert_eq!(eval(&e).to_bits(), r.to_bits(), "8-bit selector {sel}");
                }
            }
        }
    }

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

    /// The four BYTES of one register, addressed by the component selector - the 8-bit VPCK
    /// widths, at both ends.
    ///
    /// The evidence is a football title's fragment secondary: ONE `U8 -> F16` VPCK repeated
    /// four times under an SMLSI in SWIZZLE mode, whose byte `0xe1` walks selectors 1,0,2,3
    /// over a FIXED source register while the destination steps -2 registers a time. The
    /// primary stream then dots the four results against one sampled RGBA, and the four
    /// parameters the source register holds are the title's `EBR`/`EBG`/`EBB`/`EBA`, declared
    /// `Uniform U8 comps 1` - so selector 1 must be the GREEN byte of that ONE register. Under
    /// the 16-bit rule (`index + (sel >> 1)`) selector 1 would be the high HALF of the same
    /// register and selector 2 a different register entirely, and the dot would be of four
    /// values that are not those four parameters.
    #[test]
    fn an_eight_bit_pack_selector_counts_bytes_inside_one_register() {
        let src = |sel: u8| {
            let mut o = Operand::plain(Bank::SecondaryAttr, 36, 2);
            o.swizzle = [sel; 4];
            o
        };
        let dest = Operand::plain(Bank::SecondaryAttr, 60, 2);
        for (sel, lo) in [(0u8, 0u32), (1, 8), (2, 16), (3, 24)] {
            let mut i = instr(Op::PackFromInt { bits: 8, signed: false }, Some(dest), vec![src(sel)]);
            i.write_mask = [true, false, false, false];
            let wgsl = emit_fragment(&shader(vec![i])).unwrap();
            // ONE register - the operand's own - and the byte at the selector's position.
            assert!(wgsl.contains("sa[36]"), "selector {sel} must read the operand's register:
{wgsl}");
            assert!(!wgsl.contains("sa[37]"), "selector {sel} must not reach a second register:
{wgsl}");
            let want =
                if lo == 0 { "(sa[36] & 0xffu)".to_string() } else { format!("(sa[36] >> {lo}u)") };
            assert!(
                wgsl.contains(&want) || wgsl.contains(&format!("((sa[36] >> {lo}u) & 0xffu)")),
                "selector {sel} must read byte {} of sa[36]:
{wgsl}",
                lo / 8
            );
        }
        // And the same-width copy writes a byte where it read one, leaving the other three
        // alone - the byte broadcast the same title emits (`sa[2]` bytes 1 and 2 from `sa[0]`
        // byte 0). A whole-register store would clear the partners.
        let mut copy =
            instr(Op::PackIntCopy { bits: 8 }, Some(Operand::plain(Bank::SecondaryAttr, 2, 2)), vec![src(0)]);
        copy.write_mask = [false, true, false, false];
        let wgsl = emit_fragment(&shader(vec![copy])).unwrap();
        assert!(
            wgsl.contains("sa[2] = (sa[2] & 0xffff00ffu) | (((sa[36] & 0xffu) & 0xffu) << 8u);"),
            "a byte copy is a read-modify-write of ONE byte:
{wgsl}"
        );
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
        // Four F16 coefficients occupy TWO registers past the four gathered texels, each
        // register's two halves folded into one write (see `Dest::flush`).
        assert!(wgsl.contains("r[4] = gxp_hpk("), "coefficients start at dest + 4:\n{wgsl}");
        assert!(wgsl.contains("r[5] = gxp_hpk("), "coefficients span two registers:\n{wgsl}");
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
    /// >>> THE WORD THAT PANICKED A USER'S RUN, EMITTED.
    ///
    /// `0x7802019271f6a839`: the unsigned 16-bit VTSTMSK. The decoder now translates it (see
    /// `decode_grp_test_mask`); this pins what comes out, because unblocking the decoder buys
    /// nothing if the emitter refuses - the pair would hard-fail exactly as before, one layer
    /// down.
    ///
    /// It is a FACING TEST: `GLOBAL[16]` is the back-face register, which this project and the
    /// vendor's own header independently name, so the shader is asking "is this face front or
    /// back" and depositing a 16-bit all-ones mask for the answer.
    /// The OTHER half of the same idiom, and the instruction that panicked the run after the
    /// VTSTMSK above was unblocked: `0x48090881a00cc79f`, a BITWISE SHIFT-LEFT test.
    ///
    /// The program writes a `0x0000FFFF`-or-zero facing mask with the VTSTMSK, then shifts THAT
    /// register left by 31 and asks whether the result is positive - which keeps bit 0 of the
    /// mask and nothing else. Two things this pins that a reader could otherwise "simplify"
    /// away, both of which silently invert the test:
    ///
    /// * the comparison is UNSIGNED (`> 0u`). `0xFFFF << 31` is `0x80000000`, which is a large
    ///   positive number to this family and a NEGATIVE one to a signed compare.
    /// * there is no clamp on the shift amount, because the decoder refuses any amount that is
    ///   not an inline immediate below 32.
    #[test]
    fn vtst_bitwise_shift_left_tests_bit_zero_of_the_facing_mask() {
        let pa15 = Operand::plain(Bank::PrimaryAttr, 15, 2);
        let amount = Operand::plain(Bank::Immediate, 31, 2);
        let wgsl = emit_fragment(&shader(vec![instr(
            Op::Test {
                alu: TestAlu::BitShl,
                cmp: TestCmp::Gt,
                reduce: TestReduce::Channel(0),
                pdst: 0,
                write_back: false,
            },
            None,
            vec![pa15, amount],
        )]))
        .unwrap();
        assert!(wgsl.contains("p[0] = ((pa[15] << 31u) > 0u);"), "raw unsigned shift test:
{wgsl}");
        // Not through the float path: a bitcast here would read 0x0000FFFF as a denormal.
        assert!(!wgsl.contains("bitcast<f32>(pa[15])"), "no float view:
{wgsl}");
        // And it must not have become a signed compare on the way out.
        assert!(!wgsl.contains("bitcast<i32>"), "unsigned, not signed:
{wgsl}");
    }

    #[test]
    fn vtstmsk_u16_writes_a_raw_sixteen_bit_mask_on_one_channel() {
        let d = Operand::plain(Bank::PrimaryAttr, 15, 2);
        let g = Operand::plain(Bank::Global, 16, 1);
        let sa = Operand::plain(Bank::SecondaryAttr, 57, 3);
        let wgsl = emit_fragment(&shader(vec![instr(
            Op::TestMask { alu: TestAlu::IntSub16U, cmp: TestCmp::Eq },
            Some(d),
            vec![g, sa],
        )]))
        .unwrap();
        // The comparison is on the LOW 16 BITS of each raw lane. Two lanes differing only
        // above bit 15 are equal to the device, and a 32-bit compare would call them different.
        assert!(wgsl.contains("& 0xffffu"), "masked to 16 bits:
{wgsl}");
        // 0xFFFF for true, 0 for false - the pair both readings of the mask field agree on.
        assert!(wgsl.contains("select(0u, 0xffffu,"), "an all-ones 16-bit mask:
{wgsl}");
        // Written as a RAW lane. Going through the float store would bitcast the pattern and
        // change the bits a later raw-lane read sees.
        assert!(wgsl.contains("pa[15] ="), "channel x of pa[15]:
{wgsl}");
        assert!(!wgsl.contains("bitcast<u32>(g0)"), "not through the float store path:
{wgsl}");
        // ONE channel: the count comes from the ALU family, so pa[16..18] are untouched.
        for r in ["pa[16]", "pa[17]", "pa[18]"] {
            assert!(!wgsl.contains(r), "{r} must not be written:
{wgsl}");
        }
        // And it really is the facing register on the other side of the compare.
        assert!(wgsl.contains("front_facing") || wgsl.contains("FrontFacing"), "reads facing:
{wgsl}");
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
        // hard-fail rather than translate an unmodeled iterator pre-load - WHEN THE RESULT
        // REACHES THE OUTPUT. The destination is the output bank for exactly that reason: the
        // guard is about an unmodeled input the frame can see (see `live_instructions`).
        let d = Operand::plain(Bank::Output, 0, 1);
        let a = Operand::plain(Bank::PrimaryAttr, 4, 2);
        let i = Operand::plain(Bank::Internal, 0, 0); // i0, xxxx
        let err = emit_fragment(&shader(vec![instr(Op::Dot { components: 4 }, Some(d), vec![a, i])]))
            .unwrap_err();
        match err {
            EmitError::UndefinedInternal { lane, .. } => assert_eq!(lane, 0),
            other => panic!("expected UndefinedInternal, got {other:?}"),
        }
    }

    /// The other half of the guard: the same read, into a temporary nothing goes on to consume,
    /// TRANSLATES. This is the shape that was dropping fifteen of Madden's fragment blobs - an
    /// F32 write of lane 0 followed by a narrower consumer whose second channel is thrown away -
    /// and refusing it removed the pair's whole mesh from the frame over a value no one reads.
    #[test]
    fn a_dead_undefined_internal_read_translates_instead_of_refusing() {
        let d = Operand::plain(Bank::Temp, 0, 0);
        let a = Operand::plain(Bank::PrimaryAttr, 4, 2);
        let i = Operand::plain(Bank::Internal, 0, 0);
        let wgsl = emit_fragment(&shader(vec![instr(Op::Dot { components: 4 }, Some(d), vec![a, i])]))
            .expect("a dead over-read of an internal lane must emit, not refuse");
        assert!(wgsl.contains(&rd("i", 0)), "got:
{wgsl}");
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
        // sa[6] and pa[4] are each read by two channels, so `cse_unpacks` hoists them; sa[7]
        // and pa[5] are read once and stay spelled out.
        assert!(wgsl.contains("let u_sa6 = unpack2x16float(sa[6]);"), "got:
{wgsl}");
        assert!(wgsl.contains("let u_pa4 = unpack2x16float(pa[4]);"), "got:
{wgsl}");
        assert!(wgsl.contains("(u_sa6[0] * u_pa4[0])"), "got:
{wgsl}");
        assert!(wgsl.contains("(u_sa6[1] * u_pa4[1])"), "got:
{wgsl}");
        assert!(wgsl.contains("unpack2x16float(sa[7])[0] * unpack2x16float(pa[5])[0]"), "got:
{wgsl}");
        // Channels 0 and 1 are the two halves of r[0], so the pair FOLDS into one write with
        // no read-modify-write left (see `Dest::flush`) - low half first.
        assert!(
            wgsl.contains("r[0] = gxp_hpk((u_sa6[0] * u_pa4[0]), (u_sa6[1] * u_pa4[1]));"),
            "got:
{wgsl}"
        );
        // Channel 2 moves on to the LOW half of r[1]; channel 3 is masked out, so that half has
        // no partner and keeps the read-modify-write that preserves the high half.
        assert!(wgsl.contains("r[1] = gxp_hlo(r[1], "), "got:
{wgsl}");
        assert!(!wgsl.contains("gxp_hhi(r[1], "), "channel 3 masked:
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
        let (before, inside) = wgsl.split_once("loop {").unwrap_or_else(|| panic!("got:\n{wgsl}"));
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

    /// >>> A DERIVATIVE WHOSE SOURCE THE BRANCH ITSELF WROTE GETS A UNIFORM GAP, not a refusal.
    ///
    /// The hoist above the block is exact only while nothing in the block has rewritten the
    /// register being differenced. When the block computes that value, the block is CLOSED at
    /// the derivative, the builtin called where control flow is uniform, and the block
    /// re-entered on the same condition - which is what the hardware does (the USSE predicates
    /// the write-back, never the differencing). Three of a football title's world fragment
    /// programs are this shape and every one of them was dropped.
    #[test]
    fn a_derivative_over_a_register_the_branch_wrote_gets_a_uniform_gap() {
        // 0: br if p0 -> 3     (guards 1..2)
        // 1: mov r0            <- writes the register the derivative reads
        // 2: dsx r4 <- r0
        let dsx = instr(
            Op::Dsx,
            Some(Operand::plain(Bank::Temp, 4, 0)),
            vec![Operand::plain(Bank::Temp, 0, 0)],
        );
        let wgsl = emit_fragment(&shader(vec![branch(3, Predicate::IfP(0)), mov(0), dsx])).unwrap();
        // The guarded block opens, closes BEFORE the call, and opens again on the same test.
        assert_eq!(wgsl.matches("if (!p[0]) {").count(), 2, "the block is re-entered:
{wgsl}");
        let (head, tail) = wgsl.split_once("dpdx").unwrap();
        // The write the derivative reads is INSIDE the first block, above the gap.
        assert!(head.contains("r[0] ="), "the branch's own write comes first:
{wgsl}");
        // ...and the gap is OUTSIDE it: the block's closing brace stands between the write and
        // the call, which is the whole point - `dpdx` must not be inside the `if`.
        assert!(
            head.lines().rev().skip(1).find(|l| !l.trim().is_empty()).is_some_and(|l| l.trim() == "}"),
            "the call is in a uniform gap - the block closes just above it:
{wgsl}"
        );
        // The STORE is back inside the re-entered block, so only the lanes the branch admitted
        // write it - exactly the hardware's predicated write-back.
        let store = tail.split_once("r[4] =").expect("the derivative is stored").0;
        assert!(store.contains("if (!p[0]) {"), "the store is re-guarded:
{wgsl}");
    }

    /// A LOOP cannot be closed and re-entered - the rest of its iterations would run outside it -
    /// so a derivative over a register the loop body writes is still refused, by name.
    #[test]
    fn a_derivative_over_a_register_a_loop_body_wrote_hard_fails() {
        // 0: mov r0            <- loop head, and the write
        // 1: dsx r4 <- r0
        // 2: br if p0 -> 0     (the back edge)
        let dsx = instr(
            Op::Dsx,
            Some(Operand::plain(Bank::Temp, 4, 0)),
            vec![Operand::plain(Bank::Temp, 0, 0)],
        );
        let err = emit_fragment(&shader(vec![mov(0), dsx, branch(-2, Predicate::IfP(0))])).unwrap_err();
        match err {
            EmitError::Blocked { reason, .. } => {
                assert!(reason.contains("inside a LOOP"), "{reason}");
            }
            other => panic!("expected Blocked, got {other:?}"),
        }
    }

    /// A branch that rewrites a PREDICATE register before the derivative cannot be re-entered on
    /// the same condition - the re-test would admit a different set of lanes - so it is refused.
    #[test]
    fn a_derivative_in_a_branch_that_rewrites_a_predicate_hard_fails() {
        // 0: br if p0 -> 4     (guards 1..3)
        // 1: mov r0
        // 2: vtst              (writes a predicate)
        // 3: dsx r4 <- r0
        let mut test = instr(
            Op::Test {
                alu: crate::ir::TestAlu::Sub,
                cmp: crate::ir::TestCmp::Ne,
                reduce: crate::ir::TestReduce::Channel(0),
                pdst: 0,
                write_back: false,
            },
            None,
            vec![Operand::plain(Bank::Temp, 8, 0), Operand::plain(Bank::Temp, 9, 0)],
        );
        test.write_mask = [false; 4];
        let dsx = instr(
            Op::Dsx,
            Some(Operand::plain(Bank::Temp, 4, 0)),
            vec![Operand::plain(Bank::Temp, 0, 0)],
        );
        let err = emit_fragment(&shader(vec![branch(4, Predicate::IfP(0)), mov(0), test, dsx]))
            .unwrap_err();
        match err {
            EmitError::Blocked { reason, .. } => {
                assert!(reason.contains("rewrites a predicate register"), "{reason}");
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
    // ====================================================================================
    // The f16 STORE ROUNDING helpers - see `HALF_LO_FN`.
    // ====================================================================================

    /// The body must name the helpers and NEVER the builtin, because the builtin's rounding
    /// mode is the device's choice and the body is hashed and cached across devices.
    #[test]
    fn a_half_store_calls_the_helper_and_never_the_pack_builtin() {
        let body = half_stmt("r", 3, false, "x", false) + &half_stmt("r", 3, true, "y", false);
        assert_eq!(
            body,
            "  r[3] = gxp_hlo(r[3], x);\n  r[3] = gxp_hhi(r[3], y);\n",
            "got:\n{body}"
        );
        assert!(!body.contains("pack2x16float"), "the body must not name the builtin:\n{body}");
    }

    /// ...and the two adjacent halves of one register fold into the pair form, which is ONE
    /// native instruction where the two are two narrowings and a shift.
    #[test]
    fn two_adjacent_halves_of_one_register_fold_into_the_pair_helper() {
        let stmts = half_stmt("r", 3, false, "x", false) + &half_stmt("r", 3, true, "y", false);
        assert_eq!(fold_halves(&stmts), "  r[3] = gxp_hpk(x, y);\n", "got:\n{}", fold_halves(&stmts));
        // A DIFFERENT register between them is not a pair, and folding across it would move a
        // store past a statement - the thing `fold_halves` exists not to do.
        let split = half_stmt("r", 3, false, "x", false)
            + &half_stmt("r", 4, false, "z", false)
            + &half_stmt("r", 3, true, "y", false);
        assert_eq!(fold_halves(&split), split, "only adjacent lines fold:\n{split}");
    }

    /// A RAW half store moves a 16-bit BIT PATTERN, not a float, so it must not go near a
    /// rounding helper: rounding a bone index is how a skinned mesh loses its weights.
    #[test]
    fn a_raw_half_store_does_not_round() {
        let raw = half_stmt("r", 3, false, "b", true);
        assert!(!raw.contains("gxp_h"), "a raw store converts nothing:\n{raw}");
        assert!(raw.contains("& 0xffff0000u"), "it read-modify-writes the word:\n{raw}");
    }

    /// The three arms differ ONLY in the definition of the narrowing, so the module a title
    /// gets is the same text with one function swapped - which is what makes a picture A/B
    /// between them attributable.
    #[test]
    fn the_three_rounding_arms_swap_one_function_and_nothing_else() {
        let body = "  r[0] = gxp_hpk(x, y);\n";
        let module = format!("@group(0) @binding(0) var<uniform> u: vec4<u32>;\n{body}");

        set_native_f16(false);
        crate::link::set_arm(crate::link::F16_ROUND_ARM, "0");
        let pack = add_half_helpers(module.clone());
        crate::link::set_arm(crate::link::F16_ROUND_ARM, "1");
        let portable = add_half_helpers(module.clone());
        set_native_f16(true);
        let native = add_half_helpers(module.clone());
        set_native_f16(false);

        for m in [&pack, &portable, &native] {
            assert!(m.contains("fn gxp_hlo("), "every arm defines the store helpers:\n{m}");
            assert!(m.contains("fn gxp_hq("), "...and the round trip:\n{m}");
            assert!(m.ends_with(body), "the BODY is untouched by the arm:\n{m}");
        }
        // Only the control arm hands the builtin a value that still NEEDS rounding, which is
        // the thing whose mode the device chooses. The portable arm never calls it at all; the
        // native arm calls it on a value already narrowed to f16, where every mode agrees.
        assert!(pack.contains("pack2x16float(vec2<f32>(lo, hi))"), "{pack}");
        assert!(!portable.contains("pack2x16float(vec2"), "{portable}");
        assert!(native.contains("pack2x16float(vec2<f32>(gxp_f16r(lo), gxp_f16r(hi)))"), "{native}");
        assert!(native.contains("fn gxp_f16r(v: f32) -> f32 { return f32(f16(gxp_f16c(v))); }"), "{native}");
        // `enable f16;` is legal only on a device created with the feature, so exactly one arm
        // carries it - and it carries it FIRST, before any declaration.
        assert!(native.starts_with("enable f16;\n"), "{native}");
        assert!(!portable.contains("enable f16;"), "{portable}");
        assert!(!pack.contains("enable f16;"), "{pack}");
    }

    /// A module that makes no half store at all gets no helpers - most vertex programs.
    #[test]
    fn a_module_with_no_half_store_carries_no_helpers() {
        let m = "@fragment\nfn fs_main() {\n  r[0] = bitcast<u32>(1.0);\n}\n".to_string();
        assert_eq!(add_half_helpers(m.clone()), m);
    }

    /// Wrapping an already-helped module twice must not define the functions twice, which is a
    /// WGSL redefinition error and fails the whole module.
    #[test]
    fn adding_the_helpers_twice_defines_them_once() {
        let once = add_half_helpers("  r[0] = gxp_hpk(x, y);\n".to_string());
        assert_eq!(add_half_helpers(once.clone()), once);
        assert_eq!(once.matches("fn gxp_f16b(").count(), 1, "{once}");
    }

    /// >>> THE PORTABLE NARROWING IS THE ONE FUNCTION WHOSE WRONGNESS WOULD BE INVISIBLE, SO
    /// >>> ITS ALGORITHM IS CHECKED EXHAUSTIVELY HERE AGAINST THE REFERENCE'S OWN.
    ///
    /// This is a Rust TRANSCRIPTION of the WGSL, which is the half of the claim a Rust test can
    /// make: it pins the algorithm - every normal, every subnormal, both overflow ties, both
    /// underflow ties - over all 2^32 f32 patterns reachable through the f16 grid. That the
    /// EMITTED TEXT says the same thing is the other half, and `probe-f16round.mjs` measures it
    /// on a real device [[vitaslop-probe-the-shader-dont-simulate-it]].
    #[test]
    fn the_portable_narrowing_rounds_to_nearest_even_everywhere_it_can_be_asked() {
        fn wgsl_arm(v: f32) -> u32 {
            let f = v.to_bits();
            let sign = (f >> 16) & 0x8000;
            let mag = f & 0x7fff_ffff;
            if mag > 0x7f80_0000 {
                return sign | 0x7e00;
            }
            if mag == 0x7f80_0000 {
                return sign | 0x7c00;
            }
            if mag >= 0x477f_f000 {
                return sign | 0x7bff;
            }
            if mag < 0x3300_0000 {
                return sign;
            }
            let e = mag >> 23;
            if e >= 113 {
                let bits = ((e - 112) << 10) | ((mag >> 13) & 0x3ff);
                let rem = mag & 0x1fff;
                if rem > 0x1000 || (rem == 0x1000 && bits & 1 == 1) {
                    return sign | (bits + 1);
                }
                return sign | bits;
            }
            let shift = 126 - e;
            let m = (mag & 0x7f_ffff) | 0x80_0000;
            let bits = m >> shift;
            let half = 1u32 << (shift - 1);
            let rem = m & ((1u32 << shift) - 1);
            if rem > half || (rem == half && bits & 1 == 1) {
                return sign | (bits + 1);
            }
            sign | bits
        }

        // Every f16 pattern, and around each of them the quarter, the exact TIE and the
        // three-quarter point on BOTH sides - the places the two candidate modes disagree.
        // A random sweep would mostly land where every mode agrees and would say nothing.
        let decode = |h: u32| -> f32 {
            let (s, e, m) = (h >> 15, (h >> 10) & 0x1f, h & 0x3ff);
            let v = if e == 0 {
                (m as f32) * 2f32.powi(-24)
            } else if e == 0x1f {
                f32::INFINITY
            } else {
                (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15)
            };
            if s == 1 { -v } else { v }
        };
        let mut checked = 0usize;
        for h in 0..0x7c00u32 {
            let (v, w) = (decode(h), decode(h + 1));
            for frac in [0.0f64, 0.25, 0.5, 0.75] {
                for sign in [1.0f64, -1.0] {
                    let x = (sign * (v as f64 + (w - v) as f64 * frac)) as f32;
                    let want = crate::fold::f32_to_f16_bits_saturating(x) as u32;
                    assert_eq!(
                        wgsl_arm(x),
                        want,
                        "x={x:e} (0x{:08x}): the portable arm says 0x{:04x}, the reference 0x{want:04x}",
                        x.to_bits(),
                        wgsl_arm(x)
                    );
                    checked += 1;
                }
            }
        }
        // Zero, the two infinities, a NaN, and the overflow boundary in both directions.
        for x in [0.0f32, -0.0, 65504.0, -65504.0, 65520.0, -65520.0, 65519.996, 131008.0,
                  f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 5.96e-8, -5.96e-8, 2.98e-8] {
            assert_eq!(
                wgsl_arm(x),
                crate::fold::f32_to_f16_bits_saturating(x) as u32,
                "x={x:e} (0x{:08x})",
                x.to_bits()
            );
            checked += 1;
        }
        for x in [f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(wgsl_arm(x) & 0x7fff, 0x7c00, "an infinity stays one: {x}");
        }
        assert_eq!(wgsl_arm(f32::NAN) & 0x7c00, 0x7c00, "a NaN stays a NaN");
        assert!(wgsl_arm(f32::NAN) & 0x3ff != 0, "...with a payload, so it is not an infinity");
        assert!(checked > 100_000, "the sweep must be a sweep: {checked}");
    }

    /// And the EMITTED TEXT is the thing that ships, so it is pinned line for line against the
    /// transcription above. A test of an algorithm that the module does not contain is a test
    /// of nothing [[vitaslop-probe-the-shader-dont-simulate-it]].
    #[test]
    fn the_emitted_portable_helper_is_the_algorithm_that_was_checked() {
        for line in [
            "if (mag > 0x7f800000u) { return sign | 0x7e00u; }",
            "if (mag == 0x7f800000u) { return sign | 0x7c00u; }",
            "if (mag >= 0x477ff000u) { return sign | 0x7bffu; }",
            "if (mag < 0x33000000u) { return sign; }",
            "let bits = ((e - 112u) << 10u) | ((mag >> 13u) & 0x3ffu);",
            "if (rem > 0x1000u || (rem == 0x1000u && (bits & 1u) == 1u)) { return sign | (bits + 1u); }",
            "let shift = 126u - e;",
            "let half = 1u << (shift - 1u);",
            "if (rem > half || (rem == half && (bits & 1u) == 1u)) { return sign | (bits + 1u); }",
        ] {
            assert!(HALF_HELPERS_PORTABLE.contains(line), "missing from the emitted helper: {line}");
        }
    }
    /// >>> THE VARYING STAND-IN'S TWO HALVES MUST BE THE SAME FUNCTION, and one of them is WGSL
    /// >>> text this test cannot run - so what it pins is the SPELLING, line for line.
    ///
    /// The Rust half (`case_texture_value_at`) is what the reference fetches; the WGSL half is
    /// what the module computes. If they drift, every program that samples anything diverges
    /// and it reads as a translation defect in the shader rather than in the rig.
    #[test]
    fn the_varying_texture_stand_in_is_one_function_spelled_twice() {
        crate::link::set_arm(crate::link::CASE_TEX_ARM, "vary");
        let body = "  let _tex0 = textureSample(t3, s3, vec2<f32>(bitcast<f32>(pa[0]), bitcast<f32>(pa[1])));
";
        let (module, units) = wrap_compute_module_for(body, ProgramKind::Fragment, &[]);
        assert_eq!(units, vec![3], "the unit is taken from the binding name: {module}");
        // The COORDINATE reaches the stand-in, which is the whole point of the arm.
        assert!(
            module.contains("gxp_case_tex(3u, vec3<f32>(vec2<f32>(bitcast<f32>(pa[0]), bitcast<f32>(pa[1])), 0.0))"),
            "the coordinate must be passed through verbatim:
{module}"
        );
        // ...and the taming is spelled exactly as the Rust twin computes it.
        assert!(
            module.contains("let t = min(max(c, vec3<f32>(-4.0)), vec3<f32>(4.0)) * 0.25;"),
            "got:
{module}"
        );
        let v = case_texture_value(3);
        assert!(
            module.contains(&format!("{:?} + t.x", v[0])) && module.contains(&format!("{:?})", v[3])),
            "the per-unit constants are the same ones the reference adds to:
{module}"
        );
        // A 3-COORD sample keeps its third component rather than being padded.
        let cube = "  let _tex1 = textureSample(t1, s1, vec3<f32>(a, b, c));
";
        let (m3, _) = wrap_compute_module_for(cube, ProgramKind::Fragment, &[]);
        assert!(m3.contains("gxp_case_tex(1u, vec3<f32>(a, b, c))"), "got:
{m3}");
        // A sample carrying a BIAS after the coordinate must not swallow it into the argument -
        // it travels as the LOD operand, tagged with the builtin it came from.
        let biased = "  let _tex2 = textureSampleBias(t2, s2, vec2<f32>(a, b), 0.5);
";
        let (mb, _) = wrap_compute_module_for(biased, ProgramKind::Fragment, &[]);
        assert!(
            mb.contains("gxp_case_texl(2u, vec3<f32>(vec2<f32>(a, b), 0.0), 1u, vec4<f32>(0.5, 0.0, 0.0, 0.0));"),
            "got:
{mb}"
        );
        // A GRADIENT's two derivative vectors fill the four LOD channels in order.
        let grad = "  let _tex2 = textureSampleGrad(t2, s2, vec2<f32>(a, b), vec2<f32>(d0, d1), vec2<f32>(d2, d3));
";
        let (mg, _) = wrap_compute_module_for(grad, ProgramKind::Fragment, &[]);
        assert!(
            mg.contains("gxp_case_texl(2u, vec3<f32>(vec2<f32>(a, b), 0.0), 3u, vec4<f32>(vec2<f32>(d0, d1), vec2<f32>(d2, d3)));"),
            "got:
{mg}"
        );
        // ...and the LOD form's terms are the Rust twin's, in its order.
        for line in [
            "let l = min(max(lod, vec4<f32>(-4.0)), vec4<f32>(4.0)) * 0.25;",
            "let km = b.w + 0.0625 * f32(mode);",
            "return vec4<f32>(b.x + l.w, b.y + l.z, b.z + l.y, km + l.x);",
        ] {
            assert!(mg.contains(line), "missing `{line}`:\n{mg}");
        }

        // And with the arm OFF the coordinate is discarded again, which is what every
        // measurement before this arm existed used.
        crate::link::set_arm(crate::link::CASE_TEX_ARM, "const");
        let (off, _) = wrap_compute_module_for(body, ProgramKind::Fragment, &[]);
        assert!(off.contains("gxp_case_tex(3u);"), "got:
{off}");
        assert!(!off.contains("min(max(c,"), "got:
{off}");
    }

    /// The Rust half's taming, checked at the places it can go wrong: inside the range, at both
    /// clamp bounds, past them, and on a value no comparison answers usefully.
    #[test]
    fn the_varying_stand_in_tames_its_coordinate_exactly() {
        use crate::interp::TexLodArg;
        let at = |c: [f32; 3]| case_texture_value_at(2, c, TexLodArg::IMPLICIT);
        let k = case_texture_value(2);
        // Inside the range: a quarter of the coordinate, exactly.
        let v = at([1.0, -2.0, 0.5]);
        assert_eq!(v[0], k[0] + 0.25);
        assert_eq!(v[1], k[1] - 0.5);
        assert_eq!(v[2], k[2] + 0.125);
        // Channel 3 is the pure constant, so a garbage coordinate leaves one readable channel.
        assert_eq!(v[3], k[3]);
        // At and past the bounds.
        assert_eq!(at([4.0, -4.0, 0.0])[0], k[0] + 1.0);
        assert_eq!(at([1.0e38, -1.0e38, 0.0])[0], k[0] + 1.0);
        assert_eq!(at([1.0e38, -1.0e38, 0.0])[1], k[1] - 1.0);
        // A NaN lands on a bound rather than propagating - `f32::max` returns the non-NaN
        // operand, which is the same answer WGSL's `max` gives.
        assert!(at([f32::NAN, 0.0, 0.0])[0].is_finite());

        // THE LOD FORMS: a bias and a level of the same value must DIFFER (that is the whole
        // point), and each gradient channel must land in its own output channel.
        let lod = |mode, args| case_texture_value_at(2, [0.0; 3], TexLodArg { mode, args });
        let bias = lod(TexLod::Bias, [1.0, 0.0, 0.0, 0.0]);
        let level = lod(TexLod::Level, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(bias[3], k[3] + 0.0625 + 0.25);
        assert_eq!(level[3], k[3] + 0.125 + 0.25);
        assert_eq!(bias[..3], level[..3]);
        let g = lod(TexLod::Gradient, [0.5, 1.0, 2.0, -4.0]);
        assert_eq!(g, [k[0] - 1.0, k[1] + 0.5, k[2] + 0.25, k[3] + 0.1875 + 0.125]);
    }

    /// >>> THE CUBE INVERSE, AGAINST THE FORWARD RULE WRITTEN OUT SEPARATELY.
    ///
    /// The render rig does not hand a cube sampler an arbitrary direction and then try to model
    /// which face the hardware picked. It goes the other way: it decides the face and the texel,
    /// and builds the direction that names them - so the reference reads that texel directly and
    /// the hardware's major-axis selection never enters the comparison.
    ///
    /// That is only sound if the inverse is right, and a table of eighteen signed terms is
    /// exactly the shape that is wrong in one entry and agrees with itself everywhere. So this
    /// applies the FORWARD rule - the one the graphics specifications state, transcribed here
    /// and nowhere else in this crate - to the direction the table builds, and requires it to
    /// come back to the same face and the same texel.
    ///
    /// It also pins the two properties the exactness rests on: the major axis wins by a real
    /// margin (no tie to break), and the recovered coordinate is EXACT rather than merely close - every
    /// value in the chain is a small multiple of `1 / CASE_TEX_SIZE`, which f32 holds exactly.
    #[test]
    fn the_cube_direction_selects_the_face_and_texel_it_names() {
        // The forward mapping, from the specification: which face a direction names, and the
        // face coordinates it yields. Deliberately a second, independent statement.
        fn forward(d: [f32; 3]) -> (usize, f32, f32, f32) {
            let (x, y, z) = (d[0], d[1], d[2]);
            let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
            if ax >= ay && ax >= az {
                if x > 0.0 { (0, -z, -y, ax) } else { (1, z, -y, ax) }
            } else if ay >= az {
                if y > 0.0 { (2, x, z, ay) } else { (3, x, -z, ay) }
            } else if z > 0.0 {
                (4, x, -y, az)
            } else {
                (5, -x, -y, az)
            }
        }

        let n = CASE_TEX_SIZE as f32;
        for face in 0..6usize {
            for &i in &[0u32, 1, 7, 31, CASE_TEX_SIZE - 1] {
                for &j in &[0u32, 2, 30, CASE_TEX_SIZE - 1] {
                    let u = (i as f32 + 0.5) / n;
                    let v = (j as f32 + 0.5) / n;
                    let (sc, tc) = (2.0 * u - 1.0, 2.0 * v - 1.0);
                    let d = case_cube_direction(face, sc, tc);

                    // The major axis wins outright: the other two components are at most
                    // `1 - 1/size` in magnitude, so nothing here depends on how a tie is broken.
                    let mag = [d[0].abs(), d[1].abs(), d[2].abs()];
                    let major = mag.iter().cloned().fold(0.0f32, f32::max);
                    assert_eq!(major, 1.0, "face {face} texel ({i},{j}): direction {d:?}");
                    let others: f32 = mag.iter().cloned().filter(|m| *m != 1.0).fold(0.0, f32::max);
                    assert!(
                        others <= 1.0 - 1.0 / n,
                        "face {face} texel ({i},{j}): a minor axis reached {others}, so the face \
                         selection is a tie-break rather than a decision"
                    );

                    let (got_face, gsc, gtc, ma) = forward(d);
                    assert_eq!(got_face, face, "texel ({i},{j}) landed on the wrong face: {d:?}");
                    // EXACT, not within a tolerance: every value is a multiple of 1/size.
                    let (gu, gv) = (0.5 * (gsc / ma + 1.0), 0.5 * (gtc / ma + 1.0));
                    assert_eq!(gu, u, "face {face} texel ({i},{j}): u did not come back exactly");
                    assert_eq!(gv, v, "face {face} texel ({i},{j}): v did not come back exactly");
                    assert_eq!((gu * n).floor() as u32, i, "face {face}: wrong texel column");
                    assert_eq!((gv * n).floor() as u32, j, "face {face}: wrong texel row");
                }
            }
        }
    }

    /// The generated WGSL switch is the table's own text, and it is a `vec3` per face with a
    /// `default` arm - which is what makes the helper exhaustive for a `u32` selector.
    #[test]
    fn the_cube_switch_is_generated_from_the_one_table() {
        let src = case_uv_fns();
        for f in 0..5 {
            assert!(src.contains(&format!("case {f}u: {{ return vec3<f32>(")), "missing face {f}:\n{src}");
        }
        assert!(src.contains("default: { return vec3<f32>("), "no default arm:\n{src}");
        assert!(!src.contains("CASE_CUBE_SWITCH") && !src.contains("CASE_TEX_SIZEf"), "unsubstituted:\n{src}");
        // Face +X is `( 1.0, -tc, -sc)`. Spelled out once here so a table edited by accident
        // has to be edited here too.
        assert!(src.contains("case 0u: { return vec3<f32>(1.0, -tc, -sc); }"), "+X face changed:\n{src}");
    }

    /// >>> THE GATHER QUANTISER'S WHOLE REASON IS THE MARGIN, and the margin is invisible in the
    /// values it returns unless something checks it.
    ///
    /// `floor(uv * size - 0.5)` is what picks a gather's 2x2 footprint, and a coordinate that
    /// lands ON that boundary is one the hardware's subtexel rounding may take either side of -
    /// which would move all four texels and be reported as a translation defect. So the
    /// quantiser puts the sample a quarter of a texel clear of it, at every input.
    ///
    /// It must ALSO leave the fraction free to move, or the bilinear coefficients the
    /// instruction computes would be the same four constants in every case and would check
    /// nothing. Both properties at once is what the `+ 0.75 + frac * 0.5` form buys, and this
    /// is what says so.
    #[test]
    fn the_gather_quantiser_stays_clear_of_the_footprint_boundary() {
        let mut seen_low = false;
        let mut seen_high = false;
        // Inputs across the whole tamed range and past both ends, plus the shapes that break a
        // naive quantiser: a NaN, an infinity, a denormal, an exact bound.
        let mut inputs: Vec<f32> = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e-45, -4.0, 4.0, 0.0];
        for k in 0..400 {
            inputs.push(-4.5 + 9.0 * (k as f32) / 400.0);
        }
        for c in inputs {
            let (_, frac) = case_render_gather(0, [c, c]);
            for f in frac {
                assert!(
                    (0.25..=0.75).contains(&f),
                    "input {c}: the gather fraction is {f}, which is not a quarter texel clear \
                     of the boundary the footprint's floor turns on"
                );
                if f < 0.35 {
                    seen_low = true;
                }
                if f > 0.65 {
                    seen_high = true;
                }
            }
        }
        assert!(
            seen_low && seen_high,
            "the fraction never moved across its range, so the bilinear coefficients this feeds \
             are constants and check nothing"
        );
    }

    /// The footprint is the 2x2 at the texel the quantiser named, CLAMPED at the edges - which
    /// is the addressing mode the rig binds, and the one place a corner texel is read twice.
    #[test]
    fn the_gather_footprint_is_the_two_by_two_the_quantiser_named() {
        // A coordinate in the middle: four DISTINCT texels, in the platform's order.
        let c = 0.0f32;
        let (i, _) = case_tex_index_frac(c);
        let (texels, _) = case_render_gather(3, [c, c]);
        let at = |x: u32, y: u32| case_texel_value(3, 0, x, y)[0];
        assert_eq!(texels, [at(i, i + 1), at(i + 1, i + 1), at(i + 1, i), at(i, i)]);
        assert_eq!(
            texels.iter().filter(|v| **v == texels[0]).count(),
            1,
            "the four texels of a gather must differ, or the instruction checks nothing"
        );
        // At the top edge the clamp makes two pairs equal rather than reading off the texture.
        let (edge, _) = case_render_gather(3, [100.0, 100.0]);
        let last = CASE_TEX_SIZE - 1;
        assert_eq!(edge[1], at(last, last), "the clamped corner");
        assert_eq!(edge, [at(last, last); 4], "every texel of the footprint clamps to the corner");
    }

    /// A render module is real WGSL - including the cube binding, whose type must match the
    /// three-component coordinate the quantiser produces, and the `discard` rewrite, which sits
    /// inside whatever block the predicate put it in.
    #[test]
    fn a_render_case_module_is_valid_wgsl() {
        let body = "  let _tex0 = textureSample(t1, s1, vec2<f32>(bitcast<f32>(pa[0]), bitcast<f32>(pa[1])));\n\
                    \x20 r[0] = bitcast<u32>(_tex0.x);\n\
                    \x20 let _tex1 = textureSampleBias(t2, s2, vec3<f32>(bitcast<f32>(pa[2]), bitcast<f32>(pa[3]), bitcast<f32>(pa[4])), 0.5);\n\
                    \x20 r[1] = bitcast<u32>(_tex1.y);\n\
                    \x20 let _guv2 = vec2<f32>(bitcast<f32>(pa[5]), bitcast<f32>(pa[6]));\n\
                    \x20 let _g2 = textureGather(0u, t1, s1, _guv2);\n\
                    \x20 let _gf2 = fract(_guv2 * vec2<f32>(textureDimensions(t1, 0u)) - vec2<f32>(0.5));\n\
                    \x20 r[2] = bitcast<u32>(_g2.x + _gf2.x);\n\
                    \x20 r[3] = bitcast<u32>(dpdx(bitcast<f32>(pa[7])));\n\
                    \x20 if (p[0]) {\n\
                    \x20 discard;\n\
                    \x20 }\n\
                    \x20 gxp_frag_depth = gxp_depth_to_window(bitcast<f32>(pa[8]), gxp_interp_depth);\n";
        let units = [
            TexBinding { unit: 1, coords: 2, cube: false, raw: false },
            TexBinding { unit: 2, coords: 3, cube: true, raw: false },
        ];
        let (module, rewrites) =
            wrap_render_case_module_for(body, ProgramKind::Fragment, &[], &units).expect("render module");
        assert_eq!(rewrites, RenderRewrites { samples: 2, gathers: 1, kills: 1 });
        // The two uniformity-bearing forms became explicit level-0 samples; the cube one took
        // the cube quantiser and the flat one the flat quantiser.
        assert!(module.contains("textureSampleLevel(t1, s1, gxp_case_uv2("), "flat sample:\n{module}");
        assert!(module.contains("textureSampleLevel(t2, s2, gxp_case_uv3c("), "cube sample:\n{module}");
        assert!(!module.contains("textureSampleBias"), "a biased sample survived:\n{module}");
        assert!(module.contains("var t2: texture_cube<f32>;"), "cube binding:\n{module}");
        assert!(module.contains("let _guv2 = gxp_case_uv2g("), "gather coordinate:\n{module}");
        assert!(!module.contains("discard;"), "the discard was not rewritten:\n{module}");
        assert!(module.contains("gxp_killed = 1u;"), "the kill is not recorded:\n{module}");
        let m = naga::front::wgsl::parse_str(&module)
            .unwrap_or_else(|e| panic!("render module failed to parse:\n{module}\n\n{e:?}"));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&m)
            .unwrap_or_else(|e| panic!("render module failed validation:\n{module}\n\n{e:?}"));
    }

    /// >>> AN UNREWRITTEN SAMPLE MUST BE COUNTABLE, because it is not a compile error.
    ///
    /// A `textureSample` the parse missed stays a real sample on an UNQUANTISED coordinate: the
    /// module compiles, runs, and picks whichever texel the device's subtexel rounding lands on.
    /// The only thing between that and a fabricated divergence is the caller comparing these
    /// counts against the instruction stream, so the counts have to be honest about a miss.
    #[test]
    fn a_sample_the_rewrite_cannot_parse_is_not_counted() {
        let units = [TexBinding { unit: 1, coords: 2, cube: false, raw: false }];
        // A shape `emit_tex` never writes: the coordinate is not a `vecK<f32>` constructor.
        let odd = "  let _tex0 = textureSample(t1, s1, someOtherCoord);\n";
        let (module, rewrites) =
            wrap_render_case_module_for(odd, ProgramKind::Fragment, &[], &units).expect("module");
        assert_eq!(rewrites.samples, 0, "an unparsed sample was counted as rewritten:\n{module}");
        assert!(module.contains("textureSample(t1, s1, someOtherCoord)"), "left alone:\n{module}");
    }
}
