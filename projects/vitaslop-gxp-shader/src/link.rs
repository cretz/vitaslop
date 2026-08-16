//! Cross-stage LINKAGE: pair a recompiled vertex program with a recompiled fragment
//! program into a single, bindable WGSL module whose vertex outputs feed the fragment
//! inputs through matched `@location` varyings.
//!
//! A `SceGxmProgram` vertex shader and the fragment shader it draws with do not carry each
//! other's binding layout, and the two stages do NOT share a register layout. The vertex
//! writes ONE interpolated float per OUTPUT lane; the fragment receives each varying in its
//! PRIMARY-ATTRIBUTE (PA) bank at the precision its own descriptor declares, so an F16 varying
//! arrives as two packed halves in ONE PA register and costs TWO vertex lanes. Matching the
//! stages register-to-register is therefore wrong by a factor of two on every F16 varying.
//!
//! Both sides state their own layout, and linking means matching them BY USAGE:
//!
//! * the VERTEX varyings block lists each TEXCOORD's component width, placed in ascending
//!   index from output lane 6 - and the container's own total output-lane count checks that
//!   placement ([`Program::output_varyings`]);
//! * the FRAGMENT varyings block lists each interpolant's usage, PA register base, register
//!   span and precision ([`Program::interpolants`]).
//!
//! For a usage present on both sides the two statements must agree exactly: a varying of `n`
//! components occupies `n` PA registers at F32 or `ceil(n/2)` at F16. That equality is checked
//! per usage and is what pins the interface - a disagreement means one side was decoded wrong,
//! so the pair hard-fails to the fixed-function fallback rather than route every later varying
//! to the wrong component. This is the same no-guess / no-silent-degrade contract the emitter
//! and NID dispatcher hold: a wrong translation can never paint a pixel.
//!
//! A fragment interpolant whose registers the code never reads before writing is not routed at
//! all (it cannot affect the picture); one that IS read but has no matching vertex output is a
//! hard failure. PA registers above the declared interpolants are the fragment's own scratch
//! (the bank is reused for computed / dependent texture coordinates) and are not varyings.
//!
//! ## Binding namespace
//!
//! The vertex and fragment stages share one WGSL `@group`/`@binding` namespace inside a
//! pipeline, so their resources cannot collide. The linked module places:
//!
//! * the vertex default-uniform buffer (SA bank) at `@group(0) @binding(0)`,
//! * the fragment default-uniform buffer (SA bank) at `@group(1) @binding(0)`,
//! * the fragment's sampled textures + samplers at `@group(2)` (`t{u}` = binding `2*i`,
//!   `s{u}` = `2*i+1`, ascending by sampler unit).
//!
//! The vertex-output to fragment-input linkage is carried by the separate `@location`
//! interpolant namespace: shared varying lane `k` lives in `@location(k / 4)` component
//! `k % 4`, written by the vertex stage and read by the fragment stage.

use core::fmt::Write as _;

use crate::container::{
    OutputVarying, ParseError, Program, ProgramKind, VaryingOrder, VaryingUsage,
};
use crate::ir::{Bank, Op, Shader};
use crate::module::{plan_bindings, plan_vertex_bindings, BindingPlan, ColorOutput, VertexBindingPlan};
use crate::wgsl::{emit_body, EmitError, TexBinding, BANK_REGS};
use crate::{recompile_fragment, recompile_vertex, RecompileError};

/// A vertex program linked to a fragment program: one WGSL module carrying both entry points
/// with a matched varying interface, plus the binding plans the renderer needs to feed each
/// stage. Produced only when the pair links faithfully (see [`link_programs`]).
#[derive(Debug, Clone)]
pub struct LinkedProgram {
    /// The complete WGSL module source: `@vertex fn vs_main(...)` + `@fragment fn fs_main(...)`
    /// sharing the varying `@location` interface. wgpu builds a render pipeline from this one
    /// module referencing both entry points.
    pub wgsl: String,
    /// What the renderer must bind for the vertex stage (attributes + `@group(0)` uniform).
    pub vertex_bindings: VertexBindingPlan,
    /// What the renderer must bind for the fragment stage (`@group(1)` uniform + `@group(2)`
    /// samplers). Its `pa_lane_count` is fed by the varyings, not a direct binding.
    pub fragment_bindings: BindingPlan,
    /// Number of `@location` vec4 varyings the vertex stage declares (it may write more than
    /// the fragment reads; the surplus is interpolated and ignored, which WebGPU permits).
    pub vertex_varyings: u32,
    /// Number of `@location` vec4 varyings the fragment stage reads (`<= vertex_varyings`).
    pub fragment_varyings: u32,
    /// Content hash of the vertex blob (pipeline-cache key half).
    pub vertex_hash: u64,
    /// Content hash of the fragment blob (pipeline-cache key half).
    pub fragment_hash: u64,
}

/// Why a vertex+fragment pair could not be linked into a faithful WGSL module. Every variant
/// sends the renderer to its fixed-function fallback rather than bind a wrong interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkError {
    /// The vertex blob failed to parse.
    VertexParse(ParseError),
    /// The fragment blob failed to parse.
    FragmentParse(ParseError),
    /// The "vertex" program was not a vertex shader (or the "fragment" not a fragment).
    WrongKind,
    /// The vertex program could not be recompiled to WGSL (names the underlying gap).
    VertexRecompile(RecompileError),
    /// The fragment program could not be recompiled to WGSL (names the underlying gap).
    FragmentRecompile(RecompileError),
    /// The vertex program does not write all four lanes of clip POSITION into `o0..o3`, which is
    /// what the rasteriser consumes. `written` is the o-bank lane mask over `o0..o3` as the
    /// instruction walk saw it. Fall back.
    ///
    /// This and [`LinkError::VertexVaryingsUndecoded`] used to be one variant whose message ORed
    /// them together, so a real failure named two possible causes and settled neither. They are
    /// different defects - one is a gap in the OUTPUT-bank write model, the other in the varying
    /// BLOCK decode - and they are not fixed in the same place.
    VertexClipPositionNotWritten { written: [bool; 4] },
    /// The vertex program's varyings block did not decode, so where its outputs land is unknown
    /// and no fragment input can be fed from it faithfully. `why` is the decode's own reason.
    /// Fall back.
    VertexVaryingsUndecoded { why: &'static str },
    /// The vertex program's varying ORDER is not stated by its own container, and searching
    /// every permutation against this fragment did not leave exactly one that both stages'
    /// declarations admit. `surviving` is how many did - 0 means the linker's model is
    /// wrong somewhere else, more than 1 means the blobs genuinely do not pin the order and
    /// picking one would be a confident wrong picture. Fall back.
    VaryingOrderAmbiguous { varyings: usize, surviving: usize },
    /// The fragment reads an interpolant with usage `usage` that the vertex program does not
    /// produce, so it would sample an uninterpolated value. A wrong pairing, or a usage
    /// (colour, fog, position) whose vertex-side placement is not established. Fall back.
    UnfedVarying { usage: VaryingUsage },
    /// The two stages disagree about one varying's size: the vertex produces
    /// `vertex_components` interpolated components for `usage`, which must occupy exactly
    /// `vertex_components` PA registers at F32 or `ceil(vertex_components / 2)` at F16, but the
    /// fragment descriptor spans `fragment_registers`. One side is decoded wrong; routing on
    /// either would shift every later varying. Fall back.
    VaryingSizeMismatch { usage: VaryingUsage, fragment_registers: u32, vertex_components: u32, half: bool },
    /// The fragment code reads PA register `register` before writing it, but the register lies
    /// beyond the `primary_regs` the container allocates - so it is neither a declared varying
    /// nor allocated scratch. The operand decode or the interpolant span is wrong. Fall back.
    PaReadBeyondAllocation { register: u32, primary_regs: u32 },
    /// The fragment code reads PA register `register` before writing it, and no declared
    /// interpolant covers it - so nothing in the pipeline supplies its value. Emitting anyway
    /// would silently read a zero-initialised register, which is how a shader that links
    /// "successfully" can paint black. The interpolant layout decode is incomplete for this
    /// program. Fall back.
    PaReadUnfed { register: u32, varyings_error: Option<&'static str> },
    /// The linked varying count exceeds what WebGPU guarantees (16 inter-stage vec4s). Fall back.
    TooManyVaryings { needed: u32, limit: u32 },
    /// A stage reads an SA register that is neither in its default uniform buffer nor a
    /// container literal - it lives in the texture-control-word region, whose contents are GPU
    /// texture state this recompiler does not reproduce as shader-visible data. Fall back.
    SecondaryAttrOutOfRange { register: u32, uniform_regs: u32 },
    /// A varying descriptor declares a prefetched sample from a texture unit the program's own
    /// parameter table does not declare as a sampler, so its dimensionality is unknown and
    /// binding it would guess at GPU state the shader never asked for. Fall back.
    PrefetchUnitNotDeclared { unit: u8 },
    /// A prefetched sample needs `needed` coordinate components but the vertex produces only
    /// `available` for the texcoord that feeds it, so the missing coordinates would sample at an
    /// arbitrary position. Fall back.
    PrefetchCoordTooNarrow { unit: u8, needed: u32, available: u32 },
}

impl core::fmt::Display for LinkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LinkError::VertexParse(e) => write!(f, "vertex GXP parse error: {e:?}"),
            LinkError::FragmentParse(e) => write!(f, "fragment GXP parse error: {e:?}"),
            LinkError::WrongKind => write!(f, "program kind mismatch (expected a vertex + a fragment)"),
            LinkError::VertexRecompile(e) => write!(f, "vertex recompile failed: {e}"),
            LinkError::FragmentRecompile(e) => write!(f, "fragment recompile failed: {e}"),
            LinkError::VertexClipPositionNotWritten { written } => write!(
                f,
                "the vertex program does not write clip POSITION into o0..o3 - of those four lanes \
                 it writes [{}]",
                (0..4)
                    .map(|i| if written[i] { format!("o{i}") } else { format!("-{i}") })
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            LinkError::VertexVaryingsUndecoded { why } => write!(
                f,
                "the vertex program's varyings block did not decode ({why}), so where its outputs \
                 land is unknown"
            ),
            LinkError::VaryingOrderAmbiguous { varyings, surviving } => write!(
                f,
                "the vertex program's varyings block does not state the ORDER of its {varyings} \
                 outputs, and {surviving} of the {} possible orders link consistently against \
                 this fragment - so the two programs' own declarations do not pin it down",
                (1..=*varyings).product::<usize>()
            ),
            LinkError::UnfedVarying { usage } => write!(
                f,
                "fragment reads a {usage:?} varying that the vertex program does not produce"
            ),
            LinkError::VaryingSizeMismatch { usage, fragment_registers, vertex_components, half } => write!(
                f,
                "{usage:?}: vertex produces {vertex_components} components but the fragment spans \
                 {fragment_registers} PA registers at {} precision",
                if *half { "F16" } else { "F32" }
            ),
            LinkError::PaReadBeyondAllocation { register, primary_regs } => write!(
                f,
                "fragment reads PA register {register} before writing it, beyond its {primary_regs} allocated PA registers"
            ),
            LinkError::PaReadUnfed { register, varyings_error } => write!(
                f,
                "fragment reads PA register {register} before writing it but no declared interpolant \
                 covers it, so no vertex output feeds it{}",
                match varyings_error {
                    // The interpolant list is empty because the block would not DECODE, which
                    // is the actual defect - without this the message blames the pairing.
                    Some(why) => format!(" (its varyings block did not decode: {why})"),
                    None => String::new(),
                }
            ),
            LinkError::TooManyVaryings { needed, limit } => write!(
                f,
                "linked interface needs {needed} varying locations but the pipeline supports only {limit}"
            ),
            LinkError::SecondaryAttrOutOfRange { register, uniform_regs } => write!(
                f,
                "shader reads SA register {register} outside its {uniform_regs}-register default \
                 uniform buffer and outside the container literals (texture-control region)"
            ),
            LinkError::PrefetchUnitNotDeclared { unit } => write!(
                f,
                "a varying declares a prefetched sample from texture unit {unit}, which the \
                 program does not declare as a sampler"
            ),
            LinkError::PrefetchCoordTooNarrow { unit, needed, available } => write!(
                f,
                "the prefetched sample from texture unit {unit} needs {needed} coordinate \
                 components but its texcoord supplies only {available}"
            ),
        }
    }
}

/// PA registers a prefetched sample's result occupies: two, holding its four components as
/// packed F16 halves. See [`crate::container::SamplePrefetch`].
/// The widest a prefetched sample can be, in PA registers. The per-descriptor width is
/// [`crate::container::Interpolant::prefetch_regs`]; this is only the upper bound.
const PREFETCH_REGS: u32 = 2;

impl std::error::Error for LinkError {}

/// The maximum number of `@location` inter-stage varyings a linked pipeline may use. WebGPU
/// guarantees at least 16; a pair needing more is rejected (fall back) rather than fail
/// pipeline creation.
pub const MAX_VARYINGS: u32 = 16;

/// Link a vertex + fragment `SceGxmProgram` pair into a single bindable WGSL module with a
/// matched varying interface, or return why it could not be linked faithfully (which sends
/// the caller to its fixed-function fallback). Both programs are recompiled with the same
/// strict, no-guess contract as [`recompile_vertex`] / [`recompile_fragment`]; the linkage
/// itself additionally validates the vertex output layout and every sampled varying lane.
pub fn link_programs(vbytes: &[u8], fbytes: &[u8]) -> Result<LinkedProgram, LinkError> {
    let vprog = Program::parse(vbytes).map_err(LinkError::VertexParse)?;
    let fprog = Program::parse(fbytes).map_err(LinkError::FragmentParse)?;
    if vprog.kind != ProgramKind::Vertex || fprog.kind != ProgramKind::Fragment {
        return Err(LinkError::WrongKind);
    }

    let vrc = recompile_vertex(vbytes).map_err(LinkError::VertexRecompile)?;
    let frc = recompile_fragment(fbytes).map_err(LinkError::FragmentRecompile)?;

    let vplan = plan_vertex_bindings(&vprog, &vrc.shader);
    let mut fplan =
        plan_bindings(&frc.shader, fprog.default_uniform_regs, |u| fprog.sampler_is_cube(u as u32));

    // The vertex must place clip POSITION in o0..o3 (what the rasteriser consumes) and its
    // varyings block must have VALIDATED - otherwise its varying placement is unknown.
    //
    // "Validated" is `varyings_error`, not "produced at least one varying". An empty output list
    // used to stand for both "the block did not decode" and "this program outputs clip position
    // and nothing else", and those are opposite situations: the first must fall back, the second
    // is a perfectly linkable DEPTH-ONLY program. Conflating them cost this title its whole
    // 1024x1024 shadow pass - 13 of its 16 draws fell back with a message naming a layout
    // problem that did not exist.
    let written = output_written_lanes(&vrc.shader);
    let clip: [bool; 4] = std::array::from_fn(|l| written.get(l).copied().unwrap_or(false));
    if !clip.iter().all(|w| *w) {
        return Err(LinkError::VertexClipPositionNotWritten { written: clip });
    }
    if let Some(why) = vprog.varyings_error {
        return Err(LinkError::VertexVaryingsUndecoded { why });
    }

    // Match the two stages' own statements of the interface, by usage.
    let iface = plan_interface(&vprog, &fprog, &frc.shader)?;
    let varyings = &iface.components;

    // A PASSTHROUGH fragment program emits the PDS-loaded primary attributes verbatim (see
    // `plan_interface`), so its colour is the primary-attribute bank at register 0 - never the
    // OUTPUT bank, which nothing in the program ever writes. `plan_bindings` infers the colour
    // register from what the stream WRITES and cannot answer for a stream that writes nothing;
    // its default (`NativeO0`) would read four registers the hardware never fills.
    //
    // The precision comes from the interpolant that lands there: a HALF descriptor packs four
    // components into two registers, which is the shape `ColorPrecision::F16` unpacks.
    if is_passthrough(&frc.shader, &pa_read_before_write(&frc.shader).0) {
        fplan.color = ColorOutput::NonNativePa0;
        fplan.color_precision = match fprog.interpolants.iter().find(|it| it.pa_base == 0) {
            Some(it) if it.half => crate::module::ColorPrecision::F16,
            _ => crate::module::ColorPrecision::F32,
        };
    }

    // A prefetched unit is usually sampled ONLY by the PDS, so the instruction walk that built
    // the binding plan never saw it. Merge those units in so the renderer binds them.
    for pf in &iface.prefetches {
        match fplan.samplers.iter_mut().find(|b| b.unit == pf.unit) {
            Some(b) => b.coords = b.coords.max(pf.binding().coords),
            None => fplan.samplers.push(pf.binding()),
        }
    }
    fplan.samplers.sort_unstable_by_key(|b| b.unit);

    // Container literals the driver preloads into SA registers above the uniform buffer. An SA
    // read that is neither a uniform nor a literal is unmodeled texture state and hard-fails.
    let vliterals = secondary_attr_init(&vrc.shader, &vprog)?;
    let fliterals = secondary_attr_init(&frc.shader, &fprog)?;

    // Interpolated scalar components packed four per `@location` vec4. Both stages declare the
    // same interface, so the counts are equal by construction.
    let fragment_varyings = (varyings.len() as u32).div_ceil(4);
    let vertex_varyings = fragment_varyings;
    if vertex_varyings > MAX_VARYINGS {
        return Err(LinkError::TooManyVaryings { needed: vertex_varyings, limit: MAX_VARYINGS });
    }

    // Each stage's statements are its SECONDARY program followed by its primary one. The
    // secondary program runs first on the hardware and exists to leave values in SA registers
    // the primary reads, so skipping it does not lose a detail - it leaves those registers
    // holding whatever the default uniform buffer had, which is how an unrelated matrix element
    // ends up scaling a surface's colour to black.
    let vbody = format!(
        "{}{}",
        emit_secondary_body(&vprog).map_err(|e| LinkError::VertexRecompile(e.into()))?,
        emit_body(&vrc.shader).map_err(|e: EmitError| LinkError::VertexRecompile(e.into()))?
    );
    let fbody = format!(
        "{}{}",
        emit_secondary_body(&fprog).map_err(|e| LinkError::FragmentRecompile(e.into()))?,
        emit_body(&frc.shader).map_err(|e: EmitError| LinkError::FragmentRecompile(e.into()))?
    );

    let wgsl = build_linked_module(
        &vbody,
        &vplan,
        &vprog,
        &vliterals,
        &fbody,
        &fplan,
        &fprog,
        &fliterals,
        &iface,
        fragment_varyings,
        frc.shader.instrs.iter().any(|i| i.op == Op::DepthF),
    );

    Ok(LinkedProgram {
        wgsl,
        vertex_bindings: vplan,
        fragment_bindings: fplan,
        vertex_varyings,
        fragment_varyings,
        vertex_hash: vprog.hash,
        fragment_hash: fprog.hash,
    })
}

/// The set of OUTPUT-bank scalar lanes a vertex shader writes (indexed by lane). A lane is
/// written if any instruction whose destination is the OUTPUT bank has that lane in its write
/// mask. Sized to [`BANK_REGS`] so every reachable lane is representable.
fn output_written_lanes(shader: &Shader) -> Vec<bool> {
    let mut written = vec![false; BANK_REGS];
    for instr in &shader.instrs {
        let Some(d) = instr.dest.as_ref() else { continue };
        if d.bank != Bank::Output {
            continue;
        }
        for c in 0..4 {
            if instr.write_mask[c] {
                let lane = d.index as usize + c;
                if lane < written.len() {
                    written[lane] = true;
                }
            }
        }
    }
    written
}

/// The channels an instruction reads from its sources, mirroring the emitter's read model
/// ([`crate::wgsl`]): a dot reads a fixed component prefix, a texture sample reads its
/// coordinate prefix, and every other op reads a source channel only where it writes the
/// destination channel.
fn read_channels(instr: &crate::ir::Instr) -> [bool; 4] {
    match instr.op {
        Op::Dot { components } => {
            let n = (components as usize).clamp(1, 4);
            [0 < n, 1 < n, 2 < n, 3 < n]
        }
        Op::Tex { coords, .. } => {
            let n = (coords as usize).clamp(1, 4);
            [0 < n, 1 < n, 2 < n, 3 < n]
        }
        _ => instr.write_mask,
    }
}

/// The PA-bank REGISTERS a fragment shader reads before writing them, i.e. the registers whose
/// value must arrive from outside the shader. The fragment reuses the PA bank as general
/// scratch (a computed / dependent texture coordinate is written into PA and then sampled), so
/// a register written before it is read carries an intermediate value, not an interpolated
/// vertex output. Walks the stream in order (sources are evaluated before the destination is
/// assigned) and resolves each access to a 32-bit register at the accessing instruction's
/// precision: an F32 channel reads register `index + selector`, while the four F16 channels
/// share a register PAIR (`index + selector/2`).
///
/// Returns TWO answers, because the two questions the caller asks are genuinely different.
///
/// `.0` is per 16-bit HALF: register `r` is an input if ANY half of it is read before that half
/// is written. An F16 instruction writes ONE half and the emitter preserves the other (`pa[r] =
/// (pa[r] & 0xffff0000u) | ...`), so a register whose low half an early instruction fills is NOT
/// defined - a later read of its high half still needs the interpolated value. Counting the
/// whole register as written there made a material's interpolant look like scratch, so the
/// linker routed none of it and the shader read zeros for the components nothing wrote. On this
/// title's static-world material that component is the FOG FACTOR, so every wall and building
/// came out flat `fogColour` with correct textures bound and never sampled. This is the answer
/// that decides whether an interpolant's data has to be routed.
///
/// `.1` is per REGISTER: `r` is read at a point where NEITHER half has been written. That is
/// the strictly-untouched case, and it is what the unfed-varying check must use. A register the
/// shader itself partially filled and then read another lane of is reading its OWN scratch,
/// which the hardware leaves undefined too - real programs do it on lanes whose result is dead
/// (`frag_872e7aa0` computes a 4-channel MAD over a 3-channel product and discards the fourth),
/// and failing the link over one would refuse a shader that is perfectly translatable.
fn pa_read_before_write(shader: &Shader) -> (Vec<bool>, Vec<bool>) {
    // Two entries per register: index `2*r + h` is half `h`. An F32 access covers both halves.
    let mut written = vec![false; BANK_REGS * 2];
    let mut inputs = vec![false; BANK_REGS];
    let mut untouched = vec![false; BANK_REGS];
    for instr in &shader.instrs {
        // Sources and destination can be at DIFFERENT widths (a format convert), so each side
        // resolves its own registers - see [`crate::ir::Instr::source_half_precision`].
        let src_half = instr.source_half_precision();
        let half = instr.half_precision;
        let read = read_channels(instr);
        for src in &instr.srcs {
            if src.bank != Bank::PrimaryAttr {
                continue;
            }
            for c in 0..4 {
                if !read[c] {
                    continue;
                }
                let sel = src.swizzle[c] as usize;
                if sel > 3 {
                    continue; // a swizzle constant reads no register
                }
                let (reg, halves) = if src_half {
                    (src.index as usize + (sel >> 1), (sel & 1)..(sel & 1) + 1)
                } else {
                    (src.index as usize + sel, 0..2)
                };
                if reg >= BANK_REGS {
                    continue;
                }
                if halves.clone().any(|h| !written[reg * 2 + h]) {
                    inputs[reg] = true;
                }
                if !written[reg * 2] && !written[reg * 2 + 1] {
                    untouched[reg] = true;
                }
            }
        }
        if let Some(d) = instr.dest.as_ref() {
            if d.bank == Bank::PrimaryAttr {
                for c in 0..4 {
                    if !instr.write_mask[c] {
                        continue;
                    }
                    let (reg, halves) =
                        if half { (d.index as usize + (c >> 1), (c & 1)..(c & 1) + 1) } else { (d.index as usize + c, 0..2) };
                    if reg >= BANK_REGS {
                        continue;
                    }
                    for h in halves {
                        written[reg * 2 + h] = true;
                    }
                }
            }
        }
    }
    (inputs, untouched)
}

/// One interpolated scalar component crossing the stage boundary, in interface order. The
/// hardware interpolates in floats and only then packs, so an F16 varying's two halves are two
/// SEPARATE components here and are repacked in the fragment prologue - passing the packed
/// 32-bit pattern through an interpolator would blend two unrelated numbers as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VaryingComponent {
    /// OUTPUT-bank lane the vertex stage writes this component to.
    vertex_lane: u32,
    /// What the fragment stage does with it.
    dest: ComponentDest,
}

/// Where an interpolated component goes once it reaches the fragment stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComponentDest {
    /// A whole 32-bit PA register (an F32 interpolant's component).
    Register(u32),
    /// One 16-bit half of a PA register (an F16 interpolant packs two components per register).
    Half { register: u32, slot: u32 },
    /// A coordinate of a prefetched texture sample. It never reaches a PA register: the PDS
    /// consumes it before the shader runs, and only the sample's RESULT is visible to the code.
    SampleCoord { prefetch: usize, coord: u32 },
}

/// A texture sample the PDS performs before the fragment program runs, resolved to everything
/// the prologue needs to reproduce it. See [`crate::container::SamplePrefetch`].
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedPrefetch {
    /// GXM texture unit sampled.
    unit: u8,
    /// First PA register the packed F16 result components land in.
    pa_base: u32,
    /// How many PA registers the result occupies: 2 (four components) or 1 (two). See
    /// [`crate::container::Interpolant::prefetch_regs`].
    ///
    /// Writing two unconditionally is not a harmless over-write: the register after a
    /// one-register prefetch belongs to the NEXT interpolant, so it clobbers a varying the
    /// vertex stage fed correctly. That is invisible in the WGSL and shows up only as a
    /// surface shading black.
    regs: u32,
    /// Interface component indices of the sample coordinates, in order.
    coords: Vec<usize>,
    /// The sampler is a cube map, so the coordinate is a three-component direction.
    cube: bool,
}

impl PlannedPrefetch {
    /// The binding this sample needs, which the shader's own SMP instructions may not mention -
    /// a prefetched unit is often sampled ONLY by the PDS.
    fn binding(&self) -> TexBinding {
        TexBinding { unit: self.unit, coords: self.coords.len() as u8, cube: self.cube }
    }
}

/// The complete stage interface: the interpolated components plus the samples the PDS performs
/// from them before the fragment code runs.
#[derive(Debug, Clone, PartialEq)]
struct Interface {
    components: Vec<VaryingComponent>,
    prefetches: Vec<PlannedPrefetch>,
    /// PA registers the fragment reads for a varying the vertex program does not fill, and the
    /// CONSTANT the iterator supplies for them: `(register, [half0, half1], packed)`.
    ///
    /// A GXM iterator programmed for four components off a three-component vertex output does
    /// not read garbage - it produces the texture-coordinate default, `(0, 0, 0, 1)`, the same
    /// fill a vertex ATTRIBUTE gets for the components its format does not supply. One title's
    /// fog fragment declares a four-register `TexCoord(0)` against a three-component vertex
    /// output and reads all four.
    defaults: Vec<(u32, [f32; 2], bool)>,
    /// First PA register of a `Position` interpolant the fragment reads, if it declares one.
    ///
    /// This is NOT an interpolated varying: it is the rasteriser's own WINDOW coordinate -
    /// pixels in x/y, the depth-buffer value in z, and `1/w` in w - which is what Sony's Cg
    /// front end gives a fragment program's `POSITION`/`WPOS` semantic. See
    /// [`plan_interface`] for the corpus measurement that settles it.
    window_position: Option<u32>,
}

/// Match the vertex's declared varying outputs to the fragment's declared interpolants BY
/// USAGE and produce the interface between them: the flat list of interpolated components, plus
/// the texture samples the PDS takes from them before the fragment code runs.
///
/// Each side computes its own layout from its own container - the vertex from the varyings
/// block's texcoord widths, the fragment from its interpolant descriptors - so the two are
/// independent statements about one interface and must agree exactly: `n` interpolated
/// components occupy `n` PA registers at F32 or `ceil(n/2)` at F16. Any disagreement is a hard
/// [`LinkError`].
///
/// A PA register the fragment reads before writing is fed either by an interpolant's own data or
/// by a prefetched sample's result; anything else read is a decode error, because emitting it
/// would silently read a zero-initialised register. An interpolant (or a prefetch) the code
/// never reads is skipped: it cannot affect the picture, and skipping it keeps a shader that
/// merely declares an unmodeled usage linkable.
/// Say, once per distinct shape, that a fragment allocated more registers to a varying than the
/// vertex fills, so the surplus was filled with the iterator's texture-coordinate default. That
/// default is a modelling decision about hardware, not a fact read off the blob, and a shader
/// that actually depends on the value would go wrong quietly. Naming it is what makes it
/// findable.
fn report_unfilled_varying_registers(usage: VaryingUsage, vertex_components: u32, declared: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(VaryingUsage, u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((usage, vertex_components, declared)) {
        return;
    }
    eprintln!(
        "gxp link: {usage:?} is allocated {declared} fragment registers but the vertex program \
         fills only {vertex_components} components - the surplus is fed the iterator's default \
         (0, 0, 0, 1), the same fill a vertex attribute's missing components get"
    );
}

/// Report, once per distinct shape, that a fragment consumes only a PREFIX of what the vertex
/// writes for a varying.
///
/// Unconditional and deduplicated, like every other approximation this renderer makes. It used
/// to be a hard link failure; it is now an accepted reading (see `plan_interface`), and an
/// accepted reading that turns out to be wrong has to be attributable to something. If a title
/// ever renders with a texcoord that looks truncated, this line is the first thing to check.
fn report_varying_read_as_prefix(usage: VaryingUsage, vertex_components: u32, capacity: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(VaryingUsage, u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((usage, vertex_components, capacity)) {
        return;
    }
    eprintln!(
        "gxp link: the vertex program writes {vertex_components} components of {usage:?} but the \
         fragment declares room for {capacity} - the fragment reads that PREFIX and the surplus \
         is not iterated"
    );
}

/// Whether this fragment program is a PASSTHROUGH: its stream neither reads a primary-attribute
/// register nor writes a colour register, so it computes nothing and its result is whatever the
/// PDS loaded before it ran. In the corpus that is literally a single `Nop`.
///
/// BOTH halves matter. "Writes no colour register" alone is the condition
/// `RecompileError::ColorRegisterNeverWritten` names, and it is also true of a stream that reads
/// its varyings and leaves the answer in a temporary - a shape no real program has, but one the
/// unit fixtures use, and treating those as passthrough demands every declared interpolant be
/// routed whether the shader touches it or not.
fn is_passthrough(fshader: &Shader, inputs: &[bool]) -> bool {
    crate::module::writes_no_color_register(fshader) && !inputs.iter().any(|&r| r)
}


/// The most varyings a permutation search will consider. 8! is 40,320 attempts, each a
/// cheap interface plan; beyond that the factorial makes the search itself the problem, and
/// a program with nine ambiguous varyings has never been seen.
const MAX_PERMUTED_VARYINGS: usize = 8;

/// Re-lay `vout` in the order given by `perm` (indices into `vout`), recomputing each
/// varying's base lane from the origin the container established.
///
/// Only the SEQUENCE changes. The set, the widths and the lane the varying region starts
/// at are all statements the container really makes, and none of them is being re-guessed.
fn permute_varyings(vout: &[OutputVarying], perm: &[usize]) -> Vec<OutputVarying> {
    let origin = vout.iter().map(|v| v.base_lane).min().unwrap_or(0);
    let mut lane = origin;
    perm.iter()
        .map(|&i| {
            let v = OutputVarying { usage: vout[i].usage, base_lane: lane, components: vout[i].components };
            lane += vout[i].components;
            v
        })
        .collect()
}

/// Every permutation of `0..n`, smallest-first (Heap's algorithm would do, but the counts
/// here are tiny and a plain recursive build keeps the order deterministic - which matters,
/// because a tie has to be reported as a tie rather than resolved by iteration order).
fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for head in 0..n {
        for rest in permutations(n - 1) {
            let mut p = Vec::with_capacity(n);
            p.push(head);
            p.extend(rest.into_iter().map(|i| if i >= head { i + 1 } else { i }));
            out.push(p);
        }
    }
    out
}

/// Resolve a vertex program whose varying ORDER the container could not read
/// ([`VaryingOrder::Ambiguous`]) by trying every permutation against THIS fragment and
/// keeping the ones both stages' declarations admit.
///
/// # Why this is a reading and not a guess
/// The checks it filters on are not weak. Each fragment interpolant states how many PA
/// registers it spans and at what precision; the vertex states how many components it
/// produces for each usage; and the lane accounting has to close. A varying assigned to
/// the wrong slot generally lands a 2-component value where the fragment reads 4, which
/// [`plan_interface`] already rejects. So a permutation that survives is one the blobs
/// themselves permit, and if exactly ONE does, the order is determined by the data.
///
/// If several survive the pair is still refused, with the count - because "we could not
/// tell which of three" is a fact worth reporting, and picking one would be exactly the
/// confident wrong picture this whole path exists to avoid.
fn resolve_ambiguous_order(
    vprog: &Program,
    fprog: &Program,
    fshader: &Shader,
) -> Result<Vec<OutputVarying>, LinkError> {
    let vout = &vprog.output_varyings;
    if vout.len() > MAX_PERMUTED_VARYINGS {
        return Err(LinkError::VaryingOrderAmbiguous { varyings: vout.len(), surviving: 0 });
    }
    let mut survivors: Vec<Vec<OutputVarying>> = Vec::new();
    for perm in permutations(vout.len()) {
        let candidate = permute_varyings(vout, &perm);
        if plan_interface_with(vprog, fprog, fshader, &candidate).is_ok() {
            survivors.push(candidate);
        }
    }
    match survivors.len() {
        1 => Ok(survivors.pop().expect("checked")),
        n => Err(LinkError::VaryingOrderAmbiguous { varyings: vout.len(), surviving: n }),
    }
}

fn plan_interface(vprog: &Program, fprog: &Program, fshader: &Shader) -> Result<Interface, LinkError> {
    // A program whose order the container could read is planned directly against it - this
    // is every program that links today, and its behaviour is untouched.
    if vprog.output_order != VaryingOrder::Ambiguous {
        return plan_interface_with(vprog, fprog, fshader, &vprog.output_varyings);
    }
    let resolved = resolve_ambiguous_order(vprog, fprog, fshader)?;
    plan_interface_with(vprog, fprog, fshader, &resolved)
}

/// [`plan_interface`] against an EXPLICIT vertex lane layout, so the permutation search can
/// try one without mutating the program.
fn plan_interface_with(
    vprog: &Program,
    fprog: &Program,
    fshader: &Shader,
    vout: &[OutputVarying],
) -> Result<Interface, LinkError> {
    let (mut inputs, untouched) = pa_read_before_write(fshader);
    let primary_regs = fprog.primary_reg_count as u32;

    // A PASSTHROUGH fragment program - one whose instruction stream writes NEITHER colour
    // register (very often a single `Nop`) - does not compute a colour. Its result is whatever
    // the PDS left in the primary-attribute registers before it ran: the iterated varyings and
    // any prefetched sample. That is not an inference from provenance, it is what the container
    // says out loud - such a program still declares its interpolants and a non-zero
    // `primary_reg_count`, which would be meaningless for a shader that reads nothing.
    //
    // Reading the body's own reads here (which are none) routed NO varyings and returned the
    // zero-initialised register file, so every such draw painted transparent black. On one title
    // that is the engine's 2D primitive-render path: it draws a fullscreen triangle with
    // `SCE_GXM_DEPTH_FUNC_ALWAYS` and depth WRITE, and a black one over the finished world is
    // the whole frame.
    //
    // So treat the whole primary-attribute allocation as read. Anything the interpolants then
    // fail to cover is a hard error below, not a silent zero.
    let passthrough = is_passthrough(fshader, &inputs);
    if passthrough {
        inputs.resize(inputs.len().max(primary_regs as usize), false);
        inputs[..primary_regs as usize].fill(true);
    }

    // A read of a register the container does not allocate at all means the interpolant spans
    // or the operand decode are wrong - never route varyings on that.
    if let Some(reg) = (primary_regs..inputs.len() as u32).find(|&r| inputs[r as usize]) {
        return Err(LinkError::PaReadBeyondAllocation { register: reg, primary_regs });
    }
    let reads = |range: core::ops::Range<u32>| {
        range.into_iter().any(|r| inputs.get(r as usize).copied().unwrap_or(false))
    };
    // The POSITION a fragment can declare as an interpolant is the rasteriser's WINDOW
    // coordinate, not an interpolated copy of the vertex's clip position: pixels in x and y,
    // the value written to the depth buffer in z, and `1/w` in w. That is Cg's `WPOS`, and
    // Sony's shader front end for this hardware is Cg - but it is not assumed from provenance,
    // it is MEASURED on the corpus (`fragment_position_interpolant_usage`):
    //
    //   `frag_8151b0bc` computes `kDepthBias + Position.z` and writes the result as the
    //   FRAGMENT DEPTH (`0xF8 DEPTHF`). A depth write only type-checks against a value already
    //   in depth-buffer space, which the raw clip `z` is not - it still needs its `w` divide.
    //
    // The previous reading (route lanes 0..3 as an ordinary varying) differs from this one by
    // a perspective divide and a viewport scale, so it silently changed the arithmetic of
    // every shader that reprojects - soft particles, screen-space fades, depth fog.
    //
    // It is never a prefetch coordinate source (those are named by TEXCOORD index), so it only
    // has to be handled where an interpolant's own data registers are read.
    let vertex_output =
        |usage| vout.iter().find(|v| v.usage == usage).ok_or(LinkError::UnfedVarying { usage });

    let mut iface = Interface {
        components: Vec::new(),
        prefetches: Vec::new(),
        defaults: Vec::new(),
        window_position: None,
    };
    let mut fed = vec![false; inputs.len()];
    for it in &fprog.interpolants {
        let data_base = it.pa_base as u32;
        if it.usage == VaryingUsage::Position && reads(data_base..data_base + it.register_count as u32)
        {
            // Four full-precision registers is the only shape the window coordinate has; a
            // half-precision or narrower declaration would mean the descriptor means something
            // else here, and guessing a routing for it would feed the shader silent zeros.
            if it.half || it.register_count != 4 {
                return Err(LinkError::VaryingSizeMismatch {
                    usage: it.usage,
                    fragment_registers: it.register_count as u32,
                    vertex_components: 4,
                    half: it.half,
                });
            }
            iface.window_position = Some(data_base);
            for r in data_base..data_base + 4 {
                fed[r as usize] = true;
            }
        } else if reads(data_base..data_base + it.register_count as u32) {
            let vertex = vertex_output(it.usage)?;
            let n = vertex.components;
            let expected = if it.half { n.div_ceil(2) } else { n };
            let declared = it.register_count as u32;
            // The fragment may ALLOCATE more registers for a varying than the vertex fills. That
            // is only a contradiction if the code READS the surplus: the hardware iterates what
            // the vertex wrote and leaves the rest holding whatever the allocation held, so a
            // shader that reads it is reading undefined data on the console too. One title's fog
            // varying is declared four full-precision registers against a three-component vertex
            // output and never touches the fourth.
            //
            // The OPPOSITE case - the vertex writes more than the fragment declared - used to be
            // a hard failure, on the reasoning that "the surplus would land on the NEXT
            // interpolant's registers". That is true of THIS PLANNER, which copied all `n`
            // components into a smaller span, and it is what the clamp below fixes; it is not a
            // property of the hardware. The fragment reads a PREFIX.
            //
            // The evidence is a real draw plus a count. PCSA00027's tutorial pairs a vertex
            // writing four TexCoord(0) components with a fragment declaring ONE half register
            // (two components), and that draw is correct on the device - a shipping title's
            // shaders are. It is not a one-off either: across that title's 47 fragment blobs, 21
            // of 120 half-precision interpolants declare a single register while their texcoords
            // are three or four components wide
            // (`tabulate_interpolant_register_counts_by_precision`). A configuration that common
            // in a shipping title is one the hardware handles, and the only reading under which
            // it works is that the iterator fills what the fragment ASKED for.
            let capacity = if it.half { declared * 2 } else { declared };
            let n_used = n.min(capacity);
            if expected > declared {
                report_varying_read_as_prefix(it.usage, n, capacity);
            }
            // The fragment may allocate MORE registers than the vertex fills. The iterator then
            // supplies the texture-coordinate default for the trailing components - see
            // `Interface::defaults` - so those registers are fed with constants, not left zero
            // and not refused.
            if expected < declared {
                report_unfilled_varying_registers(it.usage, n, declared);
                let total = if it.half { declared * 2 } else { declared };
                let default_of = |c: u32| if c == 3 { 1.0 } else { 0.0 };
                let mut r = data_base + expected;
                while r < data_base + declared {
                    let c0 = if it.half { (r - data_base) * 2 } else { r - data_base };
                    let halves = [default_of(c0), default_of(c0 + 1)];
                    // A half-precision varying whose fill would start MID-register would need the
                    // register to mix an interpolated half with a default one. No corpus program
                    // does that, and inventing the packing would be a guess.
                    if it.half && c0 < n {
                        return Err(LinkError::VaryingSizeMismatch {
                            usage: it.usage,
                            fragment_registers: declared,
                            vertex_components: n,
                            half: it.half,
                        });
                    }
                    let _ = total;
                    iface.defaults.push((r, halves, it.half));
                    if let Some(slot) = fed.get_mut(r as usize) {
                        *slot = true;
                    }
                    r += 1;
                }
            }
            // `n_used`, not `n`: copying the vertex's surplus components would write PAST this
            // varying's declared span and into the next interpolant's registers.
            for c in 0..n_used {
                let register = data_base + if it.half { c / 2 } else { c };
                iface.components.push(VaryingComponent {
                    vertex_lane: vertex.base_lane + c,
                    dest: match it.half {
                        true => ComponentDest::Half { register, slot: c % 2 },
                        false => ComponentDest::Register(register),
                    },
                });
                fed[register as usize] = true;
            }
        }

        // The prefetched sample's packed F16 components sit in the register(s) after the data -
        // one or two, as the descriptor says (see `Interpolant::prefetch_regs`).
        let (Some(pf), Some(pa_base)) = (it.prefetch, it.prefetch_base().map(u32::from)) else {
            continue;
        };
        let prefetch_regs = u32::from(it.prefetch_regs);
        if !reads(pa_base..pa_base + prefetch_regs) {
            continue; // the PDS fetched it, but this shader never looks at the result
        }
        // The unit must be one the program itself declares, or the renderer would bind a
        // texture the shader never asked for.
        let Some(sampler) = fprog.sampler_at(pf.unit as u32) else {
            return Err(LinkError::PrefetchUnitNotDeclared { unit: pf.unit });
        };
        // The coordinate is a plain interpolated texcoord - that is what makes the fetch
        // non-dependent and lets the PDS issue it ahead of the shader. It is often NOT one of
        // this program's own interpolants: a texcoord only the PDS reads costs no PA registers.
        let usage = VaryingUsage::TexCoord(pf.source_texcoord);
        let source = vertex_output(usage)?;
        let cube = sampler.sampler_cube;
        let coords = if cube { 3 } else { 2 };
        if source.components < coords {
            return Err(LinkError::PrefetchCoordTooNarrow {
                unit: pf.unit,
                needed: coords,
                available: source.components,
            });
        }
        let prefetch = iface.prefetches.len();
        let first = iface.components.len();
        for c in 0..coords {
            iface.components.push(VaryingComponent {
                vertex_lane: source.base_lane + c,
                dest: ComponentDest::SampleCoord { prefetch, coord: c },
            });
        }
        iface.prefetches.push(PlannedPrefetch {
            unit: pf.unit,
            pa_base,
            regs: prefetch_regs,
            coords: (first..first + coords as usize).collect(),
            cube,
        });
        for r in pa_base..pa_base + prefetch_regs {
            fed[r as usize] = true;
        }
    }

    // Every register the fragment reads with NOTHING of it written must now be fed. One that is
    // not would emit as a read of a zero-initialised register - a silently wrong picture rather
    // than a fallback - so it is a hard error, not a gap to paper over.
    //
    // The strict (`untouched`) half of the analysis is deliberate here: a register the shader
    // itself half-filled and then read another lane of is its own scratch, undefined on the
    // hardware too, and never a varying this linker failed to route.
    if let Some(reg) = (0..untouched.len()).find(|&r| untouched[r] && !fed[r]) {
        return Err(LinkError::PaReadUnfed { register: reg as u32, varyings_error: fprog.varyings_error });
    }
    // A passthrough program's colour IS its primary-attribute allocation, so a register in that
    // allocation that no interpolant feeds is a colour channel we would emit as zero. Fail
    // instead: for these programs the routing is the whole translation.
    if passthrough {
        if let Some(reg) = (0..primary_regs).find(|&r| !fed.get(r as usize).copied().unwrap_or(false)) {
            return Err(LinkError::PaReadUnfed { register: reg, varyings_error: fprog.varyings_error });
        }
    }
    Ok(iface)
}

/// Emit the statements of a program's SECONDARY code stream, which run before its primary body
/// and write the SA bank (see [`crate::usse::decode_secondary_shader`]). Empty string when the
/// program has no secondary code.
///
/// An instruction the emitter cannot translate is an error, exactly as in the primary stream: a
/// secondary program that half-runs leaves some SA registers computed and others stale, and the
/// primary reads both without distinction.
fn emit_secondary_body(program: &Program) -> Result<String, EmitError> {
    if program.secondary_code.is_empty() {
        return Ok(String::new());
    }
    emit_body(&crate::usse::decode_secondary_shader(program))
}

/// Validate that every SA register a stage reads is either inside its default uniform buffer,
/// written by the program's own secondary code, or a container literal, and return the literal
/// initialisers to emit. An SA read anywhere else lands in the texture-control-word region,
/// which is GPU state rather than shader data.
///
/// BOTH streams are scanned for reads, and that is not a detail. The secondary program is where
/// a container literal is most likely to be consumed - its whole job is to fold constants into
/// the SA registers the primary then reads - so scanning only the primary makes exactly the
/// literals that matter invisible, emits no initialiser for them, and the secondary reads zero.
/// MEASURED: a title's separable-blur vertex program declares `sa[3] = 3.0h`, `sa[4] = 5.0h` -
/// the tap distances of a 6-tap kernel at +-1, +-3, +-5 texels - and reads them ONLY in its
/// secondary program. With them zeroed, all six taps collapse onto +-1 and the blur silently
/// stops blurring, with nothing in the log to say so.
fn secondary_attr_init(
    shader: &Shader,
    program: &Program,
) -> Result<Vec<(u32, u32)>, LinkError> {
    let uniform_regs = program.default_uniform_regs;
    // Registers the secondary program computes. These are legitimate sources for the primary
    // even above the uniform buffer - that is the point of a secondary program - so they must
    // not be mistaken for reads of the texture-control region.
    let secondary = crate::usse::decode_secondary_shader(program);
    let mut written = std::collections::BTreeSet::new();
    for instr in &secondary.instrs {
        let Some(d) = instr.dest.as_ref() else { continue };
        if d.bank != Bank::SecondaryAttr {
            continue;
        }
        for c in 0..4u32 {
            if instr.write_mask[c as usize] {
                written.insert(d.index as u32 + if instr.half_precision { c >> 1 } else { c });
            }
        }
    }
    let mut needed = std::collections::BTreeSet::new();
    for instr in shader.instrs.iter().chain(secondary.instrs.iter()) {
        let half = instr.source_half_precision();
        let read = read_channels(instr);
        for (i, src) in instr.srcs.iter().enumerate() {
            if src.bank != Bank::SecondaryAttr {
                continue;
            }
            // A texture sample's SECOND source is the SAMPLER, not data: it names the four
            // texture-control words describing the texture, which live above the default
            // uniform buffer by construction. The unit is resolved from the container's own
            // texture-control table at decode, so those registers are never read as uniforms -
            // counting them makes a shader look like it reads past its buffer, and the read
            // channels here are the COORDINATE's count, which says nothing about the sampler.
            if matches!(instr.op, Op::Tex { .. }) && i == 1 {
                continue;
            }
            for c in 0..4 {
                if !read[c] {
                    continue;
                }
                let sel = src.swizzle[c];
                if sel > 3 {
                    continue;
                }
                needed.insert(src.index as u32 + if half { (sel >> 1) as u32 } else { sel as u32 });
            }
        }
    }
    let mut literals = Vec::new();
    for reg in needed {
        if reg < uniform_regs || written.contains(&reg) {
            continue;
        }
        match program.literals.iter().find(|(r, _)| *r == reg) {
            Some(&(r, v)) => literals.push((r, v)),
            None => return Err(LinkError::SecondaryAttrOutOfRange { register: reg, uniform_regs }),
        }
    }
    literals.sort_unstable();
    literals.dedup();
    Ok(literals)
}

/// The vector component letter for lane `c` (0..3 -> x/y/z/w).
fn comp(c: u32) -> char {
    ['x', 'y', 'z', 'w'][(c & 3) as usize]
}

/// Assemble the linked WGSL module: both entry points sharing the `@location` varying
/// interface, with the non-colliding binding namespace documented on the module. `vbody`/
/// `fbody` are the verbatim [`emit_body`] statements for each stage.
#[allow(clippy::too_many_arguments)]
/// The pipeline-supplied depth state, and the two helpers that read it. See the call site in
/// [`build_linked_module`] for what each lane holds.
///
/// `gxp_guest_depth` is the single definition of "what a GXM depth surface holds", and it is
/// deliberately shared: the renderer's depth-conversion pass writes that value into a sampleable
/// texture, and a fragment reading its own POSITION.z reads it here. If those two ever disagree
/// the comparison a soft particle makes is between two different quantities - which renders as a
/// fade that is stuck at 0 or 1 with nothing to point at.
pub(crate) const GXP_DEPTH_DECL: &str = r#"struct GxpDepth { range: vec4<f32>, fit: vec4<f32> };
@group(3) @binding(0) var<uniform> gxp_depth: GxpDepth;

// The value the GUEST's depth buffer holds for a fragment at clip `w`. A projection makes clip
// `z` affine in clip `w` (`z = a*w + c`), so the window depth `z/w` is `a + c/w` - and `a`, `c`
// are MEASURED per pass by interpreting its own vertex programs, not guessed. Both `a` and `c`
// matter and for different reasons: a soft-particle fade takes a DIFFERENCE of two depths, where
// `a` cancels and `c` sets the scale, while a near-plane fade reads one depth on its own, where
// `a` is the whole answer.
fn gxp_guest_depth(w: f32) -> f32 {
  return select(gxp_depth.fit.x + gxp_depth.fit.y / w, 0.0, w == 0.0);
}

fn gxp_window_position(fc: vec4<f32>) -> vec4<f32> {
  // `fc` is WebGPU's fragment builtin: pixels in xy, OUR remapped depth in z, and 1/w of the
  // position this pipeline actually rasterised - which is the guest's clip position after
  // `gxp_clipfix`. Recover the guest's own clip w by undoing that fixup's sign correction
  // (both correcting modes negate w; only the value of w matters here, not x/y/z).
  var w = 1.0 / fc.w;
  if (gxp_depth.range.z != 1.0) { w = -w; }
  return vec4<f32>(fc.x, fc.y, gxp_guest_depth(w), 1.0 / w);
}

// The inverse of the whole chain above, for a fragment that WRITES its own depth (0xF8
// DEPTHF). A shader can only compute a depth in the GUEST's encoding - it builds one out of
// its POSITION interpolant, which is that encoding by construction - while the depth buffer
// holds OURS, so an unconverted write would sort against every interpolated depth at random.
// `range.w` names which forward map the vertex stage applied, because there is more than one
// and only the renderer knows which is in force.
fn gxp_depth_to_window(d: f32, interpolated: f32) -> f32 {
  // The GL-style remap: the guest's own clip z, read in [-w, w], mapped into [0, w]. The
  // guest's window depth IS z/w there, so this is just the same affine map on it.
  if (gxp_depth.range.w > 0.5 && gxp_depth.range.w < 1.5) { return clamp((d + 1.0) * 0.5, 0.0, 1.0); }
  // No remap at all: the guest's clip z rasterised untouched, so its window depth is `d`.
  if (gxp_depth.range.w >= 1.5) { return d; }
  // The default: clip z was REPLACED by the projected view distance `-1/w`, mapped linearly
  // onto [0,1] over the scene's own range. Recover `w` from `d` by inverting
  // `gxp_guest_depth`, then apply that same map. `fit.y` is the `c` of `z = a*w + c`; with
  // c == 0 the guest's depth does not depend on `w` at all, so there is nothing to invert and
  // the interpolated depth stands.
  if (gxp_depth.fit.y == 0.0) { return interpolated; }
  let q = (gxp_depth.fit.x - d) / gxp_depth.fit.y;
  return clamp((q - gxp_depth.range.x) * gxp_depth.range.y, 0.0, 1.0);
}
"#;

fn build_linked_module(
    vbody: &str,
    vplan: &VertexBindingPlan,
    vprog: &Program,
    vliterals: &[(u32, u32)],
    fbody: &str,
    fplan: &BindingPlan,
    fprog: &Program,
    fliterals: &[(u32, u32)],
    iface: &Interface,
    varying_locations: u32,
    // Whether the fragment program replaces the interpolated depth (0xF8 DEPTHF).
    writes_depth: bool,
) -> String {
    let varyings = &iface.components;
    let mut m = String::new();

    // ---- Pipeline-supplied depth state at group 3 ----
    // Declared by the LINKER rather than injected by the renderer, because both stages depend
    // on it and a module that mentions it has to be independently compilable (the oracle
    // naga-validates linked modules with no renderer in the picture). The renderer fills it:
    //   x = depth_min, y = depth_scale  - the affine remap `gxp_clipfix` puts clip depth through
    //   z = the clip-`w` sign correction that same fixup applied (1 none, -1 negate, 2 flip w)
    //   w = which value the guest's own depth buffer holds (see `gxp_guest_depth`)
    m.push_str(GXP_DEPTH_DECL);

    // ---- Vertex default-uniform buffer (SA bank) at group 0 ----
    // The buffer is the guest's raw default-uniform-buffer bytes: a run of 32-bit registers,
    // NOT an array of floats, because a register may hold two packed F16 halves. It is bound
    // as `vec4<u32>` and copied verbatim into the register file.
    let vsa_regs = vprog.default_uniform_regs;
    let vsa_vec4 = vsa_regs.div_ceil(4);
    if vsa_vec4 > 0 {
        let _ = writeln!(m, "struct VsSa {{ data: array<vec4<u32>, {vsa_vec4}> }};");
        let _ = writeln!(m, "@group(0) @binding(0) var<uniform> vs_sa: VsSa;");
    }

    // ---- Fragment default-uniform buffer (SA bank) at group 1 ----
    let fsa_regs = fprog.default_uniform_regs;
    let fsa_vec4 = fsa_regs.div_ceil(4);
    if fsa_vec4 > 0 {
        let _ = writeln!(m, "struct FsSa {{ data: array<vec4<u32>, {fsa_vec4}> }};");
        let _ = writeln!(m, "@group(1) @binding(0) var<uniform> fs_sa: FsSa;");
    }

    // ---- Fragment sampled textures + samplers at group 2 ----
    for (i, b) in fplan.samplers.iter().enumerate() {
        let (tb, sb) = (i as u32 * 2, i as u32 * 2 + 1);
        let ty = b.wgsl_type();
        let _ = writeln!(m, "@group(2) @binding({tb}) var t{}: {ty};", b.unit);
        let _ = writeln!(m, "@group(2) @binding({sb}) var s{}: sampler;", b.unit);
    }

    // ---- Vertex sampled textures + samplers, AFTER the fragment ones in group 2 ----
    // They share the group because the device guarantees only four bind groups and the other
    // three are taken; they keep their own NAMES (`vt{u}`/`vs{u}`) because the two stages number
    // their sampler units independently, so a linked module can carry a vertex unit 0 and a
    // fragment unit 0 that are different textures. A vertex fetch builds GEOMETRY from what it
    // reads, so conflating them would not shade a surface wrongly, it would draw the wrong mesh.
    let vsampler_base = fplan.samplers.len() as u32 * 2;
    for (i, b) in vplan.samplers.iter().enumerate() {
        let (tb, sb) = (vsampler_base + i as u32 * 2, vsampler_base + i as u32 * 2 + 1);
        let ty = b.wgsl_type();
        let (tex, samp) = crate::wgsl::sampler_names(ProgramKind::Vertex, b.unit);
        let _ = writeln!(m, "@group(2) @binding({tb}) var {tex}: {ty};");
        let _ = writeln!(m, "@group(2) @binding({sb}) var {samp}: sampler;");
    }

    // ---- Vertex input attributes ----
    let has_inputs = !vplan.attributes.is_empty();
    if has_inputs {
        let _ = writeln!(m, "\nstruct VsIn {{");
        for a in &vplan.attributes {
            let _ = writeln!(m, "  @location({}) a{}: vec4<f32>,", a.location, a.location);
        }
        let _ = writeln!(m, "}};");
    }

    // ---- Shared varying interface: vertex output struct ----
    let _ = writeln!(m, "\nstruct VsOut {{");
    let _ = writeln!(m, "  @builtin(position) position: vec4<f32>,");
    for j in 0..varying_locations {
        let _ = writeln!(m, "  @location({j}) v{j}: vec4<f32>,");
    }
    let _ = writeln!(m, "}};");

    // ---- Vertex entry point ----
    if has_inputs {
        let _ = writeln!(m, "\n@vertex\nfn vs_main(in: VsIn) -> VsOut {{");
    } else {
        let _ = writeln!(m, "\n@vertex\nfn vs_main() -> VsOut {{");
    }
    emit_register_banks(&mut m);
    // Load PA registers from the vertex attributes (vertex inputs are plain f32 components).
    for a in &vplan.attributes {
        for c in 0..a.components {
            let _ = writeln!(
                m,
                "  pa[{}] = bitcast<u32>(in.a{}.{});",
                a.base_lane + c,
                a.location,
                comp(c)
            );
        }
    }
    emit_secondary_attrs(&mut m, "vs_sa", vsa_regs, vliterals);
    m.push_str(vbody);
    let _ = writeln!(m, "  var out: VsOut;");
    let _ = writeln!(
        m,
        "  out.position = vec4<f32>(bitcast<f32>(o[0]), bitcast<f32>(o[1]), bitcast<f32>(o[2]), bitcast<f32>(o[3]));"
    );
    // The vertex supplies one interpolated scalar per OUTPUT lane, in interface order - the
    // lanes its own varyings block places each usage at. A lane the program never writes stays
    // zero (the container reserves the slot; the hardware value would be undefined).
    for j in 0..varying_locations {
        let c = |k: usize| match varyings.get(j as usize * 4 + k) {
            Some(v) => format!("bitcast<f32>(o[{}])", v.vertex_lane),
            None => "0.0".to_string(),
        };
        let _ = writeln!(m, "  out.v{j} = vec4<f32>({}, {}, {}, {});", c(0), c(1), c(2), c(3));
    }
    let _ = writeln!(m, "  return out;\n}}");

    // ---- Fragment input struct (the same interface the vertex declares) ----
    // `front_facing` is declared unconditionally, even by a fragment stage with no varyings:
    // it is pipeline state rather than an interpolated value, so it costs no `@location`, and
    // making it always present keeps the entry signature (and every module builder here) the
    // same shape whether or not the body happens to read the facing GLOBAL register.
    let _ = writeln!(m, "\nstruct FsIn {{");
    for j in 0..varying_locations {
        let _ = writeln!(m, "  @location({j}) v{j}: vec4<f32>,");
    }
    // Both builtins are declared unconditionally, even by a fragment stage with no varyings:
    // they are rasteriser state rather than interpolated values, so they cost no `@location`,
    // and making them always present keeps the entry signature (and every module builder here)
    // the same shape whether or not the body happens to read them.
    let _ = writeln!(m, "  @builtin(position) frag_coord: vec4<f32>,");
    let _ = writeln!(m, "  @builtin(front_facing) front_facing: bool,");
    let _ = writeln!(m, "}};");
    // A fragment that writes its own depth (0xF8 DEPTHF) returns a STRUCT carrying
    // `@builtin(frag_depth)` next to the colour. The struct is emitted only for such a
    // program: declaring the builtin unconditionally would defeat early-depth rejection on
    // every other pair in the title, which is a real cost paid for nothing.
    if writes_depth {
        let _ = writeln!(
            m,
            "\nstruct FsOut {{\n  @location(0) color: vec4<f32>,\n  @builtin(frag_depth) depth: f32,\n}};"
        );
    }
    let ret_ty = if writes_depth { "FsOut" } else { "@location(0) vec4<f32>" };
    let _ = writeln!(m, "\n@fragment\nfn fs_main(in: FsIn) -> {ret_ty} {{");
    m.push_str(crate::wgsl::FRONT_FACING_DECL);
    emit_register_banks(&mut m);
    if writes_depth {
        let _ = writeln!(m, "  let gxp_interp_depth = in.frag_coord.z;");
        let _ = writeln!(m, "  var gxp_frag_depth: f32 = gxp_interp_depth;");
    }
    // The WINDOW coordinate a fragment's POSITION interpolant reads (see `plan_interface`).
    // `gxp_window_position` undoes what the pipeline did to the guest's clip position on the
    // way here - the clip-`w` sign correction and the depth remap - and re-encodes the depth
    // the way the guest's own depth buffer holds it, so that a shader comparing its own
    // POSITION against a sampled depth surface compares two values in ONE space.
    if let Some(base) = iface.window_position {
        let _ = writeln!(m, "  let gxp_wpos = gxp_window_position(in.frag_coord);");
        for c in 0..4u32 {
            let _ = writeln!(m, "  pa[{}] = bitcast<u32>(gxp_wpos.{});", base + c, comp(c));
        }
    }
    // Rebuild the PA register file from the interpolated components, repacking each F16 pair
    // exactly as the hardware interpolator delivers it (interpolate as floats, then pack). A
    // register carrying only one half of a pair (an odd-width varying) keeps 0 in the other.
    let at = |i: usize| format!("in.v{}.{}", i / 4, comp((i % 4) as u32));
    let mut done: Vec<u32> = Vec::new();
    for (i, v) in varyings.iter().enumerate() {
        let register = match v.dest {
            ComponentDest::Register(r) | ComponentDest::Half { register: r, .. } => r,
            ComponentDest::SampleCoord { .. } => continue, // consumed by the PDS, below
        };
        if done.contains(&register) {
            continue;
        }
        done.push(register);
        match v.dest {
            ComponentDest::Register(_) => {
                let _ = writeln!(m, "  pa[{register}] = bitcast<u32>({});", at(i));
            }
            _ => {
                let half_at = |slot: u32| {
                    varyings
                        .iter()
                        .position(|o| o.dest == ComponentDest::Half { register, slot })
                        .map(at)
                        .unwrap_or_else(|| "0.0".to_string())
                };
                let _ = writeln!(
                    m,
                    "  pa[{register}] = pack2x16float(vec2<f32>({}, {}));",
                    half_at(0),
                    half_at(1)
                );
            }
        }
    }
    // Registers the iterator fills with the texture-coordinate default because the vertex
    // program produces fewer components than the fragment allocated (see `Interface::defaults`).
    for &(register, halves, half) in &iface.defaults {
        if half {
            let _ = writeln!(
                m,
                "  pa[{register}] = pack2x16float(vec2<f32>({:?}, {:?}));",
                halves[0], halves[1]
            );
        } else {
            let _ = writeln!(m, "  pa[{register}] = bitcast<u32>({:?}f);", halves[0]);
        }
    }
    // Replay the samples the PDS took before the shader started. Each leaves four components in
    // two PA registers as packed F16 halves, which is how the code reads them - the instruction
    // stream contains no SMP for these, so without this the shader would read zeros.
    // The temporary is named by the prefetch's ORDINAL, not by its texture unit: one unit can be
    // prefetched more than once (the same texture sampled at two different interpolants), and
    // naming by unit emits two `let pf1` in one scope, which is a WGSL redefinition error that
    // fails the whole module - taking a pair that recompiled correctly straight to a hard stop.
    for (i, pf) in iface.prefetches.iter().enumerate() {
        let coord = pf
            .coords
            .iter()
            .map(|&i| at(i))
            .collect::<Vec<_>>()
            .join(", ");
        let n = pf.coords.len();
        let _ = writeln!(
            m,
            "  let pf{i} = textureSample(t{0}, s{0}, vec{n}<f32>({coord}));",
            pf.unit
        );
        if pf.regs > 1 {
            // Two registers: four F16 components, packed two per register.
            let _ = writeln!(m, "  pa[{}] = pack2x16float(pf{i}.xy);", pf.pa_base);
            let _ = writeln!(m, "  pa[{}] = pack2x16float(pf{i}.zw);", pf.pa_base + 1);
        } else {
            // One register: a single FULL-PRECISION component, not a packed pair.
            //
            // MEASURED on the corpus's own reads. A title's track material prefetches four
            // samples: its `DiffuseAlphaMap`, `lightmap` and `occlusionMap` descriptors each
            // span two registers and the code reads them with `unpack2x16float`, while its
            // one-register `shadowMap` descriptor is read with a full-precision `bitcast` -
            // the correlation is exact across the corpus. Packing halves into that register
            // instead makes the shadow compare read a denormal, every fragment tests as
            // shadowed, and the whole track surface shades black.
            let _ = writeln!(m, "  pa[{}] = bitcast<u32>(pf{i}.x);", pf.pa_base);
        }
    }
    emit_secondary_attrs(&mut m, "fs_sa", fsa_regs, fliterals);
    m.push_str(fbody);
    let ret = match fplan.color {
        ColorOutput::NativeO0 => "o",
        ColorOutput::NonNativePa0 => "pa",
    };
    let color = crate::module::color_return_expr(ret, fplan.color_precision, varying_locations);
    if writes_depth {
        let _ = writeln!(m, "  return FsOut({color}, gxp_frag_depth);\n}}");
    } else {
        let _ = writeln!(m, "  return {color};\n}}");
    }

    m
}

/// Emit a stage's SA-bank initialisation: the default uniform buffer copied verbatim into
/// registers `0..uniform_regs`, then the container literals stored at their own registers.
/// Both are raw 32-bit register values - a register may hold an F32 or two packed F16 halves,
/// and only the instruction reading it decides which.
fn emit_secondary_attrs(m: &mut String, binding: &str, uniform_regs: u32, literals: &[(u32, u32)]) {
    if uniform_regs > 0 {
        let _ = writeln!(
            m,
            "  for (var k: u32 = 0u; k < {uniform_regs}u; k = k + 1u) {{ sa[k] = {binding}.data[k / 4u][k % 4u]; }}"
        );
    }
    for &(reg, value) in literals {
        let _ = writeln!(m, "  sa[{reg}] = {value:#010x}u;");
    }
}

/// Emit the per-entry-point USSE register-file locals (raw 32-bit registers, matching the
/// emitter): the `r`/`o`/`i`/`pa`/`sa` banks plus the predicate registers.
fn emit_register_banks(m: &mut String) {
    for bank in ["r", "o", "i", "pa", "sa"] {
        let _ = writeln!(m, "  var {bank}: array<u32, {BANK_REGS}>;");
    }
    let _ = writeln!(m, "  var p: array<bool, 4>;");
    // The INDEX register file, for register-INDIRECT operands. Two registers, because the
    // extension row names exactly two indexed banks (INDEXED1 -> i0, INDEXED2 -> i1).
    let _ = writeln!(m, "  var idx: array<i32, 2>;");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{
        Interpolant, OutputVarying, ParamCategory, ParamType, Parameter, ProgramKind, SamplePrefetch,
    };
    use crate::ir::{Instr, Op, Operand, Predicate};

    fn instr(op: Op, dest: Option<Operand>, srcs: Vec<Operand>, mask: [bool; 4]) -> Instr {
        Instr { op, pred: Predicate::Always, dest, write_mask: mask, srcs, half_precision: false, raw: 0, group: 0, blocked: None }
    }

    fn shader(kind: ProgramKind, instrs: Vec<Instr>) -> Shader {
        Shader { kind, instrs }
    }

    #[test]
    fn pa_reads_are_read_before_write_only() {
        // PA[4] read as a 2D coord (registers 4,5) is a true input. PA[8] is WRITTEN by the
        // first instruction and then sampled - a computed / dependent coordinate, NOT an input.
        // PA[10] is read and never written, so it is an input.
        let sh = shader(
            ProgramKind::Fragment,
            vec![
                instr(Op::Mov, Some(Operand::plain(Bank::PrimaryAttr, 8, 1)), vec![Operand::plain(Bank::PrimaryAttr, 4, 1)], [true, true, false, false]),
                instr(Op::Tex { unit: 0, coords: 2, coord_half: false, lod: crate::ir::TexLod::Implicit }, Some(Operand::plain(Bank::Temp, 0, 0)), vec![Operand::plain(Bank::PrimaryAttr, 8, 1)], [true; 4]),
                instr(Op::Tex { unit: 1, coords: 2, coord_half: false, lod: crate::ir::TexLod::Implicit }, Some(Operand::plain(Bank::Temp, 4, 0)), vec![Operand::plain(Bank::PrimaryAttr, 10, 1)], [true; 4]),
            ],
        );
        let (inputs, _) = pa_read_before_write(&sh);
        let regs: Vec<usize> = (0..BANK_REGS).filter(|&r| inputs[r]).collect();
        assert_eq!(regs, vec![4, 5, 10, 11]);
    }

    #[test]
    fn f16_reads_resolve_to_a_register_pair() {
        // An F16 operand's four channels share a REGISTER PAIR (index + selector/2), so a
        // 4-channel read at index 6 touches registers 6 and 7 - not 6..9. Getting this wrong is
        // the factor-of-two that mis-routes every F16 varying.
        let mut i = instr(
            Op::Mov,
            Some(Operand::plain(Bank::Temp, 0, 0)),
            vec![Operand::plain(Bank::PrimaryAttr, 6, 1)],
            [true; 4],
        );
        i.half_precision = true;
        let (inputs, _) = pa_read_before_write(&shader(ProgramKind::Fragment, vec![i]));
        let regs: Vec<usize> = (0..BANK_REGS).filter(|&r| inputs[r]).collect();
        assert_eq!(regs, vec![6, 7]);
    }

    /// A fragment `Program` carrying the interpolant interface + PA allocation the linker reads.
    fn fragment_program(interpolants: Vec<Interpolant>, primary_reg_count: u16) -> Program {
        Program { kind: ProgramKind::Fragment, primary_reg_count, interpolants, ..vertex_program(0, Vec::new(), 0) }
    }

    /// A minimal vertex `Program` carrying only the fields the linker reads.
    fn vertex_program(secondary_reg_count: u16, attrs: Vec<Parameter>, hash: u64) -> Program {
        Program {
            // These fixtures set `output_varyings` explicitly, so their order IS the order
            // under test - it must not be re-derived from a fragment.
            output_order: crate::container::VaryingOrder::Known,
            varyings_error: None,
            default_uniform_regs: 0,
            secondary_code: Vec::new(),
            literals: Vec::new(),
            texture_control: Vec::new(),
            kind: ProgramKind::Vertex,
            major: 1,
            minor: 4,
            size: 0,
            primary_reg_count: 0,
            secondary_reg_count,
            temp_reg_count: 0,
            parameters: attrs,
            code: Vec::new(),
            interpolants: Vec::new(),
            output_varyings: Vec::new(),
            hash,
        }
    }

    fn texcoord_out(index: u8, base_lane: u32, components: u32) -> OutputVarying {
        OutputVarying { usage: VaryingUsage::TexCoord(index), base_lane, components }
    }

    fn texcoord_in(index: u8, pa_base: u8, register_count: u8, half: bool) -> Interpolant {
        Interpolant {
            usage: VaryingUsage::TexCoord(index),
            pa_base,
            register_count,
            span: register_count,
            half,
            prefetch: None,
            prefetch_regs: 2,
        }
    }

    /// The same, plus a PDS-prefetched sample from `unit` coordinated by TEXCOORD `source` - so
    /// its span carries two more registers for the sample's four packed F16 components.
    fn texcoord_in_prefetched(
        index: u8,
        pa_base: u8,
        register_count: u8,
        unit: u8,
        source: u8,
    ) -> Interpolant {
        Interpolant {
            span: register_count + PREFETCH_REGS as u8,
            prefetch: Some(SamplePrefetch { unit, source_texcoord: source, last: true }),
            prefetch_regs: 2,
            ..texcoord_in(index, pa_base, register_count, true)
        }
    }

    fn sampler(unit: i32, cube: bool) -> Parameter {
        Parameter {
            name: String::new(),
            category: ParamCategory::Sampler,
            ptype: ParamType::F32,
            component_count: 4,
            container_index: 0,
            sampler_cube: cube,
            array_size: 1,
            resource_index: unit,
            semantic: 0,
            semantic_index: 0,
        }
    }

    /// The PA register each component of `plan` lands in, for assertions that only care about
    /// placement (a sample coordinate lands in no register and is reported as `None`).
    fn destinations(plan: &[VaryingComponent]) -> Vec<(u32, Option<u32>)> {
        plan.iter()
            .map(|c| match c.dest {
                ComponentDest::Register(r) => (r, None),
                ComponentDest::Half { register, slot } => (register, Some(slot)),
                ComponentDest::SampleCoord { .. } => (u32::MAX, None),
            })
            .collect()
    }

    /// A fragment that reads PA registers `regs` (F32, one channel each) - enough to mark them
    /// as live inputs for the interface planner.
    fn fragment_reading(regs: &[u8]) -> Shader {
        shader(
            ProgramKind::Fragment,
            regs.iter()
                .map(|&r| {
                    instr(
                        Op::Mov,
                        Some(Operand::plain(Bank::Temp, 0, 0)),
                        vec![Operand::plain(Bank::PrimaryAttr, r, 1)],
                        [true, false, false, false],
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn f16_varying_costs_two_vertex_lanes_per_register() {
        // TEXCOORD1 is 4 components on the vertex (lanes 10..13) and arrives F16-packed in TWO
        // fragment PA registers. The interface must carry all four components - one vertex lane
        // each - and pair them into the two registers, not map register-to-register.
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(0, 6, 4), texcoord_out(1, 10, 4)];
        let fprog = fragment_program(vec![texcoord_in(1, 0, 2, true)], 4);
        let iface = plan_interface(&vprog, &fprog, &fragment_reading(&[0])).unwrap();
        assert_eq!(
            iface.components,
            vec![
                VaryingComponent { vertex_lane: 10, dest: ComponentDest::Half { register: 0, slot: 0 } },
                VaryingComponent { vertex_lane: 11, dest: ComponentDest::Half { register: 0, slot: 1 } },
                VaryingComponent { vertex_lane: 12, dest: ComponentDest::Half { register: 1, slot: 0 } },
                VaryingComponent { vertex_lane: 13, dest: ComponentDest::Half { register: 1, slot: 1 } },
            ]
        );
        assert!(iface.prefetches.is_empty());
    }

    #[test]
    fn f32_varying_costs_one_register_per_component() {
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(3, 6, 4)];
        let fprog = fragment_program(vec![texcoord_in(3, 0, 4, false)], 4);
        let plan = plan_interface(&vprog, &fprog, &fragment_reading(&[0])).unwrap().components;
        assert_eq!(plan.len(), 4);
        assert!(plan.iter().enumerate().all(|(c, v)| v.vertex_lane == 6 + c as u32
            && v.dest == ComponentDest::Register(c as u32)));
    }

    #[test]
    fn odd_width_f16_varying_half_fills_its_last_register() {
        // A 3-component F16 texcoord occupies ceil(3/2) = 2 registers; the second carries only
        // its low half.
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(2, 6, 3)];
        let fprog = fragment_program(vec![texcoord_in(2, 0, 2, true)], 2);
        let plan = plan_interface(&vprog, &fprog, &fragment_reading(&[0])).unwrap().components;
        assert_eq!(plan.len(), 3);
        assert_eq!(
            plan[2],
            VaryingComponent { vertex_lane: 8, dest: ComponentDest::Half { register: 1, slot: 0 } }
        );
    }

    /// A fragment that declares LESS room than the vertex writes reads a PREFIX, and the surplus
    /// is not routed anywhere.
    ///
    /// This used to be a hard failure on the reasoning that the surplus would land on the next
    /// interpolant. It would have - in this planner - which is why the routing is clamped. The
    /// test that matters is the second assertion: nothing is planned past the declared span.
    #[test]
    fn a_fragment_narrower_than_its_vertex_reads_a_prefix() {
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(1, 6, 4)];
        let fprog = fragment_program(vec![texcoord_in(1, 0, 2, false)], 2);
        let plan = plan_interface(&vprog, &fprog, &fragment_reading(&[0])).unwrap().components;
        assert_eq!(plan.len(), 2, "only the two components the fragment declared are routed");
        assert!(
            plan.iter().all(|c| matches!(c.dest, ComponentDest::Register(r) if r < 2)),
            "nothing may be routed past the declared span: {plan:?}"
        );
        assert_eq!(plan[0].vertex_lane, 6, "and it is the FIRST components that are taken");
        assert_eq!(plan[1].vertex_lane, 7);
    }

    /// The half-precision shape a real draw hits: the vertex writes four texcoord components and
    /// the fragment declares ONE half register, which holds two.
    #[test]
    fn a_half_precision_fragment_narrower_than_its_vertex_reads_two_components() {
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(0, 6, 4)];
        let fprog = fragment_program(vec![texcoord_in(0, 0, 1, true)], 1);
        let plan = plan_interface(&vprog, &fprog, &fragment_reading(&[0])).unwrap().components;
        assert_eq!(plan.len(), 2);
        assert!(
            plan.iter().all(|c| matches!(c.dest, ComponentDest::Half { register: 0, .. })),
            "both components share the one declared register: {plan:?}"
        );
    }

    #[test]
    fn a_declared_but_unread_interpolant_is_not_routed() {
        // The fragment declares FOG (whose vertex placement is not established) but never reads
        // it. It cannot affect the picture, so it must not block the link - while the texcoord
        // it does read still routes.
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(1, 6, 4)];
        let fprog = fragment_program(
            vec![
                texcoord_in(1, 0, 2, true),
                Interpolant {
                    usage: VaryingUsage::Fog,
                    pa_base: 2,
                    register_count: 1,
                    span: 1,
                    half: false,
                    prefetch: None,
                    prefetch_regs: 2,
                },
            ],
            4,
        );
        let plan = plan_interface(&vprog, &fprog, &fragment_reading(&[0])).unwrap().components;
        assert_eq!(destinations(&plan), vec![(0, Some(0)), (0, Some(1)), (1, Some(0)), (1, Some(1))]);
    }

    #[test]
    fn a_read_interpolant_the_vertex_does_not_produce_is_a_hard_failure() {
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(1, 6, 4)];
        let fprog = fragment_program(vec![texcoord_in(4, 0, 2, true)], 4);
        assert_eq!(
            plan_interface(&vprog, &fprog, &fragment_reading(&[0])).unwrap_err(),
            LinkError::UnfedVarying { usage: VaryingUsage::TexCoord(4) }
        );
    }

    #[test]
    fn a_pa_read_nothing_feeds_is_a_hard_failure() {
        // The fragment reads PA[4] before writing it. That is inside the PA registers the
        // container allocates, but neither an interpolant's data nor a prefetched sample covers
        // it, so nothing in the pipeline supplies its value. It cannot be dismissed as scratch:
        // scratch is written before it is read. Emitting would read a zero-initialised register
        // and paint a silently wrong colour.
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(1, 6, 4)];
        let fprog = fragment_program(vec![texcoord_in(1, 0, 2, true)], 8);
        assert_eq!(
            plan_interface(&vprog, &fprog, &fragment_reading(&[0, 4])).unwrap_err(),
            LinkError::PaReadUnfed { register: 4, varyings_error: None }
        );
    }

    #[test]
    fn a_pa_register_written_before_it_is_read_is_scratch_not_a_varying() {
        // The complement of the case above: the shader writes PA[4] and only then reads it, so
        // it carries an intermediate value (a dependent texture coordinate, say) and needs no
        // vertex output. It must neither be routed nor block the link.
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(1, 6, 4)];
        let fprog = fragment_program(vec![texcoord_in(1, 0, 2, true)], 8);
        let mut fshader = fragment_reading(&[0]);
        fshader.instrs.insert(
            0,
            instr(Op::Mov, Some(Operand::plain(Bank::PrimaryAttr, 4, 0)), vec![], [true; 4]),
        );
        fshader.instrs.push(instr(
            Op::Mov,
            Some(Operand::plain(Bank::Temp, 0, 0)),
            vec![Operand::plain(Bank::PrimaryAttr, 4, 2)],
            [true; 4],
        ));
        let plan = plan_interface(&vprog, &fprog, &fshader).unwrap().components;
        assert_eq!(destinations(&plan), vec![(0, Some(0)), (0, Some(1)), (1, Some(0)), (1, Some(1))]);
    }

    #[test]
    fn a_pa_read_beyond_the_container_allocation_is_a_hard_failure() {
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(1, 6, 4)];
        let fprog = fragment_program(vec![texcoord_in(1, 0, 2, true)], 4);
        assert_eq!(
            plan_interface(&vprog, &fprog, &fragment_reading(&[0, 9])).unwrap_err(),
            LinkError::PaReadBeyondAllocation { register: 9, primary_regs: 4 }
        );
    }

    #[test]
    fn a_prefetched_sample_feeds_the_two_registers_after_the_data() {
        // TEXCOORD1's descriptor declares a sample from unit 13 coordinated by TEXCOORD0. Its
        // own data lands in PA[0..2] and the sample's four F16 components in PA[2..4] - which is
        // why the shader can read PA[2] without any SMP instruction ever naming unit 13.
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(0, 6, 4), texcoord_out(1, 10, 4)];
        let mut fprog = fragment_program(vec![texcoord_in_prefetched(1, 0, 2, 13, 0)], 4);
        fprog.parameters = vec![sampler(13, false)];

        let iface = plan_interface(&vprog, &fprog, &fragment_reading(&[0, 2])).unwrap();
        assert_eq!(
            iface.prefetches,
            vec![PlannedPrefetch { unit: 13, pa_base: 2, regs: 2, coords: vec![4, 5], cube: false }]
        );
        // The interface carries TEXCOORD1's four components, then the two sample coordinates
        // taken from TEXCOORD0 - which is NOT itself an interpolant of this fragment.
        assert_eq!(
            iface.components.iter().map(|c| c.vertex_lane).collect::<Vec<_>>(),
            vec![10, 11, 12, 13, 6, 7]
        );
        assert_eq!(
            iface.components[4].dest,
            ComponentDest::SampleCoord { prefetch: 0, coord: 0 }
        );
    }

    #[test]
    fn a_prefetched_sample_the_shader_never_reads_is_not_taken() {
        // The PDS fetched it, but this shader looks only at the interpolated data. Issuing the
        // sample anyway would bind a texture the pipeline does not need - and would fail the
        // link outright when that texture is missing.
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(0, 6, 4), texcoord_out(1, 10, 4)];
        let fprog = fragment_program(vec![texcoord_in_prefetched(1, 0, 2, 13, 0)], 4);
        let iface = plan_interface(&vprog, &fprog, &fragment_reading(&[0])).unwrap();
        assert!(iface.prefetches.is_empty());
        assert_eq!(iface.components.len(), 4);
    }

    #[test]
    fn a_prefetch_from_an_undeclared_sampler_is_a_hard_failure() {
        // Without a declared sampler the texture's dimensionality is unknown, so neither the
        // WGSL texture type nor the coordinate count can be derived. Fall back, never guess.
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(0, 6, 4), texcoord_out(1, 10, 4)];
        let fprog = fragment_program(vec![texcoord_in_prefetched(1, 0, 2, 13, 0)], 4);
        assert_eq!(
            plan_interface(&vprog, &fprog, &fragment_reading(&[0, 2])).unwrap_err(),
            LinkError::PrefetchUnitNotDeclared { unit: 13 }
        );
    }

    #[test]
    fn a_cube_prefetch_takes_three_coordinates() {
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(0, 6, 4), texcoord_out(1, 10, 4)];
        let mut fprog = fragment_program(vec![texcoord_in_prefetched(1, 0, 2, 15, 0)], 4);
        fprog.parameters = vec![sampler(15, true)];
        let iface = plan_interface(&vprog, &fprog, &fragment_reading(&[0, 2])).unwrap();
        assert_eq!(iface.prefetches[0].coords.len(), 3);
        assert!(iface.prefetches[0].cube);
        assert_eq!(iface.prefetches[0].binding().wgsl_type(), "texture_cube<f32>");
    }

    #[test]
    fn a_prefetch_whose_texcoord_is_too_narrow_is_a_hard_failure() {
        // A cube sample needs a three-component direction; this texcoord carries two.
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(0, 6, 2), texcoord_out(1, 10, 4)];
        let mut fprog = fragment_program(vec![texcoord_in_prefetched(1, 0, 2, 15, 0)], 4);
        fprog.parameters = vec![sampler(15, true)];
        assert_eq!(
            plan_interface(&vprog, &fprog, &fragment_reading(&[0, 2])).unwrap_err(),
            LinkError::PrefetchCoordTooNarrow { unit: 15, needed: 3, available: 2 }
        );
    }

    fn attribute(name: &str, resource_index: i32, component_count: u8) -> Parameter {
        Parameter {
            name: name.to_string(),
            category: ParamCategory::Attribute,
            ptype: ParamType::F32,
            component_count,
            container_index: 0,
            sampler_cube: false,
            array_size: 1,
            resource_index,
            semantic: 0,
            semantic_index: 0,
        }
    }

    #[test]
    fn build_linked_module_wires_shared_varyings_and_bindings() {
        // Vertex: position attr at pa0, uv attr at pa8. Writes clip position (o0..3) and one
        // varying group (o6..9) from the uv. Fragment: reads that varying (pa0..3), samples a
        // texture from pa lane 0, writes o0 colour.
        let vsh = shader(
            ProgramKind::Vertex,
            vec![
                instr(Op::Mad, Some(Operand::plain(Bank::Output, 0, 2)),
                    vec![Operand::plain(Bank::PrimaryAttr, 0, 1), Operand::plain(Bank::SecondaryAttr, 0, 3), Operand::plain(Bank::Constant, 2, 0)], [true; 4]),
                instr(Op::Mov, Some(Operand::plain(Bank::Output, 6, 2)),
                    vec![Operand::plain(Bank::PrimaryAttr, 8, 1)], [true; 4]),
            ],
        );
        let vprog = vertex_program(4, vec![attribute("position", 0, 4), attribute("uv", 8, 2)], 0xaa);
        let vplan = plan_vertex_bindings(&vprog, &vsh);
        let vbody = emit_body(&vsh).unwrap();

        let fsh = shader(
            ProgramKind::Fragment,
            vec![
                instr(Op::Tex { unit: 0, coords: 2, coord_half: false, lod: crate::ir::TexLod::Implicit }, Some(Operand::plain(Bank::Output, 0, 2)),
                    vec![Operand::plain(Bank::PrimaryAttr, 0, 1)], [true; 4]),
            ],
        );
        let fplan = plan_bindings(&fsh, 0, |_| false);
        let fbody = emit_body(&fsh).unwrap();

        let mut uprog = vertex_program(4, Vec::new(), 0);
        uprog.default_uniform_regs = 4;
        uprog.output_varyings = vec![texcoord_out(0, 6, 4)];
        let fprog = fragment_program(vec![texcoord_in(0, 0, 4, false)], 4);
        let iface = plan_interface(&uprog, &fprog, &fsh).unwrap();
        let locations = (iface.components.len() as u32).div_ceil(4);
        let wgsl = build_linked_module(
            &vbody, &vplan, &uprog, &[], &fbody, &fplan, &uprog, &[], &iface, locations, false,
        );

        // Vertex SA is group 0, samplers are group 2, and both stages share @location(0). Every
        // stage-crossing value moves as a raw register through the interpolated components.
        assert!(wgsl.contains("@group(0) @binding(0) var<uniform> vs_sa:"), "{wgsl}");
        assert!(wgsl.contains("@group(2) @binding(0) var t0: texture_2d<f32>;"), "{wgsl}");
        assert!(wgsl.contains("out.v0 = vec4<f32>(bitcast<f32>(o[6]), bitcast<f32>(o[7])"), "{wgsl}");
        assert!(wgsl.contains("pa[0] = bitcast<u32>(in.v0.x);"), "{wgsl}");
        assert!(wgsl.contains("fn vs_main(in: VsIn) -> VsOut"), "{wgsl}");
        assert!(wgsl.contains("fn fs_main(in: FsIn) -> @location(0) vec4<f32>"), "{wgsl}");
    }

    #[test]
    fn output_written_lanes_tracks_the_write_mask() {
        let sh = shader(
            ProgramKind::Vertex,
            vec![instr(
                Op::Mov,
                Some(Operand::plain(Bank::Output, 6, 2)),
                vec![Operand::plain(Bank::PrimaryAttr, 0, 1)],
                [true, true, false, false],
            )],
        );
        let w = output_written_lanes(&sh);
        assert!(w[6] && w[7] && !w[8]);
    }
}


