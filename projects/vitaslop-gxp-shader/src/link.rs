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
use crate::ir::{Bank, Instr, Op, Predicate, Shader};
use crate::module::{plan_bindings, plan_vertex_bindings, BindingPlan, ColorOutput, VertexBindingPlan};
use crate::ColorPrecision;
use crate::wgsl::{
    add_half_helpers, emit_body, emit_body_marked, split_marker, strip_split_markers, EmitError,
    TexBinding, BANK_REGS, HALF_HI_FN, HALF_LO_FN, HALF_PK_FN, HALF_QUANT_FN,
};
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
    /// Whether the fragment stage reads the DESTINATION colour (it blends for itself). The
    /// renderer owes such a draw a copy of the colour attachment as it stands immediately
    /// before it, bound at `@group(3) @binding(1)`. See [`crate::module::BindingPlan::
    /// reads_dest_color`].
    pub reads_dest_color: bool,
    /// Whether the fragment stage's destination read was lowered to a DUAL-SOURCE blend. The
    /// module's `fs_main` then returns two `@blend_src` colours and the renderer MUST build the
    /// pipeline with `src0 * 1 + dst * src1`. Exclusive with `reads_dest_color` - a program
    /// lowered this way needs no attachment copy. See `module::dual_source_eligible`.
    pub dual_source: bool,
    /// The blend the fragment program performed itself and that
    /// [`crate::module::lower_dest_blend`] recovered as pipeline state. When this is `Some` the
    /// emitted shader writes only the SOURCE term and the renderer MUST apply this equation, or
    /// the draw composites wrongly. It and `reads_dest_color` are mutually exclusive by
    /// construction: the lowering runs first, and what it rewrites no longer reads the output.
    pub dest_blend: Option<crate::module::DestBlend>,
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
    ///
    /// `provenance` names every place the register COULD have come from and says which of them
    /// were checked, because "sa[49] is out of range" alone does not distinguish the three
    /// defects that produce it: a uniform block whose extent was read short, a secondary write
    /// the dest model did not attribute to it, and a literal only an indexed read reaches. Each
    /// wants a different fix and the number alone points at none of them.
    SecondaryAttrOutOfRange { register: u32, uniform_regs: u32, provenance: String },
    /// A varying descriptor declares a prefetched sample from a texture unit the program's own
    /// parameter table does not declare as a sampler, so its dimensionality is unknown and
    /// binding it would guess at GPU state the shader never asked for. Fall back.
    PrefetchUnitNotDeclared { unit: u8 },
    /// A prefetched sample needs `needed` coordinate components but the vertex produces only
    /// `available` for the texcoord that feeds it, so the missing coordinates would sample at an
    /// arbitrary position. Fall back.
    PrefetchCoordTooNarrow { unit: u8, needed: u32, available: u32 },
    /// The VERTEX program's 0xE8 memory loads cannot be tied to a bindable guest-memory
    /// window ([`crate::module::resolve_mem_window`] names the specific gap). Emitting anyway
    /// would hand the loads fabricated bytes. Fall back.
    MemWindowUnresolved { why: &'static str },
    /// The FRAGMENT program's 0xE8 memory loads cannot be tied to a bindable guest-memory
    /// window. Separate from the vertex variant because the two stages bind from DIFFERENT
    /// GXM uniform-buffer tables, so a message naming the wrong one sends a reader to the
    /// wrong `sceGxmSet*UniformBuffer` call site.
    FragmentMemWindowUnresolved { why: &'static str },
    /// A unit the caller named RAW ([`LinkOptions::raw_units`]) is sampled by the program's
    /// own SMP instruction. The body was emitted before the link knew the unit is raw, so it
    /// would `textureSample` a `texture_2d<u32>`; refuse rather than mis-type the binding.
    RawUnitSampledInProgram { unit: u8 },
    /// A RAW unit's prefetch is not a two-coordinate 2D sample. A raw texel is a `textureLoad`
    /// at integer coordinates, which is established only for the 2D case.
    RawUnitNotTwoDimensional { unit: u8 },
    /// [`LinkOptions::raw64_output`] was asked of a program that cannot honour it.
    Raw64OutputUnsupported { why: &'static str },
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
            LinkError::SecondaryAttrOutOfRange { register, uniform_regs, provenance } => write!(
                f,
                "shader reads SA register {register} outside its {uniform_regs}-register default \
                 uniform buffer and outside the container literals (texture-control region); \n                 {provenance}"
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
            LinkError::MemWindowUnresolved { why } => write!(
                f,
                "the vertex program's memory loads have no bindable guest-memory window: {why}"
            ),
            LinkError::FragmentMemWindowUnresolved { why } => write!(
                f,
                "the fragment program's memory loads have no bindable guest-memory window: {why}"
            ),
            LinkError::RawUnitSampledInProgram { unit } => write!(
                f,
                "texture unit {unit} holds a 64-bit RAW texture but the fragment program samples \
                 it with its own SMP instruction, and a raw in-program sample is not emitted yet"
            ),
            LinkError::RawUnitNotTwoDimensional { unit } => write!(
                f,
                "texture unit {unit} holds a 64-bit RAW texture but its prefetch is not a \
                 two-coordinate 2D sample, and a raw load is established only for 2D"
            ),
            LinkError::Raw64OutputUnsupported { why } => write!(
                f,
                "this pass renders into a 64-bit colour surface (raw Rg32Uint attachment) and \
                 the fragment program cannot: {why}"
            ),
        }
    }
}

/// The ORDINARY width of a prefetched sample's result, in PA registers: two, holding its four
/// components as packed F16 halves. The per-descriptor width is
/// [`crate::container::Interpolant::prefetch_regs`], which is 1, 2 or 4 - this is only the
/// common case the test helpers build. See [`crate::container::SamplePrefetch`].
// Kept for the record: it names the COMMON prefetch width, which is the thing a reader
// needs when `Interpolant::prefetch_regs` returns 1 or 4 instead.
#[allow(dead_code)]
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
/// How a pair is to be linked beyond what its two programs say.
// NOT `Copy`: `guest_attrs` is a list whose length is the guest's, and a fixed-size array with
// a sentinel would be a worse answer than one clone per pipeline BUILD - a path that already
// decodes two USSE programs and emits ~27 KB of WGSL text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LinkOptions {
    /// Emit the fragment stage as a DUAL-SOURCE blend (see `module::dual_source_plan`). The
    /// caller has established that the draws this module will serve satisfy the plan's gates;
    /// a program the plan refuses links the ordinary way whatever this says.
    pub dual_source: bool,
    /// GXM fragment texture units (bit `u` = unit `u`) whose bound texture is a 64-bit RAW
    /// format the hardware hands to the shader UNCONVERTED: the texel's two 32-bit words land
    /// in two consecutive registers exactly as stored. A prefetch of such a unit is emitted as
    /// a `textureLoad` off a `texture_2d<u32>` binding and stored as raw words.
    ///
    /// MEASURED on a baseball title's crowd: an impostor atlas is RENDERED into an
    /// F16F16F16F16 surface by a program that packs RGBA8 bytes into its colour registers, then
    /// SAMPLED as a U32U32 texture by the crowd program, which reads the two words back as
    /// packed bytes (`unpack4x8unorm`) - an alpha test on byte 3 of word 0, a colour from word
    /// 1. Sampling that as a float texture handed the byte reads the bit pattern of a float
    /// instead, every fragment failed the alpha test, and the stands were empty.
    pub raw_units: u64,
    /// The pass renders into a 64-bit colour surface bound as an `Rg32Uint` attachment: the
    /// entry point returns the colour register PAIR as raw words. Refused (the link fails,
    /// naming why) for a program that blends for itself or is lowered to a dual-source blend,
    /// because an integer attachment cannot blend and a raw pair has no destination colour.
    pub raw64_output: bool,
    /// What the GUEST bound to each vertex attribute, as `(base lane, GXM attribute format,
    /// component count)` - the `SceGxmVertexAttribute` fields, keyed by the PA base lane the
    /// program's own parameter table names (`resource_index`), so no ordering has to be agreed
    /// between the two sides.
    ///
    /// Empty means "not told", which is what every caller outside the live renderer passes, and
    /// then nothing below changes: every attribute keeps the `vec4<f32>` declaration and the
    /// renderer converts on the CPU exactly as it always did. The renderer supplies it so the
    /// link can decide [`crate::module::VertexAttribute::int_fetch`] - and it MUST be folded
    /// into the module cache key by any caller that varies it, because two guest layouts over
    /// one program pair are two different modules.
    pub guest_attrs: Vec<(u32, u8, u8)>,
    /// >>> THE CALLER ASSERTS THAT THIS DRAW'S COLOUR OUTPUT CANNOT REACH A PIXEL: its guest
    /// >>> colour MASK is zero, or the pass it is issued into has no colour attachment.
    ///
    /// What it unlocks is the PASSTHROUGH treatment below (`is_passthrough`), which exists for
    /// a program that writes no colour register because its result is *whatever the PDS left in
    /// the primary-attribute registers*. That treatment marks the whole primary allocation read
    /// and then demands every register of it be fed, which is right when the register file IS
    /// the colour - and wrong for a DISCARD-ONLY program, whose entire effect is a predicated
    /// `Kill` plus the depth write and which genuinely reads nothing.
    ///
    /// MEASURED: a football title's `frag_90c054e0` is `Nop`, a `Test` on two constants, a
    /// predicated `Kill`, `Nop`, `Nop` - no varyings block (its count of 0 is TRUE, it
    /// interpolates nothing) and no write to any colour register at all. It was refused as
    /// "reads PA register 0 before writing it", and nothing in the SHADER can tell it from a
    /// passthrough: both write no colour and read no PA. Only the DRAW says which, and only
    /// because a colour that is masked off is exact whatever it holds.
    ///
    /// This is the same shape as the renderer's colour-no-op elision - "this draw's colour
    /// cannot matter", decided per draw from state the shader does not carry - and like every
    /// other per-draw option here it MUST be folded into the module cache key by any caller
    /// that varies it.
    ///
    /// Default `false`, which is every caller outside the live renderer: the passthrough
    /// treatment stands exactly as it did.
    pub colour_output_masked_off: bool,
}

pub fn link_programs(vbytes: &[u8], fbytes: &[u8]) -> Result<LinkedProgram, LinkError> {
    link_programs_with(vbytes, fbytes, LinkOptions::default())
}

pub fn link_programs_with(vbytes: &[u8], fbytes: &[u8], opts: LinkOptions) -> Result<LinkedProgram, LinkError> {
    let vprog = Program::parse(vbytes).map_err(LinkError::VertexParse)?;
    let fprog = Program::parse(fbytes).map_err(LinkError::FragmentParse)?;
    if vprog.kind != ProgramKind::Vertex || fprog.kind != ProgramKind::Fragment {
        return Err(LinkError::WrongKind);
    }

    let vrc = recompile_vertex(vbytes).map_err(LinkError::VertexRecompile)?;
    let frc = recompile_fragment(fbytes).map_err(LinkError::FragmentRecompile)?;

    // Memory loads, in EITHER stage: each must resolve to a bindable window or the pair falls
    // back NAMING the gap (the plan's own resolver call swallows the reason, because a plan has
    // no error channel).
    crate::module::resolve_mem_windows(&vprog, &vrc.shader)
        .map_err(|why| LinkError::MemWindowUnresolved { why })?;
    let fmem_windows = crate::module::resolve_mem_windows(&fprog, &frc.shader)
        .map_err(|why| LinkError::FragmentMemWindowUnresolved { why })?;

    let mut vplan = plan_vertex_bindings(&vprog, &vrc.shader);
    let mut fplan =
        plan_bindings(&frc.shader, fprog.sa_carried_extent(), |u| fprog.sampler_is_cube(u as u32));
    fplan.mem_windows = fmem_windows;

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
    let iface = plan_interface(&vprog, &fprog, &frc.shader, opts.colour_output_masked_off)?;
    let varyings = &iface.components;

    // What each attribute lane is fed when the guest binds fewer components than the program
    // declares. It is answered HERE and not in the plan because the deciding use is often in the
    // FRAGMENT stage - a surplus lane forwarded into a varying that a modulate reads - so the
    // question needs both shaders and the interface between them. See [`crate::attrflow`].
    let lands: Vec<(u32, crate::attrflow::FragLand)> = varyings
        .iter()
        .map(|c| {
            (
                c.vertex_lane,
                match c.dest {
                    ComponentDest::Register(r) => crate::attrflow::FragLand::Register(r),
                    ComponentDest::Half { register, slot } => {
                        crate::attrflow::FragLand::Half { register, slot }
                    }
                    ComponentDest::SampleCoord { .. } => crate::attrflow::FragLand::SampleCoord,
                },
            )
        })
        .collect();
    for a in &mut vplan.attributes {
        for c in 0..4u32 {
            a.surplus_fill[c as usize] =
                crate::attrflow::lane_fill(&vrc.shader, &frc.shader, a.base_lane, c, &lands);
        }
    }

    // >>> AND THEN, FOR EACH ATTRIBUTE THE CALLER TOLD US ABOUT, WHETHER IT CAN BE FETCHED AS
    // >>> AN INTEGER. Both halves of that question are answered above: the guest's format comes
    // from the caller, and the fills come from the analysis that has just run. See
    // `VertexAttribute::int_fetch` for the measurement and for why the lane test is "unobserved"
    // rather than "wants zero".
    for a in &mut vplan.attributes {
        let Some(&(_, fmt, gcomps)) = opts.guest_attrs.iter().find(|(lane, ..)| *lane == a.base_lane)
        else {
            continue;
        };
        // >>> WHAT THE GUEST ACTUALLY BINDS, so the module reads only those lanes and emits the
        // >>> fill CONSTANT for the rest - see `VertexAttribute::guest_components`. Recorded for
        // every attribute the caller named, whatever its format: it is what takes a stream off
        // the repack, and it is independent of the integer question below.
        a.guest_components = Some((gcomps as u32).clamp(1, 4));
        // GXM 0..3 are U8 / S8 / U16 / S16 - the PLAIN integer family, delivered to the shader
        // as the value itself. 4..7 are the normalised twins and 8/9 are float: those already
        // have an exact float fetch and must keep it.
        let signed = match fmt {
            0 | 2 => false,
            1 | 3 => true,
            _ => continue,
        };
        // NO LANE TEST. The narrowest integer format is two or four components wide, so a
        // one- or three-component attribute is FETCHED wider than the guest bound - but the
        // module does not READ those lanes: `guest_components`, set just above, makes every one
        // of them a baked constant. The proof the gap needs is already made, once, for both
        // halves of this.
        a.int_fetch = Some(signed);
    }

    // A PASSTHROUGH fragment program emits the PDS-loaded primary attributes verbatim (see
    // `plan_interface`), so its colour is the primary-attribute bank at register 0 - never the
    // OUTPUT bank, which nothing in the program ever writes. `plan_bindings` infers the colour
    // register from what the stream WRITES and cannot answer for a stream that writes nothing;
    // its default (`NativeO0`) would read four registers the hardware never fills.
    //
    // The precision comes from the interpolant that lands there: a HALF descriptor packs four
    // components into two registers, which is the shape `ColorPrecision::F16` unpacks.
    //
    // When that interpolant carries NO data of its own (`register_count == 0`) and a sample prefetch
    // lands at register 0, the colour IS the prefetched texel - vita2d's plain texture
    // program, every sprite of every homebrew: a body of one PHAS word and nothing else.
    // Its precision is the prefetch's: two registers hold four packed F16 channels, four
    // hold F32 (`Interpolant::prefetch_regs`). The descriptor's `half` flag describes the
    // (empty) data part and says F32 here, which read the packed halves as raw floats and
    // painted every sprite black.
    if is_passthrough(&frc.shader, &pa_read_before_write(&frc.shader).0) {
        fplan.color = ColorOutput::NonNativePa(0);
        fplan.color_precision = match fprog.interpolants.iter().find(|it| it.pa_base == 0) {
            Some(it) if it.register_count == 0 && it.prefetch.is_some() && it.prefetch_base() == Some(0) => {
                if it.prefetch_regs >= 4 {
                    crate::module::ColorPrecision::F32
                } else {
                    crate::module::ColorPrecision::F16
                }
            }
            Some(it) if it.half => crate::module::ColorPrecision::F16,
            _ => crate::module::ColorPrecision::F32,
        };
    }

    // A prefetched unit is usually sampled ONLY by the PDS, so the instruction walk that built
    // the binding plan never saw it. Merge those units in so the renderer binds them.
    for pf in &iface.prefetches {
        match fplan.samplers.iter_mut().find(|b| b.unit == pf.unit) {
            Some(b) => {
                b.coords = b.coords.max(pf.binding().coords);
                // A prefetch of a CUBE sampler types the texture, whatever the instruction walk
                // decided from the stream - the two describe the same unit.
                b.cube |= pf.cube;
            }
            None => fplan.samplers.push(pf.binding()),
        }
    }
    fplan.samplers.sort_unstable_by_key(|b| b.unit);

    // The units the caller has established hold a 64-bit RAW texture (see
    // `LinkOptions::raw_units`): their bindings become `texture_2d<u32>` and their prefetches
    // raw loads. The body was emitted from the instruction stream alone, so a unit the program
    // samples ITSELF cannot be re-typed here - refuse it by name rather than bind a uint texture
    // to a `textureSample`.
    for b in fplan.samplers.iter_mut() {
        if b.unit < 64 && opts.raw_units & (1u64 << b.unit) != 0 {
            if frc.shader.instrs.iter().any(|i| matches!(i.op, Op::Tex { unit, .. } | Op::TexGather { unit, .. } if unit == b.unit)) {
                return Err(LinkError::RawUnitSampledInProgram { unit: b.unit });
            }
            if b.coords != 2 || b.cube {
                return Err(LinkError::RawUnitNotTwoDimensional { unit: b.unit });
            }
            b.raw = true;
        }
    }

    // Container literals the driver preloads into SA registers above the uniform buffer. An SA
    // read that is neither a uniform nor a literal is unmodeled texture state and hard-fails.
    let vliterals = secondary_attr_init(&vrc.shader, &vprog)?;
    let fliterals = secondary_attr_init(&frc.shader, &fprog)?;
    fplan.dual_source = opts.dual_source
        && crate::module::dual_source_eligible(
            &crate::module::with_secondary(&fprog, &frc.shader),
            fprog.sa_carried_extent(),
            &fliterals,
        );
    // In the PRIMARY stream's index space, which is what `emit_body_marked` marks. The
    // secondary stream fills SA registers and cannot read the output bank, so a split can only
    // fall in the primary.
    fplan.dual_split = crate::module::first_dest_reader(&frc.shader);
    if opts.raw64_output {
        if fplan.reads_dest_color {
            return Err(LinkError::Raw64OutputUnsupported {
                why: "it reads the destination colour, and an integer attachment has none to blend with",
            });
        }
        if frc.shader.instrs.iter().any(|i| i.op == Op::DepthF) {
            return Err(LinkError::Raw64OutputUnsupported { why: "it writes its own depth" });
        }
        fplan.dual_source = false;
        fplan.raw64_output = true;
    }

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
        // MARKED: the dual-source module builder cuts this body at a top-level instruction
        // boundary, and `build_linked_module` strips what it does not use.
        emit_body_marked(&frc.shader)
            .map_err(|e: EmitError| LinkError::FragmentRecompile(e.into()))?
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

    let dual_source = fplan.dual_source;
    let reads_dest_color = fplan.reads_dest_color && !dual_source;
    let dest_blend = frc.dest_blend;
    Ok(LinkedProgram {
        wgsl,
        vertex_bindings: vplan,
        fragment_bindings: fplan,
        vertex_varyings,
        fragment_varyings,
        vertex_hash: vprog.hash,
        fragment_hash: fprog.hash,
        reads_dest_color,
        dual_source,
        dest_blend,
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

/// The channels an instruction reads from its sources. The model itself lives on the
/// instruction ([`crate::ir::Instr::read_channels`]) so the linker, the module extent scan and
/// the corpus checks cannot drift apart; this is the linker's local name for it.
pub(crate) fn read_channels(instr: &crate::ir::Instr) -> [bool; 4] {
    instr.read_channels()
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
/// Which instructions are NOT guaranteed to run - the ones whose writes therefore cannot kill a
/// later read.
///
/// # Why the linear reading was wrong, and what it cost
/// The read-before-write analysis above walked the stream in order, so a write ANYWHERE earlier
/// counted as defining the register. In an if/else that is exactly backwards: the two arms are
/// mutually exclusive, so a write in one arm defines nothing for a read in the other. A retail
/// title's world material is that shape -
///
/// ```text
///   [1] test p0 = (g_CloudShadowAlpha > 0)
///   [2] br  p0 -> 6            ; skip the cheap arm
///   [3] pa[4] = pa[0]          ; cheap arm: no shadow lookup
///   [4] pa[5] = pa[1]
///   [5] br      -> 11          ; over the else arm
///   [6] pa[4] = sample(unit 13, coord = pa[4..5])   ; <-- reads the varying
/// ```
///
/// - and the writes at [3]/[4] made the linker treat `pa[4..5]` as the shader's own scratch. It
/// then routed no varying into them, so the shadow lookup sampled at a coordinate of ZERO on
/// every pixel. That is silent: the pair links, every draw renders, and the picture is wrong
/// only where the branch is taken - which is why this title's front end looked perfect and its
/// courses rendered black.
///
/// The rule is the ordinary dominance one, computed structurally because these programs are
/// forward skips and loops rather than an arbitrary graph: an instruction is conditional if it
/// carries a predicate, if some forward branch jumps over it, or if it sits in a loop body. Only
/// the writes of the rest may kill. This is CONSERVATIVE in the safe direction - it can only
/// make the linker route MORE of what a fragment declares, never less.
pub(crate) fn conditionally_executed(shader: &Shader) -> Vec<bool> {
    let n = shader.instrs.len();
    let mut conditional = vec![false; n];
    for (i, instr) in shader.instrs.iter().enumerate() {
        if instr.pred != Predicate::Always {
            conditional[i] = true;
        }
        let Op::Branch { rel } = instr.op else { continue };
        let target = i as i64 + rel as i64;
        let (lo, hi) = if rel > 0 {
            // The words the branch jumps over run only when it is NOT taken.
            (i + 1, target.min(n as i64) as usize)
        } else {
            // A back edge: everything from the loop head to the edge runs a number of times the
            // stream does not fix, so none of it is guaranteed to have run at the exit.
            (target.max(0) as usize, i + 1)
        };
        for c in conditional.iter_mut().take(hi).skip(lo) {
            *c = true;
        }
    }
    conditional
}

fn pa_read_before_write(shader: &Shader) -> (Vec<bool>, Vec<bool>) {
    // Two entries per register: index `2*r + h` is half `h`. An F32 access covers both halves.
    // `written` is the LINEAR answer (any earlier write in stream order) and feeds the strict
    // `untouched` result; `written_must` counts only writes that are on EVERY path to a later
    // instruction, and is what decides which registers have to be routed - see
    // [`conditionally_executed`].
    let mut written = vec![false; BANK_REGS * 2];
    let mut written_must = vec![false; BANK_REGS * 2];
    let mut inputs = vec![false; BANK_REGS];
    let mut untouched = vec![false; BANK_REGS];
    let conditional = conditionally_executed(shader);
    for (index, instr) in shader.instrs.iter().enumerate() {
        // Sources and destination can be at DIFFERENT widths (a format convert), so each side
        // resolves its own registers - `source_register` answers for the sources, and `half`
        // here is the DESTINATION's width. See [`crate::ir::Instr::source_half_precision`].
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
                let Some((reg, halves)) = instr.source_register(src, c) else {
                    continue; // a swizzle constant reads no register
                };
                let reg = reg as usize;
                if reg >= BANK_REGS {
                    continue;
                }
                if halves.clone().any(|h| !written_must[reg * 2 + h]) {
                    inputs[reg] = true;
                }
                if !written[reg * 2] && !written[reg * 2 + 1] {
                    untouched[reg] = true;
                }
            }
        }
        if let Some(d) = instr.dest.as_ref()
            && d.bank == Bank::PrimaryAttr {
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
                        if !conditional[index] {
                            written_must[reg * 2 + h] = true;
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
    /// How many PA registers the result occupies: 4 (four unpacked F32 components), 2 (four
    /// packed F16 components) or 1 (two). See [`crate::container::Interpolant::prefetch_regs`].
    ///
    /// Writing two unconditionally is not a harmless over-write: the register after a
    /// one-register prefetch belongs to the NEXT interpolant, so it clobbers a varying the
    /// vertex stage fed correctly. That is invisible in the WGSL and shows up only as a
    /// surface shading black.
    regs: u32,
    /// Interface component indices of the sample coordinates, in order. EMPTY when
    /// [`Self::coords_unfed`] - there is no varying to take them from.
    coords: Vec<usize>,
    /// The sampler is a cube map, so the coordinate is a three-component direction.
    cube: bool,
    /// A PROJECTIVE lookup ([`crate::container::PrefetchLookup::Projective`]): [`Self::coords`]
    /// holds the texcoord's `x`, `y` and its fourth component `w`, and the sample is taken at
    /// `xy / w`. The binding stays a flat 2D texture - it is the coordinate that is projective,
    /// not the texture.
    projective: bool,
    /// >>> THE VERTEX PROGRAM PRODUCES NO SUCH TEXCOORD AT ALL, so the coordinate is the
    /// >>> texture-coordinate DEFAULT and the draw is emitted anyway.
    ///
    /// A PDS prefetch names its coordinate by TEXCOORD index alone, and that index need not be
    /// one of the fragment's own interpolants - a texcoord only the PDS reads costs no PA
    /// registers. Usually the paired vertex produces it. One pair of a baseball title's 128 does
    /// not: `vert_843374b8` writes exactly `[Color0, TexCoord(1)]` and its fragment's prefetch
    /// names TEXCOORD 0.
    ///
    /// That is not a decode error - it was checked. Read as an ORDINAL into the vertex's own
    /// texcoord list, `0` would be `TexCoord(1)` and the pair would link; but over that title's
    /// 278 real prefetches the ordinal reading resolves 4 that the semantic reading resolves and
    /// the ordinal cannot, and disagrees with it nowhere
    /// (`a_prefetch_source_texcoord_read_as_an_ordinal_agrees_wherever_the_semantic_reading_works`).
    /// The semantic reading is the right one and this pair really does ask the PDS for a
    /// coordinate its vertex never writes.
    ///
    /// On the hardware the interpolator then iterates lanes the vertex never filled, which is
    /// undefined - the same situation as an SA register that is read and never written, where
    /// this linker already binds 0.0 and emits the pair rather than dropping the draw. Refusing
    /// costs the whole draw: on the user's device it was SIX draws a scene, every scene, and a
    /// draw that renders nothing is indistinguishable from one that was never issued.
    /// `VITASLOP_GXP_PREFETCH_UNFED=0` restores the refusal.
    coords_unfed: bool,
}

impl PlannedPrefetch {
    /// The binding this sample needs, which the shader's own SMP instructions may not mention -
    /// a prefetched unit is often sampled ONLY by the PDS.
    /// The coordinate ARITY the emitted sample uses. ONE answer, because the module's
    /// `textureSample` and the BINDING that types its texture have to agree or the module does
    /// not compile - and they did not.
    ///
    /// # The bug this replaced, which only TINT could see
    /// `binding()` reported `coords.len()`, and for an UNFED prefetch that list is EMPTY - there
    /// is no varying to take the coordinates from, which is the whole meaning of
    /// [`Self::coords_unfed`]. So an unfed CUBE prefetch declared `coords: 0`, the binding plan
    /// typed its texture `texture_2d<f32>`, and the emission (which asks the sampler's shape, not
    /// the list) wrote `textureSample(t1, s1, vec3<f32>(0.0, 0.0, 0.0))`. That is no overload at
    /// all: real Chrome answers `no matching call to 'textureSample(texture_2d<f32>, sampler,
    /// vec3<f32>)'`, the pipeline is refused, and a refused pipeline poisons its pair and drops
    /// every draw of it [[vitaslop-a-failed-submit-loses-the-whole-frame]]. Found by compiling
    /// every module a corpus can produce in Chrome itself
    /// (`corpus.rs::write_every_linked_pair_wgsl` + `vitaslop-web/e2e/tintcheck.mjs`) - 1 of 595 -
    /// which is the check the naga-based corpus tests cannot make
    /// [[vitaslop-tint-rejects-what-naga-accepts]].
    fn coord_arity(&self) -> u8 {
        if self.coords_unfed || self.projective {
            if self.cube { 3 } else { 2 }
        } else {
            self.coords.len() as u8
        }
    }

    fn binding(&self) -> TexBinding {
        TexBinding { unit: self.unit, coords: self.coord_arity(), cube: self.cube, raw: false }
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
    // Only when a log filter was NAMED: this crate has no tracing, and a run nobody
    // configured prints nothing. The fill is the measured hardware value, so this is a
    // note, not a warning.
    if std::env::var_os("VITASLOP_LOG").is_some() || std::env::var_os("RUST_LOG").is_some() {
        eprintln!(
            "gxp link: {usage:?} is allocated {declared} fragment registers but the vertex program \
             fills only {vertex_components} components - the surplus is fed the iterator's default \
             (0, 0, 0, 1), the same fill a vertex attribute's missing components get"
        );
    }
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

/// Say, once per (unit, texcoord), that a PDS prefetch names a texcoord the paired vertex
/// program does not produce, so its coordinate is the texture-coordinate default.
///
/// A WARNING and not a status note: the sample lands at (0,0) for every pixel of the draw, which
/// is a flat texel where the title meant a lookup. It is still the right trade - the alternative
/// is the whole draw, and this at least paints geometry that can be seen and recognised - but it
/// is a fix we owe [[vitaslop-a-warning-means-we-owe-a-fix]].
fn report_prefetch_coord_unfed(unit: u8, texcoord: u8) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u8, u8)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((unit, texcoord)) {
        return;
    }
    eprintln!(
        "gxp link: unit {unit}'s PDS prefetch names TEXCOORD {texcoord}, which the paired VERTEX          program does not produce - the hardware would interpolate lanes the vertex never wrote,          so the coordinate is the texcoord DEFAULT (0,0) and the pair is emitted rather than the          draw dropped. The sample is one flat texel for the whole draw."
    );
}

/// `VITASLOP_GXP_PREFETCH_UNFED=0` - refuse the pair again instead, which drops the draw. See
/// [`PlannedPrefetch::coords_unfed`].
fn prefetch_unfed_default() -> bool {
    std::env::var("VITASLOP_GXP_PREFETCH_UNFED").as_deref() != Ok("0")
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
    colour_masked_off: bool,
) -> Result<Vec<OutputVarying>, LinkError> {
    let vout = &vprog.output_varyings;
    if vout.len() > MAX_PERMUTED_VARYINGS {
        return Err(LinkError::VaryingOrderAmbiguous { varyings: vout.len(), surviving: 0 });
    }
    let mut survivors: Vec<Vec<OutputVarying>> = Vec::new();
    for perm in permutations(vout.len()) {
        let candidate = permute_varyings(vout, &perm);
        if plan_interface_with(fprog, fshader, &candidate, colour_masked_off).is_ok() {
            survivors.push(candidate);
        }
    }
    match survivors.len() {
        1 => Ok(survivors.pop().expect("checked")),
        n => Err(LinkError::VaryingOrderAmbiguous { varyings: vout.len(), surviving: n }),
    }
}

/// Which OUTPUT-bank scalar lanes a vertex program's code actually writes.
///
/// This is the vertex program's own statement about its lane layout, and it is the evidence
/// the containers do not carry. A varying occupies a contiguous run of lanes, so the set of
/// written lanes - and, more tellingly, the HOLES in it - shows where the runs begin and end.
fn written_output_lanes(vshader: &Shader) -> Vec<bool> {
    let mut w = Vec::new();
    for instr in &vshader.instrs {
        let Some(d) = instr.dest.as_ref() else { continue };
        if d.bank != Bank::Output {
            continue;
        }
        for c in 0..4 {
            if instr.write_mask[c] {
                let lane = d.index as usize + c;
                if w.len() <= lane {
                    w.resize(lane + 1, false);
                }
                w[lane] = true;
            }
        }
    }
    w
}

/// Does the CONVENTION's lane layout agree with what the vertex program's code writes?
///
/// # Why this is the check that was missing
/// The varyings block states the SET of varyings and each one's width, but not their ORDER,
/// and the convention (colours, fog, then texcoords ascending) fills that in. Until now the
/// convention was either trusted outright (`Assumed`) or refused outright (`Ambiguous`, the
/// COLOR1 case) - and neither was ever checked against the one witness that can speak: the
/// program's own writes.
///
/// It can speak, and it is precise. MEASURED on `vert_82bfdfb0`, whose layout
/// the convention puts at `Fog@4..5, TexCoord(0)@6..10, TexCoord(1)@10..14, ...`: the code
/// writes lanes `[4, 6,7,8,9, 10..21]` - **lane 5 is not written**, exactly the reserved
/// second lane of a one-component Fog. A layout that placed anything else at lane 4 would
/// have to explain that hole. Across that program's siblings this hole falls where the
/// convention says every time, which is why that title renders correctly under it.
///
/// So: accept the convention when the code does not CONTRADICT it, using the same verdict the
/// corpus test `assumed_varying_orders_the_vertex_code_contradicts` established and asserts on
/// across 261 assumed-order programs of four titles - only one of the three things a layout can
/// be measured against refutes it:
///
/// * a lane the code writes that no declared run covers and that lies BELOW the top of the
///   layout REFUTES it. Only a wrong layout can produce one, because a wrong width or a wrong
///   position shifts every run after it and the writes then land in the gaps.
/// * a lane written at or ABOVE the top of the layout is not evidence about the order at all.
///   The output bank's top holds things that are not varyings - clip planes, point size, and
///   the reserved lane a varyings block can declare above its texcoords - and no run is placed
///   there for a permutation to get wrong.
/// * a declared run the program never STARTS is not evidence either. The hardware allows a
///   declared varying to go unproduced, and that is exactly what the corpus test's UNWRITTEN
///   verdict names; requiring every run to be started refuses a program for something the
///   layout does not claim. (This gate did require it, which is what refused a golf title's
///   three world vertex programs: each declares ten texture coordinates and fills eight.)
///
/// Trailing lanes of a run may legitimately be unwritten (a three-component colour in a
/// four-lane slot, a one-component fog in its two-lane slot), and those HOLES are the
/// convention's own corroboration where they fall exactly where it says.
fn convention_agrees_with_the_code(vout: &[OutputVarying], vshader: &Shader) -> bool {
    let written = written_output_lanes(vshader);
    let in_a_run = |lane: usize| {
        // The clip position owns the first lanes and is not a varying.
        lane < crate::container::VERTEX_POSITION_LANES as usize
            || vout.iter().any(|v| {
                let lo = v.base_lane as usize;
                lane >= lo && lane < lo + v.components as usize
            })
    };
    let top = vout
        .iter()
        .map(|v| v.base_lane as usize + v.components as usize)
        .max()
        .unwrap_or(0);
    !written.iter().enumerate().any(|(lane, &w)| w && lane < top && !in_a_run(lane))
}

/// The vertex lane order the paired FRAGMENT's declaration implies, or `None` when the two
/// sides do not name the same set of varyings.
///
/// # The competing reading, spelled out so it can be MEASURED rather than argued
/// The convention orders a vertex's varyings `Color0, Color1, Fog, TexCoord0..9`. The
/// fragment states an order explicitly - its interpolant descriptors accumulate a PA base in
/// declaration order - and a texcoord consumed by a PDS PREFETCH takes its place in that
/// sequence even though it never lands in a PA register of its own. The two disagree on most
/// of one title's pairs, and neither reading had a witness that could settle it.
///
/// It has one now: one title's trackside billboard (`79d26abb534d2ea4`). Under the
/// convention its `Color1` is fed the vertex's UV attribute and the panel paints a
/// green-to-red ramp; the fragment's own order puts `TexCoord(0)` in that slot instead.
/// `VITASLOP_GXP_VARYING_ORDER=fragment` selects this reading so the two can be compared on
/// that frame. It is a DIAGNOSTIC and off by default - the convention still ships until the
/// oracle says otherwise.
fn fragment_declared_order(vprog: &Program, fprog: &Program) -> Option<Vec<OutputVarying>> {
    let mut want: Vec<VaryingUsage> = Vec::new();
    for it in &fprog.interpolants {
        want.push(it.usage);
        // The prefetched texcoord is interpolated and delivered like any other varying; it is
        // simply consumed by the sampler rather than left in a register, so it occupies its
        // place in the sequence and must not be skipped.
        if let Some(pf) = &it.prefetch {
            want.push(VaryingUsage::TexCoord(pf.source_texcoord));
        }
    }
    let vout = &vprog.output_varyings;
    // All-or-nothing, for the same reason `attribute_order` is: a partial cover would order
    // some varyings by the fragment and the rest by the convention, which is neither reading.
    let mut a: Vec<String> = want.iter().map(|u| format!("{u:?}")).collect();
    let mut b: Vec<String> = vout.iter().map(|v| format!("{:?}", v.usage)).collect();
    a.sort();
    b.sort();
    if a != b {
        return None;
    }
    let perm: Vec<usize> = want
        .iter()
        .map(|u| vout.iter().position(|v| v.usage == *u).expect("cover checked above"))
        .collect();
    Some(permute_varyings(vout, &perm))
}

/// One vertex ATTRIBUTE copied straight into output lanes: `(the varying its semantic names,
/// the lanes it was copied to)`, one entry per attribute that is forwarded at all.
///
/// # Why a MOVE is evidence the varyings block is not
/// The block states the SET of varyings and each one's width; the order is the convention's to
/// fill in, and there has been no witness for it that is not itself a convention.
/// `mov Output[n] <- PrimaryAttr[m]` is one: register `m` belongs to a declared attribute, that
/// attribute's semantic names a varying, and lane `n` therefore carries that varying. It is a
/// statement about ONE lane, so unlike [`crate::container::attribute_order`] - which needs the
/// attributes to cover the declared set EXACTLY, i.e. a passthrough program - it survives the
/// other varyings being computed rather than forwarded. Almost no real program is a
/// passthrough; plenty forward one input.
///
/// Only `Mov` counts. A value that is computed with says nothing about which varying it IS.
fn forwarding_claims(vprog: &Program, vshader: &Shader) -> Vec<(VaryingUsage, Vec<u32>)> {
    use crate::container::{ParamCategory, SEMANTIC_COLOR, SEMANTIC_FOGCOORD, SEMANTIC_TEXCOORD};
    // The varying an attribute's semantic names. POSITION, normals, tangents and blend weights
    // are consumed rather than forwarded and name no varying.
    let usage_of = |p: &crate::container::Parameter| match p.semantic {
        SEMANTIC_FOGCOORD => Some(VaryingUsage::Fog),
        SEMANTIC_COLOR => match p.semantic_index {
            0 => Some(VaryingUsage::Color0),
            1 => Some(VaryingUsage::Color1),
            _ => None,
        },
        SEMANTIC_TEXCOORD => Some(VaryingUsage::TexCoord(p.semantic_index)),
        _ => None,
    };
    let mut by_attr: Vec<(i32, VaryingUsage, Vec<u32>)> = Vec::new();
    for instr in &vshader.instrs {
        if !matches!(instr.op, Op::Mov) {
            continue;
        }
        let (Some(d), Some(s)) = (instr.dest.as_ref(), instr.srcs.first()) else { continue };
        if d.bank != Bank::Output || s.bank != Bank::PrimaryAttr {
            continue;
        }
        for c in 0..4 {
            if !instr.write_mask[c] {
                continue;
            }
            let src_reg = s.index as u32 + s.swizzle[c] as u32;
            let Some(a) = vprog.parameters.iter().find(|a| {
                a.category == ParamCategory::Attribute
                    && a.resource_index >= 0
                    && src_reg >= a.resource_index as u32
                    && src_reg < a.resource_index as u32 + a.component_count as u32
            }) else {
                continue;
            };
            let Some(u) = usage_of(a) else { continue };
            let lane = d.index as u32 + c as u32;
            match by_attr.iter_mut().find(|(reg, _, _)| *reg == a.resource_index) {
                Some((_, _, lanes)) => {
                    if !lanes.contains(&lane) {
                        lanes.push(lane);
                    }
                }
                None => by_attr.push((a.resource_index, u, vec![lane])),
            }
        }
    }
    for (_, _, lanes) in &mut by_attr {
        lanes.sort_unstable();
    }
    by_attr.into_iter().map(|(_, u, lanes)| (u, lanes)).collect()
}

/// Does a candidate lane layout put a forwarded attribute into a varying that is not the one its
/// semantic names? Returns the first contradiction, spelled out.
///
/// # The bar this sets, and why it is set so high
/// A claim counts ONLY when the attribute's lanes are exactly one declared run - same first
/// lane, same length. Anything less is not an order question:
///
/// - A PARTIAL run is PACKING. One title writes `VertexColour1` into lane 9 of a four-lane
/// `TexCoord(0)@6..10` - a scalar tucked into a spare texcoord channel, which is a thing
/// shaders do and says nothing about where TexCoord(0) sits. - Lanes spanning SEVERAL runs is
/// DUPLICATION. One title's `uv1_uv2` is copied to `16..18` and `18..20` at once; one
/// attribute cannot name two varyings, so it names neither.
///
/// What is left is the shape that can only be an order statement: a whole run, filled by one
/// attribute, whose semantic names a different varying than the layout does. MEASURED across
/// four corpora, exactly that shape appears on one title's billboard family (`In.UV1` fills
/// the whole of what the convention calls `Color1`) and on ONE program of a second title, and
/// on NOTHING at all in a third - which is what makes it usable: that third renders correctly
/// under the convention today and any reading that moves its lanes is wrong.
fn forwarding_contradicts(vout: &[OutputVarying], claims: &[(VaryingUsage, Vec<u32>)]) -> Option<String> {
    for (usage, lanes) in claims {
        let (Some(&first), Some(&last)) = (lanes.first(), lanes.last()) else { continue };
        // Contiguous, or it is not a run.
        if last + 1 - first != lanes.len() as u32 {
            continue;
        }
        let Some(run) = vout
            .iter()
            .find(|v| v.base_lane == first && v.components == lanes.len() as u32)
        else {
            continue;
        };
        if run.usage != *usage {
            return Some(format!(
                "lanes {first}..{} are filled by one attribute whose semantic is {usage:?}, but \
                 this layout puts {:?} there",
                last + 1,
                run.usage
            ));
        }
    }
    None
}

/// The lanes each declared varying is filled from, when the vertex program forwards EVERY one of
/// them straight from an attribute - a COMPLETE reading of the code rather than one claim plus a
/// convention. Returns `None` the moment anything is missing or inconsistent.
///
/// # Why this is a separate, much stricter reading than [`forwarding_claims`]
/// A single `mov Output[n] <- PrimaryAttr[m]` is weak evidence and MEASURED to be wrong on real
/// programs: a golf title's course family forwards a COLOR-semantic attribute into the four
/// lanes the convention calls `TexCoord(0)`, because passing a colour down a spare texcoord
/// channel is a thing shaders do. Acting on that one claim re-orders twelve varyings and takes
/// the sky, the course and the whole UI with it (MEASURED: `hole-one.recipe` f001880, 89% of
/// pixels, world black and sky flat green).
///
/// What is not weak is a claim set that accounts for the WHOLE layout: every declared varying
/// named, each from one attribute, each run exactly its declared width, and the runs tiling the
/// lane budget with no gap and no overlap. There is nothing left for a convention to decide, and
/// nothing a channel-reuse can hide in - a program that packs a colour into a texcoord and
/// computes the rest never produces a complete cover at all, so it is refused rather than
/// mis-read.
///
/// PACKs count as forwards where `forwarding_claims` takes only moves: a float-to-float VPCK
/// changes a value's STORAGE width and not the value, which is how the fighting title's UI
/// program writes its vertex colour (`pack o[4].xy <- pa[8].zy`, `pack o[6].xy <- pa[8].xw` -
/// swizzled, but every lane from the one attribute).
fn forwarding_cover(vprog: &Program, vshader: &Shader) -> Option<Vec<OutputVarying>> {
    use crate::container::{ParamCategory, SEMANTIC_COLOR, SEMANTIC_FOGCOORD, SEMANTIC_TEXCOORD};
    let usage_of = |p: &crate::container::Parameter| match p.semantic {
        SEMANTIC_FOGCOORD => Some(VaryingUsage::Fog),
        SEMANTIC_COLOR => match p.semantic_index {
            0 => Some(VaryingUsage::Color0),
            1 => Some(VaryingUsage::Color1),
            _ => None,
        },
        SEMANTIC_TEXCOORD => Some(VaryingUsage::TexCoord(p.semantic_index)),
        _ => None,
    };
    let declared = &vprog.output_varyings;
    if declared.len() < 2 {
        return None; // one varying cannot be mis-ordered
    }
    // Lane -> the usage the code puts there. A lane written twice, or written from two different
    // attributes, is not a cover.
    let mut at: Vec<Option<VaryingUsage>> = vec![None; BANK_REGS];
    for instr in &vshader.instrs {
        if !matches!(instr.op, Op::Mov | Op::Pack { .. }) || instr.pred != Predicate::Always {
            continue;
        }
        let (Some(d), Some(src)) = (instr.dest.as_ref(), instr.srcs.first()) else { continue };
        if d.bank != Bank::Output || src.bank != Bank::PrimaryAttr {
            continue;
        }
        // A half-precision destination packs two components per register, so its lanes are not
        // one per channel and this reading does not model it. Refuse rather than guess.
        if instr.half_precision {
            return None;
        }
        for c in 0..4 {
            if !instr.write_mask[c] {
                continue;
            }
            let sel = src.swizzle[c] as u32;
            if sel > 3 {
                continue;
            }
            let src_reg = src.index as u32 + sel;
            let Some(a) = vprog.parameters.iter().find(|a| {
                a.category == ParamCategory::Attribute
                    && a.resource_index >= 0
                    && src_reg >= a.resource_index as u32
                    && src_reg < a.resource_index as u32 + a.component_count as u32
            }) else {
                continue;
            };
            let Some(u) = usage_of(a) else { continue };
            let lane = d.index as usize + c;
            match at.get(lane) {
                // Written twice with different meanings: not a cover.
                Some(Some(prev)) if *prev != u => return None,
                Some(_) => at[lane] = Some(u),
                None => return None,
            }
        }
    }

    // Every declared varying must be covered by exactly its own contiguous run of its own width.
    let origin = declared.iter().map(|v| v.base_lane).min()?;
    let mut out: Vec<OutputVarying> = Vec::with_capacity(declared.len());
    for v in declared {
        let first = (0..at.len()).find(|&l| at[l] == Some(v.usage))?;
        let count = at[first..].iter().take_while(|u| **u == Some(v.usage)).count() as u32;
        if count != v.components || at[first..].iter().filter(|u| **u == Some(v.usage)).count() as u32 != count {
            return None;
        }
        out.push(OutputVarying { usage: v.usage, base_lane: first as u32, components: v.components });
    }
    out.sort_by_key(|v| v.base_lane);
    // The runs must TILE the budget: start at the same origin the container placed them at, and
    // abut with no gap. A gap is a lane the code never wrote, which is a varying this reading
    // did not see and therefore did not cover.
    let mut lane = origin;
    for v in &out {
        if v.base_lane != lane {
            return None;
        }
        lane += v.components;
    }
    (out != *declared).then_some(out)
}

/// Re-order a convention-placed layout so that every forwarded attribute's usage STARTS at the
/// lane that attribute actually fills. Returns `None` when no permutation does.
///
/// # Why this is a reading of the VERTEX and the two refused ones were not
/// A `mov Output[n] <- PrimaryAttr[m]` says lane `n` carries the varying that attribute's
/// semantic names ([`forwarding_claims`]). The convention says which usage sits where. When they
/// disagree, the move is the statement about THIS program and the convention is a statement about
/// programs in general, so the move wins - and it wins without consulting the paired fragment
/// (objection 1 of the history at the call site) and without changing any usage's declared WIDTH
/// (objection 2). Only the ORDER moves; every usage keeps the components the container gave it,
/// and the lane budget therefore still closes exactly.
///
/// **The rule is a START match, not a whole-run match.** A claim's width is the container's
/// declared component count, which for an attribute the guest binds narrower than it declares
/// spans more lanes than the usage's run - so requiring the run to have the claim's LENGTH is
/// what made the witness unusable. The first lane is the part the copy cannot be wrong about.
///
/// MEASURED on one title's sky/background family (five vertex programs, all identical in
/// shape): the convention gives `Color0@4x4 Color1@8x4 TexCoord(0)@12x2` while the code copies
/// `In.UV1` - a TEXCOORD - from lane 8. The permutation that starts `TexCoord(0)` at 8 is
/// `Color0@4x4 TexCoord(0)@8x2 Color1@10x4`, whose budget closes at 14 lanes exactly, and it is
/// what makes the sky sample its own gradient texture instead of a vertex colour, removes the
/// vertical seam across the road, and paints the trackside billboard its artwork. On the
/// billboard's own program (17 lanes, four varyings) the same permutation leaves the computed
/// reflection vector at `TexCoord(1)@14x3`, feeding the cube map coordinates the convention had
/// pointed at lanes nothing writes.
/// Is the claim WIDTH consistency test in force (`VITASLOP_GXP_CLAIM_WIDTH=0` turns it off)?
///
/// An A/B ARM rather than a mode, so both readings are reachable from ONE build - the arms must
/// differ in one thing or the comparison is between two binaries
/// ([[vitaslop-browser-ab-needs-a-negative-control]]).
fn claim_width_test() -> bool {
    std::env::var("VITASLOP_GXP_CLAIM_WIDTH").as_deref() != Ok("0")
}

fn layout_from_forwarding_claims(
    vout: &[OutputVarying],
    claims: &[(VaryingUsage, Vec<u32>)],
    // Whether the layout being refuted is the ATTRIBUTE reading (`VaryingOrder::Known`). Only
    // then may the ordering SEARCH below run - see `exhaustive_layout_from_claims`.
    passthrough_reading: bool,
) -> Option<Vec<OutputVarying>> {
    // Which usage each claimed START lane demands, and HOW MANY LANES that claim covers. A lane
    // claimed by two attributes is not a statement about either, so it is dropped rather than
    // resolved.
    let mut want: Vec<(u32, VaryingUsage, u32)> = Vec::new();
    for (usage, lanes) in claims {
        let (Some(&first), Some(&last)) = (lanes.first(), lanes.last()) else { continue };
        if last + 1 - first != lanes.len() as u32 {
            continue; // not one run, so it names no single varying
        }
        match want.iter().position(|(l, _, _)| *l == first) {
            Some(i) if want[i].1 != *usage => {
                want.remove(i);
            }
            Some(_) => {}
            None => want.push((first, *usage, lanes.len() as u32)),
        }
    }
    if want.is_empty() {
        return None;
    }
    // Walk the lanes in order. At each run boundary the evidence gets first refusal: if a claim
    // names a usage for THIS lane and that usage is still unplaced, it goes here. Otherwise the
    // convention's own order supplies the next one.
    //
    // Greedy rather than "satisfy every claim at once" because the two are not equally strong. A
    // claim's start lane is only reachable if the widths of what precedes it add up to it, and
    // whether they do is a fact about the container, not a choice - so a claim the walk never
    // arrives at is unsatisfiable by construction and ignoring it decides nothing. Requiring all
    // of them instead makes one unreachable claim throw away a layout the reachable ones settle,
    // which is exactly what happened on the family this was written for: `In.VColor` fills lanes
    // 12..14, no width arrangement puts a four-lane `Color0` there, and demanding it refused the
    // `TexCoord(0)@8` the same program states plainly.
    // The SLOT each usage occupies, taken from the convention's own layout rather than from its
    // component count - because the two are NOT the same and the difference is a whole lane.
    //
    // A one-component Fog sits in a TWO-lane slot (the second lane is reserved and the program
    // never writes it, which is the hole `convention_agrees_with_the_code` reads as the
    // convention's own corroboration). Re-walking the lanes at `lane += components` closes that
    // hole and shifts EVERY varying after it down by one, which is a width change - the one
    // thing this resolver's contract says it never makes. MEASURED on a golf title's
    // full-screen composite: the convention places `Fog@8x1 TexCoord(0)@10x4` and the walk
    // produced `Fog@8x1 TexCoord(0)@9x4`, so the passthrough fragment read its colour from
    // `o[9..12]` - one lane below the four the vertex writes - and the whole 960x544 world
    // scene came out BLACK behind an opaque quad whose alpha came from a lane nothing wrote.
    let mut stride: Vec<u32> = Vec::with_capacity(vout.len());
    for (i, v) in vout.iter().enumerate() {
        match vout.get(i + 1).map(|n| n.base_lane) {
            // The last run has no successor to measure against, so its own components are all
            // that is known; nothing is placed after it either way.
            None => stride.push(v.components),
            Some(n) if n >= v.base_lane + v.components => stride.push(n - v.base_lane),
            // Overlapping or descending runs are not a layout this can re-walk at all, and a
            // walk over one would invent lane numbers. Refuse instead.
            Some(_) => return None,
        }
    }
    let origin = vout.iter().map(|v| v.base_lane).min().unwrap_or(0);

    let mut placed = vec![false; vout.len()];
    let mut out: Vec<OutputVarying> = Vec::with_capacity(vout.len());
    let mut lane = origin;
    let mut satisfied = false;
    while out.len() < vout.len() {
        // >>> A CLAIM MAY ONLY PLACE A VARYING WIDE ENOUGH TO HOLD THE WHOLE RUN IT NAMES, and
        // >>> that is a consistency test on the evidence rather than an extra convention.
        //
        // The claim states that lanes `first..first+len` are ALL filled from one attribute whose
        // semantic is this usage. Putting a narrower varying at `first` hands the rest of that
        // run to a DIFFERENT varying - so the layout contradicts the very copy it was derived
        // from, and the start match is then honouring half a claim.
        //
        // The width comes from the attribute's DECLARED component count, which is wider than the
        // guest's binding whenever a title binds an attribute narrower than its shader declares
        // - the objection the notes on this function record. That case is exactly the one this
        // test removes: MEASURED on a baseball title's stadium family, `In.UV1` declares four
        // components and is bound `F16x2`, so its copy fills lanes 4..8 (the last two with the
        // missing-component fill) and the claim named the two-lane `TexCoord(1)`. Honouring the
        // start alone moved `TexCoord(1)` to lane 4 and pushed `Color0` to 6..10, which put the
        // ALBEDO prefetch's texcoord on lanes 10..12 - lanes the vertex program never writes -
        // so every stadium surface sampled its texture at a constant (0,0) and the frame came out
        // flat and banded. With this test the claim is unsatisfiable, the convention stands, and
        // the stadium's signage, brickwork and scoreboard are drawn from their own textures.
        //
        // It does NOT re-impose the whole-run match this function's notes rejected: a claim whose
        // run is exactly the width of the usage it names still places it, which is the shape the
        // racing title's sky/background family is resolved by.
        let claimed = want.iter().find(|(l, _, _)| *l == lane).and_then(|(_, u, len)| {
            let need = if claim_width_test() { *len } else { 0 };
            vout.iter()
                .position(|v| v.usage == *u && v.components >= need)
                .filter(|&i| !placed[i])
        });
        satisfied |= claimed.is_some();
        let i = claimed.or_else(|| (0..vout.len()).find(|&i| !placed[i]))?;
        placed[i] = true;
        out.push(OutputVarying {
            usage: vout[i].usage,
            base_lane: lane,
            components: vout[i].components,
        });
        lane += stride[i];
    }
    // A walk that satisfied no claim has resolved NOTHING, whatever it did to the lanes: the
    // claims are the entire evidence this function acts on, and a layout that answers none of
    // them is a re-ordering with no witness behind it. A walk that lands back on the convention
    // has likewise resolved nothing. (The claims are only consulted at all when
    // `forwarding_contradicts` has already fired, so both are the case where the contradiction
    // is real but the evidence cannot reach the lane that would fix it.)
    if satisfied && out != vout {
        return Some(out);
    }
    // >>> THE GREEDY WALK CANNOT REACH A CLAIM THAT LIES PAST ITS OWN FIRST STEP, and that is
    // >>> not a claim that decides nothing - it is one that decides the order of what comes
    // >>> BEFORE it. Search the orderings.
    //
    // The walk above consults a claim only when it ARRIVES at the claimed lane, so it can never
    // discover that placing a different varying first is what makes that lane reachable.
    // MEASURED on a baseball title's menu pair (`27ef71ac110cd816`): the convention places
    // `TexCoord(0)@4x2 Color0@6x4`, and the vertex's own `mov o[8..10] <- PrimaryAttr(UV)`
    // claims `TexCoord(0)` at lane 8 - a lane the walk steps straight over (4, 6, 10). The one
    // ordering that satisfies the claim is `Color0@4x4 TexCoord(0)@8x2`, and under the
    // convention's layout the fragment read the UV where the colour belongs and sampled its
    // texture at the SQUARED VERTEX COLOUR: every menu came out a saturated green-to-cyan ramp
    // with the dialog text unreadable. With it, the menu is the real one.
    //
    // >>> AND ONLY WHERE THE ATTRIBUTE READING'S OWN PREMISE IS REFUTED. `VaryingOrder::Known`
    // means the layout was read off the ATTRIBUTES, which is a reading of a PASSTHROUGH - the
    // program's outputs are its inputs, in `resource_index` order. A forwarding move that
    // contradicts it refutes exactly that premise, and the same moves are then the best evidence
    // for the real order, so searching the orderings that satisfy them reads the witness that
    // did the refuting. An `Ambiguous` layout has no such premise (the convention placed a
    // `Color1` no attribute confirms) and has its own reading downstream
    // (`convention_agrees_with_the_code`, then `resolve_ambiguous_order`); preempting it is a
    // MEASURED regression - on a racer's two `Ambiguous` world programs the search swaps
    // `Color0` and `Color1` and the grass comes back washed out in pale streaks with the
    // foliage unshaded, one shot of twelve.
    //
    // >>> UNIQUE OR NOTHING, AND BOUNDED. Several orderings usually satisfy one claim (anything
    // after the last claimed lane is free), and picking one of them would be inventing a layout,
    // not reading one. So this acts only when EXACTLY ONE ordering satisfies EVERY claim - which
    // is what makes it a reading of the vertex - and only for a program with few enough varyings
    // to enumerate. A bigger program keeps exactly the behaviour it has today.
    if !passthrough_reading {
        return None;
    }
    exhaustive_layout_from_claims(vout, &stride, &want)
}

/// Every varying count this will enumerate orderings of. `6! = 720` layouts is nothing per
/// LINKED PAIR (which happens once per pair, not per draw), and above it the search is both
/// slow and useless: a program with ten varyings and one claim has hundreds of orderings that
/// satisfy it, so the uniqueness test refuses anyway.
const EXHAUSTIVE_LAYOUT_MAX_VARYINGS: usize = 6;

/// The ONE ordering of `vout` that puts every claimed usage at the lane it is claimed at, or
/// `None` when there is no such ordering or more than one. `stride` is each ORIGINAL varying's
/// slot width (see the caller - a slot is not the component count).
fn exhaustive_layout_from_claims(
    vout: &[OutputVarying],
    stride: &[u32],
    // `(claimed start lane, usage, lanes the claimed run covers)` - the same triples the greedy
    // walk takes, so both readings apply the width consistency test identically.
    want: &[(u32, VaryingUsage, u32)],
) -> Option<Vec<OutputVarying>> {
    // The ARM BACK, so the search can be A/B'd against the greedy walk alone on one build.
    // `VITASLOP_GXP_VARYING_RESOLVE=0` is the wider arm (it turns off the forwarding readings
    // entirely); this one keeps them and removes only the ordering search.
    if vout.len() > EXHAUSTIVE_LAYOUT_MAX_VARYINGS
        || want.is_empty()
        || std::env::var("VITASLOP_GXP_VARYING_RESOLVE").as_deref() == Ok("greedy")
    {
        return None;
    }
    let origin = vout.iter().map(|v| v.base_lane).min().unwrap_or(0);
    let mut found: Option<Vec<OutputVarying>> = None;
    let mut order: Vec<usize> = Vec::with_capacity(vout.len());
    let mut used = vec![false; vout.len()];
    // Depth-first over orderings, laying each one out as it goes so a partial layout that
    // already contradicts a claim is abandoned rather than completed.
    fn walk(
        vout: &[OutputVarying],
        stride: &[u32],
        want: &[(u32, VaryingUsage, u32)],
        origin: u32,
        order: &mut Vec<usize>,
        used: &mut [bool],
        found: &mut Option<Vec<OutputVarying>>,
        second: &mut bool,
    ) {
        if *second {
            return;
        }
        if order.len() == vout.len() {
            // Lay it out and check every claim.
            let mut lane = origin;
            let mut out = Vec::with_capacity(vout.len());
            for &i in order.iter() {
                out.push(OutputVarying {
                    usage: vout[i].usage,
                    base_lane: lane,
                    components: vout[i].components,
                });
                lane += stride[i];
            }
            let need = |len: u32| if claim_width_test() { len } else { 0 };
            let ok = want.iter().all(|(l, u, len)| {
                out.iter().any(|v| v.base_lane == *l && v.usage == *u && v.components >= need(*len))
            });
            if ok {
                match found {
                    None => *found = Some(out),
                    Some(prev) if *prev != out => *second = true,
                    Some(_) => {}
                }
            }
            return;
        }
        for i in 0..vout.len() {
            if used[i] {
                continue;
            }
            used[i] = true;
            order.push(i);
            walk(vout, stride, want, origin, order, used, found, second);
            order.pop();
            used[i] = false;
        }
    }
    let mut second = false;
    walk(vout, stride, want, origin, &mut order, &mut used, &mut found, &mut second);
    if second {
        return None;
    }
    found.filter(|out| out != vout)
}

/// Say, once per distinct contradiction, that a vertex program's own forwarding moves refuse the
/// lane layout it was about to be linked with - and which layout was used instead.
fn report_forwarding_contradiction(hash: u64, why: &str, resolution: &str) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(hash) {
        return;
    }
    // Only when a log filter was NAMED (this crate has no tracing): it is a resolved layout
    // note, and a run nobody configured prints nothing.
    if std::env::var_os("VITASLOP_LOG").is_some() || std::env::var_os("RUST_LOG").is_some() {
        eprintln!("gxp link: vertex {hash:016x}: {why} - {resolution}");
    }
}

/// Diagnostic (`VITASLOP_GXP_VARYING_LAYOUT=<vhash>:<usage>@<lane>x<comps>,...`): plan ONE
/// vertex program against a lane layout typed by hand.
///
/// # Why a knob and not another reading
/// The container comment on `parse_vertex_output_varyings` ends "settling this needs a RENDER
/// ORACLE, not more container reading", and the candidate layouts a permutation can reach are
/// not the whole space: `permute_varyings` carries each usage's WIDTH with it, so a layout that
/// gives `TexCoord(0)` four lanes and `Color1` two is unreachable however the runs are reordered
/// - and that is exactly the shape a vertex whose forwarding moves fill a four-lane run from a
/// texcoord attribute is asking for. This types a layout directly so the frame can judge it.
///
/// Usages are spelled `c0`, `c1`, `fog`, `t0`..`t9`. The vertex hash is the one the
/// `gxp pair`/`gxp link` lines print. Anything unparsable is ignored with a report rather than
/// silently dropped - a diagnostic that quietly does nothing is worse than none.
fn hand_typed_layout(vprog: &Program) -> Option<Vec<OutputVarying>> {
    let spec = std::env::var("VITASLOP_GXP_VARYING_LAYOUT").ok()?;
    // Several programs, separated by `;` - a family shares one defect and has to be judged as a
    // family, not one member at a time.
    let (hash, list) = spec.split(';').find_map(|one| {
        let (h, rest) = one.split_once(':')?;
        (u64::from_str_radix(h.trim().trim_start_matches("0x"), 16).ok()? == vprog.hash)
            .then_some((h, rest))
    })?;
    let _ = hash;
    let mut out = Vec::new();
    for item in list.split(',') {
        let item = item.trim();
        let (usage, rest) = item.split_once('@')?;
        let (lane, comps) = rest.split_once('x')?;
        let usage = match usage.trim() {
            "c0" => VaryingUsage::Color0,
            "c1" => VaryingUsage::Color1,
            "fog" => VaryingUsage::Fog,
            t if t.starts_with('t') => VaryingUsage::TexCoord(t[1..].parse().ok()?),
            other => {
                eprintln!("gxp link: VITASLOP_GXP_VARYING_LAYOUT: unknown usage {other:?}");
                return None;
            }
        };
        out.push(OutputVarying {
            usage,
            base_lane: lane.trim().parse().ok()?,
            components: comps.trim().parse().ok()?,
        });
    }
    let shown: Vec<String> = out
        .iter()
        .map(|v| format!("{:?}@{}..{}", v.usage, v.base_lane, v.base_lane + v.components))
        .collect();
    eprintln!(
        "gxp link: vertex {:016x}: HAND-TYPED lane layout {} (VITASLOP_GXP_VARYING_LAYOUT)",
        vprog.hash,
        shown.join(" ")
    );
    Some(out)
}

/// How many components of `iface` read a vertex output lane the vertex program NEVER WRITES.
///
/// # An unwritten lane is a FACT, and it decides layout questions the containers cannot
/// The vertex output ORDER is not stated by the varyings block ([`VaryingOrder`]), so the
/// linker has always had to choose between readings that the containers alone cannot separate -
/// which is why the notes on `VaryingOrder` ask for a render oracle. But one half of the
/// question needs no oracle at all: a candidate layout that routes a varying the FRAGMENT READS
/// onto a lane the VERTEX CODE never writes is refuted by the vertex program itself. The
/// hardware iterates whatever that lane held; the fragment then shades from a value the title
/// never produced.
///
/// MEASURED on a baseball title's stadium material, which is what this was written for: its
/// vertex program writes lanes 4..10 and 12..20 and leaves 10 and 11 alone, and BOTH readings
/// the linker could reach put a prefetch coordinate there - the convention put the LIGHTMAP's
/// texcoord on the dead pair, and the forwarding resolver put the ALBEDO's. Whichever won, one
/// of the two textures was sampled at a constant (0,0) for every fragment of every stadium
/// surface in the frame, which is a flat, banded, wrong picture that still looks like a
/// stadium.
///
/// This counts, rather than refusing: a layout with a dead read may still be the best available
/// (this title's is - no permutation of its declared widths avoids one), and the count is what
/// says so out loud instead of leaving it to be discovered from a picture.
fn unwritten_lane_reads(iface: &Interface, written: &[bool]) -> usize {
    dead_lane_detail(iface, written).len()
}

/// The dead reads themselves: `(vertex lane, what the fragment does with it)`. A COUNT says a
/// layout is wrong; the LANES say which surface goes wrong and how, which is the difference
/// between "this pair is suspect" and "the tangent frame's third component is uninterpolated".
fn dead_lane_detail(iface: &Interface, written: &[bool]) -> Vec<(u32, String)> {
    iface
        .components
        .iter()
        .filter(|c| !written.get(c.vertex_lane as usize).copied().unwrap_or(false))
        .map(|c| {
            let what = match c.dest {
                ComponentDest::Register(r) => format!("pa[{r}]"),
                ComponentDest::Half { register, slot } => format!("pa[{register}].h{slot}"),
                ComponentDest::SampleCoord { prefetch, coord } => {
                    format!("prefetch#{prefetch}.coord{coord}")
                }
            };
            (c.vertex_lane, what)
        })
        .collect()
}

/// Whether a projective prefetch divides by its `w` - see `container::PrefetchLookup::Projective`.
/// `VITASLOP_GXP_PREFETCH_PROJECTIVE=0` is the arm back to the plain lookup it used to get.
fn projective_prefetch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VITASLOP_GXP_PREFETCH_PROJECTIVE").as_deref() != Ok("0"))
}

/// >>> A PREFETCH COORDINATE THAT LANDS ON LANES THE VERTEX NEVER WRITES IS RE-POINTED TO THE
/// >>> LANES A FORWARDING CLAIM NAMES FOR ITS USAGE.
///
/// The PDS fetches a texture before the fragment program runs, and its coordinate is named by a
/// TEXCOORD index alone. When the declared lane widths put that texcoord on lanes the vertex
/// program demonstrably never writes, the sample is taken at whatever the allocation held -
/// which is one constant texel for the entire draw, and looks like a texture that "works" while
/// carrying none of its content.
///
/// MEASURED on a baseball title's stadium pair (`vert_71eebf0e9836ad22` + `frag_2a3e046eba8baf5e`):
/// `TexCoord(1)` feeds the LIGHTMAP prefetch - a 4096x1024 atlas - and lands on output lanes
/// 10 and 11, which none of that program's 23 instructions writes. Every world surface therefore
/// sampled the atlas at one texel, and the grandstand's seat rows, brick arches and signage were
/// flat bands of colour. The vertex COPIES its `In.UV1` attribute (semantic TEXCOORD/1) into
/// lanes 4..8, so a claim names lane 4 for that very usage, and re-pointing the coordinate there
/// draws the stand.
///
/// # Why this is not a layout change
/// It moves NO varying and changes no declared width: only the two lanes the PDS reads for one
/// sample. That matters because the declared widths of this family cannot be arranged to cover
/// the written lanes at all (see [`report_dead_lane_layout`]) - so there is no candidate ORDER
/// to switch to, and a rule that acts only on the provably dead coordinate cannot disturb a pair
/// whose coordinate is already live.
///
/// The gate is deliberately narrow: the coordinate must be dead, the claim must name the SAME
/// usage, and the claim's own lanes must all be written. `VITASLOP_GXP_PREFETCH_CLAIM=0` is the
/// arm back.
fn repoint_dead_prefetch_coords(
    vprog: &Program,
    vshader: &Shader,
    fprog: &Program,
    iface: &mut Interface,
    written: &[bool],
) -> Vec<String> {
    let mut moved = Vec::new();
    if std::env::var("VITASLOP_GXP_PREFETCH_CLAIM").as_deref() == Ok("0") {
        return moved;
    }
    let live = |lane: u32| written.get(lane as usize).copied().unwrap_or(false);
    // Nothing to do unless a coordinate is actually dead - the common case, and the one this
    // must not touch.
    if !iface.components.iter().any(|c| {
        matches!(c.dest, ComponentDest::SampleCoord { .. }) && !live(c.vertex_lane)
    }) {
        return moved;
    }
    let claims = forwarding_claims(vprog, vshader);
    for pf in iface.prefetches.iter() {
        let dead = pf
            .coords
            .iter()
            .any(|&i| iface.components.get(i).is_some_and(|c| !live(c.vertex_lane)));
        if !dead {
            continue;
        }
        // The usage this unit's coordinate is named by, read back from the fragment's own
        // descriptor list - `PlannedPrefetch` carries the unit, not the texcoord index.
        let Some(usage) = fprog
            .interpolants
            .iter()
            .filter_map(|it| it.prefetch)
            .find(|s| s.unit == pf.unit)
            .map(|s| VaryingUsage::TexCoord(s.source_texcoord))
        else {
            continue;
        };
        // The WIDTH the coordinate spans, which for a projective lookup is four lanes (its `w`
        // is the fourth) even though it routes three.
        let n = if pf.projective { 4 } else { pf.coords.len() as u32 };
        let Some((_, lanes)) = claims
            .iter()
            .find(|(u, lanes)| *u == usage && lanes.len() as u32 >= n && lanes.iter().all(|&l| live(l)))
        else {
            continue;
        };
        let base = lanes[0];
        // The claim's run must be CONTIGUOUS from its first lane for the coordinate's width -
        // a coordinate is two (or three) adjacent lanes, and a claim assembled from scattered
        // writes names no such run.
        if (0..n).any(|c| !lanes.contains(&(base + c))) {
            continue;
        }
        for &i in pf.coords.iter() {
            if let Some(comp) = iface.components.get_mut(i)
                && let ComponentDest::SampleCoord { coord, .. } = comp.dest
            {
                comp.vertex_lane = base + coord;
            }
        }
        moved.push(format!("unit {} coord <- {usage:?}@{base}", pf.unit));
    }
    moved
}

/// Plan the interface, preferring a layout whose fragment reads all land on lanes the vertex
/// actually writes.
///
/// The precedence in [`layout_by_precedence`] is unchanged and still decides first. This only
/// acts when that choice reads a DEAD lane and another candidate does not - a case where the
/// first choice is known wrong, so replacing it cannot cost a pair that is right today.
fn plan_interface(
    vprog: &Program,
    fprog: &Program,
    fshader: &Shader,
    colour_masked_off: bool,
) -> Result<Interface, LinkError> {
    let vshader = crate::usse::decode_shader(vprog);
    let chosen = layout_by_precedence(vprog, fprog, fshader, &vshader, colour_masked_off)?;
    let mut iface = plan_interface_with(fprog, fshader, &chosen, colour_masked_off)?;
    if hand_typed_layout(vprog).is_some() {
        return Ok(iface); // an explicitly typed layout is the answer, not a candidate
    }
    let written = output_written_lanes(&vshader);
    // >>> THE RE-POINT RUNS ON THE WINNER, NEVER ON THE CANDIDATES. Applying it inside the
    // search below let a layout that is only reachable BECAUSE its coordinate was repaired win
    // on dead-lane count, which moved a racer's world family onto `fragment_declared_order` -
    // the reading `layout_by_precedence` records as MEASURED to break exactly that family. So
    // the layout is chosen exactly as it is today and the coordinate is repaired afterwards.
    let repoint = |iface: &mut Interface| {
        let moved = repoint_dead_prefetch_coords(vprog, &vshader, fprog, iface, &written);
        if !moved.is_empty() {
            report_repointed_prefetch(vprog.hash, &moved);
        }
    };
    let detail = dead_lane_detail(&iface, &written);
    let dead = detail.len();
    if dead == 0 {
        return Ok(iface);
    }
    // The other readings. Order is the TIE-BREAK only: a candidate wins here by reading strictly
    // FEWER dead lanes, so a pair that is right today (dead reads on neither reading) is never
    // moved, and the precedence in `layout_by_precedence` still decides everything else.
    let mut alternatives: Vec<(&str, Vec<OutputVarying>)> = Vec::new();
    if let Some(o) = forwarding_cover(vprog, &vshader) {
        alternatives.push(("the vertex's complete forwarding cover", o));
    }
    if let Some(o) = layout_from_forwarding_claims(
        &vprog.output_varyings,
        &forwarding_claims(vprog, &vshader),
        vprog.output_order == VaryingOrder::Known,
    ) {
        alternatives.push(("the vertex's forwarding claims", o));
    }
    alternatives.push(("the container's own order", vprog.output_varyings.clone()));
    if let Some(o) = fragment_declared_order(vprog, fprog) {
        alternatives.push(("the paired fragment's declaration order", o));
    }
    let mut best: Option<(usize, &str, Interface)> = None;
    for (why, order) in &alternatives {
        if *order == chosen {
            continue;
        }
        let Ok(alt) = plan_interface_with(fprog, fshader, order, colour_masked_off) else { continue };
        let n = unwritten_lane_reads(&alt, &written);
        if n < dead && best.as_ref().is_none_or(|(b, _, _)| n < *b) {
            best = Some((n, why, alt));
        }
    }
    match best {
        Some((n, why, mut alt)) => {
            report_dead_lane_layout(vprog.hash, &detail, Some((why, n)));
            repoint(&mut alt);
            Ok(alt)
        }
        None => {
            report_dead_lane_layout(vprog.hash, &detail, None);
            repoint(&mut iface);
            Ok(iface)
        }
    }
}

/// Report - once per vertex program - that a prefetch coordinate was re-pointed off dead lanes.
fn report_repointed_prefetch(hash: u64, moved: &[String]) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(hash) {
        return;
    }
    if std::env::var_os("VITASLOP_LOG").is_none() && std::env::var_os("RUST_LOG").is_none() {
        return;
    }
    eprintln!(
        "gxp link: vertex {hash:016x}: prefetch coordinate(s) landed on vertex output lanes the program never writes - RE-POINTED from the vertex's own forwarding claims: {}",
        moved.join("; ")
    );
}

/// Report - once per vertex program - that its chosen varying layout reads vertex output lanes
/// the program never writes, and whether another reading avoided them.
fn report_dead_lane_layout(hash: u64, detail: &[(u32, String)], replaced_by: Option<(&str, usize)>) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(hash) {
        return;
    }
    // This crate has no tracing subscriber of its own; the same rule as
    // `report_forwarding_contradiction` - print only when a log filter was named.
    if std::env::var_os("VITASLOP_LOG").is_none() && std::env::var_os("RUST_LOG").is_none() {
        return;
    }
    let dead = detail.len();
    let lanes = detail.iter().map(|(l, w)| format!("o[{l}]->{w}")).collect::<Vec<_>>().join(" ");
    match replaced_by {
        Some((why, n)) => eprintln!(
            "gxp link: vertex {hash:016x}: the preferred varying layout routed {dead} fragment read(s) onto vertex output lanes the program never writes [{lanes}] - SWITCHED to {why}, which reads {n}"
        ),
        None => eprintln!(
            "gxp link: vertex {hash:016x}: {dead} fragment read(s) land on vertex output lanes the program NEVER WRITES [{lanes}], and no other reading avoids them - this pair shades from values the title never produced. No arrangement of the declared widths covers the written lanes, so the SET or the WIDTHS are being read wrongly, not just the order"
        ),
    }
}

fn layout_by_precedence(
    vprog: &Program,
    fprog: &Program,
    fshader: &Shader,
    vshader: &Shader,
    colour_masked_off: bool,
) -> Result<Vec<OutputVarying>, LinkError> {
    if let Some(order) = hand_typed_layout(vprog) {
        return Ok(order);
    }
    // A DIAGNOSTIC ARM, not the default: order the vertex's lanes the way the paired FRAGMENT
    // declares them. See `fragment_declared_order` for the two witnesses and for why the
    // default may not be this - a rule that gives one vertex program two lane orders depending
    // on its pairing is not a reading of anything. MEASURED as a default: it fixes a baseball
    // title's menus and BREAKS a racer's world (blocky foliage, striped road, banded terrain),
    // which is what a convention-vs-convention flip does. The vertex's own forwarding claims
    // settle both, and that is where the fix went - see `exhaustive_layout_from_claims`.
    if std::env::var("VITASLOP_GXP_VARYING_ORDER").as_deref() == Ok("fragment")
        && let Some(order) = fragment_declared_order(vprog, fprog) {
            return Ok(order);
        }
    // >>> ASK THE VERTEX PROGRAM'S OWN FORWARDING MOVES FIRST, WHATEVER PLACED THE REST. See
    // `forwarding_claims` for why a move can answer the order question and the varyings block
    // cannot.
    //
    // >>> THIS RUNS FOR A `Known` ORDER TOO, and that is a correction. `Known` means
    // `container::attribute_order` found an attribute carrying each declared varying's semantic
    // and took the attributes' `resource_index` order as the output order - which is a reading
    // of a PASSTHROUGH program, whose outputs are its inputs. A program that COMPUTES a varying
    // it also has an attribute for satisfies that cover without being a passthrough at all, and
    // then the inference is simply wrong. MEASURED on a retail title's UI pair
    // (`vert_81a7e8b8` + `frag_81a809b4`): the attributes are `aPosition`@0, `aTexCoord`@4,
    // `aColor`@8, so `attribute_order` places `TexCoord(0)@4x2 Color0@6x4` - while the code
    // moves `aColor` straight into lanes 4..8 and writes the texcoord it DIVIDES BY `uTexture`
    // into lanes 8..10. The fragment's prefetch then sampled the logo texture at the vertex
    // COLOUR (a constant 1,1), so every sprite came out one flat texel and the title screen
    // drew solid silhouettes.
    //
    // Consulting the moves here cannot disturb a program the attribute reading gets right: a
    // real passthrough forwards every attribute, so its claims AGREE with that order,
    // `layout_from_forwarding_claims` walks back onto the same layout and returns `None`. The
    // resolver is its own gate - it acts only where a claim names a usage the layout does not
    // put at that lane - which is why the contradiction report below is no longer what admits it.
    // >>> A COMPLETE READING FIRST. When the code forwards every declared varying and the runs
    // tile the budget exactly, that IS the layout - see [`forwarding_cover`], and see the golf
    // regression recorded there for why the weaker one-claim reading below must not be
    // strengthened into this.
    if std::env::var("VITASLOP_GXP_VARYING_RESOLVE").as_deref() != Ok("0")
        && let Some(order) = forwarding_cover(vprog, vshader) {
            let shown: Vec<String> = order
                .iter()
                .map(|v| format!("{:?}@{}..{}", v.usage, v.base_lane, v.base_lane + v.components))
                .collect();
            report_forwarding_contradiction(
                vprog.hash,
                "the code forwards EVERY declared varying and the runs tile the budget",
                &format!("COVERED by the vertex alone -> {}", shown.join(" ")),
            );
            return Ok(order);
        }
    let claims = forwarding_claims(vprog, vshader);
    // Value-sensitive, because a knob used as an A/B ARM has to be: a presence-only reader
    // turns `=0` into an ON arm and both arms then measure the same build.
    if std::env::var("VITASLOP_GXP_VARYING_RESOLVE").as_deref() != Ok("0")
        && let Some(order) = layout_from_forwarding_claims(
            &vprog.output_varyings,
            &claims,
            vprog.output_order == VaryingOrder::Known,
        ) {
            let shown: Vec<String> = order
                .iter()
                .map(|v| format!("{:?}@{}..{}", v.usage, v.base_lane, v.base_lane + v.components))
                .collect();
            report_forwarding_contradiction(
                vprog.hash,
                &forwarding_contradicts(&vprog.output_varyings, &claims).unwrap_or_else(|| {
                    "a forwarded attribute starts at a lane this layout gives another varying"
                        .to_string()
                }),
                &format!("RESOLVED from the vertex alone -> {}", shown.join(" ")),
            );
            return Ok(order);
        }
    // >>> WHAT EACH READING SAYS, for every pair that gets this far. A layout question is
    // settled by comparing the readings, and until now that comparison needed a rebuild with a
    // hand-typed layout: nothing printed the candidates.
    if let Some(order) = fragment_declared_order(vprog, fprog) {
        let show = |vs: &[OutputVarying]| {
            vs.iter()
                .map(|v| format!("{:?}@{}x{}", v.usage, v.base_lane, v.components))
                .collect::<Vec<_>>()
                .join(" ")
        };
        if order != vprog.output_varyings {
            report_forwarding_contradiction(
                vprog.hash,
                &format!("layout by {:?}: {}", vprog.output_order, show(&vprog.output_varyings)),
                &format!("the paired FRAGMENT declares: {}", show(&order)),
            );
        }
    }
    // A `Known` order was read off the attributes and the moves did not refute it.
    if vprog.output_order == VaryingOrder::Known {
        return Ok(vprog.output_varyings.clone());
    }
    // >>> EVERYTHING BELOW IS A LAYOUT THE CONVENTION PLACED.
    if let Some(why) = forwarding_contradicts(&vprog.output_varyings, &claims) {
        // >>> IT NOW ACTS, FROM THE VERTEX ALONE - see `layout_from_forwarding_claims`. The
        // history below is why it took three attempts, and every objection in it still stands
        // against the reading it refused; none of them applies to this one, which never consults
        // the fragment and never changes a usage's width.
        //
        // 2026-08-19b resolved a contradiction by switching that pair to
        // `fragment_declared_order`. It made that billboard paint its banner, and
        // it is still backed out, because a rule that gives ONE vertex program two different
        // lane orders depending on its pairing is not a reading of anything. (The lurid green
        // sky first blamed on this change was MEASURED not to be it - that title's world
        // frame is byte-identical with this resolution backed out - but the two structural
        // faults below stand on their own and the resolution goes anyway.)
        //
        // Two things are wrong with resolving it that way, and both are structural:
        //
        // 1. **It makes a vertex program's LANE ORDER depend on which fragment it is paired
        //    with.** The order is a property of the vertex program alone - it is baked into the
        //    code that writes those lanes - so a rule that gives one program two orders is not a
        //    reading of anything. MEASURED: of five vertex programs of one title with IDENTICAL
        //    contradiction evidence, two switched and three did not, decided by nothing but
        //    which fragments they happened to be paired with.
        // 2. **A claim's WIDTH is the container's declared component count, not the bound one.**
        //    `In.UV1` declares 4 components and the guest binds it `F16x2`, so PA[6] and PA[7]
        //    carry the (0,1) missing-component fill and no data - yet the claim spans four lanes
        //    and matches a four-lane run it has no business matching. The linker cannot see the
        //    bound layout; that is runtime state.
        //
        // So the witness stays, because what it FINDS is real and is the only statement about
        // lane order that is not itself a convention, and the convention stands until a reading
        // exists that resolves it from the VERTEX alone. Do not reconnect this to the fragment.
        //
        // Reaching HERE means the resolver above already declined: the contradiction is real but
        // no walk of the declared widths puts the forwarded attribute's usage at the lane it
        // fills, so there is nothing to act on and the convention stands.
        report_forwarding_contradiction(
            vprog.hash,
            &why,
            "the convention stands - no permutation puts the forwarded attribute's usage at \
             the lane it fills, so this pair's varyings may be routed wrongly",
        );
    }
    // A program whose order the container could read is planned directly against it - this
    // is every program that links today, and its behaviour is untouched.
    if vprog.output_order != VaryingOrder::Ambiguous {
        return Ok(vprog.output_varyings.clone());
    }
    // >>> ASK THE VERTEX PROGRAM'S OWN CODE before refusing. `Ambiguous` means "the
    // convention placed a COLOR1 and no attribute confirms it" - which is a statement about
    // the CONTAINERS, and the containers are not the only witness. If the code's writes
    // agree with the convention's layout, the layout is read rather than assumed.
    if convention_agrees_with_the_code(&vprog.output_varyings, vshader) {
        return Ok(vprog.output_varyings.clone());
    }
    resolve_ambiguous_order(vprog, fprog, fshader, colour_masked_off)
}

/// [`plan_interface`] against an EXPLICIT vertex lane layout, so the permutation search can
/// try one without mutating the program.
fn plan_interface_with(
    fprog: &Program,
    fshader: &Shader,
    vout: &[OutputVarying],
    // See [`LinkOptions::colour_output_masked_off`]. It reaches only the passthrough
    // treatment below, and it has to be threaded through the permutation search as well: a
    // candidate order must be judged under the same rule the winner will be linked under, or
    // the search answers a different question than the link does.
    colour_masked_off: bool,
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
    // >>> A PROGRAM THAT DECLARES NO INTERPOLANTS AT ALL IS NOT A PASSTHROUGH, AND THAT IS A
    // >>> STATEMENT THE SHADER MAKES BY ITSELF.
    //
    // The passthrough reading rests on the PDS having LOADED something into the primary
    // attribute registers before the program ran - "its result is whatever the PDS left". What
    // the PDS loads is precisely this program's DECLARED INTERPOLANTS. A program that declares
    // none has nothing loaded, so those registers are undefined ON THE HARDWARE TOO, and
    // demanding that every one of them be fed asks for a value that does not exist.
    //
    // MEASURED: a football title's `frag_90c054e0` is `Nop`, a `Test` on two constants, a
    // predicated `Kill`, `Nop`, `Nop` - an interpolant count of 0 that is TRUE, and no write to
    // any colour register. It was refused as "reads PA register 0 before writing it" and its
    // draws were DROPPED (168 fallback draws in one browser run of the kickoff recipe). A real
    // passthrough is the opposite shape: a racing title's flat vertex-coloured polygon is a
    // single `Nop` that DECLARES a Color0, and that declaration is exactly what makes its
    // register file meaningful. So the two are distinguishable after all.
    //
    // >>> A FAILED DECODE IS NOT AN EMPTY DECLARATION. An interpolant list that is empty
    // because the varyings block did not PARSE must keep the old, conservative treatment -
    // otherwise a decode gap would quietly start emitting zero-initialised colour, which is the
    // silent-approximation failure this whole path exists to avoid. `varyings_error` separates
    // the two, and it is the same field `PaReadUnfed` carries for the same reason.
    let declares_no_interpolants = fprog.interpolants.is_empty() && fprog.varyings_error.is_none();
    // ...and a draw whose colour is MASKED OFF is not a passthrough either, whatever the
    // shader looks like - see [`LinkOptions::colour_output_masked_off`]. That one cannot be
    // read from the shader: it is the guest's own per-draw blend state.
    let passthrough = is_passthrough(fshader, &inputs) && !colour_masked_off && !declares_no_interpolants;
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
            // The evidence is a real draw plus a count. One title's tutorial pairs a vertex
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
        let cube = sampler.sampler_cube;
        // >>> A PROJECTIVE PREFETCH READS THE TEXCOORD'S `w` TOO - see
        // `container::PrefetchLookup::Projective`. Never on a cube sampler: every projective
        // descriptor in every corpus names a flat one, and a projective cube is a shape nothing
        // here has seen.
        // `VITASLOP_GXP_PREFETCH_PROJECTIVE=0` is the arm back (the plain lookup).
        let projective = !cube
            && pf.lookup == crate::container::PrefetchLookup::Projective
            && projective_prefetch_enabled();
        let coords = if cube { 3 } else { 2 };
        // >>> A COORDINATE THE VERTEX NEVER PRODUCES IS THE DEFAULT, NOT A REFUSAL. See
        // `PlannedPrefetch::coords_unfed` for the corpus closure behind that.
        let source = match vertex_output(usage) {
            Ok(v) => Some(v),
            Err(e) if prefetch_unfed_default() => {
                report_prefetch_coord_unfed(pf.unit, pf.source_texcoord);
                let _ = e;
                None
            }
            Err(e) => return Err(e),
        };
        if let Some(source) = source
            && source.components < coords
        {
            return Err(LinkError::PrefetchCoordTooNarrow {
                unit: pf.unit,
                needed: coords,
                available: source.components,
            });
        }
        let prefetch = iface.prefetches.len();
        let first = iface.components.len();
        // The divisor is the texcoord's FOURTH component. A vertex that writes fewer leaves the
        // iterator's texture-coordinate default there, `w = 1` (see `Interface::defaults`), and
        // dividing by one is the plain lookup - so only a four-component source routes a lane.
        let divisor = projective && source.is_some_and(|v| v.components >= 4);
        if let Some(source) = source {
            for c in 0..coords {
                iface.components.push(VaryingComponent {
                    vertex_lane: source.base_lane + c,
                    dest: ComponentDest::SampleCoord { prefetch, coord: c },
                });
            }
            if divisor {
                iface.components.push(VaryingComponent {
                    vertex_lane: source.base_lane + 3,
                    dest: ComponentDest::SampleCoord { prefetch, coord: 3 },
                });
            }
        }
        let routed = coords as usize + usize::from(divisor);
        iface.prefetches.push(PlannedPrefetch {
            unit: pf.unit,
            pa_base,
            regs: prefetch_regs,
            coords: match source {
                Some(_) => (first..first + routed).collect(),
                None => Vec::new(),
            },
            cube,
            projective: divisor,
            coords_unfed: source.is_none(),
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
    if passthrough
        && let Some(reg) = (0..primary_regs).find(|&r| !fed.get(r as usize).copied().unwrap_or(false)) {
            return Err(LinkError::PaReadUnfed { register: reg, varyings_error: fprog.varyings_error });
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
/// The `VITASLOP_GXP_SA_UNCLAIMED` arm's value, as `(text, f32 bits)`, or `None` when it is off.
///
/// It takes a NUMBER (`zero`, `one`, or any float literal) rather than being a bare on/off,
/// because the first question to ask of a slot nobody has identified is whether its value
/// MATTERS: two runs at different constants that produce the same picture say the draws do not
/// depend on it, and two that differ say which readings are still live. An on/off arm can only
/// ever answer "does binding something help", which is the weaker half.
fn unclaimed_arm_value() -> Option<(&'static str, u32)> {
    let v = arm("VITASLOP_GXP_SA_UNCLAIMED")?;
    let f = match v {
        "zero" => 0.0f32,
        "one" => 1.0f32,
        other => other.parse::<f32>().ok()?,
    };
    Some((v, f.to_bits()))
}

/// Report an uninitialised-scratch SA read, once per (register, extent) pair.
///
/// Deduped because it is emitted while EMITTING a shader, and a title binds the same pair
/// hundreds of times a frame; undeduped it is the log.
fn report_unclaimed_scratch(reg: u32, declared_top: u32) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::BTreeSet<(u32, u32)>>> =
        std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert((reg, declared_top)) {
        eprintln!(
            "gxp link: sa[{reg}] is read but never written and lies ABOVE this program's whole declared layout (which ends at sa[{declared_top}]) - no container, literal or texture word places anything there, so on hardware it holds whatever the previous draw left and the guest cannot depend on it. Binding it as 0.0 and emitting the pair rather than dropping the draw."
        );
    }
}

/// The SA-register initialiser list one program is linked with: its container literals plus
/// whatever an unclaimed-register arm binds.
///
/// Public because this is what a DECODER-INDEPENDENT change to `secondary_attr_init` moves, and
/// a per-blob WGSL hash cannot see it - the initialiser is built during LINKING, so a blob
/// recompiled on its own emits none of it. It takes a program's bytes alone because the list
/// depends only on that program and its own two streams, never on the stage it is paired with,
/// which is what makes a blob-level census of it exact rather than a sample.
pub fn sa_literal_init(program: &Program) -> Result<Vec<(u32, u32)>, LinkError> {
    let shader = crate::usse::decode_shader(program);
    secondary_attr_init(&shader, program)
}

/// Is a container literal laid down for EVERY read that names its register
/// (`VITASLOP_GXP_SA_LITERAL_ALWAYS=0` is the arm back to suppressing it wherever the secondary
/// program writes the register at all)?
///
/// An A/B ARM rather than a mode, so both readings are reachable from ONE build - the arms must
/// differ in one thing or the comparison is between two binaries
/// ([[vitaslop-browser-ab-needs-a-negative-control]]).
fn sa_literal_always() -> bool {
    std::env::var("VITASLOP_GXP_SA_LITERAL_ALWAYS").as_deref() != Ok("0")
}

pub(crate) fn secondary_attr_init(
    shader: &Shader,
    program: &Program,
) -> Result<Vec<(u32, u32)>, LinkError> {
    // Every SA register the BINDING carries: the default uniform buffer plus any non-default
    // uniform buffer the driver copies into the register file. Reading `default_uniform_regs`
    // alone refused a program whose whole uniform block is a bound container-0 buffer, with a
    // message naming its "0-register default uniform buffer" - which was true and not the point.
    let uniform_regs = program.sa_carried_extent();
    let secondary = crate::usse::decode_secondary_shader(program);

    // The SA registers one instruction READS, addressed exactly as the emitter addresses them.
    let sa_sources = |instr: &Instr| -> Vec<u32> {
        let read = read_channels(instr);
        let mut out = Vec::new();
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
            if matches!(instr.op, Op::Tex { .. } | Op::TexGather { .. }) && i == 1 {
                continue;
            }
            for c in 0..4 {
                if !read[c] {
                    continue;
                }
                let Some((reg, _)) = instr.source_register(src, c) else {
                    continue;
                };
                out.push(reg);
            }
        }
        out
    };
    // The SA registers one instruction WRITES, by the same addressing.
    let sa_dests = |instr: &Instr| -> Vec<u32> {
        let mut out = Vec::new();
        let Some(d) = instr.dest.as_ref() else { return out };
        if d.bank != Bank::SecondaryAttr {
            return out;
        }
        // A memory load is not a four-lane write: it fills `elements` CONSECUTIVE registers
        // from `dest.index`, which is how the emitter writes it (`emit_mem_load` stores
        // `dest.index + k` for k in 0..elements). Counting it as four left the rest of the
        // span looking unwritten, so a program that loads a 16-register matrix and then reads
        // its second row was refused for reading an SA register "outside its uniform buffer" -
        // a register its own prologue had just filled.
        if let crate::ir::Op::MemLoad { elements, .. } = instr.op {
            out.extend((0..u32::from(elements)).map(|k| d.index as u32 + k));
            return out;
        }
        for c in 0..4u32 {
            if instr.write_mask[c as usize] {
                out.push(d.index as u32 + if instr.half_precision { c >> 1 } else { c });
            }
        }
        out
    };

    // Registers the secondary program computes. For a PRIMARY read these are legitimate sources
    // even above the uniform buffer - that is the point of a secondary program - so they must
    // not be mistaken for reads of the texture-control region.
    //
    // >>> A SECONDARY read is a different question, and conflating the two blacked out a whole
    // title's UI. The secondary program is a straight-line prologue, so a register it writes is
    // only "computed" for a reader that comes AFTER that write; its own earlier reads still see
    // the container literal. One title's UI vertex program declares `sa[8] = 0.5h`, multiplies
    // the screen extent by it, and only THEN reuses `sa[8]` as the scratch holding the NDC bias.
    // Suppressing that literal made the multiply read zero, the reciprocal that follows it
    // infinite and every clip position NaN - so no UI draw in the title covered a pixel, with
    // nothing refused and nothing in any log. Walk the stream in order instead.
    let mut written = std::collections::BTreeSet::new();
    // Reads that MUST be backed by the uniform buffer or a container literal: a secondary read
    // that no earlier secondary write has covered.
    let mut needed_strict = std::collections::BTreeSet::new();
    for instr in &secondary.instrs {
        for reg in sa_sources(instr) {
            if !written.contains(&reg) {
                needed_strict.insert(reg);
            }
        }
        written.extend(sa_dests(instr));
    }
    // Reads the secondary program may satisfy, wherever in it the write happens.
    let mut needed = std::collections::BTreeSet::new();
    for instr in &shader.instrs {
        needed.extend(sa_sources(instr));
    }
    // The mem-window base register is DRIVER data, not texture state: the module initialises
    // it from the bound window's own header (see `MemWindow::base_sa`), so a read of it is fed.
    let mem_base_sa: Vec<u32> = crate::module::resolve_mem_windows(program, shader)
        .unwrap_or_default()
        .iter()
        .map(|w| w.base_sa)
        .collect();
    // The top of everything the program's own tables DECLARE: the SA-carried uniform extent,
    // the DATA container, and every literal and texture-control word. A register above all of
    // that is not driver data under any reading - see the `UninitialisedScratch` arm below.
    let declared_top = {
        let data_top = program
            .containers
            .iter()
            .map(|c| u32::from(c.base_sa) + u32::from(c.size_regs))
            .max()
            .unwrap_or(0);
        let lit_top = program.literals.iter().map(|&(r, _)| r + 1).max().unwrap_or(0);
        let tex_top = program.texture_control.iter().map(|&(b, _)| b + 4).max().unwrap_or(0);
        uniform_regs.max(data_top).max(lit_top).max(tex_top)
    };
    let mut literals = Vec::new();
    for reg in needed.union(&needed_strict).copied() {
        if reg < uniform_regs || mem_base_sa.contains(&reg) {
            continue;
        }
        // >>> A CONTAINER LITERAL IS OWED WHENEVER A READ NAMES THE REGISTER, EVEN ONE THE
        // >>> SECONDARY PROGRAM ALSO WRITES. The driver lays the literal table into the SA file
        // BEFORE the secondary runs - which is the order the emitter uses - so the secondary's
        // own write simply lands on top of it. Treating "the secondary writes it" as "the
        // secondary supplies it" is only true when the write covers the WHOLE register, and a
        // half-precision write covers ONE OF TWO LANES: `sa_dests` maps channel `c` to register
        // `index + (c >> 1)`, so a mask of channel 1 alone claims the register while leaving
        // its low lane untouched.
        //
        // MEASURED, and it is the whole of one title's black batter: its fragment secondary
        // packs a value into `sa[30]`'s HIGH lane only, and the LOW lane is the literal
        // `1.0h`. Suppressed, `pa[18].x = -shadow + sa[30].x` became `-0 + 0 = 0`, and 50
        // instructions later the material multiplies its entire lit colour by that lane - so
        // the draw covered 7,909 pixels at 100.0% pure black. Nothing refused and nothing
        // reported: the pair links, the pipeline builds, the pass encodes it.
        //
        // Emitting a literal for a register the secondary DOES fully overwrite costs one dead
        // store, which is the right price for never suppressing a live one.
        if let Some(&(r, v)) = program.literals.iter().find(|(r, _)| *r == reg)
            && (sa_literal_always() || !written.contains(&reg) || needed_strict.contains(&reg))
        {
            literals.push((r, v));
            continue;
        }
        let computed = written.contains(&reg) && !needed_strict.contains(&reg);
        if computed {
            continue;
        }
        {
            // DIAGNOSTIC ARM, off by default and never a default:
            // `VITASLOP_GXP_SA_UNCLAIMED=zero` binds an unclaimed register as 0.0 instead of
            // refusing the pair.
            //
            // It exists to SEPARATE TWO QUESTIONS that a dropped draw answers together. One
            // title's whole in-game world is 18 pairs refused here, and until the draws reach
            // the rasteriser nothing says whether these registers are the reason the world is
            // black or merely the reason it is absent - a pass that also has a wrong transform
            // would still paint nothing with them bound. Zero is not a reading of the slot and
            // is not claimed to be one; it is the perturbation that makes the pass draw, so the
            // next question can be asked of a picture.
            //
            // The unclaimed registers of this title are the DATA container's first two, which
            // no +0x78 entry and no literal names, in every program of four titles' corpora.
            // What the driver puts there is NOT established, so the default stays a refusal
            // [[vitaslop-recompile-or-fail-default]] and the arm SAYS what it did.
            // >>> AN UNWRITTEN REGISTER ABOVE THE WHOLE DECLARED LAYOUT IS UNINITIALISED
            // >>> SCRATCH, AND ZERO IS FAITHFUL TO IT.
            //
            // This is NOT the arm below and not a guess at a driver value. The registers it
            // covers are above the SA-carried uniforms, above the DATA container, and above
            // every literal and texture-control word - so no table places anything there and
            // the driver writes nothing there. On the hardware they hold whatever the PREVIOUS
            // draw left in the register file, which is a value the guest cannot know and
            // therefore cannot depend on: a program reading one is reading a value its own
            // compiler never defined, and whatever it computes from it must be dead.
            //
            // MEASURED, on the program this was found on: one title's world/shadow pair reads
            // sa[63] as the second lane of a two-lane add whose first lane is sa[62] - which
            // its prologue DOES write. Binding sa[63] to 0.0, 1.0 and 0.5 in turn leaves the
            // draw's clip bounding box BIT-IDENTICAL (x[7.78,8.70] y[-1.76,-1.73] every time),
            // which is the direct statement that nothing downstream depends on it.
            //
            // The alternative - refusing - drops the draw, and a dropped draw is definitely
            // wrong where an unread value is definitely harmless. It REPORTS, so a picture that
            // does turn out to depend on one is traceable [[vitaslop-fallback-must-report]].
            //
            // >>> IT COVERS THE DATA CONTAINER'S UNNAMED SLOTS TOO, and the reason is a
            // MEASUREMENT that retired the reading those slots used to have. Across four
            // titles no +0x78 entry, literal or texture word ever lands on DATA slot 0 or 1,
            // and they were read as RESERVED DRIVER HEAD WORDS. They are not:
            //  * ONE TITLE'S WORLD PROGRAM WRITES BOTH OF THEM AS SCRATCH - its prologue moves
            //    `UVP_ShadowParameters.x/.y` into sa[2]/sa[1]. A compiler does not clobber a
            //    driver word it might need, so they are ordinary registers.
            //  * AND THE VALUE THAT BELONGS THERE IS ZERO, read out of the guest's own bound
            //    buffer rather than assumed: a sibling SKINNED program reads sa[1] without
            //    writing it, in the idiom `r[1] = max(worldY, sa[1])` - a ground-plane clamp -
            //    and that draw's `UVP_ShadowParameters` window holds
            //    (-0.20008333, 0, 0.4966981, 0.012345679). Its `.y` is 0.0, which is exactly
            //    what the world program puts in sa[1], and 0.0 is what a clamp to the ground
            //    plane wants. MEASURED on the capsule: at 0.0 the mesh covers pixels, at 1.0 it
            //    covers none.
            // A register that IS written but read earlier in the secondary than the write that
            // covers it is a different thing - an ordering problem - and still refuses.
            if !written.contains(&reg)
                && !program.texture_control.iter().any(|&(b, _)| (b..b + 4).contains(&reg))
            {
                report_unclaimed_scratch(reg, declared_top);
                literals.push((reg, 0));
            } else if let Some((text, bits)) = unclaimed_arm_value() {
                eprintln!(
                    "gxp link: sa[{reg}] has no declared home and VITASLOP_GXP_SA_UNCLAIMED={text} is BINDING IT AS that constant - this is a diagnostic arm, not a reading of the slot, and any picture it produces is unverified"
                );
                literals.push((reg, bits));
            } else {
                let windows: Vec<String> =
                    crate::module::resolve_mem_windows(program, shader).unwrap_or_default().iter()
                        .map(|w| format!("buf{}@sa[{}]x{}B", w.buffer_index, w.base_sa, w.bytes))
                        .collect();
                let lits: Vec<String> =
                    program.literals.iter().map(|(r, _)| format!("sa[{r}]")).collect();
                let provenance = format!(
                    "it is {} by the secondary program, and the read is {}; SA-carried extent sa[0..{uniform_regs}), container literals at [{}], memory windows [{}]",
                    if written.contains(&reg) { "WRITTEN" } else { "never written" },
                    if needed_strict.contains(&reg) {
                        "IN THE SECONDARY itself, ahead of any write that covers it"
                    } else {
                        "in the primary"
                    },
                    lits.join(" "),
                    windows.join(" "),
                );
                return Err(LinkError::SecondaryAttrOutOfRange { register: reg, uniform_regs, provenance });
            }
        }
    }
    // >>> AN INDEXED READ NAMES NO REGISTER, so the walk above cannot see the literals it
    // >>> reaches - and dropping one leaves a compile-time CONSTANT reading zero.
    //
    // `sa[idx[0] + 14]` is a register the shader picks at runtime; `sa_sources` only ever
    // reports the ones an operand spells out, so every literal only an indexed read can reach
    // fell out of this list. MEASURED on a particle vertex program: its four float2 corner
    // offsets are container literals at `sa[50..57]` - `(0,0) (0,1) (1,1) (1,0)`, the corners
    // of a unit quad - reached as `sa[cornerIndex*2 + 36 + 14]`. Only `sa[35..49]` were
    // emitted, so all four corners read (0,0), every particle quad collapsed to a point, and
    // the title's sparks, smoke and fire drew nothing at all. Nothing reported it: the draw is
    // prepared, the pipeline builds, the pass encodes it and it covers no pixels.
    //
    // The register cannot be narrowed - that is what "indexed" means - so every literal the
    // container carries goes in. They are constants out of the blob, so emitting one the
    // shader never reads costs a `let` the compiler drops; omitting one is a wrong picture.
    // The two exclusions above still apply: a literal inside the uniform buffer's own extent
    // would overwrite what the guest bound, and one naming a memory-window base would
    // overwrite the pointer the module puts there.
    let reads_indexed_sa = shader
        .instrs
        .iter()
        .chain(secondary.instrs.iter())
        .flat_map(|i| i.srcs.iter())
        .any(|s| s.bank == Bank::Indexed && crate::ir::indexed_sub_bank(s.index) == Bank::SecondaryAttr);
    if reads_indexed_sa {
        for &(r, v) in program.literals.iter() {
            if r >= uniform_regs && !mem_base_sa.contains(&r) {
                literals.push((r, v));
            }
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
///
/// `vp.x` is the guest viewport's VERTICAL SENSE, +1 or -1. GXM maps ndc y to the framebuffer
/// as `screen = yOffset + yScale * ndc`, and a pass that sets `yScale > 0` therefore puts ndc
/// `+1` at the BOTTOM of its rectangle. A wgpu viewport requires a positive height and cannot
/// express that, so the flip is done here instead - which is the only place it can be done at
/// all, and it has to be per DRAW because the guest sets a viewport per draw.
pub(crate) const GXP_DEPTH_DECL: &str = r#"struct GxpDepth { range: vec4<f32>, fit: vec4<f32>, vp: vec4<f32> };
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
  // The guest's OWN viewport depth mapping was applied in the vertex stage, so a depth the
  // shader computes - which is already in the guest's window encoding - needs only the clamp
  // that mapping carries. Checked first because it is the default forward map.
  if (gxp_depth.range.w >= 2.5) { return clamp(d, 0.0, 1.0); }
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
    if fplan.dual_source {
        // `@blend_src` needs the extension enabled, and an `enable` must come first.
        m.push_str("enable dual_source_blending;\n");
    }

    // ---- Pipeline-supplied depth state at group 3 ----
    // Declared by the LINKER rather than injected by the renderer, because both stages depend
    // on it and a module that mentions it has to be independently compilable (the oracle
    // naga-validates linked modules with no renderer in the picture). The renderer fills it:
    //   x = depth_min, y = depth_scale  - the affine remap `gxp_clipfix` puts clip depth through
    //   z = the clip-`w` sign correction that same fixup applied (1 none, -1 negate, 2 flip w)
    //   w = which value the guest's own depth buffer holds (see `gxp_guest_depth`)
    m.push_str(GXP_DEPTH_DECL);
    // The destination-colour texture rides in the same group - see `module::GXP_DEST_DECL`.
    if fplan.reads_dest_color && !fplan.dual_source {
        m.push_str(crate::module::GXP_DEST_DECL);
    }

    // ---- Vertex default-uniform buffer (SA bank) at group 0 ----
    // The buffer is the guest's raw default-uniform-buffer bytes: a run of 32-bit registers,
    // NOT an array of floats, because a register may hold two packed F16 halves. It is bound
    // as `vec4<u32>` and copied verbatim into the register file.
    let vsa_regs = vprog.sa_carried_extent();
    let vsa_vec4 = vsa_regs.div_ceil(4);
    if vsa_vec4 > 0 {
        let _ = writeln!(m, "struct VsSa {{ data: array<vec4<u32>, {vsa_vec4}> }};");
        let _ = writeln!(m, "@group(0) @binding(0) var<uniform> vs_sa: VsSa;");
    }

    // ---- The vertex stage's guest-memory window, beside its uniform in group 0 ----
    // Only present when the program's 0xE8 loads resolved to one (see `MemWindow`): vec4 0
    // lane x is the window's own guest base address, the window's words follow.
    if !vplan.mem_windows.is_empty() {
        let _ = writeln!(
            m,
            "@group(0) @binding(1) var<uniform> gxp_mem: array<vec4<u32>, {}>;",
            crate::module::mem_window_vec4_count(&vplan.mem_windows)
        );
        m.push_str(&crate::module::mem_window_helper(&vplan.mem_windows));
    }

    // ---- Fragment default-uniform buffer (SA bank) at group 1 ----
    let fsa_regs = fprog.sa_carried_extent();
    let fsa_vec4 = fsa_regs.div_ceil(4);
    if fsa_vec4 > 0 {
        let _ = writeln!(m, "struct FsSa {{ data: array<vec4<u32>, {fsa_vec4}> }};");
        let _ = writeln!(m, "@group(1) @binding(0) var<uniform> fs_sa: FsSa;");
    }

    // ---- The fragment stage's guest-memory window, beside its uniform in group 1 ----
    // The exact mirror of the vertex one in group 0, with its own name because WGSL has a
    // single global namespace and the two are different buffers.
    if !fplan.mem_windows.is_empty() {
        let _ = writeln!(
            m,
            "@group(1) @binding(1) var<uniform> gxp_fmem: array<vec4<u32>, {}>;",
            crate::module::mem_window_vec4_count(&fplan.mem_windows)
        );
        m.push_str(&crate::module::mem_window_helper_named(&fplan.mem_windows, "gxp_fmem"));
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
            // The TYPE follows the fetch: an integer-fetched attribute is `vec4<u32>` or
            // `vec4<i32>`, because WebGPU requires a vertex input's base type to match the
            // format bound to it. See `VertexAttribute::int_fetch`.
            let ty = crate::module::attribute_wgsl_type(a);
            let _ = writeln!(m, "  @location({}) a{}: {ty},", a.location, a.location);
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
    // The position probe draws its OWN triangle (see [`POSPROBE_ARM`]) and needs the vertex
    // index to place the three corners; nothing else in this module reads it.
    // OFF unless asked for: `arm_on` DEFAULTS TO TRUE (it is for default-on arms), so gating
    // this on it armed the probe in every ordinary render - which replaces the clip position of
    // every vertex in the frame. A diagnostic that is on by default is not a diagnostic.
    let posprobe = posprobe_on();
    let vid = if posprobe { ", @builtin(vertex_index) gxp_vid: u32" } else { "" };
    let sig = vid.trim_start_matches(", ");
    if has_inputs {
        m.push_str(&crate::module::probe_globals());
        let _ = writeln!(m, "\n@vertex\nfn vs_main(in: VsIn{vid}) -> VsOut {{");
    } else {
        let _ = writeln!(m, "\n@vertex\nfn vs_main({sig}) -> VsOut {{");
    }
    emit_register_banks(&mut m);
    // Load PA registers from the vertex attributes (vertex inputs are plain f32 components).
    for a in &vplan.attributes {
        crate::module::emit_attribute_load(&mut m, a);
    }
    emit_secondary_attrs(&mut m, "vs_sa", vsa_regs, vliterals);
    // The driver-placed pointer register: the bound buffer's guest address, exactly as the
    // hardware's PDS would leave it, so the body's address arithmetic runs bit-exact. After
    // the SA-init marker and the literals, so `resolve_sa_init` sees it as a WRITE and keeps
    // the register a compacted local slot.
    for (i, w) in vplan.mem_windows.iter().enumerate() {
        let _ = writeln!(m, "  sa[{}] = gxp_mem[{i}u].x;", w.base_sa);
    }
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
    // See [`POSPROBE_ARM`]. After the varyings, because it OVERWRITES `v0`; skipped when the
    // pair declares no varying to carry the answer.
    if varying_locations > 0 && posprobe {
        let _ = writeln!(m, "  {{");
        let _ = writeln!(m, "    let gxp_pp = out.position;");
        let _ = writeln!(m, "    let gxp_pw = max(abs(gxp_pp.w), 1e-9);");
        // The divisor is baked as a LITERAL rather than read through a uniform: the emitted
        // body is hashed and cached, so a probe run and an ordinary run must not be able to
        // produce the same module with different behaviour.
        let _ = writeln!(m, "    let gxp_nd = (gxp_pp.xy / gxp_pw) / {:?};", posprobe_divisor());
        // A COLLAPSED MESH CANNOT DRAW ITSELF: clamping a degenerate triangle leaves it
        // degenerate and a NaN survives every clamp. So the probe rasterises a triangle of
        // its OWN - one full-screen triangle per primitive - and carries the answer in `v0`.
        let _ = writeln!(m, "    let gxp_c = gxp_vid % 3u;");
        let _ = writeln!(m, "    out.position = vec4<f32>(select(-1.0, 3.0, gxp_c == 1u), select(-1.0, 3.0, gxp_c == 2u), 0.5, 1.0);");
        let _ = writeln!(m, "    let gxp_ok = gxp_pp.x == gxp_pp.x && gxp_pp.y == gxp_pp.y && gxp_pp.w == gxp_pp.w;");
        let _ = writeln!(m, "    out.v0 = vec4<f32>(select(0.0, clamp(gxp_nd.x * 0.5 + 0.5, 0.0, 1.0), gxp_ok), select(0.0, clamp(gxp_nd.y * 0.5 + 0.5, 0.0, 1.0), gxp_ok), select(0.0, 1.0, gxp_pp.w > 0.0), select(0.0, 1.0, gxp_ok));");
        let _ = writeln!(m, "  }}");
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
    // A RAW 64-bit attachment takes the colour register pair's words; the link refused the
    // depth-writing and destination-reading shapes, so this is the plain entry only.
    let ret_ty = match (writes_depth, fplan.raw64_output) {
        (true, _) => "FsOut",
        (false, true) => "@location(0) vec2<u32>",
        (false, false) => "@location(0) vec4<f32>",
    };
    // >>> A DUAL-SOURCE BODY IS CUT WHERE IT STARTS DEPENDING ON THE DESTINATION.
    //
    // The lowering needs the body at two destinations (`module::dual_source_plan` has the
    // algebra), and the straightforward way to get that is a function called twice. But the
    // PREFIX - everything before the first instruction that reads the output bank - reads
    // varyings, uniforms and textures and nothing that differs between the two calls, so
    // running it twice is pure waste. On mlb's world-family blend, which paints 78% of the
    // world pass's fragments, that prefix is most of the body.
    //
    // So when the body can be cut, `fs_main` IS the body: the prefix once, then the suffix
    // twice over a saved copy of the register file. [`split_dual_body`] decides.
    let dual_split = fplan.dual_source.then(|| split_dual_body(fbody, fplan.dual_split)).flatten();
    if fplan.dual_source && dual_split.is_none() {
        // The body becomes a FUNCTION OF THE DESTINATION and the entry point below calls it
        // twice - see `module::dual_source_eligible` for the algebra.
        let _ = writeln!(m, "\nfn gxp_body(in: FsIn, gxp_dstc_in: vec4<f32>) -> vec4<f32> {{");
    } else if fplan.dual_source {
        // The unsplit form gets this struct from `DUAL_SOURCE_ENTRY`; the split form IS the
        // entry point, so it declares it.
        m.push_str(GXP_DUAL_STRUCT);
        let _ = writeln!(m, "\n@fragment\nfn fs_main(in: FsIn) -> GxpDual {{");
    } else {
        let _ = writeln!(m, "\n@fragment\nfn fs_main(in: FsIn) -> {ret_ty} {{");
    }
    m.push_str(crate::wgsl::FRONT_FACING_DECL);
    emit_register_banks(&mut m);
    // A program that blends for ITSELF starts with the destination colour in the output bank,
    // because that is what the hardware seeds those registers with - see
    // `BindingPlan::reads_dest_color`. Before the body, and before anything else writes `o`.
    // A SPLIT body seeds the output bank inside each arm instead, at the cut - see
    // `split_dual_body`. Seeding here would put the destination in `o` before a prefix that
    // may write `o` itself, and then the second arm's reseed would erase that write.
    if fplan.reads_dest_color && dual_split.is_none() {
        m.push_str(&crate::module::dest_color_init(fplan.color_precision, fplan.dual_source));
    }
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
                    "  pa[{register}] = {HALF_PK_FN}({}, {});",
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
                "  pa[{register}] = {HALF_PK_FN}({:?}, {:?});",
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
        // The texture-coordinate DEFAULT when the vertex produces no such texcoord - see
        // `PlannedPrefetch::coords_unfed`. Same constant the surplus-register fill uses, and
        // the same one a GXM texcoord defaults to: zero.
        let n = pf.coord_arity();
        let coord = if pf.coords_unfed {
            vec!["0.0"; usize::from(n)].join(", ")
        } else if pf.projective {
            // `xy / w` - see `PlannedPrefetch::projective`. Emitted as the whole vector so the
            // `vec2` the sample below wraps it in divides both lanes by the one `w`.
            let c: Vec<String> = pf.coords.iter().map(|&i| at(i)).collect();
            format!("vec2<f32>({}, {}) / {}", c[0], c[1], c[2])
        } else {
            pf.coords.iter().map(|&i| at(i)).collect::<Vec<_>>().join(", ")
        };
        if fplan.samplers.iter().any(|b| b.unit == pf.unit && b.raw) {
            // A RAW 64-bit texel: the two stored words, unconverted, into the first two of the
            // prefetch's registers. `textureLoad` at the nearest texel of level 0 - an integer
            // texture cannot be filtered, and the hardware hands raw data over unfiltered too.
            // The coordinate wraps (REPEAT), which is what a prefetch off a plain texcoord
            // does; the registers past the pair are what the allocation held, which nothing
            // established reads before writing (the one program in the corpus overwrites them).
            let _ = writeln!(
                m,
                "  let pf{i} = textureLoad(t{0}, vec2<i32>(clamp(vec2<f32>(fract(vec2<f32>({coord}))) * vec2<f32>(textureDimensions(t{0})), vec2<f32>(0.0), vec2<f32>(textureDimensions(t{0})) - vec2<f32>(1.0))), 0);",
                pf.unit
            );
            let _ = writeln!(m, "  pa[{}] = pf{i}.x;", pf.pa_base);
            if pf.regs > 1 {
                let _ = writeln!(m, "  pa[{}] = pf{i}.y;", pf.pa_base + 1);
            }
            continue;
        }
        let _ = writeln!(
            m,
            "  let pf{i} = textureSample(t{0}, s{0}, vec{n}<f32>({coord}));",
            pf.unit
        );
        if pf.regs == 4 {
            // Four registers: the same four components UNPACKED, one full-precision component
            // each. The one program in these corpora that asks for this reads them back with an
            // F32-granular four-component swizzle off its prefetch base, so packing them in
            // halves here would feed it two registers of packed pairs and two of whatever the
            // allocation held.
            for c in 0..4 {
                let _ = writeln!(
                    m,
                    "  pa[{}] = bitcast<u32>(pf{i}.{});",
                    pf.pa_base + c,
                    ["x", "y", "z", "w"][c as usize]
                );
            }
        } else if pf.regs > 1 {
            // Two registers: four F16 components, packed two per register.
            let _ = writeln!(m, "  pa[{}] = {HALF_PK_FN}(pf{i}.x, pf{i}.y);", pf.pa_base);
            let _ = writeln!(m, "  pa[{}] = {HALF_PK_FN}(pf{i}.z, pf{i}.w);", pf.pa_base + 1);
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
    // The driver-placed pointer register, exactly as the vertex entry does it: the bound
    // buffer's guest address in the SA register the DATA container names, so the body's own
    // address arithmetic runs bit-exact. After the literals, for the same reason.
    for (i, w) in fplan.mem_windows.iter().enumerate() {
        let _ = writeln!(m, "  sa[{}] = gxp_fmem[{i}u].x;", w.base_sa);
    }
    // >>> AND WHEN THE POSITION PROBE IS ARMED, THE FRAGMENT BODY DOES NOT RUN AT ALL.
    //
    // The probe exists to answer "where did this draw send its vertices" for a draw that
    // rasterises NOTHING, and the guest fragment can throw the answer away before it is seen:
    // an alpha test is a `discard`, and a discarded fragment leaves no pixel and is counted by
    // no occlusion query. `VITASLOP_GXP_SOLID` cannot help - it rewrites the final `return`
    // and everything above it, the `discard` included, still executes - which is why a crowd
    // pair came back empty under it and read as dead geometry when the question was open.
    //
    // Skipping the body makes the two outcomes distinguishable: geometry that exists now
    // paints (its own clip position as colour, per `POSPROBE_ARM`), and geometry that is
    // degenerate still paints nothing.
    if posprobe_on() && varying_locations > 0 {
        let _ = writeln!(m, "  return vec4<f32>(in.v0.x, in.v0.y, in.v0.z, 1.0);");
        let _ = writeln!(m, "}}");
        return size_register_banks(&unpack_half_registers(&resolve_sa_init(&strip_split_markers(&m))));
    }
    m.push_str(dual_split.as_ref().map_or(fbody, |(head, _)| head.as_str()));
    let (ret, base) = match fplan.color {
        ColorOutput::NativeO0 => ("o", 0),
        ColorOutput::NonNativePa(base) => ("pa", base),
    };
    let color =
        crate::module::color_return_expr(ret, base, fplan.color_precision, varying_locations);
    if let Some((_, tail)) = dual_split.as_ref() {
        emit_dual_split_tail(&mut m, tail, &color, fplan.color_precision);
        return size_register_banks(&unpack_half_registers(&resolve_sa_init(&strip_split_markers(&m))));
    }
    if fplan.raw64_output {
        // The register pair's bits, as the hardware stores them into a 64-bit surface. Whatever
        // precision the program wrote them at - packed halves, packed bytes, a float - the
        // surface holds the words, and a later raw sample reads them back unchanged.
        let _ = writeln!(m, "  return vec2<u32>({ret}[{base}], {ret}[{}]);\n}}", base + 1);
        return size_register_banks(&unpack_half_registers(&resolve_sa_init(&strip_split_markers(&m))));
    }
    if let Some(spec) = depth_probe() {
        // `=<min>:<scale>` spreads the window `[min, min + 1/scale]` over the whole grey ramp.
        // A projection crams its whole scene into the far end of the depth buffer - one title's
        // sits above 0.99 - so the plain ramp saturates and answers nothing; this is what makes
        // the difference between two surfaces readable.
        let (lo, k) = match spec.split_once(':') {
            Some((a, b)) => (a.trim().parse::<f32>().unwrap_or(0.0), b.trim().parse::<f32>().unwrap_or(1.0)),
            None => (0.0, 1.0),
        };
        let d = format!("clamp((in.frag_coord.z - {lo:?}) * {k:?}, 0.0, 1.0)");
        let color = format!("vec4<f32>(vec3<f32>({d}), 1.0)");
        if writes_depth {
            let _ = writeln!(m, "  return FsOut({color}, gxp_frag_depth);
}}");
        } else {
            let _ = writeln!(m, "  return {color};
}}");
        }
        if fplan.dual_source {
            m.push_str(DUAL_SOURCE_ENTRY);
        }
        return size_register_banks(&unpack_half_registers(&resolve_sa_init(&strip_split_markers(&m))));
    }
    let color = match dest_probe() {
        // `opaque` forces alpha to 1: a draw whose pipeline blend is `SrcAlpha` and whose
        // destination alpha is zero composites to NOTHING, so the plain probe cannot tell
        // "the copy is black" from "the copy is transparent and the blend discarded it".
        Some("opaque") if fplan.reads_dest_color => {
            "vec4<f32>(gxp_dstc.rgb, 1.0)".to_string()
        }
        // `mark` paints a flat MAGENTA. It answers the question that has to come first and
        // that the other two arms cannot: is this pixel painted by a destination reader at
        // all? A black rectangle where a blend should be looks the same whether the copy read
        // black or the draw that made it never read a destination.
        Some("mark") if fplan.reads_dest_color => "vec4<f32>(1.0, 0.0, 1.0, 1.0)".to_string(),
        Some(_) if fplan.reads_dest_color => "gxp_dstc".to_string(),
        _ => color,
    };
    if writes_depth {
        let _ = writeln!(m, "  return FsOut({color}, gxp_frag_depth);\n}}");
    } else {
        let _ = writeln!(m, "  return {color};\n}}");
    }
    if fplan.dual_source {
        m.push_str(DUAL_SOURCE_ENTRY);
    }

    // Last, and in this order: `resolve_sa_init` is what decides whether the SA bank is
    // subscripted dynamically at all, and `size_register_banks` sizes what comes out of it.
    size_register_banks(&unpack_half_registers(&resolve_sa_init(&strip_split_markers(&m))))
}

/// Emit a stage's SA-bank initialisation: the default uniform buffer copied verbatim into
/// registers `0..uniform_regs`, then the container literals stored at their own registers.
/// Both are raw 32-bit register values - a register may hold an F32 or two packed F16 halves,
/// and only the instruction reading it decides which.
fn emit_secondary_attrs(m: &mut String, binding: &str, uniform_regs: u32, literals: &[(u32, u32)]) {
    if uniform_regs > 0 {
        // A MARKER, resolved by [`resolve_sa_init`] once the stage's whole body exists - the
        // form this becomes depends on which SA registers the body reads and which it writes,
        // and neither is known here.
        let _ = writeln!(m, "  {SA_INIT_MARKER}{binding}:{uniform_regs}");
    }
    for &(reg, value) in literals {
        let _ = writeln!(m, "  sa[{reg}] = {value:#010x}u;");
    }
}

/// The marker [`emit_secondary_attrs`] leaves for [`resolve_sa_init`], carrying the stage's
/// uniform binding name and its default-uniform register count. It is a WGSL line comment, so
/// a module that somehow reached a driver with one unresolved still compiles.
const SA_INIT_MARKER: &str = "//@@GXP_SA_INIT:";

/// Turn each stage's SA-bank marker into the reads the body actually needs.
///
/// # The bank was COPIED, and the copy was the bug
/// This used to emit one loop per stage:
///
/// ```text
///   for (var gxp_sa_k: u32 = 0u; gxp_sa_k < 78u; gxp_sa_k = gxp_sa_k + 1u)
///     { sa[gxp_sa_k] = vs_sa.data[gxp_sa_k / 4u][gxp_sa_k % 4u]; }
/// ```
///
/// - it runs per INVOCATION - 78 to 90 iterations for every vertex of every draw, to copy a
///   uniform that a constant subscript could have read directly, and
/// - `sa[gxp_sa_k]` is a DYNAMIC subscript into a function-local array, which is the one
///   construct that forces a driver to materialise the whole bank as indexable storage.
///
/// **The second point is not theoretical: it turned a retail race BLACK on an
/// Android PowerVR (img-tec D-series).** While the banks were declared at the full
/// [`BANK_REGS`] the driver had no choice but scratch memory and compiled it; once
/// [`size_register_banks`] cut them to their real extent, the four pairs with the largest
/// default-uniform banks (83, 95, 95 and 107 registers - every smaller pair in the title
/// compiled) failed pipeline creation outright with `CreateGraphicsPipelines failed with
/// VK_ERROR_UNKNOWN`, and ONE failed pipeline in a pass invalidates the whole command buffer,
/// so the frame drew nothing at all.
///
/// # What it emits instead
/// A read of an SA register the body never WRITES becomes a direct constant-subscript uniform
/// read - `vs_sa.data[8][3]` for `sa[35]` - which is what a hand-written shader would have said.
/// Only registers the body does write keep a local slot, initialised once at entry. On the four
/// pairs above that is 2 of 78 and 2 of 90: the loop and the bank both disappear.
///
/// # Where it refuses
/// A stage that subscripts `sa` DYNAMICALLY (an `indexed_element` register-indirect read) keeps
/// the loop verbatim: the index is not known here, so no substitution can be proved safe and
/// the bank has to hold every register the index could reach. That is the same boundary
/// [`bank_extent`] draws, and for the same reason.
/// The entry point of a DUAL-SOURCE module: `G` is the body at destination zero, `F` the body
/// at destination one minus `G`, and the pipeline blend `src0 + dst * src1` reassembles
/// `G + dst * F`. See `module::dual_source_eligible`.
/// Cut a dual-source fragment body into (prefix, suffix) at the first instruction that reads
/// the destination, or `None` when it cannot be cut and the body must be evaluated twice whole.
///
/// It refuses in four cases, each of which would make the shape below say something other than
/// what the body says:
///
/// * NO SPLIT INDEX, or a split at instruction 0 - there is no prefix to save.
/// * NO MARKER for that instruction: [`emit_body_marked`] marks only TOP-LEVEL boundaries, so a
///   blend that begins inside an `if` or a loop has none, and cutting anywhere else would put a
///   brace on one side of the cut and its match on the other.
/// * THE PREFIX TOUCHES THE OUTPUT BANK. It cannot READ it (that is what the split index
///   means), so any `o[` there is a WRITE, and the suffix's reseed would erase it. Rare enough
///   to refuse rather than model.
/// * A DIAGNOSTIC PROBE is armed (`VITASLOP_GXP_DEPTH_PROBE` / `VITASLOP_GXP_DEST_PROBE`): those
///   replace the returned colour, and a probe must not also change the shape of the module it
///   is probing [[vitaslop-instrument-failure-imitating-its-subject]].
fn split_dual_body(fbody: &str, split: Option<usize>) -> Option<(String, String)> {
    if depth_probe().is_some() || dest_probe().is_some() {
        return None;
    }
    let at = fbody.find(&split_marker(split.filter(|&i| i > 0)?))?;
    let (head, tail) = fbody.split_at(at);
    if head.contains("o[") {
        return None;
    }
    Some((head.to_string(), tail.to_string()))
}

/// The two evaluations of a split dual-source body's SUFFIX, and the `GxpDual` return.
///
/// The register file is function-scope `var` arrays, so "run the suffix again from the state
/// the prefix left" is a save of every bank the suffix touches and a restore between the two
/// runs. Only those banks: a bank the suffix never subscripts is one the module may not even
/// declare, and naming it here would be a reference to nothing.
///
/// Each run is WRAPPED IN A BLOCK. The suffix carries `let` bindings of its own - a gather's
/// texels, a hoisted unpack, a staged store's temporary - and two copies of the same text in
/// one scope is a WGSL redefinition, which is a module the device refuses outright rather than
/// a shader that renders wrong.
fn emit_dual_split_tail(m: &mut String, tail: &str, color: &str, precision: ColorPrecision) {
    // >>> RE-RUNNING ONLY THE DESTINATION-DEPENDENT STATEMENTS WAS TRIED 2026-09-12g AND IS A
    // >>> NULL ON THE PAIR IT WAS AIMED AT. The suffix is cut at the first instruction that
    // READS the destination, so "after the cut" is a POSITION and not a dependence, and on mlb's
    // world-family blend only three of its 82 statements touch the destination. But a second
    // evaluation that skips the other 79 is only correct if none of them overwrites a register a
    // dependent statement wrote - and this program reuses `r[]` heavily, so every one of them
    // does. Pulling those back in leaves nothing skipped. Shrinking this needs RENAMING (the
    // re-run writing its own copies), not a dependence walk.
    //
    // `p` and `idx` are declared unconditionally by `emit_register_banks`; the five register
    // banks are declared only if subscripted, which is exactly the test below.
    let banks: Vec<&str> = ["r", "o", "i", "pa", "sa", "p", "idx"]
        .into_iter()
        .filter(|b| tail.contains(&format!("{b}[")))
        .collect();
    let _ = writeln!(m, "  var gxp_g: vec4<f32>;");
    let _ = writeln!(m, "  var gxp_f: vec4<f32>;");
    for b in &banks {
        let _ = writeln!(m, "  let gxp_save_{b} = {b};");
    }
    for (pass, dst) in [("gxp_g", 0.0f32), ("gxp_f", 1.0f32)].iter().enumerate() {
        let (name, d) = *dst;
        if pass == 1 {
            for b in &banks {
                let _ = writeln!(m, "  {b} = gxp_save_{b};");
            }
        }
        let _ = writeln!(m, "  {{");
        let _ = writeln!(m, "  let gxp_dstc = vec4<f32>({d:?}, {d:?}, {d:?}, {d:?});");
        m.push_str(&crate::module::dest_seed(precision));
        m.push_str(tail);
        let _ = writeln!(m, "  {name} = {color};");
        let _ = writeln!(m, "  }}");
    }
    // `G` is the body at destination zero and `F` the CHANGE the destination makes, which is
    // what the pipeline's `src0 * 1 + dst * src1` multiplies the destination by.
    let _ = writeln!(m, "  return GxpDual(gxp_g, gxp_f - gxp_g);
}}");
}

/// The two-colour fragment output a dual-source blend takes: `g` is the source term and `f` the
/// factor the ROP multiplies the destination by. The unsplit entry below carries its own copy;
/// the SPLIT entry is `fs_main` itself, so it pushes this.
const GXP_DUAL_STRUCT: &str = "
struct GxpDual {
  @location(0) @blend_src(0) g: vec4<f32>,
  @location(0) @blend_src(1) f: vec4<f32>,
};
";

const DUAL_SOURCE_ENTRY: &str = "
struct GxpDual {
  @location(0) @blend_src(0) g: vec4<f32>,
  @location(0) @blend_src(1) f: vec4<f32>,
};
@fragment
fn fs_main(in: FsIn) -> GxpDual {
  let gxp_g = gxp_body(in, vec4<f32>(0.0, 0.0, 0.0, 0.0));
  let gxp_f = gxp_body(in, vec4<f32>(1.0, 1.0, 1.0, 1.0)) - gxp_g;
  return GxpDual(gxp_g, gxp_f);
}
";

/// Give every register the program only ever uses as a PACKED F16 PAIR an UNPACKED home, so
/// reading half of one stops being a conversion.
///
/// # The bill this exists to cut
/// WGSL has no 16-bit float without an extension, so a USSE 16-bit register is a `u32` holding
/// two halves and the emitter spells every read `unpack2x16float(r[n])[k]` and every write
/// `r[n] = gxp_hlo(r[n], ...)`, which is a narrowing plus a masked merge. Those are
/// CONVERSIONS, and a fragment
/// program runs them per fragment: mlb's world-family blend emits 466 of them, and that pair
/// paints 78% of the samples in the pass that is the phone's most expensive
/// [[vitaslop-f16-emulation-is-the-phones-world-pass]]. A desktop GPU's compiler folds most of
/// them away and prices this at zero [[vitaslop-desktop-cannot-price-a-count-win]]; a tiler
/// does not.
///
/// # What it does
/// A register whose every occurrence in a stage is one of the recognised F16 forms gets a
/// second home, `<bank>_h: array<vec2<f32>>`, holding the two halves ALREADY UNPACKED. Then
/// - a read `unpack2x16float(r[n])` becomes `r_h[n]` - no instruction at all;
/// - a paired write becomes `r_h[n] = gxp_q2(...)`, and a single half `r_h[n][k] = ...`, which
///   is a component STORE rather than a read-modify-write of the packed word;
/// - the `gxp_hq` rounding that survives is what keeps the ARITHMETIC honest: the hardware
///   register really is 16 bits, so every store still rounds to the value the guest reads back.
///   The conversions that disappear are the ones that only existed to move a value in and out
///   of a packed word.
///
/// # Why it is a TEXT pass, and why that is the safe way round
/// The same reason [`size_register_banks`] is: the emitted text is the only place where the
/// decoder's register doubling, the swizzle selectors, the write masks, the half-pair packing
/// and this file's own prologue have all already been applied. A second implementation of those
/// rules, walking the IR, would disagree somewhere and silently corrupt a shader.
///
/// And the direction of the fallback is what makes it safe: a register is rewritten only when
/// EVERY occurrence of it is recognised. An occurrence this pass does not understand - a
/// `bitcast<f32>` of the same register, a byte view, a raw integer store, an address
/// computation, a dynamic subscript - disqualifies that register, and a dynamic subscript
/// disqualifies its whole bank, because there is no telling which registers it reaches. An
/// unrecognised form therefore costs the OLD emission, never a wrong one.
///
/// `VITASLOP_GXP_HALF_REGS=1` turns it ON; it is OFF by default - see [`half_regs_on`].
fn unpack_half_registers(module: &str) -> String {
    if !half_regs_on() {
        return module.to_string();
    }
    let mut out = String::with_capacity(module.len());
    let mut rest = module;
    while let Some(at) = rest.find(BANKS_MARKER) {
        let after = at + BANKS_MARKER.len();
        out.push_str(&rest[..after]);
        let body = &rest[after..];
        // A stage's body ends where the next stage's declarations begin - the same regions
        // [`resolve_sa_init`] and [`size_register_banks`] walk.
        let end = body.find(BANKS_MARKER).unwrap_or(body.len());
        out.push_str(&unpack_half_registers_region(&body[..end]));
        rest = &body[end..];
    }
    out.push_str(rest);
    out
}

/// The five register banks, as [`size_register_banks`] names them.
const HALF_BANKS: [&str; 5] = ["r", "o", "i", "pa", "sa"];

/// The suffix an unpacked half-register array's name takes.
///
/// `_h` rather than `h` so that scanning for `r[` - which [`emit_dual_split_tail`] and
/// [`bank_extent`] both do - cannot match the half array by accident.
const HALF_SUFFIX: &str = "_h";

/// One recognised statement shape, and what the rewrite makes of it.
enum HalfStore<'a> {
    /// `X[n] = gxp_hpk(LO, HI);` - both halves at once (the `wgsl::fold_halves` shape). The
    /// slice is the two ARGUMENTS, comma-separated, not a `vec2` expression.
    Pair(&'a str),
    /// `X[n] = gxp_hlo(X[n], ARG);` - the low half.
    Low(&'a str),
    /// The same for the high half.
    High(&'a str),
    /// `X[n] = RHS;` for an RHS that does not mention `X[n]`: a uniform word, a container
    /// literal, a driver-placed pointer. Whatever those BITS mean, a register whose every other
    /// occurrence is a half read is a register whose halves are what is read back, so unpacking
    /// the word once at the store is exact and moves the conversion out of the body.
    Word(&'a str),
    /// `X[n] = Y[m];` or `X[n] = (Y[m] | 0u);` - a whole-WORD copy of one register into
    /// another, which is what a `mov` of a 16-bit PAIR emits. Called out from [`HalfStore::Word`]
    /// because both ends can then keep their unpacked homes and the copy becomes a plain
    /// assignment: a register copied this way is the one case where a raw word read is not
    /// evidence that the register is anything but two halves. See the fixpoint in
    /// [`unpack_half_registers_region`] for what happens when only one end qualifies.
    Copy(&'static str, usize),
}

/// `Y[m]` or `(Y[m] | 0u)` as a whole-word register read, and nothing else.
fn parse_word_copy(rhs: &str) -> Option<(&'static str, usize)> {
    let inner = rhs.strip_prefix('(').and_then(|r| r.strip_suffix(" | 0u)")).unwrap_or(rhs);
    let (bank, reg, after) = parse_bank_subscript(inner)?;
    after.is_empty().then_some((bank, reg))
}

/// Split `line` into the store it performs, if it is one of the recognised shapes, and the
/// right-hand text that store did not account for.
fn parse_half_store(line: &str) -> Option<(&'static str, usize, HalfStore<'_>)> {
    let t = line.trim_start();
    let (bank, reg, after) = parse_bank_subscript(t)?;
    let rest = after.strip_prefix(" = ")?.trim_end().strip_suffix(';')?;
    let dest = format!("{bank}[{reg}]");
    // The two half forms, spelled exactly as `wgsl::half_stmt` writes them. These are the
    // emitter's own helper names rather than literal text, so a change to how a half store is
    // spelled cannot leave this pass silently matching nothing and quietly doing no work.
    let low_mid = format!("{HALF_LO_FN}({dest}, ");
    let high_mid = format!("{HALF_HI_FN}({dest}, ");
    if let Some(inner) = rest.strip_prefix(&low_mid).and_then(|r| r.strip_suffix(')')) {
        return Some((bank, reg, HalfStore::Low(inner)));
    }
    if let Some(inner) = rest.strip_prefix(&high_mid).and_then(|r| r.strip_suffix(')')) {
        return Some((bank, reg, HalfStore::High(inner)));
    }
    if let Some(inner) =
        rest.strip_prefix(&format!("{HALF_PK_FN}(")).and_then(|r| r.strip_suffix(')'))
    {
        return Some((bank, reg, HalfStore::Pair(inner)));
    }
    // A RAW half store (`wgsl::Dest::store_raw_half`) read-modify-writes the packed word with a
    // 16-bit BIT PATTERN, and there is no telling a bit pattern from a float here - so a store
    // whose right-hand side mentions its own destination and is not one of the two shapes above
    // is left alone, and its destination is disqualified by that self-reference.
    if rest.contains(&dest) {
        return None;
    }
    if let Some(src) = parse_word_copy(rest) {
        return Some((bank, reg, HalfStore::Copy(src.0, src.1)));
    }
    Some((bank, reg, HalfStore::Word(rest)))
}

/// `X[n]` at the START of `t`: the bank name, the literal index, and what follows the `]`.
fn parse_bank_subscript(t: &str) -> Option<(&'static str, usize, &str)> {
    let bank = HALF_BANKS
        .into_iter()
        .find(|b| t.starts_with(b) && t.as_bytes().get(b.len()) == Some(&b'['))?;
    let sub = &t[bank.len() + 1..];
    let digits = sub.len() - sub.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    if digits == 0 || sub.as_bytes().get(digits) != Some(&b']') {
        return None;
    }
    Some((bank, sub[..digits].parse().ok()?, &sub[digits + 1..]))
}

/// Every `unpack2x16float(X[n])` in `text`, as `(bank, reg, start, end)` byte offsets.
fn half_reads(text: &str) -> Vec<(&'static str, usize, usize, usize)> {
    const CALL: &str = "unpack2x16float(";
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(rel) = text[from..].find(CALL) {
        let at = from + rel + CALL.len();
        from = at;
        let Some((bank, reg, after)) = parse_bank_subscript(&text[at..]) else { continue };
        if !after.starts_with(')') {
            continue;
        }
        let end = text.len() - after.len() + 1;
        out.push((bank, reg, at - CALL.len(), end));
        from = end;
    }
    out
}

/// Every bank subscript in `text`: `Ok` names a register, `Err` a bank subscripted by something
/// that is not a literal and therefore reaches anywhere in it.
fn stray_bank_refs(text: &str) -> Vec<Result<(&'static str, usize), &'static str>> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    for bank in HALF_BANKS {
        let mut from = 0usize;
        while let Some(rel) = text[from..].find(bank) {
            let start = from + rel;
            from = start + bank.len();
            // A bank name is a whole identifier followed by its subscript - the same two
            // boundary tests [`bank_extent`] makes, for the same reason.
            if start > 0 {
                let p = bytes[start - 1];
                if p.is_ascii_alphanumeric() || p == b'_' {
                    continue;
                }
            }
            if bytes.get(from) != Some(&b'[') {
                continue;
            }
            match parse_bank_subscript(&text[start..]) {
                Some((b, reg, _)) => out.push(Ok((b, reg))),
                None => out.push(Err(bank)),
            }
        }
    }
    out
}

/// Whether a 16-bit register gets an UNPACKED home. **OFF unless asked for**, and the reason is
/// a measurement, not caution.
///
/// The pass removes 70% of the f16 conversions a corpus emits (mlb: 4,459 -> 1,330), and on the
/// only engine here that can price it - the DESKTOP BROWSER, whose shader compiler is the
/// phone's family - it is worth NOTHING: mlb's world pass reads p50 0.91 and 0.88 ms across two
/// runs of the SAME arm, and both arms of the change land inside that. It also moves pixels on
/// five of seven titles (a mean of 0.02-0.07 of 255, on glyph edges and stipple, where a rounding
/// difference flips an edge blend) and grows a pair's declared register storage 228 -> 332 words,
/// which on a phone is occupancy.
///
/// A change that demonstrates nothing, alters the picture and costs registers is not one to ship
/// on a static count alone. What it IS is armed and ready for the device that can settle it: the
/// phone is ALU-bound on this title's world pass, which is exactly where ~60 conversions a
/// fragment over 3M fragments would be paid. `=1` takes it, `=noexact` takes it with every store
/// rounded.
fn half_regs_on() -> bool {
    // Back to OPT-IN (2026-09-18): a phone run with it on by default read the world pass at
    // 16.0 ms over 1,062 draws against 12.1 ms over 1,055 without it - no win, and the
    // register growth is the likely cost. `=1` / `=noexact` take it.
    matches!(arm(HALF_REGS_ARM), Some("1") | Some("noexact"))
}

/// A register's occurrences, split into the LIVE RANGES a whole-register write separates.
///
/// A program reuses one register at two precisions: mlb's world blend keeps a prefetched texel
/// pair in `pa[2]` as two halves for fifty reads, then overwrites the whole register with an
/// `f32` and uses it as a texture coordinate. Under a per-REGISTER rule that one `f32` use costs
/// all fifty half reads their unpacked home. Under a per-RANGE rule it costs only its own range,
/// because a write of the WHOLE register ends the previous value's life: nothing after it can
/// read what was there before, so the two ranges can live in different homes.
///
/// The split is refused where it would not be sound:
/// * a region containing a LOOP is not split at all - a back edge means a line further down the
///   text can run before one above it, and then "the previous value is dead" is not a fact the
///   line order establishes;
/// * a whole-register write inside a CONDITIONAL block does not open a range, because the old
///   value survives the path that skips it. An unconditional `{ }` - the staging block the
///   emitter puts around one instruction - is not a conditional and does not stop a split.
type Ranges = std::collections::HashMap<(&'static str, usize), usize>;

/// Whether a region has a back edge, and so cannot be range-split at all.
fn has_back_edge(region: &str) -> bool {
    ["loop {", "for (", "while ("].iter().any(|k| region.contains(k))
}

/// Walk `region` line by line, handing each line the live-range id of every register at that
/// point. Both the classification pass and the rewrite pass go through this, so the two cannot
/// disagree about which occurrence belongs to which range.
fn walk_half_ranges(region: &str, mut f: impl FnMut(&str, &Ranges)) {
    let split = !has_back_edge(region);
    let mut ranges: Ranges = Ranges::new();
    // One entry per open `{`: whether that block is conditional.
    let mut blocks: Vec<bool> = Vec::new();
    // >>> THE DUAL-SOURCE SPLIT RUNS ITS TAIL TWICE OVER A SAVED REGISTER FILE, and the
    // ranges have to be rewound with it. The second copy's reads see what the PREFIX left,
    // not what the first copy wrote, so carrying the first copy's range ids into the second
    // would label a read with a range that never reached it - and if one of those two ranges
    // lives unpacked and the other packed, the second copy reads a word nothing wrote.
    // Snapshotting at the save and rewinding at the restore is what the emitted code itself
    // does to the registers.
    let mut saved: Option<Ranges> = None;
    for line in region.split_inclusive('\n') {
        let t = line.trim_start();
        if t.starts_with("let gxp_save_") {
            saved.get_or_insert_with(|| ranges.clone());
        } else if t.contains(" = gxp_save_")
            && !t.starts_with("let ")
            && let Some(snapshot) = saved.as_ref()
        {
            ranges.clone_from(snapshot);
        }
        let conditional = blocks.iter().any(|c| *c);
        if split
            && !conditional
            && let Some((b, r, st)) = parse_half_store(line)
            && matches!(st, HalfStore::Pair(_) | HalfStore::Word(_))
        {
            *ranges.entry((b, r)).or_insert(0) += 1;
        }
        f(line, &ranges);
        let cond = t.starts_with("if ") || t.starts_with("if(") || t.starts_with("} else")
            || t.starts_with("else") || t.starts_with("for ") || t.starts_with("while ")
            || t.starts_with("switch ");
        for c in line.chars() {
            match c {
                '{' => blocks.push(cond),
                '}' => {
                    blocks.pop();
                }
                _ => {}
            }
        }
    }
}

fn unpack_half_registers_region(region: &str) -> String {
    use std::collections::BTreeSet;
    type Key = (&'static str, usize, usize);
    // A value earns an unpacked home by being read or written as a half somewhere...
    let mut half: BTreeSet<Key> = BTreeSet::new();
    // ...and loses it to any occurrence this pass does not recognise.
    let mut bad: BTreeSet<Key> = BTreeSet::new();
    let mut bad_bank: BTreeSet<&str> = BTreeSet::new();
    // A value the pass only ever sees STORED is left alone: moving it would convert at the
    // store and save nothing at all.
    let mut read: BTreeSet<Key> = BTreeSet::new();

    // `dest <- src` for every whole-word register copy, for the fixpoint below.
    let mut copies: Vec<(Key, Key)> = Vec::new();
    walk_half_ranges(region, |line, ranges| {
        let at = |b: &'static str, r: usize| (b, r, ranges.get(&(b, r)).copied().unwrap_or(0));
        let mut residue = line;
        if let Some((bank, reg, store)) = parse_half_store(line) {
            match store {
                HalfStore::Pair(inner) | HalfStore::Low(inner) | HalfStore::High(inner) => {
                    half.insert(at(bank, reg));
                    residue = inner;
                }
                // Not evidence of a half by itself, and not a disqualification either.
                HalfStore::Word(rhs) => residue = rhs,
                HalfStore::Copy(sb, sr) => {
                    half.insert(at(bank, reg));
                    half.insert(at(sb, sr));
                    read.insert(at(sb, sr));
                    copies.push((at(bank, reg), at(sb, sr)));
                    residue = "";
                }
            }
        }
        let reads = half_reads(residue);
        for (bank, reg, _, _) in &reads {
            half.insert(at(bank, *reg));
            read.insert(at(bank, *reg));
        }
        // Whatever is left once the store skeleton and the half reads are taken out.
        let mut left = String::with_capacity(residue.len());
        let mut from = 0usize;
        for (_, _, s, e) in &reads {
            left.push_str(&residue[from..*s]);
            from = *e;
        }
        left.push_str(&residue[from..]);
        for r in stray_bank_refs(&left) {
            match r {
                Ok((b, reg)) => {
                    bad.insert(at(b, reg));
                }
                Err(b) => {
                    bad_bank.insert(b);
                }
            }
        }
    });

    // >>> A COPY WHOSE DESTINATION STAYS PACKED READS ITS SOURCE AS A WORD.
    //
    // The classification above took both ends of `X[n] = Y[m];` to be halves, which is true only
    // if the statement is rewritten. If the destination turns out not to qualify - it is read as
    // an `f32` somewhere - the line stays as it is, and then it really does read `Y[m]`'s packed
    // word, so the source cannot live unpacked either. Marking the source bad can in turn sink
    // another copy's destination, so this runs to a fixpoint rather than once.
    loop {
        let settled = |k: &Key, bad: &BTreeSet<Key>| {
            half.contains(k) && read.contains(k) && !bad.contains(k) && !bad_bank.contains(k.0)
        };
        let mut changed = false;
        for (dest, src) in &copies {
            if !settled(dest, &bad) && bad.insert(*src) {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let keep = |k: Key| half.contains(&k) && read.contains(&k) && !bad.contains(&k) && !bad_bank.contains(k.0);
    if !half.iter().any(|k| keep(*k)) {
        return region.to_string();
    }

    let mut out = String::with_capacity(region.len());
    walk_half_ranges(region, |line, ranges| {
        let live = |b: &str, r: usize| {
            // `walk_half_ranges` only ever yields the bank names in `HALF_BANKS`, which are
            // `'static`, so the lookup can take the caller's borrowed name.
            let Some(bank) = HALF_BANKS.into_iter().find(|x| *x == b) else { return false };
            keep((bank, r, ranges.get(&(bank, r)).copied().unwrap_or(0)))
        };
        out.push_str(&rewrite_half_line(line, &live));
    });
    fix_dual_split_saves(&out)
}

/// One line with every qualified register moved to its unpacked home.
fn rewrite_half_line(line: &str, keep: &impl Fn(&str, usize) -> bool) -> String {
    let nl = if line.ends_with('\n') { "\n" } else { "" };
    let body = line.trim_end_matches('\n');
    let indent: String = body.chars().take_while(|c| c.is_whitespace()).collect();
    // The store first: its right-hand side is rewritten for reads like any other text, and its
    // destination decides the statement's shape.
    if let Some((bank, reg, store)) = parse_half_store(body)
        && keep(bank, reg)
    {
            let h = format!("{bank}{HALF_SUFFIX}[{reg}]");
            let stmt = match store {
                // The pair helper takes its two halves as separate arguments (one native
                // instruction narrows both), so the unpacked home - a `vec2<f32>` - builds the
                // vector back up here rather than the emitter shipping one it would then strip.
                HalfStore::Pair(inner) => {
                    let e = format!("vec2<f32>({})", rewrite_half_reads(inner, keep));
                    if f16_exact(&e) { format!("{h} = {e};") } else { format!("{h} = gxp_q2({e});") }
                }
                HalfStore::Low(inner) => half_component_store(&h, 0, &rewrite_half_reads(inner, keep)),
                HalfStore::High(inner) => half_component_store(&h, 1, &rewrite_half_reads(inner, keep)),
                HalfStore::Word(rhs) => {
                    format!("{h} = unpack2x16float({});", rewrite_half_reads(rhs, keep))
                }
                // Both ends unpacked, and the copy is a plain assignment. A source that stayed
                // packed still has to be unpacked here, which is the form this line had anyway.
                HalfStore::Copy(sb, sr) if keep(sb, sr) => format!("{h} = {sb}{HALF_SUFFIX}[{sr}];"),
                HalfStore::Copy(sb, sr) => format!("{h} = unpack2x16float({sb}[{sr}]);"),
            };
        return format!("{indent}{stmt}{nl}");
    }
    format!("{}{nl}", rewrite_half_reads(body, keep))
}

/// One half of an unpacked register, rounded only where the value is not already a 16-bit one.
///
/// Through [`crate::wgsl::HALF_QUANT_FN`] rather than WGSL's `quantizeToF16`, whose rounding
/// mode the language leaves implementation-defined exactly as it does `pack2x16float`'s. A
/// register that keeps its halves unpacked must round the same way as one that keeps them
/// packed, or turning the pass on would change every 16-bit number in the frame.
fn half_component_store(home: &str, component: usize, expr: &str) -> String {
    if f16_exact(expr) {
        format!("{home}[{component}] = {expr};")
    } else {
        format!("{home}[{component}] = {HALF_QUANT_FN}({expr});")
    }
}

/// `unpack2x16float(X[n])` -> `X_h[n]`, for every qualified register in `text`.
fn rewrite_half_reads(text: &str, keep: &impl Fn(&str, usize) -> bool) -> String {
    let mut out = String::with_capacity(text.len());
    let mut at = 0usize;
    for (bank, reg, s, e) in half_reads(text) {
        if !keep(bank, reg) {
            continue;
        }
        out.push_str(&text[at..s]);
        let _ = write!(out, "{bank}{HALF_SUFFIX}[{reg}]");
        at = e;
    }
    out.push_str(&text[at..]);
    out
}

/// Whether `expr` can only ever hold a value that is ALREADY exactly a 16-bit float, so the
/// store rounding it would round nothing.
///
/// The unpacked home keeps a half as an `f32` and rounds on every store, because that is what
/// the hardware register does. But a great many USSE instructions only MOVE a value that is
/// already in a 16-bit register - `mov`, `abs`, `min`, `max`, a predicated select - and rounding
/// the output of one of those is two conversion instructions that cannot change a bit.
///
/// The test is structural and deliberately mean: an expression qualifies only if everything in
/// it is a half-register read, one of the operations that cannot introduce a value between two
/// f16 neighbours (`abs`, negation, `min`, `max`, `select`, vector construction), or one of the
/// two literals that are exactly representable at every precision. Anything else at all - an
/// add, a multiply, a texture sample, a transcendental, a literal this does not recognise -
/// fails the test and keeps its rounding. A false NEGATIVE costs two instructions; a false
/// positive would leave a value in a register at more precision than the guest's, so the bias
/// is all one way.
fn f16_exact(expr: &str) -> bool {
    // `VITASLOP_GXP_HALF_REGS=noexact` keeps the unpacked home and rounds EVERY store, which is
    // the control for this peephole: it is the one piece of the pass that can leave a value in a
    // register at more precision than the guest would, so a picture difference has to be able to
    // ask it the question directly.
    if arm(HALF_REGS_ARM) == Some("noexact") {
        return false;
    }
    // Anything left after the recognised pieces are struck out disqualifies the expression.
    let mut rest = expr.to_string();
    // Half-register reads, with or without a component selector: `pa_h[3]`, `r_h[0][1]`.
    while let Some(at) = rest.find(HALF_SUFFIX).filter(|at| {
        rest[..*at].chars().next_back().is_some_and(|c| c.is_ascii_lowercase())
    }) {
        let start = rest[..at].rfind(|c: char| !c.is_ascii_lowercase()).map_or(0, |i| i + 1);
        let after = &rest[at + HALF_SUFFIX.len()..];
        let Some(end) = after.strip_prefix('[').and_then(|r| r.find(']').map(|i| i + 2)) else {
            return false;
        };
        let mut end = at + HALF_SUFFIX.len() + end;
        if let Some(sel) = rest[end..].strip_prefix('[') {
            let Some(close) = sel.find(']') else { return false };
            end += close + 2;
        }
        rest.replace_range(start..end, " ");
    }
    for token in [
        "vec2<f32>", "abs", "min", "max", "select", "(", ")", ",", "-", " ", "\t",
        "<", ">", "=", "!", "&", "|", "0.0", "1.0", "true", "false",
    ] {
        rest = rest.replace(token, " ");
    }
    rest.trim().is_empty()
}

/// Keep the dual-source split's register-file save and restore in step with the rewrite.
///
/// [`emit_dual_split_tail`] decides which banks to save by scanning its tail for `<bank>[`, long
/// before this pass moves half registers out of those arrays. So a bank that has just lost its
/// last packed reference would be saved and restored by a name whose declaration is gone - a
/// module that does not compile - and an unpacked array the suffix writes would not be restored
/// at all, which is a WRONG PICTURE rather than a failure. Both are settled here, where the
/// final text says which arrays actually exist.
fn fix_dual_split_saves(region: &str) -> String {
    if !region.contains("gxp_save_") {
        return region.to_string();
    }
    let mut out = String::with_capacity(region.len());
    for line in region.split_inclusive('\n') {
        let t = line.trim();
        let save = t
            .strip_prefix("let gxp_save_")
            .and_then(|r| r.split_once(" = "))
            .map(|(b, _)| b.trim());
        let restore = t
            .split_once(" = gxp_save_")
            .filter(|(lhs, _)| !lhs.contains("let "))
            .map(|(lhs, _)| lhs.trim());
        let Some(bank) = save.or(restore) else {
            out.push_str(line);
            continue;
        };
        if !HALF_BANKS.contains(&bank) {
            out.push_str(line);
            continue;
        }
        if region.contains(&format!("{bank}[")) {
            out.push_str(line);
        }
        if region.contains(&format!("{bank}{HALF_SUFFIX}[")) {
            let h = format!("{bank}{HALF_SUFFIX}");
            let _ = if save.is_some() {
                writeln!(out, "  let gxp_save_{h} = {h};")
            } else {
                writeln!(out, "  {h} = gxp_save_{h};")
            };
        }
    }
    out
}

/// Where a module's leading DIRECTIVES end, which is the earliest a declaration may appear.
pub(crate) fn directives_end(module: &str) -> usize {
    let mut at = 0usize;
    for line in module.lines() {
        let t = line.trim();
        if t.is_empty()
            || t.starts_with("//")
            || t.starts_with("enable ")
            || t.starts_with("requires ")
            || t.starts_with("diagnostic")
        {
            at += line.len() + 1;
            continue;
        }
        break;
    }
    at
}

/// The helper an unpacked pair store rounds through: the two halves as the hardware keeps them.
/// One function rather than two [`crate::wgsl::HALF_QUANT_FN`] calls at every site, so the text
/// stays readable. It is defined in terms of that one, so there is exactly one statement in the
/// module of how an f32 narrows to a half - see [`crate::wgsl::HALF_LO_FN`].
const GXP_Q2: &str = "
fn gxp_q2(v: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(gxp_hq(v.x), gxp_hq(v.y));
}
";

fn resolve_sa_init(module: &str) -> String {
    let mut out = String::with_capacity(module.len());
    let mut rest = module;
    while let Some(at) = rest.find(BANKS_MARKER) {
        let after = at + BANKS_MARKER.len();
        out.push_str(&rest[..after]);
        let body = &rest[after..];
        // A stage's body ends where the next stage's declarations begin - the same regions
        // `size_register_banks` walks, and this pass runs first so that one sees the result.
        let end = body.find(BANKS_MARKER).unwrap_or(body.len());
        out.push_str(&resolve_sa_init_region(&body[..end]));
        rest = &body[end..];
    }
    out.push_str(rest);
    out
}

fn resolve_sa_init_region(region: &str) -> String {
    let Some(mark) = region.find(SA_INIT_MARKER) else {
        return region.to_string();
    };
    let line_start = region[..mark].rfind('\n').map_or(0, |i| i + 1);
    let line_end = region[mark..].find('\n').map_or(region.len(), |i| mark + i + 1);
    let spec = region[mark + SA_INIT_MARKER.len()..line_end].trim();
    let Some((binding, regs)) = spec.split_once(':') else {
        return region.to_string();
    };
    let Ok(uniform_regs) = regs.parse::<usize>() else {
        return region.to_string();
    };
    // Everything the marker precedes. Scanning from here rather than from the top of the region
    // keeps every byte offset below relative to ONE string: the text before the marker is the
    // bank declarations and the attribute loads, which name no SA register.
    let body = &region[line_end..];

    let loop_form = || {
        // `gxp_sa_k` is named, not `k`, because `bank_extent` has to recognise this exact
        // subscript: it is the one NON-constant index into a bank that is statically bounded by
        // its own literal, and every other non-constant subscript is not bounded at all.
        format!(
            "{}  for (var gxp_sa_k: u32 = 0u; gxp_sa_k < {uniform_regs}u; \
             gxp_sa_k = gxp_sa_k + 1u) {{ sa[gxp_sa_k] = {binding}.data[gxp_sa_k / 4u]\
             [gxp_sa_k % 4u]; }}\n{}",
            &region[..line_start],
            &region[line_end..]
        )
    };
    match arm("VITASLOP_GXP_SA_DIRECT") {
        // `0` - the copy loop, verbatim.
        Some("0") => return loop_form(),
        // `unroll` - copy every uniform register into the bank one constant subscript at a
        // time, and change NOTHING else: no read substitution, no compaction. This arm exists
        // to answer one question and it is not a perf arm. The direct form moves pixels on this
        // desktop GPU (0.47/255 mean over a race frame, concentrated on one material), and the
        // two candidate explanations - a value error in the substitution, or the driver
        // contracting differently once the copy shape changes - are told apart by an arm whose
        // EXPRESSIONS are identical to the loop form's and whose copy shape is not.
        Some("unroll") => {
            let mut init = String::new();
            for reg in 0..uniform_regs {
                let _ = writeln!(init, "  sa[{reg}] = {binding}.data[{}][{}];", reg / 4, reg % 4);
            }
            return format!("{}{init}{}", &region[..line_start], &region[line_end..]);
        }
        _ => {}
    }
    let Some(uses) = sa_uses(body) else {
        // Dynamically subscripted: the bank has to hold everything the index could reach.
        return loop_form();
    };
    // >>> WRITTEN IS A PROPERTY OF THE REGISTER, NOT OF THE OCCURRENCE, and reading it per
    // occurrence is a silent wrong-value bug rather than a compile failure. A program that
    // computes `sa[39] = 1.0 / sa[39]` and reads `sa[39]` again later has one occurrence marked
    // written and two not; substituting THOSE hands the later read the register's original
    // uniform value and the shader carries on with it. MEASURED, on one title's on-track
    // run: the world came back correct in geometry and blown out in exposure, because the
    // reciprocal a lighting term divides by had reverted to the value it was taken from.
    let high = uses.iter().map(|u| u.reg).max().unwrap_or(0);
    let mut written = vec![false; high + 1];
    for u in &uses {
        written[u.reg] |= u.written;
    }
    // Rewrite every read of a NEVER-written uniform-backed register into the uniform read it is.
    let mut rewritten = String::with_capacity(body.len());
    let mut at = 0usize;
    for u in &uses {
        if written[u.reg] || u.reg >= uniform_regs {
            continue;
        }
        rewritten.push_str(&body[at..u.start]);
        let _ = write!(rewritten, "{binding}.data[{}][{}]", u.reg / 4, u.reg % 4);
        at = u.end;
    }
    rewritten.push_str(&body[at..]);

    // What survives keeps a local slot, loaded once at entry. A register the body writes may
    // also be READ before that write (the half-precision and 8-bit stores are read-modify-write
    // by construction, and the case above reads its own register), so the load is not optional.
    let mut init = String::new();
    for (reg, w) in written.iter().enumerate().take(uniform_regs) {
        if *w {
            let _ = writeln!(init, "  sa[{reg}] = {binding}.data[{}][{}];", reg / 4, reg % 4);
        }
    }
    // What is left is a SPARSE set of constant subscripts - the two or three registers the body
    // writes, plus the container literals, which sit at whatever high register the compiler that
    // built the blob chose (register 92 of 107 in one of the four pairs above). `bank_extent`
    // sizes an array by its HIGH WATER MARK, so a bank with fifteen live registers is declared
    // with a hundred, and the dead slots are exactly the storage this whole pass exists to stop
    // handing a driver. Every subscript here is a literal, so they can simply be renumbered.
    compact_sa_registers(&format!("{}{init}{rewritten}", &region[..line_start]))
}

/// Renumber a stage's surviving `sa[N]` subscripts onto `0..k`, in first-appearance order.
///
/// Only ever called on a region [`resolve_sa_init_region`] has already proved carries no dynamic
/// SA subscript, so the mapping is total: every reference is a literal this pass can see and
/// rewrite. A region it cannot prove that of is returned untouched.
fn compact_sa_registers(region: &str) -> String {
    let Some(uses) = sa_uses(region) else {
        return region.to_string();
    };
    let mut map: Vec<(usize, usize)> = Vec::new();
    let mut out = String::with_capacity(region.len());
    let mut at = 0usize;
    for u in &uses {
        let slot = match map.iter().find(|(from, _)| *from == u.reg) {
            Some((_, to)) => *to,
            None => {
                let to = map.len();
                map.push((u.reg, to));
                to
            }
        };
        out.push_str(&region[at..u.start]);
        let _ = write!(out, "sa[{slot}]");
        at = u.end;
    }
    out.push_str(&region[at..]);
    out
}

/// One `sa[N]` occurrence in an emitted stage body.
struct SaUse {
    reg: usize,
    /// Byte range of the whole `sa[N]` text, so a read can be replaced in place.
    start: usize,
    end: usize,
    /// Followed by ` = `, i.e. this occurrence is the DESTINATION of an assignment. A
    /// read-modify-write store names the same register on both sides and produces one use of
    /// each kind, which is exactly right: the register is written, so it keeps its slot.
    written: bool,
}

/// Every `sa[N]` in `body`, or `None` if any occurrence is subscripted dynamically.
fn sa_uses(body: &str) -> Option<Vec<SaUse>> {
    let mut uses = Vec::new();
    let bytes = body.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = body[from..].find("sa") {
        let start = from + rel;
        from = start + 2;
        // A bank name is a whole identifier followed immediately by its subscript, so `vs_sa`,
        // `fs_sa` and `gxp_sa_k` are excluded by the two boundary tests.
        if start > 0 {
            let p = bytes[start - 1];
            if p.is_ascii_alphanumeric() || p == b'_' {
                continue;
            }
        }
        if bytes.get(from) != Some(&b'[') {
            continue;
        }
        let sub = &body[from + 1..];
        let digits = sub.len() - sub.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        if digits == 0 || sub.as_bytes().get(digits) != Some(&b']') {
            return None;
        }
        let end = from + 1 + digits + 1;
        uses.push(SaUse {
            reg: sub[..digits].parse().ok()?,
            start,
            end,
            // ` = ` and not ` == `: a comparison reads its left operand.
            written: body[end..].starts_with(" = "),
        });
        from = end;
    }
    Some(uses)
}
/// Diagnostic (`VITASLOP_GXP_DEPTH_PROBE=<lo>:<scale>`): EVERY fragment returns its own window
/// depth as a grey value, over the window `[lo, lo + 1/scale]`.
///
/// The question "why is this surface in front of that one" has no other instrument here: the
/// depth buffer is not readable, the CPU interpreter cannot run a skinned or texture-sampling
/// vertex program, and a picture only ever says WHICH draw won, never by how much. Render two
/// draw ranges with this and the two depths can be compared at the same pixel.
///
/// # It is a COMPARISON, not a number
/// The value goes through the display pass, which gamma-encodes and upscales it, so it is not a
/// depth to read off. That transform is MONOTONIC and identical for both renders, so the
/// comparison survives it - and the comparison is the whole question.
///
/// # Why it takes a window
/// A projection crams its whole scene into the far end of the depth buffer: one title's sits
/// above 0.98, where a plain 0..1 ramp saturates and answers nothing. `0.9:10` is what made a
/// character's depth and its floor's separable.
fn depth_probe() -> Option<&'static str> {
    arm("VITASLOP_GXP_DEPTH_PROBE").filter(|v| *v != "0")
}

/// Diagnostic (`VITASLOP_GXP_DEST_PROBE=1|opaque|mark`): a fragment program that reads the
/// DESTINATION colour returns what it READ instead of what it computed.
///
/// The destination path has two halves that fail the same way on screen - the renderer's copy
/// of the attachment (wrong texture, wrong moment, wrong extent) and the shader's own
/// arithmetic over it - and a black rectangle where a blend should be is the symptom of either.
///
/// * `1` returns the seeded colour, so a draw that paints the scene behind it was fed a real
///   destination and the fault is downstream.
/// * `opaque` returns it with alpha forced to 1, because a draw whose pipeline blend is
///   `SrcAlpha` and whose destination alpha is zero composites to NOTHING - which reads exactly
///   like a black copy.
/// * `mark` returns flat MAGENTA. It answers the question that has to come FIRST and that
///   neither of the others can: is this pixel painted by a destination reader at all? A black
///   rectangle looks the same whether the copy read black or the draw that made it never read a
///   destination, and an hour went into that difference before this arm existed.
///
/// Opt-IN: an unset variable leaves the shader alone. [`arm_on`] defaults to ON, which is right
/// for a default this crate ships and wrong for a diagnostic.
fn dest_probe() -> Option<&'static str> {
    arm("VITASLOP_GXP_DEST_PROBE").filter(|v| *v != "0")
}


/// `VITASLOP_GXP_SIZE_BANKS=0` restores the pre-2026-08-20b emission - every register bank
/// declared at the full [`BANK_REGS`] - and is the A/B arm for [`size_register_banks`].
///
/// VALUE-sensitive, because an arm has to be. Kept rather than deleted because the sizing is
/// the difference between a driver keeping a program's registers in registers and spilling them
/// to scratch, which is worth **10.13 -> 4.08 ms** of warm GPU render on one title's
/// on-track run here and is a bigger and less predictable number on a phone; a session that
/// measures a device needs to be able to take both arms without rebuilding.
pub(crate) fn arm_on(name: &str) -> bool {
    arm(name).map(|v| v != "0").unwrap_or(true)
}

/// An arm that has more than two positions, as a trimmed static string. `None` is the default.
///
/// # >>> IT MUST BE READABLE IN THE BROWSER, AND THAT IS NOT A CONVENIENCE
/// `wasm32-unknown-unknown` has no environment, so until [`set_arm`] existed both arms in this
/// file were hardwired ON in the browser and could not be taken there at all. The bug that made
/// this urgent - a phone whose driver refused four pipelines and drew a BLACK RACE - would have
/// been bisected in one run by `VITASLOP_GXP_SIZE_BANKS=0`, and instead cost an offline hunt
/// through the shader corpus. The engine that ships is the engine that has to be A/B-able.
///
/// Leaked deliberately and once per distinct value: this is read while emitting a shader, the
/// set of values is the set of arms a human can type, and the alternative is threading a
/// configuration struct through the whole emitter for a diagnostic.
pub(crate) fn arm(name: &str) -> Option<&'static str> {
    if let Some(v) = arms().lock().unwrap_or_else(|e| e.into_inner()).get(name) {
        return Some(v);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let v = std::env::var(name).ok()?;
        Some(Box::leak(v.trim().to_string().into_boxed_str()))
    }
    #[cfg(target_arch = "wasm32")]
    None
}

type Arms = std::collections::HashMap<String, &'static str>;

fn arms() -> &'static std::sync::Mutex<Arms> {
    static ARMS: std::sync::OnceLock<std::sync::Mutex<Arms>> = std::sync::OnceLock::new();
    ARMS.get_or_init(Default::default)
}

/// Set one of this crate's emission arms for a platform that has no environment to read it
/// from - the browser. Takes precedence over `std::env` where there is one, so a harness can
/// pin an arm without the process's environment agreeing.
///
/// The names are [`SIZE_BANKS_ARM`] and [`SA_DIRECT_ARM`]; anything else is ignored on purpose,
/// because this is called from a generic knob table and a typo there must not become an arm.
pub fn set_arm(name: &str, value: &str) {
    if !matches!(
        name,
        SIZE_BANKS_ARM
            | SA_DIRECT_ARM
            | MEM_OFFSET16_ARM
            | HALF_REGS_ARM
            | IDX_REGDEST_ARM
            | PACK_COMP0_ARM
            | IDX_MUL_ARM
            | F16_ROUND_ARM
            | CASE_TEX_ARM
            // >>> THE SHADER PROBES, for the same reason as the arms above and more urgently.
            //
            // These are the only instruments that can say WHICH term of a lit material is the
            // zero, and until now they were `std::env::var` reads - which means they could be
            // armed on the desktop and NOWHERE ELSE. That is backwards: a football title's
            // sideline characters render pure BLACK in the browser and are culled natively at
            // the same recipe frame, so the engine that can be probed is the engine that does
            // not have the defect [[vitaslop-web-is-the-product-not-the-tool]].
            | PROBE_ARM
            | PROBE_SCALE_ARM
            | VPROBE_ARM
    ) {
        return;
    }
    let v: &'static str = Box::leak(value.trim().to_string().into_boxed_str());
    arms().lock().unwrap_or_else(|e| e.into_inner()).insert(name.to_string(), v);
}

/// >>> WHICH EMISSION ARMS THIS RUN IS ON, FOR THE DIAGNOSTIC TO SAY OUT LOUD.
///
/// A capture taken on a device is read hours later, against a capture taken from a different
/// build, and NOTHING in the file says which arm produced it - so an A/B run as two visits
/// rests entirely on remembering which visit was which. That is not a control. Every arm set
/// away from its default is named here, and the empty case says so rather than printing
/// nothing, because "no line" and "no arms" have to be distinguishable in a downloaded file.
pub fn arms_line() -> String {
    let g = arms().lock().unwrap_or_else(|e| e.into_inner());
    let mut set: Vec<String> = g.iter().map(|(k, v)| format!("{k}={v}")).collect();
    set.sort();
    if set.is_empty() {
        "no shader emission arm is set - this is the DEFAULT emitter".to_string()
    } else {
        format!("ARMED: {}", set.join(", "))
    }
}

/// `0` declares every register bank at the full [`BANK_REGS`] - see [`size_register_banks`].
pub const SIZE_BANKS_ARM: &str = "VITASLOP_GXP_SIZE_BANKS";
/// `0` restores the SA copy loop, `unroll` the constant-subscript copy - see [`resolve_sa_init`].
pub const SA_DIRECT_ARM: &str = "VITASLOP_GXP_SA_DIRECT";
/// `1` gives a 16-bit register an UNPACKED home ([`unpack_half_registers`]); unset or `0` keeps
/// every half read and write a conversion, which is the default and what ships. `noexact` is the
/// middle position: the unpacked home with every store rounded (see [`f16_exact`]).
pub const HALF_REGS_ARM: &str = "VITASLOP_GXP_HALF_REGS";

/// >>> HOW AN f32 NARROWS TO AN f16 - THE NEGATIVE CONTROL FOR THE ROUNDING FIX.
///
/// `0` puts `pack2x16float` back under the store helpers, which is **exactly what every build
/// before 2026-09-21c shipped**: the device's own implementation-defined mode, measured here as
/// truncation. Unset (the default) is round-to-nearest-even, which is what the CPU reference
/// does and what the guest's hardware does.
///
/// It is the arm a picture A/B needs, and it is a ONE-HUNK arm by construction: the emitted
/// BODY is identical in both, and only the module preamble's definition of
/// [`crate::wgsl::HALF_BITS_FN`] changes. Nothing else can drift between the two arms, which is
/// what makes a pixel difference attributable [[vitaslop-knob-is-the-gate-not-the-level]].
///
/// `native` and `portable` FORCE one of the two round-to-nearest arms instead of letting the
/// adapter choose. They exist so a case harness can run both on one device: the two arms must
/// agree bit for bit, and the only thing that can say so is a run of each.
pub const F16_ROUND_ARM: &str = "VITASLOP_GXP_F16_RTE";

/// `const` makes the execution rig's STAND-IN TEXTURE a constant again. It is a function of the
/// sample COORDINATE by default, which is the only thing in the differential that can see a
/// program computing the wrong UV; the constant arm is what every measurement before
/// 2026-09-21c used and is kept as the comparison. See [`crate::wgsl::case_tex_varies`].
pub const CASE_TEX_ARM: &str = "VITASLOP_GXP_CASE_TEX";

/// `0` reads a memory load's REGISTER offset full-width instead of as 16 bits - see
/// `wgsl::emit_mem_load`, which carries the measurement.
pub const MEM_OFFSET16_ARM: &str = "VITASLOP_GXP_MEM_OFFSET16";


/// `<bank><idx>[@<instr>][:f32|:bits=<hex>]` - return that register AS the colour. See
/// [`crate::module::ProbeSpec`].
pub const PROBE_ARM: &str = "VITASLOP_GXP_PROBE";
/// Divide a probed value before it is written, so an HDR term reads back under the
/// attachment's [0,1] clamp.
pub const PROBE_SCALE_ARM: &str = "VITASLOP_GXP_PROBE_SCALE";
/// `<n>` - return interpolated varying `v<n>` AS the colour.
pub const VPROBE_ARM: &str = "VITASLOP_GXP_VPROBE";
/// `1` - WHERE DID THIS DRAW'S VERTICES GO? Draw the mesh at CLAMPED normalised coordinates and
/// report each vertex's real clip position through varying 0.
///
/// A draw that rasterises no fragment at all - `VITASLOP_GXM_DRAW_COVERAGE` counts them by the
/// thousand, and a capsule reproduces one offline - says nothing about WHY through any other
/// instrument: every colour arm is downstream of a triangle that never existed. `SOLID` and
/// `NODEPTH` separate shading from the depth test and both come back empty on such a draw,
/// which leaves the vertex stage - whose output is the one value nothing here could read.
///
/// `v0` carries it, where `VITASLOP_GXP_VPROBE=0` already knows how to show it: `x` and `y` are
/// the NDC mapped into [0,1] (0.5 is the centre, a saturated channel is off-screen in that
/// axis), `z` is 1.0 when clip `w` is POSITIVE (0.0 is behind the eye, which no perspective
/// divide brings back) and the alpha lane is 0.0 for a NaN. A collapsed mesh is then one flat
/// colour and a merely misplaced one is a picture of where it went.
pub const POSPROBE_ARM: &str = "VITASLOP_GXP_POSPROBE";

/// Whether the vertex POSITION PROBE is armed - see [`POSPROBE_ARM`]. OFF unless asked for:
/// `arm_on` defaults to TRUE, which is right for a default-on arm and catastrophic for a
/// diagnostic that replaces every vertex position in the frame.
fn posprobe_on() -> bool {
    arm(POSPROBE_ARM).is_some_and(|v| v.trim() != "0")
}

/// >>> HOW FAR OFF-SCREEN, NOT ONLY THAT IT IS OFF-SCREEN. The probe maps NDC into `[0, 1]` and
/// >>> CLAMPS, so every vertex past the viewport paints the same saturated channel whether it
/// >>> overshot by a tenth or by a thousand. A picture that cannot tell those apart cannot say
/// >>> whether the cause is a wrong offset or a wrong scale.
///
/// `VITASLOP_GXP_POSPROBE_DIV=<d>` divides the NDC by `d` before the map, so the first divisor
/// at which a rail comes off names the OVERSHOOT'S ORDER OF MAGNITUDE. Run 1, 10, 100, 1000:
/// geometry that is merely outside the frustum comes back at 10, and geometry whose transform
/// has lost a scale does not come back at all.
///
/// It divides rather than taking a log because the sign has to survive - "above the viewport"
/// and "below it" are different defects, and a log of a negative number is not a picture.
pub const POSPROBE_DIV_ARM: &str = "VITASLOP_GXP_POSPROBE_DIV";

/// [`POSPROBE_DIV_ARM`]'s value, or 1.0. A zero or a negative divisor would turn every vertex
/// into an infinity or mirror the picture, so both fall back to 1.0 rather than being applied.
fn posprobe_divisor() -> f32 {
    arm(POSPROBE_DIV_ARM)
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|d| d.is_finite() && *d > 0.0)
        .unwrap_or(1.0)
}


/// `0` sends EVERY group-0x14 index load to the index register, which is what this decoder did
/// before the `(b8,b51)` flag was read as naming the destination - see `decode_grp_i16mad`.
///
/// It exists so the whole-corpus WGSL census can be taken twice on ONE build: the words the
/// change touches live in a single title, and "no other corpus can move" is a claim that has to
/// be MEASURED rather than argued from which bits are set.
pub const IDX_REGDEST_ARM: &str = "VITASLOP_GXP_IDX_REGDEST";

/// `0` restores bit 1 as comp0's high selector bit for a 16-bit PACK source, which is what this
/// decoder read before the corpus said bit 7 - see `decode_grp_pack`. Here for the same reason
/// as [`IDX_REGDEST_ARM`]: one title's picture and another's regression have to be A/B-able on
/// ONE build.
pub const PACK_COMP0_ARM: &str = "VITASLOP_GXP_PACK_COMP0";

/// The multiplier the ORDINARY-REGISTER index load applies to its source, as a decimal string.
/// Default 1, which is what the corpus grammar decodes today.
///
/// >>> IT IS A QUESTION, NOT A SETTING. Madden's matrix palette is THREE float4 rows per bone
/// (measured: rows 0,1,2 of the bound window are one affine transform, 3,4,5 the next, and so
/// on for every group), and its `IN.blendIndices` are plain consecutive BONE numbers - 31, 32,
/// 33 - so the row a bone's matrix starts at is `3 * index`, not `index`. No field of the
/// group-0x14 word has been shown to carry that 3. This arm is how the picture is asked whether
/// the factor is real before anything is claimed about where it is encoded.
pub const IDX_MUL_ARM: &str = "VITASLOP_GXP_IDX_MUL";

/// [`IDX_MUL_ARM`]'s value as a multiplier, or `None` when it is unset - in which case the
/// stride the program itself carries (`usse::resolve_index_load_stride`) is used.
pub(crate) fn index_load_multiplier() -> Option<i32> {
    arm(IDX_MUL_ARM).and_then(|v| v.parse().ok()).filter(|n: &i32| *n > 0)
}

/// The marker a stage's register-bank declarations are emitted as, resolved to real sizes by
/// [`size_register_banks`] once the whole module is built and every subscript is known.
const BANKS_MARKER: &str = "  //@@GXP_REGISTER_BANKS\n";

/// Emit the per-entry-point USSE register-file locals (raw 32-bit registers, matching the
/// emitter): the `r`/`o`/`i`/`pa`/`sa` banks plus the predicate registers.
///
/// Emits a MARKER, not the declarations. See [`size_register_banks`] for why the sizes cannot
/// be known here.
fn emit_register_banks(m: &mut String) {
    m.push_str(BANKS_MARKER);
    let _ = writeln!(m, "  var p: array<bool, 4>;");
    // The INDEX register file, for register-INDIRECT operands. Two registers, because the
    // extension row names exactly two indexed banks (INDEXED1 -> i0, INDEXED2 -> i1).
    let _ = writeln!(m, "  var idx: array<i32, 2>;");
}

/// Replace each stage's bank marker with declarations sized to what that stage's emitted code
/// actually subscripts.
///
/// # Why this is a text pass and not a walk of the IR
/// [`BANK_REGS`] is 512 per bank, five banks, per entry point - 10 KB of function-local storage
/// in every module we hand a driver, where a real program touches at most a couple of dozen
/// registers. That is the shape the USSE register file has, not the shape the program has, and
/// the driver pays for it twice: once compiling (it has to prove 2,560 slots dead before it can
/// keep the live ones in registers) and once at runtime, where a bank it fails to promote
/// becomes per-invocation scratch memory. A phone GPU is where that second cost lands.
///
/// The bound is taken from the EMITTED TEXT rather than re-derived from the instruction stream
/// because the text is the only thing that cannot be wrong: register indices are resolved by
/// the emitter through the decoder's doubling, swizzle selectors, write masks, half-precision
/// pairing and the linker's own prologue/epilogue, and a second implementation of those rules
/// that disagreed by one would corrupt a shader silently. Scanning `bank[N]` counts exactly
/// what the module references.
///
/// # Why under-sizing cannot corrupt a shader
/// A constant subscript past the end of a WGSL array is a VALIDATION ERROR, so a reference this
/// scan missed fails the module loudly at `create_shader_module` rather than reading a
/// neighbour. The one form that would fail silently is a DYNAMIC subscript - `indexed_element`
/// clamps to `BANK_REGS - 1`, and a smaller array would fold every high index onto its last
/// element - so a bank with any dynamic subscript keeps its full size. That is the whole safety
/// argument: constant indices are checked by the compiler, dynamic ones are not shrunk.
fn size_register_banks(module: &str) -> String {
    const BANKS: [&str; 5] = ["r", "o", "i", "pa", "sa"];
    let sized = arm_on("VITASLOP_GXP_SIZE_BANKS");
    let mut out = String::with_capacity(module.len());
    let mut rest = module;
    while let Some(at) = rest.find(BANKS_MARKER) {
        out.push_str(&rest[..at]);
        let body = &rest[at + BANKS_MARKER.len()..];
        // A stage's references end where the next stage's declarations begin.
        let region = match body.find(BANKS_MARKER) {
            Some(next) => &body[..next],
            None => body,
        };
        for bank in BANKS {
            match if sized { bank_extent(region, bank) } else { None } {
                Some(0) => {} // never referenced - declaring it would be dead storage
                Some(n) => {
                    let _ = writeln!(out, "  var {bank}: array<u32, {n}>;");
                }
                None => {
                    let _ = writeln!(out, "  var {bank}: array<u32, {BANK_REGS}>;");
                }
            }
            // The UNPACKED half-register home, when [`unpack_half_registers`] gave this bank
            // one. Always sized, even with the sizing arm off: this array exists only because
            // that pass put literal subscripts in it, so it has no dynamic form to be safe
            // about, and declaring 512 unused `vec2<f32>` would undo the cut it is part of.
            if let Some(n @ 1..) = bank_extent(region, &format!("{bank}{HALF_SUFFIX}")) {
                let _ = writeln!(out, "  var {bank}{HALF_SUFFIX}: array<vec2<f32>, {n}>;");
            }
        }
        rest = body;
    }
    out.push_str(rest);
    // WGSL wants a function declared before it is called, so the rounding helper goes at the
    // TOP of the module rather than beside the pass that introduces its calls - but AFTER the
    // directives, because `enable`/`requires`/`diagnostic` must precede every declaration in the
    // module. Putting it at byte zero instead cost a whole title its picture: a dual-source pair
    // carries `enable dual_source_blending;`, the module then read as a directive after a
    // declaration, the device refused the pipeline, and the frame went BLACK.
    if out.contains("gxp_q2(") {
        out.insert_str(directives_end(&out), GXP_Q2);
    }
    // ...and the f16 STORE helpers, which `gxp_q2` is itself written in terms of - so they must
    // be inserted AFTER it, because each insertion goes at the same place and the last one in
    // ends up first. This is the last pass every linked module goes through, which is what makes
    // it the one place the helpers can be added once rather than at each of the five returns.
    add_half_helpers(out)
}

/// How many registers of `bank` the emitted `region` references: `Some(high_water + 1)`, or
/// `None` when it is subscripted dynamically and therefore cannot be bounded here.
fn bank_extent(region: &str, bank: &str) -> Option<usize> {
    let mut high = 0usize;
    let mut any = false;
    let bytes = region.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = region[from..].find(bank) {
        let start = from + rel;
        from = start + bank.len();
        // A bank name is a whole identifier followed immediately by its subscript, so `idx[`,
        // `in.`, `gxp_sa_k` and every other identifier containing these letters are excluded by
        // the two boundary tests.
        if start > 0 {
            let p = bytes[start - 1];
            if p.is_ascii_alphanumeric() || p == b'_' {
                continue;
            }
        }
        if bytes.get(from) != Some(&b'[') {
            continue;
        }
        let sub = &region[from + 1..];
        let digits = sub.len() - sub.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        if digits > 0 && sub.as_bytes().get(digits) == Some(&b']') {
            let n: usize = sub[..digits].parse().ok()?;
            high = high.max(n);
            any = true;
            continue;
        }
        // The default-uniform copy loop is bounded by its own literal; every other dynamic
        // subscript is not bounded at all.
        if let Some(tail) = sub.strip_prefix("gxp_sa_k]") {
            let _ = tail;
            if let Some(bound) = uniform_loop_bound(region) {
                high = high.max(bound.saturating_sub(1));
                any = true;
                continue;
            }
        }
        return None;
    }
    Some(if any { high + 1 } else { 0 })
}

/// The literal register count the SA copy loop emitted by [`emit_secondary_attrs`] runs to.
fn uniform_loop_bound(region: &str) -> Option<usize> {
    let at = region.find("gxp_sa_k < ")? + "gxp_sa_k < ".len();
    let sub = &region[at..];
    let digits = sub.len() - sub.trim_start_matches(|c: char| c.is_ascii_digit()).len();
    sub[..digits].parse().ok()
}

/// The bank sizing is a TEXT pass over emitted WGSL, and the one failure mode that would not
/// announce itself is shrinking a bank a DYNAMIC subscript still clamps to `BANK_REGS - 1`.
/// Every case here is about that boundary.
#[cfg(test)]
mod bank_sizing_tests {
    use super::*;

    fn one_stage(body: &str) -> String {
        let mut m = String::from("@vertex\nfn vs_main() -> VsOut {\n");
        emit_register_banks(&mut m);
        m.push_str(body);
        m.push_str("}\n");
        size_register_banks(&m)
    }

    #[test]
    fn a_bank_is_sized_to_its_highest_constant_subscript() {
        let out = one_stage("  r[3] = pa[0];\n  o[11] = r[3];\n");
        assert!(out.contains("var r: array<u32, 4>;"), "{out}");
        assert!(out.contains("var pa: array<u32, 1>;"), "{out}");
        assert!(out.contains("var o: array<u32, 12>;"), "{out}");
    }

    #[test]
    fn a_bank_nothing_subscripts_is_not_declared_at_all() {
        let out = one_stage("  r[0] = 1u;\n");
        assert!(out.contains("var r: array<u32, 1>;"), "{out}");
        assert!(!out.contains("var sa:"), "an unreferenced bank is dead storage: {out}");
        assert!(!out.contains("var i:"), "{out}");
    }

    #[test]
    fn a_dynamically_subscripted_bank_keeps_its_full_size() {
        // This is the emission of `wgsl::indexed_element`, whose clamp names `BANK_REGS - 1`:
        // a smaller array would fold every high index onto its last element SILENTLY.
        let out = one_stage("  r[0] = sa[min(u32(max(idx[0] + 2i, 0i)), 511u)];\n");
        assert!(out.contains(&format!("var sa: array<u32, {BANK_REGS}>;")), "{out}");
        assert!(out.contains("var r: array<u32, 1>;"), "the other banks still size: {out}");
    }

    /// A default-uniform register the body only READS never reaches the bank at all: it is read
    /// straight out of the uniform, so nineteen declared registers become none.
    #[test]
    fn a_read_only_uniform_register_is_read_from_the_uniform_and_not_copied() {
        let mut m = String::from("@vertex\nfn vs_main() -> VsOut {\n");
        emit_register_banks(&mut m);
        emit_secondary_attrs(&mut m, "vs_sa", 19, &[]);
        m.push_str("  o[0] = sa[2];\n}\n");
        let out = size_register_banks(&resolve_sa_init(&m));
        assert!(out.contains("o[0] = vs_sa.data[0][2];"), "{out}");
        assert!(!out.contains("gxp_sa_k"), "the copy loop is gone: {out}");
        assert!(!out.contains("var sa:"), "nothing subscripts the bank now: {out}");
    }

    /// One the body WRITES keeps a slot, loaded once - and the slot is renumbered, because the
    /// register it came from is only an index into the guest's uniform buffer.
    #[test]
    fn a_written_uniform_register_keeps_one_compacted_slot() {
        let mut m = String::from("@vertex\nfn vs_main() -> VsOut {\n");
        emit_register_banks(&mut m);
        emit_secondary_attrs(&mut m, "vs_sa", 19, &[]);
        m.push_str("  sa[17] = 1u;\n  o[0] = sa[17];\n  o[1] = sa[2];\n}\n");
        let out = size_register_banks(&resolve_sa_init(&m));
        assert!(out.contains("var sa: array<u32, 1>;"), "one live register: {out}");
        assert!(out.contains("sa[0] = vs_sa.data[4][1];"), "loaded before its write: {out}");
        assert!(out.contains("o[1] = vs_sa.data[0][2];"), "the read-only one is direct: {out}");
    }

    /// A register the body OVERWRITES with a function of itself keeps every one of its reads on
    /// the local slot - including the ones after the write.
    ///
    /// This is the case that made a retail race come back blown out: the
    /// substitution was decided per OCCURRENCE, so `sa[5]`'s later read was rewritten to the
    /// uniform and the reciprocal computed into it was thrown away. Nothing about that fails to
    /// compile.
    #[test]
    fn a_register_the_body_recomputes_is_not_substituted_after_its_write() {
        let mut m = String::from("@vertex\nfn vs_main() -> VsOut {\n");
        emit_register_banks(&mut m);
        emit_secondary_attrs(&mut m, "vs_sa", 19, &[]);
        m.push_str("  sa[5] = bitcast<u32>(1.0 / bitcast<f32>(sa[5]));\n  o[0] = sa[5];\n}\n");
        let out = size_register_banks(&resolve_sa_init(&m));
        assert!(out.contains("sa[0] = vs_sa.data[1][1];"), "{out}");
        assert!(out.contains("o[0] = sa[0];"), "the later read stays on the slot: {out}");
        assert_eq!(out.matches("vs_sa.data").count(), 1, "only the entry load reads it: {out}");
    }

    /// A stage that indexes the bank DYNAMICALLY cannot have either rewrite proved safe, so it
    /// keeps the copy loop and the loop's own literal bounds the bank.
    #[test]
    fn a_dynamically_indexed_stage_keeps_the_uniform_copy_loop() {
        let mut m = String::from("@vertex\nfn vs_main() -> VsOut {\n");
        emit_register_banks(&mut m);
        emit_secondary_attrs(&mut m, "vs_sa", 19, &[]);
        m.push_str("  o[0] = sa[min(u32(max(idx[0] + 2i, 0i)), 511u)];\n}\n");
        let out = size_register_banks(&resolve_sa_init(&m));
        assert!(out.contains("gxp_sa_k < 19u"), "{out}");
        // And the bank keeps its FULL size, because `indexed_element`'s clamp names
        // `BANK_REGS - 1` - the case `a_dynamically_subscripted_bank_keeps_its_full_size`
        // covers. The loop is what makes the copy correct for an index nothing here bounds.
        assert!(out.contains(&format!("var sa: array<u32, {BANK_REGS}>;")), "{out}");
    }

    /// The literals the CONTAINER carries are stored at whatever register the blob's own
    /// compiler chose, which in the retail corpus is register 92 of 107. They are live and must
    /// survive - compacted, like every other survivor.
    #[test]
    fn container_literals_survive_the_compaction() {
        let mut m = String::from("@vertex\nfn vs_main() -> VsOut {\n");
        emit_register_banks(&mut m);
        emit_secondary_attrs(&mut m, "vs_sa", 19, &[(92, 0x3f80_0000)]);
        m.push_str("  o[0] = sa[92];\n}\n");
        let out = size_register_banks(&resolve_sa_init(&m));
        assert!(out.contains("var sa: array<u32, 1>;"), "{out}");
        assert!(out.contains("sa[0] = 0x3f800000u;"), "{out}");
        assert!(out.contains("o[0] = sa[0];"), "{out}");
    }

    #[test]
    fn the_two_stages_of_a_module_are_sized_independently() {
        let mut m = String::from("@vertex\nfn vs_main() -> VsOut {\n");
        emit_register_banks(&mut m);
        m.push_str("  o[30] = 1u;\n}\n@fragment\nfn fs_main() -> @location(0) vec4<f32> {\n");
        emit_register_banks(&mut m);
        m.push_str("  o[1] = 1u;\n}\n");
        let out = size_register_banks(&m);
        assert!(out.contains("var o: array<u32, 31>;"), "{out}");
        assert!(out.contains("var o: array<u32, 2>;"), "the fragment must not inherit 31: {out}");
    }

    #[test]
    fn an_identifier_that_merely_contains_a_bank_name_is_not_one() {
        // `idx[..]` and `in.a0` both contain bank letters; neither is a bank subscript, and
        // reading one as a dynamic index would silently give up the sizing for that bank.
        let out = one_stage("  idx[0] = 2i;\n  pa[1] = bitcast<u32>(in.a0.x);\n");
        assert!(out.contains("var pa: array<u32, 2>;"), "{out}");
        assert!(!out.contains("var i:"), "`idx` is not the `i` bank: {out}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{
        Interpolant, OutputVarying, ParamCategory, ParamType, Parameter, ProgramKind, SamplePrefetch,
    };
    use crate::ir::{Instr, Op, Operand, Predicate};

    /// A module region as [`unpack_half_registers`] sees one: the bank marker, then statements.
    ///
    /// The pass is OFF by default ([`half_regs_on`]), so a test of it has to ARM it. Every test
    /// here wants it on and none wants it off, so arming the process-wide table is enough and
    /// cannot race with a test that expects the other arm.
    fn half_region(body: &str) -> String {
        set_arm(HALF_REGS_ARM, "1");
        format!("{BANKS_MARKER}{body}")
    }

    /// A register whose every use is a 16-bit half moves to the unpacked home, and the two forms
    /// that cost the most - the read and the single-half store - stop converting entirely.
    #[test]
    fn a_register_used_only_as_halves_moves_to_its_unpacked_home() {
        let out = unpack_half_registers(&half_region(
            "  pa[0] = gxp_hpk(in.v0.x, in.v0.y);\n\
             \x20 r[1] = gxp_hlo(r[1], (unpack2x16float(pa[0])[0] * unpack2x16float(pa[0])[1]));\n\
             \x20 o[0] = gxp_hpk(unpack2x16float(r[1])[0], unpack2x16float(r[1])[1]);\n\
             \x20 let c = vec2<f32>(unpack2x16float(o[0])[0], unpack2x16float(o[0])[1]);\n",
        ));
        assert!(out.contains("pa_h[0] = gxp_q2(vec2<f32>(in.v0.x, in.v0.y));"), "{out}");
        assert!(out.contains("r_h[1][0] = gxp_hq((pa_h[0][0] * pa_h[0][1]));"), "{out}");
        // A store whose value is already a 16-bit one rounds nothing, so it does not round.
        assert!(out.contains("o_h[0] = vec2<f32>(r_h[1][0], r_h[1][1]);"), "{out}");
        assert!(!out.contains("unpack2x16float(pa[0])"), "no read may stay packed:\n{out}");
    }

    /// ...and a register the program ALSO reads as a 32-bit float does not, because the two
    /// views are one word and only the packed form can hold both.
    #[test]
    fn a_register_read_at_two_precisions_stays_packed() {
        let body = "  pa[0] = gxp_hpk(in.v0.x, in.v0.y);\n\
                    \x20 r[0] = bitcast<u32>(unpack2x16float(pa[0])[0] * bitcast<f32>(pa[0]));\n";
        let out = unpack_half_registers(&half_region(body));
        assert_eq!(out, half_region(body), "an f32 view disqualifies the register:\n{out}");
    }

    /// A WHOLE-register write ends the previous value's life, so an f32 use after one does not
    /// cost the half uses before it. This is the difference between moving one register and
    /// losing fifty reads to a scratch use at the end of a shader.
    #[test]
    fn a_whole_register_write_splits_the_live_range() {
        let out = unpack_half_registers(&half_region(
            "  pa[2] = gxp_hpk(in.v0.x, in.v0.y);\n\
             \x20 r[0] = gxp_hlo(r[0], unpack2x16float(pa[2])[1]);\n\
             \x20 pa[2] = bitcast<u32>(in.v1.x);\n\
             \x20 let c = bitcast<f32>(pa[2]);\n",
        ));
        assert!(out.contains("pa_h[2] = gxp_q2(vec2<f32>(in.v0.x, in.v0.y));"), "{out}");
        // `r[0]` is never read here, so it stays where it is - what moved is the READ of the
        // first range of `pa[2]`, which is the point.
        assert!(out.contains("gxp_hlo(r[0], pa_h[2][1])"), "{out}");
        // The second range keeps the packed word - it is the one the `f32` read needs.
        assert!(out.contains("pa[2] = bitcast<u32>(in.v1.x);"), "{out}");
        assert!(out.contains("let c = bitcast<f32>(pa[2]);"), "{out}");
    }

    /// A whole-register write inside a CONDITIONAL does not end anything: the path that skips it
    /// carries the old value past, so both uses are one range and the f32 read disqualifies it.
    #[test]
    fn a_conditional_write_does_not_split_a_live_range() {
        let body = "  pa[2] = gxp_hpk(in.v0.x, in.v0.y);\n\
                    \x20 if (p[0]) {\n\
                    \x20 pa[2] = bitcast<u32>(in.v1.x);\n\
                    \x20 }\n\
                    \x20 let c = bitcast<f32>(pa[2]) + unpack2x16float(pa[2])[0];\n";
        let out = unpack_half_registers(&half_region(body));
        assert_eq!(out, half_region(body), "a conditional write cannot open a range:\n{out}");
    }

    /// >>> THE DUAL-SOURCE SPLIT RUNS ITS TAIL TWICE AND THE RANGES REWIND WITH IT.
    ///
    /// The second copy reads what the PREFIX left, not what the first copy wrote. Carrying the
    /// first copy's range ids into the second put a read in a range that never reached it, and
    /// when one of those ranges lived unpacked and the other packed, the second copy read a word
    /// nothing had written: mlb's whole frame came back BLACK.
    #[test]
    fn the_dual_source_split_rewinds_the_ranges_with_the_register_file() {
        let out = unpack_half_registers(&half_region(
            "  pa[0] = gxp_hpk(in.v0.x, in.v0.y);\n\
             \x20 let gxp_save_pa = pa;\n\
             \x20 {\n\
             \x20 r[0] = gxp_hlo(r[0], unpack2x16float(pa[0])[0]);\n\
             \x20 pa[0] = bitcast<u32>(in.v1.x);\n\
             \x20 }\n\
             \x20 pa = gxp_save_pa;\n\
             \x20 {\n\
             \x20 r[0] = gxp_hlo(r[0], unpack2x16float(pa[0])[0]);\n\
             \x20 pa[0] = bitcast<u32>(in.v1.x);\n\
             \x20 }\n",
        ));
        // Both copies read the same home - the prefix's - rather than one reading a word the
        // other never wrote.
        assert_eq!(out.matches("pa_h[0][0]").count(), 2, "{out}");
        // And the save/restore covers the home, not just the packed array it no longer uses.
        assert!(out.contains("let gxp_save_pa_h = pa_h;"), "{out}");
        assert!(out.contains("pa_h = gxp_save_pa_h;"), "{out}");
    }

    /// The rounding helper is a DECLARATION, and every WGSL directive has to precede those. A
    /// dual-source module carries `enable dual_source_blending;`, and putting the helper at byte
    /// zero made the module unparseable, the device refuse the pipeline, and the frame go black.
    #[test]
    fn the_rounding_helper_goes_after_the_directives() {
        let module = format!(
            "enable dual_source_blending;\n\n@group(0) @binding(0) var<uniform> u: vec4<u32>;\n{}",
            half_region("  pa[0] = gxp_hpk(in.v0.x, in.v0.y);\n  let a = unpack2x16float(pa[0])[0];\n")
        );
        let out = size_register_banks(&unpack_half_registers(&module));
        let (directive, helper) = (out.find("enable dual_source_blending;"), out.find("fn gxp_q2("));
        assert!(directive < helper, "the helper must follow the directives:\n{out}");
        assert!(helper.is_some(), "the helper must be emitted when it is called:\n{out}");
        assert!(out.contains("var pa_h: array<vec2<f32>, 1>;"), "the home is declared:\n{out}");
    }

    /// A bank with a DYNAMIC subscript keeps every one of its registers packed: there is no
    /// telling which of them the index reaches, so none of them can move.
    #[test]
    fn a_dynamic_subscript_keeps_its_whole_bank_packed() {
        let body = "  pa[0] = gxp_hpk(in.v0.x, in.v0.y);\n\
                    \x20 let a = unpack2x16float(pa[0])[0];\n\
                    \x20 let b = pa[idx[0]];\n";
        let out = unpack_half_registers(&half_region(body));
        assert_eq!(out, half_region(body), "a dynamic subscript disqualifies the bank:\n{out}");
    }

    fn instr(op: Op, dest: Option<Operand>, srcs: Vec<Operand>, mask: [bool; 4]) -> Instr {
        Instr { op, pred: Predicate::Always, dest, write_mask: mask, srcs, half_precision: false, raw: 0, group: 0, blocked: None }
    }

    fn shader(kind: ProgramKind, instrs: Vec<Instr>) -> Shader {
        Shader { kind, instrs }
    }

    /// A claim WIDER than the usage it names places nothing - the baseball title's stadium
    /// family, and the shape `claim_width_test` exists for.
    ///
    /// `In.UV1` declares four components and is bound `F16x2`, so its copy fills lanes 4..8 with
    /// the last two carrying the missing-component fill. The claim therefore spans four lanes
    /// and names the TWO-lane `TexCoord(1)`. Honouring the start alone gives
    /// `TexCoord(1)@4x2 Color0@6x10 TexCoord(0)@10x2` - which puts the ALBEDO prefetch's
    /// coordinate on lanes 10 and 11, lanes this vertex program never writes, so every stadium
    /// surface samples its texture at a constant (0,0). MEASURED against the frame: with the
    /// claim refused the same frame draws the stadium's signage, brickwork and scoreboard.
    #[test]
    fn a_claim_wider_than_the_usage_it_names_places_nothing() {
        let vout = vec![
            OutputVarying { usage: VaryingUsage::Color0, base_lane: 4, components: 4 },
            OutputVarying { usage: VaryingUsage::TexCoord(0), base_lane: 8, components: 2 },
            OutputVarying { usage: VaryingUsage::TexCoord(1), base_lane: 10, components: 2 },
            OutputVarying { usage: VaryingUsage::TexCoord(2), base_lane: 12, components: 2 },
        ];
        let claims = vec![(VaryingUsage::TexCoord(1), vec![4, 5, 6, 7])];
        assert_eq!(
            layout_from_forwarding_claims(&vout, &claims, false),
            None,
            "a four-lane claim cannot place a two-lane varying"
        );
        // The same claim over a run its usage CAN hold still places it, so the test is about the
        // width and not about refusing every claim that names a texcoord.
        let claims_fitting = vec![(VaryingUsage::TexCoord(1), vec![4, 5])];
        assert!(layout_from_forwarding_claims(&vout, &claims_fitting, false).is_some());
    }

    #[test]
    fn a_forwarding_claim_places_its_usage_at_the_lane_the_vertex_fills() {
        // One title's sky/background family. The convention gives
        // `Color0@4x4 Color1@8x4 TexCoord(0)@12x2`; the code copies `In.UV1` - a TEXCOORD - from
        // lane 8, and copies `In.VColor` into lanes 12..14, which no four-lane `Color0` can ever
        // start at. The reachable claim settles the order and the unreachable one is ignored: an
        // earlier version demanded BOTH and therefore refused the very layout this family states.
        //
        // >>> THE CLAIM'S RUN IS THE WIDTH OF THE USAGE IT NAMES, and that is now REQUIRED - see
        // >>> `claim_width_test`. The run recorded here was four lanes because the copy fills
        // >>> four; the usage it names is two wide, and honouring the start alone hands the other
        // >>> two to a different varying, contradicting the copy the claim was read from. A
        // >>> baseball title's stadium family is that shape and comes out flat and banded under
        // >>> the start-only rule (its albedo prefetch lands on lanes the vertex never writes).
        // >>> The run here is two lanes, which is what the rule admits and what the resolution
        // >>> below needs; `VITASLOP_GXP_CLAIM_WIDTH=0` is the arm back to the start-only reading
        // >>> if this family is ever measured to need four.
        let vout = vec![
            OutputVarying { usage: VaryingUsage::Color0, base_lane: 4, components: 4 },
            OutputVarying { usage: VaryingUsage::Color1, base_lane: 8, components: 4 },
            OutputVarying { usage: VaryingUsage::TexCoord(0), base_lane: 12, components: 2 },
        ];
        let claims = vec![
            (VaryingUsage::TexCoord(0), vec![8, 9]),
            (VaryingUsage::Color0, vec![12, 13]),
        ];
        let got = layout_from_forwarding_claims(&vout, &claims, false)
            .expect("the TexCoord claim is reachable");
        assert_eq!(
            got,
            vec![
                OutputVarying { usage: VaryingUsage::Color0, base_lane: 4, components: 4 },
                OutputVarying { usage: VaryingUsage::TexCoord(0), base_lane: 8, components: 2 },
                OutputVarying { usage: VaryingUsage::Color1, base_lane: 10, components: 4 },
            ]
        );

        // No usage's WIDTH moves, so the lane budget still closes exactly - the objection that
        // sank the previous attempt at resolving this.
        assert_eq!(got.iter().map(|v| v.components).sum::<u32>(), 10);

        // A claim the convention already satisfies changes nothing, and "nothing" is reported as
        // no resolution rather than as a layout.
        assert!(
            layout_from_forwarding_claims(&vout, &[(VaryingUsage::Color0, vec![4, 5, 6, 7])], false)
                .is_none()
        );
    }

    #[test]
    fn a_forwarding_claim_overrules_a_layout_the_attributes_placed() {
        // >>> THE ATTRIBUTE ORDER IS A READING OF A PASSTHROUGH PROGRAM, AND THIS PROGRAM IS NOT
        // >>> ONE. MEASURED on a retail title's UI vertex program (`vert_81a7e8b8`): its
        // attributes are `aPosition`@0, `aTexCoord`@4, `aColor`@8, so
        // `container::attribute_order` covers the declared set exactly and reports
        // `TexCoord(0)@4x2 Color0@6x4` as `VaryingOrder::Known`. The CODE moves `aColor`
        // straight into lanes 4..8 and writes the texcoord it divides by `uTexture` into 8..10 -
        // so the cover is satisfied by a program that COMPUTES one of the two varyings, and the
        // inference is simply wrong.
        //
        // It cost the title its art: the paired fragment prefetches its sampler from TEXCOORD 0,
        // which under that layout is the vertex COLOUR (a constant 1,1), so every sprite sampled
        // one flat texel. `plan_interface` therefore asks the claims about a `Known` order too,
        // and this is the shape it has to resolve.
        let vout = vec![
            OutputVarying { usage: VaryingUsage::TexCoord(0), base_lane: 4, components: 2 },
            OutputVarying { usage: VaryingUsage::Color0, base_lane: 6, components: 4 },
        ];
        let got =
            layout_from_forwarding_claims(&vout, &[(VaryingUsage::Color0, vec![4, 5, 6, 7])], false)
            .expect("the colour attribute fills the lane the layout gives TexCoord(0)");
        assert_eq!(
            got,
            vec![
                OutputVarying { usage: VaryingUsage::Color0, base_lane: 4, components: 4 },
                OutputVarying { usage: VaryingUsage::TexCoord(0), base_lane: 8, components: 2 },
            ]
        );
        // The claim spans FOUR lanes and the run at lane 4 is two wide, so `forwarding_contradicts`
        // - which requires a whole-run match - says nothing about it. That is why the resolver,
        // not the contradiction report, is what admits a re-layout.
        assert!(forwarding_contradicts(&vout, &[(VaryingUsage::Color0, vec![4, 5, 6, 7])]).is_none());
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
    fn a_packed_byte_read_is_one_register_not_four() {
        // `CopyFx8 o[0] <- pa[0]` reads FOUR CHANNELS out of ONE register - they are its four
        // bytes. Reading it at F32 marks `pa[0..4)` as inputs, and a fragment that declares one
        // primary register is then refused for reading `pa[1]`, which is exactly how a title's
        // two-instruction passthrough (`Nop`, then this) failed to link.
        let mut i = instr(
            Op::CopyFx8,
            Some(Operand::plain(Bank::Output, 0, 1)),
            vec![Operand::plain(Bank::PrimaryAttr, 0, 2)],
            [true; 4],
        );
        i.half_precision = false;
        let (inputs, _) = pa_read_before_write(&shader(ProgramKind::Fragment, vec![i]));
        let regs: Vec<usize> = (0..BANK_REGS).filter(|&r| inputs[r]).collect();
        assert_eq!(regs, vec![0], "four bytes of pa[0], not pa[0..4)");
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
            sa_base_from_container: true,
            containers: Vec::new(),
            uniform_buffer_bindings: Vec::new(),
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
            prefetch: Some(SamplePrefetch { unit, source_texcoord: source, last: true, lookup: crate::container::PrefetchLookup::Plain }),
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
        let iface = plan_interface(&vprog, &fprog, &fragment_reading(&[0]), false).unwrap();
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
        let plan = plan_interface(&vprog, &fprog, &fragment_reading(&[0]), false).unwrap().components;
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
        let plan = plan_interface(&vprog, &fprog, &fragment_reading(&[0]), false).unwrap().components;
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
        let plan = plan_interface(&vprog, &fprog, &fragment_reading(&[0]), false).unwrap().components;
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
        let plan = plan_interface(&vprog, &fprog, &fragment_reading(&[0]), false).unwrap().components;
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
        let plan = plan_interface(&vprog, &fprog, &fragment_reading(&[0]), false).unwrap().components;
        assert_eq!(destinations(&plan), vec![(0, Some(0)), (0, Some(1)), (1, Some(0)), (1, Some(1))]);
    }

    /// >>> A PROGRAM THAT DECLARES NO INTERPOLANTS IS NOT A PASSTHROUGH - the football
    /// >>> title's `frag_90c054e0`, whose draws were being DROPPED.
    ///
    /// The passthrough treatment exists for a program whose colour IS its primary-attribute
    /// file, loaded there by the PDS. What the PDS loads is the program's DECLARED
    /// interpolants, so a program that declares none has nothing loaded and those registers
    /// are undefined on the hardware too. Demanding they be fed asks for a value that does
    /// not exist, and refusing the pair drops the draw.
    #[test]
    fn a_program_declaring_no_interpolants_is_not_treated_as_a_passthrough() {
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(1, 6, 4)];
        // No interpolants and NO decode error - the count of 0 is true.
        let fprog = fragment_program(Vec::new(), 4);
        assert_eq!(fprog.varyings_error, None, "the fixture must be an empty DECLARATION, not a failed decode");
        let iface = plan_interface(&vprog, &fprog, &fragment_reading(&[]), false)
            .expect("a program that interpolates nothing has nothing that needs feeding");
        assert!(iface.components.is_empty(), "nothing declared, so nothing routed: {:?}", iface.components);
    }

    /// >>> ...BUT AN EMPTY LIST THAT IS AN EMPTY LIST BECAUSE THE BLOCK DID NOT PARSE KEEPS
    /// >>> THE OLD, CONSERVATIVE TREATMENT.
    ///
    /// This is the one that makes the rule above safe. Without it a varyings-block decode gap
    /// would quietly start emitting zero-initialised colour - a silently wrong picture with
    /// nothing in the log - which is the exact failure the whole refusal path exists to avoid.
    #[test]
    fn an_interpolant_list_empty_because_the_block_did_not_decode_is_still_refused() {
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(1, 6, 4)];
        let mut fprog = fragment_program(Vec::new(), 4);
        fprog.varyings_error = Some("fixture: the block did not decode");
        assert_eq!(
            plan_interface(&vprog, &fprog, &fragment_reading(&[]), false).unwrap_err(),
            LinkError::PaReadUnfed {
                register: 0,
                varyings_error: Some("fixture: the block did not decode"),
            },
            "an empty list from a FAILED decode must not be read as 'declares nothing'"
        );
    }

    /// >>> A REAL PASSTHROUGH STILL DEMANDS ITS ROUTING, AND A MASKED-OFF DRAW STILL LIFTS IT.
    ///
    /// The program here DOES declare an interpolant, so the passthrough reading applies and
    /// every register of the primary allocation must be fed. The declared TexCoord covers
    /// PA0..1 and the allocation is 4, so PA2 is unfed and the pair is refused - which is the
    /// behaviour the titles that depend on the passthrough treatment rely on.
    ///
    /// The second arm is [`LinkOptions::colour_output_masked_off`]: a colour that cannot reach
    /// a pixel is exact whatever it holds, so the same program links.
    #[test]
    fn a_passthrough_that_declares_interpolants_is_refused_unless_the_colour_is_masked_off() {
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(1, 6, 4)];
        let fprog = fragment_program(vec![texcoord_in(1, 0, 2, true)], 4);
        let body = fragment_reading(&[]);

        assert_eq!(
            plan_interface(&vprog, &fprog, &body, false).unwrap_err(),
            LinkError::PaReadUnfed { register: 2, varyings_error: None },
            "a declared interpolant means the register file IS the colour, so it must be fed"
        );

        plan_interface(&vprog, &fprog, &body, true)
            .expect("a colour that cannot reach a pixel is exact whatever it holds");
    }

    #[test]
    fn a_read_interpolant_the_vertex_does_not_produce_is_a_hard_failure() {
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(1, 6, 4)];
        let fprog = fragment_program(vec![texcoord_in(4, 0, 2, true)], 4);
        assert_eq!(
            plan_interface(&vprog, &fprog, &fragment_reading(&[0]), false).unwrap_err(),
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
            plan_interface(&vprog, &fprog, &fragment_reading(&[0, 4]), false).unwrap_err(),
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
        let plan = plan_interface(&vprog, &fprog, &fshader, false).unwrap().components;
        assert_eq!(destinations(&plan), vec![(0, Some(0)), (0, Some(1)), (1, Some(0)), (1, Some(1))]);
    }

    #[test]
    fn a_pa_read_beyond_the_container_allocation_is_a_hard_failure() {
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(1, 6, 4)];
        let fprog = fragment_program(vec![texcoord_in(1, 0, 2, true)], 4);
        assert_eq!(
            plan_interface(&vprog, &fprog, &fragment_reading(&[0, 9]), false).unwrap_err(),
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

        let iface = plan_interface(&vprog, &fprog, &fragment_reading(&[0, 2]), false).unwrap();
        assert_eq!(
            iface.prefetches,
            vec![PlannedPrefetch { unit: 13, pa_base: 2, regs: 2, coords: vec![4, 5], cube: false, projective: false, coords_unfed: false }]
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
        let iface = plan_interface(&vprog, &fprog, &fragment_reading(&[0]), false).unwrap();
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
            plan_interface(&vprog, &fprog, &fragment_reading(&[0, 2]), false).unwrap_err(),
            LinkError::PrefetchUnitNotDeclared { unit: 13 }
        );
    }

    #[test]
    fn a_cube_prefetch_takes_three_coordinates() {
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(0, 6, 4), texcoord_out(1, 10, 4)];
        let mut fprog = fragment_program(vec![texcoord_in_prefetched(1, 0, 2, 15, 0)], 4);
        fprog.parameters = vec![sampler(15, true)];
        let iface = plan_interface(&vprog, &fprog, &fragment_reading(&[0, 2]), false).unwrap();
        assert_eq!(iface.prefetches[0].coords.len(), 3);
        assert!(iface.prefetches[0].cube);
        assert_eq!(iface.prefetches[0].binding().wgsl_type(), "texture_cube<f32>");
    }

    fn texcoord_in_projective(index: u8, pa_base: u8, register_count: u8, unit: u8, source: u8) -> Interpolant {
        let mut it = texcoord_in_prefetched(index, pa_base, register_count, unit, source);
        if let Some(pf) = it.prefetch.as_mut() {
            pf.lookup = crate::container::PrefetchLookup::Projective;
        }
        it
    }

    /// The fragment of `a_projective_prefetch_*`: TEXCOORD1's data in PA[0..2] and unit 4's
    /// sample in PA[2..4], coordinated by TEXCOORD0 - mlb's grass sampling `PlayerShadowsTarget`.
    fn projective_pair(texcoord0_width: u32) -> (Program, Program) {
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(0, 6, texcoord0_width), texcoord_out(1, 10, 4)];
        let mut fprog = fragment_program(vec![texcoord_in_projective(1, 0, 2, 4, 0)], 4);
        fprog.parameters = vec![sampler(4, false)];
        (vprog, fprog)
    }

    #[test]
    fn a_projective_prefetch_routes_the_texcoords_w_and_stays_a_2d_binding() {
        // TEXCOORD0 is the homogeneous screen position (x', y', z, w) at lanes 6..10, so the
        // coordinate is lanes 6 and 7 divided by lane 9 - and NOT lane 8, the depth.
        let (vprog, fprog) = projective_pair(4);
        let iface = plan_interface(&vprog, &fprog, &fragment_reading(&[0, 2]), false).unwrap();
        let pf = &iface.prefetches[0];
        assert!(pf.projective);
        let lanes: Vec<u32> = pf.coords.iter().map(|&i| iface.components[i].vertex_lane).collect();
        assert_eq!(lanes, vec![6, 7, 9]);
        assert_eq!(iface.components[pf.coords[2]].dest, ComponentDest::SampleCoord { prefetch: 0, coord: 3 });
        // The TEXTURE is flat: projection is a property of the coordinate.
        assert_eq!(pf.binding().wgsl_type(), "texture_2d<f32>");
        assert_eq!(pf.coord_arity(), 2);
    }

    #[test]
    fn a_projective_prefetch_off_a_three_component_texcoord_divides_by_the_iterators_one() {
        // The iterator fills a missing fourth component with the texcoord default w = 1, so a
        // vertex that writes three lanes gets the plain lookup - and no lane past its varying
        // is routed (lane 9 would be the NEXT varying's first component).
        let (vprog, fprog) = projective_pair(3);
        let iface = plan_interface(&vprog, &fprog, &fragment_reading(&[0, 2]), false).unwrap();
        let pf = &iface.prefetches[0];
        assert!(!pf.projective);
        let lanes: Vec<u32> = pf.coords.iter().map(|&i| iface.components[i].vertex_lane).collect();
        assert_eq!(lanes, vec![6, 7]);
    }

    #[test]
    fn a_projective_prefetch_samples_at_xy_over_w() {
        // The emitted sample, against the plain reading as its negative control: the same pair
        // with the descriptor's lookup field at 1 samples the raw (x', y') - which is what wrapped
        // `PlayerShadowsTarget` across mlb's whole infield as a grid of dark dashes.
        let emit = |projective: bool| {
            let (vprog, mut fprog) = projective_pair(4);
            if !projective {
                fprog.interpolants[0].prefetch.as_mut().unwrap().lookup = crate::container::PrefetchLookup::Plain;
            }
            let fsh = fragment_reading(&[0, 2]);
            let vsh = shader(
                ProgramKind::Vertex,
                vec![instr(Op::Mov, Some(Operand::plain(Bank::Output, 6, 2)), vec![Operand::plain(Bank::PrimaryAttr, 0, 1)], [true; 4])],
            );
            let iface = plan_interface(&vprog, &fprog, &fsh, false).unwrap();
            let fplan = plan_bindings(&fsh, 0, |_| false);
            let mut fplan = fplan;
            for pf in &iface.prefetches {
                fplan.samplers.push(pf.binding());
            }
            let vplan = plan_vertex_bindings(&vprog, &vsh);
            let locations = (iface.components.len() as u32).div_ceil(4);
            build_linked_module(
                &emit_body(&vsh).unwrap(), &vplan, &vprog, &[], &emit_body(&fsh).unwrap(), &fplan, &fprog, &[],
                &iface, locations, false,
            )
        };
        let proj = emit(true);
        assert!(
            proj.contains("textureSample(t4, s4, vec2<f32>(vec2<f32>(in.v1.x, in.v1.y) / in.v1.z));"),
            "{proj}"
        );
        let plain = emit(false);
        assert!(plain.contains("textureSample(t4, s4, vec2<f32>(in.v1.x, in.v1.y));"), "{plain}");
        assert!(!plain.contains(") / in.v"), "{plain}");
    }

    #[test]
    fn a_prefetch_whose_texcoord_is_too_narrow_is_a_hard_failure() {
        // A cube sample needs a three-component direction; this texcoord carries two.
        let mut vprog = vertex_program(0, Vec::new(), 0);
        vprog.output_varyings = vec![texcoord_out(0, 6, 2), texcoord_out(1, 10, 4)];
        let mut fprog = fragment_program(vec![texcoord_in_prefetched(1, 0, 2, 15, 0)], 4);
        fprog.parameters = vec![sampler(15, true)];
        assert_eq!(
            plan_interface(&vprog, &fprog, &fragment_reading(&[0, 2]), false).unwrap_err(),
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
        let iface = plan_interface(&uprog, &fprog, &fsh, false).unwrap();
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


