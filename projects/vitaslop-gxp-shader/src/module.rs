//! Assemble a recompiled fragment [`Shader`] into a COMPLETE, bindable WGSL fragment
//! module - the artifact the renderer's pipeline builder consumes - together with an
//! explicit [`BindingPlan`] describing exactly what the renderer must bind.
//!
//! [`crate::wgsl::emit_fragment`] produces the function *body* (the scalarised USSE
//! register-file statements). [`crate::wgsl::wrap_module`] wraps that body into a
//! STANDALONE module with zeroed private banks, used only to prove the body is valid WGSL
//! in isolation. This module goes the rest of the way: it declares the real resource
//! bindings the guest shader needs - the default uniform buffer (SA bank), the sampled
//! textures, and the interpolated varyings (PA bank) - and returns the binding plan so the
//! integration layer can wire the draw's captured inputs to them.
//!
//! ## The register-file / binding contract (what the renderer must satisfy)
//!
//! The USSE register file is modelled as scalar-`f32` local arrays (`r`/`o`/`i`/`pa`), the
//! same model the emitter targets. The externally-bound banks map as:
//!
//! * **SA (secondary attributes = the default uniform buffer)** -> a uniform buffer at
//!   `@group(0) @binding(0)`, laid out as `array<vec4<f32>, N>` so `sa[k]` reads 4-byte
//!   register `k` (`data[k/4][k%4]`). The renderer uploads the captured
//!   `bound_fragment_uniform_buf` bytes verbatim - the packing already matches (a uniform's
//!   `resource_index` is its 4-byte-register offset, exactly this indexing).
//! * **Sampled textures (group 0xE0 SMP)** -> a `texture_2d`/`texture_3d` + `sampler` pair
//!   per referenced unit at `@group(1)`, bindings `2*i` / `2*i+1`. The renderer binds the
//!   draw's texture for that sampler unit (cross-checked against the reflected sampler
//!   parameter table).
//! * **PA (primary attributes = interpolated varyings)** -> `@location(i) vec4<f32>`
//!   fragment inputs `v0..`, one vec4 per four PA lanes the shader reads. The renderer's
//!   vertex stage MUST output these varyings so `pa[lane]` receives the interpolated value.
//!   This is the cross-stage linkage: the fragment module declares the varyings it needs;
//!   feeding them faithfully requires the matching vertex program's output layout.
//! * **Output** -> `@location(0)`. Native-colour shaders leave RGBA in OUTPUT reg 0 (`o0`);
//!   non-native-colour shaders leave it in PRIMATTR reg 0 (`pa0`). Which one applies is
//!   determined here from the shader's actual writes (a shader that writes the OUTPUT bank
//!   is native), matching the SGX "the value left in o0/pa0 at program end is the colour"
//!   rule without needing to guess a header flag.

use core::fmt::Write as _;

use crate::container::{ParamCategory, Program};
use crate::ir::{Bank, Op, Shader};
use crate::wgsl::{tex_units, TexBinding, BANK_REGS};

/// Where the fragment shader's final RGBA lives at program end (SGX has no explicit colour
/// emit; the value left in a fixed register is the output - see the texflow spec F8.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorOutput {
    /// Native colour: RGBA in OUTPUT register 0 (`o0`).
    NativeO0,
    /// Non-native colour: RGBA in PRIMATTR register 0 (`pa0`).
    NonNativePa0,
}

/// The precision the final colour registers hold, which decides how many registers the four
/// components occupy and how to read them back.
///
/// The USSE register file is untyped 32-bit storage: an F32 operation leaves one component per
/// register, while an F16 one packs two per register (channel `c` = half `c & 1` of register
/// `index + (c >> 1)`), exactly as everywhere else in this translator. So the colour's layout is
/// not a property of the render target - it is a property of the instruction that produced it,
/// and reading four consecutive registers as F32 when the shader wrote F16 pairs yields
/// denormal garbage (a black frame), not an approximation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorPrecision {
    /// One component per register: `x..w` = registers 0..3, each an F32 bit pattern.
    F32,
    /// Two components per register: `x,y` = the halves of register 0, `z,w` of register 1.
    F16,
}

/// The concrete resources a [`FragmentModule`] expects the renderer to bind. Every count is
/// derived from the decoded shader + the parameter table, so the renderer can build matching
/// bind-group layouts and know exactly what to upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingPlan {
    /// Number of 4-byte SA registers the default-uniform-buffer binding must supply
    /// (`sa[0..sa_lane_count]`), taken from the container's `default_uniform_regs`. The buffer
    /// is bound as `array<vec4<u32>, ceil(n/4)>` - raw registers, since a register may hold two
    /// packed F16 halves. Zero means the shader reads no uniforms (no SA binding).
    pub sa_lane_count: u32,
    /// Number of PA registers the shader reads as varyings (`pa[0..pa_lane_count]`). How many
    /// interpolated scalar components that costs depends on each register's access width, so
    /// the linker computes the `@location` layout rather than assuming four lanes per vec4.
    pub pa_lane_count: u32,
    /// The sampler units the shader references, ascending by unit. Bound at `@group(1)`,
    /// `t{unit}` = binding `2*i`, `s{unit}` = binding `2*i+1` (matching declaration order).
    pub samplers: Vec<TexBinding>,
    /// Which register holds the final colour at `@location(0)`.
    pub color: ColorOutput,
    /// How the four colour components are laid out across those registers.
    pub color_precision: ColorPrecision,
}

impl BindingPlan {
    /// Number of `@location` vec4 varying inputs the vertex stage must provide.
    pub fn varying_count(&self) -> u32 {
        self.pa_lane_count.div_ceil(4)
    }

    /// Number of `vec4<f32>` elements in the SA uniform buffer (`0` when no SA binding).
    pub fn sa_vec4_count(&self) -> u32 {
        self.sa_lane_count.div_ceil(4)
    }
}

/// A recompiled fragment shader assembled into a complete, bindable WGSL module.
#[derive(Debug, Clone)]
pub struct FragmentModule {
    /// The full WGSL module source: `fn fs_main(...) -> @location(0) vec4<f32>`.
    pub wgsl: String,
    /// What the renderer must bind to run it.
    pub bindings: BindingPlan,
}

/// The highest 32-bit REGISTER index a bank is *read* at, plus one (0 if never read). Scans
/// every source operand: an F32 channel reads register `index + selector`, while an F16
/// channel reads a half of register `index + (selector >> 1)` (the four F16 channels share a
/// register pair). Swizzle constants (selector >= 4) read no register. Texture-coordinate
/// operands (SMP `srcs[0]`) count only their coordinate components, matching the emitter.
fn bank_read_extent(shader: &Shader, bank: Bank) -> u32 {
    let mut extent = 0u32;
    for instr in &shader.instrs {
        let read_lanes = read_lane_mask(instr);
        for src in &instr.srcs {
            if src.bank != bank {
                continue;
            }
            for c in 0..4 {
                if !read_lanes[c] {
                    continue;
                }
                let sel = src.swizzle[c];
                if sel <= 3 {
                    let step =
                        if instr.source_half_precision() { (sel >> 1) as u32 } else { sel as u32 };
                    extent = extent.max(src.index as u32 + step + 1);
                }
            }
        }
    }
    extent
}

/// The channels an instruction actually reads from its sources (mirrors
/// `crate::wgsl`'s read model: a dot/tex reads a fixed prefix, everything else reads where it
/// writes). Kept local so the module extent scan matches emitted reads exactly.
fn read_lane_mask(instr: &crate::ir::Instr) -> [bool; 4] {
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

/// True when the fragment writes NEITHER colour register, so nothing in the stream says what it
/// emits and [`color_output`]'s inference has no evidence to work from.
///
/// Such a program is not broken - it is a PASS-THROUGH, and on hardware the register its header
/// names is already full: the rasteriser wrote the interpolated varying there before the first
/// instruction ran, and a shader with nothing to do just lets it stand. A racing title's race
/// frame contains one (a single PHAS word, a declared `Color0` interpolant, and no other
/// instruction) - a flat vertex-coloured polygon.
///
/// Emitting it anyway would return a zero-initialised register file, i.e. paint transparent
/// black over whatever it covers, with no error anywhere - exactly the silent-approximation
/// failure this recompiler exists to avoid. Reproducing it instead needs the header's
/// `is_native_color` bit, which is not established, so the honest answer is to report the pair
/// and let the renderer draw its fixed-function approximation.
pub fn writes_no_color_register(shader: &Shader) -> bool {
    !writes_bank(shader, Bank::Output) && !writes_bank(shader, Bank::PrimaryAttr)
}

/// Whether the shader writes any lane of a bank (used to decide native vs non-native colour).
fn writes_bank(shader: &Shader, bank: Bank) -> bool {
    shader.instrs.iter().any(|i| {
        i.dest
            .as_ref()
            .is_some_and(|d| d.bank == bank && i.write_mask.iter().any(|&m| m))
    })
}

/// Decide where the fragment colour ends up: a shader that writes the OUTPUT bank is native
/// (`o0`); one that never writes OUTPUT but writes PRIMATTR reg 0 is non-native (`pa0`).
///
/// Which of the two applies is a header fact (spec F8.9 - `is_native_color`), and this infers it
/// from what the stream WRITES instead, which is exact for every shader that writes its colour
/// at all. A shader that writes NEITHER is the one case the inference cannot answer: see
/// [`writes_no_color_register`], which makes the caller fall back rather than let this default
/// pick a register the program never filled.
fn color_output(shader: &Shader) -> ColorOutput {
    if writes_bank(shader, Bank::Output) {
        return ColorOutput::NativeO0;
    }
    let writes_pa0 = shader.instrs.iter().any(|i| {
        i.dest.as_ref().is_some_and(|d| {
            d.bank == Bank::PrimaryAttr && d.index == 0 && i.write_mask.iter().take(4).any(|&m| m)
        })
    });
    if writes_pa0 {
        ColorOutput::NonNativePa0
    } else {
        ColorOutput::NativeO0
    }
}

/// The precision of the value left in the colour registers: that of the LAST instruction to
/// write register 0 of the colour bank, since that instruction is what produced the value the
/// hardware emits. A shader that never writes it (so the module returns the register file's
/// initial state) is reported as [`ColorPrecision::F32`], the raw-bit-pattern reading, which is
/// what the zero-initialised registers mean either way.
fn color_precision(shader: &Shader, color: ColorOutput) -> ColorPrecision {
    let bank = match color {
        ColorOutput::NativeO0 => Bank::Output,
        ColorOutput::NonNativePa0 => Bank::PrimaryAttr,
    };
    let last = shader.instrs.iter().rev().find(|i| {
        i.dest
            .as_ref()
            .is_some_and(|d| d.bank == bank && d.index == 0 && i.write_mask.iter().any(|&m| m))
    });
    match last {
        Some(i) if i.half_precision => ColorPrecision::F16,
        _ => ColorPrecision::F32,
    }
}

/// The `vec4<f32>` expression that reads the final colour out of register-file array `bank`,
/// honouring how the shader packed it. Shared by the standalone fragment wrapper and the
/// linked module so both read the colour identically.
///
/// `varyings` is how many `v<n>` locations this fragment stage actually declares, so the
/// varying probe can refuse to name one that does not exist. It used to emit `in.v<n>`
/// unconditionally, which is a WGSL parse error on any pair with fewer varyings - and since
/// every pair is compiled, one unlucky pair took the whole run down with it. A diagnostic that
/// cannot be aimed at one shader has to degrade on the others, not abort.
pub(crate) fn color_return_expr(bank: &str, precision: ColorPrecision, varyings: u32) -> String {
    // Diagnostic (`VITASLOP_GXP_PROBE=<bank><index>`, e.g. `r6` or `pa0`): return that register
    // pair AS the colour instead of the shader's own result. A recompiled shader that paints a
    // wrong colour is otherwise a black box - this bisects it by making any intermediate
    // visible, which is how a "the whole surface is black" bug is traced to the one term that
    // is zero. Read here so both the standalone and linked module paths honour it.
    // Diagnostic (`VITASLOP_GXP_VPROBE=<n>`): return interpolated varying `v<n>` AS the colour.
    // The register probe below can only see values the fragment stores into a register, and a
    // TEXTURE COORDINATE is usually not one of them - it is consumed straight out of the varying
    // by a `textureSample`. That leaves the most common "why is this sampling the wrong place"
    // question with no instrument at all, which is how a composite's UV offset stayed a matter
    // of argument for a whole session. `<n>.xy` shows as red/green, so an on-screen ramp from
    // black to yellow is UV 0..1 and anything flat is a coordinate that does not vary.
    if let Ok(n) = std::env::var("VITASLOP_GXP_VPROBE").map(|s| s.trim().to_string()) {
        if let Ok(i) = n.parse::<u32>() {
            if i < varyings {
                return format!("vec4<f32>(in.v{i}.x, in.v{i}.y, in.v{i}.z, 1.0)");
            }
            // This pair has no such varying. Return a flat MAGENTA rather than emitting a
            // field access that does not compile: the probe is asking a question this shader
            // cannot answer, and "not applicable" has to be visibly different from "zero".
            return "vec4<f32>(1.0, 0.0, 1.0, 1.0)".to_string();
        }
    }
    if let Ok(spec) = std::env::var("VITASLOP_GXP_PROBE") {
        let split = spec.trim().find(|c: char| c.is_ascii_digit()).unwrap_or(spec.len());
        let (probe_bank, idx) = spec.trim().split_at(split);
        if let Ok(i) = idx.parse::<u32>() {
            return format!(
                "vec4<f32>(unpack2x16float({probe_bank}[{i}]), unpack2x16float({probe_bank}[{}]))",
                i + 1
            );
        }
    }
    match precision {
        ColorPrecision::F32 => format!(
            "vec4<f32>(bitcast<f32>({bank}[0]), bitcast<f32>({bank}[1]), \
             bitcast<f32>({bank}[2]), bitcast<f32>({bank}[3]))"
        ),
        ColorPrecision::F16 => {
            format!("vec4<f32>(unpack2x16float({bank}[0]), unpack2x16float({bank}[1]))")
        }
    }
}

/// The vector component letter for lane `c` (0..3 -> x/y/z/w).
fn comp(c: u32) -> char {
    ['x', 'y', 'z', 'w'][(c & 3) as usize]
}

/// Build the [`BindingPlan`] for a decoded fragment shader from its operands + the program's
/// default-uniform-buffer size. `uniform_regs` is the container's `default_uniform_regs`
/// (header +0x64) - the authoritative size of the buffer that is loaded at SA register 0. It
/// is NOT the total SA register count: the SA registers above the uniform buffer hold texture
/// control words and compile-time literals, which are not part of this binding.
pub fn plan_bindings(shader: &Shader, uniform_regs: u32, is_cube: impl Fn(u8) -> bool) -> BindingPlan {
    let sa_lane_count = uniform_regs;
    let pa_lane_count = bank_read_extent(shader, Bank::PrimaryAttr);
    let color = color_output(shader);
    BindingPlan {
        sa_lane_count,
        pa_lane_count,
        samplers: tex_units(shader, is_cube),
        color,
        color_precision: color_precision(shader, color),
    }
}

/// Assemble a complete, bindable WGSL fragment module from an emitted body + its binding
/// plan. The body is the verbatim output of [`crate::wgsl::emit_fragment`]; this wraps it
/// with the real resource bindings and the register-file locals, initialising `pa` from the
/// varying inputs and `sa` from the uniform buffer before the body runs.
pub fn build_module(body: &str, plan: &BindingPlan, writes_depth: bool) -> FragmentModule {
    let mut m = String::new();

    // A depth-writing program (0xF8 DEPTHF) reads the pipeline's depth state through the same
    // group-3 block the linked module declares, so this standalone wrapper has to declare it
    // too or the module does not compile at all.
    if writes_depth {
        m.push_str(crate::link::GXP_DEPTH_DECL);
    }

    // Sampled textures + samplers at group 1 (t{unit} = binding 2*i, s{unit} = 2*i+1).
    for (i, b) in plan.samplers.iter().enumerate() {
        let (tb, sb) = (i as u32 * 2, i as u32 * 2 + 1);
        let ty = b.wgsl_type();
        let _ = writeln!(m, "@group(1) @binding({tb}) var t{}: {ty};", b.unit);
        let _ = writeln!(m, "@group(1) @binding({sb}) var s{}: sampler;", b.unit);
    }

    // Default uniform buffer (SA bank) at group 0 binding 0, as raw 32-bit registers - a
    // register may hold an F32 or two packed F16 halves, so it is never bound as floats.
    let sa_vec4 = plan.sa_vec4_count();
    if sa_vec4 > 0 {
        let _ = writeln!(m, "struct SaBuf {{ data: array<vec4<u32>, {sa_vec4}> }};");
        let _ = writeln!(m, "@group(0) @binding(0) var<uniform> sa_buf: SaBuf;");
    }

    // Interpolated varyings (PA bank) as @location vec4 inputs.
    // `front_facing` is declared unconditionally - see the note in `link::build_linked_module`.
    let varyings = plan.varying_count();
    let _ = writeln!(m, "struct FsIn {{");
    for i in 0..varyings {
        let _ = writeln!(m, "  @location({i}) v{i}: vec4<f32>,");
    }
    let _ = writeln!(m, "  @builtin(front_facing) front_facing: bool,");
    if writes_depth {
        let _ = writeln!(m, "  @builtin(position) frag_coord: vec4<f32>,");
    }
    let _ = writeln!(m, "}};");
    if writes_depth {
        let _ = writeln!(
            m,
            "\nstruct FsOut {{\n  @location(0) color: vec4<f32>,\n  @builtin(frag_depth) depth: f32,\n}};"
        );
    }
    let ret_ty = if writes_depth { "FsOut" } else { "@location(0) vec4<f32>" };
    let _ = writeln!(m, "\n@fragment\nfn fs_main(in: FsIn) -> {ret_ty} {{");
    m.push_str(crate::wgsl::FRONT_FACING_DECL);
    if writes_depth {
        let _ = writeln!(m, "  let gxp_interp_depth = in.frag_coord.z;");
        let _ = writeln!(m, "  var gxp_frag_depth: f32 = gxp_interp_depth;");
    }

    // The USSE register-file locals: raw 32-bit registers, matching the emitter.
    for bank in ["r", "o", "i", "pa", "sa"] {
        let _ = writeln!(m, "  var {bank}: array<u32, {BANK_REGS}>;");
    }
    // Predicate registers p0..p3 (written by test ops, read by predicated instructions).
    let _ = writeln!(m, "  var p: array<bool, 4>;");
    // The INDEX register file, for register-INDIRECT operands. Two registers, because the
    // extension row names exactly two indexed banks (INDEXED1 -> i0, INDEXED2 -> i1).
    let _ = writeln!(m, "  var idx: array<i32, 2>;");

    // Feed the PA registers from the varyings. This standalone wrapper carries one register
    // per interpolated component; the real linked module ([`crate::link`]) instead derives each
    // varying's F16/F32 width from both stages and interpolates F16 halves separately.
    for j in 0..plan.pa_lane_count {
        let _ = writeln!(m, "  pa[{j}] = bitcast<u32>(in.v{}.{});", j / 4, comp(j % 4));
    }
    if sa_vec4 > 0 {
        let _ = writeln!(
            m,
            "  for (var k: u32 = 0u; k < {}u; k = k + 1u) {{ sa[k] = sa_buf.data[k / 4u][k % 4u]; }}",
            plan.sa_lane_count
        );
    }

    m.push_str(body);

    let ret = match plan.color {
        ColorOutput::NativeO0 => "o",
        ColorOutput::NonNativePa0 => "pa",
    };
    // The standalone fragment wrapper has no inter-stage varyings at all (it is compiled
    // without a vertex partner), so the varying probe can never apply here.
    let color = color_return_expr(ret, plan.color_precision, 0);
    if writes_depth {
        let _ = writeln!(m, "  return FsOut({color}, gxp_frag_depth);\n}}");
    } else {
        let _ = writeln!(m, "  return {color};\n}}");
    }

    FragmentModule { wgsl: m, bindings: plan.clone() }
}

// ===================================================================================
// Vertex programs
// ===================================================================================
//
// A vertex program runs the same USSE arithmetic core (so the emitted BODY is identical to
// a fragment's - see `crate::wgsl::emit_body`); only the module I/O differs:
//
// * **Inputs** are vertex ATTRIBUTES, not interpolated varyings. Each ATTRIBUTE-category
//   parameter's `resource_index` is the base scalar lane in the PA bank the input is fetched
//   into, and `component_count` its lane span (validated against the captured vertex blobs:
//   e.g. `normal@resource_index=4,comp=4` is read by the shader at `pa[4]`). They bind as
//   `@location(i)` vertex inputs; the module loads `pa[base..base+components]` from each.
// * **Outputs** live in the O bank. The clip-space POSITION is always `o0..o3` (4 lanes -
//   confirmed: every captured vertex program writes o0,o1,o2,o3), surfaced as
//   `@builtin(position)`. Every further written output lane (`o[4..]`) is an interpolant,
//   surfaced as `@location` vec4 varyings (four lanes each) for the fragment stage to consume.
// * **Uniforms** are the SA default uniform buffer, exactly as on the fragment side.
//
// The mapping of each varying `@location` to a fragment PA input by USAGE (position/colour/
// texcoord linkage) is a separate cross-stage step; here the vertex module faithfully exposes
// every output register at a deterministic location without interpreting its usage.

/// A vertex input attribute the recompiled vertex module consumes: bound at
/// `@location(location)` and loaded into `pa[base_lane .. base_lane + components]` (the
/// primary-attribute bank the USSE code reads its inputs from). `base_lane`/`components` come
/// straight from the program's ATTRIBUTE parameter (`resource_index` = base scalar lane,
/// `component_count` = lane span).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexAttribute {
    /// The attribute's declared name (diagnostics + renderer cross-check against the vertex
    /// buffer layout).
    pub name: String,
    /// The `@location` the module binds this attribute at (assigned in ascending base-lane
    /// order; the renderer feeds the matching vertex-buffer attribute here).
    pub location: u32,
    /// Base scalar lane in the PA bank the attribute's first component loads into.
    pub base_lane: u32,
    /// Number of scalar lanes (components) the attribute spans (1..4).
    pub components: u32,
}

/// The concrete resources a [`VertexModule`] expects the renderer to bind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexBindingPlan {
    /// Vertex input attributes, ascending by base lane. Each binds at its `location`.
    pub attributes: Vec<VertexAttribute>,
    /// Number of 4-byte SA registers the default-uniform-buffer binding must supply
    /// (`sa[0..sa_lane_count]`), as `array<vec4<f32>, ceil(n/4)>` at `@group(0) @binding(0)`.
    /// Zero means the shader reads no uniforms.
    pub sa_lane_count: u32,
    /// Number of `@location` vec4 varying OUTPUTS beyond clip position (grouping `o[4..]` four
    /// lanes per location). Zero means the vertex program outputs only position.
    pub varying_vec4s: u32,
    /// Textures the VERTEX program samples (vertex texture fetch), ascending by unit. A vertex
    /// program that samples is building its GEOMETRY from the texture, so an unbound one is a
    /// missing mesh rather than an untextured surface. Empty for the usual vertex program.
    pub samplers: Vec<crate::wgsl::TexBinding>,
}

impl VertexBindingPlan {
    /// Number of `vec4<f32>` elements in the SA uniform buffer (`0` when no SA binding).
    pub fn sa_vec4_count(&self) -> u32 {
        self.sa_lane_count.div_ceil(4)
    }
}

/// A recompiled vertex shader assembled into a complete, bindable WGSL module.
#[derive(Debug, Clone)]
pub struct VertexModule {
    /// The full WGSL module source: `fn vs_main(in: VsIn) -> VsOut`.
    pub wgsl: String,
    /// What the renderer must bind to run it.
    pub bindings: VertexBindingPlan,
}

/// The highest OUTPUT-bank scalar lane a shader writes, plus one (0 if it writes none). Used
/// to size the vertex program's varying outputs (`o[4..extent]`).
fn output_write_extent(shader: &Shader) -> u32 {
    let mut extent = 0u32;
    for instr in &shader.instrs {
        let Some(d) = instr.dest.as_ref() else { continue };
        if d.bank != Bank::Output {
            continue;
        }
        for c in 0..4 {
            if instr.write_mask[c] {
                extent = extent.max(d.index as u32 + c as u32 + 1);
            }
        }
    }
    extent
}

/// Build the [`VertexBindingPlan`] for a decoded vertex program from its parameter table
/// (attributes) + the declared SA register count + the output write extent. `varying_vec4s`
/// packs every written output lane beyond clip position (`o[4..]`) four lanes per `@location`.
pub fn plan_vertex_bindings(program: &Program, shader: &Shader) -> VertexBindingPlan {
    let mut attributes: Vec<VertexAttribute> = program
        .parameters
        .iter()
        .filter(|p| p.category == ParamCategory::Attribute)
        .map(|p| VertexAttribute {
            name: p.name.clone(),
            location: 0, // assigned below in base-lane order
            base_lane: p.resource_index.max(0) as u32,
            components: (p.component_count as u32).clamp(1, 4),
        })
        .collect();
    attributes.sort_by_key(|a| a.base_lane);
    for (i, a) in attributes.iter_mut().enumerate() {
        a.location = i as u32;
    }

    // The SA binding carries the default uniform buffer only (loaded at SA register 0); the SA
    // registers above it hold texture control words and literals, which are not bound data.
    let sa_lane_count = program.default_uniform_regs;
    let extent = output_write_extent(shader);
    let varying_vec4s = extent.saturating_sub(4).div_ceil(4);
    let samplers = crate::wgsl::tex_units(shader, |u| program.sampler_is_cube(u as u32));

    VertexBindingPlan { attributes, sa_lane_count, varying_vec4s, samplers }
}

/// Assemble a complete, bindable WGSL vertex module from an emitted body + its binding plan.
/// The body is the verbatim output of [`crate::wgsl::emit_body`]; this wraps it with the
/// vertex attribute inputs (loaded into `pa`), the SA uniform buffer, and the position +
/// varying outputs.
pub fn build_vertex_module(body: &str, plan: &VertexBindingPlan) -> VertexModule {
    let mut m = String::new();

    // Default uniform buffer (SA bank) at group 0 binding 0, as raw 32-bit registers - a
    // register may hold an F32 or two packed F16 halves, so it is never bound as floats.
    let sa_vec4 = plan.sa_vec4_count();
    if sa_vec4 > 0 {
        let _ = writeln!(m, "struct SaBuf {{ data: array<vec4<u32>, {sa_vec4}> }};");
        let _ = writeln!(m, "@group(0) @binding(0) var<uniform> sa_buf: SaBuf;");
    }

    // Vertex inputs: one @location per attribute (typed vec4; unused lanes ignored). A vertex
    // program with no attributes takes no input parameter (an empty WGSL struct is invalid).
    let has_inputs = !plan.attributes.is_empty();
    if has_inputs {
        let _ = writeln!(m, "struct VsIn {{");
        for a in &plan.attributes {
            let _ = writeln!(m, "  @location({}) a{}: vec4<f32>,", a.location, a.location);
        }
        let _ = writeln!(m, "}};");
    }

    // Outputs: clip position builtin + one vec4 per varying location.
    let _ = writeln!(m, "struct VsOut {{");
    let _ = writeln!(m, "  @builtin(position) position: vec4<f32>,");
    for j in 0..plan.varying_vec4s {
        let _ = writeln!(m, "  @location({j}) v{j}: vec4<f32>,");
    }
    let _ = writeln!(m, "}};");

    if has_inputs {
        let _ = writeln!(m, "\n@vertex\nfn vs_main(in: VsIn) -> VsOut {{");
    } else {
        let _ = writeln!(m, "\n@vertex\nfn vs_main() -> VsOut {{");
    }

    // The USSE register-file locals: raw 32-bit registers, matching the emitter.
    for bank in ["r", "o", "i", "pa", "sa"] {
        let _ = writeln!(m, "  var {bank}: array<u32, {BANK_REGS}>;");
    }
    let _ = writeln!(m, "  var p: array<bool, 4>;");
    // The INDEX register file, for register-INDIRECT operands. Two registers, because the
    // extension row names exactly two indexed banks (INDEXED1 -> i0, INDEXED2 -> i1).
    let _ = writeln!(m, "  var idx: array<i32, 2>;");

    // Load PA registers from the vertex attributes (vertex inputs are plain f32 components).
    const COMP: [&str; 4] = ["x", "y", "z", "w"];
    for a in &plan.attributes {
        for c in 0..a.components {
            let _ = writeln!(
                m,
                "  pa[{}] = bitcast<u32>(in.a{}.{});",
                a.base_lane + c,
                a.location,
                COMP[c as usize]
            );
        }
    }
    // Load SA lanes from the uniform buffer.
    if sa_vec4 > 0 {
        let _ = writeln!(
            m,
            "  for (var k: u32 = 0u; k < {}u; k = k + 1u) {{ sa[k] = sa_buf.data[k / 4u][k % 4u]; }}",
            plan.sa_lane_count
        );
    }

    m.push_str(body);

    // Surface the outputs: clip position (o0..o3) + varyings (o[4..], four registers per
    // location). Standalone wrapper only - the linked module derives the real per-varying
    // F16/F32 transport from both stages (see [`crate::link`]).
    let f = |reg: u32| format!("bitcast<f32>(o[{reg}])");
    let _ = writeln!(m, "  var out: VsOut;");
    let _ = writeln!(m, "  out.position = vec4<f32>({}, {}, {}, {});", f(0), f(1), f(2), f(3));
    for j in 0..plan.varying_vec4s {
        let b = 4 + j * 4;
        let _ = writeln!(m, "  out.v{j} = vec4<f32>({}, {}, {}, {});", f(b), f(b + 1), f(b + 2), f(b + 3));
    }
    let _ = writeln!(m, "  return out;\n}}");

    VertexModule { wgsl: m, bindings: plan.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::ProgramKind;
    use crate::ir::{Instr, Op, Operand, Predicate};

    fn instr(op: Op, dest: Option<Operand>, srcs: Vec<Operand>, mask: [bool; 4]) -> Instr {
        Instr { op, pred: Predicate::Always, dest, write_mask: mask, srcs, half_precision: false, raw: 0, group: 0, blocked: None }
    }

    fn shader(instrs: Vec<Instr>) -> Shader {
        Shader { kind: ProgramKind::Fragment, instrs }
    }

    #[test]
    fn plan_counts_pa_sa_extents_and_samplers() {
        // o0 = pa[4..8] (varying) * sa[8..12] (uniform); then a tex sample from unit 3.
        let sh = shader(vec![
            instr(
                Op::Mul,
                Some(Operand::plain(Bank::Output, 0, 1)),
                vec![Operand::plain(Bank::PrimaryAttr, 4, 2), Operand::plain(Bank::SecondaryAttr, 8, 3)],
                [true; 4],
            ),
            instr(
                Op::Tex { unit: 3, coords: 2, coord_half: false, lod: crate::ir::TexLod::Implicit },
                Some(Operand::plain(Bank::Temp, 0, 0)),
                vec![Operand::plain(Bank::PrimaryAttr, 10, 2)],
                [true; 4],
            ),
        ]);
        let plan = plan_bindings(&sh, 12, |_| false);
        // PA read up to pa[10]+pa[11] (coords x,y of reg 10) -> 12 registers.
        assert_eq!(plan.pa_lane_count, 12);
        // The SA binding is exactly the declared default uniform buffer.
        assert_eq!(plan.sa_lane_count, 12);
        assert_eq!(plan.samplers, vec![TexBinding { unit: 3, coords: 2, cube: false }]);
        assert_eq!(plan.color, ColorOutput::NativeO0);
        assert_eq!(plan.varying_count(), 3); // ceil(12/4)
        assert_eq!(plan.sa_vec4_count(), 3);
    }

    #[test]
    fn sa_binding_is_the_declared_uniform_buffer_not_the_read_extent() {
        // The SA bank also holds texture control words and literals above the uniform buffer,
        // so the binding size comes from the container - never from how far the code reads.
        let sh = shader(vec![instr(
            Op::Mov,
            Some(Operand::plain(Bank::Output, 0, 1)),
            vec![Operand::plain(Bank::SecondaryAttr, 40, 3)],
            [true; 4],
        )]);
        assert_eq!(plan_bindings(&sh, 8, |_| false).sa_lane_count, 8);
    }

    #[test]
    fn non_native_color_detected_from_pa0_write() {
        // A shader that writes PRIMATTR reg 0 and never writes OUTPUT is non-native colour.
        let sh = shader(vec![instr(Op::Mov, Some(Operand::plain(Bank::PrimaryAttr, 0, 2)), vec![Operand::plain(Bank::Temp, 4, 0)], [true; 4])]);
        assert_eq!(plan_bindings(&sh, 0, |_| false).color, ColorOutput::NonNativePa0);
    }

    #[test]
    fn f16_colour_is_read_back_as_packed_halves_not_four_f32_registers() {
        // The overwhelming majority of this generation's fragment code is F16, and an F16
        // instruction leaves x,y in the halves of colour register 0 and z,w in register 1 -
        // NOT one component per register. Reading four consecutive registers as F32 bit
        // patterns there returns denormal garbage (a black frame), so the layout must follow
        // the precision of the instruction that produced the colour.
        let mut half = instr(
            Op::Mul,
            Some(Operand::plain(Bank::PrimaryAttr, 0, 1)),
            vec![Operand::plain(Bank::PrimaryAttr, 4, 2), Operand::plain(Bank::SecondaryAttr, 0, 3)],
            [true; 4],
        );
        half.half_precision = true;
        let plan = plan_bindings(&shader(vec![half]), 4, |_| false);
        assert_eq!(plan.color, ColorOutput::NonNativePa0);
        assert_eq!(plan.color_precision, ColorPrecision::F16);
        let wgsl = build_module("", &plan, false).wgsl;
        assert!(
            wgsl.contains("return vec4<f32>(unpack2x16float(pa[0]), unpack2x16float(pa[1]));"),
            "{wgsl}"
        );

        // An F32 shader keeps the one-component-per-register reading.
        let f32_sh = shader(vec![instr(
            Op::Mul,
            Some(Operand::plain(Bank::Output, 0, 1)),
            vec![Operand::plain(Bank::PrimaryAttr, 4, 2), Operand::plain(Bank::SecondaryAttr, 0, 3)],
            [true; 4],
        )]);
        assert_eq!(plan_bindings(&f32_sh, 4, |_| false).color_precision, ColorPrecision::F32);
    }

    #[test]
    fn colour_precision_follows_the_last_write_to_the_colour_register() {
        // A shader may use the colour register as scratch at one precision and produce the
        // final colour at another; only the last write decides the layout of what is emitted.
        let f32_scratch = instr(
            Op::Mov,
            Some(Operand::plain(Bank::PrimaryAttr, 0, 1)),
            vec![Operand::plain(Bank::Temp, 4, 2)],
            [true; 4],
        );
        let mut half_final = f32_scratch.clone();
        half_final.half_precision = true;
        let plan = plan_bindings(&shader(vec![f32_scratch.clone(), half_final.clone()]), 0, |_| false);
        assert_eq!(plan.color_precision, ColorPrecision::F16);
        let reversed = plan_bindings(&shader(vec![half_final, f32_scratch]), 0, |_| false);
        assert_eq!(reversed.color_precision, ColorPrecision::F32);
    }

    #[test]
    fn module_wires_pa_sa_and_returns_output() {
        let sh = shader(vec![instr(
            Op::Mul,
            Some(Operand::plain(Bank::Output, 0, 1)),
            vec![Operand::plain(Bank::PrimaryAttr, 0, 2), Operand::plain(Bank::SecondaryAttr, 0, 3)],
            [true; 4],
        )]);
        let plan = plan_bindings(&sh, 4, |_| false);
        let body = crate::wgsl::emit_fragment(&sh).unwrap();
        let module = build_module(&body, &plan, false);
        assert!(module.wgsl.contains("var<uniform> sa_buf: SaBuf;"), "{}", module.wgsl);
        assert!(module.wgsl.contains("@location(0) v0: vec4<f32>"), "{}", module.wgsl);
        assert!(module.wgsl.contains("pa[0] = bitcast<u32>(in.v0.x);"), "{}", module.wgsl);
        assert!(module.wgsl.contains("sa[k] = sa_buf.data[k / 4u][k % 4u];"), "{}", module.wgsl);
        assert!(
            module.wgsl.contains("return vec4<f32>(bitcast<f32>(o[0]), bitcast<f32>(o[1])"),
            "{}",
            module.wgsl
        );
    }

    use crate::container::{ParamType, Parameter, Program};

    /// A minimal `Program` carrying only the fields the vertex planner reads (parameters +
    /// register counts), for testing the vertex binding plan without a real blob.
    fn vertex_program(secondary_reg_count: u16, attrs: Vec<Parameter>) -> Program {
        Program {
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
            hash: 0,
        }
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
        }
    }

    #[test]
    fn vertex_plan_maps_attributes_and_outputs() {
        // pos@lane0, normal@lane4, uv@lane8 (declared out of order to check sorting); the shader
        // writes clip position (o0..3) and one varying group (o6..9).
        let sh = shader(vec![
            instr(Op::Mad, Some(Operand::plain(Bank::Output, 0, 1)),
                vec![Operand::plain(Bank::PrimaryAttr, 0, 2), Operand::plain(Bank::SecondaryAttr, 0, 3), Operand::plain(Bank::Constant, 2, 0)], [true; 4]),
            instr(Op::Mov, Some(Operand::plain(Bank::Output, 6, 1)),
                vec![Operand::plain(Bank::PrimaryAttr, 8, 2)], [true; 4]),
        ]);
        let mut prog = vertex_program(
            4,
            vec![attribute("uv", 8, 2), attribute("position", 0, 4), attribute("normal", 4, 4)],
        );
        prog.default_uniform_regs = 4;
        let plan = plan_vertex_bindings(&prog, &sh);
        // Attributes sorted by base lane, locations assigned in that order.
        assert_eq!(plan.attributes[0].name, "position");
        assert_eq!(plan.attributes[0].base_lane, 0);
        assert_eq!(plan.attributes[0].location, 0);
        assert_eq!(plan.attributes[2].name, "uv");
        assert_eq!(plan.attributes[2].base_lane, 8);
        assert_eq!(plan.attributes[2].components, 2);
        // SA binding = the declared uniform buffer (4); output extent 10 -> ceil((10-4)/4) = 2.
        assert_eq!(plan.sa_lane_count, 4);
        assert_eq!(plan.varying_vec4s, 2);
    }

    #[test]
    fn vertex_module_wires_inputs_position_and_varyings() {
        let sh = shader(vec![
            instr(Op::Mad, Some(Operand::plain(Bank::Output, 0, 1)),
                vec![Operand::plain(Bank::PrimaryAttr, 0, 2), Operand::plain(Bank::SecondaryAttr, 0, 3), Operand::plain(Bank::Constant, 2, 0)], [true; 4]),
            instr(Op::Mov, Some(Operand::plain(Bank::Output, 6, 1)),
                vec![Operand::plain(Bank::PrimaryAttr, 8, 2)], [true; 4]),
        ]);
        let mut prog = vertex_program(4, vec![attribute("position", 0, 4), attribute("uv", 8, 2)]);
        prog.default_uniform_regs = 4;
        let plan = plan_vertex_bindings(&prog, &sh);
        let body = crate::wgsl::emit_body(&sh).unwrap();
        let module = build_vertex_module(&body, &plan);
        let w = &module.wgsl;
        assert!(w.contains("@builtin(position) position: vec4<f32>,"), "{w}");
        assert!(w.contains("@location(0) a0: vec4<f32>,"), "{w}");
        assert!(w.contains("pa[0] = bitcast<u32>(in.a0.x);"), "{w}"); // position attribute
        assert!(w.contains("pa[8] = bitcast<u32>(in.a1.x);"), "{w}"); // uv attribute at reg 8
        assert!(!w.contains("pa[10] ="), "uv is 2-component, must not load a 3rd register:\n{w}");
        assert!(
            w.contains("out.position = vec4<f32>(bitcast<f32>(o[0]), bitcast<f32>(o[1])"),
            "{w}"
        );
        assert!(w.contains("out.v0 = vec4<f32>(bitcast<f32>(o[4]), bitcast<f32>(o[5])"), "{w}");
        assert!(w.contains("var<uniform> sa_buf: SaBuf;"), "{w}");
    }

    #[test]
    fn vertex_position_only_has_no_varyings() {
        // A vertex program that writes only clip position (o0..3) needs zero varying locations.
        let sh = shader(vec![instr(Op::Mov, Some(Operand::plain(Bank::Output, 0, 1)),
            vec![Operand::plain(Bank::PrimaryAttr, 0, 2)], [true; 4])]);
        let prog = vertex_program(0, vec![attribute("position", 0, 4)]);
        let plan = plan_vertex_bindings(&prog, &sh);
        assert_eq!(plan.varying_vec4s, 0);
        let body = crate::wgsl::emit_body(&sh).unwrap();
        let module = build_vertex_module(&body, &plan);
        assert!(!module.wgsl.contains("@location(0) v0"), "no varyings:\n{}", module.wgsl);
        assert!(!module.wgsl.contains("var<uniform>"), "no SA binding when none read:\n{}", module.wgsl);
    }
}

