//! Clean-room GXP (`SceGxmProgram`) -> WGSL shader recompiler.
//!
//! A Vita fragment/vertex shader is a `SceGxmProgram` container wrapping PowerVR SGX543
//! "USSE" bytecode. To render a title faithfully the guest's own shading must run, not a
//! fixed-function approximation of it. This crate:
//!
//! 1. [`container`] - parses the container (header, parameter table, USSE code location).
//! 2. [`usse`] - decodes the 64-bit USSE instructions into the [`ir`].
//! 3. [`wgsl`] - emits WGSL for shaders it can translate faithfully.
//!
//! Integrity contract: the recompiler emits WGSL only for shaders composed entirely of
//! operations whose semantics are *established clean-room facts*. It never guesses and
//! never emits an approximation. Anything it cannot translate is a HARD FAILURE - a loud
//! [`RecompileError`] that names the exact instruction and opcode to implement next, the
//! same way the NID dispatcher hard-fails on an unimplemented NID. This is an opcode
//! grind, not a silent fallback: you implement the named opcode and re-run. Every fact used
//! here - the SGX543 USSE instruction encoding, the container layout, the parameter table -
//! comes from the public hardware instruction-set encoding and vitasdk / psdevwiki
//! definitions: permissive, fact-only sources, with no copyleft or proprietary code read,
//! linked, or derived from.

pub mod container;
pub mod interp;
pub mod ir;
pub mod link;
pub mod module;
pub mod usse;
pub mod wgsl;

pub use container::{Parameter, ParamCategory, ParamType, Program, ProgramKind};
pub use ir::{Instr, Op, Shader};
pub use link::{link_programs, LinkError, LinkedProgram, MAX_VARYINGS};
pub use module::{
    BindingPlan, ColorOutput, ColorPrecision, FragmentModule, VertexAttribute, VertexBindingPlan, VertexModule,
};

/// The maximum number of `@location` varying outputs a recompiled vertex module may declare.
/// WebGPU guarantees at least 16 inter-stage variables; a vertex program that needs more than
/// this is rejected (hard-fail) rather than emitting a module that would fail pipeline creation.
pub const MAX_VERTEX_VARYINGS: u32 = 15;

/// Why a shader could not be recompiled to WGSL. Every variant is a hard failure that
/// pinpoints what to fix or implement next - never a signal to silently draw something
/// else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecompileError {
    /// The container failed to parse.
    Parse(container::ParseError),
    /// The program was a vertex shader when a fragment shader was requested (or vice
    /// versa).
    WrongKind,
    /// WGSL emission hard-failed (unsupported op named, empty, or unmapped operand).
    Emit(wgsl::EmitError),
    /// A vertex program declared more `@location` varying outputs than the pipeline supports.
    /// Names the count and the limit so the caller knows the shader exceeded the interface,
    /// not that translation failed - it falls back to fixed-function rather than mis-bind.
    TooManyVaryings { needed: u32, limit: u32 },
}

impl core::fmt::Display for RecompileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RecompileError::Parse(e) => write!(f, "GXP container parse error: {e:?}"),
            RecompileError::WrongKind => write!(f, "program kind mismatch (expected a fragment shader)"),
            RecompileError::Emit(e) => write!(f, "{e}"),
            RecompileError::TooManyVaryings { needed, limit } => write!(
                f,
                "vertex program needs {needed} varying @location outputs but the pipeline supports only {limit}",
            ),
        }
    }
}

impl std::error::Error for RecompileError {}

impl From<container::ParseError> for RecompileError {
    fn from(e: container::ParseError) -> Self {
        RecompileError::Parse(e)
    }
}
impl From<wgsl::EmitError> for RecompileError {
    fn from(e: wgsl::EmitError) -> Self {
        RecompileError::Emit(e)
    }
}

/// A successfully recompiled fragment shader plus what the pipeline builder needs to
/// bind it.
#[derive(Debug, Clone)]
pub struct RecompiledFragment {
    /// The parsed program (parameter table = the resource binding plan).
    pub program: Program,
    /// The decoded IR (for diagnostics / caching).
    pub shader: Shader,
    /// The emitted WGSL function body of `usse_main`.
    pub wgsl_body: String,
    /// Stable content hash of the source blob, for pipeline caching.
    pub hash: u64,
}

/// Decode + coverage of a program without requiring full translation. Useful for the
/// renderer to log how close a shader is to translatable and for the oracle harness.
#[derive(Debug, Clone)]
pub struct Coverage {
    pub kind: ProgramKind,
    pub total: usize,
    /// Instructions the WGSL emitter can translate today (operation wired + not blocked).
    pub supported: usize,
    /// Instructions whose operation is known from the ISA (classified), whether or not
    /// emit is wired yet. Always >= `supported`.
    pub classified: usize,
    /// Per-group instruction counts (`opcode1` -> count), for prioritising RE work.
    pub group_counts: [u32; 32],
}

impl Coverage {
    /// Fraction of instructions the emitter can translate, in `[0, 1]`.
    pub fn fraction(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.supported as f32 / self.total as f32 }
    }

    /// Fraction of instructions whose operation is known from the ISA, in `[0, 1]`.
    pub fn classified_fraction(&self) -> f32 {
        if self.total == 0 { 0.0 } else { self.classified as f32 / self.total as f32 }
    }
}

/// Parse + decode a blob and report coverage, without emitting WGSL. Never fails on an
/// unsupported opcode - that is the point (it measures them).
pub fn analyze(bytes: &[u8]) -> Result<Coverage, container::ParseError> {
    let program = Program::parse(bytes)?;
    let shader = usse::decode_shader(&program);
    let mut group_counts = [0u32; 32];
    for i in &shader.instrs {
        group_counts[(i.group & 0x1f) as usize] += 1;
    }
    Ok(Coverage {
        kind: program.kind,
        total: shader.instrs.len(),
        supported: shader.supported_count(),
        classified: shader.classified_count(),
        group_counts,
    })
}

/// One control-flow branch in a decoded program: its instruction index, its signed
/// instruction-word delta, and the target index that delta resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchSite {
    pub index: usize,
    pub rel: i32,
    pub target: i64,
}

/// Every [`Op::Branch`] in a blob, with the program's instruction count.
///
/// This exists to make the ONE unsettled fact about USSE control flow measurable on real
/// shaders rather than argued from the spec: whether a branch target is `index + rel` (what
/// [`usse::decode_shader`] produces, per the distilled ISA reference's "relative to the branch
/// instruction's own program offset") or `index + 1 + rel`. The two differ by exactly one
/// instruction, so the wrong one leaves the last instruction of every conditional block
/// running unconditionally - a wrong picture with no error, which is the failure mode this
/// recompiler exists to avoid. A branch whose target lands exactly on `total` (one past the
/// end) settles it: under the other reading that same branch would run off the end.
///
/// The captured corpus contains no branches at all, so the only source of this evidence is a
/// live title - hence a reporting hook rather than a test fixture.
pub fn branch_sites(bytes: &[u8]) -> Result<(usize, Vec<BranchSite>), container::ParseError> {
    let program = Program::parse(bytes)?;
    let shader = usse::decode_shader(&program);
    let sites = shader
        .instrs
        .iter()
        .enumerate()
        .filter_map(|(index, i)| match i.op {
            Op::Branch { rel } => Some(BranchSite { index, rel, target: index as i64 + rel as i64 }),
            _ => None,
        })
        .collect();
    Ok((shader.instrs.len(), sites))
}

/// Recompile a fragment shader blob to WGSL, or return why it could not be (which sends
/// the caller to its fixed-function fallback).
pub fn recompile_fragment(bytes: &[u8]) -> Result<RecompiledFragment, RecompileError> {
    let program = Program::parse(bytes)?;
    if program.kind != ProgramKind::Fragment {
        return Err(RecompileError::WrongKind);
    }
    let shader = usse::decode_shader(&program);
    let wgsl_body = wgsl::emit_fragment(&shader)?;
    let hash = program.hash;
    Ok(RecompiledFragment { program, shader, wgsl_body, hash })
}

/// Recompile a fragment shader blob all the way to a complete, bindable [`FragmentModule`]
/// (WGSL module source + [`BindingPlan`]) - the artifact the renderer's pipeline builder
/// consumes. Hard-fails identically to [`recompile_fragment`] on any unsupported opcode, so
/// the caller falls back to its fixed-function path only on a real translation gap, never a
/// silent degrade.
pub fn recompile_fragment_module(bytes: &[u8]) -> Result<(RecompiledFragment, FragmentModule), RecompileError> {
    let rc = recompile_fragment(bytes)?;
    let plan = module::plan_bindings(&rc.shader, rc.program.default_uniform_regs, |o| {
        rc.program.sampler_is_cube(o as u32)
    });
    let module = module::build_module(&rc.wgsl_body, &plan);
    Ok((rc, module))
}

/// A successfully recompiled vertex shader plus what the pipeline builder needs to bind it.
#[derive(Debug, Clone)]
pub struct RecompiledVertex {
    /// The parsed program (parameter table = the attribute/uniform binding plan).
    pub program: Program,
    /// The decoded IR (for diagnostics / caching).
    pub shader: Shader,
    /// The emitted WGSL function body of the vertex program.
    pub wgsl_body: String,
    /// Stable content hash of the source blob, for pipeline caching.
    pub hash: u64,
}

/// Recompile a vertex shader blob to a WGSL body, or return why it could not be (which sends
/// the caller to its fixed-function fallback). Hard-fails identically to
/// [`recompile_fragment`] on any unsupported opcode, naming exactly what to implement next.
pub fn recompile_vertex(bytes: &[u8]) -> Result<RecompiledVertex, RecompileError> {
    let program = Program::parse(bytes)?;
    if program.kind != ProgramKind::Vertex {
        return Err(RecompileError::WrongKind);
    }
    let shader = usse::decode_shader(&program);
    let wgsl_body = wgsl::emit_body(&shader)?;
    let hash = program.hash;
    Ok(RecompiledVertex { program, shader, wgsl_body, hash })
}

/// Recompile a vertex shader blob all the way to a complete, bindable [`VertexModule`] (WGSL
/// module source + [`VertexBindingPlan`]). Hard-fails identically to [`recompile_vertex`] on
/// any unsupported opcode, and rejects a program whose varying-output count exceeds
/// [`MAX_VERTEX_VARYINGS`], so the caller falls back to fixed-function only on a real gap,
/// never a silent degrade.
pub fn recompile_vertex_module(bytes: &[u8]) -> Result<(RecompiledVertex, VertexModule), RecompileError> {
    let rc = recompile_vertex(bytes)?;
    let plan = module::plan_vertex_bindings(&rc.program, &rc.shader);
    if plan.varying_vec4s > MAX_VERTEX_VARYINGS {
        return Err(RecompileError::TooManyVaryings { needed: plan.varying_vec4s, limit: MAX_VERTEX_VARYINGS });
    }
    let module = module::build_vertex_module(&rc.wgsl_body, &plan);
    Ok((rc, module))
}
