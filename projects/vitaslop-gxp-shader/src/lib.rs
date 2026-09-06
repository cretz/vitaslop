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
    BindingPlan, ColorOutput, ColorPrecision, FragmentModule, MemWindow, VertexAttribute,
    VertexBindingPlan, VertexModule,
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
    /// The fragment writes NEITHER colour register, so which one carries its result is not
    /// established for this program - see [`module::writes_no_color_register`]. Emitting it
    /// would paint a zero-initialised register file; the caller falls back instead.
    ColorRegisterNeverWritten,
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
            RecompileError::ColorRegisterNeverWritten => write!(
                f,
                "fragment writes neither OUTPUT nor PRIMATTR register 0, so the register carrying                  its colour is not established (a pass-through of an interpolated varying)"
            ),
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
    if module::writes_no_color_register(&rc.shader) {
        return Err(RecompileError::ColorRegisterNeverWritten);
    }
    let plan = module::plan_bindings(&rc.shader, rc.program.sa_carried_extent(), |o| {
        rc.program.sampler_is_cube(o as u32)
    });
    let writes_depth = rc.shader.instrs.iter().any(|i| i.op == ir::Op::DepthF);
    let module = module::build_module(&rc.wgsl_body, &plan, writes_depth);
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

/// The guest-memory windows a VERTEX blob's 0xE8 memory loads need at DRAW time, if any: for
/// each, which uniform buffer index to read the bound address for and how many bytes to
/// snapshot. EMPTY for a program with no memory loads AND for one whose windows cannot be
/// resolved - the latter refuses to link ([`LinkError::MemWindowUnresolved`] names why), so
/// a capture that snapshots nothing for it changes nothing.
///
/// The opcode scan short-circuits before parsing operands: memory loads are a handful of
/// programs in the whole captured corpus, and this runs once per registered program.
pub fn mem_windows_for_vertex_blob(bytes: &[u8]) -> Vec<module::MemWindow> {
    let Ok(program) = Program::parse(bytes) else { return Vec::new() };
    if program.kind != ProgramKind::Vertex {
        return Vec::new();
    }
    if !program.code.iter().any(|&w| usse::opcode1(w) == 0x1d) {
        return Vec::new();
    }
    let shader = usse::decode_shader(&program);
    module::resolve_mem_windows(&program, &shader).unwrap_or_default()
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

/// The destination coefficient of a fragment program's own ROP blend - see [`rop_blend`].
/// The SOURCE coefficient is `SRC_ALPHA` in every encoding this corpus establishes and is
/// refused otherwise, so it is not a field here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopDstFactor {
    /// `1` - the additive form (`sel2 = ZERO` under the complement).
    One,
    /// `1 - src.a` - straight source-over.
    OneMinusSrcAlpha,
}

/// A blend equation compiled INTO a fragment program, rather than patched in by GXM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopBlend {
    pub dst: RopDstFactor,
    /// The alpha op field (42:41) differs between titles; `true` is the second observed value.
    /// Carried so a caller can refuse rather than silently treat the two alike.
    pub alpha_op_differs: bool,
}

/// The blend equation a FRAGMENT program performs itself, in its epilogue, or `None`.
///
/// # Why this exists
/// GXM has no runtime blend state and normally bakes the equation into a fragment program from
/// the `SceGxmBlendInfo*` passed at `sceGxmShaderPatcherCreateFragmentProgram`
/// ([[vitaslop-blend-is-baked-in]]). A program created with a **NULL** `blendInfo` was therefore
/// read as "never blends" - and for a program whose epilogue is a plain colour write that is
/// right. It is NOT right for a program that ends in a group-0x80 SOP2 reading the OUTPUT
/// register as one of its two sources: the output register at the end of a fragment program is
/// the DESTINATION colour the ROP feeds back, so such a word IS the blend, compiled in by the
/// offline shader compiler instead of patched in by the driver.
///
/// [`usse::decode`]'s `decode_grp_sop2` reads these words as a COPY of the source term, on the
/// argument that `o[0]` is "a register the program never writes" and so its coefficient must be
/// zero. That argument is wrong in exactly this way, and it is why one title's entire UI - every
/// glyph, every alpha-cut sprite, the box around its logo - drew as an opaque rectangle: the
/// source term alone is the premultiplied colour with the destination thrown away. The copy is
/// still what the WGSL emits, because a fragment shader cannot read the framebuffer; what is
/// recovered here is the PIPELINE state that the second term describes.
///
/// # What the corpus establishes
/// Every group-0x80 word in every captured corpus that writes `o[0]` decodes, through the field
/// table of its fully-read sibling SOP2M, to one of two equations:
///
/// | word | `sel1`/`mod1` | `sel2`/`mod2` | equation |
/// |---|---|---|---|
/// | `809080c190000000` | `SRC1_ALPHA` / - | `ZERO` / complement | `a*src + 1*dst` |
/// | `809080d990000000` | `SRC1_ALPHA` / - | `SRC1_ALPHA` / complement | `a*src + (1-a)*dst` |
///
/// Those are the two commonest blend equations there are, and they are the SAME two the same
/// title passes explicitly as a `SceGxmBlendInfo` for its other shaders (measured:
/// `colorSrc=4 colorDst=5` and `colorSrc=4 colorDst=1`). A field table that turns unrelated bits
/// into exactly the pair of equations the title already asks for by another route is not a
/// coincidence.
///
/// **The operator is ADD, and the frame is what says so.** Bits 53:52, which SOP2M spends on its
/// colour op, are 1 in EVERY observed group-0x80 word - including both equations above and a
/// second title's - so they cannot be this group's colour op, which would have to differ. Read
/// as ADD, `809080d990000000` is source-over, and source-over is measurably the right answer:
/// forcing it made a title's dialogue text, which had been three solid white bars, render as
/// readable glyphs. Read as SUB nothing composites at all.
///
/// # What is refused, and why that matters more than what is accepted
/// Anything else returns `None` and the caller keeps GXM's own answer. In particular a program
/// created WITH a `blendInfo` never reaches here, and the corpus says it never needs to: no
/// fragment blob carries both a real `blendInfo` and an epilogue SOP2. The two are alternatives
/// - the driver patches the blend in, or the compiler compiled it in - and a program that
/// somehow had both would be double-blended, so the caller must consult this only when GXM
/// supplied nothing.
///
/// The named risk: if bits 53:52 do encode an operator that is not ADD for some word not in any
/// corpus here, that word takes this path and blends the wrong way round. It would show as a
/// surface that is too bright where it should be dark. The guard below pins every field that
/// this evidence does not read, so such a word returns `None` instead.
pub fn rop_blend(bytes: &[u8]) -> Option<RopBlend> {

    let program = Program::parse(bytes).ok()?;
    if program.kind != ProgramKind::Fragment {
        return None;
    }
    let shader = usse::decode_shader(&program);
    let mut found = None;
    for instr in &shader.instrs {
        if instr.group != 0x80 {
            continue;
        }
        let w = instr.raw;
        let bit = |hi, lo| usse::decode::bits(w, hi, lo);
        // The destination and the fed-back source must both be the OUTPUT bank: that pairing is
        // what makes the word a ROP blend rather than an ordinary combine.
        let dest_is_output = bit(33, 32) == 1 && bit(27, 21) == 0;
        let src2_is_output = bit(29, 28) == 1 && bit(6, 0) == 0;
        // >>> THE SWAPPED SHAPE IS A BLEND TOO - `dst + a*src`, THE ADDITIVE FORM.
        //
        // `0x8190002160040000` names `o[0]` in SRC1 and `pa[0]` in SRC2 (`decode_grp_sop2`'s
        // "swapped" form). It was read as a copy on the strength of a sibling pair: one title
        // registers one flat-colour shader twice, `frag_81a7f590` ending in `809080d990000000`
        // and `frag_81a7f798` ending in this word, and at the time BOTH were read as copies.
        // The first is now established above as source-over, so the pair is the same shader
        // with two BLEND equations - and its fields say which: `sel1 = ZERO` under `mod1`
        // (complement) is a destination coefficient of 1, exactly as `809080c190000000`'s
        // complemented ZERO is, with the operands the other way round.
        //
        // MEASURED, and it is the whole of one title's Velvet Room: the display pass draws 56
        // flat-colour quads through `frag_81a7f798` right after the world composite. As a copy
        // they are opaque black rectangles over the room; the room's own draws replay to
        // velvet blue in capsules and show on screen the moment those quads are cut out of
        // the pass (`VITASLOP_DRAW_RANGE=0-1`). Additive black adds nothing, and the magenta
        // ones are the glow the room is known for.
        //
        // Bits 20:14 are 16 in this word and 20 in a lit, fogged world material's epilogue
        // (`0x8190002160050000`); that material draws opaque scenery whose overlaps do not
        // brighten, so 20 stays the copy `decode_grp_sop2` emits and only 16 is a blend.
        // What bit 16 selects is not established - the split is the two observed words.
        // The source coefficient (`sel2 = 4`, a selector no reference names) is TAKEN as
        // SRC_ALPHA, the only source coefficient any compiled-in blend here has; if it is ONE,
        // a translucent additive quad reads a little too bright.
        let swapped = bit(31, 30) == 1
            && bit(13, 7) == 0
            && bit(29, 28) == 2
            && bit(58, 57) == 0
            && bit(56, 56) == 1
            && bit(53, 52) == 1
            && bit(51, 51) == 0
            && bit(49, 49) == 0
            && bit(48, 48) == 0
            && bit(47, 47) == 0
            && bit(46, 43) == 0
            && bit(42, 41) == 0
            && bit(40, 38) == 0
            && bit(37, 35) == 4
            && bit(20, 14) == 16;
        if dest_is_output && swapped {
            if found.is_some() {
                return None;
            }
            found = Some(RopBlend { dst: RopDstFactor::One, alpha_op_differs: false });
            continue;
        }
        if !dest_is_output || !src2_is_output {
            continue;
        }
        // Every field this reading does not establish, pinned to its one observed value.
        let pinned = bit(58, 57) == 0        // unpredicated
            && bit(56, 56) == 0              // mod1 - the source term is not complemented
            && bit(53, 52) == 1              // constant across every observed word (see above)
            && bit(51, 51) == 0              // no destination bank extension
            && bit(49, 49) == 0              // no src1 bank extension
            && bit(48, 48) == 0              // no src2 bank extension
            && bit(47, 47) == 1              // mod2 - the destination term IS complemented
            && bit(46, 43) == 0              // the bits SOP2M spends on its write mask
            && bit(40, 38) == 3; // sel1 = SRC1_ALPHA, the source coefficient
        if !pinned {
            return None;
        }
        let dst = match bit(37, 35) {
            0 => RopDstFactor::One,
            3 => RopDstFactor::OneMinusSrcAlpha,
            _ => return None,
        };
        let alpha_op_differs = match bit(42, 41) {
            0 => false,
            1 => true,
            _ => return None,
        };
        // Two ROP words in one program is a shape this reading says nothing about.
        if found.is_some() {
            return None;
        }
        found = Some(RopBlend { dst, alpha_op_differs });
    }
    found
}
