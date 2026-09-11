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
//!   non-native-colour shaders leave it in the PRIMATTR register their LAST write targets,
//!   which is not always `pa0` (see [`ColorOutput::NonNativePa`]). Which one applies is
//!   determined here from the shader's actual writes (a shader that writes the OUTPUT bank
//!   is native), matching the SGX "the value left in the colour register at program end is the
//!   colour" rule without needing to guess a header flag.

use core::fmt::Write as _;

use crate::container::{ParamCategory, Program};
use crate::ir::{Bank, Instr, Op, Operand, Predicate, Shader};
use crate::wgsl::{tex_units, TexBinding, BANK_REGS};

/// Where the fragment shader's final RGBA lives at program end (SGX has no explicit colour
/// emit; the value left in a fixed register is the output - see the texflow spec F8.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorOutput {
    /// Native colour: RGBA in OUTPUT register 0 (`o0`).
    NativeO0,
    /// Non-native colour: RGBA in PRIMATTR register `base` (and `base + 1` when the colour is
    /// packed F16 pairs, or `base..base + 4` at F32).
    ///
    /// # The base is NOT always 0, and assuming it was painted a surface black
    /// A fragment's primary-attribute allocation holds its interpolants FIRST - including a
    /// PDS-prefetched sample, which occupies registers of its own - and a non-native colour
    /// goes wherever the program's own writes put it, which is above them. MEASURED on one
    /// title's bright-pass (`frag_8669f600`, pair `553fa1bb8c47dce0`):
    /// `primary_reg_count = 4`, the one descriptor is prefetch-only and takes `pa[0..2)`, and
    /// both of the program's two instructions write `pa[2]`. Reading the colour at `pa0` - or,
    /// as the old code did, falling through to the OUTPUT bank because `pa0` was never written
    /// - returns registers nothing ever filled, so that pass wrote (0,0,0,0) into the 128x128
    /// surface the glare chain blurs, and the game's whole bloom/glare composite added exactly
    /// nothing for the entire run.
    NonNativePa(u32),
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
    /// FOUR components in ONE register, each a `byte / 255` unsigned-normalised channel.
    ///
    /// This is what a fragment leaves behind when its epilogue is the 8-bit pair
    /// `pack.unorm8` + `mov.fx8` (see `usse::decode::decode_grp_sop2`): the colour is already
    /// in the surface's own 8-bit-per-channel form, in one register, not spread over two or
    /// four. Reading it as [`Self::F16`] would take four bytes for two halves and emit a
    /// denormal pair - a black frame that reports success, the same failure
    /// [[vitaslop-f16-colour-output]] records for reading an F16 colour as F32.
    Fx8,
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
    /// Whether this fragment program READS THE OUTPUT BANK - i.e. reads the DESTINATION colour
    /// the ROP feeds back, and so performs its blending itself.
    ///
    /// # Why the output register is the destination
    /// On this hardware a fragment program's output registers are the on-chip pixel data, and
    /// the driver seeds them with the framebuffer's current colour. A program is therefore free
    /// to blend in ordinary ALU: `pa0 = -o[0] + src; pa0 = pa0 * a + o[0]` is a source-over
    /// lerp written out longhand, and one retail title composites its whole frame that way -
    /// its colour-grading pass is `sa4*dst + (dot(sa8, dst) * sa2 + sa0.x)`, which no
    /// fixed-function blend can express at all.
    ///
    /// [`crate::rop_blend`] already recovers the OTHER shape this takes - an epilogue group-0x80
    /// SOP2 - as a pipeline blend state. That works because a SOP2 epilogue IS one of the two
    /// standard equations; an arbitrary ALU chain is not, so it needs the real destination
    /// colour and there is no way around reading it.
    ///
    /// A module built with this set declares a destination texture and seeds the O bank from it
    /// at entry; the renderer owes it a copy of the attachment taken immediately before the
    /// draw. WebGPU has no framebuffer fetch, so that copy is the whole mechanism.
    pub reads_dest_color: bool,
    /// Whether that read is lowered to a DUAL-SOURCE blend - see [`dual_source_eligible`].
    /// When set, the linked module declares no destination texture and the renderer owes the
    /// draw a dual-source pipeline blend instead of an attachment copy.
    pub dual_source: bool,
    /// The guest-memory windows THIS (fragment) program's 0xE8 loads read through, in the order
    /// the `gxp_fmem` binding lays them out. Empty for the overwhelming majority; a fragment
    /// that loads memory reaches its buffer through `sceGxmSetFragmentUniformBuffer`, which is
    /// a different table from the vertex stage's, so the two windows are bound separately.
    pub mem_windows: Vec<MemWindow>,
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
                    // A packed-byte source spans ONE register whatever the channel - see
                    // [`crate::ir::Instr::source_packed_bytes`].
                    let step = if instr.source_packed_bytes() {
                        0
                    } else if instr.source_half_precision() {
                        (sel >> 1) as u32
                    } else {
                        sel as u32
                    };
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
    // WHICH primary-attribute register: the base of the LAST instruction that writes the bank.
    // A fragment program's final act is to leave its colour in the register the hardware emits,
    // so the last write names it - and unlike "register 0" that is a reading of the program
    // rather than a convention. It reproduces every non-native pair of the frame this was
    // measured on (the two whose colour is at `pa0` still resolve to 0) and it is the only thing
    // that answers for the one whose colour is at `pa2` - see [`ColorOutput::NonNativePa`].
    let last_pa = shader
        .instrs
        .iter()
        .rev()
        .find(|i| {
            i.dest
                .as_ref()
                .is_some_and(|d| d.bank == Bank::PrimaryAttr && i.write_mask.iter().take(4).any(|&m| m))
        })
        .and_then(|i| i.dest.as_ref())
        .map(|d| d.index as u32);
    match last_pa {
        Some(base) => ColorOutput::NonNativePa(base),
        // Writes neither bank: the caller refuses this pair rather than let a default pick a
        // register the program never filled. See [`writes_no_color_register`].
        None => ColorOutput::NativeO0,
    }
}

/// The precision of the value left in the colour registers: that of the LAST instruction to
/// write register 0 of the colour bank, since that instruction is what produced the value the
/// hardware emits. A shader that never writes it (so the module returns the register file's
/// initial state) is reported as [`ColorPrecision::F32`], the raw-bit-pattern reading, which is
/// what the zero-initialised registers mean either way.
fn color_precision(shader: &Shader, color: ColorOutput) -> ColorPrecision {
    let (bank, base) = match color {
        ColorOutput::NativeO0 => (Bank::Output, 0),
        ColorOutput::NonNativePa(base) => (Bank::PrimaryAttr, base),
    };
    let writer_of = |bank: Bank, index: u32, before: usize| -> Option<(usize, &crate::ir::Instr)> {
        shader.instrs[..before].iter().enumerate().rev().find(|(_, i)| {
            i.dest.as_ref().is_some_and(|d| {
                d.bank == bank && d.index as u32 == index && i.write_mask.iter().any(|&m| m)
            })
        })
    };
    let mut found = writer_of(bank, base, shader.instrs.len());
    // >>> A COPY DOES NOT CHANGE THE PRECISION OF WHAT IT COPIES, and taking it at face
    // >>> value is a black frame that reports success - the same failure
    // >>> [[vitaslop-f16-colour-output]] records for the F16 reading itself.
    //
    // The precision of the colour is the precision of the arithmetic that BUILT it. A
    // fragment program often assembles its colour in a working register and then moves it
    // to the colour register in one full-width copy: that copy carries `half_precision ==
    // false` because it moves 32 bits, and reading it as the colour's precision calls a
    // register holding two packed halves an F32 component. Bitcasting a packed pair to f32
    // gives a denormal, so every channel comes out ~0 and the frame is black.
    //
    // Found on a retail title whose final gamma pass ends `o[0] = pa[0]; o[1] = pa[1];`
    // over registers built with `pack2x16float` - the whole picture was black while every
    // draw, pipeline and texture reported success.
    //
    // So a full-width move is followed back to what it copied, and that register's own
    // writer is asked instead. Bounded, because a chain of copies is still a chain and a
    // cycle must not hang the compiler.
    for _ in 0..8 {
        let Some((at, i)) = found else { break };
        if !matches!(i.op, Op::Mov) || i.half_precision {
            break;
        }
        // Only a plain register-to-register copy forwards: an immediate or a constant has
        // no earlier writer to ask, and a swizzle that reorders halves is not a copy of
        // one register's layout.
        let Some(src) = i.srcs.first().filter(|s| {
            matches!(s.bank, Bank::PrimaryAttr | Bank::Temp | Bank::Internal | Bank::Output)
        }) else {
            break;
        };
        match writer_of(src.bank, u32::from(src.index), at) {
            Some(next) => found = Some(next),
            None => break,
        }
    }
    let last = found.map(|(_, i)| i);
    match last {
        // An 8-BIT write leaves four bytes in the one register, whatever `half_precision`
        // says - that flag describes a float view and neither of these ops has one. This has
        // to be asked FIRST: both carry `half_precision == false`, so the fall-through would
        // call a packed-byte colour F32 and read four registers, three of which the program
        // never wrote.
        Some(i) if matches!(i.op, Op::CopyFx8 | Op::PackUnorm8 { to_unorm8: true, .. } | Op::Sop2 { .. }) => {
            ColorPrecision::Fx8
        }
        Some(i) if i.half_precision => ColorPrecision::F16,
        _ => ColorPrecision::F32,
    }
}

/// Diagnostic (`VITASLOP_GXP_PROBE=<bank><idx>[@<instr>][:f32|:bits=<hex>]`, e.g. `pa2@54`):
/// return that register AS the colour instead of the shader's own result.
///
/// The plain form (`pa20`) reads a register at the END of the shader, and that is a trap worth
/// naming: for any register the program writes more than once - `pa0`, `pa2`, `pa4`, `pa6` and
/// `r0` in a typical lit material all are - the end value is NOT the one the arithmetic in the
/// middle used, so "this term is zero" read off a plain probe can be an artefact of a later
/// overwrite. The `@<instr>` form (`pa2@54`) snapshots the register the moment instruction 54
/// has run, which is what makes a bisection down a colour chain possible at all.
///
/// `at` is an index into the DECODED instruction list (what `print_one_blob` prints), not a
/// byte offset and not the compact disassembly's numbering, which elides nothing but does
/// renumber the 32-bit bitwise expansions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProbeSpec {
    /// Register-file array name as it appears in the emitted WGSL: `pa`, `r`, `sa`, `i`, `o`.
    pub bank: String,
    /// Index of the first register of the pair.
    pub index: u32,
    /// Snapshot after this decoded instruction index; `None` reads at the end of the shader.
    pub at: Option<usize>,
    /// Read the four lanes as two F16 pairs (the default) or as four full F32 registers.
    pub f32_lanes: bool,
    /// `:bits=<hex>` - paint a lane 1.0 when that register's RAW BITS equal this word, 0.0
    /// otherwise. The read modes above cannot answer "is this exactly the poison word": a
    /// quiet NaN reads back through `bitcast` as something a colour attachment clamps to
    /// black, which is indistinguishable from the guest having written a real zero - the very
    /// distinction `VITASLOP_GXM_UNIFORM_POISON` exists to make. A bit compare is exact.
    pub bits: Option<u32>,
}

/// Parse `VITASLOP_GXP_PROBE=<bank><index>[@<instr>][:f32]`, e.g. `pa20`, `pa2@54`, `pa8@37:f32`.
///
/// Returns `None` when the variable is unset or does not parse, so a malformed probe leaves the
/// shader alone rather than silently painting a wrong picture.
pub(crate) fn probe_spec() -> Option<ProbeSpec> {
    parse_probe_spec(&std::env::var("VITASLOP_GXP_PROBE").ok()?)
}

/// The parse itself, separated from the environment read so the tests can drive it with a
/// plain argument. They used to drive it THROUGH the environment - set the process-global
/// variable, call [`probe_spec`], remove it - under a mutex that serialised the parse tests
/// against each other. What the mutex could not cover is every OTHER test in the crate:
/// `build_module` reads the same variable, so a colour-emission test running in parallel with
/// a parse test intermittently saw a probe active and emitted probe WGSL - a two-in-three
/// flake on the whole suite that looked exactly like the emitter being nondeterministic.
/// The environment layer this no longer exercises is one `std::env::var` line.
pub(crate) fn parse_probe_spec(raw: &str) -> Option<ProbeSpec> {
    let raw = raw.trim();
    let (head, bits) = match raw.split_once(":bits=") {
        Some((h, w)) => (h, Some(u32::from_str_radix(w.trim().trim_start_matches("0x"), 16).ok()?)),
        None => (raw, None),
    };
    let (head, f32_lanes) = match head.strip_suffix(":f32") {
        Some(h) => (h, true),
        None => (head, false),
    };
    let (head, at) = match head.split_once('@') {
        Some((h, n)) => (h, Some(n.trim().parse::<usize>().ok()?)),
        None => (head, None),
    };
    let split = head.find(|c: char| c.is_ascii_digit())?;
    let (bank, idx) = head.split_at(split);
    if bank.is_empty() {
        return None;
    }
    Some(ProbeSpec { bank: bank.to_string(), index: idx.parse().ok()?, at, f32_lanes, bits })
}

/// Module-scope declarations for an `@<instr>` probe's snapshot registers.
///
/// These live at module scope rather than in the statement body because ONE WGSL function can
/// carry more than one emitted body - a fragment stage runs its SECONDARY program and then its
/// primary in the same `fs_main` - and a per-body declaration is a redefinition that takes the
/// whole pipeline down. A private var is per-invocation, so the two stages of a linked module
/// do not share one.
pub(crate) fn probe_globals() -> String {
    match probe_spec() {
        Some(spec) if spec.at.is_some() => concat!(
            "var<private> _probe0: u32 = 0x00003c00u;\n",
            "var<private> _probe1: u32 = 0x3c000000u;\n",
            "var<private> _probe2: u32 = 0x00003c00u;\n",
            "var<private> _probe3: u32 = 0x00003c00u;\n",
        )
        .to_string(),
        _ => String::new(),
    }
}

/// The WGSL that reads one probe's four lanes out of `regs` - either the two-register array
/// slice the probe names, or the snapshot locals when it is an `@<instr>` probe.
pub(crate) fn probe_read_expr(spec: &ProbeSpec, from_snapshot: bool) -> String {
    let (a, b, c, d) = if from_snapshot {
        ("_probe0".into(), "_probe1".into(), "_probe2".into(), "_probe3".into())
    } else {
        let i = spec.index;
        let bk = spec.bank.as_str();
        (
            format!("{bk}[{i}]"),
            format!("{bk}[{}]", i + 1),
            format!("{bk}[{}]", i + 2),
            format!("{bk}[{}]", i + 3),
        )
    };
    if let Some(w) = spec.bits {
        return format!(
            "vec4<f32>(select(0.0, 1.0, {a} == {w}u), select(0.0, 1.0, {b} == {w}u),              select(0.0, 1.0, {c} == {w}u), 1.0)"
        );
    }
    if spec.f32_lanes {
        format!(
            "vec4<f32>(bitcast<f32>({a}), bitcast<f32>({b}), bitcast<f32>({c}), bitcast<f32>({d}))"
        )
    } else {
        format!("vec4<f32>(unpack2x16float({a}), unpack2x16float({b}))")
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
pub(crate) fn color_return_expr(
    bank: &str,
    base: u32,
    precision: ColorPrecision,
    varyings: u32,
) -> String {
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
    if let Ok(n) = std::env::var("VITASLOP_GXP_VPROBE").map(|s| s.trim().to_string())
        && let Ok(i) = n.parse::<u32>() {
            if i < varyings {
                return format!("vec4<f32>(in.v{i}.x, in.v{i}.y, in.v{i}.z, 1.0)");
            }
            // This pair has no such varying. Return a flat MAGENTA rather than emitting a
            // field access that does not compile: the probe is asking a question this shader
            // cannot answer, and "not applicable" has to be visibly different from "zero".
            return "vec4<f32>(1.0, 0.0, 1.0, 1.0)".to_string();
        }
    if let Some(spec) = probe_spec() {
        // An `@<instr>` probe reads the SNAPSHOT locals the emitter wrote at that instruction,
        // not the register's end value - see [`ProbeSpec`] for why those differ.
        let e = probe_read_expr(&spec, spec.at.is_some());
        // >>> `VITASLOP_GXP_PROBE_SCALE=<f>`: DIVIDE the probed value before it is written.
        //
        // A colour attachment CLAMPS to [0,1], so every probe of an HDR term reads back as a
        // flat 255 and a bisection down a lit material stops at the first light multiply -
        // which on a title whose sun colour is (12.2, 11.7, 4.8) and whose ambient is
        // (97.2, 23.7, 10.7) is the very first instruction that matters. Saturation and
        // "exactly 1.0" are then indistinguishable, and so are 4x over and 40x over.
        //
        // Dividing by a known constant moves the range of interest back under the clamp and
        // costs one multiply; the reader multiplies back. It is a DIAGNOSTIC scale and it is
        // reported, because a frame whose colours were divided is not the frame the guest asked
        // for and must never be mistaken for one.
        if let Ok(v) = std::env::var("VITASLOP_GXP_PROBE_SCALE")
            && let Ok(f) = v.trim().parse::<f32>()
            && f > 0.0
        {
            return format!("(({e}) / {f:?})");
        }
        return e;
    }
    match precision {
        ColorPrecision::F32 => format!(
            "vec4<f32>(bitcast<f32>({bank}[{base}]), bitcast<f32>({bank}[{}]), \
             bitcast<f32>({bank}[{}]), bitcast<f32>({bank}[{}]))",
            base + 1,
            base + 2,
            base + 3
        ),
        ColorPrecision::F16 => format!(
            "vec4<f32>(unpack2x16float({bank}[{base}]), unpack2x16float({bank}[{}]))",
            base + 1
        ),
        // One register, four `byte/255` channels - the inverse of the store `Prec::Fx8` uses,
        // so a colour that went through the 8-bit epilogue comes back the way it went in.
        ColorPrecision::Fx8 => format!("unpack4x8unorm({bank}[{base}])"),
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
        reads_dest_color: declares_dest_color(shader),
        // Never on its own: the LINK asks for it, per draw - see `link::LinkOptions`.
        dual_source: false,
        // A plan built from the SHADER alone cannot resolve a window - that needs the
        // program's containers and parameter table - so it carries none, and
        // `link_programs` fills them in. Same shape as `VertexAttribute::surplus_fill`.
        mem_windows: Vec::new(),
    }
}

/// A blend equation a fragment program performed ITSELF in ALU over the destination colour,
/// recovered as PIPELINE state so the draw does not need the framebuffer read at all.
///
/// The operation is always ADD; only the two coefficients vary. See [`lower_dest_blend`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DestBlend {
    pub color: BlendTerm,
    pub alpha: BlendTerm,
}

/// One channel group's `src * src_factor + dst * dst_factor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlendTerm {
    pub src: BlendFactor,
    pub dst: BlendFactor,
}

/// The coefficients [`lower_dest_blend`] can produce. Every one exists in WebGPU under the same
/// name, so the renderer maps them one for one and nothing is approximated in the mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    /// The SOURCE colour, per channel - WebGPU's `Src`. What a `dst * K` modulate needs, with
    /// the shader emitting `K`.
    Src,
    SrcAlpha,
    OneMinusSrcAlpha,
}

/// Whether [`lower_dest_blend`] is allowed to run. `VITASLOP_GXP_DEST_BLEND=0` turns it off and
/// every program keeps its ALU blend, which then needs the destination texture and a render-pass
/// split - the A/B arm that proves a lowering equivalent by comparing the two frames.
///
/// A static rather than an environment read because this crate is compiled for the browser,
/// which has no environment; the renderer sets it once from its own knob table.
/// A MASK over the shapes, not a boolean, so a wrong picture can be bisected to one shape
/// without a rebuild: bit 0 lerp, bit 1 modulate, bit 2 additive.
static DEST_BLEND_LOWERING: core::sync::atomic::AtomicU32 =
    core::sync::atomic::AtomicU32::new(FORM_DEFAULT);

/// The shape bits of [`DEST_BLEND_LOWERING`].
pub const FORM_LERP: u32 = 1;
pub const FORM_MODULATE: u32 = 2;
pub const FORM_ADDITIVE: u32 = 4;
/// The LERP variant whose SOURCE TERM is not the register the colour epilogue copies, so the
/// epilogue has to be redirected at it - see form C in [`match_dest_blend`]. Its own bit
/// because it is newer than the other three and a whole-run pixel A/B has to be able to hold
/// everything else fixed while turning it off.
pub const FORM_LERP_SRC: u32 = 8;

/// The shapes that are ON by default, because each has been PROVED equivalent to the ALU form
/// it replaces:
///
/// * LERP - a capsule of a shipped HUD pair replayed both ways differs by a MAXIMUM CHANNEL
///   DELTA OF 1 over 8,043 covered pixels, which is the ROP's 8-bit rounding and nothing else.
/// * MODULATE - a 3,400-frame headless run of the same title with the shape on and off is
///   equal to a mean absolute error of 0.01 per channel on every shot, with no pixel anywhere
///   off by more than 36 and none at all off by more than 32 as a share of the frame.
///
/// * ADDITIVE - three capsules of a shipped particle pair, replayed both ways, differ by a
///   MAXIMUM CHANNEL DELTA OF 1 over every covered pixel (278, 226 and 172 pixels differing of
///   235,520, all by one). The rewrite is also equal term for term by inspection: the ALU form
///   computes `src + o[]` and copies `o[].w` into the alpha lane, and the lowered form computes
///   `src` under `One/One` colour and `Zero/One` alpha, which is `dst + src` and `dst.a`. The
///   delta of 1 is the ROP doing that sum in the attachment's 8 bits where the shader did it in
///   f32 - the same rounding LERP was proved to.
///
/// >>> ADDITIVE WAS OFF UNTIL IT WAS MEASURED, and turning it on is worth 9 of this title's 15
/// >>> remaining destination-reading fragment programs. Each one is a RENDER-PASS SPLIT per
/// draw - the pass ends, the whole attachment is copied, and a new pass begins with
/// `LoadOp::Load`, which on a tiling GPU is a full store and reload of every tile. A phone
/// measured 54 splits and 108 MB of copies in ONE frame of this title's menu; the corpus says
/// this shape alone takes its dest-reading programs from 15 to 6. No other title in any corpus
/// has a single destination-reading fragment program, so nothing else can be moved by it.
pub const FORM_DEFAULT: u32 = FORM_LERP | FORM_MODULATE | FORM_ADDITIVE | FORM_LERP_SRC;

/// Every shape, including the unproved one.
pub const FORM_ALL: u32 = FORM_LERP | FORM_MODULATE | FORM_ADDITIVE | FORM_LERP_SRC;

/// How many REGISTERS one count of an index register spans - see [`crate::wgsl`]'s
/// `emit_load_index` for the frame that settled it. `VITASLOP_GXP_IDX_SCALE=1` restores the
/// old single-register reading as the A/B arm.
pub fn index_register_scale() -> i32 {
    static CELL: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_GXP_IDX_SCALE")
            .ok()
            .and_then(|s| s.trim().parse::<i32>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(2)
    })
}

/// Set by the renderer from `VITASLOP_GXP_DEST_BLEND`. See [`DEST_BLEND_LOWERING`].
pub fn set_dest_blend_lowering(forms: u32) {
    DEST_BLEND_LOWERING.store(forms, core::sync::atomic::Ordering::Relaxed);
}

fn form_enabled(bit: u32) -> bool {
    DEST_BLEND_LOWERING.load(core::sync::atomic::Ordering::Relaxed) & bit != 0
}

/// Whether a fragment program that reads the destination colour is DECLARED as reading it.
/// `VITASLOP_GXP_DEST=0` clears this, and then such a program compiles with its output bank
/// starting at ZERO and no destination texture is bound or copied - the behaviour every build
/// before this mechanism had, kept as the A/B arm.
///
/// It gates the DECLARATION rather than the copy on purpose. Gating only the copy left the
/// module still asking for a texture the pass no longer supplied, so every such draw was
/// DROPPED - which is a third behaviour, neither arm of the A/B, and it silently made the arm
/// answer a different question than the one on its label.
static DEST_COLOR_READ: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);

/// Set by the renderer from `VITASLOP_GXP_DEST`. See [`DEST_COLOR_READ`].
pub fn set_dest_color_read(on: bool) {
    DEST_COLOR_READ.store(on, core::sync::atomic::Ordering::Relaxed);
}

/// Whether a destination-reading program that is LINEAR in the destination may be lowered to a
/// DUAL-SOURCE blend instead of a pass split. Set by the renderer from the device's features
/// (`dual-source-blending`) and `VITASLOP_GXP_DUAL_SOURCE`; off until it says so, because a
/// module using `@blend_src` does not compile on a device without the feature.
static DUAL_SOURCE_BLEND: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Set by the renderer once it knows the device. See [`DUAL_SOURCE_BLEND`].
pub fn set_dual_source_blend(on: bool) {
    DUAL_SOURCE_BLEND.store(on, core::sync::atomic::Ordering::Relaxed);
}

/// One coefficient that GATES a cross-channel destination term: a 2- or 4-byte value the
/// program multiplies a destination channel by, taken from per-draw data - a lane loaded out
/// of a memory window (pointer register `base_sa`, byte `byte` from the window's base) or a
/// register of the fragment default uniform buffer (byte `byte` of the buffer). When every
/// gate of a term is zero in a draw's data, that term is zero and the draw can take the
/// dual-source pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DualCoef {
    Window { base_sa: u32, byte: u32, len: u8 },
    Uniform { byte: u32, len: u8 },
    /// A container LITERAL that is zero: not a per-draw gate but a term that is dead in every
    /// draw. Never appears in a plan.
    Zero,
}

/// Whether THIS program's destination read may be lowered to a dual-source blend, and what a
/// DRAW has to satisfy for that lowering to be exact: the device has the feature, the program
/// reads the destination, every use of it is linear ([`dest_is_linear`]), the program does not
/// also write its own depth (the dual-source entry returns the two colour terms and nothing
/// else), and every CROSS-CHANNEL destination term is gated by window coefficients
/// ([`cross_channel_gates`]) - the returned list. Empty means the lowering is exact for every
/// draw; `None` means the program cannot be lowered.
///
/// # The lowering, and why the linearity proof is not all it needs
/// A program linear in the destination computes `out = dst * F + G` with `F` and `G` per-pixel
/// values of its own. The body is emitted as a function of the destination and evaluated
/// TWICE: `G` is the body with the destination forced to zero, `F` is the body with it forced
/// to one, minus `G`. The pipeline then blends `src0 * 1 + dst * src1` - which is exactly
/// `G + dst * F` - so the draw needs no copy of the attachment and no render-pass split. On a
/// tiler that split is a store and reload of the whole tile; the second evaluation of a short
/// blend body is far cheaper.
///
/// >>> BUT THE ROP MULTIPLIES `dst.c` BY `src1.c`, CHANNEL BY CHANNEL. A program whose red
/// output depends on the destination's ALPHA (`out.r = ... + dst.a * CDa.r`) has a term the
/// blend cannot express, and forcing all four channels to one at once folds it into `F.r` as
/// if it were `dst.r * CDa.r`. MEASURED: a baseball title's generic-blend family carries exactly
/// that term, and lowering it unconditionally drew its HUD's base-runner diamond white where
/// the split path drew it grey - 357 pixels of a frame, every one where `dst.a != dst.r`. Its
/// coefficient is a halfword in the draw's blend window, and for almost every draw it is zero;
/// so the term is gated on THAT, per draw, rather than the program refused outright.
pub fn dual_source_plan(shader: &Shader, uniform_regs: u32, literals: &[(u32, u32)]) -> Option<Vec<DualCoef>> {
    dual_source_plan_or_why(shader, uniform_regs, literals).ok()
}

/// [`dual_source_plan`], naming the reason a program is refused - a lowering that never fires
/// must be able to say why.
/// `literals` are the container literals preloaded into SA registers (`(register, raw
/// word)`): a zero literal kills the term it multiplies, a nonzero one cannot gate it.
pub fn dual_source_plan_or_why(shader: &Shader, uniform_regs: u32, literals: &[(u32, u32)]) -> Result<Vec<DualCoef>, String> {
    if !DUAL_SOURCE_BLEND.load(core::sync::atomic::Ordering::Relaxed) {
        return Err("the device gate is off".into());
    }
    if !declares_dest_color(shader) {
        return Err("no destination read".into());
    }
    if shader.instrs.iter().any(|i| i.op == Op::DepthF) {
        return Err("writes its own depth".into());
    }
    if !dest_is_linear(shader) {
        return Err("not linear in the destination".into());
    }
    cross_channel_gates(shader, uniform_regs, literals)
}

/// The program as it RUNS: its secondary stream (which fills SA registers - the memory loads
/// live there) followed by its primary stream. An analysis that follows a coefficient back to
/// the load that fetched it has to see both.
pub fn with_secondary(program: &crate::container::Program, primary: &Shader) -> Shader {
    let mut instrs = crate::usse::decode_secondary_shader(program).instrs;
    instrs.extend(primary.instrs.iter().cloned());
    Shader { kind: primary.kind, instrs }
}

/// Whether the dual-source lowering applies to this program for SOME draw. See
/// [`dual_source_plan`].
pub fn dual_source_eligible(shader: &Shader, uniform_regs: u32, literals: &[(u32, u32)]) -> bool {
    dual_source_plan(shader, uniform_regs, literals).is_some()
}

/// What a value carries of the destination: which destination channels it depends on, and
/// per channel whether that dependence is gated by window coefficients (all of which being
/// zero kills it) or cannot be killed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Dep {
    /// Per destination channel 0..3: `None` = no dependence, `Some(None)` = ungated,
    /// `Some(Some(gates))` = zero whenever every gate is zero.
    on: [Option<Option<Vec<DualCoef>>>; 4],
}

impl Dep {
    fn is_clean(&self) -> bool {
        self.on.iter().all(|d| d.is_none())
    }
    fn union(&self, o: &Dep) -> Dep {
        let mut r = Dep::default();
        for c in 0..4 {
            r.on[c] = match (&self.on[c], &o.on[c]) {
                (None, x) | (x, None) => x.clone(),
                (Some(None), _) | (_, Some(None)) => Some(None),
                (Some(Some(a)), Some(Some(b))) => {
                    let mut v = a.clone();
                    for g in b {
                        if !v.contains(g) {
                            v.push(*g);
                        }
                    }
                    Some(Some(v))
                }
            };
        }
        r
    }
    /// The dependence after multiplication by a coefficient loaded from a window: every
    /// term now vanishes when that coefficient does.
    fn gated_by(&self, coef: DualCoef) -> Dep {
        let mut r = self.clone();
        for c in 0..4 {
            r.on[c] = match &self.on[c] {
                None => None,
                Some(None) => Some(Some(vec![coef])),
                Some(Some(v)) => {
                    let mut v = v.clone();
                    if !v.contains(&coef) {
                        v.push(coef);
                    }
                    Some(Some(v))
                }
            };
        }
        r
    }
}

/// The per-channel destination dependence of the program's colour output, reduced to the
/// window coefficients that gate its CROSS-channel terms - see [`dual_source_plan`].
///
/// Values are tracked per HALF-LANE of the register file (an F16 program addresses halves; an
/// F32 instruction on register `i` covers halves `2i` and `2i+1`), which is the one addressing
/// under which a 32-bit move of a register holding two halves and a half-precision read of one
/// of them agree. Componentwise ops (the only ones the linearity proof admits) feed channel
/// `c` of the destination from channel `c` of each source, after the source swizzle - so a
/// cross-channel term can only enter through a swizzle, which is exactly where it is caught.
/// A coefficient is a lane a `ldmem` wrote from the window; multiplying a destination term by
/// one gates it. Anything this cannot follow - an indexed operand, a coefficient with no
/// window provenance - is refused rather than guessed.
fn cross_channel_gates(shader: &Shader, uniform_regs: u32, literals: &[(u32, u32)]) -> Result<Vec<DualCoef>, String> {
    use crate::ir::{Bank, BitwiseKind};
    use std::collections::HashMap;
    fn bank_id(b: Bank) -> Option<u8> {
        Some(match b {
            Bank::Temp => 0,
            Bank::PrimaryAttr => 1,
            Bank::Output => 2,
            Bank::SecondaryAttr => 3,
            Bank::Internal => 4,
            Bank::Constant => 5,
            Bank::Global => 6,
            Bank::Immediate => 7,
            Bank::Indexed | Bank::Index | Bank::Raw(_) => return None,
        })
    }
    // The half lanes an operand's channel `sel` (a lane selector 0..3) covers, low half first.
    fn lanes(bank: Bank, index: u8, sel: u32, half: bool) -> Vec<u32> {
        let half = half && !matches!(bank, Bank::Internal);
        if half {
            vec![2 * index as u32 + sel]
        } else {
            let r = index as u32 + sel;
            vec![2 * r, 2 * r + 1]
        }
    }
    let is_move = |op: &Op| matches!(op, Op::Mov | Op::Bitwise { kind: BitwiseKind::Or, imm: Some(0), .. });
    let color = color_output(shader);
    let precision = color_precision(shader, color);
    let mut dep: HashMap<(u8, u32), Dep> = HashMap::new();
    let mut prov: HashMap<(u8, u32), DualCoef> = HashMap::new();
    // The fragment default uniform buffer: SA registers below the carried extent hold
    // per-draw bytes the renderer captures (`frag_sa`), so a coefficient read from one is a
    // gate the draw can be tested against. Registers above it are container LITERALS, which
    // no draw changes - a term they multiply is refused as ungated if it crosses channels.
    for r in 0..uniform_regs {
        for h in 0..2u32 {
            prov.insert((3, 2 * r + h), DualCoef::Uniform { byte: 4 * r + 2 * h, len: 2 });
        }
    }
    for &(r, v) in literals {
        for h in 0..2u32 {
            if (v >> (16 * h)) & 0xffff == 0 {
                prov.insert((3, 2 * r + h), DualCoef::Zero);
            } else {
                prov.remove(&(3, 2 * r + h));
            }
        }
    }
    // The destination seeds the O bank in the colour's own layout (`dest_color_init`).
    for c in 0..4u32 {
        let mut d = Dep::default();
        d.on[c as usize] = Some(None);
        match precision {
            ColorPrecision::F32 => {
                dep.insert((2, 2 * c), d.clone());
                dep.insert((2, 2 * c + 1), d);
            }
            ColorPrecision::F16 => {
                dep.insert((2, c), d);
            }
            ColorPrecision::Fx8 => {
                let all = Dep { on: [Some(None), Some(None), Some(None), Some(None)] };
                dep.insert((2, 0), all.clone());
                dep.insert((2, 1), all);
            }
        }
    }
    let trace = std::env::var_os("VITASLOP_GXP_DUAL_TRACE").is_some();
    if trace {
        eprintln!("  uniform_regs={uniform_regs} literals={literals:x?}");
    }
    for instr in &shader.instrs {
        let half = instr.half_precision;
        if let Op::MemLoad { elements, offset_bytes } = instr.op {
            if std::env::var_os("VITASLOP_GXP_DUAL_TRACE").is_some() {
                eprintln!("  ldmem elements={elements} offset={offset_bytes} dest={:?} srcs={:?}", instr.dest.as_ref().map(|d| (d.bank, d.index)), instr.srcs.iter().map(|o| (o.bank, o.index)).collect::<Vec<_>>());
            }
            let (Some(d), Some(p)) = (instr.dest.as_ref(), instr.srcs.first()) else {
                return Err("a load with no destination or pointer".into());
            };
            let db = bank_id(d.bank).ok_or("an indexed load destination")?;
            for k in 0..elements as u32 {
                for h in 0..2u32 {
                    let lane = 2 * (d.index as u32 + k) + h;
                    dep.remove(&(db, lane));
                    if p.bank == Bank::SecondaryAttr && instr.srcs.len() == 1 {
                        prov.insert(
                            (db, lane),
                            DualCoef::Window { base_sa: p.index as u32, byte: offset_bytes + 4 * k + 2 * h, len: 2 },
                        );
                    } else {
                        prov.remove(&(db, lane));
                    }
                }
            }
            continue;
        }
        let Some(d) = instr.dest.as_ref() else { continue };
        let db = bank_id(d.bank).ok_or_else(|| format!("an indexed destination at {:?}", instr.op))?;
        // Every channel's sources are read BEFORE any channel's lanes are written, so a
        // swizzle reading a lane this instruction also writes sees the old value. Per source
        // and channel: the dependence and coefficient of EACH half lane it covers.
        let read = |op: &Operand, c: usize| -> Option<Vec<(Dep, Option<DualCoef>)>> {
            let b = bank_id(op.bank)?;
            let sel = op.swizzle[c] as u32;
            if sel > 3 {
                return Some(vec![(Dep::default(), None)]);
            }
            Some(
                lanes(op.bank, op.index, sel, half)
                    .into_iter()
                    .map(|l| (dep.get(&(b, l)).cloned().unwrap_or_default(), prov.get(&(b, l)).copied()))
                    .collect(),
            )
        };
        let mut per_channel: Vec<Option<Vec<Vec<(Dep, Option<DualCoef>)>>>> = Vec::with_capacity(4);
        for c in 0..4usize {
            if !instr.write_mask[c] {
                per_channel.push(None);
                continue;
            }
            let v = instr.srcs.iter().map(|s| read(s, c)).collect::<Option<Vec<_>>>();
            per_channel.push(Some(v.ok_or_else(|| format!("an indexed source at {:?}", instr.op))?));
        }
        let _ = read;
        for c in 0..4usize {
            let Some(srcs) = per_channel[c].take() else { continue };
            let dlanes = lanes(d.bank, d.index, c as u32, half);
            if trace {
                eprintln!(
                    "  {:?} half={} dest={:?}[{}] mask={:?} srcs={:?} deps={:?}",
                    instr.op, half, (d.bank, d.index), c, instr.write_mask,
                    instr.srcs.iter().map(|o| (o.bank, o.index, o.swizzle)).collect::<Vec<_>>(),
                    srcs.iter().map(|ls| ls.iter().map(|(dd, k)| (dd.on.iter().map(|x| x.as_ref().map(|g| g.as_ref().map(|v| v.len()))).collect::<Vec<_>>(), *k)).collect::<Vec<_>>()).collect::<Vec<_>>()
                );
            }
            // A 32-bit MOVE of a register holding two halves moves each half on its own; an
            // arithmetic op on a 32-bit float consumes both halves as one value.
            let per_lane_moves = is_move(&instr.op) && dlanes.len() == 2 && srcs.len() == 1 && srcs[0].len() == 2;
            let combine = |lane_of_src: &dyn Fn(&Vec<(Dep, Option<DualCoef>)>) -> (Dep, Option<DualCoef>)| -> Result<Dep, String> {
                let s: Vec<(Dep, Option<DualCoef>)> = srcs.iter().map(|ls| lane_of_src(ls)).collect();
                if !s.iter().any(|(d, _)| !d.is_clean()) {
                    return Ok(Dep::default());
                }
                Ok(match instr.op {
                    Op::Add | Op::Mov => s.iter().fold(Dep::default(), |a, (d, _)| a.union(d)),
                    Op::Bitwise { kind: BitwiseKind::Or, imm: Some(0), .. } if s.len() == 1 => s[0].0.clone(),
                    Op::Mul | Op::Mad => {
                        let (a, b) = (&s[0], s.get(1).ok_or("a one-operand multiply")?);
                        let product = match (a.0.is_clean(), b.0.is_clean()) {
                            (true, true) => Dep::default(),
                            (false, true) => match b.1 {
                                Some(DualCoef::Zero) => Dep::default(),
                                Some(k) => a.0.gated_by(k),
                                None => a.0.clone(),
                            },
                            (true, false) => match a.1 {
                                Some(DualCoef::Zero) => Dep::default(),
                                Some(k) => b.0.gated_by(k),
                                None => b.0.clone(),
                            },
                            (false, false) => return Err(format!("a product of two destination terms at {:?}", instr.op)),
                        };
                        match (instr.op, s.get(2)) {
                            (Op::Mad, Some((cd, _))) => product.union(cd),
                            _ => product,
                        }
                    }
                    op => return Err(format!("{op:?} over a destination term")),
                })
            };
            let news: Vec<Dep> = if per_lane_moves {
                vec![combine(&|ls| ls[0].clone())?, combine(&|ls| ls[1].clone())?]
            } else {
                // One value over all of a source's lanes: union the deps; a coefficient only
                // if the whole word is one (a 32-bit read of a window word or uniform register).
                let whole = |ls: &Vec<(Dep, Option<DualCoef>)>| -> (Dep, Option<DualCoef>) {
                    let d = ls.iter().fold(Dep::default(), |a, (x, _)| a.union(x));
                    let k = match ls.as_slice() {
                        [(_, k)] => *k,
                        [(_, Some(DualCoef::Window { base_sa, byte, .. })), (_, Some(DualCoef::Window { .. }))] => {
                            Some(DualCoef::Window { base_sa: *base_sa, byte: *byte, len: 4 })
                        }
                        [(_, Some(DualCoef::Uniform { byte, .. })), (_, Some(DualCoef::Uniform { .. }))] => {
                            Some(DualCoef::Uniform { byte: *byte, len: 4 })
                        }
                        [(_, Some(DualCoef::Zero)), (_, Some(DualCoef::Zero))] => Some(DualCoef::Zero),
                        _ => None,
                    };
                    (d, k)
                };
                let one = combine(&whole)?;
                dlanes.iter().map(|_| one.clone()).collect()
            };
            for (l, new) in dlanes.iter().zip(news) {
                prov.remove(&(db, *l));
                if instr.pred == Predicate::Always {
                    if new.is_clean() {
                        dep.remove(&(db, *l));
                    } else {
                        dep.insert((db, *l), new);
                    }
                } else if !new.is_clean() {
                    let merged = dep.get(&(db, *l)).map_or(new.clone(), |old| old.union(&new));
                    dep.insert((db, *l), merged);
                }
            }
        }
    }
    // The colour the program returns, channel by channel.
    let (ob, base) = match color {
        ColorOutput::NativeO0 => (2u8, 0u32),
        ColorOutput::NonNativePa(b) => (1u8, b),
    };
    let mut gates: Vec<DualCoef> = Vec::new();
    for c in 0..4u32 {
        let ls: Vec<u32> = match precision {
            ColorPrecision::F32 => vec![2 * (base + c), 2 * (base + c) + 1],
            ColorPrecision::F16 => vec![2 * base + c],
            ColorPrecision::Fx8 => vec![2 * base, 2 * base + 1],
        };
        let mut d = Dep::default();
        for l in ls {
            if let Some(x) = dep.get(&(ob, l)) {
                d = d.union(x);
            }
        }
        for (src, on) in d.on.iter().enumerate() {
            if src as u32 == c {
                continue;
            }
            match on {
                None => {}
                Some(None) => {
                    return Err(format!("output channel {c} depends on destination channel {src} with no per-draw coefficient gating it"))
                }
                Some(Some(v)) => {
                    for g in v {
                        if !gates.contains(g) {
                            gates.push(*g);
                        }
                    }
                }
            }
        }
    }
    Ok(gates)
}

/// Whether the destination read is declared at all - [`DEST_COLOR_READ`] AND the program.
pub fn declares_dest_color(shader: &Shader) -> bool {
    DEST_COLOR_READ.load(core::sync::atomic::Ordering::Relaxed) && reads_output_bank(shader)
}

/// Recognise a blend the fragment program performs ITSELF over the destination colour, rewrite
/// the program to emit only its SOURCE term, and return the equation for the pipeline.
///
/// # Why this exists
/// A fragment program's output registers are the ROP's destination colour fed back, and a
/// program is free to blend in ordinary ALU. Serving that faithfully needs a copy of the colour
/// attachment taken immediately before the draw, which costs a RENDER-PASS SPLIT - and on one
/// retail title that was **42 splits over 145 draws in a single frame**, five shader pairs
/// accounting for 41 of them. A tiling GPU stores and reloads its tiles at every split.
///
/// Most of those equations are ordinary blends written longhand, and a blend is exactly what the
/// pipeline can do for free. This is the same move [`crate::rop_blend`] makes for the SOP2
/// epilogue form, one level up: recover the equation, emit only the source term, let the ROP do
/// the rest. What is left over - a colour grade that takes a DOT PRODUCT of the destination, a
/// `max` against it - is not a blend in any hardware's sense and keeps its split.
///
/// # The shapes, and the algebra that says each is exact
/// Every one was read off a shipped program; `C` is the register the colour epilogue copies,
/// `O` the output register, `K` a uniform, `X` an ordinary value.
///
/// * **LERP, alpha already in the colour register** (`frag_8713c840`, `frag_8713c9a0` - a
///   title's whole HUD, 26 of the 42 splits):
///   ```text
///     Add  T = -O + C          ; C.w already holds the alpha (a Pack put it there)
///     Mad  C = T * C.wwww + O  ; = O + (C - O) * C.w
///     Mov  O = C
///   ```
///   `out = C*a + O*(1-a)` with `a = C.w`: `SrcAlpha / OneMinusSrcAlpha`, shader emitting `C`.
///   The ALPHA channel closes too - the program computes `(C.w - O.w)*C.w + O.w` and the blend
///   computes `C.w*C.w + O.w*(1-C.w)`, the same value.
///
/// * **LERP, factor recomputed from the same two registers** (`frag_912c9380`, `frag_9129ebe0`):
///   ```text
///     Mad  T = C * K - O
///     Mul  C = K.wwww * C.wwww ; the lerp factor, broadcast
///     Mad  C = T * C + O
///     Mov  O = C
///   ```
///   `out = O + (C*K - O) * (K.w*C.w)`, and that factor IS `(C*K).w` - so the same
///   `SrcAlpha / OneMinusSrcAlpha` with the shader emitting `C*K`, which is what `T` becomes
///   once its `- O` term is dropped. The epilogue is redirected to read `T`.
///
/// * **MODULATE** (`frag_87621c00`, 8 of the 42 splits): `Mul C = O * K; Mov O = C` is
///   `Zero / Src` with the shader emitting `K`.
///
/// * **ADDITIVE, destination alpha kept** (`frag_865af860`, `frag_907b4f80`):
///   `Mad C = X * K + O; Pack C.w = O.w; Mov O = C` is `One / One` on colour and `Zero / One`
///   on alpha. The Pack is REQUIRED for this arm: without it the alpha equation is a different
///   one and this returns `None` rather than guess which.
///
/// # What is refused, and why that matters more than what is accepted
/// Every field is pinned: the output operand must be register 0 read whole with no modifier, the
/// write masks must cover all four channels, the instructions must be unpredicated, and the
/// epilogue `Mov` must be the LAST instruction. After the rewrite the program must not read the
/// output bank AT ALL - that final check is what makes a partial match impossible, because a
/// program still reading the destination somewhere else would be blended twice. Anything
/// unmatched returns `None`, keeps its ALU blend, and pays for a destination copy.
pub fn lower_dest_blend(shader: &mut Shader) -> Option<DestBlend> {
    if DEST_BLEND_LOWERING.load(core::sync::atomic::Ordering::Relaxed) == 0 {
        return None;
    }
    if shader.kind != crate::container::ProgramKind::Fragment {
        return None;
    }
    let mut work = shader.clone();
    let blend = match_dest_blend(&mut work)?;
    // The whole-program guard: a rewrite that leaves ANY other read of the destination would be
    // blended twice, once in the shader and once by the ROP.
    if reads_output_bank(&work) {
        return None;
    }
    *shader = work;
    Some(blend)
}

/// A plain `.xyzw` operand with no modifier.
fn ident(op: &Operand) -> bool {
    op.swizzle == [0, 1, 2, 3] && !op.abs && !op.neg
}

/// The output register the colour epilogue writes, read whole: `o0.xyzw`, optionally negated.
fn is_output0(op: &Operand, neg: bool) -> bool {
    op.bank == Bank::Output
        && op.index == 0
        && op.swizzle == [0, 1, 2, 3]
        && !op.abs
        && op.neg == neg
}

/// The same register, whatever the swizzle.
fn same_reg(a: &Operand, b: &Operand) -> bool {
    a.bank == b.bank && a.index == b.index
}

/// `reg.wwww` - one channel broadcast, which is how a scalar alpha reaches four lanes.
fn broadcast_w(op: &Operand) -> bool {
    op.swizzle == [3, 3, 3, 3] && !op.abs && !op.neg
}

fn always(i: &Instr) -> bool {
    i.pred == Predicate::Always && i.write_mask == [true; 4]
}

/// The tail-matching half of [`lower_dest_blend`], on a shader it is free to mutate.
fn match_dest_blend(sh: &mut Shader) -> Option<DestBlend> {
    const LERP: DestBlend = DestBlend {
        color: BlendTerm { src: BlendFactor::SrcAlpha, dst: BlendFactor::OneMinusSrcAlpha },
        alpha: BlendTerm { src: BlendFactor::SrcAlpha, dst: BlendFactor::OneMinusSrcAlpha },
    };
    const MODULATE: DestBlend = DestBlend {
        color: BlendTerm { src: BlendFactor::Zero, dst: BlendFactor::Src },
        alpha: BlendTerm { src: BlendFactor::Zero, dst: BlendFactor::Src },
    };
    const ADDITIVE: DestBlend = DestBlend {
        color: BlendTerm { src: BlendFactor::One, dst: BlendFactor::One },
        alpha: BlendTerm { src: BlendFactor::Zero, dst: BlendFactor::One },
    };
    let n = sh.instrs.len();
    if n < 2 {
        return None;
    }
    // The colour epilogue: the LAST instruction, a plain move of one register into `o0`.
    let mov = &sh.instrs[n - 1];
    if !matches!(mov.op, Op::Mov) || mov.pred != Predicate::Always {
        return None;
    }
    let dest = mov.dest.as_ref()?;
    if dest.bank != Bank::Output || dest.index != 0 {
        return None;
    }
    let csrc = mov.srcs.first()?.clone();
    if csrc.abs || csrc.neg {
        return None;
    }

    // ---- MODULATE: `Mul C = O * K` ----
    if let Some(prev) = sh.instrs.get(n - 2)
        && matches!(prev.op, Op::Mul)
        && always(prev)
        && prev.dest.as_ref().is_some_and(|d| same_reg(d, &csrc))
        && prev.srcs.len() == 2
    {
        let (a, b) = (&prev.srcs[0], &prev.srcs[1]);
        let k = match (is_output0(a, false), is_output0(b, false)) {
            (true, false) if ident(b) => Some(b.clone()),
            (false, true) if ident(a) => Some(a.clone()),
            _ => None,
        };
        if let Some(k) = k
            && form_enabled(FORM_MODULATE)
        {
            let i = n - 2;
            sh.instrs[i].op = Op::Mov;
            sh.instrs[i].srcs = vec![k];
            return Some(MODULATE);
        }
    }

    // ---- LERP and ADDITIVE both end in a `Mad ... + O`. ADDITIVE may carry a `Pack C.w = O.w`
    // ---- between that Mad and the epilogue.
    let mut pack_at: Option<usize> = None;
    let mut mad_at = n.checked_sub(2)?;
    if let Some(p) = sh.instrs.get(mad_at)
        && matches!(p.op, Op::Pack { .. })
        && p.pred == Predicate::Always
        && p.write_mask == [false, false, false, true]
        && p.dest.as_ref().is_some_and(|d| same_reg(d, &csrc))
        && p.srcs.first().is_some_and(|s| {
            s.bank == Bank::Output && s.index == 0 && s.swizzle[3] == 3 && !s.neg && !s.abs
        })
    {
        pack_at = Some(mad_at);
        mad_at = mad_at.checked_sub(1)?;
    }
    let mad = sh.instrs.get(mad_at)?.clone();
    if !matches!(mad.op, Op::Mad)
        || !always(&mad)
        || mad.srcs.len() != 3
        || !is_output0(&mad.srcs[2], false)
    {
        return None;
    }
    let mad_dest = mad.dest.as_ref()?.clone();
    if !same_reg(&mad_dest, &csrc) {
        return None;
    }

    // ---- ADDITIVE: `Mad C = X * K + O` with the destination alpha packed back in ----
    if let Some(pack) = pack_at {
        if !ident(&mad.srcs[0]) || !ident(&mad.srcs[1]) || !form_enabled(FORM_ADDITIVE) {
            return None;
        }
        sh.instrs[mad_at].op = Op::Mul;
        sh.instrs[mad_at].srcs.truncate(2);
        sh.instrs.remove(pack);
        return Some(ADDITIVE);
    }

    // ---- LERP: `Mad C = T * a + O` ----
    let t = mad.srcs[0].clone();
    let factor = mad.srcs[1].clone();
    if !ident(&t) || !form_enabled(FORM_LERP) {
        return None;
    }
    let prev_at = mad_at.checked_sub(1)?;
    let prev = sh.instrs.get(prev_at)?.clone();

    // ---- Forms A and C: `Add D = -O + S` then `Mad C = D * S.wwww + O`.
    //
    // Both are the SAME equation - `out = S*S.w + O*(1-S.w)`, `SrcAlpha / OneMinusSrcAlpha`
    // with the shader emitting `S`. They differ only in WHICH of the two registers the colour
    // epilogue copies, and that decides whether the epilogue has to be redirected:
    //
    //   * A (`frag_8713c840`, `frag_8713c9a0`): `S` IS the register the epilogue copies, and
    //     the difference goes to a separate one. Dropping the two instructions leaves the
    //     epilogue reading `S` already.
    //   * C (`frag_87140cb0`, `frag_87151ba0`): the DIFFERENCE is written back over the
    //     register the epilogue copies, and the source term lives in another (a temp, or a
    //     second PA register). Dropping the two would leave the epilogue reading a register
    //     nothing writes, so it is redirected at `S` - the same move form B already makes.
    //
    // C is worth having on its own numbers: it is 2 of this title's 6 remaining
    // destination-reading fragment programs, and each one is a render-pass split per draw.
    if matches!(prev.op, Op::Add)
        && always(&prev)
        && prev.dest.as_ref().is_some_and(|d| same_reg(d, &t))
        && prev.srcs.len() == 2
        && is_output0(&prev.srcs[0], true)
        && ident(&prev.srcs[1])
        && broadcast_w(&factor)
        && same_reg(&factor, &prev.srcs[1])
    {
        let src_term = prev.srcs[1].clone();
        let redirect = !same_reg(&src_term, &csrc);
        if redirect && !form_enabled(FORM_LERP_SRC) {
            return None;
        }
        sh.instrs.remove(mad_at);
        sh.instrs.remove(prev_at);
        if redirect {
            let last = sh.instrs.len() - 1;
            let sw = sh.instrs[last].srcs[0].swizzle;
            sh.instrs[last].srcs[0] = Operand { swizzle: sw, ..src_term };
        }
        return Some(LERP);
    }

    // Form B: the lerp factor is recomputed as `K.w * C.w`, and `T = C * K - O`.
    if !matches!(prev.op, Op::Mul)
        || !always(&prev)
        || !prev.dest.as_ref().is_some_and(|d| same_reg(d, &csrc))
        || prev.srcs.len() != 2
        || !broadcast_w(&prev.srcs[0])
        || !broadcast_w(&prev.srcs[1])
        || !same_reg(&factor, &csrc)
        || !ident(&factor)
    {
        return None;
    }
    let src_at = prev_at.checked_sub(1)?;
    let src_mad = sh.instrs.get(src_at)?.clone();
    if !matches!(src_mad.op, Op::Mad)
        || !always(&src_mad)
        || !src_mad.dest.as_ref().is_some_and(|d| same_reg(d, &t))
        || src_mad.srcs.len() != 3
        || !is_output0(&src_mad.srcs[2], true)
        || !ident(&src_mad.srcs[0])
        || !ident(&src_mad.srcs[1])
    {
        return None;
    }
    // The two broadcasts must be the `w` of the same two registers the source term multiplies,
    // or the factor is not `(C*K).w` and this lowering would be a guess.
    let (p0, p1) = (&prev.srcs[0], &prev.srcs[1]);
    let (m0, m1) = (&src_mad.srcs[0], &src_mad.srcs[1]);
    if !((same_reg(p0, m0) && same_reg(p1, m1)) || (same_reg(p0, m1) && same_reg(p1, m0))) {
        return None;
    }
    // `T` becomes the source term itself, and the epilogue is redirected to read it.
    sh.instrs[src_at].op = Op::Mul;
    sh.instrs[src_at].srcs.truncate(2);
    sh.instrs.remove(mad_at);
    sh.instrs.remove(prev_at);
    let last = sh.instrs.len() - 1;
    let sw = sh.instrs[last].srcs[0].swizzle;
    sh.instrs[last].srcs[0] = Operand { swizzle: sw, ..t };
    Some(LERP)
}

/// >>> IS THE PROGRAM LINEAR IN THE DESTINATION COLOUR - `out = dst*F + G`?
///
/// # Why this question, and why it needs a PROOF rather than three hand-derivations
/// [`match_dest_blend`] recovers a blend by matching the last few instructions against known
/// SHAPES. That works while a title's blends are the handful of idioms its compiler emits, and
/// it ran out on a sports title: its three destination-reading fragment programs match nothing,
/// read the destination TWICE each, and interleave the two reads through eight instructions of
/// per-pixel arithmetic. Adding a fourth, fifth and sixth shape for them would be three more
/// tail patterns that the next title breaks again.
///
/// They do have one thing in common, and it is a PROPERTY rather than a pattern: every one is
/// LINEAR in the destination. Collect the terms of any of them and it is
///
/// ```text
///   out = dst * F + G      with F and G computed per pixel from varyings and uniforms
/// ```
///
/// Fixed-function blending cannot express that - it offers exactly one shader-computed factor
/// (`BlendFactor::Src`, which is what MODULATE uses) and this needs two - but DUAL-SOURCE
/// blending can: emit `G` as `src0` and `F` as `src1`, and the ROP computes `One*src0 +
/// Src1*dst`. The user's phone offers `dual-source-blending`
/// [[vitaslop-the-phone-has-dual-source-blending]].
///
/// # What this function proves, and what it deliberately refuses
/// The decomposition is only sound if the program really is linear, so this TAINTS every value
/// derived from the output bank and checks that each one is consumed only by an operation that
/// is linear in it:
///
/// * `Add` - linear in both sides.
/// * `Mul` / `Mad` - linear in a tainted operand ONLY IF the other multiplicand is untainted.
///   `dst * dst` is quadratic and is refused, which is the whole point of tracking taint rather
///   than pattern-matching a tail.
/// * `Mov` - a swizzled copy, linear.
///
/// EVERYTHING ELSE with a tainted source is refused: `Min`/`Max`/`Cmov` are piecewise (linear
/// on each side of a branch the destination itself can move), `Dot` multiplies lanes together,
/// `Rcp`/`Rsq`/`Log`/`Exp` are not linear at all, and `Sop2` is a fixed-point combiner whose
/// coefficients can BE the operand ([`SopFactor`] says so). A refusal costs a render-pass split,
/// which is what happens today; a wrong acceptance silently paints a different picture.
///
/// >>> `Bitwise` IS ACCEPTED FOR ONE SPELLING ONLY, AND IT IS NOT AN INDULGENCE. All three of
/// that title's programs read their second output register as `o1 | 0` - a 32-bit OR with an
/// immediate ZERO, which is the identity on the bit pattern and is how this compiler spells "move
/// these bits". A move is linear. Any other bitwise op, any other immediate, or a register second
/// operand is refused: a shift or a mask of a float's bits is not linear in the float.
pub fn dest_is_linear(shader: &Shader) -> bool {
    use crate::ir::{Bank, BitwiseKind};
    // Tainted (bank, index) pairs: values derived from the output bank. Whole registers, not
    // lanes - a per-lane taint would be more precise and there is no evidence any program needs
    // it, and the imprecision is on the SAFE side: it can only refuse a program, never accept a
    // nonlinear one.
    // Keyed on a small local bank id, so the IR does not grow a `Hash`/`Ord` derive for one
    // analysis's scratch set and nothing allocates per operand.
    fn bank_id(b: Bank) -> u8 {
        match b {
            Bank::Temp => 0,
            Bank::PrimaryAttr => 1,
            Bank::Output => 2,
            Bank::SecondaryAttr => 3,
            Bank::Internal => 4,
            Bank::Constant => 5,
            Bank::Global => 6,
            Bank::Immediate => 7,
            Bank::Indexed => 8,
            Bank::Index => 9,
            Bank::Raw(_) => 10,
        }
    }
    let key = |op: &Operand| (bank_id(op.bank), op.index);
    let mut tainted: std::collections::HashSet<(u8, u8)> = Default::default();
    let is_tainted = |t: &std::collections::HashSet<(u8, u8)>, op: &Operand| {
        op.bank == Bank::Output || t.contains(&key(op))
    };
    let mut any = false;

    for instr in &shader.instrs {
        let srcs = &instr.srcs;
        let hit: Vec<bool> = srcs.iter().map(|s| is_tainted(&tainted, s)).collect();
        let touches = hit.iter().any(|h| *h);
        if !touches {
            // Still have to KILL a tainted destination that is overwritten by clean data -
            // otherwise a register reused as a scratch stays tainted for the rest of the
            // program and the analysis refuses shaders it should accept.
            if let Some(d) = instr.dest.as_ref()
                && instr.write_mask == [true; 4]
                && instr.pred == Predicate::Always
            {
                tainted.remove(&key(d));
            }
            continue;
        }
        any = true;
        // `|dst|` is not linear in `dst` whatever the instruction does with it afterwards, so
        // an absolute-value modifier on a tainted operand is refused outright (negation is
        // linear and passes).
        if srcs.iter().zip(&hit).any(|(s, h)| *h && s.abs) {
            return false;
        }
        let linear = match instr.op {
            Op::Add | Op::Mov => true,
            // `Mad dest = a*b + c`: the PRODUCT must have at most one tainted side. `c` may be
            // tainted freely - that is the `+ dst` every one of these blends ends on.
            Op::Mad => !(hit.first() == Some(&true) && hit.get(1) == Some(&true)),
            Op::Mul => !(hit.first() == Some(&true) && hit.get(1) == Some(&true)),
            // See the note above: `x | 0` is a move of the bit pattern and nothing else is.
            Op::Bitwise { kind: BitwiseKind::Or, imm: Some(0), .. } => srcs.len() == 1,
            _ => false,
        };
        if !linear {
            return false;
        }
        // A predicated or partially-masked write leaves the register holding a MIX of the old
        // value and the new one. That is still linear if both are, but this analysis does not
        // track the old value once it has been overwritten, so it taints conservatively.
        if let Some(d) = instr.dest.as_ref() {
            tainted.insert(key(d));
        }
    }
    any
}

/// Whether any instruction SOURCES the output bank - see [`BindingPlan::reads_dest_color`].
///
/// Conservative on purpose: a read anywhere in the program counts, without asking whether the
/// program had already written that register. Getting it wrong the other way costs a black or
/// stale destination on a draw that blends, which is exactly the failure this exists to end;
/// getting it wrong this way costs one attachment copy on a draw that did not need it.
///
/// # The SOP2 family is excluded, and that is not an exception - it is the other answer
/// An 8-bit SOP2 ([`Op::Sop2`]) whose second operand is the output register is the ROP blend
/// *by construction*, and this translator already has an answer for that one:
/// [`crate::rop_blend`] recovers the equation as PIPELINE state and the emitter renders the
/// instruction as its source term. Counting it here as well would ask for an attachment copy
/// the draw does not need - and on a program where `rop_blend` DID recognise the word, it would
/// apply the destination twice, once in the shader and once in the blend.
///
/// MEASURED: one title's whole corpus has exactly one such program (`frag_81a7faa4`, a
/// `PackUnorm8` + SOP2 alpha epilogue whose second coefficient is ZERO), and without this it
/// would pay a render-pass split per draw for a term multiplied by nothing.
pub fn reads_output_bank(shader: &Shader) -> bool {
    shader.instrs.iter().any(|i| {
        !matches!(i.op, Op::Sop2 { .. })
            && i.srcs.iter().any(|s| s.bank == Bank::Output)
    })
}

/// The WGSL that seeds the O bank with the DESTINATION colour, for a program that reads it.
///
/// The layout is the colour's own: the register file is untyped 32-bit storage, so the halves
/// or bytes have to go back in exactly the way the program's arithmetic will read them out -
/// the same correspondence [`color_return_expr`] uses in the other direction. Reading an F16
/// destination as four F32 registers is the denormal-black failure
/// [[vitaslop-f16-colour-output]] records, run backwards.
pub(crate) fn dest_color_init(precision: ColorPrecision, dual_source: bool) -> String {
    let mut s = String::new();
    if dual_source {
        // The dual-source body takes the destination as a PARAMETER - it is evaluated once with
        // zero and once with one, see `dual_source_eligible` - so there is no texture to load.
        let _ = writeln!(s, "  let gxp_dstc = gxp_dstc_in;");
        push_dest_seed(&mut s, precision);
        return s;
    }
    // Diagnostic (`VITASLOP_GXP_DEST_POISON=<r,g,b,a>`): seed the output bank with a CONSTANT
    // instead of the attachment copy. A self-blending program mixes the destination into
    // everything it writes, so "is this draw's picture wrong because the shader is wrong or
    // because the destination it was handed is wrong" has no answer from the frame - both
    // produce a wrong colour everywhere the draw covers. A constant separates them in one run:
    // with the poison in, whatever remains of the destination in the picture is the shader's
    // own doing [[vitaslop-poison-separates-a-guest-zero-from-an-unwritten-one]].
    match dest_poison() {
        Some(c) => {
            let _ = writeln!(
                s,
                "  let gxp_dstc = vec4<f32>({:?}, {:?}, {:?}, {:?}); // POISONED",
                c[0], c[1], c[2], c[3]
            );
            let _ = writeln!(s, "  _ = textureLoad(gxp_dst, vec2<i32>(0, 0), 0);");
        }
        None => {
            let _ = writeln!(
                s,
                "  let gxp_dstc = textureLoad(gxp_dst, vec2<i32>(in.frag_coord.xy), 0);"
            );
        }
    }
    push_dest_seed(&mut s, precision);
    s
}

/// Seed the O bank from `gxp_dstc` in the colour's own layout - see [`dest_color_init`].
fn push_dest_seed(s: &mut String, precision: ColorPrecision) {
    match precision {
        ColorPrecision::F32 => {
            for c in 0..4u32 {
                let _ = writeln!(s, "  o[{c}] = bitcast<u32>(gxp_dstc.{});", comp(c));
            }
        }
        ColorPrecision::F16 => {
            let _ = writeln!(s, "  o[0] = pack2x16float(gxp_dstc.xy);");
            let _ = writeln!(s, "  o[1] = pack2x16float(gxp_dstc.zw);");
        }
        ColorPrecision::Fx8 => {
            let _ = writeln!(s, "  o[0] = pack4x8unorm(gxp_dstc);");
        }
    }
}

/// `VITASLOP_GXP_DEST_POISON=<r,g,b,a>` - the constant [`dest_color_init`] seeds the output
/// bank with instead of the attachment copy. Off (and byte-identical) when unset.
fn dest_poison() -> Option<[f32; 4]> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<[f32; 4]>> = OnceLock::new();
    *CELL.get_or_init(|| {
        let raw = std::env::var("VITASLOP_GXP_DEST_POISON").ok()?;
        let v: Vec<f32> = raw.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        (v.len() == 4).then(|| [v[0], v[1], v[2], v[3]])
    })
}

/// The destination-colour texture declaration, at `@group(3) @binding(1)`.
///
/// Group 3 because a device guarantees only FOUR bind groups and the other three are spoken for
/// (vertex uniforms, fragment uniforms, samplers); group 3 already carries the per-draw depth
/// block, so it is the one group whose bind group is rebuilt often enough to also carry a view
/// that changes within a pass.
pub(crate) const GXP_DEST_DECL: &str = "@group(3) @binding(1) var gxp_dst: texture_2d<f32>;
";

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
    // A program that blends for itself reads the DESTINATION colour out of the output bank -
    // see `BindingPlan::reads_dest_color`. The texture the renderer copies the attachment into
    // lives beside the depth block in group 3.
    if plan.reads_dest_color {
        m.push_str(GXP_DEST_DECL);
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
    // A program that writes its own depth needs the interpolated one; a program that reads the
    // DESTINATION colour needs the pixel to read it AT. Either way it is the same builtin, and
    // declaring it unconditionally would defeat early-depth rejection on every other program.
    if writes_depth || plan.reads_dest_color {
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
    m.push_str(&probe_globals());
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
    // ...and the O bank starts at the DESTINATION colour for a program that blends itself,
    // because that is what the hardware seeds those registers with.
    if plan.reads_dest_color {
        m.push_str(&dest_color_init(plan.color_precision, false));
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

    let (ret, base) = match plan.color {
        ColorOutput::NativeO0 => ("o", 0),
        ColorOutput::NonNativePa(base) => ("pa", base),
    };
    // The standalone fragment wrapper has no inter-stage varyings at all (it is compiled
    // without a vertex partner), so the varying probe can never apply here.
    let color = color_return_expr(ret, base, plan.color_precision, 0);
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
    ///
    /// # ONE PA REGISTER PER COMPONENT, INCLUDING FOR A COLOUR - MEASURED
    /// A vertex COLOUR is the one attribute a packed reading is tempting for: the
    /// fixed-function colour path is F16, and one title's sky family reads a
    /// four-component half varying out of a run its vertex fills with two lanes, which two
    /// packed pairs would explain exactly.
    /// **It is refuted by the frame.** Delivering every `SEMANTIC_COLOR` attribute as
    /// `ceil(n / 2)` registers of packed halves takes that title's tree/scenery pair
    /// (`5bcabf3a0a944a13`, 33,762 pixels) from `(50, 65, 38)` to `(0, 2, 0)` - BLACK - because
    /// its vertex reads the same attribute one F32 component per register. Colours are not
    /// packed, so whatever feeds that sky's third modulate component, it is not this.
    pub components: u32,
    /// The constant each lane is fed when the GUEST binds fewer components than the shader
    /// declares - per lane, because two shipping titles need opposite values and no property of
    /// the binding separates them. Decided by [`crate::attrflow`] from what each lane FEEDS;
    /// see that module for the rule and the two frames that fix it.
    ///
    /// [`plan_vertex_bindings`] cannot answer it - the question spans the LINKED pair, since a
    /// lane's only use is often a modulate in the fragment stage - so a plan on its own carries
    /// the standing 1.0 and [`crate::link::link_programs`] overwrites it. The renderer uses it
    /// only for lanes above the guest's binding.
    pub surplus_fill: [crate::attrflow::Fill; 4],
}

/// One guest-memory WINDOW a vertex program's 0xE8 memory loads read through: a bound uniform
/// buffer whose guest address the driver places in SA register [`MemWindow::base_sa`] and whose
/// bytes the host must upload with every draw (WGSL has no raw pointers, so the shader's loads
/// become subscripts of the bound windows - see `wgsl::emit_mem_load`).
///
/// A program can carry SEVERAL: the +0x78 table has one entry per buffer the driver hands a
/// pointer to, and a golf title's world vertex programs use three at once (its default uniform
/// buffer, a 128-byte light array and an 8,640-byte instance array). Built ONLY by
/// [`resolve_mem_windows`], which refuses (naming the reason) any program whose loads it cannot
/// tie to exactly this shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemWindow {
    /// The uniform buffer's GXM index - what `sceGxmSetVertexUniformBuffer(ctx, index, data)`
    /// binds, and what the host reads the bound address for at draw time.
    pub buffer_index: u32,
    /// The window's extent in BYTES: the `UniformBuffer` parameter's `array_size`, which is
    /// in bytes - MEASURED by exact tiling on both corpus programs that declare one (6144 =
    /// the 384-vec4 F32 member exactly; 32 = the container entry's own 8 registers exactly).
    pub bytes: u32,
    /// The SA register the driver writes the buffer's bound guest ADDRESS into
    /// (`data_container.base_sa + binding.data_slot`); the module initialises it from the
    /// window so the shader's own address arithmetic runs bit-exact.
    pub base_sa: u32,
    /// Bytes to ADD to the buffer's bound address to get the pointer the driver actually
    /// places in [`Self::base_sa`]. Zero for every buffer the driver does not also copy.
    ///
    /// # THE DEFAULT UNIFORM BUFFER'S POINTER IS NOT ITS BASE
    /// When part of a buffer is ALSO copied into the SA register file, the program reads that
    /// part as `sa[k]` and reaches only the REMAINDER through a load - so the pointer has to
    /// name the first register the driver did NOT copy, or every offset the program adds is
    /// short by the copied extent. The driver copies exactly `container 14`'s `size_regs`,
    /// which is what this is.
    ///
    /// **MEASURED by corpus closure on two of the golf title's programs, five reads, every one
    /// landing on a declared parameter's FIRST register under this reading and on nothing under
    /// `offset = 0`:**
    /// * `vert_820d6730` - container 14 is 31 registers of a declared 34, and its single
    ///   3-word read at `+0` is `sunColor` (`resource_index` 31, 3 components). The leftover is
    ///   3 registers and `sunColor` is 3 components: the container holds exactly what fits.
    /// * `vert_81d72040` - container 14 is 14 registers of a declared 28, and its four reads at
    ///   `+0`, `+8`, `+24`, `+40` are `g_DiffuseRange` (reg 14, 2), `g_Material.diffuse`
    ///   (16, 4), `g_Material.fresnel` (20, 4) and `g_Material.ambient` (24, 4) - 14 registers,
    ///   again exactly the leftover.
    ///
    /// At `offset = 0` those five reads land on `worldViewProjection[0..2]` and on
    /// `g_Material.specular`/`g_TexCoordOffset`, which is how the golf title's menu came out
    /// with a red sky: `sunColor.x` was reading the projection matrix's `1.944`.
    pub base_offset: u32,
}

impl MemWindow {
    /// Number of `vec4<u32>` elements this window's own bytes occupy.
    pub fn data_vec4s(&self) -> u32 {
        self.bytes.div_ceil(16)
    }
}

/// Where one window's words sit inside the `gxp_mem` binding: the 32-bit WORD index its first
/// byte lands on, and how many words it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemWindowPlacement {
    pub first_word: u32,
    pub words: u32,
}

/// The `gxp_mem` binding's layout for a program's windows: ONE header `vec4` per window (lane x
/// = that window's guest base address), then every window's bytes in order.
///
/// With a single window this is exactly the layout that existed before several were possible -
/// header at vec4 0, bytes from vec4 1 - so nothing about a one-window program changes.
pub fn mem_window_placements(windows: &[MemWindow]) -> Vec<MemWindowPlacement> {
    let mut vec4 = windows.len() as u32;
    windows
        .iter()
        .map(|w| {
            let at = MemWindowPlacement { first_word: vec4 * 4, words: w.bytes.div_ceil(4) };
            vec4 += w.data_vec4s();
            at
        })
        .collect()
}

/// Total `vec4<u32>` elements the `gxp_mem` uniform binding holds for these windows.
pub fn mem_window_vec4_count(windows: &[MemWindow]) -> u32 {
    windows.len() as u32 + windows.iter().map(MemWindow::data_vec4s).sum::<u32>()
}

/// The WGSL helper every module with a memory window declares: resolve a guest ADDRESS to the
/// word the bound windows hold at it.
///
/// # Why the shader dispatches on the ADDRESS rather than on which buffer a load names
/// A load's pointer register is usually not the driver-placed one - the program adds an index
/// to it first, so by the time the load runs the address lives in a temporary. Deciding which
/// window a load belongs to would therefore need dataflow through the whole program, including
/// across branches and the loops these titles use. The address itself needs none: each window
/// carries its own guest base, the windows are snapshots of guest memory, and two that overlap
/// hold the same bytes there - so the FIRST window that contains the address is always the
/// right answer.
///
/// An address inside no window reads zero. That is the same fabrication the single-window form
/// made (it clamped into the one window it had), it requires the guest to address outside every
/// buffer it declared, and no window can leak another draw's data.
pub fn mem_window_helper(windows: &[MemWindow]) -> String {
    mem_window_helper_named(windows, "gxp_mem")
}

/// [`mem_window_helper`] over a named binding, so a module can carry ONE PER STAGE. WGSL has a
/// single global namespace, and the two stages resolve different windows through different
/// bindings, so a linked pair that loads memory in both needs two helpers with two names -
/// emitting one would silently give the fragment the vertex's buffer.
pub fn mem_window_helper_named(windows: &[MemWindow], binding: &str) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(s, "fn {binding}_word(addr: u32) -> u32 {{");
    for (i, at) in mem_window_placements(windows).iter().enumerate() {
        let _ = writeln!(s, "  {{");
        let _ = writeln!(s, "    let b{i} = {binding}[{i}u].x;");
        let _ = writeln!(s, "    if (addr >= b{i}) {{");
        let _ = writeln!(s, "      let w{i} = (addr - b{i}) >> 2u;");
        let _ = writeln!(s, "      if (w{i} < {}u) {{", at.words);
        let _ = writeln!(s, "        let g{i} = {}u + w{i};", at.first_word);
        let _ = writeln!(s, "        return {binding}[g{i} >> 2u][g{i} & 3u];");
        let _ = writeln!(s, "      }}");
        let _ = writeln!(s, "    }}");
        let _ = writeln!(s, "  }}");
    }
    let _ = writeln!(s, "  return 0u;");
    let _ = writeln!(s, "}}");
    s
}

/// Bytes the driver adds to the DEFAULT uniform buffer's bound address before writing it into
/// its DATA slot - `carried * 4`, the first register it did NOT copy into the SA file.
/// `VITASLOP_GXP_DEFAULT_UNIFORM_OFFSET=0` restores the pre-2026-08-25c reading, in which the
/// pointer named the buffer's BASE.
///
/// Kept as an arm for the same reason `VITASLOP_GXP_ATTR_FILL=api` is: the offset rests on
/// corpus closure over five reads in two programs (see [`MemWindow::base_offset`]), which is
/// strong but is not a specification, and a title whose driver turns out to place the base
/// after all should cost one run to find out rather than a rebuild. It is also what PROVED
/// this change inert for the other titles - PCSE00001's first 1,600 frames are bit-identical
/// under both arms while the golf title's menu differs on 98.6% of its pixels, which is the
/// negative control an A/B needs - and a no-regression claim that needs a rebuild to check is
/// one nobody checks.
fn default_uniform_pointer_offset(carried_regs: u32) -> u32 {
    if crate::link::arm_on("VITASLOP_GXP_DEFAULT_UNIFORM_OFFSET") {
        carried_regs * 4
    } else {
        0
    }
}

/// The container index the format gives the DEFAULT uniform buffer, which is also the buffer
/// index a +0x78 entry names it by when the driver hands the program a POINTER to it as well as
/// copying part of it into the SA file.
const DEFAULT_UNIFORM_BUFFER_INDEX: u16 = 14;

/// Resolve whether (and how) a decoded VERTEX program's memory loads can be fed, per
/// [`MemWindow`]. An empty list = the program loads no memory. `Err` names exactly what is
/// unestablished - the caller must refuse to emit rather than let a load read fabricated
/// bytes ([`crate::wgsl::emit_mem_load`] cannot be reached without this having succeeded).
///
/// The checks are what make the +0x78 reading safe to act on (see
/// [`crate::container::UniformBufferBinding`]): a program where the reading is wrong cannot
/// pass them by accident, because every slot must land inside the DATA container and collide
/// with no literal and no texture-control word - and, the other way round, every DATA-container
/// register the program READS that is not a literal and not a texture-control word must be
/// covered by an entry. That second check is what makes a missing window a refusal rather than
/// a pointer register silently reading zero.
///
/// # The DEFAULT uniform buffer can be one of the windows
/// A +0x78 entry naming buffer 14 is the default uniform buffer: the driver copies part of it
/// into the SA register file (container 14) AND leaves its address in a DATA slot, so the
/// program reaches the rest by pointer. That is what a header whose `default_uniform_regs`
/// exceeds its own container's extent is describing - the copied part is the container, the
/// declared size is the whole buffer - and the window's extent is therefore the DECLARED size,
/// not the container's.
///
/// An entry whose buffer the program does not declare (and whose SA register it never reads) is
/// INERT and contributes no window: the golf title's programs carry one.
/// How a program uses the SA register a +0x78 entry would place a buffer's pointer in.
enum PointerUse {
    /// No memory load chases it. The entry is INERT whatever else the program does with the
    /// register - a plain READ of it is not a pointer read, and treating it as one refuses a
    /// program over a register that is somebody else's business.
    NotAPointer,
    /// A load chases it, but through a REGISTER-supplied byte offset whose value is not known
    /// here, so no compile-time extent bounds it.
    Unbounded,
    /// A load chases it and every offset is constant: this is the byte extent they reach.
    Bounded(u32),
}

/// How the program uses the pointer in SA register `base_sa`.
///
/// A `MemLoad`'s `srcs[0]` is the pointer and any further sources are REGISTER-supplied byte
/// offsets ([`crate::usse::decode`]), whose value is not known here - so a load carrying one is
/// [`PointerUse::Unbounded`] rather than a bound that happens to hold for the constant part.
fn pointer_use(base_sa: u32, shader: &Shader, secondary: &Shader) -> PointerUse {
    let mut extent = 0u32;
    let mut saw = false;
    for i in shader.instrs.iter().chain(secondary.instrs.iter()) {
        let crate::ir::Op::MemLoad { elements, offset_bytes } = i.op else { continue };
        let Some(ptr) = i.srcs.first() else { continue };
        if ptr.bank != crate::ir::Bank::SecondaryAttr || u32::from(ptr.index) != base_sa {
            continue;
        }
        if i.srcs.len() > 1 {
            return PointerUse::Unbounded;
        }
        saw = true;
        extent = extent.max(offset_bytes + u32::from(elements) * 4);
    }
    if saw { PointerUse::Bounded(extent) } else { PointerUse::NotAPointer }
}

pub fn resolve_mem_windows(
    program: &Program,
    shader: &Shader,
) -> Result<Vec<MemWindow>, &'static str> {
    // >>> THE LOAD CAN BE IN EITHER STREAM, and looking only at the primary refused a
    // >>> whole retail title. A program that reads a bound uniform buffer by chasing its
    // >>> pointer can issue that load from the SECONDARY (prologue) program instead - which
    // >>> is the natural place for it, since the prologue runs once and leaves the fetched
    // >>> registers in the SA file for the primary to read with no load at all. One title's
    // >>> world-transform vertex program does exactly that: `MemLoad` in the secondary,
    // >>> destination sa[20], address sa[18] from the DATA container. Scanning only the
    // >>> primary found no load, resolved no window, and the linker then rejected the
    // >>> address register as an SA read with nothing behind it - a message about uniform
    // >>> buffer extents that named neither the load nor the stream it was in.
    let secondary = crate::usse::decode_secondary_shader(program);
    let has_mem_load = |sh: &Shader| sh.instrs.iter().any(|i| matches!(i.op, crate::ir::Op::MemLoad { .. }));
    if !has_mem_load(shader) && !has_mem_load(&secondary) {
        return Ok(Vec::new());
    }
    let Some(data) = program.containers.iter().find(|c| c.index == 19) else {
        return Err("memory loads with no DATA container to hold a buffer's address");
    };
    if program.uniform_buffer_bindings.is_empty() {
        return Err("memory loads with no +0x78 buffer binding entry to place a pointer");
    }
    // A buffer the driver COPIES into the SA register file is read as `sa[k]` with no load at
    // all (see `Program::sa_uniform_buffers`), so it is not a window even when it also has a
    // +0x78 entry - the copy is what the program reads.
    let sa_resident = program.sa_uniform_buffers();
    // Both streams again: the address register is read by whichever one issues the load.
    let reads_sa = |reg: u32| {
        shader.instrs.iter().chain(secondary.instrs.iter()).flat_map(|i| i.srcs.iter()).any(|s| {
            s.bank == crate::ir::Bank::SecondaryAttr && u32::from(s.index) == reg
        })
    };

    let mut windows: Vec<MemWindow> = Vec::new();
    for binding in &program.uniform_buffer_bindings {
        if binding.data_slot >= data.size_regs {
            return Err("a buffer-address slot falls outside the DATA container");
        }
        let base_sa = u32::from(data.base_sa) + u32::from(binding.data_slot);
        if program.literals.iter().any(|&(reg, _)| reg == base_sa)
            || program.texture_control.iter().any(|&(reg, _)| reg == base_sa)
        {
            return Err("a buffer-address SA register collides with a literal or texture word");
        }
        // The buffer's EXTENT, which is what the host uploads, and the OFFSET its pointer
        // carries (see `MemWindow::base_offset`). The default uniform buffer's window is only
        // the part the driver did NOT copy into container 14; every other buffer's is its own
        // parameter's byte count, whole, at offset zero.
        let mut base_offset = 0u32;
        let bytes = if binding.buffer_index == DEFAULT_UNIFORM_BUFFER_INDEX {
            let carried = program
                .containers
                .iter()
                .find(|c| c.index == DEFAULT_UNIFORM_BUFFER_INDEX)
                .map_or(0, |c| u32::from(c.size_regs));
            // A program with no container 14 keeps its whole default buffer behind the pointer,
            // which is the shape this code always assumed and is still exactly right.
            if carried > program.default_uniform_regs {
                return Err(
                    "container 14 carries MORE registers than the header declares for the                      default uniform buffer - the leftover the pointer names cannot be sized",
                );
            }
            base_offset = default_uniform_pointer_offset(carried);
            // The extent follows the offset, or the OFF arm is not the old behaviour: a window
            // that starts at the base but is only as long as the leftover would stop short of
            // the registers the old reading addressed, and the arm would be testing a third
            // thing that has never been anyone's reading.
            program.default_uniform_regs * 4 - base_offset
        } else {
            let declared = program.parameters.iter().find(|p| {
                p.category == ParamCategory::UniformBuffer
                    && p.resource_index >= 0
                    && p.resource_index as u32 == u32::from(binding.buffer_index)
            });
            match declared {
                Some(ub) => ub.array_size,
                // An entry for a buffer the program does not declare is INERT - nothing binds
                // it and nothing can read it. Skipping it is exact as long as the pointer
                // register really is dead, which is checked here rather than assumed.
                //
                // >>> AND WHEN IT IS NOT DEAD, THE PROGRAM ITSELF SIZES THE WINDOW. The only
                // thing the parameter table was supplying is a BYTE COUNT, and a window exists
                // solely so the shader's own address arithmetic can read guest bytes - so the
                // bytes it actually loads are an exact bound, and a tighter one than the
                // declared size. `static_extent_through` returns it only when every load
                // through this pointer has a constant offset; a register-supplied (runtime)
                // offset is unbounded and still refuses by name.
                //
                // MEASURED on a fragment program whose +0x78 table names twelve buffers and
                // whose parameter table declares ONE (buffer 7): it reads buffer 11's pointer
                // at sa[14] and was refused whole for it, dropping every draw of the pair.
                None => match pointer_use(base_sa, shader, &secondary) {
                    PointerUse::Bounded(bytes) => bytes,
                    PointerUse::Unbounded => {
                        return Err(
                            "a +0x78 entry names a buffer the parameter table does not declare,                              and the program's loads through its pointer are not statically bounded",
                        );
                    }
                    // >>> A READ IS NOT A POINTER READ. This used to refuse whenever the
                    // register was read AT ALL, which is a different claim: the +0x78 table is
                    // a FULL table of where each buffer's pointer WOULD go, so an entry for a
                    // buffer the program does not declare places nothing, and the compiler is
                    // free to use that register for its own purposes. MEASURED on a fragment
                    // program whose table names twelve buffers and whose parameter table
                    // declares one: it reads sa[14] - buffer 11's would-be slot - as DATA and
                    // never loads through it, and the pair was refused whole for it.
                    PointerUse::NotAPointer => continue,
                },
            }
        };
        if bytes == 0 {
            if reads_sa(base_sa) {
                return Err("a memory-loaded uniform buffer declares a zero size");
            }
            continue;
        }
        if sa_resident.iter().any(|b| b.buffer_index == u32::from(binding.buffer_index)) {
            continue;
        }
        windows.push(MemWindow {
            buffer_index: u32::from(binding.buffer_index),
            bytes,
            base_sa,
            base_offset,
        });
    }
    if windows.is_empty() {
        return Err("memory loads with no bindable buffer among the +0x78 entries");
    }
    // The other direction: a DATA-container register the program READS that is neither a
    // literal, nor a texture-control word, nor one of the windows above is a POINTER nothing
    // feeds - which would read zero and load fabricated bytes with nothing to say so.
    // A DATA register the SECONDARY program writes is a computed value, not a pointer: the
    // prologue puts something there for the primary to read. One title's world program packs
    // a loaded matrix row into the first DATA register and reads it from the primary, which
    // is indistinguishable here from an unfed pointer unless the writes are counted - and
    // calling that "a pointer nothing feeds" refused a program whose prologue feeds it.
    let written_by_secondary = |reg: u32| {
        secondary.instrs.iter().any(|i| {
            i.dest.as_ref().is_some_and(|d| {
                if d.bank != crate::ir::Bank::SecondaryAttr {
                    return false;
                }
                // A memory load fills `elements` consecutive registers; everything else
                // writes at most the four lanes of its destination.
                let base = u32::from(d.index);
                let span = match i.op {
                    crate::ir::Op::MemLoad { elements, .. } => u32::from(elements),
                    _ => 4,
                };
                (base..base + span).contains(&reg)
            })
        })
    };
    for reg in u32::from(data.base_sa)..u32::from(data.base_sa) + u32::from(data.size_regs) {
        // >>> AND THE SAME CORRECTION AS ABOVE: only a register a load actually CHASES is a
        // pointer. Refusing on a plain read made this check fire on a DATA register the
        // program uses as ordinary data, which is a question for the SA machinery
        // (`link::secondary_attr_init`) and not evidence that a pointer is unfed.
        if !matches!(pointer_use(reg, shader, &secondary), PointerUse::Bounded(_) | PointerUse::Unbounded)
            || program.literals.iter().any(|&(r, _)| r == reg)
            || program.texture_control.iter().any(|&(r, _)| r == reg)
            || windows.iter().any(|w| w.base_sa == reg)
            || written_by_secondary(reg)
        {
            continue;
        }
        return Err("the program reads a DATA-container register no +0x78 entry feeds");
    }
    // The windows bind as ONE uniform buffer (present on every WebGPU tier); the guaranteed
    // minimum for a single binding is 64 KiB, headers included.
    if mem_window_vec4_count(&windows) * 16 > 65536 {
        return Err("the declared uniform buffers exceed a 64 KiB uniform binding");
    }
    Ok(windows)
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
    /// The guest-memory windows the program's 0xE8 loads read through, in the order the
    /// `gxp_mem` binding lays them out (see [`MemWindow`] and [`mem_window_placements`]).
    /// Empty when the program loads no memory; the renderer must bind every window's bytes
    /// with every draw.
    pub mem_windows: Vec<MemWindow>,
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
/// Emit one vertex attribute's load into the PA bank, shared by the standalone wrapper and the
/// linked module so both deliver an attribute identically.
pub(crate) fn emit_attribute_load(m: &mut String, a: &VertexAttribute) {
    const COMP: [&str; 4] = ["x", "y", "z", "w"];
    for c in 0..a.components {
        let _ = writeln!(
            m,
            "  pa[{}] = bitcast<u32>(in.a{}.{});",
            a.base_lane + c,
            a.location,
            COMP[(c & 3) as usize]
        );
    }
}

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
            // The standing fill; only a LINK can answer this - see `VertexAttribute::surplus_fill`.
            surplus_fill: [crate::attrflow::Fill::Identity; 4],
        })
        .collect();
    attributes.sort_by_key(|a| a.base_lane);
    for (i, a) in attributes.iter_mut().enumerate() {
        a.location = i as u32;
    }

    // The SA binding carries the default uniform buffer (loaded at SA register 0) PLUS every
    // non-default uniform buffer the driver copies into the SA file - see
    // `Program::sa_uniform_buffers`, and note that a program in that shape can declare a
    // default buffer of size ZERO and keep its whole transform in container 0. The SA
    // registers above all of them hold texture control words and literals, which are baked
    // into the emitted shader rather than bound.
    let sa_lane_count = program.sa_carried_extent();
    let extent = output_write_extent(shader);
    let varying_vec4s = extent.saturating_sub(4).div_ceil(4);
    let samplers = crate::wgsl::tex_units(shader, |u| program.sampler_is_cube(u as u32));

    // An Err here (memory loads whose window cannot be established) surfaces as a
    // LinkError in `link_programs`, which re-runs the resolver to NAME the reason; a plan
    // is a statement of what to bind, and there is nothing to bind for a refused program.
    let mem_windows = resolve_mem_windows(program, shader).unwrap_or_default();

    VertexBindingPlan { attributes, sa_lane_count, varying_vec4s, samplers, mem_windows }
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

    // The guest-memory windows the program's 0xE8 loads read through: one header vec4 per
    // window (lane x = its guest base address), then every window's words (see [`MemWindow`]).
    if !plan.mem_windows.is_empty() {
        let _ = writeln!(
            m,
            "@group(0) @binding(1) var<uniform> gxp_mem: array<vec4<u32>, {}>;",
            mem_window_vec4_count(&plan.mem_windows)
        );
        m.push_str(&mem_window_helper(&plan.mem_windows));
    }

    // Sampled textures + samplers at group 1, under the VERTEX stage's own names
    // (`vt{u}`/`vs{u}`, see `crate::wgsl::sampler_names`). A vertex program that fetches a
    // texture builds GEOMETRY from what it reads, so this is not an optional decoration: a
    // wrapper that omits the declaration emits a module referring to an undefined identifier,
    // which cannot be validated at all - and the shaders that need it are exactly the ones
    // worth validating, the displacement/canvas programs.
    for (i, b) in plan.samplers.iter().enumerate() {
        let (tb, sb) = (i as u32 * 2, i as u32 * 2 + 1);
        let ty = b.wgsl_type();
        let (tex, samp) =
            crate::wgsl::sampler_names(crate::container::ProgramKind::Vertex, b.unit);
        let _ = writeln!(m, "@group(1) @binding({tb}) var {tex}: {ty};");
        let _ = writeln!(m, "@group(1) @binding({sb}) var {samp}: sampler;");
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
    for a in &plan.attributes {
        emit_attribute_load(&mut m, a);
    }
    // Load SA lanes from the uniform buffer.
    if sa_vec4 > 0 {
        let _ = writeln!(
            m,
            "  for (var k: u32 = 0u; k < {}u; k = k + 1u) {{ sa[k] = sa_buf.data[k / 4u][k % 4u]; }}",
            plan.sa_lane_count
        );
    }
    // The driver-placed pointer register: the bound buffer's guest address, exactly as the
    // hardware's PDS would leave it, so the body's address arithmetic runs bit-exact.
    for (i, w) in plan.mem_windows.iter().enumerate() {
        let _ = writeln!(m, "  sa[{}] = gxp_mem[{i}u].x;", w.base_sa);
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
mod probe_tests {
    use super::*;

    /// Parsing is pinned because every form of this knob is typed by hand at a shell prompt in
    /// the middle of a bisection, and a spec that silently fails to parse renders the ORDINARY
    /// picture - which reads as "the probe says this term is fine" rather than as a typo.
    fn parse(spec: &str) -> Option<ProbeSpec> {
        // Straight into the parser, NEVER through the process environment. The earlier form
        // set `VITASLOP_GXP_PROBE`, called `probe_spec()` and removed it, under a mutex - and
        // the mutex could not cover the rest of the crate: `build_module` reads the same
        // variable, so any test building a module in parallel with this one intermittently
        // saw a probe active and emitted probe WGSL. See `parse_probe_spec`.
        parse_probe_spec(spec)
    }

    #[test]
    fn probe_spec_parses_every_documented_form() {
        assert_eq!(
            parse("pa20"),
            Some(ProbeSpec {
                bank: "pa".into(),
                index: 20,
                at: None,
                f32_lanes: false,
                bits: None
            })
        );
        assert_eq!(parse("pa2@54").unwrap().at, Some(54));
        assert_eq!(parse("pa8@37:f32").unwrap(), ProbeSpec {
            bank: "pa".into(),
            index: 8,
            at: Some(37),
            f32_lanes: true,
            bits: None
        });
        assert_eq!(parse("sa14@0:bits=7fc0dead").unwrap().bits, Some(0x7fc0_dead));
        assert_eq!(parse("sa14:bits=0x7fc0dead").unwrap().bits, Some(0x7fc0_dead));
        // A bank name is not restricted to `pa`: the internal, secondary-attribute and output
        // files are all worth reading, and they are all plain arrays in the emitted WGSL.
        assert_eq!(parse("i0@41:f32").unwrap().bank, "i");
        assert_eq!(parse("r6").unwrap().bank, "r");
    }

    #[test]
    fn probe_spec_refuses_what_it_cannot_read() {
        assert_eq!(parse(""), None);
        assert_eq!(parse("pa"), None, "no register index");
        assert_eq!(parse("12"), None, "no bank name");
        assert_eq!(parse("pa2@nope"), None, "instruction index is not a number");
        assert_eq!(parse("pa2:bits=zz"), None, "not a hex word");
    }

    /// The snapshot form must read the SNAPSHOT locals, not the bank array - reading the array
    /// at the end of the shader is exactly the value an `@<instr>` probe exists to avoid.
    #[test]
    fn probe_read_expr_uses_the_snapshot_only_for_an_at_probe() {
        let end = ProbeSpec { bank: "pa".into(), index: 20, at: None, f32_lanes: false, bits: None };
        assert!(probe_read_expr(&end, false).contains("pa[20]"));
        let at = ProbeSpec { bank: "pa".into(), index: 2, at: Some(54), f32_lanes: false, bits: None };
        let e = probe_read_expr(&at, true);
        assert!(e.contains("_probe0") && !e.contains("pa["), "{e}");
    }

    /// A bit probe is a boolean picture, so it must never bitcast: the whole point is that the
    /// poison word is a quiet NaN, which every numeric read collapses to something that looks
    /// like the zero it has to be told apart from.
    #[test]
    fn a_bit_probe_compares_raw_words_rather_than_bitcasting() {
        let spec = ProbeSpec {
            bank: "sa".into(),
            index: 14,
            at: Some(0),
            f32_lanes: false,
            bits: Some(0x7fc0_dead),
        };
        let e = probe_read_expr(&spec, true);
        assert!(e.contains("_probe0 == 2143346349u"), "{e}");
        assert!(!e.contains("bitcast") && !e.contains("unpack2x16float"), "{e}");
    }
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

    /// A fragment program that SOURCES the output bank is reading the ROP's destination colour
    /// and blending itself, so the module has to declare a destination texture and seed `o[]`
    /// from it - in the COLOUR'S OWN layout, or an F16 destination comes back as denormals.
    /// See [`BindingPlan::reads_dest_color`].
    #[test]
    fn a_program_that_reads_its_output_bank_is_given_the_destination_colour() {
        // `o0 = pa4 * sa8 + o0` - an additive blend written in ALU, the commonest shape.
        let mut mad = instr(
            Op::Mad,
            Some(Operand::plain(Bank::PrimaryAttr, 0, 2)),
            vec![
                Operand::plain(Bank::PrimaryAttr, 4, 2),
                Operand::plain(Bank::SecondaryAttr, 8, 3),
                Operand::plain(Bank::Output, 0, 1),
            ],
            [true; 4],
        );
        mad.half_precision = true;
        let mov = instr(
            Op::Mov,
            Some(Operand::plain(Bank::Output, 0, 1)),
            vec![Operand::plain(Bank::PrimaryAttr, 0, 2)],
            [true, true, false, false],
        );
        let sh = shader(vec![mad, mov]);
        let plan = plan_bindings(&sh, 12, |_| false);
        assert!(plan.reads_dest_color, "an Output-bank SOURCE is a destination read");
        assert_eq!(plan.color_precision, ColorPrecision::F16);
        let m = build_module("", &plan, false);
        assert!(m.wgsl.contains("@group(3) @binding(1) var gxp_dst: texture_2d<f32>;"), "{}", m.wgsl);
        assert!(m.wgsl.contains("textureLoad(gxp_dst, vec2<i32>(in.frag_coord.xy), 0)"), "{}", m.wgsl);
        // F16: two halves per register, the inverse of what `color_return_expr` reads back.
        assert!(m.wgsl.contains("o[0] = pack2x16float(gxp_dstc.xy);"), "{}", m.wgsl);
        assert!(m.wgsl.contains("o[1] = pack2x16float(gxp_dstc.zw);"), "{}", m.wgsl);
    }

    /// The register the colour epilogue copies, in the shape every shipped program uses:
    /// `Mov o0 = C` with the two F16-packed halves.
    fn color_epilogue(reg: u8) -> Instr {
        instr(
            Op::Mov,
            Some(Operand::plain(Bank::Output, 0, 1)),
            vec![Operand { swizzle: [0, 1, 0, 1], ..Operand::plain(Bank::PrimaryAttr, reg, 2) }],
            [true, true, false, false],
        )
    }

    fn half(mut i: Instr) -> Instr {
        i.half_precision = true;
        i
    }

    /// A source-over lerp written longhand IS a blend, and recovering it as pipeline state is
    /// what turns 42 render-pass splits a frame into one. The rewritten program must not read
    /// the output bank at all afterwards - see [`lower_dest_blend`].
    #[test]
    fn an_alu_lerp_over_the_destination_lowers_to_a_source_over_blend() {
        // t0 = -o0 + pa0 ; pa0 = t0 * pa0.wwww + o0 ; o0 = pa0
        let sub = half(instr(
            Op::Add,
            Some(Operand::plain(Bank::Temp, 0, 0)),
            vec![
                Operand { neg: true, ..Operand::plain(Bank::Output, 0, 1) },
                Operand::plain(Bank::PrimaryAttr, 0, 2),
            ],
            [true; 4],
        ));
        let lerp = half(instr(
            Op::Mad,
            Some(Operand::plain(Bank::PrimaryAttr, 0, 2)),
            vec![
                Operand::plain(Bank::Temp, 0, 0),
                Operand { swizzle: [3, 3, 3, 3], ..Operand::plain(Bank::PrimaryAttr, 0, 2) },
                Operand::plain(Bank::Output, 0, 1),
            ],
            [true; 4],
        ));
        let mut sh = shader(vec![sub, lerp, color_epilogue(0)]);
        let b = lower_dest_blend(&mut sh).expect("the lerp shape lowers");
        assert_eq!(b.color, BlendTerm { src: BlendFactor::SrcAlpha, dst: BlendFactor::OneMinusSrcAlpha });
        assert_eq!(b.alpha, BlendTerm { src: BlendFactor::SrcAlpha, dst: BlendFactor::OneMinusSrcAlpha });
        // Both blend instructions are gone and the epilogue still copies the same register.
        assert_eq!(sh.instrs.len(), 1);
        assert!(!reads_output_bank(&sh));
        assert!(!plan_bindings(&sh, 0, |_| false).reads_dest_color);
    }

    /// >>> THE LINEARITY PROOF MUST BE ABLE TO SAY NO, or it proves nothing.
    ///
    /// [`dest_is_linear`] is what would authorise rewriting a program into a DUAL-SOURCE blend
    /// (`out = dst*F + G`). Every destination-reading program in every corpus here happens to be
    /// linear, so the corpus can only ever exercise the YES answer - and an analysis that has
    /// never returned NO in a test is indistinguishable from `fn dest_is_linear() { true }`.
    /// These build the nonlinear shapes by hand.
    #[test]
    fn the_linearity_proof_refuses_a_destination_used_nonlinearly() {
        let dst = || Operand::plain(Bank::Output, 0, 1);
        let uni = || Operand::plain(Bank::SecondaryAttr, 0, 3);
        let tmp = || Operand::plain(Bank::PrimaryAttr, 0, 2);

        // `dst * dst` is QUADRATIC. This is the case a tail-matching pattern cannot see and the
        // one that would paint a silently different picture if it were accepted.
        let sq = shader(vec![
            half(instr(Op::Mul, Some(tmp()), vec![dst(), dst()], [true; 4])),
            color_epilogue(0),
        ]);
        assert!(!dest_is_linear(&sq), "dst*dst is not linear in dst");

        // `min(dst, K)` is piecewise: linear on each side of a threshold the destination itself
        // moves across, which is not the same thing as linear.
        let clamp = shader(vec![
            half(instr(Op::Min, Some(tmp()), vec![dst(), uni()], [true; 4])),
            color_epilogue(0),
        ]);
        assert!(!dest_is_linear(&clamp), "min() over the destination is piecewise");

        // A reciprocal of the destination is not linear by any reading.
        let rcp = shader(vec![
            half(instr(Op::Rcp, Some(tmp()), vec![dst()], [true; 4])),
            color_epilogue(0),
        ]);
        assert!(!dest_is_linear(&rcp), "1/dst is not linear");

        // >>> AND IT MUST FOLLOW THE TAINT, not just look at direct reads of `o`. Here the
        // destination reaches a square one instruction LATER, through a temp - which is exactly
        // how the real programs carry it, and what a tail pattern would miss.
        let laundered = shader(vec![
            half(instr(Op::Mov, Some(tmp()), vec![dst()], [true; 4])),
            half(instr(Op::Mul, Some(tmp()), vec![tmp(), tmp()], [true; 4])),
            color_epilogue(0),
        ]);
        assert!(!dest_is_linear(&laundered), "the taint has to survive a Mov");
        // `|dst|` through a Mov: the operand MODIFIER is the nonlinearity, not the op.
        let absd = shader(vec![
            half(instr(Op::Mov, Some(tmp()), vec![Operand { abs: true, ..dst() }], [true; 4])),
            color_epilogue(0),
        ]);
        assert!(!dest_is_linear(&absd), "abs(dst) is not linear in dst");


        // The YES answer, so the test is not passing by refusing everything: `dst * K + G`.
        let mad = shader(vec![
            half(instr(Op::Mad, Some(tmp()), vec![dst(), uni(), uni()], [true; 4])),
            color_epilogue(0),
        ]);
        assert!(dest_is_linear(&mad), "dst*K + G is the shape being looked for");

        // A program that never touches the destination makes no claim either way, and must not
        // report `true` - it has nothing to lower and would be a false positive in the census.
        let clean = shader(vec![
            half(instr(Op::Mul, Some(tmp()), vec![uni(), uni()], [true; 4])),
            color_epilogue(0),
        ]);
        assert!(!dest_is_linear(&clean), "no destination read is not a linear destination read");
    }

    /// The dual-source lowering's second question, beyond linearity: a destination CHANNEL
    /// feeding another output channel (`out.r = dst.a * k + ...`) is a term the ROP's
    /// per-channel blend cannot carry, so it is either GATED on a per-draw coefficient (a
    /// uniform register, a window load) that is zero for the draws that take the lowering, or
    /// the program is refused. MEASURED: a baseball title's generic blend has exactly this
    /// term on its destination alpha and drew its HUD wrong when it was folded.
    #[test]
    fn the_cross_channel_gates_name_the_coefficient_or_refuse() {
        set_dual_source_blend(true);
        let dst_w = || Operand { swizzle: [3, 3, 3, 3], ..Operand::plain(Bank::Output, 0, 1) };
        let dst = || Operand::plain(Bank::Output, 0, 1);
        let k = || Operand::plain(Bank::SecondaryAttr, 0, 3);
        let tmp = || Operand::plain(Bank::PrimaryAttr, 0, 2);
        // out = dst.a * k + dst, F16: channels 0..2 carry dst.a through k's halves.
        let prog = shader(vec![
            half(instr(Op::Mad, Some(tmp()), vec![dst_w(), k(), dst()], [true; 4])),
            color_epilogue(0),
        ]);
        // k in the uniform buffer (4 registers carried): gated on its first three halves.
        let gates = dual_source_plan_or_why(&prog, 4, &[]).expect("gated on the uniform");
        assert_eq!(
            gates,
            vec![
                DualCoef::Uniform { byte: 0, len: 2 },
                DualCoef::Uniform { byte: 2, len: 2 },
                DualCoef::Uniform { byte: 4, len: 2 }
            ]
        );
        // k a nonzero LITERAL: the term is there in every draw - refused.
        assert!(dual_source_plan_or_why(&prog, 0, &[(0, 0x3c00_3c00)]).is_err());
        // k a ZERO literal: the term is dead in every draw - exact with no gate.
        assert_eq!(dual_source_plan_or_why(&prog, 0, &[(0, 0), (1, 0)]), Ok(vec![]));
        // No provenance at all: refused, naming the channels.
        let why = dual_source_plan_or_why(&prog, 0, &[]).unwrap_err();
        assert!(why.contains("channel 0 depends on destination channel 3"), "{why}");
        // The diagonal alone (`out = dst * k + tmp`) never needs a gate.
        let diag = shader(vec![
            half(instr(Op::Mul, Some(tmp()), vec![dst(), k()], [true; 4])),
            color_epilogue(0),
        ]);
        assert_eq!(dual_source_plan_or_why(&diag, 0, &[]), Ok(vec![]));
    }

    /// `dst * K` is `Zero / Src` with the shader emitting `K` - no blend constant needed, which
    /// is why this shape is exact rather than approximated.
    #[test]
    fn an_alu_modulate_of_the_destination_lowers_to_zero_over_src() {
        let mul = half(instr(
            Op::Mul,
            Some(Operand::plain(Bank::PrimaryAttr, 0, 2)),
            vec![Operand::plain(Bank::Output, 0, 1), Operand::plain(Bank::SecondaryAttr, 0, 3)],
            [true; 4],
        ));
        let mut sh = shader(vec![mul, color_epilogue(0)]);
        let b = lower_dest_blend(&mut sh).expect("the modulate shape lowers");
        assert_eq!(b.color, BlendTerm { src: BlendFactor::Zero, dst: BlendFactor::Src });
        // The multiply became a plain copy of the uniform: that IS the source term.
        assert!(matches!(sh.instrs[0].op, Op::Mov));
        assert_eq!(sh.instrs[0].srcs[0].bank, Bank::SecondaryAttr);
        assert!(!reads_output_bank(&sh));
    }

    /// The ADDITIVE shape lowers under [`FORM_DEFAULT`] now that capsules have measured it
    /// (see the constant's own doc for the numbers), and stays off when the knob excludes it -
    /// which is what keeps `VITASLOP_GXP_DEST_BLEND=lerp,modulate` a usable bisect arm.
    #[test]
    fn the_additive_shape_lowers_by_default_and_is_off_when_excluded() {
        let add = half(instr(
            Op::Mad,
            Some(Operand::plain(Bank::PrimaryAttr, 0, 2)),
            vec![
                Operand::plain(Bank::PrimaryAttr, 4, 2),
                Operand::plain(Bank::SecondaryAttr, 2, 3),
                Operand::plain(Bank::Output, 0, 1),
            ],
            [true; 4],
        ));
        let pack = half(instr(
            Op::Pack { src_half: true },
            Some(Operand::plain(Bank::PrimaryAttr, 0, 2)),
            vec![Operand { swizzle: [0, 0, 0, 3], ..Operand::plain(Bank::Output, 0, 1) }],
            [false, false, false, true],
        ));
        let build = || shader(vec![add.clone(), pack.clone(), color_epilogue(0)]);

        // ON by default, and the destination copy it was paying for goes with it.
        let mut sh = build();
        let b = lower_dest_blend(&mut sh).expect("in the default set");
        assert_eq!(b.color, BlendTerm { src: BlendFactor::One, dst: BlendFactor::One });
        assert_eq!(b.alpha, BlendTerm { src: BlendFactor::Zero, dst: BlendFactor::One });
        assert!(!reads_output_bank(&sh), "so it no longer needs the destination copy");

        // The mask is process-wide, so this test restores it below. No other test in this
        // module builds the additive shape, so the window cannot change another one's answer.
        // Excluding the shape by name is the bisect arm, and it must still put the ALU form
        // back - that is what makes `VITASLOP_GXP_DEST_BLEND=lerp,modulate` worth having.
        set_dest_blend_lowering(FORM_LERP | FORM_MODULATE);
        let mut sh = build();
        assert_eq!(lower_dest_blend(&mut sh), None, "excluded by the knob");
        assert!(reads_output_bank(&sh), "so it pays for the destination copy again");
        set_dest_blend_lowering(FORM_DEFAULT);
    }

    /// What is NOT a blend must be refused, or the picture is a guess. A colour grade that takes
    /// a DOT PRODUCT of the destination is the case that made this whole mechanism necessary:
    /// no hardware blend can express it, so it keeps its ALU form and its render-pass split.
    #[test]
    fn an_equation_no_blend_can_express_is_refused() {
        let dot = half(instr(
            Op::Dot { components: 4 },
            Some(Operand::plain(Bank::PrimaryAttr, 0, 2)),
            vec![Operand::plain(Bank::SecondaryAttr, 8, 3), Operand::plain(Bank::Output, 0, 1)],
            [true; 4],
        ));
        let mut sh = shader(vec![dot, color_epilogue(0)]);
        assert_eq!(lower_dest_blend(&mut sh), None);
        assert!(reads_output_bank(&sh), "so the renderer still owes it the destination colour");
    }

    /// ...but an 8-bit SOP2 whose second operand is the output register is the ROP blend by
    /// construction, and [`crate::rop_blend`] already answers that one as PIPELINE state.
    /// Counting it here would ask for an attachment copy for nothing and, on a word `rop_blend`
    /// recognised, would apply the destination twice.
    #[test]
    fn a_sop2_reading_the_output_bank_is_not_a_destination_read() {
        let sop = instr(
            Op::Sop2 {
                color: crate::ir::SopOp::Add,
                alpha: crate::ir::SopOp::Add,
                f1: crate::ir::SopFactor::Src1Color,
                f1_complement: false,
                f2: crate::ir::SopFactor::Zero,
                f2_complement: false,
            },
            Some(Operand::plain(Bank::Output, 0, 1)),
            vec![Operand::plain(Bank::PrimaryAttr, 0, 2), Operand::plain(Bank::Output, 0, 1)],
            [false, false, false, true],
        );
        let plan = plan_bindings(&shader(vec![sop]), 0, |_| false);
        assert!(!plan.reads_dest_color);
        assert!(!build_module("", &plan, false).wgsl.contains("gxp_dst"));
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
        assert_eq!(plan_bindings(&sh, 0, |_| false).color, ColorOutput::NonNativePa(0));
    }

    #[test]
    fn a_non_native_colour_above_the_interpolants_is_read_where_the_program_wrote_it() {
        // One title's bright-pass (`frag_8669f600`): its one varying descriptor is
        // PREFETCH-ONLY and takes `pa[0..2)`, so the program's own writes - and its colour -
        // are at `pa[2]`. The old rule looked for `pa0`, found nothing, fell through to the
        // OUTPUT bank, and returned four registers the program never wrote: the pass emitted
        // (0,0,0,0) into the surface the glare chain blurs, and the whole bloom composite added
        // nothing. A colour register that is never written is not an approximation, it is a
        // black surface with no error anywhere - which is the failure this crate exists to
        // refuse.
        let mut mad = instr(
            Op::Mul,
            Some(Operand::plain(Bank::PrimaryAttr, 2, 1)),
            vec![Operand::plain(Bank::PrimaryAttr, 0, 2), Operand::plain(Bank::SecondaryAttr, 0, 3)],
            [true; 4],
        );
        mad.half_precision = true;
        let plan = plan_bindings(&shader(vec![mad]), 4, |_| false);
        assert_eq!(plan.color, ColorOutput::NonNativePa(2));
        assert_eq!(plan.color_precision, ColorPrecision::F16);
        let wgsl = build_module("", &plan, false).wgsl;
        assert!(
            wgsl.contains("return vec4<f32>(unpack2x16float(pa[2]), unpack2x16float(pa[3]));"),
            "{wgsl}"
        );
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
        assert_eq!(plan.color, ColorOutput::NonNativePa(0));
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
            semantic: 0,
            semantic_index: 0,
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

