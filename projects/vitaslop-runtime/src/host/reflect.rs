use super::*;

/// Walk one `SceGxmProgram`'s parameter table and reflect every constant the capture
/// path needs from it. This is the uncached body behind
/// [`VitaState::reflect_program`] - see [`ProgramReflection`] for the caching rationale
/// and why it must stay a pure function of the program blob.
///
/// Table layout (from the guest's own reflection API, the same offsets
/// `sceGxmProgramGetParameter` and friends use): `header+0x24` is the parameter count,
/// `header+0x28` holds the offset to the table, and each 16-byte entry is
/// `{ +0x0 name offset (signed, relative to the entry), +0x4 packed word, +0x8 array
/// size, +0xc resource index }`. The packed word's low nibble is the category (0
/// attribute, 1 uniform, 2 sampler), the next the component type (1 = F16), the next
/// the component count.
pub(super) fn reflect_program_uncached(ctx: &GuestCtx, header: u32) -> ProgramReflection {
    let mut r = ProgramReflection::default();
    r.default_uniform_bytes = default_uniform_buffer_bytes(ctx, header);
    let count = ctx.read_u32(header.wrapping_add(0x24));
    let base = header
        .wrapping_add(0x28)
        .wrapping_add(ctx.read_u32(header.wrapping_add(0x28)));
    let mut max_regs = 0u32;
    let mut best_albedo: Option<(i32, u32)> = None;
    // One scratch buffer for every name in the table: names are short and this walk is
    // the whole reason the per-draw version allocated.
    let mut name = String::with_capacity(64);
    for i in 0..count.min(256) {
        let p = base.wrapping_add(i.wrapping_mul(16));
        let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
        let category = word & 0xf;
        let ptype = (word >> 4) & 0xf;
        let comp = (word >> 8) & 0xf;
        let res = ctx.read_u32(p.wrapping_add(0xc));
        let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
        read_lowercase_name(ctx, name_addr, &mut name);
        match category {
            // A per-vertex attribute. Only its NAME matters: an expansion basis
            // (`scale_rotation`, or an explicit right/up axis pair) says the vertex
            // program synthesizes its own primitive rather than reading triangles.
            0 => {
                if name.contains("scale_rotation")
                    || name.contains("rightvector")
                    || name.contains("upvector")
                {
                    r.shader_expanded = true;
                }
            }
            1 => {
                let array = ctx.read_u32(p.wrapping_add(8)).max(1);
                let is_f16 = ptype == 1;
                // The extent this parameter occupies, in 32-BIT REGISTERS - which is what
                // `resource_index` counts, and what the buffer is sized in.
                //
                // A component is not a register. The width comes from the type nibble, the
                // same mapping as `ParamType::component_bytes`: F16/U16/S16/C10 pack two
                // components per register and U8/S8 four, so an `F16[4]` at register 2 ends
                // at register 4, not register 6. Counting every component as a register
                // OVER-SIZES the buffer, and the over-size is not benign: the default
                // uniform buffer is a recycled per-scene ring, so the bytes past what the
                // guest wrote are the PREVIOUS draw's uniforms. That is how a fragment
                // program declaring 16 bytes came to be captured with 24, whose tail
                // "ramped smoothly" frame over frame and read exactly like a runaway
                // exposure multiplier the guest was computing. It was our own arithmetic.
                let comps = comp.max(1).wrapping_mul(array);
                let regs = match ptype {
                    1 | 2 | 5 | 6 => comps.div_ceil(2),
                    7 | 8 => comps.div_ceil(4),
                    // F32/U32/S32, and anything whose width this table does not know: one
                    // register per component, which is the widest reading and so the one
                    // that cannot truncate a buffer.
                    _ => comps,
                };
                max_regs = max_regs.max(res.wrapping_add(regs));
                let pref = ParamRef {
                    res,
                    comp: comp.max(1) as u8,
                    f16: is_f16,
                };
                // A 4x4 matrix is declared as component_count 4, array_size 4.
                let is_matrix = comp == 4 && array == 4;
                // A single COMBINED model->clip matrix, tested first: its name contains
                // the projection substring too, and reading it as the projection half of
                // a pair would compose it with an identity world and place the mesh by a
                // matrix that is already complete.
                if is_matrix
                    && (name.contains("worldviewproj")
                        || name.contains("modelviewproj")
                        || name.contains("wvp"))
                {
                    r.mvp_off = Some(res);
                } else if is_matrix
                    && (name.contains("toprojection")
                        || name.contains("worldtoclip")
                        // `viewProjection` and the shorter `viewProj` this title's race
                        // shaders use. Its model half is named plainly `world` (below);
                        // between them they are the whole transform, and without both the
                        // renderer falls back to "the first sixteen floats" - which for
                        // these programs is the MODEL matrix alone, so the world is drawn
                        // with no camera at all and no observer can recover the eye.
                        || name.contains("viewproj"))
                {
                    r.proj_off = Some(res);
                } else if is_matrix && (name.contains("modeltoworld") || name == "world") {
                    r.world_off = Some(res);
                } else if name.contains("coarseexposure") {
                    r.exposure_off = Some(res);
                } else if (name.contains("albedocolour")
                    || name.contains("albedocolor")
                    || name.contains("primarytint"))
                    && comp >= 3
                {
                    // The primary layer's tint only: a "secondary" tint belongs to a
                    // second material layer the capture renderer does not composite.
                    r.tint = Some(pref);
                } else if name.contains("light0direction") && comp >= 3 {
                    r.light_dir = Some(pref);
                } else if (name.contains("light0colour") || name.contains("light0color"))
                    && comp >= 3
                {
                    r.light_col = Some(pref);
                }
            }
            2 => {
                // Sampler. Its resource index IS its texture unit, so the number of units
                // the program occupies is one past the highest - which is the length of
                // the array `sceGxmPrecomputed*StateSetAllTextures` is handed.
                r.texture_unit_count = r.texture_unit_count.max(res.wrapping_add(1));
                // ...and WHICH units, not just how many. A program samples the units it
                // DECLARES and no others, so every other slot of the guest's 16-unit binding
                // block is state this draw cannot read. See `ProgramReflection::sampler_units`.
                if res < 32 {
                    r.sampler_units |= 1u32 << res;
                }
                let score = albedo_name_score(&name);
                if score > 0 && best_albedo.map(|(s, _)| score > s).unwrap_or(true) {
                    best_albedo = Some((score, res));
                }
            }
            _ => {}
        }
    }
    r.albedo_unit = best_albedo.map(|(_, u)| u);
    r.uniform_size_bytes = max_regs.wrapping_mul(4);
    r
}

/// Diagnostic (`VITASLOP_DUMP_VPROG`): dump each vertex program's parameter table.
///
/// Read once and cached. These two are tested ONCE PER DRAW, and on Windows
/// `std::env::var` copies and re-encodes the whole environment block before it can
/// answer - so an unset diagnostic knob was costing more per frame than several draws.
pub(super) fn dump_vprog() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VITASLOP_DUMP_VPROG").is_some())
}

/// Diagnostic (`VITASLOP_DUMP_FPROG`): dump each fragment program's samplers. See
/// [`dump_vprog`] for why this is cached.
pub(super) fn dump_fprog() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VITASLOP_DUMP_FPROG").is_some())
}

/// Longest parameter name read while reflecting. The previous per-call code read 48 or
/// 64 bytes depending on the call site; 64 is the larger of the two, so nothing that
/// used to match stops matching.
pub(super) const MAX_PARAM_NAME: usize = 64;

/// Read the NUL-terminated parameter name at `addr` into `out`, lowercased. Reuses the
/// caller's buffer so a whole parameter table costs one allocation, not one per name.
pub(super) fn read_lowercase_name(ctx: &GuestCtx, addr: u32, out: &mut String) {
    let mut raw = [0u8; MAX_PARAM_NAME];
    ctx.read_into(addr, &mut raw);
    out.clear();
    out.extend(
        raw.iter()
            .take_while(|&&b| b != 0)
            .map(|&b| (b as char).to_ascii_lowercase()),
    );
}

/// Score a fragment-sampler name (already lowercased) for how strongly it reads as the
/// albedo / base-colour map. Positive = a diffuse colour source (higher is a better
/// match); 0 = not an albedo, or a name that clearly names a different map role
/// (normal/spec/env/etc). Used to pick the one texture the capture renderer samples.
pub(super) fn albedo_name_score(name: &str) -> i32 {
    // A name that clearly identifies a non-colour map role is never the albedo, even if
    // it also contains a generic "map"/"tex" token.
    const NON_ALBEDO: [&str; 14] = [
        "normal", "spec", "gloss", "rough", "env", "cube", "reflect", "light", "shadow", "detail",
        "bump", "height", "mask", "emiss",
    ];
    if NON_ALBEDO.iter().any(|k| name.contains(k)) {
        return 0;
    }
    // Explicit albedo names rank highest; generic colour names next; a bare "diffuse"
    // fragment (some engines name the sole colour sampler just "tex"/"map") lowest.
    //
    // NOTE (empirical, this title): an `AlbedoTexture`/`LiveryAlbedo` sampler is this
    // engine's base colour for a body panel (the blue livery) BUT is also the livery/decal
    // SOURCE sheet on other parts (a gauge/number sheet that, sampled whole at a tiling UV,
    // paints white rings on the wheels). Both use the identical sampler name and role - the
    // only difference is the mesh's UV mapping, which the (untranslated) fragment program
    // resolves. So this name-based pick CANNOT separate them; ranking diffuse over albedo
    // to kill the wheel rings also flattens the body's livery (a worse regression). The
    // rings are left as a known artifact until the GXP fragment/vertex texcoord binding is
    // reflected (or the fragment program is translated) - do not "fix" it by name score.
    for (kw, score) in [
        ("albedo", 100),
        ("basecolor", 95),
        ("basecolour", 95),
        ("diffuse", 90),
        ("diff", 70),
        ("colour", 60),
        ("color", 60),
        ("base", 40),
    ] {
        if name.contains(kw) {
            return score;
        }
    }
    0
}

/// Size in bytes of a `SceGxmProgram`'s default uniform buffer, from the container's
/// `default_uniform_buffer_count` field (header +0x64), which counts 32-bit SA registers.
///
/// The field at +0x2C that this previously read is the varyings-block offset, not a size: it
/// reads as a fixed 108 on every program in a title, so it neither bounded the buffer nor
/// tracked its real extent. Clamped so a header we failed to resolve cannot request an absurd
/// allocation.
///
/// This is also what `sceGxmProgramGetDefaultUniformBufferSize` must hand the GUEST: a title
/// uses that size as the length of the block it writes into the buffer it reserved, so a
/// wrong answer truncates its own uniform upload (see [`crate::vita::gxm`]).
pub(crate) fn default_uniform_buffer_bytes(ctx: &GuestCtx, header: u32) -> u32 {
    use crate::vita::gxm::{
        DEFAULT_UNIFORM_BUFFER_MAX_WORDS, GXP_DEFAULT_UNIFORM_BUFFER_COUNT_OFF,
    };
    if header == 0 {
        return 0;
    }
    ctx.read_u32(header.wrapping_add(GXP_DEFAULT_UNIFORM_BUFFER_COUNT_OFF))
        .min(DEFAULT_UNIFORM_BUFFER_MAX_WORDS)
        .wrapping_mul(4)
}

/// Diagnostic (`VITASLOP_GXM_UNIFORM_POISON=1`): fill a freshly reserved default uniform buffer
/// with a recognisable bit pattern instead of leaving it zeroed.
///
/// A zeroed buffer cannot distinguish "the guest wrote 0.0 here" from "the guest never wrote
/// here at all", and those have completely different causes: the first is real data, the second
/// means the value reaches the shader by a path we do not model. The poison is a quiet NaN
/// (`0x7fc0dead`), so any lane still holding it at draw time was never written - and a shader
/// consuming it produces NaN rather than a plausible-looking black, which is itself the signal.
pub(super) fn poison_uniform_buffer(ctx: &mut GuestCtx, buf: u32, size: u32) {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    // Through the knob table. This is the one experiment that separates "the guest wrote this"
    // from "this is the previous draw's uniforms still in the ring", and the value it has to
    // decide that about (`screenTintColour`, the white-out) only ever appears in the BROWSER -
    // which has no environment, so the knob could not be set on the only engine that shows it.
    if buf == 0
        || !*ON.get_or_init(|| crate::knobs::var_os("VITASLOP_GXM_UNIFORM_POISON").is_some())
    {
        return;
    }
    for i in 0..size / 4 {
        ctx.write_u32(buf + i * 4, 0x7fc0_dead);
    }
}

/// Whether a captured draw's FIXED-FUNCTION representation will be used by anything.
///
/// It will not when the recompiler is live and fixed-function is not allowed as a fallback -
/// which is the SHIPPING configuration, and `RenderSceneBuilder::gxp_only` is the same test on
/// the other side of the capture. There the flag already skips the per-vertex walk; here it
/// skips the per-draw MATERIAL reflection - tint, directional light and ambient, the last of
/// which is the average of the bound irradiance map. A recompiled draw runs the guest's own
/// shader and derives its lighting from the SA banks, so the material is computed, stored in
/// the `Draw` and thrown away.
///
/// Two neighbours are deliberately NOT in this set. The MVP is still composed, because
/// `interpret_draw` classifies a draw's coordinate space by it and that classification decides
/// depth, culling and the opaque range on the recompiled path too. The model->world matrix is
/// still reflected, because `object_locations` - the `locate` tool and every driving controller
/// over it - reads it to say where a mesh IS, and identity there answers the origin rather than
/// failing.
pub(super) fn fixed_function_wanted() -> bool {
    use std::sync::OnceLock;
    static WANTED: OnceLock<bool> = OnceLock::new();
    *WANTED.get_or_init(|| {
        !gxp_live_capture() || crate::knobs::flag("VITASLOP_GXP_ALLOW_FIXED_FUNCTION")
    })
}

/// Whether the GXP->WGSL recompiler capture path is enabled (env `VITASLOP_GXP_LIVE`).
/// Checked once and cached, so the per-draw `record_draw` gate is a cheap load rather than
/// an environment lookup per draw.
pub(super) fn gxp_live_capture() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::knobs::flag("VITASLOP_GXP_LIVE"))
}
