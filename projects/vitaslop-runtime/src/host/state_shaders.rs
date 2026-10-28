use super::*;

impl VitaState {
    /// Queue the shader PAIR `sceGxmShaderPatcherCreateFragmentProgram` just named, so the
    /// renderer can prepare it before a draw ever asks for it.
    ///
    /// # Why the pair is knowable here at all
    /// That call takes `const SceGxmProgram *vertexProgram` alongside the fragment program,
    /// because GXM needs both to patch the varying linkage - so the pair is fully determined
    /// while the title is still on its loading screen. The device's work at this point is
    /// patching pre-compiled USSE machine code; ours is producing WGSL and having a driver
    /// compile it, which is far more expensive and was happening at the first DRAW instead.
    ///
    /// Deduplicated by `(vertex header, fragment header)`: a title creates fragment programs in
    /// bursts and re-creates them after a patcher reset, and the renderer's own module cache
    /// would absorb that anyway - but the guest-memory reads and the queue would not.
    pub fn queue_shader_precompile(
        &mut self,
        ctx: &GuestCtx,
        vertex_header: u32,
        fragment_header: u32,
    ) {
        Self::report_null_vertex_program(vertex_header, fragment_header);
        if fragment_header == 0 {
            return;
        }
        if vertex_header == 0 {
            // The title named no pair. Under `VITASLOP_GXP_PRECOMPILE_CROSS` this fragment
            // program is offered against every vertex program created SO FAR - see
            // [`Self::cross_precompile`] for what that costs and why it is not the default.
            self.cross_precompile_fragment(ctx, fragment_header);
            return;
        }
        self.push_precompile_pair(ctx, vertex_header, fragment_header);
    }

    /// `VITASLOP_GXP_PRECOMPILE_CROSS`: for a title whose `sceGxmShaderPatcherCreateFragmentProgram`
    /// passes `vertexProgram = NULL`, offer the CROSS PRODUCT of the fragment programs it creates
    /// with the vertex programs it creates, and let the renderer keep the ones that LINK.
    ///
    /// # Why this is a knob and not the default
    /// Precompiling a pair the patcher NAMED is free of guesswork: the title said those two go
    /// together. A cross product does not know that. It is speculative work paid on a loading
    /// screen, and how much of it is wasted is a per-title measurement nobody has taken - which
    /// is exactly what this knob exists to take. **MEASURED on a retail title: all 17 of its
    /// distinct fragment programs are created with a NULL vertexProgram, and its race builds
    /// 160 pipelines IN FRAME**, so it is the title the question is about.
    ///
    /// `link_programs` is our own Rust and cheap, so a candidate that does not link costs a parse
    /// and nothing more; only a pair that LINKS costs a WGSL compile. The renderer reports how
    /// many of each, which is the number that decides whether this should ever become a default.
    pub(super) fn cross_precompile() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::knobs::flag("VITASLOP_GXP_PRECOMPILE_CROSS"))
    }

    /// Record a fragment program created with a NULL `vertexProgram`, and (under the knob) pair
    /// it with every vertex program created so far.
    pub(super) fn cross_precompile_fragment(&mut self, ctx: &GuestCtx, fragment_header: u32) {
        // Recorded UNCONDITIONALLY, not just under the knob: the list is what says WHEN this
        // title's pairs first become knowable, and that question is asked by
        // `report_program_creation_frame` on every run - see there for why it decides whether
        // preparing shaders ahead of a draw is possible for this title at all.
        if !self.precompile.note_null_fragment(fragment_header) {
            return;
        }
        self.report_program_creation_frame();
        if !Self::cross_precompile() {
            return;
        }
        let vertices = self.precompile.created_vertices().to_vec();
        for v in vertices {
            self.push_precompile_pair(ctx, v, fragment_header);
        }
    }

    /// Say, on a bounded ladder, HOW MANY shader programs the title has created and BY WHAT
    /// FRAME.
    ///
    /// # Why this line decides an open question
    /// Preparing a shader ahead of the draw that needs it is only possible if the PAIR is
    /// knowable before that draw. For a title whose `sceGxmShaderPatcherCreateFragmentProgram`
    /// passes `vertexProgram = NULL` the pair is never named, and the only material left is the
    /// two LISTS - so the whole question becomes "were both programs created early enough". A
    /// cross product over programs the title has not created yet cannot contain the pair the
    /// race draws, and that is indistinguishable, from the compile counters alone, from a cross
    /// product whose pairs simply do not link. **MEASURED on a retail title: 1,088 of 4,096
    /// candidates linked and compiled 4,356 ms ahead of any draw, and in-frame WGSL barely
    /// moved (744 -> 701 ms)** - which is only consistent with the race's real pairs being
    /// ABSENT from the candidate set. This line is what tells the two apart.
    ///
    /// Printed rather than gated behind a knob, because it is a handful of lines over a whole
    /// run and it answers a question every future session on the loading-screen hitch asks.
    /// The ladder is powers of two plus every ten, so a burst is visible without a line per
    /// program.
    pub(super) fn report_program_creation_frame(&self) {
        let v = self.precompile.created_vertices().len();
        let f = self.precompile.null_fragments().len();
        let n = v + f;
        if n > 4 && !n.is_power_of_two() && !n.is_multiple_of(10) {
            return;
        }
        // Status, not stderr: this is the count that answered the cross-product question,
        // and it belongs in the panel's STATUS section, not on a clean run's console.
        tracing::info!(
            target: "vitaslop::status",
            "gxm precompile: {v} vertex + {f} NULL-paired fragment programs created by frame {}",
            self.cur_frame
        );
    }

    /// Record a vertex program the patcher created, and (under the knob) pair it with every
    /// fragment program already created with a NULL `vertexProgram`.
    ///
    /// Called unconditionally from `sceGxmShaderPatcherCreateVertexProgram`, because the LIST is
    /// what the cross product needs and a title creates its programs in whatever order it likes -
    /// a vertex program created after the fragment programs would otherwise never be paired.
    pub fn note_vertex_program_created(&mut self, ctx: &GuestCtx, vertex_header: u32) {
        if !self.precompile.note_vertex(vertex_header) {
            return;
        }
        self.report_program_creation_frame();
        if !Self::cross_precompile() {
            return;
        }
        let fragments = self.precompile.null_fragments().to_vec();
        for f in fragments {
            self.push_precompile_pair(ctx, vertex_header, f);
        }
    }

    /// The shared tail of both queueing paths: dedupe and bound (see
    /// [`PrecompileQueue::admit`]), then read the two blobs.
    pub(super) fn push_precompile_pair(&mut self, ctx: &GuestCtx, vertex_header: u32, fragment_header: u32) {
        if !self.precompile.admit(vertex_header, fragment_header) {
            return;
        }
        let vprog = self.program_blob(ctx, vertex_header);
        let fprog = self.program_blob(ctx, fragment_header);
        // The vertex layout(s) this vertex program was CREATED with - one entry per distinct
        // layout, since one program can be patched against several - as the renderer's link
        // will see it (`vitaslop_gxp_shader::guest_attr_spec`, gated exactly as the draw path
        // gates it). This is what lets the ahead-of-draw translation carry the draw's options.
        let mut layouts: Vec<Vec<(u32, u8, u8)>> = Vec::new();
        if vitaslop_platform::knobs::gxp_guest_attrs_in_link() {
            for info in self
                .shader_patcher
                .vertex_programs()
                .filter(|i| i.program_header == vertex_header)
            {
                let spec = vitaslop_gxp_shader::guest_attr_spec(
                    info.packed_attributes
                        .iter()
                        .map(|a| (u32::from(a.reg_index), a.format, a.component_count)),
                );
                if !layouts.contains(&spec) {
                    layouts.push(spec);
                }
            }
        }
        if layouts.is_empty() {
            layouts.push(Vec::new());
        }
        self.precompile
            .push(layouts.into_iter().map(|attrs| vitaslop_gxp_shader::PatcherPair {
                vprog: vprog.clone(),
                fprog: fprog.clone(),
                attrs,
            }));
    }

    /// The shader pairs the patcher has named, for the renderer to prepare.
    ///
    /// # Why this does not DRAIN
    /// It used to, and the pairs were lost. A title creates its programs during boot, so the
    /// scene that happened to close next carried them - and that scene is not necessarily one
    /// anything renders. A `--headless` run without a shot window renders exactly ONE scene, at
    /// the very end, so every pair was drained into a scene nobody looked at and the preparation
    /// silently never happened. The renderer skips a pair whose module it already has, at the
    /// cost of one hash lookup, so re-offering the whole list is cheap and cannot be missed.
    pub fn shader_precompile(&self) -> std::sync::Arc<Vec<vitaslop_gxp_shader::PatcherPair>> {
        self.precompile.pending()
    }

    /// The reflected constants of the program at `header`, walking its parameter table
    /// on the first ask and returning the cached answer afterwards. See
    /// [`ProgramReflection`] for why this is one walk instead of five.
    ///
    /// A zero header (no program bound, or a handle never recorded) reflects to the
    /// default: no matrices, no material, identity behaviour at every call site.
    pub(super) fn reflect_program(&mut self, ctx: &GuestCtx, header: u32) -> ProgramReflection {
        if header == 0 {
            return ProgramReflection::default();
        }
        if let Some(r) = self.program_memos.reflection(header) {
            return r;
        }
        let r = reflect_program_uncached(ctx, header);
        self.program_memos.set_reflection(header, r);
        r
    }

    /// Compose the model->projection MVP for the bound vertex program from its captured
    /// `uniforms`, using reflection to locate the model->world and world->projection
    /// matrices by their declared names. Returns `None` when the shader has no separate
    /// projection matrix (a single-transform 2D/UI shader), so the caller keeps the
    /// offset-0 matrix as-is. Both matrices are column-major 4x4 float blocks at their
    /// reflected `resource_index` (in floats); the result is `projection * world`.
    pub(super) fn composed_mvp(&self, r: &ProgramReflection, uniforms: &[f32]) -> Option<[f32; 16]> {
        Self::composed_mvp_by(r, |i| uniforms.get(i).copied())
    }

    /// >>> THE THREE REFLECTIONS, OVER ANY SOURCE OF UNIFORM LANES.
    ///
    /// The bank is bytes in guest memory. The fixed-function pipeline wants the WHOLE thing as
    /// floats and gets a `Vec`; a recompiled draw wants none of it - its shader reads the raw
    /// bytes - and yet every draw converted the entire bank to floats so that these three could
    /// index it. They read at most 33 lanes between them, all at offsets the reflection already
    /// knows, so taking a lane READER instead of a slice lets the recompiled path pull those
    /// lanes straight out of the guest's bytes and skip the conversion and the allocation.
    pub(super) fn lanes_by(lane: &impl Fn(usize) -> Option<f32>, off: usize) -> Option<[f32; 16]> {
        let mut m = [0f32; 16];
        for (k, v) in m.iter_mut().enumerate() {
            *v = lane(off + k)?;
        }
        Some(m)
    }

    pub(super) fn reflected_world_proj_by(
        r: &ProgramReflection,
        lane: impl Fn(usize) -> Option<f32>,
    ) -> Option<([f32; 16], [f32; 16])> {
        let (wo, po) = (r.world_off? as usize, r.proj_off? as usize);
        Some((Self::lanes_by(&lane, wo)?, Self::lanes_by(&lane, po)?))
    }

    pub(super) fn composed_mvp_by(
        r: &ProgramReflection,
        lane: impl Fn(usize) -> Option<f32>,
    ) -> Option<[f32; 16]> {
        // A shader that keeps ONE combined matrix needs no composition - but it does
        // need to be found, because it is not always at offset 0 and the fallback
        // there would then read whatever else the program put first.
        if let Some(off) = r.mvp_off.map(|o| o as usize) {
            return Self::lanes_by(&lane, off);
        }
        let (world, proj) = Self::reflected_world_proj_by(r, lane)?;
        Some(Self::compose_mvp(&world, &proj))
    }

    /// Column-major 4x4 multiply: `projection * world`.
    pub(super) fn compose_mvp(world: &[f32; 16], proj: &[f32; 16]) -> [f32; 16] {
        let mut out = [0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut s = 0f32;
                for k in 0..4 {
                    s += proj[k * 4 + row] * world[col * 4 + k];
                }
                out[col * 4 + row] = s;
            }
        }
        out
    }

    /// The model-to-world matrix reflected from the vertex program's `vsModelToWorldMatrix`
    /// (column-major 4x4), or identity if the shader declares no world matrix. Used to bring
    /// the object-space vertex normal into world space for lighting.
    // Reflection accessors kept as a complete set: the `_by` variants are the ones the
    // draw path calls, and these read the same field through the plain uniform slice, which is
    // what a reader checking one against the other needs.
    #[allow(dead_code)]
    pub(super) fn reflected_world_matrix(&self, r: &ProgramReflection, uniforms: &[f32]) -> [f32; 16] {
        Self::reflected_world_matrix_by(r, |i| uniforms.get(i).copied())
    }

    // Reflection accessors kept as a complete set: the `_by` variants are the ones the
    // draw path calls, and these read the same field through the plain uniform slice, which is
    // what a reader checking one against the other needs.
    #[allow(dead_code)]
    pub(super) fn reflected_world_matrix_by(
        r: &ProgramReflection,
        lane: impl Fn(usize) -> Option<f32>,
    ) -> [f32; 16] {
        const IDENTITY: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        Self::reflected_world_proj_by(r, lane)
            .map(|(w, _)| w)
            .unwrap_or(IDENTITY)
    }

    /// Pull the vertex program's model->world and world->projection 4x4 matrices out of
    /// this draw's `uniforms`, at the offsets [`ProgramReflection`] located by name.
    /// Returns `(world, proj)` or `None` when the shader has no separate projection
    /// matrix (a single-transform 2D/UI shader) or the buffer is short of either.
    // Reflection accessors kept as a complete set: the `_by` variants are the ones the
    // draw path calls, and these read the same field through the plain uniform slice, which is
    // what a reader checking one against the other needs.
    #[allow(dead_code)]
    pub(super) fn reflected_world_proj(
        r: &ProgramReflection,
        uniforms: &[f32],
    ) -> Option<([f32; 16], [f32; 16])> {
        Self::reflected_world_proj_by(r, |i| uniforms.get(i).copied())
    }

    /// Does the bound vertex program SYNTHESIZE its primitive instead of reading it?
    ///
    /// A point-sprite/billboard shader is handed one record per sprite - a centre position
    /// plus the basis it expands into corners with - and builds the quad itself. The
    /// declared ATTRIBUTE names are what say so, and they are the guest's own reflection
    /// data, not a guess: an expansion basis is a `scale_rotation`, or an explicit
    /// right/up axis pair. Ordinary geometry declares none of these (across this title's
    /// whole shader corpus the per-vertex names are position/normal/tangent/uv/colour).
    ///
    /// Consumed by [`crate::capture::Draw::shader_expanded`]. This costs one parameter-table
    /// scan per draw, the same scan the world/projection and exposure reflection already do.
    /// Reflected once per program (category-0 attribute names only, so a UNIFORM named
    /// "rotation" cannot trip it) - see [`ProgramReflection::shader_expanded`].
    pub(super) fn reflected_shader_expanded(r: &ProgramReflection) -> bool {
        r.shader_expanded
    }

    /// Recover the scene exposure from the bound vertex program's reflected
    /// `vsCoarseExposureReg` uniform (a float4 whose first component is the linear
    /// exposure scale the shaders multiply lit albedo by before tone-mapping). Returns
    /// `1.0` when the shader declares no exposure uniform (2D/UI shaders), or when the
    /// value is not a sane positive number, so it is a safe no-op there.
    // Reflection accessors kept as a complete set: the `_by` variants are the ones the
    // draw path calls, and these read the same field through the plain uniform slice, which is
    // what a reader checking one against the other needs.
    #[allow(dead_code)]
    pub(super) fn reflected_exposure(r: &ProgramReflection, uniforms: &[f32]) -> f32 {
        Self::reflected_exposure_by(r, |i| uniforms.get(i).copied())
    }

    pub(super) fn reflected_exposure_by(r: &ProgramReflection, lane: impl Fn(usize) -> Option<f32>) -> f32 {
        let Some(off) = r.exposure_off else {
            return 1.0;
        };
        match lane(off as usize) {
            Some(e) if e.is_finite() && e > 0.0 => e,
            _ => 1.0,
        }
    }

    /// Diagnostic (VITASLOP_DUMP_VPROG): reflect the bound vertex program's parameter
    /// table (name / category / type / component_count / container / array_size /
    /// resource_index) once per unique program, so the uniform-buffer layout (which
    /// slots are the world matrix vs the shared view-projection) is known by name.
    /// Print every UNIFORM (category 1) the vertex program `ph` declares, with its
    /// name, its float offset, and the values this draw supplied - a 4x4 laid out as
    /// four columns.
    ///
    /// A capture renderer has no shader, so it can only place geometry with a matrix it
    /// RECOGNISES; when a title's shader names its transform something the reflection
    /// does not know, the renderer falls back to "the first sixteen floats" and draws
    /// the mesh wherever those happen to point. That failure is silent and looks like a
    /// pass that drew nothing. This is how you find the real transform: read the names.
    pub(super) fn dump_named_uniforms(&self, ctx: &GuestCtx, ph: u32, values: &[f32]) {
        if ph == 0 {
            return;
        }
        let count = ctx.read_u32(ph.wrapping_add(0x24));
        let base = ph
            .wrapping_add(0x28)
            .wrapping_add(ctx.read_u32(ph.wrapping_add(0x28)));
        for i in 0..count.min(64) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            if word & 0xf != 1 {
                continue; // category 1 = uniform
            }
            let comp = ((word >> 8) & 0xf).max(1);
            // >>> WHICH BUFFER THE PARAMETER LIVES IN, which this print used to leave out and
            // >>> which made every line about a non-default uniform a FICTION.
            //
            // Containers 0..13 are the ordinary uniform buffers (`sceGxmSetVertexUniformBuffer`)
            // and 14 is the DEFAULT one; `resource_index` is an offset inside the parameter's
            // OWN container. Reading every parameter out of the default buffer therefore prints
            // one buffer's bytes under another buffer's names. MEASURED on a baseball title's field
            // decal (`vert_842dce60`): `MaterialColor_V0` (container 14, res 0) and
            // `UVP_ViewProjectionMatrix` (res 0 of its own container) both printed from float 0
            // of the default buffer, so the view-projection read as `[1,1,1,1,0,...]` - not a
            // matrix, and not what the shader sees either. That cost a session's worth of
            // wrong turns before anyone checked the container field.
            let container = (word >> 12) & 0xf;
            let array = ctx.read_u32(p.wrapping_add(8)).max(1);
            let res = ctx.read_u32(p.wrapping_add(0xc)) as usize;
            let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
            let raw = ctx.read_bytes(name_addr, 48);
            let name: String = raw
                .iter()
                .take_while(|&&b| b != 0)
                .map(|&b| b as char)
                .collect();
            let n = (comp * array) as usize;
            // The bytes this parameter's OWN container holds. The default container is the
            // buffer the caller already read (`values`); any other is the address the guest
            // bound with `sceGxmSetVertexUniformBuffer`, read here at the parameter's offset.
            // `buf=none` is a parameter whose buffer the guest never bound - which is a
            // finding, not a formatting problem, so it is said rather than printed as zeros.
            const DEFAULT_UNIFORM_CONTAINER: u32 = 14;
            let (src, vals): (String, Vec<String>) = if container == DEFAULT_UNIFORM_CONTAINER {
                (
                    "default".into(),
                    (0..n.min(16))
                        .map(|k| {
                            values
                                .get(res + k)
                                .map_or("-".into(), |v| format!("{v:.4}"))
                        })
                        .collect(),
                )
            } else {
                let addr = if self.gxm_context != 0 {
                    crate::vita::gxmctx::vertex_uniform_buffer(ctx, self.gxm_context, container)
                } else {
                    0
                };
                if addr == 0 {
                    (format!("buf{container}=NONE"), vec!["-".into(); n.min(16)])
                } else {
                    (
                        format!("buf{container}@{addr:#x}"),
                        (0..n.min(16))
                            .map(|k| {
                                format!(
                                    "{:.4}",
                                    ctx.read_f32(
                                        addr.wrapping_add(((res + k) as u32).wrapping_mul(4))
                                    )
                                )
                            })
                            .collect(),
                    )
                }
            };
            eprintln!(
                "  uniform {name:?} container={container} src={src} res={res} comp={comp} array={array} = [{}]",
                vals.join(","),
            );
        }
    }

    /// `vhandle` is the DRAW's vertex program - the snapshot's, which a precomputed draw
    /// overlays - not a fresh read of the context.
    pub(super) fn dump_vertex_program_params(&self, ctx: &GuestCtx, vhandle: u32) {
        use std::sync::Mutex;
        static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        let ph = self.vertex_program_header(vhandle);
        if ph == 0 {
            return;
        }
        {
            let mut seen = SEEN.lock().unwrap();
            if seen.contains(&ph) {
                return;
            }
            seen.push(ph);
        }
        let count = ctx.read_u32(ph.wrapping_add(0x24));
        let base = ph
            .wrapping_add(0x28)
            .wrapping_add(ctx.read_u32(ph.wrapping_add(0x28)));
        eprintln!("VPROG header={ph:#x} params={count}");
        for i in 0..count.min(64) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let name_off = ctx.read_u32(p) as i32;
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            let array_size = ctx.read_u32(p.wrapping_add(8));
            let res_index = ctx.read_u32(p.wrapping_add(0xc));
            let name_addr = (p as i64 + name_off as i64) as u32;
            let raw = ctx.read_bytes(name_addr, 48);
            let name: String = raw
                .iter()
                .take_while(|&&b| b != 0)
                .map(|&b| b as char)
                .collect();
            // The upper half of the +4 word is the `semantic` u16 that
            // `sceGxmProgramParameterGetSemantic`/`GetSemanticIndex` split. Printed RAW as
            // well as split, because a title that builds its `SceGxmVertexAttribute` array
            // by matching semantics produces ZERO attributes when the split is wrong - and
            // an empty attribute array is silent: the draw arrives with real geometry and
            // no way to fetch it.
            let semantic = ctx.read_u32(p.wrapping_add(4)) >> 16;
            eprintln!(
                "  param[{i}] name={name:?} cat={} type={} comp={} container={} array={array_size} \
                 res_index={res_index} semantic_word={semantic:#06x}",
                word & 0xf,
                (word >> 4) & 0xf,
                (word >> 8) & 0xf,
                (word >> 12) & 0xf,
            );
        }
    }

    /// Reflect the bound fragment program's SAMPLER parameters (parameter category 2)
    /// and return the texture unit (`resource_index`) of the one that is the base-colour
    /// / albedo map, so the capture renderer - which samples a single texture - picks the
    /// diffuse colour rather than whatever happens to sit at unit 0 (this title binds the
    /// NORMAL map at unit 0, so a naive unit-0 pick paints every surface flat blue).
    ///
    /// Selection is by the sampler's declared name: a diffuse/albedo/base-colour name
    /// wins; names that are clearly a different map role (normal, specular, environment/
    /// cube, reflection, gloss, light, shadow, detail, bump, height, mask, ao) are
    /// rejected. Returns `None` when there is no fragment program or no sampler reads as
    /// an albedo, leaving the caller on its default unit-0 pick.
    pub(super) fn fragment_albedo_unit(r: &ProgramReflection) -> Option<u32> {
        r.albedo_unit
    }

    /// Reflect the bound fragment program's per-material inputs (base-colour tint, the
    /// directional light, and a flat ambient term) from its parameter table read against the
    /// captured fragment default uniform buffer. The material model is uniform across this
    /// engine's shaders (verified empirically): a `*tint*`/`*albedocolour*` float3 multiplies
    /// the sampled albedo, and `directionalLight0Colour` * saturate(N.L) + ambient lights it.
    ///
    /// Uniform packing (also verified empirically against the real shaders): a parameter's
    /// `resource_index` is a 4-byte-register offset into the default uniform buffer; each of
    /// its `component_count` components is an F16 (`type == 1`, 2 bytes) or an F32 (`type ==
    /// 0`, 4 bytes). `textures` is the draw's bound textures (albedo reordered to the front),
    /// used to resolve the ambient colour from the tiny `diffuseAmbientMap` irradiance map.
    pub(super) fn reflect_fragment_material(
        &mut self,
        ctx: &GuestCtx,
        r: &ProgramReflection,
        textures: &[crate::capture::BoundTexture],
        // The fragment stage's bound default uniform buffer, as the caller resolved it -
        // 0 when nothing is bound OR when the binding is stale for the program about to
        // draw. Passed in rather than re-read so every reader of this draw agrees about
        // which of those it is; see `record_draw`.
        buf: u32,
        // The fragment default uniform buffer's bytes, when the caller has ALREADY read them
        // (the GXP capture path reads the whole bank per draw). Empty when it has not.
        //
        // # Why this is passed in rather than read here
        // The three reflected parameters below are read three scalar components at a time,
        // and each of those is a `dyn GuestMemory` access - which in the browser is a
        // boundary crossing. That is up to NINE crossings per draw to fetch nine floats out
        // of a buffer the very next phase copies in one call
        // ([[vitaslop-count-calls-not-bytes-across-the-guest-boundary]]). Taking the bytes
        // the caller already has costs nothing and removes all nine.
        //
        // It is bytes rather than floats because the components can be F16.
        sa: &[u8],
    ) -> crate::capture::FragmentMaterial {
        let mut m = crate::capture::FragmentMaterial::default();
        if buf != 0 {
            // Read the first three scalar components of a reflected parameter from the
            // fragment default uniform buffer at its register offset, honouring the
            // F16/F32 component type.
            //
            // Out of `sa` when it covers the parameter, and out of guest memory when it does
            // not - a short or absent bank is not an error here (the caller may not have read
            // one at all), and falling back is the SAME read this always did.
            let read3 = |p: ParamRef| -> [f32; 3] {
                let base = p.res.wrapping_mul(4) as usize;
                let width = if p.f16 { 2 } else { 4 };
                let local = sa.get(base..base + 3 * width);
                let byte_off = buf.wrapping_add(p.res.wrapping_mul(4));
                std::array::from_fn(|i| match (local, p.f16) {
                    (Some(s), true) => {
                        crate::render::half_to_f32(u16::from_le_bytes([s[i * 2], s[i * 2 + 1]]))
                    }
                    (Some(s), false) => {
                        f32::from_le_bytes([s[i * 4], s[i * 4 + 1], s[i * 4 + 2], s[i * 4 + 3]])
                    }
                    (None, true) => crate::render::half_to_f32(
                        ctx.read_u16(byte_off.wrapping_add(i as u32 * 2)),
                    ),
                    (None, false) => ctx.read_f32(byte_off.wrapping_add(i as u32 * 4)),
                })
            };
            // The base-colour tint: the primary layer's tint (or a wheel's AlbedoColour).
            if let Some(p) = r.tint {
                let t = read3(p);
                if t.iter().all(|c| c.is_finite() && *c >= 0.0 && *c <= 8.0) {
                    m.tint = t;
                }
            }
            if let Some(p) = r.light_dir {
                let d = read3(p);
                if d.iter().all(|c| c.is_finite()) && d.iter().any(|c| *c != 0.0) {
                    m.light_dir = d;
                    m.has_light = true;
                }
            }
            if let Some(p) = r.light_col {
                let c = read3(p);
                if c.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 16.0) {
                    m.light_col = c;
                }
            }
        }
        // Ambient: the average colour of the small `diffuseAmbientMap` irradiance texture, if
        // one is bound. It is a coarse (16x16 / 128x128) light probe, so its mean is a good
        // flat ambient for a renderer that does not sample it per-normal. Leaves the default
        // grey ambient when no such map is present.
        if let Some(amb) = textures.iter().find(|t| t.unit == 15)
            && let Some(mean) = snaps!(self).mean_rgb(amb)
        {
            m.ambient = mean;
        }
        m
    }

    /// Diagnostic (VITASLOP_DUMP_FPROG): print the bound fragment program's sampler
    /// parameters (name + unit) once per unique program, so the albedo-selection name
    /// matching can be grounded in the real sampler names a title declares.
    pub(super) fn dump_fragment_program_samplers(&self, ctx: &GuestCtx) {
        use std::sync::Mutex;
        static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        let header = self.bound_fragment_program_header(ctx);
        if header == 0 {
            return;
        }
        {
            let mut seen = SEEN.lock().unwrap();
            if seen.contains(&header) {
                return;
            }
            seen.push(header);
        }
        let count = ctx.read_u32(header.wrapping_add(0x24));
        let base = header
            .wrapping_add(0x28)
            .wrapping_add(ctx.read_u32(header.wrapping_add(0x28)));
        eprintln!("FPROG header={header:#x} params={count}");
        for i in 0..count.min(256) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            if word & 0xf != 2 {
                continue;
            }
            let unit = ctx.read_u32(p.wrapping_add(0xc));
            let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
            let raw = ctx.read_bytes(name_addr, 64);
            let name: String = raw
                .iter()
                .take_while(|&&b| b != 0)
                .map(|&b| b as char)
                .collect();
            eprintln!(
                "  sampler name={name:?} unit={unit} score={}",
                albedo_name_score(&name.to_ascii_lowercase())
            );
        }
    }

    /// Diagnostic (VITASLOP_DUMP_DRAW_GXP=<frame>): dump, per draw of the matching
    /// display frame, everything needed to reason about the fragment-program compositing
    /// the capture renderer cannot yet reproduce - the vertex attribute layout (how many
    /// texcoord SETS the mesh carries), the bound fragment program's FULL parameter table
    /// (samplers with their unit, and the fragment INPUT varyings / category-0 attributes
    /// that tell which TEXCOORD each stage reads), and the bound textures with the sampler
    /// name at each unit. This is the empirical instrument for the wheel-ring artifact: it
    /// shows whether a draw feeds one UV set to several samplers (a composite we must
    /// reproduce) or picks the wrong UV set for a sampler (a binding we can correct).
    #[allow(clippy::too_many_arguments)]
    /// `VITASLOP_DUMP_GXP_BIN=<dir>`: write the raw `SceGxmProgram` blobs (the whole container -
    /// header + parameter table + USSE bytecode) for the bound fragment and vertex programs,
    /// named `<type>_<header-addr>.gxp`, deduped by header address. These are the durable
    /// artifacts the clean-room GXP->WGSL recompiler decodes, and the corpus every ISA question
    /// is settled against. `SceGxmProgram.size` is the u32 at header+0x08 (the container's total
    /// byte length; clamped defensively).
    ///
    /// This has its OWN gate rather than living inside [`Self::dump_draw_gxp`]. Capturing the
    /// corpus and reading a per-draw trace are different jobs: the trace is frame-keyed and
    /// capped because it is gigabytes of text, while a corpus wants every program the whole run
    /// ever binds. Coupling them meant the shaders reachable only late in a level could not be
    /// captured at all without also emitting the trace for every draw before them.
    pub(super) fn dump_gxp_blobs(&self, ctx: &GuestCtx, fh: u32, vh: u32) {
        // Cached: this runs per draw, and reading an unset environment variable on Windows is
        // not free (see `dump_vprog`).
        static WANT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let Some(dir) = WANT.get_or_init(|| std::env::var("VITASLOP_DUMP_GXP_BIN").ok()) else {
            return;
        };
        use std::sync::Mutex;
        static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        for (kind, header) in [("frag", fh), ("vert", vh)] {
            if header == 0 {
                continue;
            }
            {
                let mut seen = SEEN.lock().unwrap();
                if seen.contains(&header) {
                    continue;
                }
                if seen.is_empty() {
                    let _ = std::fs::create_dir_all(dir);
                }
                seen.push(header);
            }
            let size = ctx.read_u32(header.wrapping_add(0x08)).clamp(0x40, 0x40000);
            let bytes = ctx.read_bytes(header, size as usize);
            let path = std::path::Path::new(dir).join(format!("{kind}_{header:08x}.gxp"));
            let _ = std::fs::write(path, &bytes);
        }
    }
}
