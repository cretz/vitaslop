use super::*;

impl VitaState {
    /// The bound vertex program's layout, if recorded.
    #[allow(dead_code)]
    pub(super) fn bound_layout(&self, ctx: &GuestCtx) -> Option<&VertexProgramInfo> {
        self.shader_patcher
            .vertex_program(self.bound_vertex_program(ctx))
    }

    /// 1, 10, 100, ... - the schedule a repeating diagnostic reports on when what matters is
    /// its ORDER OF MAGNITUDE and printing every occurrence would drown the log.
    pub(super) fn is_power_of_ten_impl(mut n: u64) -> bool {
        if n == 0 {
            return false;
        }
        while n.is_multiple_of(10) {
            n /= 10;
        }
        n == 1
    }

    /// Is the bound default uniform buffer for `stage` left over from a DIFFERENT program
    /// than the one about to draw?
    ///
    /// A default uniform buffer is one program's SA bank: its size and its every offset
    /// come from that program's parameter table, and GXM invalidates the reservation when
    /// the program changes (the app reserves again after each `sceGxmSetVertexProgram`).
    /// A draw that arrives with the previous program's buffer still bound therefore reads
    /// another object's uniforms through this program's layout. When lane 0 is a model
    /// matrix - which it is for every 3D vertex program in this title - the result is a
    /// mesh rendered at some other object's position, following it around the world.
    ///
    /// Reported unconditionally (once per program pair): a draw silently transformed by
    /// the wrong matrix is indistinguishable from a faithful render except by eye, and
    /// that is exactly how this class of bug survives.
    pub(super) fn stale_uniforms(
        &mut self,
        stage: &'static str,
        bound_for: u32,
        drawing: u32,
        needs_bytes: u32,
    ) -> bool {
        // `drawing == 0` means the program header was never recorded (an unknown handle),
        // so there is nothing to compare against and no evidence of staleness.
        if bound_for == drawing || drawing == 0 {
            return false;
        }
        // A program with NO default uniform block reads no SA bank, so a buffer left bound
        // for a different program is not starving it of anything - there is nothing for it to
        // be starved of. Dropping the binding is still right (it is not this draw's data), but
        // it is bookkeeping, not a defect, and reporting it as one is what buried three real
        // render defects under seventy lines of noise on the race screen.
        //
        // This is the common case by construction: the direct path reserves per program, and a
        // title only omits the reserve for a program that needs no uniforms. That is why the
        // counts are in the thousands on a title whose picture is broadly correct - the warning
        // was measuring how OFTEN the guest binds a uniform-less fragment program after a
        // uniform-taking one, which is a fact about the title, not about the emulator.
        if needs_bytes == 0 {
            return true;
        }
        // A DIAGNOSTIC ARM (`VITASLOP_GXM_STALE_UNIFORMS=use`): keep the bank rather than drop
        // it. Both answers are wrong in some case - dropping starves a draw whose uniforms are
        // really there, keeping feeds it another object's - and only the frame can say which
        // case a title is in. The arm exists so that question costs one run.
        // Read ONCE: this line is reached by every draw whose bound buffer names another
        // program, which on a baseball at-bat is most of them - a knob lookup there was a mutex,
        // a string hash and an environment probe per draw (0.25% of the render worker, measured).
        static USE_STALE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *USE_STALE.get_or_init(|| {
            crate::knobs::var("VITASLOP_GXM_STALE_UNIFORMS").as_deref() == Ok("use")
        }) {
            return false;
        }
        // >>> EVERY ONE OF THEM, for the DEVICE BUDGET - not just the ones the cadence below
        // prints. A draw that reaches a stage with no uniform bank at all is the defect class,
        // and a gate cannot be built on a log line that fires once per program and then at
        // powers of a thousand. See [`stale_uniform_draws`].
        STALE_UNIFORM_DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Keyed on the program DRAWING, not the pair: what was bound for it varies with the
        // draw order and multiplied the lines - 1,108 of one browser window's status lines,
        // on a title where the affected draws are culled geometry anyway.
        let n = self
            .reported_stale_uniforms
            .entry((stage, 0, drawing))
            .or_insert(0);
        *n += 1;
        // At the first sighting and then on powers of a thousand, so the log carries the
        // SCALE without carrying a line per draw. Four dropped draws in a run is a curiosity;
        // four hundred thousand is a stage rendering with a zeroed uniform bank, which comes
        // out black and looks like a shading bug anywhere else in the pipeline.
        if *n == 1 || ((*n).is_multiple_of(1000) && Self::is_power_of_ten_impl(*n)) {
            tracing::warn!(
                target: "vitaslop::gxm",
                stage,
                bound_for = format_args!("{bound_for:#x}"),
                drawing = format_args!("{drawing:#x}"),
                needs_bytes,
                count = *n,
                // >>> WHAT THIS LINE USED TO SAY WAS NOT WHAT THE CODE DOES, and it cost a
                // session. It read "renders with NO uniform bank at all", which sent a reader
                // (me) to a device dump, a 12,000-frame A/B and a budget row gated at zero
                // before anyone checked the fallback. `record_draw`'s vertex bank read takes
                // `sa_bank_bytes` on this path: the draw renders with the
                // `sceGxmSetUniformDataF` bank, which for a program that really sets its
                // uniforms that way is the CORRECT bank. The A/B came back 24 of 24 shots
                // byte-identical across both arms.
                //
                // It is still worth a line, because the two banks are not interchangeable in
                // general - but it must describe the substitution, not an absence.
                "STALE default uniform buffer: reserved for another program, so this draw's \
                 stage falls back to the sceGxmSetUniformDataF SA bank (it declares \
                 needs_bytes of default uniforms). Right when the title sets them that way, \
                 WRONG when it expected the buffer it reserved - and only a frame can tell \
                 those apart, which `VITASLOP_GXM_STALE_UNIFORMS=use` is the arm for"
            );
        }
        true
    }

    /// The vertex uniforms in effect for the next draw. On the precomputed path
    /// (`bound_vertex_uniform_buf` set by `sceGxmSetPrecomputedVertexState`) read the
    /// default uniform buffer straight from guest memory, sized by the program's default
    /// uniform buffer size. Otherwise use the `sceGxmSetUniformDataF` capture.
    ///
    /// A buffer bound for a different vertex program is NOT this draw's SA bank - see
    /// [`Self::stale_uniforms`] - so it is dropped rather than misread.
    /// WHERE this draw's vertex uniforms live: `Some((address, byte length))` for a bound
    /// default uniform buffer, `None` for "the `sceGxmSetUniformDataF` SA bank".
    ///
    /// Split out from the two readers below because the DECISION - which includes the
    /// staleness check, which reports - must be made once and identically however the bytes
    /// are then taken. `record_draw` needs the same bank as FLOATS (for the transform
    /// reflection) and as BYTES (the recompiler's `vert_sa`), and it used to read floats and
    /// then serialise them back to bytes: a third buffer per draw for data it had already had
    /// in that form. One decision, two readers, no round trip.
    pub(super) fn current_vertex_uniform_src(
        &mut self,
        ctx: &GuestCtx,
        blk: &crate::vita::gxmctx::Block<'_>,
    ) -> Option<(u32, usize)> {
        let mut bound = blk.uniform_binding(crate::vita::gxmctx::off::VERTEX_UNIFORM);
        if bound.buf != 0 {
            let drawing =
                self.vertex_program_header(blk.word(crate::vita::gxmctx::off::VERTEX_PROGRAM));
            let refl = self.reflect_program(ctx, drawing);
            let needs = refl.uniform_size_bytes.max(refl.default_uniform_bytes);
            if bound.header == USER_UNIFORM_BUFFER {
                bound.size = needs;
            } else if self.stale_uniforms("vertex", bound.header, drawing, needs) {
                return None;
            }
        }
        (bound.size >= 4 && bound.buf != 0).then(|| (bound.buf, (bound.size / 4) as usize * 4))
    }

    #[allow(dead_code)]
    pub(super) fn current_vertex_uniforms(
        &mut self,
        ctx: &GuestCtx,
        blk: &crate::vita::gxmctx::Block<'_>,
    ) -> Vec<f32> {
        // Only a BOUND buffer can be stale, so the test belongs behind the same guard the
        // fragment stage uses in `record_draw`. With nothing bound - `vertex_uniform_header` is
        // 0 after a null or never-built precomputed state, and after a cleared reservation -
        // both arms below return the SA bank regardless, so testing staleness here
        // cannot change a pixel and can only report a drop that did not happen. It did:
        // 10,000 `bound_for=0x0` warnings in one retail run, which reads as a rendering
        // defect when the direct uniform path was serving those draws correctly.
        let mut bound = blk.uniform_binding(crate::vita::gxmctx::off::VERTEX_UNIFORM);
        if bound.buf != 0 {
            let drawing =
                self.vertex_program_header(blk.word(crate::vita::gxmctx::off::VERTEX_PROGRAM));
            // Both halves come out of the ONE cached reflection: the header's own size field
            // is memoised beside the reflected extent (see
            // `ProgramReflection::default_uniform_bytes`), so this is a map lookup rather
            // than a map lookup AND a guest word read on every draw.
            let refl = self.reflect_program(ctx, drawing);
            let needs = refl.uniform_size_bytes.max(refl.default_uniform_bytes);
            if bound.header == USER_UNIFORM_BUFFER {
                bound.size = needs;
            } else if self.stale_uniforms("vertex", bound.header, drawing, needs) {
                return self.sa_bank_floats(ctx);
            }
        }
        if bound.size >= 4 && bound.buf != 0 {
            let count = (bound.size / 4) as usize;
            ctx.read_f32s(bound.buf, count)
        } else {
            self.sa_bank_floats(ctx)
        }
    }

    /// Append a draw to the current scene, snapshotting vertex and index bytes.
    pub fn record_draw(
        &mut self,
        ctx: &GuestCtx,
        primitive: u32,
        index_format: u32,
        index_addr: u32,
        index_count: u32,
        index_wrap: u32,
    ) {
        let _all = crate::perf::scope(crate::perf::Phase::DrawTotal);
        let _x = crate::perf::exclude_scope();
        // A draw with no context block behind it reads the GXM DEFAULTS for every piece of
        // sticky state - no bound program, no streams, cull none, depth less-equal - which
        // renders as missing geometry rather than as an error. Nothing else would report it,
        // because reading a default is not a failure anywhere downstream.
        if self.gxm_context == 0 && !self.reported_no_gxm_context {
            self.reported_no_gxm_context = true;
            tracing::warn!(
                target: "vitaslop::gxm",
                "a draw was recorded before sceGxmCreateContext - every piece of sticky GXM \
                 state reads as its DEFAULT, so this draw has no bound program and no vertex \
                 streams"
            );
        }
        // >>> ONE SNAPSHOT OF THE CONTEXT BLOCK, FOR EVERY READER BELOW. See
        // [`crate::vita::gxmctx::Block`]: a draw read that one 652-byte structure through a
        // dozen separate crossings into guest memory, and the guest cannot run during a host
        // call, so one copy answers all of them identically.
        let blk = {
            let _b = crate::perf::scope(crate::perf::Phase::DrawBlockRead);
            let blk = crate::vita::gxmctx::Block::read(ctx, self.gxm_context);
            // A precomputed draw's own program and streams, over the context's - see
            // `draw_precomputed`. The program word and the stream words are not adjacent
            // (the fragment program sits between them), so two overlays. With no context there
            // is nothing to bind into, as the setters' writes were no-ops there.
            match self.pdraw_binding {
                Some((program, streams)) if self.gxm_context != 0 => {
                    use crate::vita::gxmctx::off;
                    blk.overlaid(off::VERTEX_PROGRAM, &[program])
                        .overlaid(off::STREAMS, &streams)
                }
                _ => blk,
            }
        };
        // Occlusion query: this draw contributes to whatever slot the front-face
        // visibility test currently names (no-op when no query is live).
        crate::perf::time(crate::perf::Phase::DrawPrelude, || {
            self.accumulate_visibility(&blk, index_count)
        });
        // Drop a default uniform buffer still bound for a DIFFERENT program before anything
        // below reads it (see [`Self::stale_uniforms`]). The vertex stage is handled inside
        // `current_vertex_uniforms`; the fragment stage has several independent readers
        // (material reflection, the recompiler's `frag_sa`, the draw dump), so clear it once
        // here and they all see the same truth.
        let reflect_phase = crate::perf::scope(crate::perf::Phase::DrawReflect);
        let mut frag_uniform = blk.uniform_binding(crate::vita::gxmctx::off::FRAGMENT_UNIFORM);
        let fheader =
            self.fragment_program_header(blk.word(crate::vita::gxmctx::off::FRAGMENT_PROGRAM));
        if frag_uniform.buf != 0 {
            let drawing = fheader;
            // How much SA bank the program ABOUT TO DRAW actually declares. Zero means it
            // reads none, and then a mismatched binding starves it of nothing - see
            // `stale_uniforms`.
            // Both halves come out of the ONE cached reflection: the header's own size field
            // is memoised beside the reflected extent (see
            // `ProgramReflection::default_uniform_bytes`), so this is a map lookup rather
            // than a map lookup AND a guest word read on every draw.
            let refl = self.reflect_program(ctx, drawing);
            let needs = refl.uniform_size_bytes.max(refl.default_uniform_bytes);
            if frag_uniform.header == USER_UNIFORM_BUFFER {
                frag_uniform.size = needs;
            } else if self.stale_uniforms("fragment", frag_uniform.header, drawing, needs) {
                frag_uniform.buf = 0;
                frag_uniform.size = 0;
            }
        }
        // Reflect both bound programs ONCE, here. Every name-based lookup below reads
        // these instead of re-walking a parameter table (see `ProgramReflection`).
        let vhandle = blk.word(crate::vita::gxmctx::off::VERTEX_PROGRAM);
        let vheader = self.vertex_program_header(vhandle);
        let vref = self.reflect_program(ctx, vheader);
        let fref = self.reflect_program(ctx, fheader);
        // The bound program's attributes, and the layout of each stream they name. The
        // attributes are rewritten below onto the single interleaved buffer this draw is
        // captured into, so what goes into the `Draw` is the REPACKED layout, not the
        // guest's.
        // >>> ALL OF IT PRECOMPUTED, at the moment the vertex program was created. See
        // `VertexProgramInfo::packed_attributes`: the packed stride, the per-stream bases, the
        // single-stream test and the rebased attribute list are constants of the PROGRAM, and
        // building them per draw was four allocations and two loops on the hottest path here.
        let (attributes, streams, base, layout, stride, single_stream) =
            match self.shader_patcher.vertex_program(vhandle) {
                Some(info) => (
                    info.packed_attributes.clone(),
                    info.used_streams.clone(),
                    info.stream_base.clone(),
                    info.layout.clone(),
                    info.packed_stride,
                    info.single_stream,
                ),
                None => (
                    crate::capture::no_attributes(),
                    std::sync::Arc::from(&[][..]),
                    std::sync::Arc::from(&[][..]),
                    RowLayout::default(),
                    0,
                    false,
                ),
            };
        drop(reflect_phase);
        // >>> THE DRAW'S UNIFORM BANKS IN ONE CROSSING. Their addresses are known now (the block
        // is in hand and the programs reflected), and the readers below - the fragment bank, the
        // vertex bank, the SA-resident buffers of both stages - each used to cross into guest
        // memory on their own: ~3.5 bulk reads a draw counting the block, at ~650 draws a frame,
        // where a crossing is a call across the browser's shared-memory boundary. They read
        // through `uctx` instead, which serves the prefetched bytes and falls through to guest
        // memory for anything not fetched. A DIRECT memory (native) fetches nothing: a read
        // there is already a load. The guest cannot run during a host call, so bytes fetched
        // here are the bytes those reads would have seen.
        let vert_src = if gxp_live_capture() {
            self.current_vertex_uniform_src(ctx, &blk)
        } else {
            None
        };
        let mut uniform_ranges: Vec<(usize, usize)> = Vec::new();
        if gxp_live_capture() && !ctx.mem.direct() {
            let mut addrs: Vec<(u32, usize)> = Vec::new();
            if frag_uniform.buf != 0 {
                addrs.push((frag_uniform.buf, frag_uniform.size as usize));
            }
            addrs.extend(vert_src);
            self.sa_uniform_ranges(ctx, &blk, vheader, UniformStage::Vertex, &mut addrs);
            self.sa_uniform_ranges(ctx, &blk, fheader, UniformStage::Fragment, &mut addrs);
            uniform_ranges.extend(addrs.into_iter().filter_map(|(a, l)| {
                ctx.offset(a)
                    .map(|o| (o, l.min(ctx.mem.len().saturating_sub(o))))
            }));
        }
        let mut uniform_mem = PrefetchedMemory::reads_only(&*ctx.mem, &uniform_ranges);
        let (mut uregs, mut uvfp) = ([0u32; REG_COUNT], [0u32; VFP_ARG_COUNT]);
        let uctx = GuestCtx::new(&mut uregs, &mut uvfp, &mut uniform_mem, ctx.base);
        // Index element size: U16 (0) is 2 bytes, U32 is 4.
        let index_elem = if index_format == 0 { 2 } else { 4 };
        // Snapshot exactly the vertices this draw REFERENCES, not the whole prefix of the
        // stream. A chunked world mesh draws a few hundred vertices out of a shared buffer of
        // tens of thousands, so copying `0..=max_index` per draw costs hundreds of megabytes a
        // frame (and reads far past what the draw can touch). Take the `min..=max` window and
        // rebase the indices onto it, which leaves every consumer's indexing unchanged.
        //
        // Read, scanned, rebased and shared in one call - and skipped entirely for a buffer the
        // guest provably has not written since. See `get_or_read_indices`.
        // >>> READ AT END OF SCENE, NOT HERE, unless the arm is off. See `defer_geometry`.
        let geometry_phase = crate::perf::scope(crate::perf::Phase::DrawGeometry);
        let deferring = defer_geometry();
        let bound_streams = blk.streams();
        // An instanced draw read now is expanded now (see `read_instanced_draw`); the
        // deferred arm expands it at the guest's GPU wait like everything else.
        let instanced = if deferring {
            None
        } else {
            snaps!(self).read_instanced_draw(
                ctx,
                index_addr,
                index_count as usize * index_elem,
                index_elem,
                index_wrap,
                &streams,
                &bound_streams,
                &layout,
                stride,
            )
        };
        let draw_indices = if deferring {
            DrawIndices {
                bytes: crate::capture::no_bytes(),
                first_vertex: 0,
                vertex_count: 0,
                gather: None,
            }
        } else if let Some((idx, _)) = &instanced {
            idx.clone()
        } else {
            snaps!(self).get_or_read_indices(
                ctx,
                index_addr,
                index_count as usize * index_elem,
                index_elem,
            )
        };
        let indices = draw_indices.bytes.clone();
        // Interleave every stream this draw's attributes name into ONE buffer, and rewrite
        // the attributes onto it. A vertex here is the concatenation of its row from each
        // used stream, so the result is a plain single-stream mesh that indexes exactly as
        // the guest's did - every consumer (software raster, fixed-function GPU path, the
        // recompiler's repack) keeps working unchanged and now sees the data it was
        // previously decoding out of stream 0 by mistake.
        //
        // A per-instance stream contributes its row 0 to every vertex here, which is what
        // instance 0 reads; an instanced draw took the expanded path above instead.
        let vertices: Arc<[u8]> = if deferring {
            crate::capture::no_bytes()
        } else if let Some((_, v)) = instanced {
            v
        } else {
            let mut snapshots = snaps!(self);
            crate::perf::time(crate::perf::Phase::DrawVertices, || {
                snapshots.read_draw_vertices(
                    ctx,
                    &draw_indices,
                    single_stream,
                    &streams,
                    &bound_streams,
                    &base,
                    &layout,
                    stride,
                )
            })
        };
        drop(geometry_phase);
        // The draw's fragment and vertex TEXTURE LISTS: proven now, or - under the resolver with
        // `VITASLOP_SMP_DEFER_TEXTURES` - described now and proven by the resolver with the
        // geometry (see `DeferredTextures`).
        let tex_phase = crate::perf::scope(crate::perf::Phase::DrawTexAll);
        self.sync_vertex_textures(&blk);
        let tex_defer = if deferring && defer_textures() {
            Some(self.defer_draw_textures(&blk, fheader, &fref))
        } else {
            None
        };
        let (textures, vertex_textures) = if tex_defer.is_some() {
            (crate::capture::no_textures(), crate::capture::no_textures())
        } else {
            self.draw_textures_now(ctx, &blk, fheader, &fref)
        };
        drop(tex_phase);
        if dump_fprog() {
            self.dump_fragment_program_samplers(ctx);
        }
        // Vertex uniforms: on the precomputed path the game wrote them into a default
        // uniform buffer bound by `sceGxmSetPrecomputedVertexState`, so read that guest
        // buffer now (its contents are current at draw time). On the direct path the
        // buffer is 0 and we fall back to the `sceGxmSetUniformDataF` capture.
        // >>> THE FRAGMENT SA BANK, READ ONCE FOR THE TWO READERS THAT WANT IT.
        //
        // The GXP capture below needs the whole bank; the material reflection needs nine
        // floats out of it. Read here, before either, so the material's nine reads come out
        // of these bytes instead of being nine separate crossings into guest memory - and so
        // the bank is read ONCE per draw rather than once per reader. Charged to the GXP
        // phase because that is the phase that owes the read; the material's share of it was
        // never a bulk read at all.
        //
        // Empty when the recompiler is off (the fixed-function path does not want the bank,
        // and reading it there would be pure cost) or when nothing is bound - and
        // `reflect_fragment_material` falls back to the per-word read for exactly that case.
        let frag_sa: Arc<[u8]> = {
            let _g = crate::perf::scope(crate::perf::Phase::DrawGxpCapture);
            if gxp_live_capture() && frag_uniform.buf != 0 {
                let _s = crate::perf::scope(crate::perf::Phase::DrawGxpFragSa);
                // Into the `Arc`'s own allocation, not a `Vec` that is copied into one below.
                uctx.read_bytes_arc(frag_uniform.buf, frag_uniform.size as usize)
            } else {
                crate::capture::no_bytes()
            }
        };
        let uniform_phase = crate::perf::scope(crate::perf::Phase::DrawUniforms);
        // >>> THE VERTEX BANK IS READ ONCE, IN THE FORM THE GUEST WROTE IT.
        //
        // Two consumers want it: the transform reflection wants FLOATS, and the recompiler
        // wants the RAW BYTES as the guest wrote them, BEFORE the composed MVP is stamped over
        // lanes 0..16 below (`vert_sa_raw` also covers the direct `sceGxmSetUniformDataF`
        // path, where the bound pointer is 0). It used to read floats and then serialise them
        // back to bytes - a third buffer per draw holding what the read had already produced
        // and thrown away, since a guest read is bytes to begin with.
        //
        // Off the recompiler path nothing wants the bytes, and there the float read stays as
        // it was: on native it BORROWS guest memory and converts in place, so materialising a
        // byte buffer there would be a cost with no reader.
        // >>> THE WHOLE BANK IS CONVERTED, AND READING ONLY THE REFLECTED LANES WAS SLOWER.
        //
        // The recompiled shader reads the guest's BYTES, so the floats here serve only the
        // three reflections below and `interpret_draw`'s sixteen - about fifty lanes at
        // offsets the reflection already knows. Reading exactly those lanes out of the bytes
        // and skipping this conversion is the obvious saving and it MEASURED AS A LOSS: two
        // interleaved browser runs put the lane-reader at cpu p25 8.30 and 8.07 ms/frame
        // against 7.68 for this. A bank is a few dozen floats, `chunks_exact(4)` vectorises,
        // and fifty bounds-checked per-lane reads through a closure cost more than converting
        // it. Do not re-try it without a measurement.
        let want_fixed_function = fixed_function_wanted();
        let mut uniforms = std::mem::take(&mut self.uniform_float_scratch);
        uniforms.clear();
        // The bank's REAL length in lanes - what says "shorter than an MVP" below - which on
        // the recompiled path is no longer `uniforms.len()` (see the prefix note next).
        let mut bank_lanes_raw = None;
        let vert_sa_raw = if gxp_live_capture() {
            // >>> STRAIGHT INTO THE `Arc`, because the ONLY consumer wants one (an extra
            // allocation and copy of the bank per draw per stage otherwise). `vert_src` was
            // decided above, once, so a stale binding is reported once a draw.
            let raw = match vert_src {
                Some((addr, len)) => uctx.read_bytes_arc(addr, len),
                None => self.sa_bank_bytes(ctx),
            };
            crate::perf::note_bytes(crate::perf::Phase::DrawUniforms, raw.len());
            // >>> ONLY THE PREFIX THE REFLECTIONS CAN REACH. Still the vectorised conversion the
            // note above insists on, just shorter: a lane past `need` is read by nothing on this
            // path (the reflections take 16 lanes at their offsets, `interpret_draw` lanes
            // 0..16), and a lane past the bank's end reads `None` exactly as before. MEASURED on
            // a football title's match: 1.1 KB of bank per draw on average (skinning palettes),
            // ~275 floats converted to read ~50.
            let need = if want_fixed_function {
                usize::MAX
            } else {
                [vref.mvp_off, vref.world_off, vref.proj_off]
                    .iter()
                    .flatten()
                    .map(|&o| o as usize + 16)
                    .chain(vref.exposure_off.map(|o| o as usize + 1))
                    .fold(16, usize::max)
            };
            let lanes = raw.len() / 4;
            bank_lanes_raw = Some(lanes);
            uniforms.extend(
                raw[..lanes.min(need) * 4]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
            raw
        } else {
            // The fixed-function path's float read BORROWS guest memory natively and converts
            // in place, so it stays as it was; the extra copy into the scratch is one path's
            // cost and that path is not the one that ships.
            uniforms.extend_from_slice(&self.current_vertex_uniforms(ctx, &blk));
            crate::capture::no_bytes()
        };
        let lane = |i: usize| -> Option<f32> { uniforms.get(i).copied() };
        // The model-to-world matrix (for bringing the vertex normal into world space for
        // lighting). Read from the ORIGINAL bank, before the composed MVP is stamped over
        // lanes 0..16 (which is exactly where `vsModelToWorldMatrix` usually sits).
        // >>> STILL COMPUTED ON THE RECOMPILED PATH, AND DELIBERATELY.
        // It is two 16-lane reads out of a bank already in hand, and it is not only the
        // fixed-function pipeline's: `object_locations` reads the world matrix's translation
        // and basis to say WHERE a mesh is, which is the whole of the `locate` tool and of
        // every driving controller built on it. Handing those identity would not fail, it
        // would answer the origin - see [[vitaslop-camera-not-address]].
        // >>> READ ONCE, USED TWICE. The model->world matrix and the composed MVP are both
        // functions of the SAME pair of reflected matrices, and each used to read the pair for
        // itself - 32 lanes of the bank fetched twice per draw, ~626 draws a frame, for bytes
        // the first read already had.
        const IDENTITY_4X4: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let world_proj = Self::reflected_world_proj_by(&vref, lane);
        let world = world_proj.map(|(w, _)| w).unwrap_or(IDENTITY_4X4);
        // Recover the true clip-space transform from the vertex program's reflected
        // uniforms. A 3D shader keeps a per-object model->world matrix and a shared
        // world->projection matrix as separate uniforms (named e.g. `vsModelToWorldMatrix`
        // / `vsWorldToProjectionMatrix`), and multiplies them in the shader. The capture
        // renderer has no shader, so compose that product here and stamp it over the
        // first 16 floats, which the software/GPU paths read as the MVP. A shader with a
        // single combined transform (2D UI: `vsPrimRenderTransform` at offset 0) has no
        // projection matrix, so this leaves its MVP untouched.
        let composed = match vref.mvp_off {
            Some(off) => Self::lanes_by(&lane, off as usize),
            None => world_proj.map(|(w, p)| Self::compose_mvp(&w, &p)),
        };
        if dump_vprog() {
            self.dump_vertex_program_params(ctx, vhandle);
        }
        // >>> WHAT THE `Draw` ACTUALLY CARRIES, WHICH ON THE RECOMPILED PATH IS SIXTEEN FLOATS.
        //
        // The whole bank is here only because the fixed-function pipeline reads it as its
        // uniform block. A recompiled draw reads `vert_sa` - the guest's own bytes - and the
        // ONLY consumer of this field is `interpret_draw`, which classifies the draw's
        // coordinate space from lanes 0..16 (the MVP stamped just above, or the shader's own
        // combined transform). Carrying the tail was an allocation and a copy per draw for
        // bytes nothing downstream reads. A bank SHORTER than sixteen floats keeps its length,
        // because that length is what says "this is not an MVP draw".
        let bank_lanes = bank_lanes_raw.unwrap_or(uniforms.len());
        let mut draw_uniforms: Vec<f32> = if want_fixed_function {
            uniforms.clone()
        } else {
            uniforms[..bank_lanes.min(16)].to_vec()
        };
        if let Some(mvp) = composed
            && draw_uniforms.len() >= 16
        {
            draw_uniforms[..16].copy_from_slice(&mvp);
        }
        // >>> READ AFTER THE STAMP, WHICH IS WHERE IT ALWAYS WAS.
        //
        // A shader whose reflected exposure lane sits inside 0..16 reads the composed MVP
        // there rather than its own value. That is almost certainly not what the title means,
        // but it is what this has always done and it is a fixed-function value - so the order
        // is preserved rather than quietly corrected, and the recompiled path (where nothing
        // reads it) takes the same lanes for the same reason.
        let stamped = |i: usize| -> Option<f32> {
            match &composed {
                Some(mvp) if i < 16 && bank_lanes >= 16 => Some(mvp[i]),
                _ => lane(i),
            }
        };
        let exposure = Self::reflected_exposure_by(&vref, stamped);
        // The per-material fragment inputs (tint / directional light / ambient), reflected
        // from the fragment program's uniforms so the renderer reproduces the LIT colour.
        let material = if want_fixed_function {
            self.reflect_fragment_material(ctx, &fref, &textures, frag_uniform.buf, &frag_sa)
        } else {
            // The neutral material - what this already reflects for a shader declaring none.
            crate::capture::FragmentMaterial::default()
        };
        drop(uniform_phase);
        // The per-draw diagnostic dump, after everything it reports has been computed
        // (it prints the reflected transform and material rather than recomputing them).
        self.dump_gxp_blobs(ctx, fheader, vheader);
        self.dump_draw_gxp(
            ctx,
            vhandle,
            &blk.streams(),
            &vref,
            &material,
            &textures,
            &attributes,
            primitive,
            index_count,
            stride,
            frag_uniform.buf,
            index_addr,
            if index_format == 0 { 2 } else { 4 },
        );
        // Snapshot the raw shader blobs + SA uniform bytes for the GXP->WGSL recompiler
        // path, but only when it is enabled (the reads are pure cost on the default
        // fixed-function path). The blobs come from `program_blob`, which reads each
        // container out of guest memory ONCE and hands every later draw a shared `Arc` -
        // see there for why a per-draw read is not affordable.
        let gxp_phase = crate::perf::scope(crate::perf::Phase::DrawGxpCapture);
        let (vprog, fprog, fprog_patched_vprog, vert_sa, frag_sa) = if gxp_live_capture() {
            let blobs_phase = crate::perf::scope(crate::perf::Phase::DrawGxpBlobs);
            let vprog = self.program_blob(ctx, vheader);
            let fprog = self.program_blob(ctx, fheader);
            // The vertex program the fragment was PATCHED against, when that is not the one
            // this draw binds - see `capture::Draw::fprog_patched_vprog`.
            let patched = self.fragment_program_patched_vertex(
                blk.word(crate::vita::gxmctx::off::FRAGMENT_PROGRAM),
            );
            let fprog_patched_vprog = if patched != 0 && patched != vheader {
                let pv = self.program_blob(ctx, patched);
                // Pointer identity first: `program_blob` hands every draw the same shared `Arc`
                // for one header, so this is usually decided without comparing kilobytes.
                if Arc::ptr_eq(&pv, &vprog) || pv[..] == vprog[..] {
                    crate::capture::no_program()
                } else {
                    report_patched_against_other_vertex(fheader, vheader, patched);
                    pv
                }
            } else {
                crate::capture::no_program()
            };
            // >>> THE SA BANK A RECOMPILED SHADER READS IS THE REGISTER FILE, NOT ONE BUFFER.
            //
            // The bytes captured above are the DEFAULT uniform buffer - the pre-stamp raw
            // uniforms, covering both the bound buffer and the direct `sceGxmSetUniformDataF`
            // path - and that is what the reflections above index, from zero. The emitted WGSL
            // does not: it indexes `sa[k]` by absolute SA register, and a program can have the
            // default buffer somewhere other than register 0 with another bound buffer resident
            // below it. `sa_uniform_image` lays both stages out the way the driver does; for a
            // program with the usual layout it hands the same bytes straight back.
            drop(blobs_phase);
            let _sa_phase = crate::perf::scope(crate::perf::Phase::DrawGxpSaImage);
            let vert_sa =
                self.sa_uniform_image(&uctx, &blk, vheader, UniformStage::Vertex, vert_sa_raw);
            let frag_sa =
                self.sa_uniform_image(&uctx, &blk, fheader, UniformStage::Fragment, frag_sa);
            // >>> THE FRAGMENT BANK IS THE ONE ALREADY READ, NOT A SECOND READ OF THE SAME
            // BYTES. It was read above for the material's nine floats, under the identical
            // condition, and the guest cannot run between the two - so this used to copy the
            // whole bank out of guest memory a SECOND time on every draw, which is a bulk read
            // and an allocation per draw on the hottest path in the engine. The comment above
            // that read already said "read once for the two readers"; this is the second
            // reader actually taking it.
            // Into an `Arc` HERE, once, at the only place these are produced - see
            // `capture::Draw::vert_sa`. From here to the renderer they are shared, so the
            // clone `RenderSceneBuilder::build` makes is a refcount bump.
            (vprog, fprog, fprog_patched_vprog, vert_sa, frag_sa)
        } else {
            (
                crate::capture::no_program(),
                crate::capture::no_program(),
                crate::capture::no_program(),
                crate::capture::no_bytes(),
                crate::capture::no_bytes(),
            )
        };
        // ...and WHERE the fragment bank came from, which the bytes cannot say. See
        // `capture::Draw::frag_sa_addr`.
        let frag_sa_addr = if gxp_live_capture() {
            frag_uniform.buf
        } else {
            0
        };
        // The guest-memory window the vertex program's 0xE8 loads read through, snapshotted
        // at draw time like every other guest input. One map lookup for a program without
        // loads, which is every program of every other captured title.
        let (mem_windows, frag_mem_windows) = if gxp_live_capture() {
            let _w = crate::perf::scope(crate::perf::Phase::DrawGxpWindows);
            (
                self.capture_mem_windows(ctx, &blk, vheader, ProgramStage::Vertex, deferring, &vert_sa),
                self.capture_mem_windows(ctx, &blk, fheader, ProgramStage::Fragment, deferring, &frag_sa),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        // The buffers a storing fragment program writes through, read NOW - see
        // `capture_store_windows`. One memo lookup for every other program.
        let frag_store_windows = if gxp_live_capture() {
            self.capture_store_windows(ctx, &blk, fheader)
        } else {
            Vec::new()
        };
        // Under `VITASLOP_DUMP_DRAW_GXP=<frame>`, the FRAGMENT windows' bytes ride with the
        // draw dump: a uniform that lives in a window (container 7 here) has no other
        // steady-state readout - the once-per-pair inputs dump shows its FIRST sight only.
        if Self::dump_draw_frame_matches(self.cur_frame) {
            for (i, (a, b)) in frag_mem_windows.iter().enumerate() {
                // At STATUS (not stderr): the browser has no stderr worth reading.
                tracing::info!(target: "vitaslop::status", "fwindow{i} at {a:#010x} ({} B): {:02x?}", b.len(), &b[..b.len().min(128)]);
            }
            // >>> AND THE VERTEX WINDOWS, AS FLOATS, because a TRANSFORM can live in one.
            //
            // Only the fragment side was printed, on the reading that a windowed uniform is a
            // blend coefficient. It is not always: MEASURED on a baseball title, a screen-space quad's
            // vertex program loads sixteen words through its window and MADs them against the
            // position - its whole MODEL-VIEW-PROJECTION is in there, and the `TRANSFORM` line
            // above says `STALE-ubuf`/`composed=no` because the matrix is in no bank that line
            // reads. With the window unprinted there was no way to see what the shader was
            // actually multiplying by, and a quad drawn 114x113 pixels wide came out covering
            // half the frame with nothing to say why. Floats, because that is what a matrix is.
            for (i, (a, b)) in mem_windows.iter().enumerate() {
                let f: Vec<String> = b
                    .chunks_exact(4)
                    .take(32)
                    .map(|c| format!("{:.4}", f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                    .collect();
                tracing::info!(
                    target: "vitaslop::status",
                    "vwindow{i} at {a:#010x} ({} B) f0..31 = [{}]",
                    b.len(),
                    f.join(","),
                );
            }
        }
        drop(gxp_phase);
        let render_state = {
            let _r = crate::perf::scope(crate::perf::Phase::DrawRenderState);
            use crate::vita::gxmctx::off;
            // The two contiguous regions the fixed-function state occupies: the scalars,
            // viewport and region clip below the program handles, and the back-face stencil
            // block that sits past the sampler array.
            const FRONT_LEN: usize = (off::FRONT_VISIBILITY_TEST_OP + 4 - off::CULL_MODE) as usize;
            const BACK_LEN: usize = (off::AFTER_BACK_STENCIL - off::AFTER_TEXTURES) as usize;
            let front = blk.span(off::CULL_MODE, FRONT_LEN);
            let back = blk.span(off::AFTER_TEXTURES, BACK_LEN);
            match &self.render_state_memo {
                Some((f, b, rs)) if &**f == front && &**b == back => rs.clone(),
                _ => {
                    let rs = std::sync::Arc::new(blk.render_state());
                    self.render_state_memo = Some((
                        front.to_vec().into_boxed_slice(),
                        back.to_vec().into_boxed_slice(),
                        rs.clone(),
                    ));
                    rs
                }
            }
        };
        self.uniform_float_scratch = uniforms;
        let record_phase = crate::perf::scope(crate::perf::Phase::DrawRecord);
        let draw = crate::capture::Draw {
            primitive,
            index_format,
            index_count,
            vertices,
            vertex_stride: stride,
            attributes,
            vertex_textures,
            indices,
            uniforms: draw_uniforms,
            textures,
            render_state,
            blend: self
                .fragment_program_blend(blk.word(crate::vita::gxmctx::off::FRAGMENT_PROGRAM)),
            fragment_program_header: fheader,
            exposure,
            material,
            world,
            vprog,
            fprog,
            fprog_patched_vprog,
            vert_sa,
            frag_sa,
            frag_sa_addr,
            mem_windows,
            frag_mem_windows,
            frag_store_windows,
            shader_expanded: Self::reflected_shader_expanded(&vref),
        };
        let accepted = match self.scene.as_mut() {
            Some(scene) => {
                scene.draws.push(draw);
                true
            }
            // A draw outside begin/endScene has nowhere to go. That is a real hole in the
            // frame, so it is logged rather than dropped in silence.
            None => {
                tracing::debug!(target: "vitaslop::gxm", index_count, "draw outside a scene - DROPPED");
                false
            }
        };
        // In the SAME order as the draws, and only for a draw the scene accepted -
        // `resolve_deferred_geometry` walks the two in lockstep.
        if deferring && accepted {
            self.deferred_geometry.push(DeferredGeometry {
                index_addr,
                index_len: index_count as usize * index_elem,
                index_elem,
                index_wrap,
                streams,
                bound_streams,
                base,
                layout,
                stride,
                single_stream,
                tex: tex_defer,
            });
        }
        drop(record_phase);
    }

    /// Describe the draw's texture bindings for the resolver instead of proving them - see
    /// [`DeferredTextures`]. Cheap: a 384-byte compare and, when the bytes changed, a copy.
    pub(super) fn defer_draw_textures(
        &mut self,
        blk: &crate::vita::gxmctx::Block,
        fheader: u32,
        fref: &ProgramReflection,
    ) -> Box<DeferredTextures> {
        use crate::vita::gxmctx::{MAX_TEXTURE_UNITS, TEXTURE_STRIDE, off};
        let span = blk.span(off::TEXTURES, MAX_TEXTURE_UNITS * TEXTURE_STRIDE as usize);
        let want = fref.sampler_units;
        // The failure-only report `draw_textures_now` makes on an empty list, from the same
        // bytes without decoding them: no bound slot among the units the list would keep.
        // Skipped outright once that report has been made - it says itself only once, and the
        // scan below ran on every draw to find that out.
        if want != 0 && !self.reported_draw_without_textures {
            let narrow = sampler_narrow_enabled();
            let any = (0..MAX_TEXTURE_UNITS).any(|u| {
                let at = u * TEXTURE_STRIDE as usize;
                let addr = u32::from_le_bytes([span[at], span[at + 1], span[at + 2], span[at + 3]]);
                addr != 0 && (!narrow || (u < 32 && want & (1u32 << u) != 0))
            });
            if !any {
                self.report_draw_without_bound_textures(blk, fheader, want);
            }
        }
        let same = matches!(&self.deferred_tex_last,
            Some((b, f, scene)) if *f == fheader && *scene == self.tex_scene_seq && &**b == span);
        let span = if same {
            None
        } else {
            match &mut self.deferred_tex_last {
                Some((b, f, scene)) if b.len() == span.len() => {
                    b.copy_from_slice(span);
                    *f = fheader;
                    *scene = self.tex_scene_seq;
                }
                slot => *slot = Some((span.into(), fheader, self.tex_scene_seq)),
            }
            Some(span.into())
        };
        let vert = if self.bound_vertex_textures.is_empty() {
            None
        } else {
            let generation = self.vertex_texture_gen;
            match &self.deferred_vert_last {
                Some((g, list)) if *g == generation => Some((generation, list.clone())),
                _ => {
                    let list: Arc<[TextureBinding]> = self.bound_vertex_textures.as_slice().into();
                    self.deferred_vert_last = Some((generation, list.clone()));
                    Some((generation, list))
                }
            }
        };
        Box::new(DeferredTextures {
            span,
            fheader,
            want,
            albedo: Self::fragment_albedo_unit(fref),
            vert,
        })
    }

    /// The draw's texture lists, proven NOW - the fragment list (albedo first) and the vertex
    /// list. What `record_draw` always did; see [`Self::defer_draw_textures`] for the other form.
    pub(super) fn draw_textures_now(
        &mut self,
        ctx: &GuestCtx,
        blk: &crate::vita::gxmctx::Block,
        fheader: u32,
        fref: &ProgramReflection,
    ) -> (
        Arc<[crate::capture::BoundTexture]>,
        Arc<[crate::capture::BoundTexture]>,
    ) {
        // Snapshot every bound fragment texture (decoded from its control words),
        // sorted by unit so unit 0 is first. `bound_textures` is already kept sorted by
        // unit as it is bound, so this reads it in place rather than cloning and
        // re-sorting a fresh Vec for every draw.
        // >>> THE GATE, TIMED SEPARATELY FROM WHAT IT GATES.
        //
        // Everything down to the `snapshot_sets` lookup is paid by EVERY draw, hit or miss:
        // the sampler block comes out of the guest's context, decodes into bindings, is
        // hashed, and the hash is looked up. Only what follows is the miss path. Keeping
        // them in one phase was hiding which of the two a fix would have to land on, and
        // the two fixes have nothing in common.
        let bind_phase = crate::perf::scope(crate::perf::Phase::DrawTextureBind);
        // >>> THE PREVIOUS DRAW'S ANSWER, IF THIS DRAW BINDS THE SAME BYTES.
        //
        // Everything below - the sixteen-slot decode, the fold, the map probe and the exact
        // verify - re-derives a list that a batch of draws sharing a material produces
        // identically every time. One `memcmp` of the sampler block answers it instead. See
        // `TextureSnapshots::last_set` for why a hit is exact rather than a guess, and why
        // it is refused across a scene boundary.
        let sampler_span = blk.span(
            crate::vita::gxmctx::off::TEXTURES,
            crate::vita::gxmctx::MAX_TEXTURE_UNITS * crate::vita::gxmctx::TEXTURE_STRIDE as usize,
        );
        let previous_set = snaps!(self).set_from_previous_draw(sampler_span, fheader);
        if previous_set.is_some() {
            crate::perf::note_hit(crate::perf::Phase::DrawTexSetPrev);
        }
        // The fragment bindings come from the CONTEXT BLOCK now, because that is where the
        // inlined `sceGxmSetFragmentTexture` writes them and the host no longer sees most
        // binds at all. Read in unit order, which is the order this list has always been in.
        //
        // `bound_textures` is reused as the scratch buffer rather than allocated here: it is
        // taken and put back for the same reason it always was - the decode needs `self`
        // borrowed mutably while the bindings are borrowed - and keeping the allocation
        // avoids a Vec per draw on the hottest path in the engine.
        let mut frag_binds = std::mem::take(&mut self.bound_textures);
        // On a previous-draw hit `frag_binds` is left EXACTLY as the last draw left it, which
        // is the same list these identical bytes decode to - so the decode, the fold, the map
        // probe and the exact verify are all skipped and nothing downstream can tell.
        let (set_key, cached_set) = match &previous_set {
            Some(list) => (0u64, Some(list.clone())),
            None => {
                frag_binds.clear();
                let decode_phase = crate::perf::scope(crate::perf::Phase::DrawTexBindDecode);
                let context = self.gxm_context;
                if context != 0 {
                    // ONE borrow of the whole sampler block, not forty `read_u32`s - each of
                    // those is a virtual call through `dyn GuestMemory`, and on this path (627
                    // draws a frame) the calls were the cost rather than the bytes. See
                    // `gxmctx::texture_bindings`. The scratch list is reused for the same
                    // reason `bound_textures` is.
                    let mut raw = std::mem::take(&mut self.bound_binding_scratch);
                    blk.texture_bindings(&mut raw);
                    // >>> ONLY THE UNITS THIS PROGRAM SAMPLES. See
                    // `ProgramReflection::sampler_units` for the measurement and for why a
                    // slot the program does not declare cannot reach a pixel.
                    //
                    // `VITASLOP_SAMPLER_NARROW=0` is the A/B arm: every bound slot, which is
                    // what this did before. A zero mask means the table said nothing, and then
                    // nothing is narrowed - a program that genuinely declares no samplers
                    // produces an empty list either way.
                    let want = fref.sampler_units;
                    let narrow = want != 0 && sampler_narrow_enabled();
                    for (unit, b) in raw.iter() {
                        if narrow && (*unit >= 32 || want & (1u32 << *unit) == 0) {
                            continue;
                        }
                        frag_binds.push(TextureBinding {
                            unit: *unit,
                            addr: b.addr,
                            words: b.words,
                            from_precomputed: b.from_precomputed,
                        });
                    }
                    self.bound_binding_scratch = raw;
                }
                drop(decode_phase);
                // >>> THE WHOLE LIST IS SHARED ACROSS SCENES, RE-PROVEN PER SCENE. See
                // `TextureSnapshots::snapshot_sets`.
                //
                // The key is what the list is a function of: the bindings, and the fragment
                // program header, which decides the albedo reorder below. The reorder is
                // applied BEFORE the entry is stored, so a hit is the finished list.
                let set_key = {
                    let _f = crate::perf::scope(crate::perf::Phase::DrawTexBindFold);
                    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                    let mut mix = |v: u64| {
                        h ^= v;
                        h = h.wrapping_mul(0x0000_0100_0000_01b3);
                    };
                    mix(fheader as u64);
                    for b in &frag_binds {
                        mix(b.unit as u64);
                        // >>> THE DESCRIPTOR ADDRESS IS PART OF THE KEY BECAUSE IT IS PART OF
                        // >>> THE EQUALITY TEST. `set_validated` requires `addr` to match; a
                        // fold that leaves it out therefore sends every "same words, different
                        // descriptor" set to the SAME bucket, where the check rejects it as a
                        // collision and the whole binding list is rebuilt. MEASURED before this
                        // line existed: **16,384+ such rejections in ~6,000 frames**, about
                        // three per frame, which no 64-bit hash explains.
                        // See `report_set_key_collision`.
                        mix(b.addr as u64);
                        for w in b.words {
                            mix(w as u64);
                        }
                    }
                    h
                };
                let _p = crate::perf::scope(crate::perf::Phase::DrawTexBindProbe);
                let cached = snaps!(self).set_validated(ctx, set_key, fheader as u64, &frag_binds);
                (set_key, cached)
            }
        };
        drop(bind_phase);
        // >>> THE FAILURE-ONLY REPORT FOR "the bound units are []". See
        // `report_draw_without_bound_textures`: the renderer's own message is downstream of
        // this list and cannot say what the block held, which is the only thing that separates
        // the two causes.
        if frag_binds.is_empty() && fref.sampler_units != 0 {
            self.report_draw_without_bound_textures(blk, fheader, fref.sampler_units);
        }
        let texture_phase = crate::perf::scope(crate::perf::Phase::DrawTextures);
        let mut textures = match cached_set {
            Some(_) => Vec::new(),
            None => {
                let _m = crate::perf::scope(crate::perf::Phase::DrawTexFragMiss);
                self.snapshot_bound_textures(ctx, &frag_binds)
            }
        };
        self.bound_textures = frag_binds;
        // The VERTEX stage's own samplers, decoded exactly the same way. A vertex program that
        // fetches a texture builds its geometry from it - one retail title draws its whole
        // campaign map that way - so a draw that carries none of these renders as if the fetch
        // returned nothing, which is a blank screen rather than a visible error.
        //
        // >>> SHARED WITHIN A SCENE THE SAME WAY THE FRAGMENT LIST IS, and it was not.
        // The fragment stage has had `snapshot_sets` for a while; this side re-decoded its
        // units on EVERY draw, and a decode is where the texture-data snapshot and its compare
        // live. On a title that binds a vertex texture that is the whole remaining cost of the
        // phase. Same map, so the same invalidations (scene start, freed range, re-initialised
        // texture) apply without a second cache to keep honest. The key mixes a STAGE TAG
        // first, so a vertex list and a fragment list of identical bindings cannot collide in
        // the shared map - the fragment key starts from the program header, which is not a
        // separation a vertex list could be relied on to reproduce.
        //
        // >>> SHARED, NOT COPIED, AND USUALLY NOT EVEN LOOKED UP.
        //
        // This used to `to_vec` the shared entry into a fresh `Vec` per draw because
        // `capture::Draw` held one - "the tidier follow-up" the old note here named. It is an
        // `Arc<[_]>` on both sides now, so a hit is a refcount bump. And in front of the fold
        // and the probe sits the generation check: this title binds a vertex sampler on EVERY
        // draw, so the phase ran 626 times a frame to re-derive an answer that changes only
        // when the guest actually rebinds one.
        let vert_gen = self.vertex_texture_gen;
        let vert_binds = std::mem::take(&mut self.bound_vertex_textures);
        let vertex_textures: Arc<[crate::capture::BoundTexture]> = if vert_binds.is_empty() {
            // The common case for every other title: nothing bound decodes to nothing - the
            // SHARED empty list, because an empty `Arc` slice still allocates its counts.
            crate::capture::no_textures()
        } else if let Some(list) = snaps!(self).vertex_set_from_previous_draw(vert_gen) {
            crate::perf::note_hit(crate::perf::Phase::DrawTexSetPrev);
            list
        } else {
            let _v = crate::perf::scope(crate::perf::Phase::DrawTexVertex);
            let vkey = {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                let mut mix = |v: u64| {
                    h ^= v;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                };
                mix(VERTEX_STAGE_TAG); // "VERTEX" - the stage tag, mixed first
                for b in &vert_binds {
                    mix(b.unit as u64);
                    // The same omission as the fragment fold above, and for the same reason -
                    // both feed `set_validated`, which compares `addr`.
                    mix(b.addr as u64);
                    for w in b.words {
                        mix(w as u64);
                    }
                }
                h
            };
            // Bound first: the guard must be gone before the miss path below re-locks.
            let validated = snaps!(self).set_validated(ctx, vkey, VERTEX_STAGE_TAG, &vert_binds);
            let list = match validated {
                Some(set) => set,
                None => {
                    let decoded = self.snapshot_bound_textures(ctx, &vert_binds);
                    let set: Arc<[crate::capture::BoundTexture]> = decoded.as_slice().into();
                    snaps!(self).set_insert(vkey, VERTEX_STAGE_TAG, &vert_binds, set.clone());
                    set
                }
            };
            snaps!(self).remember_last_vertex_set(vert_gen, vkey, list.clone());
            list
        };
        self.bound_vertex_textures = vert_binds;
        drop(texture_phase);
        // The capture renderer samples a single texture (`textures.first()`). This title
        // binds the NORMAL map at unit 0, so pick the albedo sampler by fragment-program
        // reflection and move it to the front; without this every surface is tinted by a
        // normal map (flat blue/purple). Falls back to the unit-0 order when reflection
        // finds no albedo or that unit is not currently bound.
        // Failing name reflection, index 0 must still be a plausible SURFACE texture: a
        // one-dimensional lookup table (a fog ramp) or a cube map (an irradiance probe) can sort
        // ahead of the real albedo by unit number, and neither is indexed by surface UV. See
        // `Draw::albedo`, which drops a leading non-surface texture rather than stretch it.
        let textures: Arc<[crate::capture::BoundTexture]> = match cached_set {
            // Already the previous draw's answer, or one just re-proven for this scene -
            // either way the memo below is current and there is nothing to remember.
            Some(set) => set,
            None => {
                if let Some(pos) = textures
                    .iter()
                    .position(|t| t.height > 1 && t.faces <= 1)
                    .filter(|&p| p > 0)
                {
                    textures[..=pos].rotate_right(1);
                }
                if let Some(unit) = Self::fragment_albedo_unit(fref)
                    && let Some(pos) = textures.iter().position(|t| t.unit == unit)
                {
                    textures[..=pos].rotate_right(1);
                }
                let set: Arc<[crate::capture::BoundTexture]> = textures.into();
                // Taken and put back: `set_insert` needs the bindings that produced the
                // list, and they were returned to `self.bound_textures` above.
                let binds = std::mem::take(&mut self.bound_textures);
                snaps!(self).set_insert(set_key, fheader as u64, &binds, set.clone());
                self.bound_textures = binds;
                set
            }
        };
        // Remember it against the bytes it came from, so the NEXT draw of this batch can skip
        // the gate entirely. Not on a previous-draw hit: that entry is this one.
        if previous_set.is_none() {
            snaps!(self).remember_last_set(sampler_span, fheader, set_key, textures.clone());
        }
        (textures, vertex_textures)
    }
}