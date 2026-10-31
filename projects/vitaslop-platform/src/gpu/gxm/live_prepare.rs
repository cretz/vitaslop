//! [`GxpLive::prepare`]: the GPU resources for one recompiled draw.

use super::*;

impl GxpLive {
    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        // The pass renders into a single-channel ALPHA surface - see
        // `alpha_single_color_format`. Part of the pipeline cache key through `raster`.
        alpha_single: bool,
        // Samples per pixel of the pass this draw is being prepared for - see
        // `GxpPrepared::samples`.
        samples: u32,
        gxp: &GxpRecompile,
        depth_range: [f32; 2],
        // The pass's guest target size, for a draw whose viewport transform is DISABLED -
        // its clip position is then already in window pixels (see `inject_clip_fixup`).
        surf: (u32, u32),
        // The pass's ATTACHMENT extent in texels - where its fragments can land, which bounds
        // the window position a fragment program reads (see `push_mem_windows_per_draw`).
        att: (u32, u32),
        // `rendered`: views of the render targets this frame has already drawn, by
        // guest address. A sampler whose bound texture points at one of these binds
        // the render rather than the guest's (stale) bytes.
        rendered: &HashMap<u32, wgpu::TextureView>,
        // The LINEAR view of every GAMMA-CORRECT target in `rendered` (which holds its sRGB
        // view): what a sampler whose texture does NOT set a gamma mode binds - see the
        // `Bind::RenderedLinear` arm of `make_sampler_bg`.
        rendered_linear: &HashMap<u32, wgpu::TextureView>,
        // The sRGB view of every LINEAR target in `rendered` that has one: what a sampler
        // whose texture DOES set a gamma mode binds - see `Bind::RenderedSrgb`.
        rendered_srgb: &HashMap<u32, wgpu::TextureView>,
        // The same, for the DEPTH of those targets, keyed by the guest's depth-surface
        // address. Checked before `rendered` - see `make_sampler_bg`.
        depth_rendered: &HashMap<u32, wgpu::TextureView>,
        // The addresses in `depth_rendered` that a DEPTH-ONLY pass owns outright (a shadow
        // map, a depth prepass): there, the guest's depth address IS the target's key, so
        // the address legitimately appears in `rendered` too and the DEPTH answer is the
        // right one. Everywhere else an address in both maps is a registration bug - see
        // `report_depth_is_also_colour`.
        depth_only: &HashSet<u32>,
        // CUBE views of the cube maps whose six faces the guest RENDERED, by the address of
        // face 0. Separate from `rendered` because the two answer different questions: that
        // map holds one 2D target per address, and a cube sampler needs a six-layer view no
        // set of 2D targets can supply. See `GxmRenderer::rtt_cubes`.
        rendered_cubes: &HashMap<u32, wgpu::TextureView>,
        // The 64-bit targets this frame rendered as RAW words, by address. A fragment unit
        // whose bound texture is a 64-bit format at one of these addresses is linked RAW and
        // bound to that target's `Rg32Uint` view. See `GxmRenderer::rtt_rendered_raw`.
        rendered_raw: &HashMap<(u32, u32, u32), wgpu::TextureView>,
        // The addresses currently resolving to a SNAPSHOT, which key a sampler bind group
        // that names a target. (The `rtt_epoch` rode here too until the group's key became
        // the VIEW'S OWN IDENTITY - see the `Bind::Rendered` arm.)
        reads_snapshot: &HashSet<u32>,
        // The pass's grow-only vertex and index arenas, appended to rather than allocated
        // from per draw. See [`GxpPrepared`].
        vdata: &mut Vec<u8>,
        idata: &mut Vec<u8>,
        // The pass's uniform arena and the device's dynamic-offset alignment.
        udata: &mut Vec<u8>,
        ubo_align: u64,
        // The DESTINATION-colour texture for this pass and a generation that changes when
        // the texture is recreated (so the group-3 cache cannot hand back a bind group
        // naming a dead view). `None` when the pass has no draw that blends for itself,
        // in which case a pipeline that needs one is DROPPED rather than bound short - see
        // `report_missing_dest_texture`.
        dest: Option<(&wgpu::TextureView, u64)>,
    ) -> Option<GxpPrepared> {
        if gxp.index_count == 0 || gxp.vertices.is_empty() {
            // NOT a translation failure - there is nothing to draw. Recorded anyway,
            // because the caller tallies every `None` from here as a fallback draw and an
            // unnamed one reads as a recompiler defect.
            record_fallback_reason(
                // The memoised key - `key` hashes both whole blobs byte by byte.
                self.pair_key(gxp),
                "empty geometry: the draw carries no indices or no vertices, so there is \
                 nothing to render (not a recompiler failure)",
            );
            return None;
        }
        // Everything below is charged to `total_ns` however this function leaves - see
        // `SplitTotal`. Placed after the empty-geometry guard, which is not a draw.
        let _total = SplitTotal::start();
        // The preamble: the pair key, the pipeline-cache lookup and, on a miss, the pipeline
        // BUILD. The build is separately counted (`pipelines_built`), so a frame that reads
        // high here with a zero build count is paying the lookups, not the compiler.
        let t_key = split_start();
        // What the pipeline BUILD below took, so it can be taken back off the preamble it
        // happens inside. See `pipe_build_ns`.
        let mut build_ns: u64 = 0;
        let t_pair = split_start();
        let key = self.pair_key(gxp);
        split_end(t_pair, &PREP.key_pair_ns);
        if !self.keys.is_empty() && !self.keys.contains(&key) {
            record_fallback_reason(key, "filtered out by VITASLOP_GXP_KEYS (a knob, not a defect)");
            return None;
        }
        // `VITASLOP_GXP_EXCLUDE=<key>,...`: send these pairs down the fixed-function
        // path instead. The complement of `VITASLOP_GXP_KEYS`, and the useful one when
        // a frame is almost right: it answers "is this ONE draw the recompiler's fault"
        // without also un-recompiling the hundreds of draws that are already correct.
        if self.exclude.contains(&key) {
            record_fallback_reason(key, "excluded by VITASLOP_GXP_EXCLUDE (a knob, not a defect)");
            return None;
        }
        report_inputs(key, gxp);
        // Does this draw's colour work change anything at all? Asked here, once per draw,
        // because the answer is in the draw's own uniform bytes and nowhere else - see
        // `draw_colour_is_noop`. A `true` here does not drop the draw on its own: the
        // encoder still owes a depth write, and only the pair of facts decides.
        let t_fold = split_start();
        let colour_noop = draw_colour_is_noop(gxp);
        split_end(t_fold, &PREP.key_fold_ns);
        // The two halves of that fact are acted on in DIFFERENT PLACES, so they are named
        // separately: a no-op draw that writes no depth does nothing at all and the encoder
        // never issues it, while one that writes depth still owes that write and takes the
        // DEPTH-ONLY pipeline variant. Only the second is a pipeline, which is why the
        // elided-outright case must not build one.
        let noop_keeps_depth = colour_noop && gxp.depth_write;
        {
            // Once per pair per ANSWER, so a pair whose coefficients change mid-run reports
            // both - and a pair that is eligible and never folds is visible too, which is
            // what tells a "no saving" run from a "never asked" one.
            // The seen-test FIRST and on a thread-local: this block runs for EVERY draw, and
            // it used to take two global locks (the fold memo, then this set) per draw only
            // to find the report had already been made. The renderer runs on one thread.
            thread_local! {
                static SEEN: std::cell::RefCell<HashSet<(u64, bool)>> = std::cell::RefCell::new(HashSet::default());
            }
            if SEEN.with(|s| s.borrow_mut().insert((key, colour_noop)))
                && let Some(fp) = fragment_fold_program(&gxp.fprog)
            {
                // The TERMS, not just the verdict. `out = G + dst * F`, so `G=0 F=1` is
                // the identity and anything else says exactly which channel of which term
                // stopped it - a `?` being a channel the fold could not pin down at all,
                // which is a gap in the fold rather than a blend that paints.
                let windows: Vec<(u32, &[u8])> =
                    gxp.frag_mem_windows.iter().map(|(a, b)| (*a, &b[..])).collect();
                let terms = vitaslop_gxp_shader::fold::fragment_colour_terms_prepared(
                    &fp,
                    &vitaslop_gxp_shader::fold::DrawUniforms { frag_sa: &gxp.frag_sa, windows: &windows },
                    &mut None,
                );
                let show = |v: [Option<f32>; 4]| {
                    v.iter()
                        .map(|c| c.map_or_else(|| "?".to_string(), |x| format!("{x}")))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                report_status!(
                    "gxp colour no-op: pair {key:016x} {} - out = G + dst*F with G=[{}] F=[{}]                          (a `?` is a channel this draw's bytes do not pin down);                          depth_write={} depth_func={:#x}; fmem windows {:?}.                          A pair reported BOTH ways is one the guest reconfigures mid-run.",
                    if colour_noop { "WRITES THE DESTINATION BACK UNCHANGED" } else { "paints" },
                    terms.map_or("-".into(), |(g, _)| show(g)),
                    terms.map_or("-".into(), |(_, f)| show(f)),
                    gxp.depth_write,
                    gxp.depth_func,
                    gxp.frag_mem_windows
                        .iter()
                        .map(|(a, b)| format!("{a:#x}: {:02x?}", &b[..b.len().min(80)]))
                        .collect::<Vec<_>>()
                );
            }
        }
        report_inputs_order(key, gxp);
        report_quads(key, gxp);
        // The fragment units bound to a 64-bit texture at an address this frame rendered
        // as RAW words. Those units are linked as `texture_2d<u32>` loads and bound to the
        // raw target, so the mask is part of the pipeline's identity - the same pair over
        // an ordinary texture is another pipeline. Per DRAW, because a title binds
        // different textures to one program across a pass.
        let PipeKey { raw_units, f32_units, raster, cache_key } = Self::pipe_key(
            gxp,
            key,
            noop_keeps_depth,
            color_format,
            alpha_single,
            samples,
            |k| rendered_raw.contains_key(k),
        );
        if !self.pipelines.contains_key(&cache_key) {
            // Name the pair's two containers by their CONTENT hash the moment it is first
            // seen. `Program::hash` is the same value the offline corpus computes, so this
            // one line is what turns a draw key from the frame into the two `.gxp` blobs an
            // offline test can open. Printed once per unique pair, not per draw.
            //
            // >>> THIS IS AN INDEX, NOT A WARNING, AND IT MUST BE ASKED FOR.
            //
            // It was promoted to `warn` with no knob because at `debug` it printed nothing
            // in the runs that needed it - which was a real problem and the wrong fix. "A
            // few dozen lines a run" is true on a menu and false on a title screen deep in
            // a game: MEASURED on the user's device, **~90 unique pairs filled the browser
            // diagnostics panel, which keeps 96 DISTINCT lines, and the capture came back
            // reading `21 earlier DISTINCT line(s) dropped`.** The dropped lines are the
            // ones a capture is taken FOR. An index that evicts the findings is worse than
            // an index nobody can see.
            //
            // So it is gated on actually wanting it - either explicitly, or implicitly by
            // setting any diagnostic that takes a pair KEY, which is the only situation in
            // which "what keys exist" is a question. That keeps the property the promotion
            // was after (a `VITASLOP_GXP_INPUTS=<key>` run that prints nothing can still be
            // told apart from the pair never being drawn) and costs a default run nothing.
            // [[vitaslop-a-diagnostic-can-bury-the-findings]]
            if gxp_key_index_wanted() {
                let ph = |b: &[u8]| {
                    vitaslop_gxp_shader::Program::parse(b).map(|p| p.hash).unwrap_or(0)
                };
                report_status!(
                    "gxp pair {key:x}: vprog hash {:016x}, fprog hash {:016x}",
                    ph(&gxp.vprog),
                    ph(&gxp.fprog)
                );
            }
            // ...and whether either half reads uniforms we cannot supply. Here, at the
            // once-per-pair site, so it names the program rather than the API call.
            report_unfed_uniforms(key, "vertex", &gxp.vprog);
            report_unfed_uniforms(key, "fragment", &gxp.fprog);
            enc(&ENC.pipelines_built, 1);
            log_pipe("sync", &cache_key, gxp);
            if self.warm_used {
                enc(&ENC.pipelines_unwarmed, 1);
            }
            PIPELINES_BUILT_THIS_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // A variant of a pair already built under another baked state: cheap to
            // count here (builds are rare), and the number that says whether creating
            // the common variants ahead of the draw would remove a mid-lap build.
            if self.pipelines.keys().any(|k| k.0 == key) {
                enc(&ENC.pipelines_built_variant, 1);
            }
            // Charged to `pipe_build_ns`, NOT to the key preamble it sits in the middle
            // of - see the counter.
            let t_build = split_start();
            let built = build_gxp_pipeline(device, color_format, alpha_single, samples, gxp.cull_mode, gxp, key, self.zfix, self.yflip, self.solid, self.nodepth, self.noblend, noop_keeps_depth, raw_units, f32_units, &mut self.modules, |d, desc| d.create_render_pipeline(desc));
            self.pipelines.insert(cache_key, built);
            if let Some(t) = t_build {
                build_ns = (t.ms() * 1.0e6) as u64;
                enc(&PREP.pipe_build_ns, build_ns);
            }
        }

        // Split the borrows: the sampler bind group needs the texture-view cache mutably
        // while the pipeline (its layouts, its sampler plan) stays borrowed.
        let negw_mode = self.negw;
        // Which clip-depth remap `inject_clip_fixup` put the vertex stage through. A
        // fragment that WRITES its own depth has to invert exactly that map, and only the
        // renderer knows which one is in force, so it travels in the depth block.
        let zfix_mode: f32 = match self.zfix {
            ZFix::Range => 0.0,
            ZFix::Gl => 1.0,
            // Both leave the guest's window depth in the buffer, so a fragment writing its
            // own depth inverts the same (identity) map.
            ZFix::Off | ZFix::Clamp => 2.0,
            // The buffer holds the guest's own WINDOW depth - the viewport map is already
            // applied - so a fragment writing one in that same encoding needs only the
            // clamp, which is the one part of the forward map it cannot have applied.
            ZFix::Viewport => 3.0,
        };
        // >>> THE PASS'S VERDICT, APPLIED ONLY TO THE DRAWS IT WAS TAKEN OVER.
        //
        // `decide_scene_negw` deliberately EXCLUDES draws with no projection from the vote,
        // on the ground that a constant `w` is what a screen-space quad writes and says
        // nothing about the perspective matrix's sign convention. The correction was then
        // applied to those very draws anyway, which negates a `w` that was never in that
        // convention and puts the quad behind the camera.
        //
        // MEASURED on the retail golf title's opening: its 960x544 pass is judged NEGATIVE
        // by 241 perspective draws, and the two screen-space quads it also carries are the
        // BACKGROUND and THE MOVIE. Both were clipped away, so the opening movie decoded
        // correctly into guest memory, was bound as a texture every frame, was drawn every
        // frame - and contributed no pixels: the frame was byte-identical to a run with the
        // movie switched off entirely (`VITASLOP_MP4_UNITS=none`), and the title looked hung
        // for the whole length of its intro. `VITASLOP_GXP_NEGW=off` renders it.
        //
        // So a pair takes the correction when it carries a PROJECTION (it shares the
        // convention), or when its own sampled vertices are all on the negative side (where
        // it is clipped away regardless and the correction can only help). A screen-space
        // draw sitting on the positive side is left exactly as the shader wrote it.
        // Never for a window-space draw (viewport disabled): it has no projection to correct.
        let scene_negw = self.scene_negw
            && self.pair_takes_negw(key)
            && gxp.viewport_enable != crate::gpu::SCE_GXM_VIEWPORT_DISABLED;
        let scene_depth_fit = self.scene_depth_fit;
        // >>> ASKED ONCE PER RENDERER, NEVER PER DRAW.
        //
        // `Device::features()` is a cheap bitflag read on a native backend and a JS property
        // access plus a walk of the whole `GPUSupportedFeatures` set in the browser, which is
        // the boundary crossing [[vitaslop-browser-host-call-cost]] measured at 91% of a host
        // call's cost. This is on the per-draw path, and a title submits hundreds of draws a
        // frame, so reading it there would have paid that hundreds of times a frame to
        // re-learn a constant. Cached per GxpLive rather than in a global, because "there is
        // only ever one device" is true today and is not a thing this code should assert.
        let bc = *self.bc_supported.get_or_insert_with(|| {
            let f = device.features();
            if f.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
                BlockFamily::Bc
            } else if f.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2) {
                BlockFamily::Etc2
            } else {
                BlockFamily::None
            }
        });
        // >>> THE TRANSCODER IS BUILT AT RENDERER CONSTRUCTION - see `GxpLive::texenc`.
        //
        // It used to be built HERE, on the first recompiled draw of the run, which put a
        // generic engine shader's compile inside a gameplay frame. A fallback build is kept
        // for a `GxpLive` that reached a draw without one (a renderer constructed with the
        // recompiler off and enabled later), because a missing transcoder is a dropped
        // texture, not a slow one.
        if self.texenc.is_none() {
            self.texenc = Some(crate::texenc::Transcoder::new(device));
        }
        // >>> THE FRAME'S TEXTURE EXPANSIONS GO TO THE QUEUE AT EVERY PASS BOUNDARY, AND
        // >>> THAT IS A DEVICE MEASUREMENT, NOT A TIDINESS CHOICE.
        //
        // Batching them across the WHOLE frame submits fewer command buffers, and a desktop
        // says that is better. The user's phone said otherwise the first time it ran: play
        // "hiccups a tad more". The mechanism fits a tile-based GPU exactly - held to the
        // end of `encode_chain`, every expansion's COMPUTE lands in one submit immediately
        // before the render pass that samples those textures, so the render cannot start
        // until the compute drains. Flushed per pass, that work is already in flight while
        // the CPU builds the rest of the frame.
        //
        // So this is a count win that a desktop cannot price
        // [[vitaslop-desktop-cannot-price-a-count-win]], and the count is not the thing that
        // matters. Batching still happens - within a pass, which is where consecutive
        // uploads actually occur - and a load frame still submits far fewer times than the
        // one-per-texture this replaced. Widening it back to the frame needs a DEVICE
        // measurement that says so.
        //
        // No report here: a batch pending at a PASS boundary is the batching working. The
        // watchdog for a genuinely missed flush is at the top of `encode_chain`, where a
        // pending batch means a whole frame went by without one.
        if let Some(t) = self.texenc.as_ref() {
            t.flush_raw(queue);
        }
        let GxpLive {
            pipelines,
            texenc,
            views: view_cache,
            views_bytes: view_cache_bytes,
            views_used,
            view_slots,
            view_dead,
            views_evicted,
            views_epoch,
            views_frame_high,
            views_frame_bytes,
            samplers_by_mode,
            force,
            depth_bgs,
            depth_retired,
            store_mirrors,
            packed,
            packed_by_alloc,
            resident_v,
            resident_i,
            resident_i_seen,
            resident_v_seen,
            resident,
            sampler_bgs,
            last_sampler_bg,
            sampler_pre,
            mem_push_memo,
            mem_push_frame,
            raw_fill,
            udata_holes,
            ..
        } = self;
        let resident = *resident;
        // Borrow the cached pipeline. >>> THE TWO `None`s HERE ARE DIFFERENT FAULTS and
        // were being conflated by one `?` chain: a MISSING entry means this draw's state
        // combination was never built (the build is attempted above, so reaching here with
        // no entry means that attempt did not run), while a PRESENT-BUT-EMPTY entry is a
        // pair whose build was tried and FAILED and is remembered as poisoned. The first
        // points at the cache-key path, the second at the shader - unrelated fixes.
        let pipe = match pipelines.get(&cache_key) {
            Some(Some(p)) => p,
            Some(None) => {
                record_fallback_reason(
                    key,
                    "the pipeline for this pair is POISONED - an earlier build of it failed \
                     and the failure is cached",
                );
                return None;
            }
            None => {
                record_fallback_reason(
                    key,
                    "no pipeline cache entry for this draw's state combination",
                );
                return None;
            }
        };
        split_end_less(t_key, &PREP.key_ns, build_ns);

        if gxp_dump() {
            let f: Vec<f32> = gxp
                .vert_sa
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let attrs: Vec<(u16, u16, u8, u8)> =
                gxp.attributes.iter().map(|a| (a.reg_index, a.offset, a.gxm_format, a.components)).collect();
            let ff: Vec<f32> = gxp
                .frag_sa
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            // The guest stream's own extent next to what the indices actually reference: a
            // mesh whose highest index is beyond the captured vertices renders only the
            // triangles that fall inside the buffer, i.e. a PREFIX of the geometry.
            let nverts = gxp.vertices.len() / (gxp.vertex_stride.max(1) as usize);
            let max_index = if gxp.index_u32 {
                gxp.indices.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).max()
            } else {
                gxp.indices.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).max()
            }
            .unwrap_or(0);
            report!(
                "gxp draw key {:x}: vsa_lanes={} vert_sa_lanes={} fsa_lanes={} frag_sa_lanes={} samplers={:?} stride={} idx={} nverts={nverts} max_index={max_index} vbytes={} attrs(reg,off,fmt,comp)={:?}\n  vsa={:?}\n  fsa={:?}",
                key, pipe.vsa_lanes, f.len(), pipe.fsa_lanes, ff.len(), pipe.samplers, gxp.vertex_stride, gxp.index_count, gxp.vertices.len(), attrs, f, ff
            );
            // For a small mesh, the attribute VALUES as fetched. A quad landing in the
            // wrong place on screen is either its positions or its transform, and those
            // are indistinguishable from the picture; this shows the positions.
            if nverts <= 8 {
                let stride = gxp.vertex_stride.max(1) as usize;
                for v in 0..nverts {
                    let vals: Vec<String> = gxp
                        .attributes
                        .iter()
                        .map(|a| {
                            let base = v * stride + a.offset as usize;
                            let c: Vec<String> = (0..a.components as usize)
                                .map(|i| format!("{:.3}", read_attr_component(&gxp.vertices, base, a.gxm_format, i)))
                                .collect();
                            format!("r{}=[{}]", a.reg_index, c.join(","))
                        })
                        .collect();
                    report!("    v{v}: {}", vals.join(" "));
                }
            }
        }

        // Repack the guest vertex stream into the tightly-packed f32 layout the pipeline
        // expects (per the cached repack plan), straight into the pass's vertex arena.
        // Same vertex count/order, so the index buffer is unchanged.
        //
        // Both arenas stay 4-byte aligned, which `set_vertex_buffer` requires and a u32
        // index buffer requires: the packed stride is a whole number of f32s and an index
        // is four bytes, so the padding below never actually fires - it is there so a
        // future format that is not cannot silently produce an unaligned slice.
        // The allocation this stream lives in, which is the fast key - see
        // `packed_by_alloc`. Both halves of the fat pointer, so a shorter stream that
        // happens to start at the same address is a different key rather than a truncation.
        // The whole geometry path, with hash/repack/arena nested inside it - see `vpath_ns`.
        let t_vpath = split_start();
        let akey = (key, std::sync::Arc::as_ptr(&gxp.vertices) as *const u8 as usize, gxp.vertices.len());
        // `get_mut`, so a hit STAMPS the entry: this is the fast path, so it is also where
        // most of a frame's uses are, and an entry the fast path serves every frame must not
        // look untouched to the eviction below.
        let alloc_hit = (!pipe.passthrough)
            .then(|| {
                packed_by_alloc
                    .get_mut(&akey)
                    .filter(|(src, ..)| std::sync::Arc::ptr_eq(src, &gxp.vertices))
                    .map(|(_, bytes, used)| {
                        *used = *views_epoch;
                        bytes.clone()
                    })
            })
            .flatten();
        // The packed bytes for this draw, however they were reached.
        //
        // >>> THE PIPELINE MAY HAVE NO REPACKING TO DO AT ALL, and then this is the guest's
        // >>> own stream `Arc`, untouched.
        //
        // Not a fast path through the cache - a path that does not reach the cache. The
        // hash, the byte-compare, the repack and BOTH packed maps exist to turn a guest row
        // into a differently-shaped one; when the pipeline binds the guest row directly
        // (`passthrough`) there is nothing to turn, nothing to remember and nothing to
        // evict. The residency map below keys on this `Arc` exactly as it keyed on the
        // packed one, and with the same meaning: a stream rebuilt into a fresh allocation
        // every frame gets a fresh `Arc` and never promotes, while a static mesh's `Arc` is
        // stable and does.
        let packed_bytes: std::sync::Arc<[u8]> = if pipe.passthrough {
            split_add(&PREP.packed_passthrough, 1);
            gxp.vertices.clone()
        } else {
            match alloc_hit {
                Some(bytes) => {
                    split_add(&PREP.packed_hits, 1);
                    bytes
                }
                None => Self::pack_vertices(
                    packed,
                    packed_by_alloc,
                    akey,
                    gxp,
                    &pipe.repack,
                    pipe.packed_stride,
                    *views_epoch,
                ),
            }
        };
        // >>> THE RESIDENCY KEY IS THE PACKED CONTENT `Arc`, AND THE HISTORY HERE HAS TWO
        // >>> TURNS - read both before changing it again.
        //
        // The ALLOCATION key served **1 vertex draw in 492** on a retail sports title's
        // front end, because that title writes its geometry into a rotating per-frame
        // arena: the allocation is new every frame while `pack_vertices`' content hash
        // says three quarters of the streams are byte-identical to one the renderer
        // already had (`366 hits vs 125 misses`). Keying on the packed `Arc` (2026-08-19)
        // measured resident vertex draws 1 -> 275 and arena copy 0.97 -> 0.39 ms - and was
        // REVERTED, because an animated UI repeats for exactly two frames, gets promoted,
        // and is then never drawn again: the heap filled its 48 MB with 6,860 meshes and
        // WHOLESALE-RESET every 100-300 frames, re-uploading everything it held.
        //
        // What un-reverted it (2026-08-28c): the heap now COMPACTS instead of resetting -
        // a full heap keeps the slices bound in the last two frames and drops the dead
        // ones in one GPU copy pass (see `ResidentHeap::grow_or_reset`), so the promoted-
        // then-abandoned geometry that killed the first attempt is exactly what a
        // compaction sheds. The promotion test stays "second CONTENT sighting"
        // (`resident_v_seen`, a `Weak` so it pins nothing) - a stream built fresh every
        // frame has a fresh packed `Arc` and never reaches the heap.
        //
        // The key folds `key` (the pipeline) because the packed LAYOUT is the pipeline's
        // repack plan; the same guest stream drawn by two pairs packs differently.
        // Where the draw's vertices will be. A stream the renderer has seen before is placed
        // in the RESIDENT heap and never copied again; a first sighting, and anything the
        // heap declines, goes through the pass arena exactly as it always did.
        let rkey =
            (key, std::sync::Arc::as_ptr(&packed_bytes) as *const u8 as usize, packed_bytes.len());
        let (v_off, v_len, v_resident) = match resident
            .then(|| resident_v.get(&rkey, &packed_bytes))
            .flatten()
        {
            Some((off, len)) => {
                split_add(&PREP.resident_v_hits, 1);
                (off, len, true)
            }
            None => {
                let t = split_start();
                let off = vdata.len() as u64;
                vdata.extend_from_slice(&packed_bytes);
                let len = vdata.len() as u64 - off;
                while !vdata.len().is_multiple_of(4) {
                    vdata.push(0);
                }
                split_end(t, &PREP.arena_ns);
                split_add(&PREP.arena_bytes, len);
                // ...and WHOSE bytes they are. Only here, in the branch that actually
                // writes: a resident draw uploads nothing and counting it would price an
                // expansion nobody pays for. See `arena_bytes_expand_excess`.
                if pipe.passthrough {
                    split_add(&PREP.arena_bytes_passthrough, len);
                } else {
                    split_add(&PREP.arena_bytes_repacked, len);
                    let widened = pipe.packed_stride.saturating_sub(pipe.packed_stride_native);
                    if widened > 0 {
                        let nverts = len / pipe.packed_stride.max(1) as u64;
                        split_add(&PREP.arena_bytes_expand_excess, nverts * widened as u64);
                    }
                }
                // Promote on the SECOND sighting, not the first - see the note above.
                if resident {
                    let v_seen = resident_v_seen
                        .get(&rkey)
                        .and_then(|(prev, _)| prev.upgrade())
                        .is_some_and(|prev| std::sync::Arc::ptr_eq(&prev, &packed_bytes));
                    if v_seen {
                        resident_v.place(queue, rkey, &packed_bytes, &packed_bytes);
                    } else {
                        if resident_v_seen.len() >= RESIDENT_SEEN_CAP {
                            prune_seen(resident_v_seen, *views_epoch, "vertex promotion map");
                        }
                        resident_v_seen.insert(
                            rkey,
                            (std::sync::Arc::downgrade(&packed_bytes), *views_epoch),
                        );
                    }
                }
                (off, len, false)
            }
        };
        // The indices, on exactly the same terms. They have their own allocation - the
        // capture expands every primitive type into a u32 triangle list and caches THAT by
        // the guest buffer it came from - so a mesh can hold its index list still while its
        // vertices move, and the two are placed independently.
        //
        // The key has no pipeline in it: an expanded index list is a function of the guest
        // buffer alone, so two pairs drawing the same mesh share one resident copy.
        let ikey = (0u64, std::sync::Arc::as_ptr(&gxp.indices) as *const u8 as usize, gxp.indices.len());
        let (i_off, i_len, i_resident) = match resident
            .then(|| resident_i.get(&ikey, &gxp.indices))
            .flatten()
        {
            Some((off, len)) => {
                split_add(&PREP.resident_i_hits, 1);
                (off, len, true)
            }
            None => {
                let t = split_start();
                let off = idata.len() as u64;
                idata.extend_from_slice(&gxp.indices);
                let len = idata.len() as u64 - off;
                while !idata.len().is_multiple_of(4) {
                    idata.push(0);
                }
                split_end(t, &PREP.arena_ns);
                split_add(&PREP.arena_bytes, len);
                split_add(&PREP.arena_bytes_index, len);
                if resident {
                    // >>> ASKED HERE, NOT ABOVE, BECAUSE ONLY THIS BRANCH READS IT.
                    // The promotion test is a map probe plus a `Weak` upgrade and it was
                    // computed for EVERY draw while only the resident MISS path uses it -
                    // and a miss is the rare case: a race on the device binds 859 of its
                    // 952 index draws in place, so nine draws in ten paid for an answer
                    // that was discarded. The vertex half above already asks inside its
                    // own miss branch; this makes the two match.
                    let i_seen = resident_i_seen
                        .get(&ikey)
                        .and_then(|(prev, _)| prev.upgrade())
                        .is_some_and(|prev| std::sync::Arc::ptr_eq(&prev, &gxp.indices));
                    if i_seen {
                        resident_i.place(queue, ikey, &gxp.indices, &gxp.indices);
                    } else {
                        if resident_i_seen.len() >= RESIDENT_SEEN_CAP {
                            prune_seen(resident_i_seen, *views_epoch, "index promotion map");
                        }
                        resident_i_seen.insert(
                            ikey,
                            (std::sync::Arc::downgrade(&gxp.indices), *views_epoch),
                        );
                    }
                }
                (off, len, false)
            }
        };

        split_end(t_vpath, &PREP.vpath_ns);

        // The two SA blocks go into the pass's uniform ARENA at dynamic-offset alignment;
        // the bind groups over that arena belong to the shader pair and are built once,
        // after the arena buffer exists (see `ensure_ubo_bgs`).
        let t_uni = split_start();
        let vert_sa = override_sa(key, 'v', &gxp.vert_sa);
        let frag_sa = override_sa(key, 'f', &gxp.frag_sa);
        // The vertex stage's guest-memory windows, when this pair's program loads
        // memory: one header vec4 per window (lane x = its guest base address) + every
        // window's bytes, in the same dynamic-offset arena as the SA blocks. A draw that
        // arrives WITHOUT the bytes its pipeline needs is dropped with a report - feeding
        // the loads zeroes would render a wrong picture with nothing to say so.
        let mem_off = if pipe.mem_bind_bytes > 0 {
            if gxp.mem_windows.len() != pipe.mem_windows.len() {
                // >>> REPORTED WITH A RUNNING COUNT, because reporting it ONCE is how a
                // whole title's world went missing quietly. One line saying a draw was
                // dropped reads like ONE draw; what matters is how many, and on the title
                // that found this it was every stage, crowd and character draw of every
                // frame, while the HUD - whose programs load no memory - rendered on top of
                // the black. [[vitaslop-a-drop-count-needs-its-draw-count]]
                static DROPPED: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let n = DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n == 1 || n.is_multiple_of(10_000) {
                    tracing::warn!(
                        target: "vitaslop::gpu",
                        key = format_args!("{key:016x}"),
                        want = pipe.mem_windows.len(),
                        got = gxp.mem_windows.len(),
                        dropped_so_far = n,
                        "a pipeline with guest-memory windows got a draw without their \
                         bytes - draw DROPPED. The capture and the link disagree about \
                         whether this program loads memory; feeding the loads zeroes would \
                         render a wrong picture with nothing to say so."
                    );
                }
                record_fallback_reason(
                    key,
                    "the pipeline expects VERTEX guest-memory windows and this draw carried \
                     none - the capture and the link disagree about whether the program \
                     loads memory",
                );
                return None;
            }
            let t_mem = split_start();
            let off = push_mem_windows_memo(
                mem_push_memo,
                mem_push_frame,
                (pipe as *const _ as usize, 0),
                udata,
                pipe.mem_bind_bytes,
                &pipe.mem_windows,
                &gxp.mem_windows,
                mem_window_align(device, ubo_align, &pipe.mem_windows),
            );
            split_end(t_mem, &PREP.uni_mem_ns);
            off
        } else {
            0
        };
        // The FRAGMENT stage's windows, the mirror of the block above.
        let mut fmem_ranges = Vec::new();
        // This draw's own window-block size, for a PER-DRAW set (0 = the pipeline's fixed one).
        let mut fmem_bytes = 0u32;
        let fmem_per_draw = vitaslop_gxp_shader::module::mem_window_per_draw(&pipe.fmem_windows);
        let fmem_off = if pipe.fmem_bind_bytes > 0 && fmem_per_draw {
            if gxp.frag_mem_windows.len() != pipe.fmem_windows.len() {
                record_fallback_reason(
                    key,
                    "the pipeline expects FRAGMENT guest-memory windows and this draw \
                     carried none",
                );
                return None;
            }
            let t_mem = split_start();
            // >>> SIZED FROM THIS DRAW'S OWN INPUTS: its SA image, its windows' addresses and the
            // extent its fragments can land in - the attachment, which bounds the window
            // position the program reads. See `vitaslop_gxp_shader::reach`.
            let addrs: Vec<u32> = gxp.frag_mem_windows.iter().map(|(a, _)| *a).collect();
            let extents = vitaslop_gxp_shader::reach::per_draw_extents(&gxp.fprog, &pipe.fmem_windows, &frag_sa, &addrs, att);
            let Some((off, bytes, ranges)) = push_mem_windows_per_draw(
                udata,
                &gxp.frag_mem_windows,
                &extents,
                mem_window_align(device, ubo_align, &pipe.fmem_windows),
                device.limits().max_storage_buffer_binding_size.min(u64::from(u32::MAX)),
                raw_fill,
                udata_holes,
            ) else {
                record_fallback_reason(
                    key,
                    "this draw's per-draw guest-memory windows exceed the device's storage binding limit",
                );
                return None;
            };
            split_add(&PREP.arena_bytes, u64::from(bytes));
            split_add(&PREP.arena_bytes_memwin, u64::from(bytes));
            report_per_draw_windows(key, &extents, bytes);
            if !rendered_raw.is_empty() && crate::gpu::mem_window_from_target() {
                fmem_ranges = ranges;
            }
            fmem_bytes = bytes;
            split_end(t_mem, &PREP.uni_mem_ns);
            off
        } else if pipe.fmem_bind_bytes > 0 {
            if gxp.frag_mem_windows.len() != pipe.fmem_windows.len() {
                static DROPPED: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let n = DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n == 1 || n.is_multiple_of(10_000) {
                    tracing::warn!(
                        target: "vitaslop::gpu",
                        key = format_args!("{key:016x}"),
                        want = pipe.fmem_windows.len(),
                        got = gxp.frag_mem_windows.len(),
                        dropped_so_far = n,
                        "a pipeline with FRAGMENT guest-memory windows got a draw without their bytes - draw DROPPED. See the vertex counterpart above."
                    );
                }
                record_fallback_reason(
                    key,
                    "the pipeline expects FRAGMENT guest-memory windows and this draw \
                     carried none",
                );
                return None;
            }
            let t_mem = split_start();
            // `VITASLOP_GXP_FMEM` substitutes lanes here, on the way to the GPU, so the
            // knob reaches every draw of a live frame the same way it reaches one capsule.
            let subst = substituted_fmem(&gxp.frag_mem_windows);
            let off = push_mem_windows_memo(
                mem_push_memo,
                mem_push_frame,
                (pipe as *const _ as usize, 1),
                udata,
                pipe.fmem_bind_bytes,
                &pipe.fmem_windows,
                subst.as_deref().unwrap_or(&gxp.frag_mem_windows),
                mem_window_align(device, ubo_align, &pipe.fmem_windows),
            );
            // Where each window's bytes landed in the arena, for the encoder to overwrite
            // from a raw target this frame rendered over the same memory - asked only when
            // there is such a target at all. See `fill_windows_from_raw_targets`.
            if !rendered_raw.is_empty() && crate::gpu::mem_window_from_target() {
                fmem_ranges = gxp
                    .frag_mem_windows
                    .iter()
                    .zip(vitaslop_gxp_shader::module::mem_window_placements(&pipe.fmem_windows))
                    // The window's extent in the BINDING, which is what the shader can read.
                    .map(|((addr, _), place)| {
                        (*addr, place.words * 4, u64::from(off) + u64::from(place.first_word) * 4)
                    })
                    .collect();
            }
            split_end(t_mem, &PREP.uni_mem_ns);
            off
        } else {
            0
        };
        // The SA half, timed apart from the memory-window half above - see `uni_sa_ns`.
        let t_sa = split_start();
        let sa_before = udata.len() as u64;
        let u_off = [
            push_sa(udata, pipe.vsa_lanes, &vert_sa, ubo_align),
            push_sa(udata, pipe.fsa_lanes, &frag_sa, ubo_align),
            mem_off,
            fmem_off,
        ];
        split_end(t_sa, &PREP.uni_sa_ns);
        split_add(&PREP.uni_sa_bytes, udata.len() as u64 - sa_before);
        if pipe.vsa_lanes == 0 {
            split_add(&PREP.uni_sa_empty, 1);
        }
        if pipe.fsa_lanes == 0 {
            split_add(&PREP.uni_sa_empty, 1);
        }
        split_end(t_uni, &PREP.uni_ns);
        let t_samp = split_start();
        let bg2 = Self::make_sampler_bg(
            device, queue, &pipe.layouts[2],
            &[
                (&pipe.samplers[..], &gxp.textures[..]),
                (&pipe.vertex_samplers[..], &gxp.vertex_textures[..]),
            ],
            key,
            view_cache, view_slots, view_dead, view_cache_bytes, views_used, views_evicted, views_epoch,
            views_frame_high, views_frame_bytes, samplers_by_mode, *force,
            rendered, rendered_linear, rendered_srgb, depth_rendered, depth_only, rendered_cubes, rendered_raw, reads_snapshot,
            sampler_bgs, last_sampler_bg, sampler_pre, bc,
            texenc.as_ref().expect("built at the top of this function"),
        );
        split_end(t_samp, &PREP.sampler_ns);
        // >>> THE ONE EXIT THAT SAID NOTHING AT ALL. `make_sampler_bg` returns `None` when
        // a texture this draw samples cannot be bound (an unresolved view, a format with no
        // wgpu equivalent, an evicted entry that could not be rebuilt), and a bare `?` here
        // dropped the draw with no report, no counter and no recorded reason.
        let Some(bg2) = bg2 else {
            record_fallback_reason(
                key,
                "the sampler bind group could not be built - a texture this draw samples \
                 could not be bound",
            );
            return None;
        };
        // A FRAGMENT program that STORES: the mirror of guest memory its stores write, and one
        // header per store window (guest base, words, mirror word) - see `store_mirror`. A draw
        // that arrives without the windows' snapshots is DROPPED by name: binding a mirror
        // seeded from nothing would hand the stencil load bytes the guest never had.
        let store = if pipe.store_windows.is_empty() {
            None
        } else {
            if gxp.frag_store_windows.len() != pipe.store_windows.len() {
                record_fallback_reason(
                    key,
                    "the pipeline's fragment program STORES to guest memory and this draw carried no \
                     snapshot of the buffers it stores through (a capture taken before the store \
                     windows were captured)",
                );
                return None;
            }
            let m = match store_mirrors.ensure(device, queue, &gxp.frag_store_windows) {
                Ok(i) => &store_mirrors.mirrors[i],
                Err(why) => {
                    record_fallback_reason(key, why);
                    return None;
                }
            };
            let header: Vec<u32> = gxp
                .frag_store_windows
                .iter()
                .flat_map(|(addr, bytes)| [*addr, bytes.len() as u32 / 4, (addr - m.lo) / 4, 0])
                .collect();
            Some((m.buf.clone(), m.id, header))
        };
        let t_depth = split_start();
        // group3: the scene depth range the injected clip fixup maps through, as one vec4
        // (min, scale, unused, unused) - the same values the fixed-function path uses, so
        // both kinds of draw write comparable depth. Per SCENE, not per draw, so it is
        // cached (see `GxpLive::depth_bgs`); the bit pattern is the key so a changed
        // range builds a new one rather than reusing a stale buffer.
        //
        // Its third lane is the clip-`w` sign correction, which is a property of the DRAW
        // (its projection), not of the scene - so it joins the cache key.
        let corrected = scene_negw;
        let sign: f32 = match (corrected, negw_mode) {
            (false, _) => 1.0,
            (true, NegW::Negate) => -1.0,
            (true, _) => 2.0,
        };
        // Lanes 4 and 5 are this pass's guest depth encoding `a + c/w`, and they are the
        // same numbers the depth-conversion pass writes with: a fragment reading its own
        // POSITION.z and a fragment sampling a converted depth surface must be looking at
        // one quantity, or every soft fade between them compares apples to oranges.
        let (fit_a, fit_c) = scene_depth_fit;
        // Lanes 6 and 7 are the guest's OWN viewport depth mapping (`zScale`, `zOffset`),
        // which `ZFix::Viewport` puts the clip depth through. It is per DRAW - the guest can
        // and does set a different viewport for an overlay than for the world - so like the
        // sign correction above it joins the cache key.
        let (z_scale, z_offset) = gxm_viewport_depth(&gxp.viewport, key);
        // The guest viewport's VERTICAL SENSE, which the clip fixup applies because a wgpu
        // viewport cannot - see `inject_clip_fixup` for the title that needs it and what it
        // cost. Per DRAW, like the depth mapping above, so it joins the cache key.
        let y_sense: f32 = if gxp.viewport[3] > 0.0 { -1.0 } else { 1.0 };
        // A draw with the viewport transform SWITCHED OFF writes its position in window
        // pixels, not in clip space: lane 9 says so and lanes 10 and 11 carry `2/width` and
        // `2/height` of the target, for `inject_clip_fixup` to map pixels onto clip space.
        let window: [f32; 3] = if gxp.viewport_enable == crate::gpu::SCE_GXM_VIEWPORT_DISABLED && surf.0 > 0 && surf.1 > 0 {
            report_window_space_pair(key, surf);
            [1.0, 2.0 / surf.0 as f32, 2.0 / surf.1 as f32]
        } else {
            [0.0; 3]
        };
        let depth_key = (depth_range[0].to_bits() as u64) << 32 | depth_range[1].to_bits() as u64;
        let depth_key = (depth_key, z_scale.to_bits(), z_offset.to_bits(), y_sense.to_bits());
        let depth_key = (depth_key, window[1].to_bits() as u64 | (window[2].to_bits() as u64) << 32);
        // A pair that reads the destination colour binds a VIEW in this group, so the
        // generation of that view joins the key: a cached bind group naming a texture the
        // renderer has since recreated is a use-after-free the validator catches, and one
        // naming a DIFFERENT pass's texture would silently blend against the wrong image.
        let needs_dest = pipe.needs_dest;
        let dest = match (needs_dest, dest) {
            (false, _) => None,
            (true, Some(d)) => Some(d),
            // The pipeline declares a destination texture and the pass has none. Binding
            // nothing is not an option (the layout would not match), and drawing without
            // the copy would blend against whatever was last there. Drop it, loudly.
            (true, None) => {
                report_missing_dest_texture(key);
                record_fallback_reason(
                    key,
                    "the pipeline declares a DESTINATION texture and this pass has none",
                );
                return None;
            }
        };
        let depth_key = (depth_key, dest.map(|(_, g)| g).unwrap_or(0));
        // Lanes 12..28: each fragment sampler unit's TEXTURE LOD bias (`tex_bias` in the
        // block). Per DRAW - one program draws a layer sharp and another blurred by binding
        // the same image at different biases - so the exact bits join the key.
        let mut tex_bias = [0f32; 16];
        for t in gxp.textures.iter() {
            if let Some(slot) = tex_bias.get_mut(t.unit as usize) {
                *slot = t.lod_bias;
            }
        }
        let bias_key: [u32; 16] = tex_bias.map(f32::to_bits);
        let depth_key = (depth_key, bias_key);
        let store_key = store.as_ref().map(|(_, id, header)| (*id, header.clone()));
        let bg3_key = (key, depth_key, corrected, store_key);
        if depth_bgs.contains_key(&bg3_key) {
            enc(&ENC.bind_groups_reused, 1);
        } else {
            // A race moves the depth range every frame, so this cache mints entries that
            // are never asked for again - see the field's doc comment for the measurement.
            // Dropped wholesale rather than growing without bound: the key is the value,
            // so every entry rebuilds byte-identical.
            let mut evicted: Vec<(wgpu::BindGroup, wgpu::Buffer)> = Vec::new();
            if drain_if_at_cap(depth_bgs, DEPTH_BG_CACHE_CAP, &mut evicted) {
                enc(&ENC.depth_bg_cache_clears, 1);
                // The buffers go to the renderer's graveyard, NOT to the collector, and not
                // to an immediate `destroy()` either - a draw prepared earlier this frame
                // may still name the bind group that owns one. See `depth_retired`.
                depth_retired.extend(evicted.into_iter().map(|(_, buf)| buf));
            }
        }
        let (bg3, _) = depth_bgs.entry(bg3_key).or_insert_with(|| {
            enc(&ENC.bind_groups_built, 1);
            enc_buffer_created();
            enc(&ENC.buffer_bytes, GXP_DEPTH_BLOCK_BYTES);
            // A storing program's window headers ride in the SAME buffer, past the depth block
            // at the uniform offset alignment, so one buffer per entry stays the one handle
            // `destroy()` needs.
            let header_at = GXP_DEPTH_BLOCK_BYTES.next_multiple_of(u64::from(device.limits().min_uniform_buffer_offset_alignment));
            let size = match &store {
                Some((_, _, header)) => header_at + header.len() as u64 * 4,
                None => GXP_DEPTH_BLOCK_BYTES,
            };
            // >>> NOT `create_buffer_init`, AND THAT IS THE WHOLE POINT.
            //
            // `create_buffer_init` is `mappedAtCreation` on the web backend, so every call
            // allocates a renderer-side shared-memory STAGING region as well as a GPU
            // buffer. This cache mints an entry per distinct depth key and the comment
            // above says why that is not rare: a title that moves its viewport depth per
            // draw produces a new key constantly. Thousands of mapped allocations later
            // Chrome refuses one, and it refuses whichever comes next - so the panic reads
            // `createBuffer failed, size (48) is too large for the implementation`, blames
            // a forty-eight byte buffer, and kills the run worker.
            //
            // That is not a hypothesis. The per-pass arena pool a few hundred lines up was
            // built for exactly this failure, with exactly this message at a different size
            // (`size (1332)`), and its note ends "the fix is to stop asking for a new one
            // every frame". This site was left behind by that fix, and it is the one in
            // every crash stack a phone has produced on this title.
            //
            // `create_buffer` + `write_buffer` puts the same 48 bytes in the same place
            // through the queue's own staging, which is pooled by the implementation and
            // not one arena allocation per buffer.
            let dbuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gxp-depth"),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(
                &dbuf,
                0,
                &[
                    depth_range[0].to_le_bytes(),
                    depth_range[1].to_le_bytes(),
                    sign.to_le_bytes(),
                    zfix_mode.to_le_bytes(),
                    fit_a.to_le_bytes(),
                    fit_c.to_le_bytes(),
                    z_scale.to_le_bytes(),
                    z_offset.to_le_bytes(),
                    // `vp`: the viewport's vertical sense, then the window-pixel mapping of
                    // a draw whose viewport is disabled (flag, 2/width, 2/height).
                    y_sense.to_le_bytes(),
                    window[0].to_le_bytes(),
                    window[1].to_le_bytes(),
                    window[2].to_le_bytes(),
                ]
                .into_iter()
                .chain(tex_bias.map(f32::to_le_bytes))
                .collect::<Vec<_>>()
                .concat(),
            );
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: if store.is_some() {
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &dbuf,
                        offset: 0,
                        size: wgpu::BufferSize::new(GXP_DEPTH_BLOCK_BYTES),
                    })
                } else {
                    dbuf.as_entire_binding()
                },
            }];
            if let Some((view, _)) = dest {
                entries.push(wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                });
            }
            if let Some((mirror, _, header)) = &store {
                let bytes: Vec<u8> = header.iter().flat_map(|w| w.to_le_bytes()).collect();
                queue.write_buffer(&dbuf, header_at, &bytes);
                entries.push(wgpu::BindGroupEntry { binding: 2, resource: mirror.as_entire_binding() });
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &dbuf,
                        offset: header_at,
                        size: wgpu::BufferSize::new(bytes.len() as u64),
                    }),
                });
            }
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gxp-depth-bind"),
                layout: &pipe.layouts[3],
                entries: &entries,
            });
            // The buffer is kept, not dropped: it is the only handle `destroy()` can ever
            // be called on, and the bind group does not offer one.
            (bg, dbuf)
        });
        let bg3 = bg3.clone();
        split_end(t_depth, &PREP.depth_ns);

        Some(GxpPrepared {
            key,
            v_off,
            v_len,
            i_off,
            i_len,
            v_resident,
            i_resident,
            index_count: gxp.index_count,
            index_u32: gxp.index_u32,
            u_off,
            bg2,
            bg3,
            needs_dest,
            blend: gxp.blend,
            format: color_format,
            samples,
            cull: gxp.cull_mode,
            layout: GxpLive::vertex_layout_key(gxp),
            raster,
            colour_noop,
            depth_write: gxp.depth_write,
            stencil_write: !crate::gpu::stencil_disabled()
                && ((gxp.stencil[5] & 0xff != 0 && gxp.stencil[1..4].iter().any(|op| op & 7 != 0))
                    || (gxp.two_sided
                        && gxp.stencil[12] & 0xff != 0
                        && gxp.stencil[8..11].iter().any(|op| op & 7 != 0))),
            stencil_ref: gxp.stencil[6] & 0xff,
            fmem_ranges,
            fmem_bytes,
            clip_sign: sign,
        })
    }
}
