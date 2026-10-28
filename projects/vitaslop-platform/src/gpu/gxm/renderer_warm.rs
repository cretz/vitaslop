use super::*;

impl GxmRenderer {
    /// Encode a whole FRAME - every GXM scene the guest submitted, in order - so that a
    /// pass which renders into an offscreen buffer is actually drawn, and a later pass
    /// that samples that buffer gets what was drawn.
    ///
    /// This is the difference between a HUD and a game. A 3D title does not build its
    /// frame in one pass: it renders the world (and its shadow maps, reflections,
    /// post-process chain) into offscreen colour surfaces and then composites them onto
    /// the display buffer. Rendering only the last scene - which is all a single-scene
    /// `encode` can do - draws only that composite, and every texture it samples comes
    /// from guest memory that the guest never wrote, because on hardware the GPU wrote
    /// it. One of the retail racers is exactly this: fourteen offscreen passes carrying
    /// the entire world, then a 24-draw composite. The result was a correct, live HUD
    /// over a black screen.
    ///
    /// Scenes are encoded in submission order (a pass may sample an EARLIER pass's
    /// target, so order is load-bearing). A scene whose target is the same buffer the
    /// final scene draws to goes straight into the caller's view, clearing only on the
    /// first such pass so later ones compose onto it; every other scene goes into an
    /// offscreen target held by its surface's guest address. A scene with no resolvable
    /// target cannot be placed at all and is reported rather than silently dropped.
    #[allow(clippy::too_many_arguments)]
    /// Compile the WGSL for every shader pair the guest's patcher has named and this
    /// renderer has not compiled yet.
    ///
    /// # This is where the hardware does its shader work
    /// A `.gxp` carries USSE machine code the SDK compiled offline, so
    /// `sceGxmShaderPatcherCreateFragmentProgram` only has to patch and link - and it is
    /// handed `const SceGxmProgram *vertexProgram` precisely so it CAN, which means the
    /// pair is fully determined while the title is still on a loading screen. This
    /// recompiler has to produce WGSL and have a driver compile it, and doing that at the
    /// first DRAW put the whole cost inside gameplay frames: MEASURED on one title's
    /// on-track run, 931 ms of WGSL compile and 449 ms of pipeline creation, 160 pipelines
    /// built ACROSS the race, single frames spending 50-100 ms building 2-6 of them.
    ///
    /// Only the MODULE is prepared here. A pipeline is bound to the draw's cull mode and
    /// depth state, which GXM sets as runtime state and the patcher genuinely does not know.
    ///
    /// A pair that fails to link is skipped in silence: the draw path reports a fallback
    /// with the reason, once, at the site that knows which draw wanted it, and reporting the
    /// same failure from here would fire for pairs the title never draws.
    /// >>> TRANSLATE THE PAIRS THE TITLE NAMED, AHEAD OF THE DRAWS THAT WILL USE THEM.
    ///
    /// Each pair the shader patcher named (`sceGxmShaderPatcherCreateFragmentProgram`, with
    /// the layout its vertex program was created with) is linked into the translation memo
    /// under the SAME options the draw path computes for its common case - see
    /// `vitaslop_gxp_shader::link_programs_memo`. The draw then finds it and pays a hash,
    /// not a translation: MEASURED, our translation was 731-780 ms of in-frame time a golf
    /// title's session and the whole of the "new character appears" stall on a fighting title. A draw whose
    /// state differs (a dual-source blend, a raw sampler slot, a masked colour) misses and
    /// translates as before, so this can cost work but never a picture.
    ///
    /// Spread over frames by a per-frame budget, on the render worker - NOT at the patcher
    /// call, which runs under the host lock every guest thread needs. Each pair is walked
    /// once per allocation ([`GxpLive::precompile_seen`]).
    pub fn precompile_pairs(
        &mut self,
        _device: &wgpu::Device,
        pairs: &[vitaslop_gxp_shader::PatcherPair],
    ) {
        // Once: that the patcher's pairs reached the renderer at all, and whether the walk
        // is armed - a "0 translated AHEAD" panel line is otherwise ambiguous between "no
        // pairs arrived" and "the walk never ran".
        {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !pairs.is_empty() && !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                report_status!(
                    "gxp prelink: {} patcher-named pair(s) reached the renderer (gxp live {}, VITASLOP_GXP_PRECOMPILE {})",
                    pairs.len(),
                    self.gxp.enabled,
                    if crate::gpu::gxp_precompile() { "on" } else { "OFF" }
                );
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.premodule_pairs(_device, pairs);
        if !self.gxp.enabled || pairs.is_empty() || !crate::gpu::gxp_precompile() {
            return;
        }
        let t = Stopwatch::start();
        for (walked, p) in pairs.iter().enumerate() {
            // The first pair of a call always runs, so progress cannot stall however slow
            // the device is; after that the budget covers the whole loop body.
            if walked > 0 && t.ms() >= PRECOMPILE_MS_PER_FRAME {
                break;
            }
            let attrs_id = p.attrs.iter().fold(0usize, |h, &(r, f, c)| {
                h.wrapping_mul(31).wrapping_add((r as usize) << 16 | (f as usize) << 8 | c as usize)
            });
            let akey = (
                std::sync::Arc::as_ptr(&p.vprog) as *const u8 as usize,
                std::sync::Arc::as_ptr(&p.fprog) as *const u8 as usize ^ attrs_id.rotate_left(17),
            );
            if self.gxp.precompile_seen.len() >= PRECOMPILE_SEEN_CAP {
                self.gxp.precompile_seen.clear();
            }
            if !self.gxp.precompile_seen.insert(akey) {
                continue;
            }
            let _owner = crate::heap::owner_scope(crate::heap::Owner::Shader);
            vitaslop_gxp_shader::prelink(
                &p.vprog,
                &p.fprog,
                vitaslop_gxp_shader::LinkOptions {
                    guest_attrs: p.attrs.clone(),
                    patched_against: Some(p.vprog.clone()),
                    ..Default::default()
                },
                crate::gpu::now_ms,
            );
        }
    }

    /// The key [`Self::pass_shapes`] is recorded and read under.
    pub(super) fn pass_id(scene: &RenderScene) -> Option<(u32, u32, u32, u32)> {
        scene.target.map(|t| (t.data_addr, t.width, t.height, t.format))
    }

    /// >>> START, WITH `createRenderPipelineAsync`, EVERY PIPELINE THESE SCENES WILL NEED
    /// >>> AND DO NOT HAVE - before the frame is encoded. See `async_pipelines`.
    ///
    /// A synchronous create compiles on the browser's GPU main thread and froze the whole
    /// browser for seconds on the phone. The encode cannot wait for a promise halfway
    /// through (a worker cannot block, and abandoning a half-encoded frame would undo RTT,
    /// residency and write-back state), so the caller awaits [`PipelineWarm::wait`] BEFORE
    /// encoding and hands the result to [`Self::install_pipelines`]: the encode then finds
    /// every one of them in the cache.
    ///
    /// The keys are [`GxpLive::pipe_key`] - the same function the encode calls - under the
    /// pass shapes this scene's target was last encoded with (`pass_shapes`). What that
    /// cannot predict builds synchronously in the encode as before and is counted as
    /// `pipelines_unwarmed`, so the prediction's coverage is a number on the panel.
    ///
    /// # A target never encoded before gets the offscreen arm's shape, as on native
    /// A synchronous create is not only slow, it is SERIAL: every one compiles on the GPU
    /// process's main thread, and the async creates queued behind it wait too. MEASURED
    /// (desktop Chrome trace, a shooter's mission load): 135 synchronous creates - 132 of them
    /// into new 64-bit raw targets - held the GPU main thread for 13.5 s straight (DXC
    /// 100-650 ms each); the worker blocked 6.4 s for wire space meanwhile, and the next
    /// present's 21 async pipelines did not START until the sync run ended, 6-9 s later.
    /// The same 176 pipelines created async ran 15 wide and took 0.6 s. So a target with no
    /// recorded shape is guessed with [`Self::offscreen_shape`] - exact for a raw-64 target
    /// (raw words, one sample) and for a single-sampled one unless its scene is drawn through
    /// a tile shadow - and a multisampled non-gamma target, whose sample count `ensure_rtt`
    /// decides, is still left to the encode. A wrong guess costs one async compile in a wide
    /// pool and the encode builds what it needs, exactly as before. Raw-64 targets rendered by
    /// an earlier scene of this batch count as raw for the samplers of later ones, as the
    /// encode's own `rtt_rendered_raw` will. `VITASLOP_WARM_NEW_TARGETS=0` is the arm back.
    pub fn warm_pipelines(&mut self, device: &wgpu::Device, scenes: &[RenderScene]) -> PipelineWarm {
        let mut items = Vec::new();
        if !self.gxp.enabled || !crate::gpu::async_pipelines() {
            return PipelineWarm { items };
        }
        self.gxp.warm_used = true;
        self.gxp.frame_needs.clear();
        let mut started: HashSet<PipeCacheKey> = HashSet::default();
        let guess_new = crate::gpu::warm_new_targets();
        let mut batch_raw: HashSet<(u32, u32, u32)> = HashSet::default();
        for scene in scenes {
            if scene.draws.is_empty() {
                continue;
            }
            let own_raw = scene
                .target
                .filter(|t| guess_new && raw64_color_format(t.format))
                .map(|t| (t.data_addr, t.width, t.height));
            let Some(shapes) = self
                .pass_shapes
                .get(&Self::pass_id(scene))
                .cloned()
                .or_else(|| {
                    scene
                        .target
                        .filter(|_| crate::gpu::warm_by_format())
                        .and_then(|t| self.pass_shapes_by_format.get(&t.format).cloned())
                })
                .or_else(|| {
                    scene
                        .target
                        .filter(|t| {
                            guess_new
                                && (raw64_color_format(t.format)
                                    || t.gamma
                                    || crate::gpu::gxm_sample_count(t.multisample) <= 1)
                        })
                        .map(|t| vec![self.offscreen_shape(&t)])
                })
            else {
                batch_raw.extend(own_raw);
                continue;
            };
            for d in &scene.draws {
                let Some(g) = &d.gxp else { continue };
                // The same early refusals as `GxpLive::prepare`.
                if g.index_count == 0 || g.vertices.is_empty() {
                    continue;
                }
                let key = self.gxp.pair_key(g);
                if (!self.gxp.keys.is_empty() && !self.gxp.keys.contains(&key)) || self.gxp.exclude.contains(&key) {
                    continue;
                }
                let noop_keeps_depth = g.depth_write && draw_colour_is_noop(g);
                for &(fmt, samples, alpha_single) in &shapes {
                    let rtt_raw = &self.rtt_raw;
                    let pk = GxpLive::pipe_key(g, key, noop_keeps_depth, fmt, alpha_single, samples, |k| {
                        rtt_raw.contains_key(k) || batch_raw.contains(k)
                    });
                    if self.gxp.in_flight.contains(&pk.cache_key) {
                        self.gxp.frame_needs.push(pk.cache_key);
                        continue;
                    }
                    if self.gxp.pipelines.contains_key(&pk.cache_key) || !started.insert(pk.cache_key) {
                        continue;
                    }
                    report_unfed_uniforms(key, "vertex", &g.vprog);
                    report_unfed_uniforms(key, "fragment", &g.fprog);
                    enc(&ENC.pipelines_built, 1);
                    log_pipe("async", &pk.cache_key, g);
                    enc(&ENC.pipelines_async, 1);
                    let l = &mut self.gxp;
                    let built = build_gxp_pipeline(
                        device,
                        fmt,
                        alpha_single,
                        samples,
                        g.cull_mode,
                        g,
                        key,
                        l.zfix,
                        l.yflip,
                        l.solid,
                        l.nodepth,
                        l.noblend,
                        noop_keeps_depth,
                        pk.raw_units,
                        pk.f32_units,
                        &mut l.modules,
                        |d, desc| d.create_render_pipeline_async(desc),
                    );
                    match built {
                        Some(p) => {
                            self.gxp.in_flight.insert(pk.cache_key);
                            self.gxp.frame_needs.push(pk.cache_key);
                            items.push((pk.cache_key, p));
                        }
                        // Refused by our own link: cached as refused, as the encode would.
                        None => {
                            self.gxp.pipelines.insert(pk.cache_key, None);
                        }
                    }
                }
            }
            batch_raw.extend(own_raw);
        }
        PipelineWarm { items }
    }

    /// >>> NATIVE: TRANSLATE AND COMPILE THE PAIRS THE PATCHER NAMED, ON BACKGROUND THREADS,
    /// >>> WHILE THE TITLE IS STILL LOADING - so the frame that first draws them finds both.
    ///
    /// [`Self::prebuild_pipelines`] spreads a frame's new pairs over threads, but the shader
    /// MODULE half does not parallelise much (MEASURED: 127-177 ms of module work in a
    /// 126-189 ms present at a fighting title's intro starts), and those presents still
    /// emptied the audio lead. The patcher names those very pairs during the loading screen
    /// before them, where there is time to spare. So each newly named pair is translated
    /// with the options the draw computes for its common case (the same guess as the
    /// translation-only [`Self::precompile_pairs`]: the translation memo then serves the
    /// draw), run through the same clip fixup, compiled into a module, and filed under the
    /// draw path's own module key. A draw whose state differs (dual-source, raw units, a
    /// masked colour, an alpha target) computes another key and builds as before - work,
    /// never a different picture. Off under the WGSL override and keycolour diagnostics,
    /// which change the module per pair.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn premodule_pairs(&mut self, device: &wgpu::Device, pairs: &[vitaslop_gxp_shader::PatcherPair]) {
        if !self.gxp.enabled || pairs.is_empty() || !crate::gpu::parallel_pipelines() {
            return;
        }
        if crate::knobs::var("VITASLOP_GXP_WGSL_OVERRIDE_DIR").is_ok()
            || crate::knobs::var("VITASLOP_GXP_WGSL_OVERRIDE_TEXT").is_ok()
            || crate::knobs::var("VITASLOP_GXP_KEYCOLOR").is_ok()
        {
            return;
        }
        if self.premodule_seen.len() >= PRECOMPILE_SEEN_CAP {
            self.premodule_seen.clear();
        }
        let fresh: Vec<vitaslop_gxp_shader::PatcherPair> = pairs
            .iter()
            .filter(|p| {
                let akey = (
                    std::sync::Arc::as_ptr(&p.vprog) as *const u8 as usize,
                    std::sync::Arc::as_ptr(&p.fprog) as *const u8 as usize,
                );
                self.premodule_seen.insert(akey)
            })
            .cloned()
            .collect();
        if fresh.is_empty() {
            return;
        }
        let (zfix, yflip, solid) = (self.gxp.zfix, self.gxp.yflip, self.gxp.solid);
        // >>> THE FORM THE TITLE'S DRAWS ACTUALLY TAKE. A draw carries the vertex program its
        // fragment was patched against on some titles and not on others (MEASURED: a
        // fighting title's every draw linked UNPATCHED, so modules built for the patched
        // form - the translation-only precompile's assumption - matched none of them). The
        // draws so far decide; before any, the patched form, as the precompile always took.
        let patched = {
            use std::sync::atomic::Ordering::Relaxed;
            crate::gpu::DRAWS_PATCHED.load(Relaxed) >= crate::gpu::DRAWS_UNPATCHED.load(Relaxed)
        };
        // Modest on purpose: MEASURED, eight threads here slowed the in-frame builds they
        // compete with (a fight start's present 218 -> 282 ms) more than they saved.
        let threads = std::thread::available_parallelism().map_or(2, |n| n.get() / 4).clamp(1, 4).min(fresh.len());
        let fresh = std::sync::Arc::new(fresh);
        for t in 0..threads {
            let (device, out, fresh) = (device.clone(), self.premodules.clone(), fresh.clone());
            let _ = std::thread::Builder::new().name("gxp-premodule".into()).spawn(move || {
                for p in fresh.iter().skip(t).step_by(threads) {
                    let opts = vitaslop_gxp_shader::LinkOptions {
                        guest_attrs: p.attrs.clone(),
                        patched_against: patched.then(|| p.vprog.clone()),
                        ..Default::default()
                    };
                    let Ok(linked) = vitaslop_gxp_shader::link_programs_memo(&p.vprog, &p.fprog, opts) else { continue };
                    // The draw path's key for this case - see the `mkey` in `build_gxp_pipeline`.
                    let mkey = GxpLive::module_key(&p.vprog, &p.fprog)
                        ^ if patched { GxpLive::module_key(&p.vprog, b"patched").rotate_left(17) } else { 0 }
                        ^ gxp_attr_interface_fold(&linked.vertex_bindings.attributes);
                    let Some(wgsl) = inject_clip_fixup(&linked.wgsl, zfix, yflip, solid, None, true) else { continue };
                    let wgsl = vitaslop_gxp_shader::wgsl::hoist_diagnostics(&wgsl);
                    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(&format!("gxp-linked:{mkey:016x}")),
                        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                    });
                    out.lock().unwrap_or_else(|e| e.into_inner()).push((mkey, module));
                }
            });
        }
    }

    /// Move the modules [`Self::premodule_pairs`] finished into the cache the draws read.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn take_premodules(&mut self) {
        let done = std::mem::take(&mut *self.premodules.lock().unwrap_or_else(|e| e.into_inner()));
        for (k, m) in done {
            self.gxp.modules.entry(k).or_insert(m);
        }
    }

    /// The `(attachment format, samples, alpha_single)` the OFFSCREEN arm of `encode_chain`
    /// renders target `t` with, at one sample: RAW words for a 64-bit format, the sRGB twin
    /// for a gamma surface, the float target for a float format, else the renderer's own
    /// format. A multisampled target is encoded at its sample count and is not guessed.
    pub(super) fn offscreen_shape(&self, t: &crate::gpu::RttTarget) -> (wgpu::TextureFormat, u32, bool) {
        let fmt = if raw64_color_format(t.format) {
            RAW64_FORMAT
        } else if t.gamma {
            srgb_twin(self.color_format).unwrap_or(self.color_format)
        } else if float_color_format(t.format) {
            FLOAT_TARGET_FORMAT
        } else {
            self.color_format
        };
        (fmt, 1, alpha_single_color_format(t.format))
    }

    /// >>> NATIVE: BUILD EVERY PIPELINE THESE SCENES WILL NEED AND DO NOT HAVE, ON SEVERAL
    /// >>> THREADS, BEFORE THE ENCODE - the desktop's form of [`Self::warm_pipelines`].
    ///
    /// A native device has no asynchronous create, so a frame that introduces a dozen pairs
    /// built them one after another inside the encode, on the one thread the guest's next
    /// frame waits behind. MEASURED in a fighting title's window at the start of each
    /// pre-fight intro: a 254-286 ms present, of it 46-57 ms our translation, 110-139 ms
    /// shader modules and 36-45 ms pipelines - long enough to empty the 120 ms audio lead,
    /// which is the choppy sound on the intros the user heard. The three are independent per
    /// pair and a native `wgpu::Device` is thread-safe, so they are split across threads
    /// here, each with its own module map, and merged into the caches the encode reads.
    ///
    /// Predicted with the same keys and pass shapes as the browser's warm; what that cannot
    /// predict still builds in the encode exactly as before. A refused pair is cached as
    /// refused, as the encode would. `VITASLOP_PARALLEL_PIPELINES=0` is the arm back.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn prebuild_pipelines(&mut self, device: &wgpu::Device, scenes: &[RenderScene]) {
        if !self.gxp.enabled || !crate::gpu::parallel_pipelines() {
            return;
        }
        self.take_premodules();
        struct Job<'a> {
            g: &'a GxpRecompile,
            key: u64,
            fmt: wgpu::TextureFormat,
            alpha_single: bool,
            samples: u32,
            noop_keeps_depth: bool,
            pk: PipeKey,
        }
        let mut jobs: Vec<Job> = Vec::new();
        let mut seen: HashSet<PipeCacheKey> = HashSet::default();
        for scene in scenes {
            if scene.draws.is_empty() {
                continue;
            }
            let Some(shapes) = self
                .pass_shapes
                .get(&Self::pass_id(scene))
                .cloned()
                .or_else(|| {
                    scene
                        .target
                        .filter(|_| crate::gpu::warm_by_format())
                        .and_then(|t| self.pass_shapes_by_format.get(&t.format).cloned())
                })
                // A target never encoded before: the shape the OFFSCREEN arm of
                // `encode_chain` gives it, at one sample. A guest COMPLETION is all new
                // targets at a load, and building their pipelines one by one in the encode is
                // what the guest waited on - MEASURED (a golf title's course load, window):
                // 1.1 s of serial pipeline creation in one completion. A wrong guess costs one
                // wasted build on a helper thread; the encode still builds what it needs.
                // Native only: in the browser a wrong guess is a whole async compile taking a
                // slot from the real ones (see the platform README).
                .or_else(|| scene.target.map(|t| vec![self.offscreen_shape(&t)]))
            else {
                continue;
            };
            for d in &scene.draws {
                let Some(g) = &d.gxp else { continue };
                if g.index_count == 0 || g.vertices.is_empty() {
                    continue;
                }
                let key = self.gxp.pair_key(g);
                if (!self.gxp.keys.is_empty() && !self.gxp.keys.contains(&key)) || self.gxp.exclude.contains(&key) {
                    continue;
                }
                let noop_keeps_depth = g.depth_write && draw_colour_is_noop(g);
                for &(fmt, samples, alpha_single) in &shapes {
                    let rtt_raw = &self.rtt_raw;
                    let pk = GxpLive::pipe_key(g, key, noop_keeps_depth, fmt, alpha_single, samples, |k| {
                        rtt_raw.contains_key(k)
                    });
                    if self.gxp.pipelines.contains_key(&pk.cache_key) || !seen.insert(pk.cache_key) {
                        continue;
                    }
                    jobs.push(Job { g, key, fmt, alpha_single, samples, noop_keeps_depth, pk });
                }
            }
        }
        // One pipeline gains nothing from a thread; the encode builds it as it always has.
        if jobs.len() < 2 {
            return;
        }
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(2, 8).min(jobs.len());
        let (zfix, yflip, solid, nodepth, noblend) =
            (self.gxp.zfix, self.gxp.yflip, self.gxp.solid, self.gxp.nodepth, self.gxp.noblend);
        #[allow(clippy::type_complexity)]
        let results: Vec<(Vec<(PipeCacheKey, Option<GxpPipeline>)>, HashMap<u64, wgpu::ShaderModule>)> =
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let jobs = &jobs;
                        s.spawn(move || {
                            let mut modules: HashMap<u64, wgpu::ShaderModule> = HashMap::default();
                            let mut built = Vec::new();
                            // By PAIR, not by job: a pair's pass shapes share its translation
                            // and module, and dealing them to two threads built both twice.
                            for j in jobs.iter().filter(|j| (j.key % threads as u64) as usize == t) {
                                report_unfed_uniforms(j.key, "vertex", &j.g.vprog);
                                report_unfed_uniforms(j.key, "fragment", &j.g.fprog);
                                enc(&ENC.pipelines_built, 1);
                                log_pipe("prebuilt", &j.pk.cache_key, j.g);
                                let p = build_gxp_pipeline(
                                    device,
                                    j.fmt,
                                    j.alpha_single,
                                    j.samples,
                                    j.g.cull_mode,
                                    j.g,
                                    j.key,
                                    zfix,
                                    yflip,
                                    solid,
                                    nodepth,
                                    noblend,
                                    j.noop_keeps_depth,
                                    j.pk.raw_units,
                                    j.pk.f32_units,
                                    &mut modules,
                                    |d, desc| d.create_render_pipeline(desc),
                                );
                                built.push((j.pk.cache_key, p));
                            }
                            (built, modules)
                        })
                    })
                    .collect();
                handles.into_iter().filter_map(|h| h.join().ok()).collect()
            });
        for (built, modules) in results {
            for (k, m) in modules {
                self.gxp.modules.entry(k).or_insert(m);
            }
            for (k, p) in built {
                self.gxp.pipelines.entry(k).or_insert(p);
            }
        }
    }

    /// Whether a pipeline the frame last warmed draws with is still being compiled - the
    /// frame cannot be encoded without building it synchronously. Compiles the frame does
    /// NOT use do not count: they are some other frame's, and this one can go ahead.
    pub fn frame_waits_on_compile(&self) -> bool {
        self.gxp.frame_needs.iter().any(|k| self.gxp.in_flight.contains(k))
    }

    /// Put what [`PipelineWarm::wait`] resolved where the encode looks for it. A key the
    /// encode built in the meantime keeps the encode's.
    pub fn install_pipelines(&mut self, ready: PipelinesReady) {
        for k in &ready.settled {
            self.gxp.in_flight.remove(k);
        }
        for (k, p) in ready.items {
            self.gxp.pipelines.entry(k).or_insert(Some(p));
        }
    }
}
