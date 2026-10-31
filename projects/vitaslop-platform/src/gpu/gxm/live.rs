//! [`GxpLive`]: configuration, the per-pair uniform bind groups, the per-pass clip verdict,
//! and the keys a pair and its pipeline are cached under.

use super::*;

impl GxpLive {
    /// Read the recompiler's configuration.
    ///
    /// Through [`crate::knobs`], not `std::env` directly: the browser has no
    /// environment, so an `std::env` read here silently disables the recompiler
    /// there and leaves the browser drawing the fixed-function approximation while
    /// the desktop oracle draws the guest's real shaders.
    pub(super) fn from_env() -> Self {
        note_view_cache_created();
        let flag = crate::knobs::flag;
        GxpLive {
            enabled: flag("VITASLOP_GXP_LIVE"),
            only: flag("VITASLOP_GXP_ONLY"),
            zfix: match crate::knobs::var("VITASLOP_GXP_ZFIX").ok().as_deref() {
                Some("gl") => ZFix::Gl,
                Some("range") => ZFix::Range,
                Some("0") | Some("off") => ZFix::Off,
                Some("clamp") => ZFix::Clamp,
                _ => ZFix::Viewport,
            },
            yflip: crate::knobs::var("VITASLOP_GXP_YFLIP").map(|v| v != "0").unwrap_or(false),
            force: flag("VITASLOP_GXP_FORCE"),
            solid: flag("VITASLOP_GXP_SOLID"),
            nodepth: flag("VITASLOP_GXP_NODEPTH"),
            noblend: flag("VITASLOP_GXP_NOBLEND"),
            keys: crate::knobs::var("VITASLOP_GXP_KEYS")
                .ok()
                .map(|v| {
                    v.split(',')
                        .filter_map(|k| u64::from_str_radix(k.trim().trim_start_matches("0x"), 16).ok())
                        .collect()
                })
                .unwrap_or_default(),
            exclude: crate::knobs::var("VITASLOP_GXP_EXCLUDE")
                .ok()
                .map(|v| {
                    v.split(',')
                        .filter_map(|k| u64::from_str_radix(k.trim().trim_start_matches("0x"), 16).ok())
                        .collect()
                })
                .unwrap_or_default(),
            pipelines: HashMap::default(),
            warm_used: false,
            in_flight: HashSet::default(),
            frame_needs: Vec::new(),
            modules: HashMap::default(),
            pair_keys: HashMap::default(),
            views: HashMap::default(),
            views_bytes: 0,
            views_used: HashMap::default(),
            view_slots: HashMap::default(),
            view_dead: HashSet::default(),
            views_evicted: HashSet::default(),
            views_epoch: 0,
            views_frame_high: 0,
            views_frame_bytes: 0,
            bc_supported: None,
            texenc: None,
            samplers_by_mode: HashMap::default(),
            negw: match crate::knobs::var("VITASLOP_GXP_NEGW").ok().as_deref() {
                Some("0") | Some("off") => NegW::Off,
                Some("negate") => NegW::Negate,
                Some("force") => NegW::Force,
                _ => NegW::Auto,
            },
            negw_by_key: HashMap::default(),
            negw_retry: HashMap::default(),
            scene_negw: false,
            scene_depth_fit: DEPTH_FIT_RECIP_W,
            negw_by_target: HashMap::default(),
            fit_by_addr: HashMap::default(),
            ubo_bgs: HashMap::default(),
            ubo_bgs_gen: HashMap::default(),
            packed: PackedCache::default(),
            packed_by_alloc: PackedAllocCache::default(),
            resident_v: ResidentHeap::new(),
            resident_i: ResidentHeap::new(),
            resident_i_seen: HashMap::default(),
            resident_v_seen: HashMap::default(),
            precompile_seen: HashSet::default(),
            // Value-sensitive, as an A/B arm has to be: `0` sends every draw back through
            // the per-frame arenas, which is what every draw did before this existed.
            resident: crate::knobs::var("VITASLOP_RESIDENT_GEOM").map(|v| v.trim() != "0").unwrap_or(true),
            // >>> 128 MB, AND THE NUMBER IS MEASURED. A budget too small for a title's live
            // set does not degrade gracefully - the heap fills, the compaction path
            // declines (the live set is over 3/4 of the budget, so compacting buys
            // nothing), and it RESETS WHOLESALE, every frame, re-uploading everything it
            // held. MEASURED on a retail baseball title's gameplay, 101 rendered frames,
            // picture BIT-IDENTICAL at every budget:
            //
            //    48 MB (the old default)  131.74 ms/frame   72 resets
            //    96 MB                    124.54 ms/frame   69 resets   <- still thrashing
            //   128 MB                    101.76 ms/frame    0 resets
            //   192 MB                    106.70 ms/frame    0 resets
            //
            // and the encode split at 48 vs 192: `prepare` 113.29 -> 8.74 ms, submit+wait
            // 87.60 -> 12.33 ms. So the cost of being one megabyte short is a THIRD of the
            // frame, not a few re-uploads.
            //
            // IT IS A CAP, NOT AN ALLOCATION: the heap starts at 1 MB and doubles only as
            // it fills (see `grow_or_reset`), so a title whose working set is 10 MB sits at
            // 16 MB whatever this says. Raising it therefore costs nothing on a title that
            // does not need it, and the titles that DO need it were paying for the miss.
            // The knob remains, because the trade it makes - memory against thrash - is a
            // device question and a phone may want the other answer.
            resident_budget: crate::knobs::var("VITASLOP_RESIDENT_GEOM_MB")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(128)
                .clamp(1, 1024)
                * 1024
                * 1024,
            depth_bgs: HashMap::default(),
            depth_retired: Vec::new(),
            store_mirrors: StoreMirrors::default(),
            sampler_bgs: SamplerBgCache::default(),
            last_sampler_bg: None,
            mem_push_memo: HashMap::default(),
            mem_push_frame: HashSet::default(),
            raw_fill: Vec::new(),
            udata_holes: Vec::new(),
            sampler_pre: HashMap::default(),
        }
    }

    /// Ensure a group0/group1 bind group exists over the pass's uniform arena for every
    /// shader pair the pass uses. Called once per pass, after the arena buffer exists -
    /// which is why it cannot happen inside `prepare`: the buffer is sized from the arena
    /// `prepare` fills.
    pub(super) fn ensure_ubo_bgs(
        &mut self,
        device: &wgpu::Device,
        buffer: &wgpu::Buffer,
        slot: usize,
        generation: u64,
        used: &[(u64, wgpu::TextureFormat, u32, u32, u64, u64, u32)],
    ) {
        // Entries are keyed by ARENA SLOT and dropped only when THAT slot's uniform buffer
        // is re-created. The pool made the buffer stable across frames, so this cache is
        // now stable across frames too - it used to be cleared on every pass of every
        // frame, because every pass got a brand-new buffer to name.
        if self.ubo_bgs_gen.get(&slot) != Some(&generation) {
            self.ubo_bgs.retain(|k, _| k.0 != slot);
            self.ubo_bgs_gen.insert(slot, generation);
        }
        // >>> AND A COUNT BOUND, BECAUSE THE GENERATION BOUND IS NOT ONE.
        //
        // An entry dies only when its SLOT's uniform buffer is re-created, and the arena
        // pool exists precisely so that stops happening - so in a steady run nothing here
        // is ever dropped, and the map grows with every distinct (slot, pair, format,
        // samples, group) the title ever draws. MEASURED on the user's device across one
        // long session: **3,182 -> 15,254 entries**, each one a live `GPUBindGroup` in the
        // browser. That is the same growth-with-run-length shape as every other cache this
        // session bounded; it was missed because "bounded by generation" reads like a
        // bound. [[vitaslop-caches-that-clear-whole-are-the-long-run-degradation]]
        if self.ubo_bgs.len() >= UBO_BG_CACHE_CAP {
            let n = evict_oldest(
                &mut self.ubo_bgs,
                evict_target(UBO_BG_CACHE_CAP),
                self.views_epoch,
                |(_, used)| *used,
            );
            note_cache_evicted("ubo bind groups", n);
        }
        for &(key, format, samples, cull, layout, raster, fmem_bytes) in used {
            let Some(Some(pipe)) = self.pipelines.get(&(key, format, samples, cull, layout, raster)) else { continue };
            for (group, lanes) in [(0u8, pipe.vsa_lanes), (1u8, pipe.fsa_lanes)] {
                // A PER-DRAW fragment window block names its own size - see
                // `GxpPrepared::fmem_bytes`. Every other binding keeps the pipeline's.
                let fmem = if group == 1 { fmem_bytes } else { 0 };
                if let Some((_, last)) = self.ubo_bgs.get_mut(&(slot, key, format, samples, group, fmem)) {
                    // A hit is a use - it is what keeps this entry out of the eviction.
                    *last = self.views_epoch;
                    enc(&ENC.bind_groups_reused, 1);
                    continue;
                }
                enc(&ENC.bind_groups_built, 1);
                let layout = &pipe.layouts[group as usize];
                let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();
                if lanes > 0 {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new((lanes.div_ceil(4) as u64) * 16),
                        }),
                    });
                }
                // The vertex stage's guest-memory window rides in group 0 beside the SA
                // uniform, over the same arena with its own dynamic offset.
                if group == 0 && pipe.mem_bind_bytes > 0 {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(pipe.mem_bind_bytes as u64),
                        }),
                    });
                }
                // ...and the FRAGMENT stage's, in group 1 beside its own SA uniform.
                if group == 1 && pipe.fmem_bind_bytes > 0 {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(if fmem > 0 { u64::from(fmem) } else { pipe.fmem_bind_bytes as u64 }),
                        }),
                    });
                }
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(if entries.is_empty() { "gxp-ubo-empty" } else { "gxp-ubo-bind" }),
                    layout,
                    entries: &entries,
                });
                self.ubo_bgs.insert((slot, key, format, samples, group, fmem), (bg, self.views_epoch));
            }
        }
    }

    /// Whether one PAIR takes the pass's negative-`w` correction. See the use site in
    /// `prepare` for the measurement; the verdict itself is [`Self::decide_scene_negw`].
    ///
    /// Unmeasured pairs follow the pass, which is what this did for every pair before.
    pub(super) fn pair_takes_negw(&self, key: u64) -> bool {
        match self.negw_by_key.get(&key) {
            Some(Some(s)) => s.depth_fit.is_some() || s.in_front == 0,
            _ => true,
        }
    }

    /// Decide, ONCE PER PASS, whether this pass's projection puts clip `w` NEGATIVE in
    /// front of the camera - in which case WebGPU clips every draw away and the pass
    /// renders black with correct shaders, textures, depth and blend (memory
    /// `vitaslop-clip-w-can-be-negative`).
    ///
    /// **It cannot be decided per shader pair, and doing so was a bug.** The sign
    /// convention is a property of the PROJECTION MATRIX, which every draw of a pass
    /// shares; "all of this pair's vertices have `w < 0`" is equally the signature of a
    /// draw that is simply BEHIND THE CAMERA, which WebGPU is then right to clip. Per pair,
    /// those two are indistinguishable. MEASURED on one retail racer's race pass: of ~60 pairs,
    /// exactly two came out negative - both engine-trail ribbons ~700 units behind the eye
    /// - and "correcting" them pulled behind-camera geometry into the frame as a smudge in
    /// the corner. Across a whole pass the two are easy to tell apart: a negative
    /// projection makes EVERY draw negative, and one stray draw cannot outvote the rest.
    ///
    /// The per-pair MEASUREMENT is still cached per key (interpreting a vertex program over
    /// its mesh is not cheap); only the verdict is taken over the pass.
    ///
    /// The same walk settles the pass's DEPTH ENCODING - `z_clip = a * w_clip + c`, hence a
    /// window depth of `a + c/w` - because both answers come from the same interpretation
    /// and neither can be read off the frame afterwards. See [`ClipStats::depth_fit`].
    /// Returns how many vertex programs it INTERPRETED to reach the verdict. Zero on a
    /// settled pass, which is the whole point of the number - see
    /// `EncodePhases::clip_measures`.
    pub(super) fn decide_scene_negw(&mut self, scene: &RenderScene) -> usize {
        if self.negw == NegW::Off || !self.enabled {
            self.scene_negw = false;
            self.scene_depth_fit = DEPTH_FIT_RECIP_W;
            return 0;
        }
        // `=force`: correct every pass without measuring, and SAY SO - a frame built on an
        // assumed projection must never be mistaken for a measured one.
        if self.negw == NegW::Force {
            let target = scene.target.as_ref().map(|t| t.data_addr).unwrap_or(0);
            report_forced_negw(target);
            self.scene_negw = true;
            self.scene_depth_fit = DEPTH_FIT_RECIP_W;
            return 0;
        }
        let target = scene.target.as_ref().map(|t| t.data_addr).unwrap_or(0);
        // The cache key is the surface's guest address AND ITS EXTENT, because a title
        // recycles its colour buffers: a golf title renders a 640x368 front end into the
        // same three buffers it later renders its 960x544 world into. Keyed on the address
        // alone, the front end's screen-space quads settled the verdict and the world - a
        // NEGATIVE projection - inherited "ordinary" and was clipped away for two of the
        // three buffers, whatever the third measured. The extent is what distinguishes the
        // two passes here; every title that renders today keeps its extents, so their keys
        // and their verdicts are unchanged.
        let key = scene
            .target
            .as_ref()
            .map(|t| (t.data_addr, t.width, t.height))
            .unwrap_or((0, 0, 0));
        // Settled passes cost nothing per frame. A pass is only settled once its draws
        // produced EVIDENCE: a frame in which every draw of a pass happens to cover nothing
        // says nothing about its projection, and freezing "not negative" from it would make
        // the answer depend on which frame the pass was first seen in.
        if let Some(&decided) = self.negw_by_target.get(&key) {
            self.scene_negw = decided.0;
            self.scene_depth_fit = decided.1;
            return 0;
        }
        let mut measured = 0usize;
        let (mut in_front, mut behind) = (0usize, 0usize);
        // The same two counts over the draws that carry a PERSPECTIVE - see the verdict
        // below for why the distinction is not a heuristic.
        let (mut in_front_p, mut behind_p) = (0usize, 0usize);
        // The fit is taken from the ONE draw with the widest spread of `w`, not averaged
        // over the pass: `z = a*w + c` is exact for every draw sharing the projection, so a
        // wider spread is simply a better-conditioned way of asking the same question, while
        // an average would let a near-degenerate draw drag the answer.
        let (mut fit, mut best_spread) = (None, 0.0f32);
        for d in &scene.draws {
            let Some(gxp) = d.gxp.as_ref() else { continue };
            // A draw with the viewport transform off writes WINDOW pixels, not a clip
            // position (see `inject_clip_fixup`): it carries no projection to vote on, and
            // measured as clip space it reads as "entirely off screen" - which is also what
            // the degenerate-position warning would then claim about it.
            if gxp.viewport_enable == crate::gpu::SCE_GXM_VIEWPORT_DISABLED {
                continue;
            }
            let key = self.pair_key(gxp);
            let stats = match self.negw_by_key.get(&key) {
                Some(s) => s.clone(),
                None => {
                    // ASK, BUT NOT EVERY TIME - see `negw_retry`. Appearances 1, 2, 4, 8...
                    // are measured; the rest are skipped and contribute zero to the vote,
                    // which is what an empty measurement contributes anyway.
                    let seen = self.negw_retry.entry(key).or_insert(0);
                    *seen = seen.saturating_add(1);
                    // `VITASLOP_GXP_CLIP_RETRY_EVERY_FRAME=1` is the OFF arm - the old
                    // behaviour, one measurement per appearance - so the identity check is
                    // ONE build with a knob per arm rather than two binaries.
                    if !seen.is_power_of_two() && !crate::gpu::gxp_clip_retry_every_frame() {
                        continue;
                    }
                    measured += 1;
                    let s = measure_clip(gxp, key);
                    // Only remember an answer the measurement supports: a draw entirely off
                    // screen decides nothing, and the next draw of the same pair may see the
                    // geometry that settles it.
                    let empty = s.as_ref().is_some_and(|s| s.in_front == 0 && s.behind == 0);
                    if !empty {
                        self.negw_by_key.insert(key, s.clone());
                    }
                    s
                }
            };
            let Some(s) = stats else { continue };
            in_front += s.in_front;
            behind += s.behind;
            if let Some(f) = s.depth_fit {
                in_front_p += s.in_front;
                behind_p += s.behind;
                if s.w_spread > best_spread {
                    best_spread = s.w_spread;
                    fit = Some(f);
                }
            }
        }
        // A negative PROJECTION leaves nothing at all on the positive side. Requiring that
        // (rather than a mere majority) is what keeps a handful of behind-camera draws from
        // turning a pass whose projection is perfectly ordinary. One vertex on the positive
        // side refutes it - which is deliberately strict, and is reported either way.
        // The verdict is taken over the pass's PERSPECTIVE draws when it has any. A draw
        // whose sampled vertices all share one `w` has no projection in it at all - a
        // constant `w` is what an orthographic or screen-space transform produces - so it
        // cannot testify about the sign convention of the perspective matrix the pass's 3D
        // draws share, and under the strict rule (one positive vertex refutes a negative
        // projection) a single full-screen quad silently decides the whole pass.
        //
        // MEASURED before it was acted on, over every title that renders: across
        // an RC racer, a skating title and two racers, the two verdicts agree on EVERY
        // pass, and they disagree on exactly ONE pass anywhere - a golf title's 960x544
        // world, where 241 perspective draws put 2,058 sampled vertices behind the eye and
        // two screen-space quads put 4 in front. Those four vertices were the whole reason
        // that world was clipped away and the frame was black.
        //
        // A pass with NO perspective draw keeps the whole-pass verdict: there is no better
        // evidence, and a screen-space pass whose every vertex is on the negative side is
        // still clipped away without the correction.
        let whole_pass = behind > 0 && in_front == 0;
        let perspective = behind_p > 0 && in_front_p == 0;
        let has_perspective = in_front_p + behind_p > 0;
        let correct = if has_perspective { perspective } else { whole_pass };
        if whole_pass != perspective && has_perspective && in_front + behind > 0 {
            Self::report_perspective_verdict_split(
                target, in_front, behind, in_front_p, behind_p, perspective,
            );
        }
        let depth_fit = fit.unwrap_or(DEPTH_FIT_RECIP_W);
        // SETTLE on the first frame that produced any evidence at all - the original
        // rule, and it is the right one now that the KEY distinguishes the passes that
        // share a buffer. Waiting for a PERSPECTIVE draw instead was tried and MEASURED
        // wrong: a racer's pass whose fit comes from a draw with no vertex inside the
        // frustum then never settles at all, so it re-decides every frame - which
        // re-interprets every pair of the pass per frame (the cost this cache exists to
        // avoid) and lets its stored depth fit drift with the camera. It moved three of
        // that title's frames.
        if in_front + behind > 0 {
            self.negw_by_target.insert(key, (correct, depth_fit));
            self.fit_by_addr.insert(target, depth_fit);
            report!(
                "gxp clip: pass into 0x{target:08x}: {in_front} sampled vertices land in the frustum \
                 with w>0 and {behind} with w<0 -> {}; guest depth = {} + {}/w{}",
                if correct {
                    "the projection is NEGATIVE, CORRECTING the clip w sign for every pair of this pass"
                } else {
                    "clip positions left as the shaders produced them"
                },
                depth_fit.0,
                depth_fit.1,
                if fit.is_some() { "" } else { " (NO draw of this pass fits one - assuming -1/w)" }
            );
        }
        self.scene_negw = correct;
        self.scene_depth_fit = depth_fit;
        measured
    }

    /// Say, once per render target, that the clip-sign verdict taken over EVERY draw of a pass
    /// disagrees with the one taken over its PERSPECTIVE draws alone - and that the perspective
    /// draws are what was believed.
    ///
    /// The verdict is per pass because the sign convention belongs to the projection matrix and
    /// every draw of a pass shares it - per pair, "all of this draw's vertices have `w < 0`" is
    /// equally the signature of a draw that is simply behind the camera. But a pass can contain
    /// draws with NO projection at all (a full-screen 2D quad, a HUD overlay), whose `w` is a
    /// constant the shader wrote rather than a depth, and one of those on the positive side is
    /// enough to refuse a negative projection under the strict rule.
    ///
    /// This exists because that is not a theoretical case: a golf title's world pass renders 241
    /// perspective draws (2,058 sampled vertices, every one with `w < 0`) beside two screen-space
    /// quads (4 vertices, `w > 0`, no `w` spread at all), and the four quad vertices are what
    /// left the whole world clipped away and the frame black.
    fn report_perspective_verdict_split(
        target: u32,
        in_front: usize,
        behind: usize,
        in_front_p: usize,
        behind_p: usize,
        perspective_says_negative: bool,
    ) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert(target) {
            return;
        }
        report_warn!(
            "gxp clip: pass into 0x{target:08x}: the whole-pass verdict and the PERSPECTIVE-draw \
             verdict DISAGREE - over every draw it is {in_front} in front / {behind} behind, over \
             the draws that carry a projection it is {in_front_p} / {behind_p}. The PERSPECTIVE \
             draws decide: this pass's projection is {}",
            if perspective_says_negative { "NEGATIVE" } else { "ordinary" }
        );
    }

    /// The measured `(a, c)` of `guest window depth = a + c/w` for the pass that RENDERS
    /// into `addr`, for a later pass that wants to sample its depth.
    ///
    /// Keyed by the target rather than taken from `scene_depth_fit`, because the pass being
    /// encoded when a depth surface is converted is not necessarily the pass that wrote it.
    /// Falls back to `-1/w` for a target no pass has yet produced evidence for - which is
    /// the honest answer, not a silent zero: it is the encoding with no projection in it.
    pub(super) fn depth_fit_for(&self, addr: u32) -> (f32, f32) {
        self.fit_by_addr.get(&addr).copied().unwrap_or(DEPTH_FIT_RECIP_W)
    }

    /// Stable cache key for a shader pair AND the fixed-function state baked into its
    /// pipeline: FNV-1a over the vertex blob, the fragment blob, the blend equation and
    /// the depth state.
    ///
    /// Everything the pipeline bakes in has to be in the key, because the cache holds
    /// compiled PIPELINES, not modules. Two draws sharing a shader pair but differing in
    /// `SceGxmBlendInfo` or in depth func/write are different pipelines, and leaving
    /// either out silently gives the second draw the first one's state.
    /// Deliberately still the BYTE-at-a-time FNV, and not the word-wise [`fnv64`] used for
    /// content keys: this value is a published identity. It is what `VITASLOP_GXP_KEYS`,
    /// `_EXCLUDE`, `_INPUTS` and `_SA` select a pair by, what the fallback and keycolour
    /// reports name, and what every recorded investigation of this title refers to. Making
    /// it faster would rename every pair and silently invalidate all of that for a fraction
    /// of a millisecond.
    /// [`GxpLive::key`], memoized on the identity of the two program blobs.
    ///
    /// `key` is FNV over the whole vertex AND fragment container, byte at a time, and it is
    /// computed for EVERY draw: a race frame here submits 465 recompiled draws over a
    /// couple of dozen distinct pairs, so it re-hashes megabytes of the same shader bytes
    /// per frame inside `prepare` - the largest single item in the render frame. The blobs
    /// are shared `Arc`s handed out by the capture's own per-program cache, so two draws
    /// bound to the same program carry the same allocation, and a pointer pair identifies
    /// it exactly.
    ///
    /// A miss simply recomputes, so this can only ever be slower, never wrong: the value is
    /// still the published key that `VITASLOP_GXP_KEYS`/`_INPUTS`/`_SA` and every recorded
    /// investigation of this title select a pair by.
    pub(super) fn pair_key(&mut self, gxp: &GxpRecompile) -> u64 {
        let id = PairIdentity {
            vptr: gxp.vprog.as_ptr() as usize,
            vlen: gxp.vprog.len(),
            fptr: gxp.fprog.as_ptr() as usize,
            flen: gxp.fprog.len(),
            blend: gxp.blend_state,
            depth_write: gxp.depth_write,
            depth_func: gxp.depth_func,
            // The bias is part of the pipeline, so two draws that differ only in it
            // must not share one - which is exactly what the identity is for.
            depth_bias: gxp.depth_bias,
            fragment_program_enabled: gxp.fragment_program_enabled,
            dual_source: draw_dual_source(gxp),
        };
        if let Some((k, _, _)) = self.pair_keys.get(&id) {
            return *k;
        }
        // Bounded like the other caches here. The key is derived from the entry's own
        // contents, so clearing wholesale costs a rehash and never correctness.
        if self.pair_keys.len() >= PAIR_KEY_CACHE_CAP {
            self.pair_keys.clear();
        }
        let k = Self::key(gxp);
        self.pair_keys.insert(id, (k, gxp.vprog.clone(), gxp.fprog.clone()));
        k
    }

    /// The repacked bytes for a vertex stream whose ALLOCATION this renderer has not packed
    /// before - the slow half of the packed-vertex cache.
    ///
    /// Reached only on an allocation miss, which is what makes the content hash affordable:
    /// it reads the whole guest stream, and it used to do so on every draw of every frame,
    /// including the ones the cache then served.
    ///
    /// >>> A HIT IS VERIFIED AGAINST THE SOURCE BYTES, not trusted on the hash alone.
    ///
    /// The entry carries the vertex stream it was repacked FROM, and a hit that does not
    /// match it byte for byte is treated as a miss. A content-hash cache whose value is
    /// GEOMETRY fails silently when it collides - the colliding draw renders the other
    /// draw's mesh, correctly and confidently, and nothing anywhere reports it. That is
    /// exactly what happened here: the word-wise hash cancelled paired top-bit flips (see
    /// [`fnv64`]), a faded text quad collided with itself one alpha level away, and the
    /// resulting flicker was chased across several sessions through the clock, the
    /// scheduler, the display queue and the presentation path - every one of which was
    /// innocent and each of which had to be excluded by measurement.
    ///
    /// The comparison is a memcmp of the same buffer the repack would otherwise READ, so it
    /// is strictly cheaper than the work it still saves, and it makes the cache's
    /// correctness independent of the hash rather than conditional on it. The hit it exists
    /// for is a stream RE-SNAPSHOTTED into a new allocation with the same contents - the
    /// same-allocation hit is served by the caller without reaching here at all.
    pub(super) fn pack_vertices(
        packed: &mut PackedCache,
        packed_by_alloc: &mut PackedAllocCache,
        akey: (u64, usize, usize),
        gxp: &GxpRecompile,
        repack: &[RepackAttr],
        packed_stride: u32,
        epoch: u64,
    ) -> std::sync::Arc<[u8]> {
        let t_hash = split_start();
        let pkey = (akey.0, fnv64(0xcbf2_9ce4_8422_2325, &gxp.vertices));
        split_end(t_hash, &PREP.hash_ns);
        split_add(&PREP.hash_bytes, gxp.vertices.len() as u64);
        // `get_mut`, because a HIT is a USE and this cache is now evicted by last use. The
        // stamp is what makes "oldest" mean anything: without it the only orders available
        // are insertion order and hash order, and neither says which meshes this title is
        // still drawing.
        let hit = packed
            .get_mut(&pkey)
            .filter(|(src, ..)| src[..] == gxp.vertices[..])
            .map(|(src, bytes, used)| {
                *used = epoch;
                // WHY this draw is here at all rather than on the allocation fast path.
                if std::sync::Arc::ptr_eq(src, &gxp.vertices) {
                    split_add(&PREP.packed_hit_same_alloc, 1);
                } else {
                    split_add(&PREP.packed_hit_new_alloc, 1);
                }
                bytes.clone()
            });
        let bytes = match hit {
            Some(bytes) => {
                split_add(&PREP.packed_hits, 1);
                bytes
            }
            None => {
                split_add(&PREP.packed_misses, 1);
                // Was this a miss on geometry the cap itself threw away? Charged BEFORE the
                // eviction below, so a key shed by this very pass cannot be counted as a
                // thrash it caused. See `PACKED_REPACK_AFTER_EVICT`.
                note_packed_miss(pkey.1);
                let t_repack = split_start();
                // Bound the cache - but by shedding the meshes this title has stopped
                // drawing, not by dropping the ones it is drawing right now. See
                // `evict_oldest` for what the wholesale clear this replaces actually cost.
                if packed.len() >= PACKED_CACHE_CAP {
                    let n = evict_oldest_noting(
                        packed,
                        evict_target(PACKED_CACHE_CAP),
                        epoch,
                        |(_, _, used)| *used,
                        |k: &(u64, u64)| note_packed_evicted(k.1),
                    );
                    note_cache_evicted("packed geometry (by content)", n);
                }
                let mut out = Vec::new();
                repack_vertices_into(&gxp.vertices, gxp.vertex_stride, repack, packed_stride, &mut out);
                split_end(t_repack, &PREP.repack_ns);
                split_add(&PREP.repack_bytes, gxp.vertices.len() as u64);
                let out: std::sync::Arc<[u8]> = out.into();
                packed.insert(pkey, (gxp.vertices.clone(), out.clone(), epoch));
                out
            }
        };
        // Both maps hold the source `Arc`, which is what keeps the allocation alive and so
        // keeps its ADDRESS from being handed to a later stream. Bounded the same way.
        if packed_by_alloc.len() >= PACKED_CACHE_CAP {
            let n = evict_oldest(
                packed_by_alloc,
                evict_target(PACKED_CACHE_CAP),
                epoch,
                |(_, _, used)| *used,
            );
            note_cache_evicted("packed geometry (by allocation)", n);
        }
        packed_by_alloc.insert(akey, (gxp.vertices.clone(), bytes.clone(), epoch));
        bytes
    }

    /// The key a compiled WGSL MODULE is cached under: the two program blobs and nothing
    /// else.
    ///
    /// # Why not the pipeline key, which is what this used to be
    /// `key` folds in blend, depth write, depth func and the fragment-enable, because a
    /// PIPELINE is bound to all of them. The WGSL is bound to none: `link_programs` reads
    /// only the two containers, and the clip fixup after it reads only run-level knobs. So
    /// keying modules by the pipeline key compiled the same source again for every variant -
    /// and, more importantly here, made the module uncomputable until a DRAW supplied the
    /// depth state, which is exactly what stopped it being prepared when the guest's shader
    /// patcher names the pair.
    ///
    /// The keycolour diagnostic is the one thing that does vary per pipeline key (it derives
    /// a colour from it), so under that knob the module stays keyed the old way rather than
    /// have two pairs share one colour.
    pub(super) fn module_key(vprog: &[u8], fprog: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in vprog.iter().chain(fprog.iter()) {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// A content key for the GUEST VERTEX LAYOUT a draw presents: its stride and, per
    /// attribute, the four fields a `RepackAttr` is built from.
    ///
    /// Part of the pipeline cache key. A pipeline owns a repack plan that reads GUEST byte
    /// offsets, and the layout is per-DRAW state that a shader pair does not fix - see
    /// `GxpLive::pipelines` for the title that submits one pair with two layouts and what
    /// sharing a plan between them did to its HUD.
    ///
    /// The attributes are folded in the order the capture recorded them, which is the same
    /// order `build_gxp_pipeline` walks when it resolves each linked attribute, so two
    /// draws that would produce the same plan hash the same.
    pub(super) fn vertex_layout_key(gxp: &GxpRecompile) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        mix(gxp.vertex_stride as u64);
        for a in gxp.attributes.iter() {
            mix(a.reg_index as u64);
            mix(a.offset as u64);
            mix(a.gxm_format as u64);
            mix(a.components as u64);
        }
        h
    }

    pub(super) fn key(gxp: &GxpRecompile) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let depth = [
            gxp.depth_write as u8,
            (gxp.depth_func >> 22) as u8 & 0x7,
            gxp.fragment_program_enabled as u8,
            draw_dual_source(gxp) as u8,
        ];
        for b in gxp
            .vprog
            .iter()
            .chain(gxp.fprog.iter())
            .chain(gxp.fprog_patched_vprog.iter())
            .chain(gxp.blend_state.iter())
            .chain(depth.iter())
        {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Prepare the GPU resources for one recompiled draw. Returns `None` (caller falls
    /// back to fixed-function) if the pair does not link or a resource cannot be built.
    /// The pipeline CACHE KEY of one draw in a pass of this shape, and the parts of it the
    /// prepared draw carries. The one place it is computed: `prepare` (the encode) and
    /// `GxmRenderer::warm_pipelines` (the async pre-pass) both call this, so a pre-pass
    /// can never start a pipeline under a key the encode would not look up.
    ///
    /// `is_raw` answers whether a (address, width, height) is a 64-bit target this frame
    /// rendered as RAW words - see `GxmRenderer::rtt_rendered_raw`.
    pub(super) fn pipe_key(
        gxp: &GxpRecompile,
        key: u64,
        noop_keeps_depth: bool,
        color_format: wgpu::TextureFormat,
        alpha_single: bool,
        samples: u32,
        is_raw: impl Fn(&(u32, u32, u32)) -> bool,
    ) -> PipeKey {
        let raw_units: u64 = gxp
            .textures
            .iter()
            .filter(|t| {
                t.unit < 64
                    // NOT 0x1b, F16F16F16F16: a program declaring that format samples the
                    // words as four halves, filtered - it gets the surface's `Rgba16Float`
                    // companion through the ordinary rendered path (`RawSurface::float_view`).
                    && (0x1c..=0x1f).contains(&t.tex.base_format)
                    && is_raw(&(t.tex.data_addr, t.tex.width, t.tex.height))
            })
            .fold(0u64, |m, t| m | (1u64 << t.unit));
        // The units (bit u = fragment unit u, bit 32+u = vertex unit u) bound to a texture on
        // the 32-bit float seam - see `SamplerDim::TwoF32`. Their layout entries differ, so
        // the mask is part of the pipeline's identity; the module is the same.
        let f32_units: u64 = gxp
            .textures
            .iter()
            .filter(|t| t.unit < 32 && raw_units & (1u64 << t.unit) == 0 && t.tex.texel == TexelSeam::Rg32Float)
            .fold(0u64, |m, t| m | (1u64 << t.unit))
            | gxp
                .vertex_textures
                .iter()
                .filter(|t| t.unit < 32 && t.tex.texel == TexelSeam::Rg32Float)
                .fold(0u64, |m, t| m | (1u64 << (32 + t.unit)));
        // The raster key with the raw mask folded in - the SAME value goes into the
        // prepared draw (`GxpPrepared::raster`), which is what the encode looks the
        // pipeline up by again.
        let raster = Self::raster_key(gxp, noop_keeps_depth)
            ^ raw_units.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ f32_units.wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
            ^ if alpha_single { 0x5151_0000_0000_0008 } else { 0 };
        let cache_key = (key, color_format, samples, gxp.cull_mode, Self::vertex_layout_key(gxp), raster);
        PipeKey { raw_units, f32_units, raster, cache_key }
    }
}
