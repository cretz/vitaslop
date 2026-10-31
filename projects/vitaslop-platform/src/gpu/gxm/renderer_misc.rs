use super::*;

impl GxmRenderer {
    /// What the last [`GxmRenderer::encode_chain`] spent, phase by phase, over EVERY pass
    /// of the frame. Reporting one pass instead described the composite and hid the world.
    /// How full every cache in the renderer that can GROW WITH RUN LENGTH is, as one line.
    ///
    /// >>> A COST THAT SCALES WITH HOW LONG YOU HAVE PLAYED IS INVISIBLE TO EVERY OTHER
    /// >>> COUNTER HERE, because they all describe the FRAME.
    ///
    /// MEASURED on a device run of 48,703 frames: `prepare` had gone from 3.2 ms over 341
    /// draws to 12.8 ms over 406 - **9.4 us a draw to 31.5 us, three and a half times** -
    /// while every counter that frame prints stayed flat or fell: `0.0` textures uploaded,
    /// `0.0` expanded, `0.1` sampler bind groups built, `0.0` view evictions. Nothing the
    /// report carried could name it, because the thing that changed was not the frame's
    /// work, it was the SIZE of the structures that work walks.
    ///
    /// Every map here is probed per draw or per bound unit, so any one of them growing into
    /// the tens of thousands is a per-draw cache miss that no per-frame count can show. This
    /// line is what turns "it degrades as you play" into a name.
    /// >>> WHICH BUFFERS THE GUEST FLIPPED WHILE THIS FRAME WAS CAPTURED.
    ///
    /// Called by the frontend before [`Self::encode_chain`], from the same drain that takes
    /// the frame's scenes, so the two describe the same window. A frontend that has nothing
    /// to say here simply does not call it and the renderer behaves exactly as it did before
    /// this existed - see the `presented` field for what it rescues and how narrowly.
    /// >>> RELEASE RENDER TARGETS THE TITLE HAS FINISHED WITH.
    ///
    /// # What this is fixing
    /// `rtt` is keyed by the guest address of a colour surface and had no removal path at
    /// all: an entry was replaced only when the SAME address was created again. A title that
    /// moves through screens - menus, a course, a results banner - allocates its targets
    /// wherever its allocator happens to put them, so the map grows for the life of the run.
    /// MEASURED here on a 48,000-frame session: **304 targets**, in a renderer process at
    /// 1.53 GB. Every one of them owns a colour texture, a depth texture, and possibly a
    /// snapshot copy, a guest-depth companion and a pair of multisampled attachments.
    ///
    /// # Why a long TTL and not a budget
    /// Re-creating a target is not merely a re-allocation: it bumps `rtt_epoch`, which is
    /// folded into the key of EVERY cached sampler bind group naming any target, so all of
    /// them become first sightings. A budget that squeezed the live working set would pay
    /// that every frame - the disease as the cure, and this renderer has already measured
    /// what per-frame target churn costs (`rtt 3.10 created` a frame against `23.5` sampler
    /// bind groups built). So this reclaims only what is unambiguously ABANDONED: untouched,
    /// as a render destination or as a sampled texture, for [`RTT_STALE_FRAMES`]. A steady
    /// screen never reaches it and the whole function is a walk of a map with a handful of
    /// entries in it.
    pub(super) fn reclaim_stale_rtt(&mut self, now: u64) {
        // A cheap gate: the walk is O(targets) and a title with a handful of them should not
        // pay even that every frame.
        if self.rtt.len() <= RTT_KEEP_FREELY {
            return;
        }
        // >>> AN UNSTAMPED TARGET IS STAMPED, NOT RECLAIMED. A target created this frame is
        // not yet in `rtt` when the stamping above runs, so its first appearance here has no
        // stamp - and reading that as "stale" would destroy it one frame after it was built.
        // On a title that uses such a target every hundred frames that is a create/destroy
        // cycle forever, each one bumping `rtt_epoch` and invalidating every sampler bind
        // group naming any target. Far worse than the leak. Give it the clock instead.
        let unstamped: Vec<(u32, u32, u32)> =
            self.rtt.keys().filter(|k| !self.rtt_used.contains_key(*k)).copied().collect();
        for k in unstamped {
            self.rtt_used.insert(k, now);
        }
        let stale: Vec<(u32, u32, u32)> = self
            .rtt
            .keys()
            .filter(|k| {
                self.rtt_used
                    .get(*k)
                    .is_some_and(|used| now.saturating_sub(*used) > RTT_STALE_FRAMES)
            })
            .copied()
            .collect();
        if stale.is_empty() {
            return;
        }
        let mut freed = 0u64;
        for k in &stale {
            if let Some(t) = self.rtt.remove(k) {
                freed += t.bytes();
                // DESTROYED, not dropped: in the browser these are `GPUTexture`s and a drop
                // only makes them collectable - which is the whole reason this reclamation
                // is worth doing. See `RttSurface::destroy`.
                t.destroy();
                enc(&ENC.rtt_destroyed, 1);
            }
            self.rtt_used.remove(k);
            // The released target has no contents left to load, so the next creation at
            // this key clears rather than loading an undefined image - which is the rule
            // `ensure_rtt` states for a recreation. It would reach the same answer on its
            // own (a reclaimed target is a first sighting there), and doing it here also
            // keeps this set from being the one thing that still grows with run length.
            self.rtt_ever_rendered.remove(k);
            // >>> A SCAN BY ADDRESS. `rtt_binds` is keyed on the address a DRAW sampled -
            // the sampler path has no extent - so this also drops the groups naming another
            // target at the same address. Conservative and cheap: it rebuilds a bind group
            // that was still good rather than risking one over a destroyed texture.
            self.rtt_binds.retain(|&(a, _, _), _| a != k.0);
            // BY VALUE, not by key: this map is `depth address -> the target's own key`, so
            // the target being released is on the RIGHT of it. Removing `k` as a key would
            // almost never match and would leave a depth address resolving to a target that
            // no longer exists.
            self.rtt_depth_addrs.retain(|_, target| target != k);
        }
        // >>> AND EVERY CACHED VIEW OF ONE IS NOW DANGLING. `rtt_epoch` is folded into the
        // key of every sampler bind group that names a target, so bumping it is what stops
        // a cached group handing a destroyed texture to a draw. This is the expensive half
        // of the reclamation and the reason the TTL is long: it costs a rebuild of those
        // groups, once, per reclamation pass.
        self.rtt_epoch = self.rtt_epoch.wrapping_add(1);
        note_rtt_reclaimed(stale.len() as u64, freed);
        report_rtt_reclaimed(stale.len(), freed, self.rtt.len(), &stale);
    }

    /// See `offscreen_only`.
    pub fn set_offscreen_only(&mut self, on: bool) {
        self.offscreen_only = on;
    }

    /// Whether the render now being encoded is an EARLY COMPLETION rather than the display
    /// frame - see `offscreen_only`.
    ///
    /// Exposed for the chain DUMP, which otherwise writes both kinds of render into one
    /// directory under one set of filenames. An early completion runs on EVERY guest frame,
    /// including the ones a headless shot window does not render, and it renders only the
    /// small scenes - so the last writer of `rtt_<addr>.png` is almost never the frame the
    /// operator aimed at, and the world-sized target in it reads nearly black. A dump whose
    /// contents belong to a render nobody asked about is a measurement of the wrong thing.
    pub fn offscreen_only(&self) -> bool {
        self.offscreen_only
    }
    pub fn set_presented(&mut self, addrs: &[u32]) {
        self.presented.clear();
        self.presented.extend_from_slice(addrs);
    }

    /// Which image this frame shows when its LAST scene is not a display buffer - see the
    /// call site in `encode_chain`. `None` is the ordinary rule (the last scene's target).
    ///
    /// A display buffer is an address the guest has FLIPPED (`sceDisplaySetFrameBuf`) at
    /// some point in the run - its own statement, kept in `flipped_ever`. The capture does
    /// not end exactly on a flip, so a frame can close with the NEXT frame's opening pass
    /// (MEASURED, a fighting title, native run `80` f6900: `... 0x9257e000:2 0x93394000:4`, the display
    /// composite then the 128x256 silhouette target). Then:
    ///   * `Scene(a)`: the last scene drawing into a display buffer is the display, and the
    ///     trailing passes are offscreen - they still render, in order, for the next frame.
    ///   * `Hold(a)`: no scene draws into any display buffer; the most recently flipped
    ///     buffer's kept image stays on screen, as it does on the console.
    ///
    /// Only once a flipped address has equalled a scene target on this run (so the two
    /// address spaces agree - otherwise a title whose flips never match would freeze).
    /// `VITASLOP_GXM_HOLD_FLIP=0` is the arm back.
    pub(super) fn display_choice(&mut self, scenes: &[RenderScene], last_target: Option<u32>) -> Option<DisplayChoice> {
        for &a in &self.presented {
            if !self.flipped_ever.contains(&a) {
                // A title rotates a handful of display buffers; this only bounds a title
                // that allocates new ones for ever.
                if self.flipped_ever.len() >= 16 {
                    self.flipped_ever.remove(0);
                }
                self.flipped_ever.push(a);
            }
        }
        if let Some(&a) = self.presented.last() {
            self.last_flipped = Some(a);
        }
        if self.offscreen_only || !crate::gpu::hold_flip_enabled() || self.flipped_ever.is_empty() {
            return None;
        }
        let is_display = |a: u32| self.flipped_ever.contains(&a);
        let last_display_scene =
            scenes.iter().rev().find_map(|s| s.target.map(|t| t.data_addr).filter(|a| is_display(*a)));
        if last_display_scene.is_some() {
            self.presented_matched = true;
        }
        if !self.presented_matched || last_target.is_none_or(is_display) {
            return None;
        }
        let choice = match last_display_scene {
            Some(a) => DisplayChoice::Scene(a),
            None => {
                let a = self.last_flipped?;
                if !self.display_images.contains_key(&a) {
                    return None;
                }
                DisplayChoice::Hold(a)
            }
        };
        report_status!(
            "gxm display: this frame's last scene targets {:#x}, which the guest has never flipped - an offscreen pass (the capture closed on the next frame's opening pass). Showing {choice:?} instead, the buffer the console would. See `GxmRenderer::display_choice`.",
            last_target.unwrap_or(0)
        );
        Some(choice)
    }

    pub fn cache_sizes(&self) -> String {
        // The promotion maps' PRUNE, beside their occupancy: an occupancy alone cannot tell
        // a map that has never reached its cap from one that reaches it every few minutes
        // and reclaims its way back down, and those two are the difference between a run
        // that degrades and one that does not. See `prune_seen`.
        let (prunes, pruned) = seen_prune_counts();
        // What each cap has actually COST this run. A cap that never fires and a cap that
        // fires every few seconds look identical in an occupancy line, and they are the
        // difference between a session that holds 30 fps and one that reaches single
        // digits - see `evict_oldest`.
        let evictions = cache_eviction_summary();
        // >>> WHAT THE RENDER TARGETS THEMSELVES COST, which no line here has ever carried.
        // A count of 304 says nothing until it is priced; these are colour and depth
        // ATTACHMENTS, so a hundred stale ones is hundreds of megabytes of GPU memory held
        // for screens the title has left. See `reclaim_stale_rtt`.
        let rtt_mb: u64 = self.rtt.values().map(|t| t.bytes()).sum::<u64>() / (1024 * 1024);
        let (rtt_gone, rtt_gone_bytes) = rtt_reclaimed_counts();
        // How well the texture-expansion batching is doing: textures per submit. Neither
        // number means anything without the other - see `texenc::RawBatch`.
        //
        // >>> AND THE RATIO ALONE STILL CANNOT SAY. A steady window expands about a tenth
        // of a texture per frame, so ~1.0 per submit is what a healthy batcher reports when
        // there is never a second texture in the same frame. `largest` and `window-limited`
        // are the two that separate that from a batch being destroyed - see
        // `texenc::RAW_BATCH_MOST`.
        let (batch_submits, batch_textures) = crate::texenc::raw_batch_counts();
        let (batch_most, batch_multi, batch_window_one) = crate::texenc::raw_batch_shape();
        const MB: u64 = 1024 * 1024;
        let rv = self.gxp.resident_v.pinned_bytes();
        let ri = self.gxp.resident_i.pinned_bytes();
        format!(
            "renderer caches: pipelines {}, sampler bind groups {}, texture views {} ({} stamped, {} dead, {} slots), packed geometry {} by content / {} by allocation, holding {} MB by content + {} MB by allocation (the two share their buffers, so the heap pays the union, not the sum) against a {} MB budget each, resident slices {} vertex / {} index PINNING {} + {} MB of streams on the Rust heap (GPU heaps {}/{} + {}/{} MB used/cap), resident seen {} / {} ({} prunes, {} dead entries reclaimed; {} MB of DEAD streams still allocated behind their Weak), ubo bind groups {}, samplers {}, rtt targets {} holding {} MB ({} reclaimed as stale, {} MB released) ({} colour binds, {} depth addrs, {} cubes), fixed-function views {} / binds {} | RETAINED BYTES: recompiler views {} MB (entries sum to {} MB; ONE FRAME needs {} MB, learned floor {} MB) against a {} MB retention bound of a {} MB texture budget, fixed-function views {} MB | EVICTIONS: {} | GPU texture expansions {} in {} submits (largest batch {}, {} submits carried more than one, {} expansions had room for no second texture) | DEFERRED ETC2 ENCODES (`VITASLOP_TEX_ENCODE_DEFER`, {} units/frame): {}",
            self.gxp.pipelines.len(),
            self.gxp.sampler_bgs.len(),
            self.gxp.views.len(),
            self.gxp.views_used.len(),
            self.gxp.view_dead.len(),
            self.gxp.view_slots.len(),
            self.gxp.packed.len(),
            self.gxp.packed_by_alloc.len(),
            // A COUNT IS NOT A COST. These two maps were reported by ENTRY for as long as
            // they existed, and the entry figure sat quietly near its cap while the bytes
            // behind it ran to gigabytes - the same mistake `rtt targets 304` made before
            // it learned to say `holding N MB`. Reported SEPARATELY because they fill for
            // different reasons: the content map fills with the title's distinct meshes,
            // and the allocation map fills with ADDRESSES, which a title writing its
            // geometry into a rotating per-frame arena makes fresh every single frame.
            packed_map_bytes(&self.gxp.packed) / (1024 * 1024),
            packed_map_bytes(&self.gxp.packed_by_alloc) / (1024 * 1024),
            packed_cache_budget_bytes() / (1024 * 1024),
            self.gxp.resident_v.slice_count(),
            self.gxp.resident_i.slice_count(),
            rv.0 / MB,
            ri.0 / MB,
            rv.1 / MB,
            rv.2 / MB,
            ri.1 / MB,
            ri.2 / MB,
            self.gxp.resident_v_seen.len(),
            self.gxp.resident_i_seen.len(),
            prunes,
            pruned,
            // A `Weak<[u8]>` frees the CONTENTS of a dead allocation, but `[u8]` has no
            // destructor and the ALLOCATION is only returned when the last Weak goes - so a
            // promotion-map entry whose stream died still holds all of its bytes.
            self.gxp
                .resident_v_seen
                .iter()
                .chain(self.gxp.resident_i_seen.iter())
                .filter(|(_, (w, _))| w.strong_count() == 0)
                .map(|(k, _)| k.2 as u64)
                .sum::<u64>()
                / MB,
            self.gxp.ubo_bgs.len(),
            self.gxp.samplers_by_mode.len(),
            self.rtt.len(),
            rtt_mb,
            rtt_gone,
            rtt_gone_bytes / (1024 * 1024),
            self.rtt_binds.len(),
            self.rtt_depth_addrs.len(),
            self.rtt_cubes.len(),
            self.views.len(),
            self.tex_binds.len(),
            self.gxp.views_bytes / (1024 * 1024),
            // >>> THE COUNTER AUDITED AGAINST WHAT IT CLAIMS TO COUNT. `views_bytes` is a
            // RUNNING total - incremented on a miss, decremented on an eviction and on a
            // supersede-in-place - and the entries themselves carry the price each one was
            // added at. If those two disagree the retained figure is a counter defect, which
            // has the opposite fix to a cache that really is holding the bytes
            // [[vitaslop-a-hash-is-half-an-instrument]]. The sum is over `views_used`, whose
            // length is already printed beside `views.len()`, so a gap in EITHER is visible.
            self.gxp.views_used.values().map(|(_, b, ..)| *b).sum::<usize>() / (1024 * 1024),
            // What ONE FRAME actually binds, and the floor the retention bound is never
            // taken below. A retained figure is only a leak against these two.
            self.gxp.views_frame_bytes / (1024 * 1024),
            self.gxp.views_frame_high / (1024 * 1024),
            tex_retain_budget_bytes() / (1024 * 1024),
            tex_cache_budget_bytes() / (1024 * 1024),
            self.views_bytes / (1024 * 1024),
            evictions,
            batch_textures,
            batch_submits,
            batch_most,
            batch_multi,
            batch_window_one,
            crate::gpu::tex_encode_defer_units(),
            {
                use std::sync::atomic::Ordering;
                let d = crate::texenc::defer_stats();
                let pending = self.gxp.texenc.as_ref().map_or(0, |t| t.deferred_pending());
                if !crate::gpu::tex_encode_defer() {
                    "OFF - every encode inline, in the frame that binds it".to_string()
                } else {
                    format!(
                        "{} queued, {} finished ({} swapped in, {} after their entry was evicted), {} pending; {} blocks queued, {} work units encoded over {} pumps; the longest wait for an ETC2 was {} frames. While it waits a texture draws from the EXACT decode (RGBA8)",
                        d.queued,
                        d.finished,
                        crate::gpu::DEFER_SWAPPED.load(Ordering::Relaxed),
                        crate::gpu::DEFER_ORPHANED.load(Ordering::Relaxed),
                        pending,
                        d.blocks_queued,
                        d.units,
                        d.pumps,
                        d.frames_waited_max,
                    )
                }
            },
        )
    }

    /// Where the last `encode_chain` went, SUMMED over its passes - the frame total, and
    /// the only figure a caller's own `encode` total can be compared with.
    ///
    /// (Named `last_phases` until clippy pointed out that a getter of that name beside a
    /// field of that name reads as returning the last PASS. It never did.)
    pub fn chain_phases(&self) -> EncodePhases {
        self.chain_phases
    }

    /// Every offscreen target this renderer holds, as `(guest address, texture, w, h)`.
    ///
    /// For a host that can read pixels back (the native headless oracle). A frame is a
    /// CHAIN of passes and only the last one reaches the caller's view, so when the
    /// finished frame is black the question is which pass is empty - and every failure
    /// mode looks identical in the composite. `VITASLOP_CHAIN_LIMIT` answers that one
    /// pass per run; this answers all of them in one, which is the difference between a
    /// bisect and a look. The colour textures already carry `COPY_SRC` for the snapshot
    /// path, so exposing them costs nothing.
    pub fn rtt_targets(&self) -> Vec<(u32, &wgpu::Texture, u32, u32)> {
        // Four-byte targets only: every consumer (the guest writeback, the chain dump) reads
        // them back as 8-bit RGBA, and a float target has no such encoding - the guest-side
        // writeback refuses its format anyway.
        let mut v: Vec<_> = self
            .rtt
            .iter()
            .filter(|(_, s)| s.color.format().block_copy_size(None) == Some(4))
            .map(|(&(a, _, _), s)| (a, &s.color, s.width, s.height))
            .collect();
        // By the WHOLE key, not by the address: an address can hold two targets now (see
        // `rtt`), and sorting on the address alone would leave their order to hash order -
        // which is exactly the kind of run-to-run wobble a dump exists to rule out.
        v.sort_by_key(|t| (t.0, t.2, t.3));
        v
    }

    /// The FLOAT (`Rgba16Float`) colour targets - every F11F11F10 surface - as `(guest colour
    /// address, texture, w, h)`, for the chain dump. Excluded from [`Self::rtt_targets`]
    /// because they have no 8-bit encoding; without this a title's HDR chain and its
    /// image-based-lighting cubes were invisible to every dump.
    pub fn rtt_float_targets(&self) -> Vec<(u32, &wgpu::Texture, u32, u32)> {
        let mut v: Vec<_> = self
            .rtt
            .iter()
            .filter(|(_, s)| s.color.format() == wgpu::TextureFormat::Rgba16Float)
            .map(|(&(a, _, _), s)| (a, &s.color, s.width, s.height))
            .collect();
        v.sort_by_key(|t| (t.0, t.2, t.3));
        v
    }

    /// The RAW 64-BIT colour surfaces, as `(guest colour address, texture, w, h)`.
    ///
    /// Separate from [`Self::rtt_targets`] because [`Self::rtt_raw`] is a separate map, and
    /// that is why a dump built on `rtt_targets` alone has never contained one: a title's
    /// 64-bit surface is invisible to it. On a baseball title that surface is the CROWD ATLAS its
    /// sprite pass reads as two packed words a texel - the last unmeasured input of that
    /// title's washed-out close-up, and unreadable until this existed.
    pub fn rtt_raw_targets(&self) -> Vec<(u32, &wgpu::Texture, u32, u32)> {
        let mut v: Vec<_> =
            self.rtt_raw.iter().map(|(&(a, _, _), s)| (a, &s.color, s.width, s.height)).collect();
        // By the WHOLE key, for the reason `rtt_targets` gives: one address can hold two.
        v.sort_by_key(|t| (t.0, t.2, t.3));
        v
    }

    /// Whether some draw of the frame just encoded SAMPLES this target. See
    /// [`Self::frame_sampled`]; the RTT writeback consults it.
    pub fn frame_samples(&self, addr: u32) -> bool {
        self.frame_sampled.contains(&addr)
    }

    /// The guest-encoded DEPTH companions of the targets that have one, as
    /// `(colour address, texture, w, h)`.
    ///
    /// Exposed for the same reason as [`Self::rtt_targets`]: when a pass that reads a
    /// depth buffer renders black, the first question is whether the depth it read holds
    /// anything, and a converted buffer nobody can look at is a buffer nobody can debug.
    pub fn rtt_depth_targets(&self) -> Vec<(u32, &wgpu::Texture, u32, u32)> {
        let mut v: Vec<_> = self
            .rtt
            .iter()
            .filter_map(|(&(a, _, _), s)| {
                s.gxm_depth.as_ref().map(|d| (a, &d.tex, s.width, s.height))
            })
            .collect();
        // The whole key, for the reason `rtt_targets` gives.
        v.sort_by_key(|t| (t.0, t.2, t.3));
        v
    }
}
