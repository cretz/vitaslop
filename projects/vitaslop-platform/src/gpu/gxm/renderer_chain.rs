use super::*;

impl GxmRenderer {
    pub fn encode_chain(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        scenes: &[RenderScene],
        surf_w: u32,
        surf_h: u32,
        // The CALLER'S FRAMEBUFFER extent, which is NOT `surf_w/surf_h`: those are the
        // resolution the GUEST declared, and the framebuffer is the window or canvas. The
        // difference is load-bearing - a title presenting a buffer smaller than the panel
        // comes out full-screen because the resolution-independent clip output stretches to
        // fill the framebuffer, which is the hardware's own upscale for free. On the
        // headless native path the two are equal, which is why assuming so was invisible
        // there and cornered the whole picture in a browser.
        fb_w: u32,
        fb_h: u32,
        clear: [u8; 4],
        // The TEXTURE behind `color_view`, when the caller owns one. Needed only to copy the
        // attachment for a draw whose fragment program reads the DESTINATION colour, and
        // only on the arms that render straight into the caller's view - a display pass has
        // its own image and finds the texture itself. `None` means such a draw is dropped
        // with a report rather than blended against something else. See
        // `GxmRenderer::dest_color`.
        color_texture: Option<&wgpu::Texture>,
    ) {
        let _owner = crate::heap::owner_scope(crate::heap::Owner::RenderFrame);
        // Last frame's sub-rectangle copies were bound by work already submitted, so they
        // are free for this frame's copies (queue order puts the new copy after that
        // work). Kept for reuse up to a cap; the rest destroyed, never left to a collector.
        {
            let mut held: usize = self.subrect_free.values().map(Vec::len).sum();
            let pool = crate::gpu::memory_bound_on("VITASLOP_SUBRECT_POOL");
            for t in self.subrect_keep.drain(..) {
                if !pool {
                    // The old behaviour: dropped, left to the collector.
                    drop(t);
                } else if held < SUBRECT_FREE_CAP {
                    held += 1;
                    self.subrect_free.entry((t.width(), t.height(), t.format())).or_default().push(t);
                } else {
                    t.destroy();
                }
            }
        }
        // Release the previous frame's arena buffers on OUR schedule. The caller submitted
        // that frame's encoder before returning here, so their work is in flight or done
        // and `destroy()` is defined for both. See `retired_buffers` for what leaving this
        // to the JS collector measured.
        // Depth-range uniform buffers evicted during LAST frame's prepare join the same
        // queue: they were evicted mid-frame, when a draw already prepared could still name
        // them, so this is the first moment they are certainly unreferenced.
        // Prepare whatever the guest's shader patcher named since the last frame, BEFORE any
        // encoding. On the frame after a loading screen this is a burst; in a gameplay frame
        // it is empty, which is the point.
        // Once per FRAME, and the ONLY per-frame count this renderer keeps: the `gxm chain:`
        // line needs it to say which frame a shape was first seen on.
        self.chain_frames_seen += 1;
        // The deferred ETC2 encodes' slice for this frame - see `pump_deferred_encodes`.
        // Timed on the frame's clock like the transcodes themselves (`lent_ts_pair`).
        if self.gxp.texenc.as_ref().is_some_and(|t| t.deferred_pending() > 0) {
            lend_ts(self.ts.take());
            self.gxp.pump_deferred_encodes(device, queue);
            self.ts = reclaim_ts();
        }
        // >>> LAST FRAME'S STAGING CHUNKS COME BACK HERE, and here is the only place they
        // safely can. `recall` may only run once the submit that consumed them has
        // happened, and this renderer does not own the submit - the encoder is the
        // CALLER'S and it is finished and queued after `encode_chain` returns. The top of
        // the NEXT chain is therefore the first moment this crate can know the previous
        // one went through. See `GxmRenderer::gxp_staging`.
        if let Some(belt) = self.gxp_staging.as_mut() {
            belt.recall();
            // >>> AND GIVE A BURST'S CHUNKS BACK. The belt's free list never shrinks, so a
            // course load's high-water mark stayed mapped for the whole run: MEASURED in the
            // desktop browser (a golf title's GPU census `long4`) 36 chunks / 147 MB, against a
            // steady frame that uses one or two. Every ten seconds, keep a reserve and
            // destroy the rest (the vendored `trim_free`, a marked VITASLOP PATCH).
            if self.chain_frames_seen.is_multiple_of(600) && crate::gpu::memory_bound_on("VITASLOP_STAGING_TRIM") {
                let n = belt.trim_free(crate::gpu::STAGING_KEEP_CHUNKS);
                enc(&ENC.buffers_destroyed, n as u64);
            }
        }
        // THE WHOLE FUNCTION, from here, so the split can CLOSE rather than just list -
        // see `EncodePhases::chain_head_ms`. Everything above this line is comments and one
        // counter bump.
        let t_chain = Stopwatch::start();
        // The three chain-level phases - see `EncodePhases::precompile_ms`. Once per FRAME,
        // so the clock reads are free where a per-draw one would not be.
        let t_precompile = Stopwatch::start();
        for s in scenes {
            if !s.precompile.is_empty() {
                let pairs = s.precompile.clone();
                self.precompile_pairs(device, &pairs);
            }
        }
        let t_retire = Stopwatch::start();
        let precompile_ms = t_precompile.ms();
        let evicted = std::mem::take(&mut self.gxp.depth_retired);
        self.retired_buffers.extend(evicted);
        // A new frame for the store mirrors' ordering rule - see `store_mirror`.
        self.gxp.store_mirrors.frame += 1;
        // >>> DROP THE PASS ARENAS ONLY A LOAD FRAME REACHED. A slot exists per pass ORDINAL
        // (times the ring), so the count is the most passes any one frame ever opened and
        // each slot keeps the largest size it ever needed: MEASURED in the desktop browser
        // (a golf title's GPU census `long4`) ~300 `gxp-vbo` + ~250 `gxp-ubo`/`gxp-ibo`,
        // ~137 MB, sized by course loads. Trailing slots idle for `ARENA_IDLE_FRAMES` go
        // to the graveyard with the rest; their last use was hundreds of submits ago. A
        // pass that reaches that ordinal again re-creates the slot under a fresh
        // generation - see `GxpArenaSlot::generation` for why that matters.
        if self.chain_frames_seen.is_multiple_of(120) && crate::gpu::memory_bound_on("VITASLOP_ARENA_TRIM") {
            let now = self.chain_frames_seen;
            while self
                .gxp_arenas
                .last()
                .is_some_and(|a| now.saturating_sub(a.last_used) > ARENA_IDLE_FRAMES)
            {
                let a = self.gxp_arenas.pop().expect("checked by last()");
                self.retired_buffers.extend([a.vbo, a.ibo, a.ubo]);
            }
        }
        enc(&ENC.buffers_destroyed, self.retired_buffers.len() as u64);
        // >>> TIMED AND COUNTED, because freeing GPU memory is GPU-PROCESS work and the
        // stall report had no lane for it. See `RESOURCES_DESTROYED_THIS_FRAME`.
        let t_destroy = Stopwatch::start();
        let n_destroyed = self.retired_buffers.len() + self.retired_textures.len();
        // >>> KEPT, NOT FREED. Handing these back to the next frame that wants the same
        // size is what removes the allocation a later `write_buffer` would have paid for -
        // see `BufferPool` for the stall that argues it. Over the cap they are destroyed,
        // so the memory ceiling is unchanged.
        for b in self.retired_buffers.drain(..) {
            self.buffer_pool.put(b);
        }
        // The textures retired during LAST frame's encode, released now for the same
        // reason and with the same argument: the caller submitted that encoder before
        // returning here, so nothing recorded still names them.
        for t in self.retired_textures.drain(..) {
            t.destroy();
        }
        {
            use std::sync::atomic::Ordering::Relaxed;
            RESOURCES_DESTROYED_THIS_FRAME.store(n_destroyed as u64, Relaxed);
            DESTROY_US_THIS_FRAME.store((t_destroy.ms() * 1000.0) as u64, Relaxed);
        }
        let retire_ms = t_retire.ms();
        // A new frame starts at pass ordinal 0, so pass N of this frame reuses the arenas
        // pass N of the last frame used. See `gxp_arenas`.
        self.gxp_arena_slot = 0;
        // >>> AND THE FRAME ROTATES WHICH COPY OF EACH PASS'S ARENAS IT WRITES INTO.
        //
        // MEASURED on the user's phone, one gameplay window: a SINGLE
        // `queue.write_buffer` of 298 KB BLOCKED for 309.6 ms - the whole of that frame's
        // 312 ms encode - while the GPU's own timestamp query read 13.1 ms of work against
        // a 34.6 ms period. The GPU was idle for two thirds of the period and the thread
        // was parked in a copy of a third of a megabyte, so the block is not GPU execution
        // and not bytes; it is this thread waiting on the staging path. With ONE buffer per
        // pass ordinal, every frame writes the same buffer the previous frame's submit may
        // still be reading, and the write can only be staged. Rotating over `arena_ring`
        // copies gives each frame a destination whose last reader retired `ring - 1` frames
        // ago. `VITASLOP_GXP_ARENA_RING=1` restores the single-buffer behaviour and is the
        // falsifier for this.
        self.gxp_arena_phase = (self.gxp_arena_phase + 1) % self.gxp_arena_ring;
        // ...and so does `VITASLOP_GXM_DEST_SPLIT_AB`'s frame counter. Bumped HERE, once per
        // frame, so every pass in the frame reads the same arm - see `dest_split_armed`.
        crate::gpu::DEST_AB_FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(ts) = self.ts.as_mut() {
            ts.begin_chain();
        }
        if let Some(c) = self.coverage.as_mut() {
            c.begin_chain();
        }
        // ...and so does the frame's write ordinal, which is what says whether a stalled
        // write was the FIRST one after the previous submit. See `BUFFER_WRITES_THIS_FRAME`.
        // The FRAME-scoped window key set, cleared where the frame starts rather than where
        // a pass does - that difference is the whole measurement. See `GxpLive::mem_push_frame`.
        self.gxp.mem_push_frame.clear();
        BUFFER_WRITES_THIS_FRAME.store(0, std::sync::atomic::Ordering::Relaxed);
        TEX_UPLOADS_THIS_FRAME.store(0, std::sync::atomic::Ordering::Relaxed);
        TEX_UPLOAD_BYTES_THIS_FRAME.store(0, std::sync::atomic::Ordering::Relaxed);
        BUFFERS_MADE_THIS_FRAME.store(0, std::sync::atomic::Ordering::Relaxed);
        PIPELINES_BUILT_THIS_FRAME.store(0, std::sync::atomic::Ordering::Relaxed);
        CREATE_US_THIS_FRAME.store(0, std::sync::atomic::Ordering::Relaxed);
        HEAP_PAGES_AT_FRAME_START.store(heap_pages_now(), std::sync::atomic::Ordering::Relaxed);
        // >>> THE ONLY PLACE THE RESIDENT HEAPS MAY CHANGE IDENTITY.
        //
        // A prepared draw carries an OFFSET; the buffer it addresses is read at encode time.
        // Recreating a heap while a chain is in flight would silently re-point every draw
        // already prepared, so `place` only ever declines and the grow it asks for lands
        // here, between frames, with the old buffer going to the graveyard above rather than
        // being dropped - the last frame's commands still name it until its submit retires.
        let t_resident = Stopwatch::start();
        if self.gxp.resident {
            let budget = self.gxp.resident_budget;
            if let Some(old) = self.gxp.resident_v.grow_or_reset(
                device,
                queue,
                budget,
                wgpu::BufferUsages::VERTEX,
                "gxp-resident-vbo",
            ) {
                self.retired_buffers.push(old);
            }
            if let Some(old) = self.gxp.resident_i.grow_or_reset(
                device,
                queue,
                budget,
                wgpu::BufferUsages::INDEX,
                "gxp-resident-ibo",
            ) {
                self.retired_buffers.push(old);
            }
        }
        let resident_ms = t_resident.ms();
        // >>> A BATCH STILL PENDING HERE IS A MISSED FLUSH FROM THE PREVIOUS FRAME, and it
        // means that frame drew with textures nothing had written yet. The flush at the
        // BOTTOM of this function is the one that must happen; this says so if it did not.
        //
        // >>> IT LIVES HERE, ONCE PER FRAME, AND IT USED TO LIVE IN `prepare`, WHICH IS ONCE
        // >>> PER PASS. That was wrong twice over and a device dump caught both: pass 2 of a
        // frame legitimately sees pass 1's batch still pending - that IS the batching - so it
        // reported a defect that had not happened; and because it also flushed, every pass
        // boundary submitted. MEASURED on the device before this fix: `GPU texture expansions
        // 360 in 332 submits`, i.e. the per-texture submit this batching exists to remove,
        // reintroduced by its own watchdog. A frame there is 6.6 passes.
        if let Some(t) = self.gxp.texenc.as_ref() {
            let pending = t.raw_batch_pending();
            if pending > 0 {
                t.flush_raw(queue);
                report_raw_batch_not_flushed(pending);
            }
        }
        // An early completion (`offscreen_only`) is PART of the frame being built: what it
        // rendered stays "rendered this frame", so a cube whose faces completed in earlier
        // early calls is still assembled before the pass that samples it.
        if !self.offscreen_only {
            self.rtt_rendered.clear();
            self.rtt_rendered_raw.clear();
            self.rtt_depth_rendered.clear();
            self.cube_faces_copied.clear();
        }
        self.rtt_hits = 0;
        self.chain_phases = EncodePhases::default();
        // >>> ASSIGNED AFTER THE RESET ABOVE, not before it. These three run BEFORE
        // `chain_phases` is cleared, so recording them into it directly would have
        // been wiped on the same line and the residual would have stayed unnamed -
        // which is the failure this split exists to remove.
        self.chain_phases.precompile_ms = precompile_ms;
        self.chain_phases.retire_ms = retire_ms;
        self.chain_phases.resident_ms = resident_ms;
        // >>> BOUND THE PACKED-GEOMETRY CACHES IN BYTES, ONCE PER FRAME.
        //
        // Their entry cap counts meshes and says nothing about memory: on one title's
        // gameplay frame this heap grew 40-80 MB per RENDERED FRAME with the entry cap
        // firing throughout, reaching 3,751 MB in a hundred frames - which under wasm32 is
        // a dead tab. See `packed_cache_budget_bytes`.
        //
        // Here rather than at the insert, and BEFORE the epoch bump: the frame just encoded
        // is still `views_epoch`, so everything it drew is protected and only geometry the
        // title has stopped drawing is shed. A sweep per insert would walk the map on every
        // miss, which is work proportional to the frame's own geometry on the path that
        // exists to make that geometry free.
        // >>> ...BUT NOT ON EVERY FRAME, BECAUSE THE DEVICE PRICED IT AND IT IS NOT FREE.
        //
        // MEASURED on the user's phone, a baseball title's stadium: `head 1.4 ms (sweeps 1.1 + scans 0.2
        // + rtt 0.0)` - so this block and the promotion sweep below were **1.1-1.4 ms of a
        // 35 ms frame**, almost all of it spent deciding that nothing needed doing. Both
        // walks are proportional to what the caches HOLD, not to what the frame draws:
        // `packed_map_bytes` sums up to 4,096 entries per map to compare one total against a
        // budget, twice, and the promotion sweep retains over up to 8,192 entries twice.
        // ~25,000 loads a frame for a no-op. The comment below says "a sweep of 8,192
        // `strong_count` loads a frame is microseconds" - on a phone it is not, and that
        // sentence was written without a device number.
        //
        // AMORTISED, NOT REMOVED. Both are BOUNDS on memory, not correctness invariants:
        // nothing reads a stale byte because a budget was checked eight frames late, and the
        // 3,751 MB blowup and the 2 GB of dead meshes these exist to stop both took hundreds
        // of frames to build. Checking every `SWEEP_EVERY` frames keeps the bound and pays
        // an eighth of the cost. `VITASLOP_GXM_SWEEP_EVERY=1` restores per-frame behaviour
        // and is the OFF arm.
        let t_head_sweep = Stopwatch::start();
        let sweep_now = {
            let every = crate::gpu::sweep_every();
            self.chain_frames_seen.is_multiple_of(every)
        };
        if sweep_now {
            let budget = packed_cache_budget_bytes();
            let epoch = self.gxp.views_epoch;
            let n = evict_to_budget(&mut self.gxp.packed, budget, Some(epoch), |k: &(u64, u64)| {
                note_packed_evicted(k.1)
            });
            note_cache_evicted("packed geometry bytes (by content)", n);
            // >>> A HARD BOUND HERE (no protection for the frame just drawn) WAS TRIED AND
            // MEASURED: same picture, same peak, same live figure, because the map's excess
            // is this frame's own geometry either way. Not carried.
            let n = evict_to_budget(&mut self.gxp.packed_by_alloc, budget, Some(epoch), |_| {});
            note_cache_evicted("packed geometry bytes (by allocation)", n);
        }
        // >>> SWEEP THE PROMOTION MAPS OF DEAD ENTRIES EVERY FRAME, NOT AT THEIR CAP.
        //
        // A `Weak<[u8]>` pins NOTHING was the reasoning, and it is wrong for an unsized
        // buffer: the bytes live INSIDE the `Arc`'s allocation, and that allocation is
        // freed only when the last `Weak` goes, not the last `Arc`. So a dead entry here
        // holds its whole mesh in the heap. MEASURED on a baseball title's stadium (the
        // heap ledger named the site): ~30 one-shot meshes of 1.1-1.6 MB a frame, dead the
        // frame after they were filed, held until the 8,192-entry cap pruned them -
        // 2,008 MB in 1,387 buffers, dropped 1,849 MB in one frame, and climbing again.
        // A sweep of 8,192 `strong_count` loads a frame is microseconds - WHICH IS FALSE ON A
        // PHONE, see the note above. It rides the same `SWEEP_EVERY` counter now: an entry
        // dead for up to eight frames holds its mesh for eight frames instead of one, which
        // is nothing against the 2 GB this exists to stop.
        if sweep_now {
            let n = sweep_dead_seen(&mut self.gxp.resident_v_seen)
                + sweep_dead_seen(&mut self.gxp.resident_i_seen);
            if n > 0 {
                note_seen_pruned(n as u64);
            }
        }
        // The cache work ends here. This is the span to watch on a LONG run - it is
        // proportional to what the caches hold, not to what the frame draws.
        self.chain_phases.head_sweep_ms = t_head_sweep.ms();
        // A new frame: everything the texture cache holds is now a candidate for eviction
        // again. Bumped HERE and nowhere else, because "used by the frame being encoded"
        // is the one property that makes an entry un-evictable, and a frame is exactly
        // this call. See `GxpLive::views_used`.
        self.gxp.views_epoch = self.gxp.views_epoch.wrapping_add(1);
        // The frame that just finished is the evidence for how big one frame's working set
        // is. Take its high-water mark BEFORE resetting, so the floor is set by a complete
        // frame and never by a partial one.
        self.gxp.views_frame_high = self.gxp.views_frame_high.max(self.gxp.views_frame_bytes);
        self.gxp.views_frame_bytes = 0;
        let Some(last) = scenes.last() else { return };
        // THE FRAME'S SCANS - three walks of every draw's every sampler, below. Timed as one
        // span because they answer one question each and the fix for all three is the same
        // one (fold them into a single walk), so splitting them further would not change
        // what to do about the number.
        let t_head_scan = Stopwatch::start();
        // Which scenes' DEPTH some draw in this frame samples. Scanned up front because
        // the reader comes AFTER the writer: the pass that fills a depth buffer has to
        // know, while it is being set up, that something later will read it - a target
        // built without `TEXTURE_BINDING` cannot be sampled at all, and the extra
        // conversion is not worth paying on the targets nothing reads.
        //
        // >>> ONE WALK, PRODUCING ALL THREE OF `sampled`, `sampled_extents` AND `cube_bases`.
        //
        // These were three separate traversals of every draw's every sampler, and the third
        // - `sampled_extents` - ran a WHOLE traversal PER sampled render-target address,
        // which on a baseball title's stadium is ~50 of them (`rtt-samples=252`). So the
        // head of the frame walked the draw list up to fifty-two times to build three
        // things that one pass over it can build, in the same order, with the same
        // first-match-wins rule. MEASURED in the browser before this: `head 0.5-0.6 ms` of
        // an 11 ms frame, all of it here.
        //
        // BOTH stages, in `sampled`. A vertex program that samples a depth surface (a
        // displacement or reprojection built in the vertex stage) names it exactly as a
        // fragment one does, and leaving vertex samplers out of this set makes the pass that
        // WROTE that depth skip the conversion - so the reader gets raw hardware depth with
        // nothing saying so.
        let mut sampled: HashSet<u32> = HashSet::default();
        let mut sampled_extents: HashMap<u32, (u32, u32, bool)> = HashMap::default();
        let mut cube_bases: HashSet<u32> = HashSet::default();
        for d in scenes.iter().flat_map(|s| s.draws.iter()) {
            // The fixed-function samplers FIRST and the recompiled ones after, which is the
            // order the per-address search used, so "the first draw that binds this address"
            // resolves to the same texture it did before.
            for t in d.texture.iter() {
                if sampled.insert(t.data_addr) && !self.rtt_extents_at(t.data_addr).is_empty() {
                    sampled_extents
                        .insert(t.data_addr, (t.width, t.height, t.guest_bytes_unwritten));
                }
            }
            for g in d.gxp.iter() {
                for t in g.textures.iter().chain(g.vertex_textures.iter()) {
                    let a = t.tex.data_addr;
                    if sampled.insert(a) && !self.rtt_extents_at(a).is_empty() {
                        sampled_extents.insert(
                            a,
                            (t.tex.width, t.tex.height, t.tex.guest_bytes_unwritten),
                        );
                    }
                    // A CUBE MAP this frame samples, by the address of face 0. The guest's
                    // own texture declares it - `faces == 6` is how a cube arrives from the
                    // capture - so this is a statement, not a guess about the layout.
                    if t.tex.faces == CUBE_FACES && a != 0 {
                        cube_bases.insert(a);
                    }
                }
            }
        }
        // The EXTENT each draw binds a RENDER TARGET's address at - see the stamping below
        // for why an address alone cannot say whether a target is still what lives there.
        //
        // >>> DERIVED FROM THE SET ABOVE, NOT FROM A SECOND WALK OF THE FRAME.
        //
        // This is a DIAGNOSTIC, and it went through two worse versions before this one: a
        // `HashMap` insert for every bound texture in the frame (several hundred a frame,
        // on a phone, to answer a question about the handful in `rtt`), then the same walk
        // with a filter. Both re-traversed every draw's every sampler for a report that
        // changes nothing. A diagnostic that scales with the frame's working set is exactly
        // the cost this renderer keeps finding in the code it replaces
        // [[vitaslop-a-diagnostic-can-bury-the-findings]].
        //
        // `sampled` has already visited them all, and `rtt` is small, so the whole thing is
        // one pass over the DISTINCT addresses - a few hundred at most - and it stops at
        // the first hit for each. The extent has to come from a draw, so it is looked up
        // only for the addresses that survive the `rtt` test, which is a handful.
        // The bound extent AND whether the guest's own bytes at that address are all zero.
        // The second is what separates the two things an extent mismatch can mean - see
        // `report_rtt_extent_mismatch`. `all(|b| *b == 0)` short-circuits on the first
        // non-zero byte, so the ordinary case (a real texture, non-zero almost immediately)
        // costs a handful of bytes; only a genuinely empty buffer is scanned through, and
        // this whole loop runs for the handful of sampled addresses that are in `rtt`.
        //
        // The `rtt` test is by ADDRESS - `rtt_extents_at` - because that walk is what
        // FINDS the extent; it has none to key on yet.
        self.frame_sampled.clone_from(&sampled);
        let depth_sampled: HashSet<u32> = scenes
            .iter()
            .map(|s| s.depth_addr)
            .filter(|a| *a != 0 && sampled.contains(a))
            .collect();
        // The separate STENCIL planes this frame samples, each with the SWIZZLE1 selector
        // of the U8 texture its first sampler declares - see `convert_gxm_stencil`.
        self.stencil_sampled.clear();
        let stencil_addrs: HashSet<u32> =
            scenes.iter().map(|s| s.stencil_addr).filter(|a| *a != 0 && sampled.contains(a)).collect();
        if !stencil_addrs.is_empty() {
            for d in scenes.iter().flat_map(|s| s.draws.iter()) {
                for g in d.gxp.iter() {
                    for t in g.textures.iter().chain(g.vertex_textures.iter()) {
                        if t.tex.base_format == 0 && stencil_addrs.contains(&t.tex.data_addr) {
                            // `swizzle` is the format's raw swizzle bits; SWIZZLE1 is 14:12.
                            self.stencil_sampled.entry(t.tex.data_addr).or_insert((t.tex.swizzle >> 12) & 0x7);
                        }
                    }
                }
            }
        }
        // `cube_bases` is collected in the walk above, up front for the same reason
        // `sampled` is: the pass that draws the last face has to be followed by the assembly
        // before the pass that reads the cube, and both sides are known only by looking at
        // the whole frame.
        //
        // NOTE there is no "may this buffer be averaged" test here any more, and its
        // absence is the point. The old one asked how a target was FILTERED - linear for
        // an image, point for data - because the pass was a 2x2 SUPERSAMPLE whose box
        // resolve really did mix values that must not be mixed. It was refuted: this
        // title's shadow map is linear-sampled and the race regressed anyway.
        //
        // The question dissolves once the pass multisamples instead of supersamples. MSAA
        // shades ONCE per pixel and hands that one value to every covered sample, so a
        // resolve inside a triangle returns it exactly, whatever the buffer means. Only
        // pixels on a silhouette mix, which is where the hardware mixes them too. So the
        // guest's `multisampleMode` is the whole decision and no heuristic stands next to
        // it. See `gxm_sample_count`.
        // The display buffer is whatever the final scene draws to. Any earlier scene
        // naming the same address is part of the same image, not an offscreen pass.
        let display = if self.offscreen_only { None } else { last.target.map(|t| t.data_addr) };
        // >>> ...UNLESS THAT SCENE IS NOT A DISPLAY BUFFER AT ALL.
        //
        // "The last scene's target is the display" put an OFFSCREEN pass on the screen
        // whenever a captured frame ended on one: MEASURED on a fighting title's K.O.
        // (browser run `web1`, native runs `79` f6600/f7050 and `80` f6900) the frame closes
        // with the NEXT frame's 128x256 fighter-silhouette pass at 0x93394000, so the whole
        // screen became that silhouette - natively on black, in the browser as a black box
        // over the last picture. The console shows the buffer the guest FLIPPED.
        //
        // So the display is the last scene into an address the guest has flipped, the
        // passes after it render offscreen (the next frame samples them), and a frame that
        // drew no display buffer at all keeps the flipped buffer's image. Narrow on purpose
        // - see `display_choice`; an ordinary frame ends on its display buffer and is
        // untouched.
        let choice = if self.ss_scale > 1 { None } else { self.display_choice(scenes, display) };
        let held_flip = match choice {
            Some(DisplayChoice::Hold(a)) => Some(a),
            _ => None,
        };
        let display = match choice {
            Some(DisplayChoice::Scene(a)) => Some(a),
            Some(DisplayChoice::Hold(_)) => None,
            None => display,
        };
        // The faces of every cube already assembled from renders. Known from `rtt_cubes`,
        // which persists, so this is empty only until the first cube is built.
        let cube_faces: HashSet<u32> = self
            .rtt_cubes
            .iter()
            .flat_map(|(base, c)| c.face_targets(*base).into_iter().map(|(a, _)| a))
            .collect();
        report_world_not_on_display(scenes, display, &sampled, &cube_faces, &mut self.orphan_candidate);
        // >>> ...AND SO IS ANY SCENE DRAWING INTO A BUFFER THE GUEST FLIPPED THIS FRAME.
        //
        // See `GxpLive`'s `presented` field for the defect this closes. The rule above is
        // right for a frame that belongs to one image and drops a whole pass on one that
        // STRADDLES A FLIP, which a title rotating three display buffers produces whenever
        // its scene list crosses the boundary: the world goes into buffer A, the HUD into
        // buffer B, and A is classified as an offscreen target nothing reads. That is a HUD
        // over black.
        //
        // Deliberately NARROW. It fires only for a target the GUEST ITSELF presented (its
        // own statement that the buffer is a display buffer - not a guess from the extent),
        // that nothing in this frame samples, and that is not a cube face. A frame with one
        // display buffer - which is every ordinary frame - produces an empty set here and is
        // encoded exactly as it was before this existed.
        let extra_display: HashSet<u32> = match display {
            Some(d) if self.presented.len() > 1 => scenes
                .iter()
                .filter_map(|s| s.target.map(|t| t.data_addr))
                .filter(|a| {
                    *a != d
                        && self.presented.contains(a)
                        && !sampled.contains(a)
                        && !cube_faces.contains(a)
                })
                .collect(),
            _ => HashSet::default(),
        };
        if !extra_display.is_empty() {
            report_frame_straddles_a_flip(&extra_display, self.presented.len());
        }
        self.chain_phases.head_scan_ms = t_head_scan.ms();
        // ...and the RENDER-TARGET BOOKKEEPING, which is proportional to the live targets.
        let t_head_rtt = Stopwatch::start();
        // >>> EVERY RENDER TARGET THIS FRAME TOUCHES, STAMPED, AND THE STALE ONES
        // >>> RECLAIMED. See `rtt_used` for the 304-target measurement this is for.
        //
        // "Touched" is deliberately both halves: a target this frame RENDERS into (a scene
        // target, or the depth address of one) and a target this frame SAMPLES (`sampled`
        // is every texture address any draw binds, and it is already computed above for the
        // depth question). A static shadow map rendered once at load and sampled every frame
        // after is touched by the second half alone, and stamping only renders would throw
        // it away and rebuild it - which is worse than the leak.
        {
            let now = self.gxp.views_epoch;
            // >>> ONLY ADDRESSES THAT ARE ACTUALLY TARGETS. `sampled` is every texture
            // address any draw binds - hundreds a frame, nearly all of them ordinary guest
            // textures - so stamping all of them would give this map an entry per distinct
            // address the title ever binds, i.e. exactly the unbounded growth with run
            // length that the map exists to fix. A target created THIS frame is not in
            // `rtt` yet; `reclaim_stale_rtt` stamps anything it finds unstamped rather than
            // treating it as stale, which is what gives a new target its full TTL.
            //
            // >>> AND THE EXTENT IS CARRIED WHEREVER THE CALLER HAS ONE. A scene's target
            // states its own; a depth-only pass's target is keyed by the depth address and
            // `scene.depth_extent` - see `encode_depth_only_pass`. See `stamp_rtt_used`,
            // which falls back to every target at the address only when there is no honest
            // extent to give it.
            for s in scenes {
                if let Some(t) = s.target {
                    self.stamp_rtt_used(t.data_addr, Some((t.width, t.height)), now);
                }
                if s.depth_addr != 0 {
                    self.stamp_rtt_used(s.depth_addr, s.depth_extent, now);
                }
            }
            // >>> A SAMPLE ONLY COUNTS AS A USE IF THE EXTENTS AGREE, and that test is
            // >>> doing more work than keeping a map tidy.
            //
            // `rtt` is keyed by the guest address of a colour surface, and the guest reuses
            // memory: a target it has finished with can be freed and an ordinary TEXTURE
            // allocated at the same address. When that happens the address is in `rtt` AND
            // bound as a texture, so stamping on the address alone marks a dead target live
            // for ever - MEASURED on the device: `rtt targets 357 holding 276 MB (0
            // reclaimed as stale)` in a 78-second window, with nothing aging out at all.
            //
            // The extent is the cheapest thing that tells the two apart: a pass sampling a
            // real render target binds it at the size it was rendered, while a fresh texture
            // that merely inherited the address is whatever size the guest made it. So an
            // address whose bound extent DISAGREES with the target's is not a use of that
            // target, it goes unstamped, and the reclamation takes it within the TTL.
            //
            // That matters beyond memory: `sample_views` offers every `rtt` entry to the
            // sampler path, so while such a target lives, a draw binding that address is
            // handed the OLD TARGET's pixels instead of the guest's texture. Reclaiming it
            // is what stops that. See `report_rtt_extent_mismatch` - the report fires first
            // and names it, because this is a suspicion the device has to confirm.
            self.rtt_alias_block.clear();
            // >>> "THE" TARGET AT AN ADDRESS IS NOW A SET OF THEM, and the test is whether
            // >>> the bound extent matches NONE of them.
            //
            // `rtt` is keyed by `(address, width, height)` (see its doc comment: one colour
            // address, two live render targets, every frame), so a draw binding one of the
            // two shapes held at an address is sampling a LIVE target and must not be read
            // as a recycled allocation. Only a draw whose extent matches none of them is
            // the recycling this test is looking for. The report names the smallest extent
            // held there, which for the single-target case - every title measured when this
            // report was written - is exactly what it printed before.
            for (a, (w, h, guest_empty)) in sampled_extents.iter() {
                let held = self.rtt_extents_at(*a);
                let Some(&first) = held.first() else { continue };
                if !held.contains(&(*w, *h)) {
                    report_rtt_extent_mismatch(*a, first, (*w, *h), *guest_empty);
                    // >>> AN EXTENT MISMATCH ALONE DOES NOT DECIDE THIS, AND TREATING IT AS
                    // >>> IF IT DID COST A TITLE ITS ARTWORK.
                    //
                    // Two different things produce a mismatch, and the repair for one is the
                    // bug for the other:
                    //   * the guest FREED the target and allocated an ordinary texture over
                    //     it - then its bytes are in guest memory and the draw must sample
                    //     those, which is what blocking the alias achieves;
                    //   * the guest is sampling a LIVE target through a descriptor of
                    //     another size - a 1024x1024 declaration over an 840x476 target is
                    //     a title rendering into part of a power-of-two texture, and there
                    //     is nothing wrong with it.
                    //
                    // Guest memory tells them apart, and it is the one witness that does: a
                    // render target's pixels live on the GPU and read as ZEROS in guest
                    // memory [[vitaslop-a-render-target-reads-empty-in-guest-memory]],
                    // whereas a recycled allocation has the guest's real bytes in it. So an
                    // all-zero buffer is not evidence of a dead target - it is evidence of a
                    // LIVE one, and blocking it hands the draw an empty texture.
                    //
                    // MEASURED on a role-playing title, which is what found this: its title-screen logo
                    // is rendered into an 840x476 target and sampled through a 1024x1024
                    // descriptor, and blocking the alias drew the logo as an untextured
                    // gradient band. The 192x192-against-1024x1024 cases the block was built
                    // for are unaffected - those buffers have guest bytes.
                    if !*guest_empty {
                        // >>> A RECYCLED ALLOCATION IS NOT A USE OF THIS TARGET, so it goes
                        // unstamped and the reclamation takes it within the TTL - which is
                        // what ends the aliasing. The device dump justifies this: the report
                        // ran unconditionally for a session first, precisely so the decision
                        // would rest on evidence, and it fired for ~20 addresses.
                        self.rtt_alias_block.insert(*a);
                        continue;
                    }
                }
                // >>> SAMPLING A LIVE TARGET IS A USE, AND IT WAS NOT BEING COUNTED AS ONE.
                //
                // The note above this block says "touched" is deliberately both halves -
                // rendered into OR sampled - and names the static shadow map, rendered once
                // at load and sampled every frame after, as the case the second half is for.
                // The loop then `continue`d past the stamp for EVERY sampled address, so
                // that half never happened: such a target aged out after its TTL and was
                // rebuilt, which the note itself calls worse than the leak.
                //
                // It matters more now that an empty-guest mismatch aliases: a logo rendered
                // once and sampled thereafter would otherwise vanish a minute into a run,
                // which no short replay would ever show.
                //
                // The DECLARED extent is handed over: when it names one of the targets held
                // here that one is stamped and the other ages on its own clock, which is the
                // whole point of keying on the extent. When it names none (the legitimate
                // odd-descriptor case above, which reaches here only when the guest's bytes
                // are empty) every target at the address is stamped, because the sampler
                // path may be reading any of them - see `stamp_rtt_used`.
                // A superseded copy is not what the draw reads (the display image is), so it
                // is not a use either - it ages out and its memory is reclaimed.
                if self.rtt_superseded.contains(a) {
                    continue;
                }
                self.stamp_rtt_used(*a, Some((*w, *h)), now);
            }
            // The six faces of every assembled cube. Taken from `rtt_cubes` rather than
            // from the `cube_faces` address set beside it, because the cube RECORDS its
            // face extent (`CubeFromRenders::size`, square by construction) - so these
            // stamp the exact targets they mean instead of every target at the address.
            let faces: Vec<(u32, (u32, u32))> = self
                .rtt_cubes
                .iter()
                .flat_map(|(base, c)| c.face_targets(*base))
                .collect();
            for (a, e) in faces {
                self.stamp_rtt_used(a, Some(e), now);
            }
            self.reclaim_stale_rtt(now);
        }
        self.chain_phases.head_rtt_ms = t_head_rtt.ms();
        let ss = self.ss_scale > 1;
        if ss {
            self.ensure_ss_target(device, queue, surf_w, surf_h);
        }
        let mut display_pass_done = false;
        // The display address whose kept image has to be blitted into the caller's
        // framebuffer after the loop. `None` when supersampling owns the path instead.
        let mut display_blit_addr: Option<u32> = None;
        let n = scenes.len();
        // HEAD ends here and the LOOP begins - the two spans that let the split close.
        // `pass_wall_ms` accumulates inside `encode_pass` itself rather than around the four
        // call sites here, so a fifth one cannot be added without being measured, and the
        // depth-only pass's own setup lands in the loop residual where it belongs.
        self.chain_phases.chain_head_ms = t_chain.ms();
        let t_scene_loop = Stopwatch::start();
        for (i, scene) in scenes.iter().enumerate() {
            // BEFORE this pass is encoded, and so after every earlier one: if the faces of
            // a cube this frame samples have all been rendered by now, copy them into their
            // cube texture. Between passes is the only correct place - the copies must
            // follow the passes that drew the faces and precede any pass that reads them,
            // and they go on THIS encoder for the same reason.
            if !cube_bases.is_empty() {
                self.assemble_rendered_cubes(device, encoder, &cube_bases);
            }
            let to_display = !self.offscreen_only && choice.is_none() && (i + 1 == n)
                || (scene.target.map(|t| t.data_addr) == display && display.is_some())
                // The straddled case - see `extra_display`. The pass is composited into the
                // SAME kept image as the rest of the frame, in scene order, so the world
                // clears it and the HUD lands on top, which is the picture the console shows
                // across the two flips.
                || (!self.offscreen_only && scene.target.is_some_and(|t| extra_display.contains(&t.data_addr)));
            if to_display {
                // >>> THE FINISHED DISPLAY IMAGE IS KEPT, because a title may SAMPLE IT ON
                // >>> A LATER FRAME - see `display_images` for the defect and for why this
                // >>> is a map of its own rather than an `rtt` entry.
                //
                // The pass renders into that image and the blit after the loop puts it in
                // the caller's framebuffer, so what reaches the screen is unchanged. The
                // supersample path keeps its own target: its attachment is `ss_scale` times
                // the size and its resolve downsamples, so `ss_scale > 1` is untouched.
                // >>> WHICH COLOUR VIEW THIS PASS RENDERS THROUGH, and it is a THIRD member
                // >>> of the tuple rather than a decision taken later, for the reason the
                // >>> extent below is: a view chosen in parallel with the thing it views is
                // >>> the bug this arm already learned once.
                //
                // A GAMMA-CORRECT display surface holds sRGB-ENCODED bytes, because the
                // console's ROP encodes every store after blending. Rendering it through
                // the linear view keeps whatever the shader produced, which is DARKER than
                // the title intends on every pixel of every draw - see
                // `report_gamma_display_surface`. MEASURED on this title: it enables gamma
                // writes on both of its 960x544 display surfaces, and the user's report is
                // "everything feels dark".
                let want_srgb = scene.target.is_some_and(|t| t.gamma);
                // Whether the kept image was (re)created just now, so holds nothing yet.
                let mut fresh_image = false;
                let disp = match (ss, display) {
                    (false, Some(addr)) => {
                        // At the FRAMEBUFFER's extent, so the pass rasterises exactly where
                        // it used to and the blit below stays an exact 1:1 copy. Sizing it
                        // to the guest surface instead both lost resolution and left the
                        // image in the corner of a larger framebuffer.
                        fresh_image = self.ensure_display_image(device, addr, fb_w, fb_h);
                        self.display_images.get(&addr).map(|d| {
                            // The sRGB view only when the guest asked AND the format has a
                            // twin. When it asked and there is none, the report below says
                            // the mode could not be honoured rather than this arm quietly
                            // picking the linear view as it always used to.
                            let srgb = want_srgb.then_some(d.view_srgb.as_ref()).flatten();
                            (addr, srgb.unwrap_or(&d.view).clone(), d.depth_view.clone(), srgb.is_some())
                        })
                    }
                    _ => None,
                };
                // >>> THE EXTENT COMES OUT OF THE SAME ARM THAT PICKS THE VIEW.
                //
                // Every rectangle this pass states in guest pixels is mapped onto this, so
                // it has to be what `cv` actually IS, not a second derivation that agrees
                // with it today. Deriving an attachment extent in parallel with the
                // attachment is precisely the bug this whole change fixes; doing it again
                // one level up would leave the same trap for the next arm added here.
                let (cv, dv, att_w, att_h) = match (&disp, ss, self.ss_target.as_ref()) {
                    // The depth is the image's OWN, at the image's extent - see
                    // `ensure_display_image`. Nothing samples a display pass's depth (the
                    // report below says so when something tries), so a private one costs
                    // only its bytes and is the only thing a render pass will accept.
                    (Some((_, v, dv, _)), _, _) => (v.clone(), dv.clone(), fb_w, fb_h),
                    (None, true, Some(t)) => {
                        (t.color_view.clone(), t.depth_view.clone(), t.width, t.height)
                    }
                    _ => (color_view.clone(), depth_view.clone(), fb_w, fb_h),
                };
                if let Some((addr, _, _, _)) = &disp {
                    display_blit_addr = Some(*addr);
                    // This pass writes the buffer's memory, so every offscreen copy held at
                    // the address is now older than it - see `rtt_superseded`. The epoch is
                    // bumped only when the mark is NEW, so a title presenting through the
                    // same buffers every frame pays it once, not per frame.
                    let addr = *addr;
                    self.rtt_rendered.retain(|k, _| k.0 != addr);
                    if self.rtt.keys().any(|k| k.0 == addr) && self.rtt_superseded.insert(addr) {
                        self.rtt_epoch = self.rtt_epoch.wrapping_add(1);
                        self.rtt_binds.retain(|&(a, _, _), _| a != addr);
                    }
                }
                let first = !display_pass_done;
                display_pass_done = true;
                self.rtt_reads_snapshot.clear();
                // A pass that SAMPLES the buffer it draws into reads a snapshot of what is
                // already there - the offscreen arm's rule (`samples_own_target`), for the
                // display image. A freshly created image holds nothing, so there is nothing
                // to copy and the read keeps its old path.
                //
                // >>> ONLY WHEN EVERY SUCH READ DECLARES THE SURFACE'S OWN EXTENT. The image
                // is the whole surface at the caller's resolution, so it answers a read of
                // the whole surface and nothing else; a different extent at the same address
                // is another surface (the stale-extent rule for `rtt`). MEASURED on a capsule
                // whose last scene is a 512x128 atlas (no flip in a capsule, so the atlas was
                // taken as the display): a 256x128 read of it was handed the whole image.
                self.display_reads_snapshot = None;
                let own_reads: Vec<(u32, u32)> = match (&disp, scene.target) {
                    (Some((addr, _, _, _)), Some(_)) => scene
                        .draws
                        .iter()
                        .flat_map(|d| {
                            d.texture.iter().map(|tx| (tx.data_addr, tx.width, tx.height)).chain(
                                d.gxp.iter().flat_map(|g| {
                                    g.textures
                                        .iter()
                                        .chain(g.vertex_textures.iter())
                                        .map(|tx| (tx.tex.data_addr, tx.tex.width, tx.tex.height))
                                }),
                            )
                        })
                        .filter(|(a, _, _)| a == addr)
                        .map(|(_, w, h)| (w, h))
                        .collect(),
                    _ => Vec::new(),
                };
                if let (Some((addr, _, _, _)), Some(t)) = (&disp, scene.target)
                    && !fresh_image
                    && crate::gpu::display_snapshot()
                    && !own_reads.is_empty()
                    && own_reads.iter().all(|e| *e == (t.width, t.height))
                {
                    let addr = *addr;
                    self.display_reads_snapshot = self.snapshot_display(device, encoder, addr).map(|v| (addr, v));
                }
                // The display target's format belongs to the surface the host handed us,
                // and the host owns whether that is already an sRGB swapchain - so a
                // gamma-mode DISPLAY surface is not reinterpreted here.
                //
                // >>> AND THAT IS A SILENT PICTURE CHANGE IF THE GUEST EVER ASKS FOR ONE.
                //
                // The OFFSCREEN arm honours `SCE_GXM_COLOR_SURFACE_GAMMA_*`: it renders
                // through the target's sRGB view, so the ROP sRGB-encodes each store after
                // blending, which is what the hardware does. This arm does not, and until
                // now it did not say so either. A display surface in gamma mode therefore
                // stores LINEAR bytes where the console stores ENCODED ones, and a linear
                // 0.5 shown where 0.73 belongs is the whole frame reading DARK - which is
                // exactly the user's report on this title ("gameplay probably too dark",
                // "everything feels dark"), and it is global, which no per-draw defect is.
                //
                // NOT fixed here on the strength of that reasoning: this is a report, and
                // whether it ever fires is the evidence. The offscreen arm's own report
                // (`report_gamma_surface`) has fired for one other title's 256x1536 target,
                // so the guest API is certainly used; whether THIS title uses it on the
                // DISPLAY surface is what nothing has ever asked. When it fires, the fix is
                // the offscreen arm's, applied here: give `ensure_display_image` an sRGB
                // view (`view_formats`) and select it and its format on this line.
                // Whether the sRGB view was actually taken. `false` with `want_srgb` set is
                // the case that still cannot be honoured - a supersampled pass, a bare host
                // view, or a colour format with no sRGB twin - and the report names it.
                let srgb_taken = matches!(disp, Some((_, _, _, true)));
                // The image now holds encoded bytes (or has stopped holding them, if the
                // guest turned the mode off under a surface it is still using - which
                // `ensure_rtt` documents as real, sticky guest state). Recorded on the
                // image so the sampler path reads the IMAGE's state, not a scene's.
                if let Some((addr, _, _, _)) = disp.as_ref()
                    && let Some(d) = self.display_images.get_mut(addr) {
                        d.gamma = srgb_taken;
                    }
                if want_srgb {
                    report_gamma_display_surface(
                        scene.target.map(|t| t.data_addr).unwrap_or(0),
                        att_w,
                        att_h,
                        srgb_taken,
                    );
                }
                // A pipeline is bound to the FORMAT of the attachment it writes, so a pass
                // rendering through the sRGB view needs its pipelines built for the sRGB
                // format - the same rule the offscreen arm follows.
                let fmt = match srgb_taken {
                    true => srgb_twin(self.color_format).unwrap_or(self.color_format),
                    false => self.color_format,
                };
                // The DISPLAY buffer. Its render target is the one the guest creates at
                // boot, and on every title measured here it is `SCE_GXM_MULTISAMPLE_NONE`
                // - the console composites the front buffer at one sample. So this pass
                // has no resolve and does not ask for one.
                // The TEXTURE behind the display attachment, when the renderer owns one.
                // A supersampled pass or a bare host view has none, and a draw that reads
                // the destination colour is dropped with a report rather than served wrong.
                let ctex = match disp.as_ref() {
                    Some((addr, _, _, _)) => self.display_images.get(addr).map(|d| d.tex.clone()),
                    // No display image: the pass renders straight into the caller's view,
                    // and only the caller can name the texture behind it.
                    None => color_texture.cloned(),
                };
                // >>> THE DISPLAY IMAGE LOADS ACROSS FRAMES; IT IS CLEARED ONLY WHEN IT IS
                // >>> NEW. The tiler loads the colour surface at `sceGxmBeginScene` for the
                // display buffer exactly as it does for a render target (see
                // `rtt_ever_rendered`), and a title that wants a clear draws one.
                //
                // MEASURED on a baseball title's loading screen (2026-09-18): its pitching
                // matchup card is drawn for ~650 frames into the three display buffers, then
                // each frame draws ONE quad - the spinner - into the buffer it presents, for
                // ~500 frames. On hardware the card stays behind the spinner (the user's
                // report); cleared every frame, the screen dropped to the frontend's
                // backdrop colour with a spinner in the corner. Twice before this arm was
                // made to persist and was refuted (a racer lost its boost chevrons, which
                // blend against the destination; a rally title moved thin rails) - both are
                // draws that READ the destination, and both were measured before a pass
                // was split at every destination read (`copy_attachment_to_dest`): the
                // destination they saw was the pass's START, which under persistence was
                // the previous frame. Re-shot under the split: see the 18b notes.
                //
                // A pass into a bare host view or a supersampled target has no kept image
                // to load, and keeps clearing.
                let clear_now = first && (fresh_image || disp.is_none());
                // A MULTISAMPLED display render target rasterises into the image's own
                // multisampled attachments and resolves into the kept image - see
                // `ensure_display_msaa`. Not for a gamma surface, for the offscreen arm's
                // reason (`ensure_rtt`): which values a resolve averages there is unmeasured.
                let want_samples = match (disp.as_ref(), scene.target) {
                    (Some(_), Some(t)) if !no_multisample() && !want_srgb => gxm_sample_count(t.multisample),
                    _ => 1,
                };
                let msaa_pass = match disp.as_ref() {
                    Some((addr, _, _, _)) if want_samples > 1 => {
                        let addr = *addr;
                        self.ensure_display_msaa(device, addr, want_samples);
                        self.display_images
                            .get(&addr)
                            .and_then(|d| d.msaa.as_ref())
                            .map(|m| (m.color_view.clone(), m.depth_view.clone(), m.samples, m.color.clone()))
                    }
                    _ => None,
                };
                // A stencil plane a fragment program STORED to, FORCE_LOADed from (and
                // FORCE_STOREd back into) its store mirror - see `store_mirror`. Only through
                // the kept image's own single-sample depth; every other attachment keeps the
                // background clear it always had.
                let guest_dtex = match (&msaa_pass, disp.as_ref()) {
                    (None, Some((addr, ..))) => self.display_images.get(addr).map(|d| d.depth.clone()),
                    _ => None,
                };
                let guest_store =
                    self.begin_guest_stencil(device, queue, encoder, scene, guest_dtex.as_ref(), (surf_w, surf_h));
                let keep_stencil_was = self.keep_stencil;
                self.keep_stencil = keep_stencil_was || guest_store;
                match msaa_pass {
                    // The multisampled attachment is what a destination read copies, sample
                    // for sample - see `copy_attachment_to_dest`.
                    Some((mcv, mdv, samples, mtex)) => self.encode_pass(device, queue, encoder, &mcv, &mdv, fmt, scene, surf_w, surf_h, att_w, att_h, clear_now.then_some(clear_color(clear, srgb_taken)), samples, Some(&cv), Some(&mtex)),
                    None => self.encode_pass(device, queue, encoder, &cv, &dv, fmt, scene, surf_w, surf_h, att_w, att_h, clear_now.then_some(clear_color(clear, srgb_taken)), 1, None, ctex.as_ref()),
                }
                self.keep_stencil = keep_stencil_was;
                self.stencil_load_now = false;
                if let (true, Some(d)) = (guest_store, guest_dtex.as_ref()) {
                    self.store_guest_stencil(device, queue, encoder, scene, d, (surf_w, surf_h));
                }
                self.display_reads_snapshot = None;
                // A display pass keeps no depth copy (its depth attachment belongs to the
                // caller and is discarded), so if something reads this scene's depth it
                // will not find it. Say so rather than let the read fall through silently.
                if depth_sampled.contains(&scene.depth_addr) {
                    report_unconverted_depth_sample(scene.depth_addr);
                }
                continue;
            }
            let Some(t) = scene.target else {
                // A colour-less pass whose DEPTH a later pass samples is load-bearing, and
                // it now gets a target of its own keyed by the DEPTH address rather than
                // being dropped. The colour attachment that target carries is a throwaway:
                // nothing reads it, and it exists so that every pipeline built for an
                // ordinary pass - all of which declare a colour target - can encode this
                // one unchanged. Building a second, colour-less pipeline variant for each
                // pair would be the only alternative, and it would double the pipeline
                // cache to serve the passes that need it least.
                if depth_sampled.contains(&scene.depth_addr)
                    && let Some((w, h)) = scene.depth_extent {
                        self.encode_depth_only_pass(device, queue, encoder, scene, w, h);
                        continue;
                    }
                // The memoised key (see `pair_key`) - this runs per draw of the scene, every
                // frame the scene stays unplaced.
                let mut keys: Vec<String> = Vec::new();
                for g in scene.draws.iter().filter_map(|d| d.gxp.as_ref()) {
                    keys.push(format!("{:016x}", self.gxp.pair_key(g)));
                }
                keys.dedup();
                report_unplaced_scene(
                    scene.draws.len(),
                    scene.depth_addr,
                    depth_sampled.contains(&scene.depth_addr),
                    scene.depth_extent.is_some(),
                    &keys.join(" "),
                );
                continue;
            };
            // >>> A SCENE WITH NO DRAWS PRODUCES NO PASS.
            //
            // MEASURED on the user's phone, one gameplay window's GPU timestamps:
            //
            //   1024x512   0 draws  30.28 ms
            //   640x368  117 draws   1.11 ms
            //   display-blit         0.66 ms
            //
            // The pass that drew the whole frame cost a millisecond; a pass that drew
            // NOTHING cost twenty-seven times as much. And it is not an occasional empty
            // frame - the desktop chain dump reports this title's every frame shape as
            // `2 scenes [0x8be8d280:1024x512/0{} <display>:.../N{...}]`. The `/0{}` is the
            // draw count: that pass is empty in every shape the title has ever produced
            // here, and it was still begun, cleared and STORED each frame - on a tiling GPU
            // a full tile allocation, a 2 MB colour store and a depth store, sixty times a
            // second, for an attachment nothing wrote and (`rtt-samples=0`) nothing read.
            //
            // # WHY THIS IS NOT "SKIP THE CLEAR"
            // The clear on a first pass into a target is OURS, not the guest's - see
            // `first_pass_here` below, where it is documented as making an intermediate
            // image blend correctly where nothing drew. Its only effect on a pass with no
            // draws is to fill the target with transparent black. So the target is left
            // ALONE and, critically, is NOT recorded as rendered: a later scene into the
            // same address this frame is still the first pass there and still clears it.
            // The only behaviour that moves is a target with no draws for a WHOLE frame
            // that something also samples - it then reads what guest memory last held
            // rather than transparent black, which is what the guest's own memory says.
            //
            // Reported rather than silent: a frame losing a pass is exactly the kind of
            // change that must not be discovered from a picture.
            if scene.draws.is_empty() {
                if let Some(t) = scene.target {
                    report_empty_scene_skipped(t.data_addr, t.width, t.height);
                }
                continue;
            }
            // >>> A 64-BIT COLOUR SURFACE IS RENDERED AS RAW WORDS into its own
            // `Rg32Uint` target - see `raw64_color_format` and `rtt_raw`. Not the UNORM
            // path: no gamma view, no multisampling, no snapshot, and cleared to zero words
            // on the frame's first pass into it, as the UNORM path clears to transparent.
            // The attachment itself is handed to the pass for one reason: a program that
            // reads the destination reads a copy of its words (`module::dest_raw_init`).
            if raw64_color_format(t.format) {
                // >>> AND ITS DEPTH IS CONVERTED FOR A LATER SAMPLER, AS AN ORDINARY PASS'S
                // >>> IS. A deferred G-buffer pass is exactly a raw pass whose depth the next
                // passes read: MEASURED on one title's gameplay frame, scene 0 renders its
                // 64-bit G-buffer over depth `0x92200000`, and scene 1 linearises that depth
                // (`viewZ = d * -9.999 + 10`) while the shadow and AO passes sample it too.
                // This arm never registered it, so the samplers fell through to the guest's
                // bytes - zero, which the GPU never writes back - and every pixel's view
                // position came out on the near plane: one flat depth for the whole lit frame.
                let want_depth = depth_sampled.contains(&scene.depth_addr);
                self.ensure_rtt_raw(device, t.data_addr, t.width, t.height, want_depth);
                // The guest's ROW PITCH, which the word view below matches a sampler's row
                // stride against - the surface's width is not its pitch.
                if t.stride_px != 0 {
                    self.rtt_pitch.insert((t.data_addr, t.width, t.height), (t.stride_px, 8));
                }
                let raw_key = (t.data_addr, t.width, t.height);
                let first_pass_here = !self.rtt_rendered_raw.contains_key(&raw_key);
                let clear = first_pass_here.then_some(wgpu::Color::TRANSPARENT);
                let (cv, dv, dtex, ctex) = {
                    let s = &self.rtt_raw[&raw_key];
                    (s.color_view.clone(), s.depth_view.clone(), s.depth.clone(), s.color.clone())
                };
                self.rtt_reads_snapshot.clear();
                self.keep_depth = false;
                let stencil_read = self.stencil_sampled.contains_key(&scene.stencil_addr);
                self.load_saved_depth(encoder, scene, t.width, t.height, Some(&dtex));
                // A stencil plane a fragment program stored to - see `store_mirror`.
                let guest_store =
                    self.begin_guest_stencil(device, queue, encoder, scene, Some(&dtex), (t.width, t.height));
                self.keep_stencil = stencil_read || guest_store;
                self.encode_pass(
                    device, queue, encoder, &cv, &dv, RAW64_FORMAT, scene, t.width, t.height,
                    t.width, t.height, clear, 1, None, Some(&ctex),
                );
                self.keep_stencil = false;
                self.stencil_load_now = false;
                if guest_store {
                    self.store_guest_stencil(device, queue, encoder, scene, &dtex, (t.width, t.height));
                }
                if stencil_read {
                    self.convert_gxm_stencil(device, encoder, &dtex, scene.stencil_addr);
                }
                self.save_depth(device, encoder, scene, t.width, t.height, &dtex);
                if want_depth {
                    self.convert_raw_gxm_depth(device, queue, encoder, raw_key, scene.depth_min, scene.depth_scale);
                    if let Some(v) = self.rtt_raw.get(&raw_key).and_then(|s| s.gxm_depth.as_ref()) {
                        self.rtt_depth_rendered.insert(scene.depth_addr, v.view.clone());
                    }
                }
                self.rtt_rendered_raw.insert(raw_key, cv);
                self.convert_raw_to_float(device, encoder, raw_key);
                continue;
            }
            let want_depth = depth_sampled.contains(&scene.depth_addr);
            // A sampled STENCIL plane needs what a sampled depth needs of the attachment - a
            // bindable, single-sample depth-stencil texture - so it asks `ensure_rtt` the
            // same way. (The depth companion that comes with it goes unused.)
            let stencil_read = self.stencil_sampled.contains_key(&scene.stencil_addr);
            // How many samples the GUEST created this render target with. Not a quality
            // setting and not a per-target judgement of ours - `ensure_rtt` decides only
            // whether it can serve the request, and reports when it cannot.
            self.ensure_rtt(
                device,
                t.data_addr,
                t.width,
                t.height,
                want_depth || stencil_read,
                t.multisample,
                t.msaa_downscale,
                t.gamma,
                float_color_format(t.format).then_some(FLOAT_TARGET_FORMAT),
            );
            // The pitch with the guest's bytes per texel - see `gxm_color_bytes_per_pixel`.
            if let (true, Some(bpp)) = (t.stride_px != 0, crate::gpu::gxm_color_bytes_per_pixel(t.format)) {
                self.rtt_pitch.insert((t.data_addr, t.width, t.height), (t.stride_px, bpp));
            }
            // >>> THE TARGET THIS PASS MEANS, ADDRESS AND EXTENT. The pass knows its extent
            // from its own colour surface, so every lookup below is exact. See `rtt`: this
            // title runs a 256x256 target and a 512x1024 target through the one colour
            // address 0x8e20b030 in the same frame, and an address-only lookup here handed
            // the 4-draw pass the 41-draw pass's target and then rebuilt it.
            let key = (t.data_addr, t.width, t.height);
            // >>> A COLOUR PASS INTO AN ADDRESS RETIRES THE DEPTH CONVERTED THERE.
            //
            // A title may reuse one block of memory as a pass's DEPTH and, later in the
            // frame, as another pass's COLOUR target. Once the colour pass begins, that memory
            // is a colour image's, so the converted depth registered at the address - which
            // the sampler side consults FIRST, by exact address - no longer describes it. Left
            // in place it is bound for any draw naming the address before the colour pass is
            // registered as rendered (the colour pass sampling its own target), and the depth
            // conversion holds its distance in the RED channel. MEASURED (a fighting title, every fight):
            // the 128x256 shadow pass's depth sits at 0x93ca1000, which is also the 512x128
            // glow atlas; the atlas's second draw sampled the depth as its first level, and
            // the glow added a translucent RED band x 480-720 over 5-9% of desktop frames, and
            // a red wash over the whole fight in the browser.
            if self.rtt_depth_rendered.remove(&t.data_addr).is_some() {
                self.rtt_depth_addrs.remove(&t.data_addr);
                report_depth_retired_by_colour(t.data_addr);
            }
            // Drawing into a buffer this frame already filled, which this pass may also
            // sample: hand it a snapshot to read so the live buffer is a target only.
            self.rtt_reads_snapshot.clear();
            // Clear a target the FIRST time this frame draws into it, and compose onto
            // it after that - the same rule the display buffer follows. A later pass
            // into a buffer an earlier pass filled is a post-process step, and it is
            // entitled to leave most of the image alone: a retail racer's race frame ends
            // with a five-draw pass over the world target, and clearing for it wiped the
            // whole world, leaving a correct HUD over black.
            let first_pass_here = !self.rtt_rendered.contains_key(&key);
            // >>> ...AND ONLY WHEN THIS PASS ACTUALLY SAMPLES THE TARGET IT DRAWS INTO.
            //
            // The snapshot exists for one situation: a pass that reads the buffer it is
            // writing. It was taken for EVERY pass after the first into a target, on the
            // reasoning that such a pass MAY sample it - so a plain second pass that
            // composes onto the target and samples nothing from it paid a full-target
            // GPU copy for a read it never makes.
            //
            // MEASURED on one racer's race frame: `rtt 2.00 snapshots (1.40 MB)` every
            // frame, against 0.47 MB of all other buffer writes combined. Three times the
            // frame's whole upload budget, and on a phone `upload` was 7.6 ms of a 19.2 ms
            // encode where here it is 0.1 of 1.3 - so this is a device-shaped cost that a
            // desktop measurement understates.
            //
            // The draws already say what they sample, and the same expression above builds
            // the frame-wide `sampled` set; this asks it of ONE scene against ONE address.
            // The failure mode if it is ever wrong is LOUD, not silent: binding a live
            // colour target as a sampled texture in the same pass is a wgpu validation
            // error, and a validation error fails the run.
            let samples_own_target = !first_pass_here
                && scene.draws.iter().any(|d| {
                    d.texture
                        .iter()
                        .map(|tx| tx.data_addr)
                        .chain(d.gxp.iter().flat_map(|g| {
                            g.textures
                                .iter()
                                .chain(g.vertex_textures.iter())
                                .map(|tx| tx.tex.data_addr)
                        }))
                        .any(|a| a == t.data_addr)
                });
            if samples_own_target
                && let Some(before) = self.snapshot_rtt(device, encoder, key) {
                    self.rtt_rendered.insert(key, before);
                    self.rtt_rendered_uncopied.insert(key);
                    // BY ADDRESS: this is read on the sampler side, where a draw has only
                    // the address it named. See `rtt_reads_snapshot`.
                    self.rtt_reads_snapshot.insert(t.data_addr);
                }
            // A GAMMA-CORRECT surface is rendered through the sRGB view of the same
            // texture, so the ROP encodes each store after blending exactly as the
            // hardware does. The same view goes into `rtt_rendered` below, so a later pass
            // sampling this target decodes on the way back in.
            let gamma = t.gamma;
            let (cv, dv, fmt) = {
                let s = &self.rtt[&key];
                match (gamma, s.color_view_srgb.as_ref()) {
                    (true, Some(v)) => (
                        v.clone(),
                        s.depth_view.clone(),
                        srgb_twin(self.color_format).unwrap_or(self.color_format),
                    ),
                    // >>> THE TEXTURE'S OWN FORMAT, NOT THE RENDERER'S DEFAULT. They are
                    // the same today, and the pipelines that write a pass are keyed on the
                    // format they target - so reading it off the texture is what keeps that
                    // true if a target is ever created in another format.
                    _ => (s.color_view.clone(), s.depth_view.clone(), s.color.format()),
                }
            };
            if gamma {
                report_gamma_surface(t.data_addr, fmt != self.color_format);
            }
            // Rasterise into the MULTISAMPLED attachments when the guest created this
            // render target multisampled, naming `cv` as the resolve target so everything
            // downstream still reads the target at its STORED size. The resolve is part of
            // ending the pass, which is where the hardware does it too - there is no
            // separate resolve pass and no box filter.
            //
            // Not for a GAMMA surface. There the ROP sRGB-encodes each store after
            // blending, and a multisample resolve would have to decide whether it averages
            // the encoded or the linear values. The two give different pixels, hardware
            // does one of them, and nothing here has measured which - so a gamma target
            // keeps the path whose behaviour is already pinned. No measured title's
            // background surface is in gamma mode, so this costs nothing today.
            // `ensure_rtt` already refused a gamma target its attachments and said so, so
            // there is nothing left to decide here: if the attachments exist, use them.
            let use_msaa = self.rtt[&key].msaa.is_some() && self.rtt[&key].samples_now > 1;
            let (pass_cv, pass_dv, pass_samples, resolve) =
                match self.rtt[&key].msaa.as_ref() {
                    Some(m) if use_msaa => {
                        (m.color_view.clone(), m.depth_view.clone(), m.samples, Some(cv.clone()))
                    }
                    _ => (cv.clone(), dv.clone(), 1, None),
                };
            // The TEXTURE behind the attachment, for a draw that reads the destination
            // colour: the MULTISAMPLED one when the pass has it, copied sample for sample
            // (`copy_attachment_to_dest`) - its resolve is not what the ROP feeds back.
            let rtt_ctex = Some(match self.rtt[&key].msaa.as_ref() {
                Some(m) if use_msaa => m.color.clone(),
                _ => self.rtt[&key].color.clone(),
            });
            // A first pass is cleared to transparent black, not to the display's clear
            // colour: it is an intermediate image, and a composite that blends it must
            // see nothing where the pass drew nothing.
            // >>> ...AND ONLY ON THE FIRST FRAME THIS TARGET IS RENDERED AT ALL, because a
            // >>> GXM TILER LOADS THE COLOUR SURFACE AT `sceGxmBeginScene`.
            //
            // The rule above is about passes WITHIN a frame. Across frames the console does
            // not clear: the tiler loads the surface from memory, so a target a title
            // ACCUMULATES into keeps what earlier frames put there. Clearing it here is the
            // one thing that cannot be recovered from later, and it cost a baseball title
            // its ambient light - see `rtt_ever_rendered` for the atlas and the decode.
            //
            // `VITASLOP_GXM_RTT_CLEAR_EVERY_FRAME=1` restores the old behaviour, because
            // "accumulating garbage" and "losing the accumulation" look equally wrong in a
            // picture and the A/B is the only thing that tells them apart.
            let first_frame_here = !self.rtt_ever_rendered.contains(&key);
            let clear = (first_pass_here
                && (first_frame_here || crate::gpu::rtt_clear_every_frame()))
                .then_some(wgpu::Color::TRANSPARENT);
            // TEMPORARY TELEMETRY (`VITASLOP_RTT_CLEAR_PROBE=<hex addr>`): the two booleans
            // that decide LOAD-vs-CLEAR for one target, every pass, with the renderer's
            // mode beside them. A baseball title's 128-slot probe atlas reads 82 of 128
            // slots painted in the EARLY completion's dump and 3 in the DISPLAY frame's,
            // which can only mean the display pass cleared it - and no target in that run
            // is RE-CREATED, so `rtt_ever_rendered` is being consulted and coming back
            // empty for a key something has already rendered. Reasoning about that has
            // twice produced a wrong answer; these four values settle it in one run.
            static CLEAR_PROBE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
            if let Some(want) = *CLEAR_PROBE.get_or_init(|| {
                crate::knobs::var("VITASLOP_RTT_CLEAR_PROBE")
                    .ok()
                    .and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok())
            })
                && want == t.data_addr
            {
                report_knob!(
                    "rtt clear probe: {:#010x} {}x{} offscreen_only={} first_pass_here={} \
                     first_frame_here={} -> {}",
                    key.0,
                    key.1,
                    key.2,
                    self.offscreen_only,
                    first_pass_here,
                    first_frame_here,
                    if clear.is_some() { "CLEAR" } else { "LOAD" },
                );
            }
            self.keep_depth = want_depth;
            // The single-sample depth attachment, which is the only one `depth_saved` can
            // copy into or out of; a multisampled pass is reported by the load.
            let dtex = (!use_msaa).then(|| self.rtt[&key].depth.clone());
            self.load_saved_depth(encoder, scene, t.width, t.height, dtex.as_ref());
            // >>> A LOADED DEPTH THAT IS SAMPLED IS CONVERTED BEFORE THE PASS, so a draw that
            // >>> reads it reads THIS frame's. The depth came from an earlier pass - on a
            // deferred title the G-buffer pass, which renders RAW and converts nothing - and
            // the newest conversion at the address was otherwise last frame's. MEASURED on
            // an action title: its lighting samples 0x8a8dee00 while it tests
            // against it, and read the previous frame's depth to rebuild positions.
            let loaded = self.depth_load_now;
            if loaded && want_depth {
                self.convert_gxm_depth(device, queue, encoder, key, scene.depth_min, scene.depth_scale);
                if let Some(v) = self.rtt.get(&key).and_then(|s| s.gxm_depth.as_ref()) {
                    self.rtt_depth_rendered.insert(scene.depth_addr, v.view.clone());
                    self.rtt_depth_addrs.insert(scene.depth_addr, key);
                }
            }
            // Whether any draw of the pass can change the depth. A fixed-function draw is
            // taken to, not proven not to.
            let writes_depth = scene.draws.iter().any(|d| d.gxp.as_ref().is_none_or(|g| g.depth_write));
            if use_msaa && clear.is_none() && self.rtt[&key].msaa_stale {
                self.seed_msaa_from_color(device, encoder, key);
            }
            if let Some(t) = self.rtt.get_mut(&key) {
                // A multisampled pass resolves into `color`, so the two agree afterwards; a
                // single-sample pass writes `color` alone.
                t.msaa_stale = !use_msaa && t.msaa.is_some();
            }
            // A stencil plane a fragment program stored to - see `store_mirror`. The
            // single-sample attachment only, as for `depth_saved`.
            let guest_store =
                self.begin_guest_stencil(device, queue, encoder, scene, dtex.as_ref(), (t.width, t.height));
            self.keep_stencil = stencil_read || guest_store;
            self.encode_pass(
                // Target extent and attachment extent are the same here: an offscreen
                // pass rasterises at the size the guest gave its render target.
                device, queue, encoder, &pass_cv, &pass_dv, fmt, scene, t.width, t.height,
                t.width, t.height, clear, pass_samples, resolve.as_ref(), rtt_ctex.as_ref(),
            );
            self.keep_depth = false;
            self.keep_stencil = false;
            self.stencil_load_now = false;
            if let (true, Some(d)) = (guest_store, dtex.as_ref()) {
                self.store_guest_stencil(device, queue, encoder, scene, d, (t.width, t.height));
            }
            if stencil_read {
                // The attachment the pass rasterised into: `dtex` is the single-sample one,
                // and a multisampled pass is reported (its stencil lives in the MSAA depth).
                match dtex.as_ref() {
                    Some(d) => self.convert_gxm_stencil(device, encoder, d, scene.stencil_addr),
                    None => report_stencil_sample_unserved(
                        scene.stencil_addr,
                        "the pass that writes it is MULTISAMPLED and a per-sample stencil conversion is not established",
                    ),
                }
            }
            if crate::gpu::no_alpha_float_format(t.format) && !self.rtt[&key].gamma {
                self.fill_alpha_one(device, encoder, key);
            }
            if let Some(d) = dtex.as_ref() {
                self.save_depth(device, encoder, scene, t.width, t.height, d);
            }
            self.depth_load_now = false;
            self.rtt_rendered.insert(key, cv);
            self.rtt_rendered_uncopied.insert(key);
            self.rtt_ever_rendered.insert(key);
            // An offscreen pass is now the LATEST write at this address - see
            // `rtt_superseded`.
            if self.rtt_superseded.remove(&t.data_addr) {
                self.rtt_epoch = self.rtt_epoch.wrapping_add(1);
            }
            // Convert this pass's depth NOW: the next pass into the same target clears the
            // depth attachment, so afterwards there is nothing left to convert. A pass that
            // LOADED its depth and wrote none of it already converted it above - the same
            // pixels again would be a full-screen pass for nothing (3.2 ms of a phone frame).
            if want_depth && (!loaded || writes_depth) {
                self.convert_gxm_depth(
                    device,
                    queue,
                    encoder,
                    key,
                    scene.depth_min,
                    scene.depth_scale,
                );
                if let Some(v) = self.rtt.get(&key).and_then(|s| s.gxm_depth.as_ref()) {
                    self.rtt_depth_rendered.insert(scene.depth_addr, v.view.clone());
                    // The VALUE is the full key: which of the targets at this colour
                    // address holds the depth this pass just converted is exactly the
                    // question an address could not answer. See `rtt_depth_addrs`.
                    self.rtt_depth_addrs.insert(scene.depth_addr, key);
                }
            }
        }
        // ...and the LOOP ends here. Everything below is the frame's TAIL.
        self.chain_phases.scene_loop_ms = t_scene_loop.ms();
        let t_chain_tail = Stopwatch::start();
        // Report the frame's pass structure whenever it CHANGES: how many scenes, where
        // each one draws, and - the load-bearing number - how many draws sampled a
        // target this frame rendered. A chain of passes with zero samples of them means
        // the composite never reads the world, which looks exactly like the passes not
        // being drawn at all but is a different bug.
        if scenes.len() > 1 || self.rtt_hits > 0 {
            // >>> THE SHAPE IS FOLDED TO 64 BITS BEFORE ANYTHING IS FORMATTED, AND THAT IS
            // >>> THE WHOLE COST OF THIS DIAGNOSTIC, NOT A TIDINESS CHOICE.
            //
            // The line used to be BUILT and then compared against the last one - so a frame
            // that printed nothing still paid for the string. Building it calls the pair
            // KEY for every draw of every scene, and `GxpLive::key` is an FNV over BOTH
            // program blobs (kilobytes each) - about 490 draws a frame on this title's race,
            // several megabytes of hashing per frame, thrown away. `pair_key` is the
            // memoised form the renderer already uses on the draw path (a pointer-keyed
            // lookup), and it is what this now folds.
            //
            // Order-independent for the sampled addresses (a `HashSet`, whose iteration
            // order is not stable) and order-DEPENDENT for the scene list, which is the
            // frame's structure and is exactly what a change should report.
            let mut fold: u64 = 0xcbf2_9ce4_8422_2325;
            let mut mix = |v: u64| {
                fold ^= v;
                fold = fold.wrapping_mul(0x0000_0100_0000_01b3);
            };
            // >>> THE KEY IS THE SHAPE, NOT THE VOLUME. Everything that DRIFTS is printed
            // in the line but kept OUT of the fold, or the memo never converges and the
            // report becomes per-frame - the same defect that made this line 38% of the
            // browser renderer, arriving from a different direction.
            //
            // MEASURED on one title's race: with the draw counts, the rtt-sample count and
            // the sampled-address set in the key, a SIX-scene frame minted about twenty
            // distinct "shapes" that described the same six passes. Draw counts drift with
            // what is on screen (1 -> 3 -> 4 in a single pass), `rtt_hits` drifts with them,
            // and the sampled set follows.
            //
            // NO ADDRESS IS IN THE KEY. A triple-buffered title cycles its display target
            // between three addresses by design, and a title with text or sprite caches
            // allocates a FRESH interior target every frame - keyed on addresses, a run of
            // the latter printed this line every frame until the cap. A target's SIZE
            // counts - a pass that changed resolution IS a new shape - and the addresses
            // are still PRINTED, so which pass a draw lands in is still carried.
            //
            // The pairs are folded as each scene's DEDUPLICATED SEQUENCE - what the line
            // prints - not per draw: a text pass drawing the same pairs 44 times and then
            // 40 is one shape, not two.
            for s in scenes.iter() {
                match s.target {
                    Some(t) => mix(((t.width as u64) << 32) | t.height as u64),
                    None => mix(u64::MAX),
                }
                let mut prev = None;
                for d in s.draws.iter() {
                    if let Some(g) = d.gxp.as_ref() {
                        let k = self.gxp.pair_key(g);
                        if prev != Some(k) {
                            mix(k);
                            prev = Some(k);
                        }
                    }
                }
                mix(0x5c3e);
            }
            let fresh = !self.chain_shapes_seen.contains(&fold);
            if fresh && self.chain_shapes_seen.len() >= CHAIN_SHAPE_CAP {
                if !self.chain_shapes_capped {
                    self.chain_shapes_capped = true;
                    report_status!(
                        "gxm chain: {CHAIN_SHAPE_CAP} distinct frame shapes reported; further \
                         NEW shapes will not be printed. A title reaching this is changing \
                         its pass structure constantly, which is itself the finding."
                    );
                }
            } else if fresh {
            self.chain_shapes_seen.insert(fold);
            // >>> WITH EACH SCENE'S PAIR KEYS. Which PASS a draw is in is the fact this
            // line exists to carry, and without the keys it cannot be checked against
            // anything: "the composite reads the world" and "the composite is in the pass
            // that reaches the screen" are different claims, and a frame that is a flat
            // colour is consistent with either failing.
            let shape = scenes
                .iter()
                .map(|s| {
                    let mut keys: Vec<String> = s
                        .draws
                        .iter()
                        .filter_map(|d| d.gxp.as_ref())
                        // The MEMOISED key, for the reason the fold above uses it.
                        .map(|g| format!("{:016x}", self.gxp.pair_key(g)))
                        .collect();
                    keys.dedup();
                    let where_ = match s.target {
                        Some(t) => format!("{:#x}:{}x{}/{}", t.data_addr, t.width, t.height, s.draws.len()),
                        None => format!("?/{}", s.draws.len()),
                    };
                    format!("{where_}{{{}}}", keys.join(","))
                })
                .collect::<Vec<_>>()
                .join(" ");
            let mut sampled: Vec<String> =
                self.sampled_addrs.iter().map(|a| format!("{a:#x}")).collect();
            sampled.sort();
            let line = format!(
                "gxm chain: FIRST SEEN on rendered frame #{}: {} scenes [{shape}]                      rtt-samples={} final-pass-sampled=[{}]",
                self.chain_frames_seen,
                scenes.len(),
                self.rtt_hits,
                sampled.join(" ")
            );
            // >>> STATUS, NOT A WARNING - and still ALWAYS SHOWN.
            //
            // This is the line that says whether the composite READ the world, which is a
            // different bug from the passes not being drawn, and a black frame cannot make
            // that distinction. So it has to remain visible on a phone, and `debug` would
            // delete it. But it describes a HEALTHY frame just as readily as a broken one:
            // there is nothing to FIX on reading it, so it is not a warning
            // [[vitaslop-a-warning-means-we-owe-a-fix]]. It goes to the panel's STATUS
            // section, which is forced on regardless of `VITASLOP_LOG`.
            //
            // One line per distinct SHAPE, decided above.
            report_status!("{line}");
            }
        }
        // Put the kept display image on the caller's framebuffer. The same fullscreen
        // triangle the supersample resolve uses, with the scale uniform at 1, which makes
        // `fres` an exact 1:1 `textureLoad` copy - the image and the caller's target are
        // the same size by construction (`ensure_display_image` takes the FRAMEBUFFER
        // extent, `fb_w/fb_h`, not the guest surface's).
        // >>> THE LINEAR VIEW, DELIBERATELY, AND IT IS LOAD-BEARING NOW THAT THE PASS
        // >>> ABOVE MAY HAVE WRITTEN sRGB-ENCODED BYTES.
        //
        // This blit is a byte passthrough: it samples the image through a NON-sRGB view
        // (so a sample returns the stored byte over 255, undecoded) and writes into the
        // caller's swapchain, which is also non-sRGB (so the value is stored as that same
        // byte). What the panel then does with those bytes - apply the sRGB transfer - is
        // what the console's display controller does with the ROP's encoded output, so a
        // gamma-mode frame reaches the screen exactly as the hardware would deliver it.
        //
        // Binding `view_srgb` here instead would DECODE on the way in and store linear,
        // undoing the encode the pass just did and putting the frame back where it was.
        if held_flip.is_some() {
            display_blit_addr = held_flip;
        }
        if let Some(addr) = display_blit_addr
            && let Some(view) = self.display_images.get(&addr).map(|d| &d.view) {
                queue.write_buffer(&self.resolve_scale_buf, 0, &1u32.to_le_bytes());
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gxm-display-blit-bind"),
                    layout: &self.resolve_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                        wgpu::BindGroupEntry { binding: 1, resource: self.resolve_scale_buf.as_entire_binding() },
                    ],
                });
                enc(&ENC.passes, 1);
                enc(&ENC.pipeline_sets, 1);
                enc(&ENC.bind_group_sets, 1);
                enc(&ENC.draw_calls, 1);
                enc(&ENC.bind_groups_built, 1);
                let ts = self.ts_pair(|| "display-blit".to_string());
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gxm-display-blit"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: Self::ts_writes(&ts),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rpass.set_pipeline(&self.resolve_pipeline);
                rpass.set_bind_group(0, &bind, &[]);
                rpass.draw(0..3, 0..1);
            }
        // Resolve the supersampled display buffer once, after the last pass into it.
        // The pair is taken before the target is borrowed, since taking one is a mutable use.
        let ts_resolve =
            if ss && self.ss_target.is_some() { self.ts_pair(|| "ss-resolve".to_string()) } else { None };
        if let (true, Some(t)) = (ss, self.ss_target.as_ref()) {
            enc(&ENC.passes, 1);
            enc(&ENC.pipeline_sets, 1);
            enc(&ENC.bind_group_sets, 1);
            enc(&ENC.draw_calls, 1);
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gxm-resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: Self::ts_writes(&ts_resolve),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rpass.set_pipeline(&self.resolve_pipeline);
            rpass.set_bind_group(0, &t.resolve_bind, &[]);
            rpass.draw(0..3, 0..1);
        }
        // >>> REPORTED WHETHER OR NOT IT FITS, at the END of the frame that built it.
        //
        // The composition used to print ONLY when the budget was exceeded and nothing was
        // evictable - the failure case - which made a successful shrink unmeasurable: three
        // arms of an A/B (decode, passthrough, transcode) produced ONE number between them,
        // because the two arms that WORKED stopped tripping the report that carried the
        // number. It is unconditional now, bounded by a high-water step instead.
        //
        // At the end, not the top. The top of this function holds the PREVIOUS frame's
        // total, which is the right number in a windowed run and is zero in a headless one -
        // headless fast-forwards the guest and encodes a single frame, so a report placed
        // there fired exactly once, before that frame had bound anything, and said 0 MB.
        // A frame-scoped statistic has to be read where the frame ends.
        report_texture_working_set(
            self.gxp.views_frame_bytes,
            &self.gxp.views_used,
            self.gxp.views_epoch,
        );
        // ...and that every texture created is still accounted for. Read at the end of the
        // frame, where the cache is settled, for the same reason the working set is.
        {
            use std::sync::atomic::Ordering;
            let _ = Ordering::Relaxed;
            report_texture_handle_drift(self.gxp.views.len());
        }
        // >>> AND SUBMIT THE FRAME'S TEXTURE EXPANSIONS, BEFORE THE CALLER SUBMITS ITS OWN.
        //
        // Every texture `expand_rgba8` handed back this frame is written by commands sitting
        // in a batch of its own; this is where they go to the queue. It has to be HERE and
        // not in the caller: the draws that sample those textures are in the caller's
        // encoder, which is submitted the moment this returns, and a queue submit made after
        // that one would write the texels a frame late. See `texenc::RawBatch`.
        if let Some(t) = self.gxp.texenc.as_ref() {
            t.flush_raw(queue);
        }
        // The last of the four wall times. `head + loop + tail` is now the whole function,
        // and the caller's `encode_ms` minus it is the encoder creation and the timestamp
        // resolve - so a residual after this points OUTSIDE `encode_chain`, which is a
        // different question from the one it used to be mistaken for.
        // >>> AND THE BELT IS CLOSED HERE, after the last pass has taken its chunks and
        // before the caller finishes and submits this encoder. `finish` unmaps what was
        // written so the copies this chain recorded are legal to execute; the chunks go
        // back at the top of the NEXT chain, which is the first point this crate can know
        // the submit happened. See `GxmRenderer::gxp_staging`.
        if let Some(belt) = self.gxp_staging.as_mut() {
            belt.finish();
        }
        self.chain_phases.chain_tail_ms = t_chain_tail.ms();
    }

    /// Encode a full scene into `encoder`: a render pass over `color_view` (cleared
    /// to `clear`) with `depth_view` (must be [`depth_format`]), drawing every draw
    /// in submission order. `(surf_w, surf_h)` are the target size (the Pixel-space
    /// projection needs them). Requires `&mut self` for the per-frame arenas and the
    /// cross-frame caches.
    ///
    /// One frame does: at most three `write_buffer` uploads (vertices, indices,
    /// uniforms), a texture upload only for a not-yet-seen texture, and one bind
    /// group only for a not-yet-seen (texture, filter) pair - then the pass. Nothing
    /// is allocated per draw once the working set is warm.
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        scene: &RenderScene,
        surf_w: u32,
        surf_h: u32,
        clear: [u8; 4],
    ) {
        self.encode_chain(
            device,
            queue,
            encoder,
            color_view,
            depth_view,
            std::slice::from_ref(scene),
            surf_w,
            surf_h,
            // This entry point renders ONE scene into a target the caller sized to the
            // guest surface, so the two extents are the same by construction.
            surf_w,
            surf_h,
            clear,
            None,
        );
    }

    /// One scene into one target. `clear` is `Some(colour)` for the first pass into a
    /// target and `None` to compose onto what is already there. Supersampling is the
    /// caller's business ([`encode_chain`](Self::encode_chain) picks the views and
    /// resolves once), so this is a plain pass.
    #[allow(clippy::too_many_arguments)]
    /// Before a pass: when its surface FORCE_LOADs, copy the depth-stencil a FORCE_STORE pass
    /// left at that address and extent into `dst`, and have the pass LOAD it. See
    /// [`Self::depth_saved`].
    pub(super) fn load_saved_depth(&mut self, encoder: &mut wgpu::CommandEncoder, scene: &RenderScene, w: u32, h: u32, dst: Option<&wgpu::Texture>) {
        self.depth_load_now = false;
        if !zls_load(scene) {
            return;
        }
        // From the LOAD surface's memory, which `sceGxmBeginSceneEx` may name apart from
        // the surface this scene stores to (`RenderScene::depth_load_addr`).
        let from = if scene.depth_load_addr != 0 { scene.depth_load_addr } else { scene.depth_addr };
        let Some(src) = self.depth_saved.get(&(from, w, h)) else { return };
        let Some(dst) = dst.filter(|d| d.sample_count() == 1 && d.format() == src.format() && d.size() == src.size()) else {
            crate::gpu::report_zls_unserved(from, w, h);
            return;
        };
        encoder.copy_texture_to_texture(src.as_image_copy(), dst.as_image_copy(), src.size());
        self.depth_load_now = true;
    }

    /// After a pass: when its surface FORCE_STOREs, keep its depth-stencil for a later
    /// FORCE_LOAD pass at the same address. See [`Self::depth_saved`].
    pub(super) fn save_depth(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, scene: &RenderScene, w: u32, h: u32, src: &wgpu::Texture) {
        self.depth_load_now = false;
        if !zls_store(scene) || src.sample_count() != 1 {
            return;
        }
        let key = (scene.depth_addr, w, h);
        if self.depth_saved.get(&key).is_none_or(|t| t.format() != src.format() || t.size() != src.size()) {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-depth-saved"),
                size: src.size(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: src.format(),
                usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.depth_saved.insert(key, t);
        }
        let dst = &self.depth_saved[&key];
        encoder.copy_texture_to_texture(src.as_image_copy(), dst.as_image_copy(), src.size());
    }
}
