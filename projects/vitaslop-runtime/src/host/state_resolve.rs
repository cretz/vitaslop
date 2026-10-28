use super::*;

impl VitaState {
    /// Read every deferred draw's vertices and indices, NOW - at `sceGxmEndScene`, which is
    /// when the GPU would. See [`defer_geometry`] for why this is not at the draw call.
    ///
    /// Walks the current scene's draws and [`Self::deferred_geometry`] in lockstep: they are
    /// pushed together and only for a draw the scene accepted, so index `i` of one is index `i`
    /// of the other. A mismatch would mean a draw got another draw's bindings, so it is an
    /// assertion rather than a `zip`.
    /// The FLIP's resolve. Normally [`Self::resolve_deferred_geometry`] on the flipping thread;
    /// under `async_flip_resolve` it only records the boundary - every scene ended so far -
    /// for the presenting worker, after settling any boundary an earlier flip still owes (so
    /// scenes are always read in order and never later than the next flip).
    ///
    /// # Why this may wait past the flip
    /// The hardware does: a flip QUEUES the frame, and the GPU reads that frame's vertex
    /// buffers after it, while the CPU is already building the next one. A title therefore
    /// cannot rewrite them at the flip (a baseball title double-buffers and syncs on nothing else - no
    /// notification wait, no `sceGxmFinish`, over a 30-frame at-bat window). Every sync point
    /// that COULD make a rewrite safe (`sceGxmFinish`, a notification wait) still resolves
    /// everything first, through `resolve_deferred_geometry`.
    pub fn resolve_at_flip(&mut self, ctx: &GuestCtx) {
        if !async_flip_resolve() {
            self.resolve_deferred_geometry(ctx);
            return;
        }
        // A RESOLVER worker is up: hand it everything ended so far, now. Taking the job is
        // cheap (descriptions and windows move, nothing is read), which is the point - the
        // reading happens on a core the game's serial main -> render chain is not waiting on.
        if resolver_attached() {
            let _x = crate::perf::exclude_scope();
            let job = self.take_resolve_job(u64::MAX);
            if !job.scenes.is_empty() {
                let mut slot = self
                    .async_resolve
                    .slot
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                slot.in_flight += 1;
                slot.queue.push_back(job);
                self.async_resolve.cv.notify_all();
            }
            return;
        }
        if let Some(b) = self.flip_resolve_upto.take() {
            self.resolve_deferred_geometry_upto(ctx, b);
        }
        if !self.pending_geometry.is_empty() {
            self.flip_resolve_upto = Some(self.deferred_serial);
        }
    }

    /// >>> A SYNC POINT'S RESOLVE, HANDED TO THE RESOLVER WORKER INSTEAD OF READ UNDER THE HOST
    /// LOCK. Returns `true` when a job was queued - the caller must then BLOCK its thread
    /// (through `pending_early`) until the run worker has applied it; see
    /// `vita::gxm::complete_scenes_through`.
    ///
    /// `resolve_deferred_geometry` does the whole read inline, inside the guest's host call, so
    /// the GLOBAL host lock is held for all of it - MEASURED on the phone at a baseball at-bat,
    /// `sceGxmFinish` held it ~3 ms a call while every other guest thread's host calls queued
    /// behind it (1.5 ms/frame of lock waiting). On the device `sceGxmFinish` stalls only the
    /// thread that called it. So: the job is taken here (cheap), read on the resolver worker,
    /// and applied by the run worker before the thread is woken - the "bytes are final when the
    /// wait returns" contract holds, and the other threads run meanwhile.
    ///
    /// Only under the asynchronous frontend with a resolver worker; elsewhere (and under
    /// `VITASLOP_SYNC_RESOLVE_ASYNC=0`) the inline read is unchanged.
    pub fn queue_sync_resolve(&mut self) -> bool {
        if !self.complete_scene_async
            || !resolver_attached()
            || !async_flip_resolve()
            || !sync_resolve_async()
        {
            return false;
        }
        self.flip_resolve_upto = None;
        let job = self.take_resolve_job(u64::MAX);
        if job.scenes.is_empty() {
            return false;
        }
        let mut slot = self
            .async_resolve
            .slot
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        slot.in_flight += 1;
        slot.queue.push_back(job);
        self.async_resolve.cv.notify_all();
        true
    }

    /// Resolve the boundary a flip left (see [`Self::resolve_at_flip`]), if any. The
    /// presenting worker calls this before it takes the flipped frame's scenes.
    pub fn resolve_flip_boundary(&mut self, ctx: &GuestCtx) {
        if let Some(b) = self.flip_resolve_upto.take() {
            self.resolve_deferred_geometry_upto(ctx, b);
        }
    }

    /// [`Self::resolve_deferred_geometry`] for the scenes whose serial is `<= upto` only; the
    /// later ones (the next frame's, already ended) stay pending for their own flip - a
    /// kinematics job may still be filling their buffers (see the crowd note below).
    pub(super) fn resolve_deferred_geometry_upto(&mut self, ctx: &GuestCtx, upto: u64) {
        self.settle_async_resolve(ctx);
        let _all = crate::perf::scope(crate::perf::Phase::DrawTotal);
        let _x = crate::perf::exclude_scope();
        let mut job = self.take_resolve_job(upto);
        if job.scenes.is_empty() {
            return;
        }
        {
            let mut snaps = snaps!(self);
            compute_resolve_job(&mut snaps, ctx, &mut job);
        }
        self.apply_resolve_job(ctx, job);
    }

    pub fn resolve_deferred_geometry(&mut self, ctx: &GuestCtx) {
        // Everything pending is read now, so no flip boundary is owed any more.
        self.flip_resolve_upto = None;
        // A job the presenting worker took at a flip boundary and is still reading holds
        // scenes older than anything pending here: wait for it and apply it first, so every
        // scene ended so far has its bytes when this returns - the contract of a sync point.
        self.settle_async_resolve(ctx);
        if self.pending_geometry.is_empty() {
            return;
        }
        // The seam census, at a cadence that costs a run a handful of lines - see
        // [`report_window_reread_census`].
        if self.cur_frame > 0 && self.cur_frame.is_multiple_of(200) {
            static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
            if LAST.swap(self.cur_frame, std::sync::atomic::Ordering::Relaxed) != self.cur_frame {
                report_window_reread_census(self.cur_frame);
            }
        }
        // Under the SAME phase the draw-time read was under, so the two arms' phase tables
        // compare - a fix that moved the work out of `DrawTotal` would read as a free frame.
        let _all = crate::perf::scope(crate::perf::Phase::DrawTotal);
        let _x = crate::perf::exclude_scope();
        let mut job = self.take_resolve_job(u64::MAX);
        {
            let mut snaps = snaps!(self);
            compute_resolve_job(&mut snaps, ctx, &mut job);
        }
        self.apply_resolve_job(ctx, job);
    }

    /// >>> THE FLIP'S RESOLVE, READ OFF THE HOST LOCK: take the job (under the lock, cheap).
    ///
    /// Called by the presenting worker (`VITASLOP_SMP_ASYNC_RESOLVE`) with the boundary the
    /// guest's flip left (see [`Self::resolve_at_flip`]). The scenes stay in the capture; their
    /// windows move into the job and come back with [`Self::apply_posted_resolve`]. `None` when
    /// no boundary is owed. The job is marked IN FLIGHT here, so a sync point on a guest thread
    /// ([`Self::resolve_deferred_geometry`]) waits for its bytes instead of reading around it.
    pub(super) fn take_flip_resolve_job(
        &mut self,
    ) -> Option<(
        ResolveJob,
        Arc<AsyncResolve>,
        Arc<std::sync::Mutex<TextureSnapshots>>,
    )> {
        let upto = self.flip_resolve_upto.take()?;
        let job = self.take_resolve_job(upto);
        if job.scenes.is_empty() {
            return None;
        }
        self.async_resolve
            .slot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .in_flight += 1;
        Some((
            job,
            self.async_resolve.clone(),
            self.texture_snapshots.clone(),
        ))
    }

    /// Apply every job a presenting worker has finished reading (see
    /// [`Self::take_flip_resolve_job`]) to the scenes it came from. Under the host lock; cheap -
    /// it moves `Arc`s into draws.
    pub fn apply_posted_resolve(&mut self, ctx: &GuestCtx) {
        let done = std::mem::take(
            &mut self
                .async_resolve
                .slot
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .done,
        );
        for job in done {
            self.apply_resolve_job(ctx, job);
        }
    }

    /// Wait for every in-flight job and apply it. The computing side takes only the snapshot
    /// lock, never the host lock this caller holds, so the wait cannot deadlock.
    pub(super) fn settle_async_resolve(&mut self, ctx: &GuestCtx) {
        {
            let mut slot = self
                .async_resolve
                .slot
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            while slot.in_flight > 0 {
                slot = self
                    .async_resolve
                    .cv
                    .wait(slot)
                    .unwrap_or_else(|e| e.into_inner());
            }
        }
        self.apply_posted_resolve(ctx);
    }

    /// A resolved draw's `(fragment, vertex)` texture lists, building any set the resolver
    /// could not prove - see [`DeferredTextures`].
    pub(super) fn finish_textures(
        &mut self,
        ctx: &GuestCtx,
        t: ResolvedTextures,
    ) -> (
        Arc<[crate::capture::BoundTexture]>,
        Arc<[crate::capture::BoundTexture]>,
    ) {
        let frag = match t.frag {
            TexSet::Hit(list) => list,
            TexSet::Miss(key, binds) => {
                // An earlier miss of this apply may already have built it.
                let validated = snaps!(self).set_validated(ctx, key, t.fheader as u64, &binds);
                match validated {
                    Some(list) => list,
                    None => {
                        let _m = crate::perf::scope(crate::perf::Phase::DrawTexFragMiss);
                        let mut textures = self.snapshot_bound_textures(ctx, &binds);
                        // The albedo reorder `draw_textures_now` applies before it stores a set.
                        if let Some(pos) = textures
                            .iter()
                            .position(|t| t.height > 1 && t.faces <= 1)
                            .filter(|&p| p > 0)
                        {
                            textures[..=pos].rotate_right(1);
                        }
                        if let Some(unit) = t.albedo
                            && let Some(pos) = textures.iter().position(|t| t.unit == unit)
                        {
                            textures[..=pos].rotate_right(1);
                        }
                        let set: Arc<[crate::capture::BoundTexture]> = textures.into();
                        snaps!(self).set_insert(key, t.fheader as u64, &binds, set.clone());
                        set
                    }
                }
            }
        };
        let vert = match t.vert {
            None => crate::capture::no_textures(),
            Some(TexSet::Hit(list)) => list,
            Some(TexSet::Miss(key, binds)) => {
                let validated = snaps!(self).set_validated(ctx, key, VERTEX_STAGE_TAG, &binds);
                match validated {
                    Some(list) => list,
                    None => {
                        let decoded = self.snapshot_bound_textures(ctx, &binds);
                        let set: Arc<[crate::capture::BoundTexture]> = decoded.as_slice().into();
                        snaps!(self).set_insert(key, VERTEX_STAGE_TAG, &binds, set.clone());
                        set
                    }
                }
            }
        };
        (frag, vert)
    }

    /// Take the pending geometry of every scene whose serial is `<= upto` into a job, with each
    /// draw's uniform windows MOVED out of the capture (they come back in
    /// [`Self::apply_resolve_job`]). Scenes the capture no longer holds, or whose draws do not
    /// pair with their descriptions, are reported here and dropped from the job.
    pub(super) fn take_resolve_job(&mut self, upto: u64) -> ResolveJob {
        let cut = self
            .pending_geometry
            .iter()
            .position(|(id, _)| *id > upto)
            .unwrap_or(self.pending_geometry.len());
        let rest = self.pending_geometry.split_off(cut);
        let entries = std::mem::replace(&mut self.pending_geometry, rest);
        let mut scenes = Vec::with_capacity(entries.len());
        for (id, geo) in entries {
            // A scene the capture no longer holds cannot receive its bytes: its draws render with
            // EMPTY geometry, which is missing geometry - so it is REPORTED, not skipped. MEASURED
            // on a football title's intro: every draw of every frame "carried no vertices", and
            // this was the only exit in the path that said nothing.
            let Some(scene) = self.capture.scenes.iter_mut().find(|s| s.deferred_id == id) else {
                // Retired by a late present's catch-up because a newer frame renders its target
                // again (`Capture::retire_oldest_untaken_frame`): nothing is owed to it.
                if self.capture.take_dropped_pending(id) {
                    continue;
                }
                tracing::warn!(
                    target: "vitaslop::gxm",
                    deferred_id = id, draws = geo.len(), scenes_held = self.capture.scenes.len(),
                    "deferred geometry: the scene it belongs to is no longer in the capture - its draws render with NO vertices"
                );
                continue;
            };
            // Not a `zip` truncation: unequal lengths would mean a draw is about to read ANOTHER
            // draw's streams, which is a smeared mesh with nothing to say why. Report and leave the
            // geometry empty, which is missing geometry - visible, and honest.
            if geo.len() != scene.draws.len() {
                scene.deferred_id = 0;
                tracing::warn!(
                    target: "vitaslop::gxm",
                    descriptions = geo.len(),
                    draws = scene.draws.len(),
                    "deferred geometry does not pair with this scene's draws - every draw in it is DROPPED rather than read through another draw's vertex streams"
                );
                continue;
            }
            let windows = scene
                .draws
                .iter_mut()
                .map(|d| {
                    (
                        std::mem::take(&mut d.mem_windows),
                        std::mem::take(&mut d.frag_mem_windows),
                    )
                })
                .collect();
            scenes.push(ResolveScene {
                id,
                continues: false,
                geo,
                windows,
                out: Vec::new(),
                tex_out: Vec::new(),
            });
        }
        ResolveJob {
            frame: self.cur_frame,
            scenes,
        }
    }

    /// Put a computed job's bytes into the draws it was taken from.
    pub(super) fn apply_resolve_job(&mut self, ctx: &GuestCtx, job: ResolveJob) {
        for mut rs in job.scenes {
            // Retired while its bytes were being read (`Capture::retire_oldest_untaken_frame`):
            // a newer frame renders its target again, so neither its bytes nor its textures are
            // wanted.
            if !self.capture.scenes.iter().any(|s| s.deferred_id == rs.id)
                && self.capture.take_dropped_pending(rs.id)
            {
                continue;
            }
            // Deferred textures: a set the resolver proved is moved in; one it could not answer
            // is built HERE, under the host lock, because building one reads the host's recorded
            // texture formats - exactly the miss path `draw_textures_now` takes.
            let tex: Vec<
                Option<(
                    Arc<[crate::capture::BoundTexture]>,
                    Arc<[crate::capture::BoundTexture]>,
                )>,
            > = std::mem::take(&mut rs.tex_out)
                .into_iter()
                .map(|t| t.map(|t| self.finish_textures(ctx, t)))
                .collect();
            let Some(scene) = self
                .capture
                .scenes
                .iter_mut()
                .find(|s| s.deferred_id == rs.id)
            else {
                tracing::warn!(
                    target: "vitaslop::gxm",
                    deferred_id = rs.id, draws = rs.geo.len(), scenes_held = self.capture.scenes.len(),
                    "deferred geometry: the scene it belongs to left the capture while its bytes were being read - its draws render with NO vertices"
                );
                continue;
            };
            scene.deferred_id = 0;
            for (d, ((indices, vertices), (mw, fw))) in scene
                .draws
                .iter_mut()
                .zip(rs.out.into_iter().zip(rs.windows))
            {
                d.indices = indices;
                d.vertices = vertices;
                d.mem_windows = mw;
                d.frag_mem_windows = fw;
            }
            for (d, t) in scene.draws.iter_mut().zip(tex) {
                if let Some((frag, vert)) = t {
                    d.textures = frag;
                    d.vertex_textures = vert;
                }
            }
        }
    }

    /// Re-read every window [`STREAM_WATCH`] holds and report the ones whose bytes CHANGED
    /// since the draw that named them. Called at `sceGxmEndScene`; a no-op unless
    /// [`dump_stream_bytes`] is on, because nothing fills the list otherwise.
    pub fn report_stream_rewrites(&mut self, ctx: &GuestCtx) {
        let watch = std::mem::take(&mut *STREAM_WATCH.lock().unwrap_or_else(|e| e.into_inner()));
        for (addr, len, was) in watch {
            let now = stream_watch_hash(&ctx.read_bytes(addr, len as usize));
            if now != was {
                eprintln!(
                    "STREAM REWRITTEN AFTER ITS DRAW: {addr:#x} {len} bytes changed between sceGxmDraw and sceGxmEndScene - this draw's captured vertices are STALE"
                );
            }
        }
        let uwatch = std::mem::take(&mut *UNIFORM_WATCH.lock().unwrap_or_else(|e| e.into_inner()));
        if !uwatch.is_empty() {
            let mut stale: std::collections::BTreeMap<(u32, u32), (u32, u32)> = Default::default();
            for &(addr, len, was, header, bi) in &uwatch {
                if stream_watch_hash(&ctx.read_bytes(addr, len as usize)) != was {
                    let e = stale.entry((header, bi)).or_insert((0, addr));
                    e.0 += 1;
                }
            }
            if !stale.is_empty() {
                let list: Vec<String> = stale
                    .iter()
                    .map(|(&(h, bi), &(n, a))| format!("prog {h:#x} buf {bi} x{n} (e.g. {a:#x})"))
                    .collect();
                tracing::warn!(
                    target: "vitaslop::gxm",
                    "f{} UNIFORM BUFFER REWRITTEN AFTER ITS DRAW ({} of {} reads): {}",
                    self.cur_frame,
                    stale.values().map(|v| v.0).sum::<u32>(),
                    uwatch.len(),
                    list.join(", ")
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Whether `VITASLOP_DUMP_DRAW_GXP` names display frame `disp`: "all", a single frame
    /// "N", or an inclusive range "LO-HI". Cached, because it runs per draw.
    pub(super) fn dump_draw_frame_matches(disp: u64) -> bool {
        static WANT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        let Some(want) =
            WANT.get_or_init(|| vitaslop_platform::knobs::var("VITASLOP_DUMP_DRAW_GXP").ok())
        else {
            return false;
        };
        want == "all"
            || match want.split_once('-') {
                Some((lo, hi)) => match (lo.parse::<u64>(), hi.parse::<u64>()) {
                    (Ok(lo), Ok(hi)) => (lo..=hi).contains(&disp),
                    _ => false,
                },
                None => want.parse::<u64>().map(|f| f == disp).unwrap_or(false),
            }
    }

    /// `vhandle` and `streams` are the DRAW's bindings - the snapshot's, which a precomputed
    /// draw overlays - not a fresh read of the context.
    pub(super) fn dump_draw_gxp(
        &self,
        ctx: &GuestCtx,
        vhandle: u32,
        streams: &[u32; MAX_VERTEX_STREAMS],
        vref: &ProgramReflection,
        material: &crate::capture::FragmentMaterial,
        textures: &[crate::capture::BoundTexture],
        attributes: &[crate::capture::VertexAttribute],
        primitive: u32,
        index_count: u32,
        stride: u32,
        frag_buf: u32,
        index_addr: u32,
        index_elem: usize,
    ) {
        // The frame the dump keys on is the scheduler frame (`cur_frame`, the display-flip /
        // yield count set at each `on_frame_boundary`), which is exactly the `fNNNN` a shot is
        // labelled with in the headless runner. (`presents.len()` is NOT it: this title renders
        // through the GXM display queue and calls the raw `present` path only a couple of times.)
        let disp = self.cur_frame;
        if !Self::dump_draw_frame_matches(disp) {
            return;
        }
        // Bound the total lines dumped (a wide window over a 90-draws/frame scene is a lot of
        // output); `VITASLOP_DUMP_DRAW_GXP_CAP` raises it when a full multi-frame trace is wanted.
        let cap = std::env::var("VITASLOP_DUMP_DRAW_GXP_CAP")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(400);
        use std::sync::atomic::{AtomicU32, Ordering};
        static DUMPED: AtomicU32 = AtomicU32::new(0);
        if DUMPED.fetch_add(1, Ordering::Relaxed) >= cap {
            return;
        }
        let seq = self.scene.as_ref().map(|s| s.draws.len()).unwrap_or(0);
        let vh = self.vertex_program_header(vhandle);
        let fh = self.bound_fragment_program_header(ctx);
        // The caller has already reflected it - recomputing here would be a second
        // (and, with the ambient-mean cache, a differently-warmed) evaluation of the
        // same thing.
        let mat = material;
        // The depth/stencil/enable state rides on the line: a draw that paints nothing on
        // hardware and everything here (or the reverse) is usually a depth test against a
        // buffer the two engines fill differently, and nothing else in the dump said so.
        let rs = self.render_state(ctx);
        eprintln!(
            "DRAW frame={disp} seq={seq} pdraw={:#x} prim={primitive:#010x} idx={index_count} stride={stride} vprog={vh:#x} vhandle={vhandle:#x} fprog={fh:#x} fubuf={frag_buf:#x} \
             depth_func={}/{} depth_write={}/{} frag_enable={}/{} stencil_func={}/{} stencil_ops={:#x},{:#x},{:#x} stencil_ref={} stencil_masks={:#x}/{:#x} two_sided={} cull={} viewport_en={} clip_mode={} \
             viewport={:?} vp_extent={}x{} clip_rect={:?}",
            self.last_precomputed_block,
            rs.front_depth_func,
            rs.back_depth_func,
            rs.front_depth_write,
            rs.back_depth_write,
            rs.front_fragment_program_enable,
            rs.back_fragment_program_enable,
            rs.front_stencil_func,
            rs.back_stencil_func,
            rs.front_stencil_op_fail,
            rs.front_stencil_op_depth_fail,
            rs.front_stencil_op_depth_pass,
            rs.front_stencil_ref,
            rs.front_stencil_compare_mask,
            rs.front_stencil_write_mask,
            rs.two_sided,
            rs.cull_mode,
            rs.viewport_enable,
            rs.region_clip_mode,
            // >>> THE VIEWPORT ITSELF, and the TARGET SIZE IT IMPLIES.
            //
            // `viewport_en` alone says only whether the transform is in effect; it cannot say
            // which surface the draw was sized for. The GXM viewport is
            // `[xOffset,xScale,yOffset,yScale,zOffset,zScale]`, so `2*|xScale| x 2*|yScale|` is
            // the extent the guest is projecting into - the same reading
            // `Scene::adopt_viewport_extent` already makes. A block of draws whose implied
            // extent is not the pass's target is a block that was sized for a DIFFERENT target,
            // and that is invisible in every other line of this dump.
            rs.viewport,
            (2.0 * rs.viewport[1].abs()) as u32,
            (2.0 * rs.viewport[3].abs()) as u32,
            rs.region_clip,
        );
        // Where this draw's transform comes from, and what it is. A mesh rendered in the
        // wrong place is always one of three things: the wrong default uniform buffer still
        // bound (`STALE-ubuf`), no reflected projection matrix so offset 0 is used raw and
        // may not be an MVP at all (`composed=no`), or a leftover `sceGxmSetUniformDataF`
        // capture (`setUniformDataF` with a buffer of 0). This line separates them, and
        // prints the MVP's translation column - the clip-space position of the object's
        // origin, which is what "it is drawn at the wrong place" actually means.
        let vbound = self.vertex_uniform(ctx);
        let vbuf = vbound.buf;
        let owned = vbound.header == vh || vbound.header == USER_UNIFORM_BUFFER;
        let vsize = if vbound.header == USER_UNIFORM_BUFFER {
            vref.uniform_size_bytes.max(vref.default_uniform_bytes)
        } else {
            vbound.size
        };
        let usable = vbuf != 0 && vsize >= 4 && owned;
        let source = match (vbuf != 0, owned) {
            (true, true) => "ubuf",
            (true, false) => "STALE-ubuf",
            (false, _) => "setUniformDataF",
        };
        let raw: Vec<f32> = if usable {
            (0..(vsize / 4) as usize)
                .map(|i| ctx.read_f32(vbuf.wrapping_add(i as u32 * 4)))
                .collect()
        } else {
            self.sa_bank_floats(ctx)
        };
        let composed = self.composed_mvp(vref, &raw);
        let eff = composed.unwrap_or_else(|| {
            let mut m = [0f32; 16];
            for (i, s) in m.iter_mut().zip(raw.iter()) {
                *i = *s;
            }
            m
        });
        eprintln!(
            "  TRANSFORM source={source} ubuf={vbuf:#x} lanes={} composed={} origin_clip=[{:.3},{:.3},{:.3},{:.3}]",
            raw.len(),
            if composed.is_some() { "yes" } else { "no" },
            eff[12],
            eff[13],
            eff[14],
            eff[15],
        );
        // >>> THE NON-DEFAULT UNIFORM BUFFER TABLE, because a transform that is not in the
        // >>> default buffer is not in anything this dump used to print.
        //
        // Fourteen slots, written by `sceGxmSetVertexUniformBuffer` (and by a precomputed
        // state's own table). A program whose view-projection lives in container N reads it
        // through the buffer bound at slot N, so "slot N is 0" and "slot N holds an address"
        // are completely different diagnoses for the same wrong-looking mesh. Only the
        // non-zero slots are listed - fourteen zeros is the common case and says nothing.
        {
            use crate::vita::gxmctx;
            let bound: Vec<String> = (0..gxmctx::MAX_UNIFORM_BUFFERS as u32)
                .filter_map(|i| {
                    let a = if self.gxm_context != 0 {
                        gxmctx::vertex_uniform_buffer(ctx, self.gxm_context, i)
                    } else {
                        0
                    };
                    (a != 0).then(|| format!("{i}:{a:#x}"))
                })
                .collect();
            eprintln!(
                "  VUBUFS {}",
                if bound.is_empty() {
                    "none bound".to_string()
                } else {
                    bound.join(" ")
                },
            );
            // >>> AND THE BYTES OF ONE OF THEM (`VITASLOP_DUMP_VUBUF=<index>`), as floats.
            //
            // Knowing WHICH buffer a parameter lives in is only half of it: a matrix read at
            // the wrong OFFSET inside the right buffer looks exactly like a matrix the guest
            // never wrote. This prints the head of the buffer so the matrix can be LOOKED FOR
            // rather than assumed to start where `resource_index` says.
            if let Ok(want) = std::env::var("VITASLOP_DUMP_VUBUF")
                && let Ok(idx) = want.trim().parse::<u32>()
            {
                let addr = if self.gxm_context != 0 {
                    gxmctx::vertex_uniform_buffer(ctx, self.gxm_context, idx)
                } else {
                    0
                };
                if addr != 0 {
                    let f: Vec<String> = (0..64)
                        .map(|k| format!("{:.4}", ctx.read_f32(addr.wrapping_add(k * 4))))
                        .collect();
                    eprintln!("  VUBUF{idx}@{addr:#x} f0..63 = [{}]", f.join(","));
                }
            }
        }
        // The whole transform, column by column. `origin_clip` above is only its last
        // column, which says WHERE the origin lands and nothing about the scale or the
        // axes - and a mesh that misses the screen because its scale is wrong looks
        // exactly like one whose translation is wrong.
        eprintln!(
            "  MVP c0=[{:.4},{:.4},{:.4},{:.4}] c1=[{:.4},{:.4},{:.4},{:.4}] \
             c2=[{:.4},{:.4},{:.4},{:.4}] c3=[{:.4},{:.4},{:.4},{:.4}]",
            eff[0],
            eff[1],
            eff[2],
            eff[3],
            eff[4],
            eff[5],
            eff[6],
            eff[7],
            eff[8],
            eff[9],
            eff[10],
            eff[11],
            eff[12],
            eff[13],
            eff[14],
            eff[15],
        );
        // Every uniform lane the program declares, by NAME, so a transform that is not
        // in the first sixteen floats can be found rather than guessed at. This is the
        // dump's whole job on a shader whose transform the reflection does not know.
        self.dump_named_uniforms(ctx, vh, &raw);
        eprintln!(
            "  MATERIAL tint=[{:.3},{:.3},{:.3}] has_light={} light_dir=[{:.3},{:.3},{:.3}] light_col=[{:.3},{:.3},{:.3}] ambient=[{:.3},{:.3},{:.3}]",
            mat.tint[0],
            mat.tint[1],
            mat.tint[2],
            mat.has_light,
            mat.light_dir[0],
            mat.light_dir[1],
            mat.light_dir[2],
            mat.light_col[0],
            mat.light_col[1],
            mat.light_col[2],
            mat.ambient[0],
            mat.ambient[1],
            mat.ambient[2],
        );
        // Vertex stream attributes (bytes -> shader register): the count of distinct UV-ish
        // float2/float4 lanes is the number of texcoord sets the mesh actually carries.
        for a in attributes {
            eprintln!(
                "  attr stream={} off={} fmt={} comp={} reg={}",
                a.stream_index, a.offset, a.format, a.component_count, a.reg_index
            );
        }
        // >>> AND THE STREAMS THEMSELVES, because the line above prints the PACKED attributes.
        //
        // `set_vertex_program` rebases every attribute into one interleaved row and forces
        // `stream_index` to 0, so a multi-stream draw reads here as a single stream and the
        // real strides - the thing that decides how the gather STEPS - are invisible. A stream
        // whose attributes reach past its own stride reads OVERLAPPING rows, which comes out
        // as a mesh whose vertices each drift a few bytes further into the buffer.
        if let Some(info) = self.shader_patcher.vertex_program(vhandle) {
            eprintln!(
                "  streams packed_stride={} columns={:?} strides={:?} per_instance={:?} raw_attrs={:?}",
                info.packed_stride,
                info.used_streams
                    .iter()
                    .map(|s| s.column)
                    .collect::<Vec<_>>(),
                info.used_streams
                    .iter()
                    .map(|s| s.stride)
                    .collect::<Vec<_>>(),
                info.used_streams
                    .iter()
                    .map(|s| s.per_instance)
                    .collect::<Vec<_>>(),
                info.attributes
                    .iter()
                    .map(|a| (
                        a.stream_index,
                        a.offset,
                        a.format,
                        a.component_count,
                        a.reg_index
                    ))
                    .collect::<Vec<_>>(),
            );
            eprintln!(
                "  streams DECLARED={:?} bound_addrs={:x?}",
                info.streams
                    .iter()
                    .map(|s| (s.stride, s.per_instance))
                    .collect::<Vec<_>>(),
                &streams[..info.used_streams.len().min(4)],
            );
            // >>> AND THE GUEST'S OWN BYTES, straight from memory, per stream.
            //
            // Everything else about a stream is read through the GATHER, which applies the
            // declared stride - so a declared stride that disagrees with the buffer's real
            // record size is invisible in every downstream print. This one is the raw window:
            // read the period off it and compare with `strides=`.
            if let Some(dump_bytes) = dump_stream_bytes() {
                let bound = *streams;
                // >>> THE WINDOW THE GATHER ACTUALLY READS, not the buffer's base.
                //
                // An indexed draw reads `min_index .. max_index` of its stream, and a title
                // that suballocates one arena for many draws puts a draw's own rows a long way
                // in. Dumping from the base therefore shows some OTHER draw's records, and a
                // period read off them says nothing about this draw's stride - which is exactly
                // the wrong turn this print existed to prevent.
                let (first_vertex, vertex_count) = if index_addr != 0 && index_count > 0 {
                    let raw = ctx.read_bytes(index_addr, index_count as usize * index_elem);
                    let (lo, hi) = raw
                        .chunks(index_elem)
                        .fold((u32::MAX, 0u32), |(lo, hi), c| {
                            let i = match index_elem {
                                2 => u16::from_le_bytes([c[0], c[1]]) as u32,
                                _ => u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                            };
                            (lo.min(i), hi.max(i))
                        });
                    if lo > hi { (0, 0) } else { (lo, hi - lo + 1) }
                } else {
                    (0, 0)
                };
                eprintln!(
                    "  indices @{index_addr:#x} count={index_count} elem={index_elem}                      first_vertex={first_vertex} vertex_count={vertex_count}"
                );
                // The OTHER guest inputs a draw snapshots at the draw call, watched for the
                // same question the streams are: the index buffer and both stages' uniform
                // banks. If any of those is written after the draw too, deferring the vertex
                // read alone would be half a fix.
                {
                    let mut w = STREAM_WATCH.lock().unwrap_or_else(|e| e.into_inner());
                    let mut watch = |addr: u32, len: usize| {
                        if addr != 0 && len > 0 {
                            let b = ctx.read_bytes(addr, len);
                            w.push((addr, len as u32, stream_watch_hash(&b)));
                        }
                    };
                    watch(index_addr, index_count as usize * index_elem);
                    watch(vbuf, vbound.size as usize);
                    watch(frag_buf, 256);
                }
                for (si, st) in info.used_streams.iter().enumerate() {
                    let base = bound.get(si).copied().unwrap_or(0);
                    if base == 0 || st.column == 0 {
                        continue;
                    }
                    // A stride-0 (or per-instance) stream reads one row wherever it is bound;
                    // every other one starts at the first vertex the indices reach.
                    let addr = if st.per_instance || st.stride == 0 {
                        base
                    } else {
                        base.wrapping_add(first_vertex.wrapping_mul(st.stride))
                    };
                    let want = dump_bytes.unwrap_or(st.column as usize * 8).min(1 << 20);
                    let raw = ctx.read_bytes(addr, want);
                    // The watch covers the draw's OWN window exactly - `vertex_count` rows of
                    // this stream, or the single row a stride-0 stream repeats - not the
                    // (larger, adjustable) hex print above. A watch that overran into the next
                    // draw's rows would report every draw as rewritten, which is no report.
                    let watch_len = if st.per_instance || st.stride == 0 {
                        st.column
                    } else {
                        vertex_count.saturating_mul(st.stride)
                    };
                    if watch_len > 0 {
                        let win = ctx.read_bytes(addr, watch_len as usize);
                        STREAM_WATCH
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .push((addr, watch_len, stream_watch_hash(&win)));
                    }
                    eprintln!(
                        "  stream{si} @{addr:#x} (base {base:#x}) stride={} column={} first {} bytes: {}",
                        st.stride,
                        st.column,
                        raw.len(),
                        raw.iter()
                            .map(|b| format!("{b:02x}"))
                            .collect::<Vec<_>>()
                            .join("")
                    );
                }
            }
        }
        // Sampler name at a given unit, by reflecting the fragment program.
        let sampler_name = |unit: u32| -> String { self.gxp_sampler_name(ctx, fh, unit) };
        for t in textures {
            // Fraction of the snapshotted bytes that are non-zero. A render-target texture the
            // engine never rendered into (a shadow map, a reflection probe) reads as an all-zero
            // buffer, and a shader multiplying by it paints only its ambient term - which looks
            // like a lighting bug rather than a missing render pass unless this is measured.
            let nonzero = if t.pixels.is_empty() {
                0.0
            } else {
                t.pixels.iter().filter(|&&b| b != 0).count() as f32 / t.pixels.len() as f32
            };
            // The guest DATA ADDRESS is printed with the rest, because an all-zero texture is
            // only half an answer: what decides whether it is a missing render pass or a buffer
            // the guest genuinely left empty is WHICH address it is, and that is the join to
            // `VITASLOP_PEEK`, the `refs` tool and the scene-target list.
            eprintln!(
                "  tex unit={} {}x{} base_fmt={:#x} swizzle={:#x} type={} addr={:#010x} stride={} levels={} mip_filter={} lod_bias={} nonzero={nonzero:.3} sampler={:?}",
                t.unit,
                t.width,
                t.height,
                t.base_format,
                t.swizzle,
                t.tex_type,
                t.data_addr,
                t.stride,
                t.levels,
                t.mip_filter,
                t.lod_bias,
                sampler_name(t.unit)
            );
        }
        if std::env::var("VITASLOP_DUMP_DRAW_GXP_FULL").is_ok() {
            eprintln!("  -- fragment program params --");
            self.dump_all_params(ctx, fh);
            eprintln!("  -- vertex program params --");
            self.dump_all_params(ctx, vh);
        }
        // VITASLOP_DUMP_TEX_DIR=<dir>: also write each bound texture, decoded to RGBA8 via the
        // exact sampler decode, as a PNG named `f<frame>_seq<seq>_u<unit>_<sampler>.png`. Lets a
        // human see the ACTUAL texel content a draw sampled (is the sampled albedo the paint, or
        // a decal atlas whose UVs the shader would remap?) - the instrument for deciding whether
        // a render artifact is a decode/pick bug or the untranslated fragment-program composite.
        if let Ok(dir) = std::env::var("VITASLOP_DUMP_TEX_DIR") {
            use std::sync::Mutex;
            // De-dup by guest data address: the shared shadow/ambient/atlas maps are bound in
            // every draw, so writing them per-draw would emit gigabytes. Each unique texture is
            // written once. Maps above `VITASLOP_DUMP_TEX_MAX_TEXELS` (default 1 Mtexel) are
            // skipped - the material inputs usually worth eyeballing are the small per-part
            // albedo/normal sheets, but raising the cap is how you inspect a render-target map
            // such as the 4096x2048 shadMap when the question is whether it holds real content.
            static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
            let max_texels = std::env::var("VITASLOP_DUMP_TEX_MAX_TEXELS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(1 << 20);
            let _ = std::fs::create_dir_all(&dir);
            for t in textures {
                if t.width == 0 || t.height == 0 || t.width * t.height > max_texels {
                    continue;
                }
                {
                    let mut seen = SEEN.lock().unwrap();
                    if seen.contains(&t.data_addr) {
                        continue;
                    }
                    seen.push(t.data_addr);
                }
                let (w, h, rgba) = crate::render::decode_texture_rgba8(t);
                let samp = self.gxp_sampler_name(ctx, fh, t.unit);
                let safe: String = samp.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
                let path = std::path::Path::new(&dir)
                    // >>> THE GUEST ADDRESS IS IN THE NAME, because the de-dup above is BY
                    // >>> address and the frame/seq in the name belong to whichever draw
                    // >>> happened to bind it FIRST - not to the draw being investigated.
                    //
                    // Every other diagnostic names a texture by its address (the sampler
                    // lines, the chain-draw trace), so without it here a dumped PNG cannot be
                    // joined to the draw that sampled it, and a question of the form "what is
                    // in the albedo THIS draw bound" has to be answered by guessing from the
                    // extent.
                    .join(format!(
                        "f{disp}_seq{seq}_{:08x}_u{}_{}.png",
                        t.data_addr, t.unit, safe
                    ));
                let _ = std::fs::write(path, crate::render::rgba_to_png(w, h, &rgba));
            }
        }
        // VITASLOP_GXP_RECOMPILE=1: the explicit clean-room shader-recompiler GRIND mode.
        // For each unique bound fragment AND vertex program, read its container from guest
        // memory and recompile the guest USSE to a complete, bindable WGSL module. This is NOT
        // a silent fixed-function fallback: on any unsupported opcode / decode failure it
        // HARD-FAILS (panics) naming the exact instruction + opcode to implement next, exactly
        // like the NID grind. The ordinary renderer runs unchanged when this is unset (the
        // recompiler is simply not engaged - a separate default renderer, not a fallback here).
        if std::env::var_os("VITASLOP_GXP_RECOMPILE").is_some() {
            use std::sync::Mutex;
            static SEEN: Mutex<Vec<u32>> = Mutex::new(Vec::new());
            for (kind, header) in [("frag", fh), ("vert", vh)] {
                if header == 0 {
                    continue;
                }
                let first = {
                    let mut seen = SEEN.lock().unwrap();
                    if seen.contains(&header) {
                        false
                    } else {
                        seen.push(header);
                        true
                    }
                };
                if !first {
                    continue;
                }
                let size = ctx.read_u32(header.wrapping_add(0x08)).clamp(0x40, 0x40000);
                let bytes = ctx.read_bytes(header, size as usize);
                match kind {
                    "frag" => match vitaslop_gxp_shader::recompile_fragment_module(&bytes) {
                        Ok((r, m)) => tracing::info!(
                            target: "vitaslop::gxp",
                            header = format_args!("{header:#010x}"),
                            instrs = r.shader.instrs.len(),
                            pa = m.bindings.pa_lane_count,
                            sa = m.bindings.sa_lane_count,
                            samplers = m.bindings.samplers.len(),
                            color = format_args!("{:?}", m.bindings.color),
                            "recompiled FRAGMENT program to a bindable WGSL module ({} chars)",
                            m.wgsl.len(),
                        ),
                        Err(e) => {
                            panic!("VITASLOP_GXP_RECOMPILE: fragment program {header:#010x}: {e}")
                        }
                    },
                    _ => match vitaslop_gxp_shader::recompile_vertex_module(&bytes) {
                        Ok((r, m)) => tracing::info!(
                            target: "vitaslop::gxp",
                            header = format_args!("{header:#010x}"),
                            instrs = r.shader.instrs.len(),
                            attributes = m.bindings.attributes.len(),
                            sa = m.bindings.sa_lane_count,
                            varyings = m.bindings.varying_vec4s,
                            "recompiled VERTEX program to a bindable WGSL module ({} chars)",
                            m.wgsl.len(),
                        ),
                        Err(e) => {
                            panic!("VITASLOP_GXP_RECOMPILE: vertex program {header:#010x}: {e}")
                        }
                    },
                }
            }
        }
    }

    /// Reflect the name of the fragment sampler bound to texture `unit`, or "" if none.
    pub(super) fn gxp_sampler_name(&self, ctx: &GuestCtx, header: u32, unit: u32) -> String {
        if header == 0 {
            return String::new();
        }
        let count = ctx.read_u32(header.wrapping_add(0x24));
        let base = header
            .wrapping_add(0x28)
            .wrapping_add(ctx.read_u32(header.wrapping_add(0x28)));
        for i in 0..count.min(256) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            if word & 0xf != 2 {
                continue;
            }
            if ctx.read_u32(p.wrapping_add(0xc)) == unit {
                let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
                let raw = ctx.read_bytes(name_addr, 64);
                return raw
                    .iter()
                    .take_while(|&&b| b != 0)
                    .map(|&b| b as char)
                    .collect();
            }
        }
        String::new()
    }

    /// Dump every parameter (all categories) of a GXP program: category / type /
    /// component_count / container_index / array_size / resource_index / name. Category
    /// 0 = attribute (a fragment program's INPUT varyings, e.g. TEXCOORD0/1), 1 = uniform,
    /// 2 = sampler, 4 = uniform buffer.
    pub(super) fn dump_all_params(&self, ctx: &GuestCtx, header: u32) {
        if header == 0 {
            eprintln!("    (no program)");
            return;
        }
        let count = ctx.read_u32(header.wrapping_add(0x24));
        let base = header
            .wrapping_add(0x28)
            .wrapping_add(ctx.read_u32(header.wrapping_add(0x28)));
        for i in 0..count.min(128) {
            let p = base.wrapping_add(i.wrapping_mul(16));
            let word = ctx.read_u32(p.wrapping_add(4)) & 0xffff;
            let array_size = ctx.read_u32(p.wrapping_add(8));
            let res_index = ctx.read_u32(p.wrapping_add(0xc));
            let name_addr = (p as i64 + ctx.read_u32(p) as i32 as i64) as u32;
            let raw = ctx.read_bytes(name_addr, 64);
            let name: String = raw
                .iter()
                .take_while(|&&b| b != 0)
                .map(|&b| b as char)
                .collect();
            let cat = match word & 0xf {
                0 => "attr",
                1 => "unif",
                2 => "samp",
                4 => "ubuf",
                _ => "?",
            };
            eprintln!(
                "    [{i}] {cat} type={} comp={} container={} array={array_size} res={res_index} name={name:?}",
                (word >> 4) & 0xf,
                (word >> 8) & 0xf,
                (word >> 12) & 0xf,
            );
        }
    }

    pub fn end_scene(&mut self) {
        if let Some(mut scene) = self.scene.take() {
            scene.adopt_viewport_extent();
            // The draws' geometry stays a DESCRIPTION until the guest's GPU wait or flip -
            // see `resolve_deferred_geometry`. Filed under a serial the scene carries.
            if !self.deferred_geometry.is_empty() {
                self.deferred_serial += 1;
                scene.deferred_id = self.deferred_serial;
                self.pending_geometry.push((
                    self.deferred_serial,
                    std::mem::take(&mut self.deferred_geometry),
                ));
            }
            // Diagnostic (`RUST_LOG=vitaslop::display=trace`): which buffer this scene drew
            // into and how many draws it holds, in submission order, beside the flip trace.
            // The renderer presents the LAST scene's target; a frame whose flipped buffer is a
            // different one is showing a different image, and only these two traces together
            // can say which.
            tracing::trace!(
                target: "vitaslop::display",
                thid = self.current_thread(),
                target_addr = format_args!("{:#010x}", scene.color.as_ref().map_or(0, |c| c.data_addr)),
                draws = scene.draws.len(),
                "end scene"
            );
            // Hand the renderer whatever the shader patcher has named since the last scene, so
            // it can prepare those pairs before it encodes. Riding the scene keeps this on the
            // path the renderer already consumes - no engine has to learn a new call - and it
            // reaches the renderer at the first scene AFTER the loading screen's creates, which
            // is exactly where the spare time is.
            scene.precompile = self.shader_precompile();
            // A raw 64-bit target this scene renders is one a later pass's memory windows are
            // filled from on the GPU - the renderer registers it under the same test (a
            // 64-bit format, a pass that draws something).
            if let Some(c) = scene.color.as_ref()
                && vitaslop_platform::gpu::raw64_color_format(c.format)
                && !scene.draws.is_empty()
            {
                let pitch = if c.stride_pixels != 0 { c.stride_pixels } else { c.width };
                let t = (c.data_addr, c.width, c.height, pitch);
                if !self.frame_raw_targets.contains(&t) {
                    self.frame_raw_targets.push(t);
                }
            }
            // Goes through `push_scene`, not a bare push, so a bounded-retention run
            // still folds every scene into the determinism signature.
            self.capture.push_scene(scene);
        }
    }

    pub fn present(&mut self, buffer_addr: u32) {
        // Under an async flip resolve the flipped frame's scenes are STILL pending here by
        // design (the presenter reads them before taking), so this report would misfire.
        if self.flip_resolve_upto.is_none() {
            report_present_no_scene_rendered(buffer_addr, &self.capture, self.cur_frame);
        }
        self.capture.presents.push(buffer_addr);
        self.flip_pacer.note_present();
    }

    /// Append bytes the guest printed to the debug console (sceClibPrintf).
    pub fn write_stdout(&mut self, bytes: &[u8]) {
        self.capture.stdout.extend_from_slice(bytes);
    }
}
