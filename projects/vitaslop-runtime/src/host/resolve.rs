use super::*;

/// [`dump_stream_bytes`] only: the (address, length, checksum) of every vertex-stream
/// window the per-draw dump read, so [`VitaState::report_stream_rewrites`] can re-read them at
/// `sceGxmEndScene` and say which the guest wrote AFTER its draw call.
///
/// GXM defers: `sceGxmDraw` records the draw and the GPU reads the vertex buffer when the scene
/// is submitted. A title that reserves, draws, and only then FILLS is therefore legal on the
/// device and catastrophic here, because this engine snapshots a draw's vertices at the draw
/// call - it would capture whatever the ring held from its previous use, and nothing else in the
/// engine can tell that apart from a wrong stride.
pub(super) static STREAM_WATCH: std::sync::Mutex<Vec<(u32, u32, u64)>> = std::sync::Mutex::new(Vec::new());

/// [`watch_uniform_rewrites`] only: `(address, length, checksum, program header, buffer index)`
/// of every SA-resident uniform buffer a draw read, re-read at `sceGxmEndScene` by
/// [`VitaState::report_stream_rewrites`]. The uniform twin of [`STREAM_WATCH`]: the GPU loads a
/// bound uniform buffer when the scene EXECUTES, and this engine copies it at the draw call.
pub(super) static UNIFORM_WATCH: std::sync::Mutex<Vec<(u32, u32, u64, u32, u32)>> =
    std::sync::Mutex::new(Vec::new());

/// `VITASLOP_WATCH_UNIFORM_REWRITES=1`: watch every SA uniform buffer a draw read for a guest
/// write between `sceGxmDraw` and `sceGxmEndScene` - see [`UNIFORM_WATCH`].
pub(super) fn watch_uniform_rewrites() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::knobs::var("VITASLOP_WATCH_UNIFORM_REWRITES")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// `VITASLOP_DUMP_STREAM_BYTES[=<n>]`: with the per-draw dump on, print each vertex stream's
/// raw guest bytes from the window the GATHER reads, and watch every such window for a rewrite.
///
/// `Some(Some(n))` prints `n` bytes per stream, `Some(None)` a default eight rows, `None` is
/// off. It is the one instrument that can see a title FILL a vertex buffer after it draws from
/// it - see [`defer_geometry`] and [`STREAM_WATCH`] - and the reason it takes a byte count is
/// that a period read off the first few records is a period of whatever was there before.
pub(super) fn dump_stream_bytes() -> Option<Option<usize>> {
    use std::sync::OnceLock;
    static ON: OnceLock<Option<Option<usize>>> = OnceLock::new();
    *ON.get_or_init(|| {
        let v = std::env::var("VITASLOP_DUMP_STREAM_BYTES").ok()?;
        Some(v.trim().parse::<usize>().ok().filter(|n| *n > 0))
    })
}

/// FNV-1a over a byte window, the checksum [`STREAM_WATCH`] compares with.
pub(super) fn stream_watch_hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(1469598103934665603u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(1099511628211)
    })
}

/// One scene's share of a [`ResolveJob`]: its draw descriptions, each draw's uniform windows
/// (moved out of the capture while the job is out), and - once computed - each draw's
/// `(indices, vertices)`.
pub struct ResolveScene {
    pub(super) id: u64,
    /// A later CHUNK of a scene whose texture epoch an earlier chunk already opened - see
    /// [`compute_posted_resolve_job`]. Its `begin_scene` is skipped.
    pub(super) continues: bool,
    pub(super) geo: Vec<DeferredGeometry>,
    pub(super) windows: Vec<(Vec<(u32, Arc<[u8]>)>, Vec<(u32, Arc<[u8]>)>)>,
    pub(super) out: Vec<(Arc<[u8]>, Arc<[u8]>)>,
    /// Per draw, its textures when they were deferred ([`DeferredTextures`]); empty when no
    /// draw of the scene deferred them.
    pub(super) tex_out: Vec<Option<ResolvedTextures>>,
}

/// The deferred geometry of some ended scenes, taken out of the host state so it can be READ
/// without the host lock - see [`VitaState::take_flip_resolve_job`]. Computed by
/// [`compute_resolve_job`], put back by [`VitaState::apply_posted_resolve`].
pub struct ResolveJob {
    pub(super) frame: u64,
    pub(super) scenes: Vec<ResolveScene>,
}

/// The hand-off between a presenting worker reading a [`ResolveJob`] and a guest thread's sync
/// point that must not return before those bytes are in the capture.
#[derive(Default)]
pub struct AsyncResolve {
    pub(super) slot: std::sync::Mutex<AsyncResolveSlot>,
    pub(super) cv: std::sync::Condvar,
}

#[derive(Default)]
pub(super) struct AsyncResolveSlot {
    /// Jobs taken and not yet posted.
    pub(super) in_flight: u32,
    /// Jobs computed and not yet applied.
    pub(super) done: Vec<ResolveJob>,
    /// Jobs a flip queued for the RESOLVER worker (see [`run_resolver`]) and it has not picked.
    pub(super) queue: std::collections::VecDeque<ResolveJob>,
    /// The resolver's loop ends.
    pub(super) stop: bool,
}

impl AsyncResolve {
    /// Block until no job is in flight. Takes no host lock; a caller holding one is fine,
    /// because computing a job needs only the snapshot cache's lock.
    pub fn wait_idle(&self) {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        while slot.in_flight > 0 {
            slot = self.cv.wait(slot).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// End [`run_resolver`]'s loop.
    pub fn stop(&self) {
        self.slot.lock().unwrap_or_else(|e| e.into_inner()).stop = true;
        self.cv.notify_all();
    }
}

/// Whether a RESOLVER worker is running [`run_resolver`] - see [`set_resolver_attached`].
pub(super) static RESOLVER_ATTACHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Tell the host a resolver worker is looping in [`run_resolver`]: from now on a flip under
/// `async_flip_resolve` QUEUES its geometry for that worker instead of leaving a boundary for
/// the presenting worker. Set only once the worker is up - a queued job with nobody to take it
/// would hold every sync point for ever.
pub fn set_resolver_attached(on: bool) {
    RESOLVER_ATTACHED.store(on, std::sync::atomic::Ordering::SeqCst);
}

/// `VITASLOP_SYNC_RESOLVE_ASYNC=0` - the arm back to reading a sync point's geometry inline under
/// the host lock. See [`VitaState::queue_sync_resolve`].
pub(super) fn sync_resolve_async() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::knobs::var("VITASLOP_SYNC_RESOLVE_ASYNC")
            .as_deref()
            .map(str::trim)
            != Ok("0")
    })
}

pub(super) fn resolver_attached() -> bool {
    RESOLVER_ATTACHED.load(std::sync::atomic::Ordering::SeqCst)
}

/// The two handles the resolver worker needs - see [`VitaEnv::resolver_handles`].
pub struct ResolverHandles {
    pub(super) post: Arc<AsyncResolve>,
    pub(super) snaps: Arc<std::sync::Mutex<TextureSnapshots>>,
}

impl ResolverHandles {
    pub fn post(&self) -> Arc<AsyncResolve> {
        self.post.clone()
    }
}

/// >>> THE RESOLVER WORKER'S LOOP: read each flip's deferred geometry OFF every guest thread
/// and off the presenting worker. The flipping thread only moves its frame's descriptions into
/// a job ([`VitaState::resolve_at_flip`]); this takes it, reads guest memory under the snapshot
/// cache's own lock, and posts the bytes for whoever holds the host lock next
/// ([`VitaState::apply_posted_resolve`] - the presenter before it takes the frame, or a sync
/// point, which waits for it). Returns when [`AsyncResolve::stop`] is called. Blocks; run it
/// on a worker of its own.
pub fn run_resolver(h: ResolverHandles, mem: &mut dyn GuestMemory, base: u32) {
    let mut regs = [0u32; REG_COUNT];
    let mut vfp = [0u32; VFP_ARG_COUNT];
    let ctx = GuestCtx::new(&mut regs, &mut vfp, mem, base);
    loop {
        let job = {
            let mut slot = h.post.slot.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if slot.stop {
                    return;
                }
                if let Some(j) = slot.queue.pop_front() {
                    break j;
                }
                slot = h.post.cv.wait(slot).unwrap_or_else(|e| e.into_inner());
            }
        };
        compute_posted_resolve_job(job, &h.post, &h.snaps, &ctx);
    }
}

/// A [`ResolveJob`] with the two handles its computation needs - see
/// [`VitaEnv::take_flip_resolve_job`].
pub struct FlipResolveJob {
    pub(super) job: ResolveJob,
    pub(super) post: Arc<AsyncResolve>,
    pub(super) snaps: Arc<std::sync::Mutex<TextureSnapshots>>,
}

/// Compute a taken flip job over guest memory `mem` (region base `base`) and post it. Takes no
/// host lock - only the snapshot cache's own.
pub fn compute_flip_resolve_job(j: FlipResolveJob, mem: &mut dyn GuestMemory, base: u32) {
    let mut regs = [0u32; REG_COUNT];
    let mut vfp = [0u32; VFP_ARG_COUNT];
    let ctx = GuestCtx::new(&mut regs, &mut vfp, mem, base);
    compute_posted_resolve_job(j.job, &j.post, &j.snaps, &ctx);
}

/// Read a taken job off the host lock (the snapshot cache has its own) and post it for
/// [`VitaState::apply_posted_resolve`]. See [`VitaState::take_flip_resolve_job`].
pub(super) fn compute_posted_resolve_job(
    mut job: ResolveJob,
    post: &AsyncResolve,
    snaps: &std::sync::Mutex<TextureSnapshots>,
    ctx: &GuestCtx,
) {
    {
        let _all = crate::perf::scope(crate::perf::Phase::DrawTotal);
        match resolve_chunk() {
            0 => {
                let mut s = snaps.lock().unwrap_or_else(|e| e.into_inner());
                crate::perf::snaps_taken(crate::site!());
                compute_resolve_job(&mut s, ctx, &mut job);
            }
            chunk => compute_resolve_job_chunked(&mut job, snaps, ctx, chunk),
        }
    }
    let mut slot = post.slot.lock().unwrap_or_else(|e| e.into_inner());
    slot.done.push(job);
    slot.in_flight -= 1;
    post.cv.notify_all();
}

/// `VITASLOP_RESOLVE_CHUNK=<draws>`: how many draws the RESOLVER reads per hold of the snapshot
/// lock (default 16); `0` holds it for the whole job, which is what this did before. 16, not 64:
/// the render thread's `sceGxmDraw` window capture waits on whole chunks - MEASURED on the phone
/// (a baseball title's pitches, the lock's waits-by-holder line) 1.07 ms/f at 64, 0.70 at 16 (039).
pub(super) fn resolve_chunk() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        crate::knobs::var("VITASLOP_RESOLVE_CHUNK")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(16)
    })
}

/// >>> [`compute_resolve_job`] in CHUNKS of draws, releasing the snapshot lock between them.
///
/// The guest's render thread takes the same lock on every `sceGxmDraw` (the memory-window
/// snapshots), and the resolver used to hold it for its whole flip-time read - MEASURED on the
/// phone at a baseball at-bat, 2.0 ms of every frame waiting on this lock, most of it the
/// render thread sitting behind the resolver's ~7 ms job while it recorded the next frame. With
/// chunks a draw waits for at most one chunk. What is read, from where and when is unchanged:
/// each chunk is the same computation over a slice of the same scene, and the outputs are put
/// back in draw order.
///
/// Two things a slice has to be handed so it computes what the whole scene would have:
/// * the SAMPLER SPAN of a draw that said "same bytes as the previous draw" (`span: None`) when
///   that previous draw is in an earlier chunk - it is filled in with those very bytes;
/// * the scene's texture EPOCH, opened once per scene (`ResolveScene::continues`).
pub(super) fn compute_resolve_job_chunked(
    job: &mut ResolveJob,
    snaps: &std::sync::Mutex<TextureSnapshots>,
    ctx: &GuestCtx,
    chunk: usize,
) {
    for rs in job.scenes.iter_mut() {
        let n = rs.geo.len();
        let mut geo = std::mem::take(&mut rs.geo);
        let mut windows = std::mem::take(&mut rs.windows);
        let (mut geo_back, mut win_back) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut out, mut tex_out) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let mut last_span: Option<Box<[u8]>> = None;
        let mut began = false;
        let mut first = true;
        while !geo.is_empty() {
            let k = geo.len().min(chunk);
            let mut g: Vec<DeferredGeometry> = geo.drain(..k).collect();
            let w: Vec<_> = windows.drain(..k).collect();
            let mut need_fill = !first;
            for d in g.iter_mut() {
                if let Some(t) = d.tex.as_mut() {
                    match &t.span {
                        Some(sp) => last_span = Some(sp.clone()),
                        None if need_fill => t.span = last_span.clone(),
                        None => {}
                    }
                    need_fill = false;
                }
            }
            let has_tex = g.iter().any(|d| d.tex.is_some());
            let mut part = ResolveJob {
                frame: job.frame,
                scenes: vec![ResolveScene {
                    id: rs.id,
                    continues: has_tex && began,
                    geo: g,
                    windows: w,
                    out: Vec::new(),
                    tex_out: Vec::new(),
                }],
            };
            began |= has_tex;
            first = false;
            {
                let mut s = snaps.lock().unwrap_or_else(|e| e.into_inner());
                crate::perf::snaps_taken(crate::site!());
                compute_resolve_job(&mut s, ctx, &mut part);
            }
            let sc = part.scenes.pop().expect("one scene in, one out");
            out.extend(sc.out);
            // A slice with no deferred textures leaves `tex_out` empty; the whole scene's list is
            // positional, so it is padded to keep every later slice in line.
            if sc.tex_out.is_empty() {
                tex_out.extend((0..sc.geo.len()).map(|_| None));
            } else {
                tex_out.extend(sc.tex_out);
            }
            geo_back.extend(sc.geo);
            win_back.extend(sc.windows);
        }
        rs.geo = geo_back;
        rs.windows = win_back;
        rs.out = out;
        rs.tex_out = if tex_out.iter().all(Option::is_none) {
            Vec::new()
        } else {
            tex_out
        };
    }
}

/// Read every draw of `job`: its index bytes, the vertex rows they reach, and any uniform
/// window the draw-time snapshot says had not been written yet. `job.scenes[..].out` is filled
/// in draw order.
///
/// >>> READ AT THE GUEST'S GPU WAIT OR ITS FLIP, NOT AT `sceGxmEndScene`.
///
/// EndScene was the deferral point until a baseball title's crowd refuted it. Its crowd
/// sections are 141-sprite vertex buffers a KINEMATICS JOB THREAD fills while the render
/// thread records and ends the scene: at EndScene the first quad was written and the
/// rest still held the arena's previous user - offsets of 100-300 units and one of
/// 2.6e33 - so three of the sections covered the whole frame with a tiled atlas and cost
/// the GPU 11 ms a frame. The title's own contract is that the bytes are in place by the
/// time the GPU consumes them, and the latest point that is knowable here is the
/// guest's own wait for that GPU work (`sceGxmFinish` / a notification wait) or its
/// flip. Every scene ended since the last of those is read now.
pub(super) fn compute_resolve_job(snaps: &mut TextureSnapshots, ctx: &GuestCtx, job: &mut ResolveJob) {
    let _job = crate::perf::scope(crate::perf::Phase::ResolveJob);
    let _owner = vitaslop_platform::heap::owner_scope(vitaslop_platform::heap::Owner::Resolve);
    // >>> THE TEXTURES FIRST, OVER LIVE MEMORY - see [`DeferredTextures`]. Before the geometry,
    // because the geometry below arms its epoch and then copies the dirty map into the overlay:
    // a texture proven through that COPY after a later epoch was armed could restamp bytes the
    // copy is too old to call dirty. Proven here, each scene re-arms the verifier as the draw-
    // time path did at `begin_scene`, and the geometry then uses the epoch the last of them
    // armed - taken before its copy, as it requires.
    for rs in job.scenes.iter_mut() {
        if rs.geo.iter().all(|g| g.tex.is_none()) {
            continue;
        }
        if !rs.continues {
            snaps.begin_scene();
        }
        let _t = crate::perf::scope(crate::perf::Phase::DrawTextureBind);
        let mut prev: Option<TexSet> = None;
        let mut prev_vert: Option<(u64, TexSet)> = None;
        let mut raw = Vec::new();
        let mut tex_out = Vec::with_capacity(rs.geo.len());
        for g in rs.geo.iter() {
            let Some(t) = &g.tex else {
                tex_out.push(None);
                continue;
            };
            let frag = match &t.span {
                // The same bytes as the previous draw of this scene: the same list, exactly as
                // `set_from_previous_draw` answers at the draw.
                None => prev
                    .clone()
                    .unwrap_or_else(|| TexSet::Hit(crate::capture::no_textures())),
                Some(span) => {
                    crate::vita::gxmctx::texture_bindings_in_span(span, &mut raw);
                    // >>> ONLY THE UNITS THIS PROGRAM SAMPLES - `draw_textures_now`'s narrowing.
                    let narrow = t.want != 0 && sampler_narrow_enabled();
                    let binds: Vec<TextureBinding> = raw
                        .iter()
                        .filter(|(unit, _)| {
                            !(narrow && (*unit >= 32 || t.want & (1u32 << *unit) == 0))
                        })
                        .map(|(unit, b)| TextureBinding {
                            unit: *unit,
                            addr: b.addr,
                            words: b.words,
                            from_precomputed: b.from_precomputed,
                        })
                        .collect();
                    let key = fold_set_key(t.fheader as u64, &binds);
                    match snaps.set_validated(ctx, key, t.fheader as u64, &binds) {
                        Some(list) => TexSet::Hit(list),
                        None => TexSet::Miss(key, binds.into()),
                    }
                }
            };
            prev = Some(frag.clone());
            let vert = t.vert.as_ref().map(|(generation, binds)| match &prev_vert {
                Some((g2, set)) if g2 == generation => set.clone(),
                _ => {
                    let key = fold_set_key(VERTEX_STAGE_TAG, binds);
                    let set = match snaps.set_validated(ctx, key, VERTEX_STAGE_TAG, binds) {
                        Some(list) => TexSet::Hit(list),
                        None => TexSet::Miss(key, binds.clone()),
                    };
                    prev_vert = Some((*generation, set.clone()));
                    set
                }
            });
            tex_out.push(Some(ResolvedTextures {
                frag,
                vert,
                fheader: t.fheader,
                albedo: t.albedo,
            }));
        }
        rs.tex_out = tex_out;
    }
    // >>> TWO CROSSINGS FOR THE WHOLE RESOLVE, NOT ONE PER RANGE. See `PrefetchedMemory`.
    // Stage 1: every draw's index bytes and uniform windows (addresses known now), plus
    // the dirty map. Stage 2, after the index scans: the vertex rows each draw covers.
    // Instanced draws keep their own path (their expansion reads through the base).
    //
    // >>> AND ONLY THE RANGES THAT ARE GOING TO BE READ. The snapshot caches below
    // (indices, single-stream vertices, windows, the gather memo) answer from the dirty
    // map without a read for anything the guest has not stored into since - most of a
    // steady frame. The first cut of this table fetched every range regardless, and the
    // DESKTOP, where a crossing was already cheap, paid for it: a baseball title's cpu p10 10.7 -> 12.1
    // ms, a football title's 16.2 -> 19.2, every byte of a hit copied into the overlay and then never
    // looked at. So the stamp copy is taken FIRST, each candidate is probed against its
    // cache exactly as the read below will probe it, and a hit stays out of the table.
    // The probe is the same test the read makes, on the same copy of the map, so a range
    // left out is a range whose read never reaches guest memory.
    //
    // The scene stamp is armed before the copy (the reads below would arm it anyway):
    // a wrapped epoch clears every stamp, and then every probe misses, honestly.
    snaps.arm_scene_stamp(ctx);
    let mut overlay = PrefetchedMemory::new(&*ctx.mem, &[]);
    let (mut fetched, mut skipped) = (0usize, 0usize);
    {
        let _s1 = crate::perf::scope(crate::perf::Phase::ResolveStage1);
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let pctx = GuestCtx::new(&mut regs, &mut vfp, &mut overlay, ctx.base);
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for rs in &job.scenes {
            for ((mw, fw), g) in rs.windows.iter().zip(rs.geo.iter()) {
                if g.index_wrap == 0 && g.index_len > 0 {
                    if snaps.indices_cached(&pctx, g.index_addr, g.index_len, g.index_elem) {
                        skipped += 1;
                    } else if let Some(o) = pctx.offset(g.index_addr) {
                        ranges.push((o, g.index_len));
                    }
                }
                for (addr, bytes) in mw.iter().chain(fw.iter()) {
                    // A window on a raw target this frame rendered was read at the draw, all
                    // but the bytes the GPU fills - see `get_or_read_covered_window`. Its
                    // re-read below touches only its uncovered bytes, so nothing is fetched.
                    if snaps.is_covered_window(bytes) {
                        skipped += 1;
                        continue;
                    }
                    // A DEFERRED window's bytes are the shared all-zero placeholder, which
                    // is blank by construction: re-read it without scanning ~a thousand
                    // placeholders a frame to rediscover that. MEASURED in the browser on a
                    // football title: this loop was half of `resolve_deferred_geometry`,
                    // itself 6.6% of the busy worker.
                    let wants = if snaps.is_window_placeholder(bytes) {
                        note_placeholder_window();
                        true
                    } else {
                        window_wants_reread(bytes)
                    };
                    if wants {
                        if snaps.window_cached(&pctx, *addr, bytes.len()) {
                            skipped += 1;
                        } else if let Some(o) = pctx.offset(*addr) {
                            ranges.push((o, bytes.len()));
                        }
                    }
                }
            }
        }
        fetched += ranges.len();
        overlay.extend(&ranges);
    }
    // Stage 2 needs each draw's row range, which is what the index scan yields; the scan
    // itself reads through the overlay (stage 1 holds the bytes) and lands in the
    // indices cache, so the loop below finds it there. The rows are probed the same way:
    // a single-stream snapshot or a gather memo the map proves current is not fetched.
    {
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let pctx = GuestCtx::new(&mut regs, &mut vfp, &mut overlay, ctx.base);
        let mut rows: Vec<(usize, usize)> = Vec::new();
        let mut key = GatherKey::default();
        for rs in &job.scenes {
            for g in rs.geo.iter() {
                if g.index_wrap != 0 || g.stride == 0 {
                    continue;
                }
                let idx = snaps.get_or_read_indices(&pctx, g.index_addr, g.index_len, g.index_elem);
                if idx.vertex_count == 0 {
                    continue;
                }
                if g.single_stream {
                    let addr =
                        g.bound_streams[0].wrapping_add(idx.first_vertex.wrapping_mul(g.stride));
                    let len = idx.vertex_count as usize * g.stride as usize;
                    if snaps.vertices_cached(&pctx, addr, len) {
                        skipped += 1;
                    } else if let Some(o) = pctx.offset(addr) {
                        rows.push((o, len));
                    }
                    continue;
                }
                // The multi-stream paths: the memo key is built exactly as
                // `read_draw_vertices` builds it, so the probe answers for the read.
                let (first_row, last_row) = match idx.gather.as_ref() {
                    Some(gather) => (gather[0], gather[gather.len() - 1]),
                    None => (idx.first_vertex, idx.first_vertex + idx.vertex_count - 1),
                };
                TextureSnapshots::gather_spans_into(
                    &mut key.spans,
                    &g.streams,
                    &g.bound_streams,
                    &g.base,
                    (first_row, last_row),
                );
                key.stride = g.stride;
                key.layout = g.layout.key;
                key.gather = idx
                    .gather
                    .as_ref()
                    .map_or((0, 0), |ga| (ga.as_ptr() as usize, ga.len()));
                if snaps.gather_memo_cached(&pctx, &key) {
                    skipped += 1;
                    continue;
                }
                for (si, st) in g.streams.iter().enumerate() {
                    let Some(&b) = g.bound_streams.get(si) else {
                        continue;
                    };
                    if b == 0 || st.stride == 0 || st.per_instance {
                        continue;
                    }
                    match idx.gather.as_ref() {
                        // A compacted draw reads its rows in RUNS - the same runs
                        // `gather_rows_into_scratch` reads, so each run is one table entry
                        // and the pool between runs is not fetched.
                        Some(gather) => {
                            let step = st.stride as usize;
                            let mut k = 0usize;
                            while k < gather.len() {
                                let mut end = k;
                                while end + 1 < gather.len()
                                    && (gather[end + 1] - gather[end]) as usize * step
                                        <= GATHER_RUN_GAP_BYTES
                                {
                                    end += 1;
                                }
                                if let Some(o) =
                                    pctx.offset(b.wrapping_add(gather[k].wrapping_mul(st.stride)))
                                {
                                    rows.push((
                                        o,
                                        (gather[end] - gather[k]) as usize * step + step,
                                    ));
                                }
                                k = end + 1;
                            }
                        }
                        None => {
                            if let Some(o) = pctx
                                .offset(b.wrapping_add(idx.first_vertex.wrapping_mul(st.stride)))
                            {
                                rows.push((o, idx.vertex_count as usize * st.stride as usize));
                            }
                        }
                    }
                }
            }
        }
        fetched += rows.len();
        overlay.extend(&rows);
    }
    crate::perf::note_prefetch(fetched, skipped);
    let mut regs = [0u32; REG_COUNT];
    let mut vfp = [0u32; VFP_ARG_COUNT];
    let pctx = GuestCtx::new(&mut regs, &mut vfp, &mut overlay, ctx.base);
    let ctx = &pctx;
    // `VITASLOP_DUMP_DRAW_GXP=<frame>` also names the frame's GEOMETRY VOLUME per draw,
    // here where the bytes are actually read: on one title the guest CPU read and
    // interleaved 33-97 MB of vertices a frame and nothing said which draws they were.
    let frame = job.frame;
    let dump_volume = VitaState::dump_draw_frame_matches(frame);
    for rs in job.scenes.iter_mut() {
        let mut out = Vec::with_capacity(rs.geo.len());
        for (i, ((mw, fw), g)) in rs.windows.iter_mut().zip(rs.geo.iter()).enumerate() {
            // An instanced draw is expanded here, one row per (instance, vertex), and then
            // needs nothing below but its uniform windows. See `read_instanced_draw`.
            if let Some((idx, vertices)) = snaps.read_instanced_draw(
                ctx,
                g.index_addr,
                g.index_len,
                g.index_elem,
                g.index_wrap,
                &g.streams,
                &g.bound_streams,
                &g.layout,
                g.stride,
            ) {
                out.push((idx.bytes, vertices));
                let _w = crate::perf::scope(crate::perf::Phase::ResolveWindows);
                for (addr, bytes) in mw.iter_mut().chain(fw.iter_mut()) {
                    if let Some(b) = snaps.reread_covered_window(ctx, bytes) {
                        *bytes = b;
                        continue;
                    }
                    // ONLY WHERE THE DRAW-TIME BYTES SAY NOTHING WAS WRITTEN YET - the same
                    // rule, and the same reason, as the non-instanced site below.
                    if snaps.is_window_placeholder(bytes) || window_wants_reread(bytes) {
                        *bytes = snaps.get_or_read_window(ctx, *addr, bytes.len());
                    }
                }
                continue;
            }
            let idx = snaps.get_or_read_indices(ctx, g.index_addr, g.index_len, g.index_elem);
            if dump_volume {
                let bytes = idx.vertex_count as usize * g.stride as usize;
                if bytes >= 64 * 1024 {
                    let streams: Vec<String> = g
                        .streams
                        .iter()
                        .enumerate()
                        .map(|(k, st)| {
                            format!(
                                "{:#x}/{}",
                                g.bound_streams.get(k).copied().unwrap_or(0),
                                st.stride
                            )
                        })
                        .collect();
                    tracing::info!(target: "vitaslop::status", "GEOM frame={frame} seq={i} verts={} first={} compacted={} stride={} bytes={bytes} idx={:#x}/{} single={} streams=[{}]", idx.vertex_count, idx.first_vertex, idx.gather.is_some(), g.stride, g.index_addr, g.index_len, g.single_stream, streams.join(" "));
                }
            }
            let vertices = crate::perf::time(crate::perf::Phase::DrawVertices, || {
                snaps.read_draw_vertices(
                    ctx,
                    &idx,
                    g.single_stream,
                    &g.streams,
                    &g.bound_streams,
                    &g.base,
                    &g.layout,
                    g.stride,
                )
            });
            out.push((idx.bytes.clone(), vertices));
            // >>> A UNIFORM BUFFER IS FILLED AFTER THE DRAW TOO, by the same contract that
            // defers the vertex bytes: the GPU reads it at submission, so a title may compute it
            // between `sceGxmDraw` and `sceGxmEndScene`. MEASURED on a baseball title's crowd:
            // every skinned crowd member draws with a 3,776-byte skinning window that reads as
            // ALL ZEROES at the draw call - a job thread fills the matrices afterwards (one frame
            // later the same address held rotations and a 1.7 m translation). Zero matrices
            // collapse every vertex to the origin, so 39 of the 41 crowd-atlas draws rasterised
            // nothing, the stands were empty, and the atlas the crowd sprites sample was never
            // painted. The old note that "uniform banks do not change" was measured on ONE
            // title; this one does. The re-read is the dirty-stamped window snapshot, so a
            // window the guest did not touch since the draw costs a stamp compare and hands back
            // the same `Arc`. The SA banks stay draw-time reads: a default uniform buffer is a
            // per-draw reservation the guest writes before the draw that names it.
            //
            // >>> BUT ONLY WHERE THE DRAW-TIME SNAPSHOT SAYS NOTHING HAD BEEN WRITTEN YET.
            //
            // "Filled after the draw" and "rewritten between draws" are the two halves of one
            // fact - that a uniform buffer's contents are not pinned to the draw that names it -
            // and they want OPPOSITE bytes. Re-reading unconditionally serves the first and
            // breaks the second: a scene that names the same buffer from many draws and rewrites
            // it in between gets the LAST version on every one of them.
            //
            // MEASURED on the golf title's tree-impostor bake, which is that second case at
            // scale: 202 offscreen scenes in one frame through one 136-byte window, every draw
            // handed an ambient of `(1.0, 0.0, 1.0)`. Green exactly zero on every impostor -
            // the grass and reed billboards baked SOLID MAGENTA from an olive-green texture,
            // and the leafier ones washed magenta under their diffuse term. Taking the
            // draw-time bytes instead, the same atlas reads green-over-red on 88% of its
            // texels, which is what its source says.
            //
            // The test is the one the writeback already uses for "nothing has written here":
            // ONE REPEATED 4-BYTE WORD over the whole window, which is what a zero fill and an
            // allocator poison fill both are, and which no computed uniform block is. The
            // crowd's skinning window is 3,776 bytes of zeroes at its draw and so still takes
            // the re-read; golf's ambient block holds real floats and so keeps its own.
            let _w = crate::perf::scope(crate::perf::Phase::ResolveWindows);
            for (addr, bytes) in mw.iter_mut().chain(fw.iter_mut()) {
                // A covered window re-reads only its blank uncovered bytes - see
                // `reread_covered_window`.
                if let Some(b) = snaps.reread_covered_window(ctx, bytes) {
                    *bytes = b;
                    continue;
                }
                // A placeholder is blank by construction - no scan to rediscover it.
                if snaps.is_window_placeholder(bytes) || window_wants_reread(bytes) {
                    *bytes = snaps.get_or_read_window(ctx, *addr, bytes.len());
                }
            }
        }
        rs.out = out;
    }
}

/// Everything [`VitaState::resolve_deferred_geometry`] needs to read one draw's vertices and
/// indices - the sticky GXM state a draw sees, snapshotted at the DRAW so a later draw's
/// bindings cannot be read in its place.
///
/// Only the DESCRIPTION is taken at the draw call; the bytes are read at `sceGxmEndScene`.
/// See [`defer_geometry`] for why.
pub(super) struct DeferredGeometry {
    pub(super) index_addr: u32,
    pub(super) index_len: usize,
    pub(super) index_elem: usize,
    /// `sceGxmDrawInstanced`'s wrap: the instance index steps every this many indices. 0
    /// for a plain draw. See `TextureSnapshots::read_instanced_draw`.
    pub(super) index_wrap: u32,
    pub(super) streams: std::sync::Arc<[VertexStreamInfo]>,
    pub(super) bound_streams: [u32; crate::vita::gxmctx::MAX_VERTEX_STREAMS],
    pub(super) base: std::sync::Arc<[u32]>,
    /// How this draw's streams scatter into the packed row - see [`RowLayout`]. Carried with
    /// the rest of the layout because the read happens at the guest's GPU wait, by which time
    /// the bound vertex program may be another one entirely.
    pub(super) layout: RowLayout,
    pub(super) stride: u32,
    pub(super) single_stream: bool,
    /// The draw's texture lists, when they are proven by the resolver rather than at the
    /// draw - see [`DeferredTextures`].
    pub(super) tex: Option<Box<DeferredTextures>>,
}

/// >>> A DRAW'S TEXTURE BINDINGS, PROVEN WHERE ITS GEOMETRY IS READ.
///
/// Under the resolver worker with `VITASLOP_SMP_DEFER_TEXTURES` (see [`defer_textures`]),
/// `sceGxmDraw` no longer decodes, folds and re-proves the draw's texture set: that was about
/// half of what the draw cost the guest's render thread (`draw: decode texture bindings` 0.95
/// ms/f of ~2.0 on the desktop, x4 on the phone) and it sits on the title's serial main ->
/// render chain. The draw keeps only what the set is a function of - the sampler block's bytes
/// (or "the same bytes as the previous draw of this scene"), the fragment program, its declared
/// units and albedo unit, and the vertex-stage bindings - and [`compute_resolve_job`] proves the
/// set through the same cache, scene by scene, when it reads the geometry. A set the cache
/// cannot answer is built at apply time under the host lock ([`VitaState::apply_resolve_job`]),
/// because building one needs the host's recorded texture formats.
///
/// The pixels are therefore read at the flip rather than at the draw - the contract the
/// geometry already relies on: the GPU reads a texture when it executes the scene, and every
/// sync point that could make a rewrite safe resolves everything first.
pub(super) struct DeferredTextures {
    /// The fragment sampler block at the draw, or `None` for "byte-identical to the previous
    /// draw of the same scene, same fragment program".
    pub(super) span: Option<Box<[u8]>>,
    pub(super) fheader: u32,
    /// The program's declared sampler units ([`ProgramReflection::sampler_units`]).
    pub(super) want: u32,
    pub(super) albedo: Option<u32>,
    /// The vertex-stage bindings and the generation they were bound at, when any are bound.
    pub(super) vert: Option<(u64, Arc<[TextureBinding]>)>,
}

/// What [`compute_resolve_job`] made of a [`DeferredTextures`]: the proven list, or the
/// bindings and set key the apply step builds it from.
#[derive(Clone)]
pub(super) enum TexSet {
    Hit(Arc<[crate::capture::BoundTexture]>),
    Miss(u64, Arc<[TextureBinding]>),
}

/// One draw's resolved textures - see [`DeferredTextures`].
pub(super) struct ResolvedTextures {
    pub(super) frag: TexSet,
    pub(super) vert: Option<TexSet>,
    pub(super) fheader: u32,
    pub(super) albedo: Option<u32>,
}

/// The set-cache key of a binding list: `first` (the fragment program header, or
/// [`VERTEX_STAGE_TAG`]) then each binding's unit, descriptor address and control words.
/// The same fold `draw_textures_now` makes, so both forms share one cache.
pub(super) fn fold_set_key(first: u64, binds: &[TextureBinding]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(first);
    for b in binds {
        mix(b.unit as u64);
        mix(b.addr as u64);
        for w in b.words {
            mix(w as u64);
        }
    }
    h
}

/// Prove each draw's texture set in the resolver - see [`DeferredTextures`]. ON whenever a
/// resolver is attached (`VITASLOP_SMP_DEFER_TEXTURES=0` is the arm back), and only on the
/// recompiled path (the fixed-function material reads the list at the draw). Proxy: render
/// thread 17.6 -> 16.0 ms/f (px27c -> px27k), picture correct.
pub(super) fn defer_textures() -> bool {
    static KNOB: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KNOB.get_or_init(|| {
        crate::knobs::var("VITASLOP_SMP_DEFER_TEXTURES")
            .as_deref()
            .map(str::trim)
            != Ok("0")
    }) && resolver_attached()
        && gxp_live_capture()
        && !fixed_function_wanted()
}

/// Whether a draw's VERTEX AND INDEX BYTES are read at `sceGxmEndScene` rather than at the
/// `sceGxmDraw` that named them. `VITASLOP_DEFER_GEOMETRY=0` restores the draw-time read.
///
/// # GXM DEFERS, AND A TITLE THAT RESERVES, DRAWS, THEN FILLS IS LEGAL
/// `sceGxmDraw` records a draw into the command buffer; the GPU reads the vertex buffer when
/// the scene is SUBMITTED. So the moment a draw's geometry is defined is the end of its scene,
/// not the draw call - and a title may legally write the buffer after calling draw.
///
/// **MEASURED on a fighting title's particle pass:** of 237 vertex windows watched across one
/// frame, exactly ONE changed between its `sceGxmDraw` and `sceGxmEndScene` - the SubUV
/// particle stream. Read at the draw call it held the previous user of that rotating arena, a
/// 48-byte 2D text quad batch, which this engine then read through the particle program's
/// 52-byte stride: every vertex four bytes further into someone else's records. That is the
/// long-standing screen-crossing particle STREAK, and it is why the buffer's period disagreed
/// with the stride the guest declared for it (the declared 52 was right all along - the bytes
/// were not this draw's).
///
/// Reading at end of scene is not extra work: it is the SAME read, once, at the moment the
/// hardware would do it. The description a draw's read needs (which streams, bound where, with
/// what strides) is still taken at the draw, because that state is sticky and the next draw
/// may change it.
/// `VITASLOP_INPLACE_COMPARE=0`: skip the in-place compare of a dirty snapshot range (see
/// `get_or_read_vertices`) - the negative control for its exactness.
pub(super) fn inplace_compare() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_INPLACE_COMPARE").as_deref() != Ok("0"))
}

pub(super) static ASYNC_FLIP_RESOLVE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Leave a flip's geometry resolve to the presenting worker - see
/// [`VitaState::resolve_at_flip`]. Set only by an engine whose presenter is NOT the flipping
/// thread (the browser's parallel scheduler) and which calls
/// [`VitaEnv::resolve_flipped_geometry`] before it takes a frame.
pub fn set_async_flip_resolve(on: bool) {
    ASYNC_FLIP_RESOLVE.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn async_flip_resolve() -> bool {
    ASYNC_FLIP_RESOLVE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Which of the draw-deferral switches are on, as one line - geometry read at end of scene,
/// texture sets proven by the resolver, and the conditions the second rests on. For an engine's
/// run report: `defer_textures` is decided ONCE, at the first draw, so a resolver attached after
/// it is a resolver nobody hands texture work to.
pub fn deferral_report() -> String {
    format!(
        "deferral: geometry {}, textures {} (resolver attached {}, flip resolve async {}, gxp live {}, fixed-function wanted {})",
        defer_geometry(),
        defer_textures(),
        resolver_attached(),
        async_flip_resolve(),
        gxp_live_capture(),
        fixed_function_wanted()
    )
}

pub(super) fn defer_geometry() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Through `crate::knobs`, not `std::env`: this is a DEFAULT-BEARING arm, and the browser
    // has no environment [[vitaslop-browser-has-no-env]] - a default that cannot be turned off
    // on the engine that ships is a default nobody can A/B.
    *ON.get_or_init(|| {
        crate::knobs::var("VITASLOP_DEFER_GEOMETRY")
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}