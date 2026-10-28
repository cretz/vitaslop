//! The live GXP->WGSL recompiler's data: pipelines and what a draw needs from one, the
//! async pipeline warm-up, the per-draw prepared resources, and [`GxpLive`] itself.

use super::*;

/// A linked + compiled pipeline for one guest shader pair, cached by shader identity.
/// The destination-colour copy a self-blending fragment program reads - see
/// [`GxmRenderer::dest_color`].
pub(super) struct DestColor {
    pub(super) texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
    pub(super) width: u32,
    pub(super) height: u32,
    /// Samples per pixel - the attachment's. A multisampled copy is filled by
    /// [`GxmRenderer::copy_attachment_to_dest`]'s per-sample draw, not a texture copy.
    pub(super) samples: u32,
    /// Which minting of a destination texture this is - part of the group-3 bind-group key.
    pub(super) generation: u64,
}

/// `GxpLive::pipelines`' key: (pair, colour format, samples, cull, vertex layout, raster).
pub(super) type PipeCacheKey = (u64, wgpu::TextureFormat, u32, u32, u64, u64);

/// Pipelines built this process, and how many of them had a key built before - see `log_pipe`.
pub static PIPE_BUILDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PIPE_REBUILDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(built, of them an identical key built before)` for the whole process.
pub fn pipe_rebuilds() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (PIPE_BUILDS.load(Relaxed), PIPE_REBUILDS.load(Relaxed))
}

/// `VITASLOP_PIPE_LOG=1`: one status line per pipeline built - the frame, the pair, and the
/// baked state it was built under (format, samples, cull, vertex layout, raster = depth,
/// stencil, blend). The question it answers: how many distinct STATES a title draws each pair
/// with, which decides whether a pipeline can be built at the patcher call (a loading
/// screen) - on the phone a state variant is a whole compile (30b, ~100 ms each).
pub(super) fn log_pipe(how: &str, k: &PipeCacheKey, g: &GxpRecompile) {
    use std::sync::OnceLock;
    // ALWAYS: count the build, and whether this exact key was built before anywhere in the
    // process - a rebuild of an identical pipeline is a whole compile paid twice (a fighting title, native:
    // 32 of 65 pairs built twice), and nothing else can say whether the browser does it too.
    {
        use std::hash::{Hash, Hasher};
        static SEEN: std::sync::Mutex<Option<HashSet<u64>>> = std::sync::Mutex::new(None);
        let mut h = std::collections::hash_map::DefaultHasher::new();
        k.hash(&mut h);
        let fresh = SEEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashSet::default).insert(h.finish());
        PIPE_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !fresh {
            PIPE_REBUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| crate::knobs::flag("VITASLOP_PIPE_LOG")) {
        return;
    }
    report_status!(
        "pipe {how} pair {:016x} fmt {:?} samples {} cull {} layout {:016x} raster {:016x} | depth func {} write {} stencil {:x?} blend {} {:?} two_sided {}",
        k.0,
        k.1,
        k.2,
        k.3,
        k.4,
        k.5,
        g.depth_func,
        g.depth_write,
        &g.stencil[..4],
        g.blend,
        g.blend_state,
        g.two_sided
    );
}

/// What [`GxpLive::pipe_key`] computes for one draw.
pub(super) struct PipeKey {
    pub(super) raw_units: u64,
    pub(super) f32_units: u64,
    pub(super) raster: u64,
    pub(super) cache_key: PipeCacheKey,
}

/// A pipeline `createRenderPipelineAsync` is compiling - see `GxmRenderer::warm_pipelines`.
pub type PipeFuture = std::pin::Pin<Box<dyn std::future::Future<Output = Result<wgpu::RenderPipeline, String>>>>;

/// Pipelines [`GxmRenderer::warm_pipelines`] STARTED. The browser compiles them off its GPU
/// main thread; [`Self::wait`] resolves them and [`GxmRenderer::install_pipelines`] puts
/// them where the encode looks.
#[derive(Default)]
pub struct PipelineWarm {
    pub(super) items: Vec<(PipeCacheKey, GxpPipelineOf<PipeFuture>)>,
}

impl PipelineWarm {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Replace each compile's future with `f(future)` - how the browser front end times when
    /// a compile really SETTLES, which a poll from the present loop cannot see (it only
    /// learns of a settled promise the next time it polls).
    pub fn map_futures(&mut self, mut f: impl FnMut(u64, PipeFuture) -> PipeFuture) {
        for (k, p) in self.items.iter_mut() {
            let fut = std::mem::replace(&mut p.pipeline, Box::pin(std::future::pending()));
            p.pipeline = f(k.0, fut);
        }
    }

    /// Hold `other`'s compiles with these - a caller that keeps pipelines in flight ACROSS
    /// presents adds each frame's newly started ones here.
    pub fn absorb(&mut self, other: PipelineWarm) {
        self.items.extend(other.items);
    }

    /// The compiles that have ALREADY settled, taken out without waiting on the rest.
    ///
    /// Polled with a no-op waker: the promise's own callbacks record its result whether or
    /// not anything is waiting on it, so a later poll finds it. What is still compiling
    /// stays here for the next call (or [`Self::wait`]).
    pub fn take_ready(&mut self) -> PipelinesReady {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        let mut ready = PipelinesReady { items: Vec::new(), settled: Vec::new() };
        let mut i = 0;
        while i < self.items.len() {
            match std::future::Future::poll(self.items[i].1.pipeline.as_mut(), &mut cx) {
                std::task::Poll::Pending => i += 1,
                std::task::Poll::Ready(r) => {
                    let (k, p) = self.items.remove(i);
                    ready.settle(k, r, p.swap(()).1);
                }
            }
        }
        ready
    }

    /// Every started pipeline, resolved. One the browser REJECTS is left out and named: the
    /// encode then builds it synchronously, which reports the validation error through the
    /// device's error scope exactly as it always has.
    pub async fn wait(self) -> PipelinesReady {
        let mut ready = PipelinesReady { items: Vec::with_capacity(self.items.len()), settled: Vec::new() };
        for (k, p) in self.items {
            let (fut, shell) = p.swap(());
            ready.settle(k, fut.await, shell);
        }
        ready
    }
}

/// Resolved pipelines, for [`GxmRenderer::install_pipelines`].
pub struct PipelinesReady {
    pub(super) items: Vec<(PipeCacheKey, GxpPipeline)>,
    /// Every key whose compile SETTLED, including the rejected ones `items` leaves out - so
    /// the renderer stops counting them as in flight and the encode may build them.
    pub(super) settled: Vec<PipeCacheKey>,
}

impl PipelinesReady {
    pub(super) fn settle(&mut self, k: PipeCacheKey, r: Result<wgpu::RenderPipeline, String>, shell: GxpPipelineOf<()>) {
        self.settled.push(k);
        match r {
            Ok(pipe) => self.items.push((k, shell.swap(pipe).1)),
            Err(e) => report_status!(
                "gxp pair {:016x}: createRenderPipelineAsync REJECTED ({e}) - the frame builds it synchronously instead",
                k.0
            ),
        }
    }
}

impl<P> GxpPipelineOf<P> {
    /// This pair with its pipeline replaced by `q`, and the pipeline it had.
    pub(super) fn swap<Q>(self, q: Q) -> (P, GxpPipelineOf<Q>) {
        let GxpPipelineOf {
            pipeline,
            layouts,
            vsa_lanes,
            fsa_lanes,
            mem_bind_bytes,
            fmem_windows,
            fmem_bind_bytes,
            mem_windows,
            samplers,
            vertex_samplers,
            repack,
            packed_stride,
            packed_stride_native,
            passthrough,
            needs_dest,
            store_windows,
        } = self;
        (
            pipeline,
            GxpPipelineOf {
                pipeline: q,
                layouts,
                vsa_lanes,
                fsa_lanes,
                mem_bind_bytes,
                fmem_windows,
                fmem_bind_bytes,
                mem_windows,
                samplers,
                vertex_samplers,
                repack,
                packed_stride,
                packed_stride_native,
                passthrough,
                needs_dest,
                store_windows,
            },
        )
    }
}

/// A built pair with its pipeline in hand - see [`GxpPipelineOf`].
pub(super) type GxpPipeline = GxpPipelineOf<wgpu::RenderPipeline>;

/// Generic over the PIPELINE ONLY so one build path serves both creations: `P` is a
/// `wgpu::RenderPipeline` for a sync build and the pending future for an async one (see
/// `GxmRenderer::warm_pipelines`), which becomes the former through [`Self::with_pipeline`].
pub(super) struct GxpPipelineOf<P> {
    /// The one pipeline for this pair. There used to be two - an "opaque" variant with
    /// LessEqual + depth write and an "overlay" variant with Always + no write - selected
    /// per draw by a HEURISTIC (see the note in `build_gxp_pipeline`). Depth now comes from
    /// the guest's own captured `SceGxmDepthFunc`/depth-write, which is part of the pair
    /// key, so the two variants would be identical by construction.
    pub(super) pipeline: P,
    /// group0 = vertex SA uniform, group1 = fragment SA uniform, group2 = samplers.
    /// Empty layouts where the stage declares nothing, so the pipeline layout still
    /// covers every group index the WGSL might reference.
    pub(super) layouts: [wgpu::BindGroupLayout; 4],
    /// Vertex SA scalar-lane count (the group0 uniform holds `ceil(n/4)` vec4s).
    pub(super) vsa_lanes: u32,
    /// Fragment SA scalar-lane count.
    pub(super) fsa_lanes: u32,
    /// Size in bytes of the vertex stage's guest-MEMORY-WINDOW uniform at group 0
    /// binding 1 (one header vec4 per window + every window's bytes - see
    /// `vitaslop_gxp_shader::module::mem_window_vec4_count`), or 0 when the pair's vertex
    /// program loads no memory. A draw for a pipeline with windows must carry their bytes
    /// or be DROPPED with a report.
    pub(super) mem_bind_bytes: u32,
    /// The FRAGMENT stage's windows and its binding's total byte size, the mirror of the
    /// two fields above at `@group(1) @binding(1)`.
    pub(super) fmem_windows: Vec<vitaslop_gxp_shader::MemWindow>,
    pub(super) fmem_bind_bytes: u32,
    /// The windows themselves, in binding order, so a draw's bytes can be laid out at the
    /// offsets the shader was emitted against.
    pub(super) mem_windows: Vec<vitaslop_gxp_shader::MemWindow>,
    /// `(sampler unit, is_3d)` per group2 sampler, in binding order.
    pub(super) samplers: Vec<(u8, SamplerDim)>,
    /// The VERTEX stage's sampler units, in group-4 binding order.
    pub(super) vertex_samplers: Vec<(u8, SamplerDim)>,
    /// How to repack the guest vertex stream into the tightly-packed `Float32xN` buffer
    /// this pipeline's vertex layout expects (one entry per attribute), + the packed
    /// stride. Repacking to f32 on the CPU sidesteps wgpu's vertex-format gaps (no
    /// `Float16x3`, etc.) and matches the recompiled shader, which reads f32 anyway.
    pub(super) repack: Vec<RepackAttr>,
    pub(super) packed_stride: u32,
    /// What `packed_stride` would have been had every attribute kept the guest's own
    /// component format. Equal to `packed_stride` when none had to widen; the difference,
    /// times the vertex count, is what the f32 expansion costs this pipeline per upload.
    pub(super) packed_stride_native: u32,
    /// This pipeline's vertex layout addresses the GUEST's interleaved row DIRECTLY - its
    /// stride, its byte offsets, its own component formats - so `repack` is the IDENTITY and
    /// the draw binds `gxp.vertices` unchanged, with no hash, no repack and no packed-cache
    /// entry. See `passthrough_vertex_format` for when a layout qualifies.
    ///
    /// `repack`/`packed_stride` are still built and still correct; they are simply not used
    /// while this is set, which is what keeps `VITASLOP_GXP_VERTEX_PASSTHROUGH=0` a real arm.
    pub(super) passthrough: bool,
    /// Whether this pair's FRAGMENT program reads the DESTINATION colour - it blends for
    /// itself in ordinary ALU rather than through a blend equation the driver patched in.
    /// Such a pipeline declares a destination texture at `@group(3) @binding(1)`, and every
    /// draw using it needs the colour attachment copied into that texture IMMEDIATELY
    /// before it, which costs a render-pass split. See
    /// `vitaslop_gxp_shader::fragment_reads_dest_color`.
    pub(super) needs_dest: bool,
    /// The buffers this pair's FRAGMENT program STORES through, in its store binding's order
    /// (`vitaslop_gxp_shader::module::resolve_store_windows`). Non-empty means the pipeline
    /// declares the store mirror at `@group(3) @binding(2)` and its window headers at binding
    /// 3, and every draw must carry the windows' guest snapshots (`frag_store_windows`) or be
    /// dropped. Empty for every other pair.
    pub(super) store_windows: Vec<vitaslop_gxp_shader::MemWindow>,
}

/// One attribute's plan for repacking the guest vertex stream to packed f32.
pub(super) struct RepackAttr {
    pub(super) guest_offset: u32,
    pub(super) gxm_format: u8,
    /// Components the GUEST's stream supplies.
    pub(super) components: u8,
    /// Components the packed slot carries - the SHADER's declared width. Anything above
    /// `components` is the fill (see `attr_fill`).
    pub(super) slots: u8,
    /// What each lane above `components` is fed, per lane - decided by the LINKER from what
    /// the lane feeds (`vitaslop_gxp_shader::attrflow`), because two titles need opposite
    /// constants and nothing about the binding separates them. See `attr_fill`.
    pub(super) fill: [f32; 4],
    pub(super) packed_offset: u32,
    /// Bytes to COPY VERBATIM from the guest row into the packed slot, or 0 to expand every
    /// component to `f32`.
    ///
    /// Non-zero means the pipeline binds this slot with the GUEST's own component format
    /// and the vertex fetch does the conversion, so the repack has nothing to convert. The
    /// count is what the guest BINDS (`components * component size`); anything the format
    /// fetches beyond that stays as the packed row's zero fill, and the fill lanes above it
    /// are the hardware's - both of which the plan only chooses when nothing reads them.
    pub(super) copy_bytes: u32,
}

/// Per-draw GPU resources for one recompiled draw, kept alive through the render pass.
pub(super) struct GxpPrepared {
    /// Cache key of the pipeline this draw uses (looked up immutably during the pass).
    pub(super) key: u64,
    /// Byte range of this draw's repacked vertices and its indices inside the pass's
    /// grow-only GXP arenas.
    ///
    /// NOT per-draw buffers. Creating a vertex and an index buffer per draw is two GPU
    /// allocations per draw, and a race frame here submits ~470 recompiled draws: measured,
    /// that per-draw allocation was 14.6 ms of the 15.2 ms `encode`, i.e. the single
    /// largest item in the whole system. The fixed-function path has always packed into
    /// shared arenas uploaded once; this is the same thing for the recompiled path.
    pub(super) v_off: u64,
    pub(super) v_len: u64,
    pub(super) i_off: u64,
    pub(super) i_len: u64,
    /// Whether those ranges address the RESIDENT heap ([`GxpLive::resident`]) rather than
    /// this pass's arena. The two are independent: a mesh whose vertices never change can
    /// still be re-indexed every frame, and a title that rebuilds a vertex stream can hold
    /// its index list still.
    pub(super) v_resident: bool,
    pub(super) i_resident: bool,
    pub(super) index_count: u32,
    /// The format of the bytes at `i_off`. A 16-bit guest index buffer is no longer widened
    /// (see the expansion in the runtime's scene builder), and the format is per DRAW at
    /// `set_index_buffer`, so one frame may carry both.
    pub(super) index_u32: bool,
    /// Byte offsets of this draw's vertex and fragment SA blocks inside the pass's uniform
    /// arena, for the group0/group1 DYNAMIC offsets, plus the vertex stage's
    /// guest-memory window block (third slot, meaningful only when the pipeline's
    /// `mem_bind_bytes` is non-zero). The bind groups themselves belong to the shader
    /// PAIR, not the draw ([`GxpLive::ubo_bgs`]).
    pub(super) u_off: [u32; 4],
    /// Bind groups for group2 (samplers) and group3 (the pass depth block, plus the
    /// destination-colour texture when this pair reads it).
    pub(super) bg2: wgpu::BindGroup,
    pub(super) bg3: wgpu::BindGroup,
    /// This draw's pair reads the DESTINATION colour, so the encoder owes it a copy of the
    /// attachment taken immediately before it - see `GxpPipeline::needs_dest`.
    pub(super) needs_dest: bool,
    /// True = alpha-blended (2D/overlay), false = opaque geometry.
    pub(super) blend: bool,
    /// The attachment format this draw's pipeline was built for - part of the pipeline
    /// cache key, because a gamma-correct surface renders through an sRGB view.
    pub(super) format: wgpu::TextureFormat,
    /// Samples per pixel of the attachments this draw's pipeline was built for, and part
    /// of the cache key for the same reason as `format`: a pipeline may only be bound in a
    /// pass whose attachments match its sample count. The SAME shader pair is used by this
    /// title on both a 4-sample world target and a 1-sample display pass, so this is not a
    /// theoretical distinction - without it the second pass binds the first one's pipeline
    /// and the driver rejects the pass.
    pub(super) samples: u32,
    /// The guest's `SceGxmCullMode` for this draw, and part of the cache key for the same
    /// reason: it is baked into the pipeline's primitive state and a title sets it PER
    /// DRAW - one retail racer asks for `CW` on its world and `NONE` on its overlays.
    pub(super) cull: u32,
    /// The guest VERTEX LAYOUT this draw's pipeline was built for, and part of the cache
    /// key for a reason the other three do not share: a pipeline owns the REPACK PLAN, and
    /// that plan reads guest byte offsets. See `GxpLive::pipelines`.
    pub(super) layout: u64,
    /// The per-draw polygon offset this draw's pipeline was built for. See
    /// `GxpLive::raster_key`.
    pub(super) raster: u64,
    /// This draw's fragment colour is provably the DESTINATION UNCHANGED, so its colour
    /// work cannot change a pixel - see `draw_colour_is_noop`. What the encoder does with
    /// that depends on the draw's OTHER effect: with no depth write there is nothing left to
    /// do and the draw is dropped whole; with one, the depth write is still owed and the
    /// draw is issued through a DEPTH-ONLY pipeline (no fragment stage at all).
    pub(super) colour_noop: bool,
    /// Whether this draw writes DEPTH, which is the half of it an elision must keep.
    pub(super) depth_write: bool,
    /// Whether this draw can write STENCIL - an op other than KEEP under a non-zero write
    /// mask. An elision must keep that too: a HUD's stencil-marking quads write no colour
    /// and no depth, and they are the whole point of the draw.
    pub(super) stencil_write: bool,
    /// The stencil REFERENCE this draw compares and replaces with (front face), set on the
    /// pass per draw - it is dynamic state, not part of the pipeline.
    pub(super) stencil_ref: u32,
    /// This draw's FRAGMENT memory windows as `(guest address, bytes, arena byte offset)`,
    /// recorded only when the frame has rendered a raw 64-bit target the windows may lie on -
    /// see `GxmRenderer::fill_windows_from_raw_targets`. Empty (no allocation) for every other
    /// draw.
    pub(super) fmem_ranges: Vec<(u32, u32, u64)>,
    /// The byte size of this draw's FRAGMENT window block when its windows are sized PER DRAW
    /// (`push_mem_windows_per_draw`), which its group-1 bind group must name; 0 for every other
    /// draw, whose block is the pipeline's fixed `fmem_bind_bytes`.
    pub(super) fmem_bytes: u32,
    /// The clip-`w` sign correction this draw's module applies (`gxp_depth.range.z`: 1, -1 or
    /// 2), which [`super::clip::draw_pixel_bounds`] has to apply to the same positions.
    pub(super) clip_sign: f32,
}

/// One entry in the submission-order draw plan: either a fixed-function [`Item`] (by index
/// into the arena-packed `items`) or a recompiled draw (by index into `gxp_prepared`). The
/// two kinds interleave in the one render pass so depth and overlay ordering stay correct.
pub(super) enum Enc {
    Fixed(usize),
    Gxp(usize),
}

/// The live recompiler's pipeline cache + config. Held by [`GxmRenderer`].
pub(super) struct GxpLive {
    /// Master switch (`VITASLOP_GXP_LIVE`).
    pub(super) enabled: bool,
    /// Render ONLY recompiled draws, skipping the fixed-function draw for any call that
    /// has a working recompiled pipeline (`VITASLOP_GXP_ONLY`) - isolates the recompiler
    /// output for review.
    pub(super) only: bool,
    /// Apply the GXM (GL-style, NDC z in [-1,1]) -> WebGPU (z in [0,1]) clip-depth remap
    /// in the vertex output (`VITASLOP_GXP_ZFIX`, default on). Off passes clip z straight.
    pub(super) zfix: ZFix,
    /// Flip clip Y (`VITASLOP_GXP_YFLIP`, default off). The fixed-function MVP path passes
    /// guest clip X/Y straight to WebGPU and renders upright, so the real shaders should
    /// too; the toggle is here for empirical confirmation.
    pub(super) yflip: bool,
    /// Diagnostic (`VITASLOP_GXP_FORCE`): bind a neutral fallback texture for a sampler
    /// unit whose real texture we could not capture/decode (e.g. a 3D/LUT format) or a 3D
    /// sampler, so the recompiled GEOMETRY renders and the vertex/clip/depth path can be
    /// validated even before the special textures are handled. Off = strict (fall back to
    /// fixed-function rather than sample a wrong texel).
    pub(super) force: bool,
    /// Diagnostic (`VITASLOP_GXP_SOLID`): every recompiled draw outputs solid magenta with
    /// the depth test disabled, to answer "does the geometry rasterize on-screen at all?"
    /// independent of fragment shading and depth. Magenta visible -> vertex/clip is right.
    pub(super) solid: bool,
    /// Diagnostic (`VITASLOP_GXP_NODEPTH`): every recompiled draw keeps its real shading and
    /// blending but stops testing depth. `solid` answers "does this geometry rasterize"
    /// while changing BOTH the shading and the depth test, so a surface that comes out
    /// black under it is still ambiguous - the fragment could be painting black, or it
    /// could be losing the depth test. This changes only the depth test, which separates
    /// the two: a surface that appears here and not in an ordinary run is depth-rejected,
    /// one that stays black is shaded black.
    pub(super) nodepth: bool,
    /// Diagnostic (`VITASLOP_GXP_NOBLEND`): force every recompiled pipeline to REPLACE with
    /// a full colour write mask, changing NOTHING else. The counterpart of `nodepth` for the
    /// other way a correctly-shaded draw can leave no mark: a shader that writes alpha 0
    /// under a src-alpha blend, or a guest colour mask that writes no channels.
    ///
    /// On a pair whose ALU blend was LOWERED to pipeline state
    /// (`vitaslop_gxp_shader::module::lower_dest_blend`) this shows the SOURCE TERM ALONE,
    /// because that is all such a shader emits - which is exactly what the diagnostic is
    /// for, but it is not "the draw without its blend", it is "the draw's source term".
    pub(super) noblend: bool,
    /// Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys
    /// (the `gxp draw key` value `VITASLOP_GXP_DUMP` prints), letting every other draw fall
    /// back. Rendering one pair at a time is how a visual artifact is attributed to the
    /// shader that produced it. Empty = no filter (recompile every linkable pair).
    pub(super) keys: Vec<u64>,
    /// Pairs forced down the fixed-function path (`VITASLOP_GXP_EXCLUDE`).
    pub(super) exclude: Vec<u64>,
    /// `(key, target format) -> Some(pipeline)` for a linkable pair, `-> None` for one that
    /// failed to link (cached so we never retry it and always fall back).
    ///
    /// The FORMAT is part of the key because a render pipeline is bound to the format of
    /// the attachment it writes, and a surface the guest put in gamma-correct mode is
    /// rendered through an sRGB view of the same texture. Only a pair that is actually
    /// drawn onto such a surface ever gets a second entry, so a title using no gamma
    /// surfaces builds exactly as many pipelines as before.
    ///
    /// >>> AND THE GUEST'S VERTEX LAYOUT IS IN THE KEY, because a pipeline OWNS a repack
    /// > > > plan and that plan is built from the layout of whichever draw happened to be first.
    ///
    /// `build_gxp_pipeline` turns each linked attribute into a `RepackAttr` carrying the
    /// GUEST byte offset, format and component count it reads from - per DRAW state, not a
    /// property of the shader pair. A title is free to submit one pair with two different
    /// vertex layouts, and this title does: the golf HUD's pair `61be04e22bf29693` is
    /// submitted both as a 96-byte vertex with `IN.color` at byte 24 and as a 92-byte
    /// vertex with `IN.color` at byte 88. With the layout out of the key, whichever arrived
    /// first built the plan and every draw of the OTHER layout read its attributes from the
    /// wrong bytes - which is not a dropped draw, it is a draw that renders confidently
    /// out of the wrong memory.
    ///
    /// MEASURED: with the earlier frames rendered, the golf HUD's varying `v0` - the vertex
    /// colour, which the guest sets to `(1,1,1,1)` on every submission - arrived at the
    /// fragment as `(0, 0, 0.57)`, so the whole HUD rendered flat blue while its geometry,
    /// its atlas, its alpha and its uniforms were all measured correct. It reproduced only
    /// when a menu draw of the other layout came first, which is why a shot window that
    /// starts after the menus never showed it.
    pub(super) pipelines: HashMap<PipeCacheKey, Option<GxpPipeline>>,
    /// The caller warms pipelines ahead of the frame (`GxmRenderer::warm_pipelines` has run),
    /// so a pipeline the encode still has to build synchronously is a pre-pass MISS and is
    /// counted as `pipelines_unwarmed`. False on every path that never warms (native).
    pub(super) warm_used: bool,
    /// Pipelines `createRenderPipelineAsync` is still compiling, which the caller holds (a
    /// [`PipelineWarm`] it may keep across presents - see `LivePlayback::warm_pipelines`).
    /// `warm_pipelines` does not start one of these again; [`GxmRenderer::install_pipelines`]
    /// clears a key when its compile SETTLES, resolved or rejected.
    pub(super) in_flight: HashSet<PipeCacheKey>,
    /// The pipeline keys the frame last passed to `warm_pipelines` draws with - what
    /// [`GxmRenderer::frame_waits_on_compile`] checks against `in_flight`.
    pub(super) frame_needs: Vec<PipeCacheKey>,
    /// Compiled WGSL modules, by shader PAIR.
    ///
    /// # A pair's module does not depend on the pipeline variant, and it used to be rebuilt
    /// # for every one of them
    /// `pipelines` is keyed by `(pair, format, samples, cull)` because a render pipeline is
    /// bound to all four. The MODULE is bound to none of them: it is a pure function of the
    /// two program blobs plus the clip/depth injections, which are read once per run. So a
    /// pair drawn onto both a 4-sample world target and a 1-sample display pass compiled the
    /// same WGSL twice, and one drawn with two cull modes compiled it twice again.
    ///
    /// MEASURED on a retail race: the run's pipeline builds cost **805 ms
    /// compiling WGSL against 271 ms creating pipelines** - three quarters of it in the
    /// half that is pair-only - across 163 pipelines built over 600 frames. In the BROWSER
    /// a WGSL compile costs far more than it does here, which is why the same race hitches
    /// there.
    ///
    /// This removes the duplicate compiles. It does NOT move the FIRST compile off the frame
    /// that needs it, which is the larger half of the problem: see the note in
    /// `build_gxp_pipeline` for why that needs the guest's shader patcher to name the pair.
    pub(super) modules: HashMap<u64, wgpu::ShaderModule>,
    /// Memoized shader-pair keys, by the IDENTITY of the two program blobs plus the state
    /// baked into the pipeline alongside them - see [`GxpLive::key`] and
    /// [`GxpLive::pair_key`].
    ///
    /// The value is the same published key `key` computes; only the recomputation is
    /// skipped. The entry keeps its own `Arc` clones of both blobs, which is what makes a
    /// POINTER safe to identify them by: while the entry lives, that allocation cannot be
    /// freed and its address cannot be handed to a different program.
    pub(super) pair_keys: HashMap<PairIdentity, (u64, std::sync::Arc<[u8]>, std::sync::Arc<[u8]>)>,
    /// Uploaded texture views, keyed by the decoded texture's content fingerprint and the
    /// view dimension it is bound as. A scene binds a handful of textures across hundreds
    /// of draws, so uploading per draw (as this path first did) re-sends the same
    /// multi-megabyte shadow map thousands of times a frame and exhausts GPU memory.
    ///
    /// >>> THE TEXTURE IS HELD BESIDE THE VIEW SO AN EVICTION CAN `destroy()` IT. Keeping
    /// > > > only the view works - the view holds the texture alive - but it leaves nothing to
    /// > > > call `destroy()` ON, and in the browser dropping a view merely makes the texture
    /// > > > collectable ([[vitaslop-browser-gpu-needs-destroy]]). This cache evicts hardly at all
    /// > > > on the desktop and CONSTANTLY on the user's device, where 83% of decodes were
    /// > > > measured to be re-decodes of something just evicted - so this is the one leak the
    /// > > > desktop is structurally unable to observe, on the engine that can least afford it.
    pub(super) views: HashMap<(u64, SamplerDim), (wgpu::Texture, wgpu::TextureView)>,
    /// Bytes currently held by `views`, against [`tex_cache_budget_bytes`]. Tracked
    /// rather than derived because a `TextureView` cannot be asked its size.
    pub(super) views_bytes: usize,
    /// The frame each cached view was last USED on, and the frame counter itself.
    ///
    /// # Why the cache needs to know what the current frame is touching
    /// The budget used to be enforced by clearing the cache WHOLESALE, on the argument
    /// that the keys are content fingerprints so a clear costs a re-upload and never
    /// correctness. That is true and it is not the problem. The problem is the shape of
    /// the failure: once a title's per-frame working set exceeds the budget, the clear
    /// fires part-way through EVERY frame, so every texture is evicted before the next
    /// frame asks for it again and the cross-frame hit rate is not degraded but ZERO.
    ///
    /// MEASURED on one title's campaign map, on the target phone: 226 distinct textures
    /// totalling 330 MB against a 256 MB budget - `0.97 cache clears` per frame, 225
    /// textures re-decoded and 76 MB re-uploaded per frame, `build 718 ms` of an 878 ms
    /// render, 1 fps. The cache was not too small by a factor of anything interesting; it
    /// was one step past a CLIFF, and a cliff is the wrong shape for a budget.
    ///
    /// So eviction is now per entry and never touches what THIS frame has already used.
    /// A working set that fits keeps hitting; one that does not degrades in proportion
    /// instead of collapsing, and the frame in flight can always complete.
    /// Per entry: `(the frame it was last used on, the bytes it holds)`. The size is here
    /// rather than derived because neither the key nor a `TextureView` carries it, and
    /// evicting one entry has to subtract exactly what that entry added.
    pub(super) views_used: HashMap<(u64, SamplerDim), (u64, usize, u32, Residency)>,
    /// >>> WHICH UPLOAD IS THE CURRENT ONE FOR A GIVEN GUEST TEXTURE: a SLOT (the guest
    /// > > > address, format, shape and sampler state - see [`view_slot_key`]) to the cache key
    /// > > > holding its latest contents.
    ///
    /// The cache key folds the source bytes, which is what makes it exact and what makes a
    /// texture the guest REWRITES arrive as a brand new entry every time. A movie is one
    /// 2 MB entry per picture, thirty a second, and every one of them stays resident until
    /// the byte budget notices - on a phone, by evicting textures the title is still using.
    /// A new entry in a slot releases the one it displaces, which is bookkeeping rather
    /// than policy: the bytes it held are gone from guest memory.
    pub(super) view_slots: HashMap<(u64, SamplerDim), u64>,
    /// Entries a newer upload of the same guest texture has displaced. They stay cached -
    /// content comes back - but eviction takes them first. See the note at the site for
    /// the measurement that says releasing them outright is a 5x RISE in upload traffic.
    pub(super) view_dead: HashSet<(u64, SamplerDim)>,
    /// Keys this run has EVICTED and not yet seen come back, so a later upload of one can be
    /// counted as THRASH rather than as a cold upload. Diagnostic only - see
    /// `EncodeWork::tex_reuploaded_after_evict` for why an upload count cannot answer this.
    pub(super) views_evicted: HashSet<(u64, SamplerDim)>,
    /// Bumped once per frame by [`GxmRenderer::encode_chain`]. Entries stamped with it are
    /// in use RIGHT NOW and are not eviction candidates at any budget.
    pub(super) views_epoch: u64,
    /// The largest ONE-FRAME texture working set seen so far, in bytes, and the floor the
    /// budget is raised to.
    ///
    /// # Why the budget has to learn this
    /// Per-entry eviction stops the cache collapsing, but it cannot make a working set fit
    /// a budget smaller than itself: whatever is evicted early in a frame is wanted again
    /// later in the same frame, and the result is still a re-decode and a re-upload of
    /// most of it, every frame.
    ///
    /// A cache that cannot hold ONE frame is worse than no cache at all, and the bytes are
    /// not optional either way - every texture a frame samples has to be resident while
    /// that frame is encoded, so this floor asks for nothing the frame was not already
    /// going to hold. What it buys is keeping them for the NEXT frame, which on a title
    /// that redraws the same screen is the entire hit rate.
    ///
    /// MEASURED on one title's campaign map: 226 textures / 330 MB against the 256 MB
    /// default, `0.97 cache clears` a frame, `build 718 ms`, 1 fps.
    pub(super) views_frame_high: usize,
    /// Bytes stamped with the CURRENT epoch, i.e. this frame's working set as it builds.
    pub(super) views_frame_bytes: usize,
    /// Which block family this renderer's DEVICE accepts, resolved on the first encode and
    /// never again - see the note at the resolution site for why asking per draw is a browser
    /// regression rather than a style question. `None` until the first frame, because the
    /// device is not available where this struct is built.
    ///
    /// A family rather than a boolean, because the two answers are not interchangeable: an
    /// ETC2-only adapter handed a BC block would be a device-lost, not a fallback.
    pub(super) bc_supported: Option<BlockFamily>,
    /// Samplers by `(linear, addr_mode_u, addr_mode_v)`. A handful per title.
    pub(super) samplers_by_mode: HashMap<(bool, u32, u32), wgpu::Sampler>,
    /// How to choose the clip-`w` sign correction (`VITASLOP_GXP_NEGW`).
    pub(super) negw: NegW,
    /// What interpreting each shader pair's vertex program over its own mesh said about the
    /// projection behind it (see `measure_clip`). Measured ONCE per key, on the first draw
    /// that uses it. `None` records a pair whose program does not interpret, so it is not
    /// re-attempted.
    ///
    /// The DECISIONS are not per pair - see [`GxpLive::decide_scene_negw`].
    pub(super) negw_by_key: HashMap<u64, Option<ClipStats>>,
    /// >>> HOW MANY TIMES A PAIR THAT PRODUCED **NO EVIDENCE** HAS BEEN ASKED AGAIN, AND THE
    /// >>> REASON THIS COUNTER EXISTS IS THAT IT WAS 76% OF ONE TITLE'S ENCODE.
    ///
    /// `negw_by_key` deliberately files only a measurement that found geometry, so a pair
    /// whose sampled vertices all land outside the frustum is re-measured on its next
    /// appearance - the next frame may show the geometry that settles it. What that costs
    /// was never measured until now: MEASURED on a baseball title's gameplay frame, **the
    /// clip verdict was 5.02 ms of a 6.62 ms encode, over THIRTEEN vertex programs
    /// interpreted every single frame** - five distinct pairs of its 960x544 composite pass,
    /// several of them twice in one frame, none of which can ever produce evidence because
    /// that pass is orthographic and its sampled `w` has zero spread. Each measurement
    /// RECOMPILES the vertex program, parses the blob three times, decodes the secondary
    /// shader and interprets up to 256 vertices; 386 us apiece.
    ///
    /// So the retry is kept and BOUNDED: a pair is re-measured on its 1st, 2nd, 4th, 8th,
    /// ... appearance and skipped in between. A pass that eventually shows geometry is still
    /// found, within a factor of two of the frame it would have been found on - and it had
    /// already been wearing the default verdict for every frame until then, so the delay
    /// cannot introduce a wrong verdict that was not already there. The steady state costs
    /// nothing, which is the whole point.
    ///
    /// THE ONE ASSUMPTION, STATED: a skipped appearance is treated as if it would also have
    /// found nothing. That is what makes the skip exact for the vote (an empty measurement
    /// and a skipped one both contribute zero), and it is an assumption, not a proof.
    pub(super) negw_retry: HashMap<u64, u32>,
    /// Whether THIS pass's projection puts clip `w` negative in front of the camera, as
    /// decided by [`GxpLive::decide_scene_negw`] before the pass's draws are walked.
    pub(super) scene_negw: bool,
    /// THIS pass's `(a, c)` in `guest window depth = a + c/w` (see [`ClipStats::depth_fit`]).
    pub(super) scene_depth_fit: (f32, f32),
    /// Both verdicts, per render-target address, once a frame's draws produced evidence for
    /// them. Keeps the measurement off the per-frame path: a projection belongs to the pass,
    /// and a pass does not change projection convention between frames.
    pub(super) negw_by_target: HashMap<(u32, u32, u32), (bool, (f32, f32))>,
    /// The depth fit of the pass that most recently WROTE each colour-surface address,
    /// keyed by the address alone - which is all a later pass sampling that surface knows.
    /// Separate from [`Self::negw_by_target`] because that one is keyed by address AND
    /// extent (a title recycles a buffer between passes of different sizes), and last
    /// writer is what "the pass that wrote this surface" means.
    pub(super) fit_by_addr: HashMap<u32, (f32, f32)>,
    /// group0/group1 bind groups over the pass's uniform ARENA, one per (shader pair,
    /// target format, group) rather than one per draw - the draw supplies only a dynamic
    /// offset. Cleared wholesale when the arena buffer is re-created, because a bind group
    /// names a specific buffer; `ubo_bgs_gen` is that buffer's generation.
    /// Each entry carries the frame epoch it was last USED in, so the cap below sheds the
    /// pairs a title has stopped drawing rather than the ones it is drawing now.
    /// The last key field is a PER-DRAW fragment window block's size (`GxpPrepared::fmem_bytes`),
    /// which its binding must name; 0 for group 0 and for every fixed-size block.
    pub(super) ubo_bgs: HashMap<(usize, u64, wgpu::TextureFormat, u32, u8, u32), (wgpu::BindGroup, u64)>,
    /// Per arena SLOT, the generation of the uniform buffer its cached entries name.
    pub(super) ubo_bgs_gen: HashMap<usize, u64>,
    /// Repacked vertex streams, keyed by `(pipeline key, content hash of the guest
    /// stream)`. The repack walks every component of every vertex, and a world pass here
    /// submits meshes of ~3800 vertices with eleven components each, EVERY FRAME, from
    /// byte-identical guest bytes - the static world does not change between frames. The
    /// key is a content hash rather than the guest address precisely because a title also
    /// has dynamic geometry at a fixed address, which an address key would serve stale.
    ///
    /// Each entry holds `(the guest stream it was repacked from, the repacked bytes)`. The
    /// source is not there for lifetime reasons - it is the hit's PROOF. See the lookup in
    /// `prepare`: a content-hash cache over geometry that trusts its hash fails by silently
    /// drawing another draw's mesh, and this one did.
    pub(super) packed: PackedCache,
    /// The same entries, reached WITHOUT hashing: `(pipeline key, allocation address,
    /// length)` of the guest stream.
    ///
    /// # Why the content hash alone was not enough
    /// The content hash is what makes the cache correct across a re-snapshot into a new
    /// allocation, and it is the reason a hit is verified against the source bytes. But it
    /// is computed on EVERY draw of EVERY frame, hit or miss, and it reads the whole guest
    /// vertex stream to do it - so the cache that exists to stop the renderer touching all
    /// of a frame's geometry was touching all of it anyway, just to find out it did not
    /// have to. That is the shape of cost this project keeps finding: work proportional to
    /// the frame's volume on the path that was supposed to make the volume free.
    ///
    /// An allocation address is a sound key here for one reason, and it is the reason it
    /// must stay: **the entry holds a strong `Arc` to the very stream it was repacked
    /// from**, so that allocation cannot be freed while the entry lives, and no later
    /// stream can be handed the same address. `Arc<[u8]>` has no interior mutability, so
    /// the bytes behind a live address cannot change either. The lookup still asserts it
    /// with `Arc::ptr_eq` rather than trusting the argument - the same discipline the
    /// content path uses, and for the same reason: a geometry cache that is wrong draws
    /// another mesh, confidently, with nothing anywhere reporting it.
    ///
    /// A miss here falls through to the content hash, so nothing that used to hit stops
    /// hitting. It is a fast path, not a replacement.
    pub(super) packed_by_alloc: PackedAllocCache,
    /// Repacked vertices and expanded indices that have not changed since the renderer first
    /// saw them, resident on the GPU instead of copied into a pass arena every frame. See
    /// [`ResidentHeap`] for the measurement this is aimed at and for why an address is a
    /// sound key here. `VITASLOP_RESIDENT_GEOM=0` is the A/B arm.
    pub(super) resident_v: ResidentHeap,
    pub(super) resident_i: ResidentHeap,
    /// Index allocations seen before, so an index list is promoted on its SECOND sighting
    /// rather than its first - the vertex side gets the same test for free from
    /// `packed_by_alloc`.
    ///
    /// >>> IT IDENTIFIES THE ALLOCATION, NOT THE ADDRESS. A set of bare addresses says
    /// > > > "seen before" for an address a FREED list used to occupy, so a title whose index
    /// > > > buffers churn promotes a fresh allocation every frame and fills the heap with
    /// > > > geometry no draw will ever ask for again. MEASURED on one of them's on-track
    /// > > > run before this distinction existed: **9,580 index meshes placed in 328 frames,
    /// > > > filling the 48 MB heap and resetting it** - about 29 dead promotions a frame.
    ///
    /// >>> AND IT IS A `Weak`, BECAUSE A STRONG ONE WOULD BE A LEAK. Holding the `Arc` makes
    /// > > > the test exact and ALSO keeps every index list this map has ever seen alive - up to
    /// > > > the cap, which on the title above is thousands of expanded index buffers the title
    /// > > > itself has finished with. A `Weak` is exactly as good a test (a dead one cannot be
    /// > > > upgraded, and an address recycled after the original died fails the upgrade, which is
    /// > > > the honest answer) and owns nothing.
    ///
    /// At its cap the DEAD entries go first, then the oldest-filed - see `prune_seen`, and
    /// note that a liveness prune alone cannot bound this map, because the packed-vertex
    /// caches hold strong references to the very buffers it points at.
    /// Each entry carries the FRAME it was filed in, so the cap can shed the candidates
    /// whose second sighting never came - see `prune_seen` for why liveness alone cannot
    /// bound this map.
    pub(super) resident_i_seen: HashMap<(u64, usize, usize), (std::sync::Weak<[u8]>, u64)>,
    /// The vertex twin of [`Self::resident_i_seen`], keyed on the PACKED content `Arc`
    /// (plus the pipeline, whose repack plan shaped those bytes). Same `Weak` argument,
    /// same cap, same clearing.
    pub(super) resident_v_seen: HashMap<(u64, usize, usize), (std::sync::Weak<[u8]>, u64)>,
    /// Shader pairs `precompile_pairs` has already considered, by the ALLOCATION of the two
    /// program blobs. The pending list is re-offered every frame and `module_key` hashes
    /// both blobs, so without this the preparation costs a few hundred kilobytes of hashing
    /// per frame FOREVER - and proportionally more the more pairs a title names.
    ///
    /// It holds no `Arc`, unlike the geometry caches, and the reason it may not is the
    /// reason it is safe: a stale address here can only make a pair be SKIPPED, and a
    /// skipped pair compiles at its first draw exactly as it did before any of this existed.
    /// A wrong answer costs a hitch, never a picture.
    pub(super) precompile_seen: HashSet<(usize, usize)>,
    pub(super) resident: bool,
    /// The byte budget for each of the two heaps (`VITASLOP_RESIDENT_GEOM_MB`, per heap).
    pub(super) resident_budget: u64,
    /// The `@group(3)` depth-range bind group, keyed by `(pipeline key, the depth
    /// range's bits)`.
    ///
    /// That group holds ONE vec4 - the scene's depth min and scale - which is the
    /// same for every draw in a scene. Building a fresh 16-byte buffer and bind
    /// group per draw meant a couple of hundred GPU allocations a frame to say the
    /// same thing each time. Keyed by pipeline as well as by value because each
    /// pipeline owns its own `BindGroupLayout` object; a shared group-3 layout
    /// would collapse this to one entry, and is the right follow-up.
    ///
    /// >>> THIS CACHE IS BOUNDED, AND IT WAS NOT. The key holds the depth range's RAW
    /// > > > BITS, and a depth range is a continuous quantity: on a menu it is constant and this
    /// > > > cache holds a handful of entries, but in a RACE the near/far pair moves every frame,
    /// > > > so every frame mints entries that will never be asked for again. Each one retains a
    /// > > > GPU buffer AND a bind group for the rest of the run. Measured on the committed race
    /// > > > playbook in desktop Chrome: **30-33 buffers created EVERY FRAME, sustained**, in a
    /// > > > renderer whose steady state should create none - and the GPU process climbed from a
    /// > > > flat 0.20 GB through the menus to 5.8 GB (13.4 GB private) by frame 9000. The two
    /// > > > sibling caches on either side of this one are both bounded (`ubo_bgs` by generation,
    /// > > > `sampler_bgs` by [`SAMPLER_BG_CACHE_CAP`]); this one was simply missed.
    ///
    /// Bounded the same way `sampler_bgs` is, and for the same reason that makes a wholesale
    /// clear safe here: the key IS the value, so a cleared entry is rebuilt to something
    /// byte-identical. Clearing costs a rebuild and can never cost a wrong answer - unlike
    /// [[vitaslop-content-hash-cache-must-verify]], where the value was geometry.
    /// **The buffer is held beside the bind group so an eviction can `destroy()` it.**
    /// It used to be dropped the instant it was built - the entry kept only the bind group,
    /// which holds the buffer alive in JS but leaves NOTHING to call `destroy()` on. So
    /// every wholesale clear handed the collector up to [`DEPTH_BG_CACHE_CAP`] buffers at
    /// once, on the engine where dropping a handle reclaims nothing on our schedule. Same
    /// class of defect as [[vitaslop-browser-gpu-needs-destroy]], and invisible on a MENU
    /// (a constant depth range caches a handful of entries) while minting ~30 a frame in a
    /// RACE, which is the screen this has to survive.
    /// The last key element is a STORING program's mirror identity and window headers (see
    /// `store_mirror`), which its group-3 bind group names; `None` for every other pair.
    #[allow(clippy::type_complexity)]
    pub(super) depth_bgs: HashMap<
        (u64, ((((u64, u32, u32, u32), u64), u64), [u32; 16]), bool, Option<(u64, Vec<u32>)>),
        (wgpu::BindGroup, wgpu::Buffer),
    >,
    /// Buffers evicted from `depth_bgs`, waiting to be destroyed on the renderer's frame
    /// schedule. They cannot be destroyed at the eviction itself: a clear happens during
    /// `prepare`, and a draw already prepared THIS frame may still name the bind group that
    /// owns one. `encode_chain` drains this into the renderer's graveyard, which destroys
    /// after the submit - the same rule the arenas follow.
    pub(super) depth_retired: Vec<wgpu::Buffer>,
    /// The writable mirrors of guest memory that fragment programs store to - see
    /// `store_mirror`. Empty unless a program stores.
    pub(super) store_mirrors: StoreMirrors,
    /// The `@group(2)` SAMPLER bind group, keyed by the pipeline and by exactly what it
    /// binds: each unit's chosen texture view and its sampler state.
    ///
    /// This group used to be built with `device.create_bind_group` on EVERY draw of every
    /// frame - a real GPU object per draw, which is the thing
    /// [[vitaslop-per-pass-render-arenas]] exists to forbid, and in the browser it is also
    /// a wasm/JS boundary crossing per draw. A bind group is a pure function of the views
    /// and samplers it names, and both of those are already content-addressed caches, so
    /// the same draw next frame asks for a group that is byte-for-byte the one it got
    /// last frame.
    ///
    /// Entries naming a render target THIS frame produced are NOT cached: those views
    /// belong to textures the frame allocates, so a cached group would name a dead one.
    /// Those draws still build fresh, and the counter says how many.
    ///
    /// Each entry carries the view-cache keys it NAMES, so an eviction can drop exactly the
    /// groups that just went stale. See the eviction site for why a generation counter was
    /// the wrong instrument here.
    pub(super) sampler_bgs: SamplerBgCache,
    /// The group the PREVIOUS draw of this pass got, against a fingerprint of what decided
    /// it - see `make_sampler_bg`. Reset at every pass boundary, because the maps a unit
    /// resolves through (this frame's rendered targets) change there and nowhere else.
    pub(super) last_sampler_bg: Option<(u64, wgpu::BindGroup)>,
    /// >>> ONE PUSH PER DISTINCT WINDOW SET PER PASS. A memory-window block is a pipeline's
    /// layout filled from the draw's window snapshots, and a pass binds the same windows
    /// (the same `Arc`s - see `capture_mem_windows`) for hundreds of draws: pushing each
    /// draw its own copy was 0.95 ms of a 10 ms prepare (`mem-windows` in the split). Keyed
    /// on the pipeline and the windows' identities, verified by `Arc::ptr_eq`, cleared at
    /// every pass because the offsets name that pass's arena.
    pub(super) mem_push_memo: HashMap<(usize, u8, u64), (u32, Vec<(u32, std::sync::Arc<[u8]>)>)>,
    /// >>> THE SAME KEYS, BUT SCOPED TO THE **FRAME** - AND IT ONLY COUNTS.
    ///
    /// `mem_push_memo` is cleared at every pass, because a pass gets a fresh arena and every
    /// memoised offset named the previous one's. So a window block a frame uses in ten
    /// passes is uploaded ten times, and the arena's byte line cannot say how much of its
    /// `guest-memory WINDOW blocks` figure is that repetition rather than distinct data.
    /// Sizing the prize before building a frame-scoped arena for them: this set is the keys
    /// pushed so far THIS FRAME, and a miss on a key it already holds charges
    /// `arena_bytes_memwin_repeat`. It buys nothing on its own - it is the measurement that
    /// says whether the thing worth building is worth building.
    pub(super) mem_push_frame: HashSet<(usize, u8, u64)>,
    /// The raw 64-bit targets this PASS's window fill copies from
    /// (`GxmRenderer::fill_windows_from_raw_targets`): what this frame rendered raw before it,
    /// less the pass's own attachment. Set at every pass; a per-draw window block leaves the
    /// bytes they cover out of its copy and its upload - see `push_mem_windows_per_draw`.
    pub(super) raw_fill: Vec<crate::gpu::RawFillTarget>,
    /// The byte ranges of this pass's uniform arena the fill writes and the upload therefore
    /// skips, as `[start, end)` offsets - see `push_mem_windows_per_draw`. Cleared at every pass.
    pub(super) udata_holes: Vec<(usize, usize)>,
    /// >>> EVERY GROUP THIS PASS HAS ALREADY DECIDED, BY THE SAME FINGERPRINT
    /// >>> `last_sampler_bg` MATCHES - `pre` -> the CONTENT key `sel` it resolved to.
    ///
    /// `last_sampler_bg` answers only "is this the draw immediately before me", which is
    /// the wrong shape for a frame that interleaves materials. This catches the rest of the
    /// repeats within a pass.
    ///
    /// >>> IT IS A SMALL WIN AND THE MEASUREMENT SAYS SO. On a retail sports title's
    /// > > > gameplay frame (browser, real GPU, 672 draws) it answers **32 draws a frame** on top
    /// > > > of the previous-draw slot's 190, and a CDP profile of the worker could not tell the
    /// > > > two builds apart. What it also established is worth more than the win: the pass
    /// > > > carries about **310 DISTINCT sampler fingerprints**, so the resolution loop is not
    /// > > > where `make_sampler_bg` spends its 10% of the thread - the ~2 TEXTURE UPLOADS a
    /// > > > frame it performs inside that loop are. Do not re-open the loop on the strength of
    /// > > > the function's total; look at the upload.
    ///
    /// It is sound on exactly the argument the one-slot cache is sound on, and no more:
    /// everything the resolution consults besides the fingerprint is CONSTANT WITHIN A
    /// PASS (this frame's rendered colour/depth/cube maps, the snapshot set, `rtt_epoch`),
    /// and this is cleared at the pass boundary where those are rebuilt. Crucially it
    /// stores the KEY and not the group, so a `sel` that has gone stale anyway - an
    /// eviction drops the entry from `sampler_bgs`, which is the same map this then probes
    /// - MISSES and falls through to the full loop. A wrong answer is not reachable
    /// through a stale entry; only a wasted probe is.
    pub(super) sampler_pre: HashMap<u64, u64>,
    /// The GPU texture transcoder's compute pipelines ([`crate::texenc`]).
    ///
    /// Built on first use rather than in `new`, for the same reason `bc_supported` is asked
    /// once: it compiles three compute shaders, and a run whose adapter takes the guest's
    /// blocks verbatim never transcodes anything and should not pay for them. Held here
    /// rather than in a global because "there is only ever one device" is true today and is
    /// not a thing this code should assert.
    pub(super) texenc: Option<crate::texenc::Transcoder>,
}

/// Which depth the recompiled vertex stage writes (`VITASLOP_GXP_ZFIX`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ZFix {
    /// The scene's `-1/w` range, normalized onto [0,1] (`=range`): the same quantity and the
    /// same range the FIXED-FUNCTION path writes, so both kinds of draw share one comparable
    /// depth buffer. Costs a dependency on the scene depth range, which is measured through
    /// the software path's own reflected transform - and that is why it is no longer the
    /// default. It is a quantity the guest never computes, so the guest's own depth FUNCTION
    /// has no meaning against it, and any draw whose projection differs from the world's
    /// (a 2D overlay has a constant `w`) falls outside the measured range and CLAMPS, to the
    /// nearest value. Keep it only for a frame that still mixes in fixed-function draws.
    Range,
    /// The GL->WebGPU clip-depth remap `(z + w) / 2` (`=gl`), for a guest whose clip z is
    /// read in [-w, w].
    ///
    /// GXM is NOT such a guest, which is measurable rather than a matter of provenance:
    /// `count_clip_w_signs` fits `z = a*w + c` per pass, and one title's world pass fits
    /// `a = 1.00003, c = -2.0`, i.e. `z/w = 1 - 2/w`, which runs from 0 at the near plane
    /// (`w = 2`) to 1 at infinity. Its clip z is already in [0, w] - the D3D/WebGPU
    /// convention, not the GL one - so this remap would squash the whole scene into the far
    /// half of the buffer.
    Gl,
    /// Pass the guest's clip z straight through (`=0`/`=off`).
    ///
    /// The depth buffer then holds the guest's own window depth `z/w` exactly. It is also
    /// what makes WebGPU CLIP every fragment whose depth leaves [0, w] - which PowerVR does
    /// NOT do. MEASURED: one title's whole front end disappears under this mode and renders
    /// perfectly under [`ZFix::Clamp`], which differs from it only by the clamp.
    Off,
    /// The guest's own window depth `z/w`, CLAMPED to [0,1] (`=clamp`).
    ///
    /// This is the faithful one, and it is two facts put together:
    ///
    /// 1. GXM clip z is already in [0, w] - the D3D/WebGPU convention, not the GL one.
    ///    `count_clip_w_signs` fits `z = a*w + c` per pass; one title's world pass fits
    ///    `a = 1.00003, c = -2.0`, so `z/w` runs from 0 at its near plane (`w = 2`) to 1 at
    ///    infinity. So the depth buffer can hold the guest's own value untouched, and its
    ///    captured `SceGxmDepthFunc` then means exactly what it means on hardware.
    /// 2. PowerVR CLAMPS depth where an immediate-mode rasteriser CLIPS. A primitive that
    ///    runs past the far plane is drawn AT the far plane, not thrown away. WebGPU only
    ///    offers that through the optional `depth-clip-control` feature, so the clamp is
    ///    done in the shader instead: scaling by `c.w` re-encodes the clamped window depth
    ///    as a clip z the rasteriser will divide back out.
    ///
    /// The result depends on nothing but the draw's own projection, so a 2D overlay lands
    /// where the guest put it rather than at the end of the world pass's range.
    Clamp,
    /// The guest's OWN viewport depth mapping, applied PROJECTIVELY (the DEFAULT).
    ///
    /// >>> IT IS `z * zScale + w * zOffset`, NOT `clamp(z/w * zScale + zOffset, 0, 1) * w`.
    /// > > > THE SECOND SPELLING ATE A CHARACTER'S FEET.
    /// The two agree for every vertex with `w > 0` whose depth does not clamp, so the
    /// difference lives entirely in the vertices where they do not - and a vertex is not
    /// where a depth remap may be decided. The divide form had to guard on `c.w > 0.0` (a
    /// vertex behind the eye has no `z/w`) and left those vertices carrying a RAW clip z on
    /// a different scale from every other vertex of the same triangle. The rasteriser then
    /// interpolates a depth plane through one correct corner and one nonsense one.
    ///
    /// MEASURED, on the fighting title's rooftop stage: its ground is ONE 4800x4800 quad
    /// grid of 36 vertices, and 15 of them are behind the camera. The centre quad - the one
    /// the fighters stand on - has two such corners, so the ground's interpolated depth came
    /// out too NEAR over a band, and the ground won the depth test against the bottom
    /// ~22 screen pixels of every boot. It reads exactly like "the characters are drawn too
    /// low", and it is not: skinning the foot mesh offline puts its lowest vertex at world
    /// `z = -0.9` on a floor plane at `z = 0`, which is correct to within a rounding.
    /// The same defect is nearly invisible on a finely tessellated stage, because no small
    /// triangle straddles the camera - which is why it looked like a per-stage art problem.
    ///
    /// The projective form needs no guard and no clamp: `z' / w` is the intended window
    /// depth for every `w`, and the CLAMP that PowerVR does (rather than clipping) is asked
    /// of the rasteriser through `unclipped_depth` where the device offers
    /// `depth-clip-control`. See [`unclipped_depth_available`].
    ///
    /// `sceGxmSetViewport` takes `zOffset` and `zScale`, and the hardware's window depth is
    /// `z/w * zScale + zOffset`. That is not a convention to be inferred, it is state the
    /// guest SET, and [`Clamp`](Self::Clamp) - which is this map with `zScale = 1,
    /// zOffset = 0` hard-coded - reads it wrong wherever the guest asked for anything else.
    ///
    /// >>> AND A TITLE DOES. MEASURED on one title's title screen: that pass reports
    /// > > > `zOffset=0.5 zScale=0.5`, i.e. clip z in [-w, w], and its whole lattice sits at clip
    /// > > > `z ~ -0.98`. Under `Clamp` every vertex of it lands at window depth EXACTLY 0, so a
    /// > > > `LESS_EQUAL` depth test can reject nothing and the mesh's own hidden faces paint over
    /// > > > the faces in front of them - which is what the over-embossed lattice was. Under the
    /// > > > guest's own mapping the same geometry spans 0.0098..0.0107, the prism tops sit in
    /// > > > front of their skirts by 0.0009, and the test does what the hardware does.
    /// > > > `VITASLOP_GXP_NODEPTH=1` being BIT-IDENTICAL to the normal render was the fingerprint:
    /// > > > a depth test that rejects nothing is not a working depth test.
    ///
    /// The PowerVR clamp of [`Clamp`](Self::Clamp) is kept - it is a separate fact about the
    /// hardware and it is still true - so this differs from it only by asking the guest.
    /// A pass the guest never gave a viewport keeps the identity mapping, which is exactly
    /// what `Clamp` did, and says so once rather than guessing quietly.
    Viewport,
}

/// How the clip-`w` sign correction is chosen (`VITASLOP_GXP_NEGW`).
///
/// A guest projection may put clip `w` NEGATIVE in front of the camera, in which case
/// WebGPU clips every world draw away and the pass renders black with correct shaders,
/// textures, depth and blend (memory `vitaslop-clip-w-can-be-negative`). The correction
/// is to negate the whole clip vector, which names the same point with `w > 0`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
/// MEASURED on one retail racer at a race frame, and this is what settles which correction
/// is right: with `w` flipped the HUD reads the right way round and the player's vehicle
/// carries readable lettering on its tail; with the WHOLE vector negated the
/// identical frame comes out turned 180 degrees, every glyph mirrored. So the guest's
/// `x`,`y`,`z` are right and the `w` reaching WebGPU has the wrong SIGN - full negation
/// preserves `x/w` exactly, which is precisely why it cannot fix a mirrored picture.
pub(super) enum NegW {
    /// Never correct (`=0`). The clip position goes to WebGPU as the shader produced it.
    Off,
    /// Measure it per PASS (`=auto`, the default) and flip `w` on every pair of a pass
    /// whose projection is negative. See [`GxpLive::decide_scene_negw`] for why the
    /// decision cannot be taken per shader pair.
    Auto,
    /// Negate the WHOLE clip vector on the pairs `Auto` would correct (`=negate`).
    ///
    /// Kept because it is the other reading of the same measurement and the two are one
    /// experiment apart: negating names the same point with `w > 0` (homogeneous coordinates
    /// are scale-invariant), so it lifts the clip and changes nothing else. On this title it
    /// renders the frame upside down - see the note above.
    Negate,
    /// Correct EVERY pass, without measuring (`=force`).
    ///
    /// A diagnostic, and specifically the one for a pass whose vertex program the CPU
    /// interpreter cannot run - a program that samples textures, or indexes a uniform
    /// array. `Auto` then has no evidence and leaves the pass uncorrected, which is
    /// indistinguishable on screen from a pass that needed no correction: both render
    /// black. This settles which, in one run.
    Force,
}

/// Diagnostic (`VITASLOP_GXP_DUMP`): print each recompiled draw's inputs.
///
/// Cached, because this is tested once per DRAW and reading an unset environment
/// variable on Windows copies and re-encodes the whole environment block.
pub(super) fn gxp_dump() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_GXP_DUMP"))
}
