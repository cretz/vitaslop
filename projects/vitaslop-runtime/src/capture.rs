//! The GXM command-stream capture: the blob-free "it works" signal. The host
//! records what the guest asked the GPU to do (surfaces, programs, per-draw
//! vertex/index/uniform snapshots) without emulating a GPU or drawing a pixel.
//! A software rasterizer or wgpu backend later consumes this to produce frames.

use std::sync::Arc;

/// One vertex attribute as declared by the guest's vertex program: which stream,
/// byte offset within a vertex, source format, and component count. `format` and
/// the counts are the raw GXM enum values so the consumer decodes exactly what
/// the guest laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    pub stream_index: u16,
    pub offset: u16,
    pub format: u8,
    pub component_count: u8,
    pub reg_index: u16,
}

/// A color surface the guest initialized: pixel format plus geometry and the
/// guest address of its pixel buffer (the target a scene renders into).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorSurface {
    pub format: u32,
    /// The `SceGxmColorSurfaceType` (LINEAR/TILED/SWIZZLED) the guest set at init,
    /// read back by `sceGxmColorSurfaceGetType`. Not consumed by the renderer (it
    /// resolves the pixel layout from the data pointer and stride), but recorded so
    /// the getter round-trips exactly what the guest set.
    pub surface_type: u32,
    pub width: u32,
    pub height: u32,
    pub stride_pixels: u32,
    pub data_addr: u32,
}

/// The fixed-function pipeline state a title sets on the GXM context between draws
/// with the `sceGxmSet*` family (cull, depth, stencil, viewport, region clip,
/// polygon mode). GXM context state is sticky - a setter mutates the current state
/// and every later draw inherits it until it is changed again - so the host tracks
/// the live values and snapshots them into each [`Draw`], exactly as uniforms and
/// bound textures are captured. A renderer reproduces a draw from this snapshot
/// without replaying the call stream. Every field is the raw GXM enum word (e.g.
/// `cull_mode` is a `SceGxmCullMode` value, `front_depth_func` a `SceGxmDepthFunc`
/// value); [`Default`] is GXM's documented context default for a field a title
/// leaves unset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderState {
    pub cull_mode: u32,
    pub two_sided: u32,
    pub front_depth_func: u32,
    pub back_depth_func: u32,
    pub front_depth_write: u32,
    pub front_fragment_program_enable: u32,
    pub back_fragment_program_enable: u32,
    pub front_polygon_mode: u32,
    pub front_point_line_width: u32,
    pub front_stencil_ref: u32,
    pub front_stencil_func: u32,
    pub front_stencil_op_fail: u32,
    pub front_stencil_op_depth_fail: u32,
    pub front_stencil_op_depth_pass: u32,
    pub front_stencil_compare_mask: u32,
    pub front_stencil_write_mask: u32,
    pub viewport_enable: u32,
    /// `xOffset, xScale, yOffset, yScale, zOffset, zScale` from sceGxmSetViewport.
    pub viewport: [f32; 6],
    pub region_clip_mode: u32,
    /// `xMin, yMin, xMax, yMax` from sceGxmSetRegionClip.
    pub region_clip: [u32; 4],
}

impl Default for RenderState {
    fn default() -> Self {
        RenderState {
            cull_mode: 0x0000_0000,               // SCE_GXM_CULL_NONE
            two_sided: 0x0000_0000,               // SCE_GXM_TWO_SIDED_DISABLED
            front_depth_func: 0x00C0_0000,        // SCE_GXM_DEPTH_FUNC_LESS_EQUAL
            back_depth_func: 0x00C0_0000,         // SCE_GXM_DEPTH_FUNC_LESS_EQUAL
            front_depth_write: 0x0000_0000,       // SCE_GXM_DEPTH_WRITE_ENABLED
            front_fragment_program_enable: 0x0,   // SCE_GXM_FRAGMENT_PROGRAM_ENABLED
            back_fragment_program_enable: 0x0,    // SCE_GXM_FRAGMENT_PROGRAM_ENABLED
            front_polygon_mode: 0x0000_0000,      // SCE_GXM_POLYGON_MODE_TRIANGLE_FILL
            front_point_line_width: 1,
            front_stencil_ref: 0,
            front_stencil_func: 0x0E00_0000,      // SCE_GXM_STENCIL_FUNC_ALWAYS
            front_stencil_op_fail: 0,             // SCE_GXM_STENCIL_OP_KEEP
            front_stencil_op_depth_fail: 0,       // SCE_GXM_STENCIL_OP_KEEP
            front_stencil_op_depth_pass: 0,       // SCE_GXM_STENCIL_OP_KEEP
            front_stencil_compare_mask: 0xff,
            front_stencil_write_mask: 0xff,
            viewport_enable: 0x0000_0000,         // SCE_GXM_VIEWPORT_ENABLED
            viewport: [0.0; 6],
            region_clip_mode: 0x0000_0000,        // SCE_GXM_REGION_CLIP_NONE
            region_clip: [0; 4],
        }
    }
}

/// A texture bound to a fragment sampler unit at draw time. Decoded from the
/// guest's 16-byte `SceGxmTexture` control words (format, dimensions, memory
/// layout, data address) with a snapshot of the referenced pixel bytes, so a
/// renderer can sample it without touching guest memory later. Fields are the
/// raw GXM enum parts: `base_format` is the high byte of `SceGxmTextureBaseFormat`
/// (e.g. `0x0c` for U8U8U8U8), `swizzle` is the format's low 24 bits, and
/// `tex_type` is the 3-bit `SceGxmTextureType` selector (`0b011` = LINEAR).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundTexture {
    pub unit: u32,
    pub base_format: u32,
    pub swizzle: u32,
    pub tex_type: u32,
    pub width: u32,
    pub height: u32,
    /// Bytes per row of the snapshotted `pixels` (== the source stride).
    pub stride: u32,
    /// How many `width x height` faces `pixels` holds: 1 for an ordinary texture, 6 for a
    /// CUBE map (`SCE_GXM_TEXTURE_CUBE`/`CUBE_ARBITRARY`), whose faces are stored back to
    /// back in +X, -X, +Y, -Y, +Z, -Z order - the same order WebGPU expects its array layers
    /// in. Face `f` starts at byte `f * face_bytes`.
    pub faces: u32,
    /// Byte size of one face in `pixels` (the whole buffer when `faces` is 1).
    pub face_bytes: u32,
    pub data_addr: u32,
    /// The raw guest bytes of the texture, exactly as laid out in guest memory.
    ///
    /// Shared (`Arc`) rather than owned: one scene binds the same few textures across
    /// hundreds of draws, and a per-draw copy of a 4096x2048 shadow map costs gigabytes a
    /// frame. Sharing also gives consumers a cheap identity key (the pointer) for caching
    /// GPU uploads, so an unchanged texture is uploaded once per frame rather than per draw.
    pub pixels: Arc<[u8]>,
    /// Sampler wrap modes (`SceGxmTextureAddrMode`, 0 = REPEAT) and LOD bias set on
    /// this texture via `sceGxmTextureSet{U,V}AddrMode[Safe]` / `SetLodBias`.
    pub u_addr_mode: u32,
    pub v_addr_mode: u32,
    pub lod_bias: u32,
    /// Minification/magnification filters (`SceGxmTextureFilter`, 0 = POINT/nearest,
    /// 1 = LINEAR) set via `sceGxmTextureSetMinFilter`/`SetMagFilter`. The renderer
    /// bilinear-samples a LINEAR magnified texture and point-samples a POINT one, so
    /// small UI/font-atlas text a title draws with LINEAR filtering is smooth rather
    /// than the broken thin strokes nearest sampling gives at sub-native scale.
    pub min_filter: u32,
    pub mag_filter: u32,
}

/// The per-material fragment-shader inputs recovered by reflecting the bound fragment
/// program's parameter table against its captured default uniform buffer. The real
/// fragment program is a standard forward-lit material - `albedo = baseTexture.rgb *
/// tint`, lit by one directional light plus an ambient term, then fogged - so these are
/// the values the capture renderer needs to reproduce the LIT colour instead of the raw
/// albedo texel. Everything is optional: a 2D/UI shader declares none of it and the fields
/// stay at their neutral defaults (tint white, no light), so the material is a no-op there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FragmentMaterial {
    /// Base-colour multiplier (`AlbedoColour` / `Primarytint`). The sampled albedo texel is
    /// multiplied by this; it is why an unlit near-white tyre albedo must be scaled down.
    pub tint: [f32; 3],
    /// World-space direction of the key directional light (`directionalLight0Direction
    /// WorldSpace`) and its colour (`directionalLight0Colour`). `has_light` is false when the
    /// shader declares no directional light (then only ambient/flat albedo applies).
    pub light_dir: [f32; 3],
    pub light_col: [f32; 3],
    pub has_light: bool,
    /// A flat ambient colour (the average of the tiny `diffuseAmbientMap` irradiance texture,
    /// resolved by the renderer) added to the directional term so surfaces facing away from
    /// the light are not pure black. Defaults to a neutral mid-grey when no ambient map is
    /// bound. This is a scene-lighting term, not per-material, but stored per draw for the
    /// stateless renderer.
    pub ambient: [f32; 3],
}

impl Default for FragmentMaterial {
    fn default() -> Self {
        FragmentMaterial {
            tint: [1.0, 1.0, 1.0],
            light_dir: [0.0, -1.0, 0.0],
            light_col: [1.0, 1.0, 1.0],
            has_light: false,
            ambient: [0.35, 0.35, 0.35],
        }
    }
}

/// A single draw call with everything needed to reproduce it, snapshotted from
/// guest memory at draw time (so later guest writes cannot perturb it).
#[derive(Clone, Debug, PartialEq)]
pub struct Draw {
    pub primitive: u32,
    pub index_format: u32,
    pub index_count: u32,
    /// The bound vertex stream buffer bytes.
    pub vertices: Vec<u8>,
    pub vertex_stride: u32,
    pub attributes: Vec<VertexAttribute>,
    /// The index buffer bytes.
    pub indices: Vec<u8>,
    /// The vertex default uniform buffer contents the guest wrote for this draw
    /// (column-major 4x4 MVP for the cube), if any.
    pub uniforms: Vec<f32>,
    /// Fragment textures bound at draw time (one per active sampler unit),
    /// snapshotted from guest memory. Empty for an untextured (vertex-color) draw.
    ///
    /// Ordered so that index 0 is the draw's surface albedo when it has one - see
    /// [`Draw::albedo`], which is what the fixed-function approximation samples. The full list
    /// is what the GXP recompiler binds by unit, so nothing is ever dropped from it.
    pub textures: Vec<BoundTexture>,
    /// The fixed-function pipeline state (cull/depth/stencil/viewport/...) in effect
    /// for this draw, snapshotted from the sticky GXM context state. See [`RenderState`].
    pub render_state: RenderState,
    /// Scene exposure (linear multiplier) recovered from the vertex program's reflected
    /// `vsCoarseExposureReg` uniform. The shaders scale lit albedo by this before
    /// tone-mapping, so a capture renderer that skips it draws the world ~10x too dark.
    /// 1.0 when the shader declares no exposure (2D/UI), so it is a no-op there.
    pub exposure: f32,
    /// The per-material fragment inputs (base-colour tint + directional/ambient light)
    /// reflected from the fragment program and its default uniform buffer. Neutral (a no-op)
    /// for 2D/UI draws and any shader that declares no lit material. See [`FragmentMaterial`].
    pub material: FragmentMaterial,
    /// The model-to-world matrix (column-major 4x4) reflected from the vertex program's
    /// `vsModelToWorldMatrix`, used to bring the per-vertex object-space normal into world
    /// space for the directional-light N.L term (the light direction is world space). Identity
    /// when the shader has no separate world matrix (2D/UI), so lighting there uses the raw
    /// normal - harmless because such draws are not depth-lit.
    pub world: [f32; 16],
    /// The bound vertex `SceGxmProgram` container bytes, snapshotted for the GXP->WGSL
    /// recompiler (live guest-shader) path. Empty unless recompile-capture is enabled
    /// (env `VITASLOP_GXP_LIVE`), so the default capture pays no read/clone cost.
    pub vprog: Vec<u8>,
    /// The bound fragment `SceGxmProgram` container bytes. Empty off the recompiler path.
    pub fprog: Vec<u8>,
    /// Raw vertex default-uniform-buffer (SA bank) bytes exactly as the guest wrote them -
    /// the recompiled vertex shader reads these directly, NOT the MVP-stamped `uniforms`
    /// above (which the fixed-function path needs but the real shader recomputes itself).
    pub vert_sa: Vec<u8>,
    /// Raw fragment default-uniform-buffer (SA bank) bytes exactly as the guest wrote them,
    /// consumed by the recompiled fragment shader's `@group(1)` uniform. Empty off-path.
    pub frag_sa: Vec<u8>,
    /// The vertex program SYNTHESIZES this draw's primitive rather than reading it: the
    /// stream holds one record per sprite (a centre plus an expansion basis - a
    /// scale/rotation, or an explicit right/up billboard axis pair) and the shader builds
    /// the corners. See [`crate::host::VitaState::reflected_shader_expanded`].
    ///
    /// The fixed-function approximation has no shader, so there is nothing here it can
    /// rasterize: joining the raw records as triangles connects unrelated sprite centres
    /// into geometry the game never draws. The software renderer skips such a draw and
    /// says so; the GXP recompiler runs the real vertex program and renders it properly.
    pub shader_expanded: bool,
}

impl Draw {
    /// The texture the fixed-function approximation samples across this draw's triangles.
    ///
    /// This is index 0 by the ordering contract on [`Draw::textures`]. Note that a draw can
    /// bind textures that are not surface albedo at all - a one-dimensional lookup table (a fog
    /// ramp, indexed by depth) or a cube map (an irradiance probe, indexed by a direction) - and
    /// for a draw that binds ONLY those, the approximation stretches one across the surface. It
    /// is visibly not what the guest draws (a start-line decal paints as vertical bands of the
    /// fog ramp), but it is better than the alternative: refusing to pick one leaves the draw
    /// with no texture at all, which the renderers paint as the magenta missing-texture marker,
    /// and it hides real geometry. Reproducing these materials needs the recompiled fragment
    /// shader that reads each sampler for what it is.
    pub fn albedo(&self) -> Option<&BoundTexture> {
        self.textures.first()
    }
}

/// One scene (BeginScene to EndScene): its render target color buffer and the
/// draws issued into it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    pub color: Option<ColorSurface>,
    pub draws: Vec<Draw>,
}

impl Scene {
    /// Reduce the scene's draws to the neutral [`DrawBatch`](vitaslop_platform::gpu::DrawBatch)
    /// list the shared cube pipeline consumes. Keeps only draws the fixed-function
    /// cube shader can render (triangle lists with a 16-byte interleaved vertex
    /// and a full 4x4 MVP), so both the native and browser GPU paths select the
    /// same draws from one place. The software rasterizer stays independent.
    pub fn draw_batches(&self) -> Vec<vitaslop_platform::gpu::DrawBatch> {
        self.draws
            .iter()
            .filter(|d| d.primitive == 0 && d.uniforms.len() >= 16 && d.vertex_stride == 16)
            .map(|d| {
                let mut mvp = [0f32; 16];
                mvp.copy_from_slice(&d.uniforms[..16]);
                vitaslop_platform::gpu::DrawBatch {
                    mvp,
                    vertices: d.vertices.clone(),
                    indices: d.indices.clone(),
                    index_count: d.index_count,
                    // GXM index format: 0 is U16, anything else U32.
                    index_u32: d.index_format != 0,
                }
            })
            .collect()
    }
}

/// A game->OS "egress" event: something the guest handed to the system that
/// carries game-authored, human-readable meaning - a savedata write, a trophy
/// unlock, a leaderboard score submission. This is the content-ful, title-agnostic
/// seam a conformance recipe asserts on: it is the Vita OS API surface, NOT the
/// game's private memory (which would be per-title RE and brittle). Every event is
/// tagged with the display frame it occurred on, so a recipe can assert both what
/// the game did and (loosely) when - the human-readable proof a run reached a game
/// milestone, replacing a screenshot assertion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EgressEvent {
    pub frame: u64,
    pub kind: EgressKind,
}

/// The kinds of game-authored egress the ledger records. Deliberately small and
/// generic - each variant is a Vita OS surface every title shares, so the ledger
/// generalizes to the next game with no per-title work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EgressKind {
    /// A writable file under a persisted mount (`savedata0:`, `ux0:`) was closed
    /// after being written: the game persisted state. `bytes` is the final size and
    /// `ascii` is a printable-ASCII preview (non-printable bytes shown as `.`), so a
    /// human-readable save - a high score, an unlocked level id - is visible without
    /// a pixel. This is where a title's score often lands on the offline path.
    SaveWrite { path: String, bytes: usize, ascii: String },
    /// A trophy was unlocked (`sceNpTrophyUnlockTrophy`): the trophy id.
    Trophy { id: i32 },
    /// A score was submitted to a leaderboard (`sceNpScore*`): board id and value.
    /// Only fires if the online path is enabled (offline forces signed-out).
    ScoreSubmit { board: u32, score: i64 },
}

/// Render up to `max` bytes of `data` as a printable-ASCII preview: printable bytes
/// verbatim, everything else as `.`. Keeps the egress ledger human-readable and
/// content-bounded (no raw blobs) for logs and assertions.
pub fn ascii_preview(data: &[u8], max: usize) -> String {
    data.iter()
        .take(max)
        .map(|&b| if (0x20..0x7f).contains(&b) { b as char } else { '.' })
        .collect()
}

/// The whole recorded stream across a run.
#[derive(Default)]
pub struct Capture {
    /// Completed scenes, in submission order (one per BeginScene/EndScene pair).
    pub scenes: Vec<Scene>,
    /// Display buffer addresses presented, in order (from the display queue).
    pub presents: Vec<u32>,
    /// (library_nid, func_nid) pairs the guest called that are not implemented
    /// yet, with a count, so gaps are visible instead of silently skipped.
    pub unimplemented: Vec<(u32, u32, String)>,
    /// Total host calls serviced, for sanity.
    pub call_count: u64,
    /// Ordered trace of recently serviced calls' function NIDs, for debugging.
    /// Bounded to the most recent [`TRACE_CAP`] entries (see [`Capture::record_call`]):
    /// a 3D title makes tens of millions of host calls during boot, and every
    /// consumer of this trace (the exit dump, the probe's trace.txt and per-thread
    /// tails) only reads the recent window.
    pub trace: Vec<u32>,
    /// The guest thread id that made each serviced call, parallel to [`trace`](Self::trace)
    /// (0 outside the preemptive scheduler). Lets a trace be split by thread - e.g. to
    /// see what the main thread did versus a worker.
    pub trace_thid: Vec<i32>,
    /// Bytes the guest wrote to the debug console (sceClibPrintf and friends), in
    /// order. This is the blob-free "it printed" signal for the hello corpus: the
    /// Vita has no framebuffer console for these, so the host is the sink. Also fed
    /// by sceIoWrite to fd 1 (the path newlib's stdout takes).
    pub stdout: Vec<u8>,
    /// Bytes written to fd 2 (stderr) via sceIoWrite.
    pub stderr: Vec<u8>,
    /// Diagnostic sample of the first few `sceKernelWaitLwCond` calls:
    /// `(cond work addr, timeout pointer, timeout value or 0)`. Lets the probe tell
    /// a producer/consumer wait (no timeout) from a timed delay loop (finite
    /// timeout) without a full trace. Capped so it costs nothing after warm-up.
    pub lwcond_wait_samples: Vec<(u32, u32, u32)>,
    /// The game->OS egress ledger: game-authored, human-readable events (savedata
    /// writes, trophies, score submissions) in occurrence order, each frame-tagged.
    /// This is the content-free conformance-assertion surface (see [`EgressEvent`]).
    pub egress: Vec<EgressEvent>,
    /// How many completed scenes to keep. `None` (the default) keeps every scene,
    /// which is what a short capture-and-inspect run wants.
    ///
    /// A LONG run must set it. Each scene holds a snapshot of every draw's vertex
    /// window and indices - on a real 3D title, hundreds of draws and megabytes per
    /// frame - so retaining them all costs gigabytes within a couple of thousand
    /// frames and eventually ends the run. That is a hard ceiling on how far into a
    /// game anything can get, which makes it a ceiling on playing the game at all.
    /// See [`Capture::push_scene`]: eviction folds the dropped scene into
    /// [`retired_digest`](Self::retired_digest), so bounding retention does NOT change
    /// the determinism signature by one bit.
    pub scene_limit: Option<usize>,
    /// The running signature fold over scenes already evicted by `scene_limit`.
    pub retired_digest: u64,
    /// How many scenes have been evicted (so `scenes.len() + retired_scenes` is the
    /// true count for a report).
    pub retired_scenes: u64,
    /// Scenes pushed since the last display flip - the frame under construction.
    frame_scenes: usize,
    /// Scenes the PREVIOUS display frame was made of, latched at the flip.
    ///
    /// A title is not one scene per frame. A racing title's race frame is fifteen: six
    /// small reflection/shadow passes, a 720x408 world pass, a post pass over it, and a
    /// composite that blits the world and draws the HUD. Every observer that wants "the
    /// scene" wants a specific one of those, and which one is a question about CONTENT,
    /// not about order - so [`Capture::frame_scenes`] hands out the whole frame and
    /// [`Capture::world_scene`] picks by what the scene contains.
    prev_frame_scenes: usize,
}

/// Upper bound on retained trace entries. When the trace reaches this, the oldest
/// half is dropped, so memory stays bounded (~32 MB at the cap) while every
/// consumer keeps at least [`TRACE_CAP`]/2 of recent history.
pub const TRACE_CAP: usize = 4 << 20;

/// The FNV-1a offset basis, the seed of the determinism signature fold.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold `bytes` into an FNV-1a accumulator.
fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(FNV_PRIME);
    }
}

/// How many independent FNV chains [`fnv_bulk`] runs over a blob. Each lane is an
/// ordinary FNV-1a over a strided subsequence, so they are independent and the CPU
/// pipelines them; one chain is bound by the 64-bit multiply's latency, not its
/// throughput, and spends most of its cycles waiting.
const FNV_LANES: usize = 8;

/// Fold a LARGE byte blob into the accumulator.
///
/// Same construction as [`fnv`] - xor-then-multiply, defined purely on the byte
/// sequence - but run as [`FNV_LANES`] interleaved chains that are combined at the
/// end, because these blobs are the scene's vertex and index buffers: megabytes per
/// frame, and the single serial chain was the whole cost of `sceGxmEndScene`.
///
/// Still exactly as deterministic and as engine-independent as the serial version:
/// the result is a pure function of the byte sequence, with no endianness, word size
/// or allocation dependence. It is NOT the same VALUE as the serial fold, so
/// signatures do not compare across this change - they never had to, since a
/// signature is only ever compared between runs of the same build (`explore`'s
/// bucketing, a recipe's `@sig`).
fn fnv_bulk(h: &mut u64, bytes: &[u8]) {
    // Length first, so appending zero bytes cannot leave the digest unchanged and two
    // differently-split blobs cannot collide by concatenation.
    fnv(h, &(bytes.len() as u64).to_le_bytes());
    let mut lanes = [*h; FNV_LANES];
    // Lane `i` takes bytes i, i+LANES, i+2*LANES, ... - a fixed, size-independent
    // assignment, so the same blob always distributes the same way.
    let chunks = bytes.chunks_exact(FNV_LANES);
    let tail = chunks.remainder();
    for c in chunks {
        for (lane, &b) in lanes.iter_mut().zip(c) {
            *lane ^= b as u64;
            *lane = lane.wrapping_mul(FNV_PRIME);
        }
    }
    for (lane, &b) in lanes.iter_mut().zip(tail) {
        *lane ^= b as u64;
        *lane = lane.wrapping_mul(FNV_PRIME);
    }
    // Combine in lane order, through the same primitive, so the lanes' contributions
    // stay ordered and one lane's change cannot be cancelled by another's.
    for lane in lanes {
        fnv(h, &lane.to_le_bytes());
    }
}

/// Fold one scene's observable content into the signature accumulator.
fn fold_scene(h: &mut u64, s: &Scene) {
    if let Some(c) = &s.color {
        fnv(h, &c.data_addr.to_le_bytes());
        fnv(h, &c.format.to_le_bytes());
    }
    for d in &s.draws {
        fnv_bulk(h, &d.vertices);
        fnv_bulk(h, &d.indices);
        // One pass over the uniform floats as bytes, rather than a call per float.
        // `to_le_bytes` per element is what makes this a copy rather than a cast.
        let mut buf = Vec::with_capacity(d.uniforms.len() * 4);
        for u in &d.uniforms {
            buf.extend_from_slice(&u.to_le_bytes());
        }
        fnv_bulk(h, &buf);
    }
}

impl Capture {
    pub fn new() -> Self {
        Capture { retired_digest: FNV_OFFSET, ..Capture::default() }
    }

    /// Record a completed scene, evicting the oldest if `scene_limit` is set.
    ///
    /// Eviction folds the dropped scene into [`retired_digest`](Self::retired_digest)
    /// FIRST, so [`signature`](Self::signature) still covers the whole run. Bounding
    /// memory must not quietly change what a run reports about itself: a signature
    /// that depended on the retention window would compare unequal between a short
    /// run and a long one and destroy the only cross-engine equivalence check there
    /// is.
    pub fn push_scene(&mut self, scene: Scene) {
        self.scenes.push(scene);
        self.frame_scenes += 1;
        let Some(limit) = self.scene_limit else { return };
        let limit = limit.max(1);
        while self.scenes.len() > limit {
            let old = self.scenes.remove(0);
            // A `Capture` built by `Default` starts at 0, not the FNV basis; seed it
            // on first use so both construction paths fold identically.
            if self.retired_scenes == 0 && self.retired_digest == 0 {
                self.retired_digest = FNV_OFFSET;
            }
            let mut h = self.retired_digest;
            crate::perf::time(crate::perf::Phase::SceneFold, || fold_scene(&mut h, &old));
            self.retired_digest = h;
            self.retired_scenes += 1;
        }
    }

    /// The determinism signature over the run's whole observable output: every
    /// scene's render stream (evicted ones via `retired_digest`) then the egress
    /// ledger. Engine-independent by construction - it covers what the guest
    /// PRODUCED, never internal RAM or thread timing - so native, headless and
    /// browser runs of the same recipe must agree on it.
    pub fn signature(&self) -> u64 {
        let mut h = if self.retired_scenes == 0 && self.retired_digest == 0 {
            FNV_OFFSET
        } else {
            self.retired_digest
        };
        for s in &self.scenes {
            fold_scene(&mut h, s);
        }
        for ev in &self.egress {
            fnv(&mut h, &ev.frame.to_le_bytes());
            fnv(&mut h, format!("{:?}", ev.kind).as_bytes());
        }
        h
    }

    /// Latch the scene count of the frame that just finished. Called once per display
    /// flip, so [`frame_scenes`](Self::frame_scenes) can hand back exactly the scenes
    /// the last completed frame was built from.
    pub fn end_frame(&mut self) {
        self.prev_frame_scenes = self.frame_scenes;
        self.frame_scenes = 0;
    }

    /// The scenes of the most recently COMPLETED display frame, oldest first.
    ///
    /// Bounded by what `scene_limit` actually retained: a run that keeps one scene gets
    /// one back, which is the old behaviour and still correct for a single-pass title.
    /// Between flips (a partially built frame) this reports the previous frame's tail
    /// rather than a mixture, because an observer asking about "this frame" during
    /// construction has no complete frame to be given.
    pub fn frame_scenes(&self) -> &[Scene] {
        let n = self.prev_frame_scenes.max(1).min(self.scenes.len());
        &self.scenes[self.scenes.len() - n..]
    }

    /// The scene of the last completed frame that holds the PLAYER'S VIEW - the pass an
    /// observer means when it asks where things are or which way the vehicle points.
    ///
    /// Never by order: the last scene of a multi-pass frame is the composite, a handful of
    /// full-screen quads and a HUD, which is why a racing title reported nothing at all
    /// from `locate` while rendering a whole racetrack.
    ///
    /// # Why triangle count is the WRONG selector, measured
    /// It was the selector, and it silently chose a different pass from frame to frame.
    /// A race frame carries a rear-view MIRROR pass, and a mirror draws the same world
    /// through a camera pointing the other way - so its triangle count sits within a few
    /// percent of the main view's and crosses over depending on what happens to be behind
    /// the car. Every crossover flipped the reported heading by 180 degrees. That is
    /// indistinguishable from a car that has spun, and it was read as one: a controller
    /// steering on it fought an imaginary spin and abandoned the lap, three times, at
    /// three different corners. The position stayed smooth throughout, which is the
    /// contradiction that gives the bug away - a car cannot swap ends and hold a straight
    /// line.
    ///
    /// # The rule
    /// Among scenes with a RECOVERABLE CAMERA, the largest render target wins (ties on
    /// world triangles). The player's view is by construction the biggest thing drawn:
    /// a mirror, a reflection and an environment-probe face are all smaller. Requiring a
    /// camera is what excludes the rest for free - a shadow pass is drawn through an
    /// ORTHOGRAPHIC matrix, which is affine, so it has no eye to recover
    /// ([`scene_eye`](crate::render::scene_eye) returns `None`), and the composite has no
    /// world-to-clip matrix at all.
    ///
    /// A title whose frame is one scene selects that scene, unchanged, and a frame with no
    /// camera anywhere falls back to the old most-world-geometry reading.
    pub fn world_scene(&self) -> Option<&Scene> {
        let frame = self.frame_scenes();
        frame
            .iter()
            .filter(|s| crate::render::scene_eye(s).is_some())
            .max_by_key(|s| {
                let area = s.color.as_ref().map_or(0u64, |c| c.width as u64 * c.height as u64);
                (area, s.world_triangles())
            })
            .or_else(|| {
                frame
                    .iter()
                    .filter(|s| s.world_triangles() > 0)
                    .max_by_key(|s| s.world_triangles())
            })
            .or_else(|| frame.last())
    }

    /// Scenes the run has completed, including any evicted by `scene_limit`.
    pub fn total_scenes(&self) -> u64 {
        self.retired_scenes + self.scenes.len() as u64
    }

    /// Record one serviced call in the bounded debug trace (hot path: a push, plus
    /// an amortized front-drain every [`TRACE_CAP`]/2 calls once the cap is hit).
    #[inline]
    pub fn record_call(&mut self, func_nid: u32, thid: i32) {
        self.call_count += 1;
        if self.trace.len() >= TRACE_CAP {
            self.trace.drain(..TRACE_CAP / 2);
            self.trace_thid.drain(..TRACE_CAP / 2);
        }
        self.trace.push(func_nid);
        self.trace_thid.push(thid);
    }

    /// Note an unimplemented call once (deduplicated by NID pair).
    pub fn note_unimplemented(&mut self, library_nid: u32, func_nid: u32, name: &str) {
        if !self.unimplemented.iter().any(|(l, f, _)| *l == library_nid && *f == func_nid) {
            self.unimplemented.push((library_nid, func_nid, name.to_string()));
        }
    }
}

#[cfg(test)]
mod fold_tests {
    use super::*;

    fn digest(bytes: &[u8]) -> u64 {
        let mut h = FNV_OFFSET;
        fnv_bulk(&mut h, bytes);
        h
    }

    /// Flipping ANY single bit of the blob changes the digest. The lanes are what make
    /// this worth asserting: a byte only ever reaches one lane, so a lane that was
    /// dropped or aliased would silently stop covering a whole eighth of every buffer -
    /// and a signature blind to an eighth of the vertex data still looks like a working
    /// signature.
    #[test]
    fn every_byte_is_covered() {
        // Longer than one lane group and NOT a multiple of it, so the tail path is
        // exercised too.
        let base: Vec<u8> = (0..67u8).collect();
        let want = digest(&base);
        for i in 0..base.len() {
            for bit in 0..8 {
                let mut v = base.clone();
                v[i] ^= 1 << bit;
                assert_ne!(digest(&v), want, "byte {i} bit {bit} did not affect the digest");
            }
        }
    }

    /// Order matters: swapping two bytes that land in DIFFERENT lanes, and two that
    /// land in the SAME lane, must both change the digest. A per-lane sum would pass
    /// the first and fail the second.
    #[test]
    fn order_matters_within_and_across_lanes() {
        let base: Vec<u8> = (0..64u8).collect();
        let want = digest(&base);
        let mut across = base.clone();
        across.swap(0, 1);
        assert_ne!(digest(&across), want, "swap across lanes");
        let mut within = base.clone();
        within.swap(0, FNV_LANES);
        assert_ne!(digest(&within), want, "swap within one lane");
    }

    /// Length is folded in, so appending zeros - which leave every lane's xor-multiply
    /// chain looking plausible - cannot leave the digest unchanged, and two blobs that
    /// concatenate to the same bytes do not collide.
    #[test]
    fn length_is_part_of_the_digest() {
        assert_ne!(digest(&[1, 2, 3]), digest(&[1, 2, 3, 0]));
        let mut a = FNV_OFFSET;
        fnv_bulk(&mut a, &[1, 2]);
        fnv_bulk(&mut a, &[3]);
        let mut b = FNV_OFFSET;
        fnv_bulk(&mut b, &[1]);
        fnv_bulk(&mut b, &[2, 3]);
        assert_ne!(a, b, "the split between two folds must be visible");
    }

    /// An empty blob is still folded (its length), so a draw with no indices is
    /// distinguishable from one with none of that field at all.
    #[test]
    fn empty_is_folded() {
        assert_ne!(digest(&[]), FNV_OFFSET);
    }
}

#[cfg(test)]
mod retention_tests {
    use super::*;

    /// A scene distinguishable by `tag` through the part of it the signature folds.
    fn scene(tag: u8) -> Scene {
        Scene {
            color: Some(ColorSurface {
                format: tag as u32,
                surface_type: 0,
                width: 960,
                height: 544,
                stride_pixels: 960,
                data_addr: 0x8000_0000 + tag as u32,
            }),
            draws: Vec::new(),
        }
    }

    /// Bounding memory must not change what the run reports about itself. If the
    /// signature moved with the retention window, a long run and a short one would
    /// compare unequal and the cross-engine equivalence check - the only one there
    /// is - would be worthless.
    #[test]
    fn the_signature_is_invariant_under_scene_retention() {
        let mut unbounded = Capture::new();
        let mut bounded = Capture::new();
        bounded.scene_limit = Some(3);
        for i in 0..40u8 {
            unbounded.push_scene(scene(i));
            bounded.push_scene(scene(i));
        }
        for c in [&mut unbounded, &mut bounded] {
            c.egress.push(EgressEvent { frame: 7, kind: EgressKind::Trophy { id: 3 } });
        }
        assert_eq!(unbounded.scenes.len(), 40);
        assert_eq!(bounded.scenes.len(), 3, "retention did not bound the scene list");
        assert_eq!(bounded.total_scenes(), 40, "evicted scenes must still be counted");
        assert_eq!(
            unbounded.signature(),
            bounded.signature(),
            "the determinism signature changed when scenes were evicted"
        );
    }

    /// A `Default`-constructed capture folds identically to a `new` one (the seed is
    /// applied lazily), so nothing depends on which constructor a host happened to use.
    #[test]
    fn default_and_new_captures_agree() {
        let mut a = Capture::new();
        let mut b = Capture::default();
        b.scene_limit = Some(1);
        for i in 0..5u8 {
            a.push_scene(scene(i));
            b.push_scene(scene(i));
        }
        assert_eq!(a.signature(), b.signature());
    }
}
