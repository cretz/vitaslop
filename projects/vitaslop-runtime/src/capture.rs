//! The GXM command-stream capture: the blob-free "it works" signal. The host
//! records what the guest asked the GPU to do (surfaces, programs, per-draw
//! vertex/index/uniform snapshots) without emulating a GPU or drawing a pixel.
//! A software rasterizer or wgpu backend later consumes this to produce frames.

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
    pub data_addr: u32,
    pub pixels: Vec<u8>,
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
    pub textures: Vec<BoundTexture>,
    /// The fixed-function pipeline state (cull/depth/stencil/viewport/...) in effect
    /// for this draw, snapshotted from the sticky GXM context state. See [`RenderState`].
    pub render_state: RenderState,
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
}

/// Upper bound on retained trace entries. When the trace reaches this, the oldest
/// half is dropped, so memory stays bounded (~32 MB at the cap) while every
/// consumer keeps at least [`TRACE_CAP`]/2 of recent history.
pub const TRACE_CAP: usize = 4 << 20;

impl Capture {
    pub fn new() -> Self {
        Capture::default()
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
