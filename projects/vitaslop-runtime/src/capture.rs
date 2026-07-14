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
    pub width: u32,
    pub height: u32,
    pub stride_pixels: u32,
    pub data_addr: u32,
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
    /// Ordered trace of every serviced call's function NID, for debugging.
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
}

impl Capture {
    pub fn new() -> Self {
        Capture::default()
    }

    /// Note an unimplemented call once (deduplicated by NID pair).
    pub fn note_unimplemented(&mut self, library_nid: u32, func_nid: u32, name: &str) {
        if !self.unimplemented.iter().any(|(l, f, _)| *l == library_nid && *f == func_nid) {
            self.unimplemented.push((library_nid, func_nid, name.to_string()));
        }
    }
}
