//! The shared cube render pipeline. The WGSL shader, vertex layout, and per-draw
//! encoding live here once so the native headless renderer (`vitaslop-native`)
//! and the browser WebGPU canvas (`vitaslop-web`) draw identically - the native
//! path stays the pixel oracle for the browser. Only the surrounding host bits
//! differ: native acquires an adapter headlessly and reads pixels back, the
//! browser acquires a device from `navigator.gpu` and presents to a canvas.
//!
//! [`DrawBatch`] is the neutral, GPU-free seam: the runtime turns its captured
//! GXM scene into a list of these, and [`CubeRenderer`] (behind the `gpu`
//! feature) draws them. Keeping `DrawBatch` free of any wgpu types lets the
//! engine-agnostic runtime produce it without pulling in a GPU stack.

/// A renderer diagnostic, at `debug` on the `vitaslop::gxm` target.
///
/// # Why these are not `eprintln!`
/// This module is the SHARED render path: the same code draws the native pixel oracle
/// and the browser canvas. On wasm32 stderr is a no-op sink, so every `eprintln!` here
/// was discarded in the browser - including the recompiler's own per-scene verdict, the
/// one line that says whether a frame's draws were really recompiled or quietly fell
/// back. The engine that is hardest to inspect was the one reporting nothing, and a
/// black frame there was indistinguishable from a correct one.
///
/// Read them with `RUST_LOG=vitaslop::gxm=debug` natively, or
/// `VITASLOP_LOG=warn,vitaslop::gxm=debug` in the browser. Anything that means the
/// renderer DEGRADED - a fallback, a dropped draw, an approximated state - goes through
/// [`report_warn`] instead, so it survives the default `warn` filter on both engines
/// without anyone having to know to ask for it.
macro_rules! report {
    ($($arg:tt)*) => { tracing::debug!(target: "vitaslop::gxm", $($arg)*) };
}

/// A renderer diagnostic that means the output is WRONG or approximated. Always visible
/// at the default filter - see [`report`].
macro_rules! report_warn {
    ($($arg:tt)*) => { tracing::warn!(target: "vitaslop::gxm", $($arg)*) };
}

/// One ready-to-draw batch: a model-view-projection matrix and the raw vertex and
/// index bytes, already snapshotted from guest memory. The vertex layout is the
/// cube's fixed-function format (see [`CUBE_SHADER`]): position `float32x3` at
/// offset 0, color `unorm8x4` at offset 12, stride 16.
#[derive(Clone, Debug, PartialEq)]
pub struct DrawBatch {
    /// Column-major 4x4 MVP matrix (the guest's vertex default uniform buffer).
    pub mvp: [f32; 16],
    /// Interleaved vertex bytes, stride 16.
    pub vertices: Vec<u8>,
    /// Index buffer bytes.
    pub indices: Vec<u8>,
    /// Number of indices to draw.
    pub index_count: u32,
    /// True if indices are 32-bit; false for 16-bit (the cube uses 16-bit).
    pub index_u32: bool,
}

/// The fixed-function cube shader, the blob-free equivalent of the guest's
/// placeholder GXP shaders. It transforms each vertex by the captured MVP and
/// interpolates per-vertex color. The `clip.z` remap converts GL-style clip space
/// (z in [-1, 1], which the guest's projection targets) to WebGPU clip space
/// (z in [0, 1]).
pub const CUBE_SHADER: &str = r#"
struct Uniforms { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> VsOut {
    var out: VsOut;
    var clip = u.mvp * vec4<f32>(position, 1.0);
    clip.z = (clip.z + clip.w) * 0.5;
    out.pos = clip;
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

// ---------------------------------------------------------------------------
// General GXM renderer seam
// ---------------------------------------------------------------------------
//
// [`CubeRenderer`] above is the fixed-function 3D-cube path (one shader, one
// vertex layout, opaque depth-tested). A real 2D title needs more: per-draw
// coordinate spaces, textured sprites, and alpha blending in submission order.
// [`RenderScene`] is the neutral, GPU-free seam for that - the runtime decodes a
// captured GXM scene into it (reusing the exact interpretation its software
// rasterizer uses, so the two agree), and [`GxmRenderer`] (behind `gpu`) draws
// it. Keeping the decode in the runtime and only the draw here means the GPU
// renderer never touches guest memory or a GXM enum - it sees canonical vertices
// and a linear RGBA8 texture, and stays the faithful GPU twin of the software
// oracle.

/// The coordinate space a general draw's vertex positions live in, recovered by
/// the runtime from the draw's vertex layout - the same [`Space`] the software
/// rasterizer uses. Determines both the vertex transform and whether the draw is
/// depth-tested (only `Mvp` is; the 2D spaces paint in submission order).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DrawSpace {
    /// Object space transformed by a captured column-major 4x4 MVP uniform (the 3D
    /// path): perspective divide, GL-style clip z, depth-tested opaque.
    Mvp([f32; 16]),
    /// Clip coordinates emitted directly, in [-1, 1] (a fullscreen pass).
    Ndc,
    /// Screen pixels in [0, surface] with Y down (2D sprite quads whose vertex
    /// program baked the pixel-to-clip transform).
    Pixel,
}

/// A decoded, GPU-ready texture: a tightly-packed linear `Rgba8Unorm` image plus a
/// content/identity `key` the renderer caches uploads by, so an unchanged texture
/// (the common case - a font atlas or background reused every frame) is uploaded
/// once and then only bound. `rgba` is shared (`Arc`) so re-handing the same
/// texture to the renderer each frame is a pointer copy, not a pixel copy.
#[derive(Clone, Debug)]
pub struct GxmTexture {
    pub key: u64,
    /// Guest address of the texture's pixel data. Identity, not content: a title that
    /// renders a pass into an offscreen buffer and samples it in a later pass binds a
    /// texture whose data pointer IS that pass's colour surface, and the guest bytes
    /// behind it are stale (the GPU wrote the real pixels, not the guest). The renderer
    /// matches this against the render targets it has drawn this frame and binds the
    /// rendered texture instead - see `GxmRenderer::encode_chain`.
    pub data_addr: u32,
    pub width: u32,
    pub height: u32,
    /// Number of `width x height` RGBA8 images `rgba` holds back to back: 1 normally, 6 for a
    /// cube map (in +X, -X, +Y, -Y, +Z, -Z order, which is WebGPU's array-layer order).
    pub faces: u32,
    pub rgba: std::sync::Arc<Vec<u8>>,
    /// The `SceGxmTextureFormat` base format and channel swizzle `rgba` was decoded THROUGH.
    /// Diagnostic only - the pixels are already RGBA8 by the time the renderer sees them - but
    /// it is the one fact that separates "this channel is genuinely zero in the asset" from "we
    /// decoded the asset through the wrong layout", and those need opposite fixes.
    pub base_format: u32,
    pub swizzle: u32,
    /// True if the guest set this texture's magnification filter to LINEAR
    /// (`SceGxmTextureFilter` == 1); the renderer then bilinear-samples it, matching
    /// the software rasterizer's `sample_texture_bilinear`. False = POINT/nearest.
    pub filter_linear: bool,
    /// The guest's `SceGxmTextureAddrMode` for U and V
    /// (`sceGxmTextureSet{U,V}AddrMode`, 0 = REPEAT). The recompiled path used to hardcode
    /// REPEAT for every sampler, which is invisible while every coordinate stays inside
    /// [0,1] and catastrophic the moment one does not: a full-screen pass reading one texel
    /// past the edge wraps to the OPPOSITE edge instead of clamping, and a title's composite
    /// showed the world tiled diagonally across the display rather than clipped.
    pub addr_mode_u: u32,
    pub addr_mode_v: u32,
    /// The guest set `sceGxmTextureSetGammaMode` on this texture, so the hardware sampler
    /// sRGB-DECODES every texel it fetches. Uploading through an sRGB texture format does the
    /// same decode in the same place, before filtering.
    ///
    /// The counterpart of [`RttTarget::gamma`], which is the write half of the same state.
    pub gamma: bool,
}

/// The per-material forward-lighting inputs the GPU `fs_opaque` needs to reproduce the
/// software rasterizer's `shade_lit`: a base-colour `tint`, one directional light
/// (`light_dir` world-space direction-to-light + `light_col`), and a flat `ambient`. The
/// runtime reflects these from the fragment program; neutral values (tint 1, light col 1,
/// modest ambient) make it a pass-through for a 2D/UI draw. Kept engine-neutral here so
/// `vitaslop-platform` stays free of any runtime dependency.
#[derive(Clone, Copy, Debug)]
pub struct GxmMaterial {
    pub tint: [f32; 3],
    pub light_dir: [f32; 3],
    pub light_col: [f32; 3],
    pub ambient: [f32; 3],
}

impl Default for GxmMaterial {
    fn default() -> Self {
        GxmMaterial {
            tint: [1.0, 1.0, 1.0],
            light_dir: [0.0, 1.0, 0.0],
            light_col: [1.0, 1.0, 1.0],
            ambient: [0.35, 0.35, 0.35],
        }
    }
}

/// One ready-to-draw call, CPU-decoded by the runtime into a canonical vertex
/// layout so the GPU never decodes a GXM attribute format. The vertex layout is
/// fixed: position `float32x3` at offset 0, texcoord `float32x2` at offset 12
/// (already divided by the draw's uv scale), color `unorm8x4` at offset 20, and the
/// world-space normal `float32x3` at offset 24 (baked by the runtime from the object
/// normal and the model-to-world matrix, for the opaque lighting term), stride 36.
/// Indices are always 32-bit. `texture` is `None` for an untextured draw (the renderer
/// binds a 1x1 white texel so one shader path serves both).
#[derive(Clone, Debug)]
pub struct GxmDraw {
    pub space: DrawSpace,
    pub vertices: Vec<u8>,
    pub indices: Vec<u8>,
    pub index_count: u32,
    pub texture: Option<GxmTexture>,
    /// The per-material lighting inputs for the opaque path (ignored when `opaque` is false).
    pub material: GxmMaterial,
    /// True for genuinely opaque 3D geometry: an MVP-space draw that also writes depth
    /// (`front_depth_write != SCE_GXM_DEPTH_WRITE_DISABLED`). Drives BOTH the pipeline
    /// (depth-tested opaque replace) AND the fragment combine (the albedo texel is taken
    /// straight, tone-mapped by `exposure`, alpha forced to 1). This is the software
    /// rasterizer's `depth_test` decision, carried here rather than re-derived from
    /// `space` alone - an MVP draw with depth writes DISABLED is a 2D alpha-blended
    /// overlay, not opaque, so keying opacity off MVP-space would wrongly replace and
    /// tonemap it. False = 2D overlay: `vertex_color * texel` modulate, alpha-blended in
    /// submission order.
    pub opaque: bool,
    /// Scene exposure (linear multiplier from `vsCoarseExposureReg`) applied to opaque
    /// draws before a Reinhard tonemap. 1.0 is a no-op (2D/UI and any shader with no
    /// exposure uniform), so the tonemap is skipped and the color passes through.
    pub exposure: f32,
    /// The guest's real vertex+fragment shaders + their draw inputs, for the GXP->WGSL
    /// recompiler (live guest-shader) path. `Some` only when the runtime captured it
    /// (`VITASLOP_GXP_LIVE`); the renderer links + caches a pipeline and ALWAYS falls back
    /// to the fixed-function fields above on any link/format error. See [`GxpRecompile`].
    pub gxp: Option<GxpRecompile>,
    /// This draw has NO valid fixed-function representation: its geometry carries neither a
    /// texcoord nor a per-vertex colour, so the fixed-function fields above have no colour
    /// source and would paint opaque white. Its real colour lives in the guest's fragment
    /// shader (typically a uniform), so it can only be drawn through `gxp`.
    ///
    /// The scene builder used to DISCARD such a draw outright, silently. That is what made a
    /// retail title's solid-colour UI fills - a button's interior, a panel backing - simply
    /// absent, indistinguishable from the guest never drawing them, and it survived being
    /// stared at in screenshots. Keeping the draw and marking it lets the recompiler render it
    /// exactly; if no recompiled pipeline is available the renderer skips it and SAYS SO,
    /// rather than either dropping it quietly or painting a white rectangle.
    pub shader_only: bool,
}

/// Report, once per run, that a draw with no fixed-function representation could not be drawn
/// by the recompiler either - so it is missing from the frame. Suppressed after the first so a
/// per-frame occurrence cannot flood the log; the first one carries what matters.
pub(crate) fn report_shader_only_skip(d: &GxmDraw, gxp_enabled: bool) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    report!(
        "render: a draw with NO fixed-function representation (position-only geometry, its \
         colour lives in the guest fragment shader) is MISSING from the frame - {}. \
         {} indices, {} vertex bytes.",
        if !gxp_enabled {
            "the GXP recompiler is not enabled (set VITASLOP_GXP_LIVE)"
        } else if d.gxp.is_none() {
            "the runtime captured no shader payload for it"
        } else {
            "its shader pair did not link (the fallback above names the reason)"
        },
        d.index_count,
        d.vertices.len(),
    );
}

/// Everything the GXP->WGSL recompiler needs to draw one call with the guest's real
/// vertex+fragment shaders, snapshotted by the runtime and carried as plain data (no GPU or
/// runtime dependency). The renderer links the pair on first sight, caches the pipeline by
/// shader identity, and ALWAYS falls back to the fixed-function path on any link/format
/// error - a wrong translation never paints a pixel.
#[derive(Clone, Debug)]
pub struct GxpRecompile {
    /// The vertex `SceGxmProgram` container bytes.
    pub vprog: Vec<u8>,
    /// The fragment `SceGxmProgram` container bytes.
    pub fprog: Vec<u8>,
    /// Raw vertex default-uniform-buffer (SA bank) bytes, as the guest wrote them.
    pub vert_sa: Vec<u8>,
    /// Raw fragment default-uniform-buffer (SA bank) bytes, as the guest wrote them.
    pub frag_sa: Vec<u8>,
    /// Raw guest vertex stream bytes (stream 0) exactly as bound.
    pub vertices: Vec<u8>,
    /// Byte stride of one guest vertex within `vertices`.
    pub vertex_stride: u32,
    /// Guest vertex attributes: stream byte offset + raw GXM format + component count, keyed
    /// to the recompiler's vertex-input `@location` by `reg_index` (the attribute base lane).
    pub attributes: Vec<GxpAttr>,
    /// Raw guest index bytes.
    pub indices: Vec<u8>,
    /// Number of indices.
    pub index_count: u32,
    /// True = 32-bit indices, false = 16-bit (GXM index format 0).
    pub index_u32: bool,
    /// GXM primitive type word (drives the pipeline topology).
    pub primitive: u32,
    /// Decoded textures bound per fragment sampler unit.
    pub textures: Vec<GxpTex>,
    /// Decoded textures bound per VERTEX sampler unit. Separate list, because the two stages
    /// number their units independently - and a vertex program that samples is building its
    /// geometry from what it reads, so binding the fragment's texture here draws a wrong mesh
    /// rather than shading a surface wrongly.
    pub vertex_textures: Vec<GxpTex>,
    /// Depth write enabled for this draw (GXM `front_depth_write != DISABLED`).
    pub depth_write: bool,
    /// GXM depth-compare function word (`SceGxmDepthFunc`).
    pub depth_func: u32,
    /// GXM cull-mode word (`SceGxmCullMode`).
    pub cull_mode: u32,
    /// Whether this draw is alpha-blended (a 2D/overlay draw, not opaque geometry).
    ///
    /// A HEURISTIC read off the geometry, kept only for the fixed-function path. The
    /// recompiler path uses `blend_state`, which is the guest's own answer.
    pub blend: bool,
    /// The blend equation the guest baked into this draw's fragment program, as raw GXM enum
    /// values: `[color_mask, color_func, alpha_func, color_src, color_dst, alpha_src,
    /// alpha_dst]`.
    ///
    /// Carried as a plain array rather than the capture type so this crate stays independent
    /// of the capture crate, exactly as `depth_func` and `cull_mode` are carried.
    pub blend_state: [u8; 7],
    /// GXM viewport `[xOffset,xScale,yOffset,yScale,zOffset,zScale]` mapping the guest clip
    /// output to the framebuffer. All-zero means the guest left the default (fullscreen).
    pub viewport: [f32; 6],
}

/// One guest vertex attribute for the recompiler path: where it sits in the stream and its
/// raw GXM format, so the pipeline builds a matching `wgpu` vertex layout.
#[derive(Clone, Copy, Debug)]
pub struct GxpAttr {
    /// The vertex program's attribute resource index = the recompiler's `@location` base lane.
    pub reg_index: u16,
    /// Byte offset of this attribute within a vertex.
    pub offset: u16,
    /// Raw `SceGxmAttributeFormat` value.
    pub gxm_format: u8,
    /// Component count (1..4).
    pub components: u8,
}

/// A decoded texture bound to a specific fragment sampler unit for the recompiler path.
#[derive(Clone, Debug)]
pub struct GxpTex {
    /// The sampler unit the fragment program samples this from (`t{unit}`/`s{unit}`).
    pub unit: u8,
    /// The decoded, GPU-ready texture.
    pub tex: GxmTexture,
}

/// Stride of the canonical [`GxmDraw`] vertex: pos(12) + uv(8) + color(4) + world-normal(12).
pub const GXM_VERTEX_STRIDE: u64 = 36;

/// Where a scene's pixels land: the guest address of its colour surface and that
/// surface's size. A frame is usually several scenes - offscreen passes that render
/// shadow maps, reflections, the 3D world - followed by one that composites them onto
/// the display buffer, and the composite samples the earlier ones as TEXTURES. Matching
/// a sampled texture's data pointer against these is what lets the renderer bind what it
/// actually drew instead of the guest bytes, which the GPU never wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RttTarget {
    pub data_addr: u32,
    pub width: u32,
    pub height: u32,
    /// The guest asked for GAMMA-CORRECT writes on this surface
    /// (`sceGxmColorSurfaceSetGammaMode`), so the hardware sRGB-encodes every value the ROP
    /// stores. Rendering through an sRGB view of the same texture reproduces that exactly -
    /// including doing it AFTER blending, which is where the hardware does it too.
    pub gamma: bool,
}

/// A whole scene reduced to general draws, in submission order. The runtime builds
/// it from a captured [`Scene`](vitaslop_runtime-side); [`GxmRenderer`] draws it.
#[derive(Clone, Debug, Default)]
pub struct RenderScene {
    pub draws: Vec<GxmDraw>,
    /// This scene's render target, when the guest's colour surface was resolvable.
    pub target: Option<RttTarget>,
    /// Linear depth-normalization for the opaque (Mvp) draws: the shader maps each
    /// fragment's post-divide depth `d = c.z/c.w` to `clamp((d - depth_min) * depth_scale,
    /// 0, 1)` before the depth test. The guest's captured matrices produce `d` in an
    /// arbitrary, often huge range (hundreds to thousands, not a normalized [0,1]); the
    /// builder scans the visible opaque geometry and picks `depth_min`/`depth_scale` so the
    /// whole range maps linearly across the [0,1] depth buffer at full f32 precision. This
    /// preserves the software oracle's exact depth ORDERING (a monotonic map) while keeping
    /// enough resolution to separate, e.g., a vehicle from the ground it sits on - which a
    /// saturating squash loses when both sit at a large depth. `depth_scale == 0` means no
    /// opaque geometry (or a single depth plane): every opaque fragment maps to 0.
    pub depth_min: f32,
    pub depth_scale: f32,
    /// Where the guest's `SceGxmDepthStencilSurface` for this scene puts its depth samples,
    /// or 0 when the scene had none.
    ///
    /// A later pass that reads this scene's depth - a soft-particle fade, fog, SSAO - binds a
    /// texture at exactly this address. It is the ONLY thing that tells such a sample apart
    /// from one of the colour target, because a title allocates the two next to each other:
    /// on one retail racer the world's colour is at `0x89204aa0` and its depth 256 bytes later
    /// at `0x89204ba0`, so an address-RANGE match against the colour target claims it first
    /// and the pass reads a colour where it wanted a distance.
    pub depth_addr: u32,
}

#[cfg(feature = "gpu")]
pub use render::{CubeRenderer, DEPTH_FORMAT};

#[cfg(feature = "gpu")]
pub use gxm::{EncodePhases, GxmRenderer};

#[cfg(feature = "gpu")]
mod render {
    use super::{DrawBatch, CUBE_SHADER};
    use wgpu::util::DeviceExt;

    /// The depth format the cube pipeline renders with. A host that supplies its
    /// own depth attachment must use this format.
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    /// The cube render pipeline, built once against a device for a given color
    /// target format. Reused across frames and scenes. Host-agnostic: it records
    /// draws into a caller-provided render pass, so the same renderer serves the
    /// native headless texture target and the browser canvas surface.
    pub struct CubeRenderer {
        pipeline: wgpu::RenderPipeline,
        bind_layout: wgpu::BindGroupLayout,
    }

    /// GPU resources for one batch, kept alive past the render pass that uses them.
    struct Prepared {
        vertices: wgpu::Buffer,
        indices: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
        index_count: u32,
        index_u32: bool,
    }

    impl CubeRenderer {
        /// Build the pipeline for a `color_format` render target.
        pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cube"),
                source: wgpu::ShaderSource::Wgsl(CUBE_SHADER.into()),
            });

            let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniforms"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bind_layout)],
                immediate_size: 0,
            });

            // Vertex layout matches the GXM stream exactly: position float32x3 at
            // offset 0, color as 4 normalized bytes at offset 12, stride 16.
            let vertex_layout = wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Unorm8x4,
                        offset: 12,
                        shader_location: 1,
                    },
                ],
            };

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cube"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[Some(vertex_layout)],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // No culling: match the software rasterizer, which draws both
                    // windings and relies on the depth test.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });

            CubeRenderer { pipeline, bind_layout }
        }

        /// Upload GPU buffers for each batch. Kept separate from encoding so the
        /// resources outlive the render pass that references them.
        fn prepare(&self, device: &wgpu::Device, batches: &[DrawBatch]) -> Vec<Prepared> {
            batches
                .iter()
                .map(|b| {
                    let mut mvp = Vec::with_capacity(64);
                    for v in &b.mvp {
                        mvp.extend_from_slice(&v.to_le_bytes());
                    }
                    let uniform =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("mvp"),
                            contents: &mvp,
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.bind_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform.as_entire_binding(),
                        }],
                    });
                    let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("vbo"),
                        contents: &b.vertices,
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("ibo"),
                        contents: &b.indices,
                        usage: wgpu::BufferUsages::INDEX,
                    });
                    Prepared {
                        vertices,
                        indices,
                        bind_group,
                        index_count: b.index_count,
                        index_u32: b.index_u32,
                    }
                })
                .collect()
        }

        /// Encode a full scene into `encoder`: a render pass over `color_view`
        /// (cleared to `clear`) with `depth_view` (must be [`DEPTH_FORMAT`]),
        /// drawing every batch. The caller owns the target textures and, on the
        /// native path, any subsequent copy-to-buffer readback.
        pub fn encode(
            &self,
            device: &wgpu::Device,
            encoder: &mut wgpu::CommandEncoder,
            color_view: &wgpu::TextureView,
            depth_view: &wgpu::TextureView,
            batches: &[DrawBatch],
            clear: [u8; 4],
        ) {
            let prepared = self.prepare(device, batches);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear[0] as f64 / 255.0,
                            g: clear[1] as f64 / 255.0,
                            b: clear[2] as f64 / 255.0,
                            a: clear[3] as f64 / 255.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            for p in &prepared {
                let fmt = if p.index_u32 {
                    wgpu::IndexFormat::Uint32
                } else {
                    wgpu::IndexFormat::Uint16
                };
                pass.set_bind_group(0, &p.bind_group, &[]);
                pass.set_vertex_buffer(0, p.vertices.slice(..));
                pass.set_index_buffer(p.indices.slice(..), fmt);
                pass.draw_indexed(0..p.index_count, 0, 0..1);
            }
        }
    }
}

/// The general GXM renderer: the GPU twin of the runtime's software rasterizer,
/// drawing a [`RenderScene`] of textured, alpha-blended, multi-space draws.
#[cfg(feature = "gpu")]
mod gxm {
    use super::{DrawSpace, RenderScene, DEPTH_FORMAT, GXM_VERTEX_STRIDE};
    use std::collections::{HashMap, HashSet};
    use wgpu::util::DeviceExt;

    /// A CPU stopwatch that is INERT in the browser.
    ///
    /// This renderer is shared with the wasm build, and `std::time::Instant` is not
    /// implemented on `wasm32-unknown-unknown`: constructing one panics at runtime.
    /// A diagnostic must never be able to take down the thing it is measuring, so on
    /// wasm this compiles to a zero-sized struct that always reports 0 ms. Native
    /// builds get the real clock.
    #[derive(Clone, Copy)]
    struct Stopwatch {
        #[cfg(not(target_arch = "wasm32"))]
        start: std::time::Instant,
    }

    impl Stopwatch {
        #[cfg(not(target_arch = "wasm32"))]
        fn start() -> Self {
            Stopwatch { start: std::time::Instant::now() }
        }
        #[cfg(target_arch = "wasm32")]
        fn start() -> Self {
            Stopwatch {}
        }
        #[cfg(not(target_arch = "wasm32"))]
        fn ms(&self) -> f64 {
            self.start.elapsed().as_secs_f64() * 1000.0
        }
        #[cfg(target_arch = "wasm32")]
        fn ms(&self) -> f64 {
            0.0
        }
    }

    /// The general fragment/vertex shader: the blob-free equivalent of a title's GXP
    /// shaders, a faithful per-fragment twin of `vitaslop_runtime::render`. The vertex
    /// stage projects per the draw's coordinate space (`u.mode`). There are two fragment
    /// entry points, selected by pipeline (never by a per-fragment branch), matching the
    /// software rasterizer's two combine modes:
    ///   - `fs_blend` (2D overlay): `vertex_color * texel` modulate. An untextured draw
    ///     binds a 1x1 white texel, so `color * white == color`.
    ///   - `fs_opaque` (3D opaque): the albedo texel is taken straight (its vertex color
    ///     is a non-color mask the real fragment program consumes, so modulating by it
    ///     would tint whole surfaces); an untextured opaque draw falls back to the vertex
    ///     color. Then scene `exposure` scales it and a Reinhard curve rolls off the
    ///     bright end, and alpha is forced to 1. `exposure == 1.0` skips the curve.
    const GXM_SHADER: &str = r#"
struct U {
    mvp: mat4x4<f32>,
    mode: u32,
    surf_w: f32,
    surf_h: f32,
    textured: u32,
    exposure: f32,
    depth_min: f32,
    depth_scale: f32,
    pad2: u32,
    tint: vec4<f32>,
    light_dir: vec4<f32>,
    light_col: vec4<f32>,
    ambient: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) wnormal: vec3<f32>,
};

@vertex
fn vs(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) color: vec4<f32>, @location(3) wnormal: vec3<f32>) -> VsOut {
    var out: VsOut;
    var clip: vec4<f32>;
    if (u.mode == 0u) {
        // Mvp: object space through the captured MVP. Depth is the projected view distance
        // `w`, NOT the clip `z` - GXM/PowerVR resolves visibility from `w`, and this title's
        // vertex programs emit a clip `z` whose post-divide value depends only on the screen
        // position (see `render::project`, which measured it identical across the ground and
        // every car draw covering one pixel). The software rasterizer stores `-1/w`, which is
        // screen-linear and increases with distance; the CPU scene builder measures that
        // quantity's visible range so it can be mapped LINEARLY onto the WebGPU depth buffer's
        // [0,1] without changing a single comparison - identical ordering to the oracle, at
        // full f32 resolution across the range. X and Y pass through.
        clip = u.mvp * vec4<f32>(position, 1.0);
        let depth = -1.0 / clip.w;
        let nz = clamp((depth - u.depth_min) * u.depth_scale, 0.0, 1.0);
        clip.z = nz * clip.w;
    } else if (u.mode == 1u) {
        // Ndc: clip coords emitted directly.
        clip = vec4<f32>(position.x, position.y, 0.0, 1.0);
    } else {
        // Pixel: screen pixels, Y down -> clip. Matches the software rasterizer,
        // which passes pixel coords straight to the viewport transform.
        let x = position.x / u.surf_w * 2.0 - 1.0;
        let y = 1.0 - position.y / u.surf_h * 2.0;
        clip = vec4<f32>(x, y, 0.0, 1.0);
    }
    out.pos = clip;
    out.uv = uv;
    out.color = color;
    out.wnormal = wnormal;
    return out;
}

@fragment
fn fs_blend(in: VsOut) -> @location(0) vec4<f32> {
    return in.color * textureSample(tex, samp, in.uv);
}

@fragment
fn fs_opaque(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    // Alpha-test opaque decal/livery layers (a BC2/BC3 albedo with a coverage alpha): discard
    // the transparent texels so the body panel behind shows, not the sheet's black background.
    // `discard` also skips the depth write, matching the software rasterizer's early `continue`.
    // Safe for the ordinary opaque BC1 albedo and the untextured white fallback (alpha == 1).
    if (texel.a < 0.5) {
        discard;
    }
    // The forward-lit material, mirroring the software rasterizer's `shade_lit` exactly:
    // albedo (the texel, or the vertex color when untextured) * per-material tint, lit by
    // one directional light (saturate(N.L) * light_col) plus a flat ambient, then scaled by
    // scene exposure and Reinhard tone-mapped so HDR light rolls off instead of clipping.
    let albedo = select(in.color.rgb, texel.rgb, u.textured != 0u);
    let n = normalize(in.wnormal);
    let l = normalize(u.light_dir.xyz);
    let ndotl = max(dot(n, l), 0.0);
    let light = u.ambient.rgb + u.light_col.rgb * ndotl;
    let lit = albedo * u.tint.rgb * light * u.exposure;
    let rgb = lit / (vec3<f32>(1.0) + lit);
    return vec4<f32>(rgb, 1.0);
}
"#;

    /// The supersample resolve shader: a fullscreen triangle whose fragment box-averages the
    /// `scale x scale` block of the offscreen colour target that maps to each output pixel -
    /// the exact integer box `Framebuffer::downsampled` applies, via `textureLoad` (no filtering)
    /// so the GPU and software AA'd frames agree. `scale` comes from a tiny uniform.
    const RESOLVE_SHADER: &str = r#"
struct RU { scale: u32 };
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> ru: RU;

@vertex
fn vres(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fres(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let s = i32(ru.scale);
    let base = vec2<i32>(i32(pos.x) * s, i32(pos.y) * s);
    var acc = vec4<f32>(0.0);
    for (var dy = 0; dy < s; dy = dy + 1) {
        for (var dx = 0; dx < s; dx = dx + 1) {
            acc = acc + textureLoad(srcTex, base + vec2<i32>(dx, dy), 0);
        }
    }
    return acc / f32(s * s);
}
"#;

    /// Convert one render target's DEPTH attachment into the value the GUEST's depth buffer
    /// would hold, so a later pass that samples that buffer reads a distance rather than a
    /// colour.
    ///
    /// The recompiled vertex stage does not write the guest's depth: `gxp_clipfix` replaces it
    /// with `clamp((-1/w - min) * scale, 0, 1)`, a monotonic remap onto the scene's visible
    /// range that keeps the depth TEST precise (see `RenderScene::depth_min`). That remap is
    /// affine in `-1/w` and therefore invertible, which is what makes this a small fullscreen
    /// pass instead of a second colour attachment on every draw of the pass: recover `w`, then
    /// re-encode it the way the guest expects.
    ///
    /// `mode` selects that encoding - see `GxmDepthEncoding`. It is a knob because the value a
    /// GXM depth surface holds is not something any clean source we hold states outright, and
    /// the honest thing is to make the choice visible and measurable rather than to bake in a
    /// guess.
    const GXM_DEPTH_SHADER: &str = r#"
struct DU { depth_min: f32, depth_scale: f32, mode: u32, konst: f32, fit_a: f32, fit_c: f32, pad0: f32, pad1: f32 };
@group(0) @binding(0) var srcDepth: texture_depth_2d;
@group(0) @binding(1) var<uniform> du: DU;

@vertex
fn vdep(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fdep(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // mode 5 is a CAUSALITY test, not an encoding: every texel becomes one known value, so a
    // pass that reads this depth either responds to it or does not. Reading a value only says
    // what a shader was fed; substituting one says whether that input is what decides the
    // picture. It reports itself, like every other substitution here.
    if (du.mode == 5u) {
        return vec4<f32>(du.konst, 0.0, 0.0, 1.0);
    }
    let d = textureLoad(srcDepth, vec2<i32>(i32(pos.x), i32(pos.y)), 0);
    // mode 4, and the degenerate scale==0 scene, pass the stored depth straight through:
    // with no range there is no `w` to recover, and inventing one would be worse than
    // handing back what is actually in the buffer.
    if (du.mode == 4u || du.depth_scale == 0.0) {
        return vec4<f32>(d, 0.0, 0.0, 1.0);
    }
    // Undo the clip fixup: q is -1/w, exactly what `gxp_clipfix` mapped onto [0,1].
    let q = d / du.depth_scale + du.depth_min;
    let w = select(-1.0 / q, 0.0, q == 0.0);
    var out = du.fit_a + du.fit_c / w;   // mode 0: the guest's own window depth `a + c/w`
    if (du.mode == 1u) { out = w; }
    else if (du.mode == 2u) { out = -w; }
    else if (du.mode == 3u) { out = q; }                                 // -1/w
    else if (du.mode == 6u) { out = -q; }                                // 1/w
    if (w == 0.0) { out = 0.0; }
    return vec4<f32>(out, 0.0, 0.0, 1.0);
}
"#;

    /// The sRGB twin of a linear colour format, when it has one.
    ///
    /// A GXM colour surface with `sceGxmColorSurfaceSetGammaMode` set has its writes
    /// sRGB-ENCODED by the ROP, and a texture with `sceGxmTextureSetGammaMode` has its reads
    /// sRGB-DECODED by the sampler. Rendering through - and sampling through - an sRGB VIEW of
    /// the same texture reproduces both exactly, including doing the encode AFTER blending,
    /// which is where the hardware does it. Nothing else in the pipeline has to change: the
    /// shader keeps writing linear values.
    fn srgb_twin(f: wgpu::TextureFormat) -> Option<wgpu::TextureFormat> {
        use wgpu::TextureFormat as F;
        match f {
            F::Rgba8Unorm => Some(F::Rgba8UnormSrgb),
            F::Bgra8Unorm => Some(F::Bgra8UnormSrgb),
            // Already sRGB, or a format with no sRGB twin (a float target needs none - it is
            // not quantised, so there is nothing for a transfer function to buy).
            _ => None,
        }
    }

    /// The format the converted guest-depth texture is stored in.
    ///
    /// A GXM `DF32` depth surface holds 32-bit floats, so `R32Float` is the exact match - but
    /// `R32Float` is NOT filterable in WebGPU core, and the recompiled sampler layout declares
    /// every unit filterable (it has to: it is one layout serving a shader's texture units,
    /// and the same unit index carries ordinary colour textures on other pairs). Binding an
    /// unfilterable format into a filterable slot is a validation error, not a soft failure.
    ///
    /// `Rgba16Float` is filterable everywhere and carries ~11 bits of mantissa, which over a
    /// view distance of a few hundred units is sub-unit precision - far finer than the soft
    /// fades that read it resolve. The cost is stated rather than assumed: see
    /// [`report_depth_conversion`], which names the encoding in every run.
    const GXM_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    /// Bytes of the depth-conversion uniform block (`DU` above).
    const GXM_DEPTH_UNIFORM_BYTES: u64 = 32;

    /// Which value a later pass reads out of a render target's depth
    /// (`VITASLOP_GXM_DEPTH_ENC`), matching the `mode` branch in [`GXM_DEPTH_SHADER`].
    /// Returns `(mode, constant)`; the constant is used only by mode 5.
    fn gxm_depth_encoding() -> (u32, f32) {
        let v = std::env::var("VITASLOP_GXM_DEPTH_ENC").unwrap_or_default();
        match v.as_str() {
            // The default is no longer a guess: `fit` writes the guest's own window depth
            // `a + c/w`, with `a` and `c` measured from the pass's vertex programs. The rest
            // stay as A/B alternatives, because a title whose depth surface turns out to hold
            // something else should cost one run to find out, not a code change.
            "fit" | "" => (0, 0.0),
            "w" => (1, 0.0),
            "negw" => (2, 0.0),
            "negrecipw" => (3, 0.0),
            "recipw" => (6, 0.0),
            "unit" => (4, 0.0),
            other => match other.strip_prefix("const:").map(str::parse::<f32>) {
                Some(Ok(k)) => (5, k),
                _ => panic!(
                    "VITASLOP_GXM_DEPTH_ENC={other} is not one of \
                     fit|w|negw|negrecipw|recipw|unit|const:<float> - refusing to guess which \
                     value the guest's depth buffer holds"
                ),
            },
        }
    }

    /// Bytes of the per-draw uniform block (matches the WGSL `U` struct): mat4 (64) +
    /// mode/surf_w/surf_h/textured (16) + exposure/depth_min/depth_scale/pad2 (16) +
    /// tint/light_dir/light_col/ambient (4 x vec4 = 64) = 160. Copies are laid into a
    /// shared buffer at [`GxmRenderer::uniform_stride`] spacing for dynamic offsets.
    const UNIFORM_BYTES: u64 = 160;

    /// Upper bound on the cross-frame texture caches before they are cleared wholesale
    /// (a re-upload, never incorrectness - the keys are content fingerprints, so a
    /// re-decoded atlas still hits).
    ///
    /// # In BYTES, because a count bounds nothing
    /// This used to be a cap of 512 ENTRIES. An entry is a texture of any size, so 512 of
    /// them is anywhere from a few megabytes to well over a gigabyte, and the cap fired at
    /// the same place either way. Native never noticed - it has an address space to spare.
    /// The browser did: a worker climbed 0.70 -> 1.81 GB while the emulator's own wasm heap
    /// stayed FLAT at 487 MB, and was killed with no error, no crash event and nothing in
    /// any log. A limit that does not track the resource it is limiting is not a limit.
    ///
    /// 256 MB is comfortably above this title's working set (so the cache still hits) and
    /// far below what a browser worker can survive. `VITASLOP_TEX_CACHE_MB` overrides.
    fn tex_cache_budget_bytes() -> usize {
        use std::sync::OnceLock;
        static CELL: OnceLock<usize> = OnceLock::new();
        *CELL.get_or_init(|| {
            crate::knobs::var("VITASLOP_TEX_CACHE_MB")
                .ok()
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(256)
                * 1024
                * 1024
        })
    }

    /// Bytes a decoded texture occupies once uploaded, for the cache budget. RGBA8 is what
    /// every upload path here produces, so this is exact rather than an estimate.
    fn texture_bytes(width: u32, height: u32) -> usize {
        (width.max(1) as usize) * (height.max(1) as usize) * 4
    }

    /// Upper bound on the repacked-geometry cache, in distinct meshes. A frame here submits a
    /// few hundred; the cap only fires on a title whose geometry genuinely changes every
    /// frame, where the cache would not have helped anyway.
    const PACKED_CACHE_CAP: usize = 4096;

    /// Which texture bind group a draw uses. `White` is the shared 1x1 opaque-white
    /// fallback for an untextured draw; `Tex` names a cached upload by content key and
    /// whether it is sampled LINEAR (so the two filter modes cache separately).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum BindKey {
        White,
        Tex(u64, bool),
        /// A texture whose pixels this frame's earlier pass rendered, named by the guest
        /// address of that pass's colour surface rather than by content - the guest bytes
        /// at that address are stale, so a content key would name the wrong image. The
        /// last field selects the pre-pass snapshot over the live target.
        Rtt(u32, bool, bool),
    }

    /// One draw resolved against the per-frame arena: byte ranges into the shared
    /// vertex/index buffers, the dynamic uniform offset, which pipeline, and which
    /// texture bind group. No owned GPU resources - everything lives in the renderer's
    /// persistent buffers and caches, so recording a frame allocates nothing.
    struct Item {
        v_off: u64,
        v_len: u64,
        i_off: u64,
        i_len: u64,
        index_count: u32,
        uniform_offset: u32,
        opaque: bool,
        bind: BindKey,
    }

    /// The general renderer: the GPU twin of the software rasterizer, built for a real
    /// title's per-frame draw volume (tens to thousands of draws).
    ///
    /// Performance shape - nothing is allocated per draw in steady state:
    ///   - Two pipelines (opaque depth-tested / alpha-blended) share one shader and
    ///     vertex layout; the fragment entry point is baked into the pipeline.
    ///   - Per-frame vertex, index and uniform data are packed into three grow-only
    ///     arena buffers uploaded in one `write_buffer` each - not a buffer per draw.
    ///   - Per-draw uniforms are addressed by DYNAMIC OFFSET into the one uniform
    ///     buffer (group 0), so there is a single uniform bind group for the whole frame.
    ///   - Texture bind groups (group 1) are cached across frames by content key + filter,
    ///     so an unchanged atlas is bound, never re-created; decoded uploads are cached by
    ///     content key. A `Nearest` and a `Linear` REPEAT sampler are both prebuilt.
    pub struct GxmRenderer {
        opaque: wgpu::RenderPipeline,
        blend: wgpu::RenderPipeline,
        /// The same two against the sRGB view of the target, for a pass whose colour surface
        /// the guest put in GAMMA-CORRECT mode. `None` when `color_format` has no sRGB twin.
        srgb: Option<(wgpu::RenderPipeline, wgpu::RenderPipeline)>,
        uniform_layout: wgpu::BindGroupLayout,
        texture_layout: wgpu::BindGroupLayout,
        sampler_point: wgpu::Sampler,
        sampler_linear: wgpu::Sampler,
        white_bind: wgpu::BindGroup,
        /// Decoded texture uploads (a view kept alive), keyed by content fingerprint.
        views: HashMap<u64, wgpu::TextureView>,
        /// Bytes currently held by `views`, against [`tex_cache_budget_bytes`].
        views_bytes: usize,
        /// Texture bind groups (view + chosen sampler), keyed by (content key, linear).
        tex_binds: HashMap<(u64, bool), wgpu::BindGroup>,
        /// Grow-only per-frame arenas + the uniform bind group over the uniform arena.
        vbo: Option<wgpu::Buffer>,
        ibo: Option<wgpu::Buffer>,
        ubo: Option<wgpu::Buffer>,
        ubo_bind: Option<wgpu::BindGroup>,
        vbo_cap: u64,
        ibo_cap: u64,
        ubo_cap: u64,
        /// The recompiled path's own grow-only vertex/index arenas, the counterpart of
        /// `vbo`/`ibo` above. Separate because the two paths pack different vertex layouts.
        /// Which PASS of the current chain is being encoded, used to invalidate the bind
        /// groups over that pass's uniform arena ([`GxpLive::ubo_bgs`]).
        ///
        /// The arenas are per PASS and not per frame, and that is not an accident: every pass
        /// of a chain records into ONE command encoder and the whole encoder is submitted at
        /// the end, so a shared buffer overwritten between passes would hand every pass the
        /// LAST pass's geometry. That is exactly what happened when this was first written -
        /// the race frame came out as sheets of shredded triangles.
        gxp_pass_gen: u64,
        /// Per-draw uniform spacing: `UNIFORM_BYTES` rounded up to the device's
        /// `min_uniform_buffer_offset_alignment` (256 by default).
        uniform_stride: u64,
        /// The caller's target colour format, needed to size the supersampled offscreen
        /// colour target (its format must match the scene pipelines' target).
        color_format: wgpu::TextureFormat,
        /// Supersample factor (1 = off, the default; N renders the scene at N x the target
        /// dimensions into an offscreen buffer and box-downsamples it into the caller's view).
        /// See [`GxmRenderer::set_supersample`] and the `resolve_*` fields below.
        ss_scale: u32,
        /// Last reported `(draws, with_payload, prepared, fixed)` recompiler summary. The
        /// summary is per-RENDER, so printing it unconditionally writes a line every frame -
        /// 60 a second in a live window. Reporting only when the tuple CHANGES keeps the whole
        /// signal (the first frame, and any draw moving between the recompiled and
        /// fixed-function paths) without the spam.
        last_gxp_summary: Option<(usize, usize, usize, usize)>,
        /// The box-downsample resolve pipeline + its bind-group layout and scale uniform,
        /// used only when `ss_scale > 1`. Built once in [`GxmRenderer::new`].
        resolve_pipeline: wgpu::RenderPipeline,
        resolve_layout: wgpu::BindGroupLayout,
        resolve_scale_buf: wgpu::Buffer,
        /// The depth-conversion pipeline + layout + uniform ([`GXM_DEPTH_SHADER`]), used only
        /// for a target whose depth a later pass samples. Built once in [`GxmRenderer::new`];
        /// a frame that samples no depth never runs it.
        gxm_depth_pipe: wgpu::RenderPipeline,
        gxm_depth_layout: wgpu::BindGroupLayout,
        gxm_depth_uniform: wgpu::Buffer,
        /// The lazily-(re)created offscreen supersample target (colour + depth + resolve bind
        /// group), sized to `ss_scale * surf`. Rebuilt when the scale or target size changes.
        ss_target: Option<SsTarget>,
        /// Offscreen render targets, keyed by the guest address of the colour surface the
        /// pass that fills them writes. Persistent across frames (a title reuses the same
        /// few targets every frame), rebuilt when a target's size changes.
        rtt: HashMap<u32, RttSurface>,
        /// Views of the targets already rendered in the CURRENT frame's chain, by guest
        /// address. A later pass only substitutes a rendered target for a sampled texture
        /// when THIS frame drew it; otherwise it would sample last frame's image, which is
        /// worse than the guest's own bytes because it looks plausible.
        rtt_rendered: HashMap<u32, wgpu::TextureView>,
        /// Fixed-function bind groups over a rendered target, keyed by (address, linear,
        /// reading-the-snapshot). The snapshot flag is part of the key because the two
        /// views are different textures - binding the live one where the snapshot is meant
        /// would make the target its own input.
        rtt_binds: HashMap<(u32, bool, bool), wgpu::BindGroup>,
        /// Addresses whose entry in `rtt_rendered` is currently the snapshot rather than
        /// the live target (the pass being encoded draws into that address).
        rtt_reads_snapshot: HashSet<u32>,
        /// Views of the guest-encoded DEPTH of the targets already rendered this frame, keyed
        /// by the guest address of the depth surface (NOT the colour one). A sampler naming
        /// one of these is asking for a distance, and must be resolved here BEFORE the
        /// colour-target range match, which would otherwise claim the address first.
        rtt_depth_rendered: HashMap<u32, wgpu::TextureView>,
        /// The pass currently being encoded writes into a target whose depth is sampled later,
        /// so its depth attachment must be STORED rather than discarded.
        keep_depth: bool,
        /// Draws in the current chain that sampled a target this frame rendered. Zero over
        /// a frame with several passes means the composite is NOT reading them, which is a
        /// different problem from the passes not being drawn - and the two look identical
        /// on screen.
        rtt_hits: usize,
        /// Last reported chain shape, so the report prints on a change rather than every
        /// frame. See the `gxm chain:` line.
        last_chain_shape: Option<String>,
        /// Guest addresses of every texture the pass being encoded sampled. Reported for
        /// the frame's LAST pass: a composite that shows none of the world is either not
        /// sampling the world target or sampling an address nothing rendered, and those
        /// need opposite fixes.
        sampled_addrs: HashSet<u32>,
        /// The live GXP->WGSL recompiler: a per-shader-pair pipeline cache. When enabled
        /// (`VITASLOP_GXP_LIVE`) a draw carrying [`super::GxpRecompile`] is rendered with the
        /// guest's real shaders; a pair that fails to link falls back to the fixed-function
        /// pipelines above. Disabled -> zero cost (the payload is simply ignored).
        gxp: GxpLive,
        /// What the last [`GxmRenderer::encode`] spent, phase by phase. See [`EncodePhases`].
        last_phases: EncodePhases,
        /// The same, summed over every pass of the last [`GxmRenderer::encode_chain`] - which
        /// is the frame, and the only figure the caller's `encode` total can be compared with.
        chain_phases: EncodePhases,
    }

    /// Where one `encode` went, in milliseconds of CPU.
    ///
    /// The recompiler path renders this title's main screen in ~30 ms warm against the
    /// fixed-function path's ~8 ms, and the interesting question is WHICH SIDE that
    /// delta is on: per-draw CPU work building GPU objects, or genuinely heavier
    /// shaders on the GPU. They call for opposite fixes, so guessing is expensive.
    ///
    /// `prepare` is the scene walk - for a recompiled draw that means creating its
    /// buffers and bind groups, where the fixed-function path only appends bytes to a
    /// grow-only arena. `upload` is the arena writes, `pass` is command encoding. All
    /// three are CPU; GPU time shows up in the caller's submit-and-wait, which is why
    /// the caller times that separately. Recorded unconditionally - three `Instant`s a
    /// frame cost nothing next to the milliseconds they measure, and a diagnostic that
    /// has to be switched on is one nobody has on when they need it.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct EncodePhases {
        pub prepare_ms: f64,
        pub upload_ms: f64,
        pub pass_ms: f64,
        /// Draws that took the recompiled path, and draws that took the fixed-function
        /// path - the denominator for any per-draw cost.
        pub gxp_draws: usize,
        pub fixed_draws: usize,
    }

    impl EncodePhases {
        /// Fold one PASS's phases into a whole-FRAME total.
        ///
        /// A frame is a CHAIN of passes, and reporting only the last one's phases describes
        /// whichever pass happened to be last - on this title's race frame that is an
        /// eleven-draw composite, so the split read `prepare 0.25 ms` while the frame was
        /// spending sixteen milliseconds encoding four hundred draws it never mentioned.
        fn add(&mut self, pass: EncodePhases) {
            self.prepare_ms += pass.prepare_ms;
            self.upload_ms += pass.upload_ms;
            self.pass_ms += pass.pass_ms;
            self.gxp_draws += pass.gxp_draws;
            self.fixed_draws += pass.fixed_draws;
        }
    }

    /// The offscreen supersample render target: an `ss_scale * surf` colour + depth buffer the
    /// scene is rendered into, plus the bind group the resolve pass samples the colour through.
    /// Kept across frames and rebuilt only when the scale or the target dimensions change.
    struct SsTarget {
        scale: u32,
        width: u32,
        height: u32,
        _color: wgpu::Texture,
        color_view: wgpu::TextureView,
        depth_view: wgpu::TextureView,
        resolve_bind: wgpu::BindGroup,
    }

    /// A scene whose colour surface could not be resolved has nowhere to be drawn, and
    /// whatever it drew is missing from the frame. Reported the first time it happens (a
    /// per-frame pass would otherwise flood the log), unconditionally - a pass silently
    /// vanishing is indistinguishable from the guest never submitting it, and that
    /// ambiguity has cost this project whole sessions.
    fn report_unplaced_scene(draws: usize) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static REPORTED: AtomicBool = AtomicBool::new(false);
        if !REPORTED.swap(true, Ordering::Relaxed) {
            report!(
                "gxm: a scene with {draws} draws has no resolvable colour surface - it renders \
                 nowhere, and any later pass that samples its target gets stale guest memory"
            );
        }
    }

    /// One offscreen render target: the colour buffer a pass draws into and a matching
    /// depth buffer. Held by the guest address of the colour surface, so the pass that
    /// samples it later can be matched to it by the texture's data pointer.
    struct RttSurface {
        width: u32,
        height: u32,
        color: wgpu::Texture,
        color_view: wgpu::TextureView,
        /// An sRGB view of the SAME `color` texture, for a surface the guest put in
        /// gamma-correct mode. Rendering through it makes the ROP sRGB-encode every store
        /// after blending, and sampling through it decodes on read - which is what the
        /// hardware does at both ends. `None` when the colour format has no sRGB twin.
        color_view_srgb: Option<wgpu::TextureView>,
        depth_view: wgpu::TextureView,
        /// A copy of `color` as it stood before the pass now drawing into it, made only
        /// when that pass also SAMPLES the buffer - see `GxmRenderer::snapshot_rtt`.
        shadow: Option<(wgpu::Texture, wgpu::TextureView)>,
        /// This target's depth, re-encoded the way the GUEST's depth buffer holds it, for a
        /// later pass that samples it. Present only when some pass in the frame actually
        /// names this scene's depth address - see `GxmRenderer::encode_chain`.
        gxm_depth: Option<GxmDepthTarget>,
    }

    /// A render target's depth in the guest's own encoding, plus what it takes to produce and
    /// sample it.
    struct GxmDepthTarget {
        /// A sampleable view of the depth ATTACHMENT (the conversion pass's input).
        src_view: wgpu::TextureView,
        /// The converted R32Float texture (the output), and the view a draw samples it through.
        _tex: wgpu::Texture,
        view: wgpu::TextureView,
    }

    /// Round `v` up to a multiple of `align` (a power of two).
    fn align_up(v: u64, align: u64) -> u64 {
        (v + align - 1) & !(align - 1)
    }

    // ==================== Live GXP->WGSL recompiler path ====================
    //
    // When enabled (`VITASLOP_GXP_LIVE`), a draw that carries the guest's real vertex +
    // fragment `SceGxmProgram` blobs (see [`super::GxpRecompile`]) is rendered with those
    // shaders, translated to one linked WGSL module by `vitaslop_gxp_shader::link_programs`
    // and executed on a real pipeline. A pair that fails to link (or uses a vertex format /
    // 3D sampler we do not yet map) falls back to the fixed-function pipeline for that draw -
    // a wrong translation never paints a pixel. Pipelines are cached by shader identity;
    // per-draw vertex/index/uniform buffers + bind groups are built each frame (a real title's
    // recompilable draw count is small, and this keeps the path simple and correct first).

    use super::{GxpAttr, GxpRecompile, GxmTexture};

    /// A linked + compiled pipeline for one guest shader pair, cached by shader identity.
    struct GxpPipeline {
        /// Opaque variant: depth test LessEqual + depth write, no blend.
        opaque: wgpu::RenderPipeline,
        /// Blend variant: depth test but no write, straight-alpha src-over.
        blend: wgpu::RenderPipeline,
        /// group0 = vertex SA uniform, group1 = fragment SA uniform, group2 = samplers.
        /// Empty layouts where the stage declares nothing, so the pipeline layout still
        /// covers every group index the WGSL might reference.
        layouts: [wgpu::BindGroupLayout; 4],
        /// Vertex SA scalar-lane count (the group0 uniform holds `ceil(n/4)` vec4s).
        vsa_lanes: u32,
        /// Fragment SA scalar-lane count.
        fsa_lanes: u32,
        /// `(sampler unit, is_3d)` per group2 sampler, in binding order.
        samplers: Vec<(u8, SamplerDim)>,
        /// The VERTEX stage's sampler units, in group-4 binding order.
        vertex_samplers: Vec<(u8, SamplerDim)>,
        /// How to repack the guest vertex stream into the tightly-packed `Float32xN` buffer
        /// this pipeline's vertex layout expects (one entry per attribute), + the packed
        /// stride. Repacking to f32 on the CPU sidesteps wgpu's vertex-format gaps (no
        /// `Float16x3`, etc.) and matches the recompiled shader, which reads f32 anyway.
        repack: Vec<RepackAttr>,
        packed_stride: u32,
    }

    /// One attribute's recipe for repacking the guest vertex stream to packed f32.
    struct RepackAttr {
        guest_offset: u32,
        gxm_format: u8,
        components: u8,
        packed_offset: u32,
    }

    /// Per-draw GPU resources for one recompiled draw, kept alive through the render pass.
    struct GxpPrepared {
        /// Cache key of the pipeline this draw uses (looked up immutably during the pass).
        key: u64,
        /// Byte range of this draw's repacked vertices and its indices inside the pass's
        /// grow-only GXP arenas.
        ///
        /// NOT per-draw buffers. Creating a vertex and an index buffer per draw is two GPU
        /// allocations per draw, and a race frame here submits ~470 recompiled draws: measured,
        /// that per-draw allocation was 14.6 ms of the 15.2 ms `encode`, i.e. the single
        /// largest item in the whole system. The fixed-function path has always packed into
        /// shared arenas uploaded once; this is the same thing for the recompiled path.
        v_off: u64,
        v_len: u64,
        i_off: u64,
        i_len: u64,
        index_count: u32,
        /// Byte offsets of this draw's vertex and fragment SA blocks inside the pass's uniform
        /// arena, for the group0/group1 DYNAMIC offsets. The bind groups themselves belong to
        /// the shader PAIR, not the draw ([`GxpLive::ubo_bgs`]).
        u_off: [u32; 2],
        /// Bind groups for group2 (samplers) and group3 (the pass depth block).
        bg2: wgpu::BindGroup,
        bg3: wgpu::BindGroup,
        /// True = alpha-blended (2D/overlay), false = opaque geometry.
        blend: bool,
        /// The guest's GXM viewport for this draw, `[xOffset,xScale,yOffset,yScale,zOffset,
        /// zScale]`. All-zero means the guest left the default (the whole target).
        viewport: [f32; 6],
        /// The attachment format this draw's pipeline was built for - part of the pipeline
        /// cache key, because a gamma-correct surface renders through an sRGB view.
        format: wgpu::TextureFormat,
    }

    /// One entry in the submission-order draw plan: either a fixed-function [`Item`] (by index
    /// into the arena-packed `items`) or a recompiled draw (by index into `gxp_prepared`). The
    /// two kinds interleave in the one render pass so depth and overlay ordering stay correct.
    enum Enc {
        Fixed(usize),
        Gxp(usize),
    }

    /// The live recompiler's pipeline cache + config. Held by [`GxmRenderer`].
    struct GxpLive {
        /// Master switch (`VITASLOP_GXP_LIVE`).
        enabled: bool,
        /// Render ONLY recompiled draws, skipping the fixed-function draw for any call that
        /// has a working recompiled pipeline (`VITASLOP_GXP_ONLY`) - isolates the recompiler
        /// output for review.
        only: bool,
        /// Apply the GXM (GL-style, NDC z in [-1,1]) -> WebGPU (z in [0,1]) clip-depth remap
        /// in the vertex output (`VITASLOP_GXP_ZFIX`, default on). Off passes clip z straight.
        zfix: ZFix,
        /// Flip clip Y (`VITASLOP_GXP_YFLIP`, default off). The fixed-function MVP path passes
        /// guest clip X/Y straight to WebGPU and renders upright, so the real shaders should
        /// too; the toggle is here for empirical confirmation.
        yflip: bool,
        /// Diagnostic (`VITASLOP_GXP_FORCE`): bind a neutral fallback texture for a sampler
        /// unit whose real texture we could not capture/decode (e.g. a 3D/LUT format) or a 3D
        /// sampler, so the recompiled GEOMETRY renders and the vertex/clip/depth path can be
        /// validated even before the special textures are handled. Off = strict (fall back to
        /// fixed-function rather than sample a wrong texel).
        force: bool,
        /// Diagnostic (`VITASLOP_GXP_SOLID`): every recompiled draw outputs solid magenta with
        /// the depth test disabled, to answer "does the geometry rasterize on-screen at all?"
        /// independent of fragment shading and depth. Magenta visible -> vertex/clip is right.
        solid: bool,
        /// Diagnostic (`VITASLOP_GXP_NODEPTH`): every recompiled draw keeps its real shading and
        /// blending but stops testing depth. `solid` answers "does this geometry rasterize"
        /// while changing BOTH the shading and the depth test, so a surface that comes out
        /// black under it is still ambiguous - the fragment could be painting black, or it
        /// could be losing the depth test. This changes only the depth test, which separates
        /// the two: a surface that appears here and not in an ordinary run is depth-rejected,
        /// one that stays black is shaded black.
        nodepth: bool,
        /// Diagnostic (`VITASLOP_GXP_NOBLEND`): force every recompiled pipeline to REPLACE with
        /// a full colour write mask, changing NOTHING else. The counterpart of `nodepth` for the
        /// other way a correctly-shaded draw can leave no mark: a shader that writes alpha 0
        /// under a src-alpha blend, or a guest colour mask that writes no channels.
        noblend: bool,
        /// Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys
        /// (the `gxp draw key` value `VITASLOP_GXP_DUMP` prints), letting every other draw fall
        /// back. Rendering one pair at a time is how a visual artifact is attributed to the
        /// shader that produced it. Empty = no filter (recompile every linkable pair).
        keys: Vec<u64>,
        /// Pairs forced down the fixed-function path (`VITASLOP_GXP_EXCLUDE`).
        exclude: Vec<u64>,
        /// `(key, target format) -> Some(pipeline)` for a linkable pair, `-> None` for one that
        /// failed to link (cached so we never retry it and always fall back).
        ///
        /// The FORMAT is part of the key because a render pipeline is bound to the format of
        /// the attachment it writes, and a surface the guest put in gamma-correct mode is
        /// rendered through an sRGB view of the same texture. Only a pair that is actually
        /// drawn onto such a surface ever gets a second entry, so a title using no gamma
        /// surfaces builds exactly as many pipelines as before.
        pipelines: HashMap<(u64, wgpu::TextureFormat), Option<GxpPipeline>>,
        /// Uploaded texture views, keyed by the decoded texture's content fingerprint and the
        /// view dimension it is bound as. A scene binds a handful of textures across hundreds
        /// of draws, so uploading per draw (as this path first did) re-sends the same
        /// multi-megabyte shadow map thousands of times a frame and exhausts GPU memory.
        views: HashMap<(u64, SamplerDim), wgpu::TextureView>,
        /// Bytes currently held by `views`, against [`tex_cache_budget_bytes`]. Tracked
        /// rather than derived because a `TextureView` cannot be asked its size.
        views_bytes: usize,
        /// Samplers by `(linear, addr_mode_u, addr_mode_v)`. A handful per title.
        samplers_by_mode: HashMap<(bool, u32, u32), wgpu::Sampler>,
        /// How to choose the clip-`w` sign correction (`VITASLOP_GXP_NEGW`).
        negw: NegW,
        /// What interpreting each shader pair's vertex program over its own mesh said about the
        /// projection behind it (see `measure_clip`). Measured ONCE per key, on the first draw
        /// that uses it. `None` records a pair whose program does not interpret, so it is not
        /// re-attempted.
        ///
        /// The DECISIONS are not per pair - see [`GxpLive::decide_scene_negw`].
        negw_by_key: HashMap<u64, Option<ClipStats>>,
        /// Whether THIS pass's projection puts clip `w` negative in front of the camera, as
        /// decided by [`GxpLive::decide_scene_negw`] before the pass's draws are walked.
        scene_negw: bool,
        /// THIS pass's `(a, c)` in `guest window depth = a + c/w` (see [`ClipStats::depth_fit`]).
        scene_depth_fit: (f32, f32),
        /// Both verdicts, per render-target address, once a frame's draws produced evidence for
        /// them. Keeps the measurement off the per-frame path: a projection belongs to the pass,
        /// and a pass does not change projection convention between frames.
        negw_by_target: HashMap<u32, (bool, (f32, f32))>,
        /// group0/group1 bind groups over the pass's uniform ARENA, one per (shader pair,
        /// target format, group) rather than one per draw - the draw supplies only a dynamic
        /// offset. Cleared wholesale when the arena buffer is re-created, because a bind group
        /// names a specific buffer; `ubo_bgs_gen` is that buffer's generation.
        ubo_bgs: HashMap<(u64, wgpu::TextureFormat, u8), wgpu::BindGroup>,
        ubo_bgs_gen: u64,
        /// Repacked vertex streams, keyed by `(pipeline key, content hash of the guest
        /// stream)`. The repack walks every component of every vertex, and a world pass here
        /// submits meshes of ~3800 vertices with eleven components each, EVERY FRAME, from
        /// byte-identical guest bytes - the static world does not change between frames. The
        /// key is a content hash rather than the guest address precisely because a title also
        /// has dynamic geometry at a fixed address, which an address key would serve stale.
        packed: HashMap<(u64, u64), std::sync::Arc<[u8]>>,
        /// The `@group(3)` depth-range bind group, keyed by `(pipeline key, the depth
        /// range's bits)`.
        ///
        /// That group holds ONE vec4 - the scene's depth min and scale - which is the
        /// same for every draw in a scene. Building a fresh 16-byte buffer and bind
        /// group per draw meant a couple of hundred GPU allocations a frame to say the
        /// same thing each time. Keyed by pipeline as well as by value because each
        /// pipeline owns its own `BindGroupLayout` object; a shared group-3 layout
        /// would collapse this to one entry, and is the right follow-up.
        depth_bgs: HashMap<(u64, u64, bool), wgpu::BindGroup>,
    }

    /// Which depth the recompiled vertex stage writes (`VITASLOP_GXP_ZFIX`).
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum ZFix {
        /// The scene's `-1/w` range, normalized onto [0,1] (default): the same quantity and the
        /// same range the fixed-function path writes, so both kinds of draw share one comparable
        /// depth buffer. Costs a dependency on the scene depth range, which is measured through
        /// the software path's own reflected transform.
        Range,
        /// The ordinary GL->WebGPU clip-depth remap `(z + w) / 2` (`=gl`), using the guest's own
        /// clip z and nothing else.
        Gl,
        /// Pass the guest's clip z straight through (`=0`).
        Off,
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
    enum NegW {
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
    fn gxp_dump() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::knobs::flag("VITASLOP_GXP_DUMP"))
    }

    impl GxpLive {
        /// Read the recompiler's configuration.
        ///
        /// Through [`crate::knobs`], not `std::env` directly: the browser has no
        /// environment, so an `std::env` read here silently disables the recompiler
        /// there and leaves the browser drawing the fixed-function approximation while
        /// the desktop oracle draws the guest's real shaders.
        fn from_env() -> Self {
            let flag = crate::knobs::flag;
            GxpLive {
                enabled: flag("VITASLOP_GXP_LIVE"),
                only: flag("VITASLOP_GXP_ONLY"),
                zfix: match crate::knobs::var("VITASLOP_GXP_ZFIX").ok().as_deref() {
                    Some("0") | Some("off") => ZFix::Off,
                    Some("gl") => ZFix::Gl,
                    _ => ZFix::Range,
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
                pipelines: HashMap::new(),
                views: HashMap::new(),
                views_bytes: 0,
                samplers_by_mode: HashMap::new(),
                negw: match crate::knobs::var("VITASLOP_GXP_NEGW").ok().as_deref() {
                    Some("0") | Some("off") => NegW::Off,
                    Some("negate") => NegW::Negate,
                    Some("force") => NegW::Force,
                    _ => NegW::Auto,
                },
                negw_by_key: HashMap::new(),
                scene_negw: false,
                scene_depth_fit: DEPTH_FIT_RECIP_W,
                negw_by_target: HashMap::new(),
                ubo_bgs: HashMap::new(),
                ubo_bgs_gen: u64::MAX,
                packed: HashMap::new(),
                depth_bgs: HashMap::new(),
            }
        }

        /// Ensure a group0/group1 bind group exists over the pass's uniform arena for every
        /// shader pair the pass uses. Called once per pass, after the arena buffer exists -
        /// which is why it cannot happen inside `prepare`: the buffer is sized from the arena
        /// `prepare` fills.
        fn ensure_ubo_bgs(
            &mut self,
            device: &wgpu::Device,
            buffer: &wgpu::Buffer,
            generation: u64,
            used: &[(u64, wgpu::TextureFormat)],
        ) {
            if self.ubo_bgs_gen != generation {
                self.ubo_bgs.clear();
                self.ubo_bgs_gen = generation;
            }
            for &(key, format) in used {
                let Some(Some(pipe)) = self.pipelines.get(&(key, format)) else { continue };
                for (group, lanes) in [(0u8, pipe.vsa_lanes), (1u8, pipe.fsa_lanes)] {
                    if self.ubo_bgs.contains_key(&(key, format, group)) {
                        continue;
                    }
                    let layout = &pipe.layouts[group as usize];
                    let bg = if lanes == 0 {
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("gxp-ubo-empty"),
                            layout,
                            entries: &[],
                        })
                    } else {
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("gxp-ubo-bind"),
                            layout,
                            entries: &[wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer,
                                    offset: 0,
                                    size: wgpu::BufferSize::new((lanes.div_ceil(4) as u64) * 16),
                                }),
                            }],
                        })
                    };
                    self.ubo_bgs.insert((key, format, group), bg);
                }
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
        fn decide_scene_negw(&mut self, scene: &RenderScene) {
            if self.negw == NegW::Off || !self.enabled {
                self.scene_negw = false;
                self.scene_depth_fit = DEPTH_FIT_RECIP_W;
                return;
            }
            // `=force`: correct every pass without measuring, and SAY SO - a frame built on an
            // assumed projection must never be mistaken for a measured one.
            if self.negw == NegW::Force {
                let target = scene.target.as_ref().map(|t| t.data_addr).unwrap_or(0);
                report_forced_negw(target);
                self.scene_negw = true;
                self.scene_depth_fit = DEPTH_FIT_RECIP_W;
                return;
            }
            let target = scene.target.as_ref().map(|t| t.data_addr).unwrap_or(0);
            // Settled passes cost nothing per frame. A pass is only settled once its draws
            // produced EVIDENCE: a frame in which every draw of a pass happens to cover nothing
            // says nothing about its projection, and freezing "not negative" from it would make
            // the answer depend on which frame the pass was first seen in.
            if let Some(&decided) = self.negw_by_target.get(&target) {
                self.scene_negw = decided.0;
                self.scene_depth_fit = decided.1;
                return;
            }
            let (mut in_front, mut behind) = (0usize, 0usize);
            // The fit is taken from the ONE draw with the widest spread of `w`, not averaged
            // over the pass: `z = a*w + c` is exact for every draw sharing the projection, so a
            // wider spread is simply a better-conditioned way of asking the same question, while
            // an average would let a near-degenerate draw drag the answer.
            let (mut fit, mut best_spread) = (None, 0.0f32);
            for d in &scene.draws {
                let Some(gxp) = d.gxp.as_ref() else { continue };
                let key = Self::key(gxp);
                let stats = match self.negw_by_key.get(&key) {
                    Some(&s) => s,
                    None => {
                        let s = measure_clip(gxp, key);
                        // Only remember an answer the measurement supports: a draw entirely off
                        // screen decides nothing, and the next draw of the same pair may see the
                        // geometry that settles it.
                        let empty = s.is_some_and(|s| s.in_front == 0 && s.behind == 0);
                        if !empty {
                            self.negw_by_key.insert(key, s);
                        }
                        s
                    }
                };
                let Some(s) = stats else { continue };
                in_front += s.in_front;
                behind += s.behind;
                if let Some(f) = s.depth_fit {
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
            let correct = behind > 0 && in_front == 0;
            let depth_fit = fit.unwrap_or(DEPTH_FIT_RECIP_W);
            if in_front + behind > 0 {
                self.negw_by_target.insert(target, (correct, depth_fit));
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
        }

        /// The measured `(a, c)` of `guest window depth = a + c/w` for the pass that RENDERS
        /// into `addr`, for a later pass that wants to sample its depth.
        ///
        /// Keyed by the target rather than taken from `scene_depth_fit`, because the pass being
        /// encoded when a depth surface is converted is not necessarily the pass that wrote it.
        /// Falls back to `-1/w` for a target no pass has yet produced evidence for - which is
        /// the honest answer, not a silent zero: it is the encoding with no projection in it.
        fn depth_fit_for(&self, addr: u32) -> (f32, f32) {
            self.negw_by_target.get(&addr).map(|v| v.1).unwrap_or(DEPTH_FIT_RECIP_W)
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
        fn key(gxp: &GxpRecompile) -> u64 {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            let depth = [gxp.depth_write as u8, (gxp.depth_func >> 22) as u8 & 0x7];
            for b in gxp
                .vprog
                .iter()
                .chain(gxp.fprog.iter())
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
        fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            color_format: wgpu::TextureFormat,
            gxp: &GxpRecompile,
            depth_range: [f32; 2],
            // `rendered`: views of the render targets this frame has already drawn, by
            // guest address. A sampler whose bound texture points at one of these binds
            // the render rather than the guest's (stale) bytes.
            rendered: &HashMap<u32, wgpu::TextureView>,
            // The same, for the DEPTH of those targets, keyed by the guest's depth-surface
            // address. Checked before `rendered` - see `make_sampler_bg`.
            depth_rendered: &HashMap<u32, wgpu::TextureView>,
            // The pass's grow-only vertex and index arenas, appended to rather than allocated
            // from per draw. See [`GxpPrepared`].
            vdata: &mut Vec<u8>,
            idata: &mut Vec<u8>,
            // The pass's uniform arena and the device's dynamic-offset alignment.
            udata: &mut Vec<u8>,
            ubo_align: u64,
        ) -> Option<GxpPrepared> {
            if gxp.index_count == 0 || gxp.vertices.is_empty() {
                return None;
            }
            let key = Self::key(gxp);
            if !self.keys.is_empty() && !self.keys.contains(&key) {
                return None;
            }
            // `VITASLOP_GXP_EXCLUDE=<key>,...`: send these pairs down the fixed-function
            // path instead. The complement of `VITASLOP_GXP_KEYS`, and the useful one when
            // a frame is almost right: it answers "is this ONE draw the recompiler's fault"
            // without also un-recompiling the hundreds of draws that are already correct.
            if self.exclude.contains(&key) {
                return None;
            }
            report_inputs(key, gxp);
            let cache_key = (key, color_format);
            if !self.pipelines.contains_key(&cache_key) {
                // Name the pair's two containers by their CONTENT hash the moment it is first
                // seen. `Program::hash` is the same value the offline corpus computes, so this
                // one line is what turns a draw key from the frame into the two `.gxp` blobs an
                // offline test can open. Printed once per unique pair (not per draw), and
                // unconditionally: a diagnostic that needs a knob set is a diagnostic nobody has
                // when the surprising frame is already in front of them.
                let ph = |b: &[u8]| {
                    vitaslop_gxp_shader::Program::parse(b).map(|p| p.hash).unwrap_or(0)
                };
                report!(
                    "gxp pair {key:x}: vprog hash {:016x}, fprog hash {:016x}",
                    ph(&gxp.vprog),
                    ph(&gxp.fprog)
                );
                let built = build_gxp_pipeline(device, color_format, gxp, key, self.zfix, self.yflip, self.solid, self.nodepth, self.noblend);
                self.pipelines.insert(cache_key, built);
            }

            // Split the borrows: the sampler bind group needs the texture-view cache mutably
            // while the pipeline (its layouts, its sampler plan) stays borrowed.
            let negw_mode = self.negw;
            // Which clip-depth remap `inject_clip_fixup` put the vertex stage through. A
            // fragment that WRITES its own depth has to invert exactly that map, and only the
            // renderer knows which one is in force, so it travels in the depth block.
            let zfix_mode: f32 = match self.zfix {
                ZFix::Range => 0.0,
                ZFix::Gl => 1.0,
                ZFix::Off => 2.0,
            };
            let scene_negw = self.scene_negw;
            let scene_depth_fit = self.scene_depth_fit;
            let GxpLive {
                pipelines,
                views: view_cache,
                views_bytes: view_cache_bytes,
                samplers_by_mode,
                force,
                depth_bgs,
                packed,
                ..
            } = self;
            // Borrow the cached pipeline; None = link failed -> fall back.
            let pipe = pipelines.get(&cache_key)?.as_ref()?;

            if gxp_dump() {
                let f: Vec<f32> = gxp
                    .vert_sa
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let attrs: Vec<(u16, u16, u8, u8)> =
                    gxp.attributes.iter().map(|a| (a.reg_index, a.offset, a.gxm_format, a.components)).collect();
                let ff: Vec<f32> = gxp
                    .frag_sa
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                // The guest stream's own extent next to what the indices actually reference: a
                // mesh whose highest index is beyond the captured vertices renders only the
                // triangles that fall inside the buffer, i.e. a PREFIX of the geometry.
                let nverts = gxp.vertices.len() / (gxp.vertex_stride.max(1) as usize);
                let max_index = if gxp.index_u32 {
                    gxp.indices.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).max()
                } else {
                    gxp.indices.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).max()
                }
                .unwrap_or(0);
                report!(
                    "gxp draw key {:x}: vsa_lanes={} vert_sa_lanes={} fsa_lanes={} frag_sa_lanes={} samplers={:?} stride={} idx={} nverts={nverts} max_index={max_index} vbytes={} attrs(reg,off,fmt,comp)={:?}\n  vsa={:?}\n  fsa={:?}",
                    key, pipe.vsa_lanes, f.len(), pipe.fsa_lanes, ff.len(), pipe.samplers, gxp.vertex_stride, gxp.index_count, gxp.vertices.len(), attrs, f, ff
                );
                // For a small mesh, the attribute VALUES as fetched. A quad landing in the
                // wrong place on screen is either its positions or its transform, and those
                // are indistinguishable from the picture; this shows the positions.
                if nverts <= 8 {
                    let stride = gxp.vertex_stride.max(1) as usize;
                    for v in 0..nverts {
                        let vals: Vec<String> = gxp
                            .attributes
                            .iter()
                            .map(|a| {
                                let base = v * stride + a.offset as usize;
                                let c: Vec<String> = (0..a.components as usize)
                                    .map(|i| format!("{:.3}", read_attr_component(&gxp.vertices, base, a.gxm_format, i)))
                                    .collect();
                                format!("r{}=[{}]", a.reg_index, c.join(","))
                            })
                            .collect();
                        report!("    v{v}: {}", vals.join(" "));
                    }
                }
            }

            // Repack the guest vertex stream into the tightly-packed f32 layout the pipeline
            // expects (per the cached repack plan), straight into the pass's vertex arena.
            // Same vertex count/order, so the index buffer is unchanged.
            //
            // Both arenas stay 4-byte aligned, which `set_vertex_buffer` requires and a u32
            // index buffer requires: the packed stride is a whole number of f32s and an index
            // is four bytes, so the padding below never actually fires - it is there so a
            // future format that is not cannot silently produce an unaligned slice.
            let v_off = vdata.len() as u64;
            let pkey = (key, fnv64(0xcbf2_9ce4_8422_2325, &gxp.vertices));
            match packed.get(&pkey) {
                Some(bytes) => vdata.extend_from_slice(bytes),
                None => {
                    // Bound the cache the way the texture caches are bounded: the key is a
                    // content hash, so clearing wholesale costs a repack and never correctness.
                    if packed.len() >= PACKED_CACHE_CAP {
                        packed.clear();
                    }
                    let mut out = Vec::new();
                    repack_vertices_into(&gxp.vertices, gxp.vertex_stride, &pipe.repack, pipe.packed_stride, &mut out);
                    vdata.extend_from_slice(&out);
                    packed.insert(pkey, out.into());
                }
            }
            let v_len = vdata.len() as u64 - v_off;
            while vdata.len() % 4 != 0 {
                vdata.push(0);
            }
            let i_off = idata.len() as u64;
            idata.extend_from_slice(&gxp.indices);
            let i_len = idata.len() as u64 - i_off;
            while idata.len() % 4 != 0 {
                idata.push(0);
            }

            // The two SA blocks go into the pass's uniform ARENA at dynamic-offset alignment;
            // the bind groups over that arena belong to the shader pair and are built once,
            // after the arena buffer exists (see `ensure_ubo_bgs`).
            let vert_sa = override_sa(key, 'v', &gxp.vert_sa);
            let frag_sa = override_sa(key, 'f', &gxp.frag_sa);
            let u_off = [
                push_sa(udata, pipe.vsa_lanes, &vert_sa, ubo_align),
                push_sa(udata, pipe.fsa_lanes, &frag_sa, ubo_align),
            ];
            let bg2 = Self::make_sampler_bg(
                device, queue, &pipe.layouts[2],
                &[
                    (&pipe.samplers[..], &gxp.textures[..]),
                    (&pipe.vertex_samplers[..], &gxp.vertex_textures[..]),
                ],
                gxp, key,
                view_cache, view_cache_bytes, samplers_by_mode, *force,
                rendered, depth_rendered,
            )?;
            // group3: the scene depth range the injected clip fixup maps through, as one vec4
            // (min, scale, unused, unused) - the same values the fixed-function path uses, so
            // both kinds of draw write comparable depth. Per SCENE, not per draw, so it is
            // cached (see `GxpLive::depth_bgs`); the bit pattern is the key so a changed
            // range builds a new one rather than reusing a stale buffer.
            //
            // Its third lane is the clip-`w` sign correction, which is a property of the DRAW
            // (its projection), not of the scene - so it joins the cache key.
            let corrected = scene_negw;
            let sign: f32 = match (corrected, negw_mode) {
                (false, _) => 1.0,
                (true, NegW::Negate) => -1.0,
                (true, _) => 2.0,
            };
            // Lanes 4 and 5 are this pass's guest depth encoding `a + c/w`, and they are the
            // same numbers the depth-conversion pass writes with: a fragment reading its own
            // POSITION.z and a fragment sampling a converted depth surface must be looking at
            // one quantity, or every soft fade between them compares apples to oranges.
            let (fit_a, fit_c) = scene_depth_fit;
            let depth_key = (depth_range[0].to_bits() as u64) << 32 | depth_range[1].to_bits() as u64;
            let bg3 = depth_bgs.entry((key, depth_key, corrected)).or_insert_with(|| {
                let dbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gxp-depth"),
                    contents: &[
                        depth_range[0].to_le_bytes(),
                        depth_range[1].to_le_bytes(),
                        sign.to_le_bytes(),
                        zfix_mode.to_le_bytes(),
                        fit_a.to_le_bytes(),
                        fit_c.to_le_bytes(),
                        [0; 4],
                        [0; 4],
                    ]
                    .concat(),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gxp-depth-bind"),
                    layout: &pipe.layouts[3],
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: dbuf.as_entire_binding() }],
                })
            });
            let bg3 = bg3.clone();

            Some(GxpPrepared {
                key,
                v_off,
                v_len,
                i_off,
                i_len,
                index_count: gxp.index_count,
                u_off,
                bg2,
                bg3,
                blend: gxp.blend,
                viewport: gxp.viewport,
                format: color_format,
            })
        }

        /// The cached group0/group1 bind group for a prepared draw's shader pair (only called
        /// after `ensure_ubo_bgs` has run for this pass).
        fn ubo_bg(&self, key: u64, format: wgpu::TextureFormat, group: u8) -> &wgpu::BindGroup {
            &self.ubo_bgs[&(key, format, group)]
        }

        /// The cached pipeline for a prepared draw (only called after `prepare` succeeded).
        fn pipeline(&self, key: u64, format: wgpu::TextureFormat) -> &GxpPipeline {
            self.pipelines
                .get(&(key, format))
                .and_then(|p| p.as_ref())
                .expect("prepared key present")
        }

        /// Build the group2 sampler bind group: for each declared sampler unit, upload the
        /// bound texture and bind it with the matching filter sampler. `None` (fall back) if a
        /// unit has no bound texture or needs a 3D texture (not yet mapped).
        #[allow(clippy::too_many_arguments)]
        fn make_sampler_bg(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            layout: &wgpu::BindGroupLayout,
            stages: &[(&[(u8, SamplerDim)], &[crate::gpu::GxpTex])],
            gxp: &GxpRecompile,
            key: u64,
            view_cache: &mut HashMap<(u64, SamplerDim), wgpu::TextureView>,
            view_cache_bytes: &mut usize,
            samplers_by_mode: &mut HashMap<(bool, u32, u32), wgpu::Sampler>,
            force: bool,
            rendered: &HashMap<u32, wgpu::TextureView>,
            depth_rendered: &HashMap<u32, wgpu::TextureView>,
        ) -> Option<wgpu::BindGroup> {
            // Both stages' samplers share this group, in declaration order: the fragment's
            // first, the vertex's after. The layout, the WGSL and this must agree on that order
            // or a sample reads the wrong texture.
            let total: usize = stages.iter().map(|(s, _)| s.len()).sum();
            if total == 0 {
                return Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gxp-samplers-empty"),
                    layout,
                    entries: &[],
                }));
            }
            // Upload every needed texture first so the views outlive the bind-group build.
            let mut views: Vec<wgpu::TextureView> = Vec::with_capacity(total);
            // Cloning a `TextureView` is a refcount bump, so cached views are shared, not copied.
            // `(linear, addr_mode_u, addr_mode_v)` per bound view - the sampler state the guest
            // set on that texture, not a global default.
            let mut sampler_state: Vec<(bool, u32, u32)> = Vec::with_capacity(total);
            for &(samplers, textures) in stages {
            for &(unit, want) in samplers {
                let bound = textures.iter().find(|t| t.unit == unit);
                // The bound texture must actually supply the dimension the shader declared: a
                // cube sampler needs the six captured faces, a 2D sampler a single image. A
                // mismatch means the container and the guest state disagree, so bind nothing.
                let usable = bound.filter(|gt| match want {
                    SamplerDim::Cube => gt.tex.faces == 6,
                    SamplerDim::Two => gt.tex.faces == 1,
                    SamplerDim::Three => false,
                });
                // A DEPTH buffer this frame rendered, matched EXACTLY and checked FIRST.
                //
                // Order is load-bearing. A title allocates a scene's depth next to its colour
                // (one racer puts them 256 bytes apart), so `rendered_alias`, which matches by
                // range, claims the depth address for the colour target and the pass reads a
                // colour where it asked for a distance - which is why its glow, blur and
                // soft-particle passes rendered pure black. Exact-matching the depth first is
                // what tells the two apart, and the address comes from the guest's own
                // `SceGxmDepthStencilSurface`, not from a guess about the layout.
                let depth_hit = (want == SamplerDim::Two)
                    .then(|| usable.and_then(|gt| depth_rendered.get(&gt.tex.data_addr)))
                    .flatten();
                let aliased = depth_hit
                    .is_none()
                    .then(|| {
                        (want == SamplerDim::Two)
                            .then(|| usable.and_then(|gt| rendered_alias(rendered, gt.tex.data_addr, key, unit)))
                            .flatten()
                    })
                    .flatten();
                match usable {
                    Some(gt) if depth_hit.is_some() => {
                        report_depth_sample_bound(key, unit, gt.tex.data_addr);
                        views.push(depth_hit.unwrap().clone());
                        sampler_state.push((gt.tex.filter_linear, gt.tex.addr_mode_u, gt.tex.addr_mode_v));
                    }
                    // Sampling a buffer an earlier pass in THIS frame rendered: bind that
                    // render. Only 2D targets - a cube face is never a GXM render target.
                    Some(gt) if aliased.is_some() => {
                        views.push(rendered[&aliased.unwrap()].clone());
                        sampler_state.push((gt.tex.filter_linear, gt.tex.addr_mode_u, gt.tex.addr_mode_v));
                    }
                    Some(gt) => {
                        // A texture whose data pointer is null decodes to a 1x1 ZERO texel. That
                        // is the faithful substitute (see the runtime's zero-handle report), but
                        // a recompiled shader sampling it multiplies its whole output by zero,
                        // and the pass comes out BLACK with nothing in the log tying the two
                        // together. Name the pair and the unit, once each: this is a draw
                        // silently losing its content, which is the same class of event as a
                        // fallback and gets the same unconditional report.
                        if gt.tex.data_addr == 0 {
                            report_zero_texel_sample(key, unit);
                        }
                        let cache_key = (gt.tex.key, want);
                        if !view_cache.contains_key(&cache_key) {
                            // Bound the cache BY BYTES: the keys are content fingerprints,
                            // so clearing wholesale only costs a re-upload, never
                            // correctness. See `tex_cache_budget_bytes` for why counting
                            // entries did not bound anything.
                            *view_cache_bytes += texture_bytes(gt.tex.width, gt.tex.height);
                            if *view_cache_bytes >= tex_cache_budget_bytes() {
                                view_cache.clear();
                                *view_cache_bytes = 0;
                            }
                            let tex = upload_gxp_texture(device, queue, &gt.tex);
                            let view = tex.create_view(&wgpu::TextureViewDescriptor {
                                dimension: Some(want.view_dimension()),
                                ..Default::default()
                            });
                            view_cache.insert(cache_key, view);
                        }
                        views.push(view_cache[&cache_key].clone());
                        sampler_state.push((gt.tex.filter_linear, gt.tex.addr_mode_u, gt.tex.addr_mode_v));
                    }
                    // A volume sampler (not yet mapped), or a unit whose real texture we could
                    // not capture/decode: strict mode falls back; force mode binds a neutral
                    // fallback so geometry still renders (a diagnostic, never the default).
                    None => {
                        if !force {
                            report_fallback(
                                key,
                                &format!(
                                    "sampler unit {unit} wants {want:?} but the bound units are {:?}",
                                    gxp.textures
                                        .iter()
                                        .map(|t| (t.unit, t.tex.faces))
                                        .collect::<Vec<_>>()
                                ),
                            );
                            return None;
                        }
                        views.push(make_fallback_view(device, queue, want.view_dimension()));
                        sampler_state.push((false, 0, 0));
                    }
                }
            }
            }
            // Create every sampler this bind group needs FIRST, so the map is not borrowed
            // mutably while the entries hold shared references into it.
            for &st in &sampler_state {
                samplers_by_mode.entry(st).or_insert_with(|| make_gxp_sampler(device, st.0, st.1, st.2));
            }
            let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(total * 2);
            for (i, view) in views.iter().enumerate() {
                let samp = &samplers_by_mode[&sampler_state[i]];
                entries.push(wgpu::BindGroupEntry { binding: i as u32 * 2, resource: wgpu::BindingResource::TextureView(view) });
                entries.push(wgpu::BindGroupEntry { binding: i as u32 * 2 + 1, resource: wgpu::BindingResource::Sampler(samp) });
            }
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gxp-samplers"), layout, entries: &entries }))
        }
    }

    /// The texture dimension a recompiled fragment's sampler binding needs. It comes from the
    /// GXP container (the sampler parameter's cube flag plus the SMP coordinate count), and it
    /// must match on both sides: the WGSL declares the type, the bind-group layout declares the
    /// view dimension, and the bound texture must be able to supply it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum SamplerDim {
        Two,
        Three,
        Cube,
    }

    impl SamplerDim {
        fn view_dimension(self) -> wgpu::TextureViewDimension {
            match self {
                SamplerDim::Two => wgpu::TextureViewDimension::D2,
                SamplerDim::Three => wgpu::TextureViewDimension::D3,
                SamplerDim::Cube => wgpu::TextureViewDimension::Cube,
            }
        }
    }

    /// A REPEAT sampler for the recompiler path (point or linear), mirroring the
    /// fixed-function samplers.
    /// One `SceGxmTextureAddrMode` as a wgpu address mode.
    ///
    /// GXM's four "border" variants differ only in what a fetch outside the texture returns
    /// (a border colour we do not model); their EDGE behaviour is the clamp, which is what
    /// matters for the coordinates a shader actually produces. Mapping them to clamp is the
    /// closest available behaviour, and far closer than the repeat they used to get.
    fn gxm_addr_mode(mode: u32) -> wgpu::AddressMode {
        match mode {
            1 | 3 => wgpu::AddressMode::MirrorRepeat, // MIRROR, MIRROR_CLAMP
            2 | 5..=7 => wgpu::AddressMode::ClampToEdge, // CLAMP + the three border clamps
            _ => wgpu::AddressMode::Repeat,          // REPEAT, REPEAT_IGNORE_BORDER
        }
    }

    fn make_gxp_sampler(device: &wgpu::Device, linear: bool, u: u32, v: u32) -> wgpu::Sampler {
        let f = if linear { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest };
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gxp-sampler"),
            address_mode_u: gxm_addr_mode(u),
            address_mode_v: gxm_addr_mode(v),
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: f,
            min_filter: f,
            // Trilinear across the generated chain (see `build_mip_chain`). A guest that asked
            // for point filtering still gets point filtering WITHIN a level; what this changes
            // is that a minified surface reads a level sized for it instead of aliasing.
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        })
    }

    /// Box-filter a mip chain down from one RGBA8 image, and return it laid out the way
    /// `TextureDataOrder::LayerMajor` wants (every level of layer 0, then every level of layer 1).
    ///
    /// The guest's textures carry their own mip levels and the hardware samples them; we decode
    /// only level 0, so a minified surface was being point-sampled from the full-size image.
    /// That is not a subtle difference: the track receding to the horizon covers a 512-texel
    /// texture in a few dozen pixels, and every pixel lands on an unrelated texel, which reads
    /// as dense white SPECKLE over the whole distant road rather than as a road.
    fn build_mip_chain(w: u32, h: u32, layers: u32, level0: &[u8]) -> (Vec<u8>, u32) {
        let levels = 32 - w.max(h).max(1).leading_zeros();
        let layer_texels = (w as usize) * (h as usize);
        let mut out = Vec::with_capacity(level0.len() * 2);
        for layer in 0..layers as usize {
            let base = layer * layer_texels * 4;
            let mut src: Vec<u8> = level0[base..base + layer_texels * 4].to_vec();
            let (mut sw, mut sh) = (w, h);
            out.extend_from_slice(&src);
            for _ in 1..levels {
                let (dw, dh) = ((sw / 2).max(1), (sh / 2).max(1));
                let mut dst = vec![0u8; (dw as usize) * (dh as usize) * 4];
                for y in 0..dh as usize {
                    for x in 0..dw as usize {
                        for c in 0..4usize {
                            // Average the up-to-four source texels this one covers. On an odd
                            // dimension the second sample repeats the first, which is the
                            // ordinary way a box filter handles a non-power-of-two level.
                            let x0 = (2 * x).min(sw as usize - 1);
                            let x1 = (2 * x + 1).min(sw as usize - 1);
                            let y0 = (2 * y).min(sh as usize - 1);
                            let y1 = (2 * y + 1).min(sh as usize - 1);
                            let at = |xx: usize, yy: usize| src[(yy * sw as usize + xx) * 4 + c] as u32;
                            dst[(y * dw as usize + x) * 4 + c] =
                                ((at(x0, y0) + at(x1, y0) + at(x0, y1) + at(x1, y1) + 2) / 4) as u8;
                        }
                    }
                }
                out.extend_from_slice(&dst);
                src = dst;
                sw = dw;
                sh = dh;
            }
        }
        (out, levels)
    }

    /// Upload a decoded [`GxmTexture`] (linear RGBA8) to a GPU texture for the recompiler path.
    fn upload_gxp_texture(device: &wgpu::Device, queue: &wgpu::Queue, t: &GxmTexture) -> wgpu::Texture {
        let (w, h) = (t.width.max(1), t.height.max(1));
        // A cube map uploads as six array layers; the view below then reads them as a cube.
        let layers = t.faces.max(1);
        // Guard against a short pixel buffer (a not-fully-decoded format): pad to w*h*layers*4.
        let need = (w as usize) * (h as usize) * (layers as usize) * 4;
        let data: std::borrow::Cow<[u8]> = if t.rgba.len() >= need {
            std::borrow::Cow::Borrowed(&t.rgba[..need])
        } else {
            let mut v = t.rgba.to_vec();
            v.resize(need, 0);
            std::borrow::Cow::Owned(v)
        };
        let (data, mip_level_count) = if std::env::var("VITASLOP_GXP_MIPS").ok().as_deref() == Some("0") {
            (data.into_owned(), 1)
        } else {
            build_mip_chain(w, h, layers, &data)
        };
        device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("gxp-tex"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: layers },
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // A GAMMA-CORRECT texture is sRGB-DECODED by the hardware sampler before
                // filtering. Uploading the same bytes through an sRGB format puts the decode
                // in exactly that place, so nothing downstream has to know.
                format: if t.gamma {
                    wgpu::TextureFormat::Rgba8UnormSrgb
                } else {
                    wgpu::TextureFormat::Rgba8Unorm
                },
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &data,
        )
    }

    /// A 1x1 (or 1x1x1) neutral-grey fallback texture view of the given dimension, for the
    /// `VITASLOP_GXP_FORCE` diagnostic: bound where a sampler's real texture is unavailable so
    /// the recompiled geometry still renders. Opaque alpha so an alpha-test does not discard.
    fn make_fallback_view(device: &wgpu::Device, queue: &wgpu::Queue, dim: wgpu::TextureViewDimension) -> wgpu::TextureView {
        let is_3d = dim == wgpu::TextureViewDimension::D3;
        let tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("gxp-fallback"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: if is_3d { wgpu::TextureDimension::D3 } else { wgpu::TextureDimension::D2 },
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[180, 180, 180, 255],
        );
        tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(dim), ..Default::default() })
    }

    /// Build a group0/group1 uniform bind group from raw guest SA bytes, sized to the WGSL
    /// `array<vec4<f32>, ceil(lanes/4)>` and zero-padded. An empty (0-lane) stage gets an
    /// empty bind group so the pipeline layout's group is still satisfied at draw time.
    /// Append one stage's SA block to the pass's uniform arena at dynamic-offset alignment,
    /// zero-padded to the `array<vec4<u32>, ceil(lanes/4)>` the WGSL declares, and return its
    /// byte offset.
    ///
    /// A zero-lane stage still gets an offset (0) because a bind group with no entries takes
    /// no dynamic offsets at all - the value is simply never used.
    fn push_sa(udata: &mut Vec<u8>, lanes: u32, guest: &[u8], align: u64) -> u32 {
        if lanes == 0 {
            return 0;
        }
        while udata.len() as u64 % align != 0 {
            udata.push(0);
        }
        let off = udata.len() as u32;
        let need = (lanes.div_ceil(4) as usize) * 16;
        let n = guest.len().min(need);
        udata.extend_from_slice(&guest[..n]);
        udata.resize(off as usize + need, 0);
        off
    }

    fn make_uniform_bg(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, lanes: u32, guest: &[u8]) -> wgpu::BindGroup {
        if lanes == 0 {
            return device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gxp-ubo-empty"), layout, entries: &[] });
        }
        let need = (lanes.div_ceil(4) as usize) * 16;
        let mut data = vec![0u8; need];
        let n = guest.len().min(need);
        data[..n].copy_from_slice(&guest[..n]);
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gxp-ubo"),
            contents: &data,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gxp-ubo-bind"),
            layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
        })
    }

    /// Convert an IEEE-754 half (binary16) to f32.
    fn half_to_f32(h: u16) -> f32 {
        let sign = (h >> 15) & 1;
        let exp = (h >> 10) & 0x1f;
        let mant = h & 0x3ff;
        let v = if exp == 0 {
            (mant as f32) * (1.0 / 16_777_216.0) // subnormal: mant * 2^-24
        } else if exp == 0x1f {
            if mant == 0 { f32::INFINITY } else { f32::NAN }
        } else {
            (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
        };
        if sign == 1 { -v } else { v }
    }

    /// Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per
    /// shader pair, every value the guest actually fed that draw - each declared uniform
    /// decoded through its own declared type, and the observed range of every vertex
    /// attribute component over the draw's whole stream.
    ///
    /// This exists because the two halves of a wrong picture are indistinguishable from the
    /// frame: a coordinate that comes out wrong is either a uniform we are reading through the
    /// wrong layout or an attribute we are decoding wrong, and every other diagnostic here
    /// (`_PROBE`, `_VPROBE`, `_KEYCOLOR`) observes the shader's OUTPUT, which is downstream of
    /// both. A uniform printed as its parameter declares it - `mainScaleBias F16[4] = (1, 1,
    /// 0.25, 0.25)` - settles in one line what a register probe cannot settle at all.
    fn report_inputs(key: u64, gxp: &GxpRecompile) {
        let Ok(spec) = std::env::var("VITASLOP_GXP_INPUTS") else { return };
        let all = spec == "all";
        if !all && !spec.split(',').any(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16) == Ok(key)) {
            return;
        }
        // Dedupe on the pair AND on the inputs themselves, not on the pair alone: one pair is
        // submitted many times a frame with DIFFERENT uniforms (that is what a per-draw uniform
        // buffer is for), and reporting only the first submission is how a diagnostic ends up
        // describing a draw that is not the one being investigated.
        use std::hash::{Hash, Hasher};
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(u64, u64)>>> = OnceLock::new();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        gxp.vert_sa.hash(&mut h);
        gxp.frag_sa.hash(&mut h);
        for t in gxp.textures.iter().chain(gxp.vertex_textures.iter()) {
            (t.unit, t.tex.data_addr, t.tex.width, t.tex.height).hash(&mut h);
        }
        let inputs_hash = h.finish();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert((key, inputs_hash)) {
            return;
        }
        for (stage, bytes, blob) in
            [("vertex", &gxp.vert_sa, &gxp.vprog), ("fragment", &gxp.frag_sa, &gxp.fprog)]
        {
            let Ok(program) = vitaslop_gxp_shader::Program::parse(blob) else { continue };
            // The raw words too, not only the decoded parameters. A parameter is decoded through
            // its declared offset/type, so a value that reads wrong is either the bytes or that
            // decode - and only the bytes can tell the two apart.
            report!(
                "gxp inputs {key:016x} {stage}: default uniform buffer is {} bytes for {} declared registers, raw = {}",
                bytes.len(),
                program.default_uniform_regs,
                bytes
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        format!("{:08x}", u32::from_le_bytes(w))
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            for p in &program.parameters {
                use vitaslop_gxp_shader::container::ParamCategory;
                if p.category != ParamCategory::Uniform {
                    continue;
                }
                let vals = decode_uniform(bytes, p);
                report!(
                    "gxp inputs {key:016x} {stage}:   {} {:?}[{}] at reg {} = {}",
                    p.name, p.ptype, p.component_count, p.resource_index, vals
                );
            }
        }
        // The guest's VIEWPORT, which is what turns the vertex program's clip output into
        // target pixels. A post-process pass that samples a source at a scale/bias only lands
        // right if the source was RENDERED where the pass thinks it was, and the viewport is
        // the one piece of that which is neither in the shader nor in the uniforms.
        report!(
            "gxp inputs {key:016x} viewport: xOffset={} xScale={} yOffset={} yScale={} zOffset={} zScale={}{}",
            gxp.viewport[0], gxp.viewport[1], gxp.viewport[2],
            gxp.viewport[3], gxp.viewport[4], gxp.viewport[5],
            if gxp.viewport.iter().all(|v| *v == 0.0) { "  (all zero = the guest left the default)" } else { "" }
        );
        // What each declared sampler is actually bound to. A post-process pass's whole content
        // is the geometry of the buffer it reads, and "unit 1" in a shader and "unit 1" in the
        // guest's texture state are only the same thing if the binding says so.
        //
        // BOTH stages, and the vertex one is not a footnote: a vertex program that samples
        // builds its GEOMETRY from what it reads, so an unbound or empty vertex texture is a
        // draw with no mesh rather than a surface with the wrong colours. Leaving it out of
        // this report is what left "the campaign map body is black" a content question nobody
        // could answer - the fragment side said everything was bound, and it was.
        for (stage, blob, bound) in [
            ("sampler", &gxp.fprog, &gxp.textures),
            ("vertex sampler", &gxp.vprog, &gxp.vertex_textures),
        ] {
            let Ok(program) = vitaslop_gxp_shader::Program::parse(blob) else { continue };
            for p in &program.parameters {
                if p.category != vitaslop_gxp_shader::container::ParamCategory::Sampler {
                    continue;
                }
                match bound.iter().find(|t| t.unit as i32 == p.resource_index) {
                    // The first few DECODED texels, not just the binding. "Which texture" and
                    // "what is in it" are different questions, and for a texture the shader
                    // reads as DATA rather than as a picture - a vector canvas, a lookup
                    // table, an index map - the second one is the whole answer, and the
                    // binding line alone has already sent one investigation down the wrong
                    // road. Also says outright when every sampled texel is the same value.
                    Some(t) => report!(
                        "gxp inputs {key:016x} {stage}: {} unit {} <- {:#x} {}x{} faces={} \
                         fmt={:#04x} swz={:#x} filter={} wrap=({},{}) texels[0..4]={:?} {}{}",
                        p.name,
                        p.resource_index,
                        t.tex.data_addr,
                        t.tex.width,
                        t.tex.height,
                        t.tex.faces,
                        t.tex.base_format,
                        t.tex.swizzle,
                        if t.tex.filter_linear { "linear" } else { "point" },
                        t.tex.addr_mode_u,
                        t.tex.addr_mode_v,
                        t.tex.rgba.chunks_exact(4).take(4).map(|c| [c[0], c[1], c[2], c[3]]).collect::<Vec<_>>(),
                        channel_spread(&t.tex.rgba),
                        match t.tex.rgba.chunks_exact(4).next() {
                            Some(first) if t.tex.rgba.chunks_exact(4).all(|c| c == first) =>
                                " (EVERY texel identical - this texture carries no image)",
                            _ => "",
                        }
                    ),
                    None => report!(
                        "gxp inputs {key:016x} {stage}: {} unit {} <- NOTHING BOUND",
                        p.name, p.resource_index
                    ),
                }
                // `VITASLOP_GXP_INPUTS_DIR=<dir>`: the decoded texture itself, as a PNG named by
                // the parameter. A channel spread says a channel is flat; only the image says
                // whether the picture in the other three is the one the shader expects, and for
                // a texture read as DATA (a displacement canvas, a lookup table) that is the
                // difference between "we decoded it wrong" and "the producer never ran".
                if let (Ok(dir), Some(t)) =
                    (std::env::var("VITASLOP_GXP_INPUTS_DIR"), bound.iter().find(|t| t.unit as i32 == p.resource_index))
                {
                    let _ = std::fs::create_dir_all(&dir);
                    let safe: String = p.name.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
                    // Raw RGBA8 with the dimensions in the NAME, not a PNG: this crate has no
                    // image encoder and must not grow one for a diagnostic. The size is in the
                    // filename so the bytes are self-describing.
                    let path = std::path::Path::new(&dir).join(format!(
                        "{key:016x}_{}_u{}_{safe}_{}x{}.rgba",
                        stage.replace(' ', "-"),
                        p.resource_index,
                        t.tex.width,
                        t.tex.height * t.tex.faces
                    ));
                    let _ = std::fs::write(path, t.tex.rgba.as_slice());
                }
            }
        }
        // Attribute RANGES, not the first vertex: a scale/bias question is a question about the
        // span of the coordinate across the mesh, and one vertex cannot answer it.
        let Ok(vprogram) = vitaslop_gxp_shader::Program::parse(&gxp.vprog) else { return };
        let stride = gxp.vertex_stride.max(1) as usize;
        let nverts = gxp.vertices.len() / stride;
        for a in &gxp.attributes {
            let name = vprogram
                .parameters
                .iter()
                .find(|p| {
                    p.category == vitaslop_gxp_shader::container::ParamCategory::Attribute
                        && p.resource_index == a.reg_index as i32
                })
                .map(|p| p.name.as_str())
                .unwrap_or("<unnamed>");
            let comps = a.components.clamp(1, 4) as usize;
            let mut lo = [f32::INFINITY; 4];
            let mut hi = [f32::NEG_INFINITY; 4];
            for v in 0..nverts {
                for c in 0..comps {
                    let f = read_attr_component(&gxp.vertices, v * stride + a.offset as usize, a.gxm_format, c);
                    lo[c] = lo[c].min(f);
                    hi[c] = hi[c].max(f);
                }
            }
            let ranges: Vec<String> =
                (0..comps).map(|c| format!("[{:.4}, {:.4}]", lo[c], hi[c])).collect();
            report!(
                "gxp inputs {key:016x} attribute: {name} lane {} at byte {} of a {}-byte vertex, \
                 fmt {} x{} over {nverts} vertices = {}",
                a.reg_index,
                a.offset,
                gxp.vertex_stride,
                a.gxm_format,
                comps,
                ranges.join(" ")
            );
        }
        // Per-VERTEX values, when the caller named this pair explicitly rather than asking for
        // the whole frame. A component RANGE only pins down an attribute if the mesh maps it
        // affinely onto the screen, and a post-process DISTORTION GRID is exactly the mesh that
        // does not - so on a small mesh, print the vertices themselves and let the reader see
        // the mapping instead of assuming one.
        if all || nverts > MAX_DUMPED_VERTICES {
            return;
        }
        for v in 0..nverts {
            let cols: Vec<String> = gxp
                .attributes
                .iter()
                .map(|a| {
                    let comps = a.components.clamp(1, 4) as usize;
                    let vals: Vec<String> = (0..comps)
                        .map(|c| {
                            format!(
                                "{:.4}",
                                read_attr_component(&gxp.vertices, v * stride + a.offset as usize, a.gxm_format, c)
                            )
                        })
                        .collect();
                    format!("lane{}=({})", a.reg_index, vals.join(","))
                })
                .collect();
            // The RAW vertex bytes too. A guest vertex often carries fields no declared attribute
            // names, and an attribute read at the wrong byte offset produces plausible numbers -
            // so the only way to tell "this really is the mesh's UV" from "this is some other
            // field that happens to look like one" is to see the whole record.
            let raw: String = gxp
                .vertices
                .get(v * stride..(v + 1) * stride)
                .map(|b| b.chunks(4).map(|c| {
                    let mut w = [0u8; 4];
                    w[..c.len()].copy_from_slice(c);
                    format!("{}", f32::from_le_bytes(w))
                }).collect::<Vec<_>>().join(" "))
                .unwrap_or_default();
            report!("gxp inputs {key:016x} vertex {v}: {}   raw-as-f32 [{raw}]", cols.join(" "));
        }
    }

    /// Most vertices `VITASLOP_GXP_INPUTS` will print individually. A post-process grid or a UI
    /// quad is small enough to read; a world mesh is not, and dumping one would bury the frame's
    /// other reports in a megabyte of numbers.
    const MAX_DUMPED_VERTICES: usize = 512;

    /// Diagnostic (`VITASLOP_GXP_SA=<key>:<v|f>:<reg>=<hexword>[,...]`): replace a default-uniform
    /// register with a chosen 32-bit word before the draw is submitted.
    ///
    /// This is the causality half of [`report_inputs`]. Reading a uniform tells you what a shader
    /// was fed; it does not tell you whether that value is what makes the picture wrong, because
    /// every downstream stage is a candidate too. Substituting the value and re-rendering is the
    /// one experiment that separates them, and it takes one run instead of a session of reasoning.
    /// A substitution is REPORTED, once per (pair, stage, register): a run whose frame came from
    /// values the guest never wrote must never be mistaken for a run of the real thing.
    fn override_sa<'a>(key: u64, stage: char, bytes: &'a [u8]) -> std::borrow::Cow<'a, [u8]> {
        let Ok(spec) = std::env::var("VITASLOP_GXP_SA") else {
            return std::borrow::Cow::Borrowed(bytes);
        };
        let mut out = std::borrow::Cow::Borrowed(bytes);
        for item in spec.split(',').filter(|s| !s.trim().is_empty()) {
            let parts: Vec<&str> = item.trim().split(':').collect();
            let [k, st, assign] = parts[..] else {
                panic!("VITASLOP_GXP_SA item {item:?} is not <key>:<v|f>:<reg>=<hexword>");
            };
            let Ok(want_key) = u64::from_str_radix(k.trim_start_matches("0x"), 16) else {
                panic!("VITASLOP_GXP_SA item {item:?} has a non-hex pair key");
            };
            let want_stage = match st {
                "v" => 'v',
                "f" => 'f',
                other => panic!("VITASLOP_GXP_SA item {item:?} names stage {other:?}, not v or f"),
            };
            let Some((reg, word)) = assign.split_once('=') else {
                panic!("VITASLOP_GXP_SA item {item:?} is missing the =<hexword>");
            };
            let (Ok(reg), Ok(word)) = (
                reg.trim().parse::<usize>(),
                u32::from_str_radix(word.trim().trim_start_matches("0x"), 16),
            ) else {
                panic!("VITASLOP_GXP_SA item {item:?} has a bad register or word");
            };
            if want_key != key || want_stage != stage {
                continue;
            }
            let buf = out.to_mut();
            if buf.len() < (reg + 1) * 4 {
                buf.resize((reg + 1) * 4, 0);
            }
            buf[reg * 4..reg * 4 + 4].copy_from_slice(&word.to_le_bytes());
            report_sa_override(key, stage, reg, word);
        }
        out
    }

    /// Report - once per (pair, stage, register) - that a uniform register was substituted by
    /// `VITASLOP_GXP_SA`. See [`override_sa`] for why this is never silent.
    fn report_sa_override(key: u64, stage: char, reg: usize, word: u32) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(u64, char, usize)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert((key, stage, reg)) {
            report!(
                "gxp pair {key:016x}: {} uniform register {reg} SUBSTITUTED with {word:#010x} - \
                 this frame is NOT what the guest asked for",
                if stage == 'v' { "vertex" } else { "fragment" }
            );
        }
    }

    /// One uniform parameter's values, read out of the raw default-uniform-buffer bytes through
    /// its OWN declared type. `resource_index` is a 4-byte register offset; the components are
    /// packed from there at the type's own component width, which is how an F16 float4 fits in
    /// two registers and an F32 float4 needs four.
    fn decode_uniform(bytes: &[u8], p: &vitaslop_gxp_shader::container::Parameter) -> String {
        use vitaslop_gxp_shader::container::ParamType;
        let Some(width) = p.ptype.component_bytes() else {
            return format!("<{:?} has no fixed component width>", p.ptype);
        };
        let base = (p.resource_index.max(0) as usize) * 4;
        let n = p.component_count as usize * p.array_size.max(1) as usize;
        let mut out: Vec<String> = Vec::with_capacity(n);
        for i in 0..n {
            let o = base + i * width as usize;
            let Some(raw) = bytes.get(o..o + width as usize) else {
                out.push("<past the end of the buffer>".to_string());
                break;
            };
            out.push(match p.ptype {
                ParamType::F32 => format!("{}", f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
                ParamType::F16 => format!("{}", half_to_f32(u16::from_le_bytes([raw[0], raw[1]]))),
                ParamType::U32 => format!("{}", u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
                ParamType::S32 => format!("{}", i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
                ParamType::U16 | ParamType::C10 => format!("{}", u16::from_le_bytes([raw[0], raw[1]])),
                ParamType::S16 => format!("{}", i16::from_le_bytes([raw[0], raw[1]])),
                ParamType::U8 => format!("{}", raw[0]),
                ParamType::S8 => format!("{}", raw[0] as i8),
                ParamType::Aggregate | ParamType::Unknown(_) => unreachable!("no component width"),
            });
        }
        format!("({})", out.join(", "))
    }

    /// Per-channel min/max over every texel of a decoded texture.
    ///
    /// "The first four texels" answers what a texture looks like at its corner; it does not answer
    /// whether a channel carries any signal at all, and for a texture a shader reads as DATA that
    /// is the only question. The map body's vertex program displaces every vertex by this
    /// texture's ALPHA, so `a[0,0]` and `a[0,255]` are two completely different bugs and the
    /// corner texel cannot tell them apart.
    fn channel_spread(rgba: &[u8]) -> String {
        let mut lo = [255u8; 4];
        let mut hi = [0u8; 4];
        let mut texels = 0usize;
        for c in rgba.chunks_exact(4) {
            texels += 1;
            for i in 0..4 {
                lo[i] = lo[i].min(c[i]);
                hi[i] = hi[i].max(c[i]);
            }
        }
        if texels == 0 {
            return "spread=(no texels)".into();
        }
        format!(
            "spread over {texels} texels r[{},{}] g[{},{}] b[{},{}] a[{},{}]",
            lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], lo[3], hi[3]
        )
    }

    /// Byte size of one component of a `SceGxmAttributeFormat`.
    fn attr_component_size(gxm_format: u8) -> usize {
        match gxm_format {
            0 | 1 | 4 | 5 => 1,     // U8 / S8 / U8N / S8N
            2 | 3 | 6 | 7 | 8 => 2, // U16 / S16 / U16N / S16N / F16
            _ => 4,                 // F32 (9) and any unknown -> treat as 4-byte float
        }
    }

    /// Read one attribute component from the guest stream and convert it to f32, matching the
    /// GXM fixed-function vertex fetch (normalized formats scale to [0,1]/[-1,1], F16 expands).
    /// Out-of-range reads yield 0.0 (a benign over-read on a short buffer, never a panic).
    fn read_attr_component(buf: &[u8], base: usize, gxm_format: u8, c: usize) -> f32 {
        let o = base + c * attr_component_size(gxm_format);
        let u16at = |o: usize| -> Option<u16> { buf.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]])) };
        match gxm_format {
            9 => buf.get(o..o + 4).map(|s| f32::from_le_bytes([s[0], s[1], s[2], s[3]])).unwrap_or(0.0),
            8 => u16at(o).map(half_to_f32).unwrap_or(0.0),
            4 => buf.get(o).map(|&b| b as f32 / 255.0).unwrap_or(0.0),
            5 => buf.get(o).map(|&b| (b as i8 as f32 / 127.0).max(-1.0)).unwrap_or(0.0),
            6 => u16at(o).map(|v| v as f32 / 65535.0).unwrap_or(0.0),
            7 => u16at(o).map(|v| (v as i16 as f32 / 32767.0).max(-1.0)).unwrap_or(0.0),
            0 => buf.get(o).map(|&b| b as f32).unwrap_or(0.0),
            1 => buf.get(o).map(|&b| b as i8 as f32).unwrap_or(0.0),
            2 => u16at(o).map(|v| v as f32).unwrap_or(0.0),
            3 => u16at(o).map(|v| v as i16 as f32).unwrap_or(0.0),
            _ => 0.0,
        }
    }

    /// FNV-1a over a byte slice, consuming EIGHT bytes per multiply instead of one.
    ///
    /// The distinction matters because this runs over megabytes a frame: every draw hashes its
    /// shader pair's two container blobs to find its pipeline, and its whole vertex stream to
    /// find its repacked geometry. A byte-at-a-time FNV over that is milliseconds of a
    /// twenty-millisecond frame. The trailing bytes are folded in individually so the result
    /// still depends on the exact length.
    fn fnv64(seed: u64, bytes: &[u8]) -> u64 {
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = seed;
        let (words, tail) = bytes.split_at(bytes.len() & !7);
        for w in words.chunks_exact(8) {
            h ^= u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]);
            h = h.wrapping_mul(PRIME);
        }
        for &b in tail {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
        h ^ bytes.len() as u64
    }

    /// Repack a guest vertex stream into the tightly-packed `Float32xN` layout the recompiled
    /// pipeline expects. One packed vertex per guest vertex, in order, so the index buffer is
    /// unchanged.
    ///
    /// Appends into a caller-owned arena rather than returning a `Vec`: the arena is uploaded
    /// once per pass, so a per-draw allocation here would put back exactly the cost the arena
    /// exists to remove.
    fn repack_vertices_into(
        vertices: &[u8],
        guest_stride: u32,
        repack: &[RepackAttr],
        packed_stride: u32,
        out: &mut Vec<u8>,
    ) {
        let gstride = guest_stride.max(1) as usize;
        let nverts = vertices.len() / gstride;
        out.reserve(nverts * packed_stride as usize);
        for i in 0..nverts {
            let vbase = i * gstride;
            // Zero-fill this packed vertex, then write each attribute at its packed offset.
            let start = out.len();
            out.resize(start + packed_stride as usize, 0);
            for a in repack {
                for c in 0..a.components as usize {
                    let f = read_attr_component(vertices, vbase + a.guest_offset as usize, a.gxm_format, c);
                    let po = start + a.packed_offset as usize + c * 4;
                    out[po..po + 4].copy_from_slice(&f.to_le_bytes());
                }
            }
        }
    }

    /// The depth encoding to assume for a pass whose own draws could not produce one: `-1/w`,
    /// i.e. `a = 0, c = -1`. It is the encoding with no projection in it, so a frame built on
    /// it is visibly a fallback rather than a plausible-looking guess.
    const DEPTH_FIT_RECIP_W: (f32, f32) = (0.0, -1.0);

    /// What interpreting a pair's vertex program over its OWN mesh says about the projection
    /// behind it. See [`count_clip_w_signs`].
    #[derive(Clone, Copy, Debug, Default)]
    struct ClipStats {
        /// Sampled vertices landing inside the frustum with clip `w > 0`, and with `w < 0`.
        in_front: usize,
        behind: usize,
        sampled: usize,
        /// NDC bounding box under `x/|w|`.
        bbox: [f32; 4],
        /// `(a, c)` of `z_clip = a * w_clip + c`, and the spread of `w` the fit was taken over.
        ///
        /// A projection matrix makes clip `z` an AFFINE function of clip `w` - both are the same
        /// dot product against the same point, differing only in which matrix column they use -
        /// so this holds EXACTLY, per vertex, for every draw sharing one projection. It is what
        /// gives the guest's own window depth `z/w = a + c/w` without reflecting the matrix, and
        /// therefore what lets a depth surface be re-encoded the way the guest wrote it rather
        /// than the way we guessed. `None` when this draw has no two vertices with different
        /// `w` to fit through.
        depth_fit: Option<(f32, f32)>,
        w_spread: f32,
    }

    /// Interpret a pair's vertex program over its OWN mesh and measure what it says about the
    /// projection: how many vertices land in front of the eye and how many behind, and how clip
    /// `z` relates to clip `w`. `None` when the program cannot be interpreted at all.
    ///
    /// Sampled with an even STRIDE through the mesh rather than as a prefix: vertex 0 of a
    /// world mesh is routinely behind the camera, so a prefix answers a different question
    /// than "does this draw cover anything".
    fn count_clip_w_signs(gxp: &GxpRecompile, max_samples: usize) -> Option<ClipStats> {
        let vrc = vitaslop_gxp_shader::recompile_vertex(&gxp.vprog).ok()?;
        let mut base = vitaslop_gxp_shader::interp::RegFile::with_lanes(512);
        for (k, c) in gxp.vert_sa.chunks_exact(4).enumerate() {
            if k < base.sa.len() {
                base.sa[k] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        // The secondary program (and the container's literals) run first and overwrite SA
        // registers the primary reads, so skipping them interprets the primary against
        // uniforms the real module has already replaced.
        if let Ok(program) = vitaslop_gxp_shader::Program::parse(&gxp.vprog) {
            for &(reg, value) in &program.literals {
                if let Some(slot) = base.sa.get_mut(reg as usize) {
                    *slot = f32::from_bits(value);
                }
            }
            let secondary = vitaslop_gxp_shader::usse::decode_secondary_shader(&program);
            if vitaslop_gxp_shader::interp::run(&secondary, &mut base).is_err() {
                return None;
            }
        }
        let stride = gxp.vertex_stride.max(1) as usize;
        let nverts = gxp.vertices.len() / stride;
        if nverts == 0 {
            return None;
        }
        let step = (nverts / max_samples.max(1)).max(1);
        // The VERTEX stage's own textures, so a program that builds its geometry from a fetch
        // can be interpreted at all. Nearest, wrapped, no mip: this is a measurement of where
        // the geometry LANDS, and a filter would change nothing about that while costing four
        // texel reads per sample.
        let fetch = |unit: u8, coord: [f32; 4]| -> Option<[f32; 4]> {
            let t = &gxp.vertex_textures.iter().find(|t| t.unit == unit)?.tex;
            let (w, h) = (t.width.max(1), t.height.max(1));
            let wrap = |v: f32, n: u32| {
                let n = n as f32;
                let p = (v * n).floor().rem_euclid(n);
                p as u32
            };
            let (x, y) = (wrap(coord[0], w), wrap(coord[1], h));
            let o = ((y * w + x) * 4) as usize;
            let px = t.rgba.get(o..o + 4)?;
            Some([
                px[0] as f32 / 255.0,
                px[1] as f32 / 255.0,
                px[2] as f32 / 255.0,
                px[3] as f32 / 255.0,
            ])
        };
        // Which PA lanes an attribute actually supplies (see the default fill below).
        let mut claimed = vec![false; base.pa.len()];
        for a in &gxp.attributes {
            for c in 0..a.components as usize {
                if let Some(slot) = claimed.get_mut(a.reg_index as usize + c) {
                    *slot = true;
                }
            }
        }
        let (mut in_front, mut behind, mut sampled) = (0usize, 0usize, 0usize);
        let mut bbox = [f32::MAX, f32::MIN, f32::MAX, f32::MIN];
        // Least-squares accumulators for `z = a*w + c` over the sampled vertices, in f64: the
        // fit's `c` is a small difference between two large dot products (on one retail racer,
        // `z` and `w` agree to four decimal places and `c` is about -1), and in f32 that
        // subtraction loses most of its significant digits.
        let (mut sw, mut sz, mut sww, mut swz) = (0f64, 0f64, 0f64, 0f64);
        let (mut wlo, mut whi) = (f32::MAX, f32::MIN);
        for v in (0..nverts).step_by(step) {
            let mut regs = base.clone();
            for a in &gxp.attributes {
                let vbase = v * stride + a.offset as usize;
                // All FOUR lanes of the attribute's register, because that is what the pipeline
                // feeds the real shader: the linked module reads `in.aN.xyzw`, and WebGPU fills
                // the components a `Float32x3` vertex format does not supply with (0,0,0,1). An
                // interpretation that zero-fills instead reads `w = 0` where the GPU reads 1,
                // which on a 3-component POSITION drops the projection's translation term - i.e.
                // it measures a different clip `w` than the frame it is supposed to explain.
                for c in 0..4usize {
                    let lane = a.reg_index as usize + c;
                    if lane >= regs.pa.len() {
                        continue;
                    }
                    if c < a.components as usize {
                        regs.pa[lane] = read_attr_component(&gxp.vertices, vbase, a.gxm_format, c);
                    } else if !claimed[lane] {
                        // Only where no OTHER attribute owns the lane - two attributes packed
                        // two lanes apart would otherwise overwrite each other's components.
                        regs.pa[lane] = if c == 3 { 1.0 } else { 0.0 };
                    }
                }
            }
            if vitaslop_gxp_shader::interp::run_watching_for_nan_with_textures(
                &vrc.shader,
                &mut regs,
                &fetch,
            )
            .is_err()
            {
                return None;
            }
            sampled += 1;
            // Count the vertices this draw would actually PUT ON SCREEN under each reading of
            // the sign, rather than the sign alone. A mesh that straddles the camera plane has
            // both signs in it, and a vote on the sign alone leaves it uncorrected - which
            // clips away exactly the half that is in view and keeps the half behind the eye.
            // Inside the frustum means `|x| <= |w|` and `|y| <= |w|` on the side being tested.
            let (x, y, w) = (regs.o[0], regs.o[1], regs.o[3]);
            let inside = x.abs() <= w.abs() && y.abs() <= w.abs();
            if w > 0.0 && inside {
                in_front += 1;
            } else if w < 0.0 && inside {
                behind += 1;
            }
            // Where the draw lands once the sign is corrected: `x/|w|`, which is the same NDC
            // either reading resolves to for the vertices that are actually on screen.
            if w != 0.0 {
                let (nx, ny) = (x / w.abs(), y / w.abs());
                bbox = [bbox[0].min(nx), bbox[1].max(nx), bbox[2].min(ny), bbox[3].max(ny)];
            }
            let z = regs.o[2];
            if w.is_finite() && z.is_finite() {
                let (wd, zd) = (w as f64, z as f64);
                sw += wd;
                sz += zd;
                sww += wd * wd;
                swz += wd * zd;
                wlo = wlo.min(w);
                whi = whi.max(w);
            }
        }
        let n = sampled as f64;
        let denom = n * sww - sw * sw;
        // A mesh whose vertices all share one `w` (a 2D overlay, a billboard) pins no line
        // through them, and inventing one from a degenerate system would put a wild `c` into
        // the depth encoding. Requiring a real spread of `w` is what keeps such a draw from
        // being asked a question it cannot answer.
        let depth_fit = (sampled >= 2 && denom.abs() > 1e-9 && (whi - wlo).abs() > 1e-3).then(|| {
            let a = (n * swz - sw * sz) / denom;
            let c = (sz - a * sw) / n;
            (a as f32, c as f32)
        });
        Some(ClipStats {
            in_front,
            behind,
            sampled,
            bbox,
            depth_fit,
            w_spread: if whi > wlo { whi - wlo } else { 0.0 },
        })
    }

    /// Weigh, ONCE per shader pair, how many of that pair's vertices land in the frustum with
    /// each sign of clip `w`. Reports what it saw unconditionally: a correction this large
    /// that applied itself silently would be indistinguishable from a faithful projection by
    /// looking at the frame.
    ///
    /// `None` = the pair's vertex program does not interpret at all, so it carries no evidence
    /// and never will.
    fn measure_clip(gxp: &GxpRecompile, key: u64) -> Option<ClipStats> {
        match count_clip_w_signs(gxp, 256) {
            Some(s) => {
                report!(
                    "gxp clip: key {key:x}: of {} sampled vertices, {} land in the frustum with w>0 \
                     and {} with w<0; ndc x[{:.2},{:.2}] y[{:.2},{:.2}]; z = {}",
                    s.sampled,
                    s.in_front,
                    s.behind,
                    s.bbox[0],
                    s.bbox[1],
                    s.bbox[2],
                    s.bbox[3],
                    match s.depth_fit {
                        Some((a, c)) => format!("{a}*w + {c} over w spread {}", s.w_spread),
                        None => "(no w spread to fit through)".into(),
                    }
                );
                Some(s)
            }
            None => {
                report!(
                    "gxp clip: key {key:x}: could not be measured (the vertex program does not \
                     interpret) - it contributes no evidence about this pass's projection"
                );
                None
            }
        }
    }

    /// Inject the GXM->WebGPU clip fixup into a linked module: wrap the vertex stage's
    /// `out.position` assignment in a helper that remaps clip Z (and optionally flips Y).
    ///
    /// The linker emits exactly one `  out.position = <expr>;` statement, but the exact shape of
    /// `<expr>` is the emitter's business and has changed. So this matches the STATEMENT and
    /// wraps whatever it assigns. Returns `None` when the statement is not found, which makes
    /// the pair fall back rather than render with no depth remap at all: this transform is not
    /// cosmetic, it is what puts the guest's clip depth inside WebGPU's `0 <= z <= w` clip
    /// volume, and without it the hardware clips away every triangle whose raw clip z runs past
    /// w (this title's does, by roughly 5x) - which looks like a mesh mysteriously missing its
    /// far half, not like a broken depth buffer.
    fn inject_clip_fixup(wgsl: &str, zfix: ZFix, yflip: bool, solid: bool, keycolor: Option<u64>) -> Option<String> {
        // Replace the guest's clip z with the SAME depth the fixed-function path writes, so
        // recompiled and fixed-function draws share one comparable depth buffer: the projected
        // view distance through `-1/w`, mapped linearly onto [0,1] over the scene's visible
        // range (see `render::project` for why the guest's own clip z is not a depth here).
        // Keeping xy exact leaves the real shader's projection untouched. w<=0 (behind the eye)
        // is left to wgpu's clip.
        let z = match zfix {
            ZFix::Range => {
                "  if (c.w > 0.0) { let q = -1.0 / c.w;\n    r.z = clamp((q - gxp_depth.range.x) * gxp_depth.range.y, 0.0, 1.0) * c.w; }\n"
            }
            // The ORDINARY GL->WebGPU depth remap: the guest's own clip z, which GXM reads in
            // [-w, w], mapped into WebGPU's [0, w]. It needs no scene statistics at all, so it
            // cannot be thrown off by a depth range measured through a different projection
            // than the one the recompiled shader actually uses.
            ZFix::Gl => "  r.z = (c.z + c.w) * 0.5;\n",
            ZFix::Off => "",
        };
        let y = if yflip { "  r.y = -c.y;\n" } else { "" };
        // The clip-w SIGN correction (`gxp_depth.range.z`, +1 or -1; see `NegW`). A title whose
        // projection puts clip `w` NEGATIVE in front of the camera names every visible point on
        // the half of the homogeneous line WebGPU clips away. Negating all four components names
        // the SAME point with `w > 0` (homogeneous coordinates are scale-invariant) and moves
        // behind-camera geometry to `w < 0`, where it belongs. It is one multiply by a UNIFORM,
        // which is what keeps the decision PER DRAW rather than per vertex - a per-vertex
        // `if (w < 0) { c = -c; }` would pull the geometry behind the eye into view.
        //
        // `gxp_depth.range.z` selects the correction: 1 = none, -1 = negate the whole vector,
        // 2 = flip the sign of `w` alone. The two are DIFFERENT renders, not two spellings of
        // one: negating all four components leaves `x/w` exactly as it was (that is what makes
        // it a rename of the same point), so it fixes the clip and nothing else, while flipping
        // `w` alone MIRRORS the draw in both axes. Which one a title needs is a measurement.
        //
        // The `GxpDepth` block itself is declared by the LINKER (`link::GXP_DEPTH_DECL`), not
        // here: the fragment stage reads the same block to reconstruct the guest's window
        // position, and a linked module has to be independently compilable.
        let helper = format!(
            "fn gxp_clipfix(cin: vec4<f32>) -> vec4<f32> {{\n\
             \x20 var c = cin;\n\
             \x20 if (gxp_depth.range.z < 0.0) {{ c = -cin; }}\n\
             \x20 else if (gxp_depth.range.z > 1.5) {{ c = vec4<f32>(cin.x, cin.y, cin.z, -cin.w); }}\n\
             \x20 var r = c;\n{z}{y}  return r;\n}}\n"
        );
        const ASSIGN: &str = "\n  out.position = ";
        let at = wgsl.find(ASSIGN)?;
        let rhs_start = at + ASSIGN.len();
        let rhs_end = rhs_start + wgsl[rhs_start..].find(";")?;
        let mut patched = String::with_capacity(wgsl.len() + 64);
        patched.push_str(&wgsl[..rhs_start]);
        patched.push_str("gxp_clipfix(");
        patched.push_str(&wgsl[rhs_start..rhs_end]);
        patched.push(')');
        patched.push_str(&wgsl[rhs_end..]);
        // Diagnostic (`VITASLOP_GXP_KEYCOLOR`): shade every pair a flat colour derived from its
        // own key, keeping its real geometry, depth and blend. One run then answers "WHICH pair
        // owns this region of the screen" for every region at once - the question that otherwise
        // costs one `VITASLOP_GXP_KEYS` run per candidate, and the one that has to be answered
        // before any question about what a surface's shader computes.
        if let Some(k) = keycolor {
            let chan = |shift: u32| ((k >> shift) & 0xff) as f32 / 255.0;
            let (r, g, b) = (0.25 + 0.75 * chan(0), 0.25 + 0.75 * chan(21), 0.25 + 0.75 * chan(42));
            match patched.rfind("\n  return ") {
                Some(at) => {
                    let end = patched[at + 1..].find(";\n").map(|e| at + 1 + e + 1).unwrap_or(patched.len());
                    patched.replace_range(
                        at + 1..end,
                        &format!("  return vec4<f32>({r:.3}, {g:.3}, {b:.3}, 1.0);"),
                    );
                    // Print the assignment: reading the colour back OFF the frame means undoing
                    // whatever transfer function the target applied, and a near-match to the
                    // wrong key is not distinguishable from a match to the right one.
                    report!("gxp keycolor: key {k:x} -> linear rgb({r:.3}, {g:.3}, {b:.3})");
                }
                None => report!(
                    "gxp build: VITASLOP_GXP_KEYCOLOR found no fragment return to replace for key \
                     {k:x} - the module shape changed; this pair is NOT key-coloured"
                ),
            }
        }
        if solid {
            // Diagnostic: force the fragment to solid magenta so any on-screen triangle is
            // visible regardless of shading. The colour expression depends on which register
            // holds the colour and at what precision, so this matches the fragment entry's
            // LAST `return` statement structurally rather than by its exact text - an
            // enumerate-the-spellings version silently stopped substituting anything the first
            // time the emitter's return line changed, which made the diagnostic lie.
            match patched.rfind("\n  return ") {
                Some(at) => {
                    let end = patched[at + 1..]
                        .find(";\n")
                        .map(|e| at + 1 + e + 1)
                        .unwrap_or(patched.len());
                    patched.replace_range(at + 1..end, "  return vec4<f32>(1.0, 0.0, 1.0, 1.0);");
                }
                None => report!(
                    "gxp build: VITASLOP_GXP_SOLID found no fragment return to replace - \
                     the module shape changed; solid-fill is NOT in effect"
                ),
            }
        }
        Some(format!("{helper}{patched}"))
    }

    /// Report, ONCE PER SHADER PAIR, that a draw is rendering the fixed-function
    /// approximation instead of the guest's own translated shaders, and exactly why.
    ///
    /// This is deliberately not gated behind a debug flag. The recompiler's contract is
    /// no-guess: an unknown opcode hard-fails naming its raw word, an unfed shader input
    /// hard-fails, a clip fixup that finds nothing to patch hard-fails. But the RENDERER
    /// answers all of those by drawing the older capture-based approximation for that pair,
    /// and a silent approximation is indistinguishable on screen from a faithful render -
    /// which is how session 8's "36 draws render with their real guest shaders" turned out
    /// to be 36 draws that linked and never ran. The reason line is the only signal that a
    /// pair is still owed work, so it must be visible in an ordinary run.
    ///
    /// The dedupe is global rather than per-renderer: a pair is submitted hundreds of times a
    /// frame, and nothing is gained by repeating it for a second render target.
    /// How far past a render target's base address a sampled texture may start and still be
    /// that target. One 960-pixel row at 4 bytes is 3840, so this covers "the same buffer,
    /// described from a slightly different origin" and cannot reach the next surface: this
    /// title's own targets are 16 KiB apart.
    const RTT_ALIAS_SLACK: u32 = 4096;

    /// The rendered target `addr` refers to, allowing for a small positive offset into it.
    ///
    /// A title does not always sample a render target through a texture describing exactly that
    /// target. This one aliases its 960x544 world buffer as a **1920x1088** texture starting 256
    /// bytes in - the 2x supersampled view of the same memory - and reads it that way from every
    /// post-process pass. Matching the data address EXACTLY misses that completely: the sample
    /// falls through to the guest bytes, which the GPU never wrote, and the light, blur and
    /// composite passes all render black. Matching by RANGE binds the render they meant.
    ///
    /// Reports the first alias it resolves per (pair, unit): substituting a different buffer
    /// than the one the guest named is exactly the kind of helpfulness that must not be silent.
    fn rendered_alias(
        rendered: &HashMap<u32, wgpu::TextureView>,
        addr: u32,
        key: u64,
        unit: u8,
    ) -> Option<u32> {
        if addr == 0 {
            return None;
        }
        if rendered.contains_key(&addr) {
            return Some(addr);
        }
        let base = rendered
            .keys()
            .copied()
            .filter(|&b| addr > b && addr - b <= RTT_ALIAS_SLACK)
            .max_by_key(|&b| b)?;
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(u64, u8)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert((key, unit)) {
            report!(
                "gxp pair {key:016x}: sampler unit {unit} names {addr:#x}, which is {} bytes into \
                 the render target at {base:#x} - binding that target's render (the guest is \
                 describing the same buffer from a different origin)",
                addr - base
            );
        }
        Some(base)
    }

    /// Report - once per (pair, unit) - that a recompiled draw is sampling a texture with a NULL
    /// data pointer, i.e. the 1x1 zero texel the runtime substitutes for an uninitialised
    /// `SceGxmTexture` handle.
    ///
    /// This is not a fallback: the draw runs, with the guest's own shader, and produces a
    /// perfectly valid picture of nothing. That is exactly why it needs saying out loud - a
    /// render target that comes out uniformly black looks like a geometry, depth or blend
    /// problem, and this is none of those.
    fn report_zero_texel_sample(key: u64, unit: u8) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(u64, u8)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert((key, unit)) {
            report!(
                "gxp pair {key:016x}: sampler unit {unit} is bound to a 1x1 ZERO texel (the guest \
                 handle had null control words) - everything this draw derives from that sample \
                 is zero"
            );
        }
    }

    /// The wgpu viewport rectangle a GXM viewport asks for, in pixels of a `w` x `h` target,
    /// or `None` when the guest left the default (all-zero) or is naming exactly the whole
    /// target - in which case there is nothing to set and nothing to restore.
    ///
    /// GXM maps normalised device coordinates to the framebuffer as `screen = offset + scale *
    /// ndc`, and `yScale` is normally NEGATIVE because ndc `+1` is the top of the screen while
    /// framebuffer row 0 is the top as well. wgpu's `set_viewport(x, y, w, h)` bakes that same
    /// flip in, so the two agree exactly whenever `yScale < 0`: the rect is centred on
    /// `(xOffset, yOffset)` with half-extents `|xScale|` and `|yScale|`.
    ///
    /// The DEPTH half (`zOffset`, `zScale`) is deliberately not applied. The recompiled vertex
    /// stage already writes a depth in whichever convention `VITASLOP_GXP_ZFIX` selects, so
    /// handing the guest's z mapping to `min_depth`/`max_depth` as well would apply it twice.
    fn gxm_viewport_rect(vp: &[f32; 6], w: u32, h: u32) -> Option<(f32, f32, f32, f32)> {
        let [xo, xs, yo, ys, _, _] = *vp;
        // All-zero is the sentinel for "the guest never called sceGxmSetViewport", not a
        // request for a zero-area viewport. A zero SCALE with a nonzero offset would be a
        // degenerate viewport and is reported below rather than silently ignored.
        if vp.iter().all(|v| *v == 0.0) {
            return None;
        }
        if !vp.iter().all(|v| v.is_finite()) {
            report_viewport_problem(vp, "it contains a non-finite component");
            return None;
        }
        if ys > 0.0 {
            // ndc +1 would land at the BOTTOM of the rect. wgpu's viewport cannot express a
            // vertical flip (it requires a positive height), so the rect below renders this
            // pass upside down. Saying so is the whole point - a silently mirrored pass is
            // indistinguishable from a correct one on a symmetric image.
            report_viewport_problem(vp, "yScale is POSITIVE, which is a vertical flip that a wgpu viewport cannot express - this pass renders mirrored");
        }
        let (x, y) = (xo - xs.abs(), yo - ys.abs());
        let (vw, vh) = (2.0 * xs.abs(), 2.0 * ys.abs());
        if vw <= 0.0 || vh <= 0.0 {
            report_viewport_problem(vp, "it has a zero-area rectangle");
            return None;
        }
        // Already the whole target: setting it is a no-op and skipping it keeps the common
        // fullscreen case free of state changes.
        if x == 0.0 && y == 0.0 && vw == w as f32 && vh == h as f32 {
            return None;
        }
        // wgpu rejects a viewport that leaves the attachment. Clamp INTO the target and say
        // so: rendering the pass at the wrong rect is wrong, but dropping the viewport
        // entirely is wrong in a way that looks like nothing happened.
        let (cx, cy) = (x.max(0.0), y.max(0.0));
        let (cw, ch) = ((vw + x.min(0.0)).min(w as f32 - cx), (vh + y.min(0.0)).min(h as f32 - cy));
        if (cx, cy, cw, ch) != (x, y, vw, vh) {
            report_viewport_problem(
                vp,
                &format!(
                    "it asks for ({x}, {y}, {vw}, {vh}) which leaves a {w}x{h} target - \
                     CLAMPED to ({cx}, {cy}, {cw}, {ch})"
                ),
            );
        }
        if cw <= 0.0 || ch <= 0.0 {
            return None;
        }
        Some((cx, cy, cw, ch))
    }

    /// Report - once per surface - that a pass is rendering into a GAMMA-CORRECT colour
    /// surface, and whether that is being honoured.
    ///
    /// `honoured == false` means the colour format has no sRGB twin, so the ROP encode the
    /// hardware performs is not happening and this surface (and everything sampling it) reads
    /// darker than the title intends. That has to be said out loud - it is precisely the kind
    /// of uniform darkening that gets chased as a lighting bug.
    fn report_gamma_surface(addr: u32, honoured: bool) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert(addr) {
            if honoured {
                report!(
                    "gxm surface: {addr:#x} is GAMMA-CORRECT - rendering through an sRGB view, \
                     so writes are sRGB-encoded after blending as the hardware does"
                );
            } else {
                report!(
                    "gxm surface: {addr:#x} is GAMMA-CORRECT but the render format has no sRGB \
                     twin - its writes stay LINEAR where the hardware would encode them, so it \
                     and anything sampling it read darker than the title intends"
                );
            }
        }
    }

    /// Report - once per (pair, unit) - that a sampler was bound to a render target's DEPTH
    /// rather than to any colour buffer.
    ///
    /// The same class of event as [`rendered_alias`]: the runtime is substituting something
    /// other than the guest bytes at the address the shader named, and that substitution
    /// decides what the pass computes.
    fn report_depth_sample_bound(key: u64, unit: u8, addr: u32) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(u64, u8)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert((key, unit)) {
            report!(
                "gxp pair {key:016x}: sampler unit {unit} names {addr:#x}, which is a render \
                 target's DEPTH surface - binding this frame's converted depth"
            );
        }
    }

    /// Report - once per target - that a render target's depth was converted into the guest's
    /// encoding for a later pass to sample, and which encoding that was.
    ///
    /// Unconditional. Which value a GXM depth surface holds is REVERSE-ENGINEERED, not read
    /// off a spec we hold, so every frame built on that reading has to say which reading it
    /// used - otherwise a soft-particle fade that comes out subtly wrong looks like a shading
    /// bug rather than a choice made here.
    fn report_depth_conversion(addr: u32, mode: u32, konst: f32, depth_min: f32, depth_scale: f32) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(u32, u32)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert((addr, mode)) {
            if mode == 5 {
                report!(
                    "gxm depth: target {addr:#x} depth REPLACED by the constant {konst} - this is \
                     a causality probe, and any frame built on it is not a real frame"
                );
                return;
            }
            let name = match mode {
                0 => "the guest's own window depth (a + c/w, measured from this pass)",
                1 => "w",
                2 => "-w",
                3 => "-1/w",
                6 => "1/w",
                _ => "the stored [0,1] depth",
            };
            let degenerate = if depth_scale == 0.0 {
                "  (the scene has NO depth range, so w cannot be recovered and the stored depth \
                 is passed through unchanged)"
            } else {
                ""
            };
            report!(
                "gxm depth: target {addr:#x} depth re-encoded as {name} for a later pass to \
                 sample (depth_min={depth_min}, depth_scale={depth_scale}){degenerate}"
            );
        }
    }

    /// Report - once per address - that a pass samples a depth buffer the frame did render,
    /// but through the DISPLAY target rather than an offscreen one, where no converted copy
    /// exists.
    fn report_unconverted_depth_sample(addr: u32) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert(addr) {
            report!(
                "gxm depth: a draw samples the depth at {addr:#x}, which belongs to a scene \
                 rendered into the DISPLAY target - that pass keeps no depth copy, so this \
                 sample reads guest bytes the GPU never wrote"
            );
        }
    }

    /// Report - once per distinct viewport and message - that a guest viewport could not be
    /// reproduced exactly. Unconditional, like every other approximation in this path.
    fn report_viewport_problem(vp: &[f32; 6], what: &str) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(String, String)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        let vps = format!("{vp:?}");
        if seen.insert((vps.clone(), what.to_string())) {
            report!("gxp viewport {vps}: {what}");
        }
    }

    fn report_fallback(key: u64, reason: &str) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        // A poisoned lock must not lose the diagnostic - recover the set and report anyway.
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert(key) {
            report_warn!("gxp pair {key:016x}: FALLS BACK to fixed-function - {reason}");
        }
        drop(seen);
        fallback_reasons().lock().unwrap_or_else(|e| e.into_inner()).insert(key, reason.to_string());
        if !allow_fixed_function() {
            panic!(
                "gxp pair {key:016x} cannot be recompiled: {reason}\n\
                 The recompiler is enabled, so this draw would have been drawn by the \
                 fixed-function APPROXIMATION instead - a different renderer, which does not \
                 run the guest's shader and cannot be told apart from a faithful render by \
                 looking at the frame. Refusing. Set VITASLOP_GXP_ALLOW_FIXED_FUNCTION=1 to \
                 approximate anyway (bring-up only: it is how a title's world silently \
                 rendered 328 of 388 draws wrong)."
            );
        }
    }

    /// Whether a shader pair the recompiler cannot translate may be drawn by the
    /// fixed-function approximation instead (`VITASLOP_GXP_ALLOW_FIXED_FUNCTION=1`).
    ///
    /// OFF by default, and that default is the point. The approximation reconstructs a draw
    /// from captured state without running the guest's fragment program, so what it produces
    /// is plausible and wrong, and indistinguishable on screen from a correct render. It
    /// remains available for bring-up, where seeing a menu at all is what lets a recipe be
    /// authored - but a run that does not ask for it now stops at the first pair it cannot
    /// translate, naming the pair and the reason.
    fn allow_fixed_function() -> bool {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| crate::knobs::flag("VITASLOP_GXP_ALLOW_FIXED_FUNCTION"))
    }

    /// Why each pair last fell back, keyed by pair.
    ///
    /// [`report_fallback`] PRINTS once per pair but RECORDS every time, so a per-scene tally
    /// can weight reasons by how many DRAWS each cost. That is the number that says what to
    /// fix next, and the printed list cannot give it: sixty pairs that fall back on one draw
    /// each and one pair that falls back on three hundred read exactly the same there.
    fn fallback_reasons() -> &'static std::sync::Mutex<HashMap<u64, String>> {
        use std::sync::{Mutex, OnceLock};
        static REASONS: OnceLock<Mutex<HashMap<u64, String>>> = OnceLock::new();
        REASONS.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// The recorded fallback reason for `key`, or a placeholder if the pair fell back before
    /// any reason was recorded (which would itself be a bug worth seeing).
    fn fallback_reason_of(key: u64) -> String {
        fallback_reasons()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .cloned()
            .unwrap_or_else(|| "(no reason recorded)".to_string())
    }

    /// Report, once per pair, every USSE control-flow branch either program contains: the
    /// branch's instruction index, its signed word delta, the target that resolves to, and the
    /// program's instruction count.
    ///
    /// This is a MEASUREMENT, not a debug aid, and that is why it is unconditional like
    /// [`report_fallback`] rather than behind a flag. Branch translation rests on one fact the
    /// distilled ISA reference states but the captured shader corpus cannot corroborate (it
    /// contains no branches at all): that a target is `index + rel` rather than `index + 1 +
    /// rel`. Those differ by exactly one instruction, so the wrong reading leaves the last
    /// instruction of every conditional block running unconditionally - a plausible wrong
    /// picture with no error anywhere. A branch whose target lands exactly on `total` (one past
    /// the end, the shape a trailing `if` compiles to) confirms the implemented reading;
    /// targets that cluster one short of it, with none ever reaching `total`, would refute it.
    fn report_branches(key: u64, gxp: &GxpRecompile) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if !seen.insert(key) {
            return;
        }
        for (kind, bytes) in [("vertex", &gxp.vprog), ("fragment", &gxp.fprog)] {
            let Ok((total, sites)) = vitaslop_gxp_shader::branch_sites(bytes) else { continue };
            for s in sites {
                report!(
                    "gxp pair {key:016x}: {kind} BR #{} rel={} -> target {} of {total} instrs",
                    s.index, s.rel, s.target
                );
            }
        }
    }

    /// Link a guest shader pair and build its two pipeline variants + bind-group layouts.
    /// `None` (fall back) on any link error or an unmappable vertex format.
    ///
    /// Every `None` here is REPORTED, unconditionally and by name - see [`report_fallback`].
    /// The translator itself never guesses (an unknown opcode hard-fails naming its raw word),
    /// but the renderer's answer to that hard failure is to draw the fixed-function
    /// approximation, and an unreported approximation reads exactly like a faithful render.
    /// Map a `SceGxmDepthFunc` to its wgpu equivalent. The enum is a SHIFTED field (vitasdk
    /// `gxm.h` spaces the values 0x00400000 apart), which is how it is stored in the sticky
    /// render state, so it is normalised back to 0..7 here.
    ///
    /// Not wired into the recompiled pipeline yet - see the note in `build_gxp_pipeline` for
    /// the measurement that says why. Kept because the mapping itself is right and is what the
    /// switch-over needs.
    #[allow(dead_code)]
    fn gxm_depth_func(f: u32) -> wgpu::CompareFunction {
        use wgpu::CompareFunction as C;
        match (f >> 22) & 0x7 {
            0 => C::Never,
            1 => C::Less,
            2 => C::Equal,
            3 => C::LessEqual,
            4 => C::Greater,
            5 => C::NotEqual,
            6 => C::GreaterEqual,
            _ => C::Always,
        }
    }

    /// Map a `SceGxmBlendFactor` to its wgpu equivalent. The enum order is the vitasdk
    /// `gxm.h` one; `SRC_ALPHA_SATURATE` has a wgpu counterpart and `DST_ALPHA_SATURATE`
    /// (which wgpu has no factor for) is reported and treated as `One` rather than silently
    /// becoming something else.
    fn gxm_blend_factor(f: u8) -> wgpu::BlendFactor {
        use wgpu::BlendFactor as F;
        match f {
            0 => F::Zero,
            1 => F::One,
            2 => F::Src,
            3 => F::OneMinusSrc,
            4 => F::SrcAlpha,
            5 => F::OneMinusSrcAlpha,
            6 => F::Dst,
            7 => F::OneMinusDst,
            8 => F::DstAlpha,
            9 => F::OneMinusDstAlpha,
            10 => F::SrcAlphaSaturated,
            _ => {
                report_unmapped_blend("DST_ALPHA_SATURATE blend factor");
                F::One
            }
        }
    }

    /// Map a `SceGxmBlendFunc` pair (func, src, dst) to a wgpu blend component. `MIN`/`MAX`
    /// ignore the factors, exactly as the hardware does.
    fn gxm_blend_component(func: u8, src: u8, dst: u8) -> wgpu::BlendComponent {
        use wgpu::BlendOperation as O;
        let operation = match func {
            2 => O::Subtract,
            3 => O::ReverseSubtract,
            4 => O::Min,
            5 => O::Max,
            _ => O::Add,
        };
        // MIN and MAX take the operands unscaled.
        let (src, dst) = if matches!(func, 4 | 5) { (1, 1) } else { (src, dst) };
        wgpu::BlendComponent {
            src_factor: gxm_blend_factor(src),
            dst_factor: gxm_blend_factor(dst),
            operation,
        }
    }

    /// The wgpu blend state for a draw's captured `SceGxmBlendInfo`, or `None` when the
    /// program does not blend at all (both funcs `NONE`), which is a REPLACE.
    fn gxm_blend_state(b: [u8; 7]) -> Option<wgpu::BlendState> {
        let [_mask, color_func, alpha_func, color_src, color_dst, alpha_src, alpha_dst] = b;
        if color_func == 0 && alpha_func == 0 {
            return Some(wgpu::BlendState::REPLACE);
        }
        Some(wgpu::BlendState {
            color: gxm_blend_component(color_func, color_src, color_dst),
            alpha: gxm_blend_component(alpha_func, alpha_src, alpha_dst),
        })
    }

    /// The wgpu write mask for a `SceGxmColorMask` (bit 0 R, 1 G, 2 B, 3 A).
    fn gxm_color_mask(mask: u8) -> wgpu::ColorWrites {
        let mut w = wgpu::ColorWrites::empty();
        for (bit, flag) in [
            (0, wgpu::ColorWrites::RED),
            (1, wgpu::ColorWrites::GREEN),
            (2, wgpu::ColorWrites::BLUE),
            (3, wgpu::ColorWrites::ALPHA),
        ] {
            if mask & (1 << bit) != 0 {
                w |= flag;
            }
        }
        w
    }

    /// Report - once per render target - that `VITASLOP_GXP_NEGW=force` corrected this pass's
    /// clip `w` without measuring anything.
    fn report_forced_negw(target: u32) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(target) {
            report!(
                "gxp clip: pass into {target:#010x}: VITASLOP_GXP_NEGW=force - correcting the \
                 clip w sign with NO measurement behind it. This is a diagnostic; the frame it \
                 produces is not evidence of anything except what the correction does."
            );
        }
    }

    /// Report - once per shader pair - that the guest's own `SceGxmBlendInfo` gives this pair a
    /// colour write mask of ZERO, so every draw using it writes no colour channel.
    ///
    /// This is not a fallback and not an approximation: it is the guest's state, faithfully
    /// applied, and a depth-only prepass is exactly what it looks like. It is reported because
    /// the RESULT on screen - a pass that renders nothing - is identical to a draw whose
    /// geometry, textures or transform we got wrong, and this is the only one of those that is
    /// correct. Knowing which costs one line here and a whole investigation otherwise.
    fn report_zero_color_mask(key: u64) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(key) {
            report!(
                "gxp pair {key:016x}: the guest's blend info gives this pair a colour write mask \
                 of 0 - it writes NO colour channel, so its draws affect depth only. That is the \
                 guest's own state, not a fallback."
            );
        }
    }

    /// Report - once per case - a GXM blend value with no exact wgpu equivalent, so the
    /// substitution is visible rather than silently changing what a draw composites like.
    fn report_unmapped_blend(what: &str) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert(what.to_string()) {
            report_warn!("gxm blend: {what} has no wgpu equivalent - substituting ONE, which is an approximation");
        }
    }

    fn build_gxp_pipeline(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        gxp: &GxpRecompile,
        key: u64,
        zfix: ZFix,
        yflip: bool,
        solid: bool,
        nodepth: bool,
        noblend: bool,
    ) -> Option<GxpPipeline> {
        let debug = std::env::var_os("VITASLOP_GXP_DEBUG").is_some();
        report_branches(key, gxp);
        let linked = match vitaslop_gxp_shader::link_programs(&gxp.vprog, &gxp.fprog) {
            Ok(l) => l,
            Err(e) => {
                report_fallback(key, &format!("link failed: {e}"));
                return None;
            }
        };

        // Vertex layout: each linked attribute (@location L, base lane B) is fed by the guest
        // stream attribute whose reg_index == B. We repack it to tightly-packed `Float32xN`
        // (converting F16/U8N/etc. on the CPU) so wgpu's vertex-format gaps (no Float16x3, no
        // Unorm8x3, ...) never block a draw, and the shader (which reads f32) gets exact values.
        let mut wattrs: Vec<wgpu::VertexAttribute> = Vec::with_capacity(linked.vertex_bindings.attributes.len());
        let mut repack: Vec<RepackAttr> = Vec::with_capacity(linked.vertex_bindings.attributes.len());
        let mut packed_offset: u32 = 0;
        for a in &linked.vertex_bindings.attributes {
            let ga: &GxpAttr = match gxp.attributes.iter().find(|g| g.reg_index as u32 == a.base_lane) {
                Some(g) => g,
                None => {
                    report_fallback(
                        key,
                        &format!(
                            "no guest attribute for linked @location {} base_lane {} (guest reg_indices {:?})",
                            a.location,
                            a.base_lane,
                            gxp.attributes.iter().map(|g| g.reg_index).collect::<Vec<_>>()
                        ),
                    );
                    return None;
                }
            };
            let comps = ga.components.clamp(1, 4);
            let format = match comps {
                1 => wgpu::VertexFormat::Float32,
                2 => wgpu::VertexFormat::Float32x2,
                3 => wgpu::VertexFormat::Float32x3,
                _ => wgpu::VertexFormat::Float32x4,
            };
            wattrs.push(wgpu::VertexAttribute { format, offset: packed_offset as u64, shader_location: a.location });
            repack.push(RepackAttr { guest_offset: ga.offset as u32, gxm_format: ga.gxm_format, components: comps, packed_offset });
            packed_offset += comps as u32 * 4;
        }
        let packed_stride = packed_offset.max(4);

        // Decisive diagnostic (`VITASLOP_GXP_INTERP`): run the recompiled vertex shader on the
        // CPU interpreter with the real captured SA + the first vertex's attributes, and print
        // the clip output o[0..3]. If the recompiler math + uniforms are right, this is a
        // sensible clip position; if it is off-screen/NaN, the problem is upstream of the GPU.
        if std::env::var_os("VITASLOP_GXP_INTERP").is_some() {
            if let Ok(vrc) = vitaslop_gxp_shader::recompile_vertex(&gxp.vprog) {
                let mut regs = vitaslop_gxp_shader::interp::RegFile::with_lanes(512);
                for (k, c) in gxp.vert_sa.chunks_exact(4).enumerate() {
                    if k < regs.sa.len() {
                        regs.sa[k] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    }
                }
                for a in &linked.vertex_bindings.attributes {
                    if let Some(ga) = gxp.attributes.iter().find(|g| g.reg_index as u32 == a.base_lane) {
                        for c in 0..ga.components as usize {
                            let lane = a.base_lane as usize + c;
                            if lane < regs.pa.len() {
                                regs.pa[lane] = read_attr_component(&gxp.vertices, ga.offset as usize, ga.gxm_format, c);
                            }
                        }
                    }
                }
                // The SECONDARY program runs before the primary and its whole purpose is to
                // leave values in SA registers the primary reads, so an interpretation that
                // skips it is reading uniforms the real module has already overwritten. Its
                // container literals go in too, as raw bit patterns, exactly as the module
                // emits them.
                if let Ok(program) = vitaslop_gxp_shader::Program::parse(&gxp.vprog) {
                    for &(reg, value) in &program.literals {
                        if let Some(slot) = regs.sa.get_mut(reg as usize) {
                            *slot = f32::from_bits(value);
                        }
                    }
                    let secondary = vitaslop_gxp_shader::usse::decode_secondary_shader(&program);
                    if let Err(e) = vitaslop_gxp_shader::interp::run(&secondary, &mut regs) {
                        report!("gxp interp: secondary program failed: {e}");
                    }
                }
                // Where the WHOLE mesh lands, not just its first vertex. Vertex 0 of a large
                // world mesh is routinely behind the camera, so its `w<0` says nothing about
                // whether the draw covers any pixels - and "this draw is missing" is exactly
                // the question. Counting the vertices in front of the eye and taking the NDC
                // bounding box over them answers it outright.
                let saved_sa = regs.sa.clone();
                let stride = gxp.vertex_stride.max(1) as usize;
                let nverts = (gxp.vertices.len() / stride).min(4096);
                let (mut in_front, mut lo, mut hi) = (0usize, [f32::MAX; 2], [f32::MIN; 2]);
                for v in 0..nverts {
                    let mut vregs = regs.clone();
                    vregs.sa.clone_from(&saved_sa);
                    for a in &linked.vertex_bindings.attributes {
                        if let Some(ga) = gxp.attributes.iter().find(|g| g.reg_index as u32 == a.base_lane) {
                            let base = v * stride + ga.offset as usize;
                            for c in 0..ga.components as usize {
                                let lane = a.base_lane as usize + c;
                                if lane < vregs.pa.len() {
                                    vregs.pa[lane] = read_attr_component(&gxp.vertices, base, ga.gxm_format, c);
                                }
                            }
                        }
                    }
                    if vitaslop_gxp_shader::interp::run(&vrc.shader, &mut vregs).is_err() {
                        break;
                    }
                    let w = vregs.o[3];
                    if w > 0.0 {
                        in_front += 1;
                        lo = [lo[0].min(vregs.o[0] / w), lo[1].min(vregs.o[1] / w)];
                        hi = [hi[0].max(vregs.o[0] / w), hi[1].max(vregs.o[1] / w)];
                    }
                }
                report!(
                    "gxp interp: key {key:x} {in_front}/{nverts} vertices have w>0, ndc bbox x[{:.3},{:.3}] y[{:.3},{:.3}]",
                    lo[0], hi[0], lo[1], hi[1]
                );
                match vitaslop_gxp_shader::interp::run_watching_for_nan(&vrc.shader, &mut regs) {
                    Ok(site) => {
                        let w = regs.o[3];
                        let ndc = if w.abs() > 1e-6 { [regs.o[0] / w, regs.o[1] / w, regs.o[2] / w] } else { [0.0; 3] };
                        report!("gxp interp: o={:?} ndc={:?} viewport(xo,xs,yo,ys,zo,zs)={:?}", &regs.o[0..4], ndc, gxp.viewport);
                        // A NaN clip position draws NOTHING, which looks exactly like a black
                        // shader; naming the instruction that produced it is the whole point.
                        if let Some(s) = site {
                            report!(
                                "gxp interp: FIRST non-finite value at instruction #{} {} -> {} channel {} = {}\n  sources: {}",
                                s.index, s.op, s.dest, s.channel, s.value, s.sources.join("  ")
                            );
                        }
                    }
                    Err(e) => report!("gxp interp: run failed: {e}"),
                }
            }
        }

        let keycolor = std::env::var_os("VITASLOP_GXP_KEYCOLOR").map(|_| key);
        let wgsl = match inject_clip_fixup(&linked.wgsl, zfix, yflip, solid, keycolor) {
            Some(w) => w,
            None => {
                report_fallback(
                    key,
                    "no `out.position` assignment to wrap with the clip fixup - refusing to \
                     render without the depth remap",
                );
                return None;
            }
        };
        // `VITASLOP_GXP_WGSL_DIR=<dir>`: write each pair's linked WGSL to `<key>.wgsl`.
        //
        // When a recompiled draw comes out wrong, the question is what the guest's shader was
        // actually translated INTO, and until now the only way to see that was to add a print
        // and rebuild. The translation is the artefact worth reading - it names the samplers,
        // the varyings and the clip position in one place.
        if let Ok(dir) = std::env::var("VITASLOP_GXP_WGSL_DIR") {
            let dir = std::path::Path::new(&dir);
            if let Err(e) = std::fs::create_dir_all(dir)
                .and_then(|()| std::fs::write(dir.join(format!("{key:016x}.wgsl")), &wgsl))
            {
                report_warn!("gxp: cannot write WGSL for pair {key:016x}: {e}");
            }
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gxp-linked"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        // group0 vertex uniform, group1 fragment uniform, group2 samplers (empty where unused).
        //
        // Both are addressed by DYNAMIC OFFSET into one arena per pass, so a pass needs one
        // bind group per shader pair rather than one per draw - the same shape the
        // fixed-function path has always used for its per-draw uniform. `min_binding_size`
        // pins the shader-visible window to exactly this stage's SA block, so the arena's
        // 256-byte offset alignment cannot let a draw read into the next draw's uniforms.
        let uniform_entry = |vis: wgpu::ShaderStages, bytes: u64| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(bytes),
            },
            count: None,
        };
        let vsa_lanes = linked.vertex_bindings.sa_lane_count;
        let fsa_lanes = linked.fragment_bindings.sa_lane_count;
        let sa_bytes = |lanes: u32| (lanes.div_ceil(4) as u64) * 16;
        let (vsa_bytes, fsa_bytes) = (sa_bytes(vsa_lanes), sa_bytes(fsa_lanes));
        let g0_entries: Vec<wgpu::BindGroupLayoutEntry> =
            if vsa_lanes > 0 { vec![uniform_entry(wgpu::ShaderStages::VERTEX, vsa_bytes)] } else { vec![] };
        let g1_entries: Vec<wgpu::BindGroupLayoutEntry> =
            if fsa_lanes > 0 { vec![uniform_entry(wgpu::ShaderStages::FRAGMENT, fsa_bytes)] } else { vec![] };
        let mut g2_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
        let mut samplers: Vec<(u8, SamplerDim)> = Vec::new();
        for (i, b) in linked.fragment_bindings.samplers.iter().enumerate() {
            // Mirrors `TexBinding::wgsl_type` exactly - the layout and the shader must agree.
            let dim = match (b.coords >= 3, b.cube) {
                (true, true) => SamplerDim::Cube,
                (true, false) => SamplerDim::Three,
                _ => SamplerDim::Two,
            };
            // The binding plan already names the GXM texture unit the guest bound to: the SMP
            // sampler operand is resolved through the container's texture-control table when the
            // instruction stream is decoded, and a prefetched sample names its unit outright.
            let gxm_unit = b.unit as u32;
            g2_entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32 * 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: dim.view_dimension(),
                    multisampled: false,
                },
                count: None,
            });
            g2_entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32 * 2 + 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
            if debug {
                report!("gxp build: gxm texture unit {gxm_unit} (coords {}, {dim:?})", b.coords);
            }
            samplers.push((gxm_unit as u8, dim));
        }
        // group3 carries the scene depth range the injected clip fixup remaps through - one
        // vec4 the renderer refills per frame, so the pipeline stays cached by shader identity.
        // BOTH stages read it: the vertex stage to remap clip depth, the fragment stage to
        // reconstruct the guest's window POSITION (see `link::GXP_DEPTH_DECL`).
        // The VERTEX stage's samplers go in the SAME group, after the fragment ones: a device
        // guarantees only four bind groups and the other three are spoken for. A vertex program
        // that fetches a texture builds its geometry from it, so these are not decoration -
        // without them the draw has no mesh. Their unit numbering is independent of the fragment
        // stage's, which is why they keep separate names (`vt{u}`/`vs{u}`).
        let vsampler_base = g2_entries.len() as u32;
        let mut vertex_samplers: Vec<(u8, SamplerDim)> = Vec::new();
        for (i, b) in linked.vertex_bindings.samplers.iter().enumerate() {
            let dim = match (b.coords >= 3, b.cube) {
                (true, true) => SamplerDim::Cube,
                (true, false) => SamplerDim::Three,
                _ => SamplerDim::Two,
            };
            g2_entries.push(wgpu::BindGroupLayoutEntry {
                binding: vsampler_base + i as u32 * 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: dim.view_dimension(),
                    multisampled: false,
                },
                count: None,
            });
            g2_entries.push(wgpu::BindGroupLayoutEntry {
                binding: vsampler_base + i as u32 * 2 + 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
            if debug {
                report!("gxp build: VERTEX gxm texture unit {} (coords {}, {dim:?})", b.unit, b.coords);
            }
            vertex_samplers.push((b.unit, dim));
        }
        let layouts = [
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gxp-g0"), entries: &g0_entries }),
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gxp-g1"), entries: &g1_entries }),
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gxp-g2"), entries: &g2_entries }),
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gxp-g3"),
                // The pass depth block is per SCENE, not per draw: one cached bind group, no
                // dynamic offset.
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            }),
        ];
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gxp-pl"),
            bind_group_layouts: &[Some(&layouts[0]), Some(&layouts[1]), Some(&layouts[2]), Some(&layouts[3])],
            immediate_size: 0,
        });

        let vbuffers: Vec<Option<wgpu::VertexBufferLayout>> = if wattrs.is_empty() {
            vec![]
        } else {
            vec![Some(wgpu::VertexBufferLayout { array_stride: packed_stride as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wattrs })]
        };

        // The guest's own blend equation, from the `SceGxmBlendInfo` its fragment program was
        // created with. This is state, not a guess: GXM has no runtime blend setter, so the
        // old "opaque draws REPLACE, the rest src-over" reading was the renderer inventing a
        // mode. It got a title's whole world composite wrong - that draw does NOT blend, and
        // forcing src-over made it invisible the moment its shader output alpha 0.
        let guest_blend = gxm_blend_state(gxp.blend_state);
        let write_mask = gxm_color_mask(gxp.blend_state[0]);
        // A colour mask of ZERO is a legal and meaningful GXM state - a depth-only prepass
        // writes no colour - but on screen it is indistinguishable from a draw we got wrong,
        // and it is the one pipeline setting that can make a perfectly recompiled, correctly
        // bound, correctly transformed draw leave no mark at all. Say it out loud, once.
        if write_mask.is_empty() {
            report_zero_color_mask(key);
        }
        // Depth is still the opaque/overlay HEURISTIC, not the guest's captured state, and
        // that is a known gap rather than an oversight. Driving it from the guest
        // (`front_depth_func` / `front_depth_write`, both captured) was tried and MEASURED to
        // be worse: with it, a race's world pass lost its track surface entirely, and only the
        // ship and a barrier survived - `VITASLOP_GXP_NODEPTH` brought them back, which places
        // the fault in the depth test rather than the shading. The likely interaction is with
        // the clip-depth remap `gxp_clipfix` applies (see the scene depth range it reads):
        // guest depth values and remapped ones cannot both be compared by the guest's own
        // function. Settle that before switching this over; until then the heuristic is what
        // the depth remap was built against.
        let make = |opaque: bool| {
            let (mut blend, mut depth_write, mut depth_compare) = if opaque {
                (guest_blend, true, wgpu::CompareFunction::LessEqual)
            } else {
                (guest_blend, false, wgpu::CompareFunction::Always)
            };
            if solid || noblend {
                // Diagnostic: REPLACE, so the fragment's colour reaches the target whatever its
                // alpha is. `solid` additionally replaces the shading and drops the depth test;
                // `noblend` changes ONLY the blend, which is what separates "this surface shades
                // black" from "this surface is composited away". A draw whose shader writes
                // alpha 0 under a src-alpha blend contributes exactly nothing, and the finished
                // frame cannot tell that apart from a fragment that computed black.
                blend = Some(wgpu::BlendState::REPLACE);
            }
            // A zero write mask is the same invisibility by a different route, so the diagnostic
            // has to lift it too or it only answers half the question.
            let write_mask = if solid || noblend { wgpu::ColorWrites::ALL } else { write_mask };
            if solid || nodepth {
                // Both diagnostics drop the depth test; `solid` additionally replaces the
                // shading, which is exactly the difference between them.
                depth_write = false;
                depth_compare = wgpu::CompareFunction::Always;
            }
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(if opaque { "gxp-opaque" } else { "gxp-blend" }),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_main"), buffers: &vbuffers, compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState { format: color_format, blend, write_mask })],
                    compilation_options: Default::default(),
                }),
                // No GPU cull yet: guest facing/winding under the recompiled clip is not yet
                // confirmed, so draw both windings (the fixed-function path does the same) and
                // rely on the depth test. A cull mode is a later refinement.
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(depth_compare),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        Some(GxpPipeline { opaque: make(true), blend: make(false), layouts, vsa_lanes, fsa_lanes, samplers, vertex_samplers, repack, packed_stride })
    }

    impl GxmRenderer {
        /// Build both pipelines for a `color_format` render target. `queue` is used
        /// once here to upload the 1x1 white fallback texel.
        pub fn new(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            color_format: wgpu::TextureFormat,
        ) -> Self {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gxm"),
                source: wgpu::ShaderSource::Wgsl(GXM_SHADER.into()),
            });

            // Group 0: the per-draw uniform, addressed by dynamic offset into one shared
            // buffer (so a whole frame needs a single uniform bind group). `min_binding_size`
            // pins the shader-visible window to the `U` struct so validation is tight.
            let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gxm-uniform"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(UNIFORM_BYTES),
                    },
                    count: None,
                }],
            });

            // Group 1: the sampled texture + its sampler. `filterable` / `Filtering` so a
            // LINEAR-magnified texture (UI/font atlas) can bilinear-sample on the GPU; a
            // Nearest sampler is still valid in a filtering slot, so the same layout serves
            // both filter modes (the sampler is chosen per draw, see `bind_for`).
            let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gxm-texture"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gxm-layout"),
                bind_group_layouts: &[Some(&uniform_layout), Some(&texture_layout)],
                immediate_size: 0,
            });

            // Canonical vertex: pos float32x3 @0, uv float32x2 @12, color unorm8x4 @20,
            // world-space normal float32x3 @24 (for the opaque lighting term).
            let vertex_layout = wgpu::VertexBufferLayout {
                array_stride: GXM_VERTEX_STRIDE,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 12,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Unorm8x4,
                        offset: 20,
                        shader_location: 2,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 24,
                        shader_location: 3,
                    },
                ],
            };

            // A pipeline builder closure differing in blend, depth AND fragment entry
            // point: the opaque (3D) path depth-tests and writes, replaces, and runs
            // `fs_opaque` (albedo texel + exposure/Reinhard); the 2D path disables the
            // depth test, straight-alpha src-over blends in submission order, and runs
            // `fs_blend` (vertex_color * texel modulate) - the exact two modes the
            // software rasterizer switches between per draw.
            let make = |opaque: bool, target_format: wgpu::TextureFormat| {
                let (blend, depth_write, depth_compare, fs) = if opaque {
                    (
                        Some(wgpu::BlendState::REPLACE),
                        true,
                        // LessEqual is GXM's default depth func and what this title's opaque
                        // 3D draws set (SCE_GXM_DEPTH_FUNC_LESS_EQUAL); the software oracle
                        // compares the same way (`depth_passes`), so a coincident later face
                        // ties and repaints identically on both paths. The CPU builder
                        // already culled back faces, so no pipeline cull state is needed.
                        wgpu::CompareFunction::LessEqual,
                        "fs_opaque",
                    )
                } else {
                    (
                        Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        false,
                        wgpu::CompareFunction::Always,
                        "fs_blend",
                    )
                };
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(if opaque { "gxm-opaque" } else { "gxm-blend" }),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        buffers: &[Some(vertex_layout.clone())],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(fs),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: target_format,
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        // No culling: match the software rasterizer, which draws both
                        // windings.
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(depth_write),
                        depth_compare: Some(depth_compare),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: Default::default(),
                    multiview_mask: None,
                    cache: None,
                })
            };

            let opaque = make(true, color_format);
            let blend = make(false, color_format);
            // The same two pipelines against the sRGB view of the same texture, for a pass
            // whose colour surface the guest put in GAMMA-CORRECT mode. Built eagerly because
            // it is two pipelines, not a family: the alternative is discovering mid-frame that
            // a fixed-function draw landed on a gamma surface and having nothing to draw it
            // with.
            let srgb = srgb_twin(color_format)
                .map(|f| (make(true, f), make(false, f)));

            let sampler = |linear: bool| {
                let f = if linear {
                    wgpu::FilterMode::Linear
                } else {
                    wgpu::FilterMode::Nearest
                };
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("gxm-sampler"),
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::Repeat,
                    address_mode_w: wgpu::AddressMode::Repeat,
                    mag_filter: f,
                    min_filter: f,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..Default::default()
                })
            };
            let sampler_point = sampler(false);
            let sampler_linear = sampler(true);

            // A 1x1 opaque-white texel: the fallback an untextured draw binds so the one
            // shader path serves both textured and vertex-color draws.
            let white_tex = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("gxm-white"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &[255, 255, 255, 255],
            );
            let white = white_tex.create_view(&Default::default());
            let white_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gxm-white-bind"),
                layout: &texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&white),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler_point),
                    },
                ],
            });

            let align = device.limits().min_uniform_buffer_offset_alignment as u64;
            let uniform_stride = align_up(UNIFORM_BYTES, align.max(1));

            // The supersample resolve: a fullscreen triangle that box-averages each
            // `scale x scale` block of the offscreen colour target into one output pixel, the
            // exact integer box `Framebuffer::downsampled` applies on the software oracle so the
            // two AA'd frames stay in lockstep. `textureLoad` (no sampler) does the exact taps.
            let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gxm-resolve"),
                source: wgpu::ShaderSource::Wgsl(RESOLVE_SHADER.into()),
            });
            let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gxm-resolve-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(16),
                        },
                        count: None,
                    },
                ],
            });
            let resolve_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gxm-resolve-pl"),
                bind_group_layouts: &[Some(&resolve_layout)],
                immediate_size: 0,
            });
            let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gxm-resolve-pipe"),
                layout: Some(&resolve_pl),
                vertex: wgpu::VertexState {
                    module: &resolve_shader,
                    entry_point: Some("vres"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &resolve_shader,
                    entry_point: Some("fres"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });
            let resolve_scale_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gxm-resolve-scale"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // The guest-depth conversion pass. Built unconditionally (it is one small
            // pipeline) but only ever ENCODED for a target whose depth some pass samples.
            let gxm_depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gxm-depth-convert"),
                source: wgpu::ShaderSource::Wgsl(GXM_DEPTH_SHADER.into()),
            });
            let gxm_depth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gxm-depth-convert-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(GXM_DEPTH_UNIFORM_BYTES),
                        },
                        count: None,
                    },
                ],
            });
            let gxm_depth_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gxm-depth-convert-pl"),
                bind_group_layouts: &[Some(&gxm_depth_layout)],
                immediate_size: 0,
            });
            let gxm_depth_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gxm-depth-convert-pipe"),
                layout: Some(&gxm_depth_pl),
                vertex: wgpu::VertexState {
                    module: &gxm_depth_shader,
                    entry_point: Some("vdep"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &gxm_depth_shader,
                    entry_point: Some("fdep"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: GXM_DEPTH_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });
            let gxm_depth_uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gxm-depth-convert-uniform"),
                size: GXM_DEPTH_UNIFORM_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            GxmRenderer {
                srgb,
                gxm_depth_pipe,
                gxm_depth_layout,
                gxm_depth_uniform,
                opaque,
                blend,
                uniform_layout,
                texture_layout,
                sampler_point,
                sampler_linear,
                white_bind,
                views: HashMap::new(),
                views_bytes: 0,
                tex_binds: HashMap::new(),
                vbo: None,
                ibo: None,
                ubo: None,
                ubo_bind: None,
                vbo_cap: 0,
                ibo_cap: 0,
                ubo_cap: 0,
                gxp_pass_gen: 0,
                uniform_stride,
                color_format,
                ss_scale: 1,
                last_gxp_summary: None,
                last_phases: EncodePhases::default(),
                chain_phases: EncodePhases::default(),
                resolve_pipeline,
                resolve_layout,
                resolve_scale_buf,
                ss_target: None,
                rtt: HashMap::new(),
                rtt_rendered: HashMap::new(),
                rtt_binds: HashMap::new(),
                rtt_reads_snapshot: HashSet::new(),
                rtt_depth_rendered: HashMap::new(),
                keep_depth: false,
                rtt_hits: 0,
                last_chain_shape: None,
                sampled_addrs: HashSet::new(),
                gxp: GxpLive::from_env(),
            }
        }

        /// Set the supersample factor: 1 (default) renders the scene straight into the caller's
        /// target; `scale > 1` renders it at `scale x` the target dimensions into an offscreen
        /// buffer and box-downsamples that into the caller's view on resolve. This antialiases
        /// the geometric aliasing of a heavily-tessellated distant mesh (many sub-pixel triangles
        /// per final pixel) and coincident-panel z-fighting - the vehicle-speckle a 1x render
        /// shows - and mirrors the software oracle's `VITASLOP_SSAA` / `Framebuffer::downsampled`
        /// so both paths stay in lockstep. Cost is `scale^2` fill; 2x is the quality/perf
        /// default a caller opts into. A no-op if `scale` is unchanged.
        pub fn set_supersample(&mut self, scale: u32) {
            let scale = scale.max(1);
            if scale != self.ss_scale {
                self.ss_scale = scale;
                self.ss_target = None; // force a rebuild at the new scale
            }
        }

        /// Ensure the offscreen supersample target exists at `scale * (surf_w, surf_h)`,
        /// rebuilding it if the scale or size changed. Also refreshes the scale uniform.
        fn ensure_ss_target(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, surf_w: u32, surf_h: u32) {
            let scale = self.ss_scale;
            let (w, h) = (surf_w * scale, surf_h * scale);
            let stale = self.ss_target.as_ref().map(|t| t.scale != scale || t.width != w || t.height != h).unwrap_or(true);
            if !stale {
                return;
            }
            let color = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-ss-color"),
                size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let color_view = color.create_view(&Default::default());
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-ss-depth"),
                size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let depth_view = depth.create_view(&Default::default());
            queue.write_buffer(&self.resolve_scale_buf, 0, &[scale, 0, 0, 0].iter().flat_map(|v: &u32| v.to_le_bytes()).collect::<Vec<u8>>());
            let resolve_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gxm-resolve-bind"),
                layout: &self.resolve_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&color_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: self.resolve_scale_buf.as_entire_binding() },
                ],
            });
            self.ss_target = Some(SsTarget { scale, width: w, height: h, _color: color, color_view, depth_view, resolve_bind });
        }

        /// Ensure a decoded texture is uploaded (cached by content key) and that a bind
        /// group exists for it at the requested filter. A cache hit does no GPU work.
        fn ensure_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, t: &super::GxmTexture) {
            // Bound the caches BY BYTES: on pathological churn, clear wholesale and
            // re-upload (keys are content fingerprints, so correctness is unaffected).
            // See `tex_cache_budget_bytes` for why an entry COUNT bounded nothing.
            if !self.views.contains_key(&t.key) {
                self.views_bytes += texture_bytes(t.width, t.height);
                if self.views_bytes >= tex_cache_budget_bytes() {
                    self.views.clear();
                    self.tex_binds.clear();
                    self.views_bytes = texture_bytes(t.width, t.height);
                }
            }
            if !self.views.contains_key(&t.key) {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("gxm-tex"),
                    size: wgpu::Extent3d {
                        width: t.width,
                        height: t.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &t.rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(t.width * 4),
                        rows_per_image: Some(t.height),
                    },
                    wgpu::Extent3d { width: t.width, height: t.height, depth_or_array_layers: 1 },
                );
                self.views.insert(t.key, tex.create_view(&Default::default()));
            }
            let bind_key = (t.key, t.filter_linear);
            if !self.tex_binds.contains_key(&bind_key) {
                let view = &self.views[&t.key];
                let samp = if t.filter_linear { &self.sampler_linear } else { &self.sampler_point };
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gxm-tex-bind"),
                    layout: &self.texture_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(samp),
                        },
                    ],
                });
                self.tex_binds.insert(bind_key, bind);
            }
        }

        /// Grow (or first-create) a per-frame arena buffer so it holds at least `need`
        /// bytes. Returns true if the buffer was (re)created (the caller rebinds if so).
        fn ensure_buffer(
            device: &wgpu::Device,
            buf: &mut Option<wgpu::Buffer>,
            cap: &mut u64,
            need: u64,
            usage: wgpu::BufferUsages,
            label: &str,
        ) -> bool {
            let need = need.max(4);
            if buf.is_some() && *cap >= need {
                return false;
            }
            // Grow geometrically so a steadily-larger frame does not reallocate every time.
            let new_cap = need.next_power_of_two().max(4096);
            *buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: new_cap,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            *cap = new_cap;
            true
        }

        /// The group-1 bind group for an item (the shared white fallback, or a cached
        /// texture bind ensured earlier this frame).
        fn bind_for(&self, key: BindKey) -> &wgpu::BindGroup {
            match key {
                BindKey::White => &self.white_bind,
                BindKey::Tex(k, l) => &self.tex_binds[&(k, l)],
                BindKey::Rtt(a, l, s) => &self.rtt_binds[&(a, l, s)],
            }
        }

        /// The offscreen target for the colour surface at guest address `addr`, created (or
        /// re-created at a new size) on demand.
        ///
        /// `sample_depth` says some later pass in this frame reads this target's DEPTH, which
        /// costs the depth texture a `TEXTURE_BINDING` usage and a converted R32Float
        /// companion. It is off for every target nothing samples, so an ordinary pass pays
        /// nothing for the feature.
        fn ensure_rtt(
            &mut self,
            device: &wgpu::Device,
            addr: u32,
            width: u32,
            height: u32,
            sample_depth: bool,
        ) {
            let stale = match self.rtt.get(&addr) {
                // Gaining a depth reader is as much a reason to rebuild as a resize: the depth
                // texture it already has was created without `TEXTURE_BINDING` and cannot be
                // sampled.
                Some(t) => t.width != width || t.height != height || (sample_depth && t.gxm_depth.is_none()),
                None => true,
            };
            if !stale {
                return;
            }
            let size = wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 };
            // Declare the sRGB twin as an allowed view format on EVERY target, not only the
            // ones currently in gamma mode: `sceGxmColorSurfaceSetGammaMode` is sticky state a
            // title may set at any point, and a texture's view formats are fixed at creation.
            // Declaring it costs nothing on any backend that matters and removes a whole class
            // of "the mode arrived after the target existed" bug.
            let srgb_fmt = srgb_twin(self.color_format);
            let view_formats: Vec<wgpu::TextureFormat> = srgb_fmt.into_iter().collect();
            let color = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-rtt-color"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &view_formats,
            });
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-rtt-depth"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: if sample_depth {
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
                } else {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                },
                view_formats: &[],
            });
            let color_view = color.create_view(&Default::default());
            let color_view_srgb = srgb_fmt.map(|f| {
                color.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("gxm-rtt-color-srgb"),
                    format: Some(f),
                    ..Default::default()
                })
            });
            let depth_view = depth.create_view(&Default::default());
            let gxm_depth = sample_depth.then(|| {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("gxm-rtt-guest-depth"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: GXM_DEPTH_FORMAT,
                    // COPY_SRC so a host that can read back (the headless chain dump) can
                    // report what the conversion actually produced. A depth buffer nobody can
                    // look at is a depth buffer nobody can debug.
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let view = tex.create_view(&Default::default());
                let src_view = depth.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("gxm-rtt-depth-sampled"),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    ..Default::default()
                });
                GxmDepthTarget { src_view, _tex: tex, view }
            });
            // A bind group over the new view is stale by construction; drop any cached ones.
            self.rtt_binds.retain(|&(a, _, _), _| a != addr);
            self.rtt.insert(
                addr,
                RttSurface {
                    width,
                    height,
                    color,
                    color_view,
                    color_view_srgb,
                    depth_view,
                    shadow: None,
                    gxm_depth,
                },
            );
        }

        /// Run the depth-conversion pass for the target at `addr`: read its depth attachment
        /// and write the guest's own depth encoding into its R32Float companion.
        ///
        /// Called right after the pass that filled the depth, because the next pass into the
        /// same target clears it again.
        fn convert_gxm_depth(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            encoder: &mut wgpu::CommandEncoder,
            addr: u32,
            depth_min: f32,
            depth_scale: f32,
        ) {
            let Some(t) = self.rtt.get(&addr) else { return };
            let Some(gd) = t.gxm_depth.as_ref() else { return };
            let (mode, konst) = gxm_depth_encoding();
            // The SAME fit the pass's own fragments use for their window POSITION - the two are
            // one quantity or every comparison between them is meaningless.
            let (fit_a, fit_c) = self.gxp.depth_fit_for(addr);
            let mut u = Vec::with_capacity(GXM_DEPTH_UNIFORM_BYTES as usize);
            u.extend_from_slice(&depth_min.to_le_bytes());
            u.extend_from_slice(&depth_scale.to_le_bytes());
            u.extend_from_slice(&mode.to_le_bytes());
            u.extend_from_slice(&konst.to_le_bytes());
            u.extend_from_slice(&fit_a.to_le_bytes());
            u.extend_from_slice(&fit_c.to_le_bytes());
            u.extend_from_slice(&0f32.to_le_bytes());
            u.extend_from_slice(&0f32.to_le_bytes());
            queue.write_buffer(&self.gxm_depth_uniform, 0, &u);
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gxm-depth-convert-bind"),
                layout: &self.gxm_depth_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&gd.src_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: self.gxm_depth_uniform.as_entire_binding() },
                ],
            });
            let view = gd.view.clone();
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gxm-depth-convert"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.gxm_depth_pipe);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..3, 0..1);
            }
            report_depth_conversion(addr, mode, konst, depth_min, depth_scale);
        }

        /// Copy the rendered target at `addr` into a side texture and return a view of it.
        ///
        /// A pass is entitled to render into a buffer it also samples - a post-process
        /// stage reading the previous stage in place is the common shape - but a texture
        /// cannot be a colour target and a sampled resource in the same pass (wgpu rejects
        /// it, and the hardware read would be undefined). The snapshot is the buffer as it
        /// stood BEFORE this pass, which is exactly what such a pass means to read.
        fn snapshot_rtt(
            &mut self,
            device: &wgpu::Device,
            encoder: &mut wgpu::CommandEncoder,
            addr: u32,
        ) -> Option<wgpu::TextureView> {
            let color_format = self.color_format;
            let t = self.rtt.get_mut(&addr)?;
            let size = wgpu::Extent3d { width: t.width.max(1), height: t.height.max(1), depth_or_array_layers: 1 };
            if t.shadow.is_none() {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("gxm-rtt-shadow"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: color_format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = tex.create_view(&Default::default());
                t.shadow = Some((tex, view));
            }
            let (shadow_tex, shadow_view) = t.shadow.as_ref()?;
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &t.color,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: shadow_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                size,
            );
            Some(shadow_view.clone())
        }

        /// Ensure a fixed-function bind group exists for sampling `view` - the rendered
        /// target at `addr`, or its snapshot - with the given filter.
        fn ensure_rtt_bind(
            &mut self,
            device: &wgpu::Device,
            addr: u32,
            linear: bool,
            snapshot: bool,
            view: &wgpu::TextureView,
        ) {
            if self.rtt_binds.contains_key(&(addr, linear, snapshot)) {
                return;
            }
            let samp = if linear { &self.sampler_linear } else { &self.sampler_point };
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gxm-rtt-bind"),
                layout: &self.texture_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(samp) },
                ],
            });
            self.rtt_binds.insert((addr, linear, snapshot), bind);
        }

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
            clear: [u8; 4],
        ) {
            self.rtt_rendered.clear();
            self.rtt_depth_rendered.clear();
            self.rtt_hits = 0;
            self.chain_phases = EncodePhases::default();
            let Some(last) = scenes.last() else { return };
            // Which scenes' DEPTH some draw in this frame samples. Scanned up front because
            // the reader comes AFTER the writer: the pass that fills a depth buffer has to
            // know, while it is being set up, that something later will read it - a target
            // built without `TEXTURE_BINDING` cannot be sampled at all, and the extra
            // conversion is not worth paying on the targets nothing reads.
            let sampled: HashSet<u32> = scenes
                .iter()
                .flat_map(|s| s.draws.iter())
                .flat_map(|d| {
                    // BOTH stages. A vertex program that samples a depth surface (a
                    // displacement or reprojection built in the vertex stage) names it exactly
                    // as a fragment one does, and leaving vertex samplers out of this set makes
                    // the pass that WROTE that depth skip the conversion - so the reader gets
                    // raw hardware depth with nothing saying so.
                    d.texture.iter().map(|t| t.data_addr).chain(
                        d.gxp.iter().flat_map(|g| {
                            g.textures
                                .iter()
                                .chain(g.vertex_textures.iter())
                                .map(|t| t.tex.data_addr)
                        }),
                    )
                })
                .collect();
            let depth_sampled: HashSet<u32> = scenes
                .iter()
                .map(|s| s.depth_addr)
                .filter(|a| *a != 0 && sampled.contains(a))
                .collect();
            // The display buffer is whatever the final scene draws to. Any earlier scene
            // naming the same address is part of the same image, not an offscreen pass.
            let display = last.target.map(|t| t.data_addr);
            let ss = self.ss_scale > 1;
            if ss {
                self.ensure_ss_target(device, queue, surf_w, surf_h);
            }
            let mut display_pass_done = false;
            let n = scenes.len();
            for (i, scene) in scenes.iter().enumerate() {
                let to_display = i + 1 == n || (scene.target.map(|t| t.data_addr) == display && display.is_some());
                if to_display {
                    let (cv, dv) = match (ss, self.ss_target.as_ref()) {
                        (true, Some(t)) => (t.color_view.clone(), t.depth_view.clone()),
                        _ => (color_view.clone(), depth_view.clone()),
                    };
                    let first = !display_pass_done;
                    display_pass_done = true;
                    self.rtt_reads_snapshot.clear();
                    // The display target's format belongs to the surface the host handed us,
                    // and the host owns whether that is already an sRGB swapchain - so a
                    // gamma-mode DISPLAY surface is not reinterpreted here.
                    let fmt = self.color_format;
                    self.encode_pass(device, queue, encoder, &cv, &dv, fmt, scene, surf_w, surf_h, first.then_some(clear));
                    // A display pass keeps no depth copy (its depth attachment belongs to the
                    // caller and is discarded), so if something reads this scene's depth it
                    // will not find it. Say so rather than let the read fall through silently.
                    if depth_sampled.contains(&scene.depth_addr) {
                        report_unconverted_depth_sample(scene.depth_addr);
                    }
                    continue;
                }
                let Some(t) = scene.target else {
                    report_unplaced_scene(scene.draws.len());
                    continue;
                };
                let want_depth = depth_sampled.contains(&scene.depth_addr);
                self.ensure_rtt(device, t.data_addr, t.width, t.height, want_depth);
                // Drawing into a buffer this frame already filled, which this pass may also
                // sample: hand it a snapshot to read so the live buffer is a target only.
                self.rtt_reads_snapshot.clear();
                // Clear a target the FIRST time this frame draws into it, and compose onto
                // it after that - the same rule the display buffer follows. A later pass
                // into a buffer an earlier pass filled is a post-process step, and it is
                // entitled to leave most of the image alone: a retail racer's race frame ends
                // with a five-draw pass over the world target, and clearing for it wiped the
                // whole world, leaving a correct HUD over black.
                let first_pass_here = !self.rtt_rendered.contains_key(&t.data_addr);
                if !first_pass_here {
                    if let Some(before) = self.snapshot_rtt(device, encoder, t.data_addr) {
                        self.rtt_rendered.insert(t.data_addr, before);
                        self.rtt_reads_snapshot.insert(t.data_addr);
                    }
                }
                // A GAMMA-CORRECT surface is rendered through the sRGB view of the same
                // texture, so the ROP encodes each store after blending exactly as the
                // hardware does. The same view goes into `rtt_rendered` below, so a later pass
                // sampling this target decodes on the way back in.
                let gamma = t.gamma;
                let (cv, dv, fmt) = {
                    let s = &self.rtt[&t.data_addr];
                    match (gamma, s.color_view_srgb.as_ref()) {
                        (true, Some(v)) => (
                            v.clone(),
                            s.depth_view.clone(),
                            srgb_twin(self.color_format).unwrap_or(self.color_format),
                        ),
                        _ => (s.color_view.clone(), s.depth_view.clone(), self.color_format),
                    }
                };
                if gamma {
                    report_gamma_surface(t.data_addr, fmt != self.color_format);
                }
                // A first pass is cleared to transparent black, not to the display's clear
                // colour: it is an intermediate image, and a composite that blends it must
                // see nothing where the pass drew nothing.
                let clear = first_pass_here.then_some([0, 0, 0, 0]);
                self.keep_depth = want_depth;
                self.encode_pass(device, queue, encoder, &cv, &dv, fmt, scene, t.width, t.height, clear);
                self.keep_depth = false;
                self.rtt_rendered.insert(t.data_addr, cv);
                // Convert this pass's depth NOW: the next pass into the same target clears the
                // depth attachment, so afterwards there is nothing left to convert.
                if want_depth {
                    self.convert_gxm_depth(
                        device,
                        queue,
                        encoder,
                        t.data_addr,
                        scene.depth_min,
                        scene.depth_scale,
                    );
                    if let Some(v) = self.rtt.get(&t.data_addr).and_then(|s| s.gxm_depth.as_ref()) {
                        self.rtt_depth_rendered.insert(scene.depth_addr, v.view.clone());
                    }
                }
            }
            // Report the frame's pass structure whenever it CHANGES: how many scenes, where
            // each one draws, and - the load-bearing number - how many draws sampled a
            // target this frame rendered. A chain of passes with zero samples of them means
            // the composite never reads the world, which looks exactly like the passes not
            // being drawn at all but is a different bug.
            if scenes.len() > 1 || self.rtt_hits > 0 {
                let shape = scenes
                    .iter()
                    .map(|s| match s.target {
                        Some(t) => format!("{:#x}:{}x{}/{}", t.data_addr, t.width, t.height, s.draws.len()),
                        None => format!("?/{}", s.draws.len()),
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                let mut sampled: Vec<String> =
                    self.sampled_addrs.iter().map(|a| format!("{a:#x}")).collect();
                sampled.sort();
                let line = format!(
                    "gxm chain: {} scenes [{shape}] rtt-samples={} final-pass-sampled=[{}]",
                    scenes.len(),
                    self.rtt_hits,
                    sampled.join(" ")
                );
                if self.last_chain_shape.as_deref() != Some(line.as_str()) {
                    report!("{line}");
                    self.last_chain_shape = Some(line);
                }
            }
            // Resolve the supersampled display buffer once, after the last pass into it.
            if let (true, Some(t)) = (ss, self.ss_target.as_ref()) {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gxm-resolve"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rpass.set_pipeline(&self.resolve_pipeline);
                rpass.set_bind_group(0, &t.resolve_bind, &[]);
                rpass.draw(0..3, 0..1);
            }
        }

        /// Encode a full scene into `encoder`: a render pass over `color_view` (cleared
        /// to `clear`) with `depth_view` (must be [`DEPTH_FORMAT`]), drawing every draw
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
                clear,
            );
        }

        /// One scene into one target. `clear` is `Some(colour)` for the first pass into a
        /// target and `None` to compose onto what is already there. Supersampling is the
        /// caller's business ([`encode_chain`](Self::encode_chain) picks the views and
        /// resolves once), so this is a plain pass.
        #[allow(clippy::too_many_arguments)]
        fn encode_pass(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            encoder: &mut wgpu::CommandEncoder,
            color_view: &wgpu::TextureView,
            depth_view: &wgpu::TextureView,
            // The FORMAT of `color_view`. Usually the renderer's `color_format`, but the sRGB
            // twin when this pass renders into a gamma-correct surface - and a pipeline is
            // bound to the format of the attachment it writes, so every pipeline this pass
            // uses has to be built for this format, not for the renderer's default.
            target_format: wgpu::TextureFormat,
            scene: &RenderScene,
            surf_w: u32,
            surf_h: u32,
            clear: Option<[u8; 4]>,
        ) {
            // 1. Walk the scene once: pack vertex/index/uniform bytes into per-frame
            //    arenas and ensure each draw's texture upload + bind group exist.
            // Before any draw is prepared: does THIS pass's projection put clip `w` negative in
            // front of the camera? It is one answer for the whole pass and every draw's
            // `@group(3)` block carries it, so it has to be settled first.
            self.gxp.decide_scene_negw(scene);
            let t_prepare = Stopwatch::start();
            let stride = self.uniform_stride as usize;
            let mut vdata: Vec<u8> = Vec::new();
            let mut idata: Vec<u8> = Vec::new();
            let mut udata: Vec<u8> = Vec::new();
            // The recompiled path's arenas (see `GxpPrepared`), packed during the same walk.
            let mut gvdata: Vec<u8> = Vec::new();
            let mut gidata: Vec<u8> = Vec::new();
            let mut gudata: Vec<u8> = Vec::new();
            let ubo_align = device.limits().min_uniform_buffer_offset_alignment as u64;
            let mut items: Vec<Item> = Vec::with_capacity(scene.draws.len());
            // The live recompiler's per-draw resources + a submission-order plan interleaving
            // recompiled and fixed-function draws (so they share one depth-tested pass).
            let gxp_enabled = self.gxp.enabled;
            let gxp_only = self.gxp.only;
            // The attachment this pass writes, NOT the renderer's default - see the parameter.
            let color_format = target_format;
            let mut gxp_prepared: Vec<GxpPrepared> = Vec::new();
            let mut order: Vec<Enc> = Vec::with_capacity(scene.draws.len());
            // Taken out for the walk so the render-target views can be read while the
            // texture caches next to them are written; restored below.
            let rendered = std::mem::take(&mut self.rtt_rendered);
            let depth_rendered = std::mem::take(&mut self.rtt_depth_rendered);
            self.sampled_addrs.clear();
            // `VITASLOP_CHAIN_DRAWS=1`: describe every draw in this pass that samples a
            // target the frame rendered. A composite that shows none of the world has
            // either no such draw, or one whose blend/space/geometry throws it away, and
            // the finished (black) frame cannot tell those apart.
            let trace_draws = std::env::var_os("VITASLOP_CHAIN_DRAWS").is_some();
            // `=all` describes EVERY draw of the pass, not only the ones sampling a
            // rendered target - what is needed when the question is "which draw was
            // supposed to put the world on screen" rather than "did this one bind right".
            let trace_all = std::env::var("VITASLOP_CHAIN_DRAWS").map(|v| v == "all").unwrap_or(false);
            // Fallback draws of THIS pass, by reason - see `fallback_reasons`.
            let mut fb_reasons: HashMap<String, usize> = HashMap::new();
            for (di, d) in scene.draws.iter().enumerate() {
                if trace_draws {
                    // The sampled texture's own dimensions matter as much as its address: a
                    // title may bind a render target through a texture describing a
                    // different extent, and the UVs the shader computes are for THAT extent.
                    let hits: Vec<String> = d
                        .texture
                        .iter()
                        .map(|t| (t.data_addr, t.width, t.height))
                        .chain(d.gxp.iter().flat_map(|g| {
                            g.textures
                                .iter()
                                .chain(g.vertex_textures.iter())
                                .map(|t| (t.tex.data_addr, t.tex.width, t.tex.height))
                        }))
                        .filter(|(a, _, _)| trace_all || rendered.contains_key(a))
                        .map(|(a, w, h)| {
                            format!("{a:#x}({w}x{h}){}", if rendered.contains_key(&a) { "*" } else { "" })
                        })
                        .collect();
                    if !hits.is_empty() || trace_all {
                        report!(
                            // "carries a payload" is NOT "is recompiled" - a payload that
                            // fails to link falls back, and labelling that `recompiled=true`
                            // is how a composite draw got read as working when it was not.
                            // The index count is the RECOMPILED one when there is a payload.
                            // The fixed-function count is zero for such a draw by design (the
                            // builder does not produce a representation the renderer will not
                            // use), and printing that made every draw of a recompiled pass
                            // read as empty geometry - a diagnostic saying exactly the wrong
                            // thing about the question it exists to answer.
                            "chain draw #{di}: samples {:?} key={:?} has_payload={} blend={:?} opaque={} space={:?} idx={}",
                            hits,
                            d.gxp.as_ref().map(|g| format!("{:x}", GxpLive::key(g))),
                            d.gxp.is_some(),
                            d.gxp.as_ref().map(|g| g.blend),
                            d.opaque,
                            d.space,
                            d.gxp.as_ref().map(|g| g.index_count).unwrap_or(d.index_count)
                        );
                    }
                }
                if let Some(t) = &d.texture {
                    self.sampled_addrs.insert(t.data_addr);
                }
                if let Some(g) = &d.gxp {
                    // Vertex samplers count too: `final-pass-sampled` is the list a reader
                    // checks a missing target against, and a target read only by a vertex
                    // program was invisible in it.
                    self.sampled_addrs.extend(
                        g.textures.iter().chain(g.vertex_textures.iter()).map(|t| t.tex.data_addr),
                    );
                }
                // Live GXP path: draw with the guest's real shaders when the pair links. On a
                // link/format failure fall through to the fixed-function packing below (unless
                // isolate mode, which renders only the recompiled draws).
                if gxp_enabled {
                    if let Some(g) = &d.gxp {
                        self.rtt_hits += g
                            .textures
                            .iter()
                            .chain(g.vertex_textures.iter())
                            .filter(|t| rendered.contains_key(&t.tex.data_addr))
                            .count();
                        if let Some(mut prep) =
                            self.gxp.prepare(device, queue, color_format, g, [scene.depth_min, scene.depth_scale], &rendered, &depth_rendered, &mut gvdata, &mut gidata, &mut gudata, ubo_align)
                        {
                            if self.gxp.solid {
                                prep.blend = false; // REPLACE + depth-Always variant (see make)
                            }
                            order.push(Enc::Gxp(gxp_prepared.len()));
                            gxp_prepared.push(prep);
                            continue;
                        }
                        // Prepared failed: this draw is one of the pass's fallbacks. Tally it
                        // against its pair's reason so the summary can rank causes by draws.
                        *fb_reasons.entry(fallback_reason_of(GxpLive::key(g))).or_insert(0) += 1;
                        if gxp_only {
                            continue;
                        }
                    } else if gxp_only {
                        continue;
                    }
                }
                // A shader-only draw has no colour source the fixed-function packing can
                // honour, so reaching here means the recompiler could not draw it and the
                // alternative would be an opaque white rectangle. Skip it - but never
                // silently: this is a draw the guest asked for that the frame will not show.
                if d.shader_only {
                    crate::gpu::report_shader_only_skip(d, gxp_enabled);
                    continue;
                }
                if d.index_count == 0 || d.vertices.is_empty() {
                    continue;
                }
                let bind = match &d.texture {
                    None => BindKey::White,
                    // A texture whose data pointer is a target THIS frame already rendered
                    // is that render, not the guest bytes behind the pointer (which the GPU,
                    // not the guest, was supposed to have written).
                    Some(t) if rendered.contains_key(&t.data_addr) => {
                        self.rtt_hits += 1;
                        let snapshot = self.rtt_reads_snapshot.contains(&t.data_addr);
                        self.ensure_rtt_bind(device, t.data_addr, t.filter_linear, snapshot, &rendered[&t.data_addr]);
                        BindKey::Rtt(t.data_addr, t.filter_linear, snapshot)
                    }
                    Some(t) => {
                        self.ensure_texture(device, queue, t);
                        BindKey::Tex(t.key, t.filter_linear)
                    }
                };
                let (mode, mvp) = match d.space {
                    DrawSpace::Mvp(m) => (0u32, m),
                    DrawSpace::Ndc => (1u32, [0f32; 16]),
                    DrawSpace::Pixel => (2u32, [0f32; 16]),
                };
                // Uniform block (must match the WGSL `U` struct), then pad to the
                // dynamic-offset stride. A vec4 in WGSL is 16-byte aligned, so each material
                // vec3 is written as 4 floats (xyz + a pad lane).
                let uniform_offset = udata.len() as u32;
                for v in &mvp {
                    udata.extend_from_slice(&v.to_le_bytes());
                }
                udata.extend_from_slice(&mode.to_le_bytes());
                udata.extend_from_slice(&(surf_w as f32).to_le_bytes());
                udata.extend_from_slice(&(surf_h as f32).to_le_bytes());
                udata.extend_from_slice(&(d.texture.is_some() as u32).to_le_bytes());
                udata.extend_from_slice(&d.exposure.to_le_bytes());
                udata.extend_from_slice(&scene.depth_min.to_le_bytes());
                udata.extend_from_slice(&scene.depth_scale.to_le_bytes());
                udata.extend_from_slice(&[0u8; 4]); // pad2
                let vec4 = |udata: &mut Vec<u8>, v: [f32; 3]| {
                    for c in v {
                        udata.extend_from_slice(&c.to_le_bytes());
                    }
                    udata.extend_from_slice(&[0u8; 4]); // vec4 pad lane
                };
                vec4(&mut udata, d.material.tint);
                vec4(&mut udata, d.material.light_dir);
                vec4(&mut udata, d.material.light_col);
                vec4(&mut udata, d.material.ambient);
                udata.resize(uniform_offset as usize + stride, 0);

                let v_off = vdata.len() as u64;
                vdata.extend_from_slice(&d.vertices);
                let i_off = idata.len() as u64;
                idata.extend_from_slice(&d.indices);

                items.push(Item {
                    v_off,
                    v_len: d.vertices.len() as u64,
                    i_off,
                    i_len: d.indices.len() as u64,
                    index_count: d.index_count,
                    uniform_offset,
                    opaque: d.opaque,
                    bind,
                });
                order.push(Enc::Fixed(items.len() - 1));
            }
            self.rtt_rendered = rendered;
            self.rtt_depth_rendered = depth_rendered;
            if gxp_enabled {
                let with_payload = scene.draws.iter().filter(|d| d.gxp.is_some()).count();
                let summary = (scene.draws.len(), with_payload, gxp_prepared.len(), items.len());
                if self.last_gxp_summary != Some(summary) {
                    self.last_gxp_summary = Some(summary);
                    report!(
                        "gxp: scene has {} draws, {} carry a shader payload, {} recompiled+prepared, {} fixed-function items",
                        summary.0, summary.1, summary.2, summary.3,
                    );
                    // Rank the causes by the draws they cost, biggest first. Without this the
                    // only ranking available is the printed pair list, which counts pairs.
                    let mut ranked: Vec<_> = fb_reasons.iter().collect();
                    ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
                    for (reason, n) in ranked {
                        report_warn!("gxp:   {n} fallback draws - {reason}");
                    }
                }
            }

            self.last_phases = EncodePhases {
                prepare_ms: t_prepare.ms(),
                upload_ms: 0.0,
                pass_ms: 0.0,
                gxp_draws: gxp_prepared.len(),
                fixed_draws: items.len(),
            };

            // 2. Size the arenas and upload. Rebuild the uniform bind group if the uniform
            //    buffer was (re)created.
            let t_upload = Stopwatch::start();
            if !items.is_empty() {
                Self::ensure_buffer(device, &mut self.vbo, &mut self.vbo_cap, vdata.len() as u64, wgpu::BufferUsages::VERTEX, "gxm-vbo");
                Self::ensure_buffer(device, &mut self.ibo, &mut self.ibo_cap, idata.len() as u64, wgpu::BufferUsages::INDEX, "gxm-ibo");
                let ubo_new = Self::ensure_buffer(device, &mut self.ubo, &mut self.ubo_cap, udata.len() as u64, wgpu::BufferUsages::UNIFORM, "gxm-ubo");
                if ubo_new || self.ubo_bind.is_none() {
                    let ubo = self.ubo.as_ref().unwrap();
                    self.ubo_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("gxm-ubo-bind"),
                        layout: &self.uniform_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: ubo,
                                offset: 0,
                                size: wgpu::BufferSize::new(UNIFORM_BYTES),
                            }),
                        }],
                    }));
                }
                queue.write_buffer(self.vbo.as_ref().unwrap(), 0, &vdata);
                queue.write_buffer(self.ibo.as_ref().unwrap(), 0, &idata);
                queue.write_buffer(self.ubo.as_ref().unwrap(), 0, &udata);
            }
            // The recompiled path's arenas: THREE buffers for the whole pass, however many
            // draws it carries, instead of four per draw. They are created fresh per pass
            // rather than reused across the chain - see `gxp_pass_gen` for why - and stay
            // alive through submit because the command encoder holds a reference to every
            // resource its passes name.
            let gxp_arena = (!gxp_prepared.is_empty()).then(|| {
                let mk = |data: &[u8], usage, label| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        // An empty arena is not a legal buffer, and a pass whose every stage
                        // declares no uniforms produces one.
                        contents: if data.is_empty() { &[0u8; 4] } else { data },
                        usage,
                    })
                };
                (
                    mk(&gvdata, wgpu::BufferUsages::VERTEX, "gxp-vbo"),
                    mk(&gidata, wgpu::BufferUsages::INDEX, "gxp-ibo"),
                    mk(&gudata, wgpu::BufferUsages::UNIFORM, "gxp-ubo"),
                )
            });
            if let Some((_, _, ubo)) = &gxp_arena {
                self.gxp_pass_gen += 1;
                let pass_gen = self.gxp_pass_gen;
                let used: Vec<(u64, wgpu::TextureFormat)> =
                    gxp_prepared.iter().map(|p| (p.key, p.format)).collect();
                self.gxp.ensure_ubo_bgs(device, ubo, pass_gen, &used);
            }

            self.last_phases.upload_ms = t_upload.ms();

            // 3. One render pass over the whole scene. When supersampling, the scene is drawn
            //    into the offscreen `scale x` target (built here) and a resolve pass below
            //    box-downsamples it into the caller's view; otherwise it is drawn straight in.
            let t_pass = Stopwatch::start();
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gxm-scene"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: match clear {
                                Some(c) => wgpu::LoadOp::Clear(wgpu::Color {
                                    r: c[0] as f64 / 255.0,
                                    g: c[1] as f64 / 255.0,
                                    b: c[2] as f64 / 255.0,
                                    a: c[3] as f64 / 255.0,
                                }),
                                None => wgpu::LoadOp::Load,
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            // Discarded by default - nothing reads a depth attachment once its
                            // pass is over. A pass whose depth a LATER pass samples has to keep
                            // it, and only that pass pays the store.
                            store: if self.keep_depth {
                                wgpu::StoreOp::Store
                            } else {
                                wgpu::StoreOp::Discard
                            },
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if order.is_empty() {
                    // Clear-only pass; the descriptor above already did the work. The phase
                    // timing is still closed out, so a caller reading it never sees a stale
                    // value from an earlier, busier frame.
                    self.last_phases.pass_ms = t_pass.ms();
                    self.chain_phases.add(self.last_phases);
                    return;
                }
                // Draw in submission order, switching between the fixed-function arenas and the
                // recompiled per-draw resources. The fixed-function handles are unwrapped only
                // inside a Fixed arm, where `items` is non-empty so the arenas were uploaded.
                let ubo_bind = self.ubo_bind.as_ref();
                let vbo = self.vbo.as_ref();
                let ibo = self.ibo.as_ref();
                // Unwrapped only inside a `Gxp` arm, where `gxp_prepared` is non-empty so the
                // arenas above were created.
                let gxp_arena = gxp_arena.as_ref();
                // The guest's viewport is per-draw state, and `set_viewport` is sticky, so a
                // draw that wants the whole target after one that did not must SAY so. The
                // pass starts at the full rect (wgpu's default), and the tracker below issues
                // a change only when the requested rect actually differs - which on a title
                // whose every pass is fullscreen means it never issues one at all.
                let full = (0.0f32, 0.0f32, surf_w as f32, surf_h as f32);
                let mut cur_vp = full;
                for e in &order {
                    let want = match e {
                        Enc::Gxp(idx) => {
                            gxm_viewport_rect(&gxp_prepared[*idx].viewport, surf_w, surf_h).unwrap_or(full)
                        }
                        // The fixed-function path packs its own screen-space geometry and has
                        // never carried a viewport; it means the whole target.
                        Enc::Fixed(_) => full,
                    };
                    if want != cur_vp {
                        pass.set_viewport(want.0, want.1, want.2, want.3, 0.0, 1.0);
                        cur_vp = want;
                    }
                    match e {
                        Enc::Fixed(i) => {
                            let it = &items[*i];
                            let (ubo_bind, vbo, ibo) = (ubo_bind.unwrap(), vbo.unwrap(), ibo.unwrap());
                            // A fixed-function draw on a gamma-correct surface needs the sRGB
                            // variant, for the same reason a recompiled one does: a pipeline is
                            // bound to its attachment's format.
                            let (op, bl) = match (&self.srgb, target_format == self.color_format) {
                                (Some((o, b)), false) => (o, b),
                                _ => (&self.opaque, &self.blend),
                            };
                            pass.set_pipeline(if it.opaque { op } else { bl });
                            pass.set_bind_group(0, ubo_bind, &[it.uniform_offset]);
                            pass.set_bind_group(1, self.bind_for(it.bind), &[]);
                            pass.set_vertex_buffer(0, vbo.slice(it.v_off..it.v_off + it.v_len));
                            pass.set_index_buffer(ibo.slice(it.i_off..it.i_off + it.i_len), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..it.index_count, 0, 0..1);
                        }
                        Enc::Gxp(idx) => {
                            let p = &gxp_prepared[*idx];
                            let (gxp_vbo, gxp_ibo, _) = gxp_arena.unwrap();
                            let pipe = self.gxp.pipeline(p.key, p.format);
                            pass.set_pipeline(if p.blend { &pipe.blend } else { &pipe.opaque });
                            // group0/group1 belong to the PAIR and take this draw's byte offset
                            // into the pass's uniform arena; a stage with no uniforms has an
                            // empty bind group, which takes no dynamic offsets at all.
                            let dyn_off = |lanes: u32, off: u32| if lanes == 0 { Vec::new() } else { vec![off] };
                            pass.set_bind_group(
                                0,
                                self.gxp.ubo_bg(p.key, p.format, 0),
                                &dyn_off(pipe.vsa_lanes, p.u_off[0]),
                            );
                            pass.set_bind_group(
                                1,
                                self.gxp.ubo_bg(p.key, p.format, 1),
                                &dyn_off(pipe.fsa_lanes, p.u_off[1]),
                            );
                            pass.set_bind_group(2, &p.bg2, &[]);
                            pass.set_bind_group(3, &p.bg3, &[]);
                            pass.set_vertex_buffer(0, gxp_vbo.slice(p.v_off..p.v_off + p.v_len));
                            pass.set_index_buffer(
                                gxp_ibo.slice(p.i_off..p.i_off + p.i_len),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.draw_indexed(0..p.index_count, 0, 0..1);
                        }
                    }
                }
            }
            self.last_phases.pass_ms = t_pass.ms();
            self.chain_phases.add(self.last_phases);
        }

        /// What the last [`GxmRenderer::encode_chain`] spent, phase by phase, over EVERY pass
        /// of the frame. Reporting one pass instead described the composite and hid the world.
        pub fn last_phases(&self) -> EncodePhases {
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
            let mut v: Vec<_> =
                self.rtt.iter().map(|(&a, s)| (a, &s.color, s.width, s.height)).collect();
            v.sort_by_key(|t| t.0);
            v
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
                .filter_map(|(&a, s)| s.gxm_depth.as_ref().map(|d| (a, &d._tex, s.width, s.height)))
                .collect();
            v.sort_by_key(|t| t.0);
            v
        }
    }
}
