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
    pub width: u32,
    pub height: u32,
    /// Number of `width x height` RGBA8 images `rgba` holds back to back: 1 normally, 6 for a
    /// cube map (in +X, -X, +Y, -Y, +Z, -Z order, which is WebGPU's array-layer order).
    pub faces: u32,
    pub rgba: std::sync::Arc<Vec<u8>>,
    /// True if the guest set this texture's magnification filter to LINEAR
    /// (`SceGxmTextureFilter` == 1); the renderer then bilinear-samples it, matching
    /// the software rasterizer's `sample_texture_bilinear`. False = POINT/nearest.
    pub filter_linear: bool,
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
    /// Depth write enabled for this draw (GXM `front_depth_write != DISABLED`).
    pub depth_write: bool,
    /// GXM depth-compare function word (`SceGxmDepthFunc`).
    pub depth_func: u32,
    /// GXM cull-mode word (`SceGxmCullMode`).
    pub cull_mode: u32,
    /// Whether this draw is alpha-blended (a 2D/overlay draw, not opaque geometry).
    pub blend: bool,
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

/// A whole scene reduced to general draws, in submission order. The runtime builds
/// it from a captured [`Scene`](vitaslop_runtime-side); [`GxmRenderer`] draws it.
#[derive(Clone, Debug, Default)]
pub struct RenderScene {
    pub draws: Vec<GxmDraw>,
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
}

#[cfg(feature = "gpu")]
pub use render::{CubeRenderer, DEPTH_FORMAT};

#[cfg(feature = "gpu")]
pub use gxm::GxmRenderer;

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
    use std::collections::HashMap;
    use wgpu::util::DeviceExt;

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

    /// Bytes of the per-draw uniform block (matches the WGSL `U` struct): mat4 (64) +
    /// mode/surf_w/surf_h/textured (16) + exposure/depth_min/depth_scale/pad2 (16) +
    /// tint/light_dir/light_col/ambient (4 x vec4 = 64) = 160. Copies are laid into a
    /// shared buffer at [`GxmRenderer::uniform_stride`] spacing for dynamic offsets.
    const UNIFORM_BYTES: u64 = 160;

    /// Upper bound on the cross-frame texture caches before they are cleared wholesale
    /// (a re-upload, never incorrectness - the keys are content fingerprints, so a
    /// re-decoded atlas still hits). Mirrors the runtime's decode-cache cap; a title's
    /// working set is far smaller, so this only fires on pathological churn.
    const TEX_CACHE_CAP: usize = 512;

    /// Which texture bind group a draw uses. `White` is the shared 1x1 opaque-white
    /// fallback for an untextured draw; `Tex` names a cached upload by content key and
    /// whether it is sampled LINEAR (so the two filter modes cache separately).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum BindKey {
        White,
        Tex(u64, bool),
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
        uniform_layout: wgpu::BindGroupLayout,
        texture_layout: wgpu::BindGroupLayout,
        sampler_point: wgpu::Sampler,
        sampler_linear: wgpu::Sampler,
        white_bind: wgpu::BindGroup,
        /// Decoded texture uploads (a view kept alive), keyed by content fingerprint.
        views: HashMap<u64, wgpu::TextureView>,
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
        /// The box-downsample resolve pipeline + its bind-group layout and scale uniform,
        /// used only when `ss_scale > 1`. Built once in [`GxmRenderer::new`].
        resolve_pipeline: wgpu::RenderPipeline,
        resolve_layout: wgpu::BindGroupLayout,
        resolve_scale_buf: wgpu::Buffer,
        /// The lazily-(re)created offscreen supersample target (colour + depth + resolve bind
        /// group), sized to `ss_scale * surf`. Rebuilt when the scale or target size changes.
        ss_target: Option<SsTarget>,
        /// The live GXP->WGSL recompiler: a per-shader-pair pipeline cache. When enabled
        /// (`VITASLOP_GXP_LIVE`) a draw carrying [`super::GxpRecompile`] is rendered with the
        /// guest's real shaders; a pair that fails to link falls back to the fixed-function
        /// pipelines above. Disabled -> zero cost (the payload is simply ignored).
        gxp: GxpLive,
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
        vbuf: wgpu::Buffer,
        ibuf: wgpu::Buffer,
        index_count: u32,
        /// Bind groups for group0/1/2 (empty where the stage declares nothing).
        bg: [wgpu::BindGroup; 4],
        /// True = alpha-blended (2D/overlay), false = opaque geometry.
        blend: bool,
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
        zfix: bool,
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
        /// Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys
        /// (the `gxp draw key` value `VITASLOP_GXP_DUMP` prints), letting every other draw fall
        /// back. Rendering one pair at a time is how a visual artifact is attributed to the
        /// shader that produced it. Empty = no filter (recompile every linkable pair).
        keys: Vec<u64>,
        /// `key -> Some(pipeline)` for a linkable pair, `key -> None` for one that failed to
        /// link (cached so we never retry it and always fall back).
        pipelines: HashMap<u64, Option<GxpPipeline>>,
        /// Uploaded texture views, keyed by the decoded texture's content fingerprint and the
        /// view dimension it is bound as. A scene binds a handful of textures across hundreds
        /// of draws, so uploading per draw (as this path first did) re-sends the same
        /// multi-megabyte shadow map thousands of times a frame and exhausts GPU memory.
        views: HashMap<(u64, SamplerDim), wgpu::TextureView>,
        sampler_point: Option<wgpu::Sampler>,
        sampler_linear: Option<wgpu::Sampler>,
    }

    impl GxpLive {
        fn from_env() -> Self {
            let flag = |k: &str| std::env::var_os(k).is_some();
            GxpLive {
                enabled: flag("VITASLOP_GXP_LIVE"),
                only: flag("VITASLOP_GXP_ONLY"),
                zfix: std::env::var("VITASLOP_GXP_ZFIX").map(|v| v != "0").unwrap_or(true),
                yflip: std::env::var("VITASLOP_GXP_YFLIP").map(|v| v != "0").unwrap_or(false),
                force: flag("VITASLOP_GXP_FORCE"),
                solid: flag("VITASLOP_GXP_SOLID"),
                keys: std::env::var("VITASLOP_GXP_KEYS")
                    .ok()
                    .map(|v| {
                        v.split(',')
                            .filter_map(|k| u64::from_str_radix(k.trim().trim_start_matches("0x"), 16).ok())
                            .collect()
                    })
                    .unwrap_or_default(),
                pipelines: HashMap::new(),
                views: HashMap::new(),
                sampler_point: None,
                sampler_linear: None,
            }
        }

        /// Stable cache key for a shader pair: FNV-1a over the vertex then fragment blob.
        fn key(gxp: &GxpRecompile) -> u64 {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in gxp.vprog.iter().chain(gxp.fprog.iter()) {
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
        ) -> Option<GxpPrepared> {
            if gxp.index_count == 0 || gxp.vertices.is_empty() {
                return None;
            }
            let key = Self::key(gxp);
            if !self.keys.is_empty() && !self.keys.contains(&key) {
                return None;
            }
            if !self.pipelines.contains_key(&key) {
                let built = build_gxp_pipeline(device, color_format, gxp, self.zfix, self.yflip, self.solid);
                self.pipelines.insert(key, built);
            }
            if self.sampler_point.is_none() {
                self.sampler_point = Some(make_gxp_sampler(device, false));
                self.sampler_linear = Some(make_gxp_sampler(device, true));
            }
            // Split the borrows: the sampler bind group needs the texture-view cache mutably
            // while the pipeline (its layouts, its sampler plan) stays borrowed.
            let GxpLive { pipelines, views: view_cache, sampler_point, sampler_linear, force, .. } = self;
            // Borrow the cached pipeline; None = link failed -> fall back.
            let pipe = pipelines.get(&key)?.as_ref()?;

            if std::env::var_os("VITASLOP_GXP_DUMP").is_some() {
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
                eprintln!(
                    "gxp draw key {:x}: vsa_lanes={} vert_sa_lanes={} fsa_lanes={} frag_sa_lanes={} samplers={:?} stride={} idx={} nverts={nverts} max_index={max_index} vbytes={} attrs(reg,off,fmt,comp)={:?}\n  vsa={:?}\n  fsa={:?}",
                    key, pipe.vsa_lanes, f.len(), pipe.fsa_lanes, ff.len(), pipe.samplers, gxp.vertex_stride, gxp.index_count, gxp.vertices.len(), attrs, f, ff
                );
            }

            // Repack the guest vertex stream into the tightly-packed f32 layout the pipeline
            // expects (per the cached repack plan). Same vertex count/order, so the index
            // buffer is unchanged.
            let packed = repack_vertices(&gxp.vertices, gxp.vertex_stride, &pipe.repack, pipe.packed_stride);
            let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gxp-vbo"),
                contents: &packed,
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gxp-ibo"),
                contents: &gxp.indices,
                usage: wgpu::BufferUsages::INDEX,
            });

            let bg0 = make_uniform_bg(device, &pipe.layouts[0], pipe.vsa_lanes, &gxp.vert_sa);
            let bg1 = make_uniform_bg(device, &pipe.layouts[1], pipe.fsa_lanes, &gxp.frag_sa);
            let bg2 = Self::make_sampler_bg(
                device, queue, &pipe.layouts[2], &pipe.samplers, gxp,
                view_cache, sampler_point.as_ref().unwrap(), sampler_linear.as_ref().unwrap(), *force,
            )?;
            // group3: the scene depth range the injected clip fixup maps through, as one vec4
            // (min, scale, unused, unused) - the same values the fixed-function path uses, so
            // both kinds of draw write comparable depth.
            let dbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gxp-depth"),
                contents: &[depth_range[0].to_le_bytes(), depth_range[1].to_le_bytes(), [0; 4], [0; 4]].concat(),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gxp-depth-bind"),
                layout: &pipe.layouts[3],
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: dbuf.as_entire_binding() }],
            });

            Some(GxpPrepared { key, vbuf, ibuf, index_count: gxp.index_count, bg: [bg0, bg1, bg2, bg3], blend: gxp.blend })
        }

        /// The cached pipeline for a prepared draw (only called after `prepare` succeeded).
        fn pipeline(&self, key: u64) -> &GxpPipeline {
            self.pipelines.get(&key).and_then(|p| p.as_ref()).expect("prepared key present")
        }

        /// Build the group2 sampler bind group: for each declared sampler unit, upload the
        /// bound texture and bind it with the matching filter sampler. `None` (fall back) if a
        /// unit has no bound texture or needs a 3D texture (not yet mapped).
        #[allow(clippy::too_many_arguments)]
        fn make_sampler_bg(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            layout: &wgpu::BindGroupLayout,
            samplers: &[(u8, SamplerDim)],
            gxp: &GxpRecompile,
            view_cache: &mut HashMap<(u64, SamplerDim), wgpu::TextureView>,
            sampler_point: &wgpu::Sampler,
            sampler_linear: &wgpu::Sampler,
            force: bool,
        ) -> Option<wgpu::BindGroup> {
            if samplers.is_empty() {
                return Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gxp-samplers-empty"),
                    layout,
                    entries: &[],
                }));
            }
            let debug = std::env::var_os("VITASLOP_GXP_DEBUG").is_some();
            // Upload every needed texture first so the views outlive the bind-group build.
            let mut views: Vec<wgpu::TextureView> = Vec::with_capacity(samplers.len());
            // Cloning a `TextureView` is a refcount bump, so cached views are shared, not copied.
            let mut linears: Vec<bool> = Vec::with_capacity(samplers.len());
            for &(unit, want) in samplers {
                let bound = gxp.textures.iter().find(|t| t.unit == unit);
                // The bound texture must actually supply the dimension the shader declared: a
                // cube sampler needs the six captured faces, a 2D sampler a single image. A
                // mismatch means the container and the guest state disagree, so bind nothing.
                let usable = bound.filter(|gt| match want {
                    SamplerDim::Cube => gt.tex.faces == 6,
                    SamplerDim::Two => gt.tex.faces == 1,
                    SamplerDim::Three => false,
                });
                match usable {
                    Some(gt) => {
                        let cache_key = (gt.tex.key, want);
                        if !view_cache.contains_key(&cache_key) {
                            // Bound the cache: the keys are content fingerprints, so clearing
                            // wholesale only costs a re-upload, never correctness.
                            if view_cache.len() >= TEX_CACHE_CAP {
                                view_cache.clear();
                            }
                            let tex = upload_gxp_texture(device, queue, &gt.tex);
                            let view = tex.create_view(&wgpu::TextureViewDescriptor {
                                dimension: Some(want.view_dimension()),
                                ..Default::default()
                            });
                            view_cache.insert(cache_key, view);
                        }
                        views.push(view_cache[&cache_key].clone());
                        linears.push(gt.tex.filter_linear);
                    }
                    // A volume sampler (not yet mapped), or a unit whose real texture we could
                    // not capture/decode: strict mode falls back; force mode binds a neutral
                    // fallback so geometry still renders (a diagnostic, never the default).
                    None => {
                        if !force {
                            if debug {
                                eprintln!(
                                    "gxp prepare: sampler unit {unit} wants {want:?} but bound units are {:?}",
                                    gxp.textures
                                        .iter()
                                        .map(|t| (t.unit, t.tex.faces))
                                        .collect::<Vec<_>>()
                                );
                            }
                            return None;
                        }
                        views.push(make_fallback_view(device, queue, want.view_dimension()));
                        linears.push(false);
                    }
                }
            }
            let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(samplers.len() * 2);
            for (i, view) in views.iter().enumerate() {
                let samp = if linears[i] { sampler_linear } else { sampler_point };
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
    fn make_gxp_sampler(device: &wgpu::Device, linear: bool) -> wgpu::Sampler {
        let f = if linear { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest };
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gxp-sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: f,
            min_filter: f,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        })
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
        device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("gxp-tex"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: layers },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
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

    /// Repack a guest vertex stream into the tightly-packed `Float32xN` layout the recompiled
    /// pipeline expects. One packed vertex per guest vertex, in order, so the index buffer is
    /// unchanged.
    fn repack_vertices(vertices: &[u8], guest_stride: u32, repack: &[RepackAttr], packed_stride: u32) -> Vec<u8> {
        let gstride = guest_stride.max(1) as usize;
        let nverts = vertices.len() / gstride;
        let mut out = Vec::with_capacity(nverts * packed_stride as usize);
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
        out
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
    fn inject_clip_fixup(wgsl: &str, zfix: bool, yflip: bool, solid: bool) -> Option<String> {
        // Replace the guest's clip z with the SAME depth the fixed-function path writes, so
        // recompiled and fixed-function draws share one comparable depth buffer: the projected
        // view distance through `-1/w`, mapped linearly onto [0,1] over the scene's visible
        // range (see `render::project` for why the guest's own clip z is not a depth here).
        // Keeping xy exact leaves the real shader's projection untouched. w<=0 (behind the eye)
        // is left to wgpu's clip.
        let z = if zfix {
            "  if (c.w > 0.0) { let q = -1.0 / c.w;\n    r.z = clamp((q - gxp_depth.range.x) * gxp_depth.range.y, 0.0, 1.0) * c.w; }\n"
        } else {
            ""
        };
        let y = if yflip { "  r.y = -c.y;\n" } else { "" };
        let helper = format!(
            "struct GxpDepth {{ range: vec4<f32> }};\n\
             @group(3) @binding(0) var<uniform> gxp_depth: GxpDepth;\n\
             fn gxp_clipfix(c: vec4<f32>) -> vec4<f32> {{\n  var r = c;\n{z}{y}  return r;\n}}\n"
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
                None => eprintln!(
                    "gxp build: VITASLOP_GXP_SOLID found no fragment return to replace - \
                     the module shape changed; solid-fill is NOT in effect"
                ),
            }
        }
        Some(format!("{helper}{patched}"))
    }

    /// Link a guest shader pair and build its two pipeline variants + bind-group layouts.
    /// `None` (fall back) on any link error or an unmappable vertex format.
    fn build_gxp_pipeline(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        gxp: &GxpRecompile,
        zfix: bool,
        yflip: bool,
        solid: bool,
    ) -> Option<GxpPipeline> {
        let debug = std::env::var_os("VITASLOP_GXP_DEBUG").is_some();
        let linked = match vitaslop_gxp_shader::link_programs(&gxp.vprog, &gxp.fprog) {
            Ok(l) => l,
            Err(e) => {
                if debug {
                    eprintln!("gxp build: link failed: {e}");
                }
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
                    if debug {
                        eprintln!(
                            "gxp build: no guest attribute for linked @location {} base_lane {} (guest reg_indices {:?})",
                            a.location, a.base_lane,
                            gxp.attributes.iter().map(|g| g.reg_index).collect::<Vec<_>>()
                        );
                    }
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
                match vitaslop_gxp_shader::interp::run(&vrc.shader, &mut regs) {
                    Ok(()) => {
                        let w = regs.o[3];
                        let ndc = if w.abs() > 1e-6 { [regs.o[0] / w, regs.o[1] / w, regs.o[2] / w] } else { [0.0; 3] };
                        eprintln!("gxp interp: o={:?} ndc={:?} viewport(xo,xs,yo,ys,zo,zs)={:?}", &regs.o[0..4], ndc, gxp.viewport);
                    }
                    Err(e) => eprintln!("gxp interp: run failed: {e}"),
                }
            }
        }

        let wgsl = match inject_clip_fixup(&linked.wgsl, zfix, yflip, solid) {
            Some(w) => w,
            None => {
                eprintln!(
                    "gxp build: link failed: no `out.position` assignment to wrap with the clip \
                     fixup - refusing to render without the depth remap"
                );
                return None;
            }
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gxp-linked"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        // group0 vertex uniform, group1 fragment uniform, group2 samplers (empty where unused).
        let uniform_entry = |vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: vis,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let vsa_lanes = linked.vertex_bindings.sa_lane_count;
        let fsa_lanes = linked.fragment_bindings.sa_lane_count;
        let g0_entries: Vec<wgpu::BindGroupLayoutEntry> = if vsa_lanes > 0 { vec![uniform_entry(wgpu::ShaderStages::VERTEX)] } else { vec![] };
        let g1_entries: Vec<wgpu::BindGroupLayoutEntry> = if fsa_lanes > 0 { vec![uniform_entry(wgpu::ShaderStages::FRAGMENT)] } else { vec![] };
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
                eprintln!("gxp build: gxm texture unit {gxm_unit} (coords {}, {dim:?})", b.coords);
            }
            samplers.push((gxm_unit as u8, dim));
        }
        // group3 carries the scene depth range the injected clip fixup remaps through - one
        // vec4 the renderer refills per frame, so the pipeline stays cached by shader identity.
        let layouts = [
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gxp-g0"), entries: &g0_entries }),
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gxp-g1"), entries: &g1_entries }),
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gxp-g2"), entries: &g2_entries }),
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gxp-g3"),
                entries: &[uniform_entry(wgpu::ShaderStages::VERTEX)],
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

        let make = |opaque: bool| {
            let (mut blend, mut depth_write, mut depth_compare) = if opaque {
                (Some(wgpu::BlendState::REPLACE), true, wgpu::CompareFunction::LessEqual)
            } else {
                (
                    Some(wgpu::BlendState {
                        color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                        alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
                    }),
                    false,
                    wgpu::CompareFunction::Always,
                )
            };
            if solid {
                // Diagnostic: REPLACE (ignore alpha) + depth Always, so a magenta triangle shows
                // unconditionally wherever geometry lands.
                blend = Some(wgpu::BlendState::REPLACE);
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
                    targets: &[Some(wgpu::ColorTargetState { format: color_format, blend, write_mask: wgpu::ColorWrites::ALL })],
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

        Some(GxpPipeline { opaque: make(true), blend: make(false), layouts, vsa_lanes, fsa_lanes, samplers, repack, packed_stride })
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
            let make = |opaque: bool| {
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
                            format: color_format,
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

            let opaque = make(true);
            let blend = make(false);

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

            GxmRenderer {
                opaque,
                blend,
                uniform_layout,
                texture_layout,
                sampler_point,
                sampler_linear,
                white_bind,
                views: HashMap::new(),
                tex_binds: HashMap::new(),
                vbo: None,
                ibo: None,
                ubo: None,
                ubo_bind: None,
                vbo_cap: 0,
                ibo_cap: 0,
                ubo_cap: 0,
                uniform_stride,
                color_format,
                ss_scale: 1,
                resolve_pipeline,
                resolve_layout,
                resolve_scale_buf,
                ss_target: None,
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
            // Bound the caches: on pathological churn, clear wholesale and re-upload
            // (keys are content fingerprints, so correctness is unaffected).
            if self.views.len() >= TEX_CACHE_CAP {
                self.views.clear();
                self.tex_binds.clear();
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
            // 1. Walk the scene once: pack vertex/index/uniform bytes into per-frame
            //    arenas and ensure each draw's texture upload + bind group exist.
            let stride = self.uniform_stride as usize;
            let mut vdata: Vec<u8> = Vec::new();
            let mut idata: Vec<u8> = Vec::new();
            let mut udata: Vec<u8> = Vec::new();
            let mut items: Vec<Item> = Vec::with_capacity(scene.draws.len());
            // The live recompiler's per-draw resources + a submission-order plan interleaving
            // recompiled and fixed-function draws (so they share one depth-tested pass).
            let gxp_enabled = self.gxp.enabled;
            let gxp_only = self.gxp.only;
            let color_format = self.color_format;
            let mut gxp_prepared: Vec<GxpPrepared> = Vec::new();
            let mut order: Vec<Enc> = Vec::with_capacity(scene.draws.len());
            for d in &scene.draws {
                // Live GXP path: draw with the guest's real shaders when the pair links. On a
                // link/format failure fall through to the fixed-function packing below (unless
                // isolate mode, which renders only the recompiled draws).
                if gxp_enabled {
                    if let Some(g) = &d.gxp {
                        if let Some(mut prep) = self.gxp.prepare(device, queue, color_format, g, [scene.depth_min, scene.depth_scale]) {
                            if self.gxp.solid {
                                prep.blend = false; // REPLACE + depth-Always variant (see make)
                            }
                            order.push(Enc::Gxp(gxp_prepared.len()));
                            gxp_prepared.push(prep);
                            continue;
                        }
                        if gxp_only {
                            continue;
                        }
                    } else if gxp_only {
                        continue;
                    }
                }
                if d.index_count == 0 || d.vertices.is_empty() {
                    continue;
                }
                let bind = match &d.texture {
                    None => BindKey::White,
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
            if gxp_enabled {
                let with_payload = scene.draws.iter().filter(|d| d.gxp.is_some()).count();
                eprintln!(
                    "gxp: scene has {} draws, {} carry a shader payload, {} recompiled+prepared, {} fixed-function items",
                    scene.draws.len(),
                    with_payload,
                    gxp_prepared.len(),
                    items.len(),
                );
            }

            // 2. Size the arenas and upload. Rebuild the uniform bind group if the uniform
            //    buffer was (re)created.
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

            // 3. One render pass over the whole scene. When supersampling, the scene is drawn
            //    into the offscreen `scale x` target (built here) and a resolve pass below
            //    box-downsamples it into the caller's view; otherwise it is drawn straight in.
            let ss = self.ss_scale > 1;
            if ss {
                self.ensure_ss_target(device, queue, surf_w, surf_h);
            }
            // The pass target (colour + depth) is the offscreen SS buffer when supersampling,
            // else the caller's views. Bound in a narrow scope so the resolve pass can follow.
            {
                let (cv, dv) = match (ss, self.ss_target.as_ref()) {
                    (true, Some(t)) => (&t.color_view, &t.depth_view),
                    _ => (color_view, depth_view),
                };
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gxm-scene"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: cv,
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
                        view: dv,
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
                if order.is_empty() && !ss {
                    return; // clear-only frame; the pass above already cleared the target.
                }
                // Draw in submission order, switching between the fixed-function arenas and the
                // recompiled per-draw resources. The fixed-function handles are unwrapped only
                // inside a Fixed arm, where `items` is non-empty so the arenas were uploaded.
                let ubo_bind = self.ubo_bind.as_ref();
                let vbo = self.vbo.as_ref();
                let ibo = self.ibo.as_ref();
                for e in &order {
                    match e {
                        Enc::Fixed(i) => {
                            let it = &items[*i];
                            let (ubo_bind, vbo, ibo) = (ubo_bind.unwrap(), vbo.unwrap(), ibo.unwrap());
                            pass.set_pipeline(if it.opaque { &self.opaque } else { &self.blend });
                            pass.set_bind_group(0, ubo_bind, &[it.uniform_offset]);
                            pass.set_bind_group(1, self.bind_for(it.bind), &[]);
                            pass.set_vertex_buffer(0, vbo.slice(it.v_off..it.v_off + it.v_len));
                            pass.set_index_buffer(ibo.slice(it.i_off..it.i_off + it.i_len), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..it.index_count, 0, 0..1);
                        }
                        Enc::Gxp(idx) => {
                            let p = &gxp_prepared[*idx];
                            let pipe = self.gxp.pipeline(p.key);
                            pass.set_pipeline(if p.blend { &pipe.blend } else { &pipe.opaque });
                            pass.set_bind_group(0, &p.bg[0], &[]);
                            pass.set_bind_group(1, &p.bg[1], &[]);
                            pass.set_bind_group(2, &p.bg[2], &[]);
                            pass.set_bind_group(3, &p.bg[3], &[]);
                            pass.set_vertex_buffer(0, p.vbuf.slice(..));
                            pass.set_index_buffer(p.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..p.index_count, 0, 0..1);
                        }
                    }
                }
            }

            // 4. Resolve: box-downsample the offscreen SS target into the caller's view.
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
    }
}
