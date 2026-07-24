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
        // Mvp: object space through the captured MVP. The software rasterizer depth-tests
        // the RAW post-divide depth c.z/c.w (an unbounded f32) with an +INF clear and a
        // Less test - so only the ORDERING of depths matters, not their range. A real
        // title's captured matrices produce c.z/c.w in an arbitrary, often huge range (the
        // composed MVP is not a normalized [0,1] projection), which a WebGPU depth buffer
        // (limited to [0,1]) would clamp to 1.0, dropping the whole 3D world to black.
        // Map the depth LINEARLY across the scene's visible opaque range (computed on the
        // CPU, see RenderScene::depth_min/scale): linear preserves every Less comparison
        // (identical ordering to the oracle) AND keeps full f32 resolution across the range
        // - enough to separate a vehicle from the ground it rests on, which a saturating
        // nonlinear squash loses when both sit at a large depth. X and Y pass through.
        clip = u.mvp * vec4<f32>(position, 1.0);
        let d = clip.z / clip.w;
        let nz = clamp((d - u.depth_min) * u.depth_scale, 0.0, 1.0);
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
            for d in &scene.draws {
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
                if items.is_empty() && !ss {
                    return; // clear-only frame; the pass above already cleared the target.
                }
                if !items.is_empty() {
                    let ubo_bind = self.ubo_bind.as_ref().unwrap();
                    let vbo = self.vbo.as_ref().unwrap();
                    let ibo = self.ibo.as_ref().unwrap();
                    for it in &items {
                        pass.set_pipeline(if it.opaque { &self.opaque } else { &self.blend });
                        pass.set_bind_group(0, ubo_bind, &[it.uniform_offset]);
                        pass.set_bind_group(1, self.bind_for(it.bind), &[]);
                        pass.set_vertex_buffer(0, vbo.slice(it.v_off..it.v_off + it.v_len));
                        pass.set_index_buffer(ibo.slice(it.i_off..it.i_off + it.i_len), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..it.index_count, 0, 0..1);
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
