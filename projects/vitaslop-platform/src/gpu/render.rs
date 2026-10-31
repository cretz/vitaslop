use super::{DrawBatch, CUBE_SHADER};


/// The depth format the cube pipeline renders with. A host that supplies its
/// own depth attachment must use this format.
pub use super::depth_format;

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

/// A buffer holding `contents`, filled through the QUEUE's pooled staging rather than a
/// `mappedAtCreation` region of its own. `write_buffer` wants a size that is a multiple of
/// `COPY_BUFFER_ALIGNMENT`, so the allocation is rounded up and the tail left as zeros.
fn write_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    let size = ((contents.len() as u64).max(align)).div_ceil(align) * align;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, contents);
    buffer
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
                format: depth_format(),
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
    ///
    /// # Why this goes through the QUEUE and not `create_buffer_init`
    /// `create_buffer_init` is `mappedAtCreation` on the web backend, so every call takes
    /// a renderer-side shared-memory STAGING region as well as a GPU buffer. This runs
    /// three times per batch per FRAME, and when that arena is exhausted Chrome refuses
    /// whichever allocation comes next: CI failed with `createBuffer failed, size (64) is
    /// too large for the implementation when mappedAtCreation == true`, naming the 64-byte
    /// mvp uniform, which is the victim rather than the cause. A desktop adapter's arena is
    /// large enough to hide this indefinitely; SwiftShader's is not, which is why the
    /// software runner is the machine that finds it.
    /// [[vitaslop-mapped-at-creation-exhausts-a-staging-arena]]
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batches: &[DrawBatch],
    ) -> Vec<Prepared> {
        batches
            .iter()
            .map(|b| {
                let mut mvp = Vec::with_capacity(64);
                for v in &b.mvp {
                    mvp.extend_from_slice(&v.to_le_bytes());
                }
                let uniform = write_buffer(
                    device,
                    queue,
                    "mvp",
                    &mvp,
                    wgpu::BufferUsages::UNIFORM,
                );
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.bind_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    }],
                });
                let vertices =
                    write_buffer(device, queue, "vbo", &b.vertices, wgpu::BufferUsages::VERTEX);
                let indices =
                    write_buffer(device, queue, "ibo", &b.indices, wgpu::BufferUsages::INDEX);
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
    /// (cleared to `clear`) with `depth_view` (must be [`depth_format`]),
    /// drawing every batch. The caller owns the target textures and, on the
    /// native path, any subsequent copy-to-buffer readback.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        batches: &[DrawBatch],
        clear: [u8; 4],
    ) {
        let prepared = self.prepare(device, queue, batches);
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
