//! The last step of a present: the console's 960x544 picture, scaled to the canvas's REAL
//! device pixels with each source pixel kept crisp.
//!
//! # Why the browser's own scaling is not good enough
//! The canvas used to be 960x544 and the browser stretched it to the screen with bilinear
//! filtering. On a phone that is a ~2x non-integer stretch, and every hard-edged pixel the
//! title draws - text above all - comes out soft: a baseball title's warning screen is a 1:1 texel-for-pixel
//! blit of hard-edged glyphs (measured: the rendered frame is crisp and matches the desktop
//! texel for texel), and the user saw it blurred on the phone while a golf title's antialiased font
//! survived the blur. Nearest-neighbour (`image-rendering: pixelated`) is crisp but at a
//! non-integer ratio it drops or doubles whole columns unevenly.
//!
//! # What this does
//! The frame renders into a 960x544 STAGE texture exactly as it rendered into the canvas before
//! (so every probe, shot and destination-colour read sees the same image), and one full-screen
//! pass samples it into the device-sized surface: inside a source pixel the output is that
//! pixel's exact colour, and across the seam between two source pixels it blends over exactly
//! ONE output pixel - the area-correct answer for a pixel boundary that falls part-way through a
//! screen pixel. At any scale that is as sharp as the screen can show the console's pixels, and
//! no column is dropped or doubled.

pub struct Scaler {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    stage: Option<Stage>,
}

pub struct Stage {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    bind: wgpu::BindGroup,
}

const SHADER: &str = r#"
struct U { src: vec2<f32>, dst: vec2<f32>, org: vec2<f32>, pad: vec2<f32> };
@group(0) @binding(0) var img: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> u: U;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  // Source texels per output pixel, per axis.
  let d = u.src / u.dst;
  // This output pixel's centre, in source texel units (relative to the viewport's corner).
  let t = (pos.xy - u.org) * d;
  // The nearest source-pixel seam; within half an output pixel of it the sample slides across
  // the seam linearly (one output pixel of blend), elsewhere it sits on a texel centre.
  let seam = floor(t + 0.5);
  let off = clamp((t - seam) / max(d, vec2<f32>(1e-6)), vec2<f32>(-0.5), vec2<f32>(0.5));
  let uv = (seam + off) / u.src;
  return textureSampleLevel(img, smp, uv, 0.0);
}
"#;

impl Scaler {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Scaler {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("present-scale"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present-scale"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("present-scale"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("present-scale"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("present-scale"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present-scale"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Scaler { pipeline, layout, sampler, uniform, stage: None }
    }

    /// The 960x544 stage the frame renders into, created on first use.
    pub fn stage(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat, w: u32, h: u32) {
        if self.stage.as_ref().is_none_or(|s| s.texture.width() != w || s.texture.height() != h) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("present-stage"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                // Everything the canvas texture was used for, plus being sampled by the scale.
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("present-scale"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    wgpu::BindGroupEntry { binding: 2, resource: self.uniform.as_entire_binding() },
                ],
            });
            self.stage = Some(Stage { texture, view, bind });
        }
    }

    /// The stage, if one has been created.
    pub fn current(&self) -> Option<&Stage> {
        self.stage.as_ref()
    }

    /// Scale the stage onto `dst` (the surface view, `dw`x`dh` device pixels).
    pub fn encode(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, dst: &wgpu::TextureView, dw: u32, dh: u32) {
        self.encode_rect(queue, encoder, dst, (0, 0, dw, dh));
    }

    /// Scale the stage into the `(x, y, w, h)` rectangle of `dst`, clearing the rest of it to
    /// black - the desktop window, whose surface is the whole window and whose picture is
    /// letterboxed inside it (the browser sizes its canvas to the picture instead).
    pub fn encode_rect(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::TextureView,
        (x, y, dw, dh): (u32, u32, u32, u32),
    ) {
        let Some(stage) = self.stage.as_ref() else { return };
        let (sw, sh) = (stage.texture.width() as f32, stage.texture.height() as f32);
        let mut u = [0u8; 32];
        for (i, v) in [sw, sh, dw as f32, dh as f32, x as f32, y as f32, 0.0, 0.0].into_iter().enumerate() {
            u[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        queue.write_buffer(&self.uniform, 0, &u);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("present-scale"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_viewport(x as f32, y as f32, dw.max(1) as f32, dh.max(1) as f32, 0.0, 1.0);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &stage.bind, &[]);
        pass.draw(0..3, 0..1);
    }
}
