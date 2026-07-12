//! A wgpu (GPU) renderer over the captured GXM stream. Same input and same
//! `Framebuffer` output as the software rasterizer in vitaslop-runtime, so the
//! two are directly comparable (software is the oracle). The shader, pipeline,
//! and per-draw encoding live in `vitaslop-platform` (`gpu::CubeRenderer`), so
//! this native path and the browser WebGPU path draw identically. What stays
//! here is native-only: acquiring an adapter headlessly and reading pixels back.
//!
//! This runs headless (render to a texture, read back). A windowed/live path and
//! the browser WebGPU backend build on the same shared pipeline.

use pollster::block_on;
use vitaslop_platform::gpu::{CubeRenderer, DEPTH_FORMAT};
use vitaslop_runtime::capture::Scene;
use vitaslop_runtime::render::Framebuffer;

const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A GPU renderer bound to a device and the shared cube pipeline. Create once,
/// render many scenes.
pub struct WgpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    cube: CubeRenderer,
    /// The adapter name, for logging which GPU serviced the render.
    pub adapter_name: String,
}

impl WgpuRenderer {
    /// Acquire a GPU and build the pipeline. Returns None if no adapter is
    /// available (e.g. a headless CI box with no GPU), so callers can skip.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .ok()?;
        let adapter_name = adapter.get_info().name;

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vitaslop-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;

        let cube = CubeRenderer::new(&device, OUTPUT_FORMAT);
        Some(WgpuRenderer { device, queue, cube, adapter_name })
    }

    /// Render one captured scene to an RGBA framebuffer on the GPU.
    pub fn render_scene(
        &self,
        scene: &Scene,
        width: u32,
        height: u32,
        clear: [u8; 4],
    ) -> Framebuffer {
        let color_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("color"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OUTPUT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&Default::default());

        let depth_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&Default::default());

        let batches = scene.draw_batches();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.cube
            .encode(&self.device, &mut encoder, &color_view, &depth_view, &batches, clear);

        // Copy the color texture into a readback buffer. width*4 is 256-aligned
        // for 960 (3840 = 15*256), so no per-row padding is needed here.
        let bytes_per_row = width * 4;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bytes_per_row * height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &color_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        self.queue.submit([encoder.finish()]);

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let rgba = slice.get_mapped_range().unwrap().to_vec();
        readback.unmap();

        Framebuffer { width, height, rgba }
    }
}
