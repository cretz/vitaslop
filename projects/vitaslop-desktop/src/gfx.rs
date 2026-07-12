//! The native wgpu surface: acquire a GPU for the winit window and present a
//! captured GXM scene to it through the shared cube pipeline. This is the
//! windowed sibling of `vitaslop-native`'s headless renderer (which renders to a
//! texture and reads pixels back) and of the browser canvas path - all three
//! feed the one `CubeRenderer` in `vitaslop-platform`, so pixels stay identical.

use std::sync::Arc;

use pollster::block_on;
use vitaslop_platform::gpu::{CubeRenderer, DEPTH_FORMAT};
use vitaslop_runtime::capture::Scene;
use winit::window::Window;

/// Background clear color, matching the native render tests and the browser path.
pub const CLEAR: [u8; 4] = [16, 16, 24, 255];

/// GPU state bound to the window's surface, plus the shared cube pipeline. Built
/// once when the window appears, then presents scenes until the window closes.
pub struct Gfx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    cube: CubeRenderer,
    depth: wgpu::TextureView,
    /// The GPU servicing this window, for the startup log.
    pub adapter_name: String,
}

impl Gfx {
    /// Acquire an adapter compatible with `window`'s surface, build the device and
    /// the cube pipeline for the surface's format, and configure at the window
    /// size. Fifo present mode paces presentation to the display's refresh.
    pub fn new(window: Arc<Window>) -> Result<Gfx, String> {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .map_err(|e| format!("create_surface: {e}"))?;

        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))
        .map_err(|_| "no GPU adapter for the window surface".to_string())?;
        let adapter_name = adapter.get_info().name;

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vitaslop-desktop"),
            required_features: wgpu::Features::empty(),
            // Keep the conservative downlevel baseline (so this path matches the
            // headless oracle's capability floor) but raise the resolution-derived
            // limits to what the adapter really supports: a high-DPI desktop window
            // is physically larger than the 2048 downlevel max texture dimension.
            required_limits: wgpu::Limits::downlevel_defaults()
                .using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("request_device: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let cube = CubeRenderer::new(&device, format);
        let depth = make_depth(&device, width, height);
        Ok(Gfx { surface, device, queue, config, cube, depth, adapter_name })
    }

    /// Reconfigure the surface and depth buffer for a new window size.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth = make_depth(&self.device, width, height);
    }

    /// Present one captured scene to the window.
    pub fn present(&mut self, scene: &Scene) {
        let batches = scene.draw_batches();

        // get_current_texture returns an enum in wgpu 30; render on Success or
        // Suboptimal, skip this frame on any transient status (resize, lost).
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.cube
            .encode(&self.device, &mut encoder, &view, &self.depth, &batches, CLEAR);
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }
}

/// The cube pipeline's depth attachment, sized to the surface.
fn make_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&Default::default())
}
