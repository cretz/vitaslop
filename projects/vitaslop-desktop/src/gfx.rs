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
    /// Presents in a row that produced no surface texture - see [`acquire`], which is where
    /// this stops being a transient and becomes a black window nobody is told about.
    acquire_failures: u32,
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
        Ok(Gfx { surface, device, queue, config, cube, depth, acquire_failures: 0, adapter_name })
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

        let Some(frame) = acquire(
            &self.surface,
            &self.device,
            &self.config,
            &mut self.acquire_failures,
            ACQUIRE_FAILURE_LIMIT,
        ) else {
            return;
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

/// Acquire the next surface texture, or say why there is none - and refuse to go on producing
/// frames nobody will ever see.
///
/// >>> "SUCCESS OR SUBOPTIMAL, ELSE `return`" IS A PERMANENTLY BLACK WINDOW.
///
/// It is the right answer for a timeout and wrong for everything else. `Outdated` and `Lost` do
/// NOT clear themselves: a surface in either state answers the same way for every subsequent
/// frame, so skipping means the emulator runs at full cost forever against a window that will
/// never update again, with nothing said. `Validation` is this renderer's own bug and nothing
/// would ever have printed it. The browser half of this had the same defect and the same fix -
/// see `vitaslop_web`'s `PresentOutcome`.
///
/// Returns `None` for a frame that is legitimately skipped, having already reconfigured the
/// surface where that is what recovers it. Panics - which on a CLI is the loud failure - once
/// the surface has refused for `limit` presents in a row, because past that it is not a
/// transient and continuing is the silent failure this exists to remove.
pub(crate) fn acquire(
    surface: &wgpu::Surface<'static>,
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    failures: &mut u32,
    limit: u32,
) -> Option<wgpu::SurfaceTexture> {
    let give_up = |what: &str, n: u32| -> ! {
        panic!(
            "the surface has not produced a texture for {n} presents in a row (last answer:              {what}). Nothing rendered since then reached the screen, so the run stops here              rather than keep paying for frames nobody will see."
        )
    };
    match surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) => {
            *failures = 0;
            Some(t)
        }
        // Acquired, but no longer matching the surface: render it and reconfigure for the next.
        wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
            *failures = 0;
            surface.configure(device, config);
            Some(t)
        }
        // A minimised or hidden window is occluded for as long as the user leaves it that way,
        // so this never escalates - it is not a failure, it is a window nobody is looking at.
        wgpu::CurrentSurfaceTexture::Occluded => None,
        wgpu::CurrentSurfaceTexture::Timeout => {
            *failures += 1;
            if *failures >= limit {
                give_up("Timeout", *failures);
            }
            None
        }
        other @ (wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost) => {
            *failures += 1;
            if *failures >= limit {
                give_up(&format!("{other:?}"), *failures);
            }
            surface.configure(device, config);
            None
        }
        wgpu::CurrentSurfaceTexture::Validation => panic!(
            "the surface refused to hand out a texture with a VALIDATION error, which means it              was configured with something this device will not accept. Nothing can be drawn."
        ),
    }
}

/// How many presents in a row may fail before [`acquire`] gives up. Deliberately generous - a
/// reconfigure takes effect on the next frame - and far below "forever".
pub(crate) const ACQUIRE_FAILURE_LIMIT: u32 = 240;
