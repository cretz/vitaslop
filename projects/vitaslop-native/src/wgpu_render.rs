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
use vitaslop_platform::gpu::{CubeRenderer, GxmRenderer, DEPTH_FORMAT};
use vitaslop_runtime::capture::Scene;
use vitaslop_runtime::render::{Framebuffer, RenderSceneBuilder};

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

/// The general GXM renderer bound to a headless device: the native oracle harness for
/// [`GxmRenderer`]. Renders a captured scene through the same shared pipeline the
/// browser canvas uses, reading pixels back so it is directly comparable to the
/// software rasterizer (`vitaslop_runtime::render::render_scene`) - the correctness
/// check for the GPU path before it runs live in a browser.
pub struct GeneralRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    gxm: GxmRenderer,
    builder: RenderSceneBuilder,
    /// The adapter name, for logging which GPU serviced the render.
    pub adapter_name: String,
    /// Where the last [`GeneralRenderer::render_scene`] went. See [`RenderSplit`].
    last_split: RenderSplit,
}

/// Where one whole `render_scene` went, in milliseconds.
///
/// The point of the breakdown is to answer a question that cannot be settled by
/// looking at the total: when one render path costs several times another, is the
/// difference CPU work building GPU objects per draw, or the GPU actually shading
/// more? Optimising the wrong one is wasted effort, and a total hides which it is.
///
/// `build` decodes the captured scene into the neutral render scene, `encode` is the
/// three CPU phases inside the renderer (see
/// [`EncodePhases`](vitaslop_platform::gpu::EncodePhases)), and `submit` is
/// submit-and-wait - the only figure here that contains GPU execution.
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderSplit {
    pub build_ms: f64,
    pub encode_ms: f64,
    pub submit_ms: f64,
    pub phases: vitaslop_platform::gpu::EncodePhases,
}

impl GeneralRenderer {
    /// Acquire a GPU and build the general pipeline. `None` if no adapter is available.
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
            label: Some("vitaslop-gxm"),
            required_features: wgpu::Features::empty(),
            // Raise the resolution-derived limits (max texture dimension, buffer/binding
            // sizes) to what the adapter really supports: a real title binds textures
            // larger than the 2048 downlevel floor (some titles have a ~2480px atlas).
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        let gxm = GxmRenderer::new(&device, &queue, OUTPUT_FORMAT);
        Some(GeneralRenderer {
            device,
            queue,
            gxm,
            builder: RenderSceneBuilder::new(),
            adapter_name,
            last_split: RenderSplit::default(),
        })
    }

    /// Set the GPU supersample factor (1 = off). Mirrors the software oracle's
    /// [`render_scene_supersampled`](vitaslop_runtime::render::render_scene_supersampled) so
    /// both paths antialias identically. See [`GxmRenderer::set_supersample`].
    pub fn set_supersample(&mut self, scale: u32) {
        self.gxm.set_supersample(scale);
    }

    /// Render one captured scene to an RGBA framebuffer on the GPU via the general path.
    ///
    /// A frame that is really several passes must go through
    /// [`render_frame`](Self::render_frame) instead - see its doc comment for why one
    /// scene is not a frame.
    pub fn render_scene(&mut self, scene: &Scene, width: u32, height: u32, clear: [u8; 4]) -> Framebuffer {
        self.render_frame(std::slice::from_ref(scene), width, height, clear)
    }

    /// Render a whole captured FRAME - every scene the guest submitted between flips, in
    /// order - to an RGBA framebuffer.
    ///
    /// A 3D title's frame is a chain: offscreen passes render the world and its
    /// intermediates, and a final pass composites them onto the display buffer by SAMPLING
    /// them. Rendering only the last scene therefore draws only the composite, over
    /// textures whose guest bytes the GPU (not the guest) was supposed to fill. See
    /// [`GxmRenderer::encode_chain`].
    pub fn render_frame(
        &mut self,
        scenes: &[Scene],
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

        // `VITASLOP_CHAIN_LIMIT=N` renders only the frame's first N scenes, which makes the
        // Nth one the image. A frame is a chain of offscreen passes feeding a composite, and
        // when the composite comes out black the question is WHICH pass is empty - a
        // question the finished frame cannot answer, because every failure mode looks like
        // black. This shows any single pass on its own.
        let limit = std::env::var("VITASLOP_CHAIN_LIMIT").ok().and_then(|s| s.trim().parse::<usize>().ok());
        let scenes = match limit {
            Some(n) if n > 0 && n < scenes.len() => &scenes[..n],
            _ => scenes,
        };
        // `VITASLOP_CHAIN_SKIP=i,j` leaves those passes out of the frame. A pass that draws
        // over a target an earlier pass filled correctly is indistinguishable, in the
        // finished frame, from the earlier pass never having drawn - removing the suspect
        // and seeing the image come back separates the two in one run.
        let skip: Vec<usize> = std::env::var("VITASLOP_CHAIN_SKIP")
            .ok()
            .map(|s| s.split(',').filter_map(|p| p.trim().parse().ok()).collect())
            .unwrap_or_default();
        let t_build = std::time::Instant::now();
        let built: Vec<_> = scenes
            .iter()
            .enumerate()
            .filter(|(i, _)| !skip.contains(i))
            .map(|(_, s)| self.builder.build(s))
            .collect();
        let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;
        let t_encode = std::time::Instant::now();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.gxm.encode_chain(
            &self.device,
            &self.queue,
            &mut encoder,
            &color_view,
            &depth_view,
            &built,
            width,
            height,
            clear,
        );
        let encode_ms = t_encode.elapsed().as_secs_f64() * 1000.0;
        let t_submit = std::time::Instant::now();

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
        self.last_split = RenderSplit {
            build_ms,
            encode_ms,
            // Submit-and-wait: the only part of this that contains GPU execution.
            submit_ms: t_submit.elapsed().as_secs_f64() * 1000.0,
            phases: self.gxm.last_phases(),
        };
        Framebuffer { width, height, rgba }
    }

    /// Where the last [`GeneralRenderer::render_scene`] went. See [`RenderSplit`].
    pub fn last_split(&self) -> RenderSplit {
        self.last_split
    }
}
