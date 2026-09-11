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
            required_features: vitaslop_platform::gpu::wanted_features(&adapter),
            required_limits: vitaslop_platform::gpu::device_limits(&adapter),
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
            .encode(&self.device, &self.queue, &mut encoder, &color_view, &depth_view, &batches, clear);

        // >>> THE ROW PITCH IS PADDED TO 256, AND ASSUMING IT NEED NOT BE COST A BLACK RUN.
        //
        // `width * 4` is 256-aligned for the 960-wide panel (3840 = 15*256), which is what the
        // comment here used to rest on. A title is free to declare a display buffer SMALLER
        // than the panel and let the display controller stretch it - MEASURED on a baseball
        // title, which switches to 720x408 the moment a game starts, and 720*4 = 2880 is not a
        // multiple of 256. The copy is then a validation error, the submit fails, and the
        // readback holds nothing: 9,101 errors in one run and every shot from the moment the
        // game loaded came back BLACK, with the guest submitting 540 draws a frame behind it.
        let bytes_per_row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
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
        let rgba = unpad_rows(&slice.get_mapped_range().unwrap(), width, height, bytes_per_row);
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

/// The adapter this renderer runs on: the high-performance one, unless `WGPU_ADAPTER_NAME`
/// names another.
///
/// # >>> THIS EXISTS SO A DEVICE-ONLY DEFECT CAN BE REPRODUCED HERE.
/// `request_adapter` returns ONE adapter, and on a laptop that is the discrete GPU. This machine
/// also has an Intel iGPU which, on Vulkan, exposes **ETC2** - the family the target phone uses
/// and the one the whole GPU texture transcode is written for. Without a way to reach it, every
/// ETC2-only path is unexercised here and its first execution is on the user's phone. That has
/// already cost one shipped build: the transcode's `copyBufferToTexture` extent was wrong, the
/// whole command buffer was invalidated, and every transcoded texture was created and never
/// written.
///
/// `WGPU_ADAPTER_NAME` is wgpu's own convention (a case-insensitive substring of the adapter
/// name), not a vitaslop knob, and it pairs with `WGPU_BACKEND`. **Unset, this is exactly the
/// call it replaces**, so the default path and the determinism oracle are untouched.
///
/// It reports what it took, because "the same run behaves differently" with no line saying which
/// GPU answered is the ambiguity every report in this renderer exists to remove.
fn pick_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    let want = std::env::var("WGPU_ADAPTER_NAME").ok().filter(|s| !s.is_empty());
    if let Some(want) = want {
        let lower = want.to_lowercase();
        let found = block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .find(|a| a.get_info().name.to_lowercase().contains(&lower));
        match found {
            Some(a) => {
                let i = a.get_info();
                eprintln!(
                    "gpu: WGPU_ADAPTER_NAME={want} selected {:?} {:?} {}",
                    i.backend, i.device_type, i.name
                );
                return Some(a);
            }
            // Falling back silently would run the test on the DEFAULT adapter and report a pass
            // for a configuration that was never exercised, which is worse than not running.
            None => {
                eprintln!("gpu: WGPU_ADAPTER_NAME={want} matched no adapter - REFUSING to fall back");
                return None;
            }
        }
    }
    block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .ok()
}

impl GeneralRenderer {
    /// Acquire a GPU and build the general pipeline. `None` if no adapter is available.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pick_adapter(&instance)?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vitaslop-gxm"),
            required_features: vitaslop_platform::gpu::wanted_features(&adapter),
            required_limits: vitaslop_platform::gpu::device_limits(&adapter),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        // A wgpu VALIDATION error is reported out of band: the offending object becomes
        // invalid and everything built from it silently does nothing. Without a handler that
        // is a black frame and no message - indistinguishable from a scene the guest never
        // submitted, which is the exact ambiguity every report in this renderer exists to
        // remove. The BROWSER device has had one since the day its black world was chased for
        // a session; this is the same device on the desktop and it never got one, so a
        // headless run could fail validation in perfect silence.
        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            let msg = e.to_string();
            tracing::error!(target: "vitaslop::gxm", "wgpu uncaptured error: {msg}");
            // The same poison list the browser fills, for the same reason: a refused pipeline
            // bound into a pass takes every other draw in the frame with it. The desktop is the
            // pixel oracle, so it must degrade the same way or the two renderers disagree about
            // what a frame contains. See `vitaslop_platform::gpu::note_device_error`.
            vitaslop_platform::gpu::note_device_error("wgpu", &msg);
        }));
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
        // Scenes completed at their own `sceGxmEndScene` are not rendered again: their target
        // already holds the image (see `Scene::completed_early`).
        let owned: Vec<Scene>;
        let scenes: &[Scene] = if scenes.iter().any(|s| s.completed_early) {
            owned = scenes.iter().filter(|s| !s.completed_early).cloned().collect();
            &owned
        } else {
            scenes
        };
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
        // A new frame for the builder's texture cache - the same boundary the browser marks.
        // Both engines share this builder, so a cache policy that depends on the frame
        // boundary has to be told about it from both, or the two diverge in how much they
        // re-decode and the desktop stops being an oracle for the browser.
        self.builder.begin_frame();
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
            // The headless target IS the guest resolution here: `color_tex` above is created at
            // exactly `width x height`, so the framebuffer extent and the surface extent agree.
            width,
            height,
            clear,
            // The offline path renders straight into this texture, so a draw whose fragment
            // program reads the DESTINATION colour is served here too - without it a capsule
            // of such a draw is dropped and replays to an empty frame.
            Some(&color_tex),
        );
        // `VITASLOP_GXM_DRAW_COVERAGE`: resolve the per-draw occlusion queries onto this same
        // encoder, so the counts describe the frame the readback below is about.
        self.gxm.ts_finish_chain(&mut encoder);
        let encode_ms = t_encode.elapsed().as_secs_f64() * 1000.0;
        let t_submit = std::time::Instant::now();

        // Padded to 256, for the reason the other readback in this file spells out.
        let bytes_per_row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
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
        let rgba = unpad_rows(&slice.get_mapped_range().unwrap(), width, height, bytes_per_row);
        readback.unmap();
        // ...and now that the submit has completed, say what every draw of that frame covered.
        // Inert unless `VITASLOP_GXM_DRAW_COVERAGE` is set.
        self.gxm.coverage_report_blocking(&self.device);
        self.last_split = RenderSplit {
            build_ms,
            encode_ms,
            // Submit-and-wait: the only part of this that contains GPU execution.
            submit_ms: t_submit.elapsed().as_secs_f64() * 1000.0,
            phases: self.gxm.chain_phases(),
        };
        if let Some(dir) = std::env::var_os("VITASLOP_GPU_CHAIN_DIR") {
            self.dump_chain_targets(std::path::Path::new(&dir));
            self.dump_chain_depth_targets(std::path::Path::new(&dir));
        }
        Framebuffer { width, height, rgba }
    }

    /// The rendered pixels of every offscreen target small enough to hand back to the GUEST,
    /// as `(guest colour address, width, height, straight RGBA8)`.
    ///
    /// # Why a render target has to reach guest memory at all
    /// A title does not only SAMPLE its render targets - it also reads texels out of them on
    /// the CPU. MEASURED on a baseball title: its ambient light is a 128x128 target the GPU
    /// paints once a frame and the guest then indexes with `row*512 + col*4` to pull one texel
    /// out. Because nothing ever put the rendered pixels back in guest memory, the read
    /// returned the game's OWN allocator poison - `0xBAADCAFE`, every word of the buffer - and
    /// the three poison bytes (254, 202, 173) went through the title's base-10 log decode into
    /// an ambient of (97.2, 23.7, 10.7). That is a hundred times an ambient, and it is the
    /// whole of that title's washed-out frame.
    ///
    /// # Why SMALL only
    /// The copy is a GPU->CPU readback, and one per frame for every target would be tens of
    /// megabytes. The split is not arbitrary: a target a title reads on the CPU is small BY
    /// CONSTRUCTION - a probe grid, a luminance reduction, an occlusion or exposure result -
    /// because the CPU has to walk it. A full-size colour buffer is consumed by the GPU and
    /// never crosses back. `VITASLOP_GXM_RTT_WRITEBACK` is the cap in TEXELS (default 65536,
    /// i.e. 256x256); `0` disables the writeback entirely and is the arm back.
    /// What the renderer's caches are holding, in the same line the browser panel prints.
    ///
    /// The desktop had no reader for it at all, which is why an unbounded cache could be found
    /// only from a device dump or a browser panel - on the one engine where a run costs a minute
    /// and the OS will tell you the process's working set. A residency question should be
    /// answerable here first.
    pub fn cache_sizes(&self) -> String {
        self.gxm.cache_sizes()
    }

    /// Complete ONE scene now - render it as an offscreen target and return the small
    /// targets' pixels - for the guest's `sceGxmEndScene` (see `VitaState::complete_scene_now`).
    /// The scene must not be taken for the display: alone in a frame it would be.
    pub fn complete_scenes(&mut self, scenes: &[Scene]) -> Vec<(u32, u32, u32, Vec<u8>)> {
        if scenes.is_empty() {
            return Vec::new();
        }
        self.gxm.set_offscreen_only(true);
        let _ = self.render_frame(scenes, 960, 544, [0, 0, 0, 0]);
        self.gxm.set_offscreen_only(false);
        self.rtt_writebacks()
    }

    pub fn rtt_writebacks(&self) -> Vec<(u32, u32, u32, Vec<u8>)> {
        let cap = rtt_writeback_texels();
        if cap == 0 {
            return Vec::new();
        }
        // Row pitch for a texture->buffer copy must be a multiple of 256 bytes, so the
        // readback is padded and the padding stripped per row on the way out.
        const ALIGN: u32 = 256;
        // ONE encoder, ONE submit and ONE poll for every target. A copy-submit-poll per target
        // is a GPU stall per target: this title has seven small ones and paid seven stalls a
        // frame for a job whose whole point is that it is cheap
        // [[vitaslop-a-stall-must-not-buy-a-sentence]].
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gxm-rtt-writeback") });
        let mut staged: Vec<(u32, u32, u32, u32, wgpu::Buffer)> = Vec::new();
        for (addr, tex, w, h) in self.gxm.rtt_targets() {
            let (w, h) = (w.max(1), h.max(1));
            if w * h > cap {
                continue;
            }
            let padded = (w * 4).div_ceil(ALIGN) * ALIGN;
            let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gxm-rtt-writeback"),
                size: (padded * h) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            staged.push((addr, w, h, padded, readback));
        }
        if staged.is_empty() {
            return Vec::new();
        }
        self.queue.submit([enc.finish()]);
        for (_, _, _, _, buf) in &staged {
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let mut out = Vec::with_capacity(staged.len());
        for (addr, w, h, padded, buf) in &staged {
            let Ok(view) = buf.slice(..).get_mapped_range() else { continue };
            out.push((*addr, *w, *h, unpad_rows(&view, *w, *h, *padded)));
            drop(view);
            buf.unmap();
        }
        out
    }

    /// `VITASLOP_GPU_CHAIN_DIR=<dir>`: write every offscreen target of the frame just
    /// rendered to `<dir>/rtt_<addr>_<w>x<h>.png`.
    ///
    /// The GPU counterpart of the software rasterizer's `VITASLOP_SW_CHAIN`. Only the
    /// composite reaches the caller's framebuffer, so a black frame says nothing about
    /// WHICH pass failed - and on a title whose draws are real recompiled shaders the
    /// software chain is not an answer either, because it cannot run them. Written after
    /// submit, so these are the finished images the composite had available to sample.
    fn dump_chain_targets(&self, dir: &std::path::Path) {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("gpu chain dump: mkdir {}: {e}", dir.display());
            return;
        }
        // Row pitch for a texture->buffer copy must be a multiple of 256 bytes, so the
        // readback is padded and the padding stripped per row on the way out.
        const ALIGN: u32 = 256;
        for (addr, tex, w, h) in self.gxm.rtt_targets() {
            let (w, h) = (w.max(1), h.max(1));
            let padded = (w * 4).div_ceil(ALIGN) * ALIGN;
            let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gxm-rtt-readback"),
                size: (padded * h) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            self.queue.submit([enc.finish()]);
            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            if rx.recv().is_err() {
                continue;
            }
            let padded_bytes = slice.get_mapped_range().unwrap().to_vec();
            readback.unmap();
            let mut rgba = Vec::with_capacity((w * h * 4) as usize);
            for row in 0..h as usize {
                let start = row * padded as usize;
                rgba.extend_from_slice(&padded_bytes[start..start + (w * 4) as usize]);
            }
            let path = dir.join(format!("rtt_{addr:08x}_{w}x{h}.png"));
            let fb = Framebuffer { width: w, height: h, rgba };
            if let Err(e) = std::fs::write(&path, fb.to_png()) {
                eprintln!("gpu chain dump: write {}: {e}", path.display());
            }
        }
    }

    /// `VITASLOP_GPU_CHAIN_DIR=<dir>`: alongside the colour targets, report every converted
    /// guest-DEPTH target numerically and write a normalised grayscale view of it.
    ///
    /// The numbers are the point, not the picture. A depth buffer holds view distances in the
    /// guest's own units - hundreds, and often negative - so a PNG of it says almost nothing,
    /// while "min -1174, max -3.8" says immediately whether the conversion produced distances
    /// at all and whether they have the sign the reading shader expects. A pass that samples a
    /// depth buffer and renders black cannot otherwise be told apart from one that samples a
    /// depth buffer full of zeroes.
    fn dump_chain_depth_targets(&self, dir: &std::path::Path) {
        const ALIGN: u32 = 256;
        for (addr, tex, w, h) in self.gxm.rtt_depth_targets() {
            let (w, h) = (w.max(1), h.max(1));
            // Rgba16Float: four halves, eight bytes a texel.
            let padded = (w * 8).div_ceil(ALIGN) * ALIGN;
            let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gxm-rtt-depth-readback"),
                size: (padded * h) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            self.queue.submit([enc.finish()]);
            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            if rx.recv().is_err() {
                continue;
            }
            let bytes = slice.get_mapped_range().unwrap().to_vec();
            readback.unmap();
            let mut vals: Vec<f32> = Vec::with_capacity((w * h) as usize);
            for row in 0..h as usize {
                let start = row * padded as usize;
                for x in 0..w as usize {
                    let o = start + x * 8;
                    vals.push(f16_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]])));
                }
            }
            let finite: Vec<f32> =
                vals.iter().copied().filter(|v| v.is_finite()).collect();
            let (min, max) = finite
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            let mean =
                if finite.is_empty() { 0.0 } else { finite.iter().sum::<f32>() / finite.len() as f32 };
            let nonzero = finite.iter().filter(|v| **v != 0.0).count();
            eprintln!(
                "gpu chain depth {addr:#010x} {w}x{h}: min={min} max={max} mean={mean} \
                 nonzero={nonzero}/{} finite={}",
                vals.len(),
                finite.len()
            );
            // A coarse grid of actual values. The min/max above say the conversion produced
            // distances; this says WHERE they are, which is the question when a pass that
            // reads the depth lights up in some places and not others. Eight columns by six
            // rows is small enough to read in a log and dense enough to show a horizon.
            for gy in 0..6 {
                let y = (h as usize * (2 * gy + 1)) / 12;
                let row: Vec<String> = (0..8)
                    .map(|gx| {
                        let x = (w as usize * (2 * gx + 1)) / 16;
                        format!("{:>9.1}", vals[y * w as usize + x])
                    })
                    .collect();
                eprintln!("gpu chain depth {addr:#010x}   y={y:<4} {}", row.join(" "));
            }
            // A normalised grayscale view, purely so the SHAPE is visible (is that the track,
            // or noise?). The scale is printed above, because the image cannot carry it.
            let span = if max > min { max - min } else { 1.0 };
            let mut rgba = Vec::with_capacity((w * h * 4) as usize);
            for v in &vals {
                let g = if v.is_finite() { (((v - min) / span) * 255.0) as u8 } else { 0 };
                rgba.extend_from_slice(&[g, g, g, 255]);
            }
            let path = dir.join(format!("depth_{addr:08x}_{w}x{h}.png"));
            let fb = Framebuffer { width: w, height: h, rgba };
            if let Err(e) = std::fs::write(&path, fb.to_png()) {
                eprintln!("gpu chain dump: write {}: {e}", path.display());
            }
        }
    }

    /// Where the last [`GeneralRenderer::render_scene`] went. See [`RenderSplit`].
    pub fn last_split(&self) -> RenderSplit {
        self.last_split
    }
}

/// Decode an IEEE binary16 bit pattern to `f32`, including subnormals, infinities and NaN.
///
/// Written out rather than pulled from a crate because it is the ONLY place this crate needs
/// it and a wrong `f16` decode would misreport exactly the values this diagnostic exists to
/// report - a silently wrong number is worse than no number.
/// Strip a readback's row PADDING: `bytes_per_row` is rounded up to
/// `COPY_BYTES_PER_ROW_ALIGNMENT` for the copy, and a `Framebuffer` is tightly packed.
/// A no-op copy when the two already agree, which is every panel-width frame.
pub use vitaslop_runtime::rtt_writeback::apply_rtt_writebacks;
use vitaslop_runtime::rtt_writeback::rtt_writeback_texels;

fn unpad_rows(padded: &[u8], width: u32, height: u32, bytes_per_row: u32) -> Vec<u8> {
    let tight = (width * 4) as usize;
    if bytes_per_row as usize == tight {
        return padded.to_vec();
    }
    let mut rgba = Vec::with_capacity(tight * height as usize);
    for row in 0..height as usize {
        let start = row * bytes_per_row as usize;
        rgba.extend_from_slice(&padded[start..start + tight]);
    }
    rgba
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        // Zero or subnormal: normalise by hand.
        0 if frac == 0 => sign << 31,
        0 => {
            // `frac * 2^-24`, renormalised. With `lz` leading zeros in the u32 the top set
            // bit sits at `31 - lz`, so the value is `2^(7 - lz) * (1 + rest)` and the biased
            // exponent is `134 - lz`; the mantissa shifts left by `lz - 8` and the implicit
            // leading one falls off the top under the mask.
            let lz = frac.leading_zeros();
            (sign << 31) | ((134 - lz) << 23) | ((frac << (lz - 8)) & 0x7f_ffff)
        }
        // Infinity or NaN.
        0x1f => (sign << 31) | 0x7f80_0000 | (frac << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod f16_tests {
    use super::f16_to_f32;

    /// Every finite half, against the reference conversion. A hand-picked list misses exactly
    /// the cases that are hard - the subnormals and the exponent boundary.
    #[test]
    fn every_finite_half_decodes_exactly() {
        for bits in 0u32..=0xffff {
            let h = bits as u16;
            let exp = (h >> 10) & 0x1f;
            if exp == 0x1f {
                continue; // inf/NaN compared separately below
            }
            let got = f16_to_f32(h);
            // Reference: assemble through f32 arithmetic from the fields.
            let sign = if h >> 15 == 1 { -1.0f32 } else { 1.0 };
            let frac = (h & 0x3ff) as f32;
            let want = if exp == 0 {
                sign * frac * 2.0f32.powi(-24)
            } else {
                sign * (1.0 + frac / 1024.0) * 2.0f32.powi(exp as i32 - 15)
            };
            assert_eq!(got, want, "half {h:#06x}");
        }
    }

    #[test]
    fn infinities_and_nan_decode() {
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}

#[cfg(test)]
mod writeback_tests {
    use vitaslop_runtime::rtt_writeback::nothing_written_here;

    /// The gate that keeps the render-target writeback off memory a title composed itself.
    #[test]
    fn a_uniform_fill_is_unwritten_and_an_image_is_not() {
        // Zeros - a freshly mapped page.
        assert!(nothing_written_here(&[0u8; 64]));
        // 0xBAADCAFE - MLB 12's allocator poison, the case that found this.
        let poison: Vec<u8> = [0xFEu8, 0xCA, 0xAD, 0xBA].repeat(16);
        assert!(nothing_written_here(&poison));
        // One texel written into the poison is enough to make it the guest's.
        let mut touched = poison.clone();
        touched[8] = 0x01;
        assert!(!nothing_written_here(&touched));
        // Too short to carry a word answers "not empty" rather than guessing.
        assert!(!nothing_written_here(&[0u8; 3]));
        // A gradient is an image.
        let ramp: Vec<u8> = (0..64u8).collect();
        assert!(!nothing_written_here(&ramp));
    }
}
