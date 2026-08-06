//! The retail-title desktop path: load an extracted Vita game directory, run it LIVE
//! in a native window, and play it with the keyboard + mouse. This is the native twin
//! of the browser's `run_game` - the same decrypt -> link -> transpile -> preemptive
//! scheduler -> general GXM->WebGPU render pipeline, wrapped in a winit window instead
//! of a canvas. The guest is stepped one display frame per 1/60 s of wall time with
//! real input injected through the `World` seam, so menus (touch, via the mouse) and
//! gameplay (buttons, via the keyboard/gamepad) reach the guest exactly as on hardware.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use vitaslop_native::{RunReport, ThreadedScheduler};
use vitaslop_platform::gpu::{GxmRenderer, DEPTH_FORMAT};
use vitaslop_runtime::capture::Scene;
use vitaslop_runtime::ingest::pipeline::decrypt_container;
use vitaslop_runtime::ingest::vfs::MemVfs;
use vitaslop_runtime::link::link;
use vitaslop_runtime::render::RenderSceneBuilder;
use vitaslop_runtime::{CtrlFrame, TouchFrame, VitaEnv, World};
use vitaslop_loader as loader;

use pollster::block_on;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::input::Input;

/// The Vita display resolution the guest draws in (pixel-space 2D coords are in this
/// space); the window opens here and the general renderer projects against it. The
/// front touch panel is twice this in each axis.
const GAME_W: u32 = 960;
const GAME_H: u32 = 544;
const PANEL_SCALE: f32 = 2.0;

/// Background clear color, matching the software oracle and the browser.
const CLEAR: [u8; 4] = [16, 16, 24, 255];

/// Step one guest frame per 1/60 s of wall time (60 Hz), regardless of the monitor's
/// refresh, so the game runs at its intended speed.
const FRAME_DT: Duration = Duration::from_micros(16_666);

/// Scheduler quantum + per-frame round cap. The quantum matches the retail boot probe
/// (a fuel slice large enough that most between-host-call work finishes in one slice);
/// the per-frame round cap is generous - the first frame runs the whole boot.
const QUANTUM_FUEL: u64 = 5_000_000;
const PER_FRAME_ROUNDS: u64 = 60_000_000;

/// The live input the window writes and the guest's world reads: the merged controller
/// frame plus the current mouse-as-touch (front panel). Shared behind a mutex because
/// `World` is `Send` (single-threaded here, so it never contends).
#[derive(Clone, Copy, Default)]
pub struct DesktopInput {
    pub ctrl: CtrlFrame,
    pub touch: Option<TouchFrame>,
}

pub type SharedInput = Arc<Mutex<DesktopInput>>;

/// The world the retail guest polls: a virtual 60 Hz clock, a seeded PRNG, and input
/// drawn from the shared [`DesktopInput`] cell the window updates each frame. An
/// optional [`RecipeWorld`] overlays a scripted TAS recipe on top of live input, so a
/// recorded playthrough replays in the window while the user watches (and can still
/// nudge with the keyboard/mouse) - the native twin of the browser's `BrowserWorld`.
struct DesktopWorld {
    input: SharedInput,
    /// A scripted recipe driving input as a function of frame, if `--recipe` was given.
    /// Frame-keyed, so it replays identically on desktop, browser, and headless.
    recipe: Option<vitaslop_runtime::RecipeWorld>,
    monotonic_us: u64,
    wall_us: u64,
    rng: u64,
}

impl DesktopWorld {
    fn new(input: SharedInput) -> Self {
        DesktopWorld {
            input,
            recipe: None,
            monotonic_us: 0,
            wall_us: 1_500_000_000_000_000,
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// With a scripted recipe overlaid on the live input.
    fn with_recipe(input: SharedInput, recipe: vitaslop_runtime::RecipeWorld) -> Self {
        let mut w = DesktopWorld::new(input);
        w.recipe = Some(recipe);
        w
    }
}

impl World for DesktopWorld {
    fn monotonic_us(&mut self) -> u64 {
        self.monotonic_us
    }
    fn wall_us(&mut self) -> u64 {
        self.wall_us
    }
    fn poll_ctrl(&mut self, port: u32) -> CtrlFrame {
        let live = self.input.lock().unwrap().ctrl;
        let Some(r) = self.recipe.as_mut() else { return live };
        // Recipe drives; live input is additive so the watcher can nudge. Buttons OR
        // together; a stick the recipe leaves centered falls back to the live stick.
        let scripted = r.poll_ctrl(port);
        CtrlFrame {
            buttons: scripted.buttons | live.buttons,
            lx: if scripted.lx != 128 { scripted.lx } else { live.lx },
            ly: if scripted.ly != 128 { scripted.ly } else { live.ly },
            rx: if scripted.rx != 128 { scripted.rx } else { live.rx },
            ry: if scripted.ry != 128 { scripted.ry } else { live.ry },
        }
    }
    fn poll_touch(&mut self, port: u32) -> TouchFrame {
        if port != 0 {
            return TouchFrame::default();
        }
        let live = self.input.lock().unwrap().touch.unwrap_or_default();
        let Some(r) = self.recipe.as_mut() else { return live };
        // Prefer the recipe's scripted finger; fall back to the live mouse-as-touch.
        let scripted = r.poll_touch(port);
        if scripted.count > 0 {
            scripted
        } else {
            live
        }
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        // Prefer the recipe's deterministic PRNG when replaying, so a scripted run is
        // reproducible; otherwise SplitMix64, matching the runtime worlds.
        if let Some(r) = self.recipe.as_mut() {
            r.fill_random(buf);
            return;
        }
        for chunk in buf.chunks_mut(8) {
            self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            for (i, b) in chunk.iter_mut().enumerate() {
                *b = (z >> (i * 8)) as u8;
            }
        }
    }
    fn set_frame(&mut self, frame: u64) {
        if let Some(r) = self.recipe.as_mut() {
            r.set_frame(frame);
        }
        self.monotonic_us = frame.wrapping_mul(16_666);
        self.wall_us = 1_500_000_000_000_000u64.wrapping_add(self.monotonic_us);
    }
}

/// Recursively collect every file under `root` as `(forward-slash relative path, bytes)`
/// - the shape `decrypt_container` expects for the extracted app directory.
fn read_dir_files(root: &Path) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("read {}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
            let path = entry.path();
            let ty = entry.file_type().map_err(|e| format!("file type: {e}"))?;
            if ty.is_dir() {
                stack.push(path);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .map_err(|e| format!("strip prefix: {e}"))?
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
                out.push((rel, bytes));
            }
        }
    }
    Ok(out)
}

/// The retail guest, stepped live on the preemptive scheduler. Owns the scheduler and
/// tracks the newest presented frame and whether the run has ended.
pub struct RetailGuest {
    sched: ThreadedScheduler<VitaEnv>,
    /// Every scene of the newest presented frame, in submission order.
    scenes: Vec<Scene>,
    finished: bool,
    err: Option<String>,
    /// Decrypt + link + transpile + instantiate time, measured once at construction.
    pub build_ms: f64,
}

impl RetailGuest {
    /// Load, decrypt, link, transpile, and instantiate the title in `dir` for live
    /// execution over the shared `input` cell. An optional scripted `recipe` (a
    /// frame-keyed TAS text) is overlaid on the live input so a recorded playthrough
    /// replays in the window.
    pub fn new(dir: &Path, input: SharedInput, recipe: Option<&str>) -> Result<RetailGuest, String> {
        let t0 = Instant::now();
        let files = read_dir_files(dir)?;
        let mut vfs = MemVfs::new();
        for (path, bytes) in files {
            vfs.insert(path, bytes);
        }
        let game = decrypt_container(&vfs).map_err(|e| format!("decrypt: {e:?}"))?;
        let modules = game
            .modules
            .iter()
            .map(|m| loader::load(&m.elf))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("load module: {e:?}"))?;
        let linked = link(modules).map_err(|e| format!("link: {e:?}"))?;

        let world: Box<dyn World + Send> = match recipe {
            Some(text) => {
                let r = vitaslop_runtime::RecipeWorld::parse(text)
                    .map_err(|e| format!("recipe parse: {e}"))?;
                Box::new(DesktopWorld::with_recipe(input, r))
            }
            None => Box::new(DesktopWorld::new(input)),
        };
        let mut env = VitaEnv::new(linked.imports.clone(), linked.base, linked.mem_bytes, world);
        env.state.set_alloc_base(linked.alloc_base);
        env.state.set_process_param(linked.process_param);
        env.state.set_modules(linked.loaded_modules.clone());
        env.state.set_tls_template(linked.tls_template);
        env.state.set_preemptive(true);
        // Move (not clone) the decrypted assets into the guest filesystem - for a
        // large 3D title this is hundreds of megabytes.
        for (path, bytes) in game.files.into_files() {
            env.state.add_file(&path, bytes);
        }

        let (sched, _stubs) = ThreadedScheduler::from_linked(&linked, env, QUANTUM_FUEL)
            .map_err(|e| format!("scheduler: {e:?}"))?;
        let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(RetailGuest { sched, scenes: Vec::new(), finished: false, err: None, build_ms })
    }

    /// Step the guest one display frame. Keeps the newest captured scene and drops the
    /// rest (render-to-texture intermediates); marks the run finished on end/trap.
    pub fn advance(&mut self) {
        if self.finished {
            return;
        }
        let target = self.sched.frames() + 1;
        let report = self.sched.run_frames(target, PER_FRAME_ROUNDS);
        {
            let mut host = self.sched.host();
            let cap = &mut host.state.capture;
            // Diagnostic (VITASLOP_DUMP_SCENES): a frame can contain several GXM scenes
            // (render-to-texture passes plus the display pass). Only the last is kept, so
            // knowing how many there were - and how big each is - is the difference between
            // "the geometry is missing" and "the geometry is in a pass we discard".
            if std::env::var_os("VITASLOP_DUMP_SCENES").is_some() && !cap.scenes.is_empty() {
                let shape: Vec<String> = cap
                    .scenes
                    .iter()
                    .map(|s| match &s.color {
                        Some(c) => format!("{}draws@{}x{}", s.draws.len(), c.width, c.height),
                        None => format!("{}draws@no-surface", s.draws.len()),
                    })
                    .collect();
                eprintln!("SCENES frame={}: [{}]", self.sched.frames(), shape.join(", "));
            }
            // Keep the WHOLE frame, not just its last scene. A 3D title renders its world
            // (and shadow maps, reflections, post chains) into offscreen surfaces and then
            // composites them; keeping only the last scene keeps only the composite, and
            // the world never gets drawn at all - a retail racer's race was a live HUD over
            // black for exactly this reason. An empty list means the guest submitted no
            // scene this frame, in which case the previous frame's stays on screen.
            if !cap.scenes.is_empty() {
                self.scenes = std::mem::take(&mut cap.scenes);
            }
            cap.scenes.clear();
            cap.trace.clear();
            cap.trace_thid.clear();
            cap.presents.clear();
        }
        match report {
            RunReport::FramesReached(_) => {}
            RunReport::Error(e) => {
                self.finished = true;
                self.err = Some(e);
            }
            _ => self.finished = true,
        }
    }

    /// The guest's OWN diagnostic output: everything it has written to fd 1/2 this run.
    ///
    /// Retail titles ship a lot of their own logging, and it is the developer's account of
    /// what the game thinks is happening - which beats reverse-engineering the answer, and
    /// has already identified a hang on another title in one line. The engine has been
    /// capturing this all along with nothing in the headless path to read it.
    /// The guest's virtual clock, in microseconds.
    ///
    /// Worth reporting next to the frame count: a title that paces its simulation off the
    /// wall clock but caps how far it will step per frame goes WRONG QUIETLY when this
    /// runs fast - the race timer counts up several times too quickly while the car
    /// crawls, and neither symptom names the clock. One frame is 1/60 s, so the ratio of
    /// this to `frames / 60` should be 1.
    pub fn clock_us(&mut self) -> u64 {
        self.sched.host().state.now_us()
    }

    pub fn guest_stdout(&mut self) -> Vec<u8> {
        self.sched.host().state.capture.stdout.clone()
    }

    /// The newest presented frame's scenes, in submission order; empty until the guest
    /// has flipped once.
    pub fn current(&self) -> &[Scene] {
        &self.scenes
    }
    pub fn finished(&self) -> bool {
        self.finished
    }
    pub fn frames(&self) -> u64 {
        self.sched.frames()
    }
    pub fn error(&self) -> Option<&str> {
        self.err.as_deref()
    }

    /// Guest memory at `addr`, for `VITASLOP_PEEK`.
    pub fn peek(&self, addr: u32, len: usize) -> Vec<u8> {
        self.sched.read_guest(addr, len)
    }
}

/// `VITASLOP_PEEK=<hex addr>:<len>[,<hex addr>:<len>]`: hex-dump guest memory at the END of a
/// headless run, as words with their addresses.
///
/// The alternative is a store watchpoint, and the two answer different questions. A watchpoint
/// says WHO wrote a word; this says WHAT IS THERE, which is what you need when the suspicion is
/// about the neighbours - "is this a table of records, and does the one next door hold the value
/// this draw should have had". A watchpoint cannot show you a record nobody wrote.
fn peek_regions(guest: &RetailGuest) {
    let Ok(spec) = std::env::var("VITASLOP_PEEK") else { return };
    for part in spec.split(',') {
        let Some((a, n)) = part.trim().split_once(':') else { continue };
        let (Ok(addr), Ok(len)) = (
            u32::from_str_radix(a.trim().trim_start_matches("0x"), 16),
            n.trim().parse::<usize>(),
        ) else {
            println!("peek: {part:?} is not <hex addr>:<len>");
            continue;
        };
        let bytes = guest.peek(addr, len);
        for (i, row) in bytes.chunks(32).enumerate() {
            let words: Vec<String> = row
                .chunks(4)
                .map(|c| {
                    let mut w = [0u8; 4];
                    w[..c.len()].copy_from_slice(c);
                    format!("{:08x}", u32::from_le_bytes(w))
                })
                .collect();
            println!("peek {:#010x}: {}", addr + i as u32 * 32, words.join(" "));
        }
    }
}

/// The window's GPU surface + the general GXM renderer. Presents a captured scene each
/// frame through the same `GxmRenderer` the browser canvas uses (so pixels match). The
/// guest draws in 960x544 game space; the renderer projects against that and fills the
/// (possibly larger) window, stretching to fit.
struct RetailGfx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    gxm: GxmRenderer,
    builder: RenderSceneBuilder,
    depth: wgpu::TextureView,
    render_format: wgpu::TextureFormat,
    adapter_name: String,
}

impl RetailGfx {
    fn new(window: Arc<Window>) -> Result<RetailGfx, String> {
        let size = window.inner_size();
        let (w, h) = (size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).map_err(|e| format!("create_surface: {e}"))?;
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))
        .map_err(|_| "no GPU adapter for the window surface".to_string())?;
        let adapter_name = adapter.get_info().name;

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vitaslop-desktop-gxm"),
            required_features: wgpu::Features::empty(),
            // Raise resolution-derived limits to the adapter's: a real title binds
            // textures past the 2048 downlevel floor (some titles have a ~2480px atlas).
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("request_device: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        // The GXM color decode already yields final display-ready (sRGB-encoded) byte
        // values - the software oracle just writes them into an `Rgba8Unorm` PNG. If the
        // surface's preferred format is an sRGB variant (native Vulkan/DX usually prefer
        // `Bgra8UnormSrgb`), writing those straight values applies a SECOND sRGB encode
        // and washes the image out. Render through a non-sRGB view of the same surface so
        // the bytes land verbatim, matching the oracle. On WebGPU the preferred canvas
        // format is already non-sRGB, so this is a no-op there.
        let render_format = format.remove_srgb_suffix();
        let view_formats = if render_format == format { vec![] } else { vec![render_format] };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats,
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let mut gxm = GxmRenderer::new(&device, &queue, render_format);
        // Antialias the native path (which has the GPU headroom): 2x supersample resolves the
        // sub-pixel-triangle / coincident-panel speckle a distant 3D vehicle shows, matching the
        // software review shots. `VITASLOP_SSAA` overrides (1 disables). See GxmRenderer::set_supersample.
        let ssaa = std::env::var("VITASLOP_SSAA").ok().and_then(|s| s.parse::<u32>().ok()).filter(|&n| n >= 1).unwrap_or(2);
        gxm.set_supersample(ssaa);
        let depth = make_depth(&device, w, h);
        Ok(RetailGfx { surface, device, queue, config, gxm, builder: RenderSceneBuilder::new(), depth, render_format, adapter_name })
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth = make_depth(&self.device, w, h);
    }

    fn present(&mut self, scenes: &[Scene]) {
        let built: Vec<_> = scenes.iter().map(|s| self.builder.build(s)).collect();
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.render_format),
            ..Default::default()
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Project against the 960x544 GAME resolution (not the window size), so pixel-
        // space 2D coords map correctly; the window-sized framebuffer stretches the
        // resolution-independent clip output to fill the window.
        self.gxm.encode_chain(&self.device, &self.queue, &mut encoder, &view, &self.depth, &built, GAME_W, GAME_H, CLEAR);
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }
}

fn make_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&Default::default())
}

/// Headless self-check of the retail path (NO window): load `dir`, optionally drive a
/// touch script through the shared input cell, and render the final scene via the
/// native GPU renderer (the same `GxmRenderer`) to `<shot_dir>/desktop.png`. This
/// exercises the whole load -> decrypt -> link -> transpile -> run -> input -> render
/// chain minus the winit window glue, so the desktop path is verifiable without a
/// display.
///
/// Env knobs (so one entry point serves any title without recompiling):
/// - `VITASLOP_HEADLESS_FRAMES` - display flips to run to (default 180). A title that
///   boots through intro/attract screens before its first interactive frame needs a
///   higher target.
/// - `VITASLOP_HEADLESS_RECIPE` - a frame-keyed TAS recipe to drive input with. When
///   set, the recipe supplies the touches/buttons; when unset, the built-in Tutorial
///   tap script runs (the historical title navigation), unless
///   `VITASLOP_HEADLESS_NO_TAPS` is set, which runs input-free to the title screen.
/// - `VITASLOP_HEADLESS_TIMING` - report per-frame GUEST cost (the emulated CPU work
///   behind one display flip) and the GPU cost of rendering the final captured scene.
/// - `VITASLOP_HEADLESS_SHOT_EVERY` - also write `<shot_dir>/fNNNNNN.png` every N display
///   flips, so one run shows the whole boot SEQUENCE rather than only its last frame.
///
/// NOTE what the timing does and does not measure. The guest advances every frame but the
/// scene is RENDERED ONCE, at the end - so a wall-clock total over a headless run is CPU
/// only, and dividing it by the frame count says nothing about the render. The two costs
/// are reported separately for that reason. Neither is a frame rate on its own: a title
/// sitting on a menu waiting for input costs almost nothing per frame, so a number
/// measured there is not a gameplay number.
/// Print the guest-CPU frame-cost distribution and the one-off render cost.
///
/// Reported as PERCENTILES over the whole run and separately over the last 300 frames,
/// because a headless run is two different workloads glued together: a boot sequence that
/// does heavy one-off work, then whatever screen it settles on. The median of the tail is
/// the closest thing here to "what a frame costs on this screen", and it is only a frame
/// RATE if the title is actually doing work on that screen rather than idling for input.
fn report_frame_timing(
    frame_ms: &[f64],
    cold_render_ms: f64,
    warm_render_ms: &[f64],
    split: vitaslop_native::RenderSplit,
) {
    if frame_ms.is_empty() {
        println!("timing: no frames advanced");
        return;
    }
    let pct = |v: &[f64], p: f64| -> f64 {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[(((s.len() - 1) as f64) * p).round() as usize]
    };
    let total: f64 = frame_ms.iter().sum();
    let tail = &frame_ms[frame_ms.len().saturating_sub(300)..];
    println!(
        "timing: guest CPU over {} frames - total {:.0} ms, p50 {:.2} ms, p95 {:.2} ms, max {:.2} ms",
        frame_ms.len(),
        total,
        pct(frame_ms, 0.50),
        pct(frame_ms, 0.95),
        pct(frame_ms, 1.0)
    );
    println!(
        "timing: guest CPU over the last {} frames - p50 {:.2} ms ({:.0} fps if CPU-bound), p95 {:.2} ms",
        tail.len(),
        pct(tail, 0.50),
        1000.0 / pct(tail, 0.50).max(1e-6),
        pct(tail, 0.95)
    );
    println!(
        "timing: GPU render of the final scene - cold {cold_render_ms:.1} ms (builds pipelines, \
         compiles shaders, uploads textures)"
    );
    if !warm_render_ms.is_empty() {
        let p50 = pct(warm_render_ms, 0.50);
        println!(
            "timing: GPU render of the final scene - warm p50 {:.2} ms ({:.0} fps if GPU-bound), \
             p95 {:.2} ms over {} samples",
            p50,
            1000.0 / p50.max(1e-6),
            pct(warm_render_ms, 0.95),
            warm_render_ms.len()
        );
    }
    // WHERE the warm render went. A total says the recompiler path is slower than the
    // fixed-function one; only the split says whether that is per-draw CPU work building GPU
    // objects or the GPU genuinely shading more, and those call for opposite fixes.
    let p = split.phases;
    println!(
        "timing: warm render split - build {:.2} ms (scene decode, CPU), encode {:.2} ms (CPU), \
         submit+wait {:.2} ms (contains the GPU)",
        split.build_ms, split.encode_ms, split.submit_ms
    );
    println!(
        "timing: encode phases - prepare {:.2} ms, upload {:.2} ms, pass {:.2} ms over \
         {} recompiled + {} fixed-function draws",
        p.prepare_ms, p.upload_ms, p.pass_ms, p.gxp_draws, p.fixed_draws
    );
    // The honest caveat, printed rather than left to the reader: a guest that is waiting for
    // input does no work, so a low per-frame CPU cost on a menu is not a gameplay figure.
    println!(
        "timing: NOTE the CPU figures are whatever the title was doing on this screen - if it \
         was idling on a menu they are not a gameplay frame rate."
    );
}

/// How many extra renders of the captured scene to time for the warm figure. Enough to get a
/// stable median past the first-render pipeline/upload cost, cheap enough to always run.
const WARM_RENDER_SAMPLES: usize = 60;

pub fn headless_check(dir: PathBuf, shot_dir: PathBuf) -> Result<(), String> {
    let input: SharedInput = Arc::new(Mutex::new(DesktopInput::default()));
    let recipe = std::env::var("VITASLOP_HEADLESS_RECIPE")
        .ok()
        .map(|p| std::fs::read_to_string(&p).map_err(|e| format!("read recipe {p}: {e}")))
        .transpose()?;
    let mut guest = RetailGuest::new(&dir, input.clone(), recipe.as_deref())?;
    let target: u64 = std::env::var("VITASLOP_HEADLESS_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(180);
    println!("headless: loaded (build {:.0} ms), running to frame {target}...", guest.build_ms);

    // The built-in Tutorial tap script (title menu navigation): used only when no
    // recipe is given and taps are not disabled. (active_from, active_until, x, y) - a
    // sticky tap held across a few frames then released.
    let use_builtin_taps = recipe.is_none() && std::env::var_os("VITASLOP_HEADLESS_NO_TAPS").is_none();
    let taps = [
        (12u64, 19u64, 450u16, 674u16),
        (30, 37, 230, 230),
        (80, 87, 930, 674),
        (112, 121, 1620, 376),
        (128, 137, 630, 870),
    ];
    // Per-frame guest cost, in milliseconds, when timing is on. Kept per frame rather than
    // as a running total so the report can separate the typical frame from the worst one - a
    // mean over a run that includes boot (or an idle menu) describes neither.
    let timing = std::env::var_os("VITASLOP_HEADLESS_TIMING").is_some();
    // Periodic GPU shots (`VITASLOP_HEADLESS_SHOT_EVERY=N`). Without this the headless path
    // renders exactly one frame - the last - which answers "what is on screen at frame N"
    // and nothing else. A title being brought up moves through a SEQUENCE (legal notice,
    // attract, title, menu), and finding the frame a given screen lives on by bisecting on
    // one final shot per run costs a whole boot per guess. The renderer is built once and
    // reused, so the extra cost is one render per sampled frame.
    //
    // Note this renders through `GeneralRenderer`, the same GPU path as the final shot, so
    // a sampled frame is directly comparable to it - and to the browser, which shares the
    // renderer. The software rasterizer is NOT used here and could not show a title whose
    // draws are fragment-shader passes.
    let shot_every: u64 = std::env::var("VITASLOP_HEADLESS_SHOT_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut periodic: Option<vitaslop_native::GeneralRenderer> = None;
    if shot_every > 0 {
        std::fs::create_dir_all(&shot_dir).map_err(|e| format!("mkdir: {e}"))?;
        periodic = Some(vitaslop_native::GeneralRenderer::new().ok_or("no GPU adapter")?);
    }
    let mut frame_ms: Vec<f64> = Vec::new();
    while guest.frames() < target && !guest.finished() {
        let f = guest.frames();
        if let (Some(r), true) = (periodic.as_mut(), shot_every > 0 && f % shot_every == 0) {
            let scenes = guest.current();
            if !scenes.is_empty() {
                let fb = r.render_frame(scenes, GAME_W, GAME_H, CLEAR);
                let path = shot_dir.join(format!("f{f:06}.png"));
                std::fs::write(&path, fb.to_png()).map_err(|e| format!("write png: {e}"))?;
            }
            // The guest clock beside the frame, so its LOCAL rate is visible. A run-total
            // ratio of 1.00x hides a stretch that ran five times fast against a stretch
            // that stalled, and a title paced off the clock behaves very differently in
            // the two.
            println!("shot f{f:06}: guest clock {:.3}s", guest.clock_us() as f64 / 1e6);
        }
        if use_builtin_taps {
            let touch = taps
                .iter()
                .find(|(a, b, _, _)| f >= *a && f < *b)
                .map(|(_, _, x, y)| TouchFrame::single(*x, *y));
            input.lock().unwrap().touch = touch;
        }
        if timing {
            let t = std::time::Instant::now();
            guest.advance();
            frame_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        } else {
            guest.advance();
        }
    }
    // Block-visit histogram (`VITASLOP_BLOCK_HIST=<top>`, with `VITASLOP_TRACE_BLOCKS=lo-hi`
    // emitting the hooks). It was only reachable from the boot-probe test, which cannot
    // replay a recipe - so the one case it is built for, "the title reached screen N and
    // stopped", could not be measured at all. Restrict the hook range to the suspect
    // functions and the tail of the recorded sequence IS the stuck loop.
    if let Ok(top) = std::env::var("VITASLOP_BLOCK_HIST") {
        vitaslop_native::dump_block_hist(top.trim().parse().unwrap_or(40));
    }
    // Sampler-unit bindings dropped at capture time, ranked by cause. Unconditional: a
    // dropped unit is why a recompiled shader falls back, and the per-cause reports are
    // deduped so only this says which cause is worth fixing first.
    vitaslop_runtime::host::report_texture_drops();
    // VITASLOP_DUMP_STDOUT=<path>: write everything the GUEST logged to fd 1/2 this run.
    // Written before the error check, because a run that ended in a trap or a hang is exactly
    // the one whose log matters most.
    if let Ok(path) = std::env::var("VITASLOP_DUMP_STDOUT") {
        let bytes = guest.guest_stdout();
        let len = bytes.len();
        std::fs::write(&path, bytes).map_err(|e| format!("write guest stdout: {e}"))?;
        println!("headless: wrote {len} bytes of GUEST log to {path}");
    }
    if let Some(e) = guest.error() {
        return Err(format!("guest error at frame {}: {e}", guest.frames()));
    }
    let scenes = guest.current().to_vec();
    if scenes.is_empty() {
        return Err("no scene captured".into());
    }

    let mut renderer = vitaslop_native::GeneralRenderer::new().ok_or("no GPU adapter")?;
    let render_t = std::time::Instant::now();
    let fb = renderer.render_frame(&scenes, GAME_W, GAME_H, CLEAR);
    let cold_render_ms = render_t.elapsed().as_secs_f64() * 1000.0;
    if timing {
        // Re-render the SAME scene to separate one-off cost from per-frame cost. The first
        // render builds every pipeline, compiles every recompiled shader and uploads every
        // texture; a steady frame does none of that. Reporting only the cold number would
        // overstate the per-frame cost by orders of magnitude, and reporting only the warm
        // one would hide a startup hitch users would actually feel.
        let mut warm: Vec<f64> = Vec::new();
        for _ in 0..WARM_RENDER_SAMPLES {
            let t = std::time::Instant::now();
            let _ = renderer.render_frame(&scenes, GAME_W, GAME_H, CLEAR);
            warm.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        report_frame_timing(&frame_ms, cold_render_ms, &warm, renderer.last_split());
    }
    std::fs::create_dir_all(&shot_dir).map_err(|e| format!("mkdir: {e}"))?;
    let path = shot_dir.join("desktop.png");
    std::fs::write(&path, fb.to_png()).map_err(|e| format!("write png: {e}"))?;
    // Diagnostic (VITASLOP_SOFTWARE=1): also render the frame's FINAL scene through the
    // software rasterizer and write it beside the GPU shot. The two paths share the scene
    // but nothing else, so a defect present in both is upstream of rendering (geometry,
    // capture, uniforms) rather than a shading bug - and the software path is the one that
    // reports per-draw `DSTAT` coverage under VITASLOP_DRAW_STATS. It renders one scene, so
    // on a title that composites offscreen passes it shows the composite only.
    if std::env::var_os("VITASLOP_SOFTWARE").is_some() {
        let sw = vitaslop_runtime::render::render_scene(scenes.last().unwrap(), GAME_W, GAME_H, CLEAR);
        let sw_path = shot_dir.join("software.png");
        std::fs::write(&sw_path, sw.to_png()).map_err(|e| format!("write png: {e}"))?;
        println!("headless: wrote {}", sw_path.display());
    }
    peek_regions(&guest);
    let frames = guest.frames();
    let clock_s = guest.clock_us() as f64 / 1e6;
    println!(
        "headless: reached frame {frames}, wrote {} (guest clock {clock_s:.1}s = {:.2}x the {:.1}s those frames are worth)",
        path.display(),
        clock_s / (frames.max(1) as f64 / 60.0),
        frames as f64 / 60.0,
    );
    Ok(())
}

/// Run the retail title in `dir` in a live window until the window closes or the guest
/// exits. Blocks until the event loop ends. With `recipe` set, a scripted playthrough
/// replays in the window (live keyboard/mouse still nudges it).
pub fn run(dir: PathBuf, recipe: Option<String>) -> Result<(), String> {
    let input: SharedInput = Arc::new(Mutex::new(DesktopInput::default()));
    let guest = RetailGuest::new(&dir, input.clone(), recipe.as_deref())?;
    println!("loaded {} (decrypt + link + transpile {:.0} ms)", dir.display(), guest.build_ms);
    if recipe.is_some() {
        println!("replaying scripted recipe (keyboard/mouse still nudge live)");
    }
    println!("controls:");
    println!("  menus  : click the window (the front-end is touch driven)");
    println!("  d-pad  : arrow keys        faces: Z cross / X circle / A square / S triangle");
    println!("  L / R  : Q / E             Start: Enter   Select: Shift");
    println!("  a gamepad also works; Space pauses, Esc quits");

    let event_loop = EventLoop::new().map_err(|e| format!("create event loop: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = RetailApp {
        guest,
        input_shared: input,
        input: Input::new(),
        window: None,
        gfx: None,
        paused: false,
        cursor: (0.0, 0.0),
        mouse_down: false,
        acc: Duration::ZERO,
        last_tick: Instant::now(),
        fps_since: Instant::now(),
        fps_frames: 0,
        guest_frames_since: 0,
        fps: 0.0,
        guest_fps: 0.0,
        reported_exit: false,
    };
    event_loop.run_app(&mut app).map_err(|e| format!("run event loop: {e}"))?;
    Ok(())
}

/// The retail window app: the live guest, the merged input, mouse-as-touch state, the
/// window + its general-renderer surface, and 60 Hz pacing + FPS bookkeeping.
struct RetailApp {
    guest: RetailGuest,
    input_shared: SharedInput,
    input: Input,
    window: Option<Arc<Window>>,
    gfx: Option<RetailGfx>,
    paused: bool,
    /// Last cursor position in physical window pixels, and whether the left button is
    /// held - together the mouse-as-touch source.
    cursor: (f64, f64),
    mouse_down: bool,
    acc: Duration,
    last_tick: Instant,
    fps_since: Instant,
    /// Frames PRESENTED since the meter window opened (redraws), and the guest display
    /// flips retired in the same window. These are different numbers and the difference is
    /// the point: the guest is stepped against real time with a catch-up budget, so when
    /// emulation cannot keep up the window keeps presenting at the display rate while the
    /// guest falls behind. Reporting only the present rate would show a steady 60 while the
    /// title ran in slow motion.
    fps_frames: u32,
    guest_frames_since: u64,
    fps: f64,
    /// Emulated display flips per second, and that as a fraction of the pace the guest is
    /// being stepped at (`FRAME_DT`) - i.e. how close to real-time speed it is running.
    guest_fps: f64,
    reported_exit: bool,
}

impl ApplicationHandler for RetailApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("vitaslop")
            .with_inner_size(LogicalSize::new(GAME_W, GAME_H));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        match RetailGfx::new(window.clone()) {
            Ok(g) => {
                println!("presenting on GPU: {}", g.adapter_name);
                self.gfx = Some(g);
            }
            Err(e) => {
                eprintln!("failed to init GPU surface: {e}");
                event_loop.exit();
                return;
            }
        }
        self.last_tick = Instant::now();
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::Escape if pressed => {
                            event_loop.exit();
                            return;
                        }
                        KeyCode::Space if pressed && !event.repeat => self.paused = !self.paused,
                        _ => {}
                    }
                    self.input.set_key(code, pressed);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = (position.x, position.y);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.mouse_down = state == ElementState::Pressed;
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
        }
    }
}

impl RetailApp {
    /// The mouse-as-touch frame this instant: a front-panel finger at the cursor when
    /// the left button is held, else no touch. Cursor physical pixels map through the
    /// window size to 960x544 game space, then double to panel coords.
    fn mouse_touch(&self) -> Option<TouchFrame> {
        if !self.mouse_down {
            return None;
        }
        let (ww, wh) = self
            .window
            .as_ref()
            .map(|w| {
                let s = w.inner_size();
                (s.width.max(1) as f64, s.height.max(1) as f64)
            })
            .unwrap_or((GAME_W as f64, GAME_H as f64));
        let sx = (self.cursor.0 / ww * GAME_W as f64).clamp(0.0, GAME_W as f64);
        let sy = (self.cursor.1 / wh * GAME_H as f64).clamp(0.0, GAME_H as f64);
        Some(TouchFrame::single((sx as f32 * PANEL_SCALE) as u16, (sy as f32 * PANEL_SCALE) as u16))
    }

    fn render(&mut self) {
        self.input.pump_gamepad();
        let ctrl = self.input.ctrl_frame();
        let touch = self.mouse_touch();
        *self.input_shared.lock().unwrap() = DesktopInput { ctrl, touch };

        let now = Instant::now();
        self.acc += now.duration_since(self.last_tick);
        self.last_tick = now;

        if self.paused {
            self.acc = Duration::ZERO;
        } else {
            if self.guest.current().is_empty() {
                self.guest.advance(); // bootstrap the first frame (runs the whole boot)
            }
            // Cap catch-up so a long boot frame does not spiral.
            let mut budget = 4;
            while self.acc >= FRAME_DT && budget > 0 {
                self.acc -= FRAME_DT;
                budget -= 1;
                self.guest.advance();
            }
            if self.acc > FRAME_DT * 4 {
                self.acc = Duration::ZERO;
            }
        }

        if self.guest.finished() && !self.reported_exit {
            self.reported_exit = true;
            match self.guest.error() {
                Some(e) => eprintln!("guest exited with error: {e}"),
                None => println!("guest exited after {} frames", self.guest.frames()),
            }
        }

        if let Some(gfx) = self.gfx.as_mut() {
            let scenes = self.guest.current();
            if !scenes.is_empty() {
                gfx.present(scenes);
            }
        }
        self.update_title(now);
    }

    /// Publish both rates to the window title, four times a second.
    ///
    /// Two numbers, deliberately: `fps` is what the window PRESENTED, `guest` is how many
    /// emulated display flips actually retired, and `speed` is that against the pace the
    /// guest is stepped at. On a machine with headroom they agree and speed sits at 100%;
    /// when emulation cannot keep up, present stays pinned to the display rate and only the
    /// guest number drops - which is the number a user actually wants when asking "is this
    /// running full speed?".
    fn update_title(&mut self, now: Instant) {
        self.fps_frames += 1;
        let since = now.duration_since(self.fps_since);
        if since < Duration::from_millis(250) {
            return;
        }
        let secs = since.as_secs_f64();
        let guest_now = self.guest.frames();
        self.fps = self.fps_frames as f64 / secs;
        self.guest_fps = guest_now.saturating_sub(self.guest_frames_since) as f64 / secs;
        self.fps_frames = 0;
        self.guest_frames_since = guest_now;
        self.fps_since = now;
        let Some(w) = self.window.as_ref() else { return };
        let state = if self.guest.finished() {
            " [exited]"
        } else if self.paused {
            " [paused]"
        } else {
            ""
        };
        // The step pace is the reference for "full speed": one FRAME_DT per emulated flip.
        let target = 1.0 / FRAME_DT.as_secs_f64();
        let speed = if self.paused || self.guest.finished() {
            String::new()
        } else {
            format!("  |  speed {:.0}%", self.guest_fps / target * 100.0)
        };
        w.set_title(&format!(
            "vitaslop  |  {:.0} fps present  |  {:.0} fps guest{speed}{state}  |  frame {}",
            self.fps,
            self.guest_fps,
            guest_now
        ));
    }
}
