//! Browser app entry. A wasm-bindgen cdylib that runs the whole blob-free
//! north-star path in the browser: load the cube velf, transpile ARM/Thumb/VFP
//! to wasm, run it on the browser's own `WebAssembly` engine (see [`web_vm`]),
//! capture the GXM command stream, and play the captured frames back to a WebGPU
//! canvas through the shared cube pipeline (`vitaslop_platform::gpu`) - the same
//! pipeline the native headless renderer uses as its pixel oracle. No Sony shader
//! blob, no server: everything runs client-side.
//!
//! # Milestone boundary
//! This first browser milestone runs the guest CPU to completion up front (a
//! scripted input world, a few hundred frames), then loops the captured scenes
//! on `requestAnimationFrame`. That is enough for the primary payoff - the first
//! real perf read on browser wasm - and stays honest about what is not here yet:
//! *live* per-frame execution and interactive input need the guest to yield each
//! frame, which is the cooperative-scheduler milestone. The keyboard/gamepad ->
//! SceCtrl seam lands with that. Until then input is the scripted world below.
//!
//! The whole crate is gated to `wasm32`: on a native host build it is empty, so a
//! workspace build does not drag the browser stack onto the desktop toolchain.
#![cfg(target_arch = "wasm32")]

mod browser_sched;
mod conformance;
mod input;
mod web_vm;

pub use conformance::run_conformance;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use input::{BrowserWorld, InputState};
use vitaslop_loader as loader;
use vitaslop_platform::gpu::{CubeRenderer, GxmRenderer};
use vitaslop_runtime::capture::Scene;
use vitaslop_runtime::render::RenderSceneBuilder;
use vitaslop_runtime::{CtrlFrame, RecipeWorld, RunReport, VitaEnv, World};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

/// The committed clean-room cube (velf), embedded so the page needs no server.
const CUBE: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.velf");

/// The Vita display resolution the cube targets.
const WIDTH: u32 = 960;
const HEIGHT: u32 = 544;
/// Background clear color, matching the native render tests.
const CLEAR: [u8; 4] = [16, 16, 24, 255];
/// Frames of guest execution to run up front. More frames give a better perf
/// sample and more captured scenes to loop.
const FRAMES: u32 = 300;

/// A scripted input world for the run-to-completion pass: a virtual 60 Hz clock,
/// no input for `frames` frames, then START to trigger the cube's clean teardown
/// (`sceGxmTerminate`), which halts the run. The deterministic twin of the native
/// `RunFor` test world. Live input arrives with the cooperative scheduler.
struct ScriptedWorld {
    polls: u32,
    frames: u32,
}

impl World for ScriptedWorld {
    fn monotonic_us(&mut self) -> u64 {
        self.polls as u64 * 16_666
    }
    fn wall_us(&mut self) -> u64 {
        0
    }
    fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
        self.polls += 1;
        let mut f = CtrlFrame::default();
        if self.polls > self.frames {
            f.buttons = 0x0000_0008; // START
        }
        f
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        buf.fill(0);
    }
}

/// The result of the CPU pass: the captured scenes plus the timings that are the
/// point of this milestone.
struct CpuRun {
    scenes: Vec<Scene>,
    transpile_ms: f64,
    run_ms: f64,
    /// How the scheduler run ended (for the status line / diagnostics).
    report: String,
}

/// Run the cube through the browser PREEMPTIVE scheduler (the JSPI engine over the
/// shared `SchedCore`), the same architecture the real multi-thread game needs. The
/// cube is single-threaded, so it exercises the loop's Continue/Yield(flip)/Halt path
/// and proves JSPI suspension drives a real transpiled guest in the browser. Returns
/// the captured scenes and timings.
async fn run_cube_scheduled() -> Result<CpuRun, JsValue> {
    let perf = web_sys::window()
        .and_then(|w| w.performance())
        .ok_or_else(|| JsValue::from_str("no performance clock"))?;

    let m = loader::load(CUBE).map_err(|e| JsValue::from_str(&format!("load: {e:?}")))?;
    // Emit an importable shared linear memory so every guest thread's instance can
    // share one address space (the scheduler's requirement).
    let mut inputs = m.program_inputs();
    inputs.import_memory = true;

    let t0 = perf.now();
    let artifact = vitaslop_transpiler::transpile(&inputs.program())
        .map_err(|e| JsValue::from_str(&format!("transpile: {e:?}")))?;
    let transpile_ms = perf.now() - t0;

    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();
    let world = Box::new(ScriptedWorld { polls: 0, frames: FRAMES });
    let mut venv = VitaEnv::new(imports, inputs.base, inputs.mem_bytes, world);
    venv.state.halt_on_terminate = true;
    venv.state.set_preemptive(true);

    let main_sp = inputs.base.wrapping_add(inputs.mem_bytes);
    let module = browser_sched::compile_module(&artifact.wasm).await?;
    let mut sched = browser_sched::BrowserSched::new(
        module,
        &inputs.code,
        inputs.base,
        artifact.mem_pages,
        m.entry & !1,
        main_sp,
        venv,
    )?;

    let t1 = perf.now();
    let report = browser_sched::run_frames(&mut sched.core, FRAMES as u64, 50_000_000).await;
    let run_ms = perf.now() - t1;

    let scenes = sched.host.lock().unwrap().state.capture.scenes.clone();
    Ok(CpuRun { scenes, transpile_ms, run_ms, report: format!("{report:?}") })
}

/// A live frames-per-second meter driven off the render loop's own wall clock. It
/// counts presented frames and, twice a second, publishes the rate to the page's
/// `#fps` element. The cost is a `performance.now()` read and an integer bump per
/// frame plus one DOM text write per half-second, so it is cheap enough to leave on
/// always rather than behind a debug flag - the number is the whole point of a
/// browser build, and users should see it live.
struct FpsMeter {
    perf: web_sys::Performance,
    report: Report,
    window_start: f64,
    window_frames: u32,
    /// Last published rate, so callers (and a headless test) can read it back.
    last_fps: f64,
}

/// How often to recompute and publish the rate. Half a second is responsive but
/// long enough to average out per-frame jitter.
const FPS_WINDOW_MS: f64 = 500.0;

/// A sink for the small status/FPS/perf strings the run publishes to the page. On the
/// main thread it writes DOM elements by id; in a Web Worker (which has no DOM) it
/// forwards `(id, text)` to a JS callback that turns it into a `postMessage` the page
/// applies - so the same live loop reports its metrics either way.
#[derive(Clone)]
struct Report {
    sink: Rc<dyn Fn(&str, &str)>,
}

impl Report {
    /// Main-thread sink: `document.getElementById(id).textContent = text`.
    fn dom() -> Report {
        let document = web_sys::window().and_then(|w| w.document());
        Report {
            sink: Rc::new(move |id: &str, text: &str| {
                if let Some(el) = document.as_ref().and_then(|d| d.get_element_by_id(id)) {
                    el.set_text_content(Some(text));
                }
            }),
        }
    }

    /// Worker sink: forward `(id, text)` to a JS callback (which posts it to the page).
    fn callback(f: js_sys::Function) -> Report {
        Report {
            sink: Rc::new(move |id: &str, text: &str| {
                let _ = f.call2(
                    &JsValue::UNDEFINED,
                    &JsValue::from_str(id),
                    &JsValue::from_str(text),
                );
            }),
        }
    }

    fn emit(&self, id: &str, text: &str) {
        (self.sink)(id, text);
    }
}

impl FpsMeter {
    fn new(perf: web_sys::Performance, report: Report) -> FpsMeter {
        let now = perf.now();
        FpsMeter { perf, report, window_start: now, window_frames: 0, last_fps: 0.0 }
    }

    /// Record one presented frame; publish the rate when the window elapses.
    fn tick(&mut self) {
        self.window_frames += 1;
        let now = self.perf.now();
        let dt = now - self.window_start;
        if dt >= FPS_WINDOW_MS {
            self.last_fps = self.window_frames as f64 * 1000.0 / dt;
            self.report.emit("fps", &format!("fps: {:.0}", self.last_fps));
            self.window_start = now;
            self.window_frames = 0;
        }
    }
}

/// GPU state for the playback loop, owned across `requestAnimationFrame` ticks.
struct Playback {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    cube: CubeRenderer,
    depth: wgpu::TextureView,
    scenes: Vec<Scene>,
    frame: usize,
    fps: FpsMeter,
}

impl Playback {
    /// Acquire WebGPU on `canvas` and build the shared cube pipeline.
    async fn new(canvas: HtmlCanvasElement, scenes: Vec<Scene>) -> Result<Playback, JsValue> {
        canvas.set_width(WIDTH);
        canvas.set_height(HEIGHT);

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| JsValue::from_str(&format!("create_surface: {e}")))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| JsValue::from_str("no WebGPU adapter (browser support required)"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("vitaslop-web"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("request_device: {e}")))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: WIDTH,
                height: HEIGHT,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let cube = CubeRenderer::new(&device, format);
        let depth = make_depth(&device);
        let perf = web_sys::window()
            .and_then(|w| w.performance())
            .ok_or_else(|| JsValue::from_str("no performance clock"))?;
        let fps = FpsMeter::new(perf, Report::dom());
        Ok(Playback { surface, device, queue, cube, depth, scenes, frame: 0, fps })
    }

    /// Render the next captured scene to the canvas.
    fn render_next(&mut self) -> Result<(), JsValue> {
        if self.scenes.is_empty() {
            return Ok(());
        }
        let scene = &self.scenes[self.frame % self.scenes.len()];
        self.frame += 1;
        let batches = scene.draw_batches();

        // get_current_texture returns an enum in wgpu 30: render on Success or
        // Suboptimal, skip this frame on any transient status (resize, lost, etc).
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return Ok(()),
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.cube
            .encode(&self.device, &mut encoder, &view, &self.depth, &batches, CLEAR);
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
        self.fps.tick();
        Ok(())
    }
}

/// The cube pipeline's depth attachment, sized to the canvas.
fn make_depth(device: &wgpu::Device) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: vitaslop_platform::gpu::DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&Default::default())
}

/// Drive `render_next` on every `requestAnimationFrame`. The classic wasm rAF
/// pattern: a closure that reschedules itself, kept alive by the `Rc` cell it
/// holds a reference to.
fn start_raf_loop(playback: Playback) {
    let playback = Rc::new(RefCell::new(playback));
    let holder: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let holder2 = holder.clone();
    *holder.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        if let Err(e) = playback.borrow_mut().render_next() {
            web_sys::console::error_1(&e);
            return; // Stop the loop on a render error rather than spamming.
        }
        request_animation_frame(holder2.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));
    request_animation_frame(holder.borrow().as_ref().unwrap());
}

fn request_animation_frame(cb: &Closure<dyn FnMut()>) {
    web_sys::window()
        .expect("window")
        .request_animation_frame(cb.as_ref().unchecked_ref())
        .expect("request_animation_frame");
}

/// Headroom below the top of the guest region for the main thread's startup stack
/// (ELF/crt scratch), matching native's `MAIN_STACK_HEADROOM`.
const MAIN_STACK_HEADROOM: u32 = 0x0010_0000;

/// The main thread's initial stack pointer: near the top of the guest region, with
/// startup headroom, 16-aligned (native's `main_stack_top`).
fn main_stack_top(base: u32, mem_bytes: u32) -> u32 {
    (base.wrapping_add(mem_bytes).wrapping_sub(MAIN_STACK_HEADROOM)) & !0xF
}

/// Live WebGPU playback of a real title through the general GXM renderer
/// ([`GxmRenderer`]) - the browser's production render path, the GPU twin of the
/// native software oracle. Unlike the cube [`Playback`], this holds the general
/// renderer and a persistent [`RenderSceneBuilder`] (so its texture-decode cache
/// stays warm across frames), and presents one freshly-executed scene per call
/// rather than looping a canned capture.
struct LivePlayback {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    gxm: GxmRenderer,
    builder: RenderSceneBuilder,
    depth: wgpu::TextureView,
    render_format: wgpu::TextureFormat,
    fps: FpsMeter,
}

impl LivePlayback {
    /// Acquire WebGPU on `target` (a canvas on the main thread, or an OffscreenCanvas
    /// in a worker) and build the general pipeline. `report` sinks the FPS meter.
    async fn new(
        target: wgpu::SurfaceTarget<'static>,
        report: Report,
    ) -> Result<LivePlayback, JsValue> {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(target)
            .map_err(|e| JsValue::from_str(&format!("create_surface: {e}")))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| JsValue::from_str("no WebGPU adapter (browser support required)"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("vitaslop-web-gxm"),
                required_features: wgpu::Features::empty(),
                // Raise the resolution-derived limits to the adapter's: a real title
                // binds textures past the conservative WebGL2 floor (some titles have
                // a ~2480px atlas). WebGPU guarantees at least 8192, so this is safe.
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("request_device: {e}")))?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        // The GXM decode yields final display-ready (sRGB-encoded) byte values, matching
        // the `Rgba8Unorm` software oracle. Render through a non-sRGB view so those bytes
        // land verbatim rather than getting a second sRGB encode (see the desktop
        // `RetailGfx` for the full note). WebGPU's preferred canvas format is normally
        // already non-sRGB, so this is usually a no-op in the browser.
        let render_format = format.remove_srgb_suffix();
        let view_formats = if render_format == format { vec![] } else { vec![render_format] };
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: WIDTH,
                height: HEIGHT,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats,
                desired_maximum_frame_latency: 2,
            },
        );

        let gxm = GxmRenderer::new(&device, &queue, render_format);
        let depth = make_depth(&device);
        let perf = global_performance().ok_or_else(|| JsValue::from_str("no performance clock"))?;
        let fps = FpsMeter::new(perf, report);
        Ok(LivePlayback {
            surface,
            device,
            queue,
            gxm,
            builder: RenderSceneBuilder::new(),
            depth,
            render_format,
            fps,
        })
    }

    /// Render one freshly-executed scene to the canvas through the general renderer.
    fn present(&mut self, scene: &Scene) {
        let render_scene = self.builder.build(scene);
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.render_format),
            ..Default::default()
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.gxm.encode(
            &self.device,
            &self.queue,
            &mut encoder,
            &view,
            &self.depth,
            &render_scene,
            WIDTH,
            HEIGHT,
            CLEAR,
        );
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
        self.fps.tick();
    }
}

/// The `performance` clock from whichever global we run in - `window.performance` on
/// the main thread, `self.performance` in a Web Worker (which has no `window`).
fn global_performance() -> Option<web_sys::Performance> {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("performance"))
        .ok()
        .and_then(|p| p.dyn_into::<web_sys::Performance>().ok())
}

/// Yield to the event loop for one frame tick, resolving asynchronously. On the main
/// thread this is `requestAnimationFrame` - it paces the live loop to the display
/// refresh (so presented FPS reflects real cadence) and keeps input listeners
/// responsive. In a Web Worker (no `requestAnimationFrame`) it falls back to a
/// zero-delay `setTimeout`, which still returns control to the worker's event loop
/// each frame (so posted input/messages are processed) but does not vsync-pace - the
/// worker then runs near its true uncapped throughput.
async fn next_tick() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let cb = Closure::once_into_js(move |_t: JsValue| {
            let _ = resolve.call0(&JsValue::UNDEFINED);
        });
        if let Some(window) = web_sys::window() {
            let _ = window.request_animation_frame(cb.as_ref().unchecked_ref());
        } else {
            // Worker: no rAF. `setTimeout(cb, 0)` off the global still yields a macrotask
            // so incoming messages are serviced between frames.
            let set_timeout =
                js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("setTimeout"))
                    .ok()
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok());
            if let Some(set_timeout) = set_timeout {
                let _ = set_timeout.call2(
                    &JsValue::UNDEFINED,
                    cb.as_ref().unchecked_ref(),
                    &JsValue::from_f64(0.0),
                );
            }
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// Per-frame scheduler round cap for the live loop. Steady frames retire far fewer
/// resumes than this, but the FIRST frame runs the entire boot (every `module_init`
/// then the eboot entry) up to the first display flip, which is millions of resumes -
/// so the cap is generous. It only backstops a livelocked frame so it terminates
/// rather than hanging the tab forever.
const PER_FRAME_ROUNDS: u64 = 60_000_000;

/// The heavy one-time setup shared by the main-thread and worker live paths: build the
/// guest filesystem from the fetched container bytes, decrypt + link + transpile the
/// title, and stand up the preemptive scheduler over a [`BrowserWorld`]. Returns the
/// ready scheduler plus the numbers for the setup status line. `live` is the shared
/// input cell the world reads (fed by DOM listeners on the main thread; unused but
/// harmless in the worker recipe path).
struct GameSetup {
    sched: browser_sched::BrowserSched,
    nfiles: usize,
    n_modules: usize,
    image_kib: usize,
    decrypt_ms: f64,
    transpile_ms: f64,
    scripted: bool,
}

impl GameSetup {
    /// The one-line setup status, tagged with where the run lives (main thread / worker).
    fn status(&self, home: &str) -> String {
        format!(
            "retail title live in browser ({home}): {} files, {} modules, image {} KiB | decrypt/link {:.0} ms + transpile {:.0} ms (one-time){} | rendering live via WebGPU",
            self.nfiles,
            self.n_modules,
            self.image_kib,
            self.decrypt_ms,
            self.transpile_ms,
            if self.scripted { " | scripted recipe" } else { "" },
        )
    }
}

async fn setup_game(
    files: JsValue,
    recipe: &str,
    live: Arc<Mutex<InputState>>,
) -> Result<GameSetup, JsValue> {
    use vitaslop_runtime::ingest::pipeline::decrypt_container;
    use vitaslop_runtime::ingest::vfs::MemVfs;
    use vitaslop_runtime::link::link;

    let perf = global_performance().ok_or_else(|| JsValue::from_str("no performance clock"))?;

    // Build the input VFS from the JS { path: Uint8Array } object.
    let obj: js_sys::Object = files.dyn_into()?;
    let mut vfs = MemVfs::new();
    let mut nfiles = 0usize;
    for entry in js_sys::Object::entries(&obj).iter() {
        let pair: js_sys::Array = entry.into();
        let path = pair.get(0).as_string().ok_or_else(|| JsValue::from_str("bad file path"))?;
        let bytes = js_sys::Uint8Array::new(&pair.get(1)).to_vec();
        vfs.insert(path, bytes);
        nfiles += 1;
    }

    let t_dec = perf.now();
    let game = decrypt_container(&vfs).map_err(|e| JsValue::from_str(&format!("decrypt: {e:?}")))?;
    let modules = game
        .modules
        .iter()
        .map(|m| loader::load(&m.elf))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| JsValue::from_str(&format!("load module: {e:?}")))?;
    let linked = link(modules).map_err(|e| JsValue::from_str(&format!("link: {e:?}")))?;
    // The raw container bytes are done with; release them now - for a large title
    // this is hundreds of megabytes the transpile step should not have to share the
    // wasm heap with.
    drop(vfs);
    let decrypt_ms = perf.now() - t_dec;

    // The input world: a scripted recipe (if given) overlaid with live pointer/keyboard
    // input, both feeding the same touch/pad seam the native probe drives.
    let scripted = !recipe.trim().is_empty();
    let recipe_world = if scripted {
        Some(RecipeWorld::parse(recipe).map_err(|e| JsValue::from_str(&format!("recipe: {e}")))?)
    } else {
        None
    };
    let world = Box::new(BrowserWorld::new(recipe_world, live));
    let mut env = VitaEnv::new(linked.imports.clone(), linked.base, linked.mem_bytes, world);
    env.state.set_alloc_base(linked.alloc_base);
    env.state.set_process_param(linked.process_param);
    env.state.set_tls_template(linked.tls_template);
    env.state.set_preemptive(true);
    // Move (not clone) the decrypted assets into the guest filesystem: for a large
    // 3D title this is hundreds of megabytes, and the browser heap is the tightest
    // memory budget we run in.
    for (path, bytes) in game.files.into_files() {
        env.state.add_file(&path, bytes);
    }

    let t_tr = perf.now();
    let built = vitaslop_transpiler::transpile_lenient(&linked.shared_program());
    let transpile_ms = perf.now() - t_tr;

    let main_sp = main_stack_top(linked.base, linked.mem_bytes);
    let module = browser_sched::compile_module(&built.artifact.wasm).await?;
    let sched = browser_sched::BrowserSched::from_linked(
        module,
        &linked.image,
        linked.base,
        built.artifact.mem_pages,
        &linked.module_inits,
        main_sp,
        env,
    )?;

    Ok(GameSetup {
        sched,
        nfiles,
        n_modules: linked.module_inits.len(),
        image_kib: linked.image.len() / 1024,
        decrypt_ms,
        transpile_ms,
        scripted,
    })
}

/// Boot the REAL retail title LIVE on the MAIN THREAD: decrypt + link + transpile, then
/// run the guest frame-by-frame through the JSPI preemptive scheduler, rendering each
/// freshly-executed frame to the WebGPU `canvas` through the general GXM renderer and
/// feeding real input (pointer/keyboard on the canvas, plus an optional scripted
/// `recipe`) through the browser [`BrowserWorld`]. Returns after setup once the live
/// loop is spawned; the loop then runs on the event loop, updating the on-page FPS
/// meter and status. `max_frames` bounds the run (display flips); `max_rounds` is unused
/// (kept for API compatibility - the live loop caps rounds per frame).
///
/// Note: instantiating the title's (large) transpiled module synchronously mid-run
/// needs the `WebAssemblyUnlimitedSyncCompilation` flag on the main thread; the worker
/// entry ([`run_game_worker`]) is the flag-free production home.
#[wasm_bindgen]
pub async fn run_game(
    canvas: JsValue,
    files: JsValue,
    recipe: String,
    max_frames: u32,
    max_rounds: f64,
) -> Result<String, JsValue> {
    let _ = max_rounds;
    console_error_panic_hook::set_once();

    let live: Arc<Mutex<InputState>> = Arc::new(Mutex::new(InputState::default()));
    let setup = setup_game(files, &recipe, live.clone()).await?;

    // Acquire WebGPU on the canvas and wire live input, then hand both to the live loop.
    let canvas: HtmlCanvasElement = canvas.dyn_into()?;
    canvas.set_width(WIDTH);
    canvas.set_height(HEIGHT);
    input::install_listeners(&canvas, &live);
    let report = Report::dom();
    let playback =
        LivePlayback::new(wgpu::SurfaceTarget::Canvas(canvas), report.clone()).await?;

    let status = setup.status("main thread");
    web_sys::console::log_1(&JsValue::from_str(&status));
    wasm_bindgen_futures::spawn_local(live_loop(setup.sched, playback, report, max_frames as u64));
    Ok(status)
}

/// Boot the REAL retail title LIVE INSIDE A WEB WORKER: the same run as [`run_game`] but
/// presenting to a transferred `OffscreenCanvas` and reporting metrics through
/// `report_fn` (which the worker turns into a `postMessage` the page applies to its FPS/
/// status elements, since a worker has no DOM). This is the production home: a worker
/// allows synchronous instantiation of the title's large transpiled module at any size,
/// so it needs no `WebAssemblyUnlimitedSyncCompilation` flag (the one main-thread
/// caveat), and it is where the later multi-worker SMP step lands. Input here is the
/// scripted `recipe`; forwarding live pointer/keyboard from the page is a follow-up.
#[wasm_bindgen]
pub async fn run_game_worker(
    offscreen: JsValue,
    files: JsValue,
    recipe: String,
    max_frames: u32,
    report_fn: js_sys::Function,
) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();

    let live: Arc<Mutex<InputState>> = Arc::new(Mutex::new(InputState::default()));
    // Register the shared input cell so the page's forwarded pointer/keyboard messages
    // (via the exported worker_input_* functions) reach this run's world.
    input::set_worker_input(live.clone());
    let setup = setup_game(files, &recipe, live).await?;

    let offscreen: web_sys::OffscreenCanvas = offscreen.dyn_into()?;
    offscreen.set_width(WIDTH);
    offscreen.set_height(HEIGHT);
    let report = Report::callback(report_fn);
    let playback =
        LivePlayback::new(wgpu::SurfaceTarget::OffscreenCanvas(offscreen), report.clone()).await?;

    let status = setup.status("web worker");
    web_sys::console::log_1(&JsValue::from_str(&status));
    wasm_bindgen_futures::spawn_local(live_loop(setup.sched, playback, report, max_frames as u64));
    Ok(status)
}

/// The live run: step the guest one display frame, render it through the general GXM
/// renderer, pace to the display refresh, repeat - until `max_frames` flips or the run
/// ends. This is what makes the browser build *live* (the guest computes each frame on
/// demand and reacts to input) rather than replaying a canned capture. The presented
/// FPS the meter shows is the true combined guest-CPU + render cadence.
async fn live_loop(
    mut sched: browser_sched::BrowserSched,
    mut playback: LivePlayback,
    report: Report,
    max_frames: u64,
) {
    let perf = global_performance();

    // Real-time pacing: advance the guest at 60 Hz of WALL-CLOCK time, not as fast as
    // the event loop ticks. Without this the game runs too fast on a >60 Hz display
    // (rAF fires 120x/s) or in a worker (setTimeout is not vsync-paced). An accumulator
    // catches up if a frame ran long, capped so the multi-second boot frame does not
    // trigger a catch-up spiral.
    const FRAME_MS: f64 = 1000.0 / 60.0;
    const MAX_CATCHUP_MS: f64 = 4.0 * FRAME_MS;

    // Rolling per-frame timing, so the page can report the CPU-bound throughput (guest
    // execution, independent of the pacing) separately from the paced present rate.
    // Early frames (boot to first flip + JIT warmup) are excluded.
    const WARMUP_FRAMES: u64 = 6;
    const PERF_WINDOW: u32 = 30;
    let mut cpu_ms = 0.0f64;
    let mut cpu_frames = 0u32;
    let mut render_ms = 0.0f64;
    let mut presents = 0u32;

    let now = || perf.as_ref().map(|p| p.now()).unwrap_or(0.0);
    let mut acc = 0.0f64;
    let mut last = now();

    'run: loop {
        next_tick().await;
        let t = now();
        acc = (acc + (t - last)).min(MAX_CATCHUP_MS);
        last = t;

        // Advance as many whole 60 Hz frames as wall time has accrued (usually one),
        // keeping only the newest scene to present - so the GPU present rate follows the
        // display/tick rate, not the catch-up count.
        let mut latest = None;
        while acc >= FRAME_MS {
            acc -= FRAME_MS;
            // Run to exactly one more display flip (the frame counter is cumulative
            // across calls, so `frames + 1` advances by a single frame).
            let target = sched.core.frames() + 1;
            let c0 = now();
            let report_step =
                browser_sched::run_frames(&mut sched.core, target, PER_FRAME_ROUNDS).await;
            let c1 = now();

            // Take the scene presented this frame and drop the rest (render-to-texture
            // intermediates); clearing the per-frame capture vectors bounds the capture's
            // memory during a long run (they exist only for post-mortem debugging).
            let scene = {
                let mut host = sched.host.lock().unwrap();
                let cap = &mut host.state.capture;
                let last_scene = cap.scenes.pop();
                cap.scenes.clear();
                cap.trace.clear();
                cap.trace_thid.clear();
                cap.presents.clear();
                last_scene
            };
            if scene.is_some() {
                latest = scene;
            }

            let frames = sched.core.frames();
            if frames > WARMUP_FRAMES {
                cpu_ms += c1 - c0;
                cpu_frames += 1;
            }
            report.emit("status", &format!("frame {frames} (live via WebGPU) | {report_step:?}"));
            match report_step {
                // Reached the next flip and still within budget: keep going.
                RunReport::FramesReached(n) if n < max_frames => {}
                // Hit the frame budget, or the run finished / deadlocked / trapped.
                _ => {
                    web_sys::console::log_1(&JsValue::from_str(&format!(
                        "live run ended at frame {frames}: {report_step:?}"
                    )));
                    if let Some(scene) = &latest {
                        playback.present(scene);
                    }
                    break 'run;
                }
            }
        }

        // Present at most one (the newest) frame per tick, and fold its render time into
        // the rolling perf report.
        if let Some(scene) = latest {
            let r0 = now();
            playback.present(&scene);
            let r1 = now();
            if sched.core.frames() > WARMUP_FRAMES {
                render_ms += r1 - r0;
                presents += 1;
            }
            if presents >= PERF_WINDOW {
                let cpu_avg = if cpu_frames > 0 { cpu_ms / cpu_frames as f64 } else { 0.0 };
                let render_avg = render_ms / presents as f64;
                let cpu_fps = if cpu_avg > 0.0 { 1000.0 / cpu_avg } else { 0.0 };
                report.emit(
                    "perf",
                    &format!(
                        "cpu {cpu_avg:.1} ms/frame ({cpu_fps:.0} fps uncapped) | render {render_avg:.1} ms"
                    ),
                );
                cpu_ms = 0.0;
                cpu_frames = 0;
                render_ms = 0.0;
                presents = 0;
            }
        }
    }
}

/// Entry point the page calls with the target `<canvas>`. Runs the cube CPU pass,
/// reports timings, then starts WebGPU playback of the captured frames. Returns a
/// short status string (also useful for the page to display).
#[wasm_bindgen]
pub async fn run(canvas: JsValue) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    let canvas: HtmlCanvasElement = canvas.dyn_into()?;

    let cpu = run_cube_scheduled().await?;
    let status = format!(
        "cube ran on browser JSPI scheduler ({}): {} scenes captured, transpile {:.1} ms, {} frames in {:.1} ms ({:.0} us/frame)",
        cpu.report,
        cpu.scenes.len(),
        cpu.transpile_ms,
        FRAMES,
        cpu.run_ms,
        cpu.run_ms * 1000.0 / FRAMES as f64,
    );
    web_sys::console::log_1(&JsValue::from_str(&status));

    let playback = Playback::new(canvas, cpu.scenes).await?;
    start_raf_loop(playback);
    Ok(status)
}
