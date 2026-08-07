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

mod audio;
mod browser_sched;
mod conformance;
mod input;
mod logging;
mod opfs;
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
        artifact.mirror_off,
        m.entry & !1,
        main_sp,
        venv,
    )?;

    let t1 = perf.now();
    let report =
        browser_sched::run_frames(&mut sched.core, FRAMES as u64, 50_000_000, &mut |_| {}).await;
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
    /// While set, the meter publishes what it is doing INSTEAD of a rate. A fast-forward
    /// presents once per event-loop tick rather than once per guest frame, so the rate it
    /// would compute describes the tick cadence and nothing about how fast the emulator
    /// runs. Publishing that number would be worse than publishing none.
    paused: bool,
}

/// How often to recompute and publish the rate. Half a second is responsive but
/// long enough to average out per-frame jitter.
const FPS_WINDOW_MS: f64 = 500.0;

/// The console's display rate, and so the reference for "full speed". The live loop
/// advances the guest exactly one display flip per presented frame, so the presented
/// rate IS the emulated rate and dividing by this gives the speed the title is running
/// at. Worth showing next to the raw number: on the main thread the loop is paced by
/// `requestAnimationFrame`, so the rate is capped at the display refresh and a healthy
/// run reads a flat 60 whether it has 2x headroom or none - the percentage is what says
/// "keeping up" and, in a Worker (uncapped), how much room is left.
const GUEST_DISPLAY_HZ: f64 = 60.0;

/// A sink for the small status/FPS/perf strings the run publishes to the page. On the
/// main thread it writes DOM elements by id; in a Web Worker (which has no DOM) it
/// forwards `(id, text)` to a JS callback that turns it into a `postMessage` the page
/// applies - so the same live loop reports its metrics either way.
#[derive(Clone)]
struct Report {
    sink: Rc<dyn Fn(&str, &str)>,
    /// When each id was last published, for the rate limit in [`Report::emit`]. Shared
    /// across clones, because the live loop clones this per frame for its progress
    /// callback and a per-clone limiter would not limit anything.
    last: Rc<RefCell<std::collections::HashMap<String, f64>>>,
    perf: Option<web_sys::Performance>,
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
            last: Rc::new(RefCell::new(std::collections::HashMap::new())),
            perf: global_performance(),
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
            last: Rc::new(RefCell::new(std::collections::HashMap::new())),
            perf: global_performance(),
        }
    }

    /// Publish `text` under `id`, at most [`REPORT_MIN_INTERVAL_MS`] apart per id.
    ///
    /// # Why a status line has to be rate-limited
    /// From a worker every emit is a `postMessage`, and the page's handler runs on the
    /// MAIN thread. The live loop emits per FRAME, and a fast-forward runs as many frames
    /// as fit each tick - so a healthy fast-forward posts thousands of messages a second
    /// at a main thread that can only drain them one task at a time. The queue then grows
    /// without bound, the page stops answering anything (including the harness asking how
    /// far the run has got), and eventually something is killed.
    ///
    /// That is the "page stopped answering while Chrome was still burning CPU" that went
    /// unexplained for two sessions. A status nobody can read is worth nothing, and ten a
    /// second is already more than a human or a 15-second poller can use.
    fn emit(&self, id: &str, text: &str) {
        let now = self.now();
        let mut last = self.last.borrow_mut();
        match last.get(id) {
            Some(&t) if now - t < REPORT_MIN_INTERVAL_MS => return,
            _ => last.insert(id.to_string(), now),
        };
        drop(last);
        (self.sink)(id, text);
    }

    /// Publish unconditionally, ignoring the rate limit. For the last word on a run - the
    /// frame it ended at and why - which must never be the one the limiter drops.
    fn emit_final(&self, id: &str, text: &str) {
        self.last.borrow_mut().insert(id.to_string(), self.now());
        (self.sink)(id, text);
    }

    fn now(&self) -> f64 {
        self.perf.as_ref().map(|p| p.now()).unwrap_or(0.0)
    }
}

/// Smallest gap between two published updates of the same id. Ten a second: faster than
/// anything reads them, slow enough that the transport cannot be saturated.
const REPORT_MIN_INTERVAL_MS: f64 = 100.0;

impl FpsMeter {
    fn new(perf: web_sys::Performance, report: Report) -> FpsMeter {
        let now = perf.now();
        FpsMeter { perf, report, window_start: now, window_frames: 0, last_fps: 0.0, paused: false }
    }

    /// Drop the in-flight window and start a fresh one.
    ///
    /// Called when the run changes character (leaving a fast-forward), so the first rate
    /// published afterwards is measured entirely under the new regime rather than
    /// averaging across the change.
    fn reset(&mut self) {
        self.window_start = self.perf.now();
        self.window_frames = 0;
        self.last_fps = 0.0;
    }

    /// Suspend or resume rate publishing. See [`FpsMeter::paused`].
    fn set_paused(&mut self, paused: bool) {
        if paused != self.paused {
            self.paused = paused;
            self.reset();
        }
    }

    /// Record one presented frame; publish the rate when the window elapses.
    fn tick(&mut self) {
        if self.paused {
            self.report.emit("fps", "fps: -- (fast-forwarding, not a real-time rate)");
            return;
        }
        self.window_frames += 1;
        let now = self.perf.now();
        let dt = now - self.window_start;
        if dt >= FPS_WINDOW_MS {
            self.last_fps = self.window_frames as f64 * 1000.0 / dt;
            self.report.emit(
                "fps",
                &format!(
                    "fps: {:.0} ({:.0}% speed)",
                    self.last_fps,
                    self.last_fps / GUEST_DISPLAY_HZ * 100.0
                ),
            );
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
                // The same limits the native pixel oracle asks for - see the note on
                // the retail device below. NOT the WebGL2 downlevel set: this is a
                // WebGPU device.
                required_limits: wgpu::Limits::downlevel_defaults()
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

/// Supersample factor for the live browser render (`VITASLOP_BROWSER_SUPERSAMPLE`).
///
/// Defaults to 2, which is what the desktop review path uses, so a browser shot and a
/// desktop shot are comparable by construction. Turn it down for a heavier scene or a
/// weaker GPU; 1 is native resolution. A value that is not a positive integer is an
/// ERROR rather than a silent fallback to the default - a run configured by a typo
/// would otherwise publish a rate for a resolution nobody asked for.
fn supersample() -> u32 {
    match vitaslop_runtime::knobs::var("VITASLOP_BROWSER_SUPERSAMPLE") {
        Err(_) => 2,
        Ok(v) => v
            .parse::<u32>()
            .ok()
            .filter(|n| *n >= 1)
            .unwrap_or_else(|| panic!("VITASLOP_BROWSER_SUPERSAMPLE={v} is not a positive integer")),
    }
}

/// Frame to fast-forward the live loop to (`VITASLOP_BROWSER_FASTFORWARD`), unpaced.
///
/// # Why an emulator that is keeping up still needs this
/// The live loop advances the guest at 60 Hz of WALL-CLOCK time, which is the right
/// behaviour for playing - but it also means that reaching a screen tens of thousands of
/// frames into a title costs that many sixtieths of a second no matter how much headroom
/// the machine has. One retail racer's race is ~44,700 flips: twelve and a half minutes
/// at best, before any per-frame cost. Below this frame the loop runs as fast as the
/// machine allows and presents only the newest frame; at and above it the loop paces
/// normally, so the fps the meter publishes is a real-time rate and not a fast-forward
/// artefact. Zero (the default) means never fast-forward.
fn fastforward_to() -> u64 {
    match vitaslop_runtime::knobs::var("VITASLOP_BROWSER_FASTFORWARD") {
        Err(_) => 0,
        Ok(v) => v
            .parse::<u64>()
            .unwrap_or_else(|_| panic!("VITASLOP_BROWSER_FASTFORWARD={v} is not a frame number")),
    }
}

/// Whether a run may proceed on a software rasteriser (`VITASLOP_ALLOW_SOFTWARE_GPU`).
///
/// OFF by default, and the default is the point. SwiftShader renders a plausible picture
/// at roughly a thirtieth of the speed, so a software run publishes a frame rate that
/// describes the rasteriser while looking exactly like a browser result. It stays
/// available for a machine with no GPU at all, where seeing the frame is still worth
/// something - but a run that does not ask for it now refuses rather than quietly
/// producing a number nobody can use.
fn allow_software_gpu() -> bool {
    vitaslop_runtime::knobs::flag("VITASLOP_ALLOW_SOFTWARE_GPU")
}

/// What the browser's WebGPU adapter actually IS, and whether it is a real GPU.
///
/// # Why this is not optional decoration
/// An fps without the backend that produced it is not a measurement. Every browser run
/// of this project so far launched Chrome with `--enable-unsafe-swiftshader`, and a
/// headless Chrome has no GPU - so every frame was CPU-rasterised and every published
/// rate described SwiftShader, not a browser anyone uses. Nothing in the run said so.
/// This type is how a run says so, and by default how it REFUSES: the house rule is that
/// a fallback reports itself, and a software rasteriser is the largest fallback in the
/// system.
struct AdapterProbe {
    /// A one-line description for the page and the harness.
    summary: String,
    /// True when the adapter is a CPU rasteriser rather than a GPU.
    software: bool,
}

/// Names a software WebGPU implementation goes by, lowercased. Chrome reports SwiftShader
/// through `architecture`; the others turn up on Linux/CI Vulkan stacks and on D3D WARP.
const SOFTWARE_ADAPTER_MARKERS: &[&str] =
    &["swiftshader", "llvmpipe", "lavapipe", "softpipe", "warp", "basic render", "microsoft basic"];

/// Ask `navigator.gpu` directly what adapter this page would get, and read the fields
/// wgpu's WebGPU backend does not surface.
///
/// wgpu's `AdapterInfo` on the WebGPU backend carries only the description string and a
/// `device_type` that is `Cpu` solely for a *fallback* adapter - which SwiftShader-behind-
/// `--enable-unsafe-swiftshader` is NOT: Chrome hands it over as an ordinary adapter. The
/// vendor/architecture fields, which do name it, are only reachable through the raw
/// `GPUAdapterInfo`. Requesting a second adapter is cheap (the page gets the same one) and
/// is the only way to answer the question honestly.
async fn probe_webgpu_adapter() -> Option<AdapterProbe> {
    use js_sys::{Function, Object, Reflect};
    let global = js_sys::global();
    // `navigator` is `Navigator` on the page and `WorkerNavigator` in a worker; both
    // carry `gpu`, so reflection covers the two homes with one path.
    let navigator = Reflect::get(&global, &JsValue::from_str("navigator")).ok()?;
    let gpu = Reflect::get(&navigator, &JsValue::from_str("gpu")).ok()?;
    let request: Function = Reflect::get(&gpu, &JsValue::from_str("requestAdapter"))
        .ok()?
        .dyn_into()
        .ok()?;
    let options = Object::new();
    Reflect::set(
        &options,
        &JsValue::from_str("powerPreference"),
        &JsValue::from_str("high-performance"),
    )
    .ok()?;
    let promise: js_sys::Promise = request.call1(&gpu, &options).ok()?.dyn_into().ok()?;
    let adapter = wasm_bindgen_futures::JsFuture::from(promise).await.ok()?;
    if adapter.is_null() || adapter.is_undefined() {
        return None;
    }
    let info = Reflect::get(&adapter, &JsValue::from_str("info")).ok()?;
    let field = |key: &str| {
        Reflect::get(&info, &JsValue::from_str(key))
            .ok()
            .and_then(|v| v.as_string())
            .filter(|s| !s.is_empty())
    };
    let vendor = field("vendor").unwrap_or_else(|| "?".into());
    let architecture = field("architecture").unwrap_or_else(|| "?".into());
    let device = field("device").unwrap_or_else(|| "?".into());
    let description = field("description").unwrap_or_else(|| "?".into());
    // A fallback adapter is software by definition, whatever it calls itself.
    let is_fallback = Reflect::get(&info, &JsValue::from_str("isFallbackAdapter"))
        .ok()
        .and_then(|v| v.as_bool())
        .or_else(|| {
            Reflect::get(&adapter, &JsValue::from_str("isFallbackAdapter"))
                .ok()
                .and_then(|v| v.as_bool())
        })
        .unwrap_or(false);
    let haystack = format!("{vendor} {architecture} {device} {description}").to_lowercase();
    let software =
        is_fallback || SOFTWARE_ADAPTER_MARKERS.iter().any(|m| haystack.contains(m));
    Some(AdapterProbe {
        summary: format!(
            "vendor={vendor} arch={architecture} device={device} desc={description}{}",
            if is_fallback { " FALLBACK" } else { "" }
        ),
        software,
    })
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

        // Say what we got, before anything renders through it, and refuse a CPU
        // rasteriser unless the run explicitly asked for one. An fps measured on
        // SwiftShader describes SwiftShader; publishing it as a browser number is the
        // mistake this check exists to make impossible.
        let info = adapter.get_info();
        let probe = probe_webgpu_adapter().await;
        let software = probe.as_ref().is_some_and(|p| p.software)
            || info.device_type == wgpu::DeviceType::Cpu;
        let summary = format!(
            "adapter: {} | {}{}",
            probe.as_ref().map(|p| p.summary.as_str()).unwrap_or("navigator.gpu unreadable"),
            if info.name.is_empty() { "wgpu name unavailable" } else { &info.name },
            if software { " | SOFTWARE RASTERISER" } else { " | GPU" },
        );
        web_sys::console::log_1(&JsValue::from_str(&summary));
        report.emit("adapter", &summary);
        if software && !allow_software_gpu() {
            return Err(JsValue::from_str(&format!(
                "{summary}\nRefusing to run: this is a CPU rasteriser, not a GPU. A frame rate \
                 measured here describes the software rasteriser and says nothing about the \
                 browser. Give the page a real GPU (a headed window, or headless Chrome with a \
                 GPU), or set VITASLOP_ALLOW_SOFTWARE_GPU=1 to accept a software run."
            )));
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("vitaslop-web-gxm"),
                required_features: wgpu::Features::empty(),
                // Raise the resolution-derived limits to the adapter's: a real title
                // binds textures past the conservative downlevel floor (some titles
                // have a ~2480px atlas). WebGPU guarantees at least 8192, so this is
                // safe.
                //
                // # These MUST be the same limits the native oracle asks for
                // This was `downlevel_webgl2_defaults`, which is not a WebGPU floor at
                // all - it is the set a GL ES 3.0 backend can promise, and it is
                // strictly tighter than the `downlevel_defaults` the native renderer
                // (`vitaslop-native::wgpu_render`, `vitaslop-desktop::retail`) requests
                // over the identical `GxmRenderer` code. Two of the differences bite a
                // real title: `max_vertex_buffer_array_stride` 255 against 2048, and
                // `max_inter_stage_shader_variables` 15 against 16.
                //
                // Asking for less than native does not make the browser safer, it makes
                // it a DIFFERENT renderer - and silently, because a pipeline that
                // exceeds a limit is not a fallback the recompiler reports, it is an
                // invalid pipeline whose render pass draws nothing. A simple scene keeps
                // working, a demanding one goes black, and the fallback count stays zero.
                // The native path can only be the browser's pixel oracle if both devices
                // are built to the same floor.
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| JsValue::from_str(&format!("request_device: {e}")))?;

        // A WebGPU validation error is not an exception - it is reported out of band and
        // the offending object becomes invalid. Without a handler the browser silently
        // draws nothing and reports NOTHING, which is indistinguishable from a scene the
        // guest never submitted. Native's wgpu default panics on this; the browser has to
        // be told. See `vitaslop-platform`'s `report!` macro for the same lesson.
        device.on_uncaptured_error(std::sync::Arc::new(|e| {
            tracing::error!(target: "vitaslop::gxm", "WebGPU uncaptured error: {e}");
        }));

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

        let mut gxm = GxmRenderer::new(&device, &queue, render_format);
        // 2x supersample: resolve the sub-pixel-triangle / coincident-panel speckle a distant 3D
        // vehicle shows, matching the software review shots and the desktop path. The car content
        // is light on fill, so 2x (4x the fragments of a 960x544 frame) stays within a flagship
        // mobile GPU's budget; it is the one knob to turn down if a heavier scene needs it.
        gxm.set_supersample(supersample());
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

    /// Render one freshly-executed FRAME - every scene the guest submitted between
    /// flips, in order - to the canvas through the general renderer.
    ///
    /// # One scene is not a frame
    /// This used to take the frame's LAST scene and draw that. A 3D title's frame is a
    /// chain: offscreen passes render the world and its intermediates, and a final pass
    /// composites them onto the display by SAMPLING them. The last scene IS that
    /// composite, so drawing it alone samples targets nothing ever rendered into - which
    /// on this title produced a live, correct HUD over a black world, with zero fallbacks
    /// and zero WebGPU errors, because nothing had gone wrong: the passes that draw the
    /// world were simply thrown away before the renderer saw them.
    ///
    /// A menu frame is one or two scenes, so the old code was right by accident
    /// everywhere cheap enough to check quickly, and wrong exactly where it mattered.
    /// The native oracle has always gone through `encode_chain`
    /// (`vitaslop-desktop::RetailGfx::present`, `WgpuRenderer::render_frame`); this is
    /// the same call over the same renderer, which is what makes the two comparable.
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.gxm.encode_chain(
            &self.device,
            &self.queue,
            &mut encoder,
            &view,
            &self.depth,
            &built,
            WIDTH,
            HEIGHT,
            CLEAR,
        );
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
        self.fps.tick();
    }
}

/// Size of the EMULATOR's own wasm linear memory, in MB.
///
/// Not the guest's - that is a separate, fixed-size shared memory and never moves. This is
/// the Rust heap: everything the host allocates, which is what a leak on this side grows.
fn wasm_heap_mb() -> usize {
    // 64 KiB pages, so pages/16 is megabytes.
    core::arch::wasm32::memory_size(0) / 16
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
    /// The parsed recipe, kept so the live loop can EVALUATE it - not just replay its
    /// input. Before this the browser shared the recipe's input timeline and none of its
    /// checks (they lived in `vitaslop-native`, which a wasm32 build cannot reach), so a
    /// browser run replayed the same buttons as a passing native run and verified nothing.
    recipe: Option<vitaslop_runtime::recipe::Recipe>,
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

/// Where this run's title comes from. The page says which explicitly - an unrecognised
/// `kind` is an error rather than a guess, because the two differ in how much memory the
/// run needs and a silent fallback to the wrong one is exactly the failure this seam
/// exists to prevent.
enum Source {
    /// Bytes handed straight over from JS. Fine for a small fixture, and the reason a
    /// retail title could not boot: see [`Source::Opfs`].
    Memory(js_sys::Object),
    /// The title stored in OPFS, read in pieces. The product path.
    Opfs(opfs::OpfsReader),
}

impl Source {
    fn from_js(v: JsValue) -> Result<Self, JsValue> {
        let obj: js_sys::Object = v.dyn_into()?;
        let kind = js_sys::Reflect::get(&obj, &JsValue::from_str("kind"))?
            .as_string()
            .unwrap_or_default();
        let payload = js_sys::Reflect::get(&obj, &JsValue::from_str("payload"))?;
        match kind.as_str() {
            "opfs" => Ok(Source::Opfs(opfs::OpfsReader::new(payload)?)),
            "memory" => Ok(Source::Memory(payload.dyn_into()?)),
            other => Err(JsValue::from_str(&format!(
                "game source kind {other:?} is not one of \"opfs\" / \"memory\""
            ))),
        }
    }
}

/// A guest module built elsewhere, plus the two layout numbers the scheduler needs from
/// the transpile that produced it. Small enough to cross a `postMessage` alongside the
/// module itself.
struct Prebuilt {
    module: js_sys::WebAssembly::Module,
    mem_pages: u32,
    mirror_off: Option<u64>,
}

impl Prebuilt {
    /// Read the `{ module, memPages, mirrorOff }` a transpile worker posted back. Every
    /// field is required: a missing one would mean running a module against the wrong
    /// memory layout, which corrupts the guest rather than failing.
    fn from_js(v: &JsValue) -> Result<Option<Self>, JsValue> {
        if v.is_undefined() || v.is_null() {
            return Ok(None);
        }
        let get = |k: &str| js_sys::Reflect::get(v, &JsValue::from_str(k));
        let module = get("module")?
            .dyn_into::<js_sys::WebAssembly::Module>()
            .map_err(|_| JsValue::from_str("prebuilt.module is not a WebAssembly.Module"))?;
        let mem_pages = get("memPages")?
            .as_f64()
            .ok_or_else(|| JsValue::from_str("prebuilt.memPages missing"))? as u32;
        let mirror = get("mirrorOff")?;
        let mirror_off = if mirror.is_null() || mirror.is_undefined() {
            None
        } else {
            Some(mirror.as_f64().ok_or_else(|| JsValue::from_str("bad prebuilt.mirrorOff"))? as u64)
        };
        Ok(Some(Prebuilt { module, mem_pages, mirror_off }))
    }
}

/// What a transpile produced, without the artifact's other baggage.
struct Transpiled {
    wasm: Vec<u8>,
    mem_pages: u32,
    mirror_off: Option<u64>,
    ms: f64,
}

/// Transpile `linked` in THIS worker. Costs ~463 MB of heap that can never be given back
/// (see the note at the call site), so the production path builds it elsewhere.
fn transpile_here(
    linked: &vitaslop_runtime::link::LinkedProgram,
    perf: &web_sys::Performance,
) -> Result<Transpiled, JsValue> {
    // Ask the transpiler for software fuel BEFORE it emits. The browser's WebAssembly
    // engine has no fuel counter of its own, so without this a guest loop that makes no
    // host call runs forever and takes the tab with it - see `browser_sched::preempt_note`.
    // Native does not do this: wasmtime interrupts a thread on real fuel, so its module
    // stays free of the counter entirely.
    vitaslop_transpiler::set_fuel_interval(browser_sched::fuel_interval());
    let t = perf.now();
    let built = vitaslop_transpiler::transpile_lenient(&linked.shared_program());
    let ms = perf.now() - t;
    web_sys::console::log_1(&JsValue::from_str(&format!(
        "[setup] transpiled wasm {} MB, guest memory {} MB, emulator heap {} MB",
        built.artifact.wasm.len() / (1024 * 1024),
        linked.mem_bytes / (1024 * 1024),
        wasm_heap_mb(),
    )));
    Ok(Transpiled {
        wasm: built.artifact.wasm,
        mem_pages: built.artifact.mem_pages,
        mirror_off: built.artifact.mirror_off,
        ms,
    })
}

/// Mount the title, link it, and transpile+compile the guest module - and NOTHING else.
///
/// Runs in a throwaway worker whose whole point is to be terminated afterwards: the
/// transpile's ~463 MB peak cannot be returned to the system by a wasm heap, so the only
/// way to stop paying for it is for the heap it happened in to cease to exist. Returns
/// `{ module, memPages, mirrorOff }` for [`run_game_worker`] to run against.
#[wasm_bindgen]
pub async fn transpile_title(source: JsValue) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    logging::init();
    let perf = global_performance().ok_or_else(|| JsValue::from_str("no performance clock"))?;
    let Mounted { linked, .. } = mount_and_link(source).await?;
    let built = transpile_here(&linked, &perf)?;
    let module = browser_sched::compile_module(&built.wasm).await?;
    let out = js_sys::Object::new();
    js_sys::Reflect::set(&out, &JsValue::from_str("module"), &module)?;
    js_sys::Reflect::set(&out, &JsValue::from_str("memPages"), &JsValue::from_f64(built.mem_pages as f64))?;
    js_sys::Reflect::set(
        &out,
        &JsValue::from_str("mirrorOff"),
        &match built.mirror_off {
            Some(v) => JsValue::from_f64(v as f64),
            None => JsValue::NULL,
        },
    )?;
    Ok(out.into())
}

/// A mounted, linked title: the program plus the guest's files, however they are served.
struct Mounted {
    linked: vitaslop_runtime::link::LinkedProgram,
    /// OPFS-backed files, served on demand. `None` for the in-memory source.
    backing: Option<Box<dyn vitaslop_runtime::host::FileBacking>>,
    /// Decrypted files held in the heap. `None` for the OPFS source.
    resident: Option<vitaslop_runtime::ingest::vfs::MemVfs>,
    nfiles: usize,
    decrypt_ms: f64,
}

/// Mount `source` and link it - everything up to, but not including, the transpile.
///
/// Shared by the run worker and the throwaway transpile worker, which BOTH need the same
/// linked program: one to build the module, one to run it. Linking twice is cheap (24 MB
/// of heap, tens of milliseconds off OPFS) and much cheaper than shipping the linked
/// program between workers.
async fn mount_and_link(source: JsValue) -> Result<Mounted, JsValue> {
    use vitaslop_runtime::ingest::pipeline::{decrypt_container, dump_root, mount_dump_lazy};
    use vitaslop_runtime::ingest::vfs::MemVfs;
    use vitaslop_runtime::link::link;

    let perf = global_performance().ok_or_else(|| JsValue::from_str("no performance clock"))?;
    let t_dec = perf.now();

    // What setup produces either way: the loadable modules, the guest's files (resident
    // or lazily backed), and how many files the title has.
    let mut backing: Option<Box<dyn vitaslop_runtime::host::FileBacking>> = None;
    let mut resident: Option<MemVfs> = None;
    let nfiles;
    let modules_elf;

    match Source::from_js(source)? {
        // The product path. Only the manifest and the loadable modules are read; the data
        // files stay in OPFS and are served a read at a time. That is what makes a
        // gigabyte-plus title fit a wasm32 heap at all - loading one costs more than the
        // whole address space allows once the JS side's copy is counted.
        Source::Opfs(reader) => {
            let vfs = opfs::OpfsVfs::new(reader);
            let root = dump_root(&vfs).ok_or_else(|| {
                JsValue::from_str(
                    "OPFS holds no decrypted dump - import the title (pkg + work.bin) first",
                )
            })?;
            let dump = mount_dump_lazy(&vfs, &root)
                .map_err(|e| JsValue::from_str(&format!("mount dump: {e:?}")))?;
            nfiles = dump.files.len();
            modules_elf = dump.modules;
            // The same open handles back the guest filesystem: a sync access handle takes
            // an exclusive lock, so opening a second set would fail rather than merely
            // cost something.
            // Correctness of this read path is covered by `e2e/opfs.mjs` against a
            // fixture, not re-proven here: verifying a gigabyte-sized title would cost
            // more than the boot it precedes.
            backing =
                Some(Box::new(opfs::OpfsBacking::new(vfs.into_reader(), dump.files_prefix)));
        }

        // Bytes in memory. Kept for fixtures and for a title small enough not to care.
        // Each entry is RELEASED from the JS object as it is copied in, so the container
        // is not resident twice for the whole of setup.
        Source::Memory(obj) => {
            let mut vfs = MemVfs::new();
            let mut n = 0usize;
            for entry in js_sys::Object::entries(&obj).iter() {
                let pair: js_sys::Array = entry.into();
                let path =
                    pair.get(0).as_string().ok_or_else(|| JsValue::from_str("bad file path"))?;
                let bytes = js_sys::Uint8Array::new(&pair.get(1)).to_vec();
                let _ = js_sys::Reflect::delete_property(&obj, &JsValue::from_str(&path));
                vfs.insert(path, bytes);
                n += 1;
            }
            nfiles = n;
            let game = decrypt_container(&mut vfs)
                .map_err(|e| JsValue::from_str(&format!("decrypt: {e:?}")))?;
            // Freed before `link`, not after: link and transpile are the allocation peak.
            drop(vfs);
            modules_elf = game.modules;
            resident = Some(game.files);
        }
    }

    let modules = modules_elf
        .iter()
        .map(|m| loader::load(&m.elf))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| JsValue::from_str(&format!("load module: {e:?}")))?;
    // The module images are linked into one program from here on; the SELF bytes they
    // came from are dead weight during transpile, which is the peak.
    drop(modules_elf);
    let linked = link(modules).map_err(|e| JsValue::from_str(&format!("link: {e:?}")))?;
    let decrypt_ms = perf.now() - t_dec;
    // The heap high-water mark is PERMANENT: wasm linear memory grows and never shrinks,
    // so whatever setup peaks at is carried for the whole run. Sampling either side of
    // each stage is the only way to attribute it - a single number at the end cannot say
    // whether it is the emulator's steady state or a transient the run is still paying for.
    web_sys::console::log_1(&JsValue::from_str(&format!(
        "[setup] heap after decrypt+link: {} MB",
        wasm_heap_mb()
    )));
    Ok(Mounted { linked, backing, resident, nfiles, decrypt_ms })
}

async fn setup_game(
    source: JsValue,
    recipe: &str,
    live: Arc<Mutex<InputState>>,
    prebuilt: Option<Prebuilt>,
    audio_ring: &JsValue,
) -> Result<GameSetup, JsValue> {
    let perf = global_performance().ok_or_else(|| JsValue::from_str("no performance clock"))?;
    let Mounted { linked, backing, resident, nfiles, decrypt_ms } = mount_and_link(source).await?;

    // The input world: a scripted recipe (if given) overlaid with live pointer/keyboard
    // input, both feeding the same touch/pad seam the native probe drives.
    let scripted = !recipe.trim().is_empty();
    // Parsed ONCE, into both halves: the input timeline the world replays and the
    // observations the live loop evaluates. Same text, same parser, same verdict as
    // native - which is the whole point of moving the evaluator into the runtime.
    let parsed_recipe = if scripted {
        Some(
            vitaslop_runtime::recipe::Recipe::parse(recipe)
                .map_err(|e| JsValue::from_str(&format!("recipe: {e}")))?,
        )
    } else {
        None
    };
    let recipe_world = if scripted {
        Some(RecipeWorld::parse(recipe).map_err(|e| JsValue::from_str(&format!("recipe: {e}")))?)
    } else {
        None
    };
    let world = Box::new(BrowserWorld::new(recipe_world, live));
    let mut env = VitaEnv::new(linked.imports.clone(), linked.base, linked.mem_bytes, world);
    env.state.set_alloc_base(linked.alloc_base);
    env.state.set_process_param(linked.process_param);
    env.state.set_modules(linked.loaded_modules.clone());
    env.state.set_tls_template(linked.tls_template);
    env.state.set_preemptive(true);
    // Audio output. The page owns the AudioContext (a main-thread API) and hands us the
    // shared ring its AudioWorklet drains; without one the default `NullSink` stands and
    // the run is silent. Say which, because a silent run has two very different causes -
    // no sink, or a guest producing nothing - and they need opposite investigations.
    match audio::WebAudioSink::new(audio_ring) {
        Some(sink) => {
            env.state.audio = Box::new(sink);
            web_sys::console::log_1(&JsValue::from_str("[audio] shared ring attached"));
        }
        None => web_sys::console::log_1(&JsValue::from_str(
            "[audio] no ring supplied - this run is SILENT (the NullSink discards every grain)",
        )),
    }
    // Give the guest its files. From OPFS that is a backing - nothing is read until the
    // guest asks - and from memory it is a MOVE of the decrypted assets (for a large 3D
    // title that is hundreds of megabytes, and cloning them would double the tightest
    // memory budget we run in).
    if let Some(backing) = backing {
        env.state.set_file_backing(backing);
    }
    if let Some(files) = resident {
        for (path, bytes) in files.into_files() {
            env.state.add_file(&path, bytes);
        }
    }

    // The compiled guest module: either handed in already built, or built here.
    //
    // # Why building it elsewhere matters so much
    // Transpiling a retail title costs ~463 MB of transient heap (measured: 24 MB after
    // link, 487 MB after transpile), and **wasm linear memory never shrinks** - so a run
    // that transpiles in its own worker carries that half-gigabyte for its entire life,
    // on top of the guest's 512 MB and the engine's machine code for the module. The
    // worker was killed part-way through every long run.
    //
    // Built in a throwaway worker instead, the peak dies with that worker and only the
    // compiled `WebAssembly.Module` crosses over (it is structured-cloneable).
    let (module, mem_pages, mirror_off, transpile_ms) = match prebuilt {
        Some(p) => {
            web_sys::console::log_1(&JsValue::from_str(&format!(
                "[setup] using a PREBUILT module (transpiled in a throwaway worker); \
                 emulator heap {} MB",
                wasm_heap_mb()
            )));
            (p.module, p.mem_pages, p.mirror_off, 0.0)
        }
        None => {
            let built = transpile_here(&linked, &perf)?;
            let module = browser_sched::compile_module(&built.wasm).await?;
            (module, built.mem_pages, built.mirror_off, built.ms)
        }
    };

    let main_sp = main_stack_top(linked.base, linked.mem_bytes);
    let sched = browser_sched::BrowserSched::from_linked(
        module,
        &linked.image,
        linked.base,
        mem_pages,
        mirror_off,
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
        recipe: parsed_recipe,
    })
}

/// Verify the OPFS read path against a `syncReader`, over every key under `prefix`.
///
/// Exported so the browser e2e suite can test the real production read path on a small
/// fixture, in seconds, instead of the property being re-proven at every boot of a
/// gigabyte-sized title. What it checks is the difference between whole-file reads (which
/// setup exercises, via the loadable modules) and OFFSET reads (which only the guest
/// exercises): if offsets are wrong, nothing complains - the guest is simply handed the
/// wrong bytes and traps deep in its own code much later, with nothing pointing at
/// storage. That is exactly how it failed the first time.
///
/// Returns a one-line summary on success; an `Err` naming the first disagreeing byte
/// otherwise.
#[wasm_bindgen]
pub fn opfs_verify(reader: JsValue, prefix: &str) -> Result<String, JsValue> {
    let backing = opfs::OpfsBacking::new(opfs::OpfsReader::new(reader)?, prefix);
    backing.verify_all().map_err(|e| JsValue::from_str(&e))
}

/// Set a `VITASLOP_*` knob for this run, from the page.
///
/// The browser has NO environment: on `wasm32-unknown-unknown` `std::env::var` always
/// reports a knob as unset and `std::env::set_var` fails outright, so without this the
/// browser build can only ever run the default configuration. That is not cosmetic - one
/// retail racer needs `VITASLOP_FRAME_TOPUP=0` to finish loading at all, and with the
/// default clock it sits on its loading screen forever, which reads as an emulator bug
/// rather than as an unreachable knob.
///
/// Call this BEFORE `run_game`/`run_game_worker`; the readers latch their values once.
/// A knob whose reader does not consult the override table panics here rather than being
/// silently ignored - see `vitaslop_runtime::knobs::OVERRIDABLE`.
#[wasm_bindgen]
pub fn set_knob(name: &str, value: &str) {
    vitaslop_runtime::knobs::set_override(name, value);
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
    logging::init();

    let live: Arc<Mutex<InputState>> = Arc::new(Mutex::new(InputState::default()));
    // The main thread has no throwaway-worker option (it cannot spawn one and wait
    // synchronously mid-run), so it always transpiles in place.
    let setup = setup_game(files, &recipe, live.clone(), None, &JsValue::UNDEFINED).await?;

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
    wasm_bindgen_futures::spawn_local(live_loop(
        setup.sched,
        playback,
        report,
        max_frames as u64,
        setup.recipe,
    ));
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
    prebuilt: JsValue,
    audio_ring: JsValue,
) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    logging::init();

    let live: Arc<Mutex<InputState>> = Arc::new(Mutex::new(InputState::default()));
    // Register the shared input cell so the page's forwarded pointer/keyboard messages
    // (via the exported worker_input_* functions) reach this run's world.
    input::set_worker_input(live.clone());
    // `prebuilt` is a module a throwaway worker already transpiled and compiled. Passing
    // one keeps the transpile's ~463 MB peak out of THIS worker's heap, which it could
    // never give back - see the note in `setup_game`.
    let setup =
        setup_game(files, &recipe, live, Prebuilt::from_js(&prebuilt)?, &audio_ring).await?;

    let offscreen: web_sys::OffscreenCanvas = offscreen.dyn_into()?;
    offscreen.set_width(WIDTH);
    offscreen.set_height(HEIGHT);
    let report = Report::callback(report_fn);
    let playback =
        LivePlayback::new(wgpu::SurfaceTarget::OffscreenCanvas(offscreen), report.clone()).await?;

    let status = setup.status("web worker");
    web_sys::console::log_1(&JsValue::from_str(&status));
    wasm_bindgen_futures::spawn_local(live_loop(
        setup.sched,
        playback,
        report,
        max_frames as u64,
        setup.recipe,
    ));
    Ok(status)
}

/// Reads guest memory for the shared recipe evaluator, through the scheduler core.
///
/// The browser's half of the `GuestRead` seam - a `SharedArrayBuffer` view where native
/// has a wasmtime store. Everything else about evaluating a recipe is shared.
struct CoreRead<'a>(&'a browser_sched::BrowserSched);

impl vitaslop_runtime::recipe_eval::GuestRead for CoreRead<'_> {
    fn read_into(&self, addr: u32, out: &mut [u8]) -> bool {
        self.0.core.read_guest(addr, out)
    }
}

/// The live run: step the guest one display frame, render it through the general GXM
/// renderer, pace to the display refresh, repeat - until `max_frames` flips or the run
/// ends. This is what makes the browser build *live* (the guest computes each frame on
/// demand and reacts to input) rather than replaying a canned capture. The presented
/// FPS the meter shows is the true combined guest-CPU + render cadence.
///
/// `recipe`, when given, is EVALUATED as well as replayed: its `@watch`/`@assert`/`@sig`
/// go through the same `vitaslop-runtime` evaluator the native runner uses, so a browser
/// run of a recipe reaches the same verdict instead of merely pressing the same buttons.
async fn live_loop(
    mut sched: browser_sched::BrowserSched,
    mut playback: LivePlayback,
    report: Report,
    max_frames: u64,
    recipe: Option<vitaslop_runtime::recipe::Recipe>,
) {
    let mut eval = recipe.as_ref().map(|r| vitaslop_runtime::recipe_eval::RecipeEval::new(r, None));
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

    // How long a fast-forward tick may run before returning to the event loop. Long
    // enough that the fast-forward is CPU-bound rather than paced by the tick rate
    // (`requestAnimationFrame` would otherwise cap it at the display refresh, which is
    // the very cap being escaped), short enough that the page keeps answering - a tab
    // that stops responding while it burns CPU is indistinguishable from a dead one.
    const FF_TICK_BUDGET_MS: f64 = 250.0;
    let ff_to = fastforward_to();
    let mut was_fast = false;
    // Host-call totals at the end of the previous frame, so each frame can report its own.
    let (mut last_hc_calls, mut last_hc_ms) = browser_sched::host_call_totals();

    // How often the live loop repeats its status on the console. Slow by default, because
    // this is a progress heartbeat for a watcher, not a metric.
    //
    // `VITASLOP_BROWSER_HEARTBEAT_MS=0` makes it every frame, which is what a run that
    // DIES at a particular frame needs: the 5-second default reports the last frame before
    // the death, which can be hundreds short of it, and a cause hundreds of frames from
    // where you are looking is not findable.
    let console_status_ms: f64 = vitaslop_runtime::knobs::var("VITASLOP_BROWSER_HEARTBEAT_MS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(5_000.0);

    let now = || perf.as_ref().map(|p| p.now()).unwrap_or(0.0);
    let mut acc = 0.0f64;
    let mut last = now();
    let mut last_console = 0.0f64;

    'run: loop {
        next_tick().await;
        let t = now();
        acc = (acc + (t - last)).min(MAX_CATCHUP_MS);
        last = t;

        // Fast-forward while below the target frame: ignore the wall clock and run as
        // many frames as fit this tick's budget. Crossing the target hands the loop back
        // to real-time pacing and resets the meters, so the published rate describes
        // paced play and never the fast-forward.
        let fast = sched.core.frames() < ff_to;
        let ff_deadline = t + FF_TICK_BUDGET_MS;
        playback.fps.set_paused(fast);
        if fast {
            acc = FRAME_MS;
        } else if was_fast {
            cpu_ms = 0.0;
            cpu_frames = 0;
            render_ms = 0.0;
            presents = 0;
            acc = 0.0;
        }
        was_fast = fast;

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
            // Say what a long frame is DOING while it does it. A frame here can be
            // millions of scheduler rounds, and the status line otherwise reports the
            // last frame FINISHED - so a healthy grind and a hang print the same text for
            // minutes. The rate is also the only direct read on browser guest-CPU speed.
            let report_progress = {
                let report = report.clone();
                let perf = perf.clone();
                move |rounds: u64| {
                    let elapsed =
                        perf.as_ref().map(|p| p.now()).unwrap_or(0.0) - c0;
                    let rate = if elapsed > 0.0 { rounds as f64 * 1000.0 / elapsed } else { 0.0 };
                    let line = format!(
                        "frame {target} in progress: {rounds} scheduler rounds in \
                         {:.0} ms ({rate:.0} rounds/s)",
                        elapsed
                    );
                    report.emit("status", &line);
                    // Under a per-frame heartbeat, say it on the console too. A frame that
                    // never COMPLETES emits no heartbeat at all, so without this the last
                    // thing a watcher sees is the previous frame finishing - and a frame
                    // that dies half way through looks identical to one that never started.
                    if console_status_ms == 0.0 && rounds % 200_000 == 0 {
                        web_sys::console::log_1(&JsValue::from_str(&format!("[live] {line}")));
                    }
                }
            };
            let report_step = browser_sched::run_frames(
                &mut sched.core,
                target,
                PER_FRAME_ROUNDS,
                &mut { report_progress },
            )
            .await;
            let c1 = now();

            // Take the scene presented this frame and drop the rest (render-to-texture
            // intermediates); clearing the per-frame capture vectors bounds the capture's
            // memory during a long run (they exist only for post-mortem debugging).
            // Take the WHOLE frame, not its last scene. See `Playback::present`: a race
            // frame is a chain of offscreen passes feeding a composite, and the composite
            // alone is a HUD over black. Draining is still what bounds memory - each
            // scene holds a snapshot of every draw's vertex window - so nothing is
            // retained across frames here, only within one.
            let frame_scenes = {
                let mut host = sched.host.lock().unwrap();
                let cap = &mut host.state.capture;
                let scenes = core::mem::take(&mut cap.scenes);
                cap.trace.clear();
                cap.trace_thid.clear();
                cap.presents.clear();
                scenes
            };
            if !frame_scenes.is_empty() {
                latest = Some(frame_scenes);
            }

            let frames = sched.core.frames();
            // Evaluate the recipe's observations for this frame. Shots are NAMED here and
            // logged rather than written: a worker has no filesystem, so the picture is
            // the harness's job (`SHOT_EVERY_MS` / the end-of-run capture) while WHICH
            // frames wanted one is the recipe's.
            if let Some(eval) = eval.as_mut() {
                let shots = {
                    let host = sched.host.lock().unwrap();
                    // Read guest memory through the core, not through the locked host -
                    // they are different objects and taking both at once would deadlock.
                    let cap = &host.state.capture;
                    eval.on_frame(frames, &CoreRead(&sched), cap)
                };
                for name in shots {
                    web_sys::console::log_1(&JsValue::from_str(&format!(
                        "[shot] {name} at frame {frames}"
                    )));
                }
            }
            if frames > WARMUP_FRAMES {
                cpu_ms += c1 - c0;
                cpu_frames += 1;
            }
            // The per-frame split, in the one line that is always visible. "This frame
            // took 900 ms" cannot be acted on; "900 ms, of which 40,000 host calls took
            // 870" names the half to fix and rules out the other.
            let (hc_calls, hc_ms) = browser_sched::host_call_totals();
            let frame_calls = hc_calls - last_hc_calls;
            let frame_hc_ms = hc_ms - last_hc_ms;
            last_hc_calls = hc_calls;
            last_hc_ms = hc_ms;
            let status = format!(
                "frame {frames}{} (live via WebGPU) | {:.0} ms, {frame_calls} host calls \
                 ({frame_hc_ms:.0} ms) | {report_step:?}",
                if fast { format!(" fast-forwarding to {ff_to}") } else { String::new() },
                c1 - c0,
            );
            report.emit("status", &status);
            // Also say it on the CONSOLE, slowly.
            //
            // The status element is the right place for a human watching the page, and
            // the wrong place for anything else to read: it lives on the MAIN thread, and
            // a harness asking for its text needs that thread to answer. During a long
            // fast-forward it does not answer reliably, so a run that was perfectly
            // healthy reported as "could not read #status" for as long as it lasted.
            // A console line is pushed, not polled, and reaches a watcher whatever the
            // main thread is doing.
            let t_now = now();
            if t_now - last_console >= console_status_ms {
                last_console = t_now;
                // With the emulator's OWN heap size. A worker that grows until it is
                // killed says nothing about WHERE it grew, and the two answers need
                // opposite fixes: a climbing wasm heap is a Rust-side leak, a flat one
                // with a climbing process is GPU or JS. Chrome's per-process number
                // cannot separate them and the kill leaves no other evidence.
                // With the emulator's own heap size AND the JSPI stack accounting.
                //
                // A worker that grows until it is killed says nothing about WHERE it grew,
                // and the answers need opposite fixes. The wasm heap separates a Rust-side
                // leak from a JS/engine-side one. The stack counters separate the JS-side
                // case further: a JSPI stack is the biggest single allocation this
                // scheduler makes, an ABANDONED one can never be reclaimed, and a
                // per-frame process growth divided by a per-frame count is the per-item
                // cost that names the culprit. Chrome's per-process number cannot separate
                // any of these and the kill leaves no other evidence.
                let (susp, starts, abandoned, released) = browser_sched::stack_stats();
                let (live_threads, finished_threads) = sched.core.thread_census();
                // The GAME CLOCK and the quanta that advanced it.
                //
                // The single most important cross-engine number, and it had no readout at
                // all. With `VITASLOP_FRAME_TOPUP=0` the game clock advances ONLY through
                // `charge_cpu_quantum`, once per scheduler quantum - so the clock rate is
                // quanta-per-frame times a constant, and a quantum is only comparable
                // across engines if both count the same unit of guest work. Both now
                // count executed wasm instructions (native through wasmtime's fuel, the
                // browser through `emit_block_charge`), but when the browser counted HOST
                // CALLS instead its clock ran 5.1x slow. A title waiting a fixed number of
                // SECONDS then waits a different number of FRAMES on each engine, and
                // nothing about that is visible from the frame counter, which is the only
                // thing both engines were reporting.
                let (quanta, flips, clock_us, clk_q, clk_idle) = {
                    let host = sched.host.lock().unwrap();
                    let (q, f) = host.state.quantum_flip_counts();
                    let (from_q, _from_topup, from_idle) = host.state.clock_sources();
                    (q, f, host.state.now_us(), from_q, from_idle)
                };
                let (preempts, on_fuel) = browser_sched::preemption_stats();
                web_sys::console::log_1(&JsValue::from_str(&format!(
                    "[live] {status} | clock {:.2}s over {flips} flips ({quanta} quanta, \
                     {:.1} us/frame; {:.2}s quanta + {:.2}s idle) \
                     | preempt {preempts} ({on_fuel} on fuel) | wasm heap {} MB \
                     | jspi {susp} susp, {starts} stacks, {abandoned} abandoned, \
                     {released} released | threads {live_threads} live, {finished_threads} finished",
                    clock_us as f64 / 1e6,
                    if flips > 0 { clock_us as f64 / flips as f64 } else { 0.0 },
                    clk_q as f64 / 1e6,
                    clk_idle as f64 / 1e6,
                    wasm_heap_mb()
                )));
            }
            // Keep spending this tick's budget while the fast-forward target is still
            // ahead; otherwise fall out and let the wall clock pace the next frame.
            if fast && now() < ff_deadline && sched.core.frames() < ff_to {
                acc = FRAME_MS;
            }
            match report_step {
                // Reached the next flip and still within budget: keep going.
                RunReport::FramesReached(n) if n < max_frames => {}
                // Hit the frame budget, or the run finished / deadlocked / trapped.
                _ => {
                    // The last word on the run, past the rate limit: the frame it ended
                    // at and why. A limiter that dropped THIS would leave the page
                    // showing a stale mid-run line forever, which is indistinguishable
                    // from a hang.
                    report.emit_final(
                        "status",
                        &format!("frame {frames} ENDED (live via WebGPU) | {report_step:?}"),
                    );
                    web_sys::console::log_1(&JsValue::from_str(&format!(
                        "live run ended at frame {frames}: {report_step:?}"
                    )));
                    // The recipe's VERDICT, on the console, at the end of the run.
                    //
                    // Without it a browser run that replayed a recipe reported only where
                    // it stopped, which says nothing about whether the title did what the
                    // recipe says it should - and assertions past the frame reached count
                    // as failures, so a run that stalled short cannot read as a pass.
                    if let Some(eval) = eval.as_mut() {
                        let sig = sched.host.lock().unwrap().state.capture.signature();
                        eval.finish(frames, sig);
                        web_sys::console::log_1(&JsValue::from_str(&format!(
                            "[recipe] {} | sig {sig:#018x}",
                            eval.summary()
                        )));
                        for a in eval.asserts.iter().filter(|a| !a.passed) {
                            web_sys::console::log_1(&JsValue::from_str(&format!(
                                "[recipe] FAIL f{} {} - {}",
                                a.frame, a.desc, a.detail
                            )));
                        }
                    }
                    if let Some(scene) = &latest {
                        playback.present(scene);
                    }
                    break 'run;
                }
            }
        }

        // Present at most one (the newest) frame per tick, and fold its render time into
        // the rolling perf report.
        //
        // NOT while fast-forwarding: nobody is watching a fast-forward, and every present
        // is a full GXM->WebGPU encode of a scene that is discarded a moment later. It is
        // pure cost on the one path whose entire purpose is to reach a later frame
        // quickly. Real-time pacing resumes presenting the moment the target is crossed,
        // so the screenshot and the published rate are unaffected.
        if fast {
            latest = None;
        }
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
                let perf_line = format!(
                    "cpu {cpu_avg:.1} ms/frame ({cpu_fps:.0} fps uncapped) | render {render_avg:.1} ms"
                );
                report.emit("perf", &perf_line);
                // Also on the console, with the frame it describes.
                //
                // The perf element holds only the LATEST window, so a run's rate can only
                // ever be read at the instant someone looks. A title's cost varies by an
                // order of magnitude between a menu and a race, so "the browser runs at N
                // fps" is meaningless without saying which frame N was measured over -
                // and a single end-of-run reading silently answers for whatever screen
                // the run happened to stop on.
                web_sys::console::log_1(&JsValue::from_str(&format!(
                    "[perf] frame {} | {perf_line}",
                    sched.core.frames()
                )));
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
