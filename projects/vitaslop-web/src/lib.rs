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
mod location;
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
        artifact.dirty_off,
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
    /// Guest display flips retired in this window, which is a DIFFERENT rate from the
    /// presented one whenever the loop runs more than one frame per present. See
    /// [`FpsMeter::note_guest_frames`].
    window_guest_frames: u32,
    /// Last published guest-flip rate, i.e. the emulated speed's numerator.
    last_guest_fps: f64,
    /// The emulated game clock, in microseconds, at the start of this window and at the
    /// last frame - the numerator of the SPEED percentage. See [`FpsMeter::note_clock`].
    window_clock_us: u64,
    last_clock_us: u64,
    /// While set, the meter publishes what it is doing INSTEAD of a rate. A fast-forward
    /// presents once per event-loop tick rather than once per guest frame, so the rate it
    /// would compute describes the tick cadence and nothing about how fast the emulator
    /// runs. Publishing that number would be worse than publishing none.
    paused: bool,
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
        FpsMeter {
            perf,
            report,
            window_start: now,
            window_frames: 0,
            last_fps: 0.0,
            window_guest_frames: 0,
            last_guest_fps: 0.0,
            window_clock_us: 0,
            last_clock_us: 0,
            paused: false,
        }
    }

    /// Record `n` retired guest display flips. Called by the live loop for every frame it
    /// runs, presented or not.
    ///
    /// # Why the presented rate is not the emulated speed
    /// The percentage this meter publishes used to be `presents / 60`, and that is only the
    /// emulated speed when the loop presents every frame it runs. It does not: it presents
    /// at most the newest scene per tick, so a tick that ran two frames showed one. A device
    /// capture read `fps 37 (61% speed)` alongside `1.5 guest frames per present` - the
    /// guest was retiring 56 flips a second, i.e. running at 93% of console speed, and the
    /// headline number understated it by the exact frames-per-present ratio.
    ///
    /// That is not a cosmetic error. It says the emulator is CPU-bound when the real defect
    /// is that it is discarding finished pictures, and those two have nothing in common: one
    /// is a month of guest-CPU work and the other is the pacing policy twenty lines below.
    /// Both numbers are published now, because both are real and they answer different
    /// questions - "is the world running at the right speed" and "how much of it can I see".
    fn note_guest_frames(&mut self, n: u32) {
        self.window_guest_frames += n;
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
        self.window_guest_frames = 0;
        self.last_guest_fps = 0.0;
        self.window_clock_us = self.last_clock_us;
    }

    /// Record the emulated game clock. See [`FpsMeter::note_clock`] for why the speed is
    /// computed from this and not from the flip rate.
    fn note_clock(&mut self, clock_us: u64) {
        if self.window_clock_us == 0 {
            self.window_clock_us = clock_us;
        }
        self.last_clock_us = clock_us;
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
            self.last_guest_fps = self.window_guest_frames as f64 * 1000.0 / dt;
            // >>> SPEED IS EMULATED SECONDS PER REAL SECOND, NOT FLIPS OVER 60.
            //
            // The old percentage was `guest flips / 60`, which assumes every title presents
            // sixty times a second. A 30 fps title does not, ON HARDWARE, so that reading
            // called a perfectly-paced run "50% speed" - and it is the headline number a
            // device capture is read from. MEASURED on one retail title's round in a browser:
            // 31 flips a second, `50% speed` by the old rule, while its emulated clock ran at
            // 1.01x real time - i.e. exactly console speed.
            //
            // The clock IS the guest's own experience of time (its timers, its animation
            // rates and its audio are all billed in it), so its rate against the wall is what
            // "how fast is this running" means, at any frame rate. The flip rate is still
            // published beside it: that is what a player SEES, and the two answer different
            // questions.
            let speed = (self.last_clock_us.saturating_sub(self.window_clock_us)) as f64
                / (dt * 1000.0)
                * 100.0;
            let text = if self.last_guest_fps > self.last_fps + 0.5 {
                format!(
                    "fps: {:.0} shown of {:.0} run ({speed:.0}% speed - {:.0}% of the frames                      the emulator computed were DISCARDED unpresented)",
                    self.last_fps,
                    self.last_guest_fps,
                    100.0 * (1.0 - self.last_fps / self.last_guest_fps),
                )
            } else {
                format!("fps: {:.0} ({speed:.0}% speed)", self.last_fps)
            };
            self.report.emit("fps", &text);
            self.window_start = now;
            self.window_frames = 0;
            self.window_guest_frames = 0;
            self.window_clock_us = self.last_clock_us;
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
                required_features: vitaslop_platform::gpu::wanted_features(&adapter),
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
    /// The surface description, kept so the diagnostics panel can carry it (see the note
    /// where it is built).
    surface_line: String,
    /// The surface stores blue first, so the probe's per-channel means need swapping.
    probe_bgra: bool,
    fps: FpsMeter,
    /// The clock `present` times itself with. `FpsMeter` owns one too, but it is behind
    /// that type's own accounting; the render split needs four reads of its own.
    perf: Option<web_sys::Performance>,
    split: RenderSplit,
    /// Reads back WHAT WE PRESENTED, when `VITASLOP_PRESENT_PROBE` asks for it.
    probe: Option<PresentProbe>,
    /// ...and how bright each offscreen target of the chain is, on the same cadence.
    targets: Option<TargetProbe>,
    /// The most recent probe description, waiting for the next diagnostics window.
    last_probe: Option<String>,
    /// Presents since the run started. NOT `split.presents`, which `take_split` resets every
    /// diagnostics window - a probe cadence driven off that one restarts at zero each window,
    /// so it fires on the same relative frame forever and every report is labelled "frame 0".
    presents_total: u64,
    /// Set by the device-lost callback installed in [`LivePlayback::new`]. `Some` means every
    /// GPU object this renderer holds is invalid and the run is over - see that callback.
    lost: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    /// The surface's configuration, kept so a recoverable acquire failure can apply it again.
    surface_config: wgpu::SurfaceConfiguration,
    /// Consecutive presents that produced no surface texture. A handful is ordinary (a
    /// reconfigure takes effect on the next frame, a tab is occluded); a run of them is a
    /// swapchain that is never coming back, and continuing past it is the black screen this
    /// whole mechanism exists to refuse.
    acquire_failures: u32,
    /// Whether the surface was occluded at the last acquire, so the report fires on the EDGE.
    occluded: bool,
}

/// How far a call to [`LivePlayback::present`] got.
///
/// >>> IT IS A RETURN VALUE BECAUSE SKIPPING A FRAME AND LOSING THE DEVICE USED TO BE THE
/// >>> SAME LINE OF CODE. `get_current_texture` was matched as "Success or Suboptimal, else
/// `return`", which is right for a timeout and catastrophic for the other three: `Outdated`
/// needs the surface configured again (it never recovers on its own, so the canvas stays
/// black for the rest of the run), `Lost` needs the same or is fatal, and `Validation` is a
/// bug in this renderer that nothing would ever have printed.
#[derive(Debug, PartialEq, Eq)]
enum PresentOutcome {
    /// A frame was encoded, submitted and presented.
    Presented,
    /// No surface texture this frame, for a reason that legitimately passes: the tab is
    /// occluded, the acquire timed out, or the surface was just reconfigured. Counted.
    Skipped,
    /// The renderer cannot draw again. The run must stop and say this.
    Fatal(String),
}

/// Sample every OFFSCREEN TARGET of a frame and describe how bright each one is.
///
/// # Why the presented surface is not enough
/// A frame is a chain: the world and its intermediates render into offscreen targets and a
/// final pass composites them. When the finished picture is wrong, the surface says only that
/// - and every pass in the chain produces the same symptom in the composite. The native oracle
/// answers this with `VITASLOP_GPU_CHAIN_DIR`, which writes a PNG per target; the browser has
/// no filesystem, and the defect being chased here (a world that goes pure white while the UI
/// survives) reproduces in the BROWSER and not, so far, anywhere a PNG can be written.
///
/// The other route was tried first and cannot work: `VITASLOP_GXP_SOLID` paints every
/// recompiled draw magenta, but the frame's last pass is a fullscreen composite that is itself
/// recompiled, so the whole screen comes back magenta and names nothing. That is recorded in
/// the notes as a trap and it caught this session anyway.
///
/// Only a small corner of each target is copied - brightness is the question, not the image -
/// so the whole probe is a few tens of kilobytes however many targets a frame has.
struct TargetProbe {
    /// One staging buffer per target address, kept across frames.
    bufs: std::collections::HashMap<u32, (wgpu::Buffer, std::sync::Arc<std::sync::atomic::AtomicBool>)>,
    /// The targets copied on the frame in flight: `(address, width, height)` of the REGION.
    pending: Vec<(u32, u32, u32)>,
    awaiting_submit: bool,
    frame: u64,
}

/// The edge of each sampled tile. 64 * 4 bytes is exactly the 256-byte row alignment WebGPU
/// requires for a texture-to-buffer copy, so no padding arithmetic is needed.
const TARGET_PROBE_EDGE: u32 = 64;

/// How many tiles are sampled per target, spread across its INTERIOR.
///
/// # Why one corner is not a sample
/// The first version copied a single 64x64 tile from each target's top-left, and it named the
/// 1024x1024 target as pure white - which it is, in that corner, on BOTH engines: it is a
/// shadow map and its corner is empty far-plane. Whole-image, the same target is 126.5 mean /
/// 49.6% white. A corner reading would have identified an innocent pass as the cause of the
/// white-out, confidently, with a number attached.
///
/// Four tiles at the quarter points sample where a render target actually has content. It is
/// still a sample and not the image - a target could be white only between the tiles - but it
/// cannot be fooled by one empty margin, which is the failure that actually happened.
const TARGET_PROBE_TILES: u32 = 4;

impl TargetProbe {
    fn new() -> TargetProbe {
        TargetProbe {
            bufs: std::collections::HashMap::new(),
            pending: Vec::new(),
            awaiting_submit: false,
            frame: 0,
        }
    }

    /// Copy a corner of every target. Call before the submit that carries `encoder`.
    fn capture(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        targets: &[(u32, &wgpu::Texture, u32, u32)],
        frame: u64,
    ) {
        self.pending.clear();
        self.frame = frame;
        let row = TARGET_PROBE_EDGE * 4;
        let tile_bytes = (row as u64) * (TARGET_PROBE_EDGE as u64);
        for &(addr, tex, w, h) in targets {
            let cw = w.min(TARGET_PROBE_EDGE);
            let ch = h.min(TARGET_PROBE_EDGE);
            if cw == 0 || ch == 0 {
                continue;
            }
            let entry = self.bufs.entry(addr).or_insert_with(|| {
                (
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("target-probe"),
                        size: tile_bytes * TARGET_PROBE_TILES as u64,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }),
                    std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                )
            });
            entry.1.store(false, std::sync::atomic::Ordering::Relaxed);
            // Four tiles at the quarter points, clamped so each one fits inside the target.
            // The tile offsets are whole multiples of `tile_bytes`, itself a multiple of the
            // 256-byte copy alignment, so no offset arithmetic can violate it.
            let origins = [(w / 4, h / 4), (3 * w / 4, h / 4), (w / 4, 3 * h / 4), (3 * w / 4, 3 * h / 4)];
            for (i, (ox, oy)) in origins.iter().enumerate() {
                let x = (*ox).min(w.saturating_sub(cw));
                let y = (*oy).min(h.saturating_sub(ch));
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x, y, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &entry.0,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: tile_bytes * i as u64,
                            bytes_per_row: Some(row),
                            rows_per_image: Some(TARGET_PROBE_EDGE),
                        },
                    },
                    wgpu::Extent3d { width: cw, height: ch, depth_or_array_layers: 1 },
                );
            }
            self.pending.push((addr, cw, ch));
        }
        self.awaiting_submit = !self.pending.is_empty();
    }

    /// Request the maps, AFTER the submit. See `PresentProbe::begin_map` for why this cannot
    /// be folded into `capture`.
    fn begin_map(&mut self) {
        if !self.awaiting_submit {
            return;
        }
        self.awaiting_submit = false;
        for (addr, _, _) in &self.pending {
            if let Some((buf, ready)) = self.bufs.get(addr) {
                let ready = ready.clone();
                buf.slice(..).map_async(wgpu::MapMode::Read, move |_| {
                    ready.store(true, std::sync::atomic::Ordering::Relaxed);
                });
            }
        }
    }

    /// Describe every target whose copy has landed, brightest first.
    fn take_report(&mut self, swizzle_bgra: bool) -> Option<String> {
        if self.pending.is_empty() || self.awaiting_submit {
            return None;
        }
        let all_ready = self.pending.iter().all(|(a, _, _)| {
            self.bufs
                .get(a)
                .is_some_and(|(_, r)| r.load(std::sync::atomic::Ordering::Relaxed))
        });
        if !all_ready {
            return None;
        }
        let mut rows: Vec<(f64, String)> = Vec::new();
        for (addr, cw, ch) in std::mem::take(&mut self.pending) {
            let Some((buf, _)) = self.bufs.get(&addr) else { continue };
            let (mut sum, mut white, mut n) = (0u64, 0u64, 0u64);
            if let Ok(view) = buf.slice(..).get_mapped_range() {
                let tile = (TARGET_PROBE_EDGE as usize) * 4 * (TARGET_PROBE_EDGE as usize);
                for t in 0..TARGET_PROBE_TILES as usize {
                    for y in 0..ch as usize {
                        let base = t * tile + y * (TARGET_PROBE_EDGE as usize) * 4;
                        for x in 0..cw as usize {
                            let p = &view[base + x * 4..base + x * 4 + 4];
                            let (r, g, b) =
                                if swizzle_bgra { (p[2], p[1], p[0]) } else { (p[0], p[1], p[2]) };
                            sum += ((r as u32 * 54 + g as u32 * 183 + b as u32 * 19) >> 8) as u64;
                            if r > 250 && g > 250 && b > 250 {
                                white += 1;
                            }
                            n += 1;
                        }
                    }
                }
            }
            buf.unmap();
            let mean = sum as f64 / n.max(1) as f64;
            rows.push((
                mean,
                format!(
                    "  {addr:#x}  mean {mean:6.1}  white {:5.1}%",
                    white as f64 * 100.0 / n.max(1) as f64
                ),
            ));
        }
        if rows.is_empty() {
            return None;
        }
        rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = format!(
            "frame {}: {TARGET_PROBE_TILES} interior {TARGET_PROBE_EDGE}x{TARGET_PROBE_EDGE} \
             tiles of every offscreen target, brightest first. A target reading ~255 mean / \
             ~100% white is where the white ENTERS the chain; the ones below it are downstream. \
             NOTE this is a sample, not the image - compare a suspect against the SAME tiles of \
             a native `VITASLOP_GPU_CHAIN_DIR` dump before believing it.\n",
            self.frame
        );
        for (_, line) in rows {
            out.push_str(&line);
            out.push('\n');
        }
        Some(out)
    }
}

/// Sample the presented surface and describe it in TEXT.
///
/// # Why a render counter cannot answer "the screen is white"
/// A white screen with a healthy panel - draws recompiled, nothing dropped, no WebGPU error,
/// textures cached, and an fps meter that ticks AFTER `queue.present` - is consistent with two
/// completely different faults: we presented white pixels, or we presented a picture the
/// compositor never showed. Every counter in this file is upstream of the surface, so none of
/// them can tell those apart, and on a phone there is no screenshot tool, no devtools and no
/// pixel to sample by hand. This reads the surface itself, after the frame is encoded, and
/// prints a summary a person can read out of the diagnostics panel.
///
/// It costs a full-surface copy and a buffer map, so it runs once every `every` presents and
/// only when the knob asks. `VITASLOP_PRESENT_PROBE=120` is a reasonable cadence: twice a
/// second at 60 fps, and the copy is 2 MB.
struct PresentProbe {
    /// Sample every N presents. Never zero.
    every: u32,
    /// The staging buffer the surface is copied into (`WIDTH * HEIGHT * 4`).
    buffer: wgpu::Buffer,
    /// `bytes_per_row` for the copy - the surface is `WIDTH` wide and 256-aligned already,
    /// but a wrong assumption here is a validation error rather than a wrong picture, so it
    /// is computed and asserted at construction.
    bytes_per_row: u32,
    /// Set by the map callback when a mapped read is ready. Shared with the callback, which
    /// wgpu requires to be `'static` and thread-safe even on wasm.
    ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// A copy is in flight (mapped or awaiting its callback), so no new one may start.
    in_flight: bool,
    /// The copy is encoded but its submit has not happened yet, so the map may not be
    /// requested. See [`PresentProbe::begin_map`].
    awaiting_submit: bool,
    /// The frame the in-flight copy was taken on, for the report.
    in_flight_frame: u64,
}

impl PresentProbe {
    fn new(device: &wgpu::Device, every: u32) -> PresentProbe {
        // WebGPU requires a 256-byte aligned `bytes_per_row` for a texture-to-buffer copy.
        // 960 * 4 = 3840 = 15 * 256, so the surface needs no padding - but the ALIGNMENT is
        // the rule, not the coincidence, so it is rounded explicitly and the buffer sized
        // from the rounded value. A hard-coded 3840 would silently corrupt every row the day
        // the surface stops being 960 wide.
        let bytes_per_row = (WIDTH * 4).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present-probe"),
            size: (bytes_per_row as u64) * (HEIGHT as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        PresentProbe {
            every: every.max(1),
            buffer,
            bytes_per_row,
            ready: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            in_flight: false,
            awaiting_submit: false,
            in_flight_frame: 0,
        }
    }

    /// Should this present be sampled?
    fn wants(&self, presents: u64) -> bool {
        !self.in_flight && presents > 0 && presents % (self.every as u64) == 0
    }

    /// Queue the copy. Call with the encoder that is about to be submitted, BEFORE
    /// `present` - the surface texture is not readable once presented.
    fn capture(&mut self, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture, frame: u64) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row),
                    rows_per_image: Some(HEIGHT),
                },
            },
            wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        );
        self.in_flight = true;
        self.awaiting_submit = true;
        self.in_flight_frame = frame;
        self.ready.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Request the map, AFTER the encoder carrying the copy has been submitted.
    ///
    /// # Why this cannot be folded into `capture`
    /// It was, and the probe then reported a 100% pure black surface on a frame that was
    /// demonstrably rendering. `map_async` resolves against the buffer as the queue knows it
    /// at the moment of the call; asking before the copy is submitted returns the buffer's
    /// prior contents, which for a never-written staging buffer is zeros. The failure is
    /// perfectly quiet - a valid map, a full buffer, every byte zero - and it reads exactly
    /// like the defect the probe was built to diagnose. An instrument whose failure mode
    /// imitates its subject has to be ordered correctly by construction, so the submit
    /// boundary is now a separate call that cannot be skipped without leaving `awaiting_submit`
    /// set and the probe visibly stuck.
    fn begin_map(&mut self) {
        if !self.awaiting_submit {
            return;
        }
        self.awaiting_submit = false;
        let ready = self.ready.clone();
        self.buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            // A failed map must not leave the probe stuck in flight forever; it is flagged
            // ready and the reader reports the failure rather than silently sampling nothing.
            let _ = r;
            ready.store(true, std::sync::atomic::Ordering::Relaxed);
        });
    }

    /// If a mapped read is ready, describe it and release the buffer. Returns the report.
    fn take_report(&mut self, swizzle_bgra: bool) -> Option<String> {
        if !self.in_flight || !self.ready.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        // A map that FAILED still flags ready (see the callback), so the failure has to be
        // reported here rather than read as an empty sample. Silence would look exactly like
        // a probe that ran and found nothing worth saying.
        let text = match self.buffer.slice(..).get_mapped_range() {
            Ok(view) => {
                let text =
                    Self::describe(&view, self.bytes_per_row, self.in_flight_frame, swizzle_bgra);
                drop(view);
                self.buffer.unmap();
                text
            }
            Err(e) => format!(
                "presented frame {}: the surface readback FAILED to map ({e}), so this window \
                 says nothing about what was presented",
                self.in_flight_frame
            ),
        };
        self.in_flight = false;
        Some(text)
    }

    /// Turn the presented pixels into something readable on a phone.
    ///
    /// Three things, because each answers a different question the others cannot:
    /// the WHITE and BLACK shares (is the surface uniform?), the channel means and extremes
    /// (is it washed out, or clipped?), and an 8x6 luminance grid (does the picture have
    /// STRUCTURE?). The grid is the one that settles "white screen": a uniform grid means we
    /// presented a blank surface, and a varied one means we presented a picture and the
    /// screen is not showing it.
    fn describe(bytes: &[u8], bytes_per_row: u32, frame: u64, swizzle_bgra: bool) -> String {
        const GRID_W: usize = 8;
        const GRID_H: usize = 6;
        let (mut white, mut black, mut total) = (0u64, 0u64, 0u64);
        let (mut sum_r, mut sum_g, mut sum_b) = (0u64, 0u64, 0u64);
        let (mut min_l, mut max_l) = (255u8, 0u8);
        let mut cells = [[0u64; GRID_W]; GRID_H];
        let mut cell_n = [[0u64; GRID_W]; GRID_H];
        for y in 0..HEIGHT as usize {
            let row = &bytes[y * bytes_per_row as usize..][..WIDTH as usize * 4];
            for x in 0..WIDTH as usize {
                let p = &row[x * 4..x * 4 + 4];
                // The surface format may be BGRA; the channel order only matters for the
                // per-channel means, never for luminance or the white/black shares.
                let (r, g, b) = if swizzle_bgra { (p[2], p[1], p[0]) } else { (p[0], p[1], p[2]) };
                total += 1;
                sum_r += r as u64;
                sum_g += g as u64;
                sum_b += b as u64;
                if r > 250 && g > 250 && b > 250 {
                    white += 1;
                }
                if r < 5 && g < 5 && b < 5 {
                    black += 1;
                }
                let l = ((r as u32 * 54 + g as u32 * 183 + b as u32 * 19) >> 8) as u8;
                min_l = min_l.min(l);
                max_l = max_l.max(l);
                let cy = y * GRID_H / HEIGHT as usize;
                let cx = x * GRID_W / WIDTH as usize;
                cells[cy][cx] += l as u64;
                cell_n[cy][cx] += 1;
            }
        }
        let pct = |n: u64| n as f64 * 100.0 / total.max(1) as f64;
        let mut out = format!(
            "presented frame {frame}: {:.1}% pure white, {:.1}% pure black, \
             mean rgb ({:.0},{:.0},{:.0}), luminance {min_l}..{max_l}\n",
            pct(white),
            pct(black),
            sum_r as f64 / total.max(1) as f64,
            sum_g as f64 / total.max(1) as f64,
            sum_b as f64 / total.max(1) as f64,
        );
        // A uniform surface and a picture are told apart by this and nothing else above it.
        out.push_str(if max_l - min_l < 8 {
            "the surface is UNIFORM - we presented a blank picture, so the fault is UPSTREAM \
             of the canvas\n"
        } else {
            "the surface HAS STRUCTURE - we presented a picture, so a blank SCREEN is a \
             compositing fault, not a render one\n"
        });
        for row in cells.iter().zip(cell_n.iter()) {
            out.push(' ');
            for (sum, n) in row.0.iter().zip(row.1.iter()) {
                let mean = (sum / (*n).max(1)) as u8;
                // A coarse ramp: the SHAPE is the reading, not the exact level.
                out.push(match mean {
                    0..=31 => '.',
                    32..=63 => ':',
                    64..=95 => '-',
                    96..=127 => '=',
                    128..=159 => '+',
                    160..=191 => '*',
                    192..=223 => '#',
                    _ => '@',
                });
                out.push(' ');
            }
            out.push('\n');
        }
        out
    }
}

/// Where a presented frame's render time went. See [`LivePlayback::present`].
#[derive(Default)]
struct RenderSplit {
    build_ms: f64,
    encode_ms: f64,
    /// `encode_ms` broken down by `encode_chain`'s own phases, summed over the frame.
    prepare_ms: f64,
    upload_ms: f64,
    /// `upload_ms` split into its two halves - see `EncodePhases::arena_ms`. The combined
    /// number named no fix: a course-load frame read 2,642 ms of "upload" while uploading zero
    /// textures and zero bytes.
    arena_ms: f64,
    arena_create_ms: f64,
    arena_write_ms: f64,
    ubo_bg_ms: f64,
    /// The parts of `encode_chain` that are not a pass - see `gpu::EncodePhases`. Summed over
    /// the window like every other phase here.
    precompile_ms: f64,
    retire_ms: f64,
    resident_ms: f64,
    pass_ms: f64,
    gxp_draws: u64,
    fixed_draws: u64,
    submit_ms: f64,
    scenes: u64,
    draws: u64,
    presents: u64,
    /// The WORST single present of the window, and what it did.
    ///
    /// A window mean cannot answer this title's question. Its frames range from 276 to 714
    /// draws depending on what is on screen, so a window averaging 509 draws can be mostly
    /// cheap frames plus two catastrophic ones - and the mean then reports a per-draw cost
    /// that no frame in the window actually paid. The worst frame is the one to explain.
    worst_build_ms: f64,
    worst_draws: usize,
    worst_work: vitaslop_runtime::render::BuildWork,
    /// The same, for the worst present by ENCODE - which is not always the same frame as the
    /// worst by build, and encode is the larger half now.
    worst_encode_ms: f64,
    worst_encode_draws: usize,
    worst_enc_work: vitaslop_platform::gpu::EncodeWork,
    /// The worst encode frame's OWN prepare/upload/pass split.
    ///
    /// # Why the counters were not enough
    /// A device capture produced a worst encode frame of 32.5 ms against a 10.8 ms window
    /// mean, at 474 draws against 473 - and with every counter in `worst_enc_work` identical
    /// to the mean's, down to the buffer count and the bytes written. Identical work, three
    /// times the time. The counters say what was DONE and cannot say where the time went, and
    /// the phase split was reported for the window mean only, so the one frame that needed
    /// explaining was the one frame with no breakdown. `prepare` and `upload` have completely
    /// different causes (bind-group construction against buffer writes and allocation), so
    /// without this the outlier cannot even be attributed to a half.
    worst_enc_phases: vitaslop_platform::gpu::EncodePhases,
    /// The window's build work, summed here rather than read globally, so it covers exactly
    /// the presents this window counted.
    work: vitaslop_runtime::render::BuildWork,
    /// What `encode_chain` DID over this window - see `vitaslop_platform::gpu::EncodeWork`.
    enc_work: vitaslop_platform::gpu::EncodeWork,
    /// >>> WHERE `prepare` WENT, ON THE ENGINE THAT PAYS FOR IT.
    ///
    /// `VITASLOP_PREPARE_SPLIT` has always fed these counters on both engines, and only the
    /// DESKTOP ever printed them - so the browser reported `encode 4.7 ms (prepare 3.6 ...)`
    /// and nothing inside a phase whose candidates want opposite fixes (hashing a vertex
    /// stream to key a cache, copying into a pass arena, building a bind group). The desktop's
    /// answer does not transfer: after resident geometry its `prepare` is 0.6 ms while the
    /// browser's is several, which is precisely a browser-only cost that the one engine
    /// printing the split cannot see. Same defect class as the phase timers before
    /// `perf::set_clock` existed.
    ///
    /// Empty and free unless the knob is on - the counters are clock reads on a path that
    /// makes no WebGPU call, which is why they are the one instrument here that is asked for.
    prep: vitaslop_platform::gpu::PrepareSplit,
}

/// Supersample factor for the live browser render (`VITASLOP_BROWSER_SUPERSAMPLE`).
///
/// Defaults to 1 - the panel's own resolution, which is what the guest asks the display
/// buffer to be rasterised at. The antialiasing a title really requests is on the render
/// targets it creates multisampled, and that is honoured now, so there is nothing left for
/// this to do but differ from the desktop. It used to default to 2 while the live page
/// always overrode it to 1, which meant no engine ever ran the documented default and a
/// phone could not be compared with a review shot at all.
///
/// A value that is not a positive integer is an ERROR rather than a silent fallback to the
/// default - a run configured by a typo would otherwise publish a rate for a resolution
/// nobody asked for.
fn supersample() -> u32 {
    match vitaslop_runtime::knobs::var("VITASLOP_BROWSER_SUPERSAMPLE") {
        Err(_) => 1,
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

/// Whether the per-window performance report is also written to the browser CONSOLE
/// (`VITASLOP_PERF_CONSOLE`).
///
/// OFF by default. The report is eight multi-line sections per window; on the page a
/// person actually plays on, that is a firehose that buries anything they might need to
/// see, and this page is the product rather than the instrument. Nothing is lost by
/// default: every section still reaches the on-screen diagnostics panel and the
/// dev-server sink, which is where a measurement is read from anyway.
fn perf_console() -> bool {
    vitaslop_runtime::knobs::flag("VITASLOP_PERF_CONSOLE")
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
/// >>> WHETHER THE ADAPTER SAYS THIS IS A PHONE, WHICH `navigator.deviceMemory` CANNOT.
///
/// `deviceMemory` is capped at 8 by its own specification, so the target phone reports the same
/// 8 GB a workstation does and every memory budget scales by 1.00 on the one device the scaling
/// was written for. The GPU vendor is not capped and is not a guess: `img-tec` is PowerVR, and
/// nothing but a mobile part ships one.
///
/// # AND IT IS DELIBERATELY NOT WIRED TO THE BUDGETS. Read this before doing that.
/// `knobs::scale_budget` only ever makes a budget SMALLER, and the texture budget is what gates
/// `transcoded_source`'s refusal to re-encode the guest's BC textures to ETC2 - a SECOND lossy
/// step over the title's own compression. Tightening the budget here would therefore buy memory
/// by spending picture quality, silently, on the exact device the change was aimed at
/// [[vitaslop-never-trade-quality]]. And the measurement does not ask for it: the phone dump
/// that raised this reports a 98 MB texture working set against a 477 MB budget and a 308 MB
/// wasm heap, with 12.7 ms of SLEEP in every 33.5 ms frame. Nothing there is short of memory.
///
/// So this reports, and changes nothing. If a device ever does show real memory pressure, the
/// budget to move is one that does not gate an encoder.
static ADAPTER_LOOKS_MOBILE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// GPU vendor strings that only ship in a phone or tablet. Matched against the WebGPU adapter's
/// own `vendor`/`architecture`, lowercased.
const MOBILE_ADAPTER_MARKERS: &[&str] =
    &["img-tec", "imagination", "powervr", "adreno", "qualcomm", "mali", "arm", "apple"];

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
/// Establish that WebGPU is genuinely usable here BEFORE any of it reaches `wgpu`, and name the
/// step that failed if it is not.
///
/// # Why this has to exist, and what it cost not to have it
/// `wgpu::Instance::request_adapter` on the WebGPU backend can return `Ok` holding an adapter
/// whose underlying JavaScript object is NULL. Nothing about that is visible from Rust: the
/// `Result` is fine, the `Adapter` exists, and the first property read off it - `adapter.features`
/// in the generated glue, which is the very first thing this renderer asks for - throws a JS
/// `TypeError` that no Rust error handling can intercept. Inside the emulator's worker that kills
/// the worker outright, and the user sees one line:
///
/// ```text
/// worker error: Uncaught TypeError: Cannot read properties of null (reading 'features')
/// ```
///
/// which names a property in generated glue and nothing about the cause. REPORTED FROM A DEVICE,
/// twice - the first time on `.info`, and removing that read only moved it one property along,
/// because it was a symptom. This is the cause: an adapter that does not exist must be refused at
/// the boundary, not carried inward.
///
/// Every step here is reflection with a guard, so this function itself can never throw.
async fn webgpu_preflight() -> Result<(), String> {
    use js_sys::{Function, Reflect};
    let global = js_sys::global();
    let navigator = Reflect::get(&global, &JsValue::from_str("navigator"))
        .map_err(|_| "no `navigator` in this context".to_string())?;
    let gpu = Reflect::get(&navigator, &JsValue::from_str("gpu"))
        .map_err(|_| "reading `navigator.gpu` threw".to_string())?;
    if gpu.is_undefined() || gpu.is_null() {
        return Err(
            "`navigator.gpu` is absent - this browser has no WebGPU, or it is disabled for this \
             origin. On Android, Chrome exposes WebGPU only on a SECURE context it trusts: a \
             self-signed certificate that was clicked through can be enough to withhold it. \
             Check chrome://gpu on the device."
                .into(),
        );
    }
    let request: Function = Reflect::get(&gpu, &JsValue::from_str("requestAdapter"))
        .map_err(|_| "`navigator.gpu.requestAdapter` is unreadable".to_string())?
        .dyn_into()
        .map_err(|_| "`navigator.gpu.requestAdapter` is not callable".to_string())?;
    // >>> ASK EVERY WAY THE SPEC ALLOWS, AND RETRY, BEFORE BELIEVING A NULL.
    //
    // Two different things make `requestAdapter` answer null, and only one of them is permanent.
    //
    // 1. TIMING. A phone that has just restarted its GPU process - which is what happens after a
    //    page crashed one, and this renderer has crashed one - answers null for a moment and then
    //    answers properly. A single ask turns a half-second race into "this device has no WebGPU".
    //
    // 2. THE REQUEST SHAPE. `powerPreference` is documented as a hint, but it is a hint an
    //    implementation is free to fail: a device with one GPU and no "high performance" tier can
    //    answer null to `high-performance` and hand over the very same adapter when asked with no
    //    preference at all. This renderer asked for `high-performance` and nothing else, so a
    //    device behaving that way looked exactly like a device with no WebGPU.
    //
    // So: every shape, several times, and the shape that works is the one the renderer then uses
    // - see `PREFERRED_POWER`. Reporting which shapes were tried is what makes the failure
    // actionable when none of them work.
    let shapes: [(&str, Option<&str>); 3] =
        [("high-performance", Some("high-performance")), ("default", None), ("low-power", Some("low-power"))];
    for round in 0..3 {
        for (name, pref) in shapes {
            let got = adapter_once(&request, &gpu, pref).await?;
            if !(got.is_null() || got.is_undefined()) {
                set_preferred_power(pref);
                if name != "high-performance" {
                    web_sys::console::log_1(&JsValue::from_str(&format!(
                        "adapter: `high-performance` was refused; this device answered to \
                         powerPreference `{name}`, which is what the renderer will use"
                    )));
                }
                return Ok(());
            }
        }
        if round < 2 {
            sleep_ms(300).await;
        }
    }
    Err(
        "`navigator.gpu.requestAdapter()` returned NULL for every powerPreference \
         (high-performance, default, low-power), three times each over a second - WebGPU is \
         present but this device will not hand over an adapter at all. That is a blocklisted or \
         repeatedly-crashed GPU process rather than a missing feature: open `chrome://gpu` on the \
         device and read `Graphics Feature Status` and `Problems Detected`. Note that Chrome \
         disables acceleration for a PROFILE after enough GPU-process crashes, and that survives \
         restarting the browser."
            .into(),
    )
}

/// The `powerPreference` this device actually answered to, chosen by [`webgpu_preflight`].
///
/// `wgpu` is asked with the same one. Preflighting with one shape and then letting the renderer
/// request another would mean the check passed for a request nobody makes.
static PREFERRED_POWER: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

fn set_preferred_power(pref: Option<&str>) {
    let v = match pref {
        Some("low-power") => 2,
        None => 1,
        _ => 0,
    };
    PREFERRED_POWER.store(v, std::sync::atomic::Ordering::Relaxed);
}

fn preferred_power() -> wgpu::PowerPreference {
    match PREFERRED_POWER.load(std::sync::atomic::Ordering::Relaxed) {
        1 => wgpu::PowerPreference::None,
        2 => wgpu::PowerPreference::LowPower,
        _ => wgpu::PowerPreference::HighPerformance,
    }
}

/// One `requestAdapter` call at a given `powerPreference` (`None` = ask with no preference at
/// all, which is a different request and can succeed where a preference is refused), guarded.
async fn adapter_once(
    request: &js_sys::Function,
    gpu: &JsValue,
    power: Option<&str>,
) -> Result<JsValue, String> {
    use js_sys::{Object, Reflect};
    let options = Object::new();
    if let Some(p) = power {
        let _ =
            Reflect::set(&options, &JsValue::from_str("powerPreference"), &JsValue::from_str(p));
    }
    let promise: js_sys::Promise = request
        .call1(gpu, &options)
        .map_err(|e| format!("`requestAdapter` threw: {e:?}"))?
        .dyn_into()
        .map_err(|_| "`requestAdapter` did not return a promise".to_string())?;
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|e| format!("`requestAdapter` rejected: {e:?}"))
}

/// `setTimeout` as an await point.
async fn sleep_ms(ms: i32) {
    let p = js_sys::Promise::new(&mut |resolve, _reject| {
        let global = js_sys::global();
        if let Ok(f) = js_sys::Reflect::get(&global, &JsValue::from_str("setTimeout")) {
            if let Ok(f) = f.dyn_into::<js_sys::Function>() {
                let _ = f.call2(&global, &resolve, &JsValue::from_f64(ms as f64));
            }
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(p).await;
}

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
    // Matched on the two fields a driver actually fills in, not on `description`, which on some
    // desktop parts carries a marketing string containing an unrelated vendor name. See
    // `ADAPTER_LOOKS_MOBILE` - this is READ ONLY BY THE PANEL and moves no budget.
    let mobile_haystack = format!("{vendor} {architecture}").to_lowercase();
    ADAPTER_LOOKS_MOBILE.store(
        !software && MOBILE_ADAPTER_MARKERS.iter().any(|m| mobile_haystack.contains(m)),
        std::sync::atomic::Ordering::Relaxed,
    );
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
        // >>> `adapter.get_info()` IS NOT CALLED HERE, and that is deliberate.
        //
        // On the WebGPU backend it compiles to a bare `adapter.info` property read in the
        // generated glue, with no guard. When that object is null the read throws a JS
        // `TypeError` that no Rust `Result` can catch, and because this runs in the emulator's
        // WORKER it takes the whole worker down - the user sees
        // `worker error: Uncaught TypeError: Cannot read properties of null (reading 'info')`
        // and nothing else. That was REPORTED from a device and never reproduced on this
        // machine, which is exactly the shape of failure worth removing rather than explaining.
        //
        // Nothing is lost. `info.name` is EMPTY on every browser measured here (the device's own
        // capture reads `wgpu name unavailable`), and the software check is what
        // `probe_webgpu_adapter` already answers, more precisely: it reads vendor, architecture
        // and `isFallbackAdapter` off `GPUAdapterInfo` through reflection, every step of which
        // returns `None` instead of throwing. `device_type == Cpu` only ever caught a FALLBACK
        // adapter, which that probe catches by name.
        let probe = probe_webgpu_adapter().await;
        let software = probe.as_ref().is_some_and(|p| p.software);
        let summary = format!(
            "adapter: {} | {}{}",
            probe.as_ref().map(|p| p.summary.as_str()).unwrap_or("navigator.gpu unreadable"),
            "wgpu name unavailable",
            if software { " | SOFTWARE RASTERISER" } else { " | GPU" },
        );
        web_sys::console::log_1(&JsValue::from_str(&summary));
        report.emit("adapter", &summary);
        // >>> WHICH COMPRESSED TEXTURE FAMILIES THIS ADAPTER OFFERS, because it decides the
        // single biggest memory question this renderer has and cannot be guessed from here.
        //
        // A guest PVRTC or UBC surface has no WebGPU format on every device we have looked at,
        // so it is CPU-decoded to RGBA8 - an 8x expansion. MEASURED on a race frame: 121 PVRTC
        // textures occupying 110 MB that are ~14 MB in their native 4bpp form, inside a 260 MB
        // working set against a 256 MB budget. On a phone the allocation past that budget is
        // what fails, and a failed texture draws WHITE.
        //
        // The fix depends entirely on what the DEVICE supports, and the two answers are
        // different pieces of work: `texture-compression-bc` means the guest's UBC1/2/3 blocks
        // can be handed over verbatim with no transcode at all, while ASTC or ETC2 alone means
        // a real transcoder. Printing the set turns that from an argument into a lookup, and it
        // costs one line per run. [[vitaslop-browser-gpu-must-be-proven]]
        let f = adapter.features();
        let compressed: Vec<&str> = [
            (wgpu::Features::TEXTURE_COMPRESSION_BC, "bc"),
            (wgpu::Features::TEXTURE_COMPRESSION_ETC2, "etc2"),
            (wgpu::Features::TEXTURE_COMPRESSION_ASTC, "astc"),
        ]
        .iter()
        .filter(|(bit, _)| f.contains(*bit))
        .map(|(_, name)| *name)
        .collect();
        let compressed = format!(
            "adapter compressed-texture support: {}",
            if compressed.is_empty() {
                "NONE - every PVRTC/UBC surface is decoded to RGBA8, ~8x expanded".to_string()
            } else {
                compressed.join(", ")
            }
        );
        web_sys::console::log_1(&JsValue::from_str(&compressed));
        // >>> ITS OWN ID. `Report::emit` RATE-LIMITS BY ID (100 ms), and this fires immediately
        // after the `adapter` summary above - so publishing it under the same name meant it was
        // DROPPED on every run this line has ever existed for. The one fact that decides whether
        // any compressed-texture work reaches a device has never been visible on a device.
        report.emit("adapter-compression", &compressed);
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
                required_features: vitaslop_platform::gpu::wanted_features(&adapter),
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
        // >>> AND IT IS INSTALLED AS A RAW JS HANDLER, NOT THROUGH `on_uncaptured_error`,
        // BECAUSE THAT ONE PANICS AND TAKES THE WORKER WITH IT.
        //
        // wgpu 30's `crate::Error::from_js` maps only `GPUValidationError` and
        // `GPUOutOfMemoryError` and ends in `panic!("Unexpected error")` for anything else -
        // which in practice is **`GPUInternalError`**, the error a driver raises when it
        // fails on a shader the validator already accepted. That is not hypothetical: an
        // Android PowerVR (img-tec D-series) device raised it on a retail title and the
        // panic killed the run worker four times over, so the ONE report that could have
        // named the bad pair was replaced by a wgpu backtrace.
        //
        // Handling it in JS means the error object never reaches that converter. A
        // `GPUInternalError` is exactly the case we most need reported - it is the device
        // telling us the shader is beyond it - and it must not be the case that crashes.
        match device.as_webgpu() {
            Some(raw) => {
                let cb = wasm_bindgen::closure::Closure::<dyn FnMut(web_sys::Event)>::new(
                    |ev: web_sys::Event| {
                        // `GPUUncapturedErrorEvent.error` is a `GPUError`, whose `message` is
                        // the device's own text. Read it reflectively: the concrete subclass
                        // (validation / out-of-memory / internal) is what wgpu chokes on, and
                        // naming it here is the diagnosis rather than a crash.
                        let err = js_sys::Reflect::get(&ev, &JsValue::from_str("error")).ok();
                        let kind = err
                            .as_ref()
                            .and_then(|e| js_sys::Reflect::get(e, &JsValue::from_str("constructor")).ok())
                            .and_then(|c| js_sys::Reflect::get(&c, &JsValue::from_str("name")).ok())
                            .and_then(|n| n.as_string())
                            .unwrap_or_else(|| "GPUError".into());
                        let msg = err
                            .and_then(|e| js_sys::Reflect::get(&e, &JsValue::from_str("message")).ok())
                            .and_then(|m| m.as_string())
                            .unwrap_or_default();
                        tracing::error!(
                            target: "vitaslop::gxm",
                            "WebGPU uncaptured error [{kind}]: {msg}"
                        );
                        // The message names the pipeline by its label, which is the shader-pair
                        // key. That is the only diagnosis a phone offers - and, more urgently,
                        // the only way to stop ONE refused pipeline invalidating every command
                        // buffer that binds it and blanking the whole frame. See
                        // `vitaslop_platform::gpu::note_device_error`.
                        vitaslop_platform::gpu::note_device_error(&kind, &msg);
                    },
                );
                raw.set_onuncapturederror(Some(cb.as_ref().unchecked_ref()));
                // Leaked deliberately: it must outlive the device, and the device outlives
                // the run. Dropping it would leave JS calling a freed closure.
                cb.forget();
            }
            // Not the WebGPU backend (a native build sharing this path): fall back to
            // wgpu's own hook, which is correct there - it is only the WEB converter that
            // panics on an unmapped class.
            None => device.on_uncaptured_error(std::sync::Arc::new(|e| {
                tracing::error!(target: "vitaslop::gxm", "WebGPU uncaptured error: {e}");
            })),
        }

        // >>> A LOST DEVICE IS THE END OF THE RUN, AND NOTHING USED TO SAY SO.
        //
        // When a WebGPU device is lost every object made from it becomes invalid: the
        // swapchain stops handing out textures, `get_current_texture` answers `Lost`, and
        // every draw, buffer write and bind-group build is silently discarded. The emulator
        // does not notice - the guest keeps executing, the capture keeps recording, the
        // encoder keeps being fed - so the run continues at full cost, forever, against a
        // BLACK CANVAS with no error anywhere. A player sees a hang; a log shows a healthy
        // frame rate. That is the worst shape a failure can have here and it is the one this
        // had [[vitaslop-fast-fail-no-silent-success]].
        //
        // The device is not re-obtained. Recovery would mean rebuilding the renderer and
        // every GPU object the frame's caches hold (pipelines, views, bind groups, the
        // resident geometry heap) from a state whose guest side has moved on, and a partial
        // rebuild renders a wrong picture rather than none - so the honest answer is to stop
        // and say why, in the copyable fatal box the device reports come out of.
        let lost: std::sync::Arc<std::sync::Mutex<Option<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(None));
        {
            let sink = lost.clone();
            device.set_device_lost_callback(move |reason, message| {
                let text = format!("{reason:?}: {message}");
                // Kept for `present` to turn into a fatal on the next frame. Reported HERE
                // too, because the callback can fire while the run is between presents and
                // the earliest possible word is the point.
                tracing::error!(target: "vitaslop::gxm", "WebGPU DEVICE LOST - {text}");
                if let Ok(mut slot) = sink.lock() {
                    if slot.is_none() {
                        *slot = Some(text);
                    }
                }
            });
        }

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        // The GXM decode yields final display-ready (sRGB-encoded) byte values, matching
        // the `Rgba8Unorm` software oracle. Render through a non-sRGB view so those bytes
        // land verbatim rather than getting a second sRGB encode (see the desktop
        // `RetailGfx` for the full note). WebGPU's preferred canvas format is normally
        // already non-sRGB, so this is usually a no-op in the browser.
        let render_format = format.remove_srgb_suffix();
        let view_formats = if render_format == format { vec![] } else { vec![render_format] };
        // OPAQUE if the platform offers it, rather than whatever it happens to list first.
        //
        // The guest's display buffer is an opaque picture: GXM's colour surface has no
        // meaningful alpha for the compositor, and whatever alpha our draws leave in the
        // framebuffer is a by-product of blending, not a transparency the page should honour.
        // Under `PreMultiplied` the browser composites the canvas against the page using
        // exactly that by-product, so a screen whose blending leaves alpha below 1 comes out
        // washed toward the page behind it - and it varies frame to frame, which reads as
        // flicker rather than as a wrong colour.
        //
        // Taking `alpha_modes[0]` hid this perfectly on every machine available here: desktop
        // adapters list `Opaque` first, so the wrong branch was never taken locally, while an
        // Android/PowerVR canvas commonly lists `PreMultiplied` first. That is the shape of
        // defect that only ever appears on the target device.
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes[0]
        };
        // The presentation oracle, off unless asked for. `COPY_SRC` is added to the surface
        // ONLY when the probe is on: it is a usage the canvas has to honour, and a page that
        // is not sampling its own output should not ask the platform for a capability it does
        // not use.
        let probe_every = vitaslop_runtime::knobs::var("VITASLOP_PRESENT_PROBE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|n| *n > 0);
        let surface_usage = if probe_every.is_some() {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        };
        // KEPT, not built inline: `present` has to be able to CONFIGURE THE SURFACE AGAIN
        // when the browser hands back `Outdated` (the canvas or its compositing changed) or
        // `Lost` (the swapchain died under a live device). Both are recoverable and both are
        // permanently black if the frame is merely skipped - see `PresentOutcome`.
        let surface_config = wgpu::SurfaceConfiguration {
            usage: surface_usage,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: WIDTH,
            height: HEIGHT,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats,
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);
        // What the SURFACE actually is, on the page, next to the adapter.
        //
        // Every field here is chosen from what the platform offers, so every one of them can
        // differ between the machine a change is written on and the phone it is judged on -
        // and all four change how the finished picture looks without changing a single draw.
        // A render defect that reproduces on one device and not the other is unfalsifiable
        // until these are visible, which is the position this cost two sessions.
        let surface_line = format!(
            "surface: format {format:?}, rendered through {render_format:?}, alpha {alpha_mode:?}, \
             present Fifo | offered formats {:?}, alpha modes {:?}",
            caps.formats, caps.alpha_modes,
        );
        web_sys::console::log_1(&JsValue::from_str(&surface_line));
        report.emit("surface", &surface_line);

        // Give the renderer a clock BEFORE it draws anything. See `perf_now`: without this the
        // encode phase split is zero here and reads as "encode costs nothing", on the one engine
        // where encode is 84% of the render.
        vitaslop_platform::gpu::set_wasm_clock(perf_now);
        let mut gxm = GxmRenderer::new(&device, &queue, render_format);
        // ONE sample per pixel, which is what the guest asks its display buffer for. How finely
        // a pass is rasterised is the guest's own call - it states samples-per-pixel per render
        // target with `SceGxmMultisampleMode`, and the renderer honours that - so a blanket
        // supersample here is a second, contradictory answer to a question already settled.
        // (This comment used to promise 2x for aliasing; the aliasing it described was measured
        // to be a depth problem, and 2x on this title's text came out very slightly SOFTER.)
        // `VITASLOP_BROWSER_SUPERSAMPLE` survives as a parity instrument against the software
        // oracle, not as a picture setting.
        gxm.set_supersample(supersample());
        let depth = make_depth(&device);
        let perf = global_performance().ok_or_else(|| JsValue::from_str("no performance clock"))?;
        let split_clock = Some(perf.clone());
        let probe = probe_every.map(|every| PresentProbe::new(&device, every));
        if probe.is_some() {
            tracing::warn!(
                target: "vitaslop::gxm",
                "VITASLOP_PRESENT_PROBE is on: the surface is sampled every {} presents and \
                 described in the diagnostics panel. This costs a full-surface copy and a \
                 buffer map on those frames.",
                probe_every.unwrap_or(0)
            );
        }
        // The surface's own description belongs in the PANEL, not only on the console.
        //
        // Every field in it is chosen from what the platform offers, so every one can differ
        // between the machine a change is written on and the phone it is judged on - which is
        // exactly the argument the line above it makes for printing it at all. It was emitted
        // to a `surface` element that only the desktop test pages define, so on a phone the
        // one report built for device-only render defects was invisible.
        let fps = FpsMeter::new(perf, report);
        Ok(LivePlayback {
            surface,
            device,
            queue,
            gxm,
            builder: RenderSceneBuilder::new(),
            depth,
            render_format,
            surface_line,
            probe_bgra: matches!(
                format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
            ),
            fps,
            perf: split_clock,
            split: RenderSplit::default(),
            targets: probe.is_some().then(TargetProbe::new),
            probe,
            last_probe: None,
            presents_total: 0,
            lost,
            surface_config,
            acquire_failures: 0,
            occluded: false,
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
    /// # Where the time goes, and why it is split HERE
    /// Mid-race this call costs about as much as the whole guest frame, and "render 32 ms"
    /// names no cause: BUILD (turning captured GXM scenes into render scenes, pure Rust in
    /// wasm), ENCODE (`encode_chain` - every pipeline, bind group, buffer write and draw,
    /// each one a call across the wasm/JS boundary into WebGPU), and SUBMIT/PRESENT (the
    /// queue and the swapchain) have completely different fixes, and the boundary one is
    /// the one this project has been caught by before
    /// [[vitaslop-browser-host-call-cost]]. Timed unconditionally - the clock is already
    /// read either side of `present` for the perf window, so this is three more reads on a
    /// path that costs tens of milliseconds.
    /// `display` is the `(width, height)` the guest declared to `sceDisplaySetFrameBuf`, which
    /// is what the frame is projected against. A title may declare a buffer SMALLER than the
    /// panel and let the display controller stretch it; projecting such a frame against the
    /// panel would put the whole picture in the top-left corner. The canvas is a fixed size
    /// and the surface stretches whatever is rendered into it, so passing the declared size
    /// here IS the hardware's upscale.
    fn present(
        &mut self,
        scenes: &[Scene],
        display: (u32, u32),
        presents: &[u32],
    ) -> PresentOutcome {
        let clock = |p: &Option<web_sys::Performance>| p.as_ref().map(|p| p.now()).unwrap_or(0.0);
        // Asked BEFORE any work: once the device is lost, building scenes and encoding a
        // command buffer is pure cost against a picture that cannot be drawn.
        if let Some(why) = self.lost.lock().ok().and_then(|s| s.clone()) {
            return PresentOutcome::Fatal(format!(
                "THE WebGPU DEVICE WAS LOST and every GPU object built from it is invalid, so \
                 nothing this run draws from here can reach the screen. The run is over.\n  \
                 reason: {why}\n  {}\n  Reload the page to start again. If this repeats on the \
                 same scene it is this renderer's fault, not the browser's: a device is lost \
                 when a driver faults or the GPU process is reset, and the shader or the \
                 allocation that did it is in the frame before this one.",
                self.surface_line
            ));
        }
        let t0 = clock(&self.perf);
        // Tell the builder a new frame starts here. Its texture cache needs the boundary to
        // know what is in use right now and how big one frame's working set is; without it the
        // cache cannot tell a texture it is about to need again from one it is finished with.
        self.builder.begin_frame();
        let built: Vec<_> = scenes.iter().map(|s| self.builder.build(s)).collect();
        let draws: usize = built.iter().map(|b| b.draws.len()).sum();
        let t1 = clock(&self.perf);
        // >>> EVERY VARIANT IS ANSWERED, AND THREE OF THEM ARE NOT "SKIP THE FRAME".
        //
        // `Outdated` and `Lost` do not clear themselves: a surface in either state answers the
        // same way for every subsequent frame, so a bare skip is a permanently black canvas
        // that costs a full frame of guest and encode work to produce. Both are recoverable by
        // configuring the surface again - which is why `surface_config` is kept - and the
        // recovery is bounded: if a run of presents in a row cannot acquire, the swapchain is
        // not coming back and continuing is the failure this refuses.
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => {
                self.acquire_failures = 0;
                self.occluded = false;
                t
            }
            // Acquired, but the texture no longer matches the surface. Render it - a slightly
            // stale frame beats none - and reconfigure so the next one matches.
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                self.acquire_failures = 0;
                self.occluded = false;
                self.surface.configure(&self.device, &self.surface_config);
                t
            }
            // >>> OCCLUDED IS THE ONE ARM THAT NEVER ESCALATES, and that is deliberate: a
            // backgrounded tab is occluded for as long as the user is looking at something
            // else, which is minutes, not frames. Counting it toward the give-up limit would
            // end a healthy run for the crime of being in another tab. It is still reported,
            // because "the emulator is running full tilt while nothing is on screen" is worth
            // knowing, and the counter is left alone so a real failure is not masked by it.
            wgpu::CurrentSurfaceTexture::Occluded => {
                self.note_occluded();
                return PresentOutcome::Skipped;
            }
            // A transient that clears on its own - unless it does not, which is a swapchain
            // that has stopped producing and a black screen for the rest of the run.
            wgpu::CurrentSurfaceTexture::Timeout => {
                self.acquire_failures += 1;
                if let Some(fatal) = self.acquire_gave_up("Timeout") {
                    return fatal;
                }
                return PresentOutcome::Skipped;
            }
            // Recoverable, but only by acting: reconfigure and take the next frame.
            other @ (wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost) => {
                self.acquire_failures += 1;
                if let Some(fatal) = self.acquire_gave_up(&format!("{other:?}")) {
                    return fatal;
                }
                tracing::warn!(
                    target: "vitaslop::gxm",
                    "the surface came back {other:?} - configuring it again and skipping this \
                     frame ({} in a row)",
                    self.acquire_failures
                );
                self.surface.configure(&self.device, &self.surface_config);
                return PresentOutcome::Skipped;
            }
            // A validation error INSIDE the acquire. That is this renderer's own bug (a
            // surface configured with something the device will not take), it does not clear,
            // and nothing else would ever print it.
            wgpu::CurrentSurfaceTexture::Validation => {
                return PresentOutcome::Fatal(format!(
                    "THE SURFACE REFUSED TO HAND OUT A TEXTURE with a VALIDATION error, which \
                     means this renderer configured it with something this device will not \
                     accept. Nothing can be drawn and the run is over.\n  {}",
                    self.surface_line
                ));
            }
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.render_format),
            ..Default::default()
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Which buffers the guest flipped while these scenes were captured - see
        // `GxmRenderer::set_presented`.
        self.gxm.set_presented(presents);
        self.gxm.encode_chain(
            &self.device,
            &self.queue,
            &mut encoder,
            &view,
            &self.depth,
            &built,
            display.0,
            display.1,
            frame.texture.width(),
            frame.texture.height(),
            CLEAR,
        );
        let t2 = clock(&self.perf);
        // Sample the surface BEFORE it is presented - once presented it is no longer ours to
        // read. The copy rides the same encoder, so it costs no extra submit.
        self.presents_total += 1;
        let sampling = self.probe.as_ref().is_some_and(|p| p.wants(self.presents_total));
        if sampling {
            let n = self.presents_total;
            if let Some(probe) = self.probe.as_mut() {
                probe.capture(&mut encoder, &frame.texture, n);
            }
            // The offscreen targets of the SAME frame, so the chain and the surface describe
            // one picture rather than two moments.
            if let Some(tp) = self.targets.as_mut() {
                let list = self.gxm.rtt_targets();
                tp.capture(&self.device, &mut encoder, &list, n);
            }
        }
        self.queue.submit([encoder.finish()]);
        // The map is requested only now: before the submit it would resolve against an
        // unwritten buffer and describe zeros. See `PresentProbe::begin_map`.
        if let Some(probe) = self.probe.as_mut() {
            probe.begin_map();
        }
        if let Some(tp) = self.targets.as_mut() {
            tp.begin_map();
        }
        // Deliver any map callback that is ready.
        //
        // On the device the probe announced itself and then produced NOTHING, on two runs -
        // the buffer was mapped and the callback never arrived. wgpu queues map callbacks
        // internally and delivers them from `poll`; native calls it with a blocking wait,
        // which a browser cannot do, so nothing was draining the queue at all. `Poll` is
        // non-blocking and is a no-op when there is nothing pending, so it is safe to call
        // every present rather than only when a probe is in flight.
        let _ = self.device.poll(wgpu::PollType::Poll);
        self.queue.present(frame);
        let t3 = clock(&self.perf);
        // `encode_chain` already splits itself over every pass of the frame - prepare (the
        // scene walk, which for a recompiled draw creates its bind groups), upload (the
        // arena writes) and pass (command encoding). Take that rather than reporting one
        // opaque encode number: the three have different fixes and only `pass` scales with
        // the wasm/JS boundary crossings per draw.
        let ph = self.gxm.last_phases();
        // Per PRESENT, not per window: the worst frame is the one that needs explaining, and
        // the counters that explain it have to come from that same frame.
        let work = vitaslop_runtime::render::take_build_work();
        let enc_work = vitaslop_platform::gpu::take_encode_work();
        if t1 - t0 > self.split.worst_build_ms {
            self.split.worst_build_ms = t1 - t0;
            self.split.worst_draws = draws;
            self.split.worst_work = work;
        }
        if t2 - t1 > self.split.worst_encode_ms {
            self.split.worst_encode_ms = t2 - t1;
            self.split.worst_encode_draws = draws;
            self.split.worst_enc_work = enc_work;
            self.split.worst_enc_phases = ph;
        }
        self.split.work.add_pub(&work);
        self.split.enc_work.add(&enc_work);
        // Taken every present whether or not the knob is on: `take_prepare_split` is a handful
        // of relaxed swaps, and taking it here keeps the window's tally over exactly the
        // presents this window counted. Everything it holds is zero when the knob is off.
        self.split.prep.add(&vitaslop_platform::gpu::take_prepare_split());
        self.split.build_ms += t1 - t0;
        self.split.encode_ms += t2 - t1;
        self.split.prepare_ms += ph.prepare_ms;
        self.split.upload_ms += ph.upload_ms;
        self.split.arena_ms += ph.arena_ms;
        self.split.arena_create_ms += ph.arena_create_ms;
        self.split.arena_write_ms += ph.arena_write_ms;
        self.split.ubo_bg_ms += ph.ubo_bg_ms;
        self.split.precompile_ms += ph.precompile_ms;
        self.split.retire_ms += ph.retire_ms;
        self.split.resident_ms += ph.resident_ms;
        self.split.pass_ms += ph.pass_ms;
        self.split.gxp_draws += ph.gxp_draws as u64;
        self.split.fixed_draws += ph.fixed_draws as u64;
        self.split.submit_ms += t3 - t2;
        self.split.scenes += scenes.len() as u64;
        self.split.draws += draws as u64;
        self.split.presents += 1;
        // Collect a finished readback. The map callback lands on an event-loop turn after the
        // copy, so this is always a LATER frame than the one it describes - which is why the
        // description carries its own frame number.
        let bgra = self.probe_bgra;
        if let Some(text) = self.probe.as_mut().and_then(|p| p.take_report(bgra)) {
            self.last_probe = Some(text);
        }
        // The offscreen targets go into the SAME panel section, under the surface, because the
        // two are one reading: the surface says the picture is wrong and the chain says where.
        if let Some(text) = self.targets.as_mut().and_then(|t| t.take_report(bgra)) {
            let mut s = self.last_probe.take().unwrap_or_default();
            s.push_str("CHAIN TARGETS, ");
            s.push_str(&text);
            self.last_probe = Some(s);
        }
        self.fps.tick();
        PresentOutcome::Presented
    }

    /// How many presents in a row may fail to acquire a surface texture before the run is
    /// declared over.
    ///
    /// Sized to be unmistakable rather than tight: a reconfigure takes effect on the next
    /// frame and a tab can be occluded for as long as the user looks elsewhere, so the bar is
    /// "this has not worked for several seconds of continuous attempts" - about four seconds
    /// at the pace this loop runs. Anything that recovers resets the counter to zero, so a
    /// healthy run can never reach it however long it plays.
    const ACQUIRE_FAILURE_LIMIT: u32 = 240;

    /// Say that the surface is occluded, once per occlusion rather than once per frame - the
    /// state lasts as long as the tab is in the background, and a line a frame would be the
    /// whole log.
    fn note_occluded(&mut self) {
        if !self.occluded {
            self.occluded = true;
            tracing::warn!(
                target: "vitaslop::gxm",
                "the surface is OCCLUDED (the page is not being composited): the emulator is \
                 still running the guest at full cost and nothing it renders is reaching the \
                 screen. This clears by itself when the page is visible again."
            );
        }
    }

    /// `Some(fatal)` once the surface has refused for [`Self::ACQUIRE_FAILURE_LIMIT`] presents
    /// in a row. Split out so both acquire arms escalate on exactly the same rule.
    fn acquire_gave_up(&self, last: &str) -> Option<PresentOutcome> {
        (self.acquire_failures >= Self::ACQUIRE_FAILURE_LIMIT).then(|| {
            PresentOutcome::Fatal(format!(
                "THE SURFACE HAS NOT PRODUCED A TEXTURE FOR {} PRESENTS IN A ROW (last answer: \
                 {last}), so nothing this run has rendered since then reached the screen. \
                 Continuing would keep the emulator at full cost against a black canvas, which \
                 is indistinguishable from a hang, so the run stops here.\n  {}",
                self.acquire_failures, self.surface_line
            ))
        })
    }

    /// The render split accumulated since the last read, and reset. Reported alongside the
    /// perf window so the two describe the same frames.
    fn take_split(&mut self) -> RenderSplit {
        core::mem::take(&mut self.split)
    }

    /// The surface description, for the diagnostics panel.
    /// How full the renderer's growing caches are - see `GxmRenderer::cache_sizes`.
    fn cache_sizes(&self) -> String {
        self.gxm.cache_sizes()
    }

    fn surface_line(&self) -> &str {
        &self.surface_line
    }

    /// The latest presented-surface description, if the probe produced one since the last
    /// window.
    fn take_probe_report(&mut self) -> Option<String> {
        self.last_probe.take()
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

/// >>> TELL THE ENGINE HOW MUCH MEMORY THIS DEVICE HAS, from `navigator.deviceMemory`.
///
/// Every cache budget in this project was an absolute constant fitted on a desktop, adding up to
/// close to a gigabyte before the guest's own memory or the wasm heap - and the target device is
/// a phone whose owner has reported, repeatedly, that a long session makes the WHOLE device
/// sluggish. Nothing anywhere read a single property of the machine. This is the property the
/// browser already publishes, and the only one it publishes.
///
/// Best-effort by design: `deviceMemory` is absent on Firefox and Safari, and its own
/// specification caps it at 8, so a machine that does not answer and every desktop that does get
/// the budgets exactly as they are today. See [`vitaslop_platform::knobs::memory_scale`].
///
/// Called once at startup, before the first frame. It is not a knob - the engine that pays for
/// this runs whatever is defaulted [[vitaslop-pick-the-default-dont-add-a-knob]].
fn report_device_memory() {
    let Some(nav) = js_sys::global().dyn_ref::<web_sys::WorkerGlobalScope>().map(|g| g.navigator())
    else {
        return;
    };
    let Ok(v) = js_sys::Reflect::get(&nav, &JsValue::from_str("deviceMemory")) else {
        return;
    };
    let Some(gb) = v.as_f64() else {
        return;
    };
    vitaslop_platform::knobs::set_device_memory_gb(gb);
}

/// `performance.now()`, as a plain `fn` the renderer can hold.
///
/// `vitaslop-platform` cannot reach the browser clock itself (it has no `js-sys`, on purpose -
/// `vitaslop-runtime` depends on it for the neutral seam types), so its own phase Stopwatch
/// read 0.0 on wasm and the whole `prepare/upload/pass` split was structurally zero here. That
/// is what "the browser cannot split `encode`" really was: a statement about
/// `std::time::Instant`, not about the browser. This is installed with
/// `gpu::set_wasm_clock` and the split is measured on both engines from then on.
///
/// The `Performance` object is looked up ONCE and cached: it is a `Reflect::get` off the
/// global, and this is called four times per pass on a path that runs hundreds of times a
/// second.
fn perf_now() -> f64 {
    thread_local! {
        static P: Option<web_sys::Performance> = global_performance();
    }
    P.with(|p| p.as_ref().map(|p| p.now()).unwrap_or(0.0))
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
/// responsive.
///
/// # >>> IN A WORKER IT IS A `MessageChannel`, BECAUSE `setTimeout(0)` IS CLAMPED TO 4 ms
/// A worker has no `requestAnimationFrame`, and the obvious fallback - `setTimeout(cb, 0)` -
/// is not zero. Every browser clamps a NESTED timeout (one scheduled from inside a timeout
/// callback, five deep) to a **4 ms minimum**, and this loop is an unbroken chain of exactly
/// that: the tick's continuation schedules the next tick. So the live loop paid ~4 ms of pure
/// waiting between every presented frame, on the engine that ships, and it was invisible in
/// every split - `cpu` measures the guest half and `render` the present, and this sits
/// between them where nothing was looking.
///
/// It matters more than its size suggests, because the race sits ON the vsync boundary: work
/// per present is ~17 ms against a 16.7 ms interval, so 4 ms decides whether a present lands
/// on one interval or two.
///
/// A `MessageChannel` message is a macrotask like a timeout - the event loop still turns, so
/// posted input and messages are serviced exactly as before - and it is not clamped. The
/// channel is built ONCE and reused: a fresh pair of ports per frame would trade the clamp
/// for an allocation.
///
/// >>> THE `window()` PROBE IS ASKED ONCE, AND THAT IS WORTH 20% OF THE WORKER THREAD.
/// `web_sys::window()` is an `instanceof Window` test on the global, and in a WORKER the
/// identifier `Window` does not exist at all - so the generated glue THROWS a `ReferenceError`
/// and catches it, on every call. That is ~20 us each with a deep JSPI stack live, and this
/// loop does not run once per frame: it spins on `next_tick` until wall time accrues, thousands
/// of times a second. MEASURED with a V8 sampling profile of the worker during a paced race:
/// `__wbg_instanceof_Window` was **17% of every sample on the thread**, the single largest
/// entry ahead of all guest code. Which global this is cannot change while the thread lives,
/// so it is one probe, cached.
///
/// It buys no frame rate on a machine that is already behind - a loop with no slack does not
/// spin - and that is not why it is here. A fifth of a core burnt continuously is heat on a
/// phone, contention with the GPU process, and battery, and thermal throttling is one of the
/// few things that can take a device from 60 to 30 in the middle of a race.
/// Wait until the next frame is DUE, rather than spinning until it is.
///
/// >>> THE SPIN WAS A TENTH OF THE WORKER THREAD, AND IT WAS WAITING.
/// [`next_tick`] is not called once per frame - the live loop calls it, checks the clock, finds
/// too little time has accrued to run a frame, and calls it again. A `MessageChannel` round
/// trip is ~20-50 us, so on a machine with slack (this desktop runs a race frame in ~10 ms of
/// a 16.7 ms budget) that is thousands of empty iterations a second. MEASURED with a V8
/// sampling profile of the worker during a paced race: `postMessage` 5.7% of all samples,
/// `JsFuture::from` + the per-tick closure another 3.3%, `set_onmessage` and `queueMicrotask`
/// most of the rest - **roughly a tenth of the thread, spent asking "is it time yet"**.
///
/// So when the next frame is more than a few milliseconds away, wait with ONE timeout instead.
/// The 4 ms nested-timeout clamp that made timeouts unusable for the tick itself is a FLOOR,
/// not a rounding: a `setTimeout(9)` waits about 9 ms. The last few milliseconds are still
/// approached on the channel, so the frame boundary keeps its old precision and a timer that
/// fires late is absorbed by the accumulator exactly as a late tick always was.
///
/// A machine that is BEHIND asks for zero and gets the old path unchanged - there is no slack
/// to sleep in, and that is the machine whose frame rate this must not touch. What it buys
/// there is on the phone: a tenth of a core not burnt is heat not made, and thermal throttling
/// is one of the few things that takes a device from 60 fps to 30 mid-race.
async fn next_tick_in(ms: f64) {
    // Below this, the channel is both cheaper and more precise than a timer.
    const MIN_SLEEP_MS: f64 = 5.0;
    // Wake this early and approach the boundary on the channel, so a coarse timer cannot
    // make a frame LATE - only slightly early, which costs one cheap tick.
    const SLACK_MS: f64 = 3.0;
    // >>> ONE EVENT-LOOP TURN PER TICK, AND THE REST OF THE WAIT COSTS NOTHING.
    //
    // Where `Atomics.wait` is available this is the whole function: sleep the wait out on it,
    // keep a millisecond back, and spend that millisecond on the ONE channel turn the tick owes
    // the event loop. That turn is not optional - a `VideoDecoder` answers on a TASK, and a
    // worker that never returns to its event loop starves the movie it is waiting for
    // ([[vitaslop-a-host-call-that-never-yields-starves-the-browser]]) - but ONE is enough, and
    // the loop was taking about fifteen.
    //
    // MEASURED before this, V8 profile of the worker during a race: `postMessage` 9.18% of all
    // samples, with `__wbindgen_cast` 3.37% + 2.05%, `_wbg_cb_unref` 1.56%, `queueMicrotask`
    // 1.30% and `set_onmessage` 0.77% behind it - roughly a sixth of the thread spent asking
    // "is it time yet". The timer path below is what a host without `Atomics.wait` gets.
    const YIELD_RESERVE_MS: f64 = 1.0;
    if ms > YIELD_RESERVE_MS {
        let deadline = now_ms().map(|t| t + ms);
        if precise_sleep(ms - YIELD_RESERVE_MS) {
            // The turn the event loop is owed, taken while there is still time in hand for it.
            next_tick().await;
            // Whatever the turn did not use. Landing ON the deadline matters: returning early
            // puts the caller straight back here with a sub-millisecond wait, which is the spin
            // this function exists to avoid.
            if let (Some(deadline), Some(now)) = (deadline, now_ms()) {
                if deadline > now {
                    precise_sleep(deadline - now);
                }
            }
            return;
        }
    }
    if ms < MIN_SLEEP_MS {
        next_tick().await;
        return;
    }
    // >>> THE APPROACH THE DOC ABOVE PROMISES, WHICH THE CODE DID NOT MAKE.
    //
    // Sleeping `ms - SLACK` and returning leaves the wake-up wherever the host's timer put
    // it. A worker's timer is coarse - the 4 ms clamp is a FLOOR and the jitter above it is
    // the host's business - so the loop woke LATE, ran its one frame late, and presented
    // late, every tick. Under one-guest-frame-per-tick that is not a wobble that averages
    // out: the guest advances by exactly one frame whether the tick took 16.7 ms or 19.5,
    // so every millisecond of lateness is emulated time the run never gets back. A device
    // measured `period 19.5 ms` where the work was 10.2 - 86% speed out of a machine with
    // 40% headroom.
    //
    // So the timer only gets the loop CLOSE, and the last few milliseconds are walked on
    // the channel (~20-50 us a turn), which is what the comment above always claimed. The
    // spin that cost a tenth of the worker thread was the OUTER loop asking "is it time
    // yet" for the whole sleep; three or four turns at the end of one is not that.
    //
    // >>> AND THE SLACK IS LEARNED, BECAUSE A FIXED 3 ms IS A GUESS ABOUT A HOST.
    //
    // The walk repairs a timer that fires EARLY. It can do nothing about one that fires LATE,
    // and late is what a phone's worker does: Android Chrome coalesces and throttles timers in
    // a busy renderer, and a device measured 9.3 ms of sleep on ticks where this function was
    // asked for less than that - the walk cannot give back time already spent. So the timer's
    // own error is measured and subtracted from the next request. A host that returns on time
    // keeps a 3 ms slack and one channel turn; a host that runs 15 ms late is asked for 15 ms
    // less until it lands, and in the limit stops being asked at all and the wait becomes the
    // walk. It costs one f64 per thread and it needs no knob, because the number it wants is a
    // property of the host and cannot be known here.
    let deadline = now_ms().map(|t| t + ms);
    let delay = (ms - SLACK_MS - timer_overshoot_ms()).clamp(0.0, 1000.0);
    // Below the host's own timer floor there is nothing to ask for: a worker clamps a nested
    // `setTimeout` to about 4 ms, so a 1 ms request comes back in 4 and the measurement below
    // would read that as the host running 3 ms late - a correction that feeds itself. Walk
    // instead, which is what the remaining time is short enough for anyway.
    const MIN_TIMER_MS: f64 = 4.0;
    if delay < MIN_TIMER_MS {
        let Some(deadline) = deadline else { return };
        let mut turns = 0u32;
        while turns < MAX_APPROACH_TURNS {
            let Some(left) = now_ms().map(|t| deadline - t).filter(|&l| l > 0.0) else { return };
            if precise_sleep(left) {
                return;
            }
            turns += 1;
            next_tick().await;
        }
        return;
    }
    let asked_at = now_ms();
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let cb = Closure::once_into_js(move |_: JsValue| {
            let _ = resolve.call0(&JsValue::UNDEFINED);
        });
        let set_timeout = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("setTimeout"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok());
        match set_timeout {
            Some(f) => {
                let _ = f.call2(
                    &JsValue::UNDEFINED,
                    cb.as_ref().unchecked_ref(),
                    &JsValue::from_f64(delay),
                );
            }
            // No `setTimeout` at all is not a thing any host does, but resolving immediately
            // degrades to the old spin rather than hanging the run.
            None => {
                let _ = js_sys::Function::from(cb).call0(&JsValue::UNDEFINED);
            }
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    if let (Some(t0), Some(t1)) = (asked_at, now_ms()) {
        note_timer_overshoot(t1 - t0 - delay);
    }
    // Walk the rest of the way, on the CHEAPEST wait this thread has - see `precise_sleep`.
    let Some(deadline) = deadline else { return };
    let mut turns = 0u32;
    while turns < MAX_APPROACH_TURNS {
        let Some(left) = now_ms().map(|t| deadline - t).filter(|&l| l > 0.0) else { break };
        if precise_sleep(left) {
            break;
        }
        turns += 1;
        next_tick().await;
    }
}

/// Block this thread for `ms` on `Atomics.wait`, returning whether it did.
///
/// >>> THE TICK'S OWN WAIT WAS A TENTH OF THE WORKER THREAD.
///
/// MEASURED with a V8 profile of the worker during a race: `postMessage` **9.18%** of all
/// samples, plus `__wbindgen_cast` 3.37% + 2.05%, `_wbg_cb_unref` 1.56%, `queueMicrotask`
/// 1.30% and `set_onmessage` 0.77% - the closure churn around it. Every one of those is the
/// `MessageChannel` round trip this loop uses to wait, and it uses it because a worker's timer
/// is clamped to 4 ms and the last few milliseconds of a frame have to come from somewhere.
///
/// A worker has a better primitive. `Atomics.wait` blocks the thread for a precise duration
/// with no task, no closure and no allocation - it is the one wait in the platform that costs
/// nothing to take. It is forbidden on the main thread, which is why this is guarded and falls
/// back to the channel: the same live loop runs there in the non-worker mode.
///
/// What it gives up is the event loop for the duration, which is exactly what a loop with
/// nothing to do wants to give up. The audio ring is serviced by the worklet on its own thread
/// out of the same `SharedArrayBuffer`, so a sleep here is not a gap in the sound.
fn precise_sleep(ms: f64) -> bool {
    if ms <= 0.0 {
        return true;
    }
    thread_local! {
        /// A one-word shared array to wait on. Its value is never changed, so the wait always
        /// runs to its timeout - this is a sleep, not a handshake. `None` where
        /// `SharedArrayBuffer` or `Atomics.wait` is unavailable (the main thread, or a page
        /// that is not cross-origin isolated).
        static SLEEPER: Option<js_sys::Int32Array> = {
            let ok = js_sys::global()
                .dyn_ref::<web_sys::DedicatedWorkerGlobalScope>()
                .is_some();
            ok.then(|| js_sys::Int32Array::new(&js_sys::SharedArrayBuffer::new(4).into()))
                .filter(|a| js_sys::Atomics::wait_with_timeout(a, 0, 1, 0.0).is_ok())
        };
    }
    // `Atomics.wait` sleeps only while the stored word EQUALS the value asked for, and returns
    // "not-equal" immediately otherwise. The word is zero and stays zero, so asking for 0 is
    // what makes this a sleep - and asking for 1, as the capability probe above does, is what
    // makes that probe return at once instead of blocking for its timeout.
    SLEEPER.with(|s| match s {
        Some(a) => js_sys::Atomics::wait_with_timeout(a, 0, 0, ms).is_ok(),
        None => false,
    })
}

/// How many channel turns `next_tick_in` will spend walking up to a deadline before it gives
/// up and returns late anyway. A turn is tens of microseconds and the timer lands within a
/// few milliseconds of its target, so the loop below normally runs a handful of times; the
/// bound exists so a host whose clock or channel misbehaves cannot turn pacing into a spin.
const MAX_APPROACH_TURNS: u32 = 200;

thread_local! {
    /// How late this host's timer has been running, in milliseconds - an EMA of
    /// `(actual - requested)` over the sleeps [`next_tick_in`] has taken. Subtracted from the
    /// next request so the wake-up lands on the frame boundary instead of past it.
    static TIMER_OVERSHOOT_MS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// The learned timer error, never negative: a host whose timer returns EARLY is already handled
/// by the channel walk, and asking it for MORE than the wait would make the loop late on
/// purpose.
fn timer_overshoot_ms() -> f64 {
    TIMER_OVERSHOOT_MS.with(|c| c.get())
}

/// Fold one sleep's error into the estimate.
///
/// Rises fast and falls slow (1/4 up, 1/32 down). Being late costs emulated time every tick it
/// happens, and being early costs one channel turn - so the estimate should chase a host that
/// starts running late and let go of it reluctantly. Clamped to a frame: an outlier from a
/// hitch or a backgrounded tab must not make the next sleep zero for the rest of the run.
fn note_timer_overshoot(error_ms: f64) {
    TIMER_OVERSHOOT_MS.with(|c| {
        let prev = c.get();
        let next = if error_ms > prev {
            prev + (error_ms - prev) * 0.25
        } else {
            prev + (error_ms - prev) / 32.0
        };
        c.set(next.clamp(0.0, 1000.0 / 60.0));
    });
}

/// `performance.now()` from whatever global this thread has, or `None` if there is none.
///
/// The live loop holds a `Performance` already; this is for the free functions that do not,
/// and it is looked up once per thread rather than per call.
fn now_ms() -> Option<f64> {
    thread_local! {
        static PERF: Option<web_sys::Performance> = js_sys::Reflect::get(
            &js_sys::global(),
            &JsValue::from_str("performance"),
        )
        .ok()
        .and_then(|p| p.dyn_into::<web_sys::Performance>().ok());
    }
    PERF.with(|p| p.as_ref().map(|p| p.now()))
}

async fn next_tick() {
    // `None` = this thread has no `window` (a worker), which is the case the loop actually
    // runs in. Asked once per thread and remembered.
    thread_local! {
        static WINDOW: Option<web_sys::Window> = web_sys::window();
    }
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let cb = Closure::once_into_js(move |_t: JsValue| {
            let _ = resolve.call0(&JsValue::UNDEFINED);
        });
        if WINDOW.with(|w| {
            w.as_ref().map(|window| {
                let _ = window.request_animation_frame(cb.as_ref().unchecked_ref());
            })
        }).is_some() {
            return;
        }
        thread_local! {
            static CHANNEL: Option<web_sys::MessageChannel> = web_sys::MessageChannel::new().ok();
        }
        let posted = CHANNEL.with(|ch| {
            let Some(ch) = ch else { return false };
            // The resolver rides ON the message, so the port needs no per-call handler
            // registration: `port1.onmessage` is set here each time (a property write, not an
            // allocation) and fires once for the message posted immediately after.
            let f: js_sys::Function = cb.as_ref().unchecked_ref::<js_sys::Function>().clone();
            ch.port1().set_onmessage(Some(&f));
            ch.port2().post_message(&JsValue::NULL).is_ok()
        });
        if posted {
            return;
        }
        // No `MessageChannel` (or it refused): the clamped timeout is still correct, just
        // slower, so it stays as the fallback rather than the run stopping.
        let set_timeout = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("setTimeout"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok());
        if let Some(set_timeout) = set_timeout {
            let _ = set_timeout.call2(
                &JsValue::UNDEFINED,
                cb.as_ref().unchecked_ref(),
                &JsValue::from_f64(0.0),
            );
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
    dirty_off: Option<u64>,
    /// Guest address of each transpiled function, in wasm function order.
    ///
    /// # Why this has to cross the worker boundary
    /// The production path transpiles in a THROWAWAY worker and hands the run worker a
    /// compiled module. The run worker therefore never sees the artifact - so it never
    /// learns which guest function each wasm index is, and a guest fault's backtrace comes
    /// back as bare module indices that nobody holding a phone can resolve. MEASURED: the
    /// first device capture after the fatal box was wired showed ten frames of
    /// `wasm-function[5362]` and nothing else, because the table was recorded in the worker
    /// that had already been thrown away.
    ///
    /// It is ~88 KB for a retail title (one u32 per function), transferred once at setup.
    func_addrs: Vec<u32>,
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
        // The guest-store dirty map. Absent is a legitimate answer here (a module built
        // without tracking has none), so unlike the two above it is not an error - what
        // would be an error is treating a missing map as "nothing was written", which is
        // why it travels as an Option all the way to `GuestMemory::dirty_since`.
        let dirty = get("dirtyOff")?;
        let dirty_off = if dirty.is_null() || dirty.is_undefined() {
            None
        } else {
            Some(dirty.as_f64().ok_or_else(|| JsValue::from_str("bad prebuilt.dirtyOff"))? as u64)
        };
        // Absent is tolerated: an older transpile worker did not send it, and a backtrace
        // with unnamed frames is worse than one with names but not worth failing the run.
        let func_addrs = match get("funcAddrs")? {
            v if v.is_undefined() || v.is_null() => Vec::new(),
            v => js_sys::Uint32Array::new(&v).to_vec(),
        };
        Ok(Some(Prebuilt { module, mem_pages, mirror_off, dirty_off, func_addrs }))
    }
}

/// What a transpile produced, without the artifact's other baggage.
struct Transpiled {
    wasm: Vec<u8>,
    mem_pages: u32,
    mirror_off: Option<u64>,
    dirty_off: Option<u64>,
    ms: f64,
    /// Guest address of each transpiled function - see [`Prebuilt::func_addrs`].
    func_addrs: Vec<u32>,
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
    // And ask it to stamp guest STORES, which lets the capture prove a texture is
    // unchanged without comparing its bytes (`TextureSnapshots`) - 40% of a race frame
    // on the desktop, and about half the browser's guest CPU. Emitted unbilled, so the
    // game clock cannot tell the difference. Native does not do this either: wasmtime
    // bills every operator it executes, so the stamps would speed its clock up.
    vitaslop_transpiler::set_dirty_tracking(true);
    // Hand the engine-agnostic runtime this engine's clock, so its per-phase timers work
    // HERE. They are `#[cfg]`-inert on wasm without one - there is no `Instant` - which is
    // why a browser frame could only ever report one undifferentiated number while the
    // desktop profiler split the same code into eight phases. Gated on `VITASLOP_PERF`
    // inside `perf`, so an ordinary run still pays nothing.
    vitaslop_runtime::perf::set_clock(browser_sched::perf_clock);
    // And whether to hold the ARM register file in wasm LOCALS along each straight-line
    // run instead of on its globals (`transpiler::promote`). Routed through the override
    // table rather than read from the environment because THIS is the engine that has to
    // answer the question: promotion adds operators and removes none, so fuel and the
    // expansion factor cannot see it, and V8 wall-clock on matched frames is the only
    // instrument that can.
    //
    // `knobs::flag` honours `=0` as OFF - it did not always, and that cost a measurement:
    // an A/B whose OFF arm was written `VITASLOP_PROMOTE_REGS=0` ran the PROMOTED build in
    // both arms and reported a clean, meaningless "no difference". See `knobs::flag`.
    vitaslop_transpiler::set_promote_registers(
        vitaslop_runtime::knobs::flag("VITASLOP_PROMOTE_REGS"),
    );
    // And which carry/overflow forms `emit_flags_add` uses. Routed the same way and for the
    // same reason: `flags-add` was 39% of every operator the transpiler emitted, the new
    // closed forms cut the module 5.3% and executed operators 8.7%, and THREE interleaved
    // desktop repeats put the wall-clock difference inside the noise. V8 on a phone is the
    // engine the answer belongs to. `=1` selects the OLD 64-bit carry.
    vitaslop_transpiler::set_flags_wide_c(vitaslop_runtime::knobs::flag("VITASLOP_FLAGS_WIDE_C"));
    // And whether every FALLTHROUGH is routed through the function's `br_table` dispatch
    // loop as well. This one is an ABLATION - it can only make the module slower - and it is
    // here because it is the only way to price what a structured-control-flow emitter (a
    // relooper) would be worth. The module carries one indirect branch per 10.5 guest
    // instructions; whether that is 2% of a browser frame or 25% decides whether the
    // relooper is the next big piece of work or a refuted idea, and neither the operator
    // count nor the fuel figure can tell the difference.
    vitaslop_transpiler::set_dispatch_all(vitaslop_runtime::knobs::flag("VITASLOP_DISPATCH_ALL"));
    // And whether the module carries the guest-address name section. Browser-reachable for
    // the profiler's sake: a V8 CPU profile of the worker is the only instrument that can rank
    // the inside of a frame without taxing what it measures, and without this section every
    // guest function in it is a bare `wasm-function[N]`.
    vitaslop_transpiler::set_wasm_names(vitaslop_runtime::knobs::flag("VITASLOP_WASM_NAMES"));
    let t = perf.now();
    let built = vitaslop_transpiler::transpile_lenient(&linked.shared_program());
    let ms = perf.now() - t;
    // >>> THE EXPANSION FACTOR IS REPORTED, because the emulated CPU's SPEED depends on
    // it. The game clock is charged per unit of fuel and a unit of fuel is one executed
    // wasm operator, so the console runs at `fuel rate / operators-per-guest-instruction`.
    // Improve this transpiler's codegen and the emulated Vita silently gets faster unless
    // the calibration moves with it - a faithfulness change with nothing to notice it by.
    // Printing it is what turns that into something a capture can be read against.
    let x = built.artifact.expansion;
    web_sys::console::log_1(&JsValue::from_str(&format!(
        "[setup] transpiled wasm {} MB, {} functions (the per-instance funcref table), \
         guest memory {} MB, emulator heap {} MB | code expansion {:.2} wasm operators \
         per guest instruction ({} instructions -> {} operators)",
        built.artifact.wasm.len() / (1024 * 1024),
        built.artifact.funcs.len(),
        linked.mem_bytes / (1024 * 1024),
        wasm_heap_mb(),
        x.per_instruction(),
        x.arm_instructions,
        x.emitted_ops,
    )));
    // >>> AND WHICH BUILD THIS IS, so an A/B can never again be a build measured against
    // itself. The register file is either on its globals or in locals, and the two are
    // indistinguishable from every other counter a run reports - the expansion factor
    // moves (9.87 -> 10.33 on one title) but nothing SAYS which arm produced it.
    web_sys::console::log_1(&JsValue::from_str(&format!(
        "[setup] register promotion {} | {} accesses would become LOCAL ({:.1}% of all \
         operators), {} left on their globals, {} operators of overhead",
        if vitaslop_transpiler::promote_registers() { "ON" } else { "OFF" },
        x.promotion.converted,
        x.promotion.converted_share(x.emitted_ops),
        x.promotion.left,
        x.promotion.overhead,
    )));
    let func_addrs: Vec<u32> = built.artifact.funcs.iter().map(|f| f.addr).collect();
    // Record it for THIS worker too: the main-thread path transpiles in place and runs the
    // guest itself, so it needs the table without any handoff.
    browser_sched::record_function_addresses(func_addrs.clone());
    Ok(Transpiled {
        wasm: built.artifact.wasm,
        mem_pages: built.artifact.mem_pages,
        mirror_off: built.artifact.mirror_off,
        dirty_off: built.artifact.dirty_off,
        ms,
        func_addrs,
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
    crate::logging::install_panic_hook();
    logging::init();
    let perf = global_performance().ok_or_else(|| JsValue::from_str("no performance clock"))?;
    let Mounted { linked, .. } = mount_and_link(source).await?;
    let built = transpile_here(&linked, &perf)?;
    let module = browser_sched::compile_module(&built.wasm).await?;
    let out = js_sys::Object::new();
    js_sys::Reflect::set(&out, &JsValue::from_str("module"), &module)?;
    js_sys::Reflect::set(&out, &JsValue::from_str("memPages"), &JsValue::from_f64(built.mem_pages as f64))?;
    // The wasm-index -> guest-address table, so the RUN worker can name guest functions in
    // a fault backtrace. See `Prebuilt::func_addrs`.
    js_sys::Reflect::set(
        &out,
        &JsValue::from_str("funcAddrs"),
        &js_sys::Uint32Array::from(&built.func_addrs[..]),
    )?;
    js_sys::Reflect::set(
        &out,
        &JsValue::from_str("mirrorOff"),
        &match built.mirror_off {
            Some(v) => JsValue::from_f64(v as f64),
            None => JsValue::NULL,
        },
    )?;
    js_sys::Reflect::set(
        &out,
        &JsValue::from_str("dirtyOff"),
        &match built.dirty_off {
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
    // The host's location provider. Created and registered HERE rather than passed in by
    // each engine, because both engines want exactly the same cell and the exported
    // `worker_location_*` entry points find it through the thread-local registry - so an
    // engine that forgot to pass one would silently have no provider.
    let location: crate::location::SharedLocation =
        Arc::new(Mutex::new(crate::location::LiveLocation::default()));
    crate::location::set_shared_location(location.clone());
    let world = Box::new(BrowserWorld::new(recipe_world, live, location));
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
    // Movie playback goes through WebCodecs, reached by the same platform seam the desktop
    // uses for Media Foundation. A browser without WebCodecs reports "no decoder" when a
    // title opens a movie, and the title skips it - the run does not fail over a movie.
    env.state.video = Box::new(vitaslop_platform::video::H264Factory);
    env.state.audio_dec = Box::new(vitaslop_platform::audio_dec::AacFactory);

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
    let (module, mem_pages, mirror_off, dirty_off, transpile_ms) = match prebuilt {
        Some(p) => {
            web_sys::console::log_1(&JsValue::from_str(&format!(
                "[setup] using a PREBUILT module (transpiled in a throwaway worker); \
                 emulator heap {} MB",
                wasm_heap_mb()
            )));
            browser_sched::record_function_addresses(p.func_addrs);
            (p.module, p.mem_pages, p.mirror_off, p.dirty_off, 0.0)
        }
        None => {
            let built = transpile_here(&linked, &perf)?;
            let module = browser_sched::compile_module(&built.wasm).await?;
            (module, built.mem_pages, built.mirror_off, built.dirty_off, built.ms)
        }
    };

    let main_sp = main_stack_top(linked.base, linked.mem_bytes);
    let sched = browser_sched::BrowserSched::from_linked(
        module,
        &linked.image,
        linked.base,
        mem_pages,
        mirror_off,
        dirty_off,
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

/// Supply the font that STANDS IN for the console's system font.
///
/// `sceFontOpen` / `scePvfOpen` open one of the console's own installed fonts by index. Those
/// are the vendor's assets and are not shipped here, so the open is refused - and a title that
/// renders its strings through the system font then draws them all from an empty glyph atlas,
/// which reaches the screen as BLANK OR BLACK areas where its dynamic text belongs (measured on
/// the golf title: an opaque black rectangle over the club list and black bars over half the
/// course-settings screen).
///
/// The desktop can probe a host font path for a substitute. The browser has no filesystem and
/// no environment, so the bytes have to come in from the page - which is what this is. Call it
/// BEFORE `run_game`/`run_game_worker`; the resolution latches on first use.
///
/// Not calling it is a supported state, not a failure: the refusal is then reported and the
/// title runs with no dynamic text, exactly as on a device with no font installed.
#[wasm_bindgen]
pub fn set_system_font(bytes: &[u8]) {
    vitaslop_runtime::font::system::set_bytes(bytes.to_vec());
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
// ===========================================================================
// THE GUEST'S OWN SAVED STATE
// ===========================================================================
//
// >>> THE EMULATOR NEVER TOUCHES STORAGE HERE. It produces a container's bytes and
// consumes them, and JS puts them somewhere (`web/gamedata.js`, an OPFS directory that is
// not the one the title lives in). That split is deliberate: writing a save is
// ASYNCHRONOUS in a browser, and a guest frame cannot await, so the only way the two can
// meet is for the frame to hand over finished bytes and let the event loop do the rest.
//
// It also means the rule about WHAT may be persisted lives in exactly one place -
// `vitaslop_runtime::gamedata`, which refuses anything that is not on the guest's writable
// mounts - rather than being restated in JS where it could drift away from the Rust.

/// How the run reaches the page's storage: what to restore before the guest starts, and
/// where to hand each later export.
struct Persist {
    /// The container stored for this title, if there is one.
    restore: Option<Vec<u8>>,
    /// Called with a `Uint8Array` of the whole container whenever the save has changed.
    /// The JS side owns the write, including coalescing writes that arrive while one is
    /// still in flight.
    save: Option<js_sys::Function>,
    /// The title id, for the container's README only.
    title: String,
}

impl Persist {
    /// Read the `{ data, save, titleId }` object the worker passes, or an empty one for a
    /// run with no persistence (the main-thread page, the e2e fixtures).
    fn from_js(v: &JsValue) -> Persist {
        if v.is_undefined() || v.is_null() {
            return Persist { restore: None, save: None, title: String::new() };
        }
        let get = |k: &str| js_sys::Reflect::get(v, &JsValue::from_str(k)).unwrap_or(JsValue::UNDEFINED);
        let data = get("data");
        let restore = (!data.is_undefined() && !data.is_null())
            .then(|| js_sys::Uint8Array::new(&data).to_vec());
        let save = get("save").dyn_into::<js_sys::Function>().ok();
        let title = get("titleId").as_string().unwrap_or_default();
        Persist { restore, save, title }
    }
}

// The live run's host, kept so an out-of-band flush (the page going away) can reach it.
//
// A `try_lock`, never a `lock`: the guest can be suspended inside a host call with the
// mutex held, and a page-hide handler that BLOCKED there would hang the worker on the way
// out. A refused flush is at worst the last few seconds of play, and it says so.
thread_local! {
    static PERSIST_HOST: RefCell<Option<Arc<Mutex<VitaEnv>>>> = const { RefCell::new(None) };
    static PERSIST_TITLE: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Export the run's game data RIGHT NOW, for the page to write before it goes away.
/// Returns the container as a `Uint8Array`, or `null` when there is no run, nothing has
/// changed, or the guest is mid-host-call and the host cannot be borrowed.
#[wasm_bindgen]
pub fn flush_game_data() -> JsValue {
    let title = PERSIST_TITLE.with(|t| t.borrow().clone());
    PERSIST_HOST.with(|h| {
        let borrowed = h.borrow();
        let Some(host) = borrowed.as_ref() else { return JsValue::NULL };
        let Ok(mut guard) = host.try_lock() else {
            web_sys::console::warn_1(&JsValue::from_str(
                "[gamedata] could not flush now - the guest is inside a host call. The last \
                 periodic flush stands; nothing is lost beyond what was saved since it.",
            ));
            return JsValue::NULL;
        };
        if !guard.state.game_data_dirty() {
            return JsValue::NULL;
        }
        let bytes = guard.state.game_data().to_zip(&title);
        guard.state.clear_game_data_dirty();
        js_sys::Uint8Array::from(&bytes[..]).into()
    })
}

/// Describe a container without running anything, for the page's upload confirmation.
///
/// The point is that the answer comes from the SAME parser the run uses, refusals and all,
/// so what the user is told at upload time is what will actually be restored. A second
/// implementation in JS would eventually say something the run does not do.
#[wasm_bindgen]
pub fn game_data_describe(zip: &[u8]) -> Result<String, JsValue> {
    let (data, refused) = vitaslop_runtime::gamedata::GameData::from_zip(zip)
        .map_err(|e| JsValue::from_str(&e))?;
    let mut out = data.summary();
    if !refused.is_empty() {
        out.push_str(&format!(
            " - and {} entr(y/ies) that name something outside the guest's saved state, \
             which will be REFUSED: {}",
            refused.len(),
            refused.join(", ")
        ));
    }
    Ok(out)
}

#[wasm_bindgen]
pub async fn run_game(
    canvas: JsValue,
    files: JsValue,
    recipe: String,
    max_frames: u32,
    max_rounds: f64,
) -> Result<String, JsValue> {
    let _ = max_rounds;
    crate::logging::install_panic_hook();
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
    // NO PERSISTENCE ON THIS PATH, deliberately. This is the harness page (`game.html`),
    // which is handed a file map rather than a stored title and so has no title id to key
    // a save by - and a scripted boot that silently wrote a save would make the NEXT run of
    // the same fixture start from a different state, which is the one thing a boot probe
    // must not do. The product path is `run_game_worker`.
    wasm_bindgen_futures::spawn_local(live_loop(
        setup.sched,
        playback,
        report,
        max_frames as u64,
        setup.recipe,
        None,
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
    persist: JsValue,
) -> Result<String, JsValue> {
    crate::logging::install_panic_hook();
    logging::init();
    // >>> BEFORE ANYTHING ALLOCATES A CACHE. Every memory budget in the engine is scaled by what
    // this reports, and a budget read before it lands would be the desktop-sized one.
    report_device_memory();
    // >>> THE PHASE CLOCK BELONGS TO THIS WORKER, AND IT WAS BEING INSTALLED IN THE WRONG ONE.
    //
    // `transpile_here` also calls this - and `transpile_here` normally runs in a THROWAWAY
    // worker, so its `set_clock` died with that worker and the RUN worker never had one. On
    // `wasm32` `perf::scope` returns `None` without a clock, so **every phase timer in the
    // browser was silently inert**: a run with `VITASLOP_PERF=1` printed the guest-access
    // counts (which need no clock) and an EMPTY phase table, which reads as "no phase costs
    // anything" rather than "nothing was measured".
    // [[vitaslop-instrument-failure-imitating-its-subject]]
    vitaslop_runtime::perf::set_clock(browser_sched::perf_clock);

    let live: Arc<Mutex<InputState>> = Arc::new(Mutex::new(InputState::default()));
    // Register the shared input cell so the page's forwarded pointer/keyboard messages
    // (via the exported worker_input_* functions) reach this run's world.
    input::set_worker_input(live.clone());
    // `prebuilt` is a module a throwaway worker already transpiled and compiled. Passing
    // one keeps the transpile's ~463 MB peak out of THIS worker's heap, which it could
    // never give back - see the note in `setup_game`.
    let setup =
        setup_game(files, &recipe, live, Prebuilt::from_js(&prebuilt)?, &audio_ring).await?;

    // >>> THE SAVE GOES BACK IN BEFORE THE GUEST RUNS, AND THERE IS NO SECOND CHANCE.
    //
    // `setup_game` mounts, links and compiles; no guest instruction has executed yet, and
    // the title's first frame is where it opens its savedata. Restoring any later would
    // hand a running title a filesystem that changed under descriptors it already holds.
    let persist = Persist::from_js(&persist);
    if let Some(bytes) = persist.restore.as_ref() {
        match vitaslop_runtime::gamedata::GameData::from_zip(bytes) {
            Ok((data, refused)) => {
                let report = setup.sched.host.lock().unwrap().state.restore_game_data(&data);
                web_sys::console::log_1(&JsValue::from_str(&format!("[gamedata] {report}")));
                if !refused.is_empty() {
                    // Loud, not swallowed: an entry outside the guest's own saved state is
                    // either a corrupted container or one that was built to reach the
                    // installed title, and both are worth saying out loud.
                    tracing::warn!(
                        target: "vitaslop::gamedata",
                        refused = refused.len(),
                        first = refused[0].as_str(),
                        "this title's stored game data names path(s) outside the guest's own \
                         saved state. They were REFUSED - nothing outside a savedata mount \
                         is ever written - and the rest of the container was restored."
                    );
                }
            }
            // A save that will not parse must not take the run down: the title boots with a
            // fresh profile, which is what a console does with a corrupt save, and the
            // reason is on the page rather than in a console nobody is holding.
            Err(e) => tracing::warn!(
                target: "vitaslop::gamedata",
                error = %e,
                "this title's stored game data could not be read, so the run starts with \
                 NOTHING restored. The stored container is left alone - download it from \
                 the launcher if you want to keep it - and this run will overwrite it when \
                 the title next saves."
            ),
        }
    }
    // Reachable from `flush_game_data`, for the flush the page asks for on its way out.
    PERSIST_HOST.with(|h| *h.borrow_mut() = Some(setup.sched.host.clone()));
    PERSIST_TITLE.with(|t| *t.borrow_mut() = persist.title.clone());

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
        Some(persist),
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
    persist: Option<Persist>,
) {
    let mut eval = recipe.as_ref().map(|r| vitaslop_runtime::recipe_eval::RecipeEval::new(r, None));
    // >>> ONLY FOLD THE DETERMINISM SIGNATURE WHEN SOMETHING WILL READ IT.
    //
    // This function is the ONLY place in the browser that asks for it, and only when it is
    // evaluating a recipe (see the `eval.finish(frames, sig)` at the end of the run). A live
    // player session has no recipe - the user's own device captures read `recipe: (none)` - and
    // folding hashes every retired scene's vertices, indices and uniforms: about 3 MB a frame on
    // this title's race, MEASURED at 7.9% of the whole frame, for a number nobody asks for.
    //
    // `Capture::signature` refuses rather than returning a partial hash when this was off, so
    // the failure mode of getting this wrong is a loud one.
    //
    // >>> AND A RECIPE IS NOT ITSELF A READER. The only consumer of the number is
    // `RecipeEval::finish`, which uses it if and only if the recipe DECLARES `@sig`; every
    // other recipe run folded 3 MB a frame to print a hash nothing compared. The user drives
    // this page with a recipe selected from the menu - that is what the recipe dropdown IS -
    // so "has a recipe" was true for every session anyone has ever measured, including every
    // browser number in the notes. Gated on the declaration instead, with a knob to ask for it
    // when the point of the run is to LEARN the signature and bless it into a recipe.
    let want_sig = eval.is_some()
        && (recipe.as_ref().and_then(|r| r.meta.sig).is_some()
            || vitaslop_runtime::knobs::flag("VITASLOP_SIGNATURE"));
    sched.host.lock().unwrap().state.capture.set_signature_wanted(want_sig);
    // `VITASLOP_SIGNATURE_EVERY=<n>`: how often the running signature is printed, mirroring the
    // desktop knob of the same name. 0 (the default) prints none.
    let sig_every: u64 = vitaslop_runtime::knobs::var("VITASLOP_SIGNATURE_EVERY")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);
    // The one frame whose per-SCENE digests are printed, if any. See the print site.
    // `<frame>` or `<frame>:<draw>` - split BEFORE parsing, or the two-part form silently
    // disables the whole diagnostic.
    let frame_digest_at: Option<u64> = vitaslop_runtime::knobs::var("VITASLOP_FRAME_DIGEST")
        .ok()
        .and_then(|v| v.split(':').next().and_then(|n| n.trim().parse().ok()));
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
    // >>> THE LAST PUBLISHED RENDER SPLIT, SO THE HEARTBEAT CARRIES IT TOO.
    //
    // The split is recomputed every `PERF_WINDOW` presents and written to the PAGE, where it
    // overwrites its predecessor - so a run's cost is only ever readable as "what it is now",
    // and a run that starts fast and ends slow reads exactly like a run that was always slow.
    // The heartbeat is the one line that goes to the CONSOLE on a cadence, so a run's cost over
    // TIME is a thing the log has only if the split rides along on it.
    let mut last_perf = String::new();
    // >>> WHERE THE WALL CLOCK WENT, WHICH NO OTHER COUNTER ON THIS PAGE ACCOUNTS FOR.
    //
    // `cpu` is per GUEST FRAME and `render` is per PRESENT, and when the loop runs more than
    // one frame per present neither of them - nor their sum - is a second of wall clock. A
    // race window here measured 11.6 ms of cpu and 2.8 of render at 28 fps and 2.0 frames per
    // present: 728 ms of the second is described and 272 ms is not, and "27% of the run is
    // somewhere else" is not a thing any existing line could have said.
    //
    // So the pacing accounts for ITSELF: ticks, the sleep it asked for, the sleep it actually
    // got, and how many guest frames each tick ran. That last one is the whole pacing policy
    // in one number - a loop that runs two frames and presents once is throwing away half the
    // pictures it computed, and it does so silently.
    let mut ticks = 0u32;
    let mut sleep_ms = 0.0f64;
    // What the loop ASKED to wait, against `sleep_ms` which is what it got. The difference is
    // the pacing error, and with one guest frame per tick every millisecond of it is emulated
    // time the run never gets back: the tick period becomes `work + sleep + overshoot` while
    // the guest advances by exactly one frame either way.
    let mut sleep_asked_ms = 0.0f64;
    // The movie counters as of the last panel report, so the panel can publish a RATE over the
    // window rather than an average over a run the movie was not playing for all of.
    let mut last_movie = (0u64, 0u64, 0u64, 0u64);
    // Artificial per-frame guest cost, for exercising the behind-the-clock branch of this loop
    // on a machine that is not behind. See where it is spent, below.
    let slow_frame_us: f64 = vitaslop_runtime::knobs::var("VITASLOP_SLOW_FRAME_US")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0.0);
    let mut tick_span_ms = 0.0f64;
    let mut acc_at_tick_ms = 0.0f64;
    let mut saturated_ticks = 0u32;
    // Guest frames run by the tick loop itself, and the largest number any one tick ran.
    //
    // Counted rather than derived from `cpu_frames / ticks`: that ratio read 2.00 in a build
    // whose loop is HARD-CAPPED at one frame per tick, which is not a number the code can
    // produce - so one of the two counters is measuring something other than what its name
    // says, and a ratio cannot say which. This pair can.
    let mut tick_frames = 0u64;
    let mut tick_frames_max = 0u32;
    // Emulated time charged to the wall-clock budget, summed over the frames the tick loop
    // ran. Divided by those frames it is the title's own frame period in GAME time - 16.7 ms
    // for a 60 Hz title, 33.3 for a 30 Hz one - which is the number the pacing now spends
    // against the wall clock, so a run that is going too fast or too slow can be read here
    // rather than inferred from the frame rate.
    let mut charged_ms = 0.0f64;
    let mut idle_ticks = 0u32;
    // Cumulative guest-store epoch wraps at the start of the current perf window, so the
    // window's own count is a difference rather than a running total that only grows.
    let mut epoch_wraps_at_window_start = 0u64;
    let mut epoch_rebases_at_window_start = 0u64;
    // >>> THE MOST EXPENSIVE FRAMES OF THE WHOLE RUN, AND THE FRAME/PRESENT TOTALS.
    //
    // Every other instrument on this page describes a WINDOW - the last thirty presents - which
    // is the right shape for a steady state and useless for the one question a user actually
    // asks: "it hung from frame 1 to frame 600, what was it doing?" By the time the panel can be
    // read, the window has moved past the thing that needs explaining, and the per-frame console
    // lines (`VITASLOP_BROWSER_HEARTBEAT_MS=0`) are on a console a phone does not show.
    //
    // These are cumulative for the run, so ONE panel grabbed after the stall answers it: a
    // single enormous frame, a few hundred merely slow ones, and presents being dropped are
    // three different defects with three different fixes, and the top-N list plus the
    // frames-against-presents ratio separates them. `(frame, guest ms, host calls)`, kept
    // smallest-first so the cheapest is always at index 0.
    const SLOWEST_KEPT: usize = 12;
    let mut slowest: Vec<(u64, f64, u64)> = Vec::new();
    let mut frames_total = 0u64;
    let mut presents_total = 0u64;
    // The display-flip count when real-time pacing began, so the movie report's
    // per-displayed-frame ratio has the same window its numerator does.
    let mut frames_at_pace_start = 0u64;

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

    // >>> THE GAME'S OWN SAVE, WRITTEN OUT WHILE IT PLAYS.
    //
    // Checked once a tick and answered by ONE BOOL on a run that is not saving (see
    // `VitaState::game_data_dirty`), so a title that never writes pays a comparison per
    // frame and nothing else.
    //
    // The interval is a floor on how often the container is REBUILT, not a delay before
    // the first write: a title that saves and is closed a second later loses nothing,
    // because the page flushes on its way out too. What it bounds is a title that rewrites
    // its save every frame - an autosave in a menu loop is exactly that - turning every
    // frame into a zip.
    const SAVE_MIN_MS: f64 = 3_000.0;
    let mut last_save_at = f64::NEG_INFINITY;
    let mut saves = 0u32;
    let mut save_ms = 0.0f64;
    let mut save_bytes = 0usize;
    let mut save_error: Option<String> = None;

    let now = || perf.as_ref().map(|p| p.now()).unwrap_or(0.0);
    let mut acc = 0.0f64;
    let mut last = now();
    let mut last_console = 0.0f64;
    // Whether this run was started with debug capture on (`VITASLOP_DEBUG_CAPTURE`). Read ONCE,
    // here: it decides whether the expensive instruments record for the whole run, and a run that
    // changed its own instrumentation part-way would publish two incomparable halves.
    let debug_capture = vitaslop_runtime::knobs::flag("VITASLOP_DEBUG_CAPTURE");
    if debug_capture {
        browser_sched::set_host_call_timing(true);
        vitaslop_runtime::vita::set_callsite_profiling(true);
        web_sys::console::log_1(&JsValue::from_str(
            "[perf] DEBUG CAPTURE is on: host calls are timed and profiled by NID. This costs \
             roughly a doubling of the guest frame cost - the ratios it reports are valid, the \
             absolute frame times are not.",
        ));
    }

    'run: loop {
        // How long until a frame is actually DUE: the accumulator carries what has already
        // accrued, so the wait is the rest of one frame's budget. Fast-forward asks for zero -
        // it is deliberately unpaced - and so does a machine that is behind, whose `acc` is
        // already at or over the budget. See [`next_tick_in`].
        let due_in = if sched.core.frames() < ff_to { 0.0 } else { FRAME_MS - acc };
        let sleep_from = now();
        next_tick_in(due_in).await;
        let t = now();
        // The pacing's own accounting, before `acc` is consumed by the frames below - see
        // the declarations. `due_in` is what was ASKED for and `t - sleep_from` what the
        // host gave, and the gap between them is the tick floor this loop cannot go under.
        ticks += 1;
        sleep_ms += t - sleep_from;
        sleep_asked_ms += due_in.max(0.0);
        tick_span_ms += t - last;
        let acc_raw = acc + (t - last);
        acc_at_tick_ms += acc_raw;
        if acc_raw >= MAX_CATCHUP_MS {
            saturated_ticks += 1;
        }
        acc = acc_raw.min(MAX_CATCHUP_MS);
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
            // The movie decoder's counters describe PACED play only - see
            // `avcdec::reset_movie_counters`. A fast-forward never returns to the JS event
            // loop, so a callback-driven decoder cannot answer during one and its whole
            // backlog would otherwise be charged to the frames that follow.
            vitaslop_runtime::vita::avcdec::reset_movie_counters();
            frames_at_pace_start = sched.core.frames();
        }
        was_fast = fast;

        // >>> ONE GUEST FRAME PER TICK, AND THEREFORE ONE PRESENT PER GUEST FRAME.
        //
        // This tick presents at most the newest scene, so every frame beyond the first is a
        // picture computed and thrown away. The rule used to be "catch up freely unless the
        // last iteration overran its 16.7 ms budget", and it cost about half the frame rate
        // on every machine measured, INCLUDING ones with headroom.
        //
        // MEASURED here (headed Chrome, desktop GPU, one title's race, 452 draws):
        //   fps 31 shown of 60 run - 100% emulated speed, 48% of frames DISCARDED
        //   period 33.2 ms, of which slept 9.9 | 2.00 guest frames/tick, 1.00 presents/tick
        //   acc 37.4 ms at tick, 0 saturated
        // A guest frame was 11.8 ms and its present 2.4, so ONE of each is 14.2 ms inside a
        // 16.7 ms budget - the machine had headroom and was still showing half the frames.
        //
        // # Why the old rule produced that, and why no threshold fixes it
        // `acc` is a wall-clock accumulator: it says "33.2 ms passed, two frames are owed".
        // That is true, and it is true BECAUSE the tick ran two frames. The loop therefore has
        // a stable fixed point at every N: at N=2 it produces exactly 60 guest frames a second,
        // so `acc` never drains and nothing pushes it back to N=1. One hitch (the boot frame,
        // leaving the fast-forward) settles it into N=2 and it stays there for the whole run,
        // with `saturated` reading 0 the entire time - the old comment's "acc pins at the cap"
        // diagnosis describes a different failure from the one that actually happens.
        //
        // The degeneracy is the bug, so the fix is to remove the choice rather than to tune a
        // threshold: N=1 is the lowest fixed point and it is never worse. If a frame plus its
        // present fits in 16.7 ms, N=1 sustains full speed and N=2 is pure loss.
        //
        // If it does not fit, catching up cannot help either, and that is now MEASURED here
        // rather than argued - see below.
        //
        // >>> CATCH-UP WAS RE-TRIED, WITH A DEVICE'S NUMBERS, AND IT IS STILL WRONG. DO NOT
        // >>> RE-TRY IT A THIRD TIME.
        //
        // A phone reported `fps 39 (65% speed)` on this title's race with `acc` pinned at the
        // 67 ms ceiling, 17 of 30 ticks saturated - which reads exactly like a loop forbidden
        // to catch up, and the user's report ("my car simply can't catch up with the others")
        // is what 65% speed feels like from the inside. So N=2-while-saturated was written, and
        // then run against `VITASLOP_SLOW_FRAME_US=15000`, which reproduces that device's frame
        // cost on this desktop:
        //
        //   N=2  cpu 20.7 ms/frame | period 44.7 ms | 2.00 frames/tick, 1.00 presents/tick
        //        23 fps shown of 45 run, 75% speed, 30 of 30 ticks saturated
        //   N=1  cpu 20.7 ms/frame | period 22.4 ms | 1.00 frames/tick, 1.00 presents/tick
        //        44 fps shown of 45 run, 75% speed
        //
        // **The same 75%, at half the frame rate.** And the arithmetic says it will always come
        // out this way. With a guest frame costing C and a present costing R, a tick that runs N
        // frames and presents once takes `N*C + R`, so
        //
        //     speed  ∝  N / (N*C + R)      rises with N, towards a ceiling of 1/C
        //     fps    =  1 / (N*C + R)      falls with N, in proportion
        //
        // Every point of speed catch-up buys is bought by amortising ONE present over more
        // frames, so the most it can ever return is R/C - and it charges half the frame rate to
        // return it. Here R/C is 1.6/20.8: N=2 offered 2% of speed for 46% of the frame rate.
        // On a phone, where R is a much larger share (a device measured 7.6 ms of texture upload
        // inside a 19.2 ms encode), the offer improves to perhaps 15% for the same 46% - still
        // the wrong side of the trade, and now with a reason rather than a reading.
        //
        // The lever that actually moves speed is C and R themselves. There is no pacing rule
        // that makes a slow frame fast.
        //
        // Which also relocates the phone's missing speed. Its work was 10.2 ms against a 19.5 ms
        // period on the front end - it had 40% headroom and was still at 86% - so what it lost
        // was not frames it was forbidden to run, but time spent inside the tick's own wait.
        // That is what `slept` against `asked` on the PACING line now names, and what the
        // deadline walk in `next_tick_in` is for.
        //
        // MEASURED on a phone (PowerVR D-series, one title's main screen): 72.8 ms of guest CPU
        // per frame against a 13.2 ms render. Four frames per tick made that 4 presents/s while
        // the guest advanced at 14 - so skipping three presents bought 11% of guest speed and
        // cost three quarters of the frame rate the user could see. One frame per tick shows all
        // 14.
        //
        // What is given up: a machine that is momentarily behind no longer sprints to catch the
        // wall clock. It cannot - a sprint is only available to a machine with spare time, and
        // one with spare time does not need it. `acc` still carries the deficit (capped at
        // `MAX_CATCHUP_MS`) so the emulated clock does not silently lose the hitch.
        //
        // Fast-forward is exempt: it presents NOTHING by design, so there is no frame rate to
        // protect and running as many frames per tick as the budget allows is its entire job.
        let mut latest = None;
        let mut frames_this_tick = 0u32;
        // Named `_per_tick` because the run's own frame LIMIT is also called `max_frames` in
        // this scope, and shadowing it here would silently end the run at the first tick.
        let max_frames_per_tick = if fast { u32::MAX } else { 1 };
        while acc >= FRAME_MS && frames_this_tick < max_frames_per_tick {
            frames_this_tick += 1;
            // >>> WHAT A FRAME COSTS THE ACCUMULATOR IS THE EMULATED TIME IT ADVANCES, NOT
            // >>> ONE DISPLAY PERIOD. See where it is subtracted, below the run.
            let clock_before_us = { sched.host.lock().unwrap().state.now_us() };
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
            // >>> A SLOW DEVICE, ON THIS MACHINE (`VITASLOP_SLOW_FRAME_US`).
            //
            // Every pacing question here is a question about a machine that CANNOT hold 60, and
            // this desktop always can: a race frame costs 5.4 ms of a 16.7 ms budget, so the
            // accumulator never saturates and the whole behind-the-clock branch of this loop is
            // dead code locally. The answers therefore came from a phone, one panel dump at a
            // time, with the person holding it in the loop - and a pacing rule shipped on that
            // basis made the game run at 65% speed for a week.
            //
            // Chrome's own `Emulation.setCPUThrottlingRate` was tried first and does NOT reach a
            // dedicated worker: 4x throttling left `cpu 5.6 ms/frame` against an unthrottled
            // 5.4. This burns the time inside the frame's own measurement window instead, so it
            // lands in `cpu` exactly as guest work would and the pacing loop cannot tell the
            // difference. A phone measured ~20 ms a frame where this machine measures 5.4, so
            // `VITASLOP_SLOW_FRAME_US=15000` is roughly that device.
            //
            // It is a spin, deliberately: a sleep would return to the event loop and change the
            // very thing under test.
            // Not during a fast-forward: that runs unpaced and presents nothing, so slowing it
            // tests no pacing and only makes the run take longer to reach the part that does.
            if slow_frame_us > 0.0 && !fast {
                let until = now() + slow_frame_us / 1000.0;
                while now() < until {}
            }
            let c1 = now();

            // >>> A FRAME COSTS THE WALL-CLOCK BUDGET THE EMULATED TIME IT ADVANCED, AND ON A
            // >>> 30 Hz TITLE THAT IS TWO DISPLAY PERIODS, NOT ONE.
            //
            // This used to charge a flat `FRAME_MS` per frame, i.e. it paced ONE GUEST FLIP
            // per 16.7 ms of wall clock. That is only right for a title whose frame is one
            // display period. A title that waits for TWO vblanks a frame - the console's other
            // characteristic rate, and what this engine's own `pace_flip` models - advances
            // 33.3 ms of game clock per flip, so a flip every 16.7 ms ran it at TWICE real
            // time. MEASURED on a retail 30 fps sports title's opening, in the browser:
            // `fps 53 (177% speed)`, `2.04 display periods per displayed frame`, and audio
            // produced at the same 1.8x - `overrun 2,166,784` frames (45 s of sound the ring
            // could not take) beside `underrun 260,224`, which is heard as the hiccup it is.
            // Nothing on the guest side was wrong: the clock per frame was the whole number it
            // should be, and the loop simply ran that clock too fast against the wall.
            //
            // So the budget is spent in GAME TIME. `now_us` is the emulated clock, one frame
            // advances it by however many display periods the title's own limiter waits for,
            // and subtracting that keeps the emulated clock tracking the wall clock 1:1 at any
            // title frame rate - which is what `% speed` measures and what the audio ring is
            // filled at.
            //
            // The deficit is floored at the same `MAX_CATCHUP_MS` that caps the surplus: the
            // boot frame advances the clock by whole seconds, and without a floor the loop
            // would then sit idle for those seconds "waiting for the wall to catch up" with a
            // blank canvas. Neither direction is allowed to bank more than four frames.
            let advanced_ms = {
                let after = sched.host.lock().unwrap().state.now_us();
                after.saturating_sub(clock_before_us) as f64 / 1000.0
            };
            charged_ms += advanced_ms.max(FRAME_MS);
            acc = (acc - advanced_ms.max(FRAME_MS)).max(-MAX_CATCHUP_MS);

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
                // THROUGH the capture, not around it: a raw `mem::take` here walks the
                // frame's scenes past the determinism fold, and the run then reports the
                // EMPTY-fold hash as though it were a real one. See `Capture::take_frame_scenes`.
                // `VITASLOP_FRAME_DIGEST=<frame>`: at ONE frame, a digest per scene, printed
                // BEFORE the drain - after it there are no scenes left to describe. The
                // desktop prints the same line from `recipe_runner`, so the two logs diff.
                if frame_digest_at == Some(sched.core.frames()) {
                    // `VITASLOP_FRAME_DIGEST=<frame>:<draw>` also dumps that draw's vertex
                    // FLOATS - the end of the bisect, where the question stops being "which
                    // draw" and becomes "which number". See `Capture::draw_vertex_floats`.
                    let want_draw: Option<usize> = vitaslop_runtime::knobs::var("VITASLOP_FRAME_DIGEST")
                        .ok()
                        .and_then(|v| v.split(':').nth(1).and_then(|d| d.trim().parse().ok()));
                    let shapes = cap.frame_scene_shapes();
                    let d: Vec<String> = cap
                        .frame_scene_digests()
                        .iter()
                        .zip(shapes.iter())
                        .map(|(h, (n, a, fmt))| format!("{h:#018x}/draws={n}/surf={a:#x}:{fmt}"))
                        .collect();
                    web_sys::console::log_1(&JsValue::from_str(&format!(
                        "framedigest f{} [{}]",
                        sched.core.frames(),
                        d.join(" ")
                    )));
                    // ...and the LAST held scene draw by draw - which DRAW of the pass,
                    // and which of its three inputs. See `Capture::scene_draw_digests`.
                    let last = d.len().saturating_sub(1);
                    if let Some(di) = want_draw {
                        if let Some((lanes, hs)) = cap.draw_lane_hashes(last, di) {
                            let l: Vec<String> = hs
                                .iter()
                                .enumerate()
                                .map(|(i, h)| format!("{i}:{h:#010x}"))
                                .collect();
                            web_sys::console::log_1(&JsValue::from_str(&format!(
                                "lanehash f{} s{last} d{di} lanes={lanes} {}",
                                sched.core.frames(),
                                l.join(" ")
                            )));
                        }
                        if let Some((stride, len, vals)) = cap.draw_vertex_floats(last, di, 64) {
                            web_sys::console::log_1(&JsValue::from_str(&format!(
                                "drawbytes f{} s{last} d{di} stride={stride} len={len} {vals:?}",
                                sched.core.frames()
                            )));
                        }
                    }
                    if let Some(draws) = cap.scene_draw_digests(last) {
                        for (i, (hv, hc, hi, hu, nu)) in draws.iter().enumerate() {
                            web_sys::console::log_1(&JsValue::from_str(&format!(
                                "drawdigest f{} s{last} d{i} verts={hv:#018x} vertsNaNc={hc:#018x} idx={hi:#018x} unis={hu:#018x} nunis={nu}",
                                sched.core.frames()
                            )));
                        }
                    }
                }
                let scenes = cap.take_frame_scenes();
                cap.trace.clear();
                cap.trace_thid.clear();
                // >>> TAKEN, NOT DISCARDED. These are the buffers the guest FLIPPED while this
                // frame's scenes were being captured, which is its own statement of what a
                // display buffer is - the renderer uses it to recognise a frame whose scenes
                // straddle a flip, where "the display is the last scene's target" drops a whole
                // pass. See `GxmRenderer::set_presented`.
                let presents = std::mem::take(&mut cap.presents);
                (scenes, presents)
            };
            let (frame_scenes, frame_presents) = frame_scenes;
            if !frame_scenes.is_empty() {
                latest = Some((frame_scenes, frame_presents));
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
                // >>> THE RUNNING SIGNATURE, the browser half of the desktop's
                // `VITASLOP_SIGNATURE_EVERY` (`recipe_runner::signature_trace_interval`).
                //
                // Two engines that disagree at the END disagree from some FIRST frame onward,
                // and the end-of-run number cannot say which. The desktop has printed this for
                // a while; without the same line here a browser-only divergence can only be
                // bisected by re-running the pair, which on this title is half an hour a
                // halving. Same text (`sigtrace f<frame> <hash>`) so the two logs diff directly.
                //
                // Inert unless a signature is actually being folded: `Capture::signature`
                // refuses a partial hash, and printing a number nothing folded is how an EMPTY
                // fold (the FNV basis) gets read as a DIFFERENT fold.
                if want_sig && sig_every > 0 && frames % sig_every == 0 {
                    // The COUNTS ride along with the hash - see `Capture::stream_counts` for
                    // why a differing signature is only half an answer without them.
                    let (sig, scenes, egress, calls) = {
                        let host = sched.host.lock().unwrap();
                        let (sc, eg, ca) = host.state.capture.stream_counts();
                        (host.state.capture.signature(), sc, eg, ca)
                    };
                    web_sys::console::log_1(&JsValue::from_str(&format!(
                        "sigtrace f{frames} {sig:#018x} scenes={scenes} egress={egress} calls={calls}"
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
            // The run's slowest frames, kept as they happen - see `slowest`. Counted from the
            // FIRST frame, warmup included: the boot frame is the whole point of the list, and
            // it is the one every other counter on this page deliberately excludes.
            frames_total += 1;
            // The rate meter counts presents; tell it about the flips too, so the speed it
            // publishes is the emulated one. See `FpsMeter::note_guest_frames`.
            playback.fps.note_guest_frames(1);
            // ...and the emulated clock, which is what the SPEED percentage is made of - see
            // `FpsMeter::note_clock`.
            playback.fps.note_clock(sched.host.lock().unwrap().state.now_us());
            if slowest.len() < SLOWEST_KEPT {
                slowest.push((frames, c1 - c0, frame_calls));
                slowest.sort_by(|a, b| a.1.total_cmp(&b.1));
            } else if c1 - c0 > slowest[0].1 {
                slowest[0] = (frames, c1 - c0, frame_calls);
                slowest.sort_by(|a, b| a.1.total_cmp(&b.1));
            }
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
                // Instances INSTANTIATED against instances reused from the pool. A title
                // that creates a guest thread per frame instantiates the whole transpiled
                // module per frame without the pool, and each instance is a funcref table
                // with an entry per translated function - which is what killed the
                // renderer at frame 22. `created` going flat while `reused` climbs is the
                // only evidence that the pool is doing its job.
                let (inst_new, _pooled, inst_reused) = browser_sched::instance_stats();
                // The FUEL the scheduler actually billed, next to the preemption count that is
                // supposed to be proportional to it. A preemption fires when the emitted counter
                // reaches `fuel_interval()`, so `fuel / on_fuel` must sit near that interval;
                // when it collapses, the preemptions are firing on a counter that is not
                // tracking work and each one costs a full JSPI suspend for nothing. Nothing else
                // on this line can tell "this frame did an enormous amount of guest work" apart
                // from "this frame suspended thousands of times and did none", and those are
                // opposite bugs.
                // Vblank wait loops parked rather than spun through. On the running line
                // because that is the one a phone run is read from, and because a title
                // whose spin guard never fires and a build where it is switched off look
                // identical from every other number here.
                let vparks = vitaslop_runtime::host::vblank_spin_parks();
                let (fuel_total, fuel_samples, fuel_max) = sched.core.fuel_report();
                let (raw_last, raw_min) = browser_sched::raw_fuel_stats();
                let (unbilled_none, unbilled_idle) = sched.core.unbilled_report();
                web_sys::console::log_1(&JsValue::from_str(&format!(
                    "[live] {status} | clock {:.2}s over {flips} flips ({quanta} quanta, \
                     {:.1} us/frame; {:.2}s quanta + {:.2}s idle) \
                     | preempt {preempts} ({on_fuel} on fuel, {vparks} vblank spins PARKED) \
                     | fuel {fuel_total} over {fuel_samples} (max {fuel_max}, \
                     raw {raw_last}/min {raw_min}, unbilled {unbilled_none}+{unbilled_idle})                      | wasm heap {} MB \
                     | jspi {susp} susp, {starts} stacks, {abandoned} abandoned, \
                     {released} released | instances {inst_new} new, {inst_reused} reused \
                     | threads {live_threads} live, {finished_threads} finished \
                     | texture working set {} MB | {last_perf}",
                    clock_us as f64 / 1e6,
                    if flips > 0 { clock_us as f64 / flips as f64 } else { 0.0 },
                    clk_q as f64 / 1e6,
                    clk_idle as f64 / 1e6,
                    wasm_heap_mb(),
                    vitaslop_platform::gpu::texture_working_set_bytes() / (1024 * 1024)
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
                    // >>> A GUEST TRAP IS A FATAL OUTCOME AND BELONGS IN THE FATAL BOX.
                    // It is not a Rust panic, so the panic hook never sees it, and the
                    // status line it lands on is rebuilt away by the next panel refresh.
                    // On the device that meant a full guest fault - backtrace and all -
                    // showed up only in the status text and had to be copied out by hand.
                    // The wasm indices in that backtrace are named here too: on their own
                    // they are module numbers nobody holding a phone can resolve.
                    if let RunReport::Error(why) = &report_step {
                        crate::logging::report_fatal(&format!(
                            "GUEST FAULT at frame {frames} - the run is over.\n{}",
                            browser_sched::name_guest_frames(why)
                        ));
                    }
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
                        let sig = if want_sig {
                            sched.host.lock().unwrap().state.capture.signature()
                        } else {
                            u64::MAX
                        };
                        eval.finish(frames, sig);
                        // Say WHICH of the two things happened rather than printing an
                        // all-ones hash that reads like a real one. A run that was meant to
                        // learn the signature and did not fold has to be told so here, at the
                        // only place it would ever have looked.
                        let sig_text = if want_sig {
                            format!("sig {sig:#018x}")
                        } else {
                            "sig NOT FOLDED (this recipe declares no @sig; set \
                             VITASLOP_SIGNATURE=1 to compute one)"
                                .to_string()
                        };
                        web_sys::console::log_1(&JsValue::from_str(&format!(
                            "[recipe] {} | {sig_text}",
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
                        let display = sched.host.lock().unwrap().state.display_size();
                        // The run is already ending on the line below, so this last frame's
                        // outcome changes nothing - but `present` reports a lost device itself
                        // before returning, so nothing is swallowed by ignoring it here.
                        let _ = playback.present(&scene.0, display, &scene.1);
                    }
                    break 'run;
                }
            }
        }

        // What this tick actually ran, counted at the one place that knows - see the
        // declarations for why this is not derived from the per-frame counters.
        tick_frames += frames_this_tick as u64;
        tick_frames_max = tick_frames_max.max(frames_this_tick);
        if frames_this_tick == 0 {
            idle_ticks += 1;
        }

        // Present at most one (the newest) frame per tick, and fold its render time into
        // the rolling perf report.
        //
        // >>> WRITE THE SAVE OUT, IF THE GUEST CHANGED IT.
        //
        // Here rather than inside the frame loop: this is a point where no lock is held and
        // the guest is between frames, so the export sees a filesystem no host call is
        // half-way through changing. Skipped during a fast-forward, which is not play and
        // whose whole point is to reach a later frame quickly.
        if let Some(p) = persist.as_ref().filter(|p| p.save.is_some() && !fast) {
            if t - last_save_at >= SAVE_MIN_MS {
                let dirty = { sched.host.lock().unwrap().state.game_data_dirty() };
                if dirty {
                    last_save_at = t;
                    let s0 = now();
                    let bytes = {
                        let mut host = sched.host.lock().unwrap();
                        let zip = host.state.game_data().to_zip(&p.title);
                        // Cleared once the bytes EXIST, not once JS has stored them: the flag
                        // records that the guest changed something, and this container now
                        // holds that change whatever happens to it downstream.
                        host.state.clear_game_data_dirty();
                        zip
                    };
                    let arr = js_sys::Uint8Array::from(&bytes[..]);
                    match p.save.as_ref().expect("filtered above").call1(&JsValue::UNDEFINED, &arr) {
                        Ok(_) => {
                            saves += 1;
                            save_bytes = bytes.len();
                            save_ms += now() - s0;
                        }
                        // A storage failure is the one thing in this loop the user MUST be
                        // told about - it is their progress - and it is otherwise completely
                        // silent (a full quota, an evicted origin). It goes to the panel as
                        // well as the console, because a phone has no console.
                        Err(e) => {
                            let text = e.as_string().unwrap_or_else(|| format!("{e:?}"));
                            tracing::error!(
                                target: "vitaslop::gamedata",
                                error = %text,
                                "THIS TITLE'S SAVE COULD NOT BE WRITTEN TO THIS BROWSER'S \
                                 STORAGE. Play continues, but progress from here will be \
                                 lost when the tab closes."
                            );
                            save_error = Some(text);
                        }
                    }
                }
            }
        }

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
            let display = sched.host.lock().unwrap().state.display_size();
            // >>> A RENDERER THAT CANNOT DRAW ENDS THE RUN HERE.
            //
            // The alternative is what this used to do: keep executing the guest, keep
            // capturing scenes, keep encoding command buffers, at full cost, against a canvas
            // that will never change again. The fatal text goes to the copyable box a device
            // report is taken from, because the one thing a black screen cannot do is say why
            // it is black [[vitaslop-fast-fail-no-silent-success]].
            // NOT `presents` - that name is already a running counter in this scope, and
            // shadowing it here silently retyped it.
            let (scene, flips) = scene;
            if let PresentOutcome::Fatal(why) = playback.present(&scene, display, &flips) {
                crate::logging::report_fatal(&format!(
                    "RENDERER FAULT at frame {} - the run is over.\n{why}",
                    sched.core.frames()
                ));
                report.emit_final(
                    "status",
                    &format!(
                        "frame {} ENDED - the renderer cannot draw (live via WebGPU)",
                        sched.core.frames()
                    ),
                );
                break 'run;
            }
            let r1 = now();
            // Counted from the first present, warmup included, for the same reason the frame
            // total is: what this is read against is `frames_total`, and a ratio whose two
            // halves start counting at different frames is not a ratio.
            presents_total += 1;
            if sched.core.frames() > WARMUP_FRAMES {
                render_ms += r1 - r0;
                presents += 1;
            }
            if presents >= PERF_WINDOW {
                let cpu_avg = if cpu_frames > 0 { cpu_ms / cpu_frames as f64 } else { 0.0 };
                let render_avg = render_ms / presents as f64;
                let cpu_fps = if cpu_avg > 0.0 { 1000.0 / cpu_avg } else { 0.0 };
                // ...and WHERE the render time went. "render 32 ms" names no cause, and
                // the three parts have three different fixes: build is Rust in wasm,
                // encode is one wasm/JS boundary crossing per WebGPU call (so it scales
                // with the DRAW COUNT, which is printed next to it), and submit is the
                // queue and swapchain.
                let s = playback.take_split();
                let np = s.presents.max(1) as f64;
                // `encode_chain` splits ITSELF into prepare / upload / pass, and it can do
                // that here now: the renderer holds `performance.now()` (see `perf_now`),
                // where it used to hold a `std::time::Instant` that does not exist on wasm32
                // and so reported zero for every phase. A zero split read as "encode costs
                // nothing anywhere" on the engine where encode is most of the render, so if
                // the clock is ever NOT installed this says which case it is rather than
                // publishing the zeros.
                let inner = if !vitaslop_platform::gpu::wasm_clock_installed() {
                    " (no inner split: no wasm clock installed - see gpu::set_wasm_clock)"
                        .to_string()
                } else {
                    // >>> AND THE RESIDUAL, WHICH IS THE HALF THE SPLIT WAS HIDING.
                    //
                    // `prepare`/`upload`/`pass` are timed inside `encode_pass`; `encode` is the
                    // whole of `encode_chain`, which also retires buffers, grows or COMPACTS the
                    // resident geometry heap, precompiles pairs and assembles rendered cube
                    // maps - none of it timed. A reader naturally sums the three and assumes
                    // that is encode, and on a long device run it is NOT: measured there,
                    // `encode 19.3` against `prepare 12.8 + upload 0.8 + pass 1.5`, and on that
                    // run's worst frame **107.3 ms against 8.8**. Ninety-eight milliseconds with
                    // no name is worse than no split at all, because the split looks complete.
                    // This is the same lesson `prepare` already learned
                    // [[vitaslop-prepare-split-reports-its-own-residual]].
                    let named = s.prepare_ms + s.upload_ms + s.pass_ms;
                    format!(
                        " (prepare {:.1}, upload {:.1} [arena {:.1} = create {:.1} + write {:.1}, ubo-bg {:.1}], pass {:.1}, CHAIN {:.1} [precompile {:.1}, retire {:.1}, resident-heap {:.1}])",
                        s.prepare_ms / np,
                        s.upload_ms / np,
                        s.arena_ms / np,
                        s.arena_create_ms / np,
                        s.arena_write_ms / np,
                        s.ubo_bg_ms / np,
                        s.pass_ms / np,
                        (s.encode_ms - named).max(0.0) / np,
                        s.precompile_ms / np,
                        s.retire_ms / np,
                        s.resident_ms / np,
                    )
                };
                // Guest frames per PRESENT, stated rather than left to be inferred.
                //
                // The `fps` meter counts PRESENTS. The cpu figure is per GUEST FRAME. When the
                // loop runs more than one guest frame per present those are different rates, and
                // reading one against the other produces a phantom: 72.8 + 13.2 ms of measured
                // work against a 4 fps display looks like 150 ms a frame going somewhere
                // unaccounted, when really the loop ran four guest frames and presented once.
                // This ratio is the whole explanation and it costs one number.
                let per_present = cpu_frames as f64 / np;
                let perf_line = format!(
                    "cpu {cpu_avg:.1} ms/frame ({cpu_fps:.0} fps uncapped, {per_present:.1} \
                     guest frames per present) | render \
                     {render_avg:.1} ms = build {:.1} + encode {:.1}{inner} + submit {:.1} \
                     over {:.0} scenes / {:.0} draws ({:.0} gxp, {:.0} fixed)",
                    s.build_ms / np,
                    s.encode_ms / np,
                    s.submit_ms / np,
                    s.scenes as f64 / np,
                    s.draws as f64 / np,
                    s.gxp_draws as f64 / np,
                    s.fixed_draws as f64 / np,
                );
                report.emit("perf", &perf_line);
                last_perf = perf_line.clone();
                // Everything below also goes to a `diag` element on the PAGE, not only to the
                // console.
                //
                // # Why the console is not enough
                // The console is unreachable on the device this most needs to be read on. Every
                // counter in this file was console-only, so a phone - the actual target, and the
                // only machine whose numbers are not a proxy - could show a frame rate and
                // nothing that explains it. A run whose diagnostics require a USB cable and
                // remote debugging is a run nobody profiles.
                let mut diag = String::new();
                let frame_no = sched.core.frames();
                // Emit one diagnostic line to BOTH sinks: the console (which a harness reads and
                // which keeps the frame number next to every line) and the page's `diag` element
                // (which is the only one a phone can show).
                // Each line carries a TAG saying which measurement it is.
                //
                // Without one the panel reads as duplicated output, and was reported as such:
                // `BuildWork::line` and `EncodeWork::line` are reused for the window MEAN and for
                // the WORST single frame, so their identical `build work/frame:` prefix appears
                // twice in a row (three times counting the decode tally), and three near-identical
                // 400-character lines look like the same line printed three times rather than
                // three different frames.
                //
                // Tagging them was not enough, and the report came back. In a STEADY window every
                // frame has the same draw count and the same counters, so the worst frame's
                // payload is not merely similar to the mean's - it is BYTE-IDENTICAL, and the tag
                // is the only thing that differs across 400 characters. That is a duplicate by any
                // reading. So an identical payload is now collapsed to one short line that SAYS it
                // is identical, which is both shorter and more informative than the repeat: "this
                // window was uniform" is a fact about the run, and it is exactly what the repeated
                // line was failing to convey. A window that is NOT uniform still prints both, which
                // is when the second line earns its space.
                // >>> THE CONSOLE IS NOT THE REPORT. Every one of these sections still
                // goes into `diag`, which is what the panel shows and what the dev-server
                // sink records - so nothing is lost. What is gated is the CONSOLE copy,
                // because eight multi-line blocks per window turn a player's dev tools
                // into a firehose, and this page is the product, not the instrument
                // ([[vitaslop-web-is-the-product-not-the-tool]]). Set
                // `VITASLOP_PERF_CONSOLE=1` to get them back while debugging.
                let to_console = perf_console();
                let mut line = |diag: &mut String, tag: &str, text: &str| {
                    if to_console {
                        web_sys::console::log_1(&JsValue::from_str(&format!(
                            "[perf] frame {frame_no} | {tag} | {text}"
                        )));
                    }
                    diag.push_str(tag);
                    diag.push('\n');
                    diag.push_str(text);
                    diag.push_str("\n\n");
                };
                // The worst-frame counterpart of `line`: `payload` is the worst frame's own
                // counters and `mean` the window mean's, already formatted. When they agree there
                // is nothing to compare, so `prefix` (its millisecond cost, which is NOT the mean's)
                // is printed on its own.
                let mut worst_line =
                    |diag: &mut String, tag: &str, prefix: &str, payload: &str, mean: &str| {
                        if payload == mean {
                            let text = format!(
                                "{prefix} - counters IDENTICAL to the window mean above, so this \
                                 window was UNIFORM (not repeated here)"
                            );
                            line(diag, tag, &text);
                        } else {
                            line(diag, tag, &format!("{prefix} | {payload}"));
                        }
                    };
                // FIRST, and only when there is one: anything the run reported at WARN or ERROR.
                // A `WebGPU uncaptured error`, a dropped draw or a renderer fallback each turn a
                // silent wrong picture into a named one, and on a phone the console they were
                // written to does not exist. Nothing is printed when the run is clean, so a
                // healthy panel is not made longer by the instrument.
                if let Some(log) = crate::logging::page_log_report() {
                    line(&mut diag, "WARNINGS AND ERRORS", &log);
                }
                // WHAT WE PRESENTED, when the probe is on. Directly under the warnings because
                // it answers the question a silent panel raises: a healthy set of counters over
                // a blank screen is either a blank picture or a picture nobody showed, and
                // nothing else in this panel can tell those apart.
                if let Some(probe) = playback.take_probe_report() {
                    line(&mut diag, "PRESENTED SURFACE", &probe);
                }
                line(&mut diag, "RENDER SPLIT", &perf_line);
                // >>> AND THE PACING, WHICH IS WHAT DECIDES HOW MANY OF THOSE FRAMES ANYONE SEES.
                //
                // `frames/tick` above 1.0 is the loop computing pictures and discarding them:
                // every tick presents at most the newest scene, so a tick that ran two frames
                // threw one away. `saturated` is how often the wall-time accumulator was pinned
                // at its catch-up cap, which is the state in which that happens forever - a
                // machine that is behind never catches up, it just stops showing half its work.
                // `slept` against `asked` is the tick floor: when the loop asks for 2 ms and
                // gets 6, the difference is the host's, not the emulator's, and no amount of
                // guest optimisation moves it.
                {
                    let nt = ticks.max(1) as f64;
                    let pacing = format!(
                        "{ticks} ticks | period {:.1} ms, of which slept {:.1} ms \
                         (asked for {:.1}) | \
                         {:.2} guest frames/tick (worst {tick_frames_max}, {idle_ticks} idle), \
                         {:.2} presents/tick | charged {:.1} ms of game time per frame | \
                         acc {:.1} ms at tick \
                         ({saturated_ticks} saturated at {MAX_CATCHUP_MS:.0} ms) | \
                         unaccounted {:.1} ms/tick | host timer runs {:.1} ms late (learned)",
                        tick_span_ms / nt,
                        sleep_ms / nt,
                        sleep_asked_ms / nt,
                        tick_frames as f64 / nt,
                        presents as f64 / nt,
                        charged_ms / tick_frames.max(1) as f64,
                        acc_at_tick_ms / nt,
                        (tick_span_ms - sleep_ms - cpu_ms - render_ms) / nt,
                        timer_overshoot_ms(),
                    );
                    line(&mut diag, "PACING", &pacing);
                }
                // >>> WHAT THE NGS MIX CARRIED, AND WHETHER IT CLIPPED.
                //
                // These counters existed and were printed by the DESKTOP binary's shutdown
                // only, so the host where audio already works could read them and the phone -
                // which has no console, whose only report is this panel, and which is the one
                // a user reports crackling on - could not. The clipping half was additionally
                // behind a `debug` tracing gate, so "does the mix clip" had no answer anyone
                // could get at without flooding the panel it would have been printed into.
                {
                    let mix = vitaslop_runtime::vita::at9::mix_report();
                    if !mix.is_empty() {
                        line(&mut diag, "NGS MIX", &mix.join("\n"));
                    }
                }
                // >>> THE TWO NUMBERS THAT SAY WHETHER THE RING COUNTERS ABOVE ARE AN AUDIO
                // >>> PROBLEM AT ALL, AND THEY MEAN DIFFERENT THINGS.
                //
                // A capture reading `UNDERRUN 25%` beside `OVERRUN 50%` looks like a broken
                // backend and is not one: it is a RATE fault, and the ring's own counters
                // cannot name it because they all describe the ring.
                //
                // `sceAudioOutOutput` parks one grain of VIRTUAL time, so SOUND / CLOCK is
                // 1.00 on a healthy path at any frame rate - and it stays 1.00 when the CLOCK
                // is the thing that is wrong. What catches that is the CLOCK PER DISPLAYED
                // FRAME in display periods: it is the title's own vblank divisor, so a whole
                // number (1 = 60 fps, 2 = 30). One retail title read 0.985 sound/clock while
                // charging 2.99 periods a frame for a limiter that waits for two, and that
                // extra period is what made it produce 1.7 s of audio per second of real time
                // on a phone - filling the ring, dropping a third of it, and starving on the
                // next hitch.
                {
                    let (produced, flips, clock_us) = {
                        let host = sched.host.lock().unwrap();
                        (
                            host.state.audio_produced_seconds(),
                            host.state.flip_count(),
                            host.state.now_us(),
                        )
                    };
                    let clock_s = clock_us as f64 / 1.0e6;
                    if flips > 0 && clock_s > 0.0 {
                        line(
                            &mut diag,
                            "CLOCK vs PICTURE vs SOUND",
                            &format!(
                                "the clock advanced {:.2} display periods per displayed frame - the title's own vblank divisor, so a WHOLE number (1 = 60 fps, 2 = 30); a fraction, or one period more than its limiter waits for, is game time no frame accounted for and is what fills the audio ring. Sound {produced:.1}s against {clock_s:.1}s of clock ({:.2}x) - audio is billed in clock time, so THAT one is 1.00 at any frame rate and does not move when the clock is wrong.",
                                clock_s * 60.0 / flips as f64,
                                if clock_s > 0.0 { produced / clock_s } else { 0.0 },
                            ),
                        );
                    }
                }
                // >>> HOW FAST THE MOVIE DECODER IS ACTUALLY DELIVERING PICTURES.
                //
                // A device reported the title-screen movie as "frame, black, frame" and
                // nothing on this panel could say what rate it was arriving at - see
                // `avcdec::movie_report`. Silent unless a movie was decoded.
                {
                    let paced_frames = sched.core.frames().saturating_sub(frames_at_pace_start);
                    let mut movie = vitaslop_runtime::vita::avcdec::movie_report(paced_frames);
                    if !movie.is_empty() {
                        // A callback-driven decoder can only answer on a TASK, so its ceiling
                        // is how often this worker reaches the event loop at all. The tick
                        // gives it one turn a displayed frame; this is what the idle path adds
                        // on top, and at zero the decoder's whole budget is that one turn.
                        // >>> THE RATE, OVER THIS WINDOW, BECAUSE THE RUN AVERAGE IS A LIE
                        // >>> WHENEVER THE MOVIE STARTS PART WAY THROUGH.
                        //
                        // The cumulative figure above divides by every frame since the pacing
                        // meters reset. This title's front end starts its movie around frame
                        // 1300 of a run counting from 900, which reads as 0.34 pictures a frame
                        // where the title is asking for exactly ONE - a factor of three, in the
                        // direction that makes a healthy decoder look starved. That is how long
                        // a movie diagnosis can be spent on a denominator.
                        let (sub, del, calls, empty) =
                            vitaslop_runtime::vita::avcdec::movie_counters();
                        let win_frames = (tick_frames as f64).max(1.0);
                        movie.push(format!(
                            "movie this window: {:.2} access units and {:.2} pictures per \
                             displayed frame over {tick_frames} frames, {} of {} calls empty. \
                             This is the RATE; the line above is the run average",
                            (sub - last_movie.0) as f64 / win_frames,
                            (del - last_movie.1) as f64 / win_frames,
                            empty - last_movie.3,
                            calls - last_movie.2,
                        ));
                        last_movie = (sub, del, calls, empty);
                        let turns = browser_sched::event_loop_turns();
                        let per_frame =
                            if paced_frames > 0 { turns as f64 / paced_frames as f64 } else { 0.0 };
                        movie.push(format!(
                            "event loop: {turns} extra turns from the idle path \
                             ({per_frame:.2} per displayed frame, on top of the one the tick \
                             always gives). A decoder answers on a TASK, so this is the \
                             ceiling on how often it CAN answer"
                        ));
                        line(&mut diag, "MOVIE", &movie.join("\n"));
                    }
                }
                // >>> THE GUEST'S OWN HEAPS, AND WHETHER THE TITLE IS DRAINING THEM.
                //
                // A device reported thousands of `sceClibMspaceMemalign` failures, and nothing
                // could tell an ordinary tight pool from a release path this engine does not
                // implement. `allocs` against `frees` is exactly that distinction, and it
                // belongs on the panel because the device is where the report came from and
                // the console there does not exist.
                {
                    let spaces = sched.host.lock().unwrap().state.mspace_report();
                    if !spaces.is_empty() {
                        line(&mut diag, "GUEST HEAPS", &spaces.join("\n"));
                    }
                }
                // >>> WHETHER THE SAVE IS ACTUALLY REACHING STORAGE.
                //
                // "did my progress get kept?" cannot be answered by looking at the screen,
                // and it is answered wrongly by silence - which is what this said before it
                // existed. A count, the container's size and what each rebuild cost the
                // frame, so a save that is written is visible and one that FAILED says so
                // in the panel the user can copy.
                if persist.is_some() {
                    let text = match &save_error {
                        Some(e) => format!(
                            "WRITING THIS TITLE'S SAVE TO THIS BROWSER FAILED: {e}. Progress \
                             from here will be lost when the tab closes. {saves} write(s) \
                             succeeded before that."
                        ),
                        None if saves == 0 => "the game has not saved anything this run \
                             (nothing to write - this is normal until it does)"
                            .to_string(),
                        None => format!(
                            "{saves} write(s), container now {:.1} KB, {:.1} ms of frame time \
                             spent building them in total",
                            save_bytes as f64 / 1024.0,
                            save_ms
                        ),
                    };
                    line(&mut diag, "GAME DATA (the guest's own save)", &text);
                }
                // >>> WHERE THE IDLE CLOCK WENT, BY (wait kind, thread) - the instrument
                // that names a PARKED thread. It found the display-wait double-charge on
                // the desktop, and the browser is the one engine where a thread parks
                // forever with everything else healthy (a movie loop stalled at its fifth
                // frame while the game runs on) - which no other counter on this panel can
                // name. Largest owner first, the same shape the desktop prints at shutdown.
                {
                    let idle = sched.host.lock().unwrap().state.idle_attribution();
                    if !idle.is_empty() {
                        let total: u64 = idle.iter().map(|(_, us, _)| us).sum();
                        let mut s = format!("idle clock {:.1}s bought by:\n", total as f64 / 1e6);
                        for (owner, us, jumps) in idle.iter().take(8) {
                            s.push_str(&format!(
                                "  {:>8.3}s over {jumps:>8} jump(s) - {} on thread {:#x}\n",
                                *us as f64 / 1e6,
                                owner.kind.name(),
                                owner.thid,
                            ));
                        }
                        line(&mut diag, "IDLE ATTRIBUTION", &s);
                    }
                }
                // >>> AND WHICH THREADS ARE PARKED RIGHT NOW, AND IN WHAT.
                //
                // IDLE ATTRIBUTION above is a TIMED-wait instrument: it accounts for the clock
                // a park BOUGHT, so a thread parked on something with no deadline - a signal
                // wait, a join, a mutex nobody releases - buys nothing and never appears there.
                // That is precisely the shape of a hang, and the browser-only stall this panel
                // exists for (a movie loop that stops after its fifth access unit while the
                // game runs on) is exactly that shape. So the panel needs the other question:
                // for every LIVE thread, is it parked, and on which object.
                //
                // Per THREAD rather than per object, and parked threads only: the page keeps a
                // bounded number of DISTINCT lines (`logging::PAGE_LOG_CAP`) and the
                // whole-machine sync dump the desktop watchdog prints
                // (`VitaState::debug_sync_dump`) would fill it by itself.
                {
                    let blocked = sched.host.lock().unwrap().state.blocked_threads();
                    if !blocked.is_empty() {
                        let mut s = format!("{} parked thread(s):
", blocked.len());
                        // A retail title runs on the order of 20 threads, so this prints them
                        // ALL rather than a head: the one that matters in a stall is exactly
                        // the one an arbitrary cut would drop, and it cost a run to find that
                        // out. The section is a single panel entry with embedded newlines, so
                        // its length does not eat the page's distinct-line budget.
                        // >>> ONE CONSTANT, USED BY BOTH THE CUT AND THE NOTE ABOUT THE CUT.
                        // These were 48 and 16, so a run with 17 parked threads printed all
                        // seventeen and then said "... and 1 more" - a thread that did not
                        // exist, in the one section a person reads when a title has STOPPED.
                        // The user reported a stuck game and this line told them the evidence
                        // was incomplete when it was not.
                        const SHOWN: usize = 48;
                        for (thid, name, state) in blocked.iter().take(SHOWN) {
                            s.push_str(&format!("  thid {thid:#x} {name:?}: {state}
"));
                        }
                        if blocked.len() > SHOWN {
                            s.push_str(&format!("  ... and {} more
", blocked.len() - SHOWN));
                        }
                        line(&mut diag, "BLOCKED THREADS", &s);
                    }
                }
                ticks = 0;
                tick_frames = 0;
                charged_ms = 0.0;
                tick_frames_max = 0;
                idle_ticks = 0;
                sleep_ms = 0.0;
                sleep_asked_ms = 0.0;
                tick_span_ms = 0.0;
                acc_at_tick_ms = 0.0;
                saturated_ticks = 0;
                // >>> AND THE RUN'S OWN WORST FRAMES, WHICH NO WINDOW CAN SHOW. See `slowest`.
                //
                // Placed high in the panel, directly under the rate: when a user reports a
                // stall, this is the first line that can answer it, and everything below
                // describes a window that has already moved past the stall.
                {
                    let worst: Vec<String> = slowest
                        .iter()
                        .rev()
                        .map(|(f, ms, calls)| format!("f{f} {ms:.0} ms ({calls} calls)"))
                        .collect();
                    line(
                        &mut diag,
                        "SLOWEST FRAMES, cumulative for the run",
                        &format!(
                            "{frames_total} guest frames, {presents_total} presented ({:.2} \
                             frames per present) | worst: {} | WORST write_buffer OF THE RUN \
                             {:.1} ms for {:.0} KB",
                            frames_total as f64 / presents_total.max(1) as f64,
                            worst.join(", "),
                            // >>> ON THIS LINE BECAUSE THIS LINE SURVIVES THE HANG. The windowed
                            // copy on the ENCODE line has rolled over by the time a frozen page
                            // can be read - see `gpu::BUFFER_WRITE_WORST_RUN`.
                            vitaslop_platform::gpu::buffer_write_worst_run_us_kb().0,
                            vitaslop_platform::gpu::buffer_write_worst_run_us_kb().1,
                        ),
                    );
                }
                // The surface's format, alpha mode and present mode. Every one is chosen from
                // what the platform offers, so every one can differ between desktop and phone.
                line(&mut diag, "SURFACE", playback.surface_line());
                // ...and WHAT `build` did to cost that, in counts rather than milliseconds.
                // `build` is the largest part of the render half here and there is no
                // `Instant` inside it on wasm32, so the only portable instrument is the
                // work itself. See `vitaslop_runtime::render::BuildWork`.
                let (bg_hit, bg_new) = vitaslop_platform::gpu::take_sampler_bg_counts();
                let build_mean = s.work.line(s.presents.max(1));
                line(
                    &mut diag,
                    "BUILD, window mean",
                    &format!(
                        "{build_mean} | sampler bind groups {:.1} reused ({:.1} from the previous                          draw, {:.1} from earlier in the pass) / {:.1} BUILT",
                        bg_hit as f64 / np,
                        vitaslop_platform::gpu::take_sampler_bg_prev() as f64 / np,
                        vitaslop_platform::gpu::take_sampler_bg_pass() as f64 / np,
                        bg_new as f64 / np,
                    ),
                );
                // ...and WHAT `encode` did, in the same units the desktop prints, for the same
                // reason: `encode` is the larger half of the render here and its three phases
                // are timed but not attributed. Bytes and call counts say whether it is upload
                // volume or per-call boundary overhead, which a millisecond never can.
                let encode_mean = s.enc_work.line(s.presents.max(1));
                line(&mut diag, "ENCODE, window mean", &encode_mean);
                // ...and how full the caches that grow with RUN LENGTH are. Every other line
                // here describes the frame; this one is the only one that can explain a cost
                // which appears only after an hour of play. See `GxmRenderer::cache_sizes`.
                line(&mut diag, "RENDERER CACHES", &playback.cache_sizes());
                // ...and the GUEST-MEMORY side of the same question. `RENDERER CACHES` covers
                // what the renderer holds; this covers what the capture holds, which is where
                // the largest budget in the project lives (192 MB of texture snapshots) and
                // where the clear-whole that the device kept reporting used to be.
                line(
                    &mut diag,
                    "SNAPSHOT CACHES",
                    &vitaslop_runtime::host::snapshot_cache_report(),
                );
                // >>> AND WHAT IS RESIDENT, WHICH THE PANEL HAS NEVER CARRIED.
                //
                // The wasm heap and the texture working set were already computed - for a
                // CONSOLE line. The diagnostics panel is what a user actually pastes, so the
                // one symptom nobody could ever act on was "the whole phone feels sluggish":
                // a report full of milliseconds and not one byte of residency. The wasm heap
                // is the number that matters most here because it is SHARED with the guest and
                // can never hand a page back - it only ever goes up, so a run that grows it is
                // taking memory from the device permanently, and no frame timing shows that.
                line(
                    &mut diag,
                    "MEMORY",
                    &format!(
                        "emulator wasm heap {} MB (shared with the guest, never returned to the                          OS - this one only goes UP) | GPU texture working set {} MB of a {} MB                          texture cache budget | device reports {} | cache budgets scaled x{:.2}{}",
                        wasm_heap_mb(),
                        vitaslop_platform::gpu::texture_working_set_bytes() / (1024 * 1024),
                        vitaslop_platform::gpu::tex_cache_budget_now() / (1024 * 1024),
                        match vitaslop_platform::knobs::device_memory_gb() {
                            Some(gb) => format!("{gb} GB"),
                            None => "nothing (no navigator.deviceMemory)".to_string(),
                        },
                        vitaslop_platform::knobs::memory_scale(),
                        // >>> WHAT THE ADAPTER SAYS, BESIDE WHAT `deviceMemory` SAYS, because on
                        // a phone the two disagree and only one of them is capped. Reported, not
                        // acted on - see `ADAPTER_LOOKS_MOBILE` for why wiring this to the
                        // budgets would spend picture quality.
                        if ADAPTER_LOOKS_MOBILE.load(std::sync::atomic::Ordering::Relaxed) {
                            " | NOTE the GPU adapter is a MOBILE part, which deviceMemory (capped                          at 8 GB by its own spec) cannot see. Budgets are deliberately NOT                          scaled from this: the texture budget gates the BC->ETC2 re-encode,                          so tightening it here would trade picture quality silently"
                        } else {
                            ""
                        },
                    ),
                );
                // ...and INSIDE `prepare`, when it was asked for. See `WindowSplit::prep`.
                if !s.prep.is_empty() {
                    line(&mut diag, "PREPARE SPLIT, window mean", &s.prep.line(s.presents.max(1)));
                }
                // >>> WHAT THE GUEST-CPU HALF MOVED, IN BYTES. The only phase instrument this
                // engine can have.
                //
                // `cpu N ms/frame` is one number covering translated guest code, the scheduler
                // and every capture phase, and wasm has no `Instant` to split it with - so a
                // device capture had NOTHING between "the guest cost 130 ms" and the render
                // counters below. That is exactly how a defect spending 44% of every frame
                // comparing 105.8 MB of texture went unseen here while being plainly visible in
                // the desktop profiler.
                //
                // A byte count needs no clock. Volume is a measurement in its own right: a phase
                // moving a hundred megabytes a frame is a volume problem whatever the clock says,
                // and this line alone would have named it.
                let bytes = vitaslop_runtime::perf::take_bytes();
                let moved: Vec<String> = bytes
                    .iter()
                    .filter(|(_, b)| *b > 0)
                    .map(|(p, b)| format!("{} {:.2} MB", p.label(), *b as f64 / np / (1024.0 * 1024.0)))
                    .collect();
                if !moved.is_empty() {
                    line(
                        &mut diag,
                        "GUEST CPU, bytes moved per frame",
                        &format!(
                            "{} | a phase moving tens of MB a frame is the cost, whatever it \
                             times at",
                            moved.join(", ")
                        ),
                    );
                }
                // >>> AND NOW THE TIME, WHICH THIS ENGINE COULD NEVER REPORT BEFORE.
                //
                // The comment above is still true about why bytes are counted unconditionally,
                // but the clock half is no longer missing: `perf::set_clock` hands the runtime
                // `performance.now()`, so the same phases the desktop profiler splits are split
                // here too, on the engine that actually pays for them. MEASURED before this
                // existed: the browser's host-call bucket is 46% of a frame at 6.3 us a call
                // against the desktop's 1.58 us for the SAME handler work - so a phase's share
                // here is nothing like its share there, and reading the desktop's ranking as if
                // it were the browser's is how the previous two attempts at this path were
                // aimed. Gated on `VITASLOP_PERF` like every other clock read.
                if vitaslop_runtime::perf::enabled() {
                    let timed: Vec<String> = vitaslop_runtime::perf::Phase::all()
                        .iter()
                        .map(|p| (p, vitaslop_runtime::perf::read(*p)))
                        // A COUNT-ONLY phase (`perf::note_hit`) has no time and is still the
                        // reading - filtering on `ns` alone dropped exactly the rows whose
                        // point is a rate.
                        .filter(|(_, (ns, hits, _))| *ns > 0 || *hits > 0)
                        .map(|(p, (ns, hits, _))| {
                            format!(
                                "{} {:.2} ms/frame over {:.0} entries",
                                p.label(),
                                ns as f64 / 1.0e6 / np,
                                hits as f64 / np,
                            )
                        })
                        .collect();
                    if !timed.is_empty() {
                        line(
                            &mut diag,
                            "GUEST CPU, phase TIME per frame",
                            &format!(
                                "{} | these are INSIDE the host-call bucket, so compare them \
                                 against its ms, not against the frame",
                                timed.join(", ")
                            ),
                        );
                    }
                    // >>> HOW OFTEN THE HOST REACHED INTO GUEST MEMORY, which is the number
                    // that names a whole CLASS of defect on sight and needs no clock.
                    //
                    // A `dyn GuestMemory` access is a bounds check and a virtual call, and
                    // here it crosses into a SharedArrayBuffer view. A structure read one
                    // word at a time and the same structure read in one block move identical
                    // bytes and differ by tens of times in cost, so the byte counters above
                    // are blind to the difference and so is a phase timer that has not been
                    // split finely enough. WORD READS PER DRAW is the tell: it should be
                    // near zero, and every time it is not, something is looping over a
                    // structure a word at a time.
                    let (words, bulk) = vitaslop_runtime::perf::guest_accesses();
                    let draws = np.max(1.0);
                    let total_wraps = vitaslop_runtime::perf::epoch_wraps();
                    let epoch_wraps_window = total_wraps - epoch_wraps_at_window_start;
                    epoch_wraps_at_window_start = total_wraps;
                    let total_rebases = vitaslop_runtime::perf::epoch_rebases();
                    let epoch_rebases_window = total_rebases - epoch_rebases_at_window_start;
                    epoch_rebases_at_window_start = total_rebases;
                    line(
                        &mut diag,
                        "GUEST CPU, guest-memory accesses per frame",
                        &format!(
                            "{:.0} single-WORD reads, {:.0} bulk reads - a word count that \
                             scales with DRAWS is a structure being read a word at a time{}",
                            words as f64 / draws,
                            bulk as f64 / draws,
                            // >>> ...UNLESS THE PROFILER IS ON, IN WHICH CASE MOST OF THEM ARE
                            // ITS OWN. `vita::dispatch`'s call-site attribution walks up to
                            // FORTY words of the guest stack per host call to find the caller,
                            // which on a race is ~31 word reads a draw - a clean, stable,
                            // per-draw number that looks exactly like the defect this line
                            // exists to name, and is the instrument
                            // ([[vitaslop-instrument-failure-imitating-its-subject]]). Read
                            // this line from a run WITHOUT debug capture.
                            if debug_capture {
                                " | DEBUG CAPTURE IS ON and the call-site profiler scans up to \
                                 40 stack words per host call - most of this count is the \
                                 INSTRUMENT, not the engine"
                            } else {
                                ""
                            },
                        ),
                    );
                    // >>> AND HOW OFTEN THE GUEST-STORE EPOCH WRAPPED IN THIS WINDOW, because
                    // a wrap is a CLIFF and the byte counter above only shows its average.
                    //
                    // Every wrap drops every texture stamp, so the next use of each retained
                    // snapshot copies the whole texture across the guest boundary to compare
                    // it. One wrap costs the working set; spread over the window it reads as a
                    // small per-draw cost with no cause. See `perf::EPOCH_WRAPS`.
                    line(
                        &mut diag,
                        "GUEST CPU, guest-store epoch wraps",
                        &format!(
                            "{} in this window ({:.2} per presented frame) - each one drops                              every snapshot stamp, so the whole texture working set is                              re-compared over the frames that follow",
                            epoch_wraps_window,
                            epoch_wraps_window as f64 / draws,
                        ),
                    );
                    line(
                        &mut diag,
                        "GUEST CPU, guest-store epoch renumberings",
                        &format!(
                            "{} in this window - each one is a wrap AVOIDED by handing back                              the unused low half of the range",
                            epoch_rebases_window,
                        ),
                    );
                    // >>> AND WHICH PHASE THEY WERE IN. The total names the class; this names
                    // the line. Nested scopes double count (an inner phase's reads are in its
                    // parent's row too), so read a child out of its parent rather than summing
                    // the row - and the difference between this table's largest row and the
                    // total above is the reads that happen in no named phase at all, which is
                    // where the last search has to go.
                    let mut rows: Vec<(&'static str, u64)> = vitaslop_runtime::perf::Phase::all()
                        .iter()
                        .map(|p| (p.label(), vitaslop_runtime::perf::word_reads(*p)))
                        .filter(|(_, n)| *n > 0)
                        .collect();
                    rows.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
                    if !rows.is_empty() {
                        line(
                            &mut diag,
                            "GUEST CPU, single-WORD reads by phase, per frame",
                            &format!(
                                "{} | NESTED SCOPES DOUBLE COUNT; what these do not account                                  for is in no named phase",
                                rows.iter()
                                    .map(|(l, n)| format!("{l} {:.0}", *n as f64 / draws))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                        );
                    }
                    // >>> AND THE HOST CALLS RANKED BY TIME, which closes the frame.
                    //
                    // The phases above are the parts of the handler someone thought to name;
                    // MEASURED, they account for about half of the host-call bucket, and the
                    // rest was attributed to nothing at all. The browser has always been able
                    // to rank selectors by TIME (`host_calls_by_ms`) and has only ever
                    // reported them by COUNT, which answers a different question - a NID called
                    // a million times cheaply and one called twice for a millisecond look the
                    // same in a count and need opposite fixes.
                    let by_ms = browser_sched::host_calls_by_ms(8);
                    if !by_ms.is_empty() {
                        let named: Vec<String> = {
                            let h = sched.host.lock().unwrap();
                            by_ms
                                .iter()
                                .map(|(sel, calls, ms)| {
                                    let name = match h.import_at(*sel) {
                                        Some((_, func_nid)) => {
                                            let n = vitaslop_runtime::nid::name(func_nid);
                                            if n.is_empty() || n == "?" {
                                                format!("{func_nid:#010x}")
                                            } else {
                                                n.to_string()
                                            }
                                        }
                                        None => format!("selector {sel}"),
                                    };
                                    format!(
                                        "{name} {:.2} us each over {calls} calls ({:.0} ms)",
                                        if *calls == 0 { 0.0 } else { ms * 1000.0 / *calls as f64 },
                                        ms,
                                    )
                                })
                                .collect()
                        };
                        // CUMULATIVE since the run started, and said so: `hostcalls` keeps
                        // running totals and there is no per-window baseline here. The
                        // us/call and the RANK are what this line is for and neither
                        // depends on the divisor; dividing a run total by a window's frame
                        // count would have printed a per-frame figure that is simply wrong.
                        line(
                            &mut diag,
                            "GUEST CPU, top host calls by TIME (cumulative, whole run)",
                            &named.join(", "),
                        );
                    }
                    // Per WINDOW, like the byte counters above: a running total looks like a
                    // per-frame figure that keeps climbing, which reads as a leak.
                    vitaslop_runtime::perf::reset();
                }
                // The single worst present of the window, with ITS OWN counters. A mean over
                // frames that differ by 2.5x in draw count describes none of them - and when they
                // do NOT differ, saying so beats printing the same counters again.
                worst_line(
                    &mut diag,
                    "BUILD, the single WORST frame of the window",
                    &format!("{:.1} ms over {} draws", s.worst_build_ms, s.worst_draws),
                    &s.worst_work.line(1),
                    &build_mean,
                );
                worst_line(
                    &mut diag,
                    "ENCODE, the single WORST frame of the window",
                    // The phase split of THAT frame, next to its millisecond cost. When the
                    // counters match the window mean exactly - which is when an outlier is most
                    // puzzling - this is the only line that says which half of encode grew.
                    &format!(
                        "{:.1} ms over {} draws (prepare {:.1} + upload {:.1} [arena {:.1} = create {:.1} + write {:.1}, ubo-bg {:.1}] + pass {:.1} + CHAIN {:.1})",
                        s.worst_encode_ms,
                        s.worst_encode_draws,
                        s.worst_enc_phases.prepare_ms,
                        s.worst_enc_phases.upload_ms,
                        s.worst_enc_phases.arena_ms,
                        s.worst_enc_phases.arena_create_ms,
                        s.worst_enc_phases.arena_write_ms,
                        s.worst_enc_phases.ubo_bg_ms,
                        s.worst_enc_phases.pass_ms,
                        // The residual - see the window line for why it is printed. It is the
                        // whole of this frame on the device run that prompted it.
                        (s.worst_encode_ms
                            - (s.worst_enc_phases.prepare_ms
                                + s.worst_enc_phases.upload_ms
                                + s.worst_enc_phases.pass_ms))
                            .max(0.0),
                    ),
                    &s.worst_enc_work.line(1),
                    &encode_mean,
                );
                // What the decoder spent its bytes on, cumulative for the run. This decides
                // whether a compressed upload path is worth building - see `decode_by_format`.
                line(
                    &mut diag,
                    "TEXTURE DECODE by format, cumulative for the run",
                    &vitaslop_runtime::render::decode_by_format_line(),
                );
                // WHICH host calls, when the profiler is on. The count and the total cost are
                // already in the status line; neither says which NIDs they are, and the only way
                // to spend less at the boundary is to cross it fewer times. Cumulative from
                // boot, so two readings a known number of frames apart give calls per frame.
                //
                // LAST, because it is the longest by far - so the shorter lines are readable on a
                // phone without scrolling past a 25-entry histogram to reach them.
                // The EXPENSIVE instruments, only when the run was started with debug capture on.
                //
                // # Why this is a decision made before the run, and never automatic
                // Per-call timing and the call-site profiler each roughly double a frame's cost
                // on a phone: the first reads the clock twice per host call, the second scans the
                // guest stack once. An earlier version of this sampled them automatically - on
                // for one window in eight - which is cheaper but is still profiling machinery
                // running in a production run by default, deciding for the user that a permanent
                // eighth of their frame budget belongs to diagnostics. It does not. Debug capture
                // is asked for, or it does not happen.
                if debug_capture {
                    let (calls, total, handler, marshal) = browser_sched::host_call_split();
                    line(
                        &mut diag,
                        "HOST CALLS, handler vs marshalling, cumulative",
                        &format!(
                            "{calls} timed, {total:.0} ms ({:.2} us/call) = \
                             {handler:.0} ms handler + {marshal:.0} ms register marshalling \
                             ({:.0}% marshalling). NOTE debug capture inflates the frame cost; \
                             the RATIO is the reading, not the total.",
                            if calls > 0 { total * 1000.0 / calls as f64 } else { 0.0 },
                            if total > 0.0 { marshal * 100.0 / total } else { 0.0 },
                        ),
                    );
                    // >>> AND BY TIME, NOT ONLY BY COUNT. A count ranks how OFTEN the
                    // boundary is crossed; it cannot rank what the crossings COST, and two
                    // NIDs called equally often can differ by an order of magnitude in what
                    // they do. This is the reading that decides whether a host call is
                    // worth inlining into the guest, and until now it existed only on the
                    // desktop - which is not the machine whose numbers matter
                    // ([[vitaslop-desktop-cannot-price-a-count-win]]).
                    {
                        let host = sched.host.lock().unwrap();
                        let rows: Vec<String> = browser_sched::host_calls_by_ms(12)
                            .into_iter()
                            .map(|(sel, calls, ms)| {
                                let name = match host.import_at(sel) {
                                    Some((_, func_nid)) => {
                                        let n = vitaslop_runtime::nid::name(func_nid);
                                        if n.is_empty() || n == "?" {
                                            format!("{func_nid:#010x}")
                                        } else {
                                            n.to_string()
                                        }
                                    }
                                    None => format!("selector {sel}"),
                                };
                                format!(
                                    "{ms:>9.0} ms {:>7.2} us/call x{calls:<9} {name}",
                                    if calls > 0 { ms * 1000.0 / calls as f64 } else { 0.0 }
                                )
                            })
                            .collect();
                        if !rows.is_empty() {
                            line(
                                &mut diag,
                                "HOST CALLS by TIME, cumulative (debug capture inflates the \
                                 totals; the SHARES and the us/call ordering are the reading)",
                                &rows.join("\n"),
                            );
                        }
                    }
                    line(&mut diag, "HOST CALLS by NID, cumulative", &vitaslop_runtime::vita::call_sites_report(20));
                }
                report.emit("diag", &diag);
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
    crate::logging::install_panic_hook();
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
