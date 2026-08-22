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
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: surface_usage,
                format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: WIDTH,
                height: HEIGHT,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode,
                view_formats,
                desired_maximum_frame_latency: 2,
            },
        );
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
    fn present(&mut self, scenes: &[Scene]) {
        let clock = |p: &Option<web_sys::Performance>| p.as_ref().map(|p| p.now()).unwrap_or(0.0);
        let t0 = clock(&self.perf);
        // Tell the builder a new frame starts here. Its texture cache needs the boundary to
        // know what is in use right now and how big one frame's working set is; without it the
        // cache cannot tell a texture it is about to need again from one it is finished with.
        self.builder.begin_frame();
        let built: Vec<_> = scenes.iter().map(|s| self.builder.build(s)).collect();
        let draws: usize = built.iter().map(|b| b.draws.len()).sum();
        let t1 = clock(&self.perf);
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
    }

    /// The render split accumulated since the last read, and reset. Reported alongside the
    /// perf window so the two describe the same frames.
    fn take_split(&mut self) -> RenderSplit {
        core::mem::take(&mut self.split)
    }

    /// The surface description, for the diagnostics panel.
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
async fn next_tick() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let cb = Closure::once_into_js(move |_t: JsValue| {
            let _ = resolve.call0(&JsValue::UNDEFINED);
        });
        if let Some(window) = web_sys::window() {
            let _ = window.request_animation_frame(cb.as_ref().unchecked_ref());
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
    crate::logging::install_panic_hook();
    logging::init();
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
    // What the previous whole ITERATION cost - one guest frame plus the render that followed it.
    //
    // The pacing decision needs to know whether this machine is keeping up, and the guest half
    // alone answers a different question. On the desktop the guest frame is 14.7 ms, inside a
    // 16.7 ms budget, while the render adds 6.5 - so by the guest figure it is keeping up and by
    // the real one it is not, and it spent the difference dropping presents that bought it
    // nothing. Starts at 0 so the first tick is free to catch up: the boot frame is enormous and
    // would otherwise pin the loop to one frame per tick for the rest of the run.
    let mut last_iter_ms = 0.0f64;
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
        //
        // # Catch-up is only catch-up when the guest can keep up
        // Running N frames per tick and presenting once trades visible frame rate for guest
        // progress, and that trade is only available to a machine that is AHEAD. On one that is
        // behind, `acc` saturates at `MAX_CATCHUP_MS` every single tick, so the loop
        // permanently runs the maximum number of frames and permanently discards all but one
        // present - it never catches up, because there is nothing to catch up to.
        //
        // MEASURED on a phone (PowerVR D-series, one title's main screen): 72.8 ms of guest CPU
        // per frame against a 13.2 ms render. Four frames per tick made that 4 presents/s while
        // the guest advanced at 14 - so skipping three presents bought 11% of guest speed and
        // cost three quarters of the frame rate the user could see. One frame per tick shows all
        // 14. So the catch-up count is capped by whether the last frame FIT in its budget: a
        // fast machine still catches up after a hitch, a slow one stops paying for a catch-up it
        // cannot have.
        //
        // Fast-forward is exempt: it presents NOTHING by design, so there is no frame rate to
        // protect and running as many frames per tick as the budget allows is its entire job.
        let mut latest = None;
        let mut frames_this_tick = 0u32;
        // `1` is not a special case - it is what this evaluates to whenever the guest is slower
        // than real time, which is the whole point. Named `_per_tick` because the run's own
        // frame LIMIT is also called `max_frames` in this scope, and shadowing it here would
        // silently end the run at the first tick.
        let max_frames_per_tick = if fast || last_iter_ms <= FRAME_MS { u32::MAX } else { 1 };
        while acc >= FRAME_MS && frames_this_tick < max_frames_per_tick {
            acc -= FRAME_MS;
            frames_this_tick += 1;
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
            // The guest half now; the render this iteration pays is added after `present` below.
            last_iter_ms = c1 - c0;
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
                let (fuel_total, fuel_samples, fuel_max) = sched.core.fuel_report();
                let (raw_last, raw_min) = browser_sched::raw_fuel_stats();
                let (unbilled_none, unbilled_idle) = sched.core.unbilled_report();
                web_sys::console::log_1(&JsValue::from_str(&format!(
                    "[live] {status} | clock {:.2}s over {flips} flips ({quanta} quanta, \
                     {:.1} us/frame; {:.2}s quanta + {:.2}s idle) \
                     | preempt {preempts} ({on_fuel} on fuel) \
                     | fuel {fuel_total} over {fuel_samples} (max {fuel_max}, \
                     raw {raw_last}/min {raw_min}, unbilled {unbilled_none}+{unbilled_idle})                      | wasm heap {} MB \
                     | jspi {susp} susp, {starts} stacks, {abandoned} abandoned, \
                     {released} released | instances {inst_new} new, {inst_reused} reused \
                     | threads {live_threads} live, {finished_threads} finished",
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
            // A present belongs to the iteration that produced it, so the pacing decision for the
            // NEXT tick sees the true cost of this one.
            last_iter_ms += r1 - r0;
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
                    format!(
                        " (prepare {:.1}, upload {:.1}, pass {:.1})",
                        s.prepare_ms / np,
                        s.upload_ms / np,
                        s.pass_ms / np
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
                             frames per present) | worst: {}",
                            frames_total as f64 / presents_total.max(1) as f64,
                            worst.join(", "),
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
                        "{build_mean} | sampler bind groups {:.1} reused / {:.1} BUILT",
                        bg_hit as f64 / np,
                        bg_new as f64 / np,
                    ),
                );
                // ...and WHAT `encode` did, in the same units the desktop prints, for the same
                // reason: `encode` is the larger half of the render here and its three phases
                // are timed but not attributed. Bytes and call counts say whether it is upload
                // volume or per-call boundary overhead, which a millisecond never can.
                let encode_mean = s.enc_work.line(s.presents.max(1));
                line(&mut diag, "ENCODE, window mean", &encode_mean);
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
                        .filter(|(_, (ns, _, _))| *ns > 0)
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
                             scales with DRAWS is a structure being read a word at a time",
                            words as f64 / draws,
                            bulk as f64 / draws,
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
                        "{:.1} ms over {} draws (prepare {:.1} + upload {:.1} + pass {:.1})",
                        s.worst_encode_ms,
                        s.worst_encode_draws,
                        s.worst_enc_phases.prepare_ms,
                        s.worst_enc_phases.upload_ms,
                        s.worst_enc_phases.pass_ms,
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
