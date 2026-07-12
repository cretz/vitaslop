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

mod conformance;
mod web_vm;

pub use conformance::run_conformance;

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_platform::gpu::CubeRenderer;
use vitaslop_runtime::capture::Scene;
use vitaslop_runtime::{CtrlFrame, VitaEnv, World};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

use web_vm::WebVm;

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
}

/// Load, transpile, and run the cube to completion on the browser wasm engine,
/// returning the captured scenes and timings.
fn run_cube_cpu() -> Result<CpuRun, JsValue> {
    let perf = web_sys::window()
        .and_then(|w| w.performance())
        .ok_or_else(|| JsValue::from_str("no performance clock"))?;

    let m = loader::load(CUBE).map_err(|e| JsValue::from_str(&format!("load: {e:?}")))?;
    let inputs = m.program_inputs();

    let t0 = perf.now();
    let artifact = vitaslop_transpiler::transpile(&inputs.program())
        .map_err(|e| JsValue::from_str(&format!("transpile: {e:?}")))?;
    let transpile_ms = perf.now() - t0;

    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();
    let world = Box::new(ScriptedWorld { polls: 0, frames: FRAMES });
    let mut venv = VitaEnv::new(imports, inputs.base, inputs.mem_bytes, world);
    venv.state.halt_on_terminate = true;
    // Keep an Rc handle to read the capture back after the run; hand the VM a
    // clone as the dispatcher (Rc<RefCell<VitaEnv>> implements ImportDispatch).
    let env = Rc::new(RefCell::new(venv));

    // The cube routes the Vita NID trap (env.import) to the VitaEnv; no svc.
    let vm = WebVm::new(
        &artifact.wasm,
        &inputs.code,
        inputs.base,
        inputs.mem_bytes,
        None,
        Some(Box::new(env.clone())),
    )?;

    let t1 = perf.now();
    vm.call(m.entry & !1)?;
    let run_ms = perf.now() - t1;

    let scenes = env.borrow().state.capture.scenes.clone();
    Ok(CpuRun { scenes, transpile_ms, run_ms })
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
        Ok(Playback { surface, device, queue, cube, depth, scenes, frame: 0 })
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

/// Entry point the page calls with the target `<canvas>`. Runs the cube CPU pass,
/// reports timings, then starts WebGPU playback of the captured frames. Returns a
/// short status string (also useful for the page to display).
#[wasm_bindgen]
pub async fn run(canvas: JsValue) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    let canvas: HtmlCanvasElement = canvas.dyn_into()?;

    let cpu = run_cube_cpu()?;
    let status = format!(
        "cube ran on browser wasm: {} scenes captured, transpile {:.1} ms, {} frames in {:.1} ms ({:.0} us/frame)",
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
