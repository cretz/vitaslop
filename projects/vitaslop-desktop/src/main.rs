//! Native desktop app: the same load -> transpile -> run -> capture -> wgpu path
//! as the browser, in a live winit window with real keyboard and gamepad input.
//!
//! This first milestone runs the cube CPU to completion up front (a scripted
//! input world), then presents the captured scenes to the window's wgpu surface,
//! reading the real pad each frame and folding it into a SceCtrl frame shown in
//! the title bar. Feeding that frame to the guest *live* (so the pad steers the
//! guest, not just the readout) needs the guest to yield per frame, which is the
//! cooperative-scheduler milestone; the input mapping and window loop here are
//! built to carry over unchanged when the live frame source lands.

mod cube;
mod frames;
mod gfx;
mod input;

use std::sync::Arc;
use std::time::{Duration, Instant};

use frames::{FrameSource, Playback};
use input::Input;
use vitaslop_runtime::CtrlFrame;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// The Vita display resolution the cube targets; the window opens at this size so
/// the captured scenes present at their native aspect.
const WIDTH: u32 = 960;
const HEIGHT: u32 = 544;

/// Playback pacing: advance one captured scene per 1/60 s of wall time, regardless
/// of the monitor's refresh, so the cube spins at the rate the guest intended.
const FRAME_DT: Duration = Duration::from_micros(16_666);

fn main() {
    // Run the guest CPU pass first so the window can present immediately.
    let run = match cube::run_cube() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to run cube: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "cube ran on native wasmtime: {} scenes captured, transpile {:.1} ms, {} frames in {:.1} ms ({:.0} us/frame)",
        run.scenes.len(),
        run.transpile_ms,
        run.frames,
        run.run_ms,
        run.run_ms * 1000.0 / run.frames as f64,
    );

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        source: Playback::new(run.scenes),
        input: Input::new(),
        window: None,
        gfx: None,
        paused: false,
        acc: Duration::ZERO,
        last_tick: Instant::now(),
        fps_since: Instant::now(),
        fps_frames: 0,
        fps: 0.0,
    };
    event_loop.run_app(&mut app).expect("run event loop");
}

/// The whole app: the frame source, live input, the window and its GPU surface,
/// and the small amount of playback-pacing and FPS bookkeeping.
struct App {
    source: Playback,
    input: Input,
    window: Option<Arc<Window>>,
    gfx: Option<gfx::Gfx>,
    paused: bool,
    /// Wall-time accumulator so scene advance is paced to 60 Hz, not the monitor.
    acc: Duration,
    last_tick: Instant,
    fps_since: Instant,
    fps_frames: u32,
    fps: f64,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // Already have a window (e.g. a spurious second resume).
        }
        let attrs = Window::default_attributes()
            .with_title("vitaslop - cube")
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        match gfx::Gfx::new(window.clone()) {
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

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
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
                    // Host controls take precedence over the pad mapping.
                    match code {
                        KeyCode::Escape if pressed => {
                            event_loop.exit();
                            return;
                        }
                        KeyCode::Space if pressed && !event.repeat => {
                            self.paused = !self.paused;
                        }
                        _ => {}
                    }
                    self.input.set_key(code, pressed);
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

impl App {
    /// Pace playback to 60 Hz off wall time, present the current scene to the
    /// surface (Fifo vsync throttles the GPU), and keep the title readout current.
    fn render(&mut self) {
        let Some(gfx) = self.gfx.as_mut() else { return };

        self.input.pump_gamepad();
        let ctrl = self.input.ctrl_frame();

        // Advance the animation by however much wall time has elapsed, so the spin
        // rate is refresh-independent. Skips advance while paused.
        let now = Instant::now();
        self.acc += now.duration_since(self.last_tick);
        self.last_tick = now;
        while self.acc >= FRAME_DT {
            self.acc -= FRAME_DT;
            if !self.paused {
                self.source.advance(ctrl);
            }
        }

        gfx.present(self.source.current());

        // FPS + live SceCtrl readout in the title, refreshed a few times a second.
        self.fps_frames += 1;
        let since = now.duration_since(self.fps_since);
        if since >= Duration::from_millis(250) {
            self.fps = self.fps_frames as f64 / since.as_secs_f64();
            self.fps_frames = 0;
            self.fps_since = now;
            if let Some(w) = self.window.as_ref() {
                let paused = if self.paused { " [paused]" } else { "" };
                w.set_title(&format!(
                    "vitaslop - cube  |  {:.0} fps{}  |  pad: {}",
                    self.fps,
                    paused,
                    describe_ctrl(ctrl),
                ));
            }
        }
    }
}

/// A short human label for the buttons and sticks held this frame, so input is
/// visibly reaching the SceCtrl mapping ahead of the live guest path.
fn describe_ctrl(f: CtrlFrame) -> String {
    use input::*;
    let mut parts = Vec::new();
    for (bit, name) in [
        (SCE_CTRL_UP, "Up"),
        (SCE_CTRL_DOWN, "Down"),
        (SCE_CTRL_LEFT, "Left"),
        (SCE_CTRL_RIGHT, "Right"),
        (SCE_CTRL_TRIANGLE, "Triangle"),
        (SCE_CTRL_CIRCLE, "Circle"),
        (SCE_CTRL_CROSS, "Cross"),
        (SCE_CTRL_SQUARE, "Square"),
        (SCE_CTRL_LTRIGGER, "L"),
        (SCE_CTRL_RTRIGGER, "R"),
        (SCE_CTRL_START, "Start"),
        (SCE_CTRL_SELECT, "Select"),
    ] {
        if f.buttons & bit != 0 {
            parts.push(name);
        }
    }
    let buttons = if parts.is_empty() { "none".to_string() } else { parts.join("+") };
    // Only mention sticks when off center, so the readout stays quiet at rest.
    if f.lx != 128 || f.ly != 128 || f.rx != 128 || f.ry != 128 {
        format!("{buttons}  L({},{}) R({},{})", f.lx, f.ly, f.rx, f.ry)
    } else {
        buttons
    }
}
