//! Native desktop app: the same load -> transpile -> run -> capture -> wgpu path
//! as the browser, in a live winit window with real keyboard and gamepad input.
//!
//! The guest runs *live* on the cooperative scheduler (a wasmtime fiber): one
//! frame is stepped per window redraw, with the real controller injected between
//! frames through the SceCtrl seam. So the pad genuinely reaches the guest - press
//! START (Enter, or the pad's Start) and the cube tears itself down and exits,
//! just as it would on hardware. The window presents each captured frame to its
//! wgpu surface through the shared cube pipeline the browser and headless oracle
//! also use.

mod gfx;
mod input;
mod live;
mod retail;

use std::sync::Arc;
use std::time::{Duration, Instant};

use input::Input;
use live::LiveGuest;
use vitaslop_runtime::CtrlFrame;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// The Vita display resolution the cube targets; the window opens at this size so
/// the frames present at their native aspect.
const WIDTH: u32 = 960;
const HEIGHT: u32 = 544;

/// Step one guest frame per 1/60 s of wall time, regardless of the monitor's
/// refresh, so the cube advances at the rate the guest intends.
const FRAME_DT: Duration = Duration::from_micros(16_666);

fn main() {
    // Surface the runtime's `tracing` diagnostics (`vitaslop::io=trace`, `vitaslop::gxm=debug`,
    // ...).
    //
    // `VITASLOP_LOG` first, `RUST_LOG` as the fallback - see `knobs::log_filter`.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            vitaslop_platform::knobs::log_filter(),
        ))
        .with_writer(std::io::stderr)
        .try_init();
    // `--game <dir>` plays a real extracted retail title (decrypt -> link -> transpile
    // -> preemptive scheduler -> general GXM renderer) in a live window. With no
    // argument, run the built-in clean-room cube demo below.
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--game" || a == "-g") {
        let Some(dir) = args.get(i + 1).cloned() else {
            eprintln!("usage: vitaslop-desktop --game <extracted-app-dir> [--headless <shot-dir>]");
            std::process::exit(2);
        };
        // `--recipe <file>` replays a scripted TAS recipe in the live window so a
        // recorded playthrough can be watched (live keyboard/mouse still nudge it).
        let recipe = args
            .iter()
            .position(|a| a == "--recipe" || a == "-r")
            .and_then(|i| args.get(i + 1))
            .map(|p| std::fs::read_to_string(p).unwrap_or_else(|e| {
                eprintln!("cannot read recipe {p}: {e}");
                std::process::exit(2);
            }));
        // `--headless <dir>` validates the retail path without opening a window (drive
        // the tutorial + render one frame to a PNG); useful on a display-less box.
        let result = match args.iter().position(|a| a == "--headless") {
            Some(h) => match args.get(h + 1) {
                Some(shot) => retail::headless_check(dir.into(), shot.into()),
                None => {
                    eprintln!("--headless requires a shot directory");
                    std::process::exit(2);
                }
            },
            None => retail::run(dir.into(), recipe),
        };
        if let Err(e) = result {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }

    // Load, transpile, and instantiate the cube for cooperative execution.
    let guest = match LiveGuest::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("failed to start cube: {e}");
            std::process::exit(1);
        }
    };
    println!("cube ready (transpile + instantiate {:.1} ms)", guest.build_ms);
    println!("controls: arrows = d-pad, Z/X/A/S = cross/circle/square/triangle, Q/E = L/R,");
    println!("          Enter = Start (exits the cube), Shift = Select, Space = pause, Esc = quit");

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        guest,
        input: Input::new(),
        window: None,
        gfx: None,
        paused: false,
        acc: Duration::ZERO,
        last_tick: Instant::now(),
        fps_since: Instant::now(),
        fps_frames: 0,
        fps: 0.0,
        reported_exit: false,
    };
    event_loop.run_app(&mut app).expect("run event loop");
}

/// The whole app: the live guest, real input, the window and its GPU surface, and
/// the pacing and FPS bookkeeping.
struct App {
    guest: LiveGuest,
    input: Input,
    window: Option<Arc<Window>>,
    gfx: Option<gfx::Gfx>,
    paused: bool,
    /// Wall-time accumulator so guest frames step at 60 Hz, not the monitor rate.
    acc: Duration,
    last_tick: Instant,
    fps_since: Instant,
    fps_frames: u32,
    fps: f64,
    /// So the guest's exit is logged once, not every redraw.
    reported_exit: bool,
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
    /// Read input, step the guest by however much wall time has elapsed (60 Hz),
    /// present the newest frame to the surface (Fifo vsync throttles the GPU), and
    /// keep the title readout current.
    fn render(&mut self) {
        self.input.pump_gamepad();
        let ctrl = self.input.ctrl_frame();

        let now = Instant::now();
        self.acc += now.duration_since(self.last_tick);
        self.last_tick = now;

        if self.paused {
            // Don't build up a backlog of frames to catch up on while paused.
            self.acc = Duration::ZERO;
        } else {
            // Bootstrap the first frame even before a full tick has accumulated,
            // then step one guest frame per elapsed 1/60 s (catching up if behind).
            if self.guest.current().is_none() {
                self.guest.advance(ctrl);
            }
            while self.acc >= FRAME_DT {
                self.acc -= FRAME_DT;
                self.guest.advance(ctrl);
            }
        }

        if self.guest.finished() && !self.reported_exit {
            self.reported_exit = true;
            match self.guest.error() {
                Some(e) => eprintln!("cube exited with error: {e}"),
                None => println!("cube exited cleanly after {} frames", self.guest.frames()),
            }
        }

        if let Some(gfx) = self.gfx.as_mut() {
            if let Some(scene) = self.guest.current() {
                gfx.present(scene);
            }
        }

        self.update_title(ctrl, now);
    }

    /// FPS + live SceCtrl readout + guest frame count in the title, a few times a
    /// second.
    fn update_title(&mut self, ctrl: CtrlFrame, now: Instant) {
        self.fps_frames += 1;
        let since = now.duration_since(self.fps_since);
        if since < Duration::from_millis(250) {
            return;
        }
        self.fps = self.fps_frames as f64 / since.as_secs_f64();
        self.fps_frames = 0;
        self.fps_since = now;

        let Some(w) = self.window.as_ref() else { return };
        let state = if self.guest.finished() {
            " [exited]"
        } else if self.paused {
            " [paused]"
        } else {
            ""
        };
        w.set_title(&format!(
            "vitaslop - cube  |  {:.0} fps{}  |  frame {}  |  pad: {}",
            self.fps,
            state,
            self.guest.frames(),
            describe_ctrl(ctrl),
        ));
    }
}

/// A short human label for the buttons and sticks held this frame, so input is
/// visibly reaching the SceCtrl mapping that the guest reads.
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
