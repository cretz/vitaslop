//! One running title, independent of who presents it: the guest, its live input,
//! the frame pacing, the pause states and the statistics. The `--game` window and
//! the shell both drive a `Session`; neither owns the loop's rules.

use std::time::{Duration, Instant};

use vitaslop_runtime::capture::Scene;
use vitaslop_runtime::TouchFrame;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::PhysicalKey;

use crate::input::Input;
use crate::retail::{DesktopInput, RetailGuest, SharedInput, FRAME_DT, GAME_H, GAME_W, PANEL_SCALE};

pub(crate) struct Session {
    pub guest: RetailGuest,
    input_shared: SharedInput,
    pub input: Input,
    /// The person's pause (Space, the menu).
    pub paused: bool,
    /// The window's: unfocused, when the settings ask for it.
    pub paused_by_blur: bool,
    pub pause_on_blur: bool,
    /// Mouse-as-touch is confined to this rectangle of the window (the game's
    /// letterboxed area), in physical pixels; `None` means the whole window.
    pub game_rect: Option<(f64, f64, f64, f64)>,
    cursor: (f64, f64),
    mouse_down: bool,
    acc: Duration,
    pub last_tick: Instant,
    fps_since: Instant,
    fps_frames: u32,
    guest_frames_since: u64,
    fps: f64,
    guest_fps: f64,
    reported_exit: bool,
}

pub(crate) struct Stats {
    pub fps: f64,
    pub guest_fps: f64,
    pub speed_pct: f64,
    pub frames: u64,
    pub finished: bool,
    pub paused: bool,
    pub paused_by_blur: bool,
}

impl Stats {
    pub fn title_line(&self) -> String {
        let state = if self.finished {
            " [exited]"
        } else if self.paused_by_blur {
            " [paused - window not focused]"
        } else if self.paused {
            " [paused]"
        } else {
            ""
        };
        let speed = if self.paused || self.paused_by_blur || self.finished {
            String::new()
        } else {
            format!("  |  speed {:.0}%", self.speed_pct)
        };
        format!("{:.0} fps present  |  {:.0} fps guest{speed}{state}  |  frame {}", self.fps, self.guest_fps, self.frames)
    }
}

impl Session {
    pub fn new(guest: RetailGuest, input_shared: SharedInput, input: Input, pause_on_blur: bool) -> Session {
        Session {
            guest,
            input_shared,
            input,
            paused: false,
            paused_by_blur: false,
            pause_on_blur,
            game_rect: None,
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
        }
    }

    /// Keyboard, mouse and focus. `window_size` is the inner size in physical pixels.
    pub fn event(&mut self, event: &WindowEvent, _window_size: Option<(f64, f64)>) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.input.set_key(code, event.state == ElementState::Pressed);
                }
            }
            WindowEvent::Focused(focused) => {
                if self.pause_on_blur {
                    self.paused_by_blur = !focused;
                }
                if !focused {
                    self.input.release_all();
                    self.mouse_down = false;
                }
            }
            WindowEvent::CursorMoved { position, .. } => self.cursor = (position.x, position.y),
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.mouse_down = *state == ElementState::Pressed;
                }
            }
            _ => {}
        }
    }

    fn mouse_touch(&self, window_size: Option<(f64, f64)>) -> Option<TouchFrame> {
        if !self.mouse_down {
            return None;
        }
        let (x0, y0, w, h) = match self.game_rect {
            Some(r) => r,
            None => {
                let (w, h) = window_size.unwrap_or((GAME_W as f64, GAME_H as f64));
                (0.0, 0.0, w, h)
            }
        };
        if w <= 0.0 || h <= 0.0 {
            return None;
        }
        let sx = ((self.cursor.0 - x0) / w * GAME_W as f64).clamp(0.0, GAME_W as f64);
        let sy = ((self.cursor.1 - y0) / h * GAME_H as f64).clamp(0.0, GAME_H as f64);
        Some(TouchFrame::single((sx as f32 * PANEL_SCALE) as u16, (sy as f32 * PANEL_SCALE) as u16))
    }

    /// Feed the input, advance the guest by however many 1/60 s ticks have elapsed.
    pub fn tick(&mut self, window_size: Option<(f64, f64)>) {
        self.input.pump_gamepad();
        let ctrl = self.input.ctrl_frame();
        let touch = self.mouse_touch(window_size);
        *self.input_shared.lock().unwrap() = DesktopInput { ctrl, touch };

        let now = Instant::now();
        self.acc += now.duration_since(self.last_tick);
        self.last_tick = now;

        if self.paused || self.paused_by_blur {
            self.acc = Duration::ZERO;
        } else {
            if self.guest.current().is_empty() {
                self.guest.advance(); // bootstrap the first frame (runs the whole boot)
            }
            if self.acc >= FRAME_DT {
                self.acc -= FRAME_DT;
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
    }

    pub fn scenes(&mut self) -> (&[Scene], (u32, u32), &[u32]) {
        let display = self.guest.display_size();
        (self.guest.current(), display, self.guest.current_presents())
    }

    /// Fresh statistics every 250 ms, `None` in between.
    pub fn stats(&mut self, now: Instant) -> Option<Stats> {
        self.fps_frames += 1;
        let since = now.duration_since(self.fps_since);
        if since < Duration::from_millis(250) {
            return None;
        }
        let secs = since.as_secs_f64();
        let guest_now = self.guest.frames();
        self.fps = self.fps_frames as f64 / secs;
        self.guest_fps = guest_now.saturating_sub(self.guest_frames_since) as f64 / secs;
        self.fps_frames = 0;
        self.guest_frames_since = guest_now;
        self.fps_since = now;
        let target = 1.0 / FRAME_DT.as_secs_f64();
        Some(Stats {
            fps: self.fps,
            guest_fps: self.guest_fps,
            speed_pct: self.guest_fps / target * 100.0,
            frames: guest_now,
            finished: self.guest.finished(),
            paused: self.paused,
            paused_by_blur: self.paused_by_blur,
        })
    }
}
