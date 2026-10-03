//! One running title, independent of who presents it: the guest, its live input,
//! the frame pacing, the pause states and the statistics. The `--game` window and
//! the shell both drive a `Session`; neither owns the loop's rules.

use std::time::{Duration, Instant};

use vitaslop_runtime::capture::Scene;
use vitaslop_runtime::TouchFrame;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::PhysicalKey;

use crate::audio_out::AudioOut;
use crate::input::Input;
use crate::retail::{DesktopInput, RetailGuest, SharedInput, FRAME_DT, GAME_H, GAME_W, PANEL_SCALE};

/// One display period, in ms.
const FRAME_MS: f64 = 1000.0 / 60.0;
/// The most wall time either direction may bank - the browser's `MAX_CATCHUP_MS`.
const MAX_CATCHUP_MS: f64 = 4.0 * FRAME_MS;

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
    /// Wall time owed to the guest, in ms - NEGATIVE when the last frame advanced more game
    /// time than the wall has since (a 30 fps title's two-period frame). See `tick`.
    acc_ms: f64,
    /// Floor charges banked to be refunded by the next long frame, and the wall-floor game time
    /// already accounted for - see `tick`.
    pace_debt_ms: f64,
    floor_seen_us: u64,
    pub last_tick: Instant,
    fps_since: Instant,
    fps_frames: u32,
    guest_frames_since: u64,
    fps: f64,
    guest_fps: f64,
    reported_exit: bool,
    /// The speakers, when this machine has any. `None` plays silent, as a headless run does.
    pub audio: Option<AudioOut>,
    /// When the guest's first frame was asked for, and the game clock then - for the run line
    /// [`Session::report_audio`] prints (the window title's speed is a 250 ms window).
    run_start: Option<(Instant, u64)>,
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
    pub fn new(mut guest: RetailGuest, input_shared: SharedInput, input: Input, pause_on_blur: bool) -> Session {
        // The window is where sound belongs; the guest is built with no sink at all (headless
        // runs keep it that way), so the speakers are attached here, before its first frame.
        let audio = match AudioOut::open() {
            Ok(out) => {
                println!("audio: {}", out.device);
                // `VITASLOP_AUDIO_MUTE=1`: start muted. The device still consumes, so every
                // counter (underrun above all) reads as it would aloud.
                if std::env::var("VITASLOP_AUDIO_MUTE").is_ok_and(|v| v.trim() == "1") {
                    out.set_muted(true);
                }
                guest.set_audio_sink(Box::new(out.sink()));
                Some(out)
            }
            Err(e) => {
                eprintln!("audio: {e} - the title will run silent");
                None
            }
        };
        Session {
            audio,
            run_start: None,
            guest,
            input_shared,
            input,
            paused: false,
            paused_by_blur: false,
            pause_on_blur,
            game_rect: None,
            cursor: (0.0, 0.0),
            mouse_down: false,
            acc_ms: 0.0,
            pace_debt_ms: 0.0,
            floor_seen_us: 0,
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
            WindowEvent::MouseInput { state, button, .. }
                if *button == MouseButton::Left => {
                    self.mouse_down = *state == ElementState::Pressed;
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
        self.acc_ms += now.duration_since(self.last_tick).as_secs_f64() * 1000.0;
        self.last_tick = now;

        let paused = self.paused || self.paused_by_blur;
        if let Some(a) = &self.audio {
            a.set_paused(paused);
        }
        if paused {
            self.acc_ms = 0.0;
        } else {
            if self.run_start.is_none() {
                self.run_start = Some((Instant::now(), self.guest.clock_us()));
            }
            if self.guest.current().is_empty() {
                self.guest.advance(); // bootstrap the first frame (runs the whole boot)
            }
            if self.acc_ms >= FRAME_MS {
                self.guest.advance();
                let charge = self.charge();
                self.acc_ms = (self.acc_ms - charge).max(-MAX_CATCHUP_MS);
            }
            // Neither direction banks more than four frames: a stall's surplus is dropped rather
            // than run back at speed.
            self.acc_ms = self.acc_ms.min(MAX_CATCHUP_MS);
        }

        if self.guest.finished() && !self.reported_exit {
            self.reported_exit = true;
            match self.guest.error() {
                Some(e) => eprintln!("guest exited with error: {e}"),
                None => println!("guest exited after {} frames", self.guest.frames()),
            }
        }
    }

    /// >>> A FRAME COSTS THE WALL TIME OF THE GAME TIME IT ADVANCED - two display periods on a
    /// 30 fps title, not one. The browser's pacer (`vitaslop-web/src/lib.rs`, the live loop's
    /// `advanced_ms`), rule for rule.
    ///
    /// This charged a flat 1/60 s per frame, which paced one guest FLIP per display period: a
    /// title whose frame waits for two vblanks ran at twice real time. MEASURED in this window
    /// (a fighting title's menus and cutscenes, 7,000 frames): 59.4 fps with the game clock
    /// at 127% of the wall. The browser measured the same thing first (`fps 53 (177% speed)`).
    ///
    /// - A frame advancing less than a period is charged a whole one, and the overcharge is
    ///   BANKED (to four frames) and refunded out of the frames that advance more - so over any
    ///   stretch the charge is the clock's own advance.
    /// - The wall FLOOR's pull is not charged: a slow guest's clock pulled up to the wall is
    ///   wall time already in `acc_ms`.
    fn charge(&mut self) -> f64 {
        let advanced = self.guest.last_advance_us() as f64 / 1000.0;
        let floor = self.guest.clock_from_wall_us();
        let gain = floor.saturating_sub(self.floor_seen_us) as f64 / 1000.0;
        self.floor_seen_us = floor;
        let mut c = (advanced - gain).max(0.0);
        let give = self.pace_debt_ms.min((c - FRAME_MS).max(0.0));
        c -= give;
        self.pace_debt_ms -= give;
        if c < FRAME_MS {
            self.pace_debt_ms = (self.pace_debt_ms + FRAME_MS - c).min(MAX_CATCHUP_MS);
            c = FRAME_MS;
        }
        c
    }

    pub fn scenes(&mut self) -> (&[Scene], (u32, u32), &[u32]) {
        let display = self.guest.display_size();
        (self.guest.current(), display, self.guest.current_presents())
    }

    /// Whether the guest is stepping (not paused by the person or the window, not ended).
    pub fn live(&self) -> bool {
        !self.paused && !self.paused_by_blur && !self.guest.finished()
    }

    /// How long until [`Self::tick`] will step the guest again - what a window SLEEPS for
    /// instead of re-presenting an unchanged frame every display period. Zero when it is due.
    pub fn due_in(&self) -> Duration {
        let owed = self.acc_ms + self.last_tick.elapsed().as_secs_f64() * 1000.0;
        Duration::from_secs_f64(((FRAME_MS - owed) / 1000.0).max(0.0))
    }

    /// The executable the guest's `sceAppMgrLoadExec` asked to be REPLACED by, once its process
    /// has halted for it - taken, so the owner boots it once. The browser and the headless run
    /// boot it in place; a window that did not left a launcher-first title (an action title)
    /// sitting on a halted launcher, unplayable.
    pub fn take_exec(&mut self) -> Option<String> {
        if !self.guest.finished() {
            return None;
        }
        self.guest.take_exec_request()
    }

    /// Put a frame's rendered small targets back in guest memory - see
    /// `vitaslop_native::apply_rtt_writebacks`, and `RetailGfx::rtt_writebacks` for where they
    /// come from. A title that reads a target it drew on the CPU otherwise reads its own
    /// allocator poison.
    pub fn apply_writebacks(&mut self, wb: &[(u32, u32, u32, Vec<u8>)]) {
        if wb.is_empty() {
            return;
        }
        let scenes = self.guest.current().to_vec();
        // On the parallel engine the guest keeps running between frames: it PAUSES for the
        // write, so no guest store lands between the whole-region check and the bytes - the
        // browser's `pause_guest`.
        self.guest.pause_guest();
        let guest = &self.guest;
        vitaslop_native::apply_rtt_writebacks(wb, &scenes, |a, n| guest.read_guest(a, n), |a, b| guest.write_guest(a, b));
        self.guest.resume_guest();
    }

    /// One line on what reached the speakers: the guest's own production beside the device's
    /// counters. Underrun is the emulator not keeping up (a performance number, as in the
    /// browser's panel); overrun and latency skips are audio produced ahead of the device.
    pub fn report_audio(&mut self) {
        if let Some((t0, c0)) = self.run_start {
            let wall = t0.elapsed().as_secs_f64();
            let game = self.guest.clock_us().saturating_sub(c0) as f64 / 1e6;
            let frames = self.guest.frames();
            println!(
                "run: {frames} frames over {wall:.1} s of wall = {:.1} fps; game clock {game:.1} s = {:.0}% of the wall",
                frames as f64 / wall.max(1e-9),
                100.0 * game / wall.max(1e-9)
            );
        }
        let produced = self.guest.audio_produced_seconds();
        match &self.audio {
            Some(a) => {
                let s = a.stats();
                println!(
                    "audio: produced {produced:.1} s, peak {:.3}, underrun {:.2} s, overrun {:.2} s, latency skip {:.2} s, rejoins {}",
                    s.peak, s.underrun_s, s.overrun_s, s.latency_skip_s, s.rejoins
                );
            }
            None => println!("audio: produced {produced:.1} s, no output device"),
        }
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
