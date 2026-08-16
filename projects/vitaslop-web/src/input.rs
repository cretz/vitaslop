//! The browser input world: the [`World`] the live run polls, fed by real pointer
//! and keyboard events (a human playing) and/or a scripted TAS recipe (a deterministic
//! headless run, e.g. the e2e that drives to the tutorial for a screenshot). Both feed
//! the same seam the native probe uses ([`World::poll_ctrl`]/[`World::poll_touch`]), so
//! the browser reaches the exact live gameplay the native probe does.
//!
//! # One shared input cell
//! DOM event listeners run on the same single thread as the scheduler, but the `World`
//! trait is `Send` (native runs it across an OS thread), so the live state is shared
//! through an `Arc<Mutex<_>>` rather than an `Rc<RefCell<_>>`. On the browser's one
//! thread the lock never contends - it only satisfies the bound.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use crate::location::SharedLocation;
use vitaslop_runtime::{CtrlFrame, LocationFix, LocationPermission, RecipeWorld, TouchFrame, World};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

/// The Vita front touch panel is twice the screen in each axis (1920x1088 vs
/// 960x544), so a screen coordinate maps to panel coordinates by doubling.
const SCREEN_W: f64 = 960.0;
const SCREEN_H: f64 = 544.0;
const PANEL_SCALE: f64 = 2.0;

/// Live input state written by DOM event handlers and read by [`BrowserWorld`]. A
/// human's keyboard maps to `SceCtrlButtons` and a pointer press to a single front-
/// panel touch point.
#[derive(Clone, Copy, Default)]
pub struct InputState {
    /// Currently-held controller buttons (`SceCtrlButtons` bitmask).
    pub buttons: u32,
    /// The live front-panel touch, if a pointer is pressed.
    pub touch: Option<TouchFrame>,
    /// The live ANALOG sticks as `(x, y)` in the guest's 0..255 encoding, or `None` when
    /// nothing is driving them.
    ///
    /// # Why `Option` and not a centred default
    /// A centred stick is not "no stick": 128 is a VALUE, and writing it every frame would
    /// overwrite a scripted recipe's steering with neutral on every poll, which is the same
    /// mistake as an over-report. `None` means "this run has no live stick, use whatever the
    /// recipe says" - the identical rule `touch` already follows. It matters because one
    /// retail racer's whole steering is `lx` (memory `vitaslop-absolute-heading-control`), so a
    /// browser that can only send buttons cannot drive it at all.
    pub left_stick: Option<(u8, u8)>,
    pub right_stick: Option<(u8, u8)>,
}

/// A shared, thread-safe handle to the live input state.
pub type SharedInput = Arc<Mutex<InputState>>;

/// The `SceCtrlButtons` bit for a `KeyboardEvent.code`, or `None` if unmapped. A
/// small, ergonomic default map: arrows/WASD for the dpad, J/K/L/I (and Z/X) for the
/// face buttons, Q/E for the shoulder buttons, Enter for start, Right-Shift for select.
pub fn key_button(code: &str) -> Option<u32> {
    Some(match code {
        "ArrowUp" | "KeyW" => 0x0000_0010,    // up
        "ArrowRight" | "KeyD" => 0x0000_0020, // right
        "ArrowDown" | "KeyS" => 0x0000_0040,  // down
        "ArrowLeft" | "KeyA" => 0x0000_0080,  // left
        "KeyQ" => 0x0000_0100,                // L
        "KeyE" => 0x0000_0200,                // R
        "Enter" => 0x0000_0008,               // start
        "ShiftRight" => 0x0000_0001,          // select
        "KeyI" => 0x0000_8000,                // square
        "KeyL" => 0x0000_1000,                // triangle
        "KeyK" | "KeyX" => 0x0000_2000,       // circle
        "KeyJ" | "KeyZ" => 0x0000_4000,       // cross
        _ => return None,
    })
}

/// Microseconds per virtual 60Hz frame (matches the runtime's `RecipeWorld`), so a
/// title reading elapsed time still sees monotonic progress.
const FRAME_US: u64 = 16_666;

/// The world the live browser run polls: a virtual 60Hz clock, a seeded PRNG, and
/// input drawn from a scripted recipe (optional) merged with live pointer/keyboard.
/// Live input takes precedence when present, so a human can take over a scripted run.
pub struct BrowserWorld {
    recipe: Option<RecipeWorld>,
    live: SharedInput,
    /// The host's location provider (see [`crate::location`]). Read on every guest
    /// query rather than cached, so a fix cannot go stale behind the title's back.
    location: SharedLocation,
    /// Whether this run has asked the host to acquire position. Guards the outbound
    /// request so a title polling its permission dialog every frame - which is what the
    /// observed retail caller does - does not post a message per frame.
    location_requested: bool,
    monotonic_us: u64,
    wall_us: u64,
    rng: u64,
}

impl BrowserWorld {
    /// A world driven by `live` input, optionally overlaid on a scripted `recipe`
    /// (frame-keyed touch/button directives). Pass `None` for a purely live session.
    pub fn new(recipe: Option<RecipeWorld>, live: SharedInput, location: SharedLocation) -> Self {
        BrowserWorld {
            recipe,
            live,
            location,
            location_requested: false,
            monotonic_us: 0,
            wall_us: 1_500_000_000_000_000,
            rng: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

impl World for BrowserWorld {
    fn monotonic_us(&mut self) -> u64 {
        self.monotonic_us
    }
    fn wall_us(&mut self) -> u64 {
        self.wall_us
    }
    fn poll_ctrl(&mut self, port: u32) -> CtrlFrame {
        let mut f = self.recipe.as_mut().map(|r| r.poll_ctrl(port)).unwrap_or_default();
        // Merge live keyboard buttons on top of any scripted state.
        let live = self.live.lock().unwrap();
        f.buttons |= live.buttons;
        // A live stick OVERRIDES the scripted one, exactly as a live touch does - and only
        // while something is actually driving it. See `InputState::left_stick`.
        if let Some((x, y)) = live.left_stick {
            f.lx = x;
            f.ly = y;
        }
        if let Some((x, y)) = live.right_stick {
            f.rx = x;
            f.ry = y;
        }
        f
    }
    fn poll_touch(&mut self, port: u32) -> TouchFrame {
        // A live pointer press overrides the scripted touch; otherwise the recipe (if
        // any) drives, and failing that there is no finger down.
        if port == 0 {
            if let Some(t) = self.live.lock().unwrap().touch {
                return t;
            }
        }
        self.recipe.as_mut().map(|r| r.poll_touch(port)).unwrap_or_default()
    }
    fn location_permission(&mut self) -> LocationPermission {
        self.location.lock().unwrap().permission
    }
    fn request_location(&mut self) {
        if self.location_requested {
            return;
        }
        self.location_requested = true;
        crate::location::start_host_watch();
    }
    fn release_location(&mut self) {
        if !self.location_requested {
            return;
        }
        self.location_requested = false;
        crate::location::stop_host_watch();
    }
    fn poll_location(&mut self) -> Option<LocationFix> {
        self.location.lock().unwrap().fix
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        // SplitMix64, matching the runtime worlds: deterministic and cheap.
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
        self.monotonic_us = frame.wrapping_mul(FRAME_US);
        self.wall_us = 1_500_000_000_000_000u64.wrapping_add(self.monotonic_us);
    }
}

/// Attach pointer and keyboard listeners that write into `state`, so a human can play
/// the live run. Pointer presses on `canvas` become front-panel touches; key presses
/// become `SceCtrlButtons`. The listener closures are leaked (they live for the page's
/// lifetime, which is the run's lifetime), so nothing needs to hold them.
pub fn install_listeners(canvas: &HtmlCanvasElement, state: &SharedInput) {
    let window = web_sys::window().expect("window");
    let document = window.document().expect("document");

    // Map a pointer event's client coords to a front-panel touch via the canvas rect.
    let to_panel_touch = {
        let canvas = canvas.clone();
        move |client_x: f64, client_y: f64| -> TouchFrame {
            let rect = canvas.get_bounding_client_rect();
            let (rw, rh) = (rect.width().max(1.0), rect.height().max(1.0));
            let sx = ((client_x - rect.left()) / rw * SCREEN_W).clamp(0.0, SCREEN_W);
            let sy = ((client_y - rect.top()) / rh * SCREEN_H).clamp(0.0, SCREEN_H);
            TouchFrame::single((sx * PANEL_SCALE) as u16, (sy * PANEL_SCALE) as u16)
        }
    };

    // Pointer down / move (while pressed) set the touch; pointer up / leave clear it.
    let down = {
        let state = state.clone();
        let to_panel_touch = to_panel_touch.clone();
        Closure::wrap(Box::new(move |e: web_sys::PointerEvent| {
            state.lock().unwrap().touch = Some(to_panel_touch(e.client_x() as f64, e.client_y() as f64));
        }) as Box<dyn FnMut(_)>)
    };
    let mv = {
        let state = state.clone();
        let to_panel_touch = to_panel_touch.clone();
        Closure::wrap(Box::new(move |e: web_sys::PointerEvent| {
            // Only track movement while the primary button is held (buttons bit 0).
            if e.buttons() & 1 != 0 {
                state.lock().unwrap().touch = Some(to_panel_touch(e.client_x() as f64, e.client_y() as f64));
            }
        }) as Box<dyn FnMut(_)>)
    };
    let up = {
        let state = state.clone();
        Closure::wrap(Box::new(move |_e: web_sys::PointerEvent| {
            state.lock().unwrap().touch = None;
        }) as Box<dyn FnMut(_)>)
    };
    let target: &web_sys::EventTarget = canvas.as_ref();
    target.add_event_listener_with_callback("pointerdown", down.as_ref().unchecked_ref()).unwrap();
    target.add_event_listener_with_callback("pointermove", mv.as_ref().unchecked_ref()).unwrap();
    target.add_event_listener_with_callback("pointerup", up.as_ref().unchecked_ref()).unwrap();
    target.add_event_listener_with_callback("pointerleave", up.as_ref().unchecked_ref()).unwrap();
    down.forget();
    mv.forget();
    up.forget();

    // Keyboard: set/clear the mapped button bit on the shared state.
    let key_down = {
        let state = state.clone();
        Closure::wrap(Box::new(move |e: web_sys::KeyboardEvent| {
            if let Some(bit) = key_button(&e.code()) {
                state.lock().unwrap().buttons |= bit;
                e.prevent_default();
            }
        }) as Box<dyn FnMut(_)>)
    };
    let key_up = {
        let state = state.clone();
        Closure::wrap(Box::new(move |e: web_sys::KeyboardEvent| {
            if let Some(bit) = key_button(&e.code()) {
                state.lock().unwrap().buttons &= !bit;
            }
        }) as Box<dyn FnMut(_)>)
    };
    let doc_target: &web_sys::EventTarget = document.as_ref();
    doc_target.add_event_listener_with_callback("keydown", key_down.as_ref().unchecked_ref()).unwrap();
    doc_target.add_event_listener_with_callback("keyup", key_up.as_ref().unchecked_ref()).unwrap();
    key_down.forget();
    key_up.forget();
}

// ---------------------------------------------------------------------------
// Worker live input
// ---------------------------------------------------------------------------
//
// A Web Worker has no DOM, so it cannot attach its own listeners. Instead the page
// forwards each pointer/keyboard event to the worker as a message, and the worker's
// JS glue calls the exported functions below to update the same shared input cell the
// worker's `BrowserWorld` reads. `run_game_worker` registers the cell here first.

thread_local! {
    /// The live input cell the worker's world reads, set by `run_game_worker`. A
    /// thread-local (not a static) because each worker has its own single thread.
    static WORKER_INPUT: RefCell<Option<SharedInput>> = const { RefCell::new(None) };
}

/// Register the shared input cell the worker input entry points feed. Called once by
/// `run_game_worker` before the live loop starts.
pub fn set_worker_input(state: SharedInput) {
    WORKER_INPUT.with(|c| *c.borrow_mut() = Some(state));
}

/// Apply `f` to the registered worker input state, if any.
fn with_worker_input(f: impl FnOnce(&mut InputState)) {
    WORKER_INPUT.with(|c| {
        if let Some(state) = c.borrow().as_ref() {
            f(&mut state.lock().unwrap());
        }
    });
}

/// Worker input entry point: a key (`KeyboardEvent.code`) went down/up. The page
/// forwards its keydown/keyup here; unmapped keys are ignored.
#[wasm_bindgen]
pub fn worker_input_key(code: &str, pressed: bool) {
    if let Some(bit) = key_button(code) {
        with_worker_input(|st| {
            if pressed {
                st.buttons |= bit;
            } else {
                st.buttons &= !bit;
            }
        });
    }
}

/// Worker input entry point: one analog stick moved, or stopped being driven.
///
/// `stick` is 0 for the left stick and 1 for the right. `(x, y)` are the guest's own 0..255
/// encoding with 128 centred, so the page owns the deadzone and the curve (an on-screen thumb
/// stick and a real gamepad axis need different ones) and this layer stays a plain conduit.
/// `active` false releases the stick back to whatever a scripted recipe says, which is not the
/// same as centring it - see [`InputState::left_stick`].
///
/// An unknown `stick` index is IGNORED rather than silently treated as the left one: a caller
/// that means a stick this build does not have is a bug in the caller, and steering the wrong
/// axis is worse than steering none.
#[wasm_bindgen]
pub fn worker_input_stick(stick: u32, x: u8, y: u8, active: bool) {
    with_worker_input(|st| {
        let slot = match stick {
            0 => &mut st.left_stick,
            1 => &mut st.right_stick,
            _ => return,
        };
        *slot = active.then_some((x, y));
    });
}

/// Worker input entry point: a pointer press/move/release at screen coordinates
/// `(sx, sy)` in the 960x544 screen space (the page maps its canvas-relative pointer
/// into that space). `down` false lifts the finger. Converted here to a front-panel
/// touch (panel = screen * 2).
#[wasm_bindgen]
pub fn worker_input_pointer(sx: f64, sy: f64, down: bool) {
    with_worker_input(|st| {
        st.touch = if down {
            let x = (sx.clamp(0.0, SCREEN_W) * PANEL_SCALE) as u16;
            let y = (sy.clamp(0.0, SCREEN_H) * PANEL_SCALE) as u16;
            Some(TouchFrame::single(x, y))
        } else {
            None
        };
    });
}
