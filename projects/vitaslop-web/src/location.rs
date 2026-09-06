//! The browser's location provider, feeding [`vitaslop_runtime::World::poll_location`].
//!
//! # Why this is split between the page and the worker
//! `navigator.geolocation` is exposed on `Window` ONLY - it does not exist in a
//! `WorkerGlobalScope`, so the worker that runs the emulator cannot call it at all. The
//! page therefore owns the real API and the worker owns the guest, exactly as it already
//! works for pointer/keyboard input and for audio (the page owns the `AudioContext`).
//! The two are joined by the same `postMessage` seam:
//!
//! ```text
//!  guest: sceLocationConfirm
//!    -> BrowserWorld::request_location
//!       -> (worker) postMessage {type:"location-request"} -> page
//!          -> navigator.geolocation.watchPosition   [raises the permission prompt]
//!             -> page postMessage {type:"location"} -> worker
//!                -> worker_location_* -> this shared cell
//!  guest: sceLocationGetLocation -> BrowserWorld::poll_location -> the cell
//! ```
//!
//! On the MAIN-THREAD engine (`run_game`, the no-worker path) there is no relay to make:
//! the same code calls `watchPosition` directly, because it is already on `Window`.
//! [`start_host_watch`] picks between the two by looking at what the global actually is,
//! so neither engine carries a flag saying which it is.
//!
//! # The permission prompt IS the guest's dialog
//! The browser raises its prompt on the first `watchPosition`, and the guest's
//! `sceLocationConfirm*` family reports exactly that prompt's progress - see
//! `vitaslop_runtime::vita::location`. Nothing here decides on the user's behalf: until
//! the browser answers, the cell holds [`LocationPermission::Pending`], which the guest
//! reads as a RUNNING dialog.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use vitaslop_runtime::world::HostLocation;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// Live location state written by the host (page listeners, or the worker's forwarded
/// messages) and read by `BrowserWorld`.
///
/// The state machine itself - which W3C error code is a refusal, what a NaN heading
/// means, when a fix must be dropped - lives in [`HostLocation`] in the runtime, where it
/// is testable on any host. This crate contributes only the DOM half.
pub type LiveLocation = HostLocation;

/// A shared, thread-safe handle to the live location state. `Arc<Mutex<_>>` rather than
/// `Rc<RefCell<_>>` for the same reason the input cell uses one: `World` is `Send`
/// because the native engine runs it across an OS thread. On the browser's single
/// thread the lock never contends.
pub type SharedLocation = Arc<Mutex<LiveLocation>>;

thread_local! {
    /// The cell this worker's world reads, registered by `run_game_worker` /`run_game`
    /// before the run starts, so the exported `worker_location_*` entry points can find
    /// it. A thread-local because each worker has its own single thread.
    static LOCATION: RefCell<Option<SharedLocation>> = const { RefCell::new(None) };

    /// The `watchPosition` id held by the MAIN-THREAD engine, so it can be cleared.
    /// Unused in the worker engine, where the page owns the watch.
    static WATCH_ID: RefCell<Option<i32>> = const { RefCell::new(None) };
}

/// Register the shared location cell. Called once before the live loop starts.
pub fn set_shared_location(state: SharedLocation) {
    LOCATION.with(|c| *c.borrow_mut() = Some(state));
}

/// Apply `f` to the registered cell, if any.
fn with_location(f: impl FnOnce(&mut LiveLocation)) {
    LOCATION.with(|c| {
        if let Some(state) = c.borrow().as_ref() {
            f(&mut state.lock().unwrap());
        }
    });
}

/// The worker global, or `None` when this is the page's main thread. This is the one
/// test that decides whether location is relayed or called directly.
fn worker_scope() -> Option<web_sys::DedicatedWorkerGlobalScope> {
    js_sys::global().dyn_into::<web_sys::DedicatedWorkerGlobalScope>().ok()
}

/// Begin acquiring position, raising the browser's permission prompt if it has not been
/// answered. Idempotent at the browser level (a second `watchPosition` does not re-prompt
/// once permission is decided), and the callers above it only fire it on a state change.
pub fn start_host_watch() {
    if let Some(scope) = worker_scope() {
        // Relay: only the page can touch navigator.geolocation.
        let msg = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&msg, &"type".into(), &"location-request".into());
        let _ = scope.post_message(&msg);
        return;
    }
    start_window_watch();
}

/// Stop acquiring position.
pub fn stop_host_watch() {
    if let Some(scope) = worker_scope() {
        let msg = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&msg, &"type".into(), &"location-release".into());
        let _ = scope.post_message(&msg);
        return;
    }
    WATCH_ID.with(|c| {
        if let Some(id) = c.borrow_mut().take() {
            if let Some(geo) = window_geolocation() {
                geo.clear_watch(id);
            }
        }
    });
}

/// `window.navigator.geolocation`, or `None` if this context has no Geolocation API at
/// all (which is a real state - an insecure origin does not expose it).
fn window_geolocation() -> Option<web_sys::Geolocation> {
    web_sys::window()?.navigator().geolocation().ok()
}

/// The MAIN-THREAD engine's own watch. Mirrors what `web/live.html` does for the worker
/// engine, so both reach the same cell through the same conversions.
fn start_window_watch() {
    let Some(geo) = window_geolocation() else {
        // No Geolocation API in this context. That is genuinely absent hardware as far
        // as a title can tell, and it is what `Unavailable` means.
        with_location(|l| l.set_unavailable());
        return;
    };
    // Already watching: do not stack a second watch.
    if WATCH_ID.with(|c| c.borrow().is_some()) {
        return;
    }
    // The prompt is up (or permission is already decided and the first callback is
    // imminent). Either way the guest's dialog is RUNNING until an answer arrives.
    with_location(|l| l.set_pending());

    let ok = Closure::wrap(Box::new(move |pos: web_sys::Position| {
        let c = pos.coords();
        apply_fix(
            c.latitude(),
            c.longitude(),
            c.altitude(),
            Some(c.accuracy()),
            c.heading(),
            c.speed(),
            pos.timestamp(),
        );
    }) as Box<dyn FnMut(web_sys::Position)>);

    let err = Closure::wrap(Box::new(move |e: web_sys::PositionError| {
        apply_error(e.code());
    }) as Box<dyn FnMut(web_sys::PositionError)>);

    let id = geo.watch_position_with_error_callback(
        ok.as_ref().unchecked_ref(),
        Some(err.as_ref().unchecked_ref()),
    );
    // The closures live for the page's lifetime, which is the run's lifetime.
    ok.forget();
    err.forget();
    if let Ok(id) = id {
        WATCH_ID.with(|c| *c.borrow_mut() = Some(id));
    }
}

/// Record a browser position error against the cell. The rules live in [`HostLocation`].
fn apply_error(code: u16) {
    with_location(|l| l.apply_w3c_error(code));
}

/// Record a fix against the cell, converting the W3C shape to the guest's.
#[allow(clippy::too_many_arguments)]
fn apply_fix(
    latitude: f64,
    longitude: f64,
    altitude: Option<f64>,
    accuracy: Option<f64>,
    heading: Option<f64>,
    speed: Option<f64>,
    timestamp_ms: f64,
) {
    with_location(|l| {
        l.apply_w3c_fix(latitude, longitude, altitude, accuracy, heading, speed, timestamp_ms)
    });
}

// ---------------------------------------------------------------------------
// Worker entry points, called by `web/worker.js` from the page's relayed messages.
// ---------------------------------------------------------------------------

/// A position from the page's `watchPosition`. `None` for any of the optional fields
/// means the browser could not supply it, and it stays unknown all the way to the
/// guest's INVALID sentinel.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn worker_location_fix(
    latitude: f64,
    longitude: f64,
    altitude: Option<f64>,
    accuracy: Option<f64>,
    heading: Option<f64>,
    speed: Option<f64>,
    timestamp_ms: f64,
) {
    apply_fix(latitude, longitude, altitude, accuracy, heading, speed, timestamp_ms);
}

/// A `PositionError` from the page's `watchPosition`, by its W3C `code`.
#[wasm_bindgen]
pub fn worker_location_error(code: u16) {
    apply_error(code);
}

/// The page reporting that this context has no Geolocation API at all (an insecure
/// origin, or a browser without it). Distinct from a refusal: nothing was asked.
#[wasm_bindgen]
pub fn worker_location_unavailable() {
    with_location(|l| l.set_unavailable());
}

/// A human-readable note from the page's geolocation relay.
///
/// It is emitted as a `tracing` WARN so it lands in the on-page WARN/ERROR mirror and in
/// the `/diag` sink alongside everything else. That matters because the note originates
/// on the PAGE, which on a phone has no console anyone is holding - and the failure this
/// replaces is a title sitting on "acquiring position" for ever with the reason (an
/// insecure origin, a refusal, a timeout) visible nowhere.
#[wasm_bindgen]
pub fn worker_location_note(message: &str) {
    tracing::warn!(target: "vitaslop::location", "{message}");
}

