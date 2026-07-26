//! What a run lets an observer SEE: sample a declared memory watch, render the
//! current frame to a PNG, and reduce the whole observable output to one
//! determinism signature.
//!
//! These three are the entire observation surface of a game run, and both drivers
//! share them - the batch [`recipe_runner`](crate::recipe_runner) and the resident
//! [`session`](crate::session). Keeping them here (rather than one copy per driver)
//! is what makes a signature printed by `play` and one printed by `session`
//! comparable, which is the property the whole equivalence-oracle idea rests on.

use std::path::{Path, PathBuf};

use vitaslop_runtime::recipe::WatchDecl;
use vitaslop_runtime::{render, VitaEnv};

use crate::ThreadedScheduler;

/// Front-panel render size and clear color (the retail titles present at 960x544).
pub const WIDTH: u32 = 960;
pub const HEIGHT: u32 = 544;
pub const CLEAR: [u8; 4] = [0, 0, 0, 255];

/// Sample one watched value from current guest memory, widened to `f64`. `None`
/// when the address is outside guest memory.
pub fn sample_watch(sched: &ThreadedScheduler<VitaEnv>, w: &WatchDecl) -> Option<f64> {
    let mut buf = [0u8; 4];
    let width = w.ty.width();
    if !sched.read_guest_into(w.addr, &mut buf[..width]) {
        return None;
    }
    w.ty.decode(&buf[..width])
}

/// Render the last captured scene to `<dir>/<name>.png`. Returns the written path,
/// or `None` if there is no scene yet or the write failed.
pub fn write_shot(
    sched: &ThreadedScheduler<VitaEnv>,
    shot_dir: Option<&Path>,
    name: &str,
) -> Option<PathBuf> {
    let dir = shot_dir?;
    let scene = {
        let host = sched.host();
        host.state.capture.scenes.last().cloned()
    }?;
    std::fs::create_dir_all(dir).ok()?;
    // Supersample the software shot (VITASLOP_SSAA=N): rasterize at N x native and
    // box-downsample. Antialiases the geometric aliasing of the heavily-tessellated
    // vehicle meshes (dozens of sub-pixel triangles per final pixel, plus coincident-
    // panel z-fighting) that one sample/pixel renders as speckle - a distant 3D vehicle
    // is unreadable at 1x and clean at 2x. A review shot is occasional, so the 4x fill
    // cost of 2x SSAA is immaterial; 2x is the quality default, overridable (1 disables,
    // higher for close scrutiny).
    let ssaa = std::env::var("VITASLOP_SSAA")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(2);
    let fb = render::render_scene_supersampled(&scene, WIDTH, HEIGHT, CLEAR, ssaa);
    let path = dir.join(format!("{name}.png"));
    std::fs::write(&path, fb.to_png()).ok()?;
    Some(path)
}

/// The determinism signature over a run's observable output. Delegates to
/// [`Capture::signature`](vitaslop_runtime::capture::Capture::signature), which is the
/// one definition - it also folds in scenes a bounded-retention run has already
/// evicted, so a long session and a short one agree.
pub fn signature(cap: &vitaslop_runtime::capture::Capture) -> u64 {
    cap.signature()
}

/// Format an `f64` compactly: integers without a trailing `.0`.
pub fn format_f64(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}
