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

/// Reads guest memory through a scheduler, for the shared recipe evaluator.
struct SchedRead<'a>(&'a ThreadedScheduler<VitaEnv>);

impl vitaslop_runtime::recipe_eval::GuestRead for SchedRead<'_> {
    fn read_into(&self, addr: u32, out: &mut [u8]) -> bool {
        self.0.read_guest_into(addr, out)
    }
}

/// Sample one watched value from current guest memory, widened to `f64`. `None`
/// when the address is outside guest memory.
///
/// Delegates to the SHARED sampler, so a `@watch` means the same thing in a resident
/// session, in a native recipe run and in the browser. A second copy of four lines of
/// decode looks harmless right up to the point where one of them is fixed.
pub fn sample_watch(sched: &ThreadedScheduler<VitaEnv>, w: &WatchDecl) -> Option<f64> {
    vitaslop_runtime::recipe_eval::sample_watch(&SchedRead(sched), w)
}

/// Render the current frame to `<dir>/<name>.png`. Returns the written path, or `None`
/// if there is no scene yet or the write failed.
///
/// The WHOLE frame, not its last scene. A 3D title builds a frame from several passes -
/// the world and its shadow/reflection/post targets offscreen, then a composite - and
/// rendering only the last one draws only the composite, whose sampled world target the
/// software path never filled. That is a live HUD over black: a shot that is not merely
/// imperfect but actively misleading, because it looks like a title that renders nothing.
/// See [`render::render_frame_chain`].
pub fn write_shot(
    sched: &ThreadedScheduler<VitaEnv>,
    shot_dir: Option<&Path>,
    name: &str,
) -> Option<PathBuf> {
    let dir = shot_dir?;
    let scenes: Vec<_> = {
        let host = sched.host();
        host.state.capture.frame_scenes().to_vec()
    };
    if scenes.is_empty() {
        return None;
    }
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
    let fb = render::render_frame_chain(&scenes, WIDTH, HEIGHT, CLEAR, ssaa);
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

/// Format an `f64` compactly: integers without a trailing `.0`. The shared formatter, so
/// a value reads identically in a session, a recipe report and the browser.
pub fn format_f64(x: f64) -> String {
    vitaslop_runtime::recipe_eval::format_f64(x)
}
