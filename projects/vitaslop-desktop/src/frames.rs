//! Where the window gets each frame's scene. Task-1 has one implementation:
//! [`Playback`] over the pre-captured scenes. The cooperative scheduler will add
//! a live implementation that steps the guest one frame per `advance`, feeding it
//! the real [`CtrlFrame`] through `poll_ctrl`. The window loop only ever calls
//! `advance` then `current`, so it never changes when the live source lands.

use vitaslop_runtime::capture::Scene;
use vitaslop_runtime::CtrlFrame;

/// The window's source of per-frame scenes.
pub trait FrameSource {
    /// Advance one frame, consuming the current input. A live guest steps its CPU
    /// here; playback just moves to the next captured scene.
    fn advance(&mut self, input: CtrlFrame);
    /// The scene to present right now.
    fn current(&self) -> &Scene;
}

/// Loop the pre-captured scenes. The input is not consumed (the guest already ran
/// to completion), but the window still surfaces it in the title so the real
/// input -> SceCtrl mapping is visible and verifiable before the live path exists.
pub struct Playback {
    scenes: Vec<Scene>,
    idx: usize,
}

impl Playback {
    pub fn new(scenes: Vec<Scene>) -> Self {
        Playback { scenes, idx: 0 }
    }
}

impl FrameSource for Playback {
    fn advance(&mut self, _input: CtrlFrame) {
        self.idx = self.idx.wrapping_add(1);
    }
    fn current(&self) -> &Scene {
        &self.scenes[self.idx % self.scenes.len()]
    }
}
