//! The live frame source: the guest running cooperatively on a fiber, stepped one
//! frame per window redraw with the real controller injected between frames. This
//! is what makes the desktop window *live* rather than canned playback - the guest
//! computes each frame on demand and reacts to input in real time (press START to
//! watch it tear down and exit). It wraps the native [`Scheduler`]; the window
//! only calls `advance` then `current`.

use std::time::Instant;

use vitaslop_loader as loader;
use vitaslop_native::{FrameStop, Scheduler};
use vitaslop_runtime::capture::Scene;
use vitaslop_runtime::CtrlFrame;

/// The committed clean-room cube (velf), embedded so the binary needs no assets.
const CUBE: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.velf");

/// The guest, stepped live. Owns the cooperative scheduler and tracks how many
/// frames it has produced and whether it has exited.
pub struct LiveGuest {
    sched: Scheduler,
    finished: bool,
    frames: u64,
    /// Transpile + instantiate time, measured once at construction.
    pub build_ms: f64,
}

impl LiveGuest {
    /// Load, transpile, and instantiate the cube for cooperative execution. The
    /// guest is suspended before its first instruction; the first `advance` runs
    /// init through the first frame.
    pub fn new() -> Result<LiveGuest, String> {
        let m = loader::load(CUBE).map_err(|e| format!("load cube.velf: {e:?}"))?;
        let inputs = m.program_inputs();
        let imports: Vec<(u32, u32)> =
            m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

        let t0 = Instant::now();
        let sched = Scheduler::new(
            &inputs.code,
            inputs.base,
            inputs.thumb_entry,
            &inputs.entries,
            &inputs.externs,
            inputs.mem_bytes,
            imports,
        )
        .map_err(|e| format!("build scheduler: {e:?}"))?;
        let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

        Ok(LiveGuest { sched, finished: false, frames: 0, build_ms })
    }

    /// Inject this frame's controller input and step the guest one frame. A frame
    /// that reaches its flip updates `current`; an exit marks the guest finished;
    /// a quantum preempt leaves the previous frame in place to resume next time.
    pub fn advance(&mut self, input: CtrlFrame) {
        if self.finished {
            return;
        }
        self.sched.set_input(input);
        match self.sched.run_frame() {
            FrameStop::Present => self.frames += 1,
            FrameStop::Preempted => {}
            FrameStop::Finished => self.finished = true,
        }
    }

    /// The most recent presented frame, or None before the first one.
    pub fn current(&self) -> Option<&Scene> {
        self.sched.current_scene()
    }

    /// True once the guest has exited (e.g. after START tore it down).
    pub fn finished(&self) -> bool {
        self.finished
    }

    /// Frames produced so far.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// The error that ended the run, if it ended in one.
    pub fn error(&self) -> Option<&str> {
        self.sched.error()
    }
}
