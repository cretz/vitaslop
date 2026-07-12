//! Run the committed clean-room cube to completion on the native wasmtime engine
//! and hand back the captured GXM scenes. This is the native twin of the browser
//! `run_cube_cpu`: load the velf, transpile ARM/Thumb/VFP to wasm, execute the
//! guest CPU through the real Vita host, and collect the per-frame capture.
//!
//! It runs run-to-completion up front (a scripted input world bounds the loop),
//! exactly like the browser's first milestone. Live per-frame execution - where
//! the window's real input reaches `poll_ctrl` - is the cooperative-scheduler
//! milestone; until then the window plays these captured scenes back.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use vitaslop_loader as loader;
use vitaslop_native::{CtrlFrame, HostAbi, VitaEnv, Vm, World};
use vitaslop_runtime::capture::Scene;

/// The committed clean-room cube (velf), embedded so the binary needs no assets.
const CUBE: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.velf");

/// Frames of guest execution to run up front. Enough captured scenes for a smooth
/// several-second playback loop.
const FRAMES: u32 = 300;

/// A scripted input world for the run-to-completion pass: a virtual 60 Hz clock,
/// no input for `frames` polls, then START so the cube runs its clean teardown
/// (`sceGxmTerminate`), which halts the run. The deterministic twin of the native
/// render test's `RunFor` and the browser's `ScriptedWorld`.
struct ScriptedWorld {
    polls: u32,
    frames: u32,
}

impl World for ScriptedWorld {
    fn monotonic_us(&mut self) -> u64 {
        self.polls as u64 * 16_666
    }
    fn wall_us(&mut self) -> u64 {
        0
    }
    fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
        self.polls += 1;
        let mut f = CtrlFrame::default();
        if self.polls > self.frames {
            f.buttons = crate::input::SCE_CTRL_START;
        }
        f
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        buf.fill(0);
    }
}

/// The captured scenes plus the timings that make the perf story concrete.
pub struct CubeRun {
    pub scenes: Vec<Scene>,
    pub transpile_ms: f64,
    pub run_ms: f64,
    pub frames: u32,
}

/// Load, transpile, and run the cube to completion, returning the captured scenes
/// and timings.
pub fn run_cube() -> Result<CubeRun, String> {
    let m = loader::load(CUBE).map_err(|e| format!("load cube.velf: {e:?}"))?;
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    let world = Box::new(ScriptedWorld { polls: 0, frames: FRAMES });
    let mut env = VitaEnv::new(imports, inputs.base, inputs.mem_bytes, world);
    env.state.halt_on_terminate = true;
    let env = Rc::new(RefCell::new(env));

    let t0 = Instant::now();
    let mut vm = Vm::new(
        &inputs.code,
        inputs.base,
        inputs.thumb_entry,
        &inputs.entries,
        &inputs.externs,
        inputs.mem_bytes,
        &HostAbi::default(),
    )
    .map_err(|e| format!("instantiate cube: {e:?}"))?;
    let transpile_ms = t0.elapsed().as_secs_f64() * 1000.0;

    vm.set_import_env(Box::new(env.clone()));

    let t1 = Instant::now();
    vm.call(m.entry & !1).map_err(|e| format!("run cube: {e:?}"))?;
    let run_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let scenes = env.borrow().state.capture.scenes.clone();
    if scenes.is_empty() {
        return Err("cube produced no scenes".to_string());
    }
    Ok(CubeRun { scenes, transpile_ms, run_ms, frames: FRAMES })
}
