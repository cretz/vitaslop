//! The cooperative scheduler: run the cube on a wasmtime async fiber, stepped one
//! frame per `run_frame`, with input injected live between frames. Two claims:
//!
//! 1. **Equivalence** - a no-input scheduled run is bit-identical to the sync
//!    run-to-completion capture. The fiber path computes the same frames as the
//!    straight-line `Vm`; suspending and resuming at each flip changes nothing.
//! 2. **Live input** - input set between frames reaches the guest's `poll_ctrl`:
//!    pressing START ends the run (the guest tears down), on the exact frame the
//!    scripted world would have.
//!
//! Run: cargo test -p vitaslop-conformance-harness --test cube_scheduler

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{
    capture::Scene, CtrlFrame, FrameStop, HostAbi, Scheduler, VitaEnv, Vm, World,
};

const CUBE: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.velf");

/// SCE_CTRL_START.
const START: u32 = 0x0000_0008;

/// Number of frames to render before pressing START.
const FRAMES: usize = 30;

/// The scripted world for the baseline sync run: no buttons for `frames` polls,
/// then START. Identical in convention to the scheduler's LiveWorld, so a no-input
/// live run and this run see the same input and clock sequence.
struct PressStartAfter {
    polls: u32,
    frames: u32,
}

impl World for PressStartAfter {
    fn monotonic_us(&mut self) -> u64 {
        self.polls as u64 * 16_666
    }
    fn wall_us(&mut self) -> u64 {
        1_500_000_000_000_000 + self.polls as u64 * 16_666
    }
    fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
        self.polls += 1;
        let mut f = CtrlFrame::default();
        if self.polls > self.frames {
            f.buttons = START;
        }
        f
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        buf.fill(0);
    }
}

/// The straight-line reference: run the cube to completion via the sync `Vm`,
/// return the captured scenes.
fn baseline_scenes() -> Vec<Scene> {
    let m = loader::load(CUBE).expect("load");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();
    let world = Box::new(PressStartAfter { polls: 0, frames: FRAMES as u32 });
    let mut env = VitaEnv::new(imports, inputs.base, inputs.mem_bytes, world);
    env.state.halt_on_terminate = true;
    let env = Rc::new(RefCell::new(env));
    let mut vm = Vm::new(
        &inputs.code,
        inputs.base,
        inputs.thumb_entry,
        &inputs.entries,
        &inputs.externs,
        inputs.mem_bytes,
        &HostAbi::default(),
    )
    .expect("instantiate");
    vm.set_import_env(Box::new(env.clone()));
    vm.call(m.entry & !1).expect("run");
    env.borrow().state.capture.scenes.clone()
}

/// Drive the cube through the scheduler, feeding no input for `FRAMES` frames then
/// START, collecting each presented scene.
fn scheduled_scenes() -> Vec<Scene> {
    let m = loader::load(CUBE).expect("load");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();
    let mut sched = Scheduler::new(
        &inputs.code,
        inputs.base,
        inputs.thumb_entry,
        &inputs.entries,
        &inputs.externs,
        inputs.mem_bytes,
        imports,
    )
    .expect("build scheduler");

    let mut scenes = Vec::new();
    loop {
        // No buttons until we have the FRAMES scenes, then press START.
        let mut input = CtrlFrame::default();
        if scenes.len() >= FRAMES {
            input.buttons = START;
        }
        sched.set_input(input);

        match sched.run_frame() {
            FrameStop::Present => {
                scenes.push(sched.current_scene().expect("a scene").clone());
            }
            FrameStop::Finished => break,
            FrameStop::Preempted => panic!("cube frame exceeded the quantum budget"),
        }

        // Safety net so a bug cannot loop forever.
        assert!(scenes.len() < FRAMES + 8, "guest did not stop after START");
    }
    if let Some(e) = sched.error() {
        panic!("scheduled run errored: {e}");
    }
    scenes
}

#[test]
fn scheduled_run_matches_sync_and_stops_on_live_start() {
    let baseline = baseline_scenes();
    assert!(baseline.len() >= FRAMES, "baseline only produced {} scenes", baseline.len());

    let scheduled = scheduled_scenes();

    // Exactly the frames rendered before START, and each a real cube (a triangle-
    // list draw the shared pipeline accepts).
    assert_eq!(scheduled.len(), FRAMES, "scheduler produced {} scenes", scheduled.len());
    for (i, s) in scheduled.iter().enumerate() {
        assert!(!s.draw_batches().is_empty(), "frame {i} has no drawable batch");
    }

    // Equivalence: the fiber path computes the same frames as the sync path.
    for i in 0..FRAMES {
        assert_eq!(scheduled[i], baseline[i], "scheduled frame {i} differs from sync");
    }

    // The cube spins: first and middle frames differ.
    assert_ne!(scheduled[0], scheduled[FRAMES / 2], "cube does not appear to rotate");

    eprintln!("scheduler produced {FRAMES} frames, bit-identical to the sync run, stopped on live START");
}

/// The fuel quantum: with a small enough quantum the guest yields mid-frame (a
/// preemptive yield, not a blocking call), and the scheduler must resume it and
/// still deliver the same frame. A tiny quantum plus a tiny per-frame cap forces
/// a `Preempted` stop, which the caller then resumes to completion.
#[test]
fn quantum_preempts_and_resumes_without_changing_frames() {
    let baseline = baseline_scenes();

    let m = loader::load(CUBE).expect("load");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    // A small quantum (yield every ~2000 retired instructions) with a low per-frame
    // cap, so the long first frame (init + first render) cannot finish in one
    // `run_frame` and must be resumed across several `Preempted` stops.
    let mut sched = Scheduler::with_quantum(
        2_000,
        4,
        &inputs.code,
        inputs.base,
        inputs.thumb_entry,
        &inputs.entries,
        &inputs.externs,
        inputs.mem_bytes,
        imports,
    )
    .expect("build scheduler");

    let mut scenes = Vec::new();
    let mut preempts = 0u32;
    let mut guard = 0u32;
    loop {
        let mut input = CtrlFrame::default();
        if scenes.len() >= FRAMES {
            input.buttons = START;
        }
        sched.set_input(input);
        match sched.run_frame() {
            FrameStop::Present => scenes.push(sched.current_scene().expect("scene").clone()),
            FrameStop::Preempted => preempts += 1,
            FrameStop::Finished => break,
        }
        guard += 1;
        assert!(guard < 1_000_000, "did not converge");
    }

    // Preemption actually happened (the quantum fired mid-frame)...
    assert!(preempts > 0, "expected the small quantum to force preemptions");
    // ...yet resuming across quanta produced exactly the same frames as the
    // uninterrupted sync run. Preemption is transparent to the guest.
    assert_eq!(scenes.len(), FRAMES);
    for i in 0..FRAMES {
        assert_eq!(scenes[i], baseline[i], "preempted frame {i} differs from sync");
    }

    eprintln!("quantum forced {preempts} preemptions; frames stayed bit-identical to the sync run");
}
