//! Determinism: record a run's World inputs, then replay them, and assert the
//! captured GXM stream is bit-identical. This is the core claim of the
//! determinism seam - all non-determinism enters through World, and everything
//! else (allocation, scheduling) is deterministic by construction - so replaying
//! the recorded World answers reproduces the run exactly.

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{
    capture::Scene, CtrlFrame, HostAbi, Record, Replay, VitaEnv, Vm, World, WorldEvent,
};

const CUBE: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.velf");

/// Presses a few buttons over the run so the recorded input is non-trivial, then
/// START to end the loop.
struct Script {
    polls: u32,
}
impl World for Script {
    fn monotonic_us(&mut self) -> u64 {
        self.polls as u64 * 16_666
    }
    fn wall_us(&mut self) -> u64 {
        1_500_000_000_000_000 + self.polls as u64 * 16_666
    }
    fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
        self.polls += 1;
        let mut f = CtrlFrame::default();
        // Wiggle the stick and hold a face button on some frames.
        f.lx = (128 + self.polls) as u8;
        if self.polls % 2 == 0 {
            f.buttons = 0x0000_4000; // SCE_CTRL_CROSS
        }
        if self.polls > 6 {
            f.buttons |= 0x0000_0008; // SCE_CTRL_START -> end loop
        }
        f
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (self.polls as u8).wrapping_add(i as u8);
        }
    }
}

/// Run the cube with `world`, returning the captured scenes and presents.
fn run(world: Box<dyn World + Send>) -> (Vec<Scene>, Vec<u32>) {
    let m = loader::load(CUBE).expect("load");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();
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
    let env = env.borrow();
    (env.state.capture.scenes.clone(), env.state.capture.presents.clone())
}

#[test]
fn record_then_replay_is_identical() {
    // Record a run.
    let recorder = Record::new(Script { polls: 0 });
    let log = recorder.events();
    let (scenes_rec, presents_rec) = run(Box::new(recorder));
    let events: Vec<WorldEvent> = log.lock().unwrap().clone();

    assert!(!events.is_empty(), "nothing was recorded");
    assert!(!scenes_rec.is_empty(), "recorded run drew nothing");

    // Replay the recorded inputs against a fresh run.
    let (scenes_replay, presents_replay) = run(Box::new(Replay::new(events.clone())));

    // The captured stream must be bit-identical.
    assert_eq!(scenes_rec, scenes_replay, "replay diverged from the recorded run");
    assert_eq!(presents_rec, presents_replay, "replay presents diverged");

    eprintln!(
        "recorded {} world events, {} scenes; replay reproduced them exactly",
        events.len(),
        scenes_rec.len()
    );
}
