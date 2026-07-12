//! A rough CPU-throughput read: run the cube's full frame loop and time it.
//! Native wasmtime, so this is an optimistic ceiling versus a mobile browser's
//! wasm engine, and the cube is a light CPU workload (matrix math plus a handful
//! of host calls per frame), so treat it as a floor-of-a-floor, not a verdict.
//! Run:
//!   cargo test --release -p vitaslop-conformance-harness --test cube_perf \
//!     -- --ignored --nocapture

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use vitaslop_loader as loader;
use vitaslop_native::{CtrlFrame, HostAbi, VitaEnv, Vm, World};

const CUBE: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.velf");

/// Never presses anything, so the cube runs its full built-in 600-frame loop and
/// then tears down and terminates.
struct Idle;
impl World for Idle {
    fn monotonic_us(&mut self) -> u64 {
        0
    }
    fn wall_us(&mut self) -> u64 {
        0
    }
    fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
        CtrlFrame::default()
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        buf.fill(0);
    }
}

#[test]
#[ignore]
fn cube_cpu_throughput() {
    let m = loader::load(CUBE).expect("load");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    // Time transpile + instantiate (the browser pays this once at module load).
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
    .expect("instantiate");
    let build = t0.elapsed();

    let mut env = VitaEnv::new(imports, inputs.base, inputs.mem_bytes, Box::new(Idle));
    env.state.halt_on_terminate = true;
    let env = Rc::new(RefCell::new(env));
    vm.set_import_env(Box::new(env.clone()));

    // Time the full run (600 frames of CPU + host calls).
    let t1 = Instant::now();
    vm.call(m.entry & !1).expect("run");
    let run = t1.elapsed();

    let env = env.borrow();
    let cap = &env.state.capture;
    let frames = cap.scenes.len().max(1);

    eprintln!("--- cube CPU throughput (native wasmtime, release) ---");
    eprintln!("transpile + instantiate: {:?}", build);
    eprintln!("run: {:?} for {} frames, {} host calls", run, frames, cap.call_count);
    eprintln!("per-frame: {:?}", run / frames as u32);
    eprintln!("frames/sec: {:.0}", frames as f64 / run.as_secs_f64());
}
