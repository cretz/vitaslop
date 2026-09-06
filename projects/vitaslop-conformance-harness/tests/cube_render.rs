//! Milestone 5: rasterize the captured cube stream into actual pixels. Runs the
//! cube through the real Vita host, then software-rasterizes each captured scene
//! (fixed-function equivalent of the placeholder shaders) and checks a real cube
//! was drawn. Set CUBE_PNG_DIR to also dump one PNG per frame for inspection:
//!   CUBE_PNG_DIR=/some/dir cargo test -p vitaslop-conformance-harness \
//!     --test cube_render -- --nocapture

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{render, CtrlFrame, HostAbi, VitaEnv, Vm, World};

const CUBE: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.velf");

const WIDTH: u32 = 960;
const HEIGHT: u32 = 544;
const CLEAR: [u8; 4] = [16, 16, 24, 255];
const FRAMES: u32 = 120;

/// Runs the cube's render loop for `frames` frames, then stops it.
struct RunFor {
    polls: u32,
    frames: u32,
}

impl World for RunFor {
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
            f.buttons = 0x0000_0008; // START
        }
        f
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        buf.fill(0);
    }
}

#[test]
fn cube_rasterizes_to_pixels() {
    let m = loader::load(CUBE).expect("load cube.velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    let world = Box::new(RunFor { polls: 0, frames: FRAMES });
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
    .expect("instantiate cube");
    vm.set_import_env(Box::new(env.clone()));
    vm.call(m.entry & !1).expect("run cube");

    let env = env.borrow();
    let scenes = &env.state.capture.scenes;
    assert!(scenes.len() as u32 >= FRAMES - 1, "expected ~{FRAMES} scenes, got {}", scenes.len());

    let out_dir = std::env::var("CUBE_PNG_DIR").ok();
    if let Some(dir) = &out_dir {
        std::fs::create_dir_all(dir).expect("create out dir");
    }

    let mut total_drawn = 0usize;
    for (i, scene) in scenes.iter().enumerate() {
        let fb = render::render_scene(scene, WIDTH, HEIGHT, CLEAR);
        let drawn = fb.drawn_pixels(CLEAR);
        total_drawn += drawn;

        // Every frame draws a substantial, centered cube.
        assert!(drawn > 20_000, "frame {i}: only {drawn} pixels drawn");
        let center = fb.pixel(WIDTH / 2, HEIGHT / 2);
        assert_ne!(center, CLEAR, "frame {i}: cube center is background");

        if let Some(dir) = &out_dir {
            let path = format!("{dir}/frame_{i:04}.png");
            std::fs::write(&path, fb.to_png()).expect("write png");
        }
    }

    // The cube spins, so the rendered pixel counts vary frame to frame (it is not
    // a static image). Confirm the projected area actually changes.
    let first = render::render_scene(&scenes[0], WIDTH, HEIGHT, CLEAR).drawn_pixels(CLEAR);
    let mid = render::render_scene(&scenes[scenes.len() / 2], WIDTH, HEIGHT, CLEAR)
        .drawn_pixels(CLEAR);
    assert_ne!(first, mid, "cube does not appear to rotate");

    eprintln!(
        "rendered {} frames, {} total drawn pixels, first={} mid={}",
        scenes.len(),
        total_drawn,
        first,
        mid
    );
}
