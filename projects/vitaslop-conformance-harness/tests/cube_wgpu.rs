//! Milestone 5, GPU path: render the captured cube with wgpu and cross-check it
//! against the software rasterizer (the oracle). Skips gracefully if no GPU
//! adapter is present. Set CUBE_PNG_DIR to dump GPU-rendered PNGs. Run:
//!   CUBE_PNG_DIR=/some/dir cargo test -p vitaslop-conformance-harness \
//!     --test cube_wgpu -- --nocapture

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{render, CtrlFrame, HostAbi, VitaEnv, Vm, WgpuRenderer, World};

const CUBE: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.velf");

const WIDTH: u32 = 960;
const HEIGHT: u32 = 544;
const CLEAR: [u8; 4] = [16, 16, 24, 255];
const FRAMES: u32 = 120;

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
fn cube_renders_on_gpu() {
    let renderer = match WgpuRenderer::new() {
        Some(r) => r,
        None => {
            eprintln!("no GPU adapter available - skipping wgpu render test");
            return;
        }
    };
    eprintln!("GPU: {}", renderer.adapter_name);

    // Run the cube and capture the GXM stream.
    let m = loader::load(CUBE).expect("load");
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
    .expect("instantiate");
    vm.set_import_env(Box::new(env.clone()));
    vm.call(m.entry & !1).expect("run");

    let env = env.borrow();
    let scenes = &env.state.capture.scenes;
    assert!(!scenes.is_empty(), "no scenes captured");

    let out_dir = std::env::var("CUBE_PNG_DIR").ok();
    if let Some(dir) = &out_dir {
        std::fs::create_dir_all(dir).expect("mkdir");
    }

    for (i, scene) in scenes.iter().enumerate() {
        let gpu = renderer.render_scene(scene, WIDTH, HEIGHT, CLEAR);
        let gpu_drawn = gpu.drawn_pixels(CLEAR);
        assert!(gpu_drawn > 20_000, "frame {i}: GPU drew only {gpu_drawn} pixels");
        assert_ne!(gpu.pixel(WIDTH / 2, HEIGHT / 2), CLEAR, "frame {i}: GPU center is background");

        if let Some(dir) = &out_dir {
            std::fs::write(format!("{dir}/gpu_{i:04}.png"), gpu.to_png()).expect("write");
        }
    }

    // Cross-check the GPU against the software oracle on a mid-spin frame: the two
    // rasterizers differ in fill/rounding rules, so the projected cube area should
    // agree closely, not bit-exactly.
    let mid = scenes.len() / 2;
    let gpu = renderer.render_scene(&scenes[mid], WIDTH, HEIGHT, CLEAR);
    let soft = render::render_scene(&scenes[mid], WIDTH, HEIGHT, CLEAR);
    let (gd, sd) = (gpu.drawn_pixels(CLEAR) as f64, soft.drawn_pixels(CLEAR) as f64);
    let ratio = gd / sd;
    eprintln!("mid frame: gpu drew {gd:.0}, software drew {sd:.0}, ratio {ratio:.3}");
    assert!(
        (0.9..1.1).contains(&ratio),
        "gpu/software cube area disagree: gpu={gd} software={sd}"
    );

    eprintln!("rendered {} frames on GPU ({})", scenes.len(), renderer.adapter_name);
}
