//! End-to-end: load the cube velf, transpile+execute its CPU code through the
//! real Vita host, and assert the captured GXM command stream. This is the
//! blob-free "it works" signal for milestone 4 - no GPU emulation, no pixels,
//! just the draw calls and buffers the guest issued.
//!
//! The guest entry is `_start`, which spins forever after `main` returns, so we
//! bound the frame loop by pressing START through the World seam and halt the run
//! on sceGxmTerminate (after teardown). Run with:
//!   cargo test -p vitaslop-conformance-harness --test cube_run

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{CtrlFrame, HostAbi, VitaEnv, Vm, World};

const CUBE: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.velf");

/// SCE_CTRL_START.
const START: u32 = 0x0000_0008;

/// A World that reports no buttons for the first `frames` polls, then presses
/// START so the cube's render loop breaks. This exercises the controller input
/// path and bounds the run deterministically.
struct PressStartAfter {
    polls: u32,
    frames: u32,
}

impl World for PressStartAfter {
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
            f.buttons = START;
        }
        f
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        buf.fill(0);
    }
}

#[test]
fn cube_runs_and_captures_gxm_stream() {
    let m = loader::load(CUBE).expect("load cube.velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    let world = Box::new(PressStartAfter { polls: 0, frames: 3 });
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

    let result = vm.call(m.entry & !1);

    let env = env.borrow();
    let cap = &env.state.capture;

    eprintln!(
        "calls={} scenes={} presents={} unimplemented={:?}",
        cap.call_count,
        cap.scenes.len(),
        cap.presents.len(),
        cap.unimplemented
    );
    result.expect("run cube main");

    // Every imported NID the cube used is implemented.
    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);

    // The render loop ran a few frames before START, each a full scene.
    assert!(cap.scenes.len() >= 3, "expected >= 3 scenes, got {}", cap.scenes.len());

    // Inspect the first scene's single draw: the cube is 12 triangles = 36
    // 16-bit indices over 8 interleaved position+color vertices (stride 16).
    let scene = &cap.scenes[0];
    assert!(scene.color.is_some(), "scene has no color surface");
    let color = scene.color.unwrap();
    assert_eq!(color.width, 960);
    assert_eq!(color.height, 544);
    assert_eq!(color.stride_pixels, 1024);

    assert_eq!(scene.draws.len(), 1, "expected one draw");
    let d = &scene.draws[0];
    assert_eq!(d.primitive, 0, "SCE_GXM_PRIMITIVE_TRIANGLES");
    assert_eq!(d.index_format, 0, "SCE_GXM_INDEX_FORMAT_U16");
    assert_eq!(d.index_count, 36);
    assert_eq!(d.indices.len(), 36 * 2);
    assert_eq!(d.vertex_stride, 16);
    assert_eq!(d.vertices.len(), 8 * 16, "8 vertices at stride 16");
    assert_eq!(d.attributes.len(), 2, "position + color attributes");
    // Attribute 0: float3 position at offset 0 (F32 = 9, 3 components).
    assert_eq!(d.attributes[0].offset, 0);
    assert_eq!(d.attributes[0].format, 9);
    assert_eq!(d.attributes[0].component_count, 3);
    // Attribute 1: 4x U8N color at offset 12 (U8N = 4, 4 components).
    assert_eq!(d.attributes[1].offset, 12);
    assert_eq!(d.attributes[1].format, 4);
    assert_eq!(d.attributes[1].component_count, 4);
    // The MVP uniform was captured (16 floats), and it is a real transform (not
    // all zero) since the cube computes perspective * view * model each frame.
    assert_eq!(d.uniforms.len(), 16, "4x4 MVP uniform");
    assert!(d.uniforms.iter().any(|&x| x != 0.0), "MVP is all zero");

    // The first vertex is the cube corner (-1,-1,-1) with color 0xff0000ff.
    let x = f32::from_le_bytes([d.vertices[0], d.vertices[1], d.vertices[2], d.vertices[3]]);
    let y = f32::from_le_bytes([d.vertices[4], d.vertices[5], d.vertices[6], d.vertices[7]]);
    let z = f32::from_le_bytes([d.vertices[8], d.vertices[9], d.vertices[10], d.vertices[11]]);
    assert_eq!((x, y, z), (-1.0, -1.0, -1.0));
    let color0 = u32::from_le_bytes([d.vertices[12], d.vertices[13], d.vertices[14], d.vertices[15]]);
    assert_eq!(color0, 0xff00_00ff);

    // The display queue presented buffers (double-buffered swap).
    assert!(!cap.presents.is_empty(), "no display presents recorded");
}
