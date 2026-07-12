//! End-to-end from a real title container: load the cube as an fSELF
//! `eboot.bin` (the form a shipped Vita app actually is), unwrap it, transpile,
//! execute through the Vita host, and assert the same GXM command stream the
//! bare-velf path produces. This proves the SELF/fSELF loader closes the gap
//! between "our hand-made velf" and "a real eboot.bin".
//!
//!   cargo test -p vitaslop-conformance-harness --test cube_eboot

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{CtrlFrame, HostAbi, VitaEnv, Vm, World};

/// The cube wrapped as an unencrypted fSELF by vita-make-fself: uncompressed
/// and zlib-compressed (`-c`). Both must run the full path identically.
const CUBE_EBOOT: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.eboot.bin");
const CUBE_EBOOT_C: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/cube-src/cube.eboot_c.bin");

/// SCE_CTRL_START.
const START: u32 = 0x0000_0008;

/// Presses START after a few frames so the cube's render loop terminates.
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
fn cube_runs_from_uncompressed_eboot() {
    run_eboot(CUBE_EBOOT);
}

#[test]
fn cube_runs_from_compressed_eboot() {
    run_eboot(CUBE_EBOOT_C);
}

/// Load a cube eboot container, run the whole CPU->GXM path, and assert the
/// captured stream matches the bare-velf path (see cube_run.rs).
fn run_eboot(eboot: &[u8]) {
    // The only difference from cube_run is the input bytes: a SELF container.
    // load() unwraps (and inflates, if compressed) it to the inner velf.
    assert!(loader::self_::is_self(eboot), "eboot is a SELF container");
    let m = loader::load(eboot).expect("load cube eboot");
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
    result.expect("run cube main");

    // Every imported NID resolved, and the render loop produced real scenes -
    // identical structure to the velf path (see cube_run.rs).
    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert!(cap.scenes.len() >= 3, "expected >= 3 scenes, got {}", cap.scenes.len());

    let scene = &cap.scenes[0];
    let color = scene.color.expect("scene has a color surface");
    assert_eq!((color.width, color.height, color.stride_pixels), (960, 544, 1024));

    assert_eq!(scene.draws.len(), 1, "expected one draw");
    let d = &scene.draws[0];
    assert_eq!(d.index_count, 36, "12 triangles = 36 indices");
    assert_eq!(d.vertices.len(), 8 * 16, "8 vertices at stride 16");
    assert_eq!(d.uniforms.len(), 16, "4x4 MVP uniform");

    // First cube corner (-1,-1,-1), color 0xff0000ff - same as the velf path.
    let x = f32::from_le_bytes([d.vertices[0], d.vertices[1], d.vertices[2], d.vertices[3]]);
    let y = f32::from_le_bytes([d.vertices[4], d.vertices[5], d.vertices[6], d.vertices[7]]);
    let z = f32::from_le_bytes([d.vertices[8], d.vertices[9], d.vertices[10], d.vertices[11]]);
    assert_eq!((x, y, z), (-1.0, -1.0, -1.0));

    assert!(!cap.presents.is_empty(), "no display presents recorded");
}
