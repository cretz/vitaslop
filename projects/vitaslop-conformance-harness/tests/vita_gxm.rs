//! End-to-end GXM render conformance on a REAL Vita executable. `gxmtri.velf`
//! (clean-room, MIT vita-headers, placeholder shaders) drives libgxm through init
//! -> one offscreen scene -> one indexed triangle draw -> finish -> exit. We run it
//! through the loader + transpiler + host NID layer under the plain single-thread
//! Vm, then render the CAPTURED GXM scene both ways (software rasterizer, and the
//! wgpu GPU path when an adapter is present) and assert the triangle appears with
//! the expected interpolated vertex colors.
//!
//! This is the render north star's runnable milestone: proof that our GXM->pixels
//! translation is faithful on a real artifact, not only on synthetic Rust scenes.
//! The draw uses NDC vertex positions and no uniform buffer, so it needs no shader
//! reflection - the capture recovers it from the app-declared vertex attributes and
//! the vertex/index streams alone (the parts independent of the absent shader blob).
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_gxm

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, GeneralRenderer, HostAbi, VitaEnv, Vm};
use vitaslop_runtime::render::{render_scene, Framebuffer};

const GXMTRI: &[u8] = include_bytes!("../../vitaslop-conformance-suite-vita/gxmtri-src/gxmtri.velf");

const W: u32 = 128;
const H: u32 = 128;
const CLEAR: [u8; 4] = [12, 12, 16, 255];

/// Count pixels differing from the clear color.
fn drawn(fb: &Framebuffer) -> usize {
    fb.drawn_pixels(CLEAR)
}

#[test]
fn gxm_triangle_renders() {
    let m = loader::load(GXMTRI).expect("load gxmtri.velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    let env = VitaEnv::new(
        imports,
        inputs.base,
        inputs.mem_bytes,
        Box::new(DeterministicWorld::default()),
    );
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
    .expect("instantiate gxmtri");
    vm.set_import_env(Box::new(env.clone()));

    vm.call(m.entry & !1).expect("run gxmtri main");

    let env = env.borrow();
    let cap = &env.state.capture;
    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);

    // The offscreen BeginScene/EndScene must have captured exactly one scene with the
    // one triangle draw.
    eprintln!("scenes captured: {}", cap.scenes.len());
    let scene = cap.scenes.first().expect("no GXM scene captured");
    eprintln!("draws in scene 0: {}", scene.draws.len());
    let d = scene.draws.first().expect("no draw in scene");
    assert_eq!(d.index_count, 3, "expected a 3-index triangle");
    assert!(d.uniforms.len() < 16, "triangle should carry no MVP uniform (NDC draw)");
    assert!(d.textures.is_empty(), "triangle is untextured");

    // Software rasterize the captured scene.
    let sw = render_scene(scene, W, H, CLEAR);
    let ds = drawn(&sw);
    eprintln!("software drawn pixels: {ds}");
    // A triangle spanning roughly the middle of a 128x128 target covers a few
    // thousand pixels; require a meaningful fill, not a stray pixel or a whiteout.
    assert!(ds > 1500 && ds < (W * H) as usize * 3 / 4, "unexpected triangle coverage: {ds}");

    // The centroid is interior and should carry a blend of the three corner colors
    // (red/green/blue), so every channel is present and it is not the clear color.
    let c = sw.pixel(64, 74);
    eprintln!("centroid pixel: {c:?}");
    assert_ne!([c[0], c[1], c[2], c[3]], CLEAR, "centroid should be painted");
    assert!(c[0] > 20 && c[1] > 20 && c[2] > 20, "centroid should blend all three corners: {c:?}");
    // A top corner (outside the upward triangle) stays clear.
    assert_eq!(sw.pixel(2, 2), CLEAR, "corner (2,2) should be background");

    // GPU path: the wgpu renderer must agree with the software oracle on the same
    // real-GXM capture (skipped cleanly when no adapter is present, like the other
    // GPU probes).
    if let Some(mut gpu) = GeneralRenderer::new() {
        eprintln!("adapter: {}", gpu.adapter_name);
        let hw = gpu.render_scene(scene, W, H, CLEAR);
        let dh = drawn(&hw);
        eprintln!("gpu drawn pixels: {dh}");
        let sum: u64 = sw
            .rgba
            .iter()
            .zip(&hw.rgba)
            .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
            .sum();
        let mean = sum as f64 / sw.rgba.len() as f64;
        eprintln!("mean_abs_diff sw vs gpu: {mean:.3}");
        assert!(mean < 6.0, "GPU diverges from software oracle on real GXM capture: {mean:.3}");
        let ratio = ds.min(dh) as f64 / ds.max(dh) as f64;
        assert!(ratio > 0.9, "drawn-pixel counts disagree: sw={ds} hw={dh}");
    } else {
        eprintln!("no GPU adapter; software-only GXM render check");
    }
}
