//! **gxmconf's REAL-SHADER scenes, on the recompiled path.**
//!
//! `vita_gxmconf.rs` asserts the app's placeholder-shader scenes, which the renderer can only
//! reconstruct fixed-function. Scenes 9 and 10 draw through a real, authored vertex + fragment
//! pair (`gxmconf-src/gxmconf_shaders.h`) and are asserted HERE, in their own test binary,
//! because the recompiled path is a process-wide switch (`VITASLOP_GXP_LIVE`): with it on, a
//! placeholder pair cannot be recompiled and its draws are dropped, so the two groups cannot
//! share a process.
//!
//! GPU only: the software oracle has no recompiled path.
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_gxmconf_real

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, GeneralRenderer, HostAbi, VitaEnv, Vm};
use vitaslop_runtime::render::Framebuffer;

const GXMCONF: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/gxmconf-src/gxmconf.velf");

const W: u32 = 128;
const H: u32 = 128;
const CLEAR: [u8; 4] = [12, 12, 16, 255];
const HALF_W: u32 = W / 2;

/// Run the app with the recompiled path on and render scene `index` on the GPU, or `None`
/// without an adapter.
fn render_on_gpu(index: usize) -> Option<Framebuffer> {
    render_frame_on_gpu(index..index + 1)
}

/// Run the app and render the scenes `range` as ONE FRAME through one renderer - what a pair of
/// scenes needs when the second samples a target only the first rendered (on the GPU, never in
/// guest memory). Returns the frame's final framebuffer, or `None` without an adapter.
fn render_frame_on_gpu(range: std::ops::Range<usize>) -> Option<Framebuffer> {
    vitaslop_runtime::knobs::set_override("VITASLOP_GXP_LIVE", "1");
    let m = loader::load(GXMCONF).expect("load gxmconf.velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> = m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();
    let env = VitaEnv::new(imports, inputs.base, inputs.mem_bytes, Box::new(DeterministicWorld::default()));
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
    .expect("instantiate gxmconf");
    vm.set_import_env(Box::new(env.clone()));
    vm.call(m.entry & !1).expect("run gxmconf main");
    let env = env.borrow();
    let cap = &env.state.capture;
    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert_eq!(cap.scenes.len(), 18, "gxmconf emits 18 scenes");
    let scenes = &cap.scenes[range.clone()];
    assert!(
        scenes.iter().flat_map(|s| s.draws.iter()).all(|d| !d.vprog.is_empty() && !d.fprog.is_empty()),
        "scenes {range:?} carry draws with no program bytes - the recompiled path is not on"
    );
    let mut gpu = GeneralRenderer::new()?;
    // >>> NOT ON WINDOWS' SOFTWARE ADAPTER. The Windows CI runner has no GPU, so this is WARP,
    // and D3D12 compiles these recompiled shaders with FXC: MEASURED 1,058 s for this one test
    // binary (the whole Linux test step is 1.9 min). The D3D12 path is still covered there by
    // every other GPU test, and these scenes run on Linux's software Vulkan and macOS's Metal.
    if cfg!(windows) && gpu.software {
        eprintln!("software adapter on Windows ({}): scenes {range:?} not checked here", gpu.adapter_name);
        return None;
    }
    Some(if scenes.len() == 1 {
        gpu.render_scene(&scenes[0], W, H, CLEAR)
    } else {
        gpu.render_frame(scenes, W, H, CLEAR)
    })
}

/// **SCENE 9 - A REVERSED DEPTH RANGE, THROUGH REAL SHADERS.**
///
/// Background depth 0.0 and GREATER_EQUAL, the near quad (z 0.9, green) over the right half
/// drawn FIRST, then the far one (z 0.1, red) over everything: left red, right STAYS green.
/// This is a fighting title's (PCSE00235) whole world. Run with
/// `VITASLOP_GXM_BACKGROUND_DEPTH=0` it must fail (nothing painted).
#[test]
fn a_reversed_depth_range_holds_through_real_shaders() {
    let Some(fb) = render_on_gpu(9) else {
        eprintln!("no GPU adapter; scene 9 not checked");
        return;
    };
    let left = fb.pixel(HALF_W / 2, H / 2);
    let right = fb.pixel(HALF_W + HALF_W / 2, H / 2);
    assert!(
        left != CLEAR || right != CLEAR,
        "NOTHING painted - every fragment failed GREATER_EQUAL: the pass started its depth from 1.0, not the surface's background 0.0"
    );
    assert!(left[0] > left[1] && left != CLEAR, "the left half should be the FAR quad's red, got {left:?}");
    assert!(
        right[1] > right[0],
        "the right half should STAY the near quad's green - red means the depth test was ignored or run the wrong way; got {right:?}"
    );
}

/// **SCENE 10 - A U8_A SURFACE STORES THE FRAGMENT'S ALPHA.**
///
/// A half-alpha red quad (`0x800000ff`) into a single-channel `U8_A` surface: the one channel
/// must hold the ALPHA, 0x80, where a renderer treating it as red stores 0xff. The renderer
/// keeps such a surface as an RGBA8 target whose RED channel is the one channel, so that byte
/// is what is read. Run with `VITASLOP_GXM_ALPHA_SINGLE=0` it must fail.
#[test]
fn a_u8_a_surface_stores_alpha() {
    let Some(fb) = render_on_gpu(10) else {
        eprintln!("no GPU adapter; scene 10 not checked");
        return;
    };
    let p = fb.pixel(W / 2, H / 2);
    assert!(
        p[0].abs_diff(0x80) <= 1,
        "the one channel holds {:#04x}, expected the alpha 0x80 (0xff = red was stored); pixel {p:?}",
        p[0]
    );
}

/// **SCENES 11 + 12 - A CUBE THE GUEST RENDERS AS ONE STACKED TARGET IS SAMPLED AS A CUBE.**
///
/// Scene 11 renders six differently coloured 16x16 faces into one 16x128 surface, face `k` in
/// rows `16k..16k+16`; scene 12 binds that memory as a 16x16 CUBE and samples it along +X, +Y
/// and +Z in three strips, which must read faces 0, 2 and 4 - red, green, blue. A fighting
/// title (PCSE00235) renders its image-based lighting this way. The faces exist only on the
/// GPU, so the two scenes are rendered as ONE frame; sampled from guest memory the strips are
/// black, and a wrong face order shows the wrong colour.
#[test]
fn a_stacked_render_target_is_sampled_as_a_cube() {
    let Some(fb) = render_frame_on_gpu(11..13) else {
        eprintln!("no GPU adapter; scenes 11-12 not checked");
        return;
    };
    let y = H / 2;
    let strips = [(W / 6, "left +X", 0usize), (W / 2, "middle +Y", 1), (5 * W / 6, "right +Z", 2)];
    let mut failures = Vec::new();
    for (x, name, channel) in strips {
        let p = fb.pixel(x, y);
        let dominant = p[channel] > 200 && (0..3).filter(|&c| c != channel).all(|c| p[c] < 60);
        if !dominant {
            failures.push(format!("{name}: expected face {} ({}), got {p:?}", channel * 2, ["red", "green", "blue"][channel]));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// **SCENE 13 - AN UNTYPED POSITION IS FETCHED AS ITS RAW WORDS.**
///
/// A full green quad whose position attribute is declared `SCE_GXM_ATTRIBUTE_FORMAT_UNTYPED`.
/// A fighting title's (PCSE00235) engine clears its targets with such a triangle; a fetch that
/// read UNTYPED as zero collapsed it to a point, the clear drew nothing, and a map it clears to
/// alpha 1 stayed at alpha 0 - which its character shader multiplies every colour by.
#[test]
fn an_untyped_position_is_fetched_as_its_raw_words() {
    let Some(fb) = render_on_gpu(13) else {
        eprintln!("no GPU adapter; scene 13 not checked");
        return;
    };
    let p = fb.pixel(W / 2, H / 2);
    assert!(p[1] > 200 && p[0] < 60 && p[2] < 60, "the untyped quad did not paint green: {p:?}");
}

/// **SCENE 14 - THE COLOUR MASK NUMBERS ALPHA AS BIT 0.**
///
/// White through `colorMask = R` over the left half, through `colorMask = A` over the right.
/// Left must be red over the clear's green and blue; right must keep the clear's colour. A
/// fighting title's (PCSE00235) hair pass (mask RGB, alpha kept) lost its red to the old
/// R-first reading and came out cyan. Run with `VITASLOP_GXM_COLOR_MASK_ORDER=0` it must fail.
#[test]
fn the_colour_mask_numbers_alpha_as_bit_zero() {
    let Some(fb) = render_on_gpu(14) else {
        eprintln!("no GPU adapter; scene 14 not checked");
        return;
    };
    let left = fb.pixel(HALF_W / 2, H / 2);
    let right = fb.pixel(HALF_W + HALF_W / 2, H / 2);
    let mut failures = Vec::new();
    if !(left[0] > 200 && left[1] < 60 && left[2] < 60) {
        failures.push(format!("left (mask R) should be red over the clear, got {left:?}"));
    }
    if right[..3] != CLEAR[..3] {
        failures.push(format!("right (mask A) must leave the clear's RGB untouched, got {right:?}"));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// **SCENE 15 - A COPY OF A RE-FORMATTED TEXTURE READS THE NEW FORMAT.**
///
/// A CPU-filled cube initialised as U2F10F10F10, set to A8B8G8R8, then copied; the COPY is
/// bound, so only its control words say what it is. The +X, +Y, +Z strips must read faces 0, 2
/// and 4 - red, green, blue. A SetFormat that leaves the init's extension bit in word 0 makes
/// the copy decode as another format entirely (PCSE00235's lighting cubes).
#[test]
fn a_copy_of_a_reformatted_texture_reads_the_new_format() {
    let Some(fb) = render_on_gpu(15) else {
        eprintln!("no GPU adapter; scene 15 not checked");
        return;
    };
    let y = H / 2;
    let strips = [(W / 6, "left +X", 0usize), (W / 2, "middle +Y", 1), (5 * W / 6, "right +Z", 2)];
    let mut failures = Vec::new();
    for (x, name, channel) in strips {
        let p = fb.pixel(x, y);
        let dominant = p[channel] > 200 && (0..3).filter(|&c| c != channel).all(|c| p[c] < 60);
        if !dominant {
            failures.push(format!("{name}: expected face {} ({}), got {p:?}", channel * 2, ["red", "green", "blue"][channel]));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// **SCENES 16 + 17 - A TEXTURE NAMING A SUB-RECTANGLE OF A RENDERED TARGET SAMPLES THAT RECTANGLE.**
///
/// Scene 16 renders four quadrants into a 64x64 target (bottom-right yellow); scene 17 binds a
/// 32x32 LINEAR_STRIDED texture at the bottom-right quadrant's INTERIOR address with the
/// target's pitch and samples it at uv (0.25, 0.25). Every pixel must be yellow. Binding the
/// whole target instead reads the top-left quadrant's red - a fighting title's (PCSE00235)
/// bloom chain, whose levels share one surface, did that. The target exists only on the GPU,
/// so both scenes render as ONE frame. Run with `VITASLOP_RTT_SUBRECT=0` it must fail.
#[test]
fn a_texture_naming_a_sub_rectangle_of_a_target_samples_that_rectangle() {
    let Some(fb) = render_frame_on_gpu(16..18) else {
        eprintln!("no GPU adapter; scenes 16-17 not checked");
        return;
    };
    let p = fb.pixel(W / 2, H / 2);
    assert!(p[0] > 200 && p[1] > 200 && p[2] < 60, "expected the bottom-right quadrant's yellow, got {p:?}");
}
