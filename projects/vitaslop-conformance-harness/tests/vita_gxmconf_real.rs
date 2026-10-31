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
use vitaslop_runtime::capture::Scene;
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
    let all = captured_scenes(&format!("scenes {range:?}"))?;
    let scenes = &all[range.clone()];
    assert!(
        scenes.iter().flat_map(|s| s.draws.iter()).all(|d| !d.vprog.is_empty() && !d.fprog.is_empty()),
        "scenes {range:?} carry draws with no program bytes - the recompiled path is not on"
    );
    let mut gpu = GeneralRenderer::new()?;
    Some(if scenes.len() == 1 {
        gpu.render_scene(&scenes[0], W, H, CLEAR)
    } else {
        gpu.render_frame(scenes, W, H, CLEAR)
    })
}

/// Render each entry of `frames` as one FRAME, in order, through ONE renderer - what a scene
/// needs when it samples a target only an EARLIER frame rendered. Returns the last frame's
/// framebuffer, or `None` without an adapter.
fn render_frames_on_gpu(frames: &[Vec<Scene>]) -> Option<Framebuffer> {
    let mut gpu = GeneralRenderer::new()?;
    let mut last = None;
    for scenes in frames {
        last = Some(gpu.render_frame(scenes, W, H, CLEAR));
    }
    last
}

/// Run the app with the recompiled path on and return every scene it captured, or `None` where
/// these scenes are not checked (no adapter, or Windows' software one). `what` names the
/// scenes for that message.
fn captured_scenes(what: &str) -> Option<Vec<Scene>> {
    // >>> NOT ON WINDOWS' SOFTWARE ADAPTER, AND DECIDED BEFORE ANYTHING IS BUILT. The Windows CI
    // runner's only adapter is the Microsoft Basic Render Driver (its `gpu_probe` step says so),
    // where D3D12 compiles in software with FXC: MEASURED 1,058 s for this one test binary against
    // 1.9 min for the whole Linux test step. Building the renderer with the recompiled path on is
    // itself minutes there, so the adapter is asked first, not the renderer. The D3D12 path is
    // still covered on that runner by every other GPU test, and these scenes run on Linux's
    // software Vulkan, macOS's Metal and any Windows machine with a GPU.
    let (adapter, software) = vitaslop_native::wgpu_render::general_adapter()?;
    if cfg!(windows) && software {
        eprintln!("software adapter on Windows ({adapter}): {what} not checked here");
        return None;
    }
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
    assert_eq!(cap.scenes.len(), 21, "gxmconf emits 21 scenes");
    Some(cap.scenes.clone())
}

/// **SCENE 9 - A REVERSED DEPTH RANGE, THROUGH REAL SHADERS.**
///
/// Background depth 0.0 and GREATER_EQUAL, the near quad (z 0.9, green) over the right half
/// drawn FIRST, then the far one (z 0.1, red) over everything: left red, right STAYS green.
/// This is a fighting title's whole world. Run with
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
/// title renders its image-based lighting this way. The faces exist only on the
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
/// A fighting title's engine clears its targets with such a triangle; a fetch that
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
/// fighting title's hair pass (mask RGB, alpha kept) lost its red to the old
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
/// the copy decode as another format entirely (a fighting title's lighting cubes).
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
/// whole target instead reads the top-left quadrant's red - a fighting title's
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

/// Scenes 16 and 17 as captured, plus a scene drawing into the display - so a frame holding
/// scene 16 alone still ends on the display and leaves the quadrant target an ordinary target.
fn quadrant_scenes() -> Option<(Scene, Scene, Scene)> {
    let all = captured_scenes("scenes 16-18")?;
    Some((all[16].clone(), all[17].clone(), all[18].clone()))
}

/// Scene 17 moved to the quadrant target's BASE address - its 32x32 top-left window, at the
/// target's pitch - and sampled at uv (0.75, 0.75) instead of (0.25, 0.25): the window's texel
/// (24, 24), inside the top-left RED quadrant. The whole 64x64 target at that uv is (48, 48),
/// the bottom-right YELLOW one, so the two readings cannot agree.
fn sample_the_base_window(mut s17: Scene) -> Scene {
    // The bottom-right quadrant's offset from the base: 32 rows of 64 texels, then 32 texels.
    const QUADRANT: u32 = (32 * 64 + 32) * 4;
    let d = &mut s17.draws[0];
    let mut textures = d.textures.to_vec();
    textures[0].data_addr -= QUADRANT;
    d.textures = textures.into();
    // The uv is the colour attribute's red and green bytes (0x40 = 0.25 in the app).
    let colour = d.attributes.iter().find(|a| a.format == 4 && a.component_count == 4).expect("a U8N colour attribute").offset;
    let mut v = d.vertices.to_vec();
    for rec in v.chunks_mut(d.vertex_stride as usize) {
        let o = colour as usize;
        if rec.len() >= o + 2 {
            rec[o] = 0xc0;
            rec[o + 1] = 0xc0;
        }
    }
    d.vertices = v.into();
    s17
}

/// **SCENES 16 + 17, AT THE BASE - A TEXTURE AT A TARGET'S OWN ADDRESS, SMALLER THAN IT, IS ITS
/// TOP-LEFT WINDOW.** Scene 17's texture moved to the base address and sampled at uv (0.75,
/// 0.75), in one frame with scene 16: every pixel must be the top-left quadrant's red. Binding
/// the whole target reads the bottom-right yellow - an action title's sky pass, reading the
/// top-left level of a bloom atlas, painted every level of it instead. Run with
/// `VITASLOP_RTT_SUBRECT_HELD=0` it must fail.
#[test]
fn a_texture_at_a_targets_base_smaller_than_it_samples_its_top_left_window() {
    let Some((s16, s17, _)) = quadrant_scenes() else {
        eprintln!("no GPU adapter; scenes 16-17 at the base not checked");
        return;
    };
    let Some(fb) = render_frames_on_gpu(&[vec![s16, sample_the_base_window(s17)]]) else { return };
    let p = fb.pixel(W / 2, H / 2);
    assert!(p[0] > 200 && p[1] < 60 && p[2] < 60, "expected the top-left quadrant's red, got {p:?}");
}

/// **SCENES 16 + 17, ACROSS FRAMES - A SUB-RECTANGLE OF A TARGET AN EARLIER FRAME RENDERED IS
/// STILL THAT RECTANGLE.** Scene 16 renders the quadrants on one frame (ending on a display
/// scene); the next frame's scene 17 samples the bottom-right window, and then the base window
/// as above. The target is held on the GPU from the earlier frame; its window must be read
/// exactly as one this frame rendered: yellow, then red. Binding the whole held target reads
/// red, then yellow - the action title's sky pass fed every bloom level back into the frame,
/// smaller each time. Run with `VITASLOP_RTT_SUBRECT_HELD=0` it must fail. The INTERIOR window
/// of a held target is checked only with `VITASLOP_RTT_SUBRECT_HELD_INTERIOR=1` - off by
/// default, see `subrect_held_interior`.
#[test]
fn a_sub_rectangle_of_a_target_held_from_an_earlier_frame_samples_that_rectangle() {
    let Some((s16, s17, s18)) = quadrant_scenes() else {
        eprintln!("no GPU adapter; scenes 16-17 across frames not checked");
        return;
    };
    let base = sample_the_base_window(s17.clone());
    if std::env::var("VITASLOP_RTT_SUBRECT_HELD_INTERIOR").is_ok_and(|v| v.trim() == "1") {
        let Some(interior) = render_frames_on_gpu(&[vec![s16.clone(), s18.clone()], vec![s17]]) else { return };
        let p = interior.pixel(W / 2, H / 2);
        assert!(p[0] > 200 && p[1] > 200 && p[2] < 60, "interior window: expected the bottom-right quadrant's yellow, got {p:?}");
    }
    let Some(at_base) = render_frames_on_gpu(&[vec![s16, s18], vec![base]]) else { return };
    let p = at_base.pixel(W / 2, H / 2);
    assert!(p[0] > 200 && p[1] < 60 && p[2] < 60, "base window: expected the top-left quadrant's red, got {p:?}");
}

/// **SCENE 18 - A DEFAULT UNIFORM BUFFER IS CONTEXT STATE, READ AT THE DRAWING PROGRAM'S SIZE.**
///
/// Each stage's buffer is set with `sceGxmSet{Vertex,Fragment}DefaultUniformBuffer` while
/// ANOTHER program is bound (one declaring 16 default-uniform registers), then the program that
/// reads it (32 registers, the colour at register 16) is set, then the draw. Left half must be
/// the vertex buffer's green and right half the fragment buffer's blue, every pixel. A binding
/// owned by the program bound at the set is judged stale for the drawing one and reads the empty
/// bank: both halves come out all-zero black. A 2D title's clear took its depth from such a bank,
/// wrote 0, and every later draw failed the depth test.
#[test]
fn a_default_uniform_buffer_outlives_a_program_change() {
    let Some(fb) = render_on_gpu(18) else {
        eprintln!("no GPU adapter; scene 18 not checked");
        return;
    };
    let mut failures = Vec::new();
    for (name, x0, x1, want) in [
        ("left (vertex stage)", 0, HALF_W, [0u8, 255, 0, 255]),
        ("right (fragment stage)", HALF_W, W, [0, 0, 255, 255]),
    ] {
        let wrong = (0..H).flat_map(|y| (x0..x1).map(move |x| (x, y))).filter(|&(x, y)| fb.pixel(x, y) != want).count();
        if wrong != 0 {
            failures.push(format!(
                "{name}: {wrong} of {} pixels are not {want:?} - mid pixel {:?} (all-zero means the \
                 binding was read as another program's and dropped)",
                (x1 - x0) * H,
                fb.pixel((x0 + x1) / 2, H / 2)
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
