//! **GXM PIPELINE-STATE conformance on a real Vita executable.**
//!
//! `gxmconf.velf` (clean-room, MIT vita-headers, placeholder shaders) drives libgxm through one
//! offscreen SCENE PER FEATURE, each drawing axis-aligned NDC quads chosen so that what should
//! appear is a statement about a RECTANGLE. We run it through the loader + transpiler + host NID
//! layer, then render each captured scene and assert that rectangle.
//!
//! # What this checks that nothing else does
//!
//! The shader conformance suite (`vitaslop-gxp-shader/tests/conformance.rs`) authors USSE
//! programs and checks that the shader the GPU runs computes what the program MEANS. It says
//! nothing about the state machine AROUND the shader - which pixels a draw may touch, what the
//! depth test does, how the result is combined with what is already there - and every one of
//! those has cost this project a title-level debugging session:
//!
//! * A fighting title's main screen rendered grey because a REGION CLIP set for a 128x128 atlas
//!   was still in force for the frame buffer, and it FIT (scene 1).
//! * A football title's stadium crowd rasterised nothing for three sessions - 23 draws and
//!   59,202 indices a frame - while every downstream cause was measured and refused one at a
//!   time, before a position probe showed the vertices were above the top of the screen
//!   (scene 2).
//!
//! # No golden image, and no tolerance on the geometry
//!
//! Every assertion is a COUNT or a NAMED PIXEL, because the quads are axis-aligned and the
//! colours are flat: "the left half is painted and the right half is not" is exact. A test that
//! compared against a stored image would fail on any renderer change and would need a human to
//! say whether the new picture was better - which is how a picture test stops being run.
//!
//! Run with: cargo test -p vitaslop-conformance-harness --test vita_gxmconf

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, GeneralRenderer, HostAbi, VitaEnv, Vm};
use vitaslop_runtime::render::{render_scene, Framebuffer};

const GXMCONF: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/gxmconf-src/gxmconf.velf");

const W: u32 = 128;
const H: u32 = 128;
const CLEAR: [u8; 4] = [12, 12, 16, 255];
/// The app's own `HALF_W`: the boundary scenes 1 and 5 put their edge on.
const HALF_W: u32 = W / 2;

/// The scenes the app emits, in order, with what each exists to check. Kept beside the
/// assertions because a scene index on its own says nothing about what failed.
const SCENES: [&str; 8] = [
    "baseline: one quad over the whole viewport",
    "region clip: the same quad clipped to the left half",
    "above the viewport: a quad entirely past y = +1",
    "depth test: a far quad, then a near quad over its right half",
    "alpha blend: an opaque quad, then a half-alpha quad over its right half",
    "viewport: the full quad through a half-width viewport",
    "viewport AND region clip, over different rectangles: only the top-left quarter",
    "stencil mask: a NEVER/fail-REPLACE mark, then an EQUAL fill: only the bottom-right quarter",
];

/// Is this pixel something other than the clear colour?
fn painted(fb: &Framebuffer, x: u32, y: u32) -> bool {
    fb.pixel(x, y) != CLEAR
}

/// WHERE a scene painted, in one line: the count in each quadrant-edge half plus the pixel at
/// the middle of each half.
///
/// A mean-abs-diff says two backends disagree and nothing more, and "the GPU applies no region
/// clip" was READ OFF THE SOURCE from such a mean - wrongly, as the renderer's own
/// `VITASLOP_GXP_VP_TRACE` then showed it issuing exactly the right scissor. A shape is what
/// separates "clipped the wrong half", "painted the whole target" and "painted nothing", and
/// every one of those is a different defect.
fn shape(fb: &Framebuffer) -> String {
    let (l, r) = (painted_in_columns(fb, 0, HALF_W), painted_in_columns(fb, HALF_W, W));
    let mut top = 0;
    for y in 0..H / 2 {
        for x in 0..W {
            if painted(fb, x, y) {
                top += 1;
            }
        }
    }
    format!(
        "painted {:5} (left {l:5} right {r:5} top {top:5} bottom {:5})  mid-left {:?} mid-right {:?}",
        l + r,
        l + r - top,
        fb.pixel(HALF_W / 2, H / 2),
        fb.pixel(HALF_W + HALF_W / 2, H / 2),
    )
}

/// How many pixels of the given column range are painted, over every row.
fn painted_in_columns(fb: &Framebuffer, x0: u32, x1: u32) -> usize {
    let mut n = 0;
    for y in 0..H {
        for x in x0..x1 {
            if painted(fb, x, y) {
                n += 1;
            }
        }
    }
    n
}

/// Make the RENDERER'S OWN diagnostics visible in this test run.
///
/// Every `report!` in `vitaslop_platform::gpu` is a `tracing::debug`, and a test binary
/// installs no subscriber - so `VITASLOP_GXP_VP_TRACE=1` printed nothing at all while a
/// scene's applied scissor was being attributed by reading source instead. A report at debug
/// with nobody listening is a diagnostic that does not exist, and a conformance failure that
/// cannot be attributed is a guess. `try_init` because every test here calls
/// `run_and_render` and only one of them can install the global default.
fn show_renderer_diagnostics() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            vitaslop_runtime::knobs::log_filter(),
        ))
        .with_writer(std::io::stderr)
        .try_init();
}

/// Run the app once and render every captured scene, software and (when an adapter is present)
/// on the GPU. Returns one framebuffer per scene from the software oracle, plus the GPU's.
fn run_and_render() -> (Vec<Framebuffer>, Option<Vec<Framebuffer>>) {
    show_renderer_diagnostics();
    let m = loader::load(GXMCONF).expect("load gxmconf.velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> = m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

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
    .expect("instantiate gxmconf");
    vm.set_import_env(Box::new(env.clone()));
    vm.call(m.entry & !1).expect("run gxmconf main");

    let env = env.borrow();
    let cap = &env.state.capture;
    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert_eq!(
        cap.scenes.len(),
        SCENES.len(),
        "the app emits one scene per feature; captured {} of {}",
        cap.scenes.len(),
        SCENES.len()
    );

    let sw: Vec<Framebuffer> =
        cap.scenes.iter().map(|s| render_scene(s, W, H, CLEAR)).collect();
    // >>> ONE RENDERER PER SCENE, AND THAT IS NOT AN OPTIMISATION QUESTION - IT IS WHAT MAKES
    // >>> EACH SCENE'S PICTURE ITS OWN.
    //
    // All six scenes render into the SAME guest colour surface, so the address is the same
    // every time, and `GxmRenderer` keeps one image per display address on purpose: a render
    // target LOADS across frames rather than clearing, because the tiler does and a title's
    // accumulating atlas depends on it. Rendering the six scenes through one renderer
    // therefore composites them: scene 1's clipped-away right half came back holding SCENE
    // 0's green, and scene 2 - which correctly drew nothing - came back holding all of scene
    // 1's. Read as a picture that is two clipping defects, and the notes carried both as
    // MEASURED renderer gaps for a session. Neither exists; the renderer's own
    // `VITASLOP_GXP_VP_TRACE` was issuing exactly the right scissor while the test called it
    // ignored.
    //
    // A fresh renderer per scene gives each scene the FRESH image the software oracle gives
    // it, which is what makes the two comparable at all. It costs an adapter per scene, which
    // at six scenes is the difference between 1.1 s and ~2 s and buys the whole comparison.
    let hw = GeneralRenderer::new().map(|gpu| {
        eprintln!("adapter: {}", gpu.adapter_name);
        drop(gpu);
        cap.scenes
            .iter()
            .map(|s| {
                let mut gpu = GeneralRenderer::new().expect("an adapter was present a moment ago");
                gpu.render_scene(s, W, H, CLEAR)
            })
            .collect()
    });
    (sw, hw)
}

/// **WHAT THE CAPTURE ACTUALLY HOLDS, PER SCENE.** Run this before believing any failure above.
///
/// A scene assertion fails for two completely different reasons and the picture cannot tell
/// them apart: the app did not issue what it meant to (a defect in `gxmconf.c`, i.e. in the
/// instrument), or the renderer did not honour what it was given. This prints the draws and the
/// per-draw state so a failure can be attributed before it is reported as a renderer gap.
///
/// ```text
/// cargo test -p vitaslop-conformance-harness --test vita_gxmconf -- --ignored --nocapture what
/// ```
#[test]
#[ignore = "diagnostic: prints the captured scenes rather than asserting a picture"]
fn what_the_capture_holds_per_scene() {
    let m = loader::load(GXMCONF).expect("load gxmconf.velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> = m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();
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
    .expect("instantiate gxmconf");
    vm.set_import_env(Box::new(env.clone()));
    vm.call(m.entry & !1).expect("run gxmconf main");

    let env = env.borrow();
    let cap = &env.state.capture;
    println!("\n=== {} scene(s) captured ===", cap.scenes.len());
    for (i, scene) in cap.scenes.iter().enumerate() {
        println!(
            "\nscene {i}: {} draw(s)   completed_early {}   {}",
            scene.draws.len(),
            scene.completed_early,
            SCENES.get(i).copied().unwrap_or("(unnamed)")
        );
        for (d, draw) in scene.draws.iter().enumerate() {
            let rs = &draw.render_state;
            let b = &draw.blend;
            println!(
                "  draw {d}: {} indices, {} vertex bytes",
                draw.index_count,
                draw.vertices.len()
            );
            println!(
                "      clip mode {:#010x} rect {:?} | viewport_enable {} viewport {:?}",
                rs.region_clip_mode, rs.region_clip, rs.viewport_enable, rs.viewport
            );
            println!(
                "      depth func {:#010x} write {:#010x} | blend mask {:#04x} \
                 colour {}<-{} alpha {}<-{}",
                rs.front_depth_func,
                rs.front_depth_write,
                b.color_mask,
                b.color_src,
                b.color_dst,
                b.alpha_src,
                b.alpha_dst
            );
            println!(
                "      stencil func {:#010x} ops fail {} zfail {} pass {} | compare {:#04x} \
                 write {:#04x} ref {}",
                rs.front_stencil_func,
                rs.front_stencil_op_fail,
                rs.front_stencil_op_depth_fail,
                rs.front_stencil_op_depth_pass,
                rs.front_stencil_compare_mask,
                rs.front_stencil_write_mask,
                rs.front_stencil_ref
            );
        }
    }
}

/// **SCENE 0 - THE BASELINE.** A quad spanning the whole viewport paints the whole target.
///
/// Its own test, and first, because every later scene's coverage number is read against it: a
/// full-screen quad that does not cover the target makes "the left half is painted" meaningless.
#[test]
fn a_full_viewport_quad_paints_every_pixel() {
    let (sw, _) = run_and_render();
    let fb = &sw[0];
    let covered = painted_in_columns(fb, 0, W);
    assert_eq!(
        covered,
        (W * H) as usize,
        "{}: a full-viewport quad left {} pixels unpainted",
        SCENES[0],
        (W * H) as usize - covered
    );
}

/// **SCENE 1 - A REGION CLIP DISCARDS WHAT FALLS OUTSIDE IT.**
///
/// The same quad as scene 0, with the clip set to the left half. Both halves are asserted: a
/// renderer that ignored the clip paints the right half too, and one that dropped the draw
/// paints neither - and only checking both tells those apart.
#[test]
fn a_region_clip_confines_a_draw_to_its_rectangle() {
    let (sw, _) = run_and_render();
    let fb = &sw[1];
    let left = painted_in_columns(fb, 0, HALF_W);
    let right = painted_in_columns(fb, HALF_W, W);
    assert_eq!(left, (HALF_W * H) as usize, "{}: the clipped-to half is not fully painted", SCENES[1]);
    assert_eq!(right, 0, "{}: {right} pixels painted OUTSIDE the region clip", SCENES[1]);
}

/// **SCENE 2 - GEOMETRY OUTSIDE THE VIEWPORT PAINTS NOTHING.**
///
/// >>> THIS IS A FOOTBALL TITLE'S STADIUM CROWD, AS A TEST.
///
/// That defect cost three sessions: 23 draws and 59,202 indices rasterising nothing every frame,
/// with the sampler, the alpha test, the depth test and the geometry each measured and refused
/// in turn before a position probe showed the vertices were simply off the top of the screen.
///
/// The failure this guards against is not "nothing is drawn" - that is the CORRECT result here -
/// it is a renderer that CLAMPS out-of-range geometry instead of clipping it, which would paint
/// a band along the top edge and look, from a distance, like scenery.
#[test]
fn geometry_above_the_viewport_paints_nothing() {
    let (sw, _) = run_and_render();
    let fb = &sw[2];
    let covered = painted_in_columns(fb, 0, W);
    assert_eq!(
        covered, 0,
        "{}: {covered} pixels painted by geometry entirely above y = +1 - \
         the mark of a renderer that CLAMPS out-of-range geometry rather than clipping it",
        SCENES[2]
    );
}

/// **SCENE 3 - THE DEPTH TEST LETS THE NEARER SURFACE WIN.**
///
/// A far quad over everything, then a near quad over its right half. Both halves are named
/// colours, so a depth test that rejected everything (the right half stays far-coloured) and one
/// that accepted everything are distinguishable, and neither can pass by painting nothing.
#[test]
fn the_depth_test_lets_the_nearer_draw_win() {
    let (sw, _) = run_and_render();
    let fb = &sw[3];
    let left = fb.pixel(HALF_W / 2, H / 2);
    let right = fb.pixel(HALF_W + HALF_W / 2, H / 2);
    assert_ne!(left, CLEAR, "{}: the far quad did not paint the left half", SCENES[3]);
    assert_ne!(right, CLEAR, "{}: nothing painted the right half", SCENES[3]);
    assert!(
        left[0] > left[1],
        "{}: the left half should still be the FAR quad's red, got {left:?}",
        SCENES[3]
    );
    assert!(
        right[1] > right[0],
        "{}: the right half should be the NEAR quad's green - a depth test that rejected it \
         would leave the far quad's red here, got {right:?}",
        SCENES[3]
    );
}

/// **SCENE 4 - A BLEND COMBINES THE SOURCE WITH WHAT IS ALREADY THERE.**
///
/// An opaque blue quad, then a half-alpha red quad over its right half through a fragment
/// program carrying a real `SceGxmBlendInfo`. The overlap must be NEITHER colour: a renderer
/// that ignored the blend paints pure red, one that dropped the draw leaves pure blue, and a
/// single-colour assertion could not tell those apart.
///
/// The blend is a property of the FRAGMENT PROGRAM on this hardware rather than of a context
/// call, which is itself worth pinning - a renderer looking for a blend state on the context
/// would find none and draw every transparent thing opaque.
#[test]
fn a_blend_combines_the_source_with_the_destination() {
    let (sw, _) = run_and_render();
    let fb = &sw[4];
    let plain = fb.pixel(HALF_W / 2, H / 2);
    let blended = fb.pixel(HALF_W + HALF_W / 2, H / 2);
    assert_ne!(plain, CLEAR, "{}: the opaque quad did not paint", SCENES[4]);
    assert_ne!(blended, CLEAR, "{}: nothing painted the blended half", SCENES[4]);
    assert_ne!(
        blended, plain,
        "{}: the blended half is identical to the unblended one - the second draw did nothing",
        SCENES[4]
    );
    assert!(
        blended[0] > 40 && blended[2] > 40,
        "{}: a half-alpha red over opaque blue must carry BOTH colours; got {blended:?}, \
         which is one of them alone",
        SCENES[4]
    );
}

/// **SCENE 5 - A VIEWPORT TRANSFORMS COORDINATES INTO ITS OWN RECTANGLE.**
///
/// The same visible result as scene 1 by a completely different mechanism - a clip DISCARDS
/// fragments, a viewport TRANSFORMS coordinates - so the pair separates a renderer that
/// implements one and silently ignores the other.
///
/// The offsets are in pixels and the scales are half-extents, which is the convention a
/// transform read the other way round gets exactly wrong: at double the intended scale the quad
/// covers the whole target, which is what the right-half assertion catches.
///
/// >>> **THIS FOUND A REAL GAP IN BOTH BACKENDS AT ONCE: NEITHER APPLIED THE GUEST VIEWPORT.**
///
/// The guest sets `[xOff 32, xScale 32, yOff 64, yScale -64, zOff 0.5, zScale 0.5]` and the
/// capture carried it faithfully all along - `what_the_capture_holds_per_scene` prints exactly
/// those six numbers. Both paths then painted all 16,384 pixels: `render.rs::project` mapped an
/// `Space::Ndc` vertex straight onto the full target and read the viewport only to INFER a
/// target extent, while the GPU's fixed-function arm passed the whole attachment on the
/// reasoning that that path "packs its own screen-space geometry and has never carried a
/// viewport".
///
/// Fixed by giving GXM's offset/half-scale convention ONE statement -
/// `vitaslop_platform::gpu::viewport_rect` - that the software rasteriser and both GPU arms
/// read. The rectangle also BOUNDS the draw (on the GPU the clip volume maps onto the viewport,
/// so ndc past +-1 falls outside it), which the software path applies by intersecting it into
/// the draw's scissor. Scene 5 now matches scene 1's picture by the other mechanism, which is
/// what the pair was built to check.
#[test]
fn a_viewport_maps_the_draw_into_its_own_rectangle() {
    let (sw, _) = run_and_render();
    let fb = &sw[5];
    let left = painted_in_columns(fb, 0, HALF_W);
    let right = painted_in_columns(fb, HALF_W, W);
    assert!(
        left > (HALF_W * H) as usize * 9 / 10,
        "{}: the viewport's own half is only {left} of {} pixels",
        SCENES[5],
        HALF_W * H
    );
    assert_eq!(
        right, 0,
        "{}: {right} pixels painted outside the viewport - the signature of a viewport scale \
         read as a full extent rather than a half-extent",
        SCENES[5]
    );
}

/// **SCENE 6 - THE VIEWPORT AND THE REGION CLIP APPLY TOGETHER, AS AN INTERSECTION.**
///
/// The viewport is the left half and the clip keeps the top half, so only the TOP-LEFT QUARTER
/// may be painted. Scenes 1 and 5 each check one of those bounds on its own, and a renderer
/// that implements one and drops the other on the SAME draw passes both of them.
///
/// Four failures, each meaning something different, which is why all four quadrants are counted:
/// the whole target means neither bound applied, the left half means the clip was lost, the top
/// half means the viewport was lost, and nothing at all means the two rectangles were composed
/// into an empty one instead of intersected.
///
/// >>> IT IS ALSO THE REGRESSION FOR THE VIEWPORT BEING APPLIED AT ALL, AND FOR THE HALF OF
/// >>> THAT FIX A PICTURE CANNOT OTHERWISE SEE. Mapping clip space INTO the viewport rectangle
/// and BOUNDING the draw by it are two separate claims; on a quad that exactly fills its
/// viewport the first alone looks identical to both. Here it does not.
#[test]
fn a_viewport_and_a_region_clip_intersect() {
    let (sw, _) = run_and_render();
    let fb = &sw[6];
    let mut quadrant = [0usize; 4]; // top-left, top-right, bottom-left, bottom-right
    for y in 0..H {
        for x in 0..W {
            if painted(fb, x, y) {
                quadrant[usize::from(x >= HALF_W) + 2 * usize::from(y >= H / 2)] += 1;
            }
        }
    }
    let quarter = (HALF_W * H / 2) as usize;
    assert_eq!(
        quadrant,
        [quarter, 0, 0, 0],
        "{}: the painted quadrants are [top-left, top-right, bottom-left, bottom-right] = \
         {quadrant:?}, and a quarter is {quarter} pixels. All four full means neither bound \
         applied; the left half means the clip was lost; the top half means the viewport was; \
         nothing at all means the two were composed rather than intersected",
        SCENES[6]
    );
}

/// The app's scene-7 colours, as the framebuffer reads them (RGBA): the MARK quad's red, which
/// must never appear, and the FILL's blue.
const STENCIL_MARK: [u8; 4] = [255, 0, 0, 255];
const STENCIL_FILL: [u8; 4] = [0, 0, 255, 255];

/// Scene 7's picture on one backend, checked pixel by pixel against the exact rectangle: the
/// fill colour over the bottom-right quarter and the clear colour everywhere else. Returns the
/// failure message, naming WHICH stencil defect the picture looks like.
fn stencil_mask_failure(fb: &Framebuffer, backend: &str) -> Option<String> {
    // [inside, outside] x [fill, mark, clear, other]
    let mut n = [[0usize; 4]; 2];
    for y in 0..H {
        for x in 0..W {
            let outside = usize::from(!(x >= HALF_W && y >= H / 2));
            let p = fb.pixel(x, y);
            let k = if p == STENCIL_FILL {
                0
            } else if p == STENCIL_MARK {
                1
            } else if p == CLEAR {
                2
            } else {
                3
            };
            n[outside][k] += 1;
        }
    }
    let quarter = (HALF_W * H / 2) as usize;
    let want = [[quarter, 0, 0, 0], [0, 0, 3 * quarter, 0]];
    if n == want {
        return None;
    }
    let [inside, outside] = n;
    let looks_like = if outside[0] == 3 * quarter && inside[0] == quarter {
        "the WHOLE target is the fill colour: the stencil was ignored ENTIRELY - the EQUAL test \
         passed everywhere (the football HUD's white boxes)"
    } else if inside[1] > 0 {
        "the MARK quad's red painted: the NEVER test did not reject its colour - the stencil \
         TEST was ignored on the mark"
    } else if outside[0] > 0 {
        "the fill painted OUTSIDE the mark: the EQUAL test did not confine it"
    } else if inside[0] == 0 {
        "no fill at all: the mark's FAIL op never wrote the mask, or EQUAL compared against the \
         wrong value / mask"
    } else {
        "a partial or off-colour rectangle"
    };
    Some(format!(
        "{} [{backend}]: counts [fill, mark, clear, other] inside the bottom-right quarter \
         {inside:?} and outside it {outside:?}; want [{quarter}, 0, 0, 0] and [0, 0, {}, 0]. \
         It looks like {looks_like}. mid-left {:?} mid-right {:?}",
        SCENES[7],
        3 * quarter,
        fb.pixel(HALF_W / 2, H / 2 + H / 4),
        fb.pixel(HALF_W + HALF_W / 2, H / 2 + H / 4),
    ))
}

/// **SCENE 7 - A STENCIL MASK CONFINES A LATER DRAW TO WHAT AN INVISIBLE DRAW MARKED.**
///
/// >>> THIS IS A FOOTBALL TITLE'S HUD, AS A TEST.
///
/// It builds its scoreboard out of stencil masks: `NEVER` quads whose stencil FAIL op REPLACEs a
/// bit (so they write no colour at all), then `EQUAL ref=1 compare-mask=0x1` fills that may only
/// paint inside the cut-out. With no stencil every masked quad painted whole - white boxes over
/// the team names and the play-call panel.
///
/// The mark covers the bottom-right quarter in red (which must never appear), then a blue quad
/// covers the whole viewport through the EQUAL test. Only the bottom-right quarter may be blue,
/// and the rest must be the clear colour, exactly.
///
/// Asserted on BOTH backends, unlike scenes 0-6: the stencil is modelled in the GPU renderer, so
/// a software-only assertion would not be testing the renderer that ships. Run with
/// `VITASLOP_GXP_NO_STENCIL=1` the GPU side must fail here - that is the check that this test can
/// see the stencil at all.
#[test]
fn a_stencil_mask_confines_a_later_draw_to_the_marked_rectangle() {
    let (sw, hw) = run_and_render();
    let mut failures: Vec<String> = stencil_mask_failure(&sw[7], "software").into_iter().collect();
    match &hw {
        Some(hw) => failures.extend(stencil_mask_failure(&hw[7], "gpu")),
        None => eprintln!("no GPU adapter; scene 7 checked on the software oracle only"),
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The GPU renderer must agree with the software oracle on every scene.
///
/// Separate from the per-scene tests on purpose: those state what the PICTURE must be, and this
/// states that the two backends produce the same one. A divergence here is a backend defect even
/// when both pictures pass their own assertions, and lumping the two together would let a scene
/// that passes on software hide a GPU path that renders it differently.
///
/// >>> **IT ONCE REPORTED TWO CLIPPING GAPS HERE. BOTH WERE THIS TEST'S OWN, AND THE LESSON IS
/// >>> WORTH MORE THAN THE FINDINGS WERE.**
///
/// It read: scene 1 (region clip) diverging by 33.875 because "the fixed-function GPU path
/// paints the whole quad", and scene 2 (above the viewport) by 67.750 because "the GPU path
/// paints it" - the football crowd from the other side, a large wrong rectangle across a frame.
/// Every other scene was at 0.000, which is what made the pair look like a real pair.
///
/// **Neither exists.** All six scenes agree at a mean-abs-diff of 0.000. `run_and_render` used
/// to drive all six through ONE `GeneralRenderer`, and every scene renders into the same guest
/// colour surface - so all six shared the one kept display image that renderer holds per
/// address, deliberately, because a render target LOADS across frames rather than clearing.
/// The six single-scene frames therefore COMPOSITED: scene 1's correctly clipped-away right
/// half came back holding scene 0's green, and scene 2 - which correctly drew nothing at all -
/// came back holding every pixel scene 1 had left. See `run_and_render`.
///
/// **What should have caught it, and did once it was asked:** the renderer's own
/// `VITASLOP_GXP_VP_TRACE` prints the rect and scissor each draw actually got, and it was
/// printing `clip mode 0x80000000 [0,0,63,127] -> scissor (0, 0, 64, 128)` - the exact right
/// scissor - through the whole session that called the clip ignored. It printed nowhere because
/// a test binary installs no `tracing` subscriber and every `report!` is a `debug`. That is now
/// fixed for good in `show_renderer_diagnostics`, and the shape of each side is printed on
/// every scene by `shape` rather than a mean that can only say "they differ".
///
/// A mean-abs-diff between two backends says nothing about WHICH is wrong, and a picture cannot
/// tell "the renderer ignored the state" from "the harness never gave it a fresh target".
#[test]
fn the_gpu_renderer_agrees_with_the_software_oracle_on_every_scene() {
    let (sw, hw) = run_and_render();
    let Some(hw) = hw else {
        eprintln!("no GPU adapter; software-only run");
        return;
    };
    let mut worst = (0usize, 0.0f64);
    for (i, (s, h)) in sw.iter().zip(&hw).enumerate() {
        let sum: u64 = s
            .rgba
            .iter()
            .zip(&h.rgba)
            .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
            .sum();
        let mean = sum as f64 / s.rgba.len() as f64;
        eprintln!("scene {i} mean_abs_diff sw vs gpu: {mean:.3}  ({})", SCENES[i]);
        // The SHAPE of each side, always, not only on the failing scene: a mean cannot say
        // which backend is wrong or how, and that is how this pair got mis-attributed once.
        eprintln!("    sw  {}", shape(s));
        eprintln!("    gpu {}", shape(h));
        if mean > worst.1 {
            worst = (i, mean);
        }
    }
    assert!(
        worst.1 < 6.0,
        "scene {} diverges between the backends by {:.3}: {}",
        worst.0,
        worst.1,
        SCENES[worst.0]
    );
}
