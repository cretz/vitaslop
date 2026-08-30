//! A GAMMA-CORRECT render target must survive being sampled by a pass that also draws
//! into it, unchanged.
//!
//! # Why this test exists
//! `sceGxmColorSurfaceSetGammaMode` puts a surface in gamma-correct mode: the ROP
//! sRGB-ENCODES every value it stores, so the surface's memory holds encoded bytes and a
//! sampler must DECODE them on the way back in. The renderer reproduces that by rendering
//! and sampling through an sRGB view of the same texture, at both ends.
//!
//! Miss the decode at ONE end and a value reads back too bright: 0.5 stored as 0.73 reads
//! as 0.73. On an ordinary texture that is a one-off error. On a FEEDBACK path - a target a
//! pass samples while drawing into it, which is what an accumulation or blur buffer is -
//! the error re-applies every iteration and walks the image to white: 0.5 -> 0.73 -> 0.88
//! -> 0.95 -> 1. That is a "renders fine at first, then slowly turns white" defect, and the
//! shape is nasty: it needs several passes to show, so any single-pass probe passes.
//!
//! It has now been found TWICE in this renderer, on the two halves of the same mechanism -
//! the cross-frame sampling path (`sample_views`) and the within-frame snapshot
//! (`snapshot_rtt`, which exists ONLY for this case and so is the one that matters most).
//! Both halves are covered here, by construction rather than by inspection: the test asserts
//! the value comes back out of a three-pass chain as it went in.
//!
//! Content-free: every value is a constant in this file. Skips cleanly with no GPU adapter.

use vitaslop_native::GeneralRenderer;
use vitaslop_runtime::capture::{
    BoundTexture, ColorSurface, Draw, FragmentMaterial, Scene, VertexAttribute,
};

const W: u32 = 64;
const H: u32 = 64;
const CLEAR: [u8; 4] = [0, 0, 0, 255];
/// The guest address the offscreen target lives at. A draw naming this as its texture is
/// sampling the target, which is what puts it on the feedback path.
const RTT_ADDR: u32 = 0x0010_0000;
/// Mid grey. Chosen because it is where linear and sRGB disagree most: encoding 0.502
/// gives ~0.735, so a single missed decode is ~60 levels - far outside any rounding.
const MID: u8 = 128;
/// The display buffer's own guest address - a real frame always has one.
const DISPLAY_ADDR: u32 = 0x0020_0000;
/// An ordinary guest texture address, not a render target.
const SEED_ADDR: u32 = 0x0030_0000;

const F32: u8 = 9;
const U8N: u8 = 4;

/// Column-major identity 4x4. Every draw here is in PIXEL space, which ignores the world
/// matrix, but `Draw` carries one and a garbage matrix in a struct is a trap for the next
/// test that copies this file.
const IDENTITY_MVP: [f32; 16] =
    [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.];

fn pixel_vertex(buf: &mut Vec<u8>, x: f32, y: f32, u: f32, v: f32, color: [u8; 4]) {
    buf.extend_from_slice(&x.to_le_bytes());
    buf.extend_from_slice(&y.to_le_bytes());
    buf.extend_from_slice(&u.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
    buf.extend_from_slice(&color);
}

fn pixel_attrs() -> Vec<VertexAttribute> {
    vec![
        VertexAttribute { stream_index: 0, offset: 0, format: F32, component_count: 2, reg_index: 0 },
        VertexAttribute { stream_index: 0, offset: 8, format: F32, component_count: 2, reg_index: 1 },
        VertexAttribute { stream_index: 0, offset: 16, format: U8N, component_count: 4, reg_index: 2 },
    ]
}

/// A full-surface quad in pixel space, with `color` as its vertex colour and `tex` bound.
fn full_quad(color: [u8; 4], tex: Option<BoundTexture>) -> Draw {
    quad_inset(color, tex, 0.0)
}

/// The same quad inset by `inset` pixels on every side, so a test can tell the DRAW from
/// the pass's CLEAR - a full-surface quad makes the two indistinguishable.
fn quad_inset(color: [u8; 4], tex: Option<BoundTexture>, inset: f32) -> Draw {
    let (lo, hi) = (inset, W as f32 - inset);
    let mut v = Vec::new();
    pixel_vertex(&mut v, lo, lo, 0.0, 0.0, color);
    pixel_vertex(&mut v, hi, lo, 1.0, 0.0, color);
    pixel_vertex(&mut v, hi, hi, 1.0, 1.0, color);
    pixel_vertex(&mut v, lo, hi, 0.0, 1.0, color);
    let indices: Vec<u8> = [0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect();
    Draw {
        fragment_program_header: 0,
        vertex_textures: std::sync::Arc::from(&[][..]),
        primitive: 0,
        index_format: 0,
        index_count: 6,
        vertices: v.into(),
        vertex_stride: 20,
        attributes: pixel_attrs().into(),
        indices: indices.into(),
        uniforms: vec![],
        textures: tex.into_iter().collect(),
        render_state: Default::default(),
        exposure: 1.0,
        material: FragmentMaterial::default(),
        world: IDENTITY_MVP,
        blend: Default::default(),
        // Fixed-function parity path: no recompiled GXP payload.
        vprog: vitaslop_runtime::capture::no_program(),
        fprog: vitaslop_runtime::capture::no_program(),
        vert_sa: std::sync::Arc::from(&[][..]),
        frag_sa: std::sync::Arc::from(&[][..]),
        frag_sa_addr: 0,
        mem_windows: Vec::new(),
        shader_expanded: false,
    }
}

/// A texture that NAMES the offscreen target. Its `pixels` are never read - the renderer
/// recognises the address as a target this frame rendered and binds that render instead -
/// but they are filled with an obviously wrong colour so a test that somehow fell through to
/// the guest bytes fails loudly rather than passing on a coincidence.
fn texture_naming_target() -> BoundTexture {
    let size = 4u32;
    let mut pixels = Vec::new();
    for _ in 0..size * size {
        pixels.extend_from_slice(&[255, 0, 255, 255]);
    }
    BoundTexture {
        // A fixture: a DISTINCT buffer, so a distinct identity - two fixtures sharing
        // one id would collide in every cache keyed on it.
        pixels_id: vitaslop_runtime::capture::next_pixels_id(),
        unit: 0,
        base_format: 0x0c,
        swizzle: 0,
        tex_type: 3,
        width: size,
        height: size,
        stride: size * 4,
        faces: 1,
        face_bytes: size * size * 4,
        levels: 1,
        data_addr: RTT_ADDR,
        pixels: pixels.into(),
        u_addr_mode: 0,
        v_addr_mode: 0,
        lod_bias: 0,
        min_filter: 0,
        mag_filter: 0,
        mip_filter: 0,
        gamma: 0,
    }
}

/// The offscreen colour surface, in the guest's gamma-correct mode when `gamma`.
fn surface(gamma: u32) -> ColorSurface {
    ColorSurface {
        format: 0,
        surface_type: 0,
        width: W,
        height: H,
        stride_pixels: W,
        data_addr: RTT_ADDR,
        scale_mode: 0,
        gamma,
    }
}

/// Render a chain of `feedback` sample-and-write-back passes over the offscreen target and
/// return the centre pixel of the finished frame.
///
/// The chain is always:
///   1. draw flat `MID` grey into the offscreen target,
///   2. `feedback` further passes into the SAME target, each SAMPLING it - the within-frame
///      feedback shape, and what forces the snapshot copy,
///   3. blit the target to the display.
///
/// Every pass after the first is a straight pass-through (white vertex colour times the
/// sampled texel), so a correct chain returns exactly what pass 1 wrote, for any `feedback`
/// and whatever the surface's gamma mode.
///
/// # The display blit is part of every arm on purpose
/// The obvious way to ask "what did pass N leave in the target" is to truncate the chain
/// (`VITASLOP_CHAIN_LIMIT`). That is WRONG here and quietly so: the display buffer is
/// whatever the LAST scene draws to, so truncating promotes an offscreen pass to being the
/// display pass and changes the very code path under test. Growing the middle while keeping
/// a real blit last is the same measurement without that confound.
fn chain_centre(gpu: &mut GeneralRenderer, gamma: u32, feedback: usize) -> [u8; 4] {
    let white = [255, 255, 255, 255];
    // The seed pass paints the target from an ordinary guest TEXTURE rather than from a
    // vertex colour. Both should work, but only one of them is the thing under test, and a
    // seed that silently fails makes every later reading meaningless.
    let mut seed = texture_naming_target();
    seed.data_addr = SEED_ADDR;
    seed.pixels = std::iter::repeat([MID, MID, MID, 255]).take(16).flatten().collect::<Vec<u8>>().into();
    let mut scenes = vec![Scene {
        precompile: Default::default(),
        color: Some(surface(gamma)),
        depth: None,
        multisample: 0,
        draws: vec![quad_inset([255, 255, 255, 255], Some(seed), 8.0)],
    }];
    for _ in 0..feedback {
        scenes.push(Scene {
            precompile: Default::default(),
            color: Some(surface(gamma)),
            depth: None,
            multisample: 0,
            draws: vec![full_quad(white, Some(texture_naming_target()))],
        });
    }
    // The display scene carries a colour surface of its own, at a DIFFERENT address. That is
    // the shape every real frame has - the guest's display buffer is guest memory with an
    // address like any other - and `encode_chain` derives which pass is the display from
    // `scenes.last().target`, so a display scene with no surface is not a smaller version of
    // the real thing, it is a different case.
    let mut display_surface = surface(0);
    display_surface.data_addr = DISPLAY_ADDR;
    scenes.push(Scene {
        precompile: Default::default(),
color: Some(display_surface),
        depth: None,
        multisample: 0,
        draws: vec![quad_inset(white, Some(texture_naming_target()), 8.0)],
    });
    let fb = gpu.render_frame(&scenes, W, H, CLEAR);
    // The display CORNER is outside the blit, so it shows the display pass's own clear. If
    // that is not the clear colour, the display pass did not run and nothing else in this
    // frame can be read as a colour at all.
    eprintln!("  chain(gamma={gamma}, feedback={feedback}) display corner={:?} centre={:?}",
        fb.pixel(2, 2), fb.pixel(W / 2, H / 2));
    fb.pixel(W / 2, H / 2)
}

#[test]
fn a_gamma_target_survives_being_sampled_by_the_pass_that_draws_into_it() {
    let Some(mut gpu) = GeneralRenderer::new() else {
        eprintln!("no GPU adapter; skipping gamma feedback probe");
        return;
    };
    eprintln!("adapter: {}", gpu.adapter_name);

    // The LINEAR arm is the control. It shares every line of this chain except the one bit
    // under test, so if it also drifted the test would be measuring the chain, not the
    // gamma handling - which is exactly how a wrong tolerance gets blessed.
    //
    // Before any of it: the two primitives the chain is built from, each rendered straight
    // to the display in one pass. If either of these is wrong then nothing further down
    // measures the renderer, it measures the test.
    let direct_color = gpu.render_frame(
        &[Scene { precompile: Default::default(), color: None, depth: None, multisample: 0, draws: vec![quad_inset([MID, MID, MID, 255], None, 8.0)] }],
        W, H, CLEAR,
    );
    eprintln!(
        "sanity inset-quad to DISPLAY: corner={:?} (want the clear {CLEAR:?}) centre={:?} (want ~{MID})",
        direct_color.pixel(2, 2),
        direct_color.pixel(W / 2, H / 2)
    );
    let direct_tex = gpu.render_frame(
        &[Scene {
            precompile: Default::default(),
            color: None,
            depth: None,
            multisample: 0,
            draws: vec![full_quad([255, 255, 255, 255], Some(texture_naming_target()))],
        }],
        W, H, CLEAR,
    );
    eprintln!(
        "sanity: untextured vertex-colour grey -> {:?} (want ~{MID}), textured -> {:?} (want the \
         guest bytes 255,0,255)",
        direct_color.pixel(W / 2, H / 2),
        direct_tex.pixel(W / 2, H / 2)
    );

    // Zero feedback passes is the second control: it is the same chain with the mechanism
    // under test removed, so it says whether a plain render-and-blit already drifts.
    let mut results = Vec::new();
    for n in [3usize, 2, 1] {
        results.push((n, chain_centre(&mut gpu, 1, n), chain_centre(&mut gpu, 0, n)));
    }
    // n=0 LAST and linear last, so a `VITASLOP_GPU_CHAIN_DIR` dump (which keeps only the
    // most recent frame) belongs to the simplest arm rather than to whichever one happened
    // to run at the end. Reading a dump from the wrong arm already cost an hour here.
    let gamma0 = chain_centre(&mut gpu, 1, 0);
    let linear0 = chain_centre(&mut gpu, 0, 0);
    results.push((0, gamma0, linear0));
    results.sort_by_key(|r| r.0);
    for (n, g, l) in &results {
        eprintln!("feedback passes {n}: gamma={g:?} linear={l:?} (wrote {MID})");
    }
    let linear = linear0;
    let gamma = gamma0;

    for (name, px) in [("linear", linear), ("gamma", gamma)] {
        for (i, c) in px.iter().take(3).enumerate() {
            let drift = (*c as i32 - MID as i32).abs();
            assert!(
                drift <= 4,
                "[{name}] channel {i} came back {c}, wrote {MID} (drift {drift}). A value that \
                 grows through a sample-and-write-back chain is the sRGB round-trip defect: one \
                 end of the pair is missing its encode or its decode. Repeated, this is the \
                 white-out."
            );
        }
    }

    // And the two arms must agree with each other. A gamma surface stores different BYTES
    // than a linear one, but a chain that encodes and decodes at both ends returns the same
    // VALUE - that equality is the whole claim of gamma-correct mode.
    for i in 0..3 {
        let d = (linear[i] as i32 - gamma[i] as i32).abs();
        assert!(d <= 4, "gamma and linear arms disagree on channel {i}: {linear:?} vs {gamma:?}");
    }
}
