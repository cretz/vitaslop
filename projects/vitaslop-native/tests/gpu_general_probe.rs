//! Parity probe for the general GXM renderer ([`GeneralRenderer`] over the shared
//! `GxmRenderer`): render synthetic 2D scenes through both the GPU path and the
//! software rasterizer (`vitaslop_runtime::render::render_scene`, the oracle) and
//! assert they agree. The two use the same per-draw interpretation and texture
//! decode, so a divergence here is a real GPU-path bug (wrong space transform,
//! blend, or vertex layout) - caught in seconds, before the browser runs it live.
//!
//! Content-free: every scene is built from constants in this file, no game data.
//! Skips cleanly if no GPU adapter is present (headless CI), like the other GPU
//! probes.

use vitaslop_native::GeneralRenderer;
use vitaslop_runtime::capture::{BoundTexture, Draw, Scene, VertexAttribute};
use vitaslop_runtime::render::{render_scene, Framebuffer};

const W: u32 = 64;
const H: u32 = 64;
const CLEAR: [u8; 4] = [16, 16, 24, 255];

/// GXM attribute format enums (`SceGxmAttributeFormat`): F32 and normalized U8.
const F32: u8 = 9;
const U8N: u8 = 4;

/// A pixel-space vertex: pos.xy (f32 @0), uv.xy (f32 @8), color rgba (u8 @16); stride 20.
fn pixel_vertex(buf: &mut Vec<u8>, x: f32, y: f32, u: f32, v: f32, color: [u8; 4]) {
    buf.extend_from_slice(&x.to_le_bytes());
    buf.extend_from_slice(&y.to_le_bytes());
    buf.extend_from_slice(&u.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
    buf.extend_from_slice(&color);
}

/// The attribute list for [`pixel_vertex`]: position, texcoord, color.
fn pixel_attrs() -> Vec<VertexAttribute> {
    vec![
        VertexAttribute { stream_index: 0, offset: 0, format: F32, component_count: 2, reg_index: 0 },
        VertexAttribute { stream_index: 0, offset: 8, format: F32, component_count: 2, reg_index: 1 },
        VertexAttribute { stream_index: 0, offset: 16, format: U8N, component_count: 4, reg_index: 2 },
    ]
}

/// A two-triangle quad (4 verts, 6 indices) as a U16-indexed draw.
fn quad(vertices: Vec<u8>, attrs: Vec<VertexAttribute>, textures: Vec<BoundTexture>) -> Draw {
    let indices: Vec<u8> = [0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect();
    Draw {
        primitive: 0,
        index_format: 0,
        index_count: 6,
        vertices,
        vertex_stride: 20,
        attributes: attrs,
        indices,
        uniforms: vec![],
        textures,
    }
}

/// A solid-color LINEAR U8U8U8U8 (ABGR-identity swizzle) texture, `size` square. A
/// solid color keeps the comparison insensitive to nearest-sample edge placement, so
/// it tests the textured path + modulate, not texel-decode (covered by unit tests).
fn solid_texture(size: u32, rgba: [u8; 4]) -> BoundTexture {
    let mut pixels = Vec::new();
    for _ in 0..size * size {
        pixels.extend_from_slice(&rgba);
    }
    BoundTexture {
        unit: 0,
        base_format: 0x0c,
        swizzle: 0,
        tex_type: 3,
        width: size,
        height: size,
        stride: size * 4,
        data_addr: 0x1000,
        pixels,
    }
}

/// Mean absolute per-channel difference between two framebuffers of the same size.
fn mean_abs_diff(a: &Framebuffer, b: &Framebuffer) -> f64 {
    assert_eq!(a.rgba.len(), b.rgba.len());
    let sum: u64 = a
        .rgba
        .iter()
        .zip(&b.rgba)
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.rgba.len() as f64
}

/// Count of pixels differing from the clear color (how much got drawn).
fn drawn(fb: &Framebuffer) -> usize {
    fb.drawn_pixels(CLEAR)
}

/// Render `scene` both ways and assert the GPU output matches the software oracle:
/// low mean difference and a comparable drawn-pixel count.
fn assert_parity(gpu: &mut GeneralRenderer, scene: &Scene, name: &str) {
    let sw = render_scene(scene, W, H, CLEAR);
    let hw = gpu.render_scene(scene, W, H, CLEAR);
    let diff = mean_abs_diff(&sw, &hw);
    let (ds, dh) = (drawn(&sw), drawn(&hw));
    eprintln!("[{name}] mean_abs_diff={diff:.3} drawn sw={ds} hw={dh}");
    // A few units of mean diff is expected (GPU unorm rounding + rasterizer fill-rule
    // at edges); a real divergence (wrong transform / blend / layout) is tens+.
    assert!(diff < 6.0, "[{name}] GPU diverges from software oracle: mean_abs_diff={diff:.3}");
    // Both must draw a comparable amount (not one blank).
    assert!(ds > 0 && dh > 0, "[{name}] one path drew nothing: sw={ds} hw={dh}");
    let ratio = ds.min(dh) as f64 / ds.max(dh) as f64;
    assert!(ratio > 0.8, "[{name}] drawn-pixel counts disagree: sw={ds} hw={dh}");
}

#[test]
fn general_renderer_matches_software_oracle() {
    let Some(mut gpu) = GeneralRenderer::new() else {
        eprintln!("no GPU adapter; skipping general renderer parity probe");
        return;
    };
    eprintln!("adapter: {}", gpu.adapter_name);

    // 1. Textured pixel-space sprite: a solid-red 4x4 texture on a centered quad,
    //    white vertex color (so color * texel = texel). Exercises the Pixel space
    //    transform, the textured path, and the atlas UV normalization staying at 1.
    {
        let mut v = Vec::new();
        let white = [255, 255, 255, 255];
        pixel_vertex(&mut v, 16.0, 16.0, 0.0, 0.0, white);
        pixel_vertex(&mut v, 48.0, 16.0, 1.0, 0.0, white);
        pixel_vertex(&mut v, 48.0, 48.0, 1.0, 1.0, white);
        pixel_vertex(&mut v, 16.0, 48.0, 0.0, 1.0, white);
        let scene = Scene {
            color: None,
            draws: vec![quad(v, pixel_attrs(), vec![solid_texture(4, [220, 40, 40, 255])])],
        };
        assert_parity(&mut gpu, &scene, "pixel-textured");
    }

    // 2. Untextured vertex-color quad in NDC space: no texcoord attribute, so the
    //    draw is untextured (color * white-fallback = color) and its positions are
    //    clip coords. Tests color interpolation and the Ndc transform.
    {
        let mut nv = Vec::new();
        for (x, y, c) in [
            (-0.8f32, -0.8f32, [255, 0, 0, 255]),
            (0.8, -0.8, [0, 255, 0, 255]),
            (0.8, 0.8, [0, 0, 255, 255]),
            (-0.8, 0.8, [255, 255, 0, 255]),
        ] {
            nv.extend_from_slice(&x.to_le_bytes());
            nv.extend_from_slice(&y.to_le_bytes());
            nv.extend_from_slice(&c);
        }
        let attrs = vec![
            VertexAttribute { stream_index: 0, offset: 0, format: F32, component_count: 2, reg_index: 0 },
            VertexAttribute { stream_index: 0, offset: 8, format: U8N, component_count: 4, reg_index: 1 },
        ];
        let indices: Vec<u8> = [0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect();
        let draw = Draw {
            primitive: 0,
            index_format: 0,
            index_count: 6,
            vertices: nv,
            vertex_stride: 12,
            attributes: attrs,
            indices,
            uniforms: vec![],
            textures: vec![],
        };
        assert_parity(&mut gpu, &Scene { color: None, draws: vec![draw] }, "ndc-vertexcolor");
    }

    // 3. Alpha blend in submission order: an opaque red quad, then a half-alpha blue
    //    quad over it. Tests the 2D straight-alpha src-over blend and draw ordering.
    {
        let mut back = Vec::new();
        let red = [220, 20, 20, 255];
        pixel_vertex(&mut back, 8.0, 8.0, 0.0, 0.0, red);
        pixel_vertex(&mut back, 56.0, 8.0, 0.0, 0.0, red);
        pixel_vertex(&mut back, 56.0, 56.0, 0.0, 0.0, red);
        pixel_vertex(&mut back, 8.0, 56.0, 0.0, 0.0, red);
        let mut front = Vec::new();
        let blue_half = [40, 40, 220, 128];
        pixel_vertex(&mut front, 24.0, 24.0, 0.0, 0.0, blue_half);
        pixel_vertex(&mut front, 56.0, 24.0, 0.0, 0.0, blue_half);
        pixel_vertex(&mut front, 56.0, 56.0, 0.0, 0.0, blue_half);
        pixel_vertex(&mut front, 24.0, 56.0, 0.0, 0.0, blue_half);
        let scene = Scene {
            color: None,
            draws: vec![
                quad(back, pixel_attrs(), vec![]),
                quad(front, pixel_attrs(), vec![]),
            ],
        };
        assert_parity(&mut gpu, &scene, "alpha-blend");
    }
}
