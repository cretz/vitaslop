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
use vitaslop_runtime::capture::{BoundTexture, Draw, FragmentMaterial, RenderState, Scene, VertexAttribute};
use vitaslop_runtime::render::{render_scene, render_scene_supersampled, Framebuffer};

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
        vertex_textures: Vec::new(),
        primitive: 0,
        index_format: 0,
        index_count: 6,
        vertices,
        vertex_stride: 20,
        attributes: attrs,
        indices,
        uniforms: vec![],
        textures,
        render_state: Default::default(),
        exposure: 1.0,
        material: FragmentMaterial::default(),
        world: IDENTITY_MVP,
        // The guest's blend equation. These probes are fixed-function parity checks, so they
        // take the default a NULL `blendInfo` gives: write every channel, no blending.
        blend: Default::default(),
        // The GXP recompiler payload: empty off that path, which is what these fixed-function
        // probes exercise.
        vprog: Vec::new(),
        fprog: Vec::new(),
        vert_sa: Vec::new(),
        frag_sa: Vec::new(),
        // These probes hand the renderer real triangles, not point-sprite records a
        // vertex program would expand into quads.
        shader_expanded: false,
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
        faces: 1,
        face_bytes: size * size * 4,
        data_addr: 0x1000,
        pixels: pixels.into(),
        u_addr_mode: 0,
        v_addr_mode: 0,
        lod_bias: 0,
        min_filter: 0,
        mag_filter: 0,
        gamma: 0,
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

/// Render `scene` both ways and assert the GPU output matches the software oracle
/// within `tol` mean-abs-diff and a comparable drawn-pixel count. Returns the software
/// framebuffer so a caller can add absolute (semantic) assertions on the oracle.
fn assert_parity_tol(gpu: &mut GeneralRenderer, scene: &Scene, name: &str, tol: f64) -> Framebuffer {
    let sw = render_scene(scene, W, H, CLEAR);
    let hw = gpu.render_scene(scene, W, H, CLEAR);
    let diff = mean_abs_diff(&sw, &hw);
    let (ds, dh) = (drawn(&sw), drawn(&hw));
    eprintln!("[{name}] mean_abs_diff={diff:.3} drawn sw={ds} hw={dh}");
    // A few units of mean diff is expected (GPU unorm rounding + rasterizer fill-rule
    // at edges, and hardware vs software bilinear); a real divergence (wrong transform /
    // blend / combine / layout) is tens+.
    assert!(diff < tol, "[{name}] GPU diverges from software oracle: mean_abs_diff={diff:.3} (tol {tol})");
    // Both must draw a comparable amount (not one blank).
    assert!(ds > 0 && dh > 0, "[{name}] one path drew nothing: sw={ds} hw={dh}");
    let ratio = ds.min(dh) as f64 / ds.max(dh) as f64;
    assert!(ratio > 0.8, "[{name}] drawn-pixel counts disagree: sw={ds} hw={dh}");
    sw
}

/// Default parity tolerance (see [`assert_parity_tol`]).
fn assert_parity(gpu: &mut GeneralRenderer, scene: &Scene, name: &str) -> Framebuffer {
    assert_parity_tol(gpu, scene, name, 6.0)
}

/// The RGBA at the framebuffer center - the interior of a centered quad, away from the
/// antialiased/fill-rule edges, for absolute (semantic) assertions.
fn center(fb: &Framebuffer) -> [u8; 4] {
    fb.pixel(W / 2, H / 2)
}

/// An MVP-space vertex: pos.xyz (f32 @0), uv.xy (f32 @12), color rgba (u8 @20); stride 24.
fn mvp_vertex(buf: &mut Vec<u8>, x: f32, y: f32, z: f32, u: f32, v: f32, color: [u8; 4]) {
    buf.extend_from_slice(&x.to_le_bytes());
    buf.extend_from_slice(&y.to_le_bytes());
    buf.extend_from_slice(&z.to_le_bytes());
    buf.extend_from_slice(&u.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
    buf.extend_from_slice(&color);
}

/// The attribute list for [`mvp_vertex`]: position float3, texcoord float2, color.
fn mvp_attrs() -> Vec<VertexAttribute> {
    vec![
        VertexAttribute { stream_index: 0, offset: 0, format: F32, component_count: 3, reg_index: 0 },
        VertexAttribute { stream_index: 0, offset: 12, format: F32, component_count: 2, reg_index: 1 },
        VertexAttribute { stream_index: 0, offset: 20, format: U8N, component_count: 4, reg_index: 2 },
    ]
}

/// Column-major identity 4x4: an MVP that maps object coords in [-1,1] straight to NDC
/// (w = 1), so a `[-0.8, 0.8]` quad lands centered on screen.
const IDENTITY_MVP: [f32; 16] =
    [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.];

/// A material that isolates the exposure/tonemap path from lighting: unit tint, no
/// directional light, and unit ambient - so `fs_opaque` reduces to `albedo * exposure`
/// then Reinhard, exactly the pre-lighting opaque model. Lets a test pin the exposure
/// curve without depending on a normal or light direction.
fn flat_material() -> FragmentMaterial {
    FragmentMaterial {
        tint: [1.0, 1.0, 1.0],
        light_dir: [0.0, 1.0, 0.0],
        light_col: [0.0, 0.0, 0.0],
        ambient: [1.0, 1.0, 1.0],
        has_light: false,
    }
}

/// A centered MVP quad (object space `[-0.8, 0.8]`, z = 0, uv 0..1, identity MVP). With
/// `depth_write_disabled` it is a 2D alpha overlay instead of opaque 3D - the runtime
/// keys "opaque" off the depth-write state, not merely MVP space.
fn mvp_quad(
    color: [u8; 4],
    textures: Vec<BoundTexture>,
    exposure: f32,
    depth_write_disabled: bool,
    material: FragmentMaterial,
) -> Draw {
    let mut v = Vec::new();
    for (x, y, u, vv) in
        [(-0.8f32, -0.8f32, 0., 0.), (0.8, -0.8, 1., 0.), (0.8, 0.8, 1., 1.), (-0.8, 0.8, 0., 1.)]
    {
        mvp_vertex(&mut v, x, y, 0.0, u, vv, color);
    }
    let indices: Vec<u8> = [0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect();
    let mut render_state = RenderState::default();
    if depth_write_disabled {
        render_state.front_depth_write = 0x0010_0000; // SCE_GXM_DEPTH_WRITE_DISABLED
    }
    Draw {
        vertex_textures: Vec::new(),
        primitive: 0,
        index_format: 0,
        index_count: 6,
        vertices: v,
        vertex_stride: 24,
        attributes: mvp_attrs(),
        indices,
        uniforms: IDENTITY_MVP.to_vec(),
        textures,
        render_state,
        exposure,
        material,
        world: IDENTITY_MVP,
        // The guest's blend equation. These probes are fixed-function parity checks, so they
        // take the default a NULL `blendInfo` gives: write every channel, no blending.
        blend: Default::default(),
        // The GXP recompiler payload: empty off that path, which is what these fixed-function
        // probes exercise.
        vprog: Vec::new(),
        fprog: Vec::new(),
        vert_sa: Vec::new(),
        frag_sa: Vec::new(),
        // These probes hand the renderer real triangles, not point-sprite records a
        // vertex program would expand into quads.
        shader_expanded: false,
    }
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
            depth: None,
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
            vertex_textures: Vec::new(),
        textures: vec![],
            render_state: Default::default(),
            exposure: 1.0,
            material: FragmentMaterial::default(),
            world: IDENTITY_MVP,
        // The guest's blend equation. These probes are fixed-function parity checks, so they
        // take the default a NULL `blendInfo` gives: write every channel, no blending.
        blend: Default::default(),
        // The GXP recompiler payload: empty off that path, which is what these fixed-function
        // probes exercise.
        vprog: Vec::new(),
        fprog: Vec::new(),
        vert_sa: Vec::new(),
        frag_sa: Vec::new(),
        // Real triangles, not point-sprite records the vertex program expands.
        shader_expanded: false,
        };
        assert_parity(&mut gpu, &Scene { color: None, depth: None, draws: vec![draw] }, "ndc-vertexcolor");
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
            depth: None,
            draws: vec![
                quad(back, pixel_attrs(), vec![]),
                quad(front, pixel_attrs(), vec![]),
            ],
        };
        assert_parity(&mut gpu, &scene, "alpha-blend");
    }

    // 4. Opaque 3D draw (MVP space + depth write) with a dark albedo texture, scene
    //    exposure, and a RED vertex color. The opaque combine takes the albedo texel
    //    STRAIGHT (ignoring the vertex color, which for real world meshes is a non-color
    //    mask), then applies exposure + a Reinhard tonemap. A regression to `color*texel`
    //    would tint the surface red - caught as a parity break here, and the absolute
    //    check pins the semantic: grey, not red.
    {
        // texel 64/255 = 0.251; with the flat material (unit ambient, no directional light)
        // the lit value reduces to albedo*exposure: l = 0.251*4 = 1.004; reinhard = l/(1+l)
        // = 0.501 -> ~128. This pins the exposure/tonemap curve independent of lighting.
        let dark = solid_texture(4, [64, 64, 64, 255]);
        let draw = mvp_quad([255, 0, 0, 255], vec![dark], 4.0, false, flat_material());
        let scene = Scene { color: None, depth: None, draws: vec![draw] };
        let sw = assert_parity(&mut gpu, &scene, "mvp-opaque-exposure");
        let c = center(&sw);
        assert!(c[0] > 100 && c[0] < 155, "exposed dark albedo should be ~mid-grey, got {c:?}");
        assert!(
            (c[0] as i32 - c[1] as i32).abs() < 12 && (c[0] as i32 - c[2] as i32).abs() < 12,
            "opaque combine must use the albedo texel, not the red vertex color: {c:?}"
        );
    }

    // 4b. LIT MATERIAL: a WHITE albedo texture under a near-black base-colour tint (a tyre:
    //     white detail albedo, tint ~0.05) lit by a directional light aligned with the quad's
    //     up-facing normal. The lit result must be DARK - this is exactly the wheel-ring fix
    //     (an unlit renderer painted the white albedo as a white ring; the reflected tint
    //     scales it to dark rubber). Asserts parity AND that tint*light actually darkens.
    {
        let white = solid_texture(4, [255, 255, 255, 255]);
        let mat = FragmentMaterial {
            tint: [0.05, 0.05, 0.05],
            light_dir: [0.0, 1.0, 0.0], // aligned with the default up normal -> N.L = 1
            light_col: [1.0, 1.0, 1.0],
            ambient: [0.0, 0.0, 0.0],
            has_light: true,
        };
        let draw = mvp_quad([255, 0, 255, 255], vec![white], 1.0, false, mat);
        let scene = Scene { color: None, depth: None, draws: vec![draw] };
        let sw = assert_parity(&mut gpu, &scene, "mvp-lit-tint");
        let c = center(&sw);
        // albedo 1.0 * tint 0.05 * (ambient 0 + light 1 * N.L 1) = 0.05; reinhard -> ~12.
        assert!(c[0] < 40 && c[1] < 40 && c[2] < 40, "dark tint must darken a white albedo, got {c:?}");
    }

    // 5. THE DECOUPLE CASE: an MVP-space draw with depth writes DISABLED is a 2D
    //    alpha-blended overlay, NOT opaque 3D - even though it uses the MVP transform.
    //    The runtime must key "opaque" off the depth-write state (so this modulates
    //    vertex_color * texel and src-over blends), not off MVP space. Half-alpha white
    //    vertex color over a blue texel: the result blends toward the clear background.
    {
        let blue = solid_texture(4, [40, 40, 220, 255]);
        let draw = mvp_quad([255, 255, 255, 180], vec![blue], 1.0, true, FragmentMaterial::default());
        let scene = Scene { color: None, depth: None, draws: vec![draw] };
        let sw = assert_parity(&mut gpu, &scene, "mvp-depthdisabled-overlay");
        let c = center(&sw);
        // Blended, not opaque-replaced: alpha 180/255 over the dark clear keeps the blue
        // partial. If it were wrongly treated as opaque it would be full-strength blue.
        assert!(c[2] > c[0] && c[2] < 220, "expected partial (blended) blue, got {c:?}");
    }

    // 6. LINEAR magnification parity: a 2x2 gradient texture drawn large so it is heavily
    //    magnified. The software path bilinear-samples (`sample_texture_bilinear`); the
    //    GPU must pick the Linear sampler and produce the same smooth gradient (a small
    //    extra tolerance covers hardware vs software bilinear rounding).
    {
        // 2x2 RGBA: distinct corners so bilinear produces a real gradient across the quad.
        let px = vec![
            20, 20, 20, 255, /* (1,0) */ 220, 40, 40, 255, /* (0,1) */ 40, 220, 40, 255,
            /* (1,1) */ 40, 40, 220, 255,
        ];
        let mut tex = BoundTexture {
            unit: 0,
            base_format: 0x0c,
            swizzle: 0,
            tex_type: 3,
            width: 2,
            height: 2,
            stride: 8,
            faces: 1,
            face_bytes: 16,
            data_addr: 0x2000,
            pixels: px.into(),
            u_addr_mode: 0,
            v_addr_mode: 0,
            lod_bias: 0,
            min_filter: 1,
            mag_filter: 1, // SCE_GXM_TEXTURE_FILTER_LINEAR
            gamma: 0,
        };
        // Nudge the address so its content key differs from any earlier texture.
        tex.data_addr = 0x2100;
        let mut v = Vec::new();
        let white = [255, 255, 255, 255];
        pixel_vertex(&mut v, 8.0, 8.0, 0.0, 0.0, white);
        pixel_vertex(&mut v, 56.0, 8.0, 1.0, 0.0, white);
        pixel_vertex(&mut v, 56.0, 56.0, 1.0, 1.0, white);
        pixel_vertex(&mut v, 8.0, 56.0, 0.0, 1.0, white);
        let scene = Scene { color: None, depth: None, draws: vec![quad(v, pixel_attrs(), vec![tex])] };
        assert_parity_tol(&mut gpu, &scene, "linear-filter", 9.0);
    }

    // 7. Untextured opaque 3D draw: MVP space + depth write, a per-vertex color, NO
    //    texcoord attribute (so no texture). The opaque combine falls back to the vertex
    //    color (not the white texel) and still applies exposure + Reinhard.
    {
        // pos.xyz f32 @0, color u8x4 @12; stride 16, no uv.
        let mut v = Vec::new();
        for (x, y) in [(-0.8f32, -0.8f32), (0.8, -0.8), (0.8, 0.8), (-0.8, 0.8)] {
            v.extend_from_slice(&x.to_le_bytes());
            v.extend_from_slice(&y.to_le_bytes());
            v.extend_from_slice(&0.0f32.to_le_bytes());
            v.extend_from_slice(&[200, 100, 50, 255]);
        }
        let attrs = vec![
            VertexAttribute { stream_index: 0, offset: 0, format: F32, component_count: 3, reg_index: 0 },
            VertexAttribute { stream_index: 0, offset: 12, format: U8N, component_count: 4, reg_index: 1 },
        ];
        let indices: Vec<u8> = [0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect();
        let draw = Draw {
            primitive: 0,
            index_format: 0,
            index_count: 6,
            vertices: v,
            vertex_stride: 16,
            attributes: attrs,
            indices,
            uniforms: IDENTITY_MVP.to_vec(),
            vertex_textures: Vec::new(),
        textures: vec![],
            render_state: RenderState::default(),
            exposure: 2.0,
            // Flat material (unit ambient, no directional light) isolates the exposure ramp.
            material: flat_material(),
            world: IDENTITY_MVP,
        // The guest's blend equation. These probes are fixed-function parity checks, so they
        // take the default a NULL `blendInfo` gives: write every channel, no blending.
        blend: Default::default(),
        // The GXP recompiler payload: empty off that path, which is what these fixed-function
        // probes exercise.
        vprog: Vec::new(),
        fprog: Vec::new(),
        vert_sa: Vec::new(),
        frag_sa: Vec::new(),
        // Real triangles, not point-sprite records the vertex program expands.
        shader_expanded: false,
        };
        let scene = Scene { color: None, depth: None, draws: vec![draw] };
        let sw = assert_parity(&mut gpu, &scene, "mvp-opaque-untextured");
        let c = center(&sw);
        // R = reinhard(200/255*2) = 1.568/2.568 = 0.611 -> ~156; must be reddish (R>G>B).
        assert!(c[0] > c[1] && c[1] > c[2], "vertex-color opaque should keep R>G>B ramp: {c:?}");
        assert!(c[0] > 130 && c[0] < 180, "exposed vertex red should be ~156, got {c:?}");
    }

    // 8. Back-face culling (SceGxmCullMode). An opaque MVP triangle with cull=CCW draws
    //    when wound one way and is fully culled when wound the other - identically on both
    //    paths (the CPU builder pre-culls into the index buffer; the software oracle culls
    //    at raster). Exactly one of the two windings survives, and the paths agree which.
    {
        let cull_tri = |order: [u16; 3]| -> Scene {
            let mut v = Vec::new();
            for (x, y) in [(-0.6f32, -0.6f32), (0.6, -0.6), (0.0, 0.6)] {
                mvp_vertex(&mut v, x, y, 0.0, 0.0, 0.0, [200, 200, 60, 255]);
            }
            let indices: Vec<u8> = order.iter().flat_map(|i| i.to_le_bytes()).collect();
            let mut render_state = RenderState::default();
            render_state.cull_mode = 0x2; // SCE_GXM_CULL_CCW
            let draw = Draw {
                primitive: 0,
                index_format: 0,
                index_count: 3,
                vertices: v,
                vertex_stride: 24,
                attributes: mvp_attrs(),
                indices,
                uniforms: IDENTITY_MVP.to_vec(),
                vertex_textures: Vec::new(),
        textures: vec![],
                render_state,
                exposure: 1.0,
                material: FragmentMaterial::default(),
                world: IDENTITY_MVP,
        // The guest's blend equation. These probes are fixed-function parity checks, so they
        // take the default a NULL `blendInfo` gives: write every channel, no blending.
        blend: Default::default(),
        // The GXP recompiler payload: empty off that path, which is what these fixed-function
        // probes exercise.
        vprog: Vec::new(),
        fprog: Vec::new(),
        vert_sa: Vec::new(),
        frag_sa: Vec::new(),
        // Real triangles, not point-sprite records the vertex program expands.
        shader_expanded: false,
            };
            Scene { color: None, depth: None, draws: vec![draw] }
        };
        let a = cull_tri([0, 1, 2]);
        let b = cull_tri([0, 2, 1]);
        let (a_sw, a_hw) = (render_scene(&a, W, H, CLEAR), gpu.render_scene(&a, W, H, CLEAR));
        let (b_sw, b_hw) = (render_scene(&b, W, H, CLEAR), gpu.render_scene(&b, W, H, CLEAR));
        eprintln!(
            "[cull] order012 sw={} hw={}  order021 sw={} hw={}",
            drawn(&a_sw), drawn(&a_hw), drawn(&b_sw), drawn(&b_hw)
        );
        // Both paths agree per winding on whether anything survived.
        assert_eq!(drawn(&a_sw) > 0, drawn(&a_hw) > 0, "cull parity, order 012");
        assert_eq!(drawn(&b_sw) > 0, drawn(&b_hw) > 0, "cull parity, order 021");
        // Exactly one winding is the front face (drawn); the other is the back (culled).
        assert!(
            (drawn(&a_sw) > 0) ^ (drawn(&b_sw) > 0),
            "exactly one winding survives CCW cull: {} vs {}",
            drawn(&a_sw), drawn(&b_sw)
        );
        // The surviving winding matches pixel-wise between the two paths.
        let (sw, hw) = if drawn(&a_sw) > 0 { (&a_sw, &a_hw) } else { (&b_sw, &b_hw) };
        assert!(mean_abs_diff(sw, hw) < 6.0, "surviving face must match across paths");
    }

    // 9. Depth-compare function. LESS_EQUAL (GXM's default and this title's opaque 3D
    //    draws) lets a coincident, later-submitted opaque face tie the depth test and
    //    repaint. Two coincident opaque MVP quads at the same depth (red then blue): blue,
    //    drawn second, wins on both paths. A regression to strict LESS would leave red.
    {
        let red = mvp_quad([220, 20, 20, 255], vec![], 1.0, false, flat_material());
        let blue = mvp_quad([20, 20, 220, 255], vec![], 1.0, false, flat_material());
        let scene = Scene { color: None, depth: None, draws: vec![red, blue] };
        let sw = assert_parity(&mut gpu, &scene, "depthfunc-lessequal");
        let c = center(&sw);
        // The winning face is blue: its dominant blue channel survives (Reinhard-compressed
        // from the untextured vertex-colour albedo), while the red channel stays low.
        assert!(
            c[2] > 90 && c[0] < 50,
            "LESS_EQUAL must let the later coincident face (blue) win, got {c:?}"
        );
    }
}

/// Supersampling parity: the GPU's `set_supersample(N)` (render at N x into an offscreen
/// target, box-downsample) must match the software oracle's `render_scene_supersampled`
/// (render at N x, `Framebuffer::downsampled`) - so the antialiased shipped path stays a
/// faithful twin of the oracle. Uses a mix of an opaque MVP quad and a textured Pixel-space
/// sprite so both the resolution-independent (Mvp) and the Pixel-scaled coordinate paths are
/// exercised through the resolve, then compares at factor 2.
#[test]
fn general_renderer_supersample_matches_software() {
    let Some(mut gpu) = GeneralRenderer::new() else {
        eprintln!("no GPU adapter; skipping supersample parity probe");
        return;
    };
    eprintln!("adapter: {}", gpu.adapter_name);

    // A textured Pixel-space sprite (a 2x2 checker so the resolve averages real texel edges)
    // over a solid opaque MVP quad, so the frame has both interior fills and antialiased edges.
    let checker: Vec<u8> = {
        let px = [[200u8, 60, 60, 255], [40, 40, 200, 255]];
        let mut b = Vec::new();
        for y in 0..2 {
            for x in 0..2 {
                b.extend_from_slice(&px[(x + y) % 2]);
            }
        }
        b
    };
    let tex = BoundTexture {
        unit: 0, base_format: 0x0c, swizzle: 0, tex_type: 0, width: 2, height: 2, stride: 8,
        faces: 1, face_bytes: 16,
        pixels: checker.into(), data_addr: 0, u_addr_mode: 0, v_addr_mode: 0, lod_bias: 0,
        min_filter: 0, mag_filter: 0, gamma: 0,
    };
    let mut sprite_v = Vec::new();
    for (x, y, u, v) in [(8.0f32, 8.0, 0.0, 0.0), (56.0, 8.0, 1.0, 0.0), (56.0, 56.0, 1.0, 1.0), (8.0, 56.0, 0.0, 1.0)] {
        pixel_vertex(&mut sprite_v, x, y, u, v, [255, 255, 255, 255]);
    }
    let sprite = quad(sprite_v, pixel_attrs(), vec![tex]);
    let backdrop = mvp_quad([60, 160, 60, 255], vec![], 1.0, false, flat_material());
    let scene = Scene { color: None, depth: None, draws: vec![backdrop, sprite] };

    // Factor-1 must still be exactly the non-supersampled path (a sanity anchor).
    gpu.set_supersample(1);
    let base_diff = mean_abs_diff(&render_scene(&scene, W, H, CLEAR), &gpu.render_scene(&scene, W, H, CLEAR));
    assert!(base_diff < 6.0, "factor-1 must equal the plain path: {base_diff:.3}");

    // Factor-2 supersampled: software resolve vs GPU resolve.
    gpu.set_supersample(2);
    let sw = render_scene_supersampled(&scene, W, H, CLEAR, 2);
    let hw = gpu.render_scene(&scene, W, H, CLEAR);
    let diff = mean_abs_diff(&sw, &hw);
    eprintln!("[supersample-2] mean_abs_diff={diff:.3}");
    // Both box-average the same N x N rasterization; a few units of unorm-rounding /
    // edge-fill difference is expected, a real resolve bug is tens+.
    assert!(diff < 8.0, "supersampled GPU diverges from oracle: mean_abs_diff={diff:.3}");
    // The supersampled frame must differ from the aliased 1x frame (AA actually happened),
    // and still draw a full frame.
    let one = render_scene(&scene, W, H, CLEAR);
    assert!(mean_abs_diff(&sw, &one) > 0.0, "supersample must change the frame");
    assert!(drawn(&hw) > 0 && drawn(&sw) > 0, "supersampled frame drew nothing");
}
