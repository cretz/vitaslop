//! The GXM command-stream capture: the blob-free "it works" signal. The host
//! records what the guest asked the GPU to do (surfaces, programs, per-draw
//! vertex/index/uniform snapshots) without emulating a GPU or drawing a pixel.
//! A software rasterizer or wgpu backend later consumes this to produce frames.

use std::sync::Arc;

/// One vertex attribute as declared by the guest's vertex program: which stream,
/// byte offset within a vertex, source format, and component count. `format` and
/// the counts are the raw GXM enum values so the consumer decodes exactly what
/// the guest laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexAttribute {
    pub stream_index: u16,
    pub offset: u16,
    pub format: u8,
    pub component_count: u8,
    pub reg_index: u16,
}

/// A color surface the guest initialized: pixel format plus geometry and the
/// guest address of its pixel buffer (the target a scene renders into).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorSurface {
    pub format: u32,
    /// The `SceGxmColorSurfaceType` (LINEAR/TILED/SWIZZLED) the guest set at init,
    /// read back by `sceGxmColorSurfaceGetType`. Not consumed by the renderer (it
    /// resolves the pixel layout from the data pointer and stride), but recorded so
    /// the getter round-trips exactly what the guest set.
    pub surface_type: u32,
    pub width: u32,
    pub height: u32,
    pub stride_pixels: u32,
    pub data_addr: u32,
    /// The `SceGxmColorSurfaceScaleMode` the guest set at init (NONE, or the
    /// downscale MSAA resolves use), read back by `sceGxmColorSurfaceGetScaleMode`.
    /// Recorded so the getter round-trips; the renderer resolves at full resolution.
    pub scale_mode: u32,
    /// The `SceGxmColorSurfaceGammaMode` set by `sceGxmColorSurfaceSetGammaMode`. Non-zero
    /// means the ROP sRGB-ENCODES every write to this surface, so its memory holds
    /// gamma-encoded bytes and a shader writing linear values still lands correct on screen.
    /// A renderer that ignores this stores the linear values instead and the whole surface -
    /// and everything that samples it - comes out far too dark.
    pub gamma: u32,
}

/// The fixed-function pipeline state a title sets on the GXM context between draws
/// with the `sceGxmSet*` family (cull, depth, stencil, viewport, region clip,
/// polygon mode). GXM context state is sticky - a setter mutates the current state
/// and every later draw inherits it until it is changed again - so the host tracks
/// the live values and snapshots them into each [`Draw`], exactly as uniforms and
/// bound textures are captured. A renderer reproduces a draw from this snapshot
/// without replaying the call stream. Every field is the raw GXM enum word (e.g.
/// `cull_mode` is a `SceGxmCullMode` value, `front_depth_func` a `SceGxmDepthFunc`
/// value); [`Default`] is GXM's documented context default for a field a title
/// leaves unset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderState {
    pub cull_mode: u32,
    pub two_sided: u32,
    pub front_depth_func: u32,
    pub back_depth_func: u32,
    pub front_depth_write: u32,
    pub back_depth_write: u32,
    pub front_fragment_program_enable: u32,
    pub back_fragment_program_enable: u32,
    pub front_polygon_mode: u32,
    pub back_polygon_mode: u32,
    pub front_point_line_width: u32,
    pub front_stencil_ref: u32,
    pub front_stencil_func: u32,
    pub front_stencil_op_fail: u32,
    pub front_stencil_op_depth_fail: u32,
    pub front_stencil_op_depth_pass: u32,
    pub front_stencil_compare_mask: u32,
    pub front_stencil_write_mask: u32,
    pub viewport_enable: u32,
    /// `xOffset, xScale, yOffset, yScale, zOffset, zScale` from sceGxmSetViewport.
    pub viewport: [f32; 6],
    pub region_clip_mode: u32,
    /// `xMin, yMin, xMax, yMax` from sceGxmSetRegionClip.
    pub region_clip: [u32; 4],
    /// Occlusion query state for front-facing primitives (`SceGxmVisibilityTestMode`,
    /// the slot index within the visibility buffer, and `SceGxmVisibilityTestOp`). The
    /// buffer itself is context state, not per draw, so it lives in `VitaState`.
    pub front_visibility_test_enable: u32,
    pub front_visibility_test_index: u32,
    pub front_visibility_test_op: u32,
}

impl Default for RenderState {
    fn default() -> Self {
        RenderState {
            cull_mode: 0x0000_0000,               // SCE_GXM_CULL_NONE
            two_sided: 0x0000_0000,               // SCE_GXM_TWO_SIDED_DISABLED
            front_depth_func: 0x00C0_0000,        // SCE_GXM_DEPTH_FUNC_LESS_EQUAL
            back_depth_func: 0x00C0_0000,         // SCE_GXM_DEPTH_FUNC_LESS_EQUAL
            front_depth_write: 0x0000_0000,       // SCE_GXM_DEPTH_WRITE_ENABLED
            back_depth_write: 0x0000_0000,        // SCE_GXM_DEPTH_WRITE_ENABLED
            front_fragment_program_enable: 0x0,   // SCE_GXM_FRAGMENT_PROGRAM_ENABLED
            back_fragment_program_enable: 0x0,    // SCE_GXM_FRAGMENT_PROGRAM_ENABLED
            front_polygon_mode: 0x0000_0000,      // SCE_GXM_POLYGON_MODE_TRIANGLE_FILL
            back_polygon_mode: 0x0000_0000,       // SCE_GXM_POLYGON_MODE_TRIANGLE_FILL
            front_point_line_width: 1,
            front_stencil_ref: 0,
            front_stencil_func: 0x0E00_0000,      // SCE_GXM_STENCIL_FUNC_ALWAYS
            front_stencil_op_fail: 0,             // SCE_GXM_STENCIL_OP_KEEP
            front_stencil_op_depth_fail: 0,       // SCE_GXM_STENCIL_OP_KEEP
            front_stencil_op_depth_pass: 0,       // SCE_GXM_STENCIL_OP_KEEP
            front_stencil_compare_mask: 0xff,
            front_stencil_write_mask: 0xff,
            viewport_enable: 0x0000_0000,         // SCE_GXM_VIEWPORT_ENABLED
            viewport: [0.0; 6],
            region_clip_mode: 0x0000_0000,        // SCE_GXM_REGION_CLIP_NONE
            region_clip: [0; 4],
            front_visibility_test_enable: 0,      // SCE_GXM_VISIBILITY_TEST_DISABLED
            front_visibility_test_index: 0,
            front_visibility_test_op: 0,          // SCE_GXM_VISIBILITY_TEST_OP_INCREMENT
        }
    }
}

/// The blend state a fragment program was CREATED with.
///
/// GXM does not have a settable blend state: the blend equation is baked into the fragment
/// program at `sceGxmShaderPatcherCreateFragmentProgram`, from a `SceGxmBlendInfo` the title
/// passes there and never mentions again. A renderer that guesses instead - "opaque draws
/// REPLACE, everything else src-over" - gets a whole class of draws wrong in ways that look
/// like a shading bug: an ADDITIVE glow renders as a translucent quad, and a draw whose real
/// mode ignores alpha vanishes entirely when its shader happens to output alpha 0. That last
/// one is not hypothetical; it is why a race's entire world composite was invisible.
///
/// Fields are the raw GXM enum values (`SceGxmBlendFunc` / `SceGxmBlendFactor` /
/// `SceGxmColorMask`), so the capture stays a faithful record and the mapping to a host
/// blend state lives in the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlendState {
    /// `SceGxmColorMask` bits: which channels this program writes.
    pub color_mask: u8,
    /// `SceGxmBlendFunc` for colour and for alpha. `NONE` (0) means no blending at all -
    /// the source replaces the destination, whatever its alpha is.
    pub color_func: u8,
    pub alpha_func: u8,
    /// `SceGxmBlendFactor` source/destination, for colour and for alpha.
    pub color_src: u8,
    pub color_dst: u8,
    pub alpha_src: u8,
    pub alpha_dst: u8,
}

impl Default for BlendState {
    /// What GXM does for a fragment program created with a NULL `blendInfo`: write every
    /// channel, no blending.
    fn default() -> Self {
        BlendState {
            color_mask: 0xf,
            color_func: 0, // SCE_GXM_BLEND_FUNC_NONE
            alpha_func: 0,
            color_src: 1, // SCE_GXM_BLEND_FACTOR_ONE
            color_dst: 0, // SCE_GXM_BLEND_FACTOR_ZERO
            alpha_src: 1,
            alpha_dst: 0,
        }
    }
}

impl BlendState {
    /// Decode the packed 4 bytes of a guest `SceGxmBlendInfo`.
    ///
    /// Layout (vitasdk `gxm.h`): byte 0 is `colorMask`; byte 1 packs `colorFunc` in the low
    /// nibble and `alphaFunc` in the high; byte 2 `colorSrc` low, `colorDst` high; byte 3
    /// `alphaSrc` low, `alphaDst` high.
    pub fn from_bytes(b: [u8; 4]) -> Self {
        BlendState {
            color_mask: b[0],
            color_func: b[1] & 0xf,
            alpha_func: b[1] >> 4,
            color_src: b[2] & 0xf,
            color_dst: b[2] >> 4,
            alpha_src: b[3] & 0xf,
            alpha_dst: b[3] >> 4,
        }
    }

    /// Whether this program blends at all. `SCE_GXM_BLEND_FUNC_NONE` on both channels means
    /// the fragment REPLACES the destination - the case a src-over guess gets wrong.
    pub fn blends(&self) -> bool {
        self.color_func != 0 || self.alpha_func != 0
    }
}

/// A texture bound to a fragment sampler unit at draw time. Decoded from the
/// guest's 16-byte `SceGxmTexture` control words (format, dimensions, memory
/// layout, data address) with a snapshot of the referenced pixel bytes, so a
/// renderer can sample it without touching guest memory later. Fields are the
/// raw GXM enum parts: `base_format` is the high byte of `SceGxmTextureBaseFormat`
/// (e.g. `0x0c` for U8U8U8U8), `swizzle` is the format's low 24 bits, and
/// `tex_type` is the 3-bit `SceGxmTextureType` selector (`0b011` = LINEAR).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundTexture {
    pub unit: u32,
    pub base_format: u32,
    pub swizzle: u32,
    pub tex_type: u32,
    pub width: u32,
    pub height: u32,
    /// Bytes per row of the snapshotted `pixels` (== the source stride).
    pub stride: u32,
    /// How many `width x height` faces `pixels` holds: 1 for an ordinary texture, 6 for a
    /// CUBE map (`SCE_GXM_TEXTURE_CUBE`/`CUBE_ARBITRARY`), whose faces are stored back to
    /// back in +X, -X, +Y, -Y, +Z, -Z order - the same order WebGPU expects its array layers
    /// in. Face `f` starts at byte `f * face_bytes`.
    pub faces: u32,
    /// Byte size of one face in `pixels` (the whole buffer when `faces` is 1), INCLUDING
    /// every mip level of that face - so level 0 of face `f` still starts at
    /// `f * face_bytes`, whether or not the chain was snapshotted.
    pub face_bytes: u32,
    /// How many mip levels of each face `pixels` holds, counting level 0. Level `l` starts
    /// at `f * face_bytes + crate::render::level_offset(.., l)`.
    ///
    /// # Why the chain is snapshotted at all
    /// The hardware samples the guest's OWN mip levels. We used to read level 0 and box-filter
    /// a chain from it, which is defensible for an image and not defensible at all for a
    /// texture that is handed to the GPU still compressed: there is no box filter for a BC
    /// block, so a compressed upload either carries the guest's levels or ships level 0 alone -
    /// and level-0-alone is the "distant road reads as white speckle" failure
    /// ([[vitaslop-textures-need-mips]]).
    ///
    /// 1 when the guest declared no mips, when the read of the fuller chain failed (a texture
    /// whose allocation genuinely ends after level 0 - reported, never assumed), or for a CUBE
    /// map, whose six chains' interleaving is not established.
    pub levels: u32,
    pub data_addr: u32,
    /// The raw guest bytes of the texture, exactly as laid out in guest memory.
    ///
    /// Shared (`Arc`) rather than owned: one scene binds the same few textures across
    /// hundreds of draws, and a per-draw copy of a 4096x2048 shadow map costs gigabytes a
    /// frame. Sharing also gives consumers a cheap identity key (the pointer) for caching
    /// GPU uploads, so an unchanged texture is uploaded once per frame rather than per draw.
    pub pixels: Arc<[u8]>,
    /// Sampler wrap modes (`SceGxmTextureAddrMode`, 0 = REPEAT) and LOD bias set on
    /// this texture via `sceGxmTextureSet{U,V}AddrMode[Safe]` / `SetLodBias`.
    pub u_addr_mode: u32,
    pub v_addr_mode: u32,
    pub lod_bias: u32,
    /// Minification/magnification filters (`SceGxmTextureFilter`, 0 = POINT/nearest,
    /// 1 = LINEAR) set via `sceGxmTextureSetMinFilter`/`SetMagFilter`. The renderer
    /// bilinear-samples a LINEAR magnified texture and point-samples a POINT one, so
    /// small UI/font-atlas text a title draws with LINEAR filtering is smooth rather
    /// than the broken thin strokes nearest sampling gives at sub-native scale.
    pub min_filter: u32,
    pub mag_filter: u32,
    /// `SceGxmTextureMipFilter` (control word 0 bit 9, `vita::gxm::texword0::MIP_FILTER`):
    /// 1 = the hardware filters BETWEEN mip levels, 0 = it does not.
    ///
    /// # Why this is worth carrying, and what it does NOT license
    /// A texture with `mip_count` 1 and `mip_filter` 0 is one the hardware samples from its
    /// base level and nothing else. Our renderer box-filters a chain for every RGBA8 texture
    /// regardless, which is more anti-aliased than the device and was added to fix real
    /// speckle - so this is not a licence to stop doing that. What it settles is narrower: for
    /// a texture handed to the GPU still COMPRESSED, whose chain cannot be generated at all,
    /// "the guest declares no levels AND does not filter between them" is the difference
    /// between shipping one level faithfully and dropping a chain the hardware was using.
    pub mip_filter: u32,
    /// The `SceGxmTextureGammaMode` set via `sceGxmTextureSetGammaMode`. Non-zero means the
    /// hardware sampler sRGB-DECODES each texel BEFORE filtering, so the shader receives
    /// linear values from memory that holds gamma-encoded ones. A renderer that ignores this
    /// hands the shader the encoded bytes as if they were linear, and everything derived from
    /// that texture comes out too BRIGHT in the mid-tones.
    pub gamma: u32,
}

/// The per-material fragment-shader inputs recovered by reflecting the bound fragment
/// program's parameter table against its captured default uniform buffer. The real
/// fragment program is a standard forward-lit material - `albedo = baseTexture.rgb *
/// tint`, lit by one directional light plus an ambient term, then fogged - so these are
/// the values the capture renderer needs to reproduce the LIT colour instead of the raw
/// albedo texel. Everything is optional: a 2D/UI shader declares none of it and the fields
/// stay at their neutral defaults (tint white, no light), so the material is a no-op there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FragmentMaterial {
    /// Base-colour multiplier (`AlbedoColour` / `Primarytint`). The sampled albedo texel is
    /// multiplied by this; it is why an unlit near-white tyre albedo must be scaled down.
    pub tint: [f32; 3],
    /// World-space direction of the key directional light (`directionalLight0Direction
    /// WorldSpace`) and its colour (`directionalLight0Colour`). `has_light` is false when the
    /// shader declares no directional light (then only ambient/flat albedo applies).
    pub light_dir: [f32; 3],
    pub light_col: [f32; 3],
    pub has_light: bool,
    /// A flat ambient colour (the average of the tiny `diffuseAmbientMap` irradiance texture,
    /// resolved by the renderer) added to the directional term so surfaces facing away from
    /// the light are not pure black. Defaults to a neutral mid-grey when no ambient map is
    /// bound. This is a scene-lighting term, not per-material, but stored per draw for the
    /// stateless renderer.
    pub ambient: [f32; 3],
}

impl Default for FragmentMaterial {
    fn default() -> Self {
        FragmentMaterial {
            tint: [1.0, 1.0, 1.0],
            light_dir: [0.0, -1.0, 0.0],
            light_col: [1.0, 1.0, 1.0],
            has_light: false,
            ambient: [0.35, 0.35, 0.35],
        }
    }
}

/// No shader container: what [`Draw::vprog`] and [`Draw::fprog`] hold off the recompiler path,
/// and what a synthetic draw (a test, a probe) carries.
///
/// A named constructor rather than a `Default` impl on `Draw`: every other field of a draw is a
/// real capture with no sensible default, and a whole-struct default would make it possible to
/// build a draw that describes nothing.
///
/// One SHARED empty container, so this is a refcount bump rather than an allocation: it is
/// called twice per draw on the fixed-function path, where the whole point is that the
/// recompiler payload costs nothing.
pub fn no_program() -> std::sync::Arc<[u8]> {
    static EMPTY: std::sync::OnceLock<std::sync::Arc<[u8]>> = std::sync::OnceLock::new();
    EMPTY.get_or_init(|| std::sync::Arc::from(&[][..])).clone()
}

/// A single draw call with everything needed to reproduce it, snapshotted from
/// guest memory at draw time (so later guest writes cannot perturb it).
#[derive(Clone, Debug, PartialEq)]
pub struct Draw {
    pub primitive: u32,
    pub index_format: u32,
    pub index_count: u32,
    /// The bound vertex stream buffer bytes.
    ///
    /// Shared, not owned: the renderer's scene builder hands these straight to the
    /// recompiled path, and a `Vec` there meant a full copy of every draw's mesh on every
    /// frame - MEASURED at 2.4-3.2 MB per frame mid-race on a 500-700 draw title, allocated
    /// and freed sixty times a second. An `Arc` makes that handoff a refcount bump, and it
    /// gives the buffer an IDENTITY, which is what lets a consumer cache anything derived
    /// from it (see the index expansion in `RenderSceneBuilder`).
    pub vertices: Arc<[u8]>,
    pub vertex_stride: u32,
    pub attributes: Vec<VertexAttribute>,
    /// The index buffer bytes. `Arc` for the same reason as `vertices` above.
    pub indices: Arc<[u8]>,
    /// The vertex default uniform buffer contents the guest wrote for this draw
    /// (column-major 4x4 MVP for the cube), if any.
    pub uniforms: Vec<f32>,
    /// Fragment textures bound at draw time (one per active sampler unit),
    /// snapshotted from guest memory. Empty for an untextured (vertex-color) draw.
    ///
    /// Ordered so that index 0 is the draw's surface albedo when it has one - see
    /// [`Draw::albedo`], which is what the fixed-function approximation samples. The full list
    /// is what the GXP recompiler binds by unit, so nothing is ever dropped from it.
    ///
    /// SHARED rather than owned: for a fixed set of bindings inside one scene this list is
    /// bitwise identical for every draw that uses it (see `TextureSnapshots::snapshot_sets`), so
    /// the ~650 draws of a race frame get one `Arc` clone each instead of ~650 allocations and
    /// as many rebuilds.
    pub textures: std::sync::Arc<[BoundTexture]>,
    /// Textures bound to VERTEX-stage sampler units at draw time. Separate from
    /// [`Self::textures`] because the two stages number their sampler units independently, and
    /// a vertex program that fetches a texture is building its GEOMETRY from it - so a draw
    /// that loses these renders no vertices, not an untextured surface.
    pub vertex_textures: Vec<BoundTexture>,
    /// The fixed-function pipeline state (cull/depth/stencil/viewport/...) in effect
    /// for this draw, snapshotted from the sticky GXM context state. See [`RenderState`].
    pub render_state: RenderState,
    /// The blend equation baked into the bound fragment program - see [`BlendState`]. This is
    /// state, not a guess: GXM has no runtime blend setter, so this is the only source of it.
    pub blend: BlendState,
    /// The bound fragment program's `SceGxmProgram*`. Diagnostic, and the one that ties a draw
    /// to a BLOB: a title can register the SAME fragment shader twice with different blend
    /// equations, so the shader bytes alone do not identify which `SceGxmFragmentProgram` a
    /// draw used - and the two render completely differently. It is also the address the shader
    /// dumps are named by (`frag_<header>.gxp`).
    pub fragment_program_header: u32,
    /// Scene exposure (linear multiplier) recovered from the vertex program's reflected
    /// `vsCoarseExposureReg` uniform. The shaders scale lit albedo by this before
    /// tone-mapping, so a capture renderer that skips it draws the world ~10x too dark.
    /// 1.0 when the shader declares no exposure (2D/UI), so it is a no-op there.
    pub exposure: f32,
    /// The per-material fragment inputs (base-colour tint + directional/ambient light)
    /// reflected from the fragment program and its default uniform buffer. Neutral (a no-op)
    /// for 2D/UI draws and any shader that declares no lit material. See [`FragmentMaterial`].
    pub material: FragmentMaterial,
    /// The model-to-world matrix (column-major 4x4) reflected from the vertex program's
    /// `vsModelToWorldMatrix`, used to bring the per-vertex object-space normal into world
    /// space for the directional-light N.L term (the light direction is world space). Identity
    /// when the shader has no separate world matrix (2D/UI), so lighting there uses the raw
    /// normal - harmless because such draws are not depth-lit.
    pub world: [f32; 16],
    /// The bound vertex `SceGxmProgram` container bytes, snapshotted for the GXP->WGSL
    /// recompiler (live guest-shader) path. Empty unless recompile-capture is enabled
    /// (env `VITASLOP_GXP_LIVE`), so the default capture pays no read/clone cost.
    ///
    /// `Arc<[u8]>`, not `Vec<u8>`, because a program container is IMMUTABLE for as long as it
    /// is registered and every draw bound to it wants the same bytes: a race frame submits
    /// 400+ draws over a couple of dozen distinct programs, so a per-draw read out of guest
    /// memory plus a per-draw clone copies the same few kilobytes hundreds of times a frame
    /// for nothing. The cache that hands these out is `VitaState::program_blobs`, invalidated
    /// at exactly the moment a header address can come to mean a different program.
    pub vprog: std::sync::Arc<[u8]>,
    /// The bound fragment `SceGxmProgram` container bytes. Empty off the recompiler path.
    pub fprog: std::sync::Arc<[u8]>,
    /// Raw vertex default-uniform-buffer (SA bank) bytes exactly as the guest wrote them -
    /// the recompiled vertex shader reads these directly, NOT the MVP-stamped `uniforms`
    /// above (which the fixed-function path needs but the real shader recomputes itself).
    pub vert_sa: Vec<u8>,
    /// Raw fragment default-uniform-buffer (SA bank) bytes exactly as the guest wrote them,
    /// consumed by the recompiled fragment shader's `@group(1)` uniform. Empty off-path.
    pub frag_sa: Vec<u8>,
    /// GUEST ADDRESS the bytes above were read from, or 0 when there is no bound buffer.
    ///
    /// The bytes alone answer "what did this draw get"; only the address answers "who put it
    /// there", and that is a different and often harder question. This block may come from our
    /// recycled reserve ring OR - on a title that binds through precomputed states, which is
    /// most of them - from guest-owned memory the guest allocated and writes itself. In the
    /// second case the address is stable enough to point `VITASLOP_WATCH_STORE_LOG` at, which
    /// is the one tool that names every guest writer of an address in a single run.
    pub frag_sa_addr: u32,
    /// The vertex program SYNTHESIZES this draw's primitive rather than reading it: the
    /// stream holds one record per sprite (a centre plus an expansion basis - a
    /// scale/rotation, or an explicit right/up billboard axis pair) and the shader builds
    /// the corners. See [`crate::host::VitaState::reflected_shader_expanded`].
    ///
    /// The fixed-function approximation has no shader, so there is nothing here it can
    /// rasterize: joining the raw records as triangles connects unrelated sprite centres
    /// into geometry the game never draws. The software renderer skips such a draw and
    /// says so; the GXP recompiler runs the real vertex program and renders it properly.
    pub shader_expanded: bool,
}

impl Draw {
    /// The texture the fixed-function approximation samples across this draw's triangles.
    ///
    /// This is index 0 by the ordering contract on [`Draw::textures`]. Note that a draw can
    /// bind textures that are not surface albedo at all - a one-dimensional lookup table (a fog
    /// ramp, indexed by depth) or a cube map (an irradiance probe, indexed by a direction) - and
    /// for a draw that binds ONLY those, the approximation stretches one across the surface. It
    /// is visibly not what the guest draws (a start-line decal paints as vertical bands of the
    /// fog ramp), but it is better than the alternative: refusing to pick one leaves the draw
    /// with no texture at all, which the renderers paint as the magenta missing-texture marker,
    /// and it hides real geometry. Reproducing these materials needs the recompiled fragment
    /// shader that reads each sampler for what it is.
    pub fn albedo(&self) -> Option<&BoundTexture> {
        self.textures.first()
    }
}

/// One scene (BeginScene to EndScene): its render target color buffer and the
/// draws issued into it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    pub color: Option<ColorSurface>,
    /// The `SceGxmDepthStencilSurface` this scene rendered its depth into, when the guest
    /// passed one to `sceGxmBeginScene`.
    ///
    /// Recorded because a later pass may SAMPLE this scene's depth - soft particles, fog and
    /// SSAO all read the depth of the pass before them - and the only way to recognise such a
    /// texture is to match its data pointer against the depth buffers the frame has rendered.
    /// Without this the address falls through to a colour target (they are allocated near each
    /// other) or to guest bytes the GPU never wrote, and the pass reads a colour as a distance.
    pub depth: Option<DepthSurface>,
    /// The `SceGxmMultisampleMode` of the RENDER TARGET this scene rasterises through
    /// (`SceGxmRenderTargetParams::multisampleMode`): 0 = NONE, 1 = 2X, 2 = 4X.
    ///
    /// This lives on the scene rather than on the colour surface because it is a property
    /// of the target, and it is the guest's own statement of how many samples per pixel the
    /// hardware rasterises this pass with. It is NOT the same fact as
    /// [`ColorSurface::scale_mode`]: the scale mode says the surface stores the RESOLVED
    /// image, the multisample mode says how many samples were resolved into it. A renderer
    /// needs both - the first to know the stored size is the sampled size, the second to
    /// know the sample count - and reading either alone has already produced a wrong
    /// conclusion here (see `report_scene_target`).
    pub multisample: u32,
    pub draws: Vec<Draw>,
}

/// A scene's `SceGxmDepthStencilSurface`, as the published struct layout gives it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DepthSurface {
    /// `zlsControl`: format, surface type, stride and the two force load/store bits.
    pub zls_control: u32,
    /// Where the depth samples live in guest memory. This is the address a later pass names
    /// when it binds this scene's depth as a texture.
    pub depth_addr: u32,
    /// Where the stencil samples live, or 0 when the format carries no stencil.
    pub stencil_addr: u32,
    /// The depth the surface clears/loads as background, as raw f32 bits (GXM's default is 1.0).
    pub background_depth: u32,
}

/// Report - once per (target, from, to) - that a scene's extent was taken from its draws'
/// VIEWPORT because every struct the guest filled in described it as degenerate.
///
/// This is a guess, and it decides the resolution a whole pass rasterises at - which every
/// later pass sampling that buffer then derives its own texel size and screen-space bias from.
/// A pass silently rasterised at a size no guest struct asked for is exactly the shape of cause
/// that makes an authored sampling bias look wrong, so it says so in every run rather than
/// behind a debug flag.
fn report_adopted_viewport_extent(data_addr: u32, from: (u32, u32), to: (u32, u32)) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32, u32, u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((data_addr, from.0, from.1, to.0, to.1)) {
        return;
    }
    eprintln!(
        "gxm scene extent: colour {data_addr:#x} was described as {}x{} by EVERY guest struct \
         (colour surface and render target), so its extent is being taken from its draws' \
         VIEWPORT instead: {}x{}. That is a guess, and everything derived from this pass's \
         resolution downstream inherits it.",
        from.0, from.1, to.0, to.1
    );
}

impl Scene {
    /// Give this scene the extent its own draws were rasterized at, when its colour
    /// surface cannot possibly be right.
    ///
    /// # Why a scene's target size is not simply the colour surface's
    /// The surface's `width`/`height` are what a title passed to
    /// `sceGxmColorSurfaceInit`, and a title is entitled to leave them meaningless for
    /// a render-to-texture pass: the render target and the viewport are what decide
    /// where the GPU rasterizes. One retail title does exactly that - its front-end
    /// renders a 20,160-triangle map through a colour surface initialised **1x1 with
    /// stride 8**, on a render target created 1x1, while every draw in the pass sets a
    /// **960x544 viewport**. Believing the surface made that whole pass one pixel, and
    /// the finished frame showed the HUD over black: a screen that looks like a title
    /// which renders nothing, rather than one pass dropped.
    ///
    /// The viewport is the guest's own statement of the pixel extent it is drawing
    /// into (GXM's viewport is offset/scale in pixels, so the width is `2*|xScale|`),
    /// so it is the sound thing to fall back to. This only fires where the surface is
    /// DEGENERATE - a zero or single-pixel dimension, which nothing can rasterize a
    /// real viewport into - so a title whose surface extents are honest is untouched,
    /// and a genuinely tiny render target (a 1x1 probe) keeps its size as long as its
    /// viewport agrees.
    pub fn adopt_viewport_extent(&mut self) {
        let Some(c) = self.color.as_mut() else { return };
        if c.width > 1 && c.height > 1 {
            return;
        }
        let Some(first) = self.draws.first() else { return };
        // Only when the viewport is actually in effect (`SCE_GXM_VIEWPORT_ENABLED` is
        // 0); with it disabled the transform is the render target's and this says
        // nothing.
        if first.render_state.viewport_enable != 0 {
            return;
        }
        let v = first.render_state.viewport;
        let (vw, vh) = ((2.0 * v[1].abs()) as u32, (2.0 * v[3].abs()) as u32);
        if vw > 1 && vh > 1 {
            report_adopted_viewport_extent(c.data_addr, (c.width, c.height), (vw, vh));
            c.width = vw;
            c.height = vh;
        }
    }

    /// Reduce the scene's draws to the neutral [`DrawBatch`](vitaslop_platform::gpu::DrawBatch)
    /// list the shared cube pipeline consumes. Keeps only draws the fixed-function
    /// cube shader can render (triangle lists with a 16-byte interleaved vertex
    /// and a full 4x4 MVP), so both the native and browser GPU paths select the
    /// same draws from one place. The software rasterizer stays independent.
    pub fn draw_batches(&self) -> Vec<vitaslop_platform::gpu::DrawBatch> {
        self.draws
            .iter()
            .filter(|d| d.primitive == 0 && d.uniforms.len() >= 16 && d.vertex_stride == 16)
            .map(|d| {
                let mut mvp = [0f32; 16];
                mvp.copy_from_slice(&d.uniforms[..16]);
                vitaslop_platform::gpu::DrawBatch {
                    mvp,
                    vertices: d.vertices.to_vec(),
                    indices: d.indices.to_vec(),
                    index_count: d.index_count,
                    // GXM index format: 0 is U16, anything else U32.
                    index_u32: d.index_format != 0,
                }
            })
            .collect()
    }
}

/// A game->OS "egress" event: something the guest handed to the system that
/// carries game-authored, human-readable meaning - a savedata write, a trophy
/// unlock, a leaderboard score submission. This is the content-ful, title-agnostic
/// seam a conformance recipe asserts on: it is the Vita OS API surface, NOT the
/// game's private memory (which would be per-title RE and brittle). Every event is
/// tagged with the display frame it occurred on, so a recipe can assert both what
/// the game did and (loosely) when - the human-readable proof a run reached a game
/// milestone, replacing a screenshot assertion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EgressEvent {
    pub frame: u64,
    pub kind: EgressKind,
}

/// The kinds of game-authored egress the ledger records. Deliberately small and
/// generic - each variant is a Vita OS surface every title shares, so the ledger
/// generalizes to the next game with no per-title work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EgressKind {
    /// A writable file under a persisted mount (`savedata0:`, `ux0:`) was closed
    /// after being written: the game persisted state. `bytes` is the final size and
    /// `ascii` is a printable-ASCII preview (non-printable bytes shown as `.`), so a
    /// human-readable save - a high score, an unlocked level id - is visible without
    /// a pixel. This is where a title's score often lands on the offline path.
    SaveWrite { path: String, bytes: usize, ascii: String },
    /// A trophy was unlocked (`sceNpTrophyUnlockTrophy`): the trophy id.
    Trophy { id: i32 },
    /// A score was submitted to a leaderboard (`sceNpScore*`): board id and value.
    /// Only fires if the online path is enabled (offline forces signed-out).
    ScoreSubmit { board: u32, score: i64 },
}

/// Render up to `max` bytes of `data` as a printable-ASCII preview: printable bytes
/// verbatim, everything else as `.`. Keeps the egress ledger human-readable and
/// content-bounded (no raw blobs) for logs and assertions.
pub fn ascii_preview(data: &[u8], max: usize) -> String {
    data.iter()
        .take(max)
        .map(|&b| if (0x20..0x7f).contains(&b) { b as char } else { '.' })
        .collect()
}

/// The whole recorded stream across a run.
#[derive(Default)]
pub struct Capture {
    /// Completed scenes, in submission order (one per BeginScene/EndScene pair).
    pub scenes: Vec<Scene>,
    /// Display buffer addresses presented, in order (from the display queue).
    pub presents: Vec<u32>,
    /// (library_nid, func_nid) pairs the guest called that are not implemented
    /// yet, with a count, so gaps are visible instead of silently skipped.
    pub unimplemented: Vec<(u32, u32, String)>,
    /// Total host calls serviced, for sanity.
    pub call_count: u64,
    /// Ordered trace of recently serviced calls' function NIDs, for debugging.
    /// Bounded to the most recent [`TRACE_CAP`] entries (see [`Capture::record_call`]):
    /// a 3D title makes tens of millions of host calls during boot, and every
    /// consumer of this trace (the exit dump, the probe's trace.txt and per-thread
    /// tails) only reads the recent window.
    pub trace: Vec<u32>,
    /// The guest thread id that made each serviced call, parallel to [`trace`](Self::trace)
    /// (0 outside the preemptive scheduler). Lets a trace be split by thread - e.g. to
    /// see what the main thread did versus a worker.
    pub trace_thid: Vec<i32>,
    /// Bytes the guest wrote to the debug console (sceClibPrintf and friends), in
    /// order. This is the blob-free "it printed" signal for the hello corpus: the
    /// Vita has no framebuffer console for these, so the host is the sink. Also fed
    /// by sceIoWrite to fd 1 (the path newlib's stdout takes).
    pub stdout: Vec<u8>,
    /// Bytes written to fd 2 (stderr) via sceIoWrite.
    pub stderr: Vec<u8>,
    /// Diagnostic sample of the first few `sceKernelWaitLwCond` calls:
    /// `(cond work addr, timeout pointer, timeout value or 0)`. Lets the probe tell
    /// a producer/consumer wait (no timeout) from a timed delay loop (finite
    /// timeout) without a full trace. Capped so it costs nothing after warm-up.
    pub lwcond_wait_samples: Vec<(u32, u32, u32)>,
    /// The game->OS egress ledger: game-authored, human-readable events (savedata
    /// writes, trophies, score submissions) in occurrence order, each frame-tagged.
    /// This is the content-free conformance-assertion surface (see [`EgressEvent`]).
    pub egress: Vec<EgressEvent>,
    /// How many completed scenes to keep. `None` (the default) keeps every scene,
    /// which is what a short capture-and-inspect run wants.
    ///
    /// A LONG run must set it. Each scene holds a snapshot of every draw's vertex
    /// window and indices - on a real 3D title, hundreds of draws and megabytes per
    /// frame - so retaining them all costs gigabytes within a couple of thousand
    /// frames and eventually ends the run. That is a hard ceiling on how far into a
    /// game anything can get, which makes it a ceiling on playing the game at all.
    /// See [`Capture::push_scene`]: eviction folds the dropped scene into
    /// [`retired_digest`](Self::retired_digest), so bounding retention does NOT change
    /// the determinism signature by one bit.
    pub scene_limit: Option<usize>,
    /// The running signature fold over scenes already evicted by `scene_limit`.
    pub retired_digest: u64,
    /// Set when a caller has said nothing will read the signature - see
    /// [`Capture::set_signature_wanted`]. Stored INVERTED so `Default` means folding is on:
    /// the safe state has to be the one you get by forgetting.
    fold_disabled: bool,
    /// Set the moment a scene is retired without being folded, so
    /// [`Capture::signature`] can refuse instead of returning a hash of part of a run.
    signature_incomplete: bool,
    /// How many scenes have been evicted (so `scenes.len() + retired_scenes` is the
    /// true count for a report).
    pub retired_scenes: u64,
    /// Scenes pushed since the last display flip - the frame under construction.
    frame_scenes: usize,
    /// Scenes the PREVIOUS display frame was made of, latched at the flip.
    ///
    /// A title is not one scene per frame. A racing title's race frame is fifteen: six
    /// small reflection/shadow passes, a 720x408 world pass, a post pass over it, and a
    /// composite that blits the world and draws the HUD. Every observer that wants "the
    /// scene" wants a specific one of those, and which one is a question about CONTENT,
    /// not about order - so [`Capture::frame_scenes`] hands out the whole frame and
    /// [`Capture::world_scene`] picks by what the scene contains.
    prev_frame_scenes: usize,
}

/// Upper bound on retained trace entries. When the trace reaches this, the oldest
/// half is dropped, so memory stays bounded (~32 MB at the cap) while every
/// consumer keeps at least [`TRACE_CAP`]/2 of recent history.
pub const TRACE_CAP: usize = 4 << 20;

/// The FNV-1a offset basis, the seed of the determinism signature fold.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold `bytes` into an FNV-1a accumulator.
fn fnv(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(FNV_PRIME);
    }
}

/// How many independent FNV chains [`fnv_bulk`] runs over a blob. Each lane is an
/// ordinary FNV-1a over a strided subsequence, so they are independent and the CPU
/// pipelines them; one chain is bound by the 64-bit multiply's latency, not its
/// throughput, and spends most of its cycles waiting.
const FNV_LANES: usize = 8;

/// Fold a LARGE byte blob into the accumulator.
///
/// Same construction as [`fnv`] - xor-then-multiply, defined purely on the byte
/// sequence - but run as [`FNV_LANES`] interleaved chains that are combined at the
/// end, because these blobs are the scene's vertex and index buffers: megabytes per
/// frame, and the single serial chain was the whole cost of `sceGxmEndScene`.
///
/// Still exactly as deterministic and as engine-independent as the serial version:
/// the result is a pure function of the byte sequence, with no endianness, word size
/// or allocation dependence. It is NOT the same VALUE as the serial fold, so
/// signatures do not compare across this change - they never had to, since a
/// signature is only ever compared between runs of the same build (`explore`'s
/// bucketing, a recipe's `@sig`).
fn fnv_bulk(h: &mut u64, bytes: &[u8]) {
    // Length first, so appending zero bytes cannot leave the digest unchanged and two
    // differently-split blobs cannot collide by concatenation.
    fnv(h, &(bytes.len() as u64).to_le_bytes());
    let mut lanes = [*h; FNV_LANES];
    // Lane `i` takes bytes i, i+LANES, i+2*LANES, ... - a fixed, size-independent
    // assignment, so the same blob always distributes the same way.
    let chunks = bytes.chunks_exact(FNV_LANES);
    let tail = chunks.remainder();
    for c in chunks {
        for (lane, &b) in lanes.iter_mut().zip(c) {
            *lane ^= b as u64;
            *lane = lane.wrapping_mul(FNV_PRIME);
        }
    }
    for (lane, &b) in lanes.iter_mut().zip(tail) {
        *lane ^= b as u64;
        *lane = lane.wrapping_mul(FNV_PRIME);
    }
    // Combine in lane order, through the same primitive, so the lanes' contributions
    // stay ordered and one lane's change cannot be cancelled by another's.
    for lane in lanes {
        fnv(h, &lane.to_le_bytes());
    }
}

/// Fold one scene's observable content into the signature accumulator.
fn fold_scene(h: &mut u64, s: &Scene) {
    if let Some(c) = &s.color {
        fnv(h, &c.data_addr.to_le_bytes());
        fnv(h, &c.format.to_le_bytes());
    }
    for d in &s.draws {
        // The fold's cost is its VOLUME, and volume is the half of it a browser can report -
        // there is no clock there to time this with. See `crate::perf::note_bytes`.
        crate::perf::note_bytes(
            crate::perf::Phase::SceneFold,
            d.vertices.len() + d.indices.len() + d.uniforms.len() * 4,
        );
        fnv_bulk(h, &d.vertices);
        fnv_bulk(h, &d.indices);
        // One pass over the uniform floats as bytes, rather than a call per float.
        // `to_le_bytes` per element is what makes this a copy rather than a cast.
        let mut buf = Vec::with_capacity(d.uniforms.len() * 4);
        for u in &d.uniforms {
            buf.extend_from_slice(&u.to_le_bytes());
        }
        fnv_bulk(h, &buf);
    }
}

/// Say - once - that this run cannot produce a signature because it was not folding scenes.
fn report_signature_incomplete() {
    static SEEN: std::sync::Once = std::sync::Once::new();
    SEEN.call_once(|| {
        tracing::warn!(
            target: "vitaslop::gxm",
            "the determinism signature was asked for on a run that had scene folding OFF, so \
             scenes were retired without being hashed and no signature exists for it. Returning \
             u64::MAX rather than a partial hash that would look real and compare unequal for no \
             recorded reason. Whoever needs the signature must call \
             `Capture::set_signature_wanted(true)` before the run."
        );
    });
}

/// Ceiling on how many scenes [`Capture::push_scene`] will retain on account of them
/// belonging to one display frame. A real frame is a handful of passes (six for a
/// front-end, fifteen for a race frame), so this is far above any title's frame and
/// exists only so a run that submits scenes without ever flipping stays bounded.
const MAX_FRAME_SCENES: usize = 64;

/// Say so, once, when a single frame carried more passes than [`MAX_FRAME_SCENES`],
/// because past that point the retained set is no longer a whole frame and a shot of
/// it is missing passes. Silence here would look exactly like a frame that rendered.
fn report_frame_scene_cap(passes: usize) {
    static SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SEEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!(
            "capture: a display frame carried {passes} passes, over the {MAX_FRAME_SCENES} this \
             retains - the oldest are evicted, so anything rendered from this frame is missing \
             them. Raise `scene_limit` if a whole frame is needed."
        );
    }
}

impl Capture {
    pub fn new() -> Self {
        Capture { retired_digest: FNV_OFFSET, ..Capture::default() }
    }

    /// Record a completed scene, evicting the oldest if `scene_limit` is set.
    ///
    /// Eviction folds the dropped scene into [`retired_digest`](Self::retired_digest)
    /// FIRST, so [`signature`](Self::signature) still covers the whole run. Bounding
    /// memory must not quietly change what a run reports about itself: a signature
    /// that depended on the retention window would compare unequal between a short
    /// run and a long one and destroy the only cross-engine equivalence check there
    /// is.
    pub fn push_scene(&mut self, scene: Scene) {
        self.scenes.push(scene);
        self.frame_scenes += 1;
        let Some(limit) = self.scene_limit else { return };
        // Never evict below ONE WHOLE FRAME - the scenes of the last completed frame
        // plus the one being built. A frame is not a scene: a title renders its world,
        // shadows and post passes offscreen and then composites them, so a capture
        // holding the last scene alone holds only the composite. Every observer that
        // renders "the frame" ([`Self::frame_scenes`], and `@shot` through it) would
        // then draw the composite over black and report it as the picture - a shot
        // that is not merely imperfect but actively misleading. The bound stays a
        // bound: this is one frame's passes, not the run's.
        //
        // Capped by [`MAX_FRAME_SCENES`], because "one frame" is only a bound while a
        // frame ends: a title that submits scenes and never flips would otherwise
        // retain all of them.
        let whole_frame = (self.prev_frame_scenes + self.frame_scenes).min(MAX_FRAME_SCENES);
        if self.prev_frame_scenes + self.frame_scenes > MAX_FRAME_SCENES {
            report_frame_scene_cap(self.prev_frame_scenes + self.frame_scenes);
        }
        let limit = limit.max(1).max(whole_frame);
        while self.scenes.len() > limit {
            let old = self.scenes.remove(0);
            // >>> THE FOLD IS SKIPPED WHEN NOTHING WILL READ IT, AND THE SKIP IS RECORDED.
            //
            // Folding hashes every retired scene's vertices, indices and uniforms - about 3 MB
            // a frame on this title's race, MEASURED at 7.9% of the whole frame. That is the
            // right price for the only cross-engine equivalence check there is, and it is pure
            // waste on a live player session, which never asks for the number: the browser reads
            // `signature()` only when it is EVALUATING a recipe, and the user's own captures say
            // `recipe: (none)`.
            //
            // Skipping silently would be far worse than the cost, so the skip is remembered and
            // `signature()` refuses to make one up - see there.
            if self.fold_disabled {
                self.signature_incomplete = true;
                self.retired_scenes += 1;
                continue;
            }
            // A `Capture` built by `Default` starts at 0, not the FNV basis; seed it
            // on first use so both construction paths fold identically.
            if self.retired_scenes == 0 && self.retired_digest == 0 {
                self.retired_digest = FNV_OFFSET;
            }
            let mut h = self.retired_digest;
            crate::perf::time(crate::perf::Phase::SceneFold, || fold_scene(&mut h, &old));
            self.retired_digest = h;
            self.retired_scenes += 1;
        }
    }

    /// Whether this run will ever be asked for its determinism [`signature`](Self::signature).
    ///
    /// ON by default, so every existing tool - `explore`, `memdiff`, the recipe runner, the
    /// session - keeps working with no change and no chance of being handed a hash of half a
    /// run. A LIVE PLAYER turns it off, because it never asks.
    ///
    /// This is not a knob and must not become one: it is a statement about whether the caller
    /// has a consumer, and a caller that turns it off and then asks gets a refusal rather than
    /// a number ([[vitaslop-fast-fail-no-silent-success]]).
    pub fn set_signature_wanted(&mut self, wanted: bool) {
        self.fold_disabled = !wanted;
    }

    /// The determinism signature over the run's whole observable output: every
    /// scene's render stream (evicted ones via `retired_digest`) then the egress
    /// ledger. Engine-independent by construction - it covers what the guest
    /// PRODUCED, never internal RAM or thread timing - so native, headless and
    /// browser runs of the same recipe must agree on it.
    ///
    /// # It REFUSES rather than answering when scenes were retired unfolded
    /// A run that turned [`set_signature_wanted`](Self::set_signature_wanted) off dropped scenes
    /// without folding them, so no honest signature exists for it. Returning the partial hash
    /// would be a number that looks exactly like a real one and compares unequal for a reason
    /// nothing records - the worst possible failure for the one value the whole cross-engine
    /// equivalence check rests on. `u64::MAX` is returned instead, loudly, so a comparison
    /// against it fails and says why.
    pub fn signature(&self) -> u64 {
        if self.signature_incomplete {
            report_signature_incomplete();
            return u64::MAX;
        }
        let mut h = if self.retired_scenes == 0 && self.retired_digest == 0 {
            FNV_OFFSET
        } else {
            self.retired_digest
        };
        for s in &self.scenes {
            fold_scene(&mut h, s);
        }
        for ev in &self.egress {
            fnv(&mut h, &ev.frame.to_le_bytes());
            fnv(&mut h, format!("{:?}", ev.kind).as_bytes());
        }
        h
    }

    /// Latch the scene count of the frame that just finished. Called once per display
    /// flip, so [`frame_scenes`](Self::frame_scenes) can hand back exactly the scenes
    /// the last completed frame was built from.
    pub fn end_frame(&mut self) {
        self.prev_frame_scenes = self.frame_scenes;
        self.frame_scenes = 0;
    }

    /// The scenes of the most recently COMPLETED display frame, oldest first.
    ///
    /// Bounded by what `scene_limit` actually retained: a run that keeps one scene gets
    /// one back, which is the old behaviour and still correct for a single-pass title.
    /// Between flips (a partially built frame) this reports the previous frame's tail
    /// rather than a mixture, because an observer asking about "this frame" during
    /// construction has no complete frame to be given.
    pub fn frame_scenes(&self) -> &[Scene] {
        let n = self.prev_frame_scenes.max(1).min(self.scenes.len());
        &self.scenes[self.scenes.len() - n..]
    }

    /// The scene of the last completed frame that holds the PLAYER'S VIEW - the pass an
    /// observer means when it asks where things are or which way the vehicle points.
    ///
    /// Never by order: the last scene of a multi-pass frame is the composite, a handful of
    /// full-screen quads and a HUD, which is why a racing title reported nothing at all
    /// from `locate` while rendering a whole racetrack.
    ///
    /// # Why triangle count is the WRONG selector, measured
    /// It was the selector, and it silently chose a different pass from frame to frame.
    /// A race frame carries a rear-view MIRROR pass, and a mirror draws the same world
    /// through a camera pointing the other way - so its triangle count sits within a few
    /// percent of the main view's and crosses over depending on what happens to be behind
    /// the car. Every crossover flipped the reported heading by 180 degrees. That is
    /// indistinguishable from a car that has spun, and it was read as one: a controller
    /// steering on it fought an imaginary spin and abandoned the lap, three times, at
    /// three different corners. The position stayed smooth throughout, which is the
    /// contradiction that gives the bug away - a car cannot swap ends and hold a straight
    /// line.
    ///
    /// # The rule
    /// Among scenes with a RECOVERABLE CAMERA, the largest render target wins (ties on
    /// world triangles). The player's view is by construction the biggest thing drawn:
    /// a mirror, a reflection and an environment-probe face are all smaller. Requiring a
    /// camera is what excludes the rest for free - a shadow pass is drawn through an
    /// ORTHOGRAPHIC matrix, which is affine, so it has no eye to recover
    /// ([`scene_eye`](crate::render::scene_eye) returns `None`), and the composite has no
    /// world-to-clip matrix at all.
    ///
    /// A title whose frame is one scene selects that scene, unchanged, and a frame with no
    /// camera anywhere falls back to the old most-world-geometry reading.
    ///
    /// # Largest is not enough: a shadow map is bigger than the screen
    /// "Largest target with a camera" assumed a shadow pass has no recoverable eye. That
    /// holds only while its matrix is not recognised: this title's race frame renders a
    /// **1024x1024** light pass whose transform IS recoverable, and it beats the
    /// 960x544 player view on area every frame. The reported eye then jumps between the
    /// light and the player from frame to frame - a controller steering on it is being
    /// handed two different cameras and drives into a wall.
    ///
    /// So the first test is STRUCTURAL rather than a size heuristic: **the player's view
    /// is the pass the final composite SAMPLES.** That is what a composite is for, and
    /// it is a fact about this frame rather than a guess about relative sizes - a shadow
    /// map is sampled by the world pass, never by the composite.
    pub fn world_scene(&self) -> Option<&Scene> {
        let frame = self.frame_scenes();
        // Every texture the final pass reads, by guest address.
        let composited: std::collections::BTreeSet<u32> = frame
            .last()
            .map(|s| s.draws.iter().flat_map(|d| d.textures.iter()).map(|t| t.data_addr).collect())
            .unwrap_or_default();
        frame
            .iter()
            .filter(|s| crate::render::scene_eye(s).is_some())
            .max_by_key(|s| {
                let area = s.color.as_ref().map_or(0u64, |c| c.width as u64 * c.height as u64);
                let sampled = s
                    .color
                    .as_ref()
                    .is_some_and(|c| composited.contains(&c.data_addr));
                (sampled, area, s.world_triangles())
            })
            .or_else(|| {
                frame
                    .iter()
                    .filter(|s| s.world_triangles() > 0)
                    .max_by_key(|s| s.world_triangles())
            })
            .or_else(|| frame.last())
    }

    /// Scenes the run has completed, including any evicted by `scene_limit`.
    pub fn total_scenes(&self) -> u64 {
        self.retired_scenes + self.scenes.len() as u64
    }

    /// Record one serviced call in the bounded debug trace (hot path: a push, plus
    /// an amortized front-drain every [`TRACE_CAP`]/2 calls once the cap is hit).
    #[inline]
    pub fn record_call(&mut self, func_nid: u32, thid: i32) {
        self.call_count += 1;
        if self.trace.len() >= TRACE_CAP {
            self.trace.drain(..TRACE_CAP / 2);
            self.trace_thid.drain(..TRACE_CAP / 2);
        }
        self.trace.push(func_nid);
        self.trace_thid.push(thid);
    }

    /// Note an unimplemented call once (deduplicated by NID pair).
    pub fn note_unimplemented(&mut self, library_nid: u32, func_nid: u32, name: &str) {
        if !self.unimplemented.iter().any(|(l, f, _)| *l == library_nid && *f == func_nid) {
            self.unimplemented.push((library_nid, func_nid, name.to_string()));
        }
    }
}

#[cfg(test)]
mod fold_tests {
    use super::*;

    fn digest(bytes: &[u8]) -> u64 {
        let mut h = FNV_OFFSET;
        fnv_bulk(&mut h, bytes);
        h
    }

    /// Flipping ANY single bit of the blob changes the digest. The lanes are what make
    /// this worth asserting: a byte only ever reaches one lane, so a lane that was
    /// dropped or aliased would silently stop covering a whole eighth of every buffer -
    /// and a signature blind to an eighth of the vertex data still looks like a working
    /// signature.
    #[test]
    fn every_byte_is_covered() {
        // Longer than one lane group and NOT a multiple of it, so the tail path is
        // exercised too.
        let base: Vec<u8> = (0..67u8).collect();
        let want = digest(&base);
        for i in 0..base.len() {
            for bit in 0..8 {
                let mut v = base.clone();
                v[i] ^= 1 << bit;
                assert_ne!(digest(&v), want, "byte {i} bit {bit} did not affect the digest");
            }
        }
    }

    /// Order matters: swapping two bytes that land in DIFFERENT lanes, and two that
    /// land in the SAME lane, must both change the digest. A per-lane sum would pass
    /// the first and fail the second.
    #[test]
    fn order_matters_within_and_across_lanes() {
        let base: Vec<u8> = (0..64u8).collect();
        let want = digest(&base);
        let mut across = base.clone();
        across.swap(0, 1);
        assert_ne!(digest(&across), want, "swap across lanes");
        let mut within = base.clone();
        within.swap(0, FNV_LANES);
        assert_ne!(digest(&within), want, "swap within one lane");
    }

    /// Length is folded in, so appending zeros - which leave every lane's xor-multiply
    /// chain looking plausible - cannot leave the digest unchanged, and two blobs that
    /// concatenate to the same bytes do not collide.
    #[test]
    fn length_is_part_of_the_digest() {
        assert_ne!(digest(&[1, 2, 3]), digest(&[1, 2, 3, 0]));
        let mut a = FNV_OFFSET;
        fnv_bulk(&mut a, &[1, 2]);
        fnv_bulk(&mut a, &[3]);
        let mut b = FNV_OFFSET;
        fnv_bulk(&mut b, &[1]);
        fnv_bulk(&mut b, &[2, 3]);
        assert_ne!(a, b, "the split between two folds must be visible");
    }

    /// An empty blob is still folded (its length), so a draw with no indices is
    /// distinguishable from one with none of that field at all.
    #[test]
    fn empty_is_folded() {
        assert_ne!(digest(&[]), FNV_OFFSET);
    }
}

#[cfg(test)]
mod extent_tests {
    use super::*;

    /// A one-draw scene whose colour surface is `w x h` and whose draw sets a
    /// `vw x vh` viewport (`enable` is GXM's, where 0 is ENABLED).
    fn scene(w: u32, h: u32, vw: f32, vh: f32, enable: u32) -> Scene {
        let mut render_state = RenderState::default();
        render_state.viewport = [vw / 2.0, vw / 2.0, vh / 2.0, -vh / 2.0, 0.5, 0.5];
        render_state.viewport_enable = enable;
        let draw = Draw {
            fragment_program_header: 0,
            vertex_textures: Vec::new(),
            primitive: 0,
            index_format: 0,
            index_count: 3,
            vertices: Arc::from(&[][..]),
            vertex_stride: 0,
            attributes: Vec::new(),
            indices: Arc::from(&[][..]),
            uniforms: Vec::new(),
            textures: Arc::from(&[][..]),
            render_state,
            blend: BlendState::default(),
            exposure: 1.0,
            material: FragmentMaterial::default(),
            world: [0.0; 16],
            vprog: no_program(),
            fprog: no_program(),
            vert_sa: Vec::new(),
            frag_sa: Vec::new(),
            frag_sa_addr: 0,
            shader_expanded: false,
        };
        Scene {
            color: Some(ColorSurface {
                format: 0,
                surface_type: 0,
                width: w,
                height: h,
                stride_pixels: w,
                data_addr: 0x8900_0000,
                scale_mode: 0,
                gamma: 0,
            }),
            depth: None,
            multisample: 0,
            draws: vec![draw],
        }
    }

    /// The case this exists for: a real pass through a colour surface a title left at
    /// 1x1. Believing the surface renders the pass into one pixel and loses it.
    #[test]
    fn a_degenerate_surface_takes_the_extent_from_the_viewport() {
        let mut s = scene(1, 1, 960.0, 544.0, 0);
        s.adopt_viewport_extent();
        let c = s.color.unwrap();
        assert_eq!((c.width, c.height), (960, 544));
    }

    /// An honest surface is never overridden, whatever the viewport says - a title may
    /// legitimately draw through a viewport smaller than its target.
    #[test]
    fn an_honest_surface_extent_is_left_alone() {
        let mut s = scene(64, 256, 960.0, 544.0, 0);
        s.adopt_viewport_extent();
        let c = s.color.unwrap();
        assert_eq!((c.width, c.height), (64, 256));
    }

    /// With the viewport DISABLED the transform is not the viewport's, so it says
    /// nothing about the extent and must not be adopted.
    #[test]
    fn a_disabled_viewport_is_not_adopted() {
        let mut s = scene(1, 1, 960.0, 544.0, 1);
        s.adopt_viewport_extent();
        let c = s.color.unwrap();
        assert_eq!((c.width, c.height), (1, 1));
    }
}

#[cfg(test)]
mod retention_tests {
    use super::*;

    /// A scene distinguishable by `tag` through the part of it the signature folds.
    fn scene(tag: u8) -> Scene {
        Scene {
            color: Some(ColorSurface {
                format: tag as u32,
                surface_type: 0,
                width: 960,
                height: 544,
                stride_pixels: 960,
                data_addr: 0x8000_0000 + tag as u32,
                scale_mode: 0,
                gamma: 0,
            }),
            depth: None,
            multisample: 0,
            draws: Vec::new(),
        }
    }

    /// Turning the fold off must never produce a plausible-looking wrong signature.
    ///
    /// # The whole risk of the optimisation is here
    /// Skipping the fold saves ~8% of a race frame on a live player session, which never asks
    /// for the number. The danger is not the saving, it is that a run which skipped scenes could
    /// still hand back a hash - one that looks exactly like a real signature, compares unequal to
    /// every other run, and records nothing about why. So the skip is remembered and the getter
    /// refuses. A caller that leaves it alone is unaffected, which is what `Default` gives.
    #[test]
    fn a_run_that_skipped_the_fold_refuses_to_produce_a_signature() {
        let mut folded = Capture::new();
        let mut skipped = Capture::new();
        folded.scene_limit = Some(2);
        skipped.scene_limit = Some(2);
        skipped.set_signature_wanted(false);
        for i in 0..12u8 {
            folded.push_scene(scene(i));
            skipped.push_scene(scene(i));
            folded.end_frame();
            skipped.end_frame();
        }
        assert_ne!(folded.signature(), u64::MAX, "a folding run gives a real signature");
        assert_eq!(
            skipped.signature(),
            u64::MAX,
            "a run that retired scenes unfolded must refuse, not guess"
        );
        // And the refusal is about SKIPPED scenes, not about the flag: a run that turned the
        // fold off but never evicted anything has folded everything it has, so it can answer.
        let mut untouched = Capture::new();
        untouched.set_signature_wanted(false);
        for i in 0..3u8 {
            untouched.push_scene(scene(i));
            untouched.end_frame();
        }
        assert_ne!(untouched.signature(), u64::MAX, "nothing was dropped, so nothing is missing");
    }

    /// Folding must be ON unless a caller explicitly says otherwise - the safe state has to be
    /// the one you get by forgetting, and `Capture` is `#[derive(Default)]`.
    #[test]
    fn the_fold_defaults_to_on() {
        for mut c in [Capture::new(), Capture::default()] {
            c.scene_limit = Some(1);
            for i in 0..6u8 {
                c.push_scene(scene(i));
                c.end_frame();
            }
            assert_ne!(c.signature(), u64::MAX, "a default Capture must fold");
        }
    }

    /// Bounding memory must not change what the run reports about itself. If the
    /// signature moved with the retention window, a long run and a short one would
    /// compare unequal and the cross-engine equivalence check - the only one there
    /// is - would be worthless.
    #[test]
    fn the_signature_is_invariant_under_scene_retention() {
        let mut unbounded = Capture::new();
        let mut bounded = Capture::new();
        bounded.scene_limit = Some(3);
        for i in 0..40u8 {
            unbounded.push_scene(scene(i));
            bounded.push_scene(scene(i));
            // One scene per display frame, as a single-pass title produces. Retention
            // never evicts below a whole frame, so a run that never ends a frame is
            // deliberately not the case under test here (see
            // `retention_keeps_a_whole_frame`).
            unbounded.end_frame();
            bounded.end_frame();
        }
        for c in [&mut unbounded, &mut bounded] {
            c.egress.push(EgressEvent { frame: 7, kind: EgressKind::Trophy { id: 3 } });
        }
        assert_eq!(unbounded.scenes.len(), 40);
        assert_eq!(bounded.scenes.len(), 3, "retention did not bound the scene list");
        assert_eq!(bounded.total_scenes(), 40, "evicted scenes must still be counted");
        assert_eq!(
            unbounded.signature(),
            bounded.signature(),
            "the determinism signature changed when scenes were evicted"
        );
    }

    /// Retention keeps the LAST COMPLETED FRAME whole, even at `scene_limit = 1`.
    /// Without this a multi-pass title's `@shot` renders only the composite - a live
    /// HUD over black - and reports it as the picture.
    #[test]
    fn retention_keeps_a_whole_frame() {
        let mut c = Capture::new();
        c.scene_limit = Some(1);
        for frame in 0..5u8 {
            for pass in 0..6u8 {
                c.push_scene(scene(frame * 6 + pass));
            }
            c.end_frame();
        }
        assert_eq!(c.frame_scenes().len(), 6, "a shot of this frame would be missing passes");
        assert_eq!(c.total_scenes(), 30, "evicted scenes must still be counted");
    }

    /// A `Default`-constructed capture folds identically to a `new` one (the seed is
    /// applied lazily), so nothing depends on which constructor a host happened to use.
    #[test]
    fn default_and_new_captures_agree() {
        let mut a = Capture::new();
        let mut b = Capture::default();
        b.scene_limit = Some(1);
        for i in 0..5u8 {
            a.push_scene(scene(i));
            b.push_scene(scene(i));
        }
        assert_eq!(a.signature(), b.signature());
    }
}
