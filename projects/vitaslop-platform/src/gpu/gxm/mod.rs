use super::{
    alpha_single_color_format, float_color_format, gxm_sample_count, raw64_color_format, BlockFamily, BlockFormat, CompressedData, FLOAT_TARGET_FORMAT,
    CompressedUpload, DrawSpace, RenderScene, TexelSeam, depth_format, GXM_VERTEX_STRIDE,
    MSAA_SAMPLES, RAW64_FORMAT,
};
// >>> HASHED WITH FxHash, NOT SipHash. Every one of these is probed per DRAW - the
// pipeline cache, the sampler bind groups, the view cache, the packed-geometry memo -
// and a race frame submits hundreds of draws each naming several of them. See
// `crate::fasthash` for what that trade is and why the keys here make it a safe one.
use crate::fasthash::{FxHashMap as HashMap, FxHashSet as HashSet};
use wgpu::util::DeviceExt;

// The renderer is split across these files by concern; every one of them sees this module's
// items through `use super::*`, and this module sees theirs through the globs below.
mod build_pipeline;
mod buffers;
mod caches;
mod clip;
mod coverage;
mod encode;
mod fallback;
mod gxp_types;
mod inputs;
mod live;
mod live_bind;
mod live_prepare;
mod pipeline_state;
mod prepare;
mod renderer_chain;
mod renderer_misc;
mod renderer_new;
mod renderer_pass;
mod renderer_resources;
mod renderer_rtt;
mod renderer_warm;
mod repack;
mod reports;
mod resident;
mod rtt;
mod store_mirror;
mod textures;
mod timestamps;
mod uniforms;
mod uploads;
mod vertex;
#[cfg(test)]
mod tests;

use build_pipeline::*;
use buffers::*;
pub use caches::*;
use clip::*;
use coverage::*;
pub use encode::*;
pub(crate) use fallback::*;
pub use gxp_types::*;
use inputs::*;
pub(in crate::gpu) use pipeline_state::*;
pub use prepare::*;
use repack::*;
use reports::*;
use resident::*;
use rtt::*;
use store_mirror::*;
pub(crate) use textures::*;
pub use timestamps::*;
use uniforms::*;
pub use uploads::*;
pub use vertex::*;

/// The millisecond clock the wasm [`Stopwatch`] reads, installed by the frontend.
///
/// # Why this is a seam and not a dependency
/// `std::time::Instant` is not implemented on `wasm32-unknown-unknown` - constructing one
/// panics at runtime - so for a long time this renderer's phase split was structurally
/// zero in the browser, and a whole session's conclusion ("the browser cannot split
/// `encode` at all") was really a statement about `Instant`, not about the browser. The
/// browser HAS a monotonic clock: `performance.now()`, in a worker as much as on a page.
/// This crate cannot reach it - it has no `js-sys`, deliberately, because
/// `vitaslop-runtime` depends on it for the neutral seam types - so the frontend that
/// already holds a `web_sys::Performance` installs it here instead.
///
/// Unset, every wasm phase reads 0.0 exactly as before, and
/// [`wasm_clock_installed`] is what lets a reporter say so rather than publish zeros
/// that read as "this phase is free".
#[cfg(target_arch = "wasm32")]
static WASM_CLOCK: std::sync::OnceLock<fn() -> f64> = std::sync::OnceLock::new();

/// Install the browser's millisecond clock for the renderer's own phase timing. Idempotent;
/// a second call is ignored (the first installer wins, and both read the same clock).
#[cfg(target_arch = "wasm32")]
pub fn set_wasm_clock(f: fn() -> f64) {
    let _ = WASM_CLOCK.set(f);
}

/// Whether the wasm phase clock has been installed. Native always true (it has `Instant`).
pub fn wasm_clock_installed() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        WASM_CLOCK.get().is_some()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        true
    }
}

#[cfg(target_arch = "wasm32")]
fn wasm_now() -> f64 {
    WASM_CLOCK.get().map(|f| f()).unwrap_or(0.0)
}

/// Milliseconds on a monotonic clock, on EITHER target - `Instant` natively, and the clock
/// the frontend installed with [`set_wasm_clock`] in the browser (0.0 until one is).
///
/// Public because a phase timer is not only a renderer's business: the SCENE BUILD lives in
/// `vitaslop-runtime`, which has no `js_sys` of its own, and a build phase that can only be
/// timed natively is a build phase whose cost is unknown on the one engine that is the
/// product [[vitaslop-web-is-the-product-not-the-tool]].
#[cfg(target_arch = "wasm32")]
pub fn now_ms() -> f64 {
    wasm_now()
}

/// A CPU stopwatch, on both engines.
///
/// Native reads `Instant`. Wasm reads whatever clock the frontend installed with
/// [`set_wasm_clock`], and 0.0 until one is - see that function for why the browser is
/// not clockless and only looked it.
#[derive(Clone, Copy)]
struct Stopwatch {
    #[cfg(not(target_arch = "wasm32"))]
    start: std::time::Instant,
    #[cfg(target_arch = "wasm32")]
    start: f64,
}

impl Stopwatch {
    #[cfg(not(target_arch = "wasm32"))]
    fn start() -> Self {
        Stopwatch { start: std::time::Instant::now() }
    }
    #[cfg(target_arch = "wasm32")]
    fn start() -> Self {
        Stopwatch { start: wasm_now() }
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
    #[cfg(target_arch = "wasm32")]
    fn ms(&self) -> f64 {
        wasm_now() - self.start
    }
}

/// The general fragment/vertex shader: the blob-free equivalent of a title's GXP
/// shaders, a faithful per-fragment twin of `vitaslop_runtime::render`. The vertex
/// stage projects per the draw's coordinate space (`u.mode`). There are two fragment
/// entry points, selected by pipeline (never by a per-fragment branch), matching the
/// software rasterizer's two combine modes:
///   - `fs_blend` (2D overlay): `vertex_color * texel` modulate. An untextured draw
///     binds a 1x1 white texel, so `color * white == color`.
///   - `fs_opaque` (3D opaque): the albedo texel is taken straight (its vertex color
///     is a non-color mask the real fragment program consumes, so modulating by it
///     would tint whole surfaces); an untextured opaque draw falls back to the vertex
///     color. Then scene `exposure` scales it and a Reinhard curve rolls off the
///     bright end, and alpha is forced to 1. `exposure == 1.0` skips the curve.
const GXM_SHADER: &str = r#"
struct U {
    mvp: mat4x4<f32>,
    mode: u32,
    surf_w: f32,
    surf_h: f32,
    textured: u32,
    exposure: f32,
    depth_min: f32,
    depth_scale: f32,
    pad2: u32,
    tint: vec4<f32>,
    light_dir: vec4<f32>,
    light_col: vec4<f32>,
    ambient: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) wnormal: vec3<f32>,
};

@vertex
fn vs(@location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) color: vec4<f32>, @location(3) wnormal: vec3<f32>) -> VsOut {
    var out: VsOut;
    var clip: vec4<f32>;
    if (u.mode == 0u) {
        // Mvp: object space through the captured MVP. Depth is the projected view distance
        // `w`, NOT the clip `z` - GXM/PowerVR resolves visibility from `w`, and this title's
        // vertex programs emit a clip `z` whose post-divide value depends only on the screen
        // position (see `render::project`, which measured it identical across the ground and
        // every car draw covering one pixel). The software rasterizer stores `-1/w`, which is
        // screen-linear and increases with distance; the CPU scene builder measures that
        // quantity's visible range so it can be mapped LINEARLY onto the WebGPU depth buffer's
        // [0,1] without changing a single comparison - identical ordering to the oracle, at
        // full f32 resolution across the range. X and Y pass through.
        clip = u.mvp * vec4<f32>(position, 1.0);
        let depth = -1.0 / clip.w;
        let nz = clamp((depth - u.depth_min) * u.depth_scale, 0.0, 1.0);
        clip.z = nz * clip.w;
    } else if (u.mode == 1u) {
        // Ndc: clip coords emitted directly.
        clip = vec4<f32>(position.x, position.y, 0.0, 1.0);
    } else {
        // Pixel: screen pixels, Y down -> clip. Matches the software rasterizer,
        // which passes pixel coords straight to the viewport transform.
        let x = position.x / u.surf_w * 2.0 - 1.0;
        let y = 1.0 - position.y / u.surf_h * 2.0;
        clip = vec4<f32>(x, y, 0.0, 1.0);
    }
    out.pos = clip;
    out.uv = uv;
    out.color = color;
    out.wnormal = wnormal;
    return out;
}

@fragment
fn fs_blend(in: VsOut) -> @location(0) vec4<f32> {
    return in.color * textureSample(tex, samp, in.uv);
}

@fragment
fn fs_opaque(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    // Alpha-test opaque decal/livery layers (a BC2/BC3 albedo with a coverage alpha): discard
    // the transparent texels so the body panel behind shows, not the sheet's black background.
    // `discard` also skips the depth write, matching the software rasterizer's early `continue`.
    // Safe for the ordinary opaque BC1 albedo and the untextured white fallback (alpha == 1).
    if (texel.a < 0.5) {
        discard;
    }
    // The forward-lit material, mirroring the software rasterizer's `shade_lit` exactly:
    // albedo (the texel, or the vertex color when untextured) * per-material tint, lit by
    // one directional light (saturate(N.L) * light_col) plus a flat ambient, then scaled by
    // scene exposure and Reinhard tone-mapped so HDR light rolls off instead of clipping.
    let albedo = select(in.color.rgb, texel.rgb, u.textured != 0u);
    let n = normalize(in.wnormal);
    let l = normalize(u.light_dir.xyz);
    let ndotl = max(dot(n, l), 0.0);
    let light = u.ambient.rgb + u.light_col.rgb * ndotl;
    let lit = albedo * u.tint.rgb * light * u.exposure;
    let rgb = lit / (vec3<f32>(1.0) + lit);
    return vec4<f32>(rgb, 1.0);
}
"#;

/// The supersample resolve shader: a fullscreen triangle whose fragment box-averages the
/// `scale x scale` block of the offscreen colour target that maps to each output pixel -
/// the exact integer box `Framebuffer::downsampled` applies, via `textureLoad` (no filtering)
/// so the GPU and software AA'd frames agree. `scale` comes from a tiny uniform.
const RESOLVE_SHADER: &str = r#"
struct RU { scale: u32 };
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> ru: RU;

@vertex
fn vres(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fres(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let s = i32(ru.scale);
    let base = vec2<i32>(i32(pos.x) * s, i32(pos.y) * s);
    var acc = vec4<f32>(0.0);
    for (var dy = 0; dy < s; dy = dy + 1) {
        for (var dx = 0; dx < s; dx = dx + 1) {
            acc = acc + textureLoad(srcTex, base + vec2<i32>(dx, dy), 0);
        }
    }
    return acc / f32(s * s);
}
"#;

/// Convert one render target's DEPTH attachment into the value the GUEST's depth buffer
/// would hold, so a later pass that samples that buffer reads a distance rather than a
/// colour.
///
/// The recompiled vertex stage does not write the guest's depth: `gxp_clipfix` replaces it
/// with `clamp((-1/w - min) * scale, 0, 1)`, a monotonic remap onto the scene's visible
/// range that keeps the depth TEST precise (see `RenderScene::depth_min`). That remap is
/// affine in `-1/w` and therefore invertible, which is what makes this a small fullscreen
/// pass instead of a second colour attachment on every draw of the pass: recover `w`, then
/// re-encode it the way the guest expects.
///
/// `mode` selects that encoding - see `GxmDepthEncoding`. It is a knob because the value a
/// GXM depth surface holds is not something any clean source we hold states outright, and
/// the honest thing is to make the choice visible and measurable rather than to bake in a
/// guess.
const GXM_DEPTH_SHADER: &str = r#"
struct DU { depth_min: f32, depth_scale: f32, mode: u32, konst: f32, fit_a: f32, fit_c: f32, pad0: f32, pad1: f32 };
@group(0) @binding(0) var srcDepth: texture_depth_2d;
@group(0) @binding(1) var<uniform> du: DU;
// >>> A SAMPLER, BECAUSE `textureLoad` IS REFUSED ON DEPTH IN COMPATIBILITY MODE.
// `textureSampleLevel` with a NON-FILTERING sampler is legal in both regimes and is exact:
// `pos.xy` is already at pixel centres, so dividing by the dimensions lands on the texel centre
// and nearest sampling returns that texel unchanged. One path serves both regimes, which is
// worth more here than saving an instruction - the alternative was a device where every pass
// that reads depth failed to build its pipeline at all.
@group(0) @binding(2) var srcSamp: sampler;

@vertex
fn vdep(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fdep(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // mode 5 is a CAUSALITY test, not an encoding: every texel becomes one known value, so a
    // pass that reads this depth either responds to it or does not. Reading a value only says
    // what a shader was fed; substituting one says whether that input is what decides the
    // picture. It reports itself, like every other substitution here.
    if (du.mode == 5u) {
        return vec4<f32>(du.konst, 0.0, 0.0, 1.0);
    }
    let dims = vec2<f32>(textureDimensions(srcDepth));
    let d = textureSampleLevel(srcDepth, srcSamp, pos.xy / dims, 0);
    // mode 4, and the degenerate scale==0 scene, pass the stored depth straight through:
    // with no range there is no `w` to recover, and inventing one would be worse than
    // handing back what is actually in the buffer.
    if (du.mode == 4u || du.depth_scale == 0.0) {
        return vec4<f32>(d, 0.0, 0.0, 1.0);
    }
    // Undo the clip fixup: q is -1/w, exactly what `gxp_clipfix` mapped onto [0,1].
    let q = d / du.depth_scale + du.depth_min;
    let w = select(-1.0 / q, 0.0, q == 0.0);
    var out = du.fit_a + du.fit_c / w;   // mode 0: the guest's own window depth `a + c/w`
    if (du.mode == 1u) { out = w; }
    else if (du.mode == 2u) { out = -w; }
    else if (du.mode == 3u) { out = q; }                                 // -1/w
    else if (du.mode == 6u) { out = -q; }                                // 1/w
    if (w == 0.0) { out = 0.0; }
    return vec4<f32>(out, 0.0, 0.0, 1.0);
}
"#;

/// The sRGB twin of a linear colour format, when it has one.
///
/// A GXM colour surface with `sceGxmColorSurfaceSetGammaMode` set has its writes
/// sRGB-ENCODED by the ROP, and a texture with `sceGxmTextureSetGammaMode` has its reads
/// sRGB-DECODED by the sampler. Rendering through - and sampling through - an sRGB VIEW of
/// the same texture reproduces both exactly, including doing the encode AFTER blending,
/// which is where the hardware does it. Nothing else in the pipeline has to change: the
/// shader keeps writing linear values.
/// A pass's clear colour as the attachment must be told it, from the bytes the caller
/// wants STORED. A clear value is written as given - the sRGB encode a gamma-correct
/// view applies to shader output is applied to the clear as well - so a clear through an
/// sRGB view has to be handed the LINEAR value whose encoding is the wanted bytes.
///
/// MEASURED: the loading screen of a baseball title is one draw into a gamma-correct
/// 512x256 surface, and its backdrop, the frontend's (16, 16, 24), came back as (71, 71,
/// 86): the bytes 16/255 and 24/255 encoded as if they were linear. A gamma surface
/// showed a different backdrop from a linear one for the same clear.
fn clear_color(bytes: [u8; 4], srgb_view: bool) -> wgpu::Color {
    let ch = |b: u8| {
        let v = b as f64 / 255.0;
        if !srgb_view {
            v
        } else if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    wgpu::Color { r: ch(bytes[0]), g: ch(bytes[1]), b: ch(bytes[2]), a: bytes[3] as f64 / 255.0 }
}

fn srgb_twin(f: wgpu::TextureFormat) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as F;
    match f {
        F::Rgba8Unorm => Some(F::Rgba8UnormSrgb),
        F::Bgra8Unorm => Some(F::Bgra8UnormSrgb),
        // Already sRGB, or a format with no sRGB twin (a float target needs none - it is
        // not quantised, so there is nothing for a transfer function to buy).
        _ => None,
    }
}

/// The format the converted guest-depth texture is stored in.
///
/// A GXM `DF32` depth surface holds 32-bit floats, so `R32Float` is the exact match - but
/// `R32Float` is NOT filterable in WebGPU core, and the recompiled sampler layout declares
/// every unit filterable (it has to: it is one layout serving a shader's texture units,
/// and the same unit index carries ordinary colour textures on other pairs). Binding an
/// unfilterable format into a filterable slot is a validation error, not a soft failure.
///
/// `Rgba16Float` is filterable everywhere and carries ~11 bits of mantissa, which over a
/// view distance of a few hundred units is sub-unit precision - far finer than the soft
/// fades that read it resolve. The cost is stated rather than assumed: see
/// [`report_depth_conversion`], which names the encoding in every run.
const GXM_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Bytes of the depth-conversion uniform block (`DU` above).
const GXM_DEPTH_UNIFORM_BYTES: u64 = 32;

/// Which value a later pass reads out of a render target's depth
/// (`VITASLOP_GXM_DEPTH_ENC`), matching the `mode` branch in [`GXM_DEPTH_SHADER`].
/// Returns `(mode, constant)`; the constant is used only by mode 5.
fn gxm_depth_encoding() -> (u32, f32) {
    // Cached: asked per depth conversion.
    static V: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let v = V.get_or_init(|| std::env::var("VITASLOP_GXM_DEPTH_ENC").unwrap_or_default());
    match v.as_str() {
        // The default is no longer a guess: `fit` writes the guest's own window depth
        // `a + c/w`, with `a` and `c` measured from the pass's vertex programs. The rest
        // stay as A/B alternatives, because a title whose depth surface turns out to hold
        // something else should cost one run to find out, not a code change.
        "fit" | "" => (0, 0.0),
        "w" => (1, 0.0),
        "negw" => (2, 0.0),
        "negrecipw" => (3, 0.0),
        "recipw" => (6, 0.0),
        "unit" => (4, 0.0),
        other => match other.strip_prefix("const:").map(str::parse::<f32>) {
            Some(Ok(k)) => (5, k),
            _ => panic!(
                "VITASLOP_GXM_DEPTH_ENC={other} is not one of \
                 fit|w|negw|negrecipw|recipw|unit|const:<float> - refusing to guess which \
                 value the guest's depth buffer holds"
            ),
        },
    }
}

/// Bytes of the per-draw uniform block (matches the WGSL `U` struct): mat4 (64) +
/// mode/surf_w/surf_h/textured (16) + exposure/depth_min/depth_scale/pad2 (16) +
/// tint/light_dir/light_col/ambient (4 x vec4 = 64) = 160. Copies are laid into a
/// shared buffer at [`GxmRenderer::uniform_stride`] spacing for dynamic offsets.
const UNIFORM_BYTES: u64 = 160;

/// Cap on the sampler bind-group cache, in ENTRIES. An entry is a small descriptor that
/// only REFERENCES textures the byte-bounded view cache already owns, so unlike that
/// cache this one bounds nothing large - the count is a runaway guard, not a memory
/// budget. See `tex_cache_budget_bytes` for the case where a count was the wrong unit.
const SAMPLER_BG_CACHE_CAP: usize = 8192;

/// A `<hex-key>[,<hex-key>]|all` knob, resolved ONCE and cached.
///
/// # Why this is not read straight from the knob table at the call site
/// [`crate::knobs::var`] takes a MUTEX and, off the browser, falls through to
/// `std::env::var`, which allocates. The two diagnostics that use this shape
/// ([`report_inputs`] and [`report_inputs_order`]) are called PER DRAW, so reading the knob
/// at the call site cost a lock, an environment lookup, a `String` allocation and a
/// `split(',')` parse **per draw, per diagnostic, in every run** - about 2,500 of each per
/// frame on the user's device at 1264 draws, entirely to decide to print nothing.
///
/// A knob cannot change after start-up on either engine (the browser fills its override
/// table before the guest runs), so resolving once is not a behaviour change - it is the
/// same answer, computed once instead of a million times. Same pattern as
/// [`tex_cache_budget_bytes`]. **A diagnostic that is off must cost approximately nothing,
/// or the instrumented build stops being the build anyone measures.**
enum KeySpec {
    Off,
    All,
    Keys(HashSet<u64>),
    AllExcept(HashSet<u64>),
}

/// `VITASLOP_GXP_TEXCOORD_NUDGE=<eps>`: add `eps` (normalised units) to every 2-D
/// `textureSample` coordinate - see [`nudge_texcoords`].
fn texcoord_nudge() -> Option<f32> {
    use std::sync::OnceLock;
    static V: OnceLock<Option<f32>> = OnceLock::new();
    *V.get_or_init(|| crate::knobs::var("VITASLOP_GXP_TEXCOORD_NUDGE").ok()?.trim().parse().ok())
}

/// Rewrite every `textureSample(t, s, vec2<f32>(..)..)` so its coordinate is `gxp_nudge2(..)`:
/// the coordinate plus a tiny positive epsilon. A coordinate computed in f16 (the GXP's half
/// registers, emulated bit-exactly) lands EXACTLY on texel boundaries, and a point sample
/// there is a tie each GPU breaks its own way - measured on the PowerVR phone: a baseball title's boot
/// light-cell blur read the cells' black gutters and came out darker from identical input.
fn nudge_texcoords(wgsl: &str, eps: f32) -> String {
    const CALL: &str = "textureSample(";
    let mut out = String::with_capacity(wgsl.len() + 256);
    let mut rest = wgsl;
    while let Some(at) = rest.find(CALL) {
        out.push_str(&rest[..at + CALL.len()]);
        rest = &rest[at + CALL.len()..];
        // Split the call's top-level arguments.
        let bytes = rest.as_bytes();
        let (mut depth, mut commas, mut end) = (0i32, Vec::new(), None);
        for (i, &c) in bytes.iter().enumerate() {
            match c {
                b'(' => depth += 1,
                b')' if depth == 0 => {
                    end = Some(i);
                    break;
                }
                b')' => depth -= 1,
                b',' if depth == 0 => commas.push(i),
                _ => {}
            }
        }
        if let (Some(e), 2) = (end, commas.len()) {
            let coord = rest[commas[1] + 1..e].trim();
            if coord.starts_with("vec2<f32>") {
                out.push_str(&rest[..commas[1] + 1]);
                out.push_str(" gxp_nudge2(");
                out.push_str(coord);
                out.push(')');
                rest = &rest[e..];
            }
        }
    }
    out.push_str(rest);
    // APPENDED, not prepended: an `enable` directive must be the first thing in a module.
    format!("{out}
fn gxp_nudge2(c: vec2<f32>) -> vec2<f32> {{ return c + vec2<f32>({eps:e}); }}
")
}

fn gxp_return_keys() -> &'static KeySpec {
    use std::sync::OnceLock;
    static S: OnceLock<KeySpec> = OnceLock::new();
    S.get_or_init(|| KeySpec::resolve("VITASLOP_GXP_RETURN_KEYS"))
}

impl KeySpec {
    /// `all`, a key list, or a `!`-prefixed key list meaning "every pair EXCEPT these".
    ///
    /// The exclusion form is what makes a whole-frame instrument survive a COMPOSITE. A
    /// title that renders its world to an offscreen surface and then blits it with one
    /// fullscreen pair destroys any per-pair marking in that surface - the blit reports
    /// itself, not what it sampled - so `all` answers "the composite owns every pixel" and
    /// nothing else. Excluding the blit leaves it passing its input through untouched, and
    /// the marking underneath survives to the frame.
    fn resolve(name: &str) -> Self {
        let Ok(spec) = crate::knobs::var(name) else { return Self::Off };
        let spec = spec.trim();
        // A bare on-switch means every pair. `1` is spelled here rather than left to fall
        // through the hex parser because it would otherwise resolve to the single key
        // 0x0000000000000001 and mark nothing - an instrument that silently does nothing
        // when it is switched on the obvious way.
        if matches!(spec, "all" | "1" | "on" | "yes" | "") {
            return Self::All;
        }
        let (negated, list) = match spec.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, spec),
        };
        let keys: HashSet<u64> = list
            .split(',')
            .filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
            .collect();
        if negated { Self::AllExcept(keys) } else { Self::Keys(keys) }
    }

    #[inline]
    fn wants(&self, key: u64) -> bool {
        match self {
            Self::Off => false,
            Self::All => true,
            Self::Keys(k) => k.contains(&key),
            Self::AllExcept(k) => !k.contains(&key),
        }
    }
}

/// Whether the once-per-pair `gxp pair <key>: vprog hash ..., fprog hash ...` INDEX should be
/// printed: explicitly via `VITASLOP_GXP_PAIRS`, or implicitly because some diagnostic that
/// takes a pair KEY is set and the operator therefore needs to know which keys exist.
///
/// Gated because it is an index and not a finding - see the call site for the device capture
/// where it filled the panel and evicted 21 distinct real lines.
fn gxp_key_index_wanted() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_GXP_PAIRS").is_ok()
            || [
                "VITASLOP_GXP_INPUTS",
                "VITASLOP_GXP_INPUTS_ORDER",
                "VITASLOP_GXP_INPUTS_VERTS",
                "VITASLOP_GXP_INPUTS_DIR",
                "VITASLOP_GXP_QUADS",
                "VITASLOP_GXP_SA",
                "VITASLOP_GXP_KEYS",
                "VITASLOP_GXP_EXCLUDE",
                // A key-colour frame is USELESS without the key list: the colours are
                // computed from the keys, so reading one back means matching against
                // every key that was drawn (`boot24/keymatch.py` takes exactly this
                // index on stdin). Leaving it out cost a whole extra run.
                "VITASLOP_GXP_KEYCOLOR",
            ]
            .iter()
            .any(|k| crate::knobs::var(k).is_ok())
    })
}

/// Bytes of the recompiled path's per-draw `@group(3)` block (`GxpDepth` in the linker): three
/// vec4s of depth and viewport state, then four of per-unit texture LOD bias.
const GXP_DEPTH_BLOCK_BYTES: u64 = 48 + 64;

/// Cap on the depth-range bind-group cache, in ENTRIES - see [`GxpLive::depth_bgs`] for the
/// leak this exists to stop and the numbers that found it.
///
/// Unlike [`SAMPLER_BG_CACHE_CAP`], an entry here DOES own something: a small uniform
/// buffer and a bind group. They are individually tiny and collectively were not - the
/// count is what has to be bounded, because the bytes per entry are too small to trip a
/// byte budget long before the object count sinks the driver. A few hundred entries covers
/// every distinct depth range a frame actually uses many times over; the cap is here to
/// stop unbounded growth, not to be a working-set estimate.
const DEPTH_BG_CACHE_CAP: usize = 1024;

/// Clear `map` wholesale if it has reached `cap`, and say whether it did.
///
/// Split out of the one call site so the BOUND itself is testable without a GPU: the thing
/// that went wrong was not the bind group, it was that a cache holding GPU objects had no
/// upper limit at all, and that is a property of the map and the cap alone. A test that
/// needs an adapter is a test that does not run in CI, and this is exactly the kind of
/// slow unbounded growth no picture and no single-frame capture can show.
///
/// Only safe for caches whose KEY determines their VALUE, which is why it is not offered
/// as a general utility - see [`GxpLive::depth_bgs`].
///
/// The evicted VALUES are handed back rather than dropped, because an entry here owns a GPU
/// buffer and dropping one only makes it collectable on the engine that matters. The caller
/// decides when to `destroy()` them; see [`GxpLive::depth_retired`].
fn drain_if_at_cap<K, V>(map: &mut HashMap<K, V>, cap: usize, out: &mut Vec<V>) -> bool {
    if map.len() < cap {
        return false;
    }
    out.extend(map.drain().map(|(_, v)| v));
    true
}

/// Upper bound on the cross-frame texture caches. The recompiler's view cache evicts to it
/// per entry, oldest-first; the fixed-function one (unused in the shipped GXP-live
/// configuration, and reported at 0 MB there) still clears wholesale, which is a re-upload
/// and never incorrectness - the keys are content fingerprints, so a re-decoded atlas still
/// hits.
///
/// # In BYTES, because a count bounds nothing
/// This used to be a cap of 512 ENTRIES. An entry is a texture of any size, so 512 of
/// them is anywhere from a few megabytes to well over a gigabyte, and the cap fired at
/// the same place either way. Native never noticed - it has an address space to spare.
/// The browser did: a worker climbed 0.70 -> 1.81 GB while the emulator's own wasm heap
/// stayed FLAT at 487 MB, and was killed with no error, no crash event and nothing in
/// any log. A limit that does not track the resource it is limiting is not a limit.
///
/// # >>> THE FIGURE IS THE CONSOLE'S, NOT A TITLE'S
/// This was 256 MB, justified as "comfortably above this title's working set". It was
/// fitted to a MENU. A race on the same title thrashes at that budget - MEASURED on the
/// user's device at **83% of decodes being re-decodes of something just evicted**, and on
/// another title's campaign map at 225 textures re-decoded and 76 MB re-uploaded per frame
/// for `build 718 ms` of an 878 ms render, 1 fps. A budget fitted to one screen is a budget
/// that will be wrong on the next one.
///
/// The hardware has no counterpart to this cache at all: the Vita's GPU samples the guest's
/// own bytes in place, so there is no copy and no expansion to bound. What the hardware DOES
/// bound is how much texture a game can have resident, and that is title-independent - it is
/// the game's memory partitions (henkaku wiki, `Memory budget`):
///
/// | partition | size |
/// |---|---|
/// | `ScePhyMemPartGame` | 256 MiB |
/// | the "+109 MiB mode" extension, from the 125 MiB remaining pool | 109 MiB |
/// | `ScePhyMemPartGameCdram` | 112 MiB of the 128 MiB CDRAM (16 is the shell's) |
///
/// So **477 MiB is the most texture any title can have resident, ever**, because it is all
/// the memory a game can address. At that budget an upload the same size as the guest's own
/// bytes can NEVER be evicted while the guest still has it live, on any title, without
/// measuring one.
///
/// Going over it therefore means WE expanded - an RGBA8 decode at 4-8x on an adapter that
/// cannot take the guest's block format - which is our overhead and not the guest's demand.
/// That is the case this budget should catch, and `report_texture_working_set` says so out
/// loud. Eviction is per entry and never touches what the current frame has used, so a
/// working set past the budget degrades in proportion instead of collapsing.
///
/// Still far below what a browser worker can survive: the run that was killed climbed to
/// 1.81 GB, and it was on the ENTRY-count cap with no byte limit at all.
/// `VITASLOP_TEX_CACHE_MB` overrides.
//
// `GAME_RESIDENT_CEILING_MB` reads as UNUSED in a non-test build and is not: the child test
// module below reaches it as `super::GAME_RESIDENT_CEILING_MB`, which resolves through this
// `use`. Dropping it on the warning's word breaks `cargo test` while `cargo build` stays
// green [[lowering-a-warning-is-silencing]] - in reverse.
#[allow(unused_imports)]
use super::{tex_cache_budget_bytes, tex_retain_budget_bytes, GAME_RESIDENT_CEILING_MB};

/// Bound on `GxpLive::precompile_seen`, a set of ALLOCATION pairs already considered for
/// precompilation. Far above any title's program count; clearing costs one re-scan.
/// Distinct frame SHAPES and scene TALLIES that will be reported before the two
/// diagnostics go quiet. Each is a finding; a title producing more than this many distinct
/// ones is not being diagnosed by them, it is being buried by them - and the volume is
/// itself measurable in the frame (see `GxmRenderer::chain_shapes_seen`).
///
/// >>> SIX WAS TOO FEW, AND IT HID THE FRAME THE DIAGNOSTIC EXISTS FOR. One title's flat
/// frame - a whole screen of clear colour with a single live tile in it - has a shape all
/// of its own, and the cap was reached long before it happened, so the one line that would
/// have named it was never printed. The cap is here to stop a title that changes structure
/// every frame from burying the log; a title with a few dozen shapes is not that, and the
/// number has to leave room for the RARE shape, which is exactly the interesting one.
const CHAIN_SHAPE_CAP: usize = 64;
const GXP_SUMMARY_CAP: usize = 64;
const PRECOMPILE_SEEN_CAP: usize = 16384;

/// How long one frame may spend preparing shader pairs AHEAD of any draw. A budget in
/// MILLISECONDS rather than a pair count, because what a pair costs to compile is a
/// property of the device's driver and this project's target is a phone: a count tuned on
/// a desktop would be an order of magnitude too large there. See the loop in
/// [`gxm::GxmRenderer::precompile_pairs`].
const PRECOMPILE_MS_PER_FRAME: f64 = 6.0;

/// Upper bound on the remembered-evicted view keys, in entries. A key pair, so this is a
/// fraction of a megabyte, and it is a DIAGNOSTIC bound: past it the thrash count
/// under-reports rather than the set growing with the run.
const VIEW_EVICTED_KEYS_CAP: usize = 1 << 16;

/// Upper bound on the memoized shader-pair keys, in distinct pairs. A title has a few
/// hundred shader pairs in total, so this never fires in practice; it exists so a title
/// that rebuilds programs every frame cannot grow the map without limit.
const PAIR_KEY_CACHE_CAP: usize = 4096;

/// What identifies a shader pair for [`GxpLive::pair_key`]: the two program blobs BY
/// ALLOCATION - address and length - plus the fixed-function state baked into the pipeline
/// beside them, which the published key also covers.
///
/// A raw pointer is a legitimate identity here only because the cache entry holds an `Arc`
/// clone of each blob, so the allocation outlives the entry and no later program can be
/// given the same address. Without that clone this would be a use-after-free waiting to
/// hand one shader another shader's key.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PairIdentity {
    vptr: usize,
    vlen: usize,
    fptr: usize,
    flen: usize,
    blend: [u8; 7],
    depth_write: bool,
    depth_func: u32,
    depth_bias: (i32, i32),
    /// Baked into the pipeline as an empty colour write mask, so it must identify the pair
    /// - one shader used both enabled and disabled is two pipelines, not one.
    fragment_program_enabled: bool,
    /// The draw takes the dual-source pipeline - a different module and blend, so a
    /// different pipeline. See `draw_dual_source`.
    dual_source: bool,
}

/// Which texture bind group a draw uses. `White` is the shared 1x1 opaque-white
/// fallback for an untextured draw; `Tex` names a cached upload by content key and
/// whether it is sampled LINEAR (so the two filter modes cache separately).
#[derive(Clone, Copy, PartialEq, Eq)]
enum BindKey {
    White,
    Tex(u64, bool),
    /// A texture whose pixels this frame's earlier pass rendered, named by the guest
    /// address of that pass's colour surface rather than by content - the guest bytes
    /// at that address are stale, so a content key would name the wrong image. The
    /// last field selects the pre-pass snapshot over the live target.
    Rtt(u32, bool, bool),
}

/// One draw resolved against the per-frame arena: byte ranges into the shared
/// vertex/index buffers, the dynamic uniform offset, which pipeline, and which
/// texture bind group. No owned GPU resources - everything lives in the renderer's
/// persistent buffers and caches, so recording a frame allocates nothing.
struct Item {
    v_off: u64,
    v_len: u64,
    i_off: u64,
    i_len: u64,
    index_count: u32,
    uniform_offset: u32,
    opaque: bool,
    bind: BindKey,
    /// A masked draw's stencil VARIANT key (see `GxmRenderer::fixed_stencil`), `None` for
    /// the default ALWAYS/KEEP state the eager pipelines already carry.
    stencil: Option<u64>,
    /// The stencil reference a masked draw sets on the pass.
    stencil_ref: u32,
}

/// The general renderer: the GPU twin of the software rasterizer, built for a real
/// title's per-frame draw volume (tens to thousands of draws).
///
/// Performance shape - nothing is allocated per draw in steady state:
///   - Two pipelines (opaque depth-tested / alpha-blended) share one shader and
///     vertex layout; the fragment entry point is baked into the pipeline.
///   - Per-frame vertex, index and uniform data are packed into three grow-only
///     arena buffers uploaded in one `write_buffer` each - not a buffer per draw.
///   - Per-draw uniforms are addressed by DYNAMIC OFFSET into the one uniform
///     buffer (group 0), so there is a single uniform bind group for the whole frame.
///   - Texture bind groups (group 1) are cached across frames by content key + filter,
///     so an unchanged atlas is bound, never re-created; decoded uploads are cached by
///     content key. A `Nearest` and a `Linear` REPEAT sampler are both prebuilt.
/// One pass ordinal's recompiled-path arenas, held across frames. See
/// [`GxmRenderer::gxp_arenas`] for why there is a slot per pass rather than one buffer.
struct GxpArenaSlot {
    vbo: wgpu::Buffer,
    ibo: wgpu::Buffer,
    ubo: wgpu::Buffer,
    vcap: u64,
    icap: u64,
    ucap: u64,
    /// Changed whenever `ubo` is RE-created (a grow). The bind groups over it name that
    /// specific buffer, so they have to be rebuilt when it changes and - the whole point
    /// of pooling - must NOT be rebuilt when it does not. Drawn from one RUN-WIDE counter
    /// ([`next_arena_generation`]), never counted per slot: a slot the trim dropped and a
    /// later pass re-created must not reuse a generation its old bind groups were cached
    /// under, or they would bind a destroyed buffer.
    generation: u64,
    /// The chain frame that last used this slot - see the trim in `encode_chain`.
    last_used: u64,
}

/// What a frame whose last scene is not a display buffer shows - see
/// [`GxmRenderer::display_choice`].
#[derive(Clone, Copy, Debug)]
enum DisplayChoice {
    /// This display buffer's scene (the last one drawing into a flipped address).
    Scene(u32),
    /// Nothing drew a display buffer: this flipped buffer's kept image.
    Hold(u32),
}

/// See [`GxpArenaSlot::generation`].
fn next_arena_generation() -> u64 {
    static GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// A pass-arena slot unused this many chain frames is dropped if it is at the TAIL.
const ARENA_IDLE_FRAMES: u64 = 600;

/// Sub-rectangle copy textures kept for reuse across frames - see `subrect_free`. A frame
/// makes a handful; the cap only has to cover one frame's worth plus shape changes.
const SUBRECT_FREE_CAP: usize = 32;

/// A kept finished DISPLAY IMAGE: the colour a display pass renders into, plus the private
/// depth that pass needs. Both are the GUEST surface's extent, which is why the depth
/// cannot be the caller's. See [`GxmRenderer::display_images`].
struct DisplayImage {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    /// The sRGB twin of `view`, for a display surface the guest put in GAMMA-CORRECT mode.
    ///
    /// The offscreen arm has carried this for every render target since gamma mode was
    /// implemented; the DISPLAY arm did not, and so silently rendered a gamma surface
    /// linear. See `report_gamma_display_surface` and the display arm of `encode_chain`.
    /// `None` when `color_format` has no sRGB twin, which is the one case where the mode
    /// cannot be honoured and the report says so.
    view_srgb: Option<wgpu::TextureView>,
    /// Whether the bytes in `tex` are sRGB-ENCODED, i.e. whether the pass that last drew
    /// this image rendered through `view_srgb`.
    ///
    /// >>> IT IS THE IMAGE'S OWN STATE, NOT THE CURRENT SCENE'S. A pass that SAMPLES this
    /// image is usually not the pass that WROTE it - an offscreen scene reading the
    /// previous frame's display buffer has its own target and its own gamma mode - so
    /// asking the scene in hand whether to decode gives the wrong answer exactly when the
    /// two disagree. Written by the display arm each time it renders into this image, and
    /// read by the sampler path.
    gamma: bool,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// The multisampled attachments a display pass rasterises into when the guest created
    /// the display buffer's render target MULTISAMPLED, resolved into `tex` as the pass
    /// ends - see `ensure_display_msaa`. Built on first use, at this image's extent.
    msaa: Option<MsaaAttachments>,
    /// A copy of `tex` taken just before a display pass that SAMPLES THIS SAME BUFFER - the
    /// display arm's twin of an offscreen target's `shadow` (see `snapshot_rtt`), with its
    /// sRGB view. Built on first use and copied into thereafter. See `snapshot_display`.
    shadow: Option<(wgpu::Texture, wgpu::TextureView, Option<wgpu::TextureView>)>,
}

/// One cube map assembled from the six render targets the guest drew its faces into.
/// See [`GxmRenderer::rtt_cubes`] for why this exists and why it outlives a frame.
struct CubeFromRenders {
    /// Six array layers, viewed below as a cube. WebGPU has no way to view six separate
    /// 2D textures as one cube, so the faces are COPIED here - `copy_texture_to_texture`
    /// on the frame's own encoder, between the pass that drew the last face and the pass
    /// that samples it, which is the only ordering that is correct.
    tex: wgpu::Texture,
    /// The `Cube` view a sampler binds. Built once with the texture.
    view: wgpu::TextureView,
    /// Face edge length in pixels. A refresh whose faces changed size rebuilds instead of
    /// copying, because `copy_texture_to_texture` requires matching extents.
    size: u32,
    /// The texture format the faces were rendered in - a copy also requires these to match.
    format: wgpu::TextureFormat,
    /// Whether those faces hold sRGB-ENCODED bytes, which is what decides whether `view`
    /// is the plain format or its sRGB twin. Part of the rebuild test because it changes
    /// the view, not just the contents.
    gamma: bool,
    /// The guest byte stride between consecutive face addresses, kept so a later frame can
    /// find the same six targets without re-deriving it from the render set. ZERO for a
    /// [`Self::stacked`] cube, whose six faces are one target.
    stride: u32,
    /// Byte offsets within a face of each SMALLER mip level the guest renders, level 1 first:
    /// level `m` of face `k` is the `size >> m` target at `base + k*stride + mips[m-1]`.
    /// Empty for a cube rendered at one level only (and for a stacked one).
    mips: Vec<u32>,
    /// The six faces were rendered into ONE `size x 6*size` target, face `k` in rows
    /// `k*size..(k+1)*size` - see `GxmRenderer::assemble_stacked_cube`.
    stacked: bool,
    /// A stacked cube's target HEIGHT, which can exceed `6*size` (the surface is taller than
    /// the six faces). Zero for six separate targets.
    stack_height: u32,
}

impl CubeFromRenders {
    /// The render targets this cube's faces come from, as `(address, (width, height))`.
    fn face_targets(&self, base: u32) -> Vec<(u32, (u32, u32))> {
        if self.stacked {
            vec![(base, (self.size, self.stack_height))]
        } else {
            (0..CUBE_FACES).map(|k| (base.wrapping_add(self.stride * k), (self.size, self.size))).collect()
        }
    }
}

pub struct GxmRenderer {
    /// Encode every scene as an OFFSCREEN target, treating none as the display. Set for the
    /// `sceGxmEndScene` completion of a small target (see `VitaState::complete_scene_now`),
    /// where the one scene handed in would otherwise be taken for the frame's display.
    offscreen_only: bool,
    opaque: wgpu::RenderPipeline,
    blend: wgpu::RenderPipeline,
    /// The same two built for [`MSAA_SAMPLES`], for a pass whose render target the guest
    /// created multisampled. A fixed-function draw shares the pass with the recompiled
    /// ones, so it needs a pipeline whose sample count matches the attachments.
    opaque_ms: wgpu::RenderPipeline,
    blend_ms: wgpu::RenderPipeline,
    /// The same two against the sRGB view of the target, for a pass whose colour surface
    /// the guest put in GAMMA-CORRECT mode. `None` when `color_format` has no sRGB twin.
    srgb: Option<(wgpu::RenderPipeline, wgpu::RenderPipeline)>,
    /// The same two against a FLOAT target ([`FLOAT_TARGET_FORMAT`]), single-sampled and
    /// multisampled - see `float_color_format`. A float surface's pass is not the default
    /// format and not its sRGB twin, and was given the sRGB pair: every fixed-function draw
    /// into one was refused by wgpu (824 validation errors in one run of a volleyball title).
    float: (wgpu::RenderPipeline, wgpu::RenderPipeline),
    float_ms: (wgpu::RenderPipeline, wgpu::RenderPipeline),
    /// The fixed-function shader and layout, kept so a MASKED draw's stencil variant can be
    /// built with exactly what the eager pipelines above were built with.
    fixed_shader: wgpu::ShaderModule,
    fixed_layout: wgpu::PipelineLayout,
    /// Fixed-function stencil variants by `(opaque, target format, samples, stencil key)`.
    /// A handful per title - a HUD's marking and testing states - built on first sight.
    fixed_stencil: HashMap<(bool, wgpu::TextureFormat, u32, u64), wgpu::RenderPipeline>,
    uniform_layout: wgpu::BindGroupLayout,
    texture_layout: wgpu::BindGroupLayout,
    sampler_point: wgpu::Sampler,
    sampler_linear: wgpu::Sampler,
    white_bind: wgpu::BindGroup,
    /// Decoded texture uploads (a view kept alive), keyed by content fingerprint.
    views: HashMap<u64, wgpu::TextureView>,
    /// Bytes currently held by `views`, against [`tex_cache_budget_bytes`].
    views_bytes: usize,
    /// Texture bind groups (view + chosen sampler), keyed by (content key, linear).
    tex_binds: HashMap<(u64, bool), wgpu::BindGroup>,
    /// Every texture handed to a draw of the pass being prepared, with its bytes - what a
    /// wholesale clear of the caches above has to KEEP (see `ensure_texture`). Reset at the
    /// start of each `encode_pass`.
    tex_pass_keys: HashMap<u64, usize>,
    /// Grow-only per-frame arenas + the uniform bind group over the uniform arena.
    vbo: Option<wgpu::Buffer>,
    ibo: Option<wgpu::Buffer>,
    ubo: Option<wgpu::Buffer>,
    ubo_bind: Option<wgpu::BindGroup>,
    vbo_cap: u64,
    ibo_cap: u64,
    ubo_cap: u64,
    /// The recompiled path's own arenas, the counterpart of `vbo`/`ibo` above. Separate
    /// because the two paths pack different vertex layouts.
    ///
    /// **One slot per PASS ORDINAL within a chain, reused across FRAMES.** The arenas are
    /// per PASS and not per frame, and that is not an accident: every pass of a chain
    /// records into ONE command encoder and the whole encoder is submitted at the end, so
    /// a shared buffer overwritten between passes would hand every pass the LAST pass's
    /// geometry. That is exactly what happened when this was first written - the race
    /// frame came out as sheets of shredded triangles. Pass ordinal `n` of every frame
    /// therefore gets its OWN buffers, and reuses them next frame, when the encoder that
    /// named them last has certainly been submitted (the invariant in `retired_buffers`).
    ///
    /// **Why this is a POOL and not a fresh allocation, which is what it used to be.**
    /// A fresh `create_buffer_init` is `mappedAtCreation` on the web backend, so each one
    /// allocates a renderer-side shared-memory staging region as well as a GPU buffer.
    /// Three per pass per frame is ~11,500 of them over 70 seconds of a menu that draws
    /// SEVEN triangleslists a frame, and the browser eventually refuses one: Chrome
    /// reports a failed mapped allocation as `createBuffer failed, size (1332) is too
    /// large for the implementation`, which wgpu unwraps into a panic that kills the run
    /// worker. A 1332-byte buffer is not too large for anything; the allocation that
    /// failed was the staging region, and the fix is to stop asking for a new one every
    /// frame rather than to make the failure survivable.
    gxp_arenas: Vec<GxpArenaSlot>,

    /// >>> THE ARENA UPLOAD'S STAGING BELT - the path that replaces `queue.write_buffer`
    /// >>> for bulk per-frame geometry, because on a real device that call BLOCKS.
    ///
    /// # The measurement
    /// On the user's phone (PowerVR D-series, Chrome Android), inside one gameplay frame:
    /// a **2,315 KB `queue.write_buffer` blocked for 208 ms**; a 4-byte write into the same
    /// buffer immediately after took **0.0 ms**; and **the same 2,315 KB again took 551.9
    /// ms**, with **zero submits in flight**. `report_write_stall_control` exists to read
    /// exactly that triple, and its verdict is unambiguous: a fast tiny control means the
    /// thread was running, and a SLOW repeat means the cost is the SIZE - a staging path
    /// charging per megabyte every time - not the destination and not queue drain. Frame
    /// totals reached `upload 233-334 ms` against a 58.8 ms period.
    ///
    /// This retires two older readings of the same symptom: "a buffer's first write costs
    /// thousands of times its second" (refuted - the repeat is slow, so it is not first
    /// touch) and "waiting on the queue to drain" (refuted - nothing was in flight).
    ///
    /// # Why a BELT and not a staging buffer per write
    /// Because this code has already been burned by the obvious version. See `gxp_arenas`
    /// above: asking for a fresh mapped allocation every frame made Chrome refuse one and
    /// report it, confusingly, against a 1,332-byte buffer - "the allocation that failed
    /// was the staging region, and the fix is to stop asking for a new one every frame".
    /// A belt keeps a POOL and hands chunks back after submit, which is that fix.
    ///
    /// `CHUNK` is sized above the largest write seen on the device so the common frame
    /// takes one chunk and allocates nothing; a larger write still works, it just gets its
    /// own buffer. `None` means the belt is switched off (`VITASLOP_GXM_STAGING=0`) and the
    /// arenas go back through `queue.write_buffer` - the A/B arm, from one build.
    gxp_staging: Option<wgpu::util::StagingBelt>,

    /// How many COPIES of each pass ordinal's arenas are kept, and which copy this frame
    /// writes. See the rotation site in `encode_chain` for the measurement that motivates
    /// it. `gxp_arenas` is indexed `ordinal * gxp_arena_ring + gxp_arena_phase`.
    gxp_arena_ring: usize,
    gxp_arena_phase: usize,
    /// Which pass ordinal of the CURRENT chain is being encoded - the index into
    /// `gxp_arenas`. Reset by `encode_chain`, bumped by each pass that has recompiled
    /// draws.
    gxp_arena_slot: usize,
    /// GPU timestamps at both ends of every pass, where the adapter offers them - see
    /// [`GpuTimestamps`]. `None` on an adapter without `timestamp-query`.
    ts: Option<GpuTimestamps>,
    /// `VITASLOP_GPU_BURN`'s compute work, built on first use - see [`Self::gpu_burn`].
    burn: Option<(u32, wgpu::ComputePipeline, wgpu::BindGroup)>,
    /// Per-draw occlusion counts, under `VITASLOP_GXM_DRAW_COVERAGE` - see [`DrawCoverage`].
    /// `None` unless the knob is set, and then it is set for the life of the renderer, so a
    /// run either measures every frame or none: a query set that comes and goes would make
    /// the query indices mean different draws from frame to frame.
    coverage: Option<DrawCoverage>,
    /// Per-draw uniform spacing: `UNIFORM_BYTES` rounded up to the device's
    /// `min_uniform_buffer_offset_alignment` (256 by default).
    uniform_stride: u64,
    /// That alignment itself, kept so `prepare` does not have to ask the device again.
    ///
    /// >>> `Device::limits()` IS A BOUNDARY CROSSING IN THE BROWSER. wgpu's WebGPU backend
    /// > > > answers it by reading the live `GPUSupportedLimits` object and building a whole
    /// > > > `wgt::Limits` from it - `map_wgt_limits` shows up by name in a V8 worker profile of
    /// > > > a race. It was being asked once per PASS, sixteen times a frame, for a constant the
    /// > > > device fixed when it was created.
    uniform_align: u64,
    /// The caller's target colour format, needed to size the supersampled offscreen
    /// colour target (its format must match the scene pipelines' target).
    color_format: wgpu::TextureFormat,
    /// Supersample factor (1 = off, the default; N renders the scene at N x the target
    /// dimensions into an offscreen buffer and box-downsamples it into the caller's view).
    /// See [`GxmRenderer::set_supersample`] and the `resolve_*` fields below.
    ss_scale: u32,
    /// Every `(draws, with_payload, prepared, fixed)` recompiler summary already reported.
    ///
    /// Was a single `Option`, on the reasoning that reporting "only when the tuple CHANGES"
    /// removes the spam. It does not: this fires per SCENE, a race frame has eleven of them
    /// with different draw counts, so the tuple changes on nearly every scene and the memo
    /// never matches. MEASURED: 41,775 of these lines in one browser race run - see
    /// [`GxmRenderer::chain_shapes_seen`] for what the pair of them cost the frame.
    gxp_summaries_seen: HashSet<(usize, usize, usize, usize)>,
    /// Whether the summary cap has been announced - see `chain_shapes_capped`.
    gxp_summaries_capped: bool,
    /// Whether the one-off "tallies are live and clean" line has been printed.
    ///
    /// The tally is FAILURE-ONLY (see its emission site), and a report that says nothing
    /// when all is well cannot be told apart from one that never ran
    /// [[report-that-only-fires-on-failure]]. This makes it say so exactly once.
    gxp_summary_all_clear: bool,
    /// The box-downsample resolve pipeline + its bind-group layout and scale uniform,
    /// used only when `ss_scale > 1`. Built once in [`GxmRenderer::new`].
    resolve_pipeline: wgpu::RenderPipeline,
    resolve_layout: wgpu::BindGroupLayout,
    resolve_scale_buf: wgpu::Buffer,
    /// The depth-conversion pipeline + layout + uniform ([`GXM_DEPTH_SHADER`]), used only
    /// for a target whose depth a later pass samples. Built once in [`GxmRenderer::new`];
    /// a frame that samples no depth never runs it.
    gxm_depth_pipe: wgpu::RenderPipeline,
    gxm_depth_layout: wgpu::BindGroupLayout,
    gxm_depth_uniform: wgpu::Buffer,
    /// The same conversion reading a MULTISAMPLED depth attachment (sample 0, by
    /// `textureLoad`) - see `GxmDepthTarget::multisampled`. `None` on a compatibility-mode
    /// adapter, which refuses `textureLoad` on depth; there a depth-sampled target keeps a
    /// single-sampled attachment instead.
    gxm_depth_ms: Option<(wgpu::BindGroupLayout, wgpu::RenderPipeline)>,
    /// The lazily-(re)created offscreen supersample target (colour + depth + resolve bind
    /// group), sized to `ss_scale * surf`. Rebuilt when the scale or target size changes.
    ss_target: Option<SsTarget>,
    /// Offscreen render targets, keyed by the guest address of the colour surface the
    /// pass that fills them writes AND by the extent it is rasterised at. Persistent
    /// across frames (a title reuses the same few targets every frame).
    ///
    /// >>> ONE GUEST ADDRESS CAN CARRY TWO LIVE RENDER TARGETS OF DIFFERENT EXTENTS IN THE
    /// >>> SAME FRAME, AND KEYING ON THE ADDRESS ALONE DESTROYED ONE OF THEM EVERY FRAME.
    ///
    /// This was `HashMap<u32, RttSurface>`, and `ensure_rtt` read a changed extent at a
    /// known address as "the guest resized its target" and REBUILT it - which discards the
    /// image. MEASURED on a retail baseball title, every frame:
    ///
    ///   colour 0x8e20b030 rasterised through render target 0x40000337 (256x256)  - 41 draws
    ///   colour 0x8e20b030 rasterised through render target 0x40000010 (512x1024) -  4 draws
    ///
    /// The 4-draw pass re-created the target and threw the 41 draws away, and
    /// `report_rtt_recreated` said so on exactly the three stadium-board addresses
    /// 0x8e55a090, 0x8e79c2e0 and 0x8ec4a8e0 - which is what those three BLACK boards with
    /// one pale slanted bar across them were. Two GXM render targets are two objects; they
    /// share a colour surface pointer and nothing else.
    ///
    /// The precedent this follows verbatim is [`Self::rtt_raw`] and
    /// [`Self::rtt_rendered_raw`] one screen down: those were keyed `(addr, width, height)`
    /// for exactly this reason (a crowd atlas aliased as one 64-bit surface and one 8-bit
    /// surface of another size), and their doc comments state it. One entry per SHAPE, so
    /// neither target recreates the other and each reader gets the one it declared.
    rtt: HashMap<(u32, u32, u32), RttSurface>,
    /// >>> THE FRAME EACH TARGET WAS LAST TOUCHED IN - see `reclaim_stale_rtt`.
    ///
    /// The comment above says "a title reuses the same few targets every frame", and on a
    /// steady screen it does. Across a SESSION it does not: the key is a guest address, an
    /// entry is only ever replaced by one at the same address, and nothing was ever removed.
    /// MEASURED on a 48,000-frame golf run in this repo: **`rtt targets 304`**, holding
    /// hundreds of megabytes of colour and depth attachments for screens the title left long
    /// ago, in a renderer process that had reached 1.53 GB. That is unbounded growth in GPU
    /// memory as a function of RUN LENGTH, which is the shape of every complaint about a
    /// long session on the target device.
    ///
    /// Keyed exactly as `rtt` is, because it names the same objects: stamping by address
    /// alone would have one of the two targets at a shared address keep the other alive for
    /// ever, and the reclamation could then never take either.
    rtt_used: HashMap<(u32, u32, u32), u64>,
    /// Addresses this frame binds as a TEXTURE at an extent that disagrees with the render
    /// target `rtt` still holds there - i.e. addresses the guest has recycled out from under
    /// a target. `encode_pass` refuses to offer those targets to the sampler, so the draw
    /// gets the guest's own texture. Rebuilt every frame by `encode_chain`; empty on any
    /// frame where the two never disagree, which is every frame of a title that does not
    /// recycle. See `report_rtt_extent_mismatch`.
    ///
    /// >>> DELIBERATELY STILL KEYED BY ADDRESS ALONE, unlike `rtt` beside it. The judgement
    /// this records is "the GUEST has freed whatever was at this address and put an
    /// ordinary texture there", which is a statement about the ADDRESS and not about any
    /// one target held at it - so it blocks every target there, which is the conservative
    /// direction (the draw falls through to the guest's own bytes, which is what the guest
    /// wrote). Now that an address may hold more than one target, the test that fills this
    /// is "the bound extent matches NONE of the targets held there" - see `encode_chain`.
    rtt_alias_block: HashSet<u32>,
    /// Addresses whose `rtt` targets are OLDER than the display image drawn there since.
    ///
    /// A title's display buffers are ordinary memory, and the same buffer can be drawn as
    /// an offscreen target on one frame (an early completion, or a frame whose scenes were
    /// taken with the next one's) and as the DISPLAY pass on every frame after. The display
    /// pass writes `display_images`, not `rtt`, so the offscreen copy stops updating - and the
    /// sampler path offers `rtt` FIRST, display images only filling gaps. A draw sampling
    /// that buffer was handed the frozen copy for the rest of the run. MEASURED on a golf
    /// title: its post pass copies the frame's display buffer into a 640x368 trail target
    /// every frame, and in the browser it read a copy hundreds of frames old - an old
    /// golfer over the swing, a black world behind the HUD. The memory holds what was
    /// written LAST, so a display pass marks the address here and `encode_pass` stops
    /// offering those targets; an offscreen pass into the address clears the mark.
    rtt_superseded: HashSet<u32>,
    /// Bumped whenever a target in `rtt` is CREATED or RE-created - which is the only event
    /// that can invalidate a `wgpu::TextureView` of one.
    ///
    /// # Why this exists: a bind group over a render target used to be rebuilt EVERY FRAME
    /// `make_sampler_bg` refused to cache any group naming a target this frame rendered, on
    /// the reading that "those views belong to textures the frame allocates". They do not.
    /// `rtt` is persistent and keyed by `(guest address, width, height)`, `ensure_rtt`
    /// rebuilds an entry only when its sample count or depth-readability changes (an
    /// extent change is a different TARGET, not a rebuild - see `rtt`), and even the snapshot
    /// texture (`RttSurface::shadow`) is created once and copied into thereafter - so every
    /// one of those views is the SAME view next frame. A bind group names views; it does not
    /// care what is inside them.
    ///
    /// MEASURED on a retail race, 600 frames: **64 sampler bind groups
    /// created per frame, forever**, against 40 pipelines and 89 textures for the whole
    /// window. That is a real GPU object per draw per frame in a steady state that never
    /// converges - the exact shape [[vitaslop-steady-state-can-be-the-defect]] names - and
    /// in the browser it is also a wasm/JS boundary crossing per draw, which is the cost
    /// [[vitaslop-browser-host-call-cost]] measures at 91% marshalling.
    ///
    /// Mixing this counter into the group's key keeps the cache exactly as correct as
    /// refusing to cache was: a target that IS rebuilt bumps it and every group naming any
    /// target is rebuilt once, which is the same wholesale invalidation the old code did
    /// every frame - just only when something actually changed.
    rtt_epoch: u64,
    /// Views of the targets already rendered in the CURRENT frame's chain, by the same
    /// `(address, width, height)` key `rtt` uses. This decides two things: which pass
    /// CLEARS a target first (the first one into it each frame) and which reads need a
    /// snapshot.
    ///
    /// >>> THE EXTENT IS IN THE KEY BECAUSE "THE FIRST PASS INTO THIS TARGET" IS A QUESTION
    /// >>> ABOUT A TARGET, NOT ABOUT AN ADDRESS. See `rtt` for the two-targets-one-address
    /// measurement: with the address alone, the 4-draw 512x1024 pass at 0x8e20b030 counted
    /// the 41-draw 256x256 pass as "already rendered here" and neither cleared nor loaded
    /// correctly. The flattening into the address-keyed map the SAMPLER path consumes
    /// happens once, in `encode_pass` - see `sample_views` there.
    ///
    /// **It no longer decides what a pass may SAMPLE - see `sample_views` in
    /// `encode_pass`, and read this before restoring the old rule.** This comment used to
    /// say a later pass substitutes a rendered target only when THIS frame drew it,
    /// "otherwise it would sample last frame's image, which is worse than the guest's own
    /// bytes because it looks plausible". That reasoning is REVERSED, on evidence:
    ///
    /// - A render target IS guest memory, and guest memory keeps what was last written
    ///   into it. Last frame's image is not a plausible fake - it is precisely what the
    ///   hardware returns. The guest bytes are the fake: the GPU wrote those pixels, the
    ///   guest never did, so they decode to black.
    /// - MEASURED on a retail title. At its title-to-menu transition the guest stops rendering
    ///   its background into `0x89204aa0` and starts BLURRING it, and the root of that
    ///   blur chain samples exactly that buffer. Under the old rule the frame went from
    ///   fully painted to black in one flip - 91% of pixels - while the guest was drawing
    ///   MORE (6 scenes / 11 draws to 8 scenes / 29). Binding the resident target restores
    ///   the menu backdrop on both engines and leaves the race frame BIT-IDENTICAL.
    ///
    /// The one real caveat the old rule was reaching for: our resident texture can diverge
    /// from the guest's bytes if the guest CPU-WRITES that buffer, since those writes land
    /// in guest memory and not in our texture. No title here has been seen doing it to a
    /// render target, and the failure it would cause is bounded and visible.
    rtt_rendered: HashMap<(u32, u32, u32), wgpu::TextureView>,
    /// Each render target's guest row pitch in pixels, recorded where the pass that renders
    /// it names its surface - see the sub-rectangle rule in `encode_pass`.
    rtt_pitch: HashMap<(u32, u32, u32), (u32, u32)>,
    /// The sub-rectangle copies this frame's passes bind (see `encode_pass`), kept alive
    /// until the next frame's encode, after the frame that used them was submitted.
    subrect_keep: Vec<wgpu::Texture>,
    /// Last frames' sub-rectangle copies, free for reuse by `(width, height, format)`.
    ///
    /// A copy used to be a NEW texture every frame, dropped at the next frame's head and
    /// left to the browser's collector: MEASURED in the desktop browser (a fighting title's
    /// GPU census `long3`) 3,748 `gxm-rtt-subrect` textures made over ~100 s, none destroyed,
    /// 680 / 106 MB of them live at once waiting on a GC - GPU memory the process could not
    /// get back on its own schedule. Reused here, and destroyed past `SUBRECT_FREE_CAP`.
    subrect_free: HashMap<(u32, u32, wgpu::TextureFormat), Vec<wgpu::Texture>>,
    /// Every offscreen target this renderer has rendered into on some EARLIER frame, by
    /// the same `(address, width, height)` key `rtt` uses - so the frame's first pass into
    /// it can LOAD what the last frame left instead of clearing.
    ///
    /// >>> A GXM TILER LOADS THE COLOUR SURFACE AT `sceGxmBeginScene`, so a target a title
    /// >>> ACCUMULATES INTO ACROSS FRAMES keeps its pixels on the console and cannot here.
    ///
    /// MEASURED on a baseball title, and it cost that title its whole ambient light. It
    /// keeps a 128x128 PROBE ATLAS - 8 columns x 16 rows of 16x8 cells, 128 slots - paints
    /// only TWO OR THREE cells a frame, and then reads one slot back on the CPU (the texel
    /// at `(8+16a, 1+8b)` for slot `n`, `a = n%8`, `b = n/8`) and decodes it as
    /// `10^(3b/255 - 1) - 0.1` into its ambient colour. Clearing the atlas every frame left
    /// every cell but this frame's two reading `(0,0,0,0)`, byte 0 decodes to EXACTLY 0.0,
    /// and the frame came back blown out.
    ///
    /// Unlike [`Self::rtt_rendered`] this is NOT cleared at the top of a frame. It is
    /// dropped for one TARGET when that target is RECREATED, because a new texture has no
    /// previous contents to load and loading an undefined one is worse than clearing.
    ///
    /// Keyed by the `rtt` key rather than by the address for the same reason
    /// `rtt_rendered` is: "has this target ever been drawn" is a question about a target,
    /// and two targets at one address answer it independently.
    rtt_ever_rendered: HashSet<(u32, u32, u32)>,
    /// The 64-bit colour surfaces (see [`raw64_color_format`]) this renderer has drawn
    /// into, by guest address, each an `Rg32Uint` attachment holding the fragments' RAW
    /// register pairs plus its own depth. Kept SEPARATE from [`Self::rtt`]: a title aliases
    /// one address as a 64-bit surface AND as an 8-bit one of another size (a crowd atlas
    /// rendered as F16F16F16F16 and mip-chained through a 512x1024 U8U8U8U8 view of the
    /// same bytes), and one map keyed by address would recreate the target every frame and
    /// hand each reader the other's image.
    rtt_raw: HashMap<(u32, u32, u32), RawSurface>,
    /// Every (colour format, samples, alpha-single) a pass into this target has been
    /// ENCODED with, by the scene's target (`None` = a scene with no colour target), recorded
    /// by `encode_pass`. It is how `warm_pipelines` knows the pass shape of a scene before
    /// the encode decides it: the decision is spread over the display, raw-64 and RTT arms
    /// of `encode_chain` and their side effects, and replaying it would be a second copy of
    /// that logic to keep in step. A target never encoded before is not predicted - its
    /// first pipelines build synchronously, as they always did, and count as UNWARMED.
    pass_shapes: HashMap<Option<(u32, u32, u32, u32)>, Vec<(wgpu::TextureFormat, u32, bool)>>,
    /// The same shapes by the target's GUEST COLOUR FORMAT alone - the fallback
    /// `warm_pipelines` predicts a never-encoded target from. MEASURED (a fighting title, desktop Chrome,
    /// 30c): the worst display gap left, 770 ms at the fight intro, was 16 pipelines built
    /// SYNCHRONOUSLY for a target at a new address - the title allocates a fresh surface of
    /// a format it has drawn before. A wrong guess costs one background compile nobody
    /// waits for; a right one moves those compiles off the GPU queue.
    pass_shapes_by_format: HashMap<u32, Vec<(wgpu::TextureFormat, u32, bool)>>,
    /// The pipeline that unpacks a RAW 64-bit surface into its `Rgba16Float` companion
    /// (`RawSurface::float_view`), built on first use. See `convert_raw_to_float`.
    raw_to_float: Option<(wgpu::RenderPipeline, wgpu::BindGroupLayout)>,
    /// The raw targets drawn THIS frame, by `(address, width, height)` - what a 64-bit
    /// texture declaring that address and extent binds (as `texture_2d<u32>`) instead of
    /// the guest's bytes. Cleared with [`Self::rtt_rendered`].
    rtt_rendered_raw: HashMap<(u32, u32, u32), wgpu::TextureView>,
    /// The finished DISPLAY image of each display buffer, by its guest colour address -
    /// held SEPARATELY from [`Self::rtt`] and never merged into it.
    ///
    /// A display pass is written into the caller's framebuffer, which we do not own and
    /// cannot sample, so the finished frame used to leave no image behind at all. A title
    /// that SNAPSHOTS each finished frame and re-blits it - the golf title does, through a
    /// fullscreen `texel * vertexColour` quad - then read the guest's own bytes at that
    /// address, which the GPU never wrote. Its ball-strike sequence went WHITE for ~150
    /// frames because of it.
    ///
    /// # Why this is not an `rtt` entry, which was tried and reverted
    /// `rtt` was keyed by ADDRESS ALONE and rebuilt a target whose extent changed, and a
    /// display buffer's address is ALSO used as an ordinary guest colour surface at a
    /// different size (this title's `0x88f00500` is both a 640x368 surface and the 960x544
    /// display). Registering the display image there made the two roles evict each other
    /// every frame and broke the very menu it fixed.
    ///
    /// >>> THAT EVICTION IS GONE - `rtt` is keyed by `(address, width, height)` now, for
    /// >>> exactly the same class of collision (see `rtt`) - AND THIS MAP STAYS ANYWAY.
    ///
    /// The rest of the reasoning never depended on the keying: the image is at the
    /// CALLER's resolution rather than the guest's (see below), it is rendered into a
    /// framebuffer whose extent has nothing to do with any guest surface, and it must not
    /// take part in the clear/load, snapshot and reclamation rules `rtt` entries obey.
    /// Kept apart, it can only FILL A GAP: it is consulted after `rtt` and never overrides
    /// it, so an address `rtt` already answers for behaves exactly as before.
    ///
    /// The image is at the CALLER's resolution, not the guest's. Every shader that samples
    /// one of these does so with NORMALISED coordinates, so the extent does not have to
    /// match - and matching it would mean rasterising the display pass at the guest's
    /// resolution, which is a quality loss no defect here justifies.
    display_images: HashMap<u32, DisplayImage>,
    /// The display pass now being encoded samples its own buffer: `(address, snapshot)`,
    /// offered to its samplers in place of the attachment. Set and cleared around that one
    /// pass by the display arm - see `snapshot_display`.
    display_reads_snapshot: Option<(u32, wgpu::TextureView)>,
    /// Native: shader modules built AHEAD on background threads for the pairs the guest's
    /// patcher named, `(module key, module)`, waiting to join `GxpLive::modules` - see
    /// [`GxmRenderer::premodule_pairs`].
    #[cfg(not(target_arch = "wasm32"))]
    premodules: std::sync::Arc<std::sync::Mutex<Vec<(u64, wgpu::ShaderModule)>>>,
    /// The allocation pairs already handed to [`GxmRenderer::premodule_pairs`].
    #[cfg(not(target_arch = "wasm32"))]
    premodule_seen: HashSet<(usize, usize)>,
    /// Fixed-function bind groups over a rendered target, keyed by (address, linear,
    /// reading-the-snapshot). The snapshot flag is part of the key because the two
    /// views are different textures - binding the live one where the snapshot is meant
    /// would make the target its own input.
    ///
    /// >>> STILL THE SAMPLED ADDRESS, NOT THE `rtt` KEY, and deliberately: this names the
    /// view the SAMPLER path resolved for a draw, and that path resolves by ADDRESS and by
    /// RANGE (see `rendered_alias` - a title samples a 960x544 target through a 1920x1088
    /// descriptor 256 bytes in, and matching the extent would miss it entirely). The draw
    /// has an address; which target answers for it is decided once, in `sample_views`.
    /// Invalidation is therefore by address too, which over-invalidates when two targets
    /// share one - it rebuilds a bind group that was still good, which costs one bind
    /// group and cannot be wrong.
    rtt_binds: HashMap<(u32, bool, bool), wgpu::BindGroup>,
    /// A CUBE MAP the guest RENDERS, assembled from the six 2D targets its six faces were
    /// rendered into, keyed by the guest address of face 0 (which is the cube texture's own
    /// `data_addr`).
    ///
    /// >>> BY ADDRESS ALONE, because that is how the SAMPLER names a cube: the guest's own
    /// texture declares `data_addr` and `faces == 6`, and `prepare` looks it up by that
    /// address. The face EXTENT is not lost - it is in `CubeFromRenders::size`, and
    /// `assemble_rendered_cubes` now searches the extents `rtt_rendered` actually holds at
    /// face 0's address rather than assuming there is only one.
    ///
    /// # A cube face IS a GXM render target, and this renderer assumed it was not
    /// Both render-target sampler paths are 2D-only, on that stated assumption. MEASURED on
    /// a golf title: ten shader pairs bind a cube at unit 11 naming `0x891e6520`, and the frame
    /// renders six 256x256 passes at exactly `0x891e6520 + n*0x40000` - six faces of one
    /// 256x256 RGBA8 cube, laid out back to back the way GXM lays out a cube's faces. With
    /// the assumption in place the cube fell through to an upload of GUEST MEMORY, which the
    /// GPU wrote and the guest never did, so a dynamic reflection sampled stale or empty
    /// bytes ([[vitaslop-a-render-target-reads-empty-in-guest-memory]]). The largest of the
    /// six passes carried 356 draws - the biggest pass in the whole frame - which is what
    /// `report_world_not_on_display` had been reporting as "renders into a target nothing
    /// reads".
    ///
    /// # Why it PERSISTS across frames rather than living in `rtt_rendered`
    /// The title re-renders the six faces periodically, not every frame, and samples the
    /// cube every frame - so a map cleared at the top of the frame would answer for the
    /// refresh frames and fall back to stale guest memory for all the others, which is the
    /// same defect with a duty cycle. The guest's own buffer persists in memory between
    /// refreshes; this mirrors that. Entries are refreshed in place whenever all six faces
    /// render again, so the contents track the guest's.
    rtt_cubes: HashMap<u32, CubeFromRenders>,
    /// The cube FACES (by `rtt` key) already copied into their cube in the frame being
    /// encoded. Cleared with `rtt_rendered` at the top of each frame; `rtt_cubes` itself is
    /// NOT, because the texture has to survive the frames between the guest's refreshes.
    ///
    /// Per FACE rather than per cube: a title refreshes a cube ONE FACE PER FRAME (the
    /// baseball title's reflection probe), and `assemble_rendered_cubes` runs before every
    /// pass, so this is what stops one face's copy being re-recorded ahead of every later
    /// pass while still letting a face drawn later in the same frame be copied when it is.
    cube_faces_copied: HashSet<(u32, u32, u32)>,
    /// Every target RENDERED since a cube last copied it, across frames - see
    /// `assemble_rendered_cubes`. A face rendered on a frame that samples no cube (a scene
    /// reload re-rendering its reflection probe behind a loading screen) is still owed its
    /// copy on the first frame that does; `rtt_rendered` forgets it at the frame's end.
    rtt_rendered_uncopied: HashSet<(u32, u32, u32)>,
    /// The target `report_world_not_on_display` found orphaned on the PREVIOUS frame, so
    /// it can require the same answer twice before reporting. See that function for why
    /// one frame is not enough to ask the question on.
    orphan_candidate: Option<u32>,
    /// >>> THE GUEST'S OWN DISPLAY BUFFERS FOR THE FRAME BEING ENCODED - every address it
    /// >>> passed to `sceDisplaySetFrameBuf` while this frame's scenes were captured.
    ///
    /// Set by the frontend before [`Self::encode_chain`] (see `set_presented`), and EMPTY
    /// on a frontend that does not supply it, which is exactly the behaviour that shipped
    /// before this existed.
    ///
    /// # What it is for
    /// The display buffer is otherwise taken to be "whatever the LAST scene draws to". That
    /// is right for a frame whose scenes all belong to one image and wrong for one that
    /// STRADDLES A FLIP: a title rotating three display buffers (0x88b00300 / 0x88d00400 /
    /// 0x88f00500 on the one measured) can draw its world into buffer A, flip, and draw its
    /// HUD into buffer B - and then A is classified as an offscreen pass, rendered into a
    /// target nothing reads, and DROPPED. The picture that produces is a HUD over black,
    /// which is the "black course" symptom exactly.
    ///
    /// A flip address is the guest's own statement that a buffer is a display buffer, so
    /// this is evidence rather than a heuristic about extents. It is used ONLY to rescue
    /// the straddling case - see `encode_chain` - and by `display_choice`.
    presented: Vec<u32>,
    /// Whether a flipped address has ever equalled a scene target on this run - the
    /// evidence `display_choice` needs before it trusts a frame that matched none.
    presented_matched: bool,
    /// Every address the guest has flipped this run (bounded) - its display buffers. See
    /// `display_choice`.
    flipped_ever: Vec<u32>,
    /// The most recent flipped address, for a frame that flipped nothing.
    last_flipped: Option<u32>,
    /// Addresses whose entry in `rtt_rendered` is currently the snapshot rather than
    /// the live target (the pass being encoded draws into that address).
    ///
    /// By ADDRESS rather than by the `rtt` key, because it is consumed on the SAMPLER side
    /// (`prepare`'s `reads_snapshot`, and the fixed-function bind below), where a draw has
    /// only the address it named - the same reason `rtt_binds` keeps an address key.
    rtt_reads_snapshot: HashSet<u32>,
    /// The DESTINATION-colour texture: a copy of the pass's colour attachment as it stood
    /// immediately before a draw whose fragment program reads it.
    ///
    /// # Why this exists at all
    /// A fragment program on this hardware can read its own output registers, which the
    /// driver seeds with the framebuffer colour - so a program can blend for itself in
    /// ordinary ALU. One retail title does exactly that for its whole frame: its
    /// colour-grading composite is `sa4*dst + dot(sa8, dst)*sa2 + sa0.x`, a per-channel
    /// function of the DESTINATION that no fixed-function blend can express, and reading
    /// the output bank as zeros painted the finished world black every frame.
    ///
    /// WebGPU has no framebuffer fetch, so the only faithful mechanism is to end the render
    /// pass, copy the attachment here, and start a new pass with the copy bound. That is a
    /// real cost (a tiler flushes and reloads its tiles at every split), so it is paid ONLY
    /// for the draws that read the destination, and the split count is reported.
    ///
    /// Sized to the ATTACHMENT, not the guest surface, because the copy is texel-for-texel -
    /// so there is one per attachment SHAPE. A frame alternates between an offscreen target
    /// and the display buffer, and keeping a single texture would recreate it twice a frame,
    /// bumping the generation and emptying the group-3 bind-group cache every time.
    dest_colors: HashMap<(u32, u32, wgpu::TextureFormat, u32), DestColor>,
    /// Source of [`DestColor::generation`]: bumped whenever one is created, so a cached
    /// group-3 bind group can never name a view the renderer has since replaced.
    dest_gen: u64,
    /// The per-sample copy into a MULTISAMPLED [`DestColor`], by (format, samples) - see
    /// [`GxmRenderer::copy_attachment_to_dest`].
    dest_ms_copy: HashMap<(wgpu::TextureFormat, u32), (wgpu::RenderPipeline, wgpu::BindGroupLayout)>,
    /// The draw that fills a multisampled attachment from its target's resolved image - see
    /// [`GxmRenderer::seed_msaa_from_color`].
    msaa_seed: HashMap<(wgpu::TextureFormat, u32), (wgpu::RenderPipeline, wgpu::BindGroupLayout)>,
    /// The half-precision ON-CHIP PIXEL DATA of an 8-bit scene whose programs blend at
    /// register precision, by attachment extent - see [`GxmRenderer::tile_shadow`].
    tile_shadows: HashMap<(u32, u32), (wgpu::Texture, wgpu::TextureView)>,
    /// The draw that sets every texel's alpha to 1.0 - see [`GxmRenderer::fill_alpha_one`].
    alpha_fill: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,

    /// Views of the guest-encoded DEPTH of the targets already rendered this frame, keyed
    /// by the guest address of the depth surface (NOT the colour one). A sampler naming
    /// one of these is asking for a distance, and must be resolved here BEFORE the
    /// colour-target range match, which would otherwise claim the address first.
    rtt_depth_rendered: HashMap<u32, wgpu::TextureView>,
    /// Which render target holds the converted depth for a given DEPTH address:
    /// `depth address -> the ``rtt`` key whose ``gxm_depth`` is that surface`. The VALUE is
    /// the full `(address, width, height)` key, not a bare colour address: two targets can
    /// share a colour address (see `rtt`) and only one of them holds the depth this surface
    /// was converted from. Kept across
    /// frames so the cross-frame carry-forward in `encode_chain` can re-offer a depth
    /// buffer an EARLIER frame filled without having to guess the key.
    ///
    /// # Why the two addresses cannot be conflated
    /// `rtt` is keyed by the COLOUR address of a pass (a depth-ONLY pass is the one case
    /// where the two coincide), while `rtt_depth_rendered` is keyed by the DEPTH address.
    /// Carrying the map forward by iterating `rtt` and using ITS key therefore registers a
    /// COLOUR address as a depth surface - and the depth path is consulted FIRST, so the
    /// next pass sampling that colour target is handed a distance where it asked for an
    /// image. That shipped: a racer's whole world rendered as a single-channel red
    /// gradient, because its composite samples the 960x544 target the frame just drew.
    rtt_depth_addrs: HashMap<u32, (u32, u32, u32)>,
    /// Buffers this renderer has finished with, held so they can be `destroy()`ed at the
    /// START of the next frame rather than left to be collected.
    ///
    /// **What feeds this is now only a GROW.** It used to take every GXP arena buffer of
    /// every pass, every frame, because those buffers were created fresh each time. They
    /// are POOLED now ([`GxmRenderer::gxp_arenas`]), so the steady state retires nothing at
    /// all and the only buffers arriving here are the ones a slot outgrew. The measurement
    /// below is what the churn used to cost and is why the pool exists; it is history, not
    /// a description of the current per-frame path.
    ///
    /// # Why dropping them is not enough, and why this is a BROWSER problem
    /// Dropping a `wgpu::Buffer` releases the handle. On a native backend that returns the
    /// allocation promptly. In the browser a `wgpu::Buffer` is a `GPUBuffer` living in
    /// JavaScript, and dropping the Rust handle only makes it GARBAGE - the GPU memory
    /// behind it is reclaimed whenever the JS collector next feels like it, which is not a
    /// schedule a renderer allocating every frame can rely on. `destroy()` is the only
    /// thing that releases it on our schedule, and before this it was called NOWHERE in
    /// the project.
    ///
    /// MEASURED, the committed race playbook to f9000 in desktop Chrome headless: the GPU
    /// process sits FLAT at 0.20 GB through boot and the menus, then climbs through the
    /// race to a 4.96 GB working set and **13.4 GB of private bytes - which is
    /// approximately the run's CUMULATIVE buffer allocation**, i.e. the shape of nothing
    /// being reclaimed at all. Everything else in the frame is steady by then: zero
    /// textures decoded, zero uploaded, zero targets created, zero pipelines built. These
    /// arenas (3 buffers x 11 passes, ~4.4 MB) are the only per-frame allocation left.
    ///
    /// Destroyed at the start of the NEXT frame, not at the end of this one: the caller
    /// submits the encoder after `encode_chain` returns, so at the end of the frame these
    /// buffers may still be named by commands that have not been submitted. By the next
    /// call the submit has happened, and WebGPU defines `destroy()` on a buffer with work
    /// in flight as completing that work before releasing the memory.
    ///
    /// >>> THE INVARIANT THIS RELIES ON, stated because it is now load-bearing rather than
    /// > > > merely true: **each caller creates one encoder, calls `encode_chain` ONCE on it, and
    /// > > > submits before calling again.** All of them do - `vitaslop-desktop/retail.rs`,
    /// > > > `vitaslop-native/wgpu_render.rs`, `vitaslop-web/lib.rs`, and the `encode` wrapper,
    /// > > > which forwards a single scene. A caller that encoded two chains into one
    /// > > > un-submitted encoder would have the second frame's start destroy the first frame's
    /// > > > arenas out from under recorded commands, so if that ever becomes a shape this
    /// > > > renderer supports, the graveyard has to be keyed to the SUBMISSION, not to the call.
    retired_buffers: Vec<wgpu::Buffer>,
    /// Retired buffers kept for reuse rather than freed - see [`BufferPool`], which carries
    /// the measurement that made it necessary.
    buffer_pool: BufferPool,
    /// TEXTURES retired mid-frame, on exactly the same schedule and for a sharper reason
    /// than the buffers: a render target REBUILT while the frame's encoder already holds a
    /// pass that names its attachments makes that whole COMMAND BUFFER invalid, and the
    /// submit fails with "Texture with 'gxm-rtt-depth' label has been destroyed" - which
    /// loses every draw in the frame, not just the one that owned the target.
    ///
    /// MEASURED on a baseball title in a live game: 11 scenes a frame, one target rebuilt
    /// among them, and **9,099 validation errors** in one run - the picture was BLACK while
    /// the guest happily submitted 540-600 draws a frame. `ensure_rtt` is not a
    /// shutdown-only path; it runs whenever the guest resizes a target, gains a depth
    /// reader or changes its sample count, and a title with several render targets does
    /// that inside a frame.
    retired_textures: Vec<wgpu::Texture>,
    /// The pass currently being encoded writes into a target whose depth is sampled later,
    /// so its depth attachment must be STORED rather than discarded.
    keep_depth: bool,
    /// The pass currently being encoded writes a separate STENCIL plane a later pass
    /// samples, so its stencil must be STORED for [`Self::convert_gxm_stencil`] to read.
    keep_stencil: bool,
    /// The separate stencil planes this frame SAMPLES, by stencil address, each with the
    /// `SWIZZLE1` selector of the U8 texture the guest reads it through. Built per frame in
    /// `encode_chain`; see [`Self::convert_gxm_stencil`].
    stencil_sampled: HashMap<u32, u32>,
    /// Each sampled stencil plane converted to the RGBA8 its sampler reads, by stencil
    /// address: `(texture, view, width, height)`. Registered in `rtt_depth_rendered` at
    /// that address after the pass that wrote it.
    stencil_planes: HashMap<u32, (wgpu::Texture, wgpu::TextureView, u32, u32)>,
    /// The stencil conversion pipelines, one per `SWIZZLE1` selector, built on first use.
    stencil_convert: HashMap<u32, (wgpu::BindGroupLayout, wgpu::RenderPipeline)>,
    /// The pass currently being encoded starts from a depth-stencil LOADED from
    /// `depth_saved` (its surface's FORCE_LOAD bit) instead of clearing it.
    depth_load_now: bool,
    /// The pass currently being encoded starts its STENCIL from a plane loaded out of a store
    /// mirror (see `store_mirror` and [`Self::load_guest_stencil`]) - independent of its depth,
    /// which keeps its own load-or-clear rule.
    stencil_load_now: bool,
    /// The word-row copy the stencil load and store go through, built on first use, and the
    /// staging buffer that holds the plane at a buffer-texture copy's pitch. See `store_mirror`.
    stencil_row_copy: Option<(wgpu::BindGroupLayout, wgpu::ComputePipeline)>,
    stencil_staging: Option<wgpu::Buffer>,
    /// >>> THE DEPTH-STENCIL A FORCE_STORE PASS LEFT IN MEMORY, by (depth address, width,
    /// >>> height) - what a later FORCE_LOAD pass at that address starts from.
    ///
    /// Every render target here has its OWN depth attachment, so two passes into different
    /// colour targets never shared depth - and on the console they do whenever the guest
    /// says so: the tiler stores a surface's depth at the end of a scene with FORCE_STORE
    /// and loads it at the start of one with FORCE_LOAD (`zlsControl`, see
    /// [`RenderScene::zls_control`]). MEASURED on an action title: its deferred
    /// frame draws the world into a G-buffer with depth at 0x8a8dee00 (store), then lights
    /// it into an HDR target with 49 draws testing EQUAL / LEQUAL / GREATER against that
    /// same depth (load + store). Each started from a CLEARED buffer, so every light and
    /// every re-drawn surface failed its test, and the lit image held only the always-pass
    /// backdrop - fog and a distant rock where the daylit ruins and the player were.
    /// Single-sample passes only (a copy needs matching sample counts); a multisampled
    /// FORCE_LOAD pass is reported. `VITASLOP_GXM_ZLS=0` is the arm back.
    depth_saved: HashMap<(u32, u32, u32), wgpu::Texture>,
    /// The word views of raw 64-bit surfaces (see the pass encoder's `raw_word_view` block),
    /// by (sampled address, width, height, format): the staging buffer the words go through
    /// and the texture a sampler binds. REUSED every frame - made fresh they were ~5 MB of
    /// GPU objects per view per frame left to the garbage collector, the churn that took
    /// a fighting title's GPU process to 2.3 GB (`subrect_free`).
    word_views: HashMap<(u32, u32, u32, wgpu::TextureFormat), (wgpu::Buffer, wgpu::Texture, wgpu::TextureView)>,
    /// Draws in the current chain that sampled a target this frame rendered. Zero over
    /// a frame with several passes means the composite is NOT reading them, which is a
    /// different problem from the passes not being drawn - and the two look identical
    /// on screen.
    rtt_hits: usize,
    /// Every chain SHAPE already reported, as a 64-bit fold. See the `gxm chain:` line.
    ///
    /// >>> A SINGLE-SLOT MEMO WAS A PER-FRAME DIAGNOSTIC IN DISGUISE, AND IT WAS MEASURED.
    ///
    /// This was `Option<String>`, holding the last line, on the reasoning that "a title
    /// whose frame structure is stable pays one line for a run". A race's structure is not
    /// stable - it CYCLES, and a one-slot memo turns a cycle of N shapes into N reports per
    /// cycle, for ever. MEASURED in the browser on a racer's race: 16,218 of these lines
    /// in one run, and with the `gxp: scene has` tally beside it **58,418 warn lines**, worth
    /// **1.9 ms of a 5.0 ms render** - 38% of the browser's whole renderer
    /// ([[vitaslop-a-diagnostic-can-bury-the-findings]] is the same lesson from the other
    /// end: there the volume hid the findings, here it costs the frame).
    ///
    /// A SET reports each distinct shape once, which is what the single slot was trying to
    /// say. Bounded, and it says so when the bound is reached rather than going quiet.
    chain_shapes_seen: HashSet<u64>,
    /// Whether the chain-shape cap has been announced, so a run that reaches it says so
    /// exactly once instead of silently dropping every later shape.
    chain_shapes_capped: bool,
    /// Every render-target address some draw of the FRAME just encoded samples. The
    /// `sampled_addrs` beside it is the LAST PASS's set, which answers a different question
    /// (did the composite read the world) and cannot answer this one. Kept so the RTT
    /// writeback can be asked to hold back a target the GPU consumes.
    frame_sampled: HashSet<u32>,
    /// How many frames this renderer has been handed, counted so the `gxm chain:` line can
    /// say WHICH one a shape was first seen on.
    ///
    /// >>> WITHOUT IT THE LINE READS AS PER-FRAME AND IS NOT. The report fires once per
    /// distinct shape, so a log's second chain line can be the 40th frame, and a reader
    /// comparing "the first line" against "the last line" is comparing two shapes, not two
    /// frames. MEASURED: a session read a 5-draw overlay pass in the first line against a
    /// 39-draw one in a later line and concluded the guest submitted different work on its
    /// first rendered frame. It did not - the two lines are different PASSES of the same
    /// steady frame. The ordinal makes that visible in the line itself.
    chain_frames_seen: u64,
    /// Guest addresses of every texture the pass being encoded sampled. Reported for
    /// the frame's LAST pass: a composite that shows none of the world is either not
    /// sampling the world target or sampling an address nothing rendered, and those
    /// need opposite fixes.
    sampled_addrs: HashSet<u32>,
    /// The live GXP->WGSL recompiler: a per-shader-pair pipeline cache. When enabled
    /// (`VITASLOP_GXP_LIVE`) a draw carrying [`super::GxpRecompile`] is rendered with the
    /// guest's real shaders; a pair that fails to link falls back to the fixed-function
    /// pipelines above. Disabled -> zero cost (the payload is simply ignored).
    gxp: GxpLive,
    /// >>> THE SIX PER-PASS STAGING ARENAS WERE POOLED ON `self` HERE, AND IT WAS
    /// > > > REVERTED. Kept as a note because the reasoning that made it look free is the trap.
    ///
    /// `encode_pass` opens each pass with six `Vec::new()`, so each grows from zero by
    /// doubling as several hundred draws append - which a sampler profile charges to the
    /// ALLOCATOR (7.3% of the busy worker thread, with `push_sa`, `prepare` and
    /// `encode_pass` named as the `malloc` callers). Holding them on the renderer and
    /// `clear()`ing instead removes all of that, and it is BIT-IDENTICAL (10 of 10 frames,
    /// frame-pinned oracle).
    ///
    /// **It still made the target device slower, and the user felt it before any instrument
    /// here saw it** - golf went from holding 30 fps to sitting in the teens. The mechanism
    /// the "it is obviously free" argument missed: `clear()` keeps CAPACITY, a loading frame
    /// pushes 16 MB through these, and six arenas pinned at a one-off peak is tens of
    /// megabytes resident forever in a linear memory a wasm heap can never give back. That
    /// tightens the heap for EVERYTHING, including the guest - which is why the
    /// "a render-side change cannot move guest `cpu`" argument was wrong. It reasons about
    /// call graphs; the allocator is shared.
    ///
    /// >>> TRIED AGAIN, WITH THE BOUND THE NOTE ASKS FOR: see `pass_arenas` below.
    /// The six per-pass staging arenas, held across passes and frames with their CAPACITY
    /// BOUNDED - the retry the note above asks for.
    ///
    /// Taken out with `mem::take` at the top of a pass and put back at the bottom, which is
    /// what keeps them usable while the rest of the renderer is borrowed mutably (the same
    /// idiom as `rtt_rendered`). Each is `clear()`ed on the way in and SHRUNK on the way out
    /// if it is holding more than [`ARENA_RETAIN_BYTES`], so a loading frame's 16 MB peak
    /// cannot stay pinned in a wasm linear memory that never gives anything back - which is
    /// the exact mechanism that made the unbounded version slower on the target device while
    /// being bit-identical and obviously free on a desktop.
    ///
    /// OFF by default; [`arena_pool_enabled`] is the knob and carries the measurement that
    /// says why.
    pass_arenas: [Vec<u8>; 6],
    /// What the last [`GxmRenderer::encode`] spent, phase by phase. See [`EncodePhases`].
    last_phases: EncodePhases,
    /// The same, summed over every pass of the last [`GxmRenderer::encode_chain`] - which
    /// is the frame, and the only figure the caller's `encode` total can be compared with.
    chain_phases: EncodePhases,
}

// ==================== Live GXP->WGSL recompiler path ====================
//
// When enabled (`VITASLOP_GXP_LIVE`), a draw that carries the guest's real vertex +
// fragment `SceGxmProgram` blobs (see [`super::GxpRecompile`]) is rendered with those
// shaders, translated to one linked WGSL module by `vitaslop_gxp_shader::link_programs`
// and executed on a real pipeline. A pair that fails to link (or uses a vertex format /
// 3D sampler we do not yet map) falls back to the fixed-function pipeline for that draw -
// a wrong translation never paints a pixel. Pipelines are cached by shader identity;
// per-draw vertex/index/uniform buffers + bind groups are built each frame (a real title's
// recompilable draw count is small, and this keeps the path simple and correct first).

use super::{GxpAttr, GxpRecompile, GxmTexture, RegionClip};
