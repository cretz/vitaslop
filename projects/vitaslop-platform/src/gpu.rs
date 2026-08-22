//! The shared cube render pipeline. The WGSL shader, vertex layout, and per-draw
//! encoding live here once so the native headless renderer (`vitaslop-native`)
//! and the browser WebGPU canvas (`vitaslop-web`) draw identically - the native
//! path stays the pixel oracle for the browser. Only the surrounding host bits
//! differ: native acquires an adapter headlessly and reads pixels back, the
//! browser acquires a device from `navigator.gpu` and presents to a canvas.
//!
//! [`DrawBatch`] is the neutral, GPU-free seam: the runtime turns its captured
//! GXM scene into a list of these, and [`CubeRenderer`] (behind the `gpu`
//! feature) draws them. Keeping `DrawBatch` free of any wgpu types lets the
//! engine-agnostic runtime produce it without pulling in a GPU stack.

/// A renderer diagnostic, at `debug` on the `vitaslop::gxm` target.
///
/// # Why these are not `eprintln!`
/// This module is the SHARED render path: the same code draws the native pixel oracle
/// and the browser canvas. On wasm32 stderr is a no-op sink, so every `eprintln!` here
/// was discarded in the browser - including the recompiler's own per-scene verdict, the
/// one line that says whether a frame's draws were really recompiled or quietly fell
/// back. The engine that is hardest to inspect was the one reporting nothing, and a
/// black frame there was indistinguishable from a correct one.
///
/// Read them with `RUST_LOG=vitaslop::gxm=debug` natively, or
/// `VITASLOP_LOG=warn,vitaslop::gxm=debug` in the browser. Anything that means the
/// renderer DEGRADED - a fallback, a dropped draw, an approximated state - goes through
/// [`report_warn`] instead, so it survives the default `warn` filter on both engines
/// without anyone having to know to ask for it.
macro_rules! report {
    ($($arg:tt)*) => { tracing::debug!(target: "vitaslop::gxm", $($arg)*) };
}

/// A renderer diagnostic that means the output is WRONG or approximated. Always visible
/// at the default filter - see [`report`].
macro_rules! report_warn {
    ($($arg:tt)*) => { tracing::warn!(target: "vitaslop::gxm", $($arg)*) };
}

/// The output of a diagnostic whose own KNOB is already the gate.
///
/// Emitted at `warn`, like [`report_warn`], and for a reason that is not about severity:
/// setting `VITASLOP_GXP_INPUTS_ORDER=all` is an unambiguous request for these lines, and
/// a second, invisible gate underneath it can only turn that request into silence. These
/// were `report!` (`debug`) while every documented repro command in the project's notes
/// says `VITASLOP_LOG=warn` - so the knob was set, the run completed, and the diagnostic
/// printed nothing, which reads exactly like "this draw never happened". It cost a run
/// here and, going by the notes, a good deal more than that elsewhere.
///
/// Use this ONLY where a knob has already decided the line should exist. A diagnostic that
/// fires on every draw regardless still belongs at `debug`.
macro_rules! report_knob {
    ($($arg:tt)*) => { tracing::warn!(target: "vitaslop::gxm", $($arg)*) };
}

/// One ready-to-draw batch: a model-view-projection matrix and the raw vertex and
/// index bytes, already snapshotted from guest memory. The vertex layout is the
/// cube's fixed-function format (see [`CUBE_SHADER`]): position `float32x3` at
/// offset 0, color `unorm8x4` at offset 12, stride 16.
#[derive(Clone, Debug, PartialEq)]
pub struct DrawBatch {
    /// Column-major 4x4 MVP matrix (the guest's vertex default uniform buffer).
    pub mvp: [f32; 16],
    /// Interleaved vertex bytes, stride 16.
    pub vertices: Vec<u8>,
    /// Index buffer bytes.
    pub indices: Vec<u8>,
    /// Number of indices to draw.
    pub index_count: u32,
    /// True if indices are 32-bit; false for 16-bit (the cube uses 16-bit).
    pub index_u32: bool,
}

/// The fixed-function cube shader, the blob-free equivalent of the guest's
/// placeholder GXP shaders. It transforms each vertex by the captured MVP and
/// interpolates per-vertex color. The `clip.z` remap converts GL-style clip space
/// (z in [-1, 1], which the guest's projection targets) to WebGPU clip space
/// (z in [0, 1]).
pub const CUBE_SHADER: &str = r#"
struct Uniforms { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> VsOut {
    var out: VsOut;
    var clip = u.mvp * vec4<f32>(position, 1.0);
    clip.z = (clip.z + clip.w) * 0.5;
    out.pos = clip;
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

// ---------------------------------------------------------------------------
// General GXM renderer seam
// ---------------------------------------------------------------------------
//
// [`CubeRenderer`] above is the fixed-function 3D-cube path (one shader, one
// vertex layout, opaque depth-tested). A real 2D title needs more: per-draw
// coordinate spaces, textured sprites, and alpha blending in submission order.
// [`RenderScene`] is the neutral, GPU-free seam for that - the runtime decodes a
// captured GXM scene into it (reusing the exact interpretation its software
// rasterizer uses, so the two agree), and [`GxmRenderer`] (behind `gpu`) draws
// it. Keeping the decode in the runtime and only the draw here means the GPU
// renderer never touches guest memory or a GXM enum - it sees canonical vertices
// and a linear RGBA8 texture, and stays the faithful GPU twin of the software
// oracle.

/// The coordinate space a general draw's vertex positions live in, recovered by
/// the runtime from the draw's vertex layout - the same [`Space`] the software
/// rasterizer uses. Determines both the vertex transform and whether the draw is
/// depth-tested (only `Mvp` is; the 2D spaces paint in submission order).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DrawSpace {
    /// Object space transformed by a captured column-major 4x4 MVP uniform (the 3D
    /// path): perspective divide, GL-style clip z, depth-tested opaque.
    Mvp([f32; 16]),
    /// Clip coordinates emitted directly, in [-1, 1] (a fullscreen pass).
    Ndc,
    /// Screen pixels in [0, surface] with Y down (2D sprite quads whose vertex
    /// program baked the pixel-to-clip transform).
    Pixel,
}

/// A decoded, GPU-ready texture: a tightly-packed linear `Rgba8Unorm` image plus a
/// content/identity `key` the renderer caches uploads by, so an unchanged texture
/// (the common case - a font atlas or background reused every frame) is uploaded
/// once and then only bound. `rgba` is shared (`Arc`) so re-handing the same
/// texture to the renderer each frame is a pointer copy, not a pixel copy.
/// The layout of [`GxmTexture::rgba`], and the only two the renderer uploads.
///
/// Deliberately a closed set of two rather than "whatever the guest format was". The decoders
/// converge every one of the guest's several dozen formats onto one of these, so the renderer,
/// the software rasterizer and the upload path each handle two cases and not fifty - and a
/// format added later has to pick one, rather than silently arriving as bytes nobody sizes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TexelSeam {
    /// Four `u8` lanes per texel, normalized. Every colour format lands here.
    Rgba8,
    /// Four IEEE binary16 lanes per texel, little-endian. Exact for the guest's own F16
    /// formats, which is the point: these carry data, and a re-encode would defeat the reason
    /// for the wider seam.
    Rgba16Float,
}

impl TexelSeam {
    /// Bytes one texel occupies in `rgba`.
    pub fn bytes_per_texel(self) -> usize {
        match self {
            TexelSeam::Rgba8 => 4,
            TexelSeam::Rgba16Float => 8,
        }
    }
}

/// The optional device features this renderer wants, of those the adapter offers.
///
/// # Every device asked for `Features::empty()`, and that was the whole reason a Vita frame
/// needed 260 MB
/// A guest frame that references 260 MB of texture is not referencing 260 MB - a Vita has
/// nowhere near it. Every compressed surface was being CPU-decoded to RGBA8 on the way in,
/// which is a 4-8x expansion, and past the cache budget the allocation fails on a phone and
/// the draw comes out WHITE. Asking for `TEXTURE_COMPRESSION_BC` is the precondition for
/// handing the guest's own blocks over instead.
///
/// It is requested only where the adapter HAS it - `request_device` fails outright otherwise -
/// so an adapter without it simply keeps the decode path, and [`CompressedUpload`] is ignored.
/// This is one function rather than four copies because four copies is how three of them come
/// to be right and one silently is not.
#[cfg(feature = "gpu")]
pub fn wanted_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    // BC AND ETC2, because the two together are what covers both halves of the hardware this
    // runs on. BC is preferred where both exist: the guest's own `UBC1/2/3` blocks ARE BC, so
    // that adapter can take them with no re-encode at all, while ETC2 always costs a lossy
    // transcode. An adapter that offers neither keeps the RGBA8 decode.
    let mut have = adapter.features();
    // >>> `VITASLOP_NO_BC=1` MAKES A DESKTOP BEHAVE LIKE THE TARGET DEVICE.
    //
    // The phone's adapter exposes `etc2, astc` and no BC, so it takes the transcode path for
    // EVERY compressed texture - which is where the GPU encoder, the ETC2 target and the whole
    // `CompressedData::Gpu` plan live. A desktop with BC never enters any of it, so all of it
    // was unexercised locally and its first real execution was on the user's phone. That has
    // already cost one shipped build.
    //
    // On this machine the Intel iGPU exposes BOTH families over Vulkan, so
    // `WGPU_ADAPTER_NAME=Arc WGPU_BACKEND=vulkan VITASLOP_NO_BC=1` is the device's texture
    // configuration, on real content, with the headless renderer.
    //
    // It only ever REMOVES a capability - the same shape as `VITASLOP_TEX_COMPRESS=0`, and the
    // reason a knob is admissible here at all ([[vitaslop-knob-is-the-gate-not-the-level]]): it
    // cannot make this build do something a real adapter would not.
    if crate::knobs::var("VITASLOP_NO_BC").ok().as_deref() == Some("1") {
        have.remove(wgpu::Features::TEXTURE_COMPRESSION_BC);
        report_warn!(
            "gxm textures: VITASLOP_NO_BC=1 - pretending this adapter has no BC, so every \
             compressed texture takes the TRANSCODE path exactly as the target device does"
        );
    }
    let want =
        have & (wgpu::Features::TEXTURE_COMPRESSION_BC | wgpu::Features::TEXTURE_COMPRESSION_ETC2);
    set_block_family(if want.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        BlockFamily::Bc
    } else if want.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2) {
        BlockFamily::Etc2
    } else {
        BlockFamily::None
    });
    // >>> PUBLISHED AS A WARNING, ON EVERY ENGINE, BECAUSE IT DECIDES A WHOLE PIECE OF WORK.
    //
    // Which block formats an adapter takes is the single fact that says whether the compressed
    // upload is available at all, and getting it off a device has failed TWICE for different
    // reasons: the browser published it under the same `Report` id as the adapter summary, so
    // the 100 ms rate limiter dropped it; and once that was fixed it landed on a panel id the
    // diagnostics dump does not carry. It is not a per-frame diagnostic and it is not something
    // a knob should have to ask for - it is one line, once, in the section people paste.
    let f = adapter.features();
    let have: Vec<&str> = [
        (wgpu::Features::TEXTURE_COMPRESSION_BC, "bc"),
        (wgpu::Features::TEXTURE_COMPRESSION_ETC2, "etc2"),
        (wgpu::Features::TEXTURE_COMPRESSION_ASTC, "astc"),
    ]
    .iter()
    .filter(|(bit, _)| f.contains(*bit))
    .map(|(_, name)| *name)
    .collect();
    report_warn!(
        "gxm textures: adapter compressed-texture support: {}. This build implements `bc` \
         (passthrough of the guest's own blocks, plus transcode) and `etc2` (transcode only, \
         4 bpp opaque / 8 bpp with alpha); taking {} for this run. A format in neither family is \
         decoded to RGBA8 at 4-8x its size.",
        if have.is_empty() { "NONE".to_string() } else { have.join(", ") },
        match block_family() {
            BlockFamily::Bc => "BC",
            BlockFamily::Etc2 => "ETC2",
            BlockFamily::None => "NOTHING",
        }
    );
    want
}

/// Whether this renderer is running on a WebGPU COMPATIBILITY-MODE adapter.
///
/// # Not a performance tier - a different validation regime
/// Compat mode is what a device gets when the full one is unavailable, and on the target device
/// that is not hypothetical: Chrome blocklisted the Imagination 25.1 driver
/// (crbug.com/520126488) and left only the compatibility adapter standing. Two of its rules bite
/// this renderer directly:
///
/// * a texture may NOT carry a view of a different format, so the sRGB twin this code declares
///   on every render target makes the target itself fail to create; and
/// * `textureLoad` is refused on a depth texture.
///
/// Neither degrades gracefully. The first produces an INVALID texture, and every view, bind
/// group, pass and submit built on it is invalid in turn - which reaches the screen as black,
/// with the cause thousands of validation errors earlier. MEASURED on the device: 5 failed
/// targets, 4,776 invalid bind groups, 1,173 invalid render passes, one black frame.
static COMPAT_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Record that the adapter in use is a compatibility-mode one. Set by the host before any
/// texture is created.
pub fn set_compat_mode(yes: bool) {
    COMPAT_MODE.store(yes, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the adapter in use is a compatibility-mode one. See [`set_compat_mode`].
pub fn compat_mode() -> bool {
    COMPAT_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Which block family the GPU this process is using accepts.
/// `0` unknown, `1` BC, `2` none, `3` ETC2.
static BLOCK_COMPRESSION: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Record what the adapter can take, so the CPU side does not prepare work the GPU cannot use.
///
/// Set by [`wanted_features`] at device creation, which is before any texture is decoded.
pub fn set_block_family(f: BlockFamily) {
    let v = match f {
        BlockFamily::Bc => 1,
        BlockFamily::None => 2,
        BlockFamily::Etc2 => 3,
    };
    BLOCK_COMPRESSION.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// The block family to encode for.
///
/// UNKNOWN answers `Bc` for the same reason [`block_compression_available`] answers `true`: the
/// caller is a cache-filling path where guessing wrong costs one texture's bytes, and the upload
/// asks the real device before it uses any of it.
pub fn block_family() -> BlockFamily {
    match BLOCK_COMPRESSION.load(std::sync::atomic::Ordering::Relaxed) {
        2 => BlockFamily::None,
        3 => BlockFamily::Etc2,
        _ => BlockFamily::Bc,
    }
}

/// Whether it is worth building a [`CompressedUpload`] at all.
///
/// # This exists so a device WITHOUT BC pays nothing for a feature it cannot use
/// The compressed source is built on the texture decode path, in the runtime, which has no
/// device and cannot ask. Without this it would de-swizzle and copy every BC texture's blocks
/// on every engine, and hold them in the decode cache for the life of the run, purely so the
/// uploader could decline them - and the engine that would pay for that is a phone whose GPU
/// exposes ASTC and ETC2 but not BC, i.e. exactly the device this whole line of work is for.
///
/// UNKNOWN reads as available: the only caller is a cache-filling path where being wrong costs
/// one texture's worth of bytes, while being wrong the other way would permanently disable the
/// passthrough on any engine that sets this late. The upload itself never trusts this - it asks
/// the real device (see `GxpLive::bc_supported`).
pub fn block_compression_available() -> bool {
    BLOCK_COMPRESSION.load(std::sync::atomic::Ordering::Relaxed) != 2
}

/// The last complete frame's uploaded texture working set, in bytes. `0` before the first frame.
static LAST_WORKING_SET: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Is the texture working set close enough to its budget that spending CPU to shrink it is still
/// worth something?
///
/// # Compression is a means, and this is the end it serves
/// Re-encoding a texture exists to make a frame's textures FIT. Once they fit with room to spare,
/// further compression buys memory nobody needs at a CPU cost this device cannot afford - and it
/// is the one resource it has least of. MEASURED on the target phone after the PVRTC transcode
/// landed: the race frame's working set fell from **256 MB against a 256 MB budget** to **82 MB**,
/// at which point the remaining BC textures could have been left alone entirely.
///
/// Two-thirds rather than the whole budget, because the answer has to be given BEFORE the work:
/// a texture skipped at 90% would be wanted again the moment the next screen loaded, and a
/// control that only reacts at the limit oscillates across it.
/// The most texture a game can have resident on the console, in MiB, and this renderer's default
/// cache budget: `ScePhyMemPartGame` (256) + the +109 MiB mode (109) + `ScePhyMemPartGameCdram`
/// (112). Spelled as its three partitions so it cannot quietly become a round number somebody
/// liked - the previous value was fitted to one title's MENU and cost a race 83% re-decodes on
/// the target device. See `gxm::GxpLive::views` for the full derivation.
///
/// >>> AT FILE SCOPE ON PURPOSE. It used to live inside `mod gxm`, which is `#[cfg(feature =
/// "gpu")]`, so `texture_budget_pressure` - which is not - could not reach it and carried its own
/// `unwrap_or(256)` instead. That is how the two disagreed by 221 MiB while a comment said they
/// could not.
pub(crate) const GAME_RESIDENT_CEILING_MB: usize = 256 + 109 + 112;

/// The texture-cache budget in bytes: [`GAME_RESIDENT_CEILING_MB`] unless
/// `VITASLOP_TEX_CACHE_MB` overrides it. Read once.
///
/// The ONE reader of that knob. Every consumer - the uploader's eviction, the pressure signal
/// above, the over-budget report - must call this rather than re-derive it.
pub(crate) fn tex_cache_budget_bytes() -> usize {
    use std::sync::OnceLock;
    static CELL: OnceLock<usize> = OnceLock::new();
    *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_TEX_CACHE_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(GAME_RESIDENT_CEILING_MB)
            * 1024
            * 1024
    })
}

/// `VITASLOP_RTT_BG_CACHE=0` restores the OLD behaviour: a sampler bind group naming a render
/// target is rebuilt every frame instead of being keyed by `rtt_epoch`. An A/B ARM, so it is
/// VALUE-sensitive rather than presence-only - a knob used as an arm that reads `NAME=0` as ON
/// has already cost this project a whole measurement
/// ([[vitaslop-knob-is-the-gate-not-the-level]]).
pub(crate) fn rtt_bg_cache() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::var("VITASLOP_RTT_BG_CACHE").map(|v| v.trim() != "0").unwrap_or(true))
}

/// `VITASLOP_GXP_CULL=0` restores the pre-2026-08-19b "draw both windings". An A/B arm, so it is
/// VALUE-sensitive rather than presence-only.
pub(crate) fn gxp_cull() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| crate::knobs::var("VITASLOP_GXP_CULL").map(|v| v.trim() != "0").unwrap_or(true))
}

/// Whether a shader pair the guest's patcher names is compiled AHEAD of the draw that binds it
/// (`VITASLOP_GXP_PRECOMPILE=0` restores compiling at the first draw). See
/// [`gxm::GxmRenderer::precompile_pairs`].
pub(crate) fn gxp_precompile() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_GXP_PRECOMPILE").map(|v| v.trim() != "0").unwrap_or(true)
    })
}

/// Microseconds spent inside `create_shader_module` and `create_render_pipeline` for recompiled
/// pairs, and how many pipelines that was. See the timing site in `build_gxp_pipeline`.
static PIPE_MODULE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PIPE_CREATE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Microseconds spent compiling WGSL AHEAD of any draw, when the guest's shader patcher named
/// the pair - see `GxmRenderer::precompile_pairs`. Reported separately from `PIPE_MODULE_US`
/// because the whole point is that this time is NOT in a gameplay frame: adding the two together
/// would hide the only thing the change is trying to move.
static PIPE_PRECOMPILE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Takes MILLISECONDS from `gxm::Stopwatch`, never a `std::time::Instant`.
///
/// >>> `std::time::Instant::now()` PANICS ON wasm32-unknown-unknown - "time not implemented on
/// this platform" - and it panics inside `GxpLive::prepare`, i.e. on the first frame the browser
/// renders anything. It shipped that way for one build and killed the run worker outright.
/// `Stopwatch` exists a few hundred lines above precisely for this and is the only clock this
/// module may use. [[vitaslop-count-bytes-when-there-is-no-clock]] is the same lesson from the
/// other direction: the browser has no phase timer, so reach for a COUNT before a clock.
fn add_build_ms(slot: &std::sync::atomic::AtomicU64, ms: f64) {
    slot.fetch_add((ms * 1000.0).max(0.0) as u64, std::sync::atomic::Ordering::Relaxed);
}

/// `(shader-module ms, pipeline-create ms)` since the last call, and reset. The split that says
/// whether building pipelines AHEAD of the draw that needs them is worth doing, and which half
/// of it could be moved.
pub fn take_pipeline_build_split() -> (f64, f64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        PIPE_MODULE_US.swap(0, Relaxed) as f64 / 1000.0,
        PIPE_CREATE_US.swap(0, Relaxed) as f64 / 1000.0,
    )
}

/// Milliseconds spent compiling WGSL ahead of the draw, since the last call, and reset.
pub fn take_precompile_ms() -> f64 {
    PIPE_PRECOMPILE_US.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0
}

/// Shader pairs whose PIPELINE the device itself refused, keyed the way every other report
/// here keys a pair.
fn poisoned_pairs() -> &'static std::sync::Mutex<std::collections::HashSet<u64>> {
    static POISONED: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<u64>>> =
        std::sync::OnceLock::new();
    POISONED.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

/// Learn from a WebGPU uncaptured error which shader pair the DEVICE could not build, so the
/// next frame can leave that one draw out instead of losing the whole picture.
///
/// # >>> ONE REFUSED PIPELINE BLANKS THE ENTIRE FRAME, AND THAT IS THE FAILURE THIS PREVENTS
/// A `setPipeline` with an invalid pipeline invalidates the render pass, which invalidates the
/// command buffer, which makes `queue.submit` a no-op. So a device that refuses FOUR of a
/// title's shader pairs does not lose four objects - it loses **every** draw of every frame,
/// and the screen is black with no visible relationship to the four errors scrolling past. That
/// is exactly how an Android PowerVR device presented a race: 1,554 draws prepared, 0 pixels.
///
/// The device's own message names the pipeline by its label, which is the pair key
/// (`[Invalid RenderPipeline "gxp:873eb144f958a48b"]`), so the text IS the diagnosis. Parsing it
/// back is not elegant, but the alternative - `pop_error_scope` - is asynchronous and the render
/// path is not, and no amount of elegance is worth the frame.
///
/// Returns how many distinct pairs this message newly poisoned.
pub fn note_device_error(kind: &str, msg: &str) -> usize {
    let mut newly = 0usize;
    let mut rest = msg;
    while let Some(at) = rest.find("gxp:") {
        rest = &rest[at + "gxp:".len()..];
        let hex = rest.len() - rest.trim_start_matches(|c: char| c.is_ascii_hexdigit()).len();
        let Ok(key) = u64::from_str_radix(&rest[..hex], 16) else { continue };
        rest = &rest[hex..];
        if poisoned_pairs().lock().unwrap_or_else(|e| e.into_inner()).insert(key) {
            ANY_POISONED.store(true, std::sync::atomic::Ordering::Relaxed);
            newly += 1;
            report_warn!(
                "gxp pair {key:016x}: THE DEVICE REFUSED THIS PIPELINE [{kind}] and every draw \
                 that uses it is being DROPPED from here on. One invalid pipeline invalidates \
                 the whole command buffer, so leaving it in loses the entire frame - this trades \
                 one object for every other object in it. THE PICTURE IS NOW INCOMPLETE and this \
                 is not a fix: the pair is named so it can be dumped with \
                 `VITASLOP_GXP_WGSL_DIR` and taken back to its two blobs by content hash. The \
                 device said: {msg}"
            );
        }
    }
    newly
}

/// Whether [`note_device_error`] has seen the device refuse this pair's pipeline.
///
/// The atomic is checked FIRST and is the whole point of it: this runs once per recompiled
/// draw, a race frame here submits ~560 of them, and taking a mutex per draw to ask a question
/// whose answer is "no" on every device that works would put the damage control in the hot
/// path of the case it exists to protect.
pub fn gxp_pair_poisoned(key: u64) -> bool {
    if !ANY_POISONED.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    poisoned_pairs().lock().unwrap_or_else(|e| e.into_inner()).contains(&key)
}

/// Set once [`note_device_error`] poisons anything. See [`gxp_pair_poisoned`].
static ANY_POISONED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// How many draws the poison list has cost this frame, since the last call, and reset.
pub fn take_poisoned_draws() -> u32 {
    POISONED_DRAWS.swap(0, std::sync::atomic::Ordering::Relaxed)
}

static POISONED_DRAWS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn texture_budget_pressure() -> bool {
    let ws = LAST_WORKING_SET.load(std::sync::atomic::Ordering::Relaxed);
    // Unknown (no frame finished yet) counts as PRESSURE: the first frames of a screen are
    // exactly when the working set is being built, and guessing "plenty of room" there would
    // skip the compression that stops the budget being blown in the first place.
    // >>> THE SAME FUNCTION the uploader enforces, not a second copy of its arithmetic.
    //
    // This used to re-read the knob here with its own `unwrap_or(256)` while
    // `gxm::tex_cache_budget_bytes` defaulted to the console's 477 MiB ceiling - so the comment
    // that used to sit here, "read the same way, so the two cannot disagree about what tight
    // means", was false, and they disagreed by 221 MiB. Pressure therefore declared itself at
    // 171 MiB where the uploader's own threshold is 318, and every screen spent its first frames
    // transcoding textures to save memory that was not short. Setting `VITASLOP_TEX_CACHE_MB`
    // hid it, because that moves BOTH readers at once - which is exactly how a duplicated
    // default stays invisible.
    let budget = tex_cache_budget_bytes();
    ws == 0 || ws * 3 >= budget * 2
}

/// The decoded texels of a [`GxmTexture`] - produced the FIRST time something reads them, and
/// never produced at all when nothing does.
///
/// # >>> THE DECODE WAS UNCONDITIONAL, AND THE PATH THAT SHIPS THROWS IT AWAY
/// Every bound texture was expanded to RGBA8 the moment it was captured, and the result was
/// then held for the life of the run in the decode cache. That is exactly right for the
/// fixed-function renderer, which samples those bytes. It is pure waste for the configuration
/// this project actually ships: the recompiler path hands the GPU the guest's own compressed
/// BLOCKS ([`CompressedUpload`]), so `upload_gxp_texture` returns before it ever looks at
/// `rgba`, and the expansion is a PVRTC or BC decode of the whole image whose only consumer is
/// the allocator that frees it.
///
/// It cost twice over. Once in CPU - the decode is the single most expensive thing a screen
/// transition does, and it ran for textures whose blocks were about to be handed over untouched.
/// And once in MEMORY - eight bytes of RGBA8 per byte of guest texture, resident in the decode
/// cache, which is what pushed a phone's working set past its budget and started the eviction
/// thrash that the compressed upload had just been written to prevent. A transcode paid it
/// TWICE: `transcoded_source` decodes each level itself, so the eager decode was a second,
/// discarded copy of level 0.
///
/// Laziness rather than a flag, deliberately: every consumer that genuinely needs texels
/// (the software rasterizer, the fixed-function upload, the vertex-texture clip probe, the
/// diagnostics) keeps working unchanged and simply pays for what it reads. There is no
/// configuration in which this returns different bytes from the eager decode - it returns the
/// same bytes, or nobody asks.
pub struct Texels(std::sync::Arc<TexelsInner>);

struct TexelsInner {
    ready: std::sync::OnceLock<Vec<u8>>,
    /// The decode itself. `None` is an already-materialised buffer (see [`Texels::ready`]).
    make: Option<Box<dyn Fn() -> Vec<u8> + Send + Sync>>,
}

impl Texels {
    /// Texels that already exist. No decode is deferred and [`Texels::resident`] is true from
    /// the start.
    pub fn ready(v: Vec<u8>) -> Self {
        let cell = std::sync::OnceLock::new();
        let _ = cell.set(v);
        Texels(std::sync::Arc::new(TexelsInner { ready: cell, make: None }))
    }

    /// Texels that will be produced by `f` if anything reads them. `f` runs AT MOST once, and
    /// every clone of this handle shares the one result.
    pub fn lazy(f: impl Fn() -> Vec<u8> + Send + Sync + 'static) -> Self {
        Texels(std::sync::Arc::new(TexelsInner {
            ready: std::sync::OnceLock::new(),
            make: Some(Box::new(f)),
        }))
    }

    /// Have these texels actually been produced? Used by accounting that must not FORCE the
    /// decode just to price it - reading `len()` there would defeat the whole mechanism.
    pub fn resident(&self) -> bool {
        self.0.ready.get().is_some()
    }

    /// Bytes currently held, WITHOUT forcing the decode: 0 while it has not run.
    pub fn resident_len(&self) -> usize {
        self.0.ready.get().map_or(0, |v| v.len())
    }
}

impl std::ops::Deref for Texels {
    type Target = Vec<u8>;

    /// Reading the texels IS the decode. Every existing `t.rgba.len()` / `&t.rgba` /
    /// `t.rgba.get(..)` call site keeps its meaning; what changes is when the work happens.
    fn deref(&self) -> &Vec<u8> {
        self.0.ready.get_or_init(|| match &self.0.make {
            Some(f) => f(),
            // Unreachable: `ready` sets the cell, `lazy` sets `make`. An empty buffer is the
            // only honest answer that is not a panic inside a `Deref`.
            None => Vec::new(),
        })
    }
}

impl Clone for Texels {
    fn clone(&self) -> Self {
        Texels(self.0.clone())
    }
}

impl std::fmt::Debug for Texels {
    /// Deliberately does NOT force the decode. A `{:?}` of a texture is a diagnostic, and a
    /// diagnostic that performs the expensive work it is reporting on is an instrument whose
    /// failure imitates its subject.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0.ready.get() {
            Some(v) => write!(f, "Texels({} bytes)", v.len()),
            None => write!(f, "Texels(not decoded)"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GxmTexture {
    pub key: u64,
    /// Guest address of the texture's pixel data. Identity, not content: a title that
    /// renders a pass into an offscreen buffer and samples it in a later pass binds a
    /// texture whose data pointer IS that pass's colour surface, and the guest bytes
    /// behind it are stale (the GPU wrote the real pixels, not the guest). The renderer
    /// matches this against the render targets it has drawn this frame and binds the
    /// rendered texture instead - see `GxmRenderer::encode_chain`.
    pub data_addr: u32,
    pub width: u32,
    pub height: u32,
    /// Number of `width x height` images `rgba` holds back to back: 1 normally, 6 for a
    /// cube map (in +X, -X, +Y, -Y, +Z, -Z order, which is WebGPU's array-layer order).
    pub faces: u32,
    /// The decoded texels, in the layout [`Self::texel`] names - produced on first READ, so a
    /// texture the GPU takes compressed is never expanded at all. See [`Texels`].
    pub rgba: Texels,
    /// Which layout `rgba` is in - four bytes a texel, or four HALVES a texel.
    ///
    /// # Why this seam is not always RGBA8
    /// It was, and for an IMAGE that is a defensible trade. It is not defensible for a texture
    /// whose texels are DATA, and the guest's 64-bit half-float formats are how a title stores
    /// data in a texture. Crushing a 16-bit lane to 8 bits leaves 256 distinct values per
    /// channel, and if those values are coordinates rather than colours, 256 is the number of
    /// distinct places the shader can address.
    ///
    /// MEASURED: this title's campaign map draws its labels through a "vector canvas" - a
    /// 512x128 `F16F16F16F16` (base format `0x1b`) lookup texture, sampled BY THE VERTEX
    /// PROGRAM, whose R and G lanes are each glyph's origin in a 2048x2048 atlas. Through an
    /// 8-bit seam one step of that origin is 2048/255 = 8 texels, so every glyph snapped to an
    /// 8-texel grid and the label rendered as skewed fragments. No amount of shader or sampler
    /// work could have fixed it: the information was gone at decode.
    pub texel: TexelSeam,
    /// The `SceGxmTextureFormat` base format and channel swizzle `rgba` was decoded THROUGH.
    /// Diagnostic only - the pixels are already RGBA8 by the time the renderer sees them - but
    /// it is the one fact that separates "this channel is genuinely zero in the asset" from "we
    /// decoded the asset through the wrong layout", and those need opposite fixes.
    pub base_format: u32,
    pub swizzle: u32,
    /// True if the guest set this texture's magnification filter to LINEAR
    /// (`SceGxmTextureFilter` == 1); the renderer then bilinear-samples it, matching
    /// the software rasterizer's `sample_texture_bilinear`. False = POINT/nearest.
    pub filter_linear: bool,
    /// The guest's `SceGxmTextureAddrMode` for U and V
    /// (`sceGxmTextureSet{U,V}AddrMode`, 0 = REPEAT). The recompiled path used to hardcode
    /// REPEAT for every sampler, which is invisible while every coordinate stays inside
    /// [0,1] and catastrophic the moment one does not: a full-screen pass reading one texel
    /// past the edge wraps to the OPPOSITE edge instead of clamping, and a title's composite
    /// showed the world tiled diagonally across the display rather than clipped.
    pub addr_mode_u: u32,
    pub addr_mode_v: u32,
    /// The guest set `sceGxmTextureSetGammaMode` on this texture, so the hardware sampler
    /// sRGB-DECODES every texel it fetches. Uploading through an sRGB texture format does the
    /// same decode in the same place, before filtering.
    ///
    /// The counterpart of [`RttTarget::gamma`], which is the write half of the same state.
    pub gamma: bool,
    /// The guest's own block-compressed bytes, when this texture may be handed to the GPU
    /// WITHOUT being decoded - see [`CompressedUpload`]. `None` is the ordinary case and
    /// costs nothing: `rgba` is still the decode, and the uploader uses it.
    pub compressed: Option<CompressedUpload>,
}

/// A block-compressed format WebGPU can be handed directly, and which of the guest's base
/// formats it IS.
///
/// The guest's `UBC1`/`UBC2`/`UBC3` (base formats `0x85`/`0x86`/`0x87`) are BC1/BC2/BC3 - the
/// same blocks, the same 4x4 geometry, the same bits. Nothing is transcoded here; the only
/// work between guest memory and the GPU is undoing the Morton BLOCK order on a swizzled
/// texture, which is a permutation of whole blocks and touches no bit inside one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockFormat {
    Bc1,
    Bc2,
    Bc3,
    /// ETC2 RGB8: 4 bits per texel, alpha discarded. The rate the guest's PVRTC was stored at.
    Etc2Rgb8,
    /// ETC2 RGBA8: the 64-bit EAC alpha block followed by the 64-bit colour block, 8 bpp.
    Etc2Rgba8,
}

impl BlockFormat {
    /// Bytes in one 4x4 block.
    pub fn block_bytes(self) -> u32 {
        match self {
            BlockFormat::Bc1 | BlockFormat::Etc2Rgb8 => 8,
            BlockFormat::Bc2 | BlockFormat::Bc3 | BlockFormat::Etc2Rgba8 => 16,
        }
    }

    /// The adapter feature this format needs.
    pub fn family(self) -> BlockFamily {
        match self {
            BlockFormat::Bc1 | BlockFormat::Bc2 | BlockFormat::Bc3 => BlockFamily::Bc,
            BlockFormat::Etc2Rgb8 | BlockFormat::Etc2Rgba8 => BlockFamily::Etc2,
        }
    }
}

/// Which family of block-compressed formats the GPU in use accepts.
///
/// # This is not a boolean, and treating it as one is what made the whole feature desktop-only
/// The first version of this recorded "does the adapter take BC". Every desktop said yes and the
/// device this work exists for said no, so the phone kept decoding every compressed texture to
/// RGBA8 at 4-8x - a 354 MB working set against a 256 MB budget - while the desktop reported a
/// 274 -> 57 MB win. The fact that decides the whole feature is WHICH family, not whether.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockFamily {
    None,
    Bc,
    Etc2,
}

/// The guest's compressed bytes, relaid into the order a WebGPU upload wants.
///
/// # Why this exists rather than a flag on the decode
/// A race frame of one retail title needs 260 MB of texture against a 256 MB budget, and the
/// reason is not the number of textures - it is that every compressed format is CPU-decoded to
/// RGBA8 on the way in. Past the budget the allocation fails on a phone and the draw comes out
/// WHITE, which is exactly the reported symptom, on exactly the screens that carry big
/// textures. Handing the blocks over unchanged removes the expansion instead of managing it.
#[derive(Clone, Debug)]
pub struct CompressedUpload {
    pub format: BlockFormat,
    /// The dimensions `data` actually describes.
    ///
    /// # >>> THESE ARE ALWAYS THE GUEST'S OWN SIZE TODAY, AND THAT IS A RULE, NOT A COINCIDENCE
    /// This field once carried a TIERED transcode: build the first version of a texture from the
    /// guest's small mip levels and replace it with larger tiers over the following frames, so a
    /// screen transition went soft for a moment instead of freezing. **That mechanism is
    /// DELETED** - it is the progressive-residency trade written up in
    /// [[vitaslop-never-trade-quality]], where the device never got past 128 texels a side and a
    /// 2048x2048 atlas rendered at a sixteenth of its resolution per axis for a whole run.
    ///
    /// The field stays because it is what the uploader must declare the texture at, and because
    /// declaring `t.width`/`t.height` over a smaller buffer is a validation error at best and a
    /// read past the buffer at worst. Anything that ever makes these differ from the guest's own
    /// size is shipping a lower-resolution picture and needs the argument that goes with that.
    pub width: u32,
    pub height: u32,
    /// The blocks themselves, or the recipe for the GPU to make them. See [`CompressedData`].
    pub data: CompressedData,
    /// Mip levels `data` holds per face, counting level 0. These are the GUEST'S levels: there
    /// is no box filter for a compressed block, so a passthrough either carries the chain the
    /// hardware itself samples or ships level 0 alone, and level-0-alone is the "distant road
    /// reads as white speckle" failure ([[vitaslop-textures-need-mips]]).
    pub levels: u32,
    /// True if these blocks were RE-ENCODED from decoded texels rather than being the guest's
    /// own - a second lossy step, not a handover.
    ///
    /// # Carried, not inferred from the format
    /// The first version of the working-set report worked this out from the guest base format:
    /// PVRTC has no WebGPU format, so PVRTC + compressed must mean transcoded. That held until
    /// a refused UBC2 texture started going through the transcode too, at which point the one
    /// report anyone reads to find out where the megabytes went described 45 MB of re-encoded
    /// blocks as "guest blocks, guest mips". A provenance that can be inferred today is a
    /// provenance that will be wrong later.
    pub transcoded: bool,
}

impl CompressedUpload {
    /// Bytes this upload will occupy on the GPU, whichever side produced them.
    ///
    /// A GPU-built chain is priced from its geometry rather than from a buffer, because the
    /// buffer does not exist yet on the CPU side and never will - which is the entire point.
    /// The CPU-built blocks, or `None` when the GPU will build them.
    ///
    /// Every caller of this is asserting something about the BYTES, which only exist on the CPU
    /// path. A GPU plan has no bytes to assert about by construction - that is what it is for -
    /// so the `Option` is the honest signature rather than an empty slice that would let a test
    /// pass by measuring nothing.
    pub fn cpu_bytes(&self) -> Option<&[u8]> {
        match &self.data {
            CompressedData::Cpu(v) => Some(v),
            CompressedData::Gpu(_) => None,
        }
    }

    pub fn byte_len(&self) -> usize {
        match &self.data {
            CompressedData::Cpu(v) => v.len(),
            CompressedData::Gpu(_) => {
                let bb = self.format.block_bytes() as usize;
                (0..self.levels)
                    .map(|l| {
                        let w = (self.width >> l).max(1);
                        let h = (self.height >> l).max(1);
                        (w.div_ceil(4) * h.div_ceil(4)) as usize * bb
                    })
                    .sum()
            }
        }
    }
}

/// Where a [`CompressedUpload`]'s blocks come from.
///
/// # >>> THE ENCODER IS THE MOST EXPENSIVE THING THIS EMULATOR DOES, AND IT RAN ON THE IDLE SIDE
/// A screen transition on the target device binds a hundred textures at once, every one of them
/// PVRTC, which no WebGPU adapter can be handed compressed. Each has to be decoded and re-encoded,
/// and the device's CPU encoder runs at about 1 Mtexel/s - so a single 1024x1024 atlas with its
/// chain is well over a second of frozen guest, and a transition frame MEASURED at
/// **BUILD 21,182 ms**. Over the same frame the GPU was doing essentially nothing: `pass 3.3 ms`
/// against 128 ms of CPU.
///
/// Block encoding is embarrassingly parallel over independent 4x4 blocks and needs no guest state
/// at all, so it belongs on the processor that is idle. [`CompressedData::Gpu`] carries the guest's
/// own bytes and a description of how to read them; the decode, the mip chain and the block encode
/// then happen in compute shaders and the result is copied straight into the compressed texture.
/// **Nothing crosses back to the CPU**, so there is no stall to schedule around and no readback.
///
/// It is not a quality trade in either direction: the GPU runs the same algorithm over the same
/// texels at the guest's own resolution. `gpu_etc2_matches_the_cpu_encoder` is what says so,
/// against the CPU encoder this replaces, over the same corpus its own error ceilings are written
/// from.
#[derive(Clone, Debug)]
pub enum CompressedData {
    /// Blocks already built on the CPU: level 0 first, then each smaller level, block-packed with
    /// no row padding, layer-major over the texture's faces - which is precisely what
    /// `TextureDataOrder::LayerMajor` reads, level by level, at each level's block-rounded
    /// PHYSICAL size.
    Cpu(std::sync::Arc<Vec<u8>>),
    /// The guest's raw bytes, to be decoded, mipped and encoded on the GPU.
    Gpu(GpuTranscode),
}

/// Everything the GPU needs to build a texture's blocks from the guest's own bytes.
///
/// The layout arithmetic (which level starts where, how many blocks, Morton or linear) is done
/// ONCE on the CPU, where `level_layout` and `morton_index` already live and are already tested,
/// and handed over as a table. A second implementation of that addressing in WGSL is exactly the
/// kind of duplicate that drifts, and its failure mode is a texture that decodes plausibly out of
/// the wrong bytes.
#[derive(Clone, Debug)]
pub struct GpuTranscode {
    /// The guest's pixel bytes for face 0. Single-face only - a cube map's six chains are not
    /// established, and the CPU path refuses them for the same reason.
    pub src: std::sync::Arc<[u8]>,
    /// How to read `src`.
    pub codec: SourceCodec,
    /// Texel dimensions of level 0.
    pub width: u32,
    pub height: u32,
    /// Levels the FINISHED texture will have: a full chain down to 1x1. Levels past
    /// `src_levels.len()` are box-filtered on the GPU from the level above, exactly as the CPU
    /// transcode does in RGBA8.
    pub levels: u32,
    /// The guest's own levels present in `src`, largest first.
    pub src_levels: Vec<SrcLevel>,
}

/// One guest mip level inside [`GpuTranscode::src`].
#[derive(Clone, Copy, Debug)]
pub struct SrcLevel {
    /// Byte offset of this level from the start of `src`.
    pub byte_offset: u32,
    pub width: u32,
    pub height: u32,
    /// Blocks needed to cover the level, in the SOURCE codec's block geometry (4x4 for PVRTC
    /// 4bpp and for BC, 8x4 for PVRTC 2bpp).
    pub blocks_x: u32,
    pub blocks_y: u32,
    /// The power-of-two-padded grid a SWIZZLED level's Morton addressing runs over. Equal to
    /// `blocks_x`/`blocks_y` when the level is stored linearly.
    pub padded_x: u32,
    pub padded_y: u32,
    /// Block rows are Morton-ordered over the padded grid rather than laid out linearly.
    pub swizzled: bool,
}

/// The wgpu format a [`BlockFormat`] uploads through, in its plain or sRGB twin.
///
/// The renderer's own `block_wgpu_format` is the definition; this forwards to it so the GPU
/// transcoder ([`crate::texenc`]) creates its texture through exactly the same mapping the
/// uploader would have. Two spellings of "which wgpu format is this" is how a gamma decode ends
/// up applied on one path and not the other.
#[cfg(feature = "gpu")]
pub fn block_wgpu_format_pub(f: BlockFormat, gamma: bool) -> wgpu::TextureFormat {
    gxm::block_wgpu_format_pub(f, gamma)
}

/// The compressed format the GUEST stored a texture in, as the GPU decoder needs to read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceCodec {
    /// PVRTC1 (`two = false`) or PVRTC2, at 2 or 4 bits per texel.
    Pvrtc { two: bool, four_bpp: bool },
    /// The guest's `UBC1`/`UBC2`/`UBC3` (base formats `0x85`/`0x86`/`0x87`), which ARE BC1/2/3.
    ///
    /// On a desktop these pass through untouched and this never runs. It exists for an adapter
    /// with no BC at all - the target device - where the alternatives are an RGBA8 expansion at
    /// four times the size or a CPU decode plus a CPU re-encode, which is the single most
    /// expensive path in the texture pipeline.
    Bc { base_format: u32 },
}

impl SourceCodec {
    /// Bytes in one SOURCE block. PVRTC is always 8; BC1 is 8 and BC2/BC3 are 16.
    pub fn block_bytes(self) -> u32 {
        match self {
            SourceCodec::Pvrtc { .. } | SourceCodec::Bc { base_format: 0x85 } => 8,
            SourceCodec::Bc { .. } => 16,
        }
    }
}

/// The per-material forward-lighting inputs the GPU `fs_opaque` needs to reproduce the
/// software rasterizer's `shade_lit`: a base-colour `tint`, one directional light
/// (`light_dir` world-space direction-to-light + `light_col`), and a flat `ambient`. The
/// runtime reflects these from the fragment program; neutral values (tint 1, light col 1,
/// modest ambient) make it a pass-through for a 2D/UI draw. Kept engine-neutral here so
/// `vitaslop-platform` stays free of any runtime dependency.
#[derive(Clone, Copy, Debug)]
pub struct GxmMaterial {
    pub tint: [f32; 3],
    pub light_dir: [f32; 3],
    pub light_col: [f32; 3],
    pub ambient: [f32; 3],
}

impl Default for GxmMaterial {
    fn default() -> Self {
        GxmMaterial {
            tint: [1.0, 1.0, 1.0],
            light_dir: [0.0, 1.0, 0.0],
            light_col: [1.0, 1.0, 1.0],
            ambient: [0.35, 0.35, 0.35],
        }
    }
}

/// One ready-to-draw call, CPU-decoded by the runtime into a canonical vertex
/// layout so the GPU never decodes a GXM attribute format. The vertex layout is
/// fixed: position `float32x3` at offset 0, texcoord `float32x2` at offset 12
/// (already divided by the draw's uv scale), color `unorm8x4` at offset 20, and the
/// world-space normal `float32x3` at offset 24 (baked by the runtime from the object
/// normal and the model-to-world matrix, for the opaque lighting term), stride 36.
/// Indices are always 32-bit. `texture` is `None` for an untextured draw (the renderer
/// binds a 1x1 white texel so one shader path serves both).
#[derive(Clone, Debug)]
pub struct GxmDraw {
    pub space: DrawSpace,
    pub vertices: Vec<u8>,
    pub indices: Vec<u8>,
    pub index_count: u32,
    pub texture: Option<GxmTexture>,
    /// The per-material lighting inputs for the opaque path (ignored when `opaque` is false).
    pub material: GxmMaterial,
    /// True for genuinely opaque 3D geometry: an MVP-space draw that also writes depth
    /// (`front_depth_write != SCE_GXM_DEPTH_WRITE_DISABLED`). Drives BOTH the pipeline
    /// (depth-tested opaque replace) AND the fragment combine (the albedo texel is taken
    /// straight, tone-mapped by `exposure`, alpha forced to 1). This is the software
    /// rasterizer's `depth_test` decision, carried here rather than re-derived from
    /// `space` alone - an MVP draw with depth writes DISABLED is a 2D alpha-blended
    /// overlay, not opaque, so keying opacity off MVP-space would wrongly replace and
    /// tonemap it. False = 2D overlay: `vertex_color * texel` modulate, alpha-blended in
    /// submission order.
    pub opaque: bool,
    /// Scene exposure (linear multiplier from `vsCoarseExposureReg`) applied to opaque
    /// draws before a Reinhard tonemap. 1.0 is a no-op (2D/UI and any shader with no
    /// exposure uniform), so the tonemap is skipped and the color passes through.
    pub exposure: f32,
    /// The guest's real vertex+fragment shaders + their draw inputs, for the GXP->WGSL
    /// recompiler (live guest-shader) path. `Some` only when the runtime captured it
    /// (`VITASLOP_GXP_LIVE`); the renderer links + caches a pipeline and ALWAYS falls back
    /// to the fixed-function fields above on any link/format error. See [`GxpRecompile`].
    pub gxp: Option<GxpRecompile>,
    /// This draw has NO valid fixed-function representation: its geometry carries neither a
    /// texcoord nor a per-vertex colour, so the fixed-function fields above have no colour
    /// source and would paint opaque white. Its real colour lives in the guest's fragment
    /// shader (typically a uniform), so it can only be drawn through `gxp`.
    ///
    /// The scene builder used to DISCARD such a draw outright, silently. That is what made a
    /// retail title's solid-colour UI fills - a button's interior, a panel backing - simply
    /// absent, indistinguishable from the guest never drawing them, and it survived being
    /// stared at in screenshots. Keeping the draw and marking it lets the recompiler render it
    /// exactly; if no recompiled pipeline is available the renderer skips it and SAYS SO,
    /// rather than either dropping it quietly or painting a white rectangle.
    pub shader_only: bool,
    /// The guest's `sceGxmSetRegionClip` for this draw - GXM's hardware SCISSOR. See
    /// [`RegionClip`].
    pub region_clip: RegionClip,
}

/// GXM's region clip (`sceGxmSetRegionClip`), which is the hardware SCISSOR, captured per
/// draw because it is per-draw state exactly as the viewport is.
///
/// # What the two enabled modes mean, and how that was settled
/// `SceGxmRegionClipMode` has four values in the top two bits: `NONE` (0), `ALL`
/// (`0x40000000`), `OUTSIDE` (`0x80000000`) and `INSIDE` (`0xC0000000`). The vitasdk
/// header's prose is ambiguous about which of the two enabled modes keeps the inside of
/// the rectangle and which keeps the outside, and reading it either way makes one of two
/// retail titles blank its own first scene. The titles settle it between them:
///
/// - A retail racer issues `INSIDE` with `0,0 .. 959,543` as the first call of its first
///   scene (a whole-target default), then a run of arbitrary rectangles - `9,0..959,543`,
///   then `0,0..719,543`, `539`, `404`, `302`, `226`, `169`, `127`, `95`, `71`, `53` -
///   which is a closing WIPE. **None of those is a multiple of 32.**
/// - A retail futuristic racer issues `OUTSIDE` with a whole-target rectangle and with
///   `0,0..127,63`, `0,64..63,95`, `0,96..31,127`, `32,96..63,127`, `64,64..95,95`,
///   `64,96..95,127`, `96,64..127,95`, `96,96..127,127` - a disjoint TILING of one
///   128x128 atlas, which only makes sense if each draw is confined to its rectangle.
///   **Every one of those is a multiple of 32.**
///
/// So BOTH enabled modes keep the INSIDE of the rectangle - an inverse reading of either
/// one blanks that title's whole-target default - and what separates them is GRANULARITY:
/// the header's wording is about clipping *tiles*, and only the mode used with arbitrary
/// rectangles can be the exact per-pixel one. That is a claim the data supports and it is
/// checked rather than assumed: [`report_region_clip_not_tile_aligned`] fires if a
/// tile-mode rectangle ever arrives unaligned, which is the case where the two readings
/// would produce different pixels.
///
/// `ALL` is "clip everything" and is reported rather than guessed at: it has never been
/// observed, and a mode that draws nothing is not something to infer from silence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegionClip {
    /// The raw `SceGxmRegionClipMode` word (the enum lives in the top two bits).
    pub mode: u32,
    /// `[xMin, yMin, xMax, yMax]`, INCLUSIVE, in target pixels - GXM's own convention.
    pub rect: [u32; 4],
}

impl RegionClip {
    pub const NONE: u32 = 0x0000_0000;
    pub const ALL: u32 = 0x4000_0000;
    pub const OUTSIDE: u32 = 0x8000_0000;
    pub const INSIDE: u32 = 0xC000_0000;
    /// Tile side in pixels, for the alignment CHECK only. Nothing rounds by it: the check
    /// exists so that the one case where tile-granularity and pixel-granularity differ
    /// cannot pass silently.
    const TILE: u32 = 32;

    /// This draw's scissor rectangle in pixels of a `w` x `h` attachment, as
    /// `(x, y, width, height)`, or `None` for "the whole attachment".
    ///
    /// The GXM rectangle is INCLUSIVE at both ends, so the width is `xMax - xMin + 1`.
    /// Clamped into the attachment: wgpu rejects a scissor that leaves it, and a rejected
    /// pass loses every draw in it, which is a far worse failure than a clamped rectangle.
    fn rect_in(&self, w: u32, h: u32) -> Option<(u32, u32, u32, u32)> {
        match self.mode & 0xC000_0000 {
            Self::NONE => None,
            Self::ALL => Some((0, 0, 0, 0)),
            _ => {
                let [x0, y0, x1, y1] = self.rect;
                let x = x0.min(w);
                let y = y0.min(h);
                // `x1`/`y1` are inclusive; `+1` cannot overflow a sane rectangle, and a
                // saturating add keeps a garbage one from wrapping to an empty scissor.
                let right = x1.saturating_add(1).min(w);
                let bottom = y1.saturating_add(1).min(h);
                Some((x, y, right.saturating_sub(x), bottom.saturating_sub(y)))
            }
        }
    }
}

/// Name each distinct region clip that reaches a draw, and say whether it actually narrows
/// anything.
///
/// A scissor that equals the whole target issues no `set_scissor_rect` at all (the pass
/// already starts there), so `scissor_sets` reads zero for a title that sets a whole-target
/// default and never narrows it - which is indistinguishable from a scissor that never
/// reached the renderer. One line per distinct rectangle tells those apart, and it is the
/// only way to know the capture is carrying the state at all.
fn report_region_clip_applied(clip: RegionClip, w: u32, h: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    let (mode, rect) = (clip.mode, clip.rect);
    if mode & 0xC000_0000 == RegionClip::NONE {
        return;
    }
    static SEEN: Mutex<Option<HashSet<(u32, [u32; 4])>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((mode, rect)) {
        return;
    }
    let whole = rect[0] == 0 && rect[1] == 0 && rect[2] + 1 >= w && rect[3] + 1 >= h;
    report!(
        "gxm region clip: mode {mode:#x} over {},{} .. {},{} on a {w}x{h} target - {}",
        rect[0], rect[1], rect[2], rect[3],
        if whole { "the WHOLE target, so no scissor is issued" } else { "SCISSORED" }
    );
    // The two cases that are not merely informational, checked here rather than in
    // `RegionClip::rect_in` so that the pure geometry stays pure and neither check costs a
    // lock on the per-draw path.
    if mode & 0xC000_0000 == RegionClip::ALL {
        report_region_clip_all(rect);
    }
    if mode & 0xC000_0000 == RegionClip::OUTSIDE
        && (rect[0] % RegionClip::TILE != 0
            || rect[1] % RegionClip::TILE != 0
            || (rect[2] + 1) % RegionClip::TILE != 0
            || (rect[3] + 1) % RegionClip::TILE != 0)
    {
        report_region_clip_not_tile_aligned(rect);
    }
}

/// Say so, once per distinct rectangle, that a TILE-granular region clip arrived with an
/// edge off a tile boundary - the one case where "clip whole tiles" and "clip exactly"
/// disagree, and therefore the one case where [`RegionClip`]'s reading of the two modes
/// could paint different pixels than the hardware. Every rectangle measured on two retail
/// titles is aligned, so this firing is new information and not noise.
fn report_region_clip_not_tile_aligned(rect: [u32; 4]) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<[u32; 4]>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(rect) {
        return;
    }
    report_warn!(
        "gxm region clip: the TILE-granular mode was given {},{} .. {},{}, which is not a \
         multiple of {} - so this scissor is applied EXACTLY where the hardware would have \
         clipped whole tiles, and up to {} pixels at each edge may be missing that the \
         console would have drawn.",
        rect[0], rect[1], rect[2], rect[3], RegionClip::TILE, RegionClip::TILE - 1
    );
}

/// Say so, once, that a draw asked for `SCE_GXM_REGION_CLIP_ALL`. It clips inside AND
/// outside the region, so nothing rasterises - which is indistinguishable from a bug
/// unless it is announced.
fn report_region_clip_all(rect: [u32; 4]) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    report_warn!(
        "gxm region clip: SCE_GXM_REGION_CLIP_ALL over {},{} .. {},{} - it clips both inside \
         and outside the region, so every draw under it rasterises NOTHING. That is what the \
         guest asked for; it is reported because an empty pass otherwise reads as a defect.",
        rect[0], rect[1], rect[2], rect[3]
    );
}

/// Report, once per run, that a draw with no fixed-function representation could not be drawn
/// by the recompiler either - so it is missing from the frame. Suppressed after the first so a
/// per-frame occurrence cannot flood the log; the first one carries what matters.
pub(crate) fn report_shader_only_skip(d: &GxmDraw, gxp_enabled: bool) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    // At WARN, and this one is not a style point. A title's FINAL COMPOSITE is a fullscreen
    // blit whose colour lives entirely in its fragment shader, so it is shader-only: without
    // `VITASLOP_GXP_LIVE` the whole visible frame is this one skipped draw, the headless shot
    // comes back as the flat CLEAR COLOUR, and at `debug` nothing said so. Every offscreen
    // pass still renders, so `VITASLOP_GPU_CHAIN_DIR` shows a correct world target beside a
    // blank presented frame - which reads as "the composite is broken" and is really "the
    // recompiler was never switched on". The message already names the knob; it just could
    // not be seen at the filter every documented repro command uses.
    report_warn!(
        "render: a draw with NO fixed-function representation (position-only geometry, its \
         colour lives in the guest fragment shader) is MISSING from the frame - {}. \
         {} indices, {} vertex bytes.",
        if !gxp_enabled {
            "the GXP recompiler is not enabled (set VITASLOP_GXP_LIVE)"
        } else if d.gxp.is_none() {
            "the runtime captured no shader payload for it"
        } else {
            "its shader pair did not link (the fallback above names the reason)"
        },
        d.index_count,
        d.vertices.len(),
    );
}

/// Everything the GXP->WGSL recompiler needs to draw one call with the guest's real
/// vertex+fragment shaders, snapshotted by the runtime and carried as plain data (no GPU or
/// runtime dependency). The renderer links the pair on first sight, caches the pipeline by
/// shader identity, and ALWAYS falls back to the fixed-function path on any link/format
/// error - a wrong translation never paints a pixel.
#[derive(Clone, Debug)]
pub struct GxpRecompile {
    /// The vertex `SceGxmProgram` container bytes. Shared, not owned: the capture reads each
    /// container out of guest memory once and every draw bound to it carries the same `Arc`,
    /// so building a frame's draw list does not copy a few kilobytes per draw.
    pub vprog: std::sync::Arc<[u8]>,
    /// The fragment `SceGxmProgram` container bytes. Shared for the same reason as `vprog`.
    pub fprog: std::sync::Arc<[u8]>,
    /// Raw vertex default-uniform-buffer (SA bank) bytes, as the guest wrote them.
    pub vert_sa: Vec<u8>,
    /// Raw fragment default-uniform-buffer (SA bank) bytes, as the guest wrote them.
    pub frag_sa: Vec<u8>,
    /// Guest address `frag_sa` was read from (0 = none). See `capture::Draw::frag_sa_addr`:
    /// the bytes say what the draw got, the address is what a store watch can be pointed at.
    pub frag_sa_addr: u32,
    /// The guest-memory WINDOW the recompiled vertex shader's 0xE8 memory loads read
    /// through: `(window guest base address, the window's bytes at draw time)`. `None` for
    /// a program that loads no memory; a draw whose PIPELINE declares a window but carries
    /// `None` here is DROPPED with a report rather than fed fabricated bytes.
    pub mem_window: Option<(u32, Vec<u8>)>,
    /// Raw guest vertex stream bytes (stream 0) exactly as bound.
    ///
    /// Shared with the capture that snapshotted it rather than copied: this is the whole
    /// mesh of one draw, and copying it per draw per frame measured 2.4-3.2 MB of
    /// allocate-and-free every frame on a 500-700 draw scene. Sharing also gives the
    /// buffer an identity anything derived from it can be cached against.
    pub vertices: std::sync::Arc<[u8]>,
    /// Byte stride of one guest vertex within `vertices`.
    pub vertex_stride: u32,
    /// Guest vertex attributes: stream byte offset + raw GXM format + component count, keyed
    /// to the recompiler's vertex-input `@location` by `reg_index` (the attribute base lane).
    pub attributes: Vec<GxpAttr>,
    /// The draw's index buffer, already expanded to a flat winding-normalized triangle-LIST
    /// of `u32`s. Shared, and CACHED by the builder against the guest index buffer it was
    /// expanded from: the guest's own index bytes do not change from frame to frame for
    /// static geometry, so expanding them again every frame is pure repetition.
    pub indices: std::sync::Arc<[u8]>,
    /// Number of indices.
    pub index_count: u32,
    /// True = 32-bit indices, false = 16-bit (GXM index format 0).
    pub index_u32: bool,
    /// GXM primitive type word (drives the pipeline topology).
    pub primitive: u32,
    /// Decoded textures bound per fragment sampler unit.
    pub textures: Vec<GxpTex>,
    /// Decoded textures bound per VERTEX sampler unit. Separate list, because the two stages
    /// number their units independently - and a vertex program that samples is building its
    /// geometry from what it reads, so binding the fragment's texture here draws a wrong mesh
    /// rather than shading a surface wrongly.
    pub vertex_textures: Vec<GxpTex>,
    /// Depth write enabled for this draw (GXM `front_depth_write != DISABLED`).
    pub depth_write: bool,
    /// GXM depth-compare function word (`SceGxmDepthFunc`).
    pub depth_func: u32,
    /// GXM cull-mode word (`SceGxmCullMode`).
    pub cull_mode: u32,
    /// Whether the guest left the FRAGMENT PROGRAM enabled for this draw
    /// (`sceGxmSetFrontFragmentProgramEnable`). `false` is a legal and common GXM state: the
    /// draw rasterises and updates DEPTH and STENCIL and writes no colour at all. It is how a
    /// title writes a depth prepass, and how one title clears its depth buffer mid-scene -
    /// a fullscreen triangle with `DEPTH_FUNC_ALWAYS`, depth write on and the fragment program
    /// off. Ignoring it painted that clear's shader output (opaque black) over the finished
    /// world every frame.
    pub fragment_program_enabled: bool,
    /// The bound fragment program's `SceGxmProgram*`. Diagnostic only - it names which
    /// `SceGxmFragmentProgram` a draw used, which the shader bytes cannot: a title can register
    /// one fragment shader twice with different blend equations (one title's 2D primitive-render
    /// program is registered both with a NULL `blendInfo` and with straight-alpha src-over), and
    /// the two render completely differently from identical bytes.
    pub fprog_header: u32,
    /// Whether this draw is alpha-blended (a 2D/overlay draw, not opaque geometry).
    ///
    /// A HEURISTIC read off the geometry, kept only for the fixed-function path. The
    /// recompiler path uses `blend_state`, which is the guest's own answer.
    pub blend: bool,
    /// The blend equation the guest baked into this draw's fragment program, as raw GXM enum
    /// values: `[color_mask, color_func, alpha_func, color_src, color_dst, alpha_src,
    /// alpha_dst]`.
    ///
    /// Carried as a plain array rather than the capture type so this crate stays independent
    /// of the capture crate, exactly as `depth_func` and `cull_mode` are carried.
    pub blend_state: [u8; 7],
    /// GXM viewport `[xOffset,xScale,yOffset,yScale,zOffset,zScale]` mapping the guest clip
    /// output to the framebuffer. All-zero means the guest left the default (fullscreen).
    pub viewport: [f32; 6],
}

/// One guest vertex attribute for the recompiler path: where it sits in the stream and its
/// raw GXM format, so the pipeline builds a matching `wgpu` vertex layout.
#[derive(Clone, Copy, Debug)]
pub struct GxpAttr {
    /// The vertex program's attribute resource index = the recompiler's `@location` base lane.
    pub reg_index: u16,
    /// Byte offset of this attribute within a vertex.
    pub offset: u16,
    /// Raw `SceGxmAttributeFormat` value.
    pub gxm_format: u8,
    /// Component count (1..4).
    pub components: u8,
}

/// A decoded texture bound to a specific fragment sampler unit for the recompiler path.
#[derive(Clone, Debug)]
pub struct GxpTex {
    /// The sampler unit the fragment program samples this from (`t{unit}`/`s{unit}`).
    pub unit: u8,
    /// The decoded, GPU-ready texture.
    pub tex: GxmTexture,
}

/// Stride of the canonical [`GxmDraw`] vertex: pos(12) + uv(8) + color(4) + world-normal(12).
pub const GXM_VERTEX_STRIDE: u64 = 36;

/// Where a scene's pixels land: the guest address of its colour surface and that
/// surface's size. A frame is usually several scenes - offscreen passes that render
/// shadow maps, reflections, the 3D world - followed by one that composites them onto
/// the display buffer, and the composite samples the earlier ones as TEXTURES. Matching
/// a sampled texture's data pointer against these is what lets the renderer bind what it
/// actually drew instead of the guest bytes, which the GPU never wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RttTarget {
    pub data_addr: u32,
    pub width: u32,
    pub height: u32,
    /// The guest asked for GAMMA-CORRECT writes on this surface
    /// (`sceGxmColorSurfaceSetGammaMode`), so the hardware sRGB-encodes every value the ROP
    /// stores. Rendering through an sRGB view of the same texture reproduces that exactly -
    /// including doing it AFTER blending, which is where the hardware does it too.
    pub gamma: bool,
    /// The guest asked for `SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE` on this surface, which
    /// means the surface STORES the resolved image: its stored size is the sampled size, and
    /// the pass behind it was rasterised with more samples than that per pixel. How many is
    /// [`Self::multisample`] - the two are halves of one setting and neither is readable alone.
    pub msaa_downscale: bool,
    /// The `SceGxmMultisampleMode` of the render target this pass rasterises through:
    /// 0 = NONE, 1 = 2X, 2 = 4X. See [`gxm_sample_count`] for the mapping to a GPU sample
    /// count and for why 2X cannot be reproduced exactly.
    pub multisample: u32,
}

/// The GPU sample count that reproduces a `SceGxmMultisampleMode`.
///
/// # This is MSAA, and the difference from supersampling is the whole point
/// Multisampling rasterises COVERAGE and DEPTH at N samples per pixel but invokes the
/// fragment shader ONCE per pixel, at the pixel centre, and gives every covered sample that
/// one result. Supersampling invokes it once per SAMPLE. On an ordinary colour image the two
/// look nearly identical, which is why rendering a 2x2 image and box-averaging it passed for
/// MSAA here for a while. They are not the same program:
///
/// - Inside a triangle, MSAA's N samples are N copies of one value, so the resolve returns it
///   EXACTLY. Supersampling's N samples are the shader evaluated at N different positions, so
///   the resolve returns their mean - which equals the centre value only when the shader is
///   linear in position.
/// - MEASURED, and it is why the 2x2 raster had to be left off: this title's 1024x1024 shadow
///   map is a fragment program writing a DEPTH into a colour surface. Averaging four depths
///   sampled at four positions gives a depth belonging to none of them, so the shadow
///   comparison biased towards occluded and the track's dappled sunlight flattened into
///   uniform shade over 17.6% of the frame. Under MSAA that surface's interior is bit-exact,
///   because there is one shaded value per pixel to begin with.
///
/// So the guest's request is reproduced by a real multisampled attachment, not by a finer
/// raster. That also makes it CHEAPER than the 2x2 path it replaces: one shader invocation
/// per pixel instead of four.
///
/// # 2X is an approximation and says so at the call site
/// WebGPU guarantees sample counts 1 and 4 only, so `SCE_GXM_MULTISAMPLE_2X` is served with 4
/// samples: more antialiasing than the title asked for rather than less, and never a
/// different image where the shader is what MSAA promises it is (one invocation per pixel).
pub fn gxm_sample_count(multisample_mode: u32) -> u32 {
    match multisample_mode {
        // SCE_GXM_MULTISAMPLE_2X and _4X. Anything else is NONE: the enum has no other
        // members, and an unrecognised value is not a licence to invent a sample count.
        1 | 2 => MSAA_SAMPLES,
        _ => 1,
    }
}

/// The one multisampled count this renderer builds pipelines for.
///
/// WebGPU guarantees sample counts 1 and 4 and nothing between, so honouring the guest's
/// modes takes exactly two pipeline variants rather than a family - which is what lets the
/// fixed-function pair be built eagerly instead of discovered mid-frame.
pub const MSAA_SAMPLES: u32 = 4;

/// A whole scene reduced to general draws, in submission order. The runtime builds
/// it from a captured [`Scene`](vitaslop_runtime-side); [`GxmRenderer`] draws it.
#[derive(Clone, Debug, Default)]
pub struct RenderScene {
    /// Shader pairs the guest's patcher has named, for [`GxmRenderer::encode_chain`] to prepare
    /// before it encodes anything. `(vertex container bytes, fragment container bytes)`.
    ///
    /// See `capture::Scene::precompile`: the device patches pre-compiled USSE code at
    /// `sceGxmShaderPatcherCreateFragmentProgram`, which titles call behind a loading screen,
    /// while this recompiler has to produce WGSL and have a driver compile it. Doing that at the
    /// first DRAW is what puts 50-100 ms of pipeline building inside gameplay frames.
    pub precompile: std::sync::Arc<Vec<(std::sync::Arc<[u8]>, std::sync::Arc<[u8]>)>>,
    pub draws: Vec<GxmDraw>,
    /// This scene's render target, when the guest's colour surface was resolvable.
    pub target: Option<RttTarget>,
    /// Linear depth-normalization for the opaque (Mvp) draws: the shader maps each
    /// fragment's post-divide depth `d = c.z/c.w` to `clamp((d - depth_min) * depth_scale,
    /// 0, 1)` before the depth test. The guest's captured matrices produce `d` in an
    /// arbitrary, often huge range (hundreds to thousands, not a normalized [0,1]); the
    /// builder scans the visible opaque geometry and picks `depth_min`/`depth_scale` so the
    /// whole range maps linearly across the [0,1] depth buffer at full f32 precision. This
    /// preserves the software oracle's exact depth ORDERING (a monotonic map) while keeping
    /// enough resolution to separate, e.g., a vehicle from the ground it sits on - which a
    /// saturating squash loses when both sit at a large depth. `depth_scale == 0` means no
    /// opaque geometry (or a single depth plane): every opaque fragment maps to 0.
    pub depth_min: f32,
    pub depth_scale: f32,
    /// Where the guest's `SceGxmDepthStencilSurface` for this scene puts its depth samples,
    /// or 0 when the scene had none.
    ///
    /// A later pass that reads this scene's depth - a soft-particle fade, fog, SSAO - binds a
    /// texture at exactly this address. It is the ONLY thing that tells such a sample apart
    /// from one of the colour target, because a title allocates the two next to each other:
    /// on one retail racer the world's colour is at `0x89204aa0` and its depth 256 bytes later
    /// at `0x89204ba0`, so an address-RANGE match against the colour target claims it first
    /// and the pass reads a colour where it wanted a distance.
    pub depth_addr: u32,
    /// The pixel extent a DEPTH-ONLY pass rasterises at, when this scene has no colour
    /// surface to take one from.
    ///
    /// A `SceGxmDepthStencilSurface` carries no width or height - it is a pointer, a stride
    /// and a format - so a scene that renders depth and no colour has nothing on it that
    /// says how big the pass is. Without a size such a pass cannot be given a target at all,
    /// which is why it used to be reported and skipped; and skipping it is only harmless
    /// while nothing samples the depth it was supposed to write.
    ///
    /// The extent comes from the draws' VIEWPORT, which is the guest's own statement of the
    /// pixel region it is drawing into. That is the same fallback
    /// [`Scene::adopt_viewport_extent`](vitaslop_runtime) already uses for a degenerate
    /// colour surface, and it is sound for the same reason - but it is still a derived
    /// number rather than one the guest wrote down, so a pass placed this way says so.
    pub depth_extent: Option<(u32, u32)>,
    /// The viewport-enabled draws of a depth-only pass did NOT all name the same rectangle, so
    /// [`Self::depth_extent`] is the largest of several rather than the one the guest stated.
    ///
    /// Carried separately because it is the only part of that extent worth reporting: an
    /// agreed extent is a measurement and needs no comment, while a disagreement means every
    /// later pass sampling this depth inherits a resolution no single draw asked for.
    pub depth_extent_ambiguous: bool,
}

#[cfg(feature = "gpu")]
pub use render::{CubeRenderer, DEPTH_FORMAT};

#[cfg(feature = "gpu")]
pub use gxm::{
    take_encode_work, take_prepare_split, take_sampler_bg_counts, wasm_clock_installed,
    EncodePhases, EncodeWork, PrepareSplit,
    GxmRenderer,
};

/// See [`gxm::set_wasm_clock`] - the browser installs its `performance.now()` here so the
/// renderer's own phase split is measured there too, not structurally zero.
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub use gxm::set_wasm_clock;

#[cfg(feature = "gpu")]
mod render {
    use super::{DrawBatch, CUBE_SHADER};
    use wgpu::util::DeviceExt;

    /// The depth format the cube pipeline renders with. A host that supplies its
    /// own depth attachment must use this format.
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    /// The cube render pipeline, built once against a device for a given color
    /// target format. Reused across frames and scenes. Host-agnostic: it records
    /// draws into a caller-provided render pass, so the same renderer serves the
    /// native headless texture target and the browser canvas surface.
    pub struct CubeRenderer {
        pipeline: wgpu::RenderPipeline,
        bind_layout: wgpu::BindGroupLayout,
    }

    /// GPU resources for one batch, kept alive past the render pass that uses them.
    struct Prepared {
        vertices: wgpu::Buffer,
        indices: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
        index_count: u32,
        index_u32: bool,
    }

    impl CubeRenderer {
        /// Build the pipeline for a `color_format` render target.
        pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cube"),
                source: wgpu::ShaderSource::Wgsl(CUBE_SHADER.into()),
            });

            let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniforms"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bind_layout)],
                immediate_size: 0,
            });

            // Vertex layout matches the GXM stream exactly: position float32x3 at
            // offset 0, color as 4 normalized bytes at offset 12, stride 16.
            let vertex_layout = wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Unorm8x4,
                        offset: 12,
                        shader_location: 1,
                    },
                ],
            };

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cube"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[Some(vertex_layout)],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // No culling: match the software rasterizer, which draws both
                    // windings and relies on the depth test.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });

            CubeRenderer { pipeline, bind_layout }
        }

        /// Upload GPU buffers for each batch. Kept separate from encoding so the
        /// resources outlive the render pass that references them.
        fn prepare(&self, device: &wgpu::Device, batches: &[DrawBatch]) -> Vec<Prepared> {
            batches
                .iter()
                .map(|b| {
                    let mut mvp = Vec::with_capacity(64);
                    for v in &b.mvp {
                        mvp.extend_from_slice(&v.to_le_bytes());
                    }
                    let uniform =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("mvp"),
                            contents: &mvp,
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.bind_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform.as_entire_binding(),
                        }],
                    });
                    let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("vbo"),
                        contents: &b.vertices,
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                    let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("ibo"),
                        contents: &b.indices,
                        usage: wgpu::BufferUsages::INDEX,
                    });
                    Prepared {
                        vertices,
                        indices,
                        bind_group,
                        index_count: b.index_count,
                        index_u32: b.index_u32,
                    }
                })
                .collect()
        }

        /// Encode a full scene into `encoder`: a render pass over `color_view`
        /// (cleared to `clear`) with `depth_view` (must be [`DEPTH_FORMAT`]),
        /// drawing every batch. The caller owns the target textures and, on the
        /// native path, any subsequent copy-to-buffer readback.
        pub fn encode(
            &self,
            device: &wgpu::Device,
            encoder: &mut wgpu::CommandEncoder,
            color_view: &wgpu::TextureView,
            depth_view: &wgpu::TextureView,
            batches: &[DrawBatch],
            clear: [u8; 4],
        ) {
            let prepared = self.prepare(device, batches);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear[0] as f64 / 255.0,
                            g: clear[1] as f64 / 255.0,
                            b: clear[2] as f64 / 255.0,
                            a: clear[3] as f64 / 255.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            for p in &prepared {
                let fmt = if p.index_u32 {
                    wgpu::IndexFormat::Uint32
                } else {
                    wgpu::IndexFormat::Uint16
                };
                pass.set_bind_group(0, &p.bind_group, &[]);
                pass.set_vertex_buffer(0, p.vertices.slice(..));
                pass.set_index_buffer(p.indices.slice(..), fmt);
                pass.draw_indexed(0..p.index_count, 0, 0..1);
            }
        }
    }
}

/// The general GXM renderer: the GPU twin of the runtime's software rasterizer,
/// drawing a [`RenderScene`] of textured, alpha-blended, multi-space draws.
#[cfg(feature = "gpu")]
mod gxm {
    use super::{
        gxm_sample_count, BlockFamily, BlockFormat, CompressedData, CompressedUpload, DrawSpace,
        RenderScene, TexelSeam, DEPTH_FORMAT, GXM_VERTEX_STRIDE, MSAA_SAMPLES,
    };
    // >>> HASHED WITH FxHash, NOT SipHash. Every one of these is probed per DRAW - the
    // pipeline cache, the sampler bind groups, the view cache, the packed-geometry memo -
    // and a race frame submits hundreds of draws each naming several of them. See
    // `crate::fasthash` for what that trade is and why the keys here make it a safe one.
    use crate::fasthash::{FxHashMap as HashMap, FxHashSet as HashSet};
    use wgpu::util::DeviceExt;

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
        let v = std::env::var("VITASLOP_GXM_DEPTH_ENC").unwrap_or_default();
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
                    "VITASLOP_GXP_SA",
                    "VITASLOP_GXP_KEYS",
                    "VITASLOP_GXP_EXCLUDE",
                ]
                .iter()
                .any(|k| crate::knobs::var(k).is_ok())
        })
    }

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

    /// Upper bound on the cross-frame texture caches before they are cleared wholesale
    /// (a re-upload, never incorrectness - the keys are content fingerprints, so a
    /// re-decoded atlas still hits).
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

    use super::{tex_cache_budget_bytes, GAME_RESIDENT_CEILING_MB};

    /// Bytes a decoded texture occupies once uploaded, for the cache budget. RGBA8 is what
    /// every upload path here produces, so this is exact rather than an estimate.
    fn texture_bytes(width: u32, height: u32) -> usize {
        (width.max(1) as usize) * (height.max(1) as usize) * 4
    }

    /// What a [`GxmTexture`] will ACTUALLY occupy on the GPU once [`upload_gxp_texture`] has
    /// uploaded it: every face, at the seam's real bytes per texel, including the mip chain.
    ///
    /// # Why the flat `width * height * 4` this replaces was wrong in three directions at once
    /// [`texture_bytes`] prices everything as one RGBA8 level. That is the correct answer for
    /// exactly one case and an under-estimate for the two that matter on a memory-constrained
    /// device:
    /// - **The mip chain is not counted at all.** `upload_gxp_texture` builds a full chain for
    ///   the byte seam, which is about 33% on top of level 0, on nearly every texture a title
    ///   binds.
    /// - **The half seam is counted at HALF its size.** `Rgba16Float` is 8 bytes per texel, not
    ///   4 ([[vitaslop-texel-seam-carries-data.md]]).
    /// - Cube maps upload six layers; only one was priced.
    ///
    /// All three err the SAME way - the cache believes its contents are cheaper than they are -
    /// so a budget that looks respected is being exceeded, and the engine that pays for that is
    /// the one that cannot afford it. MEASURED on the user's device: a frame reporting a 264 MB
    /// working set against a 256 MB budget, on an accounting that omits the mips.
    ///
    /// **This must track `upload_gxp_texture`'s decisions exactly.** The two are deliberately
    /// adjacent and the mip predicate is written the same way in both; a budget that prices a
    /// texture differently from the way it is uploaded is the same class of defect as a counter
    /// whose name stopped matching what it counts.
    fn uploaded_texture_bytes(bc: BlockFamily, t: &GxmTexture) -> usize {
        texture_upload_bytes(t, compressed_upload(bc, t).is_some())
    }

    /// [`uploaded_texture_bytes`] with the device decision already made, so the arithmetic can
    /// be tested without a GPU.
    fn texture_upload_bytes(t: &GxmTexture, passthrough: bool) -> usize {
        // A texture that reaches the GPU compressed occupies exactly the blocks it is made of -
        // no estimate needed, and no 4/3 either, because the chain is already counted. A
        // GPU-built chain is priced from its geometry, which is the same number the CPU one
        // reports for the same shape; see `CompressedUpload::byte_len`.
        if passthrough {
            if let Some(c) = t.compressed.as_ref() {
                return c.byte_len();
            }
        }
        let (w, h) = (t.width.max(1) as usize, t.height.max(1) as usize);
        let layers = t.faces.max(1) as usize;
        let level0 = w * h * layers * t.texel.bytes_per_texel();
        if mips_for_seam(t.texel) {
            // A full chain sums to just under 4/3 of level 0; the exact walk is not worth it
            // for a budget, and rounding UP is the safe direction for one.
            level0 * 4 / 3
        } else {
            level0
        }
    }

    /// The compressed source [`upload_gxp_texture`] will actually use, or `None` if it will
    /// decode instead.
    ///
    /// The DEVICE is the last gate: the block-compression features are requested wherever the
    /// adapter offers them, and an adapter with none takes the decode path. This is the single
    /// place that decision is made, so [`uploaded_texture_bytes`] and the uploader cannot
    /// disagree about what a texture costs - the failure mode the byte-accounting rewrite
    /// existed to remove.
    ///
    /// The check is per FAMILY, not a boolean. The runtime chooses a block format from what the
    /// adapter reported at device creation, so the two normally agree - but they are set in
    /// different processes' worth of state (a global atomic and a device query), and handing an
    /// ETC2-only device a BC block is a device-lost rather than a soft failure. Disagreement
    /// falls back to the decode instead of gambling.
    fn compressed_upload(family: BlockFamily, t: &GxmTexture) -> Option<&CompressedUpload> {
        t.compressed.as_ref().filter(|c| c.format.family() == family)
    }

    /// The wgpu format one [`BlockFormat`] uploads through, in its plain or sRGB twin.
    ///
    /// Gamma is a property of the SAMPLE, not of the blocks: a gamma-correct texture is
    /// sRGB-decoded by the hardware sampler before filtering, and an `...UnormSrgb` format puts
    /// that decode in exactly the same place. So the same bytes serve both, which is why gamma
    /// never disqualifies a passthrough.
    /// Report - once per guest base format - that the GPU transcoder declined a texture and it
    /// took the CPU decode instead.
    ///
    /// Not a fault and not a wrong picture: every shape the transcoder declines is one the decode
    /// handles correctly. It is reported because the whole point of the GPU path is that a screen
    /// transition stops costing seconds, and "the transition is still slow" has two completely
    /// different causes - the shaders never ran, or they ran and something else is the weight.
    /// Silence here would leave those indistinguishable.
    fn report_gpu_transcode_refused(base_format: u32) {
        use std::collections::HashSet;
        use std::sync::Mutex;
        static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
        let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if !g.get_or_insert_with(HashSet::new).insert(base_format) {
            return;
        }
        report_warn!(
            "gxm textures: base format {base_format:#04x} asked for a GPU transcode and the \
             transcoder declined it, so it is being decoded and re-encoded on the CPU instead. \
             The picture is the same; the cost is not.",
        );
    }

    /// [`block_wgpu_format`], reachable from outside this module. See
    /// [`super::block_wgpu_format_pub`].
    pub(super) fn block_wgpu_format_pub(f: BlockFormat, gamma: bool) -> wgpu::TextureFormat {
        block_wgpu_format(f, gamma)
    }

    fn block_wgpu_format(f: BlockFormat, gamma: bool) -> wgpu::TextureFormat {
        match (f, gamma) {
            (BlockFormat::Bc1, false) => wgpu::TextureFormat::Bc1RgbaUnorm,
            (BlockFormat::Bc1, true) => wgpu::TextureFormat::Bc1RgbaUnormSrgb,
            (BlockFormat::Bc2, false) => wgpu::TextureFormat::Bc2RgbaUnorm,
            (BlockFormat::Bc2, true) => wgpu::TextureFormat::Bc2RgbaUnormSrgb,
            (BlockFormat::Bc3, false) => wgpu::TextureFormat::Bc3RgbaUnorm,
            (BlockFormat::Bc3, true) => wgpu::TextureFormat::Bc3RgbaUnormSrgb,
            // `Etc2Rgb8A1` is deliberately not used: its one-bit alpha would silently turn a
            // soft edge into a hard one. A texture with any alpha at all takes the 8 bpp
            // `Etc2Rgba8` and its full EAC alpha block.
            (BlockFormat::Etc2Rgb8, false) => wgpu::TextureFormat::Etc2Rgb8Unorm,
            (BlockFormat::Etc2Rgb8, true) => wgpu::TextureFormat::Etc2Rgb8UnormSrgb,
            (BlockFormat::Etc2Rgba8, false) => wgpu::TextureFormat::Etc2Rgba8Unorm,
            (BlockFormat::Etc2Rgba8, true) => wgpu::TextureFormat::Etc2Rgba8UnormSrgb,
        }
    }

    /// Whether [`upload_gxp_texture`] builds a mip chain for this seam. Shared by the uploader
    /// and by [`uploaded_texture_bytes`] so the budget and the upload cannot disagree.
    fn mips_for_seam(texel: TexelSeam) -> bool {
        texel == TexelSeam::Rgba8 && crate::knobs::var("VITASLOP_GXP_MIPS").ok().as_deref() != Some("0")
    }

    /// Upper bound on the repacked-geometry cache, in distinct meshes. A frame here submits a
    /// few hundred; the cap only fires on a title whose geometry genuinely changes every
    /// frame, where the cache would not have helped anyway.
    const PACKED_CACHE_CAP: usize = 4096;

    /// Bound on `resident_i_seen`, which is only a "have I met this address before" set.
    const RESIDENT_SEEN_CAP: usize = 8192;

    /// Bound on `GxpLive::precompile_seen`, a set of ALLOCATION pairs already considered for
    /// precompilation. Far above any title's program count; clearing costs one re-scan.
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
        /// Baked into the pipeline as an empty colour write mask, so it must identify the pair
        /// - one shader used both enabled and disabled is two pipelines, not one.
        fragment_program_enabled: bool,
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
        /// Bumped whenever `ubo` is RE-created (a grow). The bind groups over it name that
        /// specific buffer, so they have to be rebuilt when it changes and - the whole point
        /// of pooling - must NOT be rebuilt when it does not.
        generation: u64,
    }

    /// One bump-allocated GPU buffer holding geometry that has NOT CHANGED since the renderer
    /// first saw it, uploaded once and left there.
    ///
    /// # Why this exists: the largest steady-state item in a gameplay frame
    /// The pass arenas are per FRAME. Every recompiled draw copies its repacked vertices and
    /// its expanded indices into them, and the whole arena is `write_buffer`ed, every frame,
    /// for every draw - including a track mesh that has been byte-identical since the race
    /// started. MEASURED on a retail race, per frame: **4.75 ms copying 6.94
    /// MB into the arenas** out of a 7.9 ms `prepare`, and 11.23 MB of buffer writes behind it.
    /// That is the biggest thing the renderer does in a frame, and 93% of it (541 draws of 582)
    /// is geometry it already had.
    ///
    /// It is also a divergence from the device, not just a cost. GXM does not upload geometry:
    /// the guest allocates its vertex and index buffers in shared memory and the GPU reads them
    /// where they lie. Nothing on the hardware is proportional to a frame's vertex VOLUME. So
    /// paying that per frame is the wrong SHAPE as well as the wrong price, which is the same
    /// argument that moved shader compilation to the patcher call.
    ///
    /// # What makes an address a sound key
    /// An entry holds a strong `Arc` to the guest stream it was built from, so that allocation
    /// cannot be freed while the entry lives and no later stream can be handed the same address.
    /// `Arc<[u8]>` has no interior mutability, so the bytes behind a live address cannot change.
    /// The lookup asserts it with `Arc::ptr_eq` anyway rather than trusting the argument: a
    /// geometry cache that is wrong draws ANOTHER MESH, confidently, and nothing reports it.
    /// This project has already paid for that once ([`GxpLive::packed`]).
    ///
    /// # Why it never grows or resets mid-frame
    /// A pass records into a command encoder that is submitted at the END of the chain, and a
    /// prepared draw carries only an OFFSET - the buffer it addresses is read at encode time.
    /// Recreating the buffer while a frame is in flight would silently re-point every draw
    /// already prepared. So `place` only ever declines, and the grow or reset it asks for
    /// happens at the top of the next `encode_chain`.
    struct ResidentHeap {
        buf: Option<wgpu::Buffer>,
        cap: u64,
        used: u64,
        /// Every slice handed out: key -> (the guest allocation that owns it, offset, length).
        slices: HashMap<(u64, usize, usize), (std::sync::Arc<[u8]>, u64, u64)>,
        /// Set by a `place` that could not fit, cleared by the frame boundary that acts on it.
        want_grow: bool,
        /// Times this heap was RESET wholesale because it filled at its budget, and frames since
        /// the last one. A reset is not a fault - a title that reaches new geometry has to
        /// displace old - but a reset every few frames is thrash, and then the budget is the
        /// finding rather than the heap.
        resets: u64,
        frames_since_reset: u64,
        /// Bytes uploaded into this heap, ever. Against the per-frame arena volume it replaced.
        uploaded: u64,
    }

    impl ResidentHeap {
        fn new() -> Self {
            ResidentHeap {
                buf: None,
                cap: 0,
                used: 0,
                slices: HashMap::default(),
                want_grow: false,
                resets: 0,
                frames_since_reset: 0,
                uploaded: 0,
            }
        }

        /// The slice `key` already owns, or `None`.
        ///
        /// A key whose stored `Arc` is not the caller's is treated as absent, never served. See
        /// the type's doc comment: that check is the whole soundness argument for an address key.
        fn get(&self, key: &(u64, usize, usize), src: &std::sync::Arc<[u8]>) -> Option<(u64, u64)> {
            let (stored, off, len) = self.slices.get(key)?;
            std::sync::Arc::ptr_eq(stored, src).then_some((*off, *len))
        }

        /// Copy `bytes` into the heap and remember them under `key`, or decline.
        ///
        /// Declining is not a failure - the caller falls back to the pass arena, which is what
        /// every draw did before this existed. It asks for a grow on the way out so the frame
        /// boundary can make room without moving a buffer a live command encoder names.
        #[allow(clippy::too_many_arguments)]
        fn place(
            &mut self,
            queue: &wgpu::Queue,
            key: (u64, usize, usize),
            src: &std::sync::Arc<[u8]>,
            bytes: &[u8],
        ) -> Option<(u64, u64)> {
            let buf = self.buf.as_ref()?;
            let len = bytes.len() as u64;
            // `write_buffer` copies whole 4-byte units and every consumer of these ranges wants
            // 4-byte alignment (an index is four bytes; a packed vertex stride is a whole number
            // of f32s). Round the RESERVATION, never the data.
            let need = len.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).max(4);
            if self.used + need > self.cap {
                self.want_grow = true;
                return None;
            }
            let off = self.used;
            // The tail past `len` is reserved and never read, so a short write is fine - but
            // `write_buffer` itself needs a whole number of copy units, so pad the DATA when the
            // caller's slice is not one. Padding here rather than at every call site keeps the
            // alignment argument in one place.
            if len == need {
                queue.write_buffer(buf, off, bytes);
            } else {
                let mut padded = bytes.to_vec();
                padded.resize(need as usize, 0);
                queue.write_buffer(buf, off, &padded);
            }
            enc(&ENC.buffer_bytes, need);
            split_add(&PREP.resident_placed_bytes, need);
            self.uploaded += need;
            self.used += need;
            self.slices.insert(key, (src.clone(), off, len));
            Some((off, len))
        }

        /// Make room, at a FRAME BOUNDARY and nowhere else. Returns the buffer being replaced,
        /// for the caller's graveyard - dropping a `wgpu::Buffer` on the web backend only makes
        /// it collectable, and the last frame's commands still name it until its submit retires.
        fn grow_or_reset(
            &mut self,
            device: &wgpu::Device,
            budget: u64,
            usage: wgpu::BufferUsages,
            label: &str,
        ) -> Option<wgpu::Buffer> {
            self.frames_since_reset += 1;
            if !self.want_grow && self.buf.is_some() {
                return None;
            }
            self.want_grow = false;
            // First use: start small and let the title's own working set size this.
            let want = if self.buf.is_none() {
                (self.cap.max(1024 * 1024)).min(budget)
            } else if self.cap < budget {
                (self.cap * 2).min(budget)
            } else {
                // Already at budget and still short: the working set does not fit. Reset rather
                // than stall forever holding geometry the title has moved on from, and SAY so -
                // a reset every few frames is thrash, and then the budget is the finding.
                report_resident_heap_reset(label, self.cap, self.slices.len(), self.frames_since_reset);
                self.resets += 1;
                self.frames_since_reset = 0;
                self.used = 0;
                self.slices.clear();
                return None;
            };
            enc(&ENC.buffers_created, 1);
            let new = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: want,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.cap = want;
            // A new buffer holds none of the old contents, so every slice is void. They are
            // re-placed as their draws come round again, which is the next frame for anything
            // the title is actually drawing.
            self.used = 0;
            self.slices.clear();
            self.buf.replace(new)
        }
    }

    /// Say, once per heap, that a resident geometry heap FILLED at its budget and was dropped.
    ///
    /// Unconditional, at `warn`, for the usual reason: this is the renderer throwing away work
    /// it will have to redo, and the alternative to saying so is a frame that is mysteriously
    /// slower than its neighbours with every counter reading healthy.
    fn report_resident_heap_reset(label: &str, cap: u64, entries: usize, frames: u64) {
        report_warn!(
            "resident geometry: the {label} heap filled at {:.1} MB ({entries} meshes) and was \
             reset after {frames} frames - every mesh it held is uploaded again as its draw comes \
             round. One reset as a title reaches new content is expected; one every few frames is \
             thrash, and then VITASLOP_RESIDENT_GEOM_MB is the number to change.",
            cap as f64 / (1024.0 * 1024.0)
        );
    }

    pub struct GxmRenderer {
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
        /// Which pass ordinal of the CURRENT chain is being encoded - the index into
        /// `gxp_arenas`. Reset by `encode_chain`, bumped by each pass that has recompiled
        /// draws.
        gxp_arena_slot: usize,
        /// Shader modules compiled AHEAD of any draw - see [`GxmRenderer::precompile_pairs`].
        /// Reported so a run says whether the preparation actually happened; a count of zero with
        /// pipelines still building mid-race means the patcher signal never arrived.
        gxp_precompiled: u32,
        /// Per-draw uniform spacing: `UNIFORM_BYTES` rounded up to the device's
        /// `min_uniform_buffer_offset_alignment` (256 by default).
        uniform_stride: u64,
        /// The caller's target colour format, needed to size the supersampled offscreen
        /// colour target (its format must match the scene pipelines' target).
        color_format: wgpu::TextureFormat,
        /// Supersample factor (1 = off, the default; N renders the scene at N x the target
        /// dimensions into an offscreen buffer and box-downsamples it into the caller's view).
        /// See [`GxmRenderer::set_supersample`] and the `resolve_*` fields below.
        ss_scale: u32,
        /// Last reported `(draws, with_payload, prepared, fixed)` recompiler summary. The
        /// summary is per-RENDER, so printing it unconditionally writes a line every frame -
        /// 60 a second in a live window. Reporting only when the tuple CHANGES keeps the whole
        /// signal (the first frame, and any draw moving between the recompiled and
        /// fixed-function paths) without the spam.
        last_gxp_summary: Option<(usize, usize, usize, usize)>,
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
        /// The lazily-(re)created offscreen supersample target (colour + depth + resolve bind
        /// group), sized to `ss_scale * surf`. Rebuilt when the scale or target size changes.
        ss_target: Option<SsTarget>,
        /// Offscreen render targets, keyed by the guest address of the colour surface the
        /// pass that fills them writes. Persistent across frames (a title reuses the same
        /// few targets every frame), rebuilt when a target's size changes.
        rtt: HashMap<u32, RttSurface>,
        /// Bumped whenever a target in `rtt` is CREATED or RE-created - which is the only event
        /// that can invalidate a `wgpu::TextureView` of one.
        ///
        /// # Why this exists: a bind group over a render target used to be rebuilt EVERY FRAME
        /// `make_sampler_bg` refused to cache any group naming a target this frame rendered, on
        /// the reading that "those views belong to textures the frame allocates". They do not.
        /// `rtt` is persistent and keyed by guest address, `ensure_rtt` rebuilds an entry only
        /// when its size, sample count or depth-readability changes, and even the snapshot
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
        /// Views of the targets already rendered in the CURRENT frame's chain, by guest
        /// address. This decides two things: which pass CLEARS a target first (the first one
        /// into it each frame) and which reads need a snapshot.
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
        rtt_rendered: HashMap<u32, wgpu::TextureView>,
        /// Fixed-function bind groups over a rendered target, keyed by (address, linear,
        /// reading-the-snapshot). The snapshot flag is part of the key because the two
        /// views are different textures - binding the live one where the snapshot is meant
        /// would make the target its own input.
        rtt_binds: HashMap<(u32, bool, bool), wgpu::BindGroup>,
        /// Addresses whose entry in `rtt_rendered` is currently the snapshot rather than
        /// the live target (the pass being encoded draws into that address).
        rtt_reads_snapshot: HashSet<u32>,
        /// Views of the guest-encoded DEPTH of the targets already rendered this frame, keyed
        /// by the guest address of the depth surface (NOT the colour one). A sampler naming
        /// one of these is asking for a distance, and must be resolved here BEFORE the
        /// colour-target range match, which would otherwise claim the address first.
        rtt_depth_rendered: HashMap<u32, wgpu::TextureView>,
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
        /// MEASURED, the committed race recipe to f9000 in desktop Chrome headless: the GPU
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
        /// merely true: **each caller creates one encoder, calls `encode_chain` ONCE on it, and
        /// submits before calling again.** All of them do - `vitaslop-desktop/retail.rs`,
        /// `vitaslop-native/wgpu_render.rs`, `vitaslop-web/lib.rs`, and the `encode` wrapper,
        /// which forwards a single scene. A caller that encoded two chains into one
        /// un-submitted encoder would have the second frame's start destroy the first frame's
        /// arenas out from under recorded commands, so if that ever becomes a shape this
        /// renderer supports, the graveyard has to be keyed to the SUBMISSION, not to the call.
        retired_buffers: Vec<wgpu::Buffer>,
        /// The pass currently being encoded writes into a target whose depth is sampled later,
        /// so its depth attachment must be STORED rather than discarded.
        keep_depth: bool,
        /// Draws in the current chain that sampled a target this frame rendered. Zero over
        /// a frame with several passes means the composite is NOT reading them, which is a
        /// different problem from the passes not being drawn - and the two look identical
        /// on screen.
        rtt_hits: usize,
        /// Last reported chain shape, so the report prints on a change rather than every
        /// frame. See the `gxm chain:` line.
        last_chain_shape: Option<String>,
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
        /// What the last [`GxmRenderer::encode`] spent, phase by phase. See [`EncodePhases`].
        last_phases: EncodePhases,
        /// The same, summed over every pass of the last [`GxmRenderer::encode_chain`] - which
        /// is the frame, and the only figure the caller's `encode` total can be compared with.
        chain_phases: EncodePhases,
    }

    /// Where one `encode` went, in milliseconds of CPU.
    ///
    /// The recompiler path renders this title's main screen in ~30 ms warm against the
    /// fixed-function path's ~8 ms, and the interesting question is WHICH SIDE that
    /// delta is on: per-draw CPU work building GPU objects, or genuinely heavier
    /// shaders on the GPU. They call for opposite fixes, so guessing is expensive.
    ///
    /// `prepare` is the scene walk - for a recompiled draw that means creating its
    /// buffers and bind groups, where the fixed-function path only appends bytes to a
    /// grow-only arena. `upload` is the arena writes, `pass` is command encoding. All
    /// three are CPU; GPU time shows up in the caller's submit-and-wait, which is why
    /// the caller times that separately. Recorded unconditionally - three `Instant`s a
    /// frame cost nothing next to the milliseconds they measure, and a diagnostic that
    /// has to be switched on is one nobody has on when they need it.
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct EncodePhases {
        pub prepare_ms: f64,
        pub upload_ms: f64,
        pub pass_ms: f64,
        /// Draws that took the recompiled path, and draws that took the fixed-function
        /// path - the denominator for any per-draw cost.
        pub gxp_draws: usize,
        pub fixed_draws: usize,
    }

    impl EncodePhases {
        /// Fold one PASS's phases into a whole-FRAME total.
        ///
        /// A frame is a CHAIN of passes, and reporting only the last one's phases describes
        /// whichever pass happened to be last - on this title's race frame that is an
        /// eleven-draw composite, so the split read `prepare 0.25 ms` while the frame was
        /// spending sixteen milliseconds encoding four hundred draws it never mentioned.
        fn add(&mut self, pass: EncodePhases) {
            self.prepare_ms += pass.prepare_ms;
            self.upload_ms += pass.upload_ms;
            self.pass_ms += pass.pass_ms;
            self.gxp_draws += pass.gxp_draws;
            self.fixed_draws += pass.fixed_draws;
        }
    }

    /// Sampler bind groups BUILT and sampler bind groups reused, since the last read.
    ///
    /// A count, not a time: the browser has no `Instant` inside `encode`, and "how many GPU
    /// objects did this frame create" is the question either way. See `GxpLive::sampler_bgs`.
    static SAMPLER_BG: std::sync::Mutex<(u64, u64)> = std::sync::Mutex::new((0, 0));

    /// Record one sampler bind group as reused (`hit`) or freshly built.
    pub(crate) fn note_sampler_bg(hit: bool) {
        let mut g = SAMPLER_BG.lock().unwrap();
        if hit {
            g.0 += 1;
        } else {
            g.1 += 1;
        }
        drop(g);
        // Sampler groups are also bind groups, and the encode tally's job is to account for
        // EVERY GPU object the frame creates. Counted in both places on purpose: this pair has
        // its own line because group2 is the one a draw can accidentally make per draw.
        enc(if hit { &ENC.bind_groups_reused } else { &ENC.bind_groups_built }, 1);
    }

    /// Take and reset `(reused, built)` sampler bind-group counts.
    pub fn take_sampler_bg_counts() -> (u64, u64) {
        let mut g = SAMPLER_BG.lock().unwrap();
        std::mem::take(&mut *g)
    }

    /// What one `encode_chain` DID, counted rather than timed.
    ///
    /// # Why a count when the phases are now timed on both engines
    /// A time says WHICH phase, never WHAT IN IT. `encode` costs 143 ms on a burst frame in the
    /// browser and 8 ms on the desktop, and the two obvious stories - "it uploads 178 MB of
    /// expanded texture" and "it makes six WebGPU calls per draw across the wasm/JS boundary"
    /// - live in the same phase, scale with different things, and have opposite fixes. Bytes
    /// and call counts tell them apart in one run; a millisecond cannot, ever.
    ///
    /// It is also the only figure that survives a comparison BETWEEN the two engines, which is
    /// the comparison this project actually makes. Same title, same frame: if one side uploads
    /// forty times the bytes or makes four times the calls, that is the finding. If both do the
    /// identical work and one is twenty times slower, the cost is per-call overhead and the fix
    /// is batching, not volume - and that conclusion is unreachable from two clocks on two
    /// machines.
    ///
    /// Every counter is a relaxed atomic add on a path that already makes a WebGPU call, so the
    /// instrument costs nothing next to what it measures. It is NOT behind a knob, for the usual
    /// reason: a diagnostic you have to switch on is one nobody has on when the surprising frame
    /// is already in front of them.
    macro_rules! encode_work {
        ($($(#[$m:meta])* $name:ident),+ $(,)?) => {
            #[derive(Default, Clone, Copy, Debug)]
            pub struct EncodeWork { $($(#[$m])* pub $name: u64,)+ }

            struct EncodeCounters { $($name: std::sync::atomic::AtomicU64,)+ }

            static ENC: EncodeCounters = EncodeCounters {
                $($name: std::sync::atomic::AtomicU64::new(0),)+
            };

            /// Take and RESET every encode counter. The caller owns the window it divides by.
            pub fn take_encode_work() -> EncodeWork {
                EncodeWork {
                    $($name: ENC.$name.swap(0, std::sync::atomic::Ordering::Relaxed),)+
                }
            }

            impl EncodeWork {
                /// Fold another tally in, so a caller can accumulate per PRESENT.
                pub fn add(&mut self, o: &EncodeWork) {
                    $(self.$name += o.$name;)+
                }
            }
        };
    }

    encode_work! {
        /// Render passes begun, and the draw/state calls inside them. Each one is a call
        /// across the wasm/JS boundary in the browser and a function call on the desktop,
        /// which is the whole reason to count them separately from the bytes.
        passes,
        draw_calls,
        pipeline_sets,
        bind_group_sets,
        vertex_buffer_sets,
        viewport_sets,
        /// `set_scissor_rect` calls, from the guest's `sceGxmSetRegionClip`. Zero on a title
        /// that never scissors, which is what makes a nonzero value here worth seeing: a
        /// scissor changes WHICH PIXELS a draw may touch, so a frame that looks wrong on a
        /// title with a nonzero count has a suspect this counter names.
        scissor_sets,
        /// Textures UPLOADED (a `write_texture`) and the RGBA8 bytes that went with them,
        /// against the ones a warm view cache served for free. This is the counter that
        /// answers "is encode the 178 MB".
        tex_uploaded,
        tex_upload_bytes,
        tex_view_cached,
        /// Times a view cache went over its BYTE budget and ran an EVICTION PASS, and how many
        /// entries those passes dropped.
        ///
        /// # These were `tex_view_clears`, and the name outlived the policy
        /// It counted WHOLESALE clears. When this cache went per-entry the counting site moved
        /// with it and the name did not, so a healthy cache shedding cold entries under steady
        /// pressure - which is the design - reported "2.10 view-cache clears per frame" and read
        /// as the cliff the rewrite had removed. The decode cache had exactly the same stale
        /// name and cost a session to the same misreading; this is the other half of that fix.
        ///
        /// **Neither of these is the cliff signal.** [`Self::tex_reuploaded_after_evict`] is.
        tex_view_evict_passes,
        tex_view_evicted,
        /// Times the FIXED-FUNCTION renderer's view cache was cleared WHOLESALE on reaching its
        /// byte budget. This is the last wholesale clear in the engine and it is a real cliff:
        /// once the working set exceeds the budget it fires part-way through every frame and
        /// throws away exactly what the rest of that frame is about to ask for.
        ///
        /// It is counted separately from the per-entry passes above precisely so the two are not
        /// added together - they are different failure shapes and only one of them is a bug. It
        /// reads zero on any title whose draws all recompile (`0 fixed`), which is why it has not
        /// been worth rewriting; a nonzero value here means a title is on the fallback path AND
        /// over budget, and then it should be rewritten the way the GXP one was.
        tex_view_wholesale_clears,
        /// Textures UPLOADED whose view this run had already EVICTED - the cache thrashing.
        ///
        /// An upload is expensive and a cold upload is unavoidable, so the count of uploads
        /// cannot separate "this frame reached new content" from "we are re-uploading what we
        /// threw away last frame". This can. A cache under healthy pressure evicts what is not
        /// coming back and leaves this near zero however many passes run.
        tex_reuploaded_after_evict,
        /// Bind groups and pipelines CREATED against ones a cache served. A GPU object per
        /// draw is the mistake this project has made twice.
        bind_groups_built,
        bind_groups_reused,
        pipelines_built,
        /// Buffers created, and bytes written into a buffer (`write_buffer` plus the
        /// create-with-contents arenas). Vertex/index/uniform volume, separable from texture
        /// volume because the two have completely different fixes.
        buffers_created,
        buffer_bytes,
        /// Buffers explicitly `destroy()`ed (the previous frame's arenas, released at the start
        /// of the next one). Reported beside `buffers_created` on purpose: the two should track
        /// each other in a steady frame, and a persistent gap between them is the signature of
        /// GPU memory the renderer has stopped naming but has not released.
        buffers_destroyed,
        /// Offscreen render targets created or resized, targets SNAPSHOTTED (a full
        /// texture-to-texture copy so a pass can sample what it also draws into) and the
        /// bytes in those copies, and depth surfaces converted to a sampleable float.
        /// All three are per-frame GPU work no draw count predicts.
        rtt_created,
        /// Offscreen render targets `destroy()`ed because the guest rebuilt them (a resize, a
        /// new depth reader, a changed sample count). Beside `rtt_created` for the same reason
        /// `buffers_destroyed` sits beside `buffers_created`: a target is several textures, so
        /// a standing gap between the two is a large leak on the one engine where Drop does not
        /// free - see [`RttSurface::destroy`].
        rtt_destroyed,
        rtt_snapshots,
        rtt_snapshot_bytes,
        depth_converts,
        /// Wholesale clears of the depth-range bind-group cache ([`GxpLive::depth_bgs`]).
        ///
        /// A nonzero value is NOT a fault - it is the bound doing its job on a screen whose
        /// depth range moves. It is reported because the alternative reading of a cache that
        /// never hits is a cache that is never bounded, and this one was: the counter is what
        /// makes "entries are being minted and never reused" visible from a capture instead of
        /// from a memory graph taken hours later.
        depth_bg_cache_clears,
        /// Textures explicitly `destroy()`ed on eviction from the recompiler's view cache. Reads
        /// zero wherever the cache never evicts (the desktop, on this title); on a device under
        /// budget pressure it should track the eviction count.
        textures_destroyed,
        /// Of the uploads above, how many handed the guest's own BLOCKS to the GPU instead of a
        /// decoded RGBA8 image.
        ///
        /// This has to be counted separately from the megabytes beside it. A working set that
        /// does not shrink has two completely different causes - the passthrough never fired
        /// (this reads 0), or it fired and the remaining formats are the weight (this reads
        /// high) - and the byte total alone cannot tell them apart.
        tex_uploaded_compressed,
        /// Of those, how many had their blocks BUILT ON THE GPU from the guest's own bytes,
        /// with no CPU decode and no CPU encode ([`crate::texenc`]).
        ///
        /// The number this is read against is `tex_gpu_encode_refused`. A transition that is
        /// still slow with this high is slow for some other reason; a transition that is still
        /// slow with the refusals high is one where the shaders declined the shapes that matter,
        /// and those two have nothing in common to fix.
        tex_encoded_on_gpu,
        /// Uploads that asked for a GPU transcode and were declined by it, falling back to the
        /// CPU decode. Never a wrong picture - only a slower one.
        tex_gpu_encode_refused,
    }

    /// Bump one encode counter. Relaxed: these are a diagnostic, and no reader depends on
    /// seeing two of them agree at an instant.
    #[inline]
    fn enc(c: &std::sync::atomic::AtomicU64, n: u64) {
        c.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }

    /// Where the milliseconds INSIDE one `prepare` go, plus the bytes each phase moved.
    ///
    /// `EncodePhases` says `prepare` is 7.66 ms of a 9.08 ms encode. That names the phase and
    /// nothing in it, and the candidates inside call for opposite fixes: hashing the whole
    /// vertex stream to key a cache is bytes-per-frame the cache exists to avoid, copying the
    /// packed result into the pass arena is bytes-per-frame the GPU genuinely needs uploaded,
    /// and building a bind group is a GPU object. A time alone cannot separate them, so the
    /// bytes ride along - the same argument `EncodeWork` is built on.
    ///
    /// # Why this one is behind a knob when no other counter here is
    /// Every other counter in this module is a relaxed add on a path that already makes a WebGPU
    /// call. These are CLOCK READS on a path that makes none: in the browser `performance.now()`
    /// is a wasm/JS boundary crossing ([[vitaslop-browser-host-call-cost]]), and six per draw
    /// across several hundred draws a frame would cost more than the phase they measure and move
    /// the number they exist to report. `VITASLOP_PREPARE_SPLIT=1` asks for them.
    macro_rules! prepare_split {
        ($($(#[$m:meta])* $name:ident),+ $(,)?) => {
            #[derive(Default, Clone, Copy, Debug)]
            pub struct PrepareSplit { $($(#[$m])* pub $name: u64,)+ }

            struct PrepareCounters { $($name: std::sync::atomic::AtomicU64,)+ }

            static PREP: PrepareCounters = PrepareCounters {
                $($name: std::sync::atomic::AtomicU64::new(0),)+
            };

            /// Take and RESET every prepare sub-counter. The caller owns the window.
            pub fn take_prepare_split() -> PrepareSplit {
                PrepareSplit {
                    $($name: PREP.$name.swap(0, std::sync::atomic::Ordering::Relaxed),)+
                }
            }

            impl PrepareSplit {
                /// Fold another tally in, so a caller can accumulate per PRESENT.
                pub fn add(&mut self, o: &PrepareSplit) {
                    $(self.$name += o.$name;)+
                }
            }
        };
    }

    prepare_split! {
        /// Nanoseconds in the cache-key and pipeline-lookup preamble.
        key_ns,
        /// Nanoseconds hashing the guest vertex stream to key the packed-vertex cache, and the
        /// bytes that hash read. This is pure cache overhead: it is paid on a HIT as well as a
        /// miss, and it scales with the frame's whole vertex volume rather than with its misses.
        hash_ns,
        hash_bytes,
        /// Nanoseconds repacking a guest vertex stream the packed cache did not have, and the
        /// guest bytes those repacks read.
        repack_ns,
        repack_bytes,
        /// Nanoseconds copying packed vertices and guest indices into the pass arenas, and the
        /// bytes copied. Paid every frame for every draw, including one whose geometry has not
        /// changed since the renderer started.
        arena_ns,
        arena_bytes,
        /// Nanoseconds pushing the two SA blocks into the pass's uniform arena.
        uni_ns,
        /// Nanoseconds in `make_sampler_bg` - group2, the one that is a GPU object per draw when
        /// its cache misses.
        sampler_ns,
        /// Nanoseconds in the group3 depth bind group.
        depth_ns,
        /// Draws that reached each of the two packed-vertex outcomes.
        packed_hits,
        packed_misses,
        /// Draws whose vertices, and draws whose indices, were bound where they already LIVE -
        /// the resident heaps ([`ResidentHeap`]) - against the bytes newly placed in them. A
        /// healthy steady state is high hit counts against near-zero placement: the placement is
        /// what a title pays as it reaches new geometry, and it should stop.
        resident_v_hits,
        resident_i_hits,
        resident_placed_bytes,
    }

    impl PrepareSplit {
        /// Whether anything was recorded at all - i.e. whether the knob was on.
        pub fn is_empty(&self) -> bool {
            self.key_ns == 0
                && self.hash_ns == 0
                && self.arena_ns == 0
                && self.sampler_ns == 0
                && self.packed_hits == 0
                && self.packed_misses == 0
        }

        /// One line, per FRAME (the caller divides), naming every sub-phase above.
        pub fn line(&self, frames: u64) -> String {
            let n = frames.max(1) as f64;
            let ms = |v: u64| v as f64 / n / 1.0e6;
            let mb = |v: u64| v as f64 / n / (1024.0 * 1024.0);
            let per = |v: u64| v as f64 / n;
            format!(
                "prepare split/frame: key {:.2} ms, vertex-hash {:.2} ms ({:.2} MB hashed), \
                 repack {:.2} ms ({:.2} MB read, {:.0} misses vs {:.0} hits), arena copy {:.2} ms \
                 ({:.2} MB), uniforms {:.2} ms, samplers {:.2} ms, depth {:.2} ms; RESIDENT \
                 {:.0} vertex + {:.0} index draws bound in place, {:.2} MB newly placed",
                ms(self.key_ns),
                ms(self.hash_ns),
                mb(self.hash_bytes),
                ms(self.repack_ns),
                mb(self.repack_bytes),
                per(self.packed_misses),
                per(self.packed_hits),
                ms(self.arena_ns),
                mb(self.arena_bytes),
                ms(self.uni_ns),
                ms(self.sampler_ns),
                ms(self.depth_ns),
                per(self.resident_v_hits),
                per(self.resident_i_hits),
                mb(self.resident_placed_bytes),
            )
        }
    }

    /// Whether [`take_prepare_split`]'s counters are being fed. Read once - see the macro above
    /// for why this instrument is the one thing here that has to be asked for.
    pub(crate) fn prepare_split_on() -> bool {
        use std::sync::OnceLock;
        static CELL: OnceLock<bool> = OnceLock::new();
        *CELL.get_or_init(|| crate::knobs::var("VITASLOP_PREPARE_SPLIT").map(|v| v.trim() != "0").unwrap_or(false))
    }

    /// A stopwatch that exists only when the split is on, so the OFF path reads no clock.
    #[inline]
    fn split_start() -> Option<Stopwatch> {
        prepare_split_on().then(Stopwatch::start)
    }

    /// Charge `t`'s elapsed time to `c`. A no-op when the split is off, because `t` is `None`.
    #[inline]
    fn split_end(t: Option<Stopwatch>, c: &std::sync::atomic::AtomicU64) {
        if let Some(t) = t {
            enc(c, (t.ms() * 1.0e6) as u64);
        }
    }

    /// Charge `n` to a split counter, only when the split is on - so a reader never sees bytes
    /// beside a zero time and reads the phase as free.
    #[inline]
    fn split_add(c: &std::sync::atomic::AtomicU64, n: u64) {
        if prepare_split_on() {
            enc(c, n);
        }
    }

    impl EncodeWork {
        /// One line, per FRAME (the caller divides), naming every unit above.
        pub fn line(&self, frames: u64) -> String {
            let n = frames.max(1) as f64;
            let mb = |v: u64| v as f64 / n / (1024.0 * 1024.0);
            let per = |v: u64| v as f64 / n;
            format!(
                "encode work/frame: {:.1} passes, {:.0} draws, {:.0} pipeline + {:.0} bind-group \
                 + {:.0} vertex-buffer + {:.0} viewport + {:.0} scissor sets, textures {:.1} UPLOADED \
                 ({:.2} MB, {:.1} COMPRESSED passthrough, {:.1} ENCODED ON THE GPU, {:.1} GPU \
                 encodes refused, {:.1} RE-uploaded after eviction) / {:.1} cached ({:.2} view evict \
                 passes dropping {:.1} entries, {:.2} WHOLESALE clears, {:.1} DESTROYED), bind groups {:.1} built \
                 / {:.1} reused, {:.2} pipelines built, buffers {:.1} created / {:.1} destroyed ({:.2} MB \
                 written), rtt {:.2} created / {:.2} destroyed / {:.2} snapshots ({:.2} MB) / {:.2} depth \
                 conversions, {:.2} depth-bind-cache clears",
                per(self.passes),
                per(self.draw_calls),
                per(self.pipeline_sets),
                per(self.bind_group_sets),
                per(self.vertex_buffer_sets),
                per(self.viewport_sets),
                per(self.scissor_sets),
                per(self.tex_uploaded),
                mb(self.tex_upload_bytes),
                per(self.tex_uploaded_compressed),
                per(self.tex_encoded_on_gpu),
                per(self.tex_gpu_encode_refused),
                per(self.tex_reuploaded_after_evict),
                per(self.tex_view_cached),
                per(self.tex_view_evict_passes),
                per(self.tex_view_evicted),
                per(self.tex_view_wholesale_clears),
                per(self.textures_destroyed),
                per(self.bind_groups_built),
                per(self.bind_groups_reused),
                per(self.pipelines_built),
                per(self.buffers_created),
                per(self.buffers_destroyed),
                mb(self.buffer_bytes),
                per(self.rtt_created),
                per(self.rtt_destroyed),
                per(self.rtt_snapshots),
                mb(self.rtt_snapshot_bytes),
                per(self.depth_converts),
                per(self.depth_bg_cache_clears),
            )
        }
    }

    /// The offscreen supersample render target: an `ss_scale * surf` colour + depth buffer the
    /// scene is rendered into, plus the bind group the resolve pass samples the colour through.
    /// Kept across frames and rebuilt only when the scale or the target dimensions change.
    struct SsTarget {
        scale: u32,
        width: u32,
        height: u32,
        _color: wgpu::Texture,
        color_view: wgpu::TextureView,
        depth_view: wgpu::TextureView,
        resolve_bind: wgpu::BindGroup,
    }

    /// A scene whose colour surface could not be resolved has nowhere to be drawn, and
    /// whatever it drew is missing from the frame. Reported the first time it happens (a
    /// per-frame pass would otherwise flood the log), unconditionally - a pass silently
    /// vanishing is indistinguishable from the guest never submitting it, and that
    /// ambiguity has cost this project whole sessions.
    ///
    /// `depth_addr` and `depth_is_sampled` are the two facts that decide whether this
    /// matters. A colour-less scene is usually a DEPTH-ONLY pass - a shadow map, a depth
    /// prepass - and it is only a defect when something later in the frame BINDS that depth
    /// as a texture, because then a real read is being served stale guest bytes. Without
    /// them the report says a pass vanished and leaves the reader with no way to find out
    /// whether anything wanted it, which is how this one has sat open with "no confirmed
    /// visible symptom" for several sessions.
    fn report_unplaced_scene(
        draws: usize,
        depth_addr: u32,
        depth_is_sampled: bool,
        had_extent: bool,
    ) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static REPORTED: AtomicBool = AtomicBool::new(false);
        if !REPORTED.swap(true, Ordering::Relaxed) {
            if depth_is_sampled {
                // Reaching here now means the depth-only path could not be taken, and there is
                // exactly one reason left: no extent. A `SceGxmDepthStencilSurface` carries
                // none, so the extent comes from the draws' viewport, and a pass whose draws
                // disable the viewport (or set a degenerate one) leaves nothing to size a
                // target from. Naming that is the difference between a defect someone can act
                // on and a pass that vanished.
                report_warn!(
                    "gxm: a scene with {draws} draws has no resolvable colour surface, so it \
                     renders NOWHERE - and a later pass in the same frame SAMPLES its depth at \
                     {depth_addr:#x}, which therefore reads stale guest bytes rather than this \
                     pass's depth. It could not be given a depth-only target because its extent \
                     is unknown (had_extent={had_extent}): a DepthSurface carries no extent, and \
                     this scene's draws do not supply an enabled, non-degenerate viewport to \
                     take one from."
                );
                return;
            }
            report!(
                "gxm: a scene with {draws} draws has no resolvable colour surface - it renders \
                 nowhere. Nothing in this frame samples its depth ({depth_addr:#x}), so as far \
                 as this frame is concerned nothing is missing"
            );
        }
    }

    /// Say - once per depth address - that a colour-less pass was placed into a depth-only
    /// target, and at what size.
    ///
    /// The size is the point. It is DERIVED from the draws' viewport rather than read out of a
    /// guest struct (a `SceGxmDepthStencilSurface` has no extent), and every later pass that
    /// samples this depth inherits that resolution in its own texel size and screen-space
    /// bias. A derived resolution nobody can see is exactly the kind of thing that makes a
    /// sampling bias look wrong somewhere else entirely.
    fn report_depth_only_pass(
        depth_addr: u32,
        width: u32,
        height: u32,
        draws: usize,
        ambiguous: bool,
    ) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert(depth_addr) {
            return;
        }
        // >>> THE LEVEL IS DECIDED BY WHETHER THE SIZE IS KNOWN, NOT BY WHETHER IT IS DERIVED.
        //
        // This used to warn unconditionally because the extent came from `draws.first()` and so
        // was, honestly, a guess. It is not a guess any more: the builder now reads the
        // viewport of EVERY draw in the pass, and `ambiguous` says whether they disagreed. When
        // they all name one rectangle that rectangle is the guest's own statement of the region
        // and there is nothing to report - the accommodation simply worked. When they disagree,
        // the target is the largest of several, every later pass sampling this depth inherits a
        // resolution no single draw asked for, and that is a real defect that says so.
        //
        // Lowering the agreed case is NOT silencing: the check that made it a guess was
        // replaced, so the thing being reported stopped existing. Silencing would have been
        // lowering it while `draws.first()` still decided the answer.
        if ambiguous {
            report_warn!(
                "gxm depth-only pass {depth_addr:#x}: {draws} draws with no colour surface, and \
                 they DISAGREE about the viewport - the depth-only target is {width}x{height}, \
                 the largest of several, so it is a size no single draw asked for and every \
                 later pass sampling this depth inherits it. A DepthSurface carries no extent, \
                 so there is no other source to check this against."
            );
        } else {
            report!(
                "gxm depth-only pass {depth_addr:#x}: {draws} draws with no colour surface, \
                 placed into a depth-only target at {width}x{height} - the extent every \
                 viewport-enabled draw in the pass agrees on (a DepthSurface carries no extent)."
            );
        }
    }

    /// Say so - once, unconditionally - when the frame's BIGGEST pass is neither the display
    /// pass nor sampled by anything, because then the thing that took the most work to draw is
    /// the one thing the picture cannot contain.
    ///
    /// The display is defined as the LAST scene's colour address, and everything else is an
    /// offscreen target reachable only by a later pass that SAMPLES it. That is right for a
    /// title that renders its world offscreen and composites it. It is wrong the moment a
    /// title draws its world straight into one of several DISPLAY buffers and ends the frame
    /// with a small pass into a different one: the world then goes to an offscreen target
    /// nothing reads, and the frame is the last little pass over an empty screen.
    ///
    /// That is not hypothetical. It is what a black world with a correct HUD looks like, and
    /// the shape is invisible from the frame - a pass that renders nowhere and a pass the
    /// guest never submitted produce the same picture. So this reports the addresses and lets
    /// the reader see two display-sized buffers in one frame.
    fn report_world_not_on_display(
        scenes: &[RenderScene],
        display: Option<u32>,
        sampled: &HashSet<u32>,
    ) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static REPORTED: AtomicBool = AtomicBool::new(false);
        // NOT named `display`: `tracing`'s macros bring their own `display()` helper into
        // scope, so a local of that name is shadowed inside the format string and the error
        // it produces names a trait bound, not a shadowing.
        let Some(display_addr) = display else { return };
        let Some(biggest) = scenes.iter().max_by_key(|s| s.draws.len()) else { return };
        let Some(t) = biggest.target else { return };
        if t.data_addr == display_addr || sampled.contains(&t.data_addr) || biggest.draws.len() < 2
        {
            return;
        }
        if REPORTED.swap(true, Ordering::Relaxed) {
            return;
        }
        let targets: Vec<String> = scenes
            .iter()
            .map(|s| match s.target {
                Some(t) => format!("{:#x}:{}x{}({} draws)", t.data_addr, t.width, t.height, s.draws.len()),
                None => format!("none({} draws)", s.draws.len()),
            })
            .collect();
        report_warn!(
            "gxm: the frame's BIGGEST pass ({} draws into {:#x}, {}x{}) is neither the display \
             ({:#x}) nor sampled by anything, so it renders into a target nothing reads and is \
             MISSING from the picture. The display is taken to be the LAST scene's colour \
             address; if two of these are display-sized the frame is being assembled from \
             passes that do not belong to one image. Passes: [{}]",
            biggest.draws.len(), t.data_addr, t.width, t.height, display_addr, targets.join(", ")
        );
    }

    /// One offscreen render target: the colour buffer a pass draws into and a matching
    /// depth buffer. Held by the guest address of the colour surface, so the pass that
    /// samples it later can be matched to it by the texture's data pointer.
    struct RttSurface {
        width: u32,
        height: u32,
        color: wgpu::Texture,
        color_view: wgpu::TextureView,
        /// An sRGB view of the SAME `color` texture, for a surface the guest put in
        /// gamma-correct mode. Rendering through it makes the ROP sRGB-encode every store
        /// after blending, and sampling through it decodes on read - which is what the
        /// hardware does at both ends. `None` when the colour format has no sRGB twin.
        color_view_srgb: Option<wgpu::TextureView>,
        /// Whether the GUEST put this surface in gamma-correct mode, i.e. whether the bytes in
        /// `color` are sRGB-ENCODED and a sampler must decode them on the way back in.
        ///
        /// # Not the same question as "does an sRGB view exist"
        /// `color_view_srgb` is built whenever the colour format HAS an sRGB twin, whatever the
        /// guest asked for, so it cannot stand in for this. Keeping the guest's answer here is
        /// what lets a pass sampling this target on a LATER frame make the same choice the pass
        /// that rendered it made - see `encode_chain`'s `sample_views`, and the note there for
        /// what it cost when the two disagreed.
        gamma: bool,
        /// The depth ATTACHMENT texture. Held rather than left to its view so it can be
        /// `destroy()`ed with the rest of the target - see `RttSurface::destroy`.
        depth: wgpu::Texture,
        depth_view: wgpu::TextureView,
        /// A copy of `color` as it stood before the pass now drawing into it, made only
        /// when that pass also SAMPLES the buffer - see `GxmRenderer::snapshot_rtt`.
        ///
        /// Two views over the one copy, for the same reason `color` carries two: the copy is
        /// made with `copy_texture_to_texture`, which moves BYTES, so a snapshot of a
        /// gamma-correct target holds sRGB-ENCODED bytes and a sampler must decode them on the
        /// way back in. Handing out the linear view there reads 0.5 back as 0.73, and this is
        /// the FEEDBACK path by construction - the snapshot exists only for a pass that samples
        /// the buffer it is drawing into - so the error re-applies every iteration and walks the
        /// image to white. That is the same defect `sample_views` documents on the cross-frame
        /// path; it lived here too, on the within-frame one.
        shadow: Option<(wgpu::Texture, wgpu::TextureView, Option<wgpu::TextureView>)>,
        /// This target's depth, re-encoded the way the GUEST's depth buffer holds it, for a
        /// later pass that samples it. Present only when some pass in the frame actually
        /// names this scene's depth address - see `GxmRenderer::encode_chain`.
        gxm_depth: Option<GxmDepthTarget>,
        /// The MULTISAMPLED attachments this target is rasterised into when the guest created
        /// its render target with `SCE_GXM_MULTISAMPLE_2X`/`_4X`, resolved into `color` by the
        /// pass itself. `None` when the guest asked for one sample, which is the common case.
        ///
        /// Everything outside the pass keeps reading `color` at the STORED size: that is what
        /// later passes sample, what guest coordinates are expressed in, and what the guest's
        /// own structs describe. The multisampled buffer exists only inside the pass.
        msaa: Option<MsaaAttachments>,
    }

    impl RttSurface {
        /// Release every GPU allocation this target owns.
        ///
        /// # Drop is not enough, and only one engine can see that
        /// On a native backend dropping the handle returns the allocation. In the browser a
        /// `wgpu::Texture` IS a JavaScript `GPUTexture`, so dropping it only makes it GARBAGE
        /// and the memory comes back whenever the JS collector decides to run. That is the
        /// same defect the per-pass buffer arenas and the evicted texture views were fixed for;
        /// render targets were the third owner of GPU memory and were still relying on Drop.
        ///
        /// A render target is REBUILT whenever the guest resizes it, gains a depth reader or
        /// changes its sample count, so this is not a shutdown-only path - it runs at screen
        /// transitions, which on a memory-tight device is exactly when the headroom is needed.
        fn destroy(&self) {
            self.color.destroy();
            self.depth.destroy();
            if let Some((tex, _, _)) = &self.shadow {
                tex.destroy();
            }
            if let Some(d) = &self.gxm_depth {
                d.tex.destroy();
            }
            if let Some(m) = &self.msaa {
                m.color.destroy();
                m.depth.destroy();
            }
        }
    }

    /// The multisampled colour + depth a target rasterises into. There is no resolve bind
    /// group and no resolve pass: a multisampled colour attachment names the stored-size
    /// texture as its `resolve_target` and the hardware resolves it as part of ending the
    /// pass, which is also where the hardware does it.
    struct MsaaAttachments {
        /// The textures themselves, kept only so they can be `destroy()`ed when the target is
        /// rebuilt. In the browser a `wgpu::Texture` IS a `GPUTexture`, so dropping the Rust
        /// handle makes it GARBAGE rather than freeing it - see `RttSurface::destroy`.
        color: wgpu::Texture,
        depth: wgpu::Texture,
        color_view: wgpu::TextureView,
        depth_view: wgpu::TextureView,
        /// Samples per pixel, so a pipeline built for this pass matches its attachments.
        samples: u32,
    }

    /// A/B instrument: force every pass to ONE sample, whatever the guest asked for.
    ///
    /// This is not a quality setting and there is no per-platform default to pick. The sample
    /// count is the guest's own `SceGxmRenderTargetParams::multisampleMode` and it is honoured
    /// on every engine - see [`gxm_sample_count`] for why multisampling (one shader invocation
    /// per pixel) is the faithful reading of that request and supersampling is not. What this
    /// exists for is the measurement: a render change that cannot be turned off cannot be
    /// priced, and the phone is where the price is.
    fn no_multisample() -> bool {
        use std::sync::OnceLock;
        static OFF: OnceLock<bool> = OnceLock::new();
        *OFF.get_or_init(|| crate::knobs::flag("VITASLOP_GXM_NO_MULTISAMPLE"))
    }

    /// Say - once per target - that a target the guest created MULTISAMPLED got a multisampled
    /// attachment, and at how many samples. The refusals are [`report_multisample_refused`].
    ///
    /// The guest asking and us obliging used to be indistinguishable in a log: the runtime
    /// reported the request unconditionally and the renderer silently ignored it, so a reader
    /// could see the ask and had no way to learn the answer.
    fn report_multisample_granted(addr: u32, w: u32, h: u32, mode: u32, samples: u32) {
        use std::collections::HashSet;
        use std::sync::Mutex;
        static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
        let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if !g.get_or_insert_with(HashSet::new).insert(addr) {
            return;
        }
        // 2X served with 4 samples is the one place this path is not exact, so it is named
        // every time rather than folded into the same sentence as an exact grant.
        let exact = if mode == 1 {
            " - the guest asked for 2X and WebGPU guarantees only 1 and 4, so this pass is \
             antialiased MORE finely than the title asked"
        } else {
            ""
        };
        report!("gxm rtt {addr:#x} ({w}x{h}): MULTISAMPLE granted - {samples} samples per pixel, resolved into the stored size at end of pass{exact}");
    }

    /// Say - once per target - that a target the guest created MULTISAMPLED was rasterised at
    /// one sample anyway, and which reason applied. Every refusal here is deliberate and
    /// narrow, which is exactly the kind of quiet exception that later reads as a bug
    /// somewhere else.
    fn report_multisample_refused(addr: u32, w: u32, h: u32, gamma: bool) {
        use std::collections::HashSet;
        use std::sync::Mutex;
        static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
        let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if !g.get_or_insert_with(HashSet::new).insert(addr) {
            return;
        }
        let why = if gamma {
            "it is a GAMMA-CORRECT surface, where the ROP sRGB-encodes each store after \
             blending, and whether the resolve averages the encoded or the linear values has \
             not been measured here"
        } else {
            "a later pass samples this target's DEPTH, and a multisampled depth attachment \
             cannot be fed to the conversion pass that matches the guest's own depth buffer \
             ([[vitaslop-depth-as-texture]] is emphatic that depth has to match EXACTLY)"
        };
        // `report_warn`, not `report`: this one says the output is APPROXIMATED, which is the
        // documented job of the warn-level macro, and a refusal that only shows up at
        // `vitaslop::gxm=debug` is a refusal nobody reads. The grant above stays at debug -
        // it says the renderer did what it was told, which is not news.
        report_warn!("gxm rtt {addr:#x} ({w}x{h}): MULTISAMPLE REFUSED - {why}. This pass stays more aliased than the title asked.");
    }

    /// Say - once per target - that the guest created a render target MULTISAMPLED but did NOT
    /// put its colour surface in `SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE`.
    ///
    /// The two together mean "rasterise at N samples, store the resolve". Multisampled WITHOUT
    /// the scale mode means the surface's own memory is expected to hold the raw samples, and a
    /// resolved image is then the wrong bytes at the wrong size for anything that reads it. We
    /// always resolve, so this is a real divergence rather than a curiosity - and it warns
    /// rather than reports, because nothing downstream would look wrong in an obvious way.
    fn report_unresolved_multisample_surface(addr: u32, w: u32, h: u32) {
        use std::collections::HashSet;
        use std::sync::Mutex;
        static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
        let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if !g.get_or_insert_with(HashSet::new).insert(addr) {
            return;
        }
        report_warn!(
            "gxm rtt {addr:#x} ({w}x{h}): the render target is MULTISAMPLED but its colour \
             surface is NOT in MSAA_DOWNSCALE, so the guest expects this memory to hold the raw \
             SAMPLES - we store the resolved image instead"
        );
    }

    /// A render target's depth in the guest's own encoding, plus what it takes to produce and
    /// sample it.
    struct GxmDepthTarget {
        /// A sampleable view of the depth ATTACHMENT (the conversion pass's input).
        src_view: wgpu::TextureView,
        /// The converted R32Float texture (the output), and the view a draw samples it through.
        /// The texture is held (not just its view) so it can be `destroy()`ed with the target -
        /// see `RttSurface::destroy`.
        tex: wgpu::Texture,
        view: wgpu::TextureView,
    }

    /// Round `v` up to a multiple of `align` (a power of two).
    fn align_up(v: u64, align: u64) -> u64 {
        (v + align - 1) & !(align - 1)
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

    /// A linked + compiled pipeline for one guest shader pair, cached by shader identity.
    struct GxpPipeline {
        /// The one pipeline for this pair. There used to be two - an "opaque" variant with
        /// LessEqual + depth write and an "overlay" variant with Always + no write - selected
        /// per draw by a HEURISTIC (see the note in `build_gxp_pipeline`). Depth now comes from
        /// the guest's own captured `SceGxmDepthFunc`/depth-write, which is part of the pair
        /// key, so the two variants would be identical by construction.
        pipeline: wgpu::RenderPipeline,
        /// group0 = vertex SA uniform, group1 = fragment SA uniform, group2 = samplers.
        /// Empty layouts where the stage declares nothing, so the pipeline layout still
        /// covers every group index the WGSL might reference.
        layouts: [wgpu::BindGroupLayout; 4],
        /// Vertex SA scalar-lane count (the group0 uniform holds `ceil(n/4)` vec4s).
        vsa_lanes: u32,
        /// Fragment SA scalar-lane count.
        fsa_lanes: u32,
        /// Size in bytes of the vertex stage's guest-MEMORY-WINDOW uniform at group 0
        /// binding 1 (one header vec4 + the window bytes - see
        /// `vitaslop_gxp_shader::MemWindow::vec4_count`), or 0 when the pair's vertex
        /// program loads no memory. A draw for a pipeline with a window must carry the
        /// window's bytes or be DROPPED with a report.
        mem_bind_bytes: u32,
        /// `(sampler unit, is_3d)` per group2 sampler, in binding order.
        samplers: Vec<(u8, SamplerDim)>,
        /// The VERTEX stage's sampler units, in group-4 binding order.
        vertex_samplers: Vec<(u8, SamplerDim)>,
        /// How to repack the guest vertex stream into the tightly-packed `Float32xN` buffer
        /// this pipeline's vertex layout expects (one entry per attribute), + the packed
        /// stride. Repacking to f32 on the CPU sidesteps wgpu's vertex-format gaps (no
        /// `Float16x3`, etc.) and matches the recompiled shader, which reads f32 anyway.
        repack: Vec<RepackAttr>,
        packed_stride: u32,
    }

    /// One attribute's recipe for repacking the guest vertex stream to packed f32.
    struct RepackAttr {
        guest_offset: u32,
        gxm_format: u8,
        /// Components the GUEST's stream supplies.
        components: u8,
        /// Components the packed slot carries - the SHADER's declared width. Anything above
        /// `components` is the fill (see `attr_fill`).
        slots: u8,
        packed_offset: u32,
    }

    /// Per-draw GPU resources for one recompiled draw, kept alive through the render pass.
    struct GxpPrepared {
        /// Cache key of the pipeline this draw uses (looked up immutably during the pass).
        key: u64,
        /// Byte range of this draw's repacked vertices and its indices inside the pass's
        /// grow-only GXP arenas.
        ///
        /// NOT per-draw buffers. Creating a vertex and an index buffer per draw is two GPU
        /// allocations per draw, and a race frame here submits ~470 recompiled draws: measured,
        /// that per-draw allocation was 14.6 ms of the 15.2 ms `encode`, i.e. the single
        /// largest item in the whole system. The fixed-function path has always packed into
        /// shared arenas uploaded once; this is the same thing for the recompiled path.
        v_off: u64,
        v_len: u64,
        i_off: u64,
        i_len: u64,
        /// Whether those ranges address the RESIDENT heap ([`GxpLive::resident`]) rather than
        /// this pass's arena. The two are independent: a mesh whose vertices never change can
        /// still be re-indexed every frame, and a title that rebuilds a vertex stream can hold
        /// its index list still.
        v_resident: bool,
        i_resident: bool,
        index_count: u32,
        /// Byte offsets of this draw's vertex and fragment SA blocks inside the pass's uniform
        /// arena, for the group0/group1 DYNAMIC offsets, plus the vertex stage's
        /// guest-memory window block (third slot, meaningful only when the pipeline's
        /// `mem_bind_bytes` is non-zero). The bind groups themselves belong to the shader
        /// PAIR, not the draw ([`GxpLive::ubo_bgs`]).
        u_off: [u32; 3],
        /// Bind groups for group2 (samplers) and group3 (the pass depth block).
        bg2: wgpu::BindGroup,
        bg3: wgpu::BindGroup,
        /// True = alpha-blended (2D/overlay), false = opaque geometry.
        blend: bool,
        /// The guest's GXM viewport for this draw, `[xOffset,xScale,yOffset,yScale,zOffset,
        /// zScale]`. All-zero means the guest left the default (the whole target).
        viewport: [f32; 6],
        /// The attachment format this draw's pipeline was built for - part of the pipeline
        /// cache key, because a gamma-correct surface renders through an sRGB view.
        format: wgpu::TextureFormat,
        /// Samples per pixel of the attachments this draw's pipeline was built for, and part
        /// of the cache key for the same reason as `format`: a pipeline may only be bound in a
        /// pass whose attachments match its sample count. The SAME shader pair is used by this
        /// title on both a 4-sample world target and a 1-sample display pass, so this is not a
        /// theoretical distinction - without it the second pass binds the first one's pipeline
        /// and the driver rejects the pass.
        samples: u32,
        /// The guest's `SceGxmCullMode` for this draw, and part of the cache key for the same
        /// reason: it is baked into the pipeline's primitive state and a title sets it PER
        /// DRAW - one retail racer asks for `CW` on its world and `NONE` on its overlays.
        cull: u32,
    }

    /// One entry in the submission-order draw plan: either a fixed-function [`Item`] (by index
    /// into the arena-packed `items`) or a recompiled draw (by index into `gxp_prepared`). The
    /// two kinds interleave in the one render pass so depth and overlay ordering stay correct.
    enum Enc {
        Fixed(usize),
        Gxp(usize),
    }

    /// The live recompiler's pipeline cache + config. Held by [`GxmRenderer`].
    struct GxpLive {
        /// Master switch (`VITASLOP_GXP_LIVE`).
        enabled: bool,
        /// Render ONLY recompiled draws, skipping the fixed-function draw for any call that
        /// has a working recompiled pipeline (`VITASLOP_GXP_ONLY`) - isolates the recompiler
        /// output for review.
        only: bool,
        /// Apply the GXM (GL-style, NDC z in [-1,1]) -> WebGPU (z in [0,1]) clip-depth remap
        /// in the vertex output (`VITASLOP_GXP_ZFIX`, default on). Off passes clip z straight.
        zfix: ZFix,
        /// Flip clip Y (`VITASLOP_GXP_YFLIP`, default off). The fixed-function MVP path passes
        /// guest clip X/Y straight to WebGPU and renders upright, so the real shaders should
        /// too; the toggle is here for empirical confirmation.
        yflip: bool,
        /// Diagnostic (`VITASLOP_GXP_FORCE`): bind a neutral fallback texture for a sampler
        /// unit whose real texture we could not capture/decode (e.g. a 3D/LUT format) or a 3D
        /// sampler, so the recompiled GEOMETRY renders and the vertex/clip/depth path can be
        /// validated even before the special textures are handled. Off = strict (fall back to
        /// fixed-function rather than sample a wrong texel).
        force: bool,
        /// Diagnostic (`VITASLOP_GXP_SOLID`): every recompiled draw outputs solid magenta with
        /// the depth test disabled, to answer "does the geometry rasterize on-screen at all?"
        /// independent of fragment shading and depth. Magenta visible -> vertex/clip is right.
        solid: bool,
        /// Diagnostic (`VITASLOP_GXP_NODEPTH`): every recompiled draw keeps its real shading and
        /// blending but stops testing depth. `solid` answers "does this geometry rasterize"
        /// while changing BOTH the shading and the depth test, so a surface that comes out
        /// black under it is still ambiguous - the fragment could be painting black, or it
        /// could be losing the depth test. This changes only the depth test, which separates
        /// the two: a surface that appears here and not in an ordinary run is depth-rejected,
        /// one that stays black is shaded black.
        nodepth: bool,
        /// Diagnostic (`VITASLOP_GXP_NOBLEND`): force every recompiled pipeline to REPLACE with
        /// a full colour write mask, changing NOTHING else. The counterpart of `nodepth` for the
        /// other way a correctly-shaded draw can leave no mark: a shader that writes alpha 0
        /// under a src-alpha blend, or a guest colour mask that writes no channels.
        noblend: bool,
        /// Diagnostic (`VITASLOP_GXP_KEYS=<hex>,<hex>`): recompile ONLY these shader-pair keys
        /// (the `gxp draw key` value `VITASLOP_GXP_DUMP` prints), letting every other draw fall
        /// back. Rendering one pair at a time is how a visual artifact is attributed to the
        /// shader that produced it. Empty = no filter (recompile every linkable pair).
        keys: Vec<u64>,
        /// Pairs forced down the fixed-function path (`VITASLOP_GXP_EXCLUDE`).
        exclude: Vec<u64>,
        /// `(key, target format) -> Some(pipeline)` for a linkable pair, `-> None` for one that
        /// failed to link (cached so we never retry it and always fall back).
        ///
        /// The FORMAT is part of the key because a render pipeline is bound to the format of
        /// the attachment it writes, and a surface the guest put in gamma-correct mode is
        /// rendered through an sRGB view of the same texture. Only a pair that is actually
        /// drawn onto such a surface ever gets a second entry, so a title using no gamma
        /// surfaces builds exactly as many pipelines as before.
        pipelines: HashMap<(u64, wgpu::TextureFormat, u32, u32), Option<GxpPipeline>>,
        /// Compiled WGSL modules, by shader PAIR.
        ///
        /// # A pair's module does not depend on the pipeline variant, and it used to be rebuilt
        /// # for every one of them
        /// `pipelines` is keyed by `(pair, format, samples, cull)` because a render pipeline is
        /// bound to all four. The MODULE is bound to none of them: it is a pure function of the
        /// two program blobs plus the clip/depth injections, which are read once per run. So a
        /// pair drawn onto both a 4-sample world target and a 1-sample display pass compiled the
        /// same WGSL twice, and one drawn with two cull modes compiled it twice again.
        ///
        /// MEASURED on a retail race: the run's pipeline builds cost **805 ms
        /// compiling WGSL against 271 ms creating pipelines** - three quarters of it in the
        /// half that is pair-only - across 163 pipelines built over 600 frames. In the BROWSER
        /// a WGSL compile costs far more than it does here, which is why the same race hitches
        /// there.
        ///
        /// This removes the duplicate compiles. It does NOT move the FIRST compile off the frame
        /// that needs it, which is the larger half of the problem: see the note in
        /// `build_gxp_pipeline` for why that needs the guest's shader patcher to name the pair.
        modules: HashMap<u64, wgpu::ShaderModule>,
        /// Memoized shader-pair keys, by the IDENTITY of the two program blobs plus the state
        /// baked into the pipeline alongside them - see [`GxpLive::key`] and
        /// [`GxpLive::pair_key`].
        ///
        /// The value is the same published key `key` computes; only the recomputation is
        /// skipped. The entry keeps its own `Arc` clones of both blobs, which is what makes a
        /// POINTER safe to identify them by: while the entry lives, that allocation cannot be
        /// freed and its address cannot be handed to a different program.
        pair_keys: HashMap<PairIdentity, (u64, std::sync::Arc<[u8]>, std::sync::Arc<[u8]>)>,
        /// Uploaded texture views, keyed by the decoded texture's content fingerprint and the
        /// view dimension it is bound as. A scene binds a handful of textures across hundreds
        /// of draws, so uploading per draw (as this path first did) re-sends the same
        /// multi-megabyte shadow map thousands of times a frame and exhausts GPU memory.
        ///
        /// >>> THE TEXTURE IS HELD BESIDE THE VIEW SO AN EVICTION CAN `destroy()` IT. Keeping
        /// only the view works - the view holds the texture alive - but it leaves nothing to
        /// call `destroy()` ON, and in the browser dropping a view merely makes the texture
        /// collectable ([[vitaslop-browser-gpu-needs-destroy]]). This cache evicts hardly at all
        /// on the desktop and CONSTANTLY on the user's device, where 83% of decodes were
        /// measured to be re-decodes of something just evicted - so this is the one leak the
        /// desktop is structurally unable to observe, on the engine that can least afford it.
        views: HashMap<(u64, SamplerDim), (wgpu::Texture, wgpu::TextureView)>,
        /// Bytes currently held by `views`, against [`tex_cache_budget_bytes`]. Tracked
        /// rather than derived because a `TextureView` cannot be asked its size.
        views_bytes: usize,
        /// The frame each cached view was last USED on, and the frame counter itself.
        ///
        /// # Why the cache needs to know what the current frame is touching
        /// The budget used to be enforced by clearing the cache WHOLESALE, on the argument
        /// that the keys are content fingerprints so a clear costs a re-upload and never
        /// correctness. That is true and it is not the problem. The problem is the shape of
        /// the failure: once a title's per-frame working set exceeds the budget, the clear
        /// fires part-way through EVERY frame, so every texture is evicted before the next
        /// frame asks for it again and the cross-frame hit rate is not degraded but ZERO.
        ///
        /// MEASURED on one title's campaign map, on the target phone: 226 distinct textures
        /// totalling 330 MB against a 256 MB budget - `0.97 cache clears` per frame, 225
        /// textures re-decoded and 76 MB re-uploaded per frame, `build 718 ms` of an 878 ms
        /// render, 1 fps. The cache was not too small by a factor of anything interesting; it
        /// was one step past a CLIFF, and a cliff is the wrong shape for a budget.
        ///
        /// So eviction is now per entry and never touches what THIS frame has already used.
        /// A working set that fits keeps hitting; one that does not degrades in proportion
        /// instead of collapsing, and the frame in flight can always complete.
        /// Per entry: `(the frame it was last used on, the bytes it holds)`. The size is here
        /// rather than derived because neither the key nor a `TextureView` carries it, and
        /// evicting one entry has to subtract exactly what that entry added.
        views_used: HashMap<(u64, SamplerDim), (u64, usize, u32, Residency)>,
        /// Keys this run has EVICTED and not yet seen come back, so a later upload of one can be
        /// counted as THRASH rather than as a cold upload. Diagnostic only - see
        /// `EncodeWork::tex_reuploaded_after_evict` for why an upload count cannot answer this.
        views_evicted: HashSet<(u64, SamplerDim)>,
        /// Bumped once per frame by [`GxmRenderer::encode_chain`]. Entries stamped with it are
        /// in use RIGHT NOW and are not eviction candidates at any budget.
        views_epoch: u64,
        /// The largest ONE-FRAME texture working set seen so far, in bytes, and the floor the
        /// budget is raised to.
        ///
        /// # Why the budget has to learn this
        /// Per-entry eviction stops the cache collapsing, but it cannot make a working set fit
        /// a budget smaller than itself: whatever is evicted early in a frame is wanted again
        /// later in the same frame, and the result is still a re-decode and a re-upload of
        /// most of it, every frame.
        ///
        /// A cache that cannot hold ONE frame is worse than no cache at all, and the bytes are
        /// not optional either way - every texture a frame samples has to be resident while
        /// that frame is encoded, so this floor asks for nothing the frame was not already
        /// going to hold. What it buys is keeping them for the NEXT frame, which on a title
        /// that redraws the same screen is the entire hit rate.
        ///
        /// MEASURED on one title's campaign map: 226 textures / 330 MB against the 256 MB
        /// default, `0.97 cache clears` a frame, `build 718 ms`, 1 fps.
        views_frame_high: usize,
        /// Bytes stamped with the CURRENT epoch, i.e. this frame's working set as it builds.
        views_frame_bytes: usize,
        /// Which block family this renderer's DEVICE accepts, resolved on the first encode and
        /// never again - see the note at the resolution site for why asking per draw is a browser
        /// regression rather than a style question. `None` until the first frame, because the
        /// device is not available where this struct is built.
        ///
        /// A family rather than a boolean, because the two answers are not interchangeable: an
        /// ETC2-only adapter handed a BC block would be a device-lost, not a fallback.
        bc_supported: Option<BlockFamily>,
        /// Samplers by `(linear, addr_mode_u, addr_mode_v)`. A handful per title.
        samplers_by_mode: HashMap<(bool, u32, u32), wgpu::Sampler>,
        /// How to choose the clip-`w` sign correction (`VITASLOP_GXP_NEGW`).
        negw: NegW,
        /// What interpreting each shader pair's vertex program over its own mesh said about the
        /// projection behind it (see `measure_clip`). Measured ONCE per key, on the first draw
        /// that uses it. `None` records a pair whose program does not interpret, so it is not
        /// re-attempted.
        ///
        /// The DECISIONS are not per pair - see [`GxpLive::decide_scene_negw`].
        negw_by_key: HashMap<u64, Option<ClipStats>>,
        /// Whether THIS pass's projection puts clip `w` negative in front of the camera, as
        /// decided by [`GxpLive::decide_scene_negw`] before the pass's draws are walked.
        scene_negw: bool,
        /// THIS pass's `(a, c)` in `guest window depth = a + c/w` (see [`ClipStats::depth_fit`]).
        scene_depth_fit: (f32, f32),
        /// Both verdicts, per render-target address, once a frame's draws produced evidence for
        /// them. Keeps the measurement off the per-frame path: a projection belongs to the pass,
        /// and a pass does not change projection convention between frames.
        negw_by_target: HashMap<u32, (bool, (f32, f32))>,
        /// group0/group1 bind groups over the pass's uniform ARENA, one per (shader pair,
        /// target format, group) rather than one per draw - the draw supplies only a dynamic
        /// offset. Cleared wholesale when the arena buffer is re-created, because a bind group
        /// names a specific buffer; `ubo_bgs_gen` is that buffer's generation.
        ubo_bgs: HashMap<(usize, u64, wgpu::TextureFormat, u32, u8), wgpu::BindGroup>,
        /// Per arena SLOT, the generation of the uniform buffer its cached entries name.
        ubo_bgs_gen: HashMap<usize, u64>,
        /// Repacked vertex streams, keyed by `(pipeline key, content hash of the guest
        /// stream)`. The repack walks every component of every vertex, and a world pass here
        /// submits meshes of ~3800 vertices with eleven components each, EVERY FRAME, from
        /// byte-identical guest bytes - the static world does not change between frames. The
        /// key is a content hash rather than the guest address precisely because a title also
        /// has dynamic geometry at a fixed address, which an address key would serve stale.
        ///
        /// Each entry holds `(the guest stream it was repacked from, the repacked bytes)`. The
        /// source is not there for lifetime reasons - it is the hit's PROOF. See the lookup in
        /// `prepare`: a content-hash cache over geometry that trusts its hash fails by silently
        /// drawing another draw's mesh, and this one did.
        packed: HashMap<(u64, u64), (std::sync::Arc<[u8]>, std::sync::Arc<[u8]>)>,
        /// The same entries, reached WITHOUT hashing: `(pipeline key, allocation address,
        /// length)` of the guest stream.
        ///
        /// # Why the content hash alone was not enough
        /// The content hash is what makes the cache correct across a re-snapshot into a new
        /// allocation, and it is the reason a hit is verified against the source bytes. But it
        /// is computed on EVERY draw of EVERY frame, hit or miss, and it reads the whole guest
        /// vertex stream to do it - so the cache that exists to stop the renderer touching all
        /// of a frame's geometry was touching all of it anyway, just to find out it did not
        /// have to. That is the shape of cost this project keeps finding: work proportional to
        /// the frame's volume on the path that was supposed to make the volume free.
        ///
        /// An allocation address is a sound key here for one reason, and it is the reason it
        /// must stay: **the entry holds a strong `Arc` to the very stream it was repacked
        /// from**, so that allocation cannot be freed while the entry lives, and no later
        /// stream can be handed the same address. `Arc<[u8]>` has no interior mutability, so
        /// the bytes behind a live address cannot change either. The lookup still asserts it
        /// with `Arc::ptr_eq` rather than trusting the argument - the same discipline the
        /// content path uses, and for the same reason: a geometry cache that is wrong draws
        /// another mesh, confidently, with nothing anywhere reporting it.
        ///
        /// A miss here falls through to the content hash, so nothing that used to hit stops
        /// hitting. It is a fast path, not a replacement.
        packed_by_alloc: HashMap<(u64, usize, usize), (std::sync::Arc<[u8]>, std::sync::Arc<[u8]>)>,
        /// Repacked vertices and expanded indices that have not changed since the renderer first
        /// saw them, resident on the GPU instead of copied into a pass arena every frame. See
        /// [`ResidentHeap`] for the measurement this is aimed at and for why an address is a
        /// sound key here. `VITASLOP_RESIDENT_GEOM=0` is the A/B arm.
        resident_v: ResidentHeap,
        resident_i: ResidentHeap,
        /// Index allocations seen before, so an index list is promoted on its SECOND sighting
        /// rather than its first - the vertex side gets the same test for free from
        /// `packed_by_alloc`.
        ///
        /// >>> IT IDENTIFIES THE ALLOCATION, NOT THE ADDRESS. A set of bare addresses says
        /// "seen before" for an address a FREED list used to occupy, so a title whose index
        /// buffers churn promotes a fresh allocation every frame and fills the heap with
        /// geometry no draw will ever ask for again. MEASURED on one of them's on-track
        /// run before this distinction existed: **9,580 index meshes placed in 328 frames,
        /// filling the 48 MB heap and resetting it** - about 29 dead promotions a frame.
        ///
        /// >>> AND IT IS A `Weak`, BECAUSE A STRONG ONE WOULD BE A LEAK. Holding the `Arc` makes
        /// the test exact and ALSO keeps every index list this map has ever seen alive - up to
        /// the cap, which on the title above is thousands of expanded index buffers the title
        /// itself has finished with. A `Weak` is exactly as good a test (a dead one cannot be
        /// upgraded, and an address recycled after the original died fails the upgrade, which is
        /// the honest answer) and owns nothing.
        ///
        /// Cleared wholesale at its cap, which delays a promotion and nothing else.
        resident_i_seen: HashMap<(u64, usize, usize), std::sync::Weak<[u8]>>,
        /// Shader pairs `precompile_pairs` has already considered, by the ALLOCATION of the two
        /// program blobs. The pending list is re-offered every frame and `module_key` hashes
        /// both blobs, so without this the preparation costs a few hundred kilobytes of hashing
        /// per frame FOREVER - and proportionally more the more pairs a title names.
        ///
        /// It holds no `Arc`, unlike the geometry caches, and the reason it may not is the
        /// reason it is safe: a stale address here can only make a pair be SKIPPED, and a
        /// skipped pair compiles at its first draw exactly as it did before any of this existed.
        /// A wrong answer costs a hitch, never a picture.
        precompile_seen: HashSet<(usize, usize)>,
        resident: bool,
        /// The byte budget for each of the two heaps (`VITASLOP_RESIDENT_GEOM_MB`, per heap).
        resident_budget: u64,
        /// The `@group(3)` depth-range bind group, keyed by `(pipeline key, the depth
        /// range's bits)`.
        ///
        /// That group holds ONE vec4 - the scene's depth min and scale - which is the
        /// same for every draw in a scene. Building a fresh 16-byte buffer and bind
        /// group per draw meant a couple of hundred GPU allocations a frame to say the
        /// same thing each time. Keyed by pipeline as well as by value because each
        /// pipeline owns its own `BindGroupLayout` object; a shared group-3 layout
        /// would collapse this to one entry, and is the right follow-up.
        ///
        /// >>> THIS CACHE IS BOUNDED, AND IT WAS NOT. The key holds the depth range's RAW
        /// BITS, and a depth range is a continuous quantity: on a menu it is constant and this
        /// cache holds a handful of entries, but in a RACE the near/far pair moves every frame,
        /// so every frame mints entries that will never be asked for again. Each one retains a
        /// GPU buffer AND a bind group for the rest of the run. Measured on the committed race
        /// recipe in desktop Chrome: **30-33 buffers created EVERY FRAME, sustained**, in a
        /// renderer whose steady state should create none - and the GPU process climbed from a
        /// flat 0.20 GB through the menus to 5.8 GB (13.4 GB private) by frame 9000. The two
        /// sibling caches on either side of this one are both bounded (`ubo_bgs` by generation,
        /// `sampler_bgs` by [`SAMPLER_BG_CACHE_CAP`]); this one was simply missed.
        ///
        /// Bounded the same way `sampler_bgs` is, and for the same reason that makes a wholesale
        /// clear safe here: the key IS the value, so a cleared entry is rebuilt to something
        /// byte-identical. Clearing costs a rebuild and can never cost a wrong answer - unlike
        /// [[vitaslop-content-hash-cache-must-verify]], where the value was geometry.
        /// **The buffer is held beside the bind group so an eviction can `destroy()` it.**
        /// It used to be dropped the instant it was built - the entry kept only the bind group,
        /// which holds the buffer alive in JS but leaves NOTHING to call `destroy()` on. So
        /// every wholesale clear handed the collector up to [`DEPTH_BG_CACHE_CAP`] buffers at
        /// once, on the engine where dropping a handle reclaims nothing on our schedule. Same
        /// class of defect as [[vitaslop-browser-gpu-needs-destroy]], and invisible on a MENU
        /// (a constant depth range caches a handful of entries) while minting ~30 a frame in a
        /// RACE, which is the screen this has to survive.
        depth_bgs: HashMap<(u64, (u64, u32, u32), bool), (wgpu::BindGroup, wgpu::Buffer)>,
        /// Buffers evicted from `depth_bgs`, waiting to be destroyed on the renderer's frame
        /// schedule. They cannot be destroyed at the eviction itself: a clear happens during
        /// `prepare`, and a draw already prepared THIS frame may still name the bind group that
        /// owns one. `encode_chain` drains this into the renderer's graveyard, which destroys
        /// after the submit - the same rule the arenas follow.
        depth_retired: Vec<wgpu::Buffer>,
        /// The `@group(2)` SAMPLER bind group, keyed by the pipeline and by exactly what it
        /// binds: each unit's chosen texture view and its sampler state.
        ///
        /// This group used to be built with `device.create_bind_group` on EVERY draw of every
        /// frame - a real GPU object per draw, which is the thing
        /// [[vitaslop-per-pass-render-arenas]] exists to forbid, and in the browser it is also
        /// a wasm/JS boundary crossing per draw. A bind group is a pure function of the views
        /// and samplers it names, and both of those are already content-addressed caches, so
        /// the same draw next frame asks for a group that is byte-for-byte the one it got
        /// last frame.
        ///
        /// Entries naming a render target THIS frame produced are NOT cached: those views
        /// belong to textures the frame allocates, so a cached group would name a dead one.
        /// Those draws still build fresh, and the counter says how many.
        ///
        /// Each entry carries the view-cache keys it NAMES, so an eviction can drop exactly the
        /// groups that just went stale. See the eviction site for why a generation counter was
        /// the wrong instrument here.
        sampler_bgs: HashMap<(u64, u64), (wgpu::BindGroup, Vec<(u64, SamplerDim)>)>,
        /// The GPU texture transcoder's compute pipelines ([`crate::texenc`]).
        ///
        /// Built on first use rather than in `new`, for the same reason `bc_supported` is asked
        /// once: it compiles three compute shaders, and a run whose adapter takes the guest's
        /// blocks verbatim never transcodes anything and should not pay for them. Held here
        /// rather than in a global because "there is only ever one device" is true today and is
        /// not a thing this code should assert.
        texenc: Option<crate::texenc::Transcoder>,
    }

    /// Which depth the recompiled vertex stage writes (`VITASLOP_GXP_ZFIX`).
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum ZFix {
        /// The scene's `-1/w` range, normalized onto [0,1] (`=range`): the same quantity and the
        /// same range the FIXED-FUNCTION path writes, so both kinds of draw share one comparable
        /// depth buffer. Costs a dependency on the scene depth range, which is measured through
        /// the software path's own reflected transform - and that is why it is no longer the
        /// default. It is a quantity the guest never computes, so the guest's own depth FUNCTION
        /// has no meaning against it, and any draw whose projection differs from the world's
        /// (a 2D overlay has a constant `w`) falls outside the measured range and CLAMPS, to the
        /// nearest value. Keep it only for a frame that still mixes in fixed-function draws.
        Range,
        /// The GL->WebGPU clip-depth remap `(z + w) / 2` (`=gl`), for a guest whose clip z is
        /// read in [-w, w].
        ///
        /// GXM is NOT such a guest, which is measurable rather than a matter of provenance:
        /// `count_clip_w_signs` fits `z = a*w + c` per pass, and one title's world pass fits
        /// `a = 1.00003, c = -2.0`, i.e. `z/w = 1 - 2/w`, which runs from 0 at the near plane
        /// (`w = 2`) to 1 at infinity. Its clip z is already in [0, w] - the D3D/WebGPU
        /// convention, not the GL one - so this remap would squash the whole scene into the far
        /// half of the buffer.
        Gl,
        /// Pass the guest's clip z straight through (`=0`/`=off`).
        ///
        /// The depth buffer then holds the guest's own window depth `z/w` exactly. It is also
        /// what makes WebGPU CLIP every fragment whose depth leaves [0, w] - which PowerVR does
        /// NOT do. MEASURED: one title's whole front end disappears under this mode and renders
        /// perfectly under [`ZFix::Clamp`], which differs from it only by the clamp.
        Off,
        /// The guest's own window depth `z/w`, CLAMPED to [0,1] (`=clamp`).
        ///
        /// This is the faithful one, and it is two facts put together:
        ///
        /// 1. GXM clip z is already in [0, w] - the D3D/WebGPU convention, not the GL one.
        ///    `count_clip_w_signs` fits `z = a*w + c` per pass; one title's world pass fits
        ///    `a = 1.00003, c = -2.0`, so `z/w` runs from 0 at its near plane (`w = 2`) to 1 at
        ///    infinity. So the depth buffer can hold the guest's own value untouched, and its
        ///    captured `SceGxmDepthFunc` then means exactly what it means on hardware.
        /// 2. PowerVR CLAMPS depth where an immediate-mode rasteriser CLIPS. A primitive that
        ///    runs past the far plane is drawn AT the far plane, not thrown away. WebGPU only
        ///    offers that through the optional `depth-clip-control` feature, so the clamp is
        ///    done in the shader instead: scaling by `c.w` re-encodes the clamped window depth
        ///    as a clip z the rasteriser will divide back out.
        ///
        /// The result depends on nothing but the draw's own projection, so a 2D overlay lands
        /// where the guest put it rather than at the end of the world pass's range.
        Clamp,
        /// The guest's OWN viewport depth mapping, clamped (the DEFAULT).
        ///
        /// `sceGxmSetViewport` takes `zOffset` and `zScale`, and the hardware's window depth is
        /// `z/w * zScale + zOffset`. That is not a convention to be inferred, it is state the
        /// guest SET, and [`Clamp`](Self::Clamp) - which is this map with `zScale = 1,
        /// zOffset = 0` hard-coded - reads it wrong wherever the guest asked for anything else.
        ///
        /// >>> AND A TITLE DOES. MEASURED on one title's title screen: that pass reports
        /// `zOffset=0.5 zScale=0.5`, i.e. clip z in [-w, w], and its whole lattice sits at clip
        /// `z ~ -0.98`. Under `Clamp` every vertex of it lands at window depth EXACTLY 0, so a
        /// `LESS_EQUAL` depth test can reject nothing and the mesh's own hidden faces paint over
        /// the faces in front of them - which is what the over-embossed lattice was. Under the
        /// guest's own mapping the same geometry spans 0.0098..0.0107, the prism tops sit in
        /// front of their skirts by 0.0009, and the test does what the hardware does.
        /// `VITASLOP_GXP_NODEPTH=1` being BIT-IDENTICAL to the normal render was the fingerprint:
        /// a depth test that rejects nothing is not a working depth test.
        ///
        /// The PowerVR clamp of [`Clamp`](Self::Clamp) is kept - it is a separate fact about the
        /// hardware and it is still true - so this differs from it only by asking the guest.
        /// A pass the guest never gave a viewport keeps the identity mapping, which is exactly
        /// what `Clamp` did, and says so once rather than guessing quietly.
        Viewport,
    }

    /// How the clip-`w` sign correction is chosen (`VITASLOP_GXP_NEGW`).
    ///
    /// A guest projection may put clip `w` NEGATIVE in front of the camera, in which case
    /// WebGPU clips every world draw away and the pass renders black with correct shaders,
    /// textures, depth and blend (memory `vitaslop-clip-w-can-be-negative`). The correction
    /// is to negate the whole clip vector, which names the same point with `w > 0`.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    /// MEASURED on one retail racer at a race frame, and this is what settles which correction
    /// is right: with `w` flipped the HUD reads the right way round and the player's vehicle
    /// carries readable lettering on its tail; with the WHOLE vector negated the
    /// identical frame comes out turned 180 degrees, every glyph mirrored. So the guest's
    /// `x`,`y`,`z` are right and the `w` reaching WebGPU has the wrong SIGN - full negation
    /// preserves `x/w` exactly, which is precisely why it cannot fix a mirrored picture.
    enum NegW {
        /// Never correct (`=0`). The clip position goes to WebGPU as the shader produced it.
        Off,
        /// Measure it per PASS (`=auto`, the default) and flip `w` on every pair of a pass
        /// whose projection is negative. See [`GxpLive::decide_scene_negw`] for why the
        /// decision cannot be taken per shader pair.
        Auto,
        /// Negate the WHOLE clip vector on the pairs `Auto` would correct (`=negate`).
        ///
        /// Kept because it is the other reading of the same measurement and the two are one
        /// experiment apart: negating names the same point with `w > 0` (homogeneous coordinates
        /// are scale-invariant), so it lifts the clip and changes nothing else. On this title it
        /// renders the frame upside down - see the note above.
        Negate,
        /// Correct EVERY pass, without measuring (`=force`).
        ///
        /// A diagnostic, and specifically the one for a pass whose vertex program the CPU
        /// interpreter cannot run - a program that samples textures, or indexes a uniform
        /// array. `Auto` then has no evidence and leaves the pass uncorrected, which is
        /// indistinguishable on screen from a pass that needed no correction: both render
        /// black. This settles which, in one run.
        Force,
    }

    /// Diagnostic (`VITASLOP_GXP_DUMP`): print each recompiled draw's inputs.
    ///
    /// Cached, because this is tested once per DRAW and reading an unset environment
    /// variable on Windows copies and re-encodes the whole environment block.
    fn gxp_dump() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::knobs::flag("VITASLOP_GXP_DUMP"))
    }

    impl GxpLive {
        /// Read the recompiler's configuration.
        ///
        /// Through [`crate::knobs`], not `std::env` directly: the browser has no
        /// environment, so an `std::env` read here silently disables the recompiler
        /// there and leaves the browser drawing the fixed-function approximation while
        /// the desktop oracle draws the guest's real shaders.
        fn from_env() -> Self {
            let flag = crate::knobs::flag;
            GxpLive {
                enabled: flag("VITASLOP_GXP_LIVE"),
                only: flag("VITASLOP_GXP_ONLY"),
                zfix: match crate::knobs::var("VITASLOP_GXP_ZFIX").ok().as_deref() {
                    Some("gl") => ZFix::Gl,
                    Some("range") => ZFix::Range,
                    Some("0") | Some("off") => ZFix::Off,
                    Some("clamp") => ZFix::Clamp,
                    _ => ZFix::Viewport,
                },
                yflip: crate::knobs::var("VITASLOP_GXP_YFLIP").map(|v| v != "0").unwrap_or(false),
                force: flag("VITASLOP_GXP_FORCE"),
                solid: flag("VITASLOP_GXP_SOLID"),
                nodepth: flag("VITASLOP_GXP_NODEPTH"),
                noblend: flag("VITASLOP_GXP_NOBLEND"),
                keys: crate::knobs::var("VITASLOP_GXP_KEYS")
                    .ok()
                    .map(|v| {
                        v.split(',')
                            .filter_map(|k| u64::from_str_radix(k.trim().trim_start_matches("0x"), 16).ok())
                            .collect()
                    })
                    .unwrap_or_default(),
                exclude: crate::knobs::var("VITASLOP_GXP_EXCLUDE")
                    .ok()
                    .map(|v| {
                        v.split(',')
                            .filter_map(|k| u64::from_str_radix(k.trim().trim_start_matches("0x"), 16).ok())
                            .collect()
                    })
                    .unwrap_or_default(),
                pipelines: HashMap::default(),
                modules: HashMap::default(),
                pair_keys: HashMap::default(),
                views: HashMap::default(),
                views_bytes: 0,
                views_used: HashMap::default(),
                views_evicted: HashSet::default(),
                views_epoch: 0,
                views_frame_high: 0,
                views_frame_bytes: 0,
                bc_supported: None,
                texenc: None,
                samplers_by_mode: HashMap::default(),
                negw: match crate::knobs::var("VITASLOP_GXP_NEGW").ok().as_deref() {
                    Some("0") | Some("off") => NegW::Off,
                    Some("negate") => NegW::Negate,
                    Some("force") => NegW::Force,
                    _ => NegW::Auto,
                },
                negw_by_key: HashMap::default(),
                scene_negw: false,
                scene_depth_fit: DEPTH_FIT_RECIP_W,
                negw_by_target: HashMap::default(),
                ubo_bgs: HashMap::default(),
                ubo_bgs_gen: HashMap::default(),
                packed: HashMap::default(),
                packed_by_alloc: HashMap::default(),
                resident_v: ResidentHeap::new(),
                resident_i: ResidentHeap::new(),
                resident_i_seen: HashMap::default(),
                precompile_seen: HashSet::default(),
                // Value-sensitive, as an A/B arm has to be: `0` sends every draw back through
                // the per-frame arenas, which is what every draw did before this existed.
                resident: crate::knobs::var("VITASLOP_RESIDENT_GEOM").map(|v| v.trim() != "0").unwrap_or(true),
                resident_budget: crate::knobs::var("VITASLOP_RESIDENT_GEOM_MB")
                    .ok()
                    .and_then(|v| v.trim().parse::<u64>().ok())
                    .unwrap_or(48)
                    .clamp(1, 1024)
                    * 1024
                    * 1024,
                depth_bgs: HashMap::default(),
                depth_retired: Vec::new(),
                sampler_bgs: HashMap::default(),
            }
        }

        /// Ensure a group0/group1 bind group exists over the pass's uniform arena for every
        /// shader pair the pass uses. Called once per pass, after the arena buffer exists -
        /// which is why it cannot happen inside `prepare`: the buffer is sized from the arena
        /// `prepare` fills.
        fn ensure_ubo_bgs(
            &mut self,
            device: &wgpu::Device,
            buffer: &wgpu::Buffer,
            slot: usize,
            generation: u64,
            used: &[(u64, wgpu::TextureFormat, u32, u32)],
        ) {
            // Entries are keyed by ARENA SLOT and dropped only when THAT slot's uniform buffer
            // is re-created. The pool made the buffer stable across frames, so this cache is
            // now stable across frames too - it used to be cleared on every pass of every
            // frame, because every pass got a brand-new buffer to name.
            if self.ubo_bgs_gen.get(&slot) != Some(&generation) {
                self.ubo_bgs.retain(|k, _| k.0 != slot);
                self.ubo_bgs_gen.insert(slot, generation);
            }
            for &(key, format, samples, cull) in used {
                let Some(Some(pipe)) = self.pipelines.get(&(key, format, samples, cull)) else { continue };
                for (group, lanes) in [(0u8, pipe.vsa_lanes), (1u8, pipe.fsa_lanes)] {
                    if self.ubo_bgs.contains_key(&(slot, key, format, samples, group)) {
                        enc(&ENC.bind_groups_reused, 1);
                        continue;
                    }
                    enc(&ENC.bind_groups_built, 1);
                    let layout = &pipe.layouts[group as usize];
                    let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();
                    if lanes > 0 {
                        entries.push(wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer,
                                offset: 0,
                                size: wgpu::BufferSize::new((lanes.div_ceil(4) as u64) * 16),
                            }),
                        });
                    }
                    // The vertex stage's guest-memory window rides in group 0 beside the SA
                    // uniform, over the same arena with its own dynamic offset.
                    if group == 0 && pipe.mem_bind_bytes > 0 {
                        entries.push(wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer,
                                offset: 0,
                                size: wgpu::BufferSize::new(pipe.mem_bind_bytes as u64),
                            }),
                        });
                    }
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(if entries.is_empty() { "gxp-ubo-empty" } else { "gxp-ubo-bind" }),
                        layout,
                        entries: &entries,
                    });
                    self.ubo_bgs.insert((slot, key, format, samples, group), bg);
                }
            }
        }

        /// Decide, ONCE PER PASS, whether this pass's projection puts clip `w` NEGATIVE in
        /// front of the camera - in which case WebGPU clips every draw away and the pass
        /// renders black with correct shaders, textures, depth and blend (memory
        /// `vitaslop-clip-w-can-be-negative`).
        ///
        /// **It cannot be decided per shader pair, and doing so was a bug.** The sign
        /// convention is a property of the PROJECTION MATRIX, which every draw of a pass
        /// shares; "all of this pair's vertices have `w < 0`" is equally the signature of a
        /// draw that is simply BEHIND THE CAMERA, which WebGPU is then right to clip. Per pair,
        /// those two are indistinguishable. MEASURED on one retail racer's race pass: of ~60 pairs,
        /// exactly two came out negative - both engine-trail ribbons ~700 units behind the eye
        /// - and "correcting" them pulled behind-camera geometry into the frame as a smudge in
        /// the corner. Across a whole pass the two are easy to tell apart: a negative
        /// projection makes EVERY draw negative, and one stray draw cannot outvote the rest.
        ///
        /// The per-pair MEASUREMENT is still cached per key (interpreting a vertex program over
        /// its mesh is not cheap); only the verdict is taken over the pass.
        ///
        /// The same walk settles the pass's DEPTH ENCODING - `z_clip = a * w_clip + c`, hence a
        /// window depth of `a + c/w` - because both answers come from the same interpretation
        /// and neither can be read off the frame afterwards. See [`ClipStats::depth_fit`].
        fn decide_scene_negw(&mut self, scene: &RenderScene) {
            if self.negw == NegW::Off || !self.enabled {
                self.scene_negw = false;
                self.scene_depth_fit = DEPTH_FIT_RECIP_W;
                return;
            }
            // `=force`: correct every pass without measuring, and SAY SO - a frame built on an
            // assumed projection must never be mistaken for a measured one.
            if self.negw == NegW::Force {
                let target = scene.target.as_ref().map(|t| t.data_addr).unwrap_or(0);
                report_forced_negw(target);
                self.scene_negw = true;
                self.scene_depth_fit = DEPTH_FIT_RECIP_W;
                return;
            }
            let target = scene.target.as_ref().map(|t| t.data_addr).unwrap_or(0);
            // Settled passes cost nothing per frame. A pass is only settled once its draws
            // produced EVIDENCE: a frame in which every draw of a pass happens to cover nothing
            // says nothing about its projection, and freezing "not negative" from it would make
            // the answer depend on which frame the pass was first seen in.
            if let Some(&decided) = self.negw_by_target.get(&target) {
                self.scene_negw = decided.0;
                self.scene_depth_fit = decided.1;
                return;
            }
            let (mut in_front, mut behind) = (0usize, 0usize);
            // The fit is taken from the ONE draw with the widest spread of `w`, not averaged
            // over the pass: `z = a*w + c` is exact for every draw sharing the projection, so a
            // wider spread is simply a better-conditioned way of asking the same question, while
            // an average would let a near-degenerate draw drag the answer.
            let (mut fit, mut best_spread) = (None, 0.0f32);
            for d in &scene.draws {
                let Some(gxp) = d.gxp.as_ref() else { continue };
                let key = self.pair_key(gxp);
                let stats = match self.negw_by_key.get(&key) {
                    Some(&s) => s,
                    None => {
                        let s = measure_clip(gxp, key);
                        // Only remember an answer the measurement supports: a draw entirely off
                        // screen decides nothing, and the next draw of the same pair may see the
                        // geometry that settles it.
                        let empty = s.is_some_and(|s| s.in_front == 0 && s.behind == 0);
                        if !empty {
                            self.negw_by_key.insert(key, s);
                        }
                        s
                    }
                };
                let Some(s) = stats else { continue };
                in_front += s.in_front;
                behind += s.behind;
                if let Some(f) = s.depth_fit {
                    if s.w_spread > best_spread {
                        best_spread = s.w_spread;
                        fit = Some(f);
                    }
                }
            }
            // A negative PROJECTION leaves nothing at all on the positive side. Requiring that
            // (rather than a mere majority) is what keeps a handful of behind-camera draws from
            // turning a pass whose projection is perfectly ordinary. One vertex on the positive
            // side refutes it - which is deliberately strict, and is reported either way.
            let correct = behind > 0 && in_front == 0;
            let depth_fit = fit.unwrap_or(DEPTH_FIT_RECIP_W);
            if in_front + behind > 0 {
                self.negw_by_target.insert(target, (correct, depth_fit));
                report!(
                    "gxp clip: pass into 0x{target:08x}: {in_front} sampled vertices land in the frustum \
                     with w>0 and {behind} with w<0 -> {}; guest depth = {} + {}/w{}",
                    if correct {
                        "the projection is NEGATIVE, CORRECTING the clip w sign for every pair of this pass"
                    } else {
                        "clip positions left as the shaders produced them"
                    },
                    depth_fit.0,
                    depth_fit.1,
                    if fit.is_some() { "" } else { " (NO draw of this pass fits one - assuming -1/w)" }
                );
            }
            self.scene_negw = correct;
            self.scene_depth_fit = depth_fit;
        }

        /// The measured `(a, c)` of `guest window depth = a + c/w` for the pass that RENDERS
        /// into `addr`, for a later pass that wants to sample its depth.
        ///
        /// Keyed by the target rather than taken from `scene_depth_fit`, because the pass being
        /// encoded when a depth surface is converted is not necessarily the pass that wrote it.
        /// Falls back to `-1/w` for a target no pass has yet produced evidence for - which is
        /// the honest answer, not a silent zero: it is the encoding with no projection in it.
        fn depth_fit_for(&self, addr: u32) -> (f32, f32) {
            self.negw_by_target.get(&addr).map(|v| v.1).unwrap_or(DEPTH_FIT_RECIP_W)
        }

        /// Stable cache key for a shader pair AND the fixed-function state baked into its
        /// pipeline: FNV-1a over the vertex blob, the fragment blob, the blend equation and
        /// the depth state.
        ///
        /// Everything the pipeline bakes in has to be in the key, because the cache holds
        /// compiled PIPELINES, not modules. Two draws sharing a shader pair but differing in
        /// `SceGxmBlendInfo` or in depth func/write are different pipelines, and leaving
        /// either out silently gives the second draw the first one's state.
        /// Deliberately still the BYTE-at-a-time FNV, and not the word-wise [`fnv64`] used for
        /// content keys: this value is a published identity. It is what `VITASLOP_GXP_KEYS`,
        /// `_EXCLUDE`, `_INPUTS` and `_SA` select a pair by, what the fallback and keycolour
        /// reports name, and what every recorded investigation of this title refers to. Making
        /// it faster would rename every pair and silently invalidate all of that for a fraction
        /// of a millisecond.
        /// [`GxpLive::key`], memoized on the identity of the two program blobs.
        ///
        /// `key` is FNV over the whole vertex AND fragment container, byte at a time, and it is
        /// computed for EVERY draw: a race frame here submits 465 recompiled draws over a
        /// couple of dozen distinct pairs, so it re-hashes megabytes of the same shader bytes
        /// per frame inside `prepare` - the largest single item in the render frame. The blobs
        /// are shared `Arc`s handed out by the capture's own per-program cache, so two draws
        /// bound to the same program carry the same allocation, and a pointer pair identifies
        /// it exactly.
        ///
        /// A miss simply recomputes, so this can only ever be slower, never wrong: the value is
        /// still the published key that `VITASLOP_GXP_KEYS`/`_INPUTS`/`_SA` and every recorded
        /// investigation of this title select a pair by.
        fn pair_key(&mut self, gxp: &GxpRecompile) -> u64 {
            let id = PairIdentity {
                vptr: gxp.vprog.as_ptr() as usize,
                vlen: gxp.vprog.len(),
                fptr: gxp.fprog.as_ptr() as usize,
                flen: gxp.fprog.len(),
                blend: gxp.blend_state,
                depth_write: gxp.depth_write,
                depth_func: gxp.depth_func,
                fragment_program_enabled: gxp.fragment_program_enabled,
            };
            if let Some((k, _, _)) = self.pair_keys.get(&id) {
                return *k;
            }
            // Bounded like the other caches here. The key is derived from the entry's own
            // contents, so clearing wholesale costs a rehash and never correctness.
            if self.pair_keys.len() >= PAIR_KEY_CACHE_CAP {
                self.pair_keys.clear();
            }
            let k = Self::key(gxp);
            self.pair_keys.insert(id, (k, gxp.vprog.clone(), gxp.fprog.clone()));
            k
        }

        /// The repacked bytes for a vertex stream whose ALLOCATION this renderer has not packed
        /// before - the slow half of the packed-vertex cache.
        ///
        /// Reached only on an allocation miss, which is what makes the content hash affordable:
        /// it reads the whole guest stream, and it used to do so on every draw of every frame,
        /// including the ones the cache then served.
        ///
        /// >>> A HIT IS VERIFIED AGAINST THE SOURCE BYTES, not trusted on the hash alone.
        ///
        /// The entry carries the vertex stream it was repacked FROM, and a hit that does not
        /// match it byte for byte is treated as a miss. A content-hash cache whose value is
        /// GEOMETRY fails silently when it collides - the colliding draw renders the other
        /// draw's mesh, correctly and confidently, and nothing anywhere reports it. That is
        /// exactly what happened here: the word-wise hash cancelled paired top-bit flips (see
        /// [`fnv64`]), a faded text quad collided with itself one alpha level away, and the
        /// resulting flicker was chased across several sessions through the clock, the
        /// scheduler, the display queue and the presentation path - every one of which was
        /// innocent and each of which had to be excluded by measurement.
        ///
        /// The comparison is a memcmp of the same buffer the repack would otherwise READ, so it
        /// is strictly cheaper than the work it still saves, and it makes the cache's
        /// correctness independent of the hash rather than conditional on it. The hit it exists
        /// for is a stream RE-SNAPSHOTTED into a new allocation with the same contents - the
        /// same-allocation hit is served by the caller without reaching here at all.
        fn pack_vertices(
            packed: &mut HashMap<(u64, u64), (std::sync::Arc<[u8]>, std::sync::Arc<[u8]>)>,
            packed_by_alloc: &mut HashMap<(u64, usize, usize), (std::sync::Arc<[u8]>, std::sync::Arc<[u8]>)>,
            akey: (u64, usize, usize),
            gxp: &GxpRecompile,
            repack: &[RepackAttr],
            packed_stride: u32,
        ) -> std::sync::Arc<[u8]> {
            let t_hash = split_start();
            let pkey = (akey.0, fnv64(0xcbf2_9ce4_8422_2325, &gxp.vertices));
            split_end(t_hash, &PREP.hash_ns);
            split_add(&PREP.hash_bytes, gxp.vertices.len() as u64);
            let hit = packed
                .get(&pkey)
                .filter(|(src, _)| src[..] == gxp.vertices[..])
                .map(|(_, bytes)| bytes.clone());
            let bytes = match hit {
                Some(bytes) => {
                    split_add(&PREP.packed_hits, 1);
                    bytes
                }
                None => {
                    split_add(&PREP.packed_misses, 1);
                    let t_repack = split_start();
                    // Bound the cache the way the texture caches are bounded: the key is a
                    // content hash, so clearing wholesale costs a repack and never correctness.
                    if packed.len() >= PACKED_CACHE_CAP {
                        packed.clear();
                    }
                    let mut out = Vec::new();
                    repack_vertices_into(&gxp.vertices, gxp.vertex_stride, repack, packed_stride, &mut out);
                    split_end(t_repack, &PREP.repack_ns);
                    split_add(&PREP.repack_bytes, gxp.vertices.len() as u64);
                    let out: std::sync::Arc<[u8]> = out.into();
                    packed.insert(pkey, (gxp.vertices.clone(), out.clone()));
                    out
                }
            };
            // Both maps hold the source `Arc`, which is what keeps the allocation alive and so
            // keeps its ADDRESS from being handed to a later stream. Bounded the same way.
            if packed_by_alloc.len() >= PACKED_CACHE_CAP {
                packed_by_alloc.clear();
            }
            packed_by_alloc.insert(akey, (gxp.vertices.clone(), bytes.clone()));
            bytes
        }

        /// The key a compiled WGSL MODULE is cached under: the two program blobs and nothing
        /// else.
        ///
        /// # Why not the pipeline key, which is what this used to be
        /// `key` folds in blend, depth write, depth func and the fragment-enable, because a
        /// PIPELINE is bound to all of them. The WGSL is bound to none: `link_programs` reads
        /// only the two containers, and the clip fixup after it reads only run-level knobs. So
        /// keying modules by the pipeline key compiled the same source again for every variant -
        /// and, more importantly here, made the module uncomputable until a DRAW supplied the
        /// depth state, which is exactly what stopped it being prepared when the guest's shader
        /// patcher names the pair.
        ///
        /// The keycolour diagnostic is the one thing that does vary per pipeline key (it derives
        /// a colour from it), so under that knob the module stays keyed the old way rather than
        /// have two pairs share one colour.
        fn module_key(vprog: &[u8], fprog: &[u8]) -> u64 {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in vprog.iter().chain(fprog.iter()) {
                h ^= *b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            h
        }

        fn key(gxp: &GxpRecompile) -> u64 {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            let depth = [
                gxp.depth_write as u8,
                (gxp.depth_func >> 22) as u8 & 0x7,
                gxp.fragment_program_enabled as u8,
            ];
            for b in gxp
                .vprog
                .iter()
                .chain(gxp.fprog.iter())
                .chain(gxp.blend_state.iter())
                .chain(depth.iter())
            {
                h ^= *b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            h
        }

        /// Prepare the GPU resources for one recompiled draw. Returns `None` (caller falls
        /// back to fixed-function) if the pair does not link or a resource cannot be built.
        fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            color_format: wgpu::TextureFormat,
            // Samples per pixel of the pass this draw is being prepared for - see
            // `GxpPrepared::samples`.
            samples: u32,
            gxp: &GxpRecompile,
            depth_range: [f32; 2],
            // `rendered`: views of the render targets this frame has already drawn, by
            // guest address. A sampler whose bound texture points at one of these binds
            // the render rather than the guest's (stale) bytes.
            rendered: &HashMap<u32, wgpu::TextureView>,
            // The same, for the DEPTH of those targets, keyed by the guest's depth-surface
            // address. Checked before `rendered` - see `make_sampler_bg`.
            depth_rendered: &HashMap<u32, wgpu::TextureView>,
            // The renderer's render-target view generation and the addresses currently resolving
            // to a SNAPSHOT, both of which key a sampler bind group that names a target. See
            // `GxmRenderer::rtt_epoch`.
            rtt_epoch: u64,
            reads_snapshot: &HashSet<u32>,
            // The pass's grow-only vertex and index arenas, appended to rather than allocated
            // from per draw. See [`GxpPrepared`].
            vdata: &mut Vec<u8>,
            idata: &mut Vec<u8>,
            // The pass's uniform arena and the device's dynamic-offset alignment.
            udata: &mut Vec<u8>,
            ubo_align: u64,
        ) -> Option<GxpPrepared> {
            if gxp.index_count == 0 || gxp.vertices.is_empty() {
                return None;
            }
            // The preamble: the pair key, the pipeline-cache lookup and, on a miss, the pipeline
            // BUILD. The build is separately counted (`pipelines_built`), so a frame that reads
            // high here with a zero build count is paying the lookups, not the compiler.
            let t_key = split_start();
            let key = self.pair_key(gxp);
            if !self.keys.is_empty() && !self.keys.contains(&key) {
                return None;
            }
            // `VITASLOP_GXP_EXCLUDE=<key>,...`: send these pairs down the fixed-function
            // path instead. The complement of `VITASLOP_GXP_KEYS`, and the useful one when
            // a frame is almost right: it answers "is this ONE draw the recompiler's fault"
            // without also un-recompiling the hundreds of draws that are already correct.
            if self.exclude.contains(&key) {
                return None;
            }
            report_inputs(key, gxp);
            report_inputs_order(key, gxp);
            let cache_key = (key, color_format, samples, gxp.cull_mode);
            if !self.pipelines.contains_key(&cache_key) {
                // Name the pair's two containers by their CONTENT hash the moment it is first
                // seen. `Program::hash` is the same value the offline corpus computes, so this
                // one line is what turns a draw key from the frame into the two `.gxp` blobs an
                // offline test can open. Printed once per unique pair, not per draw.
                //
                // >>> THIS IS AN INDEX, NOT A WARNING, AND IT MUST BE ASKED FOR.
                //
                // It was promoted to `warn` with no knob because at `debug` it printed nothing
                // in the runs that needed it - which was a real problem and the wrong fix. "A
                // few dozen lines a run" is true on a menu and false on a title screen deep in
                // a game: MEASURED on the user's device, **~90 unique pairs filled the browser
                // diagnostics panel, which keeps 96 DISTINCT lines, and the capture came back
                // reading `21 earlier DISTINCT line(s) dropped`.** The dropped lines are the
                // ones a capture is taken FOR. An index that evicts the findings is worse than
                // an index nobody can see.
                //
                // So it is gated on actually wanting it - either explicitly, or implicitly by
                // setting any diagnostic that takes a pair KEY, which is the only situation in
                // which "what keys exist" is a question. That keeps the property the promotion
                // was after (a `VITASLOP_GXP_INPUTS=<key>` run that prints nothing can still be
                // told apart from the pair never being drawn) and costs a default run nothing.
                // [[vitaslop-a-diagnostic-can-bury-the-findings]]
                if gxp_key_index_wanted() {
                    let ph = |b: &[u8]| {
                        vitaslop_gxp_shader::Program::parse(b).map(|p| p.hash).unwrap_or(0)
                    };
                    report_warn!(
                        "gxp pair {key:x}: vprog hash {:016x}, fprog hash {:016x}",
                        ph(&gxp.vprog),
                        ph(&gxp.fprog)
                    );
                }
                // ...and whether either half reads uniforms we cannot supply. Here, at the
                // once-per-pair site, so it names the program rather than the API call.
                report_unfed_uniforms(key, "vertex", &gxp.vprog);
                report_unfed_uniforms(key, "fragment", &gxp.fprog);
                enc(&ENC.pipelines_built, 1);
                let built = build_gxp_pipeline(device, color_format, samples, gxp.cull_mode, gxp, key, self.zfix, self.yflip, self.solid, self.nodepth, self.noblend, &mut self.modules);
                self.pipelines.insert(cache_key, built);
            }

            // Split the borrows: the sampler bind group needs the texture-view cache mutably
            // while the pipeline (its layouts, its sampler plan) stays borrowed.
            let negw_mode = self.negw;
            // Which clip-depth remap `inject_clip_fixup` put the vertex stage through. A
            // fragment that WRITES its own depth has to invert exactly that map, and only the
            // renderer knows which one is in force, so it travels in the depth block.
            let zfix_mode: f32 = match self.zfix {
                ZFix::Range => 0.0,
                ZFix::Gl => 1.0,
                // Both leave the guest's window depth in the buffer, so a fragment writing its
                // own depth inverts the same (identity) map.
                ZFix::Off | ZFix::Clamp => 2.0,
                // The buffer holds the guest's own WINDOW depth - the viewport map is already
                // applied - so a fragment writing one in that same encoding needs only the
                // clamp, which is the one part of the forward map it cannot have applied.
                ZFix::Viewport => 3.0,
            };
            let scene_negw = self.scene_negw;
            let scene_depth_fit = self.scene_depth_fit;
            // >>> ASKED ONCE PER RENDERER, NEVER PER DRAW.
            //
            // `Device::features()` is a cheap bitflag read on a native backend and a JS property
            // access plus a walk of the whole `GPUSupportedFeatures` set in the browser, which is
            // the boundary crossing [[vitaslop-browser-host-call-cost]] measured at 91% of a host
            // call's cost. This is on the per-draw path, and a title submits hundreds of draws a
            // frame, so reading it there would have paid that hundreds of times a frame to
            // re-learn a constant. Cached per GxpLive rather than in a global, because "there is
            // only ever one device" is true today and is not a thing this code should assert.
            let bc = *self.bc_supported.get_or_insert_with(|| {
                let f = device.features();
                if f.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
                    BlockFamily::Bc
                } else if f.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2) {
                    BlockFamily::Etc2
                } else {
                    BlockFamily::None
                }
            });
            // Built here rather than lazily at the upload site, because that site holds `self`
            // destructured and cannot reach back for a `&mut`. Three compute shaders, once.
            if self.texenc.is_none() {
                self.texenc = Some(crate::texenc::Transcoder::new(device));
            }
            let GxpLive {
                pipelines,
                texenc,
                views: view_cache,
                views_bytes: view_cache_bytes,
                views_used,
                views_evicted,
                views_epoch,
                views_frame_high,
                views_frame_bytes,
                samplers_by_mode,
                force,
                depth_bgs,
                depth_retired,
                packed,
                packed_by_alloc,
                resident_v,
                resident_i,
                resident_i_seen,
                resident,
                sampler_bgs,
                ..
            } = self;
            let resident = *resident;
            // Borrow the cached pipeline; None = link failed -> fall back.
            let pipe = pipelines.get(&cache_key)?.as_ref()?;
            split_end(t_key, &PREP.key_ns);

            if gxp_dump() {
                let f: Vec<f32> = gxp
                    .vert_sa
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                let attrs: Vec<(u16, u16, u8, u8)> =
                    gxp.attributes.iter().map(|a| (a.reg_index, a.offset, a.gxm_format, a.components)).collect();
                let ff: Vec<f32> = gxp
                    .frag_sa
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                // The guest stream's own extent next to what the indices actually reference: a
                // mesh whose highest index is beyond the captured vertices renders only the
                // triangles that fall inside the buffer, i.e. a PREFIX of the geometry.
                let nverts = gxp.vertices.len() / (gxp.vertex_stride.max(1) as usize);
                let max_index = if gxp.index_u32 {
                    gxp.indices.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).max()
                } else {
                    gxp.indices.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).max()
                }
                .unwrap_or(0);
                report!(
                    "gxp draw key {:x}: vsa_lanes={} vert_sa_lanes={} fsa_lanes={} frag_sa_lanes={} samplers={:?} stride={} idx={} nverts={nverts} max_index={max_index} vbytes={} attrs(reg,off,fmt,comp)={:?}\n  vsa={:?}\n  fsa={:?}",
                    key, pipe.vsa_lanes, f.len(), pipe.fsa_lanes, ff.len(), pipe.samplers, gxp.vertex_stride, gxp.index_count, gxp.vertices.len(), attrs, f, ff
                );
                // For a small mesh, the attribute VALUES as fetched. A quad landing in the
                // wrong place on screen is either its positions or its transform, and those
                // are indistinguishable from the picture; this shows the positions.
                if nverts <= 8 {
                    let stride = gxp.vertex_stride.max(1) as usize;
                    for v in 0..nverts {
                        let vals: Vec<String> = gxp
                            .attributes
                            .iter()
                            .map(|a| {
                                let base = v * stride + a.offset as usize;
                                let c: Vec<String> = (0..a.components as usize)
                                    .map(|i| format!("{:.3}", read_attr_component(&gxp.vertices, base, a.gxm_format, i)))
                                    .collect();
                                format!("r{}=[{}]", a.reg_index, c.join(","))
                            })
                            .collect();
                        report!("    v{v}: {}", vals.join(" "));
                    }
                }
            }

            // Repack the guest vertex stream into the tightly-packed f32 layout the pipeline
            // expects (per the cached repack plan), straight into the pass's vertex arena.
            // Same vertex count/order, so the index buffer is unchanged.
            //
            // Both arenas stay 4-byte aligned, which `set_vertex_buffer` requires and a u32
            // index buffer requires: the packed stride is a whole number of f32s and an index
            // is four bytes, so the padding below never actually fires - it is there so a
            // future format that is not cannot silently produce an unaligned slice.
            // The allocation this stream lives in, which is the fast key - see
            // `packed_by_alloc`. Both halves of the fat pointer, so a shorter stream that
            // happens to start at the same address is a different key rather than a truncation.
            let akey = (key, std::sync::Arc::as_ptr(&gxp.vertices) as *const u8 as usize, gxp.vertices.len());
            let alloc_hit = packed_by_alloc
                .get(&akey)
                .filter(|(src, _)| std::sync::Arc::ptr_eq(src, &gxp.vertices))
                .map(|(_, bytes)| bytes.clone());
            // Whether this exact allocation had been repacked BEFORE this draw - which is the
            // promotion test below, and has to be read before the miss path inserts it.
            let seen_before = alloc_hit.is_some();
            // The packed bytes for this draw, however they were reached.
            let packed_bytes: std::sync::Arc<[u8]> = match alloc_hit {
                Some(bytes) => {
                    split_add(&PREP.packed_hits, 1);
                    bytes
                }
                None => Self::pack_vertices(packed, packed_by_alloc, akey, gxp, &pipe.repack, pipe.packed_stride),
            };
            // Where the draw's vertices will be. A stream the renderer has seen before is placed
            // in the RESIDENT heap and never copied again; a first sighting, and anything the
            // heap declines, goes through the pass arena exactly as it always did.
            let (v_off, v_len, v_resident) = match resident
                .then(|| resident_v.get(&akey, &gxp.vertices))
                .flatten()
            {
                Some((off, len)) => {
                    split_add(&PREP.resident_v_hits, 1);
                    (off, len, true)
                }
                None => {
                    let t = split_start();
                    let off = vdata.len() as u64;
                    vdata.extend_from_slice(&packed_bytes);
                    let len = vdata.len() as u64 - off;
                    while vdata.len() % 4 != 0 {
                        vdata.push(0);
                    }
                    split_end(t, &PREP.arena_ns);
                    split_add(&PREP.arena_bytes, len);
                    // Promote on the SECOND sighting, not the first: the entry only exists
                    // because this exact allocation was repacked before, so a stream the title
                    // builds fresh every frame never reaches the heap and never displaces one
                    // that does. A promotion costs the same upload the arena write costs, once.
                    if resident && seen_before {
                        resident_v.place(queue, akey, &gxp.vertices, &packed_bytes);
                    }
                    (off, len, false)
                }
            };
            // The indices, on exactly the same terms. They have their own allocation - the
            // capture expands every primitive type into a u32 triangle list and caches THAT by
            // the guest buffer it came from - so a mesh can hold its index list still while its
            // vertices move, and the two are placed independently.
            //
            // The key has no pipeline in it: an expanded index list is a function of the guest
            // buffer alone, so two pairs drawing the same mesh share one resident copy.
            let ikey = (0u64, std::sync::Arc::as_ptr(&gxp.indices) as *const u8 as usize, gxp.indices.len());
            let i_seen = resident_i_seen
                .get(&ikey)
                .and_then(|prev| prev.upgrade())
                .is_some_and(|prev| std::sync::Arc::ptr_eq(&prev, &gxp.indices));
            let (i_off, i_len, i_resident) = match resident
                .then(|| resident_i.get(&ikey, &gxp.indices))
                .flatten()
            {
                Some((off, len)) => {
                    split_add(&PREP.resident_i_hits, 1);
                    (off, len, true)
                }
                None => {
                    let t = split_start();
                    let off = idata.len() as u64;
                    idata.extend_from_slice(&gxp.indices);
                    let len = idata.len() as u64 - off;
                    while idata.len() % 4 != 0 {
                        idata.push(0);
                    }
                    split_end(t, &PREP.arena_ns);
                    split_add(&PREP.arena_bytes, len);
                    if resident {
                        if i_seen {
                            resident_i.place(queue, ikey, &gxp.indices, &gxp.indices);
                        } else {
                            if resident_i_seen.len() >= RESIDENT_SEEN_CAP {
                                resident_i_seen.clear();
                            }
                            resident_i_seen.insert(ikey, std::sync::Arc::downgrade(&gxp.indices));
                        }
                    }
                    (off, len, false)
                }
            };

            // The two SA blocks go into the pass's uniform ARENA at dynamic-offset alignment;
            // the bind groups over that arena belong to the shader pair and are built once,
            // after the arena buffer exists (see `ensure_ubo_bgs`).
            let t_uni = split_start();
            let vert_sa = override_sa(key, 'v', &gxp.vert_sa);
            let frag_sa = override_sa(key, 'f', &gxp.frag_sa);
            // The vertex stage's guest-memory window, when this pair's program loads memory:
            // one header vec4 (lane x = the window's guest base address) + the window bytes,
            // in the same dynamic-offset arena as the SA blocks. A draw that arrives WITHOUT
            // the bytes its pipeline needs is dropped with a report - feeding the loads
            // zeroes would render a wrong picture with nothing to say so.
            let mem_off = if pipe.mem_bind_bytes > 0 {
                let Some((base, bytes)) = gxp.mem_window.as_ref() else {
                    static REPORTED: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        tracing::warn!(
                            target: "vitaslop::gpu",
                            key = format_args!("{key:016x}"),
                            "a pipeline with a guest-memory window got a draw with no window \
                             bytes - draw DROPPED"
                        );
                    }
                    return None;
                };
                let off = push_mem_window(udata, pipe.mem_bind_bytes, *base, bytes, ubo_align);
                split_add(&PREP.arena_bytes, pipe.mem_bind_bytes as u64);
                off
            } else {
                0
            };
            let u_off = [
                push_sa(udata, pipe.vsa_lanes, &vert_sa, ubo_align),
                push_sa(udata, pipe.fsa_lanes, &frag_sa, ubo_align),
                mem_off,
            ];
            split_end(t_uni, &PREP.uni_ns);
            let t_samp = split_start();
            let bg2 = Self::make_sampler_bg(
                device, queue, &pipe.layouts[2],
                &[
                    (&pipe.samplers[..], &gxp.textures[..]),
                    (&pipe.vertex_samplers[..], &gxp.vertex_textures[..]),
                ],
                gxp, key,
                view_cache, view_cache_bytes, views_used, views_evicted, views_epoch,
                views_frame_high, views_frame_bytes, samplers_by_mode, *force,
                rendered, depth_rendered, rtt_epoch, reads_snapshot,
                sampler_bgs, bc,
                texenc.as_ref().expect("built at the top of this function"),
            );
            split_end(t_samp, &PREP.sampler_ns);
            let bg2 = bg2?;
            let t_depth = split_start();
            // group3: the scene depth range the injected clip fixup maps through, as one vec4
            // (min, scale, unused, unused) - the same values the fixed-function path uses, so
            // both kinds of draw write comparable depth. Per SCENE, not per draw, so it is
            // cached (see `GxpLive::depth_bgs`); the bit pattern is the key so a changed
            // range builds a new one rather than reusing a stale buffer.
            //
            // Its third lane is the clip-`w` sign correction, which is a property of the DRAW
            // (its projection), not of the scene - so it joins the cache key.
            let corrected = scene_negw;
            let sign: f32 = match (corrected, negw_mode) {
                (false, _) => 1.0,
                (true, NegW::Negate) => -1.0,
                (true, _) => 2.0,
            };
            // Lanes 4 and 5 are this pass's guest depth encoding `a + c/w`, and they are the
            // same numbers the depth-conversion pass writes with: a fragment reading its own
            // POSITION.z and a fragment sampling a converted depth surface must be looking at
            // one quantity, or every soft fade between them compares apples to oranges.
            let (fit_a, fit_c) = scene_depth_fit;
            // Lanes 6 and 7 are the guest's OWN viewport depth mapping (`zScale`, `zOffset`),
            // which `ZFix::Viewport` puts the clip depth through. It is per DRAW - the guest can
            // and does set a different viewport for an overlay than for the world - so like the
            // sign correction above it joins the cache key.
            let (z_scale, z_offset) = gxm_viewport_depth(&gxp.viewport, key);
            let depth_key = (depth_range[0].to_bits() as u64) << 32 | depth_range[1].to_bits() as u64;
            let depth_key = (depth_key, z_scale.to_bits(), z_offset.to_bits());
            if depth_bgs.contains_key(&(key, depth_key, corrected)) {
                enc(&ENC.bind_groups_reused, 1);
            } else {
                // A race moves the depth range every frame, so this cache mints entries that
                // are never asked for again - see the field's doc comment for the measurement.
                // Dropped wholesale rather than growing without bound: the key is the value,
                // so every entry rebuilds byte-identical.
                let mut evicted: Vec<(wgpu::BindGroup, wgpu::Buffer)> = Vec::new();
                if drain_if_at_cap(depth_bgs, DEPTH_BG_CACHE_CAP, &mut evicted) {
                    enc(&ENC.depth_bg_cache_clears, 1);
                    // The buffers go to the renderer's graveyard, NOT to the collector, and not
                    // to an immediate `destroy()` either - a draw prepared earlier this frame
                    // may still name the bind group that owns one. See `depth_retired`.
                    depth_retired.extend(evicted.into_iter().map(|(_, buf)| buf));
                }
            }
            let (bg3, _) = depth_bgs.entry((key, depth_key, corrected)).or_insert_with(|| {
                enc(&ENC.bind_groups_built, 1);
                enc(&ENC.buffers_created, 1);
                enc(&ENC.buffer_bytes, 32);
                let dbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("gxp-depth"),
                    contents: &[
                        depth_range[0].to_le_bytes(),
                        depth_range[1].to_le_bytes(),
                        sign.to_le_bytes(),
                        zfix_mode.to_le_bytes(),
                        fit_a.to_le_bytes(),
                        fit_c.to_le_bytes(),
                        z_scale.to_le_bytes(),
                        z_offset.to_le_bytes(),
                    ]
                    .concat(),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gxp-depth-bind"),
                    layout: &pipe.layouts[3],
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: dbuf.as_entire_binding() }],
                });
                // The buffer is kept, not dropped: it is the only handle `destroy()` can ever
                // be called on, and the bind group does not offer one.
                (bg, dbuf)
            });
            let bg3 = bg3.clone();
            split_end(t_depth, &PREP.depth_ns);

            Some(GxpPrepared {
                key,
                v_off,
                v_len,
                i_off,
                i_len,
                v_resident,
                i_resident,
                index_count: gxp.index_count,
                u_off,
                bg2,
                bg3,
                blend: gxp.blend,
                viewport: gxp.viewport,
                format: color_format,
                samples,
                cull: gxp.cull_mode,
            })
        }

        /// The cached group0/group1 bind group for a prepared draw's shader pair (only called
        /// after `ensure_ubo_bgs` has run for this pass).
        fn ubo_bg(&self, slot: usize, key: u64, format: wgpu::TextureFormat, samples: u32, group: u8) -> &wgpu::BindGroup {
            &self.ubo_bgs[&(slot, key, format, samples, group)]
        }

        /// The cached pipeline for a prepared draw (only called after `prepare` succeeded).
        fn pipeline(&self, key: u64, format: wgpu::TextureFormat, samples: u32, cull: u32) -> &GxpPipeline {
            self.pipelines
                .get(&(key, format, samples, cull))
                .and_then(|p| p.as_ref())
                .expect("prepared key present")
        }

        /// Build the group2 sampler bind group: for each declared sampler unit, upload the
        /// bound texture and bind it with the matching filter sampler. `None` (fall back) if a
        /// unit has no bound texture or needs a 3D texture (not yet mapped).
        #[allow(clippy::too_many_arguments)]
        fn make_sampler_bg(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            layout: &wgpu::BindGroupLayout,
            stages: &[(&[(u8, SamplerDim)], &[crate::gpu::GxpTex])],
            gxp: &GxpRecompile,
            key: u64,
            view_cache: &mut HashMap<(u64, SamplerDim), (wgpu::Texture, wgpu::TextureView)>,
            view_cache_bytes: &mut usize,
            views_used: &mut HashMap<(u64, SamplerDim), (u64, usize, u32, Residency)>,
            views_evicted: &mut HashSet<(u64, SamplerDim)>,
            views_epoch: &u64,
            views_frame_high: &usize,
            views_frame_bytes: &mut usize,
            samplers_by_mode: &mut HashMap<(bool, u32, u32), wgpu::Sampler>,
            force: bool,
            rendered: &HashMap<u32, wgpu::TextureView>,
            depth_rendered: &HashMap<u32, wgpu::TextureView>,
            // What identifies a render-target VIEW for cache purposes: the `rtt_epoch` (bumped
            // when a target or its snapshot texture is created, i.e. when views die) and the set
            // of addresses currently resolving to the SNAPSHOT rather than the live target. The
            // two views of one address are different textures, so which one a group named has to
            // be in its key. See `GxmRenderer::rtt_epoch`.
            rtt_epoch: u64,
            reads_snapshot: &HashSet<u32>,
            sampler_bgs: &mut HashMap<(u64, u64), (wgpu::BindGroup, Vec<(u64, SamplerDim)>)>,
            // Which block family this device accepts - resolved once per renderer by the caller.
            bc: BlockFamily,
            // The GPU transcoder's pipelines, built once per renderer by the caller.
            texenc: &crate::texenc::Transcoder,
        ) -> Option<wgpu::BindGroup> {
            // What this group will NAME, accumulated as the units resolve, so the finished
            // group can be looked up instead of rebuilt. See `GxpLive::sampler_bgs`.
            let mut sel: u64 = 0xcbf2_9ce4_8422_2325;
            let mut mix = |v: u64| {
                sel ^= v;
                sel = sel.wrapping_mul(0x0000_0100_0000_01b3);
            };
            // A view that belongs to a target THIS frame rendered cannot be cached across
            // frames - the texture behind it is the frame's, not the cache's.
            let mut per_frame = false;
            // Both stages' samplers share this group, in declaration order: the fragment's
            // first, the vertex's after. The layout, the WGSL and this must agree on that order
            // or a sample reads the wrong texture.
            let total: usize = stages.iter().map(|(s, _)| s.len()).sum();
            if total == 0 {
                // An EMPTY group is still a GPU object, and building one per draw is the
                // same waste as building a full one per draw. It names nothing, so one per
                // pipeline is all there can ever be.
                let (bg, _) = sampler_bgs.entry((key, sel)).or_insert_with(|| {
                    note_sampler_bg(false);
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("gxp-samplers-empty"),
                        layout,
                        entries: &[],
                    });
                    // Names nothing, so no eviction can ever invalidate it.
                    (bg, Vec::new())
                });
                return Some(bg.clone());
            }
            // Upload every needed texture first so the views outlive the bind-group build.
            let mut views: Vec<wgpu::TextureView> = Vec::with_capacity(total);
            // The view-cache keys this group NAMES, kept so an eviction can invalidate exactly
            // the groups that went stale rather than all of them. A per-frame view contributes
            // nothing here because such a group is never cached in the first place.
            let mut named_views: Vec<(u64, SamplerDim)> = Vec::with_capacity(total);
            // Cloning a `TextureView` is a refcount bump, so cached views are shared, not copied.
            // `(linear, addr_mode_u, addr_mode_v)` per bound view - the sampler state the guest
            // set on that texture, not a global default.
            let mut sampler_state: Vec<(bool, u32, u32)> = Vec::with_capacity(total);
            for &(samplers, textures) in stages {
            for &(unit, want) in samplers {
                let bound = textures.iter().find(|t| t.unit == unit);
                // The bound texture must actually supply the dimension the shader declared: a
                // cube sampler needs the six captured faces, a 2D sampler a single image. A
                // mismatch means the container and the guest state disagree, so bind nothing.
                let usable = bound.filter(|gt| match want {
                    SamplerDim::Cube => gt.tex.faces == 6,
                    SamplerDim::Two => gt.tex.faces == 1,
                    SamplerDim::Three => false,
                });
                // A DEPTH buffer this frame rendered, matched EXACTLY and checked FIRST.
                //
                // Order is load-bearing. A title allocates a scene's depth next to its colour
                // (one racer puts them 256 bytes apart), so `rendered_alias`, which matches by
                // range, claims the depth address for the colour target and the pass reads a
                // colour where it asked for a distance - which is why its glow, blur and
                // soft-particle passes rendered pure black. Exact-matching the depth first is
                // what tells the two apart, and the address comes from the guest's own
                // `SceGxmDepthStencilSurface`, not from a guess about the layout.
                let depth_hit = (want == SamplerDim::Two)
                    .then(|| usable.and_then(|gt| depth_rendered.get(&gt.tex.data_addr)))
                    .flatten();
                let aliased = depth_hit
                    .is_none()
                    .then(|| {
                        (want == SamplerDim::Two)
                            .then(|| usable.and_then(|gt| rendered_alias(rendered, gt.tex.data_addr, key, unit)))
                            .flatten()
                    })
                    .flatten();
                match usable {
                    Some(gt) if depth_hit.is_some() => {
                        report_depth_sample_bound(key, unit, gt.tex.data_addr);
                        // A view of a persistent target, so it is cacheable - keyed by the
                        // address it names and by the epoch that says the view is still alive.
                        if super::rtt_bg_cache() {
                            mix(0x0d0d_0000_0000_0000 ^ gt.tex.data_addr as u64 ^ (rtt_epoch << 32));
                        } else {
                            per_frame = true;
                        }
                        views.push(depth_hit.unwrap().clone());
                        sampler_state.push((gt.tex.filter_linear, gt.tex.addr_mode_u, gt.tex.addr_mode_v));
                    }
                    // Sampling a buffer an earlier pass in THIS frame rendered: bind that
                    // render. Only 2D targets - a cube face is never a GXM render target.
                    Some(gt) if aliased.is_some() => {
                        let a = aliased.unwrap();
                        if super::rtt_bg_cache() {
                            mix(0x0c0c_0000_0000_0000
                                ^ a as u64
                                ^ (rtt_epoch << 32)
                                ^ if reads_snapshot.contains(&a) { 1 << 20 } else { 0 });
                        } else {
                            per_frame = true;
                        }
                        views.push(rendered[&a].clone());
                        sampler_state.push((gt.tex.filter_linear, gt.tex.addr_mode_u, gt.tex.addr_mode_v));
                    }
                    Some(gt) => {
                        // A texture whose data pointer is null decodes to a 1x1 ZERO texel. That
                        // is the faithful substitute (see the runtime's zero-handle report), but
                        // a recompiled shader sampling it multiplies its whole output by zero,
                        // and the pass comes out BLACK with nothing in the log tying the two
                        // together. Name the pair and the unit, once each: this is a draw
                        // silently losing its content, which is the same class of event as a
                        // fallback and gets the same unconditional report.
                        if gt.tex.data_addr == 0 {
                            report_zero_texel_sample(key, unit);
                        }
                        let cache_key = (gt.tex.key, want);
                        mix(gt.tex.key);
                        if view_cache.contains_key(&cache_key) {
                            enc(&ENC.tex_view_cached, 1);
                        }
                        if !view_cache.contains_key(&cache_key) {
                            // Bound the cache BY BYTES, evicting ENTRY BY ENTRY and never one
                            // this frame has already used. See `GxpLive::views_used` for the
                            // measurement that killed the old wholesale clear: it turned "the
                            // working set is 30% over budget" into a 0% hit rate and a 1 fps
                            // frame, because it fired part-way through every frame and threw
                            // away exactly what the next frame was about to ask for.
                            //
                            // The keys are content fingerprints, so an eviction costs a
                            // re-upload and never correctness - that part of the old argument
                            // was right, and it is what makes any eviction policy safe here.
                            *view_cache_bytes += uploaded_texture_bytes(bc, &gt.tex);
                            // The budget, but never below what ONE FRAME needs - see
                            // `GxpLive::views_frame_high`. Those bytes have to be resident
                            // during the frame regardless, so this floor costs nothing that
                            // was avoidable and is what lets the next frame hit.
                            let budget = tex_cache_budget_bytes().max(*views_frame_high);
                            if *view_cache_bytes >= budget {
                                let epoch = *views_epoch;
                                // Oldest first, so a cache under steady pressure sheds what has
                                // gone longest unused rather than whatever the hash order
                                // happens to yield.
                                let mut stale: Vec<((u64, SamplerDim), u64)> = views_used
                                    .iter()
                                    .filter(|(_, (used, ..))| *used != epoch)
                                    .map(|(k, (used, ..))| (*k, *used))
                                    .collect();
                                stale.sort_by_key(|(_, used)| *used);
                                let mut evicted = 0usize;
                                let mut just_evicted: HashSet<(u64, SamplerDim)> = HashSet::default();
                                for (k, _) in stale {
                                    if *view_cache_bytes < budget {
                                        break;
                                    }
                                    if let Some((tex, _view)) = view_cache.remove(&k) {
                                        // Release the GPU memory NOW. Dropping these two handles
                                        // only makes the texture collectable in the browser, and
                                        // a cache that evicts every frame would then be handing
                                        // the collector megabytes a frame to catch up on.
                                        tex.destroy();
                                        enc(&ENC.textures_destroyed, 1);
                                        let bytes = views_used.remove(&k).map_or(0, |(_, b, ..)| b);
                                        *view_cache_bytes = view_cache_bytes.saturating_sub(bytes);
                                        // Remember it, so a later upload of the same key can be
                                        // attributed as THRASH rather than as a cold upload.
                                        // Bounded by dropping: forgetting an old eviction
                                        // under-counts thrash and nothing else.
                                        if views_evicted.len() < VIEW_EVICTED_KEYS_CAP {
                                            views_evicted.insert(k);
                                        }
                                        just_evicted.insert(k);
                                        evicted += 1;
                                    }
                                }
                                if evicted > 0 {
                                    enc(&ENC.tex_view_evict_passes, 1);
                                    enc(&ENC.tex_view_evicted, evicted as u64);
                                    // >>> DROP EXACTLY THE SAMPLER BIND GROUPS THAT WENT STALE,
                                    // and not the whole cache.
                                    //
                                    // A cached group names specific `TextureView`s, and an
                                    // evicted key can be re-uploaded later as a DIFFERENT view,
                                    // so any group naming it is dead - but only those. This used
                                    // to bump a global `views_gen` folded into every group's
                                    // key, which was right while eviction was WHOLESALE (every
                                    // group really was dead) and became far too blunt when it
                                    // went per entry. The policy changed and the invalidation
                                    // did not follow it, exactly as the counter NAMES did not.
                                    //
                                    // MEASURED on the user's device at 1.07 eviction passes a
                                    // frame: 446.5 sampler bind groups REBUILT per frame against
                                    // 792.6 reused - one cold entry a frame was throwing away a
                                    // third of the cache that exists to stop per-draw GPU object
                                    // creation ([[vitaslop-per-pass-render-arenas]]).
                                    sampler_bgs.retain(|_, (_, named)| {
                                        !named.iter().any(|k| just_evicted.contains(k))
                                    });
                                } else {
                                    // Nothing was evictable: every entry is in use by the
                                    // frame being encoded. Exceeding the budget is the only
                                    // way to finish it, and it is REPORTED rather than
                                    // silently done - a working set this size is the thing to
                                    // fix, and on a phone it is also how a worker gets killed
                                    // with no error.
                                    report_texture_budget_exceeded(
                                        *view_cache_bytes,
                                        views_used,
                                        *views_epoch,
                                    );
                                }
                            }
                            if views_evicted.remove(&cache_key) {
                                enc(&ENC.tex_reuploaded_after_evict, 1);
                            }
                            let tex = upload_gxp_texture(
                                device,
                                queue,
                                &gt.tex,
                                bc,
                                texenc,
                            );
                            let view = tex.create_view(&wgpu::TextureViewDescriptor {
                                dimension: Some(want.view_dimension()),
                                ..Default::default()
                            });
                            view_cache.insert(cache_key, (tex, view));
                        }
                        // Stamp it as used by THIS frame, hit or miss - that stamp is what
                        // makes it ineligible for eviction while the frame is still being
                        // built. The size rides along so an eviction can subtract it.
                        let bytes = uploaded_texture_bytes(bc, &gt.tex);
                        // Count it toward THIS frame's working set the first time the frame
                        // touches it - a texture sampled by two hundred draws is two hundred
                        // lookups and one set of bytes, and counting per lookup would inflate
                        // the learned floor by that factor.
                        // The passthrough flag rides along so the working-set breakdown can say
                        // which formats are still EXPANDING, rather than repeating a static
                        // claim about what BC "would" cost. That claim was aspirational for as
                        // long as the note existed: nothing uploaded compressed, and the report
                        // said "BC, uploaded compressed" anyway.
                        let resident = match compressed_upload(bc, &gt.tex) {
                            Some(c) if c.transcoded => Residency::Transcoded,
                            Some(_) => Residency::Passthrough,
                            None => Residency::Decoded,
                        };
                        if views_used
                            .insert(cache_key, (*views_epoch, bytes, gt.tex.base_format, resident))
                            .is_none_or(|(used, ..)| used != *views_epoch)
                        {
                            *views_frame_bytes += bytes;
                        }
                        views.push(view_cache[&cache_key].1.clone());
                        named_views.push(cache_key);
                        sampler_state.push((gt.tex.filter_linear, gt.tex.addr_mode_u, gt.tex.addr_mode_v));
                    }
                    // A volume sampler (not yet mapped), or a unit whose real texture we could
                    // not capture/decode: strict mode falls back; force mode binds a neutral
                    // fallback so geometry still renders (a diagnostic, never the default).
                    None => {
                        // A fresh fallback view per draw, so this group is per-frame too.
                        per_frame = true;
                        if !force {
                            report_fallback(
                                key,
                                &format!(
                                    "sampler unit {unit} wants {want:?} but the bound units are {:?}",
                                    gxp.textures
                                        .iter()
                                        .map(|t| (t.unit, t.tex.faces))
                                        .collect::<Vec<_>>()
                                ),
                            );
                            return None;
                        }
                        views.push(make_fallback_view(device, queue, want.view_dimension()));
                        sampler_state.push((false, 0, 0));
                    }
                }
            }
            }
            // The sampler state belongs in the key too: two draws can bind the same image
            // through different filter/wrap modes, and that is a different bind group.
            for &(lin, u, v) in &sampler_state {
                mix((lin as u64) << 62 | (u as u64) << 32 | v as u64);
            }
            // The finished group is a pure function of what it names, so ask for it before
            // building it. Only when nothing in it belongs to THIS frame's render targets.
            if !per_frame {
                if let Some((bg, _)) = sampler_bgs.get(&(key, sel)) {
                    note_sampler_bg(true);
                    return Some(bg.clone());
                }
            }
            // Create every sampler this bind group needs FIRST, so the map is not borrowed
            // mutably while the entries hold shared references into it.
            for &st in &sampler_state {
                samplers_by_mode.entry(st).or_insert_with(|| make_gxp_sampler(device, st.0, st.1, st.2));
            }
            let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(total * 2);
            for (i, view) in views.iter().enumerate() {
                let samp = &samplers_by_mode[&sampler_state[i]];
                entries.push(wgpu::BindGroupEntry { binding: i as u32 * 2, resource: wgpu::BindingResource::TextureView(view) });
                entries.push(wgpu::BindGroupEntry { binding: i as u32 * 2 + 1, resource: wgpu::BindingResource::Sampler(samp) });
            }
            note_sampler_bg(false);
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gxp-samplers"), layout, entries: &entries });
            if !per_frame {
                // Bounded the way every other content-addressed cache here is: the key is a
                // fingerprint of what the group names, so dropping it costs a rebuild and
                // never correctness.
                if sampler_bgs.len() >= SAMPLER_BG_CACHE_CAP {
                    sampler_bgs.clear();
                }
                sampler_bgs.insert((key, sel), (bg.clone(), named_views));
            }
            Some(bg)
        }
    }

    /// The texture dimension a recompiled fragment's sampler binding needs. It comes from the
    /// GXP container (the sampler parameter's cube flag plus the SMP coordinate count), and it
    /// must match on both sides: the WGSL declares the type, the bind-group layout declares the
    /// view dimension, and the bound texture must be able to supply it.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum SamplerDim {
        Two,
        Three,
        Cube,
    }

    impl SamplerDim {
        fn view_dimension(self) -> wgpu::TextureViewDimension {
            match self {
                SamplerDim::Two => wgpu::TextureViewDimension::D2,
                SamplerDim::Three => wgpu::TextureViewDimension::D3,
                SamplerDim::Cube => wgpu::TextureViewDimension::Cube,
            }
        }
    }

    /// A REPEAT sampler for the recompiler path (point or linear), mirroring the
    /// fixed-function samplers.
    /// One `SceGxmTextureAddrMode` as a wgpu address mode.
    ///
    /// GXM's four "border" variants differ only in what a fetch outside the texture returns
    /// (a border colour we do not model); their EDGE behaviour is the clamp, which is what
    /// matters for the coordinates a shader actually produces. Mapping them to clamp is the
    /// closest available behaviour, and far closer than the repeat they used to get.
    fn gxm_addr_mode(mode: u32) -> wgpu::AddressMode {
        match mode {
            1 | 3 => wgpu::AddressMode::MirrorRepeat, // MIRROR, MIRROR_CLAMP
            2 | 5..=7 => wgpu::AddressMode::ClampToEdge, // CLAMP + the three border clamps
            _ => wgpu::AddressMode::Repeat,          // REPEAT, REPEAT_IGNORE_BORDER
        }
    }

    fn make_gxp_sampler(device: &wgpu::Device, linear: bool, u: u32, v: u32) -> wgpu::Sampler {
        let f = if linear { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest };
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("gxp-sampler"),
            address_mode_u: gxm_addr_mode(u),
            address_mode_v: gxm_addr_mode(v),
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: f,
            min_filter: f,
            // Trilinear across the generated chain (see `build_mip_chain`). A guest that asked
            // for point filtering still gets point filtering WITHIN a level; what this changes
            // is that a minified surface reads a level sized for it instead of aliasing.
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        })
    }

    /// Box-filter a mip chain down from one RGBA8 image, and return it laid out the way
    /// `TextureDataOrder::LayerMajor` wants (every level of layer 0, then every level of layer 1).
    ///
    /// The guest's textures carry their own mip levels and the hardware samples them; we decode
    /// only level 0, so a minified surface was being point-sampled from the full-size image.
    /// That is not a subtle difference: the track receding to the horizon covers a 512-texel
    /// texture in a few dozen pixels, and every pixel lands on an unrelated texel, which reads
    /// as dense white SPECKLE over the whole distant road rather than as a road.
    fn build_mip_chain(w: u32, h: u32, layers: u32, level0: &[u8]) -> (Vec<u8>, u32) {
        let levels = 32 - w.max(h).max(1).leading_zeros();
        let layer_texels = (w as usize) * (h as usize);
        let mut out = Vec::with_capacity(level0.len() * 2);
        for layer in 0..layers as usize {
            let base = layer * layer_texels * 4;
            let mut src: Vec<u8> = level0[base..base + layer_texels * 4].to_vec();
            let (mut sw, mut sh) = (w, h);
            out.extend_from_slice(&src);
            for _ in 1..levels {
                let (dw, dh) = ((sw / 2).max(1), (sh / 2).max(1));
                let mut dst = vec![0u8; (dw as usize) * (dh as usize) * 4];
                for y in 0..dh as usize {
                    for x in 0..dw as usize {
                        for c in 0..4usize {
                            // Average the up-to-four source texels this one covers. On an odd
                            // dimension the second sample repeats the first, which is the
                            // ordinary way a box filter handles a non-power-of-two level.
                            let x0 = (2 * x).min(sw as usize - 1);
                            let x1 = (2 * x + 1).min(sw as usize - 1);
                            let y0 = (2 * y).min(sh as usize - 1);
                            let y1 = (2 * y + 1).min(sh as usize - 1);
                            let at = |xx: usize, yy: usize| src[(yy * sw as usize + xx) * 4 + c] as u32;
                            dst[(y * dw as usize + x) * 4 + c] =
                                ((at(x0, y0) + at(x1, y0) + at(x0, y1) + at(x1, y1) + 2) / 4) as u8;
                        }
                    }
                }
                out.extend_from_slice(&dst);
                src = dst;
                sw = dw;
                sh = dh;
            }
        }
        (out, levels)
    }

    /// Upload a decoded [`GxmTexture`] to a GPU texture for the recompiler path, in whichever
    /// of the two seams it was decoded onto (see [`TexelSeam`]).
    fn upload_gxp_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        t: &GxmTexture,
        bc: BlockFamily,
        texenc: &crate::texenc::Transcoder,
    ) -> wgpu::Texture {
        let (w, h) = (t.width.max(1), t.height.max(1));
        // A cube map uploads as six array layers; the view below then reads them as a cube.
        let layers = t.faces.max(1);
        // >>> THE PASSTHROUGH: the guest's own blocks, straight to the GPU, no decode.
        //
        // Everything that decides whether this is legal was decided where the bytes were laid
        // out (`vitaslop_runtime::render::compressed_source`) - block geometry, identity channel
        // swizzle, a block-aligned size, and a real mip chain. What is left here is the DEVICE,
        // and one report: the counters have to say which path a texture took, or a working set
        // that fails to shrink is indistinguishable from a passthrough that never fired.
        if let Some(c) = compressed_upload(bc, t) {
            // >>> THE BLOCKS THAT DO NOT EXIST YET ARE MADE HERE, ON THE GPU.
            //
            // A `CompressedData::Gpu` upload carries the guest's own bytes and a description of
            // how to read them, and nothing has been decoded or encoded on the CPU at all. The
            // transcoder runs the decode, the mip chain and the block encode in compute shaders
            // and copies the result straight into the texture. A refusal here is not a failure:
            // every shape it declines falls through to the ordinary decode below, which produces
            // the same picture and only costs more.
            if let CompressedData::Gpu(plan) = &c.data {
                if let Some(tex) = texenc.run(device, queue, c, plan, t.gamma) {
                    enc(&ENC.tex_uploaded, 1);
                    enc(&ENC.tex_upload_bytes, c.byte_len() as u64);
                    enc(&ENC.tex_uploaded_compressed, 1);
                    enc(&ENC.tex_encoded_on_gpu, 1);
                    return tex;
                }
                enc(&ENC.tex_gpu_encode_refused, 1);
                report_gpu_transcode_refused(t.base_format);
            } else {
            enc(&ENC.tex_uploaded, 1);
            enc(&ENC.tex_upload_bytes, c.byte_len() as u64);
            enc(&ENC.tex_uploaded_compressed, 1);
            // The compressed data's OWN dimensions - see `CompressedUpload::width`. They equal
            // the guest's today; using `w`/`h` here anyway would let anything that ever changed
            // that declare a 2048x2048 texture over a smaller buffer, which is a validation
            // error at best and a read past the buffer at worst.
            let (w, h) = (c.width.max(1), c.height.max(1));
            let CompressedData::Cpu(bytes) = &c.data else {
                unreachable!("the GPU arm returns or falls through above")
            };
            return device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("gxp-tex-bc"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: layers },
                    mip_level_count: c.levels,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: block_wgpu_format(c.format, t.gamma),
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                bytes,
            );
            }
        }
        let bpt = t.texel.bytes_per_texel();
        // Guard against a short pixel buffer (a not-fully-decoded format): pad to the full size.
        let need = (w as usize) * (h as usize) * (layers as usize) * bpt;
        let data: std::borrow::Cow<[u8]> = if t.rgba.len() >= need {
            std::borrow::Cow::Borrowed(&t.rgba[..need])
        } else {
            let mut v = t.rgba.to_vec();
            v.resize(need, 0);
            std::borrow::Cow::Owned(v)
        };
        // >>> THE HALF SEAM UPLOADS LEVEL 0 ONLY, deliberately, and it is not the "level-0-only
        // reads as white speckle" mistake. That one was about IMAGES, which are minified and
        // need a filtered chain. A texture on this seam is a DATA lookup fetched at an explicit
        // LOD 0 by a vertex program - the case that has no derivatives and never minifies - so a
        // chain would be bytes nothing samples, and box-averaging coordinates would be
        // meaningless in exactly the way averaging depths is.
        // The SAME predicate the budget uses - see `uploaded_texture_bytes` for why they share
        // one function rather than each writing it out.
        let (data, mip_level_count) = if !mips_for_seam(t.texel) {
            (data.into_owned(), 1)
        } else {
            build_mip_chain(w, h, layers, &data)
        };
        // Count the bytes that ACTUALLY cross into the driver, mips included. The decoder's
        // output is the level-0 size; the mip chain adds a third more, and a tally taken from
        // the decoder would under-report the upload by exactly that much.
        enc(&ENC.tex_uploaded, 1);
        enc(&ENC.tex_upload_bytes, data.len() as u64);
        device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("gxp-tex"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: layers },
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // A GAMMA-CORRECT texture is sRGB-DECODED by the hardware sampler before
                // filtering. Uploading the same bytes through an sRGB format puts the decode
                // in exactly that place, so nothing downstream has to know.
                // `rgba16float` is filterable in core WebGPU, so the half seam keeps the same
                // sampler choices the byte seam has. There is no sRGB half format and there
                // needs to be none: gamma is a property of colour, and nothing that reaches
                // this seam is colour.
                format: match (t.texel, t.gamma) {
                    (TexelSeam::Rgba16Float, _) => wgpu::TextureFormat::Rgba16Float,
                    (TexelSeam::Rgba8, true) => wgpu::TextureFormat::Rgba8UnormSrgb,
                    (TexelSeam::Rgba8, false) => wgpu::TextureFormat::Rgba8Unorm,
                },
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &data,
        )
    }

    /// A 1x1 (or 1x1x1) neutral-grey fallback texture view of the given dimension, for the
    /// `VITASLOP_GXP_FORCE` diagnostic: bound where a sampler's real texture is unavailable so
    /// the recompiled geometry still renders. Opaque alpha so an alpha-test does not discard.
    fn make_fallback_view(device: &wgpu::Device, queue: &wgpu::Queue, dim: wgpu::TextureViewDimension) -> wgpu::TextureView {
        let is_3d = dim == wgpu::TextureViewDimension::D3;
        let tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("gxp-fallback"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: if is_3d { wgpu::TextureDimension::D3 } else { wgpu::TextureDimension::D2 },
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[180, 180, 180, 255],
        );
        tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(dim), ..Default::default() })
    }

    /// Build a group0/group1 uniform bind group from raw guest SA bytes, sized to the WGSL
    /// `array<vec4<f32>, ceil(lanes/4)>` and zero-padded. An empty (0-lane) stage gets an
    /// empty bind group so the pipeline layout's group is still satisfied at draw time.
    /// Append one stage's SA block to the pass's uniform arena at dynamic-offset alignment,
    /// zero-padded to the `array<vec4<u32>, ceil(lanes/4)>` the WGSL declares, and return its
    /// byte offset.
    ///
    /// A zero-lane stage still gets an offset (0) because a bind group with no entries takes
    /// no dynamic offsets at all - the value is simply never used.
    fn push_sa(udata: &mut Vec<u8>, lanes: u32, guest: &[u8], align: u64) -> u32 {
        if lanes == 0 {
            return 0;
        }
        // >>> ALIGNED IN ONE `resize`, NOT ONE BYTE AT A TIME.
        //
        // `ubo_align` is the adapter's minimum uniform-buffer dynamic offset alignment - 256
        // bytes on every desktop and mobile adapter this runs on - so the byte-at-a-time loop
        // this replaces pushed up to 255 bytes through `Vec::push`, with its length and
        // capacity check each, TWICE PER DRAW. On a race frame of ~990 draws that is a
        // quarter of a million pushes per presented frame to write padding, and it is most of
        // what `prepare split`'s `uniforms` line (0.52 ms/present) was timing.
        let pad = udata.len() as u64 % align;
        if pad != 0 {
            udata.resize(udata.len() + (align - pad) as usize, 0);
        }
        let off = udata.len() as u32;
        let need = (lanes.div_ceil(4) as usize) * 16;
        let n = guest.len().min(need);
        udata.extend_from_slice(&guest[..n]);
        udata.resize(off as usize + need, 0);
        off
    }

    /// Append one draw's guest-MEMORY-WINDOW block to the pass's uniform arena at
    /// dynamic-offset alignment: a header vec4 whose lane x is the window's guest base
    /// address, then the window's bytes, padded to the pipeline's declared binding size
    /// (`GxpPipeline::mem_bind_bytes`). The same arena discipline as [`push_sa`].
    fn push_mem_window(udata: &mut Vec<u8>, bind_bytes: u32, base: u32, window: &[u8], align: u64) -> u32 {
        let pad = udata.len() as u64 % align;
        if pad != 0 {
            udata.resize(udata.len() + (align - pad) as usize, 0);
        }
        let off = udata.len() as u32;
        udata.extend_from_slice(&base.to_le_bytes());
        udata.resize(off as usize + 16, 0);
        let need = bind_bytes as usize;
        let n = window.len().min(need - 16);
        udata.extend_from_slice(&window[..n]);
        udata.resize(off as usize + need, 0);
        off
    }

    fn make_uniform_bg(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, lanes: u32, guest: &[u8]) -> wgpu::BindGroup {
        if lanes == 0 {
            return device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("gxp-ubo-empty"), layout, entries: &[] });
        }
        let need = (lanes.div_ceil(4) as usize) * 16;
        let mut data = vec![0u8; need];
        let n = guest.len().min(need);
        data[..n].copy_from_slice(&guest[..n]);
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gxp-ubo"),
            contents: &data,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gxp-ubo-bind"),
            layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
        })
    }

    /// Convert an IEEE-754 half (binary16) to f32.
    fn half_to_f32(h: u16) -> f32 {
        let sign = (h >> 15) & 1;
        let exp = (h >> 10) & 0x1f;
        let mant = h & 0x3ff;
        let v = if exp == 0 {
            (mant as f32) * (1.0 / 16_777_216.0) // subnormal: mant * 2^-24
        } else if exp == 0x1f {
            if mant == 0 { f32::INFINITY } else { f32::NAN }
        } else {
            (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
        };
        if sign == 1 { -v } else { v }
    }

    /// Diagnostic (`VITASLOP_GXP_INPUTS=<hex-key>[,<hex-key>]` or `=all`): print, ONCE per
    /// shader pair, every value the guest actually fed that draw - each declared uniform
    /// decoded through its own declared type, and the observed range of every vertex
    /// attribute component over the draw's whole stream.
    ///
    /// This exists because the two halves of a wrong picture are indistinguishable from the
    /// frame: a coordinate that comes out wrong is either a uniform we are reading through the
    /// wrong layout or an attribute we are decoding wrong, and every other diagnostic here
    /// (`_PROBE`, `_VPROBE`, `_KEYCOLOR`) observes the shader's OUTPUT, which is downstream of
    /// both. A uniform printed as its parameter declares it - `mainScaleBias F16[4] = (1, 1,
    /// 0.25, 0.25)` - settles in one line what a register probe cannot settle at all.
    fn report_inputs(key: u64, gxp: &GxpRecompile) {
        // Through the knob table, not `std::env::var`: the browser has no environment, and this
        // is the diagnostic that answers "what was this draw FED" - which is the live question
        // for a defect that reproduces in the browser and nowhere a file can be written.
        // Resolved once - see `KeySpec` for the per-draw cost this used to pay in every run.
        use std::sync::OnceLock;
        static SPEC: OnceLock<KeySpec> = OnceLock::new();
        if !SPEC.get_or_init(|| KeySpec::resolve("VITASLOP_GXP_INPUTS")).wants(key) {
            return;
        }
        // Dedupe on the pair AND on the inputs themselves, not on the pair alone: one pair is
        // submitted many times a frame with DIFFERENT uniforms (that is what a per-draw uniform
        // buffer is for), and reporting only the first submission is how a diagnostic ends up
        // describing a draw that is not the one being investigated.
        //
        // >>> THE VERTEX BYTES ARE PART OF "the inputs", and leaving them out of this hash made
        // this diagnostic LIE. It reports attribute ranges and per-vertex values, so a draw whose
        // only per-frame change is in the vertex stream - a UI element faded through a per-vertex
        // COLOUR, which is how this title animates its text - matched an earlier submission on
        // uniforms and textures alone and was never printed again. The frame showed ten distinct
        // alphas while this said there were four sets of inputs, and the honest reading of that
        // contradiction was that the fade did not come through a uniform. It cost a wrong
        // conclusion before it cost anything else.
        //
        // Hashing the whole vertex buffer is real work on a world mesh. It is affordable because
        // nothing here runs unless the knob is set, and a diagnostic that silently under-reports
        // is worth less than one that is slow.
        use std::hash::{Hash, Hasher};
        use std::sync::Mutex;
        static SEEN: OnceLock<Mutex<HashSet<(u64, u64)>>> = OnceLock::new();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        gxp.vert_sa.hash(&mut h);
        gxp.frag_sa.hash(&mut h);
        gxp.vertices.hash(&mut h);
        gxp.vertex_stride.hash(&mut h);
        for t in gxp.textures.iter().chain(gxp.vertex_textures.iter()) {
            (t.unit, t.tex.data_addr, t.tex.width, t.tex.height).hash(&mut h);
        }
        let inputs_hash = h.finish();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert((key, inputs_hash)) {
            return;
        }
        for (stage, bytes, blob) in
            [("vertex", &gxp.vert_sa, &gxp.vprog), ("fragment", &gxp.frag_sa, &gxp.fprog)]
        {
            let Ok(program) = vitaslop_gxp_shader::Program::parse(blob) else { continue };
            // The raw words too, not only the decoded parameters. A parameter is decoded through
            // its declared offset/type, so a value that reads wrong is either the bytes or that
            // decode - and only the bytes can tell the two apart.
            // The fragment bank's guest ADDRESS goes on the line too - it is what a store watch
            // is pointed at, and without it "who wrote this uniform" needs a second run to
            // discover the address before it can even start.
            let at = match stage {
                "fragment" if gxp.frag_sa_addr != 0 => format!(" at {:#x}", gxp.frag_sa_addr),
                _ => String::new(),
            };
            report_knob!(
                "gxp inputs {key:016x} {stage}{at}: default uniform buffer is {} bytes for {} declared registers, raw = {}",
                bytes.len(),
                program.default_uniform_regs,
                bytes
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        format!("{:08x}", u32::from_le_bytes(w))
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            for p in &program.parameters {
                use vitaslop_gxp_shader::container::ParamCategory;
                if p.category != ParamCategory::Uniform {
                    continue;
                }
                let vals = decode_uniform(bytes, p);
                report_knob!(
                    "gxp inputs {key:016x} {stage}:   {} {:?}[{}] at reg {} = {}",
                    p.name, p.ptype, p.component_count, p.resource_index, vals
                );
            }
        }
        // The guest's VIEWPORT, which is what turns the vertex program's clip output into
        // The guest's CULL MODE, which no pipeline here applies (every one of them is built
        // with `cull_mode: None`). It belongs in this report for the same reason the viewport
        // does: it is state the draw was FED, it is not visible in the shader or the uniforms,
        // and ignoring it changes the picture. A fullscreen overlay triangle that the guest
        // expects to be culled is drawn instead, over everything.
        report_knob!(
            "gxp inputs {key:016x} cull: mode={} ({}){}",
            gxp.cull_mode,
            match gxp.cull_mode {
                0 => "NONE",
                1 => "CW",
                2 => "CCW",
                _ => "unknown",
            },
            match gxm_cull_face(gxp.cull_mode) {
                Some(wgpu::Face::Front) => "  - APPLIED, front faces discarded",
                Some(wgpu::Face::Back) => "  - APPLIED, back faces discarded",
                None => "",
            }
        );
        // target pixels. A post-process pass that samples a source at a scale/bias only lands
        // right if the source was RENDERED where the pass thinks it was, and the viewport is
        // the one piece of that which is neither in the shader nor in the uniforms.
        report_knob!(
            "gxp inputs {key:016x} viewport: xOffset={} xScale={} yOffset={} yScale={} zOffset={} zScale={}{}",
            gxp.viewport[0], gxp.viewport[1], gxp.viewport[2],
            gxp.viewport[3], gxp.viewport[4], gxp.viewport[5],
            if gxp.viewport.iter().all(|v| *v == 0.0) { "  (all zero = the guest left the default)" } else { "" }
        );
        // What each declared sampler is actually bound to. A post-process pass's whole content
        // is the geometry of the buffer it reads, and "unit 1" in a shader and "unit 1" in the
        // guest's texture state are only the same thing if the binding says so.
        //
        // BOTH stages, and the vertex one is not a footnote: a vertex program that samples
        // builds its GEOMETRY from what it reads, so an unbound or empty vertex texture is a
        // draw with no mesh rather than a surface with the wrong colours. Leaving it out of
        // this report is what left "the campaign map body is black" a content question nobody
        // could answer - the fragment side said everything was bound, and it was.
        for (stage, blob, bound) in [
            ("sampler", &gxp.fprog, &gxp.textures),
            ("vertex sampler", &gxp.vprog, &gxp.vertex_textures),
        ] {
            let Ok(program) = vitaslop_gxp_shader::Program::parse(blob) else { continue };
            for p in &program.parameters {
                if p.category != vitaslop_gxp_shader::container::ParamCategory::Sampler {
                    continue;
                }
                match bound.iter().find(|t| t.unit as i32 == p.resource_index) {
                    // The first few DECODED texels, not just the binding. "Which texture" and
                    // "what is in it" are different questions, and for a texture the shader
                    // reads as DATA rather than as a picture - a vector canvas, a lookup
                    // table, an index map - the second one is the whole answer, and the
                    // binding line alone has already sent one investigation down the wrong
                    // road. Also says outright when every sampled texel is the same value.
                    Some(t) => report_knob!(
                        "gxp inputs {key:016x} {stage}: {} unit {} <- {:#x} {}x{} faces={} \
                         fmt={:#04x} swz={:#x} filter={} wrap=({},{}) seam={:?} texels[0..4]={:?} {}{}",
                        p.name,
                        p.resource_index,
                        t.tex.data_addr,
                        t.tex.width,
                        t.tex.height,
                        t.tex.faces,
                        t.tex.base_format,
                        t.tex.swizzle,
                        if t.tex.filter_linear { "linear" } else { "point" },
                        t.tex.addr_mode_u,
                        t.tex.addr_mode_v,
                        t.tex.texel,
                        // Read through the seam, so a half texture prints the values the shader
                        // actually gets rather than pairs of its bytes read as texels.
                        seam_texels(&t.tex).into_iter().take(4).collect::<Vec<_>>(),
                        channel_spread_seam(&t.tex),
                        match seam_texels(&t.tex).first() {
                            Some(first) if seam_texels(&t.tex).iter().all(|c| c == first) =>
                                " (EVERY texel identical - this texture carries no image)",
                            _ => "",
                        }
                    ),
                    None => report_knob!(
                        "gxp inputs {key:016x} {stage}: {} unit {} <- NOTHING BOUND",
                        p.name, p.resource_index
                    ),
                }
                // `VITASLOP_GXP_INPUTS_DIR=<dir>`: the decoded texture itself, as a PNG named by
                // the parameter. A channel spread says a channel is flat; only the image says
                // whether the picture in the other three is the one the shader expects, and for
                // a texture read as DATA (a displacement canvas, a lookup table) that is the
                // difference between "we decoded it wrong" and "the producer never ran".
                if let (Ok(dir), Some(t)) =
                    (std::env::var("VITASLOP_GXP_INPUTS_DIR"), bound.iter().find(|t| t.unit as i32 == p.resource_index))
                {
                    let _ = std::fs::create_dir_all(&dir);
                    let safe: String = p.name.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
                    // Raw RGBA8 with the dimensions in the NAME, not a PNG: this crate has no
                    // image encoder and must not grow one for a diagnostic. The size is in the
                    // filename so the bytes are self-describing.
                    let path = std::path::Path::new(&dir).join(format!(
                        "{key:016x}_{}_u{}_{safe}_{}x{}.rgba",
                        stage.replace(' ', "-"),
                        p.resource_index,
                        t.tex.width,
                        t.tex.height * t.tex.faces
                    ));
                    let _ = std::fs::write(path, t.tex.rgba.as_slice());
                }
            }
        }
        // Attribute RANGES, not the first vertex: a scale/bias question is a question about the
        // span of the coordinate across the mesh, and one vertex cannot answer it.
        let Ok(vprogram) = vitaslop_gxp_shader::Program::parse(&gxp.vprog) else { return };
        let stride = gxp.vertex_stride.max(1) as usize;
        let nverts = gxp.vertices.len() / stride;
        // THE VERTICES THE DRAW ACTUALLY REFERENCES, not every vertex in the uploaded stream.
        //
        // A stream commonly holds more than one draw's worth of geometry - a top ring and a
        // skirt ring, an LOD chain, a shared pool - and only the INDEX BUFFER says which of it
        // this draw touches. Ranging over the whole stream reports a component as reaching a
        // value no drawn triangle ever sees, which is a statement about the buffer wearing the
        // clothes of a statement about the picture. It also makes our own index expansion
        // unfalsifiable: if we index vertices the guest never asked for, the two populations
        // differ and nothing here could ever have said so.
        let indexed: Vec<usize> = {
            let mut seen = std::collections::BTreeSet::new();
            let b = gxp.indices.as_ref();
            let n = gxp.index_count as usize;
            if gxp.index_u32 {
                for i in 0..n.min(b.len() / 4) {
                    seen.insert(u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]) as usize);
                }
            } else {
                for i in 0..n.min(b.len() / 2) {
                    seen.insert(u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]) as usize);
                }
            }
            seen.into_iter().filter(|&v| v < nverts).collect()
        };
        // The TOPOLOGY, and the first triangle it forms. A mesh whose vertices are right can
        // still be assembled wrong: read a triangle LIST as a STRIP and every triangle after the
        // first straddles two of the source's, which shows up as smooth INTERPOLATION across
        // faces that should each be flat - a shading artefact that looks like a lighting bug and
        // is nothing of the kind. The guest's primitive word never appeared in any report, so
        // that possibility could not be checked at all.
        report_knob!(
            "gxp inputs {key:016x} topology: guest primitive {:#x}, {} indices -> {} triangles \
             over {nverts} vertices; first triangle = {:?}",
            gxp.primitive,
            gxp.index_count,
            gxp.index_count / 3,
            {
                let b = gxp.indices.as_ref();
                let rd = |i: usize| -> u32 {
                    if gxp.index_u32 {
                        b.get(i * 4..i * 4 + 4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).unwrap_or(0)
                    } else {
                        b.get(i * 2..i * 2 + 2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).unwrap_or(0)
                    }
                };
                [rd(0), rd(1), rd(2), rd(3), rd(4), rd(5)]
            }
        );
        if !indexed.is_empty() && indexed.len() != nverts {
            report_knob!(
                "gxp inputs {key:016x} vertices: the stream holds {nverts} but the draw's indices \
                 reference only {} of them ({}..={}) - the ranges below are over the INDEXED set",
                indexed.len(),
                indexed.first().copied().unwrap_or(0),
                indexed.last().copied().unwrap_or(0)
            );
        }
        // Fall back to the whole stream only when there is no index buffer to narrow it.
        let sample: Vec<usize> = if indexed.is_empty() { (0..nverts).collect() } else { indexed };
        let nverts = sample.len();
        for a in &gxp.attributes {
            let name = vprogram
                .parameters
                .iter()
                .find(|p| {
                    p.category == vitaslop_gxp_shader::container::ParamCategory::Attribute
                        && p.resource_index == a.reg_index as i32
                })
                .map(|p| p.name.as_str())
                .unwrap_or("<unnamed>");
            let comps = a.components.clamp(1, 4) as usize;
            let mut lo = [f32::INFINITY; 4];
            let mut hi = [f32::NEG_INFINITY; 4];
            let mut vals: Vec<Vec<f32>> = vec![Vec::new(); comps];
            for &v in &sample {
                for c in 0..comps {
                    let f = read_attr_component(&gxp.vertices, v * stride + a.offset as usize, a.gxm_format, c);
                    lo[c] = lo[c].min(f);
                    hi[c] = hi[c].max(f);
                    vals[c].push(f);
                }
            }
            // A RANGE says a component reaches -1 somewhere; it does not say whether that is one
            // vertex or half the mesh, and those two readings send an investigation in opposite
            // directions. Eight equal buckets across the span answer it in a bounded line - the
            // reason this is not the per-vertex dump, which is capped out on any real mesh.
            let shape = |c: usize| -> String {
                if !(lo[c].is_finite() && hi[c].is_finite()) || hi[c] <= lo[c] {
                    return "constant".into();
                }
                let mut bins = [0usize; 8];
                let span = hi[c] - lo[c];
                for &f in &vals[c] {
                    let b = (((f - lo[c]) / span) * 8.0) as usize;
                    bins[b.min(7)] += 1;
                }
                let pct: Vec<String> = bins
                    .iter()
                    .map(|&n| format!("{:.0}", 100.0 * n as f32 / nverts.max(1) as f32))
                    .collect();
                pct.join("/")
            };
            let ranges: Vec<String> = (0..comps)
                .map(|c| format!("[{:.4}, {:.4}] {}%", lo[c], hi[c], shape(c)))
                .collect();
            report_knob!(
                "gxp inputs {key:016x} attribute: {name} lane {} at byte {} of a {}-byte vertex, \
                 fmt {} x{} over {nverts} vertices = {}",
                a.reg_index,
                a.offset,
                gxp.vertex_stride,
                a.gxm_format,
                comps,
                ranges.join(" ")
            );
        }
        // Per-VERTEX values, behind their OWN knob (`VITASLOP_GXP_INPUTS_VERTS=<key>|all`).
        // A component RANGE only pins down an attribute if the mesh maps it affinely onto the
        // screen, and a post-process DISTORTION GRID is exactly the mesh that does not - so
        // sometimes the vertices themselves are the answer.
        //
        // # Why naming the pair is NOT enough to ask for this
        // It used to be: naming a pair explicitly (rather than `all`) turned the dump on, capped
        // at `MAX_DUMPED_VERTICES` = 512 so it could not "bury the frame's other reports in a
        // megabyte of numbers". That cap is sized for a LOG FILE. The browser's diagnostics panel
        // keeps a bounded number of DISTINCT lines - 96 - and this title's final composite is a
        // 288-vertex grid, so one input set of per-vertex lines evicts every other finding in the
        // panel. MEASURED on the user's phone: `10378 earlier DISTINCT line(s) dropped`, and the
        // four uniform lines the run was taken FOR were among them. The knob answered, and its
        // answer was destroyed by the same knob's other output.
        //
        // So the dump is opt-in on its own name. Naming a pair in `VITASLOP_GXP_INPUTS` now gets
        // the uniforms, the samplers, the viewport and the attribute RANGES - a bounded number of
        // lines that fits any sink. [[vitaslop-a-diagnostic-can-bury-the-findings]]
        let verts_wanted = crate::knobs::var("VITASLOP_GXP_INPUTS_VERTS")
            .map(|s| {
                s.split(',').any(|k| {
                    let k = k.trim();
                    k == "all" || u64::from_str_radix(k.trim_start_matches("0x"), 16) == Ok(key)
                })
            })
            .unwrap_or(false);
        if !verts_wanted {
            return;
        }
        // A MESH TOO BIG TO DUMP STILL GETS ITS FIRST FEW VERTICES. The cap exists so one draw
        // cannot bury a frame's other findings, and that is right - but returning EMPTY makes the
        // instrument silent exactly on the meshes worth asking about, and "the knob printed
        // nothing" then reads as "there is nothing there". The question a raw record answers -
        // is this field really at this byte, is this value really the float it decodes to - is
        // answered by eight vertices as well as by seventeen thousand.
        // [[vitaslop-instrument-failure-imitating-its-subject]]
        const HEAD: usize = 8;
        let shown = if nverts > MAX_DUMPED_VERTICES { HEAD.min(nverts) } else { nverts };
        if shown < nverts {
            report_knob!(
                "gxp inputs {key:016x} vertices: {nverts} is over the {MAX_DUMPED_VERTICES} dump \
                 cap - printing the FIRST {shown} only; the attribute RANGES above cover all of them"
            );
        }
        for v in 0..shown {
            let cols: Vec<String> = gxp
                .attributes
                .iter()
                .map(|a| {
                    let comps = a.components.clamp(1, 4) as usize;
                    let vals: Vec<String> = (0..comps)
                        .map(|c| {
                            format!(
                                "{:.4}",
                                read_attr_component(&gxp.vertices, v * stride + a.offset as usize, a.gxm_format, c)
                            )
                        })
                        .collect();
                    format!("lane{}=({})", a.reg_index, vals.join(","))
                })
                .collect();
            // The RAW vertex bytes too. A guest vertex often carries fields no declared attribute
            // names, and an attribute read at the wrong byte offset produces plausible numbers -
            // so the only way to tell "this really is the mesh's UV" from "this is some other
            // field that happens to look like one" is to see the whole record.
            let raw: String = gxp
                .vertices
                .get(v * stride..(v + 1) * stride)
                .map(|b| b.chunks(4).map(|c| {
                    let mut w = [0u8; 4];
                    w[..c.len()].copy_from_slice(c);
                    format!("{}", f32::from_le_bytes(w))
                }).collect::<Vec<_>>().join(" "))
                .unwrap_or_default();
            report_knob!("gxp inputs {key:016x} vertex {v}: {}   raw-as-f32 [{raw}]", cols.join(" "));
        }
    }

    /// Most vertices `VITASLOP_GXP_INPUTS` will print individually. A post-process grid or a UI
    /// quad is small enough to read; a world mesh is not, and dumping one would bury the frame's
    /// other reports in a megabyte of numbers.
    const MAX_DUMPED_VERTICES: usize = 512;

    /// Diagnostic (`VITASLOP_GXP_INPUTS_ORDER=<hex-key>[,<hex-key>]`): print ONE compact line per
    /// SUBMISSION of that pair, in submission order, never deduped.
    ///
    /// # Why this exists next to [`report_inputs`], which already prints every input
    /// [`report_inputs`] deduplicates on `(pair, inputs-hash)` for the whole run, so what it
    /// prints is the SET of distinct input sets in FIRST-SEEN order. That is the right shape for
    /// "what is this draw fed" and the wrong shape for "in what order is it fed", and the two are
    /// indistinguishable when read off the same log: a first-seen list looks monotonic whatever
    /// the real order is, because a value's second appearance is exactly what the dedupe drops.
    ///
    /// This was not a hypothetical. A fade animated through a per-vertex COLOUR was read off that
    /// deduped list as "two clean monotonic ramps", which excluded the guest as the source of a
    /// flicker whose whole content is which alpha lands in which frame. The ordering question
    /// needs an instrument that repeats itself; that is this one.
    ///
    /// Deliberately one line per submission with RANGES only - a per-vertex dump repeated at draw
    /// rate is not readable, and a ramp is a question about the span, not about vertex 7.
    ///
    /// # Why the SMALL declared uniforms are printed by VALUE and not only as a hash
    /// This line used to carry a single `sa <hash>` for the whole uniform block, which makes it
    /// able to say WHEN a uniform changed and never WHAT it changed to. That is half an
    /// instrument, and the half it is missing is the half the ordering question needs: the
    /// project's own notes told the next session to "use `_INPUTS_ORDER` if the order must be
    /// exact", and following that instruction produces a column of opaque hashes. The values
    /// then have to be recovered from [`report_inputs`], whose output is DEDUPED and
    /// first-seen-ordered - which is the exact instrument the ordering question was trying to
    /// get away from. A fade was read off that pairing as running upward when it runs downward.
    ///
    /// Only uniforms of at most [`MAX_ORDERED_UNIFORM_COMPONENTS`] components are printed.
    /// That is not a cosmetic budget: the uniforms a fade or a transition is driven through are
    /// scalars and short vectors (`screenTintColour`, `bloomFactor`, `posOffset`), while the
    /// ones that would drown the line are matrices and sample-offset tables (a `worldViewProj`
    /// is 16 components and one map pair declares 94 registers). Printing everything at draw
    /// rate would bury the finding in exactly the way
    /// [[vitaslop-a-diagnostic-can-bury-the-findings]] describes. The hash still covers the
    /// WHOLE block, so a change in a table that is not printed still shows up as a hash change
    /// and cannot be mistaken for "nothing moved".
    fn report_inputs_order(key: u64, gxp: &GxpRecompile) {
        // Through the knob table - see `report_inputs` for why - and resolved ONCE, see
        // `KeySpec`. This is the hottest of these: it runs per SUBMISSION, not per pair.
        use std::sync::OnceLock;
        static SPEC: OnceLock<KeySpec> = OnceLock::new();
        if !SPEC.get_or_init(|| KeySpec::resolve("VITASLOP_GXP_INPUTS_ORDER")).wants(key) {
            return;
        }
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let stride = gxp.vertex_stride.max(1) as usize;
        let nverts = gxp.vertices.len() / stride;
        let mut cols: Vec<String> = Vec::new();
        for a in &gxp.attributes {
            let comps = a.components.clamp(1, 4) as usize;
            let mut lo = [f32::INFINITY; 4];
            let mut hi = [f32::NEG_INFINITY; 4];
            for v in 0..nverts {
                for c in 0..comps {
                    let f = read_attr_component(
                        &gxp.vertices,
                        v * stride + a.offset as usize,
                        a.gxm_format,
                        c,
                    );
                    lo[c] = lo[c].min(f);
                    hi[c] = hi[c].max(f);
                }
            }
            let r: Vec<String> = (0..comps)
                .map(|c| {
                    if lo[c] == hi[c] {
                        format!("{:.4}", lo[c])
                    } else {
                        format!("{:.4}..{:.4}", lo[c], hi[c])
                    }
                })
                .collect();
            cols.push(format!("lane{}=({})", a.reg_index, r.join(",")));
        }
        // The uniform bytes fold in too: a fade driven through a uniform and one driven through a
        // vertex colour produce the same picture, and this line has to be able to tell them apart
        // without a second run.
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        gxp.vert_sa.hash(&mut h);
        gxp.frag_sa.hash(&mut h);
        // The SHORT declared uniforms by value, so this line can answer "what did it change to"
        // and not only "it changed" - see the doc comment above for why the long ones are left
        // to the hash.
        let mut named: Vec<String> = Vec::new();
        for (stage, bytes, blob) in
            [("v", &gxp.vert_sa, &gxp.vprog), ("f", &gxp.frag_sa, &gxp.fprog)]
        {
            let Ok(program) = vitaslop_gxp_shader::Program::parse(blob) else { continue };
            for p in &program.parameters {
                use vitaslop_gxp_shader::container::ParamCategory;
                if p.category != ParamCategory::Uniform
                    || p.component_count as usize > MAX_ORDERED_UNIFORM_COMPONENTS
                {
                    continue;
                }
                named.push(format!("{stage}:{}={}", p.name, decode_uniform(bytes, p)));
            }
        }
        report_knob!(
            "gxp order {key:016x} #{seq}: {nverts} verts, sa {:016x}, {}{}",
            h.finish(),
            cols.join(" "),
            if named.is_empty() { String::new() } else { format!(" | {}", named.join(" ")) }
        );
    }

    /// The widest declared uniform [`report_inputs_order`] prints by VALUE on its per-submission
    /// line. See that function's doc comment: short vectors are what animations are driven
    /// through, matrices and lookup tables are what would bury the line.
    const MAX_ORDERED_UNIFORM_COMPONENTS: usize = 8;

    /// Diagnostic (`VITASLOP_GXP_SA=<key>:<v|f>:<reg>=<hexword>[,...]`): replace a default-uniform
    /// register with a chosen 32-bit word before the draw is submitted.
    ///
    /// This is the causality half of [`report_inputs`]. Reading a uniform tells you what a shader
    /// was fed; it does not tell you whether that value is what makes the picture wrong, because
    /// every downstream stage is a candidate too. Substituting the value and re-rendering is the
    /// one experiment that separates them, and it takes one run instead of a session of reasoning.
    /// A substitution is REPORTED, once per (pair, stage, register): a run whose frame came from
    /// values the guest never wrote must never be mistaken for a run of the real thing.
    fn override_sa<'a>(key: u64, stage: char, bytes: &'a [u8]) -> std::borrow::Cow<'a, [u8]> {
        // Through the knob table - see `report_inputs`. This one especially: it is the
        // causality experiment, and the defect it is aimed at reproduces only in the browser.
        // >>> THE OFF PATH IS FREE, and it was not: this runs TWICE PER DRAW (vertex and
        // fragment), and reading the knob here cost a mutex plus an `std::env::var` allocation
        // each time - ~2,500 per frame on the user's device - before re-parsing and
        // re-VALIDATING a spec that is almost always absent. Only the presence check is cached;
        // when the knob IS set the full parse still runs per draw, keeping every fail-loud
        // panic below exactly as it was. A diagnostic nobody enabled must not be on the bill.
        use std::sync::OnceLock;
        static SET: OnceLock<bool> = OnceLock::new();
        if !*SET.get_or_init(|| crate::knobs::var("VITASLOP_GXP_SA").is_ok()) {
            return std::borrow::Cow::Borrowed(bytes);
        }
        let Ok(spec) = crate::knobs::var("VITASLOP_GXP_SA") else {
            return std::borrow::Cow::Borrowed(bytes);
        };
        let mut out = std::borrow::Cow::Borrowed(bytes);
        for item in spec.split(',').filter(|s| !s.trim().is_empty()) {
            let parts: Vec<&str> = item.trim().split(':').collect();
            let [k, st, assign] = parts[..] else {
                panic!("VITASLOP_GXP_SA item {item:?} is not <key>:<v|f>:<reg>=<hexword>");
            };
            let Ok(want_key) = u64::from_str_radix(k.trim_start_matches("0x"), 16) else {
                panic!("VITASLOP_GXP_SA item {item:?} has a non-hex pair key");
            };
            let want_stage = match st {
                "v" => 'v',
                "f" => 'f',
                other => panic!("VITASLOP_GXP_SA item {item:?} names stage {other:?}, not v or f"),
            };
            let Some((reg, word)) = assign.split_once('=') else {
                panic!("VITASLOP_GXP_SA item {item:?} is missing the =<hexword>");
            };
            let (Ok(reg), Ok(word)) = (
                reg.trim().parse::<usize>(),
                u32::from_str_radix(word.trim().trim_start_matches("0x"), 16),
            ) else {
                panic!("VITASLOP_GXP_SA item {item:?} has a bad register or word");
            };
            if want_key != key || want_stage != stage {
                continue;
            }
            let buf = out.to_mut();
            if buf.len() < (reg + 1) * 4 {
                buf.resize((reg + 1) * 4, 0);
            }
            buf[reg * 4..reg * 4 + 4].copy_from_slice(&word.to_le_bytes());
            report_sa_override(key, stage, reg, word);
        }
        out
    }

    /// Report - once per (pair, stage, register) - that a uniform register was substituted by
    /// `VITASLOP_GXP_SA`. See [`override_sa`] for why this is never silent.
    fn report_sa_override(key: u64, stage: char, reg: usize, word: u32) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(u64, char, usize)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert((key, stage, reg)) {
            // At WARN, not `report!`. This one carries a correctness guarantee - "a run whose
            // frame came from values the guest never wrote must never be mistaken for a run of
            // the real thing" - and `report!` is BELOW the browser's default filter, so on the
            // engine where this knob was just made reachable the guarantee was silently void.
            // The first substituted run there produced a frame with no indication it was one.
            report_warn!(
                "gxp pair {key:016x}: {} uniform register {reg} SUBSTITUTED with {word:#010x} - \
                 this frame is NOT what the guest asked for",
                if stage == 'v' { "vertex" } else { "fragment" }
            );
        }
    }

    /// One uniform parameter's values, read out of the raw default-uniform-buffer bytes through
    /// its OWN declared type. `resource_index` is a 4-byte register offset; the components are
    /// packed from there at the type's own component width, which is how an F16 float4 fits in
    /// two registers and an F32 float4 needs four.
    fn decode_uniform(bytes: &[u8], p: &vitaslop_gxp_shader::container::Parameter) -> String {
        use vitaslop_gxp_shader::container::ParamType;
        let Some(width) = p.ptype.component_bytes() else {
            return format!("<{:?} has no fixed component width>", p.ptype);
        };
        let base = (p.resource_index.max(0) as usize) * 4;
        let n = p.component_count as usize * p.array_size.max(1) as usize;
        let mut out: Vec<String> = Vec::with_capacity(n);
        for i in 0..n {
            let o = base + i * width as usize;
            let Some(raw) = bytes.get(o..o + width as usize) else {
                out.push("<past the end of the buffer>".to_string());
                break;
            };
            out.push(match p.ptype {
                ParamType::F32 => format!("{}", f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
                ParamType::F16 => format!("{}", half_to_f32(u16::from_le_bytes([raw[0], raw[1]]))),
                ParamType::U32 => format!("{}", u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
                ParamType::S32 => format!("{}", i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])),
                ParamType::U16 | ParamType::C10 => format!("{}", u16::from_le_bytes([raw[0], raw[1]])),
                ParamType::S16 => format!("{}", i16::from_le_bytes([raw[0], raw[1]])),
                ParamType::U8 => format!("{}", raw[0]),
                ParamType::S8 => format!("{}", raw[0] as i8),
                ParamType::Aggregate | ParamType::Unknown(_) => unreachable!("no component width"),
            });
        }
        format!("({})", out.join(", "))
    }

    /// Every texel of a decoded texture as four floats, read through its own seam.
    ///
    /// The byte seam's values are `n/255`, so the two seams print on one scale and a reader does
    /// not have to know which one a given texture landed on to compare two of them.
    fn seam_texels(t: &GxmTexture) -> Vec<[f32; 4]> {
        let bpt = t.texel.bytes_per_texel();
        t.rgba
            .chunks_exact(bpt)
            .map(|c| match t.texel {
                TexelSeam::Rgba8 => [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                    c[3] as f32 / 255.0,
                ],
                TexelSeam::Rgba16Float => {
                    let h = |i: usize| half_to_f32(u16::from_le_bytes([c[i * 2], c[i * 2 + 1]]));
                    [h(0), h(1), h(2), h(3)]
                }
            })
            .collect()
    }

    /// Per-channel min/max over every texel, read through the texture's own seam.
    ///
    /// "The first four texels" answers what a texture looks like at its corner; it does not
    /// answer whether a channel carries any signal at all, and for a texture a shader reads as
    /// DATA that is the only question. The map body's vertex program displaces every vertex by
    /// this texture's ALPHA, so `a[0,0]` and `a[0,1]` are two completely different bugs and the
    /// corner texel cannot tell them apart.
    fn channel_spread_seam(t: &GxmTexture) -> String {
        let texels = seam_texels(t);
        if texels.is_empty() {
            return "spread=(no texels)".into();
        }
        let mut lo = [f32::INFINITY; 4];
        let mut hi = [f32::NEG_INFINITY; 4];
        for c in &texels {
            for i in 0..4 {
                lo[i] = lo[i].min(c[i]);
                hi[i] = hi[i].max(c[i]);
            }
        }
        format!(
            "spread over {} texels r[{:.4},{:.4}] g[{:.4},{:.4}] b[{:.4},{:.4}] a[{:.4},{:.4}]",
            texels.len(),
            lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], lo[3], hi[3]
        )
    }

    /// Byte size of one component of a `SceGxmAttributeFormat`.
    fn attr_component_size(gxm_format: u8) -> usize {
        match gxm_format {
            0 | 1 | 4 | 5 => 1,     // U8 / S8 / U8N / S8N
            2 | 3 | 6 | 7 | 8 => 2, // U16 / S16 / U16N / S16N / F16
            _ => 4,                 // F32 (9) and any unknown -> treat as 4-byte float
        }
    }

    /// Read one attribute component from the guest stream and convert it to f32, matching the
    /// GXM fixed-function vertex fetch (normalized formats scale to [0,1]/[-1,1], F16 expands).
    /// Out-of-range reads yield 0.0 (a benign over-read on a short buffer, never a panic).
    fn read_attr_component(buf: &[u8], base: usize, gxm_format: u8, c: usize) -> f32 {
        let o = base + c * attr_component_size(gxm_format);
        let u16at = |o: usize| -> Option<u16> { buf.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]])) };
        match gxm_format {
            9 => buf.get(o..o + 4).map(|s| f32::from_le_bytes([s[0], s[1], s[2], s[3]])).unwrap_or(0.0),
            8 => u16at(o).map(half_to_f32).unwrap_or(0.0),
            4 => buf.get(o).map(|&b| b as f32 / 255.0).unwrap_or(0.0),
            5 => buf.get(o).map(|&b| (b as i8 as f32 / 127.0).max(-1.0)).unwrap_or(0.0),
            6 => u16at(o).map(|v| v as f32 / 65535.0).unwrap_or(0.0),
            7 => u16at(o).map(|v| (v as i16 as f32 / 32767.0).max(-1.0)).unwrap_or(0.0),
            0 => buf.get(o).map(|&b| b as f32).unwrap_or(0.0),
            1 => buf.get(o).map(|&b| b as i8 as f32).unwrap_or(0.0),
            2 => u16at(o).map(|v| v as f32).unwrap_or(0.0),
            3 => u16at(o).map(|v| v as i16 as f32).unwrap_or(0.0),
            _ => 0.0,
        }
    }

    /// A content hash over a byte slice, consuming EIGHT bytes per multiply instead of one.
    ///
    /// The word-at-a-time shape matters because this runs over megabytes a frame: every draw
    /// hashes its whole vertex stream to find its repacked geometry, and a byte-at-a-time fold
    /// over that is milliseconds of a twenty-millisecond frame. The trailing bytes are folded in
    /// individually so the result still depends on the exact length.
    ///
    /// # Why this is not plain word-wise FNV-1a any more
    /// It was, and the word-wise form of FNV has a failure mode the byte-wise form does not:
    /// **a difference confined to bit 63 of a word survives the multiply unchanged, so two such
    /// differences cancel.** Multiplication by an odd constant is linear mod 2^64, and 2^63 is a
    /// fixed point of it (`2^63 * odd = 2^63`), so a top-bit flip in word `k` leaves the
    /// accumulator differing in exactly bit 63 - which the very next word's top-bit flip then
    /// XORs back to zero. **Any EVEN number of top-bit flips at 8-byte-aligned positions hashes
    /// identically.** No other bit position does this: a flip at bit `b < 63` becomes `2^b *
    /// PRIME`, which is spread across the word and cannot be cancelled by a single clean bit.
    ///
    /// That is not a theoretical weakness, it is a defect that shipped and was visible. This
    /// hash keys the PACKED-VERTEX CACHE, whose entries are repacked vertex buffers. A UI text
    /// quad faded through a per-vertex unorm8 COLOUR, 150 vertices of a 24-byte stride with the
    /// colour at byte 20, puts every vertex's ALPHA byte at absolute offset `24v + 23` - that is
    /// `7 mod 8`, the top byte of a word, so the alpha's high bit IS bit 63. 150 is even.
    /// So two frames of the fade whose alpha bytes differed only in bit 7 - 127 against 255, 101
    /// against 229, 50 against 178, 25 against 153, 76 against 204 - collided, and the second
    /// frame drew the FIRST frame's repacked vertices. MEASURED on the title screen: of the 15
    /// alpha levels the guest submits over its 60-frame cycle, the 5 whose bit-7 partner also
    /// appears in the cycle rendered as that partner instead, every time. The fade looked like it
    /// was stepping backwards, and the guest was emitting a perfectly monotonic ramp throughout.
    ///
    /// The rotate breaks the fixed point: a difference at bit 63 lands at bit 30 before the next
    /// multiply, where it is spread rather than preserved. The final avalanche is murmur3's
    /// `fmix64`, so the length also reaches every output bit instead of only the low ones.
    fn fnv64(seed: u64, bytes: &[u8]) -> u64 {
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = seed;
        let (words, tail) = bytes.split_at(bytes.len() & !7);
        for w in words.chunks_exact(8) {
            h ^= u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]);
            h = h.wrapping_mul(PRIME).rotate_left(31);
        }
        for &b in tail {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
        h ^= bytes.len() as u64;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^ (h >> 33)
    }

    #[cfg(test)]
    /// `std::time::Instant` must not be constructed anywhere in this file except inside
    /// [`Stopwatch`], because it PANICS on `wasm32-unknown-unknown` and this file is compiled
    /// for the browser.
    ///
    /// # Why a source grep rather than a type
    /// The panic is a RUNTIME one: `Instant::now()` compiles for wasm32 perfectly happily and
    /// dies the moment it is called, so neither the wasm build nor the desktop tests can catch
    /// it. It shipped exactly that way - a pipeline-build timer added to `build_gxp_pipeline`
    /// killed the browser's run worker on the first frame it rendered anything, with a stack
    /// reading `time not implemented on this platform`, while `cargo build --target
    /// wasm32-unknown-unknown` was green. The doc comment on `wasm_now` had warned about this
    /// for a whole session before that happened, which is the argument for a test over a
    /// comment.
    #[cfg(test)]
    mod wasm_clock_tests {
        /// The only lines allowed to name it: the `Stopwatch` field, its constructor, and prose.
        #[test]
        fn nothing_constructs_a_std_instant_outside_the_stopwatch() {
            // Spelled in halves so this test's OWN source does not match the needle - the
            // first version of it failed on itself, which is funny once and then is just a
            // broken test.
            let needle = concat!("std::time::", "Instant", "::now()");
            let src = include_str!("gpu.rs");
            let offenders: Vec<(usize, &str)> = src
                .lines()
                .enumerate()
                .filter(|(_, l)| l.contains(needle))
                .filter(|(_, l)| !l.contains("Stopwatch { start:"))
                // Prose, including the doc comment on this very test.
                .filter(|(_, l)| !l.trim_start().starts_with("///"))
                .filter(|(_, l)| !l.trim_start().starts_with("//"))
                .map(|(n, l)| (n + 1, l.trim()))
                .collect();
            assert!(
                offenders.is_empty(),
                "{needle} panics on wasm32 and this file runs in the browser - use                  `Stopwatch::start()`. Offending lines: {offenders:?}"
            );
        }
    }

    mod texture_budget_default_tests {
        /// The default budget is the CONSOLE's ceiling on resident game memory, and it is
        /// spelled as its three partitions so that it cannot quietly become a round number
        /// somebody liked. The previous value was fitted to one title's menu screen and cost a
        /// race 83% re-decodes on the target device; a figure with no derivation attached is how
        /// that happens again.
        #[test]
        fn the_default_is_the_consoles_game_partitions_and_nothing_else() {
            assert_eq!(
                super::GAME_RESIDENT_CEILING_MB,
                256 + 109 + 112,
                "ScePhyMemPartGame + the +109 MiB extension + ScePhyMemPartGameCdram"
            );
            assert_eq!(super::GAME_RESIDENT_CEILING_MB, 477);
        }

        /// The PRESSURE signal and the UPLOADER must price the budget identically.
        ///
        /// They did not. `texture_budget_pressure` carried its own copy of this arithmetic with
        /// `unwrap_or(256)` under a comment saying "the same budget the uploader enforces, read
        /// the same way, so the two cannot disagree about what tight means" - while the uploader
        /// defaulted to 477. Pressure therefore declared itself at 171 MiB where the uploader's
        /// own two-thirds threshold is 318, and on a no-BC adapter every screen re-encoded its BC
        /// textures to ETC2 - a block decode plus an alpha-carrying encode, the most expensive
        /// path there is - to save memory that was not short.
        ///
        /// The two now call one function. This test is what stops a second copy appearing: a
        /// duplicated default is invisible precisely because SETTING the knob moves both.
        #[test]
        fn the_pressure_signal_prices_the_budget_the_same_way_the_uploader_does() {
            // 0 is the "no frame has finished" sentinel, which is pressure by design, so the
            // comparison is made at a working set that is unambiguously below any threshold.
            super::super::LAST_WORKING_SET.store(1, std::sync::atomic::Ordering::Relaxed);
            let budget = super::tex_cache_budget_bytes();
            assert!(
                !super::super::texture_budget_pressure(),
                "1 byte resident is not pressure against a {} MiB budget",
                budget / (1024 * 1024)
            );
            // ...and at two-thirds of that same budget it MUST be, or the two readers are using
            // different numbers again.
            super::super::LAST_WORKING_SET.store(budget / 3 * 2 + 1, std::sync::atomic::Ordering::Relaxed);
            assert!(super::super::texture_budget_pressure());
            super::super::LAST_WORKING_SET.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    mod texture_budget_accounting_tests {
        use super::{texture_upload_bytes, BlockFormat, CompressedUpload, TexelSeam};

        /// [`texture_upload_bytes`] on the decode path - the device gate is what the other
        /// argument is, and it has its own test below.
        fn uploaded_texture_bytes(t: &super::GxmTexture) -> usize {
            texture_upload_bytes(t, false)
        }

        fn tex(width: u32, height: u32, faces: u32, texel: TexelSeam) -> super::GxmTexture {
            super::GxmTexture {
                key: 0,
                data_addr: 0,
                width,
                height,
                faces,
                rgba: crate::gpu::Texels::ready(Vec::new()),
                texel,
                base_format: 0,
                swizzle: 0,
                filter_linear: false,
                addr_mode_u: 0,
                addr_mode_v: 0,
                gamma: false,
                compressed: None,
            }
        }

        /// A texture that passes through compressed is priced at the bytes actually handed to
        /// the driver, NOT at the 4/3-of-level-0 estimate the decode path uses.
        ///
        /// The estimate is what makes this worth asserting: it is derived from `width * height *
        /// bytes_per_texel`, so applying it to a passthrough would price BC3 blocks as RGBA8 and
        /// report a working set four times what is resident. A budget that mis-prices the path
        /// it is meant to be measuring is the same defect the byte accounting was rewritten to
        /// remove, just pointed the other way.
        #[test]
        fn a_compressed_passthrough_is_priced_at_the_bytes_handed_over() {
            let mut t = tex(256, 256, 1, TexelSeam::Rgba8);
            // 256x256 BC3: 64x64 blocks of 16 bytes, plus the chain.
            let blocks = vec![0u8; 64 * 64 * 16 * 4 / 3];
            let n = blocks.len();
            t.compressed = Some(CompressedUpload {
                format: BlockFormat::Bc3,
                width: 256,
                height: 256,
                data: crate::gpu::CompressedData::Cpu(std::sync::Arc::new(blocks)),
                levels: 9,
                transcoded: false,
            });
            assert_eq!(texture_upload_bytes(&t, true), n);
            // And with the device gate shut it is the decode that is priced, not the blocks -
            // the two paths cost wildly different amounts and the flag is the only thing that
            // says which one this texture is on.
            assert_eq!(texture_upload_bytes(&t, false), 256 * 256 * 4 * 4 / 3);
        }

        /// The byte seam is uploaded WITH a mip chain, so it costs about a third more than its
        /// level 0. The flat `w * h * 4` this replaced returned exactly level 0 - which is what
        /// makes this test fail against the old accounting rather than merely differ from it.
        #[test]
        fn the_byte_seam_is_priced_with_its_mip_chain() {
            let level0 = 256 * 256 * 4;
            let got = uploaded_texture_bytes(&tex(256, 256, 1, TexelSeam::Rgba8));
            assert!(
                got > level0,
                "priced {got} for a mipped texture whose level 0 alone is {level0}"
            );
            assert_eq!(got, level0 * 4 / 3);
        }

        /// The half seam is 8 bytes per texel and is uploaded level-0-only. Pricing it as RGBA8
        /// under-charged it by exactly half, on the one seam whose texels are DATA.
        #[test]
        fn the_half_seam_is_priced_at_eight_bytes_per_texel_and_has_no_mips() {
            let as_rgba8 = 512 * 128 * 4;
            let got = uploaded_texture_bytes(&tex(512, 128, 1, TexelSeam::Rgba16Float));
            assert_eq!(got, 512 * 128 * 8, "the half seam is 8 bytes per texel");
            assert_eq!(got, as_rgba8 * 2, "the old accounting was exactly half this");
        }

        /// A cube map uploads six layers and only one was ever priced. Asserted against the
        /// exact expected total rather than `one * 6`: the mip factor is integer arithmetic, so
        /// six faces priced together and one face priced six times differ by a couple of bytes.
        #[test]
        fn every_face_of_a_cube_map_is_priced() {
            let one = uploaded_texture_bytes(&tex(64, 64, 1, TexelSeam::Rgba8));
            let six = uploaded_texture_bytes(&tex(64, 64, 6, TexelSeam::Rgba8));
            assert_eq!(six, 64 * 64 * 6 * 4 * 4 / 3);
            assert!(
                six >= one * 6 - 8 && six <= one * 6 + 8,
                "six faces priced {six}, one face priced {one}"
            );
        }
    }

    #[cfg(test)]
    mod depth_bind_cache_bound_tests {
        use super::{drain_if_at_cap, DEPTH_BG_CACHE_CAP};
        use crate::fasthash::FxHashMap as HashMap;

        /// The defect, reproduced as the access PATTERN that caused it: a race moves the depth
        /// range every frame, so every insertion carries a key never seen before and never
        /// asked for again. Without a bound the map grows once per draw for the whole run -
        /// which is how the GPU process reached 5.8 GB while every picture looked correct.
        ///
        /// Asserting on the CAP rather than on a fixed number is deliberate: the bug was the
        /// absence of any ceiling, so the test has to fail for an unbounded map at whatever
        /// the ceiling happens to be.
        #[test]
        fn a_never_repeating_key_cannot_grow_the_cache_without_bound() {
            let mut map: HashMap<u64, u64> = HashMap::default();
            let mut evicted = Vec::new();
            // Ten times the cap of distinct keys, as a long race would produce.
            for k in 0..(DEPTH_BG_CACHE_CAP as u64 * 10) {
                if !map.contains_key(&k) {
                    drain_if_at_cap(&mut map, DEPTH_BG_CACHE_CAP, &mut evicted);
                }
                map.insert(k, k);
                assert!(
                    map.len() <= DEPTH_BG_CACHE_CAP,
                    "cache reached {} entries against a cap of {DEPTH_BG_CACHE_CAP}",
                    map.len()
                );
            }
            // EVERY evicted entry is handed back, not dropped. Each one owns a GPU buffer in
            // the real cache, and one silently dropped is one that can never be destroyed -
            // the whole reason this returns the values instead of clearing.
            assert_eq!(
                evicted.len(),
                DEPTH_BG_CACHE_CAP * 9,
                "evicted entries must all be handed back for destruction"
            );
        }

        /// A cache under its cap must not be disturbed - the bound is a ceiling, not a policy
        /// that throws away a working set. A menu holds one depth range all screen.
        #[test]
        fn a_cache_under_its_cap_is_left_alone() {
            let mut map: HashMap<u64, u64> = HashMap::default();
            for k in 0..(DEPTH_BG_CACHE_CAP as u64 - 1) {
                map.insert(k, k);
            }
            let mut evicted = Vec::new();
            assert!(
                !drain_if_at_cap(&mut map, DEPTH_BG_CACHE_CAP, &mut evicted),
                "cleared while under the cap"
            );
            assert_eq!(map.len(), DEPTH_BG_CACHE_CAP - 1);
            assert!(evicted.is_empty(), "evicted something while under the cap");
            // And it reports the clear when it does happen, so a capture can show it.
            map.insert(u64::MAX, 0);
            assert!(
                drain_if_at_cap(&mut map, DEPTH_BG_CACHE_CAP, &mut evicted),
                "did not clear at the cap"
            );
            assert!(map.is_empty());
            assert_eq!(evicted.len(), DEPTH_BG_CACHE_CAP);
        }
    }

    #[cfg(test)]
    mod content_hash_tests {
        /// The exact shape that shipped: a 150-vertex, 24-byte-stride UI stream whose only
        /// difference is bit 7 of the unorm8 ALPHA at byte 23 of each vertex.
        ///
        /// Byte `24v + 23` is `7 mod 8`, so every one of those bits is bit 63 of a word, and
        /// the word-wise FNV this replaced cancelled them in pairs. 150 is even, so it hashed
        /// the two streams identically and the packed-vertex cache served the wrong mesh.
        #[test]
        fn a_paired_top_bit_flip_does_not_cancel() {
            let mut a = vec![0x5au8; 24 * 150];
            let mut b = a.clone();
            for v in 0..150 {
                a[v * 24 + 23] = 127;
                b[v * 24 + 23] = 255;
            }
            assert_ne!(
                super::fnv64(0xcbf2_9ce4_8422_2325, &a),
                super::fnv64(0xcbf2_9ce4_8422_2325, &b),
                "two vertex streams differing in the top bit of every 8-aligned word hash the \
                 same, which is the collision that made a faded text quad render at another \
                 frame's alpha"
            );
        }

        /// The general form, so the fix is not read as being about one stride: an EVEN number of
        /// top-bit flips at any 8-byte-aligned positions must change the hash.
        #[test]
        fn top_bit_flips_at_word_boundaries_change_the_hash() {
            let base = vec![0x11u8; 8 * 32];
            for pairs in 1..=8usize {
                let mut v = base.clone();
                for i in 0..(pairs * 2) {
                    v[i * 8 + 7] ^= 0x80;
                }
                assert_ne!(
                    super::fnv64(0, &base),
                    super::fnv64(0, &v),
                    "{} paired top-bit flips cancelled",
                    pairs
                );
            }
        }

        /// A hash that ignored length, or folded it in only at the end without diffusing it,
        /// would let a truncated stream collide with its own prefix.
        #[test]
        fn length_reaches_the_whole_hash() {
            let v = vec![0u8; 64];
            let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
            for n in 0..=64 {
                assert!(seen.insert(super::fnv64(0, &v[..n])), "zero-fill of length {n} collided");
            }
        }
    }

    /// Repack a guest vertex stream into the tightly-packed `Float32xN` layout the recompiled
    /// pipeline expects. One packed vertex per guest vertex, in order, so the index buffer is
    /// unchanged.
    ///
    /// Appends into a caller-owned arena rather than returning a `Vec`: the arena is uploaded
    /// once per pass, so a per-draw allocation here would put back exactly the cost the arena
    /// exists to remove.
    /// What a vertex attribute's components ABOVE the guest's binding are fed - **1.0**
    /// (`VITASLOP_GXP_ATTR_FILL=api` restores the graphics API's `(0, 0, 0, 1)`).
    ///
    /// # THE (0, 0, 0, 1) FILL WAS NEVER A READING, AND IT COST A WHOLE TITLE ITS COLOUR
    /// It is what WebGPU, GL and D3D supply, and this renderer supplied it only by binding the
    /// NARROW vertex format and letting the API decide - which is not the same thing as deciding.
    /// The guest's vertex fetch is what has to be reproduced, and nothing had ever asked what it
    /// leaves there.
    ///
    /// **MEASURED.** one title's sky/background family forwards `In.UV1` - declared four
    /// components, BOUND `F16x2` - straight into a varying its fragment reads as a colour
    /// MODULATE. Components `z`/`w` of an attribute nothing writes therefore multiply a channel,
    /// and a zero there is a DEAD CHANNEL: that is the in-race world reading green/yellow with
    /// the road at `(194, 220, 119)` and the sky's blue at 2. At 1.0 - a missing modulate
    /// component being the identity rather than zero - the road's blue comes back to 199 and the
    /// sky goes neutral. **`f008900.png` and the whole outdoor stretch stop being green.**
    ///
    /// **Blast radius, measured rather than argued:** one title's title screen
    /// (`f005600`) and another title's tutorial drive (`f002400`) are **BIT-IDENTICAL**
    /// either way - their programs
    /// do not read past what their guest binds. The only component this can change that the old
    /// default also set is `w`, which both readings make 1.0.
    fn attr_fill(component: usize) -> f32 {
        use std::sync::OnceLock;
        static MODE: OnceLock<Option<f32>> = OnceLock::new();
        let mode = *MODE.get_or_init(|| match crate::knobs::var("VITASLOP_GXP_ATTR_FILL").ok().as_deref() {
            // The graphics API's fill, kept so the change above stays an A/B.
            Some("api") => Some(f32::NAN),
            Some("zero") => Some(0.0),
            _ => None,
        });
        match mode {
            Some(v) if v.is_nan() => {
                if component == 3 {
                    1.0
                } else {
                    0.0
                }
            }
            Some(v) => v,
            None => 1.0,
        }
    }

    fn repack_vertices_into(
        vertices: &[u8],
        guest_stride: u32,
        repack: &[RepackAttr],
        packed_stride: u32,
        out: &mut Vec<u8>,
    ) {
        let gstride = guest_stride.max(1) as usize;
        let nverts = vertices.len() / gstride;
        out.reserve(nverts * packed_stride as usize);
        for i in 0..nverts {
            let vbase = i * gstride;
            // Zero-fill this packed vertex, then write each attribute at its packed offset.
            let start = out.len();
            out.resize(start + packed_stride as usize, 0);
            for a in repack {
                for c in 0..a.slots as usize {
                    let f = if c < a.components as usize {
                        read_attr_component(vertices, vbase + a.guest_offset as usize, a.gxm_format, c)
                    } else {
                        attr_fill(c)
                    };
                    let po = start + a.packed_offset as usize + c * 4;
                    out[po..po + 4].copy_from_slice(&f.to_le_bytes());
                }
            }
        }
    }

    /// The depth encoding to assume for a pass whose own draws could not produce one: `-1/w`,
    /// i.e. `a = 0, c = -1`. It is the encoding with no projection in it, so a frame built on
    /// it is visibly a fallback rather than a plausible-looking guess.
    const DEPTH_FIT_RECIP_W: (f32, f32) = (0.0, -1.0);

    /// What interpreting a pair's vertex program over its OWN mesh says about the projection
    /// behind it. See [`count_clip_w_signs`].
    #[derive(Clone, Copy, Debug, Default)]
    struct ClipStats {
        /// Sampled vertices landing inside the frustum with clip `w > 0`, and with `w < 0`.
        in_front: usize,
        behind: usize,
        sampled: usize,
        /// NDC bounding box under `x/|w|`.
        bbox: [f32; 4],
        /// `(a, c)` of `z_clip = a * w_clip + c`, and the spread of `w` the fit was taken over.
        ///
        /// A projection matrix makes clip `z` an AFFINE function of clip `w` - both are the same
        /// dot product against the same point, differing only in which matrix column they use -
        /// so this holds EXACTLY, per vertex, for every draw sharing one projection. It is what
        /// gives the guest's own window depth `z/w = a + c/w` without reflecting the matrix, and
        /// therefore what lets a depth surface be re-encoded the way the guest wrote it rather
        /// than the way we guessed. `None` when this draw has no two vertices with different
        /// `w` to fit through.
        depth_fit: Option<(f32, f32)>,
        w_spread: f32,
    }

    /// Interpret a pair's vertex program over its OWN mesh and measure what it says about the
    /// projection: how many vertices land in front of the eye and how many behind, and how clip
    /// `z` relates to clip `w`. `None` when the program cannot be interpreted at all.
    ///
    /// Sampled with an even STRIDE through the mesh rather than as a prefix: vertex 0 of a
    /// world mesh is routinely behind the camera, so a prefix answers a different question
    /// than "does this draw cover anything".
    fn count_clip_w_signs(gxp: &GxpRecompile, max_samples: usize) -> Option<ClipStats> {
        let vrc = vitaslop_gxp_shader::recompile_vertex(&gxp.vprog).ok()?;
        let mut base = vitaslop_gxp_shader::interp::RegFile::with_lanes(512);
        for (k, c) in gxp.vert_sa.chunks_exact(4).enumerate() {
            if k < base.sa.len() {
                base.sa[k] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        // The secondary program (and the container's literals) run first and overwrite SA
        // registers the primary reads, so skipping them interprets the primary against
        // uniforms the real module has already replaced.
        if let Ok(program) = vitaslop_gxp_shader::Program::parse(&gxp.vprog) {
            for &(reg, value) in &program.literals {
                if let Some(slot) = base.sa.get_mut(reg as usize) {
                    *slot = f32::from_bits(value);
                }
            }
            let secondary = vitaslop_gxp_shader::usse::decode_secondary_shader(&program);
            if vitaslop_gxp_shader::interp::run(&secondary, &mut base).is_err() {
                return None;
            }
        }
        let stride = gxp.vertex_stride.max(1) as usize;
        let nverts = gxp.vertices.len() / stride;
        if nverts == 0 {
            return None;
        }
        let step = (nverts / max_samples.max(1)).max(1);
        // The VERTEX stage's own textures, so a program that builds its geometry from a fetch
        // can be interpreted at all. Nearest, wrapped, no mip: this is a measurement of where
        // the geometry LANDS, and a filter would change nothing about that while costing four
        // texel reads per sample.
        let fetch = |unit: u8, coord: [f32; 4]| -> Option<[f32; 4]> {
            let t = &gxp.vertex_textures.iter().find(|t| t.unit == unit)?.tex;
            let (w, h) = (t.width.max(1), t.height.max(1));
            let wrap = |v: f32, n: u32| {
                let n = n as f32;
                let p = (v * n).floor().rem_euclid(n);
                p as u32
            };
            let (x, y) = (wrap(coord[0], w), wrap(coord[1], h));
            let bpt = t.texel.bytes_per_texel();
            let o = (y * w + x) as usize * bpt;
            let px = t.rgba.get(o..o + bpt)?;
            // Read through the seam the texture was decoded onto. Narrowing here would put
            // back exactly the precision loss the half seam exists to avoid, and on the one
            // path whose whole job is to say where a vertex-fetched geometry LANDS.
            Some(match t.texel {
                TexelSeam::Rgba8 => [
                    px[0] as f32 / 255.0,
                    px[1] as f32 / 255.0,
                    px[2] as f32 / 255.0,
                    px[3] as f32 / 255.0,
                ],
                TexelSeam::Rgba16Float => {
                    let h = |i: usize| half_to_f32(u16::from_le_bytes([px[i * 2], px[i * 2 + 1]]));
                    [h(0), h(1), h(2), h(3)]
                }
            })
        };
        // Which PA lanes an attribute actually supplies (see the default fill below).
        let mut claimed = vec![false; base.pa.len()];
        for a in &gxp.attributes {
            for c in 0..a.components as usize {
                if let Some(slot) = claimed.get_mut(a.reg_index as usize + c) {
                    *slot = true;
                }
            }
        }
        let (mut in_front, mut behind, mut sampled) = (0usize, 0usize, 0usize);
        let mut bbox = [f32::MAX, f32::MIN, f32::MAX, f32::MIN];
        // Least-squares accumulators for `z = a*w + c` over the sampled vertices, in f64: the
        // fit's `c` is a small difference between two large dot products (on one retail racer,
        // `z` and `w` agree to four decimal places and `c` is about -1), and in f32 that
        // subtraction loses most of its significant digits.
        let (mut sw, mut sz, mut sww, mut swz) = (0f64, 0f64, 0f64, 0f64);
        let (mut wlo, mut whi) = (f32::MAX, f32::MIN);
        for v in (0..nverts).step_by(step) {
            let mut regs = base.clone();
            for a in &gxp.attributes {
                let vbase = v * stride + a.offset as usize;
                // All FOUR lanes of the attribute's register, because that is what the pipeline
                // feeds the real shader: the linked module reads `in.aN.xyzw`, and WebGPU fills
                // the components a `Float32x3` vertex format does not supply with (0,0,0,1). An
                // interpretation that zero-fills instead reads `w = 0` where the GPU reads 1,
                // which on a 3-component POSITION drops the projection's translation term - i.e.
                // it measures a different clip `w` than the frame it is supposed to explain.
                for c in 0..4usize {
                    let lane = a.reg_index as usize + c;
                    if lane >= regs.pa.len() {
                        continue;
                    }
                    if c < a.components as usize {
                        regs.pa[lane] = read_attr_component(&gxp.vertices, vbase, a.gxm_format, c);
                    } else if !claimed[lane] {
                        // Only where no OTHER attribute owns the lane - two attributes packed
                        // two lanes apart would otherwise overwrite each other's components.
                        regs.pa[lane] = if c == 3 { 1.0 } else { 0.0 };
                    }
                }
            }
            if vitaslop_gxp_shader::interp::run_watching_for_nan_with_textures(
                &vrc.shader,
                &mut regs,
                &fetch,
            )
            .is_err()
            {
                return None;
            }
            sampled += 1;
            // Count the vertices this draw would actually PUT ON SCREEN under each reading of
            // the sign, rather than the sign alone. A mesh that straddles the camera plane has
            // both signs in it, and a vote on the sign alone leaves it uncorrected - which
            // clips away exactly the half that is in view and keeps the half behind the eye.
            // Inside the frustum means `|x| <= |w|` and `|y| <= |w|` on the side being tested.
            let (x, y, w) = (regs.o[0], regs.o[1], regs.o[3]);
            let inside = x.abs() <= w.abs() && y.abs() <= w.abs();
            if w > 0.0 && inside {
                in_front += 1;
            } else if w < 0.0 && inside {
                behind += 1;
            }
            // Where the draw lands once the sign is corrected: `x/|w|`, which is the same NDC
            // either reading resolves to for the vertices that are actually on screen.
            if w != 0.0 {
                let (nx, ny) = (x / w.abs(), y / w.abs());
                bbox = [bbox[0].min(nx), bbox[1].max(nx), bbox[2].min(ny), bbox[3].max(ny)];
            }
            let z = regs.o[2];
            if w.is_finite() && z.is_finite() {
                let (wd, zd) = (w as f64, z as f64);
                sw += wd;
                sz += zd;
                sww += wd * wd;
                swz += wd * zd;
                wlo = wlo.min(w);
                whi = whi.max(w);
            }
        }
        let n = sampled as f64;
        let denom = n * sww - sw * sw;
        // A mesh whose vertices all share one `w` (a 2D overlay, a billboard) pins no line
        // through them, and inventing one from a degenerate system would put a wild `c` into
        // the depth encoding. Requiring a real spread of `w` is what keeps such a draw from
        // being asked a question it cannot answer.
        let depth_fit = (sampled >= 2 && denom.abs() > 1e-9 && (whi - wlo).abs() > 1e-3).then(|| {
            let a = (n * swz - sw * sz) / denom;
            let c = (sz - a * sw) / n;
            (a as f32, c as f32)
        });
        Some(ClipStats {
            in_front,
            behind,
            sampled,
            bbox,
            depth_fit,
            w_spread: if whi > wlo { whi - wlo } else { 0.0 },
        })
    }

    /// Weigh, ONCE per shader pair, how many of that pair's vertices land in the frustum with
    /// each sign of clip `w`. Reports what it saw unconditionally: a correction this large
    /// that applied itself silently would be indistinguishable from a faithful projection by
    /// looking at the frame.
    ///
    /// `None` = the pair's vertex program does not interpret at all, so it carries no evidence
    /// and never will.
    fn measure_clip(gxp: &GxpRecompile, key: u64) -> Option<ClipStats> {
        match count_clip_w_signs(gxp, 256) {
            Some(s) => {
                report!(
                    "gxp clip: key {key:x}: of {} sampled vertices, {} land in the frustum with w>0 \
                     and {} with w<0; ndc x[{:.2},{:.2}] y[{:.2},{:.2}]; z = {}",
                    s.sampled,
                    s.in_front,
                    s.behind,
                    s.bbox[0],
                    s.bbox[1],
                    s.bbox[2],
                    s.bbox[3],
                    match s.depth_fit {
                        Some((a, c)) => format!("{a}*w + {c} over w spread {}", s.w_spread),
                        None => "(no w spread to fit through)".into(),
                    }
                );
                Some(s)
            }
            None => {
                report!(
                    "gxp clip: key {key:x}: could not be measured (the vertex program does not \
                     interpret) - it contributes no evidence about this pass's projection"
                );
                None
            }
        }
    }

    /// Inject the GXM->WebGPU clip fixup into a linked module: wrap the vertex stage's
    /// `out.position` assignment in a helper that remaps clip Z (and optionally flips Y).
    ///
    /// The linker emits exactly one `  out.position = <expr>;` statement, but the exact shape of
    /// `<expr>` is the emitter's business and has changed. So this matches the STATEMENT and
    /// wraps whatever it assigns. Returns `None` when the statement is not found, which makes
    /// the pair fall back rather than render with no depth remap at all: this transform is not
    /// cosmetic, it is what puts the guest's clip depth inside WebGPU's `0 <= z <= w` clip
    /// volume, and without it the hardware clips away every triangle whose raw clip z runs past
    /// w (this title's does, by roughly 5x) - which looks like a mesh mysteriously missing its
    /// far half, not like a broken depth buffer.
    fn inject_clip_fixup(wgsl: &str, zfix: ZFix, yflip: bool, solid: bool, keycolor: Option<u64>) -> Option<String> {
        // Replace the guest's clip z with the SAME depth the fixed-function path writes, so
        // recompiled and fixed-function draws share one comparable depth buffer: the projected
        // view distance through `-1/w`, mapped linearly onto [0,1] over the scene's visible
        // range (see `render::project` for why the guest's own clip z is not a depth here).
        // Keeping xy exact leaves the real shader's projection untouched. w<=0 (behind the eye)
        // is left to wgpu's clip.
        let z = match zfix {
            ZFix::Range => {
                "  if (c.w > 0.0) { let q = -1.0 / c.w;\n    r.z = clamp((q - gxp_depth.range.x) * gxp_depth.range.y, 0.0, 1.0) * c.w; }\n"
            }
            // The ORDINARY GL->WebGPU depth remap: the guest's own clip z, which GXM reads in
            // [-w, w], mapped into WebGPU's [0, w]. It needs no scene statistics at all, so it
            // cannot be thrown off by a depth range measured through a different projection
            // than the one the recompiled shader actually uses.
            ZFix::Gl => "  r.z = (c.z + c.w) * 0.5;\n",
            // The guest's own window depth, CLAMPED rather than clipped (see `ZFix::Clamp`).
            // `c.w <= 0` is behind the eye and left to wgpu's own clip, exactly as `Range` does.
            ZFix::Clamp => "  if (c.w > 0.0) { r.z = clamp(c.z / c.w, 0.0, 1.0) * c.w; }\n",
            // The guest's OWN viewport depth mapping (`fit.z` = zScale, `fit.w` = zOffset),
            // clamped exactly as `Clamp` clamps. See `ZFix::Viewport`.
            ZFix::Viewport => {
                "  if (c.w > 0.0) { r.z = clamp(c.z / c.w * gxp_depth.fit.z + gxp_depth.fit.w, 0.0, 1.0) * c.w; }\n"
            }
            ZFix::Off => "",
        };
        let y = if yflip { "  r.y = -c.y;\n" } else { "" };
        // The clip-w SIGN correction (`gxp_depth.range.z`, +1 or -1; see `NegW`). A title whose
        // projection puts clip `w` NEGATIVE in front of the camera names every visible point on
        // the half of the homogeneous line WebGPU clips away. Negating all four components names
        // the SAME point with `w > 0` (homogeneous coordinates are scale-invariant) and moves
        // behind-camera geometry to `w < 0`, where it belongs. It is one multiply by a UNIFORM,
        // which is what keeps the decision PER DRAW rather than per vertex - a per-vertex
        // `if (w < 0) { c = -c; }` would pull the geometry behind the eye into view.
        //
        // `gxp_depth.range.z` selects the correction: 1 = none, -1 = negate the whole vector,
        // 2 = flip the sign of `w` alone. The two are DIFFERENT renders, not two spellings of
        // one: negating all four components leaves `x/w` exactly as it was (that is what makes
        // it a rename of the same point), so it fixes the clip and nothing else, while flipping
        // `w` alone MIRRORS the draw in both axes. Which one a title needs is a measurement.
        //
        // The `GxpDepth` block itself is declared by the LINKER (`link::GXP_DEPTH_DECL`), not
        // here: the fragment stage reads the same block to reconstruct the guest's window
        // position, and a linked module has to be independently compilable.
        let helper = format!(
            "fn gxp_clipfix(cin: vec4<f32>) -> vec4<f32> {{\n\
             \x20 var c = cin;\n\
             \x20 if (gxp_depth.range.z < 0.0) {{ c = -cin; }}\n\
             \x20 else if (gxp_depth.range.z > 1.5) {{ c = vec4<f32>(cin.x, cin.y, cin.z, -cin.w); }}\n\
             \x20 var r = c;\n{z}{y}  return r;\n}}\n"
        );
        const ASSIGN: &str = "\n  out.position = ";
        let at = wgsl.find(ASSIGN)?;
        let rhs_start = at + ASSIGN.len();
        let rhs_end = rhs_start + wgsl[rhs_start..].find(";")?;
        let mut patched = String::with_capacity(wgsl.len() + 64);
        patched.push_str(&wgsl[..rhs_start]);
        patched.push_str("gxp_clipfix(");
        patched.push_str(&wgsl[rhs_start..rhs_end]);
        patched.push(')');
        patched.push_str(&wgsl[rhs_end..]);
        // Diagnostic (`VITASLOP_GXP_KEYCOLOR`): shade every pair a flat colour derived from its
        // own key, keeping its real geometry, depth and blend. One run then answers "WHICH pair
        // owns this region of the screen" for every region at once - the question that otherwise
        // costs one `VITASLOP_GXP_KEYS` run per candidate, and the one that has to be answered
        // before any question about what a surface's shader computes.
        if let Some(k) = keycolor {
            let chan = |shift: u32| ((k >> shift) & 0xff) as f32 / 255.0;
            let (r, g, b) = (0.25 + 0.75 * chan(0), 0.25 + 0.75 * chan(21), 0.25 + 0.75 * chan(42));
            match patched.rfind("\n  return ") {
                Some(at) => {
                    let end = patched[at + 1..].find(";\n").map(|e| at + 1 + e + 1).unwrap_or(patched.len());
                    // Keep the pair's REAL alpha. Forcing it to 1.0 does not merely recolour the
                    // frame, it changes WHICH pair owns a pixel: a blended draw that contributes
                    // a few percent becomes an opaque cover, and one late fullscreen quad then
                    // hides every pair behind it. That is the exact question this instrument
                    // exists to answer, so the alpha is carried through rather than replaced.
                    let expr = patched[at + 1 + "  return ".len()..end - 1].trim().to_string();
                    patched.replace_range(
                        at + 1..end,
                        &format!("  return vec4<f32>({r:.3}, {g:.3}, {b:.3}, ({expr}).w);"),
                    );
                    // Print the assignment: reading the colour back OFF the frame means undoing
                    // whatever transfer function the target applied, and a near-match to the
                    // wrong key is not distinguishable from a match to the right one.
                    report!("gxp keycolor: key {k:x} -> linear rgb({r:.3}, {g:.3}, {b:.3})");
                }
                // At WARN, for the same reason as the `solid` arm below: an instrument that
                // reports its own absence below the default filter reports nothing at all.
                None => report_warn!(
                    "gxp build: VITASLOP_GXP_KEYCOLOR found no fragment return to replace for key \
                     {k:x} - the module shape changed; this pair is NOT key-coloured"
                ),
            }
        }
        if solid {
            // Diagnostic: force the fragment to solid magenta so any on-screen triangle is
            // visible regardless of shading. The colour expression depends on which register
            // holds the colour and at what precision, so this matches the fragment entry's
            // LAST `return` statement structurally rather than by its exact text - an
            // enumerate-the-spellings version silently stopped substituting anything the first
            // time the emitter's return line changed, which made the diagnostic lie.
            match patched.rfind("\n  return ") {
                Some(at) => {
                    let end = patched[at + 1..]
                        .find(";\n")
                        .map(|e| at + 1 + e + 1)
                        .unwrap_or(patched.len());
                    patched.replace_range(at + 1..end, "  return vec4<f32>(1.0, 0.0, 1.0, 1.0);");
                }
                // At WARN. This is the instrument reporting that it did NOTHING, and it was
                // at `debug` while every repro command says `VITASLOP_LOG=warn` - so a
                // `VITASLOP_GXP_SOLID` run came back BIT-IDENTICAL to the unpatched one
                // (0 of 522240 pixels), the substitution silently absent, and the frame read
                // as "this geometry does not rasterize". It rasterizes fine. The comment
                // above about an enumerate-the-spellings version that "made the diagnostic
                // lie" describes this failure exactly, and structural matching did not cure
                // it - only saying so out loud does.
                None => report_warn!(
                    "gxp build: VITASLOP_GXP_SOLID found no fragment return to replace - \
                     the module shape changed; solid-fill is NOT in effect"
                ),
            }
        }
        Some(format!("{helper}{patched}"))
    }

    /// Report, ONCE PER SHADER PAIR, that a draw is rendering the fixed-function
    /// approximation instead of the guest's own translated shaders, and exactly why.
    ///
    /// This is deliberately not gated behind a debug flag. The recompiler's contract is
    /// no-guess: an unknown opcode hard-fails naming its raw word, an unfed shader input
    /// hard-fails, a clip fixup that finds nothing to patch hard-fails. But the RENDERER
    /// answers all of those by drawing the older capture-based approximation for that pair,
    /// and a silent approximation is indistinguishable on screen from a faithful render -
    /// which is how session 8's "36 draws render with their real guest shaders" turned out
    /// to be 36 draws that linked and never ran. The reason line is the only signal that a
    /// pair is still owed work, so it must be visible in an ordinary run.
    ///
    /// The dedupe is global rather than per-renderer: a pair is submitted hundreds of times a
    /// frame, and nothing is gained by repeating it for a second render target.
    /// How far past a render target's base address a sampled texture may start and still be
    /// that target. One 960-pixel row at 4 bytes is 3840, so this covers "the same buffer,
    /// described from a slightly different origin" and cannot reach the next surface: this
    /// title's own targets are 16 KiB apart.
    const RTT_ALIAS_SLACK: u32 = 4096;

    /// The rendered target `addr` refers to, allowing for a small positive offset into it.
    ///
    /// A title does not always sample a render target through a texture describing exactly that
    /// target. This one aliases its 960x544 world buffer as a **1920x1088** texture starting 256
    /// bytes in - the 2x supersampled view of the same memory - and reads it that way from every
    /// post-process pass. Matching the data address EXACTLY misses that completely: the sample
    /// falls through to the guest bytes, which the GPU never wrote, and the light, blur and
    /// composite passes all render black. Matching by RANGE binds the render they meant.
    ///
    /// Reports the first alias it resolves per (pair, unit): substituting a different buffer
    /// than the one the guest named is exactly the kind of helpfulness that must not be silent.
    fn rendered_alias(
        rendered: &HashMap<u32, wgpu::TextureView>,
        addr: u32,
        key: u64,
        unit: u8,
    ) -> Option<u32> {
        if addr == 0 {
            return None;
        }
        if rendered.contains_key(&addr) {
            return Some(addr);
        }
        let base = rendered
            .keys()
            .copied()
            .filter(|&b| addr > b && addr - b <= RTT_ALIAS_SLACK)
            .max_by_key(|&b| b)?;
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(u64, u8)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert((key, unit)) {
            report!(
                "gxp pair {key:016x}: sampler unit {unit} names {addr:#x}, which is {} bytes into \
                 the render target at {base:#x} - binding that target's render (the guest is \
                 describing the same buffer from a different origin)",
                addr - base
            );
        }
        Some(base)
    }

    /// Report - once, with the high-water mark - that ONE FRAME's texture working set does not
    /// fit the cache budget, so the budget had to be exceeded to finish the frame.
    ///
    /// The eviction policy will not throw away a texture the frame in flight has already used
    /// (see [`GxpLive::views_used`]), so this is the case where there was nothing else to give.
    /// It needs saying because the consequence is not a slow frame - it is unbounded GPU memory
    /// on a device that kills a worker for it without an error, a crash event or a log line.
    /// `VITASLOP_TEX_CACHE_MB` raises the budget; the real fix is a smaller working set.
    fn report_texture_budget_exceeded(
        bytes: usize,
        views_used: &HashMap<(u64, SamplerDim), (u64, usize, u32, Residency)>,
        epoch: u64,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static WORST: AtomicUsize = AtomicUsize::new(0);
        // >>> A NEW HIGH-WATER MARK IS NOT A NEW FINDING, and reporting every one of them
        // turned this into the second-largest source of noise in a device capture: the working
        // set climbs a megabyte at a time (256, 260, 261, 263, 264 ...) and each step was a
        // fresh `warn`. MEASURED: 1,846 lines of it in one run, against a panel that keeps 96.
        // A step is required, so the line fires when the picture MATERIALLY changes.
        const STEP_MB: usize = 32;
        let mb = bytes / (1024 * 1024);
        let prev = WORST.load(Ordering::Relaxed);
        if mb < prev + if prev == 0 { 1 } else { STEP_MB } {
            return;
        }
        WORST.store(mb, Ordering::Relaxed);
        // >>> AND IT NAMES ITS OWN CAUSE. "Reduce the working set" is not actionable without
        // knowing what the working set IS: a Vita cannot reference 264 MB of texture, so the
        // interesting question is which formats INFLATE on the way in - a PVRTC surface has no
        // WebGPU format and is decoded to RGBA8, which is an 8x expansion, while a BC one is
        // uploaded compressed and expands not at all. Without this breakdown the next step is a
        // guess; with it the top row IS the next piece of work.
        report_warn!(
            "gxm textures: one frame's working set is {mb} MB against a \
             {} MB budget, and every entry is in use by this frame - the budget is being \
             EXCEEDED to finish it. Raise VITASLOP_TEX_CACHE_MB or reduce the working set. \
             Biggest contributors this frame: {}",
            tex_cache_budget_bytes() / (1024 * 1024),
            texture_composition(views_used, epoch).join(", ")
        );
    }

    /// How a texture is actually resident on the GPU, for the working-set breakdown.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Residency {
        /// CPU-decoded to RGBA8 (or RGBA16F) - the expansion the budget is about.
        Decoded,
        /// The guest's own blocks, handed over unchanged.
        Passthrough,
        /// Blocks we re-encoded from decoded texels.
        Transcoded,
    }

    /// The frame's texture working set BY FORMAT, biggest first - the breakdown both texture
    /// reports print, written once so they cannot describe the same frame differently.
    fn texture_composition(
        views_used: &HashMap<(u64, SamplerDim), (u64, usize, u32, Residency)>,
        epoch: u64,
    ) -> Vec<String> {
        let mut by_format: HashMap<(u32, Residency), (usize, usize)> = HashMap::default();
        for (_, (used, b, fmt, resident)) in views_used.iter() {
            if *used == epoch {
                let e = by_format.entry((*fmt, *resident)).or_insert((0, 0));
                e.0 += *b;
                e.1 += 1;
            }
        }
        let mut rows: Vec<((u32, Residency), usize, usize)> =
            by_format.into_iter().map(|(f, (b, n))| (f, b, n)).collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.1));
        rows.iter()
            .take(8)
            .map(|((f, resident), b, n)| {
                format!("{:#04x} {} x{n} ({:.1} MB)", f, format_note(*f, *resident), *b as f64 / 1048576.0)
            })
            .collect()
    }

    /// Report one frame's texture working set and its composition at each new high-water mark,
    /// WHETHER OR NOT it fits the budget.
    ///
    /// # A report that only fires on failure cannot measure a fix
    /// [`report_texture_budget_exceeded`] carries the same breakdown and used to be the only
    /// thing that ever printed it - so the moment a change brought the working set under budget,
    /// the number that would have shown by how much disappeared with it. Measuring the decode
    /// path against the compressed upload gave 274 MB for the arm that FAILED and silence for
    /// the two that worked, which is the one shape of result an A/B must not have.
    ///
    /// # NOT behind a knob, because a knob nobody sets is the same as no report
    /// This is the number the whole texture-memory story is told in, and gating it would mean the
    /// engine that needs it most - a phone, which has no environment to set a knob in - never
    /// emits it at all. The volume is bounded instead: a 32 MB step over a 274 MB run is nine
    /// lines, against a device panel that keeps 96. The budget report's own history is why the
    /// step matters - it fired on every 1 MB change and produced 1,846 lines in one run.
    fn report_texture_working_set(
        bytes: usize,
        views_used: &HashMap<(u64, SamplerDim), (u64, usize, u32, Residency)>,
        epoch: u64,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static WORST: AtomicUsize = AtomicUsize::new(0);
        const STEP_MB: usize = 32;
        // Publish the frame's working set whether or not it is a new high, so the CPU side can
        // ask whether compressing anything further is still buying something. See
        // [`texture_budget_pressure`].
        crate::gpu::LAST_WORKING_SET.store(bytes, Ordering::Relaxed);
        let mb = bytes / (1024 * 1024);
        let prev = WORST.load(Ordering::Relaxed);
        if mb < prev + if prev == 0 { 1 } else { STEP_MB } {
            return;
        }
        WORST.store(mb, Ordering::Relaxed);
        // >>> NAME THE ADAPTER WHEN THE ADAPTER IS THE REASON.
        //
        // Without this the composition reads "x94 PVRTC, ~8x EXPANDED" on a device that CANNOT
        // do anything else, and every reader has to work out from the ABSENCE of refusal reports
        // that the format gate shut before any of them could fire. Absence of evidence is the
        // worst possible way to publish the single fact that decides whether this whole path is
        // available - MEASURED on the target phone, whose img-tec D-series adapter offers ASTC
        // and ETC2 but not BC, so the passthrough and the transcode are both unreachable there.
        let why = if crate::gpu::block_compression_available() {
            ""
        } else {
            " NOTE: this adapter has NO BC support, so every compressed guest format is decoded \
              to RGBA8 and no passthrough or transcode is possible here - the fix for this device \
              is an ETC2/ASTC encoder, not a refusal to chase."
        };
        report_warn!(
            "gxm textures: one frame's texture working set is {mb} MB (new high). \
             Composition: {}.{why}",
            texture_composition(views_used, epoch).join(", ")
        );
    }

    /// Whether a guest base format costs GPU memory beyond its compressed size, for the
    /// working-set breakdown. The distinction is the whole point of that report: a format with
    /// no WebGPU equivalent is CPU-decoded to RGBA8 and lands on the GPU expanded, and that
    /// expansion - not the number of textures - is what puts a phone over budget.
    /// # This used to state an aspiration and read as a measurement
    /// `0x85..=0x88` was annotated "BC, uploaded compressed" while nothing in the project
    /// uploaded anything compressed - both devices asked for `Features::empty()` and
    /// `upload_gxp_texture` only ever created `Rgba8Unorm`. So the one report whose whole job is
    /// to say WHERE the megabytes are told the reader that a third of them were already as small
    /// as they could be. The flag now comes from the upload decision itself.
    fn format_note(base_format: u32, resident: Residency) -> &'static str {
        match resident {
            // The difference between a lossless handover and a second lossy pass, in the one
            // report anyone reads to find out where the megabytes went.
            Residency::Passthrough => return "uploaded COMPRESSED (guest blocks, guest mips)",
            // >>> NAME THE FAMILY IT WAS ENCODED TO, not the one this line was written for.
            // This said "RE-ENCODED to BC" unconditionally, and a device capture caught it
            // saying so two lines under its own `RE-ENCODED to Etc2Rgb8` warning. The working-set
            // breakdown is the one report anyone reads to find out where the megabytes went, and
            // on the device that needs ETC2 it was naming the format that device cannot take.
            Residency::Transcoded => {
                return match super::block_family() {
                    BlockFamily::Etc2 => "RE-ENCODED to ETC2 (lossy)",
                    _ => "RE-ENCODED to BC (lossy)",
                }
            }
            Residency::Decoded => {}
        }
        match base_format {
            0x80..=0x84 => "PVRTC/PVRTC2 -> RGBA8, ~8x EXPANDED",
            0x85..=0x8b => "BC -> RGBA8, ~4-8x EXPANDED (no passthrough - see the report above)",
            _ => "raw",
        }
    }

    /// Report - once per (pair, unit) - that a recompiled draw is sampling a texture with a NULL
    /// data pointer, i.e. the 1x1 zero texel the runtime substitutes for an uninitialised
    /// `SceGxmTexture` handle.
    ///
    /// This is not a fallback: the draw runs, with the guest's own shader, and produces a
    /// perfectly valid picture of nothing. That is exactly why it needs saying out loud - a
    /// render target that comes out uniformly black looks like a geometry, depth or blend
    /// problem, and this is none of those.
    fn report_zero_texel_sample(key: u64, unit: u8) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(u64, u8)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert((key, unit)) {
            report!(
                "gxp pair {key:016x}: sampler unit {unit} is bound to a 1x1 ZERO texel (the guest \
                 handle had null control words) - everything this draw derives from that sample \
                 is zero"
            );
        }
    }

    /// The guest's own depth mapping `(zScale, zOffset)` for a draw, from the viewport it was
    /// submitted under: the hardware's window depth is `z/w * zScale + zOffset`.
    ///
    /// `(1, 0)` - the identity, which is what this renderer applied for every draw of every
    /// title before there was a reading - when the guest never called `sceGxmSetViewport` for
    /// this pass (the all-zero sentinel), or when it left a degenerate `zScale` of 0 that would
    /// collapse the whole scene onto one plane. Both cases are REPORTED once per pair: an
    /// assumed depth convention is exactly the kind of thing that looks right until a title
    /// whose faces sort by nine ten-thousandths of the buffer draws itself inside out.
    fn gxm_viewport_depth(vp: &[f32; 6], key: u64) -> (f32, f32) {
        let (zo, zs) = (vp[4], vp[5]);
        if vp.iter().all(|v| *v == 0.0) {
            report_viewport_depth_assumed(key, "the guest set no viewport for this draw");
            return (1.0, 0.0);
        }
        if !zs.is_finite() || !zo.is_finite() || zs == 0.0 {
            report_viewport_depth_assumed(key, "its zScale is zero or non-finite");
            return (1.0, 0.0);
        }
        (zs, zo)
    }

    /// One line per pair whose depth mapping had to be assumed rather than read. See
    /// [`gxm_viewport_depth`].
    fn report_viewport_depth_assumed(key: u64, why: &str) {
        use std::collections::HashSet;
        use std::sync::Mutex;
        static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
        let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if !g.get_or_insert_with(HashSet::new).insert(key) {
            return;
        }
        report!(
            "gxp pair {key:x}: depth mapped with the IDENTITY (zScale 1, zOffset 0) because \
             {why} - the guest's own zScale/zOffset is what decides whether a LESS_EQUAL test \
             can separate two faces at all"
        );
    }

    /// The wgpu viewport rectangle a GXM viewport asks for, in pixels of a `w` x `h` target,
    /// or `None` when the guest left the default (all-zero) or is naming exactly the whole
    /// target - in which case there is nothing to set and nothing to restore.
    ///
    /// GXM maps normalised device coordinates to the framebuffer as `screen = offset + scale *
    /// ndc`, and `yScale` is normally NEGATIVE because ndc `+1` is the top of the screen while
    /// framebuffer row 0 is the top as well. wgpu's `set_viewport(x, y, w, h)` bakes that same
    /// flip in, so the two agree exactly whenever `yScale < 0`: the rect is centred on
    /// `(xOffset, yOffset)` with half-extents `|xScale|` and `|yScale|`.
    ///
    /// The DEPTH half (`zOffset`, `zScale`) is deliberately not applied. The recompiled vertex
    /// stage already writes a depth in whichever convention `VITASLOP_GXP_ZFIX` selects, so
    /// handing the guest's z mapping to `min_depth`/`max_depth` as well would apply it twice.
    fn gxm_viewport_rect(vp: &[f32; 6], w: u32, h: u32) -> Option<(f32, f32, f32, f32)> {
        let [xo, xs, yo, ys, _, _] = *vp;
        // All-zero is the sentinel for "the guest never called sceGxmSetViewport", not a
        // request for a zero-area viewport. A zero SCALE with a nonzero offset would be a
        // degenerate viewport and is reported below rather than silently ignored.
        if vp.iter().all(|v| *v == 0.0) {
            return None;
        }
        if !vp.iter().all(|v| v.is_finite()) {
            report_viewport_problem(vp, "it contains a non-finite component");
            return None;
        }
        if ys > 0.0 {
            // ndc +1 would land at the BOTTOM of the rect. wgpu's viewport cannot express a
            // vertical flip (it requires a positive height), so the rect below renders this
            // pass upside down. Saying so is the whole point - a silently mirrored pass is
            // indistinguishable from a correct one on a symmetric image.
            report_viewport_problem(vp, "yScale is POSITIVE, which is a vertical flip that a wgpu viewport cannot express - this pass renders mirrored");
        }
        let (x, y) = (xo - xs.abs(), yo - ys.abs());
        let (vw, vh) = (2.0 * xs.abs(), 2.0 * ys.abs());
        if vw <= 0.0 || vh <= 0.0 {
            report_viewport_problem(vp, "it has a zero-area rectangle");
            return None;
        }
        // Already the whole target: setting it is a no-op and skipping it keeps the common
        // fullscreen case free of state changes.
        if x == 0.0 && y == 0.0 && vw == w as f32 && vh == h as f32 {
            return None;
        }
        // wgpu rejects a viewport that leaves the attachment. Clamp INTO the target and say
        // so: rendering the pass at the wrong rect is wrong, but dropping the viewport
        // entirely is wrong in a way that looks like nothing happened.
        let (cx, cy) = (x.max(0.0), y.max(0.0));
        let (cw, ch) = ((vw + x.min(0.0)).min(w as f32 - cx), (vh + y.min(0.0)).min(h as f32 - cy));
        if (cx, cy, cw, ch) != (x, y, vw, vh) {
            report_viewport_problem(
                vp,
                &format!(
                    "it asks for ({x}, {y}, {vw}, {vh}) which leaves a {w}x{h} target - \
                     CLAMPED to ({cx}, {cy}, {cw}, {ch})"
                ),
            );
        }
        if cw <= 0.0 || ch <= 0.0 {
            return None;
        }
        Some((cx, cy, cw, ch))
    }

    /// Report - once per (surface, new mode) - that the guest CHANGED a live target's
    /// gamma-correct mode.
    ///
    /// Worth a line of its own because it is the event that used to desynchronise the
    /// renderer from itself: the recorded mode was written at creation and never updated, so
    /// after a change the pass encoded through one view while every sampler decoded through
    /// the other, and on a feedback path the mismatch compounded to white or to black. It is
    /// handled now, and this says when a title exercises it - a fact no other line carries.
    fn report_gamma_mode_changed(addr: u32, gamma: bool) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(u32, bool)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert((addr, gamma)) {
            report!(
                "gxm surface: {addr:#x} changed GAMMA-CORRECT mode to {gamma} on a live target - \
                 the recorded mode and its cached bind groups are refreshed, so what samples this \
                 target decodes the way the pass that wrote it encoded"
            );
        }
    }

    /// Report - once per surface - that a pass is rendering into a GAMMA-CORRECT colour
    /// surface, and whether that is being honoured.
    ///
    /// `honoured == false` means the colour format has no sRGB twin, so the ROP encode the
    /// hardware performs is not happening and this surface (and everything sampling it) reads
    /// darker than the title intends. That has to be said out loud - it is precisely the kind
    /// of uniform darkening that gets chased as a lighting bug.
    fn report_gamma_surface(addr: u32, honoured: bool) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert(addr) {
            if honoured {
                report!(
                    "gxm surface: {addr:#x} is GAMMA-CORRECT - rendering through an sRGB view, \
                     so writes are sRGB-encoded after blending as the hardware does"
                );
            } else {
                report!(
                    "gxm surface: {addr:#x} is GAMMA-CORRECT but the render format has no sRGB \
                     twin - its writes stay LINEAR where the hardware would encode them, so it \
                     and anything sampling it read darker than the title intends"
                );
            }
        }
    }

    /// Report - once per (pair, unit) - that a sampler was bound to a render target's DEPTH
    /// rather than to any colour buffer.
    ///
    /// The same class of event as [`rendered_alias`]: the runtime is substituting something
    /// other than the guest bytes at the address the shader named, and that substitution
    /// decides what the pass computes.
    fn report_depth_sample_bound(key: u64, unit: u8, addr: u32) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(u64, u8)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert((key, unit)) {
            report!(
                "gxp pair {key:016x}: sampler unit {unit} names {addr:#x}, which is a render \
                 target's DEPTH surface - binding this frame's converted depth"
            );
        }
    }

    /// Report - once per target - that a render target's depth was converted into the guest's
    /// encoding for a later pass to sample, and which encoding that was.
    ///
    /// Unconditional. Which value a GXM depth surface holds is REVERSE-ENGINEERED, not read
    /// off a spec we hold, so every frame built on that reading has to say which reading it
    /// used - otherwise a soft-particle fade that comes out subtly wrong looks like a shading
    /// bug rather than a choice made here.
    fn report_depth_conversion(addr: u32, mode: u32, konst: f32, depth_min: f32, depth_scale: f32) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(u32, u32)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert((addr, mode)) {
            if mode == 5 {
                report!(
                    "gxm depth: target {addr:#x} depth REPLACED by the constant {konst} - this is \
                     a causality probe, and any frame built on it is not a real frame"
                );
                return;
            }
            let name = match mode {
                0 => "the guest's own window depth (a + c/w, measured from this pass)",
                1 => "w",
                2 => "-w",
                3 => "-1/w",
                6 => "1/w",
                _ => "the stored [0,1] depth",
            };
            let degenerate = if depth_scale == 0.0 {
                "  (the scene has NO depth range, so w cannot be recovered and the stored depth \
                 is passed through unchanged)"
            } else {
                ""
            };
            report!(
                "gxm depth: target {addr:#x} depth re-encoded as {name} for a later pass to \
                 sample (depth_min={depth_min}, depth_scale={depth_scale}){degenerate}"
            );
        }
    }

    /// Report - once per address - that a pass samples a depth buffer the frame did render,
    /// but through the DISPLAY target rather than an offscreen one, where no converted copy
    /// exists.
    fn report_unconverted_depth_sample(addr: u32) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert(addr) {
            report!(
                "gxm depth: a draw samples the depth at {addr:#x}, which belongs to a scene \
                 rendered into the DISPLAY target - that pass keeps no depth copy, so this \
                 sample reads guest bytes the GPU never wrote"
            );
        }
    }

    /// Report - once per distinct viewport and message - that a guest viewport could not be
    /// reproduced exactly. Unconditional, like every other approximation in this path.
    fn report_viewport_problem(vp: &[f32; 6], what: &str) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<(String, String)>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        let vps = format!("{vp:?}");
        if seen.insert((vps.clone(), what.to_string())) {
            report!("gxp viewport {vps}: {what}");
        }
    }

    /// Report - once per pair and stage - a program that reads uniforms this renderer does not
    /// feed it: anything declared in a NON-DEFAULT container, or a `UniformBuffer` binding.
    ///
    /// # Why the bind site cannot say this
    /// `sceGxmSet{Vertex,Fragment}UniformBuffer` and its precomputed twins already warn, once
    /// per run, that a non-default buffer was bound and that the capture does not carry it. That
    /// warning names the CALL, and the call is not the defect: a title may bind a buffer that no
    /// shader on screen ever reads, in which case nothing is wrong and the warning is a false
    /// alarm that cannot be cleared. The defect is a PROGRAM whose parameter table says it reads
    /// one, because that program renders with zeros where the guest put data - silently, with
    /// zero fallbacks and zero WebGPU errors, which is this project's signature failure shape.
    ///
    /// So this is keyed by pair and stage and names the parameters, which makes it falsifiable:
    /// a run with the bind-site warning and none of these is a run where the gap costs nothing.
    fn report_unfed_uniforms(key: u64, stage: &str, blob: &[u8]) {
        use std::sync::{Mutex, OnceLock};
        use vitaslop_gxp_shader::container::ParamCategory;
        static SEEN: OnceLock<Mutex<HashSet<(u64, String)>>> = OnceLock::new();
        let Ok(program) = vitaslop_gxp_shader::Program::parse(blob) else { return };
        // ONLY a `UniformBuffer` parameter, which is unambiguous: it IS a buffer binding.
        //
        // This deliberately does NOT also flag `Uniform` parameters by `container_index`. That
        // was the first version, on the assumption that container 0 is the default uniform
        // buffer and anything else is a bound buffer - and it fired on nearly every pair in the
        // title, naming ordinary uniforms (`bloomFactor`, `blendweights`) that demonstrably DO
        // reach the shader, because the picture is right. So container 14 is not "some other
        // buffer" and the assumption was wrong.
        //
        // What `container_index` actually indexes is not established here, and a diagnostic
        // built on an unverified reading of a field is worse than no diagnostic: it produced
        // 200 lines of confident noise on one run. It stays out until someone reads the
        // container table and can say what the index means.
        // A VERTEX program whose memory loads RESOLVED to a window is fed - the draw
        // snapshots the bound buffer's bytes and the shader's loads read them - so it is no
        // longer an unfed declaration. (A fragment program has no window path at all, and a
        // vertex one that declares a buffer but never memory-loads still reads zeroes.)
        let fed_by_window = stage == "vertex"
            && vitaslop_gxp_shader::mem_window_for_vertex_blob(blob).is_some();
        let unfed: Vec<String> = program
            .parameters
            .iter()
            .filter(|p| matches!(p.category, ParamCategory::UniformBuffer))
            .map(|p| format!("{} (container {})", p.name, p.container_index))
            .collect();
        if unfed.is_empty() || fed_by_window {
            return;
        }
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert((key, stage.to_string())) {
            return;
        }
        report_warn!(
            "gxp pair {key:016x} {stage}: the program reads {} uniform(s) from a NON-DEFAULT \
             buffer, which the capture does not carry - those registers read ZERO in the \
             recompiled shader: {}",
            unfed.len(),
            unfed.join(", ")
        );
    }

    fn report_fallback(key: u64, reason: &str) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        // A poisoned lock must not lose the diagnostic - recover the set and report anyway.
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert(key) {
            report_warn!("gxp pair {key:016x}: FALLS BACK to fixed-function - {reason}");
        }
        drop(seen);
        fallback_reasons().lock().unwrap_or_else(|e| e.into_inner()).insert(key, reason.to_string());
        if !allow_fixed_function() {
            panic!(
                "gxp pair {key:016x} cannot be recompiled: {reason}\n\
                 The recompiler is enabled, so this draw would have been drawn by the \
                 fixed-function APPROXIMATION instead - a different renderer, which does not \
                 run the guest's shader and cannot be told apart from a faithful render by \
                 looking at the frame. Refusing. Set VITASLOP_GXP_ALLOW_FIXED_FUNCTION=1 to \
                 approximate anyway (bring-up only: it is how a title's world silently \
                 rendered 328 of 388 draws wrong)."
            );
        }
    }

    /// Whether a shader pair the recompiler cannot translate may be drawn by the
    /// fixed-function approximation instead (`VITASLOP_GXP_ALLOW_FIXED_FUNCTION=1`).
    ///
    /// OFF by default, and that default is the point. The approximation reconstructs a draw
    /// from captured state without running the guest's fragment program, so what it produces
    /// is plausible and wrong, and indistinguishable on screen from a correct render. It
    /// remains available for bring-up, where seeing a menu at all is what lets a recipe be
    /// authored - but a run that does not ask for it now stops at the first pair it cannot
    /// translate, naming the pair and the reason.
    fn allow_fixed_function() -> bool {
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| crate::knobs::flag("VITASLOP_GXP_ALLOW_FIXED_FUNCTION"))
    }

    /// Why each pair last fell back, keyed by pair.
    ///
    /// [`report_fallback`] PRINTS once per pair but RECORDS every time, so a per-scene tally
    /// can weight reasons by how many DRAWS each cost. That is the number that says what to
    /// fix next, and the printed list cannot give it: sixty pairs that fall back on one draw
    /// each and one pair that falls back on three hundred read exactly the same there.
    fn fallback_reasons() -> &'static std::sync::Mutex<HashMap<u64, String>> {
        use std::sync::{Mutex, OnceLock};
        static REASONS: OnceLock<Mutex<HashMap<u64, String>>> = OnceLock::new();
        REASONS.get_or_init(|| Mutex::new(HashMap::default()))
    }

    /// The recorded fallback reason for `key`, or a placeholder if the pair fell back before
    /// any reason was recorded (which would itself be a bug worth seeing).
    fn fallback_reason_of(key: u64) -> String {
        fallback_reasons()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            .cloned()
            .unwrap_or_else(|| "(no reason recorded)".to_string())
    }

    /// Report, once per pair, every USSE control-flow branch either program contains: the
    /// branch's instruction index, its signed word delta, the target that resolves to, and the
    /// program's instruction count.
    ///
    /// This is a MEASUREMENT, not a debug aid, and that is why it is unconditional like
    /// [`report_fallback`] rather than behind a flag. Branch translation rests on one fact the
    /// distilled ISA reference states but the captured shader corpus cannot corroborate (it
    /// contains no branches at all): that a target is `index + rel` rather than `index + 1 +
    /// rel`. Those differ by exactly one instruction, so the wrong reading leaves the last
    /// instruction of every conditional block running unconditionally - a plausible wrong
    /// picture with no error anywhere. A branch whose target lands exactly on `total` (one past
    /// the end, the shape a trailing `if` compiles to) confirms the implemented reading;
    /// targets that cluster one short of it, with none ever reaching `total`, would refute it.
    fn report_branches(key: u64, gxp: &GxpRecompile) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if !seen.insert(key) {
            return;
        }
        for (kind, bytes) in [("vertex", &gxp.vprog), ("fragment", &gxp.fprog)] {
            let Ok((total, sites)) = vitaslop_gxp_shader::branch_sites(bytes) else { continue };
            for s in sites {
                report!(
                    "gxp pair {key:016x}: {kind} BR #{} rel={} -> target {} of {total} instrs",
                    s.index, s.rel, s.target
                );
            }
        }
    }

    /// Link a guest shader pair and build its two pipeline variants + bind-group layouts.
    /// `None` (fall back) on any link error or an unmappable vertex format.
    ///
    /// Every `None` here is REPORTED, unconditionally and by name - see [`report_fallback`].
    /// The translator itself never guesses (an unknown opcode hard-fails naming its raw word),
    /// but the renderer's answer to that hard failure is to draw the fixed-function
    /// approximation, and an unreported approximation reads exactly like a faithful render.
    /// Report the fixed-function state a recompiled pair's pipeline was BAKED with, once per
    /// pair. Every field here is the guest's own captured state, and a wrong one is invisible
    /// in the frame in the worst way: a draw with the wrong depth function does not look
    /// broken, it looks like a different draw won. This is the line that says which.
    fn report_gxp_pipeline_state(
        key: u64,
        gxp: &GxpRecompile,
        depth_write: bool,
        depth_compare: wgpu::CompareFunction,
        blend: Option<wgpu::BlendState>,
        write_mask: wgpu::ColorWrites,
    ) {
        report!(
            "gxp pipeline {key:x} (fragment program {:#x}{}): depth {depth_compare:?}{} (guest func {:#x}, write {}), \
             guest viewport depth zScale={} zOffset={}, \
             blend {}, colour mask {write_mask:?}, guest cull {:#x}{}",
            gxp.fprog_header,
            if gxp.fragment_program_enabled { "" } else { ", DEPTH/STENCIL ONLY - the guest disabled the fragment program" },
            if depth_write { " + WRITE" } else { "" },
            gxp.depth_func,
            gxp.depth_write,
            gxp.viewport[5],
            gxp.viewport[4],
            match blend {
                Some(b) => format!("{:?}/{:?}", b.color, b.alpha),
                None => "REPLACE (the guest supplied no SceGxmBlendInfo)".into(),
            },
            gxp.cull_mode,
            match gxm_cull_face(gxp.cull_mode) {
                Some(wgpu::Face::Front) => " - culling FRONT faces",
                Some(wgpu::Face::Back) => " - culling BACK faces",
                None => "",
            }
        );
    }

    /// Map a `SceGxmCullMode` to the wgpu face to discard, against `FrontFace::Ccw`.
    ///
    /// # How the sense is pinned, since getting it backwards turns a model inside out
    /// The software rasteriser pinned its own winding EMPIRICALLY against a title's ground
    /// plane (`render::cull_backface`): in the Y-down framebuffer space `render::project`
    /// emits, with `edge(a,b,c) = (bx-ax)(cy-ay) - (by-ay)(cx-ax)`, `SCE_GXM_CULL_CCW`
    /// discards `edge < 0` and `SCE_GXM_CULL_CW` discards `edge > 0`.
    ///
    /// The GPU side is pinned the same way, MEASURED, and it came out the OPPOSITE way round to
    /// the obvious derivation - which is why it is measured rather than reasoned. Vulkan
    /// defines its signed area as `1/2 * sum(x_i*y_{i+1} - x_{i+1}*y_i)` over framebuffer
    /// coordinates, algebraically the same expression as `edge`, and calls it counter-clockwise
    /// when POSITIVE; reading that straight across gives `CULL_CW -> Face::Front`, and a
    /// retail race frame under it LOSES ITS WHOLE ROAD SURFACE - the camera looks
    /// through the track at the terrain below. Under the
    /// mapping below the same frame is the correct picture. Something between the guest's
    /// viewport, our clip fixup and wgpu's framebuffer convention flips the sign once more than
    /// that reading accounts for.
    ///
    /// So: `CULL_CW` discards the BACK face and `CULL_CCW` the FRONT, and any future change here
    /// has to be re-rendered against that frame rather than argued from a spec.
    ///
    /// Both engines now keep the same triangles, which is the property that matters: the
    /// software path has culled since it was written and the GPU path never did, so the two
    /// halves of this renderer disagreed about which faces exist and neither could check the
    /// other.
    fn gxm_cull_face(mode: u32) -> Option<wgpu::Face> {
        // `VITASLOP_GXP_CULL=0` is the A/B arm that restores "draw both windings", the behaviour
        // every pipeline had before 2026-08-19b. VALUE-sensitive, because it is an arm.
        if !super::gxp_cull() {
            return None;
        }
        // `SceGxmCullMode`: NONE = 0, CW = 1, CCW = 2. The same three values `render.rs` names.
        match mode {
            1 => Some(wgpu::Face::Back),
            2 => Some(wgpu::Face::Front),
            _ => None,
        }
    }

    /// Map a `SceGxmDepthFunc` to its wgpu equivalent. The enum is a SHIFTED field (vitasdk
    /// `gxm.h` spaces the values 0x00400000 apart), which is how it is stored in the sticky
    /// render state, so it is normalised back to 0..7 here.
    fn gxm_depth_func(f: u32) -> wgpu::CompareFunction {
        use wgpu::CompareFunction as C;
        match (f >> 22) & 0x7 {
            0 => C::Never,
            1 => C::Less,
            2 => C::Equal,
            3 => C::LessEqual,
            4 => C::Greater,
            5 => C::NotEqual,
            6 => C::GreaterEqual,
            _ => C::Always,
        }
    }

    /// Map a `SceGxmBlendFactor` to its wgpu equivalent. The enum order is the vitasdk
    /// `gxm.h` one; `SRC_ALPHA_SATURATE` has a wgpu counterpart and `DST_ALPHA_SATURATE`
    /// (which wgpu has no factor for) is reported and treated as `One` rather than silently
    /// becoming something else.
    fn gxm_blend_factor(f: u8) -> wgpu::BlendFactor {
        use wgpu::BlendFactor as F;
        match f {
            0 => F::Zero,
            1 => F::One,
            2 => F::Src,
            3 => F::OneMinusSrc,
            4 => F::SrcAlpha,
            5 => F::OneMinusSrcAlpha,
            6 => F::Dst,
            7 => F::OneMinusDst,
            8 => F::DstAlpha,
            9 => F::OneMinusDstAlpha,
            10 => F::SrcAlphaSaturated,
            _ => {
                report_unmapped_blend("DST_ALPHA_SATURATE blend factor");
                F::One
            }
        }
    }

    /// Map a `SceGxmBlendFunc` pair (func, src, dst) to a wgpu blend component. `MIN`/`MAX`
    /// ignore the factors, exactly as the hardware does.
    fn gxm_blend_component(func: u8, src: u8, dst: u8) -> wgpu::BlendComponent {
        use wgpu::BlendOperation as O;
        let operation = match func {
            2 => O::Subtract,
            3 => O::ReverseSubtract,
            4 => O::Min,
            5 => O::Max,
            _ => O::Add,
        };
        // MIN and MAX take the operands unscaled.
        let (src, dst) = if matches!(func, 4 | 5) { (1, 1) } else { (src, dst) };
        wgpu::BlendComponent {
            src_factor: gxm_blend_factor(src),
            dst_factor: gxm_blend_factor(dst),
            operation,
        }
    }

    /// The wgpu blend state for a draw's captured `SceGxmBlendInfo`, or `None` when the
    /// program does not blend at all (both funcs `NONE`), which is a REPLACE.
    fn gxm_blend_state(b: [u8; 7]) -> Option<wgpu::BlendState> {
        let [_mask, color_func, alpha_func, color_src, color_dst, alpha_src, alpha_dst] = b;
        if color_func == 0 && alpha_func == 0 {
            return Some(wgpu::BlendState::REPLACE);
        }
        Some(wgpu::BlendState {
            color: gxm_blend_component(color_func, color_src, color_dst),
            alpha: gxm_blend_component(alpha_func, alpha_src, alpha_dst),
        })
    }

    /// The wgpu write mask for a `SceGxmColorMask` (bit 0 R, 1 G, 2 B, 3 A).
    fn gxm_color_mask(mask: u8) -> wgpu::ColorWrites {
        let mut w = wgpu::ColorWrites::empty();
        for (bit, flag) in [
            (0, wgpu::ColorWrites::RED),
            (1, wgpu::ColorWrites::GREEN),
            (2, wgpu::ColorWrites::BLUE),
            (3, wgpu::ColorWrites::ALPHA),
        ] {
            if mask & (1 << bit) != 0 {
                w |= flag;
            }
        }
        w
    }

    /// Report - once per render target - that `VITASLOP_GXP_NEGW=force` corrected this pass's
    /// clip `w` without measuring anything.
    fn report_forced_negw(target: u32) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(target) {
            report!(
                "gxp clip: pass into {target:#010x}: VITASLOP_GXP_NEGW=force - correcting the \
                 clip w sign with NO measurement behind it. This is a diagnostic; the frame it \
                 produces is not evidence of anything except what the correction does."
            );
        }
    }

    /// Report - once per shader pair - that the guest's own `SceGxmBlendInfo` gives this pair a
    /// colour write mask of ZERO, so every draw using it writes no colour channel.
    ///
    /// This is not a fallback and not an approximation: it is the guest's state, faithfully
    /// applied, and a depth-only prepass is exactly what it looks like. It is reported because
    /// the RESULT on screen - a pass that renders nothing - is identical to a draw whose
    /// geometry, textures or transform we got wrong, and this is the only one of those that is
    /// correct. Knowing which costs one line here and a whole investigation otherwise.
    fn report_zero_color_mask(key: u64) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(key) {
            report!(
                "gxp pair {key:016x}: the guest's blend info gives this pair a colour write mask \
                 of 0 - it writes NO colour channel, so its draws affect depth only. That is the \
                 guest's own state, not a fallback."
            );
        }
    }

    /// Report - once per case - a GXM blend value with no exact wgpu equivalent, so the
    /// substitution is visible rather than silently changing what a draw composites like.
    fn report_unmapped_blend(what: &str) {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
        if seen.insert(what.to_string()) {
            report_warn!("gxm blend: {what} has no wgpu equivalent - substituting ONE, which is an approximation");
        }
    }

    fn build_gxp_pipeline(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        // Samples per pixel of the attachments this pipeline will be bound against. A
        // pipeline's multisample state must match its pass exactly, so this is part of the
        // cache key - see `GxpPrepared::samples`.
        samples: u32,
        // The guest's `SceGxmCullMode` for this draw. Part of the cache key because it is baked
        // into the pipeline's primitive state, and a title sets it PER DRAW.
        cull: u32,
        gxp: &GxpRecompile,
        key: u64,
        zfix: ZFix,
        yflip: bool,
        solid: bool,
        nodepth: bool,
        noblend: bool,
        // Compiled modules by pair - see `GxpLive::modules`.
        modules: &mut HashMap<u64, wgpu::ShaderModule>,
    ) -> Option<GxpPipeline> {
        let debug = std::env::var_os("VITASLOP_GXP_DEBUG").is_some();
        report_branches(key, gxp);
        let linked = match vitaslop_gxp_shader::link_programs(&gxp.vprog, &gxp.fprog) {
            Ok(l) => l,
            Err(e) => {
                report_fallback(key, &format!("link failed: {e}"));
                return None;
            }
        };

        // Vertex layout: each linked attribute (@location L, base lane B) is fed by the guest
        // stream attribute whose reg_index == B. We repack it to tightly-packed `Float32xN`
        // (converting F16/U8N/etc. on the CPU) so wgpu's vertex-format gaps (no Float16x3, no
        // Unorm8x3, ...) never block a draw, and the shader (which reads f32) gets exact values.
        let mut wattrs: Vec<wgpu::VertexAttribute> = Vec::with_capacity(linked.vertex_bindings.attributes.len());
        let mut repack: Vec<RepackAttr> = Vec::with_capacity(linked.vertex_bindings.attributes.len());
        let mut packed_offset: u32 = 0;
        for a in &linked.vertex_bindings.attributes {
            let ga: &GxpAttr = match gxp.attributes.iter().find(|g| g.reg_index as u32 == a.base_lane) {
                Some(g) => g,
                None => {
                    report_fallback(
                        key,
                        &format!(
                            "no guest attribute for linked @location {} base_lane {} (guest reg_indices {:?})",
                            a.location,
                            a.base_lane,
                            gxp.attributes.iter().map(|g| g.reg_index).collect::<Vec<_>>()
                        ),
                    );
                    return None;
                }
            };
            let comps = ga.components.clamp(1, 4);
            // The SHADER's declared width, which is what its PA loads read. When the guest binds
            // FEWER components than that, the surplus registers are fed a fill - and which fill
            // is a question about the guest's vertex fetch that nothing here had ever asked.
            // Widening the packed slot to the declared width and writing the fill ourselves makes
            // it an A/B instead of whatever the graphics API happens to do
            // (`VITASLOP_GXP_ATTR_FILL`; the default reproduces WebGPU's (0,0,0,1) exactly).
            let slots = (a.components as u8).clamp(comps, 4);
            let format = match slots {
                1 => wgpu::VertexFormat::Float32,
                2 => wgpu::VertexFormat::Float32x2,
                3 => wgpu::VertexFormat::Float32x3,
                _ => wgpu::VertexFormat::Float32x4,
            };
            wattrs.push(wgpu::VertexAttribute { format, offset: packed_offset as u64, shader_location: a.location });
            repack.push(RepackAttr {
                guest_offset: ga.offset as u32,
                gxm_format: ga.gxm_format,
                components: comps,
                slots,
                packed_offset,
            });
            packed_offset += slots as u32 * 4;
        }
        let packed_stride = packed_offset.max(4);

        // Decisive diagnostic (`VITASLOP_GXP_INTERP`): run the recompiled vertex shader on the
        // CPU interpreter with the real captured SA + the first vertex's attributes, and print
        // the clip output o[0..3]. If the recompiler math + uniforms are right, this is a
        // sensible clip position; if it is off-screen/NaN, the problem is upstream of the GPU.
        if std::env::var_os("VITASLOP_GXP_INTERP").is_some() {
            if let Ok(vrc) = vitaslop_gxp_shader::recompile_vertex(&gxp.vprog) {
                let mut regs = vitaslop_gxp_shader::interp::RegFile::with_lanes(512);
                for (k, c) in gxp.vert_sa.chunks_exact(4).enumerate() {
                    if k < regs.sa.len() {
                        regs.sa[k] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    }
                }
                for a in &linked.vertex_bindings.attributes {
                    if let Some(ga) = gxp.attributes.iter().find(|g| g.reg_index as u32 == a.base_lane) {
                        for c in 0..ga.components as usize {
                            let lane = a.base_lane as usize + c;
                            if lane < regs.pa.len() {
                                regs.pa[lane] = read_attr_component(&gxp.vertices, ga.offset as usize, ga.gxm_format, c);
                            }
                        }
                    }
                }
                // The SECONDARY program runs before the primary and its whole purpose is to
                // leave values in SA registers the primary reads, so an interpretation that
                // skips it is reading uniforms the real module has already overwritten. Its
                // container literals go in too, as raw bit patterns, exactly as the module
                // emits them.
                if let Ok(program) = vitaslop_gxp_shader::Program::parse(&gxp.vprog) {
                    for &(reg, value) in &program.literals {
                        if let Some(slot) = regs.sa.get_mut(reg as usize) {
                            *slot = f32::from_bits(value);
                        }
                    }
                    let secondary = vitaslop_gxp_shader::usse::decode_secondary_shader(&program);
                    if let Err(e) = vitaslop_gxp_shader::interp::run(&secondary, &mut regs) {
                        report!("gxp interp: secondary program failed: {e}");
                    }
                }
                // Where the WHOLE mesh lands, not just its first vertex. Vertex 0 of a large
                // world mesh is routinely behind the camera, so its `w<0` says nothing about
                // whether the draw covers any pixels - and "this draw is missing" is exactly
                // the question. Counting the vertices in front of the eye and taking the NDC
                // bounding box over them answers it outright.
                let saved_sa = regs.sa.clone();
                let stride = gxp.vertex_stride.max(1) as usize;
                let nverts = (gxp.vertices.len() / stride).min(4096);
                let (mut in_front, mut lo, mut hi) = (0usize, [f32::MAX; 2], [f32::MIN; 2]);
                for v in 0..nverts {
                    let mut vregs = regs.clone();
                    vregs.sa.clone_from(&saved_sa);
                    for a in &linked.vertex_bindings.attributes {
                        if let Some(ga) = gxp.attributes.iter().find(|g| g.reg_index as u32 == a.base_lane) {
                            let base = v * stride + ga.offset as usize;
                            for c in 0..ga.components as usize {
                                let lane = a.base_lane as usize + c;
                                if lane < vregs.pa.len() {
                                    vregs.pa[lane] = read_attr_component(&gxp.vertices, base, ga.gxm_format, c);
                                }
                            }
                        }
                    }
                    if vitaslop_gxp_shader::interp::run(&vrc.shader, &mut vregs).is_err() {
                        break;
                    }
                    let w = vregs.o[3];
                    if w > 0.0 {
                        in_front += 1;
                        lo = [lo[0].min(vregs.o[0] / w), lo[1].min(vregs.o[1] / w)];
                        hi = [hi[0].max(vregs.o[0] / w), hi[1].max(vregs.o[1] / w)];
                    }
                }
                report!(
                    "gxp interp: key {key:x} {in_front}/{nverts} vertices have w>0, ndc bbox x[{:.3},{:.3}] y[{:.3},{:.3}]",
                    lo[0], hi[0], lo[1], hi[1]
                );
                match vitaslop_gxp_shader::interp::run_watching_for_nan(&vrc.shader, &mut regs) {
                    Ok(site) => {
                        let w = regs.o[3];
                        let ndc = if w.abs() > 1e-6 { [regs.o[0] / w, regs.o[1] / w, regs.o[2] / w] } else { [0.0; 3] };
                        report!("gxp interp: o={:?} ndc={:?} viewport(xo,xs,yo,ys,zo,zs)={:?}", &regs.o[0..4], ndc, gxp.viewport);
                        // A NaN clip position draws NOTHING, which looks exactly like a black
                        // shader; naming the instruction that produced it is the whole point.
                        if let Some(s) = site {
                            report!(
                                "gxp interp: FIRST non-finite value at instruction #{} {} -> {} channel {} = {}\n  sources: {}",
                                s.index, s.op, s.dest, s.channel, s.value, s.sources.join("  ")
                            );
                        }
                    }
                    Err(e) => report!("gxp interp: run failed: {e}"),
                }
            }
        }

        // Through `KeySpec` so the knob takes `all` / a key list / `!<key list>`. The exclusion
        // form is the one that matters here: see `KeySpec::resolve`.
        let keycolor = {
            use std::sync::OnceLock;
            static SPEC: OnceLock<KeySpec> = OnceLock::new();
            SPEC.get_or_init(|| KeySpec::resolve("VITASLOP_GXP_KEYCOLOR")).wants(key).then_some(key)
        };
        let wgsl = match inject_clip_fixup(&linked.wgsl, zfix, yflip, solid, keycolor) {
            Some(w) => w,
            None => {
                report_fallback(
                    key,
                    "no `out.position` assignment to wrap with the clip fixup - refusing to \
                     render without the depth remap",
                );
                return None;
            }
        };
        // `VITASLOP_GXP_WGSL_DIR=<dir>`: write each pair's linked WGSL to `<key>.wgsl`.
        //
        // When a recompiled draw comes out wrong, the question is what the guest's shader was
        // actually translated INTO, and until now the only way to see that was to add a print
        // and rebuild. The translation is the artefact worth reading - it names the samplers,
        // the varyings and the clip position in one place.
        if let Ok(dir) = std::env::var("VITASLOP_GXP_WGSL_DIR") {
            let dir = std::path::Path::new(&dir);
            if let Err(e) = std::fs::create_dir_all(dir)
                .and_then(|()| std::fs::write(dir.join(format!("{key:016x}.wgsl")), &wgsl))
            {
                report_warn!("gxp: cannot write WGSL for pair {key:016x}: {e}");
            }
        }
        // >>> TIMED, because "the first frame of a race costs 2.5 seconds" is not actionable
        // until it says WHICH half. A WGSL compile and a pipeline create are different costs
        // with different fixes: the compile depends on the shader pair alone and is now done
        // when `sceGxmShaderPatcherCreateFragmentProgram` names the pair (see
        // `GxmRenderer::precompile_pairs`), while the pipeline also needs the draw's blend,
        // depth, cull, format and sample count and cannot be. This lookup is what a precompiled
        // pair HITS; a miss compiles here exactly as it always did.
        let t_module = Stopwatch::start();
        let mkey = match keycolor {
            Some(_) => key,
            None => GxpLive::module_key(&gxp.vprog, &gxp.fprog),
        };
        let module = modules
            .entry(mkey)
            .or_insert_with(|| {
                // >>> LABELLED WITH THE PAIR, because the DEVICE's own error message is the
                // only diagnosis available on a phone.
                //
                // A device that rejects a module reports it as `[Invalid ShaderModule
                // "<label>"]`, and a constant label names nothing: a real report from an
                // Android PowerVR device read `[Invalid RenderPipeline "gxp"] is invalid due
                // to a previous error` thirty-two times, with no way to tell WHICH of the
                // title's pairs was bad or to look it up in the shader corpus. The module key
                // is the same hex a `gxp pair <key>` line prints, so the label joins the
                // device's complaint to the dump tooling.
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("gxp-linked:{mkey:016x}")),
                    source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                })
            })
            .clone();
        super::add_build_ms(&super::PIPE_MODULE_US, t_module.ms());

        // group0 vertex uniform, group1 fragment uniform, group2 samplers (empty where unused).
        //
        // Both are addressed by DYNAMIC OFFSET into one arena per pass, so a pass needs one
        // bind group per shader pair rather than one per draw - the same shape the
        // fixed-function path has always used for its per-draw uniform. `min_binding_size`
        // pins the shader-visible window to exactly this stage's SA block, so the arena's
        // 256-byte offset alignment cannot let a draw read into the next draw's uniforms.
        let uniform_entry = |vis: wgpu::ShaderStages, bytes: u64| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(bytes),
            },
            count: None,
        };
        let vsa_lanes = linked.vertex_bindings.sa_lane_count;
        let fsa_lanes = linked.fragment_bindings.sa_lane_count;
        let sa_bytes = |lanes: u32| (lanes.div_ceil(4) as u64) * 16;
        let (vsa_bytes, fsa_bytes) = (sa_bytes(vsa_lanes), sa_bytes(fsa_lanes));
        let mut g0_entries: Vec<wgpu::BindGroupLayoutEntry> =
            if vsa_lanes > 0 { vec![uniform_entry(wgpu::ShaderStages::VERTEX, vsa_bytes)] } else { vec![] };
        // The vertex stage's guest-memory window at group 0 binding 1, when its program
        // loads memory: the same dynamic-offset arena discipline as the SA blocks, with
        // `min_binding_size` pinning the shader-visible extent to exactly the declared
        // window (header vec4 included).
        let mem_bind_bytes = linked
            .vertex_bindings
            .mem_window
            .map(|w| w.vec4_count() as u64 * 16)
            .unwrap_or(0);
        if mem_bind_bytes > 0 {
            g0_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(mem_bind_bytes),
                },
                count: None,
            });
        }
        let g1_entries: Vec<wgpu::BindGroupLayoutEntry> =
            if fsa_lanes > 0 { vec![uniform_entry(wgpu::ShaderStages::FRAGMENT, fsa_bytes)] } else { vec![] };
        let mut g2_entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
        let mut samplers: Vec<(u8, SamplerDim)> = Vec::new();
        for (i, b) in linked.fragment_bindings.samplers.iter().enumerate() {
            // Mirrors `TexBinding::wgsl_type` exactly - the layout and the shader must agree.
            let dim = match (b.coords >= 3, b.cube) {
                (true, true) => SamplerDim::Cube,
                (true, false) => SamplerDim::Three,
                _ => SamplerDim::Two,
            };
            // The binding plan already names the GXM texture unit the guest bound to: the SMP
            // sampler operand is resolved through the container's texture-control table when the
            // instruction stream is decoded, and a prefetched sample names its unit outright.
            let gxm_unit = b.unit as u32;
            g2_entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32 * 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: dim.view_dimension(),
                    multisampled: false,
                },
                count: None,
            });
            g2_entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32 * 2 + 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
            if debug {
                report!("gxp build: gxm texture unit {gxm_unit} (coords {}, {dim:?})", b.coords);
            }
            samplers.push((gxm_unit as u8, dim));
        }
        // group3 carries the scene depth range the injected clip fixup remaps through - one
        // vec4 the renderer refills per frame, so the pipeline stays cached by shader identity.
        // BOTH stages read it: the vertex stage to remap clip depth, the fragment stage to
        // reconstruct the guest's window POSITION (see `link::GXP_DEPTH_DECL`).
        // The VERTEX stage's samplers go in the SAME group, after the fragment ones: a device
        // guarantees only four bind groups and the other three are spoken for. A vertex program
        // that fetches a texture builds its geometry from it, so these are not decoration -
        // without them the draw has no mesh. Their unit numbering is independent of the fragment
        // stage's, which is why they keep separate names (`vt{u}`/`vs{u}`).
        let vsampler_base = g2_entries.len() as u32;
        let mut vertex_samplers: Vec<(u8, SamplerDim)> = Vec::new();
        for (i, b) in linked.vertex_bindings.samplers.iter().enumerate() {
            let dim = match (b.coords >= 3, b.cube) {
                (true, true) => SamplerDim::Cube,
                (true, false) => SamplerDim::Three,
                _ => SamplerDim::Two,
            };
            g2_entries.push(wgpu::BindGroupLayoutEntry {
                binding: vsampler_base + i as u32 * 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: dim.view_dimension(),
                    multisampled: false,
                },
                count: None,
            });
            g2_entries.push(wgpu::BindGroupLayoutEntry {
                binding: vsampler_base + i as u32 * 2 + 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
            if debug {
                report!("gxp build: VERTEX gxm texture unit {} (coords {}, {dim:?})", b.unit, b.coords);
            }
            vertex_samplers.push((b.unit, dim));
        }
        let layouts = [
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gxp-g0"), entries: &g0_entries }),
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gxp-g1"), entries: &g1_entries }),
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("gxp-g2"), entries: &g2_entries }),
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gxp-g3"),
                // The pass depth block is per SCENE, not per draw: one cached bind group, no
                // dynamic offset.
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            }),
        ];
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gxp-pl"),
            bind_group_layouts: &[Some(&layouts[0]), Some(&layouts[1]), Some(&layouts[2]), Some(&layouts[3])],
            immediate_size: 0,
        });

        let vbuffers: Vec<Option<wgpu::VertexBufferLayout>> = if wattrs.is_empty() {
            vec![]
        } else {
            vec![Some(wgpu::VertexBufferLayout { array_stride: packed_stride as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &wattrs })]
        };

        // The guest's own blend equation, from the `SceGxmBlendInfo` its fragment program was
        // created with. This is state, not a guess: GXM has no runtime blend setter, so the
        // old "opaque draws REPLACE, the rest src-over" reading was the renderer inventing a
        // mode. It got a title's whole world composite wrong - that draw does NOT blend, and
        // forcing src-over made it invisible the moment its shader output alpha 0.
        let guest_blend = gxm_blend_state(gxp.blend_state);
        // The colour channels the guest asked for - AND ONLY IF it left the fragment program
        // enabled. `sceGxmSetFrontFragmentProgramEnable(DISABLED)` makes the draw depth- and
        // stencil-only on hardware, whatever its shader computes; see
        // `GxpRecompile::fragment_program_enabled`.
        let write_mask = match gxp.fragment_program_enabled {
            true => gxm_color_mask(gxp.blend_state[0]),
            false => wgpu::ColorWrites::empty(),
        };
        // A colour mask of ZERO is a legal and meaningful GXM state - a depth-only prepass
        // writes no colour - but on screen it is indistinguishable from a draw we got wrong,
        // and it is the one pipeline setting that can make a perfectly recompiled, correctly
        // bound, correctly transformed draw leave no mark at all. Say it out loud, once.
        if write_mask.is_empty() && gxp.fragment_program_enabled {
            report_zero_color_mask(key);
        }
        // Depth comes from the guest's own captured state: `SceGxmDepthFunc` and the depth-write
        // enable, exactly as the hardware would apply them.
        //
        // It used to be a HEURISTIC - "an MVP-space draw that writes depth is opaque, everything
        // else is a 2D overlay" - decided in `render.rs` off the FIXED-FUNCTION reflection, i.e.
        // off whether we could recognise a model-view-projection matrix in the guest's uniform
        // buffer. A recompiled draw does not use that matrix at all, so the heuristic was
        // answering a question about a transform the pipeline never runs. On one title it
        // classified the engine's 2D primitive-render path as an overlay and gave a FULLSCREEN
        // black triangle `Always` + no depth write: it painted over the finished world every
        // frame, and no depth knob could move it because the draw was not being depth-tested at
        // all. Three different clip-depth remaps produced bit-identical black frames, which is
        // the fingerprint of depth state that is not the guest's.
        //
        // Switching this over was tried once before and MEASURED to be worse (a race's world
        // pass lost its track surface). That measurement was real, and it was against
        // `ZFix::Range`, which writes the scene's normalised `-1/w` - NOT the guest's depth. The
        // guest's own `LESS_EQUAL` cannot mean anything against a quantity the guest never
        // computed. The two changes only work together: see `ZFix::Off`, now the default, which
        // passes the guest's clip z through untouched so the buffer holds the guest's own
        // window depth.
        let (mut blend, mut depth_write, mut depth_compare) =
            (guest_blend, gxp.depth_write, gxm_depth_func(gxp.depth_func));
        if solid || noblend {
            // Diagnostic: REPLACE, so the fragment's colour reaches the target whatever its
            // alpha is. `solid` additionally replaces the shading and drops the depth test;
            // `noblend` changes ONLY the blend, which is what separates "this surface shades
            // black" from "this surface is composited away". A draw whose shader writes
            // alpha 0 under a src-alpha blend contributes exactly nothing, and the finished
            // frame cannot tell that apart from a fragment that computed black.
            blend = Some(wgpu::BlendState::REPLACE);
        }
        // A zero write mask is the same invisibility by a different route, so the diagnostic
        // has to lift it too or it only answers half the question.
        let write_mask = if solid || noblend { wgpu::ColorWrites::ALL } else { write_mask };
        if solid || nodepth {
            // Both diagnostics drop the depth test; `solid` additionally replaces the
            // shading, which is exactly the difference between them.
            depth_write = false;
            depth_compare = wgpu::CompareFunction::Always;
        }
        report_gxp_pipeline_state(key, gxp, depth_write, depth_compare, blend, write_mask);
        let t_pipe = Stopwatch::start();
        let make = || {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                // Named by PAIR for the same reason the module above is - see there. This is
                // the label that appeared thirty-two times as a bare `"gxp"` in a device
                // failure report, naming nothing.
                label: Some(&format!("gxp:{key:016x}")),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_main"), buffers: &vbuffers, compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState { format: color_format, blend, write_mask })],
                    compilation_options: Default::default(),
                }),
                // The guest's own `SceGxmCullMode`, mapped by `gxm_cull_face` (which is where
                // the winding sense is derived and why). It used to be `None` unconditionally -
                // "draw both windings and rely on the depth test" - which is wrong twice over:
                // the software rasteriser HAS culled since it was written, so the two halves of
                // this renderer disagreed about which faces exist, and a back face that the
                // depth test happens to reject still costs its rasterisation. `solid` drops it
                // for the same reason it drops the depth test: that diagnostic exists to show
                // every fragment a pair produces.
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: if solid { None } else { gxm_cull_face(cull) },
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(depth_compare),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                // MSAA, not alpha-to-coverage and not per-sample shading: the fragment stage
                // runs ONCE per pixel and every covered sample takes that result, which is
                // what `SceGxmMultisampleMode` asks for. `mask: !0` leaves coverage entirely
                // to the rasterizer, as the hardware does with no coverage override set.
                multisample: wgpu::MultisampleState {
                    count: samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
        };

        let pipeline = make();
        super::add_build_ms(&super::PIPE_CREATE_US, t_pipe.ms());
        Some(GxpPipeline {
            pipeline,
            layouts,
            vsa_lanes,
            fsa_lanes,
            mem_bind_bytes: mem_bind_bytes as u32,
            samplers,
            vertex_samplers,
            repack,
            packed_stride,
        })
    }

    impl GxmRenderer {
        /// Build both pipelines for a `color_format` render target. `queue` is used
        /// once here to upload the 1x1 white fallback texel.
        pub fn new(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            color_format: wgpu::TextureFormat,
        ) -> Self {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gxm"),
                source: wgpu::ShaderSource::Wgsl(GXM_SHADER.into()),
            });

            // Group 0: the per-draw uniform, addressed by dynamic offset into one shared
            // buffer (so a whole frame needs a single uniform bind group). `min_binding_size`
            // pins the shader-visible window to the `U` struct so validation is tight.
            let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gxm-uniform"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(UNIFORM_BYTES),
                    },
                    count: None,
                }],
            });

            // Group 1: the sampled texture + its sampler. `filterable` / `Filtering` so a
            // LINEAR-magnified texture (UI/font atlas) can bilinear-sample on the GPU; a
            // Nearest sampler is still valid in a filtering slot, so the same layout serves
            // both filter modes (the sampler is chosen per draw, see `bind_for`).
            let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gxm-texture"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gxm-layout"),
                bind_group_layouts: &[Some(&uniform_layout), Some(&texture_layout)],
                immediate_size: 0,
            });

            // Canonical vertex: pos float32x3 @0, uv float32x2 @12, color unorm8x4 @20,
            // world-space normal float32x3 @24 (for the opaque lighting term).
            let vertex_layout = wgpu::VertexBufferLayout {
                array_stride: GXM_VERTEX_STRIDE,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 12,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Unorm8x4,
                        offset: 20,
                        shader_location: 2,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 24,
                        shader_location: 3,
                    },
                ],
            };

            // A pipeline builder closure differing in blend, depth AND fragment entry
            // point: the opaque (3D) path depth-tests and writes, replaces, and runs
            // `fs_opaque` (albedo texel + exposure/Reinhard); the 2D path disables the
            // depth test, straight-alpha src-over blends in submission order, and runs
            // `fs_blend` (vertex_color * texel modulate) - the exact two modes the
            // software rasterizer switches between per draw.
            let make = |opaque: bool, target_format: wgpu::TextureFormat, samples: u32| {
                let (blend, depth_write, depth_compare, fs) = if opaque {
                    (
                        Some(wgpu::BlendState::REPLACE),
                        true,
                        // LessEqual is GXM's default depth func and what this title's opaque
                        // 3D draws set (SCE_GXM_DEPTH_FUNC_LESS_EQUAL); the software oracle
                        // compares the same way (`depth_passes`), so a coincident later face
                        // ties and repaints identically on both paths. The CPU builder
                        // already culled back faces, so no pipeline cull state is needed.
                        wgpu::CompareFunction::LessEqual,
                        "fs_opaque",
                    )
                } else {
                    (
                        Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        false,
                        wgpu::CompareFunction::Always,
                        "fs_blend",
                    )
                };
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(if opaque { "gxm-opaque" } else { "gxm-blend" }),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        buffers: &[Some(vertex_layout.clone())],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(fs),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: target_format,
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        // No culling: match the software rasterizer, which draws both
                        // windings.
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(depth_write),
                        depth_compare: Some(depth_compare),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: samples,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview_mask: None,
                    cache: None,
                })
            };

            let opaque = make(true, color_format, 1);
            let blend = make(false, color_format, 1);
            // The same two pipelines for a MULTISAMPLED pass. A fixed-function draw and a
            // recompiled one share one render pass, so when the guest's render target asks for
            // multisampling BOTH have to be built for it - a pipeline may only be bound in a
            // pass whose sample count matches. Built eagerly for the same reason the sRGB pair
            // below is: it is two pipelines, not a family (`gxm_sample_count` yields 1 or
            // MSAA_SAMPLES and nothing else), and the alternative is discovering mid-frame that
            // a fixed-function draw landed on a multisampled target with nothing to draw it
            // with.
            let opaque_ms = make(true, color_format, MSAA_SAMPLES);
            let blend_ms = make(false, color_format, MSAA_SAMPLES);
            // The same two pipelines against the sRGB view of the same texture, for a pass
            // whose colour surface the guest put in GAMMA-CORRECT mode. Built eagerly because
            // it is two pipelines, not a family: the alternative is discovering mid-frame that
            // a fixed-function draw landed on a gamma surface and having nothing to draw it
            // with. No multisampled variant: a gamma target is refused multisampling (see
            // `report_multisample_refused`), so the pair would never be bound.
            let srgb = srgb_twin(color_format)
                .map(|f| (make(true, f, 1), make(false, f, 1)));

            let sampler = |linear: bool| {
                let f = if linear {
                    wgpu::FilterMode::Linear
                } else {
                    wgpu::FilterMode::Nearest
                };
                device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("gxm-sampler"),
                    address_mode_u: wgpu::AddressMode::Repeat,
                    address_mode_v: wgpu::AddressMode::Repeat,
                    address_mode_w: wgpu::AddressMode::Repeat,
                    mag_filter: f,
                    min_filter: f,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    ..Default::default()
                })
            };
            let sampler_point = sampler(false);
            let sampler_linear = sampler(true);

            // A 1x1 opaque-white texel: the fallback an untextured draw binds so the one
            // shader path serves both textured and vertex-color draws.
            let white_tex = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("gxm-white"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &[255, 255, 255, 255],
            );
            let white = white_tex.create_view(&Default::default());
            let white_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gxm-white-bind"),
                layout: &texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&white),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler_point),
                    },
                ],
            });

            let align = device.limits().min_uniform_buffer_offset_alignment as u64;
            let uniform_stride = align_up(UNIFORM_BYTES, align.max(1));

            // The supersample resolve: a fullscreen triangle that box-averages each
            // `scale x scale` block of the offscreen colour target into one output pixel, the
            // exact integer box `Framebuffer::downsampled` applies on the software oracle so the
            // two AA'd frames stay in lockstep. `textureLoad` (no sampler) does the exact taps.
            let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gxm-resolve"),
                source: wgpu::ShaderSource::Wgsl(RESOLVE_SHADER.into()),
            });
            let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gxm-resolve-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(16),
                        },
                        count: None,
                    },
                ],
            });
            let resolve_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gxm-resolve-pl"),
                bind_group_layouts: &[Some(&resolve_layout)],
                immediate_size: 0,
            });
            let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gxm-resolve-pipe"),
                layout: Some(&resolve_pl),
                vertex: wgpu::VertexState {
                    module: &resolve_shader,
                    entry_point: Some("vres"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &resolve_shader,
                    entry_point: Some("fres"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });
            let resolve_scale_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gxm-resolve-scale"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // The guest-depth conversion pass. Built unconditionally (it is one small
            // pipeline) but only ever ENCODED for a target whose depth some pass samples.
            let gxm_depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gxm-depth-convert"),
                source: wgpu::ShaderSource::Wgsl(GXM_DEPTH_SHADER.into()),
            });
            let gxm_depth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gxm-depth-convert-layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(GXM_DEPTH_UNIFORM_BYTES),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // NON-filtering: a depth buffer is data, and this pass must return the
                        // stored texel rather than a blend of its neighbours.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });
            let gxm_depth_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gxm-depth-convert-pl"),
                bind_group_layouts: &[Some(&gxm_depth_layout)],
                immediate_size: 0,
            });
            let gxm_depth_pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("gxm-depth-convert-pipe"),
                layout: Some(&gxm_depth_pl),
                vertex: wgpu::VertexState {
                    module: &gxm_depth_shader,
                    entry_point: Some("vdep"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &gxm_depth_shader,
                    entry_point: Some("fdep"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: GXM_DEPTH_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            });
            let gxm_depth_uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gxm-depth-convert-uniform"),
                size: GXM_DEPTH_UNIFORM_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            GxmRenderer {
                srgb,
                gxm_depth_pipe,
                gxm_depth_layout,
                gxm_depth_uniform,
                opaque,
                blend,
                opaque_ms,
                blend_ms,
                uniform_layout,
                texture_layout,
                sampler_point,
                sampler_linear,
                white_bind,
                views: HashMap::default(),
                views_bytes: 0,
                tex_binds: HashMap::default(),
                vbo: None,
                ibo: None,
                ubo: None,
                ubo_bind: None,
                vbo_cap: 0,
                ibo_cap: 0,
                ubo_cap: 0,
                gxp_arenas: Vec::new(),
                gxp_arena_slot: 0,
                gxp_precompiled: 0,
                uniform_stride,
                color_format,
                ss_scale: 1,
                last_gxp_summary: None,
                last_phases: EncodePhases::default(),
                chain_phases: EncodePhases::default(),
                resolve_pipeline,
                resolve_layout,
                resolve_scale_buf,
                ss_target: None,
                rtt: HashMap::default(),
                rtt_epoch: 0,
                rtt_rendered: HashMap::default(),
                rtt_binds: HashMap::default(),
                rtt_reads_snapshot: HashSet::default(),
                rtt_depth_rendered: HashMap::default(),
                retired_buffers: Vec::new(),
                keep_depth: false,
                rtt_hits: 0,
                last_chain_shape: None,
                sampled_addrs: HashSet::default(),
                gxp: GxpLive::from_env(),
            }
        }

        /// Set the supersample factor: 1 (default) renders the scene straight into the caller's
        /// target; `scale > 1` renders it at `scale x` the target dimensions into an offscreen
        /// buffer and box-downsamples that into the caller's view on resolve. This antialiases
        /// the geometric aliasing of a heavily-tessellated distant mesh (many sub-pixel triangles
        /// per final pixel) and coincident-panel z-fighting - the vehicle-speckle a 1x render
        /// shows - and mirrors the software oracle's `VITASLOP_SSAA` / `Framebuffer::downsampled`
        /// so both paths stay in lockstep. Cost is `scale^2` fill; 2x is the quality/perf
        /// default a caller opts into. A no-op if `scale` is unchanged.
        pub fn set_supersample(&mut self, scale: u32) {
            let scale = scale.max(1);
            if scale != self.ss_scale {
                self.ss_scale = scale;
                self.ss_target = None; // force a rebuild at the new scale
            }
        }

        /// Ensure the offscreen supersample target exists at `scale * (surf_w, surf_h)`,
        /// rebuilding it if the scale or size changed. Also refreshes the scale uniform.
        fn ensure_ss_target(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, surf_w: u32, surf_h: u32) {
            let scale = self.ss_scale;
            let (w, h) = (surf_w * scale, surf_h * scale);
            let stale = self.ss_target.as_ref().map(|t| t.scale != scale || t.width != w || t.height != h).unwrap_or(true);
            if !stale {
                return;
            }
            let color = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-ss-color"),
                size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let color_view = color.create_view(&Default::default());
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-ss-depth"),
                size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let depth_view = depth.create_view(&Default::default());
            queue.write_buffer(&self.resolve_scale_buf, 0, &[scale, 0, 0, 0].iter().flat_map(|v: &u32| v.to_le_bytes()).collect::<Vec<u8>>());
            let resolve_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gxm-resolve-bind"),
                layout: &self.resolve_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&color_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: self.resolve_scale_buf.as_entire_binding() },
                ],
            });
            self.ss_target = Some(SsTarget { scale, width: w, height: h, _color: color, color_view, depth_view, resolve_bind });
        }

        /// Ensure a decoded texture is uploaded (cached by content key) and that a bind
        /// group exists for it at the requested filter. A cache hit does no GPU work.
        fn ensure_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, t: &super::GxmTexture) {
            // Bound the caches BY BYTES: on pathological churn, clear wholesale and
            // re-upload (keys are content fingerprints, so correctness is unaffected).
            // See `tex_cache_budget_bytes` for why an entry COUNT bounded nothing.
            if self.views.contains_key(&t.key) {
                enc(&ENC.tex_view_cached, 1);
            } else {
                self.views_bytes += texture_bytes(t.width, t.height);
                if self.views_bytes >= tex_cache_budget_bytes() {
                    enc(&ENC.tex_view_wholesale_clears, 1);
                    self.views.clear();
                    self.tex_binds.clear();
                    self.views_bytes = texture_bytes(t.width, t.height);
                }
            }
            if !self.views.contains_key(&t.key) {
                enc(&ENC.tex_uploaded, 1);
                enc(&ENC.tex_upload_bytes, t.rgba.len() as u64);
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("gxm-tex"),
                    size: wgpu::Extent3d {
                        width: t.width,
                        height: t.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &t.rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(t.width * 4),
                        rows_per_image: Some(t.height),
                    },
                    wgpu::Extent3d { width: t.width, height: t.height, depth_or_array_layers: 1 },
                );
                self.views.insert(t.key, tex.create_view(&Default::default()));
            }
            let bind_key = (t.key, t.filter_linear);
            if self.tex_binds.contains_key(&bind_key) {
                enc(&ENC.bind_groups_reused, 1);
            } else {
                enc(&ENC.bind_groups_built, 1);
            }
            if !self.tex_binds.contains_key(&bind_key) {
                let view = &self.views[&t.key];
                let samp = if t.filter_linear { &self.sampler_linear } else { &self.sampler_point };
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gxm-tex-bind"),
                    layout: &self.texture_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(samp),
                        },
                    ],
                });
                self.tex_binds.insert(bind_key, bind);
            }
        }

        /// Grow (or first-create) a per-frame arena buffer so it holds at least `need`
        /// bytes. Returns true if the buffer was (re)created (the caller rebinds if so).
        ///
        /// Shared with [`GxmRenderer::ensure_gxp_arena`] so both arena pools grow on the same
        /// schedule: geometrically, so a steadily-larger frame does not reallocate every time.
        fn cap_for(need: u64) -> u64 {
            need.max(4).next_power_of_two().max(4096)
        }

        fn ensure_buffer(
            device: &wgpu::Device,
            buf: &mut Option<wgpu::Buffer>,
            cap: &mut u64,
            need: u64,
            usage: wgpu::BufferUsages,
            label: &str,
        ) -> bool {
            let need = need.max(4);
            if buf.is_some() && *cap >= need {
                return false;
            }
            let new_cap = Self::cap_for(need);
            enc(&ENC.buffers_created, 1);
            *buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: new_cap,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            *cap = new_cap;
            true
        }

        /// Make slot `slot` of the recompiled path's arena pool hold this pass's vertex,
        /// index and uniform data, creating or growing its three buffers only when the data
        /// no longer fits, and uploading with `write_buffer`.
        ///
        /// The slot is addressed by PASS ORDINAL, so it is written at most once per frame and
        /// the write is queue-ordered behind the previous frame's submit. A buffer that has to
        /// GROW is handed to the graveyard rather than dropped, for the same reason every
        /// other retired buffer is: dropping a `wgpu::Buffer` on the web backend only makes it
        /// collectable, and the previous frame's commands still name it until its submit
        /// retires. See [`GxmRenderer::gxp_arenas`] and `retired_buffers`.
        #[allow(clippy::too_many_arguments)]
        fn ensure_gxp_arena(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            arenas: &mut Vec<GxpArenaSlot>,
            retired: &mut Vec<wgpu::Buffer>,
            slot: usize,
            vdata: &[u8],
            idata: &[u8],
            udata: &[u8],
        ) {
            // `write_buffer` copies whole 4-byte units, and a packed vertex stream need not
            // land on one. Padding the LENGTH is safe where padding the arena would not be:
            // every draw addresses its own byte range, so the tail past the last draw is
            // never read.
            let padded = |n: usize| (n.max(4) as u64).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
            let (vneed, ineed, uneed) = (padded(vdata.len()), padded(idata.len()), padded(udata.len()));

            while arenas.len() <= slot {
                let n = arenas.len();
                let mk = |need: u64, usage: wgpu::BufferUsages, label: &str| {
                    enc(&ENC.buffers_created, 1);
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(label),
                        size: Self::cap_for(need),
                        usage: usage | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                };
                // A slot below `slot` that is being created only to fill the gap is sized to
                // the minimum: it belongs to a pass that did not run this frame, and it will
                // grow on the frame that first uses it.
                let (v, i, u) = if n == slot { (vneed, ineed, uneed) } else { (4, 4, 4) };
                arenas.push(GxpArenaSlot {
                    vbo: mk(v, wgpu::BufferUsages::VERTEX, "gxp-vbo"),
                    ibo: mk(i, wgpu::BufferUsages::INDEX, "gxp-ibo"),
                    ubo: mk(u, wgpu::BufferUsages::UNIFORM, "gxp-ubo"),
                    vcap: Self::cap_for(v),
                    icap: Self::cap_for(i),
                    ucap: Self::cap_for(u),
                    generation: 0,
                });
            }

            let a = &mut arenas[slot];
            let mut grow = |buf: &mut wgpu::Buffer,
                            cap: &mut u64,
                            need: u64,
                            usage: wgpu::BufferUsages,
                            label: &str,
                            retired: &mut Vec<wgpu::Buffer>|
             -> bool {
                if *cap >= need {
                    return false;
                }
                enc(&ENC.buffers_created, 1);
                let new = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: Self::cap_for(need),
                    usage: usage | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                *cap = Self::cap_for(need);
                retired.push(std::mem::replace(buf, new));
                true
            };
            grow(&mut a.vbo, &mut a.vcap, vneed, wgpu::BufferUsages::VERTEX, "gxp-vbo", retired);
            grow(&mut a.ibo, &mut a.icap, ineed, wgpu::BufferUsages::INDEX, "gxp-ibo", retired);
            // Only the UNIFORM arena's identity is baked into a bind group, so only its
            // re-creation has to invalidate one.
            if grow(&mut a.ubo, &mut a.ucap, uneed, wgpu::BufferUsages::UNIFORM, "gxp-ubo", retired) {
                a.generation += 1;
            }

            // One `write_buffer` per arena, over the padded length. An empty arena still gets
            // its 4 zero bytes: a pass whose every stage declares no uniforms produces one,
            // and a zero-length buffer is not legal.
            let mut write = |buf: &wgpu::Buffer, data: &[u8], need: u64| {
                enc(&ENC.buffer_bytes, need);
                if data.len() as u64 == need {
                    queue.write_buffer(buf, 0, data);
                } else {
                    let mut padded = vec![0u8; need as usize];
                    padded[..data.len()].copy_from_slice(data);
                    queue.write_buffer(buf, 0, &padded);
                }
            };
            write(&a.vbo, vdata, vneed);
            write(&a.ibo, idata, ineed);
            write(&a.ubo, udata, uneed);
        }

        /// The group-1 bind group for an item (the shared white fallback, or a cached
        /// texture bind ensured earlier this frame).
        fn bind_for(&self, key: BindKey) -> &wgpu::BindGroup {
            match key {
                BindKey::White => &self.white_bind,
                BindKey::Tex(k, l) => &self.tex_binds[&(k, l)],
                BindKey::Rtt(a, l, s) => &self.rtt_binds[&(a, l, s)],
            }
        }

        /// The offscreen target for the colour surface at guest address `addr`, created (or
        /// re-created at a new size) on demand.
        ///
        /// `sample_depth` says some later pass in this frame reads this target's DEPTH, which
        /// costs the depth texture a `TEXTURE_BINDING` usage and a converted R32Float
        /// companion. It is off for every target nothing samples, so an ordinary pass pays
        /// nothing for the feature.
        fn ensure_rtt(
            &mut self,
            device: &wgpu::Device,
            addr: u32,
            width: u32,
            height: u32,
            sample_depth: bool,
            // The render target's `SceGxmMultisampleMode` - see `RttTarget::multisample`.
            multisample: u32,
            // The colour surface asks for `SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE`, i.e. it
            // stores the RESOLVE. Not part of the sample-count decision - that is the render
            // target's business - but checked against it, see the report below.
            msaa_downscale: bool,
            // The colour surface is in GAMMA-CORRECT mode. Passed in rather than decided at
            // encode time so a gamma target never ALLOCATES multisampled attachments it is
            // then refused the use of - which is both wasted memory and a log that says
            // "granted" and "REFUSED" about the same target in the same frame.
            gamma: bool,
        ) {
            let want_samples =
                if no_multisample() || gamma { 1 } else { gxm_sample_count(multisample) };
            let stale = match self.rtt.get(&addr) {
                // Gaining a depth reader is as much a reason to rebuild as a resize: the depth
                // texture it already has was created without `TEXTURE_BINDING` and cannot be
                // sampled. Gaining, losing or CHANGING the sample count is the same kind of
                // event - the attachments themselves are a different texture.
                Some(t) => {
                    t.width != width
                        || t.height != height
                        || (sample_depth && t.gxm_depth.is_none())
                        || t.msaa.as_ref().map(|m| m.samples).unwrap_or(1)
                            != if sample_depth { 1 } else { want_samples }
                }
                None => true,
            };
            // >>> GAMMA MODE IS STICKY GUEST STATE AND IT CAN CHANGE UNDER AN EXISTING TARGET.
            //
            // `sceGxmColorSurfaceSetGammaMode` is set on the SURFACE, not on the render target,
            // so a title may turn it on or off for a buffer it has been using all along. The
            // texture does not need rebuilding for that - both views already exist, which is
            // why the sRGB twin is declared on every target - but the recorded answer does,
            // and it was previously written once at creation and never again.
            //
            // What that costs is a renderer that DISAGREES WITH ITSELF: `encode_chain` renders
            // through the mode the guest is asking for RIGHT NOW, while everything that SAMPLES
            // the target - `sample_views` and `snapshot_rtt` - reads the stale one. One end
            // encodes and the other does not decode, and on a feedback path that error
            // compounds every iteration, running the image to white or to black depending on
            // which way round the two ended up. Measured on a 3-pass feedback chain with a
            // stale-true flag over a linear target: 128 -> 55 -> 10 -> 1 -> 0.
            if let Some(t) = self.rtt.get_mut(&addr) {
                if t.gamma != gamma {
                    t.gamma = gamma;
                    report_gamma_mode_changed(addr, gamma);
                    // The cached fixed-function bind groups name a view chosen under the old
                    // answer, so they are stale in exactly the same way.
                    self.rtt_binds.retain(|&(a, _, _), _| a != addr);
                }
            }
            if !stale {
                return;
            }
            // Every view of the target about to be replaced dies here, so every cached bind
            // group naming ANY target is invalidated. See `rtt_epoch`.
            self.rtt_epoch += 1;
            enc(&ENC.rtt_created, 1);
            let size = wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 };
            // Declare the sRGB twin as an allowed view format on EVERY target, not only the
            // ones currently in gamma mode: `sceGxmColorSurfaceSetGammaMode` is sticky state a
            // title may set at any point, and a texture's view formats are fixed at creation.
            // Declaring it costs nothing on any backend that matters and removes a whole class
            // of "the mode arrived after the target existed" bug.
            // >>> COMPATIBILITY MODE FORBIDS A VIEW OF ANOTHER FORMAT, so the sRGB twin is not
            // declared there and no gamma view is built. The cost is real and is reported by
            // `report_gamma_surface`: a gamma-correct surface then stores LINEAR values where the
            // hardware would sRGB-encode them, so it and everything sampling it read darker than
            // the title intends. That is a colour error on some passes. The alternative is not a
            // better picture, it is NO picture: declaring it makes the target itself invalid, and
            // every view, bind group and pass built on it fails with it.
            let srgb_fmt =
                if super::compat_mode() { None } else { srgb_twin(self.color_format) };
            let view_formats: Vec<wgpu::TextureFormat> = srgb_fmt.into_iter().collect();
            let color = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-rtt-color"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.color_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &view_formats,
            });
            let depth = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gxm-rtt-depth"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: if sample_depth {
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
                } else {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                },
                view_formats: &[],
            });
            let color_view = color.create_view(&Default::default());
            let color_view_srgb = srgb_fmt.map(|f| {
                color.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("gxm-rtt-color-srgb"),
                    format: Some(f),
                    ..Default::default()
                })
            });
            let depth_view = depth.create_view(&Default::default());
            let gxm_depth = sample_depth.then(|| {
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("gxm-rtt-guest-depth"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: GXM_DEPTH_FORMAT,
                    // COPY_SRC so a host that can read back (the headless chain dump) can
                    // report what the conversion actually produced. A depth buffer nobody can
                    // look at is a depth buffer nobody can debug.
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let view = tex.create_view(&Default::default());
                let src_view = depth.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("gxm-rtt-depth-sampled"),
                    aspect: wgpu::TextureAspect::DepthOnly,
                    ..Default::default()
                });
                GxmDepthTarget { src_view, tex, view }
            });
            // The multisampled attachments the guest asked for. Skipped when a later pass
            // SAMPLES this target's depth: that depth is converted from the attachment and
            // matched against the guest's own buffer, and [[vitaslop-depth-as-texture]] is
            // emphatic that depth has to match EXACTLY - a multisampled depth texture cannot
            // be fed to that conversion pass at all. Colour-only targets take the win.
            //
            // NOTE the scale mode is NOT consulted here. It describes the surface's STORED
            // form (it holds a resolved image), which is already true of `color` below; the
            // sample count is the render target's business and comes from the guest's
            // `multisampleMode`. Conflating the two is what made this a 2x2 supersample.
            let samples = want_samples;
            let grant = samples > 1 && !sample_depth;
            let msaa = grant.then(|| {
                let mc = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("gxm-rtt-msaa-color"),
                    size,
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.color_format,
                    // RENDER_ATTACHMENT only: nothing samples the multisampled image. What
                    // downstream reads is `color`, which this resolves into.
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    // The sRGB twin, for the same reason `color` declares it: a gamma surface
                    // is rendered through an sRGB view and a texture's view formats are fixed
                    // at creation.
                    view_formats: &view_formats,
                });
                let md = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("gxm-rtt-msaa-depth"),
                    size,
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format: DEPTH_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                MsaaAttachments {
                    color_view: mc.create_view(&Default::default()),
                    depth_view: md.create_view(&Default::default()),
                    color: mc,
                    depth: md,
                    samples,
                }
            });
            // A multisampled target whose colour surface does NOT ask for the downscale resolve
            // expects its own memory to hold the raw SAMPLES, not a resolved image - and we
            // hand it a resolved one. No target on any title measured here is in that state
            // (every multisampled one is scale mode 1), which is exactly why it would go
            // unnoticed if one ever were.
            if gxm_sample_count(multisample) > 1 && !msaa_downscale {
                report_unresolved_multisample_surface(addr, width, height);
            }
            if msaa.is_some() {
                report_multisample_granted(addr, width, height, multisample, samples);
            } else if gxm_sample_count(multisample) > 1 && !no_multisample() {
                // The guest asked and did not get it. `gamma` and `sample_depth` are the two
                // reasons, and they are reported here - at the one place that knows both -
                // rather than half here and half at encode time.
                report_multisample_refused(addr, width, height, gamma);
            }
            // A bind group over the new view is stale by construction; drop any cached ones.
            self.rtt_binds.retain(|&(a, _, _), _| a != addr);
            // Hand the OLD target's allocations back before the new one is installed. `insert`
            // would otherwise drop it, which on a native backend frees it and in the browser
            // does not - see `RttSurface::destroy`. Counted beside the creations so a standing
            // gap between the two is visible rather than inferred.
            if let Some(old) = self.rtt.insert(
                addr,
                RttSurface {
                    width,
                    height,
                    color,
                    color_view,
                    color_view_srgb,
                    gamma,
                    depth,
                    depth_view,
                    shadow: None,
                    gxm_depth,
                    msaa,
                },
            ) {
                old.destroy();
                enc(&ENC.rtt_destroyed, 1);
            }
        }

        /// Run the depth-conversion pass for the target at `addr`: read its depth attachment
        /// and write the guest's own depth encoding into its R32Float companion.
        ///
        /// Called right after the pass that filled the depth, because the next pass into the
        /// same target clears it again.
        fn convert_gxm_depth(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            encoder: &mut wgpu::CommandEncoder,
            addr: u32,
            depth_min: f32,
            depth_scale: f32,
        ) {
            let Some(t) = self.rtt.get(&addr) else { return };
            let Some(gd) = t.gxm_depth.as_ref() else { return };
            let (mode, konst) = gxm_depth_encoding();
            // The SAME fit the pass's own fragments use for their window POSITION - the two are
            // one quantity or every comparison between them is meaningless.
            let (fit_a, fit_c) = self.gxp.depth_fit_for(addr);
            let mut u = Vec::with_capacity(GXM_DEPTH_UNIFORM_BYTES as usize);
            u.extend_from_slice(&depth_min.to_le_bytes());
            u.extend_from_slice(&depth_scale.to_le_bytes());
            u.extend_from_slice(&mode.to_le_bytes());
            u.extend_from_slice(&konst.to_le_bytes());
            u.extend_from_slice(&fit_a.to_le_bytes());
            u.extend_from_slice(&fit_c.to_le_bytes());
            u.extend_from_slice(&0f32.to_le_bytes());
            u.extend_from_slice(&0f32.to_le_bytes());
            enc(&ENC.depth_converts, 1);
            enc(&ENC.buffer_bytes, u.len() as u64);
            // A whole extra pass, its own bind group, its own draw - all of it per converted
            // depth surface per frame, and none of it predicted by the frame's draw count.
            enc(&ENC.bind_groups_built, 1);
            enc(&ENC.passes, 1);
            enc(&ENC.pipeline_sets, 1);
            enc(&ENC.bind_group_sets, 1);
            enc(&ENC.draw_calls, 1);
            queue.write_buffer(&self.gxm_depth_uniform, 0, &u);
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gxm-depth-convert-bind"),
                layout: &self.gxm_depth_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&gd.src_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: self.gxm_depth_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler_point) },
                ],
            });
            let view = gd.view.clone();
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gxm-depth-convert"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.gxm_depth_pipe);
                pass.set_bind_group(0, &bind, &[]);
                pass.draw(0..3, 0..1);
            }
            report_depth_conversion(addr, mode, konst, depth_min, depth_scale);
        }

        /// Copy the rendered target at `addr` into a side texture and return a view of it.
        ///
        /// A pass is entitled to render into a buffer it also samples - a post-process
        /// stage reading the previous stage in place is the common shape - but a texture
        /// cannot be a colour target and a sampled resource in the same pass (wgpu rejects
        /// it, and the hardware read would be undefined). The snapshot is the buffer as it
        /// stood BEFORE this pass, which is exactly what such a pass means to read.
        fn snapshot_rtt(
            &mut self,
            device: &wgpu::Device,
            encoder: &mut wgpu::CommandEncoder,
            addr: u32,
        ) -> Option<wgpu::TextureView> {
            let color_format = self.color_format;
            // The shadow texture is created ONCE per target and copied into thereafter, so a
            // view of it is stable across frames exactly as the target's own view is - but its
            // creation is still a new view, and every cached group has to hear about it.
            let fresh_shadow = self.rtt.get(&addr).is_some_and(|t| t.shadow.is_none());
            if fresh_shadow {
                self.rtt_epoch += 1;
            }
            let t = self.rtt.get_mut(&addr)?;
            let size = wgpu::Extent3d { width: t.width.max(1), height: t.height.max(1), depth_or_array_layers: 1 };
            if t.shadow.is_none() {
                // The sRGB twin is declared on the snapshot for exactly the reason it is
                // declared on `color`: gamma mode is sticky guest state that can arrive after
                // the texture exists, and view formats are fixed at creation.
                // Same rule as the target's own view: compatibility mode refuses a view of
                // another format, and the snapshot is a copy of that target.
                let srgb_fmt = if super::compat_mode() { None } else { srgb_twin(color_format) };
                let view_formats: Vec<wgpu::TextureFormat> = srgb_fmt.into_iter().collect();
                let tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("gxm-rtt-shadow"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: color_format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &view_formats,
                });
                let view = tex.create_view(&Default::default());
                let view_srgb = srgb_fmt.map(|f| {
                    tex.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("gxm-rtt-shadow-srgb"),
                        format: Some(f),
                        ..Default::default()
                    })
                });
                t.shadow = Some((tex, view, view_srgb));
            }
            let gamma = t.gamma;
            let (shadow_tex, shadow_view, shadow_view_srgb) = t.shadow.as_ref()?;
            // The view the pass that RENDERED this target would hand out. `encode_chain` picks
            // the sRGB view for a gamma target and `sample_views` mirrors it; a snapshot of the
            // same bytes has to make the same choice or the two disagree about what the bytes
            // mean.
            let shadow_view = match (gamma, shadow_view_srgb.as_ref()) {
                (true, Some(v)) => v,
                _ => shadow_view,
            };
            enc(&ENC.rtt_snapshots, 1);
            enc(&ENC.rtt_snapshot_bytes, texture_bytes(size.width, size.height) as u64);
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &t.color,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: shadow_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                size,
            );
            Some(shadow_view.clone())
        }

        /// Ensure a fixed-function bind group exists for sampling `view` - the rendered
        /// target at `addr`, or its snapshot - with the given filter.
        fn ensure_rtt_bind(
            &mut self,
            device: &wgpu::Device,
            addr: u32,
            linear: bool,
            snapshot: bool,
            view: &wgpu::TextureView,
        ) {
            if self.rtt_binds.contains_key(&(addr, linear, snapshot)) {
                enc(&ENC.bind_groups_reused, 1);
                return;
            }
            enc(&ENC.bind_groups_built, 1);
            let samp = if linear { &self.sampler_linear } else { &self.sampler_point };
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gxm-rtt-bind"),
                layout: &self.texture_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(samp) },
                ],
            });
            self.rtt_binds.insert((addr, linear, snapshot), bind);
        }

        /// Encode a scene that has DEPTH but no colour surface, into a target keyed by its
        /// depth address, and register the converted depth so a later pass can sample it.
        ///
        /// # Why this pass exists at all
        /// A depth prepass, a shadow map or an occlusion pass renders no colour: the whole
        /// product is the depth buffer, and a later pass in the same frame binds it as a
        /// texture. Such a scene has no colour surface, so it used to be reported and dropped -
        /// and the later pass then sampled the depth ADDRESS and found guest bytes the GPU
        /// never wrote, i.e. whatever the CPU last left there. That is not a missing pass on
        /// screen; it is a pass whose absence is laundered into a plausible-looking image by
        /// the pass that reads it, which is the hardest kind of defect to see.
        ///
        /// The colour attachment here is a throwaway sized to the pass, so that the pipelines
        /// built for ordinary passes encode this one unchanged. Its contents are never read:
        /// this address is registered in `rtt_depth_rendered` (the DEPTH map), never in
        /// `rtt_rendered` (the colour one), so no sampler can reach it by accident.
        #[allow(clippy::too_many_arguments)]
        fn encode_depth_only_pass(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            encoder: &mut wgpu::CommandEncoder,
            scene: &RenderScene,
            width: u32,
            height: u32,
        ) {
            let addr = scene.depth_addr;
            // Keyed by the DEPTH address. A depth surface and a colour surface are allocated
            // near each other but are never the same address, so this cannot collide with a
            // colour target - and `sample_depth` is true by construction, since this pass is
            // only encoded when something samples it.
            //
            // One sample per pixel, unconditionally: a multisampled depth attachment cannot be
            // fed to the conversion pass that matches the guest's own depth buffer, which is
            // the same rule `ensure_rtt` applies to any target whose depth is read.
            self.ensure_rtt(device, addr, width, height, true, 0, false, false);
            let (cv, dv) = {
                let s = &self.rtt[&addr];
                (s.color_view.clone(), s.depth_view.clone())
            };
            self.rtt_reads_snapshot.clear();
            self.keep_depth = true;
            self.encode_pass(
                device,
                queue,
                encoder,
                &cv,
                &dv,
                self.color_format,
                scene,
                width,
                height,
                1,
                Some([0, 0, 0, 0]),
                1,
                None,
            );
            self.keep_depth = false;
            self.convert_gxm_depth(device, queue, encoder, addr, scene.depth_min, scene.depth_scale);
            if let Some(v) = self.rtt.get(&addr).and_then(|s| s.gxm_depth.as_ref()) {
                self.rtt_depth_rendered.insert(addr, v.view.clone());
            }
            report_depth_only_pass(
                addr,
                width,
                height,
                scene.draws.len(),
                scene.depth_extent_ambiguous,
            );
        }

        /// Encode a whole FRAME - every GXM scene the guest submitted, in order - so that a
        /// pass which renders into an offscreen buffer is actually drawn, and a later pass
        /// that samples that buffer gets what was drawn.
        ///
        /// This is the difference between a HUD and a game. A 3D title does not build its
        /// frame in one pass: it renders the world (and its shadow maps, reflections,
        /// post-process chain) into offscreen colour surfaces and then composites them onto
        /// the display buffer. Rendering only the last scene - which is all a single-scene
        /// `encode` can do - draws only that composite, and every texture it samples comes
        /// from guest memory that the guest never wrote, because on hardware the GPU wrote
        /// it. One of the retail racers is exactly this: fourteen offscreen passes carrying
        /// the entire world, then a 24-draw composite. The result was a correct, live HUD
        /// over a black screen.
        ///
        /// Scenes are encoded in submission order (a pass may sample an EARLIER pass's
        /// target, so order is load-bearing). A scene whose target is the same buffer the
        /// final scene draws to goes straight into the caller's view, clearing only on the
        /// first such pass so later ones compose onto it; every other scene goes into an
        /// offscreen target held by its surface's guest address. A scene with no resolvable
        /// target cannot be placed at all and is reported rather than silently dropped.
        #[allow(clippy::too_many_arguments)]
        /// Compile the WGSL for every shader pair the guest's patcher has named and this
        /// renderer has not compiled yet.
        ///
        /// # This is where the hardware does its shader work
        /// A `.gxp` carries USSE machine code the SDK compiled offline, so
        /// `sceGxmShaderPatcherCreateFragmentProgram` only has to patch and link - and it is
        /// handed `const SceGxmProgram *vertexProgram` precisely so it CAN, which means the
        /// pair is fully determined while the title is still on a loading screen. This
        /// recompiler has to produce WGSL and have a driver compile it, and doing that at the
        /// first DRAW put the whole cost inside gameplay frames: MEASURED on one title's
        /// on-track run, 931 ms of WGSL compile and 449 ms of pipeline creation, 160 pipelines
        /// built ACROSS the race, single frames spending 50-100 ms building 2-6 of them.
        ///
        /// Only the MODULE is prepared here. A pipeline is bound to the draw's cull mode and
        /// depth state, which GXM sets as runtime state and the patcher genuinely does not know.
        ///
        /// A pair that fails to link is skipped in silence: the draw path reports a fallback
        /// with the reason, once, at the site that knows which draw wanted it, and reporting the
        /// same failure from here would fire for pairs the title never draws.
        pub fn precompile_pairs(
            &mut self,
            device: &wgpu::Device,
            pairs: &[(std::sync::Arc<[u8]>, std::sync::Arc<[u8]>)],
        ) {
            // `VITASLOP_GXP_PRECOMPILE=0` is the A/B arm - VALUE-sensitive, because an arm has to
            // be: a presence-only reader turns `=0` into an ON arm and both arms then measure the
            // same build.
            if !self.gxp.enabled || pairs.is_empty() || !super::gxp_precompile() {
                return;
            }
            let t = Stopwatch::start();
            let mut built = 0u32;
            let mut budget_stopped = false;
            for (vprog, fprog) in pairs {
                // >>> SPREAD OVER FRAMES, because a loading screen is not a still image.
                // The candidate list a title's precomputed states imply arrives as a burst of a
                // few hundred pairs, and compiling all of them in the first frame that sees
                // them replaces a hitch in the RACE with a hitch on the loading screen - which
                // is a better place for it but still a visible one. The list is re-offered
                // every frame and never drained, so stopping here simply resumes next frame: at
                // this budget one measured title's 256 candidates spread across a couple of
                // hundred frames of the ~780 it leaves between naming them and racing. The
                // first pair of a call always runs, so progress cannot stall however slow the
                // device is.
                if built > 0 && t.ms() >= PRECOMPILE_MS_PER_FRAME {
                    budget_stopped = true;
                    break;
                }
                // >>> SKIP BY ALLOCATION BEFORE HASHING. The pending list is RE-OFFERED every
                // frame rather than drained (see `VitaState::shader_precompile` for why), and
                // `module_key` hashes both program blobs - kilobytes each. Considering a pair
                // once per run instead of once per frame is what keeps a list that exists to
                // move work OUT of the frame from becoming work IN it, and it matters more the
                // longer the list gets. The blobs are the capture's own `Arc`s, re-offered from
                // the same allocation, so pointer identity is the whole test; a re-created
                // program is a new allocation and is considered again, which is correct.
                let akey = (
                    std::sync::Arc::as_ptr(vprog) as *const u8 as usize,
                    std::sync::Arc::as_ptr(fprog) as *const u8 as usize,
                );
                if self.gxp.precompile_seen.len() >= PRECOMPILE_SEEN_CAP {
                    self.gxp.precompile_seen.clear();
                }
                if !self.gxp.precompile_seen.insert(akey) {
                    continue;
                }
                let mkey = GxpLive::module_key(vprog, fprog);
                if self.gxp.modules.contains_key(&mkey) {
                    continue;
                }
                let Ok(linked) = vitaslop_gxp_shader::link_programs(vprog, fprog) else { continue };
                // `keycolor` is None here on purpose: a pair the diagnostic wants is keyed by its
                // PIPELINE key (see `module_key`), so precompiling it under the pair-only key
                // would put a module nothing looks up into the cache.
                let Some(wgsl) =
                    inject_clip_fixup(&linked.wgsl, self.gxp.zfix, self.gxp.yflip, self.gxp.solid, None)
                else {
                    continue;
                };
                self.gxp.modules.insert(
                    mkey,
                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("gxp-precompiled"),
                        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                    }),
                );
                built += 1;
            }
            if built > 0 {
                self.gxp_precompiled += built;
                super::add_build_ms(&super::PIPE_PRECOMPILE_US, t.ms());
            }
            // At WARN, and reported even when NOTHING was built, because that is the interesting
            // case: pairs offered but none compiled means the pairs the patcher named do not
            // LINK, and pairs never offered at all means the title names none. Both leave the
            // WGSL compile in a gameplay frame, and neither is visible anywhere else.
            //
            // Reported only on a call that WALKED THE WHOLE LIST rather than stopping on the
            // per-frame budget. A budgeted call builds a different handful every frame, so
            // deduping those on their own shape prints a line per frame and buries the answer
            // in its own diagnostic; a call that reaches the end is the state worth naming,
            // and it names the cumulative total.
            {
                use std::collections::HashSet;
                use std::sync::Mutex;
                static SEEN: Mutex<Option<HashSet<(usize, u32)>>> = Mutex::new(None);
                let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                if !budget_stopped
                    && g.get_or_insert_with(HashSet::new).insert((pairs.len(), self.gxp_precompiled))
                {
                    // >>> THE CUMULATIVE TOTAL FIRST, because `built` is this CALL's count and
                    // the two read as one number. The old wording put them the other way round
                    // ("{built} of {offered} ... ({total} total)"), so a healthy run that had
                    // warmed 39 pairs and had nothing new to do printed "0 of 256 ... (39
                    // total)" - which reads as "the mechanism did nothing", and was read that
                    // way. An instrument whose steady state is indistinguishable from its own
                    // failure is the defect this project keeps finding
                    // [[vitaslop-instrument-failure-imitating-its-subject]].
                    report_warn!(
                        "gxp precompile: {} shader modules compiled AHEAD of any draw, out of {} \
                         candidate pairs offered ({built} new in this pass) - WGSL compile time \
                         that will not land in a gameplay frame. NOTE this warms the shader \
                         MODULE, not the render PIPELINE: a pipeline bakes the blend program and \
                         the attachment formats too, so a pair whose module is warm still pays \
                         `create_render_pipeline` at the draw. The candidates are the pairs the \
                         shader patcher NAMED unless VITASLOP_GXP_PRECOMPILE_CROSS is set, in \
                         which case they are a cross product and most of this is speculative",
                        self.gxp_precompiled,
                        pairs.len(),
                    );
                }
            }
        }

        pub fn encode_chain(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            encoder: &mut wgpu::CommandEncoder,
            color_view: &wgpu::TextureView,
            depth_view: &wgpu::TextureView,
            scenes: &[RenderScene],
            surf_w: u32,
            surf_h: u32,
            clear: [u8; 4],
        ) {
            // Release the previous frame's arena buffers on OUR schedule. The caller submitted
            // that frame's encoder before returning here, so their work is in flight or done
            // and `destroy()` is defined for both. See `retired_buffers` for what leaving this
            // to the JS collector measured.
            // Depth-range uniform buffers evicted during LAST frame's prepare join the same
            // queue: they were evicted mid-frame, when a draw already prepared could still name
            // them, so this is the first moment they are certainly unreferenced.
            // Prepare whatever the guest's shader patcher named since the last frame, BEFORE any
            // encoding. On the frame after a loading screen this is a burst; in a gameplay frame
            // it is empty, which is the point.
            for s in scenes {
                if !s.precompile.is_empty() {
                    let pairs = s.precompile.clone();
                    self.precompile_pairs(device, &pairs);
                }
            }
            let evicted = std::mem::take(&mut self.gxp.depth_retired);
            self.retired_buffers.extend(evicted);
            enc(&ENC.buffers_destroyed, self.retired_buffers.len() as u64);
            for b in self.retired_buffers.drain(..) {
                b.destroy();
            }
            // A new frame starts at pass ordinal 0, so pass N of this frame reuses the arenas
            // pass N of the last frame used. See `gxp_arenas`.
            self.gxp_arena_slot = 0;
            // >>> THE ONLY PLACE THE RESIDENT HEAPS MAY CHANGE IDENTITY.
            //
            // A prepared draw carries an OFFSET; the buffer it addresses is read at encode time.
            // Recreating a heap while a chain is in flight would silently re-point every draw
            // already prepared, so `place` only ever declines and the grow it asks for lands
            // here, between frames, with the old buffer going to the graveyard above rather than
            // being dropped - the last frame's commands still name it until its submit retires.
            if self.gxp.resident {
                let budget = self.gxp.resident_budget;
                if let Some(old) = self.gxp.resident_v.grow_or_reset(
                    device,
                    budget,
                    wgpu::BufferUsages::VERTEX,
                    "gxp-resident-vbo",
                ) {
                    self.retired_buffers.push(old);
                }
                if let Some(old) = self.gxp.resident_i.grow_or_reset(
                    device,
                    budget,
                    wgpu::BufferUsages::INDEX,
                    "gxp-resident-ibo",
                ) {
                    self.retired_buffers.push(old);
                }
            }
            self.rtt_rendered.clear();
            self.rtt_depth_rendered.clear();
            self.rtt_hits = 0;
            self.chain_phases = EncodePhases::default();
            // A new frame: everything the texture cache holds is now a candidate for eviction
            // again. Bumped HERE and nowhere else, because "used by the frame being encoded"
            // is the one property that makes an entry un-evictable, and a frame is exactly
            // this call. See `GxpLive::views_used`.
            self.gxp.views_epoch = self.gxp.views_epoch.wrapping_add(1);
            // The frame that just finished is the evidence for how big one frame's working set
            // is. Take its high-water mark BEFORE resetting, so the floor is set by a complete
            // frame and never by a partial one.
            self.gxp.views_frame_high = self.gxp.views_frame_high.max(self.gxp.views_frame_bytes);
            self.gxp.views_frame_bytes = 0;
            let Some(last) = scenes.last() else { return };
            // Which scenes' DEPTH some draw in this frame samples. Scanned up front because
            // the reader comes AFTER the writer: the pass that fills a depth buffer has to
            // know, while it is being set up, that something later will read it - a target
            // built without `TEXTURE_BINDING` cannot be sampled at all, and the extra
            // conversion is not worth paying on the targets nothing reads.
            let sampled: HashSet<u32> = scenes
                .iter()
                .flat_map(|s| s.draws.iter())
                .flat_map(|d| {
                    // BOTH stages. A vertex program that samples a depth surface (a
                    // displacement or reprojection built in the vertex stage) names it exactly
                    // as a fragment one does, and leaving vertex samplers out of this set makes
                    // the pass that WROTE that depth skip the conversion - so the reader gets
                    // raw hardware depth with nothing saying so.
                    d.texture.iter().map(|t| t.data_addr).chain(
                        d.gxp.iter().flat_map(|g| {
                            g.textures
                                .iter()
                                .chain(g.vertex_textures.iter())
                                .map(|t| t.tex.data_addr)
                        }),
                    )
                })
                .collect();
            let depth_sampled: HashSet<u32> = scenes
                .iter()
                .map(|s| s.depth_addr)
                .filter(|a| *a != 0 && sampled.contains(a))
                .collect();
            // NOTE there is no "may this buffer be averaged" test here any more, and its
            // absence is the point. The old one asked how a target was FILTERED - linear for
            // an image, point for data - because the pass was a 2x2 SUPERSAMPLE whose box
            // resolve really did mix values that must not be mixed. It was refuted: this
            // title's shadow map is linear-sampled and the race regressed anyway.
            //
            // The question dissolves once the pass multisamples instead of supersamples. MSAA
            // shades ONCE per pixel and hands that one value to every covered sample, so a
            // resolve inside a triangle returns it exactly, whatever the buffer means. Only
            // pixels on a silhouette mix, which is where the hardware mixes them too. So the
            // guest's `multisampleMode` is the whole decision and no heuristic stands next to
            // it. See `gxm_sample_count`.
            // The display buffer is whatever the final scene draws to. Any earlier scene
            // naming the same address is part of the same image, not an offscreen pass.
            let display = last.target.map(|t| t.data_addr);
            report_world_not_on_display(scenes, display, &sampled);
            let ss = self.ss_scale > 1;
            if ss {
                self.ensure_ss_target(device, queue, surf_w, surf_h);
            }
            let mut display_pass_done = false;
            let n = scenes.len();
            for (i, scene) in scenes.iter().enumerate() {
                let to_display = i + 1 == n || (scene.target.map(|t| t.data_addr) == display && display.is_some());
                if to_display {
                    let (cv, dv) = match (ss, self.ss_target.as_ref()) {
                        (true, Some(t)) => (t.color_view.clone(), t.depth_view.clone()),
                        _ => (color_view.clone(), depth_view.clone()),
                    };
                    let first = !display_pass_done;
                    display_pass_done = true;
                    self.rtt_reads_snapshot.clear();
                    // The display target's format belongs to the surface the host handed us,
                    // and the host owns whether that is already an sRGB swapchain - so a
                    // gamma-mode DISPLAY surface is not reinterpreted here.
                    let fmt = self.color_format;
                    // The DISPLAY buffer. Its render target is the one the guest creates at
                    // boot, and on every title measured here it is `SCE_GXM_MULTISAMPLE_NONE`
                    // - the console composites the front buffer at one sample. So this pass
                    // has no resolve and does not ask for one.
                    // The supersampled display pass renders into an attachment `ss_scale`
                    // times the target size, so every rectangle expressed in TARGET pixels -
                    // the viewport and the guest's scissor - has to be scaled to reach the
                    // right texels. It is 1 when supersampling is off, which is the default.
                    let attach_scale = if ss { self.ss_scale } else { 1 };
                    self.encode_pass(device, queue, encoder, &cv, &dv, fmt, scene, surf_w, surf_h, attach_scale, first.then_some(clear), 1, None);
                    // A display pass keeps no depth copy (its depth attachment belongs to the
                    // caller and is discarded), so if something reads this scene's depth it
                    // will not find it. Say so rather than let the read fall through silently.
                    if depth_sampled.contains(&scene.depth_addr) {
                        report_unconverted_depth_sample(scene.depth_addr);
                    }
                    continue;
                }
                let Some(t) = scene.target else {
                    // A colour-less pass whose DEPTH a later pass samples is load-bearing, and
                    // it now gets a target of its own keyed by the DEPTH address rather than
                    // being dropped. The colour attachment that target carries is a throwaway:
                    // nothing reads it, and it exists so that every pipeline built for an
                    // ordinary pass - all of which declare a colour target - can encode this
                    // one unchanged. Building a second, colour-less pipeline variant for each
                    // pair would be the only alternative, and it would double the pipeline
                    // cache to serve the passes that need it least.
                    if depth_sampled.contains(&scene.depth_addr) {
                        if let Some((w, h)) = scene.depth_extent {
                            self.encode_depth_only_pass(device, queue, encoder, scene, w, h);
                            continue;
                        }
                    }
                    report_unplaced_scene(
                        scene.draws.len(),
                        scene.depth_addr,
                        depth_sampled.contains(&scene.depth_addr),
                        scene.depth_extent.is_some(),
                    );
                    continue;
                };
                let want_depth = depth_sampled.contains(&scene.depth_addr);
                // How many samples the GUEST created this render target with. Not a quality
                // setting and not a per-target judgement of ours - `ensure_rtt` decides only
                // whether it can serve the request, and reports when it cannot.
                self.ensure_rtt(device, t.data_addr, t.width, t.height, want_depth, t.multisample, t.msaa_downscale, t.gamma);
                // Drawing into a buffer this frame already filled, which this pass may also
                // sample: hand it a snapshot to read so the live buffer is a target only.
                self.rtt_reads_snapshot.clear();
                // Clear a target the FIRST time this frame draws into it, and compose onto
                // it after that - the same rule the display buffer follows. A later pass
                // into a buffer an earlier pass filled is a post-process step, and it is
                // entitled to leave most of the image alone: a retail racer's race frame ends
                // with a five-draw pass over the world target, and clearing for it wiped the
                // whole world, leaving a correct HUD over black.
                let first_pass_here = !self.rtt_rendered.contains_key(&t.data_addr);
                if !first_pass_here {
                    if let Some(before) = self.snapshot_rtt(device, encoder, t.data_addr) {
                        self.rtt_rendered.insert(t.data_addr, before);
                        self.rtt_reads_snapshot.insert(t.data_addr);
                    }
                }
                // A GAMMA-CORRECT surface is rendered through the sRGB view of the same
                // texture, so the ROP encodes each store after blending exactly as the
                // hardware does. The same view goes into `rtt_rendered` below, so a later pass
                // sampling this target decodes on the way back in.
                let gamma = t.gamma;
                let (cv, dv, fmt) = {
                    let s = &self.rtt[&t.data_addr];
                    match (gamma, s.color_view_srgb.as_ref()) {
                        (true, Some(v)) => (
                            v.clone(),
                            s.depth_view.clone(),
                            srgb_twin(self.color_format).unwrap_or(self.color_format),
                        ),
                        _ => (s.color_view.clone(), s.depth_view.clone(), self.color_format),
                    }
                };
                if gamma {
                    report_gamma_surface(t.data_addr, fmt != self.color_format);
                }
                // Rasterise into the MULTISAMPLED attachments when the guest created this
                // render target multisampled, naming `cv` as the resolve target so everything
                // downstream still reads the target at its STORED size. The resolve is part of
                // ending the pass, which is where the hardware does it too - there is no
                // separate resolve pass and no box filter.
                //
                // Not for a GAMMA surface. There the ROP sRGB-encodes each store after
                // blending, and a multisample resolve would have to decide whether it averages
                // the encoded or the linear values. The two give different pixels, hardware
                // does one of them, and nothing here has measured which - so a gamma target
                // keeps the path whose behaviour is already pinned. No measured title's
                // background surface is in gamma mode, so this costs nothing today.
                // `ensure_rtt` already refused a gamma target its attachments and said so, so
                // there is nothing left to decide here: if the attachments exist, use them.
                let use_msaa = self.rtt[&t.data_addr].msaa.is_some();
                let (pass_cv, pass_dv, pass_samples, resolve) =
                    match self.rtt[&t.data_addr].msaa.as_ref() {
                        Some(m) if use_msaa => {
                            (m.color_view.clone(), m.depth_view.clone(), m.samples, Some(cv.clone()))
                        }
                        _ => (cv.clone(), dv.clone(), 1, None),
                    };
                // A first pass is cleared to transparent black, not to the display's clear
                // colour: it is an intermediate image, and a composite that blends it must
                // see nothing where the pass drew nothing.
                let clear = first_pass_here.then_some([0, 0, 0, 0]);
                self.keep_depth = want_depth;
                self.encode_pass(
                    device, queue, encoder, &pass_cv, &pass_dv, fmt, scene, t.width, t.height, 1,
                    clear, pass_samples, resolve.as_ref(),
                );
                self.keep_depth = false;
                self.rtt_rendered.insert(t.data_addr, cv);
                // Convert this pass's depth NOW: the next pass into the same target clears the
                // depth attachment, so afterwards there is nothing left to convert.
                if want_depth {
                    self.convert_gxm_depth(
                        device,
                        queue,
                        encoder,
                        t.data_addr,
                        scene.depth_min,
                        scene.depth_scale,
                    );
                    if let Some(v) = self.rtt.get(&t.data_addr).and_then(|s| s.gxm_depth.as_ref()) {
                        self.rtt_depth_rendered.insert(scene.depth_addr, v.view.clone());
                    }
                }
            }
            // Report the frame's pass structure whenever it CHANGES: how many scenes, where
            // each one draws, and - the load-bearing number - how many draws sampled a
            // target this frame rendered. A chain of passes with zero samples of them means
            // the composite never reads the world, which looks exactly like the passes not
            // being drawn at all but is a different bug.
            if scenes.len() > 1 || self.rtt_hits > 0 {
                let shape = scenes
                    .iter()
                    .map(|s| match s.target {
                        Some(t) => format!("{:#x}:{}x{}/{}", t.data_addr, t.width, t.height, s.draws.len()),
                        None => format!("?/{}", s.draws.len()),
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                let mut sampled: Vec<String> =
                    self.sampled_addrs.iter().map(|a| format!("{a:#x}")).collect();
                sampled.sort();
                let line = format!(
                    "gxm chain: {} scenes [{shape}] rtt-samples={} final-pass-sampled=[{}]",
                    scenes.len(),
                    self.rtt_hits,
                    sampled.join(" ")
                );
                if self.last_chain_shape.as_deref() != Some(line.as_str()) {
                    report!("{line}");
                    self.last_chain_shape = Some(line);
                }
            }
            // Resolve the supersampled display buffer once, after the last pass into it.
            if let (true, Some(t)) = (ss, self.ss_target.as_ref()) {
                enc(&ENC.passes, 1);
                enc(&ENC.pipeline_sets, 1);
                enc(&ENC.bind_group_sets, 1);
                enc(&ENC.draw_calls, 1);
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gxm-resolve"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rpass.set_pipeline(&self.resolve_pipeline);
                rpass.set_bind_group(0, &t.resolve_bind, &[]);
                rpass.draw(0..3, 0..1);
            }
            // >>> REPORTED WHETHER OR NOT IT FITS, at the END of the frame that built it.
            //
            // The composition used to print ONLY when the budget was exceeded and nothing was
            // evictable - the failure case - which made a successful shrink unmeasurable: three
            // arms of an A/B (decode, passthrough, transcode) produced ONE number between them,
            // because the two arms that WORKED stopped tripping the report that carried the
            // number. It is unconditional now, bounded by a high-water step instead.
            //
            // At the end, not the top. The top of this function holds the PREVIOUS frame's
            // total, which is the right number in a windowed run and is zero in a headless one -
            // headless fast-forwards the guest and encodes a single frame, so a report placed
            // there fired exactly once, before that frame had bound anything, and said 0 MB.
            // A frame-scoped statistic has to be read where the frame ends.
            report_texture_working_set(
                self.gxp.views_frame_bytes,
                &self.gxp.views_used,
                self.gxp.views_epoch,
            );
        }

        /// Encode a full scene into `encoder`: a render pass over `color_view` (cleared
        /// to `clear`) with `depth_view` (must be [`DEPTH_FORMAT`]), drawing every draw
        /// in submission order. `(surf_w, surf_h)` are the target size (the Pixel-space
        /// projection needs them). Requires `&mut self` for the per-frame arenas and the
        /// cross-frame caches.
        ///
        /// One frame does: at most three `write_buffer` uploads (vertices, indices,
        /// uniforms), a texture upload only for a not-yet-seen texture, and one bind
        /// group only for a not-yet-seen (texture, filter) pair - then the pass. Nothing
        /// is allocated per draw once the working set is warm.
        #[allow(clippy::too_many_arguments)]
        pub fn encode(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            encoder: &mut wgpu::CommandEncoder,
            color_view: &wgpu::TextureView,
            depth_view: &wgpu::TextureView,
            scene: &RenderScene,
            surf_w: u32,
            surf_h: u32,
            clear: [u8; 4],
        ) {
            self.encode_chain(
                device,
                queue,
                encoder,
                color_view,
                depth_view,
                std::slice::from_ref(scene),
                surf_w,
                surf_h,
                clear,
            );
        }

        /// One scene into one target. `clear` is `Some(colour)` for the first pass into a
        /// target and `None` to compose onto what is already there. Supersampling is the
        /// caller's business ([`encode_chain`](Self::encode_chain) picks the views and
        /// resolves once), so this is a plain pass.
        #[allow(clippy::too_many_arguments)]
        fn encode_pass(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            encoder: &mut wgpu::CommandEncoder,
            color_view: &wgpu::TextureView,
            depth_view: &wgpu::TextureView,
            // The FORMAT of `color_view`. Usually the renderer's `color_format`, but the sRGB
            // twin when this pass renders into a gamma-correct surface - and a pipeline is
            // bound to the format of the attachment it writes, so every pipeline this pass
            // uses has to be built for this format, not for the renderer's default.
            target_format: wgpu::TextureFormat,
            scene: &RenderScene,
            surf_w: u32,
            surf_h: u32,
            // Attachment texels per TARGET pixel. Always 1 except on the supersampled
            // display pass, whose attachment is `ss_scale` times the target in each axis -
            // so a rectangle the guest states in target pixels (the viewport, the region
            // clip) must be multiplied by this to name the right texels.
            attach_scale: u32,
            clear: Option<[u8; 4]>,
            // Samples per pixel of `color_view`/`depth_view`. A pipeline is bound to its
            // attachments' sample count exactly as it is bound to their format, so this
            // travels with `target_format` and belongs in the pipeline cache key.
            samples: u32,
            // Where a multisampled pass resolves to: the stored-size texture everything
            // downstream samples. `None` when `samples == 1` and there is nothing to resolve.
            resolve: Option<&wgpu::TextureView>,
        ) {
            // 1. Walk the scene once: pack vertex/index/uniform bytes into per-frame
            //    arenas and ensure each draw's texture upload + bind group exist.
            // Before any draw is prepared: does THIS pass's projection put clip `w` negative in
            // front of the camera? It is one answer for the whole pass and every draw's
            // `@group(3)` block carries it, so it has to be settled first.
            self.gxp.decide_scene_negw(scene);
            let t_prepare = Stopwatch::start();
            let stride = self.uniform_stride as usize;
            let mut vdata: Vec<u8> = Vec::new();
            let mut idata: Vec<u8> = Vec::new();
            let mut udata: Vec<u8> = Vec::new();
            // The recompiled path's arenas (see `GxpPrepared`), packed during the same walk.
            let mut gvdata: Vec<u8> = Vec::new();
            let mut gidata: Vec<u8> = Vec::new();
            let mut gudata: Vec<u8> = Vec::new();
            let ubo_align = device.limits().min_uniform_buffer_offset_alignment as u64;
            let mut items: Vec<Item> = Vec::with_capacity(scene.draws.len());
            // The live recompiler's per-draw resources + a submission-order plan interleaving
            // recompiled and fixed-function draws (so they share one depth-tested pass).
            let gxp_enabled = self.gxp.enabled;
            let gxp_only = self.gxp.only;
            // The attachment this pass writes, NOT the renderer's default - see the parameter.
            let color_format = target_format;
            let mut gxp_prepared: Vec<GxpPrepared> = Vec::new();
            let mut order: Vec<Enc> = Vec::with_capacity(scene.draws.len());
            // The guest's SCISSOR for each entry of `order`, in the same positions. It rides
            // beside `order` rather than inside `GxpPrepared`/`Item` because it is state both
            // paths share and neither owns - the fixed-function path has no per-draw state
            // struct of its own that a recompiled draw also passes through.
            let mut clips: Vec<RegionClip> = Vec::with_capacity(scene.draws.len());
            // Taken out for the walk so the render-target views can be read while the
            // texture caches next to them are written; restored below.
            let rendered = std::mem::take(&mut self.rtt_rendered);
            let depth_rendered = std::mem::take(&mut self.rtt_depth_rendered);
            // Read out for the same reason: `prepare` borrows `self.gxp` mutably.
            let rtt_epoch = self.rtt_epoch;
            let reads_snapshot = std::mem::take(&mut self.rtt_reads_snapshot);
            // What this pass may SAMPLE: the targets this frame has rendered, plus every
            // target still RESIDENT from an earlier frame.
            //
            // A render target is guest memory, and guest memory keeps what was last written
            // into it until something overwrites it. `rtt_rendered` is cleared every frame, so
            // without this a pass sampling a buffer drawn on an EARLIER frame falls through to
            // decoding the guest bytes behind the pointer - and the GPU, not the guest, wrote
            // those pixels, so they decode to black.
            //
            // MEASURED on a retail title: at the title-to-menu transition the guest stops
            // rendering its background into 0x89204aa0 and starts blurring it instead. The root
            // of that blur chain samples 0x89204aa0, which is resident from the previous frame
            // and absent from `rtt_rendered`, so it read black - and every pass downstream of
            // it inherited the black, taking 91% of the frame with it in one flip.
            //
            // This is deliberately SEPARATE from `rtt_rendered`, which still decides two other
            // things that must not change: which pass clears a target first (a resident target
            // is still cleared on this frame's first pass, or a post-process would compose onto
            // a stale image forever) and which reads need a snapshot.
            let current_target = scene.target.map(|t| t.data_addr);
            let mut sample_views = rendered.clone();
            for (addr, t) in self.rtt.iter() {
                // NEVER the attachment this pass is writing: binding it as a sampler at the
                // same time is a use-after-alias the driver rejects. Within-frame feedback is
                // what `rtt_reads_snapshot` is for, and it only applies once a pass has
                // already rendered into the target - which puts it in `rendered` above.
                if Some(*addr) == current_target {
                    continue;
                }
                // >>> THE SAME VIEW THE PASS THAT RENDERED IT WOULD HAND OUT, WHICH FOR A
                // GAMMA-CORRECT TARGET IS THE sRGB ONE.
                //
                // A gamma-correct surface holds sRGB-ENCODED bytes: the ROP encodes every store,
                // and `color_view_srgb` exists precisely so a sampler DECODES them on the way
                // back in ("what the hardware does at both ends", per its own doc). The
                // within-frame path above obeys that - `rtt_rendered` gets the sRGB view. This
                // path did not, and handed out the linear view instead.
                //
                // Reading encoded bytes as linear returns them too BRIGHT (0.5 stored as 0.73
                // reads back as 0.73). On its own that is a one-off error. In a FEEDBACK chain -
                // a target sampled and written back into itself every frame, which is what an
                // accumulation/blur buffer is - it compounds: 0.5 -> 0.73 -> 0.88 -> 0.95 -> 1.
                //
                // That is the "renders fine for one frame, then slowly skews itself to white"
                // the user reported for days, and the shape is exactly right: a restore
                // re-creates the targets, so the first frame after it has nothing accumulated
                // and is CORRECT, and every frame after re-applies the extra encode. It reaches
                // only the cross-frame path, which is why a single-frame desktop headless shot
                // could never show it.
                sample_views.entry(*addr).or_insert_with(|| {
                    match (t.gamma, t.color_view_srgb.as_ref()) {
                        (true, Some(v)) => v.clone(),
                        _ => t.color_view.clone(),
                    }
                });
            }
            self.sampled_addrs.clear();
            // `VITASLOP_CHAIN_DRAWS=1`: describe every draw in this pass that samples a
            // target the frame rendered. A composite that shows none of the world has
            // either no such draw, or one whose blend/space/geometry throws it away, and
            // the finished (black) frame cannot tell those apart.
            let trace_draws = std::env::var_os("VITASLOP_CHAIN_DRAWS").is_some();
            // `=all` describes EVERY draw of the pass, not only the ones sampling a
            // rendered target - what is needed when the question is "which draw was
            // supposed to put the world on screen" rather than "did this one bind right".
            let trace_all = std::env::var("VITASLOP_CHAIN_DRAWS").map(|v| v == "all").unwrap_or(false);
            // Fallback draws of THIS pass, by reason - see `fallback_reasons`.
            let mut fb_reasons: HashMap<String, usize> = HashMap::default();
            for (di, d) in scene.draws.iter().enumerate() {
                if trace_draws {
                    // The sampled texture's own dimensions matter as much as its address: a
                    // title may bind a render target through a texture describing a
                    // different extent, and the UVs the shader computes are for THAT extent.
                    let hits: Vec<String> = d
                        .texture
                        .iter()
                        .map(|t| (t.data_addr, t.width, t.height))
                        .chain(d.gxp.iter().flat_map(|g| {
                            g.textures
                                .iter()
                                .chain(g.vertex_textures.iter())
                                .map(|t| (t.tex.data_addr, t.tex.width, t.tex.height))
                        }))
                        .filter(|(a, _, _)| trace_all || sample_views.contains_key(a))
                        // `*` this frame rendered it, `~` it is RESIDENT from an earlier frame,
                        // nothing at all means the sample decodes guest bytes. The difference
                        // between the first two used to be invisible and it was the whole of
                        // the black-background bug - the chain root read a target
                        // the frame had not redrawn, and an unmarked address looked identical
                        // to an ordinary texture.
                        .map(|(a, w, h)| {
                            let mark = if rendered.contains_key(&a) {
                                "*"
                            } else if sample_views.contains_key(&a) {
                                "~"
                            } else {
                                ""
                            };
                            format!("{a:#x}({w}x{h}){mark}")
                        })
                        .collect();
                    if !hits.is_empty() || trace_all {
                        report!(
                            // "carries a payload" is NOT "is recompiled" - a payload that
                            // fails to link falls back, and labelling that `recompiled=true`
                            // is how a composite draw got read as working when it was not.
                            // The index count is the RECOMPILED one when there is a payload.
                            // The fixed-function count is zero for such a draw by design (the
                            // builder does not produce a representation the renderer will not
                            // use), and printing that made every draw of a recompiled pass
                            // read as empty geometry - a diagnostic saying exactly the wrong
                            // thing about the question it exists to answer.
                            "chain draw #{di}: samples {:?} key={:?} has_payload={} blend={:?} opaque={} space={:?} idx={}",
                            hits,
                            d.gxp.as_ref().map(|g| format!("{:x}", GxpLive::key(g))),
                            d.gxp.is_some(),
                            d.gxp.as_ref().map(|g| g.blend),
                            d.opaque,
                            d.space,
                            d.gxp.as_ref().map(|g| g.index_count).unwrap_or(d.index_count)
                        );
                    }
                }
                if let Some(t) = &d.texture {
                    self.sampled_addrs.insert(t.data_addr);
                }
                if let Some(g) = &d.gxp {
                    // Vertex samplers count too: `final-pass-sampled` is the list a reader
                    // checks a missing target against, and a target read only by a vertex
                    // program was invisible in it.
                    self.sampled_addrs.extend(
                        g.textures.iter().chain(g.vertex_textures.iter()).map(|t| t.tex.data_addr),
                    );
                }
                // Live GXP path: draw with the guest's real shaders when the pair links. On a
                // link/format failure fall through to the fixed-function packing below (unless
                // isolate mode, which renders only the recompiled draws).
                if gxp_enabled {
                    if let Some(g) = &d.gxp {
                        self.rtt_hits += g
                            .textures
                            .iter()
                            .chain(g.vertex_textures.iter())
                            .filter(|t| sample_views.contains_key(&t.tex.data_addr))
                            .count();
                        if let Some(mut prep) =
                            self.gxp.prepare(device, queue, color_format, samples, g, [scene.depth_min, scene.depth_scale], &sample_views, &depth_rendered, rtt_epoch, &reads_snapshot, &mut gvdata, &mut gidata, &mut gudata, ubo_align)
                        {
                            if self.gxp.solid {
                                prep.blend = false; // REPLACE + depth-Always variant (see make)
                            }
                            order.push(Enc::Gxp(gxp_prepared.len()));
                            clips.push(d.region_clip);
                            gxp_prepared.push(prep);
                            continue;
                        }
                        // Prepared failed: this draw is one of the pass's fallbacks. Tally it
                        // against its pair's reason so the summary can rank causes by draws.
                        *fb_reasons.entry(fallback_reason_of(GxpLive::key(g))).or_insert(0) += 1;
                        if gxp_only {
                            continue;
                        }
                    } else if gxp_only {
                        continue;
                    }
                }
                // A shader-only draw has no colour source the fixed-function packing can
                // honour, so reaching here means the recompiler could not draw it and the
                // alternative would be an opaque white rectangle. Skip it - but never
                // silently: this is a draw the guest asked for that the frame will not show.
                if d.shader_only {
                    crate::gpu::report_shader_only_skip(d, gxp_enabled);
                    continue;
                }
                if d.index_count == 0 || d.vertices.is_empty() {
                    continue;
                }
                let bind = match &d.texture {
                    None => BindKey::White,
                    // A texture whose data pointer is a target THIS frame already rendered
                    // is that render, not the guest bytes behind the pointer (which the GPU,
                    // not the guest, was supposed to have written).
                    Some(t) if sample_views.contains_key(&t.data_addr) => {
                        self.rtt_hits += 1;
                        let snapshot = self.rtt_reads_snapshot.contains(&t.data_addr);
                        let view = sample_views[&t.data_addr].clone();
                        self.ensure_rtt_bind(device, t.data_addr, t.filter_linear, snapshot, &view);
                        BindKey::Rtt(t.data_addr, t.filter_linear, snapshot)
                    }
                    Some(t) => {
                        self.ensure_texture(device, queue, t);
                        BindKey::Tex(t.key, t.filter_linear)
                    }
                };
                let (mode, mvp) = match d.space {
                    DrawSpace::Mvp(m) => (0u32, m),
                    DrawSpace::Ndc => (1u32, [0f32; 16]),
                    DrawSpace::Pixel => (2u32, [0f32; 16]),
                };
                // Uniform block (must match the WGSL `U` struct), then pad to the
                // dynamic-offset stride. A vec4 in WGSL is 16-byte aligned, so each material
                // vec3 is written as 4 floats (xyz + a pad lane).
                let uniform_offset = udata.len() as u32;
                for v in &mvp {
                    udata.extend_from_slice(&v.to_le_bytes());
                }
                udata.extend_from_slice(&mode.to_le_bytes());
                udata.extend_from_slice(&(surf_w as f32).to_le_bytes());
                udata.extend_from_slice(&(surf_h as f32).to_le_bytes());
                udata.extend_from_slice(&(d.texture.is_some() as u32).to_le_bytes());
                udata.extend_from_slice(&d.exposure.to_le_bytes());
                udata.extend_from_slice(&scene.depth_min.to_le_bytes());
                udata.extend_from_slice(&scene.depth_scale.to_le_bytes());
                udata.extend_from_slice(&[0u8; 4]); // pad2
                let vec4 = |udata: &mut Vec<u8>, v: [f32; 3]| {
                    for c in v {
                        udata.extend_from_slice(&c.to_le_bytes());
                    }
                    udata.extend_from_slice(&[0u8; 4]); // vec4 pad lane
                };
                vec4(&mut udata, d.material.tint);
                vec4(&mut udata, d.material.light_dir);
                vec4(&mut udata, d.material.light_col);
                vec4(&mut udata, d.material.ambient);
                udata.resize(uniform_offset as usize + stride, 0);

                let v_off = vdata.len() as u64;
                vdata.extend_from_slice(&d.vertices);
                let i_off = idata.len() as u64;
                idata.extend_from_slice(&d.indices);

                items.push(Item {
                    v_off,
                    v_len: d.vertices.len() as u64,
                    i_off,
                    i_len: d.indices.len() as u64,
                    index_count: d.index_count,
                    uniform_offset,
                    opaque: d.opaque,
                    bind,
                });
                order.push(Enc::Fixed(items.len() - 1));
                clips.push(d.region_clip);
            }
            self.rtt_rendered = rendered;
            self.rtt_depth_rendered = depth_rendered;
            self.rtt_reads_snapshot = reads_snapshot;
            if gxp_enabled {
                let with_payload = scene.draws.iter().filter(|d| d.gxp.is_some()).count();
                let summary = (scene.draws.len(), with_payload, gxp_prepared.len(), items.len());
                if self.last_gxp_summary != Some(summary) {
                    self.last_gxp_summary = Some(summary);
                    report!(
                        "gxp: scene has {} draws, {} carry a shader payload, {} recompiled+prepared, {} fixed-function items",
                        summary.0, summary.1, summary.2, summary.3,
                    );
                    // Rank the causes by the draws they cost, biggest first. Without this the
                    // only ranking available is the printed pair list, which counts pairs.
                    let mut ranked: Vec<_> = fb_reasons.iter().collect();
                    ranked.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
                    for (reason, n) in ranked {
                        report_warn!("gxp:   {n} fallback draws - {reason}");
                    }
                }
            }

            self.last_phases = EncodePhases {
                prepare_ms: t_prepare.ms(),
                upload_ms: 0.0,
                pass_ms: 0.0,
                gxp_draws: gxp_prepared.len(),
                fixed_draws: items.len(),
            };

            // 2. Size the arenas and upload. Rebuild the uniform bind group if the uniform
            //    buffer was (re)created.
            let t_upload = Stopwatch::start();
            if !items.is_empty() {
                Self::ensure_buffer(device, &mut self.vbo, &mut self.vbo_cap, vdata.len() as u64, wgpu::BufferUsages::VERTEX, "gxm-vbo");
                Self::ensure_buffer(device, &mut self.ibo, &mut self.ibo_cap, idata.len() as u64, wgpu::BufferUsages::INDEX, "gxm-ibo");
                let ubo_new = Self::ensure_buffer(device, &mut self.ubo, &mut self.ubo_cap, udata.len() as u64, wgpu::BufferUsages::UNIFORM, "gxm-ubo");
                if ubo_new || self.ubo_bind.is_none() {
                    enc(&ENC.bind_groups_built, 1);
                    let ubo = self.ubo.as_ref().unwrap();
                    self.ubo_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("gxm-ubo-bind"),
                        layout: &self.uniform_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: ubo,
                                offset: 0,
                                size: wgpu::BufferSize::new(UNIFORM_BYTES),
                            }),
                        }],
                    }));
                }
                enc(&ENC.buffer_bytes, (vdata.len() + idata.len() + udata.len()) as u64);
                queue.write_buffer(self.vbo.as_ref().unwrap(), 0, &vdata);
                queue.write_buffer(self.ibo.as_ref().unwrap(), 0, &idata);
                queue.write_buffer(self.ubo.as_ref().unwrap(), 0, &udata);
            }
            // The recompiled path's arenas: THREE buffers for the whole pass, however many
            // draws it carries, instead of four per draw. This pass takes the slot for its
            // ORDINAL in the chain and keeps it across frames - see `gxp_arenas` - so the
            // steady state creates no buffer at all, and every buffer named by this frame's
            // commands stays alive through submit because the renderer owns it outright.
            let gxp_slot = (!gxp_prepared.is_empty()).then(|| {
                let slot = self.gxp_arena_slot;
                self.gxp_arena_slot += 1;
                Self::ensure_gxp_arena(
                    device,
                    queue,
                    &mut self.gxp_arenas,
                    &mut self.retired_buffers,
                    slot,
                    &gvdata,
                    &gidata,
                    &gudata,
                );
                slot
            });
            if let Some(slot) = gxp_slot {
                let generation = self.gxp_arenas[slot].generation;
                let used: Vec<(u64, wgpu::TextureFormat, u32, u32)> =
                    gxp_prepared.iter().map(|p| (p.key, p.format, p.samples, p.cull)).collect();
                self.gxp.ensure_ubo_bgs(device, &self.gxp_arenas[slot].ubo, slot, generation, &used);
            }

            self.last_phases.upload_ms = t_upload.ms();

            // 3. One render pass over the whole scene. When supersampling, the scene is drawn
            //    into the offscreen `scale x` target (built here) and a resolve pass below
            //    box-downsamples it into the caller's view; otherwise it is drawn straight in.
            let t_pass = Stopwatch::start();
            enc(&ENC.passes, 1);
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gxm-scene"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        depth_slice: None,
                        // The multisample resolve, when this pass has multisampled
                        // attachments: ending the pass writes the resolved image into the
                        // stored-size texture every later pass samples. `StoreOp::Store` below
                        // still applies to the multisampled attachment itself, which has to
                        // survive so a SECOND pass into the same target composes onto this
                        // one's samples rather than onto its resolve.
                        resolve_target: resolve,
                        ops: wgpu::Operations {
                            load: match clear {
                                Some(c) => wgpu::LoadOp::Clear(wgpu::Color {
                                    r: c[0] as f64 / 255.0,
                                    g: c[1] as f64 / 255.0,
                                    b: c[2] as f64 / 255.0,
                                    a: c[3] as f64 / 255.0,
                                }),
                                None => wgpu::LoadOp::Load,
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            // Discarded by default - nothing reads a depth attachment once its
                            // pass is over. A pass whose depth a LATER pass samples has to keep
                            // it, and only that pass pays the store.
                            store: if self.keep_depth {
                                wgpu::StoreOp::Store
                            } else {
                                wgpu::StoreOp::Discard
                            },
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if order.is_empty() {
                    // Clear-only pass; the descriptor above already did the work. The phase
                    // timing is still closed out, so a caller reading it never sees a stale
                    // value from an earlier, busier frame.
                    self.last_phases.pass_ms = t_pass.ms();
                    self.chain_phases.add(self.last_phases);
                    return;
                }
                // Draw in submission order, switching between the fixed-function arenas and the
                // recompiled per-draw resources. The fixed-function handles are unwrapped only
                // inside a Fixed arm, where `items` is non-empty so the arenas were uploaded.
                let ubo_bind = self.ubo_bind.as_ref();
                let vbo = self.vbo.as_ref();
                let ibo = self.ibo.as_ref();
                // Unwrapped only inside a `Gxp` arm, where `gxp_prepared` is non-empty so this
                // pass took an arena slot above.
                let gxp_arena = gxp_slot.map(|s| &self.gxp_arenas[s]);
                // The guest's viewport is per-draw state, and `set_viewport` is sticky, so a
                // draw that wants the whole target after one that did not must SAY so. The
                // pass starts at the full rect (wgpu's default), and the tracker below issues
                // a change only when the requested rect actually differs - which on a title
                // whose every pass is fullscreen means it never issues one at all.
                let (att_w, att_h) = (surf_w * attach_scale, surf_h * attach_scale);
                let full = (0.0f32, 0.0f32, att_w as f32, att_h as f32);
                let mut cur_vp = full;
                // The guest's REGION CLIP, tracked exactly as the viewport is and for the
                // same reason: `set_scissor_rect` is sticky within a pass, so a draw that
                // wants the whole attachment after a scissored one has to say so. The pass
                // starts at wgpu's default (the whole attachment), and a title that never
                // sets a region clip therefore issues no scissor call at all.
                let full_sc = (0u32, 0u32, att_w, att_h);
                let mut cur_sc = full_sc;
                // The clip the previous draw carried, so the report below runs on a CHANGE
                // rather than per draw. `report_region_clip_applied` takes a mutex and hits a
                // hash set; at ~500 draws a frame that would make the instrument a measurable
                // part of the `pass` phase it sits in, which is the one thing a diagnostic in
                // this loop must not be.
                let mut last_clip: Option<RegionClip> = None;
                // >>> REDUNDANT-STATE ELIMINATION WAS BUILT HERE AND REMOVED. Do not re-add it
                // without measuring the PHASE first.
                //
                // Skipping a `set_pipeline`/`set_bind_group` whose value is already bound is
                // safe (pass state is sticky within a pass) and it demonstrably works: MEASURED
                // on the user's phone, a race frame went from 975 pipeline sets to 458, and a
                // static screen came out BIT-IDENTICAL (0.00% of pixels, max channel delta 0).
                // It was still deleted, because it bought NOTHING - render 22.9 ms with it
                // against 22.7 ms without, same device, same screen, same build.
                //
                // The reason is in the phase split, and it is the part worth keeping: these
                // calls are counted in the `pass` phase, and `pass` is 1.1 ms of a 19.2 ms
                // encode, against `prepare` 10.1 ms and `upload` 7.6 ms. Halving the call count
                // of the smallest phase cannot move a frame however large the count looks next
                // to the draw count. RANK BY PHASE FIRST, THEN BY COUNT: counting alone picked
                // the one item in the frame that could not pay.
                //
                // Counted in LOCALS and folded once after the loop, not per call: these are the
                // hottest lines in the renderer and an atomic per `set_bind_group` would make
                // the instrument a measurable part of what it measures.
                let (mut n_vp, mut n_pipe, mut n_bg, mut n_vb, mut n_draw) = (0u64, 0u64, 0u64, 0u64, 0u64);
                let mut n_sc = 0u64;
                for (oi, e) in order.iter().enumerate() {
                    let want = match e {
                        Enc::Gxp(idx) => {
                            match gxm_viewport_rect(&gxp_prepared[*idx].viewport, surf_w, surf_h) {
                                Some(r) => {
                                    let s = attach_scale as f32;
                                    (r.0 * s, r.1 * s, r.2 * s, r.3 * s)
                                }
                                None => full,
                            }
                        }
                        // The fixed-function path packs its own screen-space geometry and has
                        // never carried a viewport; it means the whole target.
                        Enc::Fixed(_) => full,
                    };
                    if want != cur_vp {
                        pass.set_viewport(want.0, want.1, want.2, want.3, 0.0, 1.0);
                        cur_vp = want;
                        n_vp += 1;
                    }
                    // The guest's hardware scissor. `rect_in` works in ATTACHMENT texels, so
                    // the clip is scaled with the pass exactly as the viewport is.
                    if last_clip != Some(clips[oi]) {
                        last_clip = Some(clips[oi]);
                        super::report_region_clip_applied(clips[oi], surf_w, surf_h);
                    }
                    let want_sc = clips[oi]
                        .rect_in(surf_w, surf_h)
                        .map(|(x, y, w, h)| {
                            (x * attach_scale, y * attach_scale, w * attach_scale, h * attach_scale)
                        })
                        .unwrap_or(full_sc);
                    if want_sc != cur_sc {
                        // A zero-area scissor is legal in wgpu and draws nothing, which is
                        // exactly what SCE_GXM_REGION_CLIP_ALL asks for.
                        pass.set_scissor_rect(want_sc.0, want_sc.1, want_sc.2, want_sc.3);
                        cur_sc = want_sc;
                        n_sc += 1;
                    }
                    match e {
                        Enc::Fixed(i) => {
                            let it = &items[*i];
                            let (ubo_bind, vbo, ibo) = (ubo_bind.unwrap(), vbo.unwrap(), ibo.unwrap());
                            // A fixed-function draw on a gamma-correct surface needs the sRGB
                            // variant, and one in a multisampled pass needs the multisampled
                            // variant, for the same reason a recompiled one does: a pipeline is
                            // bound to its attachments' format AND sample count. The two never
                            // combine - a gamma target is refused multisampling - so this is a
                            // three-way choice, not a matrix.
                            let (op, bl) = match (&self.srgb, target_format == self.color_format) {
                                (Some((o, b)), false) => (o, b),
                                _ if samples > 1 => (&self.opaque_ms, &self.blend_ms),
                                _ => (&self.opaque, &self.blend),
                            };
                            pass.set_pipeline(if it.opaque { op } else { bl });
                            pass.set_bind_group(0, ubo_bind, &[it.uniform_offset]);
                            pass.set_bind_group(1, self.bind_for(it.bind), &[]);
                            pass.set_vertex_buffer(0, vbo.slice(it.v_off..it.v_off + it.v_len));
                            pass.set_index_buffer(ibo.slice(it.i_off..it.i_off + it.i_len), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..it.index_count, 0, 0..1);
                            n_pipe += 1;
                            n_bg += 2;
                            n_vb += 2;
                            n_draw += 1;
                        }
                        Enc::Gxp(idx) => {
                            let p = &gxp_prepared[*idx];
                            // >>> A PIPELINE THE DEVICE REFUSED IS NOT BOUND, because binding it
                            // would invalidate the pass, the command buffer and therefore every
                            // OTHER draw in the frame. See `note_device_error`, which is what
                            // learns the key, and which reports it loudly the first time.
                            if super::gxp_pair_poisoned(p.key) {
                                super::POISONED_DRAWS
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                continue;
                            }
                            let arena = gxp_arena.unwrap();
                            // Geometry the renderer has resident is bound where it LIVES; only
                            // what changed this frame comes from the pass arena. Both handles
                            // are read here, at encode time, which is why the resident heap may
                            // never be recreated mid-frame - see [`ResidentHeap`].
                            let gxp_vbo = if p.v_resident {
                                self.gxp.resident_v.buf.as_ref().expect("a resident slice implies its buffer")
                            } else {
                                &arena.vbo
                            };
                            let gxp_ibo = if p.i_resident {
                                self.gxp.resident_i.buf.as_ref().expect("a resident slice implies its buffer")
                            } else {
                                &arena.ibo
                            };
                            let slot = gxp_slot.unwrap();
                            let pipe = self.gxp.pipeline(p.key, p.format, p.samples, p.cull);
                            pass.set_pipeline(&pipe.pipeline);
                            // group0/group1 belong to the PAIR and take this draw's byte offset
                            // into the pass's uniform arena; a stage with no uniforms has an
                            // empty bind group, which takes no dynamic offsets at all.
                            let dyn_off = |lanes: u32, off: u32| if lanes == 0 { Vec::new() } else { vec![off] };
                            // group 0's dynamic offsets in BINDING order: the SA block, then
                            // the guest-memory window when this pipeline declares one.
                            let mut g0_offs = dyn_off(pipe.vsa_lanes, p.u_off[0]);
                            if pipe.mem_bind_bytes > 0 {
                                g0_offs.push(p.u_off[2]);
                            }
                            pass.set_bind_group(
                                0,
                                self.gxp.ubo_bg(slot, p.key, p.format, p.samples, 0),
                                &g0_offs,
                            );
                            pass.set_bind_group(
                                1,
                                self.gxp.ubo_bg(slot, p.key, p.format, p.samples, 1),
                                &dyn_off(pipe.fsa_lanes, p.u_off[1]),
                            );
                            pass.set_bind_group(2, &p.bg2, &[]);
                            pass.set_bind_group(3, &p.bg3, &[]);
                            n_pipe += 1;
                            n_bg += 4;
                            pass.set_vertex_buffer(0, gxp_vbo.slice(p.v_off..p.v_off + p.v_len));
                            pass.set_index_buffer(
                                gxp_ibo.slice(p.i_off..p.i_off + p.i_len),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.draw_indexed(0..p.index_count, 0, 0..1);
                            n_vb += 2;
                            n_draw += 1;
                        }
                    }
                }
                enc(&ENC.viewport_sets, n_vp);
                enc(&ENC.scissor_sets, n_sc);
                enc(&ENC.pipeline_sets, n_pipe);
                enc(&ENC.bind_group_sets, n_bg);
                enc(&ENC.vertex_buffer_sets, n_vb);
                enc(&ENC.draw_calls, n_draw);
            }
            self.last_phases.pass_ms = t_pass.ms();
            self.chain_phases.add(self.last_phases);
            // The arenas are NOT retired here any more: this pass's slot keeps them for the
            // next frame's pass of the same ordinal. Only a slot that had to GROW retires a
            // buffer, and `ensure_gxp_arena` puts that one in the graveyard itself.
        }

        /// What the last [`GxmRenderer::encode_chain`] spent, phase by phase, over EVERY pass
        /// of the frame. Reporting one pass instead described the composite and hid the world.
        pub fn last_phases(&self) -> EncodePhases {
            self.chain_phases
        }

        /// Every offscreen target this renderer holds, as `(guest address, texture, w, h)`.
        ///
        /// For a host that can read pixels back (the native headless oracle). A frame is a
        /// CHAIN of passes and only the last one reaches the caller's view, so when the
        /// finished frame is black the question is which pass is empty - and every failure
        /// mode looks identical in the composite. `VITASLOP_CHAIN_LIMIT` answers that one
        /// pass per run; this answers all of them in one, which is the difference between a
        /// bisect and a look. The colour textures already carry `COPY_SRC` for the snapshot
        /// path, so exposing them costs nothing.
        pub fn rtt_targets(&self) -> Vec<(u32, &wgpu::Texture, u32, u32)> {
            let mut v: Vec<_> =
                self.rtt.iter().map(|(&a, s)| (a, &s.color, s.width, s.height)).collect();
            v.sort_by_key(|t| t.0);
            v
        }

        /// The guest-encoded DEPTH companions of the targets that have one, as
        /// `(colour address, texture, w, h)`.
        ///
        /// Exposed for the same reason as [`Self::rtt_targets`]: when a pass that reads a
        /// depth buffer renders black, the first question is whether the depth it read holds
        /// anything, and a converted buffer nobody can look at is a buffer nobody can debug.
        pub fn rtt_depth_targets(&self) -> Vec<(u32, &wgpu::Texture, u32, u32)> {
            let mut v: Vec<_> = self
                .rtt
                .iter()
                .filter_map(|(&a, s)| s.gxm_depth.as_ref().map(|d| (a, &d.tex, s.width, s.height)))
                .collect();
            v.sort_by_key(|t| t.0);
            v
        }
    }
}
