//! Textures for the recompiled path: what a sampler unit binds, sampler state, the upload in
//! each of its forms, and the ledger that catches a dropped GPU texture.

use super::*;

/// Record that fingerprint `pre` resolved to content key `sel`, for the rest of this pass.
///
/// Bounded like every other content-addressed cache here, and by the same argument: the
/// key is a fingerprint, dropping an entry costs a re-resolution and never an answer. The
/// bound is per PASS, and a pass with more distinct materials than this is not a pass this
/// cache can help anyway.
/// A `TextureView`'s identity as a key word: wgpu compares and hashes views by the
/// resource behind them, and a cached group holding a clone keeps that identity from
/// being reused while the entry lives.
pub(super) fn view_identity(v: &wgpu::TextureView) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

pub(super) fn remember_sampler_pre(map: &mut HashMap<u64, u64>, pre: u64, sel: u64) {
    if map.len() >= SAMPLER_BG_CACHE_CAP {
        map.clear();
    }
    map.insert(pre, sel);
}

/// What one sampler unit will bind, recorded by `make_sampler_bg`'s resolution loop so
/// that the actual `TextureView` is taken only if the finished bind group turns out not to
/// be cached. See the `plan` declaration there for the measurement.
///
/// An ADDRESS rather than a view for the two render-target cases, and a cache KEY rather
/// than a view for the ordinary one: all three are `Copy`, so recording them costs nothing
/// and the group's identity (`sel`) is already folded from the same facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Bind {
    /// A depth buffer this frame rendered, matched exactly by its guest address.
    Depth(u32),
    /// A colour target this frame rendered, by the address `rendered_alias` resolved to.
    Rendered(u32),
    /// The same target's LINEAR view, for a texture that sets no gamma mode over a
    /// gamma-correct target - see the `rendered_linear` arm of `make_sampler_bg`.
    RenderedLinear(u32),
    /// The other direction: a LINEAR target's sRGB view, for a texture that DOES set a gamma
    /// mode over it - the sampler decodes whatever bytes it reads. See `rendered_srgb`.
    RenderedSrgb(u32),
    /// A CUBE MAP assembled from the six targets its faces were rendered into, by the
    /// address of face 0. See [`GxmRenderer::rtt_cubes`].
    RenderedCube(u32),
    /// A 64-bit colour target this frame rendered as RAW words, by its address, bound to a
    /// `SamplerDim::Raw` slot. See [`GxmRenderer::rtt_rendered_raw`].
    RenderedRaw((u32, u32, u32)),
    /// An ordinary uploaded texture, by its view-cache key.
    Cached((u64, SamplerDim)),
    /// No usable texture: a neutral view, built per draw, which makes the group per-frame.
    Fallback(SamplerDim),
}

/// The texture dimension a recompiled fragment's sampler binding needs. It comes from the
/// GXP container (the sampler parameter's cube flag plus the SMP coordinate count), and it
/// must match on both sides: the WGSL declares the type, the bind-group layout declares the
/// view dimension, and the bound texture must be able to supply it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum SamplerDim {
    Two,
    Three,
    Cube,
    /// A 2D binding read as RAW 32-bit words (`texture_2d<u32>`, `Uint` sample type, a
    /// non-filtering sampler): the shader was linked with this unit in
    /// `LinkOptions::raw_units`, and the bound view is a [`RAW64_FORMAT`] render target.
    Raw,
    /// A 2D FLOAT binding the draw fills with a texture on the 32-bit float seam
    /// ([`TexelSeam::Rg32Float`], the guest's F32F32): `texture_2d<f32>` in the WGSL as
    /// always, but laid out UNFILTERABLE with a non-filtering sampler, because core WebGPU
    /// cannot filter 32-bit floats. The sampler is nearest - which is what point sampling a
    /// data texel returns, and the only mode a matrix palette is fetched with.
    TwoF32,
}

impl SamplerDim {
    pub(super) fn view_dimension(self) -> wgpu::TextureViewDimension {
        match self {
            SamplerDim::Two | SamplerDim::Raw | SamplerDim::TwoF32 => wgpu::TextureViewDimension::D2,
            SamplerDim::Three => wgpu::TextureViewDimension::D3,
            SamplerDim::Cube => wgpu::TextureViewDimension::Cube,
        }
    }

    /// The bind-group-layout entries for one texture + sampler pair of this dimension:
    /// the WGSL declares the matching type, so the two must agree here and nowhere else.
    pub(super) fn layout_entries(self, binding: u32, visibility: wgpu::ShaderStages) -> [wgpu::BindGroupLayoutEntry; 2] {
        let raw = matches!(self, SamplerDim::Raw);
        let unfilterable = raw || matches!(self, SamplerDim::TwoF32);
        [
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Texture {
                    sample_type: if raw {
                        wgpu::TextureSampleType::Uint
                    } else {
                        wgpu::TextureSampleType::Float { filterable: !unfilterable }
                    },
                    view_dimension: self.view_dimension(),
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: binding + 1,
                visibility,
                ty: wgpu::BindingType::Sampler(if unfilterable {
                    wgpu::SamplerBindingType::NonFiltering
                } else {
                    wgpu::SamplerBindingType::Filtering
                }),
                count: None,
            },
        ]
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
pub(super) fn gxm_addr_mode(mode: u32) -> wgpu::AddressMode {
    match mode {
        1 | 3 => wgpu::AddressMode::MirrorRepeat, // MIRROR, MIRROR_CLAMP
        2 | 5..=7 => wgpu::AddressMode::ClampToEdge, // CLAMP + the three border clamps
        _ => wgpu::AddressMode::Repeat,          // REPEAT, REPEAT_IGNORE_BORDER
    }
}

/// Folded into the `u` address mode of a sampler-mode key for a `SamplerDim::Raw` slot:
/// that slot's layout is NON-FILTERING, and a sampler with ANY linear filter (the trilinear
/// mipmap filter below included) is refused there by validation. The raw load never uses
/// the sampler, so an all-nearest one is exactly what the slot wants.
pub(super) const RAW_SAMPLER_FLAG: u32 = 0x8000_0000;

/// Folded into the `u` address mode of a sampler-mode key when the guest DISABLED mip
/// filtering (`SceGxmTextureMipFilter` 0): the level is still chosen per pixel, but the
/// NEAREST one is read rather than a blend of two.
///
/// # Why it matters more than a softer or sharper picture
/// A texture can carry DATA in a channel. A baseball title's player material reads a colorization ROW
/// out of its base texture's 4-bit alpha - `floor(alpha * 15.99)` - and the guest turned
/// mip filtering off for it. Blending two levels hands that `floor` a fraction of two
/// indices, and the leg guards came out speckled with other rows' colours (red and white
/// dots on a black guard, batter and catcher alike).
pub(super) const MIP_NEAREST_SAMPLER_FLAG: u32 = 0x4000_0000;

/// The sampler-mode key of a guest texture: its filter, its address modes, and whether it
/// blends between mip levels. See [`MIP_NEAREST_SAMPLER_FLAG`].
pub(super) fn sampler_mode(t: &GxmTexture) -> (bool, u32, u32) {
    let mip = if t.mip_filter == 0 { MIP_NEAREST_SAMPLER_FLAG } else { 0 };
    (t.filter_linear, t.addr_mode_u | mip, t.addr_mode_v | (t.lod_min.min(15) << LOD_MIN_SAMPLER_SHIFT))
}

/// Where a sampler-mode key's `v` carries the texture's LOD min (4 bits) above its address
/// mode. See [`GxmTexture::lod_min`].
pub(super) const LOD_MIN_SAMPLER_SHIFT: u32 = 8;

pub(super) fn make_gxp_sampler(device: &wgpu::Device, linear: bool, u: u32, v: u32) -> wgpu::Sampler {
    let raw = u & RAW_SAMPLER_FLAG != 0;
    let mip_nearest = raw || u & MIP_NEAREST_SAMPLER_FLAG != 0;
    let f = if linear && !raw { wgpu::FilterMode::Linear } else { wgpu::FilterMode::Nearest };
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(if raw { "gxp-sampler-raw" } else { "gxp-sampler" }),
        address_mode_u: gxm_addr_mode(u & !(RAW_SAMPLER_FLAG | MIP_NEAREST_SAMPLER_FLAG)),
        address_mode_v: gxm_addr_mode(v & ((1 << LOD_MIN_SAMPLER_SHIFT) - 1)),
        address_mode_w: wgpu::AddressMode::Repeat,
        lod_min_clamp: ((v >> LOD_MIN_SAMPLER_SHIFT) & 0xf) as f32,
        mag_filter: f,
        min_filter: f,
        // A minified surface reads a level sized for it (see `build_mip_chain`), and blends
        // two levels only when the guest asked for that - see [`MIP_NEAREST_SAMPLER_FLAG`].
        mipmap_filter: if mip_nearest { wgpu::MipmapFilterMode::Nearest } else { wgpu::MipmapFilterMode::Linear },
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
pub(super) fn build_mip_chain(w: u32, h: u32, layers: u32, level0: &[u8]) -> (Vec<u8>, u32) {
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

/// The SLOT an uploaded texture belongs to: the guest texture it came from, bound the way
/// this binding binds it - everything the cache key folds EXCEPT the pixel buffer's
/// identity. See `GxpLive::view_slots`.
///
/// `None` for a texture with no guest address: a null data pointer decodes to a 1x1 zero
/// texel and every one of them would otherwise share a slot and evict each other.
/// >>> LIVE GPU TEXTURES AGAINST CACHE ENTRIES, so a handle that is dropped instead of
/// >>> destroyed is REPORTED rather than discovered as a dead worker.
///
/// In a browser a `wgpu::Texture` is a `GPUTexture` in JavaScript: dropping the Rust handle
/// makes it collectable, not freed. So a path that takes a texture out of the cache and then
/// fails to either re-insert it or `destroy()` it leaks GPU memory silently, at whatever
/// rate the frame supersedes textures - and it ends with a device that has nothing left,
/// where a FORTY-EIGHT BYTE `createBuffer` fails and the panic names the innocent caller
/// that happened to allocate next.
///
/// That is not hypothetical: it shipped for one phone run. Nothing else could see it - the
/// byte accounting was correct (the entry left the cache), the eviction counters read zero
/// (nothing was evicted), and the working-set report only ever grew by 1 MB.
///
/// The invariant is simple and holds by construction: every texture this renderer creates is
/// either IN the view cache or has been destroyed. A drift of more than a frame's worth of
/// in-flight supersedes means a handle went missing.
pub(super) fn report_texture_handle_drift(cached: usize) {
    use std::sync::atomic::Ordering::Relaxed as Rx;
    let (created, destroyed) = (LEDGER_CREATED.load(Rx), LEDGER_DESTROYED.load(Rx));
    let (inserted, removed) = (LEDGER_INSERTED.load(Rx), LEDGER_REMOVED.load(Rx));
    let caches = LEDGER_CACHES.load(Rx);
    // >>> WHAT THE CACHE SHOULD HOLD, from the ledger alone. Every creation is put in the
    // cache and every removal takes one out, so the map's size is `inserted - removed`, and
    // comparing THAT against `cached` is a statement about one map on one time base.
    // >>> THE INVARIANT IS BETWEEN TWO LEDGER TERMS, NOT AGAINST ONE MAP'S LENGTH.
    //
    // A texture is ALIVE if it was created and not destroyed, and it is HELD if it was put
    // in a view cache and not taken out. Every creation is inserted and every removal is
    // either destroyed or re-inserted, so these two must be equal - and neither depends on
    // how many caches exist or on which one a caller happens to hold.
    let live = created.saturating_sub(destroyed);
    let held = inserted.saturating_sub(removed);
    if live == held {
        return;
    }
    use std::sync::atomic::{AtomicU64, Ordering};
    static SAID: AtomicU64 = AtomicU64::new(0);
    // On a doubling ladder: this is a leak, so it grows, and one line per frame would bury
    // the run it is trying to explain.
    let prev = SAID.load(Ordering::Relaxed);
    if prev != 0 && live < prev * 2 {
        return;
    }
    SAID.store(live.max(1), Ordering::Relaxed);
    report_warn!(
        "gxm textures: {live} GPU textures are LIVE but the view caches hold {held}. Every texture this renderer creates is meant to be in a cache or destroyed, so the difference is handles that were dropped instead - which in a browser is memory the collector releases whenever it likes, and which ends as an out-of-memory failure on some unrelated allocation. created={created} destroyed={destroyed} put-in-a-cache={inserted} taken-out={removed}, over {caches} cache(s); the renderer asking holds {cached} of them"
    );
}

/// >>> THE HANDLE LEDGER'S OWN COUNTERS, AND THEY ARE DELIBERATELY NOT `ENC` ONES.
///
/// `take_encode_work()` SWAPS every `ENC` counter to zero - the browser drains them every
/// PRESENT and the desktop every sampled frame - because they are a per-window work report.
/// [`report_texture_handle_drift`] compares its terms against `views.len()`, which is
/// ABSOLUTE, so reading them from `ENC` compared a few frames' creations against the whole
/// cache and called the difference a leak.
///
/// MEASURED on a baseball title's gameplay before this split: `created=1331 destroyed=0
/// put-in-the-cache=1331 map-len-at-last-insert=307 evicted=0 superseded=23`, with EVERY
/// insert growing the map. Those numbers cannot all be true of one map - and they were not:
/// the creations were a long window's worth and the removals had been drained away in
/// earlier windows. The report named 1,024 leaked handles on that arithmetic for as long as
/// it has existed [[vitaslop-compare-like-windows]].
///
/// A ledger that must close over a run needs counters that live as long as the run.
pub(super) static LEDGER_CREATED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) static LEDGER_DESTROYED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) static LEDGER_INSERTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) static LEDGER_REMOVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// How many view CACHES have existed. A second one means the first one's textures were
/// abandoned wholesale, which is a leak no per-entry counter can see.
pub(super) static LEDGER_CACHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One view cache came into existence. See [`LEDGER_CACHES`].
pub(crate) fn note_view_cache_created() {
    LEDGER_CACHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn note_texture_created() {
    enc(&ENC.textures_created, 1);
    LEDGER_CREATED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// One entry taken OUT of the recompiler's view cache, whatever becomes of its texture.
/// The ledger's fourth term: an entry that leaves the map without being destroyed is the
/// leak, and without counting the leaving there is no way to tell that from an entry that
/// never arrived.
pub(crate) fn note_view_cache_removed() {
    LEDGER_REMOVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// One texture released back to the driver, for the ledger [`report_texture_handle_drift`]
/// reads.
///
/// >>> A HALF-KEPT LEDGER CRIES WOLF ABOUT THE ONE THING IT EXISTS TO CATCH.
///
/// The creations were counted in three files and the destroys in one, so every texture the
/// transcoder released on a refusal path was invisible. MEASURED on a baseball title's
/// gameplay: the drift report read **created=1331 destroyed=0 against 307 cache entries**
/// and named 1,024 leaked handles - on a run where `texenc::expand_rgba8` had destroyed its
/// caller's spare at every one of its eight refusals and at every shape mismatch, correctly,
/// and simply never said so.
///
/// A leak warning that fires when nothing leaked is worse than no warning at all: it is the
/// one report anybody would act on, and acting on it means hunting a handle that was
/// released. Count every release of a texture whose creation was counted - and ONLY those,
/// because a destroy with no matching create drives the ledger the other way and hides a
/// real leak behind a negative.
pub(crate) fn note_texture_destroyed() {
    enc(&ENC.textures_destroyed, 1);
    LEDGER_DESTROYED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn view_slot_key(t: &GxmTexture, want: SamplerDim) -> Option<(u64, SamplerDim)> {
    if t.data_addr == 0 {
        return None;
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in [
        t.data_addr as u64,
        t.base_format as u64,
        t.swizzle as u64,
        (t.width as u64) << 32 | t.height as u64,
        t.faces as u64,
        t.filter_linear as u64,
        (t.addr_mode_u as u64) << 32 | t.addr_mode_v as u64,
        t.gamma as u64,
    ] {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
        h = h.rotate_left(23);
    }
    Some((h, want))
}

/// Upload a decoded [`GxmTexture`] to a GPU texture for the recompiler path, in whichever
/// of the two seams it was decoded onto (see [`TexelSeam`]).
/// `reuse` is the GPU texture the PREVIOUS contents of this same guest texture were
/// uploaded into, when the caller has established that nothing this frame is still reading
/// it. Only the GPU expansion takes it today - see `make_sampler_bg`, and see there for why
/// this matters more than it looks.
pub(super) fn upload_gxp_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    t: &GxmTexture,
    bc: BlockFamily,
    texenc: &crate::texenc::Transcoder,
    reuse: Option<wgpu::Texture>,
    // >>> HOW MANY LAYERS THE VIEW NEEDS, which is not always how many faces the guest
    // >>> texture HAS.
    //
    // A program whose container declares a CUBE sampler over a unit the guest bound a FLAT
    // texture to has to sample something: a cube view needs six layers and the texture has
    // one, so the one image is REPLICATED into all six. See the `usable` test in
    // `make_sampler_bg` for the measurement and for why this beats refusing the draw.
    want_layers: u32,
    // The view-cache key a DEFERRED ETC2 encode is swapped in under when it finishes, or
    // `None` where the caller cannot swap one in - see `GxpLive::pump_deferred_encodes`.
    defer_key: Option<u64>,
) -> wgpu::Texture {
    let (w, h) = (t.width.max(1), t.height.max(1));
    // A cube map uploads as six array layers; the view below then reads them as a cube.
    let faces = t.faces.max(1);
    let layers = want_layers.max(faces);
    // How many times each face's bytes are written. Exactly 1 for every ordinary texture,
    // so nothing that already worked copies anything extra.
    let repeat = (layers / faces).max(1) as usize;
    /// The bytes for `layers` layers, from the bytes for `faces` of them: LayerMajor order,
    /// so a whole-buffer repeat IS the layer sequence.
    fn replicate(bytes: &[u8], repeat: usize) -> std::borrow::Cow<'_, [u8]> {
        if repeat <= 1 {
            return std::borrow::Cow::Borrowed(bytes);
        }
        std::borrow::Cow::Owned(bytes.repeat(repeat))
    }
    // >>> A VIDEO FRAME CONVERTS ON THE GPU, and falls through to the CPU decode below
    // if this adapter or this shape is outside what the shader covers - same picture,
    // more cost. See `texenc::Transcoder::convert_yuv420p2`.
    if let Some(planes) = t.planar_yuv.as_ref() {
        let source = crate::texenc::PlanarYuv {
            width: planes.width,
            height: planes.height,
            luma_stride: planes.luma_stride,
            chroma_stride: planes.chroma_stride,
            chroma_offset: planes.chroma_offset,
            swap_chroma: planes.swap_chroma,
            profile: planes.profile,
            data: &planes.data,
        };
        if let Some(tex) = texenc.convert_yuv420p2(device, queue, &source) {
            enc(&ENC.tex_uploaded, 1);
            enc_tex_upload(planes.data.len() as u64);
            return tex;
        }
        report_yuv_gpu_refused();
    }
    // >>> AN UNCOMPRESSED TEXTURE IS UN-SWIZZLED ON THE GPU, and falls through to the CPU
    // decode below if this shape is outside what the shader covers - same picture, more
    // cost. See `texenc::Transcoder::expand_rgba8`.
    //
    // Asked BEFORE the passthrough because the two are disjoint: the passthrough is for
    // block formats, this is for the formats that have no blocks at all, and the ordering
    // only decides which test runs first on a texture neither claims.
    // >>> EVERY PATH BELOW OWNS `reuse` AND MUST RELEASE IT.
    //
    // Only the GPU expansion can write into an existing texture. The caller does not know
    // which path a texture will take, so anything the expansion does not consume is
    // destroyed here rather than dropped - see `expand_rgba8` for what dropping a
    // `GPUTexture` in a browser actually does, and for the crash it produced.
    let mut reuse = reuse;
    // The two GPU transcode paths below produce exactly the layers the SOURCE has, so a
    // replicating upload takes the CPU decode instead. That costs one small texture's decode
    // on a path that only fires for a flat texture bound to a cube sampler.
    if let Some(raw) = t.raw.as_ref().filter(|_| repeat == 1) {
        if let Some(tex) = texenc.expand_rgba8(device, queue, raw, t.gamma, reuse.take()) {
            enc(&ENC.tex_uploaded, 1);
            enc_tex_upload(raw.src.len() as u64);
            enc(&ENC.tex_encoded_on_gpu, 1);
            return tex;
        }
        enc(&ENC.tex_gpu_encode_refused, 1);
        // Read on the same thread, immediately after the refusal that set it - see
        // `texenc::LAST_RAW_REFUSAL`.
        report_gpu_transcode_refused(t.base_format, crate::texenc::last_raw_refusal());
    }
    // Not consumed by the expansion - a refusal above, or a format that never offered it.
    if let Some(t) = reuse {
        note_texture_destroyed();
        t.destroy();
    }
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
        if let (CompressedData::Gpu(plan), 1) = (&c.data, repeat) {
            // DEFERRED where the caller can swap the finished texture in (`defer_key`),
            // inline otherwise - see `super::tex_encode_defer`.
            let deferred = defer_key
                .filter(|_| crate::gpu::tex_encode_defer())
                .and_then(|key| texenc.run_deferred(device, queue, c, plan, t.gamma, key));
            if let Some(tex) = deferred.or_else(|| texenc.run(device, queue, c, plan, t.gamma)) {
                enc(&ENC.tex_etc2_encoded, 1);
                enc(&ENC.tex_uploaded, 1);
                enc_tex_upload(c.byte_len() as u64);
                enc(&ENC.tex_uploaded_compressed, 1);
                enc(&ENC.tex_encoded_on_gpu, 1);
                return tex;
            }
            enc(&ENC.tex_gpu_encode_refused, 1);
            // The BLOCK transcoder, which records no reason of its own yet. Said so rather
            // than borrowing the uncompressed path's - see `report_gpu_transcode_refused`.
            report_gpu_transcode_refused(t.base_format, "not recorded (the block transcoder)");
        } else if let CompressedData::Cpu(bytes) = &c.data {
        // (A GPU plan with `repeat > 1` - a flat texture bound to a CUBE sampler - lands
        // neither here nor in the arm above: the transcoder makes only the source's own
        // layers, so it takes the ordinary decode below. This was an `unreachable!` that
        // PANICKED the phone's football-title run at the coin toss, job 067.)
        enc(&ENC.tex_uploaded, 1);
        enc_tex_upload(c.byte_len() as u64);
        enc(&ENC.tex_uploaded_compressed, 1);
        // The compressed data's OWN dimensions - see `CompressedUpload::width`. They equal
        // the guest's today; using `w`/`h` here anyway would let anything that ever changed
        // that declare a 2048x2048 texture over a smaller buffer, which is a validation
        // error at best and a read past the buffer at worst.
        let (w, h) = (c.width.max(1), c.height.max(1));
        note_texture_created();
        let bytes = replicate(bytes, repeat);
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
            &bytes,
        );
        }
    }
    let bpt = t.texel.bytes_per_texel();
    // Guard against a short pixel buffer (a not-fully-decoded format): pad to the full size.
    let need = (w as usize) * (h as usize) * (faces as usize) * bpt;
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
    let data = replicate(&data, repeat).into_owned();
    let (data, mip_level_count) = if !mips_for_texture(t) {
        (data, 1)
    } else {
        build_mip_chain(w, h, layers, &data)
    };
    // Count the bytes that ACTUALLY cross into the driver, mips included. The decoder's
    // output is the level-0 size; the mip chain adds a third more, and a tally taken from
    // the decoder would under-report the upload by exactly that much.
    enc(&ENC.tex_uploaded, 1);
    enc_tex_upload(data.len() as u64);
    note_texture_created();
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
                (TexelSeam::Rg32Float, _) => wgpu::TextureFormat::Rg32Float,
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

/// Say, once per (pair, unit), that a CUBE sampler was given a FLAT texture and the one
/// image was replicated into six faces. See the `usable` test in `make_sampler_bg`.
///
/// Reported at `warn` and never suppressed: the draw renders, which is the point, but the
/// texel it samples is not exactly the one the hardware would take, and a substitution
/// nothing announces is indistinguishable from a correct decode.
pub(super) fn report_flat_texture_on_cube_unit(key: u64, unit: u8, base_format: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u64, u8)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert((key, unit)) {
        return;
    }
    report_warn!(
        "gxp sampler: pair {key:016x} declares unit {unit} as a CUBE and the guest bound a              FLAT texture (base format {base_format:#x}) - the one image is replicated into six              faces so the draw RENDERS instead of being dropped. The texel a cube view takes for              a direction is not the one 2-D addressing would take, so a wrong colour on this              pair's mesh starts here."
    );
}

/// Say, once, that a video frame could not be converted on the GPU and is going through
/// the CPU decode instead. Not a failure - the picture is the same - but it is the
/// difference between a movie costing nothing per frame and costing megabytes of
/// conversion, so a run that quietly does the slow one should say so.
pub(super) fn report_yuv_gpu_refused() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        report_warn!(
            "gxm textures: a two-plane 4:2:0 (video) texture could not be converted on the GPU and is being converted on the CPU instead - the same picture, but a full RGBA expansion per frame, which for a movie is per DISPLAY frame"
        );
    }
}

/// A 1x1 (or 1x1x1) neutral-grey fallback texture view of the given dimension, for the
/// `VITASLOP_GXP_FORCE` diagnostic: bound where a sampler's real texture is unavailable so
/// the recompiled geometry still renders. Opaque alpha so an alpha-test does not discard.
pub(super) fn make_fallback_view(device: &wgpu::Device, queue: &wgpu::Queue, dim: wgpu::TextureViewDimension) -> wgpu::TextureView {
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
