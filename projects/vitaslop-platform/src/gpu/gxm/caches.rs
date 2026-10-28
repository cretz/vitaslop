//! What the renderer's caches cost and how they are bounded: texture sizing for the budget,
//! the packed-geometry and promotion maps, and the eviction every bounded cache shares.

use super::*;

/// Bytes a decoded texture occupies once uploaded, for the cache budget. RGBA8 is what
/// every upload path here produces, so this is exact rather than an estimate.
pub(super) fn texture_bytes(width: u32, height: u32) -> usize {
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
pub(super) fn uploaded_texture_bytes(bc: BlockFamily, t: &GxmTexture) -> usize {
    texture_upload_bytes(t, compressed_upload(bc, t).is_some())
}

/// [`uploaded_texture_bytes`] with the device decision already made, so the arithmetic can
/// be tested without a GPU.
pub(super) fn texture_upload_bytes(t: &GxmTexture, passthrough: bool) -> usize {
    // A texture that reaches the GPU compressed occupies exactly the blocks it is made of -
    // no estimate needed, and no 4/3 either, because the chain is already counted. A
    // GPU-built chain is priced from its geometry, which is the same number the CPU one
    // reports for the same shape; see `CompressedUpload::byte_len`.
    if passthrough
        && let Some(c) = t.compressed.as_ref() {
            return c.byte_len();
        }
    let (w, h) = (t.width.max(1) as usize, t.height.max(1) as usize);
    let layers = t.faces.max(1) as usize;
    let level0 = w * h * layers * t.texel.bytes_per_texel();
    if mips_for_texture(t) {
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
pub(super) fn compressed_upload(family: BlockFamily, t: &GxmTexture) -> Option<&CompressedUpload> {
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
///
/// `why` names the rule that declined. It is passed in rather than read here because the two
/// callers are DIFFERENT transcoders: only the uncompressed expansion records a reason, and
/// reading its thread local after the block path refused would attribute one path's refusal
/// to the other - a wrong cause printed with the same confidence as a right one.
pub(super) fn report_gpu_transcode_refused(base_format: u32, why: &str) {
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
         The picture is the same; the cost is not. The rule that declined it: {why}.",
    );
}

/// [`block_wgpu_format`], reachable from outside this module. See
/// [`super::block_wgpu_format_pub`].
pub(in crate::gpu) fn block_wgpu_format_pub(f: BlockFormat, gamma: bool) -> wgpu::TextureFormat {
    block_wgpu_format(f, gamma)
}

pub(super) fn block_wgpu_format(f: BlockFormat, gamma: bool) -> wgpu::TextureFormat {
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

/// Whether [`upload_gxp_texture`] builds a mip chain for this texture. Shared by the
/// uploader and by [`uploaded_texture_bytes`] so the budget and the upload cannot
/// disagree.
///
/// # A decoded VIDEO frame is the one texture that gets no chain
///
/// The general rule is a chain for every RGBA8 texture, and it exists to fix real
/// speckle: a title's images are minified, and level 0 alone is what makes a distant
/// road sparkle. A frame out of a video decoder is not that. It is drawn once, at 1:1,
/// by a movie player, so no level below 0 is ever sampled - and it is REPLACED every
/// frame, so the chain is built and thrown away sixty times a second. MEASURED on the
/// title-screen movie: 2.66 MB uploaded per frame, of which 0.57 MB was a chain
/// nothing read.
///
/// The guest agrees, which is what makes this a reading rather than an optimisation:
/// the movie texture declares one mip level and no filtering between levels.
pub(super) fn mips_for_texture(t: &GxmTexture) -> bool {
    if t.base_format == crate::gpu::GXM_BASE_FORMAT_YUV420P2 {
        return false;
    }
    // >>> GATING THIS ON THE GUEST'S OWN `levels`/`mip_filter` WAS TRIED 2026-08-28b AND
    // >>> REVERTED. The fields are carried (see `GxmTexture::levels`) so the next attempt
    // >>> starts with the data rather than the plumbing.
    //
    // The argument is the compressed passthrough's own, verbatim: a texture the guest gave
    // ONE level and told the hardware not to filter between levels is one the DEVICE samples
    // from its base alone, so a generated chain is filtering the Vita never did. Two paths
    // applying opposite rules to one question is exactly the kind of split this codebase
    // fixes elsewhere.
    //
    // **It was tried to explain a golf title's smudged club label, and it did not: MEASURED at
    // f2400, ZERO of the 2400 pixels in the label's own box moved.** What DID move was
    // 0.256% of the frame, confined to (282,313)-(424,466) - a patch of the golfer's lower
    // legs - and there is no evidence that patch is more faithful rather than less. A change
    // that fixes nothing demonstrable, alters a character, and risks the white-speckle
    // failure the chain exists to prevent is not one to ship on principle alone.
    //
    // What it needs is a DEVICE: the honest test is whether a 1-level atlas minified several
    // times looks sharper or worse on hardware, and that is not answerable from a desktop.
    mips_for_seam(t.texel)
}

/// Whether a chain is built for a seam, ignoring the per-texture exception above.
pub(super) fn mips_for_seam(texel: TexelSeam) -> bool {
    texel == TexelSeam::Rgba8 && {
        // Cached: asked per texture priced or uploaded.
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| crate::knobs::var("VITASLOP_GXP_MIPS").ok().as_deref() != Some("0"))
    }
}

/// Upper bound on the repacked-geometry cache, in distinct meshes. A frame here submits a
/// few hundred; the cap only fires on a title whose geometry genuinely changes every
/// frame, where the cache would not have helped anyway.
///
/// # >>> THAT LAST CLAUSE IS AN ASSUMPTION, AND A DEVICE DUMP PUT IT IN DOUBT
/// A 78-second phone run reported `packed geometry (by content) 103x/115758 entries` - the
/// cap firing every three quarters of a second and shedding a quarter of the cache each
/// time, with the map sitting at 3,077 entries when it was read. Two completely different
/// worlds produce that line:
///
///   * the title really does submit fresh geometry every frame, nothing evicted is ever
///     asked for again, and the eviction costs nothing but the walk; or
///   * its working set is simply LARGER than 4,096 meshes, in which case every pass throws
///     away geometry the next second asks for and pays a full repack plus an arena copy
///     plus a `write_buffer` to get it back - the compounding degradation this whole
///     eviction scheme exists to prevent, reintroduced by a cap that is merely too small.
///
/// A count of evictions cannot tell those apart, exactly as the texture-view cache's count
/// could not until `tex_reuploaded_after_evict` was added. [`PACKED_REPACK_AFTER_EVICT`] is
/// that same signal for this cache, and it is the one to read before this number is moved.
pub(super) const PACKED_CACHE_CAP: usize = 4096;

/// >>> AND A CAP IN ENTRIES IS NOT A BOUND ON MEMORY. THIS IS THE BOUND IN BYTES.
///
/// [`PACKED_CACHE_CAP`] counts MESHES, and a mesh here is anything from a few hundred bytes
/// to a megabyte. MEASURED on one retail title's gameplay frame, which submits ~700 draws
/// and 50.3 MB of vertex streams a frame: with only the entry cap in force the process heap
/// climbed **40-80 MB on every rendered frame, linearly, from the first one** - 252 MB
/// before the render window, **3,751 MB one hundred rendered frames later**, with the entry
/// cap firing all the while. In a browser that is the wasm32 ceiling and a dead tab; the
/// panel's own `emulator wasm heap` read 68 MB and then 3,412 MB, and the user's report of
/// it was that Chrome said "Aw, Snap".
///
/// The two maps hold the SAME `Arc`s (the guest stream and the repacked bytes), so bounding
/// each at this figure bounds the pair at about it rather than at twice it.
///
/// The default is deliberately well above one frame's working set - being a megabyte short
/// of the live set is what makes a bound thrash rather than bind
/// [[vitaslop-caches-that-clear-whole-are-the-long-run-degradation]] - and it is scaled to
/// the device like every other budget here. `VITASLOP_PACKED_CACHE_MB` sets it; `=0` removes
/// the byte bound entirely, which is the arm back to the entry cap alone.
/// How often (in resident-heap frames) [`ResidentHeap::forget_idle`] sweeps.
pub(super) const RESIDENT_SWEEP_EVERY: u64 = 120;

/// Frames a resident slice may go unbound before its record (and the stream it pins) is
/// dropped - see [`ResidentHeap::forget_idle`]. Default 600 (ten seconds at 60 Hz); `0`
/// never drops one, which is the old behaviour.
pub(super) fn resident_idle_frames() -> u64 {
    use std::sync::OnceLock;
    static CELL: OnceLock<u64> = OnceLock::new();
    *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_RESIDENT_IDLE_FRAMES")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(600)
    })
}

pub(super) fn packed_cache_budget_bytes() -> usize {
    use std::sync::OnceLock;
    static CELL: OnceLock<usize> = OnceLock::new();
    let base = *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_PACKED_CACHE_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(192)
            * 1024
            * 1024
    });
    if base == 0 {
        return usize::MAX;
    }
    crate::knobs::scale_budget(base)
}

/// Bytes a packed-geometry entry pins: the guest stream it was repacked from AND the
/// repacked bytes. Both, because both are held by the `Arc`s in the entry and neither goes
/// away until it is evicted.
pub(super) fn packed_entry_bytes(v: &(std::sync::Arc<[u8]>, std::sync::Arc<[u8]>, u64)) -> usize {
    v.0.len() + v.1.len()
}

/// Total bytes a packed-geometry map holds.
pub(super) fn packed_map_bytes<K>(
    map: &HashMap<K, (std::sync::Arc<[u8]>, std::sync::Arc<[u8]>, u64)>,
) -> usize {
    map.values().map(packed_entry_bytes).sum()
}

/// Shed the least recently used entries until the map is under `budget` BYTES.
///
/// The same rule as [`evict_oldest`] in every other respect: entries stamped with the
/// current epoch are in use by the frame being encoded and are never candidates, so the
/// bound is honoured eventually rather than inside the frame that would immediately need
/// what it threw away. Run ONCE PER FRAME rather than per insert - a walk of the map on
/// every miss is work proportional to the frame's own volume on the path that exists to
/// make the volume free, which is the cost this renderer keeps finding in what it replaces.
///
/// Returns how many entries were shed.
///
/// `protect` names the epoch whose entries are never candidates - `None` bounds the map
/// HARD, which is right only where a miss costs a lookup rather than a rebuild.
pub(super) fn evict_to_budget<K: Copy + Eq + std::hash::Hash>(
    map: &mut HashMap<K, (std::sync::Arc<[u8]>, std::sync::Arc<[u8]>, u64)>,
    budget: usize,
    protect: Option<u64>,
    mut shed: impl FnMut(&K),
) -> usize {
    let mut held: usize = packed_map_bytes(map);
    if held <= budget {
        return 0;
    }
    // Evict BELOW the budget - an eighth of headroom - so a cache sitting exactly at the
    // bound does not pay this sort every single frame.
    let target = budget - budget / 8;
    let mut candidates: Vec<(u64, K, usize)> = map
        .iter()
        .filter(|(_, v)| protect != Some(v.2))
        .map(|(k, v)| (v.2, *k, packed_entry_bytes(v)))
        .collect();
    candidates.sort_unstable_by_key(|(used, ..)| *used);
    let mut n = 0;
    for (_, k, bytes) in candidates {
        if held <= target {
            break;
        }
        shed(&k);
        map.remove(&k);
        held -= bytes;
        n += 1;
    }
    n
}

/// >>> REPACKS OF GEOMETRY THIS RUN HAD ALREADY EVICTED - the cache THRASHING.
///
/// A miss is not evidence of anything: reaching new geometry is a miss and is unavoidable.
/// A miss on a key that was in this cache and got shed is different in kind - it is work
/// being done twice because the bound was too tight, and it is the only reading that
/// justifies moving [`PACKED_CACHE_CAP`]. Near zero beside a large eviction count means the
/// cap is fine and the title's geometry is genuinely fresh.
/// >>> ARE WE UPLOADING THE SAME BYTES EVERY FRAME? See the probe in `ensure_gxp_arena`.
///
/// `seen` is the last hash written to each (arena slot, vertices|indices|uniforms);
/// `repeated` and `total` are byte counts since the last read. Behind
/// `VITASLOP_GXM_ARENA_REPEAT=1` because it hashes every uploaded byte.
#[derive(Default)]
pub struct ArenaRepeat {
    pub(super) seen: HashMap<(u32, u8), u64>,
    pub repeated: u64,
    pub total: u64,
}

pub static ARENA_REPEAT: std::sync::Mutex<Option<ArenaRepeat>> = std::sync::Mutex::new(None);

/// Take the arena-repeat tally: `(repeated bytes, total bytes)` since the last read.
pub fn take_arena_repeat() -> (u64, u64) {
    let mut g = ARENA_REPEAT.lock().unwrap_or_else(|e| e.into_inner());
    match g.as_mut() {
        None => (0, 0),
        Some(a) => {
            let out = (a.repeated, a.total);
            a.repeated = 0;
            a.total = 0;
            out
        }
    }
}

pub(super) static PACKED_REPACK_AFTER_EVICT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

thread_local! {
    /// The content keys most recently shed by [`PACKED_CACHE_CAP`], for
    /// [`PACKED_REPACK_AFTER_EVICT`]. A bounded ring rather than every key ever evicted:
    /// the question is whether the title comes BACK for what was just thrown away, and
    /// remembering 115,000 keys to answer it would cost more memory than the cache does.
    /// One cap's worth of history is a long enough look-back for that.
    pub(super) static PACKED_EVICTED: std::cell::RefCell<(HashSet<u64>, std::collections::VecDeque<u64>)> =
        std::cell::RefCell::new((HashSet::default(), std::collections::VecDeque::new()));
}

/// Remember one shed content key. See [`PACKED_EVICTED`].
pub(super) fn note_packed_evicted(key: u64) {
    PACKED_EVICTED.with(|c| {
        let (set, order) = &mut *c.borrow_mut();
        if set.insert(key) {
            order.push_back(key);
            while order.len() > PACKED_CACHE_CAP {
                if let Some(old) = order.pop_front() {
                    set.remove(&old);
                }
            }
        }
    });
}

/// Charge a miss whose geometry this cache had shed. Counted ONCE per shed key - the entry
/// is back in the cache after this, so a second charge would be counting the same eviction
/// twice. See [`PACKED_REPACK_AFTER_EVICT`].
pub(super) fn note_packed_miss(key: u64) {
    PACKED_EVICTED.with(|c| {
        let (set, _) = &mut *c.borrow_mut();
        if set.remove(&key) {
            PACKED_REPACK_AFTER_EVICT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });
}

/// What a cache is cut back to when it reaches its cap - three quarters of it, so the cut
/// happens once in a while rather than every frame at the boundary.
pub(super) fn evict_target(cap: usize) -> usize {
    cap - cap / 4
}

/// The repacked-geometry cache, by content: `(the guest stream it was packed from, the
/// packed bytes, the frame epoch it was last USED in)`. The stamp is what
/// [`evict_oldest`] orders by; see there for why insertion order will not do.
pub(super) type PackedCache = HashMap<(u64, u64), (std::sync::Arc<[u8]>, std::sync::Arc<[u8]>, u64)>;
/// The same entries reached by allocation rather than by content hash.
pub(super) type PackedAllocCache =
    HashMap<(u64, usize, usize), (std::sync::Arc<[u8]>, std::sync::Arc<[u8]>, u64)>;

/// The group2 sampler bind groups, by `(pipeline, what the group names)`: `(the group, the
/// texture-view keys it names - so a view eviction can invalidate exactly the groups that
/// named it - and the frame epoch it was last USED in)`.
pub(super) type SamplerBgCache = HashMap<(u64, u64), (wgpu::BindGroup, Vec<(u64, SamplerDim)>, u64)>;

/// Every cache eviction, by cache, so a run can be asked whether its caps are firing at all
/// and at what rate. Default-on: one atomic per eviction PASS, not per entry, and the
/// question it answers ("does this session degrade because a cache keeps being emptied")
/// took four device dumps and a 48,000-frame session to ask once.
pub(super) static CACHE_EVICTIONS: std::sync::Mutex<Option<Vec<(&'static str, u64, u64)>>> =
    std::sync::Mutex::new(None);

pub(super) fn note_cache_evicted(what: &'static str, n: usize) {
    if n == 0 {
        return;
    }
    let mut g = CACHE_EVICTIONS.lock().unwrap_or_else(|e| e.into_inner());
    let v = g.get_or_insert_with(Vec::new);
    match v.iter_mut().find(|(w, ..)| *w == what) {
        Some((_, passes, entries)) => {
            *passes += 1;
            *entries += n as u64;
        }
        None => v.push((what, 1, n as u64)),
    }
}

/// `"<cache> Nx/M entries"` for every cache that has evicted, for the report line.
///
/// The repacked-geometry cache carries one extra number, because its eviction count alone
/// is not readable - see [`PACKED_REPACK_AFTER_EVICT`].
pub(super) fn cache_eviction_summary() -> String {
    let g = CACHE_EVICTIONS.lock().unwrap_or_else(|e| e.into_inner());
    let thrash = PACKED_REPACK_AFTER_EVICT.load(std::sync::atomic::Ordering::Relaxed);
    match g.as_ref() {
        None => "none".to_string(),
        Some(v) => v
            .iter()
            .map(|(w, passes, entries)| {
                let mut s = format!("{w} {passes}x/{entries} entries");
                if *w == "packed geometry (by content)" {
                    s.push_str(&format!(" [{thrash} REPACKED AFTER EVICTION]"));
                }
                s
            })
            .collect::<Vec<_>>()
            .join(", "),
    }
}

/// >>> EVICT THE OLDEST QUARTER, NOT EVERYTHING.
///
/// The pattern every bounded cache in this file wanted and only the texture-view cache had:
/// shed what has gone longest unused, down to `keep`, and leave the working set alone.
///
/// # Why a `clear()` at a cap is the long-run degradation itself
/// A wholesale clear does not cost "a rebuild" once - it costs a rebuild of the ENTIRE
/// working set inside the next frame, and then it repeats, because whatever filled the cache
/// the first time fills it again at the same rate. MEASURED on a golf run: four dumps of the
/// same title, same code, ordered by session length, read `prepare` 9.4 / 25.6 / 36.8 / 67.3
/// us a draw at 30 / 30 / 15 / 8 fps, with every counter the frame prints FLAT or falling.
/// Nothing in the frame's work changed; what changed was how often these caps were hit.
/// [[vitaslop-caches-that-clear-whole-are-the-long-run-degradation]]
///
/// # The bound is honoured "eventually", deliberately
/// An entry stamped with the CURRENT epoch is one this frame has already used, so evicting
/// it would cost a rebuild inside the very frame that is running - the disease as the cure.
/// Those are therefore never evicted, and a frame that alone touches more distinct entries
/// than the cap leaves the map above it. That is bounded by one frame's working set rather
/// than by the run, which is the property that matters: what these caps exist to stop is
/// unbounded growth ACROSS a session, and a frame's own draws are already bounded by the
/// title. The caller reports when it happens.
///
/// Returns how many entries were shed.
pub(super) fn evict_oldest<K, V>(
    map: &mut HashMap<K, V>,
    keep: usize,
    current: u64,
    stamp: impl Fn(&V) -> u64,
) -> usize {
    evict_oldest_noting(map, keep, current, stamp, |_| {})
}

/// [`evict_oldest`], reporting each key it sheds.
///
/// Separate so the common caller stays a four-argument call: only the repacked-geometry
/// cache needs the keys, and it needs them to answer whether the title comes back for what
/// was shed - see [`PACKED_REPACK_AFTER_EVICT`].
pub(super) fn evict_oldest_noting<K, V>(
    map: &mut HashMap<K, V>,
    keep: usize,
    current: u64,
    stamp: impl Fn(&V) -> u64,
    mut shed: impl FnMut(&K),
) -> usize {
    if map.len() <= keep {
        return 0;
    }
    let want = map.len() - keep;
    // Only entries this frame has NOT touched can go - see above.
    let mut old: Vec<u64> = map.values().map(&stamp).filter(|s| *s != current).collect();
    if old.is_empty() {
        return 0;
    }
    // `select_nth_unstable` rather than a sort: the threshold is an order statistic, and
    // this runs over thousands of entries at the moment a cache is already under pressure.
    let n = want.min(old.len()) - 1;
    let cutoff = *old.select_nth_unstable(n).1;
    let before = map.len();
    // `<= cutoff` sheds every entry at the threshold rather than an arbitrary subset of
    // them, so this can shed slightly MORE than `want` when many share an epoch. Shedding
    // an extra few of the oldest is the harmless direction; picking arbitrarily among
    // equals is the one that would make two runs differ for no reason.
    map.retain(|k, v| {
        let s = stamp(v);
        let keep = s == current || s > cutoff;
        if !keep {
            shed(k);
        }
        keep
    });
    before - map.len()
}

/// How many frames a render target may go untouched - neither rendered into nor sampled -
/// before [`GxmRenderer::reclaim_stale_rtt`] releases it.
///
/// 1,800 frames is a full MINUTE at this project's 30 fps target, and the margin is
/// deliberate. The one pattern that could be hurt by this is "rendered once, then sampled
/// again much later" - a surface baked at load and read on a screen the player reaches
/// minutes afterwards. Being SAMPLED stamps a target just as being rendered into does, so
/// such a surface survives for as long as anything reads it; the exposure is only the gap
/// between the last read and the next. A minute of neither is unambiguous abandonment, and
/// combined with [`RTT_KEEP_FREELY`] it means a title that behaves never reaches this code
/// at all.
pub(super) const RTT_STALE_FRAMES: u64 = 1800;

/// How many targets a renderer holds before the reclamation walk runs at all. A title with
/// this few is not leaking and should not pay a map walk per frame to prove it.
pub(super) const RTT_KEEP_FREELY: usize = 16;

/// Targets reclaimed and bytes released, for the residency report.
pub(super) static RTT_RECLAIMED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) static RTT_RECLAIMED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(super) fn note_rtt_reclaimed(n: u64, bytes: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    RTT_RECLAIMED.fetch_add(n, Relaxed);
    RTT_RECLAIMED_BYTES.fetch_add(bytes, Relaxed);
}

/// `(targets, bytes)` reclaimed this run.
pub(super) fn rtt_reclaimed_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (RTT_RECLAIMED.load(Relaxed), RTT_RECLAIMED_BYTES.load(Relaxed))
}

/// Bound on the group-0/1 uniform bind groups. Generous, because an entry is a small handle
/// and a title's legitimate working set of (arena slot, shader pair, format, sample count)
/// is in the low thousands - this is a bound on GROWTH ACROSS A RUN, not a working limit.
pub(super) const UBO_BG_CACHE_CAP: usize = 8192;

/// Bound on `resident_i_seen`, which is only a "have I met this address before" set.
pub(super) const RESIDENT_SEEN_CAP: usize = 8192;

/// >>> DROP THE ENTRIES THAT CAN NO LONGER ANSWER, NOT ALL OF THEM.
///
/// `resident_*_seen` records "I have met these exact bytes once" as a `Weak`, and at its cap
/// it used to `clear()`. That is the most expensive thing this particular map can do. The
/// promotion test is "second CONTENT sighting", so dropping a LIVE candidate does not merely
/// cost a lookup - it sends geometry that was one sighting away from being RESIDENT back
/// through repack + arena copy + `write_buffer`, every frame, until it is sighted twice
/// again. MEASURED on a golf run at only 4,400 frames: `resident seen 8037 / 8192`, i.e. the
/// cap is reached in minutes and the reset then repeats for the life of the run. That is a
/// degradation that COMPOUNDS, and it is the one the long-session reports are about.
/// [[vitaslop-caches-that-clear-whole-are-the-long-run-degradation]]
///
/// A DEAD entry is free to drop: its `Weak` cannot be upgraded, so it can never match
/// anything again and is pure occupancy - the map is keyed on an allocation ADDRESS, and an
/// address recycled after the original died fails the upgrade anyway. Pruning those keeps
/// every live candidate's promotion progress, which is the entire content of the map.
///
/// Wholesale clearing REMAINS, as the last resort for a map whose entries are genuinely all
/// live, because the bound is not optional. It reports separately from the prune, so a run
/// where the prune is not enough says so instead of looking like a run where it never fired.
pub(super) fn prune_seen<K: Copy + Eq + std::hash::Hash>(
    map: &mut HashMap<K, (std::sync::Weak<[u8]>, u64)>,
    now: u64,
    what: &'static str,
) {
    let before = map.len();
    // `strong_count` rather than `upgrade`: this asks whether the buffer is still alive
    // without briefly resurrecting it, and it is the same answer.
    map.retain(|_, (w, _)| w.strong_count() > 0);
    let pruned = before - map.len();
    note_seen_pruned(pruned as u64);
    // >>> AND WHEN THE PRUNE FINDS NOTHING, WHICH IS THE MEASURED CASE, EVICT BY AGE.
    //
    // The first version of this stopped at the prune and cleared wholesale if the map was
    // still full. It then reported doing exactly that, on the first headless replay it ran
    // in: `the vertex promotion map reached its cap with all 8192 entries still LIVE`. The
    // reason is structural rather than surprising - the `Weak`s point at the PACKED vertex
    // buffers, which `packed` and `packed_by_alloc` hold strongly at 4,096 entries each, so
    // the seen map's 8,192 slots fill with entries that are all, correctly, still alive.
    // A liveness test cannot bound a map whose contents are kept alive by another cache.
    //
    // So each entry carries the frame it was filed in, and the oldest quarter goes. An entry
    // here is short-lived by design - it is consumed on the SECOND sighting of the same
    // content, which happens within a frame or two if it happens at all - so an entry that
    // has been waiting for thousands of frames is one whose second sighting never came.
    // Those are precisely the ones to drop, and dropping them costs nothing at all.
    if map.len() >= RESIDENT_SEEN_CAP {
        let n = evict_oldest(map, evict_target(RESIDENT_SEEN_CAP), now, |(_, filed)| *filed);
        note_cache_evicted(what, n);
        // Only if even that found nothing - every entry filed in the CURRENT frame, which
        // would mean one frame alone offered more distinct meshes than the whole cap.
        if map.len() >= RESIDENT_SEEN_CAP {
            map.clear();
            report_seen_cleared(what, before);
        }
    }
}

/// Drop every entry whose buffer is gone - see the sweep in `encode_chain` for why this
/// runs every frame: a dead `Weak<[u8]>` is what keeps the dead mesh's bytes allocated.
pub(super) fn sweep_dead_seen<K: Copy + Eq + std::hash::Hash>(
    map: &mut HashMap<K, (std::sync::Weak<[u8]>, u64)>,
) -> usize {
    let before = map.len();
    map.retain(|_, (w, _)| w.strong_count() > 0);
    before - map.len()
}

/// How many dead `resident_*_seen` entries the prune above has reclaimed, and how many times
/// it ran. Default-on and two atomics, because the question this answers - "is the prune
/// enough, or is the map genuinely full of live candidates" - is not answerable from the
/// occupancy line alone, and the wholesale clear it replaces was the largest compounding
/// cost in a long run.
pub(super) static SEEN_PRUNES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) static SEEN_PRUNED_ENTRIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(super) fn note_seen_pruned(n: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    SEEN_PRUNES.fetch_add(1, Relaxed);
    SEEN_PRUNED_ENTRIES.fetch_add(n, Relaxed);
}

/// `(times pruned, dead entries reclaimed)` for the report line.
pub(super) fn seen_prune_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (SEEN_PRUNES.load(Relaxed), SEEN_PRUNED_ENTRIES.load(Relaxed))
}

pub(super) fn report_seen_cleared(what: &'static str, held: usize) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert(what) {
        return;
    }
    report_warn!(
        "gxm resident: the {what} promotion map reached its cap with all {held} entries \
         still LIVE, so it was cleared wholesale. Every mesh awaiting its second content \
         sighting has to be sighted twice again before it can become resident, which costs \
         a repack, an arena copy and a write_buffer per draw until it is. Reported once."
    );
}
