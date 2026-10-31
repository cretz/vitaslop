//! Uniforms into the pass's arena: the SA blocks, the guest-memory windows, and the
//! half-float conversions the window substitution needs.

use super::*;

/// Set once any pipeline binds its guest-memory windows as STORAGE (they outgrew a uniform
/// binding - see `vitaslop_gxp_shader::module::mem_window_storage`). Those windows ride in
/// the same per-pass uniform arena as everything else, so from then on the arena is created
/// able to back a storage binding too. A run that never builds such a pipeline never sets it,
/// and its arena is created exactly as before.
static MEM_STORAGE_PIPELINES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Note that a pipeline with storage-bound windows exists - see [`MEM_STORAGE_PIPELINES`].
pub(super) fn note_mem_storage_pipeline() {
    MEM_STORAGE_PIPELINES.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// The usage the per-pass uniform ARENA must be created with: `UNIFORM`, plus `STORAGE` once
/// [`note_mem_storage_pipeline`] has been called.
pub(super) fn arena_ubo_usage() -> wgpu::BufferUsages {
    if MEM_STORAGE_PIPELINES.load(std::sync::atomic::Ordering::Relaxed) {
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE
    } else {
        wgpu::BufferUsages::UNIFORM
    }
}

/// The dynamic-offset alignment one stage's window block is pushed at: the uniform one, or
/// for a storage-bound block the larger of it and the storage one - the block's offset is
/// handed to a STORAGE binding, which validates against its own limit.
pub(super) fn mem_window_align(device: &wgpu::Device, ubo_align: u64, windows: &[vitaslop_gxp_shader::MemWindow]) -> u64 {
    if vitaslop_gxp_shader::module::mem_window_storage(windows) {
        ubo_align.max(u64::from(device.limits().min_storage_buffer_offset_alignment))
    } else {
        ubo_align
    }
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
pub(super) fn push_sa(udata: &mut Vec<u8>, lanes: u32, guest: &[u8], align: u64) -> u32 {
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
/// dynamic-offset alignment: one header vec4 per window whose lane x is that window's
/// guest base address, then every window's bytes at the offsets the shader was emitted
/// against (`vitaslop_gxp_shader::module::mem_window_placements`), padded to the
/// pipeline's declared binding size (`GxpPipeline::mem_bind_bytes`). The same arena
/// discipline as [`push_sa`].
/// [`push_mem_windows`] behind the per-pass memo - see `GxpLive::mem_push_memo`.
pub(super) fn push_mem_windows_memo(
    memo: &mut HashMap<(usize, u8, u64), (u32, Vec<(u32, std::sync::Arc<[u8]>)>)>,
    // The FRAME-scoped key set - see `GxpLive::mem_push_frame`. Counting only.
    frame_seen: &mut HashSet<(usize, u8, u64)>,
    pipe: (usize, u8),
    udata: &mut Vec<u8>,
    bind_bytes: u32,
    spec: &[vitaslop_gxp_shader::MemWindow],
    windows: &[(u32, std::sync::Arc<[u8]>)],
    align: u64,
) -> u32 {
    let mut fold: u64 = 0xcbf2_9ce4_8422_2325;
    for (base, bytes) in windows {
        for v in [*base as u64, std::sync::Arc::as_ptr(bytes) as *const u8 as usize as u64, bytes.len() as u64] {
            fold ^= v;
            fold = fold.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    let key = (pipe.0, pipe.1, fold);
    if let Some((off, held)) = memo.get(&key)
        && held.len() == windows.len()
        && held.iter().zip(windows).all(|(h, w)| h.0 == w.0 && std::sync::Arc::ptr_eq(&h.1, &w.1))
    {
        split_add(&PREP.uni_mem_hits, 1);
        return *off;
    }
    // >>> CHARGED HERE, ON THE MISS, AND THAT IS THE WHOLE POINT OF THE COUNTER.
    //
    // It used to be charged at the CALL SITE, which charges the memo's HITS too - and 918 of
    // this title's 1,305 pushes a frame are hits, so the arena's byte line credited the
    // guest-memory windows with 1.17 MB a frame of which about 0.35 MB was ever written. An
    // over-reported byte count is the one direction a ranking cannot survive: it names the
    // wrong phase as the biggest thing the renderer uploads.
    let off = push_mem_windows(udata, bind_bytes, spec, windows, align);
    split_add(&PREP.arena_bytes, bind_bytes as u64);
    split_add(&PREP.arena_bytes_memwin, bind_bytes as u64);
    // >>> AND WHETHER THIS FRAME HAS ALREADY UPLOADED THESE EXACT BYTES IN ANOTHER PASS.
    // Not a memo and not a saving - a measurement of one. See `GxpLive::mem_push_frame`.
    if !frame_seen.insert(key) {
        split_add(&PREP.arena_bytes_memwin_repeat, bind_bytes as u64);
    }
    memo.insert(key, (off, windows.to_vec()));
    off
}

pub(super) fn push_mem_windows(
    udata: &mut Vec<u8>,
    bind_bytes: u32,
    spec: &[vitaslop_gxp_shader::MemWindow],
    windows: &[(u32, std::sync::Arc<[u8]>)],
    align: u64,
) -> u32 {
    let pad = udata.len() as u64 % align;
    if pad != 0 {
        udata.resize(udata.len() + (align - pad) as usize, 0);
    }
    let off = udata.len();
    let need = bind_bytes as usize;
    udata.resize(off + need, 0);
    for (i, (base, _)) in windows.iter().enumerate() {
        let at = off + i * 16;
        udata[at..at + 4].copy_from_slice(&base.to_le_bytes());
    }
    for ((_, bytes), place) in
        windows.iter().zip(vitaslop_gxp_shader::module::mem_window_placements(spec))
    {
        let start = place.first_word as usize * 4;
        let n = bytes.len().min(need.saturating_sub(start));
        udata[off + start..off + start + n].copy_from_slice(&bytes[..n]);
    }
    off as u32
}

/// The granule a PER-DRAW window block is padded to. Its binding size is the block's, and every
/// distinct size is its own bind group (`GxpLive::ubo_bgs`), so rounding up keeps a pair whose
/// extent moves by a few bytes from draw to draw on one group - for at most this many zero bytes.
const PER_DRAW_BLOCK_GRANULE: usize = 64 << 10;

/// Append one draw's PER-DRAW window block (`vitaslop_gxp_shader::module::MemWindow::per_draw`)
/// and return `(offset, block bytes, raw-target fill ranges)`.
///
/// `extents` is each window's reach for THIS draw (`vitaslop_gxp_shader::reach::per_draw_extents`).
/// The block is one header vec4 per window - `(guest base, words bound, first word in the block,
/// 0)`, which is what the per-draw helper reads - then the bytes, with windows that overlap in
/// guest memory placed over ONE copy of their union. Every window is a snapshot of the same
/// memory at the same moment, so the overlap holds the same bytes whichever window supplied them;
/// a depth-reconstruct pass's four windows, each a surface row apart, upload their ~4 MiB union
/// once instead of four times.
///
/// A window whose captured bytes are SHORTER than its extent (a capsule recorded before the
/// window was sized per draw, or a capture told a smaller target) leaves the rest zero, and the
/// first such draw is reported - a raw target this frame rendered over the range still fills it
/// on the GPU (`fill_windows_from_raw_targets`), so the ranges returned are the merged spans.
///
/// >>> AND THE BYTES THAT FILL OVERWRITES ARE NEITHER COPIED NOR UPLOADED. `raw_fill` is what the
/// pass's fill copies from (`GxpLive::raw_fill`); every span byte it covers
/// (`crate::gpu::raw_window_fill_cover`, the fill's own plan) is left out of the copy here and
/// appended to `holes` as an arena range the upload skips. MEASURED on a shooter's stairwell: a
/// depth-reconstruct pass's ~4 MiB block over its G-buffer was copied and uploaded every frame
/// only for the fill to replace all of it - 8.6 GB of the run's 9.9 GB of arena uploads.
pub(super) fn push_mem_windows_per_draw(
    udata: &mut Vec<u8>,
    windows: &[(u32, std::sync::Arc<[u8]>)],
    extents: &[u32],
    align: u64,
    // The largest block the device can bind: past it nothing is pushed and `None` returned.
    max_bytes: u64,
    raw_fill: &[crate::gpu::RawFillTarget],
    holes: &mut Vec<(usize, usize)>,
) -> Option<(u32, u32, Vec<(u32, u32, u64)>)> {
    let n = windows.len();
    let addrs: Vec<u32> = windows.iter().map(|(a, _)| *a).collect();
    let (placed, span_of) = crate::gpu::per_draw_window_spans(&addrs, extents);
    // `(start, end, first word in the block)` per span.
    let mut spans: Vec<(u64, u64, u32)> = Vec::with_capacity(placed.len());
    let mut word = n as u64 * 4;
    for &(s, e) in &placed {
        spans.push((s, e, word as u32));
        word += (e - s).div_ceil(4).div_ceil(4) * 4;
    }
    let bytes = (word * 4).div_ceil(PER_DRAW_BLOCK_GRANULE as u64) * PER_DRAW_BLOCK_GRANULE as u64;
    if bytes > max_bytes {
        return None;
    }
    let bytes = bytes as usize;
    let pad = udata.len() as u64 % align;
    if pad != 0 {
        udata.resize(udata.len() + (align - pad) as usize, 0);
    }
    let off = udata.len();
    udata.resize(off + bytes, 0);
    // Each span's fill cover, relative to the span's start; the arena ranges it names are
    // holes in the upload.
    let covers: Vec<Vec<(u32, u32)>> = spans
        .iter()
        .map(|s| {
            let mut c = Vec::new();
            if !raw_fill.is_empty() {
                crate::gpu::raw_window_fill_cover(s.0 as u32, (s.1 - s.0) as u32, raw_fill, &mut c);
                let at = off + s.2 as usize * 4;
                holes.extend(c.iter().map(|&(a, b)| (at + a as usize, at + b as usize)));
            }
            c
        })
        .collect();
    let mut short = false;
    for (i, (base, data)) in windows.iter().enumerate() {
        let s = spans[span_of[i]];
        let first = s.2 + ((u64::from(*base) - s.0) / 4) as u32;
        let words = extents.get(i).copied().unwrap_or(0).div_ceil(4);
        let h = off + i * 16;
        udata[h..h + 4].copy_from_slice(&base.to_le_bytes());
        udata[h + 4..h + 8].copy_from_slice(&words.to_le_bytes());
        udata[h + 8..h + 12].copy_from_slice(&first.to_le_bytes());
        let want = words as usize * 4;
        let take = data.len().min(want);
        short |= take < want;
        let at = off + first as usize * 4;
        // Only the bytes the fill does not cover; `d` is this window's start within its span.
        let d = (u64::from(*base) - s.0) as usize;
        let mut from = 0usize;
        for &(cs, ce) in &covers[span_of[i]] {
            let (cs, ce) = ((cs as usize).saturating_sub(d).min(take), (ce as usize).saturating_sub(d).min(take));
            if cs > from {
                udata[at + from..at + cs].copy_from_slice(&data[from..cs]);
            }
            from = from.max(ce);
        }
        if take > from {
            udata[at + from..at + take].copy_from_slice(&data[from..take]);
        }
    }
    if short {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                target: "vitaslop::gpu",
                "a per-draw guest-memory window carried fewer bytes than this draw can reach - the rest reads zero unless a raw target this frame covers it (an older capsule, or a capture sized for a smaller target)"
            );
        });
    }
    let ranges = spans.iter().map(|s| (s.0 as u32, (s.1 - s.0) as u32, off as u64 + u64::from(s.2) * 4)).collect();
    Some((off as u32, bytes as u32, ranges))
}

/// Report - once per (pair, block size) - what a PER-DRAW window set bound: each window's extent
/// and the block the draw uploaded, which is the number the per-draw sizing exists to keep small.
pub(super) fn report_per_draw_windows(key: u64, extents: &[u32], bytes: u32) {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<HashSet<(u64, u32)>>> = std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert((key, bytes)) {
        return;
    }
    tracing::info!(
        target: "vitaslop::status",
        key = format_args!("{key:016x}"),
        "gxp per-draw windows: extents {extents:?} -> {bytes} bytes uploaded for the draw"
    );
}

/// Diagnostic (`VITASLOP_GXP_FMEM=<lane>=<value>` / `<lane>*<factor>`, comma separated):
/// rewrite F16 lanes of every FRAGMENT memory window before they are uploaded.
///
/// # Why this exists, and why it is the fragment window rather than the SA bank
/// `VITASLOP_GXP_SA` substitutes the SA bank, which is where a title's uniforms live when
/// the driver COPIES them into the register file. A title whose material block is a bound
/// uniform buffer the shader chases a pointer into has none of its material there - the
/// whole of it is in `gxp_fmem` - so there was no way to ask "which uniform carries this
/// factor" for exactly the programs whose output is a lit surface.
///
/// The lane is counted in F16 components from the window's base, which for a declared F16
/// parameter is `resource_index * 2 + component` - the numbering the capsule replay tool's
/// `VITASLOP_CAPSULE_DUMP_SA=1` prints beside each parameter name, and the same numbering
/// its `--fmem` flag takes, so a reading taken offline transfers to a live frame unchanged.
pub(super) fn fmem_substitutions() -> &'static [(usize, bool, f32)] {
    static SPEC: std::sync::OnceLock<Vec<(usize, bool, f32)>> = std::sync::OnceLock::new();
    SPEC.get_or_init(|| {
        let Ok(v) = crate::knobs::var("VITASLOP_GXP_FMEM") else { return Vec::new() };
        v.split(',')
            .filter(|s| !s.trim().is_empty())
            .filter_map(|item| {
                let (lane, scale, val) = match item.split_once('*') {
                    Some((l, f)) => (l, true, f),
                    None => {
                        let (l, f) = item.split_once('=')?;
                        (l, false, f)
                    }
                };
                Some((lane.trim().parse().ok()?, scale, val.trim().parse().ok()?))
            })
            .collect()
    })
}

/// Apply [`fmem_substitutions`] to a draw's fragment windows, returning `None` when the knob
/// is unset so the ordinary path costs nothing but one pointer test.
pub(super) fn substituted_fmem(windows: &[(u32, std::sync::Arc<[u8]>)]) -> Option<Vec<(u32, std::sync::Arc<[u8]>)>> {
    let spec = fmem_substitutions();
    if spec.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(windows.len());
    for (at, shared) in windows {
        let mut bytes = shared.to_vec();
        for &(lane, scale, v) in spec {
            let off = lane * 2;
            let Some(dst) = bytes.get_mut(off..off + 2) else { continue };
            let was = half_to_f32(u16::from_le_bytes([dst[0], dst[1]]));
            let now = if scale { was * v } else { v };
            dst.copy_from_slice(&f32_to_half(now).to_le_bytes());
        }
        out.push((*at, std::sync::Arc::from(bytes)));
    }
    Some(out)
}

/// f32 -> IEEE-754 half bits, round-to-nearest-even. The inverse of [`half_to_f32`], needed
/// only by [`substituted_fmem`]: a substituted value has to land in the window as the exact
/// bits the shader unpacks, or the reading taken through it is of some other number.
pub(super) fn f32_to_half(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let mut mant = (x & 0x007f_ffff) as i32;
    let exp = ((x >> 23) & 0xff) as i32 - 127 + 15;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        mant |= 0x0080_0000;
        let shift = 14 - exp;
        let half = (mant >> shift) as u16;
        let rem = mant & ((1 << shift) - 1);
        let tie = 1 << (shift - 1);
        let round = u16::from(rem > tie || (rem == tie && half & 1 == 1));
        return sign | (half + round);
    }
    let half = ((exp as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    let round = u16::from(rem > 0x1000 || (rem == 0x1000 && (mant >> 13) & 1 == 1));
    sign | (half + round)
}


/// Convert an IEEE-754 half (binary16) to f32.
pub(super) fn half_to_f32(h: u16) -> f32 {
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
