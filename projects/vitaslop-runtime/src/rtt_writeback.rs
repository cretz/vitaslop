//! Putting a rendered render target's pixels BACK into guest memory.
//!
//! A GXM title does not only sample its render targets - it also reads texels out of them on
//! the CPU. MEASURED on a baseball title: its ambient light is a 128x128 target the GPU paints
//! once a frame and the guest then indexes with `row*512 + col*4` to pull one texel out.
//! Because nothing ever put the rendered pixels back in guest memory, the read returned the
//! game's OWN allocator poison - `0xBAADCAFE`, every word of the buffer - and the three poison
//! bytes went through the title's base-10 log decode into an ambient a hundred times too
//! bright. That was the whole of that title's washed-out frame.
//!
//! The GPU half (copy the target into a mapped buffer) is engine-specific: native can block on
//! the map, the browser cannot and delivers it a frame or two later. Everything after the
//! bytes exist is here, shared, so the two engines cannot drift apart on WHICH targets are
//! written, WHEN a target's memory is ours to write, and at what stride.

use crate::capture::{ColorSurface, Scene};

/// `VITASLOP_GXM_RTT_WRITEBACK=<max texels>` - the size cap on a written-back target, in
/// texels. `0` turns the writeback off. Default 65536 (256x256), which covers every CPU-read
/// target seen so far and costs one 256 KB readback.
///
/// Why SMALL only: a target a title reads on the CPU is small BY CONSTRUCTION - a probe grid,
/// a luminance reduction, an occlusion or exposure result - because the CPU has to walk it. A
/// full-size colour buffer is consumed by the GPU and never crosses back, and reading one back
/// every frame would be tens of megabytes.
///
/// Read through a `OnceLock` because it is consulted once per target per frame. Goes through
/// `knobs::var`, so the browser's knob box reaches it.
///
/// The value may also carry `skip=<hex addr>[+<hex addr>...]` after the cap (`65536,skip=
/// 0x8d0d0cf0+0x8cb94200`), which holds those addresses BACK from the writeback while leaving
/// every other target alone. The cap alone cannot bisect a title whose targets share a size -
/// this one has eight 128x128s and six 256x64s - so "which ONE of these writebacks changes the
/// picture" is unanswerable by cap, and that is the question a writeback regression asks.
///
/// `skip=sampled` is the same list stated as a RULE rather than by address: hold back every
/// target some draw of the frame SAMPLES. The seam exists for a target the CPU walks, and a
/// target the GPU samples is consumed on the GPU - so the two sets should barely overlap, and
/// writing the second kind is work whose only effect is to change what the guest's own memory
/// says about a buffer it was not going to read.
pub fn rtt_writeback_texels() -> u32 {
    writeback_spec().0
}

/// Whether `addr` is held back by the `skip=` list of `VITASLOP_GXM_RTT_WRITEBACK`.
pub fn writeback_skipped(addr: u32) -> bool {
    writeback_spec().1.contains(&addr)
}

/// Whether the knob asked for `skip=sampled` - hold back every target the frame samples.
pub fn writeback_skips_sampled() -> bool {
    writeback_spec().2
}

/// Whether the knob asked for `skip=write`: do all the GPU work - the copy, the submit, the
/// map and the poll - and then write NOTHING into guest memory.
///
/// >>> THIS IS THE ONE A/B THAT SEPARATES THE BYTES FROM THE STALL. The writeback does a
/// readback per small target per frame, and a readback is a GPU stall; `=0` removes the bytes
/// AND the stall together, so a picture that moves between `=0` and `=65536` does not say which
/// of the two did it. With `skip=write` the stall is identical to a full writeback and only the
/// bytes are gone, so the two arms differ in exactly one thing.
pub fn writeback_skips_write() -> bool {
    writeback_spec().3
}

/// TEMPORARY TELEMETRY (`fill=<rrggbbaa>`): write this CONSTANT texel over the target instead
/// of the rendered pixels, so the picture can be read as a function of the VALUE the guest
/// finds in its probe rather than of whether anything is there at all. Sweeping it measures
/// the guest's own transfer function end to end.
pub fn writeback_fill(addr: u32) -> Option<[u8; 4]> {
    let (_, _, _, _, fill, at, _) = writeback_spec();
    let t = (*fill)?;
    (at.is_empty() || at.contains(&addr)).then_some(t)
}

/// TEMPORARY TELEMETRY (`rows=<lo>-<hi>`): restrict `fill=` to a BAND OF ROWS, leaving every
/// other row the rendered pixels. Poisoning only the rows the render leaves black, and only
/// the rows it paints, is what separates "the guest reads where we painted" from "the guest
/// reads where we did not" - and the picture answers, so it needs no guest watchpoint.
pub fn writeback_fill_rows() -> Option<(usize, usize)> {
    writeback_spec().6
}

/// TEMPORARY TELEMETRY (`VITASLOP_RTT_PROBE_LOG=1`): print every written-back target's
/// texel(24,1) EVERY frame, not once. The trajectory of that value across frames is the input
/// side of the feedback loop the writeback closes; a once-only line shows only its first step.
fn probe_log(addr: u32) -> bool {
    static SPEC: std::sync::OnceLock<Option<Vec<u32>>> = std::sync::OnceLock::new();
    let spec = SPEC.get_or_init(|| {
        let v = vitaslop_platform::knobs::var("VITASLOP_RTT_PROBE_LOG").ok()?;
        let v = v.trim();
        if v.is_empty() || v == "0" {
            return None;
        }
        // `=1` is every target; anything else is a `+`-separated address list, because a
        // per-frame line for every target of every frame is a log nobody can read.
        Some(if v == "1" {
            Vec::new()
        } else {
            v.split('+').filter_map(|a| u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()).collect()
        })
    });
    match spec {
        None => false,
        Some(list) => list.is_empty() || list.contains(&addr),
    }
}

/// The parsed knob: (texel cap, addresses held back, hold back sampled targets, write nothing).
fn writeback_spec() -> &'static (u32, Vec<u32>, bool, bool, Option<[u8; 4]>, Vec<u32>, Option<(usize, usize)>) {
    static SPEC: std::sync::OnceLock<(u32, Vec<u32>, bool, bool, Option<[u8; 4]>, Vec<u32>, Option<(usize, usize)>)> = std::sync::OnceLock::new();
    SPEC.get_or_init(|| {
        let Ok(v) = vitaslop_platform::knobs::var("VITASLOP_GXM_RTT_WRITEBACK") else {
            return (256 * 256, Vec::new(), false, false, None, Vec::new(), None);
        };
        let mut cap = 256 * 256;
        let mut skip = Vec::new();
        let mut sampled = false;
        let mut no_write = false;
        let mut fill: Option<[u8; 4]> = None;
        let mut at: Vec<u32> = Vec::new();
        let mut rows: Option<(usize, usize)> = None;
        for field in v.split(',').map(str::trim).filter(|f| !f.is_empty()) {
            match field.strip_prefix("skip=") {
                Some("sampled") => sampled = true,
                Some("write") => no_write = true,
                _ if field.starts_with("rows=") => {
                    if let Some((a, b)) = field.trim_start_matches("rows=").split_once('-')
                        && let (Ok(a), Ok(b)) = (a.trim().parse(), b.trim().parse())
                    {
                        rows = Some((a, b));
                    }
                }
                _ if field.starts_with("at=") => {
                    at.extend(field.trim_start_matches("at=").split('+').filter_map(|a| {
                        u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()
                    }));
                }
                _ if field.starts_with("fill=") => {
                    let h = field.trim_start_matches("fill=").trim_start_matches("0x");
                    if let Ok(v) = u32::from_str_radix(h, 16) {
                        fill = Some(v.to_be_bytes());
                    }
                }
                Some(list) => skip.extend(list.split('+').filter_map(|a| {
                    u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()
                })),
                // A cap that does not parse is the DEFAULT, not zero: reading a malformed knob
                // as "writeback off" would look exactly like the arm-back it is spelled `=0`.
                None => cap = field.parse::<u32>().unwrap_or(cap),
            }
        }
        if !skip.is_empty() || sampled || no_write {
            let list: Vec<String> = skip.iter().map(|a| format!("{a:#010x}")).collect();
            tracing::info!(
                target: "vitaslop::status",
                "gxm rtt writeback: cap {cap} texels, HOLDING BACK {}{}{}",
                if sampled { "every target the frame SAMPLES" } else { "" },
                if no_write { "THE WRITE ITSELF (the readback still runs, so the GPU stall is unchanged)" } else { "" },
                if list.is_empty() { String::new() } else { format!(" {}", list.join(" ")) }
            );
        }
        (cap, skip, sampled, no_write, fill, at, rows)
    })
}

/// The colour surface this frame's scenes render into at `addr`, if any.
///
/// >>> THIS IS THE SAFETY BOUND, not just a lookup. A target the frame did NOT render into may
/// have been freed and the memory handed to something else entirely; writing pixels over it
/// would corrupt whatever the guest put there. Only a target this frame's own scenes name as a
/// colour surface is written, which is also the only one whose stride and format are known.
pub fn surface_for(scenes: &[Scene], addr: u32) -> Option<ColorSurface> {
    scenes.iter().filter_map(|s| s.color.as_ref()).find(|c| c.data_addr == addr).copied()
}

/// Put the rendered pixels of every readback in `writebacks` - `(guest colour address, width,
/// height, straight RGBA8)` - back into guest memory. Returns how many targets were written, so
/// a caller can report a run in which the writeback did nothing.
pub fn apply_rtt_writebacks(
    writebacks: &[(u32, u32, u32, Vec<u8>)],
    scenes: &[Scene],
    mut read: impl FnMut(u32, usize) -> Vec<u8>,
    mut write: impl FnMut(u32, &[u8]),
) -> usize {
    let mut done = 0usize;
    for (addr, w, h, rgba) in writebacks {
        let Some(c) = surface_for(scenes, *addr) else { continue };
        // TEMPORARY TELEMETRY. WHERE in the target the guest asked its draws to paint. A CPU-
        // read probe whose content lands away from the origin is either a title packing an
        // atlas or a placement bug, and the guest's own viewport/region clip is the only
        // witness that tells those apart - the picture cannot.
        report_probe_geometry(*addr, &c, scenes);
        if apply_one(*addr, *w, *h, rgba, &c, &mut read, &mut write) {
            done += 1;
        }
    }
    done
}

/// Write ONE target's rendered pixels (`rgba`, tightly packed `w*h*4`, memory order R,G,B,A)
/// into the guest memory of the colour surface `c` at `addr`. True if it was written.
///
/// # What it will and will not write
/// Only base format `U8U8U8U8` - which is what the renderer actually holds, since every target
/// is created in the swapchain's 8-bit format whatever the guest asked for. Writing a 16-bit
/// or float surface from an 8-bit readback would put the wrong WIDTH of pixel in guest memory,
/// which is worse than leaving it alone, so those are counted and reported rather than
/// converted.
///
/// The byte order is the identity permutation (memory-order R,G,B,A = GXM's `ABGR` selector),
/// the same assumption the renderer makes at every other seam - it holds no per-target swizzle.
/// A surface declaring a different selector is reported by format, once per address.
pub fn apply_one(
    addr: u32,
    w: u32,
    h: u32,
    rgba: &[u8],
    c: &ColorSurface,
    read: &mut impl FnMut(u32, usize) -> Vec<u8>,
    write: &mut impl FnMut(u32, &[u8]),
) -> bool {
    // The guest's pixel WIDTH, and whether this writeback can produce it. The renderer holds
    // every target as 8-bit RGBA whatever the guest asked for, so a writeback of a narrower
    // surface has to CONVERT - writing 4 bytes where the guest reads 2 shears the image, which
    // is why this used to refuse outright. See `writeback_bytes_per_pixel`.
    let Some(bpp) = writeback_bytes_per_pixel(c.format) else {
        report_writeback_skipped(addr, c.format);
        return false;
    };
    // Held back by the knob's `skip=` list - a bisect, reported once by `writeback_spec`.
    if writeback_skipped(addr) || writeback_skips_write() {
        return false;
    }
    if (w as usize) * (h as usize) * 4 > rgba.len() {
        return false;
    }
    // The guest's own row pitch, which is not the target's width: a surface is allocated at
    // its STRIDE and a writeback that ignored it would shear every row after the first.
    let stride = c.stride_pixels.max(w) as usize * bpp;
    let rows = h.min(c.height) as usize;
    let cols = w.min(c.width) as usize;
    // >>> ONLY WHERE NOTHING HAS BEEN WRITTEN. A target's guest memory is not always idle: a
    // title may COMPOSE into it on the CPU and expect its own bytes back. MEASURED - the first
    // cut wrote every rendered target unconditionally and repainted a racer's car livery into
    // a mottled green-blue camo, because that livery is the guest's own composition over
    // memory that had once been a render target.
    //
    // The test is the same one the texture path uses for "nothing has written here": ONE
    // REPEATED 4-BYTE WORD over a bounded prefix, which is what a zero fill and an allocator's
    // poison fill both are (this title's is `0xBAADCAFE`) and which no composed image is.
    //
    // It is LATCHED per address, because the writeback's own first write makes the memory
    // non-uniform and a re-test would then refuse every later frame - and a probe grid the
    // guest re-reads every frame needs the newest pixels, not the first ones.
    if !writeback_owns(addr, || nothing_written_here(&read(addr, 4096))) {
        // >>> AND SAY SO, ONCE. A refusal here is the writeback deciding the guest owns these
        // bytes, and it is the ONE outcome that leaves a CPU-reading title reading something
        // other than its rendered pixels - which is the whole defect this module exists for.
        // Silent, it is indistinguishable from a target that was never rendered, from one the
        // frame's scenes do not name, and from the writeback working: every reading that
        // matters looks the same in the log. A title that pools its probe targets sees a NEW
        // address every few frames, and one of them landing on memory that is not uniform
        // refuses that target for the rest of the run [[vitaslop-fallback-must-report]].
        report_writeback_unowned(addr, w, h);
        return false;
    }
    // >>> AND IT MUST STILL BE OURS. See `writeback_still_ours`: the latch above answers only
    // for the FIRST sight of an address, and a deferred write that lands after the title has
    // recycled a pooled target writes pixels over whatever the allocator has since put there.
    // The probe is the same 4096 bytes the latch read, so this costs one hash of one row.
    let fp_bytes = (cols * writeback_bytes_per_pixel(c.format).unwrap_or(4)).min(4096);
    if fp_bytes > 0 && !writeback_still_ours(addr, &read(addr, fp_bytes), fp_bytes) {
        report_writeback_recycled(addr, w, h);
        writeback_fingerprints().lock().unwrap_or_else(|e| e.into_inner()).remove(&addr);
        return false;
    }
    // TEMPORARY TELEMETRY. The per-frame form of the line below: the probe texel EVERY frame,
    // which is the input side of the feedback loop, plus the frame's own mean so the loop can
    // be read as a trajectory rather than a first step.
    if probe_log(addr) {
        let t = if w > 24 && h > 1 { &rgba[(w as usize + 24) * 4..(w as usize + 24) * 4 + 4] } else { &[][..] };
        let sum: u64 = rgba.iter().map(|b| *b as u64).sum();
        // Eight ROW means down the target (rgb only), so the readback answers for the whole
        // image and not one texel: a chain-dump PNG of a written-back target can be stale, and
        // this is the only readback of these pixels there is.
        let row_bytes = w as usize * 4;
        let rows: Vec<u32> = (0..8u32)
            .filter_map(|i| {
                let y = (h as usize * i as usize) / 8;
                let r = rgba.get(y * row_bytes..(y + 1) * row_bytes)?;
                let s: u64 = r.chunks_exact(4).map(|px| px[0] as u64 + px[1] as u64 + px[2] as u64).sum();
                Some((s / (3 * w as u64).max(1)) as u32)
            })
            .collect();
        // >>> AND ITS ALPHA, BECAUSE AN ALPHA TEST IS WHAT READS IT.
        //
        // The row means above are RGB only, and a target whose colour is perfect can still be
        // invisible: this title's CROWD samples a 1024x128 atlas and DISCARDS every fragment
        // whose alpha is under 0.75 (`frag_90b70060`: `pf0.w - 0.75 >= 0` or `discard`), so
        // "how much of this target would pass that test" is the whole question for it and an
        // RGB mean cannot answer it. Reported as a PERCENTAGE of texels at or above 191/255.
        let a_sum: u64 = rgba.chunks_exact(4).map(|px| px[3] as u64).sum();
        let a_px = (rgba.len() / 4).max(1) as u64;
        let a_hi = rgba.chunks_exact(4).filter(|px| px[3] >= 191).count();
        tracing::info!(
            target: "vitaslop::status",
            "rtt probe: {addr:#010x} {w}x{h} texel(24,1)={t:?} mean={:.2} rows={rows:?} alpha_mean={:.1} alpha_ge_191={:.1}%",
            sum as f64 / rgba.len().max(1) as f64,
            a_sum as f64 / a_px as f64,
            100.0 * a_hi as f64 / a_px as f64,
        );
    }
    if report_once(0x7100_0000_0000_0000 ^ addr as u64) {
        let t = if w > 24 && h > 1 { rgba[(w as usize + 24) * 4..(w as usize + 24) * 4 + 4].to_vec() } else { Vec::new() };
        tracing::info!(target: "vitaslop::status", "gxm rtt writeback: first frame-end write of {addr:#010x} ({w}x{h}): texel(24,1)={t:?}");
    }
    report_writeback_target(addr, w, h, c, rows, cols, stride);
    // TEMPORARY TELEMETRY. `fill=` replaces the rendered pixels with a constant, so the arms
    // of a sweep differ in the VALUE the guest reads and in nothing else - same targets, same
    // readback, same stall, same bytes written.
    let constant = writeback_fill(addr).map(|t| t.repeat(cols));
    let band = writeback_fill_rows();
    let mut packed: Vec<u8> = Vec::new();
    for y in 0..rows {
        let src = y * (w as usize) * 4;
        let filled = band.is_none_or(|(lo, hi)| y >= lo && y <= hi);
        let row = match constant.as_deref() {
            Some(c) if filled => c,
            _ => &rgba[src..src + cols * 4],
        };
        // The telemetry fill and the rendered pixels are both RGBA8, so the conversion is the
        // LAST step and both arms of a `fill=` sweep go through it.
        //
        // The test is the FORMAT, not the pixel width: `U2F10F10F10` is four bytes a pixel and
        // is not the readback's layout, and keying this on `bpp == 4` wrote its packed HDR
        // words out as plain RGBA8.
        let row = if c.format & 0xF180_0000 == 0 {
            row
        } else {
            encode_row(c.format, row, &mut packed);
            &packed
        };
        if y == 0 {
            writeback_remember(addr, row, fp_bytes);
        }
        write(addr.wrapping_add((y * stride) as u32), row);
    }
    true
}

/// The guest's bytes per pixel for a colour format this writeback can produce, or `None` for
/// one it cannot.
///
/// # Why the narrow formats are converted rather than refused
/// A target is created in the swapchain's 8-bit format whatever the guest asked for, so a
/// readback is always RGBA8. Refusing every other format left a whole CLASS of title with no
/// writeback at all: golf bakes its tree impostors into **202 U1U5U5U5 targets**, samples them
/// as ordinary textures for the rest of the hole, and destroys the targets in between. With the
/// pixels never reaching guest memory the sample decoded memory nothing had written - white on
/// the desktop, the allocator's own leftovers on the device, and DIFFERENT leftovers from run
/// to run, which is exactly the "red trees, and they flicker" a user reported.
///
/// The conversion is anchored on this renderer's own texture DECODER (`render.rs`, base formats
/// 0x04 and 0x05) rather than on a fresh reading of the layout, so what is written back is by
/// construction what reads back. A format with no encoder here still refuses and still reports.
fn writeback_bytes_per_pixel(format: u32) -> Option<usize> {
    match format & 0xF180_0000 {
        0x0000_0000 => Some(4),               // U8U8U8U8 - the readback's own layout
        0x3000_0000 | 0x4000_0000 => Some(2), // U5U6U5, U1U5U5U5
        // >>> U4U4U4U4, AND LEAVING IT OUT HID A TARGET AN ALPHA TEST READS.
        //
        // MEASURED on PCSE00084: its CROWD atlas (0x94111900, 1024x128, painted by 103
        // draws in four 256-wide tiles) is this format, so it was refused here and no
        // instrument in the project could read a texel of it - while the crowd program
        // that samples it DISCARDS every fragment whose alpha is under 0.75, which is
        // the empty grey seating on screen. A format with an alpha channel that no
        // readback can see is the worst of both: the picture depends on it and nothing
        // can say what it holds.
        0x5000_0000 => Some(2),               // U4U4U4U4
        0x4100_0000 => Some(4),               // U2F10F10F10 - packed, and NOT the readback's
        // >>> U8 AND U8U8 - the SINGLE-CHANNEL family, and leaving it out left a whole class of
        // >>> target with no write-back at all AND no way to look at one.
        //
        // MEASURED on PCSE00084: its shadow chain is U8 throughout - a 720x408 target the player
        // meshes are rendered into and several 960x544 targets a blur pass filters it through -
        // and every one of them was refused here by format, so nothing has ever read a texel of
        // any of them back. That is not only a lost write-back: it is the reason "what does the
        // shadow map hold" could not be asked at all, on either engine, while the characters
        // that sample it render BLACK.
        //
        // The renderer holds every target as 8-bit RGBA, so the single channel is the RED one -
        // the same channel `color_format_to_texture_format` maps this format's texture form
        // onto, so what is written back is by construction what reads back.
        0x8080_0000 => Some(1),               // U8
        0xB080_0000 => Some(2),               // U8U8
        _ => None,
    }
}

/// Pack one RGBA8 row into the guest's 16-bit colour format, into `out`.
///
/// The inverse of the texture decoder's `0x04`/`0x05` cases, SWIZZLE included: a colour format
/// carries its swizzle in bits 21:20 and the texture enumeration agrees name for name across
/// this family (see `color_format_to_texture_format`), so one selector decides the channel
/// order at both ends.
fn encode_row(format: u32, rgba: &[u8], out: &mut Vec<u8>) {
    let swizzle = (format >> 20) & 3;
    let five_five_five = format & 0xF180_0000 == 0x4000_0000;
    if format & 0xF180_0000 == 0x4100_0000 {
        encode_row_u2f10(swizzle, rgba, out);
        return;
    }
    // The single-channel family: take the channels the format has, in memory order, starting at
    // RED. A one-component texture reads its texel out of the first channel here and out of the
    // first channel on the way in, which is what makes the round trip exact rather than a guess
    // about a swizzle this family does not carry across (see `color_format_to_texture_format`).
    if matches!(format & 0xF180_0000, 0x8080_0000 | 0xB080_0000) {
        let n = if format & 0xF180_0000 == 0x8080_0000 { 1 } else { 2 };
        out.clear();
        out.reserve(rgba.len() / 4 * n);
        for px in rgba.chunks_exact(4) {
            out.extend_from_slice(&px[..n]);
        }
        return;
    }
    // U4U4U4U4: four equal nibbles. Swizzle 1 is ARGB - alpha in the high nibble, then red,
    // green, blue - and swizzle 0 is ABGR, which is the same order with red and blue swapped.
    // The alpha nibble is the point: this is the only 16-bit colour format in the set that
    // carries a usable alpha, and an alpha TEST is what reads this title's atlas back.
    if format & 0xF180_0000 == 0x5000_0000 {
        out.clear();
        out.reserve(rgba.len() / 2);
        for px in rgba.chunks_exact(4) {
            let to4 = |v: u8| ((v as u32 * 15 + 127) / 255) as u16;
            let (r, g, b, a) = (to4(px[0]), to4(px[1]), to4(px[2]), to4(px[3]));
            let (c1, c2, c3) = if swizzle == 1 { (r, g, b) } else { (b, g, r) };
            let w: u16 = (a << 12) | (c1 << 8) | (c2 << 4) | c3;
            out.extend_from_slice(&w.to_le_bytes());
        }
        return;
    }
    out.clear();
    out.reserve(rgba.len() / 2);
    for px in rgba.chunks_exact(4) {
        let (r, g, b, a) = (px[0] as u32, px[1] as u32, px[2] as u32, px[3] as u32);
        let to5 = |v: u32| ((v * 31 + 127) / 255) as u16;
        let w: u16 = if five_five_five {
            // U1U5U5U5: [A1][5][5][5], MSB to LSB. Swizzle 1 is ARGB, so red is the high lane.
            let (hi, mid, lo) = if swizzle == 1 { (r, g, b) } else { (b, g, r) };
            let alpha = if a >= 128 { 0x8000u16 } else { 0 };
            alpha | (to5(hi) << 10) | (to5(mid) << 5) | to5(lo)
        } else {
            // U5U6U5: swizzle 0 is BGR, so blue is the high lane; anything else is RGB.
            let (hi, lo) = if swizzle == 0 { (b, r) } else { (r, b) };
            let g6 = ((g * 63 + 127) / 255) as u16;
            (to5(hi) << 11) | (g6 << 5) | to5(lo)
        };
        out.extend_from_slice(&w.to_le_bytes());
    }
}

/// Pack one RGBA8 row into `U2F10F10F10` - three unsigned 10-bit FLOATS under a 2-bit unorm
/// alpha - into `out`.
///
/// The inverse of the texture decoder's `0x9a` case, and the second format this writeback can
/// produce for the golf title: it bakes six 256x256 atlases in this format beside its 202
/// `U1U5U5U5` ones, and those six were the remaining refusal.
///
/// # Why a TABLE and not a formula
/// A 10-bit float is five bits of exponent over five of mantissa, so near 1.0 its quantum is
/// `1/32` - FOUR TIMES COARSER than the 8-bit byte being encoded. There is no round trip to
/// preserve, only a nearest value to choose, and "nearest" has to be nearest AS THE DECODER
/// READS IT, not in the reals: the decoder rounds its own product to a byte and clamps, so two
/// different words can decode to the same byte and a word can be nearer in float and further
/// in byte. The table is therefore built BY RUNNING THE DECODER over all 1,024 words, which
/// makes it exact by construction and immune to any later change of the decode that this file
/// does not follow - the same discipline as `encode_row`'s 16-bit arm.
fn encode_row_u2f10(swizzle: u32, rgba: &[u8], out: &mut Vec<u8>) {
    let table = ten_bit_nearest();
    out.clear();
    out.reserve(rgba.len());
    for px in rgba.chunks_exact(4) {
        let (r, g, b, a) = (px[0], px[1], px[2], px[3]);
        // Swizzle 1 is ARGB - red in the TOP colour lane - and anything else is ABGR, exactly
        // as the decoder reads them.
        let (hi, mid, lo) = if swizzle == 1 { (r, g, b) } else { (b, g, r) };
        let a2 = ((a as u32 * 3 + 127) / 255) & 3;
        let w = (a2 << 30)
            | ((table[hi as usize] as u32) << 20)
            | ((table[mid as usize] as u32) << 10)
            | table[lo as usize] as u32;
        out.extend_from_slice(&w.to_le_bytes());
    }
}

/// For each byte, the 10-bit word whose DECODE is nearest to it; ties take the smaller word.
fn ten_bit_nearest() -> &'static [u16; 256] {
    static T: std::sync::OnceLock<[u16; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut best = [0u16; 256];
        let mut err = [i32::MAX; 256];
        for word in 0u16..1024 {
            let decoded = decode_ten_bit_float(word) as i32;
            for (c, (b, e)) in best.iter_mut().zip(err.iter_mut()).enumerate() {
                let d = (decoded - c as i32).abs();
                if d < *e {
                    *e = d;
                    *b = word;
                }
            }
        }
        best
    })
}

/// One 10-bit lane of `U2F10F10F10`, as the texture decoder reads it: five bits of exponent
/// over five of mantissa, bias 15, no sign, clamped into the byte range.
///
/// Kept beside the encoder rather than shared with `render.rs` on purpose - this is the
/// SPECIFICATION the table is fitted to, and a copy that drifts is caught by
/// `a_u2f10f10f10_writeback_reads_back_as_the_nearest_pixel_the_format_holds`, which runs the
/// real decoder.
fn decode_ten_bit_float(bits: u16) -> u8 {
    let (exp, mant) = (bits >> 5, bits & 0x1f);
    let v = if exp == 0 {
        // Denormal: no implicit leading one.
        mant as f32 / 32.0 * 2f32.powi(-14)
    } else {
        (1.0 + mant as f32 / 32.0) * 2f32.powi(exp as i32 - 15)
    };
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// TEMPORARY TELEMETRY: report, once per target, every distinct draw geometry the frame's
/// scenes use when rendering into it - the viewport offset/scale and the region clip the guest
/// itself set, against the surface's own extent.
fn report_probe_geometry(addr: u32, c: &ColorSurface, scenes: &[Scene]) {
    if !probe_log(addr) {
        return;
    }
    let mut seen: Vec<String> = Vec::new();
    for s in scenes.iter().filter(|s| s.color.as_ref().is_some_and(|k| k.data_addr == addr)) {
        for d in &s.draws {
            let r = &d.render_state;
            let v = r.viewport;
            // The first four vertices' leading floats too: a reduction quad carries its
            // positions straight in the stream, so where the content LANDS is decided here and
            // nowhere else, and a viewport that covers the whole target says nothing about it.
            let stride = (d.vertex_stride as usize).max(4);
            let pos: Vec<String> = (0..4)
                .filter_map(|i| {
                    let o = i * stride;
                    (o + 16 <= d.vertices.len()).then(|| {
                        let f = |k: usize| {
                            f32::from_le_bytes(d.vertices[o + k * 4..o + k * 4 + 4].try_into().unwrap())
                        };
                        format!("({:.3},{:.3},{:.3},{:.3})", f(0), f(1), f(2), f(3))
                    })
                })
                .collect();
            let line = format!(
                "vp_en={} vp=[{:.1},{:.1} {:.1},{:.1}] rc_mode={} rc=[{},{} {},{}] prim={} idx={} vstride={} nverts={} v={}",
                r.viewport_enable, v[0], v[2], v[1], v[3], r.region_clip_mode,
                r.region_clip[0], r.region_clip[1], r.region_clip[2], r.region_clip[3],
                d.primitive, d.index_count, d.vertex_stride,
                d.vertices.len() / stride, pos.join(" "),
            );
            if !seen.contains(&line) {
                seen.push(line);
            }
        }
    }
    for line in &seen {
        if report_once(0x7200_0000_0000_0000 ^ addr as u64 ^ fold_str(line)) {
            tracing::info!(
                target: "vitaslop::status",
                "rtt probe geom: {addr:#010x} surface {}x{} stride {} type {} scale {} format {:#010x} - {line}",
                c.width, c.height, c.stride_pixels, c.surface_type, c.scale_mode, c.format
            );
        }
    }
}

/// FNV-1a of a string, so a per-target report can be keyed on the geometry it names.
fn fold_str(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Whether a bounded prefix of a buffer says NOTHING HAS BEEN WRITTEN HERE.
///
/// One repeated 4-byte word, which is what a zero fill and an allocator's poison fill both are
/// - MLB 12 fills a fresh allocation with `0xBAADCAFE` - and which no composed image is. A
/// shorter prefix than one word cannot answer the question and is not treated as empty.
pub fn nothing_written_here(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes.chunks_exact(4).all(|word| word == &bytes[..4])
}

/// Whether ANY 16-byte slot of a window still says nothing has been written into it, while
/// other slots hold real values.
///
/// # Why the whole-window test is not enough, and what it misses
/// [`nothing_written_here`] answers "is this buffer untouched", which is the right question for
/// a window a job thread fills WHOLESALE after the draw that names it - a baseball crowd's
/// skinning block is 3,776 bytes of zeroes at its draw, so it takes the re-read.
///
/// A buffer that is PARTLY filled at the draw passes neither test: it is not one repeated word,
/// so the re-read is skipped, and the draw keeps a snapshot in which some slots are real and the
/// rest are still zero. For a BONE PALETTE that is not a subtle error - a vertex weighted to an
/// unwritten bone multiplies by a zero matrix and collapses to the origin, so the mesh renders
/// correctly as far as the last bone the job thread had reached and the remainder VANISHES.
/// A football title's players come out cut off at the waist, with whichever limbs were late
/// detached and adrift.
///
/// The slot is 16 bytes because that is a `vec4` - one matrix ROW - which is the granularity a
/// palette is written at and the smallest unit whose all-zero state is meaningful. A trailing
/// partial slot is ignored: it cannot be a whole row.
pub fn any_slot_unwritten(bytes: &[u8]) -> bool {
    bytes.len() >= 16 && bytes.chunks_exact(16).any(nothing_written_here)
}

/// Whether the writeback owns this target's guest memory, deciding once per address.
///
/// `unwritten` is only consulted the FIRST time an address is seen; after that the answer is
/// the latched one. See the call site for why re-testing would be self-defeating.
fn writeback_owns(addr: u32, unwritten: impl FnOnce() -> bool) -> bool {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static OWNED: Mutex<Option<HashMap<u32, bool>>> = Mutex::new(None);
    let mut g = OWNED.lock().unwrap_or_else(|e| e.into_inner());
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(&v) = m.get(&addr) {
        return v;
    }
    let v = unwritten();
    m.insert(addr, v);
    v
}

/// What the write-back last LEFT at this address: a fingerprint of the first row it wrote.
///
/// # >>> A DEFERRED WRITE MUST CHECK THAT THE DESTINATION IS STILL ITS OWN
///
/// [`writeback_owns`] decides ONCE, the first time an address is seen, and after that the
/// answer is latched. That is right for the question it asks ("did the guest compose here
/// before we ever touched it") and it is blind to the one that follows: the browser's ring is
/// ASYNCHRONOUS, the bytes land one or two frames after the copy, and in that window a pooled
/// target can be handed back to the title's allocator and re-used for something that is not an
/// image. Writing pixels over that corrupts it, and the corruption surfaces hundreds of frames
/// later as a call through a pointer that used to be a pointer.
///
/// The guest UNMAPPING the memory would say so, and on the title this was measured on it never
/// does - so the only witness left is the memory itself: if what is there now is neither the
/// uniform fill it started as nor the bytes we last wrote, somebody else owns it.
///
/// FNV-1a over at most the first row, which is what the 4096-byte probe can see.
fn writeback_fingerprints() -> &'static std::sync::Mutex<std::collections::HashMap<u32, u64>> {
    static F: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<u32, u64>>> =
        std::sync::OnceLock::new();
    F.get_or_init(Default::default)
}

fn fingerprint(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Whether the first row at `addr` still holds what this module last wrote there.
///
/// `None` recorded means we have not written this address yet, which is not a mismatch.
fn writeback_still_ours(addr: u32, probe: &[u8], n: usize) -> bool {
    let g = writeback_fingerprints().lock().unwrap_or_else(|e| e.into_inner());
    match g.get(&addr) {
        None => true,
        Some(&want) => probe.len() >= n && fingerprint(&probe[..n]) == want,
    }
}

fn writeback_remember(addr: u32, row: &[u8], n: usize) {
    let mut g = writeback_fingerprints().lock().unwrap_or_else(|e| e.into_inner());
    if row.len() >= n {
        g.insert(addr, fingerprint(&row[..n]));
    } else {
        g.remove(&addr);
    }
}

/// Say - once per address - that a deferred readback was DROPPED because the destination no
/// longer holds what this module last wrote there.
///
/// At `warn`, unconditionally, because both readings need a fix and they are opposite ones: a
/// target a CPU-reading title still wants has silently lost its pixels, or a target it has
/// recycled is one we were about to corrupt.
fn report_writeback_recycled(addr: u32, w: u32, h: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(addr) {
        return;
    }
    tracing::warn!(
        target: "vitaslop::gxm",
        "gxm rtt writeback: DROPPED the readback of {addr:#010x} ({w}x{h}) - the bytes there are          neither the uniform fill this address started as nor the row this module last wrote, so          something else owns that memory now and writing pixels over it would corrupt it. A          pooled render target handed back to the title's allocator between the GPU copy and the          map landing is how that happens."
    );
}

/// The guest's bytes per pixel, or 4 when the format has no encoder here. Used ONLY to size
/// the report's byte span, never to write with.
fn bytes_per_pixel_or_four(format: u32) -> usize {
    writeback_bytes_per_pixel(format).unwrap_or(4)
}

/// Report - once per address - a render target the writeback DOES write, with the format and
/// stride it wrote it at.
///
/// A fix that cannot say what it touched is a fix nobody can bound: this is a blast-radius
/// question first (which of a title's targets now have their pixels in guest memory) and a
/// correctness one second (the SWIZZLE selector, bits 21:20 of the colour format, says the
/// byte order a CPU reader expects and a wrong one silently recolours whatever reads it).
fn report_writeback_target(addr: u32, w: u32, h: u32, c: &ColorSurface, rows: usize, cols: usize, stride: usize) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(addr) {
        return;
    }
    let format = c.format;
    let swz = match (format >> 20) & 0x3 {
        1 => "ARGB",
        2 => "RGBA",
        3 => "BGRA",
        _ => "ABGR",
    };
    // >>> THE BYTE RANGE, BECAUSE AN EXTENT IS NOT A RANGE AND A WRITEBACK IS A WRITE INTO
    // >>> GUEST MEMORY.
    //
    // This printed the RENDERER's extent and the guest's stride and stopped there, which says
    // nothing about how far the write REACHES - and two of one title's targets are 110,592
    // bytes apart while both report `256x256`, so one is written over by the other and the only
    // evidence was a LATER target being refused for holding "the guest own composition",
    // which was our own pixels. Printing the span makes an overlap ARITHMETIC instead of an
    // inference: sort these lines by address and an overrun is visible.
    let span = rows.saturating_sub(1) * stride + cols * bytes_per_pixel_or_four(format);
    tracing::info!(
        target: "vitaslop::status",
        "gxm rtt writeback: {addr:#010x} render {w}x{h}, guest surface {}x{} stride {} format \
         {format:#010x} (swizzle {swz}) - writes {rows} rows x {cols} cols, span {span} bytes, \
         end {:#010x}",
        c.width,
        c.height,
        c.stride_pixels,
        u64::from(addr) + span as u64,
    );
}

/// Report - once per address - a render target whose guest format the writeback cannot
/// express.
/// Report - once per address - a rendered target the writeback REFUSED because its guest
/// memory already holds something that is not a uniform fill.
///
/// At WARN, because the consequence is a wrong picture and not a curiosity: a title that reads
/// this target on the CPU gets whatever is in that memory instead of what the GPU painted.
/// The two readings a reader has to be able to tell apart are named, because the refusal is
/// RIGHT in one of them (the guest composed those bytes itself and wants them back) and wrong
/// in the other (the memory happened not to be uniform when the target was first seen).
fn report_writeback_unowned(addr: u32, w: u32, h: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(addr) {
        return;
    }
    tracing::warn!(
        target: "vitaslop::gxm",
        "gxm rtt writeback: {addr:#010x} ({w}x{h}) is RENDERED but NOT written back - its guest \
         memory was not a uniform fill when this address was first seen, so the writeback \
         treats the bytes as the guest's own composition. If the guest instead READS this \
         target on the CPU, it is now reading something the GPU did not paint. Latched for the \
         run: this address will not be written back again."
    );
}

fn report_writeback_skipped(addr: u32, format: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(addr) {
        return;
    }
    tracing::warn!(
        "gxm rtt writeback: {addr:#010x} is colour format {format:#010x}, which this \
         writeback has no encoder for - every target is held as 8-bit RGBA, and writing \
         that into a surface of another pixel width would shear it. Left alone; a guest \
         that reads this target on the CPU still reads whatever its allocator left \
         there. See `writeback_bytes_per_pixel` for the formats that ARE converted."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `encode_row` writes for a 16-bit surface is what the texture decoder reads back.
    ///
    /// The decoder is the anchor, so it is repeated here from `render.rs` rather than called:
    /// the two are in different crates, and a test that called the real one could only fail
    /// when they were BOTH wrong in the same direction. Written out, a change to either side
    /// has to be made twice on purpose.
    #[test]
    fn a_sixteen_bit_writeback_reads_back_as_the_pixel_it_wrote() {
        // U1U5U5U5, swizzle 0 (ABGR): [A1][B5][G5][R5] from the top.
        let decode_1555 = |w: u16| -> [u8; 4] {
            let a = if w & 0x8000 != 0 { 255 } else { 0 };
            let hi = (((w >> 10) & 0x1f) as u32 * 255 / 31) as u8;
            let mid = (((w >> 5) & 0x1f) as u32 * 255 / 31) as u8;
            let lo = ((w & 0x1f) as u32 * 255 / 31) as u8;
            [lo, mid, hi, a]
        };
        let mut out = Vec::new();
        // The impostor atlas's own background colour, and the two extremes.
        let src = [93u8, 129, 79, 255, 0, 0, 0, 0, 255, 255, 255, 255];
        encode_row(0x4000_0000, &src, &mut out);
        assert_eq!(out.len(), 6, "three texels at two bytes each");
        let px = |i: usize| decode_1555(u16::from_le_bytes([out[i * 2], out[i * 2 + 1]]));
        // 5 bits hold 32 levels, so a channel comes back within half a step (255/31 = 8.2).
        for (i, want) in [[93u8, 129, 79, 255], [0, 0, 0, 0], [255, 255, 255, 255]].iter().enumerate() {
            let got = px(i);
            for c in 0..3 {
                let d = got[c].abs_diff(want[c]);
                assert!(d <= 5, "texel {i} channel {c}: wrote {}, read {} (off by {d})", want[c], got[c]);
            }
            assert_eq!(got[3], want[3], "texel {i} alpha");
        }
        // U5U6U5, swizzle 0 (BGR): green keeps six bits, so it is tighter than the others.
        let mut out565 = Vec::new();
        encode_row(0x3000_0000, &[93, 129, 79, 255], &mut out565);
        let w = u16::from_le_bytes([out565[0], out565[1]]);
        let r = ((w & 0x1f) as u32 * 255 / 31) as u8;
        let g = (((w >> 5) & 0x3f) as u32 * 255 / 63) as u8;
        let b = (((w >> 11) & 0x1f) as u32 * 255 / 31) as u8;
        assert!(r.abs_diff(93) <= 5 && g.abs_diff(129) <= 3 && b.abs_diff(79) <= 5, "565 gave {r},{g},{b}");
    }

    /// What `encode_row` writes for a `U2F10F10F10` surface is the NEAREST pixel that format
    /// can hold, as the texture decoder reads it - and the lanes land where the decoder looks.
    ///
    /// Not a round trip, because there is not one to have: near 1.0 a 10-bit float's quantum
    /// is `1/32`, four times coarser than the byte, so "wrote 255, read 255" is the only exact
    /// case and a tolerance alone would pass an encoder that quantised in the wrong DIRECTION
    /// everywhere. The contract is therefore exhaustive and absolute: over all 256 bytes and
    /// all 1,024 words, no word decodes closer than the one chosen.
    ///
    /// The decoder is repeated here from `render.rs` for the reason the 16-bit test gives -
    /// different crates, and a shared copy could only fail when both sides were wrong the same
    /// way.
    #[test]
    fn a_u2f10f10f10_writeback_reads_back_as_the_nearest_pixel_the_format_holds() {
        let decode_ten = |bits: u32| -> u8 {
            let (exp, mant) = (bits >> 5, bits & 0x1f);
            let v = if exp == 0 {
                mant as f32 / 32.0 * 2f32.powi(-14)
            } else {
                (1.0 + mant as f32 / 32.0) * 2f32.powi(exp as i32 - 15)
            };
            (v.clamp(0.0, 1.0) * 255.0).round() as u8
        };
        // Swizzle 1 is ARGB: red in the top colour lane, then green, then blue, under a 2-bit
        // alpha. The atlas background, transparent black, and white.
        let src = [93u8, 129, 79, 255, 0, 0, 0, 0, 255, 255, 255, 255];
        let mut out = Vec::new();
        encode_row(0x4110_0000, &src, &mut out);
        assert_eq!(out.len(), 12, "three texels at FOUR bytes each");
        let word = |i: usize| u32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
        for (i, want) in [[93u8, 129, 79, 255], [0, 0, 0, 0], [255, 255, 255, 255]].iter().enumerate() {
            let w = word(i);
            let got = [decode_ten(w >> 20 & 0x3ff), decode_ten(w >> 10 & 0x3ff), decode_ten(w & 0x3ff)];
            for (c, &want_c) in want[..3].iter().enumerate() {
                let best = (0u32..1024).map(|x| decode_ten(x).abs_diff(want_c)).min().unwrap();
                assert_eq!(
                    got[c].abs_diff(want_c),
                    best,
                    "texel {i} channel {c}: wrote {want_c}, read {} - the nearest this format \
                     holds is off by {best}",
                    got[c]
                );
            }
            // Two bits of alpha: 0, 85, 170, 255.
            let a = ((w >> 30) as f32 / 3.0 * 255.0).round() as u8;
            let best_a = [0u8, 85, 170, 255].iter().map(|x| x.abs_diff(want[3])).min().unwrap();
            assert_eq!(a.abs_diff(want[3]), best_a, "texel {i} alpha: wrote {}, read {a}", want[3]);
        }
        // Every byte, on its own, against every word there is.
        for c in 0..=255u8 {
            let mut one = Vec::new();
            encode_row(0x4110_0000, &[c, c, c, 255], &mut one);
            let w = u32::from_le_bytes(one[..4].try_into().unwrap());
            let best = (0u32..1024).map(|x| decode_ten(x).abs_diff(c)).min().unwrap();
            assert_eq!(decode_ten(w >> 20 & 0x3ff).abs_diff(c), best, "byte {c}");
        }
        // Swizzle 0 is ABGR - the same three lanes in the opposite order.
        let mut abgr = Vec::new();
        encode_row(0x4100_0000, &[255, 0, 0, 255], &mut abgr);
        let w = u32::from_le_bytes(abgr[..4].try_into().unwrap());
        assert_eq!(decode_ten(w & 0x3ff), 255, "red is the LOW lane under ABGR");
        assert_eq!(decode_ten(w >> 20 & 0x3ff), 0, "and the top lane is blue");
    }

    /// The deferred-window rule: this predicate has to separate a uniform block the guest fills
    /// AFTER its draw from one it REWRITES BETWEEN draws, because the end-of-scene re-read is
    /// right for the first and wrong for the second.
    ///
    /// Both real windows, verbatim. The baseball crowd's skinning window is 3,776 bytes of
    /// zeroes at the draw call and a job thread fills it afterwards, so it MUST be re-read or
    /// every crowd vertex collapses to the origin. The golf title's 136-byte vertex window
    /// already holds its ambient floats at the draw, and re-reading it handed all 202
    /// tree-impostor bakes of one frame the LAST draw's ambient - `(1.0, 0.0, 1.0)`, magenta.
    ///
    /// Pinned here rather than at the resolve site because the resolve needs a live guest and
    /// this is the whole of its decision.
    #[test]
    fn a_window_filled_after_its_draw_is_unwritten_and_one_rewritten_between_draws_is_not() {
        // The crowd's skinning window as it reads at `sceGxmDraw`.
        assert!(nothing_written_here(&[0u8; 3776]), "all-zero skinning window must be re-read");
        // Golf's vertex window at 0x882c4200: 34 words of real floats. Only the shape matters -
        // a leading run of zeroes, then values - because that leading run is what a prefix-only
        // test would have stopped at.
        let mut win = Vec::new();
        for w in [
            0x00000000u32, 0x00000000, 0x00000000, 0x3f800000, 0x3f800000, 0x80000000,
            0x3fa3126e, 0x3fa3126e, 0x3f9e147a, 0x00000000, 0x3f800000, 0x00000000,
        ] {
            win.extend_from_slice(&w.to_le_bytes());
        }
        assert!(
            !nothing_written_here(&win),
            "a window holding real uniform floats must KEEP its draw-time bytes"
        );
        // And the poison case, which is the same decision for the same reason.
        assert!(nothing_written_here(&[0xFE, 0xCA, 0xAD, 0xBA].repeat(944)));
    }

    #[test]
    fn a_poison_fill_and_a_zero_fill_are_both_unwritten_and_an_image_is_not() {
        assert!(nothing_written_here(&[0; 64]));
        assert!(nothing_written_here(&[0xFE, 0xCA, 0xAD, 0xBA].repeat(16)));
        let mut img = [0xFE, 0xCA, 0xAD, 0xBA].repeat(16);
        img[37] ^= 1;
        assert!(!nothing_written_here(&img));
        assert!(!nothing_written_here(&[1, 2, 3]));
    }

    /// A HALF-FILLED bone palette is the case that falls between the two whole-window answers,
    /// and it is the one that cuts a player off at the waist.
    #[test]
    fn a_partly_filled_palette_is_unwritten_by_slot_and_written_by_the_whole_window_test() {
        // 8 bone rows: the first four hold real matrix values, the last four nothing yet.
        let mut win = Vec::new();
        for i in 0..4u32 {
            for lane in 0..4u32 {
                win.extend_from_slice(&(1.0f32 + i as f32 + lane as f32).to_le_bytes());
            }
        }
        let filled = win.len();
        win.extend(std::iter::repeat_n(0, filled));
        // The whole-window test says "something is here", which is true and not the question.
        assert!(!nothing_written_here(&win));
        // The slot test sees the rows the job thread has not reached.
        assert!(any_slot_unwritten(&win), "an all-zero 16-byte row must count as unwritten");
        // And once every row holds a value, nothing is owed a re-read - this is the golf
        // ambient block's case, which must keep its own draw-time bytes.
        let full = &win[..filled];
        assert!(!nothing_written_here(full));
        assert!(!any_slot_unwritten(full), "a fully written window must KEEP its draw-time bytes");
    }

    /// The whole-window answer must stay a SPECIAL CASE of the slot answer, or a window the old
    /// rule re-read would stop being re-read and a crowd would vanish.
    #[test]
    fn every_window_the_whole_window_test_calls_unwritten_the_slot_test_does_too() {
        for fill in [vec![0u8; 64], [0xFE, 0xCA, 0xAD, 0xBA].repeat(16)] {
            assert!(nothing_written_here(&fill));
            assert!(any_slot_unwritten(&fill));
        }
    }

    /// Shorter than one row cannot answer the question, exactly as shorter than one word cannot.
    #[test]
    fn a_window_shorter_than_one_row_is_not_slot_unwritten() {
        assert!(!any_slot_unwritten(&[0; 15]));
        assert!(!any_slot_unwritten(&[]));
        // A trailing partial row is ignored: it is not a whole matrix row. The leading row must
        // carry DISTINCT words - a row of one repeated byte is itself "unwritten" by the same
        // definition, which is the answer this predicate is built on.
        let mut win: Vec<u8> = (0..16u8).collect();
        win.extend_from_slice(&[0; 8]);
        assert!(!any_slot_unwritten(&win));
    }

    /// A U8 target round-trips through the RED channel, and a U8U8 one through red and green.
    ///
    /// The single-channel family is what this title's whole shadow chain is, and it was refused
    /// by format - so no U8 target's pixels had ever reached guest memory on either engine.
    #[test]
    fn a_single_channel_writeback_reads_back_as_the_channel_it_wrote() {
        let mut out = Vec::new();
        let rgba = [10u8, 20, 30, 40, 50, 60, 70, 80];
        encode_row(0x8080_0000, &rgba, &mut out);
        assert_eq!(out, vec![10, 50]);
        encode_row(0xB080_0000, &rgba, &mut out);
        assert_eq!(out, vec![10, 20, 50, 60]);
        assert_eq!(writeback_bytes_per_pixel(0x8080_0000), Some(1));
        assert_eq!(writeback_bytes_per_pixel(0xB080_0000), Some(2));
    }

    /// An address this module has never written is not a mismatch: the fingerprint guard must
    /// be silent until it has something to compare against, or the FIRST write of every target
    /// would be refused and the whole write-back would be a no-op.
    #[test]
    fn an_address_never_written_is_still_ours() {
        assert!(writeback_still_ours(0x1000_0000, &[1, 2, 3, 4], 4));
    }

    /// The row we left is the row we find: this is the steady state on every engine, and it is
    /// what makes the guard free on the desktop, where the write-back is synchronous.
    #[test]
    fn the_row_this_module_wrote_is_still_ours() {
        let row = [9u8, 8, 7, 6, 5, 4, 3, 2];
        writeback_remember(0x1000_1000, &row, 8);
        assert!(writeback_still_ours(0x1000_1000, &row, 8));
    }

    /// ONE byte different is somebody else's write. A pooled target handed back to the title's
    /// allocator between the GPU copy and the map landing looks exactly like this, and writing
    /// over it is the memory corruption the guard exists to stop.
    #[test]
    fn a_row_somebody_else_overwrote_is_not_ours() {
        let row = [9u8, 8, 7, 6, 5, 4, 3, 2];
        writeback_remember(0x1000_2000, &row, 8);
        let mut other = row;
        other[5] = 0;
        assert!(!writeback_still_ours(0x1000_2000, &other, 8));
    }

    /// A probe SHORTER than the row it is compared against cannot say the bytes agree, so it
    /// must not: a guard that reads "no evidence" as "still ours" is not a guard.
    #[test]
    fn a_probe_shorter_than_the_row_is_not_ours() {
        let row = [9u8, 8, 7, 6, 5, 4, 3, 2];
        writeback_remember(0x1000_3000, &row, 8);
        assert!(!writeback_still_ours(0x1000_3000, &row[..4], 8));
    }
}

/// Whether `key` has NOT been reported before in this process - the once-per-fact gate a
/// per-frame diagnostic goes through so a title that repeats a call every frame does not
/// repeat its line. Keys are the caller's own (fact-space tag XOR the fact's fields).
pub fn report_once(key: u64) -> bool {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    g.get_or_insert_with(HashSet::new).insert(key)
}
