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
pub fn rtt_writeback_texels() -> u32 {
    static CAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        vitaslop_platform::knobs::var("VITASLOP_GXM_RTT_WRITEBACK")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(256 * 256)
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
    if c.format & 0xF180_0000 != 0 {
        report_writeback_skipped(addr, c.format);
        return false;
    }
    if (w as usize) * (h as usize) * 4 > rgba.len() {
        return false;
    }
    // The guest's own row pitch, which is not the target's width: a surface is allocated at
    // its STRIDE and a writeback that ignored it would shear every row after the first.
    let stride = c.stride_pixels.max(w) as usize * 4;
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
        return false;
    }
    if report_once(0x7100_0000_0000_0000 ^ addr as u64) {
        let t = if w > 24 && h > 1 { rgba[(w as usize + 24) * 4..(w as usize + 24) * 4 + 4].to_vec() } else { Vec::new() };
        tracing::info!(target: "vitaslop::status", "gxm rtt writeback: first frame-end write of {addr:#010x} ({w}x{h}): texel(24,1)={t:?}");
    }
    report_writeback_target(addr, w, h, c.format, c.stride_pixels);
    for y in 0..rows {
        let src = y * (w as usize) * 4;
        write(addr.wrapping_add((y * stride) as u32), &rgba[src..src + cols * 4]);
    }
    true
}

/// Whether a bounded prefix of a buffer says NOTHING HAS BEEN WRITTEN HERE.
///
/// One repeated 4-byte word, which is what a zero fill and an allocator's poison fill both are
/// - MLB 12 fills a fresh allocation with `0xBAADCAFE` - and which no composed image is. A
/// shorter prefix than one word cannot answer the question and is not treated as empty.
pub fn nothing_written_here(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes.chunks_exact(4).all(|word| word == &bytes[..4])
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

/// Report - once per address - a render target the writeback DOES write, with the format and
/// stride it wrote it at.
///
/// A fix that cannot say what it touched is a fix nobody can bound: this is a blast-radius
/// question first (which of a title's targets now have their pixels in guest memory) and a
/// correctness one second (the SWIZZLE selector, bits 21:20 of the colour format, says the
/// byte order a CPU reader expects and a wrong one silently recolours whatever reads it).
fn report_writeback_target(addr: u32, w: u32, h: u32, format: u32, stride: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(addr) {
        return;
    }
    let swz = match (format >> 20) & 0x3 {
        1 => "ARGB",
        2 => "RGBA",
        3 => "BGRA",
        _ => "ABGR",
    };
    tracing::info!(
        target: "vitaslop::status",
        "gxm rtt writeback: {addr:#010x} {w}x{h} stride {stride} format {format:#010x} \
         (swizzle {swz}) - its rendered pixels now reach guest memory"
    );
}

/// Report - once per address - a render target whose guest format the writeback cannot
/// express.
fn report_writeback_skipped(addr: u32, format: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(addr) {
        return;
    }
    tracing::warn!(
        "gxm rtt writeback: {addr:#010x} is colour format {format:#010x}, which is not \
         U8U8U8U8 - the renderer holds it as 8-bit RGBA, so writing it back would put the \
         wrong pixel WIDTH in guest memory. Left alone; a guest that reads this target on the \
         CPU still reads whatever its allocator left there."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_poison_fill_and_a_zero_fill_are_both_unwritten_and_an_image_is_not() {
        assert!(nothing_written_here(&[0; 64]));
        assert!(nothing_written_here(&[0xFE, 0xCA, 0xAD, 0xBA].repeat(16)));
        let mut img = [0xFE, 0xCA, 0xAD, 0xBA].repeat(16);
        img[37] ^= 1;
        assert!(!nothing_written_here(&img));
        assert!(!nothing_written_here(&[1, 2, 3]));
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
