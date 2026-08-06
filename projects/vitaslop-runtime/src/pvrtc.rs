//! PVRTC (PowerVR Texture Compression) texel decode, for the GXM base formats
//! `PVRT2BPP` / `PVRT4BPP` (PVRTC1) and `PVRTII2BPP` / `PVRTII4BPP` (PVRTC2).
//!
//! # Why this exists
//!
//! PVRTC is the SGX543's native compressed format, so a Vita title uses it for most of its
//! world textures. Without a decode the capture leaves every unit bound to one of these
//! formats EMPTY, and the recompiled shader that samples it falls back to fixed-function -
//! which is how a whole race came out as untextured grey geometry.
//!
//! # Why it is not a per-block decode like BC
//!
//! A BC block is self-contained: sixteen texels, decodable from its own eight bytes. PVRTC
//! is not. Each block stores two low-frequency colours (A and B) that are BILINEARLY
//! UPSCALED across the whole image, plus a full-resolution per-texel modulation signal that
//! blends between the two upscaled images. So a single texel needs the FOUR blocks whose
//! centres surround it, and the format wraps at the edges by construction.
//!
//! # Sources
//!
//! Written from the published format description only (the Khronos Data Format PVRTC
//! specification and the PVR file-format specification) - facts, no code. Nothing here is
//! derived from any decoder implementation.
//!
//! # What is exact and what is not
//!
//! Exact: the block bit layout, colour expansion by bit replication, the bilinear upscale
//! and its sample positions, the 4bpp modulation weights, punch-through, and PVRTC2's
//! non-interpolated mode. The interpolation is carried out on the expanded 8-bit channels
//! rather than in the specification's fixed-point intermediate, which can differ by one
//! least-significant bit; nothing here rounds a channel into a visibly different value.
//!
//! NOT modelled: PVRTC1 2bpp's sub-sampled modulation (`M = 1`) and PVRTC2's local-palette
//! mode (`M = 1, H = 1`). Both are reported unconditionally the first time a texture uses
//! them rather than approximated silently - an approximation that is never announced is
//! indistinguishable on screen from a faithful decode, which is exactly how a wrong render
//! survives for sessions.

use crate::render::morton_index;

/// Which PVRTC variant a GXM base format selects.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Variant {
    /// PVRTC2 (`PVRTII*`) rather than PVRTC1 (`PVRT*`).
    pub two: bool,
    /// 4 bits per texel (4x4 texels a block) rather than 2 (8x4 texels a block).
    pub four_bpp: bool,
}

impl Variant {
    /// The GXM base format's variant, or `None` if it is not a PVRTC format.
    pub fn from_base_format(base_format: u32) -> Option<Variant> {
        Some(match base_format {
            0x80 => Variant { two: false, four_bpp: false },
            0x81 => Variant { two: false, four_bpp: true },
            0x82 => Variant { two: true, four_bpp: false },
            0x83 => Variant { two: true, four_bpp: true },
            _ => return None,
        })
    }

    /// Texels covered by one 8-byte block: 4x4 at 4bpp, 8x4 at 2bpp.
    pub fn block_size(self) -> (u32, u32) {
        if self.four_bpp { (4, 4) } else { (8, 4) }
    }
}

/// One block's decoded contents.
#[derive(Clone, Copy)]
struct Block {
    /// Colour A and colour B, expanded to straight RGBA8.
    a: [u8; 4],
    b: [u8; 4],
    /// Modulation mode flag.
    m: bool,
    /// PVRTC2 hard-transition flag (always false for PVRTC1).
    h: bool,
    /// The raw 32-bit modulation word.
    modulation: u32,
}

/// Expand an `n`-bit channel to 8 bits by replicating its high bits downward, which is what
/// the specification's expansion to ARGB:8888 does (e.g. a 4-bit `C3C2C1C0` becomes
/// `C3C2C1C0C3C2C1C0`). A zero channel stays zero and a full channel becomes 255.
fn expand(value: u32, bits: u32) -> u8 {
    debug_assert!((1..=8).contains(&bits));
    let mut v = value & ((1 << bits) - 1);
    let mut have = bits;
    while have < 8 {
        v = (v << bits) | (value & ((1 << bits) - 1));
        have += bits;
    }
    (v >> (have - 8)) as u8
}

/// Decode the 64-bit block at `off` into its two colours, its flags and its modulation word.
fn decode_block(bytes: &[u8], off: usize, variant: Variant) -> Block {
    let word = |i: usize| -> u32 {
        let s = off + i * 4;
        if s + 4 <= bytes.len() {
            u32::from_le_bytes([bytes[s], bytes[s + 1], bytes[s + 2], bytes[s + 3]])
        } else {
            0
        }
    };
    let modulation = word(0);
    let c = word(1);
    let m = c & 1 != 0;

    // Colour B occupies the top 16 bits in both variants, with its opacity flag at bit 31.
    let op_b = c & 0x8000_0000 != 0;
    let b = if op_b {
        [expand((c >> 26) & 0x1f, 5), expand((c >> 21) & 0x1f, 5), expand((c >> 16) & 0x1f, 5), 255]
    } else {
        // The 3-bit alpha is padded to 4 bits before expansion (a fully opaque translucent
        // encoding is 0b1110, not 0b1111 - the format cannot express 255 here).
        [
            expand((c >> 24) & 0xf, 4),
            expand((c >> 20) & 0xf, 4),
            expand((c >> 16) & 0xf, 4),
            expand(((c >> 28) & 0x7) << 1 | 1, 4),
        ]
    };

    // Colour A is 15 bits in PVRTC1 (its own opacity flag at bit 15) and 14 bits in PVRTC2,
    // whose bit 15 is the hard-transition flag and whose opacity comes from colour B's flag.
    let (op_a, h) = if variant.two { (op_b, c & 0x8000 != 0) } else { (c & 0x8000 != 0, false) };
    let a = if op_a {
        // Opaque A: 5 / 5 / 4 bits, the blue channel one bit shorter than colour B's.
        [expand((c >> 10) & 0x1f, 5), expand((c >> 5) & 0x1f, 5), expand((c >> 1) & 0xf, 4), 255]
    } else {
        [
            expand((c >> 8) & 0xf, 4),
            expand((c >> 4) & 0xf, 4),
            expand((c >> 1) & 0x7, 3),
            expand(((c >> 12) & 0x7) << 1, 4),
        ]
    };
    Block { a, b, m, h, modulation }
}

/// Byte offset of block `(bx, by)` in a GXM PVRTC image.
///
/// GXM stores these textures with the block grid in the GPU's Morton order over a
/// power-of-two-padded grid, exactly as it does for the block-compressed formats, so the
/// same addressing applies here.
fn block_offset(bx: u32, by: u32, blocks_x: u32, blocks_y: u32, swizzled: bool) -> usize {
    let index = if swizzled {
        morton_index(bx, by, blocks_x.next_power_of_two(), blocks_y.next_power_of_two())
    } else {
        by * blocks_x + bx
    };
    index as usize * 8
}

/// Decode the texel at `(x, y)` of a PVRTC image.
///
/// `bytes` is the face's pixel data, `swizzled` whether the block grid is Morton-ordered
/// (the GXM `SWIZZLED` texture types). `(x, y)` must already be inside `(width, height)`.
pub fn texel(
    bytes: &[u8],
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    variant: Variant,
    swizzled: bool,
) -> [u8; 4] {
    let (bw, bh) = variant.block_size();
    let blocks_x = width.div_ceil(bw).max(1);
    let blocks_y = height.div_ceil(bh).max(1);

    // The block CONTAINING the texel supplies the modulation and (in PVRTC2's
    // non-interpolated modes) the colours.
    let own = decode_block(
        bytes,
        block_offset(x / bw, y / bh, blocks_x, blocks_y, swizzled),
        variant,
    );

    // A block's two colours are considered to sit at its CENTRE, so the four blocks whose
    // centres surround this texel are the ones to interpolate, and they wrap.
    let shift_x = bw / 2;
    let shift_y = bh / 2;
    let sx = (x + width - shift_x) % width;
    let sy = (y + height - shift_y) % height;
    let bx0 = sx / bw;
    let by0 = sy / bh;
    let xr = sx % bw;
    let yr = sy % bh;
    let bx1 = (bx0 + 1) % blocks_x;
    let by1 = (by0 + 1) % blocks_y;

    // PVRTC2 with the hard-transition flag set skips the upscale in this region: the block's
    // own colours apply directly, which is what makes a hard edge possible at all.
    let hard = variant.two && own.h;
    let (a, b) = if hard {
        (own.a, own.b)
    } else {
        let n00 = decode_block(bytes, block_offset(bx0, by0, blocks_x, blocks_y, swizzled), variant);
        let n10 = decode_block(bytes, block_offset(bx1, by0, blocks_x, blocks_y, swizzled), variant);
        let n01 = decode_block(bytes, block_offset(bx0, by1, blocks_x, blocks_y, swizzled), variant);
        let n11 = decode_block(bytes, block_offset(bx1, by1, blocks_x, blocks_y, swizzled), variant);
        let w00 = (bw - xr) * (bh - yr);
        let w10 = xr * (bh - yr);
        let w01 = (bw - xr) * yr;
        let w11 = xr * yr;
        let total = bw * bh;
        let lerp = |p: fn(&Block) -> [u8; 4]| -> [u8; 4] {
            let (c00, c10, c01, c11) = (p(&n00), p(&n10), p(&n01), p(&n11));
            let mut out = [0u8; 4];
            for (ch, o) in out.iter_mut().enumerate() {
                let s = c00[ch] as u32 * w00
                    + c10[ch] as u32 * w10
                    + c01[ch] as u32 * w01
                    + c11[ch] as u32 * w11;
                *o = (s / total) as u8;
            }
            out
        };
        (lerp(|b| b.a), lerp(|b| b.b))
    };

    // Modulation: how far this texel sits between the two upscaled images, in eighths.
    let tx = x % bw;
    let ty = y % bh;
    let (weight, punched) = if variant.four_bpp {
        let bit = (ty * 4 + tx) * 2;
        let v = (own.modulation >> bit) & 0x3;
        if own.m && !hard {
            // Punch-through: the middle two codes are a half blend, and one of them forces
            // the texel fully transparent.
            match v {
                0 => (0, false),
                1 => (4, false),
                2 => (4, true),
                _ => (8, false),
            }
        } else if own.m && hard {
            // PVRTC2 M=1,H=1 is the local-palette mode, whose palette construction is not
            // established here. Report it and fall back to the plain two-colour blend rather
            // than invent a palette.
            report_unmodelled("PVRTC2 local-palette mode (M=1, H=1)");
            ([0, 3, 5, 8][v as usize], false)
        } else {
            ([0, 3, 5, 8][v as usize], false)
        }
    } else {
        // 2bpp carries one modulation bit per texel in its base mode.
        if own.m {
            report_unmodelled("PVRTC 2bpp sub-sampled modulation (M=1)");
        }
        let bit = ty * 8 + tx;
        (((own.modulation >> bit) & 1) * 8, false)
    };

    if punched {
        return [0, 0, 0, 0];
    }
    let mut out = [0u8; 4];
    for (ch, o) in out.iter_mut().enumerate() {
        *o = ((a[ch] as u32 * (8 - weight) + b[ch] as u32 * weight) / 8) as u8;
    }
    out
}

/// Report - once per distinct case - that a PVRTC sub-mode this decoder does not model was
/// encountered, so the texels it covers are an approximation rather than a decode.
fn report_unmodelled(what: &str) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(what.to_string()) {
        return;
    }
    eprintln!(
        "pvrtc: {what} is NOT modelled - the texels it covers are approximated by the plain \
         two-colour blend and are not a faithful decode"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit replication must map a full channel to 255 and an empty one to 0, at every width
    /// the format uses. A channel that saturates to 254 instead of 255 tints a whole texture.
    #[test]
    fn expansion_covers_the_full_range() {
        for bits in 3..=5 {
            assert_eq!(expand(0, bits), 0, "zero at {bits} bits");
            assert_eq!(expand((1 << bits) - 1, bits), 255, "full at {bits} bits");
        }
        assert_eq!(expand(0b1000, 4), 0b1000_1000);
        assert_eq!(expand(0b10000, 5), 0b10000_100);
    }

    /// A block whose colours are both the same and whose modulation is uniform must decode to
    /// exactly that colour everywhere, whatever the modulation code says - the blend of a
    /// colour with itself. This is the one property that holds independently of the weights,
    /// so it checks the addressing and the interpolation without assuming either.
    #[test]
    fn a_uniform_image_decodes_to_its_colour() {
        let variant = Variant { two: false, four_bpp: true };
        // Opaque A = B = white: colour word with both opacity flags and all channel bits set.
        let colour: u32 = 0xFFFF_FFFE; // every channel bit set, M = 0
        let mut bytes = Vec::new();
        for _ in 0..4 {
            bytes.extend_from_slice(&0xAAAA_AAAAu32.to_le_bytes()); // mixed modulation
            bytes.extend_from_slice(&colour.to_le_bytes());
        }
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(
                    texel(&bytes, 8, 8, x, y, variant, false),
                    [255, 255, 255, 255],
                    "at ({x}, {y})"
                );
            }
        }
    }

    /// Modulation code 0 must give colour A and code 3 colour B, exactly, when the two
    /// colours are uniform across the image (so interpolation cannot move them).
    #[test]
    fn modulation_endpoints_select_a_and_b() {
        let variant = Variant { two: false, four_bpp: true };
        // A opaque black, B opaque white: A payload zero, B payload all ones.
        let colour: u32 = 0xFFFF_0000 | 0x8000;
        let mut bytes = Vec::new();
        for _ in 0..4 {
            // Texel (0,0) -> code 0, texel (1,0) -> code 3.
            bytes.extend_from_slice(&0b1100u32.to_le_bytes());
            bytes.extend_from_slice(&colour.to_le_bytes());
        }
        assert_eq!(texel(&bytes, 8, 8, 0, 0, variant, false), [0, 0, 0, 255]);
        assert_eq!(texel(&bytes, 8, 8, 1, 0, variant, false), [255, 255, 255, 255]);
    }

    /// A 2bpp block covers 8x4 texels and a 4bpp block 4x4 - the sizes the whole addressing
    /// rests on, and getting one wrong silently halves or doubles a texture.
    #[test]
    fn block_sizes_match_the_bit_rate() {
        assert_eq!(Variant::from_base_format(0x81).unwrap().block_size(), (4, 4));
        assert_eq!(Variant::from_base_format(0x80).unwrap().block_size(), (8, 4));
        assert_eq!(Variant::from_base_format(0x83).unwrap(), Variant { two: true, four_bpp: true });
        assert!(Variant::from_base_format(0x0c).is_none());
    }
}
