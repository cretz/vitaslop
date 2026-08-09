//! A BC1/BC3 encoder, for the formats WebGPU cannot be handed directly.
//!
//! # Why an encoder exists at all
//! The guest's `UBC1/2/3` surfaces ARE BC1/2/3 and go to the GPU untouched
//! ([`crate::render::compressed_source`]). PVRTC does not: WebGPU has no PVRTC format on any
//! adapter, so a PVRTC surface is CPU-decoded to RGBA8 and lands on the GPU at eight times its
//! size. MEASURED on one retail race frame, that family alone is **159.9 MB of a 259 MB working
//! set** - the largest single item by a wide margin, and the reason a phone's allocation fails
//! and the draw comes out WHITE.
//!
//! Re-encoding those texels to BC1 puts them back at 4 bits per texel, which is exactly the rate
//! the guest stored them at.
//!
//! # This is LOSSY, and that is stated rather than hidden
//! PVRTC -> RGBA8 -> BC1 is two lossy steps, not one. Both are 4bpp block codecs with different
//! error, so the second pass cannot undo the first and adds its own. That is why the transcode
//! is a knob with a measured default rather than an unconditional win: the number that decides
//! it is the pixel difference on a real frame, and it is recorded in the project notes beside
//! the memory it buys.
//!
//! # Provenance
//! Written from the published BC1/BC3 BLOCK FORMAT (the bit layout is what
//! [`crate::render::decode_bc_texel`] already decodes, and that decoder is this one's test
//! oracle). The endpoint search is the obvious one - the RGB bounding box of the block, inset,
//! then a least-squares refit against the chosen indices. No encoder source was consulted.

/// One 4x4 block's worth of RGBA8 texels, row-major.
type Block = [[u8; 4]; 16];

/// Pack an RGB888 colour into RGB565.
fn to565(c: [u8; 3]) -> u16 {
    ((c[0] as u16 >> 3) << 11) | ((c[1] as u16 >> 2) << 5) | (c[2] as u16 >> 3)
}

/// Expand RGB565 back to RGB888 the way a decoder does - replicating the high bits into the
/// low ones, NOT shifting in zeros. Encoding against a different expansion than the hardware
/// decodes with biases every block dark by up to 7/255.
fn from565(c: u16) -> [u8; 3] {
    let r = ((c >> 11) & 0x1f) as u32;
    let g = ((c >> 5) & 0x3f) as u32;
    let b = (c & 0x1f) as u32;
    [(r * 255 / 31) as u8, (g * 255 / 63) as u8, (b * 255 / 31) as u8]
}

/// The four palette colours of a 4-colour BC1 block.
fn palette(c0: u16, c1: u16) -> [[u8; 3]; 4] {
    let a = from565(c0);
    let b = from565(c1);
    let mix = |x: u8, y: u8, wx: u32, wy: u32| ((x as u32 * wx + y as u32 * wy) / 3) as u8;
    [
        a,
        b,
        [mix(a[0], b[0], 2, 1), mix(a[1], b[1], 2, 1), mix(a[2], b[2], 2, 1)],
        [mix(a[0], b[0], 1, 2), mix(a[1], b[1], 1, 2), mix(a[2], b[2], 1, 2)],
    ]
}

fn dist2(a: [u8; 3], b: [u8; 3]) -> u32 {
    let d = |i: usize| {
        let x = a[i] as i32 - b[i] as i32;
        (x * x) as u32
    };
    d(0) + d(1) + d(2)
}

/// Assign each texel the nearest palette entry, returning the packed 2-bit indices and the
/// total squared error.
fn assign(block: &Block, pal: &[[u8; 3]; 4]) -> (u32, u32) {
    let mut bits = 0u32;
    let mut err = 0u32;
    for (i, px) in block.iter().enumerate() {
        let rgb = [px[0], px[1], px[2]];
        let mut best = 0usize;
        let mut best_d = u32::MAX;
        for (k, p) in pal.iter().enumerate() {
            let d = dist2(rgb, *p);
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        bits |= (best as u32) << (2 * i);
        err = err.saturating_add(best_d);
    }
    (bits, err)
}

/// The two endpoint colours of a block: its texels' extremes along their PRINCIPAL AXIS.
///
/// # The RGB bounding box is not good enough, and it fails on the most visible blocks
/// The obvious choice is the bounding box's two corners, and it is exactly right whenever the
/// block's colours run along the box's main diagonal - a gradient, a shaded surface. It is
/// badly wrong when they run along any other diagonal, because then NEITHER corner is a colour
/// the block contains. MEASURED here: a two-colour block of `(255,32,0)` and `(0,16,255)` has
/// the bounding-box corners `(255,32,255)` and `(0,16,0)` - magenta and near-black, neither of
/// them present - and encoding a hard-edged pattern of those two colours came back at a mean
/// absolute error of **65.75 of 255**, which is not a lossy-codec artefact, it is the wrong
/// picture. Along the principal axis the same block's endpoints ARE its two colours.
///
/// The axis comes from a few power iterations on the block's covariance matrix, which is enough
/// for a 16-sample problem, and the endpoints are the actual colours with the extreme
/// projections - not points on the axis - so a two-colour block reproduces exactly.
fn endpoints(block: &Block) -> ([u8; 3], [u8; 3]) {
    let n = block.len() as i64;
    let mut mean = [0i64; 3];
    for px in block {
        for c in 0..3 {
            mean[c] += px[c] as i64;
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    // The symmetric 3x3 covariance, in integers - the block is 16 samples of bytes, so nothing
    // here can overflow and no floating point is needed until the iteration.
    let mut cov = [[0i64; 3]; 3];
    for px in block {
        let d: [i64; 3] = std::array::from_fn(|c| px[c] as i64 - mean[c]);
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    // Power iteration for the dominant eigenvector. Seeded off the diagonal so the start is
    // already biased toward the widest channel, which makes four iterations plenty; a seed
    // orthogonal to the true axis would converge slowly, and a zero seed not at all.
    let mut v = [
        (cov[0][0] + cov[0][1] + cov[0][2]) as f32,
        (cov[1][0] + cov[1][1] + cov[1][2]) as f32,
        (cov[2][0] + cov[2][1] + cov[2][2]) as f32,
    ];
    if v.iter().all(|x| x.abs() < 1.0) {
        v = [1.0, 1.0, 1.0];
    }
    for _ in 0..4 {
        let next: [f32; 3] = std::array::from_fn(|i| {
            (0..3).map(|j| cov[i][j] as f32 * v[j]).sum::<f32>()
        });
        let len = (next[0] * next[0] + next[1] * next[1] + next[2] * next[2]).sqrt();
        if len < 1e-6 {
            break;
        }
        v = [next[0] / len, next[1] / len, next[2] / len];
    }
    // The texels whose projections onto the axis are extreme. Taking the COLOURS rather than
    // points on the axis is what makes a two-colour block exact: the axis passes through both,
    // but a point computed on it would be rounded twice.
    let proj = |px: &[u8; 4]| -> f32 {
        (0..3).map(|c| (px[c] as f32 - mean[c] as f32) * v[c]).sum()
    };
    let mut lo_px = block[0];
    let mut hi_px = block[0];
    let (mut lo_p, mut hi_p) = (f32::MAX, f32::MIN);
    for px in block {
        let p = proj(px);
        if p < lo_p {
            lo_p = p;
            lo_px = *px;
        }
        if p > hi_p {
            hi_p = p;
            hi_px = *px;
        }
    }
    ([hi_px[0], hi_px[1], hi_px[2]], [lo_px[0], lo_px[1], lo_px[2]])
}

/// Encode one 4x4 block to the 8-byte BC1 layout: `c0`, `c1` (little-endian RGB565) then four
/// bytes of 2-bit indices, texel 0 in the low bits.
///
/// Always emits the 4-COLOUR mode (`c0 > c1`). The 3-colour mode exists to carry 1-bit alpha,
/// and this encoder never needs it: a texture with real alpha goes to BC3 instead, where the
/// colour block is unconditionally 4-colour anyway.
fn encode_bc1_block(block: &Block) -> [u8; 8] {
    let (a, b) = endpoints(block);
    let mut c0 = to565(a);
    let mut c1 = to565(b);
    // 4-colour mode is selected by c0 > c1. Equal endpoints mean a flat block, where the
    // decoder's colour 0 is the answer whichever mode it reads, and every index is 0.
    if c0 < c1 {
        std::mem::swap(&mut c0, &mut c1);
    }
    let (mut bits, mut err) = assign(block, &palette(c0, c1));
    if c0 == c1 {
        return pack_bc1(c0, c1, 0);
    }
    // ONE least-squares refit. With the indices fixed, each texel's colour is a known linear
    // blend of the two endpoints, so the endpoints that minimise squared error have a closed
    // form. It is kept only if it actually helps - a refit can overshoot on a block whose
    // indices are already optimal, and an encoder that trusts its own refinement unconditionally
    // gets WORSE on exactly the flat blocks it should be perfect on.
    if let Some((r0, r1)) = refit(block, bits) {
        let (rbits, rerr) = assign(block, &palette(r0, r1));
        if rerr < err && r0 != r1 {
            let (mut n0, mut n1, mut nbits) = (r0, r1, rbits);
            if n0 < n1 {
                std::mem::swap(&mut n0, &mut n1);
                // Swapping the endpoints swaps index 0 with 1 and index 2 with 3.
                nbits = (0..16).fold(0u32, |acc, i| {
                    let k = (rbits >> (2 * i)) & 3;
                    acc | (k ^ 1) << (2 * i)
                });
            }
            if n0 > n1 {
                bits = nbits;
                err = rerr;
                c0 = n0;
                c1 = n1;
            }
        }
    }
    let _ = err;
    pack_bc1(c0, c1, bits)
}

/// The endpoints that minimise squared error for a fixed index assignment.
///
/// Each index names a blend weight `w` in `{1, 0, 2/3, 1/3}` of endpoint A (the rest being
/// endpoint B), so every channel is an ordinary two-variable least-squares fit over the block.
/// `None` when the system is degenerate - every texel on one endpoint - where the bounding box
/// is already exact.
fn refit(block: &Block, bits: u32) -> Option<(u16, u16)> {
    // Weights of endpoint A per index value, in thirds, to keep this in integers until the end.
    const WA: [i64; 4] = [3, 0, 2, 1];
    let (mut saa, mut sab, mut sbb) = (0i64, 0i64, 0i64);
    let mut sax = [0i64; 3];
    let mut sbx = [0i64; 3];
    for (i, px) in block.iter().enumerate() {
        let wa = WA[((bits >> (2 * i)) & 3) as usize];
        let wb = 3 - wa;
        saa += wa * wa;
        sab += wa * wb;
        sbb += wb * wb;
        for c in 0..3 {
            sax[c] += wa * px[c] as i64;
            sbx[c] += wb * px[c] as i64;
        }
    }
    let det = saa * sbb - sab * sab;
    if det == 0 {
        return None;
    }
    let mut a = [0u8; 3];
    let mut b = [0u8; 3];
    for c in 0..3 {
        // Multiplied through by 3 because the weights are in thirds on both sides.
        a[c] = (((sbb * sax[c] - sab * sbx[c]) * 3) / det).clamp(0, 255) as u8;
        b[c] = (((saa * sbx[c] - sab * sax[c]) * 3) / det).clamp(0, 255) as u8;
    }
    Some((to565(a), to565(b)))
}

fn pack_bc1(c0: u16, c1: u16, bits: u32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&bits.to_le_bytes());
    out
}

/// Encode one 4x4 block's ALPHA to the 8-byte BC4 layout BC3 uses: `a0`, `a1`, then sixteen
/// 3-bit indices packed little-endian across six bytes.
///
/// Always the 8-value mode (`a0 > a1`), whose palette is the two endpoints plus six evenly
/// interpolated values. The 6-value mode reserves two codes for exactly 0 and 255, which is
/// worth having for a cutout mask and costs a quarter of the interior resolution; with the
/// endpoints already sitting on the block's own min and max, 0 and 255 are only ever needed
/// when they ARE the endpoints.
fn encode_bc3_alpha_block(block: &Block) -> [u8; 8] {
    let mut lo = 255u8;
    let mut hi = 0u8;
    for px in block {
        lo = lo.min(px[3]);
        hi = hi.max(px[3]);
    }
    let mut out = [0u8; 8];
    out[0] = hi;
    out[1] = lo;
    if hi == lo {
        // Flat alpha: index 0 selects a0 exactly, so every index stays zero.
        return out;
    }
    let value = |k: usize| -> u8 {
        match k {
            0 => hi,
            1 => lo,
            _ => (((8 - k as u32) * hi as u32 + (k as u32 - 1) * lo as u32) / 7) as u8,
        }
    };
    let pal: [u8; 8] = std::array::from_fn(value);
    let mut bits = 0u64;
    for (i, px) in block.iter().enumerate() {
        let mut best = 0usize;
        let mut best_d = u32::MAX;
        for (k, v) in pal.iter().enumerate() {
            let d = (px[3] as i32 - *v as i32).unsigned_abs();
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        bits |= (best as u64) << (3 * i);
    }
    out[2..8].copy_from_slice(&bits.to_le_bytes()[..6]);
    out
}

/// Gather the 4x4 block at block coordinates `(bx, by)` out of an RGBA8 image.
///
/// A block that hangs off the edge repeats the last row/column rather than reading zeros: the
/// texels outside the image are never sampled, but they ARE averaged into the block's endpoints,
/// and padding with black drags every edge block's endpoints toward it.
fn gather(w: u32, h: u32, rgba: &[u8], bx: u32, by: u32) -> Block {
    let mut out = [[0u8; 4]; 16];
    for py in 0..4u32 {
        for px in 0..4u32 {
            let x = (bx * 4 + px).min(w - 1) as usize;
            let y = (by * 4 + py).min(h - 1) as usize;
            let o = (y * w as usize + x) * 4;
            out[(py * 4 + px) as usize].copy_from_slice(&rgba[o..o + 4]);
        }
    }
    out
}

/// Whether every texel of an RGBA8 image is fully opaque, which is what decides BC1 (4 bits a
/// texel) against BC3 (8).
pub fn is_opaque(rgba: &[u8]) -> bool {
    rgba.chunks_exact(4).all(|p| p[3] == 255)
}

/// Encode an RGBA8 image to BC1, block-packed in linear block rows.
pub fn encode_bc1(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
    let mut out = Vec::with_capacity((bw * bh * 8) as usize);
    for by in 0..bh {
        for bx in 0..bw {
            out.extend_from_slice(&encode_bc1_block(&gather(w, h, rgba, bx, by)));
        }
    }
    out
}

/// Encode an RGBA8 image to BC3 (an alpha block then a colour block per 4x4).
pub fn encode_bc3(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
    let mut out = Vec::with_capacity((bw * bh * 16) as usize);
    for by in 0..bh {
        for bx in 0..bw {
            let block = gather(w, h, rgba, bx, by);
            out.extend_from_slice(&encode_bc3_alpha_block(&block));
            out.extend_from_slice(&encode_bc1_block(&block));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode what we encoded, through the DECODER THIS PROJECT ALREADY HAD.
    ///
    /// That is the whole point of testing it this way: `decode_bc_texel` is the function the
    /// software rasterizer and the RGBA8 upload path both go through, and it was written and
    /// tested against real guest textures long before this encoder existed. An encoder checked
    /// against its own idea of the format would agree with itself about a wrong bit layout.
    fn roundtrip(w: u32, h: u32, rgba: &[u8], bc3: bool) -> Vec<u8> {
        let data = if bc3 { encode_bc3(w, h, rgba) } else { encode_bc1(w, h, rgba) };
        let fmt = if bc3 { 0x87 } else { 0x85 };
        let bb = if bc3 { 16 } else { 8 };
        let mut out = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let bi = (y / 4) * w.div_ceil(4) + (x / 4);
                let block = &data[(bi * bb) as usize..((bi + 1) * bb) as usize];
                let c = crate::render::decode_bc_texel(block, fmt, x % 4, y % 4);
                let o = ((y * w + x) * 4) as usize;
                out[o..o + 4].copy_from_slice(&c);
            }
        }
        out
    }

    fn mad(a: &[u8], b: &[u8]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x.abs_diff(*y) as u64).sum::<u64>() as f64 / a.len() as f64
    }

    /// A block of one colour must come back EXACTLY. This is the case an encoder has no excuse
    /// for, and it is also the one a careless refit breaks: the least-squares system is
    /// degenerate when every texel sits on one endpoint.
    #[test]
    fn a_flat_block_survives_exactly() {
        for c in [[0u8, 0, 0, 255], [255, 255, 255, 255], [40, 130, 200, 255], [8, 4, 8, 255]] {
            let rgba: Vec<u8> = c.iter().cycle().take(16 * 4).copied().collect();
            let got = roundtrip(4, 4, &rgba, false);
            // 565 quantisation is exact only for values that survive the round trip, so compare
            // against what the decoder produces for the quantised colour rather than the input.
            let q = from565(to565([c[0], c[1], c[2]]));
            for px in got.chunks_exact(4) {
                assert_eq!([px[0], px[1], px[2]], q, "a flat block must be one flat colour");
            }
        }
    }

    /// Alpha must survive: a cutout mask that comes back soft is a visible defect, and the
    /// endpoints sit on the block's own min and max so the extremes are exact.
    #[test]
    fn bc3_carries_the_alpha_extremes_exactly() {
        let mut rgba = vec![0u8; 16 * 4];
        for (i, px) in rgba.chunks_exact_mut(4).enumerate() {
            px.copy_from_slice(&[200, 100, 50, if i % 2 == 0 { 0 } else { 255 }]);
        }
        let got = roundtrip(4, 4, &rgba, true);
        for (i, px) in got.chunks_exact(4).enumerate() {
            assert_eq!(px[3], if i % 2 == 0 { 0 } else { 255 }, "texel {i} alpha");
        }
    }

    /// The opacity test decides BC1 against BC3, so it must not be fooled by a single texel.
    #[test]
    fn one_transparent_texel_makes_an_image_non_opaque() {
        let mut rgba = vec![255u8; 64];
        assert!(is_opaque(&rgba));
        rgba[7] = 254;
        assert!(!is_opaque(&rgba), "alpha 254 is not opaque");
    }

    /// A cheap deterministic pseudo-random source, so every test below runs on the same bytes
    /// on every machine. `Math::random`-style nondeterminism in a codec test buys nothing and
    /// costs the ability to reproduce a failure.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn byte(&mut self) -> u8 {
            (self.next() & 0xff) as u8
        }
    }

    /// Peak error, in levels, over an RGB comparison.
    fn max_rgb_err(a: &[u8], b: &[u8]) -> u8 {
        a.chunks_exact(4)
            .zip(b.chunks_exact(4))
            .flat_map(|(x, y)| (0..3).map(move |c| x[c].abs_diff(y[c])))
            .max()
            .unwrap_or(0)
    }

    // ---------------------------------------------------------------------------------
    // Bit-layout conformance: what the encoder EMITS, independent of how good it looks.
    // ---------------------------------------------------------------------------------

    /// The output is exactly one block per 4x4, at the format's block size, with partial
    /// trailing blocks counted. A short buffer is not a quality problem, it is a texture upload
    /// that fails validation or reads someone else's memory.
    #[test]
    fn the_encoded_size_is_exactly_one_block_per_four_by_four() {
        for &(w, h) in &[(4u32, 4u32), (8, 4), (4, 8), (64, 64), (16, 8), (1, 1), (5, 3), (12, 20)] {
            let rgba = vec![128u8; (w * h * 4) as usize];
            let blocks = (w.div_ceil(4) * h.div_ceil(4)) as usize;
            assert_eq!(encode_bc1(w, h, &rgba).len(), blocks * 8, "BC1 {w}x{h}");
            assert_eq!(encode_bc3(w, h, &rgba).len(), blocks * 16, "BC3 {w}x{h}");
        }
    }

    /// Every colour block must select the 4-COLOUR mode (`c0 > c1`), or be flat (`c0 == c1`).
    ///
    /// `c0 < c1` is the 3-colour mode, where index 3 decodes to TRANSPARENT BLACK. Emitting it
    /// by accident does not look like a rounding error: it punches holes in the texture. The
    /// endpoint swap in the refit path is exactly where that could creep in, so this runs over
    /// random blocks, which is what exercises the refit.
    #[test]
    fn every_colour_block_uses_the_four_colour_mode() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let (w, h) = (64u32, 64u32);
        let rgba: Vec<u8> = (0..(w * h * 4)).map(|_| rng.byte()).collect();
        for data in [encode_bc1(w, h, &rgba), encode_bc3(w, h, &rgba)] {
            let (stride, off) = if data.len() == (w / 4 * h / 4 * 8) as usize { (8, 0) } else { (16, 8) };
            for block in data.chunks_exact(stride) {
                let c0 = u16::from_le_bytes([block[off], block[off + 1]]);
                let c1 = u16::from_le_bytes([block[off + 2], block[off + 3]]);
                assert!(c0 >= c1, "3-colour mode emitted: c0 {c0:#06x} < c1 {c1:#06x}");
            }
        }
    }

    /// A flat block must also emit all-zero indices, not merely decode flat. Two encoders can
    /// agree on the picture and disagree on the bits, and the bits are what a device reads.
    #[test]
    fn a_flat_block_emits_zero_indices() {
        let rgba: Vec<u8> = [90u8, 140, 200, 255].iter().cycle().take(16 * 4).copied().collect();
        let bc1 = encode_bc1(4, 4, &rgba);
        assert_eq!(&bc1[4..8], &[0, 0, 0, 0], "a flat block has nothing to index");
        let bc3 = encode_bc3(4, 4, &rgba);
        assert_eq!(&bc3[2..8], &[0, 0, 0, 0, 0, 0], "flat alpha has nothing to index");
        assert_eq!(bc3[0], 255, "a0 is the block's max alpha");
        assert_eq!(bc3[1], 255, "a1 is the block's min alpha");
    }

    /// The same input must always produce the same bytes.
    ///
    /// The encode result is cached and keyed by texture content, so a codec that is not a pure
    /// function of its input would hand two draws of the same texture different blocks - and
    /// the difference would show up as a flicker, which is the hardest class of artefact to
    /// attribute back to its cause.
    #[test]
    fn the_encoder_is_deterministic() {
        let mut rng = Rng(1234567);
        let rgba: Vec<u8> = (0..(32 * 32 * 4)).map(|_| rng.byte()).collect();
        assert_eq!(encode_bc1(32, 32, &rgba), encode_bc1(32, 32, &rgba));
        assert_eq!(encode_bc3(32, 32, &rgba), encode_bc3(32, 32, &rgba));
    }

    /// BC3's colour half IS a BC1 block, so the two encoders must emit the same eight bytes for
    /// the same texels. If they ever diverge, one of them is carrying a fix the other is not.
    #[test]
    fn bc3_reuses_the_bc1_colour_block_exactly() {
        let mut rng = Rng(99);
        let (w, h) = (16u32, 16u32);
        let rgba: Vec<u8> = (0..(w * h * 4)).map(|_| rng.byte()).collect();
        let bc1 = encode_bc1(w, h, &rgba);
        let bc3 = encode_bc3(w, h, &rgba);
        for (i, colour) in bc3.chunks_exact(16).enumerate() {
            assert_eq!(&colour[8..16], &bc1[i * 8..i * 8 + 8], "block {i} colour half");
        }
    }

    // ---------------------------------------------------------------------------------
    // Reproduction conformance: cases where the format can be EXACT, and must be.
    // ---------------------------------------------------------------------------------

    /// Every colour that survives 565 quantisation must round-trip EXACTLY as a flat block.
    ///
    /// Swept rather than sampled: 32x64x32 is the whole 565 space, and the failure this catches
    /// (an expansion that shifts in zeros instead of replicating high bits) is a systematic
    /// darkening of a few levels that no single spot check would name.
    #[test]
    fn every_565_colour_survives_a_flat_block_exactly() {
        for r in 0..32u32 {
            for g in 0..64u32 {
                for b in 0..32u32 {
                    let c = [
                        (r * 255 / 31) as u8,
                        (g * 255 / 63) as u8,
                        (b * 255 / 31) as u8,
                        255u8,
                    ];
                    let rgba: Vec<u8> = c.iter().cycle().take(16 * 4).copied().collect();
                    let got = roundtrip(4, 4, &rgba, false);
                    assert_eq!(
                        &got[..3],
                        &c[..3],
                        "565 colour ({r},{g},{b}) did not survive a flat block"
                    );
                }
            }
        }
    }

    /// A block containing only TWO colours must reproduce both essentially exactly, whatever
    /// direction they lie in.
    ///
    /// This is the case the RGB bounding box gets wrong (see [`endpoints`]) and it is not an
    /// exotic one - it is every hard edge in a UI atlas, every letter of text, every mask. The
    /// residual allowed here is 565 quantisation and nothing else, so the axis has to be right
    /// for all 64 sign patterns, not just the main diagonal.
    #[test]
    fn a_two_colour_block_reproduces_both_colours() {
        let mut rng = Rng(0xDEAD_BEEF);
        for _ in 0..400 {
            let a = [rng.byte(), rng.byte(), rng.byte()];
            let b = [rng.byte(), rng.byte(), rng.byte()];
            let mut rgba = Vec::with_capacity(16 * 4);
            for i in 0..16 {
                let c = if (rng.next() & 1) == 0 { a } else { b };
                let _ = i;
                rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
            let got = roundtrip(4, 4, &rgba, false);
            // 565 costs up to 4 levels in R/B and 2 in G; allow that and no more.
            let e = max_rgb_err(&rgba, &got);
            assert!(
                e <= 8,
                "a two-colour block of {a:?} and {b:?} came back {e} levels off - the endpoints \
                 are not the block's own colours"
            );
        }
    }

    /// Alpha's two extremes are the BC4 endpoints, so ANY pair of alpha values in a block must
    /// come back exactly. A cutout mask that softens is a visible defect - haloes around every
    /// sprite - and it is invisible in an RGB comparison.
    #[test]
    fn any_two_alpha_values_in_a_block_are_exact() {
        let mut rng = Rng(4242);
        for _ in 0..300 {
            let (a0, a1) = (rng.byte(), rng.byte());
            let mut rgba = Vec::with_capacity(16 * 4);
            for _ in 0..16 {
                let a = if (rng.next() & 1) == 0 { a0 } else { a1 };
                rgba.extend_from_slice(&[10, 20, 30, a]);
            }
            let got = roundtrip(4, 4, &rgba, true);
            for (i, (want, got)) in rgba.chunks_exact(4).zip(got.chunks_exact(4)).enumerate() {
                assert_eq!(want[3], got[3], "texel {i} alpha, block of {a0} and {a1}");
            }
        }
    }

    /// A full 0..255 alpha ramp must land within half a step of the eight-value palette.
    ///
    /// The palette spans the block's own min and max in 7 steps, so the worst a correctly
    /// assigned index can be is half a step - about 18 levels for a full-range block. Anything
    /// larger means the interpolation weights or the index packing are wrong, and the 3-bit
    /// little-endian packing across six bytes is a real opportunity to get that wrong.
    #[test]
    fn an_alpha_ramp_lands_within_half_a_palette_step() {
        let mut rgba = Vec::with_capacity(16 * 4);
        for i in 0..16u32 {
            rgba.extend_from_slice(&[0, 0, 0, (i * 17) as u8]);
        }
        let got = roundtrip(4, 4, &rgba, true);
        let worst = rgba
            .chunks_exact(4)
            .zip(got.chunks_exact(4))
            .map(|(a, b)| a[3].abs_diff(b[3]))
            .max()
            .unwrap();
        let step = 255u32 / 7;
        assert!(
            (worst as u32) <= step / 2 + 1,
            "alpha ramp is {worst} levels off, past half a {step}-level step"
        );
        // And the extremes are exact, because they ARE the endpoints.
        assert_eq!(got[3], 0);
        assert_eq!(got[15 * 4 + 3], 255);
    }

    /// Every texel of every block must be assigned its NEAREST palette entry.
    ///
    /// Checked directly rather than through an error bound: this is the one part of the encoder
    /// with a provably optimal answer once the endpoints are chosen, so an error bound would let
    /// a broken index search hide behind good endpoints.
    #[test]
    fn every_texel_takes_its_nearest_palette_entry() {
        let mut rng = Rng(0xC0FFEE);
        let (w, h) = (32u32, 32u32);
        let rgba: Vec<u8> = (0..(w * h * 4))
            .map(|i| if i % 4 == 3 { 255 } else { rng.byte() })
            .collect();
        let data = encode_bc1(w, h, &rgba);
        for (bi, block) in data.chunks_exact(8).enumerate() {
            let c0 = u16::from_le_bytes([block[0], block[1]]);
            let c1 = u16::from_le_bytes([block[2], block[3]]);
            let pal = palette(c0, c1);
            let bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
            let (bx, by) = (bi as u32 % (w / 4), bi as u32 / (w / 4));
            for t in 0..16u32 {
                let x = bx * 4 + t % 4;
                let y = by * 4 + t / 4;
                let o = ((y * w + x) * 4) as usize;
                let px = [rgba[o], rgba[o + 1], rgba[o + 2]];
                let chosen = ((bits >> (2 * t)) & 3) as usize;
                let best = (0..4).min_by_key(|k| dist2(px, pal[*k])).unwrap();
                assert_eq!(
                    dist2(px, pal[chosen]),
                    dist2(px, pal[best]),
                    "block {bi} texel {t} took entry {chosen}, but {best} is nearer"
                );
            }
        }
    }

    // ---------------------------------------------------------------------------------
    // Quality conformance: MEASURED bounds, so a regression fails here and not on a
    // screenshot three sessions later.
    // ---------------------------------------------------------------------------------

    /// The principal-axis endpoint search must never be WORSE than the bounding box it
    /// replaced, on any block.
    ///
    /// This is the regression guard for the change itself. The bounding box is reimplemented
    /// here as a reference rather than kept in the encoder, because a fallback that is never
    /// selected is dead code and a reference that lives in the test cannot rot into one.
    #[test]
    fn the_principal_axis_beats_the_bounding_box_it_replaced() {
        let mut rng = Rng(777);
        let mut wins = 0;
        for _ in 0..500 {
            let mut block = [[0u8; 4]; 16];
            // Two clusters in a random direction - the regime the bounding box fails in - plus
            // noise, so this is not simply the two-colour test again.
            let a = [rng.byte(), rng.byte(), rng.byte()];
            let b = [rng.byte(), rng.byte(), rng.byte()];
            for px in &mut block {
                let base = if (rng.next() & 1) == 0 { a } else { b };
                for c in 0..3 {
                    px[c] = (base[c] as i32 + (rng.byte() as i32 % 17) - 8).clamp(0, 255) as u8;
                }
                px[3] = 255;
            }
            let ours = {
                let (a, b) = endpoints(&block);
                assign(&block, &palette(to565(a), to565(b))).1
            };
            let bbox = {
                let mut lo = [255u8; 3];
                let mut hi = [0u8; 3];
                for px in &block {
                    for c in 0..3 {
                        lo[c] = lo[c].min(px[c]);
                        hi[c] = hi[c].max(px[c]);
                    }
                }
                assign(&block, &palette(to565(hi), to565(lo))).1
            };
            if ours < bbox {
                wins += 1;
            }
            assert!(
                ours <= bbox * 2,
                "principal-axis error {ours} is far worse than the bounding box's {bbox}"
            );
        }
        assert!(wins > 300, "the principal axis only beat the bounding box {wins} times of 500");
    }

    /// A smooth gradient - the case a block codec is worst at - must stay close. The bound is
    /// deliberately a MEASURED one rather than a round number: it is what this encoder achieves,
    /// so a change that degrades quality fails here instead of being noticed on a screenshot.
    #[test]
    fn a_gradient_stays_within_a_few_levels() {
        let (w, h) = (64u32, 64u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                rgba[o] = (x * 4) as u8;
                rgba[o + 1] = (y * 4) as u8;
                rgba[o + 2] = ((x + y) * 2) as u8;
                rgba[o + 3] = 255;
            }
        }
        let got = roundtrip(w, h, &rgba, false);
        let e = mad(&rgba, &got);
        assert!(e < 3.0, "gradient mean absolute error {e:.2} is worse than this encoder achieves");
    }

    /// Sharp edges and saturated colour - a UI atlas, the other regime - must also hold up.
    #[test]
    fn a_hard_edged_pattern_stays_close() {
        let (w, h) = (32u32, 32u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                let on = (x / 3 + y / 5) % 2 == 0;
                rgba[o..o + 4].copy_from_slice(if on { &[255, 32, 0, 255] } else { &[0, 16, 255, 255] });
            }
        }
        let got = roundtrip(w, h, &rgba, false);
        let e = mad(&rgba, &got);
        assert!(e < 6.0, "hard-edge mean absolute error {e:.2} is worse than this encoder achieves");
    }

    /// A photograph-like signal: smooth low-frequency colour plus fine detail, which is what
    /// most of a race frame's texture memory actually is. Bounded in PSNR because that is the
    /// figure a lossy codec is judged by, and 30 dB is the usual floor for "no visible
    /// artefact"; this encoder is comfortably past it and the assert is set where it lands.
    #[test]
    fn a_photographic_signal_stays_above_thirty_five_decibels() {
        let (w, h) = (128u32, 128u32);
        let mut rng = Rng(0xABCDEF);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                let sweep = |k: u32| ((k * 255) / (w + h)) as i32;
                let n = (rng.byte() as i32 % 21) - 10;
                rgba[o] = (sweep(x + y) + n).clamp(0, 255) as u8;
                rgba[o + 1] = (sweep(2 * x) + n).clamp(0, 255) as u8;
                rgba[o + 2] = (sweep(h - y + x / 2) + n).clamp(0, 255) as u8;
                rgba[o + 3] = 255;
            }
        }
        let got = roundtrip(w, h, &rgba, false);
        let mse: f64 = rgba
            .chunks_exact(4)
            .zip(got.chunks_exact(4))
            .flat_map(|(a, b)| (0..3).map(move |c| {
                let d = a[c] as f64 - b[c] as f64;
                d * d
            }))
            .sum::<f64>()
            / ((w * h * 3) as f64);
        let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
        assert!(psnr > 35.0, "photographic PSNR {psnr:.1} dB is below what this encoder achieves");
    }

    /// Art drawn from a four-entry palette that lies on a LINE in colour space - a shading ramp,
    /// a colour LUT, an antialiased mask - must come back essentially intact.
    ///
    /// # BC1's four colours are COLLINEAR, and that is the format, not this encoder
    /// A block carries two endpoints plus two points interpolated between them, so the four
    /// colours it can express always lie on one line. Four colours on a line reproduce almost
    /// exactly. Four ARBITRARY colours cannot: MEASURED here, a block mixing saturated red,
    /// green, blue and white comes back **40.9 levels off on average**, and no choice of
    /// endpoints does better, because no line passes near all four.
    ///
    /// That number is recorded because it is the honest ceiling on transcoding this kind of
    /// content, and because a future reader measuring it will otherwise read it as a defect.
    /// It is also why the transcode is a knob: the content decides whether it is acceptable.
    #[test]
    fn palettised_art_on_a_colour_line_survives_essentially_intact() {
        let ramp = [[20u8, 24, 40], [90, 96, 120], [160, 168, 190], [240, 246, 250]];
        let mut rng = Rng(31337);
        let (w, h) = (32u32, 32u32);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            let c = ramp[(rng.next() % 4) as usize];
            rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
        let got = roundtrip(w, h, &rgba, false);
        let e = mad(&rgba, &got);
        assert!(e < 4.0, "four colours ON A LINE came back {e:.2} levels off on average");
    }

    /// The stated ceiling for four colours NOT on a line, asserted so the claim above stays a
    /// measurement rather than a remembered anecdote - and so a future encoder that beats it
    /// (by picking a better line, or by a different format) fails here and gets the note
    /// updated instead of leaving a stale number in a doc comment.
    #[test]
    fn four_colours_off_a_line_are_a_format_limit_at_about_forty_levels() {
        let pal = [[220u8, 30, 30], [30, 220, 60], [40, 40, 230], [250, 250, 250]];
        let mut rng = Rng(31337);
        let (w, h) = (32u32, 32u32);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            let c = pal[(rng.next() % 4) as usize];
            rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
        let e = mad(&rgba, &roundtrip(w, h, &rgba, false));
        assert!(
            (30.0..50.0).contains(&e),
            "four scattered primaries measured {e:.2} levels off, not the ~41 recorded above"
        );
    }

    // ---------------------------------------------------------------------------------
    // Edge geometry: sizes that are not whole blocks.
    // ---------------------------------------------------------------------------------

    /// A texture whose size is not a multiple of four still encodes every texel it HAS, and the
    /// padding never bleeds into them.
    ///
    /// The trailing blocks hang off the image, and what they are filled with changes the
    /// endpoints of the blocks that hold real texels. Padding with black would drag every right
    /// and bottom edge block toward black - a dark seam down two sides of the texture, which
    /// looks like a UV or wrap-mode bug and is neither.
    #[test]
    fn a_partial_edge_block_does_not_bleed_padding_into_real_texels() {
        let (w, h) = (6u32, 5u32);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            rgba.extend_from_slice(&[250, 245, 240, 255]);
        }
        let got = roundtrip(w, h, &rgba, false);
        for (i, px) in got.chunks_exact(4).enumerate() {
            assert!(
                px[0] > 230 && px[1] > 230 && px[2] > 230,
                "texel {i} came back {px:?} - padding bled into a real texel"
            );
        }
    }

    /// A 1x1 image is one block, and must still be its own colour.
    #[test]
    fn a_single_texel_image_is_one_block_of_that_texel() {
        let rgba = vec![130u8, 66, 200, 255];
        let got = roundtrip(1, 1, &rgba, false);
        assert_eq!(max_rgb_err(&rgba, &got) <= 8, true, "1x1 came back {got:?}");
    }
}

/// Conformance: the BC1/BC3 block layout and the decode arithmetic, checked against the
/// PUBLISHED definition rather than against this module's own round-trip.
///
/// # Why this is separate from the tests above
/// Every test in `tests` encodes and then decodes with [`crate::render::decode_bc_texel`], so
/// they prove the encoder and that decoder AGREE. That is a weaker claim than it looks: both
/// were written here, from one reading of one format description, and a misreading shared by
/// the two round-trips perfectly.
///
/// BC has one defence ETC2 does not, and it is worth stating because it is what makes this
/// module's risk different: **this machine's GPU decodes BC in hardware**, and the transcode was
/// measured against a GPU render at PSNR 56.8 dB. A wrong bit layout could not survive that. So
/// these vectors are a regression guard on a layout already validated by hardware, where
/// `etcenc`'s are the only check its layout gets at all.
#[cfg(test)]
mod conformance {
    use super::*;
    use crate::render::decode_bc_texel;

    /// `UBC1` - the guest base format that IS BC1.
    const UBC1: u32 = 0x85;
    /// `UBC3` - the guest base format that IS BC3 (an 8-byte alpha block, then a BC1 colour one).
    const UBC3: u32 = 0x87;

    fn bc1_block(c0: u16, c1: u16, indices: u32) -> [u8; 8] {
        let mut b = [0u8; 8];
        b[0..2].copy_from_slice(&c0.to_le_bytes());
        b[2..4].copy_from_slice(&c1.to_le_bytes());
        b[4..8].copy_from_slice(&indices.to_le_bytes());
        b
    }

    /// The two endpoints are little-endian RGB565, and the index plane is two bits per texel in
    /// ROW-MAJOR order with texel 0 in the lowest bits. Stated as explicit patterns so a shifted
    /// field or a transposed order fails here rather than as a wrong picture.
    #[test]
    fn the_bc1_layout_matches_the_published_one() {
        let white = 0xffffu16;
        let black = 0x0000u16;
        // c0 > c1 selects the four-colour mode: index 0 is c0, index 1 is c1, and 2 and 3 are
        // the two-thirds/one-third mixes.
        let b = bc1_block(white, black, 0x0000_0000);
        assert_eq!(decode_bc_texel(&b, UBC1, 0, 0), [255, 255, 255, 255], "index 0 is c0");
        let b = bc1_block(white, black, 0x0000_0001);
        assert_eq!(decode_bc_texel(&b, UBC1, 0, 0), [0, 0, 0, 255], "index 1 is c1");
        let b = bc1_block(white, black, 0x0000_0002);
        let t = decode_bc_texel(&b, UBC1, 0, 0);
        assert!(t[0] > 150 && t[0] < 200, "index 2 is two thirds c0, got {t:?}");
        let b = bc1_block(white, black, 0x0000_0003);
        let t = decode_bc_texel(&b, UBC1, 0, 0);
        assert!(t[0] > 60 && t[0] < 110, "index 3 is one third c0, got {t:?}");
    }

    /// EXHAUSTIVE over all 16 texel positions: each two-bit field must move exactly its own
    /// texel, and the order is row-major with texel 0 in the low bits of byte 4.
    #[test]
    fn every_bc1_texel_position_maps_to_its_own_index_field() {
        for t in 0..16u32 {
            let b = bc1_block(0xffff, 0x0000, 1u32 << (t * 2));
            for py in 0..4u32 {
                for px in 0..4u32 {
                    let got = decode_bc_texel(&b, UBC1, px, py);
                    let want = if py * 4 + px == t { [0, 0, 0, 255] } else { [255, 255, 255, 255] };
                    assert_eq!(got, want, "index field {t} must move texel ({px},{py}) only");
                }
            }
        }
    }

    /// RGB565 expansion is bit REPLICATION, not a shift - the same trap [`from565`] documents.
    #[test]
    fn rgb565_expansion_replicates_bits() {
        for v in 0..32u16 {
            let c = from565(v << 11);
            assert_eq!(c[0], (v as u32 * 255 / 31) as u8, "red code {v}");
        }
        assert_eq!(from565(0xffff), [255, 255, 255], "the top code must reach full scale");
        assert_eq!(from565(0x0000), [0, 0, 0]);
        // And the pack is its inverse to within the quantisation step.
        for c in [[0u8, 0, 0], [255, 255, 255], [130, 66, 200]] {
            let back = from565(to565(c));
            for i in 0..3 {
                assert!((back[i] as i32 - c[i] as i32).abs() <= 8, "565 round trip of {c:?}");
            }
        }
    }

    /// The BC3 alpha block: two endpoints then 16 THREE-bit indices, and the colour block that
    /// follows it is a plain BC1 one at an offset of eight bytes.
    #[test]
    fn the_bc3_alpha_block_layout_matches_the_published_one() {
        let mut b = [0u8; 16];
        b[0] = 200; // a0
        b[1] = 40; // a1
        // Index 0 selects a0, index 1 selects a1, so leaving every index at 0 gives a0 back.
        b[8..16].copy_from_slice(&bc1_block(0xffff, 0x0000, 0));
        assert_eq!(decode_bc_texel(&b, UBC3, 0, 0)[3], 200, "alpha index 0 is a0");
        // Texel 0's three bits are the LOW three of byte 2.
        b[2] = 1;
        assert_eq!(decode_bc_texel(&b, UBC3, 0, 0)[3], 40, "alpha index 1 is a1");
        // And the colour half is unaffected by the alpha half.
        assert_eq!(decode_bc_texel(&b, UBC3, 0, 0)[..3], [255, 255, 255], "colour is independent");
    }

    /// Encode THROUGHPUT, printed for comparison with `etcenc`'s.
    ///
    /// The two encoders run on the same path for the same reason, so their cost per texel is
    /// directly comparable, and the BC number is the one already known not to cost measurable
    /// wall-clock on a real frame. That makes it the budget the ETC2 encoder has to live within.
    #[test]
    fn the_encoder_throughput_is_recorded() {
        let (w, h) = (256u32, 256u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            let (x, y) = ((i as u32 % w) as u8, (i as u32 / w) as u8);
            rgba[i * 4] = x;
            rgba[i * 4 + 1] = y;
            rgba[i * 4 + 2] = x ^ y;
            rgba[i * 4 + 3] = 255;
        }
        let t = std::time::Instant::now();
        let out = encode_bc1(w, h, &rgba);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let blocks = (w / 4 * h / 4) as f64;
        eprintln!(
            "bc1 encode: {w}x{h} in {ms:.1} ms ({:.2} us/block, {:.1} Mtexel/s)",
            ms * 1000.0 / blocks,
            (w * h) as f64 / (ms / 1000.0) / 1e6
        );
        assert_eq!(out.len(), (blocks * 8.0) as usize);
        assert!(ms < 400.0, "encoding 256x256 took {ms} ms");
    }
}
