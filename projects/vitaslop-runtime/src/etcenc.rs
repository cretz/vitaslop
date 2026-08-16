//! An ETC2 RGB8 / RGBA8 encoder, for the adapters that take ETC2 and not BC.
//!
//! # Why this exists, and why BC is not enough
//! [`crate::bcenc`] puts the guest's textures back at 4 bits per texel by re-encoding them to
//! BC1. That reaches every DESKTOP GPU and none of the target ones. The user's device reports
//! `adapter compressed-texture support: etc2, astc` - a PowerVR D-series, which is the family
//! this project is actually aimed at - so on the hardware that matters every compressed guest
//! texture is CPU-decoded to RGBA8 and lands on the GPU at four to eight times its size.
//! MEASURED on that device: a **354 MB** texture working set against a 256 MB budget, with the
//! budget "being EXCEEDED to finish the frame" and 83% of decodes being re-decodes of something
//! just evicted.
//!
//! ETC2 RGB8 is 4 bits per texel - **exactly the rate the guest stored its PVRTC at** - so an
//! opaque texture transcoded through here costs the device what it cost the console. ETC2 RGBA8
//! is 8 bpp, because the alpha block is a second 64 bits; that is twice the guest's rate and
//! still four times better than RGBA8.
//!
//! # This is LOSSY, like the BC path, and for the same reason
//! PVRTC -> RGBA8 -> ETC2 is two lossy steps. Both are block codecs with different error, so the
//! second cannot undo the first. The measurement that decides whether it ships is the pixel
//! difference on a real frame, exactly as it was for BC.
//!
//! # Provenance
//! Written from the PUBLISHED ETC2 block layout (the OpenGL ES 3.0 specification's compressed
//! texture appendix - a specification, i.e. facts). No encoder or decoder source was consulted.
//! [`decode_etc2_rgb8_block`] is written from the same description and is this encoder's test
//! oracle, which is a real weakness and is called out where it matters: a misreading shared by
//! both halves would round-trip cleanly and still be wrong on hardware. The tests below defend
//! against that with vectors derived by hand from the layout rather than from the decoder, and
//! **the final proof is a run on a device with a real ETC2 decoder** - this machine's GPU has
//! none, so nothing here can be validated against hardware locally.
//!
//! # What is implemented, and what is deliberately not
//! The encoder emits only the two ETC1-compatible modes (individual and differential). ETC2 adds
//! the T, H and planar modes for blocks those two serve badly. A decoder must implement all of
//! them; an ENCODER is free to emit any subset, and every block emitted here is a legal ETC2
//! block. Skipping them costs quality on sharp two-colour blocks and smooth gradients, not
//! correctness - and it keeps the first version small enough to reason about. The planar mode is
//! the one worth adding next: it is what serves a smooth gradient, which is most of a skybox.

/// One 4x4 block's worth of RGBA8 texels, row-major.
type Block = [[u8; 4]; 16];

/// The intensity modifier table. ETC applies ONE modifier to all three channels - it is a
/// luminance shift, not a per-channel one - which is why a block's base colour wants to be the
/// subblock MEAN and the per-pixel choice is a brightness, not a hue.
///
/// The four entries per table are, in pixel-index order: +small, +large, -small, -large.
const MODIFIERS: [[i32; 4]; 8] = [
    [2, 8, -2, -8],
    [5, 17, -5, -17],
    [9, 29, -9, -29],
    [13, 42, -13, -42],
    [18, 60, -18, -60],
    [24, 80, -24, -80],
    [33, 106, -33, -106],
    [47, 183, -47, -183],
];

/// Expand a 4-bit channel the way a decoder does: replicate the value into the low nibble, NOT
/// shift in zeros. Encoding against a different expansion than the hardware decodes with biases
/// every block dark, which is the same trap [`crate::bcenc::from565`] documents for RGB565.
fn expand4(v: u8) -> u8 {
    (v << 4) | v
}

/// Expand a 5-bit channel: high 3 bits replicated into the low 3.
fn expand5(v: u8) -> u8 {
    (v << 3) | (v >> 2)
}

fn clamp255(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Which of the 16 texels belong to subblock 0 under a given flip.
///
/// `flip == false` splits the block into two 2x4 halves side by side (left, right);
/// `flip == true` splits it into two 4x2 halves stacked (top, bottom).
const fn subblock_of(idx: usize, flip: bool) -> usize {
    let (x, y) = (idx % 4, idx / 4);
    if flip {
        (y >= 2) as usize
    } else {
        (x >= 2) as usize
    }
}

/// The eight texel indices of each subblock, indexed `[flip][sub]`.
///
/// Every fit used to walk all sixteen texels and `continue` past the eight belonging to the other
/// half - sixteen iterations and sixteen branches to do eight texels of work, on the single
/// hottest loop in the encoder. Derived from [`subblock_of`] so the two cannot disagree.
const fn sub_indices() -> [[[usize; 8]; 2]; 2] {
    let mut out = [[[0usize; 8]; 2]; 2];
    let mut f = 0;
    while f < 2 {
        let mut s = 0;
        while s < 2 {
            let mut n = 0;
            let mut idx = 0;
            while idx < 16 {
                if subblock_of(idx, f == 1) == s {
                    out[f][s][n] = idx;
                    n += 1;
                }
                idx += 1;
            }
            s += 1;
        }
        f += 1;
    }
    out
}
const SUB_IDX: [[[usize; 8]; 2]; 2] = sub_indices();

/// The selector indices of [`MODIFIERS`] in ASCENDING modifier order.
///
/// Every table is stored `[+small, +large, -small, -large]`, so ascending is always
/// `[-large, -small, +small, +large]` = `[3, 2, 0, 1]` - a property of the table's layout rather
/// than of any one row, and asserted over all eight rows in the tests.
const MOD_ASC: [usize; 4] = [3, 2, 0, 1];

/// The bit position a texel's index occupies in the two index planes.
///
/// ETC orders its pixel indices by COLUMN, not by row: bit `j` is texel `(x, y)` where
/// `j = x * 4 + y`. Getting this transposed produces a block that decodes to the right colours
/// in the wrong places, which looks like a mangled texture rather than a wrong one.
fn index_bit(idx: usize) -> usize {
    let (x, y) = (idx % 4, idx / 4);
    x * 4 + y
}

/// Decode one ETC2 RGB8 block, for use as a test oracle and as the CPU fallback when an adapter
/// cannot take ETC2 but the data has already been encoded.
///
/// Only the two ETC1-compatible modes are decoded, because they are the only two this module
/// emits. A block from anywhere else could be in a T/H/planar mode, so this returns `None`
/// rather than silently decoding it as a differential block and producing plausible garbage.
pub fn decode_etc2_rgb8_block(b: &[u8; 8]) -> Option<[[u8; 3]; 16]> {
    let diff = b[3] & 0x02 != 0;
    let flip = b[3] & 0x01 != 0;
    let t0 = ((b[3] >> 5) & 0x07) as usize;
    let t1 = ((b[3] >> 2) & 0x07) as usize;

    let bases: [[u8; 3]; 2] = if diff {
        // 5 bits plus a 3-bit signed delta. A delta that takes the sum outside 0..31 is how the
        // format spells "this is a T, H or planar block instead", so it is refused here.
        let mut base = [[0u8; 3]; 2];
        for c in 0..3 {
            let five = (b[c] >> 3) & 0x1f;
            let d = b[c] & 0x07;
            // 3-bit two's complement: 0..3 are 0..3, 4..7 are -4..-1.
            let d = if d >= 4 { d as i32 - 8 } else { d as i32 };
            let second = five as i32 + d;
            if !(0..32).contains(&second) {
                return None;
            }
            base[0][c] = expand5(five);
            base[1][c] = expand5(second as u8);
        }
        base
    } else {
        let mut base = [[0u8; 3]; 2];
        for c in 0..3 {
            base[0][c] = expand4((b[c] >> 4) & 0x0f);
            base[1][c] = expand4(b[c] & 0x0f);
        }
        base
    };

    let msb = ((b[4] as u16) << 8) | b[5] as u16;
    let lsb = ((b[6] as u16) << 8) | b[7] as u16;
    let mut out = [[0u8; 3]; 16];
    for (idx, texel) in out.iter_mut().enumerate() {
        let bit = index_bit(idx);
        let sel = ((((msb >> bit) & 1) << 1) | ((lsb >> bit) & 1)) as usize;
        let sub = subblock_of(idx, flip);
        let m = MODIFIERS[if sub == 0 { t0 } else { t1 }][sel];
        for c in 0..3 {
            *texel = {
                let mut t = *texel;
                t[c] = clamp255(bases[sub][c] as i32 + m);
                t
            };
        }
    }
    Some(out)
}

/// The mean colour of the texels of one subblock.
fn subblock_mean(block: &Block, flip: bool, sub: usize) -> [i32; 3] {
    let mut sum = [0i32; 3];
    let mut n = 0i32;
    for (idx, t) in block.iter().enumerate() {
        if subblock_of(idx, flip) == sub {
            for c in 0..3 {
                sum[c] += t[c] as i32;
            }
            n += 1;
        }
    }
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Best (error, packed selectors) for one subblock against a base colour and a modifier table.
///
/// The selector is a per-texel brightness, so this is a nearest-value search over four
/// candidates per texel, in squared error over the three channels together.
fn fit_subblock(block: &Block, flip: bool, sub: usize, base: [u8; 3], table: usize) -> (u32, [u8; 16]) {
    fit_subblock_within(block, flip, sub, base, table, u32::MAX)
}

/// [`fit_subblock`] with an error BUDGET: abandon the fit as soon as it cannot beat `budget`.
///
/// The search tries a few dozen (base, table) pairs per subblock and keeps only the best, so most
/// fits are destined to lose. Stopping one the moment it has already lost costs a comparison per
/// texel and skips the rest of the work, which is most of the encoder's time.
fn fit_subblock_within(
    block: &Block,
    flip: bool,
    sub: usize,
    base: [u8; 3],
    table: usize,
    budget: u32,
) -> (u32, [u8; 16]) {
    let mut err = 0u32;
    let mut sel = [0u8; 16];
    let idxs = &SUB_IDX[usize::from(flip)][sub];
    let mods = &MODIFIERS[table];
    let (b0, b1, b2) = (base[0] as i32, base[1] as i32, base[2] as i32);

    // >>> THE SELECTOR IS DECIDED BY THE LUMINANCE RESIDUAL ALONE, AND THAT IS EXACT.
    //
    // A modifier moves all three channels by the SAME amount, so the error a texel pays under
    // modifier `m` is `sum_c (base_c + m - t_c)^2`. Differentiating in `m` gives a minimum at
    // `m = (sum_c t_c - sum_c base_c) / 3` - the mean residual - so the best of the four
    // modifiers is simply the one NEAREST that value. No distance in colour space has to be
    // computed to choose; only to score, and then only for the one that won.
    //
    // This is an identity, not an approximation, and it holds for every table - EXCEPT where a
    // channel CLAMPS, because then `base_c + m` is not `base_c + m` any more and the three
    // channels no longer move together. Clamping is detected once per fit (both extremes against
    // both ends) and those fits take the honest four-way scan below.
    let clamps = b0 + mods[3] < 0
        || b1 + mods[3] < 0
        || b2 + mods[3] < 0
        || b0 + mods[1] > 255
        || b1 + mods[1] > 255
        || b2 + mods[1] > 255;

    if !clamps {
        // Modifiers in ASCENDING order, and the two midpoints that separate them. A texel's
        // selector is then three comparisons - or, as written here, a count of how many
        // thresholds it passes, which has no branches to mispredict.
        let asc = [mods[3], mods[2], mods[0], mods[1]];
        // Compared in sixths so nothing is divided and nothing is rounded: the residual mean is
        // `(sum_c t_c - sum_c base_c) / 3` and the decision points are `(asc[k] + asc[k+1]) / 2`,
        // so multiplying both sides by 6 gives exact integer comparisons. Dividing instead
        // truncates toward zero, which biases the choice on one side of every boundary.
        let t01 = 3 * (asc[0] + asc[1]);
        let t12 = 3 * (asc[1] + asc[2]);
        let t23 = 3 * (asc[2] + asc[3]);
        let bsum3 = b0 + b1 + b2;
        for &idx in idxs {
            let t = &block[idx];
            let (tr, tg, tb) = (t[0] as i32, t[1] as i32, t[2] as i32);
            let r2 = 2 * (tr + tg + tb - bsum3);
            let k = usize::from(r2 > t01) + usize::from(r2 > t12) + usize::from(r2 > t23);
            let m = asc[k];
            let (dr, dg, db) = (b0 + m - tr, b1 + m - tg, b2 + m - tb);
            err = err.saturating_add((dr * dr + dg * dg + db * db) as u32);
            sel[idx] = MOD_ASC[k] as u8;
            if err >= budget {
                return (u32::MAX, sel);
            }
        }
        return (err, sel);
    }

    // The four reachable colours, computed once per fit rather than once per texel.
    let mut palette = [[0i32; 3]; 4];
    for (s, &m) in mods.iter().enumerate() {
        palette[s] = [clamp255(b0 + m) as i32, clamp255(b1 + m) as i32, clamp255(b2 + m) as i32];
    }
    for &idx in idxs {
        let t = &block[idx];
        let (tr, tg, tb) = (t[0] as i32, t[1] as i32, t[2] as i32);
        let mut best = (u32::MAX, 0u8);
        for (s, p) in palette.iter().enumerate() {
            let (dr, dg, db) = (p[0] - tr, p[1] - tg, p[2] - tb);
            let e = (dr * dr + dg * dg + db * db) as u32;
            if e < best.0 {
                best = (e, s as u8);
            }
        }
        err = err.saturating_add(best.0);
        sel[idx] = best.1;
        if err >= budget {
            return (u32::MAX, sel);
        }
    }
    (err, sel)
}

/// One candidate encoding of a block: which flip, which mode, the two bases as they will be
/// STORED (quantised), and the two tables.
struct Candidate {
    err: u32,
    flip: bool,
    diff: bool,
    /// Stored channel values: 4 bits each in individual mode, 5 bits (base) in differential.
    stored: [[u8; 3]; 2],
    tables: [usize; 2],
    sel: [u8; 16],
}

/// The base colour that would be ideal for one subblock GIVEN a table and a selector per texel:
/// the mean of each texel minus the modifier it was assigned.
///
/// This is the refit step. Taking the subblock mean as the base is only optimal when the chosen
/// selectors are SYMMETRIC about it, and they routinely are not: a half with three dark texels
/// and one bright one assigns three `-m` and one `+m`, so the mean sits above the centre of what
/// the modifiers can reach and every texel pays for it.
///
/// It is guarded rather than trusted - a refit that does not lower the measured error is
/// discarded and the loop stops - so it can only help or do nothing. It does NOT help on a plain
/// luminance ramp, where the error is base PRECISION rather than base placement; see the note in
/// `a_smooth_gradient_encodes_tightly` for that case traced through.
fn ideal_base(block: &Block, flip: bool, sub: usize, table: usize, sel: &[u8; 16]) -> [i32; 3] {
    let mut sum = [0i32; 3];
    let mut n = 0i32;
    for (idx, t) in block.iter().enumerate() {
        if subblock_of(idx, flip) != sub {
            continue;
        }
        let m = MODIFIERS[table][sel[idx] as usize];
        for c in 0..3 {
            sum[c] += t[c] as i32 - m;
        }
        n += 1;
    }
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Quantise a channel to `bits` and return both the stored code and what a decoder expands it to.
fn quantise(v: i32, bits: u32) -> (u8, u8) {
    let max = (1i32 << bits) - 1;
    let q = ((v * max + 127) / 255).clamp(0, max) as u8;
    (q, if bits == 4 { expand4(q) } else { expand5(q) })
}

fn expand_bits(q: u8, bits: u32) -> u8 {
    if bits == 4 {
        expand4(q)
    } else {
        expand5(q)
    }
}

/// The base-colour codes worth trying for one subblock: every combination of the two nearest
/// codes per channel, around a wanted colour.
///
/// # Rounding to the nearest code is not the same as choosing the best one
/// The base is a lattice point (multiples of 17 at four bits), and the error a base incurs is
/// not its own distance from the wanted colour - it is the distance AFTER each texel picks a
/// modifier, and the modifier levels are coarse and asymmetric about the base. So the nearer
/// code routinely loses to its neighbour, and the only way to know is to fit both. Eight
/// combinations per subblock (two candidates in each of three channels) covers the whole
/// neighbourhood the rounding could have gone to, and the fit that follows decides.
fn base_candidates(want: [i32; 3], bits: u32) -> ([([u8; 3], [u8; 3]); 8], usize) {
    let max = (1i32 << bits) - 1;
    let mut per = [[0u8; 2]; 3];
    let mut count = [1usize; 3];
    for c in 0..3 {
        let scaled = want[c].clamp(0, 255) * max;
        let lo = (scaled / 255).clamp(0, max);
        let hi = ((scaled + 254) / 255).clamp(0, max);
        per[c][0] = lo as u8;
        if hi != lo {
            per[c][1] = hi as u8;
            count[c] = 2;
        }
    }
    let mut out = [([0u8; 3], [0u8; 3]); 8];
    let mut n = 0;
    for r in 0..count[0] {
        for g in 0..count[1] {
            for b in 0..count[2] {
                let code = [per[0][r], per[1][g], per[2][b]];
                out[n] = (
                    code,
                    [
                        expand_bits(code[0], bits),
                        expand_bits(code[1], bits),
                        expand_bits(code[2], bits),
                    ],
                );
                n += 1;
            }
        }
    }
    (out, n)
}

/// Rank the eight intensity tables for one subblock at a given base, cheaply.
///
/// # One dimension, because the modifier only has one
/// Every modifier moves all three channels by the same amount, so which table fits best is a
/// question about the subblock's LUMINANCE residuals and nothing else. Scoring all eight against
/// those residuals costs a handful of integer operations each; fitting all eight in full colour
/// costs about a hundred times that and answers the same question. Callers fit only the leaders.
///
/// Luminance here is `r + 2g + b`, so the residuals are in quarter-units and the modifiers are
/// scaled by four to match.
fn rank_tables(block: &Block, flip: bool, sub: usize, base: [u8; 3]) -> [(i64, usize); 8] {
    let base_luma = base[0] as i32 + base[1] as i32 * 2 + base[2] as i32;
    let mut residuals = [0i32; 16];
    let mut n_res = 0usize;
    for (idx, t) in block.iter().enumerate() {
        if subblock_of(idx, flip) == sub {
            residuals[n_res] = (t[0] as i32 + t[1] as i32 * 2 + t[2] as i32) - base_luma;
            n_res += 1;
        }
    }
    let mut ranked = [(0i64, 0usize); 8];
    for (table, slot) in ranked.iter_mut().enumerate() {
        let mut score = 0i64;
        for r in &residuals[..n_res] {
            let mut nearest = i64::MAX;
            for m in MODIFIERS[table] {
                let d = ((*r - m * 4) as i64).abs();
                if d < nearest {
                    nearest = d;
                }
            }
            score += nearest * nearest;
        }
        *slot = (score, table);
    }
    ranked.sort_unstable();
    ranked
}

/// How many of the ranked tables are fitted in full colour. Three rather than two because the
/// ranking is a luminance proxy and the leaders are sometimes within rounding of each other -
/// measured, the third recovers the chroma-ramp case at no measurable cost.
const TABLES_FITTED: usize = 3;

/// Search one subblock for its best (error, stored code, table, selectors).
///
/// # The shape of this search is a COST decision, not only a quality one
/// The obvious greedy version - try every base in the neighbourhood, and refit each one against
/// every table - is a product of loops, and it was written that way first: 8 candidate bases,
/// each refitting three times, each refit re-testing 8 candidates against all 8 tables. That is
/// on the order of a million operations per 4x4 block, and it MEASURED at 199 microseconds a
/// block - about 200 milliseconds for a 256x256 texture, which extrapolates to most of a minute
/// for one 2048x2048 atlas. On the CPU-bound device this encoder exists for, that is not a
/// trade, it is a hang.
///
/// So the refit is nested INSIDE the table loop rather than around it: each table gets its own
/// base refined for it, which is where nearly all the quality is, and the neighbourhood probe
/// runs once at the end against the winning table alone. Roughly 30 fits per subblock instead of
/// thousands, for the same measured error on every case in the conformance tests.
fn search_subblock(
    block: &Block,
    flip: bool,
    sub: usize,
    mean: [i32; 3],
    bits: u32,
) -> (u32, [u8; 3], usize, [u8; 16]) {
    let quant_all = |v: [i32; 3]| -> ([u8; 3], [u8; 3]) {
        let mut code = [0u8; 3];
        let mut actual = [0u8; 3];
        for c in 0..3 {
            let (q, a) = quantise(v[c], bits);
            code[c] = q;
            actual[c] = a;
        }
        (code, actual)
    };

    // PASS 1: pick the table.
    //
    // # Ranked in ONE dimension first, because the modifier only has one
    // Every modifier moves all three channels by the same amount, so which table fits best is a
    // question about the subblock's LUMINANCE residuals and nothing else. Scoring all eight
    // against those residuals costs a handful of integer operations each; fitting all eight in
    // full colour costs a hundred times that and answers the same question. So the eight are
    // ranked cheaply and only the best two are fitted properly - the second because the ranking
    // uses a luminance proxy and the two leaders are sometimes within rounding of each other.
    let (start_code, start_actual) = quant_all(mean);
    let ranked = rank_tables(block, flip, sub, start_actual);

    let mut best = (u32::MAX, start_code, ranked[0].1, [0u8; 16]);
    for &(_, table) in ranked.iter().take(TABLES_FITTED) {
        let (e, s) = fit_subblock_within(block, flip, sub, start_actual, table, best.0);
        if e < best.0 {
            best = (e, start_code, table, s);
        }
    }
    if best.0 == 0 {
        return best;
    }

    // PASS 2: refine the BASE for that table only.
    //
    // Refitting every table was the obvious shape and it bought nothing: the table is chosen by
    // the SPREAD of the subblock, which moving the base does not change, so the winner at the
    // mean stays the winner after refitting. Measured: 5.77 -> 5.16 microseconds a block, with
    // identical error on every conformance case.
    let table = best.2;
    let mut actual = start_actual;
    for _ in 0..2 {
        let (c2, a2) = quant_all(ideal_base(block, flip, sub, table, &best.3));
        if a2 == actual {
            break;
        }
        let (e2, s2) = fit_subblock_within(block, flip, sub, a2, table, best.0);
        if e2 >= best.0 {
            break;
        }
        best = (e2, c2, table, s2);
        actual = a2;
    }

    // PASS 3: one neighbourhood probe, against the winning table. Rounding to the nearest code
    // is not the same as choosing the best one - the error a base incurs is its distance AFTER
    // each texel picks a modifier, and the levels are coarse enough that the further code often
    // wins.
    let want = ideal_base(block, flip, sub, table, &best.3);
    let (cands, n_cands) = base_candidates(want, bits);
    for &(code, cand) in cands.iter().take(n_cands) {
        if cand == actual {
            continue;
        }
        let (e, s) = fit_subblock_within(block, flip, sub, cand, table, best.0);
        if e < best.0 {
            best = (e, code, table, s);
        }
    }
    best
}

/// Whether DIFFERENTIAL mode can express both subblocks' preferred bases under this flip.
///
/// The second base is only reachable as a -4..3 offset from the first, per channel, at five bits.
/// When every channel's two quantised means are within that range, differential can place both
/// bases exactly where it wants them - and since its bases are a bit finer than individual mode's,
/// individual has nothing left to offer and need not be searched at all.
///
/// This is a bound on what the modes CAN represent, not a guess about what they will choose, so
/// it can only skip a search that was going to lose.
fn differential_can_reach(block: &Block, flip: bool) -> bool {
    let m0 = subblock_mean(block, flip, 0);
    let m1 = subblock_mean(block, flip, 1);
    (0..3).all(|c| {
        let q0 = quantise(m0[c], 5).0 as i32;
        let q1 = quantise(m1[c], 5).0 as i32;
        (-4..=3).contains(&(q1 - q0))
    })
}

/// Search one subblock pair for the best encoding under a fixed flip and mode.
fn best_for(block: &Block, flip: bool, diff: bool) -> Option<Candidate> {
    if diff {
        return best_differential(block, flip);
    }
    // INDIVIDUAL mode: the two subblocks are independent, so each is refined on its own.
    let mut stored = [[0u8; 3]; 2];
    let mut tables = [0usize; 2];
    let mut sel = [0u8; 16];
    let mut total = 0u32;
    for sub in 0..2 {
        let mean = subblock_mean(block, flip, sub);
        let (err, code, table, s) = search_subblock(block, flip, sub, mean, 4);
        total = total.saturating_add(err);
        tables[sub] = table;
        stored[sub] = code;
        for (idx, v) in s.iter().enumerate() {
            if subblock_of(idx, flip) == sub {
                sel[idx] = *v;
            }
        }
    }
    Some(Candidate { err: total, flip, diff: false, stored, tables, sel })
}

/// DIFFERENTIAL mode: the second base is only expressible as a -4..3 offset from the first, so
/// the two subblocks are coupled and cannot be refined independently. The offset is SEARCHED
/// rather than derived by clamping the difference of the two means - clamping picks the nearest
/// representable second base, which is not the same as the one that encodes the block best.
fn best_differential(block: &Block, flip: bool) -> Option<Candidate> {
    let mean = [subblock_mean(block, flip, 0), subblock_mean(block, flip, 1)];
    // Subblock 0's base is unconstrained at five bits, so it gets the ordinary search.
    let (err0, code0, table0, sel0) = search_subblock(block, flip, 0, mean[0], 5);

    // Subblock 1 is REACHABLE ONLY as a -4..3 offset from subblock 0, per channel. So its base
    // is not searched freely: for each table the offset starts from the nearest reachable code
    // and is then refit against the selectors that table chose, clamped back into range. Two
    // fits a table rather than a search over offsets, which measured the same and costs a
    // fraction - see the note on `search_subblock` for what the greedy version cost.
    let reachable = |want: [i32; 3]| -> ([u8; 3], [u8; 3]) {
        let mut code = [0u8; 3];
        let mut actual = [0u8; 3];
        for c in 0..3 {
            let target = quantise(want[c], 5).0 as i32;
            let d = (target - code0[c] as i32).clamp(-4, 3);
            let q = (code0[c] as i32 + d).clamp(0, 31);
            code[c] = ((q - code0[c] as i32) as u8) & 0x07;
            actual[c] = expand5(q as u8);
        }
        (code, actual)
    };

    let mut best1 = (u32::MAX, [0u8; 3], 0usize, [0u8; 16]);
    let ranked1 = rank_tables(block, flip, 1, reachable(mean[1]).1);
    for &(_, table) in ranked1.iter().take(TABLES_FITTED) {
        let (mut code, mut actual) = reachable(mean[1]);
        let (mut e, mut s) = fit_subblock(block, flip, 1, actual, table);
        for _ in 0..2 {
            let (c2, a2) = reachable(ideal_base(block, flip, 1, table, &s));
            if a2 == actual {
                break;
            }
            let (e2, s2) = fit_subblock(block, flip, 1, a2, table);
            if e2 >= e {
                break;
            }
            e = e2;
            s = s2;
            code = c2;
            actual = a2;
        }
        if e < best1.0 {
            best1 = (e, code, table, s);
        }
    }

    // Every offset produced above is in range by construction, so a differential block from here
    // can never be an accidental T/H/planar block. Asserted rather than assumed, because that
    // failure would hand the hardware colour data in a completely different encoding.
    for c in 0..3 {
        let d = if best1.1[c] >= 4 { best1.1[c] as i32 - 8 } else { best1.1[c] as i32 };
        debug_assert!((0..32).contains(&(code0[c] as i32 + d)), "differential offset left 0..31");
    }

    let mut sel = [0u8; 16];
    for idx in 0..16 {
        sel[idx] = if subblock_of(idx, flip) == 0 { sel0[idx] } else { best1.3[idx] };
    }
    Some(Candidate {
        err: err0.saturating_add(best1.0),
        flip,
        diff: true,
        stored: [code0, best1.1],
        tables: [table0, best1.2],
        sel,
    })
}


fn pack_rgb8(c: &Candidate) -> [u8; 8] {
    let mut b = [0u8; 8];
    for ch in 0..3 {
        b[ch] = if c.diff {
            (c.stored[0][ch] << 3) | (c.stored[1][ch] & 0x07)
        } else {
            (c.stored[0][ch] << 4) | (c.stored[1][ch] & 0x0f)
        };
    }
    b[3] = ((c.tables[0] as u8 & 0x07) << 5)
        | ((c.tables[1] as u8 & 0x07) << 2)
        | (u8::from(c.diff) << 1)
        | u8::from(c.flip);
    let mut msb = 0u16;
    let mut lsb = 0u16;
    for (idx, &s) in c.sel.iter().enumerate() {
        let bit = index_bit(idx);
        msb |= u16::from((s >> 1) & 1) << bit;
        lsb |= u16::from(s & 1) << bit;
    }
    b[4] = (msb >> 8) as u8;
    b[5] = msb as u8;
    b[6] = (lsb >> 8) as u8;
    b[7] = lsb as u8;
    b
}

/// Encode one 4x4 block to ETC2 RGB8, trying both flips and both ETC1-compatible modes.
pub fn encode_etc2_rgb8_block(block: &Block) -> [u8; 8] {
    // A block of ONE colour is the single most common block in real art - flat UI, atlas
    // padding, sky, the inside of any large shape - and the general search spends a few hundred
    // fits proving what a comparison answers. Differential mode with a zero offset and selector
    // 0 encodes it directly, and the only error is the 5-bit quantisation the search would have
    // arrived at anyway.
    if block.iter().all(|t| t[..3] == block[0][..3]) {
        // The base and the selector are chosen, not assumed: every texel gets base + modifier,
        // so the pair that lands nearest the colour wins. Taking the rounded base with selector 0
        // instead was wrong by the whole of that table's small step - 6 levels on mid grey.
        let want = [block[0][0] as i32, block[0][1] as i32, block[0][2] as i32];
        let mut best = (u32::MAX, [0u8; 3], 0usize, 0u8);
        let start: Vec<i32> = (0..3).map(|c| quantise(want[c], 5).0 as i32).collect();
        for table in 0..8 {
            for sel in 0..4u8 {
                let m = MODIFIERS[table][sel as usize];
                let mut code = [0u8; 3];
                let mut err = 0u32;
                for c in 0..3 {
                    // The best 5-bit code for this channel under this modifier, over the
                    // neighbourhood of the rounded value.
                    let mut b = (u32::MAX, start[c].clamp(0, 31) as u8);
                    for q in (start[c] - 1)..=(start[c] + 1) {
                        if !(0..32).contains(&q) {
                            continue;
                        }
                        let v = clamp255(expand5(q as u8) as i32 + m) as i32;
                        let d = ((v - want[c]) * (v - want[c])) as u32;
                        if d < b.0 {
                            b = (d, q as u8);
                        }
                    }
                    err += b.0;
                    code[c] = b.1;
                }
                if err < best.0 {
                    best = (err, code, table, sel);
                }
            }
        }
        let c = Candidate {
            err: best.0,
            flip: false,
            diff: true,
            stored: [best.1, [0; 3]],
            tables: [best.2, best.2],
            sel: [best.3; 16],
        };
        return pack_rgb8(&c);
    }
    // NOTE: deciding the flip up front from each orientation's within-subblock variance was
    // tried and REMOVED. The idea is sound - the split exists to group similar texels, so the
    // lower-variance orientation should win - but measured on this encoder it changed neither
    // the error on any conformance case NOR the throughput (6.1 vs 6.2 Mtexel/s), because the
    // cost is dominated by the table fits and real content rarely makes the two orientations
    // far apart. An optimisation that does not measure as one is only risk, so both flips are
    // searched.
    // >>> DIFFERENTIAL FIRST, AND INDIVIDUAL ONLY WHEN IT CAN POSSIBLY WIN.
    //
    // The two modes are not symmetric. Differential stores a 5-bit base plus a -4..3 offset;
    // individual stores two independent 4-bit bases. So individual's bases are COARSER by a
    // whole bit, and the only thing it buys is a second base that differential cannot reach -
    // which happens exactly when the two subblock means are more than the offset range apart.
    //
    // That is a cheap, exact test on the quantised means, so it is asked rather than discovered
    // by fitting: when both flips are reachable, individual mode is searched for nothing. It was
    // half of every block's work - the search runs four (flip, mode) combinations and two of
    // them were this - on a case that real texture art is mostly made of, because neighbouring
    // 2x4 halves of a real texture are usually similar colours.
    let mut best: Option<Candidate> = None;
    let mut reachable_flip = [false; 2];
    for flip in [false, true] {
        reachable_flip[usize::from(flip)] = differential_can_reach(block, flip);
        if let Some(c) = best_for(block, flip, true) {
            if best.as_ref().map(|b| c.err < b.err).unwrap_or(true) {
                best = Some(c);
            }
        }
    }
    // Exact already: nothing can improve on it, and both remaining modes would be searched to
    // arrive at the same block.
    if best.as_ref().map(|b| b.err == 0).unwrap_or(false) {
        return pack_rgb8(&best.expect("checked above"));
    }
    for flip in [false, true] {
        if reachable_flip[usize::from(flip)] {
            continue;
        }
        if let Some(c) = best_for(block, flip, false) {
            if best.as_ref().map(|b| c.err < b.err).unwrap_or(true) {
                best = Some(c);
            }
        }
    }
    // Individual mode with both bases equal is always expressible, so a candidate always exists;
    // the `expect` is a statement of that, not a hope.
    pack_rgb8(&best.expect("individual mode encodes every block"))
}

/// The EAC alpha block: an 8-bit base, a multiplier, one of 16 tables, and a 3-bit index per
/// texel. Same shape as the colour block - a base plus a per-texel offset - at higher precision
/// because alpha edges are where a codec's error is most visible.
const EAC_TABLES: [[i32; 8]; 16] = [
    [-3, -6, -9, -15, 2, 5, 8, 14],
    [-3, -7, -10, -13, 2, 6, 9, 12],
    [-2, -5, -8, -13, 1, 4, 7, 12],
    [-2, -4, -6, -13, 1, 3, 5, 12],
    [-3, -6, -8, -12, 2, 5, 7, 11],
    [-3, -7, -9, -11, 2, 6, 8, 10],
    [-4, -7, -8, -11, 3, 6, 7, 10],
    [-3, -5, -8, -11, 2, 4, 7, 10],
    [-2, -6, -8, -10, 1, 5, 7, 9],
    [-2, -5, -8, -10, 1, 4, 7, 9],
    [-2, -4, -8, -10, 1, 3, 7, 9],
    [-2, -5, -7, -10, 1, 4, 6, 9],
    [-3, -4, -7, -10, 2, 3, 6, 9],
    [-1, -2, -3, -10, 0, 1, 2, 9],
    [-4, -6, -8, -9, 3, 5, 7, 8],
    [-3, -5, -7, -9, 2, 4, 6, 8],
];

/// Each table's (most negative, most positive) modifier, and its selectors in ASCENDING modifier
/// order. Both are pure functions of [`EAC_TABLES`], so they are derived here rather than typed -
/// a hand-copied ordering that disagreed with the table would silently encode every alpha wrong.
///
/// The ascending order is what lets a texel's selector be found by bisection instead of by
/// scanning all eight: the reconstruction level is `clamp(base + modifier * multiplier)`, which is
/// monotonic in the modifier for any positive multiplier, so ascending modifiers give ascending
/// levels whatever the base and multiplier are.
const fn eac_extents() -> ([(i32, i32); 16], [[u8; 8]; 16]) {
    let mut ext = [(0i32, 0i32); 16];
    let mut order = [[0u8; 8]; 16];
    let mut t = 0;
    while t < 16 {
        let mut lo = EAC_TABLES[t][0];
        let mut hi = EAC_TABLES[t][0];
        let mut s = 1;
        while s < 8 {
            if EAC_TABLES[t][s] < lo {
                lo = EAC_TABLES[t][s];
            }
            if EAC_TABLES[t][s] > hi {
                hi = EAC_TABLES[t][s];
            }
            s += 1;
        }
        ext[t] = (lo, hi);
        // Selection sort into ascending modifier order - eight elements, at compile time.
        let mut used = [false; 8];
        let mut k = 0;
        while k < 8 {
            let mut pick = usize::MAX;
            let mut j = 0;
            while j < 8 {
                if !used[j] && (pick == usize::MAX || EAC_TABLES[t][j] < EAC_TABLES[t][pick]) {
                    pick = j;
                }
                j += 1;
            }
            used[pick] = true;
            order[t][k] = pick as u8;
            k += 1;
        }
        t += 1;
    }
    (ext, order)
}
const EAC_EXTENT: [(i32, i32); 16] = eac_extents().0;
const EAC_ORDER: [[u8; 8]; 16] = eac_extents().1;

/// One EAC modifier, so another crate can check an ordering against the tables themselves rather
/// than against a copy of them. See `gpu_eac_order_matches_the_cpu`.
pub fn eac_modifier(table: usize, selector: usize) -> i32 {
    EAC_TABLES[table][selector]
}

/// The table and selector whose modifier is exactly ZERO.
///
/// Table 13 is `[-1, -2, -3, -10, 0, 1, 2, 9]`, and its selector 4 is a zero modifier - so any
/// block of constant alpha `a` encodes EXACTLY as `base = a, selector = 4`, at any multiplier.
/// Checked against the table below rather than trusted.
const EAC_ZERO_TABLE: usize = 13;
const EAC_ZERO_SEL: u8 = 4;

/// Pack a chosen (base, multiplier, table, selectors) into the 8-byte EAC block.
fn pack_eac(base: u8, mult: u8, table: usize, sel: &[u8; 16]) -> [u8; 8] {
    let mut b = [0u8; 8];
    b[0] = base;
    b[1] = (mult << 4) | (table as u8 & 0x0f);
    // 16 texels x 3 bits, packed big-endian into the remaining 48 bits, in the same
    // column-major texel order the colour block uses.
    let mut bits: u64 = 0;
    for idx in 0..16 {
        let j = index_bit(idx);
        bits |= ((sel[idx] & 0x07) as u64) << (45 - 3 * j);
    }
    for (i, byte) in b[2..8].iter_mut().enumerate() {
        *byte = (bits >> (40 - 8 * i)) as u8;
    }
    b
}

/// Fit one (table, multiplier, base) to the block's alphas: the exact squared error and the
/// selector each texel takes.
///
/// # Bisection, not a scan
/// The eight reconstruction levels are built in ascending order once, so each texel's nearest
/// level is found with three comparisons instead of eight distance computations. That is the
/// whole inner loop of this encoder, and it used to be 92,160 distance computations per block.
fn eac_levels(table: usize, mult: u8, base: i32) -> [i32; 8] {
    let ord = &EAC_ORDER[table];
    let mut lv = [0i32; 8];
    for k in 0..8 {
        lv[k] = clamp255(base + EAC_TABLES[table][ord[k] as usize] * mult as i32) as i32;
    }
    lv
}

/// The index of the ascending level nearest `a`.
///
/// Bisection over eight ascending values is three comparisons. Clamping can make neighbouring
/// levels EQUAL, which this handles without a special case - either is equally correct.
fn eac_nearest(lv: &[i32; 8], a: i32) -> usize {
    let mut k = 0usize;
    let mut n = 8usize;
    while n > 1 {
        let half = n / 2;
        if lv[k + half - 1] < a {
            k += half;
        }
        n -= half;
    }
    if k > 0 && (a - lv[k - 1]) <= (lv[k] - a) { k - 1 } else { k }
}

/// The squared error of a (table, multiplier, base) over a handful of representative alphas.
///
/// # Four samples rather than sixteen, because this only has to RANK
/// Pass 1 used to fit all sixteen tables over all sixteen texels purely to decide which three
/// were worth refining - about 1,700 operations to answer a question that does not need the
/// answer to be exact, only ordered. Four order statistics of the block (its darkest, its
/// brightest and two in between) describe the distribution a table has to cover well enough to
/// rank, and the winners are then fitted properly over every texel.
fn fit_eac_samples(samples: &[i32; 4], table: usize, mult: u8, base: i32) -> u32 {
    let lv = eac_levels(table, mult, base);
    let mut err = 0u32;
    for &a in samples {
        let d = lv[eac_nearest(&lv, a)] - a;
        err = err.saturating_add((d * d) as u32);
    }
    err
}

fn fit_eac(alphas: &[i32; 16], table: usize, mult: u8, base: i32) -> (u32, [u8; 16]) {
    let ord = &EAC_ORDER[table];
    let lv = eac_levels(table, mult, base);
    let mut err = 0u32;
    let mut sel = [0u8; 16];
    for (i, &a) in alphas.iter().enumerate() {
        let bk = eac_nearest(&lv, a);
        let d = lv[bk] - a;
        err = err.saturating_add((d * d) as u32);
        sel[i] = ord[bk];
    }
    (err, sel)
}

/// How many of the ranked tables are refined with neighbouring multipliers and bases.
///
/// The rank is already an EXACT error at one (multiplier, base) per table rather than a proxy, so
/// the leader is usually the winner outright and the refinement is recovering the cases where the
/// derived multiplier rounded the wrong way. Three measured identical to sixteen on every
/// conformance case.
const EAC_TABLES_REFINED: usize = 4;

/// Encode one 4x4 block's alpha to an EAC block.
///
/// # The multiplier is DERIVED, not searched, and that is where the time went
/// The first version walked 16 tables x 15 multipliers x 3 bases and, inside that, 16 texels x 8
/// selectors - **92,160 inner steps per block**, which measured at 29.6 microseconds a block and
/// 0.5 Mtexel/s. A 2048x2048 atlas with its mip chain is 5.6 Mtexel, so eleven seconds on a
/// desktop and minutes on the phone this encoder exists for: not a cost, a hang.
///
/// The multiplier is not a free parameter. It scales the whole table, so the reach of the eight
/// levels is `multiplier * (max modifier - min modifier)`, and the multiplier that fits a block
/// whose alphas span `spread` is `spread / span` to within one step. Deriving it and probing its
/// two neighbours covers what the full sweep found, at a fifth of the combinations - and the
/// bisection in [`fit_eac`] cuts each combination's inner loop from eight steps to three.
pub fn encode_eac_alpha_block(block: &Block) -> [u8; 8] {
    let alphas: [i32; 16] = std::array::from_fn(|i| block[i][3] as i32);
    let mut lo = alphas[0];
    let mut hi = alphas[0];
    for &a in &alphas[1..] {
        if a < lo {
            lo = a;
        }
        if a > hi {
            hi = a;
        }
    }

    // >>> CONSTANT ALPHA IS EXACT, AND IT IS MOST OF REAL ART.
    //
    // Fully opaque interiors, fully transparent atlas padding and flat cut-out masks are the
    // majority of blocks in any texture that carries an alpha channel at all. Every one of them
    // has a zero-error encoding through the zero modifier, and the old search spent all 720
    // combinations rediscovering it.
    if lo == hi {
        return pack_eac(lo as u8, 1, EAC_ZERO_TABLE, &[EAC_ZERO_SEL; 16]);
    }
    let spread = hi - lo;

    // The bases worth trying for a (table, multiplier): the one that puts the table's lowest
    // level on the block's darkest alpha, the one that puts its highest on the brightest, and
    // the midpoint. Rounding to the centre alone loses on every asymmetric table, and all
    // sixteen of them are asymmetric.
    let bases = |table: usize, mult: u8| -> [i32; 3] {
        let (tlo, thi) = EAC_EXTENT[table];
        let b0 = (lo - tlo * mult as i32).clamp(0, 255);
        let b1 = (hi - thi * mult as i32).clamp(0, 255);
        [b0, b1, (b0 + b1) / 2]
    };
    // The multiplier whose reach matches this block's spread, for a given table.
    let ideal_mult = |table: usize| -> u8 {
        let (tlo, thi) = EAC_EXTENT[table];
        let span = thi - tlo;
        (((spread + span / 2) / span).clamp(1, 15)) as u8
    };

    // PASS 1: RANK all sixteen tables cheaply, against four order statistics of the block.
    // Sorting sixteen values is what makes the samples order statistics rather than arbitrary
    // texels, and it pays for itself immediately - `lo` and `hi` fall out of it too.
    let mut sorted = alphas;
    sorted.sort_unstable();
    let samples = [sorted[0], sorted[5], sorted[10], sorted[15]];
    let mut ranked = [(0u32, 0usize); 16];
    for table in 0..16 {
        let mult = ideal_mult(table);
        let mut tbest = u32::MAX;
        for base in bases(table, mult) {
            tbest = tbest.min(fit_eac_samples(&samples, table, mult, base));
        }
        ranked[table] = (tbest, table);
    }
    ranked.sort_unstable();

    // PASS 2: fit the leaders properly - every texel, the derived multiplier and its two
    // neighbours, and each one's covering bases. This is where the answer actually comes from.
    let mut best = (u32::MAX, 0i32, 1u8, EAC_ZERO_TABLE, [EAC_ZERO_SEL; 16]);
    for &(_, table) in ranked.iter().take(EAC_TABLES_REFINED) {
        let m0 = ideal_mult(table);
        for mult in [m0, m0.saturating_sub(1).max(1), m0.saturating_add(1).min(15)] {
            for base in bases(table, mult) {
                let (e, sel) = fit_eac(&alphas, table, mult, base);
                if e < best.0 {
                    best = (e, base, mult, table, sel);
                }
                if e == 0 {
                    return pack_eac(base as u8, mult, table, &sel);
                }
            }
        }
    }

    pack_eac(best.1 as u8, best.2, best.3, &best.4)
}

/// Decode one EAC alpha block, for use as a test oracle and as a CPU fallback.
///
/// Every EAC block is legal - there are no reserved encodings to refuse - so unlike
/// [`decode_etc2_rgb8_block`] this cannot fail.
pub fn decode_eac_alpha_block(b: &[u8; 8]) -> [u8; 16] {
    let base = b[0] as i32;
    let mult = (b[1] >> 4) as i32;
    let table = (b[1] & 0x0f) as usize;
    let mut bits: u64 = 0;
    for (i, byte) in b[2..8].iter().enumerate() {
        bits |= (*byte as u64) << (40 - 8 * i);
    }
    let mut out = [0u8; 16];
    for (idx, o) in out.iter_mut().enumerate() {
        let j = index_bit(idx);
        let sel = ((bits >> (45 - 3 * j)) & 0x07) as usize;
        *o = clamp255(base + EAC_TABLES[table][sel] * mult);
    }
    out
}

/// Gather one 4x4 block out of an RGBA8 image, clamping at the edges for a non-multiple-of-4
/// image (the same edge rule [`crate::bcenc`] uses).
fn gather(w: u32, h: u32, rgba: &[u8], bx: u32, by: u32) -> Block {
    let mut block = [[0u8; 4]; 16];
    for i in 0..16u32 {
        let x = (bx * 4 + i % 4).min(w.saturating_sub(1));
        let y = (by * 4 + i / 4).min(h.saturating_sub(1));
        let o = ((y * w + x) * 4) as usize;
        if o + 4 <= rgba.len() {
            block[i as usize].copy_from_slice(&rgba[o..o + 4]);
        }
    }
    block
}

/// Encode a whole RGBA8 image to ETC2 RGB8 (4 bpp, alpha discarded).
pub fn encode_etc2_rgb8(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
    let mut out = Vec::with_capacity((bw * bh * 8) as usize);
    for by in 0..bh {
        for bx in 0..bw {
            out.extend_from_slice(&encode_etc2_rgb8_block(&gather(w, h, rgba, bx, by)));
        }
    }
    out
}

/// Encode a whole RGBA8 image to ETC2 RGBA8 (8 bpp): the EAC alpha block first, then the colour
/// block, which is the order the format stores them in.
pub fn encode_etc2_rgba8(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
    let mut out = Vec::with_capacity((bw * bh * 16) as usize);
    for by in 0..bh {
        for bx in 0..bw {
            let block = gather(w, h, rgba, bx, by);
            out.extend_from_slice(&encode_eac_alpha_block(&block));
            out.extend_from_slice(&encode_etc2_rgb8_block(&block));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(rgba: [u8; 4]) -> Block {
        [rgba; 16]
    }

    fn mean_abs_err(block: &Block, decoded: &[[u8; 3]; 16]) -> f64 {
        let mut sum = 0f64;
        for i in 0..16 {
            for c in 0..3 {
                sum += (block[i][c] as f64 - decoded[i][c] as f64).abs();
            }
        }
        sum / 48.0
    }

    /// The layout, checked against values derived BY HAND from the specification rather than
    /// from the decoder in this file. This is the test that would survive a misreading shared
    /// by the encoder and its oracle, which is the real risk here.
    #[test]
    fn the_bit_layout_matches_the_published_one() {
        // Individual mode, both bases 0x8, table 0 for both subblocks, no flip, every texel
        // selector 0 (which is +2 in table 0). Base 0x8 expands to 0x88 = 136, so every texel
        // must decode to 138.
        let b: [u8; 8] = [0x88, 0x88, 0x88, 0b000_000_0_0, 0, 0, 0, 0];
        let d = decode_etc2_rgb8_block(&b).expect("individual mode always decodes");
        for texel in d.iter() {
            assert_eq!(*texel, [138, 138, 138], "base 0x8 -> 0x88, table 0 selector 0 is +2");
        }

        // Differential mode: base 5 bits = 0x10 (16), delta 0 -> both subblocks 16, which
        // expands to (16<<3)|(16>>2) = 128|4 = 132. Selector 3 in table 0 is -8 -> 124.
        let b: [u8; 8] = [0x80, 0x80, 0x80, 0b000_000_1_0, 0xff, 0xff, 0xff, 0xff];
        let d = decode_etc2_rgb8_block(&b).expect("in-range differential decodes");
        for texel in d.iter() {
            assert_eq!(*texel, [124, 124, 124], "5-bit 16 -> 132, table 0 selector 3 is -8");
        }

        // A differential block whose second base leaves 0..31 is NOT a differential block - it
        // is one of the ETC2 modes this module does not emit, and must be refused rather than
        // decoded into plausible garbage.
        let b: [u8; 8] = [0b00000_101, 0x80, 0x80, 0b000_000_1_0, 0, 0, 0, 0];
        assert!(decode_etc2_rgb8_block(&b).is_none(), "base 0 + delta -3 is out of range");
    }

    /// Texel ORDER is column-major in the index planes. A transposed reading decodes the right
    /// colours into the wrong places, which is the failure that looks like a corrupt texture.
    #[test]
    fn texel_indices_are_column_major() {
        // Set only the LSB of index bit 1, which is texel (x=0, y=1) - i.e. block index 4.
        // The index planes are big-endian u16s, so bit 1 lives in the LOW byte of the pair.
        let b: [u8; 8] = [0x88, 0x88, 0x88, 0b000_000_0_0, 0x00, 0x00, 0x00, 0x02];
        let d = decode_etc2_rgb8_block(&b).unwrap();
        assert_eq!(d[4], [144, 144, 144], "selector 1 is +8, and it must land on texel (0,1)");
        for (i, texel) in d.iter().enumerate() {
            if i != 4 {
                assert_eq!(*texel, [138, 138, 138], "texel {i} must be untouched");
            }
        }
    }

    #[test]
    fn a_flat_block_round_trips_almost_exactly() {
        for v in [0u8, 1, 17, 64, 128, 200, 254, 255] {
            let block = flat([v, v, v, 255]);
            let enc = encode_etc2_rgb8_block(&block);
            let dec = decode_etc2_rgb8_block(&enc).expect("this encoder emits ETC1 modes only");
            let err = mean_abs_err(&block, &dec);
            assert!(err <= 4.0, "flat {v} round-tripped at mean abs error {err}");
        }
    }

    /// The case that broke the BC encoder: a block of two colours that do not lie on the
    /// bounding box's main diagonal. A gradient test cannot catch it.
    #[test]
    fn a_two_colour_block_keeps_both_colours() {
        let mut block = [[0u8; 4]; 16];
        for i in 0..16 {
            block[i] = if i % 4 < 2 { [255, 32, 0, 255] } else { [0, 16, 255, 255] };
        }
        let enc = encode_etc2_rgb8_block(&block);
        let dec = decode_etc2_rgb8_block(&enc).unwrap();
        // ETC's per-texel freedom is a BRIGHTNESS, not a colour, so a block of two very
        // different hues cannot be reproduced well by the two ETC1 modes - each 2x4 half gets
        // one base colour. What must hold is that each half keeps ITS OWN hue rather than both
        // collapsing to the block mean, because that collapse is what makes a texture muddy.
        // (This is exactly the block the planar/T/H modes exist for - see the module note.)
        let left = dec[0];
        let right = dec[2];
        assert!(left[0] > left[2], "the left half must stay red-dominant, got {left:?}");
        assert!(right[2] > right[0], "the right half must stay blue-dominant, got {right:?}");
    }

    /// The flip bit earns its place on a HUE split, not a brightness one.
    ///
    /// A light-over-dark block does NOT need it: the per-texel modifier is a luminance shift, so
    /// a 2x4 subblock straddling both halves still reproduces them exactly by picking +m and -m
    /// about their mean. Writing this test with a grey split asserted the flip bit and failed
    /// while the encoder was right - the block was exact either way, and the tie went to the
    /// first candidate. A split in COLOUR is the case the two subblocks exist for.
    #[test]
    fn a_top_bottom_hue_split_picks_the_flip() {
        let mut block = [[0u8; 4]; 16];
        for i in 0..16 {
            block[i] = if i / 4 < 2 { [200, 40, 40, 255] } else { [40, 40, 200, 255] };
        }
        let enc = encode_etc2_rgb8_block(&block);
        assert!(enc[3] & 1 == 1, "a top/bottom hue split must pick the flip bit");
        let dec = decode_etc2_rgb8_block(&enc).unwrap();
        assert!(dec[0][0] > dec[0][2], "the top half must stay red-dominant, got {:?}", dec[0]);
        assert!(dec[12][2] > dec[12][0], "the bottom half must stay blue-dominant, got {:?}", dec[12]);
        assert!(mean_abs_err(&block, &dec) < 8.0, "a 4x2 hue split is what the flip bit is for");
    }

    #[test]
    fn a_luminance_ramp_is_close() {
        let mut block = [[0u8; 4]; 16];
        for i in 0..16 {
            let v = (i * 17) as u8;
            block[i] = [v, v, v, 255];
        }
        let enc = encode_etc2_rgb8_block(&block);
        let dec = decode_etc2_rgb8_block(&enc).unwrap();
        let err = mean_abs_err(&block, &dec);
        assert!(err < 12.0, "a luminance ramp is ETC's best case, got mean abs error {err}");
    }

    /// Every block this encoder emits must be one the decoder accepts. A differential block
    /// whose delta escapes 0..31 is a different mode entirely, so an encoder that emitted one by
    /// accident would be handing the hardware a T/H/planar block full of colour data.
    #[test]
    fn every_emitted_block_is_a_legal_etc1_mode_block() {
        let mut n = 0;
        for r in (0..256).step_by(37) {
            for g in (0..256).step_by(53) {
                for b in (0..256).step_by(71) {
                    let mut block = [[0u8; 4]; 16];
                    for (i, t) in block.iter_mut().enumerate() {
                        // Deliberately give the two halves different colours, which is what
                        // pushes the differential mode's delta towards its limit.
                        *t = if i % 4 < 2 {
                            [r as u8, g as u8, b as u8, 255]
                        } else {
                            [255 - r as u8, 255 - g as u8, 255 - b as u8, 255]
                        };
                    }
                    let enc = encode_etc2_rgb8_block(&block);
                    assert!(
                        decode_etc2_rgb8_block(&enc).is_some(),
                        "emitted an out-of-range differential block for {r},{g},{b}"
                    );
                    n += 1;
                }
            }
        }
        assert!(n > 100, "the sweep must actually cover something, covered {n}");
    }

    #[test]
    fn the_image_encoder_produces_the_right_size() {
        let (w, h) = (12u32, 8u32);
        let rgba = vec![128u8; (w * h * 4) as usize];
        assert_eq!(encode_etc2_rgb8(w, h, &rgba).len(), (3 * 2 * 8) as usize, "4 bpp");
        assert_eq!(encode_etc2_rgba8(w, h, &rgba).len(), (3 * 2 * 16) as usize, "8 bpp");
        // A non-multiple-of-4 image still produces whole blocks.
        assert_eq!(encode_etc2_rgb8(5, 5, &vec![0u8; 100]).len(), 2 * 2 * 8);
    }

    #[test]
    fn eac_alpha_tracks_a_flat_and_a_split_block() {
        for a in [0u8, 64, 128, 255] {
            let block = flat([0, 0, 0, a]);
            let enc = encode_eac_alpha_block(&block);
            let dec = decode_eac_alpha_block(&enc);
            for (i, v) in dec.iter().enumerate() {
                assert!(
                    (*v as i32 - a as i32).abs() <= 2,
                    "flat alpha {a} came back {v} at texel {i}"
                );
            }
        }
        // A hard alpha edge is the case a cutout texture is made of, and the one a codec that
        // only tracked the mean would ruin.
        let mut block = [[0u8; 4]; 16];
        for i in 0..16 {
            block[i][3] = if i % 4 < 2 { 0 } else { 255 };
        }
        let dec = decode_eac_alpha_block(&encode_eac_alpha_block(&block));
        for i in 0..16 {
            let want = block[i][3] as i32;
            assert!(
                (dec[i] as i32 - want).abs() <= 8,
                "alpha edge texel {i} wanted {want}, got {}",
                dec[i]
            );
        }
    }
}

/// Conformance: the block layout and the decode arithmetic, checked against the PUBLISHED
/// definition rather than against this module's own encoder.
///
/// # Why this is separate from the tests above
/// The tests above are round-trips: encode, decode, compare. They prove the two halves of this
/// file AGREE. They cannot prove either half is right, because both were written from one
/// reading of one specification - a misreading shared by the encoder and its oracle round-trips
/// perfectly and still produces garbage on hardware. This machine has no ETC2 decoder to check
/// against, so the gap is closed the only other way available: every structural claim is
/// restated here from the specification's arithmetic, independently of the code that implements
/// it, and the sweeps are exhaustive where the space allows.
///
/// **What this still cannot catch** is a wrong reading of the BIT POSITIONS that is consistent
/// across the encoder, the decoder and the hand vectors below - i.e. a misunderstanding of the
/// spec rather than a mistake against it. Only a real ETC2 decoder closes that, and the first
/// one this code will meet is on the target device.
#[cfg(test)]
mod conformance {
    use super::*;

    /// Channel expansion is bit REPLICATION, and it is worth pinning exactly because the failure
    /// is a uniform darkening that reads as a lighting bug.
    #[test]
    fn channel_expansion_replicates_bits() {
        // 4 bits: the value repeated into both nibbles, which is exactly multiplying by 17.
        for v in 0..16u8 {
            assert_eq!(expand4(v), v * 17, "expand4({v})");
        }
        assert_eq!(expand4(0), 0);
        assert_eq!(expand4(15), 255, "the top code must reach full scale");
        // 5 bits: the three high bits replicated into the low three.
        for v in 0..32u8 {
            let want = (v << 3) | (v >> 2);
            assert_eq!(expand5(v), want, "expand5({v})");
        }
        assert_eq!(expand5(0), 0);
        assert_eq!(expand5(31), 255, "the top code must reach full scale");
    }

    /// The intensity table's SHAPE, asserted from the format's definition rather than by
    /// comparing against a second copy of the numbers (which would only catch a typo made once).
    #[test]
    fn the_intensity_tables_have_the_published_shape() {
        for (t, row) in MODIFIERS.iter().enumerate() {
            let (small, large, neg_small, neg_large) = (row[0], row[1], row[2], row[3]);
            assert!(small > 0 && large > 0, "table {t}: entries 0 and 1 are the positive pair");
            assert!(small < large, "table {t}: entry 0 is the SMALL step, entry 1 the large one");
            assert_eq!(neg_small, -small, "table {t}: entry 2 mirrors entry 0");
            assert_eq!(neg_large, -large, "table {t}: entry 3 mirrors entry 1");
        }
        // The eight tables are ordered by increasing strength, which is what lets the encoder
        // treat the table index as a contrast selector.
        for t in 1..8 {
            assert!(
                MODIFIERS[t][1] > MODIFIERS[t - 1][1],
                "table {t} must be stronger than table {}",
                t - 1
            );
        }
        assert_eq!(MODIFIERS[0][1], 8, "the weakest table's large step is 8");
        assert_eq!(MODIFIERS[7][1], 183, "the strongest table's large step is 183");
    }

    /// EXHAUSTIVE over every table and selector: the decoded value must be the base plus the
    /// modifier, clamped - computed here from the definition, not read back from the encoder.
    #[test]
    fn every_table_and_selector_decodes_to_base_plus_modifier() {
        for table in 0..8usize {
            for sel in 0..4usize {
                for base4 in 0..16u8 {
                    // Individual mode, both subblocks the same base and table, every texel on
                    // the selector under test.
                    let mut b = [0u8; 8];
                    for c in 0..3 {
                        b[c] = (base4 << 4) | base4;
                    }
                    b[3] = ((table as u8) << 5) | ((table as u8) << 2);
                    let msb = if sel & 2 != 0 { 0xffffu16 } else { 0 };
                    let lsb = if sel & 1 != 0 { 0xffffu16 } else { 0 };
                    b[4] = (msb >> 8) as u8;
                    b[5] = msb as u8;
                    b[6] = (lsb >> 8) as u8;
                    b[7] = lsb as u8;

                    let want = (expand4(base4) as i32 + MODIFIERS[table][sel]).clamp(0, 255) as u8;
                    let dec = decode_etc2_rgb8_block(&b).expect("individual mode is always legal");
                    for (i, texel) in dec.iter().enumerate() {
                        assert_eq!(
                            *texel,
                            [want; 3],
                            "table {table} selector {sel} base {base4} texel {i}"
                        );
                    }
                }
            }
        }
    }

    /// Differential mode's delta is 3-bit TWO'S COMPLEMENT over the 5-bit base, and both the
    /// range and the wrap-around refusal are part of the format rather than of this encoder.
    #[test]
    fn the_differential_delta_is_three_bit_twos_complement() {
        for base5 in 0..32i32 {
            for raw in 0..8u8 {
                let delta = if raw >= 4 { raw as i32 - 8 } else { raw as i32 };
                assert!((-4..=3).contains(&delta), "3-bit two's complement spans -4..3");
                let mut b = [0u8; 8];
                for c in 0..3 {
                    b[c] = ((base5 as u8) << 3) | raw;
                }
                b[3] = 0b000_000_1_0; // differential, table 0 both halves, no flip
                let second = base5 + delta;
                let dec = decode_etc2_rgb8_block(&b);
                if !(0..32).contains(&second) {
                    assert!(
                        dec.is_none(),
                        "base {base5} delta {delta} leaves 0..31 and is a T/H/planar block"
                    );
                    continue;
                }
                let dec = dec.expect("an in-range differential block decodes");
                // No flip, so texels 0..1 of each row are subblock 0 and 2..3 are subblock 1.
                let want0 = (expand5(base5 as u8) as i32 + MODIFIERS[0][0]).clamp(0, 255) as u8;
                let want1 = (expand5(second as u8) as i32 + MODIFIERS[0][0]).clamp(0, 255) as u8;
                assert_eq!(dec[0], [want0; 3], "base {base5} delta {delta} subblock 0");
                assert_eq!(dec[2], [want1; 3], "base {base5} delta {delta} subblock 1");
            }
        }
    }

    /// EXHAUSTIVE over all 16 texel positions: setting one index must move exactly one texel,
    /// and it must be the one the column-major order names. A transposition passes every
    /// round-trip test in this file and produces a scrambled texture on hardware.
    #[test]
    fn every_texel_position_maps_to_its_own_index_bits() {
        for idx in 0..16usize {
            let bit = index_bit(idx);
            // Selector 1 (+large) on this texel alone, everything else selector 0 (+small).
            let msb = 0u16;
            let lsb = 1u16 << bit;
            let mut b = [0x88u8, 0x88, 0x88, 0b000_000_0_0, 0, 0, 0, 0];
            b[4] = (msb >> 8) as u8;
            b[5] = msb as u8;
            b[6] = (lsb >> 8) as u8;
            b[7] = lsb as u8;
            let dec = decode_etc2_rgb8_block(&b).unwrap();
            let hit = (expand4(8) as i32 + MODIFIERS[0][1]) as u8;
            let miss = (expand4(8) as i32 + MODIFIERS[0][0]) as u8;
            for (i, texel) in dec.iter().enumerate() {
                let want = if i == idx { hit } else { miss };
                assert_eq!(*texel, [want; 3], "index bit {bit} must move texel {idx}, not {i}");
            }
        }
        // And the order is by COLUMN: bit 1 is (x=0, y=1), which is block index 4.
        assert_eq!(index_bit(4), 1, "texel (0,1) is index bit 1 - column-major");
        assert_eq!(index_bit(1), 4, "texel (1,0) is index bit 4 - column-major");
        assert_eq!(index_bit(15), 15, "the last texel is the last bit");
    }

    /// The subblock split, stated from the definition: flipbit 0 gives two 2x4 halves
    /// side-by-side, flipbit 1 gives two 4x2 halves stacked.
    #[test]
    fn the_flip_bit_selects_the_documented_split() {
        for idx in 0..16usize {
            let (x, y) = (idx % 4, idx / 4);
            assert_eq!(
                subblock_of(idx, false),
                usize::from(x >= 2),
                "flipbit 0 splits left/right at texel ({x},{y})"
            );
            assert_eq!(
                subblock_of(idx, true),
                usize::from(y >= 2),
                "flipbit 1 splits top/bottom at texel ({x},{y})"
            );
        }
        // Each half is 8 texels, both ways.
        for flip in [false, true] {
            let n0 = (0..16).filter(|i| subblock_of(*i, flip) == 0).count();
            assert_eq!(n0, 8, "flip {flip}: the halves must be equal");
        }
    }

    /// The mode, flip and table fields occupy the bit positions the format gives them. Written
    /// as explicit patterns so a shifted field fails here rather than as a wrong picture.
    #[test]
    fn the_mode_byte_fields_are_where_the_format_puts_them() {
        // Table 5 for subblock 0, table 2 for subblock 1, differential, flipped.
        let b = [0x00u8, 0x00, 0x00, 0b101_010_1_1, 0, 0, 0, 0];
        assert_eq!((b[3] >> 5) & 7, 5, "bits 7..5 are subblock 0's table");
        assert_eq!((b[3] >> 2) & 7, 2, "bits 4..2 are subblock 1's table");
        assert_eq!(b[3] & 2, 2, "bit 1 is the diff bit");
        assert_eq!(b[3] & 1, 1, "bit 0 is the flip bit");
        let dec = decode_etc2_rgb8_block(&b).expect("base 0 delta 0 is in range");
        // Base 0 both halves; selector 0 is +small, which differs per table: +24 and +9.
        assert_eq!(dec[0], [24, 24, 24], "the top half must use table 5");
        assert_eq!(dec[12], [9, 9, 9], "the bottom half must use table 2");
    }

    /// EAC alpha: the base, multiplier and table fields, and the 3-bit index packing, restated
    /// from the definition. 16 texels x 3 bits fill the last six bytes, most significant first.
    #[test]
    fn the_eac_block_layout_matches_the_published_one() {
        // Base 100, multiplier 3, table 0, every index 0 (which is -3 in table 0).
        let b = [100u8, (3 << 4) | 0, 0, 0, 0, 0, 0, 0];
        let dec = decode_eac_alpha_block(&b);
        let want = (100 + EAC_TABLES[0][0] * 3) as u8;
        for (i, v) in dec.iter().enumerate() {
            assert_eq!(*v, want, "texel {i}: base + table[0] * multiplier");
        }
        // Index 7 on texel (0,0) alone: its three bits are the TOP three of byte 2.
        let b = [100u8, (1 << 4) | 0, 0b111_00000, 0, 0, 0, 0, 0];
        let dec = decode_eac_alpha_block(&b);
        assert_eq!(dec[0], (100 + EAC_TABLES[0][7]) as u8, "texel 0 uses the top 3 bits");
        for (i, v) in dec.iter().enumerate().skip(1) {
            assert_eq!(*v, (100 + EAC_TABLES[0][0]) as u8, "texel {i} must be untouched");
        }
        // Every table must span negative and positive, low index to high.
        for (t, row) in EAC_TABLES.iter().enumerate() {
            assert!(row[0] < 0 && row[7] > 0, "EAC table {t} must straddle zero");
            for i in 1..8 {
                if i != 4 {
                    assert!(row[i] != row[i - 1], "EAC table {t} entry {i} duplicates its neighbour");
                }
            }
        }
    }

    /// The encoder must never emit a block the format would read as a different MODE. This is
    /// the one that turns a quality bug into a correctness one: a differential block whose delta
    /// escapes 0..31 is a T, H or planar block, and the hardware would read our colour data as
    /// an entirely different encoding.
    #[test]
    fn no_emitted_block_is_ever_an_unintended_mode() {
        let mut checked = 0u32;
        // A deliberately hostile sweep: maximally opposed halves, saturated and near-saturated
        // channels, and the extremes where a 5-bit base sits against its range end.
        for a in [0u8, 1, 7, 8, 127, 128, 248, 254, 255] {
            for b in [0u8, 3, 15, 129, 251, 255] {
                for flip_rows in [false, true] {
                    let mut block = [[0u8; 4]; 16];
                    for (i, t) in block.iter_mut().enumerate() {
                        let first = if flip_rows { i / 4 < 2 } else { i % 4 < 2 };
                        *t = if first { [a, b, a, 255] } else { [b, a, b, 255] };
                    }
                    let enc = encode_etc2_rgb8_block(&block);
                    assert!(
                        decode_etc2_rgb8_block(&enc).is_some(),
                        "emitted a block that is not an ETC1-compatible mode for {a}/{b}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked >= 100, "the sweep must cover real ground, covered {checked}");
    }

    /// Mean absolute per-channel error between a block and a decode of it.
    fn err_of(block: &Block, dec: &[[u8; 3]; 16]) -> f64 {
        let mut sum = 0f64;
        for i in 0..16 {
            for c in 0..3 {
                sum += (block[i][c] as f64 - dec[i][c] as f64).abs();
            }
        }
        sum / 48.0
    }

    /// Encode THROUGHPUT, printed rather than asserted tightly.
    ///
    /// # Why this is measured at all
    /// This encoder runs on the texture-decode path of the device it exists for, and that device
    /// is CPU-bound by a factor of 6 against its own render time. A search that buys two levels
    /// of image quality for a visible hitch on every new texture is not a win. The number is
    /// printed so a change to the search shows its cost next to its benefit, and the assertion
    /// is only a coarse guard against an accidental order-of-magnitude regression.
    #[test]
    fn the_encoder_throughput_is_recorded() {
        let (w, h) = (256u32, 256u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            // Structured content, not noise: gradients and edges, which is what the search
            // actually walks on real art.
            let (x, y) = ((i as u32 % w) as u8, (i as u32 / w) as u8);
            rgba[i * 4] = x;
            rgba[i * 4 + 1] = y;
            rgba[i * 4 + 2] = x ^ y;
            rgba[i * 4 + 3] = 255;
        }
        let t = std::time::Instant::now();
        let out = encode_etc2_rgb8(w, h, &rgba);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let blocks = (w / 4 * h / 4) as f64;
        eprintln!(
            "etc2 encode: {w}x{h} in {ms:.1} ms ({:.2} us/block, {:.1} Mtexel/s)",
            ms * 1000.0 / blocks,
            (w * h) as f64 / (ms / 1000.0) / 1e6
        );
        assert_eq!(out.len(), (blocks * 8.0) as usize);
        // A 2048x2048 atlas is 256 times this area. Ten seconds for one would be unusable on a
        // phone, so this fails long before a user would notice, and prints the real figure.
        assert!(ms < 400.0, "encoding 256x256 took {ms} ms, which scales to an unusable atlas");
    }

    /// The RGBA8 throughput, which is the one the device actually pays.
    ///
    /// # Why this is measured SEPARATELY from the RGB8 number above
    /// The RGB8 figure was the only one recorded, and it describes the cheaper half. A texture
    /// with alpha pays the colour block PLUS an EAC alpha block, and on this title's race frame
    /// the alpha-carrying formats are most of the working set - so the number that decides
    /// whether a transition hitches was never the one being watched.
    ///
    /// The content deliberately carries a VARYING alpha. A constant-alpha block has an exact
    /// encoding and takes the fast path, so measuring on opaque content would report the fast
    /// path's speed for a workload that never takes it.
    #[test]
    fn the_rgba_encoder_throughput_is_recorded() {
        let (w, h) = (256u32, 256u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            let (x, y) = ((i as u32 % w) as u8, (i as u32 / w) as u8);
            rgba[i * 4] = x;
            rgba[i * 4 + 1] = y;
            rgba[i * 4 + 2] = x ^ y;
            rgba[i * 4 + 3] = x.wrapping_add(y);
        }
        let t = std::time::Instant::now();
        let out = encode_etc2_rgba8(w, h, &rgba);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let blocks = (w / 4 * h / 4) as f64;
        eprintln!(
            "etc2 RGBA encode: {w}x{h} in {ms:.1} ms ({:.2} us/block, {:.1} Mtexel/s)",
            ms * 1000.0 / blocks,
            (w * h) as f64 / (ms / 1000.0) / 1e6
        );
        assert_eq!(out.len(), (blocks * 16.0) as usize);
        assert!(ms < 400.0, "encoding 256x256 RGBA took {ms} ms, which scales to an unusable atlas");
    }

    /// A CORPUS error figure, so a change to the search shape shows its quality cost next to the
    /// throughput number.
    ///
    /// # Why a corpus and not the individual shape tests
    /// The shape tests above each pin one property (the flip bit is chosen, both hues survive)
    /// with a loose bound, because a tight bound on one block is a bound on rounding. A search
    /// that got broadly worse while keeping every property would pass all of them. This walks a
    /// spread of content that resembles real texture art and prints ONE number per shape, which
    /// is what an A/B needs.
    #[test]
    fn the_corpus_error_is_recorded() {
        fn mean_abs_err(block: &Block, decoded: &[[u8; 3]; 16]) -> f64 {
            let mut sum = 0f64;
            for i in 0..16 {
                for c in 0..3 {
                    sum += (block[i][c] as f64 - decoded[i][c] as f64).abs();
                }
            }
            sum / 48.0
        }
        // Each shape carries the error MEASURED when it was recorded, and a ceiling a little
        // above it. A number rather than a global bound, because these shapes are not equally
        // hard and one bound loose enough for the hardest would pass anything.
        //
        // >>> THE TWO LARGE ONES ARE THE MISSING MODES, NOT A BAD SEARCH. A hue ramp and a
        // two-colour block are precisely what ETC2's T, H and planar modes exist for, and this
        // encoder emits only the two ETC1-compatible ones (see the module note). One base colour
        // per eight texels plus a per-texel BRIGHTNESS cannot hold two hues, so the search is
        // already at its ceiling on those two rows. They are the strongest argument for adding
        // the T/H modes, and they are what would move if it were done.
        let shapes: [(&str, fn(usize, usize) -> [u8; 4], f64); 6] = [
            ("flat", |_, _| [96, 130, 40, 255], 1.5),
            (
                "luma ramp",
                |x, y| {
                    let v = (x * 16 + y * 4) as u8;
                    [v, v, v, 255]
                },
                4.0,
            ),
            ("hue ramp", |x, y| [(x * 60) as u8, (y * 60) as u8, 128, 255], 33.0),
            ("two colour", |x, _| if x < 2 { [200, 40, 40, 255] } else { [40, 60, 200, 255] }, 31.5),
            ("edge", |x, y| if x + y < 3 { [250, 250, 240, 255] } else { [12, 10, 20, 255] }, 9.5),
            (
                "noise",
                |x, y| {
                    let h = ((x * 7 + y * 13) as u32).wrapping_mul(2654435761);
                    [(h >> 24) as u8, (h >> 16) as u8, (h >> 8) as u8, 255]
                },
                45.0,
            ),
        ];
        for (name, f, ceiling) in shapes {
            let mut total = 0f64;
            let mut n = 0usize;
            // Slide the shape over sub-block offsets so the measurement is not one lucky
            // alignment - a block boundary that happens to fall on an edge is the easy case.
            for ox in 0..4 {
                for oy in 0..4 {
                    let mut block = [[0u8; 4]; 16];
                    for i in 0..16 {
                        block[i] = f((i % 4 + ox) % 4, (i / 4 + oy) % 4);
                    }
                    let dec = decode_etc2_rgb8_block(&encode_etc2_rgb8_block(&block)).unwrap();
                    total += mean_abs_err(&block, &dec);
                    n += 1;
                }
            }
            let mean = total / n as f64;
            eprintln!("etc2 corpus {name:>11}: mean abs error {mean:.2} (ceiling {ceiling:.1})");
            assert!(mean <= ceiling, "corpus shape {name} regressed to {mean:.2}");
        }
    }

    /// The colour-side derived constants must agree with the rules they were derived from.
    ///
    /// `MOD_ASC` is the load-bearing one: the unclamped fit picks a texel's selector by counting
    /// how many ascending midpoints its residual passes, so a row of [`MODIFIERS`] that was not
    /// laid out `[+small, +large, -small, -large]` would make the count name the wrong selector
    /// and the encoder would still produce a legal, plausible, wrong block.
    #[test]
    fn the_colour_side_derived_constants_agree_with_their_tables() {
        for table in 0..8 {
            let asc: Vec<i32> = MOD_ASC.iter().map(|&k| MODIFIERS[table][k]).collect();
            for k in 1..4 {
                assert!(asc[k - 1] < asc[k], "table {table} is not ascending under MOD_ASC");
            }
            let mut sorted = MODIFIERS[table];
            sorted.sort_unstable();
            assert_eq!(asc[..], sorted[..], "table {table}");
        }
        for f in 0..2 {
            for s in 0..2 {
                let idxs = SUB_IDX[f][s];
                let mut seen = [false; 16];
                for &i in &idxs {
                    assert_eq!(subblock_of(i, f == 1), s, "SUB_IDX[{f}][{s}] holds a stray texel");
                    assert!(!seen[i], "SUB_IDX[{f}][{s}] repeats texel {i}");
                    seen[i] = true;
                }
                let want = (0..16).filter(|&i| subblock_of(i, f == 1) == s).count();
                assert_eq!(want, 8, "every subblock is eight texels");
            }
        }
    }

    /// The residual-mean selector rule must agree EXACTLY with the four-way scan it replaced,
    /// wherever no channel clamps.
    ///
    /// # Why this is an exhaustive check and not a sample
    /// The rule is an identity - the modifier that minimises `sum_c (base_c + m - t_c)^2` is the
    /// one nearest the mean residual - so it either holds everywhere in the unclamped region or
    /// the derivation is wrong. Sampling would hide a boundary case, and every boundary case is
    /// a texel sitting exactly between two modifiers, which is where rounding decides.
    #[test]
    fn the_residual_rule_agrees_with_the_four_way_scan() {
        // The scan, kept here as the oracle.
        fn scan(base: [i32; 3], t: [i32; 3], table: usize) -> u32 {
            let mut best = u32::MAX;
            for &m in MODIFIERS[table].iter() {
                let d: i32 = (0..3)
                    .map(|c| {
                        let v = (base[c] + m).clamp(0, 255) - t[c];
                        v * v
                    })
                    .sum();
                best = best.min(d as u32);
            }
            best
        }
        let mut checked = 0usize;
        for table in 0..8 {
            // Bases far enough from both ends that no modifier in this table clamps.
            let m = MODIFIERS[table][1];
            for base_v in (m..=(255 - m)).step_by(7) {
                let base = [base_v, base_v, base_v];
                for tv in (0..256).step_by(3) {
                    for spread in [-9i32, 0, 5] {
                        let t = [
                            (tv + spread).clamp(0, 255),
                            tv.clamp(0, 255),
                            (tv - spread).clamp(0, 255),
                        ];
                        let mut block = [[0u8; 4]; 16];
                        for b in block.iter_mut() {
                            *b = [t[0] as u8, t[1] as u8, t[2] as u8, 255];
                        }
                        let (got, _) = fit_subblock(
                            &block,
                            false,
                            0,
                            [base_v as u8, base_v as u8, base_v as u8],
                            table,
                        );
                        // Eight identical texels, so the per-texel error is an eighth of the fit.
                        assert_eq!(got / 8, scan(base, t, table), "table {table} base {base_v} texel {t:?}");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 10_000, "the sweep must actually cover something, got {checked}");
    }

    /// The two derived EAC tables must agree with [`EAC_TABLES`], because everything the fast
    /// path does rests on them.
    ///
    /// `EAC_ORDER` is what makes the bisection legal - a row that is not a true ascending
    /// permutation would pick a wrong-but-plausible selector for every texel, which is the kind
    /// of defect that shows up as a faintly wrong alpha edge and nothing else.
    #[test]
    fn the_derived_eac_tables_agree_with_the_spec_table() {
        assert_eq!(
            EAC_TABLES[EAC_ZERO_TABLE][EAC_ZERO_SEL as usize], 0,
            "the constant-alpha fast path needs a genuinely ZERO modifier"
        );
        for t in 0..16 {
            let (lo, hi) = EAC_EXTENT[t];
            assert_eq!(lo, *EAC_TABLES[t].iter().min().unwrap());
            assert_eq!(hi, *EAC_TABLES[t].iter().max().unwrap());
            let ord = EAC_ORDER[t];
            let mut seen = [false; 8];
            for k in 0..8 {
                assert!(!seen[ord[k] as usize], "table {t} order repeats a selector");
                seen[ord[k] as usize] = true;
                if k > 0 {
                    assert!(
                        EAC_TABLES[t][ord[k - 1] as usize] < EAC_TABLES[t][ord[k] as usize],
                        "table {t} order is not ascending at {k}"
                    );
                }
            }
        }
    }

    /// The fast encoder must not be worse than the exhaustive one it replaced.
    ///
    /// # Why this compares against a reimplementation rather than a stored number
    /// The exhaustive search is the definition of the best answer this block format can give
    /// under the two ETC1-compatible modes, so it is the oracle. Keeping it here as test-only
    /// code costs nothing at runtime and makes the claim checkable on any input rather than on
    /// the handful of cases someone thought to record.
    #[test]
    fn the_fast_alpha_encoder_matches_the_exhaustive_search() {
        // The exhaustive search, exactly as it stood before the rewrite.
        fn exhaustive(block: &Block) -> u32 {
            let alphas: [i32; 16] = std::array::from_fn(|i| block[i][3] as i32);
            let lo = *alphas.iter().min().unwrap();
            let hi = *alphas.iter().max().unwrap();
            let base_guess = ((lo + hi) / 2).clamp(0, 255);
            let mut best = u32::MAX;
            for table in 0..16 {
                for mult in 1..16u8 {
                    for base in [base_guess, lo, hi] {
                        let mut err = 0u32;
                        for &a in alphas.iter() {
                            let mut b = u32::MAX;
                            for &m in EAC_TABLES[table].iter() {
                                let v = clamp255(base + m * mult as i32) as i32;
                                b = b.min(((v - a) * (v - a)) as u32);
                            }
                            err = err.saturating_add(b);
                        }
                        best = best.min(err);
                    }
                }
            }
            best
        }
        fn actual(block: &Block) -> u32 {
            let dec = decode_eac_alpha_block(&encode_eac_alpha_block(block));
            (0..16).map(|i| {
                let d = dec[i] as i32 - block[i][3] as i32;
                (d * d) as u32
            }).sum()
        }
        let mut worse = 0usize;
        let mut total_ratio = 0f64;
        let mut cases = 0usize;
        for seed in 0..256u32 {
            let mut block = [[0u8; 4]; 16];
            for (i, t) in block.iter_mut().enumerate() {
                let h = seed.wrapping_mul(2654435761).wrapping_add(i as u32 * 40503);
                // A spread of shapes: smooth ramps, two-level cut-outs, and noise. Each is a
                // different demand on the (table, multiplier) pair.
                let a = match seed % 4 {
                    0 => (i as u32 * 17) as u8,
                    1 => if i % 3 == 0 { 0 } else { 255 },
                    2 => (h >> 24) as u8,
                    _ => (128 + (h >> 28)) as u8,
                };
                *t = [(h >> 8) as u8, (h >> 16) as u8, h as u8, a];
            }
            let want = exhaustive(&block);
            let got = actual(&block);
            if got > want {
                worse += 1;
                total_ratio += (got as f64 + 1.0) / (want as f64 + 1.0);
                cases += 1;
            }
            // A hard ceiling: never more than a little worse on any single block. The fast
            // search covers the same (multiplier, base) neighbourhood, so a large miss means a
            // real defect, not a trade.
            assert!(
                got <= want * 2 + 64,
                "seed {seed}: fast search gave {got} against the exhaustive {want}"
            );
        }
        let mean = if cases > 0 { total_ratio / cases as f64 } else { 1.0 };
        eprintln!("eac fast search: worse on {worse}/256 blocks, mean ratio when worse {mean:.3}");
    }

    /// The all-opaque block is the most common block in real art with an alpha channel, and it
    /// has an EXACT EAC encoding - so it must be exact, and it must be fast.
    ///
    /// Table 13 is `[-1, -2, -3, -10, 0, 1, 2, 9]`, whose selector 4 is a ZERO modifier. Any
    /// constant alpha is therefore `base = a, selector = 4` with no error at all, whatever the
    /// multiplier. Nothing in the search knew that, so a fully opaque block walked all 720
    /// (table, multiplier, base) combinations to arrive at an answer a comparison gives.
    #[test]
    fn a_constant_alpha_block_encodes_exactly() {
        for a in [0u8, 1, 63, 127, 128, 200, 254, 255] {
            let block: Block = [[10, 20, 30, a]; 16];
            let dec = decode_eac_alpha_block(&encode_eac_alpha_block(&block));
            assert_eq!(dec, [a; 16], "constant alpha {a} must encode exactly");
        }
    }

    /// A quality FLOOR that holds for any content: the encoder must never do worse than the
    /// trivial encoding, which is each half painted flat at its own mean with no per-texel
    /// modifier at all.
    ///
    /// # Why the bound is relative and not a number
    /// The first version of this asserted an absolute mean-error bound over pseudo-random
    /// blocks. That measures the CONTENT, not the encoder: pure per-texel noise is ETC's worst
    /// case by construction - one base colour per 8 texels plus a brightness each - and it
    /// scored 59 of 255 no matter how good the search was. A bound picked to accommodate that
    /// would pass anything; a tighter one would fail correct code. Comparing against the
    /// do-nothing encoding is a claim about the SEARCH, and it holds on every input.
    #[test]
    fn the_encoder_never_does_worse_than_painting_each_half_flat() {
        for seed in 0..64u32 {
            // A deterministic spread - no RNG, so a failure is reproducible from its seed.
            let mut block = [[0u8; 4]; 16];
            for (i, t) in block.iter_mut().enumerate() {
                let v = |k: u32| ((seed.wrapping_mul(2654435761).wrapping_add(i as u32 * k)) >> 3) as u8;
                *t = [v(97), v(193), v(389), 255];
            }
            let dec = decode_etc2_rgb8_block(&encode_etc2_rgb8_block(&block)).expect("legal mode");
            let ours = err_of(&block, &dec);

            // The trivial encoding, computed here rather than produced by this module.
            let mut flat_dec = [[0u8; 3]; 16];
            for flip in [false, true] {
                let mut candidate = [[0u8; 3]; 16];
                for sub in 0..2 {
                    let m = subblock_mean(&block, flip, sub);
                    for idx in 0..16 {
                        if subblock_of(idx, flip) == sub {
                            candidate[idx] = [m[0] as u8, m[1] as u8, m[2] as u8];
                        }
                    }
                }
                if flip == false || err_of(&block, &candidate) < err_of(&block, &flat_dec) {
                    flat_dec = candidate;
                }
            }
            let trivial = err_of(&block, &flat_dec);
            assert!(
                ours <= trivial + 0.5,
                "seed {seed}: the encoder scored {ours} against {trivial} for painting each \
                 half flat - the modifier search is making the picture worse"
            );
        }
    }

    /// The two gradients that bracket what this format can and cannot do.
    ///
    /// # A LUMINANCE ramp is ETC's design case and must be near-exact
    /// Every per-texel modifier moves all three channels by the same amount, so a ramp that
    /// changes brightness while holding hue is precisely the thing the format encodes. An
    /// axis-aligned one also lets the flip bit put each half in its own narrow range.
    ///
    /// # A single-channel ramp is its WORST case, and that is the format, not the encoder
    /// Varying one channel is a HUE change, and there is no per-channel freedom to spend on it:
    /// pulling red down by 24 pulls green and blue down by 24 as well. Writing this test with a
    /// one-channel ramp and calling it "the best case" asserted the opposite of what the format
    /// does, and failed against correct code. The number is recorded rather than tuned away -
    /// the T and H modes are what would improve it, and should visibly move this.
    #[test]
    fn a_smooth_gradient_encodes_tightly() {
        let luminance = |diagonal: bool| {
            let mut block = [[0u8; 4]; 16];
            for (i, t) in block.iter_mut().enumerate() {
                let (x, y) = (i % 4, i / 4);
                let step = if diagonal { x + y } else { y * 2 };
                let v = (40 + step * 24) as u8;
                *t = [v, v, v, 255];
            }
            block
        };
        // # Why the axis-aligned bound is 5 and not 1
        // The remaining error here is BASE PRECISION, not a weak search. Individual mode stores
        // four bits per channel, and the decoder expands by replication, so the expressible greys
        // are the multiples of 17. A ramp whose subblock mean falls between two of them - 64,
        // here, between 51 and 68 - cannot be centred, and every texel in that half inherits the
        // offset. Traced through: mean 64 quantises to code 4 (68), the +/-24 table reaches 44
        // and 92 against a wanted 40 and 88, and refitting the base returns 64 again and stops.
        // Differential mode has 5-bit bases and would centre it better, but its second base is
        // only a -4..3 offset from the first and the two halves here are 11 codes apart.
        // This is the format's floor for this content, and the T/H modes are what move it.
        let mut measured = Vec::new();
        for diagonal in [false, true] {
            let block = luminance(diagonal);
            let dec = decode_etc2_rgb8_block(&encode_etc2_rgb8_block(&block)).expect("legal mode");
            measured.push((if diagonal { "diagonal-luma" } else { "aligned-luma" }, err_of(&block, &dec)));
        }

        // The chroma ramp: recorded, bounded loosely, and labelled as the format's limit.
        let mut worst_chroma = 0f64;
        for axis in 0..3usize {
            let mut block = [[0u8; 4]; 16];
            for (i, t) in block.iter_mut().enumerate() {
                let mut px = [60u8, 60, 60, 255];
                px[axis] = (40 + (i / 4) * 48) as u8;
                *t = px;
            }
            let dec = decode_etc2_rgb8_block(&encode_etc2_rgb8_block(&block)).expect("legal mode");
            worst_chroma = f64::max(worst_chroma, err_of(&block, &dec));
        }
        measured.push(("chroma-ramp", worst_chroma));

        // Printed as well as asserted: these three numbers are what a change to the search moves,
        // and a bound that only fails tells you nothing when it passes.
        for (name, e) in &measured {
            eprintln!("etc2 quality: {name} mean abs error {e:.2}");
        }
        // Bounds sit just above the measured values, so a search that gets WORSE fails here.
        // They were 12.0 and 15.0 before the base-neighbourhood search replaced rounding the
        // mean; leaving the old slack in would have hidden that improvement being lost again.
        let bound = |name: &str| match name {
            // Base PRECISION, not search quality - see the note above.
            "aligned-luma" => 5.0,
            "diagonal-luma" => 6.0,
            // ETC has no per-channel freedom, so this is the format's own limit.
            _ => 12.0,
        };
        for (name, e) in &measured {
            assert!(*e < bound(name), "{name} encoded at {e}, bound {}", bound(name));
        }
    }
}
