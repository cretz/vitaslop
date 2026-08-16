// Guest texture -> compressed blocks, entirely on the GPU.
//
// Three entry points, run in sequence over one scratch RGBA8 buffer:
//   `decode_pvrtc`  guest PVRTC blocks -> RGBA8, one invocation per source block REGION
//   `halve`         one RGBA8 level -> the next, a box filter, one invocation per destination texel
//   `encode_etc2`   RGBA8 -> ETC2 RGB8 / RGBA8, one invocation per 4x4 destination block
//
// Every one of these is a faithful port of the CPU implementation it replaces
// (`vitaslop_runtime::pvrtc`, `render::halve_rgba8`, `render::build_mip_chain`,
// `vitaslop_runtime::etcenc`) - same arithmetic, same integer widths, same rounding. The point is
// to move the work, not to change the answer, and the ports are held to that by
// `gpu_etc2_matches_the_cpu_encoder` and `gpu_pvrtc_matches_the_cpu_decoder`.
//
// Integer only, everywhere. Not a style choice: a float port would have to argue about rounding
// on every adapter it ever runs on, and the CPU encoders these mirror are integer already.

struct Params {
    // The level being worked on.
    width: u32,
    height: u32,
    // Source block grid (PVRTC block geometry, so 4x4 or 8x4 texels).
    blocks_x: u32,
    blocks_y: u32,
    padded_x: u32,
    padded_y: u32,
    // Word (not byte) offsets into the three buffers.
    src_word: u32,
    rgba_word: u32,
    out_word: u32,
    // `halve` only: the level being filtered DOWN from.
    src_width: u32,
    src_height: u32,
    // Words per destination block row, so a compressed copy can be handed a 256-byte-aligned
    // `bytes_per_row` - which `copyBufferToTexture` requires and a packed block row is not.
    out_row_words: u32,
    // bit 0 swizzled, bit 1 PVRTC2, bit 2 4bpp, bit 3 target carries alpha (ETC2 RGBA8)
    flags: u32,
    // `decode_bc` only: the guest base format (0x85 BC1 / 0x86 BC2 / 0x87 BC3), which decides
    // both the block size and whether the colour sub-block may take its punch-through mode.
    src_format: u32,
    // Words in one SOURCE block: 2 for PVRTC and BC1, 4 for BC2/BC3.
    src_block_words: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> rgba: array<u32>;
@group(0) @binding(3) var<storage, read_write> outb: array<u32>;

/// The 4bpp modulation weights, in eighths, for codes 0..3 in the ordinary (non-punch-through)
/// mode. Named rather than written inline because WGSL cannot index an array LITERAL.
const MOD_WEIGHTS: vec4<u32> = vec4<u32>(0u, 3u, 5u, 8u);

const FLAG_SWIZZLED: u32 = 1u;
const FLAG_PVRTC2: u32 = 2u;
const FLAG_4BPP: u32 = 4u;
const FLAG_ALPHA: u32 = 8u;

fn flag(f: u32) -> bool { return (P.flags & f) != 0u; }

// ---------------------------------------------------------------------------------------------
// PVRTC decode
// ---------------------------------------------------------------------------------------------

// `morton_index` from the runtime: interleave the low bits up to the square formed by the
// smaller dimension, then append the longer axis's remaining bits linearly.
fn morton_index(x_in: u32, y_in: u32, pw: u32, ph: u32) -> u32 {
    let min_log = countTrailingZeros(min(pw, ph));
    var index = 0u;
    for (var i = 0u; i < min_log; i = i + 1u) {
        index = index | (((x_in >> i) & 1u) << (2u * i + 1u));
        index = index | (((y_in >> i) & 1u) << (2u * i));
    }
    let x = x_in >> min_log;
    let y = y_in >> min_log;
    let interleaved = 2u * min_log;
    if (pw >= ph) {
        return index | (((y * (pw >> min_log)) + x) << interleaved);
    }
    return index | (((x * (ph >> min_log)) + y) << interleaved);
}

fn block_word_offset(bx: u32, by: u32) -> u32 {
    var index: u32;
    if (flag(FLAG_SWIZZLED)) {
        index = morton_index(bx, by, P.padded_x, P.padded_y);
    } else {
        index = by * P.blocks_x + bx;
    }
    return P.src_word + index * P.src_block_words;
}

// ---------------------------------------------------------------------------------------------
// BC1 / BC2 / BC3 decode. A port of `render::decode_bc_texel` and `render::bc3_alpha`.
//
// The guest's `UBC1/2/3` ARE BC1/2/3, so on a desktop these blocks pass through untouched and
// this shader never runs. It exists for the adapter that has no BC at all - the target device -
// where the only alternatives are an RGBA8 expansion at four times the size or a CPU decode plus
// a CPU re-encode, which is the most expensive path in the whole texture pipeline.
// ---------------------------------------------------------------------------------------------

fn rgb565(c: u32) -> vec3<u32> {
    let r = (c >> 11u) & 0x1fu;
    let g = (c >> 5u) & 0x3fu;
    let b = c & 0x1fu;
    return vec3<u32>(r * 255u / 31u, g * 255u / 63u, b * 255u / 31u);
}

fn bc_byte(base: u32, i: u32) -> u32 {
    return (src[base + i / 4u] >> ((i % 4u) * 8u)) & 0xffu;
}

// The interpolated alpha of texel `t` in a BC3 block: two 8-bit endpoints and 3-bit indices
// packed little-endian across bytes 2..8.
fn bc3_alpha(base: u32, t: u32) -> u32 {
    let a0 = bc_byte(base, 0u);
    let a1 = bc_byte(base, 1u);
    let bit = t * 3u;
    let byte = 2u + bit / 8u;
    let shift = bit % 8u;
    let raw = bc_byte(base, byte) | (bc_byte(base, byte + 1u) << 8u);
    let code = (raw >> shift) & 0x7u;
    if (code == 0u) { return a0; }
    if (code == 1u) { return a1; }
    if (a0 > a1) { return ((8u - code) * a0 + (code - 1u) * a1) / 7u; }
    if (code == 6u) { return 0u; }
    if (code == 7u) { return 255u; }
    return ((6u - code) * a0 + (code - 1u) * a1) / 5u;
}

@compute @workgroup_size(8, 8, 1)
fn decode_bc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x;
    let by = gid.y;
    if (bx >= P.blocks_x || by >= P.blocks_y) { return; }
    let base = block_word_offset(bx, by);
    // The BC1 colour sub-block sits after the 8-byte alpha block for BC2/BC3.
    var color_off = 8u;
    if (P.src_format == 0x85u) { color_off = 0u; }
    let c0 = bc_byte(base, color_off) | (bc_byte(base, color_off + 1u) << 8u);
    let c1 = bc_byte(base, color_off + 2u) | (bc_byte(base, color_off + 3u) << 8u);
    let e0 = rgb565(c0);
    let e1 = rgb565(c1);
    // BC1 with c0 <= c1 selects the 3-colour + punch-through mode; BC2/BC3 colours always take
    // the 4-colour interpolation.
    let punch = P.src_format == 0x85u && c0 <= c1;

    for (var t = 0u; t < 16u; t = t + 1u) {
        let px = t % 4u;
        let py = t / 4u;
        let x = bx * 4u + px;
        let y = by * 4u + py;
        if (x >= P.width || y >= P.height) { continue; }
        let idx = (bc_byte(base, color_off + 4u + t / 4u) >> ((t % 4u) * 2u)) & 0x3u;
        var rgb = e0;
        if (idx == 1u) {
            rgb = e1;
        } else if (idx == 2u) {
            if (punch) { rgb = (e0 + e1) / vec3<u32>(2u); }
            else { rgb = (e0 * 2u + e1) / vec3<u32>(3u); }
        } else if (idx == 3u) {
            if (punch) { rgb = vec3<u32>(0u); }
            else { rgb = (e0 + e1 * 2u) / vec3<u32>(3u); }
        }
        var a = 255u;
        if (P.src_format == 0x85u) {
            if (punch && idx == 3u) { a = 0u; }
        } else if (P.src_format == 0x86u) {
            // BC2: 4-bit alpha per texel, two texels per byte, low nibble first.
            let byte = bc_byte(base, t / 2u);
            var a4 = byte & 0xfu;
            if (t % 2u == 1u) { a4 = byte >> 4u; }
            a = a4 * 255u / 15u;
        } else {
            a = bc3_alpha(base, t);
        }
        rgba[P.rgba_word + y * P.width + x] = pack_rgba(vec4<u32>(rgb, a));
    }
}

// Bit replication to 8 bits, the specification's ARGB:8888 expansion.
fn expand_bits_n(value: u32, bits: u32) -> u32 {
    let mask = (1u << bits) - 1u;
    var v = value & mask;
    var have = bits;
    loop {
        if (have >= 8u) { break; }
        v = (v << bits) | (value & mask);
        have = have + bits;
    }
    return (v >> (have - 8u)) & 0xffu;
}

struct PvrtcBlock {
    a: vec4<u32>,
    b: vec4<u32>,
    m: bool,
    h: bool,
    modulation: u32,
};

fn decode_pvrtc_block(bx: u32, by: u32) -> PvrtcBlock {
    let o = block_word_offset(bx, by);
    var blk: PvrtcBlock;
    blk.modulation = src[o];
    let c = src[o + 1u];
    blk.m = (c & 1u) != 0u;

    let op_b = (c & 0x80000000u) != 0u;
    if (op_b) {
        blk.b = vec4<u32>(
            expand_bits_n((c >> 26u) & 0x1fu, 5u),
            expand_bits_n((c >> 21u) & 0x1fu, 5u),
            expand_bits_n((c >> 16u) & 0x1fu, 5u),
            255u,
        );
    } else {
        blk.b = vec4<u32>(
            expand_bits_n((c >> 24u) & 0xfu, 4u),
            expand_bits_n((c >> 20u) & 0xfu, 4u),
            expand_bits_n((c >> 16u) & 0xfu, 4u),
            expand_bits_n((((c >> 28u) & 0x7u) << 1u) | 1u, 4u),
        );
    }

    var op_a: bool;
    if (flag(FLAG_PVRTC2)) {
        op_a = op_b;
        blk.h = (c & 0x8000u) != 0u;
    } else {
        op_a = (c & 0x8000u) != 0u;
        blk.h = false;
    }
    if (op_a) {
        blk.a = vec4<u32>(
            expand_bits_n((c >> 10u) & 0x1fu, 5u),
            expand_bits_n((c >> 5u) & 0x1fu, 5u),
            expand_bits_n((c >> 1u) & 0xfu, 4u),
            255u,
        );
    } else {
        blk.a = vec4<u32>(
            expand_bits_n((c >> 8u) & 0xfu, 4u),
            expand_bits_n((c >> 4u) & 0xfu, 4u),
            expand_bits_n((c >> 1u) & 0x7u, 3u),
            expand_bits_n(((c >> 12u) & 0x7u) << 1u, 4u),
        );
    }
    return blk;
}

// The bilinear upscale of the four surrounding blocks' A and B colours. The block area is 16 or
// 32, always a power of two, so the divide is the shift the CPU version also takes.
fn upscale(n0: PvrtcBlock, n1: PvrtcBlock, n2: PvrtcBlock, n3: PvrtcBlock,
           xr: u32, yr: u32, bw: u32, bh: u32) -> array<vec4<u32>, 2> {
    let w00 = (bw - xr) * (bh - yr);
    let w10 = xr * (bh - yr);
    let w01 = (bw - xr) * yr;
    let w11 = xr * yr;
    let sh = countTrailingZeros(bw * bh);
    let a = (n0.a * w00 + n1.a * w10 + n2.a * w01 + n3.a * w11) >> vec4<u32>(sh);
    let b = (n0.b * w00 + n1.b * w10 + n2.b * w01 + n3.b * w11) >> vec4<u32>(sh);
    return array<vec4<u32>, 2>(a & vec4<u32>(0xffu), b & vec4<u32>(0xffu));
}

fn pack_rgba(c: vec4<u32>) -> u32 {
    return (c.x & 0xffu) | ((c.y & 0xffu) << 8u) | ((c.z & 0xffu) << 16u) | ((c.w & 0xffu) << 24u);
}

fn unpack_rgba(v: u32) -> vec4<u32> {
    return vec4<u32>(v & 0xffu, (v >> 8u) & 0xffu, (v >> 16u) & 0xffu, (v >> 24u) & 0xffu);
}

// One invocation per source block, walking that block's REGION of the shifted upscale grid -
// the same region walk `pvrtc::decode_face` does, and for the same reason: inside one region the
// four surrounding blocks are fixed, so their addressing and expansion happen once for a whole
// block of texels instead of once per texel.
@compute @workgroup_size(8, 8, 1)
fn decode_pvrtc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx0 = gid.x;
    let by0 = gid.y;
    if (bx0 >= P.blocks_x || by0 >= P.blocks_y) { return; }

    var bw = 8u;
    if (flag(FLAG_4BPP)) { bw = 4u; }
    let bh = 4u;

    let bx1 = (bx0 + 1u) % P.blocks_x;
    let by1 = (by0 + 1u) % P.blocks_y;
    let n0 = decode_pvrtc_block(bx0, by0);
    let n1 = decode_pvrtc_block(bx1, by0);
    let n2 = decode_pvrtc_block(bx0, by1);
    let n3 = decode_pvrtc_block(bx1, by1);

    let shift_x = bw / 2u;
    let shift_y = bh / 2u;

    for (var yr = 0u; yr < bh; yr = yr + 1u) {
        let sy = by0 * bh + yr;
        if (sy >= P.height) { break; }
        let y = (sy + shift_y) % P.height;
        let ty = y % bh;
        for (var xr = 0u; xr < bw; xr = xr + 1u) {
            let sx = bx0 * bw + xr;
            if (sx >= P.width) { break; }
            let x = (sx + shift_x) % P.width;
            let own = decode_pvrtc_block(x / bw, y / bh);
            let hard = flag(FLAG_PVRTC2) && own.h;

            var a: vec4<u32>;
            var b: vec4<u32>;
            if (hard) {
                a = own.a;
                b = own.b;
            } else {
                let ab = upscale(n0, n1, n2, n3, xr, yr, bw, bh);
                a = ab[0];
                b = ab[1];
            }

            // Modulation, in eighths, and whether punch-through forces this texel transparent.
            var weight = 0u;
            var punched = false;
            if (flag(FLAG_4BPP)) {
                let bit = ((ty * 4u) + (x % bw)) * 2u;
                let v = (own.modulation >> bit) & 0x3u;
                if (own.m && !hard) {
                    if (v == 0u) { weight = 0u; }
                    else if (v == 1u) { weight = 4u; }
                    else if (v == 2u) { weight = 4u; punched = true; }
                    else { weight = 8u; }
                } else {
                    var w4 = MOD_WEIGHTS;
                    weight = w4[v];
                }
            } else {
                let bit = ty * 8u + (x % bw);
                weight = ((own.modulation >> bit) & 1u) * 8u;
            }

            var c = vec4<u32>(0u);
            if (!punched) {
                c = (a * (8u - weight) + b * weight) / 8u;
            }
            rgba[P.rgba_word + y * P.width + x] = pack_rgba(c);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Box filter, one destination texel per invocation. `render::halve_rgba8`'s arithmetic exactly:
// the four covered source texels, clamped on an odd dimension, `(sum + 2) / 4`.
// ---------------------------------------------------------------------------------------------

@compute @workgroup_size(8, 8, 1)
fn halve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= P.width || y >= P.height) { return; }
    let x0 = min(2u * x, P.src_width - 1u);
    let x1 = min(2u * x + 1u, P.src_width - 1u);
    let y0 = min(2u * y, P.src_height - 1u);
    let y1 = min(2u * y + 1u, P.src_height - 1u);
    let s = P.src_word;
    let a = unpack_rgba(rgba[s + y0 * P.src_width + x0]);
    let b = unpack_rgba(rgba[s + y0 * P.src_width + x1]);
    let c = unpack_rgba(rgba[s + y1 * P.src_width + x0]);
    let d = unpack_rgba(rgba[s + y1 * P.src_width + x1]);
    let sum = a + b + c + d + vec4<u32>(2u);
    rgba[P.rgba_word + y * P.width + x] = pack_rgba(sum / vec4<u32>(4u));
}

// ---------------------------------------------------------------------------------------------
// ETC2 encode. A port of `vitaslop_runtime::etcenc`, block for block.
// ---------------------------------------------------------------------------------------------

var<private> MODIFIERS: array<vec4<i32>, 8> = array<vec4<i32>, 8>(
    vec4<i32>(2, 8, -2, -8),
    vec4<i32>(5, 17, -5, -17),
    vec4<i32>(9, 29, -9, -29),
    vec4<i32>(13, 42, -13, -42),
    vec4<i32>(18, 60, -18, -60),
    vec4<i32>(24, 80, -24, -80),
    vec4<i32>(33, 106, -33, -106),
    vec4<i32>(47, 183, -47, -183),
);

// The four modifiers of a table in ASCENDING order are entries 3, 2, 0, 1.
var<private> MOD_ASC: vec4<i32> = vec4<i32>(3, 2, 0, 1);

// The block being encoded, as 16 RGBA texels. Private rather than passed, because WGSL has no
// references and copying a 16-element array into every helper is what a port like this cannot
// afford.
var<private> BLK: array<vec4<i32>, 16>;

fn clamp255(v: i32) -> i32 { return clamp(v, 0, 255); }

fn expand4(v: i32) -> i32 { return (v << 4) | v; }
fn expand5(v: i32) -> i32 { return (v << 3) | (v >> 2); }
fn expand_bits_q(q: i32, bits: u32) -> i32 {
    if (bits == 4u) { return expand4(q); }
    return expand5(q);
}

// `(quantised code, what a decoder expands it to)`.
fn quantise(v: i32, bits: u32) -> vec2<i32> {
    let maxv = (1 << bits) - 1;
    let q = clamp((v * maxv + 127) / 255, 0, maxv);
    return vec2<i32>(q, expand_bits_q(q, bits));
}

// Which subblock texel `idx` belongs to, under a given flip. Flip false splits left/right
// (columns 0-1 vs 2-3), flip true splits top/bottom.
fn subblock_of(idx: u32, flip: bool) -> u32 {
    if (flip) { return idx / 8u; }
    return (idx % 4u) / 2u;
}

// The texel index that carries selector bit position `j`. The format stores selectors in
// COLUMN-major order, so bit `j` is texel `(j % 4) * 4 + j / 4` read back the other way round.
fn index_bit(idx: u32) -> u32 {
    let x = idx % 4u;
    let y = idx / 4u;
    return x * 4u + y;
}

fn subblock_mean(flip: bool, sub: u32) -> vec3<i32> {
    var sum = vec3<i32>(0);
    var n = 0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        if (subblock_of(i, flip) == sub) {
            sum = sum + BLK[i].xyz;
            n = n + 1;
        }
    }
    return sum / vec3<i32>(n);
}

// The selectors chosen by the last `fit_subblock` call, and its error. WGSL cannot return an
// array cheaply, so the fit writes here.
var<private> FIT_SEL: array<i32, 16>;

// `fit_subblock_within`: the best selector per texel of one subblock against a base and a table,
// abandoning the fit as soon as it cannot beat `budget`. Returns the squared error, or the
// sentinel 0x7fffffff when it gave up.
fn fit_subblock(flip: bool, sub: u32, base: vec3<i32>, table: i32, budget: i32) -> i32 {
    var err = 0;
    let mods = MODIFIERS[table];

    // The luminance-residual shortcut is EXACT except where a channel clamps, because then the
    // three channels no longer move together. Detected once per fit, both extremes against both
    // ends, exactly as the CPU version does.
    let clamps = (base.x + mods[3] < 0) || (base.y + mods[3] < 0) || (base.z + mods[3] < 0)
        || (base.x + mods[1] > 255) || (base.y + mods[1] > 255) || (base.z + mods[1] > 255);

    if (!clamps) {
        let asc = vec4<i32>(mods[3], mods[2], mods[0], mods[1]);
        // Compared in sixths so nothing is divided and nothing is rounded.
        let t01 = 3 * (asc[0] + asc[1]);
        let t12 = 3 * (asc[1] + asc[2]);
        let t23 = 3 * (asc[2] + asc[3]);
        let bsum3 = base.x + base.y + base.z;
        for (var i = 0u; i < 16u; i = i + 1u) {
            if (subblock_of(i, flip) != sub) { continue; }
            let t = BLK[i].xyz;
            let r2 = 2 * (t.x + t.y + t.z - bsum3);
            var k = 0;
            if (r2 > t01) { k = k + 1; }
            if (r2 > t12) { k = k + 1; }
            if (r2 > t23) { k = k + 1; }
            let m = asc[k];
            let d = vec3<i32>(base.x + m - t.x, base.y + m - t.y, base.z + m - t.z);
            err = err + d.x * d.x + d.y * d.y + d.z * d.z;
            FIT_SEL[i] = MOD_ASC[k];
            if (err >= budget) { return 0x7fffffff; }
        }
        return err;
    }

    var palette: array<vec3<i32>, 4>;
    for (var s = 0u; s < 4u; s = s + 1u) {
        let m = mods[s];
        palette[s] = vec3<i32>(clamp255(base.x + m), clamp255(base.y + m), clamp255(base.z + m));
    }
    for (var i = 0u; i < 16u; i = i + 1u) {
        if (subblock_of(i, flip) != sub) { continue; }
        let t = BLK[i].xyz;
        var bestE = 0x7fffffff;
        var bestS = 0;
        for (var s = 0u; s < 4u; s = s + 1u) {
            let d = palette[s] - t;
            let e = d.x * d.x + d.y * d.y + d.z * d.z;
            if (e < bestE) { bestE = e; bestS = i32(s); }
        }
        err = err + bestE;
        FIT_SEL[i] = bestS;
        if (err >= budget) { return 0x7fffffff; }
    }
    return err;
}

// The base a subblock would ideally have GIVEN a table and the selectors it chose: the mean of
// each texel minus the modifier assigned to it.
fn ideal_base(flip: bool, sub: u32, table: i32, sel: ptr<function, array<i32, 16>>) -> vec3<i32> {
    var sum = vec3<i32>(0);
    var n = 0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        if (subblock_of(i, flip) != sub) { continue; }
        let m = MODIFIERS[table][(*sel)[i]];
        sum = sum + BLK[i].xyz - vec3<i32>(m);
        n = n + 1;
    }
    return sum / vec3<i32>(n);
}

// Rank the eight tables for a subblock by a LUMINANCE proxy (`r + 2g + b`, so residuals are in
// quarter-units and the modifiers scale by four). Returns the table indices, best first.
fn rank_tables(flip: bool, sub: u32, base: vec3<i32>) -> array<i32, 8> {
    let base_luma = base.x + base.y * 2 + base.z;
    var residuals: array<i32, 16>;
    var n_res = 0u;
    for (var i = 0u; i < 16u; i = i + 1u) {
        if (subblock_of(i, flip) == sub) {
            let t = BLK[i].xyz;
            residuals[n_res] = (t.x + t.y * 2 + t.z) - base_luma;
            n_res = n_res + 1u;
        }
    }
    var score: array<i32, 8>;
    var order: array<i32, 8>;
    for (var table = 0u; table < 8u; table = table + 1u) {
        var s = 0;
        for (var r = 0u; r < n_res; r = r + 1u) {
            var nearest = 0x7fffffff;
            for (var k = 0u; k < 4u; k = k + 1u) {
                let d = abs(residuals[r] - MODIFIERS[table][k] * 4);
                if (d < nearest) { nearest = d; }
            }
            s = s + nearest * nearest;
        }
        score[table] = s;
        order[table] = i32(table);
    }
    // Insertion sort by (score, table) - the CPU sorts the same pair, so ties break the same way.
    for (var i = 1u; i < 8u; i = i + 1u) {
        let sv = score[i];
        let ov = order[i];
        var j = i32(i) - 1;
        loop {
            if (j < 0) { break; }
            if (score[j] < sv || (score[j] == sv && order[j] <= ov)) { break; }
            score[j + 1] = score[j];
            order[j + 1] = order[j];
            j = j - 1;
        }
        score[j + 1] = sv;
        order[j + 1] = ov;
    }
    return order;
}

const TABLES_FITTED: u32 = 3u;

// The winner of `search_subblock`, written here for the same reason `FIT_SEL` exists.
var<private> SEARCH_SEL: array<i32, 16>;
var<private> SEARCH_CODE: vec3<i32>;
var<private> SEARCH_TABLE: i32;

fn quant_all(v: vec3<i32>, bits: u32) -> array<vec3<i32>, 2> {
    let r = quantise(v.x, bits);
    let g = quantise(v.y, bits);
    let b = quantise(v.z, bits);
    return array<vec3<i32>, 2>(vec3<i32>(r.x, g.x, b.x), vec3<i32>(r.y, g.y, b.y));
}

fn search_subblock(flip: bool, sub: u32, mean: vec3<i32>, bits: u32) -> i32 {
    let start = quant_all(mean, bits);
    let start_code = start[0];
    let start_actual = start[1];
    let ranked = rank_tables(flip, sub, start_actual);

    var best_err = 0x7fffffff;
    SEARCH_CODE = start_code;
    SEARCH_TABLE = ranked[0];
    // PASS 1: pick the table.
    for (var i = 0u; i < TABLES_FITTED; i = i + 1u) {
        let table = ranked[i];
        let e = fit_subblock(flip, sub, start_actual, table, best_err);
        if (e < best_err) {
            best_err = e;
            SEARCH_CODE = start_code;
            SEARCH_TABLE = table;
            SEARCH_SEL = FIT_SEL;
        }
    }
    if (best_err == 0) { return 0; }

    // PASS 2: refine the BASE for the winning table only.
    let table = SEARCH_TABLE;
    var actual = start_actual;
    for (var refit = 0u; refit < 2u; refit = refit + 1u) {
        var sel = SEARCH_SEL;
        let want = ideal_base(flip, sub, table, &sel);
        let q = quant_all(want, bits);
        if (all(q[1] == actual)) { break; }
        let e2 = fit_subblock(flip, sub, q[1], table, best_err);
        if (e2 >= best_err) { break; }
        best_err = e2;
        SEARCH_CODE = q[0];
        SEARCH_SEL = FIT_SEL;
        actual = q[1];
    }

    // PASS 3: one neighbourhood probe against the winning table. Rounding to the nearest code is
    // not the same as choosing the best one.
    var sel3 = SEARCH_SEL;
    let want = ideal_base(flip, sub, table, &sel3);
    let maxv = (1 << bits) - 1;
    var per: array<vec2<i32>, 3>;
    var count: array<u32, 3>;
    for (var c = 0u; c < 3u; c = c + 1u) {
        var w = want.x;
        if (c == 1u) { w = want.y; }
        if (c == 2u) { w = want.z; }
        let scaled = clamp(w, 0, 255) * maxv;
        let lo = clamp(scaled / 255, 0, maxv);
        let hi = clamp((scaled + 254) / 255, 0, maxv);
        per[c] = vec2<i32>(lo, hi);
        if (hi != lo) { count[c] = 2u; } else { count[c] = 1u; }
    }
    for (var r = 0u; r < count[0]; r = r + 1u) {
        for (var g = 0u; g < count[1]; g = g + 1u) {
            for (var b = 0u; b < count[2]; b = b + 1u) {
                let code = vec3<i32>(per[0][r], per[1][g], per[2][b]);
                let cand = vec3<i32>(
                    expand_bits_q(code.x, bits),
                    expand_bits_q(code.y, bits),
                    expand_bits_q(code.z, bits),
                );
                if (all(cand == actual)) { continue; }
                let e = fit_subblock(flip, sub, cand, table, best_err);
                if (e < best_err) {
                    best_err = e;
                    SEARCH_CODE = code;
                    SEARCH_SEL = FIT_SEL;
                }
            }
        }
    }
    return best_err;
}

// The best candidate found so far, in the form `pack_rgb8` needs.
struct Candidate {
    err: i32,
    flip: bool,
    diff: bool,
    stored0: vec3<i32>,
    stored1: vec3<i32>,
    table0: i32,
    table1: i32,
};
var<private> CAND_SEL: array<i32, 16>;
var<private> BEST_SEL: array<i32, 16>;

fn sat_add(a: i32, b: i32) -> i32 {
    if (a >= 0x7fffffff - b) { return 0x7fffffff; }
    return a + b;
}

// INDIVIDUAL mode: the two subblocks are independent, so each is refined on its own.
fn best_individual(flip: bool) -> Candidate {
    var c: Candidate;
    c.flip = flip;
    c.diff = false;
    var total = 0;
    for (var sub = 0u; sub < 2u; sub = sub + 1u) {
        let mean = subblock_mean(flip, sub);
        let err = search_subblock(flip, sub, mean, 4u);
        total = sat_add(total, err);
        if (sub == 0u) { c.stored0 = SEARCH_CODE; c.table0 = SEARCH_TABLE; }
        else { c.stored1 = SEARCH_CODE; c.table1 = SEARCH_TABLE; }
        for (var i = 0u; i < 16u; i = i + 1u) {
            if (subblock_of(i, flip) == sub) { CAND_SEL[i] = SEARCH_SEL[i]; }
        }
    }
    c.err = total;
    return c;
}

// DIFFERENTIAL mode: the second base is reachable only as a -4..3 offset from the first, per
// channel, so the two subblocks are coupled and cannot be refined independently.
fn best_differential(flip: bool) -> Candidate {
    var c: Candidate;
    c.flip = flip;
    c.diff = true;
    let mean0 = subblock_mean(flip, 0u);
    let mean1 = subblock_mean(flip, 1u);
    let err0 = search_subblock(flip, 0u, mean0, 5u);
    let code0 = SEARCH_CODE;
    c.stored0 = code0;
    c.table0 = SEARCH_TABLE;
    var sel0 = SEARCH_SEL;

    // The nearest base reachable from `code0`, and what a decoder expands it to.
    var reach_code: vec3<i32>;
    var reach_actual: vec3<i32>;
    // (written by `reachable`, which WGSL makes a pair of globals rather than a tuple return)
    var want = mean1;

    var best1 = 0x7fffffff;
    var best1_code = vec3<i32>(0);
    var best1_table = 0;
    var best1_sel: array<i32, 16>;

    // `reachable(mean1)` once, to rank tables against.
    for (var c3 = 0u; c3 < 3u; c3 = c3 + 1u) {
        let tgt = quantise(want[c3], 5u).x;
        let d = clamp(tgt - code0[c3], -4, 3);
        let q = clamp(code0[c3] + d, 0, 31);
        reach_code[c3] = (q - code0[c3]) & 0x07;
        reach_actual[c3] = expand5(q);
    }
    let ranked1 = rank_tables(flip, 1u, reach_actual);

    for (var ti = 0u; ti < TABLES_FITTED; ti = ti + 1u) {
        let table = ranked1[ti];
        var code = reach_code;
        var actual = reach_actual;
        var e = fit_subblock(flip, 1u, actual, table, 0x7fffffff);
        var s = FIT_SEL;
        for (var refit = 0u; refit < 2u; refit = refit + 1u) {
            var sc = s;
            let w2 = ideal_base(flip, 1u, table, &sc);
            var c2: vec3<i32>;
            var a2: vec3<i32>;
            for (var ch = 0u; ch < 3u; ch = ch + 1u) {
                let tgt = quantise(w2[ch], 5u).x;
                let d = clamp(tgt - code0[ch], -4, 3);
                let q = clamp(code0[ch] + d, 0, 31);
                c2[ch] = (q - code0[ch]) & 0x07;
                a2[ch] = expand5(q);
            }
            if (all(a2 == actual)) { break; }
            let e2 = fit_subblock(flip, 1u, a2, table, 0x7fffffff);
            if (e2 >= e) { break; }
            e = e2;
            s = FIT_SEL;
            code = c2;
            actual = a2;
        }
        if (e < best1) {
            best1 = e;
            best1_code = code;
            best1_table = table;
            best1_sel = s;
        }
    }

    c.stored1 = best1_code;
    c.table1 = best1_table;
    c.err = sat_add(err0, best1);
    for (var i = 0u; i < 16u; i = i + 1u) {
        if (subblock_of(i, flip) == 0u) { CAND_SEL[i] = sel0[i]; }
        else { CAND_SEL[i] = best1_sel[i]; }
    }
    return c;
}

// Whether DIFFERENTIAL mode can express both subblocks' preferred bases under this flip. A bound
// on what the modes CAN represent, so it can only skip a search that was going to lose.
fn differential_can_reach(flip: bool) -> bool {
    let m0 = subblock_mean(flip, 0u);
    let m1 = subblock_mean(flip, 1u);
    for (var c = 0u; c < 3u; c = c + 1u) {
        let q0 = quantise(m0[c], 5u).x;
        let q1 = quantise(m1[c], 5u).x;
        let d = q1 - q0;
        if (d < -4 || d > 3) { return false; }
    }
    return true;
}

fn pack_rgb8(c: Candidate, sel: ptr<function, array<i32, 16>>) -> vec2<u32> {
    var b: array<u32, 8>;
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        if (c.diff) {
            b[ch] = u32((c.stored0[ch] << 3) | (c.stored1[ch] & 0x07)) & 0xffu;
        } else {
            b[ch] = u32((c.stored0[ch] << 4) | (c.stored1[ch] & 0x0f)) & 0xffu;
        }
    }
    var t = ((u32(c.table0) & 7u) << 5u) | ((u32(c.table1) & 7u) << 2u);
    if (c.diff) { t = t | 2u; }
    if (c.flip) { t = t | 1u; }
    b[3] = t;
    var msb = 0u;
    var lsb = 0u;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let bit = index_bit(i);
        let s = u32((*sel)[i]);
        msb = msb | (((s >> 1u) & 1u) << bit);
        lsb = lsb | ((s & 1u) << bit);
    }
    b[4] = (msb >> 8u) & 0xffu;
    b[5] = msb & 0xffu;
    b[6] = (lsb >> 8u) & 0xffu;
    b[7] = lsb & 0xffu;
    return vec2<u32>(
        b[0] | (b[1] << 8u) | (b[2] << 16u) | (b[3] << 24u),
        b[4] | (b[5] << 8u) | (b[6] << 16u) | (b[7] << 24u),
    );
}

fn encode_etc2_rgb8_block() -> vec2<u32> {
    // A block of ONE colour is the most common block in real art, and differential mode with a
    // zero offset encodes it directly.
    var flat = true;
    for (var i = 1u; i < 16u; i = i + 1u) {
        if (any(BLK[i].xyz != BLK[0].xyz)) { flat = false; break; }
    }
    if (flat) {
        let want = BLK[0].xyz;
        var bestE = 0x7fffffff;
        var bestCode = vec3<i32>(0);
        var bestTable = 0;
        var bestSel = 0;
        var start: vec3<i32>;
        for (var c = 0u; c < 3u; c = c + 1u) { start[c] = quantise(want[c], 5u).x; }
        for (var table = 0; table < 8; table = table + 1) {
            for (var sel = 0u; sel < 4u; sel = sel + 1u) {
                let m = MODIFIERS[table][sel];
                var code = vec3<i32>(0);
                var err = 0;
                for (var c = 0u; c < 3u; c = c + 1u) {
                    var bd = 0x7fffffff;
                    var bq = clamp(start[c], 0, 31);
                    for (var q = start[c] - 1; q <= start[c] + 1; q = q + 1) {
                        if (q < 0 || q > 31) { continue; }
                        let v = clamp255(expand5(q) + m);
                        let d = (v - want[c]) * (v - want[c]);
                        if (d < bd) { bd = d; bq = q; }
                    }
                    err = err + bd;
                    code[c] = bq;
                }
                if (err < bestE) { bestE = err; bestCode = code; bestTable = table; bestSel = i32(sel); }
            }
        }
        var c: Candidate;
        c.err = bestE;
        c.flip = false;
        c.diff = true;
        c.stored0 = bestCode;
        c.stored1 = vec3<i32>(0);
        c.table0 = bestTable;
        c.table1 = bestTable;
        var sel: array<i32, 16>;
        for (var i = 0u; i < 16u; i = i + 1u) { sel[i] = bestSel; }
        return pack_rgb8(c, &sel);
    }

    var best: Candidate;
    let reach0 = differential_can_reach(false);
    let reach1 = differential_can_reach(true);
    // >>> WRITTEN OUT RATHER THAN LOOPED, AND THAT IS NOT A STYLE CHOICE.
    //
    // Each of these calls leaves its selectors in `CAND_SEL`, so the winner's selectors have to
    // be taken before the next call overwrites them. Expressed as a loop over the two flips, the
    // capture and the comparison sit in the same `if` and the whole thing reads as though the
    // candidate carried its own selectors - which it does not, because WGSL has no way to return
    // a 16-element array cheaply. Four straight-line steps make the ordering the code's shape
    // rather than something a reader has to hold in their head.
    best = best_differential(false);
    BEST_SEL = CAND_SEL;
    let d1 = best_differential(true);
    if (d1.err < best.err) {
        best = d1;
        BEST_SEL = CAND_SEL;
    }
    // Individual mode is searched only where differential CANNOT reach the second base, which is
    // the only thing individual's coarser 4-bit bases can buy. Exact already means nothing can
    // improve on it.
    if (best.err != 0 && !reach0) {
        let i0 = best_individual(false);
        if (i0.err < best.err) {
            best = i0;
            BEST_SEL = CAND_SEL;
        }
    }
    if (best.err != 0 && !reach1) {
        let i1 = best_individual(true);
        if (i1.err < best.err) {
            best = i1;
            BEST_SEL = CAND_SEL;
        }
    }
    var sel = BEST_SEL;
    return pack_rgb8(best, &sel);
}

// ---------------------------------------------------------------------------------------------
// EAC alpha
// ---------------------------------------------------------------------------------------------

var<private> EAC_TABLES: array<array<i32, 8>, 16> = array<array<i32, 8>, 16>(
    array<i32, 8>(-3, -6, -9, -15, 2, 5, 8, 14),
    array<i32, 8>(-3, -7, -10, -13, 2, 6, 9, 12),
    array<i32, 8>(-2, -5, -8, -13, 1, 4, 7, 12),
    array<i32, 8>(-2, -4, -6, -13, 1, 3, 5, 12),
    array<i32, 8>(-3, -6, -8, -12, 2, 5, 7, 11),
    array<i32, 8>(-3, -7, -9, -11, 2, 6, 8, 10),
    array<i32, 8>(-4, -7, -8, -11, 3, 6, 7, 10),
    array<i32, 8>(-3, -5, -8, -11, 2, 4, 7, 10),
    array<i32, 8>(-2, -6, -8, -10, 1, 5, 7, 9),
    array<i32, 8>(-2, -5, -8, -10, 1, 4, 7, 9),
    array<i32, 8>(-2, -4, -8, -10, 1, 3, 7, 9),
    array<i32, 8>(-2, -5, -7, -10, 1, 4, 6, 9),
    array<i32, 8>(-3, -4, -7, -10, 2, 3, 6, 9),
    array<i32, 8>(-1, -2, -3, -10, 0, 1, 2, 9),
    array<i32, 8>(-4, -6, -8, -9, 3, 5, 7, 8),
    array<i32, 8>(-3, -5, -7, -9, 2, 4, 6, 8),
);

// Every table above lists its four negative modifiers first, most negative LAST among them, then
// its four positive ones ascending. So ascending order is 3, 2, 1, 0, 4, 5, 6, 7 for all sixteen -
// derived from the tables by the CPU and asserted there; written out here because WGSL has no
// const evaluation that could rebuild it, and checked by `gpu_eac_order_matches_the_cpu`.
var<private> EAC_ORDER: array<i32, 8> = array<i32, 8>(3, 2, 1, 0, 4, 5, 6, 7);

const EAC_ZERO_TABLE: i32 = 13;
const EAC_ZERO_SEL: i32 = 4;
const EAC_TABLES_REFINED: u32 = 4u;

fn eac_extent(table: i32) -> vec2<i32> {
    return vec2<i32>(EAC_TABLES[table][EAC_ORDER[0]], EAC_TABLES[table][EAC_ORDER[7]]);
}

var<private> EAC_LV: array<i32, 8>;

fn eac_levels(table: i32, mult: i32, base: i32) {
    for (var k = 0u; k < 8u; k = k + 1u) {
        EAC_LV[k] = clamp255(base + EAC_TABLES[table][EAC_ORDER[k]] * mult);
    }
}

// The index of the ascending level nearest `a`, by bisection over eight values.
fn eac_nearest(a: i32) -> u32 {
    var k = 0u;
    var n = 8u;
    loop {
        if (n <= 1u) { break; }
        let half = n / 2u;
        if (EAC_LV[k + half - 1u] < a) { k = k + half; }
        n = n - half;
    }
    if (k > 0u && (a - EAC_LV[k - 1u]) <= (EAC_LV[k] - a)) { return k - 1u; }
    return k;
}

var<private> EAC_SEL: array<i32, 16>;

fn fit_eac(table: i32, mult: i32, base: i32) -> i32 {
    eac_levels(table, mult, base);
    var err = 0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let a = BLK[i].w;
        let bk = eac_nearest(a);
        let d = EAC_LV[bk] - a;
        err = sat_add(err, d * d);
        EAC_SEL[i] = EAC_ORDER[bk];
    }
    return err;
}

fn pack_eac(base: i32, mult: i32, table: i32, sel: ptr<function, array<i32, 16>>) -> vec2<u32> {
    // The 48 selector bits, as two 24-bit halves so no field ever straddles a word: bit position
    // `45 - 3j` for texel bit `j`, and every such position is a multiple of 3, so a field lies
    // wholly in `hi` (bits 24..47) or wholly in `lo` (bits 0..23).
    var hi = 0u;
    var lo = 0u;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let j = index_bit(i);
        let s = u32((*sel)[i]) & 7u;
        let shift = 45u - 3u * j;
        if (shift >= 24u) { hi = hi | (s << (shift - 24u)); }
        else { lo = lo | (s << shift); }
    }
    // Bytes 2..8, big-endian: byte i covers bits 40-8i .. 47-8i.
    var b: array<u32, 8>;
    b[0] = u32(base) & 0xffu;
    b[1] = ((u32(mult) & 0xfu) << 4u) | (u32(table) & 0x0fu);
    for (var i = 0u; i < 6u; i = i + 1u) {
        let shift = 40u - 8u * i;
        var v: u32;
        if (shift >= 24u) { v = hi >> (shift - 24u); }
        else { v = (lo >> shift) | (hi << (24u - shift)); }
        b[2u + i] = v & 0xffu;
    }
    return vec2<u32>(
        b[0] | (b[1] << 8u) | (b[2] << 16u) | (b[3] << 24u),
        b[4] | (b[5] << 8u) | (b[6] << 16u) | (b[7] << 24u),
    );
}

fn encode_eac_alpha_block() -> vec2<u32> {
    var sorted: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { sorted[i] = BLK[i].w; }
    for (var i = 1u; i < 16u; i = i + 1u) {
        let v = sorted[i];
        var j = i32(i) - 1;
        loop {
            if (j < 0 || sorted[j] <= v) { break; }
            sorted[j + 1] = sorted[j];
            j = j - 1;
        }
        sorted[j + 1] = v;
    }
    let lo = sorted[0];
    let hi = sorted[15];

    // Constant alpha is exact through the zero modifier, and it is most of real art.
    if (lo == hi) {
        var sel: array<i32, 16>;
        for (var i = 0u; i < 16u; i = i + 1u) { sel[i] = EAC_ZERO_SEL; }
        return pack_eac(lo, 1, EAC_ZERO_TABLE, &sel);
    }
    let spread = hi - lo;
    let samples = vec4<i32>(sorted[0], sorted[5], sorted[10], sorted[15]);

    var score: array<i32, 16>;
    var order: array<i32, 16>;
    for (var table = 0; table < 16; table = table + 1) {
        let ex = eac_extent(table);
        let span = ex.y - ex.x;
        let mult = clamp((spread + span / 2) / span, 1, 15);
        let b0 = clamp(lo - ex.x * mult, 0, 255);
        let b1 = clamp(hi - ex.y * mult, 0, 255);
        var tbest = 0x7fffffff;
        for (var bi = 0u; bi < 3u; bi = bi + 1u) {
            var base = b0;
            if (bi == 1u) { base = b1; }
            if (bi == 2u) { base = (b0 + b1) / 2; }
            eac_levels(table, mult, base);
            var e = 0;
            for (var k = 0u; k < 4u; k = k + 1u) {
                let a = samples[k];
                let d = EAC_LV[eac_nearest(a)] - a;
                e = sat_add(e, d * d);
            }
            if (e < tbest) { tbest = e; }
        }
        score[table] = tbest;
        order[table] = table;
    }
    for (var i = 1u; i < 16u; i = i + 1u) {
        let sv = score[i];
        let ov = order[i];
        var j = i32(i) - 1;
        loop {
            if (j < 0) { break; }
            if (score[j] < sv || (score[j] == sv && order[j] <= ov)) { break; }
            score[j + 1] = score[j];
            order[j + 1] = order[j];
            j = j - 1;
        }
        score[j + 1] = sv;
        order[j + 1] = ov;
    }

    var bestE = 0x7fffffff;
    var bestBase = 0;
    var bestMult = 1;
    var bestTable = EAC_ZERO_TABLE;
    var bestSel: array<i32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) { bestSel[i] = EAC_ZERO_SEL; }

    for (var ti = 0u; ti < EAC_TABLES_REFINED; ti = ti + 1u) {
        let table = order[ti];
        let ex = eac_extent(table);
        let span = ex.y - ex.x;
        let m0 = clamp((spread + span / 2) / span, 1, 15);
        for (var mi = 0u; mi < 3u; mi = mi + 1u) {
            var mult = m0;
            if (mi == 1u) { mult = max(m0 - 1, 1); }
            if (mi == 2u) { mult = min(m0 + 1, 15); }
            let b0 = clamp(lo - ex.x * mult, 0, 255);
            let b1 = clamp(hi - ex.y * mult, 0, 255);
            for (var bi = 0u; bi < 3u; bi = bi + 1u) {
                var base = b0;
                if (bi == 1u) { base = b1; }
                if (bi == 2u) { base = (b0 + b1) / 2; }
                let e = fit_eac(table, mult, base);
                if (e < bestE) {
                    bestE = e;
                    bestBase = base;
                    bestMult = mult;
                    bestTable = table;
                    bestSel = EAC_SEL;
                }
                if (e == 0) {
                    var s = EAC_SEL;
                    return pack_eac(base, mult, table, &s);
                }
            }
        }
    }
    return pack_eac(bestBase, bestMult, bestTable, &bestSel);
}

// One invocation per 4x4 destination block. Edge blocks of a non-multiple-of-4 level clamp to
// the last texel, which is the rule `etcenc::gather` and `bcenc` both use.
@compute @workgroup_size(8, 8, 1)
fn encode_etc2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bx = gid.x;
    let by = gid.y;
    let bw = (P.width + 3u) / 4u;
    let bh = (P.height + 3u) / 4u;
    if (bx >= bw || by >= bh) { return; }

    for (var i = 0u; i < 16u; i = i + 1u) {
        let x = min(bx * 4u + (i % 4u), P.width - 1u);
        let y = min(by * 4u + (i / 4u), P.height - 1u);
        let c = unpack_rgba(rgba[P.rgba_word + y * P.width + x]);
        BLK[i] = vec4<i32>(i32(c.x), i32(c.y), i32(c.z), i32(c.w));
    }

    // Rows are padded so `copyBufferToTexture` gets a 256-byte-aligned `bytes_per_row`.
    if (flag(FLAG_ALPHA)) {
        let a = encode_eac_alpha_block();
        let c = encode_etc2_rgb8_block();
        let o = P.out_word + by * P.out_row_words + bx * 4u;
        outb[o] = a.x;
        outb[o + 1u] = a.y;
        outb[o + 2u] = c.x;
        outb[o + 3u] = c.y;
    } else {
        let c = encode_etc2_rgb8_block();
        let o = P.out_word + by * P.out_row_words + bx * 2u;
        outb[o] = c.x;
        outb[o + 1u] = c.y;
    }
}
