//! A software rasterizer over the captured GXM stream: the first, blob-free way
//! to turn a recorded scene into pixels. It is a fixed-function equivalent of the
//! cube's (placeholder) shaders - transform each vertex position by the captured
//! MVP uniform, interpolate the per-vertex color, depth-test - which is exactly
//! what the real vertex/fragment programs would do. No Sony shader blob needed.
//!
//! This is the CPU reference. A wgpu backend over the same capture comes later;
//! keeping this pure and engine-agnostic makes it the oracle for that.

use crate::capture::{BoundTexture, Draw, Scene};

/// An RGBA8 image.
pub struct Framebuffer {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl Framebuffer {
    fn new(width: u32, height: u32, clear: [u8; 4]) -> Self {
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..width * height {
            rgba.extend_from_slice(&clear);
        }
        Framebuffer { width, height, rgba }
    }

    /// Count of pixels not equal to `clear` (how much got drawn).
    pub fn drawn_pixels(&self, clear: [u8; 4]) -> usize {
        self.rgba.chunks(4).filter(|p| *p != clear).count()
    }

    /// The RGBA color at `(x, y)`.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }

    /// Encode as a PNG (8-bit RGBA). Self-contained: uncompressed DEFLATE (stored
    /// blocks) so there is no compression dependency. Fine for reference dumps.
    pub fn to_png(&self) -> Vec<u8> {
        let mut raw = Vec::with_capacity((self.width * self.height * 4 + self.height) as usize);
        for y in 0..self.height {
            raw.push(0); // filter: None
            let row = (y * self.width * 4) as usize;
            raw.extend_from_slice(&self.rgba[row..row + (self.width * 4) as usize]);
        }

        let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&self.width.to_be_bytes());
        ihdr.extend_from_slice(&self.height.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, RGBA, deflate, no filter, no interlace
        write_chunk(&mut png, b"IHDR", &ihdr);
        write_chunk(&mut png, b"IDAT", &zlib_stored(&raw));
        write_chunk(&mut png, b"IEND", &[]);
        png
    }
}

/// Append a PNG chunk (length, type, data, CRC32).
fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc = Crc32::new();
    crc.update(kind);
    crc.update(data);
    out.extend_from_slice(&crc.finish().to_be_bytes());
}

/// Wrap `data` in a zlib stream using only stored (uncompressed) DEFLATE blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // zlib header, no compression
    let mut i = 0;
    while i < data.len() || data.is_empty() {
        let n = (data.len() - i).min(0xFFFF);
        let final_block = i + n >= data.len();
        out.push(if final_block { 1 } else { 0 });
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&(!(n as u16)).to_le_bytes());
        out.extend_from_slice(&data[i..i + n]);
        i += n;
        if final_block {
            break;
        }
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

/// Adler-32 checksum (zlib trailer).
fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// CRC-32 (PNG chunk checksum).
struct Crc32 {
    value: u32,
}

impl Crc32 {
    fn new() -> Self {
        Crc32 { value: 0xFFFF_FFFF }
    }
    fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.value ^= byte as u32;
            for _ in 0..8 {
                let mask = (self.value & 1).wrapping_neg();
                self.value = (self.value >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
    }
    fn finish(self) -> u32 {
        !self.value
    }
}

/// GXM vertex attribute formats we decode (`SceGxmAttributeFormat`).
const FORMAT_U8N: u8 = 4;
const FORMAT_F32: u8 = 9;

/// A vertex pulled out of the stream in its native form: the raw position lanes
/// (object space for the 3D path, screen pixels or NDC for the 2D path), plus the
/// texcoord and per-vertex color the fragment stage needs. Projection to the screen
/// happens per draw in [`render_scene`] according to the draw's [`Space`].
struct Vertex {
    pos: [f32; 3],
    uv: [f32; 2],
    color: [u8; 4],
}

/// The coordinate space a draw's vertex positions live in - i.e. what the guest's
/// (unavailable) vertex program would have done. We recover the intent from the
/// vertex layout, which is the premise of this capture-based fixed-function
/// renderer:
/// - `Mvp`: object space transformed by a captured 4x4 MVP uniform (the 3D cube).
/// - `Ndc`: clip coordinates emitted directly, in [-1, 1] (a fullscreen pass).
/// - `Pixel`: screen pixels in [0, surface] with Y down (2D sprite quads, whose
///   vertex program bakes the pixel-to-clip transform for the render target).
enum Space {
    Mvp([f32; 16]),
    Ndc,
    Pixel,
}

/// How a draw's interleaved vertex maps to position / texcoord / color, resolved
/// once per draw from its GXM attribute list. Position is the lowest-offset float
/// attribute; a second float2 attribute is the texcoord; a normalized-u8 attribute
/// is the color.
struct Layout {
    pos_off: usize,
    pos_comps: usize,
    uv_off: Option<usize>,
    color_off: Option<usize>,
}

fn layout_of(d: &Draw) -> Layout {
    // Float attributes, sorted by byte offset: the first is position, a second
    // float2 is the texcoord (the near-universal 2D sprite layout pos.xy, uv.xy).
    let mut floats: Vec<(usize, usize)> = d
        .attributes
        .iter()
        .filter(|a| a.format == FORMAT_F32 && a.component_count >= 2)
        .map(|a| (a.offset as usize, a.component_count as usize))
        .collect();
    floats.sort_unstable();
    let (pos_off, pos_comps) = floats.first().copied().unwrap_or((0, 3));
    let uv_off = floats.get(1).map(|(o, _)| *o);
    let color_off = d
        .attributes
        .iter()
        .find(|a| a.format == FORMAT_U8N && a.component_count >= 3)
        .map(|a| a.offset as usize);
    Layout { pos_off, pos_comps, uv_off, color_off }
}

/// Multiply column-major 4x4 `m` by the column vector `(x, y, z, 1)`.
fn transform(m: &[f32; 16], x: f32, y: f32, z: f32) -> [f32; 4] {
    [
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
        m[3] * x + m[7] * y + m[11] * z + m[15],
    ]
}

/// Decode vertex `i` from a draw's interleaved buffer into raw lanes (position,
/// texcoord, color) per the resolved [`Layout`]. Projection is deferred to the
/// caller, which knows the draw's [`Space`].
fn decode_vertex(d: &Draw, layout: &Layout, i: usize) -> Vertex {
    let stride = d.vertex_stride.max(1) as usize;
    let base = i * stride;
    let f = |off: usize| -> f32 {
        let o = base + off;
        if o + 4 <= d.vertices.len() {
            f32::from_le_bytes([d.vertices[o], d.vertices[o + 1], d.vertices[o + 2], d.vertices[o + 3]])
        } else {
            0.0
        }
    };
    let px = f(layout.pos_off);
    let py = f(layout.pos_off + 4);
    let pz = if layout.pos_comps >= 3 { f(layout.pos_off + 8) } else { 0.0 };
    let uv = match layout.uv_off {
        Some(o) => [f(o), f(o + 4)],
        None => [0.0, 0.0],
    };
    let color = match layout.color_off {
        Some(o) => {
            let c = base + o;
            if c + 4 <= d.vertices.len() {
                [d.vertices[c], d.vertices[c + 1], d.vertices[c + 2], d.vertices[c + 3]]
            } else {
                [255, 255, 255, 255]
            }
        }
        None => [255, 255, 255, 255],
    };
    Vertex { pos: [px, py, pz], uv, color }
}

/// Project a vertex's raw position into screen space for the given draw `space`,
/// returning `[screen_x, screen_y, depth, 1/w]`, or `None` if the vertex is behind
/// the eye (perspective `w <= 0`) and the triangle must be dropped.
fn project(v: &Vertex, space: &Space, width: u32, height: u32) -> Option<[f32; 4]> {
    let (wf, hf) = (width as f32, height as f32);
    match space {
        Space::Mvp(m) => {
            let c = transform(m, v.pos[0], v.pos[1], v.pos[2]);
            if c[3] <= 0.0 {
                return None;
            }
            let inv_w = 1.0 / c[3];
            let sx = (c[0] * inv_w * 0.5 + 0.5) * wf;
            // Flip Y: NDC +Y is up, image +Y is down.
            let sy = (1.0 - (c[1] * inv_w * 0.5 + 0.5)) * hf;
            Some([sx, sy, c[2] * inv_w, inv_w])
        }
        Space::Ndc => {
            let sx = (v.pos[0] * 0.5 + 0.5) * wf;
            let sy = (1.0 - (v.pos[1] * 0.5 + 0.5)) * hf;
            Some([sx, sy, 0.0, 1.0])
        }
        // Screen pixels already, Y down - straight through.
        Space::Pixel => Some([v.pos[0], v.pos[1], 0.0, 1.0]),
    }
}

/// Read index `i` from a draw (U16 or U32 index buffer).
fn index_at(d: &Draw, i: usize) -> usize {
    if d.index_format == 0 {
        let o = i * 2;
        if o + 2 <= d.indices.len() {
            return u16::from_le_bytes([d.indices[o], d.indices[o + 1]]) as usize;
        }
    } else {
        let o = i * 4;
        if o + 4 <= d.indices.len() {
            return u32::from_le_bytes([d.indices[o], d.indices[o + 1], d.indices[o + 2], d.indices[o + 3]]) as usize;
        }
    }
    0
}

/// Texel block geometry for a `SceGxmTextureBaseFormat` high byte:
/// `(block_width, block_height, bytes_per_block)`. Uncompressed formats are 1x1
/// texel blocks; the BC/DXT family is 4x4 blocks of 8 bytes (BC1/BC4) or 16 bytes
/// (BC2/BC3/BC5). Shared with the host-side snapshot so both agree on the byte
/// layout; `None` for a format whose size we do not know.
pub fn block_layout(base_format: u32) -> Option<(u32, u32, u32)> {
    Some(match base_format {
        // 8-bit single channel (U8/S8) and 8-bit paletted (P8).
        0x00 | 0x01 | 0x95 => (1, 1, 1),
        // 16-bit packed (U4U4U4U4, U1U5U5U5, U5U6U5, ...).
        0x02..=0x0b => (1, 1, 2),
        // 24-bit three-channel (U8U8U8, S8S8S8).
        0x98 | 0x99 => (1, 1, 3),
        // 32-bit (U8U8U8U8, ..., F32) and 32-bit single (U32/S32).
        0x0c..=0x1a => (1, 1, 4),
        // BC1 (DXT1) and BC4: 8-byte 4x4 blocks.
        0x85 | 0x88 => (4, 4, 8),
        // BC2 (DXT3), BC3 (DXT5), BC5: 16-byte 4x4 blocks.
        0x86 | 0x87 | 0x8a => (4, 4, 16),
        _ => return None,
    })
}

/// Whether a `SceGxmTextureType` selector uses the GPU's Morton (Z-order) swizzled
/// memory layout: `SCE_GXM_TEXTURE_SWIZZLED` (0) and `SWIZZLED_ARBITRARY` (5).
pub fn swizzled_type(tex_type: u32) -> bool {
    matches!(tex_type, 0 | 5)
}

/// Interleave the low bits of `x` and `y` (Morton / Z-order) up to the square
/// formed by the smaller dimension, then append the remaining high bits of the
/// larger dimension linearly. This is the GXM swizzle for a power-of-two-padded
/// block grid `pw x ph`, matching how the GPU addresses a SWIZZLED texture.
fn morton_index(mut x: u32, mut y: u32, pw: u32, ph: u32) -> u32 {
    let min_log = pw.min(ph).trailing_zeros();
    let mut index = 0u32;
    for i in 0..min_log {
        index |= ((x >> i) & 1) << (2 * i + 1);
        index |= ((y >> i) & 1) << (2 * i);
    }
    // Strip the interleaved low bits; the remaining bits of the longer axis follow
    // the square in row/column-major order.
    x >>= min_log;
    y >>= min_log;
    let interleaved_bits = 2 * min_log;
    if pw >= ph {
        // Wider than tall: leftover columns tile to the right of the square.
        index | (((y * (pw >> min_log)) + x) << interleaved_bits)
    } else {
        // Taller than wide: leftover rows tile below the square.
        index | (((x * (ph >> min_log)) + y) << interleaved_bits)
    }
}

/// Expand a 16-bit 5:6:5 color to RGB8.
fn rgb565(c: u16) -> [u8; 3] {
    let r = ((c >> 11) & 0x1f) as u32;
    let g = ((c >> 5) & 0x3f) as u32;
    let b = (c & 0x1f) as u32;
    [(r * 255 / 31) as u8, (g * 255 / 63) as u8, (b * 255 / 31) as u8]
}

/// Decode one texel `(px, py)` in `[0,4)` from a 4x4 BC/DXT block. Handles BC1
/// (`0x85`, 8-byte block, optional 1-bit alpha), BC2 (`0x86`, explicit 4-bit
/// alpha) and BC3 (`0x87`, interpolated alpha). Returns straight RGBA8.
fn decode_bc_texel(block: &[u8], base_format: u32, px: u32, py: u32) -> [u8; 4] {
    let t = (py * 4 + px) as usize;
    // The BC1 color sub-block sits after the 8-byte alpha block for BC2/BC3.
    let color_off = if base_format == 0x85 { 0 } else { 8 };
    let g = |i: usize| -> u8 { *block.get(i).unwrap_or(&0) };
    let c0 = u16::from_le_bytes([g(color_off), g(color_off + 1)]);
    let c1 = u16::from_le_bytes([g(color_off + 2), g(color_off + 3)]);
    let e0 = rgb565(c0);
    let e1 = rgb565(c1);
    let idx = (g(color_off + 4 + t / 4) >> ((t % 4) * 2)) & 0x3;
    // BC1 with c0 <= c1 selects the 3-color + punch-through-alpha mode; BC2/BC3
    // colors always use the 4-color interpolation.
    let punchthrough = base_format == 0x85 && c0 <= c1;
    let mix = |a: [u8; 3], b: [u8; 3], na: u32, nb: u32, d: u32| -> [u8; 3] {
        [
            ((a[0] as u32 * na + b[0] as u32 * nb) / d) as u8,
            ((a[1] as u32 * na + b[1] as u32 * nb) / d) as u8,
            ((a[2] as u32 * na + b[2] as u32 * nb) / d) as u8,
        ]
    };
    let rgb = match idx {
        0 => e0,
        1 => e1,
        2 if punchthrough => mix(e0, e1, 1, 1, 2),
        2 => mix(e0, e1, 2, 1, 3),
        _ if punchthrough => [0, 0, 0],
        _ => mix(e0, e1, 1, 2, 3),
    };
    let a = match base_format {
        // BC1: opaque, except the punch-through index in 3-color mode.
        0x85 => {
            if punchthrough && idx == 3 {
                0
            } else {
                255
            }
        }
        // BC2: 4-bit alpha per texel, two texels per byte, low nibble first.
        0x86 => {
            let byte = g(t / 2);
            let a4 = if t % 2 == 0 { byte & 0xf } else { byte >> 4 };
            (a4 as u32 * 255 / 15) as u8
        }
        // BC3: two 8-bit endpoints + 3-bit interpolation indices.
        0x87 => bc3_alpha(block, t),
        _ => 255,
    };
    [rgb[0], rgb[1], rgb[2], a]
}

/// Decode the interpolated alpha of texel `t` from a BC3 (DXT5) 16-byte block.
fn bc3_alpha(block: &[u8], t: usize) -> u8 {
    let a0 = *block.first().unwrap_or(&0);
    let a1 = *block.get(1).unwrap_or(&0);
    // 16 texels x 3-bit indices packed little-endian across bytes 2..8.
    let bit = t * 3;
    let byte = 2 + bit / 8;
    let shift = bit % 8;
    let raw = (*block.get(byte).unwrap_or(&0) as u32)
        | ((*block.get(byte + 1).unwrap_or(&0) as u32) << 8);
    let code = ((raw >> shift) & 0x7) as u8;
    let (a0i, a1i) = (a0 as u32, a1 as u32);
    match code {
        0 => a0,
        1 => a1,
        c if a0 > a1 => (((8 - c as u32) * a0i + (c as u32 - 1) * a1i) / 7) as u8,
        6 => 0,
        7 => 255,
        c => (((6 - c as u32) * a0i + (c as u32 - 1) * a1i) / 5) as u8,
    }
}

/// Point-sample a captured texture at normalized `(u, v)` (REPEAT wrap) and decode
/// the texel to straight RGBA8. Covers the uncompressed formats a 2D title uses;
/// an unknown format returns opaque magenta so it is visible, not silent.
fn sample_texture(t: &BoundTexture, u: f32, v: f32) -> [u8; 4] {
    if t.width == 0 || t.height == 0 {
        return [255, 0, 255, 255];
    }
    // REPEAT wrap: fractional part into [0, 1).
    let uu = u - u.floor();
    let vv = v - v.floor();
    let x = ((uu * t.width as f32) as i64).clamp(0, t.width as i64 - 1) as u32;
    let y = ((vv * t.height as f32) as i64).clamp(0, t.height as i64 - 1) as u32;
    texel_rgba(t, x, y)
}

/// Decode a whole captured texture to a tightly-packed linear RGBA8 image at its
/// native `width x height`, using the exact per-texel decode the software sampler
/// uses. This is the seam the wgpu backend samples through: the format/swizzle/BC/
/// Morton complexity stays here (one tested place), and the GPU sees a plain
/// `Rgba8Unorm` image it can point-sample with a REPEAT sampler for the identical
/// result. Returns `(width, height, rgba)`; a zero-sized texture yields a 1x1 opaque
/// magenta so an unexpected empty binding is visible rather than a GPU error.
pub fn decode_texture_rgba8(t: &BoundTexture) -> (u32, u32, Vec<u8>) {
    if t.width == 0 || t.height == 0 {
        return (1, 1, vec![255, 0, 255, 255]);
    }
    let mut rgba = Vec::with_capacity((t.width * t.height * 4) as usize);
    for y in 0..t.height {
        for x in 0..t.width {
            rgba.extend_from_slice(&texel_rgba(t, x, y));
        }
    }
    (t.width, t.height, rgba)
}

/// Decode the single texel at integer coordinates `(x, y)` (already wrapped/clamped
/// into `[0, width) x [0, height)`) to straight RGBA8. Handles the block-compressed
/// (BC/DXT, optionally Morton-swizzled) and the uncompressed format families; an
/// unknown format returns opaque magenta so it is visible, not silent.
fn texel_rgba(t: &BoundTexture, x: u32, y: u32) -> [u8; 4] {
    let Some((block_w, block_h, block_bytes)) = block_layout(t.base_format) else {
        return [255, 0, 255, 255];
    };
    // Block-compressed (BC/DXT): locate the 4x4 block (Morton-addressed when the
    // texture is swizzled, else row-major), decode it, and apply the channel
    // swizzle to the decoded RGBA (ABGR/field 0 is the identity).
    if block_w > 1 {
        let (bx, by) = (x / block_w, y / block_h);
        let block_index = if swizzled_type(t.tex_type) {
            let pw = t.width.div_ceil(block_w).next_power_of_two();
            let ph = t.height.div_ceil(block_h).next_power_of_two();
            morton_index(bx, by, pw, ph)
        } else {
            by * (t.stride / block_bytes) + bx
        };
        let off = (block_index * block_bytes) as usize;
        let block = t.pixels.get(off..off + block_bytes as usize).unwrap_or(&[]);
        let rgba = decode_bc_texel(block, t.base_format, x % block_w, y % block_h);
        let swizzle = (t.swizzle >> 12) & 0x7;
        return swizzle4(rgba[0], rgba[1], rgba[2], rgba[3], swizzle);
    }
    let off = (y * t.stride + x * block_bytes) as usize;
    let px = &t.pixels;
    let byte = |i: usize| -> u8 { *px.get(off + i).unwrap_or(&0) };
    // Channel swizzle field (bits 12..14 of the full SceGxmTextureFormat).
    let swizzle = (t.swizzle >> 12) & 0x7;
    match t.base_format {
        // 24-bit three-channel (U8U8U8): opaque. SWIZZLE3 BGR(0)/RGB(1); the 24-bit
        // value's MSB is memory byte b2, so RGB(1) -> [b2,b1,b0], BGR(0) -> [b0,b1,b2].
        0x98 | 0x99 => {
            let (b0, b1, b2) = (byte(0), byte(1), byte(2));
            match swizzle {
                1 => [b2, b1, b0, 255], // RGB
                _ => [b0, b1, b2, 255], // BGR
            }
        }
        // 32-bit four-channel (U8U8U8U8 et al). SWIZZLE4 permutes the memory bytes.
        0x0c..=0x1a => swizzle4(byte(0), byte(1), byte(2), byte(3), swizzle),
        // U1U5U5U5: little-endian 16-bit, MSB->LSB lanes = [1-bit A, 5, 5, 5]. The
        // 1-bit lane is the alpha; the three 5-bit lanes permute by swizzle.
        0x04 => {
            let w = u16::from_le_bytes([byte(0), byte(1)]);
            let a = if w & 0x8000 != 0 { 255 } else { 0 };
            let hi = (((w >> 10) & 0x1f) as u32 * 255 / 31) as u8; // bits 14..10
            let mid = (((w >> 5) & 0x1f) as u32 * 255 / 31) as u8; // bits 9..5
            let lo = ((w & 0x1f) as u32 * 255 / 31) as u8; // bits 4..0
            match swizzle {
                1 => [hi, mid, lo, a], // ARGB: R,G,B = hi,mid,lo
                _ => [lo, mid, hi, a], // ABGR: R,G,B = lo,mid,hi
            }
        }
        // U5U6U5: opaque 5-6-5 (SWIZZLE3 BGR/RGB); high 5 bits are R for RGB order.
        0x05 => {
            let w = u16::from_le_bytes([byte(0), byte(1)]);
            let hi = (((w >> 11) & 0x1f) as u32 * 255 / 31) as u8;
            let g = (((w >> 5) & 0x3f) as u32 * 255 / 63) as u8;
            let lo = ((w & 0x1f) as u32 * 255 / 31) as u8;
            match swizzle {
                0 => [lo, g, hi, 255], // BGR
                _ => [hi, g, lo, 255], // RGB
            }
        }
        // U4U4U4U4: MSB->LSB nibble lanes permute by swizzle (as for 8888).
        0x02 => {
            let w = u16::from_le_bytes([byte(0), byte(1)]);
            let n = |sh: u32| (((w >> sh) & 0xf) as u32 * 255 / 15) as u8;
            // n0 = LSB nibble .. n3 = MSB nibble, matching b0..b3 lane roles.
            swizzle4(n(0), n(4), n(8), n(12), swizzle)
        }
        // Single channel U8 (fonts / masks): replicate to grey, full alpha.
        0x00 | 0x01 => {
            let c = byte(0);
            [c, c, c, 255]
        }
        _ => [255, 0, 255, 255],
    }
}

/// Permute four memory-order lanes (`b0` = least-significant byte .. `b3` =
/// most-significant) into straight RGBA per a GXM `SWIZZLE4` selector. The swizzle
/// names channels most-significant to least-significant, and the texel's MSB is the
/// high memory byte `b3` (little-endian), so e.g. ARGB is `b3=A, b2=R, b1=G, b0=B`,
/// giving RGBA `[b2, b1, b0, b3]` - which correctly keeps a low-alpha byte as alpha
/// rather than turning it into a color channel.
fn swizzle4(b0: u8, b1: u8, b2: u8, b3: u8, swizzle: u32) -> [u8; 4] {
    match swizzle {
        1 => [b2, b1, b0, b3], // ARGB
        2 => [b3, b2, b1, b0], // RGBA
        3 => [b1, b2, b3, b0], // BGRA
        _ => [b0, b1, b2, b3], // ABGR
    }
}

/// The recovered rendering intent of one draw: how its interleaved vertex maps to
/// position/texcoord/color, what coordinate space its positions live in, whether it
/// samples a texture, and the texcoord divisor. Shared by the software rasterizer and
/// the [`RenderScene`](vitaslop_platform::gpu::RenderScene) builder so the two paths
/// make identical per-draw decisions - the GPU renderer stays the faithful twin of
/// the CPU oracle.
struct DrawInterp {
    layout: Layout,
    space: Space,
    /// The texture the draw samples (its first bound texture), or `None` if it reads
    /// no texcoord.
    textured: bool,
    /// Divisor applied to texcoords to normalize atlas-in-pixels coords to [0,1].
    uv_div: [f32; 2],
    /// Skip the draw entirely: it carries neither a texcoord nor a per-vertex color, so
    /// its fragment color comes only from the (uncaptured) fragment program / a uniform
    /// we cannot interpret. Such position-only geometry is typically a full-screen effect
    /// pass (a fog/tint/clear over an NDC cover triangle); with no color source we would
    /// otherwise fall back to OPAQUE WHITE and white out the background wherever nothing
    /// draws over it. Skipping (leaving what is behind) is the strictly-safer fallback -
    /// an invisible pass we can't reproduce, not a screen-filling white artifact.
    skip: bool,
}

/// Recover a draw's [`DrawInterp`] the same way for both render paths.
fn interpret_draw(d: &Draw) -> DrawInterp {
    let layout = layout_of(d);
    // Recover the draw's coordinate space (see `Space`). A 4x4 MVP uniform is the 3D
    // cube path (depth-tested, opaque). Otherwise a 2D draw: a texcoord marks a
    // pixel-space sprite, a bare position is an NDC fullscreen pass.
    let space = if d.uniforms.len() >= 16 {
        let mut m = [0f32; 16];
        m.copy_from_slice(&d.uniforms[..16]);
        Space::Mvp(m)
    } else if layout.uv_off.is_some() {
        Space::Pixel
    } else {
        Space::Ndc
    };
    // Texture the draw only if it actually reads a texcoord; a sticky texture binding
    // left over from a previous draw must not tint an untextured fill.
    let textured = layout.uv_off.is_some() && !d.textures.is_empty();

    // GXM texcoords are either normalized [0,1] or in texel units, chosen by the
    // (unavailable) fragment program's sampler. Infer from magnitude: a coord well
    // past 1 is a texel index into an atlas (OlliOlli's text quads index a font atlas
    // in pixels), which we normalize by the texture size. Normalized sprite UVs
    // (<= ~1) pass through unchanged.
    let uv_div = match (textured, d.textures.first()) {
        (true, Some(tex)) => {
            let stride = d.vertex_stride.max(1) as usize;
            let nverts = d.vertices.len() / stride;
            let mut max_uv = 0f32;
            for i in 0..nverts {
                let vv = decode_vertex(d, &layout, i);
                max_uv = max_uv.max(vv.uv[0].abs()).max(vv.uv[1].abs());
            }
            if max_uv > 1.5 {
                [tex.width.max(1) as f32, tex.height.max(1) as f32]
            } else {
                [1.0, 1.0]
            }
        }
        _ => [1.0, 1.0],
    };
    // Position-only geometry (no texcoord AND no per-vertex color) has no color source
    // we can honor - see `DrawInterp::skip`. Skip it rather than paint opaque white.
    let skip = layout.uv_off.is_none() && layout.color_off.is_none();
    DrawInterp { layout, space, textured, uv_div, skip }
}

/// Rasterize one scene into a fresh framebuffer. `clear` is the background color.
pub fn render_scene(scene: &Scene, width: u32, height: u32, clear: [u8; 4]) -> Framebuffer {
    let mut fb = Framebuffer::new(width, height, clear);
    let mut depth = vec![f32::INFINITY; (width * height) as usize];

    // Diagnostic: VITASLOP_PIXEL_TRACE=x,y logs every draw that writes that pixel (index,
    // whether textured/depth-tested, the source RGBA) - the definitive "which draw painted
    // this glitch" probe. Zero cost when unset.
    let trace: Option<(i32, i32)> = std::env::var("VITASLOP_PIXEL_TRACE").ok().and_then(|s| {
        let (a, b) = s.split_once(',')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    });

    for (di, d) in scene.draws.iter().enumerate() {
        // Only triangle lists are handled (SCE_GXM_PRIMITIVE_TRIANGLES = 0).
        if d.primitive != 0 {
            continue;
        }
        let DrawInterp { layout, space, textured, uv_div, skip } = interpret_draw(d);
        if skip {
            continue;
        }
        // Depth-test and replace only in the 3D path; 2D draws paint in submission
        // order with alpha blending.
        let depth_test = matches!(space, Space::Mvp(_));
        let texture = if textured { d.textures.first() } else { None };

        let tri_count = d.index_count as usize / 3;
        for t in 0..tri_count {
            let vs = [index_at(d, t * 3), index_at(d, t * 3 + 1), index_at(d, t * 3 + 2)];
            let verts: Vec<Vertex> = vs.iter().map(|&i| decode_vertex(d, &layout, i)).collect();

            // Project to screen; drop the triangle if any vertex is behind the eye.
            let mut screen = [[0f32; 4]; 3]; // x, y, depth, 1/w
            let mut behind = false;
            for (k, v) in verts.iter().enumerate() {
                match project(v, &space, width, height) {
                    Some(s) => screen[k] = s,
                    None => {
                        behind = true;
                        break;
                    }
                }
            }
            if behind {
                continue;
            }
            raster_triangle(&mut fb, &mut depth, &screen, &verts, texture, uv_div, depth_test, trace, di);
        }
    }
    fb
}

/// Rasterize one screen-space triangle with perspective-correct interpolation of
/// texcoord and color. `depth_test` gates the z-buffer (3D opaque); when off the
/// fragment is alpha-blended over the framebuffer in submission order (2D sprites).
fn raster_triangle(
    fb: &mut Framebuffer,
    depth: &mut [f32],
    s: &[[f32; 4]; 3],
    verts: &[Vertex],
    texture: Option<&BoundTexture>,
    uv_div: [f32; 2],
    depth_test: bool,
    trace: Option<(i32, i32)>,
    draw_idx: usize,
) {
    let (w, h) = (fb.width as i32, fb.height as i32);
    let min_x = s.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min).floor().max(0.0) as i32;
    let max_x = s.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max).ceil().min((w - 1) as f32) as i32;
    let min_y = s.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min).floor().max(0.0) as i32;
    let max_y = s.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max).ceil().min((h - 1) as f32) as i32;
    if min_x > max_x || min_y > max_y {
        return;
    }

    // Edge/area via the 2D cross product. A zero area is a degenerate triangle.
    let area = edge(&s[0], &s[1], &s[2]);
    if area.abs() < 1e-6 {
        return;
    }

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = [x as f32 + 0.5, y as f32 + 0.5];
            let w0 = edge(&s[1], &s[2], &[p[0], p[1], 0.0, 0.0]);
            let w1 = edge(&s[2], &s[0], &[p[0], p[1], 0.0, 0.0]);
            let w2 = edge(&s[0], &s[1], &[p[0], p[1], 0.0, 0.0]);
            // Inside if all barycentrics share the triangle's sign (either winding).
            let inside = (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0)
                || (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);
            if !inside {
                continue;
            }
            let (b0, b1, b2) = (w0 / area, w1 / area, w2 / area);
            let idx = (y * w + x) as usize;
            if depth_test {
                let z = b0 * s[0][2] + b1 * s[1][2] + b2 * s[2][2];
                if z >= depth[idx] {
                    continue;
                }
                depth[idx] = z;
            }

            // Perspective-correct interpolation: weight each attribute by 1/w and
            // renormalize by the interpolated 1/w.
            let iw = b0 * s[0][3] + b1 * s[1][3] + b2 * s[2][3];
            let interp = |a: f32, b: f32, c: f32| -> f32 {
                (b0 * s[0][3] * a + b1 * s[1][3] * b + b2 * s[2][3] * c) / iw
            };
            let mut src = [0f32; 4];
            for ch in 0..4 {
                src[ch] = interp(
                    verts[0].color[ch] as f32,
                    verts[1].color[ch] as f32,
                    verts[2].color[ch] as f32,
                );
            }
            // Modulate by the sampled texel (texture * vertex color, the standard
            // 2D sprite fragment program).
            if let Some(tex) = texture {
                let u = interp(verts[0].uv[0], verts[1].uv[0], verts[2].uv[0]) / uv_div[0];
                let v = interp(verts[0].uv[1], verts[1].uv[1], verts[2].uv[1]) / uv_div[1];
                let texel = sample_texture(tex, u, v);
                for ch in 0..4 {
                    src[ch] = src[ch] * texel[ch] as f32 / 255.0;
                }
            }

            if let Some((tx, ty)) = trace {
                if x == tx && y == ty {
                    eprintln!(
                        "PXTRACE ({tx},{ty}) draw {draw_idx} textured={} depth_test={} src=[{:.0},{:.0},{:.0},{:.0}]",
                        texture.is_some(), depth_test, src[0], src[1], src[2], src[3]
                    );
                }
            }
            let dst = idx * 4;
            if depth_test {
                // Opaque replace (with the z-buffer already updated).
                for ch in 0..4 {
                    fb.rgba[dst + ch] = src[ch].round().clamp(0.0, 255.0) as u8;
                }
                fb.rgba[dst + 3] = 255;
            } else {
                // Straight-alpha src-over blend for 2D sprites.
                let a = (src[3] / 255.0).clamp(0.0, 1.0);
                for ch in 0..3 {
                    let out = src[ch] * a + fb.rgba[dst + ch] as f32 * (1.0 - a);
                    fb.rgba[dst + ch] = out.round().clamp(0.0, 255.0) as u8;
                }
                let out_a = src[3] + fb.rgba[dst + 3] as f32 * (1.0 - a);
                fb.rgba[dst + 3] = out_a.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Twice the signed area of triangle (a, b, c) in screen space (only x,y used).
fn edge(a: &[f32; 4], b: &[f32; 4], c: &[f32; 4]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

use std::collections::HashMap;
use std::sync::Arc;
use vitaslop_platform::gpu::{DrawSpace, GxmDraw, GxmTexture, RenderScene, GXM_VERTEX_STRIDE};

/// Map the software rasterizer's internal [`Space`] to the neutral GPU
/// [`DrawSpace`] the renderer consumes. One-to-one; kept as a small function so the
/// two enums can not drift silently.
fn to_draw_space(space: &Space) -> DrawSpace {
    match space {
        Space::Mvp(m) => DrawSpace::Mvp(*m),
        Space::Ndc => DrawSpace::Ndc,
        Space::Pixel => DrawSpace::Pixel,
    }
}

/// A cheap content/identity fingerprint of a captured texture, used to cache its
/// decode (here) and its GPU upload (in the renderer). It folds the control words
/// (address, format, swizzle, type, geometry) and a strided sample of the pixel
/// bytes - enough to notice a same-address atlas whose contents changed without
/// hashing every byte every frame. FNV-1a/64.
fn tex_key(t: &BoundTexture) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
    };
    mix(t.data_addr as u64);
    mix(t.base_format as u64);
    mix(t.swizzle as u64);
    mix(t.tex_type as u64);
    mix(((t.width as u64) << 32) | t.height as u64);
    mix(((t.stride as u64) << 32) | t.pixels.len() as u64);
    // Sample at most ~256 bytes spread across the buffer so a content change is seen
    // cheaply regardless of texture size.
    let n = t.pixels.len();
    if n > 0 {
        let step = (n / 256).max(1);
        let mut i = 0;
        while i < n {
            mix(t.pixels[i] as u64);
            i += step;
        }
    }
    h
}

/// Builds a neutral [`RenderScene`] from a captured GXM [`Scene`] for the GPU
/// renderer, reusing the exact per-draw interpretation ([`interpret_draw`]) and
/// texture decode ([`decode_texture_rgba8`]) the software rasterizer uses, so the
/// GPU output matches the CPU oracle. Holds a cross-frame texture-decode cache keyed
/// by [`tex_key`] so an unchanged atlas is decoded once and thereafter only its
/// shared `Arc` is handed back; persist one builder across a run's frames to keep the
/// cache warm.
pub struct RenderSceneBuilder {
    decode_cache: HashMap<u64, GxmTexture>,
}

/// Cap on the decode cache; cleared wholesale if exceeded (a title's working texture
/// set is far smaller, so this only fires on a pathological churn and just forces a
/// re-decode, never incorrectness).
const DECODE_CACHE_CAP: usize = 512;

impl Default for RenderSceneBuilder {
    fn default() -> Self {
        RenderSceneBuilder::new()
    }
}

impl RenderSceneBuilder {
    pub fn new() -> Self {
        RenderSceneBuilder { decode_cache: HashMap::new() }
    }

    /// Decode (or reuse a cached) GPU-ready texture for `t`.
    fn texture(&mut self, t: &BoundTexture) -> GxmTexture {
        let key = tex_key(t);
        if let Some(g) = self.decode_cache.get(&key) {
            return g.clone();
        }
        if self.decode_cache.len() >= DECODE_CACHE_CAP {
            self.decode_cache.clear();
        }
        let (width, height, rgba) = decode_texture_rgba8(t);
        let g = GxmTexture { key, width, height, rgba: Arc::new(rgba) };
        self.decode_cache.insert(key, g.clone());
        g
    }

    /// Reduce a captured scene to general draws. Each triangle-list draw is decoded
    /// into the canonical vertex layout (position, uv already divided by the draw's uv
    /// scale, color) with a 32-bit index buffer, and its texture (if it samples one)
    /// decoded to linear RGBA8. Non-triangle-list draws are skipped, matching the
    /// software rasterizer.
    pub fn build(&mut self, scene: &Scene) -> RenderScene {
        let mut draws = Vec::with_capacity(scene.draws.len());
        for d in &scene.draws {
            if d.primitive != 0 || d.index_count == 0 {
                continue;
            }
            let interp = interpret_draw(d);
            if interp.skip {
                continue;
            }
            let layout = &interp.layout;

            // The largest index referenced, so the vertex buffer covers every index
            // (an out-of-range index decodes to a zero vertex, matching the software
            // path's clamped reads, rather than a GPU out-of-bounds fetch).
            let mut max_idx = 0usize;
            for i in 0..d.index_count as usize {
                max_idx = max_idx.max(index_at(d, i));
            }
            let stride = d.vertex_stride.max(1) as usize;
            let nverts = (d.vertices.len() / stride).max(max_idx + 1);

            let mut vertices = Vec::with_capacity(nverts * GXM_VERTEX_STRIDE as usize);
            for i in 0..nverts {
                let v = decode_vertex(d, layout, i);
                vertices.extend_from_slice(&v.pos[0].to_le_bytes());
                vertices.extend_from_slice(&v.pos[1].to_le_bytes());
                vertices.extend_from_slice(&v.pos[2].to_le_bytes());
                // Fold the uv divisor in here (constant per draw, so pre-dividing per
                // vertex is identical to dividing the interpolated coord).
                vertices.extend_from_slice(&(v.uv[0] / interp.uv_div[0]).to_le_bytes());
                vertices.extend_from_slice(&(v.uv[1] / interp.uv_div[1]).to_le_bytes());
                vertices.extend_from_slice(&v.color);
            }

            let mut indices = Vec::with_capacity(d.index_count as usize * 4);
            for i in 0..d.index_count as usize {
                indices.extend_from_slice(&(index_at(d, i) as u32).to_le_bytes());
            }

            let texture = if interp.textured {
                d.textures.first().map(|t| self.texture(t))
            } else {
                None
            };

            draws.push(GxmDraw {
                space: to_draw_space(&interp.space),
                vertices,
                indices,
                index_count: d.index_count,
                texture,
            });
        }
        RenderScene { draws }
    }
}

#[cfg(test)]
mod texture_tests {
    //! Content-free decode conformance for the GXM texture formats a 2D title
    //! uses. Each builds a tiny texture with known bytes and asserts the sampled,
    //! swizzled RGBA - no game data, no fixture. These pin the format/swizzle
    //! decode that the render output depends on.
    use super::*;

    fn tex(base_format: u32, swizzle: u32, w: u32, h: u32, stride: u32, pixels: Vec<u8>) -> BoundTexture {
        BoundTexture {
            unit: 0,
            base_format,
            // sample_texture reads bits 12..14; pass the swizzle in that field.
            swizzle: swizzle << 12,
            tex_type: 3,
            width: w,
            height: h,
            stride,
            data_addr: 0,
            pixels,
        }
    }

    // u that lands squarely in texel `i` of a `w`-wide row.
    fn u_of(i: u32, w: u32) -> f32 {
        (i as f32 + 0.5) / w as f32
    }

    #[test]
    fn u8u8u8u8_swizzles() {
        // Memory bytes per texel: b0,b1,b2,b3. The swizzle names channels MSB->LSB,
        // and the texel MSB is byte b3 (little-endian).
        let px = vec![10, 20, 30, 40, /* t1 */ 50, 60, 70, 80];
        // ABGR (0): b3=A,b2=B,b1=G,b0=R -> RGBA = [b0,b1,b2,b3].
        let t = tex(0x0c, 0, 2, 1, 8, px.clone());
        assert_eq!(sample_texture(&t, u_of(0, 2), 0.0), [10, 20, 30, 40]);
        assert_eq!(sample_texture(&t, u_of(1, 2), 0.0), [50, 60, 70, 80]);
        // ARGB (1): b3=A,b2=R,b1=G,b0=B -> RGBA = [b2,b1,b0,b3].
        let t = tex(0x0c, 1, 2, 1, 8, px.clone());
        assert_eq!(sample_texture(&t, u_of(0, 2), 0.0), [30, 20, 10, 40]);
        // RGBA (2): b3=R,b2=G,b1=B,b0=A -> RGBA = [b3,b2,b1,b0].
        let t = tex(0x0c, 2, 2, 1, 8, px.clone());
        assert_eq!(sample_texture(&t, u_of(0, 2), 0.0), [40, 30, 20, 10]);
        // BGRA (3): b3=B,b2=G,b1=R,b0=A -> RGBA = [b1,b2,b3,b0].
        let t = tex(0x0c, 3, 2, 1, 8, px);
        assert_eq!(sample_texture(&t, u_of(0, 2), 0.0), [20, 30, 40, 10]);
    }

    #[test]
    fn argb_keeps_low_alpha_translucent_not_a_color() {
        // The regression that motivated the swizzle fix: OlliOlli's "Loading" text
        // highlight is an ARGB U8U8U8U8 texel [ff ff ff 40] = white at 25% alpha.
        // A wrong swizzle turned the 0x40 alpha byte into a color -> opaque yellow.
        let t = tex(0x0c, 1, 1, 1, 4, vec![0xff, 0xff, 0xff, 0x40]);
        assert_eq!(sample_texture(&t, 0.5, 0.0), [0xff, 0xff, 0xff, 0x40]);
    }

    #[test]
    fn u8u8u8_is_24bit_opaque_rgb() {
        // 0x98 must not be read as 32-bit (it aliases S32 under a 5-bit field).
        let t = tex(0x98, 0, 2, 1, 6, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(sample_texture(&t, u_of(0, 2), 0.0), [1, 2, 3, 255]);
        assert_eq!(sample_texture(&t, u_of(1, 2), 0.0), [4, 5, 6, 255]);
    }

    #[test]
    fn u1u5u5u5_alpha_and_channels() {
        // ABGR: high 5 bits map to B (output[0]=c2=low 5), 1-bit alpha at 0x8000.
        // Opaque with only the low 5 bits set -> red 255 in ABGR order.
        let w: u16 = 0x8000 | 0x001f;
        let t = tex(0x04, 0, 1, 1, 2, w.to_le_bytes().to_vec());
        assert_eq!(sample_texture(&t, 0.5, 0.0), [255, 0, 0, 255]);
        // Alpha bit clear -> transparent.
        let t = tex(0x04, 0, 1, 1, 2, 0x001fu16.to_le_bytes().to_vec());
        assert_eq!(sample_texture(&t, 0.5, 0.0)[3], 0);
    }

    #[test]
    fn u5u6u5_opaque_565() {
        // Middle 6 bits are green in either channel order.
        let t = tex(0x05, 0, 1, 1, 2, 0x07e0u16.to_le_bytes().to_vec());
        assert_eq!(sample_texture(&t, 0.5, 0.0), [0, 255, 0, 255]);
        // BGR (swizzle 0): top 5 bits = blue.
        let t = tex(0x05, 0, 1, 1, 2, 0xf800u16.to_le_bytes().to_vec());
        assert_eq!(sample_texture(&t, 0.5, 0.0), [0, 0, 255, 255]);
        // RGB (swizzle 1): top 5 bits = red.
        let t = tex(0x05, 1, 1, 1, 2, 0xf800u16.to_le_bytes().to_vec());
        assert_eq!(sample_texture(&t, 0.5, 0.0), [255, 0, 0, 255]);
    }

    #[test]
    fn repeat_wrap_is_fractional() {
        let t = tex(0x0c, 0, 2, 1, 8, vec![10, 0, 0, 255, 20, 0, 0, 255]);
        // u = 1.25 wraps to 0.25 -> texel 0; u = -0.25 wraps to 0.75 -> texel 1.
        assert_eq!(sample_texture(&t, 1.0 + u_of(0, 2), 0.0)[0], 10);
        assert_eq!(sample_texture(&t, -1.0 + u_of(1, 2), 0.0)[0], 20);
    }

    #[test]
    fn second_row_uses_stride() {
        // 1x2 texture, stride 8 (padded): row 1 begins at byte 8.
        let t = tex(0x0c, 0, 1, 2, 8, vec![1, 2, 3, 4, 0, 0, 0, 0, 9, 8, 7, 6]);
        assert_eq!(sample_texture(&t, 0.5, u_of(0, 2)), [1, 2, 3, 4]);
        assert_eq!(sample_texture(&t, 0.5, u_of(1, 2)), [9, 8, 7, 6]);
    }

    // Build a texture with an explicit `SceGxmTextureType` selector (LINEAR = 3,
    // SWIZZLED = 0), so the block-compressed / swizzled paths can be exercised.
    fn tex_typed(base_format: u32, tex_type: u32, w: u32, h: u32, stride: u32, pixels: Vec<u8>) -> BoundTexture {
        BoundTexture { unit: 0, base_format, swizzle: 0, tex_type, width: w, height: h, stride, data_addr: 0, pixels }
    }

    #[test]
    fn bc2_block_alpha_and_color() {
        // One BC2 (DXT3) 4x4 block: 8 bytes 4-bit alpha, then a BC1 color sub-block.
        // Color endpoint c0 = white 565 with all-zero indices, so every texel is
        // white; alpha byte 0 holds texel 0 (low nibble = 15) and texel 1 (high
        // nibble = 0). This is the OlliOlli menu-font case: white RGB, glyph coverage
        // carried entirely in the 4-bit alpha.
        let mut block = vec![0u8; 16];
        block[0] = 0x0F; // texel0 alpha=15 (opaque), texel1 alpha=0 (transparent)
        block[8] = 0xFF; // c0 low byte
        block[9] = 0xFF; // c0 high byte -> 0xFFFF = white
        let t = tex_typed(0x86, 3, 4, 4, 16, block); // LINEAR layout
        assert_eq!(sample_texture(&t, u_of(0, 4), u_of(0, 4)), [255, 255, 255, 255]);
        assert_eq!(sample_texture(&t, u_of(1, 4), u_of(0, 4)), [255, 255, 255, 0]);
    }

    #[test]
    fn bc1_punchthrough_alpha() {
        // BC1 with c0 <= c1 selects the 3-color + punch-through mode: index 3 is
        // transparent black. c0 = 0, c1 = white, index 3 for texel 0.
        let mut block = vec![0u8; 8];
        block[0] = 0x00;
        block[1] = 0x00; // c0 = 0
        block[2] = 0xFF;
        block[3] = 0xFF; // c1 = white (c0 <= c1 -> punch-through)
        block[4] = 0x03; // texel 0 index = 3 -> transparent
        let t = tex_typed(0x85, 3, 4, 4, 8, block);
        assert_eq!(sample_texture(&t, u_of(0, 4), u_of(0, 4)), [0, 0, 0, 0]);
    }

    #[test]
    fn morton_zorder_matches_reference() {
        // Z-order over a square power-of-two block grid: bit(2i)=y_i, bit(2i+1)=x_i.
        assert_eq!(morton_index(0, 0, 4, 4), 0);
        assert_eq!(morton_index(1, 0, 4, 4), 2);
        assert_eq!(morton_index(0, 1, 4, 4), 1);
        assert_eq!(morton_index(1, 1, 4, 4), 3);
        assert_eq!(morton_index(2, 1, 4, 4), 9);
        // Wider-than-tall grid: leftover columns tile as whole 4x4 squares to the
        // right, so block (4,0) begins the second square at index 16.
        assert_eq!(morton_index(4, 0, 8, 4), 16);
    }

    #[test]
    fn bc2_swizzled_block_addressing() {
        // A 16x16-texel swizzled BC2 texture = a 4x4 grid of 16-byte blocks. Block
        // (2,1) is at Morton index 9 (vs linear index 6), so a correct swizzle read
        // must fetch from byte 9*16. Mark that block white-transparent and the rest
        // opaque, then sample a texel inside block (2,1) and require alpha 0.
        let mut px = vec![0u8; 16 * 16]; // 16 blocks
        for b in 0..16 {
            px[b * 16] = 0xFF; // both low texels opaque
            px[b * 16 + 8] = 0xFF;
            px[b * 16 + 9] = 0xFF; // c0 white
        }
        let target = morton_index(2, 1, 4, 4) as usize; // = 9
        assert_eq!(target, 9);
        px[target * 16] = 0x00; // block (2,1): texel 0 alpha 0
        let t = tex_typed(0x86, 0, 16, 16, 0, px); // SWIZZLED
                                                   // Texel (8,4) is texel 0 of block (2,1).
        assert_eq!(sample_texture(&t, u_of(8, 16), u_of(4, 16))[3], 0);
        // A neighboring block (texel (0,0), block (0,0)) stays opaque.
        assert_eq!(sample_texture(&t, u_of(0, 16), u_of(0, 16))[3], 255);
    }
}
