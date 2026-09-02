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

    /// Box-downsample by an integer `factor` (supersample resolve): each output pixel is
    /// the average of its `factor x factor` source block. This is how the renderer turns a
    /// scene rasterized at `factor`x resolution into an antialiased final frame - averaging
    /// resolves the geometric aliasing of a heavily-tessellated mesh (dozens of sub-pixel
    /// triangles per final pixel, e.g. a distant vehicle) and coincident-panel z-fighting
    /// into smooth pixels, the same edge/coverage integration hardware MSAA gives. `factor`
    /// == 1 returns an identical image. A remainder row/column (source size not a multiple of
    /// factor) is ignored, so callers should rasterize at exactly `factor * out` dimensions.
    pub fn downsampled(&self, factor: u32) -> Framebuffer {
        if factor <= 1 {
            return Framebuffer { width: self.width, height: self.height, rgba: self.rgba.clone() };
        }
        let ow = self.width / factor;
        let oh = self.height / factor;
        let mut rgba = Vec::with_capacity((ow * oh * 4) as usize);
        let inv = (factor * factor) as u32;
        for oy in 0..oh {
            for ox in 0..ow {
                let mut acc = [0u32; 4];
                for sy in 0..factor {
                    let row = ((oy * factor + sy) * self.width + ox * factor) as usize * 4;
                    for sx in 0..factor as usize {
                        let p = row + sx * 4;
                        acc[0] += self.rgba[p] as u32;
                        acc[1] += self.rgba[p + 1] as u32;
                        acc[2] += self.rgba[p + 2] as u32;
                        acc[3] += self.rgba[p + 3] as u32;
                    }
                }
                rgba.extend_from_slice(&[
                    (acc[0] / inv) as u8,
                    (acc[1] / inv) as u8,
                    (acc[2] / inv) as u8,
                    (acc[3] / inv) as u8,
                ]);
            }
        }
        Framebuffer { width: ow, height: oh, rgba }
    }

    /// Bilinearly scale to `(w, h)` - what the display controller does to a framebuffer
    /// smaller than the panel.
    ///
    /// A title is free to render its front end at a fraction of the panel and declare that
    /// smaller buffer to `sceDisplaySetFrameBuf`; the hardware stretches it. Returning the
    /// image unscaled puts the whole screen in a corner, so this is not a refinement, it is
    /// the difference between the picture and a corner of it.
    ///
    /// Bilinear rather than nearest because the hardware filters: a nearest-neighbour 1.5x
    /// of a 640-wide image doubles every other column, which is a visible comb on text and
    /// is not what the device shows. Same size in and out returns a copy and touches
    /// nothing, so every title that already declares the panel is unaffected.
    pub fn scaled_to(&self, w: u32, h: u32) -> Framebuffer {
        if (w, h) == (self.width, self.height) {
            return Framebuffer { width: w, height: h, rgba: self.rgba.clone() };
        }
        assert!(w > 0 && h > 0 && self.width > 0 && self.height > 0, "empty framebuffer scale");
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        // Sample at pixel CENTRES, so the scale does not shift the image by half a texel -
        // a half-pixel offset on a 2D front end is a visibly soft edge on every glyph.
        let sx = self.width as f32 / w as f32;
        let sy = self.height as f32 / h as f32;
        for oy in 0..h {
            let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
            let y0 = fy as u32;
            let y1 = (y0 + 1).min(self.height - 1);
            let ty = fy - y0 as f32;
            for ox in 0..w {
                let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
                let x0 = fx as u32;
                let x1 = (x0 + 1).min(self.width - 1);
                let tx = fx - x0 as f32;
                let (a, b, c, d) = (
                    self.pixel(x0, y0),
                    self.pixel(x1, y0),
                    self.pixel(x0, y1),
                    self.pixel(x1, y1),
                );
                let mut out = [0u8; 4];
                for k in 0..4 {
                    let top = a[k] as f32 + (b[k] as f32 - a[k] as f32) * tx;
                    let bot = c[k] as f32 + (d[k] as f32 - c[k] as f32) * tx;
                    out[k] = (top + (bot - top) * ty).round().clamp(0.0, 255.0) as u8;
                }
                rgba.extend_from_slice(&out);
            }
        }
        Framebuffer { width: w, height: h, rgba }
    }

    /// The RGBA color at `(x, y)`.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.width + x) * 4) as usize;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2], self.rgba[i + 3]]
    }

    /// Encode as a PNG (8-bit RGBA). Self-contained: uncompressed DEFLATE (stored
    /// blocks) so there is no compression dependency. Fine for reference dumps.
    pub fn to_png(&self) -> Vec<u8> {
        rgba_to_png(self.width, self.height, &self.rgba)
    }
}

/// Encode a tightly-packed RGBA8 image to PNG bytes (8-bit, no filter, stored deflate).
/// The blob-free image sink shared by [`Framebuffer::to_png`] and diagnostics that want to
/// write a decoded texture to disk (e.g. inspecting the albedo a draw sampled).
pub fn rgba_to_png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let row_bytes = (width * 4) as usize;
    let mut raw = Vec::with_capacity(rgba.len() + height as usize);
    for y in 0..height as usize {
        raw.push(0); // filter: None
        let row = y * row_bytes;
        raw.extend_from_slice(&rgba[row..row + row_bytes]);
    }
    let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, RGBA, deflate, no filter, no interlace
    write_chunk(&mut png, b"IHDR", &ihdr);
    write_chunk(&mut png, b"IDAT", &zlib_stored(&raw));
    write_chunk(&mut png, b"IEND", &[]);
    png
}

/// Decode a PNG written by [`rgba_to_png`] back to `(width, height, rgba)`.
///
/// Deliberately narrow: it reads 8-bit RGBA, non-interlaced, filter-0 rows out of
/// stored (uncompressed) DEFLATE - exactly and only what this module writes. That
/// is enough for every consumer that exists (tools which montage, diff or measure
/// screenshots THIS emulator produced) and it keeps a general PNG/zlib decoder out
/// of the dependency set. Anything else is an ERROR naming what it found, never a
/// silent partial decode: a tool that quietly renders half an image is worse than
/// one that stops.
pub fn png_to_rgba(png: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    if png.len() < 8 || png[..8] != [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return Err("not a PNG (bad signature)".into());
    }
    let mut pos = 8;
    let (mut width, mut height) = (0u32, 0u32);
    let mut idat = Vec::new();
    let mut saw_ihdr = false;
    while pos + 8 <= png.len() {
        let len = u32::from_be_bytes([png[pos], png[pos + 1], png[pos + 2], png[pos + 3]]) as usize;
        let kind = &png[pos + 4..pos + 8];
        let body_at = pos + 8;
        let end = body_at.checked_add(len).ok_or("chunk length overflow")?;
        if end + 4 > png.len() {
            return Err(format!("truncated {} chunk", String::from_utf8_lossy(kind)));
        }
        let body = &png[body_at..end];
        match kind {
            b"IHDR" => {
                if len < 13 {
                    return Err("short IHDR".into());
                }
                width = u32::from_be_bytes([body[0], body[1], body[2], body[3]]);
                height = u32::from_be_bytes([body[4], body[5], body[6], body[7]]);
                let (depth, color, comp, filter, interlace) =
                    (body[8], body[9], body[10], body[11], body[12]);
                if (depth, color, comp, filter, interlace) != (8, 6, 0, 0, 0) {
                    return Err(format!(
                        "unsupported PNG: depth={depth} color={color} compression={comp} filter={filter} interlace={interlace} (only 8-bit RGBA, non-interlaced)"
                    ));
                }
                saw_ihdr = true;
            }
            b"IDAT" => idat.extend_from_slice(body),
            b"IEND" => break,
            _ => {}
        }
        pos = end + 4;
    }
    if !saw_ihdr {
        return Err("no IHDR chunk".into());
    }
    let raw = zlib_stored_inflate(&idat)?;
    let row_bytes = width as usize * 4;
    let expect = (row_bytes + 1) * height as usize;
    if raw.len() != expect {
        return Err(format!("image data is {} bytes, expected {expect}", raw.len()));
    }
    let mut rgba = Vec::with_capacity(row_bytes * height as usize);
    for y in 0..height as usize {
        let at = y * (row_bytes + 1);
        if raw[at] != 0 {
            return Err(format!("row {y} uses filter {} (only filter 0 is supported)", raw[at]));
        }
        rgba.extend_from_slice(&raw[at + 1..at + 1 + row_bytes]);
    }
    Ok((width, height, rgba))
}

/// Undo [`zlib_stored`]: read a zlib stream made only of stored DEFLATE blocks.
fn zlib_stored_inflate(z: &[u8]) -> Result<Vec<u8>, String> {
    if z.len() < 2 {
        return Err("zlib stream too short".into());
    }
    let mut out = Vec::new();
    let mut i = 2; // skip the 2-byte zlib header
    loop {
        if i + 5 > z.len() {
            return Err("truncated DEFLATE block header".into());
        }
        let header = z[i];
        if header & 0x06 != 0 {
            return Err(format!(
                "DEFLATE block type {} is compressed (this decoder reads only stored blocks)",
                (header >> 1) & 3
            ));
        }
        let final_block = header & 1 == 1;
        let n = u16::from_le_bytes([z[i + 1], z[i + 2]]) as usize;
        let inv = u16::from_le_bytes([z[i + 3], z[i + 4]]);
        if inv != !(n as u16) {
            return Err("stored block length check failed".into());
        }
        let at = i + 5;
        if at + n > z.len() {
            return Err("truncated stored block".into());
        }
        out.extend_from_slice(&z[at..at + n]);
        i = at + n;
        if final_block {
            return Ok(out);
        }
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
const FORMAT_F16: u8 = 8;
const FORMAT_F32: u8 = 9;

/// A vertex pulled out of the stream in its native form: the raw position lanes
/// (object space for the 3D path, screen pixels or NDC for the 2D path), plus the
/// texcoord and per-vertex color the fragment stage needs. Projection to the screen
/// happens per draw in [`render_scene`] according to the draw's [`Space`].
struct Vertex {
    pos: [f32; 3],
    uv: [f32; 2],
    color: [u8; 4],
    /// Object-space vertex normal (unit-ish), or `[0,0,0]` when the mesh carries none. The
    /// lit-material path brings it into world space (via the draw's model-to-world matrix)
    /// for the directional-light N.L term.
    normal: [f32; 3],
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
#[derive(Clone, Copy)]
struct Layout {
    pos_off: usize,
    pos_comps: usize,
    pos_fmt: u8,
    uv_off: Option<usize>,
    uv_fmt: u8,
    color_off: Option<usize>,
    /// The vertex normal attribute (byte offset + float format), for lighting. It is the
    /// lowest-offset >= 3-component float attribute that is NOT the position - the universal
    /// interleaved layout puts the normal right after the position (`pos, normal, [tangent],
    /// uv, color`). `None` when the mesh carries no such attribute (unlit / 2D geometry).
    normal_off: Option<usize>,
    normal_fmt: u8,
}

/// Whether an attribute format is a float lane the position/texcoord path decodes:
/// F32 (`9`) or F16 half-float (`8`). 3D meshes commonly store UV/normal as F16.
fn is_float_fmt(fmt: u8) -> bool {
    fmt == FORMAT_F32 || fmt == FORMAT_F16
}

fn layout_of(d: &Draw) -> Layout {
    // Float-lane attributes (F32 or F16), sorted by byte offset. The lowest-offset one
    // with >= 3 components is the position; a 2-component float attribute is the
    // texcoord (pos.xyz + uv.xy - the universal mesh layout, and pos.xy + uv.xy for a
    // 2D sprite). Formats are tracked per lane so an F16 texcoord decodes correctly.
    let mut floats: Vec<(usize, usize, u8)> = d
        .attributes
        .iter()
        .filter(|a| is_float_fmt(a.format) && a.component_count >= 2)
        .map(|a| (a.offset as usize, a.component_count as usize, a.format))
        .collect();
    floats.sort_unstable();
    // Position: the first attribute with >= 3 components, else the first of any.
    let pos = floats.iter().find(|(_, c, _)| *c >= 3).or_else(|| floats.first());
    let (pos_off, pos_comps, pos_fmt) = pos.copied().unwrap_or((0, 3, FORMAT_F32));
    // Texcoord: prefer a dedicated 2-component attribute (the classic pos.xyz + uv.xy
    // layout); otherwise fall back to the lowest-offset non-position float attribute
    // with >= 2 components, using its first two lanes. This title's world meshes pack
    // the texcoord as the xy of a float4 (uv + an unused pair), so a strict comp==2
    // match would miss it and draw the geometry untextured (a flat white fill).
    let uv = floats
        .iter()
        .find(|(o, c, _)| *c == 2 && *o != pos_off)
        .or_else(|| floats.iter().find(|(o, c, _)| *c >= 2 && *o != pos_off));
    let (uv_off, uv_fmt) = match uv {
        Some((o, _, f)) => (Some(*o), *f),
        None => (None, FORMAT_F32),
    };
    let color_off = d
        .attributes
        .iter()
        .find(|a| a.format == FORMAT_U8N && a.component_count >= 3)
        .map(|a| a.offset as usize);
    // Normal: the lowest-offset >= 3-component float attribute that is not the position
    // (position is `floats[0]` after the sort; the normal is the next such attribute).
    let (normal_off, normal_fmt) = match floats.iter().find(|(o, c, _)| *c >= 3 && *o != pos_off) {
        Some((o, _, f)) => (Some(*o), *f),
        None => (None, FORMAT_F32),
    };
    Layout { pos_off, pos_comps, pos_fmt, uv_off, uv_fmt, color_off, normal_off, normal_fmt }
}

/// Decode an IEEE-754 half-float (F16) to f32.
pub fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f32;
    let v = if exp == 0 {
        mant * 2f32.powi(-24)
    } else if exp == 0x1f {
        if h & 0x3ff == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (mant / 1024.0 + 1.0) * 2f32.powi(exp - 15)
    };
    if sign == 1 { -v } else { v }
}

/// Encode an f32 as an IEEE-754 half-float (F16), round-to-nearest-even, with overflow
/// saturating to infinity and underflow going through the subnormal range.
///
/// The inverse of [`half_to_f32`], and needed wherever the GUEST hands over a float that the
/// hardware stores at half width - `sceGxmSetUniformDataF` writing an F16-declared uniform is
/// the case that matters: the shader reads that register back as two packed halves.
pub fn f32_to_half(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        // Inf/NaN. A NaN must stay a NaN, so keep a non-zero mantissa.
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflows the half range
    }
    if e <= 0 {
        // Subnormal (or zero): shift the implicit leading 1 into the mantissa.
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        // Round to nearest, ties to even, on the bit shifted out.
        let round = u32::from((m >> (shift - 1)) & 1 == 1 && (m & ((1 << (shift - 1)) - 1) != 0 || half & 1 == 1));
        return sign | (half + round) as u16;
    }
    let half = ((e as u32) << 10) | (mant >> 13);
    let round = u32::from(mant & 0x1000 != 0 && (mant & 0x0fff != 0 || half & 1 == 1));
    sign | (half + round) as u16
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
    // Read one lane as F32 (4 bytes) or F16 (2 bytes) per the attribute's format, so
    // an F16-packed position or texcoord decodes to the right value. `stride_of` is the
    // per-lane byte step (4 for F32, 2 for F16).
    let lane = |off: usize, fmt: u8| -> f32 {
        let o = base + off;
        if fmt == FORMAT_F16 {
            if o + 2 <= d.vertices.len() {
                half_to_f32(u16::from_le_bytes([d.vertices[o], d.vertices[o + 1]]))
            } else {
                0.0
            }
        } else if o + 4 <= d.vertices.len() {
            f32::from_le_bytes([d.vertices[o], d.vertices[o + 1], d.vertices[o + 2], d.vertices[o + 3]])
        } else {
            0.0
        }
    };
    let pstep = if layout.pos_fmt == FORMAT_F16 { 2 } else { 4 };
    let px = lane(layout.pos_off, layout.pos_fmt);
    let py = lane(layout.pos_off + pstep, layout.pos_fmt);
    let pz = if layout.pos_comps >= 3 { lane(layout.pos_off + 2 * pstep, layout.pos_fmt) } else { 0.0 };
    let ustep = if layout.uv_fmt == FORMAT_F16 { 2 } else { 4 };
    let uv = match layout.uv_off {
        Some(o) => [lane(o, layout.uv_fmt), lane(o + ustep, layout.uv_fmt)],
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
    let normal = match layout.normal_off {
        Some(o) => {
            let nstep = if layout.normal_fmt == FORMAT_F16 { 2 } else { 4 };
            [lane(o, layout.normal_fmt), lane(o + nstep, layout.normal_fmt), lane(o + 2 * nstep, layout.normal_fmt)]
        }
        None => [0.0, 0.0, 0.0],
    };
    Vertex { pos: [px, py, pz], uv, color, normal }
}

/// Just the POSITION of vertex `i`, decoded exactly as [`decode_vertex`] decodes it.
///
/// For a draw that will be rendered with the guest's recompiled shaders, the canonical
/// vertex is dead (see `gxp_only`) and the only thing the per-vertex walk still produces
/// is the opaque depth RANGE - which reads the position and nothing else. Decoding the
/// uv, colour and normal for it is per-vertex work on a race frame's several hundred
/// thousand vertices, thrown away immediately.
fn decode_vertex_pos(d: &Draw, layout: &Layout, i: usize) -> [f32; 3] {
    let stride = d.vertex_stride.max(1) as usize;
    let base = i * stride;
    let lane = |off: usize, fmt: u8| -> f32 {
        let o = base + off;
        if fmt == FORMAT_F16 {
            if o + 2 <= d.vertices.len() {
                half_to_f32(u16::from_le_bytes([d.vertices[o], d.vertices[o + 1]]))
            } else {
                0.0
            }
        } else if o + 4 <= d.vertices.len() {
            f32::from_le_bytes([d.vertices[o], d.vertices[o + 1], d.vertices[o + 2], d.vertices[o + 3]])
        } else {
            0.0
        }
    };
    let pstep = if layout.pos_fmt == FORMAT_F16 { 2 } else { 4 };
    [
        lane(layout.pos_off, layout.pos_fmt),
        lane(layout.pos_off + pstep, layout.pos_fmt),
        if layout.pos_comps >= 3 { lane(layout.pos_off + 2 * pstep, layout.pos_fmt) } else { 0.0 },
    ]
}

/// Transform an object-space normal by the model-to-world matrix's upper 3x3 (column-major)
/// and normalize. Car/world parts use near-uniform scale, so the plain 3x3 (not the
/// inverse-transpose) is faithful. A zero/degenerate result falls back to `[0,1,0]` (up).
fn world_normal(n: [f32; 3], world: &[f32; 16]) -> [f32; 3] {
    let x = world[0] * n[0] + world[4] * n[1] + world[8] * n[2];
    let y = world[1] * n[0] + world[5] * n[1] + world[9] * n[2];
    let z = world[2] * n[0] + world[6] * n[1] + world[10] * n[2];
    let len = (x * x + y * y + z * z).sqrt();
    if len > 1e-6 {
        [x / len, y / len, z / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// The lit fragment colour for an opaque 3D draw: the standard forward-lit material the real
/// fragment programs run - `albedo * tint`, lit by one directional light (`saturate(N.L) *
/// light_col`) plus a flat `ambient` term - then scaled by scene `exposure` and Reinhard
/// tone-mapped so HDR light (this title's key light is ~2.8x) rolls off instead of clipping.
/// `n_world` is the interpolated, world-space surface normal (already normalized). Shared by
/// the software rasterizer and mirrored exactly in the WGSL `fs_opaque` so the two agree.
fn shade_lit(albedo: [f32; 3], n_world: [f32; 3], mat: &crate::capture::FragmentMaterial, exposure: f32) -> [u8; 3] {
    // Light direction is world-space "direction to the light"; N.L is clamped to the front.
    let l = {
        let d = mat.light_dir;
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len > 1e-6 { [d[0] / len, d[1] / len, d[2] / len] } else { [0.0, 1.0, 0.0] }
    };
    let ndotl = (n_world[0] * l[0] + n_world[1] * l[1] + n_world[2] * l[2]).max(0.0);
    let mut out = [0u8; 3];
    for ch in 0..3 {
        let base = albedo[ch] / 255.0 * mat.tint[ch];
        let light = mat.ambient[ch] + mat.light_col[ch] * ndotl;
        let l = base * light * exposure;
        // Reinhard tone-map keeps the HDR bright end from hard-clipping to flat white.
        out[ch] = (l / (1.0 + l) * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Project a vertex's raw position into screen space for the given draw `space`,
/// returning `[screen_x, screen_y, depth, 1/w]`, or `None` if the vertex is behind
/// the eye (perspective `w <= 0`) and the triangle must be dropped. `width`/`height` are
/// the RASTER dimensions (already scaled by any supersample factor). `ssaa` is that factor:
/// clip-space draws (Mvp/Ndc) fill the raster via `width`/`height` and so scale for free, but
/// a Pixel-space draw carries absolute native pixel coords, so they are multiplied by `ssaa`
/// to cover the enlarged raster - exactly what the GPU's Pixel path does by mapping through the
/// native `surf_w`/`surf_h` into a full-viewport clip space.
/// Project a vertex to screen space, returning `[x, y, depth, 1/w]`.
///
/// DEPTH IS `-1/w`, NOT the clip `z/w`. GXM/PowerVR SGX resolves visibility from the clip
/// `w` (the projected view distance), not from a normalized clip `z`, and this title proves
/// it: its vertex programs emit a clip `z` that is a linear combination of the `x`, `y` and
/// `w` rows, so `z/w` is a function of the SCREEN POSITION ALONE - measured identical to
/// nine digits across the ground and every car draw covering one pixel, while `w` correctly
/// separated them (car 20.81 in front of ground 21.62). A `z/w` depth buffer therefore
/// carries no depth at all for this title and rejects geometry essentially at random.
///
/// `-1/w` is the faithful choice, not merely a workaround: it is screen-linear (so it
/// interpolates exactly across a triangle, like any projection's depth), it increases with
/// view distance so the guest's own depth func keeps its meaning ("nearer wins" for a LESS
/// test against a `+INF` clear), and for a well-formed projection - where `z/w = A + B/w` -
/// it is an increasing affine function of that `z/w`, so it produces the IDENTICAL ordering
/// a correct clip `z` would. It agrees with a conventional title and rescues this one.
fn project(v: &Vertex, space: &Space, width: u32, height: u32, ssaa: f32) -> Option<[f32; 4]> {
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
            Some([sx, sy, -inv_w, inv_w])
        }
        Space::Ndc => {
            let sx = (v.pos[0] * 0.5 + 0.5) * wf;
            let sy = (1.0 - (v.pos[1] * 0.5 + 0.5)) * hf;
            Some([sx, sy, 0.0, 1.0])
        }
        // Screen pixels already, Y down - scaled to the (possibly supersampled) raster.
        Space::Pixel => Some([v.pos[0] * ssaa, v.pos[1] * ssaa, 0.0, 1.0]),
    }
}

/// SceGxmPrimitiveType selectors (the high-bit-encoded enum, gxm.h): triangle list,
/// strip and fan all rasterize to triangles. Lines and points rasterize to neither, and
/// the SOFTWARE path has no representation for them - but the recompiled GPU path does
/// (`gpu::gxm_topology`), so they are carried through rather than dropped.
const PRIM_TRIANGLES: u32 = 0x0000_0000;
const PRIM_LINES: u32 = 0x0400_0000;
const PRIM_POINTS: u32 = 0x0800_0000;
const PRIM_TRIANGLE_STRIP: u32 = 0x0C00_0000;
const PRIM_TRIANGLE_FAN: u32 = 0x1000_0000;
/// `SCE_GXM_PRIMITIVE_TRIANGLE_EDGES` - the sixth value of the enum. It rasterises the
/// EDGES of a triangle list, with `SceGxmEdgeEnableFlags` choosing which of the three
/// (0x100 / 0x200 / 0x400 for edges 01 / 12 / 20). The enum above stopped at five values,
/// so this arrived as an unnamed constant in a dropped-draw report on a phone.
///
/// # Where the flags live - MEASURED, not read from a header
/// No header we have says whether the edge flags are packed into the index words or carried
/// out of band, so the drop report printed the raw indices - and the guest's own buffer
/// answered: `[0x0, 0x1, 0x2, 0x700, 0x3, 0x4, 0x5, 0x500, 0x6, 0x7, 0x8, 0x300, ...]`.
/// Groups of FOUR words: three vertex indices, then a word carrying ONLY
/// `SceGxmEdgeEnableFlags` bits. So an edge list is expanded here into the LINE segments
/// its flags enable, and drawn as a `LineList` (`gpu::gxm_topology`). A draw whose fourth
/// words carry anything outside the three flag bits does not match that reading and is
/// still dropped - and reported with its words, exactly as the one above was.
const PRIM_TRIANGLE_EDGES: u32 = 0x1400_0000;

/// The three `SceGxmEdgeEnableFlags` bits, as they appear in an edge list's per-triangle
/// flags word: edge 01, edge 12, edge 20.
const EDGE_FLAG_BITS: u32 = 0x100 | 0x200 | 0x400;

/// Does this edge-list draw match the measured encoding - groups of four index words whose
/// fourth carries only `SceGxmEdgeEnableFlags` bits? A draw that does not is dropped and
/// reported rather than drawn under a reading its own buffer contradicts.
fn edge_list_matches_packed_reading(d: &Draw) -> bool {
    let groups = d.index_count as usize / 4;
    (0..groups).all(|g| index_at(d, g * 4 + 3) as u32 & !EDGE_FLAG_BITS == 0)
}

/// How many raw index words an edge-list drop prints. Enough to see the pattern of a few
/// triangles - the whole point is to read the flag bits, not to transcribe the buffer.
const EDGE_LIST_INDEX_DUMP: usize = 24;

/// Vertices per primitive for a topology the GPU can draw directly from the guest's own
/// index list, or `None` for one this renderer expands into triangles first.
fn direct_topology_stride(primitive: u32) -> Option<usize> {
    match primitive {
        PRIM_LINES => Some(2),
        PRIM_POINTS => Some(1),
        _ => None,
    }
}

/// SceGxmCullMode: which screen-space winding the GPU discards. NONE draws both
/// faces; CW/CCW discard clockwise/counter-clockwise triangles respectively.
const SCE_GXM_CULL_NONE: u32 = 0x0000_0000;
const SCE_GXM_CULL_CW: u32 = 0x0000_0001;
const SCE_GXM_CULL_CCW: u32 = 0x0000_0002;

/// SCE_GXM_DEPTH_WRITE_DISABLED - depth writes off (a 2D alpha overlay, not opaque 3D).
const SCE_GXM_DEPTH_WRITE_DISABLED: u32 = 0x0010_0000;
/// `SCE_GXM_FRAGMENT_PROGRAM_DISABLED` (vitasdk `gxm.h`) - the draw rasterises into DEPTH and
/// STENCIL only and writes no colour. `SCE_GXM_FRAGMENT_PROGRAM_ENABLED` is 0, which is also
/// the context default, so a title that never calls the setter is enabled throughout.
const SCE_GXM_FRAGMENT_PROGRAM_DISABLED: u32 = 0x0020_0000;

/// SCE_GXM_DEPTH_FUNC_LESS_EQUAL - GXM's default depth test, and what [`render_map`]
/// asks for explicitly: its depth buffer holds negated world height rather than a
/// projected z, so the draw's own recorded func does not apply.
const SCE_GXM_DEPTH_FUNC_LESS_EQUAL: u32 = 0x00C0_0000;

/// The number of triangles a draw's topology emits from its index count.
fn triangle_count(d: &Draw) -> usize {
    match d.primitive {
        PRIM_TRIANGLES => d.index_count as usize / 3,
        PRIM_TRIANGLE_STRIP | PRIM_TRIANGLE_FAN => (d.index_count as usize).saturating_sub(2),
        _ => 0,
    }
}

/// The three vertex indices of triangle `t`, with winding NORMALIZED so every
/// triangle presents the facing a triangle-list triangle would - a strip flips
/// winding on odd triangles, so the last two indices are swapped there to undo it.
/// Fill is winding-agnostic, but a consistent winding is what lets both the software
/// rasterizer and the GPU cull back faces uniformly (the GPU sees an expanded
/// triangle LIST, so the strip's alternation must be folded out here).
fn tri_indices(d: &Draw, t: usize) -> [usize; 3] {
    match d.primitive {
        PRIM_TRIANGLE_STRIP => {
            let (a, b, c) = (index_at(d, t), index_at(d, t + 1), index_at(d, t + 2));
            if t & 1 == 0 { [a, b, c] } else { [a, c, b] }
        }
        PRIM_TRIANGLE_FAN => [index_at(d, 0), index_at(d, t + 1), index_at(d, t + 2)],
        _ => [index_at(d, t * 3), index_at(d, t * 3 + 1), index_at(d, t * 3 + 2)],
    }
}

/// Whether a triangle with signed screen-space area `area` (from [`edge`], in the
/// Y-down framebuffer space [`project`] emits) is culled under `cull_mode`. The winding
/// sign is pinned EMPIRICALLY against this title's ground plane: the ground is a
/// `SCE_GXM_CULL_CCW` mesh whose camera-facing front faces must survive, and in the
/// Y-down screen space `project` emits (with strip winding normalized by `tri_indices`)
/// those front faces have POSITIVE `edge` area - so CCW-cull discards `area < 0` and
/// CW-cull discards `area > 0`. A near-zero area (an edge-on / degenerate triangle) is
/// left to the caller's separate degeneracy check.
fn cull_backface(area: f32, cull_mode: u32) -> bool {
    match cull_mode {
        SCE_GXM_CULL_CCW => area < 0.0,
        SCE_GXM_CULL_CW => area > 0.0,
        _ => false,
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
/// `SCE_GXM_TEXTURE_BASE_FORMAT_YUV420P2 >> 24` - see the definition for what it is. The
/// uploader needs the same constant, so it is defined once, on the side both can reach.
pub use vitaslop_platform::gpu::GXM_BASE_FORMAT_YUV420P2 as YUV420P2;

/// `SCE_GXM_TEXTURE_BASE_FORMAT_P4 >> 24` - four-bit paletted, two texels per byte.
pub const P4: u32 = 0x94;
/// `SCE_GXM_TEXTURE_BASE_FORMAT_P8 >> 24` - eight-bit paletted, one index per texel.
pub const P8: u32 = 0x95;
/// `SCE_GXM_TEXTURE_BASE_FORMAT_U8U8U8U8 >> 24` - what a paletted texture becomes once its
/// indices have been looked up through its colour table (see `expand_paletted_texture`). A
/// palette ENTRY is a 32-bit texel in the texture's own declared swizzle, so the expansion
/// copies entries verbatim and the swizzle rides along unchanged.
pub const U8U8U8U8: u32 = 0x0c;

/// How many entries a paletted base format's colour table holds, or `None` when the format is
/// not paletted.
pub fn palette_entries(base_format: u32) -> Option<u32> {
    match base_format {
        P4 => Some(16),
        P8 => Some(256),
        _ => None,
    }
}

/// Byte offset of texel `(x, y)` within one stored LEVEL, as a `(byte, nibble)` pair - the
/// nibble is `1` for the high half of the byte and only ever non-zero for a 4-bit format.
///
/// Shared by the palette expansion's source walk and its destination walk so the two cannot
/// address a level differently. `bits` is 4, 8 or 32.
pub fn texel_element(
    tex_type: u32,
    l: &LevelLayout,
    bits: u32,
    x: u32,
    y: u32,
) -> (usize, u32) {
    if swizzled_type(tex_type) {
        // Morton order over the TEXEL grid, power-of-two padded. Every width here is a whole
        // number of texels per element or a whole number of elements per texel, so the texel
        // index is the addressing unit for all three.
        let pw = l.width.next_power_of_two();
        let ph = l.height.next_power_of_two();
        let m = morton_index(x, y, pw, ph) as usize;
        return match bits {
            4 => (m / 2, (m % 2) as u32),
            _ => (m * (bits as usize / 8), 0),
        };
    }
    let row = (y * l.stride) as usize;
    match bits {
        4 => (row + (x / 2) as usize, x % 2),
        _ => (row + (x as usize) * (bits as usize / 8), 0),
    }
}

pub fn block_layout(base_format: u32) -> Option<(u32, u32, u32)> {
    Some(match base_format {
        // 8-bit single channel (U8/S8) and 8-bit paletted (P8 - one INDEX per texel).
        0x00 | 0x01 | 0x95 => (1, 1, 1),
        // P4: four-bit paletted, so a "block" is the two texels that share one byte. The
        // capture EXPANDS both paletted formats through their colour table before anything
        // downstream sees them (`expand_paletted_texture`), so this geometry only ever has to
        // size and address the INDEX data.
        0x94 => (2, 1, 1),
        // 16-bit packed (U4U4U4U4, U1U5U5U5, U5U6U5, ...).
        0x02..=0x0b => (1, 1, 2),
        // 24-bit three-channel (U8U8U8, S8S8S8).
        0x98 | 0x99 => (1, 1, 3),
        // U2F10F10F10: a 32-bit packed HDR format, numbered up with the odd-sized formats
        // rather than with the other 32-bit ones. One retail racer renders its whole world
        // pass into a colour surface of this format, so the texture it reads that pass back
        // through is exactly this - and an unsized format is DROPPED, not approximated.
        0x9a => (1, 1, 4),
        // 32-bit (U8U8U8U8, ..., F32) and 32-bit single (U32/S32).
        0x0c..=0x1a => (1, 1, 4),
        // 64-bit four/two-channel: F16F16F16F16, U16U16U16U16, S16S16S16S16, F32F32, U32U32.
        0x1b..=0x1f => (1, 1, 8),
        // PVRTC (PVRTC1 and PVRTC2): 8-byte blocks covering 4x4 texels at 4bpp and 8x4 at
        // 2bpp. Unlike BC, a block is not decodable on its own - see `crate::pvrtc` - but its
        // GEOMETRY is what sizing and addressing need, and that is all this reports.
        0x80 | 0x82 => (8, 4, 8),
        0x81 | 0x83 => (4, 4, 8),
        // BC1 (DXT1) and BC4 (both signs): 8-byte 4x4 blocks.
        0x85 | 0x88 | 0x89 => (4, 4, 8),
        // BC2 (DXT3), BC3 (DXT5), BC5 (both signs): 16-byte 4x4 blocks.
        0x86 | 0x87 | 0x8a | 0x8b => (4, 4, 16),
        // YUV420P2: two PLANES, not one surface. What is reported here is the geometry of the
        // luma plane alone - one byte per texel - because that is what stride and addressing
        // are in terms of. The second plane is accounted for in [`level_layout`], which is the
        // only place that has to know the whole thing's size.
        YUV420P2 => (1, 1, 1),
        _ => return None,
    })
}

/// Round `v` up to the next multiple of `to` (a power of two).
fn align_up_to(v: u32, to: u32) -> u32 {
    v.div_ceil(to) * to
}

/// The guest memory layout of ONE mip level of a texture: its own dimensions, the bytes
/// per block-row, and the total bytes the level occupies.
///
/// Level `l` of a `width x height` image is `max(1, width >> l) x max(1, height >> l)`, laid
/// out exactly the way level 0 is - which is why this is one function rather than the level-0
/// arithmetic written out at the snapshot site plus a second copy for the chain. A SWIZZLED
/// (Morton) level covers a power-of-two-padded BLOCK grid; a LINEAR one is row-major, with
/// uncompressed rows padded to a multiple of 8 texels (the GXM linear alignment) and
/// compressed rows block-packed.
///
/// `None` for a format [`block_layout`] cannot size.
pub fn level_layout(base_format: u32, tex_type: u32, width: u32, height: u32, level: u32) -> Option<LevelLayout> {
    let (block_w, block_h, block_bytes) = block_layout(base_format)?;
    let w = (width >> level).max(1);
    let h = (height >> level).max(1);
    if base_format == YUV420P2 {
        // Two planes, each laid out linearly with GXM's 8-texel row alignment: `h` rows of
        // luma at one byte a texel, then `h/2` rows of interleaved Cb/Cr at two bytes per
        // chroma sample. `stride` stays the LUMA stride, because that is what addressing a
        // texel is in terms of; only `bytes` has to cover both planes, and it is what decides
        // how much guest memory is snapshotted - too little and the chroma half of every
        // frame is whatever was in the buffer before.
        let luma_stride = align_up_to(w, 8);
        let chroma_stride = align_up_to(w.div_ceil(2), 8) * 2;
        return Some(LevelLayout {
            width: w,
            height: h,
            blocks_x: w,
            blocks_y: h,
            stride: luma_stride,
            bytes: luma_stride * h + chroma_stride * h.div_ceil(2),
        });
    }
    let blocks_x = w.div_ceil(block_w);
    let blocks_y = h.div_ceil(block_h);
    let (stride, bytes) = if swizzled_type(tex_type) {
        let padded_x = blocks_x.next_power_of_two();
        let padded_y = blocks_y.next_power_of_two();
        (padded_x * block_bytes, padded_x * padded_y * block_bytes)
    } else {
        let row_blocks = if block_w == 1 { align_up_to(w, 8) } else { blocks_x };
        (row_blocks * block_bytes, row_blocks * block_bytes * blocks_y)
    };
    Some(LevelLayout { width: w, height: h, blocks_x, blocks_y, stride, bytes })
}

/// One mip level's geometry in guest memory - see [`level_layout`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelLayout {
    /// The level's own texel dimensions (`max(1, base >> level)`).
    pub width: u32,
    pub height: u32,
    /// Blocks needed to COVER those dimensions. For a swizzled level the stored grid is
    /// each of these rounded up to a power of two, which is why `bytes` is not simply
    /// `blocks_x * blocks_y * block_bytes`.
    pub blocks_x: u32,
    pub blocks_y: u32,
    /// Bytes per stored block-row.
    pub stride: u32,
    /// Bytes the whole level occupies.
    pub bytes: u32,
}

/// The most mip levels a `width x height` image can have, counting level 0: the chain ends
/// at the level whose larger dimension is 1.
pub fn max_mip_levels(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

/// Byte offset of level `level` from the start of ONE face, and the sum of the whole
/// `levels`-long chain when `level == levels`. Levels are stored consecutively, largest
/// first. That ordering is a claim about the DEVICE and was MEASURED, not assumed: a real mip
/// level is approximately a box filter of the one above it, and on this title's 2048x2048 atlas
/// the mean absolute difference against that reference runs 2.4-6.6 for nine levels against
/// 22-30 for an X-flipped control. See [[vitaslop-guest-mip-chain-layout]].
pub fn level_offset(base_format: u32, tex_type: u32, width: u32, height: u32, level: u32) -> Option<u32> {
    let mut off = 0u32;
    for l in 0..level {
        off += level_layout(base_format, tex_type, width, height, l)?.bytes;
    }
    Some(off)
}

/// One mip level of `t`, as a standalone single-face [`BoundTexture`] of that level's own
/// dimensions.
///
/// Every decoder here reads `width`, `height`, `stride` and `face_bytes` off the texture and
/// nothing else, so a level presented this way decodes through EXACTLY the same code as a
/// level-0 texture of the same size - which is the point. The alternative was threading a level
/// index through `texel_rgba_face`, `decode_face_fast`, `texel_byte_offset` and the PVRTC path,
/// four places that would then each have to be right.
///
/// `None` when `t` does not hold that level, or when the format cannot be sized.
pub fn level_view(t: &BoundTexture, face: u32, level: u32) -> Option<BoundTexture> {
    let l = level_layout(t.base_format, t.tex_type, t.width, t.height, level)?;
    let off = (face * t.face_bytes + level_offset(t.base_format, t.tex_type, t.width, t.height, level)?) as usize;
    let end = off.checked_add(l.bytes as usize)?;
    if level >= t.levels || end > t.pixels.len() {
        return None;
    }
    Some(BoundTexture {
        width: l.width,
        height: l.height,
        stride: l.stride,
        faces: 1,
        face_bytes: l.bytes,
        levels: 1,
        pixels: Arc::from(&t.pixels[off..end]),
        ..t.clone()
    })
}

/// The guest base formats that ARE a WebGPU block format, bit for bit.
///
/// `UBC1`/`UBC2`/`UBC3` are BC1/BC2/BC3 (DXT1/3/5) - the same 4x4 blocks with the same
/// endpoint and index encoding, which is why [`decode_bc_texel`] decodes all three with the
/// stock algorithm. BC4/BC5 (`0x88`..`0x8b`) are deliberately absent: they are one- and
/// two-channel formats, and this decoder expands them into RGBA on its own terms, so handing
/// the blocks to a `Bc4`/`Bc5` view would change which channel a shader reads. That is a real
/// piece of work, not an oversight, and it is worth ~0 MB on the measured frame.
fn block_format_for(base_format: u32) -> Option<BlockFormat> {
    Some(match base_format {
        0x85 => BlockFormat::Bc1,
        0x86 => BlockFormat::Bc2,
        0x87 => BlockFormat::Bc3,
        _ => return None,
    })
}

/// Lay the guest's own compressed blocks out the way a WebGPU upload reads them, or return
/// `None` with a REPORT naming which condition stopped it.
///
/// # The four conditions, and why each one is a hard gate rather than a best effort
/// 1. **The format must be one WebGPU has** ([`block_format_for`]).
/// 2. **The channel swizzle must be the identity.** The swizzle is applied DURING the decode
///    (`swizzle4` at the end of `decode_face_fast`) and there is no shader path for it, so a
///    permuted texture handed over raw renders with its channels wrong - the failure that looks
///    like a lighting or alpha bug and is neither.
/// 3. **Both dimensions must be a multiple of the 4x4 block.** WebGPU refuses to CREATE a
///    compressed texture whose size is not, so this one fails loudly either way; gating here
///    turns a device-lost into a decode.
/// 4. **The guest must actually have a mip chain**, unless the texture is small enough to have
///    no chain at all. There is no box filter for a compressed block, so a passthrough without
///    the guest's levels ships level 0 alone, and that is the "distant road reads as white
///    speckle" failure ([[vitaslop-textures-need-mips]]) - trading a memory defect for an image
///    defect, which is not a trade this makes silently.
///
/// A cube map is excluded for the same reason its chain is not snapshotted: how six chains
/// interleave in guest memory is not established.
fn compressed_source(t: &BoundTexture, force_format: Option<BlockFormat>) -> Option<CompressedUpload> {
    // Nothing here is worth doing on a GPU that cannot take the result - and this runs on the
    // texture decode path of every engine, including one whose adapter exposes ASTC and ETC2
    // but not BC.
    if !vitaslop_platform::gpu::block_compression_available() || !compression_enabled() {
        return None;
    }
    // The lossless handover first; the lossy re-encode is the fallback, never the preference.
    // EVERY reason the passthrough declines - an unrepresentable format, a permuted channel
    // swizzle, a missing mip chain - is a reason the transcode can handle, because it decodes to
    // RGBA8 first and owns the encode from there. Routing refusals here rather than to the plain
    // decode is worth 45 MB of one measured race frame: two 4096x2048 UBC2 surfaces that declare
    // one mip level while asking the hardware to filter between levels, which the passthrough
    // must refuse and the transcode can give a full chain.
    // The passthrough is not tiered: handing over blocks the guest already built costs nothing
    // to produce, so there is no work to spread out and no reason to ship a soft version first.
    passthrough_source(t).or_else(|| transcoded_source(t, force_format))
}

/// How fast the block encoder this adapter needs actually runs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum EncodeRate {
    /// BC, measured at 95 Mtexel/s. A whole 2048x2048 atlas is about 60 ms - a hitch on a first
    /// sight, once, on a machine that has the headroom.
    Fast,
    /// ETC2, measured at 12.2 Mtexel/s opaque and 4.1 with alpha on a desktop, and about
    /// 1 Mtexel/s on the mobile device this path exists for. A 2048x2048 atlas is SECONDS there.
    Slow,
}

fn block_encode_rate() -> EncodeRate {
    match vitaslop_platform::gpu::block_family() {
        BlockFamily::Etc2 => EncodeRate::Slow,
        _ => EncodeRate::Fast,
    }
}

/// Texels of a single inline encode that a frame will tolerate on the SLOW encoder.
///
/// Four million covers every texture up to 1024x1024 with its chain, which is nearly all of them;
/// past that the encode is refused and the texture is decoded at full resolution instead. The
/// number bounds a HITCH, not a steady cost - it is paid once, the first time a texture is seen.
///
/// It is not a quality control and must never become one: nothing on either side of this test
/// changes what the picture looks like, only whether those bytes reach the GPU compressed or
/// expanded.
const INLINE_ENCODE_TEXEL_LIMIT: u64 = 4 << 20;

/// The lossless half of [`compressed_source`]: the guest's own blocks, or `None` with a report.
fn passthrough_source(t: &BoundTexture) -> Option<CompressedUpload> {
    let why = |reason: &'static str| -> Option<CompressedUpload> {
        report_passthrough_refused(t.base_format, reason);
        None
    };
    // PVRTC has no WebGPU format on any adapter, so it can never pass through. Not reported as a
    // refusal - there is nothing to fix about it, and a line that fires for every PVRTC texture
    // in the title would bury the refusals that ARE actionable.
    let format = block_format_for(t.base_format)?;
    // >>> A PASSTHROUGH THIS ADAPTER CANNOT TAKE IS NOT A PASSTHROUGH, IT IS A DEAD END.
    //
    // The guest's `UBC1/2/3` are BC, and this returned a BC upload for them on EVERY adapter.
    // The uploader then discards it by family (`compressed_upload`) and decodes to RGBA8. That is
    // only a wasted block copy on a desktop - but `compressed_source` is `passthrough.or_else
    // (transcode)`, so returning `Some` here also meant the TRANSCODE never ran, and the whole
    // "re-encode a BC texture to ETC2 when the budget is tight" path in `transcoded_source` was
    // unreachable on exactly the device it was written for. The refusal it prints, the budget
    // pressure it consults, and the tests around it all described a branch nothing could enter.
    //
    // The family is the DEVICE's answer and the transcode's answer is the same object read the
    // other way, so asking it here costs one atomic load and makes the `or_else` mean what it
    // reads as. Not reported: an ETC2-only adapter would print this for every BC texture in the
    // title, and the transcode that follows reports what it did.
    if format.family() != vitaslop_platform::gpu::block_family() {
        return None;
    }
    if (t.swizzle >> 12) & 0x7 != 0 {
        return why("the channel swizzle is not the identity, and it is applied during the decode");
    }
    if t.faces != 1 {
        return why("it is a cube map, whose six mip chains' interleaving is not established");
    }
    if t.width % 4 != 0 || t.height % 4 != 0 {
        return why("its size is not a multiple of the 4x4 block, which WebGPU requires");
    }
    // >>> "NO CHAIN" AND "NO MIP FILTER" ARE DIFFERENT FACTS, AND ONLY ONE OF THEM BLOCKS THIS.
    //
    // A texture the guest gave one level AND told the hardware not to filter between levels is
    // one the DEVICE samples from its base level alone. Handing over that single level is then
    // the faithful answer, not a shortcut - what would be lost is a chain we invented, which the
    // Vita never had. MEASURED on a race frame: this is one 4096x2048 UBC2 surface priced at
    // 42.7 MB as RGBA8 against 8 MB of blocks, i.e. the single largest texture in the frame.
    //
    // A texture with one level and mip filtering ON is the opposite case and stays refused: the
    // hardware IS interpolating between levels, so dropping to level 0 is the "distant road
    // reads as white speckle" failure ([[vitaslop-textures-need-mips]]).
    let full_chain = max_mip_levels(t.width, t.height);
    if t.levels < 2 && full_chain > 1 && t.mip_filter != 0 {
        return why(
            "the guest declares no mip chain for it yet asks the hardware to FILTER between \
             levels, and a compressed level 0 on its own is the white-speckle failure a \
             generated chain exists to prevent",
        );
    }
    // `stride / block_bytes` is what the LINEAR decoder uses for its row pitch, so it is what
    // this uses too - a second, independently written row walk is how two readers of the same
    // bytes come to disagree.
    let bb = format.block_bytes();
    let swizzled = swizzled_type(t.tex_type);
    let mut data: Vec<u8> = Vec::with_capacity(t.face_bytes as usize);
    for level in 0..t.levels {
        let l = level_layout(t.base_format, t.tex_type, t.width, t.height, level)?;
        let base = (level_offset(t.base_format, t.tex_type, t.width, t.height, level)?) as usize;
        let (pw, ph) = (l.blocks_x.next_power_of_two(), l.blocks_y.next_power_of_two());
        for by in 0..l.blocks_y {
            for bx in 0..l.blocks_x {
                // The ONLY transformation between guest memory and the GPU: a swizzled texture
                // stores its blocks in Morton order over a power-of-two-padded grid, and WebGPU
                // wants them in linear block rows. It permutes whole blocks and touches no bit
                // inside one, so it is still a passthrough - the picture the GPU decodes is the
                // picture the hardware decodes.
                let index = if swizzled { morton_index(bx, by, pw, ph) } else { by * (l.stride / bb) + bx };
                let off = base + (index * bb) as usize;
                match t.pixels.get(off..off + bb as usize) {
                    Some(block) => data.extend_from_slice(block),
                    // Unreachable given the snapshot sized the read from the same arithmetic -
                    // and if that ever stops being true, a short read must not become a texture
                    // of stale blocks.
                    None => return why("its snapshot is shorter than its own block layout"),
                }
            }
        }
    }
    Some(CompressedUpload {
        format,
        width: t.width,
        height: t.height,
        data: vitaslop_platform::gpu::CompressedData::Cpu(Arc::new(data)),
        levels: t.levels,
        transcoded: false,
    })
}

/// The guest's bytes and layout for a texture the GPU can expand itself, or `None` when this
/// format or shape is not one the shader covers.
///
/// # Which formats, and why the list is short on purpose
/// Only the 32-bit four-channel family whose entire decode is `swizzle4` over the four memory
/// bytes - `0x0c` and the three siblings that reach the same arm of
/// [`decode_uncompressed_at`]. Every other uncompressed format does arithmetic (a 10-bit lane
/// normalised, a half float widened, a signed lane biased), and porting those one at a time
/// would trade a provable equality for a set of measured ones.
///
/// `0x0c` alone is what matters: the target device's own report reads `texture decode by
/// format: 2988.8 MB total - 0x0c raw 2964.5 MB`.
///
/// # What the shader is given
/// The same addressing the CPU uses, from the same functions - `level_offset`/`level_layout`
/// and the power-of-two padding a swizzled level's Morton order runs over. A second
/// implementation of that addressing is exactly the kind of duplicate that drifts, and its
/// failure mode is a texture read plausibly out of the wrong bytes.
pub fn raw_source(t: &BoundTexture) -> Option<vitaslop_platform::gpu::GpuRawExpand> {
    // Byte-permutation formats only. These are the arms `decode_uncompressed_at` answers with a
    // bare `swizzle4(byte(0), byte(1), byte(2), byte(3), swizzle)` - the specials inside the
    // same numeric range (two-lane 16-bit, packed floats, depth+stencil) are matched ahead of it
    // there and must be matched ahead of it here.
    if !matches!(t.base_format, 0x0c | 0x0d | 0x14 | 0x16) {
        return None;
    }
    // One face only, like the transcode: a cube map's six chains' interleaving is not
    // established, and the CPU path refuses them for the same reason.
    if t.faces != 1 {
        return None;
    }
    let (w, h) = (t.width.max(1), t.height.max(1));
    let swizzled = swizzled_type(t.tex_type);
    let mut src_levels = Vec::new();
    // >>> LEVEL 0 ONLY, BECAUSE THE CPU PATH THIS REPLACES USES LEVEL 0 ONLY.
    //
    // `build_mip_chain` box-filters the whole chain down from the decoded level 0 and never
    // looks at the guest's own levels; the GPU `halve` shader is that same filter, arithmetic
    // for arithmetic. Feeding the guest's levels in here instead would produce a DIFFERENT
    // picture - probably a more faithful one, since the hardware samples exactly those - but
    // this change is a cost change and has to be bit-identical to be judged as one. The guest's
    // own mips are a separate question with its own evidence to gather.
    for level in 0..1u32.min(t.levels.max(1)) {
        let l = level_layout(t.base_format, t.tex_type, w, h, level)?;
        let off = level_offset(t.base_format, t.tex_type, w, h, level)?;
        // A level the guest's allocation does not actually reach is not a level - the CPU path
        // discovers this through `level_view` returning `None` and box-filters from the level
        // above instead, and stopping here is the same rule.
        if (off as usize).saturating_add(l.bytes as usize) > t.pixels.len() {
            break;
        }
        src_levels.push(vitaslop_platform::gpu::SrcLevel {
            byte_offset: off,
            width: l.width,
            height: l.height,
            // For an uncompressed level the "block" is one texel, so this carries the ROW
            // STRIDE IN TEXELS, which is what a LINEAR level addresses through.
            blocks_x: if level == 0 { t.stride / 4 } else { l.width },
            blocks_y: l.height,
            padded_x: if swizzled { l.width.next_power_of_two() } else { l.width },
            padded_y: if swizzled { l.height.next_power_of_two() } else { l.height },
            swizzled,
        });
    }
    if src_levels.is_empty() {
        return None;
    }
    Some(vitaslop_platform::gpu::GpuRawExpand {
        src: t.pixels.clone(),
        width: w,
        height: h,
        levels: max_mip_levels(w, h),
        swizzle: (t.swizzle >> 12) & 0x7,
        src_levels,
        codec: None,
    })
}

/// The same plan for a texture the guest gave us as BC BLOCKS, so the expansion to RGBA8 runs
/// on the GPU instead of the CPU. `None` when this texture is not one of them.
///
/// # Why this exists beside the transcode, and why it is not the transcode
/// A GPU with no BC support cannot take the guest's blocks verbatim, and `transcoded_source`
/// deliberately declines to re-encode a BC texture that already fits the budget: ETC2 is a
/// second LOSSY step over the guest's own compression and buys only megabytes. So such a
/// texture fell through to the CPU decode - and on the device that is the largest thing left.
/// MEASURED there: one frame's texture working set is 130 MB with `0x85 BC -> RGBA8 x136
/// (98.1 MB)`, and the run's slowest frames are ~600 ms at ORDINARY host-call counts, which
/// makes them ours rather than the guest's.
///
/// This changes where that decode runs, not what it produces. Same RGBA8, same generated mip
/// chain, same memory - so it trades no image quality, which is the whole reason it can be
/// unconditional where the ETC2 re-encode cannot be.
/// [[vitaslop-never-trade-quality]] [[vitaslop-phone-gpu-has-no-bc]]
pub fn block_source(t: &BoundTexture) -> Option<vitaslop_platform::gpu::GpuRawExpand> {
    use vitaslop_platform::gpu::SourceCodec;
    if !matches!(t.base_format, 0x85 | 0x86 | 0x87) {
        return None;
    }
    // A cube map's six chains interleave in a way that is not established - the same refusal
    // the passthrough and the transcode both make.
    if t.faces != 1 {
        return None;
    }
    // >>> NO CHANNEL SWIZZLE. `decode_bc` spends `src_format` on the BASE FORMAT, so it has
    // nowhere to carry a SWIZZLE4 selector and does not apply one; the CPU decoder does. A
    // swizzled texture must therefore keep the CPU path or it renders with its channels in the
    // wrong order. `gpu_transcode` refuses it for exactly this reason.
    if (t.swizzle >> 12) & 0x7 != 0 {
        return None;
    }
    let (w, h) = (t.width.max(1), t.height.max(1));
    let swizzled = swizzled_type(t.tex_type);
    let mut src_levels = Vec::new();
    // >>> LEVEL 0 ONLY, AND EVERY LEVEL BELOW IT GENERATED - because that is what the CPU path
    // >>> this replaces does, and this change is about WHERE the decode runs, nothing else.
    //
    // Taking the guest's OWN declared levels here instead would be a quiet change of mip
    // policy: `decode_texture_rgba8_counted` decodes level 0 and `build_mip_chain` box-filters
    // the rest, so a texture whose guest chain differs from a generated one would start
    // sampling differently the moment this path claimed it - visible only as a minified
    // surface changing, on the device, with nothing to say why. Whether the guest's chain
    // should be preferred is a real question and it already has a history
    // (`mips_for_texture`: tried 2026-08-28b and REVERTED); it does not get decided as a side
    // effect of moving a decode onto the GPU. `raw_source` takes level 0 only for the same
    // reason.
    for level in 0..1u32 {
        let l = level_layout(t.base_format, t.tex_type, w, h, level)?;
        let off = level_offset(t.base_format, t.tex_type, w, h, level)?;
        if (off as usize).saturating_add(l.bytes as usize) > t.pixels.len() {
            break;
        }
        src_levels.push(vitaslop_platform::gpu::SrcLevel {
            byte_offset: off,
            width: l.width,
            height: l.height,
            blocks_x: l.blocks_x,
            blocks_y: l.blocks_y,
            // A swizzled block image is Morton-addressed over BLOCKS, so the padding is the
            // block count rounded up - not the texel count. Same as `gpu_transcode`.
            padded_x: if swizzled { l.blocks_x.next_power_of_two() } else { l.blocks_x },
            padded_y: if swizzled { l.blocks_y.next_power_of_two() } else { l.blocks_y },
            swizzled,
        });
    }
    if src_levels.is_empty() {
        return None;
    }
    Some(vitaslop_platform::gpu::GpuRawExpand {
        src: t.pixels.clone(),
        width: w,
        height: h,
        levels: max_mip_levels(w, h),
        // Unused by the block decoder - see the refusal above.
        swizzle: 0,
        src_levels,
        codec: Some(SourceCodec::Bc { base_format: t.base_format }),
    })
}

/// Whether compressed textures reach the GPU compressed at all. ON, and the only reason to set
/// `VITASLOP_TEX_COMPRESS=0` is to A/B against the plain decode.
///
/// # ONE knob, and it only turns the feature OFF
/// This was three: a passthrough switch, a transcode switch defaulting to OFF, and a mip probe.
/// That is three forks the default path never takes, and the transcode one was worse than
/// surface area - it put the large half of the win (232 MB -> 57 MB, at a measured mean error of
/// 1.78/255) behind a flag that the engine the work exists for would never have set. A knob is
/// not a way to defer a decision: if the measurement says it is better, it is the default.
///
/// Cached, like every knob on this path: this is tested once per distinct texture, and reading
/// an unset environment variable on Windows copies and re-encodes the whole environment block.
fn compression_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_TEX_COMPRESS").ok().as_deref() != Some("0"))
}

/// Re-encode a texture WebGPU has no format for (PVRTC), or one the passthrough refused, so it
/// can live on the GPU compressed instead of landing there as RGBA8 at eight times its size.
///
/// # The trade, stated plainly, and why it is the DEFAULT
/// PVRTC is 172.4 MB of a measured 274 MB race frame - the largest item by far, and the reason a
/// phone's allocation fails and the draw comes out WHITE. It is also the one family that CANNOT
/// pass through: no WebGPU adapter exposes a PVRTC format, so the choice is RGBA8 or a
/// re-encode, and a re-encode means PVRTC -> RGBA8 -> BC, two lossy block codecs in series.
///
/// MEASURED against the plain decode on that frame: the working set goes **232 MB -> 57 MB**,
/// the wall clock does not move (62.2 s against 63.9 s - the encode is one-off per texture and
/// pays for itself in evictions not taken), and the picture costs **mean 1.78 of 255,
/// PSNR 38.2 dB, 99.7% of channels within 16 levels** - indistinguishable side by side at
/// 960x544. An 8x memory cut for an error smaller than the dithering already in the source is
/// not a decision worth deferring to a flag nobody sets.
///
/// It is not "unfaithful" in any direction that was available: the Vita samples PVRTC directly
/// and we cannot, so the real choice is between a picture that is slightly re-quantised and one
/// that does not fit in memory.
///
/// # It supplies a full chain whichever way the guest stored the texture
/// Unlike the passthrough, this path OWNS the encode, so a texture the guest left mipless can
/// still be given a box-filtered chain - built in RGBA8, where a box filter is meaningful, and
/// encoded level by level. That removes the one condition that refuses a passthrough outright.
fn transcoded_source(t: &BoundTexture, force_format: Option<BlockFormat>) -> Option<CompressedUpload> {
    // Only formats that are ALREADY block-compressed on the guest. Re-encoding an uncompressed
    // texture would be a pure quality loss to save memory the title never asked to spend, and
    // the uncompressed formats are 0.8 MB of a 274 MB frame - there is nothing there to win.
    if crate::pvrtc::Variant::from_base_format(t.base_format).is_none()
        && block_format_for(t.base_format).is_none()
    {
        return None;
    }
    let why = |reason: &'static str| -> Option<CompressedUpload> {
        report_passthrough_refused(t.base_format, reason);
        None
    };
    if t.faces != 1 {
        return why("it is a cube map, whose six mip chains' interleaving is not established");
    }
    if t.width % 4 != 0 || t.height % 4 != 0 {
        return why("its size is not a multiple of the 4x4 block, which WebGPU requires");
    }
    // >>> THE GPU BUILDS THE BLOCKS WHEN IT CAN, AND THEN NOTHING BELOW THIS RUNS.
    //
    // Everything after this point - the per-level decode, the box filter, the block encode - is
    // the work that MEASURED at `BUILD 21,182 ms` on one transition frame of the target device,
    // while its GPU sat at `pass 3.3 ms`. `gpu_transcode` produces the same chain from the same
    // bytes in compute shaders and hands back a plan instead of megabytes, so a texture that
    // takes this path costs one buffer write of the guest's own compressed bytes and nothing
    // else on the CPU.
    //
    // >>> IT IS ASKED FIRST, AND THAT MOVES BOTH REFUSALS BELOW OUT OF ITS WAY.
    //
    // Both of them are statements about the CPU ENCODER - one says a BC texture is not worth
    // re-encoding because the CPU cost outweighs the megabytes, the other says a large texture
    // cannot be encoded inside a frame at all. Neither is true of a compute shader, and leaving
    // them in front of this would have kept the two most expensive cases in the title - the big
    // atlases, and every BC texture on an adapter that cannot take BC - on the CPU path they
    // exist to describe. They still guard the CPU fallback, which is exactly what they are about.
    if let Some(plan) = gpu_transcode(t, force_format) {
        return Some(plan);
    }

    // >>> NO COMPRESSED FORMAT IS RE-ENCODED UNLESS THE BUDGET IS ACTUALLY TIGHT, AND THAT
    // >>> NOW INCLUDES PVRTC.
    //
    // `UBC1/2/3` ARE BC1/2/3: they pass through untouched on any desktop, and on an ETC2-only
    // adapter re-encoding them ON THE CPU is the most expensive path there is (decode the blocks
    // to RGBA8, then encode ETC2 *with* an EAC alpha block).
    //
    // MEASURED on the device once the PVRTC half landed: the race frame's working set fell to
    // **82 MB against a 256 MB budget**, of which 82 MB WAS those BC textures sitting as RGBA8.
    // They already fit. Spending the scarcest resource on this machine to shrink something that
    // fits is the trade backwards, so it is made only while the budget is actually tight.
    //
    // >>> PVRTC IS EXEMPT FROM THAT TEST, AND EXTENDING THE TEST TO IT WAS TRIED AND MEASURED
    // >>> AND IS WRONG. DO NOT RE-PROPOSE IT.
    //
    // The argument for extending it looks airtight on paper, which is why it is written out
    // here rather than left to be re-derived. It runs: PVRTC's exemption rests on "its only
    // alternative is an eight-fold expansion", but BC1 expands by the same 8x; both fallbacks
    // are lossless with respect to the guest's asset; and the re-encode is a SECOND lossy step
    // on an already-lossy one, which the GPU-encoder gate 150 lines below refuses in as many
    // words ("a cheap encoder is a reason to spend CPU, never a reason to spend QUALITY").
    // Every one of those statements is true.
    //
    // **MEASURED 2026-08-28b on PCSA00015's campaign race, and the trade is not close:
    // one frame's texture working set went 62 MB -> 207 MB, 3.3x.** The picture did improve -
    // 48% of pixels at max delta 15 across the whole front end, which is exactly the BC1
    // re-encode error disappearing - but a delta of 15 is not worth 145 MB on a device where
    // going over the budget is not a slow frame, it is unbounded GPU memory and a worker the
    // browser kills with no error, no crash event and no log line (see
    // `report_texture_budget_exceeded`).
    //
    // **The asymmetry the original wording asserted loosely is real and QUANTITATIVE**: titles
    // use PVRTC for nearly everything (this race binds 99 PVRTC textures against 8 BC ones), so
    // "the same 8x" applies to a completely different share of the working set. That is what
    // makes the BC case affordable and this one not.
    if block_format_for(t.base_format).is_some() && !vitaslop_platform::gpu::texture_budget_pressure() {
        return why(
            "it is a BC format that already fits the texture budget as RGBA8, and re-encoding it \
             to ETC2 costs a block decode plus an alpha-carrying encode - CPU this device needs \
             more than it needs the megabytes. It will be re-encoded if the budget tightens",
        );
    }
    // >>> AN ENCODE TOO BIG FOR A FRAME IS REFUSED. IT IS NEVER MADE SMALLER.
    //
    // A screen transition binds a hundred textures at once, and this encoder runs at about
    // 1 Mtexel/s on the device it exists for, so encoding a 2048x2048 atlas inline is seconds of
    // frozen guest. The previous answer was to encode a REDUCED-resolution version instead and
    // grow it later. That is a quality trade and it does not belong here: on the device it never
    // got past 128 texels a side, so an atlas rendered at a sixteenth of its resolution per axis.
    //
    // The picture is not the variable. If an encode cannot be afforded, the texture takes the
    // ordinary decode path at the guest's own resolution and costs memory instead - which is
    // what the texture budget and its eviction are for. Fidelity is fixed; memory is managed.
    //
    // The real fix for the cost is a RESUMABLE encode - one texture's blocks spread over frames
    // at full resolution - and until that exists this refusal is the honest behaviour rather
    // than a softer picture.
    let full_texels = (t.width as u64 * t.height as u64 * 4) / 3;
    if full_texels > INLINE_ENCODE_TEXEL_LIMIT && block_encode_rate() == EncodeRate::Slow {
        return why(
            "encoding it at the guest's own resolution would stall the frame on this device's \
             CPU encoder, and a smaller version is not an option - the picture is not something \
             to trade. It takes the ordinary decode path at full resolution and costs memory \
             instead. A resumable encode is what removes this refusal",
        );
    }

    // Every level as RGBA8: the guest's own where it has them, box-filtered from the level
    // above where it does not. Mixing the two would be worse than either - the chain has to be
    // consistent, so the guest's levels are used for as long as they last.
    let want = max_mip_levels(t.width, t.height);
    let mut levels: Vec<(u32, u32, Vec<u8>)> = Vec::new();
    for level in 0..want {
        if level < t.levels {
            if let Some(view) = level_view(t, 0, level) {
                let (w, h, rgba, seam) = decode_texture_seam(&view);
                if seam != TexelSeam::Rgba8 {
                    return why("it decodes onto the half seam, which is DATA and not colour");
                }
                levels.push((w, h, rgba));
                continue;
            }
        }
        let Some((pw, ph, prev)) = levels.last() else {
            return why("its level 0 could not be decoded");
        };
        let (w, h, rgba) = halve_rgba8(*pw, *ph, prev);
        levels.push((w, h, rgba));
    }
    let (tw, th) = (levels[0].0, levels[0].1);
    // ONE format for the whole texture: a mip chain is a single GPU texture, so a level that
    // happens to be opaque cannot be BC1 while its neighbour is BC3. Any alpha anywhere means
    // the alpha-carrying format for all of it.
    //
    // >>> AND ONE FORMAT FOR THE WHOLE TEXTURE'S LIFE, ACROSS TIERS.
    //
    // The opacity test can only see the levels THIS tier decoded, and a preview tier decodes
    // only the small ones. A texture whose small levels happen to be opaque while its level 0 is
    // not would be encoded without alpha at preview and with it after promotion - so an
    // alpha-tested surface would render as a solid block for the second or so before it was
    // promoted, and then quietly fix itself. That is a defect that only appears during a
    // transition and repairs itself before anyone can look at it. The first tier's decision is
    // carried instead, so a promotion changes resolution and nothing else.
    let format = match force_format {
        Some(f) => f,
        None => {
            let opaque = levels.iter().all(|(_, _, px)| crate::bcenc::is_opaque(px));
            // Encode for the family the ADAPTER takes. A desktop takes BC; the phone this work
            // exists for takes ETC2 and nothing else, and encoding BC for it produced a 354 MB
            // working set while the report claimed a win.
            match (vitaslop_platform::gpu::block_family(), opaque) {
                (BlockFamily::Etc2, true) => BlockFormat::Etc2Rgb8,
                (BlockFamily::Etc2, false) => BlockFormat::Etc2Rgba8,
                (_, true) => BlockFormat::Bc1,
                (_, false) => BlockFormat::Bc3,
            }
        }
    };
    let mut data = Vec::new();
    for (w, h, rgba) in &levels {
        let block = match format {
            BlockFormat::Bc1 => crate::bcenc::encode_bc1(*w, *h, rgba),
            BlockFormat::Bc3 => crate::bcenc::encode_bc3(*w, *h, rgba),
            BlockFormat::Etc2Rgb8 => crate::etcenc::encode_etc2_rgb8(*w, *h, rgba),
            BlockFormat::Etc2Rgba8 => crate::etcenc::encode_etc2_rgba8(*w, *h, rgba),
            // BC2 is never CHOSEN as a transcode target - its 4-bit uncompressed alpha is
            // strictly worse than BC3's interpolated block at the same size.
            BlockFormat::Bc2 => unreachable!("BC2 is a passthrough format, never a transcode one"),
        };
        data.extend_from_slice(&block);
    }
    report_transcoded(t.base_format, format);
    Some(CompressedUpload {
        format,
        width: tw,
        height: th,
        data: vitaslop_platform::gpu::CompressedData::Cpu(Arc::new(data)),
        levels: levels.len() as u32,
        transcoded: true,
    })
}

/// Describe a texture the GPU can transcode by itself, or `None` if this one has to go through
/// the CPU.
///
/// # What this function is careful NOT to do
/// It reads no texels. Every decision here comes from the guest's format, its layout arithmetic,
/// and - for the one question that genuinely needs the content - the source blocks' own opacity
/// FLAGS, which are one word per eight bytes rather than a decode. That matters because the whole
/// value of the GPU path is that the CPU never touches the image, and a "cheap" CPU pass to decide
/// something about it would put most of the cost straight back.
///
/// # Why only PVRTC, and only to ETC2
/// PVRTC is the family that cannot pass through on any adapter and is 172 MB of one measured race
/// frame, so it is the whole problem. ETC2 is the family the target device takes, and it is the
/// only one whose encoder was ever the bottleneck - BC runs at 95 Mtexel/s, and it is also what
/// produces the blocks in the desktop's headless render, which is the determinism oracle every
/// capture in this project is compared against. Moving that encoder would retire the comparison
/// to fix a cost that does not exist on that engine.
fn gpu_transcode(t: &BoundTexture, force_format: Option<BlockFormat>) -> Option<CompressedUpload> {
    use vitaslop_platform::gpu::{CompressedData, GpuTranscode, SourceCodec, SrcLevel};

    if vitaslop_platform::gpu::block_family() != BlockFamily::Etc2 {
        return None;
    }
    // The channel swizzle is applied during the CPU decode and the shaders do not implement it,
    // exactly as `texel_rgba_face` does not apply it to PVRTC. Identity only.
    if (t.swizzle >> 12) & 0x7 != 0 {
        return None;
    }
    // >>> ALPHA IS DECIDED FROM THE SOURCE BLOCKS' OWN FLAGS, NOT FROM DECODED TEXELS.
    //
    // Both source families state their alpha structurally. A PVRTC block declares whether each of
    // its two colours is opaque, and 4bpp punch-through is a property of its modulation mode - so
    // an image whose every block is opaque and punch-through-free decodes to alpha 255 everywhere,
    // whatever the interpolation does, because the interpolation is between two 255s. BC1 is
    // opaque unless a block takes its 3-colour punch-through mode, which is a comparison of the
    // two stored endpoints; BC2 and BC3 carry a real alpha channel and are never assumed opaque.
    //
    // Either way the question is answered by reading a word or two per block instead of expanding
    // the image - which matters, because the entire value of this path is that the CPU never
    // touches the texels.
    let face = &t.pixels[..t.face_bytes.min(t.pixels.len() as u32) as usize];
    let (codec, opaque) = match crate::pvrtc::Variant::from_base_format(t.base_format) {
        Some(v) => (
            SourceCodec::Pvrtc { two: v.two, four_bpp: v.four_bpp },
            crate::pvrtc::face_is_opaque(face, v),
        ),
        None => match t.base_format {
            0x85 | 0x86 | 0x87 => {
                // >>> A BC SOURCE IS RE-ENCODED ONLY WHILE THE BUDGET IS TIGHT, exactly as the
                // CPU path decides it - and for a reason the CPU path's own cost argument
                // happens to share but does not state: **decoding BC to RGBA8 is EXACT**. Those
                // are the texels the hardware samples, so RGBA8 is the faithful upload and an
                // ETC2 re-encode is a SECOND lossy step on top of the guest's own compression.
                //
                // PVRTC above is not subject to this: no adapter has a PVRTC format at all, so
                // its only alternative is an eight-fold expansion, and transcoding it is the
                // whole reason this path exists. BC is the opposite case - it passes through
                // untouched on any desktop and, where it cannot, it still DECODES exactly.
                //
                // This gate was missing here while the CPU path had it, so the cheap GPU encoder
                // took the trade unconditionally. MEASURED on the user's device: format 0x87
                // re-encoded to `Etc2Rgba8` on a frame whose whole texture working set was
                // **1 MB against a 256 MB budget** - paying picture quality to shrink something
                // that fit many times over. A cheap encoder is a reason to spend CPU, never a
                // reason to spend QUALITY. [[vitaslop-never-trade-quality]]
                if !vitaslop_platform::gpu::texture_budget_pressure() {
                    return None;
                }
                (
                    SourceCodec::Bc { base_format: t.base_format },
                    t.base_format == 0x85 && bc1_face_is_opaque(face),
                )
            }
            _ => return None,
        },
    };
    let format = force_format.unwrap_or(if opaque { BlockFormat::Etc2Rgb8 } else { BlockFormat::Etc2Rgba8 });
    if format.family() != BlockFamily::Etc2 {
        return None;
    }

    // The guest's own levels, addressed by the same `level_layout` the CPU path uses. A second
    // implementation of this addressing in WGSL is exactly the kind of duplicate that drifts, and
    // its failure mode is a texture decoded plausibly out of the wrong bytes.
    let swizzled = swizzled_type(t.tex_type);
    let mut src_levels = Vec::new();
    for level in 0..t.levels {
        let l = level_layout(t.base_format, t.tex_type, t.width, t.height, level)?;
        let off = level_offset(t.base_format, t.tex_type, t.width, t.height, level)?;
        // A level the guest's allocation does not actually reach is not a level. The CPU path
        // discovers this through `level_view` returning `None` and falls back to box-filtering
        // from the level above; the same rule applies here, and stopping is how it is expressed.
        if (off as usize).saturating_add(l.bytes as usize) > t.pixels.len() {
            break;
        }
        src_levels.push(SrcLevel {
            byte_offset: off,
            width: l.width,
            height: l.height,
            blocks_x: l.blocks_x,
            blocks_y: l.blocks_y,
            padded_x: if swizzled { l.blocks_x.next_power_of_two() } else { l.blocks_x },
            padded_y: if swizzled { l.blocks_y.next_power_of_two() } else { l.blocks_y },
            swizzled,
        });
    }
    if src_levels.is_empty() {
        return None;
    }
    let levels = max_mip_levels(t.width, t.height);
    report_transcoded(t.base_format, format);
    Some(CompressedUpload {
        format,
        width: t.width,
        height: t.height,
        levels,
        transcoded: true,
        data: CompressedData::Gpu(GpuTranscode {
            src: t.pixels.clone(),
            codec,
            width: t.width,
            height: t.height,
            levels,
            src_levels,
        }),
    })
}

/// Whether a BC1 face decodes to alpha 255 everywhere, WITHOUT decoding it.
///
/// BC1 is opaque except in its 3-colour mode, which a block selects by storing `c0 <= c1`, and
/// even then only the texels whose index is 3 are transparent. So a face where every block has
/// `c0 > c1` is opaque, full stop - two 16-bit reads per block against a whole-image decode.
///
/// Conservative in the safe direction: a 3-colour block none of whose texels actually take index
/// 3 reads as translucent here, which costs the 8 bpp target instead of 4 bpp. Memory, never a
/// wrong picture. Checking the indices too would be exact and would read the whole block; the
/// mode word is the cheap 90%, and BC1 is the format least likely to carry alpha in the first
/// place.
fn bc1_face_is_opaque(bytes: &[u8]) -> bool {
    for block in bytes.chunks_exact(8) {
        let c0 = u16::from_le_bytes([block[0], block[1]]);
        let c1 = u16::from_le_bytes([block[2], block[3]]);
        if c0 <= c1 {
            return false;
        }
    }
    true
}

/// Report - once per (guest format, chosen block format) - that a texture was RE-ENCODED.
///
/// A transcode is not a passthrough and must never read like one in a log: the blocks the GPU
/// gets are ours, not the guest's, and the picture is a second lossy step away from the asset.
/// Whoever reads a capture and sees the working set collapse is entitled to know which of the
/// two things did it.
fn report_transcoded(base_format: u32, to: BlockFormat) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let key = (base_format, to as u32);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(key) {
        return;
    }
    // >>> "HAS NO WebGPU FORMAT" WAS WRONG FOR HALF THE FORMATS THAT REACH HERE.
    //
    // It is true of PVRTC, which no adapter can take. It is FALSE of `UBC1/2/3`, which are BC1/2/3
    // and pass through untouched on any desktop - they reach this path only because THIS adapter
    // has no BC. The old wording told a reader that a BC texture was unrepresentable in WebGPU,
    // which sends the next investigation looking for a missing format rather than at the adapter.
    let why = if crate::pvrtc::Variant::from_base_format(base_format).is_some() {
        "no WebGPU adapter has a PVRTC format at all"
    } else {
        "this adapter does not accept the block family it is already in"
    };
    tracing::warn!(
        target: "vitaslop::gxm",
        "gxm textures: base format {base_format:#04x} is being RE-ENCODED to {to:?} because \
         {why}. This is a second lossy step on top of the guest's own compression - it buys \
         roughly 8x the GPU memory and costs image quality"
    );
}

/// Report - once per (format, reason) - that a texture WebGPU has a block format for was
/// decoded to RGBA8 anyway.
///
/// Every one of these is megabytes on a device that has none to spare, and each reason has a
/// different fix. Silence here would leave "the working set did not shrink" with no way to tell
/// a passthrough that never fired from one that fired and was not enough.
fn report_passthrough_refused(base_format: u32, reason: &'static str) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, &'static str)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((base_format, reason)) {
        return;
    }
    tracing::warn!(
        target: "vitaslop::gxm",
        "gxm textures: base format {base_format:#04x} is a block format the GPU could take \
         verbatim, but it is being DECODED to RGBA8 instead because {reason}"
    );
}

/// Box-filter one RGBA8 image down by two, for the mip levels a transcode has to invent when
/// the guest supplied none.
pub fn halve_rgba8(w: u32, h: u32, src: &[u8]) -> (u32, u32, Vec<u8>) {
    let (dw, dh) = ((w / 2).max(1), (h / 2).max(1));
    let mut dst = vec![0u8; (dw * dh * 4) as usize];
    for y in 0..dh as usize {
        for x in 0..dw as usize {
            for c in 0..4usize {
                let x0 = (2 * x).min(w as usize - 1);
                let x1 = (2 * x + 1).min(w as usize - 1);
                let y0 = (2 * y).min(h as usize - 1);
                let y1 = (2 * y + 1).min(h as usize - 1);
                let at = |xx: usize, yy: usize| src[(yy * w as usize + xx) * 4 + c] as u32;
                dst[(y * dw as usize + x) * 4 + c] =
                    ((at(x0, y0) + at(x1, y0) + at(x0, y1) + at(x1, y1) + 2) / 4) as u8;
            }
        }
    }
    (dw, dh, dst)
}

/// Whether a `SceGxmTextureType` selector uses the GPU's Morton (Z-order) swizzled
/// memory layout: `SCE_GXM_TEXTURE_SWIZZLED` (0), `SWIZZLED_ARBITRARY` (5), and both CUBE
/// layouts (2, 7), whose individual faces are swizzled images.
pub fn swizzled_type(tex_type: u32) -> bool {
    matches!(tex_type, 0 | 2 | 5 | 7)
}

/// Whether a `SceGxmTextureType` selector is a CUBE map, whose data holds six faces rather
/// than one image: `SCE_GXM_TEXTURE_CUBE` (2) and `SCE_GXM_TEXTURE_CUBE_ARBITRARY` (7). Both
/// use the swizzled per-face layout [`swizzled_type`] describes.
pub fn cube_type(tex_type: u32) -> bool {
    matches!(tex_type, 2 | 7)
}

/// Interleave the low bits of `x` and `y` (Morton / Z-order) up to the square
/// formed by the smaller dimension, then append the remaining high bits of the
/// larger dimension linearly. This is the GXM swizzle for a power-of-two-padded
/// block grid `pw x ph`, matching how the GPU addresses a SWIZZLED texture.
pub(crate) fn morton_index(mut x: u32, mut y: u32, pw: u32, ph: u32) -> u32 {
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

/// Split [`morton_index`] into a per-COLUMN and a per-ROW table, so that
/// `morton_index(x, y, pw, ph) == xs[x] + ys[y]` exactly for every `x < pw`, `y < ph`.
///
/// # Why this is exact rather than an approximation
/// Both halves of the index are separable, and both compose by OR over DISJOINT bits, which is
/// addition:
/// * the interleaved low half puts `x`'s bits at odd positions and `y`'s at even ones, so the
///   two never collide;
/// * the leftover-tiling high half is `((y_hi * cols) + x_hi) << 2*min_log` (or the transpose),
///   which is a sum of one term in `x` and one in `y`, and sits entirely above the low bits.
///
/// # Why it is worth a table
/// `morton_index` runs a loop over `min_log` bits - about ten iterations for a 1024-wide level -
/// and the callers that matter run it ONCE PER TEXEL, twice for a paletted expand (source and
/// destination). MEASURED with a V8 sampling profile of the browser worker on the golf title:
/// `texel_element` alone was **30.5% of the whole thread** and the iterator inside it another
/// **31.9%**, against 4.6 ms of actual rendering. With the tables the interleave is paid once
/// per row and once per column instead of once per texel.
pub(crate) fn morton_tables(pw: u32, ph: u32, w: u32, h: u32) -> (Vec<u32>, Vec<u32>) {
    let min_log = pw.min(ph).trailing_zeros();
    let wide = pw >= ph;
    // Columns of the leftover strip, in whole squares - the same `pw >> min_log` /
    // `ph >> min_log` the scalar form multiplies by.
    let step = if wide { pw >> min_log } else { ph >> min_log };
    let mut xs = Vec::with_capacity(w as usize);
    for x in 0..w {
        let mut v = 0u32;
        for i in 0..min_log {
            v |= ((x >> i) & 1) << (2 * i + 1);
        }
        let hi = x >> min_log;
        v |= (if wide { hi } else { hi * step }) << (2 * min_log);
        xs.push(v);
    }
    let mut ys = Vec::with_capacity(h as usize);
    for y in 0..h {
        let mut v = 0u32;
        for i in 0..min_log {
            v |= ((y >> i) & 1) << (2 * i);
        }
        let hi = y >> min_log;
        v |= (if wide { hi * step } else { hi }) << (2 * min_log);
        ys.push(v);
    }
    (xs, ys)
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
pub(crate) fn decode_bc_texel(block: &[u8], base_format: u32, px: u32, py: u32) -> [u8; 4] {
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

/// Decode a whole BC1/BC2/BC3 block to its sixteen RGBA8 texels at once.
///
/// >>> THE ENDPOINTS ARE A PROPERTY OF THE BLOCK AND WERE BEING DERIVED PER TEXEL.
///
/// [`decode_bc_texel`] re-reads both 565 endpoints, re-expands them, re-decides the
/// punch-through mode and re-interpolates the palette entry it needs - and for BC3 re-reads
/// the two alpha endpoints as well - for EVERY ONE of a block's sixteen texels. The block
/// walker called it sixteen times per block, so all of that ran sixteen times to produce four
/// colours and eight alphas that do not change inside a block.
///
/// It matters most on the device that needs it most: a phone GPU with no BC support decodes
/// every compressed guest texture on the CPU [[vitaslop-phone-gpu-has-no-bc]], and MEASURED on
/// a retail golf title under that rig (`VITASLOP_NO_BC=1`) the BC families are **95.9 MB of
/// the run's 270.5 MB of texture decode**.
///
/// The per-texel entry point stays: it is what a single-texel sampler read
/// ([`texel_rgba_face`], [`texture_mean_rgb`]) uses, and it is the oracle this is asserted
/// against by `the_block_decoder_matches_the_per_texel_one`.
fn decode_bc_block(block: &[u8], base_format: u32) -> [[u8; 4]; 16] {
    let color_off = if base_format == 0x85 { 0 } else { 8 };
    let g = |i: usize| -> u8 { *block.get(i).unwrap_or(&0) };
    let c0 = u16::from_le_bytes([g(color_off), g(color_off + 1)]);
    let c1 = u16::from_le_bytes([g(color_off + 2), g(color_off + 3)]);
    let (e0, e1) = (rgb565(c0), rgb565(c1));
    let punchthrough = base_format == 0x85 && c0 <= c1;
    let mix = |a: [u8; 3], b: [u8; 3], na: u32, nb: u32, d: u32| -> [u8; 3] {
        [
            ((a[0] as u32 * na + b[0] as u32 * nb) / d) as u8,
            ((a[1] as u32 * na + b[1] as u32 * nb) / d) as u8,
            ((a[2] as u32 * na + b[2] as u32 * nb) / d) as u8,
        ]
    };
    // The four colour entries, in index order, exactly as the per-texel `match idx` picks
    // them - including the 3-colour mode's black fourth entry.
    let palette: [[u8; 3]; 4] = if punchthrough {
        [e0, e1, mix(e0, e1, 1, 1, 2), [0, 0, 0]]
    } else {
        [e0, e1, mix(e0, e1, 2, 1, 3), mix(e0, e1, 1, 2, 3)]
    };
    // ...and the eight BC3 alpha entries, on the same terms.
    let (a0, a1) = (g(0), g(1));
    let (a0i, a1i) = (a0 as u32, a1 as u32);
    let alpha: [u8; 8] = if a0 > a1 {
        [
            a0,
            a1,
            ((6 * a0i + a1i) / 7) as u8,
            ((5 * a0i + 2 * a1i) / 7) as u8,
            ((4 * a0i + 3 * a1i) / 7) as u8,
            ((3 * a0i + 4 * a1i) / 7) as u8,
            ((2 * a0i + 5 * a1i) / 7) as u8,
            ((a0i + 6 * a1i) / 7) as u8,
        ]
    } else {
        [
            a0,
            a1,
            ((4 * a0i + a1i) / 5) as u8,
            ((3 * a0i + 2 * a1i) / 5) as u8,
            ((2 * a0i + 3 * a1i) / 5) as u8,
            ((a0i + 4 * a1i) / 5) as u8,
            0,
            255,
        ]
    };
    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let idx = ((g(color_off + 4 + t / 4) >> ((t % 4) * 2)) & 0x3) as usize;
        let rgb = palette[idx];
        let a = match base_format {
            0x85 => {
                if punchthrough && idx == 3 {
                    0
                } else {
                    255
                }
            }
            0x86 => {
                let byte = g(t / 2);
                let a4 = if t % 2 == 0 { byte & 0xf } else { byte >> 4 };
                (a4 as u32 * 255 / 15) as u8
            }
            0x87 => {
                let bit = t * 3;
                let byte = 2 + bit / 8;
                let raw = (g(byte) as u32) | ((g(byte + 1) as u32) << 8);
                alpha[((raw >> (bit % 8)) & 0x7) as usize]
            }
            _ => 255,
        };
        *texel = [rgb[0], rgb[1], rgb[2], a];
    }
    out
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

/// The mean RGB (each channel in [0,1]) of a captured texture, decoded through the exact
/// per-texel path the sampler uses. Used to reduce a small `diffuseAmbientMap` irradiance
/// probe to a single flat ambient colour for the lit material model. Samples on a bounded
/// grid (at most ~32x32 taps) so a large texture costs no more than a small one. Returns
/// `None` for an empty or undecodable texture (the caller keeps its default ambient).
pub fn texture_mean_rgb(t: &BoundTexture) -> Option<[f32; 3]> {
    if t.width == 0 || t.height == 0 || block_layout(t.base_format).is_none() {
        return None;
    }
    let steps_x = t.width.min(32);
    let steps_y = t.height.min(32);
    let (mut r, mut g, mut b) = (0f32, 0f32, 0f32);
    let mut n = 0f32;
    for sy in 0..steps_y {
        for sx in 0..steps_x {
            let x = sx * t.width / steps_x;
            let y = sy * t.height / steps_y;
            let px = texel_rgba(t, x.min(t.width - 1), y.min(t.height - 1));
            r += px[0] as f32;
            g += px[1] as f32;
            b += px[2] as f32;
            n += 1.0;
        }
    }
    if n == 0.0 {
        return None;
    }
    Some([r / n / 255.0, g / n / 255.0, b / n / 255.0])
}

/// Point-sample a captured texture at normalized `(u, v)` (REPEAT wrap) and decode
/// the texel to straight RGBA8. Covers the uncompressed formats a 2D title uses;
/// an unknown format returns opaque magenta so it is visible, not silent.
fn sample_texture(t: &BoundTexture, u: f32, v: f32) -> [u8; 4] {
    if t.width == 0 || t.height == 0 {
        return [255, 0, 255, 255];
    }
    // Honor the guest-set magnification filter (SceGxmTextureFilter: 0 = POINT,
    // 1 = LINEAR). LINEAR is what UI/font-atlas text is drawn with; point-sampling it
    // at sub-native scale breaks thin glyph strokes.
    const SCE_GXM_TEXTURE_FILTER_LINEAR: u32 = 1;
    if t.mag_filter == SCE_GXM_TEXTURE_FILTER_LINEAR {
        return sample_texture_bilinear(t, u, v);
    }
    // POINT: nearest texel, REPEAT wrap (fractional part into [0, 1)).
    let uu = u - u.floor();
    let vv = v - v.floor();
    let x = ((uu * t.width as f32) as i64).clamp(0, t.width as i64 - 1) as u32;
    let y = ((vv * t.height as f32) as i64).clamp(0, t.height as i64 - 1) as u32;
    texel_rgba(t, x, y)
}

/// Bilinear texel fetch: the four texels around the sample point, lerped by the
/// sub-texel fraction. Texel centers sit at integer+0.5, so the sample coordinate is
/// `uv * size - 0.5`; the four integer taps wrap REPEAT (matching the point path's
/// REPEAT assumption, and what a 2D title's tiled/atlas textures use).
fn sample_texture_bilinear(t: &BoundTexture, u: f32, v: f32) -> [u8; 4] {
    let uu = u - u.floor();
    let vv = v - v.floor();
    let fx = uu * t.width as f32 - 0.5;
    let fy = vv * t.height as f32 - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let dx = fx - x0;
    let dy = fy - y0;
    // REPEAT-wrap an integer texel coordinate into range.
    let wrap = |c: i64, n: u32| -> u32 { c.rem_euclid(n as i64) as u32 };
    let x0i = wrap(x0 as i64, t.width);
    let x1i = wrap(x0 as i64 + 1, t.width);
    let y0i = wrap(y0 as i64, t.height);
    let y1i = wrap(y0 as i64 + 1, t.height);
    let c00 = texel_rgba(t, x0i, y0i);
    let c10 = texel_rgba(t, x1i, y0i);
    let c01 = texel_rgba(t, x0i, y1i);
    let c11 = texel_rgba(t, x1i, y1i);
    let mut out = [0u8; 4];
    for ch in 0..4 {
        let top = c00[ch] as f32 * (1.0 - dx) + c10[ch] as f32 * dx;
        let bot = c01[ch] as f32 * (1.0 - dx) + c11[ch] as f32 * dx;
        out[ch] = (top * (1.0 - dy) + bot * dy).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Decode a whole captured texture to a tightly-packed linear RGBA8 image at its
/// native `width x height`, using the exact per-texel decode the software sampler
/// uses. This is the seam the wgpu backend samples through: the format/swizzle/BC/
/// Morton complexity stays here (one tested place), and the GPU sees a plain
/// `Rgba8Unorm` image it can point-sample with a REPEAT sampler for the identical
/// result. Returns `(width, height, rgba)`; a zero-sized texture yields a 1x1 opaque
/// magenta so an unexpected empty binding is visible rather than a GPU error.
///
/// Callers that must handle the 64-bit half-float family exactly want
/// [`decode_texture_seam`]: this one narrows it back to bytes, which is right for a
/// screenshot or a colour probe and wrong for a texture whose texels are data.
pub fn decode_texture_rgba8(t: &BoundTexture) -> (u32, u32, Vec<u8>) {
    let (w, h, data, seam) = decode_texture_seam(t);
    match seam {
        TexelSeam::Rgba8 => (w, h, data),
        // Narrow, and say so in the only way that survives: the values are still right, only
        // coarser. A caller wanting the exact lanes has the function above.
        TexelSeam::Rgba16Float => (
            w,
            h,
            data.chunks_exact(2)
                .map(|c| {
                    let v = half_to_f32(u16::from_le_bytes([c[0], c[1]]));
                    (v.clamp(0.0, 1.0) * 255.0).round() as u8
                })
                .collect(),
        ),
    }
}

/// The full-precision decode: `(width, height, texels, the layout they are in)`.
pub fn decode_texture_seam(t: &BoundTexture) -> (u32, u32, Vec<u8>, TexelSeam) {
    decode_texture_rgba8_counted(t, &mut BuildWork::default())
}

/// [`decode_texture_rgba8`], recording which path decoded it. See [`BuildWork`].
/// The seam a guest base format decodes ONTO - see [`TexelSeam`].
///
/// Only `0x1b` (F16F16F16F16) widens, and the restriction is deliberate.
///
/// It is the one format the half seam represents EXACTLY - the decode is a lane-for-lane copy of
/// the guest's own halves through the channel swizzle, so nothing is converted and the narrowed
/// reading stays identical to what the byte decoder produces from the same bytes. That identity
/// is what keeps [`texel_rgba_face`] usable as the per-texel oracle for this format.
///
/// `0x1c`/`0x1d` (U16/S16 normalized) are deliberately NOT here. Core WebGPU has no filterable
/// `rgba16unorm`, so they would have to go through f16, which is 11 bits of mantissa - better
/// than 8, but still lossy, and lossy in a DIFFERENT way from the byte path, which makes the two
/// decoders disagree for no exactness gained. Swapping one lossy conversion for another is not
/// worth losing the oracle over. If a title is ever measured to store data in one of them, the
/// answer is a third seam, not this one.
pub fn seam_for_format(base_format: u32) -> TexelSeam {
    match base_format {
        0x1b => TexelSeam::Rgba16Float,
        _ => TexelSeam::Rgba8,
    }
}

/// The `(width, height, seam)` a decode of `t` WOULD produce, without performing it.
///
/// Every caller of the decode used all three of those and then, in the shipped configuration,
/// discarded the pixels ([`vitaslop_platform::gpu::Texels`]). They are pure metadata - the seam
/// comes from the base format and the dimensions are the guest's - so nothing has to be decoded
/// to know them. The 1x1 case mirrors [`decode_texture_rgba8_counted`]'s magenta placeholder
/// exactly, because a zero-sized texture must report the shape its texels will actually be in.
pub fn decoded_texture_shape(t: &BoundTexture) -> (u32, u32, TexelSeam) {
    if t.width == 0 || t.height == 0 {
        return (1, 1, TexelSeam::Rgba8);
    }
    (t.width, t.height, seam_for_format(t.base_format))
}

/// What one decode-cache entry will hold, priced WITHOUT forcing the decode.
///
/// # Why this is a prediction and not a measurement
/// The obvious implementation is `rgba.len()`, and it is the one that was here. Under
/// [`vitaslop_platform::gpu::Texels`] reading that length IS the decode, so an accounting call
/// would perform the work the accounting exists to avoid - the instrument destroying its own
/// subject.
///
/// So the price is derived from the shape instead. A texture the adapter will take as BLOCKS is
/// priced at the blocks, because its texels are never produced; anything else is priced at the
/// full expansion it will be read through, mips included, which is what the uploader will hold.
///
/// It can be wrong in one direction and it is worth naming: a texture handed over compressed
/// that some DIAGNOSTIC then reads (the vertex-texture clip probe, `VITASLOP_GXP_INPUTS`) does
/// materialise its texels, and this under-prices it until it is re-inserted. That is bounded by
/// the number of textures a probe touches, and the alternative - pricing every compressed
/// texture as if it were expanded - would put back exactly the inflated working set that made
/// the cache thrash.
fn predicted_texture_bytes(
    width: u32,
    height: u32,
    faces: u32,
    texel: TexelSeam,
    compressed: Option<&CompressedUpload>,
) -> usize {
    if let Some(c) = compressed {
        if c.format.family() == vitaslop_platform::gpu::block_family() {
            return c.byte_len();
        }
    }
    let level0 =
        (width.max(1) as usize) * (height.max(1) as usize) * (faces.max(1) as usize) * texel.bytes_per_texel();
    // The same 4/3 the uploader's own budget uses for a chain it builds, and the same reason:
    // rounding UP is the safe direction for a budget.
    if texel == TexelSeam::Rgba8 {
        level0 * 4 / 3
    } else {
        level0
    }
}

fn decode_texture_rgba8_counted(
    t: &BoundTexture,
    work: &mut BuildWork,
) -> (u32, u32, Vec<u8>, TexelSeam) {
    let seam = seam_for_format(t.base_format);
    if t.width == 0 || t.height == 0 {
        return (1, 1, vec![255, 0, 255, 255], TexelSeam::Rgba8);
    }
    if seam == TexelSeam::Rgba16Float {
        return decode_texture_rgba16f(t, work);
    }
    if t.base_format == YUV420P2 {
        return decode_texture_yuv420p2(t, work);
    }
    // A cube map decodes to its six faces stacked in `BoundTexture::faces` order, which is the
    // layer order the GPU binds them in.
    let faces = t.faces.max(1);
    let mut rgba = vec![0u8; (t.width * t.height * faces * 4) as usize];
    for f in 0..faces {
        let face_out = (f * t.width * t.height * 4) as usize;
        let face_len = (t.width * t.height * 4) as u64;
        DECODE_BY_FORMAT.lock().unwrap()[(t.base_format & 0xff) as usize] += face_len;
        if decode_face_fast(t, f, &mut rgba[face_out..]) {
            work.tex_out_blockwise += face_len;
            continue;
        }
        work.tex_out_per_texel += face_len;
        let mut o = face_out;
        for y in 0..t.height {
            for x in 0..t.width {
                rgba[o..o + 4].copy_from_slice(&texel_rgba_face(t, f, x, y));
                o += 4;
            }
        }
    }
    (t.width, t.height, rgba, TexelSeam::Rgba8)
}

/// Decode a two-plane 4:2:0 texture - a decoded video frame - to RGBA8.
///
/// # Why the conversion happens HERE rather than in a shader
///
/// The hardware sampler converts YUV to RGB on the way to the fragment program: the guest's
/// shader reads ordinary colour and knows nothing about planes. Converting on upload puts
/// the conversion in exactly that place, so no shader, bind-group layout or sampler
/// declaration has to change - a two-plane texture arrives at the recompiler as the same
/// `Two`-dimensional RGBA texture as everything else.
///
/// # What the matrix is, and what is assumed
///
/// The swizzle field selects the channel order (YUV or YVU) and WHICH of two conversion
/// profiles applies (`CSC0`/`CSC1`); the profiles themselves are set per context by
/// `sceGxmSetYuvProfile`. A title that never calls it - and this one does not - leaves both
/// at the default, `SCE_GXM_YUV_PROFILE_BT601_STANDARD`: BT.601 coefficients over the
/// studio-swing ranges (luma 16..235, chroma 16..240). That default is the ASSUMPTION here,
/// and it is reported once; the channel order is read, not assumed.
fn decode_texture_yuv420p2(
    t: &BoundTexture,
    work: &mut BuildWork,
) -> (u32, u32, Vec<u8>, TexelSeam) {
    let (w, h) = (t.width, t.height);
    let luma_stride = align_up_to(w, 8) as usize;
    let chroma_stride = align_up_to(w.div_ceil(2), 8) as usize * 2;
    let chroma_base = luma_stride * h as usize;
    // Bits 12..13 of the format word: bit 12 swaps Cb and Cr, bit 13 selects the second
    // conversion profile. Only the swap changes anything here - see the doc comment.
    let swizzle = (t.swizzle >> 12) & 0x3;
    let swapped = swizzle & 1 != 0;
    report_yuv_profile_assumed(t.swizzle);

    let (w_us, h_us) = (w as usize, h as usize);
    let mut rgba = vec![0u8; w_us * h_us * 4];
    let px = &t.pixels[..];
    // The FAST path: this walks whole rows through slices, and the counter has to say so or
    // the working-set report keeps calling the most expensive texture in the frame cheap.
    work.tex_out_blockwise += (w_us * h_us * 4) as u64;
    DECODE_BY_FORMAT.lock().unwrap()[YUV420P2 as usize] += (w_us * h_us * 4) as u64;

    // >>> TWO ROWS AND TWO COLUMNS AT A TIME, THROUGH SLICES.
    //
    // A video frame is re-decoded EVERY frame - its content changes, so no cache can help -
    // which makes this the one texture conversion whose cost is paid sixty times a second.
    // MEASURED on a phone, when this was a per-pixel loop with a bounds-checked read per
    // sample: 336 MB of conversion over one run and 7.7 ms in the worst frame's `prepare`,
    // which was the single largest render cost on the device.
    //
    // Each chroma sample serves a 2x2 group, so the group is the unit: the chroma pair is
    // read and converted ONCE and applied to four luma samples, and the row slices are taken
    // outside the inner loop so the bounds checks are hoisted out of it.
    let tables = &*BT601_TABLES;
    for cy in 0..h_us.div_ceil(2) {
        let chroma = chroma_base + cy * chroma_stride;
        let Some(chroma_row) = px.get(chroma..chroma + (w_us.div_ceil(2)) * 2) else {
            break;
        };
        for dy in 0..2 {
            let y = cy * 2 + dy;
            if y >= h_us {
                break;
            }
            let Some(luma_row) = px.get(y * luma_stride..y * luma_stride + w_us) else {
                break;
            };
            let out_row = &mut rgba[y * w_us * 4..(y + 1) * w_us * 4];
            for cx in 0..w_us.div_ceil(2) {
                let (a, b) = (chroma_row[cx * 2], chroma_row[cx * 2 + 1]);
                let (cb, cr) = if swapped { (b, a) } else { (a, b) };
                // The three chroma contributions, per chroma sample rather than per pixel.
                let (r_off, g_off, b_off) = (
                    tables.cr_r[cr as usize],
                    tables.cb_g[cb as usize] + tables.cr_g[cr as usize],
                    tables.cb_b[cb as usize],
                );
                for dx in 0..2 {
                    let x = cx * 2 + dx;
                    if x >= w_us {
                        break;
                    }
                    let y_term = tables.luma[luma_row[x] as usize];
                    let px_out = &mut out_row[x * 4..x * 4 + 4];
                    px_out[0] = clamp8(y_term + r_off);
                    px_out[1] = clamp8(y_term + g_off);
                    px_out[2] = clamp8(y_term + b_off);
                    px_out[3] = 255;
                }
            }
        }
    }
    (w, h, rgba, TexelSeam::Rgba8)
}

/// The BT.601 studio-swing conversion, precomputed per input byte.
///
/// Every term is a function of ONE sample, so all of them fit in 256-entry tables and the
/// per-pixel work becomes three adds and three clamps. Built once.
struct Bt601Tables {
    /// `(y - 16) * 255/219`, in 16.16.
    luma: [i32; 256],
    cr_r: [i32; 256],
    cr_g: [i32; 256],
    cb_g: [i32; 256],
    cb_b: [i32; 256],
}

static BT601_TABLES: std::sync::LazyLock<Bt601Tables> = std::sync::LazyLock::new(|| {
    let mut t = Bt601Tables {
        luma: [0; 256],
        cr_r: [0; 256],
        cr_g: [0; 256],
        cb_g: [0; 256],
        cb_b: [0; 256],
    };
    for i in 0..256usize {
        t.luma[i] = (i as i32 - 16) * 76309;
        let c = i as i32 - 128;
        t.cr_r[i] = 104597 * c;
        t.cr_g[i] = -53279 * c;
        t.cb_g[i] = -25675 * c;
        t.cb_b[i] = 132201 * c;
    }
    t
});

/// Round a 16.16 fixed-point channel to a byte.
fn clamp8(v: i32) -> u8 {
    ((v + 32768) >> 16).clamp(0, 255) as u8
}

/// BT.601 studio-swing YUV to full-range RGB, the `SCE_GXM_YUV_PROFILE_BT601_STANDARD`
/// conversion. Integer arithmetic in 16.16, which is exact enough that no channel differs
/// from the float form by more than one step.
///
/// The bulk path uses [`BT601_TABLES`]; this is the same arithmetic written out, and the
/// test below holds the two together.
fn bt601_studio_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let y = (y as i32 - 16) * 76309;
    let u = cb as i32 - 128;
    let v = cr as i32 - 128;
    let clamp = |v: i32| ((v + 32768) >> 16).clamp(0, 255) as u8;
    [
        clamp(y + 104597 * v),
        clamp(y - 25675 * u - 53279 * v),
        clamp(y + 132201 * u),
    ]
}

#[cfg(test)]
mod yuv_tests {
    use super::*;

    /// The table-driven bulk path and the written-out reference must agree EXACTLY.
    ///
    /// The bulk path exists because a video frame is re-converted every frame and the naive
    /// loop was the largest render cost on a phone. A faster path that is not the same
    /// arithmetic is a different picture, and the difference would be a colour shift nobody
    /// would trace back to a lookup table.
    #[test]
    fn the_fast_path_matches_the_reference_conversion() {
        let tables = &*BT601_TABLES;
        for y in (0..=255u8).step_by(5) {
            for cb in (0..=255u8).step_by(17) {
                for cr in (0..=255u8).step_by(17) {
                    let want = bt601_studio_to_rgb(y, cb, cr);
                    let luma = tables.luma[y as usize];
                    let got = [
                        clamp8(luma + tables.cr_r[cr as usize]),
                        clamp8(luma + tables.cb_g[cb as usize] + tables.cr_g[cr as usize]),
                        clamp8(luma + tables.cb_b[cb as usize]),
                    ];
                    assert_eq!(got, want, "y={y} cb={cb} cr={cr}");
                }
            }
        }
    }

    /// Studio swing: video black is 16 and video white is 235, not 0 and 255.
    #[test]
    fn studio_swing_endpoints_map_to_full_range() {
        assert_eq!(bt601_studio_to_rgb(16, 128, 128), [0, 0, 0]);
        assert_eq!(bt601_studio_to_rgb(235, 128, 128), [255, 255, 255]);
    }
}

/// Say, once per swizzle, that a YUV texture's conversion profile is the DEFAULT rather than
/// one the title chose.
///
/// `sceGxmSetYuvProfile` is what picks between BT.601 and BT.709 and between studio and full
/// range, and it is per context, so nothing about the texture itself records which applies.
/// A title that never sets one gets the default - but a title that DOES set one and is
/// converted with the default here comes out with washed-out or over-saturated video that
/// looks like a decoder problem and is not.
fn report_yuv_profile_assumed(swizzle: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((swizzle >> 12) & 0x3) {
        return;
    }
    let csc = if (swizzle >> 13) & 1 == 0 { "CSC0" } else { "CSC1" };
    let order = if (swizzle >> 12) & 1 == 0 { "YUV" } else { "YVU" };
    eprintln!(
        "gxm texture: a two-plane 4:2:0 (video) texture is bound as {order}/{csc}. The channel          order is read from the format; the CONVERSION is BT.601 studio-swing, which is the          GXM default profile - if the title ever calls sceGxmSetYuvProfile, this run is          converting with the wrong one and the picture's colour is an assumption."
    );
}

/// Decode an F16F16F16F16 texture onto the HALF seam, lane for lane.
///
/// A straight copy of the guest's own halves through the channel swizzle - bit-exact, which is
/// the whole reason for the wider seam. See [`seam_for_format`] for why no other format is here.
fn decode_texture_rgba16f(
    t: &BoundTexture,
    work: &mut BuildWork,
) -> (u32, u32, Vec<u8>, TexelSeam) {
    let faces = t.faces.max(1);
    let texels = (t.width * t.height * faces) as usize;
    let mut out = vec![0u8; texels * 8];
    // The channel swizzle field, bits 12..14 of the full `SceGxmTextureFormat` - the same
    // extraction `decode_uncompressed_at` does. Passing `t.swizzle` raw permutes by whatever the
    // whole word happens to be, which is a plausible-looking wrong picture rather than an error.
    let swizzle = (t.swizzle >> 12) & 0x7;
    for f in 0..faces {
        let face_len = (t.width * t.height * 8) as u64;
        DECODE_BY_FORMAT.lock().unwrap()[(t.base_format & 0xff) as usize] += face_len;
        work.tex_out_per_texel += face_len;
        let mut o = (f * t.width * t.height * 8) as usize;
        for y in 0..t.height {
            for x in 0..t.width {
                // The lane offsets are the byte-seam decoder's, so the two paths agree about
                // WHICH lane is which; only the width of what is kept differs.
                let base = texel_byte_offset(t, f, x, y);
                let raw = |i: usize| -> u16 {
                    let b = |k: usize| t.pixels.get(base + k).copied().unwrap_or(0);
                    u16::from_le_bytes([b(i * 2), b(i * 2 + 1)])
                };
                let lanes = swizzle4_u16(raw(0), raw(1), raw(2), raw(3), swizzle);
                for (i, l) in lanes.iter().enumerate() {
                    out[o + i * 2..o + i * 2 + 2].copy_from_slice(&l.to_le_bytes());
                }
                o += 8;
            }
        }
    }
    (t.width, t.height, out, TexelSeam::Rgba16Float)
}

/// A texture whose texel is ONE channel, and how that channel reduces to the shared RGBA8
/// seam. Resolved once per face by [`SingleChannel::of`].
///
/// >>> THE FORMAT IS A CONSTANT OF THE FACE AND WAS BEING RE-DECIDED PER TEXEL.
///
/// `decode_uncompressed_at` opens with a fifteen-arm `match t.base_format` and re-reads the
/// swizzle field, and the fast walk called it once per texel - so a 1024x1024 face ran that
/// dispatch a million times to reach the same arm every time. Single-channel formats are the
/// bulk of what a real title decodes here: MEASURED on a retail golf title, the 16-bit
/// single-channel family alone is **151.9 MB of the run's 270.5 MB**, with 8-bit single
/// channel (font atlases and coverage masks) another 10.1 MB.
///
/// Only families whose lane is a pure function of its own bytes are listed. Anything else
/// falls back to the general per-texel decode, unchanged.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SingleChannel {
    /// U8 / S8 - the byte IS the lane.
    Bits8,
    /// U16, reduced by [`unorm16_to_u8`].
    Unorm16,
    /// S16, reduced by [`snorm16_to_u8`].
    Snorm16,
    /// F16, which is not a fixed-point range and keeps the float reduction.
    Half16,
}

impl SingleChannel {
    fn of(base_format: u32) -> Option<SingleChannel> {
        match base_format {
            0x00 | 0x01 => Some(SingleChannel::Bits8),
            0x09 => Some(SingleChannel::Unorm16),
            0x0a => Some(SingleChannel::Snorm16),
            0x0b => Some(SingleChannel::Half16),
            _ => None,
        }
    }

    /// The lane at `off`, reading zeros past the end of the buffer exactly as
    /// `decode_uncompressed_at`'s bounds-checked `byte` helper does.
    #[inline]
    fn lane(self, px: &[u8], off: usize) -> u8 {
        let byte = |i: usize| -> u8 { px.get(off + i).copied().unwrap_or(0) };
        match self {
            SingleChannel::Bits8 => byte(0),
            SingleChannel::Unorm16 => unorm16_to_u8(u16::from_le_bytes([byte(0), byte(1)])),
            SingleChannel::Snorm16 => snorm16_to_u8(u16::from_le_bytes([byte(0), byte(1)])),
            SingleChannel::Half16 => {
                let v = half_to_f32(u16::from_le_bytes([byte(0), byte(1)]));
                (v.clamp(0.0, 1.0) * 255.0).round() as u8
            }
        }
    }
}


/// Decode one face of a BLOCK-COMPRESSED texture a block at a time, into `out`. Returns
/// false for anything this path does not cover, leaving the caller on the per-texel path.
///
/// # Why this exists
/// [`texel_rgba_face`] answers for ONE texel, and for a block-compressed format answering
/// for one texel means decoding the whole 4x4 block it belongs to - so the plain loop
/// decodes every block SIXTEEN TIMES, and recomputes that block's Morton address and
/// re-dispatches on the format sixteen times with it. That is invisible until a title
/// streams: MEASURED in the browser mid-race, a single frame that first sees a new stretch
/// of track decoded 263 textures / 45.6 MB and spent **2,498 ms** inside `build`, while a
/// frame of 444 draws that decoded nothing built in **1.2 ms**. Build time tracked decoded
/// megabytes and nothing else.
///
/// This walks blocks instead: one decode, one address, one dispatch, sixteen texels
/// written. It is the same arithmetic per texel, so the output is byte-for-byte what the
/// per-texel path produces - `blockwise_decode_matches_per_texel` is the test that says so,
/// and the per-texel function stays as the oracle it checks against.
fn decode_face_fast(t: &BoundTexture, face: u32, out: &mut [u8]) -> bool {
    let Some((block_w, block_h, block_bytes)) = block_layout(t.base_format) else {
        return false;
    };
    // PVRTC is not block-LOCAL (a texel reads the four blocks whose centres surround it), so
    // it cannot go through the block walker below - it has a whole-IMAGE decode of its own,
    // which is the same arithmetic with the block decodes and their addressing hoisted out of
    // the texel loop. It is the largest decoded family measured (47%) and the one WebGPU
    // cannot be handed compressed, so this path is not optional. The channel swizzle is not
    // applied here, exactly as `texel_rgba_face` does not apply it to PVRTC.
    if let Some(variant) = crate::pvrtc::Variant::from_base_format(t.base_format) {
        if !pvrtc_whole_image() {
            return false;
        }
        let face_base = (face * t.face_bytes) as usize;
        let face_bytes = t.pixels.get(face_base..).unwrap_or(&[]);
        crate::pvrtc::decode_face(
            face_bytes,
            t.width,
            t.height,
            variant,
            swizzled_type(t.tex_type),
            out,
        );
        return true;
    }
    // UNCOMPRESSED: one texel per "block". Same hoist, different inner step - the per-texel
    // path re-derives the block layout, the PVRTC test, the swizzle mode and the two
    // power-of-two paddings for EVERY texel, and MEASURED on a mid-race desktop frame this
    // family is 2.11 MB of the 3.83 MB decoded, i.e. the larger half.
    if block_w == 1 && block_h == 1 {
        let face_base = (face * t.face_bytes) as usize;
        // Both constants of the FACE - see `SingleChannel`, and `decode_uncompressed_at` for
        // the fifteen-arm dispatch this is hoisting out of the texel loop.
        let single = SingleChannel::of(t.base_format);
        let swz = (t.swizzle >> 12) & 0x7;
        if swizzled_type(t.tex_type) {
            // The interleave once per row and once per column instead of once per texel - see
            // `morton_tables`, which is asserted to be the same function as `morton_index`.
            let (pw, ph) = (t.width.next_power_of_two(), t.height.next_power_of_two());
            let (xs, ys) = morton_tables(pw, ph, t.width, t.height);
            // The same identity as the linear branch below: a swizzled `U8U8U8U8` in swizzle 0
            // still decodes to its own four bytes, so only the ADDRESSING is Morton. Worth
            // splitting because this is the other half of the paletted working set.
            let identity32 =
                block_bytes == 4 && t.base_format == U8U8U8U8 && ((t.swizzle >> 12) & 0x7) == 0;
            let mut o = 0usize;
            for y in 0..t.height {
                let yb = ys[y as usize];
                for x in 0..t.width {
                    let off = face_base + ((xs[x as usize] + yb) * block_bytes) as usize;
                    match (identity32, t.pixels.get(off..off + 4)) {
                        (true, Some(src)) => out[o..o + 4].copy_from_slice(src),
                        _ => match single {
                            Some(k) => out[o..o + 4]
                                .copy_from_slice(&swizzle1(k.lane(&t.pixels, off), swz)),
                            None => out[o..o + 4].copy_from_slice(&decode_uncompressed_at(t, off)),
                        },
                    }
                    o += 4;
                }
            }
        } else {
            // >>> A LINEAR U8U8U8U8 IN THE IDENTITY SWIZZLE IS A ROW COPY.
            // SWIZZLE4 selector 0 (ABGR) is the identity permutation, so
            // `decode_uncompressed_at` returns the four memory bytes unchanged - the row is
            // already the RGBA8 this function is producing. This is the shape a PALETTED
            // texture arrives in after `expand_paletted_texture` (which writes `U8U8U8U8` and
            // keeps the guest's swizzle), and that is the golf title's whole texture working
            // set: 608 MB of the run's 618 MB of decode. Walking it per texel through a format
            // match was 15% of the browser worker on top of the addressing above.
            //
            // ONLY `U8U8U8U8`, deliberately. `0x0c..=0x1a` is the arm that catches the 32-bit
            // four-channel family, but it is a FALLTHROUGH: `0x0e`, `0x0f..=0x11`, `0x12/0x13`,
            // `0x15`, `0x17/0x18` are all inside that range and are decoded by their own arms
            // above it. Taking the range at face value made `0x0e` (U2U10U10U10) copy its raw
            // bytes, which `uncompressed_fast_path_matches_per_texel` caught immediately.
            let identity32 =
                block_bytes == 4 && t.base_format == U8U8U8U8 && ((t.swizzle >> 12) & 0x7) == 0;
            let mut o = 0usize;
            for y in 0..t.height {
                let row = face_base + (y * t.stride) as usize;
                let n = (t.width * 4) as usize;
                if identity32 {
                    match t.pixels.get(row..row + n) {
                        Some(src) => out[o..o + n].copy_from_slice(src),
                        // Short source: fall back to the per-texel walk for THIS row, which
                        // zero-fills past the end exactly as it always did.
                        None => {
                            for x in 0..t.width {
                                let off = row + (x * block_bytes) as usize;
                                out[o + (x * 4) as usize..o + (x * 4) as usize + 4]
                                    .copy_from_slice(&decode_uncompressed_at(t, off));
                            }
                        }
                    }
                    o += n;
                    continue;
                }
                // The single-channel walk is split out rather than branched inside the
                // loop so the reduction and the SWIZZLE1 routing are each decided once for
                // the whole row instead of once per texel.
                if let Some(k) = single {
                    for x in 0..t.width {
                        let off = row + (x * block_bytes) as usize;
                        out[o..o + 4].copy_from_slice(&swizzle1(k.lane(&t.pixels, off), swz));
                        o += 4;
                    }
                    continue;
                }
                for x in 0..t.width {
                    let off = row + (x * block_bytes) as usize;
                    out[o..o + 4].copy_from_slice(&decode_uncompressed_at(t, off));
                    o += 4;
                }
            }
        }
        return true;
    }
    if block_w <= 1 {
        return false;
    }
    let face_base = (face * t.face_bytes) as usize;
    let swizzle = (t.swizzle >> 12) & 0x7;
    let swizzled = swizzled_type(t.tex_type);
    let (pw, ph) = if swizzled {
        (t.width.div_ceil(block_w).next_power_of_two(), t.height.div_ceil(block_h).next_power_of_two())
    } else {
        (0, 0)
    };
    let blocks_x = t.width.div_ceil(block_w);
    let blocks_y = t.height.div_ceil(block_h);
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block_index = if swizzled {
                morton_index(bx, by, pw, ph)
            } else {
                by * (t.stride / block_bytes) + bx
            };
            let off = face_base + (block_index * block_bytes) as usize;
            let block = t.pixels.get(off..off + block_bytes as usize).unwrap_or(&[]);
            // ONE palette per block - see `decode_bc_block`.
            let texels = decode_bc_block(block, t.base_format);
            // The trailing blocks of a non-multiple-of-4 texture are partly outside the
            // image; only the texels inside it are written, exactly as the per-texel loop
            // (which never asks for the others) would.
            for py in 0..block_h.min(t.height - by * block_h) {
                let y = by * block_h + py;
                for px in 0..block_w.min(t.width - bx * block_w) {
                    let x = bx * block_w + px;
                    let c = texels[(py * 4 + px) as usize];
                    let o = ((y * t.width + x) * 4) as usize;
                    out[o..o + 4].copy_from_slice(&swizzle4(c[0], c[1], c[2], c[3], swizzle));
                }
            }
        }
    }
    true
}

/// Decode the single texel at integer coordinates `(x, y)` (already wrapped/clamped
/// into `[0, width) x [0, height)`) to straight RGBA8. Handles the block-compressed
/// (BC/DXT, optionally Morton-swizzled) and the uncompressed format families; an
/// unknown format returns opaque magenta so it is visible, not silent.
fn texel_rgba(t: &BoundTexture, x: u32, y: u32) -> [u8; 4] {
    texel_rgba_face(t, 0, x, y)
}

/// [`texel_rgba`] for a chosen `face` of a cube map (face 0 for an ordinary texture). Faces
/// are stored back to back, each laid out exactly like a standalone texture of the same size,
/// so a face is just a byte offset applied to every fetch.
fn texel_rgba_face(t: &BoundTexture, face: u32, x: u32, y: u32) -> [u8; 4] {
    let Some((block_w, block_h, block_bytes)) = block_layout(t.base_format) else {
        report_undecodable_texture_format(t.base_format, t.tex_type);
        return [255, 0, 255, 255];
    };
    let face_base = (face * t.face_bytes) as usize;
    // PVRTC is block-based but NOT block-local: a texel needs the four blocks whose centres
    // surround it, so it cannot go through the single-block path below.
    if let Some(variant) = crate::pvrtc::Variant::from_base_format(t.base_format) {
        let face = t.pixels.get(face_base..).unwrap_or(&[]);
        return crate::pvrtc::texel(face, t.width, t.height, x, y, variant, swizzled_type(t.tex_type));
    }
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
        let off = face_base + (block_index * block_bytes) as usize;
        let block = t.pixels.get(off..off + block_bytes as usize).unwrap_or(&[]);
        let rgba = decode_bc_texel(block, t.base_format, x % block_w, y % block_h);
        let swizzle = (t.swizzle >> 12) & 0x7;
        return swizzle4(rgba[0], rgba[1], rgba[2], rgba[3], swizzle);
    }
    // Uncompressed. A SWIZZLED texture is Morton-addressed over a power-of-two-padded grid
    // exactly like the block-compressed case above - the only difference is that the "block"
    // is one texel, so the interleave runs over texel coordinates directly. Reading one
    // row-major instead scrambles it into blocky noise that still carries the right COLOURS,
    // which is what made this look like intentional "data readout" art rather than a bug: a
    // retail title's small cyan-on-black UI panels came out as static while every other
    // texture on the same screen was fine, because the others were LINEAR (type 3), solid
    // 8x8 fills (where a permutation is invisible), or block-compressed (already handled).
    let off = if swizzled_type(t.tex_type) {
        let pw = t.width.next_power_of_two();
        let ph = t.height.next_power_of_two();
        face_base + (morton_index(x, y, pw, ph) * block_bytes) as usize
    } else {
        face_base + (y * t.stride + x * block_bytes) as usize
    };
    decode_uncompressed_at(t, off)
}

/// Decode ONE uncompressed texel, given the byte offset its lanes start at.
///
/// Split out of [`texel_rgba_face`] so a whole-face decode can compute that offset in a
/// tight loop instead of re-deriving the block layout, the PVRTC test, the swizzle mode and
/// the power-of-two padding for every texel. The arithmetic below is unchanged, so the two
/// callers cannot disagree.
fn decode_uncompressed_at(t: &BoundTexture, off: usize) -> [u8; 4] {
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
        // Two-channel 16-bit lanes (U16U16 / S16S16 / F16F16), 32 bits total. These sit
        // inside the byte-wise 8888 range below but are NOT four 8-bit lanes: decoding them
        // as bytes splits each 16-bit value in half and produces noise, so they are pulled
        // out ahead of it. SWIZZLE2, low lane first.
        0x0f | 0x10 | 0x11 => {
            let lane = |i: usize| -> u8 {
                let raw = u16::from_le_bytes([byte(i * 2), byte(i * 2 + 1)]);
                let v = match t.base_format {
                    0x0f => raw as f32 / 65535.0,
                    0x10 => ((raw as i16) as f32 / 32767.0).max(0.0),
                    _ => half_to_f32(raw),
                };
                (v.clamp(0.0, 1.0) * 255.0).round() as u8
            };
            swizzle2(lane(0), lane(1), swizzle)
        }
        // Single-channel 32-bit (F32 / F32M / U32 / S32). One value, routed by SWIZZLE1.
        // F32M is F32 with the sign bit used as a flag; the magnitude is what a colour
        // read wants, so it decodes as the absolute value.
        0x12 | 0x13 | 0x17 | 0x18 => {
            let raw = u32::from_le_bytes([byte(0), byte(1), byte(2), byte(3)]);
            let v = match t.base_format {
                0x12 => f32::from_bits(raw),
                0x13 => f32::from_bits(raw).abs(),
                0x17 => raw as f32 / u32::MAX as f32,
                _ => ((raw as i32) as f32 / i32::MAX as f32).max(0.0),
            };
            swizzle1((v.clamp(0.0, 1.0) * 255.0).round() as u8, swizzle)
        }
        // X8U24: an 8-bit stencil byte over a 24-bit unsigned depth. SWIZZLE2 names the two
        // (SD / DS); the depth is the value a colour read means, normalized over 24 bits.
        0x15 => {
            let raw = u32::from_le_bytes([byte(0), byte(1), byte(2), byte(3)]);
            let d = ((raw & 0x00ff_ffff) as f32 / 16_777_215.0 * 255.0).round() as u8;
            let s = (raw >> 24) as u8;
            match swizzle {
                1 => [d, s, 0, 255], // DS
                _ => [s, d, 0, 255], // SD
            }
        }
        // U2U10U10U10: three 10-bit lanes under a 2-bit alpha, permuted by SWIZZLE4 with the
        // 2-bit lane in the alpha role (as U1U5U5U5 does with its 1-bit lane).
        0x0e => {
            let w = u32::from_le_bytes([byte(0), byte(1), byte(2), byte(3)]);
            let ten = |sh: u32| ((((w >> sh) & 0x3ff) as f32 / 1023.0) * 255.0).round() as u8;
            let a = (((w >> 30) & 0x3) as f32 / 3.0 * 255.0).round() as u8;
            match swizzle {
                1 => [ten(20), ten(10), ten(0), a], // ARGB
                _ => [ten(0), ten(10), ten(20), a], // ABGR
            }
        }
        // U2F10F10F10: three unsigned packed 10-bit FLOATS under a 2-bit unorm alpha - the
        // float sibling of U2U10U10U10 above, and it shares that format's lane order (the
        // 2-bit lane at the top, in the alpha role, so SWIZZLE4 applies unchanged). Each
        // 10-bit lane is 5 bits of exponent over 5 of mantissa, bias 15, no sign - the same
        // encoding as F11F11F10's third lane.
        //
        // Without this arm every draw sampling one was painted MAGENTA, which is what a golf
        // title's club shaft was.
        0x9a => {
            let w = u32::from_le_bytes([byte(0), byte(1), byte(2), byte(3)]);
            let ten = |sh: u32| -> u8 {
                let bits = (w >> sh) & 0x3ff;
                let (exp, mant) = (bits >> 5, bits & 0x1f);
                let v = if exp == 0 {
                    // Denormal: no implicit leading one.
                    mant as f32 / 32.0 * 2f32.powi(-14)
                } else {
                    (1.0 + mant as f32 / 32.0) * 2f32.powi(exp as i32 - 15)
                };
                (v.clamp(0.0, 1.0) * 255.0).round() as u8
            };
            let a = (((w >> 30) & 0x3) as f32 / 3.0 * 255.0).round() as u8;
            match swizzle {
                1 => [ten(20), ten(10), ten(0), a], // ARGB
                _ => [ten(0), ten(10), ten(20), a], // ABGR
            }
        }
        // F11F11F10: three unsigned packed floats (no sign bit) - 11/11/10 bits from the
        // low end, exponent bias 15, exactly the half-float layout minus the sign and with
        // a shorter mantissa. SWIZZLE3 BGR/RGB.
        0x1a => {
            let w = u32::from_le_bytes([byte(0), byte(1), byte(2), byte(3)]);
            let small = |bits: u32, mant_bits: u32| -> f32 {
                let exp = bits >> mant_bits;
                let mant = bits & ((1 << mant_bits) - 1);
                let scale = (1u32 << mant_bits) as f32;
                if exp == 0 {
                    // Denormal: no implicit leading one.
                    mant as f32 / scale * 2f32.powi(-14)
                } else {
                    (1.0 + mant as f32 / scale) * 2f32.powi(exp as i32 - 15)
                }
            };
            let c0 = small(w & 0x7ff, 6);
            let c1 = small((w >> 11) & 0x7ff, 6);
            let c2 = small((w >> 22) & 0x3ff, 5);
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            match swizzle {
                1 => [q(c2), q(c1), q(c0), 255], // RGB
                _ => [q(c0), q(c1), q(c2), 255], // BGR
            }
        }
        // SE5M9M9M9: three 9-bit mantissas sharing one 5-bit exponent (bias 15, no implicit
        // leading one - the RGB9E5 layout). SWIZZLE3 BGR/RGB.
        0x19 => {
            let w = u32::from_le_bytes([byte(0), byte(1), byte(2), byte(3)]);
            let scale = 2f32.powi((w >> 27) as i32 - 15 - 9);
            let m = |sh: u32| (((w >> sh) & 0x1ff) as f32 * scale).clamp(0.0, 1.0);
            let q = |v: f32| (v * 255.0).round() as u8;
            match swizzle {
                1 => [q(m(18)), q(m(9)), q(m(0)), 255], // RGB
                _ => [q(m(0)), q(m(9)), q(m(18)), 255], // BGR
            }
        }
        // 32-bit four-channel (U8U8U8U8 et al). SWIZZLE4 permutes the memory bytes.
        0x0c..=0x1a => swizzle4(byte(0), byte(1), byte(2), byte(3), swizzle),
        // U8U3U3U2: one 16-bit word, lanes MSB->LSB 8 / 3 / 3 / 2, so the same SWIZZLE4
        // selector applies with the 8-bit lane in the b3 role.
        0x03 => {
            let w = u16::from_le_bytes([byte(0), byte(1)]);
            let b3 = (w >> 8) as u8;
            let b2 = ((((w >> 5) & 0x7) as u32 * 255) / 7) as u8;
            let b1 = ((((w >> 2) & 0x7) as u32 * 255) / 7) as u8;
            let b0 = (((w & 0x3) as u32 * 255) / 3) as u8;
            swizzle4(b0, b1, b2, b3, swizzle)
        }
        // S5S5U6: opaque 16-bit three-channel, lanes MSB->LSB 5 / 5 / 6 with the two 5-bit
        // lanes SIGNED. SWIZZLE3 names them BGR(0)/RGB(1) as U5U6U5 does.
        0x06 => {
            let w = u16::from_le_bytes([byte(0), byte(1)]);
            let s5 = |v: u32| -> u8 {
                // Sign-extend a 5-bit two's-complement lane, then map [-1,1] -> [0,255].
                let s = if v & 0x10 != 0 { v as i32 - 32 } else { v as i32 };
                (((s as f32 / 15.0).clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8
            };
            let hi = s5((w >> 11) as u32 & 0x1f);
            let mid = s5((w >> 6) as u32 & 0x1f);
            let lo = (((w & 0x3f) as u32 * 255) / 63) as u8;
            match swizzle {
                1 => [hi, mid, lo, 255], // RGB
                _ => [lo, mid, hi, 255], // BGR
            }
        }
        // Two-channel 8-bit (U8U8 / S8S8): SWIZZLE2 over (low byte, high byte).
        0x07 | 0x08 => {
            let lane = |b: u8| -> u8 {
                if t.base_format == 0x08 {
                    (((b as i8) as f32 / 127.0).max(0.0) * 255.0).round() as u8
                } else {
                    b
                }
            };
            swizzle2(lane(byte(0)), lane(byte(1)), swizzle)
        }
        // Single-channel 16-bit (U16 / S16 / F16), reduced to 8 bits for the shared RGBA8
        // seam and routed by SWIZZLE1 exactly as the 8-bit single-channel case below.
        0x09 | 0x0a | 0x0b => {
            let raw = u16::from_le_bytes([byte(0), byte(1)]);
            let lane = match t.base_format {
                0x09 => unorm16_to_u8(raw),
                0x0a => snorm16_to_u8(raw),
                // F16 is not a fixed-point range, so it keeps the float reduction. It is also
                // not the family this costs anything on - see `unorm16_to_u8`.
                _ => (half_to_f32(raw).clamp(0.0, 1.0) * 255.0).round() as u8,
            };
            swizzle1(lane, swizzle)
        }
        // Two-channel 32-bit lanes (F32F32 / U32U32), 64 bits total. SWIZZLE2.
        0x1e | 0x1f => {
            let lane = |i: usize| -> u8 {
                let raw = u32::from_le_bytes([byte(i * 4), byte(i * 4 + 1), byte(i * 4 + 2), byte(i * 4 + 3)]);
                let v = match t.base_format {
                    0x1e => f32::from_bits(raw),
                    _ => raw as f32 / u32::MAX as f32,
                };
                (v.clamp(0.0, 1.0) * 255.0).round() as u8
            };
            swizzle2(lane(0), lane(1), swizzle)
        }
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
        // 64-bit four-channel. The four lanes sit in memory order exactly as the 8888 case's
        // four bytes do, so the same SWIZZLE4 selector permutes them. Each lane is reduced to
        // 8 bits for the shared RGBA8 texture seam: F16 saturates to [0,1] (these are HDR
        // lookup tables - a value above 1.0 clamps, which the seam cannot represent), while
        // U16/S16 are normalized ranges that map exactly.
        0x1b | 0x1c | 0x1d => {
            let lane = |i: usize| -> u8 {
                let raw = u16::from_le_bytes([byte(i * 2), byte(i * 2 + 1)]);
                match t.base_format {
                    0x1b => (half_to_f32(raw).clamp(0.0, 1.0) * 255.0).round() as u8,
                    0x1c => unorm16_to_u8(raw),
                    _ => snorm16_to_u8(raw),
                }
            };
            swizzle4(lane(0), lane(1), lane(2), lane(3), swizzle)
        }
        // Single channel U8/S8 (fonts, coverage masks): route the one channel to RGBA
        // per the format's SWIZZLE1 selector. Font atlases are typically RRRR (coverage
        // in every channel, so alpha carries it) or 000R/111R (coverage in alpha);
        // forcing alpha to 255 would turn the transparent inter-glyph gaps into opaque
        // boxes that overwrite neighbouring glyphs.
        0x00 | 0x01 => swizzle1(byte(0), swizzle),
        // Unknown format: opaque magenta, and it says so. Magenta on screen is only useful
        // if it can be traced back to a format number, and a texel decoder is far too hot to
        // report per call - so the report is deduped by format and printed once.
        _ => {
            report_undecodable_texture_format(t.base_format, t.tex_type);
            [255, 0, 255, 255]
        }
    }
}

/// Report - once per distinct base format, unconditionally - that a texture's format has no
/// texel decode, so every draw sampling it is painted magenta.
///
/// The counterpart of the capture side's unsized-format report: that one covers a format
/// whose SIZE is unknown (the unit ends up unbound), this one a format that is sized but not
/// decodable (the unit binds a magenta image). Both are silent failures otherwise, and they
/// look nothing alike on screen.
pub(crate) fn report_undecodable_texture_format(base_format: u32, tex_type: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(base_format) {
        return;
    }
    eprintln!(
        "gxm texture: base format {base_format:#04x} (type {tex_type}) has no texel decode - \
         every draw sampling it is painted MAGENTA"
    );
}

/// A 16-bit UNORM lane reduced to the shared RGBA8 seam, in integers.
///
/// >>> THIS IS THE SAME NUMBER `(raw as f32 / 65535.0 * 255.0).round()` PRODUCES, and it is
/// >>> asserted to be over all 65,536 inputs by
/// >>> `the_integer_sixteen_bit_reductions_match_the_float_ones`.
///
/// Exactly, not nearly: the float form can only differ where the true value lands on a half,
/// which needs `510 * raw == (2k+1) * 65535`; 65535 and 510 share the factor 255, so that
/// reduces to `2 * raw == (2k+1) * 257` - an even number equal to an odd one, which has no
/// solution. So round-to-nearest is unambiguous and the integer form is it.
///
/// # Why it is worth having at all
/// The float form is a divide, a clamp, a multiply and a call to `roundf` - a libm call in
/// wasm - PER TEXEL. MEASURED on a retail golf title, browser, real GPU: the single-channel
/// 16-bit family is **151.9 MB of the run's 270.5 MB of texture decode**, the largest of any
/// format, and `decode_uncompressed_at` was 2.4% of the whole worker thread in a GAMEPLAY
/// window where it decodes half a megabyte a frame. A load decodes hundreds of times that.
#[inline]
fn unorm16_to_u8(raw: u16) -> u8 {
    ((raw as u32 * 255 + 32767) / 65535) as u8
}

/// A 16-bit SNORM lane reduced the same way: negatives clamp to zero (the seam is unsigned),
/// and the positive half divides by 32767. Pinned by the same test, over all 65,536 inputs.
#[inline]
fn snorm16_to_u8(raw: u16) -> u8 {
    let v = raw as i16;
    if v <= 0 {
        return 0;
    }
    ((v as u32 * 255 + 16383) / 32767) as u8
}


/// Route a single-channel (U8/S8) texel to straight RGBA per its GXM `SWIZZLE1`
/// selector (already reduced to `(format >> 12) & 0x7` by the caller, exactly as
/// `swizzle4` receives its selector).
///
/// # THE SELECTOR NAMES ITS COMPONENTS HIGH-TO-LOW, i.e. ALPHA FIRST AND RED LAST
///
/// `SceGxmTextureSwizzle1Mode` gives the eight modes as `R`, `000R`, `111R`, `RRRR`, `0RRR`,
/// `1RRR`, `R000`, `R111`. The header names them and nothing more, and the four-letter names
/// read equally well in either direction - `R111` is "the byte in red, ones elsewhere" if the
/// name runs R,G,B,A, and "ones in rgb, the byte in ALPHA" if it runs A,B,G,R. The two readings
/// are exact mirrors of each other (`000R` <-> `R000`, `111R` <-> `R111`, `0RRR` <-> `RRR0`),
/// so no amount of re-reading the header settles it. **This used to take the first reading, and
/// it was wrong.**
///
/// MEASURED on PCSA00009, which is the only kind of evidence there is here. Its single-channel
/// textures use exactly three of the eight modes - `1RRR` x485, `R000` x1052 and `R111` x5 in
/// one frame - and under the low-to-high reading those decode to `[1,r,r,r]`, `[r,0,0,0]` and
/// `[r,1,1,1]`: one of them opaque with the byte in three channels, one of them **alpha zero on
/// a thousand bindings**, and one of them a texture whose only varying channel is RED. The last
/// is the title's GLYPH ATLAS, and its fragment program is a flat `texel * vertex_colour` on all
/// four channels, so that reading paints every string as a CYAN BOX with pale letters - which is
/// exactly what the screen showed.
///
/// Under the high-to-low reading the same three modes are `1RRR` = opaque greyscale, `R000` =
/// black with the byte as coverage, `R111` = white with the byte as coverage: the three roles a
/// single-channel texture actually has (a luminance/detail map, a dark mask, a font atlas). The
/// glyph atlas then composites as `vertex_colour` with `alpha = coverage`, and the text renders.
///
/// `RRRR` is identical either way, and no title in any captured corpus uses `000R`, `111R` or
/// `0RRR` - so the mirrored pair below is pinned by the modes that ARE used and left consistent
/// for the ones that are not.
///
/// `SWIZZLE2` carries the same ambiguity and is NOT changed here: it has no measurement behind
/// it yet, and flipping a convention on a hunch is how the first reading got in.
fn swizzle1(r: u8, swizzle: u32) -> [u8; 4] {
    match swizzle {
        // A one-letter name says only which channel is defined, so it reads the same way round:
        // the byte in red, opaque.
        0 => [r, 0, 0, 255],     // R
        1 => [r, 0, 0, 0],       // 000R
        2 => [r, 255, 255, 255], // 111R
        3 => [r, r, r, r],       // RRRR
        4 => [r, r, r, 0],       // 0RRR
        5 => [r, r, r, 255],     // 1RRR - opaque greyscale
        6 => [0, 0, 0, r],       // R000 - black, the byte as coverage
        _ => [255, 255, 255, r], // R111 (7) - white, the byte as coverage: a font atlas
    }
}

/// Route a two-channel texel to straight RGBA per its GXM `SWIZZLE2` selector, with `r`
/// the low (first) lane and `g` the high (second) lane.
///
/// The selector names the output channels left to right, the same reading `swizzle1` uses,
/// and a channel the name does not mention is 0 - except alpha, which is opaque so a
/// two-channel texture does not render invisible.
fn swizzle2(r: u8, g: u8, swizzle: u32) -> [u8; 4] {
    match swizzle {
        1 => [0, 0, g, r],       // 00GR
        2 => [g, r, r, r],       // GRRR
        3 => [r, g, g, g],       // RGGG
        4 => [g, r, g, r],       // GRGR
        5 => [0, 0, r, g],       // 00RG
        _ => [g, r, 0, 255],     // GR
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

/// [`swizzle4`] over 16-bit lanes, for the half seam. The selector means the same thing - it
/// names CHANNELS, not byte widths - so the permutation is identical and only the lane type
/// differs; `swizzle4_agrees_with_swizzle4_u16` is the test that keeps the two from drifting.
fn swizzle4_u16(l0: u16, l1: u16, l2: u16, l3: u16, swizzle: u32) -> [u16; 4] {
    match swizzle {
        1 => [l2, l1, l0, l3], // ARGB
        2 => [l3, l2, l1, l0], // RGBA
        3 => [l1, l2, l3, l0], // BGRA
        _ => [l0, l1, l2, l3], // ABGR
    }
}

/// Byte offset of one texel of an UNCOMPRESSED texture, honouring the swizzled (Morton)
/// layout exactly as [`texel_rgba_face`] does.
///
/// Factored out rather than duplicated because a second copy of this addressing is how the
/// two decoders would come to disagree about where a texel is - and a texture read at the
/// wrong offset produces plausible values, not obvious garbage.
fn texel_byte_offset(t: &BoundTexture, face: u32, x: u32, y: u32) -> usize {
    let bytes = block_layout(t.base_format).map_or(4, |(_, _, b)| b);
    let face_base = (face * t.face_bytes) as usize;
    if swizzled_type(t.tex_type) {
        let pw = t.width.next_power_of_two();
        let ph = t.height.next_power_of_two();
        face_base + (morton_index(x, y, pw, ph) * bytes) as usize
    } else {
        face_base + (y * t.stride + x * bytes) as usize
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

/// Latch for the once-per-run notice that shader-expanded draws are being skipped.
static SKIPPED_EXPANDED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Latch for the once-per-run notice that a frame held a scene with no colour surface,
/// so [`render_frame_chain`] could not place it in the chain.
static UNPLACED_SCENE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Latch for the once-per-run notice that post-process passes are being skipped.
static SKIPPED_POST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Latch for the once-per-run notice that a pass was dropped for having a zero-sized
/// target.
static ZERO_SIZED_TARGET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Print `msg` the first time this latch is raised. Used for "the renderer cannot
/// reproduce this class of draw" notices, which are properties of the title's shaders and
/// so would otherwise repeat every frame forever.
fn report_once(latch: &std::sync::atomic::AtomicBool, msg: &str) {
    if !latch.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!("{msg}");
    }
}

/// Recover a draw's [`DrawInterp`] the same way for both render paths.
/// Whether a non-MVP draw's positions are already in NORMALIZED DEVICE coordinates: the
/// whole position bounding box lies inside [-1, 1] and spans a real area.
///
/// # Why this is not a refinement
/// "A texcoord means a pixel-space sprite" is a guess about what a 2D draw is for, and a
/// composite's fullscreen blit breaks it: it reads a texcoord (it is sampling the world
/// that earlier passes rendered) while its positions are the NDC corners `+-1`. Read as
/// PIXEL coordinates those corners are a FOUR-PIXEL quad in the top-left of the screen -
/// so the entire 3D view composited into a dot, and the finished frame was a correct HUD
/// over black. Which is indistinguishable, by eye, from the world never having been drawn.
///
/// The positions themselves settle it and no guess is needed: nothing measured in pixels
/// on a 960x544 panel fits inside a two-unit box, and a fullscreen NDC quad fills it
/// exactly. The area requirement keeps a genuinely sub-pixel 2D element (which would be
/// invisible either way) from being stretched across the screen.
fn positions_are_ndc(d: &Draw, layout: &Layout) -> bool {
    let stride = d.vertex_stride.max(1) as usize;
    let nverts = d.vertices.len() / stride.max(1);
    if nverts == 0 {
        return false;
    }
    let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
    for i in 0..nverts {
        let v = decode_vertex(d, layout, i);
        for k in 0..2 {
            if !v.pos[k].is_finite() {
                return false;
            }
            lo[k] = lo[k].min(v.pos[k]);
            hi[k] = hi[k].max(v.pos[k]);
        }
    }
    let inside = lo.iter().chain(hi.iter()).all(|c| c.abs() <= 1.001);
    let spans = (hi[0] - lo[0]) >= 0.5 || (hi[1] - lo[1]) >= 0.5;
    inside && spans
}

fn interpret_draw(d: &Draw) -> DrawInterp {
    let layout = layout_of(d);
    // Recover the draw's coordinate space (see `Space`). A 4x4 MVP uniform is the 3D
    // cube path (depth-tested, opaque). Otherwise a 2D draw: a texcoord marks a
    // pixel-space sprite, a bare position is an NDC fullscreen pass.
    let space = if d.uniforms.len() >= 16 {
        let mut m = [0f32; 16];
        m.copy_from_slice(&d.uniforms[..16]);
        Space::Mvp(m)
    } else if layout.uv_off.is_some() && !positions_are_ndc(d, &layout) {
        Space::Pixel
    } else {
        Space::Ndc
    };
    // Texture the draw only if it actually reads a texcoord; a sticky texture binding
    // left over from a previous draw must not tint an untextured fill.
    let textured = layout.uv_off.is_some() && !d.textures.is_empty();

    // Texcoord divisor. A 2D UI text quad can index a font atlas in PIXEL units (coords
    // well past 1), which we normalize by the texture size. A 3D mesh (MVP space) instead
    // uses normalized coords that legitimately exceed 1 to TILE the texture (REPEAT wrap),
    // e.g. a ground plane whose detail texture repeats ~17x - dividing those by the texture
    // size would collapse the whole surface onto one corner texel (a flat smear). So only
    // apply the texel-unit normalization to non-MVP (2D) draws; 3D UVs pass through and the
    // sampler's REPEAT wrap handles the tiling.
    let uv_div = match (textured, d.albedo()) {
        (true, Some(tex)) if !matches!(space, Space::Mvp(_)) => {
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
    //
    // A shader-expanded draw is skipped for the same reason one step earlier: its stream
    // holds sprite RECORDS, not vertices, so there is no primitive here to rasterize at
    // all. Joining the records as triangles stretches one textured smear across unrelated
    // sprite centres - which is exactly how a shadow-blob pass reads on screen as a
    // striped quad welded to the object casting it.
    let skip = d.shader_expanded || (layout.uv_off.is_none() && layout.color_off.is_none());
    DrawInterp { layout, space, textured, uv_div, skip }
}

/// One locatable object in a captured scene: the draws that share a model-to-world
/// placement, and where that placement puts them.
///
/// See [`locate_scene`] for why this is worth having.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectLoc {
    /// A stable identity for this object across frames: a hash of its OBJECT-SPACE
    /// geometry (each contributing draw's vertex and index bytes).
    ///
    /// The draw index cannot serve as an identity. A scene's draw list is rebuilt
    /// every frame and its length changes as things come into view, so "draw 13" is a
    /// different object one frame later - and a delta computed against it reports
    /// enormous motion for a world that barely moved. A rigid object's vertex buffer,
    /// by contrast, is in object space and does not change as the object moves: only
    /// its world matrix does, which is exactly the quantity being measured.
    pub id: u64,
    /// Indices into [`Scene::draws`] of every draw at this placement.
    pub draws: Vec<usize>,
    /// The model-to-world translation these draws share - the object's world
    /// position, straight out of the vertex program's reflected world matrix.
    pub world: [f32; 3],
    /// The object's HEADING: the compass bearing, in degrees, of its local +X and
    /// local +Z axes after the world matrix rotates them, measured in the world XZ
    /// plane. `None` for a degenerate (non-rotating) matrix.
    ///
    /// Position alone makes steering an integration problem: to learn which way a
    /// vehicle is pointing you have to drive it and watch where it ends up, which
    /// takes long enough that it drives into something first, and the answer is
    /// contaminated by the turn it made getting there. The rotation is already in the
    /// same matrix as the translation, so the heading is a direct reading taken in one
    /// frame. Both axes are reported because which one a given mesh calls "forward" is
    /// a property of that mesh, and the difference is a constant that cancels the
    /// moment you compare two readings.
    pub heading: Option<[f32; 2]>,
    /// Screen-space bounding box `[min_x, min_y, max_x, max_y]` over every projected
    /// vertex, in the requested raster size. `None` when nothing projected in front
    /// of the eye (entirely off-camera or behind it).
    pub screen: Option<[f32; 4]>,
    /// Centre of `screen`, for the common "where is it on screen" question.
    pub centroid: Option<[f32; 2]>,
    /// Mean projected view distance (the clip `w`) - how far from the camera it is.
    pub distance: Option<f32>,
    pub triangles: usize,
    /// This placement's draws are shader-expanded sprite records, so `screen` locates
    /// the sprite CENTRES and not a rasterized shape. Flagged rather than dropped:
    /// the centres are still where the object is.
    pub sprites: bool,
}

/// Where every object in a captured scene is, in world and on screen.
///
/// # Why this exists
/// An agent driving a game through this emulator is otherwise blind. It can take a
/// screenshot, but "the car is a few pixels left of the concrete circle" is not a
/// quantity - it cannot be compared, asserted on, or fed back into the next input, so
/// every navigation decision needs a human to look at a PNG. That is the slowest step
/// in the whole loop and it is what keeps a playthrough from running unattended.
///
/// The position is already in the capture and needs no reverse engineering at all:
/// [`Draw::world`] is the model-to-world matrix reflected from the vertex program, so
/// its translation column IS the object's world position, and the draw's MVP projects
/// it to exactly where the renderer puts it. Grouping draws by that placement turns a
/// scene of hundreds of draws into the handful of OBJECTS the frame actually contains
/// - one of which is the thing the player is steering.
///
/// This deliberately does not try to name anything. Which group is the player is a
/// title-specific fact, and the way to establish it is behavioural: hold the throttle,
/// see which world position moves. That belongs in a recipe, not in the engine.
///
impl Scene {
    /// Triangles this scene draws through a model-to-world matrix, i.e. how much WORLD
    /// it contains. Zero for a composite, a HUD pass or a fullscreen post pass, whose
    /// geometry is emitted straight in clip or pixel space.
    ///
    /// This is what lets [`Capture::world_scene`](crate::capture::Capture::world_scene)
    /// pick the world pass out of a multi-pass frame by content rather than by order.
    /// The same [`interpret_draw`] classification every observer here uses decides it,
    /// so "the scene `locate` reports on" and "the scene selected" can never disagree.
    pub fn world_triangles(&self) -> usize {
        self.draws
            .iter()
            .filter(|d| matches!(interpret_draw(d).space, Space::Mvp(_)))
            .map(triangle_count)
            .sum()
    }
}

/// `width`/`height` are the raster size the screen coordinates are expressed in.
pub fn locate_scene(scene: &Scene, width: u32, height: u32) -> Vec<ObjectLoc> {
    // Group by quantized world translation. Millimetre buckets: fine enough that two
    // genuinely distinct objects never merge, coarse enough that float noise in a
    // shared placement does not split one object into several.
    let key_of = |w: [f32; 3]| {
        [
            (w[0] * 1000.0).round() as i64,
            (w[1] * 1000.0).round() as i64,
            (w[2] * 1000.0).round() as i64,
        ]
    };
    let mut groups: HashMap<[i64; 3], ObjectLoc> = HashMap::new();
    for (di, d) in scene.draws.iter().enumerate() {
        let interp = interpret_draw(d);
        let Space::Mvp(mvp) = interp.space else { continue };
        let stride = d.vertex_stride.max(1) as usize;
        let nverts = d.vertices.len() / stride;
        if nverts == 0 {
            continue;
        }
        // Column-major 4x4: columns 0..2 are the rotated basis vectors, column 3 the
        // translation.
        let world = [d.world[12], d.world[13], d.world[14]];
        // Bearing of each in-plane basis axis, in the same convention the pad's polar
        // stick directive uses: 0 along world +X, increasing toward world -Z.
        let bearing = |x: f32, z: f32| -> f32 { (-z).atan2(x).to_degrees() };
        let ax = [d.world[0], d.world[2]];
        let az = [d.world[8], d.world[10]];
        let heading = if ax[0].hypot(ax[1]) > 1e-6 && az[0].hypot(az[1]) > 1e-6 {
            Some([bearing(ax[0], ax[1]), bearing(az[0], az[1])])
        } else {
            None
        };
        let mut bbox = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let mut wsum = 0.0f64;
        let mut wcount = 0usize;
        for i in 0..nverts {
            let v = decode_vertex(d, &interp.layout, i);
            let Some(p) = project(&v, &Space::Mvp(mvp), width, height, 1.0) else { continue };
            bbox[0] = bbox[0].min(p[0]);
            bbox[1] = bbox[1].min(p[1]);
            bbox[2] = bbox[2].max(p[0]);
            bbox[3] = bbox[3].max(p[1]);
            // `project` returns 1/w in slot 3; the view distance is its reciprocal.
            if p[3] > 0.0 {
                wsum += (1.0 / p[3]) as f64;
                wcount += 1;
            }
        }
        // FNV-1a over the draw's object-space geometry: the identity that survives the
        // draw list being rebuilt. Folded in per draw so a multi-draw object's id
        // covers all of its parts.
        let mut geo: u64 = 0xcbf2_9ce4_8422_2325;
        for b in d.vertices.iter().chain(d.indices.iter()) {
            geo ^= *b as u64;
            geo = geo.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let e = groups.entry(key_of(world)).or_insert_with(|| ObjectLoc {
            id: 0,
            draws: Vec::new(),
            world,
            heading,
            screen: None,
            centroid: None,
            distance: None,
            triangles: 0,
            sprites: false,
        });
        e.draws.push(di);
        // Commutative fold, so the id does not depend on the order draws were visited.
        e.id ^= geo;
        e.triangles += triangle_count(d);
        e.sprites |= d.shader_expanded;
        if wcount > 0 {
            let mean = (wsum / wcount as f64) as f32;
            e.distance = Some(match e.distance {
                Some(prev) => (prev + mean) * 0.5,
                None => mean,
            });
            e.screen = Some(match e.screen {
                Some(b) => [b[0].min(bbox[0]), b[1].min(bbox[1]), b[2].max(bbox[2]), b[3].max(bbox[3])],
                None => bbox,
            });
        }
    }
    let mut out: Vec<ObjectLoc> = groups.into_values().collect();
    for o in &mut out {
        o.draws.sort_unstable();
        o.centroid = o.screen.map(|b| [(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5]);
    }
    // Lowest draw index first, so the ordering is the scene's own submission order and
    // stays stable frame to frame - which is what makes two `locate`s comparable.
    out.sort_by_key(|o| o.draws.first().copied().unwrap_or(usize::MAX));
    out
}

/// How much a scene's coordinate ORIGIN moved between two [`locate_scene`] reports, and
/// how much of the scene agreed about it.
///
/// # Why this is not optional
/// The matrix a title calls "model to world" need not be measured from a fixed origin. On
/// one retail racer it is measured from a frame that travels with the camera - so while the
/// player drives, EVERY static object's reported position changes by the same vector, and
/// the player's own barely changes at all. Read naively that says the scenery is flying
/// past a stationary car, which inverts the one question a navigator asks. Worse, it is
/// not visibly wrong: the numbers are smooth, plausible, and self-consistent.
///
/// The scene itself supplies the correction. Most of what is in view is bolted down, so
/// the MODAL displacement across id-matched objects is the origin's own motion, and
/// subtracting it leaves true world motion. It is a mode rather than a mean because a
/// mean is dragged by the very objects being measured against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OriginDrift {
    /// The displacement of the coordinate origin, in the later report's units.
    pub delta: [f32; 3],
    /// How many matched objects agreed with `delta` (within `tol`).
    pub agreed: usize,
    /// How many objects were matched between the two reports at all.
    pub matched: usize,
}

impl OriginDrift {
    /// Whether the estimate can be trusted: a clear majority of a non-trivial sample
    /// agreed. A scene cut, or a frame in which almost everything genuinely moves, gives
    /// no majority - and then the honest answer is "unknown", not a plausible vector.
    pub fn reliable(&self) -> bool {
        self.matched >= 8 && self.agreed * 2 > self.matched
    }
}

/// The most common displacement in a set, and how many agreed with it.
///
/// A MODE rather than a mean, because a mean is dragged by the very objects being measured
/// against it. Found by taking the member with the most neighbours within `tol` and then
/// averaging that cluster - O(n^2) over a few hundred items, which is nothing, and it needs
/// no bin alignment (a histogram splits one cluster across two bins whenever it straddles
/// an edge).
fn modal_shift(deltas: &[[f32; 3]], tol: f32) -> Option<([f32; 3], usize)> {
    if deltas.is_empty() {
        return None;
    }
    let near = |a: &[f32; 3], b: &[f32; 3]| {
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() <= tol
    };
    let mut best = (0usize, 0usize);
    for (i, a) in deltas.iter().enumerate() {
        let n = deltas.iter().filter(|b| near(a, b)).count();
        if n > best.1 {
            best = (i, n);
        }
    }
    let centre = deltas[best.0];
    let mut sum = [0f64; 3];
    let mut n = 0usize;
    for b in deltas.iter().filter(|b| near(&centre, b)) {
        for k in 0..3 {
            sum[k] += b[k] as f64;
        }
        n += 1;
    }
    Some((
        [(sum[0] / n as f64) as f32, (sum[1] / n as f64) as f32, (sum[2] / n as f64) as f32],
        n,
    ))
}

/// Estimate the coordinate-origin displacement between two [`locate_scene`] reports.
/// `tol` is the world distance within which two displacements count as the same.
/// Returns `None` when nothing could be matched.
pub fn origin_drift(prev: &[ObjectLoc], now: &[ObjectLoc], tol: f32) -> Option<OriginDrift> {
    // Match by GEOMETRY id, and among same-id candidates (a row of identical cones) take
    // the nearest - over a short span nothing outruns its own spacing.
    let mut deltas: Vec<[f32; 3]> = Vec::new();
    for o in now {
        let best = prev
            .iter()
            .filter(|p| p.id == o.id)
            .map(|p| {
                let d = [o.world[0] - p.world[0], o.world[1] - p.world[1], o.world[2] - p.world[2]];
                (d, d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((d, _)) = best {
            deltas.push(d);
        }
    }
    let (delta, agreed) = modal_shift(&deltas, tol)?;
    Some(OriginDrift { delta, agreed, matched: deltas.len() })
}

/// One 2D drawn thing in a captured scene: where it is ON SCREEN, and an identity that
/// survives it moving.
///
/// # Why 3D locating cannot cover this
/// [`locate_scene`] needs a model-to-world matrix, and a 2D title has none: a sprite's
/// POSITION lives in its vertex data, in screen pixels. That breaks both halves of the 3D
/// approach at once - there is no placement to group by, and the geometry hash that gives a
/// 3D mesh its stable identity changes every single time a sprite moves, because the
/// geometry IS the position.
///
/// So identity comes from what does not change: the bound texture, the REGION of it this
/// quad samples, and the quad's size. That is exactly what makes one sprite in an atlas
/// distinguishable from another - two draws sampling different parts of the same sheet are
/// different sprites, and the same sprite drawn a hundred pixels along is the same sprite.
#[derive(Clone, Debug, PartialEq)]
pub struct SpriteLoc {
    pub id: u64,
    /// The draw this came from. One entry per draw: a 2D pass is normally one quad or one
    /// batch, and merging batches would throw away the positions that are the point.
    pub draw: usize,
    /// Screen bounding box `[min_x, min_y, max_x, max_y]` in the requested raster size.
    pub bbox: [f32; 4],
    pub centroid: [f32; 2],
    pub size: [f32; 2],
    pub triangles: usize,
    /// Whether the draw sampled a texture at all (an untextured 2D fill - a bar, a fade -
    /// has no atlas region, so its identity rests on shape alone and is weaker).
    pub textured: bool,
}

/// Where every 2D drawn thing in a captured scene is on screen. See [`SpriteLoc`].
///
/// `width`/`height` are the raster size the coordinates are expressed in.
pub fn locate_sprites(scene: &Scene, width: u32, height: u32) -> Vec<SpriteLoc> {
    let mut out = Vec::new();
    for (di, d) in scene.draws.iter().enumerate() {
        if triangle_count(d) == 0 {
            continue;
        }
        let interp = interpret_draw(d);
        // 3D draws belong to `locate_scene`; a shader-expanded stream has no primitive here.
        if matches!(interp.space, Space::Mvp(_)) || interp.skip {
            continue;
        }
        let stride = d.vertex_stride.max(1) as usize;
        let nverts = d.vertices.len() / stride;
        if nverts == 0 {
            continue;
        }
        let mut bbox = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let mut uv = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let mut seen = 0usize;
        for i in 0..nverts {
            let v = decode_vertex(d, &interp.layout, i);
            if let Some(p) = project(&v, &interp.space, width, height, 1.0) {
                bbox[0] = bbox[0].min(p[0]);
                bbox[1] = bbox[1].min(p[1]);
                bbox[2] = bbox[2].max(p[0]);
                bbox[3] = bbox[3].max(p[1]);
                seen += 1;
            }
            if interp.layout.uv_off.is_some() {
                uv[0] = uv[0].min(v.uv[0]);
                uv[1] = uv[1].min(v.uv[1]);
                uv[2] = uv[2].max(v.uv[0]);
                uv[3] = uv[3].max(v.uv[1]);
            }
        }
        if seen == 0 {
            continue;
        }
        let size = [bbox[2] - bbox[0], bbox[3] - bbox[1]];
        // Identity: the texture, the atlas region, and the shape - all of which are
        // properties of the sprite rather than of where it currently is. Quantized so
        // sub-pixel jitter in a scrolling layer does not split one sprite into two.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        let textured = d.albedo().is_some() && interp.layout.uv_off.is_some();
        // The BINDING, not the snapshot: see `tex_binding_key`. A sprite has to match itself
        // across frames, and every frame snapshots the atlas into a different buffer.
        mix(d.albedo().map(tex_binding_key).unwrap_or(0));
        for c in uv {
            mix(if c.is_finite() { (c * 4096.0).round() as i64 as u64 } else { 0 });
        }
        mix((size[0].round() as i64 as u64) << 32 | size[1].round() as i64 as u64);
        mix(d.index_count as u64);
        out.push(SpriteLoc {
            id: h,
            draw: di,
            bbox,
            centroid: [(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5],
            size,
            triangles: triangle_count(d),
            textured,
        });
    }
    out
}

/// How much a 2D scene SCROLLED between two [`locate_sprites`] reports.
///
/// The same problem the 3D path has, in screen space: when the camera pans, every
/// background sprite moves and the player - which the camera is following - appears not to.
/// Reported in pixels, from the modal displacement of id-matched sprites.
pub fn scroll_drift(prev: &[SpriteLoc], now: &[SpriteLoc], tol: f32) -> Option<OriginDrift> {
    let mut deltas: Vec<[f32; 3]> = Vec::new();
    for s in now {
        let best = prev
            .iter()
            .filter(|p| p.id == s.id)
            .map(|p| {
                let d = [s.centroid[0] - p.centroid[0], s.centroid[1] - p.centroid[1], 0.0];
                (d, d[0] * d[0] + d[1] * d[1])
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((d, _)) = best {
            deltas.push(d);
        }
    }
    let (delta, agreed) = modal_shift(&deltas, tol)?;
    Some(OriginDrift { delta, agreed, matched: deltas.len() })
}

/// A sprite's motion on screen with the scene's scroll removed, and its magnitude.
/// Matched against the scroll-corrected expected position, for the same reason
/// [`world_motion`] is.
pub fn sprite_motion(
    prev: &[SpriteLoc],
    now: &SpriteLoc,
    scroll: [f32; 3],
) -> Option<([f32; 2], f32)> {
    let expect = [now.centroid[0] - scroll[0], now.centroid[1] - scroll[1]];
    prev.iter()
        .filter(|p| p.id == now.id)
        .map(|p| {
            let d = [expect[0] - p.centroid[0], expect[1] - p.centroid[1]];
            (d, (d[0] * d[0] + d[1] * d[1]).sqrt())
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

/// One object's motion through the WORLD between two [`locate_scene`] reports: the
/// displacement with the origin's own drift removed, and its magnitude. `None` when the
/// object was not in the earlier report.
///
/// Matching is done against the DRIFT-CORRECTED expected position, which matters as soon
/// as a title repeats a mesh. A row of identical fence posts shares one geometry id, so
/// candidates can only be told apart by position - and if the origin drifted 23 units
/// while the posts stand 20 apart, matching on raw proximity pairs each post with its
/// NEIGHBOUR and reports the whole fence as moving. Removing the drift first puts every
/// static object back on top of its own previous position, where the nearest candidate is
/// itself.
pub fn world_motion(
    prev: &[ObjectLoc],
    now: &ObjectLoc,
    drift: [f32; 3],
) -> Option<([f32; 3], f32)> {
    let expect = [now.world[0] - drift[0], now.world[1] - drift[1], now.world[2] - drift[2]];
    prev.iter()
        .filter(|p| p.id == now.id)
        .map(|p| {
            let d = [expect[0] - p.world[0], expect[1] - p.world[1], expect[2] - p.world[2]];
            (d, (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

/// The orthographic top-down window a [`WorldMap`] covers, and the transform between
/// world XZ and map pixels.
///
/// Screen convention: +X is right, world -Z is UP the image. That is the same frame the
/// pad's polar stick directive and [`ObjectLoc::heading`] use (bearing 0 along world +X,
/// increasing toward world -Z), so a bearing read off the map is directly a bearing to
/// command - no per-title sign to rediscover.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapView {
    /// World-space window in the XZ plane: `[min_x, min_z, max_x, max_z]`.
    pub extent: [f32; 4],
    pub width: u32,
    pub height: u32,
}

impl MapView {
    /// World units per pixel, `[x, z]`.
    pub fn scale(&self) -> [f32; 2] {
        [
            (self.extent[2] - self.extent[0]) / self.width.max(1) as f32,
            (self.extent[3] - self.extent[1]) / self.height.max(1) as f32,
        ]
    }

    /// Map pixel (may be fractional or outside the image) for a world XZ position.
    /// Image Y grows DOWNWARD, so the most negative Z is at the top of the image.
    pub fn pixel_of(&self, wx: f32, wz: f32) -> [f32; 2] {
        let s = self.scale();
        [(wx - self.extent[0]) / s[0], (wz - self.extent[1]) / s[1]]
    }

    /// The world XZ position at a map pixel centre - the inverse of [`Self::pixel_of`].
    /// This is what makes the map a measuring instrument: a feature spotted at pixel
    /// (px, py) has an exact world coordinate to steer toward.
    pub fn world_of(&self, px: f32, py: f32) -> [f32; 2] {
        let s = self.scale();
        [self.extent[0] + px * s[0], self.extent[1] + py * s[1]]
    }
}

/// A top-down orthographic render of a captured scene's world geometry, with the height
/// of the topmost surface at every pixel.
///
/// # Why this exists
/// [`locate_scene`] answers "where is each OBJECT", which is enough to steer a vehicle
/// and to tell a moving thing from scenery. It cannot answer the two questions that
/// actually block a playthrough of a driving tutorial:
///
/// - **Where is the route?** On this title the trail the tutorial asks you to follow is
///   painted into the GROUND TEXTURE. It is not an object, has no placement matrix, and
///   so does not appear in a `locate` report at all. It does appear in a top-down render,
///   at a pixel this view converts straight back to a world coordinate.
/// - **What will I hit?** Internal railings and benches are what a hand-guessed waypoint
///   ring catches on. They are a metre of extra height over the ground, which
///   [`Self::top_y`] measures per pixel, so an obstacle is a reading rather than a
///   surprise.
///
/// The projection needs no reverse engineering: [`Draw::world`] is the reflected
/// model-to-world matrix, so transforming the object-space vertices by it gives true
/// world positions, and the ortho projection is ours to choose. Only 3D (`Mvp`) draws
/// take part - a 2D overlay has no world position to place.
pub struct WorldMap {
    pub view: MapView,
    /// The rendered image (same shading as the ordinary software render, so the ground
    /// markings look as they do in-game).
    pub fb: Framebuffer,
    /// World Y of the topmost surface at each pixel, row-major, `f32::NAN` where no
    /// geometry covered the pixel. A bird's-eye view keeps the HIGHEST surface, so this
    /// is a height field: ground where the pixel is open, higher where something stands
    /// on it.
    pub top_y: Vec<f32>,
}

impl WorldMap {
    /// The topmost surface height at a world XZ position, or `None` when that position is
    /// outside the view or no geometry covered it.
    pub fn height_at(&self, wx: f32, wz: f32) -> Option<f32> {
        let p = self.view.pixel_of(wx, wz);
        if p[0] < 0.0 || p[1] < 0.0 {
            return None;
        }
        let (x, y) = (p[0] as u32, p[1] as u32);
        if x >= self.fb.width || y >= self.fb.height {
            return None;
        }
        let h = self.top_y[(y * self.fb.width + x) as usize];
        if h.is_nan() { None } else { Some(h) }
    }

    /// The most common covered height, quantized to `bucket` world units - the ground
    /// level of the mapped area. Taken as a mode rather than a mean or a minimum because
    /// a scene contains both a large flat drivable surface and a few tall things, and it
    /// is the flat surface that defines "ground".
    pub fn ground_level(&self, bucket: f32) -> Option<f32> {
        let bucket = if bucket > 0.0 { bucket } else { 1.0 };
        let mut hist: HashMap<i64, u32> = HashMap::new();
        for h in self.top_y.iter().filter(|h| !h.is_nan()) {
            *hist.entry((h / bucket).round() as i64).or_insert(0) += 1;
        }
        hist.into_iter().max_by_key(|(_, n)| *n).map(|(k, _)| k as f32 * bucket)
    }

    /// An ASCII height field over `cols` x `rows` cells: a machine-readable obstacle map
    /// to plan a route against, in the same orientation as the image.
    ///
    /// A cell reports the MAXIMUM height in it, because for navigation the worst case in
    /// a cell is what matters. Legend, relative to `ground` (see [`Self::ground_level`]):
    /// `' '` nothing mapped, `'.'` at ground, `':'` up to `step`, `'+'` up to `4*step`,
    /// `'#'` higher. `step` is the height a vehicle can be expected to ignore - a kerb -
    /// so `.` and `:` are drivable and `+`/`#` are things to go around.
    pub fn height_grid(&self, cols: u32, rows: u32, ground: f32, step: f32) -> String {
        let (cols, rows) = (cols.max(1), rows.max(1));
        let step = if step > 0.0 { step } else { 1.0 };
        let mut out = String::with_capacity(((cols + 1) * rows) as usize);
        for r in 0..rows {
            for c in 0..cols {
                // Cell -> pixel span. Integer division deliberately: every pixel belongs
                // to exactly one cell, so nothing is sampled twice or missed.
                let x0 = c * self.fb.width / cols;
                let x1 = ((c + 1) * self.fb.width / cols).max(x0 + 1).min(self.fb.width);
                let y0 = r * self.fb.height / rows;
                let y1 = ((r + 1) * self.fb.height / rows).max(y0 + 1).min(self.fb.height);
                let mut peak = f32::NEG_INFINITY;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let h = self.top_y[(y * self.fb.width + x) as usize];
                        if !h.is_nan() && h > peak {
                            peak = h;
                        }
                    }
                }
                out.push(if peak == f32::NEG_INFINITY {
                    ' '
                } else {
                    let d = peak - ground;
                    if d <= step * 0.5 {
                        '.'
                    } else if d <= step {
                        ':'
                    } else if d <= step * 4.0 {
                        '+'
                    } else {
                        '#'
                    }
                });
            }
            out.push('\n');
        }
        out
    }

    /// How the mapped surface heights are distributed, as `(height, pixel_count)` bins of
    /// `bucket` world units, tallest first.
    ///
    /// This is the instrument that turns "pick a ceiling" from a guess into a reading. A
    /// scene with a drivable floor and a roof over part of it shows as two dense bands
    /// with a gap; a scene whose sky writes depth shows one enormous band far above
    /// everything else. Without it, a map that came out wrong looks exactly like a map of
    /// somewhere uninteresting.
    pub fn height_bins(&self, bucket: f32) -> Vec<(f32, u32)> {
        let bucket = if bucket > 0.0 { bucket } else { 1.0 };
        let mut hist: HashMap<i64, u32> = HashMap::new();
        for h in self.top_y.iter().filter(|h| !h.is_nan()) {
            *hist.entry((h / bucket).floor() as i64).or_insert(0) += 1;
        }
        let mut v: Vec<(f32, u32)> = hist.into_iter().map(|(k, n)| (k as f32 * bucket, n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }

    /// Stamp a hollow square marker into the image at a world position, so a rendered map
    /// carries the player's own location and the waypoints being steered to. `size` is the
    /// marker's half-extent in pixels.
    pub fn mark(&mut self, wx: f32, wz: f32, size: i32, rgb: [u8; 3]) {
        let p = self.view.pixel_of(wx, wz);
        let (cx, cy) = (p[0] as i32, p[1] as i32);
        let (w, h) = (self.fb.width as i32, self.fb.height as i32);
        for dy in -size..=size {
            for dx in -size..=size {
                // Outline only: a filled marker hides the very thing it points at.
                if dx.abs() != size && dy.abs() != size {
                    continue;
                }
                let (x, y) = (cx + dx, cy + dy);
                if x < 0 || y < 0 || x >= w || y >= h {
                    continue;
                }
                let i = ((y * w + x) * 4) as usize;
                self.fb.rgba[i..i + 3].copy_from_slice(&rgb);
                self.fb.rgba[i + 3] = 255;
            }
        }
    }
}

/// The world XZ extent of a scene's 3D geometry, robust to a skybox.
///
/// `keep` is the central fraction of vertices to cover (e.g. 0.98). A skydome or a
/// distant backdrop is a handful of vertices spanning kilometres, so a strict min/max
/// puts the playable area in four pixels; taking a percentile instead lets vertex DENSITY
/// decide, and the surface a game tessellates most is the one it is played on. Returns
/// `None` for a scene with no 3D draws.
pub fn world_extent(scene: &Scene, keep: f32) -> Option<[f32; 4]> {
    let mut xs: Vec<f32> = Vec::new();
    let mut zs: Vec<f32> = Vec::new();
    for d in &scene.draws {
        if !is_map_surface(d) {
            continue;
        }
        let interp = interpret_draw(d);
        let stride = d.vertex_stride.max(1) as usize;
        let nverts = d.vertices.len() / stride;
        for i in 0..nverts {
            let v = decode_vertex(d, &interp.layout, i);
            let w = transform(&d.world, v.pos[0], v.pos[1], v.pos[2]);
            if w[0].is_finite() && w[2].is_finite() {
                xs.push(w[0]);
                zs.push(w[2]);
            }
        }
    }
    if xs.is_empty() {
        return None;
    }
    let keep = keep.clamp(0.01, 1.0);
    let cut = ((1.0 - keep) * 0.5 * xs.len() as f32) as usize;
    let pick = |mut v: Vec<f32>| -> (f32, f32) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        (v[cut.min(v.len() - 1)], v[(v.len() - 1 - cut).min(v.len() - 1)])
    };
    let (x0, x1) = pick(xs);
    let (z0, z1) = pick(zs);
    // A degenerate axis (everything at one X) would divide by zero in the transform.
    let pad = |a: f32, b: f32| if (b - a).abs() < 1e-3 { (a - 1.0, b + 1.0) } else { (a, b) };
    let (x0, x1) = pad(x0, x1);
    let (z0, z1) = pad(z0, z1);
    Some([x0, z0, x1, z1])
}

/// A traversability mask over a [`WorldMap`]: which pixels a ground vehicle could stand
/// on, with room to spare.
///
/// Derived from the map's height field by SLOPE rather than by absolute height, because
/// absolute height cannot tell a ramp from a wall - a driveable slope and a kerb differ in
/// how fast the surface rises, not in where it ends up. Anything unmapped is impassable:
/// a hole in the map is a place nothing is known about, and routing through it would be
/// routing through a guess.
pub struct Traversable {
    pub width: u32,
    pub height: u32,
    /// Row-major, `true` where a vehicle of the requested clearance fits.
    pub open: Vec<bool>,
}

impl Traversable {
    /// Build the mask. `rise` is the largest height difference between neighbouring
    /// pixels that still counts as drivable ground - a slope limit in world units per
    /// map pixel. `clearance` erodes the result by that many pixels, so a route planned
    /// on it keeps a body's width away from what it must not touch; 0 hugs the walls.
    pub fn from_map(map: &WorldMap, rise: f32, clearance: u32) -> Traversable {
        let (w, h) = (map.fb.width, map.fb.height);
        let at = |x: u32, y: u32| -> f32 { map.top_y[(y * w + x) as usize] };
        let mut open = vec![false; (w * h) as usize];
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let c = at(x, y);
                if c.is_nan() {
                    continue;
                }
                // A rise to ANY 4-neighbour that exceeds the limit makes this a lip, a
                // kerb or the foot of a wall - all of which stop a vehicle.
                let ok = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)].iter().all(|&(nx, ny)| {
                    let n = at(nx, ny);
                    !n.is_nan() && (n - c).abs() <= rise
                });
                open[(y * w + x) as usize] = ok;
            }
        }
        // Erode. Done as `clearance` single-pixel passes: a Chebyshev erosion by N, which
        // is what "keep N pixels away from anything blocked" means.
        for _ in 0..clearance {
            let prev = open.clone();
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) as usize;
                    if !prev[i] {
                        continue;
                    }
                    let edge = x == 0 || y == 0 || x + 1 >= w || y + 1 >= h;
                    let blocked_neighbour = !edge
                        && [
                            (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                            (x - 1, y), (x + 1, y),
                            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1),
                        ]
                        .iter()
                        .any(|&(nx, ny)| !prev[(ny * w + nx) as usize]);
                    if edge || blocked_neighbour {
                        open[i] = false;
                    }
                }
            }
        }
        Traversable { width: w, height: h, open }
    }

    /// The mask as an RGBA image: open ground pale, blocked dark.
    ///
    /// A planner whose input cannot be looked at is a planner whose failures are all
    /// mysterious. "No route" and "a route straight through a fence" have the same cause -
    /// the mask disagreeing with the world - and one glance at this settles which.
    pub fn to_rgba(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.open.len() * 4);
        for o in &self.open {
            let v = if *o { 220u8 } else { 25u8 };
            out.extend_from_slice(&[v, v, v, 255]);
        }
        out
    }

    pub fn is_open(&self, x: i64, y: i64) -> bool {
        x >= 0
            && y >= 0
            && (x as u32) < self.width
            && (y as u32) < self.height
            && self.open[(y as u32 * self.width + x as u32) as usize]
    }

    /// How much of the mask is open, as a fraction - a sanity reading for the caller. A
    /// mask that came out at 0.01 means the slope limit or clearance is wrong, and a route
    /// failure is then a configuration problem rather than a walled-off destination.
    pub fn open_fraction(&self) -> f32 {
        if self.open.is_empty() {
            return 0.0;
        }
        self.open.iter().filter(|o| **o).count() as f32 / self.open.len() as f32
    }

    /// The nearest pixel to `(x, y)` within `radius` that satisfies `ok`, or `None`.
    ///
    /// Endpoints need this: a vehicle's own centre often sits on a pixel the mask calls
    /// blocked (it is pressed against a kerb, or the clearance erosion clipped it), and
    /// refusing to plan from where the player actually is would make the planner useless
    /// exactly when it is needed. Callers pass an `ok` stricter than "open" - see
    /// [`plan_route`], which requires the pixel to be REACHABLE, because the flat top of a
    /// wall is open ground that happens to be an island.
    pub fn nearest(&self, x: i64, y: i64, radius: u32, ok: impl Fn(i64, i64) -> bool) -> Option<(i64, i64)> {
        if ok(x, y) {
            return Some((x, y));
        }
        for r in 1..=radius as i64 {
            let mut best: Option<((i64, i64), i64)> = None;
            for dy in -r..=r {
                for dx in -r..=r {
                    // The ring at Chebyshev distance r.
                    if dx.abs() != r && dy.abs() != r {
                        continue;
                    }
                    if ok(x + dx, y + dy) {
                        let d2 = dx * dx + dy * dy;
                        if best.is_none_or(|(_, bd)| d2 < bd) {
                            best = Some(((x + dx, y + dy), d2));
                        }
                    }
                }
            }
            if let Some((p, _)) = best {
                return Some(p);
            }
        }
        None
    }
}

/// A route through a [`Traversable`] mask from `from` to `to`, in WORLD coordinates, or
/// `None` when the two are not connected through open ground.
///
/// # Why a planner and not a list of waypoints
/// Waypoints picked by eye off a map encode only what the eye noticed. Every railing,
/// bench and kerb between two of them is an obstacle the route knows nothing about, and
/// the vehicle finds it by driving into it - which is exactly how a hand-guessed ring of
/// waypoints wedges a car six times in one run. The height field already contains those
/// obstacles, so the route should be computed from it.
///
/// The cost field is flooded from the GOAL over the whole mask, and only then is the start
/// chosen - as the nearest pixel that flood actually reached. Snapping first and searching
/// afterwards looks equivalent and is not: a slope test cannot tell the flat top of a wall
/// from the floor, so "nearest open pixel" to a vehicle pressed against a railing can be
/// the top of the railing, and a search from there correctly finds no route at all.
///
/// The path is then simplified by line of sight: a point is dropped while the straight run
/// past it stays on open ground, so the result is the handful of turns the route actually
/// makes rather than one waypoint per pixel.
pub fn plan_route(
    map: &WorldMap,
    mask: &Traversable,
    from: [f32; 2],
    to: [f32; 2],
    snap_radius: u32,
) -> Option<Vec<[f32; 2]>> {
    let w = mask.width as i64;
    let px = |p: [f32; 2]| -> (i64, i64) {
        let q = map.view.pixel_of(p[0], p[1]);
        (q[0].round() as i64, q[1].round() as i64)
    };
    let gp = px(to);
    let goal = mask.nearest(gp.0, gp.1, snap_radius, |x, y| mask.is_open(x, y))?;

    // Dijkstra from the goal over open ground. Costs in thousandths of a pixel so the
    // diagonal step is exact enough as an integer.
    let n = (mask.width * mask.height) as usize;
    let idx = |x: i64, y: i64| -> usize { (y * w + x) as usize };
    let mut dist = vec![u32::MAX; n];
    let mut heap: std::collections::BinaryHeap<(std::cmp::Reverse<u32>, i64, i64)> =
        std::collections::BinaryHeap::new();
    dist[idx(goal.0, goal.1)] = 0;
    heap.push((std::cmp::Reverse(0), goal.0, goal.1));
    while let Some((std::cmp::Reverse(d), x, y)) = heap.pop() {
        if d > dist[idx(x, y)] {
            continue;
        }
        for (dx, dy) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
            let (nx, ny) = (x + dx, y + dy);
            if !mask.is_open(nx, ny) {
                continue;
            }
            // No corner-cutting: a diagonal step is legal only when both orthogonal
            // neighbours are open, or the route squeezes through a gap a body cannot.
            if dx != 0 && dy != 0 && !(mask.is_open(x + dx, y) && mask.is_open(x, y + dy)) {
                continue;
            }
            let step = if dx != 0 && dy != 0 { 1414 } else { 1000 };
            let nd = d.saturating_add(step);
            if nd < dist[idx(nx, ny)] {
                dist[idx(nx, ny)] = nd;
                heap.push((std::cmp::Reverse(nd), nx, ny));
            }
        }
    }

    let sp = px(from);
    let reached = |x: i64, y: i64| -> bool {
        mask.is_open(x, y) && dist[idx(x, y)] != u32::MAX
    };
    let start = mask.nearest(sp.0, sp.1, snap_radius, reached)?;

    // Walk downhill on the cost field. Guaranteed to terminate at the goal: every open
    // reached pixel except the goal has a neighbour with strictly smaller cost.
    let mut path: Vec<(i64, i64)> = vec![start];
    let mut cur = start;
    while cur != goal {
        let mut best: Option<((i64, i64), u32)> = None;
        for (dx, dy) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
            let (nx, ny) = (cur.0 + dx, cur.1 + dy);
            if !reached(nx, ny) {
                continue;
            }
            if dx != 0 && dy != 0 && !(mask.is_open(cur.0 + dx, cur.1) && mask.is_open(cur.0, cur.1 + dy)) {
                continue;
            }
            let d = dist[idx(nx, ny)];
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some(((nx, ny), d));
            }
        }
        match best {
            Some((p, d)) if d < dist[idx(cur.0, cur.1)] => {
                cur = p;
                path.push(cur);
            }
            // Cannot happen for a reached pixel, but a silent infinite loop here would be
            // far worse than an honest `None`.
            _ => return None,
        }
    }

    // Line of sight, sampled with the SAME rounding a caller walking the route would use.
    // Truncating integer division instead (the first version) samples a line up to a pixel
    // off the real one, which passes a leg that grazes blocked ground.
    let clear = |a: (i64, i64), b: (i64, i64)| -> bool {
        let steps = (b.0 - a.0).abs().max((b.1 - a.1).abs()).max(1);
        (0..=steps).all(|s| {
            let t = s as f64 / steps as f64;
            let x = (a.0 as f64 + (b.0 - a.0) as f64 * t).round() as i64;
            let y = (a.1 as f64 + (b.1 - a.1) as f64 * t).round() as i64;
            mask.is_open(x, y)
        })
    };
    let mut keep: Vec<(i64, i64)> = vec![path[0]];
    let mut i = 0usize;
    while i + 1 < path.len() {
        // The furthest point still in line of sight from the current one.
        let mut j = i + 1;
        let mut best = i + 1;
        while j < path.len() {
            if clear(path[i], path[j]) {
                best = j;
            }
            j += 1;
        }
        keep.push(path[best]);
        i = best;
    }
    Some(
        keep.iter()
            .map(|&(x, y)| {
                let wpt = map.view.world_of(x as f32, y as f32);
                [wpt[0], wpt[1]]
            })
            .collect(),
    )
}

/// Whether a draw is world SURFACE - something a top-down map should show and a vehicle
/// could stand on - rather than an overlay.
///
/// Two exclusions, both of which cost a map its meaning if skipped:
/// - Not 3D (`Mvp`), or a shader-expanded sprite stream: no world placement, or no
///   triangles in the stream to place.
/// - Depth writes disabled. A skydome is the highest geometry in the scene at every
///   single pixel, so a bird's-eye view that includes it is a picture of the inside of
///   the sky and its height field is the sky's. GXM titles draw the sky and other
///   backdrop/overlay passes with `SCE_GXM_DEPTH_WRITE_DISABLED` precisely because they
///   are not surfaces, which makes the guest's own render state the filter - no height
///   threshold to guess, no per-title constant.
fn is_map_surface(d: &Draw) -> bool {
    let interp = interpret_draw(d);
    matches!(interp.space, Space::Mvp(_))
        && !interp.skip
        && d.render_state.front_depth_write != SCE_GXM_DEPTH_WRITE_DISABLED
}

/// Where the camera is and which way it looks, recovered from a draw's own
/// world-to-clip matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eye {
    /// Camera position in world coordinates.
    pub pos: [f32; 3],
    /// Unit view direction in world coordinates.
    pub dir: [f32; 3],
    /// Compass bearing of `dir` in the `lang=`/`locate` convention (0 = world +X,
    /// increasing toward world -Z).
    pub bearing: f32,
}

/// Invert a column-major 4x4. `None` when singular.
fn invert4(m: &[f32; 16]) -> Option<[f32; 16]> {
    // Cofactor expansion, written out: a general inverse of a projection*view product,
    // which is not affine (its last row is a perspective divide), so the cheap
    // transpose-and-negate trick for rigid transforms does not apply here.
    let mut inv = [0f32; 16];
    inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
    inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
    inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
    inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
    inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
    inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];
    let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if !det.is_finite() || det.abs() < 1e-20 {
        return None;
    }
    for v in inv.iter_mut() {
        *v /= det;
    }
    Some(inv)
}

/// Recover the camera from a scene's world-to-clip matrix.
///
/// # Why this exists
/// Steering needs to know where the vehicle is and which way it points, and the obvious
/// source - an address in guest memory found by diffing two runs - is not dependable:
/// on a real title those matrices live in a per-frame scratch pool, so the slot that
/// tracked the car for two thousand frames silently stops updating and the reading
/// freezes at a plausible value. A controller cannot tell that apart from a car against
/// a wall, and both of this project's driving controllers were fooled by it.
///
/// The camera cannot go stale, because it is reconstructed from the matrix the guest
/// used to draw THIS frame. A chase camera sits behind the vehicle and looks where it is
/// going, so its position and bearing are the vehicle's, to within a car length - which
/// is well inside the width of a road.
///
/// The maths: the matrix `M` maps world to clip, and the eye is the view-space origin,
/// which a perspective projection sends to `(0, 0, c, 0)`. So `[eye; 1]` is parallel to
/// `M^-1 * (0, 0, 1, 0)`, and dehomogenizing that gives the world position exactly. The
/// direction is then the difference between the eye and any unprojected point on the
/// central view ray.
pub fn scene_eye(scene: &Scene) -> Option<Eye> {
    // The world-to-clip matrix SHARED BY THE MOST DRAWS, with triangles only as a tiebreak.
    //
    // A draw's matrix is MODEL-to-clip, so the world's own matrix is the one that many
    // separate draws happen to share - static scenery is dozens of draws with an identity
    // model transform, all carrying the same matrix - while a single articulated MODEL is
    // one or two draws carrying its own. Ranking by TRIANGLES therefore asks "what is the
    // biggest mesh on screen", which is not the same question, and on a retail racer the
    // answer is the player's own ship.
    //
    // MEASURED, and it is why a driving controller went blind for a quarter of the circuit:
    // in enclosed sections less of the track is visible, the player ship's 9,211 triangles
    // out-weigh what is left of the world, and the eye reconstructed from the SHIP's
    // model-to-clip matrix lands 11 units from the origin - in the ship's own frame, where
    // the camera really is 11 units behind it. `camera` then reported (0.70, 3.66, -11.04)
    // while a smaller pass in the same frame carried the true (1247, 154, 1200), and every
    // position, heading and sighting downstream was wrong. `camera --passes` shows both.
    //
    // Counting draws makes the discriminator structural rather than a threshold on world
    // scale: there is no "near the origin" constant here, and nothing per title.
    let mut tally: Vec<([f32; 16], (usize, usize))> = Vec::new();
    for d in &scene.draws {
        let Space::Mvp(m) = interpret_draw(d).space else { continue };
        let n = triangle_count(d);
        if n == 0 {
            continue;
        }
        match tally.iter_mut().find(|(k, _)| k == &m) {
            Some((_, c)) => {
                c.0 += 1;
                c.1 += n;
            }
            None => tally.push((m, (1, n))),
        }
    }
    let (m, _) = tally.into_iter().max_by_key(|&(_, c)| c)?;
    let inv = invert4(&m)?;
    let mul = |v: [f32; 4]| -> [f32; 4] {
        [
            inv[0] * v[0] + inv[4] * v[1] + inv[8] * v[2] + inv[12] * v[3],
            inv[1] * v[0] + inv[5] * v[1] + inv[9] * v[2] + inv[13] * v[3],
            inv[2] * v[0] + inv[6] * v[1] + inv[10] * v[2] + inv[14] * v[3],
            inv[3] * v[0] + inv[7] * v[1] + inv[11] * v[2] + inv[15] * v[3],
        ]
    };
    let e = mul([0.0, 0.0, 1.0, 0.0]);
    if e[3].abs() < 1e-12 {
        return None;
    }
    let pos = [e[0] / e[3], e[1] / e[3], e[2] / e[3]];
    // A point on the central view ray, part way into the scene. Which clip depth is
    // immaterial - every one of them is on the same ray - so this only has to be a depth
    // the projection actually maps.
    let f = mul([0.0, 0.0, 0.5, 1.0]);
    if f[3].abs() < 1e-12 {
        return None;
    }
    let mut dir = [f[0] / f[3] - pos[0], f[1] / f[3] - pos[1], f[2] / f[3] - pos[2]];
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if !len.is_finite() || len < 1e-9 {
        return None;
    }
    for c in dir.iter_mut() {
        *c /= len;
    }
    if !pos.iter().all(|c| c.is_finite()) {
        return None;
    }
    let bearing = (-dir[2]).atan2(dir[0]).to_degrees();
    Some(Eye { pos, dir, bearing })
}

/// One world-surface triangle, in world coordinates, tagged with what drew it.
///
/// See [`surface_at`] for why the tag is the important half.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceTri {
    /// Index into [`Scene::draws`].
    pub draw: usize,
    /// Guest address of the draw's albedo texture - a stable identity for the MATERIAL
    /// across draws and frames. 0 when the draw is untextured.
    pub tex: u32,
    /// The triangle's world-space vertices.
    pub v: [[f32; 3]; 3],
}

impl SurfaceTri {
    pub fn centroid(&self) -> [f32; 3] {
        [
            (self.v[0][0] + self.v[1][0] + self.v[2][0]) / 3.0,
            (self.v[0][1] + self.v[1][1] + self.v[2][1]) / 3.0,
            (self.v[0][2] + self.v[1][2] + self.v[2][2]) / 3.0,
        ]
    }

    /// Height of the triangle's plane at `(x, z)`, if `(x, z)` is inside it seen from
    /// above. Barycentric, so a point on an edge belongs to both neighbours - which is
    /// what a surface query wants.
    pub fn height_at(&self, x: f32, z: f32) -> Option<f32> {
        let (a, b, c) = (self.v[0], self.v[1], self.v[2]);
        let det = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2]);
        if det.abs() < 1e-9 {
            return None;
        }
        let l0 = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / det;
        let l1 = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / det;
        let l2 = 1.0 - l0 - l1;
        let eps = -1e-4;
        (l0 >= eps && l1 >= eps && l2 >= eps).then(|| l0 * a[1] + l1 * b[1] + l2 * c[1])
    }
}

/// Every world-surface triangle in a scene, in world coordinates.
///
/// # Why this exists
/// A height field says how high the ground is, and that is enough to route around a wall.
/// It is NOT enough to stay on a ROAD: a racing circuit and the grass beside it are the
/// same height, so a slope-based traversable mask calls the whole valley drivable and a
/// route through it drives straight across the infield. What separates them is the
/// MATERIAL - the road is a different surface drawn with a different texture - and that
/// is already in the capture, one field away from the geometry.
///
/// So this returns the triangles WITH their material tag, and [`surface_at`] turns "which
/// surface is the car standing on" into a reading rather than a guess: put the vehicle
/// somewhere it is certainly legal (a starting grid), ask what is underneath it, and every
/// later query for that tag is the drivable surface itself, at full extent, with no
/// threshold to tune.
pub fn surface_tris(scene: &Scene) -> Vec<SurfaceTri> {
    let mut out = Vec::new();
    for (di, d) in scene.draws.iter().enumerate() {
        if !is_map_surface(d) {
            continue;
        }
        let interp = interpret_draw(d);
        let tex = d.albedo().map(|t| t.data_addr).unwrap_or(0);
        let stride = d.vertex_stride.max(1) as usize;
        let nverts = d.vertices.len() / stride;
        if nverts == 0 {
            continue;
        }
        for t in 0..triangle_count(d) {
            let idx = tri_indices(d, t);
            if idx.iter().any(|&i| i >= nverts) {
                continue;
            }
            let mut v = [[0f32; 3]; 3];
            for (k, &i) in idx.iter().enumerate() {
                let vert = decode_vertex(d, &interp.layout, i);
                let w = transform(&d.world, vert.pos[0], vert.pos[1], vert.pos[2]);
                v[k] = [w[0], w[1], w[2]];
            }
            if v.iter().any(|p| p.iter().any(|c| !c.is_finite())) {
                continue;
            }
            out.push(SurfaceTri { draw: di, tex, v });
        }
    }
    out
}

/// The surface directly under a world XZ position: the HIGHEST triangle covering it that
/// is not above `ceiling`.
///
/// Highest-under-a-ceiling rather than simply highest, because a title that draws a roof
/// or a banner gantry with depth writes would otherwise answer with that. The ceiling is
/// the caller's statement of "the thing I am standing on is below this", which for a
/// vehicle is just above its own roof.
pub fn surface_at(tris: &[SurfaceTri], x: f32, z: f32, ceiling: Option<f32>) -> Option<(SurfaceTri, f32)> {
    let mut best: Option<(SurfaceTri, f32)> = None;
    for t in tris {
        let Some(y) = t.height_at(x, z) else { continue };
        if ceiling.is_some_and(|c| y > c) {
            continue;
        }
        if best.is_none_or(|(_, by)| y > by) {
            best = Some((*t, y));
        }
    }
    best
}

/// Render a scene's 3D geometry as a top-down map. See [`WorldMap`] for why.
///
/// Only surfaces take part ([`is_map_surface`]). `ceiling`, when given, additionally
/// drops any triangle lying entirely above that Y - the escape hatch for a title whose
/// sky or roof DOES write depth, and the way to see the floor of a covered area.
///
/// `origin` is subtracted from every world position, so `view.extent`, the height field
/// and `ceiling` are all measured from it. Passing a static object's placement (see
/// [`origin_drift`] for why one is needed) makes the map's coordinates stable from frame
/// to frame on a title whose own origin travels with the camera; `[0.0; 3]` maps raw
/// coordinates.
///
/// `ssaa` supersamples as [`render_scene_supersampled`] does; the height field is
/// downsampled by MAXIMUM (an obstacle must not be averaged away by the ground beside
/// it), while the image is box-averaged as usual.
pub fn render_map(
    scene: &Scene,
    view: MapView,
    clear: [u8; 4],
    ssaa: u32,
    ceiling: Option<f32>,
    origin: [f32; 3],
) -> WorldMap {
    let s = ssaa.clamp(1, 8);
    let (rw, rh) = (view.width.max(1) * s, view.height.max(1) * s);
    let raster = MapView { extent: view.extent, width: rw, height: rh };
    let mut fb = Framebuffer::new(rw, rh, clear);
    // The depth buffer holds NEGATED world Y, so the rasterizer's existing "smaller
    // wins" test keeps the HIGHEST surface - which is what looking down from above
    // means. Reusing the real triangle rasterizer (rather than a second one written for
    // maps) is what makes the map show the same textures and shading the game does.
    let mut depth = vec![f32::INFINITY; (rw * rh) as usize];
    for d in &scene.draws {
        let tri_count = triangle_count(d);
        if tri_count == 0 {
            continue;
        }
        if !is_map_surface(d) {
            continue;
        }
        let DrawInterp { layout, textured, uv_div, .. } = interpret_draw(d);
        let texture = if textured { d.albedo() } else { None };
        for t in 0..tri_count {
            let vs = tri_indices(d, t);
            let verts: Vec<Vertex> = vs.iter().map(|&i| decode_vertex(d, &layout, i)).collect();
            let mut screen = [[0f32; 4]; 3];
            let mut low = f32::INFINITY;
            for (k, v) in verts.iter().enumerate() {
                let mut w = transform(&d.world, v.pos[0], v.pos[1], v.pos[2]);
                if !w[0].is_finite() || !w[1].is_finite() || !w[2].is_finite() {
                    screen[k][3] = f32::NAN;
                    break;
                }
                for c in 0..3 {
                    w[c] -= origin[c];
                }
                let p = raster.pixel_of(w[0], w[2]);
                low = low.min(w[1]);
                // 1/w of exactly 1: an ortho projection has no perspective, and the
                // rasterizer's perspective-correct interpolation degenerates to the
                // correct affine one when every weight is equal.
                screen[k] = [p[0], p[1], -w[1], 1.0];
            }
            if screen.iter().any(|s| s[3].is_nan()) {
                continue;
            }
            // Entirely above the ceiling. Tested on the LOWEST vertex so a wall that
            // rises through the ceiling still contributes the part below it.
            if ceiling.is_some_and(|c| low > c) {
                continue;
            }
            // No back-face cull: the map wants whatever surface is on top, and culling by
            // the guest's winding here would punch holes in single-sided ground panels
            // seen from a direction the game never views them from.
            raster_triangle(
                &mut fb, &mut depth, &screen, &verts, texture, uv_div, true,
                SCE_GXM_DEPTH_FUNC_LESS_EQUAL, d.exposure, &d.material, &d.world, None, 0, false,
                // This is the top-down HEIGHT-FIELD render, not the guest's screen: its
                // projection is the tool's own, so a screen-space scissor has no meaning here.
                None,
            );
        }
    }
    // Resolve: average the image, take the MAX height per output cell.
    let top_y: Vec<f32> = if s == 1 {
        depth.iter().map(|z| if z.is_finite() { -z } else { f32::NAN }).collect()
    } else {
        let (ow, oh) = (rw / s, rh / s);
        let mut out = Vec::with_capacity((ow * oh) as usize);
        for oy in 0..oh {
            for ox in 0..ow {
                let mut peak = f32::NEG_INFINITY;
                for sy in 0..s {
                    for sx in 0..s {
                        let z = depth[(((oy * s + sy) * rw) + ox * s + sx) as usize];
                        if z.is_finite() && -z > peak {
                            peak = -z;
                        }
                    }
                }
                out.push(if peak == f32::NEG_INFINITY { f32::NAN } else { peak });
            }
        }
        out
    };
    WorldMap { view: MapView { extent: view.extent, width: rw / s, height: rh / s }, fb: fb.downsampled(s), top_y }
}

/// Rasterize one scene into a fresh framebuffer at native resolution. `clear` is the
/// background color. This is the 1x path (the oracle the GPU parity probe compares against);
/// [`render_scene_supersampled`] wraps it with antialiasing.
pub fn render_scene(scene: &Scene, width: u32, height: u32, clear: [u8; 4]) -> Framebuffer {
    render_scene_raster(scene, width, height, clear, 1)
}

/// Rasterize a scene with `ssaa`x supersampling: render at `ssaa * (width, height)` and
/// box-downsample to `width x height`. This antialiases the geometric aliasing a
/// heavily-tessellated distant mesh (dozens of sub-pixel triangles per final pixel) and
/// coincident-panel z-fighting produce as speckle at 1x, and mirrors the GPU renderer's
/// `set_supersample` so the software oracle and the GPU stay in lockstep at any factor.
/// `ssaa == 1` is identical to [`render_scene`].
pub fn render_scene_supersampled(scene: &Scene, width: u32, height: u32, clear: [u8; 4], ssaa: u32) -> Framebuffer {
    let s = ssaa.max(1);
    if s == 1 {
        return render_scene(scene, width, height, clear);
    }
    render_scene_raster(scene, width * s, height * s, clear, s).downsampled(s)
}

/// Rasterize a whole FRAME - every scene the guest submitted between two display flips,
/// in submission order - into one image. This is the software twin of
/// `GxmRenderer::encode_chain`, and without it a 3D title's shot is a lie.
///
/// A frame is not a scene. A 3D title renders its world (plus shadow maps, reflections
/// and a post-process chain) into OFFSCREEN colour surfaces and then composites those
/// onto the display buffer. Rendering only the last scene - which is all
/// [`render_scene`] can do - draws only that composite, and every texture the composite
/// samples comes from guest memory the guest never wrote, because on hardware the GPU
/// wrote it. One of the retail racers is exactly this shape: fourteen offscreen passes
/// carrying the entire world, then a 24-draw composite. Rendering the last scene alone
/// gave a correct, live HUD over pure black, and that is worse than no picture at all - it
/// looks like a title that renders nothing rather than one whose passes were dropped.
///
/// The two rules that make the chain come out right, both learned on the GPU side:
///   * a target is CLEARED the first time this frame draws into it and COMPOSED onto
///     after that. A later pass into a buffer an earlier pass filled is a post-process
///     step and is entitled to leave most of the image alone; clearing for it wipes the
///     world the pass before it drew.
///   * a draw that samples a texture whose guest address is a target this frame has
///     already rendered samples THAT RENDER, not the guest bytes - which are stale by
///     construction, since on hardware nothing but the GPU ever writes them.
pub fn render_frame_chain(
    scenes: &[Scene],
    width: u32,
    height: u32,
    clear: [u8; 4],
    ssaa: u32,
) -> Framebuffer {
    let s = ssaa.max(1);
    let (rw, rh) = (width * s, height * s);
    let Some(last) = scenes.last() else {
        return Framebuffer::new(width, height, clear);
    };
    if scenes.len() == 1 {
        return render_scene_supersampled(last, width, height, clear, s);
    }
    // The display buffer is whatever the FINAL scene draws to; any earlier scene naming
    // the same address is part of the same image rather than an offscreen pass.
    let display = last.color.as_ref().map(|c| c.data_addr);

    let mut fb = Framebuffer::new(rw, rh, clear);
    let mut depth = vec![f32::INFINITY; (rw * rh) as usize];
    // Each offscreen target's rendered image, at its NATIVE size (that is the size a
    // later pass samples it at), keyed by the colour surface's guest address.
    let mut rendered: HashMap<u32, Framebuffer> = HashMap::new();
    // Live offscreen framebuffers plus their depth buffers, so a second pass into the
    // same target composes onto the first instead of starting from nothing.
    let mut targets: HashMap<u32, (Framebuffer, Vec<f32>)> = HashMap::new();

    // Diagnostic (VITASLOP_SW_CHAIN=1): per pass, what it drew into and how many pixels it
    // actually wrote, plus - the load-bearing part - which draws sampled a target this
    // frame rendered. A composite that samples NONE of them produces a picture that is
    // indistinguishable from the passes never having run, and those are different bugs.
    let debug = std::env::var("VITASLOP_SW_CHAIN").is_ok();

    let n = scenes.len();
    for (i, scene) in scenes.iter().enumerate() {
        let addr = scene.color.as_ref().map(|c| c.data_addr);
        if debug {
            let sampled: Vec<String> = scene
                .draws
                .iter()
                .filter_map(|d| d.albedo())
                .map(|t| {
                    let hit = if rendered.contains_key(&t.data_addr) { "*" } else { "" };
                    format!("{:#x}{hit}", t.data_addr)
                })
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            // Exposure and material tint are the two fixed-function inputs that can turn a
            // pass that draws every pixel into a pass that draws every pixel BLACK, so they
            // are reported next to the pass rather than left to be guessed at afterwards.
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for d in &scene.draws {
                lo = lo.min(d.exposure);
                hi = hi.max(d.exposure);
            }
            eprintln!(
                "SWCHAIN pass{i} target={} draws={} exposure={lo:.3}..{hi:.3} samples=[{}]",
                addr.map(|a| format!("{a:#x}")).unwrap_or_else(|| "none".into()),
                scene.draws.len(),
                sampled.join(" ")
            );
        }
        if i + 1 == n || (addr.is_some() && addr == display) {
            render_scene_onto(&mut fb, &mut depth, scene, s as f32, clear, &rendered);
            if debug {
                let px = &fb.rgba;
                let cnt = (px.len() / 4).max(1) as u64;
                let mut acc = [0u64; 4];
                for p in px.chunks_exact(4) {
                    for (a, v) in acc.iter_mut().zip(p) {
                        *a += *v as u64;
                    }
                }
                eprintln!(
                    "SWCHAIN   -> DISPLAY after pass{i} mean=({},{},{},{})",
                    acc[0] / cnt, acc[1] / cnt, acc[2] / cnt, acc[3] / cnt
                );
                if let Some(dir) = std::env::var_os("VITASLOP_SW_CHAIN_DIR") {
                    let dir = std::path::PathBuf::from(dir);
                    let _ = std::fs::create_dir_all(&dir);
                    let _ = std::fs::write(
                        dir.join(format!("pass{i:02}_display.png")),
                        fb.downsampled(s).to_png(),
                    );
                }
            }
            continue;
        }
        let Some(c) = scene.color.as_ref() else {
            // A scene with no resolvable target cannot be placed at all. Say so rather
            // than dropping it: an unplaced pass and a pass that drew nothing look
            // identical in the finished frame and are completely different bugs.
            report_once(
                &UNPLACED_SCENE,
                "software render: a scene of this frame has NO colour surface, so its \
                 draws cannot be placed in the chain and are not rendered. The finished \
                 frame is missing whatever that pass carried.",
            );
            continue;
        };
        if c.width == 0 || c.height == 0 {
            // Nothing can be rasterized into a zero-sized target, but dropping the
            // pass in silence is how a whole pass goes missing without a trace: a
            // frame whose world pass was dropped looks exactly like a title that
            // drew no world. If this fires, the extent is almost certainly wrong
            // rather than the title's - `sceGxmBeginScene` takes it from the RENDER
            // TARGET, and only falls back to the colour surface for a target this
            // implementation never saw created.
            report_once(
                &ZERO_SIZED_TARGET,
                "software render: DROPPING a pass whose colour target has a zero extent. \
                 Whatever that pass carried is missing from the finished frame.",
            );
            continue;
        }
        // A SECOND pass into a target this frame already filled is a post-process step -
        // depth of field, bloom, a radial blur - and post-process is pure SHADER work. The
        // fixed-function approximation has no shader to run, so it paints the pass's mask
        // and blur textures as if they were surface albedo: this title's DOF pass covers
        // the whole frame in the flat white of its lens mask and leaves a porthole of world
        // in the middle. Dropping the pass keeps the sharp world the pass before it drew,
        // which is far closer to what the title puts on screen than the mask is.
        //
        // Reported unconditionally, because an approximation that silently discards a pass
        // is indistinguishable from one that reproduces it. `VITASLOP_SW_POST=keep` renders
        // them anyway, which is how you look at what a post pass is actually doing.
        if targets.contains_key(&c.data_addr)
            && std::env::var("VITASLOP_SW_POST").as_deref() != Ok("keep")
        {
            report_once(
                &SKIPPED_POST,
                "software render: SKIPPING post-process passes (a second pass into a target \
                 already rendered this frame). They are shader work - blur, depth of field, \
                 bloom - and the fixed-function approximation would paint their mask and \
                 blur textures as flat surface colour over the finished image. The GXP \
                 recompiler renders them properly; set VITASLOP_SW_POST=keep to see them.",
            );
            if debug {
                eprintln!("SWCHAIN   -> SKIPPED post pass{i} into {:#x}", c.data_addr);
            }
            continue;
        }
        let (tw, th) = (c.width * s, c.height * s);
        let entry = targets.entry(c.data_addr).or_insert_with(|| {
            // An intermediate image clears to TRANSPARENT black, not to the display's
            // clear colour: a composite that blends it must see nothing where the pass
            // drew nothing.
            (Framebuffer::new(tw, th, [0, 0, 0, 0]), vec![f32::INFINITY; (tw * th) as usize])
        });
        render_scene_onto(&mut entry.0, &mut entry.1, scene, s as f32, [0, 0, 0, 0], &rendered);
        if debug {
            // The MEAN colour, not just a drawn-pixel count. "Drawn" only means "not the
            // transparent clear", which opaque black satisfies - so a pass that rendered
            // every one of its pixels black reports as fully drawn and reads like a
            // success. The mean is what separates "this pass did not run" from "this pass
            // ran and produced black", and those have nothing in common as bugs.
            let px = &entry.0.rgba;
            let n = (px.len() / 4).max(1) as u64;
            let mut acc = [0u64; 4];
            for p in px.chunks_exact(4) {
                for (a, v) in acc.iter_mut().zip(p) {
                    *a += *v as u64;
                }
            }
            eprintln!(
                "SWCHAIN   -> rendered {:#x} {}x{} drawn={} of {} px mean=({},{},{},{})",
                c.data_addr,
                entry.0.width,
                entry.0.height,
                entry.0.drawn_pixels([0, 0, 0, 0]),
                entry.0.width * entry.0.height,
                acc[0] / n, acc[1] / n, acc[2] / n, acc[3] / n
            );
            // VITASLOP_SW_CHAIN_DIR=<dir> writes each pass's own image. This is the software
            // twin of VITASLOP_CHAIN_LIMIT: when a composite comes out wrong the question is
            // always WHICH pass is wrong, and the finished frame cannot answer it because
            // every failure mode looks the same once composited.
            if let Some(dir) = std::env::var_os("VITASLOP_SW_CHAIN_DIR") {
                let dir = std::path::PathBuf::from(dir);
                let _ = std::fs::create_dir_all(&dir);
                let _ = std::fs::write(
                    dir.join(format!("pass{i:02}_{:08x}.png", c.data_addr)),
                    entry.0.downsampled(s).to_png(),
                );
            }
        }
        rendered.insert(c.data_addr, entry.0.downsampled(s));
    }
    fb.downsampled(s)
}

/// Build the substitute texture a draw sampling a render target reads: the image this
/// frame rendered into that target, as a plain linear RGBA8 texture. Everything about
/// HOW the draw samples (unit, filters, wrap modes) is kept from the guest's binding -
/// only the pixels and their layout change.
fn rtt_substitute(image: &Framebuffer, proto: &BoundTexture) -> BoundTexture {
    BoundTexture {
        unit: proto.unit,
        // A buffer built HERE, from this frame's rendered image, so it is a new one every
        // time it is built and gets an identity of its own. Inheriting the prototype's would
        // hand two different rendered images one cache entry.
        pixels_id: crate::capture::next_pixels_id(),
        // 0x0c is the 32-bit four-channel family and swizzle 0 (ABGR) is the identity
        // permutation over memory bytes, so `b0,b1,b2,b3` decode as `R,G,B,A` - exactly
        // the order a `Framebuffer` stores. `tex_type` 3 is LINEAR: a rendered image is
        // row-major, not Morton-swizzled.
        base_format: 0x0c,
        swizzle: 0,
        tex_type: 3,
        width: image.width,
        height: image.height,
        stride: image.width * 4,
        faces: 1,
        face_bytes: image.width * image.height * 4,
        // A rendered image is exactly one level: nothing generated a chain for it.
        levels: 1,
        data_addr: proto.data_addr,
        pixels: Arc::from(image.rgba.as_slice()),
        u_addr_mode: proto.u_addr_mode,
        v_addr_mode: proto.v_addr_mode,
        lod_bias: proto.lod_bias,
        min_filter: proto.min_filter,
        mag_filter: proto.mag_filter,
        mip_filter: proto.mip_filter,
        // The substitute holds the pixels this frame RENDERED, which are already linear in
        // whatever space the pass wrote them - there is no gamma-encoded memory here for a
        // sampler to decode, so the mode does not carry over from the prototype binding.
        gamma: 0,
    }
}

/// The rasterizer core. `width`/`height` are the RASTER dimensions and `ssaa` the supersample
/// factor those already fold in (so Pixel-space draws scale by it - see [`project`]); the
/// caller downsamples the result. `clear` is the background color.
fn render_scene_raster(scene: &Scene, width: u32, height: u32, clear: [u8; 4], ssaa: u32) -> Framebuffer {
    let mut fb = Framebuffer::new(width, height, clear);
    let mut depth = vec![f32::INFINITY; (width * height) as usize];
    render_scene_onto(&mut fb, &mut depth, scene, ssaa.max(1) as f32, clear, &HashMap::new());
    fb
}

/// Draw one scene onto an EXISTING framebuffer and depth buffer, composing with whatever
/// is already there. Split out of [`render_scene_raster`] so that
/// [`render_frame_chain`] can render a frame's passes into their own targets and let a
/// later pass compose onto an earlier one's image.
///
/// `rtt` maps a colour surface's guest address to the image this frame already rendered
/// into it: a draw sampling one of those addresses samples that image instead of the
/// guest bytes, which is the whole reason a composite can show a world at all.
/// `clear` is only the colour the per-draw statistics count "drawn" pixels against.
fn render_scene_onto(
    fb: &mut Framebuffer,
    depth: &mut [f32],
    scene: &Scene,
    ssaa: f32,
    clear: [u8; 4],
    rtt: &HashMap<u32, Framebuffer>,
) {
    let (width, height) = (fb.width, fb.height);

    // Diagnostic: VITASLOP_PIXEL_TRACE=x,y logs every draw that writes that pixel (index,
    // whether textured/depth-tested, the source RGBA) - the definitive "which draw painted
    // this glitch" probe. Zero cost when unset.
    let trace: Option<(i32, i32)> = std::env::var("VITASLOP_PIXEL_TRACE").ok().and_then(|s| {
        let (a, b) = s.split_once(',')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    });

    // Diagnostic: VITASLOP_DRAW_STATS prints, per draw, whether it was skipped/culled and
    // how many framebuffer pixels it actually wrote - so a big environment mesh that renders
    // to nothing (off-screen projection, wrongly skipped, or fully culled) is visible instead
    // of a silently-missing background. Off by default (a per-draw drawn-pixel diff).
    let stats = std::env::var("VITASLOP_DRAW_STATS").is_ok();

    // Diagnostic: VITASLOP_UV_DEBUG paints each depth-tested draw by its interpolated texcoord
    // (R=u.fract, G=v.fract) - a coherent per-panel UV reads as smooth gradients, a scrambled one
    // as noise. Read once here (not per triangle) so the hot path stays clean.
    let uv_debug = std::env::var("VITASLOP_UV_DEBUG").is_ok();

    // Diagnostic: VITASLOP_DRAW_ONLY=<i>,<i>.. rasterizes only those draw indices (the `di`
    // DSTAT prints), leaving the rest of the frame at the clear colour. Attributing a hole
    // or an artifact to one draw is otherwise guesswork on a 90-draw scene.
    let only: Option<Vec<usize>> = std::env::var("VITASLOP_DRAW_ONLY").ok().map(|s| {
        s.split(',').filter_map(|p| p.trim().parse().ok()).collect()
    });

    // Diagnostic: VITASLOP_DUMP_TRIS=<draw>:<n> prints the first `n` triangles of that draw
    // as (index triple, model-space positions, screen positions). This is what distinguishes
    // "the mesh really is a set of ribbons" from "we are decoding its vertices wrongly".
    let dump_tris: Option<(usize, usize)> = std::env::var("VITASLOP_DUMP_TRIS").ok().and_then(|s| {
        let (a, b) = s.split_once(':')?;
        Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
    });

    for (di, d) in scene.draws.iter().enumerate() {
        if only.as_ref().is_some_and(|list| !list.contains(&di)) {
            continue;
        }
        // A list emits idx/3 triangles; a strip or fan emits idx-2 (each new index adds
        // one triangle). Any other topology (lines, points) emits none and is skipped.
        let tri_count = triangle_count(d);
        if tri_count == 0 {
            if stats {
                eprintln!("DSTAT draw {di} SKIP=nontriangle prim={:#x} idx={}", d.primitive, d.index_count);
            }
            continue;
        }
        let DrawInterp { layout, space, textured, uv_div, skip } = interpret_draw(d);
        if skip {
            // An approximation that drops geometry must say so unconditionally - a silently
            // skipped draw is indistinguishable on screen from one the game never made. This
            // is a static property of the title's shaders rather than a per-frame event, so
            // it is announced once per kind instead of once per draw.
            if d.shader_expanded {
                report_once(
                    &SKIPPED_EXPANDED,
                    "software render: SKIPPING shader-expanded (point-sprite/billboard) draws - \
                     their vertex stream holds sprite records, not triangles, so the \
                     fixed-function approximation cannot build the primitive. The GXP \
                     recompiler renders these correctly; `play` shots will not show them.",
                );
            }
            if stats {
                let why = if d.shader_expanded { "shader-expanded" } else { "no-color-no-uv" };
                eprintln!("DSTAT draw {di} SKIP={why} tris={tri_count} stride={}", d.vertex_stride);
            }
            continue;
        }
        let pixels_before = if stats { fb.drawn_pixels(clear) } else { 0 };
        // Opaque, z-buffered replace only for genuinely opaque 3D geometry: an MVP draw
        // that also WRITES depth. A 2D UI overlay positions itself with an MVP too but
        // disables depth writes (SCE_GXM_DEPTH_WRITE_DISABLED) and is alpha-blended in
        // submission order - so keying "opaque" off the mere presence of an MVP splats
        // such a sprite's transparent texels as solid colour over everything behind it.
        let depth_test = matches!(space, Space::Mvp(_))
            && d.render_state.front_depth_write != SCE_GXM_DEPTH_WRITE_DISABLED;
        let depth_func = d.render_state.front_depth_func;
        // The guest's REGION CLIP for this draw, in RASTER pixels (so it scales with `ssaa`
        // exactly as a Pixel-space vertex does). GXM states the rectangle INCLUSIVE at both
        // ends. Both enabled modes keep the INSIDE of it - see
        // `vitaslop_platform::gpu::RegionClip`, where two titles' rectangles settle which
        // reading of the mode enum is right; `ALL` clips everything and is expressed here as
        // an empty rectangle.
        let scissor: Option<[i32; 4]> = match d.render_state.region_clip_mode & 0xC000_0000 {
            0x0000_0000 => None,
            0x4000_0000 => Some([0, 0, -1, -1]),
            _ => {
                let r = d.render_state.region_clip;
                let sc = |v: u32| (v as f32 * ssaa) as i32;
                Some([sc(r[0]), sc(r[1]), sc(r[2].saturating_add(1)) - 1, sc(r[3].saturating_add(1)) - 1])
            }
        };
        // Back-face culling as the GPU does it, per the draw's SceGxmCullMode. Real 3D
        // titles enable it on nearly every world/vehicle mesh; without it the hidden
        // interior faces of a thin shell z-fight the outer faces into speckle. Only
        // 3D (MVP) draws are culled - a 2D sprite has no meaningful facing and its
        // winding is submission-defined. NONE (the double-sided body panels) skips it.
        let cull_mode = if matches!(space, Space::Mvp(_)) {
            d.render_state.cull_mode
        } else {
            SCE_GXM_CULL_NONE
        };
        // A draw sampling a colour surface this frame already rendered reads THAT image.
        // The guest bytes at that address are stale by construction - on hardware nothing
        // but the GPU ever writes a render target - so sampling them is what turns a
        // composite of a finished world into a composite of whatever was last left there.
        let substitute;
        let texture = match (textured, d.albedo()) {
            (true, Some(t)) => match rtt.get(&t.data_addr) {
                Some(image) => {
                    substitute = rtt_substitute(image, t);
                    Some(&substitute)
                }
                None => Some(t),
            },
            _ => None,
        };
        let (mut n_behind, mut n_off, mut n_on, mut n_culled) = (0u32, 0u32, 0u32, 0u32);
        let (mut bb_lo, mut bb_hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);

        for t in 0..tri_count {
            // Winding-normalized triangle (a strip's odd triangles are un-flipped) so the
            // cull test below is uniform across list/strip/fan.
            let vs = tri_indices(d, t);
            let verts: Vec<Vertex> = vs.iter().map(|&i| decode_vertex(d, &layout, i)).collect();

            // Project to screen; drop the triangle if any vertex is behind the eye.
            let mut screen = [[0f32; 4]; 3]; // x, y, depth, 1/w
            let mut behind = false;
            for (k, v) in verts.iter().enumerate() {
                match project(v, &space, width, height, ssaa) {
                    Some(s) => screen[k] = s,
                    None => {
                        behind = true;
                        break;
                    }
                }
            }
            if behind {
                n_behind += 1;
                continue;
            }
            if dump_tris.is_some_and(|(want, n)| want == di && t < n) {
                let p = |v: &Vertex| format!("({:.2},{:.2},{:.2})", v.pos[0], v.pos[1], v.pos[2]);
                eprintln!(
                    "TRI draw {di} #{t} idx={vs:?} model={} {} {} screen=({:.0},{:.0}) ({:.0},{:.0}) ({:.0},{:.0})",
                    p(&verts[0]), p(&verts[1]), p(&verts[2]),
                    screen[0][0], screen[0][1], screen[1][0], screen[1][1], screen[2][0], screen[2][1]
                );
            }
            if stats {
                let on = screen.iter().any(|s| s[0] >= 0.0 && s[0] < width as f32 && s[1] >= 0.0 && s[1] < height as f32);
                if on { n_on += 1 } else { n_off += 1 }
                for s in &screen {
                    bb_lo[0] = bb_lo[0].min(s[0]); bb_lo[1] = bb_lo[1].min(s[1]);
                    bb_hi[0] = bb_hi[0].max(s[0]); bb_hi[1] = bb_hi[1].max(s[1]);
                }
            }
            // Cull back faces before rasterizing (screen-space winding).
            if cull_mode != SCE_GXM_CULL_NONE && cull_backface(edge(&screen[0], &screen[1], &screen[2]), cull_mode) {
                n_culled += 1;
                continue;
            }
            raster_triangle(fb, depth, &screen, &verts, texture, uv_div, depth_test, depth_func, d.exposure, &d.material, &d.world, trace, di, uv_debug, scissor);
        }
        if stats {
            let wrote = fb.drawn_pixels(clear).saturating_sub(pixels_before);
            let sp = match space { Space::Mvp(_) => "mvp", Space::Ndc => "ndc", Space::Pixel => "pixel" };
            eprintln!(
                "DSTAT draw {di} space={sp} dtest={depth_test} cull={:#x} tris={tri_count} on={n_on} off={n_off} behind={n_behind} culled={n_culled} wrote+{wrote} bbox=[{:.0},{:.0}..{:.0},{:.0}] tex={}",
                cull_mode, bb_lo[0], bb_lo[1], bb_hi[0], bb_hi[1],
                d.textures.first().map(|t| format!("{}x{}", t.width, t.height)).unwrap_or_default()
            );
        }
    }
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
    depth_func: u32,
    exposure: f32,
    material: &crate::capture::FragmentMaterial,
    world: &[f32; 16],
    trace: Option<(i32, i32)>,
    draw_idx: usize,
    uv_debug: bool,
    // The draw's region clip as an INCLUSIVE `[x0, y0, x1, y1]` in raster pixels, or `None`
    // for the whole framebuffer. An empty rectangle (`x1 < x0`) draws nothing, which is what
    // `SCE_GXM_REGION_CLIP_ALL` asks for.
    scissor: Option<[i32; 4]>,
) {
    let (w, h) = (fb.width as i32, fb.height as i32);
    // World-space normals at the three vertices (constant per triangle), interpolated per
    // pixel below for the lit opaque path. Object-space normals are brought to world space
    // by the draw's model-to-world matrix so N.L matches the world-space light direction.
    let wn: [[f32; 3]; 3] =
        [world_normal(verts[0].normal, world), world_normal(verts[1].normal, world), world_normal(verts[2].normal, world)];
    let has_normal = verts.iter().any(|v| v.normal != [0.0, 0.0, 0.0]);
    let mut min_x = s.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min).floor().max(0.0) as i32;
    let mut max_x = s.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max).ceil().min((w - 1) as f32) as i32;
    let mut min_y = s.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min).floor().max(0.0) as i32;
    let mut max_y = s.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max).ceil().min((h - 1) as f32) as i32;
    // The guest's REGION CLIP, which is GXM's hardware scissor. The GPU path issues it as
    // `set_scissor_rect`; here the only thing that restricts a triangle is its bounding box,
    // so the clip narrows that. Same restriction, expressed in the terms this rasterizer has.
    if let Some([sx0, sy0, sx1, sy1]) = scissor {
        min_x = min_x.max(sx0);
        max_x = max_x.min(sx1);
        min_y = min_y.max(sy0);
        max_y = max_y.min(sy1);
    }
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

            // Perspective-correct interpolation: weight each attribute by 1/w and
            // renormalize by the interpolated 1/w.
            let iw = b0 * s[0][3] + b1 * s[1][3] + b2 * s[2][3];
            let interp = |a: f32, b: f32, c: f32| -> f32 {
                (b0 * s[0][3] * a + b1 * s[1][3] * b + b2 * s[2][3] * c) / iw
            };
            // Depth COMPARE first (an early reject that avoids sampling occluded pixels), then
            // the texel, then - for an opaque decal layer - the alpha-test, then the depth WRITE.
            // Splitting compare from write lets a transparent decal texel be discarded WITHOUT
            // writing depth, while occluded pixels still skip the texture sample (perf).
            let z = b0 * s[0][2] + b1 * s[1][2] + b2 * s[2][2];
            if depth_test && !depth_passes(z, depth[idx], depth_func) {
                continue;
            }
            // Sample the albedo/detail texel. `uv_div` normalizes an atlas-in-pixels 2D coord.
            let texel = texture.map(|tex| {
                let u = interp(verts[0].uv[0], verts[1].uv[0], verts[2].uv[0]) / uv_div[0];
                let v = interp(verts[0].uv[1], verts[1].uv[1], verts[2].uv[1]) / uv_div[1];
                sample_texture(tex, u, v)
            });
            // Alpha-test for opaque LIVERY / DECAL layers: this title draws a car's numbers and
            // logos as a separate opaque, depth-writing mesh whose picked albedo is a BC2/BC3
            // sheet carrying a COVERAGE alpha (transparent between the marks). Rendered as a flat
            // opaque replace this paints the sheet's black background as speckle over the body;
            // discarding the transparent texels (and NOT writing depth) lets the body panel drawn
            // behind it show through, the faithful decal result. Safe for the ordinary opaque
            // BC1 albedo (its alpha is always 255, so nothing is discarded).
            const ALPHA_TEST: u8 = 128;
            if depth_test {
                if let Some(t) = texel {
                    if t[3] < ALPHA_TEST {
                        continue;
                    }
                }
                depth[idx] = z;
            }

            let mut src = [0f32; 4];
            for ch in 0..4 {
                src[ch] = interp(
                    verts[0].color[ch] as f32,
                    verts[1].color[ch] as f32,
                    verts[2].color[ch] as f32,
                );
            }
            // Combine the sampled texel with the interpolated vertex color as the GXM
            // fixed-function "modulate" default does: per-channel product after the
            // texture's own swizzle is applied. This is correct for every texel role
            // because the swizzle already encodes intent:
            //   - a color texture (RGBA) modulates the vertex tint straight;
            //   - a font/coverage atlas (single channel swizzled 111R/000R) leaves rgb
            //     at 1/0 and carries coverage in alpha, so rgb passes the vertex color
            //     through and alpha becomes vertex-alpha * coverage - no double-darkening;
            //   - a single-channel LUMINANCE map (swizzled RRRR, e.g. this title's ground
            //     detail texture) replicates into rgba, tinting the vertex color by
            //     luminance instead of being mistaken for a pure alpha mask.
            if let Some(texel) = texel {
                if depth_test {
                    // 3D opaque geometry (world / vehicles): the base colour is the albedo
                    // texture. This title's meshes carry NON-colour data in the vertex
                    // colour attribute (e.g. [255,0,255] mask/lighting encodings the real
                    // fragment program consumes), so modulating the albedo by it would
                    // tint whole surfaces magenta. Take the albedo texel straight; the
                    // opaque path forces alpha to 255 below.
                    src[..3].copy_from_slice(&[texel[0] as f32, texel[1] as f32, texel[2] as f32]);
                } else {
                    // 2D / overlay: vertex colour is the real tint and the texture supplies
                    // coverage/detail, so modulate per channel (swizzle already applied).
                    for ch in 0..4 {
                        src[ch] = src[ch] * texel[ch] as f32 / 255.0;
                    }
                }
            }

            if let Some((tx, ty)) = trace {
                if x == tx && y == ty {
                    let (tu, tv, texel) = match texture {
                        Some(tex) => {
                            let u = interp(verts[0].uv[0], verts[1].uv[0], verts[2].uv[0]) / uv_div[0];
                            let v = interp(verts[0].uv[1], verts[1].uv[1], verts[2].uv[1]) / uv_div[1];
                            (u, v, sample_texture(tex, u, v))
                        }
                        None => (0.0, 0.0, [0, 0, 0, 0]),
                    };
                    eprintln!(
                        "PXTRACE ({tx},{ty}) draw {draw_idx} textured={} depth_test={} src=[{:.0},{:.0},{:.0},{:.0}] vcol0={:?} uv=({tu:.3},{tv:.3}) texel={:?}",
                        texture.is_some(), depth_test, src[0], src[1], src[2], src[3],
                        verts[0].color, texel
                    );
                }
            }
            let dst = idx * 4;
            if depth_test {
                // Opaque 3D replace (z-buffer already updated): run the reflected forward-lit
                // material. `src[..3]` is the albedo (the sampled texel, or the vertex colour
                // for an untextured opaque draw); shade it by the per-material tint + the
                // directional light (interpolated world-space normal) + ambient, then apply
                // scene exposure and tone-map. This is what turns a near-white tyre albedo
                // (tint ~0.01) into dark rubber instead of a white ring, and gives the body
                // panels form instead of a flat over-bright fill.
                let n = if has_normal {
                    let nx = interp(wn[0][0], wn[1][0], wn[2][0]);
                    let ny = interp(wn[0][1], wn[1][1], wn[2][1]);
                    let nz = interp(wn[0][2], wn[1][2], wn[2][2]);
                    let len = (nx * nx + ny * ny + nz * nz).sqrt();
                    if len > 1e-6 { [nx / len, ny / len, nz / len] } else { [0.0, 1.0, 0.0] }
                } else {
                    [0.0, 1.0, 0.0]
                };
                let lit = shade_lit([src[0], src[1], src[2]], n, material, exposure);
                // DEBUG (VITASLOP_UV_DEBUG): paint the interpolated texcoord as R=u.fract,
                // G=v.fract so a coherent per-panel UV mapping reads as smooth red/green
                // gradients, while a scrambled/mis-decoded UV reads as noise - the decisive
                // test for whether a speckled textured draw is a UV bug or genuine atlas content.
                if uv_debug && texture.is_some() {
                    let u = interp(verts[0].uv[0], verts[1].uv[0], verts[2].uv[0]) / uv_div[0];
                    let v = interp(verts[0].uv[1], verts[1].uv[1], verts[2].uv[1]) / uv_div[1];
                    fb.rgba[dst] = ((u - u.floor()) * 255.0) as u8;
                    fb.rgba[dst + 1] = ((v - v.floor()) * 255.0) as u8;
                    fb.rgba[dst + 2] = 40;
                    fb.rgba[dst + 3] = 255;
                } else {
                    fb.rgba[dst..dst + 3].copy_from_slice(&lit);
                    fb.rgba[dst + 3] = 255;
                }
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

/// Whether a fragment at depth `z` passes the z-test against the stored depth
/// `stored` under a `SceGxmDepthFunc`. The z-buffer stores nearer = smaller (cleared
/// to +inf), so `LESS`/`LESS_EQUAL` are the ordinary "closer wins" tests. Real titles
/// use `LESS_EQUAL` (its default), which lets a coincident later face tie and repaint -
/// the correct behaviour for the double-sided body panels a strict `LESS` would leave to
/// whichever face happened to draw first. The full enum is honoured so an unusual pass
/// (an ALWAYS/GREATER overlay) reproduces rather than being forced to LESS.
fn depth_passes(z: f32, stored: f32, func: u32) -> bool {
    const NEVER: u32 = 0x0000_0000;
    const LESS: u32 = 0x0040_0000;
    const EQUAL: u32 = 0x0080_0000;
    const LESS_EQUAL: u32 = 0x00C0_0000;
    const GREATER: u32 = 0x0100_0000;
    const NOT_EQUAL: u32 = 0x0140_0000;
    const GREATER_EQUAL: u32 = 0x0180_0000;
    const ALWAYS: u32 = 0x01C0_0000;
    match func {
        NEVER => false,
        LESS => z < stored,
        EQUAL => z == stored,
        GREATER => z > stored,
        NOT_EQUAL => z != stored,
        GREATER_EQUAL => z >= stored,
        ALWAYS => true,
        // LESS_EQUAL is the GXM default; treat any unrecognized encoding as it.
        LESS_EQUAL | _ => z <= stored,
    }
}

use std::collections::HashMap;
use std::sync::Arc;
use vitaslop_platform::gpu::{
    BlockFamily, BlockFormat, CompressedUpload, DrawSpace, GxmDraw, GxmTexture, RenderScene,
    TexelSeam, GXM_VERTEX_STRIDE,
};

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

/// An EXACT identity fingerprint of a captured texture, used to cache its decode (here)
/// and its GPU upload (in the renderer). It folds the control words (address, format,
/// swizzle, type, geometry), the sampler state, and the IDENTITY of the pixel buffer.
/// FNV-1a/64.
///
/// # Why the buffer's identity and not its contents
/// This used to fold a 256-byte STRIDED SAMPLE of the pixels, on the reasoning that it
/// would notice a same-address atlas whose contents changed without hashing every byte.
/// A sample is not a proof: a content change the stride steps over reused a stale DECODE,
/// however exactly the capture had compared the bytes. So the whole per-scene `memcmp`
/// upstream ([`TextureSnapshots`], 40% of a race frame) was buying an exactness this key
/// then threw away - end-to-end detection was sampled either way.
///
/// The capture already answers the question exactly, and its answer is the buffer: it hands
/// back the SAME `Arc` when the bytes are unchanged and a NEW one when they are not
/// (including on the re-read path, which compares before allocating). So the buffer's
/// address IS "these are the same pixels" - no sampling, and no hashing of a 4 MB atlas
/// either. `RenderSceneBuilder::decode_cache` holds a strong reference to the buffer it
/// keyed on, which is what makes the address valid as a key: without it a freed buffer's
/// address could be reused by a different texture. `TextureSnapshots::means` is keyed the
/// same way for the same reason.
///
/// Two distinct buffers holding identical bytes get two entries. That is wasteful, never
/// wrong, and the capture avoids it wherever it can tell.
fn tex_key(t: &BoundTexture) -> u64 {
    // >>> NINE ROUNDS, NOT SEVENTY-TWO. This ran FNV one BYTE at a time over nine 64-bit
    // values, and it is called once per bound texture per draw - **4,632 times per presented
    // frame** on a race, which is a third of a million xor-multiply rounds to look up a cache.
    //
    // The mixer is the crate's own ([[crate::fasthash]]): rotate, xor, multiply, a round per
    // WORD. The rotate is what makes that admissible where a word-at-a-time FNV would not be -
    // plain `h ^= word; h *= odd` is linear mod 2^64 and cannot diffuse bit 63, which is the
    // flaw that made a geometry cache render another draw's mesh
    // ([[vitaslop-content-hash-cache-must-verify]]). A finaliser avalanche follows, because
    // this value is used as a cache KEY with no verification behind it, and the inputs here
    // (an address, a format, two dimensions) differ in low bits far more often than high ones.
    let mut st = crate::fasthash::FxHasher::default();
    let mut h: u64 = 0;
    let mut mix = |v: u64| {
        use std::hash::Hasher;
        st.write_u64(v);
        h = st.finish();
    };
    mix(t.data_addr as u64);
    mix(t.base_format as u64);
    mix(t.swizzle as u64);
    mix(t.tex_type as u64);
    mix(((t.width as u64) << 32) | t.height as u64);
    mix(((t.stride as u64) << 32) | t.pixels.len() as u64);
    // The SAMPLER state belongs in the key too. It does not change the decoded pixels, but the
    // cached `GxmTexture` carries it to the renderer, so leaving it out hands the second binding
    // of the same image the FIRST binding's filter and wrap modes.
    mix((t.mag_filter as u64) << 32 | t.min_filter as u64);
    mix((t.u_addr_mode as u64) << 32 | t.v_addr_mode as u64);
    // GAMMA belongs here for the same reason: it does not change the decoded bytes, but it
    // changes the FORMAT they are uploaded through (sRGB decodes on fetch), so two bindings of
    // one image differing only in gamma must not share a cache entry.
    mix(t.gamma as u64);
    // >>> THE PIXEL BUFFER'S IDENTITY, WHICH IS A MINTED NUMBER AND NOT ITS ADDRESS.
    //
    // This folded `pixels.as_ptr()`, and an address is only an identity while the buffer is
    // ALIVE. It is not: the snapshot layer frees a texture's buffer and allocates a new one
    // the moment the guest rewrites that texture, and an allocator that hands the freed
    // address straight back gives the NEW contents the SAME key as the old - so this cache
    // returns the previous upload for bytes that have changed, with nothing to report it.
    //
    // MEASURED: a change that moved nothing but allocation addresses took the frame's texture
    // expansions from 1.25-1.54 MB to 4.31-4.72 MB over the same draws and the same textures
    // built. Work does not appear from nowhere - what moved was how often an address was
    // recycled, which is the one input a cache key must not depend on.
    //
    // `pixels_id` is minted once per buffer and never reused in a run, so two different
    // buffers cannot collide here however the allocator behaves, and two snapshots that share
    // an `Arc` - which is exactly when the snapshot layer has PROVEN the bytes identical -
    // share an id and keep hitting. See [`crate::capture::BoundTexture::pixels_id`].
    mix(t.pixels_id);
    // The avalanche. `splitmix64`'s finaliser: two xor-shift-multiply rounds, which is what
    // turns a rotate-xor-multiply accumulator into a value whose every bit depends on every
    // input bit. Three instructions' worth of insurance on a key nothing verifies.
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

/// A texture's IDENTITY as a bound resource - the guest address, format and shape, WITHOUT
/// the pixel buffer's address.
///
/// [`tex_key`] answers "are these the same decoded bytes", which is what a decode cache
/// wants and why it folds the buffer's address. Sprite identity is a different question:
/// two frames of a title showing the same sprite are two different SNAPSHOTS of the same
/// guest texture, so they hold different buffers by construction, and keying a sprite on
/// `tex_key` means a sprite never matches itself from one frame to the next. That silently
/// disabled `scroll_drift` and `sprite_motion` - the two things a 2D title is driven by -
/// and it is the kind of break that looks like the game moving, not like a bug.
///
/// The SLOT a decode belongs to: everything [`tex_key`] folds except the pixel buffer's
/// address, which is exactly "the same guest texture, bound the same way".
///
/// Two decodes with the same slot key differ only in their contents, so the newer one has
/// replaced the older - see [`RenderSceneBuilder::decode_slots`]. The sampler state is in
/// here for the same reason it is in [`tex_key`]: two bindings of one image that differ only
/// in filter or gamma are two legitimate entries, and calling them one slot would make each
/// draw evict the other's.
fn tex_slot_key(t: &BoundTexture) -> u64 {
    let mut h = tex_binding_key(t);
    for v in [
        (t.mag_filter as u64) << 32 | t.min_filter as u64,
        (t.u_addr_mode as u64) << 32 | t.v_addr_mode as u64,
        t.gamma as u64,
    ] {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// So this is what a SPRITE folds: what the guest bound, not what we copied out of it.
fn tex_binding_key(t: &BoundTexture) -> u64 {
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
    h
}

/// Builds a neutral [`RenderScene`] from a captured GXM [`Scene`] for the GPU
/// renderer, reusing the exact per-draw interpretation ([`interpret_draw`]) and
/// texture decode ([`decode_texture_rgba8`]) the software rasterizer uses, so the
/// GPU output matches the CPU oracle. Holds a cross-frame texture-decode cache keyed
/// by [`tex_key`] so an unchanged atlas is decoded once and thereafter only its
/// shared `Arc` is handed back; persist one builder across a run's frames to keep the
/// cache warm.
/// One frame's derived `GxpTex` list for a captured binding set, and the set it came from.
/// See [`RenderSceneBuilder::gxp_tex_sets`].
struct GxpTexSet {
    /// The capture's own list, held so its ADDRESS stays a valid identity.
    src: Arc<[BoundTexture]>,
    out: Arc<[vitaslop_platform::gpu::GxpTex]>,
    /// The `decode_epoch` this was derived in. A hit is refused across frames so the stamping
    /// `texture()` does is paid once per frame, exactly as it was before this cache existed.
    epoch: u64,
}

/// One derived `GxpAttr` list, and the attribute list it came from.
/// See [`RenderSceneBuilder::gxp_attr_sets`].
struct GxpAttrSet {
    src: Arc<[crate::capture::VertexAttribute]>,
    out: Arc<[vitaslop_platform::gpu::GxpAttr]>,
}

/// How many derived sets either cache holds before dropping the lot.
///
/// A frame binds a couple of dozen distinct sets; this is generous by two orders of magnitude
/// so a title that cycles materials never thrashes, and dropping an entry costs a re-derivation
/// and never an answer.
const GXP_SET_CACHE_CAP: usize = 2048;

pub struct RenderSceneBuilder {
    /// Whether a draw carrying a shader payload can SKIP its fixed-function representation.
    ///
    /// True only when the recompiler is live AND the fixed-function fallback is refused, which
    /// is the shipped configuration: there, the renderer either draws the guest's real shaders
    /// or hard-fails, so the canonical vertex buffer, the CPU-culled index buffer, the
    /// per-vertex screen projection and the albedo decode are provably dead - they are built,
    /// paid for per vertex, and then skipped by a `continue` in `encode`. Read once here rather
    /// than per draw.
    ///
    /// It is deliberately NOT "the recompiler is live": with the fallback allowed, a pair that
    /// fails to link still needs all of it, and producing an empty draw instead would lose
    /// geometry silently.
    gxp_only: bool,
    /// Decoded textures by [`tex_key`], each holding a strong reference to the SOURCE pixel
    /// buffer it was keyed on. The reference is not decoration: the key folds that buffer's
    /// address, and a freed buffer's address can be reused by an unrelated texture.
    decode_cache: crate::fasthash::FxHashMap<u64, (GxmTexture, Arc<[u8]>)>,
    /// Per cached decode: `(the frame it was last used on, the decoded bytes it holds)`.
    ///
    /// # Why the budget cannot be enforced by clearing
    /// This cache used to be emptied WHOLESALE when it went over budget. That is safe - the
    /// keys fold the source bytes, so a re-decode is only work - but its failure shape is a
    /// cliff, not a slope: once a title's per-frame working set exceeds the budget the clear
    /// fires part-way through EVERY frame, evicting exactly what the next frame is about to
    /// ask for, and the cross-frame hit rate goes to zero rather than down.
    ///
    /// MEASURED on one title's campaign map, on the target phone (PowerVR D-series):
    /// `0.97 cache clears` per frame, 225 textures and 51 MB RE-DECODED every frame,
    /// `build 718.3 ms` of an `878.2 ms` render against `cpu 94.1 ms` - the render was 90% of
    /// the frame and the decode was 82% of the render. 1 fps. Nothing about that frame was a
    /// shader or a draw-call problem; it was this cache missing every single time.
    ///
    /// So eviction is per entry, it never takes something the frame in flight has already
    /// used, and the budget is floored at one frame's working set - see
    /// [`RenderSceneBuilder::decode_frame_high`].
    decode_used: crate::fasthash::FxHashMap<u64, (u64, usize)>,
    /// >>> WHICH DECODE IS THE CURRENT ONE FOR A GIVEN GUEST TEXTURE: [`tex_slot_key`] ->
    /// [`tex_key`].
    ///
    /// The decode key folds the SOURCE BUFFER's address, because that is what makes it exact
    /// ("are these the same decoded bytes"). The consequence is that a texture the guest -
    /// or the engine, for a video picture - rewrites produces a BRAND NEW entry every time
    /// it changes, and the entry holding the previous contents stays until the budget
    /// notices it. On a movie that is a 0.75 MB picture arriving 30 times a second and an
    /// eviction pass nearly every frame, evicting whatever happened to be oldest.
    ///
    /// A slot is the same guest texture - same address, same format, same shape, same
    /// sampler state - so a new entry in a slot SUPERSEDES the one it displaces: the bytes
    /// it held are gone from guest memory and nothing can ask for them again. Dropping it
    /// there is not a policy but bookkeeping, and it is what keeps a video texture out of a
    /// budget it would otherwise churn.
    decode_slots: crate::fasthash::FxHashMap<u64, u64>,
    /// Bumped by [`RenderSceneBuilder::begin_frame`]. Entries stamped with it are in use by
    /// the frame being built and are not eviction candidates at any budget.
    decode_epoch: u64,
    /// >>> THE FINISHED `GxpTex` LIST FOR ONE CAPTURED BINDING SET, PER FRAME.
    ///
    /// Keyed by the IDENTITY of the capture's `Arc<[BoundTexture]>` (its pointer and length,
    /// with the source held strongly so a freed address cannot be recycled into a stale hit -
    /// the same discipline as `tex_key` [[vitaslop-an-address-is-not-an-identity]]).
    ///
    /// # Why it is exactly equivalent, which is the only thing that makes it admissible
    /// The capture hands every draw with the same bindings ONE `Arc`
    /// (`TextureSnapshots::snapshot_sets`), and `texture()` is a pure function of each
    /// `BoundTexture` apart from two SIDE EFFECTS: the `decode_used` stamp that keeps an entry
    /// out of this frame's eviction, and the `decode_frame_bytes` tally. Both are PER FRAME and
    /// idempotent within one - the stamp writes the current epoch, and the tally adds only on
    /// the first sighting in a frame. So re-deriving the list once per frame per set performs
    /// every effect the per-draw derivation did, and the draws after the first in that frame
    /// were paying a hash and two map probes per bound unit to reproduce a list byte for byte.
    /// MEASURED on a retail sports title: **3,782 of those lookups a frame** over 672 draws,
    /// for about a dozen distinct sets.
    ///
    /// The epoch is part of the entry rather than the key so a set that survives into the next
    /// frame is REBUILT there (paying the stamp) instead of hit stale.
    gxp_tex_sets: crate::fasthash::FxHashMap<(usize, usize), GxpTexSet>,
    /// The same, for the `GxpAttr` list - which has no side effects at all, being a pure
    /// rewrite of the vertex PROGRAM's own declared attributes. The capture already shares
    /// those (`capture::Draw::attributes`), so this holds the derived form against the same
    /// identity and no epoch is needed.
    gxp_attr_sets: crate::fasthash::FxHashMap<(usize, usize), GxpAttrSet>,
    /// The largest ONE-FRAME decode working set seen, and the floor the budget is raised to.
    ///
    /// A cache that cannot hold one frame is worse than none: every entry it drops mid-frame
    /// is re-decoded before that same frame ends.
    ///
    /// # Why this floor cannot raise the PEAK heap
    /// The obvious objection is that a 330 MB floor adds 330 MB to a wasm heap that is already
    /// near what a phone will allow. It does not. Every decoded texture a frame samples is
    /// held by that frame's built scenes for as long as the frame is being built and encoded -
    /// the `GxmTexture`s carry `Arc`s to exactly these buffers - so the bytes are live during
    /// the frame whether or not a cache also references them. The floor only extends their
    /// LIFETIME across the frame boundary; the high-water mark is set by the frame itself and
    /// is unchanged. What it trades is memory held between frames against a re-decode, which
    /// is the trade a cache exists to make.
    decode_frame_high: usize,
    /// Bytes first touched by the CURRENT frame, accumulating toward [`Self::decode_frame_high`].
    decode_frame_bytes: usize,
    /// Decoded RGBA8 bytes held by `decode_cache`, against `decode_cache_budget_bytes`.
    decode_cache_bytes: usize,
    /// Keys this run has EVICTED and not yet seen come back, so a re-decode of one can be
    /// counted as thrash rather than as ordinary miss traffic. Keys only - the point is
    /// identity, and holding the pixels would defeat the eviction that put them here.
    decode_evicted: crate::fasthash::FxHashSet<u64>,
    /// The last reported "the whole scene was dropped" tally, so [`DropTally::report`]
    /// prints when the shape CHANGES rather than sixty times a second.
    last_empty: Option<DropTally>,
    /// Expanded triangle-LIST `u32` index buffers, keyed by the CONTENT of the guest index
    /// buffer they were expanded from plus the topology that decides the expansion.
    ///
    /// The expansion is a pure function of those inputs, and a title re-submits the same
    /// indices for its static geometry every frame - MEASURED at 0.74-0.98 MB of freshly
    /// allocated index bytes per frame mid-race, rebuilt from scratch sixty times a second
    /// to the same answer.
    ///
    /// Keyed by CONTENT, not by the source buffer's identity, and that is not a preference:
    /// the capture reads a draw's indices out of guest memory into a fresh allocation every
    /// frame and then REBASES them in place, so the buffer's address is different every
    /// frame and an identity key would never hit once while growing without bound. Hashing
    /// the bytes is O(n) but sequential, and it replaces an allocation plus an expansion of
    /// three times the size.
    index_cache: crate::fasthash::FxHashMap<IndexKey, (Arc<[u8]>, Arc<[u8]>)>,
}

/// What an expanded index buffer is a function of - see `RenderSceneBuilder::index_cache`.
///
/// # This was a CONTENT hash, and it stopped needing to be
/// The key used to be FNV-1a over the whole rebased index buffer, on the stated grounds that
/// "the capture reads a draw's indices into a fresh allocation every frame and then rebases them
/// in place, so the buffer's address is different every frame and an identity key would never hit
/// once while growing without bound". That was true and is not any more: the capture now shares
/// ONE rebased buffer across frames for a mesh the guest has not rewritten
/// (`TextureSnapshots::get_or_read_indices`), so its address is stable for exactly as long as its
/// contents are.
///
/// So the key is the buffer's IDENTITY, and the O(n) hash of every draw's indices is gone. The
/// cache holds a strong reference to the source buffer alongside the expansion, which is what
/// makes an address a valid key: without it a freed buffer's address could be reused by an
/// unrelated mesh and serve it someone else's triangles
/// ([[vitaslop-content-hash-cache-must-verify]] is the same lesson from the other side).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct IndexKey {
    /// The rebased index buffer's address, which the entry holds a strong reference to.
    buffer: usize,
    len: usize,
    index_count: u32,
    primitive: u32,
    index_format: u32,
}

/// What [`RenderSceneBuilder::build`] actually DID, in units neither engine has to have a
/// clock for.
///
/// `build` is the largest item in the browser's render half, and the browser has no
/// `std::time::Instant` at all - `encode_chain`'s inner split is structurally zero there for
/// exactly that reason. So the instrument that works on BOTH engines is a count of the work
/// itself: vertices walked, indices scanned, textures decoded. A count also survives the
/// comparison a time cannot, since the two engines run on different hardware: 41 us a draw
/// against 2 us a draw is only a mystery until one of them says it walked forty times as
/// many vertices.
///
/// Bumped once per draw (not per vertex), so the cost of the instrument is a handful of
/// relaxed adds per draw against the hundreds of thousands of vertices it counts.
#[derive(Default, Clone, Copy)]
pub struct BuildWork {
    /// Vertices decoded and transformed by the MAIN loop - the fixed-function
    /// representation, or an inline depth-range walk.
    pub verts_walked: u64,
    /// Vertices walked by PASS TWO, purely to measure the scene depth range for some other
    /// draw that reads it. This is the walk the two-pass gate exists to avoid.
    pub verts_deferred: u64,
    /// Indices read by the max-index scan that sizes the walk.
    pub indices_scanned: u64,
    /// Textures DECODED (a cache miss) and textures served from the decode cache.
    pub tex_decoded: u64,
    pub tex_cached: u64,
    /// Draws served a whole `GxpTex` list from [`RenderSceneBuilder::gxp_tex_sets`] - i.e.
    /// draws that did NOT look a single bound texture up. Counted apart from `tex_cached`
    /// because the two answer different questions: that one is "the decode was cached", this
    /// one is "the lookup did not happen either", and the second is what the per-set cache
    /// bought. `tex_cached` falling while this rises is the fix working, not work vanishing.
    pub tex_set_reused: u64,
    /// Cache misses whose texels were actually EXPANDED to RGBA8, and the bytes that expansion
    /// produced.
    ///
    /// # Why this is separate from `tex_decoded`
    /// A miss builds a texture; it no longer decodes one ([`vitaslop_platform::gpu::Texels`]).
    /// In the shipped configuration a texture the GPU takes as blocks is never expanded at all,
    /// so `tex_decoded` counts the work that USED to happen and this counts what still does.
    /// Without the split, "3.7 textures decoded per frame" would keep reading as 3.7 whole-image
    /// decodes long after the decode stopped happening - a live counter whose name outlived what
    /// it counts, which is the failure the eviction-pass counter already taught once.
    ///
    /// Charged where the expansion happens, which is not where the texture was captured, so it
    /// lands in the process-wide tally rather than in a scene's own.
    pub tex_expanded: u64,
    pub tex_expanded_bytes: u64,
    /// Bytes of texture the decoder read, so a small count of huge textures is separable
    /// from a large count of small ones.
    pub tex_bytes: u64,
    /// Decoded OUTPUT bytes, split by which decode path produced them. The block-wise path
    /// decodes each compressed block once; everything else answers one texel at a time. A
    /// burst that is expensive despite the fast path is a burst of formats the fast path
    /// does not cover, and only this split says which.
    pub tex_out_blockwise: u64,
    pub tex_out_per_texel: u64,
    /// Times a decode went over budget and ran an EVICTION PASS, and how many entries those
    /// passes dropped.
    ///
    /// # This counter used to mean something else, and the old name outlived the change
    /// It was `tex_cache_clears`, and it counted WHOLESALE clears of the decode cache. When
    /// eviction went per-entry the counting site moved with it but the name and the doc did
    /// not, so a healthy LRU under steady pressure - which runs an eviction pass whenever the
    /// budget is touched, by design - reported "2.13 cache clears per frame" and read as the
    /// exact cliff the per-entry rewrite had been done to remove. A stale name on a live
    /// counter is worse than no counter: it does not merely fail to answer, it answers wrongly
    /// and with a number attached.
    ///
    /// **Neither of these is the cliff signal.** A pass that sheds cold entries is the cache
    /// working. What says the working set has outgrown the budget is
    /// [`Self::tex_redecoded_after_evict`].
    pub tex_evict_passes: u64,
    pub tex_evicted: u64,
    /// Textures decoded whose key had been EVICTED earlier in the run - the cache thrashing.
    ///
    /// This is the number the old `tex_cache_clears` was being read as. A cache under healthy
    /// pressure evicts what is not coming back, so this stays near zero however many eviction
    /// passes run; a cache whose budget is below the working set evicts exactly what the next
    /// frame asks for, and then every eviction shows up here one frame later. It separates
    /// "the cache is doing its job" from "the cache is a re-decode loop with extra steps",
    /// which is the distinction a pass count cannot make in either direction.
    pub tex_redecoded_after_evict: u64,
    /// Decodes DROPPED because a newer decode took their slot - the same guest texture with
    /// new contents. Not evictions: the bytes they held are gone from guest memory, so this
    /// is the cache staying the size of what is reachable rather than the size of the run.
    /// A movie is 30 of these a second and they used to be eviction pressure.
    pub tex_superseded: u64,
    /// Draws built, and how many of them ran their fixed-function representation.
    pub draws: u64,
    pub draws_fixed_function: u64,
    /// Bytes of RAW guest vertex stream COPIED for the recompiled path (`d.vertices.clone()`,
    /// once per draw per frame), and bytes of the expanded u32 index buffer built beside it.
    /// Both are O(mesh) per draw and neither is shared between frames, so they are the two
    /// candidates for a `build` that scales with geometry rather than with draw count.
    pub gxp_vertex_bytes: u64,
    pub gxp_index_bytes: u64,
    /// Bytes of default-uniform-buffer (SA bank) copied per draw for the two stages.
    pub gxp_sa_bytes: u64,
    /// Index buffers EXPANDED from the guest topology, and index buffers served from the
    /// expansion cache. `gxp_index_bytes` counts only the expanded ones - a cache hit
    /// allocates nothing.
    pub index_expanded: u64,
    pub index_expand_cached: u64,
    pub index_cache_clears: u64,
}

impl BuildWork {
    /// Fold another tally in. Public so a caller can accumulate PER PRESENT and keep the
    /// worst one apart from the window total.
    pub fn add_pub(&mut self, o: &BuildWork) {
        self.add(o);
    }

    fn add(&mut self, o: &BuildWork) {
        self.verts_walked += o.verts_walked;
        self.verts_deferred += o.verts_deferred;
        self.indices_scanned += o.indices_scanned;
        self.tex_decoded += o.tex_decoded;
        self.tex_cached += o.tex_cached;
        self.tex_set_reused += o.tex_set_reused;
        self.tex_expanded += o.tex_expanded;
        self.tex_expanded_bytes += o.tex_expanded_bytes;
        self.tex_bytes += o.tex_bytes;
        self.tex_out_blockwise += o.tex_out_blockwise;
        self.tex_out_per_texel += o.tex_out_per_texel;
        self.tex_evict_passes += o.tex_evict_passes;
        self.tex_evicted += o.tex_evicted;
        self.tex_redecoded_after_evict += o.tex_redecoded_after_evict;
        self.tex_superseded += o.tex_superseded;
        self.draws += o.draws;
        self.draws_fixed_function += o.draws_fixed_function;
        self.gxp_vertex_bytes += o.gxp_vertex_bytes;
        self.gxp_index_bytes += o.gxp_index_bytes;
        self.gxp_sa_bytes += o.gxp_sa_bytes;
        self.index_expanded += o.index_expanded;
        self.index_expand_cached += o.index_expand_cached;
        self.index_cache_clears += o.index_cache_clears;
    }

    /// One line, per FRAME (the caller divides), naming every unit above.
    pub fn line(&self, frames: u64) -> String {
        let n = frames.max(1) as f64;
        format!(
            "build work/frame: {:.0} draws ({:.0} fixed-function), {:.0} vertices walked \
             (+{:.0} deferred depth-range), {:.0} indices scanned, textures {:.1} built \
             / {:.1} cached / {:.1} whole SETS reused over {:.2} MB of guest bytes, {:.1} EXPANDED to RGBA8 \
             ({:.2} MB: {:.2} MB fast-path + {:.2} MB per-texel), \
             {:.2} evict passes dropping {:.1} entries, {:.1} RE-decoded after eviction, \
             {:.1} superseded in place, indices {:.1} expanded \
             / {:.1} cached ({:.2} clears), gxp shares {:.2} MB vertices, copies {:.2} MB \
             indices + {:.2} MB uniforms",
            self.draws as f64 / n,
            self.draws_fixed_function as f64 / n,
            self.verts_walked as f64 / n,
            self.verts_deferred as f64 / n,
            self.indices_scanned as f64 / n,
            self.tex_decoded as f64 / n,
            self.tex_cached as f64 / n,
            self.tex_set_reused as f64 / n,
            self.tex_bytes as f64 / n / (1024.0 * 1024.0),
            self.tex_expanded as f64 / n,
            self.tex_expanded_bytes as f64 / n / (1024.0 * 1024.0),
            self.tex_out_blockwise as f64 / n / (1024.0 * 1024.0),
            self.tex_out_per_texel as f64 / n / (1024.0 * 1024.0),
            self.tex_evict_passes as f64 / n,
            self.tex_evicted as f64 / n,
            self.tex_redecoded_after_evict as f64 / n,
            self.tex_superseded as f64 / n,
            self.index_expanded as f64 / n,
            self.index_expand_cached as f64 / n,
            self.index_cache_clears as f64 / n,
            self.gxp_vertex_bytes as f64 / n / (1024.0 * 1024.0),
            self.gxp_index_bytes as f64 / n / (1024.0 * 1024.0),
            self.gxp_sa_bytes as f64 / n / (1024.0 * 1024.0),
        )
    }
}

static BUILD_WORK: std::sync::Mutex<BuildWork> = std::sync::Mutex::new(BuildWork {
    verts_walked: 0,
    verts_deferred: 0,
    indices_scanned: 0,
    tex_decoded: 0,
    tex_cached: 0,
    tex_set_reused: 0,
    tex_expanded: 0,
    tex_expanded_bytes: 0,
    tex_bytes: 0,
    tex_out_blockwise: 0,
    tex_out_per_texel: 0,
    tex_evict_passes: 0,
    tex_evicted: 0,
    tex_redecoded_after_evict: 0,
    tex_superseded: 0,
    draws: 0,
    draws_fixed_function: 0,
    gxp_vertex_bytes: 0,
    gxp_index_bytes: 0,
    gxp_sa_bytes: 0,
    index_expanded: 0,
    index_expand_cached: 0,
    index_cache_clears: 0,
});

/// Take and RESET the accumulated build work. The caller owns the window it divides by.
pub fn take_build_work() -> BuildWork {
    let mut g = BUILD_WORK.lock().unwrap();
    std::mem::take(&mut *g)
}

/// Decoded RGBA8 output bytes per guest `base_format`, for the whole run.
///
/// # Why the format, and not just the total
/// The decode volume is the browser's largest single cost, and the obvious next move -
/// hand WebGPU the guest's COMPRESSED bytes instead of expanding them - is only available
/// for some formats. WebGPU has `texture-compression-bc`; it has no PVRTC format at all,
/// and PVRTC is the PowerVR-native family this console's titles are most likely to use.
/// So "replace the decoder with a compressed upload" is worth a session or worth nothing
/// depending entirely on this split, and it is not something to guess from the platform.
///
/// Indexed by base format (the field is 8 bits), so this is a flat array and costs one
/// add per decoded face.
static DECODE_BY_FORMAT: std::sync::Mutex<[u64; 256]> = std::sync::Mutex::new([0; 256]);

/// The formats this run decoded, largest first, as `(base_format, output MB)`.
/// Only formats that actually decoded something appear.
pub fn decode_by_format() -> Vec<(u32, f64)> {
    let g = DECODE_BY_FORMAT.lock().unwrap();
    let mut v: Vec<(u32, f64)> = g
        .iter()
        .enumerate()
        .filter(|(_, b)| **b > 0)
        .map(|(f, b)| (f as u32, *b as f64 / (1024.0 * 1024.0)))
        .collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// One line naming what the decoder actually spent its bytes on.
pub fn decode_by_format_line() -> String {
    let v = decode_by_format();
    if v.is_empty() {
        return "texture decode by format: nothing decoded".to_string();
    }
    let total: f64 = v.iter().map(|(_, mb)| mb).sum();
    let parts: Vec<String> = v
        .iter()
        .take(8)
        .map(|(f, mb)| format!("{:#04x} {} {:.1} MB", f, format_family(*f), mb))
        .collect();
    format!("texture decode by format: {total:.1} MB total - {}", parts.join(", "))
}

/// Which upload family a base format belongs to, which is the question the split exists to
/// answer: `bc` can be handed to WebGPU compressed, `pvrtc` cannot (no WebGPU format), and
/// `raw` is already uncompressed so there is nothing to hand over.
fn format_family(base_format: u32) -> &'static str {
    if crate::pvrtc::Variant::from_base_format(base_format).is_some() {
        return "PVRTC(no WebGPU format)";
    }
    match block_layout(base_format) {
        Some((bw, _, _)) if bw > 1 => "BC(uploadable)",
        Some(_) => "raw",
        None => "undecodable",
    }
}

/// Why [`RenderSceneBuilder::build`] discarded draws from a captured scene.
///
/// # Why this is reported rather than silent
/// `build` is allowed to drop a captured draw - a line/point topology has no triangles,
/// and a position-only stream has no colour source the fixed-function path can honour.
/// Each drop is individually correct, but the SUM of them is not: a scene where every
/// draw is dropped renders a bare clear colour, which on screen is indistinguishable from
/// "the guest submitted nothing". That ambiguity cost a whole session on one title, where
/// the capture reported `5draws@960x544` every frame and the renderer reported `0 draws`
/// and the two facts sat side by side without either being wrong.
///
/// So a drop reports itself unconditionally, naming the reason - the same rule the
/// recompiler's fixed-function fallback follows.
///
/// # Every drop reports, not just a total one
/// Reporting only the all-dropped case was itself a silent failure: a single discarded draw
/// among hundreds is a missing button fill, a missing decal, a missing glyph - invisible in a
/// summary that says nothing, and indistinguishable from the guest not drawing it. The FIRST
/// drop of each kind now prints regardless of how many survive; a repeat of the same kind is
/// suppressed so a per-frame drop cannot flood the log. `VITASLOP_STRICT_DRAWS` turns any
/// drop into a hard failure, which is the mode to run when a frame looks wrong.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct DropTally {
    /// Draws in the captured scene.
    total: usize,
    /// Dropped: not a triangle topology (lines/points emit no triangles).
    topology: usize,
    /// Dropped: the vertex stream holds shader-expanded sprite RECORDS, not vertices.
    expanded: usize,
    /// Dropped: position-only geometry - no texcoord and no vertex colour, so the
    /// fixed-function path has no colour source and would paint opaque white.
    colorless: usize,
    /// Of the dropped draws, how many carried the guest's real shaders. These are the
    /// expensive ones to lose: the recompiler could have drawn them exactly.
    with_shaders: usize,
}

impl DropTally {
    fn dropped(&self) -> usize {
        self.topology + self.expanded + self.colorless
    }

    /// Report a scene that produced no drawable geometry at all. Returns whether this
    /// tally differs from `last`, so the caller can print on change only.
    fn report_if_total(&self, last: &mut Option<DropTally>) {
        if self.total == 0 || self.dropped() < self.total || *last == Some(*self) {
            return;
        }
        *last = Some(*self);
        tracing::warn!(
            target: "vitaslop::render",
            "render: ALL {} captured draws were dropped - {} non-triangle topology, {} \
             shader-expanded sprite records, {} position-only (no texcoord, no vertex colour); \
             {} of them carried the guest's real shaders. This frame renders as a bare clear \
             colour, which is NOT the same as the guest drawing nothing.",
            self.total, self.topology, self.expanded, self.colorless, self.with_shaders,
        );
    }
}

/// Why one captured draw was discarded, for [`report_drop`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DropKind {
    Topology,
    Expanded,
    Colorless,
}

impl DropKind {
    fn describe(self) -> &'static str {
        match self {
            DropKind::Topology => {
                "not a triangle topology (a line/point list emits no triangles)"
            }
            DropKind::Expanded => {
                "the vertex stream holds shader-expanded sprite RECORDS, not vertices"
            }
            DropKind::Colorless => {
                "position-only geometry: no texcoord and no vertex colour, so the \
                 fixed-function path has no colour source"
            }
        }
    }
}

/// `VITASLOP_STRICT_DRAWS`: turn any dropped draw into a hard failure instead of a report.
/// Off by default because several drops are legitimate on titles that render correctly (a
/// point-list topology genuinely has no triangles), but a frame that looks wrong should be
/// re-run under this so the FIRST missing draw stops the run at its own draw index rather
/// than being reasoned about from a screenshot.
fn strict_draws() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| std::env::var_os("VITASLOP_STRICT_DRAWS").is_some())
}

/// Report one dropped draw the first time a drop of that kind happens, and panic instead
/// when `VITASLOP_STRICT_DRAWS` is set.
///
/// Suppressing repeats is what keeps this usable: a draw dropped every frame would otherwise
/// print sixty times a second. The first one carries the information that matters - which
/// draw, why, and whether it had the guest's real shaders attached (a drop that did is one
/// the recompiler could have drawn exactly, so it is the expensive kind to lose).
fn report_drop(kind: DropKind, di: usize, d: &Draw, tri_count: usize) {
    // The PRIMITIVE and the INDEX COUNT ride the topology drop, and they are not
    // decoration: "not a triangle topology" names a family, and lines, points and packed
    // edge lists are now DRAWN, so a drop that survives is either an edge list whose words
    // REFUSE the packed reading, a shaderless line/point draw, or a draw with too few
    // indices to make one primitive. Without these numbers those are indistinguishable,
    // which cost a re-run.
    // >>> AN EDGE-LIST DROP STILL CARRIES ITS RAW INDICES. The packed-flags encoding was
    // established by exactly this dump (see `PRIM_TRIANGLE_EDGES`); a draw that lands here
    // now is one whose fourth words carry something OTHER than the three flag bits, and its
    // words are the evidence the next reading starts from. Guessing instead would be an
    // approximation, and this project does not ship one
    // ([[vitaslop-no-approximation-no-omission]]).
    let edges = if d.primitive == PRIM_TRIANGLE_EDGES {
        let n = (d.index_count as usize).min(EDGE_LIST_INDEX_DUMP);
        let vals: Vec<String> = (0..n).map(|i| format!("{:#x}", index_at(d, i))).collect();
        format!(
            " EDGE LIST - the first {n} of {} index words, {}-bit, are [{}]; bits 0x100/0x200/0x400 \
             on these are SceGxmEdgeEnableFlags if they are packed here at all.",
            d.index_count,
            if d.index_format == 0 { 16 } else { 32 },
            vals.join(", ")
        )
    } else {
        String::new()
    };
    let detail = format!(
        "render: DROPPED draw {di} - {}. primitive={:#010x}, indices={}, tris={tri_count}, \
         stride={}, {} attributes, shaders={}. This draw is MISSING from the frame; the \
         guest asked for it.{edges}",
        kind.describe(),
        d.primitive,
        d.index_count,
        d.vertex_stride,
        d.attributes.len(),
        if d.vprog.is_empty() { "none" } else { "yes (the recompiler could draw this)" },
    );
    if strict_draws() {
        panic!("{detail}\n(VITASLOP_STRICT_DRAWS is set, so a dropped draw is fatal)");
    }
    use std::sync::Mutex;
    static SEEN: Mutex<Vec<DropKind>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap();
    if seen.contains(&kind) {
        return;
    }
    seen.push(kind);
    tracing::warn!(target: "vitaslop::render", "{detail}");
}

/// Does anything in this build CONSUME the scene depth range?
///
/// `RenderScene::depth_min`/`depth_scale` are read by exactly two things: the
/// FIXED-FUNCTION MVP pipeline, and the recompiled path under `VITASLOP_GXP_ZFIX=range`.
/// The default is `ZFix::Clamp`, which writes the guest's own window depth `z/w` and - as
/// its own documentation says - needs no scene statistics at all.
///
/// Measuring the range costs a position decode and a 4x4 transform for every vertex of
/// every opaque draw. On a race frame where every draw is recompiled, that is the largest
/// single item in `build` (measured in the browser: `build` 21.6 ms against 6.6 ms for
/// every WebGPU call of the frame put together) and NOTHING reads the answer.
///
/// Read once, not per scene.
fn zfix_consumes_scene_depth_range() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::knobs::var("VITASLOP_GXP_ZFIX").map(|v| v.trim() == "range").unwrap_or(false)
    })
}

/// Name the first draw that makes a scene measure its depth range, once per run.
///
/// The range costs a position decode and a 4x4 transform for every vertex of every opaque
/// MVP draw in the scene, and exactly one kind of draw reads it: a FIXED-FUNCTION, opaque,
/// MVP one. On a fully-recompiled frame there is none and the walk never runs - measured,
/// a race scene of 690 recompiled draws builds in 1.57 ms, and one that walks costs about
/// eighteen times that per draw.
///
/// So "which draw turned it on" is the difference between a legitimate cost and a whole
/// scene walking for a draw that is dropped or invisible, and it is not something to infer
/// from a frame time. Reported unconditionally, once, for the same reason a fallback is
/// [[vitaslop-fallback-must-report]].
fn report_depth_range_reader(di: usize, d: &Draw) {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        // A `tracing` event, not an `eprintln!`: the browser has no stdio, so a report
        // printed that way is invisible on the one engine whose `build` cost raised the
        // question. [[vitaslop-fallback-must-report]]
        tracing::warn!(
            target: "vitaslop::render",
            "render: draw {di} is FIXED-FUNCTION, opaque and MVP, so its scene measures the \
             opaque depth range - a per-vertex walk over every opaque MVP draw in that scene. \
             tris={}, stride={}, {} attributes, {} uniform floats, primitive {:#x}. Every other \
             scene of the frame skips the walk.",
            triangle_count(d),
            d.vertex_stride,
            d.attributes.len(),
            d.uniform_bank_floats(),
            d.primitive,
        );
    });
}

/// Budget for the decode cache, in BYTES of decoded RGBA8, before it is cleared wholesale.
///
/// # In bytes, because a count of entries bounded nothing
/// This was a cap of 512 ENTRIES, on the reasoning that "a title's working texture set is
/// far smaller". MEASURED on a mid-race frame of a 705-draw title, it is not: the cache
/// reached the cap and was dumped roughly once every thirty frames, and each dump cost
/// about five hundred re-decodes. Averaged over the perf window that read as **15.7
/// textures decoded per frame against 0.3 on a lighter frame, and it took `build` from
/// 9.7 ms to 166.8 ms in the browser** - a cliff, not a gradient, which is exactly what a
/// wholesale clear produces. The same unit error, on the same kind of cache, is written up
/// in `tex_cache_budget_bytes` in the platform crate; this one was left in the old unit.
///
/// An entry is bounded by what it costs to REBUILD (a decode) and by what it holds (the
/// decoded pixels), and only the second is measurable here, so that is the budget.
/// `VITASLOP_DECODE_CACHE_MB` overrides. 256 MB matches the view cache it feeds.
fn decode_cache_budget_bytes() -> usize {
    static CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let base = *CELL.get_or_init(|| {
        crate::knobs::var("VITASLOP_DECODE_CACHE_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(256)
            * 1024
            * 1024
    });
    // Scaled to the device - see [`crate::knobs::memory_scale`]. The FLOOR at one frame's
    // working set (`decode_frame_high`, applied by the caller) still holds and is what stops a
    // scaled-down budget from re-decoding inside a frame.
    crate::knobs::scale_budget(base)
}

/// Whether PVRTC decodes a whole face at a time (the default) or one texel at a time.
///
/// # Why the slow path is kept reachable
/// `VITASLOP_PVRTC_DECODE=per-texel` forces every PVRTC texture through [`crate::pvrtc::texel`],
/// the oracle the whole-image pass was written against. The two are supposed to be
/// byte-for-byte identical, so a run that differs only in this knob must render an IDENTICAL
/// frame - which is a falsifier over the title's REAL textures, in their real sizes,
/// addressing modes and sub-modes, rather than over the handful a unit test can construct.
/// That is the same contract, and the same reason, as `VITASLOP_TEXTURE_CHECK=bytes`.
///
/// It is a diagnostic, not a tuning knob: the slow path is the one that decodes every block
/// eighty times. An unrecognised value PANICS rather than silently picking one.
fn pvrtc_whole_image() -> bool {
    static CELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| match crate::knobs::var("VITASLOP_PVRTC_DECODE") {
        Ok(s) if s.trim() == "per-texel" => {
            tracing::warn!(
                target: "vitaslop::render",
                "VITASLOP_PVRTC_DECODE=per-texel: PVRTC decodes one texel at a time, which \
                 re-decodes every block eighty times. This is the exactness falsifier, not a \
                 setting to run in."
            );
            false
        }
        Ok(s) if s.trim() == "whole-image" || s.trim().is_empty() => true,
        Ok(s) => panic!(
            "VITASLOP_PVRTC_DECODE={s:?} is not a mode - use `whole-image` (default) or \
             `per-texel`"
        ),
        Err(_) => true,
    })
}

/// Cap on the expanded-index cache, in ENTRIES. Same contract as `DECODE_CACHE_CAP`: the key
/// is a content fingerprint, so clearing wholesale costs a re-expansion and never
/// correctness. Larger than the decode cap because an entry is one mesh's indices rather
/// than a decoded image, and a scene can carry hundreds of distinct meshes.
const INDEX_CACHE_CAP: usize = 4096;

/// Cap on the remembered-evicted key set (`RenderSceneBuilder::decode_evicted`), in ENTRIES.
/// Eight bytes each, so this is a fraction of a megabyte, and it is a DIAGNOSTIC bound: past it
/// the thrash count under-reports rather than the set growing with the run.
const EVICTED_KEYS_CAP: usize = 1 << 16;

impl Default for RenderSceneBuilder {
    fn default() -> Self {
        RenderSceneBuilder::new()
    }
}

impl RenderSceneBuilder {
    pub fn new() -> Self {
        RenderSceneBuilder {
            gxp_only: crate::knobs::flag("VITASLOP_GXP_LIVE")
                && !crate::knobs::flag("VITASLOP_GXP_ALLOW_FIXED_FUNCTION"),
            decode_cache: Default::default(),
            decode_slots: Default::default(),
            decode_cache_bytes: 0,
            decode_used: Default::default(),
            decode_epoch: 0,
            gxp_tex_sets: Default::default(),
            gxp_attr_sets: Default::default(),
            decode_frame_high: 0,
            decode_frame_bytes: 0,
            decode_evicted: Default::default(),
            last_empty: None,
            index_cache: Default::default(),
        }
    }

    /// Start a new FRAME's worth of scene building.
    ///
    /// A frame is many scenes ([[vitaslop-a-frame-is-many-scenes]]), so the builder cannot see
    /// the boundary on its own - and the boundary is exactly what the texture cache needs, for
    /// two things: which entries are in use right now (and so un-evictable), and how big one
    /// frame's working set really is. Both engines call this immediately before building the
    /// frame's scenes; a caller that forgets simply never bumps the epoch, which leaves every
    /// entry looking in-use and the cache growing to the whole run, so it is a loud kind of
    /// wrong rather than a silent one.
    pub fn begin_frame(&mut self) {
        // Take the finished frame's working set BEFORE resetting, so the floor is only ever
        // set by a COMPLETE frame.
        //
        // It RELAXES rather than ratcheting: a plain running maximum would let one heavy
        // loading frame pin the budget for the rest of the run, which is how a cache that was
        // supposed to adapt turns into an unbounded one. Decaying by 1/16 a frame means a
        // floor set by a burst is forgotten within a second of gameplay, while a working set
        // that is genuinely this size re-asserts it on every frame and never decays at all.
        self.decode_frame_high = self
            .decode_frame_bytes
            .max(self.decode_frame_high - self.decode_frame_high / 16);
        self.decode_frame_bytes = 0;
        self.decode_epoch = self.decode_epoch.wrapping_add(1);
        // >>> THE PER-SET CACHES ARE DROPPED AT THE FRAME BOUNDARY, AND THAT IS A BOUND, NOT
        // >>> TIDINESS.
        //
        // A `GxpTexSet` holds `GxmTexture`s, and a `GxmTexture` holds its PIXELS. Keeping them
        // across frames would pin texture bytes outside `decode_cache`'s budget - a second,
        // unbudgeted copy of the working set in a wasm heap that can never hand a page back,
        // which is precisely the shape of the last pooling change that cost the user frame
        // rate. Cleared here, the caches can hold at most what THIS frame binds, which
        // `decode_cache` is already holding anyway, so they add no resident bytes at all - and
        // an entry could not be used across frames regardless, because a hit requires the
        // current epoch (see `gxp_tex_sets`).
        self.gxp_tex_sets.clear();
        // The attribute lists carry no pixels, only a handful of `u16`s per attribute, so they
        // are kept - they are a function of the vertex PROGRAM and are the same every frame.
    }

    /// The recompiler's `GxpTex` list for a captured binding set, derived once per frame per
    /// set. See [`RenderSceneBuilder::gxp_tex_sets`] for why once per frame is exactly what the
    /// per-draw derivation did.
    fn gxp_textures(
        &mut self,
        src: &Arc<[BoundTexture]>,
        work: &mut BuildWork,
    ) -> Arc<[vitaslop_platform::gpu::GxpTex]> {
        if src.is_empty() {
            return Arc::from(&[][..]);
        }
        let key = (Arc::as_ptr(src) as *const BoundTexture as usize, src.len());
        let epoch = self.decode_epoch;
        if let Some(e) = self.gxp_tex_sets.get(&key) {
            // The pointer alone is not the identity: hold the source and compare it, so a
            // freed set whose address was handed to a different one cannot answer here.
            if e.epoch == epoch && Arc::ptr_eq(&e.src, src) {
                work.tex_set_reused += 1;
                return e.out.clone();
            }
        }
        let out: Arc<[vitaslop_platform::gpu::GxpTex]> = src
            .iter()
            .map(|t| vitaslop_platform::gpu::GxpTex {
                unit: t.unit as u8,
                tex: self.texture(t, work),
            })
            .collect();
        if self.gxp_tex_sets.len() >= GXP_SET_CACHE_CAP {
            self.gxp_tex_sets.clear();
        }
        self.gxp_tex_sets.insert(key, GxpTexSet { src: src.clone(), out: out.clone(), epoch });
        out
    }

    /// The recompiler's `GxpAttr` list for a vertex program's attributes, derived once per
    /// distinct attribute list. See [`RenderSceneBuilder::gxp_attr_sets`].
    fn gxp_attributes(
        &mut self,
        src: &Arc<[crate::capture::VertexAttribute]>,
    ) -> Arc<[vitaslop_platform::gpu::GxpAttr]> {
        if src.is_empty() {
            return Arc::from(&[][..]);
        }
        let key = (Arc::as_ptr(src) as *const crate::capture::VertexAttribute as usize, src.len());
        if let Some(e) = self.gxp_attr_sets.get(&key) {
            if Arc::ptr_eq(&e.src, src) {
                return e.out.clone();
            }
        }
        let out: Arc<[vitaslop_platform::gpu::GxpAttr]> = src
            .iter()
            .map(|a| vitaslop_platform::gpu::GxpAttr {
                reg_index: a.reg_index,
                offset: a.offset,
                gxm_format: a.format,
                components: a.component_count,
            })
            .collect();
        if self.gxp_attr_sets.len() >= GXP_SET_CACHE_CAP {
            self.gxp_attr_sets.clear();
        }
        self.gxp_attr_sets.insert(key, GxpAttrSet { src: src.clone(), out: out.clone() });
        out
    }

    /// Decode (or reuse a cached) GPU-ready texture for `t`.
    fn texture(&mut self, t: &BoundTexture, work: &mut BuildWork) -> GxmTexture {
        let key = tex_key(t);
        if let Some((g, _)) = self.decode_cache.get(&key) {
            work.tex_cached += 1;
            let g = g.clone();
            // The price recorded when this entry was inserted, NOT a fresh one: `decode_cache_
            // bytes` was incremented by that number and eviction subtracts it, so re-pricing an
            // entry here would drift the running total against the entries it is meant to
            // describe. Falls back to the shape estimate only for an entry with no record.
            //
            // >>> READ AND RE-STAMPED IN ONE MAP LOOKUP. This was a `get` followed by
            // `touch_decode`'s `insert` - two probes of the same key, on a path taken once per
            // bound texture per draw (4,632 times a presented frame on a race). Same
            // arithmetic, same counters, one hash.
            let epoch = self.decode_epoch;
            let (cost, first_this_frame) = match self.decode_used.entry(key) {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    let (used, bytes) = *e.get();
                    e.insert((epoch, bytes));
                    (bytes, used != epoch)
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    let bytes = predicted_texture_bytes(
                        g.width, g.height, g.faces, g.texel, g.compressed.as_ref(),
                    );
                    e.insert((epoch, bytes));
                    (bytes, true)
                }
            };
            if first_this_frame {
                self.decode_frame_bytes += cost;
            }
            return g;
        }
        work.tex_decoded += 1;
        // Was this key here before and thrown out? That, and not the number of eviction passes,
        // is what says the budget is under the working set - see `tex_redecoded_after_evict`.
        // Removed on the way past, so one eviction can only ever be blamed once.
        if self.decode_evicted.remove(&key) {
            work.tex_redecoded_after_evict += 1;
        }
        work.tex_bytes += t.pixels.len() as u64;
        let (width, height, texel) = decoded_texture_shape(t);
        // >>> THE COMPRESSED HANDOVER IS DECIDED BEFORE THE DECODE, BECAUSE IT DECIDES THE DECODE.
        //
        // This used to sit below, after an unconditional expansion to RGBA8, on the argument that
        // "whether the device can take blocks at all is not knowable here". It is: the adapter's
        // family is published (`gpu::block_family`) and `compressed_source` already gates on it.
        // What was not knowable was the ORDER - and getting it wrong meant every texture the GPU
        // was about to take as blocks was decoded first and the decode thrown away.
        let compressed = compressed_source(t, None);
        // Read before `compressed` is moved into the struct below, and read for the reason the
        // `raw` field explains: it is the test for "nothing else claimed this texture".
        let has_compressed = compressed.is_some();
        // What this entry will actually HOLD, priced without forcing the decode - see
        // [`resident_texture_bytes`].
        let cost = predicted_texture_bytes(width, height, t.faces.max(1), texel, compressed.as_ref());
        self.decode_cache_bytes += cost;
        // The budget, floored at ONE FRAME's working set - see `decode_frame_high`. Eviction
        // is per entry and never takes something this frame has already decoded, because that
        // is precisely the entry the rest of the frame is about to ask for again.
        let budget = decode_cache_budget_bytes().max(self.decode_frame_high);
        if self.decode_cache_bytes >= budget {
            let epoch = self.decode_epoch;
            // Oldest first, so steady pressure sheds what has gone longest unused rather than
            // whatever the hash order yields.
            let mut stale: Vec<(u64, u64)> = self
                .decode_used
                .iter()
                .filter(|(_, (used, _))| *used != epoch)
                .map(|(k, (used, _))| (*k, *used))
                .collect();
            stale.sort_by_key(|(_, used)| *used);
            let mut evicted = 0usize;
            for (k, _) in stale {
                if self.decode_cache_bytes < budget {
                    break;
                }
                if self.decode_cache.remove(&k).is_some() {
                    let bytes = self.decode_used.remove(&k).map_or(0, |(_, b)| b);
                    self.decode_cache_bytes = self.decode_cache_bytes.saturating_sub(bytes);
                    // Bounded, and bounded by DROPPING rather than by clearing: this set exists
                    // only to attribute a later re-decode, so forgetting an old eviction costs
                    // an under-count of thrash and nothing else. Clearing it wholesale on a cap
                    // would be the same failure shape the cache itself was rewritten to remove.
                    if self.decode_evicted.len() < EVICTED_KEYS_CAP {
                        self.decode_evicted.insert(k);
                    }
                    evicted += 1;
                }
            }
            if evicted > 0 {
                work.tex_evict_passes += 1;
                work.tex_evicted += evicted as u64;
            }
            // If nothing was evictable every entry belongs to this frame, and the frame has to
            // finish. The floor above means that state lasts exactly one frame: the next
            // `begin_frame` records this working set and the budget rises to meet it.
        }
        // Carry the magnification filter so the GPU picks the matching sampler (LINEAR ->
        // bilinear, as the software `sample_texture_bilinear` does). SceGxmTextureFilter:
        // 1 = LINEAR, 0 = POINT.
        let filter_linear = t.mag_filter == 1;
        // >>> EVERY TEXTURE IS BUILT AT THE GUEST'S OWN RESOLUTION. THERE IS NO OTHER OPTION.
        //
        // A previous version of this served a REDUCED-resolution version of a texture while a
        // full one was encoded over later frames, so that a screen transition would not freeze.
        // The reduced version is what a player sees, and on the device it never got past
        // 128 texels on a side - a 2048x2048 atlas rendered at one sixteenth of its resolution
        // per axis. That is not a trade this emulator gets to make. Whatever it costs, the
        // picture is the one the guest asked for.
        //
        // The freeze it was avoiding is real and is dealt with where it belongs: by not doing
        // an unaffordable encode at all (see `transcoded_source`), never by showing something
        // else in the meantime.
        //
        // >>> AND THE DECODE ITSELF IS DEFERRED TO ITS FIRST READER. See [`Texels`]: the
        // recompiled path never reads these bytes for a texture it takes compressed, so for that
        // texture this closure is built and never run. Everything that does read them - the
        // software rasterizer, the fixed-function upload, the vertex-texture clip probe - gets
        // exactly the bytes the eager decode produced, at its own expense.
        let src = t.clone();
        let g = GxmTexture {
            key,
            data_addr: t.data_addr,
            width,
            height,
            faces: t.faces.max(1),
            rgba: vitaslop_platform::gpu::Texels::lazy(move || {
                // The per-format byte tallies are charged HERE, where the work happens, rather
                // than to the `BuildWork` of whichever scene happened to capture the texture -
                // by the time this runs that scene is long finished. `DECODE_BY_FORMAT` was
                // always a process-wide tally for the same reason.
                let mut w = BuildWork::default();
                let (_, _, rgba, _) = decode_texture_rgba8_counted(&src, &mut w);
                w.tex_expanded += 1;
                w.tex_expanded_bytes += rgba.len() as u64;
                BUILD_WORK.lock().unwrap_or_else(|e| e.into_inner()).add(&w);
                rgba
            }),
            texel,
            // The guest's own planes for a video frame, so the uploader can convert on the
            // GPU instead of expanding to RGBA here. `rgba` above is still built lazily and
            // is still the fallback - see `GxmTexture::planar_yuv`.
            planar_yuv: (t.base_format == YUV420P2).then(|| {
                vitaslop_platform::gpu::PlanarYuvSource {
                    width: t.width,
                    height: t.height,
                    luma_stride: align_up_to(t.width, 8),
                    chroma_stride: align_up_to(t.width.div_ceil(2), 8) * 2,
                    chroma_offset: align_up_to(t.width, 8) * t.height,
                    swap_chroma: (t.swizzle >> 12) & 1 != 0,
                    data: t.pixels.clone(),
                }
            }),
            // The guest's own mip declaration, so the uploader can read it rather than assume a
            // chain - see `GxmTexture::levels`.
            levels: t.levels,
            mip_filter: t.mip_filter,
            base_format: t.base_format,
            swizzle: t.swizzle,
            // >>> BOUNDED, AND THE BOUND IS THE POINT. This runs for every texture of every
            // draw, and an unbounded `all(|b| *b == 0)` is the per-frame byte scan this
            // renderer spent real effort deleting: the short circuit makes a texture with a
            // non-zero first byte free, but a large mostly-transparent one is walked to its
            // end, every draw. MEASURED as a phase shift on a golf title's late frames when it
            // was unbounded - the arms rendered the same content at different animation phases
            // [[vitaslop-inline-ab-moves-the-animation-phase]].
            //
            // 4 KB is as good a witness as 4 MB for the question actually being asked: a live
            // render target's guest memory is empty EVERYWHERE, not just past some offset (see
            // `GxmTexture::guest_bytes_all_zero`). The one case the bound gets wrong is a real
            // texture whose first 4 KB are zero AND whose address is also held as a render
            // target AND whose bound extent disagrees with it - and that texture was being
            // handed the stale target's pixels before any of this existed.
            guest_bytes_all_zero: t.pixels.iter().take(4096).all(|b| *b == 0),
            filter_linear,
            addr_mode_u: t.u_addr_mode,
            addr_mode_v: t.v_addr_mode,
            gamma: t.gamma != 0,
            compressed,
            // The guest's own bytes for a texture whose decode is a permutation, so the
            // uploader can do it on the GPU instead of expanding it here. `rgba` above is still
            // built lazily and is still the fallback.
            // >>> ONLY WHEN NOTHING ELSE CLAIMED THE TEXTURE, and the order is load-bearing:
            // the uploader tries `raw` BEFORE the compressed paths, so offering a block plan
            // for a texture the passthrough could take verbatim, or the GPU transcoder could
            // turn into ETC2 under budget pressure, would preempt both with a strictly worse
            // answer. `compressed` being `None` is exactly the statement "this one is going to
            // be decoded to RGBA8" - so this decides only WHERE that happens.
            raw: raw_source(t).or_else(|| if has_compressed { None } else { block_source(t) }),
        };
        // >>> THE PREVIOUS DECODE OF THIS SAME GUEST TEXTURE IS DROPPED HERE, and the choice
        // between dropping it and merely marking it for eviction was MEASURED, both ways, on
        // the same screen of the same title with a movie playing:
        //
        // | policy | evict passes/frame | re-decoded after evict | GPU uploads/frame |
        // |---|---|---|---|
        // | neither (the budget finds it) | 1.05 | 0.0 | 2.8 (2.17 MB) |
        // | marked, evicted first | 1.73 | 0.6 | 2.9 (2.22 MB) |
        // | DROPPED here | **0.00** | **0.0** | **0.6 (0.51 MB)** |
        //
        // Marking does not free the bytes, so the budget is still met every frame and an
        // eviction pass still runs - and an eviction invalidates the GPU-side entry keyed on
        // this decode, which is why upload traffic tracks it. Dropping keeps the cache the
        // size of what the guest can still ask for, and the eviction pass stops running at
        // all. The re-decode column is what says the "content comes back" worry does not
        // happen HERE, on the CPU side; the same experiment on the GPU-side view cache says
        // the opposite, and that cache marks instead (see `GxpLive::view_dead`).
        if let Some(stale) = self.decode_slots.insert(tex_slot_key(t), key) {
            if stale != key && self.decode_cache.remove(&stale).is_some() {
                let bytes = self.decode_used.remove(&stale).map_or(0, |(_, b)| b);
                self.decode_cache_bytes = self.decode_cache_bytes.saturating_sub(bytes);
                work.tex_superseded += 1;
            }
        }
        self.decode_cache.insert(key, (g.clone(), t.pixels.clone()));
        self.touch_decode(key, cost);
        g
    }

    /// Mark a decoded texture as used by the frame being built, and count it toward that
    /// frame's working set the FIRST time this frame touches it.
    ///
    /// Counting per lookup rather than per texture would inflate the learned floor by the
    /// number of draws that sample it - a shadow map bound by two hundred draws is two hundred
    /// lookups and one texture - which would then raise the budget to something the frame
    /// never actually needed resident.
    fn touch_decode(&mut self, key: u64, bytes: usize) {
        let epoch = self.decode_epoch;
        let first_this_frame = match self.decode_used.insert(key, (epoch, bytes)) {
            Some((used, _)) => used != epoch,
            None => true,
        };
        if first_this_frame {
            self.decode_frame_bytes += bytes;
        }
    }

    /// Reduce a captured scene to general draws. Each triangle-list draw is decoded
    /// into the canonical vertex layout (position, uv already divided by the draw's uv
    /// scale, color) with a triangle-LIST 32-bit index buffer, and its texture (if it
    /// samples one) decoded to linear RGBA8. Triangle list/strip/fan are all supported and
    /// expanded to a flat list (the GPU pipeline is TriangleList topology); non-triangle
    /// topologies (lines/points) are skipped, matching the software rasterizer. Getting
    /// strips right is essential: a real 3D title emits the overwhelming majority of its
    /// world/vehicle meshes as triangle STRIPS, so dropping them (as an earlier
    /// list-only build did) rendered the whole 3D world black while the 2D UI still showed.
    pub fn build(&mut self, scene: &Scene) -> RenderScene {
        // What this scene made `build` DO, folded into the process-wide tally at the end.
        // See [`BuildWork`]: the browser has no clock inside `build`, so the count IS the
        // measurement.
        let mut work = BuildWork::default();
        let mut draws = Vec::with_capacity(scene.draws.len());
        // Visible opaque depth range (post-divide c.z/c.w), accumulated across draws for
        // the GPU's linear depth normalization.
        let mut dmin = f32::INFINITY;
        let mut dmax = f32::NEG_INFINITY;
        // Diagnostic: VITASLOP_DRAW_STATS also reports each opaque draw's own visible depth
        // span, which is what the GPU's normalization has to keep separable.
        let stats = std::env::var("VITASLOP_DRAW_STATS").is_ok();
        // Will ANY draw of this scene read `depth_min`/`depth_scale`? Only a draw that is
        // rendered FIXED-FUNCTION *and* opaque *and* MVP does, or - for every draw - the
        // recompiled path under `ZFix::Range`.
        //
        // That question cannot be answered before the draws are interpreted (opaque and MVP
        // both come out of `interpret_draw`, and calling it twice would cost more than the
        // walk saves), so the range is measured in TWO passes. The main loop below walks a
        // draw's vertices only when its own fixed-function representation needs them; every
        // OTHER opaque MVP draw is remembered in `deferred` and walked afterwards, and only
        // if a reader actually turned up. The range is still a property of the whole scene -
        // every opaque MVP draw contributes exactly as before, whatever pass measures it -
        // and `min`/`max` do not care what order they see their inputs in, so the result is
        // bit-identical.
        //
        // On a frame where every draw is recompiled - which is every frame of every title
        // now - the second pass never runs at all. The earlier gate asked only whether ANY
        // draw was fixed-function, and a race scene has TWO draws with no shader payload out
        // of 461 (2D overlays, which are neither opaque nor MVP and so read nothing), so it
        // forced the walk on for the whole scene.
        let zfix_range = zfix_consumes_scene_depth_range();
        // Opaque MVP draws whose contribution the main loop did NOT measure, as
        // (draw index, layout, model-view-projection).
        let mut deferred: Vec<(usize, Layout, [f32; 16])> = Vec::new();
        // Has a reader turned up? `ZFix::Range` makes every recompiled draw one.
        let mut range_has_reader = zfix_range;
        let mut tally = DropTally { total: scene.draws.len(), ..Default::default() };
        for (di, d) in scene.draws.iter().enumerate() {
            // A list emits idx/3 triangles; a strip or fan emits idx-2.
            let tri_count = triangle_count(d);
            // >>> A LINE OR POINT LIST IS NOT A DRAW THIS RENDERER GETS TO SKIP.
            //
            // It emits no triangles, so the software rasteriser and the fixed-function
            // packing genuinely have nothing to do with it - but the RECOMPILED path draws
            // it exactly, with the guest's own shaders, once the pipeline is given the
            // matching topology (`gpu::gxm_topology`). Dropping it here is what made one
            // retail title report, every frame, `DROPPED draw N - not a triangle topology
            // ... shaders=yes (the recompiler could draw this)` - a draw the guest asked
            // for, that this renderer had everything it needed to produce, missing from the
            // frame and saying so in its own log.
            //
            // A line or point list with NO shader payload is still dropped: there is no
            // second path for it and the report is the honest answer.
            let direct = direct_topology_stride(d.primitive)
                .filter(|&n| !d.vprog.is_empty() && d.index_count as usize >= n);
            // An edge list draws by the same recompiled-only rule as lines and points, but
            // only under the encoding its own index words confirm - see
            // `PRIM_TRIANGLE_EDGES`. One that does not confirm falls through to the drop
            // report, which prints the words that refused it.
            let edge_list = d.primitive == PRIM_TRIANGLE_EDGES
                && !d.vprog.is_empty()
                && d.index_count >= 4
                && edge_list_matches_packed_reading(d);
            if tri_count == 0 && direct.is_none() && !edge_list {
                tally.topology += 1;
                tally.with_shaders += !d.vprog.is_empty() as usize;
                report_drop(DropKind::Topology, di, d, tri_count);
                if stats {
                    println!(
                        "draw {di:>3}: DROPPED - primitive {:#x} is not a triangle topology \
                         ({} indices)",
                        d.primitive, d.index_count
                    );
                }
                continue;
            }
            let interp = interpret_draw(d);
            // A position-only draw whose colour lives in the guest's shader is NOT dropped
            // when the recompiler can have it: the fixed-function packing has no colour
            // source, but the recompiled pair does. It is carried through marked
            // `shader_only`, and the renderer draws it with the real shaders or reports it
            // missing. Dropping it here is what silently erased a title's solid-colour UI
            // fills. A shader-expanded draw is still dropped: its stream holds sprite
            // RECORDS, so there is no primitive to draw by any path.
            let shader_only = interp.skip && !d.shader_expanded && !d.vprog.is_empty();
            if interp.skip && !shader_only {
                let kind = if d.shader_expanded {
                    tally.expanded += 1;
                    DropKind::Expanded
                } else {
                    tally.colorless += 1;
                    DropKind::Colorless
                };
                tally.with_shaders += !d.vprog.is_empty() as usize;
                report_drop(kind, di, d, tri_count);
                if stats {
                    println!(
                        "draw {di:>3}: DROPPED - {} (tris={tri_count}, stride={}, {} attributes, \
                         {} uniform floats, shaders={})",
                        if d.shader_expanded {
                            "shader-expanded sprite records"
                        } else {
                            "position-only: no texcoord and no vertex colour"
                        },
                        d.vertex_stride,
                        d.attributes.len(),
                        d.uniform_bank_floats(),
                        if d.vprog.is_empty() { "none" } else { "captured" },
                    );
                }
                continue;
            }
            let layout = &interp.layout;

            // The opaque (depth-tested, texel-only, tone-mapped) decision - identical to
            // the software rasterizer's `depth_test`: an MVP-space draw that also writes
            // depth. An MVP draw with depth writes DISABLED is a 2D alpha-blended overlay.
            let opaque = matches!(interp.space, Space::Mvp(_))
                && d.render_state.front_depth_write != SCE_GXM_DEPTH_WRITE_DISABLED;
            // The MVP of any MVP-space draw (for the cull test / behind-eye drop below);
            // depth-range accumulation additionally requires it be opaque.
            let mvp = match &interp.space {
                Space::Mvp(m) => Some(*m),
                _ => None,
            };
            // Cull mode: only 3D (MVP) draws have a meaningful facing; a 2D sprite's
            // winding is submission-defined, so it is never culled (matches `render_scene`).
            let cull_mode = if mvp.is_some() { d.render_state.cull_mode } else { SCE_GXM_CULL_NONE };

            // Whether this draw's FIXED-FUNCTION representation will be used at all - see
            // `gxp_only`, and the note further down.
            let fixed_function = !(self.gxp_only && !d.vprog.is_empty());
            // ...and if it will not, the ONLY thing the per-vertex walk still produces is
            // the opaque depth RANGE, which the recompiled path maps its clip depth
            // through. When that is not wanted either, the whole walk - and the index scan
            // that sizes it - is dead. MEASURED in the browser, where this shows up as
            // `build`: 22.6 ms of a 29.7 ms render, against 6.8 ms for every WebGPU call
            // of the frame put together.
            // Only `ZFix::Range` makes a RECOMPILED draw read the range, and that is known
            // before the scene starts. Everything else is deferred - see the two-pass note
            // at the top of this function. `VITASLOP_DRAW_STATS` also walks inline, because
            // it reports each draw's OWN span and the second pass does not keep those
            // apart; that only affects a diagnostic run.
            let need_depth_range = opaque && mvp.is_some() && (zfix_range || stats);
            let walk = fixed_function || need_depth_range;
            if opaque && mvp.is_some() {
                if fixed_function {
                    // This draw's own fixed-function pipeline maps its depth through the
                    // range, so the scene has a reader. (It also walks, just below, so its
                    // own contribution is measured inline.)
                    if !range_has_reader {
                        report_depth_range_reader(di, d);
                    }
                    range_has_reader = true;
                } else if !need_depth_range {
                    deferred.push((di, *layout, mvp.expect("checked just above")));
                }
            }

            // The largest index referenced, so the vertex buffer covers every index
            // (an out-of-range index decodes to a zero vertex, matching the software
            // path's clamped reads, rather than a GPU out-of-bounds fetch).
            let mut max_idx = 0usize;
            if walk {
                for i in 0..d.index_count as usize {
                    max_idx = max_idx.max(index_at(d, i));
                }
                work.indices_scanned += d.index_count as u64;
            }
            let stride = d.vertex_stride.max(1) as usize;
            let nverts = if walk { (d.vertices.len() / stride).max(max_idx + 1) } else { 0 };
            work.draws += 1;
            work.draws_fixed_function += fixed_function as u64;
            work.verts_walked += nverts as u64;

            // Screen positions of every vertex for MVP draws, so the cull test and
            // behind-eye drop below reuse one projection per vertex (not per triangle).
            // `project` applies the same Y-flip the software rasterizer uses; only the
            // winding SIGN matters here, so any positive surface size gives the identical
            // cull decision as the real target. `None` = behind the eye (w <= 0).
            // Everything below that only feeds the fixed-function representation is skipped
            // when it is dead: on a race frame that is a few hundred thousand vertices'
            // worth of buffer writes, two matrix multiplies per vertex and an index
            // expansion, all of which `encode` then steps over.
            let mut screen_pos: Vec<Option<[f32; 4]>> =
                if fixed_function && mvp.is_some() { Vec::with_capacity(nverts) } else { Vec::new() };

            let (mut draw_dmin, mut draw_dmax) = (f32::INFINITY, f32::NEG_INFINITY);
            let mut vertices =
                Vec::with_capacity(if fixed_function { nverts * GXM_VERTEX_STRIDE as usize } else { 0 });
            for i in 0..nverts {
                if !fixed_function {
                    // Only the depth range is left, and it reads the position alone. Same
                    // arithmetic as the general arm below, on the same decoded position.
                    let p = decode_vertex_pos(d, layout, i);
                    let m = mvp.expect("need_depth_range implies an MVP");
                    let c = transform(&m, p[0], p[1], p[2]);
                    if c[3] > 1e-4 {
                        let (nx, ny, depth) = (c[0] / c[3], c[1] / c[3], -1.0 / c[3]);
                        if nx.abs() <= 1.0 && ny.abs() <= 1.0 && depth.is_finite() {
                            dmin = dmin.min(depth);
                            dmax = dmax.max(depth);
                            draw_dmin = draw_dmin.min(depth);
                            draw_dmax = draw_dmax.max(depth);
                        }
                    }
                    continue;
                }
                let v = decode_vertex(d, layout, i);
                if fixed_function {
                    vertices.extend_from_slice(&v.pos[0].to_le_bytes());
                    vertices.extend_from_slice(&v.pos[1].to_le_bytes());
                    vertices.extend_from_slice(&v.pos[2].to_le_bytes());
                    // Fold the uv divisor in here (constant per draw, so pre-dividing per
                    // vertex is identical to dividing the interpolated coord).
                    vertices.extend_from_slice(&(v.uv[0] / interp.uv_div[0]).to_le_bytes());
                    vertices.extend_from_slice(&(v.uv[1] / interp.uv_div[1]).to_le_bytes());
                    vertices.extend_from_slice(&v.color);
                    // World-space normal for the opaque lighting term, baked here (object normal
                    // through the draw's model-to-world matrix) so the GPU shader uses it directly
                    // and matches the software rasterizer's `world_normal` exactly. A mesh with no
                    // normal yields `[0,1,0]` (up), the same fallback the software path uses.
                    let wn = world_normal(v.normal, &d.world);
                    vertices.extend_from_slice(&wn[0].to_le_bytes());
                    vertices.extend_from_slice(&wn[1].to_le_bytes());
                    vertices.extend_from_slice(&wn[2].to_le_bytes());
                }
                if fixed_function && mvp.is_some() {
                    // Cull only needs the winding SIGN, so any uniform scale works; ssaa is 1
                    // here (the GPU applies supersampling itself via an enlarged render target).
                    screen_pos.push(project(&v, &interp.space, 4096, 4096, 1.0));
                }
                // Accumulate the visible opaque depth range (post-divide c.z/c.w over
                // on-screen vertices) so the GPU can linearly normalize depth into [0,1]
                // at full precision - see `RenderScene::depth_min`/`depth_scale`.
                if opaque {
                    if let Some(m) = mvp {
                        let c = transform(&m, v.pos[0], v.pos[1], v.pos[2]);
                        if c[3] > 1e-4 {
                            let (nx, ny, depth) = (c[0] / c[3], c[1] / c[3], -1.0 / c[3]);
                            if nx.abs() <= 1.0 && ny.abs() <= 1.0 && depth.is_finite() {
                                dmin = dmin.min(depth);
                                dmax = dmax.max(depth);
                                draw_dmin = draw_dmin.min(depth);
                                draw_dmax = draw_dmax.max(depth);
                            }
                        }
                    }
                }
            }

            // Expand the topology into a flat triangle-LIST index buffer with winding
            // NORMALIZED (`tri_indices` un-flips a strip's odd triangles), CPU-culling back
            // faces and dropping behind-eye triangles exactly as the software rasterizer
            // does. Doing the cull here (not via GPU pipeline state) keeps the GPU a
            // pixel-faithful twin with one cull-free pipeline and no per-draw facing state.
            let mut indices = Vec::with_capacity(if fixed_function { tri_count * 3 * 4 } else { 0 });
            for t in 0..tri_count {
                if !fixed_function {
                    break;
                }
                let vs = tri_indices(d, t);
                if mvp.is_some() {
                    let s: [[f32; 4]; 3] = match [screen_pos[vs[0]], screen_pos[vs[1]], screen_pos[vs[2]]] {
                        [Some(a), Some(b), Some(c)] => [a, b, c],
                        _ => continue, // a vertex is behind the eye - drop the triangle
                    };
                    if cull_mode != SCE_GXM_CULL_NONE && cull_backface(edge(&s[0], &s[1], &s[2]), cull_mode) {
                        continue;
                    }
                }
                for k in vs {
                    indices.extend_from_slice(&(k as u32).to_le_bytes());
                }
            }
            let index_count = (indices.len() / 4) as u32;
            if stats && opaque && draw_dmax >= draw_dmin {
                println!(
                    "draw {di:>3}: tris={tri_count:<5} depth [{draw_dmin:.9}, {draw_dmax:.9}] func={:#x} write={:#x} cull={:#x}",
                    d.render_state.front_depth_func,
                    d.render_state.front_depth_write,
                    d.render_state.cull_mode
                );
            }

            let texture = if interp.textured && fixed_function {
                d.albedo().map(|t| self.texture(t, &mut work))
            } else {
                None
            };

            // When the runtime captured the raw shader blobs (recompiler path enabled),
            // attach everything the GXP->WGSL recompiler needs to draw this call with the
            // guest's real shaders. The renderer links + caches a pipeline and falls back to
            // the fixed-function fields above on any link/format error. We carry the RAW guest
            // vertex/index buffers (not the culled canonical ones) so the recompiled pipeline
            // does its own attribute fetch + facing cull.
            let gxp = if !d.vprog.is_empty() {
                // All three lists are DERIVED ONCE per distinct source and shared thereafter -
                // see `gxp_attributes` and `gxp_textures`. The capture already hands every
                // draw with the same bindings one `Arc`; rebuilding a `Vec` from it per draw
                // threw that sharing away one layer down.
                let attributes = self.gxp_attributes(&d.attributes);
                let textures = self.gxp_textures(&d.textures, &mut work);
                // The VERTEX stage's own bindings, uploaded the same way. A vertex program that
                // samples builds its geometry from the fetch, so these decide whether the draw
                // has a mesh at all.
                let vertex_textures = self.gxp_textures(&d.vertex_textures, &mut work);
                // Expand the guest topology into a flat, winding-normalized triangle-LIST u32
                // index buffer (NO CPU cull - the recompiled pipeline culls on the GPU via the
                // guest cull mode, using its own real-shader projection). Indexes into the RAW
                // guest vertex stream `d.vertices`.
                let ikey = IndexKey {
                    buffer: d.indices.as_ptr() as usize,
                    len: d.indices.len(),
                    index_count: d.index_count,
                    primitive: d.primitive,
                    index_format: d.index_format,
                };
                let gxp_indices = match self.index_cache.get(&ikey) {
                    Some((cached, _)) => {
                        work.index_expand_cached += 1;
                        cached.clone()
                    }
                    None => {
                        if self.index_cache.len() >= INDEX_CACHE_CAP {
                            work.index_cache_clears += 1;
                            self.index_cache.clear();
                        }
                        let mut out = Vec::with_capacity(tri_count * 3 * 4);
                        match direct {
                            // A line or point list needs no expansion at all - the guest's
                            // own index order IS the primitive order, and the pipeline is
                            // built with the matching topology. Widened to u32 like every
                            // other stream so one index format serves the whole frame, and
                            // TRUNCATED to whole primitives: wgpu draws what the count says,
                            // and a trailing half-line would read a vertex the guest did not
                            // name.
                            Some(n) => {
                                let whole = (d.index_count as usize / n) * n;
                                for i in 0..whole {
                                    out.extend_from_slice(&(index_at(d, i) as u32).to_le_bytes());
                                }
                            }
                            // An edge list: groups of four words - three vertex indices
                            // and a flags word - expanded into the LINE segments the flags
                            // enable. See `PRIM_TRIANGLE_EDGES` for the measurement that
                            // established this encoding; `edge_list` above already
                            // confirmed it holds for this draw. Truncated to whole groups
                            // for the same reason `Some(n)` truncates to whole primitives.
                            None if edge_list => {
                                for g in 0..d.index_count as usize / 4 {
                                    let [i0, i1, i2] = [
                                        index_at(d, g * 4) as u32,
                                        index_at(d, g * 4 + 1) as u32,
                                        index_at(d, g * 4 + 2) as u32,
                                    ];
                                    let flags = index_at(d, g * 4 + 3) as u32;
                                    for (bit, a, b) in
                                        [(0x100, i0, i1), (0x200, i1, i2), (0x400, i2, i0)]
                                    {
                                        if flags & bit != 0 {
                                            out.extend_from_slice(&a.to_le_bytes());
                                            out.extend_from_slice(&b.to_le_bytes());
                                        }
                                    }
                                }
                            }
                            None => {
                                for t in 0..tri_count {
                                    for k in tri_indices(d, t) {
                                        out.extend_from_slice(&(k as u32).to_le_bytes());
                                    }
                                }
                            }
                        }
                        let out: Arc<[u8]> = out.into();
                        work.index_expanded += 1;
                        work.gxp_index_bytes += out.len() as u64;
                        // The SOURCE buffer rides along, and it is not decoration: the key is its
                        // address, and a dropped buffer's address can be reused by an unrelated
                        // mesh. Holding it is what stops that entry from serving someone else's
                        // triangles.
                        self.index_cache.insert(ikey, (out.clone(), d.indices.clone()));
                        out
                    }
                };
                let gxp_index_count = (gxp_indices.len() / 4) as u32;
                work.gxp_vertex_bytes += d.vertices.len() as u64;
                work.gxp_sa_bytes += (d.vert_sa.len() + d.frag_sa.len()) as u64;
                // Diagnostic (`VITASLOP_GXP_CAPSULE`): the one place a finished `Draw` and its
                // guest programs are both in hand, which is what a capsule needs. Free when the
                // knob is unset - see `maybe_capture`.
                crate::capsule::maybe_capture(d);
                Some(vitaslop_platform::gpu::GxpRecompile {
                    vprog: d.vprog.clone(),
                    fprog: d.fprog.clone(),
                    vert_sa: d.vert_sa.clone(),
                    frag_sa: d.frag_sa.clone(),
                    frag_sa_addr: d.frag_sa_addr,
                    mem_windows: d.mem_windows.clone(),
                    vertices: d.vertices.clone(),
                    vertex_stride: d.vertex_stride,
                    attributes,
                    indices: gxp_indices,
                    index_count: gxp_index_count,
                    index_u32: true,
                    primitive: d.primitive,
                    textures,
                    vertex_textures,
                    depth_write: d.render_state.front_depth_write != SCE_GXM_DEPTH_WRITE_DISABLED,
                    depth_func: d.render_state.front_depth_func,
                    depth_bias: (
                        d.render_state.front_depth_bias_factor,
                        d.render_state.front_depth_bias_units,
                    ),
                    cull_mode: d.render_state.cull_mode,
                    fragment_program_enabled: d.render_state.front_fragment_program_enable
                        != SCE_GXM_FRAGMENT_PROGRAM_DISABLED,
                    fprog_header: d.fragment_program_header,
                    blend: !opaque,
                    blend_state: [
                        d.blend.color_mask,
                        d.blend.color_func,
                        d.blend.alpha_func,
                        d.blend.color_src,
                        d.blend.color_dst,
                        d.blend.alpha_src,
                        d.blend.alpha_dst,
                    ],
                    viewport: d.render_state.viewport,
                })
            } else {
                None
            };

            draws.push(GxmDraw {
                space: to_draw_space(&interp.space),
                vertices,
                indices,
                index_count,
                texture,
                opaque,
                exposure: d.exposure,
                material: vitaslop_platform::gpu::GxmMaterial {
                    tint: d.material.tint,
                    light_dir: d.material.light_dir,
                    light_col: d.material.light_col,
                    ambient: d.material.ambient,
                },
                gxp,
                shader_only,
                region_clip: vitaslop_platform::gpu::RegionClip {
                    mode: d.render_state.region_clip_mode,
                    rect: d.render_state.region_clip,
                },
            });
        }
        // PASS TWO: a reader turned up, so the opaque MVP draws the main loop stepped over
        // still have to contribute. On a fully-recompiled scene `range_has_reader` is false
        // and this is skipped entirely, which is the whole point of deferring it.
        if range_has_reader {
            for (di, layout, m) in deferred {
                let d = &scene.draws[di];
                let mut max_idx = 0usize;
                for i in 0..d.index_count as usize {
                    max_idx = max_idx.max(index_at(d, i));
                }
                let stride = d.vertex_stride.max(1) as usize;
                let nverts = (d.vertices.len() / stride).max(max_idx + 1);
                work.indices_scanned += d.index_count as u64;
                work.verts_deferred += nverts as u64;
                for i in 0..nverts {
                    let p = decode_vertex_pos(d, &layout, i);
                    let c = transform(&m, p[0], p[1], p[2]);
                    if c[3] > 1e-4 {
                        let (nx, ny, depth) = (c[0] / c[3], c[1] / c[3], -1.0 / c[3]);
                        if nx.abs() <= 1.0 && ny.abs() <= 1.0 && depth.is_finite() {
                            dmin = dmin.min(depth);
                            dmax = dmax.max(depth);
                        }
                    }
                }
            }
        }
        // Linear depth-normalization params: map the visible opaque depth range to [0,1].
        // A degenerate range (no opaque geometry, or a single coplanar depth) yields
        // scale 0, so every opaque fragment maps to depth 0 (submission order decides, as
        // the software oracle's +INF-clear Less test does among equal depths).
        let (depth_min, depth_scale) = if dmax > dmin {
            (dmin, 1.0 / (dmax - dmin))
        } else {
            (0.0, 0.0)
        };
        BUILD_WORK.lock().unwrap().add(&work);
        tally.report_if_total(&mut self.last_empty);
        // Carry where this scene draws to, so a renderer can keep the result addressable
        // for a later pass that samples it (see `RttTarget`).
        let target = scene.color.map(|c| vitaslop_platform::gpu::RttTarget {
            data_addr: c.data_addr,
            width: c.width,
            height: c.height,
            gamma: c.gamma != 0,
            // `SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE` == 1. Anything else is NONE as far
            // as rasterisation goes - the enum's other values do not ask for a finer raster.
            msaa_downscale: c.scale_mode == 1,
            // How many samples per pixel the RENDER TARGET was created with. The scale mode
            // above says this surface stores a resolved image; this says what was resolved
            // into it. See `gpu::gxm_sample_count`.
            multisample: scene.multisample,
        });
        // Where this scene's DEPTH lands, for the same reason as `target` above: a later pass
        // that samples this depth names exactly this address.
        let depth_addr = scene.depth.map(|d| d.depth_addr).unwrap_or(0);
        // A depth-only pass has no colour surface to take an extent from, and a
        // `SceGxmDepthStencilSurface` carries none. Take the draws' viewport, which is the
        // guest's own statement of the pixel region it rasterises into (GXM's viewport is
        // offset/scale in pixels, so the width is `2*|xScale|`) - and only when the viewport
        // is actually in effect, because with it disabled the transform is the render
        // target's and says nothing. See `RenderScene::depth_extent`.
        //
        // >>> IT IS TAKEN FROM EVERY DRAW, NOT THE FIRST, AND THE AGREEMENT IS THE POINT.
        // Reading only `draws.first()` makes the extent a GUESS that happens to be right
        // whenever the pass is uniform - which is every pass anyone has looked at, so the
        // reporting had to hedge forever and say "this size is DERIVED" on every run. A derived
        // value that every contributing draw agrees on is not a guess, it is a measurement: if
        // all the viewport-enabled draws in the pass name one rectangle, that rectangle IS the
        // guest's statement of the region, and there is nothing left to warn about. If they
        // disagree there is a real ambiguity, and THAT is worth saying out loud - so the caller
        // is told which of the two it got instead of being told the size is derived either way.
        let depth_extent = if target.is_none() && depth_addr != 0 {
            let mut seen: Option<(u32, u32)> = None;
            let mut agreed = true;
            for d in scene.draws.iter().filter(|d| d.render_state.viewport_enable == 0) {
                let v = d.render_state.viewport;
                let e = ((2.0 * v[1].abs()) as u32, (2.0 * v[3].abs()) as u32);
                if e.0 <= 1 || e.1 <= 1 {
                    continue;
                }
                match seen {
                    None => seen = Some(e),
                    // Keep the LARGEST, so an ambiguous pass still gets a target big enough for
                    // every draw in it rather than one that clips the others.
                    Some(prev) if prev != e => {
                        agreed = false;
                        seen = Some((prev.0.max(e.0), prev.1.max(e.1)));
                    }
                    Some(_) => {}
                }
            }
            seen.map(|(w, h)| (w, h, agreed))
        } else {
            None
        };
        let depth_extent_ambiguous = matches!(depth_extent, Some((_, _, false)));
        let depth_extent = depth_extent.map(|(w, h, _)| (w, h));
        RenderScene {
            // Carried through untouched: the builder turns DRAWS into render state, and a pair
            // the patcher named has no draw yet - that is the whole point of it arriving early.
            precompile: scene.precompile.clone(),
            draws,
            target,
            depth_min,
            depth_scale,
            depth_addr,
            depth_extent,
            depth_extent_ambiguous,
        }
    }
}

#[cfg(test)]
mod geometry_tests {
    //! Content-free tests for the geometry-stage logic the render output depends on:
    //! triangle-strip winding normalization, per-`SceGxmCullMode` back-face culling, and
    //! the depth-compare function. No game data. These pin the decisions the software
    //! rasterizer and the GPU builder ([`RenderSceneBuilder::build`]) both make, so the two
    //! paths stay in lockstep.
    use super::*;
    use crate::capture::RenderState;

    fn strip_draw(indices: &[u16]) -> Draw {
        Draw {
            fragment_program_header: 0,
            primitive: PRIM_TRIANGLE_STRIP,
            index_format: 0,
            index_count: indices.len() as u32,
            vertices: Arc::from(&[][..]),
            vertex_stride: 1,
            attributes: vec![].into(),
            indices: indices.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>().into(),
            uniforms: vec![],
            textures: vec![].into(),
            vertex_textures: std::sync::Arc::from(&[][..]),
            render_state: std::sync::Arc::new(RenderState::default()),
            blend: crate::capture::BlendState::default(),
            exposure: 1.0,
            material: crate::capture::FragmentMaterial::default(),
            world: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vprog: crate::capture::no_program(),
            fprog: crate::capture::no_program(),
            vert_sa: std::sync::Arc::from(&[][..]),
            frag_sa: std::sync::Arc::from(&[][..]),
            frag_sa_addr: 0,
            mem_windows: Vec::new(),
            shader_expanded: false,
        }
    }

    /// A minimal MVP triangle-list draw placed at `world`, with a position-only vertex
    /// stream and a per-vertex colour (so `interpret_draw` does not skip it).
    fn located_draw(world: [f32; 3], verts: &[[f32; 3]], mvp: [f32; 16]) -> Draw {
        let mut d = strip_draw(&[0, 1, 2]);
        d.primitive = PRIM_TRIANGLES;
        d.vertex_stride = 16;
        d.vertices = verts
            .iter()
            .flat_map(|p| {
                let mut b: Vec<u8> = p.iter().flat_map(|c| c.to_le_bytes()).collect();
                b.extend_from_slice(&[255, 255, 255, 255]); // colour, at offset 12
                b
            })
            .collect();
        d.attributes = vec![
            crate::capture::VertexAttribute {
                stream_index: 0,
                offset: 0,
                format: 3, // F32
                component_count: 3,
                reg_index: 0,
            },
            crate::capture::VertexAttribute {
                stream_index: 0,
                offset: 12,
                format: 0, // U8N
                component_count: 4,
                reg_index: 1,
            },
        ].into();
        d.uniforms = mvp.to_vec();
        d.world[12] = world[0];
        d.world[13] = world[1];
        d.world[14] = world[2];
        d
    }

    #[test]
    fn locate_groups_draws_by_placement_and_projects_them() {
        // An MVP that just translates z into w, so a vertex at z=1 lands at the centre
        // of the raster: the projection is exercised without inventing a camera.
        let mut mvp = [0f32; 16];
        mvp[0] = 1.0;
        mvp[5] = 1.0;
        mvp[10] = 1.0;
        mvp[11] = 1.0; // w = z
        let tri = [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
        let scene = Scene {
            precompile: Default::default(),
            color: None,
            depth: None,
            multisample: 0,
            draws: vec![
                located_draw([10.0, 0.0, 5.0], &tri, mvp),
                // Same placement as the first: one object drawn in two passes.
                located_draw([10.0, 0.0, 5.0], &tri, mvp),
                located_draw([-3.0, 1.0, 2.0], &tri, mvp),
            ],
        };
        let found = locate_scene(&scene, 100, 100);
        assert_eq!(found.len(), 2, "draws sharing a placement are ONE object");
        assert_eq!(found[0].draws, vec![0, 1]);
        assert_eq!(found[0].world, [10.0, 0.0, 5.0]);
        assert_eq!(found[1].world, [-3.0, 1.0, 2.0]);
        // Ordering is the scene's own submission order, so two reports line up.
        assert!(found[0].draws[0] < found[1].draws[0]);
        // x=0,y=0 with w=z projects to the centre of the raster.
        assert_eq!(found[0].centroid, Some([50.0, 50.0]));
        assert_eq!(found[0].distance, Some(1.0));
    }

    #[test]
    fn locate_reports_object_heading_in_pad_bearing_convention() {
        let mut mvp = [0f32; 16];
        mvp[0] = 1.0;
        mvp[5] = 1.0;
        mvp[10] = 1.0;
        mvp[11] = 1.0;
        let tri = [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];

        // Identity rotation: local +X lies along world +X (bearing 0) and local +Z
        // along world +Z (bearing -90, since bearings increase toward world -Z). Same
        // convention as the `lang=` stick directive, so a commanded bearing and a
        // measured heading are directly comparable numbers.
        let d = located_draw([0.0, 0.0, 0.0], &tri, mvp);
        let found = locate_scene(&Scene { precompile: Default::default(), color: None, depth: None, multisample: 0, draws:vec![d.clone()] }, 100, 100);
        let h = found[0].heading.expect("an identity rotation has a heading");
        assert!((h[0] - 0.0).abs() < 1e-3, "local +X is bearing 0, got {}", h[0]);
        assert!((h[1] + 90.0).abs() < 1e-3, "local +Z is bearing -90, got {}", h[1]);

        // Rotate 90 degrees so local +X points along world -Z: bearing 90.
        let mut turned = d.clone();
        turned.world[0] = 0.0;
        turned.world[2] = -1.0;
        turned.world[8] = 1.0;
        turned.world[10] = 0.0;
        let found = locate_scene(&Scene { precompile: Default::default(), color: None, depth: None, multisample: 0, draws:vec![turned] }, 100, 100);
        let h = found[0].heading.unwrap();
        assert!((h[0] - 90.0).abs() < 1e-3, "expected bearing 90, got {}", h[0]);

        // A world matrix with no in-plane rotation at all reports no heading rather
        // than a fabricated zero.
        let mut flat = d;
        flat.world[0] = 0.0;
        flat.world[2] = 0.0;
        let found = locate_scene(&Scene { precompile: Default::default(), color: None, depth: None, multisample: 0, draws:vec![flat] }, 100, 100);
        assert_eq!(found[0].heading, None);
    }

    #[test]
    fn locate_ids_track_geometry_not_draw_index() {
        // The identity has to survive the draw list being rebuilt, because that is what
        // happens every frame: a delta matched on draw index reports huge motion for a
        // world that barely moved.
        let mut mvp = [0f32; 16];
        mvp[0] = 1.0;
        mvp[5] = 1.0;
        mvp[10] = 1.0;
        mvp[11] = 1.0;
        let car = [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]];
        let other = [[0.0, 0.0, 2.0], [5.0, 0.0, 2.0], [0.0, 5.0, 2.0]];

        let before = Scene { precompile: Default::default(), color: None, depth: None, multisample: 0, draws:vec![located_draw([0.0, 0.0, 0.0], &car, mvp)] };
        // Next frame: something new is submitted first, and the car has moved.
        let after = Scene {
            precompile: Default::default(),
            color: None,
            depth: None,
            multisample: 0,
            draws: vec![
                located_draw([99.0, 0.0, 0.0], &other, mvp),
                located_draw([1.0, 0.0, 0.0], &car, mvp),
            ],
        };
        let a = locate_scene(&before, 100, 100);
        let b = locate_scene(&after, 100, 100);
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 2);
        // The car is at draw 0 in one frame and draw 1 in the next, but keeps its id.
        assert_eq!(a[0].draws, vec![0]);
        assert_eq!(b[1].draws, vec![1]);
        assert_eq!(a[0].id, b[1].id, "same geometry, same identity");
        assert_ne!(a[0].id, b[0].id, "different geometry, different identity");
    }

    #[test]
    fn strip_winding_is_normalized_per_triangle() {
        // A strip's triangles alternate winding; `tri_indices` un-flips the odd ones so
        // every triangle presents the facing a triangle-list triangle would - the premise
        // the cull test relies on.
        let d = strip_draw(&[0, 1, 2, 3, 4]);
        assert_eq!(tri_indices(&d, 0), [0, 1, 2]); // even: as-is
        assert_eq!(tri_indices(&d, 1), [1, 3, 2]); // odd: [1,2,3] with last two swapped
        assert_eq!(tri_indices(&d, 2), [2, 3, 4]); // even: as-is
    }

    #[test]
    fn list_and_fan_indices() {
        let mut d = strip_draw(&[0, 1, 2, 3, 4, 5]);
        d.primitive = PRIM_TRIANGLES;
        assert_eq!(tri_indices(&d, 0), [0, 1, 2]);
        assert_eq!(tri_indices(&d, 1), [3, 4, 5]);
        d.primitive = PRIM_TRIANGLE_FAN;
        assert_eq!(tri_indices(&d, 0), [0, 1, 2]);
        assert_eq!(tri_indices(&d, 1), [0, 2, 3]);
        assert_eq!(tri_indices(&d, 2), [0, 3, 4]);
    }

    #[test]
    fn triangle_count_by_topology() {
        let mut d = strip_draw(&[0, 1, 2, 3, 4]); // 5 indices
        assert_eq!(triangle_count(&d), 3); // strip / fan: idx - 2
        d.primitive = PRIM_TRIANGLE_FAN;
        assert_eq!(triangle_count(&d), 3);
        d.primitive = PRIM_TRIANGLES;
        d.index_count = 6;
        assert_eq!(triangle_count(&d), 2); // list: idx / 3
        d.primitive = PRIM_LINES;
        assert_eq!(triangle_count(&d), 0, "a line list emits no TRIANGLES");
    }

    /// A line or point list emits no triangles, and used to be dropped for it. It is now
    /// carried to the recompiled path with the guest's OWN index order and the matching
    /// wgpu topology - see `direct_topology_stride` and `gpu::gxm_topology`.
    ///
    /// The truncation half is the one that would fail silently: wgpu draws exactly the
    /// index count it is given, so a trailing half-line would make it read a vertex the
    /// guest never named.
    #[test]
    fn a_line_or_point_list_is_carried_through_at_its_own_stride() {
        assert_eq!(direct_topology_stride(PRIM_LINES), Some(2));
        assert_eq!(direct_topology_stride(PRIM_POINTS), Some(1));
        for p in [PRIM_TRIANGLES, PRIM_TRIANGLE_STRIP, PRIM_TRIANGLE_FAN] {
            assert_eq!(
                direct_topology_stride(p),
                None,
                "a triangle topology is EXPANDED, not passed through"
            );
        }
        // Five indices of a line list are two whole lines and a leftover; the leftover is
        // dropped rather than read as half of a line that has no second vertex.
        let d = strip_draw(&[0, 1, 2, 3, 4]);
        let n = direct_topology_stride(PRIM_LINES).expect("lines have a stride");
        assert_eq!((d.index_count as usize / n) * n, 4);
    }

    /// The edge-list encoding is groups of four index words - three vertex indices and a
    /// `SceGxmEdgeEnableFlags` word (see `PRIM_TRIANGLE_EDGES` for the measurement). The
    /// validity test is what keeps a buffer that CONTRADICTS that reading from being drawn
    /// under it.
    #[test]
    fn an_edge_list_is_validated_against_the_packed_reading() {
        // The measured buffer shape: [i0, i1, i2, flags] x groups, flags in {0x100..0x700}.
        let mut d = strip_draw(&[0, 1, 2, 0x700, 3, 4, 5, 0x500, 6, 7, 8, 0x300]);
        d.primitive = PRIM_TRIANGLE_EDGES;
        assert!(edge_list_matches_packed_reading(&d));
        // A fourth word with a bit outside the three flag bits refuses the reading.
        let mut bad = strip_draw(&[0, 1, 2, 0x701]);
        bad.primitive = PRIM_TRIANGLE_EDGES;
        assert!(!edge_list_matches_packed_reading(&bad));
        // Flags of ZERO are a valid (empty) triangle, not a refusal.
        let mut none = strip_draw(&[0, 1, 2, 0]);
        none.primitive = PRIM_TRIANGLE_EDGES;
        assert!(edge_list_matches_packed_reading(&none));
        // A trailing partial group is outside the whole-group count and does not refuse.
        let mut tail = strip_draw(&[0, 1, 2, 0x700, 9, 9]);
        tail.primitive = PRIM_TRIANGLE_EDGES;
        assert!(edge_list_matches_packed_reading(&tail));
    }

    /// A screen triangle (Y-down, as `project` emits) with a chosen facing. `[a,b,c]` here
    /// has POSITIVE `edge` area (the front face by the convention pinned in `cull_backface`
    /// and validated against the title's ground plane); swapping the last two flips it.
    fn tri(front: bool) -> [[f32; 4]; 3] {
        let a = [0.0, 0.0, 0.0, 1.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let c = [0.0, 1.0, 0.0, 1.0];
        if front { [a, b, c] } else { [a, c, b] }
    }

    #[test]
    fn cull_mode_discards_the_back_face_only() {
        let (f, b) = (tri(true), tri(false));
        let (af, ab) = (edge(&f[0], &f[1], &f[2]), edge(&b[0], &b[1], &b[2]));
        assert!(af > 0.0 && ab < 0.0, "front area positive, back negative: {af} {ab}");
        // CCW-cull discards the back face (area < 0), keeps the front.
        assert!(!cull_backface(af, SCE_GXM_CULL_CCW));
        assert!(cull_backface(ab, SCE_GXM_CULL_CCW));
        // CW-cull is the mirror image.
        assert!(cull_backface(af, SCE_GXM_CULL_CW));
        assert!(!cull_backface(ab, SCE_GXM_CULL_CW));
        // NONE keeps both faces (the double-sided body panels).
        assert!(!cull_backface(af, SCE_GXM_CULL_NONE));
        assert!(!cull_backface(ab, SCE_GXM_CULL_NONE));
    }

    #[test]
    fn depth_func_semantics() {
        const NEVER: u32 = 0x0000_0000;
        const LESS: u32 = 0x0040_0000;
        const EQUAL: u32 = 0x0080_0000;
        const LESS_EQUAL: u32 = 0x00C0_0000;
        const GREATER: u32 = 0x0100_0000;
        const ALWAYS: u32 = 0x01C0_0000;
        // LESS_EQUAL (GXM default + this title's opaque draws): nearer passes AND a
        // coincident later face ties and repaints.
        assert!(depth_passes(4.0, 5.0, LESS_EQUAL));
        assert!(depth_passes(5.0, 5.0, LESS_EQUAL));
        assert!(!depth_passes(6.0, 5.0, LESS_EQUAL));
        // LESS is strict: a coincident face does NOT repaint (first-drawn wins).
        assert!(depth_passes(4.0, 5.0, LESS));
        assert!(!depth_passes(5.0, 5.0, LESS));
        // The rest of the enum reproduces rather than collapsing to LESS.
        assert!(depth_passes(6.0, 5.0, GREATER) && !depth_passes(4.0, 5.0, GREATER));
        assert!(depth_passes(5.0, 5.0, EQUAL) && !depth_passes(4.0, 5.0, EQUAL));
        assert!(depth_passes(9.0, 5.0, ALWAYS) && !depth_passes(9.0, 5.0, NEVER));
        // The +inf-cleared buffer: any finite fragment passes LESS_EQUAL.
        assert!(depth_passes(100.0, f32::INFINITY, LESS_EQUAL));
    }

    /// [`project`] must resolve visibility from the projected view distance `w`, not the clip
    /// `z` - see its doc comment for the measurement that established this. The two properties
    /// that matter are pinned here: a nearer surface must produce a SMALLER depth (so the
    /// guest's LESS/LESS_EQUAL func means "nearer wins"), and the clip `z` must not influence
    /// it at all (this title emits a clip `z` that is a combination of the x/y/w rows, so
    /// `z/w` is constant per screen position and carries no depth).
    #[test]
    fn depth_comes_from_w_not_clip_z() {
        // A projection whose z row is a copy of the w row plus an x/y mix - the degenerate
        // shape this title's vertex programs emit, where z/w depends only on screen position.
        // Column-major, as `transform` reads it: m[2],m[6],m[10],m[14] is the z column.
        let mut m = [0.0f32; 16];
        m[0] = 1.0; // x = X
        m[5] = 1.0; // y = Y
        m[11] = 1.0; // w = Z  (view distance)
        m[2] = 0.5; // z = 0.5*X + 0.25*Y + Z  -> z/w = 0.5*(x/w) + 0.25*(y/w) + 1
        m[6] = 0.25;
        m[10] = 1.0;
        let space = Space::Mvp(m);
        let at = |x: f32, y: f32, z: f32| {
            let v = Vertex { pos: [x, y, z], uv: [0.0; 2], color: [255; 4], normal: [0.0; 3] };
            project(&v, &space, 100, 100, 1.0).expect("in front of the eye")
        };

        // Same screen position (x/w, y/w equal), different distances.
        let near = at(2.0, 4.0, 10.0);
        let far = at(4.0, 8.0, 20.0);
        assert!((near[0] - far[0]).abs() < 1e-5 && (near[1] - far[1]).abs() < 1e-5, "same pixel");
        assert!(near[2] < far[2], "nearer must be smaller: {} vs {}", near[2], far[2]);
        assert!(depth_passes(near[2], far[2], 0x00C0_0000), "nearer wins under LESS_EQUAL");

        // The clip z of both is identical (the degenerate case), so a z/w depth would tie and
        // let submission order decide - which is exactly the failure this model removes.
        let zw = |x: f32, y: f32, z: f32| {
            let c = transform(&m, x, y, z);
            c[2] / c[3]
        };
        assert!(
            (zw(2.0, 4.0, 10.0) - zw(4.0, 8.0, 20.0)).abs() < 1e-6,
            "the fixture's clip z/w carries no depth, by construction"
        );

        // Depth is screen-linear (affine in 1/w), so it interpolates exactly across a triangle:
        // the midpoint in screen space of two vertices has the mean of their depths.
        let mid = at(3.0, 6.0, 15.0);
        let half = 0.5 * (1.0 / 10.0 + 1.0 / 20.0);
        assert!((mid[3] - 1.0 / 15.0).abs() < 1e-6);
        assert!((-half - (near[2] + far[2]) / 2.0).abs() < 1e-6);
    }

    // --- the top-down map, and the travelling-origin correction ---------------------

    /// An axis-aligned quad on the XZ plane at height `y`, spanning `[x0,x1] x [z0,z1]`,
    /// as a two-triangle list whose vertices are already in world space (identity world
    /// matrix). `depth_write` off marks it an overlay - a skydome, in the case that
    /// matters.
    fn ground_quad(y: f32, x0: f32, z0: f32, x1: f32, z1: f32, depth_write: bool) -> Draw {
        let mut d = located_draw(
            [0.0, 0.0, 0.0],
            &[[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]],
            {
                let mut m = [0f32; 16];
                m[0] = 1.0;
                m[5] = 1.0;
                m[10] = 1.0;
                m[11] = 1.0;
                m
            },
        );
        d.primitive = PRIM_TRIANGLES;
        d.index_count = 6;
        d.indices = [0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect();
        // The REAL GXM attribute format codes, so `layout_of` recognizes the position and
        // the colour and `interpret_draw` therefore does not skip the draw as having no
        // colour source. (`located_draw` predates the map and gets away with placeholder
        // codes because `locate_scene` never consults `DrawInterp::skip`.)
        // Through a fresh list rather than in place: the field is a shared `Arc` now (see
        // `capture::Draw::attributes`), which is exactly the point - a draw does not own its
        // layout, the vertex program does.
        let mut attrs = d.attributes.to_vec();
        attrs[0].format = FORMAT_F32;
        attrs[1].format = FORMAT_U8N;
        d.attributes = attrs.into();
        if !depth_write {
            // The state is shared now, so a test that wants a variant makes its own.
            let mut rs = *d.render_state;
            rs.front_depth_write = SCE_GXM_DEPTH_WRITE_DISABLED;
            d.render_state = std::sync::Arc::new(rs);
        }
        d
    }

    fn square_view(extent: [f32; 4], n: u32) -> MapView {
        MapView { extent, width: n, height: n }
    }

    #[test]
    fn map_view_round_trips_and_puts_minus_z_up_the_image() {
        let v = square_view([-100.0, -50.0, 100.0, 150.0], 200);
        // Round trip: a pixel centre maps to a world position and back.
        for (px, py) in [(0.0, 0.0), (57.0, 133.0), (199.0, 199.0)] {
            let w = v.world_of(px, py);
            let back = v.pixel_of(w[0], w[1]);
            assert!((back[0] - px).abs() < 1e-3 && (back[1] - py).abs() < 1e-3, "{px},{py}");
        }
        // Orientation, which is the part a caller cannot check by eye without a title in
        // front of it: +X is right, and world -Z is UP the image (smaller y).
        let origin = v.pixel_of(0.0, 0.0);
        assert!(v.pixel_of(50.0, 0.0)[0] > origin[0], "+X goes right");
        assert!(v.pixel_of(0.0, -40.0)[1] < origin[1], "-Z goes up the image");
        // Scale is world units per pixel on each axis, independently.
        assert!((v.scale()[0] - 1.0).abs() < 1e-6 && (v.scale()[1] - 1.0).abs() < 1e-6);
    }

    /// The whole point of [`origin_drift`]: a frame in which the coordinate origin moved
    /// must not be read as a frame in which the scenery moved.
    #[test]
    fn origin_drift_finds_the_shared_shift_and_leaves_the_real_mover() {
        let obj = |id: u64, w: [f32; 3]| ObjectLoc {
            id,
            draws: vec![0],
            world: w,
            heading: None,
            screen: None,
            centroid: None,
            distance: None,
            triangles: 1,
            sprites: false,
        };
        let shift = [-11.7, -0.66, -19.92];
        let mut prev = Vec::new();
        let mut now = Vec::new();
        for i in 0..20u64 {
            let p = [i as f32 * 3.0, 0.0, i as f32 * 7.0];
            prev.push(obj(i, p));
            // Bolted down: its coordinates change by the origin's displacement alone.
            now.push(obj(i, [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]]));
        }
        // One object that genuinely moved, by a completely different vector.
        let car = [500.0, -50.0, 300.0];
        prev.push(obj(999, car));
        now.push(obj(999, [car[0] + shift[0] + 12.0, car[1] + shift[1], car[2] + shift[2] + 8.0]));

        let d = origin_drift(&prev, &now, 0.05).expect("a match");
        assert!(d.reliable(), "a scene that is 20/21 static must yield a majority");
        for k in 0..3 {
            assert!((d.delta[k] - shift[k]).abs() < 1e-3, "axis {k}: {:?}", d.delta);
        }
        assert_eq!(d.agreed, 20, "the mover must not be counted in the cluster");
        assert_eq!(d.matched, 21);

        // And with the drift removed, the mover's residual is its TRUE motion while every
        // static object reads zero - the property navigation depends on.
        for (p, n) in prev.iter().zip(now.iter()) {
            let resid: Vec<f32> = (0..3).map(|k| n.world[k] - p.world[k] - d.delta[k]).collect();
            let mag = (resid[0] * resid[0] + resid[1] * resid[1] + resid[2] * resid[2]).sqrt();
            if n.id == 999 {
                assert!((mag - (12.0f32 * 12.0 + 8.0 * 8.0).sqrt()).abs() < 1e-2, "mover {mag}");
            } else {
                assert!(mag < 1e-3, "static object read as moving: {mag}");
            }
        }
    }

    #[test]
    fn origin_drift_refuses_to_guess_when_nothing_agrees() {
        let obj = |id: u64, w: [f32; 3]| ObjectLoc {
            id,
            draws: vec![0],
            world: w,
            heading: None,
            screen: None,
            centroid: None,
            distance: None,
            triangles: 1,
            sprites: false,
        };
        // Every object moved by a different amount: there is no origin displacement to
        // find, and inventing one would silently corrupt every delta in the report.
        let prev: Vec<ObjectLoc> = (0..12u64).map(|i| obj(i, [0.0, 0.0, 0.0])).collect();
        let now: Vec<ObjectLoc> =
            (0..12u64).map(|i| obj(i, [i as f32 * 5.0, 0.0, i as f32 * -3.0])).collect();
        let d = origin_drift(&prev, &now, 0.05).expect("objects did match");
        assert!(!d.reliable(), "no majority must be reported as unreliable, not as a vector");
        // Too small a sample is also not a majority worth trusting.
        let d2 = origin_drift(&prev[..3], &now[..3], 0.05).unwrap();
        assert!(!d2.reliable());
    }

    /// A row of identical fence posts drifting further than their own spacing. Matching on
    /// raw proximity pairs each post with its NEIGHBOUR and reports the fence as moving;
    /// matching against the drift-corrected expectation pairs each post with itself.
    #[test]
    fn world_motion_matches_a_repeated_mesh_to_itself_not_its_neighbour() {
        let post = |w: [f32; 3]| ObjectLoc {
            id: 0xfeed_face,
            draws: vec![0],
            world: w,
            heading: None,
            screen: None,
            centroid: None,
            distance: None,
            triangles: 143,
            sprites: false,
        };
        let spacing = 20.0;
        let drift = [-23.0, 0.0, 0.0];
        let prev: Vec<ObjectLoc> = (0..8).map(|i| post([i as f32 * spacing, 0.0, 0.0])).collect();
        let now: Vec<ObjectLoc> =
            (0..8).map(|i| post([i as f32 * spacing + drift[0], 0.0, 0.0])).collect();
        for o in &now {
            let (_, mag) = world_motion(&prev, o, drift).expect("matched");
            assert!(mag < 1e-3, "a static post must read 0, got {mag} at {:?}", o.world);
        }
        // The naive version this replaced: nearest RAW candidate, i.e. drift of zero.
        // Kept as a test so the bug cannot come back quietly.
        let naive: Vec<f32> =
            now.iter().filter_map(|o| world_motion(&prev, o, [0.0; 3]).map(|(_, m)| m)).collect();
        assert!(
            naive.iter().any(|m| *m > 1.0),
            "the uncorrected match should mis-pair posts - if it no longer does, this \
             fixture stopped exercising the bug: {naive:?}"
        );
    }

    #[test]
    fn map_keeps_the_higher_surface_and_measures_its_height() {
        // A wide floor with a small block standing on it.
        let scene = Scene {
            precompile: Default::default(),
            color: None,
            depth: None,
            multisample: 0,
            draws: vec![
                ground_quad(0.0, -50.0, -50.0, 50.0, 50.0, true),
                ground_quad(8.0, 0.0, 0.0, 20.0, 20.0, true),
            ],
        };
        let view = square_view([-50.0, -50.0, 50.0, 50.0], 100);
        let map = render_map(&scene, view, [0, 0, 0, 255], 1, None, [0.0; 3]);
        // Looking down keeps the block, not the floor beneath it.
        assert_eq!(map.height_at(10.0, 10.0), Some(8.0), "the block is on top");
        assert_eq!(map.height_at(-25.0, -25.0), Some(0.0), "open floor reads the floor");
        // The floor is the bulk of the covered area, so it is the ground level.
        assert_eq!(map.ground_level(0.25), Some(0.0));
        // Densest band first: the floor, then the block.
        let bins = map.height_bins(2.0);
        assert_eq!(bins[0].0, 0.0);
        assert!(bins.iter().any(|(h, _)| *h == 8.0), "the block has its own band: {bins:?}");
        // In the grid the block is an obstacle and the open floor is drivable.
        let grid = map.height_grid(10, 10, 0.0, 1.0);
        let g: Vec<&str> = grid.lines().collect();
        // Grid row 0 is the TOP of the image, which is the most negative Z; the block
        // occupies world x 0..20, z 0..20, so it lands below and right of centre.
        assert_eq!(g[0].chars().next(), Some('.'), "a corner of open floor");
        assert!(g.iter().any(|row| row.contains('#')), "the block reads as an obstacle");
        let blocked = g.iter().map(|r| r.chars().filter(|c| *c == '#').count()).sum::<usize>();
        assert_eq!(blocked, 4, "a 20x20 block over a 100x100 map is 2x2 of a 10x10 grid");
    }

    /// The failure that made the first map of this title a picture of the inside of the
    /// sky: a skydome is above everything at every pixel, so it wins the whole height
    /// field. The guest's own depth-write state is the filter.
    #[test]
    fn map_excludes_geometry_that_does_not_write_depth() {
        let sky = ground_quad(5000.0, -50.0, -50.0, 50.0, 50.0, false);
        let scene = Scene { precompile: Default::default(), color: None, depth: None, multisample: 0, draws:vec![ground_quad(0.0, -50.0, -50.0, 50.0, 50.0, true), sky] };
        let map = render_map(&scene, square_view([-50.0, -50.0, 50.0, 50.0], 40), [0, 0, 0, 255], 1, None, [0.0; 3]);
        assert_eq!(map.height_at(0.0, 0.0), Some(0.0), "the floor, not the sky");
        assert_eq!(map.ground_level(0.25), Some(0.0));
    }

    #[test]
    fn map_ceiling_drops_geometry_above_it_and_reveals_the_floor_below() {
        // A depth-WRITING roof over half the floor: the ceiling option is the only way to
        // see what is under it.
        let scene = Scene {
            precompile: Default::default(),
            color: None,
            depth: None,
            multisample: 0,
            draws: vec![
                ground_quad(0.0, -50.0, -50.0, 50.0, 50.0, true),
                ground_quad(30.0, -50.0, -50.0, 0.0, 50.0, true),
            ],
        };
        let view = square_view([-50.0, -50.0, 50.0, 50.0], 40);
        let roofed = render_map(&scene, view, [0, 0, 0, 255], 1, None, [0.0; 3]);
        assert_eq!(roofed.height_at(-25.0, 0.0), Some(30.0), "without a ceiling, the roof wins");
        let under = render_map(&scene, view, [0, 0, 0, 255], 1, Some(10.0), [0.0; 3]);
        assert_eq!(under.height_at(-25.0, 0.0), Some(0.0), "with a ceiling, the floor shows");
        assert_eq!(under.height_at(25.0, 0.0), Some(0.0), "the open half is unaffected");
    }

    #[test]
    fn map_origin_shifts_every_coordinate_into_the_anchored_frame() {
        let scene = Scene { precompile: Default::default(), color: None, depth: None, multisample: 0, draws:vec![ground_quad(4.0, -10.0, -10.0, 10.0, 10.0, true)] };
        let origin = [100.0, 4.0, -200.0];
        // The same geometry, asked for in a frame measured from `origin`: the quad now
        // lives at x -110..-90, z 190..210, and its height is 0 rather than 4.
        let view = square_view([-160.0, 140.0, -40.0, 260.0], 120);
        let map = render_map(&scene, view, [0, 0, 0, 255], 1, None, origin);
        assert_eq!(map.height_at(-100.0, 200.0), Some(0.0), "shifted in x, z AND y");
        assert_eq!(map.height_at(0.0, 0.0), None, "nothing at the raw position any more");
    }

    // --- 2D sprite locating -----------------------------------------------------------

    /// A textured screen-space quad: the shape a 2D title draws everything with. `Pixel`
    /// space (a texcoord present and no MVP uniform) is what `interpret_draw` infers for it.
    fn sprite_quad(x0: f32, y0: f32, x1: f32, y1: f32, u0: f32, v0: f32, tex_byte: u8) -> Draw {
        let mut d = located_draw([0.0, 0.0, 0.0], &[], [0f32; 16]);
        d.primitive = PRIM_TRIANGLES;
        d.index_count = 6;
        d.indices = [0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect();
        // pos.xy at 0, uv.xy at 8, colour at 16.
        d.vertex_stride = 20;
        let corners = [(x0, y0, u0, v0), (x1, y0, u0 + 0.1, v0), (x1, y1, u0 + 0.1, v0 + 0.1), (x0, y1, u0, v0 + 0.1)];
        d.vertices = corners
            .iter()
            .flat_map(|(x, y, u, v)| {
                let mut b: Vec<u8> = Vec::new();
                for f in [x, y, &0.0, u, v] {
                    b.extend_from_slice(&f.to_le_bytes());
                }
                b.truncate(16);
                b.extend_from_slice(&[255, 255, 255, 255]);
                b
            })
            .collect();
        d.attributes = vec![
            crate::capture::VertexAttribute { stream_index: 0, offset: 0, format: FORMAT_F32, component_count: 3, reg_index: 0 },
            crate::capture::VertexAttribute { stream_index: 0, offset: 12, format: FORMAT_F32, component_count: 2, reg_index: 1 },
            crate::capture::VertexAttribute { stream_index: 0, offset: 16, format: FORMAT_U8N, component_count: 4, reg_index: 2 },
        ].into();
        // No uniforms: that is what makes this 2D rather than MVP, which is the whole point.
        d.uniforms = vec![];
        d.textures = [BoundTexture {
            // A fixture: a DISTINCT buffer, so a distinct identity - two fixtures sharing
            // one id would collide in every cache keyed on it.
            pixels_id: crate::capture::next_pixels_id(),
            unit: 0,
            base_format: 0x0c,
            swizzle: 0,
            tex_type: 0,
            width: 4,
            height: 4,
            stride: 16,
            faces: 1,
            face_bytes: 64,
            levels: 1,
            data_addr: 0x1000,
            pixels: vec![tex_byte; 64].into(),
            u_addr_mode: 0,
            v_addr_mode: 0,
            lod_bias: 0,
            min_filter: 0,
            mag_filter: 0,
            mip_filter: 0,
            gamma: 0,
        }]
        .into();
        d
    }

    #[test]
    fn sprites_are_located_on_screen_and_keep_their_identity_when_they_move() {
        let scene = Scene {
            precompile: Default::default(),
            color: None,
            depth: None,
            multisample: 0,
            draws: vec![sprite_quad(100.0, 200.0, 180.0, 280.0, 0.0, 0.0, 7)],
        };
        let found = locate_sprites(&scene, 960, 544);
        assert_eq!(found.len(), 1, "one 2D quad is one sprite: {found:?}");
        let s = &found[0];
        assert_eq!(s.centroid, [140.0, 240.0]);
        assert_eq!(s.size, [80.0, 80.0]);
        assert!(s.textured);

        // The SAME sprite 300 pixels along keeps its id - which a 3D geometry hash could
        // not do, because a 2D sprite's position IS its vertex data.
        let moved = Scene {
            precompile: Default::default(),
            color: None,
            depth: None,
            multisample: 0,
            draws: vec![sprite_quad(400.0, 200.0, 480.0, 280.0, 0.0, 0.0, 7)],
        };
        let after = locate_sprites(&moved, 960, 544);
        assert_eq!(after[0].id, s.id, "identity must survive motion");
        // A different region of the same sheet is a DIFFERENT sprite.
        let other = Scene {
            precompile: Default::default(),
            color: None,
            depth: None,
            multisample: 0,
            draws: vec![sprite_quad(100.0, 200.0, 180.0, 280.0, 0.5, 0.5, 7)],
        };
        assert_ne!(locate_sprites(&other, 960, 544)[0].id, s.id, "another atlas region");
    }

    #[test]
    fn sprite_motion_removes_the_scene_scroll() {
        // A backdrop of many sprites panning left by 6px, and one that moves against it.
        let build = |shift: f32, hero_extra: f32| Scene {
            precompile: Default::default(),
            color: None,
            depth: None,
            multisample: 0,
            draws: (0..12)
                .map(|i| {
                    let x = 40.0 * i as f32 + shift;
                    sprite_quad(x, 100.0, x + 30.0, 130.0, 0.05 * i as f32, 0.0, i as u8)
                })
                .chain(std::iter::once(sprite_quad(
                    500.0 + shift + hero_extra,
                    300.0,
                    540.0 + shift + hero_extra,
                    340.0,
                    0.9,
                    0.9,
                    99,
                )))
                .collect(),
        };
        let before = locate_sprites(&build(0.0, 0.0), 960, 544);
        let after = locate_sprites(&build(-6.0, 20.0), 960, 544);
        let drift = scroll_drift(&before, &after, 0.75).expect("matched");
        assert!(drift.reliable(), "12 of 13 sprites agree, so this is a majority");
        assert!((drift.delta[0] + 6.0).abs() < 1e-3, "the pan is -6px: {:?}", drift.delta);

        for s in &after {
            let (_, mag) = sprite_motion(&before, s, drift.delta).expect("matched");
            if s.size[0] == 40.0 {
                // The hero moved 20px through the world on top of the scroll.
                assert!((mag - 20.0).abs() < 1e-2, "hero motion {mag}");
            } else {
                assert!(mag < 1e-2, "a backdrop sprite must read as still, got {mag}");
            }
        }
    }

    #[test]
    fn sprites_ignore_3d_draws_and_locate_ignores_2d_ones() {
        // The two locators must partition the scene, or an object gets counted twice - or,
        // worse, a title gets an empty report from the one that does not apply to it.
        let scene = Scene {
            precompile: Default::default(),
            color: None,
            depth: None,
            multisample: 0,
            draws: vec![
                sprite_quad(10.0, 10.0, 50.0, 50.0, 0.0, 0.0, 1),
                ground_quad(0.0, -10.0, -10.0, 10.0, 10.0, true),
            ],
        };
        let two_d = locate_sprites(&scene, 960, 544);
        let three_d = locate_scene(&scene, 960, 544);
        assert_eq!(two_d.len(), 1, "only the quad with no MVP is a sprite");
        assert_eq!(two_d[0].draw, 0);
        assert_eq!(three_d.len(), 1, "only the MVP draw is a placed object");
        assert_eq!(three_d[0].draws, vec![1]);
    }

    // --- traversability and route planning ------------------------------------------

    /// A 200x200 floor with a wall across it at z = 0, with a gap. The wall is a tall thin
    /// quad, so on the height field it is a fast rise, which is what stops a vehicle.
    fn walled_scene(gap: Option<(f32, f32)>) -> Scene {
        let mut draws = vec![ground_quad(0.0, -100.0, -100.0, 100.0, 100.0, true)];
        match gap {
            None => draws.push(ground_quad(20.0, -100.0, -4.0, 100.0, 4.0, true)),
            Some((g0, g1)) => {
                draws.push(ground_quad(20.0, -100.0, -4.0, g0, 4.0, true));
                draws.push(ground_quad(20.0, g1, -4.0, 100.0, 4.0, true));
            }
        }
        Scene { precompile: Default::default(), color: None, depth: None, multisample: 0, draws }
    }

    fn walled_map(gap: Option<(f32, f32)>) -> WorldMap {
        render_map(
            &walled_scene(gap),
            MapView { extent: [-100.0, -100.0, 100.0, 100.0], width: 200, height: 200 },
            [0, 0, 0, 255],
            1,
            None,
            [0.0; 3],
        )
    }

    #[test]
    fn traversable_blocks_a_wall_and_keeps_the_open_floor() {
        let map = walled_map(Some((10.0, 30.0)));
        let mask = Traversable::from_map(&map, 1.0, 0);
        let open_at = |wx: f32, wz: f32| {
            let p = map.view.pixel_of(wx, wz);
            mask.is_open(p[0] as i64, p[1] as i64)
        };
        assert!(open_at(-50.0, -50.0), "open floor is drivable");
        assert!(open_at(20.0, 0.0), "the gap in the wall is drivable");
        // The wall's FOOT is blocked, which is the property that stops a vehicle crossing
        // it. Its flat top is not distinguishable from floor by slope alone - so it stays
        // "open" here, as an island nothing can reach. `plan_route` is what must not be
        // fooled by that, and it flood-fills from the goal rather than trusting a snap.
        assert!(!open_at(-50.0, -4.5), "the foot of the wall is blocked");
        assert!(!open_at(-50.0, 4.5), "and so is the far foot");
        // A mask that came out almost entirely closed means the slope limit is wrong, and
        // the caller needs to be able to see that rather than just get "no route".
        let f = mask.open_fraction();
        assert!(f > 0.5 && f < 1.0, "most of a mostly-open floor should be open, got {f}");
    }

    #[test]
    fn traversable_calls_a_ramp_drivable_and_a_step_not() {
        // Two surfaces rising the same total height: one over 60 world units, one abruptly.
        let mut draws = vec![ground_quad(0.0, -100.0, -100.0, 100.0, 100.0, true)];
        for i in 0..60 {
            let x = -80.0 + i as f32;
            draws.push(ground_quad(i as f32 * 0.1, x, -50.0, x + 1.0, -20.0, true));
        }
        draws.push(ground_quad(6.0, 20.0, -50.0, 60.0, -20.0, true));
        let scene = Scene { precompile: Default::default(), color: None, depth: None, multisample: 0, draws };
        let map = render_map(
            &scene,
            MapView { extent: [-100.0, -100.0, 100.0, 100.0], width: 200, height: 200 },
            [0, 0, 0, 255],
            1,
            None,
            [0.0; 3],
        );
        let mask = Traversable::from_map(&map, 0.5, 0);
        let open_at = |wx: f32, wz: f32| {
            let p = map.view.pixel_of(wx, wz);
            mask.is_open(p[0] as i64, p[1] as i64)
        };
        assert!(open_at(-50.0, -35.0), "a gentle ramp is drivable ground");
        // The abrupt slab's own top is flat, but its EDGE is not passable - which is the
        // property that matters, since a route has to cross the edge to get on it.
        let p = map.view.pixel_of(20.0, -35.0);
        assert!(!mask.is_open(p[0] as i64, p[1] as i64), "the foot of a step is blocked");
    }

    #[test]
    fn plan_route_goes_through_the_gap_and_stays_on_open_ground() {
        let map = walled_map(Some((10.0, 30.0)));
        let mask = Traversable::from_map(&map, 1.0, 2);
        let route = plan_route(&map, &mask, [-60.0, -60.0], [-60.0, 60.0], 20)
            .expect("a gap exists, so a route must be found");
        assert!(route.len() >= 3, "a route around a wall has turns in it: {route:?}");
        // Every straight leg must stay on open ground - the property that makes a route
        // followable rather than a set of suggestions.
        for pair in route.windows(2) {
            let (a, b) = (map.view.pixel_of(pair[0][0], pair[0][1]), map.view.pixel_of(pair[1][0], pair[1][1]));
            let steps = ((b[0] - a[0]).abs().max((b[1] - a[1]).abs()).max(1.0)) as i64;
            for s in 0..=steps {
                let t = s as f32 / steps as f32;
                let x = (a[0] + (b[0] - a[0]) * t).round() as i64;
                let y = (a[1] + (b[1] - a[1]) * t).round() as i64;
                assert!(mask.is_open(x, y), "leg {pair:?} crosses blocked ground at ({x},{y})");
            }
        }
        // And it must actually pass through the gap, not teleport across the wall.
        assert!(
            route.iter().any(|p| p[0] > 5.0 && p[0] < 35.0 && p[1].abs() < 20.0),
            "the route should thread the gap at x 10..30: {route:?}"
        );
    }

    #[test]
    fn plan_route_admits_when_the_goal_is_walled_off() {
        let map = walled_map(None);
        let mask = Traversable::from_map(&map, 1.0, 1);
        // A solid wall: the honest answer is that there is no route, not a path through it.
        assert!(plan_route(&map, &mask, [-60.0, -60.0], [-60.0, 60.0], 10).is_none());
        // The same two points on the SAME side are still connected.
        assert!(plan_route(&map, &mask, [-60.0, -60.0], [60.0, -60.0], 10).is_some());
    }

    #[test]
    fn plan_route_simplifies_open_ground_to_two_points() {
        let scene = Scene { precompile: Default::default(), color: None, depth: None, multisample: 0, draws:vec![ground_quad(0.0, -100.0, -100.0, 100.0, 100.0, true)] };
        let map = render_map(
            &scene,
            MapView { extent: [-100.0, -100.0, 100.0, 100.0], width: 200, height: 200 },
            [0, 0, 0, 255],
            1,
            None,
            [0.0; 3],
        );
        let mask = Traversable::from_map(&map, 1.0, 0);
        let route = plan_route(&map, &mask, [-60.0, -60.0], [60.0, 60.0], 10).expect("open ground");
        assert_eq!(route.len(), 2, "nothing in the way means start and finish: {route:?}");
        assert!((route[1][0] - 60.0).abs() < 3.0 && (route[1][1] - 60.0).abs() < 3.0);
    }

    #[test]
    fn plan_route_snaps_an_endpoint_that_sits_on_blocked_ground() {
        let map = walled_map(Some((10.0, 30.0)));
        let mask = Traversable::from_map(&map, 1.0, 2);
        // Start pressed against the wall's foot, where the clearance erosion has blocked
        // the vehicle's own pixel. Refusing to plan from there would fail exactly when the
        // planner is needed.
        let route = plan_route(&map, &mask, [-50.0, -5.0], [-60.0, 60.0], 30)
            .expect("a start beside an obstacle must snap to open ground");
        assert!(route.len() >= 2);
        // Snapping must land on ground the goal can actually be reached from. The flat top
        // of the wall is open, is nearer to a point on the wall than the floor is, and is
        // an island - so a planner that snapped first and searched afterwards would either
        // fail here or route along the wall.
        let on_wall = plan_route(&map, &mask, [-50.0, 0.0], [-60.0, 60.0], 30);
        if let Some(r) = &on_wall {
            for p in r {
                let q = map.view.pixel_of(p[0], p[1]);
                assert!(
                    map.height_at(p[0], p[1]).is_some_and(|h| h < 10.0),
                    "route point {p:?} (pixel {q:?}) is on top of the wall"
                );
            }
        }
        // And a snap radius of zero must not silently invent a start position.
        assert!(plan_route(&map, &mask, [-50.0, -5.0], [-60.0, 60.0], 0).is_none());
    }

    #[test]
    fn world_extent_lets_vertex_density_decide_and_ignores_a_sparse_shell() {
        // A densely tessellated small floor, plus a distant 4-vertex backdrop. A strict
        // min/max would size the map to the backdrop and leave the floor sub-pixel.
        let mut draws = vec![ground_quad(1000.0, -9000.0, -9000.0, 9000.0, 9000.0, true)];
        for i in 0..60 {
            let x = -30.0 + i as f32;
            draws.push(ground_quad(0.0, x, -30.0, x + 1.0, 30.0, true));
        }
        let scene = Scene { precompile: Default::default(), color: None, depth: None, multisample: 0, draws };
        let strict = world_extent(&scene, 1.0).unwrap();
        assert!(strict[0] < -8000.0, "at keep=1.0 the backdrop sets the extent");
        let dense = world_extent(&scene, 0.90).unwrap();
        assert!(
            dense[0] > -200.0 && dense[2] < 200.0,
            "the tessellated floor should set the extent, got {dense:?}"
        );
    }
}

#[cfg(test)]
mod supersample_tests {
    //! Content-free tests for the supersample antialiasing path: the box-downsample resolve,
    //! Pixel-space scaling under supersampling, and that supersampling actually reduces the
    //! per-pixel variance a sub-pixel-triangle mesh produces. No game data.
    use super::*;
    use crate::capture::{Draw, RenderState, Scene, VertexAttribute};

    /// A 2x2 block of four solid colours box-downsamples to their exact integer average.
    #[test]
    fn downsample_box_averages() {
        let mut fb = Framebuffer::new(2, 2, [0, 0, 0, 0]);
        fb.rgba = vec![
            40, 0, 0, 255, 80, 0, 0, 255, // row 0: two reds
            0, 40, 0, 255, 0, 80, 0, 255, // row 1: two greens
        ];
        let d = fb.downsampled(2);
        assert_eq!((d.width, d.height), (1, 1));
        // avg of (40,0,0),(80,0,0),(0,40,0),(0,80,0) = (30,30,0), alpha 255.
        assert_eq!(d.pixel(0, 0), [30, 30, 0, 255]);
    }

    /// `downsampled(1)` is the identity, and a solid uniform image is unchanged by any factor.
    #[test]
    fn downsample_identity_and_uniform() {
        let fb = Framebuffer::new(4, 4, [11, 22, 33, 255]);
        assert_eq!(fb.downsampled(1).rgba, fb.rgba);
        let d = fb.downsampled(2);
        assert_eq!((d.width, d.height), (2, 2));
        assert!(d.rgba.chunks(4).all(|p| p == [11, 22, 33, 255]));
    }

    /// A full-frame Pixel-space quad must cover the frame identically at ssaa 1 and 2: the
    /// Pixel projection scales native coords by the factor, so supersampling then downsampling
    /// reproduces the same solid fill (not a quarter-size sprite in the corner).
    #[test]
    fn pixel_space_scales_under_supersample() {
        let (w, h) = (16u32, 12u32);
        // A pixel-space quad covering [0,w]x[0,h]: pos.xy f32 @0, uv.xy f32 @8, no color.
        let mut verts = Vec::new();
        for (x, y, u, v) in [(0.0, 0.0, 0.0, 0.0), (w as f32, 0.0, 1.0, 0.0), (w as f32, h as f32, 1.0, 1.0), (0.0, h as f32, 0.0, 1.0)] {
            for f in [x, y, u, v] {
                verts.extend_from_slice(&(f as f32).to_le_bytes());
            }
        }
        let tex = BoundTexture {
            // A fixture: a DISTINCT buffer, so a distinct identity - two fixtures sharing
            // one id would collide in every cache keyed on it.
            pixels_id: crate::capture::next_pixels_id(),
            unit: 0, base_format: 0x0c, swizzle: 0, tex_type: 0, width: 1, height: 1, stride: 4,
            faces: 1, face_bytes: 4, levels: 1,
            pixels: vec![200, 100, 50, 255].into(), data_addr: 0, u_addr_mode: 0, v_addr_mode: 0,
            lod_bias: 0, min_filter: 0, mag_filter: 0, mip_filter: 0, gamma: 0,
        };
        let draw = Draw {
            fragment_program_header: 0,
            primitive: PRIM_TRIANGLES, index_format: 0, index_count: 6,
            vertices: verts.into(), vertex_stride: 16,
            attributes: vec![
                VertexAttribute { stream_index: 0, offset: 0, format: FORMAT_F32, component_count: 2, reg_index: 0 },
                VertexAttribute { stream_index: 0, offset: 8, format: FORMAT_F32, component_count: 2, reg_index: 1 },
            ].into(),
            indices: [0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect(),
            uniforms: vec![], textures: vec![tex].into(), vertex_textures: std::sync::Arc::from(&[][..]), render_state: std::sync::Arc::new(RenderState::default()),
            blend: crate::capture::BlendState::default(),
            exposure: 1.0, material: Default::default(), world: [0.0; 16],
            vprog: crate::capture::no_program(), fprog: crate::capture::no_program(),
            vert_sa: std::sync::Arc::from(&[][..]), frag_sa: std::sync::Arc::from(&[][..]), frag_sa_addr: 0, mem_windows: Vec::new(), shader_expanded: false,
        };
        let scene = Scene { precompile: Default::default(), color: None, depth: None, multisample: 0, draws:vec![draw] };
        let a = render_scene_supersampled(&scene, w, h, [0, 0, 0, 255], 1);
        let b = render_scene_supersampled(&scene, w, h, [0, 0, 0, 255], 2);
        assert_eq!((b.width, b.height), (w, h));
        // Center pixel is the (blend-modulated) texel in both; supersampling must not shrink it.
        assert_eq!(a.pixel(w / 2, h / 2), b.pixel(w / 2, h / 2));
        // The whole interior is covered (not a corner quarter): a mid pixel is not the clear.
        assert_ne!(b.pixel(w / 2, h / 2), [0, 0, 0, 255]);
    }

    /// Supersampling reduces the per-pixel high-frequency variance minified high-detail content
    /// produces - the aliasing "speckle" the car's fine textures and sub-pixel-triangle body
    /// show at 1x. A fine black/white checker texture minified onto a quad point-samples to a
    /// harsh scatter at 1x and averages toward a smooth mid-grey at 4x; assert the 4x frame's
    /// neighbouring-pixel variance is far lower.
    #[test]
    fn supersample_reduces_speckle_variance() {
        // A 96x96 1-bit checker as a U8U8U8U8 texture: alternating black/white texels. Minified
        // 3x onto the 32px frame it point-samples to a per-pixel black/white checker (maximal
        // adjacent-pixel variance) at 1x - the aliasing a single sample per pixel cannot resolve.
        let (tw, th) = (96u32, 96u32);
        let mut pixels = Vec::with_capacity((tw * th * 4) as usize);
        for y in 0..th {
            for x in 0..tw {
                let c = if (x + y) % 2 == 0 { 255 } else { 0 };
                pixels.extend_from_slice(&[c, c, c, 255]);
            }
        }
        let tex = BoundTexture {
            // A fixture: a DISTINCT buffer, so a distinct identity - two fixtures sharing
            // one id would collide in every cache keyed on it.
            pixels_id: crate::capture::next_pixels_id(),
            unit: 0, base_format: 0x0c, swizzle: 0, tex_type: 0, width: tw, height: th, stride: tw * 4,
            faces: 1, face_bytes: tw * th * 4, levels: 1,
            pixels: pixels.into(), data_addr: 0, u_addr_mode: 0, v_addr_mode: 0, lod_bias: 0, min_filter: 0, mag_filter: 0, mip_filter: 0, gamma: 0,
        };
        // A full-frame Pixel-space quad over a 32px frame, uv 0..1 across the 64px checker, so
        // it is minified 2x - the aliasing regime one sample per pixel cannot resolve.
        let (w, h) = (32u32, 32u32);
        let mut verts = Vec::new();
        for (x, y, u, v) in [(0.0f32, 0.0, 0.0, 0.0), (w as f32, 0.0, 1.0, 0.0), (w as f32, h as f32, 1.0, 1.0), (0.0, h as f32, 0.0, 1.0)] {
            for f in [x, y, u, v] {
                verts.extend_from_slice(&f.to_le_bytes());
            }
        }
        let draw = Draw {
            fragment_program_header: 0,
            primitive: PRIM_TRIANGLES, index_format: 0, index_count: 6,
            vertices: verts.into(), vertex_stride: 16,
            attributes: vec![
                VertexAttribute { stream_index: 0, offset: 0, format: FORMAT_F32, component_count: 2, reg_index: 0 },
                VertexAttribute { stream_index: 0, offset: 8, format: FORMAT_F32, component_count: 2, reg_index: 1 },
            ].into(),
            indices: [0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect(),
            uniforms: vec![], textures: vec![tex].into(), vertex_textures: std::sync::Arc::from(&[][..]), render_state: std::sync::Arc::new(RenderState::default()),
            blend: crate::capture::BlendState::default(),
            exposure: 1.0, material: Default::default(), world: [0.0; 16],
            vprog: crate::capture::no_program(), fprog: crate::capture::no_program(),
            vert_sa: std::sync::Arc::from(&[][..]), frag_sa: std::sync::Arc::from(&[][..]), frag_sa_addr: 0, mem_windows: Vec::new(), shader_expanded: false,
        };
        let s = Scene { precompile: Default::default(), color: None, depth: None, multisample: 0, draws:vec![draw] };
        // Mean absolute difference between horizontally-adjacent pixels (a speckle proxy).
        fn h_variance(fb: &Framebuffer) -> f64 {
            let mut acc = 0f64;
            let mut n = 0f64;
            for y in 0..fb.height {
                for x in 1..fb.width {
                    acc += (fb.pixel(x, y)[0] as f64 - fb.pixel(x - 1, y)[0] as f64).abs();
                    n += 1.0;
                }
            }
            acc / n.max(1.0)
        }
        let one = h_variance(&render_scene_supersampled(&s, w, h, [0, 0, 0, 255], 1));
        let four = h_variance(&render_scene_supersampled(&s, w, h, [0, 0, 0, 255], 4));
        // 4x must markedly smooth the aliased checker: adjacent-pixel jumps shrink substantially.
        assert!(four < one * 0.6, "supersampling should reduce speckle variance: 1x={one:.1} 4x={four:.1}");
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
            // A fixture: a DISTINCT buffer, so a distinct identity - two fixtures sharing
            // one id would collide in every cache keyed on it.
            pixels_id: crate::capture::next_pixels_id(),
            unit: 0,
            base_format,
            // sample_texture reads bits 12..14; pass the swizzle in that field.
            swizzle: swizzle << 12,
            tex_type: 3,
            width: w,
            height: h,
            stride,
            faces: 1,
            face_bytes: pixels.len() as u32,
            levels: 1,
            data_addr: 0,
            pixels: pixels.into(),
            u_addr_mode: 0,
            v_addr_mode: 0,
            lod_bias: 0,
            min_filter: 0,
            mag_filter: 0,
            mip_filter: 0,
            gamma: 0,
        }
    }

    // u that lands squarely in texel `i` of a `w`-wide row.
    fn u_of(i: u32, w: u32) -> f32 {
        (i as f32 + 0.5) / w as f32
    }

    /// U2F10F10F10 (base format 0x9a): three unsigned 10-bit FLOATS under a 2-bit unorm
    /// alpha. Until this was decoded, every draw sampling one was painted the magenta
    /// missing-format marker - a golf title's club shaft, on every frame.
    ///
    /// Pinned on exact values rather than "not magenta": a 10-bit float is five bits of
    /// exponent over five of mantissa at bias 15, and getting the split wrong still produces
    /// a plausible colour.
    #[test]
    fn u2f10f10f10_decodes_its_three_packed_floats_and_two_bit_alpha() {
        // Lane at bit 20 = 1.0 (exp 15, mant 0), at bit 10 = 0.5 (exp 14), at bit 0 = 0.0,
        // alpha = 3 (full).
        let w: u32 = (3 << 30) | (0x1e0 << 20) | (0x1c0 << 10);
        let px = w.to_le_bytes().to_vec();
        // swizzle 1 is ARGB: the top colour lane first.
        let t = tex(0x9a, 1, 1, 1, 4, px.clone());
        assert_eq!(texel_rgba_face(&t, 0, 0, 0), [255, 128, 0, 255]);
        // swizzle 0 is ABGR: the same three lanes in the opposite order.
        let t = tex(0x9a, 0, 1, 1, 4, px);
        assert_eq!(texel_rgba_face(&t, 0, 0, 0), [0, 128, 255, 255]);
        // A zero word is transparent black, not magenta - the marker must be gone.
        let t = tex(0x9a, 1, 1, 1, 4, vec![0, 0, 0, 0]);
        assert_eq!(texel_rgba_face(&t, 0, 0, 0), [0, 0, 0, 0]);
    }

    /// The UNCOMPRESSED fast path must produce EXACTLY what the per-texel path produces.
    ///
    /// Same contract as the block-compressed case below, over the one-texel-per-block
    /// formats - which are the LARGER half of what a mid-race frame decodes. Covers every
    /// lane width the decoder distinguishes (8/16/24/32/64-bit), both addressing modes, and
    /// non-power-of-two sizes, where a swizzled texture's padding is not the image.
    /// The integer 16-bit reductions are the float ones, over every input there is.
    ///
    /// Not a spot check: both are total functions of a `u16`, so the whole domain is 65,536
    /// cases and costs microseconds to enumerate. This is what makes the substitution a
    /// REWRITE OF THE COST rather than of the answer [[vitaslop-identical-output-is-evidence]]
    /// - the largest texture family a retail title decodes goes through it, and a
    /// one-in-65,536 disagreement would show up as a single wrong texel nobody could find.
    #[test]
    fn the_integer_sixteen_bit_reductions_match_the_float_ones() {
        for raw in 0..=u16::MAX {
            let unorm = ((raw as f32 / 65535.0).clamp(0.0, 1.0) * 255.0).round() as u8;
            assert_eq!(unorm16_to_u8(raw), unorm, "U16 raw {raw}");
            let snorm =
                ((((raw as i16) as f32 / 32767.0).max(0.0)).clamp(0.0, 1.0) * 255.0).round() as u8;
            assert_eq!(snorm16_to_u8(raw), snorm, "S16 raw {raw}");
        }
    }

    /// The block decoder is the per-texel one, for every BC family and every block.
    ///
    /// The per-texel path is the ORACLE: it is what a single sampler read still goes through,
    /// and it is the code the block decoder hoists work out of. So this enumerates blocks
    /// rather than arguing - including the two modes that are decided by the ENDPOINT ORDER
    /// (BC1's punch-through when `c0 <= c1`, and BC3's six-entry alpha ramp when `a0 <= a1`),
    /// which are exactly the branches a hoist can get backwards.
    #[test]
    fn the_block_decoder_matches_the_per_texel_one() {
        // A deterministic spread: a counter run through a cheap mixer, so the endpoints land
        // on both sides of both order tests and the index bits take every value.
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for fmt in [0x85u32, 0x86, 0x87] {
            for case in 0..2000 {
                let mut block = [0u8; 16];
                for b in block.iter_mut() {
                    *b = (next() >> 24) as u8;
                }
                // Force both endpoint orders to be exercised rather than left to chance.
                if case % 2 == 0 {
                    block[1] = 0x00;
                    block[3] = 0xff;
                    block[9] = 0x00;
                }
                let whole = decode_bc_block(&block, fmt);
                for t in 0..16usize {
                    let (px, py) = ((t % 4) as u32, (t / 4) as u32);
                    assert_eq!(
                        whole[t],
                        decode_bc_texel(&block, fmt, px, py),
                        "format {fmt:#x} case {case} texel {t} of {block:02x?}"
                    );
                }
                // A SHORT block: both paths read zeros past the end, and the walker hands one
                // in whenever the guest buffer ends inside the last row of blocks.
                let short = &block[..5];
                let whole_short = decode_bc_block(short, fmt);
                for t in 0..16usize {
                    let (px, py) = ((t % 4) as u32, (t / 4) as u32);
                    assert_eq!(
                        whole_short[t],
                        decode_bc_texel(short, fmt, px, py),
                        "format {fmt:#x} short block texel {t}"
                    );
                }
            }
        }
    }

    #[test]
    fn uncompressed_fast_path_matches_per_texel() {
        // One format from each width family the match arms distinguish.
        let formats = [
            0x00u32, 0x01, // 8-bit single channel
            0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, // 16-bit
            0x09, 0x0a, 0x0b, // 16-bit single channel
            0x0c, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x15, 0x17, 0x18, 0x19, 0x1a, // 32-bit
            0x1b, 0x1c, 0x1d, 0x1e, 0x1f, // 64-bit
            0x9a, // 32-bit packed float + 2-bit alpha
            0x98, 0x99, // 24-bit
        ];
        for fmt in formats {
            let Some((bw, bh, bytes)) = block_layout(fmt) else { continue };
            assert_eq!((bw, bh), (1, 1), "format {fmt:#x} is not one texel per block");
            for &(w, h) in &[(4u32, 4u32), (8, 4), (5, 3)] {
                for &tex_type in &[3u32 /* linear */, 0 /* swizzled */] {
                    for &chan_swizzle in &[0u32, 1, 2, 3] {
                        let pw = w.next_power_of_two() as usize;
                        let ph = h.next_power_of_two() as usize;
                        let n = (pw * ph + w as usize * h as usize + 8) * bytes as usize;
                        let pixels: Vec<u8> =
                            (0..n).map(|i| ((i * 53 + 7) % 251) as u8).collect();
                        let mut t = tex(fmt, chan_swizzle, w, h, w * bytes, pixels);
                        t.tex_type = tex_type;
                        let (_, _, fast) = decode_texture_rgba8(&t);
                        let mut slow = Vec::with_capacity(fast.len());
                        for y in 0..h {
                            for x in 0..w {
                                slow.extend_from_slice(&texel_rgba_face(&t, 0, x, y));
                            }
                        }
                        assert_eq!(
                            fast, slow,
                            "fast path differs from per-texel for format {fmt:#x} {w}x{h} \
                             tex_type {tex_type} swizzle {chan_swizzle}"
                        );
                    }
                }
            }
        }
    }

    /// A mip chain's levels are consecutive and largest-first, and every level is laid out
    /// exactly the way level 0 is.
    ///
    /// This is the arithmetic the snapshot sizes its read from and the passthrough finds its
    /// blocks with, so it is asserted against hand-computed numbers rather than against a
    /// second copy of itself. The SWIZZLED case is the one worth spelling out: a level's grid is
    /// padded to a power of two in BLOCKS, so a 12x12 BC3 level is 3x3 blocks stored as 4x4.
    #[test]
    fn a_mip_level_is_laid_out_like_a_level_zero_of_its_own_size() {
        // BC3 (16-byte 4x4 blocks), LINEAR: level l is w>>l wide, block-packed rows.
        let l0 = level_layout(0x87, 3, 64, 32, 0).unwrap();
        assert_eq!((l0.width, l0.height, l0.blocks_x, l0.blocks_y), (64, 32, 16, 8));
        assert_eq!((l0.stride, l0.bytes), (16 * 16, 16 * 8 * 16));
        let l2 = level_layout(0x87, 3, 64, 32, 2).unwrap();
        assert_eq!((l2.width, l2.height, l2.blocks_x, l2.blocks_y), (16, 8, 4, 2));
        assert_eq!(l2.bytes, 4 * 2 * 16);
        // Offsets accumulate, so level 2 sits after levels 0 and 1.
        let l1 = level_layout(0x87, 3, 64, 32, 1).unwrap();
        assert_eq!(level_offset(0x87, 3, 64, 32, 2).unwrap(), l0.bytes + l1.bytes);
        // SWIZZLED: the stored grid is padded to a power of two IN BLOCKS, which is why a
        // level's size is not simply its block count times the block size.
        let s = level_layout(0x87, 0, 12, 12, 0).unwrap();
        assert_eq!((s.blocks_x, s.blocks_y), (3, 3));
        assert_eq!(s.bytes, 4 * 4 * 16, "a 3x3 block grid is STORED as 4x4");
        // The chain bottoms out at 1x1, which is still one whole block.
        let last = level_layout(0x87, 3, 64, 32, 5).unwrap();
        assert_eq!((last.width, last.height, last.bytes), (2, 1, 16));
        assert_eq!(max_mip_levels(64, 32), 7);
    }

    /// The compressed passthrough must hand the GPU blocks that decode to the SAME image the
    /// decoder produces from the guest's own bytes.
    ///
    /// A swizzled texture stores its blocks in Morton order and WebGPU wants linear block rows,
    /// so the passthrough permutes them - and a permutation is exactly the kind of change that
    /// produces a plausible picture with its 4x4 tiles shuffled. The check is therefore end to
    /// end: decode the guest texture, then decode a LINEAR texture built from the passthrough
    /// bytes, and require the two images to be identical. Nothing here trusts `morton_index`
    /// twice over - the reference decode reaches its blocks by the decoder's own addressing.
    #[test]
    fn the_passthrough_preserves_the_image_it_hands_over() {
        for &(fmt, bb) in &[(0x85u32, 8u32), (0x86, 16), (0x87, 16)] {
            for &tex_type in &[0u32 /* swizzled */, 3 /* linear */] {
                let (w, h) = (16u32, 8u32);
                let levels = max_mip_levels(w, h);
                let total = level_offset(fmt, tex_type, w, h, levels).unwrap() as usize;
                let pixels: Vec<u8> = (0..total).map(|i| ((i * 37 + 11) % 251) as u8).collect();
                let mut t = tex(fmt, 0, w, h, w.div_ceil(4) * bb, pixels);
                t.tex_type = tex_type;
                t.levels = levels;
                t.face_bytes = total as u32;
                let c = compressed_source(&t, None).expect("a mipped, identity-swizzled, block-aligned texture");
                assert_eq!(c.levels, levels);
                let bytes = c.cpu_bytes().expect("a passthrough is always built on the CPU");
                assert_eq!(bytes.len(), (0..levels).map(|l| {
                    let ll = level_layout(fmt, tex_type, w, h, l).unwrap();
                    (ll.blocks_x * ll.blocks_y * bb) as usize
                }).sum::<usize>());
                // Level 0 of the passthrough, read back as a LINEAR texture of the same size.
                let l0 = level_layout(fmt, tex_type, w, h, 0).unwrap();
                let n0 = (l0.blocks_x * l0.blocks_y * bb) as usize;
                let linear = tex(fmt, 0, w, h, l0.blocks_x * bb, bytes[..n0].to_vec());
                let (_, _, want) = decode_texture_rgba8(&t);
                let (_, _, got) = decode_texture_rgba8(&linear);
                assert_eq!(got, want, "passthrough changed the image for {fmt:#x} type {tex_type}");
            }
        }
    }

    /// Every condition that refuses a LOSSLESS passthrough refuses it, and a texture meeting all
    /// of them is accepted.
    ///
    /// Each condition is a different WRONG PICTURE rather than a slow one, so the temptation to
    /// let one through "just this once" has to fail here. The mip condition is the subtle one:
    /// dropping the chain trades a memory defect for the white-speckle image defect.
    ///
    /// Asserted against `passthrough_source`, NOT `compressed_source` - a refusal is not a
    /// decision to decode, it is a decision to re-encode instead (see the test below). Pointing
    /// this at the outer function would silently stop testing anything the day the fallback
    /// landed, because every refusal would still come back `Some`.
    #[test]
    fn a_passthrough_is_refused_for_each_stated_reason() {
        let build = |fmt: u32, tex_type: u32, w: u32, h: u32| {
            let levels = max_mip_levels(w, h);
            let total = level_offset(fmt, tex_type, w, h, levels).unwrap() as usize;
            let mut t = tex(fmt, 0, w, h, w.div_ceil(4) * 16, vec![7u8; total]);
            t.tex_type = tex_type;
            t.levels = levels;
            t.face_bytes = total as u32;
            t
        };
        assert!(passthrough_source(&build(0x87, 3, 16, 8)).is_some(), "the accepted case");
        // A format WebGPU does not have.
        assert!(passthrough_source(&build(0x83, 3, 16, 8)).is_none(), "PVRTC has no WebGPU format");
        assert!(passthrough_source(&build(0x88, 3, 16, 8)).is_none(), "BC4 is not a four-channel format");
        // A non-identity channel swizzle - applied during the decode, with no shader path.
        let mut s = build(0x87, 3, 16, 8);
        s.swizzle = 1 << 12;
        assert!(passthrough_source(&s).is_none(), "a permuted texture must not pass through raw");
        // A size that is not a multiple of the 4x4 block: WebGPU refuses to create it.
        assert!(passthrough_source(&build(0x87, 3, 14, 8)).is_none(), "14 is not a multiple of 4");
        // No guest mip chain on a texture big enough to need one, while the guest asks the
        // hardware to FILTER between levels - so the device really is reading levels we would
        // not be supplying.
        let mut m = build(0x87, 3, 16, 8);
        m.levels = 1;
        m.mip_filter = 1;
        assert!(passthrough_source(&m).is_none(), "level 0 alone is the white-speckle failure");
        // The same texture with mip filtering OFF is one the DEVICE samples from its base level
        // alone, so one level is the faithful answer and not a dropped chain. This is the
        // largest single texture in a measured race frame - 42.7 MB as RGBA8, 8 MB as blocks.
        m.mip_filter = 0;
        assert!(
            passthrough_source(&m).is_some(),
            "a texture the hardware itself samples mipless has no chain to lose"
        );
        // ... but a 4x4 texture HAS no chain to have, so one level is the whole truth.
        let mut tiny = build(0x87, 3, 4, 4);
        tiny.levels = 1;
        assert!(passthrough_source(&tiny).is_some(), "a 4x4 texture's chain is one level");
        // A cube map: how six chains interleave is not established.
        let mut cube = build(0x87, 2, 16, 8);
        cube.faces = 6;
        assert!(passthrough_source(&cube).is_none(), "a cube map's chain layout is not established");
    }

    /// A refusal falls through to the TRANSCODE, not to the plain decode.
    ///
    /// This is where the megabytes actually are. Two 4096x2048 UBC2 surfaces in a measured race
    /// frame declare one mip level while asking the hardware to filter between levels - the
    /// passthrough must refuse them, and letting them land as RGBA8 costs 45.3 MB against the
    /// 11.3 MB a re-encode with a generated chain costs. The distinction the test protects is
    /// that a refusal is a decision about WHICH compressed path, never a decision to expand.
    #[test]
    fn a_refused_passthrough_falls_through_to_the_transcode() {
        let (w, h) = (16u32, 8u32);
        let levels = max_mip_levels(w, h);
        let total = level_offset(0x87, 3, w, h, levels).unwrap() as usize;
        let mut t = tex(0x87, 0, w, h, w.div_ceil(4) * 16, vec![7u8; total]);
        t.tex_type = 3;
        t.levels = 1;
        t.mip_filter = 1;
        t.face_bytes = total as u32;
        assert!(passthrough_source(&t).is_none(), "the passthrough must refuse this");
        let c = compressed_source(&t, None).expect("but it must still reach the GPU compressed");
        assert!(c.transcoded, "and it must be labelled as re-encoded, not as guest blocks");
        assert_eq!(c.levels, levels, "the transcode owns the encode, so it supplies the chain");
    }

    /// The PVRTC transcode must produce a texture the GPU can actually be given: one block
    /// format for the whole chain, a FULL chain down to 1x1, and each level exactly its own
    /// block count in bytes.
    ///
    /// The chain length is the part worth pinning. The passthrough carries the guest's levels
    /// and refuses when there are none; this path owns the encode, so it can and must build the
    /// rest - a transcode that quietly shipped one level would be the white-speckle failure with
    /// a different cause. Here the source declares ONE level, and the result must still be the
    /// full seven.
    #[test]
    fn a_transcode_builds_a_full_chain_even_from_a_mipless_source() {
        // A PVRTC1 4bpp (0x83) source. The bytes are arbitrary - what is under test is the
        // chain's SHAPE, and the decoder's own correctness has its own tests.
        let (w, h) = (64u32, 64u32);
        let level0 = level_layout(0x83, 0, w, h, 0).unwrap();
        let mut t = tex(0x83, 0, w, h, level0.stride, vec![0x5Au8; level0.bytes as usize]);
        t.tex_type = 0;
        t.levels = 1;
        t.face_bytes = level0.bytes;
        let c = transcoded_source(&t, None).expect("PVRTC is exactly what this path is for");
        assert_eq!(c.levels, max_mip_levels(w, h), "the chain must reach 1x1");
        assert_eq!(c.levels, 7);
        let bb = c.format.block_bytes();
        let want: usize = (0..c.levels)
            .map(|l| {
                let (lw, lh) = ((w >> l).max(1), (h >> l).max(1));
                (lw.div_ceil(4) * lh.div_ceil(4) * bb) as usize
            })
            .sum();
        let bytes = c.cpu_bytes().expect("the BC target is always encoded on the CPU");
        assert_eq!(bytes.len(), want, "every level is its own block count, no more and no less");
    }

    /// The transcode picks ONE format for the whole texture, and picks it by whether any level
    /// has alpha.
    ///
    /// A mip chain is a single GPU texture, so a level that happens to be opaque cannot be BC1
    /// while its neighbour is BC3 - and the decision has to look at every level, because a
    /// box-filtered level can carry alpha that level 0's corner texels only hinted at.
    #[test]
    fn a_transcode_picks_one_format_for_the_whole_chain() {
        let (w, h) = (16u32, 16u32);
        let l0 = level_layout(0x83, 0, w, h, 0).unwrap();
        // An all-ones PVRTC block decodes opaque, so this exercises the BC1 arm; the point of
        // the assert is that the format does not vary level to level, whichever arm it takes.
        let mut t = tex(0x83, 0, w, h, l0.stride, vec![0xFFu8; l0.bytes as usize]);
        t.tex_type = 0;
        t.levels = 1;
        t.face_bytes = l0.bytes;
        let c = transcoded_source(&t, None).unwrap();
        let bb = c.format.block_bytes();
        assert!(matches!(c.format, BlockFormat::Bc1 | BlockFormat::Bc3));
        // Consistency check: the byte total is only reachable with ONE block size throughout.
        let want: usize = (0..c.levels)
            .map(|l| (((w >> l).max(1).div_ceil(4)) * ((h >> l).max(1).div_ceil(4)) * bb) as usize)
            .sum();
        assert_eq!(c.cpu_bytes().expect("the BC target is always encoded on the CPU").len(), want);
    }

    /// >>> A COMPRESSED UPLOAD IS ALWAYS THE GUEST'S OWN RESOLUTION. NEVER SMALLER.
    ///
    /// # The test that would have caught the worst regression in this file's history
    /// A previous version served a REDUCED-resolution encode while a full one was built over
    /// later frames, so a screen transition would not freeze. On the target device the reduced
    /// version never got past 128 texels a side, so a 2048x2048 atlas rendered at a sixteenth of
    /// its resolution per axis - for the whole run. It shipped, because every test asked whether
    /// the encode was well FORMED and none asked whether it was the right SIZE.
    ///
    /// Compression may change how bytes are stored. It may not change how much of the picture
    /// there is. If an encode cannot be afforded it is refused and the texture is decoded at full
    /// resolution instead - `None` here, never a smaller texture.
    #[test]
    fn a_compressed_upload_is_never_smaller_than_the_guests_texture() {
        for (w, h, levels) in [
            (64u32, 64u32, 6u32),
            (256, 256, 8),
            (512, 256, 8),
            (1024, 1024, 10),
        ] {
            let total = level_offset(0x83, 0, w, h, levels).unwrap() as usize;
            let pixels: Vec<u8> = (0..total).map(|i| ((i * 31 + 7) % 251) as u8).collect();
            let mut t = tex(0x83, 0, w, h, w.div_ceil(4), pixels);
            t.tex_type = 0;
            t.levels = levels;
            t.face_bytes = total as u32;

            // A refusal is allowed. A SMALLER texture is not.
            if let Some(c) = transcoded_source(&t, None) {
                assert_eq!((c.width, c.height), (w, h), "{w}x{h} was encoded at a reduced size");
                assert_eq!(
                    c.levels,
                    max_mip_levels(w, h),
                    "{w}x{h} must carry its whole chain, down to the last level"
                );
                // And the bytes must actually be there for every level of that full chain - a
                // correct header over a short buffer is the same defect wearing a disguise.
                let bb = c.format.block_bytes();
                let want: usize = (0..c.levels)
                    .map(|l| {
                        let (lw, lh) = ((w >> l).max(1), (h >> l).max(1));
                        (lw.div_ceil(4) * lh.div_ceil(4) * bb) as usize
                    })
                    .sum();
                let bytes = c.cpu_bytes().expect("the BC target is always encoded on the CPU");
                assert_eq!(bytes.len(), want, "{w}x{h} is short of its own declared chain");
            }
        }
    }

    /// The passthrough carries the guest's blocks, so it is full resolution by construction - but
    /// assert it, because `CompressedUpload::width` is a field somebody could fill in wrongly.
    #[test]
    fn a_passthrough_reports_the_guests_own_size() {
        let (w, h, levels) = (256u32, 256u32, 9u32);
        let bb = 8u32;
        let total = level_offset(0x85, 0, w, h, levels).unwrap() as usize;
        let pixels: Vec<u8> = (0..total).map(|i| ((i * 17 + 3) % 249) as u8).collect();
        let mut t = tex(0x85, 0, w, h, w.div_ceil(4) * bb, pixels);
        t.tex_type = 0;
        t.levels = levels;
        t.face_bytes = total as u32;
        if let Some(c) = passthrough_source(&t) {
            assert_eq!((c.width, c.height), (w, h));
            assert!(!c.transcoded, "the guest's own blocks are not a re-encode");
        }
    }

    /// A format that is neither a WebGPU block format nor PVRTC is left alone entirely.
    #[test]
    fn an_ordinary_uncompressed_texture_is_not_transcoded() {
        let t = tex(0x0c, 0, 16, 16, 64, vec![9u8; 16 * 16 * 4]);
        assert!(transcoded_source(&t, None).is_none(), "U8U8U8U8 has nothing to transcode");
        assert!(compressed_source(&t, None).is_none());
    }

    /// The block-wise decode path must produce EXACTLY what the per-texel path produces.
    ///
    /// [`decode_face_fast`] exists only to stop re-decoding each 4x4 block sixteen
    /// times; it is not allowed to be a different decoder. The per-texel function is the
    /// oracle, so this compares the two over every block-compressed format, both swizzled
    /// (Morton) and linear addressing, and at sizes that are NOT multiples of the block -
    /// the partial trailing blocks are where a block walker goes wrong.
    #[test]
    fn blockwise_decode_matches_per_texel() {
        // BC1 (0x85, 8-byte blocks), BC2 (0x86) and BC3 (0x87, 16-byte blocks).
        for &(fmt, block_bytes) in &[(0x85u32, 8usize), (0x86, 16), (0x87, 16)] {
            for &(w, h) in &[(4u32, 4u32), (8, 8), (16, 8), (7, 5), (13, 3)] {
                for &tex_type in &[3u32 /* linear */, 0 /* swizzled */] {
                    for &chan_swizzle in &[0u32, 1, 3] {
                        let bx = w.div_ceil(4).next_power_of_two() as usize;
                        let by = h.div_ceil(4).next_power_of_two() as usize;
                        // Enough bytes for the padded Morton grid, filled with a pattern
                        // that varies per byte so every endpoint and index bit differs.
                        let n = bx * by * block_bytes + block_bytes;
                        let pixels: Vec<u8> =
                            (0..n).map(|i| ((i * 37 + 11) % 251) as u8).collect();
                        let mut t = tex(fmt, chan_swizzle, w, h, w.div_ceil(4) * block_bytes as u32, pixels);
                        t.tex_type = tex_type;
                        let (_, _, fast) = decode_texture_rgba8(&t);
                        let mut slow = Vec::with_capacity(fast.len());
                        for y in 0..h {
                            for x in 0..w {
                                slow.extend_from_slice(&texel_rgba_face(&t, 0, x, y));
                            }
                        }
                        assert_eq!(
                            fast, slow,
                            "block-wise decode differs from per-texel for format {fmt:#x} \
                             {w}x{h} tex_type {tex_type} swizzle {chan_swizzle}"
                        );
                    }
                }
            }
        }
    }

    /// The whole-image PVRTC decode must produce EXACTLY what the per-texel path produces.
    ///
    /// Same contract as the two above, over the family that carries 47% of everything a race
    /// frame decodes. PVRTC is the awkward one: a texel reads five blocks, the upscale wraps
    /// at the edges, and the whole-image pass walks the SHIFTED grid rather than the texel
    /// grid - so an off-by-half-a-block there would still fill every texel, just with its
    /// neighbour's colours. Only a comparison against the oracle catches that.
    ///
    /// Covers both variants (PVRTC1 `0x80`/`0x81`, PVRTC2 `0x82`/`0x83`), both bit rates
    /// (whose blocks are 4x4 and 8x4), both addressing modes, and sizes that are not
    /// multiples of the block, including one narrower than a single block.
    #[test]
    fn pvrtc_whole_image_matches_per_texel() {
        for fmt in [0x80u32, 0x81, 0x82, 0x83] {
            for &(w, h) in &[(8u32, 8u32), (16, 16), (32, 8), (8, 4), (12, 6), (5, 3)] {
                for &tex_type in &[3u32 /* linear */, 0 /* swizzled */] {
                    let variant = crate::pvrtc::Variant::from_base_format(fmt).unwrap();
                    let (bw, bh) = variant.block_size();
                    // Enough bytes for the padded Morton block grid, filled with a pattern
                    // that varies per byte so the colours, the flags and every modulation
                    // code differ block to block.
                    let bx = w.div_ceil(bw).next_power_of_two() as usize;
                    let by = h.div_ceil(bh).next_power_of_two() as usize;
                    let n = (bx * by + 2) * 8;
                    let pixels: Vec<u8> = (0..n).map(|i| ((i * 41 + 17) % 251) as u8).collect();
                    let mut t = tex(fmt, 0, w, h, w, pixels);
                    t.tex_type = tex_type;
                    let (_, _, fast) = decode_texture_rgba8(&t);
                    let mut slow = Vec::with_capacity(fast.len());
                    for y in 0..h {
                        for x in 0..w {
                            slow.extend_from_slice(&texel_rgba_face(&t, 0, x, y));
                        }
                    }
                    assert_eq!(
                        fast, slow,
                        "whole-image PVRTC decode differs from per-texel for format \
                         {fmt:#x} {w}x{h} tex_type {tex_type}"
                    );
                }
            }
        }
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
        // The regression that motivated the swizzle fix: a "Loading" text
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
    fn u8_single_channel_swizzle1() {
        // A single-channel U8 (base 0x00) coverage texel routes to RGBA per SWIZZLE1.
        // Regression guard: the selector must be read once (not double-shifted), or a
        // font atlas's RRRR coverage would decode to red [r,0,0,255] instead of grey.
        let c = 200u8;
        // The three modes retail titles are MEASURED to use, and the roles that make them
        // meaningful. These are what pin the high-to-low reading of the selector name; see
        // `swizzle1`'s own doc for the measurement and for why the header cannot settle it.
        //
        // R111 (7): white RGB with the byte as ALPHA - a glyph atlas. Decoding this as
        // `[c,255,255,255]` is what painted PCSA00009's every string as a cyan box.
        let t = tex(0x00, 7, 1, 1, 1, vec![c]);
        assert_eq!(sample_texture(&t, 0.5, 0.5), [255, 255, 255, c]);
        // R000 (6): black with the byte as alpha - a mask. The mirror of R111, and the mode
        // this title binds a thousand times a frame; the other reading makes every one of
        // those bindings alpha-zero.
        let t = tex(0x00, 6, 1, 1, 1, vec![c]);
        assert_eq!(sample_texture(&t, 0.5, 0.5), [0, 0, 0, c]);
        // 1RRR (5): opaque greyscale - a luminance/detail map.
        let t = tex(0x00, 5, 1, 1, 1, vec![c]);
        assert_eq!(sample_texture(&t, 0.5, 0.5), [c, c, c, 255]);
        // RRRR (3): the byte in all four channels. Symmetric - it reads the same either way
        // round, which is why it could never have caught the mirrored table.
        let t = tex(0x00, 3, 1, 1, 1, vec![c]);
        assert_eq!(sample_texture(&t, 0.5, 0.5), [c, c, c, c]);
        // R (0): one named channel, so also direction-free - the byte in red, opaque.
        let t = tex(0x00, 0, 1, 1, 1, vec![c]);
        assert_eq!(sample_texture(&t, 0.5, 0.5), [c, 0, 0, 255]);
    }

    #[test]
    fn linear_filter_bilerps() {
        // A 2x1 U8 RRRR texture [0, 200] with LINEAR magnify: sampling the midpoint
        // between the two texel centers yields the average, not a nearest texel.
        let mut t = tex(0x00, 3, 2, 1, 2, vec![0, 200]);
        t.mag_filter = 1; // SCE_GXM_TEXTURE_FILTER_LINEAR
        // Texel centers at u=0.25 and u=0.75; the midpoint u=0.5 averages to ~100.
        let mid = sample_texture(&t, 0.5, 0.0);
        assert!((mid[0] as i32 - 100).abs() <= 1, "expected ~100, got {}", mid[0]);
        // POINT (default) at the same point snaps to one texel (0 or 200), not ~100.
        let p = tex(0x00, 3, 2, 1, 2, vec![0, 200]);
        assert!(sample_texture(&p, 0.5, 0.0)[0] == 0 || sample_texture(&p, 0.5, 0.0)[0] == 200);
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
        let face_bytes = pixels.len() as u32;
        BoundTexture { pixels_id: crate::capture::next_pixels_id(), unit: 0, base_format, swizzle: 0, tex_type, width: w, height: h, stride, faces: 1, face_bytes, levels: 1, data_addr: 0, pixels: pixels.into(), u_addr_mode: 0, v_addr_mode: 0, lod_bias: 0, min_filter: 0, mag_filter: 0, mip_filter: 0, gamma: 0 }
    }

    #[test]
    fn bc2_block_alpha_and_color() {
        // One BC2 (DXT3) 4x4 block: 8 bytes 4-bit alpha, then a BC1 color sub-block.
        // Color endpoint c0 = white 565 with all-zero indices, so every texel is
        // white; alpha byte 0 holds texel 0 (low nibble = 15) and texel 1 (high
        // nibble = 0). This is the common menu-font case: white RGB, glyph coverage
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

    /// UNCOMPRESSED textures are Morton-addressed when swizzled too, which the decoder once
    /// applied only to the block-compressed path. The failure it caused is worth naming: a
    /// retail title's small cyan-on-black UI panels rendered as blocky two-colour static,
    /// which reads as deliberate "data readout" art rather than a bug, because a permutation
    /// preserves every colour and only moves it. Nothing else on those screens showed it -
    /// the rest were LINEAR, solid 8x8 fills, or block-compressed.
    #[test]
    fn u8x4_swizzled_texel_addressing() {
        // An 8x8 U8U8U8U8 swizzled texture. Texel (2,1) sits at Morton index 9 but linear
        // index 1*8+2 = 10, so the two readings disagree and the test can tell them apart.
        let (w, h) = (8u32, 8u32);
        let mut px = vec![0u8; (w * h * 4) as usize];
        let target = morton_index(2, 1, w, h) as usize;
        assert_eq!(target, 9);
        assert_ne!(target, (1 * w + 2) as usize, "row-major and Morton must differ here");
        px[target * 4] = 0xFF; // mark that texel's first channel
        let t = tex_typed(0x0c, 0, w, h, w * 4, px);
        assert_eq!(sample_texture(&t, u_of(2, 8), u_of(1, 8))[0], 0xFF);
        // The texel a row-major reader would have returned instead must stay clear, so a
        // regression cannot pass by marking everything.
        assert_eq!(sample_texture(&t, u_of(1, 8), u_of(1, 8))[0], 0x00);
    }

    /// The LINEAR path must stay row-major - the swizzle fix above applies only to swizzled
    /// types, and most correctly-rendering UI textures are LINEAR.
    #[test]
    fn u8x4_linear_texel_addressing_is_row_major() {
        let (w, h) = (8u32, 8u32);
        let mut px = vec![0u8; (w * h * 4) as usize];
        px[((1 * w + 2) * 4) as usize] = 0xFF;
        let t = tex_typed(0x0c, 3, w, h, w * 4, px); // LINEAR
        assert_eq!(sample_texture(&t, u_of(2, 8), u_of(1, 8))[0], 0xFF);
    }
}

#[cfg(test)]
mod morton_table_tests {
    use super::*;

    /// The tables are an OPTIMISATION of `morton_index`, so the only thing worth asserting is
    /// that they are the SAME FUNCTION - over both leftover-tiling branches (wider-than-tall
    /// and taller-than-wide), the square case, and non-power-of-two extents where the padded
    /// grid is larger than the level.
    #[test]
    fn morton_tables_reproduce_morton_index_exactly() {
        for &(w, h) in &[
            (1u32, 1u32),
            (8, 8),
            (16, 4),
            (4, 16),
            (32, 8),
            (5, 3),
            (13, 27),
            (64, 64),
            (128, 32),
            (7, 1),
            (1, 7),
        ] {
            let (pw, ph) = (w.next_power_of_two(), h.next_power_of_two());
            let (xs, ys) = morton_tables(pw, ph, w, h);
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(
                        xs[x as usize] + ys[y as usize],
                        morton_index(x, y, pw, ph),
                        "level {w}x{h} (padded {pw}x{ph}) at ({x}, {y})"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod png_tests {
    use super::*;

    /// The decoder exists to read this module's own output, so the round trip is the
    /// whole specification.
    #[test]
    fn png_round_trips_through_encode_and_decode() {
        let (w, h) = (7u32, 3u32);
        let rgba: Vec<u8> = (0..(w * h * 4)).map(|i| (i * 7 % 251) as u8).collect();
        let png = rgba_to_png(w, h, &rgba);
        let (dw, dh, out) = png_to_rgba(&png).expect("decode");
        assert_eq!((dw, dh), (w, h));
        assert_eq!(out, rgba);
    }

    /// A big image spans several 64 KiB stored blocks; the block walk must not stop
    /// at the first one.
    #[test]
    fn png_round_trips_across_multiple_stored_blocks() {
        let (w, h) = (200u32, 120u32); // 200*4+1 per row * 120 = 96120 bytes > 65535
        let rgba: Vec<u8> = (0..(w * h * 4)).map(|i| (i % 256) as u8).collect();
        let (dw, dh, out) = png_to_rgba(&rgba_to_png(w, h, &rgba)).expect("decode");
        assert_eq!((dw, dh), (w, h));
        assert_eq!(out, rgba);
    }

    /// The F16 encoder against the decoder that already reads real guest data, over EVERY
    /// finite half - so the round trip is exhaustive rather than sampled.
    ///
    /// Exhaustive matters here because the interesting cases are the ones a hand-picked list
    /// misses: the subnormal range below 2^-14 (where the implicit leading one has to be
    /// shifted back in) and the rounding ties, which is where a plausible-looking encoder
    /// quietly loses a bit.
    #[test]
    fn every_finite_half_survives_a_round_trip_through_f32() {
        for h in 0u16..=0xffff {
            if (h >> 10) & 0x1f == 0x1f {
                continue; // Inf/NaN: not a value equality holds for
            }
            let back = f32_to_half(half_to_f32(h));
            assert_eq!(back, h, "half {h:#06x} -> {} -> {back:#06x}", half_to_f32(h));
        }
    }

    /// The encoder's edges, stated as values rather than as bit patterns.
    #[test]
    fn f32_to_half_saturates_and_rounds() {
        assert_eq!(f32_to_half(1.0), 0x3c00);
        assert_eq!(f32_to_half(0.25), 0x3400);
        assert_eq!(f32_to_half(-0.25), 0xb400);
        // Past the half range, in both directions.
        assert_eq!(f32_to_half(1.0e6), 0x7c00);
        assert_eq!(f32_to_half(-1.0e6), 0xfc00);
        // Below the smallest subnormal: flushes to a signed zero, not to a wrong tiny value.
        assert_eq!(f32_to_half(1.0e-12), 0x0000);
        assert_eq!(f32_to_half(-1.0e-12), 0x8000);
        // Round to nearest, ties to even, on the first bit the half cannot hold.
        assert_eq!(half_to_f32(f32_to_half(0.3)), half_to_f32(0x34cd));
    }

    /// Anything outside the narrow supported shape must NAME what it found rather
    /// than decode part of an image.
    #[test]
    fn png_decode_rejects_what_it_cannot_read() {
        assert!(png_to_rgba(b"not a png at all").unwrap_err().contains("signature"));
        let mut png = rgba_to_png(2, 2, &[0u8; 16]);
        // Flip the colour type to greyscale (0) in IHDR: byte 8+8+4+4+1 = 25.
        png[25] = 0;
        let err = png_to_rgba(&png).unwrap_err();
        assert!(err.contains("color=0"), "error should name the field: {err}");
    }
}
