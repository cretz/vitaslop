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
/// strip and fan all rasterize to triangles; lines/points/edges are not drawn.
const PRIM_TRIANGLES: u32 = 0x0000_0000;
const PRIM_TRIANGLE_STRIP: u32 = 0x0C00_0000;
const PRIM_TRIANGLE_FAN: u32 = 0x1000_0000;

/// SceGxmCullMode: which screen-space winding the GPU discards. NONE draws both
/// faces; CW/CCW discard clockwise/counter-clockwise triangles respectively.
const SCE_GXM_CULL_NONE: u32 = 0x0000_0000;
const SCE_GXM_CULL_CW: u32 = 0x0000_0001;
const SCE_GXM_CULL_CCW: u32 = 0x0000_0002;

/// SCE_GXM_DEPTH_WRITE_DISABLED - depth writes off (a 2D alpha overlay, not opaque 3D).
const SCE_GXM_DEPTH_WRITE_DISABLED: u32 = 0x0010_0000;

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
        // 64-bit four/two-channel: F16F16F16F16, U16U16U16U16, S16S16S16S16, F32F32, U32U32.
        0x1b..=0x1f => (1, 1, 8),
        // BC1 (DXT1) and BC4 (both signs): 8-byte 4x4 blocks.
        0x85 | 0x88 | 0x89 => (4, 4, 8),
        // BC2 (DXT3), BC3 (DXT5), BC5 (both signs): 16-byte 4x4 blocks.
        0x86 | 0x87 | 0x8a | 0x8b => (4, 4, 16),
        _ => return None,
    })
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
pub fn decode_texture_rgba8(t: &BoundTexture) -> (u32, u32, Vec<u8>) {
    if t.width == 0 || t.height == 0 {
        return (1, 1, vec![255, 0, 255, 255]);
    }
    // A cube map decodes to its six faces stacked in `BoundTexture::faces` order, which is the
    // layer order the GPU binds them in.
    let faces = t.faces.max(1);
    let mut rgba = Vec::with_capacity((t.width * t.height * faces * 4) as usize);
    for f in 0..faces {
        for y in 0..t.height {
            for x in 0..t.width {
                rgba.extend_from_slice(&texel_rgba_face(t, f, x, y));
            }
        }
    }
    (t.width, t.height, rgba)
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
        return [255, 0, 255, 255];
    };
    let face_base = (face * t.face_bytes) as usize;
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
    let off = face_base + (y * t.stride + x * block_bytes) as usize;
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
        // 64-bit four-channel. The four lanes sit in memory order exactly as the 8888 case's
        // four bytes do, so the same SWIZZLE4 selector permutes them. Each lane is reduced to
        // 8 bits for the shared RGBA8 texture seam: F16 saturates to [0,1] (these are HDR
        // lookup tables - a value above 1.0 clamps, which the seam cannot represent), while
        // U16/S16 are normalized ranges that map exactly.
        0x1b | 0x1c | 0x1d => {
            let lane = |i: usize| -> u8 {
                let raw = u16::from_le_bytes([byte(i * 2), byte(i * 2 + 1)]);
                let v = match t.base_format {
                    0x1b => half_to_f32(raw),
                    0x1c => raw as f32 / 65535.0,
                    _ => ((raw as i16) as f32 / 32767.0).max(0.0),
                };
                (v.clamp(0.0, 1.0) * 255.0).round() as u8
            };
            swizzle4(lane(0), lane(1), lane(2), lane(3), swizzle)
        }
        // Single channel U8/S8 (fonts, coverage masks): route the one channel to RGBA
        // per the format's SWIZZLE1 selector. Font atlases are typically RRRR (coverage
        // in every channel, so alpha carries it) or 000R/111R (coverage in alpha);
        // forcing alpha to 255 would turn the transparent inter-glyph gaps into opaque
        // boxes that overwrite neighbouring glyphs.
        0x00 | 0x01 => swizzle1(byte(0), swizzle),
        _ => [255, 0, 255, 255], // unknown format: opaque magenta
    }
}

/// Route a single-channel (U8/S8) texel to straight RGBA per its GXM `SWIZZLE1`
/// selector (already reduced to `(format >> 12) & 0x7` by the caller, exactly as
/// `swizzle4` receives its selector). Each output channel is either the channel byte
/// `r`, constant 0, or constant 255 (`1`), in the order the selector names RGBA (e.g.
/// `111R` = white RGB with the byte as alpha, `RRRR` = the byte in all four).
/// `SWIZZLE1_R` (0) maps to R in red, opaque.
fn swizzle1(r: u8, swizzle: u32) -> [u8; 4] {
    match swizzle {
        0 => [r, 0, 0, 255],     // R
        1 => [0, 0, 0, r],       // 000R
        2 => [255, 255, 255, r], // 111R
        3 => [r, r, r, r],       // RRRR
        4 => [0, r, r, r],       // 0RRR
        5 => [255, r, r, r],     // 1RRR
        6 => [r, 0, 0, 0],       // R000
        _ => [r, 255, 255, 255], // R111 (7)
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

/// Latch for the once-per-run notice that shader-expanded draws are being skipped.
static SKIPPED_EXPANDED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Print `msg` the first time this latch is raised. Used for "the renderer cannot
/// reproduce this class of draw" notices, which are properties of the title's shaders and
/// so would otherwise repeat every frame forever.
fn report_once(latch: &std::sync::atomic::AtomicBool, msg: &str) {
    if !latch.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!("{msg}");
    }
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

/// The rasterizer core. `width`/`height` are the RASTER dimensions and `ssaa` the supersample
/// factor those already fold in (so Pixel-space draws scale by it - see [`project`]); the
/// caller downsamples the result. `clear` is the background color.
fn render_scene_raster(scene: &Scene, width: u32, height: u32, clear: [u8; 4], ssaa: u32) -> Framebuffer {
    let ssaa = ssaa.max(1) as f32;
    let mut fb = Framebuffer::new(width, height, clear);
    let mut depth = vec![f32::INFINITY; (width * height) as usize];

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
        let texture = if textured { d.albedo() } else { None };
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
            raster_triangle(&mut fb, &mut depth, &screen, &verts, texture, uv_div, depth_test, depth_func, d.exposure, &d.material, &d.world, trace, di, uv_debug);
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
    depth_func: u32,
    exposure: f32,
    material: &crate::capture::FragmentMaterial,
    world: &[f32; 16],
    trace: Option<(i32, i32)>,
    draw_idx: usize,
    uv_debug: bool,
) {
    let (w, h) = (fb.width as i32, fb.height as i32);
    // World-space normals at the three vertices (constant per triangle), interpolated per
    // pixel below for the lit opaque path. Object-space normals are brought to world space
    // by the draw's model-to-world matrix so N.L matches the world-space light direction.
    let wn: [[f32; 3]; 3] =
        [world_normal(verts[0].normal, world), world_normal(verts[1].normal, world), world_normal(verts[2].normal, world)];
    let has_normal = verts.iter().any(|v| v.normal != [0.0, 0.0, 0.0]);
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
        // Carry the magnification filter so the GPU picks the matching sampler (LINEAR ->
        // bilinear, as the software `sample_texture_bilinear` does). SceGxmTextureFilter:
        // 1 = LINEAR, 0 = POINT.
        let filter_linear = t.mag_filter == 1;
        let g =
            GxmTexture { key, width, height, faces: t.faces.max(1), rgba: Arc::new(rgba), filter_linear };
        self.decode_cache.insert(key, g.clone());
        g
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
        let mut draws = Vec::with_capacity(scene.draws.len());
        // Visible opaque depth range (post-divide c.z/c.w), accumulated across draws for
        // the GPU's linear depth normalization.
        let mut dmin = f32::INFINITY;
        let mut dmax = f32::NEG_INFINITY;
        // Diagnostic: VITASLOP_DRAW_STATS also reports each opaque draw's own visible depth
        // span, which is what the GPU's normalization has to keep separable.
        let stats = std::env::var("VITASLOP_DRAW_STATS").is_ok();
        for (di, d) in scene.draws.iter().enumerate() {
            // A list emits idx/3 triangles; a strip or fan emits idx-2. Any other topology
            // (lines/points) emits none and is skipped.
            let tri_count = triangle_count(d);
            if tri_count == 0 {
                continue;
            }
            let interp = interpret_draw(d);
            if interp.skip {
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

            // The largest index referenced, so the vertex buffer covers every index
            // (an out-of-range index decodes to a zero vertex, matching the software
            // path's clamped reads, rather than a GPU out-of-bounds fetch).
            let mut max_idx = 0usize;
            for i in 0..d.index_count as usize {
                max_idx = max_idx.max(index_at(d, i));
            }
            let stride = d.vertex_stride.max(1) as usize;
            let nverts = (d.vertices.len() / stride).max(max_idx + 1);

            // Screen positions of every vertex for MVP draws, so the cull test and
            // behind-eye drop below reuse one projection per vertex (not per triangle).
            // `project` applies the same Y-flip the software rasterizer uses; only the
            // winding SIGN matters here, so any positive surface size gives the identical
            // cull decision as the real target. `None` = behind the eye (w <= 0).
            let mut screen_pos: Vec<Option<[f32; 4]>> =
                if mvp.is_some() { Vec::with_capacity(nverts) } else { Vec::new() };

            let (mut draw_dmin, mut draw_dmax) = (f32::INFINITY, f32::NEG_INFINITY);
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
                // World-space normal for the opaque lighting term, baked here (object normal
                // through the draw's model-to-world matrix) so the GPU shader uses it directly
                // and matches the software rasterizer's `world_normal` exactly. A mesh with no
                // normal yields `[0,1,0]` (up), the same fallback the software path uses.
                let wn = world_normal(v.normal, &d.world);
                vertices.extend_from_slice(&wn[0].to_le_bytes());
                vertices.extend_from_slice(&wn[1].to_le_bytes());
                vertices.extend_from_slice(&wn[2].to_le_bytes());
                if mvp.is_some() {
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
            let mut indices = Vec::with_capacity(tri_count * 3 * 4);
            for t in 0..tri_count {
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

            let texture =
                if interp.textured { d.albedo().map(|t| self.texture(t)) } else { None };

            // When the runtime captured the raw shader blobs (recompiler path enabled),
            // attach everything the GXP->WGSL recompiler needs to draw this call with the
            // guest's real shaders. The renderer links + caches a pipeline and falls back to
            // the fixed-function fields above on any link/format error. We carry the RAW guest
            // vertex/index buffers (not the culled canonical ones) so the recompiled pipeline
            // does its own attribute fetch + facing cull.
            let gxp = if !d.vprog.is_empty() {
                let attributes = d
                    .attributes
                    .iter()
                    .map(|a| vitaslop_platform::gpu::GxpAttr {
                        reg_index: a.reg_index,
                        offset: a.offset,
                        gxm_format: a.format,
                        components: a.component_count,
                    })
                    .collect();
                let textures = d
                    .textures
                    .iter()
                    .map(|t| vitaslop_platform::gpu::GxpTex { unit: t.unit as u8, tex: self.texture(t) })
                    .collect();
                // Expand the guest topology into a flat, winding-normalized triangle-LIST u32
                // index buffer (NO CPU cull - the recompiled pipeline culls on the GPU via the
                // guest cull mode, using its own real-shader projection). Indexes into the RAW
                // guest vertex stream `d.vertices`.
                let mut gxp_indices = Vec::with_capacity(tri_count * 3 * 4);
                for t in 0..tri_count {
                    for k in tri_indices(d, t) {
                        gxp_indices.extend_from_slice(&(k as u32).to_le_bytes());
                    }
                }
                let gxp_index_count = (gxp_indices.len() / 4) as u32;
                Some(vitaslop_platform::gpu::GxpRecompile {
                    vprog: d.vprog.clone(),
                    fprog: d.fprog.clone(),
                    vert_sa: d.vert_sa.clone(),
                    frag_sa: d.frag_sa.clone(),
                    vertices: d.vertices.clone(),
                    vertex_stride: d.vertex_stride,
                    attributes,
                    indices: gxp_indices,
                    index_count: gxp_index_count,
                    index_u32: true,
                    primitive: d.primitive,
                    textures,
                    depth_write: d.render_state.front_depth_write != SCE_GXM_DEPTH_WRITE_DISABLED,
                    depth_func: d.render_state.front_depth_func,
                    cull_mode: d.render_state.cull_mode,
                    blend: !opaque,
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
            });
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
        RenderScene { draws, depth_min, depth_scale }
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
            primitive: PRIM_TRIANGLE_STRIP,
            index_format: 0,
            index_count: indices.len() as u32,
            vertices: vec![],
            vertex_stride: 1,
            attributes: vec![],
            indices: indices.iter().flat_map(|i| i.to_le_bytes()).collect(),
            uniforms: vec![],
            textures: vec![],
            render_state: RenderState::default(),
            exposure: 1.0,
            material: crate::capture::FragmentMaterial::default(),
            world: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vprog: vec![],
            fprog: vec![],
            vert_sa: vec![],
            frag_sa: vec![],
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
        ];
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
            color: None,
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
        let found = locate_scene(&Scene { color: None, draws: vec![d.clone()] }, 100, 100);
        let h = found[0].heading.expect("an identity rotation has a heading");
        assert!((h[0] - 0.0).abs() < 1e-3, "local +X is bearing 0, got {}", h[0]);
        assert!((h[1] + 90.0).abs() < 1e-3, "local +Z is bearing -90, got {}", h[1]);

        // Rotate 90 degrees so local +X points along world -Z: bearing 90.
        let mut turned = d.clone();
        turned.world[0] = 0.0;
        turned.world[2] = -1.0;
        turned.world[8] = 1.0;
        turned.world[10] = 0.0;
        let found = locate_scene(&Scene { color: None, draws: vec![turned] }, 100, 100);
        let h = found[0].heading.unwrap();
        assert!((h[0] - 90.0).abs() < 1e-3, "expected bearing 90, got {}", h[0]);

        // A world matrix with no in-plane rotation at all reports no heading rather
        // than a fabricated zero.
        let mut flat = d;
        flat.world[0] = 0.0;
        flat.world[2] = 0.0;
        let found = locate_scene(&Scene { color: None, draws: vec![flat] }, 100, 100);
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

        let before = Scene { color: None, draws: vec![located_draw([0.0, 0.0, 0.0], &car, mvp)] };
        // Next frame: something new is submitted first, and the car has moved.
        let after = Scene {
            color: None,
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
        d.primitive = 0x0400_0000; // lines: not drawn
        assert_eq!(triangle_count(&d), 0);
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
            unit: 0, base_format: 0x0c, swizzle: 0, tex_type: 0, width: 1, height: 1, stride: 4,
            faces: 1, face_bytes: 4,
            pixels: vec![200, 100, 50, 255].into(), data_addr: 0, u_addr_mode: 0, v_addr_mode: 0,
            lod_bias: 0, min_filter: 0, mag_filter: 0,
        };
        let draw = Draw {
            primitive: PRIM_TRIANGLES, index_format: 0, index_count: 6,
            vertices: verts, vertex_stride: 16,
            attributes: vec![
                VertexAttribute { stream_index: 0, offset: 0, format: FORMAT_F32, component_count: 2, reg_index: 0 },
                VertexAttribute { stream_index: 0, offset: 8, format: FORMAT_F32, component_count: 2, reg_index: 1 },
            ],
            indices: [0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect(),
            uniforms: vec![], textures: vec![tex], render_state: RenderState::default(),
            exposure: 1.0, material: Default::default(), world: [0.0; 16],
            vprog: vec![], fprog: vec![], vert_sa: vec![], frag_sa: vec![], shader_expanded: false,
        };
        let scene = Scene { color: None, draws: vec![draw] };
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
            unit: 0, base_format: 0x0c, swizzle: 0, tex_type: 0, width: tw, height: th, stride: tw * 4,
            faces: 1, face_bytes: tw * th * 4,
            pixels: pixels.into(), data_addr: 0, u_addr_mode: 0, v_addr_mode: 0, lod_bias: 0, min_filter: 0, mag_filter: 0,
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
            primitive: PRIM_TRIANGLES, index_format: 0, index_count: 6,
            vertices: verts, vertex_stride: 16,
            attributes: vec![
                VertexAttribute { stream_index: 0, offset: 0, format: FORMAT_F32, component_count: 2, reg_index: 0 },
                VertexAttribute { stream_index: 0, offset: 8, format: FORMAT_F32, component_count: 2, reg_index: 1 },
            ],
            indices: [0u16, 1, 2, 0, 2, 3].iter().flat_map(|i| i.to_le_bytes()).collect(),
            uniforms: vec![], textures: vec![tex], render_state: RenderState::default(),
            exposure: 1.0, material: Default::default(), world: [0.0; 16],
            vprog: vec![], fprog: vec![], vert_sa: vec![], frag_sa: vec![], shader_expanded: false,
        };
        let s = Scene { color: None, draws: vec![draw] };
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
            data_addr: 0,
            pixels: pixels.into(),
            u_addr_mode: 0,
            v_addr_mode: 0,
            lod_bias: 0,
            min_filter: 0,
            mag_filter: 0,
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
        // RRRR (3): the byte in all four channels - a coverage mask carrying its own
        // alpha (what this title's UI font uses); must NOT come out red.
        let t = tex(0x00, 3, 1, 1, 1, vec![c]);
        assert_eq!(sample_texture(&t, 0.5, 0.5), [c, c, c, c]);
        // 111R (2): white RGB, coverage in alpha.
        let t = tex(0x00, 2, 1, 1, 1, vec![c]);
        assert_eq!(sample_texture(&t, 0.5, 0.5), [255, 255, 255, c]);
        // 000R (1): coverage in alpha only.
        let t = tex(0x00, 1, 1, 1, 1, vec![c]);
        assert_eq!(sample_texture(&t, 0.5, 0.5), [0, 0, 0, c]);
        // R (0): the byte in red, opaque.
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
        BoundTexture { unit: 0, base_format, swizzle: 0, tex_type, width: w, height: h, stride, faces: 1, face_bytes, data_addr: 0, pixels: pixels.into(), u_addr_mode: 0, v_addr_mode: 0, lod_bias: 0, min_filter: 0, mag_filter: 0 }
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
