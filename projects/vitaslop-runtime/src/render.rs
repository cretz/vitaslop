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

/// How much a scene's coordinate ORIGIN moved between two [`locate_scene`] reports, and
/// how much of the scene agreed about it.
///
/// # Why this is not optional
/// The matrix a title calls "model to world" need not be measured from a fixed origin. On
/// PCSA00027 it is measured from a frame that travels with the camera - so while the
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
        mix(d.albedo().map(tex_key).unwrap_or(0));
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
    /// The last reported "the whole scene was dropped" tally, so [`DropTally::report`]
    /// prints when the shape CHANGES rather than sixty times a second.
    last_empty: Option<DropTally>,
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
        eprintln!(
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
    let detail = format!(
        "render: DROPPED draw {di} - {}. tris={tri_count}, stride={}, {} attributes, \
         shaders={}. This draw is MISSING from the frame; the guest asked for it.",
        kind.describe(),
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
    eprintln!("{detail}");
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
        RenderSceneBuilder { decode_cache: HashMap::new(), last_empty: None }
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
        let g = GxmTexture {
            key,
            data_addr: t.data_addr,
            width,
            height,
            faces: t.faces.max(1),
            rgba: Arc::new(rgba),
            filter_linear,
        };
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
        let mut tally = DropTally { total: scene.draws.len(), ..Default::default() };
        for (di, d) in scene.draws.iter().enumerate() {
            // A list emits idx/3 triangles; a strip or fan emits idx-2. Any other topology
            // (lines/points) emits none and is skipped.
            let tri_count = triangle_count(d);
            if tri_count == 0 {
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
                        d.uniforms.len(),
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
                shader_only,
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
        tally.report_if_total(&mut self.last_empty);
        // Carry where this scene draws to, so a renderer can keep the result addressable
        // for a later pass that samples it (see `RttTarget`).
        let target = scene.color.map(|c| vitaslop_platform::gpu::RttTarget {
            data_addr: c.data_addr,
            width: c.width,
            height: c.height,
        });
        RenderScene { draws, target, depth_min, depth_scale }
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
        d.attributes[0].format = FORMAT_F32;
        d.attributes[1].format = FORMAT_U8N;
        if !depth_write {
            d.render_state.front_depth_write = SCE_GXM_DEPTH_WRITE_DISABLED;
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
            color: None,
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
        let scene = Scene { color: None, draws: vec![ground_quad(0.0, -50.0, -50.0, 50.0, 50.0, true), sky] };
        let map = render_map(&scene, square_view([-50.0, -50.0, 50.0, 50.0], 40), [0, 0, 0, 255], 1, None, [0.0; 3]);
        assert_eq!(map.height_at(0.0, 0.0), Some(0.0), "the floor, not the sky");
        assert_eq!(map.ground_level(0.25), Some(0.0));
    }

    #[test]
    fn map_ceiling_drops_geometry_above_it_and_reveals_the_floor_below() {
        // A depth-WRITING roof over half the floor: the ceiling option is the only way to
        // see what is under it.
        let scene = Scene {
            color: None,
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
        let scene = Scene { color: None, draws: vec![ground_quad(4.0, -10.0, -10.0, 10.0, 10.0, true)] };
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
        ];
        // No uniforms: that is what makes this 2D rather than MVP, which is the whole point.
        d.uniforms = vec![];
        d.textures = vec![BoundTexture {
            unit: 0,
            base_format: 0x0c,
            swizzle: 0,
            tex_type: 0,
            width: 4,
            height: 4,
            stride: 16,
            faces: 1,
            face_bytes: 64,
            data_addr: 0x1000,
            pixels: vec![tex_byte; 64].into(),
            u_addr_mode: 0,
            v_addr_mode: 0,
            lod_bias: 0,
            min_filter: 0,
            mag_filter: 0,
        }];
        d
    }

    #[test]
    fn sprites_are_located_on_screen_and_keep_their_identity_when_they_move() {
        let scene = Scene {
            color: None,
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
            color: None,
            draws: vec![sprite_quad(400.0, 200.0, 480.0, 280.0, 0.0, 0.0, 7)],
        };
        let after = locate_sprites(&moved, 960, 544);
        assert_eq!(after[0].id, s.id, "identity must survive motion");
        // A different region of the same sheet is a DIFFERENT sprite.
        let other = Scene {
            color: None,
            draws: vec![sprite_quad(100.0, 200.0, 180.0, 280.0, 0.5, 0.5, 7)],
        };
        assert_ne!(locate_sprites(&other, 960, 544)[0].id, s.id, "another atlas region");
    }

    #[test]
    fn sprite_motion_removes_the_scene_scroll() {
        // A backdrop of many sprites panning left by 6px, and one that moves against it.
        let build = |shift: f32, hero_extra: f32| Scene {
            color: None,
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
            color: None,
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
        Scene { color: None, draws }
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
        let scene = Scene { color: None, draws };
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
        let scene = Scene { color: None, draws: vec![ground_quad(0.0, -100.0, -100.0, 100.0, 100.0, true)] };
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
        let scene = Scene { color: None, draws };
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
