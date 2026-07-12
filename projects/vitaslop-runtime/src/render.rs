//! A software rasterizer over the captured GXM stream: the first, blob-free way
//! to turn a recorded scene into pixels. It is a fixed-function equivalent of the
//! cube's (placeholder) shaders - transform each vertex position by the captured
//! MVP uniform, interpolate the per-vertex color, depth-test - which is exactly
//! what the real vertex/fragment programs would do. No Sony shader blob needed.
//!
//! This is the CPU reference. A wgpu backend over the same capture comes later;
//! keeping this pure and engine-agnostic makes it the oracle for that.

use crate::capture::{Draw, Scene};

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

/// GXM attribute formats we decode.
const FORMAT_U8N: u8 = 4;
const FORMAT_F32: u8 = 9;

/// A vertex pulled out of the stream: object-space position and RGBA color.
struct Vertex {
    pos: [f32; 3],
    color: [u8; 4],
}

/// Multiply column-major 4x4 `m` by the column vector `(x,y,z,1)`.
fn transform(m: &[f32; 16], x: f32, y: f32, z: f32) -> [f32; 4] {
    [
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
        m[3] * x + m[7] * y + m[11] * z + m[15],
    ]
}

/// Decode vertex `i` from a draw's buffer using its attribute layout. Falls back
/// to the cube's layout (float3 pos at 0, u8x4 color at 12) if an attribute is
/// missing.
fn decode_vertex(d: &Draw, i: usize) -> Vertex {
    let stride = d.vertex_stride.max(1) as usize;
    let base = i * stride;
    let mut pos_off = 0usize;
    let mut color_off = 12usize;
    for a in &d.attributes {
        if a.format == FORMAT_F32 && a.component_count >= 3 {
            pos_off = a.offset as usize;
        } else if a.format == FORMAT_U8N && a.component_count >= 3 {
            color_off = a.offset as usize;
        }
    }
    let f = |off: usize| -> f32 {
        let o = base + off;
        if o + 4 <= d.vertices.len() {
            f32::from_le_bytes([d.vertices[o], d.vertices[o + 1], d.vertices[o + 2], d.vertices[o + 3]])
        } else {
            0.0
        }
    };
    let pos = [f(pos_off), f(pos_off + 4), f(pos_off + 8)];
    let c = base + color_off;
    // Color bytes are laid out R,G,B,A (the cube stores 0xAABBGGRR, little-endian
    // so byte 0 is R). U8N normalization is applied at blend time, not here.
    let color = if c + 4 <= d.vertices.len() {
        [d.vertices[c], d.vertices[c + 1], d.vertices[c + 2], d.vertices[c + 3]]
    } else {
        [255, 255, 255, 255]
    };
    Vertex { pos, color }
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

/// Rasterize one scene into a fresh framebuffer. `clear` is the background color.
pub fn render_scene(scene: &Scene, width: u32, height: u32, clear: [u8; 4]) -> Framebuffer {
    let mut fb = Framebuffer::new(width, height, clear);
    let mut depth = vec![f32::INFINITY; (width * height) as usize];

    for d in &scene.draws {
        // Only triangle lists are handled (SCE_GXM_PRIMITIVE_TRIANGLES = 0).
        if d.primitive != 0 || d.uniforms.len() < 16 {
            continue;
        }
        let mut mvp = [0f32; 16];
        mvp.copy_from_slice(&d.uniforms[..16]);

        let tri_count = d.index_count as usize / 3;
        for t in 0..tri_count {
            let vs = [index_at(d, t * 3), index_at(d, t * 3 + 1), index_at(d, t * 3 + 2)];
            let verts: Vec<Vertex> = vs.iter().map(|&i| decode_vertex(d, i)).collect();

            // Clip space -> NDC -> screen. Skip triangles behind the eye.
            let mut screen = [[0f32; 4]; 3]; // x, y, depth, 1/w
            let mut behind = false;
            for (k, v) in verts.iter().enumerate() {
                let c = transform(&mvp, v.pos[0], v.pos[1], v.pos[2]);
                if c[3] <= 0.0 {
                    behind = true;
                    break;
                }
                let inv_w = 1.0 / c[3];
                let ndc_x = c[0] * inv_w;
                let ndc_y = c[1] * inv_w;
                let ndc_z = c[2] * inv_w;
                let sx = (ndc_x * 0.5 + 0.5) * width as f32;
                // Flip Y: NDC +Y is up, image +Y is down.
                let sy = (1.0 - (ndc_y * 0.5 + 0.5)) * height as f32;
                screen[k] = [sx, sy, ndc_z, inv_w];
            }
            if behind {
                continue;
            }
            raster_triangle(&mut fb, &mut depth, &screen, &verts);
        }
    }
    fb
}

/// Rasterize one screen-space triangle with a depth test and perspective-correct
/// color interpolation.
fn raster_triangle(
    fb: &mut Framebuffer,
    depth: &mut [f32],
    s: &[[f32; 4]; 3],
    verts: &[Vertex],
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
            let z = b0 * s[0][2] + b1 * s[1][2] + b2 * s[2][2];
            let idx = (y * w + x) as usize;
            if z >= depth[idx] {
                continue;
            }
            depth[idx] = z;

            // Perspective-correct color: weight by 1/w, then renormalize.
            let iw = b0 * s[0][3] + b1 * s[1][3] + b2 * s[2][3];
            let mut rgba = [0u8; 4];
            for ch in 0..4 {
                let c = (b0 * s[0][3] * verts[0].color[ch] as f32
                    + b1 * s[1][3] * verts[1].color[ch] as f32
                    + b2 * s[2][3] * verts[2].color[ch] as f32)
                    / iw;
                rgba[ch] = c.round().clamp(0.0, 255.0) as u8;
            }
            rgba[3] = 255;
            fb.rgba[idx * 4..idx * 4 + 4].copy_from_slice(&rgba);
        }
    }
}

/// Twice the signed area of triangle (a, b, c) in screen space (only x,y used).
fn edge(a: &[f32; 4], b: &[f32; 4], c: &[f32; 4]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}
