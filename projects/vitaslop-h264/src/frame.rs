//! Decoded pictures, and the pixel plumbing around them.
//!
//! Every backend hands back the same `Frame`: a planar YUV picture in ONE allocation, with
//! per-plane offsets and strides. The strides are the platform's own - a Media Foundation
//! sample and a VA-API surface both come out padded, and copying that padding away costs a
//! full frame of memory traffic for nothing - so the layout is described rather than
//! normalised. Callers that want a tight buffer ask for one explicitly
//! ([`Frame::copy_to_i420`]), and callers uploading to a GPU can use the strides directly.

use crate::error::{Error, Result};

/// Pixel layout of a decoded frame.
///
/// These are the two layouts platform decoders actually produce for 8-bit 4:2:0, which is
/// the only chroma format this crate decodes ([`Error::Unsupported`] otherwise).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// Y plane, then one interleaved Cb/Cr plane at half resolution in both axes.
    /// What Media Foundation, VideoToolbox and VA-API all prefer.
    Nv12,
    /// Y, Cb, Cr as three separate planes, chroma at half resolution in both axes.
    I420,
    /// One packed plane of 8-bit R, G, B, A.
    ///
    /// No H.264 decoder produces RGB natively - the format is 4:2:0 by construction - so a
    /// frame in this form has already been converted by whoever decoded it. It appears
    /// because a platform decoder is entitled to hand back what suits its own pipeline:
    /// MEASURED, Chrome on an Android PowerVR device delivers WebCodecs frames as `RGBA`,
    /// where the same browser on a desktop delivers `I420`. Refusing it means no video on
    /// that device, so it is carried, and [`Frame::copy_to_i420`] converts back.
    Rgba,
}

impl PixelFormat {
    /// How many planes a frame in this format carries.
    pub fn plane_count(self) -> usize {
        match self {
            PixelFormat::Nv12 => 2,
            PixelFormat::I420 => 3,
            PixelFormat::Rgba => 1,
        }
    }
}

/// The colour matrix, range and primaries a frame's samples are expressed in.
///
/// Taken from the SPS's VUI when the stream carries one. It is NOT guessed when absent:
/// `unspecified` stays unspecified, and [`ColorInfo::matrix_or_default`] applies the
/// H.264-conventional fallback (BT.601 for SD, BT.709 for HD) at the one place that has to
/// pick, which is RGB conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ColorInfo {
    /// `video_full_range_flag`: false = studio swing (Y 16..235), true = full 0..255.
    pub full_range: bool,
    /// `matrix_coefficients` (ITU-T H.264 Table E-5). 2 = unspecified.
    pub matrix: u8,
    /// `colour_primaries` (Table E-3). 2 = unspecified.
    pub primaries: u8,
    /// `transfer_characteristics` (Table E-4). 2 = unspecified.
    pub transfer: u8,
}

impl ColorInfo {
    /// Unspecified everything, studio range - what a stream with no VUI means in practice.
    pub const UNSPECIFIED: ColorInfo =
        ColorInfo { full_range: false, matrix: 2, primaries: 2, transfer: 2 };

    /// The matrix to actually convert with: the signalled one, or the conventional guess
    /// keyed on picture height when the stream says "unspecified".
    pub fn matrix_or_default(&self, height: u32) -> ColorMatrix {
        match self.matrix {
            1 => ColorMatrix::Bt709,
            5 | 6 => ColorMatrix::Bt601,
            9 => ColorMatrix::Bt2020Ncl,
            // 0 (identity/GBR), 2 (unspecified), and everything reserved: the only sane
            // convention is resolution-keyed, and it is applied here and nowhere else.
            _ => {
                if height > 576 {
                    ColorMatrix::Bt709
                } else {
                    ColorMatrix::Bt601
                }
            }
        }
    }
}

/// A YUV-to-RGB matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMatrix {
    /// ITU-R BT.601 (SD).
    Bt601,
    /// ITU-R BT.709 (HD).
    Bt709,
    /// ITU-R BT.2020 non-constant luminance.
    Bt2020Ncl,
}

impl ColorMatrix {
    /// `(kr, kb)` luma coefficients.
    fn kr_kb(self) -> (f32, f32) {
        match self {
            ColorMatrix::Bt601 => (0.299, 0.114),
            ColorMatrix::Bt709 => (0.2126, 0.0722),
            ColorMatrix::Bt2020Ncl => (0.2627, 0.0593),
        }
    }
}

/// Where one plane sits inside a frame's single allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    /// Byte offset of the plane's first row within [`Frame::data`].
    pub offset: usize,
    /// Bytes from one row to the next. Always >= the plane's used width in bytes, and often
    /// larger: it is the platform decoder's own pitch.
    pub stride: usize,
    /// Rows in the plane (luma height, or half of it for chroma).
    pub rows: usize,
    /// Used bytes per row, excluding the stride padding.
    pub row_bytes: usize,
}

/// One decoded picture.
///
/// Owns its pixels. The buffer can be handed back to the decoder with
/// [`crate::Decoder::recycle`] to be reused for a later frame, which is what keeps steady
/// playback from allocating a frame-sized block sixty times a second.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Visible width in luma samples (cropping already applied).
    pub width: u32,
    /// Visible height in luma samples (cropping already applied).
    pub height: u32,
    /// Layout of `data`.
    pub format: PixelFormat,
    /// Colour signalling from the stream's VUI.
    pub color: ColorInfo,
    /// Presentation timestamp, in whatever units the caller used on the input packet. When
    /// the caller supplied none, this is the picture's presentation-order index, which is
    /// still a correct ordering key.
    pub pts: i64,
    /// Plane descriptors; `format.plane_count()` of them are meaningful.
    pub planes: [Plane; 3],
    /// The pixels. One allocation for all planes.
    pub data: Vec<u8>,
}

const EMPTY_PLANE: Plane = Plane { offset: 0, stride: 0, rows: 0, row_bytes: 0 };

impl Frame {
    /// A frame with tightly packed planes for `format` at `width` x `height`, contents
    /// zeroed. Used by the backends to shape a buffer before filling it; `data` is a
    /// recycled allocation when the caller returned one.
    pub(crate) fn alloc(format: PixelFormat, width: u32, height: u32, mut data: Vec<u8>) -> Frame {
        let (w, h) = (width as usize, height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let (planes, size) = match format {
            PixelFormat::Nv12 => {
                let y = Plane { offset: 0, stride: w, rows: h, row_bytes: w };
                let uv = Plane { offset: w * h, stride: cw * 2, rows: ch, row_bytes: cw * 2 };
                ([y, uv, EMPTY_PLANE], w * h + cw * 2 * ch)
            }
            PixelFormat::I420 => {
                let y = Plane { offset: 0, stride: w, rows: h, row_bytes: w };
                let u = Plane { offset: w * h, stride: cw, rows: ch, row_bytes: cw };
                let v = Plane { offset: w * h + cw * ch, stride: cw, rows: ch, row_bytes: cw };
                ([y, u, v], w * h + 2 * cw * ch)
            }
            PixelFormat::Rgba => {
                let rgba = Plane { offset: 0, stride: w * 4, rows: h, row_bytes: w * 4 };
                ([rgba, EMPTY_PLANE, EMPTY_PLANE], w * h * 4)
            }
        };
        data.clear();
        data.resize(size, 0);
        Frame { width, height, format, color: ColorInfo::UNSPECIFIED, pts: 0, planes, data }
    }

    /// Read-only view of plane `i`, including its stride padding.
    pub fn plane(&self, i: usize) -> &[u8] {
        let p = self.planes[i];
        &self.data[p.offset..p.offset + p.stride * p.rows]
    }

    /// Mutable view of plane `i`.
    pub(crate) fn plane_mut(&mut self, i: usize) -> &mut [u8] {
        let p = self.planes[i];
        &mut self.data[p.offset..p.offset + p.stride * p.rows]
    }

    /// One row of plane `i`, trimmed to the used bytes.
    pub fn row(&self, i: usize, y: usize) -> &[u8] {
        let p = self.planes[i];
        let start = p.offset + y * p.stride;
        &self.data[start..start + p.row_bytes]
    }

    /// Bytes a tightly packed I420 copy of this frame needs.
    pub fn i420_size(&self) -> usize {
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        w * h + 2 * cw * ch
    }

    /// Copy into a tightly packed I420 buffer (Y, then Cb, then Cr, no padding).
    ///
    /// This is the "give me something predictable" path: it de-interleaves NV12 and drops
    /// stride padding. It costs a full frame copy, so a caller uploading to a texture is
    /// usually better off reading [`Frame::planes`] directly.
    pub fn copy_to_i420(&self, out: &mut Vec<u8>) {
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        out.clear();
        out.resize(self.i420_size(), 0);
        let (y_out, rest) = out.split_at_mut(w * h);
        let (u_out, v_out) = rest.split_at_mut(cw * ch);

        if self.format == PixelFormat::Rgba {
            self.rgba_to_i420(y_out, u_out, v_out);
            return;
        }
        for r in 0..h {
            y_out[r * w..(r + 1) * w].copy_from_slice(&self.row(0, r)[..w]);
        }
        match self.format {
            PixelFormat::Rgba => unreachable!("handled above"),
            PixelFormat::I420 => {
                for r in 0..ch {
                    u_out[r * cw..(r + 1) * cw].copy_from_slice(&self.row(1, r)[..cw]);
                    v_out[r * cw..(r + 1) * cw].copy_from_slice(&self.row(2, r)[..cw]);
                }
            }
            PixelFormat::Nv12 => {
                for r in 0..ch {
                    let src = &self.row(1, r)[..cw * 2];
                    let u_row = &mut u_out[r * cw..(r + 1) * cw];
                    let v_row = &mut v_out[r * cw..(r + 1) * cw];
                    for x in 0..cw {
                        u_row[x] = src[x * 2];
                        v_row[x] = src[x * 2 + 1];
                    }
                }
            }
        }
    }

    /// Bytes a [`Frame::copy_packed`] of this frame occupies.
    pub fn packed_size(&self) -> usize {
        let (w, h) = (self.width as usize, self.height as usize);
        match self.format {
            PixelFormat::Rgba => w * h * 4,
            PixelFormat::Nv12 | PixelFormat::I420 => {
                let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
                w * h + 2 * cw * ch
            }
        }
    }

    /// Copy the frame into `out` in ITS OWN format, tightly packed - stride padding removed,
    /// no colour conversion of any kind.
    ///
    /// # Why a caller usually wants this and not [`Frame::copy_to_i420`]
    ///
    /// Converting to one layout is convenient and, for a caller that is going to write the
    /// frame somewhere in a THIRD layout, pure waste. A decoder that produced NV12 and a
    /// consumer that wants NV12 were, through `copy_to_i420`, doing NV12 -> I420 -> NV12:
    /// two conversions of every pixel, per frame, to arrive back where they started. This
    /// hands the bytes over as they are and lets the consumer convert once, or not at all.
    pub fn copy_packed(&self, out: &mut Vec<u8>) {
        let (w, h) = (self.width as usize, self.height as usize);
        out.clear();
        out.resize(self.packed_size(), 0);
        match self.format {
            PixelFormat::Rgba => {
                for r in 0..h {
                    out[r * w * 4..(r + 1) * w * 4].copy_from_slice(&self.row(0, r)[..w * 4]);
                }
            }
            PixelFormat::Nv12 => {
                let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
                let (y_out, uv_out) = out.split_at_mut(w * h);
                for r in 0..h {
                    y_out[r * w..(r + 1) * w].copy_from_slice(&self.row(0, r)[..w]);
                }
                for r in 0..ch {
                    uv_out[r * cw * 2..(r + 1) * cw * 2]
                        .copy_from_slice(&self.row(1, r)[..cw * 2]);
                }
            }
            PixelFormat::I420 => {
                let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
                let (y_out, rest) = out.split_at_mut(w * h);
                let (u_out, v_out) = rest.split_at_mut(cw * ch);
                for r in 0..h {
                    y_out[r * w..(r + 1) * w].copy_from_slice(&self.row(0, r)[..w]);
                }
                for r in 0..ch {
                    u_out[r * cw..(r + 1) * cw].copy_from_slice(&self.row(1, r)[..cw]);
                    v_out[r * cw..(r + 1) * cw].copy_from_slice(&self.row(2, r)[..cw]);
                }
            }
        }
    }

    /// Convert a packed RGBA frame back to 4:2:0.
    ///
    /// Chroma is BOX-FILTERED over each 2x2 group rather than point-sampled: the samples
    /// being discarded are real, and averaging them is what the 4:2:0 subsampling the frame
    /// came from would have done. The matrix is BT.601 studio-swing, the inverse of the one
    /// [`Frame::copy_to_rgba`] applies, so a frame that made the round trip comes back
    /// where it started to within a step of rounding.
    fn rgba_to_i420(&self, y_out: &mut [u8], u_out: &mut [u8], v_out: &mut [u8]) {
        let (w, h) = (self.width as usize, self.height as usize);
        let cw = w.div_ceil(2);
        let rgb = |x: usize, y: usize| -> (i32, i32, i32) {
            let row = self.row(0, y.min(h - 1));
            let at = x.min(w - 1) * 4;
            (row[at] as i32, row[at + 1] as i32, row[at + 2] as i32)
        };
        // 16.16 fixed point, studio swing: Y in 16..235, Cb/Cr in 16..240 around 128.
        let luma = |(r, g, b): (i32, i32, i32)| -> i32 { 16589 * r + 32558 * g + 6321 * b + (16 << 16) };
        for y in 0..h {
            for x in 0..w {
                y_out[y * w + x] = ((luma(rgb(x, y)) + 32768) >> 16).clamp(0, 255) as u8;
            }
        }
        for cy in 0..h.div_ceil(2) {
            for cx in 0..cw {
                // The 2x2 group this chroma sample covers, averaged.
                let mut sum = (0i32, 0i32, 0i32);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (r, g, b) = rgb(cx * 2 + dx, cy * 2 + dy);
                    sum = (sum.0 + r, sum.1 + g, sum.2 + b);
                }
                let (r, g, b) = (sum.0 / 4, sum.1 / 4, sum.2 / 4);
                let yv = luma((r, g, b));
                let cb = (-9713 * r - 19070 * g + 28784 * b + (128 << 16) + 32768) >> 16;
                let cr = (28784 * r - 24103 * g - 4681 * b + (128 << 16) + 32768) >> 16;
                let _ = yv;
                u_out[cy * cw + cx] = cb.clamp(0, 255) as u8;
                v_out[cy * cw + cx] = cr.clamp(0, 255) as u8;
            }
        }
    }

    /// Convert to 8-bit RGBA (alpha 255) into `out`, which is resized to `width*height*4`.
    ///
    /// Fixed-point integer arithmetic, chroma upsampled by nearest neighbour - the same
    /// thing every video path does before a GPU gets involved, and about as fast as a scalar
    /// loop gets. A caller that cares about quality or speed should upload the YUV planes
    /// and convert in a shader; this exists so a caller with no GPU (a test, a thumbnail, a
    /// software blit) has one call to make.
    pub fn copy_to_rgba(&self, out: &mut Vec<u8>) {
        let (w, h) = (self.width as usize, self.height as usize);
        out.clear();
        out.resize(w * h * 4, 0);

        let matrix = self.color.matrix_or_default(self.height);
        let (kr, kb) = matrix.kr_kb();
        let kg = 1.0 - kr - kb;
        // Studio swing expands Y 16..235 and C 16..240; full range does not scale.
        let (y_scale, y_off, c_scale) = if self.color.full_range {
            (1.0f32, 0.0f32, 1.0f32)
        } else {
            (255.0 / 219.0, 16.0, 255.0 / 224.0)
        };
        // R = Y + 2(1-kr)V ; B = Y + 2(1-kb)U ; G = Y - (2 kr (1-kr) V + 2 kb (1-kb) U)/kg
        let fx = |v: f32| (v * (1 << 14) as f32).round() as i32;
        let cy = fx(y_scale);
        let cvr = fx(2.0 * (1.0 - kr) * c_scale);
        let cub = fx(2.0 * (1.0 - kb) * c_scale);
        let cvg = fx(2.0 * kr * (1.0 - kr) / kg * c_scale);
        let cug = fx(2.0 * kb * (1.0 - kb) / kg * c_scale);
        let y_bias = fx(-y_off * y_scale);
        let co = Coeffs { cy, y_bias, cvr, cub, cvg, cug };

        // Already RGBA: a straight row copy, and no matrix is applied at all - whoever
        // produced it did the conversion, and doing it again would be a second one.
        if self.format == PixelFormat::Rgba {
            for r in 0..h {
                out[r * w * 4..(r + 1) * w * 4].copy_from_slice(&self.row(0, r)[..w * 4]);
            }
            return;
        }
        for r in 0..h {
            let y_row = self.row(0, r);
            let cr = r / 2;
            let dst = &mut out[r * w * 4..(r + 1) * w * 4];
            match self.format {
                PixelFormat::Rgba => unreachable!("handled above"),
                PixelFormat::Nv12 => {
                    let uv = self.row(1, cr);
                    for x in 0..w {
                        let u = uv[(x / 2) * 2] as i32 - 128;
                        let v = uv[(x / 2) * 2 + 1] as i32 - 128;
                        write_rgba(dst, x, y_row[x] as i32, u, v, &co);
                    }
                }
                PixelFormat::I420 => {
                    let u_row = self.row(1, cr);
                    let v_row = self.row(2, cr);
                    for x in 0..w {
                        let u = u_row[x / 2] as i32 - 128;
                        let v = v_row[x / 2] as i32 - 128;
                        write_rgba(dst, x, y_row[x] as i32, u, v, &co);
                    }
                }
            }
        }
    }

    /// Sanity check used by the backends before a frame is handed out: every plane has to
    /// sit inside `data`. A backend that miscomputed a stride from a platform struct fails
    /// here rather than handing the caller a frame with garbage at the bottom.
    pub(crate) fn validate(&self) -> Result<()> {
        for i in 0..self.format.plane_count() {
            let p = self.planes[i];
            if p.row_bytes > p.stride || p.offset + p.stride * p.rows > self.data.len() {
                return Err(Error::platform(
                    "frame layout",
                    0,
                    format!("plane {i} {p:?} does not fit {} bytes", self.data.len()),
                ));
            }
        }
        Ok(())
    }
}

/// Fixed-point YUV->RGB coefficients, 14 fractional bits.
struct Coeffs {
    cy: i32,
    y_bias: i32,
    cvr: i32,
    cub: i32,
    cvg: i32,
    cug: i32,
}

#[inline(always)]
fn write_rgba(dst: &mut [u8], x: usize, y: i32, u: i32, v: i32, c: &Coeffs) {
    let yy = y * c.cy + c.y_bias + (1 << 13);
    let px = &mut dst[x * 4..x * 4 + 4];
    px[0] = clamp8(yy + v * c.cvr);
    px[1] = clamp8(yy - v * c.cvg - u * c.cug);
    px[2] = clamp8(yy + u * c.cub);
    px[3] = 255;
}

#[inline(always)]
fn clamp8(v: i32) -> u8 {
    (v >> 14).clamp(0, 255) as u8
}

#[cfg(test)]
mod rgba_round_trip_tests {
    use super::*;

    /// A packed-RGBA frame must survive the trip to 4:2:0 and back.
    ///
    /// # Why this test exists at all
    ///
    /// The RGBA path only happens on a device: Chrome on an Android PowerVR phone delivers
    /// WebCodecs frames as `RGBA`, and the same browser on this desktop delivers `I420`, so
    /// the conversion cannot be reached by running anything here. Testing it against the
    /// INVERSE conversion needs no device and no decoder - the two matrices either agree or
    /// they do not.
    fn rgba_frame(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [u8; 4]) -> Frame {
        let mut frame = Frame::alloc(PixelFormat::Rgba, width, height, Vec::new());
        for y in 0..height {
            let row = frame.plane_mut(0);
            for x in 0..width {
                let at = (y * width + x) as usize * 4;
                row[at..at + 4].copy_from_slice(&pixel(x, y));
            }
        }
        frame
    }

    #[test]
    fn flat_colours_survive_the_round_trip() {
        // Flat blocks have no chroma detail to lose, so the only error is the matrix's own
        // rounding - which is what this is measuring.
        for colour in [
            [0, 0, 0, 255],
            [255, 255, 255, 255],
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [128, 64, 192, 255],
        ] {
            let frame = rgba_frame(16, 16, |_, _| colour);
            let mut i420 = Vec::new();
            frame.copy_to_i420(&mut i420);

            // Back again, through the decoder-side conversion every other format uses.
            let mut yuv = Frame::alloc(PixelFormat::I420, 16, 16, Vec::new());
            // 6 is BT.601 in the VUI's own numbering; studio swing is the default.
            yuv.color.matrix = 6;
            yuv.color.full_range = false;
            let (w, h) = (16usize, 16usize);
            let (cw, ch) = (8usize, 8usize);
            yuv.plane_mut(0)[..w * h].copy_from_slice(&i420[..w * h]);
            yuv.plane_mut(1)[..cw * ch].copy_from_slice(&i420[w * h..w * h + cw * ch]);
            yuv.plane_mut(2)[..cw * ch].copy_from_slice(&i420[w * h + cw * ch..]);
            let mut back = Vec::new();
            yuv.copy_to_rgba(&mut back);

            for (i, chunk) in back.chunks_exact(4).enumerate() {
                for c in 0..3 {
                    let (got, want) = (chunk[c] as i32, colour[c] as i32);
                    assert!(
                        (got - want).abs() <= 3,
                        "pixel {i} channel {c} of {colour:?} came back {got}, off by {}",
                        (got - want).abs()
                    );
                }
            }
        }
    }

    #[test]
    fn luma_tracks_brightness_monotonically() {
        // A grey ramp: whatever the matrix is, a brighter input must not produce a darker
        // luma. This catches a transposed or sign-flipped coefficient, which a single flat
        // colour can pass by luck.
        let frame = rgba_frame(64, 2, |x, _| {
            let v = (x * 4) as u8;
            [v, v, v, 255]
        });
        let mut i420 = Vec::new();
        frame.copy_to_i420(&mut i420);
        for x in 1..64usize {
            assert!(
                i420[x] >= i420[x - 1],
                "luma fell from {} to {} across a rising grey ramp",
                i420[x - 1],
                i420[x]
            );
        }
        // Studio swing: black sits at 16, not 0.
        assert!((i420[0] as i32 - 16).abs() <= 1, "black came out at {} not 16", i420[0]);
    }
}
