//! SceJpegEnc: the hardware JPEG encoder, at the API level.
//!
//! A title reaches this to turn a captured framebuffer into a JPEG - a photo mode, a
//! shared screenshot, a save-slot thumbnail. `psp2/jpegenc.h` publishes the prototypes, so
//! unlike [`super::gesture`] nothing about the interface is guessed here.
//!
//! What IS and IS NOT modelled, and the line is deliberate:
//!
//! - The SETUP calls are real and complete. The encoder context is opaque to the guest -
//!   it asks how big it is, allocates that much, and hands the pointer back - so a context
//!   whose size and contents this module chooses is not an approximation of anything, it
//!   IS the context.
//! - `Csc` and `Encode` really convert and really encode: the title gets a complete baseline
//!   JPEG of its own pixels, which is what the hardware encoder hands back by default. The
//!   bytes are this module's encoder's, not the hardware's - nothing published pins the
//!   hardware's quantisation or Huffman choices - but every decoder reads the same picture.
//!
//! What is established and what is chosen:
//! - `ARGB8888` input is the display framebuffer's own byte order, R,G,B,A in memory - an
//!   open capture plugin running on the console hands the framebuffer to `Csc` unswapped.
//! - 4:2:0 YCbCr is `w*h*3/2` bytes and 4:2:2 is `w*h*2`. The PLANE ORDER inside is not
//!   published; it is laid out here as this family's decoder lays its own YCbCr out
//!   (`sceJpegDecodeMJpegYCbCr`): planar Y, then Cb, then Cr.
//! - A higher compression ratio is more compression (vitasdk); how the 0..255 ratio maps to
//!   quantisation is not published, so it scales the standard tables linearly from quality
//!   100 at ratio 0, with the sibling ARM encoder's default ratio of 64 when none was set.
//! - The MJPEG header mode's markers are not published; both modes emit a complete JPEG.

// The `#[hostcall]` macro emits fully-qualified paths for these, so a module of nothing
// but host calls needs no plain import of them.
use crate::hostcall;

/// Bytes of encoder context [`get_context_size`] asks the guest to allocate.
///
/// The console's own figure is not published and it is not observable either: the guest
/// only ever asks for this number, allocates it, and passes the pointer back to this same
/// module, so the context is opaque by construction and this IS its size. Chosen large
/// enough that a real implementation's state could live in it.
const JPEGENC_CONTEXT_BYTES: i32 = 4096;

/// Magic stamped at the head of an encoder context, so a later call can tell a context
/// this run initialised from an uninitialised buffer. In the guest's own allocation
/// rather than a host table keyed by its address, for the reason in
/// memory `vitaslop-host-call-reference-semantics`.
const JPEGENC_MAGIC: u32 = 0x4A45_4E43; // "JENC"

/// `SCE_JPEGENC_ERROR_INVALID_POINTER` (`psp2common/jpegenc.h`: 0x80650205).
const SCE_JPEGENC_ERROR_INVALID_POINTER: i32 = 0x8065_0205u32 as i32;

/// int sceJpegEncoderGetContextSize(void)
///
/// Takes no arguments - the `r0`/`r1` an observed call arrives with are leftovers from
/// the caller's previous work, not parameters. See [`JPEGENC_CONTEXT_BYTES`] for why a
/// chosen size is exact rather than approximate here.
#[hostcall]
pub(super) fn get_context_size(_ctx: &mut GuestCtx, _st: &mut VitaState) -> i32 {
    JPEGENC_CONTEXT_BYTES
}

/// int sceJpegEncoderInit(SceJpegEncoderContext context, int inWidth, int inHeight,
///     SceJpegEncoderPixelFormat pixelformat, void *outBuffer, SceSize outSize)
///
/// Records the request in the guest's own context block. Nothing is encoded here and
/// nothing is written to `outBuffer` - an encoder that has been initialised has not yet
/// produced a frame, so an untouched output buffer is the correct state, not a gap.
#[hostcall]
pub(super) fn init(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    context: Ptr,
    in_width: u32,
    in_height: u32,
    pixelformat: u32,
    out_buffer: u32,
    out_size: u32,
) -> i32 {
    record_init(
        ctx,
        context,
        in_width,
        in_height,
        pixelformat,
        out_buffer,
        out_size,
    )
}

/// The shared body of [`init`] and [`init_with_param`]: the request, in the guest's own block.
fn record_init(
    ctx: &mut crate::host::GuestCtx,
    context: crate::host::Ptr,
    w: u32,
    h: u32,
    fmt: u32,
    out: u32,
    size: u32,
) -> i32 {
    if context.is_null() {
        SCE_JPEGENC_ERROR_INVALID_POINTER
    } else {
        let a = context.addr();
        ctx.write_u32(a, JPEGENC_MAGIC);
        ctx.write_u32(a + 4, w);
        ctx.write_u32(a + 8, h);
        ctx.write_u32(a + 12, fmt);
        ctx.write_u32(a + 16, out);
        ctx.write_u32(a + 20, size);
        0
    }
}

/// int sceJpegEncoderInitWithParam(SceJpegEncoderContext context, const SceJpegEncoderInitParam *initParam)
///
/// [`init`] with its arguments in a struct (`psp2/jpegenc.h`, asserted 0x1C bytes): `size`,
/// `inWidth`, `inHeight`, `pixelFormat`, `outBuffer`, `outSize`, `option`. The `option` (LPDDR2
/// instead of CDRAM) names where the hardware works, which nothing here observes. First called by
/// a 2011 adventure title at boot.
#[hostcall]
pub(super) fn init_with_param(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    context: Ptr,
    param: Ptr,
) -> i32 {
    if param.is_null() {
        SCE_JPEGENC_ERROR_INVALID_POINTER
    } else {
        let p = param.addr();
        let (w, h, fmt, out, size) = (
            ctx.read_u32(p + 4),
            ctx.read_u32(p + 8),
            ctx.read_u32(p + 12),
            ctx.read_u32(p + 16),
            ctx.read_u32(p + 20),
        );
        record_init(ctx, context, w, h, fmt, out, size)
    }
}

/// int sceJpegEncoderSetHeaderMode(SceJpegEncoderContext context, int mode)
///
/// Selects the header the encoder will emit (JPEG or MJPEG). Recorded in the context block next
/// to the other settings; nothing is encoded here (see the module docs), so it has no effect yet.
#[hostcall]
pub(super) fn set_header_mode(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    context: Ptr,
    mode: u32,
) -> i32 {
    if context.is_null() {
        SCE_JPEGENC_ERROR_INVALID_POINTER
    } else {
        // +36, its own slot: +28/+32 are `set_valid_region`'s width and height, and sharing +28
        // let either call overwrite the other's setting.
        ctx.write_u32(context.addr() + 36, mode);
        0
    }
}

/// int sceJpegEncoderEnd(SceJpegEncoderContext context)
/// Tears the context down: clears the marker so a use-after-end is visible rather than
/// silently working.
#[hostcall]
pub(super) fn end(ctx: &mut GuestCtx, _st: &mut VitaState, context: Ptr) -> i32 {
    if context.is_null() {
        SCE_JPEGENC_ERROR_INVALID_POINTER
    } else {
        ctx.write_u32(context.addr(), 0);
        0
    }
}

/// int sceJpegEncoderSetOutputAddr(SceJpegEncoderContext context, void *outBuffer,
///     SceSize outSize)
#[hostcall]
pub(super) fn set_output_addr(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    context: Ptr,
    out_buffer: u32,
    out_size: u32,
) -> i32 {
    if context.is_null() {
        SCE_JPEGENC_ERROR_INVALID_POINTER
    } else {
        ctx.write_u32(context.addr() + 16, out_buffer);
        ctx.write_u32(context.addr() + 20, out_size);
        0
    }
}

/// int sceJpegEncoderSetCompressionRatio(SceJpegEncoderContext context, int ratio)
/// Purely a quality setting on a context; recorded, and it changes nothing else because
/// nothing is encoded.
#[hostcall]
pub(super) fn set_compression_ratio(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    context: Ptr,
    ratio: u32,
) -> i32 {
    if context.is_null() {
        SCE_JPEGENC_ERROR_INVALID_POINTER
    } else {
        ctx.write_u32(context.addr() + 24, ratio);
        0
    }
}

// --- the context block, as `record_init` and the setters lay it out -----------------------
const CTX_WIDTH: u32 = 4;
const CTX_HEIGHT: u32 = 8;
const CTX_FORMAT: u32 = 12;
const CTX_OUT: u32 = 16;
const CTX_OUT_SIZE: u32 = 20;
const CTX_RATIO: u32 = 24;
const CTX_VALID_W: u32 = 28;
const CTX_VALID_H: u32 = 32;

/// `SCE_JPEGENC_PIXELFORMAT_*` (psp2common/jpegenc.h).
const PIXELFORMAT_YCBCR420: u32 = 8;
const PIXELFORMAT_YCBCR422: u32 = 9;
/// `SCE_JPEGENC_ERROR_*` (psp2common/jpegenc.h).
const SCE_JPEGENC_ERROR_IMAGE_SIZE: i32 = 0x8065_0200u32 as i32;
const SCE_JPEGENC_ERROR_INSUFFICIENT_BUFFER: i32 = 0x8065_0201u32 as i32;
const SCE_JPEGENC_ERROR_INVALID_PIXELFORMAT: i32 = 0x8065_0203u32 as i32;
/// The compression ratio a context has before `sceJpegEncoderSetCompressionRatio`.
const DEFAULT_RATIO: u32 = 64;

/// What an initialised context is set up to encode: the size (the valid region when one was
/// set) and whether the chroma is halved vertically too (4:2:0) or only horizontally (4:2:2).
struct Setup {
    w: usize,
    h: usize,
    vsub: bool,
}

fn setup(ctx: &crate::host::GuestCtx, context: u32) -> Result<Setup, i32> {
    if context == 0 || ctx.read_u32(context) != JPEGENC_MAGIC {
        return Err(SCE_JPEGENC_ERROR_INVALID_POINTER);
    }
    let vsub = match ctx.read_u32(context + CTX_FORMAT) & 0xf {
        PIXELFORMAT_YCBCR420 => true,
        PIXELFORMAT_YCBCR422 => false,
        _ => return Err(SCE_JPEGENC_ERROR_INVALID_PIXELFORMAT),
    };
    let (vw, vh) = (
        ctx.read_u32(context + CTX_VALID_W),
        ctx.read_u32(context + CTX_VALID_H),
    );
    let (w, h) = if vw != 0 && vh != 0 {
        (vw, vh)
    } else {
        (
            ctx.read_u32(context + CTX_WIDTH),
            ctx.read_u32(context + CTX_HEIGHT),
        )
    };
    if w == 0 || h == 0 || w > 4096 || h > 4096 {
        return Err(SCE_JPEGENC_ERROR_IMAGE_SIZE);
    }
    Ok(Setup {
        w: w as usize,
        h: h as usize,
        vsub,
    })
}

impl Setup {
    fn chroma(&self) -> (usize, usize) {
        (
            self.w.div_ceil(2),
            if self.vsub {
                self.h.div_ceil(2)
            } else {
                self.h
            },
        )
    }
    fn ycbcr_bytes(&self) -> usize {
        let (cw, ch) = self.chroma();
        self.w * self.h + 2 * cw * ch
    }
}

/// int sceJpegEncoderCsc(SceJpegEncoderContext context, void *outBuffer, const void *inBuffer,
///     int inPitch, SceJpegEncoderPixelFormat inPixelFormat)
///
/// ARGB8888 (R,G,B,A in memory, `inPitch` pixels a row) to the context's YCbCr: planar Y, Cb,
/// Cr, the chroma averaged over each 2x2 (4:2:0) or 2x1 (4:2:2) block. JFIF's BT.601
/// full-range conversion, the one this family's decoder inverts.
#[hostcall]
pub(super) fn csc(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    context: Ptr,
    out: Ptr,
    input: Ptr,
    pitch: u32,
    _fmt: u32,
) -> i32 {
    match setup(ctx, context.addr()) {
        Err(e) => e,
        Ok(_) if out.is_null() || input.is_null() => SCE_JPEGENC_ERROR_INVALID_POINTER,
        Ok(s) => {
            let pitch = (pitch as usize).max(s.w);
            let rgba = ctx.read_bytes(input.addr(), pitch * 4 * (s.h - 1) + s.w * 4);
            let (cw, ch) = s.chroma();
            let mut y = vec![0u8; s.w * s.h];
            let mut cb = vec![0u32; cw * ch];
            let mut cr = vec![0u32; cw * ch];
            let mut n = vec![0u32; cw * ch];
            for row in 0..s.h {
                for col in 0..s.w {
                    let p = (row * pitch + col) * 4;
                    let (yy, b, r) = super::jpeg::rgb_to_ycbcr(rgba[p], rgba[p + 1], rgba[p + 2]);
                    y[row * s.w + col] = yy;
                    let c = (if s.vsub { row / 2 } else { row }) * cw + col / 2;
                    cb[c] += u32::from(b);
                    cr[c] += u32::from(r);
                    n[c] += 1;
                }
            }
            let mut planes = y;
            planes.extend(cb.iter().zip(&n).map(|(v, k)| ((v + k / 2) / k) as u8));
            planes.extend(cr.iter().zip(&n).map(|(v, k)| ((v + k / 2) / k) as u8));
            ctx.write_bytes(out.addr(), &planes);
            0
        }
    }
}

/// int sceJpegEncoderEncode(SceJpegEncoderContext context, const void *inBuffer)
///
/// The context's YCbCr (as [`csc`] lays it out) to a baseline JPEG in the context's output
/// buffer, answering its byte count. A JPEG that would not fit is INSUFFICIENT_BUFFER and the
/// output is left alone - a truncated JPEG is a corrupt file a title would save.
#[hostcall]
pub(super) fn encode(ctx: &mut GuestCtx, _st: &mut VitaState, context: Ptr, input: Ptr) -> i32 {
    match setup(ctx, context.addr()) {
        Err(e) => e,
        Ok(_) if input.is_null() => SCE_JPEGENC_ERROR_INVALID_POINTER,
        Ok(s) => {
            let a = context.addr();
            let (out, out_size) = (ctx.read_u32(a + CTX_OUT), ctx.read_u32(a + CTX_OUT_SIZE));
            let ratio = match ctx.read_u32(a + CTX_RATIO) {
                0 => DEFAULT_RATIO,
                r => r.min(255),
            };
            let planes = ctx.read_bytes(input.addr(), s.ycbcr_bytes());
            let jpeg = encode_baseline(&s, &planes, ratio);
            if out == 0 {
                SCE_JPEGENC_ERROR_INVALID_POINTER
            } else if jpeg.len() > out_size as usize {
                SCE_JPEGENC_ERROR_INSUFFICIENT_BUFFER
            } else {
                ctx.write_bytes(out, &jpeg);
                jpeg.len() as i32
            }
        }
    }
}

// --- a baseline JPEG encoder (ITU T.81, the Annex K tables) ------------------------------

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
const LUMA_Q: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
const CHROMA_Q: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];
const DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_VALUES: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
const AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const AC_LUMA_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];
const AC_CHROMA_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// Canonical Huffman codes from a table's per-length counts: `(code, length)` per symbol.
fn huff_codes(bits: &[u8; 16], values: &[u8]) -> [(u16, u8); 256] {
    let mut out = [(0u16, 0u8); 256];
    let (mut code, mut k) = (0u16, 0usize);
    for (len, &count) in bits.iter().enumerate() {
        for _ in 0..count {
            out[values[k] as usize] = (code, len as u8 + 1);
            code += 1;
            k += 1;
        }
        code <<= 1;
    }
    out
}

struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitWriter {
    fn put(&mut self, code: u32, len: u32) {
        self.acc = (self.acc << len) | (code & ((1 << len) - 1));
        self.n += len;
        while self.n >= 8 {
            let b = (self.acc >> (self.n - 8)) as u8;
            self.out.push(b);
            if b == 0xff {
                self.out.push(0);
            }
            self.n -= 8;
        }
        self.acc &= (1 << self.n) - 1;
    }
    fn flush(&mut self) {
        if self.n > 0 {
            let pad = 8 - self.n;
            self.put((1 << pad) - 1, pad);
        }
    }
}

/// The quantisation table for a ratio: the standard table scaled as quality `100 - ratio*100/255`.
fn quant(base: &[u16; 64], ratio: u32) -> [u16; 64] {
    let q = (100 - (ratio * 100 / 255)).clamp(1, 100);
    let scale = if q < 50 { 5000 / q } else { 200 - 2 * q };
    base.map(|v| ((u32::from(v) * scale + 50) / 100).clamp(1, 255) as u16)
}

/// The magnitude category of a coefficient and its bits, as T.81 F.1.2.1 codes them.
fn category(v: i32) -> (u32, u32) {
    let m = v.unsigned_abs();
    let size = 32 - m.leading_zeros();
    let bits = if v < 0 {
        (v - 1) as u32 & ((1 << size) - 1)
    } else {
        v as u32
    };
    (size, bits)
}

#[allow(clippy::too_many_arguments)]
fn encode_block(
    w: &mut BitWriter,
    plane: &[u8],
    pw: usize,
    ph: usize,
    bx: usize,
    by: usize,
    q: &[u16; 64],
    dc: &[(u16, u8); 256],
    ac: &[(u16, u8); 256],
    pred: &mut i32,
) {
    let mut px = [0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            let (sx, sy) = ((bx + x).min(pw - 1), (by + y).min(ph - 1));
            px[y * 8 + x] = f32::from(plane[sy * pw + sx]) - 128.0;
        }
    }
    let mut coef = [0i32; 64];
    for v in 0..8 {
        for u in 0..8 {
            let cu = if u == 0 {
                std::f32::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let cv = if v == 0 {
                std::f32::consts::FRAC_1_SQRT_2
            } else {
                1.0
            };
            let mut s = 0f32;
            for y in 0..8 {
                for x in 0..8 {
                    s += px[y * 8 + x]
                        * (((2 * x + 1) as f32 * u as f32 * std::f32::consts::PI) / 16.0).cos()
                        * (((2 * y + 1) as f32 * v as f32 * std::f32::consts::PI) / 16.0).cos();
                }
            }
            coef[v * 8 + u] = (0.25 * cu * cv * s / f32::from(q[v * 8 + u])).round() as i32;
        }
    }
    let (size, bits) = category(coef[0] - *pred);
    *pred = coef[0];
    let (c, l) = dc[size as usize];
    w.put(u32::from(c), u32::from(l));
    w.put(bits, size);
    let mut run = 0;
    for &z in &ZIGZAG[1..] {
        let v = coef[z];
        if v == 0 {
            run += 1;
            continue;
        }
        while run > 15 {
            let (c, l) = ac[0xf0];
            w.put(u32::from(c), u32::from(l));
            run -= 16;
        }
        let (size, bits) = category(v);
        let (c, l) = ac[((run << 4) | size) as usize];
        w.put(u32::from(c), u32::from(l));
        w.put(bits, size);
        run = 0;
    }
    if run > 0 {
        let (c, l) = ac[0];
        w.put(u32::from(c), u32::from(l));
    }
}

/// Planar Y, Cb, Cr (see [`csc`]) to a complete baseline JFIF JPEG.
fn encode_baseline(s: &Setup, planes: &[u8], ratio: u32) -> Vec<u8> {
    let (cw, ch) = s.chroma();
    let (y, rest) = planes.split_at(s.w * s.h);
    let (cb, cr) = rest.split_at(cw * ch);
    let (lq, cq) = (quant(&LUMA_Q, ratio), quant(&CHROMA_Q, ratio));
    let mut out = vec![
        0xff, 0xd8, 0xff, 0xe0, 0, 16, b'J', b'F', b'I', b'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0,
    ];
    for (id, table) in [(0u8, &lq), (1, &cq)] {
        out.extend([0xff, 0xdb, 0, 67, id]);
        out.extend(ZIGZAG.iter().map(|&z| table[z] as u8));
    }
    let vsamp = if s.vsub { 2 } else { 1 };
    out.extend([0xff, 0xc0, 0, 17, 8]);
    out.extend((s.h as u16).to_be_bytes());
    out.extend((s.w as u16).to_be_bytes());
    out.extend([3, 1, 0x20 | vsamp, 0, 2, 0x11, 1, 3, 0x11, 1]);
    for (class_id, bits, values) in [
        (0x00u8, &DC_LUMA_BITS, &DC_VALUES[..]),
        (0x10, &AC_LUMA_BITS, &AC_LUMA_VALUES[..]),
        (0x01, &DC_CHROMA_BITS, &DC_VALUES[..]),
        (0x11, &AC_CHROMA_BITS, &AC_CHROMA_VALUES[..]),
    ] {
        out.extend([0xff, 0xc4]);
        out.extend(((3 + 16 + values.len()) as u16).to_be_bytes());
        out.push(class_id);
        out.extend(bits);
        out.extend(values);
    }
    out.extend([0xff, 0xda, 0, 12, 3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0]);
    let (dcl, acl) = (
        huff_codes(&DC_LUMA_BITS, &DC_VALUES),
        huff_codes(&AC_LUMA_BITS, &AC_LUMA_VALUES),
    );
    let (dcc, acc) = (
        huff_codes(&DC_CHROMA_BITS, &DC_VALUES),
        huff_codes(&AC_CHROMA_BITS, &AC_CHROMA_VALUES),
    );
    let mut w = BitWriter {
        out: Vec::new(),
        acc: 0,
        n: 0,
    };
    let mcu_h = 8 * vsamp as usize;
    let mut pred = [0i32; 3];
    for my in (0..s.h).step_by(mcu_h) {
        for mx in (0..s.w).step_by(16) {
            for dy in (0..mcu_h).step_by(8) {
                for dx in [0, 8] {
                    encode_block(
                        &mut w,
                        y,
                        s.w,
                        s.h,
                        mx + dx,
                        my + dy,
                        &lq,
                        &dcl,
                        &acl,
                        &mut pred[0],
                    );
                }
            }
            let (bx, by) = (mx / 2, if s.vsub { my / 2 } else { my });
            encode_block(&mut w, cb, cw, ch, bx, by, &cq, &dcc, &acc, &mut pred[1]);
            encode_block(&mut w, cr, cw, ch, bx, by, &cq, &dcc, &acc, &mut pred[2]);
        }
    }
    w.flush();
    out.extend(w.out);
    out.extend([0xff, 0xd9]);
    out
}

/// int sceJpegEncoderSetValidRegion(SceJpegEncoderContext context, int inWidth,
///     int inHeight)
#[hostcall]
pub(super) fn set_valid_region(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    context: Ptr,
    in_width: u32,
    in_height: u32,
) -> i32 {
    if context.is_null() {
        SCE_JPEGENC_ERROR_INVALID_POINTER
    } else {
        ctx.write_u32(context.addr() + 28, in_width);
        ctx.write_u32(context.addr() + 32, in_height);
        0
    }
}

#[cfg(test)]
mod tests {
    use crate::host::VitaState;
    use crate::{DeterministicWorld, SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::{REG_COUNT, SP};

    const CTX: u32 = 0x100;
    const RGBA: u32 = 0x2000;
    const YUV: u32 = 0x8000;
    const OUT: u32 = 0xC000;
    const STACK: u32 = 0x1F00;

    fn call(bytes: &mut [u8], st: &mut VitaState, nid: u32, args: &[u32]) -> i32 {
        let mut regs = [0u32; REG_COUNT];
        for (i, a) in args.iter().take(4).enumerate() {
            regs[i] = *a;
        }
        regs[SP] = STACK;
        for (i, a) in args.iter().skip(4).enumerate() {
            let at = (STACK as usize) + 4 * i;
            bytes[at..at + 4].copy_from_slice(&a.to_le_bytes());
        }
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut mem = SliceMemory(bytes);
        let mut ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        super::super::dispatch(crate::nid::lib::SCE_GXM, nid, &mut ctx, st);
        regs[0] as i32
    }

    /// A gradient through Csc and Encode comes back out of a JPEG decoder as the same picture:
    /// the output is a complete JPEG of the title's pixels, in the framebuffer's R,G,B,A order.
    #[test]
    fn csc_then_encode_is_a_jpeg_of_the_input() {
        use crate::nid::services as sv;
        let (w, h) = (40u32, 24u32);
        let mut bytes = vec![0u8; 0x20000];
        for y in 0..h {
            for x in 0..w {
                let p = (RGBA + (y * w + x) * 4) as usize;
                bytes[p..p + 4].copy_from_slice(&[(x * 6) as u8, (y * 10) as u8, 200, 255]);
            }
        }
        let mut st = VitaState::new(0, 0x20000, Box::new(DeterministicWorld::default()));
        assert_eq!(
            call(
                &mut bytes,
                &mut st,
                sv::JPEGENC_INIT,
                &[CTX, w, h, 8 | 16, OUT, 0x8000]
            ),
            0
        );
        assert_eq!(
            call(
                &mut bytes,
                &mut st,
                sv::JPEGENC_CSC,
                &[CTX, YUV, RGBA, w, 0]
            ),
            0
        );
        let n = call(&mut bytes, &mut st, sv::JPEGENC_ENCODE, &[CTX, YUV]);
        assert!(n > 0, "encode answered {n:#x}");
        let jpeg = &bytes[OUT as usize..OUT as usize + n as usize];
        let (dw, dh, rgb) = super::super::jpeg::decode_rgb(jpeg).expect("a decodable JPEG");
        assert_eq!((dw, dh), (w, h));
        let mut err = 0u64;
        for y in 0..h {
            for x in 0..w {
                let p = ((y * w + x) * 3) as usize;
                let want = [(x * 6) as i64, (y * 10) as i64, 200];
                for c in 0..3 {
                    err += (i64::from(rgb[p + c]) - want[c]).unsigned_abs();
                }
            }
        }
        let mean = err as f64 / f64::from(w * h * 3);
        assert!(mean < 6.0, "mean error {mean}");
        // An output buffer too small for the JPEG is refused, not truncated.
        call(
            &mut bytes,
            &mut st,
            sv::JPEGENC_SET_OUTPUT_ADDR,
            &[CTX, OUT, 16],
        );
        assert_eq!(
            call(&mut bytes, &mut st, sv::JPEGENC_ENCODE, &[CTX, YUV]),
            super::SCE_JPEGENC_ERROR_INSUFFICIENT_BUFFER
        );
    }
}
