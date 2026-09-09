//! SceJpeg: the MJPEG decoder, lifecycle and decode.
//!
//! `psp2/jpeg.h` publishes the prototypes and the 0x34-byte `SceJpegOutputInfo` layout, so
//! the interface is not guessed. The NIDs are the henkaku wiki's `SceAvcodecUser` export
//! list.
//!
//! The lifecycle calls are bookkeeping: a decoder pool's only observable property here is
//! that it exists.
//!
//! >>> THE DECODE HALF USED TO BE ABSENT ON PURPOSE, and it is here now because the reason
//! it was absent stopped being true. The old note said a decode that hands back an untouched
//! buffer puts a frame on screen no decoder produced, and that the hard-fail was the right
//! outcome - both correct. What changed is that this module now DECODES, with a real JPEG
//! decoder, so there is no untouched buffer to hand back. A retail fighting title decodes
//! JPEGs for its loading art before it reaches any menu, so the alternative was not a
//! cautious stub, it was never seeing a fight.
//!
//! # The four decode entry points, and only those
//! `sceJpegGetOutputInfo`, `sceJpegDecodeMJpegYCbCr`, `sceJpegMJpegCsc` and `sceJpegCsc` -
//! the four that title imports [[vitaslop-implement-all-called-nids]]. It never calls the
//! init/finish pair, which is why they stay as they were rather than growing state the
//! decode depends on.
//!
//! # WHERE THE SHAPES COME FROM, AND WHERE THEY DO NOT
//! The argument lists and the output struct are FACTS from the header. What no published
//! source carries is the `format` / `mode` / `colorSpace` / `sampling` ENUMERATIONS - the
//! ENCODER's constants are published (`psp2common/jpegenc.h`), the decoder's are not. So
//! every one of those values is REPORTED on its first sighting rather than silently
//! interpreted, and the report is what a later reader widens this module from. See
//! [`report_arguments`].
//!
//! # The layout this decoder produces, and why the picture is exact
//! `sceJpegDecodeMJpegYCbCr` fills a buffer the guest allocated from the `outputSize` this
//! module reported, and `sceJpegMJpegCsc` converts that buffer to RGBA. Both halves are
//! ours, so the layout is one this module CHOOSES and STATES: planar 4:4:4, a full-resolution
//! Y plane followed by full-resolution Cb and Cr planes, `width` bytes per row. Chroma is
//! upsampled to full resolution by the decode, which is what lets one layout serve every
//! JPEG the title carries whatever that image's own subsampling is.
//!
//! The planes hold REAL luma and chroma, not a private encoding: a title that reads the
//! buffer itself - uploading it as a texture, say - sees what the API promises.
//!
//! And the RGBA the guest finally gets does NOT come back through those planes. The decode
//! keeps the RGB it produced, keyed by the buffer address it filled, and the colour-space
//! conversion hands that back when the pointers match. Two 8-bit integer conversions in a row
//! are a real loss on saturated colour, and it would be a loss this emulator introduced
//! rather than one the guest's own data has. The planes are still written and the conversion
//! still works from them when the cache does not match, so nothing depends on the cache being
//! warm. [[vitaslop-never-trade-quality]]

use crate::host::{GuestCtx, Ptr, VitaState};
// The `#[hostcall]` macro emits fully-qualified paths for these, so a module of nothing
// but host calls needs no plain import of them.
use crate::hostcall;

/// Whether an MJPEG decoder pool is currently initialised, so a double-init or a finish
/// without an init is answered rather than silently accepted.
static INITIALISED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `SCE_JPEG_ERROR_INVALID_STATE`, per the facility's published error range.
const SCE_JPEG_ERROR_INVALID_STATE: i32 = 0x8065_0004u32 as i32;

/// int sceJpegInitMJpeg(SceInt32 decoderCount)
///
/// Stands up a pool of `decoderCount` MJPEG decoders. Nothing is allocated on the host:
/// the pool's only observable property is that it exists, and the decode calls that would
/// consume it are not implemented (see the module docs).
#[hostcall]
pub(super) fn init_mjpeg(_ctx: &mut GuestCtx, _st: &mut VitaState, _decoder_count: u32) -> i32 {
    use std::sync::atomic::Ordering;
    if INITIALISED.swap(true, Ordering::Relaxed) {
        // Already initialised - a title that inits twice has a bug, and hearing about it
        // is more useful than a second success.
        SCE_JPEG_ERROR_INVALID_STATE
    } else {
        0
    }
}

/// int sceJpegFinishMJpeg(void)
#[hostcall]
pub(super) fn finish_mjpeg(_ctx: &mut GuestCtx, _st: &mut VitaState) -> i32 {
    use std::sync::atomic::Ordering;
    if INITIALISED.swap(false, Ordering::Relaxed) {
        0
    } else {
        SCE_JPEG_ERROR_INVALID_STATE
    }
}

/// `SceJpegOutputInfo`, 0x34 bytes (vitasdk `psp2/jpeg.h`):
/// `colorSpace:i32, width:u16, height:u16, outputSize:u32, unk_0xc:u32, unk_0x10:u32,
/// pitch[4]: { x:u32, y:u32 }`.
const OUTPUT_INFO_BYTES: usize = 0x34;

/// The `colorSpace` this module reports, and the one layout it is prepared to produce.
///
/// >>> NOT A PUBLISHED CONSTANT. The decoder-side enumeration appears in no header and in
/// neither wiki. What IS published is that the field is signed and that the ENCODER - a
/// different enum on the other side of the API - numbers its YCbCr formats 8 (4:2:0) and
/// 9 (4:2:2) with 0 for a packed 32-bit colour, so a planar 4:4:4 decode has no published
/// number at all.
///
/// Zero is therefore reported, and the first call SAYS so. If a title branches on this field,
/// that branch is what will name the real encoding - and the report is what makes the branch
/// visible instead of a silent wrong turn.
const REPORTED_COLOR_SPACE: i32 = 0;

/// Error codes. The `0x8065_02xx` facility is the published JPEG one (the encoder's codes are
/// in `psp2common/jpegenc.h`); the decoder's own list is not published, so these use the same
/// facility with the meanings their names carry and are only ever returned for a condition
/// this module can state exactly.
const SCE_JPEG_ERROR_INVALID_POINTER: i32 = 0x8065_0205u32 as i32;
const SCE_JPEG_ERROR_INVALID_DATA: i32 = 0x8065_0203u32 as i32;
const SCE_JPEG_ERROR_INSUFFICIENT_BUFFER: i32 = 0x8065_0201u32 as i32;

/// What one decode produced, kept so the colour-space conversion does not have to invert the
/// planes it wrote. Keyed by the guest buffer the decode filled: a second decode into the same
/// buffer replaces it, and a conversion from any other pointer ignores it.
#[derive(Default)]
pub struct JpegState {
    /// The guest address and byte length of the YCbCr buffer the last decode filled.
    cached_at: Option<(u32, u32)>,
    /// The last decode's dimensions and its RGB bytes (3 per pixel, row-major).
    cached: Option<(u32, u32, Vec<u8>)>,
    /// Whether the argument report has been made for each entry point, so the panel carries
    /// one line per entry point rather than one per decode.
    reported: [bool; 4],
    /// Whether an undecodable image has already been reported.
    reported_undecodable: bool,
}

/// Say, ONCE per entry point, what arguments the title actually passes.
///
/// The `format`, `mode` and `sampling` enumerations are not published anywhere this project
/// can read, so the values a real title uses ARE the documentation. A STATUS line, not a
/// warning: nothing is wrong with a title passing them, and what the line records is the
/// evidence the next reader needs.
fn report_arguments(st: &mut VitaState, slot: usize, what: &str, args: &str) {
    if st.jpeg.reported[slot] {
        return;
    }
    st.jpeg.reported[slot] = true;
    tracing::info!(
        target: "vitaslop::status",
        "{what}: {args}. The decoder-side format/mode/sampling enumerations are published in \
         no header and in neither wiki, so these values are the only statement of what this \
         title asks for - they are recorded here rather than interpreted."
    );
}

/// Say, once, that bytes the title called a JPEG did not decode as one.
///
/// A WARNING and not a status: a title asking this library to decode something expects a
/// picture back, and an image that silently fails to appear is exactly the class of defect
/// this project reports rather than absorbs.
fn report_undecodable(st: &mut VitaState, size: u32, bytes: &[u8]) {
    if st.jpeg.reported_undecodable {
        return;
    }
    st.jpeg.reported_undecodable = true;
    let head: Vec<String> = bytes.iter().take(8).map(|b| format!("{b:02x}")).collect();
    tracing::warn!(
        target: "vitaslop::gxm",
        size,
        head = head.join(" "),
        "sceJpeg: the bytes this title handed the decoder are not a JPEG it can read, so no \
         picture is produced. The first bytes are above; a JPEG begins ff d8 ff."
    );
}

/// Decode a JPEG's HEADERS only, for width and height. `None` means the bytes are not a JPEG
/// this decoder can read, which is a real answer rather than a failure to try.
fn header_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    let mut d = zune_jpeg::JpegDecoder::new(zune_core::bytestream::ZCursor::new(bytes));
    d.decode_headers().ok()?;
    let info = d.info()?;
    Some((u32::from(info.width), u32::from(info.height)))
}

/// Decode a JPEG to interleaved RGB, three bytes per pixel, row-major.
fn decode_rgb(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let opts = zune_core::options::DecoderOptions::default()
        .jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGB);
    let mut d =
        zune_jpeg::JpegDecoder::new_with_options(zune_core::bytestream::ZCursor::new(bytes), opts);
    let pixels = d.decode().ok()?;
    let info = d.info()?;
    Some((u32::from(info.width), u32::from(info.height), pixels))
}

/// BT.601 full-range RGB -> YCbCr, the conversion JFIF itself specifies. Integer and rounded.
fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let (r, g, b) = (i32::from(r), i32::from(g), i32::from(b));
    let y = (19595 * r + 38470 * g + 7471 * b + 32768) >> 16;
    let cb = 128 + ((-11056 * r - 21712 * g + 32768 * b + 32768) >> 16);
    let cr = 128 + ((32768 * r - 27440 * g - 5328 * b + 32768) >> 16);
    (y.clamp(0, 255) as u8, cb.clamp(0, 255) as u8, cr.clamp(0, 255) as u8)
}

/// BT.601 full-range YCbCr -> RGB (JFIF), the inverse of `rgb_to_ycbcr`.
fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = i32::from(y);
    let cb = i32::from(cb) - 128;
    let cr = i32::from(cr) - 128;
    let r = y + ((91881 * cr + 32768) >> 16);
    let g = y - ((22554 * cb + 46802 * cr + 32768) >> 16);
    let b = y + ((116130 * cb + 32768) >> 16);
    (r.clamp(0, 255) as u8, g.clamp(0, 255) as u8, b.clamp(0, 255) as u8)
}

/// The two calls the browser-side bench drives, so the number it reports is THIS decoder on
/// the real path rather than a copy of it. See `vitaslop_web::frontend::jpeg_bench`.
pub fn bench_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    header_dimensions(bytes)
}

/// Decode and return the pixel count, so the caller cannot have the work optimised away.
pub fn bench_decode(bytes: &[u8]) -> usize {
    decode_rgb(bytes).map_or(0, |(_, _, rgb)| rgb.len())
}

/// `int sceJpegGetOutputInfo(const SceUInt8 *jpegData, SceSize jpegSize, SceInt32 format,
/// SceInt32 mode, SceJpegOutputInfo *output)`
///
/// The sizes reported here are what the guest allocates from, so they describe the layout
/// `decode_mjpeg_ycbcr` actually writes: planar 4:4:4, three full-resolution planes.
#[hostcall]
pub(super) fn get_output_info(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    jpeg_data: Ptr,
    jpeg_size: u32,
    format: i32,
    mode: i32,
    output: Ptr,
) -> i32 {
    do_get_output_info(ctx, st, jpeg_data, jpeg_size, format, mode, output)
}

/// The body, as a plain function: a `#[hostcall]` body is spliced into a generated wrapper,
/// so a guard clause's `return` would leave the wrapper instead of the call.
/// [[vitaslop-hostcall-body-cannot-early-return]]
fn do_get_output_info(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    jpeg_data: Ptr,
    jpeg_size: u32,
    format: i32,
    mode: i32,
    output: Ptr,
) -> i32 {
    report_arguments(
        st,
        0,
        "sceJpegGetOutputInfo",
        &format!("format={format:#x} mode={mode:#x} jpegSize={jpeg_size}"),
    );
    if jpeg_data.0 == 0 || output.0 == 0 || jpeg_size == 0 {
        return SCE_JPEG_ERROR_INVALID_POINTER;
    }
    let bytes = ctx.read_bytes(jpeg_data.0, jpeg_size as usize);
    let Some((w, h)) = header_dimensions(&bytes) else {
        report_undecodable(st, jpeg_size, &bytes);
        return SCE_JPEG_ERROR_INVALID_DATA;
    };
    let plane = w * h;
    let mut info = [0u8; OUTPUT_INFO_BYTES];
    fn put32(info: &mut [u8; OUTPUT_INFO_BYTES], off: usize, v: u32) {
        info[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    put32(&mut info, 0x00, REPORTED_COLOR_SPACE as u32);
    info[0x04..0x06].copy_from_slice(&(w as u16).to_le_bytes());
    info[0x06..0x08].copy_from_slice(&(h as u16).to_le_bytes());
    // Three full-resolution planes, so three times one plane. `unk_0xc` and `unk_0x10` stay
    // zero: no published source says what they carry, and a number invented for them would be
    // one a title could branch on.
    put32(&mut info, 0x08, plane * 3);
    // `pitch[i] = { x, y }`: bytes per row and number of rows, for the four possible planes.
    // The fourth is empty - a JPEG has at most three components.
    for (i, (x, y)) in [(w, h), (w, h), (w, h), (0, 0)].into_iter().enumerate() {
        put32(&mut info, 0x14 + i * 8, x);
        put32(&mut info, 0x18 + i * 8, y);
    }
    ctx.write_bytes(output.0, &info);
    0
}

/// `int sceJpegDecodeMJpegYCbCr(const SceUInt8 *jpegData, SceSize jpegSize, SceUInt8 *output,
/// SceSize outputSize, void *buffer, SceSize bufferSize)`
///
/// >>> SIX ARGUMENTS, AND vitasdk'S HEADER SAYS SEVEN. `psp2/jpeg.h` puts a `SceInt32 mode`
/// third; the title does not. It is MEASURED from the callsite, and the closure is exact:
///
/// ```text
///   r0 = 0x916c0000   the JPEG bytes
///   r1 = 69687        their length
///   r2 = 0x91940000   a GUEST ADDRESS - so not a mode
///   r3 = 0x000aaa00   699,392, which is the 698,880 this module reported as `outputSize`
///                     for that image ROUNDED UP TO 512 - so not a pointer either
///   sp+0, sp+4        both zero: the coefficient buffer and its size
/// ```
///
/// Read as seven, `outputSize` would be the zero at `sp+0` and `output` would be `0xaaa00`,
/// which is not an address the guest allocated anything at. Read as six, every argument is the
/// value its name says, and `r3` is a number only this module could have produced - it came
/// back out of the struct `sceJpegGetOutputInfo` filled two calls earlier. The header's
/// sibling `psp2/jpegarm.h` orders its own two enum arguments the other way round from
/// `psp2/jpeg.h`, so the published headers do not agree with each other about this family
/// either. [[vitaslop-re-undocumented-nid-from-callsite]]
///
/// Returns the decoded byte count, which is what the caller of a decode that succeeded expects
/// to be able to hand to the colour-space conversion.
#[hostcall]
pub(super) fn decode_mjpeg_ycbcr(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    jpeg_data: Ptr,
    jpeg_size: u32,
    output: Ptr,
    output_size: u32,
    work: Ptr,
    work_size: u32,
) -> i32 {
    do_decode_mjpeg_ycbcr(ctx, st, jpeg_data, jpeg_size, output, output_size, work, work_size)
}

/// The body, as a plain function - see [`do_get_output_info`] for why.
#[allow(clippy::too_many_arguments)]
fn do_decode_mjpeg_ycbcr(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    jpeg_data: Ptr,
    jpeg_size: u32,
    output: Ptr,
    output_size: u32,
    work: Ptr,
    work_size: u32,
) -> i32 {
    report_arguments(
        st,
        1,
        "sceJpegDecodeMJpegYCbCr",
        &format!(
            "jpegSize={jpeg_size} output={:#010x} outputSize={output_size:#x}              work={:#010x} workSize={work_size:#x}",
            output.0, work.0
        ),
    );
    if jpeg_data.0 == 0 || output.0 == 0 {
        return SCE_JPEG_ERROR_INVALID_POINTER;
    }
    let bytes = ctx.read_bytes(jpeg_data.0, jpeg_size as usize);
    let Some((w, h, rgb)) = decode_rgb(&bytes) else {
        report_undecodable(st, jpeg_size, &bytes);
        return SCE_JPEG_ERROR_INVALID_DATA;
    };
    let plane = (w as usize) * (h as usize);
    let needed = plane * 3;
    if (output_size as usize) < needed {
        // The guest sized its buffer from `sceJpegGetOutputInfo`, so a short one means the two
        // disagree about the layout - a finding, not a case to truncate through.
        tracing::warn!(
            target: "vitaslop::gxm",
            output_size, needed, width = w, height = h,
            "sceJpegDecodeMJpegYCbCr: the guest's output buffer is SMALLER than the planar \
             4:4:4 layout sceJpegGetOutputInfo reported for this image, so the two disagree \
             about what this decoder produces. Nothing is written."
        );
        return SCE_JPEG_ERROR_INSUFFICIENT_BUFFER;
    }
    let mut planes = vec![0u8; needed];
    for i in 0..plane {
        let (y, cb, cr) = rgb_to_ycbcr(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
        planes[i] = y;
        planes[plane + i] = cb;
        planes[plane * 2 + i] = cr;
    }
    ctx.write_bytes(output.0, &planes);
    st.jpeg.cached_at = Some((output.0, needed as u32));
    st.jpeg.cached = Some((w, h, rgb));
    needed as i32
}

/// `int sceJpegMJpegCsc(SceUInt8 *rgba, const SceUInt8 *yuv, SceSize yuvSize,
/// SceInt32 imageWidth, SceInt32 format, SceInt32 sampling)`
#[hostcall]
pub(super) fn mjpeg_csc(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    rgba: Ptr,
    yuv: Ptr,
    yuv_size: u32,
    image_width: i32,
    format: i32,
    sampling: i32,
) -> i32 {
    report_arguments(
        st,
        2,
        "sceJpegMJpegCsc",
        &format!(
            "format={format:#x} sampling={sampling:#x} imageWidth={image_width} \
             yuvSize={yuv_size:#x}"
        ),
    );
    csc(ctx, st, rgba, yuv, image_width)
}

/// `int sceJpegCsc(...)` - the non-MJpeg twin. No published header carries it, so its argument
/// list is taken to be the same as its sibling's and REPORTED, in the way
/// [[vitaslop-re-undocumented-nid-from-callsite]] describes; on every other member of this
/// library the two names differ only in which container the image came from, not in what the
/// call does.
#[hostcall]
pub(super) fn plain_csc(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    rgba: Ptr,
    yuv: Ptr,
    yuv_size: u32,
    image_width: i32,
    format: i32,
    sampling: i32,
) -> i32 {
    report_arguments(
        st,
        3,
        "sceJpegCsc",
        &format!(
            "format={format:#x} sampling={sampling:#x} imageWidth={image_width} \
             yuvSize={yuv_size:#x}"
        ),
    );
    csc(ctx, st, rgba, yuv, image_width)
}

/// The shared conversion: exact from the decode cache when the source pointer is the buffer
/// the last decode filled, and read back from the planes that decode wrote otherwise.
fn csc(ctx: &mut GuestCtx, st: &mut VitaState, rgba: Ptr, yuv: Ptr, image_width: i32) -> i32 {
    if rgba.0 == 0 || yuv.0 == 0 || image_width <= 0 {
        return SCE_JPEG_ERROR_INVALID_POINTER;
    }
    let width = image_width as u32;
    let Some((cached_addr, cached_len)) = st.jpeg.cached_at else {
        tracing::warn!(
            target: "vitaslop::gxm",
            "sceJpeg colour conversion was asked for a buffer no decode of this run has \
             filled, so the plane heights are not knowable from it. Nothing is written."
        );
        return SCE_JPEG_ERROR_INVALID_DATA;
    };
    let (w, h, rgb) = match (cached_addr == yuv.0, st.jpeg.cached.as_ref()) {
        (true, Some((w, h, rgb))) => (*w, *h, rgb.clone()),
        _ => {
            // A different buffer: read the planes back. The layout is the one this module
            // wrote and reported, and the caller's width names the row stride.
            let plane = (cached_len / 3) as usize;
            let h = (plane as u32) / width.max(1);
            let planes = ctx.read_bytes(yuv.0, cached_len as usize);
            if planes.len() < plane * 3 {
                return SCE_JPEG_ERROR_INVALID_DATA;
            }
            let mut rgb = vec![0u8; plane * 3];
            for i in 0..plane {
                let (r, g, b) = ycbcr_to_rgb(planes[i], planes[plane + i], planes[plane * 2 + i]);
                rgb[i * 3] = r;
                rgb[i * 3 + 1] = g;
                rgb[i * 3 + 2] = b;
            }
            (width, h, rgb)
        }
    };
    if w == 0 || h == 0 {
        return SCE_JPEG_ERROR_INVALID_DATA;
    }
    // RGBA8888, `image_width` pixels per output row: the destination stride is the caller's
    // width, which for a texture is not always the image's own.
    let mut out = vec![0u8; (width as usize) * (h as usize) * 4];
    for y in 0..h as usize {
        for x in 0..width as usize {
            let src = y * (w as usize) + x.min(w as usize - 1);
            let dst = (y * (width as usize) + x) * 4;
            if src * 3 + 2 >= rgb.len() {
                break;
            }
            out[dst] = rgb[src * 3];
            out[dst + 1] = rgb[src * 3 + 1];
            out[dst + 2] = rgb[src * 3 + 2];
            out[dst + 3] = 0xff;
        }
    }
    ctx.write_bytes(rgba.0, &out);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two conversions are inverses to within the rounding each does, which is what makes
    /// the planes this module writes real luma and chroma rather than a private encoding.
    #[test]
    fn ycbcr_round_trips_within_two_steps_on_ordinary_colour() {
        let mut worst = 0i32;
        for r in (0..=255i32).step_by(17) {
            for g in (0..=255i32).step_by(17) {
                for b in (0..=255i32).step_by(17) {
                    let (y, cb, cr) = rgb_to_ycbcr(r as u8, g as u8, b as u8);
                    let (r2, g2, b2) = ycbcr_to_rgb(y, cb, cr);
                    for (a, c) in [(r, i32::from(r2)), (g, i32::from(g2)), (b, i32::from(b2))] {
                        worst = worst.max((a - c).abs());
                    }
                }
            }
        }
        assert!(worst <= 2, "round trip drifted by {worst} steps");
    }

    /// Grey stays grey: the chroma planes sit at 128 and the luma is the grey level. The one
    /// point of the conversion a reader can check by eye.
    #[test]
    fn grey_has_neutral_chroma() {
        for v in [0u8, 64, 128, 200, 255] {
            let (y, cb, cr) = rgb_to_ycbcr(v, v, v);
            assert_eq!((y, cb, cr), (v, 128, 128));
        }
    }
}

/// What this decoder costs, on a real image, so "is it fast enough" is a number.
///
/// `VITASLOP_JPEG_BENCH=<path to a .jpg>` selects the image. It is not a comparison against
/// another decoder - it is the only figure that decides anything here, which is whether a
/// title's loading art costs a frame or a blink.
#[cfg(test)]
#[test]
#[ignore = "needs an image; set VITASLOP_JPEG_BENCH=<path to a .jpg>"]
fn how_long_one_decode_takes() {
    let Ok(path) = std::env::var("VITASLOP_JPEG_BENCH") else { return };
    let bytes = std::fs::read(&path).expect("the bench image");
    let (w, h, _) = decode_rgb(&bytes).expect("it decodes");
    // Warm, then time a run of decodes: one decode is short enough that the clock's own
    // resolution would be part of the answer.
    for _ in 0..3 {
        decode_rgb(&bytes);
    }
    const N: u32 = 50;
    let t = std::time::Instant::now();
    for _ in 0..N {
        std::hint::black_box(decode_rgb(&bytes));
    }
    let per = t.elapsed().as_secs_f64() / f64::from(N);
    println!(
        "{path}: {w}x{h} from {} KB in {:.2} ms per decode = {:.1} Mpixel/s",
        bytes.len() / 1024,
        per * 1000.0,
        f64::from(w * h) / per / 1.0e6
    );
}
