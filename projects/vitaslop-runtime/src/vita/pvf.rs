//! ScePvf: the PVF (Vita) vector-font engine.
//!
//! A title creates a font library, opens its own TrueType/OpenType file (or a system
//! font), sets the em/resolution/char size, queries metrics, and rasterizes glyphs to
//! coverage bitmaps it uploads as textures. The actual scaling/hinting/rasterization
//! lives behind a swappable backend in [`crate::font`]; these handlers only marshal the
//! Vita structs and drive the [`FontLibrary`](crate::font::FontLibrary) held in
//! `VitaState`.
//!
//! Handle model: `ScePvfLibId`/`ScePvfFontId` are opaque `void *` to the guest; we hand
//! back the font library's own handles and resolve them back on every call.
//!
//! Metrics structs use 26.6 fixed point (the `*64` fields): a pixel value `v` is encoded
//! as `round(v * 64)`, matching FreeType (which libpvf wraps on hardware).

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;

// ScePvf error codes (from the MIT vita-headers pvf.h).
const SCE_PVF_ERROR_LIBID: i32 = 0x8046_0002u32 as i32;
const SCE_PVF_ERROR_ARG: i32 = 0x8046_0003u32 as i32;
const SCE_PVF_ERROR_NOFILE: i32 = 0x8046_0004u32 as i32;
const SCE_PVF_ERROR_NOGLYPH: i32 = 0x8046_000Fu32 as i32;

// ScePvfImageBufferPixelFormatType.
const SCE_PVF_USERIMAGE_DIRECT4_L: u32 = 0;
const SCE_PVF_USERIMAGE_DIRECT8: u32 = 2;

/// Encode a fractional-pixel value as 26.6 fixed point (the ScePvf `*64` convention).
fn fx(v: f32) -> i32 {
    (v * 64.0).round() as i32
}

/// ScePvfLibId scePvfNewLib(ScePvfInitRec *initParam, ScePvfError *errorCode)
/// Create a font library instance, returning its (non-null) handle. The init record
/// (custom allocators / external glyph cache) is not needed - the engine owns its own
/// allocation and cache.
#[hostcall]
pub(super) fn new_lib(ctx: &mut GuestCtx, st: &mut VitaState, _init_param: Ptr, error_code: Ptr) -> u32 {
    let lib = st.fonts.new_lib();
    if !error_code.is_null() {
        ctx.write_u32(error_code.addr(), 0);
    }
    lib
}

/// ScePvfError scePvfDoneLib(ScePvfLibId libID)
/// Destroy a library and every font opened under it.
#[hostcall]
pub(super) fn done_lib(st: &mut VitaState, lib: u32) -> i32 {
    if st.fonts.done_lib(lib) {
        0
    } else {
        SCE_PVF_ERROR_LIBID
    }
}

/// ScePvfError scePvfSetEM(ScePvfLibId libID, ScePvfFloat32 emValue)
#[hostcall]
pub(super) fn set_em(st: &mut VitaState, lib: u32, em_value: f32) -> i32 {
    if st.fonts.set_em(lib, em_value) {
        0
    } else {
        SCE_PVF_ERROR_LIBID
    }
}

/// ScePvfError scePvfSetResolution(ScePvfLibId libID, ScePvfFloat32 hResolution,
///     ScePvfFloat32 vResolution)
#[hostcall]
pub(super) fn set_resolution(st: &mut VitaState, lib: u32, h_resolution: f32, v_resolution: f32) -> i32 {
    if st.fonts.set_resolution(lib, h_resolution, v_resolution) {
        0
    } else {
        SCE_PVF_ERROR_LIBID
    }
}

/// ScePvfFontId scePvfOpen(ScePvfLibId libID, ScePvfFontIndex fontIndex, ScePvfU32 mode,
///     ScePvfError *errorCode)
/// Open one of the system fonts by index. This offline oracle ships no system fonts
/// (they are Sony's copyrighted assets), so there is nothing to open; report NOFILE
/// rather than hand back a font id with no backing face. Titles that render their own
/// text ship a font and use `scePvfOpenUserFile` instead.
#[hostcall]
pub(super) fn open(ctx: &mut GuestCtx, _st: &mut VitaState, _lib: u32, _font_index: i32, _mode: u32, error_code: Ptr) -> u32 {
    if !error_code.is_null() {
        ctx.write_u32(error_code.addr(), SCE_PVF_ERROR_NOFILE as u32);
    }
    0
}

/// ScePvfFontId scePvfOpenUserFile(ScePvfLibId libID, ScePvfPointer filename,
///     ScePvfU32 mode, ScePvfError *errorCode)
/// Open a font from a file the title ships. The bytes are read from the guest
/// filesystem and parsed by the backend; a non-null font handle is returned on success.
#[hostcall]
pub(super) fn open_user_file(ctx: &mut GuestCtx, st: &mut VitaState, lib: u32, filename: Ptr, _mode: u32, error_code: Ptr) -> u32 {
    let path = super::iofilemgr::read_cstr(ctx, filename.addr());
    let (font, err) = match st.read_file(&path) {
        Some(bytes) => match st.fonts.open_user_file(lib, &bytes) {
            Some(f) => (f, 0),
            // Lib unknown or the bytes are not a font we can parse.
            None if !st.fonts.lib_exists(lib) => (0, SCE_PVF_ERROR_LIBID),
            None => (0, SCE_PVF_ERROR_NOFILE),
        },
        None => (0, SCE_PVF_ERROR_NOFILE),
    };
    tracing::debug!(target: "vitaslop::cb", path, font, err, "scePvfOpenUserFile");
    if !error_code.is_null() {
        ctx.write_u32(error_code.addr(), err as u32);
    }
    font
}

/// ScePvfFontId scePvfOpenUserMemory(ScePvfLibId libID, ScePvfPointer addr,
///     ScePvfU32 size, ScePvfError *errorCode)
///
/// Open a font from bytes the title already holds in GUEST MEMORY. Titles use this for a
/// font they unpacked from their own archive, which a path-based open cannot reach at all.
///
/// A zero-length or null buffer is rejected before the read rather than after: reading
/// zero bytes and then failing to parse them reports the same error by a longer route, but
/// a huge bogus `size` would first copy that much guest memory.
#[hostcall]
pub(super) fn open_user_memory(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    lib: u32,
    addr: Ptr,
    size: u32,
    error_code: Ptr,
) -> u32 {
    do_open_user_memory(ctx, st, lib, addr.addr(), size, error_code.addr())
}

fn do_open_user_memory(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    lib: u32,
    addr: u32,
    size: u32,
    error_code: u32,
) -> u32 {
    let (font, err) = if addr == 0 || size == 0 {
        (0, SCE_PVF_ERROR_ARG)
    } else {
        let bytes = ctx.read_bytes(addr, size as usize);
        match st.fonts.open_user_memory(lib, &bytes) {
            Some(f) => (f, 0),
            None if !st.fonts.lib_exists(lib) => (0, SCE_PVF_ERROR_LIBID),
            // The lib is fine, so the bytes are not a font this backend can parse.
            None => (0, SCE_PVF_ERROR_NOFILE),
        }
    };
    tracing::debug!(target: "vitaslop::cb", lib, size, font, err, "scePvfOpenUserMemory");
    if error_code != 0 {
        ctx.write_u32(error_code, err as u32);
    }
    font
}

/// ScePvfError scePvfClose(ScePvfFontId fontID)
///
/// Drop a font handle and, once no open font still uses its face, its cached glyphs.
///
/// A handle that was never issued - or was already closed - is refused with
/// `SCE_PVF_ERROR_ARG` rather than succeeding: a double close that reads as success hides
/// a title's own use-after-free from it, and this is the one call positioned to notice.
#[hostcall]
pub(super) fn close(st: &mut VitaState, font: u32) -> i32 {
    if st.fonts.close(font) {
        0
    } else {
        SCE_PVF_ERROR_ARG
    }
}

/// ScePvfError scePvfSetSkewValue(ScePvfFontId fontID, ScePvfFloat32 angleX,
///     ScePvfFloat32 angleY)
/// Faux-italic skew is a visual-only transform not yet modeled by the backend; accept
/// it as a no-op so a title that sets it still proceeds (glyphs render upright).
#[hostcall]
pub(super) fn set_skew_value(st: &mut VitaState, font: u32, _angle_x: f32, _angle_y: f32) -> i32 {
    if st.fonts.font_exists(font) {
        0
    } else {
        SCE_PVF_ERROR_ARG
    }
}

/// ScePvfError scePvfSetCharSize(ScePvfFontId fontID, ScePvfFloat32 hSize,
///     ScePvfFloat32 vSize)
/// Set the pixel em size for a font's subsequent metrics and rasterization.
#[hostcall]
pub(super) fn set_char_size(st: &mut VitaState, font: u32, h_size: f32, v_size: f32) -> i32 {
    if st.fonts.set_char_size(font, h_size, v_size) {
        0
    } else {
        SCE_PVF_ERROR_ARG
    }
}

/// ScePvfBool scePvfIsElement(ScePvfFontId fontID, ScePvfCharCode charCode)
/// Whether the font can render this character (SCE_PVF_TRUE / SCE_PVF_FALSE).
#[hostcall]
pub(super) fn is_element(st: &mut VitaState, font: u32, char_code: u32) -> u32 {
    u32::from(st.fonts.has_glyph(font, char_code & 0xFFFF))
}

/// ScePvfFloat32 scePvfPixelToPointH(ScePvfLibId libID, ScePvfFloat32 pixel,
///     ScePvfError *errorCode)
#[hostcall]
pub(super) fn pixel_to_point_h(ctx: &mut GuestCtx, st: &mut VitaState, lib: u32, pixel: f32, error_code: Ptr) -> f32 {
    let err = if st.fonts.lib_exists(lib) { 0 } else { SCE_PVF_ERROR_LIBID };
    if !error_code.is_null() {
        ctx.write_u32(error_code.addr(), err as u32);
    }
    st.fonts.pixel_to_point(lib, pixel, false)
}

/// ScePvfFloat32 scePvfPixelToPointV(ScePvfLibId libID, ScePvfFloat32 pixel,
///     ScePvfError *errorCode)
#[hostcall]
pub(super) fn pixel_to_point_v(ctx: &mut GuestCtx, st: &mut VitaState, lib: u32, pixel: f32, error_code: Ptr) -> f32 {
    let err = if st.fonts.lib_exists(lib) { 0 } else { SCE_PVF_ERROR_LIBID };
    if !error_code.is_null() {
        ctx.write_u32(error_code.addr(), err as u32);
    }
    st.fonts.pixel_to_point(lib, pixel, true)
}

/// Write a ScePvfIGlyphMetricsInfo (0x28: ten 26.6 fixed-point fields) at `off` in a
/// buffer, from fractional-pixel values in the field order the struct declares.
fn write_iglyph_metrics(buf: &mut [u8], off: usize, m: &crate::font::GlyphMetrics) {
    let fields = [
        m.width, m.height, m.ascender, m.descender,
        m.h_bearing_x, m.h_bearing_y, m.v_bearing_x, m.v_bearing_y,
        m.h_advance, m.v_advance,
    ];
    for (i, v) in fields.iter().enumerate() {
        buf[off + i * 4..off + i * 4 + 4].copy_from_slice(&fx(*v).to_le_bytes());
    }
}

/// ScePvfError scePvfGetCharInfo(ScePvfFontId fontID, ScePvfCharCode charCode,
///     ScePvfCharInfo *charInfo)
/// Fill the per-glyph info: bitmap dimensions/placement plus the 26.6 glyph metrics.
#[hostcall]
pub(super) fn get_char_info(ctx: &mut GuestCtx, st: &mut VitaState, font: u32, char_code: u32, char_info: Ptr) -> i32 {
    let exists = st.fonts.font_exists(font);
    match st.fonts.glyph(font, char_code & 0xFFFF) {
        Some((bmp, metrics)) => {
            // ScePvfCharInfo (0x40): bitmapWidth/Height/Pitch, bitmapLeft/Top,
            // glyphMetrics (0x28), reserved.
            let mut buf = [0u8; 0x40];
            buf[0..4].copy_from_slice(&bmp.width.to_le_bytes());
            buf[4..8].copy_from_slice(&bmp.height.to_le_bytes());
            buf[8..12].copy_from_slice(&bmp.width.to_le_bytes()); // 8-bit pitch == width
            buf[12..16].copy_from_slice(&metrics.bitmap_left.to_le_bytes());
            buf[16..20].copy_from_slice(&metrics.bitmap_top.to_le_bytes());
            write_iglyph_metrics(&mut buf, 20, metrics);
            if !char_info.is_null() {
                ctx.write_bytes(char_info.addr(), &buf);
            }
            0
        }
        None if exists => SCE_PVF_ERROR_NOGLYPH,
        None => SCE_PVF_ERROR_ARG,
    }
}

/// ScePvfError scePvfGetCharImageRect(ScePvfFontId fontID, ScePvfCharCode charCode,
///     ScePvfIrect *rect)
/// The glyph's bitmap dimensions (u16 width, u16 height) - what a title needs to size
/// the destination buffer before `scePvfGetCharGlyphImage`.
#[hostcall]
pub(super) fn get_char_image_rect(ctx: &mut GuestCtx, st: &mut VitaState, font: u32, char_code: u32, rect: Ptr) -> i32 {
    let exists = st.fonts.font_exists(font);
    match st.fonts.glyph(font, char_code & 0xFFFF) {
        Some((bmp, _)) => {
            let (w, h) = (bmp.width as u16, bmp.height as u16);
            if !rect.is_null() {
                let mut buf = [0u8; 4];
                buf[0..2].copy_from_slice(&w.to_le_bytes());
                buf[2..4].copy_from_slice(&h.to_le_bytes());
                ctx.write_bytes(rect.addr(), &buf);
            }
            0
        }
        None if exists => SCE_PVF_ERROR_NOGLYPH,
        None => SCE_PVF_ERROR_ARG,
    }
}

/// ScePvfError scePvfGetFontInfo(ScePvfFontId fontID, ScePvfFontInfo *fontInfo)
/// Fill the face-wide info: the maximum glyph metrics (26.6 and float forms) and the
/// glyph count. The style-info block (family/style codes, font/style/file names) is
/// left zeroed - it describes a system font selected from the font list, not a user
/// file, and a title that opened its own file does not read it back.
#[hostcall]
pub(super) fn get_font_info(ctx: &mut GuestCtx, st: &mut VitaState, font: u32, font_info: Ptr) -> i32 {
    match st.fonts.face_metrics(font) {
        None => SCE_PVF_ERROR_ARG,
        Some(fm) => {
            // Synthesize the "maximum glyph" metrics from the face metrics.
            let max = crate::font::GlyphMetrics {
                h_advance: fm.max_advance,
                v_advance: fm.height,
                h_bearing_x: 0.0,
                h_bearing_y: fm.ascender,
                v_bearing_x: 0.0,
                v_bearing_y: fm.ascender,
                width: fm.max_advance,
                height: fm.ascender - fm.descender,
                ascender: fm.ascender,
                descender: fm.descender,
                bitmap_left: 0,
                bitmap_top: 0,
                bitmap_width: 0,
                bitmap_height: 0,
            };
            // ScePvfFontInfo (0x130): maxIGlyphMetrics @0 (0x28), maxFGlyphMetrics @0x28
            // (0x28), numChars @0x50, fontStyleInfo @0x54 (0xD8), reserved @0x12C.
            let mut buf = [0u8; 0x130];
            write_iglyph_metrics(&mut buf, 0, &max);
            let f_fields = [
                max.width, max.height, max.ascender, max.descender,
                max.h_bearing_x, max.h_bearing_y, max.v_bearing_x, max.v_bearing_y,
                max.h_advance, max.v_advance,
            ];
            for (i, v) in f_fields.iter().enumerate() {
                buf[0x28 + i * 4..0x28 + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            buf[0x50..0x54].copy_from_slice(&fm.num_glyphs.to_le_bytes());
            if !font_info.is_null() {
                ctx.write_bytes(font_info.addr(), &buf);
            }
            0
        }
    }
}

/// ScePvfError scePvfGetCharGlyphImage(ScePvfFontId fontID, ScePvfCharCode charCode,
///     ScePvfUserImageBufferRec *imageBuffer)
/// Rasterize the glyph and blit its 8-bit coverage into the caller's destination buffer
/// at the pen position, respecting the buffer's pixel format, stride, and rect clip.
#[hostcall]
pub(super) fn get_char_glyph_image(ctx: &mut GuestCtx, st: &mut VitaState, font: u32, char_code: u32, image_buffer: Ptr) -> i32 {
    glyph_image_impl(ctx, st, font, char_code & 0xFFFF, image_buffer.addr())
}

/// The glyph-blit logic, as a plain function so it can use early returns (the
/// `#[hostcall]` wrapper's body cannot). `image_buffer` is the guest address of the
/// `ScePvfUserImageBufferRec`; a zero address is a null pointer.
fn glyph_image_impl(ctx: &mut GuestCtx, st: &mut VitaState, font: u32, ch: u32, image_buffer: u32) -> i32 {
    if image_buffer == 0 {
        return SCE_PVF_ERROR_ARG;
    }
    // ScePvfUserImageBufferRec (0x18): pixelFormat, xPos64, yPos64, rect(w:u16,h:u16),
    // bytesPerLine:u16, reserved:u16, buffer.
    let pixel_format = ctx.read_u32(image_buffer);
    let x_pos64 = ctx.read_u32(image_buffer + 4) as i32;
    let y_pos64 = ctx.read_u32(image_buffer + 8) as i32;
    let rect_word = ctx.read_u32(image_buffer + 12);
    let rect_w = (rect_word & 0xFFFF) as i32;
    let rect_h = (rect_word >> 16) as i32;
    let bytes_per_line = (ctx.read_u32(image_buffer + 16) & 0xFFFF) as u32;
    let buffer = ctx.read_u32(image_buffer + 20);

    if pixel_format != SCE_PVF_USERIMAGE_DIRECT8 && pixel_format != SCE_PVF_USERIMAGE_DIRECT4_L {
        return SCE_PVF_ERROR_ARG;
    }

    // Copy the cached glyph out so the fonts borrow ends before we touch guest memory.
    let exists = st.fonts.font_exists(font);
    let glyph = st.fonts.glyph(font, ch).map(|(bmp, m)| {
        (bmp.coverage.clone(), bmp.width as i32, bmp.height as i32, m.bitmap_left, m.bitmap_top)
    });
    let (coverage, gw, gh, left, top) = match glyph {
        Some(g) => g,
        None if exists => return SCE_PVF_ERROR_NOGLYPH,
        None => return SCE_PVF_ERROR_ARG,
    };
    if buffer == 0 || gw <= 0 || gh <= 0 {
        return 0; // Nothing to draw (e.g. whitespace) - a valid, empty result.
    }

    // The pen origin (26.6) is the glyph baseline origin in the destination buffer; the
    // bitmap sits at pen + (bitmap_left, -bitmap_top).
    let dst_x0 = (x_pos64 >> 6) + left;
    let dst_y0 = (y_pos64 >> 6) - top;

    for gy in 0..gh {
        let dy = dst_y0 + gy;
        if dy < 0 || dy >= rect_h {
            continue;
        }
        // Clip the glyph row to [0, rect_w).
        let gx0 = (-dst_x0).max(0);
        let gx1 = gw.min(rect_w - dst_x0);
        if gx1 <= gx0 {
            continue;
        }
        let row = &coverage[(gy * gw + gx0) as usize..(gy * gw + gx1) as usize];
        if pixel_format == SCE_PVF_USERIMAGE_DIRECT8 {
            let off = buffer + dy as u32 * bytes_per_line + (dst_x0 + gx0) as u32;
            ctx.write_bytes(off, row);
        } else {
            // DIRECT4_L: 4 bits per pixel, two pixels per byte (high nibble = even col).
            for (i, &cov) in row.iter().enumerate() {
                let dx = (dst_x0 + gx0) + i as i32;
                let byte_off = buffer + dy as u32 * bytes_per_line + (dx as u32 >> 1);
                let existing = ctx.read_u32(byte_off & !3);
                let shift = (byte_off & 3) * 8;
                let old = ((existing >> shift) & 0xFF) as u8;
                let nib = cov >> 4;
                let merged = if dx & 1 == 0 {
                    (old & 0x0F) | (nib << 4)
                } else {
                    (old & 0xF0) | nib
                };
                ctx.write_bytes(byte_off, &[merged]);
            }
        }
    }
    0
}
