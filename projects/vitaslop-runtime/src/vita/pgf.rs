//! ScePgf: the PSP-compatible font API the Vita also ships.
//!
//! # The same engine as ScePvf, a different set of structs
//! A title picks one of the two font libraries and never mixes them. They wrap the same
//! job - open a font, ask a glyph's metrics, rasterize it into a buffer the title
//! uploads as a texture - so both marshal onto the one [`crate::font::FontLibrary`] in
//! `VitaState`, and everything about hinting, scaling and caching is shared. What is NOT
//! shared is the struct layouts and the error codes, and those are the whole of this
//! file: `SceFontCharInfo` is 0x3c bytes where `ScePvfCharInfo` is 0x40, `SceFontInfo`
//! carries a 0xa8-byte style block ScePvf does not, and the pixel-format enum has values
//! ScePvf has no name for. Both are read straight from `psp2/pgf.h`.
//!
//! # Metrics are 26.6 fixed point, and the float copies are not optional
//! `SceFontInfo` carries every maximum metric TWICE - once as `...I` in 26.6 and once as
//! `...F` in float - and a title is free to read either. Writing only one and leaving the
//! other zero is the kind of half-filled struct that produces text laid out at zero
//! width with no error anywhere, so both are written from the same values.

use crate::host::{GuestCtx, Ptr, VitaState};
use crate::hostcall;

/// `SceFontErrorCode`, from `psp2/pgf.h`. The facility is shared with ScePvf, so the
/// two libraries' codes for the same condition are literally the same number.
const SCE_FONT_ERROR_INVALID_LIBID: i32 = 0x8046_0002u32 as i32;
const SCE_FONT_ERROR_INVALID_PARAMETER: i32 = 0x8046_0003u32 as i32;
const SCE_FONT_ERROR_HANDLER_OPEN_FAILED: i32 = 0x8046_0005u32 as i32;
const SCE_FONT_ERROR_INVALID_FONT_DATA: i32 = 0x8046_000Au32 as i32;

/// `SceFontPixelFormatCode`.
const SCE_FONT_PIXELFORMAT_4: u32 = 0;
const SCE_FONT_PIXELFORMAT_4_REV: u32 = 1;
const SCE_FONT_PIXELFORMAT_8: u32 = 2;

/// The character the library substitutes for one the face has no glyph for. `?` is the
/// library's own default (a title may change it with `sceFontSetAltCharacterCode`), and
/// substituting is what hardware does - returning an error for one missing glyph would
/// abort a whole string a console renders fine.
const DEFAULT_ALT_CHAR: u32 = '?' as u32;

/// Encode a fractional-pixel value as 26.6 fixed point.
fn fx(v: f32) -> i32 {
    (v * 64.0).round() as i32
}

/// Say, once, that a character was substituted. Not silent: a font that is missing the
/// glyphs a title needs shows up on screen as the wrong letters, and the log is the only
/// place that can name which ones.
fn report_substituted(ch: u32) {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        tracing::warn!(
            target: "vitaslop::cb",
            char_code = format_args!("{ch:#06x}"),
            "sceFont: the opened face has no glyph for this character, so the alt character \
             ('?') was substituted - which is what the library does, but it means the text on \
             screen is not the text the title asked for"
        );
    });
}

/// Resolve `ch` against `font`, falling back to the alt character. Returns the character
/// actually used, or `None` when the font handle is not live.
fn resolve_char(st: &mut VitaState, font: u32, ch: u32) -> Option<u32> {
    if !st.fonts.font_exists(font) {
        return None;
    }
    if st.fonts.has_glyph(font, ch) {
        Some(ch)
    } else {
        report_substituted(ch);
        Some(DEFAULT_ALT_CHAR)
    }
}

/// SceFontLibHandle sceFontNewLib(SceFontNewLibParams *params, unsigned int *errorCode)
///
/// `params` carries the title's own allocator and file-IO callbacks. They are NOT called
/// here, and that is a real difference from hardware rather than an omission: the glyph
/// cache and the parsed face live on the host side of the boundary, in host memory, so
/// there is nothing for a guest allocator to allocate. `numFonts` is likewise the size of
/// a system-font list this host does not have (see [`open`]).
#[hostcall]
pub(super) fn new_lib(ctx: &mut GuestCtx, st: &mut VitaState, params: Ptr, error_code: Ptr) -> u32 {
    let num_fonts = if params.is_null() { 0 } else { ctx.read_u32(params.addr() + 4) };
    let lib = st.fonts.new_lib();
    tracing::debug!(target: "vitaslop::cb", lib, num_fonts, "sceFontNewLib");
    if !error_code.is_null() {
        ctx.write_u32(error_code.addr(), 0);
    }
    lib
}

/// int sceFontDoneLib(SceFontLibHandle libHandle)
#[hostcall]
pub(super) fn done_lib(_ctx: &mut GuestCtx, st: &mut VitaState, lib: u32) -> i32 {
    if st.fonts.done_lib(lib) {
        0
    } else {
        SCE_FONT_ERROR_INVALID_LIBID
    }
}

/// SceFontHandle sceFontOpen(SceFontLibHandle libHandle, int index, int mode,
///                           unsigned int *errorCode)
///
/// Opens one of the SYSTEM fonts by index. This engine ships none - they are Sony's
/// copyrighted assets - so there is nothing to open, and reporting the open failure is
/// the honest answer. A title that renders its own text ships its own font and reaches
/// [`open_user_memory`] instead, which works.
#[hostcall]
pub(super) fn open(ctx: &mut GuestCtx, st: &mut VitaState, lib: u32, index: i32, _mode: i32, error_code: Ptr) -> u32 {
    let font = crate::font::system::bytes()
        .and_then(|bytes| {
            st.fonts.open_system_substitute(lib, &bytes, crate::font::system::SUBSTITUTE_PX)
        })
        .inspect(|_| report_substitute_font(index));
    match font {
        Some(font) => {
            if !error_code.is_null() {
                ctx.write_u32(error_code.addr(), 0);
            }
            font
        }
        None => {
            report_no_system_font(index);
            if !error_code.is_null() {
                ctx.write_u32(error_code.addr(), SCE_FONT_ERROR_HANDLER_OPEN_FAILED as u32);
            }
            0
        }
    }
}

/// Say - once per (rows lost, which edge) - that a glyph did not fit the buffer the title gave
/// it and was truncated. See the call site for why silence here is the problem.
fn report_glyph_truncated(lost: i32, gh: i32, top_edge: bool) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(i32, bool)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((lost, top_edge)) {
        return;
    }
    tracing::warn!(
        target: "vitaslop::cb",
        "sceFontGetCharGlyphImage: a glyph did not fit the buffer the title supplied and {lost} \
         of its {gh} rows were DROPPED off the {} edge. The substitute face is rasterized at THIS \
         host's metrics, not the console font's, so a glyph can be taller than the room the \
         title left for it - and what reaches the screen is a sliver that reads as a font-size \
         or a geometry bug somewhere else entirely. Set VITASLOP_SYSTEM_FONT to a face whose \
         metrics suit the title, or read the pen positions at `vitaslop::cb=debug` to see what \
         room it is actually reserving.",
        if top_edge { "TOP" } else { "BOTTOM" }
    );
}

/// Say, once, that a system-font open was answered with a SUBSTITUTE.
///
/// Not silent, and not phrased as success: the glyph shapes and the metrics are another face's,
/// so a string can wrap or centre differently from the console. That is a visible difference and
/// the log is the only place it can be stated ([[vitaslop-fallback-must-report]]).
pub(super) fn report_substitute_font(index: i32) {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        tracing::warn!(
            target: "vitaslop::cb",
            font_index = index,
            source = crate::font::system::describe().unwrap_or("?"),
            "sceFontOpen: the title asked for a SYSTEM font. This host ships none - they are the \
             console vendor's assets - so a SUBSTITUTE face is opened in its place. The text now \
             renders, but its LETTERFORMS AND METRICS ARE NOT THE CONSOLE'S: expect different \
             glyph shapes and slightly different wrapping or centring. Set VITASLOP_SYSTEM_FONT \
             to choose the face."
        );
    });
}

/// Say, once, that a SYSTEM font was asked for and refused.
///
/// This used to be silent, and silence here is the worst possible answer: a title that opens
/// the system font renders every string through it, so the failure shows up hundreds of frames
/// later as **blank or black areas where dynamic text belongs** - a club list, a course name, a
/// menu label - with nothing anywhere connecting the two. It cost a session to trace one such
/// black rectangle back to this call. A refusal is still the right RESULT (the fonts are Sony's
/// assets and are not shipped); what was wrong was not saying so ([[vitaslop-fallback-must-report]]).
fn report_no_system_font(index: i32) {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        tracing::warn!(
            target: "vitaslop::cb",
            font_index = index,
            "sceFontOpen: the title asked for a SYSTEM font, and this host ships none - they are \
             the console vendor's assets. The open is refused, exactly as it would be on a device \
             with no font installed, so every string the title renders through this library comes \
             out EMPTY: expect blank or black areas where dynamic text belongs. A title that ships \
             its own font reaches sceFontOpenUserMemory instead and is unaffected."
        );
    });
}

/// SceFontHandle sceFontOpenUserMemory(SceFontLibHandle libHandle, void *pMemoryFont,
///                                     SceSize pMemoryFontSize, unsigned int *errorCode)
///
/// Open a font from bytes the title already holds in guest memory - which is how a title
/// that unpacked a font from its own archive gets it into the library.
#[hostcall]
pub(super) fn open_user_memory(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    lib: u32,
    addr: Ptr,
    size: u32,
    error_code: Ptr,
) -> u32 {
    // Rejected before the read, not after: a bogus `size` would otherwise copy that much
    // guest memory only to fail to parse it.
    let (font, err) = if addr.is_null() || size == 0 {
        (0, SCE_FONT_ERROR_INVALID_PARAMETER)
    } else {
        let bytes = ctx.read_bytes(addr.addr(), size as usize);
        match st.fonts.open_user_memory(lib, &bytes) {
            Some(f) => (f, 0),
            None if !st.fonts.lib_exists(lib) => (0, SCE_FONT_ERROR_INVALID_LIBID),
            // The lib is live, so it is the bytes that are not a font we can parse.
            None => (0, SCE_FONT_ERROR_INVALID_FONT_DATA),
        }
    };
    tracing::debug!(target: "vitaslop::cb", lib, size, font, err, "sceFontOpenUserMemory");
    if !error_code.is_null() {
        ctx.write_u32(error_code.addr(), err as u32);
    }
    font
}

/// int sceFontClose(SceFontHandle fontHandle)
///
/// A handle that was never issued, or was already closed, is REFUSED rather than quietly
/// succeeding: a double close that reads as success hides a title's own use-after-free.
#[hostcall]
pub(super) fn close(_ctx: &mut GuestCtx, st: &mut VitaState, font: u32) -> i32 {
    if st.fonts.close(font) {
        0
    } else {
        SCE_FONT_ERROR_INVALID_PARAMETER
    }
}

/// int sceFontGetCharInfo(SceFontHandle fontHandle, unsigned int charCode,
///                        SceFontCharInfo *charInfo)
///
/// `SceFontCharInfo` is 0x3c bytes: four whole-pixel bitmap fields, then ten 26.6 metric
/// fields, then two `short` shadow fields. The shadow pair stays zero - a shadow is a
/// second glyph the PGF format can carry and nothing here synthesizes one, so claiming an
/// id for it would point the title at a glyph that does not exist.
#[hostcall]
pub(super) fn get_char_info(ctx: &mut GuestCtx, st: &mut VitaState, font: u32, char_code: u32, char_info: Ptr) -> i32 {
    char_info_impl(ctx, st, font, char_code & 0xFFFF, char_info.addr())
}

/// The body of [`get_char_info`], as a plain function so it can use early returns.
fn char_info_impl(ctx: &mut GuestCtx, st: &mut VitaState, font: u32, char_code: u32, char_info: u32) -> i32 {
    let Some(ch) = resolve_char(st, font, char_code) else {
        return SCE_FONT_ERROR_INVALID_PARAMETER;
    };
    let Some((bmp, m)) = st.fonts.glyph(font, ch) else {
        return SCE_FONT_ERROR_INVALID_PARAMETER;
    };
    let mut buf = [0u8; 0x3C];
    let words: [i32; 14] = [
        bmp.width as i32,
        bmp.height as i32,
        m.bitmap_left,
        m.bitmap_top,
        fx(m.width),
        fx(m.height),
        fx(m.ascender),
        fx(m.descender),
        fx(m.h_bearing_x),
        fx(m.h_bearing_y),
        fx(m.v_bearing_x),
        fx(m.v_bearing_y),
        fx(m.h_advance),
        fx(m.v_advance),
    ];
    for (i, w) in words.iter().enumerate() {
        buf[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    if char_info != 0 {
        ctx.write_bytes(char_info, &buf);
    }
    0
}

/// int sceFontGetFontInfo(SceFontHandle fontHandle, SceFontInfo *fontInfo)
///
/// 0x108 bytes: ten 26.6 maxima, the same ten as floats, two `short` bitmap maxima, the
/// two charmap lengths, the 0xa8-byte `SceFontStyle`, and the bits-per-pixel.
///
/// The style block is left zeroed. It describes a font CHOSEN FROM THE SYSTEM LIST
/// (family, weight, language, file name), and a title that opened its own bytes did not
/// choose from that list - there is nothing true to put there. `BPP` is 8, which is what
/// this engine's coverage bitmaps are and what [`glyph_image`] blits.
#[hostcall]
pub(super) fn get_font_info(ctx: &mut GuestCtx, st: &mut VitaState, font: u32, font_info: Ptr) -> i32 {
    font_info_impl(ctx, st, font, font_info.addr())
}

/// The body of [`get_font_info`], as a plain function so it can use early returns.
fn font_info_impl(ctx: &mut GuestCtx, st: &mut VitaState, font: u32, font_info: u32) -> i32 {
    let Some(fm) = st.fonts.face_metrics(font) else {
        return SCE_FONT_ERROR_INVALID_PARAMETER;
    };
    // The "maximum glyph" the face can produce, from the face-wide metrics.
    let (w, h) = (fm.max_advance, fm.ascender - fm.descender);
    let maxima: [f32; 10] = [
        w,           // width
        h,           // height
        fm.ascender, // ascender
        fm.descender,
        0.0,         // left x
        fm.ascender, // base y
        w / 2.0,     // centre x
        fm.ascender, // top y
        w,           // advance x
        fm.height,   // advance y
    ];
    let mut buf = [0u8; 0x108];
    for (i, v) in maxima.iter().enumerate() {
        buf[i * 4..i * 4 + 4].copy_from_slice(&fx(*v).to_le_bytes());
        buf[0x28 + i * 4..0x28 + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    buf[0x50..0x52].copy_from_slice(&(w.ceil() as i16).to_le_bytes());
    buf[0x52..0x54].copy_from_slice(&(h.ceil() as i16).to_le_bytes());
    buf[0x54..0x58].copy_from_slice(&fm.num_glyphs.to_le_bytes());
    // shadowMapLength stays 0: this engine synthesizes no shadow glyphs.
    buf[0x104] = 8; // BPP
    if font_info != 0 {
        ctx.write_bytes(font_info, &buf);
    }
    0
}

/// int sceFontGetCharGlyphImage(SceFontHandle fontHandle, unsigned int charCode,
///                              SceFontGlyphImage *glyphImage)
///
/// Rasterize the glyph and blit its 8-bit coverage into the caller's buffer at the pen
/// position, in the buffer's own pixel format and stride.
#[hostcall]
pub(super) fn get_char_glyph_image(ctx: &mut GuestCtx, st: &mut VitaState, font: u32, char_code: u32, image: Ptr) -> i32 {
    glyph_image(ctx, st, font, char_code & 0xFFFF, image.addr())
}

/// The blit, as a plain function so it can use early returns (a `#[hostcall]` body is
/// inlined into a wrapper that returns `()`, so `return` in one is a type error).
fn glyph_image(ctx: &mut GuestCtx, st: &mut VitaState, font: u32, char_code: u32, image: u32) -> i32 {
    if image == 0 {
        return SCE_FONT_ERROR_INVALID_PARAMETER;
    }
    // SceFontGlyphImage (0x18): pixelFormat, xPos64, yPos64, bufWidth:u16, bufHeight:u16,
    // bytesPerLine:u16, pad:u16, bufferPtr.
    let pixel_format = ctx.read_u32(image);
    let x_pos64 = ctx.read_u32(image + 4) as i32;
    let y_pos64 = ctx.read_u32(image + 8) as i32;
    let dims = ctx.read_u32(image + 12);
    let (buf_w, buf_h) = ((dims & 0xFFFF) as i32, (dims >> 16) as i32);
    let bytes_per_line = ctx.read_u32(image + 16) & 0xFFFF;
    let buffer = ctx.read_u32(image + 20);

    // 24- and 32-bit destinations are colour formats; this engine rasterizes COVERAGE and
    // has no colour to put in them, so they are refused rather than filled with a guess.
    if !matches!(pixel_format, SCE_FONT_PIXELFORMAT_4 | SCE_FONT_PIXELFORMAT_4_REV | SCE_FONT_PIXELFORMAT_8)
    {
        return SCE_FONT_ERROR_INVALID_PARAMETER;
    }
    let Some(ch) = resolve_char(st, font, char_code) else {
        return SCE_FONT_ERROR_INVALID_PARAMETER;
    };
    // Copy the glyph out so the `fonts` borrow ends before guest memory is touched.
    let Some((coverage, gw, gh, left, top)) = st.fonts.glyph(font, ch).map(|(b, m)| {
        (b.coverage.clone(), b.width as i32, b.height as i32, m.bitmap_left, m.bitmap_top)
    }) else {
        return SCE_FONT_ERROR_INVALID_PARAMETER;
    };
    if buffer == 0 || gw <= 0 || gh <= 0 {
        // Nothing to draw - whitespace has valid metrics and an empty bitmap, which is a
        // successful result, not a failure.
        return 0;
    }
    // >>> THE PEN IS THE BITMAP'S TOP-LEFT, NOT ITS BASELINE. MEASURED, not read from a
    // header: PCSA00009 rasterises its glyph cache through this call and then draws quads
    // sampling exactly `[pen_x, pen_x+width) x [pen_y, pen_y+height)` - 'S' at pen (2,2),
    // glyph 8x11, sampled u 2..10 v 2..13; 't' at pen (742,112), glyph 6x10, sampled
    // u 742..748 v 112.. - with NO bearing offsets, on a 20-px cell grid. The title renders
    // correctly on the console, so the console places the bitmap at the pen. The FreeType
    // baseline convention this used to apply (`pen + (left, -top)`) pushed a glyph placed at
    // `pen_y = 2` up to rows -9..2, clipped it to a 2-row sliver, and left the title's quad
    // sampling an empty box - which is what a phone reported as menu text rendering as a
    // 2-pixel smudge.
    let _ = (left, top);
    let dst_x0 = x_pos64 >> 6;
    let dst_y0 = y_pos64 >> 6;
    // The PEN POSITIONS the title chooses are the only statement it makes about the SIZE it
    // expects: this library has no set-char-size call, so a PGF font's size is intrinsic and the
    // substitute has to be rasterized at something. The step between successive pens is that
    // something, measured rather than assumed.
    tracing::debug!(
        target: "vitaslop::cb",
        ch = format_args!("{ch:#06x}"),
        pen = format_args!("({}, {})", x_pos64 >> 6, y_pos64 >> 6),
        glyph = format_args!("{gw}x{gh}+{left}+{top}"),
        buf = format_args!("{buf_w}x{buf_h} pitch {bytes_per_line}"),
        "sceFontGetCharGlyphImage"
    );
    // >>> A GLYPH THAT DOES NOT FIT THE DESTINATION IS TRUNCATED IN SILENCE, AND THAT SILENCE
    // >>> LOOKS EXACTLY LIKE A RENDERING BUG FURTHER DOWNSTREAM.
    //
    // The row loop below drops any row outside the buffer, which is correct - the title owns
    // that buffer and we must not write past it. What is not correct is saying nothing: the
    // stand-in is rasterized at OUR metrics, not the console font's, so a glyph whose ascent
    // exceeds the headroom the title left above its baseline loses its top and nothing
    // anywhere connects the two. MEASURED on PCSA00009's glyph atlas: one glyph at
    // `pen_y = 2` with `top = 11` keeps 2 of its 11 rows, and the 2-pixel band that leaves in
    // a 1024x512 atlas was read as a font-size defect for a whole session.
    //
    // Reported once per (rows lost, direction) so a title that clips one glyph says it once
    // and a title that clips every glyph still says it once - the number is what matters, not
    // the repetition. It is a WARN because it is lost ink, not a style choice.
    if dst_y0 < 0 || dst_y0 + gh > buf_h {
        let lost = (-dst_y0).max(0) + (dst_y0 + gh - buf_h).max(0);
        report_glyph_truncated(lost.min(gh), gh, dst_y0 < 0);
    }
    for gy in 0..gh {
        let dy = dst_y0 + gy;
        if dy < 0 || dy >= buf_h {
            continue;
        }
        let gx0 = (-dst_x0).max(0);
        let gx1 = gw.min(buf_w - dst_x0);
        if gx1 <= gx0 {
            continue;
        }
        let row = &coverage[(gy * gw + gx0) as usize..(gy * gw + gx1) as usize];
        if pixel_format == SCE_FONT_PIXELFORMAT_8 {
            ctx.write_bytes(buffer + dy as u32 * bytes_per_line + (dst_x0 + gx0) as u32, row);
            continue;
        }
        // The two 4-bit formats pack two pixels per byte and differ ONLY in which nibble
        // the even column lands in - `_4` puts it in the high nibble, `_4_REV` in the low.
        // Getting that backwards is not subtly wrong, it interleaves every glyph with its
        // neighbour.
        let even_high = pixel_format == SCE_FONT_PIXELFORMAT_4;
        for (i, &cov) in row.iter().enumerate() {
            let dx = (dst_x0 + gx0) + i as i32;
            let byte_off = buffer + dy as u32 * bytes_per_line + (dx as u32 >> 1);
            let existing = ctx.read_u32(byte_off & !3);
            let shift = (byte_off & 3) * 8;
            let old = ((existing >> shift) & 0xFF) as u8;
            let nib = cov >> 4;
            let high = (dx & 1 == 0) == even_high;
            let merged = if high { (old & 0x0F) | (nib << 4) } else { (old & 0xF0) | nib };
            ctx.write_bytes(byte_off, &[merged]);
        }
    }
    0
}
