//! ScePvf: the PVF (Vita font) library. A title opens the system font library,
//! configures the em-square, resolution, and skew for its text rendering, and
//! opens one or more fonts. There is no glyph rasterizer here (text rendering is
//! the renderer's job over the captured draw stream); the surface is satisfied so a
//! title's font subsystem initializes and opens fonts without stalling or
//! dereferencing a null library/font id.
//!
//! Handle model: `ScePvfLibId` and `ScePvfFontId` are opaque `void *` to the guest.
//! We hand back fresh, non-null opaque handles so the title can hold and pass them;
//! the per-font configuration setters accept and succeed.

use crate::hostcall;

/// ScePvfLibId scePvfNewLib(ScePvfInitRec *initParam, ScePvfError *errorCode)
/// Create a font library instance. Reports success through `errorCode` and returns
/// a fresh non-null library id.
#[hostcall]
pub(super) fn new_lib(ctx: &mut GuestCtx, st: &mut VitaState, _init_param: Ptr, error_code: Ptr) -> u32 {
    if !error_code.is_null() {
        ctx.write_u32(error_code.addr(), 0);
    }
    st.new_handle()
}

/// ScePvfFontId scePvfOpen(ScePvfLibId libID, ScePvfFontIndex fontIndex,
///     ScePvfU32 mode, ScePvfError *errorCode)
/// Open a font from the library. Reports success through `errorCode` and returns a
/// fresh non-null font id.
#[hostcall]
pub(super) fn open(ctx: &mut GuestCtx, st: &mut VitaState, _lib: u32, _font_index: i32, _mode: u32, error_code: Ptr) -> u32 {
    if !error_code.is_null() {
        ctx.write_u32(error_code.addr(), 0);
    }
    st.new_handle()
}

/// ScePvfError scePvfSetEM(ScePvfLibId libID, ScePvfFloat32 emValue)
#[hostcall]
pub(super) fn set_em(_st: &mut VitaState, _lib: u32, _em_value: f32) -> i32 {
    0
}

/// ScePvfError scePvfSetResolution(ScePvfLibId libID, ScePvfFloat32 hResolution,
///     ScePvfFloat32 vResolution)
#[hostcall]
pub(super) fn set_resolution(_st: &mut VitaState, _lib: u32, _h_resolution: f32, _v_resolution: f32) -> i32 {
    0
}

/// ScePvfError scePvfSetSkewValue(ScePvfFontId fontID, ScePvfFloat32 angleX,
///     ScePvfFloat32 angleY)
#[hostcall]
pub(super) fn set_skew_value(_st: &mut VitaState, _font: u32, _angle_x: f32, _angle_y: f32) -> i32 {
    0
}
