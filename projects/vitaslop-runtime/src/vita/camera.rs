//! SceCamera: the two hardware cameras, modelled as NOT PRESENT.
//!
//! This is the same shape as [`crate::vita::net`]'s offline interface, and for the same
//! reason: the honest thing to report is a real console state, not a fabricated success.
//! A Vita has a front and a back camera; a machine running this emulator does not have
//! them, and inventing a frame would put pixels on screen that no camera ever saw. The
//! API's own error table has a value for exactly this - `SCE_CAMERA_ERROR_NOT_MOUNTED` -
//! so every entry point reports it and no title is told a camera opened.
//!
//! A title reaching this is not broken by it: a camera is optional hardware behaviour
//! (an AR mode, a photo feature), and the path a title takes when the camera cannot be
//! opened is a path it ships. What it must NOT be told is that the camera opened and then
//! handed back a black frame, which is indistinguishable on screen from a camera pointed
//! at something dark.
//!
//! Unlike [`super::gesture`], nothing here is guessed: `psp2/camera.h` publishes the
//! prototypes, the device numbers and the error enum, so these are the real signatures.

// The `#[hostcall]` macro emits fully-qualified paths for these, so a module of nothing
// but host calls needs no plain import of them.
use crate::hostcall;

/// `SCE_CAMERA_ERROR_NOT_MOUNTED` (`psp2/camera.h`): the requested camera is not
/// attached. The exact condition being modelled.
const SCE_CAMERA_ERROR_NOT_MOUNTED: i32 = 0x802E_0010u32 as i32;

/// `SCE_CAMERA_ERROR_NOT_OPEN` (`psp2/camera.h`): an operation was attempted on a device
/// that was never opened. Since [`open`] always fails, every later call is in this state -
/// and a title that ignores a failed open (which at least one is known to do for
/// `SceMp4`; see [`crate::vita::video`]) gets a coherent answer rather than a success.
const SCE_CAMERA_ERROR_NOT_OPEN: i32 = 0x802E_0004u32 as i32;

/// Report, once, that a title asked for a camera and was told there is none.
///
/// This is not a failure of the emulator, but it IS a difference from the console that
/// changes what a title does, so it says so rather than being silently absent from the
/// log.
fn note_no_camera() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DONE: AtomicBool = AtomicBool::new(false);
    if !DONE.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            target: "vitaslop::camera",
            "sceCamera: this title opened a camera and there is none - reporting \
             SCE_CAMERA_ERROR_NOT_MOUNTED. Any camera-driven feature (an AR or photo \
             mode) will be unavailable, which is a real console state, not a stub."
        );
    }
}

/// int sceCameraOpen(int devnum, SceCameraInfo *pInfo)
///
/// `pInfo` is left exactly as the caller filled it in: it is an IN parameter describing
/// the format being requested, and on a failed open there is nothing to report back
/// through it. Writing to it would overwrite the caller's own request.
#[hostcall]
pub(super) fn open(_ctx: &mut GuestCtx, _st: &mut VitaState, _devnum: u32, _info: Ptr) -> i32 {
    note_no_camera();
    SCE_CAMERA_ERROR_NOT_MOUNTED
}

/// int sceCameraClose(int devnum)
#[hostcall]
pub(super) fn close(_ctx: &mut GuestCtx, _st: &mut VitaState, _devnum: u32) -> i32 {
    SCE_CAMERA_ERROR_NOT_OPEN
}

/// int sceCameraStart(int devnum)
#[hostcall]
pub(super) fn start(_ctx: &mut GuestCtx, _st: &mut VitaState, _devnum: u32) -> i32 {
    SCE_CAMERA_ERROR_NOT_OPEN
}

/// int sceCameraStop(int devnum)
#[hostcall]
pub(super) fn stop(_ctx: &mut GuestCtx, _st: &mut VitaState, _devnum: u32) -> i32 {
    SCE_CAMERA_ERROR_NOT_OPEN
}

/// int sceCameraRead(int devnum, SceCameraRead *pRead)
///
/// Deliberately does NOT touch `pRead`. On success this would fill in the frame's status,
/// timestamp and buffer pointers; on a device that was never opened there is no frame, and
/// zeroing the struct would hand back a well-formed descriptor for a frame that does not
/// exist - which a title could then read pixels from.
#[hostcall]
pub(super) fn read(_ctx: &mut GuestCtx, _st: &mut VitaState, _devnum: u32, _read: Ptr) -> i32 {
    SCE_CAMERA_ERROR_NOT_OPEN
}

/// int sceCameraGetReverse(int devnum, int *pReverse)
///
/// A getter on an unopened device reports the error and leaves the out-parameter alone,
/// so a caller that ignores the return code does not read a fabricated setting as if the
/// hardware had reported it.
#[hostcall]
pub(super) fn get_reverse(
    _ctx: &mut GuestCtx,
    _st: &mut VitaState,
    _devnum: u32,
    _out: Ptr,
) -> i32 {
    SCE_CAMERA_ERROR_NOT_OPEN
}

/// int sceCameraSetReverse(int devnum, int reverse)
#[hostcall]
pub(super) fn set_reverse(_ctx: &mut GuestCtx, _st: &mut VitaState, _devnum: u32, _v: u32) -> i32 {
    SCE_CAMERA_ERROR_NOT_OPEN
}

/// int sceCameraSetBacklight(int devnum, int backlight)
#[hostcall]
pub(super) fn set_backlight(
    _ctx: &mut GuestCtx,
    _st: &mut VitaState,
    _devnum: u32,
    _v: u32,
) -> i32 {
    SCE_CAMERA_ERROR_NOT_OPEN
}

/// Every other `int sceCameraGet<Setting>(int devnum, int *out)` - anti-flicker, auto-control
/// hold, backlight, brightness, contrast, device location, EV, effect, exposure ceiling, gain,
/// ISO, image quality, night mode, noise reduction, saturation, sharpness (and its off
/// switch), white balance, zoom. The device was never opened, so each reports that and leaves
/// `out` alone - see [`get_reverse`].
#[hostcall]
pub(super) fn get_setting(
    _ctx: &mut GuestCtx,
    _st: &mut VitaState,
    _devnum: u32,
    _out: Ptr,
) -> i32 {
    SCE_CAMERA_ERROR_NOT_OPEN
}

/// Every other `int sceCameraSet<Setting>(int devnum, int value)`, the setters of the
/// settings [`get_setting`] lists: refused on a device that was never opened.
#[hostcall]
pub(super) fn set_setting(_ctx: &mut GuestCtx, _st: &mut VitaState, _devnum: u32, _v: u32) -> i32 {
    SCE_CAMERA_ERROR_NOT_OPEN
}

/// SceBool sceCameraIsActive(int devnum): whether the camera is started. None ever is.
#[hostcall]
pub(super) fn is_active(_ctx: &mut GuestCtx, _st: &mut VitaState, _devnum: u32) -> u32 {
    0
}

/// int sceCameraSetWhiteBalance(int devnum, int wb)
#[hostcall]
pub(super) fn set_white_balance(
    _ctx: &mut GuestCtx,
    _st: &mut VitaState,
    _devnum: u32,
    _v: u32,
) -> i32 {
    SCE_CAMERA_ERROR_NOT_OPEN
}
