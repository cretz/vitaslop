//! SceJpeg: the hardware MJPEG decoder's lifecycle.
//!
//! Companion to [`super::jpegenc`], and the same line is drawn in the same place and for
//! the same reason. `psp2/jpeg.h` publishes the prototypes, so the interface is not
//! guessed.
//!
//! The lifecycle calls are real: initialising a decoder pool and tearing it down is
//! bookkeeping, and there is nothing about it that a decoder would do differently. The
//! DECODE entry points are deliberately NOT here - they produce an image the title goes on
//! to display, and handing back an untouched or zeroed buffer would put a frame on screen
//! that no decoder produced. The engine's hard-fail on reaching one is the right outcome
//! and names it exactly.

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
