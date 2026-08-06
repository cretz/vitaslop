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
//! - `Encode` and `Csc` are NOT implemented, and are deliberately left to the engine's
//!   hard-fail. They produce DATA the title goes on to use, and there is no honest way to
//!   hand back a JPEG that was never encoded: a zeroed output buffer is a corrupt file
//!   that a title would happily write to a save slot, and a silent success would make it
//!   look like a photo feature works. Reaching one is a fact about the call sequence worth
//!   learning, exactly as [`crate::vita::video`] argues for `SceMp4`.

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

/// `SCE_JPEGENC_ERROR_INVALID_POINTER`, per `psp2/jpegenc.h`'s error range.
const SCE_JPEGENC_ERROR_INVALID_POINTER: i32 = 0x8081_0e01u32 as i32;

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
    if context.is_null() {
        SCE_JPEGENC_ERROR_INVALID_POINTER
    } else {
        let a = context.addr();
        ctx.write_u32(a, JPEGENC_MAGIC);
        ctx.write_u32(a + 4, in_width);
        ctx.write_u32(a + 8, in_height);
        ctx.write_u32(a + 12, pixelformat);
        ctx.write_u32(a + 16, out_buffer);
        ctx.write_u32(a + 20, out_size);
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
