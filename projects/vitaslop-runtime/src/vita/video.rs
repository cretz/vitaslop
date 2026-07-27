//! SceMp4 / video playback.
//!
//! A title's full-motion video is an MP4 container demuxed by `SceMp4`, an H.264 video
//! stream decoded by `SceVideodecUser`/`sceAvcdec*`, and an audio stream decoded by
//! `SceAudiodecUser`. None of that decoding exists in this engine yet, and none of it can
//! be faked: there is no frame to hand back.
//!
//! So `sceMp4OpenFile` reports that the stream cannot be opened. That is a real path a
//! title has - a movie file can be missing or corrupt on hardware too - and it lets a
//! title skip its movie and carry on to the part that is a game. It is an
//! APPROXIMATION, not a faithful result, so it reports itself unconditionally the first
//! time it happens rather than quietly changing what the title shows.
//!
//! MEASURED, and it is the thing to know before doing more here: at least one title
//! IGNORES the failed `sceMp4OpenFile` and calls `sceMp4StartFileStreaming` on the
//! handle it never received. So the assumption that "no handle" makes the rest of the
//! library unreachable is wrong, and `sceMp4StartFileStreaming` needs its own honest
//! failure for that title to give up on the movie and carry on.
//!
//! Every other `SceMp4` entry point is still deliberately left unimplemented, and the
//! engine's hard-fail on an unimplemented NID is exactly the right way to find out which
//! one a title reaches next - each one that appears is a fact about the real call
//! sequence, which is the only way this undocumented library gets mapped.

// The `#[hostcall]` macro rewrites these signatures and emits its own fully-qualified
// paths, so a module of nothing but host calls has no use for a plain `use` of them -
// hence the qualified types below rather than an import that reads as unused.
use crate::hostcall;

/// `SCE_ERROR_ERRNO_ENOENT`. The API's own error table is not documented anywhere
/// clean-room, so this reports the failure a missing movie file would produce - the
/// closest thing to "this stream is not available" that is a known-good Sce error value.
const SCE_ERROR_ERRNO_ENOENT: i32 = 0x8001_0002u32 as i32;

/// int sceMp4OpenFile(...)
/// Report that the movie cannot be opened, so the title skips it. See the module docs:
/// this is an approximation and says so, every run, the first time it is reached.
#[hostcall]
pub(super) fn mp4_open_file(
    _ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
    _a0: crate::host::Ptr,
    _a1: crate::host::Ptr,
    _a2: crate::host::Ptr,
    _a3: crate::host::Ptr,
) -> i32 {
    report_no_video(st);
    SCE_ERROR_ERRNO_ENOENT
}

/// Say once per run that this title wanted to play a movie and did not get one.
/// Unconditional, not behind a debug flag: the picture the run produces is missing
/// whatever the movie would have shown, and nothing else in the output says so.
fn report_no_video(st: &mut crate::host::VitaState) {
    if st.reported_no_video {
        return;
    }
    st.reported_no_video = true;
    eprintln!(
        "SceMp4: video playback is NOT IMPLEMENTED (the MP4 demuxer exists but there is no \
         H.264 decoder behind it) - reporting the movie as unavailable so the title skips \
         it. Anything the movie would have shown is missing from this run."
    );
}

/// int sceMp4StartFileStreaming(...)
/// Report that streaming cannot start. Reached because a title ignored the failed
/// `sceMp4OpenFile` above and called this on a handle it was never given - so returning
/// success here would hand it a session with no stream behind it, and it would then ask
/// for units that do not exist. An error is a state its own error path can act on.
#[hostcall]
pub(super) fn mp4_start_file_streaming(
    _ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
    _a0: crate::host::Ptr,
    _a1: crate::host::Ptr,
    _a2: crate::host::Ptr,
    _a3: crate::host::Ptr,
) -> i32 {
    report_no_video(st);
    SCE_ERROR_ERRNO_ENOENT
}

/// int sceMp4CloseFile(...)
/// Nothing was ever opened, so there is nothing to release and no state to get wrong.
/// Succeeds: a close of a session that does not exist is the one call here that CAN
/// honestly report success, and failing it would only send the title into a teardown
/// error path over a resource this engine never created.
#[hostcall]
pub(super) fn mp4_close_file(
    _ctx: &mut crate::host::GuestCtx,
    _st: &mut crate::host::VitaState,
    _a0: crate::host::Ptr,
) -> i32 {
    0
}

/// int <unnamed SceMp4 0x7b4832fe>(handle, unit)
/// The buffer release the movie teardown makes after `sceMp4CloseFile` - see
/// [`crate::nid::services::MP4_RELEASE_BUFFER_7B4832FE`] for how that role was recovered.
/// Nothing was ever streamed, so there is no buffer held and nothing to give back;
/// succeeds for the same reason the close does. Deliberately does NOT write through the
/// unit pointer: this is a release, and inventing unit fields is exactly the hollow
/// success this module refuses to produce.
#[hostcall]
pub(super) fn mp4_release_buffer(
    _ctx: &mut crate::host::GuestCtx,
    _st: &mut crate::host::VitaState,
    _handle: crate::host::Ptr,
    _unit: crate::host::Ptr,
) -> i32 {
    0
}
