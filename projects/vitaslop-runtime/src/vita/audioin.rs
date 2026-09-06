//! SceAudioIn: the console's microphone, modelled as a MUTED one.
//!
//! # Why muted rather than absent, and why that is a real state
//! A Vita has a built-in microphone, so "there is no such device" is not a state this
//! API can report - it has no not-mounted error, unlike [`crate::vita::camera`]. What it
//! does have is `SCE_AUDIO_IN_GETSTATUS_MUTE`: a console whose input is muted is an
//! ordinary, first-class state, every title ships a path for it, and it is exactly what
//! this host is. Nothing here opens a host capture device - the same boundary
//! [`crate::vita::net`] draws, and for the same reason: a run has to be reproducible
//! from its own inputs, and a room's noise is not one of them.
//!
//! So a port really opens, `sceAudioInInput` really delivers a grain, and the grain is
//! DIGITAL SILENCE. That is not a stub standing in for capture - it is what a muted
//! input records, and [`report_silent`] says so out loud the first time, because a title
//! that is listening for the player deserves to have that visible in the log rather than
//! inferred from a voice command that never registers.
//!
//! # The pacing is the part that is easy to get wrong
//! On hardware `sceAudioInInput` BLOCKS until a grain has been captured, so a capture
//! loop built on it runs at the input's own rate whatever else it does. Returning
//! instantly would let that loop free-run - the same defect the pad's
//! `sceCtrlReadBufferPositive` had, where a non-blocking read turned a title's update
//! loop into a spin. So this parks the caller for the grain's own duration, exactly as
//! [`crate::vita::audio::out_output`] does on the output side.

use crate::host::{GuestCtx, VitaState};
use crate::SvcOutcome;
use crate::hostcall;

/// `SceAudioInErrorCode` values this surface uses, from `psp2/audioin.h`.
const SCE_AUDIO_IN_ERROR_INVALID_PORT: i32 = 0x8026_0101u32 as i32;
const SCE_AUDIO_IN_ERROR_INVALID_SIZE: i32 = 0x8026_0102u32 as i32;
const SCE_AUDIO_IN_ERROR_INVALID_SAMPLE_FREQ: i32 = 0x8026_0103u32 as i32;
const SCE_AUDIO_IN_ERROR_INVALID_POINTER: i32 = 0x8026_0105u32 as i32;

/// The sample rates `SceAudioIn` accepts. A title that asks for anything else has a bug
/// hardware would tell it about, so this refuses rather than quietly resampling nothing.
const VALID_FREQS: [i32; 3] = [16_000, 48_000, 8_000];

/// Say, once, that every captured grain is silence. Unconditional, not behind a knob:
/// an emulator that reports success while capturing nothing has to announce it, or "the
/// voice feature does not work" has no visible cause.
fn report_silent(grain: i32, freq: i32) {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        tracing::warn!(
            target: "vitaslop::audio",
            grain, freq,
            "sceAudioInOpenPort: the title opened the MICROPHONE. This host captures nothing, \
             so every grain it reads is digital silence - the console's own muted-input state. \
             Anything the title drives from voice will never trigger."
        );
    });
}

/// int sceAudioInOpenPort(SceAudioInPortType portType, int grain, int freq,
///                        SceAudioInParam param)
///
/// The parameters are really validated, because they are the one thing here that can be
/// wrong independently of the missing capture: a bad grain or rate is a caller bug and
/// hardware says so. `portType` is not checked - VOICE (0) and RAW (2) both exist, and
/// they differ in the processing applied to a signal there is none of.
#[hostcall]
pub(super) fn in_open_port(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    ty: i32,
    grain: i32,
    freq: i32,
    _param: i32,
) -> i32 {
    if grain <= 0 {
        SCE_AUDIO_IN_ERROR_INVALID_SIZE
    } else if !VALID_FREQS.contains(&freq) {
        SCE_AUDIO_IN_ERROR_INVALID_SAMPLE_FREQ
    } else {
        report_silent(grain, freq);
        st.audio_state.in_open(ty, grain as u32, freq as u32)
    }
}

/// int sceAudioInReleasePort(int port)
#[hostcall]
pub(super) fn in_release_port(_ctx: &mut GuestCtx, st: &mut VitaState, port: i32) -> i32 {
    if st.audio_state.in_close(port) {
        0
    } else {
        SCE_AUDIO_IN_ERROR_INVALID_PORT
    }
}

/// int sceAudioInInput(int port, void *destPtr)
///
/// Fill one grain and block for its duration. The buffer is S16 MONO - the one input
/// format the API defines (`SCE_AUDIO_IN_PARAM_FORMAT_S16_MONO`) - so a grain is
/// `grain * 2` bytes.
///
/// The buffer is WRITTEN rather than left alone. A caller that ignores the return code
/// would otherwise process whatever the buffer already held as if it were freshly
/// captured audio, which is worse than silence: on a reused buffer it is the previous
/// grain, played back forever.
pub(super) fn in_input(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let (port, dest) = (ctx.arg(0) as i32, ctx.arg(1));
    let Some((grain, freq)) = st.audio_state.in_format(port) else {
        ctx.ret(SCE_AUDIO_IN_ERROR_INVALID_PORT as u32);
        return SvcOutcome::Continue;
    };
    if dest == 0 {
        ctx.ret(SCE_AUDIO_IN_ERROR_INVALID_POINTER as u32);
        return SvcOutcome::Continue;
    }
    ctx.write_bytes(dest, &vec![0u8; grain as usize * 2]);
    ctx.ret(0);
    // One grain is captured in `grain / freq` seconds; park for exactly that, so a
    // capture loop runs at the microphone's rate instead of spinning.
    let paced_us = if freq > 0 { (u64::from(grain) * 1_000_000) / u64::from(freq) } else { 0 };
    if paced_us > 0 && st.is_preemptive() {
        st.sleep_park(paced_us);
        SvcOutcome::Block
    } else {
        SvcOutcome::Continue
    }
}

/// int sceAudioInGetAdopt(SceAudioInPortType portType)
/// Whether a port of `portType` is currently open, read off the same table the open and
/// release calls maintain - the input-side twin of `sceAudioOutGetAdopt`.
#[hostcall]
pub(super) fn in_get_adopt(_ctx: &mut GuestCtx, st: &mut VitaState, ty: i32) -> i32 {
    i32::from(st.audio_state.in_adopted(ty))
}

/// int sceAudioInGetStatus(int select)
///
/// `select` is `SCE_AUDIO_IN_GETSTATUS_MUTE` (1). This host captures nothing, and MUTED
/// is precisely how the console describes that - so reporting it is not a workaround,
/// it is the one status query that already has a word for our state, and it agrees with
/// the silence [`in_input`] delivers.
#[hostcall]
pub(super) fn in_get_status(_ctx: &mut GuestCtx, _st: &mut VitaState, _select: i32) -> i32 {
    1
}
