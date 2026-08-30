//! `SceAudiodec`: the guest's own audio decoder, which is how a movie gets its sound.
//!
//! The title creates one decoder per movie and then, once per audio frame, hands it an
//! elementary-stream pointer and a PCM destination and expects the PCM to be there when the
//! call returns. Prototypes and struct layouts are documented (`psp2/audiodec.h`); what had
//! to be recovered from the call sites is which of them this title uses and what it does
//! with the results, and that is recorded on each handler.
//!
//! # >>> HOW A SYNCHRONOUS GUEST API IS SERVED BY AN ASYNCHRONOUS DECODER
//!
//! `sceAudiodecDecode` must fill `pPcm` before it returns. In a browser the host decoder
//! answers on a callback that only runs when the worker returns to the event loop, so
//! nothing this call does can make the PCM appear during it.
//!
//! It does not have to. The engine is the one demultiplexing the movie: it knows every
//! audio access unit BEFORE the title asks for it, because it is the thing that handed the
//! unit over ([`crate::vita::video`]). So each unit is submitted to the host decoder at
//! DELIVERY time and the decoded frames queue up; by the time the title's audio thread asks
//! for a frame, it is already here, and the call is an ordinary synchronous one.
//!
//! The queue is FIFO and so is the title's use of it - a demux thread hands units to a
//! decode thread in order - and the ES the guest passes is CHECKED against the unit that
//! produced the frame being served, so a stream that ever went out of order says so rather
//! than playing the wrong audio.

use crate::host::{GuestCtx, Ptr, VitaState};
use crate::hostcall;

/// Field offsets in `SceAudiodecCtrl` (0x28 bytes), from `psp2/audiodec.h`.
mod ctrl {
    pub const HANDLE: u32 = 0x04;
    pub const P_ES: u32 = 0x08;
    /// OUT: elementary-stream bytes actually consumed.
    pub const INPUT_ES_SIZE: u32 = 0x0c;
    pub const MAX_ES_SIZE: u32 = 0x10;
    pub const P_PCM: u32 = 0x14;
    /// OUT: PCM bytes actually produced.
    pub const OUTPUT_PCM_SIZE: u32 = 0x18;
    pub const MAX_PCM_SIZE: u32 = 0x1c;
    pub const WORD_LENGTH: u32 = 0x20;
    pub const P_INFO: u32 = 0x24;
}

/// Field offsets in `SceAudiodecInfoAac` (0x14 bytes), from `psp2/audiodec.h`.
mod info_aac {
    pub const IS_ADTS: u32 = 0x04;
    pub const CH: u32 = 0x08;
    pub const SAMPLING_RATE: u32 = 0x0c;
    pub const IS_SBR: u32 = 0x10;
}

/// Field offsets in `SceAudiodecInfoAt9` (0x1c bytes), from `psp2/audiodec.h`. The
/// first four are what the TITLE fills in before creating a decoder; the rest are what
/// the library derives from `configData` and writes BACK, which is why they are named
/// here and not just skipped.
mod info_at9 {
    /// IN: the 4-byte ATRAC9 config word - the whole of the stream's geometry.
    pub const CONFIG_DATA: u32 = 0x04;
    /// OUT: everything below is derived from the config word.
    pub const CH: u32 = 0x08;
    pub const BIT_RATE: u32 = 0x0c;
    pub const SAMPLING_RATE: u32 = 0x10;
    pub const SUPER_FRAME_SIZE: u32 = 0x14;
    pub const FRAMES_IN_SUPER_FRAME: u32 = 0x18;
}

/// `SCE_AUDIODEC_TYPE_AT9`.
const TYPE_AT9: u32 = 0x1003;
/// `SCE_AUDIODEC_TYPE_AAC`.
const TYPE_AAC: u32 = 0x1005;

/// `SCE_AUDIODEC_AT9_MAX_ES_SIZE` / `_MAX_SAMPLES`, from `psp2/audiodec.h`. The library
/// publishes these as the ceiling a caller sizes its buffers by, and `sceAudiodecDecode`
/// decodes ONE frame - `sceAudiodecDecodeNFrames` is the call that does more - so 256
/// samples per channel is one call's output.
const AT9_MAX_ES_SIZE: u32 = 1024;
const AT9_MAX_SAMPLES: u32 = 256;

/// `SCE_AUDIODEC_ERROR_INVALID_TYPE`, for a codec this engine does not decode.
const ERROR_INVALID_TYPE: i32 = 0x807F_0001u32 as i32;
/// `SCE_AUDIODEC_ERROR_INVALID_PTR`.
const ERROR_INVALID_PTR: i32 = 0x807F_0008u32 as i32;
/// `SCE_AUDIODEC_ERROR_INVALID_HANDLE`.
const ERROR_INVALID_HANDLE: i32 = 0x807F_0009u32 as i32;
/// `SCE_AUDIODEC_ERROR_INVALID_INIT_PARAM`.
const ERROR_INVALID_INIT_PARAM: i32 = 0x807F_0002u32 as i32;
/// `SCE_AUDIODEC_ERROR_ALREADY_INITIALIZED`.
const ERROR_ALREADY_INITIALIZED: i32 = 0x807F_0003u32 as i32;
/// `SCE_AUDIODEC_ERROR_NOT_INITIALIZED`.
const ERROR_NOT_INITIALIZED: i32 = 0x807F_0005u32 as i32;
/// `SCE_AUDIODEC_AT9_ERROR_INVALID_CONFIG` - the 4-byte config word does not describe a
/// stream. Our decoder is the one that says so, by refusing to be built from it.
const ERROR_AT9_INVALID_CONFIG: i32 = 0x807F_2000u32 as i32;

/// The decoder context size handed back to a caller that allocates its own.
///
/// The value is this engine's to choose - the context is OURS, and nothing in it lives in
/// guest memory - but it is not free to choose badly: MEASURED at the one call site, the
/// title rounds `size + 0x20ff` UP to a megabyte and allocates that from its own pool, so a
/// large answer is a large allocation in a heap this engine does not own. One page is
/// enough to be a plausible decoder context and small enough to round to the same single
/// megabyte any real answer would.
const CONTEXT_SIZE: u32 = 0x1000;

/// Per-run `SceAudiodec` state.
#[derive(Default)]
pub struct AudiodecState {
    /// Live decoders, by the handle handed back in `SceAudiodecCtrl::handle`.
    pub sessions: Vec<AudiodecSession>,
    pub next_handle: u32,
    /// Codec types `sceAudiodecInitLibrary` has brought up and `TermLibrary` has not
    /// torn down. The library really is per-codec on hardware - each type is initialised
    /// separately, a second init of the same type is an error, and creating a decoder
    /// before its type is up fails - so it is tracked rather than assumed.
    pub initialised: Vec<u32>,
    /// One-shot reports.
    reported_open: bool,
    reported_starved: bool,
    reported_out_of_order: bool,
}

/// One decoder the title created.
pub struct AudiodecSession {
    pub handle: u32,
    /// The `SceAudiodecType` this decoder was created for. The two codecs are served
    /// completely differently - see [`do_decode`] - so the session has to remember which
    /// it is rather than inferring it from which fields happen to be set.
    pub codec: u32,
    pub channels: u32,
    pub sample_rate: u32,
    /// The guest context address, handed back at delete.
    pub context: u32,
    /// Frames decoded and handed to the guest, for the one line a run says about sound.
    pub delivered: u64,
    /// Frames this session REFUSED. Counted beside `delivered` because one number alone
    /// cannot say whether a codec is broken or a single frame is: a decoder that fails once
    /// in a thousand frames and one that fails every frame produce the same first error line.
    pub refused: u64,
    /// The ATRAC9 decoder for an AT9 session, built from the config word the title put
    /// in `SceAudiodecInfoAt9`. `None` for AAC, whose frames are decoded by the host
    /// ahead of the call (see the module header).
    ///
    /// It is STATEFUL - MDCT overlap and delta-coding history carry frame to frame - so
    /// it lives for the life of the decoder and is reset only by
    /// [`audiodec_clear_context`], which is exactly what that call is for.
    pub at9: Option<vitaslop_atrac9::Atrac9Decoder>,
    /// The config word, kept so `sceAudiodecClearContext` can rebuild the decoder from
    /// the same stream description rather than needing the guest struct again.
    pub at9_config: [u8; 4],
}

impl AudiodecState {
    fn session(&mut self, handle: u32) -> Option<&mut AudiodecSession> {
        self.sessions.iter_mut().find(|s| s.handle == handle)
    }
}

/// SceUInt32 sceAudiodecGetContextSize(SceAudiodecCtrl *pCtrl, SceUInt32 codecType)
///
/// **The SIZE is the return value, and zero is the failure.** MEASURED at the call site:
/// the result is tested with `bne` and only a non-zero value continues; a zero takes the
/// title's "could not size the decoder" path and the movie plays silently.
#[hostcall]
pub(super) fn audiodec_get_context_size(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    _p_ctrl: Ptr,
    codec_type: u32,
) -> i32 {
    do_get_context_size(st, codec_type)
}

pub(crate) fn do_get_context_size(st: &mut VitaState, codec_type: u32) -> i32 {
    if codec_type != TYPE_AAC {
        report_unsupported_codec(st, codec_type);
        return 0;
    }
    CONTEXT_SIZE as i32
}

/// SceInt32 sceAudiodecCreateDecoderExternal(SceAudiodecCtrl *pCtrl, SceUInt32 codecType,
///                                           SceUIntVAddr vaContext, SceUInt32 contextSize)
///
/// The title allocates the decoder's working memory itself and hands it over; this engine
/// does not use it (the decoder is the host's) but it is remembered so the delete call can
/// hand the same address back, which is what the caller frees.
#[hostcall]
pub(super) fn audiodec_create_decoder_external(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    p_ctrl: Ptr,
    codec_type: u32,
    va_context: u32,
    _context_size: u32,
) -> i32 {
    do_create_decoder_external(ctx, st, p_ctrl, codec_type, va_context)
}

pub(crate) fn do_create_decoder_external(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    p_ctrl: Ptr,
    codec_type: u32,
    va_context: u32,
) -> i32 {
    if p_ctrl.addr() == 0 {
        return ERROR_INVALID_PTR;
    }
    if codec_type != TYPE_AAC {
        report_unsupported_codec(st, codec_type);
        return ERROR_INVALID_TYPE;
    }
    let ctrl_addr = p_ctrl.addr();
    let info = ctx.read_u32(ctrl_addr + ctrl::P_INFO);
    // The stream, as the TITLE describes it. `SceAudiodecInfoAac` carries no
    // `AudioSpecificConfig`, so these three fields are all there is to configure a host
    // decoder from - see `vitaslop_platform::audio_dec::synth_asc`.
    let (channels, sample_rate, is_adts, is_sbr) = if info != 0 {
        (
            ctx.read_u32(info + info_aac::CH),
            ctx.read_u32(info + info_aac::SAMPLING_RATE),
            ctx.read_u32(info + info_aac::IS_ADTS),
            ctx.read_u32(info + info_aac::IS_SBR),
        )
    } else {
        (2, 48_000, 0, 0)
    };
    let handle = {
        st.audiodec.next_handle += 1;
        st.audiodec.next_handle
    };
    st.audiodec.sessions.push(AudiodecSession {
        handle,
        codec: TYPE_AAC,
        channels: channels.max(1),
        sample_rate: if sample_rate == 0 { 48_000 } else { sample_rate },
        context: va_context,
        delivered: 0,
        refused: 0,
        at9: None,
        at9_config: [0; 4],
    });
    ctx.write_u32(ctrl_addr + ctrl::HANDLE, handle);
    ctx.write_u32(ctrl_addr + ctrl::WORD_LENGTH, 16);
    if !st.audiodec.reported_open {
        st.audiodec.reported_open = true;
        // WARN and unconditional, like the video path's first-picture line: it is the one
        // line that says a movie's SOUND was set up at all, and a device's default log
        // level is `warn`.
        tracing::warn!(
            target: "vitaslop::movie",
            channels, sample_rate, is_adts, is_sbr,
            "sceAudiodecCreateDecoderExternal: the title opened an AAC decoder for a movie"
        );
    }
    0
}

// ---------------------------------------------------------------------------------
// The AT9 path. Unlike the movie's AAC - which the ENGINE demultiplexes, so it can decode
// ahead of the guest's call - an AT9 stream is the title's own: it hands over the
// elementary stream itself, one frame at a time, and expects the PCM back before the call
// returns. That is a straight synchronous decode, and `vitaslop-atrac9` is the same
// decoder the NGS voice path already uses ([`crate::vita::at9`]), so nothing here is an
// approximation: the samples are the stream's samples.
// ---------------------------------------------------------------------------------

/// SceInt32 sceAudiodecInitLibrary(SceUInt32 codecType, SceAudiodecInitParam *pInitParam)
///
/// Brings a codec's library up. The parameter is a union whose every arm is
/// `{ SceUInt32 size; SceUInt32 totalCh-or-totalStreams; }` - for AT9 it is the total
/// CHANNELS the title will decode across all its decoders, which is a budget the real
/// library allocates against. Nothing here allocates per-channel, so the value is
/// recorded in the log line and not enforced: refusing a request this engine can serve
/// would be inventing a limit.
///
/// A `size` of 0 is refused, because that field is how the library tells the union's
/// arms apart and a caller that left it unset has not filled the struct in.
#[hostcall]
pub(super) fn audiodec_init_library(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    codec_type: u32,
    p_init_param: Ptr,
) -> i32 {
    if p_init_param.addr() == 0 {
        ERROR_INVALID_PTR
    } else if ctx.read_u32(p_init_param.addr()) == 0 {
        ERROR_INVALID_INIT_PARAM
    } else if codec_type != TYPE_AT9 && codec_type != TYPE_AAC {
        report_unsupported_codec(st, codec_type);
        ERROR_INVALID_TYPE
    } else if st.audiodec.initialised.contains(&codec_type) {
        ERROR_ALREADY_INITIALIZED
    } else {
        let (size, total) =
            (ctx.read_u32(p_init_param.addr()), ctx.read_u32(p_init_param.addr() + 4));
        st.audiodec.initialised.push(codec_type);
        tracing::info!(
            target: "vitaslop::audio",
            codec_type = format_args!("{codec_type:#x}"), size, total,
            "sceAudiodecInitLibrary: the title brought up a decoder library"
        );
        0
    }
}

/// SceInt32 sceAudiodecTermLibrary(SceUInt32 codecType)
///
/// Tearing down a library that was never brought up is the library's own
/// `NOT_INITIALIZED` error, not a no-op: a title that terms twice has a bug and hardware
/// tells it so.
#[hostcall]
pub(super) fn audiodec_term_library(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    codec_type: u32,
) -> i32 {
    if st.audiodec.initialised.contains(&codec_type) {
        st.audiodec.initialised.retain(|&c| c != codec_type);
        0
    } else {
        ERROR_NOT_INITIALIZED
    }
}

/// SceInt32 sceAudiodecCreateDecoder(SceAudiodecCtrl *pCtrl, SceUInt32 codecType)
///
/// The library allocates the decoder's context itself here (the External form is where
/// the title supplies it), so there is no context address to remember.
///
/// **This call is where a `SceAudiodecInfoAt9` is COMPLETED.** The title fills in `size`
/// and the 4-byte `configData` and leaves the rest; the library derives channels, sample
/// rate, superframe size and frames-per-superframe from that word and writes them back,
/// and the title reads them to size its own buffers. Our decoder parses the same word,
/// so every value written here is the config word's own, not a default.
#[hostcall]
pub(super) fn audiodec_create_decoder(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    p_ctrl: Ptr,
    codec_type: u32,
) -> i32 {
    if p_ctrl.addr() == 0 {
        ERROR_INVALID_PTR
    } else if codec_type != TYPE_AT9 {
        // AAC arrives through the External form (that is what the movie path uses), and
        // any other codec is one this engine does not decode.
        report_unsupported_codec(st, codec_type);
        ERROR_INVALID_TYPE
    } else if !st.audiodec.initialised.contains(&codec_type) {
        ERROR_NOT_INITIALIZED
    } else {
        create_at9_decoder(ctx, st, p_ctrl.addr())
    }
}

/// The body of [`audiodec_create_decoder`] once the arguments are known good.
fn create_at9_decoder(ctx: &mut GuestCtx, st: &mut VitaState, ctrl_addr: u32) -> i32 {
    let info = ctx.read_u32(ctrl_addr + ctrl::P_INFO);
    if info == 0 {
        return ERROR_INVALID_PTR;
    }
    let cfg = ctx.read_bytes(info + info_at9::CONFIG_DATA, 4);
    let config = [cfg[0], cfg[1], cfg[2], cfg[3]];
    let decoder = match vitaslop_atrac9::Atrac9Decoder::new(config) {
        Ok(d) => d,
        Err(_) => {
            // The config word did not parse. That is precisely what the library's own
            // AT9-specific config error means, and it is a real refusal, not a fallback.
            tracing::error!(
                target: "vitaslop::audio",
                config = format_args!("{config:02x?}"),
                "sceAudiodecCreateDecoder: the AT9 config word does not describe a stream"
            );
            return ERROR_AT9_INVALID_CONFIG;
        }
    };
    let geom = decoder.info();
    // The bit rate is the one derived field our decoder does not carry, and it is
    // arithmetic on the ones it does: a superframe's bits over a superframe's seconds.
    let per_superframe = geom.frame_samples * geom.frames_in_superframe;
    let bit_rate = if per_superframe > 0 {
        (geom.superframe_size as i64 * 8 * geom.sample_rate as i64 / per_superframe as i64) as u32
    } else {
        0
    };
    ctx.write_u32(info + info_at9::CH, geom.channels as u32);
    ctx.write_u32(info + info_at9::BIT_RATE, bit_rate);
    ctx.write_u32(info + info_at9::SAMPLING_RATE, geom.sample_rate as u32);
    ctx.write_u32(info + info_at9::SUPER_FRAME_SIZE, geom.superframe_size as u32);
    ctx.write_u32(info + info_at9::FRAMES_IN_SUPER_FRAME, geom.frames_in_superframe as u32);

    let channels = geom.channels.max(1) as u32;
    st.audiodec.next_handle += 1;
    let handle = st.audiodec.next_handle;
    st.audiodec.sessions.push(AudiodecSession {
        handle,
        codec: TYPE_AT9,
        channels,
        sample_rate: geom.sample_rate as u32,
        context: 0,
        delivered: 0,
        refused: 0,
        at9: Some(decoder),
        at9_config: config,
    });
    ctx.write_u32(ctrl_addr + ctrl::HANDLE, handle);
    ctx.write_u32(ctrl_addr + ctrl::WORD_LENGTH, 16);
    ctx.write_u32(ctrl_addr + ctrl::MAX_ES_SIZE, AT9_MAX_ES_SIZE);
    ctx.write_u32(ctrl_addr + ctrl::MAX_PCM_SIZE, AT9_MAX_SAMPLES * 2 * channels);
    tracing::info!(
        target: "vitaslop::audio",
        handle,
        channels = geom.channels,
        sample_rate = geom.sample_rate,
        superframe = geom.superframe_size,
        frames_in_superframe = geom.frames_in_superframe,
        bit_rate,
        "sceAudiodecCreateDecoder: opened an ATRAC9 decoder"
    );
    0
}

/// SceInt32 sceAudiodecDeleteDecoder(SceAudiodecCtrl *pCtrl)
#[hostcall]
pub(super) fn audiodec_delete_decoder(ctx: &mut GuestCtx, st: &mut VitaState, p_ctrl: Ptr) -> i32 {
    if p_ctrl.addr() == 0 {
        ERROR_INVALID_PTR
    } else {
        let handle = ctx.read_u32(p_ctrl.addr() + ctrl::HANDLE);
        let before = st.audiodec.sessions.len();
        st.audiodec.sessions.retain(|s| s.handle != handle);
        if st.audiodec.sessions.len() == before {
            ERROR_INVALID_HANDLE
        } else {
            // The handle field is cleared for the same reason the out-params elsewhere
            // are written: a title that reuses the ctrl struct must not find a
            // live-looking handle in it. Our handles start at 1, so 0 is "none".
            ctx.write_u32(p_ctrl.addr() + ctrl::HANDLE, 0);
            0
        }
    }
}

/// SceInt32 sceAudiodecClearContext(SceAudiodecCtrl *pCtrl)
///
/// Drops the decoder's carried-over state. An ATRAC9 frame is not independent - the MDCT
/// overlap and the delta-coded scale factors come from the previous frame - so after a
/// SEEK the title must say so, or the first frame at the new position is reconstructed
/// against the wrong history and clicks. Rebuilding the decoder from the same config word
/// is exactly that reset, and it is the whole of what this call means.
#[hostcall]
pub(super) fn audiodec_clear_context(ctx: &mut GuestCtx, st: &mut VitaState, p_ctrl: Ptr) -> i32 {
    if p_ctrl.addr() == 0 {
        ERROR_INVALID_PTR
    } else {
        clear_at9_context(ctx, st, p_ctrl.addr())
    }
}

/// The body of [`audiodec_clear_context`] once the ctrl pointer is known good.
fn clear_at9_context(ctx: &mut GuestCtx, st: &mut VitaState, ctrl_addr: u32) -> i32 {
    let handle = ctx.read_u32(ctrl_addr + ctrl::HANDLE);
    match st.audiodec.session(handle) {
        None => ERROR_INVALID_HANDLE,
        // An AAC session's frames are decoded by the host ahead of the call, so there is
        // no per-call history here to clear; the queue is the video path's to manage.
        Some(s) if s.at9.is_none() => 0,
        Some(s) => match vitaslop_atrac9::Atrac9Decoder::new(s.at9_config) {
            Ok(fresh) => {
                s.at9 = Some(fresh);
                0
            }
            // The same config word built a decoder at create time, so this cannot fail -
            // but if it ever did, reporting it beats silently keeping stale state.
            Err(_) => ERROR_AT9_INVALID_CONFIG,
        },
    }
}

/// SceInt32 sceAudiodecGetInternalError(SceAudiodecCtrl *pCtrl, SceInt32 *pInternalError)
///
/// The codec-internal detail behind a failed `sceAudiodecDecode`. Every decode that
/// reaches the guest here either succeeded or returned its own error, and this engine
/// keeps no second, finer error behind that - so the honest report is 0, "no internal
/// error", written through the out-param rather than left for the caller to read off its
/// own stack.
#[hostcall]
pub(super) fn audiodec_get_internal_error(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    p_ctrl: Ptr,
    p_internal_error: Ptr,
) -> i32 {
    if p_ctrl.addr() == 0 {
        ERROR_INVALID_PTR
    } else if st.audiodec.session(ctx.read_u32(p_ctrl.addr() + ctrl::HANDLE)).is_none() {
        ERROR_INVALID_HANDLE
    } else {
        if p_internal_error.addr() != 0 {
            ctx.write_u32(p_internal_error.addr(), 0);
        }
        0
    }
}

/// One AT9 frame, decoded straight into the guest's PCM buffer. Called from
/// [`do_decode`] when the session is an AT9 one.
fn decode_at9(ctx: &mut GuestCtx, st: &mut VitaState, ctrl_addr: u32, handle: u32) -> i32 {
    let es = ctx.read_u32(ctrl_addr + ctrl::P_ES);
    let pcm_dest = ctx.read_u32(ctrl_addr + ctrl::P_PCM);
    let max_es = ctx.read_u32(ctrl_addr + ctrl::MAX_ES_SIZE).max(1);
    let max_pcm = ctx.read_u32(ctrl_addr + ctrl::MAX_PCM_SIZE);
    if es == 0 || pcm_dest == 0 {
        return ERROR_INVALID_PTR;
    }
    // Read the whole window the caller declared. A frame is self-delimiting inside it -
    // `decode_frame` reports how many bytes it actually consumed, which is what the
    // title advances its own read cursor by.
    let input = ctx.read_bytes(es, max_es as usize);
    let Some(session) = st.audiodec.session(handle) else { return ERROR_INVALID_HANDLE };
    let (channels, sample_rate) = (session.channels, session.sample_rate);
    let Some(decoder) = session.at9.as_mut() else { return ERROR_INVALID_HANDLE };
    let mut pcm = vec![0i16; decoder.frame_samples() * decoder.channels()];
    let used = match decoder.decode_frame(&input, &mut pcm) {
        Ok(used) => used,
        Err(e) => {
            // A frame that does not decode is a real failure and is reported as one. It
            // is NOT answered with silence: the title's own error path is the right place
            // for a broken stream to be handled, and a silent frame would hide it.
            // >>> WHICH frame, and what it was handed. A bitstream error is almost always a
            // frame BOUNDARY error - the wrong bytes, or the right bytes at the wrong offset
            // - so the ordinal (is this the first frame of the stream? the one after a
            // reset?), the window the caller declared and the first bytes of it are what
            // separate "our decoder is wrong" from "our caller is feeding it wrong". Without
            // them the line says only that something failed, which cost a re-run.
            let ordinal = st.audiodec.session(handle).map_or(0, |s| s.delivered + s.refused);
            let head: Vec<String> =
                input.iter().take(8).map(|b| format!("{b:02x}")).collect();
            tracing::error!(
                target: "vitaslop::audio",
                handle, error = %e, ordinal, max_es, es = format_args!("{es:#x}"),
                head = %head.join(" "),
                "sceAudiodecDecode: an ATRAC9 frame did not decode"
            );
            if let Some(session) = st.audiodec.session(handle) {
                session.refused += 1;
            }
            return ERROR_AT9_INVALID_CONFIG;
        }
    };
    let mut bytes: Vec<u8> = Vec::with_capacity(pcm.len() * 2);
    for s in &pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    if max_pcm != 0 && bytes.len() > max_pcm as usize {
        bytes.truncate(max_pcm as usize);
    }
    ctx.write_bytes(pcm_dest, &bytes);
    ctx.write_u32(ctrl_addr + ctrl::OUTPUT_PCM_SIZE, bytes.len() as u32);
    ctx.write_u32(ctrl_addr + ctrl::INPUT_ES_SIZE, used as u32);
    // The cursor question, answerable only from a sequence of these: does the title
    // advance `pEs` by the `used` we report, and does a short frame (used < the config's
    // fixed frame size) precede the once-per-stream failure at a superframe boundary?
    tracing::trace!(
        target: "vitaslop::audio",
        handle, es = format_args!("{es:#x}"), used,
        ordinal = st.audiodec.session(handle).map_or(0, |s| s.delivered + s.refused),
        "sceAudiodecDecode: AT9 frame ok"
    );
    if let Some(session) = st.audiodec.session(handle) {
        session.delivered += 1;
        if session.delivered == 1 {
            tracing::info!(
                target: "vitaslop::audio",
                handle, channels, sample_rate, samples = pcm.len(),
                "the FIRST ATRAC9 frame from sceAudiodecDecode reached guest memory"
            );
        }
    }
    0
}

/// SceInt32 sceAudiodecDecode(SceAudiodecCtrl *pCtrl)
///
/// Fill `pPcm` with one frame of PCM. MEASURED at the call site: the caller sets `pEs` from
/// the demuxed unit record and `pPcm` from its own output struct immediately before the
/// call, and treats any NEGATIVE result as a failure it logs and stops on.
#[hostcall]
pub(super) fn audiodec_decode(ctx: &mut GuestCtx, st: &mut VitaState, p_ctrl: Ptr) -> i32 {
    do_decode(ctx, st, p_ctrl)
}

pub(crate) fn do_decode(ctx: &mut GuestCtx, st: &mut VitaState, p_ctrl: Ptr) -> i32 {
    let ctrl_addr = p_ctrl.addr();
    if ctrl_addr == 0 {
        return ERROR_INVALID_PTR;
    }
    let handle = ctx.read_u32(ctrl_addr + ctrl::HANDLE);
    let Some((codec, channels, sample_rate)) =
        st.audiodec.session(handle).map(|s| (s.codec, s.channels, s.sample_rate))
    else {
        return ERROR_INVALID_HANDLE;
    };
    // AT9 is the guest's OWN stream and decodes synchronously right here; everything
    // below this line is the movie's AAC, served from the queue the demuxer filled.
    if codec == TYPE_AT9 {
        return decode_at9(ctx, st, ctrl_addr, handle);
    }
    let es = ctx.read_u32(ctrl_addr + ctrl::P_ES);
    let pcm_dest = ctx.read_u32(ctrl_addr + ctrl::P_PCM);
    let max_pcm = ctx.read_u32(ctrl_addr + ctrl::MAX_PCM_SIZE);

    // The frame the demuxer decoded ahead, and the access unit it came from.
    let Some(frame) = crate::vita::video::take_decoded_audio(st) else {
        // Nothing decoded ahead. On a host with no decoder that is every frame, and on a
        // browser it can be the first frame or two of a movie while the decoder fills.
        // Reported once, and answered with a frame of silence rather than with a failure:
        // the title stops playing the movie's sound entirely on a negative result.
        report_starved(st);
        let bytes = max_pcm.min(1024 * 2 * channels.max(1)) as usize;
        if pcm_dest != 0 && bytes > 0 {
            ctx.write_bytes(pcm_dest, &vec![0u8; bytes]);
        }
        ctx.write_u32(ctrl_addr + ctrl::OUTPUT_PCM_SIZE, bytes as u32);
        ctx.write_u32(ctrl_addr + ctrl::INPUT_ES_SIZE, ctx.read_u32(ctrl_addr + ctrl::MAX_ES_SIZE));
        return 0;
    };

    // >>> THE ONE CHECK THAT KEEPS THE QUEUE HONEST. The frames are served in the order the
    // units were handed over, which is only right if the title consumes them in that order
    // too. It does - a demux thread feeding a decode thread - and if it ever does not, this
    // says so instead of playing the wrong audio silently.
    if es != 0 && frame.es_head != 0 {
        let head = ctx.read_u32(es);
        if head != frame.es_head {
            report_out_of_order(st, head, frame.es_head);
        }
    }

    let mut bytes: Vec<u8> = Vec::with_capacity(frame.samples.len() * 2);
    for s in &frame.samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    if max_pcm != 0 && bytes.len() > max_pcm as usize {
        bytes.truncate(max_pcm as usize);
    }
    if pcm_dest != 0 {
        ctx.write_bytes(pcm_dest, &bytes);
    }
    ctx.write_u32(ctrl_addr + ctrl::OUTPUT_PCM_SIZE, bytes.len() as u32);
    ctx.write_u32(ctrl_addr + ctrl::INPUT_ES_SIZE, frame.es_size);
    if let Some(session) = st.audiodec.session(handle) {
        session.delivered += 1;
        if session.delivered == 1 {
            tracing::warn!(
                target: "vitaslop::movie",
                channels, sample_rate,
                samples = frame.samples.len(),
                "the movie's FIRST AUDIO FRAME reached guest memory"
            );
        }
    }
    0
}

/// SceInt32 sceAudiodecDeleteDecoderExternal(SceAudiodecCtrl *pCtrl, SceUIntVAddr *pvaContext)
///
/// Hands the context address back through the out-parameter, which is what the caller then
/// frees. A delete of a handle this engine does not know is not an error worth failing on -
/// the title is tearing down either way.
#[hostcall]
pub(super) fn audiodec_delete_decoder_external(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    p_ctrl: Ptr,
    pva_context: Ptr,
) -> i32 {
    do_delete_decoder_external(ctx, st, p_ctrl, pva_context)
}

pub(crate) fn do_delete_decoder_external(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    p_ctrl: Ptr,
    pva_context: Ptr,
) -> i32 {
    let ctrl_addr = p_ctrl.addr();
    if ctrl_addr == 0 {
        return ERROR_INVALID_PTR;
    }
    let handle = ctx.read_u32(ctrl_addr + ctrl::HANDLE);
    let context = st
        .audiodec
        .sessions
        .iter()
        .find(|s| s.handle == handle)
        .map(|s| s.context)
        .unwrap_or(0);
    st.audiodec.sessions.retain(|s| s.handle != handle);
    if pva_context.addr() != 0 {
        ctx.write_u32(pva_context.addr(), context);
    }
    0
}

/// Say, once, that a codec this engine does not decode was asked for.
fn report_unsupported_codec(st: &mut VitaState, codec_type: u32) {
    if st.audiodec.reported_open {
        return;
    }
    st.audiodec.reported_open = true;
    tracing::error!(
        target: "vitaslop::movie",
        codec_type = format_args!("{codec_type:#x}"),
        "sceAudiodec: the title asked for a codec this engine does not decode. Its sound \
         will be silent; the picture is unaffected."
    );
}

/// Say, once, that the guest asked for a frame the decoder had not produced.
fn report_starved(st: &mut VitaState) {
    if st.audiodec.reported_starved {
        return;
    }
    st.audiodec.reported_starved = true;
    tracing::warn!(
        target: "vitaslop::movie",
        "sceAudiodecDecode: no decoded frame was ready, so this one is SILENCE. On a host \
         with no AAC decoder that is every frame of the movie; otherwise it is the decoder \
         filling, and it should not repeat once playback has settled."
    );
}

/// Say, once, that the title consumed audio units in a different order than they were
/// handed over - which would make the queue serve the wrong frame.
fn report_out_of_order(st: &mut VitaState, guest_head: u32, ours: u32) {
    if st.audiodec.reported_out_of_order {
        return;
    }
    st.audiodec.reported_out_of_order = true;
    tracing::error!(
        target: "vitaslop::movie",
        guest = format_args!("{guest_head:#010x}"), ours = format_args!("{ours:#010x}"),
        "sceAudiodecDecode: the elementary stream the title passed is not the access unit \
         the queued frame was decoded from. The audio served from here is out of step with \
         the picture."
    );
}
