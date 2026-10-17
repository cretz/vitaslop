//! SceAvPlayer: the system movie player.
//!
//! Unlike `SceMp4` (which only demuxes, and leaves decoding to the title through
//! `SceAvcdec`/`SceAudiodec` - see [`crate::vita::video`]), AvPlayer is a whole player: the
//! title names a file and is handed decoded pictures and PCM, already paced to the movie's
//! own clock. So everything is here - the container ([`crate::mp4`]), the host's H.264 and
//! AAC decoders, the clock, and the title's callbacks.
//!
//! # The shape of a playback
//! `sceAvPlayerInit` copies the title's init block (allocators, optional file functions, an
//! optional event callback) and returns a handle. `sceAvPlayerAddSource(path)` opens and
//! parses the movie and delivers `READY` through the event callback - a title typically
//! enables its streams and calls `sceAvPlayerStart` from inside that callback, so the call is
//! re-entrant by design. From `Start` the title polls `sceAvPlayerGetVideoData` once per
//! frame and `sceAvPlayerGetAudioData` from an audio thread, while `sceAvPlayerIsActive` says
//! the movie is still running; it turns false at the natural end, and the title closes.
//!
//! # The data it hands out
//! - **Video**: two-plane 8-bit 4:2:0 - a luma plane of `width * height` bytes immediately
//!   followed by one plane of interleaved chroma pairs, pitch equal to the width (a title
//!   samples the frame as a linear GXM texture of exactly the reported size, so there is no
//!   padding). Each pair is **Cr then Cb**: MEASURED, the title that plays its cutscenes
//!   through this library samples the frame as `YVU420P2` (swizzle `0x3000`), and a title
//!   picks the format its library delivers - written Cb first, every face in the movie came
//!   out with red and blue exchanged (dark skin rendered blue). The buffers come from the title's own `allocateTexture`, because the title
//!   maps that memory for the GPU and samples the frame where it lies.
//! - **Audio**: signed 16-bit interleaved PCM, exactly 1024 frames per call (one AAC frame) -
//!   a title passes the buffer straight to an audio port opened with a 1024-sample grain, so
//!   a short last frame is padded with silence rather than handed out short.
//!
//! # The clock
//! Guest time since `Start`, with paused stretches removed. A picture is handed out once its
//! presentation time is due on that clock; when several are due the NEWEST is handed out
//! and the older ones are dropped, so a caller that falls behind catches up rather than
//! drifting. Audio is not gated: the title paces it by its blocking audio port.
//!
//! # Re-entrancy
//! Event delivery and the allocators are calls INTO the title, made with
//! [`VitaState::call_guest`]: a handler that makes N of them runs N+1 times. Every state change
//! here is therefore idempotent (guarded by what has already happened) or made after the
//! handler's last guest call.

use std::collections::VecDeque;

use crate::host::{GuestCall, GuestCtx, VitaState};
use crate::hostcall;

/// `SCE_AVPLAYER_ERROR_INVALID_PARAM`: a bad handle, pointer or argument.
const ERROR_INVALID_PARAM: i32 = 0x806A_0001u32 as i32;
/// The PS-family "operation failed" code: a call made in a state it cannot run in.
const ERROR_OPERATION_FAILED: i32 = 0x806A_0002u32 as i32;
/// The PS-family "not supported" code: a source that cannot be played.
const ERROR_NOT_SUPPORTED: i32 = 0x806A_0004u32 as i32;

/// Event ids delivered through `SceAvPlayerEventReplacement::eventCallback`.
mod event {
    pub const STATE_STOP: u32 = 0x01;
    pub const STATE_READY: u32 = 0x02;
    pub const STATE_PLAY: u32 = 0x03;
    pub const STATE_PAUSE: u32 = 0x04;
}

/// `SceAvPlayerStreamType`.
mod stream_type {
    pub const VIDEO: u32 = 0;
    pub const AUDIO: u32 = 1;
}

/// PCM frames per `sceAvPlayerGetAudioData`.
const AUDIO_FRAME: usize = 1024;
/// Output picture buffers when the title asks for none.
const DEFAULT_VIDEO_BUFFERS: u32 = 4;
/// Pictures decoded ahead of the clock, and audio access units fed ahead of the PCM queue.
const VIDEO_AHEAD: usize = 4;
/// Access units at most inside the video decoder, unanswered - see `pump_video`. H.264 holds
/// back at most 16 pictures (the DPB), so this never starves a stream's own reordering.
const VIDEO_IN_FLIGHT: usize = 16 + VIDEO_AHEAD;
const AUDIO_AHEAD: usize = 8;

/// The title's callbacks, from `SceAvPlayerInitData`.
#[derive(Clone, Copy, Debug, Default)]
struct Callbacks {
    mem_object: u32,
    allocate: u32,
    deallocate: u32,
    allocate_texture: u32,
    deallocate_texture: u32,
    file_open: u32,
    event_object: u32,
    event: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// Initialised; no source yet.
    Idle,
    /// A source is parsed and READY has been (or is being) delivered.
    Ready,
    Playing,
    Paused,
    /// Stopped by the title, or reached the end, or failed.
    Stopped,
}

/// One movie stream as the title numbers them: container track order.
struct Stream {
    track: usize,
    kind: u32,
    enabled: bool,
}

/// What a source holds once it is open.
struct Source {
    path: String,
    fd: i32,
    mp4: crate::mp4::Mp4,
    streams: Vec<Stream>,
    video: Option<VideoPath>,
    audio: Option<AudioPath>,
}

struct VideoPath {
    track: usize,
    decoder: Box<dyn vitaslop_platform::video::VideoDecode>,
    /// Next sample to submit.
    next: usize,
    /// Pictures decoded and not yet due, oldest first.
    ready: VecDeque<vitaslop_platform::video::DecodedPicture>,
    width: u32,
    height: u32,
}

struct AudioPath {
    track: usize,
    decoder: Box<dyn vitaslop_platform::audio_dec::AudioDecode>,
    next: usize,
    channels: u32,
    rate: u32,
    /// Decoded PCM not yet cut into an output frame, interleaved.
    pcm: VecDeque<i16>,
    /// PCM frames handed out since the timeline began - the next frame's timestamp.
    frames_out: u64,
}

/// One player, from `sceAvPlayerInit` to `sceAvPlayerClose`.
pub struct Player {
    handle: i32,
    cb: Callbacks,
    video_buffers: u32,
    auto_start: bool,
    state: State,
    source: Option<Source>,
    looping: bool,
    /// Whether READY has been delivered for the current source.
    ready_sent: bool,
    /// Whether STOP has been delivered for this playback.
    stop_sent: bool,
    /// Guest-clock microsecond of the timeline's 0 ms, and when a pause began.
    origin_us: u64,
    paused_at: Option<u64>,
    /// Output picture buffers from `allocateTexture`, and which one is handed out next.
    frame_bufs: Vec<u32>,
    frame_next: usize,
    frame_bytes: u32,
    /// Two PCM buffers from `allocate`, alternated.
    pcm_bufs: Vec<u32>,
    pcm_next: usize,
    /// Pictures and audio frames handed out, for the line a run says about the movie.
    pictures_out: u64,
    dropped: u64,
    /// Host milliseconds spent in `GetVideoData` keeping the decoder fed (`pump`) and writing
    /// the picture out as two-plane 4:2:0 (`deliver`), since the last report - see
    /// [`report_picture_costs`].
    pump_ms: f64,
    deliver_ms: f64,
}

/// Every player alive.
#[derive(Default)]
pub struct Players {
    list: Vec<Player>,
    next_handle: i32,
}

impl Players {
    fn get(&mut self, handle: i32) -> Option<&mut Player> {
        self.list.iter_mut().find(|p| p.handle == handle)
    }
}

impl Player {
    /// Milliseconds on the movie timeline now.
    fn clock_ms(&self, now_us: u64) -> u64 {
        let at = self.paused_at.unwrap_or(now_us);
        at.saturating_sub(self.origin_us) / 1000
    }

    fn duration_ms(&self) -> u64 {
        self.source.as_ref().map_or(0, |s| s.mp4.duration_us() / 1000)
    }

    /// The end of a non-looping movie: every sample submitted and every picture and audio
    /// frame handed out.
    fn finished(&self) -> bool {
        let Some(src) = &self.source else { return false };
        let video_done = src.video.as_ref().is_none_or(|v| {
            v.next >= src.mp4.tracks[v.track].samples.len() && v.ready.is_empty() && !v.decoder.owes_frames()
        });
        let audio_done = src.audio.as_ref().is_none_or(|a| {
            a.next >= src.mp4.tracks[a.track].samples.len() && a.pcm.is_empty()
        });
        video_done && audio_done
    }
}

/// Deliver an event to the title's callback. `None` while the call is still to run (the
/// handler must return at once; see the module docs).
fn deliver(st: &mut VitaState, cb: &Callbacks, id: u32) -> Option<()> {
    if cb.event == 0 {
        return Some(());
    }
    let call = GuestCall { entry: cb.event, args: vec![cb.event_object, id, 0, 0] };
    st.call_guest(call, None).map(|_| ())
}

/// SceAvPlayerHandle sceAvPlayerInit(SceAvPlayerInitData *data)
#[hostcall]
pub(super) fn init(ctx: &mut GuestCtx, st: &mut VitaState, data: Ptr) -> i32 {
    if data.is_null() {
        ERROR_INVALID_PARAM
    } else {
        let a = data.addr();
        let w = |o: u32| ctx.read_u32(a + o);
        let cb = Callbacks {
            mem_object: w(0x00),
            allocate: w(0x04),
            deallocate: w(0x08),
            allocate_texture: w(0x0c),
            deallocate_texture: w(0x10),
            file_open: w(0x18),
            event_object: w(0x28),
            event: w(0x2c),
        };
        let buffers = w(0x38) as i32;
        // `autoStart` is a boolean at 0x3c; its width is not pinned, so its first byte is read
        // - a zero-initialised block reads the same either way.
        let auto_start = ctx.read_bytes(a + 0x3c, 1)[0] != 0;
        st.avplayers.next_handle += 1;
        let handle = st.avplayers.next_handle;
        tracing::info!(
            target: "vitaslop::status",
            handle,
            allocate_texture = format_args!("{:#010x}", cb.allocate_texture),
            file_functions = cb.file_open != 0,
            event_callback = format_args!("{:#010x}", cb.event),
            buffers, auto_start,
            "SceAvPlayer: a player was created"
        );
        st.avplayers.list.push(Player {
            handle,
            cb,
            video_buffers: if buffers > 0 { buffers as u32 } else { DEFAULT_VIDEO_BUFFERS },
            auto_start,
            state: State::Idle,
            source: None,
            looping: false,
            ready_sent: false,
            stop_sent: false,
            origin_us: 0,
            paused_at: None,
            frame_bufs: Vec::new(),
            frame_next: 0,
            frame_bytes: 0,
            pcm_bufs: Vec::new(),
            pcm_next: 0,
            pictures_out: 0,
            dropped: 0,
            pump_ms: 0.0,
            deliver_ms: 0.0,
        });
        handle
    }
}

/// Open and parse a movie on the engine's own file system.
fn open_source(st: &mut VitaState, path: &str) -> Result<Source, String> {
    const SCE_O_RDONLY: u32 = 0x0001;
    let size = st.io_size(path).filter(|&s| s > 0).ok_or_else(|| format!("{path} does not exist or is empty"))?;
    let fd = st.io_open(path, SCE_O_RDONLY);
    if fd < 0 {
        return Err(format!("{path} cannot be opened ({fd:#x})"));
    }
    let read = |st: &mut VitaState, at: u64, len: usize| -> Vec<u8> {
        if st.io_lseek(fd, at as i64, 0) < 0 {
            return Vec::new();
        }
        st.io_read(fd, len).unwrap_or_default()
    };
    let parsed = (|| {
        let (moov_at, moov_len) = crate::mp4::find_moov(size, |at, len| read(st, at, len))
            .map_err(|e| format!("{path}: no moov box ({e:?})"))?;
        if moov_len > 16 * 1024 * 1024 {
            return Err(format!("{path}: its moov box is {moov_len} bytes"));
        }
        let moov = read(st, moov_at, moov_len as usize);
        crate::mp4::Mp4::parse_moov(&moov).map_err(|e| format!("{path}: {e:?}"))
    })();
    let mp4 = match parsed {
        Ok(m) => m,
        Err(e) => {
            st.io_close(fd);
            return Err(e);
        }
    };
    let streams = mp4
        .tracks
        .iter()
        .enumerate()
        .filter_map(|(i, t)| match t.kind {
            crate::mp4::TrackKind::Video => Some(Stream { track: i, kind: stream_type::VIDEO, enabled: false }),
            crate::mp4::TrackKind::Audio => Some(Stream { track: i, kind: stream_type::AUDIO, enabled: false }),
            _ => None,
        })
        .collect();
    Ok(Source { path: path.to_string(), fd, mp4, streams, video: None, audio: None })
}

/// int sceAvPlayerAddSource(SceAvPlayerHandle handle, const char *filename)
#[hostcall]
pub(super) fn add_source(ctx: &mut GuestCtx, st: &mut VitaState, handle: i32, filename: Ptr) -> i32 {
    if filename.is_null() {
        ERROR_INVALID_PARAM
    } else {
        let path = ctx.read_cstr(filename.addr(), 256);
        do_add_source(st, handle, &path)
    }
}

fn do_add_source(st: &mut VitaState, handle: i32, path: &str) -> i32 {
    let Some(p) = st.avplayers.get(handle) else { return ERROR_INVALID_PARAM };
    // Idempotent across the replays the READY delivery causes: the source is opened once.
    if p.source.is_none() {
        if p.cb.file_open != 0 {
            // A plain file is read as one whatever functions the title also handed over, as the
            // demuxer does (see `vita::video`); a path only the title's functions understand is
            // not supported yet and says so.
            if st.io_size(path).is_none_or(|s| s == 0) {
                tracing::warn!(
                    target: "vitaslop::movie",
                    %path,
                    "SceAvPlayer: the movie is not a file of ours and would have to be read through \
                     the title's own file functions, which this player does not do yet - it will NOT play"
                );
                return ERROR_NOT_SUPPORTED;
            }
        }
        let source = match open_source(st, path) {
            Ok(s) => s,
            Err(reason) => {
                tracing::warn!(target: "vitaslop::movie", %path, %reason, "SceAvPlayer: this movie will NOT play");
                return ERROR_NOT_SUPPORTED;
            }
        };
        let tracks: Vec<String> = source
            .mp4
            .tracks
            .iter()
            .map(|t| format!("{} {}x{} {}ms", String::from_utf8_lossy(&t.codec), t.width, t.height, t.duration_us() / 1000))
            .collect();
        tracing::info!(target: "vitaslop::status", handle, %path, ?tracks, "SceAvPlayer: source added");
        let Some(p) = st.avplayers.get(handle) else { return ERROR_INVALID_PARAM };
        p.source = Some(source);
        p.state = State::Ready;
        p.ready_sent = false;
        p.stop_sent = false;
    } else if p.source.as_ref().is_some_and(|s| s.path != path) {
        // One source per player.
        return ERROR_OPERATION_FAILED;
    }
    let p = st.avplayers.get(handle).expect("checked above");
    if p.ready_sent {
        return 0;
    }
    let (cb, auto_start) = (p.cb, p.auto_start);
    if auto_start || cb.event == 0 {
        // The library picks the first video and audio streams and starts itself.
        if let Some(src) = &mut p.source {
            for kind in [stream_type::VIDEO, stream_type::AUDIO] {
                if let Some(s) = src.streams.iter_mut().find(|s| s.kind == kind) {
                    s.enabled = true;
                }
            }
        }
    }
    if deliver(st, &cb, event::STATE_READY).is_none() {
        return 0;
    }
    let p = st.avplayers.get(handle).expect("checked above");
    p.ready_sent = true;
    if auto_start && p.state == State::Ready {
        return do_start(st, handle);
    }
    0
}

/// The natural end, seen from whichever call notices it first: STOP is delivered and the
/// player stops. On the device the library's own thread does this the moment the last frame
/// is consumed, and a title may wait for that event rather than poll `IsActive`. Returns
/// whether the player has ended (and so the call should answer "no data").
fn end_if_finished(st: &mut VitaState, handle: i32) -> bool {
    let Some(p) = st.avplayers.get(handle) else { return true };
    if p.state != State::Playing || p.looping {
        return false;
    }
    if !p.finished() {
        if let Some(src) = &p.source {
            tracing::debug!(
                target: "vitaslop::movie",
                handle,
                video = ?src.video.as_ref().map(|v| (v.next, src.mp4.tracks[v.track].samples.len(), v.ready.len(), v.decoder.owes_frames())),
                audio = ?src.audio.as_ref().map(|a| (a.next, src.mp4.tracks[a.track].samples.len(), a.pcm.len())),
                "SceAvPlayer: not finished (next, samples, queued[, owed])"
            );
        }
        return false;
    }
    if deliver_stop(st, handle)
        && let Some(p) = st.avplayers.get(handle)
    {
        p.state = State::Stopped;
    }
    true
}

/// int sceAvPlayerStreamCount(SceAvPlayerHandle handle)
#[hostcall]
pub(super) fn stream_count(st: &mut VitaState, handle: i32) -> i32 {
    match st.avplayers.get(handle).and_then(|p| p.source.as_ref()) {
        Some(s) => s.streams.len() as i32,
        None => ERROR_OPERATION_FAILED,
    }
}

/// int sceAvPlayerGetStreamInfo(SceAvPlayerHandle handle, uint32_t id, SceAvPlayerStreamInfo *info)
#[hostcall]
pub(super) fn get_stream_info(ctx: &mut GuestCtx, st: &mut VitaState, handle: i32, id: u32, info: Ptr) -> i32 {
    let found = st
        .avplayers
        .get(handle)
        .and_then(|p| p.source.as_ref())
        .and_then(|s| s.streams.get(id as usize).map(|stream| (stream.kind, &s.mp4.tracks[stream.track])));
    match found {
        Some((kind, track)) if !info.is_null() => {
            let mut b = [0u8; 0x28];
            b[0..4].copy_from_slice(&kind.to_le_bytes());
            if kind == stream_type::VIDEO {
                b[0x08..0x0c].copy_from_slice(&track.width.to_le_bytes());
                b[0x0c..0x10].copy_from_slice(&track.height.to_le_bytes());
                let aspect = if track.height == 0 { 0.0 } else { track.width as f32 / track.height as f32 };
                b[0x10..0x14].copy_from_slice(&aspect.to_bits().to_le_bytes());
            } else {
                let (channels, rate) = audio_shape(track);
                b[0x08..0x0a].copy_from_slice(&(channels as u16).to_le_bytes());
                b[0x0c..0x10].copy_from_slice(&rate.to_le_bytes());
            }
            b[0x18..0x20].copy_from_slice(&(track.duration_us() / 1000).to_le_bytes());
            ctx.write_bytes(info.addr(), &b);
            0
        }
        _ => ERROR_INVALID_PARAM,
    }
}

/// The channel count and sample rate an AAC track declares: the AudioSpecificConfig's own
/// fields, the track timescale as the fallback for the rate.
fn audio_shape(track: &crate::mp4::Track) -> (u32, u32) {
    let asc = track.audio_specific_config();
    const RATES: [u32; 13] = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];
    if asc.len() >= 2 {
        let freq_index = (((asc[0] & 0x07) << 1) | (asc[1] >> 7)) as usize;
        let channels = ((asc[1] >> 3) & 0x0f) as u32;
        let rate = RATES.get(freq_index).copied().unwrap_or(track.timescale);
        return (channels.max(1), rate);
    }
    (2, track.timescale)
}

/// int sceAvPlayerEnableStream(SceAvPlayerHandle handle, uint32_t id)
#[hostcall]
pub(super) fn enable_stream(st: &mut VitaState, handle: i32, id: u32) -> i32 {
    set_stream(st, handle, id, true)
}

/// int sceAvPlayerDisableStream(SceAvPlayerHandle handle, uint32_t id)
#[hostcall]
pub(super) fn disable_stream(st: &mut VitaState, handle: i32, id: u32) -> i32 {
    set_stream(st, handle, id, false)
}

fn set_stream(st: &mut VitaState, handle: i32, id: u32, on: bool) -> i32 {
    match st.avplayers.get(handle).and_then(|p| p.source.as_mut()).and_then(|s| s.streams.get_mut(id as usize)) {
        Some(s) => {
            s.enabled = on;
            0
        }
        None => ERROR_INVALID_PARAM,
    }
}

/// int sceAvPlayerStart(SceAvPlayerHandle handle)
#[hostcall]
pub(super) fn start(st: &mut VitaState, handle: i32) -> i32 {
    do_start(st, handle)
}

fn do_start(st: &mut VitaState, handle: i32) -> i32 {
    let Some(p) = st.avplayers.get(handle) else { return ERROR_INVALID_PARAM };
    if p.source.is_none() {
        return ERROR_OPERATION_FAILED;
    }
    let cb = p.cb;
    let src = p.source.as_ref().expect("checked");
    let video_track = src.streams.iter().find(|s| s.enabled && s.kind == stream_type::VIDEO).map(|s| s.track);
    let audio_track = src.streams.iter().find(|s| s.enabled && s.kind == stream_type::AUDIO).map(|s| s.track);
    let (width, height) = video_track.map_or((0, 0), |t| (src.mp4.tracks[t].width, src.mp4.tracks[t].height));
    let frame_bytes = width * height * 3 / 2;
    let audio_channels = audio_track.map_or(0, |t| audio_shape(&src.mp4.tracks[t]).0);
    let want_frames = if video_track.is_some() && p.frame_bufs.is_empty() { p.video_buffers } else { 0 };
    let want_pcm = audio_track.is_some() && p.pcm_bufs.is_empty();
    // The title's allocators, all before anything changes (see the module docs).
    let mut frames = Vec::new();
    for _ in 0..want_frames {
        let call = GuestCall { entry: cb.allocate_texture, args: vec![cb.mem_object, 256, frame_bytes] };
        let Some(r) = st.call_guest(call, None) else { return 0 };
        frames.push(r.r0);
    }
    let mut pcm = Vec::new();
    if want_pcm {
        for _ in 0..2 {
            let call = GuestCall {
                entry: cb.allocate,
                args: vec![cb.mem_object, 64, (AUDIO_FRAME as u32) * audio_channels * 2],
            };
            let Some(r) = st.call_guest(call, None) else { return 0 };
            pcm.push(r.r0);
        }
    }
    if frames.contains(&0) || pcm.contains(&0) {
        tracing::warn!(target: "vitaslop::movie", handle, "SceAvPlayer: the title's allocator refused a buffer");
        return ERROR_OUT_OF_MEMORY;
    }
    // Open the decoders once.
    let needs_open = st.avplayers.get(handle).is_some_and(|p| {
        p.source.as_ref().is_some_and(|s| (video_track.is_some() && s.video.is_none()) || (audio_track.is_some() && s.audio.is_none()))
    });
    if needs_open {
        let (video, audio) = {
            let p = st.avplayers.get(handle).expect("checked");
            let src = p.source.as_ref().expect("checked");
            let avcc = video_track.map(|t| src.mp4.tracks[t].codec_config.clone());
            let asc = audio_track.map(|t| (src.mp4.tracks[t].audio_specific_config(), audio_shape(&src.mp4.tracks[t])));
            (avcc, asc)
        };
        let video_dec = match video {
            Some(avcc) => match crate::mp4::AvcC::parse(&avcc) {
                Ok(parsed) => match st.video.open_h264(&avcc, parsed.length_size) {
                    Ok(d) => Some(d),
                    Err(e) => {
                        tracing::warn!(target: "vitaslop::movie", handle, error = ?e, "SceAvPlayer: no video decoder - the movie will show no pictures");
                        None
                    }
                },
                Err(e) => {
                    tracing::warn!(target: "vitaslop::movie", handle, error = ?e, "SceAvPlayer: the avcC record cannot be read");
                    None
                }
            },
            None => None,
        };
        let audio_dec = match audio {
            Some((asc, (channels, rate))) => {
                let stream = vitaslop_platform::audio_dec::AudioStream { asc, channels, sample_rate: rate };
                match st.audio_dec.open_aac(&stream) {
                    Ok(d) => Some((d, channels, rate)),
                    Err(e) => {
                        tracing::warn!(target: "vitaslop::movie", handle, error = ?e, "SceAvPlayer: no audio decoder - the movie will be silent");
                        None
                    }
                }
            }
            None => None,
        };
        let p = st.avplayers.get(handle).expect("checked");
        let src = p.source.as_mut().expect("checked");
        if let (Some(t), Some(decoder)) = (video_track, video_dec) {
            src.video = Some(VideoPath { track: t, decoder, next: 0, ready: VecDeque::new(), width, height });
        }
        if let (Some(t), Some((decoder, channels, rate))) = (audio_track, audio_dec) {
            src.audio = Some(AudioPath { track: t, decoder, next: 0, channels, rate, pcm: VecDeque::new(), frames_out: 0 });
        }
    }
    // PLAY is the last call into the title, so everything below happens exactly once.
    if deliver(st, &cb, event::STATE_PLAY).is_none() {
        return 0;
    }
    let now = st.guest_mono_us();
    let p = st.avplayers.get(handle).expect("checked");
    if !frames.is_empty() {
        p.frame_bufs = frames;
        p.frame_bytes = frame_bytes;
    }
    if !pcm.is_empty() {
        p.pcm_bufs = pcm;
    }
    if matches!(p.state, State::Ready | State::Stopped) {
        p.state = State::Playing;
        p.origin_us = now;
        p.paused_at = None;
        p.stop_sent = false;
        tracing::info!(target: "vitaslop::status", handle, width, height, audio_channels, "SceAvPlayer: playback started");
    }
    0
}

/// `SCE_AVPLAYER_ERROR_OUT_OF_MEMORY`.
const ERROR_OUT_OF_MEMORY: i32 = 0x806A_0003u32 as i32;

/// int sceAvPlayerPause(SceAvPlayerHandle handle)
#[hostcall]
pub(super) fn pause(st: &mut VitaState, handle: i32) -> i32 {
    do_pause(st, handle)
}

fn do_pause(st: &mut VitaState, handle: i32) -> i32 {
    let now = st.guest_mono_us();
    let Some(p) = st.avplayers.get(handle) else { return ERROR_INVALID_PARAM };
    let cb = p.cb;
    if p.state != State::Playing {
        return ERROR_OPERATION_FAILED;
    }
    if deliver(st, &cb, event::STATE_PAUSE).is_none() {
        return 0;
    }
    let p = st.avplayers.get(handle).expect("checked");
    p.state = State::Paused;
    p.paused_at = Some(now);
    0
}

/// int sceAvPlayerResume(SceAvPlayerHandle handle)
#[hostcall]
pub(super) fn resume(st: &mut VitaState, handle: i32) -> i32 {
    do_resume(st, handle)
}

fn do_resume(st: &mut VitaState, handle: i32) -> i32 {
    let now = st.guest_mono_us();
    let Some(p) = st.avplayers.get(handle) else { return ERROR_INVALID_PARAM };
    let cb = p.cb;
    if p.state != State::Paused {
        return ERROR_OPERATION_FAILED;
    }
    if deliver(st, &cb, event::STATE_PLAY).is_none() {
        return 0;
    }
    let p = st.avplayers.get(handle).expect("checked");
    if let Some(at) = p.paused_at.take() {
        p.origin_us += now.saturating_sub(at);
    }
    p.state = State::Playing;
    0
}

/// int sceAvPlayerStop(SceAvPlayerHandle handle)
#[hostcall]
pub(super) fn stop(st: &mut VitaState, handle: i32) -> i32 {
    do_stop(st, handle)
}

fn do_stop(st: &mut VitaState, handle: i32) -> i32 {
    let Some(p) = st.avplayers.get(handle) else { return ERROR_INVALID_PARAM };
    if p.source.is_none() {
        return ERROR_OPERATION_FAILED;
    }
    if deliver_stop(st, handle)
        && let Some(p) = st.avplayers.get(handle)
    {
        p.state = State::Stopped;
    }
    0
}

/// Deliver STOP once per playback. Returns whether it has been delivered.
fn deliver_stop(st: &mut VitaState, handle: i32) -> bool {
    let Some(p) = st.avplayers.get(handle) else { return true };
    if p.stop_sent {
        return true;
    }
    let cb = p.cb;
    if deliver(st, &cb, event::STATE_STOP).is_none() {
        return false;
    }
    if let Some(p) = st.avplayers.get(handle) {
        p.stop_sent = true;
        tracing::info!(target: "vitaslop::status", handle, pictures = p.pictures_out, dropped = p.dropped, "SceAvPlayer: playback stopped");
    }
    true
}

/// int sceAvPlayerClose(SceAvPlayerHandle handle)
#[hostcall]
pub(super) fn close(st: &mut VitaState, handle: i32) -> i32 {
    do_close(st, handle)
}

fn do_close(st: &mut VitaState, handle: i32) -> i32 {
    let Some(p) = st.avplayers.get(handle) else { return ERROR_INVALID_PARAM };
    let cb = p.cb;
    let frames = p.frame_bufs.clone();
    let pcm = p.pcm_bufs.clone();
    for f in frames {
        let call = GuestCall { entry: cb.deallocate_texture, args: vec![cb.mem_object, f] };
        if st.call_guest(call, None).is_none() {
            return 0;
        }
    }
    for b in pcm {
        let call = GuestCall { entry: cb.deallocate, args: vec![cb.mem_object, b] };
        if st.call_guest(call, None).is_none() {
            return 0;
        }
    }
    if let Some(i) = st.avplayers.list.iter().position(|p| p.handle == handle) {
        let p = st.avplayers.list.remove(i);
        if let Some(src) = p.source {
            st.io_close(src.fd);
        }
    }
    0
}

/// SceBool sceAvPlayerIsActive(SceAvPlayerHandle handle)
#[hostcall]
pub(super) fn is_active(st: &mut VitaState, handle: i32) -> i32 {
    do_is_active(st, handle)
}

fn do_is_active(st: &mut VitaState, handle: i32) -> i32 {
    let Some(p) = st.avplayers.get(handle) else { return 0 };
    match p.state {
        State::Idle | State::Stopped => 0,
        State::Ready | State::Paused => 1,
        State::Playing => i32::from(!end_if_finished(st, handle)),
    }
}

/// uint64_t sceAvPlayerCurrentTime(SceAvPlayerHandle handle) - milliseconds, in r0:r1.
pub(super) fn current_time(ctx: &mut GuestCtx, st: &mut VitaState) {
    let handle = ctx.arg(0) as i32;
    let now = st.guest_mono_us();
    let ms = st.avplayers.get(handle).map_or(0, |p| match p.state {
        State::Idle | State::Ready => 0,
        _ => p.clock_ms(now).min(p.duration_ms().max(1)),
    });
    ctx.regs[0] = ms as u32;
    ctx.regs[1] = (ms >> 32) as u32;
}

/// int sceAvPlayerJumpToTime(SceAvPlayerHandle handle, uint64_t offset) - offset in ms, r2:r3.
pub(super) fn jump_to_time(ctx: &mut GuestCtx, st: &mut VitaState) {
    let handle = ctx.arg(0) as i32;
    let target_ms = u64::from(ctx.arg(2)) | (u64::from(ctx.arg(3)) << 32);
    let now = st.guest_mono_us();
    let r = match st.avplayers.get(handle) {
        Some(p) if p.source.is_some() => {
            let target = target_ms.min(p.duration_ms());
            seek(p, target);
            p.origin_us = now.saturating_sub(target * 1000);
            if p.paused_at.is_some() {
                p.paused_at = Some(now);
            }
            0
        }
        _ => ERROR_INVALID_PARAM,
    };
    ctx.ret(r as u32);
}

/// Restart both decoders at `ms` on the timeline: video from the sync sample at or before it,
/// audio from the first sample at or after it.
fn seek(p: &mut Player, ms: u64) {
    let Some(src) = &mut p.source else { return };
    if let Some(v) = &mut src.video {
        let t = &src.mp4.tracks[v.track];
        let pts = ms * t.timescale as u64 / 1000;
        let at = t.sample_at_pts(pts).unwrap_or(t.samples.len());
        v.next = t.sync_at_or_before(at).unwrap_or(0);
        v.ready.clear();
        let _ = v.decoder.reset();
    }
    if let Some(a) = &mut src.audio {
        let t = &src.mp4.tracks[a.track];
        let pts = ms * t.timescale as u64 / 1000;
        a.next = t.sample_at_pts(pts).unwrap_or(t.samples.len());
        a.pcm.clear();
        a.frames_out = ms * a.rate as u64 / 1000;
        let _ = a.decoder.reset();
    }
}

/// int sceAvPlayerSetLooping(SceAvPlayerHandle handle, SceBool loop)
#[hostcall]
pub(super) fn set_looping(st: &mut VitaState, handle: i32, looping: u32) -> i32 {
    match st.avplayers.get(handle) {
        Some(p) => {
            p.looping = looping != 0;
            0
        }
        None => ERROR_INVALID_PARAM,
    }
}

/// Read one sample of `track` from the movie file.
fn read_sample(st: &mut VitaState, fd: i32, sample: &crate::mp4::Sample) -> Vec<u8> {
    if st.io_lseek(fd, sample.offset as i64, 0) < 0 {
        return Vec::new();
    }
    st.io_read(fd, sample.size as usize).unwrap_or_default()
}

/// Keep the video decoder fed [`VIDEO_AHEAD`] pictures ahead and collect what it produced.
fn pump_video(st: &mut VitaState, handle: i32) {
    loop {
        let Some(p) = st.avplayers.get(handle) else { return };
        let looping = p.looping;
        let Some(src) = &mut p.source else { return };
        let fd = src.fd;
        let Some(v) = &mut src.video else { return };
        while let Ok(Some(pic)) = v.decoder.poll() {
            v.ready.push_back(pic);
        }
        let samples = &src.mp4.tracks[v.track].samples;
        if v.ready.len() >= VIDEO_AHEAD {
            return;
        }
        // >>> AND NOT MORE IN FLIGHT THAN THAT. On an asynchronous decoder (the browser's) a
        // picture is never ready inside the call that fed it, so the bound above never held and
        // every call fed the rest of the movie: MEASURED (job 009) 1430 units queued in the
        // decoder, 600 pictures decoded for 64 shown and 333 dropped late - each decoded one
        // copied out first - and a 400 MB rise in the heap. `VIDEO_IN_FLIGHT` is above any
        // decoder's reorder pipeline, so a decoder holding pictures back still gets the units
        // it needs to release them.
        if v.decoder.in_flight() >= VIDEO_IN_FLIGHT {
            return;
        }
        if v.next >= samples.len() {
            if looping && v.ready.is_empty() && !v.decoder.owes_frames() {
                // The wrap: both streams restart at 0 ms and so does the clock.
                seek(p, 0);
                let now = st.guest_mono_us();
                st.avplayers.get(handle).expect("present").origin_us = now;
                continue;
            }
            let _ = v.decoder.finish();
            v.decoder.drain_owed();
            while let Ok(Some(pic)) = v.decoder.poll() {
                v.ready.push_back(pic);
            }
            return;
        }
        let sample = samples[v.next];
        let ts = src.mp4.tracks[v.track].timescale.max(1) as u64;
        v.next += 1;
        let bytes = read_sample(st, fd, &sample);
        let p = st.avplayers.get(handle).expect("present");
        let v = p.source.as_mut().and_then(|s| s.video.as_mut()).expect("present");
        let pts_us = (sample.pts * 1_000_000 / ts) as i64;
        if let Err(e) = v.decoder.submit(&bytes, pts_us) {
            tracing::warn!(target: "vitaslop::movie", handle, error = ?e, "SceAvPlayer: a video access unit did not decode");
        }
        // Retire the submit before polling, so how many pictures are ready is a function of the
        // input and not of how far the decoder thread got (see `VideoDecode::drain_owed`).
        v.decoder.drain_owed();
    }
}

/// Write a decoded picture as two-plane 4:2:0 at pitch `width`, chroma pairs Cr first (see
/// the module docs).
fn to_two_plane(pic: &vitaslop_platform::video::DecodedPicture, width: u32, height: u32) -> Vec<u8> {
    use vitaslop_platform::video::PictureFormat;
    let (w, h) = (width as usize, height as usize);
    let (pw, ph) = (pic.width as usize, pic.height as usize);
    let mut out = vec![0u8; w * h * 3 / 2];
    // Neutral chroma where the picture does not reach.
    out[w * h..].fill(128);
    let (cw, ch) = (pw.div_ceil(2), ph.div_ceil(2));
    match pic.format {
        PictureFormat::Nv12 => {
            for y in 0..ph.min(h) {
                out[y * w..y * w + pw.min(w)].copy_from_slice(&pic.data[y * pw..y * pw + pw.min(w)]);
            }
            // The decoder's pairs are Cb first; swap each into Cr first.
            let src_c = pw * ph;
            for y in 0..ch.min(h / 2) {
                for x in 0..cw.min(w / 2) {
                    let s = src_c + y * cw * 2 + 2 * x;
                    out[w * h + y * w + 2 * x] = pic.data[s + 1];
                    out[w * h + y * w + 2 * x + 1] = pic.data[s];
                }
            }
        }
        PictureFormat::I420 => {
            for y in 0..ph.min(h) {
                out[y * w..y * w + pw.min(w)].copy_from_slice(&pic.data[y * pw..y * pw + pw.min(w)]);
            }
            let (cb, cr) = (pic.chroma_offset(), pic.cr_offset());
            for y in 0..ch.min(h / 2) {
                for x in 0..cw.min(w / 2) {
                    out[w * h + y * w + 2 * x] = pic.data[cr + y * cw + x];
                    out[w * h + y * w + 2 * x + 1] = pic.data[cb + y * cw + x];
                }
            }
        }
        PictureFormat::Rgba => {
            // BT.601 studio swing: the GXM default YUV profile the frame is sampled back through,
            // so a picture the browser's decoder handed over as RGBA round-trips to itself.
            let px = |x: usize, y: usize| {
                let i = (y * pw + x) * 4;
                (pic.data[i] as f32, pic.data[i + 1] as f32, pic.data[i + 2] as f32)
            };
            for y in 0..ph.min(h) {
                for x in 0..pw.min(w) {
                    let (r, g, b) = px(x, y);
                    out[y * w + x] = (16.0 + 0.2568 * r + 0.5041 * g + 0.0979 * b).round().clamp(0.0, 255.0) as u8;
                }
            }
            for y in 0..ch.min(h / 2) {
                for x in 0..cw.min(w / 2) {
                    let (r, g, b) = px((2 * x).min(pw - 1), (2 * y).min(ph - 1));
                    let u = 128.0 - 0.1482 * r - 0.2910 * g + 0.4392 * b;
                    let v = 128.0 + 0.4392 * r - 0.3678 * g - 0.0714 * b;
                    out[w * h + y * w + 2 * x] = v.round().clamp(0.0, 255.0) as u8;
                    out[w * h + y * w + 2 * x + 1] = u.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    out
}

/// Pictures between two [`report_picture_costs`] lines.
const PICTURE_REPORT_EVERY: u64 = 64;

/// >>> WHAT A MOVIE PICTURE COSTS THE HOST, reported as the movie plays. A movie that plays
/// choppily on a device and smoothly on the desktop has three suspects with three different
/// remedies - the decoder's own delivery (in the browser, `VideoFrame.copyTo`, a GPU readback
/// when the decoder hands over GPU frames), keeping it fed, and writing the picture into guest
/// memory - and only the decoder's `describe` names which kind of decoder this device chose.
fn report_picture_costs(handle: i32, p: &mut Player) {
    let n = PICTURE_REPORT_EVERY as f64;
    let decoder = p.source.as_ref().and_then(|s| s.video.as_ref()).map(|v| v.decoder.describe()).unwrap_or_default();
    tracing::info!(
        target: "vitaslop::status",
        handle,
        pictures = p.pictures_out,
        dropped = p.dropped,
        "SceAvPlayer: per picture over the last {PICTURE_REPORT_EVERY}: feeding the decoder {:.2} ms, writing it to the guest {:.2} ms | decoder {decoder}",
        p.pump_ms / n,
        p.deliver_ms / n,
    );
    p.pump_ms = 0.0;
    p.deliver_ms = 0.0;
}

/// SceBool sceAvPlayerGetVideoData(SceAvPlayerHandle handle, SceAvPlayerFrameInfo *info)
#[hostcall]
pub(super) fn get_video_data(ctx: &mut GuestCtx, st: &mut VitaState, handle: i32, info: Ptr) -> i32 {
    if info.is_null() {
        0
    } else {
        i32::from(do_get_video_data(ctx, st, handle, info.addr()))
    }
}

fn do_get_video_data(ctx: &mut GuestCtx, st: &mut VitaState, handle: i32, info: u32) -> bool {
    match st.avplayers.get(handle) {
        Some(p) if p.state == State::Playing && !p.frame_bufs.is_empty() => {}
        _ => return false,
    }
    let t_pump = crate::perf::now_ms();
    pump_video(st, handle);
    let t_pumped = crate::perf::now_ms();
    if let (Some(a), Some(b), Some(p)) = (t_pump, t_pumped, st.avplayers.get(handle)) {
        p.pump_ms += b - a;
    }
    if end_if_finished(st, handle) {
        return false;
    }
    let now = st.guest_mono_us();
    let p = st.avplayers.get(handle).expect("checked");
    let clock = p.clock_ms(now);
    let Some(v) = p.source.as_mut().and_then(|s| s.video.as_mut()) else { return false };
    // The newest picture already due; older due ones are dropped.
    let mut pick = None;
    while v.ready.front().is_some_and(|pic| (pic.pts.max(0) as u64) / 1000 <= clock) {
        if pick.is_some() {
            p.dropped += 1;
        }
        pick = v.ready.pop_front();
    }
    let Some(pic) = pick else { return false };
    let (width, height) = (v.width, v.height);
    let t_deliver = crate::perf::now_ms();
    let frame = to_two_plane(&pic, width, height);
    let buf = p.frame_bufs[p.frame_next % p.frame_bufs.len()];
    p.frame_next += 1;
    p.pictures_out += 1;
    let first = p.pictures_out == 1;
    ctx.write_bytes(buf, &frame);
    let mut b = [0u8; 0x20];
    b[0..4].copy_from_slice(&buf.to_le_bytes());
    b[0x08..0x10].copy_from_slice(&((pic.pts.max(0) as u64) / 1000).to_le_bytes());
    b[0x10..0x14].copy_from_slice(&width.to_le_bytes());
    b[0x14..0x18].copy_from_slice(&height.to_le_bytes());
    let aspect = if height == 0 { 0.0 } else { width as f32 / height as f32 };
    b[0x18..0x1c].copy_from_slice(&aspect.to_bits().to_le_bytes());
    ctx.write_bytes(info, &b);
    if let (Some(a), Some(b)) = (t_deliver, crate::perf::now_ms()) {
        p.deliver_ms += b - a;
    }
    if p.pictures_out.is_multiple_of(PICTURE_REPORT_EVERY) {
        report_picture_costs(handle, p);
    }
    if first {
        tracing::info!(target: "vitaslop::status", handle, buffer = format_args!("{buf:#010x}"), width, height, "SceAvPlayer: first picture handed out");
    }
    true
}

/// Keep the audio decoder fed and its PCM queued.
fn pump_audio(st: &mut VitaState, handle: i32) {
    loop {
        let Some(p) = st.avplayers.get(handle) else { return };
        let Some(src) = &mut p.source else { return };
        let fd = src.fd;
        let Some(a) = &mut src.audio else { return };
        while let Ok(Some(out)) = a.decoder.poll() {
            a.pcm.extend(out.samples.iter().copied());
        }
        let samples = &src.mp4.tracks[a.track].samples;
        let have = a.pcm.len() / a.channels.max(1) as usize;
        if have >= AUDIO_FRAME * AUDIO_AHEAD || a.next >= samples.len() {
            return;
        }
        let sample = samples[a.next];
        a.next += 1;
        let bytes = read_sample(st, fd, &sample);
        let p = st.avplayers.get(handle).expect("present");
        let a = p.source.as_mut().and_then(|s| s.audio.as_mut()).expect("present");
        if let Err(e) = a.decoder.submit(&bytes, sample.pts as i64) {
            tracing::warn!(target: "vitaslop::movie", handle, error = ?e, "SceAvPlayer: an audio access unit did not decode");
        }
    }
}

/// SceBool sceAvPlayerGetAudioData(SceAvPlayerHandle handle, SceAvPlayerFrameInfo *info)
#[hostcall]
pub(super) fn get_audio_data(ctx: &mut GuestCtx, st: &mut VitaState, handle: i32, info: Ptr) -> i32 {
    if info.is_null() {
        0
    } else {
        i32::from(do_get_audio_data(ctx, st, handle, info.addr()))
    }
}

fn do_get_audio_data(ctx: &mut GuestCtx, st: &mut VitaState, handle: i32, info: u32) -> bool {
    match st.avplayers.get(handle) {
        Some(p) if p.state == State::Playing && !p.pcm_bufs.is_empty() => {}
        _ => return false,
    }
    pump_audio(st, handle);
    if end_if_finished(st, handle) {
        return false;
    }
    let p = st.avplayers.get(handle).expect("checked");
    let buf = p.pcm_bufs[p.pcm_next % p.pcm_bufs.len()];
    let Some(src) = &mut p.source else { return false };
    let Some(a) = &mut src.audio else { return false };
    let ch = a.channels.max(1) as usize;
    let whole = AUDIO_FRAME * ch;
    let ended = a.next >= src.mp4.tracks[a.track].samples.len();
    if a.pcm.len() < whole && !(ended && !a.pcm.is_empty()) {
        return false;
    }
    let mut out = vec![0u8; whole * 2];
    for (i, s) in a.pcm.drain(..whole.min(a.pcm.len())).enumerate() {
        out[2 * i..2 * i + 2].copy_from_slice(&s.to_le_bytes());
    }
    let ts_ms = a.frames_out * 1000 / a.rate.max(1) as u64;
    a.frames_out += AUDIO_FRAME as u64;
    let (channels, rate) = (a.channels, a.rate);
    p.pcm_next += 1;
    ctx.write_bytes(buf, &out);
    let mut b = [0u8; 0x20];
    b[0..4].copy_from_slice(&buf.to_le_bytes());
    b[0x08..0x10].copy_from_slice(&ts_ms.to_le_bytes());
    b[0x10..0x12].copy_from_slice(&(channels as u16).to_le_bytes());
    b[0x14..0x18].copy_from_slice(&rate.to_le_bytes());
    b[0x18..0x1c].copy_from_slice(&(out.len() as u32).to_le_bytes());
    ctx.write_bytes(info, &b);
    true
}
